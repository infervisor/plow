//! `costmodel` — the shared hardware cost model (design §6.3).
//!
//! Wraps a [`hwspec::GpuSpec`] and turns its raw capabilities into the things
//! the rewriter (now) and the scheduler (later) need: the legal MMA tile shapes,
//! a paged SRAM budget, and cycle/DMA cost estimates. Keeping both halves on one
//! cost model is what keeps egg's choices and the scheduler's realizations
//! consistent.

pub mod cost;
pub mod dominance;
pub mod dtype_cost;
pub mod mma;
pub mod optile;
pub mod sram;
pub mod tile;
pub mod unit;

pub use cost::{handoff_costs, xunit_handoff_costs, Cycles, HandoffCosts, XunitHandoffCosts};
pub use dtype_cost::{to_mma_dtype, CostParams};
pub use hwspec::MmaDtype;
pub use mma::MmaShape;
pub use optile::{AttnShape, FlashTile, RowShape, RowTile};
pub use sram::{SramModel, SramPolicy};
pub use tile::{GemmShape, TileShape};
pub use unit::{MemoryModel, Region, Soc, Unit, UnitId, UnitKind};

/// Re-exported so downstream crates can obtain a `GpuSpec` without depending on
/// `hwspec` directly (the cost-model API surfaces `GpuSpec`).
pub use hwspec;

use hwspec::{Arch, GpuSpec};

/// Default page size for SRAM staging: 8 KiB. Configurable (4/8/16 KiB) — this
/// is the recommended default that balances metadata overhead against page-slot
/// utilisation across all supported targets (MI300X LDS 64 KiB → 8 pages,
/// RTX 4090 100 KiB → 12 pages, Hopper/Blackwell 228 KiB → 28 pages).
pub const DEFAULT_PAGE_BYTES: u64 = 8 * 1024;

/// Bytes per TMEM column: each column holds 128 lanes × 4-byte f32 accumulator
/// words (Blackwell `tcgen05`). Used for the TMEM capacity filter in
/// [`CostModel::candidates`].
pub const TMEM_COL_BYTES: u64 = 128 * 4;

/// Conservative kernel reservation: bytes of shared memory reserved by the SM
/// kernel itself (barriers, TMA descriptors, LDS scratch) that are NOT available
/// for operand tile staging. Subtracted from `SmSpec.shared_mem` in the default
/// [`CostModel::new`] path.
pub fn kernel_reservation_bytes(arch: Arch) -> u64 {
    match arch {
        // Hopper: 4 KiB for TMA descriptors + CTA barriers + smem_bar.
        Arch::Hopper => 4 * 1024,
        // Blackwell (consumer + datacenter): same TMA/barrier structure as Hopper.
        Arch::Blackwell => 4 * 1024,
        // Ada Lovelace: smaller reservation (no TMA); barriers + shared constants.
        Arch::AdaLovelace => 2 * 1024,
        // CDNA3 (MI300): LDS barrier slots + workgroup scratch.
        Arch::CdnaV3 => 4 * 1024,
        // CDNA4 (MI350): same LDS barrier/scratch structure as CDNA3. The
        // reservation does not scale with the larger 160 KiB LDS — it is a fixed
        // per-workgroup cost, so it is a much smaller fraction here.
        Arch::CdnaV4 => 4 * 1024,
    }
}

/// A hardware cost model bound to one GPU and SRAM page configuration.
pub struct CostModel<'a> {
    pub spec: &'a GpuSpec,
    pub sram: SramModel,
    /// Element size of the staged operands (e.g. 2 for bf16/f16).
    pub elem_bytes: u64,
    /// SRAM buffering depth (e.g. 2 = double-buffer, 3 = triple).
    pub buffering: u64,
    /// Matrix-engine operand dtype (selects the per-dtype compute rate).
    pub mma_dtype: MmaDtype,
    /// Max split-K factor to consider for small-M (decode) GEMMs. `1` (default)
    /// disables split-K so the candidate set + scheduler path are unchanged; a
    /// caller able to emit the split grid + partial-sum reduction raises it.
    pub split_k_max: u64,
}

impl<'a> CostModel<'a> {
    /// Full-SM SRAM budget minus a conservative kernel reservation (barriers,
    /// TMA descriptors, LDS scratch), double-buffered, bf16 operands.
    pub fn new(spec: &'a GpuSpec, page_bytes: u64) -> CostModel<'a> {
        let reserve = kernel_reservation_bytes(spec.arch);
        let available = spec.sm.shared_mem.0.saturating_sub(reserve);
        CostModel {
            spec,
            sram: SramModel::with_available(available, page_bytes),
            elem_bytes: 2,
            buffering: 2,
            mma_dtype: MmaDtype::Bf16,
            split_k_max: 1,
        }
    }

    /// Kernel-dependent SRAM budget: the SM kernel reserves part of shared
    /// memory, leaving `available_bytes` for tile staging.
    pub fn with_available(
        spec: &'a GpuSpec,
        available_bytes: u64,
        page_bytes: u64,
    ) -> CostModel<'a> {
        CostModel {
            spec,
            sram: SramModel::with_available(available_bytes, page_bytes),
            elem_bytes: 2,
            buffering: 2,
            mma_dtype: MmaDtype::Bf16,
            split_k_max: 1,
        }
    }

    /// Candidate tile shapes for `g` under the given SRAM policy.
    /// On Blackwell datacenter, also filters out tiles whose MMA accumulator
    /// would exceed the per-SM TMEM budget.
    pub fn candidates(&self, g: GemmShape, policy: SramPolicy) -> Vec<TileShape> {
        let mut cands = tile::candidates(
            self.spec.arch,
            g,
            &self.sram,
            policy,
            self.elem_bytes,
            self.buffering,
            self.split_k_max,
        );
        // TMEM joint filter: reject tiles whose accumulator exceeds per-SM TMEM
        // (only relevant on Blackwell datacenter where tmem > 0).
        let tmem_bytes = self.spec.sm.tmem.0;
        if tmem_bytes > 0 {
            let tmem_cols = tmem_bytes / TMEM_COL_BYTES;
            cands.retain(|t| Self::accumulator_cols(t) <= tmem_cols);
            // If the filter emptied the list, keep the smallest candidate from
            // the original set (the fallback from tile::candidates guarantees at
            // least one entry before the TMEM filter).
            if cands.is_empty() {
                // Re-run without TMEM filter and pick the tile with fewest cols.
                let all = tile::candidates(
                    self.spec.arch,
                    g,
                    &self.sram,
                    policy,
                    self.elem_bytes,
                    self.buffering,
                    self.split_k_max,
                );
                if let Some(best) = all.into_iter().min_by_key(|t| Self::accumulator_cols(t)) {
                    cands.push(best);
                }
            }
        }
        cands
    }

    /// TMEM columns a GEMM tile's accumulator would occupy (one column = 128
    /// f32 lanes on Blackwell `tcgen05`).
    fn accumulator_cols(t: &TileShape) -> u64 {
        ((t.bm.max(0) * t.bn.max(0)) as u64).div_ceil(128)
    }

    /// Streaming passes a tile needs at this budget (1 ⇒ fits resident).
    pub fn passes(&self, tile: TileShape) -> u64 {
        let ws = self
            .sram
            .working_set_pages(tile, self.elem_bytes, self.buffering);
        self.sram.loop_passes(ws)
    }

    /// SRAM page footprint of a tile's working set.
    pub fn sram_pages(&self, tile: TileShape) -> u64 {
        self.sram
            .working_set_pages(tile, self.elem_bytes, self.buffering)
    }

    /// Estimated total cycles to compute `g` with `tile`.
    pub fn gemm_cost(&self, g: GemmShape, tile: TileShape) -> Cycles {
        cost::gemm_cycles(
            self.spec,
            g,
            tile,
            self.elem_bytes,
            self.buffering,
            self.passes(tile),
            self.mma_dtype,
        )
    }

    /// The per-op **decode dispatch floor** in this GPU's clock cycles — the
    /// counter-gate rendezvous ("dead-air") every single-token (M=1) op pays,
    /// independent of tile work (~4.6 µs on MI350X). A fusion that merges `k`
    /// decode ops into one dispatch removes `(k−1)` of these; this is the
    /// op-count lever the fusion selector ranks with. See [`cost::decode_dispatch_cycles`].
    pub fn decode_op_floor(&self) -> Cycles {
        cost::decode_dispatch_cycles(self.spec)
    }

    /// Lowest-cost tile for `g` under `policy`, with its cost.
    pub fn best_tile(&self, g: GemmShape, policy: SramPolicy) -> Option<(TileShape, Cycles)> {
        self.candidates(g, policy)
            .into_iter()
            .map(|t| (t, self.gemm_cost(g, t)))
            .min_by_key(|&(_, c)| c)
    }

    // --- non-GEMM ops (see `optile`) ----------------------------------------

    /// Streaming passes a flash-attention tile needs at this budget.
    pub fn flash_passes(&self, a: AttnShape, tile: FlashTile) -> u64 {
        let ws =
            self.sram
                .pages(tile.working_set_bytes(a.head_dim, self.elem_bytes, self.buffering));
        self.sram.loop_passes(ws)
    }

    /// Candidate flash-attention tilings under `policy`.
    pub fn flash_candidates(&self, a: AttnShape, policy: SramPolicy) -> Vec<FlashTile> {
        optile::flash_candidates(
            self.spec.arch,
            a,
            &self.sram,
            policy,
            self.elem_bytes,
            self.buffering,
        )
    }

    /// Estimated cycles for attention `a` with `tile`.
    pub fn flash_cost(&self, a: AttnShape, tile: FlashTile) -> Cycles {
        optile::flash_cycles(
            self.spec,
            a,
            tile,
            self.elem_bytes,
            self.buffering,
            self.flash_passes(a, tile),
            self.mma_dtype,
        )
    }

    /// Lowest-cost flash-attention tile for `a` under `policy`, with its cost.
    pub fn best_flash_tile(&self, a: AttnShape, policy: SramPolicy) -> Option<(FlashTile, Cycles)> {
        self.flash_candidates(a, policy)
            .into_iter()
            .map(|t| (t, self.flash_cost(a, t)))
            .min_by_key(|&(_, c)| c)
    }

    /// Candidate row tilings (norm/reduce/element-wise) under `policy`.
    pub fn row_candidates(&self, r: RowShape, policy: SramPolicy) -> Vec<RowTile> {
        optile::row_candidates(r, &self.sram, policy, self.elem_bytes, self.buffering)
    }

    /// Estimated cycles for a row op `r` (memory-bound; tile-independent).
    pub fn row_cost(&self, r: RowShape) -> Cycles {
        optile::row_cycles(self.spec, r, self.elem_bytes)
    }

    /// Largest row tile that fits under `policy` (memory-bound ⇒ fewer, bigger
    /// blocks win on launch overhead; cost itself is tile-independent).
    pub fn best_row_tile(&self, r: RowShape, policy: SramPolicy) -> Option<(RowTile, Cycles)> {
        self.row_candidates(r, policy)
            .into_iter()
            .max_by_key(|t| t.br)
            .map(|t| (t, self.row_cost(r)))
    }

    /// Cycles for a layout-only op moving `bytes` (free when SRAM-`resident`).
    pub fn layout_cost(&self, bytes: u64, resident: bool) -> Cycles {
        optile::layout_cycles(self.spec, bytes, resident)
    }

    // --- dtype-aware methods (Phase 2B) ----------------------------------------

    /// GEMM cost with asymmetric operand dtypes. Uses the weight dtype for SRAM
    /// staging cost and the compute dtype's MMA rate for throughput.
    pub fn gemm_cost_typed(&self, g: GemmShape, tile: TileShape, params: CostParams) -> Cycles {
        let ws_bytes = params.working_set_bytes(tile, self.buffering);
        let ws_pages = self.sram.pages(ws_bytes);
        let passes = self.sram.loop_passes(ws_pages);
        cost::gemm_cycles(
            self.spec,
            g,
            tile,
            params.activation_elem,
            self.buffering,
            passes,
            params.mma_dtype,
        )
    }

    /// Lowest-cost tile for `g` under `policy` with typed operands.
    pub fn best_tile_typed(
        &self,
        g: GemmShape,
        policy: SramPolicy,
        params: CostParams,
    ) -> Option<(TileShape, Cycles)> {
        self.candidates_typed(g, policy, params)
            .into_iter()
            .map(|t| (t, self.gemm_cost_typed(g, t, params)))
            .min_by_key(|&(_, c)| c)
    }

    /// Candidate tiles filtered with asymmetric SRAM working-set.
    pub fn candidates_typed(
        &self,
        g: GemmShape,
        policy: SramPolicy,
        params: CostParams,
    ) -> Vec<TileShape> {
        // Generate candidates using the activation elem_bytes (the A-operand
        // determines the MMA shape selection), then re-filter by asymmetric SRAM.
        let mut cands = tile::candidates(
            self.spec.arch,
            g,
            &self.sram,
            policy,
            params.activation_elem,
            self.buffering,
            self.split_k_max,
        );
        // Re-filter: use the asymmetric working-set for SRAM fit check.
        if policy == SramPolicy::Filter {
            cands.retain(|t| {
                let ws_pages = self
                    .sram
                    .pages(params.working_set_bytes(*t, self.buffering));
                ws_pages <= self.sram.pages_per_sm.max(1)
            });
        }
        // TMEM filter (Blackwell)
        let tmem_bytes = self.spec.sm.tmem.0;
        if tmem_bytes > 0 {
            let tmem_cols = tmem_bytes / TMEM_COL_BYTES;
            cands.retain(|t| Self::accumulator_cols(t) <= tmem_cols);
        }
        // Guarantee at least one candidate (same fallback as the symmetric path).
        if cands.is_empty() {
            let all = tile::candidates(
                self.spec.arch,
                g,
                &self.sram,
                SramPolicy::Stream,
                params.activation_elem,
                self.buffering,
                self.split_k_max,
            );
            if let Some(best) = all
                .into_iter()
                .min_by_key(|t| params.working_set_bytes(*t, self.buffering))
            {
                cands.push(best);
            }
        }
        cands
    }

    /// Streaming passes for a tile with typed operands.
    pub fn passes_typed(&self, tile: TileShape, params: CostParams) -> u64 {
        let ws_pages = self
            .sram
            .pages(params.working_set_bytes(tile, self.buffering));
        self.sram.loop_passes(ws_pages)
    }

    /// SRAM page footprint of a tile's working set with typed operands.
    pub fn sram_pages_typed(&self, tile: TileShape, params: CostParams) -> u64 {
        self.sram
            .pages(params.working_set_bytes(tile, self.buffering))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> &'static GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn pages_from_sram() {
        // H100: 228 KiB shared_mem - 4 KiB kernel reservation = 224 KiB available.
        // 224 KiB / 8 KiB page = 28 pages.
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        assert_eq!(cm.sram.pages_per_sm, (228 * 1024 - 4 * 1024) / (8 * 1024));
    }

    #[test]
    fn prefill_candidates_are_legal_wgmma_tiles() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let cands = cm.candidates(g, SramPolicy::Stream);
        assert!(!cands.is_empty());
        // BK = 4×16 = 64; BN from the {128,256} half of the wgmma-n family.
        assert!(cands
            .iter()
            .all(|t| t.bk == 64 && (t.bn == 128 || t.bn == 256)));
        assert!(cands.iter().any(|t| t.bm == 128)); // prefill keeps square BM
    }

    #[test]
    fn decode_prefers_skinny_bm() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 1,
            n: 4096,
            k: 4096,
        };
        let cands = cm.candidates(g, SramPolicy::Stream);
        assert!(
            cands.iter().all(|t| t.bm == 64),
            "decode should only use skinny BM"
        );
        // Split-K is off by default ⇒ no split variants in the candidate set.
        assert!(cands.iter().all(|t| t.split_k == 1));
    }

    #[test]
    fn decode_selects_split_k_to_fill_the_chip() {
        let g = GemmShape {
            m: 1,
            n: 4096,
            k: 4096,
        };
        // Default (split-K disabled): the single short row-block stays unsplit.
        let cm0 = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        assert_eq!(cm0.best_tile(g, SramPolicy::Stream).unwrap().0.split_k, 1);

        // With split-K allowed, decode fans the K reduction across SMs.
        let mut cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        cm.split_k_max = 8;
        let (tile, _) = cm.best_tile(g, SramPolicy::Stream).unwrap();
        assert!(
            tile.split_k > 1,
            "small-M GEMM should split K to use idle SMs, got {tile:?}"
        );

        // A prefill (large M) never splits, even with split-K enabled.
        let mut cmp = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        cmp.split_k_max = 8;
        let prefill = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        assert!(cmp
            .candidates(prefill, SramPolicy::Stream)
            .iter()
            .all(|t| t.split_k == 1));
    }

    #[test]
    fn filter_drops_oversized_but_stream_keeps_it() {
        // A tiny per-SM budget so big tiles overflow: 1 page of 4 KiB.
        let cm = CostModel::with_available(h100(), 4 * 1024, 4 * 1024);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };

        let big = TileShape::new(128, 256, 64);
        assert!(!cm.sram.fits(big, cm.elem_bytes, cm.buffering));
        assert!(
            cm.passes(big) > 1,
            "oversized tile should stream over >1 pass"
        );

        let streamed = cm.candidates(g, SramPolicy::Stream);
        let filtered = cm.candidates(g, SramPolicy::Filter);
        assert!(streamed.contains(&big), "Stream keeps oversized tiles");
        assert!(!filtered.contains(&big), "Filter drops oversized tiles");
    }

    #[test]
    fn best_tile_picks_a_candidate() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let (tile, cost) = cm.best_tile(g, SramPolicy::Stream).unwrap();
        assert!(cost > 0);
        assert!(cm.candidates(g, SramPolicy::Stream).contains(&tile));
    }

    #[test]
    fn flash_decode_uses_skinny_query_block() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        // Decode: one query row, long KV cache.
        let a = AttnShape {
            heads: 32,
            kv_heads: 32,
            seq_q: 1,
            seq_kv: 4096,
            head_dim: 128,
            causal: true,
            sliding_window: 0,
        };
        let cands = cm.flash_candidates(a, SramPolicy::Stream);
        assert!(!cands.is_empty());
        assert!(
            cands.iter().all(|t| t.bq == 64),
            "decode pads to one MMA row block"
        );
        let (tile, cost) = cm.best_flash_tile(a, SramPolicy::Stream).unwrap();
        assert!(cost > 0 && cands.contains(&tile));
    }

    #[test]
    fn flash_prefill_offers_square_query_blocks() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let a = AttnShape {
            heads: 32,
            kv_heads: 32,
            seq_q: 4096,
            seq_kv: 4096,
            head_dim: 128,
            causal: true,
            sliding_window: 0,
        };
        let cands = cm.flash_candidates(a, SramPolicy::Stream);
        assert!(cands.iter().any(|t| t.bq == 128), "prefill keeps square BQ");
    }

    #[test]
    fn reduce_costs_more_than_pointwise() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        let rows = 4096;
        let feat = 4096;
        let norm = RowShape {
            rows,
            feat,
            operands: 2,
            reduce: true,
        };
        let ew = RowShape {
            rows,
            feat,
            operands: 2,
            reduce: false,
        };
        // Same data, but a reduction sweeps each row twice.
        assert!(cm.row_cost(norm) > cm.row_cost(ew));
        // Larger row blocks are preferred (fewer launches).
        let (tile, _) = cm.best_row_tile(ew, SramPolicy::Stream).unwrap();
        assert_eq!(tile.br, 256);
    }

    #[test]
    fn layout_is_free_when_resident() {
        let cm = CostModel::new(h100(), DEFAULT_PAGE_BYTES);
        assert_eq!(cm.layout_cost(1 << 20, true), 0);
        assert!(cm.layout_cost(1 << 20, false) > 0);
    }

    // --- F7: parametric multi-GPU × page-size sweep ----------------------------

    /// All hardware specs in the registry.
    fn all_specs() -> Vec<&'static GpuSpec> {
        hwspec::registry::ALL.to_vec()
    }

    /// The page sizes the system supports (4/8/16 KiB).
    const PAGE_SIZES: [u64; 3] = [4 * 1024, 8 * 1024, 16 * 1024];

    /// Representative GEMM shapes (prefill, decode, skinny-M, large square).
    fn gemm_shapes() -> Vec<GemmShape> {
        vec![
            GemmShape {
                m: 4096,
                n: 4096,
                k: 4096,
            }, // large prefill
            GemmShape {
                m: 1,
                n: 4096,
                k: 4096,
            }, // decode (single token)
            GemmShape {
                m: 64,
                n: 4096,
                k: 4096,
            }, // small batch decode
            GemmShape {
                m: 2048,
                n: 8192,
                k: 4096,
            }, // wide projection
        ]
    }

    /// For every (spec, page_size, gemm_shape) combination:
    /// 1. Candidates must be non-empty (F4 guarantee).
    /// 2. Under Filter policy, every candidate must fit SRAM (pages ≤ budget).
    /// 3. The page budget must not exceed the hardware shared_mem (no over-claim).
    /// 4. On Blackwell datacenter (tmem > 0), candidates must not exceed TMEM.
    #[test]
    fn parametric_sram_budget_sweep() {
        for spec in all_specs() {
            for &page_bytes in &PAGE_SIZES {
                let cm = CostModel::new(spec, page_bytes);

                // Verify the page budget is sensible:
                // pages_per_sm * page_bytes ≤ shared_mem (after reservation).
                let reserve = kernel_reservation_bytes(spec.arch);
                let available = spec.sm.shared_mem.0.saturating_sub(reserve);
                let expected_pages = available / page_bytes;
                assert_eq!(
                    cm.sram.pages_per_sm,
                    expected_pages,
                    "pages_per_sm mismatch for {} @ {} KiB pages",
                    spec.name,
                    page_bytes / 1024
                );
                assert!(
                    cm.sram.pages_per_sm * page_bytes <= spec.sm.shared_mem.0,
                    "{}: page slots ({} × {} B = {} B) exceed shared_mem ({} B)",
                    spec.name,
                    cm.sram.pages_per_sm,
                    page_bytes,
                    cm.sram.pages_per_sm * page_bytes,
                    spec.sm.shared_mem.0
                );

                for g in gemm_shapes() {
                    // Filter policy: every candidate must fit resident in SRAM.
                    let filter_cands = cm.candidates(g, SramPolicy::Filter);
                    assert!(
                        !filter_cands.is_empty(),
                        "{} @ {} KiB: Filter candidates empty for {:?}",
                        spec.name,
                        page_bytes / 1024,
                        g,
                    );
                    for t in &filter_cands {
                        let ws_pages = cm.sram_pages(*t);
                        assert!(
                            ws_pages <= cm.sram.pages_per_sm,
                            "{} @ {} KiB: tile {:?} needs {} pages > budget {}",
                            spec.name,
                            page_bytes / 1024,
                            t,
                            ws_pages,
                            cm.sram.pages_per_sm,
                        );
                    }

                    // Stream policy: must also be non-empty (superset of filter).
                    let stream_cands = cm.candidates(g, SramPolicy::Stream);
                    assert!(
                        !stream_cands.is_empty(),
                        "{} @ {} KiB: Stream candidates empty for {:?}",
                        spec.name,
                        page_bytes / 1024,
                        g,
                    );
                    assert!(
                        stream_cands.len() >= filter_cands.len(),
                        "{}: Stream candidates fewer than Filter for {:?}",
                        spec.name,
                        g,
                    );

                    // TMEM check (Blackwell datacenter only).
                    if spec.sm.tmem.0 > 0 {
                        let tmem_cols = spec.sm.tmem.0 / TMEM_COL_BYTES;
                        for t in &filter_cands {
                            let cols = CostModel::accumulator_cols(t);
                            assert!(
                                cols <= tmem_cols,
                                "{}: tile {:?} needs {} TMEM cols > budget {}",
                                spec.name,
                                t,
                                cols,
                                tmem_cols,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Verify that switching from 16 KiB → 8 KiB pages strictly increases slot
    /// count for all targets (better utilisation of small SRAMs like MI300X LDS).
    #[test]
    fn smaller_pages_increase_slot_count() {
        for spec in all_specs() {
            let cm16 = CostModel::new(spec, 16 * 1024);
            let cm8 = CostModel::new(spec, 8 * 1024);
            assert!(
                cm8.sram.pages_per_sm >= cm16.sram.pages_per_sm,
                "{}: 8 KiB pages ({}) should give ≥ slots than 16 KiB ({})",
                spec.name,
                cm8.sram.pages_per_sm,
                cm16.sram.pages_per_sm,
            );
        }
    }

    /// The non-empty guarantee (F4) holds even for adversarial N values that
    /// don't divide any MMA-N option.
    #[test]
    fn candidates_non_empty_for_odd_n() {
        for spec in all_specs() {
            let cm = CostModel::new(spec, DEFAULT_PAGE_BYTES);
            // N=100 is not divisible by any standard MMA-N (8, 16, 32, 64, 128, 256)
            let g = GemmShape {
                m: 64,
                n: 100,
                k: 64,
            };
            let cands = cm.candidates(g, SramPolicy::Filter);
            assert!(
                !cands.is_empty(),
                "{}: candidates must never be empty (F4 guarantee), got empty for N=100",
                spec.name,
            );
        }
    }
}
