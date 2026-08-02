//! Tiling for the non-GEMM operators.
//!
//! GEMM/Linear tiling lives in [`crate::tile`]; everything else in a transformer
//! layer collapses into three more patterns, each modeled here the same way a
//! GEMM tile is — candidate enumeration, a paged-SRAM working set, and a cycle
//! cost — so they all funnel through one cost-driven extractor:
//!
//! * **FlashAttention** ([`FlashTile`]) — query-block tiling (`BQ` rows) with the
//!   K/V blocks streamed (`BKV`) and an online softmax. Working set is the
//!   Q/K/V tiles; the running max/sum live in registers.
//! * **Row** ([`RowTile`]) — RmsNorm/LayerNorm/GroupNorm/Reduce/Softmax and
//!   element-wise/activation/scale/RoPE. A block of `BR` rows is tiled; the
//!   feature dim is reduced (norm/reduce ⇒ two passes over the row) or mapped
//!   point-wise (one pass). The distinction is just the pass count and operand
//!   fan-in.
//! * **Layout-only** ([`layout_cycles`]) — reshape/transpose/slice/concat/
//!   broadcast/embedding. No compute; a pure DMA (gather/scatter), and free when
//!   the value is already SRAM-resident from a colocated producer (§2.5).

use crate::cost::{self, Cycles};
use crate::mma;
use crate::sram::{SramModel, SramPolicy};
use hwspec::{Arch, GpuSpec, MmaDtype};

// --- attention --------------------------------------------------------------

/// Attention problem with dims bound by the shape bucket.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    /// Parallel dim: `batch × num_heads` (each tile-stream is independent).
    pub heads: i64,
    /// Query rows.
    pub seq_q: i64,
    /// Key/value rows attended over.
    pub seq_kv: i64,
    /// Per-head feature width.
    pub head_dim: i64,
}

/// FlashAttention tiling: `BQ` query rows resident, `BKV` key/value rows streamed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashTile {
    pub bq: i64,
    pub bkv: i64,
}

impl FlashTile {
    /// Q-tile + K-tile + V-tile staged in SRAM (each `head_dim` wide), buffered.
    pub fn working_set_bytes(&self, head_dim: i64, elem_bytes: u64, buffering: u64) -> u64 {
        let q = (self.bq * head_dim) as u64;
        let kv = (self.bkv * head_dim) as u64 * 2; // K and V
        (q + kv) * elem_bytes * buffering
    }
}

/// Candidate FlashAttention tilings, filtered by MMA-row legality and `policy`.
pub fn flash_candidates(
    arch: Arch,
    a: AttnShape,
    sram: &SramModel,
    policy: SramPolicy,
    elem_bytes: u64,
    buffering: u64,
) -> Vec<FlashTile> {
    let mma_m = mma::mma_m(arch);
    // BQ: skinny for decode (seq_q == 1), square-ish for prefill — mirrors GEMM.
    let bq_opts: Vec<i64> = if a.seq_q <= mma_m {
        vec![mma_m]
    } else {
        vec![mma_m, mma_m * 2]
    };
    // BKV: the larger half of the MMA-n family, clamped to seq_kv.
    let max_n = mma::max_n(arch);
    let mut bkv_opts: Vec<i64> = mma::shapes_for(arch)
        .iter()
        .map(|s| s.n as i64)
        .filter(|&n| n >= max_n / 2 && n <= a.seq_kv)
        .collect();
    if bkv_opts.is_empty() {
        bkv_opts.push(a.seq_kv.max(1));
    }

    let mut out = Vec::new();
    for &bq in &bq_opts {
        for &bkv in &bkv_opts {
            let tile = FlashTile { bq, bkv };
            let ws = sram.pages(tile.working_set_bytes(a.head_dim, elem_bytes, buffering));
            if policy == SramPolicy::Filter && ws > sram.pages_per_sm.max(1) {
                continue;
            }
            out.push(tile);
        }
    }
    out
}

/// Estimated cycles for the whole attention with `tile`, `buffering`-deep
/// pipeline, streamed over `passes`, at operand dtype `dtype`.
///
/// Same shape as [`crate::cost::gemm_cycles`]: a total-work core (QKᵀ/PV compute
/// plus streamed KV DMA over every `(head, q_block, kv_block)` step) plus the
/// modeled deltas — a per-stream Q load + output write-back, KV-DMA overlap when
/// double-buffered and resident, and the wave-quantization tail when the
/// `heads · q_blocks` streams spill past a wave (decode — `seq_q = 1` — fills
/// only `heads` SMs, so the tail flags its under-utilization).
pub fn flash_cycles(
    spec: &GpuSpec,
    a: AttnShape,
    tile: FlashTile,
    elem_bytes: u64,
    buffering: u64,
    passes: u64,
    dtype: MmaDtype,
) -> Cycles {
    let q_blocks = (a.seq_q as u64).div_ceil(tile.bq as u64).max(1);
    let kv_blocks = (a.seq_kv as u64).div_ceil(tile.bkv as u64).max(1);
    let streams = a.heads as u64 * q_blocks;
    let steps = streams * kv_blocks;

    // Core: QKᵀ (BQ·BKV·head_dim) + PV MACs and the streamed KV DMA per step.
    let macs_per_kv = 2 * (tile.bq * tile.bkv * a.head_dim) as u64;
    let compute = steps * cost::macs_cycles(spec, macs_per_kv, dtype);
    let q_bytes = (tile.bq * a.head_dim) as u64 * elem_bytes;
    let kv_bytes = (tile.bkv * a.head_dim * 2) as u64 * elem_bytes;
    let dma = steps * cost::dma_cycles(spec, kv_bytes * passes, false)
        + streams * cost::dma_cycles(spec, q_bytes * 2, false); // Q load + output write

    // Delta — overlap the streamed KV behind compute when the pipeline can hide it.
    let hidden = buffering >= 2 && passes == 1;
    let base = if hidden {
        compute.max(dma) + cost::dma_cycles(spec, kv_bytes, false)
    } else {
        compute + dma
    };

    // Delta — wave-quantization tail over the independent (head, q_block) streams.
    let sm_count = spec.sm_count.max(1) as u64;
    let tail = cost::wave_tail_penalty(base, streams, sm_count);

    // Delta — decode dispatch floor (single-query attention pays the same per-op
    // rendezvous as a decode GEMV). Gated to seq_q == 1 ⇒ prefill unaffected.
    let decode_floor = if a.seq_q == 1 {
        cost::decode_dispatch_cycles(spec)
    } else {
        0
    };

    base + tail + cost::LAUNCH_CYCLES + decode_floor
}

// --- row ops (norm / reduce / element-wise / activation) --------------------

/// A row-tiled op: `rows` independent rows of width `feat`, with `operands`
/// tensors staged. `reduce` ⇒ a reduction over `feat` (two passes over the row:
/// compute the statistic, then apply); otherwise point-wise (one pass).
#[derive(Clone, Copy, Debug)]
pub struct RowShape {
    pub rows: i64,
    pub feat: i64,
    pub operands: i64,
    pub reduce: bool,
}

/// Row tiling: a block of `BR` rows handled together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowTile {
    pub br: i64,
}

impl RowTile {
    /// `BR` rows × `feat` × operand fan-in, buffered.
    pub fn working_set_bytes(&self, r: RowShape, elem_bytes: u64, buffering: u64) -> u64 {
        (self.br * r.feat * r.operands) as u64 * elem_bytes * buffering
    }
}

/// Candidate row tilings, filtered by `policy`.
pub fn row_candidates(
    r: RowShape,
    sram: &SramModel,
    policy: SramPolicy,
    elem_bytes: u64,
    buffering: u64,
) -> Vec<RowTile> {
    let br_opts = [256, 128, 64];
    let mut out = Vec::new();
    for &br in &br_opts {
        if br > r.rows.max(1) {
            continue;
        }
        let tile = RowTile { br };
        let ws = sram.pages(tile.working_set_bytes(r, elem_bytes, buffering));
        if policy == SramPolicy::Filter && ws > sram.pages_per_sm.max(1) {
            continue;
        }
        out.push(tile);
    }
    if out.is_empty() {
        out.push(RowTile { br: r.rows.max(1) });
    }
    out
}

/// Estimated cycles for a row op: memory-bound, so cost is the bytes traversed.
/// A reduction sweeps each row twice; a point-wise op once.
pub fn row_cycles(spec: &GpuSpec, r: RowShape, elem_bytes: u64) -> Cycles {
    let sweeps = if r.reduce { 2 } else { 1 };
    let bytes = (r.rows * r.feat * r.operands) as u64 * elem_bytes * sweeps;
    // Decode dispatch floor — a single-token (rows == 1) norm/residual/rope pays
    // the per-op rendezvous just like a decode GEMV, so fusing two decode row ops
    // (e.g. B1's Residual+RmsNorm → AddNorm) removes a floor. Gated to rows == 1.
    let decode_floor = if r.rows == 1 {
        cost::decode_dispatch_cycles(spec)
    } else {
        0
    };
    cost::dma_cycles(spec, bytes, false) + decode_floor
}

// --- layout-only ------------------------------------------------------------

/// Cycles for a layout-only op (reshape/transpose/slice/concat/broadcast/
/// embedding): a pure DMA of `bytes`, free when already SRAM-`resident`.
pub fn layout_cycles(spec: &GpuSpec, bytes: u64, resident: bool) -> Cycles {
    cost::dma_cycles(spec, bytes, resident)
}
