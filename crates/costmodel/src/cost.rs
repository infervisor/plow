//! Cycle / DMA cost estimates. Rough absolute numbers; the point is *relative*
//! ordering of tile shapes, which is what extraction needs.
//!
//! The compute side is modeled per output tile, then quantized into **waves** of
//! `sm_count` tiles (so SM under-utilization and the partial last wave — the
//! decode M=1 cliff — are visible), with the operand DMA overlapped behind
//! compute when the tile is double-buffered and resident (otherwise serialized).

use crate::sram::SramModel;
use crate::tile::{GemmShape, TileShape};
use hwspec::{GpuSpec, MmaDtype};

pub type Cycles = u64;

/// Fixed per-op launch / grid-setup + epilogue overhead. Modest vs a real
/// GEMM's millions of cycles, but it makes a many-tiny-tiles shape pay for its
/// extra dispatch — a real inefficiency the wave model alone doesn't capture.
pub(crate) const LAUNCH_CYCLES: Cycles = 500;

/// Per-op **decode dispatch floor** (µs): the counter-gate rendezvous "dead-air"
/// every single-token (M=1) op pays at each op→op hand-off, independent of its
/// tile work. Measured ≈4.6 µs on MI350X gfx950 — the decode autopsy's 164 µs/
/// layer counter-gate dead-air ÷ ~36 ops/MoE-layer (`plans/glm-decode-gap-
/// autopsy.md`), and cross-checked by E2: removing 234 ops cut TPOT ~6 %.
///
/// Unlike [`LAUNCH_CYCLES`] (grid setup, <1 % of a GEMV) this DOMINATES a decode
/// GEMV/GEMV-sized op, so it is what lets the model **value op count**: fusing K
/// decode ops into one removes (K−1) floors — a saving that swamps the tile-
/// divisibility deltas the wave model alone sees. Applied only in the M=1 regime,
/// so prefill (large-M) tile ranking is untouched.
pub(crate) const DECODE_DISPATCH_FLOOR_US: f64 = 4.6;

/// [`DECODE_DISPATCH_FLOOR_US`] converted to this spec's clock cycles (the model's
/// unit), so the floor tracks the target's boost clock. Callers add it once per
/// decode op and gate it to their M=1 regime (`g.m`/`seq_q`/`rows == 1`).
pub(crate) fn decode_dispatch_cycles(spec: &GpuSpec) -> Cycles {
    (DECODE_DISPATCH_FLOOR_US * 1.0e-6 * spec.clock_boost.0 as f64).round() as Cycles
}

/// Cycles to retire `macs` multiply-accumulates on this SM's matrix engines at
/// operand dtype `dtype`. An unsupported dtype (`mma.of` is `None`) yields a
/// deliberately huge cost so extraction never picks an unaccelerated path.
pub fn macs_cycles(spec: &GpuSpec, macs: u64, dtype: MmaDtype) -> Cycles {
    let per_core = spec.sm.mma.of(dtype).unwrap_or(0) as u64;
    let throughput = spec.sm.tensor_cores as u64 * per_core;
    macs.div_ceil(throughput.max(1))
}

/// Compute cycles for one `BM×BN×BK` tile-step on this SM's matrix engines.
pub fn tile_compute_cycles(spec: &GpuSpec, tile: TileShape, dtype: MmaDtype) -> Cycles {
    macs_cycles(spec, (tile.bm * tile.bn * tile.bk) as u64, dtype)
}

/// Resident thread-blocks a tile's working set allows per SM, capped by the
/// block limit. Two-or-more concurrent blocks let the SM hide one block's DMA
/// behind another's compute even without intra-block double-buffering.
fn occupancy(spec: &GpuSpec, working_set_bytes: u64) -> u64 {
    let by_smem = spec.sm.shared_mem.0 / working_set_bytes.max(1);
    by_smem.min(spec.sm.max_blocks as u64)
}

/// HBM bytes one SM can move per cycle (whole-GPU bandwidth shared across SMs).
///
/// DATASHEET peak (`mem.bandwidth`) on purpose, NOT `bandwidth_for_bound()`. This
/// feeds [`dma_cycles`], which `devgen::pick_tile` uses to RANK candidate tiles —
/// a relative comparison in which a constant derate cancels exactly. Switching it
/// to the measured figure would change no ranking and would change emitted bytes
/// only by rounding. Anything that REPORTS an absolute floor must use
/// `bandwidth_for_bound()` instead; see `plowc --lean-oracle`.
fn sm_bytes_per_cycle(spec: &GpuSpec) -> f64 {
    let bw_bytes_per_s = spec.mem.bandwidth.0 * 1.0e9; // GB/s → B/s
    let clock_hz = spec.clock_boost.0 as f64;
    (bw_bytes_per_s / clock_hz) / spec.sm_count as f64
}

/// Cycles to DMA `bytes` into/out of SRAM. `resident` (an SRAM hand-off, no HBM
/// round-trip) is ~free.
pub fn dma_cycles(spec: &GpuSpec, bytes: u64, resident: bool) -> Cycles {
    if resident {
        return 0;
    }
    (bytes as f64 / sm_bytes_per_cycle(spec).max(1.0)).ceil() as u64
}

/// On-die DSM (distributed shared memory) is faster than a single SM's share of
/// HBM bandwidth — a rough GPC-fabric speedup over an HBM round-trip.
const DSM_SPEEDUP: u64 = 4;

/// Cycles to move `bytes` SM→SM over the GPC's distributed-shared-memory fabric.
/// `Cycles::MAX` (unavailable) on architectures without DSM (`spec.dsm == None`).
pub fn dsm_cycles(spec: &GpuSpec, bytes: u64) -> Cycles {
    if spec.dsm.is_none() {
        return Cycles::MAX;
    }
    (dma_cycles(spec, bytes, false) / DSM_SPEEDUP).max(1)
}

/// L2-partition speedup over an HBM round-trip: producer's L2 write is read
/// from the same partition without HBM traffic. Chosen as strictly less than
/// [`DSM_SPEEDUP`] (a GPC-fabric SM-to-SM channel is a bit faster than an
/// L2 partition hit) but faster than HBM per-SM share.
const L2_SPEEDUP: u64 = 3;

/// Cycles to move `bytes` within one L2 partition (per-GPC on H100, per-XCD
/// on MI300). `Cycles::MAX` when the spec has no `l2_partitioning`.
///
/// Modeled analogously to [`dsm_cycles`]: the HBM per-SM share divided by
/// [`L2_SPEEDUP`]. Uses the *per-SM* HBM share as the baseline (like DSM)
/// because a single tile still executes on one SM even when reading from
/// the shared L2 partition.
pub fn l2_local_cycles(spec: &GpuSpec, bytes: u64) -> Cycles {
    if spec.l2_partitioning.is_none() {
        return Cycles::MAX;
    }
    (dma_cycles(spec, bytes, false) / L2_SPEEDUP).max(1)
}

/// The cost of the four physical realizations of a producer→consumer hand-off.
/// `hbm` round-trips through DRAM (parallel-friendly); `sram_same_sm` keeps the
/// value resident but serializes producer+consumer on one SM; `dsm` shares it
/// across SMs in one GPC (parallel, no round-trip); `l2_local` keeps the
/// producer's write in the L2 partition for the consumer to read at L2
/// bandwidth without an HBM round-trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandoffCosts {
    pub hbm: Cycles,
    pub sram_same_sm: Cycles,
    pub dsm: Cycles,
    pub l2_local: Cycles,
}

pub fn handoff_costs(spec: &GpuSpec, producer_cycles: Cycles, bytes: u64) -> HandoffCosts {
    HandoffCosts {
        hbm: 2 * dma_cycles(spec, bytes, false),
        sram_same_sm: producer_cycles,
        dsm: dsm_cycles(spec, bytes),
        l2_local: l2_local_cycles(spec, bytes),
    }
}

/// Cross-unit (cross-node) fence cost under unified memory: no data moves, but
/// the two units still synchronize across the fabric.
const BARRIER_CYCLES: Cycles = 300;
/// Inter-node network bandwidth (GB/s) for the RDMA fallback — far below the
/// on-node fast fabric; a tunable default (~400 Gb/s-class NIC).
const RDMA_GBPS: f64 = 50.0;
/// Fixed RDMA setup / round-trip latency (cycles), on top of the byte transfer.
const RDMA_LATENCY: Cycles = 5_000;

/// The cost of the three cross-unit hand-off realizations the tile-egg rules
/// enumerate (design §2.5 / `egl/tile.egg`): `barrier` (unified memory — a
/// fence, no copy), `p2p` (a direct read over the fast peer fabric, within one
/// node-domain), and `rdma` (DPU-routed over a slow link / between nodes — the
/// always-available fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XunitHandoffCosts {
    pub barrier: Cycles,
    pub p2p: Cycles,
    pub rdma: Cycles,
}

/// Cost of moving `bytes` between two units. `p2p` uses the producer's fast peer
/// fabric (NVLink / Infinity Fabric), or `Cycles::MAX` where the spec has none.
pub fn xunit_handoff_costs(spec: &GpuSpec, bytes: u64) -> XunitHandoffCosts {
    let clock = spec.clock_boost.0 as f64;
    let p2p = match spec.interconnect {
        Some(ic) => {
            let bpc = (ic.per_gpu_bandwidth.0 * 1.0e9 / clock).max(1.0);
            (bytes as f64 / bpc).ceil() as Cycles
        }
        None => Cycles::MAX,
    };
    let net_bpc = (RDMA_GBPS * 1.0e9 / clock).max(1.0);
    let rdma = (bytes as f64 / net_bpc).ceil() as Cycles + RDMA_LATENCY;
    XunitHandoffCosts {
        barrier: BARRIER_CYCLES,
        p2p,
        rdma,
    }
}

/// Total estimated cycles to compute the whole GEMM `g` with tile shape `tile`,
/// `buffering`-deep SRAM pipeline, streamed over `passes` SRAM passes, at operand
/// dtype `dtype`.
///
/// A proven **total-work core** (every `BM×BN×BK` step's compute + its operand
/// DMA, the relative ranking the extractor relies on) with four modeled deltas
/// layered on top — none of which flip the core's ordering:
///  * **dtype** — the per-step matrix rate (bf16 anchors at the historical value).
///  * **overlap** — a double-buffered, resident tile hides DMA behind compute.
///  * **output write-back** — the C store (tiling-invariant; previously uncounted).
///  * **wave-quantization tail** — a *multi-wave* op pays for its partial last
///    wave (the 133-vs-132-tile cliff); a single-wave op is left to the core.
pub fn gemm_cycles(
    spec: &GpuSpec,
    g: GemmShape,
    tile: TileShape,
    elem_bytes: u64,
    buffering: u64,
    passes: u64,
    dtype: MmaDtype,
) -> Cycles {
    let tiles_m = (g.m as u64).div_ceil(tile.bm as u64).max(1);
    let tiles_n = (g.n as u64).div_ceil(tile.bn as u64).max(1);
    let k_iters = (g.k as u64).div_ceil(tile.bk as u64).max(1);
    let split = (tile.split_k.max(1)) as u64;
    let out_tiles = tiles_m * tiles_n;

    // Split-K fans the K reduction across `split` SMs: each partial walks only
    // `k_iters / split` of the K loop, and the partials run concurrently — so the
    // critical-path step count divides by `split`. (`split == 1` ⇒ the plain core.)
    let steps = (out_tiles * k_iters).div_ceil(split);

    // Core total-work model (per-SM bandwidth share). Output write-back is the C
    // store, summed over output tiles (tiling-invariant ⇒ ranking-neutral).
    let compute = steps * tile_compute_cycles(spec, tile, dtype);
    let operand_bytes = ((tile.bm * tile.bk + tile.bk * tile.bn) as u64) * elem_bytes;
    let out_bytes = (tile.bm * tile.bn) as u64 * elem_bytes;
    let dma = steps * dma_cycles(spec, operand_bytes * passes, false)
        + out_tiles * dma_cycles(spec, out_bytes, false);

    // Delta — pipeline overlap (else the core serializes compute + dma).
    let mut base = overlap(
        spec,
        tile,
        elem_bytes,
        buffering,
        passes,
        compute,
        dma,
        operand_bytes,
    );

    // Delta — split-K reduction: sum the `split` partials per output tile (read
    // all partials + write the final). Paid only when actually splitting.
    if split > 1 {
        let reduce_bytes = out_bytes * (split + 1);
        base += out_tiles * dma_cycles(spec, reduce_bytes, false);
    }

    // Delta — wave-quantization tail (multi-wave only). Split-K partials are extra
    // independent tiles, so they fill more SMs before the tail bites.
    let sm_count = spec.sm_count.max(1) as u64;
    let tail = wave_tail_penalty(base, out_tiles * split, sm_count);

    // Delta — decode dispatch floor. A single-token (M=1) GEMV is op-overhead
    // bound: its counter-gate rendezvous dwarfs its tile work, so removing an op
    // (fusion) is the real lever. Gated to M=1 ⇒ prefill ranking is unchanged.
    let decode_floor = if g.m == 1 {
        decode_dispatch_cycles(spec)
    } else {
        0
    };

    base + tail + LAUNCH_CYCLES + decode_floor
}

/// The partial-last-wave penalty. Tiles spread `sm_count` at a time; an op that
/// spills past a wave boundary (e.g. 133 tiles on 132 SMs ⇒ 2 waves) wastes the
/// final wave's idle SM-slots, scaled into the op's own cost. Zero under a single
/// wave — there the total-work core governs (and already favours filling more
/// SMs), so this never re-ranks the under-filling case.
pub(crate) fn wave_tail_penalty(base: Cycles, out_tiles: u64, sm_count: u64) -> Cycles {
    if out_tiles <= sm_count {
        return 0;
    }
    let waves = out_tiles.div_ceil(sm_count);
    let waste = waves * sm_count - out_tiles;
    base.saturating_mul(waste) / out_tiles
}

/// Combine a tile's compute and DMA into its effective per-tile cost. A
/// double-buffered tile that fits (`passes == 1`), or one with ≥2 resident
/// blocks, hides DMA behind compute — only one buffer-fill stays exposed.
/// Otherwise (single-buffered, or streamed over multiple passes) the two
/// serialize. This is what makes double-buffering and occupancy pay off.
#[allow(clippy::too_many_arguments)]
fn overlap(
    spec: &GpuSpec,
    tile: TileShape,
    elem_bytes: u64,
    buffering: u64,
    passes: u64,
    compute: Cycles,
    dma: Cycles,
    fill_bytes: u64,
) -> Cycles {
    let ws = SramModel::working_set_bytes(tile, elem_bytes, buffering);
    let hidden = (buffering >= 2 && passes == 1) || occupancy(spec, ws) >= 2;
    if hidden {
        compute.max(dma) + dma_cycles(spec, fill_bytes, false)
    } else {
        compute + dma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> &'static GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }
    fn mi300() -> &'static GpuSpec {
        hwspec::registry::lookup("MI300X").unwrap()
    }
    fn b200() -> &'static GpuSpec {
        hwspec::registry::lookup("B200").unwrap()
    }

    #[test]
    fn decode_floor_calibrated_and_prefill_neutral() {
        // The decode dispatch floor is ~4.6 µs in the target's clock cycles.
        let spec = mi300(); // MI300X: 2.1 GHz boost.
        let floor = decode_dispatch_cycles(spec);
        let expect = (DECODE_DISPATCH_FLOOR_US * 1.0e-6 * spec.clock_boost.0 as f64).round() as u64;
        assert_eq!(floor, expect);
        // Sanity: within 1 % of 4.6 µs at this clock.
        let us = floor as f64 / spec.clock_boost.0 as f64 * 1.0e6;
        assert!(
            (us - DECODE_DISPATCH_FLOOR_US).abs() < 0.05,
            "floor = {us} µs"
        );

        // Gating: only M==1 pays the floor. M=1 and M=16 both fit one bm=16
        // row-block (div_ceil ⇒ identical tiles/steps/base work), so their cost
        // difference is EXACTLY one floor — proving the floor is added at M==1 and
        // NOT at M>1 (prefill ranking untouched).
        let tile = TileShape::new(16, 256, 128);
        let m1 = GemmShape {
            m: 1,
            n: 4096,
            k: 4096,
        };
        let m16 = GemmShape {
            m: 16,
            n: 4096,
            k: 4096,
        };
        let c1 = gemm_cycles(h100(), m1, tile, 2, 2, 1, MmaDtype::Bf16);
        let c16 = gemm_cycles(h100(), m16, tile, 2, 2, 1, MmaDtype::Bf16);
        assert_eq!(c1 - c16, decode_dispatch_cycles(h100()));
    }

    #[test]
    fn dtype_throughput_ratios() {
        let macs = 1 << 20;
        // Hopper: fp8 is 2× bf16 (half the cycles); no fp4 path.
        assert!(
            macs_cycles(h100(), macs, MmaDtype::Fp8) < macs_cycles(h100(), macs, MmaDtype::Bf16)
        );
        assert!(!h100().supports(MmaDtype::Fp4));
        // Blackwell datacenter: fp4 < fp8 < bf16, and fp4 is supported.
        assert!(b200().supports(MmaDtype::Fp4));
        let (bf16, fp8, fp4) = (
            macs_cycles(b200(), macs, MmaDtype::Bf16),
            macs_cycles(b200(), macs, MmaDtype::Fp8),
            macs_cycles(b200(), macs, MmaDtype::Fp4),
        );
        assert!(fp4 < fp8 && fp8 < bf16);
    }

    #[test]
    fn wave_quantization_cliff() {
        // A shape whose output tiles just exceed one wave costs ~2× one that fills
        // exactly one wave — the partial-second-wave tail.
        let sm = h100().sm_count as i64;
        let tile = TileShape::new(64, 256, 64);
        // One BN column-block, BM rows ⇒ tiles_n = 1, tiles_m = rows/bm.
        let one_wave = GemmShape {
            m: sm * tile.bm,
            n: tile.bn,
            k: tile.bk,
        };
        let over_wave = GemmShape {
            m: (sm + 1) * tile.bm,
            n: tile.bn,
            k: tile.bk,
        };
        let c1 = gemm_cycles(h100(), one_wave, tile, 2, 2, 1, MmaDtype::Bf16);
        let c2 = gemm_cycles(h100(), over_wave, tile, 2, 2, 1, MmaDtype::Bf16);
        assert!(c2 > c1, "one extra tile must spill into a second wave");
        // Near-2× (allow slack for the fixed launch term).
        assert!(c2 as f64 > 1.8 * c1 as f64);
    }

    #[test]
    fn overlap_pipelines_when_resident_else_serializes() {
        let (compute, dma) = (100u64, 1000u64); // DMA-bound
                                                // High-occupancy, double-buffered, resident ⇒ DMA hidden behind compute
                                                // (fill_bytes = 0 ⇒ no exposed fill), so cost = max(compute, dma).
        let small = TileShape::new(64, 256, 64);
        assert_eq!(overlap(h100(), small, 2, 2, 1, compute, dma, 0), dma);
        // A tile that fills SRAM (occupancy 1) and streams (passes > 1) can hide
        // nothing ⇒ compute and DMA serialize.
        let big = TileShape::new(256, 256, 64);
        assert_eq!(
            overlap(h100(), big, 2, 2, 2, compute, dma, 0),
            compute + dma
        );
    }

    #[test]
    fn handoff_cost_tradeoffs() {
        let bytes = 256 * 256 * 2;
        // Hopper has DSM: a DSM hand-off beats an HBM round-trip but isn't free.
        let h = handoff_costs(h100(), 1_000, bytes);
        assert!(h.dsm < h.hbm && h.dsm > 0);
        assert_eq!(h.sram_same_sm, 1_000); // serialization = producer cycles
                                           // A cheap producer ⇒ same-SM handoff wins; an expensive one ⇒ it loses.
        assert!(handoff_costs(h100(), 1, bytes).sram_same_sm < h.hbm);
        assert!(handoff_costs(h100(), 10_000_000, bytes).sram_same_sm > h.hbm);
        // CDNA has no DSM ⇒ that option is unavailable.
        assert_eq!(handoff_costs(mi300(), 1_000, bytes).dsm, Cycles::MAX);
    }
}
