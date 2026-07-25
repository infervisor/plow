//! Tile-shape candidates for a GEMM, built from the architecture's MMA shapes.

use crate::mma;
use crate::sram::{SramModel, SramPolicy};
use hwspec::Arch;

/// The concrete GEMM being tiled (dims bound by the shape bucket).
#[derive(Clone, Copy, Debug)]
pub struct GemmShape {
    pub m: i64,
    pub n: i64,
    pub k: i64,
}

/// A tile of the output: `BM×BN` accumulated over `BK`-chunks of K.
///
/// `split_k` (default 1) is the **split-K** factor: the `K` reduction is
/// partitioned across `split_k` SMs, each computing a partial `BM×BN`, which are
/// then summed. It exists to fill the chip when `M` is small (decode), where a
/// single `BM×BN` tile leaves most SMs idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileShape {
    pub bm: i64,
    pub bn: i64,
    pub bk: i64,
    pub split_k: i64,
}

impl TileShape {
    /// A plain (non-split) tile.
    pub fn new(bm: i64, bn: i64, bk: i64) -> TileShape {
        TileShape {
            bm,
            bn,
            bk,
            split_k: 1,
        }
    }
}

/// Candidate tile shapes for `g`, filtered by MMA legality and divisibility.
/// SRAM is applied per `policy`: `Filter` drops tiles whose working set exceeds
/// the budget; `Stream` keeps them (they stream over multiple passes).
///
/// When `split_k_max > 1` and `M` is small (decode — a single `BM` row-block
/// covers it), split-K variants are added: the `K` reduction is fanned across up
/// to `split_k_max` SMs (powers of two), each a partial `BM×BN`, to fill a chip
/// that one tile would leave mostly idle. The cost of the partial-sum reduction
/// is priced in [`crate::cost::gemm_cycles`].
pub fn candidates(
    arch: Arch,
    g: GemmShape,
    sram: &SramModel,
    policy: SramPolicy,
    elem_bytes: u64,
    buffering: u64,
    split_k_max: u64,
) -> Vec<TileShape> {
    let mma_m = mma::mma_m(arch);
    let mma_k = mma::mma_k(arch);
    let bk_preferred = mma_k * 4; // e.g. 64 on Hopper (4×16)
    let max_n = mma::max_n(arch);

    // BK: prefer the larger block (fewer mainloop iterations), fall back to
    // the minimum MMA-K granularity when K is not aligned to the larger one.
    let bk = if g.k % bk_preferred == 0 {
        bk_preferred
    } else if g.k % mma_k == 0 {
        mma_k
    } else {
        // K is not a multiple of even the minimum MMA-K; round up so the tile
        // is legal (the kernel masks unused lanes in the last K-step).
        mma_k
    };

    // BM: skinny for decode (small M), square-ish otherwise.
    // CDNA4 (MI350X) has enough VGPRs (512 KiB/CU) for wider M blocks too.
    let bm_opts: Vec<i64> = if g.m <= mma_m {
        vec![mma_m]
    } else if arch == Arch::CdnaV4 {
        vec![mma_m, mma_m * 2, mma_m * 4] // 16, 32, 64
    } else {
        vec![mma_m, mma_m * 2]
    };
    // BN: the larger half of the MMA-n family.
    //
    // CDNA4 override: MFMA instructions top out at n=32, but a CDNA workgroup
    // reaches wider macro-tiles by *repeating* MFMAs across N. The hand-written
    // gfx950 kernels use BN=64/128/256. With 160 KiB LDS these fit comfortably
    // (e.g. BM=64, BN=256, BK=32, double-buffered bf16 = 40 KiB). Without this
    // override the tile selector is stuck at BN≤32 and leaves 90%+ of LDS idle.
    let bn_opts: Vec<i64> = if arch == Arch::CdnaV4 {
        vec![64, 128, 256]
    } else {
        mma::shapes_for(arch)
            .iter()
            .map(|s| s.n as i64)
            .filter(|&n| n >= max_n / 2)
            .collect()
    };

    // Split-K factors: only when M is small (one BM row-block ⇒ few output
    // tiles, idle SMs) and K has enough BK-chunks to divide. Powers of two up to
    // `split_k_max`, bounded by the number of K-iterations.
    let k_iters = (g.k / bk).max(1);
    let split_opts: Vec<i64> = if g.m <= mma_m && split_k_max > 1 {
        std::iter::successors(Some(1i64), |&s| Some(s * 2))
            .take_while(|&s| s as u64 <= split_k_max && s <= k_iters)
            .collect()
    } else {
        vec![1]
    };

    let mut out = Vec::new();
    for &bm in &bm_opts {
        for &bn in &bn_opts {
            // Divisibility along N (M/K remainder is handled by masking).
            if g.n % bn != 0 {
                continue;
            }
            for &split_k in &split_opts {
                let tile = TileShape {
                    bm,
                    bn,
                    bk,
                    split_k,
                };
                if policy == SramPolicy::Filter && !sram.fits(tile, elem_bytes, buffering) {
                    continue;
                }
                out.push(tile);
            }
        }
    }

    // Guarantee at least one candidate — fall back to the smallest MMA-legal
    // tile when nothing passed the divisibility or SRAM filter. The kernel
    // handles N-remainder via predicated stores (masking), and if the tile
    // exceeds SRAM it simply streams over multiple passes.
    if out.is_empty() {
        let min_bn = mma::shapes_for(arch)
            .iter()
            .map(|s| s.n as i64)
            .min()
            .unwrap_or(8);
        let fallback = TileShape {
            bm: mma_m,
            bn: min_bn,
            bk,
            split_k: 1,
        };
        // Under `Filter` the fit constraint is hard (drop oversized): only inject
        // the fallback when it actually fits, so an exhausted budget still yields
        // an empty list. `Stream` always takes it (it streams over passes).
        if policy != SramPolicy::Filter || sram.fits(fallback, elem_bytes, buffering) {
            out.push(fallback);
        }
    }

    out
}
