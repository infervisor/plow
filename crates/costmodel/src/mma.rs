//! Matrix-engine instruction shapes per architecture.
//!
//! These are the hardware MMA tiles the tensor cores / MFMA units accept; the
//! tile-shape candidates ([`crate::tile`]) are built up from them.

use hwspec::Arch;

/// One matrix-multiply-accumulate instruction shape `C[m,n] += A[m,k]·B[k,n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// Ada Lovelace `mma.sync` family (bf16/f16): m16n8k16 (like Ampere).
/// The hardware issues these in pairs to fill a warp, so tile candidates scale
/// by multiples of 16×8.
const ADA: [MmaShape; 3] = [
    MmaShape { m: 16, n: 8, k: 16 },
    MmaShape {
        m: 16,
        n: 16,
        k: 16,
    },
    MmaShape {
        m: 16,
        n: 32,
        k: 16,
    },
];

/// Hopper `wgmma.m64.nN.k16` family (bf16/f16 operands).
const HOPPER: [MmaShape; 5] = [
    MmaShape {
        m: 64,
        n: 16,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 32,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 64,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 128,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 256,
        k: 16,
    },
];

/// Blackwell `wgmma` family (same shapes as Hopper; 5th-gen tensor cores with
/// higher throughput but identical instruction interface for bf16/f16).
const BLACKWELL: [MmaShape; 5] = [
    MmaShape {
        m: 64,
        n: 16,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 32,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 64,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 128,
        k: 16,
    },
    MmaShape {
        m: 64,
        n: 256,
        k: 16,
    },
];

/// CDNA3 (MI300) MFMA shapes (bf16).
const CDNA3: [MmaShape; 2] = [
    MmaShape {
        m: 16,
        n: 16,
        k: 16,
    },
    MmaShape { m: 32, n: 32, k: 8 },
];

/// CDNA4 (MI350, gfx950) MFMA shapes (bf16): `v_mfma_f32_16x16x32_bf16` and
/// `v_mfma_f32_32x32x16_bf16`. Both double the contraction depth of their CDNA3
/// counterparts, which is where the generation's bf16 rate increase comes from.
///
/// NOTE (known modelling gap, see `plans/gemma4-mi350x-sprint.md`): unlike
/// Hopper/Blackwell `wgmma` — where a *single instruction* spans n up to 256 —
/// an MFMA is wave-scoped and tops out at n=32. A CDNA workgroup reaches a wide
/// macro-tile by *repeating* MFMAs across N, not by issuing a wider one. So
/// `max_n`/`bn_opts` in [`crate::tile`] must not be read as "the widest legal
/// BN" on CDNA — doing so pins BN to {16, 32} and yields tiles ~8× too skinny.
/// The hand-written gfx950 kernels choose their own macro-tiles (BN=128/256);
/// this list exists for `mma_k` (contraction granularity = 32) and cost lookup.
const CDNA4: [MmaShape; 2] = [
    MmaShape {
        m: 16,
        n: 16,
        k: 32,
    },
    MmaShape {
        m: 32,
        n: 32,
        k: 16,
    },
];

pub fn shapes_for(arch: Arch) -> &'static [MmaShape] {
    match arch {
        Arch::AdaLovelace => &ADA,
        Arch::Hopper => &HOPPER,
        Arch::Blackwell => &BLACKWELL,
        Arch::CdnaV3 => &CDNA3,
        Arch::CdnaV4 => &CDNA4,
    }
}

/// Smallest MMA `m` (the minimum useful tile-row granularity).
pub fn mma_m(arch: Arch) -> i64 {
    shapes_for(arch).iter().map(|s| s.m).min().unwrap_or(16) as i64
}

/// MMA `k` (contraction granularity); the family shares one `k`.
pub fn mma_k(arch: Arch) -> i64 {
    shapes_for(arch).iter().map(|s| s.k).max().unwrap_or(16) as i64
}

/// Largest MMA `n`.
pub fn max_n(arch: Arch) -> i64 {
    shapes_for(arch).iter().map(|s| s.n).max().unwrap_or(64) as i64
}
