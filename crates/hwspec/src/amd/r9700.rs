//! AMD Radeon AI PRO R9700 (RDNA 4, gfx1201, Navi 48) descriptor.
//!
//! Same [`SmSpec`]-as-CU mapping as [`crate::amd::mi300`]. RDNA's native wave is
//! 32 lanes, but plow builds its gfx1201 objects with `-mwavefrontsize64` from
//! the CDNA sources, so `warp_lanes` is the 64 the objects actually run at.
//! The matrix engines are WMMA "AI accelerators" (2 per CU), not MFMA.
//!
//! Matrix rates are AMD RDNA 4 whitepaper / R9700 spec-sheet dense peaks at the
//! 2920 MHz boost: FP16/BF16 ~191 TFLOPS, FP8 ~383 TFLOPS, INT8 ~383 TOPS. In the
//! [`MatrixThroughput`] unit (MACs/cycle/core) that is 191e12 / (2 x 64 CUs x 2
//! cores x 2.92e9) = 256 for bf16, the same anchor as a CDNA 3 MFMA core.

use crate::spec::{Arch, GpuSpec, MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor};
use crate::units::{Bytes, GBps, Hertz};

/// RDNA 4 compute unit as plow uses it (wave64 objects).
const RDNA4_CU: SmSpec = SmSpec {
    warp_lanes: 64,                  // plow builds wave64; native is wave32
    shared_mem: Bytes::kib(64),      // a WGP's 128 KiB LDS is shared by 2 CUs
    l1_shared_total: Bytes::kib(64), // LDS is a dedicated pool
    regs_32bit: 98_304,              // 2 SIMD32 x 192 KiB VGPR file (approx.)
    max_threads: 2048,               // approx.
    max_warps: 32,                   // wave64 residency (approx.)
    max_blocks: 16,                  // approx.
    tensor_cores: 2,                 // AI accelerators per CU
    mma: MatrixThroughput {
        fp16: 256,
        bf16: 256,
        fp8: Some(512),
        fp4: None,
        int8: Some(512),
    },
    tmem: Bytes(0),
};

/// Radeon AI PRO R9700 — 64 CUs, 32 GiB GDDR6 (256-bit), 644.6 GB/s.
pub const R9700: GpuSpec = GpuSpec {
    name: "R9700",
    vendor: Vendor::Amd,
    arch: Arch::RdnaV4,
    compute_cap: (12, 0), // gfx1201
    sm_count: 64,
    sm: RDNA4_CU,
    dsm: None,
    l2: Bytes::mib(8), // plus 64 MiB Infinity Cache (MALL) behind it
    mem: MemorySpec {
        kind: MemKind::Gddr6,
        capacity: Bytes::gib(32),
        bandwidth: GBps(644.6),
        bandwidth_measured: None,
        bus_width_bits: 256,
    },
    copy_engines: 2,
    interconnect: None, // PCIe Gen5 x16 only
    chiplet: None,      // monolithic Navi 48
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(2920),
    soc: None,
};
