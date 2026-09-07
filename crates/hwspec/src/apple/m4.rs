//! Apple M4 family (2024–2025). Numbers marked "measured" come from
//! `runtime/apple/probe/probe.m` on an M4 Pro (plans/apple-silicon-backend.md §9); the rest are
//! datasheet figures. Memory capacity is the smallest shipping configuration of each chip — the
//! runtime reads the real size, and a planner must never assume more than the floor.

use crate::spec::{Arch, GpuSpec, MatrixThroughput, MemKind, MemorySpec, SmSpec, SocSpec, Vendor};
use crate::units::{Bytes, GBps, Hertz};

/// Apple GPU cores (M1–M4) have no dedicated matrix engine: `simdgroup_matrix` bf16/fp16 runs
/// on the 128 FP32 ALUs of a core, so the matrix rate equals the FMA rate. fp8/fp4/int8 are
/// not accelerated. (M5 adds per-core "neural accelerators"; that is a different entry.)
const APPLE_G16_MMA: MatrixThroughput = MatrixThroughput {
    fp16: 128,
    bf16: 128,
    fp8: None,
    fp4: None,
    int8: None,
};

/// One Apple GPU core, Apple GPU family 9 (M3/M4). Threadgroup memory is a hard 32 KiB; the
/// register file is dynamically shared with the cache ("Dynamic Caching"), so `regs_32bit` and
/// the occupancy caps are conservative figures rather than documented limits — Metal exposes no
/// occupancy query, and a persistent grid is sized by the co-residency probe at load, not by
/// these fields.
const APPLE_M4_CORE: SmSpec = SmSpec {
    warp_lanes: 32,
    shared_mem: Bytes::kib(32),
    l1_shared_total: Bytes::kib(64),
    regs_32bit: 32_768,
    max_threads: 1536,
    max_warps: 48,
    max_blocks: 32,
    tensor_cores: 0,
    mma: APPLE_G16_MMA,
    tmem: Bytes(0),
};

/// The 16-core Neural Engine every M4 chip carries: 38 int8 TOPS (datasheet). Measured on the
/// M4 Pro through a CoreML fp16 GEMM (128x3072x8192): 4.05 TFLOPS.
const M4_ANE_CORES: u32 = 16;
const M4_ANE_INT8_TOPS: u32 = 38;

/// Apple M4 — 10-core GPU, 120 GB/s LPDDR5X, 4P+4E (the 8-core CPU; the 10-core SKU is 4P+6E).
pub const APPLE_M4: GpuSpec = GpuSpec {
    name: "Apple M4",
    vendor: Vendor::Apple,
    arch: Arch::AppleM4,
    compute_cap: (4, 0),
    sm_count: 10,
    sm: APPLE_M4_CORE,
    dsm: None,
    l2: Bytes::mib(8), // system level cache
    mem: MemorySpec {
        kind: MemKind::Lpddr5x,
        capacity: Bytes::gib(16),
        bandwidth: GBps(120.0),
        bandwidth_measured: None,
        bus_width_bits: 128,
    },
    copy_engines: 1,
    interconnect: None,
    chiplet: None,
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(1470),
    soc: Some(SocSpec {
        cpu_p_cores: 4,
        cpu_e_cores: 4,
        cpu_bandwidth: GBps(100.0),
        cpu_bandwidth_measured: None,
        ane_cores: M4_ANE_CORES,
        ane_int8_tops: M4_ANE_INT8_TOPS,
        ane_fp16_tflops_measured: None,
        unified_memory: true,
    }),
};

/// Apple M4 Pro — 16-core GPU (a 20-core SKU exists), 273 GB/s, 8P+4E (the 12-core CPU; the
/// 14-core SKU is 10P+4E). This is the bring-up box for plans/apple-silicon-backend.md.
pub const APPLE_M4_PRO: GpuSpec = GpuSpec {
    name: "Apple M4 Pro",
    vendor: Vendor::Apple,
    arch: Arch::AppleM4,
    compute_cap: (4, 0),
    sm_count: 16,
    sm: APPLE_M4_CORE,
    dsm: None,
    l2: Bytes::mib(24),
    mem: MemorySpec {
        kind: MemKind::Lpddr5x,
        capacity: Bytes::gib(24),
        bandwidth: GBps(273.0),
        // probe (e): bf16 GEMV over 2 GiB, 255.1 GB/s from the GPU alone.
        bandwidth_measured: Some(GBps(255.0)),
        bus_width_bits: 256,
    },
    copy_engines: 1,
    interconnect: None,
    chiplet: None,
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(1578),
    soc: Some(SocSpec {
        cpu_p_cores: 8,
        cpu_e_cores: 4,
        cpu_bandwidth: GBps(240.0),
        // probe (e): 8 P-core threads streaming 2 GiB, 239.2 GB/s — but the bus is shared:
        // GPU + CPU together measured 247.5 GB/s, so the CPU adds nothing to a GPU stream.
        cpu_bandwidth_measured: Some(GBps(239.0)),
        ane_cores: M4_ANE_CORES,
        ane_int8_tops: M4_ANE_INT8_TOPS,
        ane_fp16_tflops_measured: Some(4.05),
        unified_memory: true,
    }),
};

/// Apple M4 Max — 32-core GPU (a 40-core SKU at 546 GB/s exists), 410 GB/s, 10P+4E.
pub const APPLE_M4_MAX: GpuSpec = GpuSpec {
    name: "Apple M4 Max",
    vendor: Vendor::Apple,
    arch: Arch::AppleM4,
    compute_cap: (4, 0),
    sm_count: 32,
    sm: APPLE_M4_CORE,
    dsm: None,
    l2: Bytes::mib(32),
    mem: MemorySpec {
        kind: MemKind::Lpddr5x,
        capacity: Bytes::gib(36),
        bandwidth: GBps(410.0),
        bandwidth_measured: None,
        bus_width_bits: 384,
    },
    copy_engines: 1,
    interconnect: None,
    chiplet: None,
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(1578),
    soc: Some(SocSpec {
        cpu_p_cores: 10,
        cpu_e_cores: 4,
        cpu_bandwidth: GBps(300.0),
        cpu_bandwidth_measured: None,
        ane_cores: M4_ANE_CORES,
        ane_int8_tops: M4_ANE_INT8_TOPS,
        ane_fp16_tflops_measured: None,
        unified_memory: true,
    }),
};
