//! AMD Instinct MI300 series (CDNA 3) descriptors.
//!
//! The NVIDIA-shaped [`SmSpec`] maps onto a CDNA *compute unit* (CU): the
//! concurrency unit is a 64-wide wavefront, "shared memory" is the LDS, and the
//! matrix engines are MFMA units. CDNA has no cross-CU shared memory, so
//! `dsm` is `None`.
//!
//! MI300X and MI325X are the same compute die (304 CUs); they differ only in
//! the HBM subsystem. Numbers are theoretical peaks from AMD's CDNA 3 / MI300
//! datasheets except where a sustained machine measurement is recorded explicitly.

use crate::spec::{
    Arch, ChipletGrouping, GpuSpec, Interconnect, InterconnectKind, L2Partitioning,
    MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor,
};
use crate::units::{Bytes, GBps, Hertz};

/// MI300 L2 partitioning: 32 MiB split across 8 XCDs = 4 MiB per XCD.
/// Per-partition L2 bandwidth ~250 GB/s (much lower than aggregate Infinity
/// Cache, but still 5× HBM per-CU share). Aligns with `ChipletGrouping`.
const MI300_L2: L2Partitioning = L2Partitioning {
    partition_count: 8,
    bytes_per_partition: Bytes::mib(4),
    sms_per_partition: 38,
    bandwidth_per_partition: GBps(250.0),
};

/// CDNA 3 compute unit: identical across the MI300 series.
const CDNA3_CU: SmSpec = SmSpec {
    warp_lanes: 64,                  // wavefront width
    shared_mem: Bytes::kib(64),      // LDS per CU
    l1_shared_total: Bytes::kib(64), // LDS is a dedicated pool (not unified with L1)
    regs_32bit: 131_072,             // 512 KiB VGPR file / CU (4 SIMDs × 128 KiB)
    max_threads: 2048,               // 32 wavefronts × 64 lanes per CU
    max_warps: 32,                   // resident wavefronts per CU
    max_blocks: 16,                  // resident workgroups per CU (approx.)
    tensor_cores: 4,                 // MFMA matrix cores (one per SIMD)
    // CDNA3 MFMA: fp16/bf16 at the 256 anchor, fp8/int8 at 2×; no fp4.
    mma: MatrixThroughput {
        fp16: 256,
        bf16: 256,
        fp8: Some(512),
        fp4: None,
        int8: Some(512),
    },
    tmem: Bytes(0),
};

/// MI300X — 304 CUs, 192 GiB HBM3, ~5.3 TB/s.
pub const MI300X: GpuSpec = GpuSpec {
    name: "MI300X",
    vendor: Vendor::Amd,
    arch: Arch::CdnaV3,
    compute_cap: (9, 4), // gfx942
    sm_count: 304,
    sm: CDNA3_CU,
    dsm: None,
    l2: Bytes::mib(32), // 8 XCDs × 4 MiB L2 (separate from 256 MiB Infinity Cache)
    mem: MemorySpec {
        kind: MemKind::Hbm3,
        capacity: Bytes::gib(192),
        bandwidth: GBps(5325.0),
        // Not measured on this part; a reported bound falls back to the datasheet peak.
        bandwidth_measured: None,
        bus_width_bits: 8192,
    },
    copy_engines: 4, // SDMA engines (approx.)
    // Infinity Fabric (xGMI): ~896 GB/s/GPU, 8-GPU fully-connected mesh.
    interconnect: Some(Interconnect {
        kind: InterconnectKind::InfinityFabric,
        per_gpu_bandwidth: GBps(896.0),
        domain_size: 8,
    }),
    // 8 XCDs, each with 38 enabled CUs sharing a 4 MiB L2 slice.
    chiplet: Some(ChipletGrouping {
        chiplet_count: 8,
        sms_per_chiplet: 38,
        l2_per_chiplet: Bytes::mib(4),
    }),
    l2_partitioning: Some(MI300_L2),
    clock_boost: Hertz::from_mhz(2100),
};

/// MI325X — same 304-CU die, 256 GiB HBM3e, ~6.0 TB/s.
pub const MI325X: GpuSpec = GpuSpec {
    name: "MI325X",
    vendor: Vendor::Amd,
    arch: Arch::CdnaV3,
    compute_cap: (9, 4), // gfx942
    sm_count: 304,
    sm: CDNA3_CU,
    dsm: None,
    l2: Bytes::mib(32),
    mem: MemorySpec {
        kind: MemKind::Hbm3e,
        capacity: Bytes::gib(256),
        bandwidth: GBps(6000.0),
        // Median stream result over three clean 16 GB leased runs on this MI325X with the
        // flake-pinned ROCm 7.14 toolchain (runtime/tests/decode_bw_probe.hip).
        bandwidth_measured: Some(GBps(4164.0)),
        bus_width_bits: 8192,
    },
    copy_engines: 4, // SDMA engines (approx.)
    // Infinity Fabric (xGMI): ~896 GB/s/GPU, 8-GPU fully-connected mesh.
    interconnect: Some(Interconnect {
        kind: InterconnectKind::InfinityFabric,
        per_gpu_bandwidth: GBps(896.0),
        domain_size: 8,
    }),
    // 8 XCDs, each with 38 enabled CUs sharing a 4 MiB L2 slice.
    chiplet: Some(ChipletGrouping {
        chiplet_count: 8,
        sms_per_chiplet: 38,
        l2_per_chiplet: Bytes::mib(4),
    }),
    l2_partitioning: Some(MI300_L2),
    clock_boost: Hertz::from_mhz(2100),
};
