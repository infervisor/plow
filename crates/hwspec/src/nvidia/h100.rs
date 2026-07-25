//! NVIDIA H100 (GH100, Hopper) descriptors.
//!
//! The three SKUs share an identical SM and GPC/DSM structure; they differ only
//! in enabled SM count and the memory subsystem. Numbers are theoretical peaks
//! from NVIDIA's Hopper architecture / H100 datasheets.

use crate::spec::{
    Arch, DsmDomainKind, DsmGrouping, GpuSpec, Interconnect, InterconnectKind, L2Partitioning,
    MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor,
};
use crate::units::{Bytes, GBps, Hertz};

/// Hopper SM: identical across H100 SKUs.
const HOPPER_SM: SmSpec = SmSpec {
    warp_lanes: 32,
    shared_mem: Bytes::kib(228),      // max configurable shared mem / SM
    l1_shared_total: Bytes::kib(256), // unified L1 + shared / SM
    regs_32bit: 65_536,               // 256 KiB register file / SM
    max_threads: 2048,
    max_warps: 64,
    max_blocks: 32,
    tensor_cores: 4, // 4th-gen
    // Hopper accelerates fp16/bf16 at the 256 anchor, fp8 (and int8) at 2×; no fp4.
    mma: MatrixThroughput {
        fp16: 256,
        bf16: 256,
        fp8: Some(512),
        fp4: None,
        int8: Some(512),
    },
    tmem: Bytes(0),
};

/// GPC structure: 8 GPCs, each a DSM reachability domain. A thread-block
/// cluster spans SMs within one GPC (up to 16 blocks with opt-in, 8 portable).
const GH100_DSM: DsmGrouping = DsmGrouping {
    domain: DsmDomainKind::Gpc,
    domain_count: 8,
    sms_per_domain: 18, // full GPC = 9 TPCs × 2 SMs
    max_cluster_blocks: 16,
};

/// H100 L2 partitioning: 50 MiB split across 8 GPCs = 6.25 MiB per GPC.
/// Per-partition L2 bandwidth is ~1.5 TB/s (aggregate 12 TB/s / 8), well
/// above HBM per-SM share and enabling intra-GPC data reuse.
const GH100_L2: L2Partitioning = L2Partitioning {
    partition_count: 8,
    bytes_per_partition: Bytes(6_553_600), // 6.25 MiB
    sms_per_partition: 18,
    bandwidth_per_partition: GBps(1500.0),
};

/// H100 SXM5 — the flagship; 132 SMs, 80 GiB HBM3, ~3.35 TB/s.
pub const H100_SXM5: GpuSpec = GpuSpec {
    name: "H100 SXM5",
    vendor: Vendor::Nvidia,
    arch: Arch::Hopper,
    compute_cap: (9, 0),
    sm_count: 132,
    sm: HOPPER_SM,
    dsm: Some(GH100_DSM),
    l2: Bytes::mib(50),
    mem: MemorySpec {
        kind: MemKind::Hbm3,
        capacity: Bytes::gib(80),
        bandwidth: GBps(3352.0),
        bus_width_bits: 5120,
    },
    copy_engines: 3,
    // NVLink-4: 18 links × 50 GB/s = 900 GB/s/GPU, NVSwitch domain of 8 (HGX H100).
    interconnect: Some(Interconnect {
        kind: InterconnectKind::NvLink,
        per_gpu_bandwidth: GBps(900.0),
        domain_size: 8,
    }),
    chiplet: None, // Monolithic GH100 die.
    l2_partitioning: Some(GH100_L2),
    clock_boost: Hertz::from_mhz(1980),
};

/// H200 SXM — Hopper GH100 with 141 GiB HBM3e and 4.8 TB/s memory bandwidth.
pub const H200_SXM: GpuSpec = GpuSpec {
    name: "H200 SXM",
    vendor: Vendor::Nvidia,
    arch: Arch::Hopper,
    compute_cap: (9, 0),
    sm_count: 132,
    sm: HOPPER_SM,
    dsm: Some(GH100_DSM),
    l2: Bytes::mib(50),
    mem: MemorySpec {
        kind: MemKind::Hbm3e,
        capacity: Bytes::gib(141),
        bandwidth: GBps(4800.0),
        bus_width_bits: 6144,
    },
    copy_engines: 3,
    interconnect: Some(Interconnect {
        kind: InterconnectKind::NvLink,
        per_gpu_bandwidth: GBps(900.0),
        domain_size: 8,
    }),
    chiplet: None,
    l2_partitioning: Some(GH100_L2),
    clock_boost: Hertz::from_mhz(1830),
};

/// H100 PCIe — 114 SMs, 80 GiB HBM2e, ~2.0 TB/s, lower boost clock.
pub const H100_PCIE: GpuSpec = GpuSpec {
    name: "H100 PCIe",
    vendor: Vendor::Nvidia,
    arch: Arch::Hopper,
    compute_cap: (9, 0),
    sm_count: 114,
    sm: HOPPER_SM,
    dsm: Some(GH100_DSM),
    l2: Bytes::mib(50),
    mem: MemorySpec {
        kind: MemKind::Hbm2e,
        capacity: Bytes::gib(80),
        bandwidth: GBps(2039.0),
        bus_width_bits: 5120,
    },
    copy_engines: 3,
    // PCIe card with optional NVLink bridge: 600 GB/s between a bonded pair.
    interconnect: Some(Interconnect {
        kind: InterconnectKind::NvLink,
        per_gpu_bandwidth: GBps(600.0),
        domain_size: 2,
    }),
    chiplet: None, // Monolithic GH100 die.
    l2_partitioning: Some(GH100_L2),
    clock_boost: Hertz::from_mhz(1755),
};

/// H100 NVL (per-GPU of the paired board) — 132 SMs, 94 GiB HBM3e, ~3.9 TB/s.
pub const H100_NVL: GpuSpec = GpuSpec {
    name: "H100 NVL",
    vendor: Vendor::Nvidia,
    arch: Arch::Hopper,
    compute_cap: (9, 0),
    sm_count: 132,
    sm: HOPPER_SM,
    dsm: Some(GH100_DSM),
    l2: Bytes::mib(50),
    mem: MemorySpec {
        kind: MemKind::Hbm3e,
        capacity: Bytes::gib(94),
        bandwidth: GBps(3938.0),
        bus_width_bits: 6144,
    },
    copy_engines: 3,
    // NVL bridges two GPUs at 600 GB/s.
    interconnect: Some(Interconnect {
        kind: InterconnectKind::NvLink,
        per_gpu_bandwidth: GBps(600.0),
        domain_size: 2,
    }),
    chiplet: None, // Monolithic GH100 die.
    l2_partitioning: Some(GH100_L2),
    clock_boost: Hertz::from_mhz(1785),
};
