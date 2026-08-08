//! NVIDIA Blackwell consumer / pro descriptors.
//!
//! Consumer/workstation Blackwell (GB202) is SM 12.0 with **Ada-class per-SM
//! limits** (100 KiB max shared, 128 KiB L1, 1536 threads / 48 warps) — the
//! big-SM configuration (228 KiB shared, 2048 threads) is datacenter-only
//! (B200/GB200, SM 10.0 `tcgen05`).
//!
//! RTX 5090 — GB202, 170 SMs, 32 GiB GDDR7, ~1.8 TB/s.
//! RTX PRO 6000 Blackwell — full-fat GB202, 188 SMs, 96 GB GDDR7, ~1.8 TB/s.

use crate::spec::{
    Arch, DsmDomainKind, DsmGrouping, GpuSpec, Interconnect, InterconnectKind, L2Partitioning,
    MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor,
};

/// Blackwell 5th-gen matrix tiers: fp16/bf16 at 256, fp8/int8 at 2×, and fp4 at
/// 4× via the new low-precision path (consumer 5th-gen + datacenter `tcgen05`).
const BLACKWELL_MMA: MatrixThroughput = MatrixThroughput {
    fp16: 256,
    bf16: 256,
    fp8: Some(512),
    fp4: Some(1024),
    int8: Some(512),
};
use crate::units::{Bytes, GBps, Hertz};

/// Consumer Blackwell SM (SM 12.0): 5th-gen tensor cores, Ada-class occupancy
/// and shared-memory limits. Tiles budgeted against the datacenter 228 KiB
/// configuration cannot launch on these parts.
const CONSUMER_BLACKWELL_SM: SmSpec = SmSpec {
    warp_lanes: 32,
    shared_mem: Bytes::kib(100),      // max configurable shared mem / SM
    l1_shared_total: Bytes::kib(128), // unified L1 + shared / SM
    regs_32bit: 65_536,               // 256 KiB register file / SM
    max_threads: 1536,
    max_warps: 48,
    max_blocks: 24,
    tensor_cores: 4, // 5th-gen
    mma: BLACKWELL_MMA,
    tmem: Bytes(0), // consumer Blackwell: no tcgen05 TMEM
};

// GB202 has NO distributed shared memory. Measured on an RTX 5090 (sm_120a):
// thread-block cluster size is 1, so there are no DSM reachability domains and
// no SM-to-SM handoff. NVIDIA's generic "cc 12.x" table lumps 12.0 together with
// 12.1 and wrongly lists clusters as available -- that entry is for 12.1.
//
// A `GB202_DSM` grouping used to live here (12 GPCs x 16 SMs, transcribed from
// the GH100 Hopper entry) and was wired into both GB202 SKUs. It made
// `cost.rs` return a finite HBM/4 instead of `Cycles::MAX`, and `collapse.rs`
// emit `DsmHandoff` edges that `cheapest()` then preferred over HBM -- i.e. the
// planner was scheduling SM-to-SM handoffs this silicon cannot perform, and
// `machine.rs` modelled 12 domains of 16 SMs rather than one flat 170/188.
// No test failed when it was removed: the fiction was unasserted in both
// directions. Do not reintroduce it without hardware evidence.

/// Datacenter Blackwell SM (B200, GB200; SM 10.0 `tcgen05`). Adds Tensor Memory
/// (TMEM): a separate 256 KiB/SM on-chip space holding MMA accumulators.
const B200_SM: SmSpec = SmSpec {
    warp_lanes: 32,
    shared_mem: Bytes::kib(228),
    l1_shared_total: Bytes::kib(256),
    regs_32bit: 65_536,
    max_threads: 2048,
    max_warps: 64,
    max_blocks: 32,
    tensor_cores: 4, // 5th-gen, tcgen05
    mma: BLACKWELL_MMA,
    tmem: Bytes::kib(256),
};

/// GB100 GPC/DSM structure (one die of the B200 package).
const GB100_DSM: DsmGrouping = DsmGrouping {
    domain: DsmDomainKind::Gpc,
    domain_count: 8,
    sms_per_domain: 18,
    max_cluster_blocks: 16,
};

/// B200 L2 partitioning: 126 MiB across 8 GPCs = ~16 MiB per GPC.
/// Per-partition bandwidth ~2 TB/s (aggregate ~16 TB/s / 8).
const GB100_L2: L2Partitioning = L2Partitioning {
    partition_count: 8,
    bytes_per_partition: Bytes(16_515_072), // ~15.75 MiB (126 / 8)
    sms_per_partition: 18,
    bandwidth_per_partition: GBps(2000.0),
};

/// B200 — datacenter Blackwell, 148 SMs (per die), 192 GiB HBM3e, ~8 TB/s.
pub const B200: GpuSpec = GpuSpec {
    name: "B200",
    vendor: Vendor::Nvidia,
    arch: Arch::Blackwell,
    compute_cap: (10, 0),
    sm_count: 148,
    sm: B200_SM,
    dsm: Some(GB100_DSM),
    l2: Bytes::mib(126),
    mem: MemorySpec {
        kind: MemKind::Hbm3e,
        capacity: Bytes::gib(192),
        bandwidth: GBps(8000.0),
        // Not measured on this part; a reported bound falls back to the datasheet peak.
        bandwidth_measured: None,
        bus_width_bits: 8192,
    },
    copy_engines: 3,
    // NVLink-5: 1800 GB/s/GPU, NVSwitch domain of 8 (HGX B200).
    interconnect: Some(Interconnect {
        kind: InterconnectKind::NvLink,
        per_gpu_bandwidth: GBps(1800.0),
        domain_size: 8,
    }),
    chiplet: None, // Each die is a separate unit in the Soc; no intra-unit chiplets.
    l2_partitioning: Some(GB100_L2),
    clock_boost: Hertz::from_mhz(1965),
};

/// RTX 5090 — 170 SMs, 32 GiB GDDR7, ~1.8 TB/s.
pub const RTX_5090: GpuSpec = GpuSpec {
    name: "RTX 5090",
    vendor: Vendor::Nvidia,
    arch: Arch::Blackwell,
    compute_cap: (12, 0),
    sm_count: 170,
    sm: CONSUMER_BLACKWELL_SM,
    dsm: None, // GB202 cluster size 1 -- no DSM (measured)
    l2: Bytes::mib(96),
    mem: MemorySpec {
        kind: MemKind::Gddr7,
        capacity: Bytes::gib(32),
        bandwidth: GBps(1792.0),
        // Not measured on this part; a reported bound falls back to the datasheet peak.
        bandwidth_measured: None,
        bus_width_bits: 512,
    },
    copy_engines: 2,
    interconnect: None, // consumer Blackwell dropped NVLink — PCIe only.
    chiplet: None,      // Monolithic GB202 die.
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(2407),
};

/// RTX PRO 6000 Blackwell — full-fat GB202, 188 SMs, 96 GB GDDR7, ~1.8 TB/s.
pub const RTX_6000_PRO: GpuSpec = GpuSpec {
    name: "RTX 6000 Pro Blackwell",
    vendor: Vendor::Nvidia,
    arch: Arch::Blackwell,
    compute_cap: (12, 0),
    sm_count: 188,
    sm: CONSUMER_BLACKWELL_SM,
    dsm: None, // GB202 cluster size 1 -- no DSM (measured)
    l2: Bytes::mib(96),
    mem: MemorySpec {
        kind: MemKind::Gddr7,
        capacity: Bytes::gib(96),
        bandwidth: GBps(1792.0),
        // Not measured on this part; a reported bound falls back to the datasheet peak.
        bandwidth_measured: None,
        bus_width_bits: 512,
    },
    copy_engines: 2,
    interconnect: None, // workstation SKU — PCIe only.
    chiplet: None,      // Monolithic GB202 die.
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(2617),
};
