//! NVIDIA Ada Lovelace (AD102) descriptors.
//!
//! The RTX 4090 is the full AD102 die — 128 SMs, 24 GiB GDDR6X, ~1.0 TB/s.
//! Ada uses SM 8.9 with `mma.sync` instructions (m16n8k16 family, similar to
//! Ampere but with improved tensor-core throughput). No distributed shared
//! memory (DSM is Hopper+).

use crate::spec::{
    Arch, GpuSpec, MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor,
};
use crate::units::{Bytes, GBps, Hertz};

/// Ada SM: 4th-gen tensor cores, mma.sync m16n8k16.
const ADA_SM: SmSpec = SmSpec {
    warp_lanes: 32,
    shared_mem: Bytes::kib(100),      // max configurable shared mem / SM
    l1_shared_total: Bytes::kib(128), // unified L1 + shared / SM
    regs_32bit: 65_536,               // 256 KiB register file / SM
    max_threads: 1536,
    max_warps: 48,
    max_blocks: 24,
    tensor_cores: 4, // 4th-gen
    // Ada matches Hopper's matrix tiers for fp16/bf16/fp8/int8; no fp4.
    mma: MatrixThroughput {
        fp16: 256,
        bf16: 256,
        fp8: Some(512),
        fp4: None,
        int8: Some(512),
    },
    tmem: Bytes(0),
};

/// RTX 4090 — 128 SMs, 24 GiB GDDR6X, ~1.0 TB/s.
pub const RTX_4090: GpuSpec = GpuSpec {
    name: "RTX 4090",
    vendor: Vendor::Nvidia,
    arch: Arch::AdaLovelace,
    compute_cap: (8, 9),
    sm_count: 128,
    sm: ADA_SM,
    dsm: None, // No distributed shared memory pre-Hopper.
    l2: Bytes::mib(72),
    mem: MemorySpec {
        kind: MemKind::Gddr6x,
        capacity: Bytes::gib(24),
        bandwidth: GBps(1008.0),
        // Not measured on this part; a reported bound falls back to the datasheet peak.
        bandwidth_measured: None,
        bus_width_bits: 384,
    },
    copy_engines: 2,
    interconnect: None, // Ada removed NVLink — PCIe only.
    chiplet: None,      // Monolithic AD102 die.
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(2520),
};
