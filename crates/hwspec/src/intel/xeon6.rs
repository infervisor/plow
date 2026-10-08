//! Intel Xeon 6 P-core (Granite Rapids). Core/cache figures from `lscpu` on a 6975P-C; memory
//! bandwidth is the DDR5-6400 x 12-channel datasheet peak (not measured on this part).

use crate::spec::{Arch, ChipletGrouping, GpuSpec, MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor};
use crate::units::{Bytes, GBps, Hertz};

/// AMX TMUL per core: 1024 int8 / 512 bf16 / 512 fp16 (AMX-FP16) MACs per cycle. No fp8/fp4.
const GNR_AMX: MatrixThroughput = MatrixThroughput {
    fp16: 512,
    bf16: 512,
    fp8: None,
    fp4: None,
    int8: Some(1024),
};

/// One Granite Rapids P-core: 2 SMT threads, 32 zmm registers of 16 fp32 lanes, 48 KiB L1d,
/// 2 MiB private L2.
const GNR_CORE: SmSpec = SmSpec {
    warp_lanes: 16,
    shared_mem: Bytes::mib(2),
    l1_shared_total: Bytes::kib(48),
    regs_32bit: 32 * 16,
    max_threads: 2,
    max_warps: 2,
    max_blocks: 2,
    tensor_cores: 1,
    mma: GNR_AMX,
    tmem: Bytes(0),
};

/// Xeon 6975P-C — 96 P-cores on three compute dies (SNC3), 480 MiB L3, 12-channel DDR5.
pub const XEON_6975P_C: GpuSpec = GpuSpec {
    name: "Xeon 6975P-C",
    vendor: Vendor::Intel,
    arch: Arch::GraniteRapids,
    compute_cap: (6, 0),
    sm_count: 96,
    sm: GNR_CORE,
    dsm: None,
    l2: Bytes::mib(480),
    mem: MemorySpec {
        kind: MemKind::Ddr5,
        capacity: Bytes::gib(384),
        bandwidth: GBps(614.4),
        bandwidth_measured: None,
        bus_width_bits: 768,
    },
    // DSA engines exist but the CPU engine copies with cores; one DMA timeline for the scheduler.
    copy_engines: 1,
    interconnect: None,
    chiplet: Some(ChipletGrouping {
        chiplet_count: 3,
        sms_per_chiplet: 32,
        l2_per_chiplet: Bytes::mib(160),
    }),
    l2_partitioning: None,
    clock_boost: Hertz::from_mhz(3900),
    soc: None,
};
