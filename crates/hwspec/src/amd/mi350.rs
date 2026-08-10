//! AMD Instinct MI350 series (CDNA 4, gfx950) descriptors.
//!
//! Same [`SmSpec`]-as-CU mapping as the MI300 series (see [`crate::amd::mi300`]):
//! 64-wide wavefronts, LDS as "shared memory", MFMA units as matrix cores, and
//! no cross-CU shared memory (`dsm: None`).
//!
//! The CDNA 4 deltas that matter to the cost model and the kernels:
//!
//! * **LDS grows 64 KiB → 160 KiB per CU.** This is the big one — it is what
//!   lets a head_dim=512 attention tile stage Q, K and V simultaneously
//!   (Tier C).
//! * **Double-K bf16 MFMA.** CDNA 3 tops out at `v_mfma_f32_16x16x16_bf16`;
//!   gfx950 adds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_32x32x16_bf16`,
//!   doubling the contraction depth per instruction and with it the per-CU
//!   bf16 rate.
//! * **8 XCDs × 32 CUs** (MI300 was 8 × 38).
//! * **FP4/FP6 datapaths** exist (unused by this sprint — Gemma 4 is bf16).
//!
//! The CU count, LDS size, clock, HBM capacity and L2 slice below are *measured*
//! from `hipGetDeviceProperties` on an MI350X (`gfx950:sramecc+:xnack-`):
//! 256 CUs, 163840 B LDS/CU, 2200 MHz, 309.2 GB, 4 MiB L2 per XCD, wave64.
//! The MFMA throughput figures and the xGMI bandwidth remain datasheet peaks —
//! they are not observable through the HIP device API.

use crate::spec::{
    Arch, ChipletGrouping, GpuSpec, Interconnect, InterconnectKind, L2Partitioning,
    MatrixThroughput, MemKind, MemorySpec, SmSpec, Vendor,
};
use crate::units::{Bytes, GBps, Hertz};

/// MI350 L2 partitioning: 8 XCDs × 4 MiB. Mirrors [`crate::amd::mi300`], but at
/// 32 CUs per XCD rather than 38, and with a higher per-partition bandwidth to
/// track the faster HBM3E behind it.
const MI350_L2: L2Partitioning = L2Partitioning {
    partition_count: 8,
    bytes_per_partition: Bytes::mib(4),
    sms_per_partition: 32,
    bandwidth_per_partition: GBps(375.0),
};

/// CDNA 4 compute unit (gfx950).
const CDNA4_CU: SmSpec = SmSpec {
    warp_lanes: 64,                   // wavefront width (unchanged from CDNA 3)
    shared_mem: Bytes::kib(160),      // LDS per CU — 2.5× CDNA 3
    l1_shared_total: Bytes::kib(160), // LDS is a dedicated pool (not unified with L1)
    regs_32bit: 131_072,              // 512 KiB VGPR file / CU (4 SIMDs × 128 KiB)
    max_threads: 2048,                // 32 wavefronts × 64 lanes per CU
    max_warps: 32,                    // resident wavefronts per CU
    max_blocks: 16,                   // resident workgroups per CU (approx.)
    tensor_cores: 4,                  // MFMA matrix cores (one per SIMD)
    // CDNA4 doubles the bf16/fp16 contraction depth per MFMA (k16 → k32), so the
    // per-cycle MAC rate per matrix core doubles against the CDNA3 256 anchor.
    // fp8 keeps its 2× ratio over bf16; fp4 is new on this generation.
    mma: MatrixThroughput {
        fp16: 512,
        bf16: 512,
        fp8: Some(1024),
        fp4: Some(2048),
        int8: Some(1024),
    },
    tmem: Bytes(0),
};

/// MI350X — 256 CUs (8 XCDs × 32), 288 GiB HBM3E, ~8 TB/s.
pub const MI350X: GpuSpec = GpuSpec {
    name: "MI350X",
    vendor: Vendor::Amd,
    arch: Arch::CdnaV4,
    compute_cap: (9, 5), // gfx950
    sm_count: 256,
    sm: CDNA4_CU,
    dsm: None,
    l2: Bytes::mib(32), // 8 XCDs × 4 MiB L2 (separate from Infinity Cache)
    mem: MemorySpec {
        kind: MemKind::Hbm3e,
        capacity: Bytes::gib(288),
        bandwidth: GBps(8000.0),
        // 6200 GB/s MEASURED whole-GPU streaming read (`runtime/amd/op_gemm.h:38`), i.e. 77.5% of
        // the 8000 GB/s datasheet peak above. The measured figure GOVERNS every reported floor:
        // Gemma-4-31B bf16 streams 61.4 GB of weights per decode step, which is 9.90 ms at 6200
        // and 7.68 ms at 8000 — and the isolated decode GEMV is measured at 95–103% of the 6200
        // ceiling, so 6200 is where the hardware actually is, not a derate we chose. Reporting the
        // 8000-based number claims 2.2 ms of headroom that does not exist.
        // Inherited by MI355X below (`..MI350X`) — same die, same HBM.
        bandwidth_measured: Some(GBps(6200.0)),
        bus_width_bits: 8192,
    },
    copy_engines: 4, // SDMA engines (approx.)
    // Infinity Fabric (xGMI): ~1075 GB/s/GPU, 8-GPU fully-connected mesh.
    interconnect: Some(Interconnect {
        kind: InterconnectKind::InfinityFabric,
        per_gpu_bandwidth: GBps(1075.0),
        domain_size: 8,
    }),
    // 8 XCDs, each with 32 enabled CUs sharing a 4 MiB L2 slice.
    chiplet: Some(ChipletGrouping {
        chiplet_count: 8,
        sms_per_chiplet: 32,
        l2_per_chiplet: Bytes::mib(4),
    }),
    l2_partitioning: Some(MI350_L2),
    clock_boost: Hertz::from_mhz(2200), // measured
};

/// MI355X — same 256-CU die as [`MI350X`], liquid-cooled, higher sustained clock.
pub const MI355X: GpuSpec = GpuSpec {
    name: "MI355X",
    clock_boost: Hertz::from_mhz(2400),
    ..MI350X
};
