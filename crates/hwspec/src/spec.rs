//! Static hardware capability descriptors.
//!
//! A [`GpuSpec`] is the source of truth for "what is an H100" — independent of
//! how many you have (that is the scheduler's `ClusterTopology`) and of which
//! physical SM a tile lands on (that is the runtime `executor_target`). The
//! cost model reads bandwidth/SRAM limits from here; the parallelism planner
//! reads `mem.capacity` to check that a shard fits.

use crate::units::{Bytes, GBps, Hertz};

/// Silicon vendor. Determines which concurrency-unit vocabulary applies
/// (warp vs wave) and whether distributed shared memory exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
}

/// Microarchitecture generation. Drives tile-shape selection in the cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// NVIDIA Ada Lovelace (RTX 4090, L40S). SM 8.9, mma.sync m16n8k16.
    AdaLovelace,
    Hopper,
    /// NVIDIA Blackwell (RTX 5090, B200, RTX 6000 Pro). SM 10.0, wgmma.
    Blackwell,
    /// AMD CDNA 3 (MI300 series).
    CdnaV3,
    /// AMD CDNA 4 (MI350 series). gfx950: 160 KiB LDS, double-K bf16 MFMA.
    CdnaV4,
}

/// On-package memory technology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Hbm2e,
    Hbm3,
    Hbm3e,
    /// GDDR6X (Ada consumer: RTX 4090).
    Gddr6x,
    /// GDDR7 (Blackwell consumer: RTX 5090, RTX 6000 Pro).
    Gddr7,
}

/// The cross-SM shared-memory domain kind. On Hopper, SMs within one GPC
/// (Graphics Processing Cluster) share an SM-to-SM network and can address
/// each other's shared memory via thread-block clusters — "distributed shared
/// memory" (DSM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsmDomainKind {
    /// NVIDIA Graphics Processing Cluster.
    Gpc,
}

/// Matrix-engine operand dtype — selects the tensor-core / MFMA throughput tier
/// (and, via [`GpuSpec::supports`], whether the engine accelerates it at all).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaDtype {
    Fp16,
    Bf16,
    Fp8,
    /// 4-bit float — Blackwell `tcgen05` (and consumer 5th-gen tensor cores).
    Fp4,
    Int8,
}

/// MACs one tensor core / MFMA unit retires per cycle, by operand dtype. The
/// absolute numbers are order-of-magnitude (bf16 = 256 anchors the model); the
/// **ratios** between dtypes and across archs are what the cost model uses.
/// `None` ⇒ the matrix engine does not accelerate that dtype.
#[derive(Clone, Copy, Debug)]
pub struct MatrixThroughput {
    pub fp16: u32,
    pub bf16: u32,
    pub fp8: Option<u32>,
    pub fp4: Option<u32>,
    pub int8: Option<u32>,
}

impl MatrixThroughput {
    /// MACs/cycle for `d`, or `None` where the engine doesn't accelerate it.
    pub const fn of(&self, d: MmaDtype) -> Option<u32> {
        match d {
            MmaDtype::Fp16 => Some(self.fp16),
            MmaDtype::Bf16 => Some(self.bf16),
            MmaDtype::Fp8 => self.fp8,
            MmaDtype::Fp4 => self.fp4,
            MmaDtype::Int8 => self.int8,
        }
    }
}

/// Capabilities of a single streaming multiprocessor (NVIDIA SM; the AMD CU
/// analog reuses this struct with `warp_lanes = 64` and LDS in `shared_mem`).
#[derive(Clone, Copy, Debug)]
pub struct SmSpec {
    /// Lanes per warp/wave (32 on NVIDIA, 64 on CDNA).
    pub warp_lanes: u32,
    /// Maximum *configurable* shared memory per SM (the cap a kernel may opt
    /// into; the scheduler's SRAM-slot constraint is bounded by this).
    pub shared_mem: Bytes,
    /// Unified L1 + shared memory per SM (shared is carved from this).
    pub l1_shared_total: Bytes,
    /// 32-bit register file slots per SM (occupancy input for the cost model).
    pub regs_32bit: u32,
    /// Resident-thread cap per SM.
    pub max_threads: u32,
    /// Resident-warp cap per SM.
    pub max_warps: u32,
    /// Resident-block cap per SM.
    pub max_blocks: u32,
    /// Matrix-engine count per SM (tensor cores / MFMA units).
    pub tensor_cores: u32,
    /// Per-dtype matrix-engine throughput (MACs/cycle/core); the cost model's
    /// compute-rate source. bf16 anchors the model at 256.
    pub mma: MatrixThroughput,
    /// Tensor Memory per SM — Blackwell datacenter (`tcgen05`) only. A separate
    /// on-chip space holding MMA accumulators, distinct from `shared_mem`.
    /// `Bytes(0)` on architectures without it (Hopper, Ada, CDNA, consumer
    /// Blackwell), so the scheduler tracks it only where it exists.
    pub tmem: Bytes,
}

/// How SMs group for distributed shared-memory access. `None` for hardware
/// without cross-SM shared memory (e.g. AMD CDNA, pre-Hopper NVIDIA).
#[derive(Clone, Copy, Debug)]
pub struct DsmGrouping {
    pub domain: DsmDomainKind,
    /// Number of such domains on the die.
    pub domain_count: u32,
    /// SMs physically belonging to one domain (the DSM reachability set).
    pub sms_per_domain: u32,
    /// Maximum thread blocks in one cluster. Hopper allows 16 with opt-in
    /// (8 is the portable default).
    pub max_cluster_blocks: u32,
}

/// On-package chiplet / L2-domain grouping. CUs/SMs within one chiplet share a
/// private L2 cache slice; cross-chiplet traffic traverses an on-package fabric
/// at higher latency. `None` on monolithic dies (H100, consumer Blackwell, Ada).
///
/// This is orthogonal to [`DsmGrouping`]: DSM describes cross-SM shared-memory
/// reachability (a hard constraint), while chiplets describe L2 cache locality
/// (a soft scheduling preference). MI300X has chiplets but no DSM; Hopper has
/// DSM but no chiplets; a future part could have both.
#[derive(Clone, Copy, Debug)]
pub struct ChipletGrouping {
    /// Number of chiplets (L2 domains) on the package.
    pub chiplet_count: u32,
    /// Enabled SMs/CUs per chiplet.
    pub sms_per_chiplet: u32,
    /// L2 cache capacity per chiplet.
    pub l2_per_chiplet: Bytes,
}

/// L2 cache partitioning — every SM sees only its own partition's slice at
/// full L2 bandwidth. Cross-partition traffic traverses the on-die fabric at
/// a lower effective bandwidth. `None` on architectures with a truly unified
/// L2 (rare on modern datacenter GPUs).
///
/// Populated on:
/// - **H100**: `partition_count = 8` matching the 8 GPCs; each GPC owns
///   ~6.25 MiB of the 50 MiB L2. Aligns with [`DsmGrouping`].
/// - **MI300X**: `partition_count = 8` matching the 8 XCDs; each XCD owns
///   4 MiB. Aligns with [`ChipletGrouping`].
///
/// The scheduler uses this to pin producer + consumer to the same L2 slice
/// on the [`LocalityReq::SameL2Partition`] tier — one step weaker than
/// `SameDomain` (no DSM messaging required) but stronger than `SameNode`
/// (must sit within one L2 slice, not just the same GPU).
#[derive(Clone, Copy, Debug)]
pub struct L2Partitioning {
    /// Number of L2 slices on the die.
    pub partition_count: u32,
    /// Bytes per partition (per-GPC on H100, per-XCD on MI300).
    pub bytes_per_partition: Bytes,
    /// SMs bound to each L2 partition — same as `sms_per_domain` on H100
    /// and `sms_per_chiplet` on MI300 by design.
    pub sms_per_partition: u32,
    /// Approximate per-partition L2 bandwidth cap (GB/s). Used by the cost
    /// model to price intra-partition `L2Local` handoffs vs full-L2 access.
    pub bandwidth_per_partition: GBps,
}

/// Fast peer-to-peer fabric between GPUs in one domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterconnectKind {
    /// NVIDIA NVLink (switched via NVSwitch, or bridged point-to-point).
    NvLink,
    /// AMD Infinity Fabric (xGMI).
    InfinityFabric,
}

/// A GPU's fast peer interconnect. `None` on a [`GpuSpec`] means no fast fabric:
/// peers are reachable only over PCIe, which the scheduler treats as a slow link.
#[derive(Clone, Copy, Debug)]
pub struct Interconnect {
    pub kind: InterconnectKind,
    /// Aggregate per-GPU link bandwidth, bidirectional — the figure vendors
    /// quote (e.g. NVLink-4 = 900 GB/s). Becomes the scheduler's interconnect
    /// `BandwidthSet` limit.
    pub per_gpu_bandwidth: GBps,
    /// GPUs reachable over the fast (switched/bridged) fabric in one domain —
    /// the largest tensor-parallel group the planner can keep on fast links.
    pub domain_size: u32,
}

/// On-package memory: size and theoretical peak bandwidth.
#[derive(Clone, Copy, Debug)]
pub struct MemorySpec {
    pub kind: MemKind,
    /// HBM/DRAM capacity.
    pub capacity: Bytes,
    /// Theoretical peak bandwidth (the cost model's HBM-bandwidth limit).
    pub bandwidth: GBps,
    /// ACHIEVED streaming-read bandwidth, measured on this part — `None` when
    /// nobody has measured it.
    ///
    /// Separate from [`Self::bandwidth`] because the two answer different
    /// questions and only one of them is a datasheet quote. `bandwidth` is the
    /// peak the part is SOLD as; a relative cost model (which tile is cheaper?)
    /// wants that, because the factor cancels. Anything that reports an
    /// ABSOLUTE floor — "this decode step cannot go below X µs" — must use the
    /// measured figure, or it hands back headroom that does not exist.
    ///
    /// The instance that motivated the split: `plowc --lean-oracle` divided by
    /// MI350X's 8000 GB/s datasheet number and reported a Gemma-4-31B decode
    /// lower bound of 7719.3 µs, where the measured denominator gives 9.96 ms
    /// — 22.5% optimistic, in the optimistic direction, on the one number whose
    /// whole job is to be a bound.
    pub bandwidth_measured: Option<GBps>,
    pub bus_width_bits: u32,
}

impl MemorySpec {
    /// The denominator for an ABSOLUTE bandwidth bound: measured where we have
    /// it, datasheet peak otherwise.
    ///
    /// Callers computing a lower bound / roofline they intend to REPORT must go
    /// through this rather than reading `bandwidth` directly.
    pub fn bandwidth_for_bound(&self) -> GBps {
        match self.bandwidth_measured {
            Some(b) => b,
            None => self.bandwidth,
        }
    }
}

/// A complete static descriptor for one GPU model/SKU.
#[derive(Clone, Copy, Debug)]
pub struct GpuSpec {
    pub name: &'static str,
    pub vendor: Vendor,
    pub arch: Arch,
    /// CUDA compute capability `(major, minor)` (or vendor equivalent).
    pub compute_cap: (u32, u32),
    /// Enabled SM count on this SKU (may be below the full die).
    pub sm_count: u32,
    pub sm: SmSpec,
    /// SM grouping for distributed shared memory; `None` if unsupported.
    pub dsm: Option<DsmGrouping>,
    /// L2 cache size (shared across all SMs).
    pub l2: Bytes,
    pub mem: MemorySpec,
    /// Asynchronous copy (DMA) engines that overlap transfers with compute; the
    /// scheduler tracks one exclusive DMA timeline per engine.
    pub copy_engines: u32,
    /// Fast peer interconnect (NVLink / Infinity Fabric); `None` ⇒ PCIe-only,
    /// so cross-GPU transfers take the scheduler's slow-link path.
    pub interconnect: Option<Interconnect>,
    /// Chiplet / L2-domain grouping; `None` on monolithic dies.
    pub chiplet: Option<ChipletGrouping>,
    /// L2 cache partitioning; `None` on truly unified-L2 archs. Present on
    /// both H100 (per-GPC L2) and MI300 (per-XCD L2); the scheduler uses
    /// this for the `SameL2Partition` locality tier + intra-partition
    /// placement preference.
    pub l2_partitioning: Option<L2Partitioning>,
    /// Peak boost clock.
    pub clock_boost: Hertz,
}

impl GpuSpec {
    /// Aggregate shared-memory budget across all enabled SMs.
    pub const fn total_shared_mem(&self) -> Bytes {
        Bytes(self.sm.shared_mem.0 * self.sm_count as u64)
    }

    /// Whether this part's matrix engine accelerates operand dtype `d`.
    pub fn supports(&self, d: MmaDtype) -> bool {
        self.sm.mma.of(d).is_some()
    }
}
