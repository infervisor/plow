//! Static per-GPU hardware capability descriptors.
//!
//! This crate answers "what is an H100" — SM count, per-SM shared memory,
//! register file, theoretical HBM bandwidth, HBM/DRAM size, L2, and the GPC
//! grouping of SMs for distributed-shared-memory (DSM) access. It is the
//! source of truth that the scheduler's cost model and parallelism planner
//! read, and from which a runtime `executor_target` is derived.
//!
//! It deliberately does **not** model topology (device count, interconnect) —
//! that is the scheduler's `ClusterTopology`.
//!
//! ```
//! let h100 = hwspec::registry::lookup("H100 SXM5").unwrap();
//! assert_eq!(h100.sm_count, 132);
//! assert_eq!(h100.sm.shared_mem.as_kib(), 228);
//! assert_eq!(h100.mem.capacity.as_gib(), 80);
//! assert_eq!(h100.dsm.unwrap().domain_count, 8);
//! ```

pub mod amd;
pub mod isa;
pub mod nvidia;
pub mod registry;
pub mod spec;
pub mod units;

pub use isa::{CalibrationTier, HardwareFingerprint, IsaCaps, IsaLevel};
pub use spec::{
    Arch, ChipletGrouping, DsmDomainKind, DsmGrouping, GpuSpec, Interconnect, InterconnectKind,
    L2Partitioning, MatrixThroughput, MemKind, MemorySpec, MmaDtype, SmSpec, Vendor,
};
pub use units::{Bytes, GBps, Hertz};
