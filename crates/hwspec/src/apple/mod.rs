//! Apple Silicon SoC descriptors. One [`GpuSpec`] per chip describes the GPU the way the
//! NVIDIA/AMD entries do (executor = GPU core, 32-lane simdgroups, 32 KiB threadgroup memory)
//! and carries the rest of the SoC in [`crate::spec::SocSpec`]: CPU performance/efficiency
//! cores, the Neural Engine, and the one unified memory bus they all share.
pub mod m4;
