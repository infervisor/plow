//! DeepSeek-V4.1-Flash on NVIDIA sm_90a (H100/H200): a host-driven engine over dedicated kernels
//! (`runtime/nvidia/dsv41/`, built by `scripts/build_dsv41_sm90a.sh`), pipeline-parallel across
//! GPUs, with the Engram tables left in host memory.
//!
//! This is not the packet/megakernel path: V4.1 has no NVIDIA emit, no decode emit on any backend,
//! and the sm_90a interpreter has no multi-GPU collectives yet. The kernels reproduce the checkpoint
//! reference's numerics (`inference/kernel.py`), checked by `scripts/dsv41_nv/test_kernels.py` and,
//! per layer against the reference model, `scripts/dsv41_nv/test_layers.py`.

pub mod config;
pub mod engine;
pub mod kernels;
pub mod serve;
pub mod stage;
pub mod weights;
