//! Block contract definitions for kernel capabilities and qualification.
//!
//! Pairs kernel specifications with analytical arithmetic intensity, memory bounds,
//! and empirical certification gates.

use hwspec::{GpuSpec, MmaDtype};

use crate::spec::{KernelSpec, Phase};

/// Bottleneck category for a block or kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundCategory {
    MemoryBoundHbm,
    MemoryBoundL2,
    ComputeBoundTensorCore,
    ComputeBoundVectorAlu,
}

impl BoundCategory {
    pub fn is_memory_bound(&self) -> bool {
        matches!(
            self,
            BoundCategory::MemoryBoundHbm | BoundCategory::MemoryBoundL2
        )
    }

    pub fn is_compute_bound(&self) -> bool {
        matches!(
            self,
            BoundCategory::ComputeBoundTensorCore | BoundCategory::ComputeBoundVectorAlu
        )
    }
}

/// Formal computational and memory contract for an operator or kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockContract {
    pub op_name: String,
    pub bound_category: BoundCategory,
    pub dtype: MmaDtype,
    pub theoretical_flops: u64,
    pub minimum_bytes_transferred: u64,
}

impl BlockContract {
    pub fn new(
        op_name: impl Into<String>,
        bound_category: BoundCategory,
        dtype: MmaDtype,
        theoretical_flops: u64,
        minimum_bytes_transferred: u64,
    ) -> Self {
        Self {
            op_name: op_name.into(),
            bound_category,
            dtype,
            theoretical_flops,
            minimum_bytes_transferred,
        }
    }

    /// Derive an approximate block contract from a KernelSpec and matrix dimensions M, N, K.
    pub fn from_gemm_spec(spec: &KernelSpec, m: u64, n: u64, k: u64, dtype: MmaDtype) -> Self {
        let flops = 2 * m * n * k;
        let elem_bytes = match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1,
            MmaDtype::Fp4 => 1,
        };
        let bytes = (m * k + k * n + m * n) * elem_bytes;
        let bound = match spec.phase {
            Phase::Decode => BoundCategory::MemoryBoundHbm,
            Phase::Prefill => {
                if m >= 256 {
                    BoundCategory::ComputeBoundTensorCore
                } else {
                    BoundCategory::MemoryBoundHbm
                }
            }
            Phase::Mixed => BoundCategory::MemoryBoundHbm,
        };
        Self::new(spec.id.c_name(), bound, dtype, flops, bytes)
    }

    /// Arithmetic intensity in FLOPs per byte transferred.
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.minimum_bytes_transferred == 0 {
            return f64::INFINITY;
        }
        self.theoretical_flops as f64 / self.minimum_bytes_transferred as f64
    }

    /// Theoretical minimum latency in seconds on a GPU spec.
    pub fn theoretical_min_latency_secs(&self, gpu: &GpuSpec) -> f64 {
        let bw_gbps = gpu.mem.bandwidth_for_bound().0;
        let t_mem = if bw_gbps > 0.0 {
            (self.minimum_bytes_transferred as f64) / (bw_gbps * 1e9)
        } else {
            0.0
        };

        // Estimate matrix compute capability based on available spec data
        let tflops = match self.dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 989.0, // baseline H100 SXM5 FP16/BF16 dense
            MmaDtype::Fp8 | MmaDtype::Int8 => 1979.0,
            MmaDtype::Fp4 => 3958.0,
        };

        let t_comp = (self.theoretical_flops as f64) / (tflops * 1e12);

        match self.bound_category {
            BoundCategory::MemoryBoundHbm => t_mem,
            BoundCategory::MemoryBoundL2 => t_mem * 0.25,
            BoundCategory::ComputeBoundTensorCore | BoundCategory::ComputeBoundVectorAlu => t_comp,
        }
    }

    /// Calculate empirical roofline efficiency given measured execution latency in seconds.
    pub fn empirical_efficiency(&self, gpu: &GpuSpec, measured_latency_secs: f64) -> f64 {
        if measured_latency_secs <= 0.0 {
            return 0.0;
        }
        let t_min = self.theoretical_min_latency_secs(gpu);
        (t_min / measured_latency_secs).clamp(0.0, 1.0)
    }

    /// Verify if a measured kernel satisfies the target certification threshold:
    /// >= 80% for memory-bound ops (e.g. Decode GEMV, linear RNN scan)
    /// >= 75% for compute-bound ops (e.g. Prefill GEMM, MoE dense)
    pub fn satisfies_certification(&self, gpu: &GpuSpec, measured_latency_secs: f64) -> bool {
        let efficiency = self.empirical_efficiency(gpu, measured_latency_secs);
        match self.bound_category {
            BoundCategory::MemoryBoundHbm | BoundCategory::MemoryBoundL2 => efficiency >= 0.80,
            BoundCategory::ComputeBoundTensorCore | BoundCategory::ComputeBoundVectorAlu => {
                efficiency >= 0.75
            }
        }
    }
}
