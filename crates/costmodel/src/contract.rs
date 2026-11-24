//! Block contract for computational sub-DAGs and kernels.
//!
//! Defines formal arithmetic, memory transfer, and roofline bounds for an
//! isolated block or operator. Enables autonomous kernel campaign qualification
//! by computing expected lower bounds and attained empirical efficiency.

use hwspec::{GpuSpec, MmaDtype};

use crate::roofline::peak_compute_tflops;

/// The architectural bottleneck category for a block or kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundCategory {
    /// Bounded by high-bandwidth memory (HBM/GDDR) throughput.
    MemoryBoundHbm,
    /// Bounded by on-chip SRAM / L2 cache bandwidth.
    MemoryBoundL2,
    /// Bounded by dense or sparse matrix multiply units (Tensor Core / MFMA).
    ComputeBoundTensorCore,
    /// Bounded by general-purpose vector ALUs (VALU).
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

    /// The certification gate: the fraction of the floor a kernel must attain.
    pub fn certification_threshold(&self) -> f64 {
        if self.is_memory_bound() {
            CERTIFY_MEMORY_BOUND
        } else {
            CERTIFY_COMPUTE_BOUND
        }
    }
}

/// Memory-bound ops (decode GEMV, linear recurrent scan) certify at >= 80% of the floor.
pub const CERTIFY_MEMORY_BOUND: f64 = 0.80;
/// Compute-bound ops (prefill GEMM, MoE dense) certify at >= 75% of the floor.
pub const CERTIFY_COMPUTE_BOUND: f64 = 0.75;

/// Formal computational and memory contract for a single block / operator invocation.
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

    /// Arithmetic intensity in FLOPs per byte transferred.
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.minimum_bytes_transferred == 0 {
            return f64::INFINITY;
        }
        self.theoretical_flops as f64 / self.minimum_bytes_transferred as f64
    }

    /// Theoretical minimum latency in seconds on the target GPU spec.
    pub fn theoretical_min_latency_secs(&self, gpu: &GpuSpec) -> f64 {
        let bw_gbps = gpu.mem.bandwidth_for_bound().0;
        let t_mem = if bw_gbps > 0.0 {
            (self.minimum_bytes_transferred as f64) / (bw_gbps * 1e9)
        } else {
            0.0
        };

        let tflops = peak_compute_tflops(gpu, self.dtype);
        let t_comp = if tflops > 0.0 {
            (self.theoretical_flops as f64) / (tflops * 1e12)
        } else {
            0.0
        };

        match self.bound_category {
            BoundCategory::MemoryBoundHbm => t_mem,
            BoundCategory::MemoryBoundL2 => t_mem * 0.25, // L2 is typically ~4x HBM bandwidth
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
        self.empirical_efficiency(gpu, measured_latency_secs)
            >= self.bound_category.certification_threshold()
    }

    /// Standard decode GEMV block contract helper.
    pub fn decode_gemv(
        op_name: impl Into<String>,
        batch: usize,
        n: usize,
        k: usize,
        dtype: MmaDtype,
    ) -> Self {
        let flops = 2 * (batch as u64) * (n as u64) * (k as u64);
        let elem_bytes = match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1,
            MmaDtype::Fp4 => 1, // half byte packed into byte arithmetic
        };
        // Weight matrix read: N*K*elem_bytes; vector read: B*K*elem_bytes; out: B*N*elem_bytes
        let bytes = (n as u64) * (k as u64) * elem_bytes
            + (batch as u64) * (k as u64) * elem_bytes
            + (batch as u64) * (n as u64) * elem_bytes;
        Self::new(op_name, BoundCategory::MemoryBoundHbm, dtype, flops, bytes)
    }

    /// Standard prefill GEMM block contract helper.
    pub fn prefill_gemm(
        op_name: impl Into<String>,
        m: usize,
        n: usize,
        k: usize,
        dtype: MmaDtype,
    ) -> Self {
        let flops = 2 * (m as u64) * (n as u64) * (k as u64);
        let elem_bytes = match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1,
            MmaDtype::Fp4 => 1,
        };
        let bytes = ((m * k) + (k * n) + (m * n)) as u64 * elem_bytes;
        let bound = if m >= 256 {
            BoundCategory::ComputeBoundTensorCore
        } else {
            BoundCategory::MemoryBoundHbm
        };
        Self::new(op_name, bound, dtype, flops, bytes)
    }

    /// Standard FlashAttention prefill block contract helper.
    pub fn flash_attention(
        op_name: impl Into<String>,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: MmaDtype,
    ) -> Self {
        let flops = 4
            * (batch as u64)
            * (num_heads as u64)
            * (seq_len as u64)
            * (seq_len as u64)
            * (head_dim as u64);
        let elem_bytes = match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1,
            MmaDtype::Fp4 => 1,
        };
        let bytes = 4
            * (batch as u64)
            * (num_heads as u64)
            * (seq_len as u64)
            * (head_dim as u64)
            * elem_bytes;
        let bound = if seq_len >= 512 {
            BoundCategory::ComputeBoundTensorCore
        } else {
            BoundCategory::MemoryBoundHbm
        };
        Self::new(op_name, bound, dtype, flops, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> &'static GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn test_contract_decode_gemv_bounds() {
        let contract = BlockContract::decode_gemv("gemv_b1", 1, 4096, 4096, MmaDtype::Bf16);
        assert_eq!(contract.bound_category, BoundCategory::MemoryBoundHbm);
        assert!(contract.arithmetic_intensity() < 5.0);

        let gpu = h100();
        let t_min = contract.theoretical_min_latency_secs(gpu);
        assert!(t_min > 0.0);

        // At 1.1x t_min, efficiency is ~90.9% -> satisfies >= 80%
        assert!(contract.satisfies_certification(gpu, t_min * 1.1));
        // At 1.5x t_min, efficiency is ~66.7% -> fails >= 80%
        assert!(!contract.satisfies_certification(gpu, t_min * 1.5));
    }

    #[test]
    fn test_contract_prefill_gemm_bounds() {
        let contract = BlockContract::prefill_gemm("gemm_m2048", 2048, 4096, 4096, MmaDtype::Bf16);
        assert_eq!(
            contract.bound_category,
            BoundCategory::ComputeBoundTensorCore
        );
        assert!(contract.arithmetic_intensity() > 50.0);

        let gpu = h100();
        let t_min = contract.theoretical_min_latency_secs(gpu);
        assert!(t_min > 0.0);

        // At 1.25x t_min, efficiency is 80% -> satisfies >= 75%
        assert!(contract.satisfies_certification(gpu, t_min * 1.25));
        // At 1.4x t_min, efficiency is ~71.4% -> fails >= 75%
        assert!(!contract.satisfies_certification(gpu, t_min * 1.4));
    }
}
