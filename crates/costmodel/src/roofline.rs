//! Roofline model for prefill and decode across hardware targets.
//!
//! Provides theoretical bounds (memory bandwidth ceiling vs matrix compute ceiling),
//! arithmetic intensity, ridge points, and attained efficiency calculations.
//!
//! # References
//! - Design §6.3: Shared Hardware Cost Model
//! - `docs/bringup/07-perf-campaign.md`: Roofline sanity and efficiency metrics.

use hwspec::{GpuSpec, MmaDtype};

/// The primary performance bound governing execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundKind {
    /// Performance is bounded by memory subsystem throughput (HBM/GDDR).
    Memory,
    /// Performance is bounded by matrix compute units (Tensor Cores / MFMA).
    Compute,
}

impl BoundKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BoundKind::Memory => "Memory",
            BoundKind::Compute => "Compute",
        }
    }
}

/// Specifications of a transformer model architecture for roofline analysis.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: String,
    pub active_params: u64,
    pub layers: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub weight_dtype: MmaDtype,
    pub kv_dtype: MmaDtype,
}

impl ModelSpec {
    pub fn new(
        name: impl Into<String>,
        active_params: u64,
        layers: u32,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        weight_dtype: MmaDtype,
        kv_dtype: MmaDtype,
    ) -> Self {
        Self {
            name: name.into(),
            active_params,
            layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            weight_dtype,
            kv_dtype,
        }
    }

    /// Preconfigured Gemma-4-12B dense model.
    pub fn gemma4_12b(weight_dtype: MmaDtype) -> Self {
        Self::new(
            "Gemma-4-12B",
            12_000_000_000,
            40,
            3840,
            30,
            10,
            256,
            weight_dtype,
            weight_dtype,
        )
    }

    /// Preconfigured Gemma-4-26B model.
    pub fn gemma4_26b(weight_dtype: MmaDtype) -> Self {
        Self::new(
            "Gemma-4-26B",
            26_000_000_000,
            46,
            5120,
            40,
            16,
            256,
            weight_dtype,
            weight_dtype,
        )
    }

    /// Preconfigured Llama-3-70B model.
    pub fn llama3_70b(weight_dtype: MmaDtype) -> Self {
        Self::new(
            "Llama-3-70B",
            70_000_000_000,
            80,
            8192,
            64,
            8,
            128,
            weight_dtype,
            weight_dtype,
        )
    }

    /// Bytes per element for a given dtype.
    pub fn dtype_bytes(d: MmaDtype) -> f64 {
        match d {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1.0,
            MmaDtype::Fp4 => 0.5,
        }
    }

    /// Total weight memory bytes for the active parameters.
    pub fn weight_bytes(&self) -> u64 {
        (self.active_params as f64 * Self::dtype_bytes(self.weight_dtype)).round() as u64
    }

    /// KV cache bytes per token position across all layers.
    pub fn kv_bytes_per_token(&self) -> u64 {
        let elem = Self::dtype_bytes(self.kv_dtype);
        // 2 for K and V
        (2.0 * self.layers as f64 * self.num_kv_heads as f64 * self.head_dim as f64 * elem).round()
            as u64
    }
}

/// Roofline analysis output for a decode step (TPOT).
#[derive(Clone, Debug)]
pub struct DecodeRoofline {
    pub batch_size: u32,
    pub ctx_len: u32,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub total_bytes: u64,
    pub flops: u64,
    pub arithmetic_intensity: f64,
    pub memory_roof_ms: f64,
    pub compute_roof_ms: f64,
    pub roofline_ms: f64,
    pub bound: BoundKind,
}

impl DecodeRoofline {
    /// Calculate achieved memory bandwidth and percent of memory roofline.
    /// Returns `(achieved_gbps, pct_roofline)`.
    pub fn attained_bandwidth(&self, measured_tpot_ms: f64, gpu_bw_gbps: f64) -> (f64, f64) {
        if measured_tpot_ms <= 0.0 {
            return (0.0, 0.0);
        }
        let secs = measured_tpot_ms * 1e-3;
        let achieved_gbps = (self.total_bytes as f64 / secs) / 1e9;
        let pct = if gpu_bw_gbps > 0.0 {
            (achieved_gbps / gpu_bw_gbps) * 100.0
        } else {
            0.0
        };
        (achieved_gbps, pct)
    }

    /// Ratio of measured time to theoretical roofline time.
    pub fn headroom(&self, measured_tpot_ms: f64) -> f64 {
        if self.roofline_ms > 0.0 {
            measured_tpot_ms / self.roofline_ms
        } else {
            0.0
        }
    }
}

/// Roofline analysis output for prefill (TTFT).
#[derive(Clone, Debug)]
pub struct PrefillRoofline {
    pub prompt_tokens: u32,
    pub gemm_flops: u64,
    pub attn_flops: u64,
    pub total_flops: u64,
    pub weight_bytes: u64,
    pub act_bytes: u64,
    pub total_bytes: u64,
    pub arithmetic_intensity: f64,
    pub ridge_point: f64,
    pub memory_roof_ms: f64,
    pub compute_roof_ms: f64,
    pub roofline_ms: f64,
    pub bound: BoundKind,
}

impl PrefillRoofline {
    /// Calculate achieved compute throughput (TFLOP/s) and percent of compute roofline.
    /// Returns `(achieved_tflops, pct_roofline)`.
    pub fn attained_compute(&self, measured_ttft_ms: f64, gpu_tflops: f64) -> (f64, f64) {
        if measured_ttft_ms <= 0.0 {
            return (0.0, 0.0);
        }
        let secs = measured_ttft_ms * 1e-3;
        let achieved_tflops = (self.total_flops as f64 / secs) / 1e12;
        let pct = if gpu_tflops > 0.0 {
            (achieved_tflops / gpu_tflops) * 100.0
        } else {
            0.0
        };
        (achieved_tflops, pct)
    }

    /// Ratio of measured time to theoretical roofline time.
    pub fn headroom(&self, measured_ttft_ms: f64) -> f64 {
        if self.roofline_ms > 0.0 {
            measured_ttft_ms / self.roofline_ms
        } else {
            0.0
        }
    }
}

/// Evaluates roofline performance for a specific GPU and model.
pub struct RooflineModel<'a> {
    pub gpu: &'a GpuSpec,
    pub model: ModelSpec,
}

impl<'a> RooflineModel<'a> {
    pub fn new(gpu: &'a GpuSpec, model: ModelSpec) -> Self {
        Self { gpu, model }
    }

    /// Peak theoretical matrix compute in TFLOP/s (dense).
    pub fn peak_compute_tflops(&self, dtype: MmaDtype) -> f64 {
        peak_compute_tflops(self.gpu, dtype)
    }

    /// Sustained / bound memory bandwidth in GB/s.
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        self.gpu.mem.bandwidth_for_bound().0
    }

    /// Arithmetic intensity ridge point (FLOP/byte) where compute bound meets memory bound.
    pub fn ridge_point(&self, dtype: MmaDtype) -> f64 {
        let compute = self.peak_compute_tflops(dtype) * 1e12;
        let bw = self.memory_bandwidth_gbps() * 1e9;
        if bw > 0.0 {
            compute / bw
        } else {
            0.0
        }
    }

    /// Compute roofline metrics for a decode step.
    pub fn decode(&self, batch_size: u32, ctx_len: u32) -> DecodeRoofline {
        let weight_bytes = self.model.weight_bytes();
        let kv_bytes = self.model.kv_bytes_per_token() * batch_size as u64 * ctx_len as u64;
        let total_bytes = weight_bytes + kv_bytes;

        // 2 FLOPs per param per token
        let flops = 2 * self.model.active_params * batch_size as u64;
        let arithmetic_intensity = if total_bytes > 0 {
            flops as f64 / total_bytes as f64
        } else {
            0.0
        };

        let bw_bytes_s = self.memory_bandwidth_gbps() * 1e9;
        let memory_roof_ms = if bw_bytes_s > 0.0 {
            (total_bytes as f64 / bw_bytes_s) * 1e3
        } else {
            0.0
        };

        let compute_flops_s = self.peak_compute_tflops(self.model.weight_dtype) * 1e12;
        let compute_roof_ms = if compute_flops_s > 0.0 {
            (flops as f64 / compute_flops_s) * 1e3
        } else {
            0.0
        };

        let roofline_ms = memory_roof_ms.max(compute_roof_ms);
        let bound = if memory_roof_ms >= compute_roof_ms {
            BoundKind::Memory
        } else {
            BoundKind::Compute
        };

        DecodeRoofline {
            batch_size,
            ctx_len,
            weight_bytes,
            kv_bytes,
            total_bytes,
            flops,
            arithmetic_intensity,
            memory_roof_ms,
            compute_roof_ms,
            roofline_ms,
            bound,
        }
    }

    /// Compute roofline metrics for prefill.
    pub fn prefill(&self, prompt_tokens: u32) -> PrefillRoofline {
        let gemm_flops = 2 * self.model.active_params * prompt_tokens as u64;
        // Causal attention FLOPs: 4 * layers * heads * head_dim * tokens^2 / 2 = 2 * layers * heads * head_dim * tokens^2
        let attn_flops = 2
            * self.model.layers as u64
            * self.model.num_heads as u64
            * self.model.head_dim as u64
            * (prompt_tokens as u64 * prompt_tokens as u64);
        let total_flops = gemm_flops + attn_flops;

        let weight_bytes = self.model.weight_bytes();
        // Activation movement: ~2 read/write passes through hidden activations
        let act_bytes = (2.0
            * self.model.layers as f64
            * self.model.hidden_size as f64
            * prompt_tokens as f64
            * ModelSpec::dtype_bytes(self.model.weight_dtype))
        .round() as u64;
        let total_bytes = weight_bytes + act_bytes;

        let arithmetic_intensity = if total_bytes > 0 {
            total_flops as f64 / total_bytes as f64
        } else {
            0.0
        };
        let ridge_point = self.ridge_point(self.model.weight_dtype);

        let bw_bytes_s = self.memory_bandwidth_gbps() * 1e9;
        let memory_roof_ms = if bw_bytes_s > 0.0 {
            (total_bytes as f64 / bw_bytes_s) * 1e3
        } else {
            0.0
        };

        let compute_flops_s = self.peak_compute_tflops(self.model.weight_dtype) * 1e12;
        let compute_roof_ms = if compute_flops_s > 0.0 {
            (total_flops as f64 / compute_flops_s) * 1e3
        } else {
            0.0
        };

        let roofline_ms = memory_roof_ms.max(compute_roof_ms);
        let bound = if arithmetic_intensity < ridge_point {
            BoundKind::Memory
        } else {
            BoundKind::Compute
        };

        PrefillRoofline {
            prompt_tokens,
            gemm_flops,
            attn_flops,
            total_flops,
            weight_bytes,
            act_bytes,
            total_bytes,
            arithmetic_intensity,
            ridge_point,
            memory_roof_ms,
            compute_roof_ms,
            roofline_ms,
            bound,
        }
    }
}

/// Compute peak theoretical dense TFLOP/s for a known GPU and precision.
pub fn peak_compute_tflops(gpu: &GpuSpec, dtype: MmaDtype) -> f64 {
    let name = gpu.name.to_uppercase();
    // Calibrated datasheet peak TFLOP/s (dense tensor core / matrix engine):
    if name.contains("H100") || name.contains("H200") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 989.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1979.0,
            MmaDtype::Fp4 => 1979.0,
        }
    } else if name.contains("MI300") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 1307.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 2614.0,
            MmaDtype::Fp4 => 2614.0,
        }
    } else if name.contains("MI350") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2300.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 4600.0,
            MmaDtype::Fp4 => 9200.0,
        }
    } else if name.contains("5090") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 835.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1670.0,
            MmaDtype::Fp4 => 3340.0,
        }
    } else if name.contains("4090") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 330.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 660.0,
            MmaDtype::Fp4 => 660.0,
        }
    } else {
        // Analytical fallback from hardware spec
        let cores = gpu.sm_count as f64 * gpu.sm.tensor_cores as f64;
        let macs_core = gpu.sm.mma.of(dtype).unwrap_or(256) as f64;
        let clock_hz = gpu.clock_boost.0 as f64;
        // 2 FLOPs per MAC
        (cores * macs_core * 2.0 * clock_hz) / 1e12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> &'static GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    fn mi300x() -> &'static GpuSpec {
        hwspec::registry::lookup("MI300X").unwrap()
    }

    #[test]
    fn h100_roofline_sanity() {
        let gpu = h100();
        let model = ModelSpec::gemma4_12b(MmaDtype::Bf16);
        let rf = RooflineModel::new(gpu, model);

        assert_eq!(rf.memory_bandwidth_gbps(), 3352.0);
        assert_eq!(rf.peak_compute_tflops(MmaDtype::Bf16), 989.0);
        assert!((rf.ridge_point(MmaDtype::Bf16) - (989.0e12 / 3352.0e9)).abs() < 1.0);

        // Single-token decode (B=1, ctx=1024)
        let dec = rf.decode(1, 1024);
        assert_eq!(dec.bound, BoundKind::Memory);
        // Weight is 24 GB, 24 GB / 3352 GB/s ≈ 7.16 ms
        assert!((dec.memory_roof_ms - 7.16).abs() < 0.2);
        assert!(dec.roofline_ms > 7.0 && dec.roofline_ms < 7.5);

        // Prefill at 4096 tokens
        let pf = rf.prefill(4096);
        assert_eq!(pf.bound, BoundKind::Compute);
        assert!(pf.arithmetic_intensity > rf.ridge_point(MmaDtype::Bf16));
        assert!(pf.roofline_ms > 50.0);
    }

    #[test]
    fn mi300x_roofline_uses_measured_bandwidth() {
        let gpu = mi300x();
        let model = ModelSpec::gemma4_12b(MmaDtype::Bf16);
        let rf = RooflineModel::new(gpu, model);

        // MI300X measured read bandwidth is ~4091.9 GB/s, not datasheet 5325.0
        assert_eq!(rf.memory_bandwidth_gbps(), 4091.9);
        assert_eq!(rf.peak_compute_tflops(MmaDtype::Bf16), 1307.0);

        let dec = rf.decode(1, 1024);
        assert_eq!(dec.bound, BoundKind::Memory);
        // 24 GB / 4091.9 GB/s ≈ 5.86 ms
        assert!((dec.roofline_ms - 5.86).abs() < 0.2);
    }

    #[test]
    fn attained_metrics_calculation() {
        let gpu = h100();
        let model = ModelSpec::gemma4_12b(MmaDtype::Bf16);
        let rf = RooflineModel::new(gpu, model);

        let dec = rf.decode(1, 1024);
        // Suppose measured TPOT is 10.0 ms
        let (achieved_bw, pct_bw) = dec.attained_bandwidth(10.0, rf.memory_bandwidth_gbps());
        assert!(achieved_bw > 2300.0 && achieved_bw < 2500.0);
        assert!(pct_bw > 70.0 && pct_bw < 75.0);
        assert!((dec.headroom(10.0) - (10.0 / dec.roofline_ms)).abs() < 1e-4);

        let pf = rf.prefill(4096);
        // Suppose measured TTFT is 150.0 ms
        let (achieved_tf, pct_tf) = pf.attained_compute(150.0, rf.peak_compute_tflops(MmaDtype::Bf16));
        assert!(achieved_tf > 600.0 && achieved_tf < 800.0);
        assert!(pct_tf > 60.0 && pct_tf < 80.0);
    }
}
