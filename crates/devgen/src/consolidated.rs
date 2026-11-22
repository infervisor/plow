//! Consolidated primitive operations and transforms across model emitters.
//!
//! Replaces duplicated emitter logic for RoPE tables, RMSNorm parameter packing,
//! attention scale factor absorption, and weight layout verification.

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::Builder;
use packet::rope::{GenTensor, RopeScale};
use packet::typed_abi::{GemmFp8MxArgs, RmsNormArgs, RopeArgs, TypedOp};

/// Consolidated RoPE parameter configuration and tensor generator.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsolidatedRope {
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f64,
    pub scale: RopeScale,
    pub max_ctx: u32,
}

impl ConsolidatedRope {
    pub fn new(
        head_dim: u32,
        rotary_fraction: f64,
        theta: f64,
        scale: RopeScale,
        max_ctx: u32,
    ) -> Self {
        let rotary_dim = ((head_dim as f64) * rotary_fraction).round() as u32;
        assert!(
            rotary_dim <= head_dim && rotary_dim % 2 == 0,
            "rotary_dim ({rotary_dim}) must be even and <= head_dim ({head_dim})"
        );
        Self {
            head_dim,
            rotary_dim,
            theta,
            scale,
            max_ctx,
        }
    }

    /// Generate the canonical recipe for the cos/sin table pair.
    pub fn gen_tables(&self) -> [GenTensor; 2] {
        GenTensor::rope_pair(
            self.max_ctx,
            self.head_dim,
            self.theta,
            self.rotary_dim as f64 / self.head_dim as f64,
            self.scale,
        )
    }

    /// Emit a typed RoPE operation into the builder.
    pub fn emit(
        &self,
        b: &mut Builder,
        cus: Vec<u32>,
        out: u32,
        x: u32,
        cos: u32,
        sin: u32,
        pos: u32,
        heads: u32,
        seq_len: u32,
        deps: &[u32],
    ) -> u32 {
        let args = RopeArgs::simple_rope(out, x, cos, sin, pos, seq_len, heads, self.head_dim);
        b.emit(DevOp::HeadNormRope, cus, deps, |d| {
            args.apply(d);
        })
    }
}

/// Consolidated RMSNorm emit configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsolidatedRmsNorm {
    pub eps: f32,
    pub residual_f32: bool,
    pub q128_quant: bool,
}

impl ConsolidatedRmsNorm {
    pub fn new(eps: f32) -> Self {
        Self {
            eps,
            residual_f32: false,
            q128_quant: false,
        }
    }

    pub fn with_residual_f32(mut self, enabled: bool) -> Self {
        self.residual_f32 = enabled;
        self
    }

    pub fn with_q128_quant(mut self, enabled: bool) -> Self {
        self.q128_quant = enabled;
        self
    }

    /// Emit a validated RMSNorm operation.
    pub fn emit(
        &self,
        b: &mut Builder,
        cus: Vec<u32>,
        out: u32,
        x: u32,
        weight: u32,
        rows: u32,
        hidden: u32,
        deps: &[u32],
    ) -> u32 {
        assert!(
            hidden > 0 && hidden % 8 == 0,
            "hidden dimension must be a multiple of 8"
        );
        assert!(out != TENSOR_NONE && x != TENSOR_NONE && weight != TENSOR_NONE);

        let args = RmsNormArgs {
            out,
            x,
            gamma: weight,
            rows,
            feat: hidden,
            eps: self.eps,
            ..Default::default()
        };

        b.emit(DevOp::RmsNorm, cus, deps, |d| {
            args.apply(d);
            if self.residual_f32 {
                d.i[6] |= packet::dev::RESIDUAL_F32_FLAG;
            }
            if self.q128_quant {
                d.i[7] = 1; // Mark block-128 quantization enabled
            }
        })
    }
}

/// Consolidated Attention scale absorption policy.
pub struct ConsolidatedAttentionScale;

impl ConsolidatedAttentionScale {
    /// Compute the effective softmax scale factor.
    pub fn softmax_scale(head_dim: u32, rope_scale: RopeScale, query_scale: Option<f64>) -> f32 {
        let base_scale = query_scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
        let mscale = rope_scale.mscale();
        // If mscale is applied to cos/sin tables, the softmax scale factor may need adjustments
        (base_scale / (mscale * mscale)) as f32
    }

    /// Determine whether the scale should be absorbed into Query projection weights
    /// or explicitly passed to the attention kernel.
    pub fn can_absorb_into_q(is_fp8: bool, has_q_norm: bool) -> bool {
        // When Q-Norm is present or weights are FP8-quantized, absorbing scale into Q weights
        // risks numerical overflow or invalidates RMSNorm scale invariant.
        !is_fp8 && !has_q_norm
    }
}

/// Consolidated Weight layout verification and shape calculations.
pub struct ConsolidatedWeightLayout;

impl ConsolidatedWeightLayout {
    /// Calculate exact byte size for FP8 block-scaled weights with UE8M0 32x32 scales.
    pub fn fp8_mx_bytes(n: u32, k: u32) -> (u64, u64) {
        let weight_bytes = n as u64 * k as u64;
        let scale_bytes = (n as u64).div_ceil(32) * (k as u64).div_ceil(32);
        (weight_bytes, scale_bytes)
    }

    /// Emit a validated FP8 MX block GEMM operation.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_gemm_fp8_mx(
        b: &mut Builder,
        cus: &[u32],
        out: u32,
        x: u32,
        wt: u32,
        scale: u32,
        m: u32,
        n: u32,
        k: u32,
        row0: u32,
        c_row0: u32,
        deps: &[u32],
    ) -> u32 {
        assert!(
            wt != TENSOR_NONE && scale != TENSOR_NONE,
            "GemmFp8Mx requires weight and scale"
        );
        assert!(
            k % 32 == 0,
            "K must be divisible by 32 for FP8 MX block scale"
        );

        let (want_wt, want_sc) = Self::fp8_mx_bytes(n, k);
        let actual_wt = b.tensor_bytes(wt);
        let actual_sc = b.tensor_bytes(scale);
        assert_eq!(
            actual_wt,
            want_wt,
            "weight tensor {:?} declared {actual_wt} bytes, want {want_wt}",
            b.tensor_name(wt)
        );
        assert_eq!(
            actual_sc,
            want_sc,
            "scale tensor {:?} declared {actual_sc} bytes, want {want_sc}",
            b.tensor_name(scale)
        );

        let args = GemmFp8MxArgs {
            out,
            x,
            w: wt,
            scale,
            T: m,
            N: n,
            K: k,
        };

        b.emit(DevOp::GemmFp8Mx, cus.to_vec(), deps, |d| {
            args.apply(d);
            d.i[4] = row0;
            d.i[5] = c_row0;
        })
    }
}

/// Multi-Head Latent Attention (MLA) geometry specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaGeometry {
    pub q_lora_rank: Option<u32>,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub num_heads: u32,
}

impl MlaGeometry {
    pub fn new(
        q_lora_rank: Option<u32>,
        kv_lora_rank: u32,
        qk_nope_head_dim: u32,
        qk_rope_head_dim: u32,
        v_head_dim: u32,
        num_heads: u32,
    ) -> Self {
        assert!(kv_lora_rank > 0 && kv_lora_rank % 8 == 0, "kv_lora_rank must be divisible by 8");
        assert!(qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even for RoPE");
        assert!(num_heads > 0, "num_heads must be positive");
        Self {
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            num_heads,
        }
    }

    pub fn total_q_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn total_kv_head_dim(&self) -> u32 {
        self.qk_rope_head_dim + self.v_head_dim
    }

    pub fn latent_kv_bytes(&self, batch: u32, dtype_bytes: u32) -> u64 {
        batch as u64 * self.kv_lora_rank as u64 * dtype_bytes as u64
    }

    pub fn absorbed_matrix_shape(&self) -> (u32, u32) {
        (self.qk_nope_head_dim, self.kv_lora_rank)
    }
}

/// Consolidated Multi-Head Latent Attention (MLA) adapter.
pub struct ConsolidatedMlaAdapter {
    pub geometry: MlaGeometry,
}

impl ConsolidatedMlaAdapter {
    pub fn new(geometry: MlaGeometry) -> Self {
        Self { geometry }
    }

    /// Emit decoupled RoPE operation for query or key states.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_decoupled_rope(
        &self,
        b: &mut Builder,
        cus: Vec<u32>,
        out: u32,
        x: u32,
        cos: u32,
        sin: u32,
        pos: u32,
        seq_len: u32,
        deps: &[u32],
    ) -> u32 {
        let args = RopeArgs::simple_rope(
            out,
            x,
            cos,
            sin,
            pos,
            seq_len,
            self.geometry.num_heads,
            self.geometry.qk_rope_head_dim,
        );
        b.emit(DevOp::HeadNormRope, cus, deps, |d| {
            args.apply(d);
        })
    }
}

/// Linear Recurrent / Gated Delta Networks (GDN / KDA) geometry specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdaGeometry {
    pub chunk_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub hidden: u32,
}

impl KdaGeometry {
    pub fn new(chunk_size: u32, num_heads: u32, head_dim: u32, hidden: u32) -> Self {
        assert!(chunk_size > 0 && chunk_size % 16 == 0, "chunk_size must be a multiple of 16");
        assert!(head_dim > 0 && head_dim % 8 == 0, "head_dim must be a multiple of 8");
        assert!(num_heads > 0, "num_heads must be positive");
        Self {
            chunk_size,
            num_heads,
            head_dim,
            hidden,
        }
    }
}

/// Consolidated KDA / GDN linear recurrence adapter.
pub struct ConsolidatedKdaAdapter {
    pub geometry: KdaGeometry,
}

impl ConsolidatedKdaAdapter {
    pub fn new(geometry: KdaGeometry) -> Self {
        Self { geometry }
    }

    /// Emit KDA chunk weight update recurrence step.
    pub fn emit_chunk_wu(
        &self,
        b: &mut Builder,
        cus: Vec<u32>,
        args: &packet::typed_abi::KdaChunkWuArgs,
        deps: &[u32],
    ) -> u32 {
        b.emit(DevOp::KdaChunkWu, cus, deps, |d| {
            args.apply(d);
        })
    }
}

/// Hybrid sliding-window / global-attention layer interleaving schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridPattern {
    /// Every Nth layer is sliding-window attention (e.g. 2 for alternating).
    SlidingEvery(u32),
    /// Pattern of sliding:full ratio (e.g. sliding: 5, full: 1 for Gemma 2/4).
    Ratio { sliding: u32, full: u32 },
    /// All layers use sliding-window attention.
    AllSliding,
    /// All layers use full global attention.
    AllFull,
}

/// Consolidated hybrid attention scheduler.
pub struct ConsolidatedHybridAttention;

impl ConsolidatedHybridAttention {
    /// Determine whether a layer uses sliding-window attention.
    pub fn is_sliding_layer(layer_idx: u32, pattern: HybridPattern) -> bool {
        match pattern {
            HybridPattern::SlidingEvery(period) => {
                if period == 0 {
                    false
                } else {
                    (layer_idx + 1) % period == 0
                }
            }
            HybridPattern::Ratio { sliding, full } => {
                let cycle = sliding + full;
                if cycle == 0 {
                    false
                } else {
                    (layer_idx % cycle) < sliding
                }
            }
            HybridPattern::AllSliding => true,
            HybridPattern::AllFull => false,
        }
    }

    /// Return sliding window size for the layer if sliding, or None if full attention.
    pub fn window_for_layer(
        layer_idx: u32,
        window_size: u32,
        pattern: HybridPattern,
    ) -> Option<u32> {
        if Self::is_sliding_layer(layer_idx, pattern) {
            Some(window_size)
        } else {
            None
        }
    }

    /// Configure sliding window on FlashPrefillArgs according to the layer and pattern.
    pub fn configure_prefill_args(
        args: &mut packet::typed_abi::FlashPrefillArgs,
        layer_idx: u32,
        window_size: u32,
        pattern: HybridPattern,
    ) {
        args.window = Self::window_for_layer(layer_idx, window_size, pattern).unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consolidated_rope_properties() {
        let rope = ConsolidatedRope::new(128, 0.5, 10000.0, RopeScale::None, 8192);
        assert_eq!(rope.rotary_dim, 64);
        let [cos, sin] = rope.gen_tables();
        assert_eq!(cos.hd, 128);
        assert_eq!(sin.ctx, 8192);
    }

    #[test]
    fn consolidated_rmsnorm_flags() {
        let norm = ConsolidatedRmsNorm::new(1e-5).with_residual_f32(true);
        assert!(norm.residual_f32);
        assert_eq!(norm.eps, 1e-5);
    }

    #[test]
    fn attention_scale_absorption() {
        assert!(ConsolidatedAttentionScale::can_absorb_into_q(false, false));
        assert!(!ConsolidatedAttentionScale::can_absorb_into_q(true, false));
        assert!(!ConsolidatedAttentionScale::can_absorb_into_q(false, true));
    }

    #[test]
    fn weight_layout_sizing() {
        let (wt, sc) = ConsolidatedWeightLayout::fp8_mx_bytes(4096, 4096);
        assert_eq!(wt, 16_777_216);
        assert_eq!(sc, 128 * 128);
    }

    #[test]
    fn mla_geometry_calculations() {
        let geom = MlaGeometry::new(Some(1536), 512, 128, 64, 128, 128);
        assert_eq!(geom.total_q_head_dim(), 192);
        assert_eq!(geom.total_kv_head_dim(), 192);
        assert_eq!(geom.latent_kv_bytes(4, 1), 2048);
        assert_eq!(geom.absorbed_matrix_shape(), (128, 512));
    }

    #[test]
    fn kda_geometry_validation() {
        let geom = KdaGeometry::new(64, 32, 128, 4096);
        assert_eq!(geom.chunk_size, 64);
        assert_eq!(geom.num_heads, 32);
        assert_eq!(geom.head_dim, 128);
    }

    #[test]
    fn hybrid_attention_schedule() {
        // Gemma 2 style: 1 sliding : 1 full alternating
        let alt = HybridPattern::SlidingEvery(2);
        assert!(!ConsolidatedHybridAttention::is_sliding_layer(0, alt));
        assert!(ConsolidatedHybridAttention::is_sliding_layer(1, alt));
        assert_eq!(ConsolidatedHybridAttention::window_for_layer(1, 4096, alt), Some(4096));
        assert_eq!(ConsolidatedHybridAttention::window_for_layer(0, 4096, alt), None);

        // Ratio pattern: 5 sliding : 1 full
        let ratio = HybridPattern::Ratio { sliding: 5, full: 1 };
        for i in 0..5 {
            assert!(ConsolidatedHybridAttention::is_sliding_layer(i, ratio));
        }
        assert!(!ConsolidatedHybridAttention::is_sliding_layer(5, ratio));
        assert!(ConsolidatedHybridAttention::is_sliding_layer(6, ratio));
    }
}
