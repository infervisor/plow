//! DeepSeek-V2 / V3 (MLA + DeepSeekMoE) configuration.

use serde::Deserialize;

use crate::DType;

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekQuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub weight_block_size: [i64; 2],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DeepSeekConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    // --- MLA (multi-head latent attention) ---
    /// Rank of the query down-projection (0 ⇒ no query LoRA, V2-lite).
    pub q_lora_rank: u32,
    /// Rank of the shared key/value down-projection.
    pub kv_lora_rank: u32,
    /// RoPE-carrying portion of the per-head query/key dim.
    pub qk_rope_head_dim: u32,
    /// Non-RoPE (content) portion of the per-head query/key dim.
    pub qk_nope_head_dim: u32,
    /// Per-head value dim.
    pub v_head_dim: u32,

    // --- DeepSeekMoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    /// First `first_k_dense_replace` layers use a dense MLP, the rest use MoE.
    pub first_k_dense_replace: u32,
    pub scoring_func: String,
    pub topk_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub num_nextn_predict_layers: u32,
    pub quantization_config: Option<DeepSeekQuantizationConfig>,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        // DeepSeek-V3-ish defaults (scaled-down layer count for testing).
        DeepSeekConfig {
            vocab_size: 129_280,
            hidden_size: 7168,
            intermediate_size: 18_432,
            num_hidden_layers: 61,
            num_attention_heads: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_intermediate_size: 2048,
            first_k_dense_replace: 3,
            scoring_func: "sigmoid".into(),
            topk_method: "noaux_tc".into(),
            n_group: 8,
            topk_group: 4,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            num_nextn_predict_layers: 0,
            quantization_config: None,
            torch_dtype: None,
        }
    }
}

impl DeepSeekConfig {
    /// Full per-head query/key dim = nope + rope portions.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_routed_experts > 0 {
            if self.scoring_func != "sigmoid" || self.topk_method != "noaux_tc" {
                return Err(format!(
                    "DeepSeek MoE requires sigmoid/noaux_tc routing, got {}/{}",
                    self.scoring_func, self.topk_method
                ));
            }
            if self.n_group == 0
                || self.topk_group == 0
                || self.topk_group > self.n_group
                || self.n_group > 255
                || self.topk_group > 255
                || self.n_routed_experts % self.n_group != 0
            {
                return Err(
                    "DeepSeek noaux_tc requires divisible experts and valid 8-bit group counts"
                        .into(),
                );
            }
        }
        if let Some(q) = &self.quantization_config {
            if q.quant_method != "fp8"
                || q.fmt != "e4m3"
                || q.activation_scheme != "dynamic"
                || q.weight_block_size != [128, 128]
            {
                return Err(format!(
                    "unsupported DeepSeek quantization: expected dynamic fp8/e4m3 [128,128], got {}/{}/{} {:?}",
                    q.quant_method, q.fmt, q.activation_scheme, q.weight_block_size
                ));
            }
        }
        Ok(())
    }

    pub fn projection_weight_dtype(&self) -> DType {
        if self.quantization_config.is_some() {
            DType::F8E4M3
        } else {
            DType::BF16
        }
    }

    pub fn fp8_scale_shape(&self, out_features: i64, in_features: i64) -> Option<[i64; 2]> {
        let block = self.quantization_config.as_ref()?.weight_block_size;
        Some([
            (out_features + block[0] - 1) / block[0],
            (in_features + block[1] - 1) / block[1],
        ])
    }
}
