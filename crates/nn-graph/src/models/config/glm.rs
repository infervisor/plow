//! GLM-MoE-DSA configuration (HuggingFace `config.json`).

use serde::Deserialize;

use super::ConfigError;

#[derive(Debug, Clone, Deserialize)]
pub struct GlmRopeParams {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlmQuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub weight_block_size: [i64; 2],
}

/// GLM-5.2/5.3 MoE-DSA text decoder config.
///
/// Fields that affect graph structure are required. In particular, missing
/// indexer or MTP metadata must not silently fall back to a dense MLA graph.
#[derive(Debug, Clone, Deserialize)]
pub struct GlmConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub hidden_act: String,

    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_head_dim: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,

    pub rope_interleave: bool,
    pub rope_parameters: GlmRopeParams,

    pub first_k_dense_replace: u32,
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    pub mlp_layer_types: Vec<String>,
    pub scoring_func: String,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub n_group: u32,
    pub topk_group: u32,
    pub topk_method: String,
    pub moe_router_dtype: String,

    /// `"full"` computes fresh top-k indices; `"shared"` consumes the most
    /// recent full layer's indices. Both still execute sparse attention.
    pub indexer_types: Vec<String>,
    pub index_head_dim: u32,
    pub index_n_heads: u32,
    pub index_topk: u32,
    pub index_topk_freq: u32,
    pub indexer_rope_interleave: bool,
    pub index_skip_topk_offset: u32,

    pub num_nextn_predict_layers: u32,
    pub index_share_for_mtp_iteration: bool,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    pub quantization_config: GlmQuantizationConfig,
}

impl GlmConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let unsupported = |message: String| Err(ConfigError::Unsupported(message));
        if self.num_hidden_layers == 0
            || self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.vocab_size <= 0
        {
            return unsupported("glm_moe_dsa dimensions must be positive".into());
        }
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim {
            return unsupported(format!(
                "glm_moe_dsa qk_head_dim={} but qk_nope_head_dim + qk_rope_head_dim = {}",
                self.qk_head_dim,
                self.qk_nope_head_dim + self.qk_rope_head_dim
            ));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return unsupported(
                "glm_moe_dsa num_attention_heads must be divisible by num_key_value_heads".into(),
            );
        }
        if self.mlp_layer_types.len() != self.num_hidden_layers as usize {
            return unsupported(format!(
                "glm_moe_dsa mlp_layer_types has {} entries for {} base layers",
                self.mlp_layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if let Some(kind) = self
            .mlp_layer_types
            .iter()
            .find(|kind| !matches!(kind.as_str(), "dense" | "sparse"))
        {
            return unsupported(format!("glm_moe_dsa unsupported MLP layer type {kind:?}"));
        }
        if self.indexer_types.len() != self.num_hidden_layers as usize {
            return unsupported(format!(
                "glm_moe_dsa indexer_types has {} entries for {} base layers",
                self.indexer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.indexer_types.first().map(String::as_str) != Some("full") {
            return unsupported(
                "glm_moe_dsa indexer_types must start with `full`; a `shared` layer has no prior top-k indices"
                    .into(),
            );
        }
        if let Some(kind) = self
            .indexer_types
            .iter()
            .find(|kind| !matches!(kind.as_str(), "full" | "shared"))
        {
            return unsupported(format!("glm_moe_dsa unsupported indexer type {kind:?}"));
        }
        if self.index_topk == 0
            || self.index_head_dim == 0
            || self.index_n_heads == 0
            || self.index_topk_freq == 0
        {
            return unsupported("glm_moe_dsa indexer dimensions must be positive".into());
        }
        if !self.rope_interleave || !self.indexer_rope_interleave {
            return unsupported("glm_moe_dsa requires interleaved main and indexer RoPE".into());
        }
        if self.rope_parameters.rope_type != "default" {
            return unsupported(format!(
                "glm_moe_dsa unsupported RoPE type {:?}",
                self.rope_parameters.rope_type
            ));
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sigmoid"
            || self.topk_method != "noaux_tc"
            || self.moe_router_dtype != "float32"
        {
            return unsupported(format!(
                "glm_moe_dsa requires silu experts and float32 sigmoid/noaux_tc routing; got hidden_act={:?}, scoring_func={:?}, topk_method={:?}, moe_router_dtype={:?}",
                self.hidden_act, self.scoring_func, self.topk_method, self.moe_router_dtype
            ));
        }
        if self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.n_group == 0
            || self.n_routed_experts % self.n_group != 0
            || self.topk_group == 0
            || self.topk_group > self.n_group
        {
            return unsupported("glm_moe_dsa invalid expert routing geometry".into());
        }
        if !self.norm_topk_prob || self.routed_scaling_factor <= 0.0 {
            return unsupported(
                "glm_moe_dsa requires normalized, positively scaled top-k expert weights".into(),
            );
        }
        if self.num_nextn_predict_layers > 1 {
            return unsupported(
                "glm_moe_dsa compiler currently supports the released single MTP layer only".into(),
            );
        }
        if self.num_nextn_predict_layers == 1 && !self.index_share_for_mtp_iteration {
            return unsupported(
                "glm_moe_dsa MTP requires index_share_for_mtp_iteration=true".into(),
            );
        }
        let q = &self.quantization_config;
        if q.quant_method != "fp8"
            || q.fmt != "e4m3"
            || q.activation_scheme != "dynamic"
            || q.weight_block_size != [128, 128]
        {
            return unsupported(format!(
                "glm_moe_dsa requires dynamic e4m3 block-FP8 [128,128]; got method={:?}, fmt={:?}, activation={:?}, block={:?}",
                q.quant_method, q.fmt, q.activation_scheme, q.weight_block_size
            ));
        }
        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return unsupported("glm_moe_dsa requires dtype=bfloat16".into());
        }
        if self.attention_bias {
            return unsupported("glm_moe_dsa attention_bias=true is not supported".into());
        }
        Ok(())
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    pub fn layer_is_dense(&self, layer: u32) -> bool {
        self.mlp_layer_types[layer as usize] == "dense"
    }

    pub fn layer_computes_index(&self, layer: u32) -> bool {
        self.indexer_types[layer as usize] == "full"
    }

    pub fn fp8_scale_shape(&self, out_features: i64, in_features: i64) -> [i64; 2] {
        let [out_block, in_block] = self.quantization_config.weight_block_size;
        [
            (out_features + out_block - 1) / out_block,
            (in_features + in_block - 1) / in_block,
        ]
    }
}
