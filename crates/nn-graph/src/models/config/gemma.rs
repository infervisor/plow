//! Gemma (1/2/3/4) text decoder configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GemmaConfig {
    pub model_type: String,
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Sliding window size for local-attention layers (Gemma2/3).
    pub sliding_window: u32,
    /// Every Nth layer is global attention; the rest are sliding-window local.
    /// `1` (or absent) means all-global (Gemma1).
    pub sliding_window_pattern: u32,
    /// Gemma2/3 normalize queries by sqrt of this; 0 ⇒ use head_dim.
    pub query_pre_attn_scalar: f32,
    /// Gemma3 adds RMSNorm on q and k before attention.
    pub use_qk_norm: bool,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    pub tie_word_embeddings: bool,
    pub final_logit_softcapping: Option<f32>,
    #[serde(default, rename = "_plow_weight_prefix")]
    pub weight_prefix: Option<String>,

    // --- Gemma 4 (gemma4_unified_text) additions ---
    /// Per-layer attention type, `"full_attention"` or `"sliding_attention"`.
    /// When non-empty this drives the local/global pattern (supersedes
    /// `sliding_window_pattern`).
    pub layer_types: Vec<String>,
    /// Head dim used by full-attention (global) layers; 0 ⇒ same as `head_dim`.
    pub global_head_dim: u32,
    /// KV heads used by full-attention (global) layers; 0 ⇒ same as
    /// `num_key_value_heads`.
    pub num_global_key_value_heads: u32,
    /// Gemma4 full-attention layers derive V from K; sliding layers keep V.
    pub attention_k_eq_v: bool,
    /// Per-layer-type RoPE (theta + partial rotary factor).
    pub rope_parameters: Option<RopeParameters>,

    // --- Gemma 4 MoE (gemma-4-26B-A4B) additions ---
    /// Number of routed experts per MoE layer. 0 ⇒ dense (no MoE).
    pub num_local_experts: u32,
    /// Number of experts activated per token (top-k routing).
    pub num_experts_per_tok: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: Option<RopeSpec>,
    pub sliding_attention: Option<RopeSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSpec {
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    /// Fraction of head_dim that receives RoPE (1.0 = full rotary).
    #[serde(default = "default_rotary_factor")]
    pub partial_rotary_factor: f32,
}

fn default_theta() -> f32 {
    10_000.0
}
fn default_rotary_factor() -> f32 {
    1.0
}

impl Default for GemmaConfig {
    fn default() -> Self {
        // Gemma3-4B-ish defaults.
        GemmaConfig {
            model_type: "gemma3".to_string(),
            vocab_size: 262_208,
            hidden_size: 2560,
            intermediate_size: 10_240,
            num_hidden_layers: 34,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
            use_qk_norm: true,
            torch_dtype: None,
            tie_word_embeddings: false,
            final_logit_softcapping: None,
            weight_prefix: None,
            layer_types: Vec::new(),
            global_head_dim: 0,
            num_global_key_value_heads: 0,
            attention_k_eq_v: false,
            rope_parameters: None,
            num_local_experts: 0,
            num_experts_per_tok: 0,
        }
    }
}

impl GemmaConfig {
    pub fn is_gemma4(&self) -> bool {
        matches!(
            self.model_type.as_str(),
            "gemma4" | "gemma4_text" | "gemma4_unified_text"
        )
    }

    /// Whether this model uses Mixture-of-Experts.
    pub fn is_moe(&self) -> bool {
        self.num_local_experts > 0
    }

    /// Whether layer `i` uses MoE FFN. In Gemma 4 MoE variants, `layer_types`
    /// contains entries like `"moe_sliding_attention"` or `"moe_full_attention"`
    /// for MoE layers. If `layer_types` doesn't encode MoE info, all layers use
    /// MoE when `num_local_experts > 0`.
    pub fn layer_is_moe(&self, layer: u32) -> bool {
        if !self.is_moe() {
            return false;
        }
        if let Some(t) = self.layer_types.get(layer as usize) {
            return t.starts_with("moe_");
        }
        // If layer_types doesn't distinguish MoE layers, all layers are MoE.
        true
    }

    /// Whether layer `i` uses full (global) attention.
    pub fn layer_is_global(&self, layer: u32) -> bool {
        if let Some(t) = self.layer_types.get(layer as usize) {
            return t == "full_attention" || t == "moe_full_attention";
        }
        // Gemma1/2/3 fallback: every Nth layer global; pattern ≤ 1 ⇒ all global.
        if self.sliding_window_pattern <= 1 {
            return true;
        }
        (layer + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// Per-head dim for this layer type.
    pub fn head_dim_for(&self, global: bool) -> i64 {
        if global && self.global_head_dim > 0 {
            self.global_head_dim as i64
        } else {
            self.head_dim as i64
        }
    }

    /// KV-head count for this layer type.
    pub fn kv_heads_for(&self, global: bool) -> u32 {
        if global && self.num_global_key_value_heads > 0 {
            self.num_global_key_value_heads
        } else {
            self.num_key_value_heads
        }
    }

    /// `(rope_theta, partial_rotary_factor)` for this layer type.
    pub fn rope_for(&self, global: bool) -> (f32, f32) {
        if let Some(rp) = &self.rope_parameters {
            let spec = if global {
                rp.full_attention.as_ref()
            } else {
                rp.sliding_attention.as_ref()
            };
            if let Some(s) = spec {
                return (s.rope_theta, s.partial_rotary_factor);
            }
        }
        (self.rope_theta, 1.0)
    }

    pub fn rope_frequency_dim(&self, global: bool, rotary_dim: u32, head_dim: u32) -> u32 {
        let spec = self.rope_parameters.as_ref().and_then(|parameters| {
            if global {
                parameters.full_attention.as_ref()
            } else {
                parameters.sliding_attention.as_ref()
            }
        });
        if spec.is_some_and(|spec| spec.rope_type == "proportional") {
            head_dim
        } else {
            rotary_dim
        }
    }
}
