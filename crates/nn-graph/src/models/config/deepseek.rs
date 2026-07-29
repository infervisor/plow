//! DeepSeek-V2 / V3 (MLA + DeepSeekMoE) configuration.

use serde::Deserialize;

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
            torch_dtype: None,
        }
    }
}

impl DeepSeekConfig {
    /// Full per-head query/key dim = nope + rope portions.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}
