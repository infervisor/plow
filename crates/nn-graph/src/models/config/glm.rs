//! GLM-5 / GLM-MoE-DSA configuration (HuggingFace `config.json`).

use serde::Deserialize;

/// GLM-5.2 MoE-DSA text decoder config. MLA attention (same low-rank Q/KV
/// decomposition as DeepSeek-V2/V3), SwiGLU MLP, DeepSeekMoE-style routing,
/// and a novel Dense-Sparse Attention (DSA) indexer for select layers.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GlmConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    /// Per-head "logical" dim (used for Q/K nope width in the weight naming).
    pub head_dim: u32,
    pub rms_norm_eps: f32,

    // --- MLA (multi-head latent attention) ---
    /// Rank of the query down-projection (>0 enables query LoRA path).
    pub q_lora_rank: u32,
    /// Rank of the shared key/value down-projection.
    pub kv_lora_rank: u32,
    /// Full per-head query/key dim (nope + rope).
    pub qk_head_dim: u32,
    /// Non-RoPE (content) portion of the per-head query/key dim.
    pub qk_nope_head_dim: u32,
    /// RoPE-carrying portion of the per-head query/key dim.
    pub qk_rope_head_dim: u32,
    /// Per-head value dim.
    pub v_head_dim: u32,

    // --- RoPE ---
    /// Use interleaved RoPE pairing: (x[0],x[1]), (x[2],x[3])... instead of
    /// the split-halves Llama convention.
    #[serde(default)]
    pub rope_interleave: bool,
    /// Nested RoPE parameters.
    pub rope_parameters: Option<GlmRopeParams>,

    // --- MoE ---
    /// First N layers use a dense MLP; the rest use MoE.
    pub first_k_dense_replace: u32,
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    /// Per-layer MLP type: "dense" or "sparse" (MoE). When non-empty, overrides
    /// `first_k_dense_replace`.
    #[serde(default)]
    pub mlp_layer_types: Vec<String>,

    // --- DSA (Dense-Sparse Attention) ---
    /// Per-layer indexer type: "full" (dense attention) or "shared" (sparse
    /// top-k indexed attention).
    #[serde(default)]
    pub indexer_types: Vec<String>,
    /// Head dim for the DSA index scoring projection.
    #[serde(default)]
    pub index_head_dim: u32,
    /// Number of attention heads used in the DSA index scoring.
    #[serde(default)]
    pub index_n_heads: u32,
    /// Top-k KV positions selected per token in sparse layers.
    #[serde(default)]
    pub index_topk: u32,
    /// Frequency of top-k recomputation (every N tokens).
    #[serde(default = "default_topk_freq")]
    pub index_topk_freq: u32,
    /// Use interleaved RoPE in the indexer (matches main RoPE convention).
    #[serde(default)]
    pub indexer_rope_interleave: bool,
    /// First N layers skip DSA (use full attention regardless of indexer_types).
    #[serde(default)]
    pub index_skip_topk_offset: u32,

    // --- Multi-token prediction ---
    /// Number of additional next-token prediction heads (0 = greedy only).
    #[serde(default)]
    pub num_nextn_predict_layers: u32,

    // --- MoE routing specifics ---
    /// Scoring function for the router ("softmax" or "sigmoid").
    #[serde(default = "default_scoring")]
    pub scoring_func: String,
    /// Multiplicative scaling applied to routed expert outputs.
    #[serde(default = "default_routed_scale")]
    pub routed_scaling_factor: f32,
    /// Normalize top-k probabilities to sum to 1.
    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

fn default_topk_freq() -> u32 {
    4
}
fn default_scoring() -> String {
    "sigmoid".into()
}
fn default_routed_scale() -> f32 {
    1.0
}

impl Default for GlmConfig {
    fn default() -> Self {
        // GLM-5.2 defaults.
        GlmConfig {
            vocab_size: 154_880,
            hidden_size: 6144,
            intermediate_size: 12_288,
            num_hidden_layers: 78,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            head_dim: 192,
            rms_norm_eps: 1e-5,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_head_dim: 256,
            qk_nope_head_dim: 192,
            qk_rope_head_dim: 64,
            v_head_dim: 256,
            rope_interleave: true,
            rope_parameters: None,
            first_k_dense_replace: 3,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_intermediate_size: 2048,
            mlp_layer_types: Vec::new(),
            indexer_types: Vec::new(),
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk: 2048,
            index_topk_freq: 4,
            indexer_rope_interleave: true,
            index_skip_topk_offset: 3,
            num_nextn_predict_layers: 1,
            scoring_func: "sigmoid".into(),
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
            torch_dtype: None,
        }
    }
}

/// Nested RoPE parameters for GLM.
#[derive(Debug, Clone, Deserialize)]
pub struct GlmRopeParams {
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_theta() -> f32 {
    8_000_000.0
}
fn default_rope_type() -> String {
    "default".into()
}

impl GlmConfig {
    /// RoPE theta, respecting nested rope_parameters if present.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map(|rp| rp.rope_theta)
            .unwrap_or(8_000_000.0)
    }

    /// Whether layer `i` uses a dense MLP (vs MoE).
    pub fn layer_is_dense(&self, layer: u32) -> bool {
        if let Some(t) = self.mlp_layer_types.get(layer as usize) {
            return t == "dense";
        }
        layer < self.first_k_dense_replace
    }

    /// Whether layer `i` uses full (dense) attention or sparse DSA.
    pub fn layer_is_full_attn(&self, layer: u32) -> bool {
        // Layers below offset always use full attention.
        if layer < self.index_skip_topk_offset {
            return true;
        }
        if let Some(t) = self.indexer_types.get(layer as usize) {
            return t == "full";
        }
        true
    }
}
