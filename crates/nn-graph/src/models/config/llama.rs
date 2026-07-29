//! Llama / Mistral configuration (HuggingFace `config.json`).

use serde::Deserialize;

/// Llama / Mistral text decoder config. Covers Llama 2/3, Mistral 7B, and
/// derivatives (CodeLlama, Vicuna, etc.) — all share the same architecture:
/// pre-norm RMSNorm, GQA, SwiGLU MLP, RoPE.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    #[serde(default = "default_vocab")]
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<i64>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    /// Whether the lm_head shares weights with the embedding table. Llama 2
    /// and most smaller models tie; Llama 3/3.1 8B+ have a separate lm_head.
    /// Defaults to `true` (HuggingFace convention).
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    pub fn kv_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> i64 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads as i64)
    }
}

fn default_vocab() -> i64 {
    32000
}
fn default_head_dim() -> Option<i64> {
    None
}
fn default_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_tie() -> bool {
    true
}
