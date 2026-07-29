//! Qwen3 / Qwen2.5 configuration (HuggingFace `config.json`).
//!
//! Qwen3 is a standard dense GQA + SwiGLU architecture, structurally
//! identical to the LLaMA family but with its own config naming and
//! rope_theta defaults.

use serde::Deserialize;

fn default_tie() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Qwen3Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    /// Explicit head dim — Qwen3 sets this independently of hidden/heads
    /// (e.g. Qwen3-4B: hidden 2560, 32 heads, head_dim 128, not 80).
    pub head_dim: Option<i64>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        // Qwen3-8B defaults.
        Qwen3Config {
            vocab_size: 151_936,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: None,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            torch_dtype: None,
        }
    }
}

impl Qwen3Config {
    pub fn kv_heads(&self) -> u32 {
        self.num_key_value_heads
    }

    pub fn head_dim(&self) -> i64 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads as i64)
    }
}
