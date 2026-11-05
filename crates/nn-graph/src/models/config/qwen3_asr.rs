//! Qwen3-ASR configuration (HuggingFace `config.json`).
//!
//! Qwen3-ASR pairs an audio front-end (three strided 2D convolutions + transformer encoder +
//! projection) with a Qwen3 text thinker decoder. Checkpoints typically nest them
//! under `thinker_config`.

use super::qwen3::Qwen3Config;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Qwen3AudioConfig {
    /// Hidden dimension of the audio encoder transformer.
    pub d_model: i64,
    /// Number of transformer encoder layers.
    pub encoder_layers: u32,
    /// Number of attention heads in the audio encoder.
    pub encoder_attention_heads: u32,
    /// Intermediate dimension of the audio encoder feed-forward network.
    pub encoder_ffn_dim: i64,
    /// Output dimension of the audio-to-text projection, matching text hidden size.
    pub output_dim: i64,
    /// Input mel-spectrogram filterbank dimension.
    pub num_mel_bins: i64,
    /// Channel count of the 2D-conv front-end.
    pub downsample_hidden_size: i64,
    /// Temporal downsampling factor of the 2D conv front-end.
    pub downsample_rate: u32,
    /// Maximum source positions / frames accepted by the encoder.
    pub max_source_positions: u32,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for Qwen3AudioConfig {
    fn default() -> Self {
        // Qwen3-ASR-1.7B audio encoder defaults.
        Qwen3AudioConfig {
            d_model: 1024,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            output_dim: 2048,
            num_mel_bins: 128,
            downsample_hidden_size: 480,
            downsample_rate: 8,
            max_source_positions: 1500,
            torch_dtype: None,
        }
    }
}

impl Qwen3AudioConfig {
    pub fn head_dim(&self) -> i64 {
        self.d_model / self.encoder_attention_heads as i64
    }
}

/// Composite Qwen3-ASR configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Qwen3AsrConfig {
    pub audio_config: Qwen3AudioConfig,
    pub text_config: Qwen3Config,
}

impl Default for Qwen3AsrConfig {
    fn default() -> Self {
        Qwen3AsrConfig {
            audio_config: Qwen3AudioConfig::default(),
            text_config: Qwen3Config {
                hidden_size: 2048,
                intermediate_size: 6144,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: Some(128),
                vocab_size: 151936,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                tie_word_embeddings: true,
                ..Default::default()
            },
        }
    }
}
