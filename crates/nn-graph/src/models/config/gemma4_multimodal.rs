//! Gemma 4 multimodal (unified) configuration: vision encoder + projector + text decoder.

use super::{GemmaConfig, SiglipConfig};
use serde::Deserialize;

/// Top-level Gemma 4 multimodal config wrapping vision, projector, and text.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Gemma4MultimodalConfig {
    /// Vision encoder (SigLIP-based).
    pub vision_config: SiglipConfig,
    /// Multimodal projector that maps vision embeddings to text hidden size.
    pub mm_input_projection_config: MmProjectorConfig,
    /// Text decoder configuration.
    pub text_config: GemmaConfig,
    /// Number of soft tokens per image (how many token slots one image occupies
    /// in the text sequence). Typically `image_seq_length = (image_size / patch_size)^2`.
    /// When 0, computed from vision_config.
    pub image_seq_length: i64,
}

/// Multi-modal projector: linear layer(s) mapping vision hidden → text hidden.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MmProjectorConfig {
    /// Input dimension (vision encoder hidden size).
    pub input_size: i64,
    /// Output dimension (text decoder hidden size). 0 ⇒ use text_config.hidden_size.
    pub output_size: i64,
}

impl Default for MmProjectorConfig {
    fn default() -> Self {
        MmProjectorConfig {
            input_size: 1152,
            output_size: 0,
        }
    }
}

impl Default for Gemma4MultimodalConfig {
    fn default() -> Self {
        Gemma4MultimodalConfig {
            vision_config: SiglipConfig::default(),
            mm_input_projection_config: MmProjectorConfig::default(),
            text_config: GemmaConfig::default(),
            image_seq_length: 0,
        }
    }
}

impl Gemma4MultimodalConfig {
    /// Number of image tokens injected into the text sequence per image.
    pub fn image_token_count(&self) -> i64 {
        if self.image_seq_length > 0 {
            self.image_seq_length
        } else {
            self.vision_config.num_patches()
        }
    }

    /// Output dimension of the projector (= text_config.hidden_size when 0).
    pub fn projector_output(&self) -> i64 {
        if self.mm_input_projection_config.output_size > 0 {
            self.mm_input_projection_config.output_size
        } else {
            self.text_config.hidden_size
        }
    }
}
