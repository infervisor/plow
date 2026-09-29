//! SNAC decoder, as exported by `scripts/tts/snac_export.py` (`model_type: "snac"`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SnacBlock {
    pub stride: u32,
    pub kernel: u32,
    pub padding: u32,
    pub output_padding: u32,
    pub cin: i64,
    pub cout: i64,
    pub dilations: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SnacConfig {
    pub codebook_size: i64,
    /// Latent rows per code at each level, finest first.
    pub vq_strides: Vec<i64>,
    pub latent_dim: i64,
    pub blocks: Vec<SnacBlock>,
}
