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

impl Default for SnacConfig {
    fn default() -> Self {
        SnacConfig {
            codebook_size: 4096,
            vq_strides: vec![8, 4, 2, 1],
            latent_dim: 1024,
            blocks: vec![
                SnacBlock {
                    stride: 2,
                    kernel: 4,
                    padding: 1,
                    output_padding: 0,
                    cin: 1024,
                    cout: 512,
                    dilations: vec![1, 3, 9],
                },
                SnacBlock {
                    stride: 4,
                    kernel: 8,
                    padding: 2,
                    output_padding: 0,
                    cin: 512,
                    cout: 256,
                    dilations: vec![1, 3, 9],
                },
                SnacBlock {
                    stride: 8,
                    kernel: 16,
                    padding: 4,
                    output_padding: 0,
                    cin: 256,
                    cout: 128,
                    dilations: vec![1, 3, 9],
                },
            ],
        }
    }
}
