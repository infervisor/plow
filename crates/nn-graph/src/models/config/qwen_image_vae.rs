//! Qwen-Image VAE (AutoencoderKLQwenImage) — 3D causal autoencoder configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QwenImageVaeConfig {
    pub base_dim: i64,
    pub dim_mult: Vec<i64>,
    pub num_res_blocks: u32,
    /// Latent channel count (encoder outputs `2 * z_dim`: mean + logvar).
    pub z_dim: i64,
    /// Per-downsample-stage temporal downsampling flags.
    pub temperal_downsample: Vec<bool>,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for QwenImageVaeConfig {
    fn default() -> Self {
        QwenImageVaeConfig {
            base_dim: 96,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_dim: 16,
            temperal_downsample: vec![false, true, true],
            torch_dtype: None,
        }
    }
}

impl QwenImageVaeConfig {
    /// Channel width at each stage: `base_dim * dim_mult[i]`.
    pub fn stage_dims(&self) -> Vec<i64> {
        self.dim_mult.iter().map(|m| self.base_dim * m).collect()
    }

    /// Total spatial downsample factor (one halving per stage transition).
    pub fn spatial_downsample(&self) -> i64 {
        1 << (self.dim_mult.len() as u32 - 1)
    }
}
