//! Qwen-Image DiT (QwenImageTransformer2DModel) — MMDiT diffusion transformer configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QwenImageDitConfig {
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub attention_head_dim: i64,
    /// Patchified latent channels (z_dim * patch_size²).
    pub in_channels: i64,
    /// Output latent channels (z_dim).
    pub out_channels: i64,
    pub patch_size: i64,
    /// Hidden size of the text-encoder stream fed into joint attention.
    pub joint_attention_dim: i64,
    pub pooled_projection_dim: i64,
    /// Per-axis RoPE dims [temporal, height, width]; sums to head_dim.
    pub axes_dims_rope: Vec<u32>,
    pub guidance_embeds: bool,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for QwenImageDitConfig {
    fn default() -> Self {
        QwenImageDitConfig {
            num_layers: 60,
            num_attention_heads: 24,
            attention_head_dim: 128,
            in_channels: 64,
            out_channels: 16,
            patch_size: 2,
            joint_attention_dim: 3584,
            pooled_projection_dim: 768,
            axes_dims_rope: vec![16, 56, 56],
            guidance_embeds: false,
            torch_dtype: None,
        }
    }
}

impl QwenImageDitConfig {
    /// Inner hidden size = heads * head_dim.
    pub fn hidden_size(&self) -> i64 {
        self.num_attention_heads as i64 * self.attention_head_dim
    }
}
