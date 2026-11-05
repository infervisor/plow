//! FLUX.1 DiT (diffusers `FluxTransformer2DModel`) configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FluxConfig {
    pub patch_size: i64,
    /// Packed latent channels (VAE z_dim * 2x2 packing).
    pub in_channels: i64,
    /// `null` in the checkpoint means `in_channels`.
    pub out_channels: Option<i64>,
    /// Double-stream (MMDiT) blocks: `transformer_blocks.*`.
    pub num_layers: u32,
    /// Single-stream blocks: `single_transformer_blocks.*`.
    pub num_single_layers: u32,
    pub attention_head_dim: i64,
    pub num_attention_heads: u32,
    /// T5 hidden size fed to `context_embedder`.
    pub joint_attention_dim: i64,
    /// CLIP pooled embedding size fed to `time_text_embed.text_embedder`.
    pub pooled_projection_dim: i64,
    /// FLUX.1-dev embeds the guidance scale; FLUX.1-schnell does not.
    pub guidance_embeds: bool,
    pub axes_dims_rope: Vec<u32>,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for FluxConfig {
    fn default() -> Self {
        // FLUX.1-dev.
        FluxConfig {
            patch_size: 1,
            in_channels: 64,
            out_channels: None,
            num_layers: 19,
            num_single_layers: 38,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 4096,
            pooled_projection_dim: 768,
            guidance_embeds: true,
            axes_dims_rope: vec![16, 56, 56],
            torch_dtype: None,
        }
    }
}

impl FluxConfig {
    pub fn hidden_size(&self) -> i64 {
        self.num_attention_heads as i64 * self.attention_head_dim
    }
}
