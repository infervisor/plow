//! SigLIP vision tower (embedding encoder) configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SiglipConfig {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_channels: i64,
    pub image_size: i64,
    pub patch_size: i64,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for SiglipConfig {
    fn default() -> Self {
        // siglip-so400m-patch14-384 defaults.
        SiglipConfig {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 384,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            torch_dtype: None,
        }
    }
}

impl SiglipConfig {
    /// Number of patches along one side.
    pub fn patches_per_side(&self) -> i64 {
        self.image_size / self.patch_size
    }

    /// Total patch count (sequence length).
    pub fn num_patches(&self) -> i64 {
        let p = self.patches_per_side();
        p * p
    }
}
