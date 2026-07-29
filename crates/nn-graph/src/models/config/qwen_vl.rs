//! Qwen2-VL / Qwen2.5-VL vision encoder configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QwenVlVisionConfig {
    /// Number of transformer blocks in the vision tower.
    pub depth: u32,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_heads: u32,
    pub in_channels: i64,
    pub patch_size: i64,
    /// Spatial merge factor for the patch merger (e.g. 2 ⇒ 2x2 patches merged).
    pub spatial_merge_size: i64,
    pub temporal_patch_size: i64,
    /// Output hidden size after the patch merger (projection into the LLM).
    pub out_hidden_size: i64,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
}

impl Default for QwenVlVisionConfig {
    fn default() -> Self {
        // Qwen2.5-VL-7B vision config defaults.
        QwenVlVisionConfig {
            depth: 32,
            hidden_size: 1280,
            intermediate_size: 3420,
            num_heads: 16,
            in_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 3584,
            torch_dtype: None,
        }
    }
}

impl QwenVlVisionConfig {
    pub fn head_dim(&self) -> u32 {
        (self.hidden_size / self.num_heads as i64) as u32
    }

    /// Patch-embed conv flattens `in_c * temporal * patch * patch` channels.
    pub fn patch_input_dim(&self) -> i64 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn merged_dim(&self) -> i64 {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    /// Vision RoPE base; not present in the sub-config, Qwen uses 10000.
    pub fn rope_theta(&self) -> f32 {
        10_000.0
    }
}
