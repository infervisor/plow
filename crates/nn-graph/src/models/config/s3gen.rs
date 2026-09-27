//! Chatterbox S3Gen, as exported by `scripts/tts/s3gen_export.py`
//! (`model_type: "chatterbox_s3gen"`). The export records the HiFT geometry; the encoder and
//! estimator widths are the architecture's fixed values.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct S3GenConfig {
    pub vocab: i64,
    /// `[stride, kernel, padding]` per HiFT upsampling stage.
    pub upsample: Vec<[u32; 3]>,
    /// `[kernel, stride, padding]` per source downsampling conv.
    pub source_downs: Vec<[u32; 3]>,
    /// `(kernel, dilations)` per source ResBlock.
    pub source_resblocks: Vec<(u32, Vec<u32>)>,
    /// `(kernel, dilations)` per ResBlock, `resblocks.len() / upsample.len()` per stage.
    pub resblocks: Vec<(u32, Vec<u32>)>,
    pub n_fft: u32,
    pub harmonics: i64,
    #[serde(default = "d_enc")]
    pub encoder_dim: i64,
    #[serde(default = "enc_layers")]
    pub encoder_layers: (u32, u32),
    #[serde(default = "d_cfm")]
    pub estimator_dim: i64,
    #[serde(default = "heads")]
    pub heads: u32,
    #[serde(default = "mel")]
    pub mel_bins: i64,
    #[serde(default = "tblocks")]
    pub transformer_blocks: u32,
    #[serde(default = "mid_blocks")]
    pub mid_blocks: u32,
    #[serde(default = "f0_convs")]
    pub f0_convs: u32,
    #[serde(default = "f0_channels")]
    pub f0_channels: i64,
    #[serde(default = "hift_channels")]
    pub hift_channels: i64,
}

fn d_enc() -> i64 {
    512
}
fn enc_layers() -> (u32, u32) {
    (6, 4)
}
fn d_cfm() -> i64 {
    256
}
fn heads() -> u32 {
    8
}
fn mel() -> i64 {
    80
}
fn tblocks() -> u32 {
    4
}
fn mid_blocks() -> u32 {
    12
}
fn f0_convs() -> u32 {
    5
}
fn f0_channels() -> i64 {
    512
}
fn hift_channels() -> i64 {
    512
}
