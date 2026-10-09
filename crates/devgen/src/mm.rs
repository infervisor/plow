//! Multimodal encoders: a checkpoint's vision / audio towers and their projections into the LM
//! embedding space, lowered to `forward.v1` sidecar packets with capacity rungs, plus the LM-side
//! contract (`plow.multimodal.v1`) that tells the runtime how to place the encoder rows.
//!
//! The runtime never names a model: the contract carries the placeholder token ids, the soft-token
//! wrapping, the processor parameters and the sidecar names; each sidecar carries its own inputs.
use std::collections::BTreeMap;

use packet::dev::{DevOp, ACT_CLAMP, ACT_GELU_TANH, ACT_RELU, ACT_ROUND_BF16, ACT_SCALE_SHIFT, ACT_SILU, TENSOR_NONE};
use plow_asset::multimodal::{MmContract, MmModality};
use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor};
use serde_json::Value;

use crate::checkpoint::TensorReader;
use crate::pipeline::{
    AttentionF32Stage, DenseActivation, DenseF32ProgramStage, DenseWeight, GatherRowsF32Stage, PacketPrefix,
    StageProgram, TensorRef,
};

pub const VISION_PACKET: &str = "mm_vision.pkt";
pub const AUDIO_PACKET: &str = "mm_audio.pkt";
pub const PIPELINE: &str = "mm.encode";

/// Encoder rows the runtime may hold for in-flight requests (`in.mm_slab`), default.
pub const DEFAULT_SLAB_ROWS: u32 = 8192;

fn f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn round_bf16(v: f32) -> f32 {
    crate::config::bf16_round(v)
}

struct Ckpt {
    reader: TensorReader,
}

impl Ckpt {
    fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let (dtype, bytes) = self.reader.read(name)?;
        match dtype {
            "BF16" => Ok(bf16_to_f32(&bytes)),
            "F32" => Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
            other => Err(format!("{name}: unsupported encoder dtype {other}")),
        }
    }
    fn scalar(&self, name: &str) -> Result<f32, String> {
        self.f32s(name)?.first().copied().ok_or_else(|| format!("{name} is empty"))
    }
    fn bf16_raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let (dtype, bytes) = self.reader.read(name)?;
        match dtype {
            "BF16" => Ok(bytes),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .flat_map(|b| ((round_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_bits() >> 16) as u16).to_le_bytes())
                .collect()),
            other => Err(format!("{name}: unsupported encoder dtype {other}")),
        }
    }
}

/// How a tower's tensors get their bytes: checkpoint BF16 (dense weights), checkpoint as F32
/// (norm gammas, tables, conv weights), or computed at lowering.
enum Init {
    Bf16(String),
    F32(String),
    Bytes(Vec<u8>),
}

/// One encoder program sequence under construction: every op waits on the previous one.
struct Seq {
    prefix: Option<PacketPrefix>,
    program: Option<StageProgram>,
    last: Option<u32>,
    init: BTreeMap<String, Init>,
    tag: u32,
}

#[derive(Clone, Copy)]
struct Clip {
    lo: f32,
    hi: f32,
}

impl Seq {
    fn p(&mut self) -> &mut StageProgram {
        if self.program.is_none() {
            let prefix = self.prefix.take().expect("encoder prefix");
            self.program = Some(prefix.program());
            self.last = None;
        }
        self.program.as_mut().unwrap()
    }

    fn deps(&self) -> Vec<u32> {
        self.last.into_iter().collect()
    }

    /// Close the current program (one launch).
    fn cut(&mut self) {
        if let Some(program) = self.program.take() {
            self.prefix = Some(program.finish(self.tag));
            self.last = None;
        }
    }

    fn t(&mut self, name: &str, elements: u64) -> u32 {
        self.p().declare(name, elements * 4)
    }

    fn weight_bf16(&mut self, name: &str, elements: u64) -> u32 {
        self.init.entry(name.to_string()).or_insert_with(|| Init::Bf16(name.to_string()));
        self.p().declare(name, elements * 2)
    }

    fn weight_f32(&mut self, ckpt_name: &str, elements: u64) -> u32 {
        let name = format!("{ckpt_name}.f32");
        self.init.entry(name.clone()).or_insert_with(|| Init::F32(ckpt_name.to_string()));
        self.p().declare(&name, elements * 4)
    }

    fn constant(&mut self, name: &str, values: Vec<f32>) -> u32 {
        let bytes = values.len() as u64 * 4;
        self.init.entry(name.to_string()).or_insert_with(|| Init::Bytes(f32_le(&values)));
        self.p().declare(name, bytes)
    }

    fn constant_u32(&mut self, name: &str, values: &[u32]) -> u32 {
        let bytes = values.len() as u64 * 4;
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.init.entry(name.to_string()).or_insert_with(|| Init::Bytes(raw));
        self.p().declare(name, bytes)
    }

    fn done(&mut self, e: crate::pipeline::Emitted) -> u32 {
        self.last = Some(e.done);
        e.output
    }

    fn unary(&mut self, x: u32, out: Option<&str>, rows: u32, width: u32, kind: u32, p0: f32, p1: f32) -> Result<u32, String> {
        let deps = self.deps();
        let output = match out {
            Some(name) => self.t(name, u64::from(rows) * u64::from(width)),
            None => x,
        };
        let elements = u64::from(rows) * u64::from(width);
        let e = self.p().raw(DevOp::UnaryF32, elements.div_ceil(2048), &deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, x, TENSOR_NONE]);
            d.i[..5].copy_from_slice(&[rows, width, kind, 0, 0]);
            d.f = [p0, p1];
        })?;
        Ok(self.done(e))
    }

    fn round(&mut self, x: u32, rows: u32, width: u32) -> Result<u32, String> {
        self.unary(x, None, rows, width, ACT_ROUND_BF16, 0.0, 0.0)
    }

    fn clamp(&mut self, x: u32, out: Option<&str>, rows: u32, width: u32, clip: Option<Clip>) -> Result<u32, String> {
        match clip {
            Some(c) => self.unary(x, out, rows, width, ACT_CLAMP, c.lo, c.hi),
            None => Ok(x),
        }
    }

    /// `out = x . W^T (+ bias)`, bf16 weights, rounded to bf16 like a bf16 torch Linear.
    #[allow(clippy::too_many_arguments)]
    fn dense(&mut self, x: u32, out: &str, rows: u32, k: u32, n: u32, weight: &str, bias: Option<&str>) -> Result<u32, String> {
        let deps = self.deps();
        let w = self.weight_bf16(weight, u64::from(n) * u64::from(k));
        let b = bias.map(|name| self.weight_f32(name, u64::from(n)));
        let e = self.p().dense_f32(
            x,
            &deps,
            DenseF32ProgramStage {
                output: TensorRef::Named(out),
                weight: TensorRef::Handle(w),
                bias: b.map(TensorRef::Handle),
                rows,
                input_width: k,
                output_width: n,
                activation: DenseActivation::None,
                round_bf16: true,
                weight_type: DenseWeight::Bf16,
                operands_bf16_exact: true,
                tf32x3: false,
                layer_norm: None,
            },
        )?;
        Ok(self.done(e))
    }

    /// Dense with a computed f32 weight (`[n][k]`, rounded to bf16 by the caller when it must be).
    #[allow(clippy::too_many_arguments)]
    fn dense_f32w(&mut self, x: u32, out: &str, rows: u32, k: u32, n: u32, weight: u32, bf16_exact: bool) -> Result<u32, String> {
        let deps = self.deps();
        let e = self.p().dense_f32(
            x,
            &deps,
            DenseF32ProgramStage {
                output: TensorRef::Named(out),
                weight: TensorRef::Handle(weight),
                bias: None,
                rows,
                input_width: k,
                output_width: n,
                activation: DenseActivation::None,
                round_bf16: true,
                weight_type: DenseWeight::F32,
                operands_bf16_exact: bf16_exact,
                tf32x3: !bf16_exact,
                layer_norm: None,
            },
        )?;
        Ok(self.done(e))
    }

    /// Clipped linear: clamp input, dense, clamp output (Gemma4ClippableLinear).
    #[allow(clippy::too_many_arguments)]
    fn clipped(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        k: u32,
        n: u32,
        weight: &str,
        clip_in: Option<Clip>,
        clip_out: Option<Clip>,
    ) -> Result<u32, String> {
        let input = self.clamp(x, Some(&format!("{out}.in")), rows, k, clip_in)?;
        let y = self.dense(input, out, rows, k, n, weight, None)?;
        self.clamp(y, None, rows, n, clip_out)
    }

    /// Grouped RMSNorm (RmsNormF32), bf16-rounded output.
    #[allow(clippy::too_many_arguments)]
    fn rms(&mut self, x: u32, out: Option<&str>, rows: u32, groups: u32, width: u32, gamma: Option<&str>, eps: f32) -> Result<u32, String> {
        let deps = self.deps();
        let output = match out {
            Some(name) => self.t(name, u64::from(rows) * u64::from(groups * width)),
            None => x,
        };
        let g = gamma.map(|name| self.weight_f32(name, u64::from(width))).unwrap_or(TENSOR_NONE);
        let units = (u64::from(rows) * u64::from(groups)).div_ceil(8);
        let e = self.p().raw(DevOp::RmsNormF32, units, &deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, x, g]);
            d.i[..6].copy_from_slice(&[rows, groups, width, 0, 2, 0]);
            d.f[0] = eps;
        })?;
        Ok(self.done(e))
    }

    /// LayerNormF32 over `width` with optional gamma/beta and a post activation, bf16-rounded.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        width: u32,
        gamma: Option<&str>,
        beta: Option<&str>,
        eps: f32,
        act: u32,
    ) -> Result<u32, String> {
        let deps = self.deps();
        let output = self.t(out, u64::from(rows) * u64::from(width));
        let g = gamma.map(|n| self.weight_f32(n, u64::from(width))).unwrap_or(TENSOR_NONE);
        let b = beta.map(|n| self.weight_f32(n, u64::from(width))).unwrap_or(TENSOR_NONE);
        let e = self.p().raw(DevOp::LayerNormF32, u64::from(rows).div_ceil(8), &deps, output, |d| {
            d.t[..5].copy_from_slice(&[output, x, g, b, TENSOR_NONE]);
            d.i[..3].copy_from_slice(&[rows, width, 1 | (act << 4)]);
            d.f[0] = eps;
        })?;
        Ok(self.done(e))
    }

    /// `out = a op b` over `[items][rows][width]` with `b` strided; bf16-rounded afterwards.
    #[allow(clippy::too_many_arguments)]
    fn binary(&mut self, a: u32, b: u32, out: Option<&str>, op: u32, dims: [u32; 3], b_strides: [u32; 3], round: bool) -> Result<u32, String> {
        let deps = self.deps();
        let elements = dims.iter().map(|&v| u64::from(v)).product::<u64>();
        let output = match out {
            Some(name) => self.t(name, elements),
            None => a,
        };
        let e = self.p().raw(DevOp::BinaryF32, elements.div_ceil(2048), &deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, a, b]);
            d.i = [dims[0], dims[1], dims[2], op, b_strides[0], b_strides[1], b_strides[2], 0];
        })?;
        let y = self.done(e);
        if round {
            self.round(y, 1, u32::try_from(elements).map_err(|_| "binary too large")?)
        } else {
            Ok(y)
        }
    }

    fn add(&mut self, a: u32, b: u32, rows: u32, width: u32) -> Result<u32, String> {
        self.binary(a, b, None, 0, [1, rows, width], [0, width, 1], true)
    }

    #[allow(clippy::too_many_arguments)]
    fn gather(
        &mut self,
        out: u32,
        table: u32,
        table_rows: u32,
        index: u32,
        rows: u32,
        width: u32,
        accumulate: bool,
    ) -> Result<u32, String> {
        let deps = self.deps();
        let e = self.p().gather_rows_f32(
            &deps,
            GatherRowsF32Stage {
                output: TensorRef::Handle(out),
                table: TensorRef::Handle(table),
                table_rows,
                index: Some(index),
                rows,
                width,
                vocab: table_rows,
                rows_per_item: 0,
                repeat: 0,
                index_item_stride: 0,
                table_item_stride: 0,
                table_f16: false,
                accumulate,
                out_stride: 0,
                out_col0: 0,
            },
        )?;
        Ok(self.done(e))
    }
}

/// Gemma-4 SigLIP-style vision tower (`gemma4_vision`) and its `embed_vision` projection.
#[derive(Clone, Debug)]
pub struct VisionTower {
    hidden: u32,
    layers: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    patch: u32,
    pool: u32,
    pos_size: u32,
    theta: f32,
    eps: f32,
    clipped: bool,
    standardize: bool,
    text_hidden: u32,
    max_soft_tokens: u32,
    prefix: String,
    embed: String,
}

/// Gemma-4 USM conformer audio tower (`gemma4_audio`) and its `embed_audio` projection.
#[derive(Clone, Debug)]
pub struct AudioTower {
    hidden: u32,
    layers: u32,
    heads: u32,
    chunk: u32,
    context_left: u32,
    context_right: u32,
    conv_kernel: u32,
    channels: [u32; 2],
    output_dims: u32,
    softcap: f32,
    residual_weight: f32,
    eps: f32,
    clipped: bool,
    text_hidden: u32,
    mel_bins: u32,
    prefix: String,
    embed: String,
}

/// Encoder-free audio (`gemma4_unified_audio`): each token is a frame of raw samples, projected
/// by `embed_audio` (scale-free RMSNorm, then Linear).
#[derive(Clone, Debug)]
pub struct FrameAudio {
    samples: u32,
    eps: f32,
    text_hidden: u32,
    embed: String,
}

#[derive(Clone, Debug)]
pub enum Audio {
    Conformer(AudioTower),
    Frames(FrameAudio),
}

pub struct Towers {
    pub vision: Option<VisionTower>,
    pub audio: Option<Audio>,
    /// Declared towers this build leaves out, with why (logged; their modality answers 400).
    pub skipped: Vec<String>,
    config: Value,
    processor: Value,
}

fn u(v: &Value, key: &str) -> Result<u32, String> {
    v[key].as_u64().and_then(|x| u32::try_from(x).ok()).ok_or_else(|| format!("config field {key} missing"))
}
fn f(v: &Value, key: &str) -> Result<f32, String> {
    v[key].as_f64().map(|x| x as f32).ok_or_else(|| format!("config field {key} missing"))
}

impl Towers {
    /// The towers a checkpoint declares. A declared tower this lowering does not know is an error,
    /// never a silent text-only build.
    pub fn from_checkpoint(dir: &std::path::Path) -> Result<Self, String> {
        let json = |file: &str| -> Result<Value, String> {
            serde_json::from_slice(&std::fs::read(dir.join(file)).map_err(|e| format!("{file}: {e}"))?)
                .map_err(|e| format!("{file}: {e}"))
        };
        let config = json("config.json")?;
        let processor = json("processor_config.json").unwrap_or(Value::Null);
        let text = &config["text_config"];
        let text_hidden = u(text, "hidden_size")?;
        let mut skipped = Vec::new();
        let vision = match config["vision_config"]["model_type"].as_str() {
            None => None,
            // The LM attends bidirectionally within each image on its sliding layers; the LM
            // attention kernels are causal-only, so image prompts would diverge from the checkpoint.
            Some(t) if text["use_bidirectional_attention"].as_str() == Some("vision") => {
                skipped.push(format!("vision ({t}): the LM needs bidirectional attention within images"));
                None
            }
            Some("gemma4_vision") => {
                let v = &config["vision_config"];
                let image = &processor["image_processor"];
                if v["rope_parameters"]["rope_type"].as_str().is_some_and(|t| t != "default" && t != "axial") {
                    return Err("gemma4_vision: unsupported rope type".into());
                }
                Some(VisionTower {
                    hidden: u(v, "hidden_size")?,
                    layers: u(v, "num_hidden_layers")?,
                    heads: u(v, "num_attention_heads")?,
                    head_dim: u(v, "head_dim")?,
                    inter: u(v, "intermediate_size")?,
                    patch: u(v, "patch_size")?,
                    pool: u(v, "pooling_kernel_size")?,
                    pos_size: u(v, "position_embedding_size")?,
                    theta: v["rope_parameters"]["rope_theta"].as_f64().unwrap_or(100.0) as f32,
                    eps: f(v, "rms_norm_eps")?,
                    clipped: v["use_clipped_linears"].as_bool().unwrap_or(false),
                    standardize: v["standardize"].as_bool().unwrap_or(false),
                    text_hidden,
                    max_soft_tokens: u(image, "max_soft_tokens").or_else(|_| u(&config, "vision_soft_tokens_per_image"))?,
                    prefix: "model.vision_tower.".into(),
                    embed: "model.embed_vision.".into(),
                })
            }
            Some(other) => return Err(format!("vision tower {other:?} has no multimodal lowering")),
        };
        let audio = match config["audio_config"]["model_type"].as_str() {
            None => None,
            Some("gemma4_unified_audio") => {
                let a = &config["audio_config"];
                Some(Audio::Frames(FrameAudio {
                    samples: u(&processor["feature_extractor"], "audio_samples_per_token").or_else(|_| u(a, "audio_embed_dim"))?,
                    eps: f(a, "rms_norm_eps")?,
                    text_hidden,
                    embed: "model.embed_audio.".into(),
                }))
            }
            Some("gemma4_audio") => {
                let a = &config["audio_config"];
                let channels = a["subsampling_conv_channels"]
                    .as_array()
                    .filter(|c| c.len() == 2)
                    .ok_or("audio subsampling_conv_channels must have two layers")?;
                let feature = &processor["feature_extractor"];
                Some(Audio::Conformer(AudioTower {
                    hidden: u(a, "hidden_size")?,
                    layers: u(a, "num_hidden_layers")?,
                    heads: u(a, "num_attention_heads")?,
                    chunk: u(a, "attention_chunk_size")?,
                    context_left: u(a, "attention_context_left")?,
                    context_right: u(a, "attention_context_right")?,
                    conv_kernel: u(a, "conv_kernel_size")?,
                    channels: [
                        channels[0].as_u64().ok_or("conv channels")? as u32,
                        channels[1].as_u64().ok_or("conv channels")? as u32,
                    ],
                    output_dims: u(a, "output_proj_dims")?,
                    softcap: f(a, "attention_logit_cap")?,
                    residual_weight: f(a, "residual_weight")?,
                    eps: f(a, "rms_norm_eps")?,
                    clipped: a["use_clipped_linears"].as_bool().unwrap_or(false),
                    text_hidden,
                    mel_bins: u(feature, "feature_size").unwrap_or(128),
                    prefix: "model.audio_tower.".into(),
                    embed: "model.embed_audio.".into(),
                }))
            }
            Some(other) => return Err(format!("audio tower {other:?} has no multimodal lowering")),
        };
        if let Some(Audio::Conformer(a)) = &audio {
            if a.hidden % a.heads != 0 || a.mel_bins % 4 != 0 || a.context_right != 0 {
                return Err("gemma4_audio geometry is unsupported".into());
            }
        }
        if let Some(v) = &vision {
            if !matches!(v.head_dim, 64 | 128) || v.hidden != v.heads * v.head_dim {
                return Err(format!("vision head_dim {} has no attention lowering (64 or 128)", v.head_dim));
            }
        }
        Ok(Self { vision, audio, skipped, config, processor })
    }

    pub fn is_empty(&self) -> bool {
        self.vision.is_none() && self.audio.is_none()
    }

    /// The LM contract: placeholder ids, wrapping, processor parameters and sidecars.
    pub fn contract(&self, slab_rows: u32) -> Result<MmContract, String> {
        let c = &self.config;
        let text = &c["text_config"];
        let id = |key: &str| c[key].as_u64().and_then(|v| u32::try_from(v).ok());
        let mut modalities = Vec::new();
        if let Some(v) = &self.vision {
            let image = &self.processor["image_processor"];
            let mut params = BTreeMap::new();
            params.insert("patch_size".into(), u64::from(v.patch));
            params.insert("pool".into(), u64::from(v.pool));
            params.insert("max_soft_tokens".into(), u64::from(v.max_soft_tokens));
            let rescale = image["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0) as f32;
            params.insert("rescale_f32".into(), u64::from(rescale.to_bits()));
            params.insert("resample".into(), image["resample"].as_u64().unwrap_or(3));
            params.insert("normalize".into(), u64::from(image["do_normalize"].as_bool().unwrap_or(false)));
            for (k, key) in [("mean", "image_mean"), ("std", "image_std")] {
                for ch in 0..3 {
                    let value = image[key][ch].as_f64().unwrap_or(if k == "mean" { 0.0 } else { 1.0 }) as f32;
                    params.insert(format!("{k}{ch}_f32"), u64::from(value.to_bits()));
                }
            }
            modalities.push(MmModality {
                kind: "image".into(),
                packet: VISION_PACKET.into(),
                placeholder: id("image_token_id").ok_or("image_token_id missing")?,
                begin: id("boi_token_id"),
                end: id("eoi_token_id"),
                processor: "aspect_patches".into(),
                parameters: params,
            });
        }
        if let Some(Audio::Frames(a)) = &self.audio {
            let fe = &self.processor["feature_extractor"];
            let mut params = BTreeMap::new();
            params.insert("sample_rate".into(), fe["sampling_rate"].as_u64().unwrap_or(16_000));
            params.insert("frame_samples".into(), u64::from(a.samples));
            params.insert("max_soft_tokens".into(), self.processor["audio_seq_length"].as_u64().unwrap_or(750));
            modalities.push(MmModality {
                kind: "audio".into(),
                packet: AUDIO_PACKET.into(),
                placeholder: id("audio_token_id").ok_or("audio_token_id missing")?,
                begin: id("boa_token_id"),
                end: id("eoa_token_id").or_else(|| id("eoa_token_index")),
                processor: "waveform_frames".into(),
                parameters: params,
            });
        }
        if let Some(Audio::Conformer(a)) = &self.audio {
            let fe = &self.processor["feature_extractor"];
            let mut params = BTreeMap::new();
            let get = |key: &str, default: u64| fe[key].as_u64().unwrap_or(default);
            let getf = |key: &str, default: f64| u64::from((fe[key].as_f64().unwrap_or(default) as f32).to_bits());
            params.insert("sample_rate".into(), get("sampling_rate", 16_000));
            params.insert("frame_length".into(), get("frame_length", 320));
            params.insert("hop_length".into(), get("hop_length", 160));
            params.insert("fft_length".into(), get("fft_length", 512));
            params.insert("mel_bins".into(), u64::from(a.mel_bins));
            params.insert("min_frequency_f32".into(), getf("min_frequency", 0.0));
            params.insert("max_frequency_f32".into(), getf("max_frequency", 8000.0));
            params.insert("mel_floor_f32".into(), getf("mel_floor", 1e-3));
            params.insert("pad_multiple".into(), 128);
            params.insert("max_samples".into(), 480_000);
            params.insert("subsample".into(), 4);
            params.insert("max_soft_tokens".into(), self.processor["audio_seq_length"].as_u64().unwrap_or(750));
            if fe["preemphasis"].as_f64().unwrap_or(0.0) != 0.0
                || fe["dither"].as_f64().unwrap_or(0.0) != 0.0
                || !fe["per_bin_mean"].is_null()
                || fe["input_scale_factor"].as_f64().unwrap_or(1.0) != 1.0
            {
                return Err("audio feature extractor options are unsupported".into());
            }
            modalities.push(MmModality {
                kind: "audio".into(),
                packet: AUDIO_PACKET.into(),
                placeholder: id("audio_token_id").ok_or("audio_token_id missing")?,
                begin: id("boa_token_id"),
                end: id("eoa_token_id").or_else(|| id("eoa_token_index")),
                processor: "semicausal_log_mel".into(),
                parameters: params,
            });
        }
        let pad = text["pad_token_id"].as_u64().unwrap_or(0) as u32;
        let hidden = u(text, "hidden_size")?;
        Ok(MmContract {
            version: plow_asset::multimodal::VERSION,
            hidden,
            pad_token: pad,
            slab_rows,
            table_capacity: (2 * slab_rows).next_power_of_two(),
            modalities,
        })
    }

    pub fn lower_vision(&self, dir: &std::path::Path, n_cu: u32, target: u32, images: &[u32]) -> Result<Sidecar, String> {
        let v = self.vision.as_ref().ok_or("checkpoint has no vision tower")?;
        let ckpt = Ckpt { reader: TensorReader::open(dir)? };
        let mut images: Vec<u32> = images.to_vec();
        images.sort_unstable_by(|a, b| b.cmp(a));
        images.dedup();
        let mut seq = new_seq(n_cu, target);
        let mut programs = BTreeMap::new();
        for &n in &images {
            let first = seq.prefix.as_ref().map_or(0, |p| p.programs.len());
            lower_vision_capacity(&mut seq, v, &ckpt, n)?;
            seq.cut();
            let prefix = seq.prefix.as_ref().unwrap();
            programs.insert(n, prefix.programs[first..].to_vec());
        }
        finish_sidecar(seq, &ckpt, programs, "vision", |pipeline, prefix| {
            let max = images[0];
            let patches = max * v.max_soft_tokens * v.pool * v.pool;
            let tensors = [
                ("posx", "in.mm.v.posx", PipelineDType::U32, vec![u64::from(patches)]),
                ("posy", "in.mm.v.posy", PipelineDType::U32, vec![u64::from(patches)]),
                ("rope", "in.mm.v.rope", PipelineDType::U32, vec![u64::from(patches), 2]),
                ("valid", "in.mm.v.valid", PipelineDType::U32, vec![u64::from(max)]),
            ];
            add_tensors(pipeline, prefix, &tensors)?;
            let pools: Vec<(String, String)> =
                (0..v.pool * v.pool).map(|j| (format!("pool.{j}"), format!("in.mm.v.pool.{j}"))).collect();
            for (key, name) in &pools {
                add_tensors(pipeline, prefix, &[(key, name, PipelineDType::U32, vec![u64::from(max * v.max_soft_tokens)])])?;
            }
            let p = &mut pipeline.parameters;
            p.insert("item_rows".into(), u64::from(v.max_soft_tokens * v.pool * v.pool));
            p.insert("item_tokens".into(), u64::from(v.max_soft_tokens));
            p.insert("patch_values".into(), u64::from(3 * v.patch * v.patch));
            p.insert("pool".into(), u64::from(v.pool));
            p.insert("output_width".into(), u64::from(v.text_hidden));
            p.insert("position_limit".into(), u64::from(v.pos_size));
            // Host input transform: pixel' = bf16(pixel * scale + shift) (the tower's 2 * (x - 0.5)).
            p.insert("input.scale_f32".into(), u64::from(2.0f32.to_bits()));
            p.insert("input.shift_f32".into(), u64::from((-1.0f32).to_bits()));
            p.insert("input.round_bf16".into(), 1);
            Ok(())
        })
    }

    pub fn lower_audio(&self, dir: &std::path::Path, n_cu: u32, target: u32, frames: &[u32]) -> Result<Sidecar, String> {
        let ckpt = Ckpt { reader: TensorReader::open(dir)? };
        let a = match self.audio.as_ref().ok_or("checkpoint has no audio tower")? {
            Audio::Conformer(a) => a,
            Audio::Frames(a) => return lower_frames(a, &ckpt, n_cu, target, frames),
        };
        let mut frames: Vec<u32> = frames.iter().map(|f| f.next_multiple_of(4)).collect();
        frames.sort_unstable_by(|a, b| b.cmp(a));
        frames.dedup();
        let mut seq = new_seq(n_cu, target);
        let mut programs = BTreeMap::new();
        for &n in &frames {
            let first = seq.prefix.as_ref().map_or(0, |p| p.programs.len());
            lower_audio_capacity(&mut seq, a, &ckpt, n)?;
            seq.cut();
            let prefix = seq.prefix.as_ref().unwrap();
            programs.insert(n, prefix.programs[first..].to_vec());
        }
        finish_sidecar(seq, &ckpt, programs, "audio", |pipeline, prefix| {
            let max = frames[0];
            let tensors = [
                ("mask1", "in.mm.a.mask1", PipelineDType::F32, vec![u64::from(max / 2)]),
                ("valid", "in.mm.a.valid", PipelineDType::U32, vec![1]),
            ];
            add_tensors(pipeline, prefix, &tensors)?;
            let p = &mut pipeline.parameters;
            p.insert("mel_bins".into(), u64::from(a.mel_bins));
            p.insert("output_width".into(), u64::from(a.text_hidden));
            p.insert("subsample".into(), 4);
            p.insert("input.round_bf16".into(), 1);
            Ok(())
        })
    }
}

/// Encoder-free audio: rungs hold `mel frames / 4` tokens (the conformer's seconds per rung).
fn lower_frames(a: &FrameAudio, ckpt: &Ckpt, n_cu: u32, target: u32, frames: &[u32]) -> Result<Sidecar, String> {
    let mut tokens: Vec<u32> = frames.iter().map(|f| f.div_ceil(4).max(1)).collect();
    tokens.sort_unstable_by(|a, b| b.cmp(a));
    tokens.dedup();
    let mut seq = new_seq(n_cu, target);
    let mut programs = BTreeMap::new();
    for &n in &tokens {
        let first = seq.prefix.as_ref().map_or(0, |p| p.programs.len());
        seq.tag = n;
        let x = seq.t("in.mm.a.frames", u64::from(n * a.samples));
        seq.rms(x, None, n, 1, a.samples, None, a.eps)?;
        let out = seq.dense(x, "act.mm.a.out", n, a.samples, a.text_hidden, &format!("{}embedding_projection.weight", a.embed), None)?;
        seq.cut();
        let prefix = seq.prefix.as_mut().unwrap();
        prefix.output = out;
        prefix.input = x;
        programs.insert(n, prefix.programs[first..].to_vec());
    }
    finish_sidecar(seq, ckpt, programs, "frames", |pipeline, _| {
        let p = &mut pipeline.parameters;
        p.insert("frame_samples".into(), u64::from(a.samples));
        p.insert("output_width".into(), u64::from(a.text_hidden));
        p.insert("input.round_bf16".into(), 1);
        Ok(())
    })
}

/// A lowered encoder: its model (programs, tensors with weights) and pipeline section.
pub struct Sidecar {
    pub model: packet::devbuild::Model,
    pub section: packet::devbuild::SectionData,
}

fn new_seq(n_cu: u32, target: u32) -> Seq {
    let model = packet::devbuild::Model {
        n_cu,
        target,
        tensors: Vec::new(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    Seq {
        prefix: Some(PacketPrefix { model, programs: Vec::new(), input: 0, output: 0, input_shape: Vec::new() }),
        program: None,
        last: None,
        init: BTreeMap::new(),
        tag: 0,
    }
}

fn add_tensors(
    pipeline: &mut PacketPipeline,
    prefix: &PacketPrefix,
    tensors: &[(&str, &str, PipelineDType, Vec<u64>)],
) -> Result<(), String> {
    for (key, name, dtype, shape) in tensors {
        if !prefix.model.tensors.iter().any(|t| t.name == *name) {
            return Err(format!("encoder tensor {name} was never declared"));
        }
        pipeline.tensors.insert((*key).into(), PipelineTensor { name: (*name).into(), dtype: *dtype, shape: shape.clone() });
    }
    Ok(())
}

fn finish_sidecar(
    seq: Seq,
    ckpt: &Ckpt,
    programs: BTreeMap<u32, Vec<usize>>,
    modality: &str,
    extra: impl FnOnce(&mut PacketPipeline, &PacketPrefix) -> Result<(), String>,
) -> Result<Sidecar, String> {
    let Seq { prefix, init, .. } = seq;
    let mut prefix = prefix.ok_or("encoder has no programs")?;
    for tensor in &mut prefix.model.tensors {
        let Some(source) = init.get(&tensor.name) else { continue };
        let bytes = match source {
            Init::Bf16(name) => ckpt.bf16_raw(name)?,
            Init::F32(name) => f32_le(&ckpt.f32s(name)?),
            Init::Bytes(bytes) => bytes.clone(),
        };
        if bytes.len() as u64 != tensor.bytes {
            return Err(format!("{}: {} bytes, the encoder expects {}", tensor.name, bytes.len(), tensor.bytes));
        }
        tensor.init = Some(bytes);
    }
    let (_, sequence) = programs.last_key_value().ok_or("encoder has no capacity")?;
    // The pipeline's input/output are the largest capacity's.
    prefix.programs = sequence.clone();
    // Flat shapes over the largest rung's (shared, max-sized) tensors.
    prefix.input_shape = vec![prefix.model.tensors[prefix.input as usize].bytes / 4];
    let output_shape = vec![prefix.model.tensors[prefix.output as usize].bytes / 4];
    let section = prefix.forward_pipeline_section(PIPELINE, PipelineDType::F32, PipelineDType::F32, output_shape)?;
    let mut metadata: PacketPipelines = serde_json::from_slice(&section.data).map_err(|e| e.to_string())?;
    let pipeline = metadata.pipelines.first_mut().ok_or("encoder pipeline missing")?;
    for (capacity, programs) in &programs {
        for (stage, &program) in programs.iter().enumerate() {
            pipeline.programs.insert(format!("forward.{capacity}.{stage}"), program as u32);
        }
    }
    pipeline.strings.insert("modality".into(), modality.into());
    extra(pipeline, &prefix)?;
    let mut section = section;
    section.data = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
    Ok(Sidecar { model: prefix.model, section })
}

fn clip(ckpt: &Ckpt, on: bool, linear: &str, side: &str) -> Result<Option<Clip>, String> {
    if !on {
        return Ok(None);
    }
    Ok(Some(Clip { lo: ckpt.scalar(&format!("{linear}.{side}_min"))?, hi: ckpt.scalar(&format!("{linear}.{side}_max"))? }))
}

/// One vision rung: `images` images of `max_soft_tokens * pool^2` patch rows each.
fn lower_vision_capacity(seq: &mut Seq, v: &VisionTower, ckpt: &Ckpt, images: u32) -> Result<(), String> {
    let item = v.max_soft_tokens * v.pool * v.pool;
    let rows = images * item;
    let out_rows = images * v.max_soft_tokens;
    let (h, hd, nh) = (v.hidden, v.head_dim, v.heads);
    let pv = 3 * v.patch * v.patch;
    seq.tag = rows;
    let px = seq.t("in.mm.v.pixels", u64::from(rows * pv));
    let posx = seq.p().declare("in.mm.v.posx", u64::from(rows) * 4);
    let posy = seq.p().declare("in.mm.v.posy", u64::from(rows) * 4);
    let rope = seq.p().declare("in.mm.v.rope", u64::from(rows) * 8);
    let valid = seq.p().declare("in.mm.v.valid", u64::from(images) * 4);
    let pool_index: Vec<u32> =
        (0..v.pool * v.pool).map(|j| seq.p().declare(&format!("in.mm.v.pool.{j}"), u64::from(out_rows) * 4)).collect();
    let pre = &v.prefix;
    let mut x = seq.dense(px, "act.mm.v.h", rows, pv, h, &format!("{pre}patch_embedder.input_proj.weight"), None)?;
    // Position table halves as f32 [pos_size][hidden] (x then y).
    let table_name = format!("{pre}patch_embedder.position_embedding_table");
    let table = ckpt.f32s(&table_name)?;
    let half = (v.pos_size * h) as usize;
    if table.len() != 2 * half {
        return Err("vision position table shape".into());
    }
    let tx = seq.constant(&format!("{table_name}.x"), table[..half].to_vec());
    let ty = seq.constant(&format!("{table_name}.y"), table[half..].to_vec());
    let pos = seq.t("act.mm.v.pos", u64::from(rows * h));
    seq.gather(pos, tx, v.pos_size, posx, rows, h, false)?;
    seq.round(pos, rows, h)?;
    seq.gather(pos, ty, v.pos_size, posy, rows, h, true)?;
    seq.round(pos, rows, h)?;
    x = seq.add(x, pos, rows, h)?;
    seq.cut();
    for l in 0..v.layers {
        let lp = format!("{pre}encoder.layers.{l}.");
        let attn = format!("{lp}self_attn.");
        let n1 = seq.rms(x, Some("act.mm.v.n1"), rows, 1, h, Some(&format!("{lp}input_layernorm.weight")), v.eps)?;
        let mut qkv = [0u32; 3];
        for (i, name) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            let linear = format!("{attn}{name}");
            qkv[i] = seq.clipped(
                n1,
                &format!("act.mm.v.{name}"),
                rows,
                h,
                nh * hd,
                &format!("{linear}.linear.weight"),
                clip(ckpt, v.clipped, &linear, "input")?,
                clip(ckpt, v.clipped, &linear, "output")?,
            )?;
        }
        seq.rms(qkv[0], None, rows, nh, hd, Some(&format!("{attn}q_norm.weight")), v.eps)?;
        seq.rms(qkv[1], None, rows, nh, hd, Some(&format!("{attn}k_norm.weight")), v.eps)?;
        seq.rms(qkv[2], None, rows, nh, hd, None, v.eps)?;
        for &t in &qkv[..2] {
            let deps = seq.deps();
            let e = seq.p().raw(DevOp::RopeAxialF32, u64::from(rows * nh * hd / 2).div_ceil(2048), &deps, t, |d| {
                d.t[..2].copy_from_slice(&[t, rope]);
                d.i[..7].copy_from_slice(&[rows, nh, hd, 0, 2, 0, 1]);
                d.f[0] = v.theta;
            })?;
            seq.done(e);
        }
        let deps = seq.deps();
        let e = seq.p().attention_f32(
            qkv[0],
            qkv[1],
            qkv[2],
            &deps,
            AttentionF32Stage {
                output: TensorRef::Named("act.mm.v.attn"),
                key_lengths: Some(valid),
                bias: None,
                batch: images,
                q_rows: item,
                kv_rows: item,
                heads: nh,
                head_width: hd,
                in_stride: 0,
                k_col0: 0,
                v_col0: 0,
                causal: false,
                scale: 1.0,
                bias_head_stride: 0,
                relative: false,
                key_length_heads: 0,
                prefix: None,
            },
        )?;
        let a = seq.done(e);
        seq.round(a, rows, h)?;
        let linear = format!("{attn}o_proj");
        let o = seq.clipped(
            a,
            "act.mm.v.o",
            rows,
            nh * hd,
            h,
            &format!("{linear}.linear.weight"),
            clip(ckpt, v.clipped, &linear, "input")?,
            clip(ckpt, v.clipped, &linear, "output")?,
        )?;
        seq.rms(o, None, rows, 1, h, Some(&format!("{lp}post_attention_layernorm.weight")), v.eps)?;
        x = seq.add(x, o, rows, h)?;
        let n2 = seq.rms(x, Some("act.mm.v.n2"), rows, 1, h, Some(&format!("{lp}pre_feedforward_layernorm.weight")), v.eps)?;
        let mlp = format!("{lp}mlp.");
        let mut gu = [0u32; 2];
        for (i, name) in ["gate_proj", "up_proj"].iter().enumerate() {
            let linear = format!("{mlp}{name}");
            gu[i] = seq.clipped(
                n2,
                &format!("act.mm.v.{name}"),
                rows,
                h,
                v.inter,
                &format!("{linear}.linear.weight"),
                clip(ckpt, v.clipped, &linear, "input")?,
                clip(ckpt, v.clipped, &linear, "output")?,
            )?;
        }
        seq.unary(gu[0], None, rows, v.inter, ACT_GELU_TANH, 0.0, 0.0)?;
        seq.round(gu[0], rows, v.inter)?;
        let m = seq.binary(gu[0], gu[1], None, 2, [1, rows, v.inter], [0, v.inter, 1], true)?;
        let linear = format!("{mlp}down_proj");
        let d = seq.clipped(
            m,
            "act.mm.v.down",
            rows,
            v.inter,
            h,
            &format!("{linear}.linear.weight"),
            clip(ckpt, v.clipped, &linear, "input")?,
            clip(ckpt, v.clipped, &linear, "output")?,
        )?;
        seq.rms(d, None, rows, 1, h, Some(&format!("{lp}post_feedforward_layernorm.weight")), v.eps)?;
        x = seq.add(x, d, rows, h)?;
        seq.cut();
    }
    // Pool k x k patches by position (index rows u32::MAX contribute nothing), then
    // `/ k^2` (bf16, as the pooler's matmul result is cast), `* sqrt(hidden)` in f32.
    let k2 = v.pool * v.pool;
    let pooled = seq.t("act.mm.v.pooled", u64::from(out_rows * h));
    for j in 0..k2 {
        let deps = seq.deps();
        let e = seq.p().gather_rows_f32(
            &deps,
            GatherRowsF32Stage {
                output: TensorRef::Handle(pooled),
                table: TensorRef::Handle(x),
                table_rows: rows,
                index: Some(pool_index[j as usize]),
                rows: out_rows,
                width: h,
                vocab: rows,
                rows_per_item: 0,
                repeat: 0,
                index_item_stride: 0,
                table_item_stride: 0,
                table_f16: false,
                accumulate: j > 0,
                out_stride: 0,
                out_col0: 0,
            },
        )?;
        seq.done(e);
    }
    seq.unary(pooled, None, out_rows, h, ACT_SCALE_SHIFT, 1.0 / k2 as f32, 0.0)?;
    seq.round(pooled, out_rows, h)?;
    seq.unary(pooled, None, out_rows, h, ACT_SCALE_SHIFT, (h as f32).sqrt(), 0.0)?;
    if v.standardize {
        let bias = seq.weight_f32(&format!("{pre}std_bias"), u64::from(h));
        let scale = seq.weight_f32(&format!("{pre}std_scale"), u64::from(h));
        seq.binary(pooled, bias, None, 1, [1, out_rows, h], [0, 0, 1], false)?;
        seq.binary(pooled, scale, None, 2, [1, out_rows, h], [0, 0, 1], false)?;
    }
    seq.round(pooled, out_rows, h)?;
    seq.rms(pooled, None, out_rows, 1, h, None, v.eps)?;
    let out = seq.dense(pooled, "act.mm.v.out", out_rows, h, v.text_hidden, &format!("{}embedding_projection.weight", v.embed), None)?;
    seq.cut();
    let prefix = seq.prefix.as_mut().unwrap();
    prefix.output = out;
    prefix.input = px;
    prefix.input_shape = vec![u64::from(rows), u64::from(pv)];
    Ok(())
}

/// Sinusoidal relative positions `[context/2 + 1][hidden]` (sin | cos), descending.
fn audio_positions(a: &AudioTower) -> Vec<f32> {
    let context = a.chunk + a.context_left - 1 + a.context_right;
    let n = context / 2 + 1;
    let half = (a.hidden / 2) as usize;
    let inc = (10000.0f32 / 1.0).ln() / ((half as f32) - 1.0).max(1.0);
    let inv: Vec<f32> = (0..half).map(|i| (-(i as f32) * inc).exp()).collect();
    let mut out = Vec::with_capacity(n as usize * a.hidden as usize);
    for p in (0..n).rev() {
        let t: Vec<f32> = inv.iter().map(|&w| p as f32 * w).collect();
        out.extend(t.iter().map(|x| round_bf16(x.sin())));
        out.extend(t.iter().map(|x| round_bf16(x.cos())));
    }
    out
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// One audio rung: `frames` log-mel frames (a multiple of 4) of one clip.
fn lower_audio_capacity(seq: &mut Seq, a: &AudioTower, ckpt: &Ckpt, frames: u32) -> Result<(), String> {
    let pre = &a.prefix;
    let (c0, c1) = (a.channels[0], a.channels[1]);
    let (f0, f1, f2) = (a.mel_bins, a.mel_bins / 2, a.mel_bins / 4);
    let (t1, t2) = (frames / 2, frames / 4);
    let h = a.hidden;
    seq.tag = t2;
    let mel = seq.t("in.mm.a.mel", u64::from(frames * f0));
    let mask1 = seq.t("in.mm.a.mask1", u64::from(t1));
    let valid = seq.p().declare("in.mm.a.valid", 4);
    // im2col indices of a 3x3 stride-2 pad-1 conv: output (t, f) tap (dt, df) reads input row
    // (2t + dt - 1, 2f + df - 1); out of range is u32::MAX (zero).
    let taps = |t_out: u32, f_out: u32, t_in: u32, f_in: u32| -> Vec<u32> {
        let mut index = Vec::with_capacity((t_out * f_out * 9) as usize);
        for t in 0..t_out {
            for fo in 0..f_out {
                for dt in 0..3i64 {
                    for df in 0..3i64 {
                        let (ti, fi) = (2 * i64::from(t) + dt - 1, 2 * i64::from(fo) + df - 1);
                        index.push(if ti < 0 || fi < 0 || ti >= i64::from(t_in) || fi >= i64::from(f_in) {
                            u32::MAX
                        } else {
                            (ti * i64::from(f_in) + fi) as u32
                        });
                    }
                }
            }
        }
        index
    };
    let idx0 = seq.constant_u32(&format!("const.mm.a.im2col0.{frames}"), &taps(t1, f1, frames, f0));
    let idx1 = seq.constant_u32(&format!("const.mm.a.im2col1.{frames}"), &taps(t2, f2, t1, f1));
    // Conv weights `[out][in][3][3]` as `[out][(dt*3 + df)*in + i]`, bf16 values in f32.
    let conv_weight = |name: &str, cout: u32, cin: u32| -> Result<Vec<f32>, String> {
        let w = ckpt.f32s(name)?;
        let mut out = vec![0f32; (cout * cin * 9) as usize];
        for o in 0..cout {
            for i in 0..cin {
                for k in 0..9 {
                    out[(o * cin * 9 + k * cin + i) as usize] = w[(o * cin * 9 + i * 9 + k) as usize];
                }
            }
        }
        Ok(out)
    };
    let sub = format!("{pre}subsample_conv_projection.");
    let w0 = seq.constant(&format!("{sub}layer0.conv.weight.taps"), conv_weight(&format!("{sub}layer0.conv.weight"), c0, 1)?);
    let w1 = seq.constant(&format!("{sub}layer1.conv.weight.taps"), conv_weight(&format!("{sub}layer1.conv.weight"), c1, c0)?);
    let col0 = seq.t("act.mm.a.col0", u64::from(t1 * f1 * 9));
    seq.gather(col0, mel, frames * f0, idx0, t1 * f1 * 9, 1, false)?;
    let y0 = seq.dense_f32w(col0, "act.mm.a.conv0", t1 * f1, 9, c0, w0, true)?;
    let y0 = seq.layer_norm(y0, "act.mm.a.ln0", t1 * f1, c0, Some(&format!("{sub}layer0.norm.weight")), None, a.eps, ACT_RELU)?;
    seq.binary(y0, mask1, None, 2, [t1, f1, c0], [1, 0, 0], false)?;
    let col1 = seq.t("act.mm.a.col1", u64::from(t2 * f2 * 9 * c0));
    seq.gather(col1, y0, t1 * f1, idx1, t2 * f2 * 9, c0, false)?;
    let y1 = seq.dense_f32w(col1, "act.mm.a.conv1", t2 * f2, 9 * c0, c1, w1, true)?;
    let y1 = seq.layer_norm(y1, "act.mm.a.ln1", t2 * f2, c1, Some(&format!("{sub}layer1.norm.weight")), None, a.eps, ACT_RELU)?;
    let mut x = seq.dense(y1, "act.mm.a.x", t2, f2 * c1, h, &format!("{sub}input_proj_linear.weight"), None)?;
    seq.cut();
    let positions = audio_positions(a);
    let npos = (positions.len() / h as usize) as u32;
    let dh = h / a.heads;
    let q_scale = (dh as f32).powf(-0.5) / std::f32::consts::LN_2;
    let k_scale = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
    let ffw = |seq: &mut Seq, x: u32, block: &str| -> Result<u32, String> {
        let n = seq.rms(x, Some("act.mm.a.ffn_in"), t2, 1, h, Some(&format!("{block}pre_layer_norm.weight")), a.eps)?;
        let l1 = format!("{block}ffw_layer_1");
        let y = seq.clipped(
            n,
            "act.mm.a.ffn_mid",
            t2,
            h,
            4 * h,
            &format!("{l1}.linear.weight"),
            clip(ckpt, a.clipped, &l1, "input")?,
            clip(ckpt, a.clipped, &l1, "output")?,
        )?;
        seq.unary(y, None, t2, 4 * h, ACT_SILU, 0.0, 0.0)?;
        seq.round(y, t2, 4 * h)?;
        let l2 = format!("{block}ffw_layer_2");
        let y = seq.clipped(
            y,
            "act.mm.a.ffn_out",
            t2,
            4 * h,
            h,
            &format!("{l2}.linear.weight"),
            clip(ckpt, a.clipped, &l2, "input")?,
            clip(ckpt, a.clipped, &l2, "output")?,
        )?;
        seq.rms(y, None, t2, 1, h, Some(&format!("{block}post_layer_norm.weight")), a.eps)?;
        seq.unary(y, None, t2, h, ACT_SCALE_SHIFT, a.residual_weight, 0.0)?;
        seq.round(y, t2, h)?;
        seq.add(x, y, t2, h)
    };
    for l in 0..a.layers {
        let lp = format!("{pre}layers.{l}.");
        x = ffw(seq, x, &format!("{lp}feed_forward1."))?;
        let attn = format!("{lp}self_attn.");
        let n = seq.rms(x, Some("act.mm.a.attn_in"), t2, 1, h, Some(&format!("{lp}norm_pre_attn.weight")), a.eps)?;
        let mut qkv = [0u32; 3];
        for (i, name) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            let linear = format!("{attn}{name}");
            qkv[i] = seq.clipped(
                n,
                &format!("act.mm.a.{name}"),
                t2,
                h,
                h,
                &format!("{linear}.linear.weight"),
                clip(ckpt, a.clipped, &linear, "input")?,
                clip(ckpt, a.clipped, &linear, "output")?,
            )?;
        }
        // Relative keys: relative_k_proj(positions) in bf16, a per-layer constant.
        let wr = ckpt.f32s(&format!("{attn}relative_k_proj.weight"))?;
        let mut relk = vec![0f32; (npos * h) as usize];
        for p in 0..npos as usize {
            for o in 0..h as usize {
                let mut s = 0f32;
                for i in 0..h as usize {
                    s += positions[p * h as usize + i] * wr[o * h as usize + i];
                }
                relk[p * h as usize + o] = round_bf16(s);
            }
        }
        let relk = seq.constant(&format!("{attn}relative_k.f32"), relk);
        let pds = ckpt.f32s(&format!("{attn}per_dim_scale"))?;
        let qs: Vec<f32> = pds.iter().map(|&p| q_scale * round_bf16(softplus(p))).collect();
        let qs = seq.constant(&format!("{attn}q_scale.f32"), qs);
        let out = seq.t("act.mm.a.attn", u64::from(t2 * h));
        let deps = seq.deps();
        let window = (a.context_left - 1) | (a.context_right << 16);
        let e = seq.p().raw(DevOp::ChunkAttentionF32, u64::from(t2.div_ceil(a.chunk) * a.heads), &deps, out, |d| {
            d.t[..7].copy_from_slice(&[out, qkv[0], qkv[1], qkv[2], relk, qs, valid]);
            d.i = [t2, a.heads, dh, a.chunk, a.context_left - 1, a.context_right, npos, window];
            d.f = [k_scale, a.softcap];
        })?;
        seq.done(e);
        seq.round(out, t2, h)?;
        let linear = format!("{attn}post");
        let o = seq.clipped(
            out,
            "act.mm.a.post",
            t2,
            h,
            h,
            &format!("{linear}.linear.weight"),
            clip(ckpt, a.clipped, &linear, "input")?,
            clip(ckpt, a.clipped, &linear, "output")?,
        )?;
        seq.rms(o, None, t2, 1, h, Some(&format!("{lp}norm_post_attn.weight")), a.eps)?;
        x = seq.add(x, o, t2, h)?;
        // Light conv: norm, linear_start, GLU, causal depthwise conv, norm, SiLU, linear_end.
        let lc = format!("{lp}lconv1d.");
        let n = seq.rms(x, Some("act.mm.a.lc_in"), t2, 1, h, Some(&format!("{lc}pre_layer_norm.weight")), a.eps)?;
        let linear = format!("{lc}linear_start");
        let y = seq.clipped(
            n,
            "act.mm.a.lc_start",
            t2,
            h,
            2 * h,
            &format!("{linear}.linear.weight"),
            clip(ckpt, a.clipped, &linear, "input")?,
            clip(ckpt, a.clipped, &linear, "output")?,
        )?;
        let g = seq.t("act.mm.a.lc_glu", u64::from(t2 * h));
        let deps = seq.deps();
        let e = seq.p().raw(DevOp::GluF32, u64::from(t2 * h).div_ceil(2048), &deps, g, |d| {
            d.t[..2].copy_from_slice(&[g, y]);
            d.i[..2].copy_from_slice(&[t2, h]);
        })?;
        seq.done(e);
        seq.round(g, t2, h)?;
        // Causal depthwise conv as shifted row gathers times per-channel taps, accumulated in f32
        // and rounded once: y[t] = sum_k w[:, k] * g[t - (K - 1) + k]. (The speech Conv1d arm
        // would add ~300k PTX lines to the speech object for one op.)
        let wk = ckpt.f32s(&format!("{lc}depthwise_conv1d.weight"))?;
        let kn = a.conv_kernel;
        let cv = seq.t("act.mm.a.lc_conv", u64::from(t2 * h));
        let tap = seq.t("act.mm.a.lc_tap", u64::from(t2 * h));
        for k in 0..kn {
            let shift = kn - 1 - k;
            let index: Vec<u32> = (0..t2).map(|t| if t >= shift { t - shift } else { u32::MAX }).collect();
            let index = seq.constant_u32(&format!("const.mm.a.shift{shift}.{t2}"), &index);
            let w: Vec<f32> = (0..h).map(|c| wk[(c * kn + k) as usize]).collect();
            let w = seq.constant(&format!("{lc}depthwise_conv1d.tap{k}"), w);
            let dst = if k == 0 { cv } else { tap };
            seq.gather(dst, g, t2, index, t2, h, false)?;
            seq.binary(dst, w, None, 2, [1, t2, h], [0, 0, 1], false)?;
            if k > 0 {
                seq.binary(cv, tap, None, 0, [1, t2, h], [0, h, 1], false)?;
            }
        }
        seq.round(cv, t2, h)?;
        seq.rms(cv, None, t2, 1, h, Some(&format!("{lc}conv_norm.weight")), a.eps)?;
        seq.unary(cv, None, t2, h, ACT_SILU, 0.0, 0.0)?;
        seq.round(cv, t2, h)?;
        let linear = format!("{lc}linear_end");
        let y = seq.clipped(
            cv,
            "act.mm.a.lc_end",
            t2,
            h,
            h,
            &format!("{linear}.linear.weight"),
            clip(ckpt, a.clipped, &linear, "input")?,
            clip(ckpt, a.clipped, &linear, "output")?,
        )?;
        x = seq.add(x, y, t2, h)?;
        x = ffw(seq, x, &format!("{lp}feed_forward2."))?;
        seq.rms(x, None, t2, 1, h, Some(&format!("{lp}norm_out.weight")), a.eps)?;
        seq.cut();
    }
    let y = seq.dense(x, "act.mm.a.proj", t2, h, a.output_dims, &format!("{pre}output_proj.weight"), Some(&format!("{pre}output_proj.bias")))?;
    seq.rms(y, None, t2, 1, a.output_dims, None, a.eps)?;
    let out = seq.dense(y, "act.mm.a.out", t2, a.output_dims, a.text_hidden, &format!("{}embedding_projection.weight", a.embed), None)?;
    seq.cut();
    let prefix = seq.prefix.as_mut().unwrap();
    prefix.output = out;
    prefix.input = mel;
    prefix.input_shape = vec![u64::from(frames), u64::from(f0)];
    Ok(())
}

/// The `plow.multimodal.v1` LM section.
pub fn contract_section(contract: &MmContract) -> Result<packet::devbuild::SectionData, String> {
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: plow_asset::multimodal::SECTION.into(),
        data: serde_json::to_vec(contract).map_err(|e| e.to_string())?,
    })
}

/// Vision rungs (images per launch) and audio rungs (mel frames), from the emit knobs.
pub fn ladders() -> (Vec<u32>, Vec<u32>) {
    let parse = |s: Option<&str>, default: &[u32]| -> Vec<u32> {
        s.map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).filter(|&v| v > 0).collect())
            .filter(|v: &Vec<u32>| !v.is_empty())
            .unwrap_or_else(|| default.to_vec())
    };
    let cfg = crate::emit_config::active();
    (parse(cfg.mm_vision_ladder.as_deref(), &[1, 2]), parse(cfg.mm_audio_ladder.as_deref(), &[400, 1000, 2000, 3000]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_positions_match_the_reference_layout() {
        let a = AudioTower {
            hidden: 8,
            layers: 1,
            heads: 1,
            chunk: 12,
            context_left: 13,
            context_right: 0,
            conv_kernel: 5,
            channels: [4, 2],
            output_dims: 8,
            softcap: 50.0,
            residual_weight: 0.5,
            eps: 1e-6,
            clipped: false,
            text_hidden: 8,
            mel_bins: 8,
            prefix: String::new(),
            embed: String::new(),
        };
        let p = audio_positions(&a);
        assert_eq!(p.len(), 13 * 8);
        // First row is position 12: sin(12) then cos(12) at the slowest timescale last.
        assert!((p[0] - round_bf16(12f32.sin())).abs() < 1e-6);
        assert!((p[4] - round_bf16(12f32.cos())).abs() < 1e-6);
        // Last row is position 0: sin 0, cos 1.
        assert_eq!(p[12 * 8], 0.0);
        assert_eq!(p[12 * 8 + 4], 1.0);
    }
}
