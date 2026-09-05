//! Network definitions: HuggingFace configs → symbolic operator graphs.
//!
//! Each architecture builder (gemma, deepseek, siglip, qwen_vl, qwen_image_*)
//! consumes a typed config (see [`config`]) and emits a [`crate::Graph`] over
//! the IR. This is the model zoo layer of `nn-graph`; it is gated behind the
//! `models` feature so the pure IR can be used without `serde`. Resolving a
//! model *identifier* over the network lives downstream (the `hub` layer), not
//! here.

pub mod config;

mod deepseek;
mod gemma;
mod gemma4_multimodal;
mod glm;
mod kimi;
mod kimi_k3;
mod llama;
mod qwen3;
mod qwen3_5;
mod qwen_image_dit;
mod qwen_image_vae;
mod qwen_vl;
mod siglip;

pub use config::{ConfigError, ModelConfig};

use crate::{infer_shapes, Graph, InferError};

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Infer(#[from] InferError),
    #[error("architecture does not support encoder build: {0}")]
    EncoderUnsupported(&'static str),
}

/// Compile-time shape bucket. Text models keep batch/sequence symbolic and
/// ignore this; diffusion models (Qwen-Image VAE/DiT) are specialized to a
/// concrete image resolution — compiling at a different resolution produces a
/// different graph, as the design's shape-bucket scheme intends.
#[derive(Clone, Copy, Debug)]
pub struct ShapeBucket {
    pub image_height: i64,
    pub image_width: i64,
}

impl ShapeBucket {
    pub fn square(size: i64) -> ShapeBucket {
        ShapeBucket {
            image_height: size,
            image_width: size,
        }
    }
}

impl Default for ShapeBucket {
    fn default() -> Self {
        ShapeBucket::square(1024)
    }
}

/// Build a shape-inferred operator graph from a parsed config, specialized to
/// `bucket` (used only by the resolution-dependent diffusion models).
pub fn build_graph(cfg: &ModelConfig, bucket: &ShapeBucket) -> Result<Graph, BuildError> {
    let mut graph = match cfg {
        ModelConfig::Gemma(c) => gemma::build(c),
        ModelConfig::Gemma4Multimodal(c) => gemma4_multimodal::build(c),
        ModelConfig::Glm(c) => glm::build(c),
        ModelConfig::Kimi(c) => kimi::build(c),
        ModelConfig::KimiK3(c) => kimi_k3::build(c)?,
        ModelConfig::Llama(c) => llama::build(c),
        ModelConfig::Qwen3(c) => qwen3::build(c),
        ModelConfig::Qwen35(c) => qwen3_5::build(c),
        ModelConfig::DeepSeek(c) => deepseek::build(c),
        ModelConfig::Siglip(c) => siglip::build(c),
        ModelConfig::QwenVl(c) => qwen_vl::build(c),
        ModelConfig::QwenImageDit(c) => qwen_image_dit::build(c, bucket),
        ModelConfig::QwenImageVae(c) => qwen_image_vae::build(c, bucket),
    };
    infer_shapes(&mut graph)?;
    Ok(graph)
}

/// Build a text model as an ENCODER: no lm_head, the final hidden states are
/// the last graph output, with optional intermediate residual-stream taps
/// (`taps` = layer indices, marked as outputs in order before the final
/// hidden states). FLUX.2 taps three encoder layers; Z-Image uses `&[]`.
pub fn build_encoder_graph(cfg: &ModelConfig, taps: &[u32]) -> Result<Graph, BuildError> {
    let mut graph = match cfg {
        ModelConfig::Llama(c) => llama::build_encoder(c, taps),
        ModelConfig::Qwen3(c) => qwen3::build_encoder(c, taps),
        other => {
            return Err(BuildError::EncoderUnsupported(match other {
                ModelConfig::Gemma(_) => "gemma",
                ModelConfig::Gemma4Multimodal(_) => "gemma4_multimodal",
                ModelConfig::Glm(_) => "glm",
                ModelConfig::Kimi(_) => "kimi",
                ModelConfig::KimiK3(_) => "kimi_k3",
                ModelConfig::DeepSeek(_) => "deepseek",
                ModelConfig::Siglip(_) => "siglip",
                ModelConfig::QwenVl(_) => "qwen_vl",
                ModelConfig::QwenImageDit(_) => "qwen_image_dit",
                ModelConfig::QwenImageVae(_) => "qwen_image_vae",
                ModelConfig::Qwen35(_) => "qwen3_5",
                ModelConfig::Llama(_) | ModelConfig::Qwen3(_) => unreachable!(),
            }))
        }
    };
    infer_shapes(&mut graph)?;
    Ok(graph)
}

/// Parse a `config.json` string and build the graph at the default resolution
/// bucket.
pub fn build_from_config_json(json: &str) -> Result<Graph, BuildError> {
    build_from_config_json_at(json, &ShapeBucket::default())
}

/// Parse a `config.json` string and build the graph specialized to `bucket`.
pub fn build_from_config_json_at(json: &str, bucket: &ShapeBucket) -> Result<Graph, BuildError> {
    let cfg = ModelConfig::from_json(json)?;
    build_graph(&cfg, bucket)
}

/// Build the complete graph executed by a text-generation endpoint. A multimodal checkpoint may
/// wrap that network in `text_config`; selecting it is endpoint semantics, not an architecture
/// special case. Plain text checkpoints pass through unchanged.
pub fn build_text_generation_from_config_json_at(
    json: &str,
    bucket: &ShapeBucket,
) -> Result<Graph, BuildError> {
    let mut outer: serde_json::Value = serde_json::from_str(json).map_err(ConfigError::from)?;
    let wrapper_type = outer
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let language_model_wrapper = matches!(
        wrapper_type.as_str(),
        "qwen3_5" | "gemma4" | "gemma4_unified" | "kimi_k3"
    );
    if let Some(mut text) = outer.get_mut("text_config").map(serde_json::Value::take) {
        if let serde_json::Value::Object(fields) = &mut text {
            if !fields.contains_key("model_type") && !fields.contains_key("architectures") {
                let text_type = match wrapper_type.as_str() {
                    "gemma3" => Some("gemma3_text"),
                    "gemma4" | "gemma4_unified" => Some("gemma4_text"),
                    "qwen3_5" => Some("qwen3_5_text"),
                    "kimi_k3" => Some("kimi_linear"),
                    _ => None,
                };
                if let Some(text_type) = text_type {
                    fields.insert("model_type".into(), text_type.into());
                }
            }
            if !fields.contains_key("dtype") && !fields.contains_key("torch_dtype") {
                for key in ["dtype", "torch_dtype"] {
                    if let Some(dtype) = outer.get(key) {
                        fields.insert(key.into(), dtype.clone());
                        break;
                    }
                }
            }
            if !fields.contains_key("quantization_config") {
                if let Some(quant) = outer.get("quantization_config") {
                    fields.insert("quantization_config".into(), quant.clone());
                }
            }
            if language_model_wrapper {
                let weight_prefix = if wrapper_type == "kimi_k3" {
                    "language_model.model"
                } else {
                    "model.language_model"
                };
                fields.insert(
                    "_plow_weight_prefix".to_string(),
                    serde_json::Value::String(weight_prefix.to_string()),
                );
                if wrapper_type == "kimi_k3" {
                    fields.insert(
                        "_plow_head_prefix".to_string(),
                        serde_json::Value::String("language_model".to_string()),
                    );
                }
            }
        }
        outer = text;
    }
    let cfg = ModelConfig::from_json(&outer.to_string())?;
    build_graph(&cfg, bucket)
}

/// One sub-network of a multi-network pipeline checkpoint.
#[derive(Debug, Clone)]
pub struct PipelineNetwork {
    /// Component name (diffusers directory name: "text_encoder",
    /// "transformer", "vae", ...).
    pub name: String,
    pub config: ModelConfig,
    /// `Some(taps)` builds this network as a text encoder (see
    /// [`build_encoder_graph`]); `None` is the standard build.
    pub encoder_taps: Option<Vec<u32>>,
}

/// A diffusers-style multi-network model (e.g. FLUX / Z-Image: text encoder +
/// DiT + VAE). Each sub-network is built as its OWN graph — the downstream
/// compiler emits an independent packet-program set per network, and the
/// runtime pipeline sequences them.
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    pub networks: Vec<PipelineNetwork>,
}

impl PipelineConfig {
    /// Parse named `config.json` strings (one per component). Networks
    /// default to the standard build; set `encoder_taps` on entries that
    /// should build as encoders.
    pub fn from_json_parts<'a>(
        parts: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<PipelineConfig, ConfigError> {
        let networks = parts
            .into_iter()
            .map(|(name, json)| {
                Ok(PipelineNetwork {
                    name: name.to_string(),
                    config: ModelConfig::from_json(json)?,
                    encoder_taps: None,
                })
            })
            .collect::<Result<Vec<_>, ConfigError>>()?;
        Ok(PipelineConfig { networks })
    }

    /// Build every sub-network into its own shape-inferred graph, in
    /// declaration order.
    pub fn build_all(&self, bucket: &ShapeBucket) -> Result<Vec<(String, Graph)>, BuildError> {
        self.networks
            .iter()
            .map(|n| {
                let g = match &n.encoder_taps {
                    Some(taps) => build_encoder_graph(&n.config, taps)?,
                    None => build_graph(&n.config, bucket)?,
                };
                Ok((n.name.clone(), g))
            })
            .collect()
    }
}
