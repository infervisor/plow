//! HuggingFace `config.json` parsing and architecture detection.
//!
//! Configs are parsed leniently: we read into `serde_json::Value`, detect the
//! architecture from `model_type` / `architectures`, descend into the relevant
//! sub-config (`text_config` / `vision_config`) for multimodal checkpoints, and
//! then deserialize into a typed per-architecture struct. Unknown fields are
//! ignored and missing fields fall back to architecture defaults.

mod deepseek;
mod gemma;
mod gemma4_multimodal;
mod glm;
mod kimi;
mod kimi_k3;
mod llama;
mod qwen3;
mod qwen_image_dit;
mod qwen_image_vae;
mod qwen_vl;
mod siglip;

pub use deepseek::DeepSeekConfig;
pub use gemma::{GemmaConfig, RopeParameters, RopeSpec};
pub use gemma4_multimodal::{Gemma4MultimodalConfig, MmProjectorConfig};
pub use glm::GlmConfig;
pub use kimi::KimiConfig;
pub use kimi_k3::{K3Layer, KimiK3Config, KimiK3TextConfig, LinearAttnConfig};
pub use llama::LlamaConfig;
pub use qwen3::Qwen3Config;
pub use qwen_image_dit::QwenImageDitConfig;
pub use qwen_image_vae::QwenImageVaeConfig;
pub use qwen_vl::QwenVlVisionConfig;
pub use siglip::SiglipConfig;

use crate::DType;

#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("could not determine architecture (no recognized model_type/architectures)")]
    UnknownArch,
    #[error("unsupported architecture: {0}")]
    Unsupported(String),
}

/// Parsed, typed model configuration, tagged by family.
#[derive(Debug, Clone)]
pub enum ModelConfig {
    Gemma(GemmaConfig),
    /// Gemma 4 multimodal: SigLIP vision encoder + projector + text decoder.
    Gemma4Multimodal(Gemma4MultimodalConfig),
    /// GLM-5 MoE-DSA: MLA + MoE + Dense-Sparse Attention.
    Glm(GlmConfig),
    /// Kimi K2 (Moonshot): MLA + MoE.
    Kimi(KimiConfig),
    /// Kimi K3 (Moonshot): hybrid MLA/KDA + AttnRes block residual + latent MoE.
    KimiK3(KimiK3Config),
    Llama(LlamaConfig),
    /// Qwen3 / Qwen2.5: GQA + SwiGLU (dense).
    Qwen3(Qwen3Config),
    DeepSeek(DeepSeekConfig),
    Siglip(SiglipConfig),
    QwenVl(QwenVlVisionConfig),
    QwenImageDit(QwenImageDitConfig),
    QwenImageVae(QwenImageVaeConfig),
}

impl ModelConfig {
    pub fn from_json(s: &str) -> Result<ModelConfig, ConfigError> {
        let v: serde_json::Value = serde_json::from_str(s)?;

        // Diffusers component configs identify by `_class_name`, not model_type.
        if let Some(cls) = v.get("_class_name").and_then(|c| c.as_str()) {
            match cls {
                "QwenImageTransformer2DModel" => {
                    return Ok(ModelConfig::QwenImageDit(serde_json::from_value(v)?))
                }
                "AutoencoderKLQwenImage" => {
                    return Ok(ModelConfig::QwenImageVae(serde_json::from_value(v)?))
                }
                _ => {}
            }
        }

        let mt = model_type(&v).ok_or(ConfigError::UnknownArch)?;
        match mt.as_str() {
            // Only Gemma 3/4 are supported (the 4-norm residual block layout).
            // Gemma 1/2 have a different 2-norm layout and are not modeled.
            "gemma" | "gemma2" => Err(ConfigError::Unsupported(format!(
                "{} (only Gemma 3/4 are supported)",
                mt
            ))),
            "gemma3" | "gemma3_text" => {
                // Gemma3 multimodal nests the decoder under `text_config`.
                let sub = sub_config(&v, "text_config");
                Ok(ModelConfig::Gemma(serde_json::from_value(sub)?))
            }
            "gemma4" | "gemma4_text" | "gemma4_unified_text" => {
                // Text-only Gemma 4 (or already inside a text_config).
                let sub = sub_config(&v, "text_config");
                Ok(ModelConfig::Gemma(serde_json::from_value(sub)?))
            }
            "gemma4_unified" => {
                // Gemma4 multimodal top-level: if vision_config is present,
                // build the full multimodal composite; else text-only.
                if v.get("vision_config").is_some() {
                    Ok(ModelConfig::Gemma4Multimodal(serde_json::from_value(v)?))
                } else {
                    let sub = sub_config(&v, "text_config");
                    Ok(ModelConfig::Gemma(serde_json::from_value(sub)?))
                }
            }
            "llama" | "mistral" | "codellama" => Ok(ModelConfig::Llama(serde_json::from_value(v)?)),
            "kimi" | "moonshot" => Ok(ModelConfig::Kimi(serde_json::from_value(v)?)),
            // Kimi-K3. Claimed BY NAME rather than falling through to the
            // generic `other` arm — the `architectures` fallback below maps any
            // `KimiLinear*`/`KimiK3*` prefix here, and before this arm existed
            // it landed on `"kimi"`, i.e. K3 silently built as a 61-layer K2.
            "kimi_k3" => Ok(ModelConfig::KimiK3(serde_json::from_value(v)?)),
            // The INNER text tower on its own. Wrapped rather than parsed
            // directly so there is exactly one `KimiK3Config` shape to reason
            // about, and so `vision_config` is unambiguously absent (a bare
            // `kimi_linear` document is the text tower by definition).
            "kimi_linear" => Ok(ModelConfig::KimiK3(KimiK3Config {
                text_config: serde_json::from_value(v)?,
                vision_config: None,
            })),
            "qwen3" | "qwen2_5" => Ok(ModelConfig::Qwen3(serde_json::from_value(v)?)),
            "deepseek" | "deepseek_v2" | "deepseek_v3" => {
                Ok(ModelConfig::DeepSeek(serde_json::from_value(v)?))
            }
            "glm_moe_dsa" | "glm" | "glm4" => Ok(ModelConfig::Glm(serde_json::from_value(v)?)),
            "siglip" | "siglip_vision_model" => {
                let sub = sub_config(&v, "vision_config");
                Ok(ModelConfig::Siglip(serde_json::from_value(sub)?))
            }
            // Only Qwen2.5-VL is supported (gated SiLU MLP). Qwen2-VL uses a
            // different non-gated MLP architecture (fc1→QuickGELU→fc2).
            "qwen2_vl" | "qwen2vl" => Err(ConfigError::Unsupported(format!(
                "{} (only Qwen2.5-VL is supported)",
                mt
            ))),
            "qwen2_5_vl" => {
                let sub = sub_config(&v, "vision_config");
                Ok(ModelConfig::QwenVl(serde_json::from_value(sub)?))
            }
            other => Err(ConfigError::Unsupported(other.to_string())),
        }
    }
}

/// Determine the model_type, falling back to mapping known `architectures`.
fn model_type(v: &serde_json::Value) -> Option<String> {
    if let Some(mt) = v.get("model_type").and_then(|m| m.as_str()) {
        return Some(mt.to_string());
    }
    let arch = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|a| a.as_str())?;
    let mapped = match arch {
        a if a.starts_with("Gemma") => "gemma3",
        a if a.starts_with("DeepseekV3") || a.starts_with("DeepseekV2") => "deepseek_v3",
        a if a.starts_with("GlmMoeDsa") || a.starts_with("Glm") => "glm_moe_dsa",
        // `KimiLinear*` is Kimi-K3, NOT K2, and the two share almost no
        // geometry: K3 runs KDA (linear attention) on 69 of its 93 layers, a
        // latent MoE, and the AttnRes block residual. Mapping it to "kimi"
        // would hand a K3 checkpoint to the K2 MLA+MoE builder, and because
        // `KimiConfig` used to be `#[serde(default)]` end to end that build
        // SUCCEEDED — emitting a plausible 61-layer K2 graph for a 93-layer K3
        // model. Claim the prefix by name and reject it; K3 is emitted by
        // `devgen`, which does not go through this crate at all.
        a if a.starts_with("KimiLinear") || a.starts_with("KimiK3") => "kimi_k3",
        a if a.starts_with("Kimi") || a.starts_with("Moonshot") => "kimi",
        a if a.starts_with("Qwen3") => "qwen3",
        a if a.starts_with("Qwen2.5") => "qwen2_5",
        a if a.starts_with("Siglip") => "siglip",
        a if a.starts_with("Qwen2_5_VL") => "qwen2_5_vl",
        a if a.starts_with("Qwen2VL") => "qwen2_vl",
        _ => return None,
    };
    Some(mapped.to_string())
}

/// Return `v[key]` if present (a nested sub-config), else `v` itself. Newer
/// multimodal configs put `dtype` at the top level, so inherit it into the
/// sub-config when the sub-config doesn't carry its own.
fn sub_config(v: &serde_json::Value, key: &str) -> serde_json::Value {
    let mut sub = v.get(key).cloned().unwrap_or_else(|| v.clone());
    if let serde_json::Value::Object(map) = &mut sub {
        if !map.contains_key("dtype") && !map.contains_key("torch_dtype") {
            for k in ["dtype", "torch_dtype"] {
                if let Some(dt) = v.get(k) {
                    map.insert("torch_dtype".to_string(), dt.clone());
                    break;
                }
            }
        }
    }
    sub
}

/// Parse a torch dtype string into our [`DType`]; defaults to bf16.
pub fn parse_dtype(s: Option<&str>) -> DType {
    match s {
        Some("float16") | Some("fp16") | Some("half") => DType::F16,
        Some("float32") | Some("fp32") => DType::F32,
        Some("float8_e4m3fn") => DType::F8E4M3,
        _ => DType::BF16,
    }
}
