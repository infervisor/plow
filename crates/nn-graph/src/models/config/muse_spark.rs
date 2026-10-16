use serde::Deserialize;

use super::ConfigError;
use crate::DType;

#[derive(Debug, Clone, Deserialize)]
pub struct MuseSparkRopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuarkTensorSpec {
    pub dtype: String,
    pub qscheme: Option<String>,
    pub group_size: Option<u32>,
    pub scale_format: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuarkGlobalQuantConfig {
    pub weight: QuarkTensorSpec,
    pub input_tensors: Option<QuarkTensorSpec>,
}

/// AMD Quark export. Only the MXFP4 form (fp4 weights, 32-wide groups, e8m0
/// scales) is modeled; `exclude` lists the modules kept in bf16.
#[derive(Debug, Clone, Deserialize)]
pub struct MuseSparkQuantizationConfig {
    pub quant_method: String,
    pub global_quant_config: QuarkGlobalQuantConfig,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub layer_quant_config: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub layer_type_quant_config: serde_json::Map<String, serde_json::Value>,
}

/// Muse Spark `text_config` (`muse_spark_text`). The graph is the language model
/// only, like vLLM `--language-model-only`: the vision tower, audio tower and
/// vision adapter/projection are not modeled, and their checkpoint tensors are
/// absent from the manifest. `omitted_towers` names the ones the source config
/// carried.
#[derive(Debug, Clone, Deserialize)]
pub struct MuseSparkConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: i64,
    pub rms_norm_eps: f32,
    pub post_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    pub hidden_act: String,
    pub sliding_window: u32,
    pub layer_types: Vec<String>,
    pub layer_rope_theta: Vec<f32>,
    pub rope_parameters: MuseSparkRopeParameters,
    pub mlp_layer_types: Vec<String>,
    pub qk_scale_factor: f32,
    pub output_multiplier: f32,
    pub final_logit_softcapping: f32,
    pub residual_gate_temperature: f32,
    pub num_local_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    pub moe_hidden_size: i64,
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub num_shared_experts: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub quantization_config: Option<MuseSparkQuantizationConfig>,
    #[serde(skip)]
    pub omitted_towers: Vec<String>,
}

pub const MUSE_SPARK_MXFP4_GROUP: i64 = 32;
/// `MuseSparkResidualGate(residual_gate_eps=1e-3)`.
pub const MUSE_SPARK_RESIDUAL_GATE_EPS: f32 = 1e-3;

impl MuseSparkConfig {
    pub fn is_sliding(&self, layer: u32) -> bool {
        self.layer_types[layer as usize] == "sliding_attention"
    }

    pub fn is_sparse(&self, layer: u32) -> bool {
        self.mlp_layer_types[layer as usize] == "sparse"
    }

    /// RoPE theta, or `None` for a NoPE layer. The reference uses one global
    /// rotary table; `layer_rope_theta` only switches it off (`0`).
    pub fn rope_theta(&self, layer: u32) -> Option<f32> {
        (self.layer_rope_theta[layer as usize] != 0.0).then_some(self.rope_parameters.rope_theta)
    }

    /// Storage dtype of linear `module` (no `.weight`): F4 for Quark MXFP4
    /// modules (F4 storage accounting includes the e8m0 group scale), else bf16.
    pub fn module_weight_dtype(&self, module: &str) -> DType {
        match &self.quantization_config {
            Some(q) if !q.exclude.iter().any(|m| m == module) => DType::F4,
            _ => DType::BF16,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |msg: String| {
            Err(ConfigError::Unsupported(format!(
                "muse_spark (text tower only; vision/audio not modeled): {msg}"
            )))
        };
        let n = self.num_hidden_layers as usize;
        for (key, len) in [
            ("layer_types", self.layer_types.len()),
            ("layer_rope_theta", self.layer_rope_theta.len()),
            ("mlp_layer_types", self.mlp_layer_types.len()),
        ] {
            if len != n {
                return bad(format!("{key} has {len} entries for {n} layers"));
            }
        }
        if let Some(t) = self
            .layer_types
            .iter()
            .find(|t| !matches!(t.as_str(), "sliding_attention" | "full_attention"))
        {
            return bad(format!("unknown layer type {t:?}"));
        }
        if let Some(t) = self
            .mlp_layer_types
            .iter()
            .find(|t| !matches!(t.as_str(), "sparse" | "dense"))
        {
            return bad(format!("unknown mlp layer type {t:?}"));
        }
        if self.rope_parameters.rope_type != "default" {
            return bad(format!(
                "rope_type {:?} (only default)",
                self.rope_parameters.rope_type
            ));
        }
        if let Some(t) = self
            .layer_rope_theta
            .iter()
            .find(|t| **t != 0.0 && **t != self.rope_parameters.rope_theta)
        {
            return bad(format!(
                "layer_rope_theta {t} differs from rope_parameters.rope_theta {}; the reference \
                 applies the global theta to every RoPE layer",
                self.rope_parameters.rope_theta
            ));
        }
        if self.hidden_act != "silu" {
            return bad(format!("hidden_act {} (only silu)", self.hidden_act));
        }
        if self.num_shared_experts != 0 {
            return bad(format!("num_shared_experts = {}", self.num_shared_experts));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return bad("num_attention_heads must be divisible by num_key_value_heads".into());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_local_experts {
            return bad(format!(
                "num_experts_per_tok {} must be in 1..={}",
                self.num_experts_per_tok, self.num_local_experts
            ));
        }
        if self.head_dim <= 0 || self.head_dim % 2 != 0 {
            return bad(format!(
                "head_dim {} must be positive and even",
                self.head_dim
            ));
        }
        if let Some(q) = &self.quantization_config {
            let w = &q.global_quant_config.weight;
            if q.quant_method != "quark"
                || w.dtype != "fp4"
                || w.qscheme.as_deref() != Some("per_group")
                || w.group_size != Some(MUSE_SPARK_MXFP4_GROUP as u32)
                || w.scale_format.as_deref() != Some("e8m0")
            {
                return bad(format!(
                    "quantization: only Quark MXFP4 (fp4, per_group 32, e8m0) is modeled, got \
                     method={:?} dtype={:?} qscheme={:?} group={:?} scale={:?}",
                    q.quant_method, w.dtype, w.qscheme, w.group_size, w.scale_format
                ));
            }
            if !q.layer_quant_config.is_empty() || !q.layer_type_quant_config.is_empty() {
                return bad("Quark per-layer quant overrides are not modeled".into());
            }
            if q.exclude.iter().any(|m| m.contains('*')) {
                return bad("Quark wildcard exclude patterns are not modeled".into());
            }
            if let Some(module) = self.quantized_non_expert_module() {
                return bad(format!(
                    "{module} is MXFP4; only routed experts are modeled as MXFP4"
                ));
            }
        }
        Ok(())
    }

    fn quantized_non_expert_module(&self) -> Option<String> {
        let mut modules = vec!["lm_head".to_string()];
        for layer in 0..self.num_hidden_layers {
            let p = format!("model.language_model.layers.{layer}");
            for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] {
                modules.push(format!("{p}.self_attn.{m}"));
            }
            let mlp = if self.is_sparse(layer) {
                ["gate", "pre_expert_proj", "post_expert_proj"].as_slice()
            } else {
                ["gate_proj", "up_proj", "down_proj"].as_slice()
            };
            modules.extend(mlp.iter().map(|m| format!("{p}.mlp.{m}")));
        }
        modules
            .into_iter()
            .find(|m| self.module_weight_dtype(m) != DType::BF16)
    }
}
