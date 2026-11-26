use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

use super::ConfigError;

/// One `layers_block_type` entry. `linear_attention` / `full_attention` are the
/// current spellings of `mamba` / `attention` (`configuration_nemotron_h_omni.py`
/// `_nemotron_h_compatible_config`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronHLayer {
    Mamba,
    Attention,
    Moe,
}

/// Storage of one linear, resolved from ModelOpt `quantized_layers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronHLinearQuant {
    Bf16,
    /// `F8E4M3` weight, fp32 per-tensor `weight_scale`, static `input_scale`.
    Fp8,
    /// Packed e2m1 `U8` weight, e4m3 block `weight_scale` per `group` inputs,
    /// fp32 `weight_scale_2`, static `input_scale`.
    Nvfp4 {
        group: u32,
    },
}

pub const NEMOTRON_H_NVFP4_GROUP: u32 = 16;

#[derive(Debug, Clone, Deserialize)]
pub struct ModeloptLayerQuant {
    pub quant_algo: String,
    #[serde(default)]
    pub group_size: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModeloptKvCacheScheme {
    pub num_bits: u32,
    #[serde(rename = "type")]
    pub kind: String,
    pub dynamic: bool,
}

/// ModelOpt `modelopt_mixed` export: per-module FP8 / NVFP4 in
/// `quantized_layers`, every other module BF16.
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHQuantizationConfig {
    pub quant_method: String,
    pub quant_algo: String,
    #[serde(default)]
    pub quantized_layers: BTreeMap<String, ModeloptLayerQuant>,
    #[serde(default)]
    pub kv_cache_scheme: Option<ModeloptKvCacheScheme>,
}

/// Nemotron-H language model (`nemotron_h`, or the `llm_config` of
/// `nemotron_h_omni`): pre-norm residual blocks whose mixer is Mamba-2, NoPE GQA
/// or a sigmoid-routed (optionally latent) relu² MoE with one shared expert.
///
/// Text tower only. Not modeled, and listed in `omitted`: the omni wrapper's
/// RADIO vision tower and `mlp1` projector (CPE position-embedding resampling and
/// pixel shuffle have no IR op), its optional Parakeet sound encoder, and the
/// `mtp.*` draft layers (the reference does not execute them, and the config's
/// `quantization_config` does not describe their NVFP4 experts).
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub layers_block_type: Vec<String>,
    pub layer_norm_epsilon: f32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub num_nextn_predict_layers: u32,

    pub mamba_num_heads: u32,
    pub mamba_head_dim: i64,
    pub ssm_state_size: i64,
    pub n_groups: u32,
    pub conv_kernel: u32,
    pub chunk_size: u32,
    pub mamba_hidden_act: String,
    pub use_conv_bias: bool,
    pub use_bias: bool,
    /// `(min, max)` clamp of `softplus(dt + dt_bias)`. The CUDA path
    /// (`mamba_chunk_scan_combined(dt_limit=...)`) and vLLM apply this; the HF
    /// torch fallback clamps at `time_step_min` instead.
    #[serde(deserialize_with = "time_step_limit")]
    pub time_step_limit: (f32, f32),

    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: i64,
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<u32>,

    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    #[serde(default)]
    pub moe_latent_size: Option<i64>,
    pub moe_shared_expert_intermediate_size: i64,
    pub n_shared_experts: u32,
    pub n_group: u32,
    pub topk_group: u32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub mlp_hidden_act: String,
    pub mlp_bias: bool,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub quantization_config: Option<NemotronHQuantizationConfig>,

    /// Checkpoint name prefix before `backbone.` / `lm_head` (`language_model.`
    /// under the omni wrapper, empty for a bare `NemotronHForCausalLM`).
    #[serde(skip)]
    pub weight_prefix: String,
    #[serde(skip)]
    pub layers: Vec<NemotronHLayer>,
    #[serde(skip)]
    pub omitted: Vec<String>,
}

/// `[0.0, {"__float__": "Infinity"}]`: transformers encodes non-finite floats as objects.
fn time_step_limit<'de, D: serde::Deserializer<'de>>(d: D) -> Result<(f32, f32), D::Error> {
    use serde::de::Error;
    let v: Vec<serde_json::Value> = Deserialize::deserialize(d)?;
    let num = |v: &serde_json::Value| -> Result<f32, D::Error> {
        if let Some(x) = v.as_f64() {
            return Ok(x as f32);
        }
        match v.get("__float__").and_then(serde_json::Value::as_str) {
            Some("Infinity") => Ok(f32::INFINITY),
            Some("-Infinity") => Ok(f32::NEG_INFINITY),
            _ => Err(D::Error::custom(format!(
                "time_step_limit entry {v} is not a float"
            ))),
        }
    };
    match v.as_slice() {
        [lo, hi] => Ok((num(lo)?, num(hi)?)),
        _ => Err(D::Error::custom("time_step_limit must have two entries")),
    }
}

impl NemotronHConfig {
    pub fn d_inner(&self) -> i64 {
        self.mamba_num_heads as i64 * self.mamba_head_dim
    }

    /// `B`/`C` width: `n_groups * ssm_state_size`.
    pub fn bc_width(&self) -> i64 {
        self.n_groups as i64 * self.ssm_state_size
    }

    /// Depthwise-conv channels: `x ‖ B ‖ C`.
    pub fn conv_dim(&self) -> i64 {
        self.d_inner() + 2 * self.bc_width()
    }

    /// `in_proj` output: `z ‖ xBC ‖ dt`.
    pub fn in_proj_width(&self) -> i64 {
        self.d_inner() + self.conv_dim() + self.mamba_num_heads as i64
    }

    pub fn linear_quant(&self, module: &str) -> NemotronHLinearQuant {
        let Some(q) = self
            .quantization_config
            .as_ref()
            .and_then(|q| q.quantized_layers.get(module))
        else {
            return NemotronHLinearQuant::Bf16;
        };
        match q.quant_algo.as_str() {
            "FP8" => NemotronHLinearQuant::Fp8,
            _ => NemotronHLinearQuant::Nvfp4 {
                group: q.group_size.unwrap_or(NEMOTRON_H_NVFP4_GROUP),
            },
        }
    }

    pub fn kv_cache_fp8(&self) -> bool {
        self.quantization_config
            .as_ref()
            .is_some_and(|q| q.kv_cache_scheme.is_some())
    }

    /// Every linear the graph builds, with its `(in, out)` features.
    pub fn linear_modules(&self) -> Vec<(String, i64, i64)> {
        let p = &self.weight_prefix;
        let h = self.hidden_size;
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let m = format!("{p}backbone.layers.{i}.mixer");
            match layer {
                NemotronHLayer::Mamba => {
                    out.push((format!("{m}.in_proj"), h, self.in_proj_width()));
                    out.push((format!("{m}.out_proj"), self.d_inner(), h));
                }
                NemotronHLayer::Attention => {
                    let q = self.num_attention_heads as i64 * self.head_dim;
                    let kv = self.num_key_value_heads as i64 * self.head_dim;
                    out.push((format!("{m}.q_proj"), h, q));
                    out.push((format!("{m}.k_proj"), h, kv));
                    out.push((format!("{m}.v_proj"), h, kv));
                    out.push((format!("{m}.o_proj"), q, h));
                }
                NemotronHLayer::Moe => {
                    let e_in = self.moe_latent_size.unwrap_or(h);
                    if let Some(l) = self.moe_latent_size {
                        out.push((format!("{m}.fc1_latent_proj"), h, l));
                        out.push((format!("{m}.fc2_latent_proj"), l, h));
                    }
                    let si = self.moe_shared_expert_intermediate_size;
                    out.push((format!("{m}.shared_experts.up_proj"), h, si));
                    out.push((format!("{m}.shared_experts.down_proj"), si, h));
                    let ei = self.moe_intermediate_size;
                    for e in 0..self.n_routed_experts {
                        out.push((format!("{m}.experts.{e}.up_proj"), e_in, ei));
                        out.push((format!("{m}.experts.{e}.down_proj"), ei, e_in));
                    }
                }
            }
        }
        if !self.tie_word_embeddings {
            out.push((format!("{p}lm_head"), h, self.vocab_size));
        }
        out
    }

    pub fn validate(&mut self) -> Result<(), ConfigError> {
        let bad = |msg: String| {
            Err(ConfigError::Unsupported(format!(
                "nemotron_h (text tower only): {msg}"
            )))
        };
        self.layers = Vec::with_capacity(self.layers_block_type.len());
        for t in &self.layers_block_type {
            self.layers.push(match t.as_str() {
                "mamba" | "linear_attention" => NemotronHLayer::Mamba,
                "attention" | "full_attention" => NemotronHLayer::Attention,
                "moe" => NemotronHLayer::Moe,
                "mlp" => return bad("dense `mlp` blocks are not modeled".into()),
                other => return bad(format!("unknown layers_block_type entry {other:?}")),
            });
        }
        if self.layers.is_empty() {
            return bad("layers_block_type is empty".into());
        }
        if self.mamba_hidden_act != "silu" {
            return bad(format!(
                "mamba_hidden_act {} (only silu)",
                self.mamba_hidden_act
            ));
        }
        if self.mlp_hidden_act != "relu2" {
            return bad(format!(
                "mlp_hidden_act {} (only relu2)",
                self.mlp_hidden_act
            ));
        }
        // The reference hardcodes bias-free attention and expert tensors; a true
        // flag would describe tensors the reference never builds.
        for (key, on) in [
            ("use_bias", self.use_bias),
            ("mlp_bias", self.mlp_bias),
            ("attention_bias", self.attention_bias),
        ] {
            if on {
                return bad(format!("{key}=true is not modeled"));
            }
        }
        if let Some(w) = self.sliding_window {
            return bad(format!("sliding_window {w} is not modeled"));
        }
        let (nh, ng) = (self.mamba_num_heads, self.n_groups);
        if nh == 0 || ng == 0 || nh % ng != 0 {
            return bad(format!("n_groups {ng} must divide mamba_num_heads {nh}"));
        }
        if self.mamba_head_dim <= 0 || self.ssm_state_size <= 0 || self.conv_kernel == 0 {
            return bad("Mamba head_dim, ssm_state_size and conv_kernel must be positive".into());
        }
        if self.chunk_size == 0 {
            return bad("chunk_size must be positive".into());
        }
        let (lo, hi) = self.time_step_limit;
        if !(lo >= 0.0 && hi > lo) {
            return bad(format!(
                "time_step_limit ({lo}, {hi}) must satisfy 0 <= min < max"
            ));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return bad("num_attention_heads must be divisible by num_key_value_heads".into());
        }
        if self.head_dim <= 0 {
            return bad(format!("head_dim {} must be positive", self.head_dim));
        }
        if self.n_shared_experts != 1 {
            return bad(format!(
                "n_shared_experts {} (the reference always builds exactly one shared expert)",
                self.n_shared_experts
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            return bad(format!(
                "num_experts_per_tok {} must be in 1..={}",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.n_group == 0 || self.n_routed_experts % self.n_group != 0 {
            return bad(format!(
                "n_group {} must divide n_routed_experts {}",
                self.n_group, self.n_routed_experts
            ));
        }
        if self.topk_group == 0 || self.topk_group > self.n_group {
            return bad(format!(
                "topk_group {} must be in 1..={}",
                self.topk_group, self.n_group
            ));
        }
        if self.moe_latent_size.is_some_and(|l| l <= 0) {
            return bad("moe_latent_size must be positive".into());
        }
        self.validate_quantization().or_else(|msg| bad(msg))?;
        if self.num_nextn_predict_layers > 0 {
            self.omitted.push(format!(
                "mtp ({} draft layer(s))",
                self.num_nextn_predict_layers
            ));
        }
        Ok(())
    }

    fn validate_quantization(&self) -> Result<(), String> {
        let Some(q) = &self.quantization_config else {
            return Ok(());
        };
        if q.quant_method != "modelopt_mixed" || q.quant_algo != "MIXED_PRECISION" {
            return Err(format!(
                "quantization: only ModelOpt modelopt_mixed / MIXED_PRECISION is modeled, got \
                 method={:?} algo={:?}",
                q.quant_method, q.quant_algo
            ));
        }
        if let Some(kv) = &q.kv_cache_scheme {
            if kv.num_bits != 8 || kv.kind != "float" || kv.dynamic {
                return Err(format!(
                    "kv_cache_scheme {kv:?}: only static 8-bit float (FP8) is modeled"
                ));
            }
        }
        // The router gate and embeddings are not in this set: a quantized one is refused.
        let inputs: HashMap<String, i64> = self
            .linear_modules()
            .into_iter()
            .map(|(m, input, _)| (m, input))
            .collect();
        for (module, lq) in &q.quantized_layers {
            let Some(&input) = inputs.get(module) else {
                return Err(format!(
                    "quantized module {module:?} is not a linear of the modeled graph"
                ));
            };
            match (lq.quant_algo.as_str(), lq.group_size) {
                ("FP8", None) => {}
                ("NVFP4", Some(g)) if g == NEMOTRON_H_NVFP4_GROUP => {
                    if input % g as i64 != 0 {
                        return Err(format!(
                            "{module}: NVFP4 needs in_features {input} divisible by {g}"
                        ));
                    }
                }
                (algo, group) => {
                    return Err(format!(
                        "{module}: quant_algo {algo:?} group {group:?} is not modeled (FP8, or \
                         NVFP4 with group {NEMOTRON_H_NVFP4_GROUP})"
                    ))
                }
            }
        }
        Ok(())
    }
}
