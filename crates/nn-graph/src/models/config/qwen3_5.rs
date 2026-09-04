use serde::Deserialize;

use crate::DType;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rope_type: String,
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35QuantizationConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub weight_block_size: [i64; 2],
    pub modules_to_not_convert: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: i64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: u32,
    pub linear_key_head_dim: i64,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_value_head_dim: i64,
    pub rms_norm_eps: f32,
    pub rope_parameters: Qwen35RopeParameters,
    pub attention_bias: bool,
    pub attn_output_gate: bool,
    pub hidden_act: String,
    pub mamba_ssm_dtype: String,
    pub output_gate_type: String,
    pub tie_word_embeddings: bool,
    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    #[serde(default, rename = "_plow_weight_prefix")]
    pub weight_prefix: Option<String>,
    #[serde(default)]
    pub quantization_config: Option<Qwen35QuantizationConfig>,
}

impl Qwen35Config {
    pub fn rotary_dim(&self) -> u32 {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as u32
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(format!(
                "qwen3_5 layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if let Some(layer_type) = self
            .layer_types
            .iter()
            .find(|t| !matches!(t.as_str(), "linear_attention" | "full_attention"))
        {
            return Err(format!("qwen3_5 unsupported layer type {layer_type}"));
        }
        if self.hidden_act != "silu" {
            return Err(format!(
                "qwen3_5 unsupported hidden_act {} (only silu is supported)",
                self.hidden_act
            ));
        }
        if self.mamba_ssm_dtype != "float32" {
            return Err("qwen3_5 requires mamba_ssm_dtype=float32".to_string());
        }
        if self.output_gate_type != "swish" {
            return Err("qwen3_5 requires output_gate_type=swish".to_string());
        }
        if self.rope_parameters.rope_type != "default" || !self.rope_parameters.mrope_interleaved {
            return Err(
                "qwen3_5 requires default, mrope-interleaved rotary embeddings".to_string(),
            );
        }
        if !self.attn_output_gate {
            return Err("qwen3_5 requires attn_output_gate=true".to_string());
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(
                "qwen3_5 num_attention_heads must be divisible by num_key_value_heads".to_string(),
            );
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return Err(
                "qwen3_5 linear_num_value_heads must be divisible by linear_num_key_heads"
                    .to_string(),
            );
        }
        if self.linear_key_head_dim != self.linear_value_head_dim {
            return Err(
                "qwen3_5 differing linear key/value head dimensions are unsupported".to_string(),
            );
        }
        if self.linear_conv_kernel_dim == 0
            || self.linear_key_head_dim <= 0
            || self.linear_value_head_dim <= 0
        {
            return Err("qwen3_5 linear-attention dimensions must be positive".to_string());
        }
        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return Err("qwen3_5 target checkpoint requires dtype=bfloat16".to_string());
        }
        if let Some(q) = &self.quantization_config {
            if q.quant_method != "fp8"
                || q.fmt != "e4m3"
                || q.activation_scheme != "dynamic"
                || q.weight_block_size != [128, 128]
            {
                return Err(format!(
                    "qwen3_5 unsupported quantization: expected dynamic fp8/e4m3 with \
                     weight_block_size=[128,128], got method={:?} fmt={:?} activation={:?} \
                     block={:?}",
                    q.quant_method, q.fmt, q.activation_scheme, q.weight_block_size
                ));
            }
            if !q.modules_to_not_convert.iter().any(|m| m == "lm_head") {
                return Err("qwen3_5 FP8 config must exclude lm_head".to_string());
            }
        }
        let rotary_dim = self.head_dim as f32 * self.rope_parameters.partial_rotary_factor;
        if self.head_dim <= 0
            || !(0.0..=1.0).contains(&self.rope_parameters.partial_rotary_factor)
            || self.rope_parameters.partial_rotary_factor == 0.0
            || rotary_dim.fract() != 0.0
            || rotary_dim as u32 % 2 != 0
        {
            return Err(
                "qwen3_5 partial RoPE dimension must be a positive even integer".to_string(),
            );
        }
        Ok(())
    }

    pub fn projection_weight_dtype(&self, module: &str) -> DType {
        match &self.quantization_config {
            Some(q) if !q.modules_to_not_convert.iter().any(|m| m == module) => DType::F8E4M3,
            _ => DType::BF16,
        }
    }

    pub fn fp8_scale_shape(&self, out_features: i64, in_features: i64) -> Option<[i64; 2]> {
        let q = self.quantization_config.as_ref()?;
        Some([
            (out_features + q.weight_block_size[0] - 1) / q.weight_block_size[0],
            (in_features + q.weight_block_size[1] - 1) / q.weight_block_size[1],
        ])
    }
}
