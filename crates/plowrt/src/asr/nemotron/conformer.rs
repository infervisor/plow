use gguf_rs_lib::format::types::GGUFTensorType;

use crate::asr::conformer::{
    AttentionMask, ConformerBlockPlan, ConformerEncoderPlan, ConvolutionPlan, FeedForwardPlan,
    RelativeAttentionPlan,
};
use crate::asset::gguf::GgufFile;
use crate::ops::linear::Q8Matrix;
use crate::ops::norm::LayerNormPlan;
use crate::{Result, RuntimeError};

const LAYER_NORM_EPSILON: f32 = 1e-5;

pub fn conformer_encoder_plan(model: &GgufFile) -> Result<ConformerEncoderPlan<'_>> {
    let layers = usize_metadata(model, "asr.encoder.n_layers")?;
    let blocks = (0..layers)
        .map(|layer| conformer_block_plan(model, layer))
        .collect::<Result<Vec<_>>>()?;
    ConformerEncoderPlan::new(blocks)
}

pub fn conformer_block_plan(model: &GgufFile, layer: usize) -> Result<ConformerBlockPlan<'_>> {
    let metadata = model.metadata();
    let width = usize_metadata(model, "asr.encoder.d_model")?;
    let doubled_width = width
        .checked_mul(2)
        .ok_or_else(|| rejected("encoder width overflows"))?;
    let hidden = usize_metadata(model, "asr.encoder.d_ff")?;
    let heads = usize_metadata(model, "asr.encoder.n_heads")?;
    let layers = usize_metadata(model, "asr.encoder.n_layers")?;
    let kernel = usize_metadata(model, "asr.encoder.conv_kernel_size")?;
    let position_center = usize_metadata(model, "asr.encoder.pos_emb_max_len")?;
    if layer >= layers {
        return Err(rejected(format!(
            "layer {layer} is outside {layers} layers"
        )));
    }
    if metadata.get_string("asr.encoder.conv_context") != Some("causal")
        || metadata.get_string("asr.encoder.conv_norm") != Some("layer_norm")
        || metadata.get_string("asr.encoder.att_context_style") != Some("chunked_limited")
        || metadata.get_string("asr.encoder.q8_layout") != Some("block_q8_0")
        || metadata.get_string("asr.encoder.qkv_q8_layout") != Some("block_q8_0")
        || metadata.get_bool("asr.encoder.use_bias") != Some(false)
        || metadata.get_bool("asr.encoder.xscaling") != Some(false)
    {
        return Err(rejected("unsupported encoder configuration"));
    }
    let left_context = usize::try_from(
        metadata
            .get_i64("asr.encoder.offline_left_ctx")
            .ok_or_else(|| rejected("missing asr.encoder.offline_left_ctx"))?,
    )
    .map_err(|_| rejected("negative asr.encoder.offline_left_ctx"))?;
    let right_context = usize::try_from(
        metadata
            .get_i64("asr.encoder.offline_right_ctx")
            .ok_or_else(|| rejected("missing asr.encoder.offline_right_ctx"))?,
    )
    .map_err(|_| rejected("negative asr.encoder.offline_right_ctx"))?;
    let chunk_size = right_context
        .checked_add(1)
        .ok_or_else(|| rejected("attention chunk size overflows"))?;
    if !left_context.is_multiple_of(chunk_size) {
        return Err(rejected("left context is not a whole number of chunks"));
    }
    let prefix = format!("encoder.layers.{layer}");
    let feed_forward1 = feed_forward(model, &prefix, "1", width, hidden)?;
    let feed_forward2 = feed_forward(model, &prefix, "2", width, hidden)?;
    let attention_prefix = format!("{prefix}.self_attn");
    let head_width = width
        .checked_div(heads)
        .filter(|_| width.is_multiple_of(heads))
        .ok_or_else(|| rejected("attention head geometry mismatch"))?;
    let position = model.tensor("encoder.pos_enc.pe")?;
    if position.dtype != GGUFTensorType::F32
        || position.dimensions.len() != 2
        || position.dimensions[0] != width as u64
    {
        return Err(rejected("invalid encoder.pos_enc.pe"));
    }
    let position_count = usize::try_from(position.dimensions[1])
        .map_err(|_| rejected("position count overflows usize"))?;
    let attention = RelativeAttentionPlan {
        norm: layer_norm(model, &format!("{prefix}.norm_self_att"), width)?,
        query: q8(
            model,
            &format!("{attention_prefix}.linear_q.weight"),
            width,
            width,
        )?,
        key: q8(
            model,
            &format!("{attention_prefix}.linear_k.weight"),
            width,
            width,
        )?,
        value: q8(
            model,
            &format!("{attention_prefix}.linear_v.weight"),
            width,
            width,
        )?,
        position: q8(
            model,
            &format!("{attention_prefix}.linear_pos.weight"),
            width,
            width,
        )?,
        output: q8(
            model,
            &format!("{attention_prefix}.linear_out.weight"),
            width,
            width,
        )?,
        bias_u_f32_le: f32_tensor(
            model,
            &format!("{attention_prefix}.pos_bias_u"),
            &[head_width, heads],
        )?,
        bias_v_f32_le: f32_tensor(
            model,
            &format!("{attention_prefix}.pos_bias_v"),
            &[head_width, heads],
        )?,
        position_f32_le: position.bytes,
        position_count,
        position_center,
        heads,
        mask: AttentionMask::ChunkedLimited {
            chunk_size,
            left_chunks: left_context / chunk_size,
        },
    };
    let convolution_prefix = format!("{prefix}.conv");
    let depthwise = model.tensor(&format!("{convolution_prefix}.depthwise_conv.weight"))?;
    let depthwise_bytes = kernel
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| rejected("depthwise convolution size overflows"))?;
    if depthwise.dtype != GGUFTensorType::F16
        || depthwise.dimensions != [kernel as u64, 1, width as u64]
        || depthwise.bytes.len() != depthwise_bytes
    {
        return Err(rejected("invalid depthwise convolution weight"));
    }
    let convolution = ConvolutionPlan {
        norm: layer_norm(model, &format!("{prefix}.norm_conv"), width)?,
        pointwise_in: q8(
            model,
            &format!("{convolution_prefix}.pointwise_conv1.weight"),
            width,
            doubled_width,
        )?,
        depthwise_f16_le: depthwise.bytes,
        kernel,
        channel_norm: layer_norm(model, &format!("{convolution_prefix}.batch_norm"), width)?,
        pointwise_out: q8(
            model,
            &format!("{convolution_prefix}.pointwise_conv2.weight"),
            width,
            width,
        )?,
    };
    ConformerBlockPlan::new(
        width,
        feed_forward1,
        attention,
        convolution,
        feed_forward2,
        layer_norm(model, &format!("{prefix}.norm_out"), width)?,
    )
}

fn feed_forward<'a>(
    model: &'a GgufFile,
    prefix: &str,
    suffix: &str,
    width: usize,
    hidden: usize,
) -> Result<FeedForwardPlan<'a>> {
    let name = format!("feed_forward{suffix}");
    Ok(FeedForwardPlan {
        norm: layer_norm(model, &format!("{prefix}.norm_{name}"), width)?,
        expand: q8(
            model,
            &format!("{prefix}.{name}.linear1.weight"),
            width,
            hidden,
        )?,
        contract: q8(
            model,
            &format!("{prefix}.{name}.linear2.weight"),
            hidden,
            width,
        )?,
    })
}

fn layer_norm<'a>(model: &'a GgufFile, prefix: &str, width: usize) -> Result<LayerNormPlan<'a>> {
    LayerNormPlan::new(
        width,
        LAYER_NORM_EPSILON,
        f32_tensor(model, &format!("{prefix}.weight"), &[width])?,
        f32_tensor(model, &format!("{prefix}.bias"), &[width])?,
    )
}

fn q8<'a>(model: &'a GgufFile, name: &str, k: usize, n: usize) -> Result<Q8Matrix<'a>> {
    let tensor = model.tensor(name)?;
    let matrix = tensor.q8_0_matrix()?;
    if matrix.k() != k || matrix.n() != n {
        return Err(rejected(format!(
            "matrix {name} is [{}, {}], expected [{k}, {n}]",
            matrix.k(),
            matrix.n()
        )));
    }
    Ok(matrix)
}

fn f32_tensor<'a>(model: &'a GgufFile, name: &str, dimensions: &[usize]) -> Result<&'a [u8]> {
    let tensor = model.tensor(name)?;
    if tensor.dtype != GGUFTensorType::F32
        || tensor.dimensions.len() != dimensions.len()
        || !tensor
            .dimensions
            .iter()
            .zip(dimensions)
            .all(|(&actual, &expected)| actual == expected as u64)
        || dimensions
            .iter()
            .try_fold(4usize, |bytes, &dimension| bytes.checked_mul(dimension))
            != Some(tensor.bytes.len())
    {
        return Err(rejected(format!("invalid tensor {name}")));
    }
    Ok(tensor.bytes)
}

fn usize_metadata(model: &GgufFile, key: &str) -> Result<usize> {
    model
        .metadata()
        .get_u64(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rejected(format!("missing or invalid {key}")))
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid Nemotron Conformer: {}", message.into()))
}
