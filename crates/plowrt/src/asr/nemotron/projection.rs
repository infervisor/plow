use gguf_rs_lib::format::types::GGUFTensorType;

use crate::asr::nemotron::subsampling::subsampling_plan;
use crate::asset::gguf::GgufFile;
use crate::ops::linear::Q8LinearPlan;
use crate::{Result, RuntimeError};

pub fn pre_encode_projection(model: &GgufFile) -> Result<Q8LinearPlan<'_>> {
    let stem = subsampling_plan(model)?;
    let shape = *stem
        .shapes(1)?
        .last()
        .ok_or_else(|| rejected("subsampling plan has no output"))?;
    let input_width = shape
        .width
        .checked_mul(shape.channels)
        .ok_or_else(|| rejected("projection input width overflows"))?;
    let output_width = model
        .metadata()
        .get_u64("asr.encoder.d_model")
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rejected("missing asr.encoder.d_model"))?;

    let weight_tensor = model.tensor("encoder.pre_encode.out.weight")?;
    let weight = weight_tensor.q8_0_matrix()?;
    if weight.k() != input_width || weight.n() != output_width {
        return Err(rejected(format!(
            "projection matrix is [{}, {}], expected [{input_width}, {output_width}]",
            weight.k(),
            weight.n()
        )));
    }
    let bias = model.tensor("encoder.pre_encode.out.bias")?;
    if bias.dtype != GGUFTensorType::F32 || bias.dimensions != [output_width as u64] {
        return Err(rejected("invalid encoder.pre_encode.out.bias"));
    }
    Q8LinearPlan::new(weight, bias.bytes)
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!(
        "invalid Nemotron pre-encoder projection: {}",
        message.into()
    ))
}
