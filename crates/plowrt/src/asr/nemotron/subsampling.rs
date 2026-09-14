use gguf_rs_lib::format::types::GGUFTensorType;

use crate::asr::frontend::LogMelFeatures;
use crate::asr::subsampling::{
    Activation, Conv2dSpec, Conv2dStage, ConvKind, CpuSubsampler, Padding, SubsampledFeatures,
    SubsamplingPlan,
};
use crate::asset::gguf::{GgufFile, GgufTensor};
use crate::{Result, RuntimeError};

pub struct NemotronSubsampler {
    inner: CpuSubsampler,
}

impl NemotronSubsampler {
    pub fn load(model: &GgufFile) -> Result<Self> {
        Ok(Self {
            inner: CpuSubsampler::from_plan(&subsampling_plan(model)?)?,
        })
    }

    pub fn run(&self, input: &LogMelFeatures) -> Result<SubsampledFeatures> {
        self.inner.run(input)
    }
}

pub fn subsampling_plan(model: &GgufFile) -> Result<SubsamplingPlan<'_>> {
    let metadata = model.metadata();
    let feature_bins = metadata
        .get_u64("asr.encoder.feat_in")
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rejected("missing asr.encoder.feat_in"))?;
    let channels = metadata
        .get_u64("asr.encoder.subsampling_conv_channels")
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rejected("missing asr.encoder.subsampling_conv_channels"))?;
    if metadata.get_u64("asr.encoder.subsampling_factor") != Some(8)
        || metadata.get_string("asr.encoder.conv_context") != Some("causal")
    {
        return Err(rejected("requires causal 8x subsampling"));
    }

    let definitions = [
        (0, 3, 2, 1, channels, ConvKind::Standard, Activation::Relu),
        (
            2,
            3,
            2,
            channels,
            channels,
            ConvKind::Depthwise,
            Activation::None,
        ),
        (
            3,
            1,
            1,
            channels,
            channels,
            ConvKind::Standard,
            Activation::Relu,
        ),
        (
            5,
            3,
            2,
            channels,
            channels,
            ConvKind::Depthwise,
            Activation::None,
        ),
        (
            6,
            1,
            1,
            channels,
            channels,
            ConvKind::Standard,
            Activation::Relu,
        ),
    ];
    let mut stages = Vec::with_capacity(definitions.len());
    for (index, kernel, stride, input_channels, output_channels, kind, activation) in definitions {
        stages.push(load_stage(
            model,
            index,
            Conv2dSpec {
                kernel,
                stride,
                padding: if kernel == 3 {
                    Padding {
                        before: 2,
                        after: 1,
                    }
                } else {
                    Padding {
                        before: 0,
                        after: 0,
                    }
                },
                input_channels,
                output_channels,
                kind,
                activation,
            },
        )?);
    }
    SubsamplingPlan::new(feature_bins, stages)
}

fn load_stage(model: &GgufFile, index: usize, spec: Conv2dSpec) -> Result<Conv2dStage<'_>> {
    let prefix = format!("encoder.pre_encode.conv.{index}");
    let weight = model.tensor(&format!("{prefix}.weight"))?;
    let stored_channels = if spec.kind == ConvKind::Depthwise {
        1
    } else {
        spec.input_channels
    };
    require_tensor(
        &weight,
        &[
            spec.kernel as u64,
            spec.kernel as u64,
            stored_channels as u64,
            spec.output_channels as u64,
        ],
        GGUFTensorType::F16,
        2,
    )?;
    let bias = model.tensor(&format!("{prefix}.bias"))?;
    require_tensor(
        &bias,
        &[spec.output_channels as u64],
        GGUFTensorType::F32,
        4,
    )?;
    Ok(Conv2dStage {
        spec,
        weights_f16_le: weight.bytes,
        bias_f32_le: bias.bytes,
    })
}

fn require_tensor(
    tensor: &GgufTensor<'_>,
    expected: &[u64],
    dtype: GGUFTensorType,
    element_bytes: usize,
) -> Result<()> {
    let elements = expected.iter().try_fold(1usize, |count, &dimension| {
        usize::try_from(dimension)
            .ok()
            .and_then(|dimension| count.checked_mul(dimension))
    });
    if tensor.dimensions != expected
        || tensor.dtype != dtype
        || elements.and_then(|count| count.checked_mul(element_bytes)) != Some(tensor.bytes.len())
    {
        return Err(rejected(format!("invalid tensor {}", tensor.name)));
    }
    Ok(())
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid Nemotron subsampler: {}", message.into()))
}
