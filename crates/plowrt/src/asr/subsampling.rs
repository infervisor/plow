use crate::asr::frontend::LogMelFeatures;
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    None,
    Relu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvKind {
    Standard,
    Depthwise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Padding {
    pub before: usize,
    pub after: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dSpec {
    pub kernel: usize,
    pub stride: usize,
    pub padding: Padding,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kind: ConvKind,
    pub activation: Activation,
}

pub struct Conv2dStage<'a> {
    pub spec: Conv2dSpec,
    pub weights_f16_le: &'a [u8],
    pub bias_f32_le: &'a [u8],
}

pub struct SubsamplingPlan<'a> {
    input_width: usize,
    stages: Vec<Conv2dStage<'a>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureShape {
    pub frames: usize,
    pub width: usize,
    pub channels: usize,
}

pub struct SubsampledFeatures {
    /// Rows are `[channel, frequency]`, matching the convolution stem's projection flatten.
    pub values: Vec<f32>,
    pub frames: usize,
    pub width: usize,
    pub channels: usize,
}

struct Conv2d {
    spec: Conv2dSpec,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

pub struct CpuSubsampler {
    input_width: usize,
    stages: Vec<Conv2d>,
}

impl<'a> SubsamplingPlan<'a> {
    pub fn new(input_width: usize, stages: Vec<Conv2dStage<'a>>) -> Result<Self> {
        if input_width == 0 || stages.is_empty() {
            return Err(rejected("empty input or stage list"));
        }
        let mut channels = 1usize;
        for (index, stage) in stages.iter().enumerate() {
            let spec = stage.spec;
            if spec.kernel == 0
                || spec.stride == 0
                || spec.input_channels != channels
                || spec.output_channels == 0
                || (spec.kind == ConvKind::Depthwise && spec.input_channels != spec.output_channels)
            {
                return Err(rejected(format!("invalid stage {index} geometry")));
            }
            let stored_channels = if spec.kind == ConvKind::Depthwise {
                1
            } else {
                spec.input_channels
            };
            let weight_elements = spec
                .kernel
                .checked_mul(spec.kernel)
                .and_then(|count| count.checked_mul(stored_channels))
                .and_then(|count| count.checked_mul(spec.output_channels));
            if weight_elements.and_then(|count| count.checked_mul(2))
                != Some(stage.weights_f16_le.len())
                || spec.output_channels.checked_mul(4) != Some(stage.bias_f32_le.len())
            {
                return Err(rejected(format!("invalid stage {index} tensor size")));
            }
            channels = spec.output_channels;
        }
        Ok(Self {
            input_width,
            stages,
        })
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn stages(&self) -> &[Conv2dStage<'a>] {
        &self.stages
    }

    pub fn shapes(&self, frames: usize) -> Result<Vec<FeatureShape>> {
        if frames == 0 {
            return Err(rejected("input has no frames"));
        }
        let mut shape = FeatureShape {
            frames,
            width: self.input_width,
            channels: 1,
        };
        let mut shapes = Vec::with_capacity(self.stages.len() + 1);
        shapes.push(shape);
        for stage in &self.stages {
            if stage.spec.input_channels != shape.channels {
                return Err(rejected("stage channel mismatch"));
            }
            shape = FeatureShape {
                frames: output_extent(shape.frames, stage.spec)?,
                width: output_extent(shape.width, stage.spec)?,
                channels: stage.spec.output_channels,
            };
            shapes.push(shape);
        }
        Ok(shapes)
    }
}

impl CpuSubsampler {
    pub fn from_plan(plan: &SubsamplingPlan<'_>) -> Result<Self> {
        let stages = plan
            .stages()
            .iter()
            .map(|stage| {
                Ok(Conv2d {
                    spec: stage.spec,
                    weights: decode_f16(stage.weights_f16_le),
                    bias: decode_f32(stage.bias_f32_le),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_width: plan.input_width(),
            stages,
        })
    }

    pub fn run(&self, input: &LogMelFeatures) -> Result<SubsampledFeatures> {
        if input.bins != self.input_width
            || input.values.len() != input.frames.checked_mul(input.bins).unwrap_or(usize::MAX)
        {
            return Err(rejected("input feature shape mismatch"));
        }
        let mut shape = FeatureShape {
            frames: input.frames,
            width: input.bins,
            channels: 1,
        };
        let mut output = self.stages[0].run(&input.values, shape);
        for stage in &self.stages[1..] {
            shape = FeatureShape {
                frames: output.frames,
                width: output.width,
                channels: output.channels,
            };
            output = stage.run(&output.values, shape);
        }
        output.values =
            channels_first(&output.values, output.frames, output.width, output.channels);
        Ok(output)
    }
}

pub(crate) fn channels_first(
    values: &[f32],
    frames: usize,
    width: usize,
    channels: usize,
) -> Vec<f32> {
    let mut packed = vec![0.0; values.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            for column in 0..width {
                packed[(frame * channels + channel) * width + column] =
                    values[(frame * width + column) * channels + channel];
            }
        }
    }
    packed
}

impl Conv2d {
    fn run(&self, input: &[f32], input_shape: FeatureShape) -> SubsampledFeatures {
        debug_assert_eq!(input_shape.channels, self.spec.input_channels);
        let left_pad = self.spec.padding.before;
        let output_frames = output_extent(input_shape.frames, self.spec).unwrap();
        let output_width = output_extent(input_shape.width, self.spec).unwrap();
        let mut output = vec![0.0; output_frames * output_width * self.spec.output_channels];
        for output_frame in 0..output_frames {
            for output_x in 0..output_width {
                for output_channel in 0..self.spec.output_channels {
                    let mut sum = self.bias[output_channel];
                    let input_channel_start = if self.spec.kind == ConvKind::Depthwise {
                        output_channel
                    } else {
                        0
                    };
                    let input_channel_end = if self.spec.kind == ConvKind::Depthwise {
                        output_channel + 1
                    } else {
                        self.spec.input_channels
                    };
                    for kernel_y in 0..self.spec.kernel {
                        let Some(input_frame) =
                            (output_frame * self.spec.stride + kernel_y).checked_sub(left_pad)
                        else {
                            continue;
                        };
                        if input_frame >= input_shape.frames {
                            continue;
                        }
                        for kernel_x in 0..self.spec.kernel {
                            let Some(input_x) =
                                (output_x * self.spec.stride + kernel_x).checked_sub(left_pad)
                            else {
                                continue;
                            };
                            if input_x >= input_shape.width {
                                continue;
                            }
                            for input_channel in input_channel_start..input_channel_end {
                                let weight_channel = if self.spec.kind == ConvKind::Depthwise {
                                    0
                                } else {
                                    input_channel
                                };
                                let stored_channels = if self.spec.kind == ConvKind::Depthwise {
                                    1
                                } else {
                                    self.spec.input_channels
                                };
                                let weight = (((output_channel * stored_channels
                                    + weight_channel)
                                    * self.spec.kernel
                                    + kernel_y)
                                    * self.spec.kernel)
                                    + kernel_x;
                                let source = (input_frame * input_shape.width + input_x)
                                    * input_shape.channels
                                    + input_channel;
                                sum = self.weights[weight].mul_add(input[source], sum);
                            }
                        }
                    }
                    output[(output_frame * output_width + output_x) * self.spec.output_channels
                        + output_channel] = match self.spec.activation {
                        Activation::None => sum,
                        Activation::Relu => sum.max(0.0),
                    };
                }
            }
        }
        SubsampledFeatures {
            values: output,
            frames: output_frames,
            width: output_width,
            channels: self.spec.output_channels,
        }
    }
}

fn output_extent(input: usize, spec: Conv2dSpec) -> Result<usize> {
    input
        .checked_add(spec.padding.before)
        .and_then(|value| value.checked_add(spec.padding.after))
        .and_then(|value| value.checked_sub(spec.kernel))
        .map(|value| value / spec.stride + 1)
        .ok_or_else(|| rejected("convolution extent overflows"))
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|bytes| half_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn half_to_f32(value: u16) -> f32 {
    let sign = u32::from(value >> 15) << 31;
    let exponent = i32::from((value >> 10) & 0x1f);
    let fraction = u32::from(value & 0x03ff);
    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << shift) & 0x03ff;
            sign | ((127 - 14 - shift) << 23) | (normalized << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((u32::try_from(exponent - 15 + 127).unwrap()) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid ASR subsampler: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_stride_two_uses_two_left_and_one_right_padding() {
        let stage = Conv2d {
            weights: vec![1.0; 9],
            bias: vec![0.0],
            spec: Conv2dSpec {
                kernel: 3,
                stride: 2,
                padding: Padding {
                    before: 2,
                    after: 1,
                },
                input_channels: 1,
                output_channels: 1,
                kind: ConvKind::Standard,
                activation: Activation::None,
            },
        };
        let input = [1.0, 2.0, 3.0, 4.0];
        let input_shape = FeatureShape {
            frames: 2,
            width: 2,
            channels: 1,
        };
        let output = stage.run(&input, input_shape);
        assert_eq!((output.frames, output.width, output.channels), (2, 2, 1));
        assert_eq!(output.values, [1.0, 3.0, 4.0, 10.0]);
    }

    #[test]
    fn projection_rows_flatten_channels_before_frequency() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(
            channels_first(&values, 1, 2, 4),
            [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
        );
    }
}
