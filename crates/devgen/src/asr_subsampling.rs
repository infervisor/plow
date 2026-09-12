//! Audio subsampling composition built from generic convolution and projection packets.

use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

use crate::conv2d::{self, Conv2dSpec, ConvLayout};
use crate::pipeline::PacketPrefix;

pub use crate::conv2d::{Conv2dStage, ConvActivation, ConvKind, ConvWeight};

#[derive(Clone, Copy)]
pub struct Projection<'a> {
    pub weight: &'a str,
    pub bias: &'a str,
    pub output_width: u32,
}

#[derive(Clone, Copy)]
pub struct SubsamplingSpec {
    pub input_frames: u32,
    pub input_width: u32,
}

pub struct SubsamplingPackets {
    pub model: Model,
    pub programs: Vec<usize>,
    pub input: u32,
    pub output: u32,
    pub output_frames: u32,
    pub output_width: u32,
    input_shape: Vec<u64>,
}

impl SubsamplingPackets {
    pub fn into_prefix(self) -> PacketPrefix {
        PacketPrefix {
            model: self.model,
            programs: self.programs,
            input: self.input,
            output: self.output,
            input_shape: self.input_shape,
        }
    }
}

pub fn lower(
    spec: SubsamplingSpec,
    stages: &[Conv2dStage<'_>],
    projection: Projection<'_>,
    n_cu: u32,
) -> Result<SubsamplingPackets, String> {
    if projection.output_width == 0
        || stages
            .last()
            .is_none_or(|stage| stage.output_layout != ConvLayout::FrameChannelsWidth)
    {
        return Err("subsampling projection requires frame-major channel-first rows".into());
    }
    let convolution = conv2d::lower(
        Conv2dSpec {
            batches: 1,
            input_frames: spec.input_frames,
            input_width: spec.input_width,
            input_channels: 1,
            input_layout: ConvLayout::ChannelsLast,
        },
        stages,
        n_cu,
    )?;
    let projection_input = convolution
        .output_width
        .checked_mul(convolution.output_channels)
        .ok_or("projection input width overflows")?;
    if projection_input % 32 != 0 {
        return Err("projection input width is not Q8 aligned".into());
    }
    let mut builder = Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let input = convolution.input;
    let source = convolution.output;
    let frames = convolution.output_frames;
    let input_shape = convolution.input_shape;
    let mut model = convolution.model;
    builder.adopt_tensors(std::mem::take(&mut model.tensors));
    let output = builder.tensor(
        "act.asr.io",
        u64::from(frames) * u64::from(projection.output_width) * 4,
    );
    let weight = builder.tensor(
        projection.weight,
        u64::from(projection.output_width) * u64::from(projection_input / 32) * 34,
    );
    let bias = builder.tensor(projection.bias, u64::from(projection.output_width) * 4);
    builder.emit(
        DevOp::Q8GemmF32,
        repeated(
            n_cu,
            frames.div_ceil(128) * projection.output_width.div_ceil(64),
        ),
        &[],
        |instruction| {
            instruction.t[..4].copy_from_slice(&[output, source, weight, bias]);
            instruction.i[..3].copy_from_slice(&[
                frames,
                projection.output_width,
                projection_input,
            ]);
        },
    );
    let projection_program = model.progs.len();
    model.tensors = builder.tensors();
    model.progs.push(builder.finish());
    model.prog_t.push(frames);
    Ok(SubsamplingPackets {
        model,
        programs: vec![0, projection_program],
        input,
        output,
        output_frames: frames,
        output_width: projection.output_width,
        input_shape,
    })
}

fn repeated(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks.max(1)).map(|index| index % n_cu).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_conv_chain_and_channels_first_projection() {
        let stages = [
            Conv2dStage {
                weight: "conv.0.weight",
                bias: "conv.0.bias",
                kernel: 3,
                stride: 2,
                pad_before: 2,
                pad_after: 1,
                input_channels: 1,
                output_channels: 32,
                kind: ConvKind::Standard,
                activation: ConvActivation::Relu,
                output_layout: ConvLayout::ChannelsLast,
                weight_type: ConvWeight::F16,
            },
            Conv2dStage {
                weight: "conv.1.weight",
                bias: "conv.1.bias",
                kernel: 1,
                stride: 1,
                pad_before: 0,
                pad_after: 0,
                input_channels: 32,
                output_channels: 32,
                kind: ConvKind::Standard,
                activation: ConvActivation::Relu,
                output_layout: ConvLayout::FrameChannelsWidth,
                weight_type: ConvWeight::F16,
            },
        ];
        let packets = lower(
            SubsamplingSpec {
                input_frames: 7,
                input_width: 7,
            },
            &stages,
            Projection {
                weight: "projection.weight",
                bias: "projection.bias",
                output_width: 64,
            },
            4,
        )
        .unwrap();
        assert_eq!(packets.output_frames, 4);
        assert_eq!(packets.output_width, 64);
        assert_eq!(packets.model.progs.len(), 2);
        assert_eq!(packets.model.progs[0].insts.len(), 2);
        assert_eq!(packets.model.progs[1].insts.len(), 1);
        let prefix = packets.into_prefix();
        assert_eq!(prefix.programs, [0, 1]);
        assert_eq!(prefix.input_shape, [7, 7, 1]);
    }
}
