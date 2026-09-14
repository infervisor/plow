//! Backend-neutral 2D convolution packet lowering.

use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

use crate::pipeline::PacketPrefix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKind {
    Standard,
    Depthwise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvLayout {
    ChannelsLast,
    FrameChannelsWidth,
    ChannelsFramesWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvActivation {
    None,
    Relu,
    GeluErfBf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvWeight {
    F16,
    F32,
}

#[derive(Clone, Copy)]
pub struct Conv2dStage<'a> {
    pub weight: &'a str,
    pub bias: &'a str,
    pub kernel: u32,
    pub stride: u32,
    pub pad_before: u32,
    pub pad_after: u32,
    pub input_channels: u32,
    pub output_channels: u32,
    pub kind: ConvKind,
    pub activation: ConvActivation,
    pub output_layout: ConvLayout,
    pub weight_type: ConvWeight,
}

#[derive(Clone, Copy)]
pub struct Conv2dSpec {
    pub batches: u32,
    pub input_frames: u32,
    pub input_width: u32,
    pub input_channels: u32,
    pub input_layout: ConvLayout,
}

pub struct Conv2dPackets {
    pub model: Model,
    pub program: usize,
    pub input: u32,
    pub output: u32,
    pub output_frames: u32,
    pub output_width: u32,
    pub output_channels: u32,
    pub output_layout: ConvLayout,
    pub output_batches: u32,
    pub input_shape: Vec<u64>,
}

impl Conv2dPackets {
    pub fn into_prefix(self) -> PacketPrefix {
        PacketPrefix {
            model: self.model,
            programs: vec![self.program],
            input: self.input,
            output: self.output,
            input_shape: self.input_shape,
        }
    }

    pub fn pack_ncfw_rows(self, rows: u32) -> Result<PacketPrefix, String> {
        let capacity = self
            .output_batches
            .checked_mul(self.output_width)
            .ok_or("row-pack capacity overflows")?;
        if self.output_layout != ConvLayout::ChannelsFramesWidth || rows == 0 || rows > capacity {
            return Err("invalid NCFW row-pack geometry".into());
        }
        let row_width = self
            .output_channels
            .checked_mul(self.output_frames)
            .ok_or("packed row width overflows")?;
        let mut model = self.model;
        let mut builder = Builder::new(model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut model.tensors));
        let output = builder.tensor(
            "act.conv2d.rows",
            u64::from(rows) * u64::from(row_width) * 4,
        );
        let elements = rows
            .checked_mul(row_width)
            .ok_or("row-pack element count overflows")?;
        builder.emit(
            DevOp::PackNcfwRowsF32,
            repeated(model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..2].copy_from_slice(&[output, self.output]);
                instruction.i[..5].copy_from_slice(&[
                    rows,
                    self.output_channels,
                    self.output_frames,
                    self.output_width,
                    self.output_batches,
                ]);
            },
        );
        let program = model.progs.len();
        model.tensors = builder.tensors();
        model.progs.push(builder.finish());
        model.prog_t.push(rows);
        Ok(PacketPrefix {
            model,
            programs: vec![self.program, program],
            input: self.input,
            output,
            input_shape: self.input_shape,
        })
    }
}

pub fn lower(
    spec: Conv2dSpec,
    stages: &[Conv2dStage<'_>],
    n_cu: u32,
) -> Result<Conv2dPackets, String> {
    if n_cu == 0
        || spec.batches == 0
        || spec.input_frames == 0
        || spec.input_width == 0
        || spec.input_channels == 0
        || stages.is_empty()
    {
        return Err("invalid convolution geometry".into());
    }
    let mut builder = Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let input_shape = shape(
        spec.batches,
        spec.input_frames,
        spec.input_width,
        spec.input_channels,
        spec.input_layout,
    );
    let input = builder.tensor(
        "in.conv2d",
        tensor_bytes(
            spec.batches,
            spec.input_frames,
            spec.input_width,
            spec.input_channels,
            4,
        )?,
    );
    let mut source = input;
    let mut frames = spec.input_frames;
    let mut width = spec.input_width;
    let mut channels = spec.input_channels;
    let mut layout = spec.input_layout;
    let mut dependency = None;
    for (index, stage) in stages.iter().enumerate() {
        if stage.kernel == 0
            || stage.stride == 0
            || stage.input_channels != channels
            || stage.output_channels == 0
            || (stage.kind == ConvKind::Depthwise && stage.input_channels != stage.output_channels)
        {
            return Err(format!("invalid convolution stage {index}"));
        }
        let output_frames = output_extent(
            frames,
            stage.kernel,
            stage.stride,
            stage.pad_before,
            stage.pad_after,
        )?;
        let output_width = output_extent(
            width,
            stage.kernel,
            stage.stride,
            stage.pad_before,
            stage.pad_after,
        )?;
        let destination = builder.tensor(
            &format!("act.conv2d.{index}"),
            tensor_bytes(
                spec.batches,
                output_frames,
                output_width,
                stage.output_channels,
                4,
            )?,
        );
        let stored_channels = if stage.kind == ConvKind::Depthwise {
            1
        } else {
            stage.input_channels
        };
        let weight_elements = u64::from(stage.output_channels)
            .checked_mul(u64::from(stored_channels))
            .and_then(|count| count.checked_mul(u64::from(stage.kernel)))
            .and_then(|count| count.checked_mul(u64::from(stage.kernel)))
            .ok_or("convolution weight size overflows")?;
        let weight = builder.tensor(
            stage.weight,
            weight_elements
                * match stage.weight_type {
                    ConvWeight::F16 => 2,
                    ConvWeight::F32 => 4,
                },
        );
        let bias = builder.tensor(stage.bias, u64::from(stage.output_channels) * 4);
        let pointwise = stage.kind == ConvKind::Standard
            && stage.kernel == 1
            && stage.stride == 1
            && stage.pad_before == 0
            && stage.pad_after == 0
            && layout == ConvLayout::ChannelsLast;
        let tiled_3x3_nchw = stage.kind == ConvKind::Standard
            && stage.kernel == 3
            && (layout == ConvLayout::ChannelsFramesWidth
                || (layout == ConvLayout::ChannelsLast && channels == 1))
            && stage.output_layout == ConvLayout::ChannelsFramesWidth;
        let spatial = spec
            .batches
            .checked_mul(output_frames)
            .and_then(|count| count.checked_mul(output_width))
            .ok_or("convolution output size overflows")?;
        let blocks = if pointwise || tiled_3x3_nchw {
            spatial
                .div_ceil(128)
                .checked_mul(stage.output_channels.div_ceil(64))
                .ok_or("pointwise block count overflows")?
        } else {
            spatial
                .checked_mul(stage.output_channels)
                .ok_or("convolution element count overflows")?
                .div_ceil(256)
        };
        dependency = Some(builder.emit(
            DevOp::Conv2dF32,
            repeated(n_cu, blocks),
            &dependency.into_iter().collect::<Vec<_>>(),
            |instruction| {
                instruction.t[..4].copy_from_slice(&[destination, source, weight, bias]);
                instruction.i.copy_from_slice(&[
                    frames,
                    width,
                    stage.input_channels,
                    stage.output_channels,
                    stage.kernel,
                    stage.stride,
                    stage.pad_before,
                    stage.pad_after,
                ]);
                instruction.j[0] = flags(stage, layout);
                instruction.j[1] = spec.batches;
            },
        ));
        source = destination;
        frames = output_frames;
        width = output_width;
        channels = stage.output_channels;
        layout = stage.output_layout;
    }
    let tensors = builder.tensors();
    let program = builder.finish();
    Ok(Conv2dPackets {
        model: Model {
            n_cu,
            target: 0,
            tensors,
            progs: vec![program],
            prog_t: vec![spec.input_frames],
            kv_row_insts: vec![],
            gen: vec![],
        },
        program: 0,
        input,
        output: source,
        output_frames: frames,
        output_width: width,
        output_channels: channels,
        output_layout: layout,
        output_batches: spec.batches,
        input_shape,
    })
}

fn flags(stage: &Conv2dStage<'_>, input_layout: ConvLayout) -> u32 {
    u32::from(stage.kind == ConvKind::Depthwise)
        | (u32::from(stage.activation == ConvActivation::Relu) << 1)
        | (layout_code(stage.output_layout) << 2)
        | (layout_code(input_layout) << 4)
        | (u32::from(stage.weight_type == ConvWeight::F32) << 6)
        | (u32::from(stage.activation == ConvActivation::GeluErfBf16) << 7)
}

fn layout_code(layout: ConvLayout) -> u32 {
    match layout {
        ConvLayout::ChannelsLast => 0,
        ConvLayout::FrameChannelsWidth => 1,
        ConvLayout::ChannelsFramesWidth => 2,
    }
}

fn shape(batches: u32, frames: u32, width: u32, channels: u32, layout: ConvLayout) -> Vec<u64> {
    let (b, f, w, c) = (batches as u64, frames as u64, width as u64, channels as u64);
    match (batches, layout) {
        (1, ConvLayout::ChannelsLast) => vec![f, w, c],
        (1, ConvLayout::FrameChannelsWidth) => vec![f, c, w],
        (1, ConvLayout::ChannelsFramesWidth) => vec![c, f, w],
        (_, ConvLayout::ChannelsLast) => vec![b, f, w, c],
        (_, ConvLayout::FrameChannelsWidth) => vec![b, f, c, w],
        (_, ConvLayout::ChannelsFramesWidth) => vec![b, c, f, w],
    }
}

pub(crate) fn output_extent(
    input: u32,
    kernel: u32,
    stride: u32,
    pad_before: u32,
    pad_after: u32,
) -> Result<u32, String> {
    input
        .checked_add(pad_before)
        .and_then(|value| value.checked_add(pad_after))
        .and_then(|value| value.checked_sub(kernel))
        .map(|value| value / stride + 1)
        .ok_or_else(|| "convolution extent overflows".into())
}

fn tensor_bytes(
    batches: u32,
    frames: u32,
    width: u32,
    channels: u32,
    element_bytes: u32,
) -> Result<u64, String> {
    [batches, frames, width, channels, element_bytes]
        .into_iter()
        .try_fold(1u64, |count, value| count.checked_mul(u64::from(value)))
        .ok_or_else(|| "tensor size overflows".into())
}

fn repeated(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks.max(1)).map(|index| index % n_cu).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_batched_nchw_chain() {
        let packets = lower(
            Conv2dSpec {
                batches: 3,
                input_frames: 8,
                input_width: 10,
                input_channels: 1,
                input_layout: ConvLayout::ChannelsFramesWidth,
            },
            &[Conv2dStage {
                weight: "conv.weight",
                bias: "conv.bias",
                kernel: 3,
                stride: 2,
                pad_before: 1,
                pad_after: 1,
                input_channels: 1,
                output_channels: 16,
                kind: ConvKind::Standard,
                activation: ConvActivation::GeluErfBf16,
                output_layout: ConvLayout::ChannelsFramesWidth,
                weight_type: ConvWeight::F32,
            }],
            4,
        )
        .unwrap();
        assert_eq!((packets.output_frames, packets.output_width), (4, 5));
        let inst = &packets.model.progs[0].insts[0];
        assert_eq!(inst.j, [2 << 4 | 2 << 2 | 1 << 6 | 1 << 7, 3]);
        let prefix = packets.pack_ncfw_rows(15).unwrap();
        assert_eq!(prefix.input_shape, [3, 1, 8, 10]);
        assert_eq!(prefix.programs, [0, 1]);
        assert_eq!(prefix.model.progs[1].insts[0].i[..4], [15, 16, 4, 5]);
    }

    #[test]
    fn tiles_long_mono_input_when_output_is_nchw() {
        let packets = lower(
            Conv2dSpec {
                batches: 1,
                input_frames: 3000,
                input_width: 128,
                input_channels: 1,
                input_layout: ConvLayout::ChannelsLast,
            },
            &[Conv2dStage {
                weight: "conv.weight",
                bias: "conv.bias",
                kernel: 3,
                stride: 2,
                pad_before: 2,
                pad_after: 1,
                input_channels: 1,
                output_channels: 256,
                kind: ConvKind::Standard,
                activation: ConvActivation::Relu,
                output_layout: ConvLayout::ChannelsFramesWidth,
                weight_type: ConvWeight::F16,
            }],
            16,
        )
        .unwrap();
        assert_eq!((packets.output_frames, packets.output_width), (1501, 65));
        assert_eq!(packets.model.progs[0].insts[0].blocks, 3052);
        assert_eq!(packets.model.progs[0].insts[0].j[0], 2 << 2 | 1 << 1);
    }
}
