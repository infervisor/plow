//! Backend-neutral Conformer lowering to the Plow device ISA.

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Model};

use crate::pipeline::PacketPrefix;

#[derive(Clone, Copy, Debug)]
pub struct NormWeights<'a> {
    pub gamma: &'a str,
    pub beta: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct FeedForwardWeights<'a> {
    pub norm: NormWeights<'a>,
    pub expand: &'a str,
    pub contract: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionWeights<'a> {
    pub norm: NormWeights<'a>,
    pub query: &'a str,
    pub key: &'a str,
    pub value: &'a str,
    pub position: &'a str,
    pub output: &'a str,
    pub bias_u: &'a str,
    pub bias_v: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct ConvolutionWeights<'a> {
    pub norm: NormWeights<'a>,
    pub pointwise_in: &'a str,
    pub depthwise: &'a str,
    pub channel_norm: NormWeights<'a>,
    pub pointwise_out: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct ConformerLayerWeights<'a> {
    pub feed_forward1: FeedForwardWeights<'a>,
    pub attention: AttentionWeights<'a>,
    pub convolution: ConvolutionWeights<'a>,
    pub feed_forward2: FeedForwardWeights<'a>,
    pub output_norm: NormWeights<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct ConformerSpec<'a> {
    pub frames: u32,
    pub width: u32,
    pub feed_forward_width: u32,
    pub heads: u32,
    pub convolution_kernel: u32,
    pub chunk_size: u32,
    pub left_chunks: u32,
    pub position_table: &'a str,
    pub position_count: u32,
    pub position_center: u32,
    pub epsilon: f32,
}

pub struct ConformerPackets {
    pub model: Model,
    pub programs: Vec<usize>,
    pub input: u32,
    pub output: u32,
    input_shape: Vec<u64>,
    frames: u32,
    width: u32,
}

impl ConformerPackets {
    pub fn into_prefix(self) -> PacketPrefix {
        PacketPrefix {
            model: self.model,
            programs: self.programs,
            input: self.input,
            output: self.output,
            input_shape: self.input_shape,
        }
    }

    /// Materialize checkpoint tensors into a self-contained packet asset. This is useful for
    /// formats such as GGUF until their generic name-based runtime binder is available.
    pub fn embed_weights(
        &mut self,
        mut resolve: impl FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<(), String> {
        for tensor in &mut self.model.tensors {
            if packet::names::is_checkpoint_weight(&tensor.name) && tensor.init.is_none() {
                let bytes = resolve(&tensor.name)?;
                if bytes.len() as u64 != tensor.bytes {
                    return Err(format!(
                        "tensor {} has {} bytes, expected {}",
                        tensor.name,
                        bytes.len(),
                        tensor.bytes
                    ));
                }
                tensor.init = Some(bytes);
            }
        }
        Ok(())
    }

    pub fn pipeline_section(&self) -> Result<packet::devbuild::SectionData, String> {
        use plow_asset::packet_pipeline::{
            PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
        };
        use std::collections::BTreeMap;

        let input = self
            .model
            .tensors
            .get(self.input as usize)
            .ok_or("Conformer input tensor is missing")?;
        let output = self
            .model
            .tensors
            .get(self.output as usize)
            .ok_or("Conformer output tensor is missing")?;
        let programs = self
            .programs
            .iter()
            .enumerate()
            .map(|(stage, &program)| (format!("forward.{stage}"), program as u32))
            .collect();
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                name: "encode".into(),
                driver: "forward.v1".into(),
                programs,
                tensors: BTreeMap::from([
                    (
                        "input".into(),
                        PipelineTensor {
                            name: input.name.clone(),
                            dtype: PipelineDType::F32,
                            shape: self.input_shape.clone(),
                        },
                    ),
                    (
                        "output".into(),
                        PipelineTensor {
                            name: output.name.clone(),
                            dtype: PipelineDType::F32,
                            shape: vec![self.frames as u64, self.width as u64],
                        },
                    ),
                ]),
                parameters: BTreeMap::from([
                    ("frames".into(), self.frames as u64),
                    ("width".into(), self.width as u64),
                    ("forward_program_count".into(), self.programs.len() as u64),
                ]),
            }],
        };
        metadata.validate(self.model.progs.len(), |name| {
            self.model
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .map(|tensor| tensor.bytes)
        })?;
        Ok(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
        })
    }
}

pub fn lower(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    n_cu: u32,
) -> Result<ConformerPackets, String> {
    lower_inner(spec, layers, n_cu, None)
}

pub fn append(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    prefix: PacketPrefix,
) -> Result<ConformerPackets, String> {
    let n_cu = prefix.model.n_cu;
    lower_inner(spec, layers, n_cu, Some(prefix))
}

fn lower_inner(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    n_cu: u32,
    prefix: Option<PacketPrefix>,
) -> Result<ConformerPackets, String> {
    validate(spec, layers, n_cu)?;
    let mut b = Builder::new(n_cu);
    b.set_tensor_dedup(true);
    let fw = u64::from(spec.frames) * u64::from(spec.width) * 4;
    let (
        input,
        input_shape,
        io,
        mut programs,
        mut program_t,
        mut pipeline_programs,
        target,
        kv_row_insts,
        gen,
    ) = if let Some(prefix) = prefix {
        let input_bytes = prefix
            .input_shape
            .iter()
            .try_fold(4u64, |bytes, &dim| bytes.checked_mul(dim));
        if prefix.programs.is_empty()
            || prefix
                .programs
                .iter()
                .any(|&program| program >= prefix.model.progs.len())
            || prefix.input as usize >= prefix.model.tensors.len()
            || prefix.output as usize >= prefix.model.tensors.len()
            || prefix.model.tensors[prefix.output as usize].bytes != fw
            || prefix.input_shape.is_empty()
            || prefix.input_shape.contains(&0)
            || input_bytes != Some(prefix.model.tensors[prefix.input as usize].bytes)
        {
            return Err("invalid Conformer packet prefix".into());
        }
        let input = prefix.input;
        let io = prefix.output;
        let input_shape = prefix.input_shape;
        let Model {
            n_cu: _,
            target,
            tensors,
            progs,
            prog_t,
            kv_row_insts,
            gen,
        } = prefix.model;
        b.adopt_tensors(tensors);
        (
            input,
            input_shape,
            io,
            progs,
            prog_t,
            prefix.programs,
            target,
            kv_row_insts,
            gen,
        )
    } else {
        let io = b.tensor("act.asr.io", fw);
        (
            io,
            vec![spec.frames as u64, spec.width as u64],
            io,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            Vec::new(),
        )
    };
    let large_width = spec.feed_forward_width.max(spec.width * 2);
    let large_bytes = u64::from(spec.frames) * u64::from(large_width) * 4;
    let position_rows = spec.frames * 2 - 1;
    let position_bytes = u64::from(position_rows) * u64::from(spec.width) * 4;
    let d0 = b.tensor("act.asr.d0", fw);
    let d1 = b.tensor("act.asr.d1", fw);
    let large = b.tensor("act.asr.large", large_bytes);
    let q = b.tensor("act.asr.q", fw);
    let k = b.tensor("act.asr.k", fw);
    let v = b.tensor("act.asr.v", fw);
    let position = b.tensor("act.asr.position", position_bytes);
    let position_table = b.tensor(
        spec.position_table,
        u64::from(spec.position_count) * u64::from(spec.width) * 4,
    );
    let mut dep = None;
    for layer in layers {
        dep = Some(feed_forward(&mut b, spec, *layer, io, d0, large, dep, true));
        let norm = emit_norm(&mut b, spec, io, d0, layer.attention.norm, dep);
        let pos = emit_q8(
            &mut b,
            position,
            position_table,
            layer.attention.position,
            position_rows,
            spec.width,
            spec.width,
            0,
            spec.position_center - spec.frames,
            dep,
        );
        let qd = emit_q8(
            &mut b,
            q,
            d0,
            layer.attention.query,
            spec.frames,
            spec.width,
            spec.width,
            0,
            0,
            Some(norm),
        );
        let kd = emit_q8(
            &mut b,
            k,
            d0,
            layer.attention.key,
            spec.frames,
            spec.width,
            spec.width,
            0,
            0,
            Some(norm),
        );
        let vd = emit_q8(
            &mut b,
            v,
            d0,
            layer.attention.value,
            spec.frames,
            spec.width,
            spec.width,
            0,
            0,
            Some(norm),
        );
        let bu = b.tensor(layer.attention.bias_u, u64::from(spec.width) * 4);
        let bv = b.tensor(layer.attention.bias_v, u64::from(spec.width) * 4);
        let attention = b.emit(
            DevOp::RelativeAttentionF32,
            repeated_cus(n_cu, (spec.frames * spec.heads).div_ceil(8)),
            &[qd, kd, vd, pos],
            |d| {
                d.t[..7].copy_from_slice(&[d1, q, k, v, position, bu, bv]);
                d.i[..5].copy_from_slice(&[
                    spec.frames,
                    spec.width,
                    spec.heads,
                    spec.chunk_size,
                    spec.left_chunks,
                ]);
            },
        );
        let projected = emit_q8(
            &mut b,
            d0,
            d1,
            layer.attention.output,
            spec.frames,
            spec.width,
            spec.width,
            0,
            0,
            Some(attention),
        );
        dep = Some(emit_add(&mut b, spec, io, io, d0, 1.0, Some(projected)));

        let norm = emit_norm(&mut b, spec, io, d0, layer.convolution.norm, dep);
        let pointwise = emit_q8(
            &mut b,
            large,
            d0,
            layer.convolution.pointwise_in,
            spec.frames,
            spec.width * 2,
            spec.width,
            0,
            0,
            Some(norm),
        );
        let glu = b.emit(DevOp::GluF32, b.all(), &[pointwise], |d| {
            d.t[0] = d1;
            d.t[1] = large;
            d.i[0] = spec.frames;
            d.i[1] = spec.width;
        });
        let depthwise_weight = b.tensor(
            layer.convolution.depthwise,
            u64::from(spec.width) * u64::from(spec.convolution_kernel) * 2,
        );
        let depthwise = b.emit(DevOp::CausalDepthwiseConv1dF32, b.all(), &[glu], |d| {
            d.t[..3].copy_from_slice(&[d0, d1, depthwise_weight]);
            d.i[..3].copy_from_slice(&[spec.frames, spec.width, spec.convolution_kernel]);
        });
        let channel_norm = emit_norm(
            &mut b,
            spec,
            d0,
            d1,
            layer.convolution.channel_norm,
            Some(depthwise),
        );
        let activated = b.emit(DevOp::SiluF32, b.all(), &[channel_norm], |d| {
            d.t[..2].copy_from_slice(&[d1, d1]);
            d.i[0] = spec.frames * spec.width;
        });
        let convolution = emit_q8(
            &mut b,
            d0,
            d1,
            layer.convolution.pointwise_out,
            spec.frames,
            spec.width,
            spec.width,
            0,
            0,
            Some(activated),
        );
        dep = Some(emit_add(&mut b, spec, io, io, d0, 1.0, Some(convolution)));
        dep = Some(feed_forward(
            &mut b, spec, *layer, io, d0, large, dep, false,
        ));
        dep = Some(emit_norm(&mut b, spec, io, io, layer.output_norm, dep));
    }
    let tensors = b.tensors();
    let program = b.finish();
    let conformer_program = programs.len();
    programs.push(program);
    program_t.push(spec.frames);
    pipeline_programs.push(conformer_program);
    Ok(ConformerPackets {
        model: Model {
            n_cu,
            target,
            tensors,
            progs: programs,
            prog_t: program_t,
            kv_row_insts,
            gen,
        },
        programs: pipeline_programs,
        input,
        output: io,
        input_shape,
        frames: spec.frames,
        width: spec.width,
    })
}

fn feed_forward(
    b: &mut Builder,
    spec: ConformerSpec<'_>,
    layer: ConformerLayerWeights<'_>,
    io: u32,
    normalized: u32,
    hidden: u32,
    dep: Option<u32>,
    first: bool,
) -> u32 {
    let weights = if first {
        layer.feed_forward1
    } else {
        layer.feed_forward2
    };
    let norm = emit_norm(b, spec, io, normalized, weights.norm, dep);
    let expand = emit_q8(
        b,
        hidden,
        normalized,
        weights.expand,
        spec.frames,
        spec.feed_forward_width,
        spec.width,
        1,
        0,
        Some(norm),
    );
    let contract = emit_q8(
        b,
        normalized,
        hidden,
        weights.contract,
        spec.frames,
        spec.width,
        spec.feed_forward_width,
        0,
        0,
        Some(expand),
    );
    emit_add(b, spec, io, io, normalized, 0.5, Some(contract))
}

fn emit_norm(
    b: &mut Builder,
    spec: ConformerSpec<'_>,
    input: u32,
    output: u32,
    weights: NormWeights<'_>,
    dep: Option<u32>,
) -> u32 {
    let gamma = b.tensor(weights.gamma, u64::from(spec.width) * 4);
    let beta = b.tensor(weights.beta, u64::from(spec.width) * 4);
    b.emit(
        DevOp::LayerNormF32,
        repeated_cus(b.n_cu(), spec.frames),
        &deps(dep),
        |d| {
            d.t[..4].copy_from_slice(&[output, input, gamma, beta]);
            d.i[0] = spec.frames;
            d.i[1] = spec.width;
            d.f[0] = spec.epsilon;
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_q8(
    b: &mut Builder,
    output: u32,
    input: u32,
    weight_name: &str,
    m: u32,
    n: u32,
    k: u32,
    activation: u32,
    a_row0: u32,
    dep: Option<u32>,
) -> u32 {
    let weight = b.tensor(weight_name, u64::from(n) * u64::from(k / 32) * 34);
    let tiles = m.div_ceil(128) * n.div_ceil(64);
    b.emit(
        DevOp::Q8GemmF32,
        repeated_cus(b.n_cu(), tiles),
        &deps(dep),
        |d| {
            d.t[..4].copy_from_slice(&[output, input, weight, TENSOR_NONE]);
            d.i[..5].copy_from_slice(&[m, n, k, activation, a_row0]);
        },
    )
}

fn emit_add(
    b: &mut Builder,
    spec: ConformerSpec<'_>,
    output: u32,
    a: u32,
    other: u32,
    scale: f32,
    dep: Option<u32>,
) -> u32 {
    b.emit(DevOp::ScaledAddF32, b.all(), &deps(dep), |d| {
        d.t[..3].copy_from_slice(&[output, a, other]);
        d.i[0] = spec.frames * spec.width;
        d.f[0] = scale;
    })
}

fn deps(dep: Option<u32>) -> Vec<u32> {
    dep.into_iter().collect()
}

fn repeated_cus(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks).map(|block| block % n_cu).collect()
}

fn validate(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    n_cu: u32,
) -> Result<(), String> {
    if n_cu == 0
        || spec.frames == 0
        || spec.width == 0
        || spec.feed_forward_width == 0
        || spec.heads == 0
        || spec.chunk_size == 0
        || spec.convolution_kernel == 0
        || layers.is_empty()
    {
        return Err("Conformer geometry must be non-zero".into());
    }
    if spec.width % spec.heads != 0 || spec.width % 32 != 0 || spec.feed_forward_width % 32 != 0 {
        return Err("Conformer projection geometry must divide heads and Q8_0 blocks".into());
    }
    let window = spec
        .chunk_size
        .checked_mul(spec.left_chunks + 1)
        .ok_or("attention window overflows")?;
    if window > 64 {
        return Err("relative attention packet currently supports at most 64 keys".into());
    }
    if spec.position_center < spec.frames
        || spec.position_center + spec.frames > spec.position_count + 1
    {
        return Err("relative position table does not cover the frame count".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: NormWeights<'static> = NormWeights {
        gamma: "encoder.norm.weight",
        beta: "encoder.norm.bias",
    };

    #[test]
    fn lowering_emits_only_generic_packet_ops() {
        let ff = FeedForwardWeights {
            norm: N,
            expand: "encoder.ff.expand",
            contract: "encoder.ff.contract",
        };
        let layer = ConformerLayerWeights {
            feed_forward1: ff,
            attention: AttentionWeights {
                norm: N,
                query: "encoder.att.q",
                key: "encoder.att.k",
                value: "encoder.att.v",
                position: "encoder.att.p",
                output: "encoder.att.o",
                bias_u: "encoder.att.u",
                bias_v: "encoder.att.vbias",
            },
            convolution: ConvolutionWeights {
                norm: N,
                pointwise_in: "encoder.conv.in",
                depthwise: "encoder.conv.dw",
                channel_norm: N,
                pointwise_out: "encoder.conv.out",
            },
            feed_forward2: ff,
            output_norm: N,
        };
        let packets = lower(
            ConformerSpec {
                frames: 8,
                width: 32,
                feed_forward_width: 64,
                heads: 2,
                convolution_kernel: 3,
                chunk_size: 4,
                left_chunks: 1,
                position_table: "encoder.pos",
                position_count: 31,
                position_center: 15,
                epsilon: 1e-5,
            },
            &[layer],
            4,
        )
        .unwrap();
        let ops: Vec<_> = packets.model.progs[0]
            .insts
            .iter()
            .map(|d| DevOp::from_u16(d.op).unwrap())
            .collect();
        assert_eq!(ops.len(), 25);
        assert!(ops.iter().all(|op| matches!(
            op,
            DevOp::Q8GemmF32
                | DevOp::LayerNormF32
                | DevOp::ScaledAddF32
                | DevOp::GluF32
                | DevOp::CausalDepthwiseConv1dF32
                | DevOp::RelativeAttentionF32
                | DevOp::SiluF32
        )));
        assert_eq!(packets.input, packets.output);
        let section = packets.pipeline_section().unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        assert_eq!(metadata.pipelines[0].driver, "forward.v1");
        assert_eq!(metadata.pipelines[0].parameters["frames"], 8);
    }
}
