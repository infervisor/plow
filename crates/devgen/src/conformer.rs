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
    /// FP16 `[channels, kernel]` for a causal convolution; with `depthwise_bias`, FP32
    /// `[channels, 1, kernel]` with the channel norm (batch norm) already folded in.
    pub depthwise: &'a str,
    pub channel_norm: NormWeights<'a>,
    /// FP32 `[channels]`: set for a non-causal convolution whose batch norm is folded into
    /// `depthwise` (`channel_norm` is then unused).
    pub depthwise_bias: Option<&'a str>,
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
    /// Causal depthwise convolution (zero left pad, layer-norm channel norm); otherwise the
    /// convolution is centred and its batch norm is folded (`ConvolutionWeights::depthwise_bias`).
    pub causal_convolution: bool,
    pub chunk_size: u32,
    /// `u32::MAX`: full-context attention (every key; `chunk_size` is then ignored).
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
                strings: Default::default(),
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
    // Full context: the host writes the valid frame count of the padded bucket, which masks the
    // attention keys and the centred convolution's input past it.
    let valid_rows = (spec.left_chunks == u32::MAX).then(|| b.tensor(VALID_ROWS, 4));
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
        // Full context runs a block per (row, head); the chunked body a warp per (row, head).
        let attention_blocks = if spec.left_chunks == u32::MAX {
            spec.frames * spec.heads
        } else {
            (spec.frames * spec.heads).div_ceil(8)
        };
        let attention = b.emit(
            DevOp::RelativeAttentionF32,
            repeated_cus(n_cu, attention_blocks),
            &[qd, kd, vd, pos],
            |d| {
                d.t[..8].copy_from_slice(&[d1, q, k, v, position, bu, bv, valid_rows.unwrap_or(TENSOR_NONE)]);
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
        let (activated, activated_rows) = if spec.causal_convolution {
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
            (activated, d1)
        } else {
            // `large` is free once the GLU has read it.
            (emit_centred_depthwise(&mut b, spec, layer.convolution, large, d1, glu, valid_rows), large)
        };
        let convolution = emit_q8(
            &mut b,
            d0,
            activated_rows,
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

/// Centred depthwise convolution with its batch norm folded into `weights.depthwise` (FP32
/// `[channels, 1, kernel]`) and `weights.depthwise_bias`, then SiLU: one [`DevOp::Conv1dF32`].
/// With `valid_rows`, input rows past it are padding (zeros), as the reference masks them.
fn emit_centred_depthwise(
    b: &mut Builder,
    spec: ConformerSpec<'_>,
    weights: ConvolutionWeights<'_>,
    output: u32,
    input: u32,
    dep: u32,
    valid_rows: Option<u32>,
) -> u32 {
    let (k, width) = (spec.convolution_kernel, spec.width);
    let weight = b.tensor(weights.depthwise, u64::from(width) * u64::from(k) * 4);
    let bias = b.tensor(
        weights.depthwise_bias.expect("validated: a centred convolution has a folded bias"),
        u64::from(width) * 4,
    );
    let pad = (k - 1) / 2;
    // Depthwise Conv1dF32: one block per 64 output rows x 64 channels.
    let blocks = spec.frames.div_ceil(64) * width.div_ceil(64);
    b.emit(DevOp::Conv1dF32, repeated_cus(b.n_cu(), blocks), &[dep], |d| {
        d.t[..4].copy_from_slice(&[output, input, weight, bias]);
        d.t[6] = valid_rows.unwrap_or(TENSOR_NONE);
        d.i = [1, spec.frames, width, width, k, 1, 1, width];
        d.j = [pad | (pad << 16), packet::dev::ACT_SILU << 8];
    })
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
    // At most 8 rows the kernel walks columns warp by warp (SPQ_GEMV_ROWS): every CU takes a share.
    let tiles = if m <= 8 { b.n_cu() } else { m.div_ceil(64) * n.div_ceil(64) };
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

/// The NVIDIA row-parallel relative attention keeps a row's scores and probabilities in the
/// speech arena (`SPR_MAX_KEYS`); this bound stays well inside it.
pub const FULL_CONTEXT_MAX_KEYS: u32 = 4096;

/// Full-context encoders: u32 valid frame count of the padded bucket, written by the host.
pub const VALID_ROWS: &str = "in.asr.valid_rows";

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
    if spec.left_chunks == u32::MAX {
        if spec.frames > FULL_CONTEXT_MAX_KEYS {
            return Err(format!(
                "full-context relative attention supports at most {FULL_CONTEXT_MAX_KEYS} frames"
            ));
        }
    } else {
        let window = spec
            .chunk_size
            .checked_mul(spec.left_chunks + 1)
            .ok_or("attention window overflows")?;
        if window > 64 {
            return Err("chunk-limited relative attention supports at most 64 keys".into());
        }
    }
    let centred_ok = |layer: &ConformerLayerWeights<'_>| {
        layer.convolution.depthwise_bias.is_some() && spec.convolution_kernel % 2 == 1
    };
    if spec.causal_convolution && layers.iter().any(|l| l.convolution.depthwise_bias.is_some())
        || !spec.causal_convolution && !layers.iter().all(centred_ok)
    {
        return Err("a centred convolution needs an odd kernel and a folded bias per layer; a causal one none".into());
    }
    if spec.position_center < spec.frames
        || spec.position_center + spec.frames > spec.position_count + 1
    {
        return Err("relative position table does not cover the frame count".into());
    }
    Ok(())
}

/// Cache-aware streaming (chunk-limited attention, causal convolutions): the attention's left
/// context in rows. A stream step keeps this many key/value rows per layer.
pub fn stream_left_rows(spec: &ConformerSpec<'_>) -> u32 {
    spec.chunk_size * spec.left_chunks
}

/// Name of the per-step key mask: keys before row `key_start` of the attention window are not
/// filled yet (the stream's first chunks). Written by the host every step.
pub const STREAM_KEY_START: &str = "in.stream.key_start";

/// One streaming encoder step: `rows` new encoder frames (whole attention chunks) taken from rows
/// `keep_row0..` of the prefix output (the subsampling of the step's mel window) run through every
/// layer. Row-local work runs on the new rows only; each layer's attention sees a window of
/// `[key/value cache | new rows]` and its depthwise convolution `[conv cache | new rows]`, the
/// caches (`state.stream.*`, zeroed when a stream opens) shifting by `rows` per step. Every output
/// row is computed in the offline encoder's order, so the step reproduces its rows. Output:
/// `act.stream.x` (`rows` x width). Run [`stream_init`] once before the first step.
pub fn append_stream_step(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    prefix: PacketPrefix,
    keep_row0: u32,
    rows: u32,
) -> Result<PacketPrefix, String> {
    validate(spec, layers, prefix.model.n_cu)?;
    if !spec.causal_convolution || spec.left_chunks == u32::MAX {
        return Err("a cache-aware stream needs causal convolutions and chunk-limited attention".into());
    }
    let left = stream_left_rows(&spec);
    let window = left + rows;
    let tail = spec.convolution_kernel - 1;
    if rows == 0 || rows % spec.chunk_size != 0 || left == 0 || keep_row0 + rows > spec.frames {
        return Err("stream step rows must be whole attention chunks inside the prefix output".into());
    }
    if spec.position_center < window || spec.position_center + window > spec.position_count + 1 {
        return Err("relative position table does not cover the stream window".into());
    }
    let n_cu = prefix.model.n_cu;
    let mut b = Builder::new(n_cu);
    b.set_tensor_dedup(true);
    b.adopt_tensors(prefix.model.tensors.clone());
    let w = spec.width;
    let row_bytes = u64::from(w) * 4;
    let rs = ConformerSpec { frames: rows, ..spec };
    let x = b.tensor("act.stream.x", u64::from(rows) * row_bytes);
    let d0 = b.tensor("act.stream.d0", u64::from(rows) * row_bytes);
    let d1 = b.tensor("act.stream.d1", u64::from(rows) * row_bytes);
    let large = b.tensor(
        "act.stream.large",
        u64::from(rows) * u64::from(spec.feed_forward_width.max(2 * w)) * 4,
    );
    let [q, k, v] = ["q", "k", "v"].map(|n| b.tensor(&format!("act.stream.{n}"), u64::from(rows) * row_bytes));
    let [qw, kw, vw, ctx] =
        ["qw", "kw", "vw", "ctx"].map(|n| b.tensor(&format!("act.stream.{n}"), u64::from(window) * row_bytes));
    let gw = b.tensor("act.stream.gw", u64::from(tail + rows) * row_bytes);
    let cw = b.tensor("act.stream.cw", u64::from(tail + rows) * row_bytes);
    let key_start = b.tensor(STREAM_KEY_START, 4);
    let mut dep = Some(copy_rows(&mut b, x, 0, prefix.output, keep_row0, rows, w, None));
    for (l, layer) in layers.iter().enumerate() {
        let k_cache = b.tensor(&format!("state.stream.k.{l}"), u64::from(left) * row_bytes);
        let v_cache = b.tensor(&format!("state.stream.v.{l}"), u64::from(left) * row_bytes);
        let conv_cache = b.tensor(&format!("state.stream.conv.{l}"), u64::from(tail) * row_bytes);
        let position = b.tensor(&stream_position(l), u64::from(2 * window - 1) * row_bytes);
        dep = Some(feed_forward(&mut b, rs, *layer, x, d0, large, dep, true));
        let norm = emit_norm(&mut b, rs, x, d0, layer.attention.norm, dep);
        let mut last = norm;
        for (out, weight) in [(q, layer.attention.query), (k, layer.attention.key), (v, layer.attention.value)] {
            last = emit_q8(&mut b, out, d0, weight, rows, w, w, 0, 0, Some(last));
        }
        last = copy_rows(&mut b, kw, 0, k_cache, 0, left, w, Some(last));
        last = copy_rows(&mut b, kw, left, k, 0, rows, w, Some(last));
        last = copy_rows(&mut b, vw, 0, v_cache, 0, left, w, Some(last));
        last = copy_rows(&mut b, vw, left, v, 0, rows, w, Some(last));
        last = copy_rows(&mut b, qw, left, q, 0, rows, w, Some(last));
        let bu = b.tensor(layer.attention.bias_u, u64::from(w) * 4);
        let bv = b.tensor(layer.attention.bias_v, u64::from(w) * 4);
        last = b.emit(
            DevOp::RelativeAttentionF32,
            // query_row0 > 0: a block per (new row, head).
            repeated_cus(n_cu, rows * spec.heads),
            &[last],
            |d| {
                d.t[..8].copy_from_slice(&[ctx, qw, kw, vw, position, bu, bv, key_start]);
                d.i[..6].copy_from_slice(&[window, w, spec.heads, spec.chunk_size, spec.left_chunks, left]);
            },
        );
        last = copy_rows(&mut b, k_cache, 0, kw, rows, left, w, Some(last));
        last = copy_rows(&mut b, v_cache, 0, vw, rows, left, w, Some(last));
        last = copy_rows(&mut b, d1, 0, ctx, left, rows, w, Some(last));
        last = emit_q8(&mut b, d0, d1, layer.attention.output, rows, w, w, 0, 0, Some(last));
        dep = Some(emit_add(&mut b, rs, x, x, d0, 1.0, Some(last)));

        let norm = emit_norm(&mut b, rs, x, d0, layer.convolution.norm, dep);
        let pointwise = emit_q8(&mut b, large, d0, layer.convolution.pointwise_in, rows, w * 2, w, 0, 0, Some(norm));
        let glu = b.emit(DevOp::GluF32, b.all(), &[pointwise], |d| {
            d.t[0] = d1;
            d.t[1] = large;
            d.i[0] = rows;
            d.i[1] = w;
        });
        last = copy_rows(&mut b, gw, 0, conv_cache, 0, tail, w, Some(glu));
        last = copy_rows(&mut b, gw, tail, d1, 0, rows, w, Some(last));
        let depthwise_weight = b.tensor(
            layer.convolution.depthwise,
            u64::from(w) * u64::from(spec.convolution_kernel) * 2,
        );
        last = b.emit(DevOp::CausalDepthwiseConv1dF32, b.all(), &[last], |d| {
            d.t[..3].copy_from_slice(&[cw, gw, depthwise_weight]);
            d.i[..3].copy_from_slice(&[tail + rows, w, spec.convolution_kernel]);
        });
        last = copy_rows(&mut b, conv_cache, 0, gw, rows, tail, w, Some(last));
        last = copy_rows(&mut b, d0, 0, cw, tail, rows, w, Some(last));
        let channel_norm = emit_norm(&mut b, rs, d0, d1, layer.convolution.channel_norm, Some(last));
        let activated = b.emit(DevOp::SiluF32, b.all(), &[channel_norm], |d| {
            d.t[..2].copy_from_slice(&[d1, d1]);
            d.i[0] = rows * w;
        });
        let convolution = emit_q8(&mut b, d0, d1, layer.convolution.pointwise_out, rows, w, w, 0, 0, Some(activated));
        dep = Some(emit_add(&mut b, rs, x, x, d0, 1.0, Some(convolution)));
        dep = Some(feed_forward(&mut b, rs, *layer, x, d0, large, dep, false));
        dep = Some(emit_norm(&mut b, rs, x, x, layer.output_norm, dep));
    }
    let tensors = b.tensors();
    let program = b.finish();
    let mut model = prefix.model;
    model.tensors = tensors;
    let index = model.progs.len();
    model.progs.push(program);
    model.prog_t.push(rows);
    let mut programs = prefix.programs;
    programs.push(index);
    Ok(PacketPrefix { model, programs, input: prefix.input, output: x, input_shape: prefix.input_shape })
}

/// The stream's per-layer relative-position projections: input independent, so one program fills
/// them once per packet load (`window` = left context + step rows).
pub fn stream_init(
    spec: ConformerSpec<'_>,
    layers: &[ConformerLayerWeights<'_>],
    rows: u32,
    n_cu: u32,
) -> Result<Model, String> {
    if !spec.causal_convolution || spec.left_chunks == u32::MAX {
        return Err("a cache-aware stream needs causal convolutions and chunk-limited attention".into());
    }
    let window = stream_left_rows(&spec) + rows;
    if spec.position_center < window || spec.position_center + window > spec.position_count + 1 {
        return Err("relative position table does not cover the stream window".into());
    }
    let mut b = Builder::new(n_cu);
    b.set_tensor_dedup(true);
    let table = b.tensor(spec.position_table, u64::from(spec.position_count) * u64::from(spec.width) * 4);
    let mut dep = None;
    for (l, layer) in layers.iter().enumerate() {
        let position = b.tensor(&stream_position(l), u64::from(2 * window - 1) * u64::from(spec.width) * 4);
        dep = Some(emit_q8(
            &mut b,
            position,
            table,
            layer.attention.position,
            2 * window - 1,
            spec.width,
            spec.width,
            0,
            spec.position_center - window,
            dep,
        ));
    }
    let tensors = b.tensors();
    let program = b.finish();
    Ok(Model { n_cu, target: 0, tensors, progs: vec![program], prog_t: vec![1], kv_row_insts: Vec::new(), gen: Vec::new() })
}

fn stream_position(layer: usize) -> String {
    format!("act.stream.position.{layer}")
}

/// `out[out_row0..+rows] = x[x_row0..+rows]` (`width` FP32 columns per row).
#[allow(clippy::too_many_arguments)]
fn copy_rows(b: &mut Builder, out: u32, out_row0: u32, x: u32, x_row0: u32, rows: u32, width: u32, dep: Option<u32>) -> u32 {
    b.emit(DevOp::CopyColsF32, b.all(), &deps(dep), |d| {
        d.t[..2].copy_from_slice(&[out, x]);
        d.i[..7].copy_from_slice(&[1, rows, width, width, x_row0 * width, width, out_row0 * width]);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: NormWeights<'static> = NormWeights {
        gamma: "encoder.norm.weight",
        beta: "encoder.norm.bias",
    };

    fn layer() -> ConformerLayerWeights<'static> {
        let ff = FeedForwardWeights {
            norm: N,
            expand: "encoder.ff.expand",
            contract: "encoder.ff.contract",
        };
        ConformerLayerWeights {
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
                depthwise_bias: None,
                pointwise_out: "encoder.conv.out",
            },
            feed_forward2: ff,
            output_norm: N,
        }
    }

    fn spec() -> ConformerSpec<'static> {
        ConformerSpec {
            frames: 8,
            width: 32,
            feed_forward_width: 64,
            heads: 2,
            convolution_kernel: 3,
            causal_convolution: true,
            chunk_size: 4,
            left_chunks: 1,
            position_table: "encoder.pos",
            position_count: 31,
            position_center: 15,
            epsilon: 1e-5,
        }
    }

    #[test]
    fn full_context_centred_convolution_folds_the_norm_into_one_conv1d() {
        let mut layer = layer();
        layer.convolution.depthwise = "encoder.conv.dw.folded";
        layer.convolution.depthwise_bias = Some("encoder.conv.dw.folded_bias");
        let spec = ConformerSpec { causal_convolution: false, left_chunks: u32::MAX, ..spec() };
        let packets = lower(spec, &[layer], 4).unwrap();
        let insts = &packets.model.progs[0].insts;
        let ops: Vec<_> = insts.iter().map(|d| DevOp::from_u16(d.op).unwrap()).collect();
        assert_eq!(ops.len(), 23);
        assert!(!ops.iter().any(|op| matches!(op, DevOp::CausalDepthwiseConv1dF32 | DevOp::SiluF32)));
        let conv = insts.iter().find(|d| d.op == DevOp::Conv1dF32 as u16).unwrap();
        assert_eq!(conv.i, [1, 8, 32, 32, 3, 1, 1, 32]);
        assert_eq!(conv.j, [1 | (1 << 16), packet::dev::ACT_SILU << 8]);
        let attention = insts.iter().find(|d| d.op == DevOp::RelativeAttentionF32 as u16).unwrap();
        assert_eq!(attention.i[4], u32::MAX);
        // One host-written valid-row count masks both the attention keys and the conv input.
        let valid = attention.t[7];
        assert_eq!(packets.model.tensors[valid as usize].name, VALID_ROWS);
        assert_eq!(conv.t[6], valid);
        // A stream over full context, a window wider than the kernel's key bound, and a centred
        // convolution without a folded bias are all refused.
        let prefix = lower(self::spec(), &[self::layer()], 4).unwrap().into_prefix();
        assert!(append_stream_step(spec, &[layer], prefix, 0, 4).is_err());
        let max = FULL_CONTEXT_MAX_KEYS;
        let wide = ConformerSpec { frames: max + 1, position_count: 2 * max + 1, position_center: max + 1, ..spec };
        assert!(lower(wide, &[layer], 4).is_err());
        layer.convolution.depthwise_bias = None;
        assert!(lower(spec, &[layer], 4).is_err());
    }

    #[test]
    fn lowering_emits_only_generic_packet_ops() {
        let packets = lower(spec(), &[layer()], 4).unwrap();
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
