use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

#[derive(Clone, Copy)]
pub struct CausalPipelineTensors {
    pub tokens: u32,
    pub positions: u32,
    pub kv_lengths: u32,
    pub overlay: Option<u32>,
    pub overlay_index: Option<u32>,
}

#[derive(Clone, Copy)]
pub struct CausalPipelineSpec<'a> {
    pub name: &'a str,
    pub max_context: u32,
    pub hidden: u32,
    pub decode_capacity: u32,
    pub overlay_rows: u32,
    /// Execute dependency-bearing programs in compiler order. Backends whose normal
    /// execution is already ordered may ignore this hint.
    pub ordered_dispatch: bool,
    pub tensors: CausalPipelineTensors,
}

pub fn causal_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    use std::collections::BTreeMap;

    if spec.max_context == 0 || spec.hidden == 0 || spec.decode_capacity == 0 {
        return Err("invalid causal pipeline geometry".into());
    }
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    if prefill_count == 0 || prefill_count == model.progs.len() {
        return Err("causal pipeline requires prefill and decode programs".into());
    }
    let mut programs = BTreeMap::new();
    for (program, &rows) in model.prog_t.iter().enumerate() {
        let phase = if program < prefill_count {
            "prefill"
        } else {
            "decode"
        };
        if programs
            .insert(format!("{phase}.{rows}"), program as u32)
            .is_some()
        {
            return Err(format!("duplicate {phase} program for {rows} rows"));
        }
    }
    let binding = |handle: u32, dtype, shape: Vec<u64>| -> Result<PipelineTensor, String> {
        let tensor = model
            .tensors
            .get(handle as usize)
            .ok_or_else(|| format!("causal pipeline tensor handle {handle} is missing"))?;
        Ok(PipelineTensor {
            name: tensor.name.clone(),
            dtype,
            shape,
        })
    };
    let mut tensors = BTreeMap::from([
        (
            "tokens".into(),
            binding(
                spec.tensors.tokens,
                PipelineDType::U32,
                vec![u64::from(spec.max_context)],
            )?,
        ),
        (
            "positions".into(),
            binding(
                spec.tensors.positions,
                PipelineDType::U32,
                vec![u64::from(spec.max_context)],
            )?,
        ),
        (
            "kv_lengths".into(),
            binding(
                spec.tensors.kv_lengths,
                PipelineDType::U32,
                vec![u64::from(spec.decode_capacity)],
            )?,
        ),
    ]);
    match (spec.tensors.overlay, spec.tensors.overlay_index) {
        (Some(overlay), Some(index)) if spec.overlay_rows > 0 => {
            tensors.insert(
                "overlay".into(),
                binding(
                    overlay,
                    PipelineDType::F32,
                    vec![u64::from(spec.overlay_rows), u64::from(spec.hidden)],
                )?,
            );
            tensors.insert(
                "overlay_index".into(),
                binding(index, PipelineDType::U32, vec![u64::from(spec.max_context)])?,
            );
        }
        (None, None) if spec.overlay_rows == 0 => {}
        _ => return Err("causal overlay tensors and geometry disagree".into()),
    }
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            strings: Default::default(),
            name: spec.name.into(),
            driver: "causal.v1".into(),
            programs,
            tensors,
            parameters: BTreeMap::from([
                ("max_context".into(), u64::from(spec.max_context)),
                ("hidden".into(), u64::from(spec.hidden)),
                ("decode_capacity".into(), u64::from(spec.decode_capacity)),
                ("overlay_rows".into(), u64::from(spec.overlay_rows)),
                (
                    "ordered_dispatch".into(),
                    u64::from(spec.ordered_dispatch),
                ),
                ("prefill_program_count".into(), prefill_count as u64),
                (
                    "decode_program_count".into(),
                    (model.progs.len() - prefill_count) as u64,
                ),
            ]),
        }],
    };
    metadata.validate(model.progs.len(), |name| {
        model
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

pub struct PacketPrefix {
    pub model: Model,
    pub programs: Vec<usize>,
    pub input: u32,
    pub output: u32,
    pub input_shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseActivation {
    None,
    Relu,
    GeluErfBf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseWeight {
    F32,
    Bf16,
}

#[derive(Clone, Copy)]
pub struct DenseF32Stage<'a> {
    pub output: &'a str,
    pub weight: &'a str,
    pub bias: Option<&'a str>,
    pub input_width: u32,
    pub output_width: u32,
    pub activation: DenseActivation,
    pub round_bf16: bool,
    pub weight_type: DenseWeight,
    /// The input holds only bf16-representable values (its producer rounds to bf16). With bf16
    /// weights the GEMM then runs on bf16 tensor cores with exact products.
    pub input_bf16_exact: bool,
    /// Normalize the input on load (DenseGemmF32 LayerNorm prologue).
    pub layer_norm: Option<DenseLayerNorm<'a>>,
    pub split: DenseSplit,
}

/// How a tensor-core DenseGemmF32 splits K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseSplit {
    /// Over idle blocks while the tiles do not fill the machine (the order depends on `rows`).
    Parallel,
    /// Every row sums K as a `Parallel` launch of `rows[row]` rows would (a u32 per row; flag
    /// 128): rows packed from several requests give each request's own bits.
    Reference(u32),
}

/// DenseGemmF32's LayerNorm prologue (flag bit 5): `A' = (A - mean) * inv * gamma + beta` with
/// per-row `stats` (`[rows][2]` = mean, inverse std, from RowStatsF32) — LayerNormF32's exact
/// arithmetic; `round_bf16` (flag bit 6) rounds `A'` to bf16 like LayerNormF32's flag bit 0.
#[derive(Clone, Copy)]
pub struct DenseLayerNorm<'a> {
    pub stats: u32,
    pub gamma: Option<TensorRef<'a>>,
    pub beta: Option<TensorRef<'a>>,
    pub round_bf16: bool,
}

/// RowStatsF32: LayerNormF32's per-row `(mean, inverse std)` into `[rows][2]`.
#[derive(Clone, Copy)]
pub struct RowStatsF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
    /// LayerNormF32 flag bit 1: float sums in row order (else double).
    pub ordered_statistics: bool,
}

/// DenseGemmF32 inside a [`StageProgram`] (no split-K: ops of one program may overlap).
#[derive(Clone, Copy)]
pub struct DenseF32ProgramStage<'a> {
    pub output: TensorRef<'a>,
    pub weight: TensorRef<'a>,
    pub bias: Option<TensorRef<'a>>,
    pub rows: u32,
    pub input_width: u32,
    pub output_width: u32,
    pub activation: DenseActivation,
    pub round_bf16: bool,
    pub weight_type: DenseWeight,
    /// Input (after any prologue) and f32 weights are bf16-representable: bf16 tensor cores.
    pub operands_bf16_exact: bool,
    /// Otherwise run 3xTF32 tensor cores (FP32-accurate) instead of FP32 FFMA.
    pub tf32x3: bool,
    pub layer_norm: Option<DenseLayerNorm<'a>>,
}

pub struct InitializedAddF32Stage<'a> {
    pub output: &'a str,
    pub addend: &'a str,
    pub values: &'a [f32],
    pub scale: f32,
    pub round_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct ScaledAddF32Stage<'a> {
    pub output: &'a str,
    pub scale: f32,
    pub round_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct LayerNormF32Stage<'a> {
    pub output: &'a str,
    pub gamma: Option<&'a str>,
    pub beta: Option<&'a str>,
    pub width: u32,
    pub epsilon: f32,
    pub round_bf16: bool,
    pub ordered_statistics: bool,
}

#[derive(Clone, Copy)]
pub struct GroupedAttentionF32Stage<'a> {
    pub output: &'a str,
    pub valid_rows: Option<u32>,
    /// `valid_rows` is a group table `{count, (first row, rows) x count}` (flag 8): packed
    /// utterances, each attending within its own windows.
    pub group_table: bool,
    pub width: u32,
    pub head_width: u32,
    pub group_rows: u32,
    pub round_score_bf16: bool,
    pub round_probability_bf16: bool,
    pub round_output_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct EmbedOverlayBf16Stage<'a> {
    pub output: &'a str,
    pub table: &'a str,
    pub tokens: &'a str,
    pub overlay_index: &'a str,
    pub rows: u32,
    pub width: u32,
    pub vocabulary: u32,
    pub overlay_rows: u32,
}

/// An operand that is either an existing tensor or a new one the stage declares by name (sized
/// from the stage geometry; a re-declared name keeps its handle).
#[derive(Clone, Copy, Debug)]
pub enum TensorRef<'a> {
    Handle(u32),
    Named(&'a str),
}

/// [`packet::dev`] `ACT_*` activation, shared by [`UnaryF32Stage`] and the convolution stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    Tanh,
    Sin,
    Cos,
    Exp,
    Abs,
    Sigmoid,
    Silu,
    Elu,
    LeakyRelu,
    Mish,
    GeluErf,
    Snake,
    Clamp,
    ScaleShift,
    Relu,
}

impl Activation {
    pub fn code(self) -> u32 {
        use packet::dev::*;
        match self {
            Activation::None => ACT_NONE,
            Activation::Tanh => ACT_TANH,
            Activation::Sin => ACT_SIN,
            Activation::Cos => ACT_COS,
            Activation::Exp => ACT_EXP,
            Activation::Abs => ACT_ABS,
            Activation::Sigmoid => ACT_SIGMOID,
            Activation::Silu => ACT_SILU,
            Activation::Elu => ACT_ELU,
            Activation::LeakyRelu => ACT_LEAKY_RELU,
            Activation::Mish => ACT_MISH,
            Activation::GeluErf => ACT_GELU_ERF,
            Activation::Snake => ACT_SNAKE,
            Activation::Clamp => ACT_CLAMP,
            Activation::ScaleShift => ACT_SCALE_SHIFT,
            Activation::Relu => ACT_RELU,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    Zero,
    Reflect,
    Replicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// [`DevOp::RandF32`] hash coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandCoord {
    Row,
    Column,
    Item,
}

/// What one emitted stage produced: its output tensor and the counter a later op in the same
/// [`StageProgram`] passes as a dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emitted {
    pub output: u32,
    pub done: u32,
}

#[derive(Clone, Copy)]
pub struct GatherRowsF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub table: TensorRef<'a>,
    /// Rows a named table is declared with.
    pub table_rows: u32,
    pub index: Option<u32>,
    pub rows: u32,
    pub width: u32,
    pub vocab: u32,
    pub rows_per_item: u32,
    pub repeat: u32,
    pub index_item_stride: u32,
    pub table_item_stride: u32,
    pub table_f16: bool,
    pub accumulate: bool,
    pub out_stride: u32,
    pub out_col0: u32,
}

#[derive(Clone, Copy)]
pub struct CopyColsF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub items: u32,
    pub rows: u32,
    pub cols: u32,
    pub in_item_stride: u32,
    pub in_stride: u32,
    pub in_offset: u32,
    pub out_item_stride: u32,
    pub out_stride: u32,
    pub out_offset: u32,
}

/// [`DevOp::Conv1dF32`] (`transpose == false`) or [`DevOp::ConvTranspose1dF32`]. For the
/// transpose, `dilation_or_output_padding` is the output padding and the pads are crops.
#[derive(Clone, Copy)]
pub struct Conv1dF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub weight: TensorRef<'a>,
    pub bias: Option<TensorRef<'a>>,
    pub alpha: Option<TensorRef<'a>>,
    pub residual: Option<u32>,
    pub lengths: Option<u32>,
    pub batch: u32,
    pub in_rows: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel: u32,
    pub stride: u32,
    pub dilation_or_output_padding: u32,
    pub groups: u32,
    pub pad_before: u32,
    pub pad_after: u32,
    pub pad_mode: PadMode,
    pub input_activation: Activation,
    pub output_activation: Activation,
    pub slope: f32,
    pub weight_f16: bool,
    /// Flag bit 14: split-bf16 wide tensor-core tiles where they fill the machine (~2^-17 relative
    /// per product; the default 3xTF32 keeps ~2^-21) at half the tensor-core cost.
    pub split_bf16: bool,
    /// Flag bit 15: Conv1d weights `[cout][kernel][cin/groups]` instead of
    /// `[cout][cin/groups][kernel]`; ConvTranspose1d weights `[stride][cout][ceil(kernel /
    /// stride)][cin/groups]` (zero past the kernel) instead of `[cin][cout/groups][kernel]`. Either
    /// way a GEMM column run (tap, channels) is contiguous.
    pub weight_tap_major: bool,
    /// Flag bit 17: on sm_90a a split-bf16 conv (pointwise, or tap-major with 64-channel groups)
    /// runs the same product on warpgroup MMA; the interpreter falls back to the wide tile
    /// elsewhere.
    pub wgmma: bool,
}

impl Conv1dF32Stage<'_> {
    /// Output rows per item, or `None` for an empty output.
    pub fn out_rows(&self, transpose: bool) -> Option<u32> {
        let (before, after) = (u64::from(self.pad_before), u64::from(self.pad_after));
        let rows = if transpose {
            let full = u64::from(self.in_rows.checked_sub(1)?) * u64::from(self.stride)
                + u64::from(self.kernel)
                + u64::from(self.dilation_or_output_padding);
            full.checked_sub(before + after)?
        } else {
            let span = u64::from(self.dilation_or_output_padding)
                * u64::from(self.kernel.checked_sub(1)?)
                + 1;
            (u64::from(self.in_rows) + before + after).checked_sub(span)? / u64::from(self.stride) + 1
        };
        u32::try_from(rows).ok().filter(|&rows| rows > 0)
    }
}

#[derive(Clone, Copy)]
pub struct UnaryF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub param: Option<TensorRef<'a>>,
    pub rows: u32,
    pub width: u32,
    /// Row stride of `x` and `out` (0 = `width`).
    pub stride: u32,
    pub col0: u32,
    pub kind: Activation,
    pub p0: f32,
    pub p1: f32,
}

#[derive(Clone, Copy)]
pub struct BinaryF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub op: BinaryOp,
    pub items: u32,
    pub rows: u32,
    pub width: u32,
    pub b_item_stride: u32,
    pub b_row_stride: u32,
    pub b_col_stride: u32,
    pub scale: Option<f32>,
}

#[derive(Clone, Copy)]
pub struct CumSumF64Stage<'a> {
    pub output: TensorRef<'a>,
    pub column_scale: Option<TensorRef<'a>>,
    pub lengths: Option<u32>,
    pub items: u32,
    pub rows: u32,
    pub width: u32,
    pub x_width: u32,
    pub exclusive: bool,
    pub wrap: bool,
    pub f64_output: bool,
    pub scale: f32,
    pub post_scale: f32,
}

#[derive(Clone, Copy)]
pub struct RandF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub items: u32,
    pub rows: u32,
    pub width: u32,
    pub stream: u32,
    pub stream_shift: u32,
    pub a: RandCoord,
    pub b: RandCoord,
    pub a_offset: u32,
    pub b_offset: u32,
    pub normal: bool,
    pub shared_seed: bool,
    /// `out = (v + offset) * scale`.
    pub scale: f32,
    pub offset: f32,
}

#[derive(Clone, Copy)]
pub struct AttentionF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub key_lengths: Option<u32>,
    pub bias: Option<TensorRef<'a>>,
    pub batch: u32,
    pub q_rows: u32,
    pub kv_rows: u32,
    pub heads: u32,
    pub head_width: u32,
    /// Row stride of query/key/value (0 = `heads * head_width`).
    pub in_stride: u32,
    pub k_col0: u32,
    pub v_col0: u32,
    pub causal: bool,
    pub scale: f32,
    pub bias_head_stride: u32,
    /// Keys before each item's own: rows of a cached prefix block (see [`AttentionPrefix`]).
    pub prefix: Option<AttentionPrefix>,
}

/// A key prefix for [`AttentionF32Stage`]: item `b` first attends to the `rows` rows of block
/// `index[b]` of `table` (`[blocks][rows][K | V]`, each `2 * heads * head_width` wide).
#[derive(Clone, Copy)]
pub struct AttentionPrefix {
    pub table: u32,
    pub index: u32,
    pub blocks: u32,
    pub rows: u32,
}

/// Several stages emitted into ONE program (one launch), ordered by explicit dependencies:
/// each call returns the counter later calls list in `deps`. [`PacketPrefix::program`] opens it,
/// [`StageProgram::finish`] appends the program and makes the last output the prefix output.
pub struct StageProgram {
    prefix: PacketPrefix,
    builder: Builder,
    output: Option<u32>,
}

impl PacketPrefix {
    pub fn forward_pipeline_section(
        &self,
        name: &str,
        input_dtype: plow_asset::packet_pipeline::PipelineDType,
        output_dtype: plow_asset::packet_pipeline::PipelineDType,
        output_shape: Vec<u64>,
    ) -> Result<packet::devbuild::SectionData, String> {
        use plow_asset::packet_pipeline::{
            PacketPipeline, PacketPipelines, PipelineTensor, SECTION, VERSION,
        };
        use std::collections::BTreeMap;

        let input = self
            .model
            .tensors
            .get(self.input as usize)
            .ok_or("pipeline input tensor is missing")?;
        let output = self
            .model
            .tensors
            .get(self.output as usize)
            .ok_or("pipeline output tensor is missing")?;
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
                name: name.into(),
                driver: "forward.v1".into(),
                programs,
                tensors: BTreeMap::from([
                    (
                        "input".into(),
                        PipelineTensor {
                            name: input.name.clone(),
                            dtype: input_dtype,
                            shape: self.input_shape.clone(),
                        },
                    ),
                    (
                        "output".into(),
                        PipelineTensor {
                            name: output.name.clone(),
                            dtype: output_dtype,
                            shape: output_shape,
                        },
                    ),
                ]),
                parameters: BTreeMap::from([(
                    "forward_program_count".into(),
                    self.programs.len() as u64,
                )]),
            }],
        };
        metadata.validate(self.model.progs.len(), |tensor_name| {
            self.model
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
                .map(|tensor| tensor.bytes)
        })?;
        Ok(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
        })
    }

    pub fn append_dense_f32(self, rows: u32, stage: DenseF32Stage<'_>) -> Result<Self, String> {
        let source = self.output;
        self.append_dense_f32_from(source, rows, stage)
    }

    pub fn append_dense_f32_from(
        mut self,
        source: u32,
        rows: u32,
        stage: DenseF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || rows == 0 || stage.input_width == 0 || stage.output_width == 0 {
            return Err("invalid dense geometry".into());
        }
        if source as usize >= self.model.tensors.len() {
            return Err("dense source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.output_width, 4)?);
        let weight = builder.tensor(
            stage.weight,
            tensor_bytes(
                stage.output_width,
                stage.input_width,
                match stage.weight_type {
                    DenseWeight::F32 => 4,
                    DenseWeight::Bf16 => 2,
                },
            )?,
        );
        let bias = stage
            .bias
            .map(|name| builder.tensor(name, u64::from(stage.output_width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        let activation = u32::from(stage.activation == DenseActivation::Relu);
        let ln_round = stage.layer_norm.is_some_and(|ln| ln.round_bf16);
        let tensor_cores = (stage.input_bf16_exact || ln_round) && stage.weight_type == DenseWeight::Bf16;
        let flags = u32::from(stage.round_bf16)
            | (u32::from(stage.activation == DenseActivation::GeluErfBf16) << 1)
            | (u32::from(stage.weight_type == DenseWeight::Bf16) << 2)
            | (u32::from(tensor_cores) << 3)
            | (u32::from(stage.layer_norm.is_some()) << 5)
            | (u32::from(ln_round) << 6);
        let prologue = match stage.layer_norm {
            Some(ln) => {
                if ln.stats as usize >= builder.n_tensors() || builder.tensor_bytes(ln.stats) < tensor_bytes(rows, 2, 4)? {
                    return Err("dense layer-norm stats tensor is missing or short".into());
                }
                let width = u64::from(stage.input_width) * 4;
                let affine = |b: &mut Builder, t: Option<TensorRef<'_>>| match t {
                    Some(TensorRef::Named(name)) => Ok(b.tensor(name, width)),
                    Some(TensorRef::Handle(h)) if (h as usize) < b.n_tensors() && b.tensor_bytes(h) >= width => Ok(h),
                    Some(TensorRef::Handle(_)) => Err("dense layer-norm affine tensor is missing".to_string()),
                    None => Ok(packet::dev::TENSOR_NONE),
                };
                [ln.stats, affine(&mut builder, ln.gamma)?, affine(&mut builder, ln.beta)?]
            }
            None => [packet::dev::TENSOR_NONE; 3],
        };
        let blocks = dense_blocks(
            rows,
            stage.output_width,
            stage.input_width,
            tensor_cores,
            stage.split == DenseSplit::Parallel,
            self.model.n_cu,
        );
        if let DenseSplit::Reference(rows_ref) = stage.split {
            if !tensor_cores
                || stage.layer_norm.is_some()
                || stage.input_width % 8 != 0
                || rows_ref as usize >= builder.n_tensors()
                || builder.tensor_bytes(rows_ref) < u64::from(rows) * 4
            {
                return Err("reference split needs bf16 tensor cores, no prologue and a row table".into());
            }
        }
        // Declared under either split, so both lowerings share one tensor table.
        let scratch = if tensor_cores {
            let scratch = builder.tensor(
                DENSE_SPLITK_SCRATCH,
                (u64::from(DENSE_SPLITK_TICKETS) + u64::from(self.model.n_cu) * 64 * 64) * 4,
            );
            match stage.split {
                DenseSplit::Parallel => scratch,
                DenseSplit::Reference(rows_ref) => rows_ref,
            }
        } else {
            packet::dev::TENSOR_NONE
        };
        let reference = matches!(stage.split, DenseSplit::Reference(_));
        let flags = flags | (u32::from(reference) << 7);
        let n_cu = self.model.n_cu;
        builder.emit(
            DevOp::DenseGemmF32,
            repeated(self.model.n_cu, blocks),
            &[],
            |instruction| {
                instruction.t[..5].copy_from_slice(&[output, source, weight, bias, scratch]);
                instruction.t[5..8].copy_from_slice(&prologue);
                instruction.i[..4].copy_from_slice(&[
                    rows,
                    stage.output_width,
                    stage.input_width,
                    activation,
                ]);
                instruction.i[7] = flags;
                if reference {
                    instruction.j[0] = n_cu;
                }
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_initialized_add_f32(
        mut self,
        stage: InitializedAddF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || stage.values.is_empty() || !stage.scale.is_finite() {
            return Err("invalid scaled-add stage".into());
        }
        let elements: u32 = stage
            .values
            .len()
            .try_into()
            .map_err(|_| "scaled-add tensor is too large")?;
        let mut init = Vec::with_capacity(stage.values.len() * 4);
        for value in stage.values {
            init.extend_from_slice(&value.to_le_bytes());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, u64::from(elements) * 4);
        let addend = builder.tensor_init(stage.addend, init);
        builder.emit(
            DevOp::ScaledAddF32,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..3].copy_from_slice(&[output, self.output, addend]);
                instruction.i[..2].copy_from_slice(&[elements, u32::from(stage.round_bf16)]);
                instruction.f[0] = stage.scale;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(elements);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_scaled_add_f32(
        mut self,
        a: u32,
        b: u32,
        elements: u32,
        stage: ScaledAddF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || elements == 0 || !stage.scale.is_finite() {
            return Err("invalid scaled-add stage".into());
        }
        if [a, b]
            .into_iter()
            .any(|handle| handle as usize >= self.model.tensors.len())
        {
            return Err("scaled-add source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, u64::from(elements) * 4);
        builder.emit(
            DevOp::ScaledAddF32,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..3].copy_from_slice(&[output, a, b]);
                instruction.i[..2].copy_from_slice(&[elements, u32::from(stage.round_bf16)]);
                instruction.f[0] = stage.scale;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(elements);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_layer_norm_f32(
        mut self,
        rows: u32,
        stage: LayerNormF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || rows == 0
            || stage.width == 0
            || !stage.epsilon.is_finite()
            || stage.epsilon <= 0.0
        {
            return Err("invalid layer-normalization stage".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.width, 4)?);
        let gamma = stage
            .gamma
            .map(|name| builder.tensor(name, u64::from(stage.width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        let beta = stage
            .beta
            .map(|name| builder.tensor(name, u64::from(stage.width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        builder.emit(
            DevOp::LayerNormF32,
            repeated(self.model.n_cu, self.model.n_cu),
            &[],
            |instruction| {
                instruction.t[..4].copy_from_slice(&[output, self.output, gamma, beta]);
                let flags =
                    u32::from(stage.round_bf16) | (u32::from(stage.ordered_statistics) << 1);
                instruction.i[..3].copy_from_slice(&[rows, stage.width, flags]);
                instruction.f[0] = stage.epsilon;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_grouped_attention_f32(
        mut self,
        query: u32,
        key: u32,
        value: u32,
        rows: u32,
        stage: GroupedAttentionF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || rows == 0
            || stage.width == 0
            || stage.head_width == 0
            || stage.width % stage.head_width != 0
            || !(1..=256).contains(&stage.group_rows)
            || (stage.group_table && stage.valid_rows.is_none())
        {
            return Err("invalid grouped-attention geometry".into());
        }
        if [Some(query), Some(key), Some(value), stage.valid_rows]
            .into_iter()
            .flatten()
            .any(|handle| handle as usize >= self.model.tensors.len())
        {
            return Err("grouped-attention source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.width, 4)?);
        let flags = u32::from(stage.round_score_bf16)
            | (u32::from(stage.round_probability_bf16) << 1)
            | (u32::from(stage.round_output_bf16) << 2)
            | (u32::from(stage.group_table) << 3);
        let heads = stage.width / stage.head_width;
        // One slice per (group, head, 8-row query chunk), as the interpreter decomposes it.
        let blocks = rows
            .div_ceil(stage.group_rows.max(1))
            .checked_mul(heads)
            .and_then(|items| items.checked_mul(stage.group_rows.div_ceil(8)))
            .ok_or("grouped-attention dispatch geometry overflows")?
            .clamp(1, self.model.n_cu);
        builder.emit(
            DevOp::GroupedAttentionF32,
            repeated(self.model.n_cu, blocks),
            &[],
            |instruction| {
                instruction.t[..4].copy_from_slice(&[output, query, key, value]);
                instruction.t[4] = stage.valid_rows.unwrap_or(packet::dev::TENSOR_NONE);
                instruction.i[..5].copy_from_slice(&[
                    rows,
                    stage.width,
                    stage.head_width,
                    stage.group_rows,
                    flags,
                ]);
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_embed_overlay_bf16(
        mut self,
        stage: EmbedOverlayBf16Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || stage.rows == 0
            || stage.width == 0
            || stage.vocabulary == 0
            || stage.overlay_rows == 0
        {
            return Err("invalid embedding-overlay geometry".into());
        }
        let overlay_bytes = tensor_bytes(stage.overlay_rows, stage.width, 4)?;
        if self
            .model
            .tensors
            .get(self.output as usize)
            .is_none_or(|source| source.bytes != overlay_bytes)
        {
            return Err("embedding-overlay source has the wrong size".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(stage.rows, stage.width, 2)?);
        let table = builder.tensor(stage.table, tensor_bytes(stage.vocabulary, stage.width, 2)?);
        let tokens = builder.tensor(stage.tokens, u64::from(stage.rows) * 4);
        let overlay_index = builder.tensor(stage.overlay_index, u64::from(stage.rows) * 4);
        let elements = stage
            .rows
            .checked_mul(stage.width)
            .ok_or("embedding-overlay dispatch geometry overflows")?;
        builder.emit(
            DevOp::EmbedOverlayBf16,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..5].copy_from_slice(&[
                    output,
                    table,
                    tokens,
                    self.output,
                    overlay_index,
                ]);
                instruction.i[..4].copy_from_slice(&[
                    stage.rows,
                    stage.width,
                    stage.vocabulary,
                    stage.overlay_rows,
                ]);
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(stage.rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }
}

fn f32_bytes(elements: u64) -> Result<u64, String> {
    elements.checked_mul(4).ok_or_else(|| "tensor size overflows".into())
}

fn product(values: &[u32]) -> Result<u64, String> {
    values
        .iter()
        .try_fold(1u64, |acc, &v| acc.checked_mul(u64::from(v)))
        .ok_or_else(|| "geometry overflows".into())
}

/// Elements a strided `[items][rows][cols]` view spans: the last element's offset plus one.
fn strided_span(items: u32, item_stride: u32, rows: u32, stride: u32, offset: u32, cols: u32) -> u64 {
    u64::from(items.saturating_sub(1)) * u64::from(item_stride)
        + u64::from(rows.saturating_sub(1)) * u64::from(stride)
        + u64::from(offset)
        + u64::from(cols)
}

fn u32_elements(count: u64, what: &str) -> Result<u32, String> {
    u32::try_from(count).map_err(|_| format!("{what}: more than u32::MAX elements"))
}

impl StageProgram {
    fn resolve(&mut self, tensor: TensorRef<'_>, bytes: u64, what: &str) -> Result<u32, String> {
        match tensor {
            TensorRef::Named(name) => Ok(self.builder.tensor(name, bytes)),
            TensorRef::Handle(handle) => self.input(handle, bytes, what),
        }
    }

    fn input(&self, handle: u32, bytes: u64, what: &str) -> Result<u32, String> {
        if (handle as usize) < self.builder.n_tensors() && self.builder.tensor_bytes(handle) >= bytes {
            Ok(handle)
        } else {
            Err(format!("{what}: tensor {handle} is missing or smaller than {bytes} bytes"))
        }
    }

    fn slices(&self, units: u64) -> Result<Vec<u32>, String> {
        let n_cu = self.builder.n_cu();
        if n_cu == 0 {
            return Err("program has no compute units".into());
        }
        Ok(repeated(n_cu, units.clamp(1, u64::from(n_cu)) as u32))
    }

    fn emit(
        &mut self,
        op: DevOp,
        units: u64,
        deps: &[u32],
        output: u32,
        fill: impl FnOnce(&mut packet::dev::DevInst),
    ) -> Result<Emitted, String> {
        let cus = self.slices(units)?;
        let done = self.builder.emit(op, cus, deps, fill);
        self.output = Some(output);
        Ok(Emitted { output, done })
    }

    /// Handle of an already-declared tensor.
    pub fn handle_of(&self, name: &str) -> Option<u32> {
        (0..self.builder.n_tensors() as u32).find(|&h| self.builder.tensor_name(h) == name)
    }

    /// Append the program to the prefix; `tag` is its `prog_t` entry.
    pub fn finish(mut self, tag: u32) -> PacketPrefix {
        let program = self.prefix.model.progs.len();
        self.prefix.model.tensors = self.builder.tensors();
        self.prefix.model.progs.push(self.builder.finish());
        self.prefix.model.prog_t.push(tag);
        self.prefix.programs.push(program);
        if let Some(output) = self.output {
            self.prefix.output = output;
        }
        self.prefix
    }

    pub fn gather_rows_f32(
        &mut self,
        deps: &[u32],
        stage: GatherRowsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        let per_item = if stage.rows_per_item == 0 { stage.rows } else { stage.rows_per_item };
        if stage.rows == 0 || stage.width == 0 || per_item == 0 {
            return Err("invalid row-gather geometry".into());
        }
        let items = stage.rows.div_ceil(per_item);
        let repeat = stage.repeat.max(1);
        let element = if stage.table_f16 { 2 } else { 4 };
        let table_bytes = product(&[stage.table_rows.max(1), stage.width])? * element;
        let table = self.resolve(stage.table, table_bytes, "row-gather table")?;
        if let Some(index) = stage.index {
            let span = u64::from(items - 1) * u64::from(stage.index_item_stride)
                + u64::from((per_item - 1) / repeat)
                + 1;
            self.input(index, span * 4, "row-gather index")?;
        }
        let out_stride = if stage.out_stride == 0 { stage.width } else { stage.out_stride };
        let span = strided_span(1, 0, stage.rows, out_stride, stage.out_col0, stage.width);
        let output = self.resolve(stage.output, f32_bytes(span)?, "row-gather output")?;
        let elements = product(&[stage.rows, stage.width])?;
        u32_elements(elements, "row gather")?;
        let flags = u32::from(stage.table_f16) | (u32::from(stage.accumulate) << 1);
        self.emit(DevOp::GatherRowsF32, elements.div_ceil(2048), deps, output, |d| {
            d.t[..3].copy_from_slice(&[
                output,
                table,
                stage.index.unwrap_or(packet::dev::TENSOR_NONE),
            ]);
            d.i = [
                stage.rows,
                stage.width,
                stage.vocab,
                stage.rows_per_item,
                stage.repeat,
                stage.index_item_stride,
                stage.table_item_stride,
                flags,
            ];
            d.j = [stage.out_stride, stage.out_col0];
        })
    }

    pub fn copy_cols_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: CopyColsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.items == 0 || stage.rows == 0 || stage.cols == 0 {
            return Err("invalid column-copy geometry".into());
        }
        let span = strided_span(
            stage.items,
            stage.in_item_stride,
            stage.rows,
            stage.in_stride,
            stage.in_offset,
            stage.cols,
        );
        self.input(x, f32_bytes(span)?, "column-copy source")?;
        let span = strided_span(
            stage.items,
            stage.out_item_stride,
            stage.rows,
            stage.out_stride,
            stage.out_offset,
            stage.cols,
        );
        let output = self.resolve(stage.output, f32_bytes(span)?, "column-copy output")?;
        let elements = product(&[stage.items, stage.rows, stage.cols])?;
        u32_elements(elements, "column copy")?;
        self.emit(DevOp::CopyColsF32, elements.div_ceil(2048), deps, output, |d| {
            d.t[..2].copy_from_slice(&[output, x]);
            d.i[..7].copy_from_slice(&[
                stage.items,
                stage.rows,
                stage.cols,
                stage.in_stride,
                stage.in_offset,
                stage.out_stride,
                stage.out_offset,
            ]);
            d.j = [stage.in_item_stride, stage.out_item_stride];
        })
    }

    /// [`DevOp::Conv1dF32`], or [`DevOp::ConvTranspose1dF32`] with `transpose`.
    pub fn conv1d_f32(
        &mut self,
        x: u32,
        transpose: bool,
        deps: &[u32],
        stage: Conv1dF32Stage<'_>,
    ) -> Result<Emitted, String> {
        self.conv1d_f32_with(x, transpose, deps, stage, packet::dev::TENSOR_NONE)
    }

    fn conv1d_f32_with(
        &mut self,
        x: u32,
        transpose: bool,
        deps: &[u32],
        stage: Conv1dF32Stage<'_>,
        row_scale: u32,
    ) -> Result<Emitted, String> {
        let valid_activation = |a: Activation| !matches!(a, Activation::Clamp | Activation::ScaleShift);
        if stage.batch == 0
            || stage.kernel == 0
            || stage.stride == 0
            || (!transpose && stage.dilation_or_output_padding == 0)
            || stage.groups == 0
            || stage.in_channels % stage.groups != 0
            || stage.out_channels % stage.groups != 0
            || stage.pad_before > 0xFFFF
            || stage.pad_after > 0xFFFF
            || (transpose && stage.pad_mode != PadMode::Zero)
            || !valid_activation(stage.input_activation)
            || !valid_activation(stage.output_activation)
            // One alpha tensor: a snake output activation reads it per output channel, so the
            // input takes no activation.
            || (stage.output_activation == Activation::Snake && stage.input_activation != Activation::None)
        {
            return Err("invalid convolution geometry".into());
        }
        let out_rows = stage.out_rows(transpose).ok_or("convolution output is empty")?;
        let out_elements = product(&[stage.batch, out_rows, stage.out_channels])?;
        u32_elements(out_elements, "convolution output")?;
        self.input(
            x,
            f32_bytes(product(&[stage.batch, stage.in_rows, stage.in_channels])?)?,
            "convolution input",
        )?;
        let taps = if transpose && stage.weight_tap_major {
            stage.kernel.div_ceil(stage.stride) * stage.stride
        } else {
            stage.kernel
        };
        let weight_bytes = product(&[stage.out_channels, stage.in_channels / stage.groups, taps])?
            * if stage.weight_f16 { 2 } else { 4 };
        let weight = self.resolve(stage.weight, weight_bytes, "convolution weight")?;
        let bias = match stage.bias {
            Some(bias) => self.resolve(bias, u64::from(stage.out_channels) * 4, "convolution bias")?,
            None => packet::dev::TENSOR_NONE,
        };
        let alpha = match stage.alpha {
            Some(alpha) => self.resolve(
                alpha,
                u64::from(stage.in_channels.max(stage.out_channels)) * 4,
                "convolution alpha",
            )?,
            None => packet::dev::TENSOR_NONE,
        };
        if let Some(residual) = stage.residual {
            self.input(residual, f32_bytes(out_elements)?, "convolution residual")?;
        }
        if let Some(lengths) = stage.lengths {
            self.input(lengths, u64::from(stage.batch) * 4, "convolution lengths")?;
        }
        let output = self.resolve(stage.output, f32_bytes(out_elements)?, "convolution output")?;
        let pad_mode = match stage.pad_mode {
            PadMode::Zero => 0,
            PadMode::Reflect => 1,
            PadMode::Replicate => 2,
        };
        // Bit 13: the GEMM-shaped cases run 3xTF32 tensor cores (FP32-accurate) on 64x64 tiles.
        let flags = pad_mode
            | (stage.input_activation.code() << 4)
            | (stage.output_activation.code() << 8)
            | (u32::from(stage.weight_f16) << 12)
            | (1 << 13)
            | (u32::from(stage.split_bf16) << 14)
            | (u32::from(stage.weight_tap_major) << 15)
            | (u32::from(stage.wgmma && stage.split_bf16) << 17);
        let op = if transpose { DevOp::ConvTranspose1dF32 } else { DevOp::Conv1dF32 };
        let units = conv1d_units(&stage, transpose, out_rows, self.builder.n_cu());
        self.emit(op, units, deps, output, |d| {
            d.t[..7].copy_from_slice(&[
                output,
                x,
                weight,
                bias,
                alpha,
                stage.residual.unwrap_or(packet::dev::TENSOR_NONE),
                stage.lengths.unwrap_or(packet::dev::TENSOR_NONE),
            ]);
            d.t[7] = row_scale;
            d.i = [
                stage.batch,
                stage.in_rows,
                stage.in_channels,
                stage.out_channels,
                stage.kernel,
                stage.stride,
                stage.dilation_or_output_padding,
                stage.groups,
            ];
            d.f[0] = stage.slope;
            d.j = [stage.pad_before | (stage.pad_after << 16), flags];
        })
    }

    /// In place when `output` is `TensorRef::Handle(x)`.
    pub fn unary_f32(&mut self, x: u32, deps: &[u32], stage: UnaryF32Stage<'_>) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.width == 0 {
            return Err("invalid unary geometry".into());
        }
        let stride = if stage.stride == 0 { stage.width } else { stage.stride };
        let bytes = f32_bytes(strided_span(1, 0, stage.rows, stride, stage.col0, stage.width))?;
        self.input(x, bytes, "unary input")?;
        let output = self.resolve(stage.output, bytes, "unary output")?;
        let param = match stage.param {
            Some(param) => self.resolve(param, u64::from(stage.width) * 4, "unary parameter")?,
            None => packet::dev::TENSOR_NONE,
        };
        let elements = product(&[stage.rows, stage.width])?;
        u32_elements(elements, "unary")?;
        self.emit(DevOp::UnaryF32, elements.div_ceil(2048), deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, x, param]);
            d.i[..5].copy_from_slice(&[stage.rows, stage.width, stage.kind.code(), stage.stride, stage.col0]);
            d.f = [stage.p0, stage.p1];
        })
    }

    pub fn binary_f32(
        &mut self,
        a: u32,
        b: u32,
        deps: &[u32],
        stage: BinaryF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.items == 0 || stage.rows == 0 || stage.width == 0 {
            return Err("invalid binary geometry".into());
        }
        let elements = product(&[stage.items, stage.rows, stage.width])?;
        u32_elements(elements, "binary")?;
        self.input(a, f32_bytes(elements)?, "binary a")?;
        let b_span = u64::from(stage.items - 1) * u64::from(stage.b_item_stride)
            + u64::from(stage.rows - 1) * u64::from(stage.b_row_stride)
            + u64::from(stage.width - 1) * u64::from(stage.b_col_stride)
            + 1;
        self.input(b, f32_bytes(b_span)?, "binary b")?;
        let output = self.resolve(stage.output, f32_bytes(elements)?, "binary output")?;
        let op = match stage.op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
            BinaryOp::Max => 4,
            BinaryOp::Min => 5,
        };
        self.emit(DevOp::BinaryF32, elements.div_ceil(2048), deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, a, b]);
            d.i = [
                stage.items,
                stage.rows,
                stage.width,
                op,
                stage.b_item_stride,
                stage.b_row_stride,
                stage.b_col_stride,
                u32::from(stage.scale.is_some()),
            ];
            d.f[0] = stage.scale.unwrap_or(1.0);
        })
    }

    pub fn cumsum_f64(&mut self, x: u32, deps: &[u32], stage: CumSumF64Stage<'_>) -> Result<Emitted, String> {
        let x_width = if stage.x_width == 0 { stage.width } else { stage.x_width };
        if stage.items == 0 || stage.rows == 0 || stage.width == 0 {
            return Err("invalid prefix-sum geometry".into());
        }
        self.input(x, f32_bytes(product(&[stage.items, stage.rows, x_width])?)?, "prefix-sum input")?;
        let elements = product(&[stage.items, stage.rows, stage.width])?;
        let output = self.resolve(
            stage.output,
            elements * if stage.f64_output { 8 } else { 4 },
            "prefix-sum output",
        )?;
        let column_scale = match stage.column_scale {
            Some(scale) => self.resolve(scale, u64::from(stage.width) * 4, "prefix-sum column scale")?,
            None => packet::dev::TENSOR_NONE,
        };
        if let Some(lengths) = stage.lengths {
            self.input(lengths, u64::from(stage.items) * 4, "prefix-sum lengths")?;
        }
        let flags = u32::from(stage.exclusive) | (u32::from(stage.wrap) << 1) | (u32::from(stage.f64_output) << 2);
        let units = product(&[stage.items, stage.width])?;
        self.emit(DevOp::CumSumF64, units, deps, output, |d| {
            d.t[..4].copy_from_slice(&[
                output,
                x,
                column_scale,
                stage.lengths.unwrap_or(packet::dev::TENSOR_NONE),
            ]);
            d.i[..5].copy_from_slice(&[stage.items, stage.rows, stage.width, stage.x_width, flags]);
            d.f = [stage.scale, stage.post_scale];
        })
    }

    pub fn rand_f32(&mut self, seed: u32, deps: &[u32], stage: RandF32Stage<'_>) -> Result<Emitted, String> {
        if stage.items == 0 || stage.rows == 0 || stage.width == 0 || stage.stream_shift > 63 {
            return Err("invalid random geometry".into());
        }
        let seeds = if stage.shared_seed { 1 } else { stage.items };
        self.input(seed, u64::from(seeds) * 8, "random seed")?;
        let elements = product(&[stage.items, stage.rows, stage.width])?;
        u32_elements(elements, "random")?;
        let output = self.resolve(stage.output, f32_bytes(elements)?, "random output")?;
        let coord = |c: RandCoord| match c {
            RandCoord::Row => 0,
            RandCoord::Column => 1,
            RandCoord::Item => 2,
        };
        let flags = u32::from(stage.normal) | (u32::from(stage.shared_seed) << 1);
        self.emit(DevOp::RandF32, elements.div_ceil(2048), deps, output, |d| {
            d.t[..2].copy_from_slice(&[output, seed]);
            d.i = [
                stage.items,
                stage.rows,
                stage.width,
                stage.stream,
                stage.stream_shift,
                coord(stage.a) | (coord(stage.b) << 2),
                stage.a_offset,
                stage.b_offset,
            ];
            d.f = [stage.scale, stage.offset];
            d.j[1] = flags;
        })
    }

    pub fn attention_f32(
        &mut self,
        query: u32,
        key: u32,
        value: u32,
        deps: &[u32],
        stage: AttentionF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.batch == 0
            || stage.q_rows == 0
            || stage.kv_rows == 0
            || stage.heads == 0
            || !matches!(stage.head_width, 64 | 128)
        {
            return Err("invalid attention geometry".into());
        }
        let width = stage.heads.checked_mul(stage.head_width).ok_or("attention width overflows")?;
        let stride = if stage.in_stride == 0 { width } else { stage.in_stride };
        let q_total = u32_elements(product(&[stage.batch, stage.q_rows])?, "attention query rows")?;
        let kv_total = u32_elements(product(&[stage.batch, stage.kv_rows])?, "attention key rows")?;
        self.input(query, f32_bytes(strided_span(1, 0, q_total, stride, 0, width))?, "attention query")?;
        for (tensor, col0, what) in [(key, stage.k_col0, "attention key"), (value, stage.v_col0, "attention value")] {
            self.input(tensor, f32_bytes(strided_span(1, 0, kv_total, stride, col0, width))?, what)?;
        }
        if let Some(lengths) = stage.key_lengths {
            self.input(lengths, u64::from(stage.batch) * 4, "attention key lengths")?;
        }
        if let Some(p) = stage.prefix {
            if stage.bias.is_some() || stage.causal || p.rows == 0 {
                return Err("attention prefix: no bias or causal mask, and at least one row".into());
            }
            self.input(p.table, f32_bytes(product(&[p.blocks, p.rows, 2 * width])?)?, "attention prefix")?;
            self.input(p.index, u64::from(stage.batch) * 4, "attention prefix index")?;
        }
        let bias = match stage.bias {
            Some(bias) => {
                let span = u64::from(stage.heads - 1) * u64::from(stage.bias_head_stride)
                    + product(&[stage.q_rows, stage.kv_rows])?;
                self.resolve(bias, f32_bytes(span)?, "attention bias")?
            }
            None => packet::dev::TENSOR_NONE,
        };
        let out_elements = product(&[stage.batch, stage.q_rows, width])?;
        let output = self.resolve(stage.output, f32_bytes(out_elements)?, "attention output")?;
        // head_width 64 runs 3xTF32 tensor cores (flag bit 1) on 128-query tiles.
        let tc = stage.head_width == 64;
        let units = product(&[stage.batch, stage.heads, stage.q_rows.div_ceil(if tc { 128 } else { 64 })])?;
        self.emit(DevOp::AttentionF32, units, deps, output, |d| {
            d.t.copy_from_slice(&[
                output,
                query,
                key,
                value,
                stage.key_lengths.unwrap_or(packet::dev::TENSOR_NONE),
                bias,
                stage.prefix.map_or(packet::dev::TENSOR_NONE, |p| p.table),
                stage.prefix.map_or(packet::dev::TENSOR_NONE, |p| p.index),
            ]);
            d.i = [
                stage.batch,
                stage.q_rows,
                stage.kv_rows,
                stage.heads,
                stage.head_width,
                stage.in_stride,
                u32::from(stage.causal) | (u32::from(tc) << 1),
                stage.prefix.map_or(stage.bias_head_stride, |p| p.rows),
            ];
            d.f[0] = stage.scale;
            d.j = [stage.k_col0, stage.v_col0];
        })
    }
}

/// [`DevOp::LayerNormF32`] inside a [`StageProgram`] (FP64 statistics).
#[derive(Clone, Copy)]
pub struct LayerNormRowsF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub gamma: Option<TensorRef<'a>>,
    pub beta: Option<TensorRef<'a>>,
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
}

impl StageProgram {
    /// In place when `output` is `TensorRef::Handle(x)`.
    pub fn layer_norm_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: LayerNormRowsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.width == 0 || !stage.epsilon.is_finite() || stage.epsilon <= 0.0 {
            return Err("invalid layer-normalization stage".into());
        }
        let bytes = f32_bytes(product(&[stage.rows, stage.width])?)?;
        self.input(x, bytes, "layer-norm input")?;
        let output = self.resolve(stage.output, bytes, "layer-norm output")?;
        let mut affine = |t: Option<TensorRef<'_>>, what| match t {
            Some(t) => self.resolve(t, u64::from(stage.width) * 4, what),
            None => Ok(packet::dev::TENSOR_NONE),
        };
        let gamma = affine(stage.gamma, "layer-norm gamma")?;
        let beta = affine(stage.beta, "layer-norm beta")?;
        // One warp per row, eight warps per slice.
        self.emit(DevOp::LayerNormF32, u64::from(stage.rows).div_ceil(8), deps, output, |d| {
            d.t[..4].copy_from_slice(&[output, x, gamma, beta]);
            d.i[..3].copy_from_slice(&[stage.rows, stage.width, 0]);
            d.f[0] = stage.epsilon;
        })
    }
}

impl StageProgram {
    /// RowStatsF32: `[rows][2]` = LayerNormF32's (mean, inverse std) per row.
    pub fn row_stats_f32(&mut self, x: u32, deps: &[u32], stage: RowStatsF32Stage<'_>) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.width == 0 || !stage.epsilon.is_finite() || stage.epsilon <= 0.0 {
            return Err("invalid row-statistics stage".into());
        }
        self.input(x, f32_bytes(product(&[stage.rows, stage.width])?)?, "row-statistics input")?;
        let output = self.resolve(stage.output, f32_bytes(product(&[stage.rows, 2])?)?, "row statistics")?;
        // One warp per row, eight warps per slice (LayerNormF32's decomposition).
        self.emit(DevOp::RowStatsF32, u64::from(stage.rows).div_ceil(8), deps, output, |d| {
            d.t[..2].copy_from_slice(&[output, x]);
            d.i[..3].copy_from_slice(&[stage.rows, stage.width, u32::from(stage.ordered_statistics) << 1]);
            d.f[0] = stage.epsilon;
        })
    }

    /// DenseGemmF32, optionally with the LayerNorm prologue.
    pub fn dense_f32(&mut self, x: u32, deps: &[u32], stage: DenseF32ProgramStage<'_>) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.input_width == 0 || stage.output_width == 0 {
            return Err("invalid dense geometry".into());
        }
        self.input(x, f32_bytes(product(&[stage.rows, stage.input_width])?)?, "dense input")?;
        let esize = match stage.weight_type {
            DenseWeight::F32 => 4,
            DenseWeight::Bf16 => 2,
        };
        let weight =
            self.resolve(stage.weight, product(&[stage.output_width, stage.input_width])? * esize, "dense weight")?;
        let bias = match stage.bias {
            Some(b) => self.resolve(b, u64::from(stage.output_width) * 4, "dense bias")?,
            None => packet::dev::TENSOR_NONE,
        };
        let prologue = match stage.layer_norm {
            Some(ln) => {
                self.input(ln.stats, f32_bytes(product(&[stage.rows, 2])?)?, "dense layer-norm stats")?;
                let width = u64::from(stage.input_width) * 4;
                let mut affine = |t: Option<TensorRef<'_>>, what| match t {
                    Some(t) => self.resolve(t, width, what),
                    None => Ok(packet::dev::TENSOR_NONE),
                };
                [ln.stats, affine(ln.gamma, "dense layer-norm gamma")?, affine(ln.beta, "dense layer-norm beta")?]
            }
            None => [packet::dev::TENSOR_NONE; 3],
        };
        let output =
            self.resolve(stage.output, f32_bytes(product(&[stage.rows, stage.output_width])?)?, "dense output")?;
        let ln_round = stage.layer_norm.is_some_and(|ln| ln.round_bf16);
        let tensor_cores = stage.operands_bf16_exact || (ln_round && stage.weight_type == DenseWeight::Bf16);
        let flags = u32::from(stage.round_bf16)
            | (u32::from(stage.activation == DenseActivation::GeluErfBf16) << 1)
            | (u32::from(stage.weight_type == DenseWeight::Bf16) << 2)
            | (u32::from(tensor_cores) << 3)
            | (u32::from(!tensor_cores && stage.tf32x3) << 4)
            | (u32::from(stage.layer_norm.is_some()) << 5)
            | (u32::from(ln_round) << 6);
        let units = if tensor_cores || stage.tf32x3 {
            u64::from(stage.rows.div_ceil(64)) * u64::from(stage.output_width.div_ceil(64))
        } else {
            u64::from(stage.rows.div_ceil(128)) * u64::from(stage.output_width.div_ceil(128))
        };
        self.emit(DevOp::DenseGemmF32, units, deps, output, |d| {
            d.t[..4].copy_from_slice(&[output, x, weight, bias]);
            d.t[5..8].copy_from_slice(&prologue);
            d.i[..4].copy_from_slice(&[
                stage.rows,
                stage.output_width,
                stage.input_width,
                u32::from(stage.activation == DenseActivation::Relu),
            ]);
            d.i[7] = flags;
        })
    }

    /// Conv1dF32 with the row-scaled residual epilogue: `out = residual + row_scale[b][t] *
    /// post(conv(x) + bias)`, `row_scale` f32 `[batch][out_rows]` (slot t7).
    pub fn conv1d_f32_row_scaled(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: Conv1dF32Stage<'_>,
        row_scale: u32,
    ) -> Result<Emitted, String> {
        let out_rows = stage.out_rows(false).ok_or("convolution output is empty")?;
        self.input(row_scale, f32_bytes(product(&[stage.batch, out_rows])?)?, "convolution row scale")?;
        self.conv1d_f32_with(x, false, deps, stage, row_scale)
    }
}

impl PacketPrefix {
    /// Append a single-op RowStatsF32 program; its output becomes the prefix output.
    pub fn append_row_stats_f32(self, x: u32, stage: RowStatsF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.row_stats_f32(x, &[], stage))
    }

    /// Open a multi-stage program; see [`StageProgram`].
    pub fn program(mut self) -> StageProgram {
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        StageProgram {
            prefix: self,
            builder,
            output: None,
        }
    }

    fn append_one(
        self,
        tag: u32,
        stage: impl FnOnce(&mut StageProgram) -> Result<Emitted, String>,
    ) -> Result<Self, String> {
        let mut program = self.program();
        stage(&mut program)?;
        Ok(program.finish(tag))
    }

    pub fn append_gather_rows_f32(self, stage: GatherRowsF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.gather_rows_f32(&[], stage))
    }

    pub fn append_copy_cols_f32(self, x: u32, stage: CopyColsF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.copy_cols_f32(x, &[], stage))
    }

    pub fn append_conv1d_f32(self, x: u32, stage: Conv1dF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.in_rows, |p| p.conv1d_f32(x, false, &[], stage))
    }

    pub fn append_conv_transpose1d_f32(self, x: u32, stage: Conv1dF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.in_rows, |p| p.conv1d_f32(x, true, &[], stage))
    }

    pub fn append_unary_f32(self, x: u32, stage: UnaryF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.unary_f32(x, &[], stage))
    }

    pub fn append_binary_f32(self, a: u32, b: u32, stage: BinaryF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.binary_f32(a, b, &[], stage))
    }

    pub fn append_cumsum_f64(self, x: u32, stage: CumSumF64Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.cumsum_f64(x, &[], stage))
    }

    pub fn append_rand_f32(self, seed: u32, stage: RandF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.rand_f32(seed, &[], stage))
    }

    pub fn append_attention_f32(
        self,
        query: u32,
        key: u32,
        value: u32,
        stage: AttentionF32Stage<'_>,
    ) -> Result<Self, String> {
        self.append_one(stage.q_rows, |p| p.attention_f32(query, key, value, &[], stage))
    }
}

fn tensor_bytes(rows: u32, columns: u32, element_bytes: u32) -> Result<u64, String> {
    u64::from(rows)
        .checked_mul(u64::from(columns))
        .and_then(|count| count.checked_mul(u64::from(element_bytes)))
        .ok_or_else(|| "tensor size overflows".into())
}

/// Conv1dF32 / ConvTranspose1dF32 slices, mirroring the interpreter's paths (op_speech_f32.cuh
/// d_conv1d_f32): depthwise 64-row x 64-channel tiles; the direct form per 256 outputs; the GEMM
/// form 64x64 tiles per (group, phase). `slices` caps the result at the machine.
fn conv1d_units(stage: &Conv1dF32Stage<'_>, transpose: bool, out_rows: u32, n_cu: u32) -> u64 {
    let (batch, cout) = (u64::from(stage.batch), u64::from(stage.out_channels));
    let cg = stage.in_channels / stage.groups;
    let ng = stage.out_channels / stage.groups;
    if !transpose && cg == 1 && ng == 1 {
        return batch * u64::from(out_rows).div_ceil(64) * cout.div_ceil(64);
    }
    if ng < 16 || cg * stage.kernel < 16 {
        return (batch * u64::from(out_rows) * cout).div_ceil(256);
    }
    let phases = if transpose { stage.stride } else { 1 };
    let per_phase = u64::from(out_rows.div_ceil(phases));
    let tiles = |tile: u64| {
        u64::from(stage.groups) * u64::from(phases) * (batch * per_phase).div_ceil(tile) * u64::from(ng).div_ceil(tile)
    };
    // 3xTF32 on 64x64 tiles, except FP32 FFMA on 128x128 tiles when those fill the machine for
    // a transposed conv or one with an input activation.
    let ffma = ((transpose && !stage.weight_tap_major) || (!transpose && stage.input_activation != Activation::None))
        && tiles(128) >= u64::from(n_cu);
    if ffma { tiles(128) } else { tiles(64) }
}

/// Split-K scratch shared by every tensor-core DenseGemmF32 (each in its own program, so they
/// never overlap): `DENSE_SPLITK_TICKETS` zeroed u32 tickets, then one 64x64 f32 partial per block.
const DENSE_SPLITK_SCRATCH: &str = "scratch.dense_f32_splitk";
const DENSE_SPLITK_TICKETS: u32 = 1024;

/// DenseGemmF32 slices, mirroring the interpreter's decomposition (op_speech_f32.cuh): FP32 FFMA
/// uses 128x128 tiles; the tensor-core path 64x64 tiles, split over K (BK = 64) until the tiles
/// fill the machine.
fn dense_blocks(rows: u32, n: u32, k: u32, tensor_cores: bool, split_k: bool, n_cu: u32) -> u32 {
    if !tensor_cores {
        return (rows.div_ceil(128) * n.div_ceil(128)).clamp(1, n_cu);
    }
    let tiles = rows.div_ceil(64) * n.div_ceil(64);
    if !split_k || tiles > DENSE_SPLITK_TICKETS || tiles * 2 > n_cu {
        return tiles.min(n_cu);
    }
    let steps = k.div_ceil(64).max(1);
    let chunk = steps.div_ceil((n_cu / tiles).min(steps));
    tiles * steps.div_ceil(chunk)
}

fn repeated(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks.max(1)).map(|index| index % n_cu).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_backend_neutral_causal_pipeline() {
        let mut tensors = Builder::new(2);
        let tokens = tensors.tensor("in.ids", 512 * 4);
        let positions = tensors.tensor("in.pos", 512 * 4);
        let kv_lengths = tensors.tensor("in.kvlen", 4 * 4);
        let overlay = tensors.tensor("in.encoder", 8 * 16 * 4);
        let overlay_index = tensors.tensor("in.encoder_index", 512 * 4);
        let model = Model {
            n_cu: 2,
            target: 0,
            tensors: tensors.tensors(),
            progs: (0..4).map(|_| Builder::new(2).finish()).collect(),
            kv_row_insts: Vec::new(),
            prog_t: vec![128, 512, 1, 4],
            gen: Vec::new(),
        };
        let section = causal_pipeline_section(
            &model,
            CausalPipelineSpec {
                name: "decode",
                max_context: 512,
                hidden: 16,
                decode_capacity: 4,
                overlay_rows: 8,
                ordered_dispatch: false,
                tensors: CausalPipelineTensors {
                    tokens,
                    positions,
                    kv_lengths,
                    overlay: Some(overlay),
                    overlay_index: Some(overlay_index),
                },
            },
        )
        .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let pipeline = &metadata.pipelines[0];
        assert_eq!(pipeline.driver, "causal.v1");
        assert_eq!(pipeline.programs["prefill.128"], 0);
        assert_eq!(pipeline.programs["prefill.512"], 1);
        assert_eq!(pipeline.programs["decode.1"], 2);
        assert_eq!(pipeline.programs["decode.4"], 3);
        assert_eq!(pipeline.tensors["overlay"].shape, [8, 16]);
        assert_eq!(pipeline.parameters["decode_capacity"], 4);
        assert_eq!(pipeline.parameters["ordered_dispatch"], 0);
    }

    #[test]
    fn appends_dense_program_to_prefix() {
        let mut builder = Builder::new(4);
        let input = builder.tensor("input", 3 * 8 * 4);
        let model = Model {
            n_cu: 4,
            target: 0,
            tensors: builder.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        };
        let prefix = PacketPrefix {
            model,
            programs: Vec::new(),
            input,
            output: input,
            input_shape: vec![3, 8],
        }
        .append_dense_f32(
            3,
            DenseF32Stage {
                output: "output",
                weight: "weight",
                bias: None,
                input_width: 8,
                output_width: 5,
                activation: DenseActivation::GeluErfBf16,
                round_bf16: false,
                weight_type: DenseWeight::F32,
                input_bf16_exact: false,
                layer_norm: None,
                split: DenseSplit::Parallel,
            },
        )
        .unwrap();
        assert_eq!(prefix.programs, [0]);
        let instruction = &prefix.model.progs[0].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::DenseGemmF32));
        assert_eq!(&instruction.i[..4], &[3, 5, 8, 0]);
        assert_eq!(instruction.i[7], 2);
        assert_eq!(instruction.t[3], packet::dev::TENSOR_NONE);
        assert!(!prefix.model.to_blob().is_empty());
        let values = [0.0f32; 15];
        let prefix = prefix
            .append_initialized_add_f32(InitializedAddF32Stage {
                output: "positioned",
                addend: "position",
                values: &values,
                scale: 1.0,
                round_bf16: true,
            })
            .unwrap();
        assert_eq!(prefix.programs, [0, 1]);
        let instruction = &prefix.model.progs[1].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::ScaledAddF32));
        assert_eq!(&instruction.i[..2], &[15, 1]);
        let prefix = prefix
            .append_layer_norm_f32(
                3,
                LayerNormF32Stage {
                    output: "normalized",
                    gamma: Some("gamma"),
                    beta: Some("beta"),
                    width: 5,
                    epsilon: 1e-5,
                    round_bf16: true,
                    ordered_statistics: true,
                },
            )
            .unwrap();
        assert_eq!(prefix.programs, [0, 1, 2]);
        let instruction = &prefix.model.progs[2].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::LayerNormF32));
        assert_eq!(&instruction.i[..3], &[3, 5, 3]);

        let normalized = prefix.output;
        let prefix = prefix
            .append_dense_f32_from(
                normalized,
                3,
                DenseF32Stage {
                    output: "query",
                    weight: "query.weight",
                    bias: Some("query.bias"),
                    input_width: 5,
                    output_width: 5,
                    activation: DenseActivation::None,
                    round_bf16: true,
                    weight_type: DenseWeight::F32,
                    input_bf16_exact: false,
                    layer_norm: None,
                    split: DenseSplit::Parallel,
                },
            )
            .unwrap();
        let query = prefix.output;
        let prefix = prefix
            .append_dense_f32_from(
                normalized,
                3,
                DenseF32Stage {
                    output: "key",
                    weight: "key.weight",
                    bias: Some("key.bias"),
                    input_width: 5,
                    output_width: 5,
                    activation: DenseActivation::None,
                    round_bf16: true,
                    weight_type: DenseWeight::F32,
                    input_bf16_exact: false,
                    layer_norm: None,
                    split: DenseSplit::Parallel,
                },
            )
            .unwrap();
        let key = prefix.output;
        assert_eq!(prefix.programs, [0, 1, 2, 3, 4]);
        assert_eq!(prefix.model.progs[3].insts[0].t[1], normalized);
        assert_eq!(prefix.model.progs[4].insts[0].t[1], normalized);
        let prefix = prefix
            .append_grouped_attention_f32(
                query,
                key,
                query,
                3,
                GroupedAttentionF32Stage {
                    output: "attention",
                    valid_rows: None,
                    group_table: false,
                    width: 5,
                    head_width: 5,
                    group_rows: 3,
                    round_score_bf16: true,
                    round_probability_bf16: true,
                    round_output_bf16: true,
                },
            )
            .unwrap();
        let instruction = &prefix.model.progs[5].insts[0];
        assert_eq!(
            DevOp::from_u16(instruction.op),
            Some(DevOp::GroupedAttentionF32)
        );
        assert_eq!(&instruction.t[1..4], &[query, key, query]);
        assert_eq!(instruction.t[4], packet::dev::TENSOR_NONE);
        assert_eq!(&instruction.i[..5], &[3, 5, 5, 3, 7]);
        let prefix = prefix
            .append_scaled_add_f32(
                query,
                key,
                15,
                ScaledAddF32Stage {
                    output: "residual",
                    scale: 0.5,
                    round_bf16: true,
                },
            )
            .unwrap();
        let instruction = &prefix.model.progs[6].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::ScaledAddF32));
        assert_eq!(&instruction.t[1..3], &[query, key]);
        assert_eq!(&instruction.i[..2], &[15, 1]);
        assert_eq!(instruction.f[0], 0.5);
        let prefix = prefix
            .append_embed_overlay_bf16(EmbedOverlayBf16Stage {
                output: "act.embeddings",
                table: "embedding.weight",
                tokens: "in.tokens",
                overlay_index: "in.overlay_index",
                rows: 3,
                width: 5,
                vocabulary: 11,
                overlay_rows: 3,
            })
            .unwrap();
        let instruction = &prefix.model.progs[7].insts[0];
        assert_eq!(
            DevOp::from_u16(instruction.op),
            Some(DevOp::EmbedOverlayBf16)
        );
        assert_eq!(&instruction.i[..4], &[3, 5, 11, 3]);
        assert_eq!(prefix.model.tensors[prefix.output as usize].bytes, 30);
        let section = prefix
            .forward_pipeline_section(
                "encode",
                plow_asset::packet_pipeline::PipelineDType::F32,
                plow_asset::packet_pipeline::PipelineDType::Bf16,
                vec![3, 5],
            )
            .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        assert_eq!(metadata.pipelines[0].programs.len(), 8);
        assert_eq!(metadata.pipelines[0].programs["forward.7"], 7);
        assert_eq!(metadata.pipelines[0].tensors["input"].shape, [3, 8]);
        assert_eq!(metadata.pipelines[0].tensors["output"].shape, [3, 5]);
    }

    fn prefix_with(tensors: &[(&str, u64)]) -> (PacketPrefix, Vec<u32>) {
        let mut builder = Builder::new(4);
        let handles: Vec<u32> = tensors.iter().map(|&(name, bytes)| builder.tensor(name, bytes)).collect();
        let model = Model {
            n_cu: 4,
            target: 0,
            tensors: builder.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        };
        let prefix = PacketPrefix {
            model,
            programs: Vec::new(),
            input: handles[0],
            output: handles[0],
            input_shape: vec![],
        };
        (prefix, handles)
    }

    #[test]
    fn stage_program_emits_dependent_signal_ops_into_one_program() {
        // A snake resblock: y = x + conv1x1(conv_k7_dil3(snake(x))), then an upsample and noise.
        let (prefix, h) = prefix_with(&[("x", 2 * 100 * 64 * 4), ("lengths", 8), ("seed", 16)]);
        let (x, lengths, seed) = (h[0], h[1], h[2]);
        let conv = |output, weight, kernel, dilation, pad, act, residual| Conv1dF32Stage {
            output: TensorRef::Named(output),
            weight: TensorRef::Named(weight),
            bias: Some(TensorRef::Named("b")),
            alpha: Some(TensorRef::Named("alpha")),
            residual,
            lengths: Some(lengths),
            batch: 2,
            in_rows: 100,
            in_channels: 64,
            out_channels: 64,
            kernel,
            stride: 1,
            dilation_or_output_padding: dilation,
            groups: 1,
            pad_before: pad,
            pad_after: pad,
            pad_mode: PadMode::Zero,
            input_activation: act,
            output_activation: Activation::None,
            slope: 0.0,
            weight_f16: false,
            split_bf16: false,
            weight_tap_major: false,
            wgmma: false,
        };
        let mut p = prefix.program();
        let a = p.conv1d_f32(x, false, &[], conv("h", "w1", 7, 3, 9, Activation::Snake, None)).unwrap();
        let b = p
            .conv1d_f32(a.output, false, &[a.done], conv("y", "w2", 1, 1, 0, Activation::Snake, Some(x)))
            .unwrap();
        let up = p
            .conv1d_f32(
                b.output,
                true,
                &[b.done],
                Conv1dF32Stage {
                    output: TensorRef::Named("up"),
                    weight: TensorRef::Named("wt"),
                    out_channels: 32,
                    kernel: 16,
                    stride: 8,
                    dilation_or_output_padding: 0,
                    pad_before: 4,
                    pad_after: 4,
                    input_activation: Activation::LeakyRelu,
                    slope: 0.1,
                    residual: None,
                    bias: None,
                    alpha: None,
                    ..conv("", "", 1, 1, 0, Activation::None, None)
                },
            )
            .unwrap();
        let noise = p
            .rand_f32(
                seed,
                &[],
                RandF32Stage {
                    output: TensorRef::Named("noise"),
                    items: 2,
                    rows: 800,
                    width: 1,
                    stream: 3,
                    stream_shift: 58,
                    a: RandCoord::Item,
                    b: RandCoord::Row,
                    a_offset: 0,
                    b_offset: 0,
                    normal: true,
                    shared_seed: true,
                    scale: 1.0,
                    offset: 0.0,
                },
            )
            .unwrap();
        let mixed = p
            .binary_f32(
                up.output,
                noise.output,
                &[up.done, noise.done],
                BinaryF32Stage {
                    output: TensorRef::Handle(up.output),
                    op: BinaryOp::Add,
                    items: 2,
                    rows: 800,
                    width: 32,
                    b_item_stride: 800,
                    b_row_stride: 1,
                    b_col_stride: 0,
                    scale: None,
                },
            )
            .unwrap();
        let prefix = p.finish(100);
        assert_eq!(prefix.programs, [0]);
        assert_eq!(prefix.output, mixed.output);
        let insts = &prefix.model.progs[0].insts;
        assert_eq!(insts.len(), 5);
        let ops: Vec<_> = insts.iter().map(|i| DevOp::from_u16(i.op).unwrap()).collect();
        assert_eq!(
            ops,
            [DevOp::Conv1dF32, DevOp::Conv1dF32, DevOp::ConvTranspose1dF32, DevOp::RandF32, DevOp::BinaryF32]
        );
        let c = &insts[0];
        assert_eq!(c.i, [2, 100, 64, 64, 7, 1, 3, 1]);
        assert_eq!(c.j[0], 9 | (9 << 16));
        assert_eq!(c.j[1], packet::dev::ACT_SNAKE << 4 | 1 << 13);
        assert_eq!(c.t[6], lengths);
        assert_eq!(insts[1].t[5], x);
        let t = &insts[2];
        assert_eq!(t.f[0], 0.1);
        assert_eq!(t.j[1], packet::dev::ACT_LEAKY_RELU << 4 | 1 << 13);
        assert_eq!(prefix.model.tensors[up.output as usize].bytes, 2 * 800 * 32 * 4);
        assert_eq!(prefix.model.tensors[insts[2].t[2] as usize].bytes, 64 * 32 * 16 * 4);
        assert_eq!(insts[3].i[5], 2);
        assert_eq!(insts[3].j[1], 3);
        assert_eq!(insts[4].t[0], insts[4].t[1]);
        assert!(!prefix.model.to_blob().is_empty());
    }

    #[test]
    fn single_stage_appends_and_validation() {
        let (prefix, h) = prefix_with(&[("q", 2 * 50 * 128 * 4), ("f0", 2 * 480 * 4)]);
        let prefix = prefix
            .append_attention_f32(
                h[0],
                h[0],
                h[0],
                AttentionF32Stage {
                    output: TensorRef::Named("ctx"),
                    key_lengths: None,
                    bias: None,
                    batch: 2,
                    q_rows: 50,
                    kv_rows: 50,
                    heads: 2,
                    head_width: 64,
                    in_stride: 0,
                    k_col0: 0,
                    v_col0: 0,
                    causal: true,
                    scale: 0.125,
                    bias_head_stride: 0,
                    prefix: None,
                },
            )
            .unwrap()
            .append_cumsum_f64(
                h[1],
                CumSumF64Stage {
                    output: TensorRef::Named("phase"),
                    column_scale: Some(TensorRef::Named("harmonics")),
                    lengths: None,
                    items: 2,
                    rows: 480,
                    width: 9,
                    x_width: 1,
                    exclusive: false,
                    wrap: true,
                    f64_output: false,
                    scale: 1.0 / 24000.0,
                    post_scale: std::f32::consts::TAU,
                },
            )
            .unwrap();
        assert_eq!(prefix.programs, [0, 1]);
        let a = &prefix.model.progs[0].insts[0];
        assert_eq!(DevOp::from_u16(a.op), Some(DevOp::AttentionF32));
        assert_eq!(a.i[6], 1 | 2);
        assert_eq!(prefix.model.tensors[prefix.output as usize].bytes, 2 * 480 * 9 * 4);
        let c = &prefix.model.progs[1].insts[0];
        assert_eq!(c.i[..5], [2, 480, 9, 1, 2]);
        let error = prefix
            .append_unary_f32(
                h[1],
                UnaryF32Stage {
                    output: TensorRef::Handle(h[1]),
                    param: None,
                    rows: 2,
                    width: 481,
                    stride: 0,
                    col0: 0,
                    kind: Activation::Tanh,
                    p0: 0.0,
                    p1: 0.0,
                },
            )
            .err()
            .unwrap();
        assert!(error.contains("smaller"), "{error}");
    }
}
