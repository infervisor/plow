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
                ("ordered_dispatch".into(), u64::from(spec.ordered_dispatch)),
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
    /// [`packet::dev::ACT_SNAKE_BETA`]: unary only (its parameter is two arrays).
    SnakeBeta,
    /// [`packet::dev::ACT_LOG`]: unary only.
    Log,
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
            Activation::SnakeBeta => ACT_SNAKE_BETA,
            Activation::Log => ACT_LOG,
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

/// [`DevOp::StreamTailF32`] mode.
#[derive(Clone, Copy)]
pub enum StreamTailF32Mode<'a> {
    /// `output` = `z`: the carried tail (zeros on a fresh stream) then `rows` source rows.
    Gather {
        output: TensorRef<'a>,
        src: u32,
        pos: u32,
        src_stride: u32,
    },
    /// The state takes `z` rows `[n[b], n[b] + tail)`.
    Store { z: u32 },
}

/// [`DevOp::ReduceRowsF32`] reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceRowsOp {
    Sum,
    Mean,
    Max,
    /// Unbiased (`n - 1`).
    Std,
    /// Mean then std: the output is `2 * width` wide.
    MeanStd,
}

#[derive(Clone, Copy)]
pub struct ReduceRowsF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub lengths: Option<u32>,
    pub items: u32,
    pub rows: u32,
    pub width: u32,
    pub op: ReduceRowsOp,
    /// 0 = `rows` (one segment).
    pub seg_rows: u32,
    /// 0 = `width`.
    pub stride: u32,
    pub col0: u32,
}

#[derive(Clone, Copy)]
pub struct LengthAffineU32Stage<'a> {
    pub output: TensorRef<'a>,
    pub min_with: Option<u32>,
    pub items: u32,
    pub mul: u32,
    pub add: i32,
    pub div: u32,
    pub lo: u32,
    pub hi: u32,
}

/// [`DevOp::TrimSilenceF32`]: `output` `[items][cap]` f32, `len_out` `[items]` u32.
#[derive(Clone, Copy)]
pub struct TrimSilenceF32Stage<'a> {
    pub output: TensorRef<'a>,
    pub len_out: TensorRef<'a>,
    pub items: u32,
    pub cap: u32,
    pub frame: u32,
    pub hop: u32,
    pub top_db: f32,
}

#[derive(Clone, Copy)]
pub struct StreamTailF32Stage<'a> {
    pub mode: StreamTailF32Mode<'a>,
    pub state: TensorRef<'a>,
    pub slot: u32,
    pub n: u32,
    pub items: u32,
    pub slots: u32,
    pub tail: u32,
    pub rows: u32,
    pub cols: u32,
    pub z_cols: u32,
    pub z_col0: u32,
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
    /// Flag bit 17: on sm_90a a split-bf16 conv (pointwise, grouped pointwise with group widths
    /// % 4 == 0, or tap-major with 64-channel groups) runs the same product on warpgroup MMA; the
    /// interpreter falls back to the other paths elsewhere.
    pub wgmma: bool,
    /// Flag bit 18: the weight tensor carries [`wgmma_weight_split`]'s image after the f32 weights,
    /// which an ungrouped wgmma conv copies as its B operand; without it such a conv takes the
    /// other paths.
    pub weight_split: bool,
    /// Flag bit 19: `alpha` is `[in_channels | out_channels]`, a snake input activation's
    /// parameters then a snake output activation's (one conv applying both).
    pub dual_alpha: bool,
}

/// The pre-split B operand of an ungrouped wgmma conv (flag bit 18), appended to its f32 weights
/// `[n][k]` (`k` = kernel * cin, tap-major): zero padding to the next 1024 bytes, then per 64-column
/// k-tile the bf16 hi halves then the lo halves (`lo = bf16(v - hi)`, both round-to-nearest-even)
/// of `n` rows padded to 128, each row 128 bytes whose 16-byte chunk `c` sits at `c ^ (row & 7)`.
pub fn wgmma_weight_split(weights: &[u8], n: u32, k: u32) -> Vec<u8> {
    let (n, k) = (n as usize, k as usize);
    let (base, np, tiles) = (
        weights.len().next_multiple_of(1024),
        n.next_multiple_of(128),
        k.div_ceil(64),
    );
    let mut out = weights.to_vec();
    out.resize(base + tiles * 2 * np * 128, 0);
    for (r, row) in weights.chunks_exact(4 * k).enumerate().take(n) {
        for (j, v) in row.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            let hi = crate::config::bf16_round(v);
            let lo = crate::config::bf16_round(v - hi);
            let at = base
                + (j / 64) * 2 * np * 128
                + r * 128
                + ((j % 64 / 8) ^ (r & 7)) * 16
                + j % 8 * 2;
            out[at..at + 2].copy_from_slice(&((hi.to_bits() >> 16) as u16).to_le_bytes());
            let at = at + np * 128;
            out[at..at + 2].copy_from_slice(&((lo.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    out
}

fn wgmma_weight_split_bytes(f32_bytes: u64, n: u32, k: u32) -> u64 {
    f32_bytes.next_multiple_of(1024)
        + u64::from(k.div_ceil(64)) * 2 * u64::from(n.next_multiple_of(128)) * 128
}

impl Conv1dF32Stage<'_> {
    /// A transposed conv the merged-phase wgmma path takes (flag bits 17 | 18; op_speech_f32.cuh
    /// `spg_ct_eligible`).
    pub fn convt_wgmma_ok(&self) -> bool {
        self.wgmma
            && self.split_bf16
            && self.weight_tap_major
            && !self.weight_f16
            && self.groups == 1
            && self.kernel == 2 * self.stride
            && self.dilation_or_output_padding == 0
            && self.input_activation == Activation::None
            && self.output_activation == Activation::None
            && self.alpha.is_none()
            && self.residual.is_none()
            && self.in_channels % 64 == 0
            && self.out_channels % 64 == 0
    }

    /// `(n, k)` of the pre-split wgmma B operand: `[out][kernel * in]`, or a transposed conv's
    /// merged phases `[stride * out][2 * in]`.
    pub fn split_geometry(&self, transpose: bool) -> (u32, u32) {
        if transpose {
            (self.stride * self.out_channels, 2 * self.in_channels)
        } else {
            (self.out_channels, self.in_channels * self.kernel)
        }
    }

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
            (u64::from(self.in_rows) + before + after).checked_sub(span)? / u64::from(self.stride)
                + 1
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
    /// The bias is a relative-position table `[q_rows][heads][bias_head_stride]`: score `(h, r, j)`
    /// adds entry `kv_rows - 1 - r + j` of row `r`, head `h` (q_rows == kv_rows, head_width 64).
    pub relative: bool,
    /// `key_lengths` holds one length per group of this many heads (0: one per item).
    pub key_length_heads: u32,
    /// Keys before each item's own: rows of a cached prefix block (see [`AttentionPrefix`]).
    pub prefix: Option<AttentionPrefix>,
    /// Causal sliding window (flag bits 16..31): key `j` is visible from row `r` iff
    /// `r - window < j <= r`; 0 = unbounded.
    pub window: u32,
    /// Stream mode (flag bit 3): see [`AttentionStream`].
    pub stream: Option<AttentionStream>,
}

/// Stream mode for [`AttentionF32Stage`]: query, key and value share each item's `kv_rows` rows;
/// the first `history` are carried keys (a [`DevOp::StreamTailF32`] gather), query `r` is row
/// `history + r` (`kv_rows = history + q_rows`), and keys before `history - min(pos[b], history)`
/// are absent (`pos`: u32 stream position per item).
#[derive(Clone, Copy)]
pub struct AttentionStream {
    pub pos: u32,
    pub history: u32,
}

/// A key prefix for [`AttentionF32Stage`]: item `b` first attends to the `rows` rows of block
/// `index[b]` of `table` (`[blocks][rows][K | V]`, each `2 * heads * head_width` wide, or with
/// `h16` blocks of [`h16_prefix_floats`] written by [`StageProgram::format_prefix_h16`]).
#[derive(Clone, Copy)]
pub struct AttentionPrefix {
    pub table: u32,
    pub index: u32,
    pub blocks: u32,
    pub rows: u32,
    pub h16: bool,
    /// u32[batch] (flag bit 5): item `b` reads only the first `min(rows_per_item[b], rows)` rows
    /// of its block; its own keys follow them directly.
    pub rows_per_item: Option<u32>,
}

/// Floats of one block of a pre-formatted attention prefix (`AttentionF32` flag bit 4): per
/// head, `rows / 16` (rounded up) tiles of 2072 (op_speech_f32.cuh SPH_XS).
pub fn h16_prefix_floats(heads: u32, rows: u32) -> u64 {
    u64::from(heads) * u64::from(rows.div_ceil(16)) * 2072
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
        let tensor_cores =
            (stage.input_bf16_exact || ln_round) && stage.weight_type == DenseWeight::Bf16;
        let flags = u32::from(stage.round_bf16)
            | (u32::from(stage.activation == DenseActivation::GeluErfBf16) << 1)
            | (u32::from(stage.weight_type == DenseWeight::Bf16) << 2)
            | (u32::from(tensor_cores) << 3)
            | (u32::from(stage.layer_norm.is_some()) << 5)
            | (u32::from(ln_round) << 6);
        let prologue = match stage.layer_norm {
            Some(ln) => {
                if ln.stats as usize >= builder.n_tensors()
                    || builder.tensor_bytes(ln.stats) < tensor_bytes(rows, 2, 4)?
                {
                    return Err("dense layer-norm stats tensor is missing or short".into());
                }
                let width = u64::from(stage.input_width) * 4;
                let affine = |b: &mut Builder, t: Option<TensorRef<'_>>| match t {
                    Some(TensorRef::Named(name)) => Ok(b.tensor(name, width)),
                    Some(TensorRef::Handle(h))
                        if (h as usize) < b.n_tensors() && b.tensor_bytes(h) >= width =>
                    {
                        Ok(h)
                    }
                    Some(TensorRef::Handle(_)) => {
                        Err("dense layer-norm affine tensor is missing".to_string())
                    }
                    None => Ok(packet::dev::TENSOR_NONE),
                };
                [
                    ln.stats,
                    affine(&mut builder, ln.gamma)?,
                    affine(&mut builder, ln.beta)?,
                ]
            }
            None => [packet::dev::TENSOR_NONE; 3],
        };
        // A reference split spreads small tiles over every CU (sp_tcn_gemm's packed split-K).
        let blocks = match stage.split {
            DenseSplit::Reference(_) if tensor_cores => self.model.n_cu,
            _ => dense_blocks(
                rows,
                stage.output_width,
                stage.input_width,
                tensor_cores,
                stage.split == DenseSplit::Parallel,
                self.model.n_cu,
            ),
        };
        if let DenseSplit::Reference(rows_ref) = stage.split {
            if !tensor_cores
                || stage.layer_norm.is_some()
                || stage.input_width % 8 != 0
                || rows_ref as usize >= builder.n_tensors()
                || builder.tensor_bytes(rows_ref) < u64::from(rows) * 4
            {
                return Err(
                    "reference split needs bf16 tensor cores, no prologue and a row table".into(),
                );
            }
        }
        // Declared under either split, so both lowerings share one tensor table.
        let splitk = if tensor_cores {
            builder.tensor(
                DENSE_SPLITK_SCRATCH,
                (u64::from(DENSE_SPLITK_TICKETS)
                    + u64::from(self.model.n_cu) * DENSE_SPLITK_SLOTS_PER_CU * 64 * 64)
                    * 4,
            )
        } else {
            packet::dev::TENSOR_NONE
        };
        let scratch = match stage.split {
            DenseSplit::Reference(rows_ref) if tensor_cores => rows_ref,
            _ => splitk,
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
                if reference {
                    instruction.t[5] = splitk;
                }
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
            repeated(
                self.model.n_cu,
                elements.div_ceil(1024).min(self.model.n_cu),
            ),
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
            repeated(
                self.model.n_cu,
                elements.div_ceil(1024).min(self.model.n_cu),
            ),
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
    elements
        .checked_mul(4)
        .ok_or_else(|| "tensor size overflows".into())
}

fn product(values: &[u32]) -> Result<u64, String> {
    values
        .iter()
        .try_fold(1u64, |acc, &v| acc.checked_mul(u64::from(v)))
        .ok_or_else(|| "geometry overflows".into())
}

/// Elements a strided `[items][rows][cols]` view spans: the last element's offset plus one.
fn strided_span(
    items: u32,
    item_stride: u32,
    rows: u32,
    stride: u32,
    offset: u32,
    cols: u32,
) -> u64 {
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
        if (handle as usize) < self.builder.n_tensors()
            && self.builder.tensor_bytes(handle) >= bytes
        {
            Ok(handle)
        } else {
            Err(format!(
                "{what}: tensor {handle} is missing or smaller than {bytes} bytes"
            ))
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

    /// Any op with explicit operands, sliced over `units` blocks (ops without a stage helper).
    pub(crate) fn raw(
        &mut self,
        op: DevOp,
        units: u64,
        deps: &[u32],
        output: u32,
        fill: impl FnOnce(&mut packet::dev::DevInst),
    ) -> Result<Emitted, String> {
        self.emit(op, units, deps, output, fill)
    }

    /// Declare (or re-declare, keeping the larger size) a tensor by name.
    pub(crate) fn declare(&mut self, name: &str, bytes: u64) -> u32 {
        self.builder.tensor(name, bytes)
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
        let per_item = if stage.rows_per_item == 0 {
            stage.rows
        } else {
            stage.rows_per_item
        };
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
        let out_stride = if stage.out_stride == 0 {
            stage.width
        } else {
            stage.out_stride
        };
        let span = strided_span(1, 0, stage.rows, out_stride, stage.out_col0, stage.width);
        let output = self.resolve(stage.output, f32_bytes(span)?, "row-gather output")?;
        let elements = product(&[stage.rows, stage.width])?;
        u32_elements(elements, "row gather")?;
        let flags = u32::from(stage.table_f16) | (u32::from(stage.accumulate) << 1);
        self.emit(
            DevOp::GatherRowsF32,
            elements.div_ceil(2048),
            deps,
            output,
            |d| {
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
            },
        )
    }

    /// With `group` (u32[1], flag bit 1) the items land at output items `g*items ..` with `g`
    /// read on device; a `g` whose items would pass the output tensor's end writes nothing.
    pub fn copy_cols_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: CopyColsF32Stage<'_>,
        group: Option<u32>,
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
        let tail = span - u64::from(stage.items - 1) * u64::from(stage.out_item_stride);
        let flags = self.copy_group(group, output, &stage, tail)?;
        let elements = product(&[stage.items, stage.rows, stage.cols])?;
        u32_elements(elements, "column copy")?;
        self.emit(
            DevOp::CopyColsF32,
            elements.div_ceil(2048),
            deps,
            output,
            |d| {
                d.t[..3].copy_from_slice(&[output, x, group.unwrap_or(packet::dev::TENSOR_NONE)]);
                d.i = [
                    stage.items,
                    stage.rows,
                    stage.cols,
                    stage.in_stride,
                    stage.in_offset,
                    stage.out_stride,
                    stage.out_offset,
                    flags,
                ];
                d.j = [stage.in_item_stride, stage.out_item_stride];
            },
        )
    }

    /// CopyColsF32 flag bits for an optional device group index: bit 1 and, in bits 8..31, how
    /// many items of `out_item_stride` fit the output when the last one spans `tail` floats.
    fn copy_group(
        &self,
        group: Option<u32>,
        output: u32,
        stage: &CopyColsF32Stage<'_>,
        tail: u64,
    ) -> Result<u32, String> {
        let Some(group) = group else {
            return Ok(0);
        };
        self.input(group, 4, "column-copy group")?;
        if stage.out_item_stride == 0 {
            return Err("column-copy group: out_item_stride must be nonzero".into());
        }
        let floats = self.builder.tensor_bytes(output) / 4;
        let capacity = (floats - tail) / u64::from(stage.out_item_stride) + 1;
        Ok(2 | (capacity.min(0xFF_FFFF) as u32) << 8)
    }

    /// [`DevOp::CopyColsF32`] format mode: `items` blocks of `rows` keys (`cols` = K | V of
    /// `cols / 128` heads of width 64) from the column-copy source geometry into an attention
    /// prefix table for [`AttentionPrefix::h16`]; block `b` at `b * out_item_stride + out_offset`
    /// (`out_stride` unused).
    /// `group` as [`Self::copy_cols_f32`].
    pub fn format_prefix_h16(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: CopyColsF32Stage<'_>,
        group: Option<u32>,
    ) -> Result<Emitted, String> {
        if stage.items == 0 || stage.rows == 0 || stage.cols == 0 || stage.cols % 128 != 0 {
            return Err("invalid prefix-format geometry".into());
        }
        let heads = stage.cols / 128;
        let block = h16_prefix_floats(heads, stage.rows);
        if u64::from(stage.out_item_stride) < block {
            return Err("prefix-format blocks overlap".into());
        }
        let span = strided_span(
            stage.items,
            stage.in_item_stride,
            stage.rows,
            stage.in_stride,
            stage.in_offset,
            stage.cols,
        );
        self.input(x, f32_bytes(span)?, "prefix-format source")?;
        let span = u64::from(stage.out_offset)
            + u64::from(stage.items - 1) * u64::from(stage.out_item_stride)
            + block;
        let output = self.resolve(stage.output, f32_bytes(span)?, "prefix-format output")?;
        let flags = self.copy_group(group, output, &stage, u64::from(stage.out_offset) + block)?;
        let units = product(&[stage.items, heads, stage.rows.div_ceil(16)])?.div_ceil(2);
        self.emit(DevOp::CopyColsF32, units, deps, output, |d| {
            d.t[..3].copy_from_slice(&[output, x, group.unwrap_or(packet::dev::TENSOR_NONE)]);
            d.i = [
                stage.items,
                stage.rows,
                stage.cols,
                stage.in_stride,
                stage.in_offset,
                0,
                stage.out_offset,
                1 | flags,
            ];
            d.j = [stage.in_item_stride, stage.out_item_stride];
        })
    }

    /// [`DevOp::StreamTailF32`]: the emitted output is `z` (gather) or the state (store).
    pub fn stream_tail_f32(
        &mut self,
        deps: &[u32],
        stage: StreamTailF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.items == 0
            || stage.slots == 0
            || stage.tail == 0
            || stage.rows == 0
            || stage.cols == 0
            || u64::from(stage.z_col0) + u64::from(stage.cols) > u64::from(stage.z_cols)
        {
            return Err("invalid stream-tail geometry".into());
        }
        let state_bytes = f32_bytes(product(&[stage.slots, stage.tail, stage.cols])?)?;
        let state = self.resolve(stage.state, state_bytes, "stream-tail state")?;
        let per_item = u64::from(stage.items) * 4;
        self.input(stage.slot, per_item, "stream-tail slots")?;
        self.input(stage.n, per_item, "stream-tail new rows")?;
        let z_rows = stage
            .tail
            .checked_add(stage.rows)
            .ok_or("stream-tail rows overflow")?;
        let z_elements = product(&[stage.items, z_rows, stage.z_cols])?;
        let z_bytes = f32_bytes(z_elements)?;
        let ints = |src_stride, mode| {
            [
                stage.items,
                stage.tail,
                stage.rows,
                stage.cols,
                stage.z_cols,
                stage.z_col0,
                src_stride,
                mode,
            ]
        };
        match stage.mode {
            StreamTailF32Mode::Gather {
                output,
                src,
                pos,
                src_stride,
            } => {
                self.input(pos, per_item, "stream-tail positions")?;
                let span = (product(&[stage.items, stage.rows])? - 1) * u64::from(src_stride)
                    + u64::from(stage.z_cols);
                self.input(src, f32_bytes(span)?, "stream-tail source")?;
                let z = self.resolve(output, z_bytes, "stream-tail output")?;
                let elements = u32_elements(z_elements, "stream-tail gather")?;
                self.emit(
                    DevOp::StreamTailF32,
                    u64::from(elements).div_ceil(2048),
                    deps,
                    z,
                    |d| {
                        d.t[..6].copy_from_slice(&[z, src, state, stage.slot, stage.n, pos]);
                        d.i = ints(src_stride, 0);
                        d.j = [stage.slots, 0];
                    },
                )
            }
            StreamTailF32Mode::Store { z } => {
                self.input(z, z_bytes, "stream-tail z")?;
                let elements = product(&[stage.items, stage.tail, stage.cols])?;
                u32_elements(elements, "stream-tail store")?;
                self.emit(
                    DevOp::StreamTailF32,
                    elements.div_ceil(2048),
                    deps,
                    state,
                    |d| {
                        d.t[1..5].copy_from_slice(&[z, state, stage.slot, stage.n]);
                        d.i = ints(0, 1);
                        d.j = [stage.slots, 0];
                    },
                )
            }
        }
    }

    /// [`DevOp::ReduceRowsF32`]: output `[items][ceil(rows / seg_rows)][width or 2 * width]`.
    pub fn reduce_rows_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: ReduceRowsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        let seg_rows = if stage.seg_rows == 0 {
            stage.rows
        } else {
            stage.seg_rows
        };
        let stride = if stage.stride == 0 {
            stage.width
        } else {
            stage.stride
        };
        if stage.items == 0
            || stage.rows == 0
            || stage.width == 0
            || u64::from(stage.col0) + u64::from(stage.width) > u64::from(stride)
        {
            return Err("invalid row-reduction geometry".into());
        }
        let x_rows = product(&[stage.items, stage.rows])?;
        self.input(
            x,
            f32_bytes(x_rows * u64::from(stride))?,
            "row-reduction input",
        )?;
        if let Some(lengths) = stage.lengths {
            self.input(lengths, u64::from(stage.items) * 4, "row-reduction lengths")?;
        }
        let (op, out_width) = match stage.op {
            ReduceRowsOp::Sum => (0, stage.width),
            ReduceRowsOp::Mean => (1, stage.width),
            ReduceRowsOp::Max => (2, stage.width),
            ReduceRowsOp::Std => (3, stage.width),
            ReduceRowsOp::MeanStd => (4, stage.width.checked_mul(2).ok_or("width overflows")?),
        };
        let segs = stage.rows.div_ceil(seg_rows);
        let out_elements = product(&[stage.items, segs, out_width])?;
        u32_elements(out_elements, "row reduction")?;
        let output = self.resolve(
            stage.output,
            f32_bytes(out_elements)?,
            "row-reduction output",
        )?;
        let threads = product(&[stage.items, segs, stage.width])?;
        self.emit(
            DevOp::ReduceRowsF32,
            threads.div_ceil(256),
            deps,
            output,
            |d| {
                d.t[..3].copy_from_slice(&[
                    output,
                    x,
                    stage.lengths.unwrap_or(packet::dev::TENSOR_NONE),
                ]);
                d.i[..7].copy_from_slice(&[
                    stage.items,
                    stage.rows,
                    stage.width,
                    op,
                    stage.seg_rows,
                    stage.stride,
                    stage.col0,
                ]);
            },
        )
    }

    /// [`DevOp::LengthAffineU32`].
    pub fn length_affine_u32(
        &mut self,
        input: u32,
        deps: &[u32],
        stage: LengthAffineU32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.items == 0 || stage.div == 0 || stage.lo > stage.hi {
            return Err("invalid length-affine stage".into());
        }
        let bytes = u64::from(stage.items) * 4;
        self.input(input, bytes, "length-affine input")?;
        if let Some(min_with) = stage.min_with {
            self.input(min_with, bytes, "length-affine bound")?;
        }
        let output = self.resolve(stage.output, bytes, "length-affine output")?;
        self.emit(DevOp::LengthAffineU32, 1, deps, output, |d| {
            d.t[..3].copy_from_slice(&[
                output,
                input,
                stage.min_with.unwrap_or(packet::dev::TENSOR_NONE),
            ]);
            d.i[..6].copy_from_slice(&[
                stage.items,
                stage.mul,
                stage.add as u32,
                stage.div,
                stage.lo,
                stage.hi,
            ]);
        })
    }

    /// [`DevOp::TrimSilenceF32`] over `pcm` (`[items][cap]` f32) and `len_in` (`[items]` u32).
    /// Returns the trimmed signal and the `len_out` handle.
    pub fn trim_silence_f32(
        &mut self,
        pcm: u32,
        len_in: u32,
        deps: &[u32],
        stage: TrimSilenceF32Stage<'_>,
    ) -> Result<(Emitted, u32), String> {
        if stage.items == 0
            || stage.cap == 0
            || stage.frame == 0
            || stage.hop == 0
            || !stage.top_db.is_finite()
        {
            return Err("invalid silence-trim stage".into());
        }
        let bytes = f32_bytes(product(&[stage.items, stage.cap])?)?;
        self.input(pcm, bytes, "silence-trim input")?;
        let lengths = u64::from(stage.items) * 4;
        self.input(len_in, lengths, "silence-trim lengths")?;
        let output = self.resolve(stage.output, bytes, "silence-trim output")?;
        let len_out = self.resolve(stage.len_out, lengths, "silence-trim output lengths")?;
        if output == pcm || len_out == len_in {
            return Err("silence trim: outputs must not alias the inputs".into());
        }
        let e = self.emit(
            DevOp::TrimSilenceF32,
            u64::from(stage.items),
            deps,
            output,
            |d| {
                d.t[..4].copy_from_slice(&[output, pcm, len_in, len_out]);
                d.i[..4].copy_from_slice(&[stage.items, stage.cap, stage.frame, stage.hop]);
                d.f[0] = stage.top_db;
            },
        )?;
        Ok((e, len_out))
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
        let valid_activation = |a: Activation| {
            !matches!(
                a,
                Activation::Clamp | Activation::ScaleShift | Activation::Log
            )
        };
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
            // input takes no activation unless bit 19 splits the tensor between two snakes.
            || (stage.output_activation == Activation::Snake
                && stage.input_activation != Activation::None
                && !stage.dual_alpha)
            || (stage.dual_alpha
                && !(stage.input_activation == Activation::Snake
                    && stage.output_activation == Activation::Snake
                    && stage.alpha.is_some()))
            || (stage.weight_split && transpose && !stage.convt_wgmma_ok())
            || (stage.weight_split
                && !transpose
                && (stage.groups != 1
                    || stage.weight_f16
                    || !(stage.wgmma && stage.split_bf16)
                    || !(stage.kernel == 1 || stage.weight_tap_major)))
        {
            return Err("invalid convolution geometry".into());
        }
        let out_rows = stage
            .out_rows(transpose)
            .ok_or("convolution output is empty")?;
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
        let mut weight_bytes =
            product(&[stage.out_channels, stage.in_channels / stage.groups, taps])?
                * if stage.weight_f16 { 2 } else { 4 };
        if stage.weight_split {
            let (n, k) = stage.split_geometry(transpose);
            weight_bytes = wgmma_weight_split_bytes(weight_bytes, n, k);
        }
        let weight = self.resolve(stage.weight, weight_bytes, "convolution weight")?;
        let bias = match stage.bias {
            Some(bias) => {
                self.resolve(bias, u64::from(stage.out_channels) * 4, "convolution bias")?
            }
            None => packet::dev::TENSOR_NONE,
        };
        let alpha_channels = if stage.dual_alpha {
            stage.in_channels + stage.out_channels
        } else {
            stage.in_channels.max(stage.out_channels)
        };
        let alpha = match stage.alpha {
            Some(alpha) => {
                self.resolve(alpha, u64::from(alpha_channels) * 4, "convolution alpha")?
            }
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
            | (u32::from(stage.wgmma && stage.split_bf16) << 17)
            | (u32::from(stage.weight_split) << 18)
            | (u32::from(stage.dual_alpha) << 19);
        let op = if transpose {
            DevOp::ConvTranspose1dF32
        } else {
            DevOp::Conv1dF32
        };
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
    pub fn unary_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: UnaryF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.width == 0 {
            return Err("invalid unary geometry".into());
        }
        let stride = if stage.stride == 0 {
            stage.width
        } else {
            stage.stride
        };
        let bytes = f32_bytes(strided_span(
            1,
            0,
            stage.rows,
            stride,
            stage.col0,
            stage.width,
        ))?;
        self.input(x, bytes, "unary input")?;
        let output = self.resolve(stage.output, bytes, "unary output")?;
        let param = match stage.param {
            Some(param) => {
                let arrays = if stage.kind == Activation::SnakeBeta {
                    2
                } else {
                    1
                };
                self.resolve(
                    param,
                    u64::from(stage.width) * 4 * arrays,
                    "unary parameter",
                )?
            }
            None if stage.kind == Activation::SnakeBeta => {
                return Err("SnakeBeta needs its alpha/beta parameter".into())
            }
            None => packet::dev::TENSOR_NONE,
        };
        let elements = product(&[stage.rows, stage.width])?;
        u32_elements(elements, "unary")?;
        self.emit(
            DevOp::UnaryF32,
            elements.div_ceil(2048),
            deps,
            output,
            |d| {
                d.t[..3].copy_from_slice(&[output, x, param]);
                d.i[..5].copy_from_slice(&[
                    stage.rows,
                    stage.width,
                    stage.kind.code(),
                    stage.stride,
                    stage.col0,
                ]);
                d.f = [stage.p0, stage.p1];
            },
        )
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
        self.emit(
            DevOp::BinaryF32,
            elements.div_ceil(2048),
            deps,
            output,
            |d| {
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
            },
        )
    }

    pub fn cumsum_f64(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: CumSumF64Stage<'_>,
    ) -> Result<Emitted, String> {
        let x_width = if stage.x_width == 0 {
            stage.width
        } else {
            stage.x_width
        };
        if stage.items == 0 || stage.rows == 0 || stage.width == 0 {
            return Err("invalid prefix-sum geometry".into());
        }
        self.input(
            x,
            f32_bytes(product(&[stage.items, stage.rows, x_width])?)?,
            "prefix-sum input",
        )?;
        let elements = product(&[stage.items, stage.rows, stage.width])?;
        let output = self.resolve(
            stage.output,
            elements * if stage.f64_output { 8 } else { 4 },
            "prefix-sum output",
        )?;
        let column_scale = match stage.column_scale {
            Some(scale) => {
                self.resolve(scale, u64::from(stage.width) * 4, "prefix-sum column scale")?
            }
            None => packet::dev::TENSOR_NONE,
        };
        if let Some(lengths) = stage.lengths {
            self.input(lengths, u64::from(stage.items) * 4, "prefix-sum lengths")?;
        }
        let flags = u32::from(stage.exclusive)
            | (u32::from(stage.wrap) << 1)
            | (u32::from(stage.f64_output) << 2);
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

    pub fn rand_f32(
        &mut self,
        seed: u32,
        deps: &[u32],
        stage: RandF32Stage<'_>,
    ) -> Result<Emitted, String> {
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
            || !(stage.head_width == 64
                || stage.head_width == 256
                || (stage.head_width % 8 == 0 && stage.head_width <= 128))
        {
            return Err("invalid attention geometry".into());
        }
        if stage.head_width == 256
            && (stage.prefix.is_some() || stage.stream.is_some() || stage.relative)
        {
            return Err("attention head_width 256: no prefix, stream or relative bias".into());
        }
        let width = stage
            .heads
            .checked_mul(stage.head_width)
            .ok_or("attention width overflows")?;
        let stride = if stage.in_stride == 0 {
            width
        } else {
            stage.in_stride
        };
        let q_total = u32_elements(
            product(&[stage.batch, stage.q_rows])?,
            "attention query rows",
        )?;
        let kv_total = u32_elements(
            product(&[stage.batch, stage.kv_rows])?,
            "attention key rows",
        )?;
        self.input(
            query,
            f32_bytes(strided_span(1, 0, q_total, stride, 0, width))?,
            "attention query",
        )?;
        for (tensor, col0, what) in [
            (key, stage.k_col0, "attention key"),
            (value, stage.v_col0, "attention value"),
        ] {
            self.input(
                tensor,
                f32_bytes(strided_span(1, 0, kv_total, stride, col0, width))?,
                what,
            )?;
        }
        let groups = match stage.key_length_heads {
            0 => 1,
            g if g <= 255 && stage.heads % g == 0 => stage.heads / g,
            _ => {
                return Err(
                    "attention key length groups must divide the heads (at most 255 per group)"
                        .into(),
                )
            }
        };
        if let Some(lengths) = stage.key_lengths {
            self.input(
                lengths,
                u64::from(stage.batch) * u64::from(groups) * 4,
                "attention key lengths",
            )?;
        }
        if stage.relative
            && (stage.bias.is_none()
                || stage.q_rows != stage.kv_rows
                || stage.head_width != 64
                || stage.q_rows <= 112
                || stage.bias_head_stride < 2 * stage.kv_rows - 1)
        {
            return Err(
                "relative attention: a table bias, q_rows == kv_rows > 112, head_width 64".into(),
            );
        }
        if stage.key_length_heads != 0 && (!stage.relative || stage.q_rows <= 112) {
            return Err("attention key length groups: relative attention only".into());
        }
        if stage.window != 0 && (!stage.causal || stage.window > 0xFFFF || stage.prefix.is_some()) {
            return Err("attention window: causal, at most 65535 rows, no prefix".into());
        }
        if let Some(st) = stage.stream {
            if !stage.causal
                || stage.prefix.is_some()
                || stage.bias.is_some()
                || stage.relative
                || stage.key_length_heads != 0
                || st.history == 0
                || stage.kv_rows != st.history + stage.q_rows
            {
                return Err(
                    "attention stream: causal, no prefix or bias, kv_rows = history + q_rows"
                        .into(),
                );
            }
            self.input(
                st.pos,
                u64::from(stage.batch) * 4,
                "attention stream positions",
            )?;
        }
        if let Some(p) = stage.prefix {
            if stage.bias.is_some() || stage.causal || p.rows == 0 {
                return Err(
                    "attention prefix: no bias or causal mask, and at least one row".into(),
                );
            }
            if let Some(rows) = p.rows_per_item {
                self.input(
                    rows,
                    u64::from(stage.batch) * 4,
                    "attention prefix rows per item",
                )?;
            }
            if p.h16 && (stage.head_width != 64 || stage.q_rows.div_ceil(16) < 4) {
                return Err(
                    "pre-formatted attention prefix: head_width 64 and at least 4 query tiles"
                        .into(),
                );
            }
            let floats = if p.h16 {
                u64::from(p.blocks) * h16_prefix_floats(stage.heads, p.rows)
            } else {
                product(&[p.blocks, p.rows, 2 * width])?
            };
            self.input(p.table, f32_bytes(floats)?, "attention prefix")?;
            self.input(
                p.index,
                u64::from(stage.batch) * 4,
                "attention prefix index",
            )?;
        }
        let bias = match (stage.bias, stage.prefix.and_then(|p| p.rows_per_item)) {
            (_, Some(rows)) => rows,
            (Some(bias), None) => {
                let span = if stage.relative {
                    product(&[stage.q_rows, stage.heads, stage.bias_head_stride])?
                } else {
                    u64::from(stage.heads - 1) * u64::from(stage.bias_head_stride)
                        + product(&[stage.q_rows, stage.kv_rows])?
                };
                self.resolve(bias, f32_bytes(span)?, "attention bias")?
            }
            (None, None) => packet::dev::TENSOR_NONE,
        };
        let out_elements = product(&[stage.batch, stage.q_rows, width])?;
        let output = self.resolve(stage.output, f32_bytes(out_elements)?, "attention output")?;
        // head_width 64 runs 3xTF32 tensor cores (flag bit 1), one block per run of 8 16-row query
        // tiles in each item's (head, tile) order (per head when a head has a single tile).
        let tc = stage.head_width == 64 && stage.stream.is_none();
        let n16 = stage.q_rows.div_ceil(16);
        let units = match (tc, n16 >= 2) {
            (true, true) => {
                let tiles = stage
                    .heads
                    .checked_mul(n16)
                    .ok_or("attention tiles overflow")?;
                product(&[stage.batch, tiles.div_ceil(8)])?
            }
            (true, false) => product(&[stage.batch, stage.heads])?,
            // head_width 256 runs 32-query tiles.
            (false, _) if stage.head_width == 256 => {
                product(&[stage.batch, stage.heads, stage.q_rows.div_ceil(32)])?
            }
            (false, _) => product(&[stage.batch, stage.heads, stage.q_rows.div_ceil(64)])?,
        };
        self.emit(DevOp::AttentionF32, units, deps, output, |d| {
            d.t.copy_from_slice(&[
                output,
                query,
                key,
                value,
                stage.key_lengths.unwrap_or(packet::dev::TENSOR_NONE),
                bias,
                stage.prefix.map_or(packet::dev::TENSOR_NONE, |p| p.table),
                stage
                    .prefix
                    .map(|p| p.index)
                    .or(stage.stream.map(|st| st.pos))
                    .unwrap_or(packet::dev::TENSOR_NONE),
            ]);
            d.i = [
                stage.batch,
                stage.q_rows,
                stage.kv_rows,
                stage.heads,
                stage.head_width,
                stage.in_stride,
                u32::from(stage.causal)
                    | (u32::from(tc) << 1)
                    | (u32::from(stage.relative) << 2)
                    | (u32::from(stage.stream.is_some()) << 3)
                    | (u32::from(stage.prefix.is_some_and(|p| p.h16)) << 4)
                    | (u32::from(stage.prefix.is_some_and(|p| p.rows_per_item.is_some())) << 5)
                    | (stage.key_length_heads << 8)
                    | (stage.window << 16),
                stage.prefix.map_or(
                    stage.stream.map_or(stage.bias_head_stride, |st| st.history),
                    |p| p.rows,
                ),
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
    /// Flag bits 4-7: a parameterless activation after the affine.
    pub activation: Activation,
    /// A `[width]` row added after the activation.
    pub add_row: Option<TensorRef<'a>>,
}

impl StageProgram {
    /// In place when `output` is `TensorRef::Handle(x)`.
    pub fn layer_norm_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: LayerNormRowsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        let parameterless = !matches!(
            stage.activation,
            Activation::LeakyRelu
                | Activation::Snake
                | Activation::Clamp
                | Activation::ScaleShift
                | Activation::Log
        );
        if stage.rows == 0
            || stage.width == 0
            || !stage.epsilon.is_finite()
            || stage.epsilon <= 0.0
            || !parameterless
        {
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
        let add = affine(stage.add_row, "layer-norm row")?;
        // One warp per row, eight warps per slice.
        self.emit(
            DevOp::LayerNormF32,
            u64::from(stage.rows).div_ceil(8),
            deps,
            output,
            |d| {
                d.t[..5].copy_from_slice(&[output, x, gamma, beta, add]);
                d.i[..3].copy_from_slice(&[stage.rows, stage.width, stage.activation.code() << 4]);
                d.f[0] = stage.epsilon;
            },
        )
    }
}

impl StageProgram {
    /// RowStatsF32: `[rows][2]` = LayerNormF32's (mean, inverse std) per row.
    pub fn row_stats_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: RowStatsF32Stage<'_>,
    ) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.width == 0 || !stage.epsilon.is_finite() || stage.epsilon <= 0.0
        {
            return Err("invalid row-statistics stage".into());
        }
        self.input(
            x,
            f32_bytes(product(&[stage.rows, stage.width])?)?,
            "row-statistics input",
        )?;
        let output = self.resolve(
            stage.output,
            f32_bytes(product(&[stage.rows, 2])?)?,
            "row statistics",
        )?;
        // One warp per row, eight warps per slice (LayerNormF32's decomposition).
        self.emit(
            DevOp::RowStatsF32,
            u64::from(stage.rows).div_ceil(8),
            deps,
            output,
            |d| {
                d.t[..2].copy_from_slice(&[output, x]);
                d.i[..3].copy_from_slice(&[
                    stage.rows,
                    stage.width,
                    u32::from(stage.ordered_statistics) << 1,
                ]);
                d.f[0] = stage.epsilon;
            },
        )
    }

    /// DenseGemmF32, optionally with the LayerNorm prologue.
    pub fn dense_f32(
        &mut self,
        x: u32,
        deps: &[u32],
        stage: DenseF32ProgramStage<'_>,
    ) -> Result<Emitted, String> {
        if stage.rows == 0 || stage.input_width == 0 || stage.output_width == 0 {
            return Err("invalid dense geometry".into());
        }
        self.input(
            x,
            f32_bytes(product(&[stage.rows, stage.input_width])?)?,
            "dense input",
        )?;
        let esize = match stage.weight_type {
            DenseWeight::F32 => 4,
            DenseWeight::Bf16 => 2,
        };
        let weight = self.resolve(
            stage.weight,
            product(&[stage.output_width, stage.input_width])? * esize,
            "dense weight",
        )?;
        let bias = match stage.bias {
            Some(b) => self.resolve(b, u64::from(stage.output_width) * 4, "dense bias")?,
            None => packet::dev::TENSOR_NONE,
        };
        let prologue = match stage.layer_norm {
            Some(ln) => {
                self.input(
                    ln.stats,
                    f32_bytes(product(&[stage.rows, 2])?)?,
                    "dense layer-norm stats",
                )?;
                let width = u64::from(stage.input_width) * 4;
                let mut affine = |t: Option<TensorRef<'_>>, what| match t {
                    Some(t) => self.resolve(t, width, what),
                    None => Ok(packet::dev::TENSOR_NONE),
                };
                [
                    ln.stats,
                    affine(ln.gamma, "dense layer-norm gamma")?,
                    affine(ln.beta, "dense layer-norm beta")?,
                ]
            }
            None => [packet::dev::TENSOR_NONE; 3],
        };
        let output = self.resolve(
            stage.output,
            f32_bytes(product(&[stage.rows, stage.output_width])?)?,
            "dense output",
        )?;
        let ln_round = stage.layer_norm.is_some_and(|ln| ln.round_bf16);
        let tensor_cores =
            stage.operands_bf16_exact || (ln_round && stage.weight_type == DenseWeight::Bf16);
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
        self.input(
            row_scale,
            f32_bytes(product(&[stage.batch, out_rows])?)?,
            "convolution row scale",
        )?;
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
        self.append_one(stage.rows, |p| p.copy_cols_f32(x, &[], stage, None))
    }

    pub fn append_conv1d_f32(self, x: u32, stage: Conv1dF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.in_rows, |p| p.conv1d_f32(x, false, &[], stage))
    }

    pub fn append_conv_transpose1d_f32(
        self,
        x: u32,
        stage: Conv1dF32Stage<'_>,
    ) -> Result<Self, String> {
        self.append_one(stage.in_rows, |p| p.conv1d_f32(x, true, &[], stage))
    }

    pub fn append_unary_f32(self, x: u32, stage: UnaryF32Stage<'_>) -> Result<Self, String> {
        self.append_one(stage.rows, |p| p.unary_f32(x, &[], stage))
    }

    pub fn append_binary_f32(
        self,
        a: u32,
        b: u32,
        stage: BinaryF32Stage<'_>,
    ) -> Result<Self, String> {
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
        self.append_one(stage.q_rows, |p| {
            p.attention_f32(query, key, value, &[], stage)
        })
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
        u64::from(stage.groups)
            * u64::from(phases)
            * (batch * per_phase).div_ceil(tile)
            * u64::from(ng).div_ceil(tile)
    };
    // 3xTF32 on 64x64 tiles, except FP32 FFMA on 128x128 tiles when those fill the machine for
    // a transposed conv or one with an input activation.
    let ffma = ((transpose && !stage.weight_tap_major)
        || (!transpose && stage.input_activation != Activation::None))
        && tiles(128) >= u64::from(n_cu);
    if ffma {
        tiles(128)
    } else {
        tiles(64)
    }
}

/// Split-K scratch shared by every tensor-core DenseGemmF32 (each in its own program, so they
/// never overlap): `DENSE_SPLITK_TICKETS` zeroed u32 tickets, then `DENSE_SPLITK_SLOTS_PER_CU`
/// 64x64 f32 partials per CU (a parallel split uses one per block; a reference split's chunk
/// slots, SPN_SPLIT_FLOATS in op_speech_f32.cuh, use all of them).
const DENSE_SPLITK_SCRATCH: &str = "scratch.dense_f32_splitk";
const DENSE_SPLITK_TICKETS: u32 = 1024;
const DENSE_SPLITK_SLOTS_PER_CU: u64 = 8;

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
    fn convt_wgmma_split_geometry() {
        let mut s = Conv1dF32Stage {
            output: TensorRef::Named("o"),
            weight: TensorRef::Named("w"),
            bias: None,
            alpha: None,
            residual: None,
            lengths: None,
            batch: 2,
            in_rows: 33,
            in_channels: 1024,
            out_channels: 512,
            kernel: 16,
            stride: 8,
            dilation_or_output_padding: 0,
            groups: 1,
            pad_before: 4,
            pad_after: 4,
            pad_mode: PadMode::Zero,
            input_activation: Activation::None,
            output_activation: Activation::None,
            slope: 0.0,
            weight_f16: false,
            split_bf16: true,
            weight_tap_major: true,
            wgmma: true,
            weight_split: true,
            dual_alpha: false,
        };
        assert!(s.convt_wgmma_ok());
        assert_eq!(s.split_geometry(true), (8 * 512, 2 * 1024));
        assert_eq!(s.split_geometry(false), (512, 16 * 1024));
        s.input_activation = Activation::Snake;
        assert!(!s.convt_wgmma_ok());
        s.input_activation = Activation::None;
        s.out_channels = 96;
        assert!(!s.convt_wgmma_ok());
        s.out_channels = 512;
        s.kernel = 8;
        assert!(!s.convt_wgmma_ok());
    }

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
        let handles: Vec<u32> = tensors
            .iter()
            .map(|&(name, bytes)| builder.tensor(name, bytes))
            .collect();
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
            weight_split: false,
            dual_alpha: false,
        };
        let mut p = prefix.program();
        let a = p
            .conv1d_f32(
                x,
                false,
                &[],
                conv("h", "w1", 7, 3, 9, Activation::Snake, None),
            )
            .unwrap();
        let b = p
            .conv1d_f32(
                a.output,
                false,
                &[a.done],
                conv("y", "w2", 1, 1, 0, Activation::Snake, Some(x)),
            )
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
        let ops: Vec<_> = insts
            .iter()
            .map(|i| DevOp::from_u16(i.op).unwrap())
            .collect();
        assert_eq!(
            ops,
            [
                DevOp::Conv1dF32,
                DevOp::Conv1dF32,
                DevOp::ConvTranspose1dF32,
                DevOp::RandF32,
                DevOp::BinaryF32
            ]
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
        assert_eq!(
            prefix.model.tensors[up.output as usize].bytes,
            2 * 800 * 32 * 4
        );
        assert_eq!(
            prefix.model.tensors[insts[2].t[2] as usize].bytes,
            64 * 32 * 16 * 4
        );
        assert_eq!(insts[3].i[5], 2);
        assert_eq!(insts[3].j[1], 3);
        assert_eq!(insts[4].t[0], insts[4].t[1]);
        assert!(!prefix.model.to_blob().is_empty());
    }

    #[test]
    fn h16_prefix_format_and_attention() {
        let (b, rows, pre, heads) = (2u32, 72u32, 306u32, 8u32);
        let width = heads * 64;
        let block = u32::try_from(h16_prefix_floats(heads, pre)).unwrap();
        let (prefix, h) = prefix_with(&[
            ("qkv", u64::from(pre * 3 * width) * 4),
            ("idx", u64::from(b) * 4),
        ]);
        let mut p = prefix.program();
        let copy = |output| CopyColsF32Stage {
            output,
            items: 2,
            rows: pre,
            cols: 2 * width,
            in_item_stride: 0,
            in_stride: 3 * width,
            in_offset: width,
            out_item_stride: block,
            out_stride: 0,
            out_offset: 0,
        };
        let f = p
            .format_prefix_h16(h[0], &[], copy(TensorRef::Named("cache")), None)
            .unwrap();
        let attention = |rows, h16| AttentionF32Stage {
            output: TensorRef::Named("ctx"),
            key_lengths: None,
            bias: None,
            batch: b,
            q_rows: rows,
            kv_rows: rows,
            heads,
            head_width: 64,
            in_stride: 3 * width,
            k_col0: width,
            v_col0: 2 * width,
            causal: false,
            scale: 0.125,
            bias_head_stride: 0,
            relative: false,
            key_length_heads: 0,
            prefix: Some(AttentionPrefix {
                table: f.output,
                index: h[1],
                blocks: 2,
                rows: pre,
                h16,
                rows_per_item: None,
            }),
            window: 0,
            stream: None,
        };
        assert!(p
            .attention_f32(h[0], h[0], h[0], &[f.done], attention(48, true))
            .is_err());
        p.attention_f32(h[0], h[0], h[0], &[f.done], attention(rows, true))
            .unwrap();
        let prefix = p.finish(0);
        let insts = &prefix.model.progs[0].insts;
        assert_eq!(
            prefix.model.tensors[f.output as usize].bytes,
            2 * u64::from(block) * 4
        );
        assert_eq!(
            (DevOp::from_u16(insts[0].op), insts[0].i[7]),
            (Some(DevOp::CopyColsF32), 1)
        );
        assert_eq!(insts[1].i[6], 2 | 16);
    }

    #[test]
    fn prefix_rows_per_item_and_copy_group() {
        let (b, rows, pre, heads) = (3u32, 72u32, 40u32, 2u32);
        let width = heads * 64;
        let block = u32::try_from(h16_prefix_floats(heads, pre)).unwrap();
        let (prefix, h) = prefix_with(&[
            ("qkv", u64::from(b * rows * 3 * width) * 4),
            ("idx", u64::from(b) * 4),
            ("pre", u64::from(b) * 4),
            ("g", 4),
            ("table", u64::from(5 * pre * 2 * width) * 4),
            ("fmt", u64::from(4 * block + 7) * 4),
            ("short", 8),
        ]);
        let mut p = prefix.program();
        let copy = |output, out_item_stride| CopyColsF32Stage {
            output,
            items: 2,
            rows: pre,
            cols: 2 * width,
            in_item_stride: rows * 3 * width,
            in_stride: 3 * width,
            in_offset: width,
            out_item_stride,
            out_stride: 2 * width,
            out_offset: 0,
        };
        let plain = copy(TensorRef::Handle(h[4]), pre * 2 * width);
        assert!(p.copy_cols_f32(h[0], &[], plain, Some(h[6])).is_ok());
        assert!(p.copy_cols_f32(h[0], &[], plain, Some(99)).is_err());
        assert!(p
            .copy_cols_f32(h[0], &[], copy(TensorRef::Handle(h[4]), 0), Some(h[3]))
            .is_err());
        let c = p.copy_cols_f32(h[0], &[], plain, Some(h[3])).unwrap();
        let f = p
            .format_prefix_h16(h[0], &[], copy(TensorRef::Handle(h[5]), block), Some(h[3]))
            .unwrap();
        let attention = |rows_per_item, bias, causal| AttentionF32Stage {
            output: TensorRef::Named("ctx"),
            key_lengths: None,
            bias,
            batch: b,
            q_rows: rows,
            kv_rows: rows,
            heads,
            head_width: 64,
            in_stride: 3 * width,
            k_col0: width,
            v_col0: 2 * width,
            causal,
            scale: 0.125,
            bias_head_stride: 0,
            relative: false,
            key_length_heads: 0,
            prefix: Some(AttentionPrefix {
                table: h[4],
                index: h[1],
                blocks: 5,
                rows: pre,
                h16: false,
                rows_per_item,
            }),
            window: 0,
            stream: None,
        };
        assert!(p
            .attention_f32(
                h[0],
                h[0],
                h[0],
                &[c.done],
                attention(Some(h[6]), None, false)
            )
            .is_err());
        assert!(p
            .attention_f32(
                h[0],
                h[0],
                h[0],
                &[c.done],
                attention(Some(h[2]), None, true)
            )
            .is_err());
        assert!(p
            .attention_f32(
                h[0],
                h[0],
                h[0],
                &[c.done],
                attention(Some(h[2]), Some(TensorRef::Handle(h[4])), false)
            )
            .is_err());
        p.attention_f32(
            h[0],
            h[0],
            h[0],
            &[c.done, f.done],
            attention(Some(h[2]), None, false),
        )
        .unwrap();
        p.attention_f32(h[0], h[0], h[0], &[c.done], attention(None, None, false))
            .unwrap();
        let prefix = p.finish(0);
        let insts = &prefix.model.progs[0].insts;
        let n = insts.len();
        // "table" holds 5 items of pre * 2 * width, "fmt" 4 blocks (+7 floats).
        assert_eq!((insts[n - 4].t[2], insts[n - 4].i[7]), (h[3], 2 | 5 << 8));
        assert_eq!(
            (insts[n - 3].t[2], insts[n - 3].i[7]),
            (h[3], 1 | 2 | 4 << 8)
        );
        assert_eq!((insts[n - 2].t[5], insts[n - 2].i[6]), (h[2], 2 | 32));
        assert_eq!(
            (insts[n - 1].t[5], insts[n - 1].i[6]),
            (packet::dev::TENSOR_NONE, 2)
        );
    }

    #[test]
    fn reduce_rows_and_length_affine_emit_and_validate() {
        let (items, rows, stride) = (2u32, 250u32, 96u32);
        let (prefix, h) = prefix_with(&[
            ("x", u64::from(items * rows * stride) * 4),
            ("len", 8),
            ("tok", 8),
        ]);
        let mut p = prefix.program();
        let stage = |op, seg_rows, width, col0| ReduceRowsF32Stage {
            output: TensorRef::Named("seg"),
            lengths: Some(h[1]),
            items,
            rows,
            width,
            op,
            seg_rows,
            stride,
            col0,
        };
        assert!(p
            .reduce_rows_f32(h[0], &[], stage(ReduceRowsOp::Mean, 100, 80, 17))
            .is_err());
        assert!(p
            .reduce_rows_f32(h[0], &[], stage(ReduceRowsOp::Mean, 100, 0, 0))
            .is_err());
        let r = p
            .reduce_rows_f32(h[0], &[], stage(ReduceRowsOp::MeanStd, 100, 80, 16))
            .unwrap();
        let affine = |div, lo, hi| LengthAffineU32Stage {
            output: TensorRef::Named("tok.len"),
            min_with: Some(h[2]),
            items,
            mul: 1,
            add: -51,
            div,
            lo,
            hi,
        };
        assert!(p.length_affine_u32(h[1], &[], affine(0, 1, 9)).is_err());
        assert!(p.length_affine_u32(h[1], &[], affine(77, 9, 1)).is_err());
        let a = p
            .length_affine_u32(h[1], &[r.done], affine(77, 1, u32::MAX))
            .unwrap();
        let prefix = p.finish(0);
        let tensors = &prefix.model.tensors;
        let insts = &prefix.model.progs[0].insts;
        assert_eq!(
            tensors[r.output as usize].bytes,
            u64::from(items * 3 * 160) * 4
        );
        assert_eq!(DevOp::from_u16(insts[0].op), Some(DevOp::ReduceRowsF32));
        assert_eq!(insts[0].t[..3], [r.output, h[0], h[1]]);
        assert_eq!(insts[0].i[..7], [items, rows, 80, 4, 100, stride, 16]);
        assert_eq!(tensors[a.output as usize].bytes, u64::from(items) * 4);
        assert_eq!(DevOp::from_u16(insts[1].op), Some(DevOp::LengthAffineU32));
        assert_eq!(insts[1].t[..3], [a.output, h[1], h[2]]);
        assert_eq!(
            insts[1].i[..6],
            [items, 1, (-51i32) as u32, 77, 1, u32::MAX]
        );
    }

    #[test]
    fn trim_silence_emit_and_validate() {
        let (items, cap) = (2u32, 1000u32);
        let (prefix, h) = prefix_with(&[("pcm", u64::from(items * cap) * 4), ("len", 8)]);
        let mut p = prefix.program();
        let stage = |frame, top_db| TrimSilenceF32Stage {
            output: TensorRef::Named("trim"),
            len_out: TensorRef::Named("trim.len"),
            items,
            cap,
            frame,
            hop: 512,
            top_db,
        };
        assert!(p.trim_silence_f32(h[0], h[1], &[], stage(0, 20.0)).is_err());
        assert!(p
            .trim_silence_f32(h[0], h[1], &[], stage(2048, f32::NAN))
            .is_err());
        let mut alias = stage(2048, 20.0);
        alias.output = TensorRef::Handle(h[0]);
        assert!(p.trim_silence_f32(h[0], h[1], &[], alias).is_err());
        let (e, len_out) = p
            .trim_silence_f32(h[0], h[1], &[], stage(2048, 20.0))
            .unwrap();
        let prefix = p.finish(0);
        let tensors = &prefix.model.tensors;
        let inst = &prefix.model.progs[0].insts[0];
        assert_eq!(tensors[e.output as usize].bytes, u64::from(items * cap) * 4);
        assert_eq!(tensors[len_out as usize].bytes, u64::from(items) * 4);
        assert_eq!(DevOp::from_u16(inst.op), Some(DevOp::TrimSilenceF32));
        assert_eq!(inst.t[..4], [e.output, h[0], h[1], len_out]);
        assert_eq!(inst.i[..4], [items, cap, 2048, 512]);
        assert_eq!(inst.f[0], 20.0);
    }

    #[test]
    fn stream_tail_gather_store_and_validation() {
        let (b, slots, tail, rows, cols, z_cols, z_col0) =
            (3u32, 4u32, 6u32, 10u32, 32u32, 48u32, 8u32);
        let (prefix, h) = prefix_with(&[
            ("src", u64::from(b * rows * 64) * 4),
            ("slot", 12),
            ("n", 12),
            ("pos", 12),
        ]);
        let mut p = prefix.program();
        let stage = |mode, tail, rows, cols, z_col0| StreamTailF32Stage {
            mode,
            state: TensorRef::Named("state.tail"),
            slot: h[1],
            n: h[2],
            items: b,
            slots,
            tail,
            rows,
            cols,
            z_cols,
            z_col0,
        };
        let gather = |src_stride| StreamTailF32Mode::Gather {
            output: TensorRef::Named("z"),
            src: h[0],
            pos: h[3],
            src_stride,
        };
        for (tail, rows, cols, z_col0) in [
            (0, rows, cols, z_col0),
            (tail, 0, cols, z_col0),
            (tail, rows, z_cols + 1, 0),
            (tail, rows, cols, z_cols - cols + 1),
        ] {
            assert!(p
                .stream_tail_f32(&[], stage(gather(64), tail, rows, cols, z_col0))
                .is_err());
        }
        assert!(p
            .stream_tail_f32(&[], stage(gather(65), tail, rows, cols, z_col0))
            .is_err());
        let g = p
            .stream_tail_f32(&[], stage(gather(64), tail, rows, cols, z_col0))
            .unwrap();
        let st = p
            .stream_tail_f32(
                &[g.done],
                stage(
                    StreamTailF32Mode::Store { z: g.output },
                    tail,
                    rows,
                    cols,
                    z_col0,
                ),
            )
            .unwrap();
        let prefix = p.finish(0);
        let tensors = &prefix.model.tensors;
        let insts = &prefix.model.progs[0].insts;
        assert_eq!(insts.len(), 2);
        assert_eq!(
            tensors[g.output as usize].bytes,
            u64::from(b * (tail + rows) * z_cols) * 4
        );
        assert_eq!(
            tensors[st.output as usize].bytes,
            u64::from(slots * tail * cols) * 4
        );
        assert_eq!(DevOp::from_u16(insts[0].op), Some(DevOp::StreamTailF32));
        assert_eq!(
            insts[0].t[..6],
            [g.output, h[0], st.output, h[1], h[2], h[3]]
        );
        assert_eq!(insts[0].i, [b, tail, rows, cols, z_cols, z_col0, 64, 0]);
        assert_eq!(insts[0].j, [slots, 0]);
        assert_eq!(insts[1].t[0], packet::dev::TENSOR_NONE);
        assert_eq!(insts[1].t[1..5], [g.output, st.output, h[1], h[2]]);
        assert_eq!(insts[1].t[5], packet::dev::TENSOR_NONE);
        assert_eq!(insts[1].i, [b, tail, rows, cols, z_cols, z_col0, 0, 1]);
        assert!(!prefix.model.to_blob().is_empty());
    }

    #[test]
    fn attention_stream_encodes_history_and_positions() {
        let (b, hist, n, width) = (3u32, 127u32, 17u32, 4 * 64u32);
        let (prefix, h) = prefix_with(&[
            ("z", u64::from(b * (hist + n) * 3 * width) * 4),
            ("pos", u64::from(b) * 4),
        ]);
        let stage = |kv_rows, causal, stream| AttentionF32Stage {
            output: TensorRef::Named("ctx"),
            key_lengths: None,
            bias: None,
            batch: b,
            q_rows: n,
            kv_rows,
            heads: 4,
            head_width: 64,
            in_stride: 3 * width,
            k_col0: width,
            v_col0: 2 * width,
            causal,
            scale: 0.125,
            bias_head_stride: 0,
            relative: false,
            key_length_heads: 0,
            prefix: None,
            window: 128,
            stream,
        };
        let st = Some(AttentionStream {
            pos: h[1],
            history: hist,
        });
        let mut p = prefix.program();
        assert!(p
            .attention_f32(h[0], h[0], h[0], &[], stage(hist + n + 1, true, st))
            .is_err());
        assert!(p
            .attention_f32(h[0], h[0], h[0], &[], stage(hist + n, false, st))
            .is_err());
        p.attention_f32(h[0], h[0], h[0], &[], stage(hist + n, true, st))
            .unwrap();
        let prefix = p.finish(0);
        let d = &prefix.model.progs[0].insts[0];
        assert_eq!(d.t[7], h[1]);
        assert_eq!(d.i[6] & 0xF, 1 | 8, "causal + stream, no tensor-core bit");
        assert_eq!(d.i[6] >> 16, 128);
        assert_eq!(d.i[7], hist);
    }

    #[test]
    fn attention_head_width_256_general_path_only() {
        let (b, q, kv, heads) = (2u32, 32u32, 150u32, 4u32);
        let width = heads * 256;
        let (prefix, h) = prefix_with(&[
            ("q", u64::from(b * q * 2 * width) * 4),
            ("kv", u64::from(b * kv * 2 * width) * 4),
            ("len", 8),
            ("pos", 8),
        ]);
        let stage = |q_rows, kv_rows, in_stride, v_col0| AttentionF32Stage {
            output: TensorRef::Named("ctx"),
            key_lengths: Some(h[2]),
            bias: None,
            batch: b,
            q_rows,
            kv_rows,
            heads,
            head_width: 256,
            in_stride,
            k_col0: 0,
            v_col0,
            causal: false,
            scale: 0.0625,
            bias_head_stride: 0,
            relative: false,
            key_length_heads: 0,
            prefix: None,
            window: 0,
            stream: None,
        };
        let mut p = prefix.program();
        let mut bad = stage(q, kv, 2 * width, width);
        bad.stream = Some(AttentionStream {
            pos: h[3],
            history: 1,
        });
        bad.causal = true;
        bad.kv_rows = q + 1;
        assert!(p.attention_f32(h[1], h[1], h[1], &[], bad).is_err());
        let mut bad = stage(q, kv, 2 * width, width);
        bad.head_width = 192;
        assert!(p.attention_f32(h[1], h[1], h[1], &[], bad).is_err());
        let e = p
            .attention_f32(h[0], h[1], h[1], &[], stage(q, kv, 2 * width, width))
            .unwrap();
        let prefix = p.finish(0);
        let d = &prefix.model.progs[0].insts[0];
        assert_eq!(
            prefix.model.tensors[e.output as usize].bytes,
            u64::from(b * q * width) * 4
        );
        assert_eq!(d.i[..6], [b, q, kv, heads, 256, 2 * width]);
        assert_eq!(d.i[6], 0, "no tensor-core bit");
        assert_eq!(d.j, [0, width]);
        assert_eq!(d.blocks, 4);
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
                    relative: false,
                    key_length_heads: 0,
                    prefix: None,
                    window: 0,
                    stream: None,
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
        assert_eq!(
            prefix.model.tensors[prefix.output as usize].bytes,
            2 * 480 * 9 * 4
        );
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
