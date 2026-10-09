//! cuBLASLt grouped-GEMM route for the Gemma-4 MoE experts (`PLOW_MOE_PF_LT` prefill,
//! `PLOW_MOE_DEC_LT` decode).
//!
//! A packet emitted with `PLOW_EMIT_MOE_PF_LT` isolates each layer's `MoeGroupGluGemmaPf` +
//! `MoeGroupDownGemmaPf` pair in a `MOE_PREFILL_CUBLASLT` segment. The route serves that segment
//! with two grouped matmuls whose per-expert row counts and matrix pointers live on the device
//! (no host sync, graph-capturable), plus four glue kernels (`runtime/nvidia/moe_lt_sm90.cu`).
//! `MoeAlignGemmaPf` still runs in the interpreter and its tables are the route's input, so the
//! gate-scaled f32 `part[token*k+slot]` contract and the combine op are unchanged.
//!
//! `PLOW_EMIT_MOE_DEC_LT` does the same on the decode rungs that run the grouped arm
//! (`MoeExpertGluNormGemma` + `MoeExpertDownGemma` with the align tables, role
//! `MOE_DECODE_CUBLASLT`). That GLU fuses the RMS norm, so the decode route first stages
//! `xn2 = norm(x) * gamma` with a fifth glue kernel and then runs the prefill chain unchanged;
//! an ABI 3 object does setup + norm + gather in one launch (`plow_moe_lt_norm_gather`).

use super::*;
use crate::asset::devblob::DevTensor;
use crate::device::cuda::lt::{GroupedDims, GroupedPlan, Lt};

pub(super) const OBJECT_FILE: &str = "interp_sm90a_moe_lt.cubin";
const GLUE_THREADS: u32 = 256;

/// One validated `MOE_PREFILL_CUBLASLT` or `MOE_DECODE_CUBLASLT` segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoeLtSegment {
    pub(super) glu: usize,
    pub(super) down: usize,
    hidden: u32,
    inter: u32,
    experts: u32,
    act: u32,
    /// Gathered-row capacity of the align tables and `fu_g`.
    capacity: u32,
    average_rows: u32,
    /// Tensor handles: ewt, meta, row_token, row_partidx, row_gate, xn2, fu, part.
    handles: [u16; 8],
    /// Decode only: the fused norm the route stages before the gather.
    norm: Option<Norm>,
    /// Decode only: the layer tail (`MoeCombineNormGemma` + `NormResidualNorm`) the segment
    /// carries after the pair; the route runs it from the down rows instead of scattering `part`.
    tail: Option<Tail>,
    /// Prefill only: the `MoeCombineNormGemmaPf` the segment carries after the pair.
    pf_tail: Option<PfTail>,
    /// W8A8 prefill: the GEMMs run e4m3 in cuBLASLt, the glue applies the scales.
    fp8: Option<Fp8Ops>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fp8Ops {
    /// The QuantFp8 of `fu` between the GLU and the DOWN (the glue re-quantizes instead).
    quant: usize,
    /// Activation row scales (`ascale`), the per-expert channel-scale table (`est`) and the
    /// `fu` row scales (`fs`).
    handles: [u16; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PfTail {
    combine: usize,
    /// out, h1, gamma.
    handles: [u16; 3],
    tokens: u32,
    eps_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Norm {
    x: u16,
    gamma: u16,
    rows: u32,
    eps_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tail {
    combine: usize,
    nrn: usize,
    /// hn, comb, h1, g_pf2, g_po, gn (the NRN weights may be TENSOR_NONE: weightless).
    handles: [u16; 6],
    eps_comb_bits: u32,
    eps_bits: u32,
    scale_bits: u32,
}

/// `plow_moe_lt_combine_nrn` is op70's k = 8 body: its float4 partition covers H <= 12 * 256.
const TAIL_TOP_K: u32 = 8;
const TAIL_MAX_HIDDEN: u32 = 12 * GLUE_THREADS;

/// The align op pads each expert's row segment to a tile multiple (at most `PGM_BM` = 128 rows),
/// which bounds a decode rung's gathered-row extent by `rows * k + experts * ALIGN_PAD`.
const ALIGN_PAD: u64 = 128;

/// `(segment, instruction)` pairs a set of routes executes, for `cublaslt::ordered_waits_for`.
pub(super) fn instructions(segments: &[Option<MoeLtSegment>]) -> Vec<(usize, usize)> {
    segments
        .iter()
        .enumerate()
        .filter_map(|(seg, route)| {
            route.map(|route| {
                let tail = route.tail.map(|t| [(seg, t.combine), (seg, t.nrn)]);
                let pf_tail = route.pf_tail.map(|t| (seg, t.combine));
                let quant = route.fp8.map(|f| (seg, f.quant));
                [(seg, route.glu), (seg, route.down)]
                    .into_iter()
                    .chain(tail.into_iter().flatten())
                    .chain(pf_tail)
                    .chain(quant)
            })
        })
        .flatten()
        .collect()
}

pub(super) fn segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
) -> Result<Vec<Option<MoeLtSegment>>> {
    let fail = || RuntimeError::Rejected("invalid packet-declared MoE prefill segment".into());
    let count = program.gq_seg_ofs.len().checked_sub(1).ok_or_else(fail)?;
    if roles.len() != count {
        return Err(fail());
    }
    let mut routes = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        if roles[segment] != plow_asset::segment_roles::MOE_PREFILL_CUBLASLT {
            continue;
        }
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(fail)?;
        let glu = entries.first().ok_or_else(fail)?.inst as usize;
        if program.insts.get(glu).is_some_and(|g| g.op == DevOp::MoeGroupGluGemmaPfW8a8 as u16) {
            routes[segment] =
                Some(w8a8_segment(program, tensors, segment, entries, glu).ok_or_else(fail)?);
            continue;
        }
        let down = glu + 1;
        let (Some(align), Some(g), Some(d)) = (
            glu.checked_sub(1).and_then(|i| program.insts.get(i)),
            program.insts.get(glu),
            program.insts.get(down),
        ) else {
            return Err(fail());
        };
        let [inter, hidden, experts] = [g.i[0], g.i[1], g.i[2]];
        let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
        let capacity = bytes(g.t[0]) / (u64::from(inter.max(1)) * 2);
        let none = packet::dev::TENSOR_NONE16;
        let last = entries.iter().map(|e| e.inst as usize).max().ok_or_else(fail)?;
        let members = glu..=last;
        let pf_tail = match last.checked_sub(down) {
            Some(0) => None,
            Some(1) => Some(prefill_tail(program, tensors, down).ok_or_else(fail)?),
            _ => return Err(fail()),
        };
        if align.op != DevOp::MoeAlignGemmaPf as u16
            || g.op != DevOp::MoeGroupGluGemmaPf as u16
            || d.op != DevOp::MoeGroupDownGemmaPf as u16
            || entries.iter().any(|e| !members.contains(&(e.inst as usize)))
            || program
                .stream
                .iter()
                .any(|e| members.contains(&(e.inst as usize)) != (e.seg as usize == segment))
            || pf_tail.is_some_and(|t| {
                align.i[2] != TAIL_TOP_K
                    || hidden > TAIL_MAX_HIDDEN
                    || bytes(d.t[0]) < u64::from(t.tokens) * u64::from(TAIL_TOP_K) * 4
            })
            || g.t[5..].iter().chain(&d.t[6..]).any(|&t| t != none)
            || [g.t[3], g.t[4]] != [align.t[0], align.t[2]]
            || [d.t[1], d.t[2], d.t[3], d.t[4], d.t[5]]
                != [g.t[0], g.t[2], align.t[0], align.t[3], align.t[4]]
            || [d.i[0], d.i[1], d.i[2]] != [hidden, inter, experts]
            || align.i[1] != experts
            || experts == 0
            || inter == 0
            || hidden == 0
            || inter % 8 != 0
            || hidden % 8 != 0
            || g.i[5] > 1
            || capacity == 0
            || capacity > u64::from(u32::MAX)
            || bytes(g.t[2]) != u64::from(experts) * 16
            || bytes(g.t[3]) < (3 * u64::from(experts) + 1) * 4
            || [g.t[4], d.t[4], d.t[5]].iter().any(|&t| bytes(t) < capacity * 4)
        {
            return Err(fail());
        }
        routes[segment] = Some(MoeLtSegment {
            glu,
            down,
            hidden,
            inter,
            experts,
            act: g.i[5],
            capacity: capacity as u32,
            average_rows: (u64::from(program.t) * u64::from(align.i[2]) / u64::from(experts))
                as u32,
            handles: [g.t[2], g.t[3], g.t[4], d.t[4], d.t[5], g.t[1], g.t[0], d.t[0]],
            norm: None,
            tail: None,
            pf_tail,
            fp8: None,
        });
    }
    Ok(routes)
}

/// A W8A8 prefill segment: `MoeGroupGluGemmaPfW8a8`, the `QuantFp8` of its `fu`,
/// `MoeGroupDownGemmaPfW8a8` and the combine, after the align and the activation `QuantFp8`.
/// The gather reads the quantized activation rows (`xq`, `ascale`); `fu` holds the e4m3 rows
/// of the re-quantized GLU output and the quant's scale tensor their row scales.
fn w8a8_segment(
    program: &DevProg,
    tensors: &[DevTensor],
    segment: usize,
    entries: &[packet::dev::StreamEnt],
    glu: usize,
) -> Option<MoeLtSegment> {
    let insts = &program.insts;
    let (align, aq, g) = (insts.get(glu.checked_sub(2)?)?, insts.get(glu - 1)?, insts.get(glu)?);
    let (q, d) = (insts.get(glu + 1)?, insts.get(glu + 2)?);
    let pf_tail = prefill_tail(program, tensors, glu + 2)?;
    let members = glu..=glu + 3;
    let [inter, hidden, experts] = [g.i[0], g.i[1], g.i[2]];
    let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
    let capacity = bytes(g.t[0]) / (u64::from(inter.max(1)) * 2);
    let tokens = u64::from(program.t);
    let none = packet::dev::TENSOR_NONE16;
    // The activation quant runs in an earlier segment: the route's gather reads its output.
    let aq_segments = || {
        program
            .stream
            .iter()
            .filter(|e| e.inst as usize == glu - 1)
            .map(|e| e.seg as usize)
    };
    (align.op == DevOp::MoeAlignGemmaPf as u16
        && aq.op == DevOp::QuantFp8 as u16
        && [aq.t[0], aq.t[2]] == [g.t[1], g.t[5]]
        && aq.t[3..].iter().all(|&t| t == none)
        && [aq.i[0], aq.i[1]] == [program.t, hidden]
        && bytes(aq.t[1]) >= tokens * u64::from(hidden) * 2
        && aq_segments().next().is_some()
        && aq_segments().all(|s| s < segment)
        && q.op == DevOp::QuantFp8 as u16
        && d.op == DevOp::MoeGroupDownGemmaPfW8a8 as u16
        && entries.iter().all(|e| members.contains(&(e.inst as usize)))
        && entries.iter().map(|e| e.inst as usize).max() == Some(glu + 3)
        && program
            .stream
            .iter()
            .all(|e| members.contains(&(e.inst as usize)) == (e.seg as usize == segment))
        && [q.t[1], q.t[0], q.t[2]] == [g.t[0], d.t[1], d.t[7]]
        && q.i[1] == inter
        && [g.t[3], g.t[4]] == [align.t[0], align.t[2]]
        && [d.t[2], d.t[3], d.t[4], d.t[5], d.t[6]]
            == [g.t[2], align.t[0], align.t[3], align.t[4], g.t[6]]
        && [d.i[0], d.i[1], d.i[2]] == [hidden, inter, experts]
        && [align.i[0], align.i[1], align.i[2]] == [program.t, experts, TAIL_TOP_K]
        && pf_tail.tokens == program.t
        && experts > 0
        && inter > 0
        && inter % 16 == 0
        && inter <= 96 * 8
        && hidden % 16 == 0
        && hidden <= TAIL_MAX_HIDDEN
        && g.i[5] <= 1
        && capacity > 0
        && capacity <= u64::from(u32::MAX)
        && bytes(g.t[2]) == u64::from(experts) * 16
        && bytes(g.t[6]) == u64::from(experts) * 16
        && bytes(g.t[3]) >= (3 * u64::from(experts) + 1) * 4
        && [g.t[4], d.t[4], d.t[5], d.t[7]].iter().all(|&t| bytes(t) >= capacity * 4)
        && bytes(g.t[1]) >= tokens * u64::from(hidden)
        && bytes(g.t[5]) >= tokens * 4
        && bytes(d.t[0]) >= tokens * u64::from(TAIL_TOP_K) * 4)
        .then(|| MoeLtSegment {
            glu,
            down: glu + 2,
            hidden,
            inter,
            experts,
            act: g.i[5],
            capacity: capacity as u32,
            average_rows: (tokens * u64::from(TAIL_TOP_K) / u64::from(experts)) as u32,
            handles: [g.t[2], g.t[3], g.t[4], d.t[4], d.t[5], g.t[1], g.t[0], d.t[0]],
            norm: None,
            tail: None,
            pf_tail: Some(pf_tail),
            fp8: Some(Fp8Ops {
                quant: glu + 1,
                handles: [g.t[5], g.t[6], d.t[7]],
            }),
        })
}

/// The decode twin of [`segments`]: a `MOE_DECODE_CUBLASLT` segment holds the layer's
/// `MoeExpertGluNormGemma` + `MoeExpertDownGemma` pair on a rung at or above the grouped arm's
/// threshold, preceded by the align op whose tables the route reads.
pub(super) fn decode_segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
) -> Result<Vec<Option<MoeLtSegment>>> {
    let fail = || RuntimeError::Rejected("invalid packet-declared MoE decode segment".into());
    let count = program.gq_seg_ofs.len().checked_sub(1).ok_or_else(fail)?;
    if roles.len() != count {
        return Err(fail());
    }
    let rows = packet::devbuild::program_rows(program.t);
    let mut routes = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        if roles[segment] != plow_asset::segment_roles::MOE_DECODE_CUBLASLT {
            continue;
        }
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(fail)?;
        let glu = entries.first().ok_or_else(fail)?.inst as usize;
        let down = glu + 1;
        let (Some(align), Some(g), Some(d)) = (
            glu.checked_sub(1).and_then(|i| program.insts.get(i)),
            program.insts.get(glu),
            program.insts.get(down),
        ) else {
            return Err(fail());
        };
        let [k, inter, hidden, experts] = [g.i[0], g.i[1], g.i[2], g.i[3]];
        let min = g.i[6];
        let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
        let capacity = (bytes(g.t[0]) / (u64::from(inter.max(1)) * 2))
            .min(u64::from(rows) * u64::from(k) + u64::from(experts) * ALIGN_PAD);
        let none = packet::dev::TENSOR_NONE16;
        let row_bytes = u64::from(rows) * u64::from(hidden) * 2;
        let last = entries.iter().map(|e| e.inst as usize).max().ok_or_else(fail)?;
        let members = glu..=last;
        let tail = match last - down {
            0 => None,
            2 => Some(decode_tail(program, tensors, down, rows, hidden).ok_or_else(fail)?),
            _ => return Err(fail()),
        };
        if align.op != DevOp::MoeAlignGemmaPf as u16
            || g.op != DevOp::MoeExpertGluNormGemma as u16
            || d.op != DevOp::MoeExpertDownGemma as u16
            || entries.iter().any(|e| !members.contains(&(e.inst as usize)))
            || program
                .stream
                .iter()
                .any(|e| members.contains(&(e.inst as usize)) != (e.seg as usize == segment))
            || tail.is_some_and(|t| {
                let c = &program.insts[t.combine];
                c.t[1] != d.t[0] || [c.i[1], c.i[2]] != [k, g.i[5]] || k != TAIL_TOP_K
            })
            || [g.t[0], g.t[1], g.t[3], g.t[4], g.t[5], g.t[6], g.t[7], d.t[0], d.t[6], d.t[7]]
                .contains(&none)
            || [align.i[0], align.i[1], align.i[2], align.i[3]] != [rows, experts, k, min]
            || min == 0
            || rows < min
            || [align.t[0], align.t[2], align.t[3], align.t[4]] != [g.t[6], g.t[7], d.t[6], d.t[7]]
            || [d.t[1], d.t[2], d.t[3], d.t[5]] != [g.t[0], g.t[2], g.t[3], g.t[6]]
            || [d.i[0], d.i[1], d.i[2], d.i[3], d.i[5], d.i[6]]
                != [k, hidden, inter, experts, g.i[5], min]
            || g.i[5] != if rows > 1 { rows } else { 0 }
            || g.i[4] > 1
            || k == 0
            || experts == 0
            || inter == 0
            || hidden == 0
            || inter % 8 != 0
            || hidden % 8 != 0
            || capacity == 0
            || capacity > u64::from(u32::MAX)
            || bytes(g.t[3]) != u64::from(experts) * 16
            || bytes(g.t[6]) < (3 * u64::from(experts) + 1) * 4
            || [g.t[7], d.t[6], d.t[7]].iter().any(|&t| bytes(t) < capacity * 4)
            || bytes(g.t[1]) < row_bytes
            || bytes(g.t[5]) < row_bytes
            || bytes(g.t[4]) < u64::from(hidden) * 2
            || bytes(d.t[0]) < row_bytes * u64::from(k) * 2
        {
            return Err(fail());
        }
        routes[segment] = Some(MoeLtSegment {
            glu,
            down,
            hidden,
            inter,
            experts,
            act: g.i[4],
            capacity: capacity as u32,
            average_rows: ((u64::from(rows) * u64::from(k)) / u64::from(experts)).max(1) as u32,
            handles: [g.t[3], g.t[6], g.t[7], d.t[6], d.t[7], g.t[5], g.t[0], d.t[0]],
            norm: Some(Norm {
                x: g.t[1],
                gamma: g.t[4],
                rows,
                eps_bits: g.fj[0],
            }),
            tail,
            pf_tail: None,
            fp8: None,
        });
    }
    Ok(routes)
}

/// The `MoeCombineNormGemma` + in-place `NormResidualNorm` pair right after a decode down op,
/// on the residual stream `x` the GLU read (checked by the caller through the combine's `part`).
fn decode_tail(
    program: &DevProg,
    tensors: &[DevTensor],
    down: usize,
    rows: u32,
    hidden: u32,
) -> Option<Tail> {
    let (c, n) = (program.insts.get(down + 1)?, program.insts.get(down + 2)?);
    let x = program.insts[down - 1].t[1];
    let none = packet::dev::TENSOR_NONE16;
    let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
    let row_bytes = u64::from(rows) * u64::from(hidden) * 2;
    let weight = |handle: u16| handle == none || bytes(handle) >= u64::from(hidden) * 2;
    (c.op == DevOp::MoeCombineNormGemma as u16
        && n.op == DevOp::NormResidualNorm as u16
        && c.i[0] == hidden
        && hidden <= TAIL_MAX_HIDDEN
        && [n.i[0], n.i[1]] == [rows, hidden]
        && [n.t[1], n.t[2], n.t[3]] == [x, x, c.t[0]]
        && ![c.t[0], c.t[2], c.t[3], n.t[0]].contains(&none)
        && [c.t[0], c.t[2], n.t[0], x].iter().all(|&t| bytes(t) >= row_bytes)
        && bytes(c.t[3]) >= u64::from(hidden) * 2
        && weight(n.t[4])
        && weight(n.t[5]))
    .then(|| Tail {
        combine: down + 1,
        nrn: down + 2,
        handles: [n.t[0], c.t[0], c.t[2], c.t[3], n.t[4], n.t[5]],
        eps_comb_bits: c.fj[0],
        eps_bits: n.fj[0],
        scale_bits: n.fj[1],
    })
}

/// The `MoeCombineNormGemmaPf` right after a prefill down op, on that op's `part`
/// (`plow_moe_lt_combine_pf` runs it from the down rows).
fn prefill_tail(program: &DevProg, tensors: &[DevTensor], down: usize) -> Option<PfTail> {
    let (d, c) = (&program.insts[down], program.insts.get(down + 1)?);
    let none = packet::dev::TENSOR_NONE16;
    let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
    let [hidden, k, tokens] = [c.i[0], c.i[1], c.i[2]];
    let row_bytes = u64::from(tokens) * u64::from(hidden) * 2;
    (c.op == DevOp::MoeCombineNormGemmaPf as u16
        && c.t[1] == d.t[0]
        && hidden == d.i[0]
        && k == TAIL_TOP_K
        && tokens > 0
        && ![c.t[0], c.t[2], c.t[3]].contains(&none)
        && bytes(c.t[0]) >= row_bytes
        && bytes(c.t[2]) >= row_bytes
        && bytes(c.t[3]) >= u64::from(hidden) * 2)
        .then(|| PfTail {
            combine: down + 1,
            handles: [c.t[0], c.t[2], c.t[3]],
            tokens,
            eps_bits: c.fj[0],
        })
}

/// Glue kernels, scratch and device-side group tables shared by every route of an engine.
pub(super) struct MoeLt {
    be: Arc<CudaBackend>,
    lt: Arc<Lt>,
    _module: Arc<DecodeModule>,
    setup: KernelFn,
    gather: KernelFn,
    glu: KernelFn,
    scatter: KernelFn,
    /// ABI 2 objects only; the decode route needs it.
    norm: Option<KernelFn>,
    /// ABI 3: the decode route's setup + norm + gather in one launch.
    norm_gather: Option<KernelFn>,
    /// ABI 4: a decode segment's layer tail in place of the scatter.
    combine_nrn: Option<KernelFn>,
    /// ABI 5: a prefill segment's combine in place of the scatter (gather + inverse row map).
    gather_inv: Option<KernelFn>,
    combine_pf: Option<KernelFn>,
    /// ABI 6, the W8A8 prefill chain: setup, gather, scaled GLU + re-quantization, combine.
    fp8_glue: Option<[KernelFn; 4]>,
    /// e4m3 grouped matmuls (a W8A8 packet); fixed per engine.
    fp8: bool,
    blocks: u32,
    hidden: u32,
    inter: u32,
    experts: u32,
    /// `xs` (the gathered input; reused as the down output once gate|up consumed it) and `gu`.
    xs: DeviceMem,
    gu: DeviceMem,
    /// i32 `[rows | 2I | H | I] x experts`, then u64 `[xs | gu | fu | dn] x experts`.
    tables: DeviceMem,
    /// Per expert-weight table: device `[gate_up | down] x experts` matrix pointers.
    weights: std::sync::Mutex<std::collections::HashMap<u64, Arc<DeviceMem>>>,
    plans: std::sync::Mutex<std::collections::HashMap<u32, [Arc<GroupedPlan>; 2]>>,
}

pub(super) struct MoeLtRoute {
    owner: Arc<MoeLt>,
    plans: [Arc<GroupedPlan>; 2],
    weights: Arc<DeviceMem>,
    /// Decode: `(x, gamma, rows, eps)` of the staged norm.
    norm: Option<(u64, u64, u32, f32)>,
    /// Decode: the layer tail's `[hn, comb, h1, g_pf2, g_po, gn]` and `(eps_comb, eps, scale)`.
    tail: Option<([u64; 6], [f32; 3])>,
    /// Prefill: the combine's `[out, h1, gamma]`, tokens, gathered-row capacity and eps.
    pf_tail: Option<([u64; 3], u32, u32, f32)>,
    /// W8A8: `[ascale, est, fs]`.
    fp8: Option<[u64; 3]>,
    meta: u64,
    row_token: u64,
    row_partidx: u64,
    row_gate: u64,
    xn2: u64,
    fu: u64,
    part: u64,
    act: u32,
}

impl MoeLt {
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        lt: &Arc<Lt>,
        directories: &[&Path],
        profile: &str,
        shape: &MoeLtSegment,
        min_capacity: u32,
    ) -> Result<Arc<Self>> {
        let image = directories
            .iter()
            .find_map(|dir| std::fs::read(dir.join(OBJECT_FILE)).ok())
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "the MoE cuBLASLt route needs {OBJECT_FILE} next to the packet objects"
                ))
            })?;
        let abi = plow_asset::cubin::global_u32(&image, "plow_moe_lt_abi");
        if profile != "sm90a"
            || plow_asset::cubin::inspect(&image).is_none_or(|i| i.sm != 90)
            || !matches!(abi, Some(1..=6))
            || (shape.fp8.is_some() && abi < Some(6))
        {
            return Err(RuntimeError::Rejected(format!(
                "{OBJECT_FILE}: not an sm90a MoE cuBLASLt glue object of ABI 1 to 6 \
                 (a W8A8 segment needs 6)"
            )));
        }
        let module = DecodeModule::load(be, &image)?;
        let function = |name: &str| be.get_function(&module, name);
        let (hidden, inter, experts) = (shape.hidden, shape.inter, shape.experts);
        let capacity = u64::from(shape.capacity.max(min_capacity));
        let xs = be.alloc(0, capacity * u64::from(hidden) * 2)?;
        let gu = be.alloc(0, capacity * u64::from(inter) * 4)?;
        let tables = be.alloc(0, u64::from(experts) * (4 * 4 + 4 * 8))?;
        let constants: Vec<i32> = [0, 2 * inter, hidden, inter]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value as i32, experts as usize))
            .collect();
        be.upload(&tables, 0, bytemuck::cast_slice(&constants))?;
        let gather = function("plow_moe_lt_gather")?;
        let blocks = be.occupancy_blocks_per_sm(gather, GLUE_THREADS, 0)?.max(1) * be.sm_count();
        tracing::info!(
            scratch_mib = (xs.len + gu.len) >> 20,
            capacity,
            blocks,
            decode = shape.norm.is_some(),
            "MoE cuBLASLt grouped route loaded"
        );
        Ok(Arc::new(Self {
            be: Arc::clone(be),
            lt: Arc::clone(lt),
            setup: function("plow_moe_lt_setup")?,
            gather,
            glu: function("plow_moe_lt_glu")?,
            scatter: function("plow_moe_lt_scatter")?,
            norm: (abi >= Some(2))
                .then(|| function("plow_moe_lt_norm"))
                .transpose()?,
            norm_gather: (abi >= Some(3))
                .then(|| function("plow_moe_lt_norm_gather"))
                .transpose()?,
            combine_nrn: (abi >= Some(4))
                .then(|| function("plow_moe_lt_combine_nrn"))
                .transpose()?,
            gather_inv: (abi >= Some(5))
                .then(|| function("plow_moe_lt_gather_inv"))
                .transpose()?,
            combine_pf: (abi >= Some(5))
                .then(|| function("plow_moe_lt_combine_pf"))
                .transpose()?,
            fp8_glue: (abi >= Some(6))
                .then(|| -> Result<[KernelFn; 4]> {
                    Ok([
                        function("plow_moe_lt_setup8")?,
                        function("plow_moe_lt_gather8_inv")?,
                        function("plow_moe_lt_glu8")?,
                        function("plow_moe_lt_combine_pf8")?,
                    ])
                })
                .transpose()?,
            fp8: shape.fp8.is_some(),
            _module: module,
            blocks,
            hidden,
            inter,
            experts,
            xs,
            gu,
            tables,
            weights: Default::default(),
            plans: Default::default(),
        }))
    }

    fn table(&self, index: u64) -> u64 {
        self.tables.base + index * u64::from(self.experts) * 4
    }

    fn pointers(&self, index: u64) -> u64 {
        self.table(4) + index * u64::from(self.experts) * 8
    }

    fn plans(
        self: &Arc<Self>,
        average_rows: u32,
        weights: u64,
        fu: u64,
    ) -> Result<[Arc<GroupedPlan>; 2]> {
        let mut plans = self.plans.lock().expect("MoE Lt plans");
        if let Some(found) = plans.get(&average_rows) {
            return Ok(found.clone());
        }
        let plan = |n: u32, n_table: u64, k: u32, k_table: u64| {
            self.lt.grouped_plan(&GroupedDims {
                groups: self.experts,
                n,
                k,
                rows: self.table(0),
                n_array: self.table(n_table),
                k_array: self.table(k_table),
                average_rows,
                fp8: self.fp8,
            })
        };
        let mut pair = [
            plan(2 * self.inter, 1, self.hidden, 2)?,
            plan(self.hidden, 2, self.inter, 3)?,
        ];
        self.tune(&mut pair, average_rows, weights, fu)?;
        let pair = pair.map(Arc::new);
        plans.insert(average_rows, pair.clone());
        Ok(pair)
    }

    /// Time every heuristic candidate of the gate|up and down matmuls on this route's expert
    /// weights and keep the fastest. The rows follow a skewed routing (log-normal expert loads,
    /// sigma 0.6, `average_rows` mean): the heuristic's first pick is made for the uniform
    /// average and measured up to 20% slower than the best candidate on a served routing.
    /// The operands are laid out as the route's setup kernel lays them out (e4m3 `xs` and `fu`
    /// rows on W8A8; the down matmul reads the segment's `fu`). The scratch, `fu` and the group
    /// tables are free here; the route rewrites them every run.
    fn tune(
        &self,
        pair: &mut [GroupedPlan; 2],
        average_rows: u32,
        weights: u64,
        fu: u64,
    ) -> Result<()> {
        let be = &self.be;
        let (h, i, e) = (u64::from(self.hidden), u64::from(self.inter), u64::from(self.experts));
        let counts = skewed_rows(self.experts, u64::from(average_rows.max(1)), self.xs.len / (h * 2));
        be.upload(&self.tables, 0, bytemuck::cast_slice(&counts))?;
        let (xs, gu) = (self.xs.base, self.gu.base);
        let mut offsets = Vec::with_capacity(counts.len());
        let mut row = 0u64;
        for &count in &counts {
            offsets.push(row);
            row += count as u64;
        }
        let element = if self.fp8 { 1 } else { 2 };
        let pointers: Vec<u64> = [(xs, h * element), (gu, i * 4), (fu, i * element), (xs, h * 2)]
            .into_iter()
            .flat_map(|(base, width)| offsets.iter().map(move |&o| base + o * width))
            .collect();
        be.upload(&self.tables, self.pointers(0) - self.tables.base, bytemuck::cast_slice(&pointers))?;
        let stream = be.stream_create()?;
        be.memset_d8_async(xs, 0x3c, self.xs.len as usize, &stream)?;
        be.memset_d8_async(gu, 0x3c, self.gu.len as usize, &stream)?;
        be.memset_d8_async(fu, 0x3c, (row * i * element) as usize, &stream)?;
        let (start, end) = (be.event_create(true)?, be.event_create(true)?);
        let operands = [
            (self.pointers(0), weights, self.pointers(1)),
            (self.pointers(2), weights + e * 8, self.pointers(3)),
        ];
        for (plan, (a, w, c)) in pair.iter_mut().zip(operands) {
            let mut times = Vec::with_capacity(plan.candidates());
            let mut best: Option<(f32, usize)> = None;
            for index in 0..plan.candidates() {
                let mut ms = f32::INFINITY;
                for _ in 0..3 {
                    be.event_record(&start, &stream)?;
                    if let Err(error) = plan.run_candidate(index, a, w, c, &stream) {
                        be.stream_synchronize(&stream)?;
                        if error.is_fatal() {
                            return Err(error);
                        }
                        tracing::warn!(%error, index, "MoE grouped candidate rejected");
                        ms = f32::INFINITY;
                        break;
                    }
                    be.event_record(&end, &stream)?;
                    be.event_synchronize(&end)?;
                    ms = ms.min(be.event_elapsed_ms(&start, &end)?);
                }
                times.push(ms);
                if ms.is_finite() && best.is_none_or(|(b, _)| ms < b) {
                    best = Some((ms, index));
                }
            }
            let (ms, index) = best.ok_or_else(|| {
                RuntimeError::Device("no runnable MoE cuBLASLt grouped candidate".into())
            })?;
            plan.select(index);
            tracing::info!(
                average_rows,
                index,
                ms = format!("{ms:.3}").as_str(),
                candidates = ?times,
                "MoE cuBLASLt grouped algorithm measured"
            );
        }
        be.stream_synchronize(&stream)
    }

    /// Split the engine's `[gate_up, down] x experts` table into the two pointer arrays a
    /// grouped matmul takes.
    fn weights(&self, table: &DeviceMem) -> Result<Arc<DeviceMem>> {
        let mut cache = self.weights.lock().expect("MoE Lt weights");
        if let Some(found) = cache.get(&table.base) {
            return Ok(Arc::clone(found));
        }
        let experts = self.experts as usize;
        let mut interleaved = vec![0u64; experts * 2];
        self.be
            .download(table, 0, bytemuck::cast_slice_mut(&mut interleaved))?;
        if interleaved.iter().any(|&p| p == 0 || p % 16 != 0) {
            return Err(RuntimeError::Rejected(
                "MoE cuBLASLt route needs bound, 16-byte aligned BF16 expert tensors".into(),
            ));
        }
        let split: Vec<u64> = (0..2)
            .flat_map(|which| interleaved.iter().skip(which).step_by(2).copied())
            .collect();
        let device = Arc::new(self.be.alloc(0, (experts * 16) as u64)?);
        self.be.upload(&device, 0, bytemuck::cast_slice(&split))?;
        cache.insert(table.base, Arc::clone(&device));
        Ok(device)
    }

    /// Same expert geometry, and the scratch holds the segment's gathered rows.
    pub(super) fn fits(&self, segment: &MoeLtSegment) -> bool {
        [segment.hidden, segment.inter, segment.experts] == [self.hidden, self.inter, self.experts]
            && segment.fp8.is_some() == self.fp8
            && u64::from(segment.capacity) * u64::from(self.hidden) * 2 <= self.xs.len
    }

    pub(super) fn route(
        self: &Arc<Self>,
        segment: &MoeLtSegment,
        insts: &mut [DevInst64],
        devp: &[DeviceMem],
    ) -> Result<MoeLtRoute> {
        if !self.fits(segment) {
            return Err(RuntimeError::Rejected(
                "MoE cuBLASLt segments disagree on the expert geometry".into(),
            ));
        }
        if segment.norm.is_some() && self.norm.is_none() {
            return Err(RuntimeError::Rejected(format!(
                "PLOW_MOE_DEC_LT needs an ABI 2 {OBJECT_FILE} (plow_moe_lt_norm)"
            )));
        }
        if segment.tail.is_some() && self.combine_nrn.is_none() {
            return Err(RuntimeError::Rejected(format!(
                "a decode segment with the layer tail needs an ABI 4 {OBJECT_FILE}"
            )));
        }
        if segment.pf_tail.is_some() && self.combine_pf.is_none() {
            return Err(RuntimeError::Rejected(format!(
                "a prefill segment with the combine needs an ABI 5 {OBJECT_FILE}"
            )));
        }
        let base = |handle: u16| devp[handle as usize].base;
        let weight = |handle: u16| {
            if handle == packet::dev::TENSOR_NONE16 { 0 } else { base(handle) }
        };
        let [ewt, meta, row_token, row_partidx, row_gate, xn2, fu, part] = segment.handles;
        let weights = self.weights(&devp[ewt as usize])?;
        let route = MoeLtRoute {
            owner: Arc::clone(self),
            plans: self.plans(segment.average_rows, weights.base, base(fu))?,
            weights,
            norm: segment
                .norm
                .map(|n| (base(n.x), base(n.gamma), n.rows, f32::from_bits(n.eps_bits))),
            tail: segment.tail.map(|t| {
                let [hn, comb, h1, g_pf2, g_po, gn] = t.handles;
                (
                    [base(hn), base(comb), base(h1), base(g_pf2), weight(g_po), weight(gn)],
                    [t.eps_comb_bits, t.eps_bits, t.scale_bits].map(f32::from_bits),
                )
            }),
            pf_tail: segment.pf_tail.map(|t| {
                let [out, h1, gamma] = t.handles;
                (
                    [base(out), base(h1), base(gamma)],
                    t.tokens,
                    segment.capacity,
                    f32::from_bits(t.eps_bits),
                )
            }),
            fp8: segment.fp8.map(|f| f.handles.map(base)),
            meta: base(meta),
            row_token: base(row_token),
            row_partidx: base(row_partidx),
            row_gate: base(row_gate),
            xn2: base(xn2),
            fu: base(fu),
            part: base(part),
            act: segment.act,
        };
        insts[segment.glu].op = DevOp::Nop as u16;
        insts[segment.down].op = DevOp::Nop as u16;
        if let Some(tail) = segment.tail {
            insts[tail.combine].op = DevOp::Nop as u16;
            insts[tail.nrn].op = DevOp::Nop as u16;
        }
        if let Some(tail) = segment.pf_tail {
            insts[tail.combine].op = DevOp::Nop as u16;
        }
        if let Some(fp8) = segment.fp8 {
            insts[fp8.quant].op = DevOp::Nop as u16;
        }
        Ok(route)
    }
}

impl MoeLtRoute {
    /// The glue kernels; none touches cuBLASLt's workspace.
    pub(super) fn glue(&self) -> impl Iterator<Item = KernelFn> + '_ {
        let o = &*self.owner;
        [o.setup, o.gather, o.glu, o.scatter]
            .into_iter()
            .chain(o.norm)
            .chain(o.norm_gather)
            .chain(o.combine_nrn)
            .chain(o.gather_inv)
            .chain(o.combine_pf)
            .chain(o.fp8_glue.into_iter().flatten())
    }

    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        if let (Some(fp8), Some(glue)) = (self.fp8, self.owner.fp8_glue) {
            return self.run_fp8(fp8, glue, stream);
        }
        let o = &*self.owner;
        let arg = |value: &mut u64| (value as *mut u64).cast::<std::ffi::c_void>();
        let arg32 = |value: &mut u32| (value as *mut u32).cast::<std::ffi::c_void>();
        let (mut meta, mut xn2, mut fu, mut part) = (self.meta, self.xn2, self.fu, self.part);
        let (mut row_token, mut row_partidx, mut row_gate) =
            (self.row_token, self.row_partidx, self.row_gate);
        let (mut xs, mut gu) = (o.xs.base, o.gu.base);
        let (mut rows, mut pointers) = (o.table(0), o.pointers(0));
        let (mut experts, mut hidden, mut inter) = (o.experts, o.hidden, o.inter);
        let (mut act, mut blocks) = (self.act, o.blocks);
        let mut dn = xs;

        if let (Some((x, gamma, _, eps)), Some(fused)) = (self.norm, o.norm_gather) {
            let (mut x, mut gamma, mut eps) = (x, gamma, eps);
            let mut params = [
                arg(&mut xs),
                arg(&mut x),
                arg(&mut gamma),
                arg(&mut row_token),
                arg(&mut rows),
                arg(&mut pointers),
                arg(&mut meta),
                arg(&mut gu),
                arg(&mut fu),
                arg(&mut dn),
                arg32(&mut experts),
                arg32(&mut hidden),
                arg32(&mut inter),
                (&mut eps as *mut f32).cast::<std::ffi::c_void>(),
                arg32(&mut blocks),
            ];
            o.be.launch_kernel(fused, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        } else {
            if let Some((x, gamma, norm_rows, eps)) = self.norm {
                let (mut x, mut gamma, mut eps) = (x, gamma, eps);
                let mut params = [
                    arg(&mut xn2),
                    arg(&mut x),
                    arg(&mut gamma),
                    arg32(&mut hidden),
                    (&mut eps as *mut f32).cast::<std::ffi::c_void>(),
                ];
                let norm = o.norm.expect("checked by route()");
                o.be.launch_kernel(norm, norm_rows, GLUE_THREADS, 0, &mut params, Some(stream))?;
            }
            let mut params = [
                arg(&mut rows),
                arg(&mut pointers),
                arg(&mut meta),
                arg(&mut xs),
                arg(&mut gu),
                arg(&mut fu),
                arg(&mut dn),
                arg32(&mut experts),
                arg32(&mut hidden),
                arg32(&mut inter),
            ];
            o.be.launch_kernel(o.setup, 1, GLUE_THREADS, 0, &mut params, Some(stream))?;
            if let (Some(_), Some(gather_inv)) = (self.pf_tail, o.gather_inv) {
                let mut params = [
                    arg(&mut xs),
                    arg(&mut part),
                    arg(&mut xn2),
                    arg(&mut row_token),
                    arg(&mut row_partidx),
                    arg(&mut meta),
                    arg32(&mut experts),
                    arg32(&mut hidden),
                    arg32(&mut blocks),
                ];
                o.be.launch_kernel(gather_inv, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
            } else {
                let mut params = [
                    arg(&mut xs),
                    arg(&mut xn2),
                    arg(&mut row_token),
                    arg(&mut meta),
                    arg32(&mut experts),
                    arg32(&mut hidden),
                    arg32(&mut blocks),
                ];
                o.be.launch_kernel(o.gather, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
            }
        }
        self.plans[0].run(o.pointers(0), self.weights.base, o.pointers(1), stream)?;
        let mut params = [
            arg(&mut fu),
            arg(&mut gu),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut inter),
            arg32(&mut act),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(o.glu, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        self.plans[1].run(
            o.pointers(2),
            self.weights.base + u64::from(o.experts) * 8,
            o.pointers(3),
            stream,
        )?;
        if let (Some(([hn, comb, h1, g_pf2, g_po, gn], [eps_comb, eps, scale])), Some(tail)) =
            (self.tail, o.combine_nrn)
        {
            let (mut x, _, rows, _) = self.norm.expect("a decode tail rides a decode route");
            let (mut hn, mut comb, mut h1, mut g_pf2, mut g_po, mut gn) = (hn, comb, h1, g_pf2, g_po, gn);
            let (mut eps_comb, mut eps, mut scale) = (eps_comb, eps, scale);
            let argf = |value: &mut f32| (value as *mut f32).cast::<std::ffi::c_void>();
            let mut params = [
                arg(&mut hn),
                arg(&mut x),
                arg(&mut comb),
                arg(&mut dn),
                arg(&mut row_partidx),
                arg(&mut row_gate),
                arg(&mut meta),
                arg(&mut h1),
                arg(&mut g_pf2),
                arg(&mut g_po),
                arg(&mut gn),
                arg32(&mut experts),
                arg32(&mut hidden),
                argf(&mut eps_comb),
                argf(&mut eps),
                argf(&mut scale),
            ];
            return o.be.launch_kernel(tail, rows, GLUE_THREADS, 0, &mut params, Some(stream));
        }
        if let (Some(([out, h1, gamma], tokens, capacity, eps)), Some(combine)) =
            (self.pf_tail, o.combine_pf)
        {
            let (mut out, mut h1, mut gamma) = (out, h1, gamma);
            let (mut tokens, mut capacity, mut eps) = (tokens, capacity, eps);
            let mut params = [
                arg(&mut out),
                arg(&mut dn),
                arg(&mut part),
                arg(&mut row_gate),
                arg(&mut h1),
                arg(&mut gamma),
                arg32(&mut hidden),
                arg32(&mut tokens),
                arg32(&mut capacity),
                (&mut eps as *mut f32).cast::<std::ffi::c_void>(),
            ];
            return o.be.launch_kernel(combine, tokens, GLUE_THREADS, 0, &mut params, Some(stream));
        }
        let mut params = [
            arg(&mut part),
            arg(&mut dn),
            arg(&mut row_partidx),
            arg(&mut row_gate),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(o.scatter, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))
    }
}

/// Per-expert row counts of a skewed routing: log-normal loads (sigma 0.6, a fixed seed)
/// scaled to `average` rows per expert, the total capped at `capacity` rows.
fn skewed_rows(experts: u32, average: u64, capacity: u64) -> Vec<i32> {
    let mut x = 0x2545_f491u32;
    let mut uniform = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        f64::from(x) / f64::from(u32::MAX)
    };
    let loads: Vec<f64> = (0..experts)
        .map(|_| (0.6 * ((0..12).map(|_| uniform()).sum::<f64>() - 6.0)).exp())
        .collect();
    let sum: f64 = loads.iter().sum();
    let total = (average * u64::from(experts)).min(capacity) as f64;
    loads.iter().map(|l| (l / sum * total).floor() as i32).collect()
}

impl MoeLtRoute {
    /// The W8A8 chain: setup + e4m3 gather (+ inverse row map), e4m3 gate|up matmul, scaled
    /// GLU + row re-quantization into `fu`, e4m3 down matmul, scaled combine.
    /// The matmuls round unscaled fast-accumulated sums to BF16 before the glue scales them
    /// (native W8A8 scales the FP32 accumulators). Measured on the 26B last layer vs FP32
    /// (dequantized experts, same xn2 and routing), 128-row to 4096-row rungs: library rel-L2
    /// 0.020-0.031, native 0.020-0.031, within 1.3% of each other at every rung; the
    /// activation quantization both share dominates.
    fn run_fp8(
        &self,
        [ascale, est, fs]: [u64; 3],
        [setup, gather, glu, combine]: [KernelFn; 4],
        stream: &CudaStream,
    ) -> Result<()> {
        let o = &*self.owner;
        let arg = |value: &mut u64| (value as *mut u64).cast::<std::ffi::c_void>();
        let arg32 = |value: &mut u32| (value as *mut u32).cast::<std::ffi::c_void>();
        let ([mut out, mut h1, mut gamma], tokens, capacity, eps) =
            self.pf_tail.expect("a W8A8 segment carries the combine");
        let (mut tokens, mut capacity, mut eps) = (tokens, capacity, eps);
        let (mut ascale, mut est, mut fs) = (ascale, est, fs);
        let (mut meta, mut xq, mut fu, mut part) = (self.meta, self.xn2, self.fu, self.part);
        let (mut row_token, mut row_partidx, mut row_gate) =
            (self.row_token, self.row_partidx, self.row_gate);
        let (mut xs, mut gu, mut dn) = (o.xs.base, o.gu.base, o.xs.base);
        let (mut rows, mut pointers) = (o.table(0), o.pointers(0));
        let (mut experts, mut hidden, mut inter) = (o.experts, o.hidden, o.inter);
        let (mut act, mut blocks) = (self.act, o.blocks);
        let mut params = [
            arg(&mut rows),
            arg(&mut pointers),
            arg(&mut meta),
            arg(&mut xs),
            arg(&mut gu),
            arg(&mut fu),
            arg(&mut dn),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut inter),
        ];
        o.be.launch_kernel(setup, 1, GLUE_THREADS, 0, &mut params, Some(stream))?;
        let mut params = [
            arg(&mut xs),
            arg(&mut part),
            arg(&mut xq),
            arg(&mut row_token),
            arg(&mut row_partidx),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(gather, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        self.plans[0].run(o.pointers(0), self.weights.base, o.pointers(1), stream)?;
        let mut params = [
            arg(&mut fu),
            arg(&mut fs),
            arg(&mut gu),
            arg(&mut ascale),
            arg(&mut est),
            arg(&mut row_token),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut inter),
            arg32(&mut act),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(glu, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        self.plans[1].run(
            o.pointers(2),
            self.weights.base + u64::from(o.experts) * 8,
            o.pointers(3),
            stream,
        )?;
        let mut params = [
            arg(&mut out),
            arg(&mut dn),
            arg(&mut part),
            arg(&mut row_gate),
            arg(&mut fs),
            arg(&mut est),
            arg(&mut meta),
            arg(&mut h1),
            arg(&mut gamma),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut tokens),
            arg32(&mut capacity),
            (&mut eps as *mut f32).cast::<std::ffi::c_void>(),
        ];
        o.be.launch_kernel(combine, tokens, GLUE_THREADS, 0, &mut params, Some(stream))
    }
}

/// The widest gathered-row capacity among `segments` (0 when none routes).
pub(super) fn max_capacity(segments: &[Option<MoeLtSegment>]) -> u32 {
    segments.iter().flatten().map(|s| s.capacity).max().unwrap_or(0)
}

/// The routed decode object (`<decode stem>_routed.cubin`, `PLOW_NV_DECODE_ROUTED`): the decode
/// megakernel without the expert GEMV arms that a routed rung never dispatches.
pub(super) struct RoutedDecode {
    pub(super) function: KernelFn,
    pub(super) smem: u32,
    _module: Arc<DecodeModule>,
}

impl RoutedDecode {
    const DROPPED: [DevOp; 2] = [DevOp::MoeExpertGluNormGemma, DevOp::MoeExpertDownGemma];

    /// `None` when the packet ships no routed object (`_routed_fp8kv` under FP8 KV). One whose
    /// ABI, KV dtype or pairing differs from the main decode object is refused.
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        assets: &Path,
        stem: &str,
        main: &Module,
        grid: u32,
    ) -> Result<Option<Arc<Self>>> {
        let kv_abi = be.module_global_u32(main, "plow_fp8_kv_abi")?;
        let file = format!("{stem}_routed{}.cubin", if kv_abi == Some(1) { "_fp8kv" } else { "" });
        let Ok(image) = std::fs::read(assets.join(&file)) else {
            return Ok(None);
        };
        if plow_asset::cubin::global_u32(&image, "plow_fp8_kv_abi") != kv_abi {
            return Err(RuntimeError::Rejected(format!(
                "{file}: FP8 KV capability differs from the decode object"
            )));
        }
        let module = DecodeModule::load(be, &image)?;
        let function =
            be.get_function(&module, &format!("_Z{}{stem}_routed11PlowProgram", stem.len() + 7))?;
        GpuEngine::check_packet_pairing_suffix(be, &module, assets, "_routed")?;
        for symbol in ["plow_block", "plow_dyn_kvrow", "plow_segment_gq_abi", "plow_gemv_mm_cap"] {
            if be.module_global_u32(&module, &format!("{symbol}_routed"))?
                != be.module_global_u32(main, symbol)?
            {
                return Err(RuntimeError::Rejected(format!(
                    "{file}: {symbol} differs from the decode object"
                )));
            }
        }
        let smem = match crate::config::RuntimeConfig::get().nv.smem {
            Some(smem) => smem,
            None => {
                let full = be
                    .module_global_u32(&module, "plow_arena_bytes_routed")?
                    .unwrap_or(12352);
                be.module_global_u32(&module, "plow_arena_bytes_narrow_routed")?
                    .map_or(full, |narrow| narrow.min(full))
            }
        };
        if smem > 48 * 1024 {
            be.set_max_dynamic_smem(function, smem)?;
        }
        let resident = be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count();
        if resident < grid {
            return Err(RuntimeError::Rejected(format!(
                "{file}: {resident} resident blocks cannot hold the decode grid {grid}"
            )));
        }
        tracing::info!(smem, "routed decode object loaded");
        Ok(Some(Arc::new(Self {
            function,
            smem,
            _module: module,
        })))
    }

    /// The routed rung's instructions reach none of the dropped arms.
    pub(super) fn serves(&self, insts: &[DevInst64]) -> bool {
        !insts
            .iter()
            .any(|d| Self::DROPPED.iter().any(|&op| d.op == op as u16))
    }
}

/// Route every segment of one decode program, loading the shared glue once with a scratch of at
/// least `min_capacity` gathered rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_routes(
    be: &Arc<CudaBackend>,
    owner: &mut Option<Arc<MoeLt>>,
    lt: &Arc<Lt>,
    directories: &[&Path],
    profile: &str,
    segments: &[Option<MoeLtSegment>],
    min_capacity: u32,
    insts: &mut [DevInst64],
    devp: &[DeviceMem],
) -> Result<Vec<Option<super::cublaslt::LibraryRoute>>> {
    let mut routes = Vec::with_capacity(segments.len());
    for segment in segments {
        let route = match segment {
            Some(segment) => {
                if owner.is_none() {
                    *owner = Some(MoeLt::load(
                        be,
                        lt,
                        directories,
                        profile,
                        segment,
                        min_capacity,
                    )?);
                }
                let moe = owner.as_ref().expect("loaded above");
                Some(super::cublaslt::LibraryRoute::Moe(moe.route(segment, insts, devp)?))
            }
            None => None,
        };
        routes.push(route);
    }
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    const ROLE: u8 = plow_asset::segment_roles::MOE_PREFILL_CUBLASLT;
    const CAPACITY: u64 = 1024 * 8 + 128 * 128;

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let mut insts = vec![
            DevInst64 {
                op: DevOp::Nop as u16,
                blocks: 1,
                ..Default::default()
            };
            4
        ];
        for inst in &mut insts {
            inst.t.fill(packet::dev::TENSOR_NONE16);
        }
        // Handles: 0 fug, 1 xn2, 2 ewt, 3 meta, 4 rowtok, 5 part, 6 rowpart, 7 rowgate, 8 table.
        insts[0].op = DevOp::MoeAlignGemmaPf as u16;
        insts[0].t[..5].copy_from_slice(&[3, 8, 4, 6, 7]);
        insts[0].i[..3].copy_from_slice(&[1024, 128, 8]);
        insts[1].op = DevOp::MoeGroupGluGemmaPf as u16;
        insts[1].t[..5].copy_from_slice(&[0, 1, 2, 3, 4]);
        insts[1].i[..3].copy_from_slice(&[704, 2816, 128]);
        insts[2].op = DevOp::MoeGroupDownGemmaPf as u16;
        insts[2].t[..6].copy_from_slice(&[5, 0, 2, 3, 6, 7]);
        insts[2].i[..3].copy_from_slice(&[2816, 704, 128]);
        let stream: Vec<_> = [(0, 0), (1, 1), (2, 1), (3, 2)]
            .into_iter()
            .map(|(inst, seg)| StreamEnt {
                inst,
                seg,
                ..Default::default()
            })
            .collect();
        let tensors = [
            CAPACITY * 704 * 2,
            1024 * 2816 * 2,
            128 * 16,
            (3 * 128 + 2) * 4,
            CAPACITY * 4,
            1024 * 8 * 2816 * 4,
            CAPACITY * 4,
            CAPACITY * 4,
            1024 * 8 * 8,
        ]
        .into_iter()
        .map(|bytes| DevTensor {
            name: "act".into(),
            bytes,
            init: None,
        })
        .collect();
        (
            DevProg {
                t: 1024,
                role: packet::devbuild::ProgramRole::PrefillBucket { rows: 1024 },
                n_counter: 0,
                insts,
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![4],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 1, 3, 4],
                l2_domains: 0,
            },
            tensors,
        )
    }

    #[test]
    fn accepts_the_emitted_glu_down_pair() {
        let (program, tensors) = fixture();
        let routes = segments(&program, &tensors, &[0, ROLE, 0]).unwrap();
        let route = routes[1].expect("routed pair");
        assert_eq!((route.glu, route.down), (1, 2));
        assert_eq!(
            (route.hidden, route.inter, route.experts, route.average_rows),
            (2816, 704, 128, 64)
        );
        assert_eq!(u64::from(route.capacity), CAPACITY);
        assert!(routes[0].is_none() && routes[2].is_none());
        assert!(segments(&program, &tensors, &[0, 0, 0])
            .unwrap()
            .iter()
            .all(Option::is_none));
        assert!(segments(&program, &tensors, &[0, ROLE]).is_err());
    }

    #[test]
    fn accepts_the_pair_with_the_prefill_combine() {
        let (mut program, mut tensors) = fixture();
        let comb = &mut program.insts[3];
        comb.op = DevOp::MoeCombineNormGemmaPf as u16;
        comb.t[..4].copy_from_slice(&[9, 5, 10, 11]);
        comb.i[..3].copy_from_slice(&[2816, 8, 1024]);
        for bytes in [1024 * 2816 * 2, 1024 * 2816 * 2, 2816 * 2] {
            tensors.push(DevTensor {
                name: "act".into(),
                bytes,
                init: None,
            });
        }
        program.stream[3].seg = 1;
        program.gq_stream[3].seg = 1;
        program.gq_seg_ofs = vec![0, 1, 4];
        let roles = [0, ROLE];
        assert_eq!(packet_role_segments(&program, &roles, &tensors).unwrap(), roles);
        let route = segments(&program, &tensors, &roles).unwrap()[1].expect("routed tail");
        let tail = route.pf_tail.expect("prefill combine");
        assert_eq!((tail.combine, tail.handles, tail.tokens), (3, [9, 10, 11], 1024));
        assert_eq!(instructions(&[None, Some(route)]), [(1, 1), (1, 2), (1, 3)]);
        // The combine must read the down op's part, at k = 8.
        for edit in [|p: &mut DevProg| p.insts[3].t[1] = 1, |p: &mut DevProg| p.insts[3].i[1] = 4] {
            let saved = program.insts[3];
            edit(&mut program);
            assert!(segments(&program, &tensors, &roles).is_err());
            program.insts[3] = saved;
        }
    }

    #[test]
    fn accepts_the_w8a8_chain() {
        // 0 align, 1 activation quant, 2 GLU, 3 fu quant, 4 DOWN, 5 combine.
        let mut insts = vec![
            DevInst64 {
                op: DevOp::Nop as u16,
                blocks: 1,
                ..Default::default()
            };
            6
        ];
        for inst in &mut insts {
            inst.t.fill(packet::dev::TENSOR_NONE16);
        }
        // Handles: 0 fug, 1 xq, 2 ewt, 3 meta, 4 rowtok, 5 part, 6 rowpart, 7 rowgate, 8 table,
        // 9 ascale, 10 est, 11 fuq, 12 fus, 13 comb, 14 h1, 15 gamma, 16 xn2.
        insts[0].op = DevOp::MoeAlignGemmaPf as u16;
        insts[0].t[..5].copy_from_slice(&[3, 8, 4, 6, 7]);
        insts[0].i[..3].copy_from_slice(&[1024, 128, 8]);
        insts[1].op = DevOp::QuantFp8 as u16;
        insts[1].t[..3].copy_from_slice(&[1, 16, 9]);
        insts[1].i[..2].copy_from_slice(&[1024, 2816]);
        insts[2].op = DevOp::MoeGroupGluGemmaPfW8a8 as u16;
        insts[2].t[..7].copy_from_slice(&[0, 1, 2, 3, 4, 9, 10]);
        insts[2].i[..3].copy_from_slice(&[704, 2816, 128]);
        insts[3].op = DevOp::QuantFp8 as u16;
        insts[3].t[..3].copy_from_slice(&[11, 0, 12]);
        insts[3].i[..2].copy_from_slice(&[CAPACITY as u32, 704]);
        insts[4].op = DevOp::MoeGroupDownGemmaPfW8a8 as u16;
        insts[4].t[..8].copy_from_slice(&[5, 11, 2, 3, 6, 7, 10, 12]);
        insts[4].i[..3].copy_from_slice(&[2816, 704, 128]);
        insts[5].op = DevOp::MoeCombineNormGemmaPf as u16;
        insts[5].t[..4].copy_from_slice(&[13, 5, 14, 15]);
        insts[5].i[..3].copy_from_slice(&[2816, 8, 1024]);
        let segs = [0, 0, 1, 1, 1, 1];
        let stream: Vec<_> = (0..6u32)
            .map(|inst| StreamEnt {
                inst,
                seg: segs[inst as usize],
                ..Default::default()
            })
            .collect();
        let tensors = [
            CAPACITY * 704 * 2,
            1024 * 2816,
            128 * 16,
            (3 * 128 + 2) * 4,
            CAPACITY * 4,
            1024 * 8 * 2816 * 4,
            CAPACITY * 4,
            CAPACITY * 4,
            1024 * 8 * 8,
            1024 * 4,
            128 * 16,
            CAPACITY * 704,
            CAPACITY * 4,
            1024 * 2816 * 2,
            1024 * 2816 * 2,
            2816 * 2,
            1024 * 2816 * 2,
        ]
        .into_iter()
        .map(|bytes| DevTensor {
            name: "act".into(),
            bytes,
            init: None,
        })
        .collect::<Vec<_>>();
        let mut program = DevProg {
            t: 1024,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: 1024 },
            n_counter: 0,
            insts,
            stream: stream.clone(),
            stream_ofs: vec![0],
            stream_len: vec![6],
            waits: vec![],
            succs: vec![],
            gq_stream: stream,
            gq_seg_ofs: vec![0, 2, 6],
            l2_domains: 0,
        };
        let roles = [0, ROLE];
        assert_eq!(packet_role_segments(&program, &roles, &tensors).unwrap(), roles);
        let route = segments(&program, &tensors, &roles).unwrap()[1].expect("routed chain");
        assert_eq!((route.glu, route.down), (2, 4));
        assert_eq!(route.fp8.map(|f| (f.quant, f.handles)), Some((3, [9, 10, 12])));
        assert_eq!(route.pf_tail.map(|t| t.combine), Some(5));
        assert_eq!(instructions(&[None, Some(route)]), [(1, 2), (1, 4), (1, 5), (1, 3)]);
        // The fu quant must feed the DOWN, and the DOWN must read the GLU's scale table. The
        // activation quant must write the GLU's xq and ascale over [tokens, hidden], unfolded,
        // in an earlier segment.
        let edits: [fn(&mut DevProg); 9] = [
            |p| p.insts[3].t[0] = 1,
            |p| p.insts[4].t[6] = 9,
            |p| p.insts[1].op = DevOp::Nop as u16,
            |p| p.insts[1].t[0] = 11,
            |p| p.insts[1].t[2] = 12,
            |p| p.insts[1].t[3] = 0,
            |p| p.insts[1].i[0] = 512,
            |p| p.insts[1].i[1] = 704,
            |p| {
                p.stream.remove(1);
            },
        ];
        for edit in edits {
            let saved = (program.insts.clone(), program.stream.clone());
            edit(&mut program);
            assert!(segments(&program, &tensors, &roles).is_err());
            (program.insts, program.stream) = saved;
        }
    }

    #[test]
    fn packet_role_validator_takes_the_two_instruction_library_segment() {
        let (program, tensors) = fixture();
        assert_eq!(
            packet_role_segments(&program, &[0, ROLE, 0], &tensors).unwrap(),
            [0, ROLE, 0]
        );
        // A segment of one instruction (or of three) is not this role.
        for (bounds, segs) in [([0, 1, 2, 4], [0, 1, 2, 2]), ([0, 1, 4, 4], [0, 1, 1, 1])] {
            let (mut split, tensors) = fixture();
            split.gq_seg_ofs = bounds.to_vec();
            for entry in split.stream.iter_mut().chain(&mut split.gq_stream) {
                entry.seg = segs[entry.inst as usize];
            }
            assert!(packet_role_segments(&split, &[0, ROLE, 0], &tensors).is_err());
        }
    }

    #[test]
    fn rejects_foreign_operands_geometry_and_short_tables() {
        let roles = [0, ROLE, 0];
        let edits: [fn(&mut DevProg, &mut Vec<DevTensor>); 6] = [
            |p, _| p.insts[2].t[1] = 1,
            |p, _| p.insts[2].i[1] = 712,
            |p, _| {
                p.insts[1].i[0] = 700;
                p.insts[2].i[1] = 700;
            },
            |p, _| p.insts[0].op = DevOp::Nop as u16,
            |p, _| {
                p.stream[3].seg = 1;
                p.gq_stream[3].seg = 1;
                p.gq_seg_ofs = vec![0, 1, 4, 4];
            },
            |_, t| t[6].bytes -= 4,
        ];
        for edit in edits {
            let (mut program, mut tensors) = fixture();
            edit(&mut program, &mut tensors);
            assert!(segments(&program, &tensors, &roles).is_err());
        }
    }

    const DECODE_ROLE: u8 = plow_asset::segment_roles::MOE_DECODE_CUBLASLT;

    /// A 16-row decode rung of the grouped arm (threshold 8): align | glu, down | nop.
    fn decode_fixture() -> (DevProg, Vec<DevTensor>) {
        let (mut program, mut tensors) = fixture();
        program.t = 16;
        program.role = packet::devbuild::ProgramRole::DecodeRung { rows: 16 };
        // Handles: 0 fug, 1 xn2, 2 ewt, 3 meta, 4 rowtok, 5 part, 6 rowpart, 7 rowgate,
        // 8 table, 9 x, 10 gamma.
        tensors.extend([16 * 2816 * 2, 2816 * 2].map(|bytes| DevTensor {
            name: "act".into(),
            bytes,
            init: None,
        }));
        program.insts[0].i[..4].copy_from_slice(&[16, 128, 8, 8]);
        let glu = &mut program.insts[1];
        glu.op = DevOp::MoeExpertGluNormGemma as u16;
        glu.t = [0, 9, 8, 2, 10, 1, 3, 4];
        glu.i = [8, 704, 2816, 128, 0, 16, 8, 0];
        glu.fj[0] = 1e-6f32.to_bits();
        let down = &mut program.insts[2];
        down.op = DevOp::MoeExpertDownGemma as u16;
        down.t = [5, 0, 8, 2, 2, 3, 6, 7];
        down.i = [8, 2816, 704, 128, 0, 16, 8, 0];
        (program, tensors)
    }

    #[test]
    fn accepts_the_grouped_decode_glu_down_pair() {
        let (program, tensors) = decode_fixture();
        let routes = decode_segments(&program, &tensors, &[0, DECODE_ROLE, 0]).unwrap();
        let route = routes[1].expect("routed pair");
        assert_eq!((route.glu, route.down), (1, 2));
        assert_eq!(
            (route.hidden, route.inter, route.experts, route.average_rows, route.act),
            (2816, 704, 128, 1, 0)
        );
        assert_eq!(u64::from(route.capacity), 16 * 8 + 128 * ALIGN_PAD);
        assert_eq!(route.handles, [2, 3, 4, 6, 7, 1, 0, 5]);
        assert_eq!(
            route.norm,
            Some(Norm {
                x: 9,
                gamma: 10,
                rows: 16,
                eps_bits: 1e-6f32.to_bits()
            })
        );
        assert_eq!(instructions(&routes), [(1, 1), (1, 2)]);
        assert!(routes[0].is_none() && routes[2].is_none());
        assert!(decode_segments(&program, &tensors, &[0, 0, 0])
            .unwrap()
            .iter()
            .all(Option::is_none));
        // The prefill validator does not take the decode pair and vice versa.
        assert!(segments(&program, &tensors, &[0, ROLE, 0]).is_err());
        let (prefill, tensors) = fixture();
        assert!(decode_segments(&prefill, &tensors, &[0, DECODE_ROLE, 0]).is_err());
        assert_eq!(
            packet_role_segments(&program, &[0, DECODE_ROLE, 0], &tensors).unwrap(),
            [0, DECODE_ROLE, 0]
        );
    }

    /// `decode_fixture` with the layer tail in the routed segment: align | glu, down, comb, nrn | nop.
    fn decode_tail_fixture() -> (DevProg, Vec<DevTensor>) {
        let (mut program, mut tensors) = decode_fixture();
        // Handles: 11 comb, 12 h1, 13 g_pf2, 14 hn, 15 g_po, 16 gn.
        let row = 16 * 2816 * 2;
        tensors.extend([row, row, 2816 * 2, row, 2816 * 2, 2816 * 2].map(|bytes| DevTensor {
            name: "act".into(),
            bytes,
            init: None,
        }));
        let nop = program.insts[3];
        program.insts.splice(3..3, [nop, nop]);
        let comb = &mut program.insts[3];
        comb.op = DevOp::MoeCombineNormGemma as u16;
        comb.t[..4].copy_from_slice(&[11, 5, 12, 13]);
        comb.i[..3].copy_from_slice(&[2816, 8, 16]);
        comb.fj[0] = 1e-6f32.to_bits();
        let nrn = &mut program.insts[4];
        nrn.op = DevOp::NormResidualNorm as u16;
        nrn.t[..6].copy_from_slice(&[14, 9, 9, 11, 15, 16]);
        nrn.i[..2].copy_from_slice(&[16, 2816]);
        nrn.fj[..2].copy_from_slice(&[1e-6f32.to_bits(), 0.5f32.to_bits()]);
        let stream: Vec<_> = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2)]
            .into_iter()
            .map(|(inst, seg)| StreamEnt {
                inst,
                seg,
                ..Default::default()
            })
            .collect();
        program.stream = stream.clone();
        program.gq_stream = stream;
        program.stream_len = vec![6];
        program.gq_seg_ofs = vec![0, 1, 5, 6];
        (program, tensors)
    }

    #[test]
    fn accepts_the_decode_pair_with_the_layer_tail() {
        let (program, tensors) = decode_tail_fixture();
        let roles = [0, DECODE_ROLE, 0];
        let routes = decode_segments(&program, &tensors, &roles).unwrap();
        let route = routes[1].expect("routed pair");
        assert_eq!(
            route.tail,
            Some(Tail {
                combine: 3,
                nrn: 4,
                handles: [14, 11, 12, 13, 15, 16],
                eps_comb_bits: 1e-6f32.to_bits(),
                eps_bits: 1e-6f32.to_bits(),
                scale_bits: 0.5f32.to_bits(),
            })
        );
        assert_eq!(instructions(&routes), [(1, 1), (1, 2), (1, 3), (1, 4)]);
        assert_eq!(packet_role_segments(&program, &roles, &tensors).unwrap(), roles);
        // A prefill segment never carries the decode tail (four instructions pass the role
        // validator only as the W8A8 chain, which the route checks).
        assert!(segments(&program, &tensors, &[0, ROLE, 0]).is_err());

        let edits: [fn(&mut DevProg, &mut Vec<DevTensor>); 7] = [
            |p, _| p.insts[3].t[1] = 7,
            |p, _| p.insts[3].i[1] = 4,
            |p, _| p.insts[4].t[2] = 14,
            |p, _| p.insts[4].t[3] = 12,
            |p, _| p.insts[4].i[0] = 8,
            |p, _| p.insts[4].op = DevOp::RmsNorm as u16,
            |_, t| t[12].bytes -= 2,
        ];
        for edit in edits {
            let (mut program, mut tensors) = decode_tail_fixture();
            edit(&mut program, &mut tensors);
            assert!(decode_segments(&program, &tensors, &roles).is_err());
        }
        // Three instructions: the tail is all or nothing.
        let (mut program, tensors) = decode_tail_fixture();
        program.gq_seg_ofs = vec![0, 1, 4, 6];
        for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
            entry.seg = [0, 1, 1, 1, 2, 2][entry.inst as usize];
        }
        assert!(decode_segments(&program, &tensors, &roles).is_err());
    }

    #[test]
    fn decode_pair_rejects_rungs_below_the_threshold_and_foreign_operands() {
        let roles = [0, DECODE_ROLE, 0];
        let edits: [fn(&mut DevProg, &mut Vec<DevTensor>); 8] = [
            |p, _| p.insts[0].i[0] = 4,
            |p, _| p.insts[1].i[6] = 32,
            |p, _| p.insts[1].i[5] = 8,
            |p, _| p.insts[2].t[5] = 4,
            |p, _| p.insts[1].t[4] = packet::dev::TENSOR_NONE16,
            |p, _| p.insts[1].i[4] = 2,
            |_, t| t[9].bytes -= 2,
            |_, t| t[5].bytes = 16 * 8 * 2816 * 4 - 4,
        ];
        for edit in edits {
            let (mut program, mut tensors) = decode_fixture();
            edit(&mut program, &mut tensors);
            assert!(decode_segments(&program, &tensors, &roles).is_err());
        }
    }
}
