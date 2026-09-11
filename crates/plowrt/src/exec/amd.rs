//! AMD HSA execution: model loading, per-phase dispatch and slot state.
//!
//! Object qualification lives in `object`; prefix and token-batch adapters
//! own their feature state. Counter resets precede the whole dispatch group,
//! and AQL barriers order its segments before the final drain.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use packet::dev::{DevInst64, DevOp, DevProgram, PrefillSpan, SE_XCTR};
use packet::devbuild::{lean_attn_res_f32mix_inst64, static_seg_ofs};
use serde::Serialize;

use super::kv_layout::kv_tensor_name;
use crate::asset::devblob::{DevBlob, DevProg};
use crate::device::hsa::{HsaBackend, HsaKernel, HsaPinned};
use crate::device::{DeviceMem, Module};
use crate::exec::amd::packed::validate_rows as validate_amd_packed_rows;
use crate::exec::device_api::EngineDevice;
use crate::exec::kvrow::{
    derive_kvrow, derive_mla_nsplit, is_lm_head_matmul, kvrow_span, mla_live_nsplit,
    prefill_row_field, rebase_chunk_rows, RowField,
};
use crate::exec::{amd_gemm_lt, amd_index_tp, amd_mla_fold, amd_moe_aiter, amd_sparse_mla};
use crate::memory::slab_pad;
use crate::memory::vmm::{VmmGeometry, VmmKv, VmmOps, WeightSlab};
#[cfg(test)]
use crate::memory::SLAB_ALIGN;
use crate::{Result, RuntimeError};

mod object;
mod packed;
mod segment;
pub(super) use object::elf_symbol_names;
use object::{
    build_requires, check_attn_res_f32mix_symbols, check_compiled_opcode_marker_set,
    check_compiled_opcode_markers, check_dec_stage_capacity, check_decode_object,
    check_dsa_decode_batch, check_dsa_select_local, check_dsa_pf_arm, check_gate_hier_object, check_gemv_capacity,
    check_interpreter_waves,
    check_k3_arms, check_kda_carry_regstate_symbols, check_kda_chunk, check_kda_conv_step_db,
    check_kda_intra_wave_items_symbols, check_kv_encoding, check_materialized_residual_input,
    check_mla_nope_arm, check_mla_v2_sv_raw_symbols, check_moe_ep_symbols, check_moe_gemma_arms,
    check_moe_pf_a4w4, check_packed_family_kv_encoding, check_packed_prefill_abi,
    check_packet_pairing_stamp, check_prefill_object, check_qwen_gdn_arms, check_sparse_fp8_object,
    check_sparse_fp8_packet, check_xargmax_capacity, elf_symbol_u32, first_op_in,
    graph_phase_xreduce_segments, l2_pairing_refusal, log_gate_hier_status,
    packet_decode_arm_requirements, packet_prefill_arm_requirements, read_attn_res_f32mix_object,
    read_kda_carry_regstate_object, read_kda_intra_wave_items_object, required_gemv_m,
    required_k3_op, required_kda_chunk, required_kda_conv_step_db, required_kv_op,
    required_moe_pf_a4w4, required_moe_pf_accum, required_qwen_gdn_op, resolve_flash_object_load,
    sparse_fp8, BF16_KV_OPS, FP8_KV_OPS, FP8_KV_SYM, GATE_HIER_SYM, KDA_CARRY_KEYFEED_MARKERS,
    KDA_CARRY_REGSTATE_LDS, KDA_CHUNK_QPRE_SYM, KDA_CONV_STEP_DB_REPLACED_OPS, KDA_FAMILY_SEG_SYM,
    KDA_WU_LEAN_LDS, KDA_WU_LEAN_MARKERS, L2_DISPATCH_SYM, MLA_PF_V2_FP8_SYM, MLA_PF_V2_SYM,
    MOE_ENC_MXFP4, MOE_GEMMA_OPS, MOE_GEMMA_PF_OPS, PACKED_PREFILL_ABI_SYM,
    PACKED_PREFILL_KDA_CHUNK_SEG_SYM, PACKED_PREFILL_KDA_SEG_SYM, PACKED_PREFILL_MLA_FLASH_SEG_SYM,
    PACKED_PREFILL_MLA_NORM_SEG_SYM, XR_ATTNRES_RESOURCE_SYM, XR_ATTNRES_SEG_SYM, XR_TAGGED_SYM,
};
pub use object::{mla_pf_v2_enabled, object_name, symbol_name, Phase, PrefillArm, Sched, Variant};

/// u32 slots per counter (`PLOW_CTR_STRIDE`), i.e. one 128 B cache line.
const CTR_STRIDE_U32: usize = 32;

/// DOUBLE-BUFFER the counter/cursor banks so the per-dispatch zeroing overlaps
/// GPU execution instead of standing in front of it. `--amd-ctr-dbuf false` /
/// `PLOW_CTR_DBUF=0` reverts to the single bank cleared synchronously before
/// every launch.
///
/// The measured case for it, on this box (MI300X, Gemma-4-12B fp8, ctx 4096,
/// `PLOW_DSTEP_LOG=1`): the synchronous `rearm` is **56 µs of the 11.69 ms
/// token**, 0.48%, and it is 57% of the entire host phase (99 µs, 0.85%).
/// It costs that much because the clear is two blocking
/// `hsa_amd_memory_async_copy` round trips over `n_counter * 128 B`, and it
/// sits between "host has staged this token" and "the GPU may start".
///
/// The alternative the review priced — a kernel prologue that self-clears —
/// removes the same 56 µs from the host but ADDS ~2 µs to the GPU critical
/// path and needs a kernel/ABI change. This needs neither: the copy still
/// happens, it just happens while 304 CUs are busy, over the SDMA engine that
/// the persistent megakernel does not contend for.
fn ctr_dbuf() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| crate::config::RuntimeConfig::get().amd.ctr_dbuf)
}

/// `sizeof(PlowTraceRec)` (`runtime/common/dev_isa.h`, static-asserted at 40).
/// One record per (workgroup, packet), slotted at `stream_ofs[cu] + pc`.
const TRACE_REC_BYTES: usize = 40;

/// Wave-class 8 is `PLOW_WG_WAVES` = 8 waves of 64.
const WG_THREADS_8: u32 = Phase::Prefill.interpreter_threads();
/// The flash object is built 4-wave. Dispatching it at 512 threads is an
/// `INVALID_ISA`, not a slowdown.
const WG_THREADS_4: u32 = Phase::Flash.interpreter_threads();

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AmdOwnedRange {
    pub name: String,
    pub address_space: &'static str,
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AmdOverlapRankEvidence {
    pub rank: usize,
    pub queue_count: usize,
    pub prefill_ranges: Vec<AmdOwnedRange>,
    pub decode_ranges: Vec<AmdOwnedRange>,
    pub prefill_queue_ids: Vec<u64>,
    pub decode_queue_ids: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AmdOverlapCapability {
    pub scratch_isolated: bool,
    pub queue_isolated: bool,
    pub overlap_safe: bool,
    pub queue_scope: &'static str,
    pub queue_count: usize,
    pub per_xcd_queues: bool,
    pub ranks: usize,
}

fn ranges_overlap(a: &AmdOwnedRange, b: &AmdOwnedRange) -> bool {
    a.address_space == b.address_space && a.start < b.end && b.start < a.end
}

pub fn derive_overlap_capability(ranks: &[AmdOverlapRankEvidence]) -> AmdOverlapCapability {
    let scratch_isolated = !ranks.is_empty()
        && ranks.iter().all(|rank| {
            !rank.prefill_ranges.is_empty()
                && !rank.decode_ranges.is_empty()
                && !rank
                    .prefill_ranges
                    .iter()
                    .any(|a| rank.decode_ranges.iter().any(|b| ranges_overlap(a, b)))
        });
    let queue_isolated = !ranks.is_empty()
        && ranks.iter().all(|rank| {
            rank.queue_count >= 2
                && !rank.prefill_queue_ids.is_empty()
                && !rank.decode_queue_ids.is_empty()
                && !rank
                    .prefill_queue_ids
                    .iter()
                    .any(|id| rank.decode_queue_ids.contains(id))
        });
    let queue_count = ranks.iter().map(|rank| rank.queue_count).min().unwrap_or(0);
    AmdOverlapCapability {
        scratch_isolated,
        queue_isolated,
        overlap_safe: scratch_isolated && queue_isolated,
        queue_scope: "global_per_rank",
        queue_count,
        per_xcd_queues: false,
        ranks: ranks.len(),
    }
}

/// Host dispatch semantics for ordered segments and device-side XCD queues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgramDispatch {
    /// Legacy placed blob: `seg` was the XCD and there is one host launch.
    L2Domains(u32),
    /// Current placed blob: every ordered segment owns one queue per XCD.
    L2Segments {
        domains: u32,
        segments: usize,
    },
    WaveSegments(usize),
}

impl ProgramDispatch {
    pub(crate) fn classify(l2_domains: u32, n_segments: usize, queue_windows: usize) -> Self {
        if l2_domains == 0 {
            return Self::WaveSegments(n_segments);
        }
        if queue_windows == n_segments.saturating_mul(l2_domains as usize) {
            return Self::L2Segments {
                domains: l2_domains,
                segments: n_segments,
            };
        }
        Self::L2Domains(l2_domains)
    }

    pub(crate) fn launches(self) -> usize {
        match self {
            Self::L2Domains(_) => 1,
            Self::L2Segments { segments, .. } => segments,
            Self::WaveSegments(n) => n,
        }
    }
}

fn prefill_segment_specialization_allowed(dispatch: ProgramDispatch) -> bool {
    !matches!(dispatch, ProgramDispatch::L2Domains(_))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeSegmentKind {
    Interpreter,
    MlaAttention,
    KdaDecodeFused(usize),
    MoeAiter,
    GemmLt,
    GroupedMoeMxfp4 { glu: usize, down: usize },
}

fn decode_segment_kinds(prog: &DevProg) -> Result<Vec<DecodeSegmentKind>> {
    let n_segments = derive_segments(prog)?.len();
    let mut kinds = vec![DecodeSegmentKind::Interpreter; n_segments];
    let mut raw_segment_owner = vec![None; prog.insts.len()];
    for seg in 0..n_segments {
        let mut fused_inst = None;
        let mut mla_flash_inst = None;
        let mut mla_merge_inst = None;
        let mut multiple_mla_flash = false;
        let mut multiple_mla_merge = false;
        let mut moe_glu_inst = None;
        let mut moe_down_inst = None;
        let mut multiple_moe_glu = false;
        let mut multiple_moe_down = false;
        let mut has_other = false;
        for e in prog.stream.iter().filter(|e| e.seg as usize == seg) {
            let inst = prog.insts.get(e.inst as usize).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "decode segment {seg} references instruction {} of {}",
                    e.inst,
                    prog.insts.len()
                ))
            })?;
            if inst.op == DevOp::FlashMlaDecode as u16 {
                match mla_flash_inst {
                    None => mla_flash_inst = Some(e.inst as usize),
                    Some(i) if i == e.inst as usize => {}
                    Some(_) => multiple_mla_flash = true,
                }
            } else if inst.op == DevOp::MlaMergeFold as u16 {
                match mla_merge_inst {
                    None => mla_merge_inst = Some(e.inst as usize),
                    Some(i) if i == e.inst as usize => {}
                    Some(_) => multiple_mla_merge = true,
                }
            } else if inst.op == DevOp::KdaDecodeFused as u16
                || inst.op == DevOp::MoeAiterFp8Pf as u16
                || inst.op == DevOp::GemmLtPf as u16
            {
                let name = if inst.op == DevOp::MoeAiterFp8Pf as u16 {
                    "MoeAiterFp8Pf"
                } else if inst.op == DevOp::GemmLtPf as u16 {
                    "GemmLtPf"
                } else {
                    "KdaDecodeFused"
                };
                match fused_inst {
                    None => fused_inst = Some(e.inst as usize),
                    Some(i) if i == e.inst as usize => {}
                    Some(i) => {
                        return Err(RuntimeError::Device(format!(
                            "decode segment {seg} mixes {name} instructions {i} and {}",
                            e.inst
                        )))
                    }
                }
                match raw_segment_owner[e.inst as usize] {
                    Some(owner) if owner != seg => {
                        return Err(RuntimeError::Device(format!(
                            "raw instruction {} is referenced by segments {owner} and {seg}; a stateful raw boundary may execute exactly once",
                            e.inst
                        )))
                    }
                    None => raw_segment_owner[e.inst as usize] = Some(seg),
                    _ => {}
                }
                if e.wait_len != 0 || e.succ_len != 0 || e.flags & SE_XCTR != 0 {
                    return Err(RuntimeError::Device(format!(
                        "{name} instruction {} has counter obligations in segment {seg} (waits={}, succs={}, flags={:#x}); a raw kernel cannot service interpreter counters",
                        e.inst, e.wait_len, e.succ_len, e.flags
                    )));
                }
            } else if inst.op == DevOp::MoeGroupGluFp8Blk as u16
                || inst.op == DevOp::MoeGroupDownFp8Blk as u16
            {
                let (slot, multiple) = if inst.op == DevOp::MoeGroupGluFp8Blk as u16 {
                    (&mut moe_glu_inst, &mut multiple_moe_glu)
                } else {
                    (&mut moe_down_inst, &mut multiple_moe_down)
                };
                match *slot {
                    None => *slot = Some(e.inst as usize),
                    Some(i) if i == e.inst as usize => {}
                    Some(_) => *multiple = true,
                }
            } else {
                has_other = true;
            }
        }
        if let Some(inst) = fused_inst {
            let name = if prog.insts[inst].op == DevOp::MoeAiterFp8Pf as u16 {
                "MoeAiterFp8Pf"
            } else if prog.insts[inst].op == DevOp::GemmLtPf as u16 {
                "GemmLtPf"
            } else {
                "KdaDecodeFused"
            };
            if has_other
                || mla_flash_inst.is_some()
                || mla_merge_inst.is_some()
                || moe_glu_inst.is_some()
                || moe_down_inst.is_some()
            {
                return Err(RuntimeError::Device(format!(
                    "decode segment {seg} mixes {name} with interpreter opcodes; standalone dispatch requires a pure segment"
                )));
            }
            kinds[seg] = if prog.insts[inst].op == DevOp::MoeAiterFp8Pf as u16 {
                DecodeSegmentKind::MoeAiter
            } else if prog.insts[inst].op == DevOp::GemmLtPf as u16 {
                DecodeSegmentKind::GemmLt
            } else {
                DecodeSegmentKind::KdaDecodeFused(inst)
            };
        } else if !has_other && moe_glu_inst.is_some() && moe_down_inst.is_some() {
            let (Some(glu), Some(down)) = (moe_glu_inst, moe_down_inst) else {
                unreachable!()
            };
            if multiple_moe_glu || multiple_moe_down || down != glu + 1 {
                return Err(RuntimeError::Device(format!(
                    "decode segment {seg} is not a pure adjacent grouped MoE GLU+DOWN pair"
                )));
            }
            for e in prog.stream.iter().filter(|e| {
                e.seg as usize == seg && matches!(e.inst as usize, i if i == glu || i == down)
            }) {
                if e.wait_len != 0 || e.succ_len != 0 || e.flags & SE_XCTR != 0 {
                    return Err(RuntimeError::Device(format!(
                        "grouped MoE instruction {} has counter obligations in segment {seg}",
                        e.inst
                    )));
                }
                match raw_segment_owner[e.inst as usize] {
                    Some(owner) if owner != seg => {
                        return Err(RuntimeError::Device(format!(
                            "raw instruction {} is referenced by segments {owner} and {seg}",
                            e.inst
                        )))
                    }
                    None => raw_segment_owner[e.inst as usize] = Some(seg),
                    _ => {}
                }
            }
            kinds[seg] = DecodeSegmentKind::GroupedMoeMxfp4 { glu, down };
        } else if !has_other && (mla_flash_inst.is_some() || mla_merge_inst.is_some()) {
            let (Some(flash), Some(merge)) = (mla_flash_inst, mla_merge_inst) else {
                return Err(RuntimeError::Device(format!(
                    "decode segment {seg} contains only half of the FlashMlaDecode+MlaMergeFold pair"
                )));
            };
            if multiple_mla_flash || multiple_mla_merge || merge != flash + 1 {
                return Err(RuntimeError::Device(format!(
                    "decode segment {seg} is not a pure adjacent FlashMlaDecode+MlaMergeFold pair"
                )));
            }
            kinds[seg] = DecodeSegmentKind::MlaAttention;
        }
    }
    for (i, inst) in prog.insts.iter().enumerate() {
        if matches!(
            DevOp::from_u16(inst.op),
            Some(DevOp::KdaDecodeFused | DevOp::MoeAiterFp8Pf)
        ) && raw_segment_owner[i].is_none()
        {
            return Err(RuntimeError::Device(format!(
                "raw instruction {i} is absent from every stream segment"
            )));
        }
    }

    // Ordinary L2 placement also uses `seg`, but does not relaunch the program at those
    // boundaries. Its interpreter counters may therefore cross domains. Only the standalone
    // raw-kernel route replaces cross-segment counter ordering with ordered AQL launches.
    if kinds
        .iter()
        .all(|kind| matches!(kind, DecodeSegmentKind::Interpreter))
    {
        return Ok(kinds);
    }

    let mut producer_segments: std::collections::HashMap<u32, Vec<u16>> =
        std::collections::HashMap::new();
    for e in &prog.stream {
        if e.flags & SE_XCTR != 0 {
            continue;
        }
        let end = e.succ_ofs as usize + e.succ_len as usize;
        let succs = prog.succs.get(e.succ_ofs as usize..end).ok_or_else(|| {
            RuntimeError::Device(format!(
                "decode stream successor range {}..{end} is out of bounds ({})",
                e.succ_ofs,
                prog.succs.len()
            ))
        })?;
        for &counter in succs {
            producer_segments.entry(counter).or_default().push(e.seg);
        }
    }
    for e in &prog.stream {
        if e.flags & SE_XCTR != 0 {
            continue;
        }
        let end = e.wait_ofs as usize + e.wait_len as usize;
        let waits = prog.waits.get(e.wait_ofs as usize..end).ok_or_else(|| {
            RuntimeError::Device(format!(
                "decode stream wait range {}..{end} is out of bounds ({})",
                e.wait_ofs,
                prog.waits.len()
            ))
        })?;
        for wait in waits {
            if producer_segments
                .get(&wait.id)
                .is_some_and(|segs| segs.iter().any(|&s| s != e.seg))
            {
                return Err(RuntimeError::Device(format!(
                    "decode counter {} crosses into segment {}; segmented raw/interpreter dispatch requires cross-segment waits to be removed",
                    wait.id, e.seg
                )));
            }
        }
    }
    Ok(kinds)
}

fn validate_decode_dispatch(progs: &[DevProg], dec_ix: usize) -> Result<()> {
    for (rung, prog) in progs[dec_ix..].iter().enumerate() {
        let kinds = decode_segment_kinds(prog)?;
        let n_segments = kinds.len();
        let dispatch = ProgramDispatch::classify(
            prog.l2_domains,
            n_segments,
            prog.gq_seg_ofs.len().saturating_sub(1),
        );
        if matches!(dispatch, ProgramDispatch::L2Domains(_))
            && kinds
                .iter()
                .any(|k| !matches!(k, DecodeSegmentKind::Interpreter))
        {
            return Err(RuntimeError::Device(format!(
                "decode program {} (rung {rung}, t={}) mixes raw kernels and interpreter segments                  with L2-domain placement; raw boundaries require ordered wave segments",
                dec_ix + rung,
                prog.t,
            )));
        }
        if prog.l2_domains == 0
            && n_segments > 1
            && !kinds
                .iter()
                .any(|k| !matches!(k, DecodeSegmentKind::Interpreter))
        {
            return Err(RuntimeError::Device(format!(
                "decode program {} (rung {rung}, t={}) has {n_segments} wave segments but no                  standalone raw boundary; ordinary AMD decode remains single-launch",
                dec_ix + rung,
                prog.t
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaDecodeFusedArgs {
    y: u64,
    q_raw: u64,
    k_raw: u64,
    v_raw: u64,
    wq: u64,
    wk: u64,
    wv: u64,
    csq: u64,
    csk: u64,
    csv: u64,
    forget_raw: u64,
    beta_raw: u64,
    output_gate_raw: u64,
    a_log: u64,
    dt_bias: u64,
    state: u64,
    norm_w: u64,
    parked: u64,
    rows: u32,
    heads: u32,
    dim: u32,
    bv: u32,
    conv_w: u32,
    flags: u32,
    gate_mode: u32,
    lower_bound: f32,
    scale: f32,
    norm_eps: f32,
}

const _: () = assert!(std::mem::size_of::<KdaDecodeFusedArgs>() == 184);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct GroupedMoeGluArgs {
    fu: u64,
    x: u64,
    table: u64,
    weights: u64,
    scales: u64,
    topk: u32,
    intermediate: u32,
    hidden: u32,
    experts: u32,
    act: u32,
    enc: u32,
    beta: f32,
    linear_beta: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct GroupedMoeDownArgs {
    partial: u64,
    fu: u64,
    table: u64,
    weights: u64,
    scales: u64,
    topk: u32,
    hidden: u32,
    intermediate: u32,
    experts: u32,
    enc: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<GroupedMoeGluArgs>() == 72);
const _: () = assert!(std::mem::size_of::<GroupedMoeDownArgs>() == 64);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkIntraCachedArgs {
    aqk: u64,
    ainv: u64,
    q: u64,
    k: u64,
    g_prefix: u64,
    beta: u64,
    t: u32,
    heads: u32,
    dim: u32,
    scale: f32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkIntraCachedArgs>() == 64);

/// Kernarg ABI of `plow_attn_res_f32mix_gfx950` (runtime/amd/attn_res_f32mix_gfx950.hip).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct AttnResF32MixArgs {
    out: u64,
    prefix: u64,
    ring: u64,
    score_w: u64,
    push_src: u64,
    gamma: u64,
    res_a: u64,
    res_b: u64,
    res_pre: u64,
    t: u32,
    hid: u32,
    nb: u32,
    push_row: u32,
    nbcap: u32,
    eps: f32,
    out_eps: f32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<AttnResF32MixArgs>() == 104);

/// Persistent workgroups for the f32-mix AttnRes object: three 4-wave tokens per CU on 256 CUs
/// measured 0.260 ms at T8192 against 0.279 (512) and 0.283 (1024); `PLOW_ATTNRES_F32MIX_GRID`
/// overrides for a sweep.
fn attn_res_f32mix_grid(t: u32) -> u32 {
    let grid = crate::config::RuntimeConfig::get()
        .amd
        .attnres_f32mix_grid
        .filter(|&g| g != 0)
        .unwrap_or(768);
    grid.min(t).max(1)
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkKeyFactorWuArgs {
    w: u64,
    u: u64,
    key_hi: u64,
    key_lo: u64,
    q: u64,
    ainv: u64,
    k: u64,
    v: u64,
    g: u64,
    beta: u64,
    t: u32,
    heads: u32,
    dim: u32,
    value_dim: u32,
    scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkKeyFactorWuArgs>() == 104);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkKeyFactorCarryArgs {
    out: u64,
    state: u64,
    q: u64,
    k: u64,
    key_hi: u64,
    key_lo: u64,
    w: u64,
    u: u64,
    aqk: u64,
    g: u64,
    t: u32,
    heads: u32,
    dim: u32,
    value_dim: u32,
    scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkKeyFactorCarryArgs>() == 104);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkCarryRegstateArgs {
    out: u64,
    state: u64,
    q: u64,
    k: u64,
    w: u64,
    u: u64,
    aqk: u64,
    g: u64,
    t: u32,
    heads: u32,
    dim: u32,
    value_dim: u32,
    scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkCarryRegstateArgs>() == 88);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkWuLeanArgs {
    w: u64,
    u: u64,
    q: u64,
    key_hi: u64,
    key_lo: u64,
    ainv: u64,
    k: u64,
    v: u64,
    g: u64,
    beta: u64,
    t: u32,
    heads: u32,
    dim: u32,
    value_dim: u32,
    scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkWuLeanArgs>() == 104);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KdaChunkCarryKeyfeedArgs {
    out: u64,
    state: u64,
    q: u64,
    k: u64,
    w: u64,
    u: u64,
    aqk: u64,
    g: u64,
    key_hi: u64,
    key_lo: u64,
    t: u32,
    heads: u32,
    dim: u32,
    value_dim: u32,
    scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<KdaChunkCarryKeyfeedArgs>() == 104);

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct XReduceAttnResArgs {
    reduced: u64,
    prefix: u64,
    out: u64,
    residual: u64,
    ring: u64,
    score: u64,
    gamma: u64,
    peer_scratch: u64,
    xctr: u64,
    status: u64,
    n: u32,
    slot_bytes: u32,
    gate_rs: u32,
    gate_ag: u32,
    rank: u32,
    nranks: u32,
    row_w: u32,
    nb: u32,
    nbcap: u32,
    eps: f32,
}

const _: () = assert!(std::mem::size_of::<XReduceAttnResArgs>() == 120);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeStage2Mxfp4Args {
    part: u64,
    activation: u64,
    weight_table: u64,
    activation_scale: u64,
    weight_scale_table: u64,
    meta: u64,
    row_partidx: u64,
    row_gate: u64,
    model_dim: u32,
    inter_dim: u32,
    experts: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<MoeStage2Mxfp4Args>() == 80);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeStage1Mxfp4Args {
    out: u64,
    activation: u64,
    weight_table: u64,
    weight_scale_table: u64,
    meta: u64,
    row_token: u64,
    row_partidx: u64,
    out_scale: u64,
    inter_dim: u32,
    model_dim: u32,
    experts: u32,
    act: u32,
    beta: f32,
    linear_beta: f32,
    reserved: u32,
    reserved2: u32,
}

const _: () = assert!(std::mem::size_of::<MoeStage1Mxfp4Args>() == 96);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeStage1A4QuantArgs {
    out: u64,
    out_scale: u64,
    activation: u64,
    row_token: u64,
    meta: u64,
    row_capacity: u32,
    experts: u32,
    hidden: u32,
}

const _: () = assert!(std::mem::size_of::<MoeStage1A4QuantArgs>() == 56);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeStage1A4ReuseArgs {
    out: u64,
    activation: u64,
    weight_table: u64,
    activation_scale: u64,
    weight_scale_table: u64,
    meta: u64,
    row_partidx: u64,
    out_scale: u64,
    inter: u32,
    hidden: u32,
    experts: u32,
    act: u32,
    beta: f32,
    linear_beta: f32,
}

const _: () = assert!(std::mem::size_of::<MoeStage1A4ReuseArgs>() == 88);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeCombineArgs {
    out: u64,
    residual: u64,
    shared: u64,
    part: u64,
    hidden: u32,
    topk: u32,
    tokens: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<MoeCombineArgs>() == 48);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeEpAlignArgs {
    routes: u64,
    meta: u64,
    partial: u64,
    row_token: u64,
    row_partidx: u64,
    row_gate: u64,
    tokens: u32,
    topk: u32,
    experts: u32,
    expert_begin: u32,
    expert_end: u32,
    row_capacity: u32,
    phase: u32,
    npart: u32,
}

const _: () = assert!(std::mem::size_of::<MoeEpAlignArgs>() == 80);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MoeEpCombineArgs {
    out: u64,
    part: u64,
    routes: u64,
    tokens: u32,
    hidden: u32,
    topk: u32,
    expert_begin: u32,
    expert_end: u32,
}

const _: () = assert!(std::mem::size_of::<MoeEpCombineArgs>() == 48);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MlaMaterializePackArgs {
    k: u64,
    v: u64,
    kv: u64,
    k_rope: u64,
    t: u32,
    heads: u32,
    qk_nope: u32,
    qk_rope: u32,
    v_head: u32,
}

const _: () = assert!(std::mem::size_of::<MlaMaterializePackArgs>() == 56);

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MlaMaterializedPrefillArgs {
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    b: i32,
    n: i32,
    n_kv: i32,
    h: i32,
    h_kv: i32,
    d_qk: i32,
    d_v: i32,
    stride_q_b: i32,
    stride_q_n: i32,
    stride_q_h: i32,
    stride_o_b: i32,
    stride_o_n: i32,
    stride_o_h: i32,
    stride_k_b: i32,
    stride_k_n: i32,
    stride_k_h: i32,
    stride_v_b: i32,
    stride_v_n: i32,
    stride_v_h: i32,
    softmax_scale: f32,
    seqstart_q: u64,
    seqstart_k: u64,
    seqstart_q_pad: u64,
    seqstart_k_pad: u64,
    opt: i32,
    _pad: u32,
    lse: u64,
    stride_lse_b: i32,
    stride_lse_h: i32,
}

const _: () = assert!(std::mem::size_of::<MlaMaterializedPrefillArgs>() == 168);

#[derive(Clone, Copy, Debug)]
struct MoeStage1Mxfp4Route {
    args: MoeStage1Mxfp4Args,
    grid: u32,
}

#[derive(Clone, Copy, Debug)]
struct MoeStage1A4ReuseRoute {
    quant_args: MoeStage1A4QuantArgs,
    args: MoeStage1A4ReuseArgs,
    quant_grid: u32,
    grid: u32,
}

#[derive(Clone, Copy, Debug)]
struct MoeStage2Mxfp4Route {
    args: MoeStage2Mxfp4Args,
    grid: u32,
}

#[derive(Clone, Copy, Debug)]
struct MoeCombineRoute {
    args: MoeCombineArgs,
    grid: u32,
}

#[derive(Clone, Copy, Debug)]
struct MoeEpAlignRoute {
    args: MoeEpAlignArgs,
}

#[derive(Clone, Copy, Debug)]
struct MoeEpCombineRoute {
    args: MoeEpCombineArgs,
    grid: u32,
}

#[derive(Clone, Copy, Debug)]
enum PrefillSegmentRoute {
    Interpreter,
    SparseMla(amd_sparse_mla::Route),
    MoeAiter(amd_moe_aiter::Route),
    IndexTp(amd_index_tp::Route),
    GemmLt(amd_gemm_lt::Route),
    MlaFold(amd_mla_fold::Route),
    XReduceWaveRs,
    GraphPhaseXReduceWaveRs,
    XReduceAttnRes {
        args: XReduceAttnResArgs,
        device_args: u64,
        grid: u32,
    },
    KdaChunkIntraCached {
        args: KdaChunkIntraCachedArgs,
        grid: u32,
    },
    KdaChunkIntraWaveItems {
        args: KdaChunkIntraCachedArgs,
        grid: u32,
    },
    AttnResF32Mix {
        args: AttnResF32MixArgs,
        grid: u32,
    },
    KdaChunkKeyFactorWu {
        args: KdaChunkKeyFactorWuArgs,
        grid: u32,
    },
    KdaChunkKeyFactorCarry {
        args: KdaChunkKeyFactorCarryArgs,
        grid: u32,
    },
    KdaChunkCarryRegstate {
        args: KdaChunkCarryRegstateArgs,
        grid: u32,
        /// Packet tensor ids of the eight operands, in args order, so a launch can rebase the
        /// KV-slot-strided ones (the recurrent `state`) to the active slot.
        tens: [u16; 8],
    },
    /// Lean four-wave Wu; `args.key_hi != 0` selects the key-emitting object.
    KdaChunkWuLean {
        args: KdaChunkWuLeanArgs,
        grid: u32,
        tens: [u16; 8],
    },
    /// The register-state carry fed with the preceding lean Wu's scaled-key pair.
    KdaChunkCarryKeyfeed {
        args: KdaChunkCarryKeyfeedArgs,
        grid: u32,
        tens: [u16; 8],
    },
    MoeStage1Mxfp4(MoeStage1Mxfp4Route),
    MoeStage1A4Reuse(MoeStage1A4ReuseRoute),
    MoeStage2Mxfp4(MoeStage2Mxfp4Route),
    MoeCombine(MoeCombineRoute),
    MoeEpAlign(MoeEpAlignRoute),
    MoeEpStage2(MoeStage2Mxfp4Route),
    MoeEpCombine(MoeEpCombineRoute),
    MlaMaterializePack {
        args: MlaMaterializePackArgs,
        grid: u32,
        /// Packet tensor id of `K_rope` (the `kv.*.krot` cache), so a launch can rebase it to
        /// the active KV slot; the other three operands are transients.
        k_rope_ten: u16,
        /// Rows the K/V transients can hold — the bound on the per-chunk `kv_len` patch.
        kv_rows_cap: u32,
    },
    MlaMaterializedPrefill {
        args: MlaMaterializedPrefillArgs,
        grid: u32,
    },
}

const MLA_MATERIALIZED_Q_BLOCK: u32 = 256;

fn mla_materialized_flat_grid(n: u32, heads: u32, batches: u32) -> Result<u32> {
    if n == 0 || heads == 0 || batches == 0 {
        return Err(RuntimeError::Device(
            "materialized MLA grid dimensions must be nonzero".into(),
        ));
    }
    n.div_ceil(MLA_MATERIALIZED_Q_BLOCK)
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(batches))
        .ok_or_else(|| RuntimeError::Device("materialized MLA grid overflows".into()))
}

fn mla_materialized_routes(
    prog: &DevProg,
    devp: &[DeviceMem],
    routes: &mut [PrefillSegmentRoute],
) -> Result<()> {
    let mut members = vec![std::collections::BTreeSet::new(); routes.len()];
    let mut raw_segment_owner = vec![None; prog.insts.len()];
    for entry in &prog.stream {
        let seg = entry.seg as usize;
        let inst_ix = entry.inst as usize;
        let inst = prog.insts.get(inst_ix).ok_or_else(|| {
            RuntimeError::Device(format!(
                "materialized MLA stream references instruction {inst_ix} of {}",
                prog.insts.len()
            ))
        })?;
        let set = members.get_mut(seg).ok_or_else(|| {
            RuntimeError::Device(format!(
                "materialized MLA stream references segment {seg} of {}",
                routes.len()
            ))
        })?;
        set.insert(inst_ix);
        if matches!(
            DevOp::from_u16(inst.op),
            Some(DevOp::MlaMaterializePack | DevOp::FlashMlaMaterializedPrefill)
        ) {
            match raw_segment_owner[inst_ix] {
                Some(owner) if owner != seg => {
                    return Err(RuntimeError::Device(format!(
                        "materialized MLA raw instruction {inst_ix} is referenced by segments {owner} and {seg}"
                    )))
                }
                None => raw_segment_owner[inst_ix] = Some(seg),
                _ => {}
            }
            if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
                return Err(RuntimeError::Device(format!(
                    "materialized MLA raw instruction {inst_ix} has interpreter counter obligations in segment {seg}"
                )));
            }
        }
    }
    let addr = |h: u16, what: &str| -> Result<u64> {
        if h == packet::dev::TENSOR_NONE16 {
            return Err(RuntimeError::Device(format!(
                "materialized MLA operand `{what}` is absent"
            )));
        }
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "materialized MLA operand `{what}` handle {h} is outside {} tensors",
                devp.len()
            ))
        })
    };
    for (seg, set) in members.into_iter().enumerate() {
        let raw = set.iter().any(|&i| {
            matches!(
                DevOp::from_u16(prog.insts[i].op),
                Some(DevOp::MlaMaterializePack | DevOp::FlashMlaMaterializedPrefill)
            )
        });
        if !raw {
            continue;
        }
        if set.len() != 1 {
            return Err(RuntimeError::Device(format!(
                "materialized MLA segment {seg} mixes a standalone opcode with interpreter work"
            )));
        }
        let d = &prog.insts[*set.first().unwrap()];
        if d.op == DevOp::MlaMaterializePack as u16 {
            if d.i[..5] != [prog.t, d.i[1], 128, 64, 128]
                || d.i[1] == 0
                || d.blocks == 0
                || d.i[5..].iter().any(|&v| v != 0)
                || d.fj.iter().any(|&v| v != 0)
            {
                return Err(RuntimeError::Device(format!(
                    "materialized MLA pack segment {seg} has unsupported geometry"
                )));
            }
            let rows_of = |h: u16, width: u64| -> u32 {
                let bytes = devp.get(h as usize).map_or(0, |m| m.len);
                u32::try_from(bytes / (u64::from(d.i[1]) * width * 2)).unwrap_or(u32::MAX)
            };
            let kv_rows_cap = rows_of(d.t[0], 192)
                .min(rows_of(d.t[1], 128))
                .min(rows_of(d.t[2], 256));
            if kv_rows_cap < prog.t {
                return Err(RuntimeError::Device(format!(
                    "materialized MLA pack segment {seg}: K/V transients hold {kv_rows_cap} rows, fewer than the bucket's {}",
                    prog.t
                )));
            }
            routes[seg] = PrefillSegmentRoute::MlaMaterializePack {
                args: MlaMaterializePackArgs {
                    k: addr(d.t[0], "K")?,
                    v: addr(d.t[1], "V")?,
                    kv: addr(d.t[2], "KV")?,
                    k_rope: addr(d.t[3], "K_rope")?,
                    t: d.i[0],
                    heads: d.i[1],
                    qk_nope: d.i[2],
                    qk_rope: d.i[3],
                    v_head: d.i[4],
                },
                grid: u32::from(d.blocks),
                k_rope_ten: d.t[3],
                kv_rows_cap,
            };
        } else if d.op == DevOp::FlashMlaMaterializedPrefill as u16 {
            let (n, h, h_kv, d_qk, d_v, abi) = (d.i[0], d.i[1], d.i[2], d.i[3], d.i[4], d.i[5]);
            let scale = f32::from_bits(d.fj[0]);
            if n != prog.t
                || h == 0
                || h_kv != h
                || d_qk != 192
                || d_v != 128
                || abi != 1
                || d.i[6..].iter().any(|&v| v != 0)
                || !scale.is_finite()
                || scale <= 0.0
            {
                return Err(RuntimeError::Device(format!(
                    "materialized MLA prefill segment {seg} has unsupported capability: T={n} H={h} H_KV={h_kv} DQK={d_qk} DV={d_v} abi={abi}"
                )));
            }
            let n = i32::try_from(n).map_err(|_| RuntimeError::Device("T exceeds i32".into()))?;
            let h = i32::try_from(h).map_err(|_| RuntimeError::Device("H exceeds i32".into()))?;
            let qn = h
                .checked_mul(192)
                .ok_or_else(|| RuntimeError::Device("Q stride overflows".into()))?;
            let on = h
                .checked_mul(128)
                .ok_or_else(|| RuntimeError::Device("O stride overflows".into()))?;
            let stride_q_b = n
                .checked_mul(qn)
                .ok_or_else(|| RuntimeError::Device("Q batch stride overflows".into()))?;
            let stride_o_b = n
                .checked_mul(on)
                .ok_or_else(|| RuntimeError::Device("O batch stride overflows".into()))?;
            let grid = mla_materialized_flat_grid(n.unsigned_abs(), h.unsigned_abs(), 1)?;
            routes[seg] = PrefillSegmentRoute::MlaMaterializedPrefill {
                args: MlaMaterializedPrefillArgs {
                    q: addr(d.t[1], "Q")?,
                    k: addr(d.t[2], "K")?,
                    v: addr(d.t[3], "V")?,
                    o: addr(d.t[0], "O")?,
                    b: 1,
                    n,
                    n_kv: n,
                    h,
                    h_kv: h,
                    d_qk: 192,
                    d_v: 128,
                    stride_q_b,
                    stride_q_n: qn,
                    stride_q_h: 192,
                    stride_o_b,
                    stride_o_n: on,
                    stride_o_h: 128,
                    stride_k_b: stride_q_b,
                    stride_k_n: qn,
                    stride_k_h: 192,
                    stride_v_b: stride_o_b,
                    stride_v_n: on,
                    stride_v_h: 128,
                    softmax_scale: scale,
                    seqstart_q: 0,
                    seqstart_k: 0,
                    seqstart_q_pad: 0,
                    seqstart_k_pad: 0,
                    opt: 0,
                    _pad: 0,
                    lse: 0,
                    stride_lse_b: 0,
                    stride_lse_h: 0,
                },
                grid,
            };
        }
    }
    for (i, inst) in prog.insts.iter().enumerate() {
        if matches!(
            DevOp::from_u16(inst.op),
            Some(DevOp::MlaMaterializePack | DevOp::FlashMlaMaterializedPrefill)
        ) && raw_segment_owner[i].is_none()
        {
            return Err(RuntimeError::Device(format!(
                "materialized MLA raw instruction {i} is absent from every stream segment"
            )));
        }
    }
    Ok(())
}

fn xreduce_attnres_encoded(d: &DevInst64) -> bool {
    d.op == DevOp::XReduceTwoShot as u16 && d.t[3] != packet::dev::TENSOR_NONE16
}

fn xreduce_attnres_inst(d: &DevInst64) -> bool {
    xreduce_attnres_encoded(d)
        && d.blocks != 0
        && d.i[0] != 0
        && d.i[1] > 1
        && d.i[5] != 0
        && d.i[0].is_multiple_of(d.i[5])
        && (d.i[0] / d.i[5]).is_multiple_of(d.i[1])
        && d.i[6] <= d.i[7]
        && d.t[0] != packet::dev::TENSOR_NONE16
        && d.t[2..=6].iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && d.t[7] == packet::dev::TENSOR_NONE16
        && f32::from_bits(d.fj[0]).is_finite()
        && f32::from_bits(d.fj[0]) > 0.0
        && d.fj[1] == 0
}

fn moe_stage1_mxfp4_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeGroupGluPf as u16
        && d.i[0] >= 256
        && d.i[0].is_multiple_of(32)
        && d.i[1] != 0
        && d.i[1].is_multiple_of(128)
        && d.i[2] != 0
        && d.i[3] == MOE_ENC_MXFP4
        && d.i[4] == 0
        && d.i[5] <= 2
        && d.i[6] == 0
        && d.i[7] == 0
        && d.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
}

fn moe_stage1_a4_reuse_inst(d: &DevInst64) -> bool {
    moe_stage1_mxfp4_inst(d) && d.i[0].div_ceil(256) >= 2
}

fn moe_stage2_mxfp4_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeGroupDownPf as u16
        && d.i[3] == MOE_ENC_MXFP4
        && d.i[1] == 384
        && d.i[0] != 0
        && d.i[0].is_multiple_of(16)
        && d.i[2] != 0
        && d.i[4] == 0
        && d.i[5] == 0
        && d.t[5] != packet::dev::TENSOR_NONE16
        && d.t[6] != packet::dev::TENSOR_NONE16
        && d.t[7] != packet::dev::TENSOR_NONE16
}

fn moe_combine_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeCombinePf as u16
        && d.t[0] != packet::dev::TENSOR_NONE16
        && d.t[3] != packet::dev::TENSOR_NONE16
        && d.i[0] != 0
        && d.i[1] == 16
        && d.i[2] != 0
        && d.i[3..].iter().all(|&v| v == 0)
        && d.fj.iter().all(|&v| v == 0)
}

fn moe_ep_degree(d: &DevInst64) -> Option<u32> {
    let degree = match DevOp::from_u16(d.op) {
        Some(DevOp::MoeAlignPf | DevOp::MoeCombinePf) => d.i[5],
        Some(DevOp::MoeGroupGluPf | DevOp::MoeGroupDownPf) => d.i[6],
        _ => 0,
    };
    (degree > 1).then_some(degree)
}

fn moe_ep_stage1_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeGroupGluPf as u16
        && d.i[0] >= 256
        && d.i[0].is_multiple_of(128)
        && d.i[1].is_multiple_of(128)
        && d.i[2] >= d.i[6]
        && d.i[3] == MOE_ENC_MXFP4
        && d.i[4] == 0
        && d.i[5] <= 2
        && d.i[6] > 1
        && d.i[7] == 0
        && d.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
}

fn moe_ep_stage2_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeGroupDownPf as u16
        && d.i[0].is_multiple_of(16)
        && d.i[1].is_multiple_of(128)
        && d.i[2] >= d.i[6]
        && d.i[3] == MOE_ENC_MXFP4
        && d.i[4] == 0
        && d.i[5] == 0
        && d.i[6] > 1
        && d.i[7] == 0
        && d.t[5..=7].iter().all(|&t| t != packet::dev::TENSOR_NONE16)
}

fn moe_ep_combine_inst(d: &DevInst64) -> bool {
    d.op == DevOp::MoeCombinePf as u16
        && d.t[0] != packet::dev::TENSOR_NONE16
        && d.t[3] != packet::dev::TENSOR_NONE16
        && d.t[4] != packet::dev::TENSOR_NONE16
        && d.i[0] != 0
        && d.i[1] == 16
        && d.i[2] > 1
        && d.i[3] == 0
        && d.i[4] == 0
        && d.i[5] > 1
        && d.i[6] >= d.i[5]
        && d.i[7] == 0
        && d.fj.iter().all(|&v| v == 0)
}

fn moe_stage2_mxfp4_pair(d: &DevInst64, c: &DevInst64) -> bool {
    moe_stage2_mxfp4_inst(d)
        && c.op == DevOp::MoeCombinePf as u16
        && c.t[1] == packet::dev::TENSOR_NONE16
        && c.t[2] == packet::dev::TENSOR_NONE16
        && c.t[3] == d.t[0]
        && c.i[0] == d.i[0]
        && c.i[1] != 0
        && c.i[2] != 0
        && c.i[3] == 0
        && c.i[4] == 0
        && c.i[7] == 0
}

fn kda_chunk_intra_cached_inst(d: &DevInst64) -> bool {
    d.op == DevOp::KdaChunkIntra as u16
        && d.blocks != 0
        && d.i[0] >= 512
        && d.i[1] != 0
        && d.i[2] == 128
        && d.i[3..].iter().all(|&v| v == 0)
        && d.t[..6].iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && d.t[6..].iter().all(|&t| t == packet::dev::TENSOR_NONE16)
        && f32::from_bits(d.fj[0]).is_finite()
        && f32::from_bits(d.fj[0]) > 0.0
        && d.fj[1..].iter().all(|&v| v == 0)
}

fn kda_key_factor_pair(w: &DevInst64, c: &DevInst64) -> bool {
    w.op == DevOp::KdaChunkWu as u16
        && c.op == DevOp::KdaChunkCarry as u16
        && w.blocks != 0
        && c.blocks != 0
        && w.i[0] >= 512
        && w.i[0] == c.i[0]
        && w.i[1] != 0
        && w.i[1] == c.i[1]
        && w.i[2] == 128
        && w.i[2] == c.i[2]
        && w.i[3] == 128
        && w.i[3] == c.i[3]
        && w.i[4] == 1
        && c.i[4] == 1
        && w.i[5..].iter().all(|&v| v == 0)
        && c.i[5..].iter().all(|&v| v == 0)
        && w.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && c.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && c.t[2] == w.t[7]
        && c.t[3] == w.t[3]
        && c.t[4] == w.t[0]
        && c.t[5] == w.t[1]
        && c.t[7] == w.t[5]
        && w.fj[0] == c.fj[0]
        && f32::from_bits(w.fj[0]).is_finite()
        && f32::from_bits(w.fj[0]) > 0.0
        && w.fj[1..].iter().all(|&v| v == 0)
        && c.fj[1..].iter().all(|&v| v == 0)
}

fn kda_key_factor_segment_pairs(prog: &DevProg) -> Vec<(usize, usize, usize, usize)> {
    let mut members = std::collections::BTreeMap::<u16, std::collections::BTreeSet<usize>>::new();
    for entry in &prog.stream {
        members
            .entry(entry.seg)
            .or_default()
            .insert(entry.inst as usize);
    }
    let singleton_seg = |inst: usize| {
        members
            .iter()
            .find_map(|(&seg, set)| (set.len() == 1 && set.contains(&inst)).then_some(seg as usize))
    };
    prog.insts
        .windows(2)
        .enumerate()
        .filter_map(|(i, pair)| {
            if !kda_key_factor_pair(&pair[0], &pair[1]) {
                return None;
            }
            let wu_seg = singleton_seg(i)?;
            let carry_seg = singleton_seg(i + 1)?;
            (wu_seg != carry_seg).then_some((wu_seg, carry_seg, i, i + 1))
        })
        .collect()
}

fn kda_key_factor_scratch_half_bytes(progs: &[DevProg]) -> Result<u64> {
    let mut max_half = 0u64;
    for prog in progs {
        for (_, _, wu, _) in kda_key_factor_segment_pairs(prog) {
            let d = &prog.insts[wu];
            let bytes = u64::from(d.i[0])
                .checked_mul(u64::from(d.i[1]))
                .and_then(|v| v.checked_mul(u64::from(d.i[2])))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| {
                    RuntimeError::Device("KDA key-factor scratch size overflows".into())
                })?;
            max_half = max_half.max(bytes);
        }
    }
    Ok(max_half)
}

fn add_kda_key_factor_routes(
    prog: &DevProg,
    devp: &[DeviceMem],
    routes: &mut [PrefillSegmentRoute],
    scratch: Option<(u64, u64)>,
) -> Result<()> {
    let Some((key_hi, half_bytes)) = scratch else {
        return Ok(());
    };
    let key_lo = key_hi
        .checked_add(half_bytes)
        .ok_or_else(|| RuntimeError::Device("KDA key-factor scratch address overflows".into()))?;
    let addr = |h: u16, what: &str| -> Result<u64> {
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "KDA key-factor operand `{what}` handle {h} is invalid"
            ))
        })
    };
    for (wu_seg, carry_seg, wi, ci) in kda_key_factor_segment_pairs(prog) {
        let w = &prog.insts[wi];
        let c = &prog.insts[ci];
        let need = u64::from(w.i[0]) * u64::from(w.i[1]) * u64::from(w.i[2]) * 2;
        if need > half_bytes {
            return Err(RuntimeError::Device(format!(
                "KDA key-factor route needs {need} bytes per half, allocation has {half_bytes}"
            )));
        }
        routes[wu_seg] = PrefillSegmentRoute::KdaChunkKeyFactorWu {
            args: KdaChunkKeyFactorWuArgs {
                w: addr(w.t[0], "W")?,
                u: addr(w.t[1], "U")?,
                key_hi,
                key_lo,
                q: addr(w.t[7], "q")?,
                ainv: addr(w.t[2], "Ainv")?,
                k: addr(w.t[3], "k")?,
                v: addr(w.t[4], "v")?,
                g: addr(w.t[5], "g")?,
                beta: addr(w.t[6], "beta")?,
                t: w.i[0],
                heads: w.i[1],
                dim: w.i[2],
                value_dim: w.i[3],
                scale: f32::from_bits(w.fj[0]),
                _pad: 0,
            },
            grid: u32::from(w.blocks),
        };
        routes[carry_seg] = PrefillSegmentRoute::KdaChunkKeyFactorCarry {
            args: KdaChunkKeyFactorCarryArgs {
                out: addr(c.t[0], "out")?,
                state: addr(c.t[1], "state")?,
                q: addr(c.t[2], "q")?,
                k: addr(c.t[3], "k")?,
                key_hi,
                key_lo,
                w: addr(c.t[4], "W")?,
                u: addr(c.t[5], "U")?,
                aqk: addr(c.t[6], "Aqk")?,
                g: addr(c.t[7], "g")?,
                t: c.i[0],
                heads: c.i[1],
                dim: c.i[2],
                value_dim: c.i[3],
                scale: f32::from_bits(c.fj[0]),
                _pad: 0,
            },
            grid: u32::from(c.blocks),
        };
    }
    Ok(())
}

fn rebase_kda_key_factor_routes(routes: &mut [PrefillSegmentRoute], t: u32) {
    for route in routes {
        match route {
            PrefillSegmentRoute::KdaChunkKeyFactorWu { args, .. } => args.t = t,
            PrefillSegmentRoute::KdaChunkKeyFactorCarry { args, .. } => args.t = t,
            PrefillSegmentRoute::KdaChunkCarryRegstate { args, .. } => args.t = t,
            PrefillSegmentRoute::KdaChunkWuLean { args, .. } => args.t = t,
            PrefillSegmentRoute::KdaChunkCarryKeyfeed { args, .. } => args.t = t,
            _ => {}
        }
    }
}

/// Per-chunk operands of the materialized MLA route.
///
/// `rows` is the chunk's query count (the bucket width, or `clen` under RAGGED-M) and
/// `n_kv = c0 + rows` the cached rows its keys span — the same pair `in.kvlen` carries for the
/// absorbed flash. A continuation chunk has no K/V for `[0, c0)` anywhere but the latent cache,
/// so its projection (`kv.*.ckv[0, n_kv) · kv_b`, the GEMM writing `kv_materialized`), the pack
/// (rope half from `kv.*.krot`) and the attention's `N_KV` all address `[0, n_kv)`. The
/// standalone object aligns its causal mask bottom-right (`causal_offset = N_KV - N`), which puts
/// query `i` at absolute position `c0 + i` — exactly `qpos = kv_len - n_tok + t` in
/// `d_flash_mla_prefill`. Runs after [`rebase_chunk_rows`], whose RAGGED-M shrink would otherwise
/// leave the projection at `clen` rows.
fn rebase_mla_materialized_routes(
    insts: &mut [DevInst64],
    names: &[String],
    routes: &mut [PrefillSegmentRoute],
    rows: u32,
    n_kv: u32,
) -> Result<()> {
    if !routes.iter().any(|r| {
        matches!(
            r,
            PrefillSegmentRoute::MlaMaterializePack { .. }
                | PrefillSegmentRoute::MlaMaterializedPrefill { .. }
        )
    }) {
        return Ok(());
    }
    for d in insts.iter_mut() {
        if matches!(prefill_row_field(d.op), Some(RowField::Rows(0)))
            && names
                .get(d.t[0] as usize)
                .is_some_and(|n| n.ends_with(".kv_materialized"))
        {
            d.i[0] = n_kv;
        }
    }
    for route in routes {
        match route {
            PrefillSegmentRoute::MlaMaterializePack {
                args, kv_rows_cap, ..
            } => {
                if n_kv > *kv_rows_cap {
                    return Err(RuntimeError::Device(format!(
                        "materialized MLA chunk spans {n_kv} cached rows, but the K/V transients hold {kv_rows_cap}"
                    )));
                }
                args.t = n_kv;
            }
            PrefillSegmentRoute::MlaMaterializedPrefill { args, grid } => {
                let n = i32::try_from(rows)
                    .map_err(|_| RuntimeError::Device("chunk rows exceed i32".into()))?;
                let nk = i32::try_from(n_kv)
                    .map_err(|_| RuntimeError::Device("kv_len exceeds i32".into()))?;
                args.n = n;
                args.n_kv = nk;
                args.stride_q_b = n * args.stride_q_n;
                args.stride_o_b = n * args.stride_o_n;
                args.stride_k_b = nk * args.stride_k_n;
                args.stride_v_b = nk * args.stride_v_n;
                *grid = mla_materialized_flat_grid(rows, args.h.unsigned_abs(), 1)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn promote_kda_intra_wave_items_routes(
    prog: &DevProg,
    classes: &[u8],
    routes: &mut [PrefillSegmentRoute],
) -> Result<()> {
    for (seg, &class) in classes.iter().enumerate() {
        if class != 20 {
            continue;
        }
        if prog.stream.iter().any(|e| {
            e.seg as usize == seg
                && (e.wait_len != 0
                    || e.succ_len != 0
                    || e.flags & packet::dev::SE_KDA_INTRA_WAVE_ITEMS == 0)
        }) {
            return Err(RuntimeError::Device(format!(
                "KDA-intra wave-item segment {seg} has counter obligations or an unmarked entry"
            )));
        }
        routes[seg] = match routes[seg] {
            PrefillSegmentRoute::KdaChunkIntraCached { args, grid } => {
                PrefillSegmentRoute::KdaChunkIntraWaveItems { args, grid }
            }
            _ => {
                return Err(RuntimeError::Device(format!(
                    "KDA-intra wave-item segment {seg} is not an exact BT64/D128 singleton"
                )))
            }
        };
    }
    Ok(())
}

fn kda_carry_regstate_inst(d: &DevInst64) -> bool {
    d.op == DevOp::KdaChunkCarry as u16
        && d.blocks != 0
        && d.i[0] >= 512
        && d.i[1] != 0
        && d.i[2] == 128
        && d.i[3] == 128
        && d.i[4] == 1
        && d.i[5..].iter().all(|&v| v == 0)
        && d.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && f32::from_bits(d.fj[0]).is_finite()
        && f32::from_bits(d.fj[0]) > 0.0
        && d.fj[1..].iter().all(|&v| v == 0)
}

/// One workgroup per (head, V16) state tile.
fn kda_carry_regstate_grid(heads: u32) -> u32 {
    heads * 8
}

fn promote_kda_carry_regstate_routes(
    prog: &DevProg,
    classes: &[u8],
    devp: &[DeviceMem],
    routes: &mut [PrefillSegmentRoute],
) -> Result<()> {
    let addr = |h: u16, what: &str| -> Result<u64> {
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "KDA carry regstate operand `{what}` handle {h} is invalid"
            ))
        })
    };
    for (seg, &class) in classes.iter().enumerate() {
        if class != 23 {
            continue;
        }
        let mut inst = None;
        for e in prog.stream.iter().filter(|e| e.seg as usize == seg) {
            if e.wait_len != 0
                || e.succ_len != 0
                || e.flags & packet::dev::SE_KDA_CARRY_REGSTATE == 0
                || inst.is_some_and(|i| i != e.inst as usize)
            {
                return Err(RuntimeError::Device(format!(
                    "KDA carry regstate segment {seg} has counter obligations or is not a singleton"
                )));
            }
            inst = Some(e.inst as usize);
        }
        let d = inst
            .and_then(|i| prog.insts.get(i))
            .filter(|d| kda_carry_regstate_inst(d));
        let Some(d) = d else {
            return Err(RuntimeError::Device(format!(
                "KDA carry regstate segment {seg} is not an exact qpre BT64/D128/V128 carry"
            )));
        };
        routes[seg] = PrefillSegmentRoute::KdaChunkCarryRegstate {
            args: KdaChunkCarryRegstateArgs {
                out: addr(d.t[0], "out")?,
                state: addr(d.t[1], "state")?,
                q: addr(d.t[2], "q")?,
                k: addr(d.t[3], "k")?,
                w: addr(d.t[4], "W")?,
                u: addr(d.t[5], "U")?,
                aqk: addr(d.t[6], "Aqk")?,
                g: addr(d.t[7], "g")?,
                t: d.i[0],
                heads: d.i[1],
                dim: d.i[2],
                value_dim: d.i[3],
                scale: f32::from_bits(d.fj[0]),
                _pad: 0,
            },
            grid: kda_carry_regstate_grid(d.i[1]),
            tens: d.t,
        };
    }
    Ok(())
}

fn kda_wu_lean_inst(d: &DevInst64) -> bool {
    d.op == DevOp::KdaChunkWu as u16
        && d.blocks != 0
        && d.i[0] >= 512
        && d.i[1] != 0
        && d.i[2] == 128
        && d.i[3] == 128
        && d.i[4] == 1
        && d.i[5] <= 1
        && d.i[6..].iter().all(|&v| v == 0)
        && d.t.iter().all(|&t| t != packet::dev::TENSOR_NONE16)
        && f32::from_bits(d.fj[0]).is_finite()
        && f32::from_bits(d.fj[0]) > 0.0
        && d.fj[1..].iter().all(|&v| v == 0)
}

/// One four-wave workgroup per (chunk, head) item, at most three per CU.
fn kda_wu_lean_grid(t: u32, heads: u32) -> u32 {
    (t.div_ceil(64) * heads).clamp(1, 768)
}

/// Widest `[T][H][D]` bf16 half of the reusable key-factor scratch pair the keyfeed Wu writes.
fn kda_keyfeed_scratch_half_bytes(progs: &[DevProg]) -> Result<u64> {
    let mut max_half = 0u64;
    for prog in progs {
        for d in &prog.insts {
            if !(kda_wu_lean_inst(d) && d.i[5] == 1) {
                continue;
            }
            let bytes = u64::from(d.i[0])
                .checked_mul(u64::from(d.i[1]))
                .and_then(|v| v.checked_mul(u64::from(d.i[2])))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| RuntimeError::Device("KDA keyfeed scratch size overflows".into()))?;
            max_half = max_half.max(bytes);
        }
    }
    Ok(max_half)
}

/// Marked lean Wu segments; a key-emitting Wu (`i[5] == 1`) also converts the regstate carry
/// route of the instruction that follows it into the key-fed carry. Runs after
/// `promote_kda_carry_regstate_routes`.
fn promote_kda_wu_lean_routes(
    prog: &DevProg,
    classes: &[u8],
    devp: &[DeviceMem],
    routes: &mut [PrefillSegmentRoute],
    scratch: Option<(u64, u64)>,
) -> Result<()> {
    let addr = |h: u16, what: &str| -> Result<u64> {
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "KDA Wu lean operand `{what}` handle {h} is invalid"
            ))
        })
    };
    for (seg, &class) in classes.iter().enumerate() {
        if class != 25 {
            continue;
        }
        let mut inst = None;
        for e in prog.stream.iter().filter(|e| e.seg as usize == seg) {
            if e.wait_len != 0
                || e.succ_len != 0
                || e.flags & packet::dev::SE_KDA_WU_LEAN == 0
                || inst.is_some_and(|i| i != e.inst as usize)
            {
                return Err(RuntimeError::Device(format!(
                    "KDA Wu lean segment {seg} has counter obligations or is not a singleton"
                )));
            }
            inst = Some(e.inst as usize);
        }
        let Some((wi, d)) = inst
            .and_then(|i| prog.insts.get(i).map(|d| (i, d)))
            .filter(|(_, d)| kda_wu_lean_inst(d))
        else {
            return Err(RuntimeError::Device(format!(
                "KDA Wu lean segment {seg} is not an exact qpre BT64/D128/V128 Wu"
            )));
        };
        let keys = d.i[5] == 1;
        let (key_hi, key_lo) = if keys {
            let Some((key_hi, half_bytes)) = scratch else {
                return Err(RuntimeError::Device(format!(
                    "KDA carry keyfeed segment {seg} has no key-factor scratch pair"
                )));
            };
            let need = u64::from(d.i[0]) * u64::from(d.i[1]) * u64::from(d.i[2]) * 2;
            if need > half_bytes {
                return Err(RuntimeError::Device(format!(
                    "KDA carry keyfeed route needs {need} bytes per half, allocation has {half_bytes}"
                )));
            }
            (key_hi, key_hi + half_bytes)
        } else {
            (0, 0)
        };
        routes[seg] = PrefillSegmentRoute::KdaChunkWuLean {
            args: KdaChunkWuLeanArgs {
                w: addr(d.t[0], "W")?,
                u: addr(d.t[1], "U")?,
                q: addr(d.t[7], "q")?,
                key_hi,
                key_lo,
                ainv: addr(d.t[2], "Ainv")?,
                k: addr(d.t[3], "k")?,
                v: addr(d.t[4], "v")?,
                g: addr(d.t[5], "g")?,
                beta: addr(d.t[6], "beta")?,
                t: d.i[0],
                heads: d.i[1],
                dim: d.i[2],
                value_dim: d.i[3],
                scale: f32::from_bits(d.fj[0]),
                _pad: 0,
            },
            grid: kda_wu_lean_grid(d.i[0], d.i[1]),
            tens: [
                d.t[0], d.t[1], d.t[7], d.t[2], d.t[3], d.t[4], d.t[5], d.t[6],
            ],
        };
        if !keys {
            continue;
        }
        let carry_seg = prog
            .stream
            .iter()
            .find(|e| e.inst as usize == wi + 1)
            .map(|e| e.seg as usize);
        let c = prog.insts.get(wi + 1);
        let paired = c.is_some_and(|c| {
            c.t[2] == d.t[7]
                && c.t[3] == d.t[3]
                && c.t[4] == d.t[0]
                && c.t[5] == d.t[1]
                && c.t[7] == d.t[5]
                && c.i[0] == d.i[0]
                && c.i[1] == d.i[1]
        });
        let Some((cs, PrefillSegmentRoute::KdaChunkCarryRegstate { args, grid, tens })) =
            carry_seg.filter(|_| paired).map(|cs| (cs, routes[cs]))
        else {
            return Err(RuntimeError::Device(format!(
                "KDA carry keyfeed Wu segment {seg} is not followed by its paired regstate carry"
            )));
        };
        routes[cs] = PrefillSegmentRoute::KdaChunkCarryKeyfeed {
            args: KdaChunkCarryKeyfeedArgs {
                out: args.out,
                state: args.state,
                q: args.q,
                k: args.k,
                w: args.w,
                u: args.u,
                aqk: args.aqk,
                g: args.g,
                key_hi,
                key_lo,
                t: args.t,
                heads: args.heads,
                dim: args.dim,
                value_dim: args.value_dim,
                scale: args.scale,
                _pad: 0,
            },
            grid,
            tens,
        };
    }
    Ok(())
}

fn has_moe_stage2_mxfp4_segment(prog: &DevProg) -> bool {
    let n_seg = prog
        .stream
        .iter()
        .map(|entry| entry.seg as usize + 1)
        .max()
        .unwrap_or(1);
    let mut members = vec![std::collections::BTreeSet::new(); n_seg];
    for entry in &prog.stream {
        members[entry.seg as usize].insert(entry.inst as usize);
    }
    members.into_iter().any(|set| {
        if set.len() != 1 {
            return false;
        }
        moe_stage2_mxfp4_inst(&prog.insts[*set.first().unwrap()])
    })
}

fn has_moe_stage1_mxfp4_segment(prog: &DevProg) -> bool {
    let mut members = std::collections::BTreeMap::<u16, std::collections::BTreeSet<usize>>::new();
    for entry in &prog.stream {
        members
            .entry(entry.seg)
            .or_default()
            .insert(entry.inst as usize);
    }
    members
        .values()
        .any(|set| set.len() == 1 && moe_stage1_mxfp4_inst(&prog.insts[*set.first().unwrap()]))
}

fn moe_stage1_a4_scratch_bytes(
    progs: &[DevProg],
    tensors: &[crate::asset::devblob::DevTensor],
) -> Result<(u64, u64)> {
    let mut payload = 0u64;
    let mut scales = 0u64;
    for prog in progs {
        for d in &prog.insts {
            if !moe_stage1_a4_reuse_inst(d) {
                continue;
            }
            let row_bytes = tensors
                .get(d.t[5] as usize)
                .ok_or_else(|| {
                    RuntimeError::Device("lean MoE stage-1 row-token handle is invalid".into())
                })?
                .bytes;
            if !row_bytes.is_multiple_of(4) {
                return Err(RuntimeError::Device(
                    "lean MoE stage-1 row-token bytes are not u32-aligned".into(),
                ));
            }
            let rows = row_bytes / 4;
            let next_payload = rows.checked_mul(u64::from(d.i[1] / 2)).ok_or_else(|| {
                RuntimeError::Device("lean MoE stage-1 A4 scratch size overflows".into())
            })?;
            let next_scales = rows.checked_mul(u64::from(d.i[1] / 32)).ok_or_else(|| {
                RuntimeError::Device("lean MoE stage-1 scale scratch size overflows".into())
            })?;
            let next_total = next_payload.checked_add(next_scales).ok_or_else(|| {
                RuntimeError::Device("lean MoE stage-1 total scratch size overflows".into())
            })?;
            if next_total > payload + scales {
                payload = next_payload;
                scales = next_scales;
            }
        }
    }
    Ok((payload, scales))
}

fn has_moe_combine_segment(prog: &DevProg) -> bool {
    let mut members = std::collections::BTreeMap::<u16, std::collections::BTreeSet<usize>>::new();
    for entry in &prog.stream {
        members
            .entry(entry.seg)
            .or_default()
            .insert(entry.inst as usize);
    }
    members
        .values()
        .any(|set| set.len() == 1 && moe_combine_inst(&prog.insts[*set.first().unwrap()]))
}

fn has_moe_prefill_ep(prog: &DevProg) -> bool {
    prog.insts.iter().any(|d| moe_ep_degree(d).is_some())
}

fn moe_prefill_ep_extra_bytes(
    progs: &[DevProg],
    tensors: &[crate::asset::devblob::DevTensor],
    n_gpu: u32,
) -> Result<u64> {
    let mut tables = BTreeSet::new();
    let mut total = 0u64;
    for d in progs.iter().flat_map(|p| &p.insts) {
        if !moe_ep_stage1_inst(d) || !tables.insert(d.t[2]) {
            continue;
        }
        let degree = moe_ep_degree(d).expect("EP stage-1 carries a degree");
        if degree != n_gpu || d.i[3] != MOE_ENC_MXFP4 || d.i[0] % 32 != 0 {
            return Err(RuntimeError::Device(
                "EP memory budget has unsupported geometry".into(),
            ));
        }
        let local_experts = packet::moe_ep::balanced_expert_range(d.i[2], n_gpu, 0).len() as u64;
        let h = u64::from(d.i[1]);
        let i = u64::from(d.i[0]);
        let matrix_payload = h
            .checked_mul(i)
            .and_then(|n| n.checked_div(2))
            .ok_or_else(|| RuntimeError::Device("EP payload budget overflows".into()))?;
        let matrix_scales = h
            .checked_mul(i / 32)
            .ok_or_else(|| RuntimeError::Device("EP scale budget overflows".into()))?;
        let main = matrix_payload
            .checked_add(matrix_scales)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| RuntimeError::Device("EP primary expert budget overflows".into()))?;
        let has_moe2 = tensors.get(d.t[2] as usize).is_some_and(|td| {
            td.name
                .strip_suffix("expert_weight_table_ep")
                .is_some_and(|pfx| {
                    tensors
                        .iter()
                        .any(|t| t.name == format!("{pfx}expert_weight_table_moe2_ep"))
                })
        });
        let moe2 = if has_moe2 {
            let padded_scale = h.div_ceil(256) * 256 * (i.div_ceil(32).div_ceil(8) * 8);
            matrix_payload
                .checked_add(padded_scale)
                .ok_or_else(|| RuntimeError::Device("EP stage-2 budget overflows".into()))?
        } else {
            0
        };
        total = total
            .checked_add(
                local_experts
                    .checked_mul(main + moe2)
                    .ok_or_else(|| RuntimeError::Device("EP resident budget overflows".into()))?,
            )
            .ok_or_else(|| RuntimeError::Device("EP resident budget overflows".into()))?;
    }
    Ok(total)
}

fn moe_mxfp4_routes_with_scratch(
    prog: &DevProg,
    tensors: &[crate::asset::devblob::DevTensor],
    devp: &[DeviceMem],
    stage1_a4_scratch: Option<(u64, u64)>,
    ep_bind: Option<(u32, u32)>,
) -> Result<Vec<PrefillSegmentRoute>> {
    let n_seg = prog
        .stream
        .iter()
        .map(|entry| entry.seg as usize + 1)
        .max()
        .unwrap_or(1);
    let mut members = vec![std::collections::BTreeSet::new(); n_seg];
    for entry in &prog.stream {
        if let Some(set) = members.get_mut(entry.seg as usize) {
            set.insert(entry.inst as usize);
        }
    }
    let addr = |h: u16, what: &str| -> Result<u64> {
        if h == packet::dev::TENSOR_NONE16 {
            return Err(RuntimeError::Device(format!(
                "standalone lean MoE operand `{what}` is absent"
            )));
        }
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "standalone lean MoE operand `{what}` handle {h} is outside {} tensors",
                devp.len()
            ))
        })
    };
    let companion = |h: u16, from: &str, to: &str| -> Result<Option<u64>> {
        let td = tensors.get(h as usize).ok_or_else(|| {
            RuntimeError::Device(format!(
                "lean MoE stage-2 source table handle {h} is outside {} tensors",
                tensors.len()
            ))
        })?;
        let name = if let Some(pfx) = td.name.strip_suffix(&format!("{from}_ep")) {
            format!("{pfx}{to}_ep")
        } else if let Some(pfx) = td.name.strip_suffix(from) {
            format!("{pfx}{to}")
        } else {
            return Ok(None);
        };
        let Some(ix) = tensors.iter().position(|t| t.name == name) else {
            return Ok(None);
        };
        devp.get(ix).map(|m| Some(m.base)).ok_or_else(|| {
            RuntimeError::Device(format!(
                "lean MoE stage-2 companion table `{name}` has no device allocation"
            ))
        })
    };
    let mut routes = Vec::with_capacity(n_seg);
    for (seg, set) in members.into_iter().enumerate() {
        let ep_members: Vec<_> = set
            .iter()
            .filter_map(|&i| {
                prog.insts
                    .get(i)
                    .and_then(|d| moe_ep_degree(d).map(|n| (d, n)))
            })
            .collect();
        if !ep_members.is_empty() {
            let (rank, n_gpu) = ep_bind.ok_or_else(|| {
                RuntimeError::Device(format!(
                    "program T={} segment {seg} declares expert parallelism without a TP binding",
                    prog.t
                ))
            })?;
            if rank >= n_gpu
                || ep_members.len() != set.len()
                || ep_members.iter().any(|(_, degree)| *degree != n_gpu)
            {
                return Err(RuntimeError::Device(format!(
                    "program T={} segment {seg} has a mixed or topology-mismatched EP boundary",
                    prog.t
                )));
            }
            if ep_members
                .iter()
                .all(|(d, _)| d.op == DevOp::MoeAlignPf as u16)
            {
                let first = ep_members[0].0;
                if first.i[0] != prog.t
                    || first.i[1] == 0
                    || first.i[2] != 16
                    || ep_members.iter().any(|(d, _)| {
                        d.t[..5] != first.t[..5]
                            || d.i[0] != first.i[0]
                            || d.i[1] != first.i[1]
                            || d.i[2] != first.i[2]
                    })
                {
                    return Err(RuntimeError::Device(format!(
                        "program T={} segment {seg} has an invalid EP align packet set",
                        prog.t
                    )));
                }
                let range = packet::moe_ep::balanced_expert_range(first.i[1], n_gpu, rank);
                let row_bytes = tensors
                    .get(first.t[2] as usize)
                    .ok_or_else(|| RuntimeError::Device("EP row-token handle is invalid".into()))?
                    .bytes;
                if !row_bytes.is_multiple_of(4) {
                    return Err(RuntimeError::Device(
                        "EP row-token tensor is misaligned".into(),
                    ));
                }
                let meta = addr(first.t[0], "meta")?;
                let meta_words = u64::from(first.i[1])
                    .checked_mul(67)
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(|| RuntimeError::Device("EP metadata size overflows".into()))?;
                let meta_bytes = meta_words.checked_mul(4).ok_or_else(|| {
                    RuntimeError::Device("EP metadata byte size overflows".into())
                })?;
                let declared_meta_bytes = tensors
                    .get(first.t[0] as usize)
                    .ok_or_else(|| RuntimeError::Device("EP metadata handle is invalid".into()))?
                    .bytes;
                if declared_meta_bytes < meta_bytes {
                    return Err(RuntimeError::Device(format!(
                        "EP metadata allocation is {declared_meta_bytes} B; {meta_bytes} B required"
                    )));
                }
                let partial = meta
                    + u64::from(first.i[1])
                        .checked_mul(3)
                        .and_then(|n| n.checked_add(1))
                        .and_then(|n| n.checked_mul(4))
                        .ok_or_else(|| {
                            RuntimeError::Device("EP partial-histogram offset overflows".into())
                        })?;
                routes.push(PrefillSegmentRoute::MoeEpAlign(MoeEpAlignRoute {
                    args: MoeEpAlignArgs {
                        routes: addr(first.t[1], "routes")?,
                        meta,
                        partial,
                        row_token: addr(first.t[2], "row_token")?,
                        row_partidx: addr(first.t[3], "row_partidx")?,
                        row_gate: addr(first.t[4], "row_gate")?,
                        tokens: first.i[0],
                        topk: first.i[2],
                        experts: first.i[1],
                        expert_begin: range.start,
                        expert_end: range.end,
                        row_capacity: u32::try_from(row_bytes / 4).map_err(|_| {
                            RuntimeError::Device("EP row capacity exceeds u32".into())
                        })?,
                        phase: 0,
                        npart: 64,
                    },
                }));
                continue;
            }
            if set.len() != 1 {
                return Err(RuntimeError::Device(format!(
                    "program T={} segment {seg} mixes multiple EP specialist packets",
                    prog.t
                )));
            }
        }
        // An f32-mix AttnRes packet is only ever emitted for the object; a segment that also
        // holds other work cannot be handed to it and must not fall back to the BF16-seam arm.
        if set.len() != 1
            && set
                .iter()
                .any(|&i| prog.insts.get(i).is_some_and(lean_attn_res_f32mix_inst64))
        {
            return Err(RuntimeError::Device(format!(
                "program T={} segment {seg} carries an f32-mix AttnRes packet in a mixed segment",
                prog.t
            )));
        }
        let encoded_xreduce_attnres = set
            .iter()
            .find_map(|&i| prog.insts.get(i).filter(|d| xreduce_attnres_encoded(d)));
        if let Some(d) = encoded_xreduce_attnres {
            if set.len() == 1 && xreduce_attnres_inst(d) {
                // Routed below after resolving its tensor addresses.
            } else {
                return Err(RuntimeError::Device(format!(
                    "program T={} segment {seg} carries an invalid or mixed fused \
                     XReduceTwoShot+AttnRes packet",
                    prog.t
                )));
            }
        }
        if set.len() == 1 {
            let i = *set.first().unwrap();
            let d = &prog.insts[i];
            if moe_ep_stage1_inst(d) {
                let (a4, a4_scale) = stage1_a4_scratch.ok_or_else(|| {
                    RuntimeError::Device(
                        "EP stage-1 requires the reusable A4 quantization scratch".into(),
                    )
                })?;
                let row_bytes = tensors
                    .get(d.t[5] as usize)
                    .ok_or_else(|| RuntimeError::Device("EP row-token handle is invalid".into()))?
                    .bytes;
                let row_capacity = u32::try_from(row_bytes / 4)
                    .map_err(|_| RuntimeError::Device("EP row capacity exceeds u32".into()))?;
                let grid = row_capacity
                    .div_ceil(64)
                    .checked_mul(d.i[0].div_ceil(128))
                    .ok_or_else(|| RuntimeError::Device("EP stage-1 grid overflows".into()))?;
                routes.push(PrefillSegmentRoute::MoeStage1A4Reuse(
                    MoeStage1A4ReuseRoute {
                        quant_args: MoeStage1A4QuantArgs {
                            out: a4,
                            out_scale: a4_scale,
                            activation: addr(d.t[1], "activation")?,
                            row_token: addr(d.t[5], "row_token")?,
                            meta: addr(d.t[4], "meta")?,
                            row_capacity,
                            experts: d.i[2],
                            hidden: d.i[1],
                        },
                        args: MoeStage1A4ReuseArgs {
                            out: addr(d.t[0], "out")?,
                            activation: a4,
                            weight_table: addr(d.t[2], "weight_table")?,
                            activation_scale: a4_scale,
                            weight_scale_table: addr(d.t[3], "weight_scale_table")?,
                            meta: addr(d.t[4], "meta")?,
                            row_partidx: addr(d.t[6], "row_partidx")?,
                            out_scale: addr(d.t[7], "out_scale")?,
                            inter: d.i[0],
                            hidden: d.i[1],
                            experts: d.i[2],
                            act: d.i[5],
                            beta: f32::from_bits(d.fj[0]),
                            linear_beta: f32::from_bits(d.fj[1]),
                        },
                        quant_grid: 1024,
                        grid,
                    },
                ));
                continue;
            }
            if moe_ep_combine_inst(d) {
                let (rank, n_gpu) = ep_bind.expect("EP segment binding checked above");
                let range = packet::moe_ep::balanced_expert_range(d.i[6], n_gpu, rank);
                let grid = d.i[2]
                    .checked_mul(d.i[0].div_ceil(256))
                    .ok_or_else(|| RuntimeError::Device("EP combine grid overflows".into()))?;
                routes.push(PrefillSegmentRoute::MoeEpCombine(MoeEpCombineRoute {
                    args: MoeEpCombineArgs {
                        out: addr(d.t[0], "out")?,
                        part: addr(d.t[3], "part")?,
                        routes: addr(d.t[4], "routes")?,
                        tokens: d.i[2],
                        hidden: d.i[0],
                        topk: d.i[1],
                        expert_begin: range.start,
                        expert_end: range.end,
                    },
                    grid,
                }));
                continue;
            }
            if moe_ep_stage2_inst(d) {
                let weight_table =
                    companion(d.t[2], "expert_weight_table", "expert_weight_table_moe2")?
                        .ok_or_else(|| {
                            RuntimeError::Device(
                                "EP stage-2 requires its shuffled down-weight companion table"
                                    .into(),
                            )
                        })?;
                let weight_scale_table =
                    companion(d.t[3], "expert_scale_table", "expert_scale_table_moe2")?
                        .ok_or_else(|| {
                            RuntimeError::Device(
                                "EP stage-2 requires its shuffled down-scale companion table"
                                    .into(),
                            )
                        })?;
                let rows = tensors
                    .get(d.t[6] as usize)
                    .ok_or_else(|| RuntimeError::Device("EP row-part handle is invalid".into()))?
                    .bytes
                    / 4;
                let grid = u64::from(d.i[0].div_ceil(256))
                    .checked_mul(rows.div_ceil(64) * 2)
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| RuntimeError::Device("EP stage-2 grid overflows".into()))?;
                routes.push(PrefillSegmentRoute::MoeEpStage2(MoeStage2Mxfp4Route {
                    args: MoeStage2Mxfp4Args {
                        part: addr(d.t[0], "part")?,
                        activation: addr(d.t[1], "activation")?,
                        weight_table,
                        activation_scale: addr(d.t[5], "activation_scale")?,
                        weight_scale_table,
                        meta: addr(d.t[4], "meta")?,
                        row_partidx: addr(d.t[6], "row_partidx")?,
                        row_gate: addr(d.t[7], "row_gate")?,
                        model_dim: d.i[0],
                        inter_dim: d.i[1],
                        experts: d.i[2],
                        reserved: 0,
                    },
                    grid,
                }));
                continue;
            }
            if xreduce_attnres_inst(d) {
                routes.push(PrefillSegmentRoute::XReduceAttnRes {
                    args: XReduceAttnResArgs {
                        reduced: addr(d.t[0], "reduced")?,
                        prefix: addr(d.t[6], "prefix")?,
                        out: addr(d.t[2], "out")?,
                        residual: if d.t[1] == packet::dev::TENSOR_NONE16 {
                            0
                        } else {
                            addr(d.t[1], "residual")?
                        },
                        ring: addr(d.t[3], "ring")?,
                        score: addr(d.t[4], "score")?,
                        gamma: addr(d.t[5], "gamma")?,
                        n: d.i[0],
                        slot_bytes: d.i[2],
                        gate_rs: d.i[3],
                        gate_ag: d.i[4],
                        row_w: d.i[5],
                        nb: d.i[6],
                        nbcap: d.i[7],
                        eps: f32::from_bits(d.fj[0]),
                        status: u64::from(d.fj[2]),
                        ..Default::default()
                    },
                    device_args: 0,
                    grid: u32::from(d.blocks),
                });
                continue;
            }
            if lean_attn_res_f32mix_inst64(d) {
                let opt = |h: u16, what: &str| -> Result<u64> {
                    if h == packet::dev::TENSOR_NONE16 {
                        Ok(0)
                    } else {
                        addr(h, what)
                    }
                };
                let res_pre = if d.i[5] == packet::dev::TENSOR_NONE_I {
                    0
                } else {
                    u16::try_from(d.i[5])
                        .ok()
                        .map(|h| addr(h, "res_pre"))
                        .transpose()?
                        .ok_or_else(|| {
                            RuntimeError::Device(format!(
                                "f32-mix AttnRes res_pre handle {} is not a tensor handle",
                                d.i[5]
                            ))
                        })?
                };
                routes.push(PrefillSegmentRoute::AttnResF32Mix {
                    args: AttnResF32MixArgs {
                        out: addr(d.t[0], "out")?,
                        prefix: addr(d.t[1], "prefix")?,
                        ring: addr(d.t[2], "ring")?,
                        score_w: addr(d.t[3], "score_w")?,
                        push_src: opt(d.t[4], "push_src")?,
                        gamma: addr(d.t[5], "gamma")?,
                        res_a: opt(d.t[6], "res_a")?,
                        res_b: opt(d.t[7], "res_b")?,
                        res_pre,
                        t: d.i[0],
                        hid: d.i[1],
                        nb: d.i[2],
                        push_row: d.i[3],
                        nbcap: d.i[4],
                        eps: f32::from_bits(d.fj[0]),
                        out_eps: f32::from_bits(d.fj[1]),
                        reserved: 0,
                    },
                    grid: attn_res_f32mix_grid(d.i[0]),
                });
                continue;
            }
            if kda_chunk_intra_cached_inst(d) {
                routes.push(PrefillSegmentRoute::KdaChunkIntraCached {
                    args: KdaChunkIntraCachedArgs {
                        aqk: addr(d.t[0], "aqk")?,
                        ainv: addr(d.t[1], "ainv")?,
                        q: addr(d.t[2], "q")?,
                        k: addr(d.t[3], "k")?,
                        g_prefix: addr(d.t[4], "g_prefix")?,
                        beta: addr(d.t[5], "beta")?,
                        t: d.i[0],
                        heads: d.i[1],
                        dim: d.i[2],
                        scale: f32::from_bits(d.fj[0]),
                    },
                    grid: u32::from(d.blocks),
                });
                continue;
            }
            if moe_combine_inst(d) {
                routes.push(PrefillSegmentRoute::MoeCombine(MoeCombineRoute {
                    args: MoeCombineArgs {
                        out: addr(d.t[0], "out")?,
                        residual: if d.t[1] == packet::dev::TENSOR_NONE16 {
                            0
                        } else {
                            addr(d.t[1], "residual")?
                        },
                        shared: if d.t[2] == packet::dev::TENSOR_NONE16 {
                            0
                        } else {
                            addr(d.t[2], "shared")?
                        },
                        part: addr(d.t[3], "part")?,
                        hidden: d.i[0],
                        topk: d.i[1],
                        tokens: d.i[2],
                        reserved: 0,
                    },
                    grid: d.i[2].min(512),
                }));
                continue;
            }
            if !moe_stage1_mxfp4_inst(d) {
                if moe_stage2_mxfp4_inst(d) {
                    let Some(c) = prog.insts.get(i + 1) else {
                        routes.push(PrefillSegmentRoute::Interpreter);
                        continue;
                    };
                    if !moe_stage2_mxfp4_pair(d, c) {
                        routes.push(PrefillSegmentRoute::Interpreter);
                        continue;
                    }
                    let Some(weight_table) =
                        companion(d.t[2], "expert_weight_table", "expert_weight_table_moe2")?
                    else {
                        routes.push(PrefillSegmentRoute::Interpreter);
                        continue;
                    };
                    let Some(weight_scale_table) =
                        companion(d.t[3], "expert_scale_table", "expert_scale_table_moe2")?
                    else {
                        routes.push(PrefillSegmentRoute::Interpreter);
                        continue;
                    };
                    let half_tiles = c.i[2]
                        .checked_mul(c.i[1])
                        .and_then(|rows| rows.div_ceil(64).checked_add(d.i[2]))
                        .and_then(|tiles| tiles.checked_mul(2));
                    let grid = d.i[0].div_ceil(256).checked_mul(half_tiles.unwrap_or(0));
                    let Some(grid) = grid.filter(|&grid| grid != 0) else {
                        return Err(RuntimeError::Device(format!(
                            "lean MoE stage-2 segment {seg} launch grid overflows"
                        )));
                    };
                    routes.push(PrefillSegmentRoute::MoeStage2Mxfp4(MoeStage2Mxfp4Route {
                        args: MoeStage2Mxfp4Args {
                            part: addr(d.t[0], "part")?,
                            activation: addr(d.t[1], "activation")?,
                            weight_table,
                            activation_scale: addr(d.t[5], "activation_scale")?,
                            weight_scale_table,
                            meta: addr(d.t[4], "meta")?,
                            row_partidx: addr(d.t[6], "row_partidx")?,
                            row_gate: addr(d.t[7], "row_gate")?,
                            model_dim: d.i[0],
                            inter_dim: d.i[1],
                            experts: d.i[2],
                            reserved: 0,
                        },
                        grid,
                    }));
                    continue;
                }
                routes.push(PrefillSegmentRoute::Interpreter);
                continue;
            }
            let align = prog.insts.iter().find(|a| {
                a.op == DevOp::MoeAlignPf as u16
                    && a.t[0] == d.t[4]
                    && a.i[0] == prog.t
                    && a.i[1] == d.i[2]
                    && a.i[2] == 16
            });
            if align.is_none() {
                return Err(RuntimeError::Device(format!(
                    "segment {seg} isolates a MoE stage-1 packet without its exact T/top-k align producer"
                )));
            }
            if d.blocks == 0 {
                return Err(RuntimeError::Device(format!(
                    "segment {seg} has a zero-grid MoE stage-1 packet"
                )));
            }
            if moe_stage1_a4_reuse_inst(d)
                && crate::config::RuntimeConfig::get().amd.moe_stage1_a4_reuse
            {
                if let Some((a4, a4_scale)) = stage1_a4_scratch {
                    let row_bytes = tensors
                        .get(d.t[5] as usize)
                        .ok_or_else(|| {
                            RuntimeError::Device(
                                "lean MoE stage-1 row-token handle is invalid".into(),
                            )
                        })?
                        .bytes;
                    if !row_bytes.is_multiple_of(4) {
                        return Err(RuntimeError::Device(
                            "lean MoE stage-1 row-token bytes are not u32-aligned".into(),
                        ));
                    }
                    let row_capacity = u32::try_from(row_bytes / 4).map_err(|_| {
                        RuntimeError::Device("lean MoE stage-1 row capacity exceeds u32".into())
                    })?;
                    let weight_table = addr(d.t[2], "expert_weight_table")?;
                    let weight_scale_table = addr(d.t[3], "expert_scale_table")?;
                    let grid = row_capacity
                        .div_ceil(64)
                        .checked_mul(d.i[0].div_ceil(128))
                        .ok_or_else(|| {
                            RuntimeError::Device("lean MoE stage-1 A4 reuse grid overflows".into())
                        })?;
                    routes.push(PrefillSegmentRoute::MoeStage1A4Reuse(
                        MoeStage1A4ReuseRoute {
                            quant_args: MoeStage1A4QuantArgs {
                                out: a4,
                                out_scale: a4_scale,
                                activation: addr(d.t[1], "activation")?,
                                row_token: addr(d.t[5], "row_token")?,
                                meta: addr(d.t[4], "meta")?,
                                row_capacity,
                                experts: d.i[2],
                                hidden: d.i[1],
                            },
                            args: MoeStage1A4ReuseArgs {
                                out: addr(d.t[0], "out")?,
                                activation: a4,
                                weight_table,
                                activation_scale: a4_scale,
                                weight_scale_table,
                                meta: addr(d.t[4], "meta")?,
                                row_partidx: addr(d.t[6], "row_partidx")?,
                                out_scale: addr(d.t[7], "out_scale")?,
                                inter: d.i[0],
                                hidden: d.i[1],
                                experts: d.i[2],
                                act: d.i[5],
                                beta: f32::from_bits(d.fj[0]),
                                linear_beta: f32::from_bits(d.fj[1]),
                            },
                            quant_grid: 1024,
                            grid,
                        },
                    ));
                    continue;
                }
            }
            routes.push(PrefillSegmentRoute::MoeStage1Mxfp4(MoeStage1Mxfp4Route {
                args: MoeStage1Mxfp4Args {
                    out: addr(d.t[0], "out")?,
                    activation: addr(d.t[1], "activation")?,
                    weight_table: addr(d.t[2], "weight_table")?,
                    weight_scale_table: addr(d.t[3], "weight_scale_table")?,
                    meta: addr(d.t[4], "meta")?,
                    row_token: addr(d.t[5], "row_token")?,
                    row_partidx: addr(d.t[6], "row_partidx")?,
                    out_scale: addr(d.t[7], "out_scale")?,
                    inter_dim: d.i[0],
                    model_dim: d.i[1],
                    experts: d.i[2],
                    act: d.i[5],
                    beta: f32::from_bits(d.fj[0]),
                    linear_beta: f32::from_bits(d.fj[1]),
                    reserved: 0,
                    reserved2: 0,
                },
                grid: u32::from(d.blocks),
            }));
            continue;
        }
        routes.push(PrefillSegmentRoute::Interpreter);
    }
    Ok(routes)
}

#[cfg(test)]
fn moe_mxfp4_routes(
    prog: &DevProg,
    tensors: &[crate::asset::devblob::DevTensor],
    devp: &[DeviceMem],
) -> Result<Vec<PrefillSegmentRoute>> {
    moe_mxfp4_routes_with_scratch(prog, tensors, devp, None, None)
}

#[derive(Clone, Copy, Debug)]
enum DecodeSegmentRoute {
    Interpreter,
    MlaAttention,
    KdaDecodeFused(KdaDecodeFusedArgs),
    MoeAiter(amd_moe_aiter::Route),
    GemmLt(amd_gemm_lt::Route),
    GroupedMoeMxfp4 {
        glu: GroupedMoeGluArgs,
        down: GroupedMoeDownArgs,
        grid: u32,
    },
}

fn requires_segmented_decode(routes: &[DecodeSegmentRoute]) -> bool {
    routes
        .iter()
        .any(|route| !matches!(route, DecodeSegmentRoute::Interpreter))
}

fn decode_segment_routes(
    prog: &DevProg,
    tensors: &[crate::asset::devblob::DevTensor],
    init: &[u8],
    devp: &[DeviceMem],
) -> Result<Vec<DecodeSegmentRoute>> {
    let kinds = decode_segment_kinds(prog)?;
    let aiter = amd_moe_aiter::routes(prog, tensors, kinds.len())?;
    let gemm_lt = amd_gemm_lt::routes(prog, tensors, kinds.len())?;
    let mut routes = Vec::with_capacity(kinds.len());
    for (seg, kind) in kinds.into_iter().enumerate() {
        let inst_ix = match kind {
            DecodeSegmentKind::Interpreter => {
                routes.push(DecodeSegmentRoute::Interpreter);
                continue;
            }
            DecodeSegmentKind::MlaAttention => {
                routes.push(DecodeSegmentRoute::MlaAttention);
                continue;
            }
            DecodeSegmentKind::MoeAiter => {
                routes.push(DecodeSegmentRoute::MoeAiter(aiter[seg].ok_or_else(
                    || RuntimeError::Device("native MoE segment has no validated route".into()),
                )?));
                continue;
            }
            DecodeSegmentKind::GemmLt => {
                routes.push(DecodeSegmentRoute::GemmLt(gemm_lt[seg].ok_or_else(
                    || RuntimeError::Device("native GEMM segment has no validated route".into()),
                )?));
                continue;
            }
            DecodeSegmentKind::KdaDecodeFused(inst_ix) => inst_ix,
            DecodeSegmentKind::GroupedMoeMxfp4 { glu, down } => {
                let g = &prog.insts[glu];
                let d = &prog.insts[down];
                if g.i[0] == 0
                    || g.i[1] == 0
                    || g.i[2] == 0
                    || g.i[3] == 0
                    || g.i[6] != 2
                    || d.i[6] != 2
                    || d.i[0] != g.i[0]
                    || d.i[1] != g.i[2]
                    || d.i[2] != g.i[1]
                    || d.i[3] != g.i[3]
                    || d.t[1] != g.t[0]
                    || d.t[2] != g.t[2]
                    || d.blocks == 0
                    || g.blocks == 0
                {
                    return Err(RuntimeError::Device(format!(
                        "grouped MoE instructions {glu}/{down} have unsupported or inconsistent MXFP4 geometry"
                    )));
                }
                let addr = |h: u16, what: &str| -> Result<u64> {
                    devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "grouped MoE operand `{what}` handle {h} is outside {} tensors",
                            devp.len()
                        ))
                    })
                };
                routes.push(DecodeSegmentRoute::GroupedMoeMxfp4 {
                    glu: GroupedMoeGluArgs {
                        fu: addr(g.t[0], "fu")?,
                        x: addr(g.t[1], "x")?,
                        table: addr(g.t[2], "table")?,
                        weights: addr(g.t[3], "weights")?,
                        scales: addr(g.t[4], "scales")?,
                        topk: g.i[0],
                        intermediate: g.i[1],
                        hidden: g.i[2],
                        experts: g.i[3],
                        act: g.i[5],
                        enc: g.i[6],
                        beta: f32::from_bits(g.fj[0]),
                        linear_beta: f32::from_bits(g.fj[1]),
                    },
                    down: GroupedMoeDownArgs {
                        partial: addr(d.t[0], "partial")?,
                        fu: addr(d.t[1], "fu")?,
                        table: addr(d.t[2], "table")?,
                        weights: addr(d.t[3], "weights")?,
                        scales: addr(d.t[4], "scales")?,
                        topk: d.i[0],
                        hidden: d.i[1],
                        intermediate: d.i[2],
                        experts: d.i[3],
                        enc: d.i[6],
                        reserved: 0,
                    },
                    grid: u32::from(g.blocks).saturating_mul(3),
                });
                continue;
            }
        };
        let d = &prog.insts[inst_ix];
        let (rows, heads, dim, bv, conv_w, flags, gate_mode, version) = (
            d.i[0], d.i[1], d.i[2], d.i[3], d.i[4], d.i[5], d.i[6], d.i[7],
        );
        let lower_bound = f32::from_bits(d.fj[1]);
        let scale = f32::from_bits(d.fj[0]);
        let norm_eps = f32::from_bits(d.fj[2]);
        if !matches!(rows, 1 | 8)
            || heads == 0
            || dim != 128
            || bv != 8
            || conv_w != 4
            || flags & 1 == 0
            || flags & !3 != 0
            || (rows > 1 && flags & 2 == 0)
            || gate_mode != 1
            || version != 2
            || d.blocks as u32 != rows.saturating_mul(heads)
            || !lower_bound.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
            || !norm_eps.is_finite()
            || norm_eps <= 0.0
        {
            return Err(RuntimeError::Device(format!(
                "KdaDecodeFused instruction {inst_ix} has unsupported ABI/geometry:                  rows={rows} H={heads} D={dim} BV={bv} W={conv_w} blocks={} flags={flags:#x}                  gate={gate_mode} version={version} lower={lower_bound} scale={scale} eps={norm_eps}",
                d.blocks
            )));
        }
        let desc_h = d.t[7] as usize;
        let desc = tensors.get(desc_h).ok_or_else(|| {
            RuntimeError::Device(format!(
                "KdaDecodeFused descriptor handle {desc_h} is outside {} tensors",
                tensors.len()
            ))
        })?;
        if desc.bytes != 44 {
            return Err(RuntimeError::Device(format!(
                "KdaDecodeFused descriptor `{}` is {} bytes, expected 44",
                desc.name, desc.bytes
            )));
        }
        let range = desc.init.clone().ok_or_else(|| {
            RuntimeError::Device(format!(
                "KdaDecodeFused descriptor `{}` has no initialized handle table",
                desc.name
            ))
        })?;
        let bytes = init.get(range).ok_or_else(|| {
            RuntimeError::Device(format!(
                "KdaDecodeFused descriptor `{}` init range is invalid",
                desc.name
            ))
        })?;
        if bytes.len() != 44 {
            return Err(RuntimeError::Device(format!(
                "KdaDecodeFused descriptor `{}` init is {} bytes, expected 44",
                desc.name,
                bytes.len()
            )));
        }
        let mut handles = [0u32; 11];
        for (slot, word) in handles.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = u32::from_le_bytes(word.try_into().expect("4"));
        }
        let addr = |h: u32, what: &str, optional: bool| -> Result<u64> {
            if h == packet::dev::TENSOR_NONE_I {
                return if optional {
                    Ok(0)
                } else {
                    Err(RuntimeError::Device(format!(
                        "KdaDecodeFused required operand `{what}` is absent"
                    )))
                };
            }
            devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "KdaDecodeFused operand `{what}` handle {h} is outside {} tensors",
                    devp.len()
                ))
            })
        };
        let direct = |slot: usize, what: &str| addr(d.t[slot] as u32, what, false);
        let a = KdaDecodeFusedArgs {
            y: direct(0, "y")?,
            q_raw: direct(1, "q_raw")?,
            k_raw: direct(2, "k_raw")?,
            v_raw: direct(3, "v_raw")?,
            wq: addr(handles[0], "wq", false)?,
            wk: addr(handles[1], "wk", false)?,
            wv: addr(handles[2], "wv", false)?,
            csq: addr(handles[3], "csq", false)?,
            csk: addr(handles[4], "csk", false)?,
            csv: addr(handles[5], "csv", false)?,
            forget_raw: direct(4, "forget_raw")?,
            beta_raw: direct(5, "beta_raw")?,
            output_gate_raw: addr(handles[9], "output_gate_raw", false)?,
            a_log: addr(handles[6], "A_log", false)?,
            dt_bias: addr(handles[7], "dt_bias", false)?,
            state: direct(6, "state")?,
            norm_w: addr(handles[8], "norm_w", false)?,
            parked: addr(handles[10], "parked", true)?,
            rows,
            heads,
            dim,
            bv,
            conv_w,
            flags,
            gate_mode,
            lower_bound,
            scale,
            norm_eps,
        };
        routes.push(DecodeSegmentRoute::KdaDecodeFused(a));
    }
    Ok(routes)
}

/// Sanity bound on `seg`, so a corrupt stream cannot make the host allocate
/// unboundedly. Was 512 — the width of the reference driver's `seg_class[512]`.
///
/// Raised to the full `u16` range because `PLOW_SEG_PER_OP` (see
/// `packet::devbuild::Builder::finish`) emits one segment per op to measure
/// host-side AQL chaining against the counter protocol, and K3's decode program
/// alone is 2459 ops. `seg` is a `u16` in `StreamEnt`, so 65536 is the real
/// representable ceiling and anything under it is arbitrary; the allocations
/// keyed off `n_seg` are `[n_cu][n_seg+1]` window bounds and `[n_seg]` wave
/// classes, i.e. bounded and linear.
const MAX_SEG: u32 = u16::MAX as u32 + 1;

/// Per-segment wave class, derived from the stream.
///
/// A segment is class 4 (the flash interpreter, 256 threads) iff ANY stream
/// entry in it points at a flash-prefill instruction; everything else is class
/// 8. The fp8 twin is included here and is NOT in the reference — omitting it
/// silently ran fp8-KV flash segments on the 8-wave object.
///
/// Under `PLOW_MLA_PF_V2=1`, the bf16 and fp8-KV MLA prefill ops (51 and 110) are class 4 too — but only in
/// programs whose bucket is big enough to fill the machine at the V2 kernel's BQ=64 work
/// decomposition (`t >= 2048`: 256+ items over 304 CUs). Smaller buckets keep the 8-wave
/// kernel, whose BQ=32 fills at half the tokens.
pub fn derive_segments(prog: &DevProg) -> Result<Vec<u8>> {
    derive_segments_for(prog, mla_pf_v2_enabled() && prog.t >= 2048)
}

fn derive_segments_for(prog: &DevProg, v2: bool) -> Result<Vec<u8>> {
    let mut n_seg: u32 = 1;
    for e in &prog.stream {
        let s = e.seg as u32 + 1;
        if s > n_seg {
            n_seg = s;
        }
    }
    if n_seg > MAX_SEG {
        return Err(RuntimeError::Device(format!(
            "program declares {n_seg} segments (max {MAX_SEG}) — corrupt stream?"
        )));
    }
    let mut class = vec![8u8; n_seg as usize];
    // MLA-V2 routing needs PURE segments: the flash object's dispatch skips every op it does
    // not carry, so a segment is only sent there if EVERY entry in it is FlashMlaPrefill —
    // which is exactly what an emit under PLOW_MLA_PF_V2=1 produces. A blob emitted without
    // the split fails the purity test and stays whole on the 8-wave object; a split blob run
    // without the env falls to the t/env guard and likewise runs 8-wave. Either mismatch
    // degrades, never corrupts.
    let mut mla_pure = vec![v2; n_seg as usize];
    let mut needs_v2: Vec<(usize, u32)> = Vec::new();
    let mut mla_any = vec![false; n_seg as usize];
    let mut mla_nope = vec![false; n_seg as usize];
    let mut xr_wave_any = vec![false; n_seg as usize];
    let mut xr_wave_pure = vec![true; n_seg as usize];
    let mut kda_wave_any = vec![false; n_seg as usize];
    let mut kda_wave_pure = vec![true; n_seg as usize];
    let mut kda_carry_pure = vec![true; n_seg as usize];
    let mut kda_wu_pure = vec![true; n_seg as usize];
    for e in &prog.stream {
        let op = prog
            .insts
            .get(e.inst as usize)
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "stream entry references instruction {} of {}",
                    e.inst,
                    prog.insts.len()
                ))
            })?
            .op;
        let seg = e.seg as usize;
        let xr_marked = e.flags & packet::dev::SE_XR_WAVE_RS != 0;
        xr_wave_any[seg] |= xr_marked;
        xr_wave_pure[seg] &= xr_marked && op == DevOp::XReduceTwoShot as u16;
        let kda_wave_marked = e.flags & packet::dev::SE_KDA_INTRA_WAVE_ITEMS != 0;
        kda_wave_any[seg] |= kda_wave_marked;
        kda_wave_pure[seg] &= kda_wave_marked && op == DevOp::KdaChunkIntra as u16;
        kda_carry_pure[seg] &= kda_wave_marked && op == DevOp::KdaChunkCarry as u16;
        kda_wu_pure[seg] &= kda_wave_marked && op == DevOp::KdaChunkWu as u16;
        if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
            class[e.seg as usize] = 4;
        } else if op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16 {
            // A dense causal KV-split packet (i6 = ns, no union table in t7) writes ns
            // partials per (token, head) and the merge reads ns of them. The 8-wave
            // fallback kernel writes the nsplit=1 layout — so the env/routing mismatches
            // that DEGRADE for plain V2 blobs must REFUSE here instead of corrupting.
            let d = &prog.insts[e.inst as usize];
            mla_nope[seg] |= d.i[3] & 0x8000_0000 != 0;
            // DEFERRED to after the class pass, and that is the whole point. This used to
            // refuse on `!v2` -- the ENV -- which misses the case that actually corrupts:
            // PLOW_MLA_PF_V2=1 SET at serve against a blob emitted WITHOUT it. `PLOW_MLA_PF_V2`
            // is read at emit too (`packet/src/devbuild.rs:1123`) and is what splits
            // FlashMlaPrefill into its own wave-class-4 segment; without it at emit the segment
            // is IMPURE, `mla_pure` stays false, `class` never becomes 4, and the ns packet runs
            // on the 8-wave kernel anyway -- while `v2` is true so the old guard stayed silent.
            // The requirement is not "the env is set", it is "THIS packet's segment was actually
            // routed to the V2 arm", which is only known once the class pass below has run.
            if (op == DevOp::FlashMlaPrefill as u16
                && d.i[6] > 1
                && d.t[7] == packet::dev::TENSOR_NONE16)
                || sparse_fp8(d)
            {
                needs_v2.push((e.seg as usize, d.i[6]));
            }
            mla_any[e.seg as usize] = true;
        } else {
            mla_pure[e.seg as usize] = false;
        }
    }
    for s in 0..n_seg as usize {
        if mla_any[s] && mla_pure[s] {
            class[s] = 4;
        }
        // NoPE MLA prefill USED TO BE REFUSED HERE unconditionally: the four-wave V2 kernel had
        // no zero-rope instantiation, so a DR=0 segment routed to class 4 would have read a rope
        // half that does not exist. It now has one (`d_flash_mla_prefill_v2<512, 0>`, the same
        // body with DR=0), so the question is no longer "can this kernel family do it" but "does
        // THIS OBJECT carry the arm" — which is an object fact, not a packet fact, and cannot be
        // decided here where no symbol table is in hand.
        //
        // So the gate moved to load, keyed on `plow_mla_pf2_nope_arm` (check_mla_nope_arm), where
        // an object that predates the arm is refused by name. Deciding it here instead would
        // either refuse every object forever or trust every object blindly. `mla_nope` is still
        // computed above because the load check needs the same predicate from the manifest side.
        let _ = mla_nope[s];
        if xr_wave_any[s] {
            if !xr_wave_pure[s] {
                return Err(RuntimeError::Device(format!(
                    "segment {s} carries SE_XR_WAVE_RS but is not pure XReduceTwoShot"
                )));
            }
            class[s] = 19;
        }
        if kda_wave_any[s] {
            if kda_wave_pure[s] {
                class[s] = 20;
            } else if kda_carry_pure[s] {
                class[s] = 23;
            } else if kda_wu_pure[s] {
                class[s] = 25;
            } else {
                return Err(RuntimeError::Device(format!(
                    "segment {s} carries SE_KDA_INTRA_WAVE_ITEMS but is not pure KdaChunkIntra \
                     (nor SE_KDA_CARRY_REGSTATE / SE_KDA_WU_LEAN on pure KdaChunkCarry / \
                     KdaChunkWu)"
                )));
            }
        }
    }
    for &(seg, ns) in &needs_v2 {
        if !(v2 && class[seg] == 4) {
            return Err(RuntimeError::Device(format!(
                "this packet's FlashMlaPrefill carries a causal KV-split (ns={ns}) which only \
                 the V2 flash arm honors, and V2 routing is NOT live for it: \
                 PLOW_MLA_PF_V2={} at serve, segment {seg} routed to wave class {} (needs 4). \
                 Serving it on the 8-wave kernel would write nsplit=1 partials under an ns-wide \
                 merge -- silently wrong output, not a crash. Either the serve env is unset, or \
                 the BLOB was emitted without PLOW_MLA_PF_V2=1 (it is read at emit too, \
                 packet/src/devbuild.rs, and is what puts FlashMlaPrefill in its own \
                 wave-class-4 segment) so the segment is impure. Re-emit with \
                 PLOW_MLA_PF_V2=1, or re-emit without PLOW_GLM_PF_NS.",
                if v2 { "1" } else { "unset" },
                class[seg]
            )));
        }
    }
    Ok(class)
}

fn derive_packed_segment_families(prog: &DevProg) -> Result<Vec<u8>> {
    let n_seg = prog
        .stream
        .iter()
        .map(|e| e.seg as usize + 1)
        .max()
        .unwrap_or(1);
    let mut family = vec![None; n_seg];
    let mut pure = vec![true; n_seg];
    for e in &prog.stream {
        let op = prog
            .insts
            .get(e.inst as usize)
            .ok_or_else(|| {
                RuntimeError::Device(format!("stream entry references instruction {}", e.inst))
            })?
            .op;
        let next = if op == DevOp::RmsNorm as u16
            || op == DevOp::HeadNormRope as u16
            || op == DevOp::HeadNormRopeFp8 as u16
        {
            Some(5)
        } else if op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16 {
            Some(6)
        } else if op == DevOp::KdaStateStep as u16
            || op == DevOp::KdaConv3 as u16
            || op == DevOp::KdaStateStepG as u16
            || op == DevOp::KdaChunkPrepare as u16
            || op == DevOp::KdaChunkIntra as u16
            || op == DevOp::KdaChunkWu as u16
            || op == DevOp::KdaChunkCarry as u16
        {
            Some(7)
        } else {
            None
        };
        let s = e.seg as usize;
        match (family[s], next) {
            (None, Some(f)) => family[s] = Some(f),
            (Some(a), Some(b)) if a == b => {}
            _ => pure[s] = false,
        }
    }
    Ok(family
        .into_iter()
        .zip(pure)
        .map(|(f, p)| if p { f.unwrap_or(0) } else { 0 })
        .collect())
}

/// Segments eligible for the dedicated gfx950 MLA V2+SV object.
///
/// That object intentionally contains only the dense bf16 opcode. A gathered MLA instruction
/// uses the same opcode but selects another body through `t7`; routing it here would silently
/// execute the dense body. This predicate is deliberately independent of model names.
fn derive_raw_mla_v2_segments(prog: &DevProg) -> Result<Vec<bool>> {
    let n_seg = prog
        .stream
        .iter()
        .map(|e| e.seg as usize + 1)
        .max()
        .unwrap_or(1);
    let mut eligible = vec![prog.t >= 2048; n_seg];
    let mut seen = vec![false; n_seg];
    for e in &prog.stream {
        let inst = prog.insts.get(e.inst as usize).ok_or_else(|| {
            RuntimeError::Device(format!("stream entry references instruction {}", e.inst))
        })?;
        let dense_bf16 = inst.op == DevOp::FlashMlaPrefill as u16
            && inst.t[7] == packet::dev::TENSOR_NONE16
            && inst.i[3] & 0x8000_0000 == 0;
        eligible[e.seg as usize] &= dense_bf16;
        seen[e.seg as usize] |= dense_bf16;
    }
    Ok(eligible
        .into_iter()
        .zip(seen)
        .map(|(eligible, seen)| eligible && seen)
        .collect())
}

fn packed_family_segments_cover(prog: &DevProg, families: &[u8], wanted: &[u8]) -> bool {
    let mut seen = false;
    let covered = prog.stream.iter().all(|e| {
        let Some(in_) = prog.insts.get(e.inst as usize) else {
            return false;
        };
        let expected = if in_.op == DevOp::RmsNorm as u16
            || in_.op == DevOp::HeadNormRope as u16
            || in_.op == DevOp::HeadNormRopeFp8 as u16
        {
            5
        } else if in_.op == DevOp::FlashMlaPrefill as u16
            || in_.op == DevOp::FlashMlaPrefillFp8 as u16
        {
            6
        } else if in_.op == DevOp::KdaStateStep as u16
            || in_.op == DevOp::KdaConv3 as u16
            || in_.op == DevOp::KdaStateStepG as u16
            || in_.op == DevOp::KdaChunkPrepare as u16
            || in_.op == DevOp::KdaChunkIntra as u16
            || in_.op == DevOp::KdaChunkWu as u16
            || in_.op == DevOp::KdaChunkCarry as u16
        {
            7
        } else {
            0
        };
        if expected != 0 && wanted.contains(&expected) {
            seen = true;
            families.get(e.seg as usize).copied() == Some(expected)
        } else {
            true
        }
    });
    seen && covered
}

fn check_packed_dense_program(insts: &[DevInst64]) -> Result<()> {
    for d in insts {
        let op = DevOp::from_u16(d.op);
        if !matches!(
            op,
            Some(
                DevOp::Embed
                    | DevOp::RmsNorm
                    | DevOp::HeadNormRope
                    | DevOp::Gemv
                    | DevOp::Gemm
                    | DevOp::FlashPrefill
                    | DevOp::FlashMerge
                    | DevOp::Glu
                    | DevOp::Residual
                    | DevOp::Argmax
                    | DevOp::ArgmaxFin
                    | DevOp::SoftCap
                    | DevOp::RowRms
                    | DevOp::GemmNorm
                    | DevOp::GemmSmall
                    | DevOp::GemmMed
                    | DevOp::GemmWide
                    | DevOp::GemmGlu
                    | DevOp::NormResidual
                    | DevOp::NormResidualNorm
            )
        ) {
            return Err(RuntimeError::Device(format!(
                "packed dense prefill does not support {op:?}"
            )));
        }
        if (d.op == DevOp::FlashPrefill as u16
            && (d.i[7] == 0 || !matches!(d.i[6], 128 | 256 | 512)))
            || (d.op == DevOp::RmsNorm as u16 && d.i[2] != 0)
        {
            return Err(RuntimeError::Device("packed dense prefill requires direct BF16 attention and row-local RMS normalization".into()));
        }
    }
    Ok(())
}

fn packed_mla_compatible(prog: &DevProg) -> bool {
    !prog.insts.iter().any(|d| {
        sparse_fp8(d)
            || d.op == DevOp::MoeAiterFp8Pf as u16
            || d.op == DevOp::IndexTpPf as u16
            || d.op == DevOp::GemmLtPf as u16
            || amd_mla_fold::native(d)
            || d.op == DevOp::FlashGatherPrefill as u16
            || ((d.op == DevOp::FlashMlaPrefill as u16 || d.op == DevOp::FlashMlaPrefillFp8 as u16)
                && (d.i[3] & 0x8000_0000 != 0
                    || (d.t[7] != packet::dev::TENSOR_NONE16
                        && d.op != DevOp::FlashMlaPrefillFp8 as u16)))
    })
}

fn packed_kda_compatible(prog: &DevProg) -> bool {
    if prog.insts.iter().any(|d| {
        matches!(
            DevOp::from_u16(d.op),
            Some(DevOp::KdaConv | DevOp::KdaConvStateStepG)
        )
    }) {
        return false;
    }
    let count = |op| prog.insts.iter().filter(|d| d.op == op as u16).count();
    let chunks = [
        count(DevOp::KdaChunkPrepare),
        count(DevOp::KdaChunkIntra),
        count(DevOp::KdaChunkWu),
        count(DevOp::KdaChunkCarry),
    ];
    chunks.iter().all(|&n| n == 0) || chunks.iter().all(|&n| n == chunks[0] && n != 0)
}

/// Rows-equivalent cost charged per launch in the chunk DP. Tuned in the
/// reference; `PLOW_LAUNCH_ROWS` overrides. It is what stops the DP from
/// choosing a hundred tiny chunks that each pay a full dispatch.
///
/// **416 is about 4x low and is deliberately left alone.** Measured over five
/// plan pairs on GLM-5.2 gfx942, a launch is ~231 ms and a padded row costs
/// 0.078-0.202 ms with no trend in context, so the honest price is ~1650 rows;
/// `PLOW_LAUNCH_ROWS=1780` is -16% at 1025 tokens and -31% at 3073 and never
/// regressed at any of 14 lengths up to 71808. It is not landed because
/// [`crate::config::AmdConfig::ragged_chunk`] DOMINATES it at every one of those
/// lengths and its output-visible blast radius is a strict subset of ragged's —
/// the reprice is a partial ragged, not an alternative to it, and under ragged
/// this constant is never read (see the early return in [`plan_chunks_cfg`]).
/// Raise this only if ragged-M is ruled out for a reason other than speed.
/// `perf-data/plow-gfx942/glm52-chunk-policy.md`.
const LAUNCH_ROWS: u32 = 416;

/// Cover `n_prompt` tokens with chunks drawn from the compiled bucket ladder.
///
/// Not a fixed chunk size: buckets are a ladder and the DP mixes them, so a
/// 1500-token prompt can be 1024+512 rather than two 1024s with 548 padded rows
/// that cost full compute. Each launch is charged [`LAUNCH_ROWS`] rows so the
/// DP trades padding against dispatch count instead of minimising one alone.
///
/// Returned largest-first, which puts the ragged chunk LAST — the tail is where
/// padding lands, and a padded row writes KV nothing reads.
///
/// Under `PLOW_RAGGED_CHUNK` the trade the DP exists to make DISAPPEARS: the row
/// shrink in [`rebase_chunk_rows`] makes padded rows cost nothing, so the cover
/// is simply the fewest launches, `ceil(n / max_bucket)`. See the branch below.
pub fn plan_chunks(buckets: &[u32], n_prompt: u32) -> Result<Vec<u32>> {
    plan_chunks_capped(buckets, n_prompt, u32::MAX)
}

/// Plan from compiled prefill rungs no wider than `max_bucket`.
pub fn plan_chunks_capped(buckets: &[u32], n_prompt: u32, max_bucket: u32) -> Result<Vec<u32>> {
    let cfg = &crate::config::RuntimeConfig::get().amd;
    let eligible: Vec<u32> = buckets
        .iter()
        .copied()
        .filter(|&b| b > 0 && b <= max_bucket)
        .collect();
    plan_chunks_cfg(
        &eligible,
        n_prompt,
        cfg.launch_rows.unwrap_or(LAUNCH_ROWS),
        cfg.ragged_chunk,
    )
}

/// [`plan_chunks`] with its two policy inputs passed in rather than read from the
/// process-wide [`crate::config::RuntimeConfig`], which is a `OnceLock` and so
/// cannot be toggled by a unit test. Every cover rule is argued here; the public
/// wrapper only supplies the config.
pub fn plan_chunks_cfg(
    buckets: &[u32],
    n_prompt: u32,
    launch_rows: u32,
    ragged: bool,
) -> Result<Vec<u32>> {
    // THE CAP IS THE PACKET'S OWN LADDER, and there is deliberately no second
    // constant here to disagree with it. The widest compiled prefill bucket IS
    // `shapes.max_chunk` in the manifest (`devgen::manifest` defines it as
    // `max(prefill_buckets)`), and the same emit sizes the KV ring from it — so a
    // runtime `MAX_CHUNK` could only ever be a stale copy that silently discards
    // rungs the blob was built to use. It was 8192, and it is what made a packet
    // carrying a 16384 rung serve as if the rung were absent.
    //
    // The `RING >= window + chunk - 1` invariant that constant stood for is
    // enforced where the ring is SIZED, at emit (`devgen::kv_ring`), and it is
    // vacuous for the models that can exceed 8192 today: the MLA family is
    // full-causal (`window = 0`), so `kv_ring` returns `(ctx, MASK_NONE)` and the
    // chunk does not size the cache at all. The generic (windowed) path cannot
    // reach a wider bucket in the first place — its ladder derives from
    // `max_chunk()`, which `MAX_CHUNK_MAX` caps at 8192.
    let mut bkt: Vec<u32> = buckets.iter().copied().filter(|&b| b > 0).collect();
    bkt.sort_unstable();
    bkt.dedup();
    if bkt.is_empty() {
        return Err(RuntimeError::Device(
            "no prefill bucket at or under the max chunk — is this a decode-only blob?".into(),
        ));
    }
    if n_prompt == 0 {
        return Ok(Vec::new());
    }
    // RAGGED-M: the padding is free, so the ONLY thing worth minimising is the
    // launch count, and the minimum is `ceil(n / max_bucket)`. Take the widest
    // bucket while more than one launch is left, then cover the remainder with
    // the SMALLEST bucket that holds it — `rebase_chunk_rows` runs that chunk at
    // its real row count, so the choice of rung costs nothing and only has to be
    // big enough.
    //
    // This is why repricing `LAUNCH_ROWS` is NOT an alternative to the shrink but
    // a consequence of it. Under the padded regime the DP is right to refuse a
    // wider tail: covering 4097 with one 8192 chunk really does cost ~4095 rows
    // of dead compute, which is worse than the second launch. Only once the
    // padding is free does "fewest launches" become the cheapest cover.
    if ragged {
        let max_b = *bkt.last().expect("non-empty");
        let mut out = Vec::new();
        let mut rem = n_prompt;
        while rem > max_b {
            out.push(max_b);
            rem -= max_b;
        }
        if rem > 0 {
            out.push(*bkt.iter().find(|&&b| b >= rem).expect("rem <= max bucket"));
        }
        return Ok(out);
    }
    let quant = bkt[0];
    let rows = n_prompt.div_ceil(quant) as usize;

    // cost[r] = cheapest cover of r quanta; pick[r] = the bucket that achieved it.
    let mut cost = vec![u64::MAX; rows + 1];
    let mut pick = vec![0u32; rows + 1];
    cost[0] = 0;
    for r in 1..=rows {
        for &b in &bkt {
            let step = (b / quant).max(1) as usize;
            let prev = r.saturating_sub(step);
            if cost[prev] == u64::MAX {
                continue;
            }
            let c = cost[prev] + (b + launch_rows) as u64;
            if c < cost[r] {
                cost[r] = c;
                pick[r] = b;
            }
        }
    }
    if cost[rows] == u64::MAX {
        return Err(RuntimeError::Device(format!(
            "no chunk cover for {n_prompt} tokens from buckets {bkt:?}"
        )));
    }
    let mut out = Vec::new();
    let mut r = rows;
    while r > 0 {
        let b = pick[r];
        out.push(b);
        r = r.saturating_sub((b / quant).max(1) as usize);
    }
    // Reconstruction walks backwards, so this is already smallest-last after
    // the reverse — i.e. largest first, ragged chunk at the tail.
    out.sort_unstable_by(|a, b| b.cmp(a));
    Ok(out)
}

// The packet declares the expert POINTER TABLES per layer and NO checkpoint contains them: they
// are tables of DEVICE POINTERS the host computes after the named weights land.
//
// Two families, same reason. `expert_*` are the routed experts, packed by `bind_packed_experts`.
// `dense_*` are the first `first_k_dense_replace` layers' FFN, whose PREFILL runs on the grouped
// expert arms with degenerate 1-expert routing and therefore also reaches its weights only through
// a pointer table (`bind_dense_ffn_tables`).
//
// Missing one there is not a subtle bug: the named-weight loop fails the load with
// `MISSING WEIGHT`, because no checkpoint has a tensor by these names. The predicate is
// `packet::names::is_host_filled_table`, shared with the CUDA loader and the VRAM planner —
// the AMD loader was the only one of the five sites that had it.

/// Pack a MoE model's routed experts and fill its expert POINTER TABLES.
///
/// Fill the DENSE-FFN pointer tables a PREFILL packet declares for its first
/// `first_k_dense_replace` layers (3 on GLM-5.2, 1 on Kimi).
///
/// # Why a dense layer has a pointer table at all
///
/// Its prefill runs on the GROUPED EXPERT arms (ops 85/86) with degenerate
/// 1-expert routing — see `emit_glm_dense_block_prefill` in
/// `crates/devgen/src/mla.rs` and the header of `d_moe_align_pf`. Those ops
/// reach their weights only through `wtab[e*3 + j]`, so even a single "expert"
/// needs the indirection. The reason is a real ISA gap: there is no block-fp8
/// tiled GEMM opcode, and ops 85/86 already are one.
///
/// # Why this is trivial next to [`bind_packed_experts`]
///
/// Nothing is packed and nothing is sliced. Unlike the 256 routed experts, the
/// dense `gate_proj`/`up_proj`/`down_proj` and their `weight_scale_inv` grids
/// ARE declared tensors, so they are already uploaded, already TP-sharded on the
/// right axis by the ordinary named-weight path, and already at a stable device
/// address. All this owes the packet is the three addresses, in the same
/// `{gate, up, down}` order `expert_weight_table` uses.
///
/// A decode-only blob declares no such table and this is a no-op — which is what
/// keeps every existing GLM asset loading unchanged.
fn bind_dense_ffn_tables(
    be: &HsaBackend,
    blob: &DevBlob,
    devp: &[DeviceMem],
    names: &[String],
) -> Result<usize> {
    const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
    let mut filled = 0usize;
    for (i, td) in blob.tensors.iter().enumerate() {
        // `mlp.dense_weight_table` -> weights; `mlp.dense_scale_table` -> the
        // scale twins.
        //
        // TWO scale spellings, tried in this order, because the quantisations
        // that reach this table spell theirs differently and a packet carries
        // whichever its checkpoint had:
        //
        //   `.weight_scale_inv`  block-fp8, an f32 [N/128][K/128] grid (GLM-5.2)
        //   `.weight_scale`      MX microscaling, one E8M0 byte per 32 elements
        //
        // `_inv` is probed FIRST and a block-fp8 packet therefore resolves on the
        // first candidate, to exactly the name this loop used to build — that is
        // what makes the GLM path byte-identical rather than merely equivalent.
        // A packet with neither is still a hard error naming both.
        let (pfx, suffixes): (&str, &[&str]) = match td
            .name
            .strip_suffix("dense_weight_table")
            .map(|p| (p, &[".weight"][..]))
            .or_else(|| {
                td.name
                    .strip_suffix("dense_scale_table")
                    .map(|p| (p, &[".weight_scale_inv", ".weight_scale"][..]))
            }) {
            Some(v) => v,
            None => continue,
        };
        let mut addrs = [0u64; 3];
        for (j, proj) in PROJ.iter().enumerate() {
            let cands: Vec<String> = suffixes.iter().map(|s| format!("{pfx}{proj}{s}")).collect();
            let (want, k) = cands
                .iter()
                .find_map(|w| names.iter().position(|n| n == w).map(|k| (w, k)))
                .ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "dense-FFN prefill table `{}` needs one of {cands:?}, none of which \
                         the packet declares. The table and the three projections are emitted \
                         together by declare_glm_rows; a packet with one and not the other is \
                         malformed.",
                        td.name
                    ))
                })?;
            // A zero base would be read by the kernel as the EP "not my expert"
            // sentinel and the tile would be silently skipped, so an unbound
            // weight must fail here rather than produce a layer that computes
            // nothing.
            if devp[k].base == 0 {
                return Err(RuntimeError::Device(format!(
                    "`{want}` has no device allocation; the grouped arm reads a null weight base \
                     as the 'not my expert' sentinel and would skip the whole layer"
                )));
            }
            addrs[j] = devp[k].base;
        }
        let bytes: Vec<u8> = addrs.iter().flat_map(|a| a.to_le_bytes()).collect();
        EngineDevice::upload(be, &devp[i], 0, &bytes)?;
        filled += 1;
    }
    Ok(filled)
}

/// GLM-5.2 (and DeepSeek-shaped models generally) keep every expert as its own
/// checkpoint tensor — 256 x {gate, up, down} x {weight, block scale} per layer,
/// 115k lookups a rank. `devgen` deliberately declares none of them
/// (`mla.rs`: "declaring 75*256*6 handles would bloat the tensor table for zero
/// emit benefit"): the ops only ever reach an expert through
/// `expert_weight_table[eid*3 + j]`. So the host owes the packet, per MoE layer,
/// ONE contiguous weight buffer + ONE scale buffer holding this rank's experts,
/// and the two tables of addresses into them.
///
/// Ported from `runtime/tests/glm52_decode.c` (the TP4 decode this is checked
/// against), with the per-projection slicing delegated to
/// [`crate::asset::shard`] rather than re-derived — the expert `gate`/`up`/
/// `down` shard on exactly the same axes as their dense counterparts, and the
/// classifier already keys on those names.
///
/// # TP vs EP is read off the packet, not off a flag
///
/// The gate/up op carries the `I_moe` it will stream per expert: the FULL
/// `moe_intermediate_size` under expert-parallel (whole experts, `E/N` of them
/// per rank) and `moe_intermediate_size / n_gpu` under tensor-parallel (a slice
/// of every expert). Against the checkpoint's own expert shape that decides
/// which layout to build, so a packet emitted `--ep` cannot be bound as if it
/// were TP. It matters that this is not a host-side flag: the mismatch has no
/// symptom other than wrong tokens.
///
/// The same instruction is found by OPCODE, not by the name of the tensor it
/// writes — see the note at `GLU_ARMS`.
///
/// # The expert NAME is read off the checkpoint, for the same reason
///
/// Three checkpoints reach this function and none of them spells an expert the
/// way the other two do (see [`ExpertNames`]). [`resolve_expert_names`] probes
/// for the one that is there and [`check_expert_geometry`] then makes the
/// weight and its scale agree with each other, so a checkpoint whose scale is
/// the wrong SIZE fails by name rather than by producing wrong numbers of the
/// right length.
#[allow(clippy::too_many_arguments)]
/// Where the weight-load wall clock actually goes.
///
/// There was NO load timing in this engine: it logged GiB and a tensor count and
/// never a second, so every claim about load cost came from a runbook rather
/// than from the runtime. The phases below are the four that can each be the
/// whole cost depending on the machine and the page-cache state, and which one
/// dominates decides which optimisation is worth anything:
///
/// * `fault` — first touch of the checkpoint mmap. Zero on a warm cache;
///   NVMe-bound on a genuinely cold one, in which case nothing else matters.
/// * `gather` — [`crate::asset::shard::slice_for`]. Free (a borrow) for a
///   replicated or column shard, a full row-by-row copy for a row shard.
/// * `memcpy` — page cache → pinned staging slab, on one core.
/// * `dma` — `hsa_amd_memory_async_copy` submit + signal wait.
///
/// `Cell`, not atomics: one rank's load is one thread, and the counters are
/// touched once per 64 MiB chunk, so this must not appear in any profile.
#[derive(Default)]
struct LoadProf {
    fault_ns: Cell<u64>,
    gather_ns: Cell<u64>,
    memcpy_ns: Cell<u64>,
    dma_ns: Cell<u64>,
    alloc_ns: Cell<u64>,
    memset_ns: Cell<u64>,
    /// Chunks pushed through the staging slab — the signal round-trip count.
    chunks: Cell<u64>,
}

impl LoadProf {
    fn add(c: &Cell<u64>, t: Instant) {
        c.set(c.get() + t.elapsed().as_nanos() as u64);
    }

    /// One `Instant::now()` reused as both "stop A" and "start B".
    fn split(c: &Cell<u64>, t: Instant) -> Instant {
        let now = Instant::now();
        c.set(c.get() + now.duration_since(t).as_nanos() as u64);
        now
    }

    fn ms(c: &Cell<u64>) -> f64 {
        c.get() as f64 / 1e6
    }

    fn report(&self, what: &str, wall: std::time::Duration, bytes: u64) {
        let gib = bytes as f64 / (1u64 << 30) as f64;
        let s = wall.as_secs_f64();
        tracing::info!(
            phase = what,
            wall_s = format!("{s:.2}").as_str(),
            gib = format!("{gib:.2}").as_str(),
            gib_s = format!("{:.2}", if s > 0.0 { gib / s } else { 0.0 }).as_str(),
            fault_ms = format!("{:.0}", Self::ms(&self.fault_ns)).as_str(),
            gather_ms = format!("{:.0}", Self::ms(&self.gather_ns)).as_str(),
            memcpy_ms = format!("{:.0}", Self::ms(&self.memcpy_ns)).as_str(),
            dma_ms = format!("{:.0}", Self::ms(&self.dma_ns)).as_str(),
            alloc_ms = format!("{:.0}", Self::ms(&self.alloc_ns)).as_str(),
            memset_ms = format!("{:.0}", Self::ms(&self.memset_ns)).as_str(),
            chunks = self.chunks.get(),
            "LOAD PHASES"
        );
    }
}

/// Is the mmap-fault phase being measured separately?
///
/// A page fault on an mmap'd checkpoint is charged to whoever touches the page
/// FIRST, and here that is the `copy_from_slice` into the staging slab — so on a
/// cold cache `memcpy` silently contains all of the NVMe time and the breakdown
/// says nothing. Reading one byte per page first moves that cost onto its own
/// counter. It is off by default because it is a second pass over the source
/// (cheap when warm, but not free) and this is the load path, not a benchmark.
fn profile_faults() -> bool {
    crate::config::RuntimeConfig::get().load_profile
}

/// Touch one byte of every page so the fault cost lands on `fault_ns` rather
/// than hiding inside the staging memcpy. Returns without reading anything the
/// caller can observe; `read_volatile` is what stops LLVM deleting it.
///
/// Must be handed the bytes the rank will ACTUALLY read ([`touched`]), never the
/// whole tensor. A column-parallel rank binds a contiguous 1/tp slice, so
/// prefaulting the whole thing pulls four times the bytes off the drive and the
/// profile then describes a load nobody runs — a mistake this made once, and one
/// that flattered nothing: it inflated the very phase it was there to measure.
fn prefault(src: &[u8], prof: &LoadProf) {
    let ns = prefault_ns(src);
    prof.fault_ns.set(prof.fault_ns.get() + ns);
}

/// [`prefault`] without the profile handle, for gather workers that cannot
/// touch the `Cell` counters; the caller folds the returned ns in.
fn prefault_ns(src: &[u8]) -> u64 {
    let t = Instant::now();
    let mut acc = 0u8;
    let mut i = 0usize;
    while i < src.len() {
        // SAFETY: `i < src.len()`, and a `u8` read is always aligned.
        acc ^= unsafe { std::ptr::read_volatile(src.as_ptr().add(i)) };
        i += 4096;
    }
    std::hint::black_box(acc);
    t.elapsed().as_nanos() as u64
}

use crate::asset::checkpoint::{prefetch_depth, prefetch_threads};

/// The bytes rank `rank` will actually TOUCH for `name`, as a queueable span.
///
/// Not the whole tensor: a column-parallel rank binds one contiguous 1/tp range
/// and faulting all of it in would read four times what that rank needs. The
/// range comes from [`crate::asset::shard::slice_for`] itself — the same
/// function that will do the real bind — so it cannot drift from the real read.
///
/// The one case that must NOT go through `slice_for` is a row-parallel gather:
/// it returns `Cow::Owned`, and doing the gather twice would cost more than the
/// prefetch saves. A row gather is strided over every row, so the bytes it
/// touches ARE the whole tensor, which is what this returns for it.
fn touched<'a>(
    ckpt: &'a crate::asset::checkpoint::Checkpoint,
    name: &str,
    want: u64,
    rank: u32,
    n_gpu: u32,
) -> Option<&'a [u8]> {
    let (src, shape) = ckpt.tensor_ex(name)?;
    let strided = n_gpu > 1
        && shape.len() == 2
        && crate::asset::shard::shard_of(name) == crate::asset::shard::Shard::Row;
    if strided {
        return Some(src);
    }
    match crate::asset::shard::slice_for(name, src, shape, want, rank, n_gpu) {
        Ok(std::borrow::Cow::Borrowed(s)) => Some(s),
        // An `Owned` here would mean the row guard above missed a case; the
        // gather has already happened, so fall back to the whole tensor rather
        // than pretend the range is known.
        _ => Some(src),
    }
}

/// [`touched`], resolved into a span the prefetch pool can carry to a thread.
fn weight_span(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    name: &str,
    want: u64,
    rank: u32,
    n_gpu: u32,
) -> Option<crate::asset::checkpoint::Span> {
    let (src, _) = ckpt.tensor_ex(name)?;
    let span = touched(ckpt, name, want, rank, n_gpu)?;
    let off = span.as_ptr() as usize - src.as_ptr() as usize;
    ckpt.span(name, off, span.len())
}

/// How ONE routed expert is spelled in the checkpoint on disk.
///
/// Three spellings reach this loader and they disagree on all four axes:
///
/// | checkpoint | sub-namespace | projections | payload | scale |
/// |---|---|---|---|---|
/// | GLM-5.2 / DeepSeek block-fp8 | `…mlp.` | `gate_proj`/`up_proj`/`down_proj` | `.weight` | `.weight_scale_inv` |
/// | Kimi-K2.7-Code MXFP4 | `…mlp.` | the same three | `.weight` | `.weight_scale` |
/// | Kimi-K3 (compressed-tensors mxfp4) | `…block_sparse_moe.` | `w1`/`w3`/`w2` | `.weight_packed` | `.weight_scale` |
///
/// The middle row is the reason this is RESOLVED and not switched on a flag: a
/// K2.7 checkpoint is the standard projection names with an E8M0 scale, so
/// "mxfp4" and "Mixtral-spelled" are independent facts and no single boolean
/// carries both. A flag that disagrees with the bytes is the failure this file
/// keeps finding; the bytes are the only thing that cannot disagree with itself.
///
/// `proj` is in `expert_weight_table` slot order — gate, up, down — which is why
/// the Mixtral row reads `w1`/`w3`/`w2` and not `w1`/`w2`/`w3`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpertNames {
    /// Everything up to and including `experts.`; an expert index follows.
    ns: String,
    /// gate, up, down.
    proj: [&'static str; 3],
    /// `.weight` or `.weight_packed`.
    payload: &'static str,
    /// `.weight_scale_inv` (block-fp8 f32 grid) or `.weight_scale` (E8M0 row).
    scale: &'static str,
}

impl ExpertNames {
    fn weight_of(&self, e: u32, j: usize) -> String {
        format!("{}{e}.{}{}", self.ns, self.proj[j], self.payload)
    }

    fn scale_of(&self, e: u32, j: usize) -> String {
        format!("{}{e}.{}{}", self.ns, self.proj[j], self.scale)
    }

    /// Is the scale an MX microscaling row (one E8M0 byte per 32 elements along
    /// K) rather than a block-fp8 `[N/128][K/128]` f32 grid?
    fn microscaled(&self) -> bool {
        self.scale == ".weight_scale"
    }
}

/// Which spelling THIS checkpoint uses, decided by probing it.
///
/// `pfx` is what is left of the packet's `…expert_weight_table` after the suffix
/// is stripped, and it is not always a checkpoint prefix: the GLM emitter
/// declares the table under the model prefix (`model.layers.{l}.mlp.`), the K3
/// emitter under its own `moe.` namespace (`moe.language_model.model.layers.{l}.`)
/// because `packet::names` classifies compiler-owned tensors by that prefix. So
/// `moe.` is stripped and the MoE sub-namespace is probed rather than assumed.
///
/// ORDER IS THE COMPATIBILITY GUARANTEE. The first candidate is `{pfx}experts.0.
/// gate_proj.weight` + `.weight_scale_inv` — character for character the two
/// names this function replaced hardcoded — so a block-fp8 packet resolves on
/// probe one and every name built downstream is the name it was built before.
fn resolve_expert_names(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    pfx: &str,
) -> Result<ExpertNames> {
    const TEMPLATES: [([&str; 3], &str); 2] = [
        (["gate_proj", "up_proj", "down_proj"], ".weight"),
        (["w1", "w3", "w2"], ".weight_packed"),
    ];
    const SCALES: [&str; 2] = [".weight_scale_inv", ".weight_scale"];
    let base = pfx.strip_prefix("moe.").unwrap_or(pfx);
    let mut tried: Vec<String> = Vec::new();
    for sub in ["", "mlp.", "block_sparse_moe."] {
        for (proj, payload) in TEMPLATES {
            let ns = format!("{base}{sub}experts.");
            let probe = format!("{ns}0.{}{payload}", proj[0]);
            if ckpt.tensor_ex(&probe).is_none() {
                tried.push(probe);
                continue;
            }
            // The payload is there, so this IS the layout — a missing scale is
            // now a broken checkpoint and not a wrong guess, and saying so beats
            // falling through to a spelling that cannot be right.
            for scale in SCALES {
                if ckpt
                    .tensor_ex(&format!("{ns}0.{}{scale}", proj[0]))
                    .is_some()
                {
                    return Ok(ExpertNames {
                        ns,
                        proj,
                        payload,
                        scale,
                    });
                }
            }
            return Err(RuntimeError::Device(format!(
                "MISSING EXPERT SCALE: `{probe}` is in the checkpoint but neither \
                 `{ns}0.{}{}` nor `{ns}0.{}{}` is. A quantized expert without its scale \
                 cannot be dequantized, and binding the payload alone would decode from \
                 4-bit or 8-bit mantissas read as if they were already scaled.",
                proj[0], SCALES[0], proj[0], SCALES[1]
            )));
        }
    }
    Err(RuntimeError::Device(format!(
        "MISSING EXPERT WEIGHT: the packet declares `{pfx}expert_weight_table` but the \
         checkpoint has no routed experts under any spelling this loader knows. Probed: \
         {tried:?}"
    )))
}

/// Fail unless expert 0's three scale twins are the right SIZE for the weights
/// they scale.
///
/// Every expert in a layer is the same shape, and `slice_for` re-checks each one
/// against the stride derived here — so this is the only place the WEIGHT and its
/// SCALE are compared to each other at all. Getting it wrong is silent in the
/// worst way: an E8M0 row and a block-fp8 grid can be the same number of bytes
/// for some geometries, so a size that merely "looks plausible" is exactly the
/// thing that must not be accepted.
fn check_expert_geometry(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    n: &ExpertNames,
) -> Result<()> {
    let miss = |name: &str| {
        RuntimeError::Device(format!(
            "MISSING EXPERT WEIGHT: {name} (expert 0 resolved to the `{}` + `{}` layout \
             under `{}`, so every projection must be present in it)",
            n.payload, n.scale, n.ns
        ))
    };
    for j in 0..3 {
        let (wn, sn) = (n.weight_of(0, j), n.scale_of(0, j));
        let (w, ws) = ckpt.tensor_ex(&wn).ok_or_else(|| miss(&wn))?;
        let (s, ss) = ckpt.tensor_ex(&sn).ok_or_else(|| miss(&sn))?;
        let bad = |m: String| {
            Err(RuntimeError::Device(format!(
                "EXPERT SCALE GEOMETRY: `{sn}` {ss:?} ({} B) cannot be the scale of `{wn}` \
                 {ws:?} ({} B): {m}",
                s.len(),
                w.len()
            )))
        };
        if ws.len() != 2 || ss.len() != 2 {
            return bad(
                "both must be 2-D — a routed expert is a matrix and its scale is \
                        a grid or a per-group row, never a vector"
                    .into(),
            );
        }
        let (wn0, wn1, sn0, sn1) = (ws[0], ws[1], ss[0], ss[1]);
        if n.microscaled() {
            // MX: payload is [N, K/2] (two fp4 per byte), scale is [N, K/32]
            // (one E8M0 byte per group of 32 along K). Both are u8, so the byte
            // count IS the element count.
            if w.len() != wn0 * wn1 || s.len() != sn0 * sn1 {
                return bad("an mxfp4 payload and its E8M0 scale are both u8, so each \
                            must be exactly the product of its shape"
                    .into());
            }
            if sn0 != wn0 {
                return bad(format!("the output dim disagrees: {wn0} vs {sn0}"));
            }
            if wn1 * 2 != sn1 * 32 {
                return bad(format!(
                    "K disagrees: the payload packs {} elements per row, the scale covers {}",
                    wn1 * 2,
                    sn1 * 32
                ));
            }
        } else {
            // Block-fp8: payload is [N, K] e4m3 (1 B/element), scale is
            // [ceil(N/128), ceil(K/128)] f32. Verified against
            // zai-org/GLM-5.2-FP8: [2048, 6144] -> [16, 48].
            const B: usize = 128;
            if w.len() != wn0 * wn1 {
                return bad("an fp8 e4m3 payload is 1 B/element, so it must be exactly \
                            the product of its shape"
                    .into());
            }
            let (gn, gk) = (wn0.div_ceil(B), wn1.div_ceil(B));
            if (sn0, sn1) != (gn, gk) || s.len() != gn * gk * 4 {
                return bad(format!(
                    "a block-fp8 scale grid must be [{gn}, {gk}] f32 ({} B)",
                    gn * gk * 4
                ));
            }
        }
    }
    Ok(())
}

/// One expert plan entry gathered on a worker, minus the ring push.
///
/// `data` borrows the checkpoint mmap for a column/replicated shard and owns a
/// `Vec` for a row gather; either way the bytes are exactly what the old
/// sequential loop handed the ring for this destination.
struct GatheredExpert<'a> {
    dst: u64,
    want: u64,
    scrub: bool,
    data: std::borrow::Cow<'a, [u8]>,
    /// (device dst, permuted payload) when the preshuffled pf slab is declared.
    pf: Option<(u64, Vec<u8>)>,
    /// Lean stage-2 companion payload or scale, already in the object's exact layout.
    moe2: Option<(u64, Vec<u8>)>,
    gather_ns: u64,
    fault_ns: u64,
}

pub(super) fn shuffle_moe_weight_16x32(src: &[u8], rows: usize, kbytes: usize) -> Result<Vec<u8>> {
    if rows == 0 || !rows.is_multiple_of(16) || kbytes == 0 || !kbytes.is_multiple_of(32) {
        return Err(RuntimeError::Device(format!(
            "MoE 16x32 weight layout requires rows%16=0 and Kbytes%32=0, got {rows}x{kbytes}"
        )));
    }
    if src.len() != rows.saturating_mul(kbytes) {
        return Err(RuntimeError::Device(format!(
            "MoE 16x32 weight layout got {} B for {rows}x{kbytes}",
            src.len()
        )));
    }
    let mut dst = vec![0u8; src.len()];
    let mut at = 0usize;
    for nb in 0..rows / 16 {
        for kb in 0..kbytes / 32 {
            for kh in 0..2 {
                for nr in 0..16 {
                    let s = (nb * 16 + nr) * kbytes + kb * 32 + kh * 16;
                    dst[at..at + 16].copy_from_slice(&src[s..s + 16]);
                    at += 16;
                }
            }
        }
    }
    Ok(dst)
}

pub(super) fn resident_moe_scales(src: &[u8]) -> Result<Vec<u8>> {
    if src.len() != 96 * 4 {
        return Err(RuntimeError::Device("resident MoE requires 96 f32 scales per projection".into()));
    }
    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let scaled = 2.0 * f32::from_le_bytes(chunk.try_into().unwrap());
        if !scaled.is_finite() {
            return Err(RuntimeError::Device("resident MoE scale overflows or is not finite".into()));
        }
        out.extend_from_slice(&scaled.to_le_bytes());
    }
    Ok(out)
}

fn shuffle_mxfp4_moe2_scale(src: &[u8], rows: usize, groups: usize) -> Result<Vec<u8>> {
    if rows == 0 || groups == 0 || src.len() != rows.saturating_mul(groups) {
        return Err(RuntimeError::Device(format!(
            "lean MoE stage-2 scale layout got {} B for {rows}x{groups}",
            src.len()
        )));
    }
    let padded_rows = rows.div_ceil(256) * 256;
    let padded_groups = groups.div_ceil(8) * 8;
    let mut dst = vec![127u8; padded_rows * padded_groups];
    let mut at = 0usize;
    for nb in 0..padded_rows / 32 {
        for gb in 0..padded_groups / 8 {
            for gi in 0..4 {
                for nr in 0..16 {
                    for gh in 0..2 {
                        for nh in 0..2 {
                            let row = nb * 32 + nh * 16 + nr;
                            let group = gb * 8 + gh * 4 + gi;
                            dst[at] = if row < rows && group < groups {
                                src[row * groups + group]
                            } else {
                                127
                            };
                            at += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(dst)
}

/// Expert-gather workers per rank. Every rank of a TP group loads at once, so
/// the box runs `n_gpu *` this many gather threads; the clamp keeps one rank
/// from claiming the whole socket while still passing the ~16-thread knee of
/// [`crate::asset::checkpoint::Checkpoint::populate`]'s scaling table.
fn expert_gather_threads(n_gpu: u32) -> usize {
    let cores = std::thread::available_parallelism().map_or(16, |n| n.get());
    (cores / n_gpu.max(1) as usize).clamp(8, 16)
}

/// The host-side work for one plan entry: fault the span in, slice this rank's
/// shard, optionally preshuffle. Pure reads of the checkpoint — safe from any
/// worker — and all device interaction stays with the caller.
fn gather_expert_entry<'a>(
    ckpt: &'a crate::asset::checkpoint::Checkpoint,
    entry: &(String, u64, u64, bool, u64, u64, u64, u64, u64, u64, bool),
    shard_rank: u32,
    shard_n: u32,
    populate: bool,
    do_prefault: bool,
    resident: bool,
) -> Result<GatheredExpert<'a>> {
    let (name, dst, want, scrub, pf_dst, pf_rows, pf_k, moe2_dst, moe2_rows, moe2_k, moe2_scale) =
        entry;
    let (src, shape) = ckpt
        .tensor_ex(name)
        .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {name}")))?;
    if populate {
        if let Some(s) = weight_span(ckpt, name, *want, shard_rank, shard_n) {
            ckpt.populate(s);
        }
    }
    let mut fault_ns = 0u64;
    if do_prefault {
        if let Some(s) = touched(ckpt, name, *want, shard_rank, shard_n) {
            fault_ns = prefault_ns(s);
        }
    }
    // `tp = 1` is the EP/single-GPU case: bind the expert whole. Otherwise the
    // classifier sees the gate/up projection (`gate_proj.weight` or
    // `.w1.weight`: a contiguous output-row slice) and the down projection
    // (`down_proj.weight` or `.w2.weight`: a strided input-column gather), and
    // BOTH scale spellings ride the same substring tests onto the same axis —
    // which is exactly the C reference's hand-rolled
    // `j < 2 ? offset : gather_row` split.
    let t = Instant::now();
    let mut data = crate::asset::shard::slice_for(name, src, shape, *want, shard_rank, shard_n)?;
    if resident {
        data = std::borrow::Cow::Owned(if *pf_rows == 0 {
            resident_moe_scales(&data)?
        } else {
            shuffle_moe_weight_16x32(&data, *pf_rows as usize, *pf_k as usize)?
        });
    }
    let mut gather_ns = t.elapsed().as_nanos() as u64;
    let pf = if *pf_dst != 0 {
        // Preshuffled copy: out[((kt*R)+r)*64 + b] = in[r*K + kt*64 + b]. A pure
        // permutation of the SAME bytes — the scrub-during-push commutes with it,
        // so the pf slab sees exactly the values the row-major slab does.
        let t = Instant::now();
        let (rows, kb) = (*pf_rows as usize, *pf_k as usize);
        debug_assert_eq!(data.len(), rows * kb);
        debug_assert_eq!(kb % 64, 0);
        let mut shuffled = vec![0u8; data.len()];
        let nkt = kb / 64;
        for r in 0..rows {
            for kt in 0..nkt {
                let s = r * kb + kt * 64;
                let d = (kt * rows + r) * 64;
                shuffled[d..d + 64].copy_from_slice(&data[s..s + 64]);
            }
        }
        gather_ns += t.elapsed().as_nanos() as u64;
        Some((*pf_dst, shuffled))
    } else {
        None
    };
    let moe2 = if *moe2_dst == 0 {
        None
    } else {
        let t = Instant::now();
        let shuffled = if *moe2_scale {
            shuffle_mxfp4_moe2_scale(&data, *moe2_rows as usize, *moe2_k as usize)?
        } else {
            shuffle_moe_weight_16x32(&data, *moe2_rows as usize, *moe2_k as usize)?
        };
        gather_ns += t.elapsed().as_nanos() as u64;
        Some((*moe2_dst, shuffled))
    };
    Ok(GatheredExpert {
        dst: *dst,
        want: *want,
        scrub: *scrub,
        data,
        pf,
        moe2,
        gather_ns,
        fault_ns,
    })
}

fn bind_packed_experts(
    be: &HsaBackend,
    blob: &DevBlob,
    ckpt: &crate::asset::checkpoint::Checkpoint,
    devp: &[DeviceMem],
    names: &[String],
    ring: &mut crate::device::hsa::HsaUploadRing,
    rank: u32,
    n_gpu: u32,
    prof: &LoadProf,
    do_prefault: bool,
    populate: bool,
    resident_tables: &[Option<u16>],
) -> Result<(Vec<DeviceMem>, u64, Vec<(u16, amd_moe_aiter::ResidentWeights)>)> {
    let layers: Vec<(usize, String, bool)> = blob
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(i, td)| {
            if let Some(pfx) = td.name.strip_suffix("expert_weight_table_ep") {
                Some((i, pfx.to_string(), true))
            } else {
                td.name
                    .strip_suffix("expert_weight_table")
                    .map(|pfx| (i, pfx.to_string(), false))
            }
        })
        .collect();
    if layers.is_empty() {
        return Ok((Vec::new(), 0, Vec::new()));
    }
    let t0 = std::time::Instant::now();
    // `I_moe` is not inferred: it is read off the very instruction that will
    // stream these weights. Every gate/up arm — per-slot bf16, per-slot
    // block-fp8, and the grouped fp8 collapse — reads the layer's
    // `expert_weight_table` from `t[3]` and carries `I_moe` in `i[1]`
    // (`crates/packet/src/slots.rs`).
    //
    // The `down` arm shares `t[3]` and carries H there instead, so the two must
    // be told apart. This USED to key on `t[0] == act.fu`, the tensor the gate/up
    // arm writes — which worked only because every emitter before Kimi-K3 named
    // that tensor exactly `act.fu`. K3's does not: its MoE activations are
    // per-layer (`act.l1.moe.fu`), so the name lookup failed outright and no K3
    // packet could bind an expert. The OPCODE is what actually distinguishes the
    // two arms, it is the same fact devgen encoded when it chose the op, and it
    // cannot be renamed. On every pre-K3 packet this selects the identical
    // instruction: the ops below are exactly the ones that write `act.fu`.
    const GLU_ARMS: [DevOp; 3] = [
        DevOp::MoeExpertGlu,
        DevOp::MoeExpertGluFp8Blk,
        DevOp::MoeGroupGluFp8Blk,
    ];
    // A BATCHED decode program routes its MoE through the GROUPED PREFILL chain, not the
    // per-slot decode chain: at T rows the emitter picks `MoeGroupGluPf`, which sorts the
    // (row, expert) pairs by expert so one expert's weights cross HBM once for every row that
    // chose it. That is the correct chain for a batch — MoE has no cross-row state, and sharing
    // the weight traffic across rows is the whole point of batching a memory-bound decode.
    //
    // Its operands sit in DIFFERENT SLOTS, which is the entire reason this needs its own arm:
    // the table is `t[2]` and `I_moe` is `i[0]` (`crates/packet/src/slots.rs:196`), against
    // `t[3]` / `i[1]` on the three arms above. Matching it with the same slot indices would
    // find nothing, and the loader would report `expert_weight_table is declared but no decode
    // instruction streams experts through it` on a program that streams them perfectly well.
    let dec = blob.progs.last().expect("checked non-empty");
    let i_moe_of = |i_ewt: usize| -> Option<u64> {
        let ep_full = blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .find(|d| {
                d.op == DevOp::MoeGroupGluPf as u16 && d.t[2] as usize == i_ewt && d.i[6] == n_gpu
            })
            .map(|d| d.i[0] as u64);
        if ep_full.is_some() {
            return ep_full;
        }
        dec.insts.iter().find_map(|d| {
            if d.t[3] as usize == i_ewt && GLU_ARMS.iter().any(|&o| o as u16 == d.op) {
                Some(d.i[1] as u64)
            } else if d.t[2] as usize == i_ewt && d.op == DevOp::MoeGroupGluPf as u16 {
                Some(d.i[0] as u64)
            } else if d.t[2] as usize == i_ewt && d.op == DevOp::MoeAiterFp8Pf as u16 {
                Some(d.i[2] as u64)
            } else {
                None
            }
        })
    };

    let mut bufs = Vec::with_capacity(layers.len() * 2);
    let mut resident_weights = Vec::new();
    let mut i_moe = 0u64;
    // Which spelling the checkpoint turned out to have, for the one log line
    // that says so. A load that binds the wrong layout is silent by nature, so
    // the resolved answer belongs in the record rather than in a debug session.
    let mut layout = String::from("none");
    let mut wbytes = 0u64;
    for (i_ewt, pfx, ep_table) in &layers {
        let resident = resident_tables[*i_ewt].is_some();
        let table_suffix = if *ep_table {
            "expert_scale_table_ep"
        } else {
            "expert_scale_table"
        };
        let i_est = names
            .iter()
            .position(|x| *x == format!("{pfx}{table_suffix}"))
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "{pfx}expert_weight_table has no matching expert_scale_table"
                ))
            })?;
        // The declared table size IS the expert count: `[E][3]` u64.
        let n_exp = (blob.tensors[*i_ewt].bytes / 24) as u32;
        if n_exp == 0 || !blob.tensors[*i_ewt].bytes.is_multiple_of(24) {
            return Err(RuntimeError::Device(format!(
                "{pfx}expert_weight_table is {} B, not a whole [E][3] u64 table",
                blob.tensors[*i_ewt].bytes
            )));
        }
        i_moe = i_moe_of(*i_ewt).ok_or_else(|| {
            RuntimeError::Device(format!(
                "{pfx}expert_weight_table is declared but no decode instruction \
                 streams experts through it — nothing to pack against"
            ))
        })?;
        // WHICH SPELLING, from the checkpoint, before anything is sized against
        // it — then the weight/scale size agreement, once, for expert 0.
        let en = resolve_expert_names(ckpt, pfx)?;
        check_expert_geometry(ckpt, &en)?;
        layout = format!("{}{{gate,up,down}}{}+{}", en.ns, en.payload, en.scale);
        // Geometry from expert 0; every expert in a layer is the same shape.
        let probe = en.weight_of(0, 0);
        let (w0, shape0) = ckpt
            .tensor_ex(&probe)
            .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {probe}")))?;
        // `I_moe` is the gate projection's OUTPUT dim under every spelling —
        // `[I_moe, K]` for `gate_proj.weight`, `[I_moe, latent/2]` for a packed
        // `w1.weight_packed`. The packing halves K, never N, so this comparison
        // against the packet's declared `I_moe` is unaffected by it.
        let i_moe_full = *shape0.first().unwrap_or(&0) as u64;
        let (owned, whole) = if i_moe == i_moe_full {
            // EP: this rank owns a contiguous block of WHOLE experts.
            (
                packet::moe_ep::balanced_expert_range(n_exp, n_gpu, rank),
                true,
            )
        } else if i_moe * n_gpu as u64 == i_moe_full {
            // TP: every rank slices every expert.
            (0..n_exp, false)
        } else {
            return Err(RuntimeError::Device(format!(
                "{pfx}: the packet streams I_moe={i_moe} per expert but the checkpoint's \
                 experts are {i_moe_full} wide, which is neither the whole expert (EP) \
                 nor a 1/{n_gpu} slice (TP)"
            )));
        };
        let n_local = owned.len() as u64;
        // Slot strides: what ONE {expert, proj} occupies in the packed buffer.
        let w_stride = w0.len() as u64 / if whole { 1 } else { n_gpu as u64 };
        let s_probe = en.scale_of(0, 0);
        let s_stride = ckpt
            .tensor_ex(&s_probe)
            .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT SCALE: {s_probe}")))?
            .0
            .len() as u64
            / if whole { 1 } else { n_gpu as u64 };

        if resident && (*ep_table || whole || n_gpu != 8 || n_exp != 256
            || i_moe != 256 || en.microscaled() || shape0 != [2048, 6144]
            || w_stride != 256 * 6144 || s_stride != 96 * 4
            || resident_tables[*i_ewt] != Some(i_est as u16))
        {
            return Err(RuntimeError::Device(format!("{pfx}: resident MoE requires TP8 block-FP8 H6144/I256/E256 weights")));
        }
        let t_alloc = Instant::now();
        let d_w = EngineDevice::alloc(be, (n_local * 3 * w_stride).max(1))?;
        let d_s = EngineDevice::alloc(be, (n_local * 3 * s_stride).max(1))?;
        LoadProf::add(&prof.alloc_ns, t_alloc);
        let (wtab, stab) = if resident {
            resident_weights.push((*i_ewt as u16, amd_moe_aiter::ResidentWeights::new(d_w.base, d_s.base)));
            (amd_moe_aiter::resident_expert_table(d_w.base, w_stride),
             amd_moe_aiter::resident_expert_table(d_s.base, s_stride))
        } else {
            (crate::orch::moe::packed_expert_table(d_w.base, w_stride, n_exp, owned.clone()),
             crate::orch::moe::packed_expert_table(d_s.base, s_stride, n_exp, owned.clone()))
        };
        // PRESHUFFLED PREFILL SLAB (PLOW_MOE_PF_SHUF): the blob declaring
        // `{pfx}expert_weight_table_pf` is the emit-time opt-in. A SECOND slab holds every
        // projection permuted to B'[K/64][R][64] so the grouped prefill GEMM's per-k-tile B
        // stream is one contiguous 16 KiB block (full 128 B lines) instead of 64 B row-slices
        // at K-stride — the aiter-asm preshuffle, done once at bind. Decode keeps streaming
        // whole rows from the row-major slab; the cost is one extra slab of HBM and one host
        // permutation pass per projection.
        let i_ewt_pf = names
            .iter()
            .position(|x| *x == format!("{pfx}expert_weight_table_pf"));
        let (d_wp, wptab) = if i_ewt_pf.is_some() {
            if en.microscaled() {
                return Err(RuntimeError::Device(format!(
                    "{pfx}expert_weight_table_pf declared for an MXFP4 expert layout — the \
                     preshuffle transform is defined for 1 B/element block-fp8 payloads only"
                )));
            }
            let t_a = Instant::now();
            let d = EngineDevice::alloc(be, (n_local * 3 * w_stride).max(1))?;
            LoadProf::add(&prof.alloc_ns, t_a);
            let t = crate::orch::moe::packed_expert_table(d.base, w_stride, n_exp, owned.clone());
            (Some(d), Some(t))
        } else {
            (None, None)
        };
        let moe2_weight_suffix = if *ep_table {
            "expert_weight_table_moe2_ep"
        } else {
            "expert_weight_table_moe2"
        };
        let moe2_scale_suffix = if *ep_table {
            "expert_scale_table_moe2_ep"
        } else {
            "expert_scale_table_moe2"
        };
        let i_ewt_moe2 = names
            .iter()
            .position(|x| *x == format!("{pfx}{moe2_weight_suffix}"));
        let i_est_moe2 = names
            .iter()
            .position(|x| *x == format!("{pfx}{moe2_scale_suffix}"));
        if i_ewt_moe2.is_some() != i_est_moe2.is_some() {
            return Err(RuntimeError::Device(format!(
                "{pfx}: lean MoE stage-2 weight and scale companion tables must be declared together"
            )));
        }
        let (d_w2, d_s2, w2tab, s2tab) = if i_ewt_moe2.is_some() {
            if !en.microscaled() {
                return Err(RuntimeError::Device(format!(
                    "{pfx}: lean MoE stage-2 companion requires MXFP4 payloads with E8M0 scales"
                )));
            }
            let kbytes = usize::try_from(i_moe / 2).unwrap_or(0);
            let groups = usize::try_from(i_moe / 32).unwrap_or(0);
            if kbytes == 0 || groups == 0 || w_stride as usize % kbytes != 0 {
                return Err(RuntimeError::Device(format!(
                    "{pfx}: lean MoE stage-2 cannot derive down geometry from I={i_moe}, stride={w_stride}"
                )));
            }
            let rows = w_stride as usize / kbytes;
            if s_stride as usize != rows.saturating_mul(groups) {
                return Err(RuntimeError::Device(format!(
                    "{pfx}: down scale stride {s_stride} disagrees with {rows}x{groups} E8M0 rows"
                )));
            }
            let scale_stride = rows.div_ceil(256) * 256 * (groups.div_ceil(8) * 8);
            let t_a = Instant::now();
            let dw = EngineDevice::alloc(be, (n_local * w_stride).max(1))?;
            let ds = EngineDevice::alloc(be, (n_local * scale_stride as u64).max(1))?;
            LoadProf::add(&prof.alloc_ns, t_a);
            let mut wt = vec![0u64; n_exp as usize * 3];
            let mut st = vec![0u64; n_exp as usize * 3];
            for (local, e) in owned.clone().enumerate() {
                wt[e as usize * 3 + 2] = dw.base + local as u64 * w_stride;
                st[e as usize * 3 + 2] = ds.base + local as u64 * scale_stride as u64;
            }
            (Some(dw), Some(ds), Some(wt), Some(st))
        } else {
            (None, None, None, None)
        };

        // The layer's reads, IN ORDER, before any of them happens.
        //
        // Materialised up front (~3 k `String`s per layer, the same ones a
        // build-at-use loop would allocate) so a worker pool can claim entries
        // by index; workers claim in plan order, which keeps the disk reads
        // roughly sequential.
        let (shard_rank, shard_n) = if whole { (0, 1) } else { (rank, n_gpu) };
        // Scrub 0x80 (OCP -0) out of BLOCK-FP8 expert payloads on the way through the staging
        // slab — value-identical, and it is what lets the CDNA3 grouped-GEMM staging decode
        // drop its neg-0 mask (`mpf_fp8x4_to_bf16_h`, runtime/amd/op_moe.h). Never the scales
        // (f32) and never an MXFP4 payload (0x80 there is two live fp4 nibbles).
        let scrub_w = !en.microscaled();
        // Per-entry: (name, dst, bytes, scrub, pf_dst, pf_rows, pf_kbytes). pf_dst == 0 means no
        // preshuffled copy (scale entries, or the pf table not declared). Geometry per
        // projection: gate/up are [I_moe][K] row-major shards, down is [H][I_moe] — in both
        // cases the shard is [rows][kbytes] with rows*kbytes == w_stride.
        let mut plan: Vec<(String, u64, u64, bool, u64, u64, u64, u64, u64, u64, bool)> =
            Vec::with_capacity(owned.len() * 6);
        for e in owned {
            for j in 0..3 {
                let idx = e as usize * 3 + j;
                let (pf_rows, pf_k) = if j < 2 {
                    (i_moe, w_stride / i_moe)
                } else {
                    (w_stride / i_moe, i_moe)
                };
                let pf_dst = wptab.as_ref().map_or(0, |t| t[idx]);
                let w2_dst = w2tab.as_ref().map_or(0, |t| t[idx]);
                let s2_dst = s2tab.as_ref().map_or(0, |t| t[idx]);
                let down_kbytes = i_moe / 2;
                let down_rows = if down_kbytes == 0 {
                    0
                } else {
                    w_stride / down_kbytes
                };
                let down_groups = i_moe / 32;
                plan.push((
                    en.weight_of(e, j),
                    wtab[idx],
                    w_stride,
                    scrub_w,
                    pf_dst,
                    pf_rows,
                    pf_k,
                    w2_dst,
                    down_rows,
                    down_kbytes,
                    false,
                ));
                plan.push((
                    en.scale_of(e, j),
                    stab[idx],
                    s_stride,
                    false,
                    0,
                    0,
                    0,
                    s2_dst,
                    down_rows,
                    down_groups,
                    true,
                ));
            }
        }
        // THE GATHER RUNS ON A WORKER POOL; the ring stays on this thread.
        //
        // One thread walking the plan was the whole load on a DeepSeek-shaped
        // checkpoint: the down projection and every 2-D scale are row shards, so
        // `slice_for` is thousands of sub-KiB strided copies per expert, and the
        // pf preshuffle is another full pass — page-fault- and latency-bound
        // work one core cannot saturate the page cache with (measured 57 s of
        // gather on a warm GLM-5.2 TP8 rank). Workers do the fault + slice +
        // preshuffle in parallel and each faults its own span in via
        // `Checkpoint::populate` first — the ~16-way concurrency the drive
        // needs on a cold cache comes from the pool itself, so this loop no
        // longer feeds the shared `Prefetcher`.
        //
        // Push order is COMPLETION order, not plan order: every entry carries
        // its precomputed device address, so the destination bytes are
        // identical either way. The ring keeps a single owner because pinned
        // staging is the SDMA correctness rule (see the note below), and the
        // bounded channel is the memory cap: at most `2 * workers` gathered
        // entries (~a few MiB each) exist at once.
        //
        // `gather_ns`/`fault_ns` become worker-summed parallel time — same
        // convention as `PrefetchStats`, and they can exceed wall clock.
        let workers = expert_gather_threads(n_gpu).min(plan.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let plan = &plan;
        // Through a PINNED slab, always. The copy does not pin its source, so
        // handing it a `slice_for` gather buffer (an ordinary `Vec`) faults the
        // SDMA engine — the one trap the C reference calls out by name.
        let mut push_gathered = |g: GatheredExpert| -> Result<()> {
            prof.gather_ns.set(prof.gather_ns.get() + g.gather_ns);
            prof.fault_ns.set(prof.fault_ns.get() + g.fault_ns);
            let stage_bytes = ring.chunk();
            for (o, chunk) in g.data.chunks(stage_bytes).enumerate() {
                let t = Instant::now();
                let at = g.dst + (o * stage_bytes) as u64;
                if g.scrub {
                    ring.push_scrub_fp8_neg0(at, chunk)?;
                } else {
                    ring.push(at, chunk)?;
                }
                LoadProf::add(&prof.memcpy_ns, t);
                prof.chunks.set(prof.chunks.get() + 1);
            }
            wbytes += g.want;
            if let Some((pf_dst, shuffled)) = g.pf {
                for (o, chunk) in shuffled.chunks(stage_bytes).enumerate() {
                    let t = Instant::now();
                    ring.push_scrub_fp8_neg0(pf_dst + (o * stage_bytes) as u64, chunk)?;
                    LoadProf::add(&prof.memcpy_ns, t);
                    prof.chunks.set(prof.chunks.get() + 1);
                }
                wbytes += g.want;
            }
            if let Some((dst, shuffled)) = g.moe2 {
                for (o, chunk) in shuffled.chunks(stage_bytes).enumerate() {
                    let t = Instant::now();
                    ring.push(dst + (o * stage_bytes) as u64, chunk)?;
                    LoadProf::add(&prof.memcpy_ns, t);
                    prof.chunks.set(prof.chunks.get() + 1);
                }
                wbytes += shuffled.len() as u64;
            }
            Ok(())
        };
        std::thread::scope(|s| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<GatheredExpert>>(workers * 2);
            for _ in 0..workers {
                let tx = tx.clone();
                let (next, stop) = (&next, &stop);
                s.spawn(move || loop {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(entry) = plan.get(i) else { return };
                    let r = gather_expert_entry(
                        ckpt,
                        entry,
                        shard_rank,
                        shard_n,
                        populate,
                        do_prefault,
                        resident,
                    );
                    if tx.send(r).is_err() {
                        return;
                    }
                });
            }
            drop(tx);
            // On error: stop the claimers, then keep draining so no worker is
            // left blocked in `send` when the scope joins — that would hang.
            let mut first_err = None;
            while let Ok(r) = rx.recv() {
                if first_err.is_some() {
                    continue;
                }
                if let Err(e) = r.and_then(&mut push_gathered) {
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    first_err = Some(e);
                }
            }
            first_err.map_or(Ok(()), Err)
        })?;
        EngineDevice::upload(be, &devp[*i_ewt], 0, as_bytes(&wtab))?;
        EngineDevice::upload(be, &devp[i_est], 0, as_bytes(&stab))?;
        if let (Some(ip), Some(tab)) = (i_ewt_pf, &wptab) {
            EngineDevice::upload(be, &devp[ip], 0, as_bytes(tab))?;
        }
        if let (Some(iw), Some(is), Some(wt), Some(st)) = (i_ewt_moe2, i_est_moe2, &w2tab, &s2tab) {
            EngineDevice::upload(be, &devp[iw], 0, as_bytes(wt))?;
            EngineDevice::upload(be, &devp[is], 0, as_bytes(st))?;
        }
        bufs.push(d_w);
        bufs.push(d_s);
        if let Some(d) = d_wp {
            bufs.push(d);
        }
        if let Some(d) = d_w2 {
            bufs.push(d);
        }
        if let Some(d) = d_s2 {
            bufs.push(d);
        }
    }
    // No expert is bound until its copy has retired. The pointer tables above
    // are uploaded through the blocking path and name a DIFFERENT address, so
    // they are ordered by construction; the expert bytes are not.
    ring.drain()?;
    tracing::info!(
        layers = layers.len(),
        gib = format!("{:.2}", wbytes as f64 / (1u64 << 30) as f64).as_str(),
        i_moe,
        layout = layout.as_str(),
        secs = format!("{:.1}", t0.elapsed().as_secs_f64()).as_str(),
        "routed experts packed; expert pointer tables filled"
    );
    Ok((bufs, wbytes, resident_weights))
}

/// Contiguous instruction span covering every KV-row patch site, or `None` when
/// there are none.
///
/// The sites are SCATTERED in k/v pairs across all layers (Gemma-31B: `[4,664]`
/// of 676), so one contiguous slice beats a per-instruction scatter: fewer
/// bytes than the whole stream and, more importantly, ONE h2d submission
/// instead of `n_kvrow` of them. Submission overhead, not bytes, is what costs
/// here.
/// May this compiler-owned tensor skip the load-time zeroing?
///
/// The skip is a PERFORMANCE optimisation with a precondition, not a property
/// of the `kv.` namespace: it is sound only where every element is WRITTEN
/// BEFORE IT IS READ. An append-only KV cache satisfies that — attention reads
/// only `[0, kvlen)` and each row is written on the step that admits it — which
/// is why skipping it saves 11.5 GiB of memset on GLM and changes nothing.
///
/// Kimi-K3 put two things under `kv.` that do NOT satisfy it, and both were
/// silently inheriting the skip:
///
/// * the KDA RECURRENT STATE (`kv.{l}.state`, `kv.{l}.conv_state.*`). The
///   recurrence is `state = state * decay(gate) + beta * (v - state·k) ⊗ k`, so
///   it READS `state` on the very first token of the very first sequence. From
///   uninitialised HBM that is garbage folded into an accumulator over up to
///   10^6 rank-1 updates, and it never washes out.
/// * the ATTNRES SNAPSHOT RING (`kv.blkres`), which AttnRes mixes over from the
///   first layer that has a snapshot.
///
/// Neither faults and neither reports a missing weight. They are `kv.`-named
/// because `packet::names::is_checkpoint_weight` classifies by EXCLUSION, so
/// the prefix is what stops the loader demanding them of the checkpoint — it
/// was never a claim about their write-before-read discipline.
fn kv_skips_zeroing(name: &str) -> bool {
    name.starts_with("kv.") && !is_carried_state(name)
}

/// Kimi-K3's AttnRes score weight, folded from the TWO tensors the checkpoint
/// actually ships. `None` when `name` is not one of them.
///
/// `runtime/amd/op_k3.h` states the relation as the reference implementation:
///
/// ```text
/// score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
/// scores       = (k * score_weight).sum(-1)
/// ```
///
/// so the emitter declares ONE f32 `[hidden]` per site while
/// `models--moonshotai--Kimi-K3` ships two bf16 tensors — `*_res_norm.weight`
/// `[7168]` and `*_res_proj.weight` `[1, 7168]` — 93 of each, at both the
/// attention and the MLP site. Without this fold every one of those 186 handles
/// resolves to MISSING WEIGHT and no real-weight K3 run can start.
///
/// The fold is exact, not an approximation, and it is worth saying why it is
/// ALLOWED to be a plain elementwise product. The score is
/// `proj · rmsnorm(x, norm)`, and RMS normalisation scales the whole row by the
/// single scalar `1/rms(x)`. A scalar commutes out of the dot product, so
/// `proj · (x/rms · norm) == (proj ⊙ norm) · (x/rms)` — the gain can be folded
/// into the projection ahead of time and the kernel divides by the RMS itself.
/// If the norm were anything per-element-nonlinear this would not hold.
///
/// f32 because the fold is a PRODUCT OF TWO bf16 VALUES: keeping the result in
/// bf16 would round away most of what the multiply just computed, and the packet
/// declares f32 for that reason. There is no TP axis here — `[hidden]` is
/// replicated on every rank — so this runs identically at any `--num-gpus`.
fn fold_res_score(c: &crate::asset::checkpoint::Checkpoint, name: &str) -> Option<Result<Vec<u8>>> {
    let stem = name.strip_suffix("_res_score.weight")?;
    let bf16 = |n: &str| -> Option<Vec<f32>> {
        let (raw, _) = c.tensor_ex(n)?;
        Some(
            raw.chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
        )
    };
    let (nn, pn) = (
        format!("{stem}_res_norm.weight"),
        format!("{stem}_res_proj.weight"),
    );
    let (Some(g), Some(p)) = (bf16(&nn), bf16(&pn)) else {
        // Name matched the pattern but the sources are absent: report the pair
        // rather than the derived name, which is in no checkpoint by design.
        return Some(Err(RuntimeError::Device(format!(
            "MISSING WEIGHT: {name} is DERIVED and needs both `{nn}` and `{pn}`; \
             at least one is not in the checkpoint"
        ))));
    };
    if g.len() != p.len() {
        return Some(Err(RuntimeError::Device(format!(
            "{name}: norm is {} wide and proj is {} — they must agree",
            g.len(),
            p.len()
        ))));
    }
    let mut out = Vec::with_capacity(g.len() * 4);
    for (a, b) in g.iter().zip(&p) {
        out.extend_from_slice(&(a * b).to_le_bytes());
    }
    Some(Ok(out))
}

/// Resolve every checkpoint weight the blob declares, BEFORE anything is uploaded.
///
/// The load path used to answer "is this weight in the checkpoint, and does it bind to the
/// declared byte count?" one tensor at a time inside the upload loop, so the answer arrived
/// after every earlier tensor had already been DMA'd. On a 181 GiB/rank GLM-5.3 that is
/// minutes of upload on four ranks before a refusal that was decidable from the safetensors
/// index at t=0. This is that decision, taken once, up front.
///
/// It reads NO payload: `tensor_ex` returns a subrange of an mmap without touching it, and
/// `slice_for` only measures. The `n_gpu > 1` row-parallel skip is `touched`'s guard for the
/// same reason it has one — a row gather is the single arm that copies, and doing it twice
/// would cost more than the check is worth. Those tensors keep the old late failure; every
/// other class now fails at t=0.
///
/// The three rules below are the upload loop's own, not a second opinion about them:
/// `is_checkpoint_weight` as the gate, `fp8/` routing to the twin checkpoint, and the
/// two-spelling lookup. Getting any of them wrong here would refuse a load that works.
fn preflight_weights(
    blob: &DevBlob,
    ckpt: Option<&crate::asset::checkpoint::Checkpoint>,
    fp8_ckpt: Option<&crate::asset::checkpoint::Checkpoint>,
    rank: u32,
    n_gpu: u32,
) -> Result<()> {
    for td in &blob.tensors {
        if !packet::names::is_checkpoint_weight(&td.name) {
            continue;
        }
        let is_fp8 = td.name.starts_with("fp8/");
        let Some(c) = (if is_fp8 { fp8_ckpt } else { ckpt }) else {
            // No checkpoint at all is not this function's error: the loop reports the
            // missing `PLOW_FP8_DIR`, and a checkpoint-less bind is a legal packet-only load.
            continue;
        };
        let stripped = td.name.strip_prefix("fp8/").unwrap_or(&td.name);
        // DERIVED, and in no checkpoint by design — `fold_res_score` builds it from a norm
        // and a proj. Without this exemption a Kimi-K3 load reports 186 false MISSING
        // WEIGHTs before it starts.
        if stripped.ends_with("_res_score.weight") {
            continue;
        }
        let Some((src, shape)) = c.tensor_ex(&td.name).or_else(|| c.tensor_ex(stripped)) else {
            return Err(RuntimeError::Device(format!(
                "MISSING WEIGHT: {} (tried that name and the `fp8/`-stripped form{}) \
                 — refused before the upload, not after it",
                td.name,
                if is_fp8 { " in PLOW_FP8_DIR" } else { "" }
            )));
        };
        // The DSA indexer's two projections are declared bf16 and may be block-fp8 on disk;
        // `plan` is the only thing that can tell a bindable one from a refusable one, and it
        // must be asked BEFORE `slice_for`, which would reject the fp8 byte count outright.
        if let Some(plan) = crate::asset::dsa_indexer::plan(c, stripped, td.bytes) {
            plan?;
            continue;
        }
        // A row-parallel gather is the one arm that allocates. Skipped for exactly the
        // shapes `touched` skips it for; everything else is measured here.
        let row = n_gpu > 1
            && shape.len() == 2
            && crate::asset::shard::shard_of(stripped) == crate::asset::shard::Shard::Row;
        if !row {
            crate::asset::shard::slice_for(stripped, src, shape, td.bytes, rank, n_gpu)?;
        }
    }
    Ok(())
}

/// A `kv.`-namespace tensor that CARRIES STATE ACROSS TOKENS rather than being
/// appended to — the KDA recurrent state, its three conv windows, and the AttnRes
/// snapshot ring.
///
/// Two callers, and separating them was the bug. [`kv_skips_zeroing`] asks whether
/// LOAD may skip the memset; [`AmdEngine::begin_slot`] asks what a NEW SEQUENCE
/// must clear. Those are the same set for the same reason — these tensors are read
/// before they are written — but they were not the same code, and only the first
/// existed. So the state was correctly zeroed once at model load and then never
/// again: request 2 inherited request 1's recurrence and conv windows, and a
/// linear-attention model conditioned on the previous conversation. Nothing faults
/// and nothing reports a missing weight; the second answer is merely wrong, in a
/// way that reads as fluent.
///
/// Substring, not suffix: `conv_state.q`/`.k`/`.v` are three tensors under one
/// idea, and a future `kv.{l}.state.v` must not slip through a match written
/// against today's exact spellings.
fn is_carried_state(name: &str) -> bool {
    name.starts_with("kv.") && (name.contains("state") || name.contains("blkres"))
}

fn patch_tp_xaudit(insts: &mut [DevInst64], status_id: u32) {
    for d in insts {
        if matches!(
            DevOp::from_u16(d.op),
            Some(
                DevOp::XReduce
                    | DevOp::XReduceTwoShot
                    | DevOp::XReduceAddNorm
                    | DevOp::XArgmaxFin
                    | DevOp::XReduceScatter
                    | DevOp::XAllGather
            )
        ) {
            d.fj[2] = status_id + 1;
        }
    }
}

/// This rank's place in a TP group — everything the engine needs from
/// [`crate::exec::tp::TpGroup`], as plain device addresses.
///
/// Deliberately NOT a borrow of `TpRank`. The engine needs four numbers and a
/// base address; taking the rank itself would couple the AMD engine to the group
/// type, drag an `Arc<dyn Backend>` alongside the `Arc<HsaBackend>` it already
/// holds, and make the binding untestable without eight GPUs. Every field here
/// is one `TpRank` accessor.
#[derive(Clone, Copy, Debug)]
pub struct TpBind {
    /// This rank's index in the group — `PlowProgram::rank`.
    pub rank: u32,
    /// TP degree — `PlowProgram::n_gpu`.
    ///
    /// The interpreter's convention is that **0 means single-GPU** (`n_gpu: 0`
    /// in [`AmdEngine::kernarg`] when there is no group), so this is never 0
    /// here; a group of one is still `1`.
    pub n_gpu: u32,
    /// `[n_gpu]` device table of every rank's peer-region base —
    /// `PlowProgram::peer_scratch`. From `TpRank::peer_scratch_table`.
    pub peer_table: u64,
    /// This rank's cross-GPU counters, inside its own peer region —
    /// `PlowProgram::xctr`. From `TpRank::xctr`.
    pub xctr: u64,
    /// Reserved xctr id used by the compact device audit status line.
    pub xstatus_id: u32,
    /// This rank's peer-region base (`peer_scratch[rank]`). `act.og_tp` binds at
    /// offset 0 and `act.dg_tp` at [`TpBind::slot_b`], so the row-parallel
    /// o_proj/down write their partials where peers can read them.
    pub scratch_base: u64,
    /// Byte offset of partial slot B — `DevBlob::tp`'s `slot_bytes`, which is
    /// what `devgen` baked into every `XReduce`'s `i[2]`. Read from the blob
    /// rather than recomputed: a host that computed its own would put `dg_tp`
    /// where no peer reads it, and nothing would say so.
    pub slot_b: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct XAuditArgs {
    insts: u64,
    n_inst: u32,
    _pad: u32,
    xctr: u64,
    n_xctr: u32,
    n_gpu: u32,
    status: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct TokenCaptureArgs {
    ids: u64,
    ring: u64,
    step: u32,
    quantum: u32,
    batch: u32,
}

pub(crate) const DEFERRED_TOKEN_MAX_STEPS: usize = 4;

#[derive(Clone, Copy)]
#[repr(C)]
struct StateClearRange {
    base: u64,
    slot_stride: u64,
    words: u32,
    _pad: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StateClearArgs {
    ranges: u64,
    n_ranges: u32,
    slot: u32,
}

const STATE_CLEAR_CHUNK: u64 = 256 * 1024;

fn state_clear_ranges(
    devp: &[DeviceMem],
    carried: &[(usize, u64)],
) -> Result<Vec<StateClearRange>> {
    let mut out = Vec::new();
    for &(i, stride) in carried {
        let m = devp.get(i).ok_or_else(|| {
            RuntimeError::Device(format!("carried-state tensor index {i} is out of range"))
        })?;
        if m.base == 0 || stride == 0 {
            continue;
        }
        if m.base % 4 != 0 || stride % 4 != 0 {
            return Err(RuntimeError::Device(format!(
                "carried-state range {i} is not u32-aligned: base={:#x} stride={stride}",
                m.base
            )));
        }
        let mut off = 0;
        while off < stride {
            let bytes = (stride - off).min(STATE_CLEAR_CHUNK);
            out.push(StateClearRange {
                base: m.base + off,
                slot_stride: stride,
                words: u32::try_from(bytes / 4).map_err(|_| {
                    RuntimeError::Device(format!("carried-state clear chunk is too large: {bytes}"))
                })?,
                _pad: 0,
            });
            off += bytes;
        }
    }
    Ok(out)
}

/// One chunk of a prefill plan: which bucket program runs it, and over which
/// absolute token range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkStep {
    /// Index of the compiled bucket program.
    pub prog: usize,
    /// Absolute position of the chunk's first row.
    pub c0: u32,
    /// REAL rows in the chunk; rows `[clen, t)` are padding.
    pub clen: u32,
}

/// One decode rung's MLA KV-split sites, plus the host image the patch edits.
///
/// The image is OWNED here rather than taken from `h_inst`: that pinned slab holds
/// only the WIDEST decode program, and a `PLOW_DECODE_BATCH_LADDER` blob dispatches
/// a NARROWER rung whenever the mux admits fewer sequences
/// (`decode_step_batched_at(.., dp)`). Patching only the widest would leave the rung
/// that actually runs on its baked split count — a silent no-op that reads as a
/// clean null. Every rung is therefore patched, so whichever the mux picks is right.
struct MlaNsplitProg {
    /// Index into `progs`.
    prog: usize,
    /// Offsets into `image` (NOT instruction indices) whose `i[4]` is the split
    /// count — flash AND merge, which stride the same partials.
    sites: Vec<usize>,
    /// First instruction of `image` within the program.
    lo: usize,
    /// `insts[lo ..= hi]`, patched in place and uploaded whole.
    image: Vec<DevInst64>,
}

/// MLA KV-split state under `PLOW_MLA_NS_LIVE`.
///
/// `cur` exists so the step pays nothing in the common case: the live count is a
/// step function of `kv_len` (it moves only at a multiple of `NS_PER`, and not at
/// all below `NS_FLOOR * NS_PER`), so the patch + upload runs a handful of times
/// over a whole generation rather than every token. That is also why the upload
/// is an ordinary `upload` (which page-locks its source per call) rather than the
/// pinned-slab path `patch_kvrow` needs: this is not the hot path.
struct MlaNsplit {
    /// One entry per decode rung, all sharing `baked`/`cur`.
    progs: Vec<MlaNsplitProg>,
    /// What the emitter wrote, from its `max_ctx`. A ceiling: buffers are sized
    /// for it and the live value only ever goes down.
    baked: u32,
    /// What is resident on the device right now.
    cur: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedPrefillBinding {
    prog: usize,
    n_spans: u32,
    n_rows: u32,
}

fn validate_packed_prefill(
    prog: usize,
    rung: u32,
    batch: usize,
    spans: &[PrefillSpan],
    parked: &[u32],
) -> Result<PackedPrefillBinding> {
    if spans.is_empty() {
        return Err(RuntimeError::Device(
            "packed prefill requires at least one span".into(),
        ));
    }
    let prog_u32 = u32::try_from(prog)
        .map_err(|_| RuntimeError::Device(format!("prefill program index {prog} exceeds u32")))?;
    let rows = validate_amd_packed_rows(prog_u32, 0, rung, batch, spans, parked)
        .map_err(|error| RuntimeError::Device(format!("packed prefill: {error}")))?;
    Ok(PackedPrefillBinding {
        prog,
        n_spans: rows.n_spans,
        n_rows: rung,
    })
}

fn validate_packed_prompt_slices(
    rung: u32,
    spans: &[PrefillSpan],
    prompt_slices: &[&[u32]],
) -> Result<u32> {
    if spans.len() != prompt_slices.len() {
        return Err(RuntimeError::Device(format!(
            "packed prefill has {} spans but {} prompt slices",
            spans.len(),
            prompt_slices.len()
        )));
    }
    let mut rows = 0u32;
    for (i, (span, prompt)) in spans.iter().zip(prompt_slices).enumerate() {
        if prompt.len() != span.n_rows as usize {
            return Err(RuntimeError::Device(format!(
                "packed prefill span {i} has {} rows but its prompt slice has {} tokens",
                span.n_rows,
                prompt.len()
            )));
        }
        rows = rows.checked_add(span.n_rows).ok_or_else(|| {
            RuntimeError::Device("packed prefill prompt row count overflows u32".into())
        })?;
    }
    if rows == 0 || rows > rung {
        return Err(RuntimeError::Device(format!(
            "packed prefill prompt slices contribute {rows} rows for rung {rung}"
        )));
    }
    Ok(rows)
}

fn stage_packed_prompt_rows(
    stage: &mut [u8],
    rung: u32,
    spans: &[PrefillSpan],
    prompt_slices: &[&[u32]],
) -> Result<u32> {
    let rows = validate_packed_prompt_slices(rung, spans, prompt_slices)?;
    let bytes = rung as usize * 4;
    if stage.len() < bytes * 2 {
        return Err(RuntimeError::Device(format!(
            "packed prefill input staging has {} bytes, needs {}",
            stage.len(),
            bytes * 2
        )));
    }
    stage[..bytes * 2].fill(0);
    for (span, prompt) in spans.iter().zip(prompt_slices) {
        for (local, &id) in prompt.iter().enumerate() {
            let row = span.row0 as usize + local;
            stage[row * 4..row * 4 + 4].copy_from_slice(&id.to_le_bytes());
            let pos = span.kv_row0 + local as u32;
            let off = bytes + row * 4;
            stage[off..off + 4].copy_from_slice(&pos.to_le_bytes());
        }
    }
    Ok(rows)
}

fn check_packed_prefill_dispatch(binding: Option<PackedPrefillBinding>, prog: usize) -> Result<()> {
    if let Some(bound) = binding.filter(|b| b.prog != prog) {
        return Err(RuntimeError::Device(format!(
            "packed-prefill metadata is staged for program {}, refusing to dispatch program {prog}; clear or restage it first",
            bound.prog
        )));
    }
    Ok(())
}

fn packed_prefill_kernarg(
    binding: Option<PackedPrefillBinding>,
    prog: usize,
    spans: u64,
    parked: u64,
) -> (u64, u64, u32, u32) {
    match binding.filter(|b| b.prog == prog) {
        Some(b) => (spans, parked, b.n_spans, b.n_rows),
        None => (0, 0, 0, 0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackedSegmentRoute {
    Primary,
    MlaNorm,
    MlaFlash,
    Kda,
}

fn packed_segment_route(
    active: bool,
    class: u8,
    mla_norm: bool,
    mla_flash: bool,
    kda: bool,
) -> Result<PackedSegmentRoute> {
    // The KDA family object is also the spill-isolation object for ordinary prefill.  Its
    // capability marker proves it supports null packed metadata; older packed-only objects
    // are never loaded into `kda` and retain the primary fallback here.
    if class == 7 && kda {
        return Ok(PackedSegmentRoute::Kda);
    }
    if !active {
        return Ok(PackedSegmentRoute::Primary);
    }
    if class == 0 || class == 8 {
        return Ok(PackedSegmentRoute::Primary);
    }
    let (available, route, name) = match class {
        5 => (mla_norm, PackedSegmentRoute::MlaNorm, "MLA norm/cache"),
        6 => (mla_flash, PackedSegmentRoute::MlaFlash, "MLA flash"),
        7 => (kda, PackedSegmentRoute::Kda, "KDA"),
        _ => {
            return Err(RuntimeError::Device(format!(
                "packed-prefill segment has unknown operator-family class {class}"
            )))
        }
    };
    if !available {
        return Err(RuntimeError::Device(format!(
            "packed-prefill {name} segment has no exact capability object; refusing to fall \
             back to the production interpreter"
        )));
    }
    Ok(route)
}

fn packed_prefill_topology_index(
    requested: usize,
    dec_lo: usize,
    mut role: impl FnMut(usize) -> Option<(u32, bool)>,
) -> Option<usize> {
    if requested >= dec_lo {
        return None;
    }
    let (rows, packed_only) = role(requested)?;
    if packed_only {
        return Some(requested);
    }
    (0..dec_lo)
        .find(|&candidate| role(candidate) == Some((rows, true)))
        .or(Some(requested))
}

/// Per-program local counter-bank state.
///
/// TP keeps `current` on the bank used by the last dispatch so diagnostic
/// snapshots still read that dispatch's counters. The other bank must be
/// re-armed successfully before a later TP dispatch may select it.
struct CounterBankState {
    current: Cell<u32>,
    inactive_ready: Cell<bool>,
}

impl CounterBankState {
    fn new() -> Self {
        Self {
            current: Cell::new(0),
            // Both banks are zeroed at allocation time.
            inactive_ready: Cell::new(true),
        }
    }

    fn current(&self) -> u32 {
        self.current.get()
    }

    fn inactive(&self) -> u32 {
        1 - self.current()
    }

    fn inactive_ready(&self) -> bool {
        self.inactive_ready.get()
    }

    /// Select the already-clean inactive bank for a TP dispatch.
    ///
    /// Returns `false` for the single-bank fallback, whose caller must re-arm
    /// the current bank synchronously. A failed/omitted inactive clear leaves
    /// `inactive_ready == false`, so stale local counters cannot be reused.
    fn begin_tp(&self, double_buffered: bool) -> std::result::Result<bool, ()> {
        if !double_buffered {
            return Ok(false);
        }
        if !self.inactive_ready.replace(false) {
            return Err(());
        }
        self.current.set(self.inactive());
        Ok(true)
    }

    fn mark_inactive_ready(&self) {
        self.inactive_ready.set(true);
    }

    fn select_rearmed_inactive(&self) {
        self.current.set(self.inactive());
        // The old current bank was dirtied by the dispatch that just launched.
        self.inactive_ready.set(false);
    }
}

/// One program's device-resident tables.
struct AmdProg {
    t: u32,
    packed_prefill_only: bool,
    packed_dense: bool,
    packed_dense_error: Option<String>,
    packed_needs_mla: bool,
    packed_mla_compatible: bool,
    packed_mla_segmented: bool,
    packed_needs_kda: bool,
    packed_kda_compatible: bool,
    packed_kda_segmented: bool,
    /// §3/§5.4 of `plans/unified-token-batch.md`, from
    /// [`crate::exec::amd::packed::recurrent_span_limit`]: `Ok(n)` = at most `n` request spans in
    /// one packed launch, `Err(msg)` = this program's recurrent operators have no packed form at
    /// all and `msg` names which one. Computed once at load, not per tick.
    packed_recurrent_spans: std::result::Result<u32, String>,
    n_inst: u32,
    trace_records: usize,
    n_counter: u32,
    d_inst: DeviceMem,
    d_stream: DeviceMem,
    d_sofs: DeviceMem,
    d_slen: DeviceMem,
    d_waits: DeviceMem,
    d_succs: DeviceMem,
    d_ctr: DeviceMem,
    /// `[n_cu][n_seg+1]` per-(CU, segment) window bounds into each CU's own
    /// stream slice, so a segment launch starts at its own first entry instead
    /// of rescanning the whole stream and filtering. The static path's analogue
    /// of `AmdGq::d_seg_ofs`. Derived from the uploaded stream, never read from
    /// the blob — see `PlowProgram::seg_ofs` in `runtime/common/dev_isa.h`.
    d_seg_ofs: DeviceMem,
    /// `[n_seg]` wave classes.
    seg_class: Vec<u8>,
    /// Ordered decode route for each segment. Raw KDA entries carry fully resolved rank-local
    /// kernargs, so the hot launch path performs no descriptor parsing or allocation.
    decode_routes: Vec<DecodeSegmentRoute>,
    /// Optional raw gfx950 route for a pure MXFP4 grouped-MoE Down+Combine segment.
    prefill_routes: Vec<PrefillSegmentRoute>,
    /// Immutable device-side typed arguments for raw XR+AttnRes routes. Routes retain only their
    /// addresses; this owner therefore keeps every pointer valid across queued segment launches.
    _xreduce_attnres_args: Vec<DeviceMem>,
    /// `[n_seg]` pure packed-consumer family: 5=MLA norm/cache, 6=MLA flash,
    /// 7=serial KDA, 0=not safely routable to a family object.
    packed_seg_family: Vec<u8>,
    /// Pure dense bf16 MLA segments eligible for the dedicated gfx950 V2+SV object.
    raw_mla_v2_segment: Vec<bool>,
    /// Global-queue tables; `None` when the blob carries no GQ appendix.
    gq: Option<AmdGq>,
    /// L2-domain placement (`PLOW_L2_PLACE`): XCD queues per ordered segment, or 0.
    l2_domains: u32,
    /// Base counter id of the two-level maintenance scratch; 0 = off. See `DevProgram::hier_base`.
    hier_base: u32,
    /// Bytes in ONE counter bank — `n_counter * CTR_STRIDE_U32 * 4`, i.e. what
    /// the old single-bank allocation was in total.
    ctr_span: u64,
    /// Local counter/cursor bank state. See [`AmdEngine::run`] and the TP
    /// begin/post-launch methods.
    bank: CounterBankState,
}

struct AmdGq {
    d_stream: DeviceMem,
    d_seg_ofs: DeviceMem,
    /// One cursor LINE per segment, not one word. RUNSEG enqueues every
    /// segment without a host wait and zeroes state once before the loop, so a
    /// shared cursor would be corrupted by the segment that ran first.
    d_cursor: DeviceMem,
    n_seg: u32,
    /// Bytes in ONE cursor bank. The allocation holds `ctr_banks` of them.
    cur_span: u64,
}

mod mixed_step;
mod prefix;
mod shared_prefix;
mod token_batch;

/// The AMD serving engine.
pub struct AmdEngine {
    mixed_step: Option<mixed_step::MixedAmdStep>,
    /// The unified token-batch route. A SECOND instance of the same type on a different code
    /// object: `interp_tokbatch_gq.elf`, spans covering `[0, M)`, no decode prefix. Never
    /// loaded beside `mixed_step` — they are alternatives for one (backend, family) pair, and
    /// a build that armed both would double the resident executables to measure neither.
    token_batch_step: Option<mixed_step::MixedAmdStep>,
    be: Arc<HsaBackend>,
    arch: String,
    n_cu: u32,
    progs: Vec<AmdProg>,
    /// Index of the WIDEST decode program — always last (`n_prog - 1`).
    decode: usize,
    /// Index of the FIRST decode program: the bottom of the DECODE BATCH LADDER
    /// (`PLOW_DECODE_BATCH_LADDER`). Without a ladder this equals [`Self::decode`], so
    /// `dec_lo..=decode` is a one-element range and every path is what it was.
    ///
    /// Programs `[0, dec_lo)` are the prefill bucket ladder; `[dec_lo, n_prog)` are decode
    /// rungs at ascending sequence widths, all sharing ONE tensor table sized at the widest.
    dec_lo: usize,
    devp: Vec<DeviceMem>,
    /// Owner of the one allocation the ordinarily-allocated tensors are carved
    /// out of; `devp` then holds **views** into it. Unlike the CUDA side, this
    /// wins on BOTH axes, measured on 8×MI355X loading Kimi-K3 TP8 (5408 carved
    /// tensors per rank, 22.84 GiB of named weights):
    ///
    /// | | slab | per-tensor |
    /// |---|---|---|
    /// | peak VRAM per card | 204 579 MiB | 205 904 MiB |
    /// | `alloc_ms`, named tensors | 96–266 | 6410–8802 |
    /// | wall, named tensors | 6.0–6.7 s | 13.0–16.7 s |
    ///
    /// **Memory: 1325 MiB per card, 10.35 GiB across the eight.** ROCr reports a
    /// 4 KiB granule and then ignores it — under 2 MiB it hands back the next
    /// POWER OF TWO with a 32 KiB floor, at or above 2 MiB it rounds to a 2 MiB
    /// multiple. A 1.4 MiB expert projection commits 2 MiB, 42.9% lost; a 12 KiB
    /// norm vector commits 32 KiB. One allocation pays that rounding once.
    ///
    /// **Time: ~7–8.5 s of driver time per rank, halving this phase.** This one
    /// contradicts the obvious microbenchmark, so do not re-derive it from one:
    /// 737 uniform 30 MiB allocations on an IDLE card cost 8.8 ms total, which
    /// says the call is nearly free and is why the first version of this comment
    /// claimed the slab was memory-only. The real load is not that shape — 5408
    /// unevenly sized tensors interleaved with 168 GiB of expert buffers, eight
    /// ranks against one driver — and there the per-call cost is three orders of
    /// magnitude worse. Measure this on the model, never in isolation.
    ///
    /// The packed-expert buffers are unaffected (`alloc_ms` ~1.5–2.1 s either
    /// way): `bind_packed_experts` already carves all of a layer's experts out
    /// of two allocations, which is this same trick applied earlier.
    ///
    /// Views never free, so this owner must outlive them; both live on this
    /// struct, and a view's `Drop` is a no-op, so field order cannot matter.
    ///
    /// `PerTensor` when the single allocation was refused and the loader fell
    /// back to per-tensor allocation — a fragmented card can decline one big
    /// block and still satisfy many small ones. The `Vmm` arm (lazy commit
    /// overlapped with the upload, as `exec::gpu`) is **opt-in** here
    /// (`PLOW_WEIGHT_VMM=1`) until it is measured on AMD hardware — see
    /// `asset::checkpoint::weight_vmm_amd_enabled`.
    _weight_slab: WeightSlab,
    d_tens: DeviceMem,
    tensor_names: Vec<String>,
    /// Per-MoE-layer PACKED expert buffers (weights, then block scales). Never
    /// read through here — the packet reaches them only through the addresses in
    /// `expert_weight_table`/`expert_scale_table` — but they are owning handles,
    /// so dropping them would free the memory those tables point at.
    _expert_bufs: Vec<DeviceMem>,

    /// Per-(workgroup, packet) `PlowTraceRec` buffer for the widest program,
    /// allocated only when `PLOW_TRACE_RAW` is set. The interpreter treats a
    /// null `trace` pointer as "tracing off" and then does not even read the
    /// clock, so an untraced build pays nothing for this field being here.
    ///
    d_trace: Option<DeviceMem>,
    trace_bytes: usize,
    /// Extent of the last program dispatched into `d_trace`.
    trace_write_bytes: Cell<usize>,

    k_prefill: HsaKernel,
    k_decode: HsaKernel,
    /// Packet-paired interpreter containing only FlashMlaDecode+MlaMergeFold.
    k_decode_mla: Option<HsaKernel>,
    k_kda_decode_fused: Option<HsaKernel>,
    k_grouped_moe_glu: Option<HsaKernel>,
    k_grouped_moe_down: Option<HsaKernel>,
    k_kda_chunk_intra_cached: Option<HsaKernel>,
    k_kda_chunk_intra_wave_items: Option<HsaKernel>,
    k_kda_chunk_carry_regstate: Option<HsaKernel>,
    k_kda_key_factor_wu: Option<HsaKernel>,
    k_kda_key_factor_carry: Option<HsaKernel>,
    /// One reusable `[hi | lo]` BF16 key-factor pair, sized for the widest eligible program.
    _kda_key_factor_scratch: Option<DeviceMem>,
    k_kda_chunk_wu_lean: Option<HsaKernel>,
    k_kda_chunk_wu_lean_keys: Option<HsaKernel>,
    k_kda_chunk_carry_keyfeed: Option<HsaKernel>,
    /// The lean Wu's `[hi | lo]` scaled-key pair for the key-fed carry.
    _kda_keyfeed_scratch: Option<DeviceMem>,
    k_moe_stage1_mxfp4: Option<HsaKernel>,
    k_moe_stage1_a4_quant: Option<HsaKernel>,
    k_moe_stage1_a4_reuse: Option<HsaKernel>,
    _moe_stage1_a4_scratch: Option<DeviceMem>,
    k_moe_stage2_mxfp4: Option<HsaKernel>,
    k_moe_combine: Option<HsaKernel>,
    /// The f32-mix AttnRes object and the workgroup size it advertises.
    k_attn_res_f32mix: Option<(HsaKernel, u32)>,
    k_moe_ep_align: Option<HsaKernel>,
    k_moe_ep_stage2: Option<HsaKernel>,
    k_moe_ep_combine: Option<HsaKernel>,
    /// Spill-free interpreter containing only marked XReduceTwoShot segments.
    k_xreduce_wave_rs: Option<HsaKernel>,
    k_mla_materialize_pack: Option<HsaKernel>,
    k_mla_materialized_prefill: Option<HsaKernel>,
    sparse_mla: Option<amd_sparse_mla::SparseMla>,
    moe_aiter: Option<amd_moe_aiter::MoeAiter>,
    index_tp: Option<amd_index_tp::IndexTp>,
    gemm_lt: Option<amd_gemm_lt::GemmLt>,
    mla_fold: Option<amd_mla_fold::MlaFold>,
    k_xaudit: Option<HsaKernel>,
    k_state_clear: Option<HsaKernel>,
    k_token_capture: Option<HsaKernel>,
    d_token_ring: Option<DeviceMem>,
    /// Task-13: low-rung decode tier ladder, ascending (max_rung, kernel).
    decode_tiers: Vec<(u32, HsaKernel)>,
    k_flash: Option<HsaKernel>,
    /// Dedicated scratch-free gfx950 V2+SV interpreter for pure bf16 MLA-flash segments.
    k_mla_v2_sv_raw: Option<HsaKernel>,
    k_packed_mla_norm: Option<HsaKernel>,
    k_packed_mla_flash: Option<HsaKernel>,
    k_packed_kda: Option<HsaKernel>,
    k_xr_attnres: Option<HsaKernel>,
    packed_prefill_dense: bool,
    packed_prefill_prefill_abi: bool,
    sched_prefill: Sched,
    sched_decode: Sched,
    _modules: Vec<Module>,

    /// Pinned copy of the decode program's instructions, patched in place so
    /// the per-step slice upload is contiguous in PINNED memory.
    h_inst: HsaPinned,
    /// Pinned scalar staging. `hsa_amd_memory_lock` is syscall-class; the
    /// reference driver measured that pinning per step "cost more than the
    /// whole forward pass", so every hot-path transfer uses pre-pinned memory.
    h_scalar: HsaPinned,
    /// Pinned zero page for the counter/cursor re-arm.
    h_zero: HsaPinned,
    /// Pinned staging for a prefill program's whole instruction array.
    h_pf_inst: HsaPinned,
    /// Persistent device and pinned-host storage for ragged packed-prefill metadata. Sized once
    /// for the widest compiled prefill rung; staging performs no allocation.
    d_prefill_spans: DeviceMem,
    d_prefill_parked: DeviceMem,
    h_prefill_meta: HsaPinned,
    prefill_span_capacity: usize,
    prefill_row_capacity: usize,
    packed_prefill: Option<PackedPrefillBinding>,
    d_state_clear: Option<DeviceMem>,
    n_state_clear: u32,
    /// Pristine host copy of each program's instructions. Prefill patching
    /// rebuilds from these every chunk: `c0` changes per chunk, so patches must
    /// not accumulate.
    pf_src: Vec<Vec<DevInst64>>,

    kvrow: Vec<u32>,
    /// KV-append sites whose write row is `i[2]`, not `i[3]` — GLM-5.2's latent
    /// `RmsNorm` half. Empty for every packet that declared its own sites.
    kvrow_i2: Vec<u32>,
    kvrow_span: Option<(usize, usize)>,
    /// LIVE-`kv_len` MLA split policy (`PLOW_MLA_NS_LIVE`): the decode program's
    /// `i[4]` split sites, their contiguous span, the count the emitter baked,
    /// and the count currently resident on the device. `None` when the knob is
    /// off or the program is not a plain dense MLA decode — see
    /// [`derive_mla_nsplit`].
    mla_nsplit: Option<MlaNsplit>,
    t_ids: Option<usize>,
    t_pos: Option<usize>,
    t_kvlen: Option<usize>,
    /// `in.parked`, the per-row SKIP mask (non-zero = park). Present only on a sequence-rows
    /// (batched-decode) blob; `None` means every row always participates.
    t_active: Option<usize>,
    t_logits: Option<usize>,
    max_ctx: usize,

    weights_bound: bool,
    /// Decode batch — the number of sequences one decode dispatch advances.
    ///
    /// Derived from `in.kvlen`, which the compiler sizes at `batch * 4` bytes,
    /// NOT from the decode program's `t`. The two agree on a well-formed blob
    /// and the tensor is the one the kernel actually indexes, so a disagreement
    /// should surface as a bind-time error rather than as every sequence past
    /// the first reading a length nobody wrote.
    batch: usize,
    /// Host mirror of the device tensor-pointer table, so a KV rebase is one
    /// edit + one upload instead of a read-modify-write off the device.
    tens_table: Vec<u8>,
    /// `(tensor index, per-sequence byte stride)` for every `kv.*` buffer.
    ///
    /// The cache is allocated `[batch][kv_head][ring][hd]` (`devgen`
    /// `b.tensor("kv.{l}.k", db * ...)`), so sequence `s`'s block is exactly
    /// `s * bytes/batch` in and a base-pointer shift addresses it exactly.
    /// That is what lets the SINGLE-SEQUENCE prefill program fill any slot: it
    /// writes with the legacy `hh * out_stride + row` formula, which is
    /// sequence 0's block relative to whatever base the pointer table holds.
    kv_slot_stride: Vec<(usize, u64)>,
    /// Which sequence slot the KV pointers are currently rebased onto.
    kv_slot: usize,
    /// `(tensor, per-slot bytes)` for the CARRIED recurrent state — KDA `state`
    /// and `conv_state`, per-slot strided, `blkres` excluded for the reason
    /// [`AmdEngine::begin_slot`] gives. This is what a prefix snapshot copies.
    carried_slot: Vec<(usize, u64)>,
    /// Per-slot snapshot of recurrent state and sliding KV at a prefix boundary.
    prefix_snap: Vec<Option<DeviceMem>>,
    prefix_regions: Option<Vec<prefix::Region>>,
    prefix_rows: Vec<u32>,
    prefix_used: Vec<u64>,
    prefix_tick: u64,
    /// `(alternate bank, legacy bank, bytes)` for the B1 KDA conv-window ping-pong arm.
    kda_conv_bank_pairs: Vec<(u64, u64, u64)>,
    /// Legacy prefill updates only bank 0; a set bit requires one bank0→bank1 mirror before
    /// decode selects a source by absolute-position parity.
    kda_conv_alt_stale: Vec<bool>,
    /// VMM-backed FULL-attention KV, or `None` for the flat allocation.
    ///
    /// The tensor table still holds ONE base per `kv.{l}.{k,v}` and
    /// [`AmdEngine::kv_rebase`] still shifts it by `slot * bytes/batch` — the
    /// base is simply a VA reservation instead of a `hsa_amd_memory_pool_allocate`
    /// result, and physical granules are mapped at each sequence's decode
    /// frontier. No block table, no per-block indirection, no kernel change.
    vmm: Option<VmmKv>,
    shared_prefix: Option<shared_prefix::SharedPrefix>,
    /// `(blocks, i[8])` of the last lm_head found — the fields that differ
    /// between two packets whose op and operands are identical.
    lm_detail: std::cell::RefCell<Option<(u16, [u32; 8], usize, Vec<u16>)>>,
    /// Pristine stream per program, for the scheduling diagnostic above.
    pf_stream: Vec<Vec<packet::dev::StreamEnt>>,

    /// This rank's place in its TP group, or `None` for single-GPU. Drives the
    /// four cross-GPU kernarg fields; the peer bindings it implies were made at
    /// load.
    tp: Option<TpBind>,

    /// Host-side accounting of segmented dispatch: enqueue time proves the host
    /// is NOT in the loop between segments; drain time is the GPU running every
    /// segment back to back.
    pub seg_enq_us: f64,
    pub seg_drain_us: f64,
    pub seg_launches: u64,

    /// A/B CONTROL, not a tuning knob (`PLOW_SEG_WINDOW=0`). Clears
    /// `PlowProgram::seg_ofs`, so the interpreter falls back to scanning each
    /// CU's whole stream and filtering on `seg` — the pre-window behaviour.
    ///
    /// It exists because the two arms must be compared in ONE process against
    /// ONE code object: rebuilding to compare would confound the measurement
    /// with a different build, and a 31 GB weight load per arm prices the
    /// comparison out. Both arms must produce IDENTICAL tokens; if they do not,
    /// the window is wrong and no timing from either arm means anything.
    seg_window: bool,
}

fn discover_lowrung_tiers(hsaco_dir: &Path, object: &str) -> Option<String> {
    let mut tiers = std::fs::read_dir(hsaco_dir)
        .ok()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let width = name.to_str()?.strip_prefix("lowrung")?.parse::<u32>().ok()?;
            let dir = entry.path();
            (width > 0 && dir.join(object).is_file()).then_some((width, dir))
        })
        .collect::<Vec<_>>();
    tiers.sort_by_key(|(width, _)| *width);
    (!tiers.is_empty()).then(|| {
        tiers
            .iter()
            .map(|(width, dir)| format!("{}:{width}", dir.display()))
            .collect::<Vec<_>>()
            .join(",")
    })
}

impl AmdEngine {
    pub fn overlap_evidence(&self, rank: usize) -> AmdOverlapRankEvidence {
        let range = |name: String, address_space, start, len: u64| AmdOwnedRange {
            name,
            address_space,
            start,
            end: start.checked_add(len).unwrap_or(u64::MAX),
        };
        let mut ranges = Vec::with_capacity(self.devp.len() + 3);
        ranges.push(range(
            "d_tens".into(),
            "device",
            self.d_tens.base,
            self.d_tens.len,
        ));
        ranges.extend(
            self.tensor_names
                .iter()
                .zip(&self.devp)
                .filter(|(name, _)| !packet::names::is_checkpoint_weight(name))
                .map(|(name, mem)| range(format!("devp:{name}"), "device", mem.base, mem.len)),
        );
        ranges.push(range(
            "h_scalar".into(),
            "host_pinned",
            self.h_scalar.as_ptr() as usize as u64,
            self.h_scalar.len() as u64,
        ));
        if let Some(mem) = &self.d_trace {
            ranges.push(range("d_trace".into(), "device", mem.base, mem.len));
        }
        let queue_ids = if self.be.queue_count() == 0 {
            Vec::new()
        } else {
            vec![self.be.queue_identity()]
        };
        AmdOverlapRankEvidence {
            rank,
            queue_count: self.be.queue_count(),
            prefill_ranges: ranges.clone(),
            decode_ranges: ranges,
            prefill_queue_ids: queue_ids.clone(),
            decode_queue_ids: queue_ids,
        }
    }

    /// Bring the engine up from a compiled blob and a directory of gfx950 code
    /// objects.
    ///
    /// With `checkpoint`, the model weights are bound by tensor name and the
    /// engine produces real tokens. Without it, every tensor is still allocated
    /// and the schedule still runs at full size — so the TIMING is real and the
    /// TOKENS are not. That mode exists because it isolates dispatch cost from
    /// a multi-minute weight upload, and it must never be mistaken for the
    /// other; the load log says which one you got.
    pub fn load(
        be: Arc<HsaBackend>,
        blob_path: &Path,
        hsaco_dir: &Path,
        checkpoint: Option<&Path>,
    ) -> Result<Self> {
        Self::load_rank(be, blob_path, hsaco_dir, checkpoint, None, None)
    }

    /// Bring up the VMM-backed KV pool, or `None` to keep the flat allocation.
    ///
    /// **This is growth, not paging.** One contiguous VA reservation per
    /// (full layer, K|V) spans the whole `[batch][kvh][max_ctx][hd]` tensor —
    /// exactly the shape `devgen` declares — so the tensor table keeps ONE base
    /// per tensor and [`AmdEngine::kv_rebase`]'s `base + slot·stride` is
    /// unchanged. Physical granules are mapped at each sequence's decode
    /// frontier. There is no block table and no indirection in the hot loop.
    ///
    /// What it buys: a slot's HBM follows its live context instead of
    /// `max_ctx`. On Gemma-4-31B (10 full layers, `kvh_full` 4, `hd_full` 512
    /// bf16) the head window is `max_ctx * 1024 B` and the granule is 2 MiB, so
    /// at `max_ctx` 16384 that is 16 MiB in 8 blocks of 2048 rows each.
    /// MEASURED at B=8, ~1k live context: 75.55 GiB resident vs 83.05 flat —
    /// **7.50 GiB of the 10.0 GiB full-layer reservation reclaimed**, TPOT
    /// unchanged (38.0 vs 39.0 ms). The ratio is `max_ctx / live_ctx`, so it
    /// grows with the emitted context: at 128k the same pool would hold 2.5 GiB
    /// of an 80 GiB tensor.
    ///
    /// Every failure is a warn + `None` — the flat path is always correct, so a
    /// missing `config.json` or a geometry the blob does not match must not
    /// stop the engine loading. Off by default (`--amd-vmm-kv` /
    /// `PLOW_VMM_KV=1`).
    ///
    /// Only FULL-attention `kv.{l}.k`/`.v` are backed. Sliding-window rings are
    /// bounded by `window`, not by context, so they have nothing to grow into,
    /// and fp8 scale tensors are 1/128th the size — both stay flat, which is
    /// also what the CUDA path does.
    fn vmm_bringup(
        be: &Arc<HsaBackend>,
        blob: &DevBlob,
        checkpoint: Option<&Path>,
        batch: usize,
    ) -> Option<VmmKv> {
        if !crate::config::RuntimeConfig::get().amd.vmm_kv {
            return None;
        }
        if !be.has_vmm() {
            tracing::warn!("PLOW_VMM_KV=1 but this ROCr has no hsa_amd_vmem_* — vmm off");
            return None;
        }
        let ckpt = checkpoint?;
        let find = |name: &str| blob.tensors.iter().position(|t| t.name == name);
        let bytes_of = |name: &str| find(name).map(|i| blob.tensors[i].bytes);
        let max_ctx = (bytes_of("in.pos")? / 4) as u32;
        let batch = u32::try_from(batch).ok()?;

        let mut geo = match VmmGeometry::from_config(ckpt, max_ctx, batch) {
            Some(g) => g,
            None => {
                tracing::warn!("vmm off: no usable KV geometry in config.json");
                return None;
            }
        };
        // KV dtype from the blob, not from config.json: the emitter declares
        // `kv.{l}.k_scale` iff that layer's cache is fp8 e4m3 (1 B/elem).
        // Presence is the discriminator — byte size alone is ambiguous
        // (2x ring vs 2x elem).
        geo.elem = match geo.full_layers.first() {
            Some(&l) => match find(&format!("kv.{l}.k_scale")) {
                Some(_) => 1,
                None => 2,
            },
            None => return None,
        };
        // Every full layer's declared bytes must equal the batch-major shape at
        // that elem. A mismatch is geometry drift, and backing a VA range the
        // kernel indexes differently is a silent wrong token.
        for &l in &geo.full_layers {
            for t in ["k", "v"] {
                match bytes_of(&format!("kv.{l}.{t}")) {
                    Some(b) if b == geo.full_tensor_bytes() => {}
                    other => {
                        tracing::warn!(
                            layer = l,
                            declared = ?other,
                            expected = geo.full_tensor_bytes(),
                            "vmm off: full-layer KV bytes mismatch"
                        );
                        return None;
                    }
                }
            }
        }

        // The block is the GROWTH quantum. On ROCr `hsa_amd_vmem_set_access`
        // costs ~3-4 us per CALL and is flat in size (measured, gfx950/ROCm
        // 7.2.4 — `tests/hsa_vmm.rs`), where CUDA measured `cuMemSetAccess` at
        // ~69 us per 2 MiB granule. That 20x is why the AMD default is the
        // granule itself instead of the 64 MiB-class block the CUDA feasibility
        // review settled on: the finest quantum the hardware can map costs
        // nothing extra here, and finer means less HBM held per slot.
        // 0 = query the device granularity (2 MiB measured on gfx950).
        let block_hint = (crate::config::RuntimeConfig::get().amd_vmm_block_mib() as u64) << 20;
        let block_hint = match block_hint {
            0 => VmmOps::granularity(&**be).ok()?,
            b => b,
        };
        match VmmKv::new(Arc::clone(be) as Arc<dyn VmmOps>, geo, block_hint, 0) {
            Ok(mut kv) => {
                kv.enable_block_pool(crate::memory::vmm::kv_pool_cap());
                Some(kv)
            }
            Err(e) => {
                tracing::warn!(error = %e, "vmm off: pool bringup failed");
                None
            }
        }
    }

    /// Bring up ONE RANK of a tensor-parallel group.
    ///
    /// With `tp = None` this is [`AmdEngine::load`] and every path below is the
    /// single-GPU one, bit-for-bit. With `Some`, three things change and nothing
    /// else does:
    ///
    /// 1. weights bind this rank's **shard** (`crate::asset::shard`);
    /// 2. `act.og_tp`/`act.dg_tp` are bound into the **peer region** instead of
    ///    ordinary VRAM, so the row-parallel partials are peer-visible;
    /// 3. the kernarg carries `rank`/`n_gpu`/`xctr`/`peer_scratch`.
    ///
    /// The blob's own declared TP degree must match the group's — see the check
    /// below, which is the difference between a clear refusal and a rank
    /// silently binding a quarter of a weight it needed all of.
    pub fn load_rank(
        be: Arc<HsaBackend>,
        blob_path: &Path,
        hsaco_dir: &Path,
        checkpoint: Option<&Path>,
        tp: Option<TpBind>,
        shared_ckpt: Option<Arc<crate::asset::checkpoint::Checkpoint>>,
    ) -> Result<Self> {
        let t_rank = Instant::now();
        let raw = std::fs::read(blob_path)
            .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
        let arch = EngineDevice::arch(&*be);
        let n_cu_dev = EngineDevice::sm_count(&*be);
        // ACCEPT L2-DOMAIN PLACEMENT AND CHECK IT AT OBJECT LOAD, rather than making the operator
        // assert it by env. scripts/build_gfx942.sh now ships -DPLOW_L2_PLACE_DISPATCH by default
        // and plowc places gfx942 blobs by default, so the env gate had become a broken default:
        // a stock build would emit a placed blob and then refuse to load it. The real guard is
        // stronger than the env var ever was -- `plow_l2_place_dispatch_1` is checked against the
        // OBJECT below, so a genuinely mismatched pairing is still refused, by inspection instead
        // of by assertion. PLOW_L2_PLACE_DISPATCH=1 still works for anyone scripting it.
        let mut blob = DevBlob::parse_l2(&raw, true)?;
        let has_fine_xctr = blob
            .progs
            .iter()
            .flat_map(|p| &p.stream)
            .any(|e| e.flags & SE_XCTR != 0);
        let tp_audit_compact = tp.is_some()
            && crate::config::RuntimeConfig::get().amd.tp_audit_compact
            && !has_fine_xctr;
        if tp_audit_compact {
            let status_id = tp.expect("checked").xstatus_id;
            for p in &mut blob.progs {
                patch_tp_xaudit(&mut p.insts, status_id);
            }
        }
        // WHICH PHASE is L2-placed, not "is anything placed". `Builder::finish` skips placement
        // per PROGRAM when that program is segmented, and AMD prefill always is -- so a normal
        // gfx942 blob has a PLACED DECODE program and UNPLACED prefill ones. Requiring the
        // dispatch axis on every object would then reject the stock build over its prefill
        // objects, which correctly do not carry it (the axis is scoped to the decode rows because
        // a set-wide define deadlocks -- see scripts/build_gfx942.sh).
        //
        // WHICH PROGRAMS ARE DECODE. Everything from `dec_ix` on is a decode rung of the
        // DECODE BATCH LADDER (`PLOW_DECODE_BATCH_LADDER`); everything before it is a prefill
        // bucket. Without a ladder this is `progs.len() - 1` and the split is the one every
        // caller has always used.
        let dec_ix = {
            let pt: Vec<u32> = blob.progs.iter().map(|p| p.t).collect();
            packet::devbuild::decode_rung_lo(&pt)
        };
        validate_decode_dispatch(&blob.progs, dec_ix)?;
        let max_decode_batch = blob.progs[dec_ix..].iter().map(|p| p.t).max().unwrap_or(1);

        // THIS USED TO ASK `p.t == 1` / `p.t > 1`, WHICH IS A BUG A BATCHED BLOB ALREADY HAD.
        // A decode program emitted at `PLOW_DECODE_BATCH=16` has `t == 16`, so it counted as a
        // PREFILL program: its (correct, default-on) L2 placement was attributed to the prefill
        // object, which is not built with the axis, and every batched gfx942 blob was refused
        // at load unless it was re-emitted with PLOW_L2_PLACE=0 — giving up the -12% placement
        // win to work around a misclassification. Splitting at `dec_ix` asks each object about
        // the programs it will actually be handed.
        let decode_l2_placed = blob.progs[dec_ix..].iter().any(|p| p.l2_domains > 0);
        let prefill_l2_placed = blob.progs[..dec_ix].iter().any(|p| p.l2_domains > 0);

        // The blob's n_cu is the grid the schedule was COMPILED for. A device
        // with a different CU count cannot run it: `stream_ofs`/`stream_len` are
        // [n_cu] and workgroup w reads slot w, so a smaller grid silently drops
        // every stream above it and a larger one reads past the table.
        //
        // `--amd-oversub` / PLOW_OVERSUB=1 (expert): accept an OVERSUBSCRIBED
        // grid — blob.n_cu a
        // multiple of the device's CU count — to co-locate several workgroups
        // per CU so one workgroup's gate poll hides behind a sibling's body.
        // The launch grid follows blob.n_cu, so this is only sound when the
        // OBJECT's resource envelope actually fits that many co-resident
        // workgroups (e.g. the occ4 profile at 2/CU: 104 VGPR, 30.7 KB LDS);
        // the persistent kernel SPINS on counters, so a non-resident workgroup
        // is not "slow", it is a DEADLOCK. No occupancy oracle is consulted
        // here — that is why this is env-gated instead of a default. L2-domain
        // placement must be OFF in the blob (the wg->domain map assumes
        // grid == n_cu_dev).
        let oversub_ok = blob.n_cu > n_cu_dev
            && blob.n_cu % n_cu_dev == 0
            && crate::config::RuntimeConfig::get().amd.oversub;
        if blob.n_cu != n_cu_dev && !oversub_ok {
            return Err(RuntimeError::Device(format!(
                "blob compiled for n_cu={} but this device has {n_cu_dev} CUs — \
                 recompile the packet with --n-cu {n_cu_dev} (or, for an oversubscribed \
                 grid on a co-resident object, set PLOW_OVERSUB=1)",
                blob.n_cu
            )));
        }
        if blob.progs.is_empty() {
            return Err(RuntimeError::Device("blob carries no programs".into()));
        }

        // The blob's declared sharding and the caller's group must agree, and
        // the mismatch is refused HERE, before a byte of the checkpoint is read.
        //
        // Without this a sharded blob on the single-GPU path dies ~60 GiB later
        // at the first projection with `SIZE MISMATCH
        // model.layers.0.self_attn.q_proj.weight (blob says 5505024 B,
        // checkpoint has 22020096 B)` — which names the symptom, not the cause,
        // and reads as a corrupt packet rather than as "you asked for one GPU
        // and handed me a four-way shard". The blob knows the answer (`DevTp`,
        // recovered from its own collectives); nobody was asking it.
        //
        // The reverse mismatch is worse and equally refused: an UNSHARDED blob
        // run under a TP group has no `XReduce` at all, so the ranks would each
        // compute the whole layer, never reduce, and produce N identical
        // single-GPU tokens at N times the cost — a "working" run that has
        // silently done nothing parallel.
        let n_gpu = tp.map_or(1, |t| t.n_gpu);
        let rank = tp.map_or(0, |t| t.rank);
        match (blob.tp, tp) {
            (Some(b), _) if b.n_gpu != n_gpu => {
                return Err(RuntimeError::Device(format!(
                    "this packet is SHARDED for tp={} (hidden={}, partial slot {} B) but \
                     this engine is bringing up {n_gpu} rank(s). Every projection in it \
                     is 1/{} wide, so binding it here would fail at the first weight \
                     with a size mismatch. Run it on {} devices, or recompile: \
                     plowc ... --num-gpus {n_gpu}",
                    b.n_gpu, b.hidden, b.slot_bytes, b.n_gpu, b.n_gpu
                )));
            }
            (None, Some(_)) => {
                return Err(RuntimeError::Device(format!(
                    "this packet carries NO collective, so it is compiled for a single \
                     GPU, but a TP group of {n_gpu} was requested. Each rank would run \
                     the whole model and never all-reduce — {n_gpu} identical tokens for \
                     {n_gpu} times the hardware. Recompile: plowc ... --num-gpus {n_gpu}"
                )));
            }
            _ => {}
        }
        if let (Some(b), Some(t)) = (blob.tp, tp) {
            // `dg_tp` is bound at this offset and peers read it at the offset
            // THEY were told. devgen bakes one value into every program's
            // `XReduce.i[2]`; if the host's differs, each rank publishes its
            // `down` partial where no peer looks and the reduction silently sums
            // stale memory.
            if b.slot_bytes != t.slot_b {
                return Err(RuntimeError::Device(format!(
                    "peer layout disagrees with the packet: the host binds partial slot \
                     B at {} B but the packet's XReduce reads it at {} B. Every rank's \
                     `down` partial would land where no peer reads it.",
                    t.slot_b, b.slot_bytes
                )));
            }
            if rank >= n_gpu {
                return Err(RuntimeError::Device(format!(
                    "rank {rank} is outside a group of {n_gpu}"
                )));
            }
        }

        // --- scheduler selection -------------------------------------------
        // Default is the global queue on both phases; it is bit-exact to static
        // and measured faster (the kernel side has GQ beating static prefill by
        // 8.4% on 31B at T=1024). It is downgraded, never silently: no GQ
        // appendix in the blob, or no `_gq` object on disk, and this says so.
        let has_gq = blob.progs.iter().all(|p| !p.gq_stream.is_empty());
        let rt = &crate::config::RuntimeConfig::get().amd;
        let mut sched_prefill = Sched::GlobalQueue;
        let mut sched_decode = Sched::GlobalQueue;
        if let Some(v) = rt.global_queue.as_deref() {
            let s = if v != "0" {
                Sched::GlobalQueue
            } else {
                Sched::Static
            };
            sched_prefill = s;
            sched_decode = s;
        }
        if rt.static_both {
            sched_prefill = Sched::Static;
            sched_decode = Sched::Static;
        }
        if rt.static_prefill {
            sched_prefill = Sched::Static;
        }
        if rt.static_decode {
            sched_decode = Sched::Static;
        }
        if !has_gq && (sched_prefill == Sched::GlobalQueue || sched_decode == Sched::GlobalQueue) {
            tracing::info!("blob carries no GQ appendix — both phases fall back to static");
            sched_prefill = Sched::Static;
            sched_decode = Sched::Static;
        }

        let variant = Variant::detect(&blob.progs);
        // Which MLA/MoE-prefill object the PREFILL phase needs — scanned from
        // every program (the prefill buckets carry these opcodes, the decode
        // program never does), so a bucket-only decode packet stays `None` and
        // a whole-layer GLM/Kimi/DeepSeek prefill packet selects the object
        // that actually has the arms instead of silently falling back to
        // `interp_prefill{,_gq}.elf`, whose `default:` case does not trap.
        let prefill_arm = PrefillArm::detect(&blob.progs);

        // THE WIDEST GEMV EACH OBJECT WILL BE HANDED, split the way the objects
        // are. The decode program runs on the decode object; every prefill
        // bucket runs on the prefill object OR the flash object (a flash segment
        // is a prefill segment), so both of those must cover the prefill maximum.
        // Split rather than one global max because the buckets are compiled
        // independently: prefill and flash take `op_gemm.h`'s default MM=1 and
        // legitimately serve the M=1 lm_head GEMV, and folding a batched decode's
        // M into their requirement would refuse them for work they never do.
        // `dec_ix` (defined above) is the split, NOT `progs.len() - 1`: with a DECODE BATCH
        // LADDER the last four programs are also decode, and putting them in the prefill half
        // asks the prefill object to cover a batched decode's GEMV — which it legitimately was
        // not built for, so a laddered blob refused itself at load with "widest GEMV asks for
        // M=8 ... interp_prefill_fp8_gq.elf was compiled PLOW_GEMV_MM=1". Loudly, which is the
        // guard working; the other shape of that mistake is rows 1..7 left STALE with no fault.
        let need_m_decode = required_gemv_m(&blob.progs[dec_ix..]);
        let need_m_prefill = required_gemv_m(&blob.progs[..dec_ix]);
        // Split the same way as the GEMV bucket, and for the same reason: the K3/KDA arms are in
        // BOTH buckets (a K3 layer runs the same graph at T=1 and T>1), so each object has to be
        // asked about the phase it actually serves rather than about the blob as a whole.
        let need_k3_decode = required_k3_op(&blob.progs[dec_ix..]);
        let need_k3_prefill = required_k3_op(&blob.progs[..dec_ix]);
        let need_qwen_decode = required_qwen_gdn_op(&blob.progs[dec_ix..]);
        let need_qwen_prefill = required_qwen_gdn_op(&blob.progs[..dec_ix]);
        let need_kda_chunk_decode = required_kda_chunk(&blob.progs[dec_ix..]);
        let need_kda_chunk_prefill = required_kda_chunk(&blob.progs[..dec_ix]);
        let need_kda_decode_fused = first_op_in(&blob.progs[dec_ix..], &[DevOp::KdaDecodeFused]);
        let need_decode_mla_segments = blob.progs[dec_ix..].iter().try_fold(false, |need, p| {
            Ok::<_, RuntimeError>(
                need || decode_segment_kinds(p)?
                    .iter()
                    .any(|kind| matches!(kind, DecodeSegmentKind::MlaAttention)),
            )
        })?;
        let need_grouped_moe = blob.progs[dec_ix..].iter().try_fold(false, |need, p| {
            Ok::<_, RuntimeError>(
                need || decode_segment_kinds(p)?
                    .iter()
                    .any(|kind| matches!(kind, DecodeSegmentKind::GroupedMoeMxfp4 { .. })),
            )
        })?;
        let graph_phase_segments = graph_phase_xreduce_segments(
            blob_path,
            &blob.progs,
            dec_ix,
            crate::config::RuntimeConfig::get().amd.phase_objects,
        )?;
        let need_graph_phase_xreduce = graph_phase_segments.iter().any(|s| !s.is_empty());
        let need_xreduce_wave_rs = need_graph_phase_xreduce
            || blob.progs[..dec_ix].iter().any(|p| {
                p.stream
                    .iter()
                    .any(|e| e.flags & packet::dev::SE_XR_WAVE_RS != 0)
            });
        let marked_op = |p: &DevProg, op: DevOp| {
            p.stream.iter().any(|e| {
                e.flags & packet::dev::SE_KDA_INTRA_WAVE_ITEMS != 0
                    && p.insts.get(e.inst as usize).map(|d| d.op) == Some(op as u16)
            })
        };
        let need_kda_intra_wave_items = blob.progs[..dec_ix]
            .iter()
            .any(|p| marked_op(p, DevOp::KdaChunkIntra));
        let need_kda_carry_regstate = blob.progs[..dec_ix]
            .iter()
            .any(|p| marked_op(p, DevOp::KdaChunkCarry));
        let marked_wu = |p: &DevProg, keys: bool| {
            p.stream.iter().any(|e| {
                e.flags & packet::dev::SE_KDA_WU_LEAN != 0
                    && p.insts
                        .get(e.inst as usize)
                        .is_some_and(|d| d.op == DevOp::KdaChunkWu as u16 && (d.i[5] == 1) == keys)
            })
        };
        let need_kda_wu_lean = blob.progs[..dec_ix].iter().any(|p| marked_wu(p, false));
        let need_kda_carry_keyfeed = blob.progs[..dec_ix].iter().any(|p| marked_wu(p, true));
        if let Some(p) = blob.progs[..dec_ix]
            .iter()
            .find(|p| p.insts.iter().any(|i| i.op == DevOp::KdaDecodeFused as u16))
        {
            return Err(RuntimeError::Device(format!(
                "KdaDecodeFused is decode-only, but prefill program T={} dispatches it",
                p.t
            )));
        }
        let need_a4w4_decode = required_moe_pf_a4w4(&blob.progs[dec_ix..]);
        let need_moe_pf_atomic_decode = required_moe_pf_accum(&blob.progs[dec_ix..], 4);
        let need_moe_pf_det_decode = required_moe_pf_accum(&blob.progs[dec_ix..], 5);
        let need_kda_conv_step_db = required_kda_conv_step_db(&blob.progs[dec_ix..]);
        let legacy_kda_decode = first_op_in(&blob.progs[dec_ix..], KDA_CONV_STEP_DB_REPLACED_OPS);
        if let Some(p) = blob.progs[..dec_ix].iter().find(|p| {
            p.insts
                .iter()
                .any(|i| i.op == DevOp::KdaConvStateStepG as u16)
        }) {
            return Err(RuntimeError::Device(format!(
                "KdaConvStateStepG is B1 decode-only, but prefill program T={} dispatches it",
                p.t
            )));
        }
        if let Some(p) = blob.progs[dec_ix..].iter().find(|p| {
            p.t != 1
                && p.insts
                    .iter()
                    .any(|i| i.op == DevOp::KdaConvStateStepG as u16)
        }) {
            return Err(RuntimeError::Device(format!(
                "KdaConvStateStepG is B1-only, but decode program T={} dispatches it",
                p.t
            )));
        }
        // Gemma-4 MoE, both halves, per phase. Same silent-NOP argument as K3 above.
        let need_gm_decode = first_op_in(&blob.progs[dec_ix..], MOE_GEMMA_OPS);
        let need_gm_prefill = first_op_in(&blob.progs[..dec_ix], MOE_GEMMA_OPS);
        let need_gmpf_decode = first_op_in(&blob.progs[dec_ix..], MOE_GEMMA_PF_OPS);
        let need_gmpf_prefill = first_op_in(&blob.progs[..dec_ix], MOE_GEMMA_PF_OPS);
        // The KV-encoding SWAP, split per phase for the same reason: the decode object carries
        // FLASH_*_DECODE and the prefill object FLASH_*_PREFILL, so asking each about the blob as
        // a whole would refuse a decode object for a prefill opcode it never runs.
        let need_fp8kv_decode = required_kv_op(&blob.progs[dec_ix..], FP8_KV_OPS);
        let need_fp8kv_prefill = required_kv_op(&blob.progs[..dec_ix], FP8_KV_OPS);
        let need_bf16kv_decode = required_kv_op(&blob.progs[dec_ix..], BF16_KV_OPS);
        let need_bf16kv_prefill = required_kv_op(&blob.progs[..dec_ix], BF16_KV_OPS);
        // Will derive_segments route any FlashMlaPrefill segment to the flash object?
        // Then that object MUST carry the V2 arm — the dispatch default is a silent skip.
        let need_mla_v2 = mla_pf_v2_enabled()
            && blob.progs[..dec_ix].iter().any(|p| {
                p.t >= 2048
                    && p.insts
                        .iter()
                        .any(|i| i.op == DevOp::FlashMlaPrefill as u16)
            });
        let need_mla_v2_fp8 = mla_pf_v2_enabled()
            && blob.progs[..dec_ix].iter().any(|p| {
                p.t >= 2048
                    && p.insts
                        .iter()
                        .any(|i| i.op == DevOp::FlashMlaPrefillFp8 as u16)
            });
        let need_moe_stage2_lean = blob.progs[..dec_ix]
            .iter()
            .any(has_moe_stage2_mxfp4_segment);
        let need_moe_ep = blob.progs[..dec_ix].iter().any(has_moe_prefill_ep);
        let need_moe_stage1_lean = need_moe_ep
            || blob.progs[..dec_ix]
                .iter()
                .any(has_moe_stage1_mxfp4_segment);
        let need_moe_combine_lean = blob.progs[..dec_ix].iter().any(has_moe_combine_segment);
        let need_attn_res_f32mix = blob.progs[..dec_ix]
            .iter()
            .any(|p| p.insts.iter().any(lean_attn_res_f32mix_inst64));
        if need_moe_ep {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "replicated-input MoE EP requires gfx950 specialist objects, but this device is {arch}"
                )));
            }
            let bind = tp.ok_or_else(|| {
                RuntimeError::Device(
                    "replicated-input MoE EP packet requires a tensor-parallel binding".into(),
                )
            })?;
            if blob.progs[..dec_ix]
                .iter()
                .flat_map(|p| &p.insts)
                .filter_map(moe_ep_degree)
                .any(|degree| degree != bind.n_gpu)
            {
                return Err(RuntimeError::Device(format!(
                    "replicated-input MoE EP packet topology does not match TP{}",
                    bind.n_gpu
                )));
            }
            let extra =
                moe_prefill_ep_extra_bytes(&blob.progs[..dec_ix], &blob.tensors, bind.n_gpu)?;
            let allowed = crate::config::RuntimeConfig::get()
                .amd
                .moe_prefill_ep_max_extra_bytes
                .ok_or_else(|| RuntimeError::Device(format!(
                    "replicated-input MoE EP requires {extra} additional resident bytes per rank for decode-safe companion weights; set PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES to an audited capacity >= that value"
                )))?;
            if extra > allowed {
                return Err(RuntimeError::Device(format!(
                    "replicated-input MoE EP requires {extra} additional resident bytes per rank, exceeding the audited {allowed}-byte allowance"
                )));
            }
            tracing::info!(
                extra_resident_bytes_per_rank = extra,
                "accepted decode-safe MoE EP companion-weight budget"
            );
        }
        let need_mla_materialized = blob.progs[..dec_ix].iter().any(|p| {
            p.insts.iter().any(|d| {
                d.op == DevOp::MlaMaterializePack as u16
                    || d.op == DevOp::FlashMlaMaterializedPrefill as u16
            })
        });
        if need_mla_materialized && arch != "gfx950" {
            return Err(RuntimeError::Device(format!(
                "materialized MLA prefill requires gfx950, but this device is {arch}"
            )));
        }
        let has_mla_fold = |p: &DevProg| p.insts.iter().any(amd_mla_fold::native);
        let use_mla_fold = blob.progs.iter().any(has_mla_fold);
        if use_mla_fold
            && (arch != "gfx942" || tp.is_none_or(|t| t.n_gpu != 8)
                || blob.progs[dec_ix..].iter().any(has_mla_fold))
        {
            return Err(RuntimeError::Device(
                "native MLA fold requires gfx942 TP8 prefill".into(),
            ));
        }
        let has_gemm_lt = |p: &DevProg| p.insts.iter().any(|d| d.op == DevOp::GemmLtPf as u16);
        let use_gemm_lt = blob.progs.iter().any(has_gemm_lt);
        if use_gemm_lt
            && (arch != "gfx942"
                || tp.is_none_or(|b| b.n_gpu != 8)
                || blob.progs.iter().enumerate().any(|(ix, p)| {
                    p.insts.iter().any(|d| {
                        d.op == DevOp::GemmLtPf as u16 && d.i[3] != u32::from(ix >= dec_ix)
                    })
                }))
        {
            return Err(RuntimeError::Device(
                "hipBLASLt projection requires gfx942 TP8 and a matching prefill/decode mode"
                    .into(),
            ));
        }
        let has_index_tp = |p: &DevProg| p.insts.iter().any(|d| d.op == DevOp::IndexTpPf as u16);
        let use_index_tp = blob.progs.iter().any(has_index_tp);
        if use_index_tp
            && (arch != "gfx942"
                || tp.is_none_or(|b| b.n_gpu != 8)
                || blob.progs[dec_ix..].iter().any(has_index_tp)
                || !crate::device::Backend::peer(&*be).is_some_and(|p| p.peer_host_writable()))
        {
            return Err(RuntimeError::Device(
                "TP indexer requires gfx942 TP8 prefill with host-mapped peer status".into(),
            ));
        }
        let has_moe_aiter =
            |p: &DevProg| p.insts.iter().any(|d| d.op == DevOp::MoeAiterFp8Pf as u16);
        let use_moe_aiter = blob.progs.iter().any(has_moe_aiter);
        let resident_tables = amd_moe_aiter::resident_tables(&blob.progs, &blob.tensors)?;
        let use_resident_moe = resident_tables.iter().any(Option::is_some);
        if use_moe_aiter && (arch != "gfx942" || tp.is_none_or(|b| b.n_gpu != 8)) {
            return Err(RuntimeError::Device("AITER MoE requires gfx942 TP8".into()));
        }
        let use_sparse_mla = crate::config::RuntimeConfig::get().amd.mla_pf_aiter;
        check_sparse_fp8_packet(&blob.progs, &blob.tensors, &arch)?;
        check_dsa_select_local(&blob.progs, &blob.tensors, dec_ix, &arch,
                               tp.is_some_and(|t| t.n_gpu == 8))?;
        if use_sparse_mla && arch != "gfx942" {
            return Err(RuntimeError::Device(
                "sparse AITER MLA requires gfx942".into(),
            ));
        }
        let need_xr_attnres = blob.progs[..dec_ix]
            .iter()
            .any(|p| p.insts.iter().any(xreduce_attnres_inst));

        // --- code objects ---------------------------------------------------
        // Resolve the symbol immediately after each load: the HSA backend
        // creates a fresh executable per load, so a later load makes an earlier
        // handle unreachable even though its resolved kernel object stays valid.
        //
        // The packet's `build.json` says which arms the PREFILL object must have
        // been compiled with; `check_prefill_object` refuses a pairing that does
        // not, because the AMD `default:` writes nothing rather than trapping.
        // Read once, here, so a broken manifest fails before any object is
        // loaded rather than between two of them.
        let requires = build_requires(blob_path)?;
        let packet_decode_requires = packet_decode_arm_requirements(&blob.progs[dec_ix..]);
        let packet_prefill_requires = packet_prefill_arm_requirements(&blob.progs[..dec_ix]);
        let mut modules = Vec::new();
        let mut k_xaudit = None;
        let mut k_state_clear = None;
        let mut k_token_capture = None;
        let mut prefill_moe_align_bm64 = false;
        let state_clear_device = crate::config::RuntimeConfig::get().amd.state_clear_device;
        // Task-13 per-rung co-load: an optional SECOND decode object for the low
        // rungs (PLOW_HSACO_LOWRUNG=<dir>), so rung 1-2 traffic runs the tight
        // single-slot codegen while wide rungs keep the batched object. Same
        // resolution, same pairing checks, different directory.
        let lowrung_dir = crate::config::RuntimeConfig::get()
            .amd
            .hsaco_lowrung
            .clone()
            // DISCOVERED, not required. `scripts/build_gfx942.sh PLOW_DECODE_TIERS=…`
            // writes the matched objects to `<objdir>/lowrung<w>/`, and until now the only
            // thing that turned them into a `dir:w` spec was a loop in
            // `scripts/glm53_serve_inner.sh` — so every OTHER caller (bringup_gate.sh, a
            // bare `plowrt serve`, anyone following docs/BUILD.md) silently served without
            // them even when they sat right there next to the object it did load.
            //
            // That is not a small default. `PLOW_GEMV_MM` is a compiled CEILING, so a
            // rung-1 packet running a width-4 object computes four rows per decode GEMV and
            // discards three: building the tiers and naming them is worth +24.9% output
            // tok/s and -20.7% TPOT at concurrency 1 on GLM-5.3 TP8
            // (docs/amd/tp-bringup-mi300x.md §7b), and the campaign that froze that
            // baseline had `hsaco_lowrung: None`.
            //
            // An explicit setting still wins, and an empty one still means "off" — this
            // only fills in the value the directory layout already implies.
            .or_else(|| {
                let name = object_name(Phase::Decode, variant, prefill_arm, sched_decode);
                let found = discover_lowrung_tiers(hsaco_dir, &name);
                // LOGGED, because a derived decision is the one thing an env dump cannot show.
                // `serve_replay` records what the operator set; this is what the layout decided
                // for them, and the difference between the two is 24.9% output tok/s.
                match &found {
                    Some(spec) => tracing::info!(
                        spec = %spec,
                        "decode tiers discovered next to the object dir (PLOW_HSACO_LOWRUNG unset)"
                    ),
                    None => tracing::info!(
                        dir = %hsaco_dir.display(),
                        "no decode tiers: no matching {name} in lowrung<w>/ beside the object dir. A packet whose \
                         decode ladder is wider than 1 runs the wide object's body at every rung \
                         — build them with scripts/build_gfx942.sh PLOW_DECODE_TIERS=…"
                    ),
                }
                found
            })
            .filter(|d| !d.is_empty());
        let mut dense_prefill_object = false;
        // ARMED-ness of the decode object, for the one status line at the end of load.
        // Every decode object opened must carry it, low rungs included: a ladder whose
        // rungs disagree runs the hierarchy on some ticks and not others.
        let mut decode_objects = 0usize;
        let mut decode_objects_gate_hier = 0usize;
        let mut dense_flash_object = false;
        type LK = (HsaKernel, bool);
        let mut load_one_in = |phase: Phase,
                               sched: Sched,
                               dir: &Path,
                               gemv_need: Option<u32>|
         -> Result<LK> {
            let name = object_name(phase, variant, prefill_arm, sched);
            let path = dir.join(&name);
            // WHICH OBJECT, BY NAME, AT INFO — and this line is not cosmetic.
            //
            // `variant` and `prefill_arm` are DETECTED from the packet's opcodes
            // ([`Variant::detect`]), so which object a run opens is a DERIVED fact, not a build
            // choice, and nothing printed it. `Variant::detect` matches `GemvFp8` and the three
            // fp8-KV flash ops; it does NOT match the block-scaled `*Fp8Blk` family, so a
            // GLM-5.2 packet — every one of whose fp8 kernels is block-scaled — detects as
            // `Bf16` and runs on `interp_decode_gq.elf`. That is correct (the `*Fp8Blk` cases in
            // interp.hip are outside `#if PLOW_FP8`, deliberately), but it is the opposite of
            // what the object names suggest, and a whole campaign of decode-kernel arms was
            // built into `interp_decode_fp8_gq.elf` and measured against a run that never
            // opened it. Its ablation — delete the kernel entirely — read as "the packet costs
            // the same", which was taken as evidence for a protocol floor that does not exist.
            // Rebuilt into the object this line names, the same ablation moves the token by
            // 11.8% (perf-data/plow-gfx942/glm52-packet-protocol-xcd.md).
            tracing::info!(object = %name, path = %path.display(), ?phase, ?variant, ?prefill_arm, ?sched,
                           "code object");
            let image = std::fs::read(&path).map_err(|e| {
                if phase == Phase::Prefill && prefill_arm != PrefillArm::None {
                    RuntimeError::Device(format!(
                        "code object {}: {e} — this packet's prefill programs contain {} \
                         opcodes, which requires {name} in {}, and it is not there. Build it \
                         (scripts/build_gfx950.sh with {}), or serve a packet that does not \
                         need it; falling back to an object without the arms is the AMD \
                         `default:`-does-not-trap bug this check exists to prevent.",
                        path.display(),
                        match prefill_arm {
                            PrefillArm::MlaMoe => "MLA+MoE prefill",
                            PrefillArm::Mla => "MLA prefill",
                            PrefillArm::K3 => "Kimi-K3 block",
                            PrefillArm::K3Moe => "Kimi-K3 block + grouped MoE prefill",
                            PrefillArm::K3MoeA4w4 => "Kimi-K3 block + grouped A4W4 MoE prefill",
                            PrefillArm::None => unreachable!(),
                        },
                        hsaco_dir.display(),
                        match prefill_arm {
                            PrefillArm::MlaMoe => "PLOW_MOE_PREFILL=1",
                            PrefillArm::Mla => "PLOW_MLA_PREFILL=1",
                            PrefillArm::K3 => "PLOW_K3=1 PLOW_MLA_PREFILL=1",
                            PrefillArm::K3Moe => {
                                "PLOW_K3=1 PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1"
                            }
                            PrefillArm::K3MoeA4w4 => {
                                "PLOW_K3=1 PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 \
                                 PLOW_MOE_PF_A4W4=1 PLOW_MXFP4=1"
                            }
                            PrefillArm::None => unreachable!(),
                        },
                    ))
                } else {
                    RuntimeError::Device(format!("code object {}: {e}", path.display()))
                }
            })?;
            check_interpreter_waves(
                elf_symbol_u32(&image, "plow_geom_PLOW_WG_WAVES"), phase, &path,
            )?;
            let syms = elf_symbol_names(&image);
            if phase == Phase::Prefill {
                prefill_moe_align_bm64 = syms.contains(&"plow_moe_align_bm64_1");
                if blob.progs[..dec_ix].iter().any(has_moe_aiter) && !prefill_moe_align_bm64 {
                    return Err(RuntimeError::Device(
                        "AITER MoE requires an MPF_BM=64 align object".into(),
                    ));
                }
            }
            check_gate_hier_object(&syms, &path, phase, sched)?;
            if phase == Phase::Decode {
                decode_objects += 1;
                decode_objects_gate_hier += usize::from(syms.contains(&GATE_HIER_SYM));
            }
            let packed_prefill_abi = syms.contains(&PACKED_PREFILL_ABI_SYM);
            let dense = syms.contains(&"plow_packed_prefill_dense_consumers_1");
            match phase {
                Phase::Prefill => dense_prefill_object = dense,
                Phase::Flash => dense_flash_object = dense,
                _ => {}
            }
            if let (Phase::Prefill, Some(req)) = (phase, requires.as_ref()) {
                check_prefill_object(&syms, &path, req)?;
            }
            if phase == Phase::Prefill {
                check_prefill_object(&syms, &path, &packet_prefill_requires)?;
            }
            if let (Phase::Decode, Some(req)) = (phase, requires.as_ref()) {
                check_decode_object(
                    &syms,
                    &path,
                    req,
                    need_moe_pf_atomic_decode,
                    need_moe_pf_det_decode,
                )?;
            }
            if phase == Phase::Decode {
                check_decode_object(
                    &syms,
                    &path,
                    &packet_decode_requires,
                    need_moe_pf_atomic_decode,
                    need_moe_pf_det_decode,
                )?;
            }
            if let (Phase::Decode, true, Some(bind)) = (phase, syms.contains(&XR_TAGGED_SYM), tp) {
                // A tagged one-shot object spins on data tags instead of the xctr gate;
                // it finds its region from the status id every collective carries and
                // needs a blob whose XReduce packets keep the parity/width contract, at
                // most 8 peers (one tag word per rank), and the compact TP audit (the
                // exact copy audit reads the counter gate the tagged arm never bumps).
                // Refuse here rather than trap on the device.
                if bind.n_gpu > 8 {
                    return Err(RuntimeError::Device(format!(
                        "{} ({XR_TAGGED_SYM}): tagged one-shot XReduce supports at most 8 \
                         ranks, got TP{}",
                        path.display(),
                        bind.n_gpu
                    )));
                }
                if !crate::config::RuntimeConfig::get().amd.tp_audit_compact {
                    return Err(RuntimeError::Device(format!(
                        "{} ({XR_TAGGED_SYM}): tagged one-shot XReduce requires the compact TP \
                         audit; unset PLOW_TP_AUDIT_COMPACT=0 or build with PLOW_XR_TAGGED=OFF",
                        path.display()
                    )));
                }
                super::amd_tp::check_xr_tagged_blob(&blob.progs, blob.tp.map_or(0, |b| b.hidden))
                    .map_err(|e| {
                    RuntimeError::Device(format!("{} ({XR_TAGGED_SYM}): {e}", path.display()))
                })?;
            }
            // The W_ofold fusion's arm lives in the FLASH object (the V2 MLA-prefill arm's
            // ofold epilogue), and it additionally needs the V2 routing itself: without
            // PLOW_MLA_PF_V2=1 the MLA segments run on the 8-wave prefill kernel, which
            // ignores packet i[6] and leaves unnormalized f32 partials for the fused GEMM
            // to read as bf16 — finite, fluent, wrong. Both must hold to serve this blob.
            if let (Phase::Flash, Some(req)) = (phase, requires.as_ref()) {
                if req.iter().any(|r| r == "PLOW_GLM_OFOLD=1") {
                    if !crate::config::RuntimeConfig::get().amd.mla_pf_v2 {
                        return Err(RuntimeError::Device(
                            "this packet fuses MlaMergeFold+o_proj (W_ofold) and REQUIRES the \
                             V2 MLA-prefill routing: serve with PLOW_MLA_PF_V2=1, or emit \
                             without PLOW_GLM_OFOLD"
                                .into(),
                        ));
                    }
                    if !syms.iter().any(|s| s.contains("plow_glm_ofold_arm")) {
                        return Err(RuntimeError::Device(format!(
                            "packet/object MISMATCH: this packet requires PLOW_GLM_OFOLD=1 \
                             but {} lacks the ofold-aware V2 arm (plow_glm_ofold_arm) — its \
                             flash would write unnormalized f32 partials that the fused \
                             o-GEMM reads as bf16 garbage. Rebuild the flash object from a \
                             tree that carries the arm.",
                            path.display()
                        )));
                    }
                }
                // The DSA SPARSE V2 prefill arm (op 51 t[7] = the per-64-query-tile union
                // table). Same two-part check as ofold above and for the same reason, but a
                // strictly worse failure if it is skipped: the fallback is not garbage that a
                // reader would notice, it is FULL CAUSAL ATTENTION on a model trained sparse —
                // a finite, fluent answer to a different question. Both halves must hold.
                //
                //   * the V2 ROUTING, because without it the MLA segments run on the 8-wave
                //     prefill kernel, whose `exec_flash_mla_prefill` never reads t[7] on op 51
                //     (it now traps there instead, but refusing at load says why).
                //   * the ARM ITSELF, because PLOW_DSA_PF_ARM is off by default: the gathered
                //     instantiation raised the flash object's spill 98 -> 287 for every blob,
                //     so a stock object genuinely has no gathered body to dispatch.
                check_dsa_pf_arm(
                    &syms,
                    &path,
                    req,
                    crate::config::RuntimeConfig::get().amd.mla_pf_v2,
                )?;
                check_mla_nope_arm(&syms, &path, req)?;
            }
            // The standalone flash object contains only flash-prefill arms. Model and
            // GEMV capability checks belong to the general prefill/decode objects; applying
            // them here rejects the intentionally model-neutral flash object.
            if phase != Phase::Flash {
                let need = match phase {
                    Phase::Decode => gemv_need.unwrap_or(need_m_decode),
                    Phase::Prefill => need_m_prefill,
                    Phase::Flash => unreachable!(),
                };
                check_gemv_capacity(&syms, &path, need)?;
            }
            if phase == Phase::Decode {
                check_xargmax_capacity(&syms, &path, gemv_need.unwrap_or(max_decode_batch))?;
                check_dec_stage_capacity(&image, &path, &blob.progs[dec_ix..])?;
                check_dsa_decode_batch(&syms, &path, &blob.progs[dec_ix..], arch == "gfx942")?;
                check_sparse_fp8_object(&syms, &path, &blob.progs[dec_ix..], true)?;
            } else if phase == Phase::Flash {
                check_sparse_fp8_object(&syms, &path, &blob.progs[..dec_ix], false)?;
            }
            // Whether this object carries the PLOW_K3 arms the packet dispatches. Refused here
            // rather than tolerated, because AMD's dispatch default is a silent NOP: the run
            // would otherwise complete on untouched buffers instead of failing.
            let need_k3 = match phase {
                Phase::Decode => need_k3_decode,
                Phase::Prefill => need_k3_prefill,
                Phase::Flash => None,
            };
            check_k3_arms(&syms, &path, need_k3)?;
            check_qwen_gdn_arms(
                &syms,
                &path,
                match phase {
                    Phase::Decode => need_qwen_decode,
                    Phase::Prefill => need_qwen_prefill,
                    Phase::Flash => None,
                },
            )?;
            if phase != Phase::Flash {
                let phase_progs = if phase == Phase::Decode {
                    &blob.progs[dec_ix..]
                } else {
                    &blob.progs[..dec_ix]
                };
                check_compiled_opcode_markers(&syms, &path, phase_progs)?;
                check_materialized_residual_input(&syms, &path, phase_progs)?;
            }
            let need_chunk = match phase {
                Phase::Decode => need_kda_chunk_decode,
                Phase::Prefill => need_kda_chunk_prefill,
                Phase::Flash => None,
            };
            check_kda_chunk(&syms, &path, need_chunk)?;
            if phase == Phase::Decode {
                check_moe_pf_a4w4(&syms, &path, need_a4w4_decode)?;
                check_kda_conv_step_db(&syms, &path, need_kda_conv_step_db, legacy_kda_decode)?;
            }
            let (need_gm, need_gmpf) = match phase {
                Phase::Decode => (need_gm_decode, need_gmpf_decode),
                Phase::Prefill => (need_gm_prefill, need_gmpf_prefill),
                Phase::Flash => (None, None),
            };
            check_moe_gemma_arms(&syms, &path, need_gm, need_gmpf)?;
            if phase == Phase::Flash && need_mla_v2 && !syms.contains(&MLA_PF_V2_SYM) {
                return Err(RuntimeError::Device(format!(
                    "PLOW_MLA_PF_V2=1 routes FlashMlaPrefill segments to {}, but it was \
                     compiled without the V2 arm (no `{MLA_PF_V2_SYM}`). The dispatch default \
                     writes NOTHING, so those packets would silently skip. Rebuild the flash \
                     object (`-DPLOW_MLA_PF_V2_ARM=ON` in runtime/CMakeLists.txt; \
                     scripts/build_gfx942.sh enables it) or unset PLOW_MLA_PF_V2.",
                    path.display()
                )));
            }
            if phase == Phase::Flash && need_mla_v2_fp8 && !syms.contains(&MLA_PF_V2_FP8_SYM) {
                return Err(RuntimeError::Device(format!(
                    "PLOW_MLA_PF_V2=1 routes FlashMlaPrefillFp8 segments to {}, but it was \
                     compiled without the fp8 V2 arm (no `{MLA_PF_V2_FP8_SYM}`). Rebuild the \
                     fp8-KV flash object or unset PLOW_MLA_PF_V2.",
                    path.display()
                )));
            }
            // L2-PLACED BLOB vs OBJECT. This is the guard the PLOW_L2_PLACE_DISPATCH env var used
            // to stand in for, moved to where it can be VERIFIED. A placed program's `seg` is an
            // L2 domain, not a wave class, so an object built without the axis would run every
            // packet on the wrong domain -- plausible output, inverted locality, no error.
            let phase_l2_placed = match phase {
                Phase::Decode => decode_l2_placed,
                Phase::Prefill | Phase::Flash => prefill_l2_placed,
            };
            if phase_l2_placed && !syms.contains(&L2_DISPATCH_SYM) {
                return Err(RuntimeError::Device(l2_pairing_refusal(&path, phase)));
            }
            // Whether this object's KV ENCODING matches the packet's. Both directions — the axis
            // is a swap, so each object is missing an arm the other has.
            let (need_fp8, need_bf16) = match phase {
                Phase::Decode => (need_fp8kv_decode, need_bf16kv_decode),
                Phase::Prefill | Phase::Flash => (need_fp8kv_prefill, need_bf16kv_prefill),
            };
            check_kv_encoding(&syms, &path, need_fp8, need_bf16)?;
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let m = EngineDevice::module_load(&*be, &image).map_err(|e| {
                RuntimeError::Device(format!(
                    "{name}: {e} — a BUNDLED object gives exactly this; was it \
                     run through clang-offload-bundler --unbundle?"
                ))
            })?;
            let sym = symbol_name(phase, sched, &arch);
            let k = EngineDevice::get_function(&*be, &m, &sym)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {sym}: {e}")))?;
            if phase == Phase::Decode && tp_audit_compact && k_xaudit.is_none() {
                k_xaudit = Some(
                    EngineDevice::get_function(&*be, &m, "plow_xctr_audit").map_err(|e| {
                        RuntimeError::Device(format!(
                            "{name}: compact TP audit requested but plow_xctr_audit is absent: {e}"
                        ))
                    })?,
                );
            }
            if phase == Phase::Decode && state_clear_device && k_state_clear.is_none() {
                k_state_clear = Some(
                    EngineDevice::get_function(&*be, &m, "plow_state_clear").map_err(|e| {
                        RuntimeError::Device(format!(
                            "{name}: device recurrent-state clear requested but \
                                 plow_state_clear is absent: {e}. Rebuild the decode object"
                        ))
                    })?,
                );
            }
            if phase == Phase::Decode
                && k_token_capture.is_none()
                && syms.contains(&"plow_token_capture")
            {
                k_token_capture = Some(EngineDevice::get_function(&*be, &m, "plow_token_capture")?);
            }
            // STALE-OBJECT REFUSAL. An object's kernarg segment is its explicit
            // args, 8-aligned, plus the COv5 implicit block — a FIXED 256 B tail
            // that hipcc emits only when the kernel uses a hidden arg (the flash
            // object does not, and reports the bare struct size; prefill and
            // decode do, and report that + 256). Those are the only two legal
            // values for a kernel whose one argument is `PlowProgram`, and both
            // are DERIVED from `size_of::<DevProgram>()` below rather than
            // written as literals — the struct has grown twice already
            // (128 -> 136 with `seg_ofs`, 136 -> 144 with `l2_domains`), and a
            // literal here goes stale exactly when it is most needed.
            //
            // This matters because the launcher writes that implicit block at
            // OUR `size_of::<DevProgram>()`. An object built against a different
            // struct loads and resolves happily, then reads its own fields, or
            // its block/grid dimensions, from the wrong offsets and faults
            // somewhere unrelated. Refuse it by name here instead.
            const IMPLICIT: u32 = 256;
            let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
            let got = k.kernarg_size();
            if got != want && got != want + IMPLICIT {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; this build's PlowProgram needs {want} \
                     (or {} with the COv5 implicit block) — the code object is STALE. Rebuild it \
                     with scripts/build_gfx950.sh. A mismatched object does not fail to load; it \
                     faults mid-run.",
                    want + IMPLICIT
                )));
            }
            modules.push(m);
            Ok((k, packed_prefill_abi))
        };

        let (k_prefill, packed_prefill_prefill_abi) =
            load_one_in(Phase::Prefill, sched_prefill, hsaco_dir, None)?;
        let (k_decode, _) = load_one_in(Phase::Decode, sched_decode, hsaco_dir, None)?;
        // Task-13: the low-rung decode tier ladder. Each tier's pairing checks
        // run with ITS need (the widest rung it will serve), not the blob-wide
        // max — an MM=4 object legitimately serves rungs 1-2 of a B=32 blob.
        // `PLOW_HSACO_LOWRUNG` is either `<dir>` (max = PLOW_LOWRUNG_MAX,
        // default 2) or `dir:max[,dir:max]...`; selection takes the narrowest
        // tier that fits, so the list is sorted ascending here.
        let mut decode_tiers: Vec<(u32, HsaKernel)> = Vec::new();
        if let Some(spec) = &lowrung_dir {
            let mut tiers: Vec<(String, u32)> = Vec::new();
            if spec.contains(':') {
                for ent in spec.split(',').filter(|s| !s.is_empty()) {
                    let (d, m) = ent.rsplit_once(':').ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "PLOW_HSACO_LOWRUNG entry `{ent}`: expected dir:max"
                        ))
                    })?;
                    let m: u32 = m.parse().map_err(|_| {
                        RuntimeError::Device(format!(
                            "PLOW_HSACO_LOWRUNG entry `{ent}`: max `{m}` is not a u32"
                        ))
                    })?;
                    tiers.push((d.to_string(), m));
                }
            } else {
                let max = crate::config::RuntimeConfig::get().amd.lowrung_max;
                tiers.push((spec.clone(), max));
            }
            tiers.sort_by_key(|&(_, m)| m);
            for (d, max) in tiers {
                let (k, _) = load_one_in(Phase::Decode, sched_decode, Path::new(&d), Some(max))?;
                decode_tiers.push((max, k));
            }
        }
        // Flash follows the PREFILL scheduler — a flash segment is a prefill
        // segment. Optional: without it every segment runs class 8, which is
        // correct and merely slower.
        let flash_load = load_one_in(Phase::Flash, sched_prefill, hsaco_dir, None);
        let flash_error = flash_load.as_ref().err().map(ToString::to_string);
        let k_flash =
            resolve_flash_object_load(flash_load, need_mla_v2 || need_mla_v2_fp8)?.map(|(k, _)| k);
        if let (None, Some(e)) = (&k_flash, flash_error) {
            tracing::info!(%e, "no flash object — flash segments run on the 8-wave interpreter");
        }
        drop(load_one_in);

        let k_decode_mla = if need_decode_mla_segments {
            const MARKER: &str = "plow_decode_mla_segment_object_1";
            let name = format!("interp_decode_mla{}.elf", sched_decode.suffix());
            let path = hsaco_dir.join(&name);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "decode MLA segments require packet-paired object {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            if !syms.contains(&MARKER)
                || !syms.contains(&"plow_packet_hash_lo")
                || !syms.contains(&"plow_packet_hash_hi")
            {
                return Err(RuntimeError::Device(format!(
                    "{} is not a packet-paired decode MLA segment object",
                    path.display()
                )));
            }
            check_gate_hier_object(&syms, &path, Phase::Decode, sched_decode)?;
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let symbol = format!("plow_interp_decode_mla_{arch}{}", sched_decode.suffix());
            let kernel = EngineDevice::get_function(&*be, &module, &symbol)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}")))?;
            const IMPLICIT: u32 = 256;
            let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
            let got = kernel.kernarg_size();
            if got != want && got != want + IMPLICIT {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; decode MLA interpreter needs {want} (or {} with implicit args)",
                    want + IMPLICIT
                )));
            }
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let k_xreduce_wave_rs = if need_xreduce_wave_rs {
            const MARKER: &str = "plow_xreduce_wave_rs_segments_1";
            let name = format!("interp_xreduce{}.elf", sched_prefill.suffix());
            let path = hsaco_dir.join(&name);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "marked XReduceTwoShot segments require {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            if !syms.contains(&MARKER)
                || !syms.contains(&"plow_packet_hash_lo")
                || !syms.contains(&"plow_packet_hash_hi")
            {
                return Err(RuntimeError::Device(format!(
                    "{} is not a packet-paired XReduce wave-RS segment object",
                    path.display()
                )));
            }
            if need_graph_phase_xreduce {
                for marker in [
                    "plow_phase_inventory_xreduce_only_1",
                    "plow_phase_xreduce_wave64_occ2_nospill_1",
                ] {
                    if !syms.contains(&marker) {
                        return Err(RuntimeError::Device(format!(
                            "{} lacks graph phase-object marker `{marker}`",
                            path.display()
                        )));
                    }
                }
                check_compiled_opcode_marker_set(&syms, &path, [DevOp::XReduceTwoShot])?;
            }
            if prefill_l2_placed && !syms.contains(&L2_DISPATCH_SYM) {
                return Err(RuntimeError::Device(l2_pairing_refusal(
                    &path,
                    Phase::Prefill,
                )));
            }
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let symbol = format!("plow_interp_xreduce_{arch}{}", sched_prefill.suffix());
            let kernel = EngineDevice::get_function(&*be, &module, &symbol)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}")))?;
            const IMPLICIT: u32 = 256;
            const MAX_LDS: u32 = 16 * 1024;
            let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
            let got = kernel.kernarg_size();
            if got != want && got != want + IMPLICIT {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; XReduce interpreter needs {want} (or {} with implicit args)",
                    want + IMPLICIT
                )));
            }
            let lds = HsaBackend::kernel_lds_bytes(&kernel);
            if lds > MAX_LDS || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(format!(
                    "{name}: XReduce specialist resource gate failed: lds={lds}, private={} B",
                    kernel.private_segment_size()
                )));
            }
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let k_mla_v2_sv_raw = if arch == "gfx950" && variant != Variant::Fp8Kv && need_mla_v2 {
            let name = format!("interp_mla_v2_sv{}.elf", sched_prefill.suffix());
            let path = hsaco_dir.join(&name);
            if !path.exists() {
                tracing::info!(object = %path.display(),
                    "no raw MLA V2+SV object — pure segments use the interpreter fallback");
                None
            } else {
                let load = (|| -> Result<(Module, HsaKernel)> {
                    let image = std::fs::read(&path).map_err(|e| {
                        RuntimeError::Device(format!("code object {}: {e}", path.display()))
                    })?;
                    let syms = elf_symbol_names(&image);
                    check_mla_v2_sv_raw_symbols(&syms, &path, prefill_l2_placed)?;
                    check_interpreter_waves(
                        elf_symbol_u32(&image, "plow_geom_PLOW_WG_WAVES"), Phase::Flash, &path,
                    )?;
                    check_packet_pairing_stamp(&image, blob_path, &path)?;
                    let module = EngineDevice::module_load(&*be, &image)?;
                    let symbol = symbol_name(Phase::Flash, sched_prefill, &arch);
                    let kernel =
                        EngineDevice::get_function(&*be, &module, &symbol).map_err(|e| {
                            RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}"))
                        })?;
                    const IMPLICIT: u32 = 256;
                    // The GQ twin contributes one 8-byte shared cursor after the 58,368-byte
                    // flash arena; the static twin contains only the arena.
                    const MAX_LDS: u32 = 58_376;
                    let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
                    let got = kernel.kernarg_size();
                    if got != want && got != want + IMPLICIT {
                        return Err(RuntimeError::Device(format!(
                            "{name}: kernarg segment is {got} B; raw MLA ABI needs {want} \
                             (or {} with COv5 implicit args)",
                            want + IMPLICIT
                        )));
                    }
                    let lds = HsaBackend::kernel_lds_bytes(&kernel);
                    if lds > MAX_LDS || kernel.private_segment_size() != 0 {
                        return Err(RuntimeError::Device(format!(
                            "{name}: raw MLA resource gate failed: LDS={lds} (max {MAX_LDS}), \
                             private={} (required 0)",
                            kernel.private_segment_size()
                        )));
                    }
                    Ok((module, kernel))
                })();
                match load {
                    Ok((module, kernel)) => {
                        modules.push(module);
                        Some(kernel)
                    }
                    Err(e) => {
                        tracing::warn!(%e,
                            "raw MLA V2+SV object rejected — using the interpreter fallback");
                        None
                    }
                }
            }
        } else {
            None
        };

        let mut packed_kda_ops: Vec<DevOp> = blob.progs[..dec_ix]
            .iter()
            .flat_map(|prog| &prog.insts)
            .filter_map(|inst| DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op))
            .filter(|op| {
                matches!(
                    op,
                    DevOp::KdaStateStep
                        | DevOp::KdaConv3
                        | DevOp::KdaStateStepG
                        | DevOp::KdaChunkPrepare
                        | DevOp::KdaChunkIntra
                        | DevOp::KdaChunkWu
                        | DevOp::KdaChunkCarry
                )
            })
            .collect();
        packed_kda_ops.sort_unstable_by_key(|op| *op as u16);
        packed_kda_ops.dedup_by_key(|op| *op as u16);
        let mut load_packed_family = |stem: &str,
                                      symbol_base: &str,
                                      phase: Phase,
                                      markers: &[&str],
                                      fp8_kv: Option<bool>,
                                      required_ops: &[DevOp]|
         -> Result<Option<HsaKernel>> {
            let name = format!("{stem}{}.elf", sched_prefill.suffix());
            let path = hsaco_dir.join(&name);
            if !path.exists() {
                return Ok(None);
            }
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!("code object {}: {e}", path.display()))
            })?;
            let syms = elf_symbol_names(&image);
            if !syms.contains(&PACKED_PREFILL_ABI_SYM)
                || markers.iter().any(|marker| !syms.contains(marker))
            {
                return Err(RuntimeError::Device(format!(
                    "packed-prefill family object {} must advertise `{PACKED_PREFILL_ABI_SYM}` \
                         and markers {:?}; refusing a stale or wrong-family object",
                    path.display(),
                    markers
                )));
            }
            check_interpreter_waves(
                elf_symbol_u32(&image, "plow_geom_PLOW_WG_WAVES"), phase, &path,
            )?;
            check_compiled_opcode_marker_set(&syms, &path, required_ops.iter().copied())?;
            if let Some(want_fp8) = fp8_kv {
                let has_fp8 = syms.contains(&FP8_KV_SYM);
                check_packed_family_kv_encoding(has_fp8, want_fp8)
                    .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            }
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let symbol = format!("{symbol_base}_{arch}{}", sched_prefill.suffix());
            let kernel = EngineDevice::get_function(&*be, &module, &symbol)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}")))?;
            const IMPLICIT: u32 = 256;
            let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
            let got = kernel.kernarg_size();
            if got != want && got != want + IMPLICIT {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; packed family ABI needs {want} \
                         (or {} with implicit args)",
                    want + IMPLICIT
                )));
            }
            modules.push(module);
            Ok(Some(kernel))
        };
        let packed_kv_infix = if variant == Variant::Fp8Kv {
            "_fp8kv"
        } else {
            ""
        };
        let packed_route = crate::config::RuntimeConfig::get().amd.packed_prefill_route;
        let kda_family_route = crate::config::RuntimeConfig::get().amd.kda_family_route;
        let k_packed_mla_norm = if packed_route {
            load_packed_family(
                &format!("interp_packed_mla_norm{packed_kv_infix}"),
                "plow_interp_packed_mla_norm",
                Phase::Prefill,
                &[PACKED_PREFILL_MLA_NORM_SEG_SYM],
                Some(variant == Variant::Fp8Kv),
                &[],
            )?
        } else {
            None
        };
        let k_packed_mla_flash = if packed_route {
            load_packed_family(
                &format!("interp_packed_mla_flash{packed_kv_infix}"),
                "plow_interp_packed_mla_flash",
                Phase::Flash,
                &[PACKED_PREFILL_MLA_FLASH_SEG_SYM],
                Some(variant == Variant::Fp8Kv),
                &[],
            )?
        } else {
            None
        };
        let kda_qpre_required = requires.as_ref().is_some_and(|requires| {
            requires
                .iter()
                .any(|requirement| requirement == "PLOW_KDA_CHUNK_QPRE=1")
        });
        let k_packed_kda = if packed_route || kda_family_route {
            let marker = if need_kda_chunk_prefill.is_some() {
                PACKED_PREFILL_KDA_CHUNK_SEG_SYM
            } else {
                PACKED_PREFILL_KDA_SEG_SYM
            };
            let mut markers = vec![KDA_FAMILY_SEG_SYM, marker];
            if kda_qpre_required {
                markers.push(KDA_CHUNK_QPRE_SYM);
            }
            load_packed_family(
                "interp_packed_kda",
                "plow_interp_packed_kda",
                Phase::Prefill,
                &markers,
                None,
                &packed_kda_ops,
            )?
        } else {
            None
        };
        drop(load_packed_family);
        let k_xr_attnres = if need_xr_attnres {
            const NAME: &str = "xreduce_attnres_gfx950.elf";
            const SYMBOL: &str = "plow_xreduce_attnres_gfx950";
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "fused XReduceTwoShot+AttnRes has no qualified object for {arch}"
                )));
            }
            let path = hsaco_dir.join(NAME);
            if !path.exists() {
                return Err(RuntimeError::Device(format!(
                    "fused XReduceTwoShot+AttnRes requires {NAME}"
                )));
            }
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!("code object {}: {e}", path.display()))
            })?;
            let syms = elf_symbol_names(&image);
            for marker in [XR_ATTNRES_SEG_SYM, XR_ATTNRES_RESOURCE_SYM] {
                if !syms.contains(&marker) {
                    return Err(RuntimeError::Device(format!(
                        "{NAME} lacks required ABI/resource marker `{marker}`"
                    )));
                }
            }
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, SYMBOL)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}")))?;
            let got = kernel.kernarg_size();
            if got != 8 && got != 264 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: kernarg segment is {got} B; raw XR+AttnRes ABI needs 8 (or 264 with implicit args)"
                )));
            }
            if kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: private segment is {} B; raw XR+AttnRes requires zero",
                    kernel.private_segment_size()
                )));
            }
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let k_kda_decode_fused = if need_kda_decode_fused.is_some() {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "KdaDecodeFused opcode 125 requires gfx950, but this device is {arch}"
                )));
            }
            const NAME: &str = "kda_decode_fused_gfx950.elf";
            const MARKER: &str = "plow_kda_decode_fused_256x16_2";
            const SYMBOL: &str = "plow_kda_decode_fused_256x16_v2";
            let path = hsaco_dir.join(NAME);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "opcode 125 requires standalone object {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            if !syms.contains(&MARKER) {
                return Err(RuntimeError::Device(format!(
                    "standalone KDA object {} does not advertise required marker `{MARKER}`;                      refusing a missing, wrong, or stale object",
                    path.display()
                )));
            }
            let module = EngineDevice::module_load(&*be, &image)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            let kernel = EngineDevice::get_function(&*be, &module, SYMBOL)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}")))?;
            const IMPLICIT: u32 = 256;
            let want = (std::mem::size_of::<KdaDecodeFusedArgs>() as u32 + 7) & !7;
            let got = kernel.kernarg_size();
            if got != want && got != want + IMPLICIT {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: kernarg segment is {got} B; fused KDA ABI needs {want}                      (or {} with COv5 implicit args). Rebuild the standalone object.",
                    want + IMPLICIT
                )));
            }
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let (k_grouped_moe_glu, k_grouped_moe_down) = if need_grouped_moe {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "standalone grouped MXFP4 MoE requires gfx950, but this device is {arch}"
                )));
            }
            const NAME: &str = "moe_decode_grouped_mxfp4_gfx950.elf";
            const MARKER: &str = "plow_moe_decode_grouped_mxfp4_abi_1";
            const GLU: &str = "plow_moe_decode_grouped_glu_mxfp4_gfx950";
            const DOWN: &str = "plow_moe_decode_grouped_down_mxfp4_gfx950";
            let path = hsaco_dir.join(NAME);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "grouped MoE segment requires {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            if !syms.contains(&MARKER) {
                return Err(RuntimeError::Device(format!(
                    "{} lacks required ABI/resource marker `{MARKER}`",
                    path.display()
                )));
            }
            let module = EngineDevice::module_load(&*be, &image)?;
            let glu = EngineDevice::get_function(&*be, &module, GLU)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {GLU}: {e}")))?;
            let down = EngineDevice::get_function(&*be, &module, DOWN)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {DOWN}: {e}")))?;
            const IMPLICIT: u32 = 256;
            for (kernel, want, role) in [
                (glu, std::mem::size_of::<GroupedMoeGluArgs>() as u32, "GLU"),
                (
                    down,
                    std::mem::size_of::<GroupedMoeDownArgs>() as u32,
                    "DOWN",
                ),
            ] {
                let got = kernel.kernarg_size();
                if got != want && got != want + IMPLICIT {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: {role} kernarg segment is {got} B; expected {want} (or {} with implicit args)",
                        want + IMPLICIT
                    )));
                }
                if kernel.private_segment_size() != 0 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: {role} private segment is {} B; zero scratch is required",
                        kernel.private_segment_size()
                    )));
                }
            }
            modules.push(module);
            (Some(glu), Some(down))
        } else {
            (None, None)
        };

        let k_kda_chunk_intra_cached = if arch == "gfx950" {
            const NAME: &str = "kda_chunk_intra_cached_gfx950.elf";
            const SYMBOL: &str = "plow_kda_chunk_intra_cached_gfx950";
            const MARKERS: [&str; 6] = [
                "plow_kda_intra_cached_abi_1",
                "plow_kda_intra_cached_bt64_d128_1",
                "plow_kda_intra_cached_wave64_1",
                "plow_kda_intra_cached_no_spill_1",
                "plow_kda_intra_cached_static_lds_114688",
                "plow_kda_intra_cached_vgpr_le_96",
            ];
            let path = hsaco_dir.join(NAME);
            if !path.exists() {
                None
            } else {
                let load = (|| -> Result<(Module, HsaKernel)> {
                    let image = std::fs::read(&path).map_err(|e| {
                        RuntimeError::Device(format!("code object {}: {e}", path.display()))
                    })?;
                    let syms = elf_symbol_names(&image);
                    for marker in MARKERS {
                        if !syms.contains(&marker) {
                            return Err(RuntimeError::Device(format!(
                                "cached KDA-intra object {} lacks required ABI/resource marker `{marker}`",
                                path.display()
                            )));
                        }
                    }
                    let module = EngineDevice::module_load(&*be, &image)?;
                    let kernel =
                        EngineDevice::get_function(&*be, &module, SYMBOL).map_err(|e| {
                            RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}"))
                        })?;
                    let want = std::mem::size_of::<KdaChunkIntraCachedArgs>() as u32;
                    let got = kernel.kernarg_size();
                    let lds = HsaBackend::kernel_lds_bytes(&kernel);
                    let private = kernel.private_segment_size();
                    if got != want && got != want + 256 {
                        return Err(RuntimeError::Device(format!(
                            "{NAME}: kernarg segment is {got} B; cached KDA-intra ABI needs {want} (or {} with COv5 implicit args)",
                            want + 256
                        )));
                    }
                    if lds != 114_688 || private != 0 {
                        return Err(RuntimeError::Device(format!(
                            "{NAME}: resource gate failed: LDS={lds} (required 114688), private={private} (required 0)"
                        )));
                    }
                    Ok((module, kernel))
                })();
                match load {
                    Ok((module, kernel)) => {
                        tracing::info!(
                            object = %path.display(),
                            symbol = SYMBOL,
                            "cached KDA-intra object accepted"
                        );
                        modules.push(module);
                        Some(kernel)
                    }
                    Err(e) => {
                        tracing::warn!(%e, "cached KDA-intra object rejected — using interpreter fallback");
                        None
                    }
                }
            }
        } else {
            None
        };

        let k_kda_chunk_intra_wave_items = if need_kda_intra_wave_items {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "marked KDA-intra wave-item segments require gfx950, but this device is {arch}"
                )));
            }
            const NAME: &str = "kda_chunk_intra_wave_items_gfx950.elf";
            const SYMBOL: &str = "plow_kda_chunk_intra_wave_items_gfx950";
            let path = hsaco_dir.join(NAME);
            let image = read_kda_intra_wave_items_object(&path)?;
            let syms = elf_symbol_names(&image);
            check_kda_intra_wave_items_symbols(&syms, &path)?;
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, SYMBOL)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}")))?;
            let want = std::mem::size_of::<KdaChunkIntraCachedArgs>() as u32;
            let got = kernel.kernarg_size();
            let lds = HsaBackend::kernel_lds_bytes(&kernel);
            let private = kernel.private_segment_size();
            if got != want && got != want + 256 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: kernarg segment is {got} B; wave-item KDA-intra ABI needs {want} (or {} with COv5 implicit args)",
                    want + 256
                )));
            }
            if lds != 131_072 || private != 0 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: resource gate failed: LDS={lds} (required 131072), private={private} (required 0)"
                )));
            }
            tracing::info!(
                object = %path.display(),
                symbol = SYMBOL,
                "KDA-intra wave-item object accepted"
            );
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let k_kda_chunk_carry_regstate = if need_kda_carry_regstate {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "marked KDA carry regstate segments require gfx950, but this device is {arch}"
                )));
            }
            const NAME: &str = "kda_chunk_carry_regstate_gfx950.elf";
            const SYMBOL: &str = "plow_kda_chunk_carry_regstate_gfx950";
            let path = hsaco_dir.join(NAME);
            let image = read_kda_carry_regstate_object(&path)?;
            let syms = elf_symbol_names(&image);
            check_kda_carry_regstate_symbols(&syms, &path)?;
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, SYMBOL)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}")))?;
            let want = std::mem::size_of::<KdaChunkCarryRegstateArgs>() as u32;
            let got = kernel.kernarg_size();
            let lds = HsaBackend::kernel_lds_bytes(&kernel);
            let private = kernel.private_segment_size();
            if got != want && got != want + 256 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: kernarg segment is {got} B; KDA carry regstate ABI needs {want} (or {} with COv5 implicit args)",
                    want + 256
                )));
            }
            if lds != KDA_CARRY_REGSTATE_LDS || private != 0 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: resource gate failed: LDS={lds} (required {KDA_CARRY_REGSTATE_LDS}), private={private} (required 0)"
                )));
            }
            tracing::info!(
                object = %path.display(),
                symbol = SYMBOL,
                "KDA carry regstate object accepted"
            );
            modules.push(module);
            Some(kernel)
        } else {
            None
        };

        let mut load_kda_lean = |name: &str,
                                 symbol: &str,
                                 what: &str,
                                 markers: &[&str],
                                 kernarg: u32,
                                 lds: u32|
         -> Result<HsaKernel> {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "marked {what} segments require gfx950, but this device is {arch}"
                )));
            }
            let path = hsaco_dir.join(name);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "marked {what} segments require {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            for marker in markers {
                if !syms.contains(marker) {
                    return Err(RuntimeError::Device(format!(
                        "{what} object {} lacks required ABI/resource marker `{marker}`",
                        path.display()
                    )));
                }
            }
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, symbol)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}")))?;
            let got = kernel.kernarg_size();
            if got != kernarg && got != kernarg + 256 {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; {what} ABI needs {kernarg} (or {} with COv5 implicit args)",
                    kernarg + 256
                )));
            }
            let got_lds = HsaBackend::kernel_lds_bytes(&kernel);
            let private = kernel.private_segment_size();
            if got_lds != lds || private != 0 {
                return Err(RuntimeError::Device(format!(
                    "{name}: resource gate failed: LDS={got_lds} (required {lds}), private={private} (required 0)"
                )));
            }
            tracing::info!(object = %path.display(), symbol, "{what} object accepted");
            modules.push(module);
            Ok(kernel)
        };
        let k_kda_chunk_wu_lean = if need_kda_wu_lean {
            Some(load_kda_lean(
                "kda_chunk_wu_lean_gfx950.elf",
                "plow_kda_chunk_wu_lean_gfx950",
                "KDA Wu lean",
                &KDA_WU_LEAN_MARKERS,
                std::mem::size_of::<KdaChunkWuLeanArgs>() as u32,
                KDA_WU_LEAN_LDS,
            )?)
        } else {
            None
        };
        let (k_kda_chunk_wu_lean_keys, k_kda_chunk_carry_keyfeed) = if need_kda_carry_keyfeed {
            let wu = load_kda_lean(
                "kda_chunk_wu_lean_keys_gfx950.elf",
                "plow_kda_chunk_wu_lean_keys_gfx950",
                "KDA Wu lean keys",
                &KDA_WU_LEAN_MARKERS,
                std::mem::size_of::<KdaChunkWuLeanArgs>() as u32,
                KDA_WU_LEAN_LDS,
            )?;
            let carry = load_kda_lean(
                "kda_chunk_carry_regstate_keyfeed_gfx950.elf",
                "plow_kda_chunk_carry_regstate_keyfeed_gfx950",
                "KDA carry keyfeed",
                &KDA_CARRY_KEYFEED_MARKERS,
                std::mem::size_of::<KdaChunkCarryKeyfeedArgs>() as u32,
                KDA_CARRY_REGSTATE_LDS,
            )?;
            (Some(wu), Some(carry))
        } else {
            (None, None)
        };

        let need_kda_key_factor = arch == "gfx950"
            && blob.progs[..dec_ix]
                .iter()
                .any(|p| !kda_key_factor_segment_pairs(p).is_empty());
        let (k_kda_key_factor_wu, k_kda_key_factor_carry) = if need_kda_key_factor {
            const COMMON: [&str; 7] = [
                "plow_kda_key_factor_abi_1",
                "plow_kda_key_factor_pair_1",
                "plow_kda_key_factor_bt64_d128_v128_1",
                "plow_kda_key_factor_qpre_1",
                "plow_kda_key_factor_wave64_1",
                "plow_kda_key_factor_nospill_1",
                "plow_kda_key_factor_scratch_pair_bf16_1",
            ];
            let specs = [
                (
                    "kda_chunk_key_factor_wu_gfx950.elf",
                    "plow_kda_chunk_key_factor_wu_gfx950",
                    "plow_kda_key_factor_wu_1",
                    "plow_kda_key_factor_wu_vgpr_le_160",
                    std::mem::size_of::<KdaChunkKeyFactorWuArgs>() as u32,
                    0,
                ),
                (
                    "kda_chunk_key_factor_carry_gfx950.elf",
                    "plow_kda_chunk_key_factor_carry_gfx950",
                    "plow_kda_key_factor_carry_1",
                    "plow_kda_key_factor_carry_vgpr_le_160",
                    std::mem::size_of::<KdaChunkKeyFactorCarryArgs>() as u32,
                    14_336,
                ),
            ];
            let paths = specs.map(|spec| hsaco_dir.join(spec.0));
            match (paths[0].exists(), paths[1].exists()) {
                (false, false) => {
                    tracing::warn!(
                        "KDA key-factor segments have no paired objects — using interpreter fallback"
                    );
                    (None, None)
                }
                (a, b) if a != b => {
                    return Err(RuntimeError::Device(format!(
                        "KDA key-factor object pair is incomplete: {} exists={a}, {} exists={b}",
                        paths[0].display(),
                        paths[1].display()
                    )));
                }
                _ => {
                    let mut kernels = Vec::with_capacity(2);
                    for (spec, path) in specs.into_iter().zip(paths) {
                        let image = std::fs::read(&path).map_err(|e| {
                            RuntimeError::Device(format!("code object {}: {e}", path.display()))
                        })?;
                        let syms = elf_symbol_names(&image);
                        for marker in COMMON.into_iter().chain([spec.2, spec.3]) {
                            if !syms.contains(&marker) {
                                return Err(RuntimeError::Device(format!(
                                    "KDA key-factor object {} lacks required marker `{marker}`",
                                    path.display()
                                )));
                            }
                        }
                        check_packet_pairing_stamp(&image, blob_path, &path)?;
                        let module = EngineDevice::module_load(&*be, &image)?;
                        let kernel =
                            EngineDevice::get_function(&*be, &module, spec.1).map_err(|e| {
                                RuntimeError::Device(format!(
                                    "{}: no symbol {}: {e}",
                                    spec.0, spec.1
                                ))
                            })?;
                        let got = kernel.kernarg_size();
                        if got != spec.4 && got != spec.4 + 256 {
                            return Err(RuntimeError::Device(format!(
                                "{}: kernarg segment is {got} B; expected {} (or {} with COv5 implicit args)",
                                spec.0, spec.4, spec.4 + 256
                            )));
                        }
                        let lds = HsaBackend::kernel_lds_bytes(&kernel);
                        let private = kernel.private_segment_size();
                        if lds != spec.5 || private != 0 {
                            return Err(RuntimeError::Device(format!(
                                "{}: resource gate failed: LDS={lds} (required {}), private={private} (required 0)",
                                spec.0, spec.5
                            )));
                        }
                        modules.push(module);
                        kernels.push(kernel);
                    }
                    (Some(kernels[0]), Some(kernels[1]))
                }
            }
        } else {
            (None, None)
        };

        let (k_moe_stage1_mxfp4, k_moe_stage1_a4_quant, k_moe_stage1_a4_reuse) = if arch == "gfx950"
            && need_moe_stage1_lean
        {
            const NAME: &str = "moe_stage1_mxfp4_gfx950.elf";
            const SYMBOL: &str = "plow_moe1_mxfp4_bk256_gfx950";
            const MARKERS: [&str; 6] = [
                "plow_moe1_mxfp4_stage1_abi_1",
                "plow_moe1_mxfp4_stage1_bm64_bn256_bk256_1",
                "plow_moe1_mxfp4_stage1_wave64_1",
                "plow_moe1_mxfp4_stage1_no_spill_1",
                "plow_moe1_mxfp4_stage1_dynamic_lds_119808",
                "plow_moe1_mxfp4_stage1_vgpr_le_192",
            ];
            let path = hsaco_dir.join(NAME);
            if !path.exists() {
                (None, None, None)
            } else {
                if !prefill_moe_align_bm64 {
                    return Err(RuntimeError::Device(format!(
                        "lean MoE stage-1 object {} requires a producer advertising plow_moe_align_bm64_1",
                        path.display()
                    )));
                }
                let image = std::fs::read(&path).map_err(|e| {
                    RuntimeError::Device(format!("code object {}: {e}", path.display()))
                })?;
                let syms = elf_symbol_names(&image);
                for marker in MARKERS {
                    if !syms.contains(&marker) {
                        return Err(RuntimeError::Device(format!(
                            "lean MoE stage-1 object {} lacks required ABI/resource marker `{marker}`",
                            path.display()
                        )));
                    }
                }
                let module = EngineDevice::module_load(&*be, &image)?;
                let kernel = EngineDevice::get_function(&*be, &module, SYMBOL).map_err(|e| {
                    RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}"))
                })?;
                let want = std::mem::size_of::<MoeStage1Mxfp4Args>() as u32;
                let got = kernel.kernarg_size();
                if got != want && got != want + 256 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: kernarg segment is {got} B; lean stage-1 ABI needs {want} (or {} with COv5 implicit args)",
                        want + 256
                    )));
                }
                if kernel.private_segment_size() != 0 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: private segment is {} B; lean stage-1 requires zero",
                        kernel.private_segment_size()
                    )));
                }
                EngineDevice::set_max_dynamic_smem(&*be, kernel, 119_808)?;
                let reuse_markers = [
                    "plow_moe1_a4_reuse_abi_1",
                    "plow_moe1_a4_reuse_wave64_1",
                    "plow_moe1_a4_reuse_four_wave_1",
                    "plow_moe1_a4_reuse_a_only_lds_8192",
                    "plow_moe1_a4_reuse_register_b_1",
                ];
                let (quant, reuse) = if crate::config::RuntimeConfig::get().amd.moe_stage1_a4_reuse
                    && reuse_markers.iter().all(|m| syms.contains(m))
                {
                    let quant =
                        EngineDevice::get_function(&*be, &module, "plow_moe1_quant_sort_a4_gfx950")
                            .map_err(|e| {
                                RuntimeError::Device(format!(
                                    "{NAME}: no A4 quant/sort symbol: {e}"
                                ))
                            })?;
                    let reuse = EngineDevice::get_function(
                        &*be,
                        &module,
                        "plow_moe1_a4_reuse_16x16x128_gfx950",
                    )
                    .map_err(|e| {
                        RuntimeError::Device(format!("{NAME}: no A4 reuse symbol: {e}"))
                    })?;
                    let qwant = std::mem::size_of::<MoeStage1A4QuantArgs>() as u32;
                    let rwant = std::mem::size_of::<MoeStage1A4ReuseArgs>() as u32;
                    if ![qwant, qwant + 256].contains(&quant.kernarg_size())
                        || ![rwant, rwant + 256].contains(&reuse.kernarg_size())
                        || quant.private_segment_size() != 0
                        || reuse.private_segment_size() != 0
                    {
                        return Err(RuntimeError::Device(format!(
                            "{NAME}: A4 reuse resource/ABI gate failed (quant args={}, private={}; reuse args={}, private={})",
                            quant.kernarg_size(), quant.private_segment_size(),
                            reuse.kernarg_size(), reuse.private_segment_size()
                        )));
                    }
                    EngineDevice::set_max_dynamic_smem(&*be, reuse, 32_768)?;
                    (Some(quant), Some(reuse))
                } else {
                    (None, None)
                };
                modules.push(module);
                (Some(kernel), quant, reuse)
            }
        } else {
            (None, None, None)
        };

        let k_moe_stage2_mxfp4 = if arch == "gfx950" && need_moe_stage2_lean {
            const NAME: &str = "moe_stage2_mxfp4_gfx950.elf";
            const SYMBOL: &str = "plow_moe2_mxfp4_16x16x128_gfx950";
            const MARKERS: [&str; 6] = [
                "plow_moe2_mxfp4_stage2_abi_3",
                "plow_moe2_mxfp4_stage2_layout_shuffled_1",
                "plow_moe2_mxfp4_stage2_no_spill_1",
                "plow_moe2_mxfp4_stage2_f32_scatter_1",
                "plow_moe2_mxfp4_stage2_dynamic_lds_4352",
                "plow_moe2_mxfp4_stage2_vgpr_le_100",
            ];
            let path = hsaco_dir.join(NAME);
            if !path.exists() {
                None
            } else {
                if !prefill_moe_align_bm64 {
                    return Err(RuntimeError::Device(format!(
                        "lean MoE object {} requires a prefill producer advertising plow_moe_align_bm64_1",
                        path.display()
                    )));
                }
                let image = std::fs::read(&path).map_err(|e| {
                    RuntimeError::Device(format!("code object {}: {e}", path.display()))
                })?;
                let syms = elf_symbol_names(&image);
                for marker in MARKERS {
                    if !syms.contains(&marker) {
                        return Err(RuntimeError::Device(format!(
                            "lean MoE object {} lacks required ABI/resource marker `{marker}`",
                            path.display()
                        )));
                    }
                }
                let module = EngineDevice::module_load(&*be, &image)?;
                let kernel = EngineDevice::get_function(&*be, &module, SYMBOL).map_err(|e| {
                    RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}"))
                })?;
                let want = std::mem::size_of::<MoeStage2Mxfp4Args>() as u32;
                let got = kernel.kernarg_size();
                if got != want && got != want + 256 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: kernarg segment is {got} B; lean MoE ABI needs {want} (or {} with COv5 implicit args)",
                        want + 256
                    )));
                }
                EngineDevice::set_max_dynamic_smem(&*be, kernel, 4_352)?;
                modules.push(module);
                Some(kernel)
            }
        } else {
            None
        };

        let k_moe_combine = if arch == "gfx950" && need_moe_combine_lean {
            const NAME: &str = "moe_combine_gfx950.elf";
            const SYMBOL: &str = "plow_moe_combine_fixed_order_gfx950";
            const MARKERS: [&str; 5] = [
                "plow_moe_combine_fixed_order_abi_1",
                "plow_moe_combine_fixed_order_slots16_1",
                "plow_moe_combine_materialized_f32_1",
                "plow_moe_combine_wave64_1",
                "plow_moe_combine_no_spill_1",
            ];
            let path = hsaco_dir.join(NAME);
            if !path.exists() {
                None
            } else {
                let image = std::fs::read(&path).map_err(|e| {
                    RuntimeError::Device(format!("code object {}: {e}", path.display()))
                })?;
                let syms = elf_symbol_names(&image);
                for marker in MARKERS {
                    if !syms.contains(&marker) {
                        return Err(RuntimeError::Device(format!(
                            "lean MoE combine object {} lacks required ABI/resource marker `{marker}`",
                            path.display()
                        )));
                    }
                }
                let module = EngineDevice::module_load(&*be, &image)?;
                let kernel = EngineDevice::get_function(&*be, &module, SYMBOL).map_err(|e| {
                    RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}"))
                })?;
                let want = std::mem::size_of::<MoeCombineArgs>() as u32;
                let got = kernel.kernarg_size();
                if got != want && got != want + 256 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: kernarg segment is {got} B; lean combine ABI needs {want} (or {} with COv5 implicit args)",
                        want + 256
                    )));
                }
                let lds = HsaBackend::kernel_lds_bytes(&kernel);
                let private = kernel.private_segment_size();
                if lds != 0 || private != 0 {
                    return Err(RuntimeError::Device(format!(
                        "{NAME}: resource gate failed: LDS={lds} (required 0), private={private} (required 0)"
                    )));
                }
                modules.push(module);
                Some(kernel)
            }
        } else {
            None
        };

        let k_attn_res_f32mix = if need_attn_res_f32mix {
            if arch != "gfx950" {
                return Err(RuntimeError::Device(format!(
                    "f32-mix AttnRes packets require gfx950, but this device is {arch}"
                )));
            }
            const NAME: &str = "attn_res_f32mix_gfx950.elf";
            const SYMBOL: &str = "plow_attn_res_f32mix_gfx950";
            let path = hsaco_dir.join(NAME);
            let image = read_attn_res_f32mix_object(&path)?;
            let syms = elf_symbol_names(&image);
            check_attn_res_f32mix_symbols(&syms, &path)?;
            check_packet_pairing_stamp(&image, blob_path, &path)?;
            let threads = elf_symbol_u32(&image, "plow_attn_res_f32mix_threads")
                .filter(|&n| n != 0 && n % 64 == 0 && n <= 1024)
                .ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "{NAME}: plow_attn_res_f32mix_threads is missing or not a wave64 workgroup"
                    ))
                })?;
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, SYMBOL)
                .map_err(|e| RuntimeError::Device(format!("{NAME}: no symbol {SYMBOL}: {e}")))?;
            let want = std::mem::size_of::<AttnResF32MixArgs>() as u32;
            let got = kernel.kernarg_size();
            if got != want && got != want + 256 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: kernarg segment is {got} B; f32-mix AttnRes ABI needs {want} (or {} with COv5 implicit args)",
                    want + 256
                )));
            }
            let private = kernel.private_segment_size();
            if private != 0 {
                return Err(RuntimeError::Device(format!(
                    "{NAME}: resource gate failed: private={private} (required 0)"
                )));
            }
            tracing::info!(
                object = %path.display(),
                symbol = SYMBOL,
                threads,
                "f32-mix AttnRes object accepted"
            );
            modules.push(module);
            Some((kernel, threads))
        } else {
            None
        };

        let (k_moe_ep_align, k_moe_ep_stage2, k_moe_ep_combine) = if need_moe_ep {
            let mut load_ep = |name: &str,
                               symbol: &str,
                               markers: &[&str],
                               want: u32,
                               dynamic_lds: u32|
             -> Result<HsaKernel> {
                let path = hsaco_dir.join(name);
                if !path.exists() {
                    return Err(RuntimeError::Device(format!(
                        "EP packet requires missing specialist object {}",
                        path.display()
                    )));
                }
                let image = std::fs::read(&path).map_err(|e| {
                    RuntimeError::Device(format!("code object {}: {e}", path.display()))
                })?;
                let syms = elf_symbol_names(&image);
                check_moe_ep_symbols(&syms, &path, markers)?;
                let module = EngineDevice::module_load(&*be, &image)?;
                let kernel = EngineDevice::get_function(&*be, &module, symbol).map_err(|e| {
                    RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}"))
                })?;
                if ![want, want + 256].contains(&kernel.kernarg_size())
                    || kernel.private_segment_size() != 0
                {
                    return Err(RuntimeError::Device(format!(
                        "{name}: EP ABI/resource gate failed (args={}, private={})",
                        kernel.kernarg_size(),
                        kernel.private_segment_size()
                    )));
                }
                if dynamic_lds != 0 {
                    EngineDevice::set_max_dynamic_smem(&*be, kernel, dynamic_lds)?;
                }
                modules.push(module);
                Ok(kernel)
            };
            let align = load_ep(
                "moe_ep_align_gfx950.elf",
                "plow_moe_ep_filter_align_gfx950",
                &[
                    "plow_moe_ep_filter_align_abi_1",
                    "plow_moe_ep_filter_align_wave64_1",
                    "plow_moe_ep_filter_align_stable_1",
                    "plow_moe_ep_filter_align_parallel_1",
                    "plow_moe_ep_filter_align_no_spill_1",
                ],
                std::mem::size_of::<MoeEpAlignArgs>() as u32,
                0,
            )?;
            let stage2 = load_ep(
                "moe_ep_stage2_gfx950.elf",
                "plow_moe2_ep_full_i_16x16x128_gfx950",
                &[
                    "plow_moe2_mxfp4_stage2_abi_3",
                    "plow_moe2_mxfp4_stage2_no_spill_1",
                    "plow_moe2_ep_full_i_3072",
                    "plow_moe2_ep_full_i_vgpr_le_128",
                ],
                std::mem::size_of::<MoeStage2Mxfp4Args>() as u32,
                4_352,
            )?;
            let combine = load_ep(
                "moe_ep_combine_gfx950.elf",
                "plow_moe_ep_combine_gfx950",
                &[
                    "plow_moe_ep_combine_abi_1",
                    "plow_moe_ep_combine_fixed_slot_1",
                    "plow_moe_ep_combine_wave64_1",
                    "plow_moe_ep_combine_no_spill_1",
                ],
                std::mem::size_of::<MoeEpCombineArgs>() as u32,
                0,
            )?;
            (Some(align), Some(stage2), Some(combine))
        } else {
            (None, None, None)
        };

        let mut load_mla_raw = |name: &str,
                                symbol: &str,
                                markers: &[&str],
                                kernarg: u32,
                                lds: u32|
         -> Result<Option<HsaKernel>> {
            if !need_mla_materialized {
                return Ok(None);
            }
            let path = hsaco_dir.join(name);
            let image = std::fs::read(&path).map_err(|e| {
                RuntimeError::Device(format!(
                    "materialized MLA requires code object {}: {e}",
                    path.display()
                ))
            })?;
            let syms = elf_symbol_names(&image);
            for marker in markers {
                if !syms.contains(marker) {
                    return Err(RuntimeError::Device(format!(
                        "materialized MLA object {} lacks required marker `{marker}`",
                        path.display()
                    )));
                }
            }
            let module = EngineDevice::module_load(&*be, &image)?;
            let kernel = EngineDevice::get_function(&*be, &module, symbol)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {symbol}: {e}")))?;
            let got = kernel.kernarg_size();
            if got != kernarg && got != kernarg + 256 {
                return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; expected {kernarg} (or {} with COv5 implicit args)",
                    kernarg + 256
                )));
            }
            let got_lds = HsaBackend::kernel_lds_bytes(&kernel);
            if got_lds != lds || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(format!(
                    "{name}: resource gate failed: LDS={got_lds} (required {lds}), private={} (required 0)",
                    kernel.private_segment_size()
                )));
            }
            modules.push(module);
            Ok(Some(kernel))
        };
        let k_mla_materialize_pack = load_mla_raw(
            "mla_materialize_pack_gfx950.elf",
            "plow_mla_materialize_pack_gfx950",
            &[
                "plow_mla_materialize_pack_abi_1",
                "plow_mla_materialize_pack_hd192_v128_1",
                "plow_mla_materialize_pack_nospill_1",
            ],
            std::mem::size_of::<MlaMaterializePackArgs>() as u32,
            0,
        )?;
        let k_mla_materialized_prefill = load_mla_raw(
            "mla_materialized_hd192_v128_gfx950.elf",
            "plow_mla_materialized_hd192_v128_gfx950",
            &[
                "plow_mla_materialized_opus_abi_1",
                "plow_mla_materialized_hd192_v128_1",
                "plow_mla_materialized_wave64_nospill_1",
            ],
            std::mem::size_of::<MlaMaterializedPrefillArgs>() as u32,
            149_760,
        )?;
        let sparse_mla = if use_sparse_mla {
            let candidates = blob.progs[..dec_ix]
                .iter()
                .filter(|p| !p.packed_prefill_only && p.t >= 2048)
                .flat_map(|p| {
                    p.insts
                        .iter()
                        .filter(|d| amd_sparse_mla::union_handle(d).is_some())
                        .map(move |d| (p.t, d.i[2]))
                });
            let (rows, ctx) = candidates.fold((0, 0), |(r, c), (rr, cc)| (r.max(rr), c.max(cc)));
            if rows == 0 || rows > 8192 || ctx < 2048 || ctx > 81920 {
                return Err(RuntimeError::Device(
                    "sparse AITER MLA requires eligible rows<=8192 and ctx<=81920".into(),
                ));
            }
            Some(amd_sparse_mla::SparseMla::load(
                &be,
                hsaco_dir,
                rows,
                ctx,
                variant == Variant::Fp8Kv,
                &mut modules,
            )?)
        } else {
            None
        };

        let gemm_lt = if use_gemm_lt {
            Some(amd_gemm_lt::GemmLt::load(&be, &hsaco_dir, &mut modules)?)
        } else {
            None
        };
        let index_tp = if use_index_tp {
            Some(amd_index_tp::IndexTp::load(&be, &hsaco_dir, &mut modules)?)
        } else {
            None
        };
        let mut moe_aiter = if use_moe_aiter {
            let rows = blob
                .progs
                .iter()
                .filter(|p| has_moe_aiter(p))
                .map(|p| p.t)
                .max()
                .unwrap();
            Some(amd_moe_aiter::MoeAiter::load(
                &be,
                hsaco_dir,
                rows.max(128),
                blob.progs[dec_ix..].iter().any(has_moe_aiter),
                use_resident_moe,
                &mut modules,
            )?)
        } else {
            None
        };

        // --- tensors + weights ------------------------------------------------
        // Staging is one pinned slab, filled and pushed in `STAGE` chunks. The
        // source is an mmap of the checkpoint, and `upload` would pin it per
        // call — asking the kernel to lock tens of GiB of page-cache mappings.
        // Copying through a fixed pinned buffer keeps the locked set at 64 MiB.
        const STAGE: usize = 64 << 20;
        // `Arc`, because the prefetch pool madvises these mappings from other
        // threads and must be joined before they can be unmapped. Under TP the
        // caller passes ONE checkpoint for the whole group — see `shared_ckpt`.
        let ckpt = match (shared_ckpt, checkpoint) {
            (Some(c), _) => Some(c),
            (None, Some(dir)) => Some(Arc::new(crate::asset::checkpoint::Checkpoint::open(dir)?)),
            (None, None) => None,
        };
        // The fp8 weight TWINS live in their own checkpoint, not the bf16 one:
        // they are a separate quantisation artifact, and the packet names them
        // with an `fp8/` prefix that is stripped before lookup. Without this an
        // fp8 packet fails at the first weight with "MISSING WEIGHT", which
        // reads as a broken packet rather than a missing directory.
        let fp8_ckpt = match crate::config::RuntimeConfig::get().amd.fp8_dir.as_deref() {
            Some(d) => Some(crate::asset::checkpoint::Checkpoint::open(Path::new(d))?),
            None => None,
        };
        // `--amd-upload-slots 1` / `PLOW_UPLOAD_SLOTS=1` is the pre-pipeline
        // shape exactly: one slab, one copy, waited on before the next memcpy
        // starts. Kept so the pipelining can be A/B'd on one binary rather
        // than argued about.
        let slots = (crate::config::RuntimeConfig::get().amd.upload_slots as usize).max(1);
        let mut ring = be.upload_ring(slots, STAGE)?;
        // v7 blobs carry the RoPE tables as RECIPES, not bytes. Materialising
        // them is not optional: a reader that skips this leaves cos=sin=0 and
        // serves fluent-looking garbage with no error anywhere.
        let gen_by_tensor: std::collections::HashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();

        // Must precede the tensor loop: it decides whether each full-layer KV
        // tensor gets an allocation or a view onto the pool's VA reservation.
        let config = crate::config::RuntimeConfig::get();
        let shared_requested = config.prefix_cache && config.amd.shared_prefix != Some(false);
        let shared_layout = if shared_requested && arch == "gfx942" && be.has_vmm() && !config.fusion {
            blob.tensors.iter().find(|t| t.name == "in.pos")
                .and_then(|t| u32::try_from(t.bytes / 4).ok())
                .and_then(|context| shared_prefix::Layout::from_blob(&blob, max_decode_batch as usize, context))
        } else {
            None
        };
        if shared_requested && config.amd.shared_prefix == Some(true) && shared_layout.is_none() {
            return Err(RuntimeError::Rejected(
                "AMD shared prefixes require gfx942, ROCr VMM, no legacy fusion, and complete MLA cache writes with supported geometry on every rung".into()));
        }
        let mut shared_prefix = shared_layout.map(|layout| {
            shared_prefix::SharedPrefix::new(be.clone(), layout,
                u64::from(config.prefix_cache_mib()) << 20,
                crate::memory::vmm::kv_pool_cap())
        }).transpose()?;
        let vmm = if shared_prefix.is_none() {
            Self::vmm_bringup(&be, &blob, checkpoint, max_decode_batch as usize)
        } else {
            None
        };
        let cache_va: Vec<_> = blob.tensors.iter().enumerate().map(|(id, tensor)| {
            shared_prefix.as_ref().and_then(|v| v.tensor_va(id)).or_else(|| {
                let (layer, role) = kv_tensor_name(&tensor.name)?;
                vmm.as_ref()?.tensor_va(layer, role)
            })
        }).collect();

        let prof = LoadProf::default();
        let do_prefault = profile_faults();
        let depth = prefetch_depth();
        let prefetch = ckpt.as_ref().and_then(|c| {
            crate::asset::checkpoint::Prefetcher::start(
                Arc::clone(c),
                prefetch_threads(),
                depth,
                None,
            )
        });
        // The pool runs `depth` WEIGHT tensors ahead of the copy, over the same
        // list in the same order, so `pf` only ever moves forward and each
        // tensor is queued exactly once. Skipping the non-weights keeps the
        // depth denominated in reads rather than in table entries — most of this
        // blob's tensors are scratch that touches no checkpoint at all.
        let mut pf = 0usize;
        let prefetch_ahead = |cur: &mut usize, budget: usize| {
            let (Some(pool), Some(c)) = (prefetch.as_ref(), ckpt.as_ref()) else {
                return;
            };
            let mut n = 0;
            while *cur < blob.tensors.len() && n < budget {
                let td = &blob.tensors[*cur];
                *cur += 1;
                // `fp8/` weights live in the twin checkpoint, which is not the
                // one the pool holds, and they are a rounding error next to the
                // experts — so they are simply not prefetched.
                if !packet::names::is_checkpoint_weight(&td.name) || td.name.starts_with("fp8/") {
                    continue;
                }
                if let Some(s) = weight_span(c, &td.name, td.bytes, rank, n_gpu) {
                    pool.push(s);
                }
                n += 1;
            }
        };
        prefetch_ahead(&mut pf, depth);

        // ---- one allocation for every tensor that would otherwise get its own
        //
        // Both passes below must agree, tensor for tensor, on which tensors are
        // carved and how much each consumes: the sizing pass decides how big the
        // slab is and the upload loop walks the cursor through it, so a filter
        // that disagreed in either direction would either overrun the end or
        // silently overlap two tensors. They are kept in step by sharing these
        // two closures rather than by two hand-copied conditions.
        //
        // The two arms that take a view instead of an allocation:
        //   * TP peer slots — storage owned by the `TpRank` peer region, which
        //     `XReduce` reads over XGMI. Carving these out of local VRAM would
        //     have every rank reduce slots its peers never wrote.
        //   * full-layer KV under VMM — the pool's VA reservation, mapped lazily
        //     at the per-sequence frontier.
        let is_peer_slot = |name: &str| {
            matches!(
                (tp.is_some(), name),
                (true, "act.og_tp")
                    | (true, "act.dg_tp")
                    | (true, "act.ug_tp")
                    | (true, "act.h2_tp")
                    | (true, "act.xe_tp")
                    | (true, "act.rt_tp")
            )
        };
        // Rank-relative BAND VIEWS (`<base>@band<t>`, sequence-parallel seams): rows
        // `[rank*t/tp, (rank+1)*t/tp)` of an already-bound base, i.e. `base + rank * bytes`.
        // Storage belongs to the base, so they are views like the peer slots.
        let is_band_view = |name: &str| tp.is_some() && name.contains("@band");
        // `.max(1)`, exactly as the per-tensor arm does — a zero-byte tensor
        // still needs a distinct address, and a zero-length carve would hand the
        // next tensor the same one.
        let slab_need = |bytes: u64| bytes.max(1);
        let slab_bytes: u64 = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(id, td)| !is_peer_slot(&td.name) && !is_band_view(&td.name) && cache_va[*id].is_none())
            .map(|(_, td)| slab_pad(slab_need(td.bytes)))
            .sum();

        // ---- EVERY checkpoint weight is RESOLVED before a byte is uploaded
        //
        // Weight resolution used to happen only inside the upload loop, tensor by tensor, so
        // a name the checkpoint does not have — or a dtype the loader cannot bind — was
        // discovered when the cursor reached it. On GLM-5.3 that was ~200 GiB of DMA before
        // `MISSING WEIGHT` or a `slice_for` byte-count error, per rank, on four ranks. This
        // pass answers the same questions from the safetensors index alone: no page is
        // faulted, no payload is read, and the whole walk is a hash lookup per tensor.
        //
        // It must mirror the loop's resolution EXACTLY or it fails loads that work: same
        // `is_checkpoint_weight` gate, same `fp8/` -> `fp8_ckpt` routing, same two-spelling
        // lookup, and the same exemption for the derived `_res_score.weight` (in no
        // checkpoint by design — `fold_res_score`).
        preflight_weights(&blob, ckpt.as_deref(), fp8_ckpt.as_ref(), rank, n_gpu)?;

        let t_slab = Instant::now();
        let weight_slab = if slab_bytes == 0 || !crate::asset::checkpoint::weight_slab_enabled() {
            WeightSlab::PerTensor
        } else {
            // Opt-in (`PLOW_WEIGHT_VMM=1`): the lazy-commit slab is unmeasured
            // on AMD hardware — see `weight_vmm_amd_enabled` for why the
            // default differs from CUDA's. Every failure falls through to the
            // flat allocation, which falls through to per-tensor.
            let vmm_slab = if crate::asset::checkpoint::weight_vmm_amd_enabled() {
                match crate::memory::vmm::VmmSlab::new(
                    Arc::clone(&be) as Arc<dyn VmmOps>,
                    slab_bytes,
                    crate::memory::vmm::WEIGHT_SLAB_CHUNK,
                ) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::warn!(
                            bytes = slab_bytes,
                            error = %e,
                            "vmm weight slab refused — falling back to flat allocation"
                        );
                        None
                    }
                }
            } else {
                None
            };
            match vmm_slab {
                Some(s) => WeightSlab::Vmm(s),
                None => match EngineDevice::alloc(&*be, slab_bytes) {
                    Ok(m) => WeightSlab::Flat(m),
                    // Not fatal: the per-tensor arm below still works and is only
                    // slower and hungrier. Better a fat load than a refused one.
                    Err(e) => {
                        tracing::warn!(
                            bytes = slab_bytes,
                            error = %e,
                            "single weight allocation refused — falling back to per-tensor alloc"
                        );
                        WeightSlab::PerTensor
                    }
                },
            }
        };
        LoadProf::add(&prof.alloc_ns, t_slab);
        let mut slab_off: u64 = 0;

        let t_tensors = Instant::now();
        let mut devp = Vec::with_capacity(blob.tensors.len());
        let mut names = Vec::with_capacity(blob.tensors.len());
        let (mut wbytes, mut nweights) = (0u64, 0usize);
        // Tensors that took a view into storage someone else owns (peer region
        // or VMM reservation) rather than a carve out of the slab — the exact
        // set the sizing pass filtered out.
        let mut n_view = 0usize;
        for (i, td) in blob.tensors.iter().enumerate() {
            // §7a: the two row-parallel partials live in the PEER region, not in
            // ordinary VRAM. `o_proj`/`down` write straight into them and the
            // peers' `XReduce` reads them over XGMI, so an ordinary local
            // allocation here would have every rank reduce three slots its peers
            // never wrote — a wrong token, with no fault and no message.
            //
            // A non-owning view: the storage belongs to the `TpRank`'s peer
            // allocation, which outlives the engine. `devgen` only declares
            // these two tensors when tp > 1.
            let peer_slot = match (tp, td.name.as_str()) {
                (Some(t), "act.og_tp") => Some(t.scratch_base),
                (Some(t), "act.dg_tp") => Some(t.scratch_base + t.slot_b),
                // Slot 2, the GATHER slot: a column-parallel partial the reduce out of
                // slot 0 folds in (`PARTIAL_SLOTS`). Only K3's LatentMoE declares it.
                (Some(t), "act.ug_tp") => Some(t.scratch_base + 2 * t.slot_b),
                // Slots 3/4/5: the sequence-parallel seams' band results (normed hidden,
                // latent xe, route table), all-gathered by op 26. K3 declares them only
                // under `PLOW_SEQ_PAR_SEAMS`; the group sizes the region to six slots then.
                (Some(t), "act.h2_tp") => Some(t.scratch_base + 3 * t.slot_b),
                (Some(t), "act.xe_tp") => Some(t.scratch_base + 4 * t.slot_b),
                (Some(t), "act.rt_tp") => Some(t.scratch_base + 5 * t.slot_b),
                _ => None,
            };
            if let Some(base) = peer_slot {
                tracing::debug!(
                    name = %td.name, base = format_args!("{base:#x}"), bytes = td.bytes,
                    "bound into the peer region"
                );
                devp.push(DeviceMem::view(base, td.bytes.max(1)));
                names.push(td.name.clone());
                n_view += 1;
                continue;
            }
            if is_band_view(&td.name) {
                let base_name = td.name.split("@band").next().unwrap_or_default();
                let base_ix = names.iter().position(|n| n == base_name).ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "band view {} names a base that is not bound before it",
                        td.name
                    ))
                })?;
                let rank = u64::from(tp.map_or(0, |t| t.rank));
                let base_mem: &DeviceMem = &devp[base_ix];
                let off = rank * td.bytes;
                if off + td.bytes > base_mem.len {
                    return Err(RuntimeError::Device(format!(
                        "band view {} at rank {rank} ends at {} B past its base {base_name} \
                         ({} B)",
                        td.name,
                        off + td.bytes,
                        base_mem.len
                    )));
                }
                devp.push(DeviceMem::view(base_mem.base + off, td.bytes.max(1)));
                names.push(td.name.clone());
                n_view += 1;
                continue;
            }
            // Full-layer KV under VMM: the base is the pool's VA reservation,
            // a non-owning view (the pool owns unmap/release). No allocation
            // and no memset — the VA is mapped lazily at each sequence's
            // frontier, and KV is always written before it is read.
            let vmm_va = cache_va[i];
            let mem = match (vmm_va, &weight_slab) {
                (Some(va), _) => {
                    n_view += 1;
                    DeviceMem::view(va, td.bytes.max(1))
                }
                // Carve from the VMM reservation, in blob order, waiting for
                // the mapper to commit through this tensor's padded end BEFORE
                // handing the view out — every write path (upload ring, shard
                // gather, memset tail) starts after the carve, so this one
                // wait point covers them all. The mapper outruns the upload,
                // so the wait is ~0 after the first chunk.
                (None, WeightSlab::Vmm(slab)) => {
                    let m = DeviceMem::view(slab.base() + slab_off, slab_need(td.bytes));
                    slab_off += slab_pad(slab_need(td.bytes));
                    slab.wait_mapped(slab_off)?;
                    m
                }
                // Carve from the one allocation, in blob order. The sizing pass
                // walked this same list with the same filter and the same
                // `slab_need`, so the cursor cannot run past the end.
                (None, WeightSlab::Flat(slab)) => {
                    let m = DeviceMem::view(slab.base + slab_off, slab_need(td.bytes));
                    slab_off += slab_pad(slab_need(td.bytes));
                    m
                }
                (None, WeightSlab::PerTensor) => {
                    let t = Instant::now();
                    let m = EngineDevice::alloc(&*be, td.bytes.max(1))?;
                    LoadProf::add(&prof.alloc_ns, t);
                    m
                }
            };
            let prof = &prof;
            // `scrub` — rewrite 0x80 (OCP e4m3 `-0`) to 0x00 inside the slab copy. Set for
            // every F8_E4M3 checkpoint payload (dense-FFN projections, fp8-twin weights…),
            // value-identical everywhere; the point is the CDNA3 maskless staging decode
            // (`mpf_fp8x4_to_bf16_h`, op_moe.h), whose contract is "no 0x80 can reach me".
            // The routed-expert packing loop applies the same rule on its own path.
            let push = |ring: &mut crate::device::hsa::HsaUploadRing,
                        src: &[u8],
                        scrub: bool|
             -> Result<()> {
                for (o, chunk) in src.chunks(STAGE).enumerate() {
                    let t = Instant::now();
                    let at = mem.base + (o * STAGE) as u64;
                    if scrub {
                        ring.push_scrub_fp8_neg0(at, chunk)?;
                    } else {
                        ring.push(at, chunk)?;
                    }
                    // Staging memcpy and DMA wait are no longer separable: the
                    // point of the ring is that they overlap. One `stage_ns`
                    // counter is the honest shape; `dma_ns` stays zero.
                    LoadProf::add(&prof.memcpy_ns, t);
                    prof.chunks.set(prof.chunks.get() + 1);
                }
                Ok(())
            };

            // The two MoE pointer tables are named like weights and are not
            // weights: `expert_weight_table`/`expert_scale_table` hold DEVICE
            // ADDRESSES the host computes after packing the experts, and no
            // checkpoint contains them. Left to the branch below they resolve to
            // `MISSING WEIGHT` and GLM cannot bind at all. They fall through to
            // the zeroing tail — a zero entry is the kernel's "not my expert" —
            // and `bind_packed_experts` fills them once the packing is done.
            //
            // Classified by EXCLUSION of the compiler's own namespaces (`packet::names`),
            // not by an allowlist of weight prefixes. The allowlist here was
            // `model.` | `fp8/` | `lm_head` — three of the four arms the five sites in the
            // tree each spelled differently — and it holds for every model shipped so far
            // only because their weights happen to be under `model.`. Kimi-K3's are not:
            // all 497 052 language-tower tensors are `language_model.model.…` and NONE
            // starts with `model.`, so an allowlist binds nothing, uploads nothing, and
            // decodes from zeroed weights without a word.
            let is_weight = packet::names::is_checkpoint_weight(&td.name);
            if is_weight {
                prefetch_ahead(&mut pf, 1);
                // `fp8/` routes to the twin checkpoint with the prefix
                // stripped; everything else to the base one.
                let is_fp8 = td.name.starts_with("fp8/");
                let src_ckpt = if is_fp8 {
                    fp8_ckpt.as_ref()
                } else {
                    ckpt.as_deref()
                };
                // KIMI-K3's ATTNRES SCORE WEIGHT IS DERIVED, NOT STORED. Resolved
                // before the ordinary lookup, which would otherwise report MISSING
                // WEIGHT for 186 tensors on a 93-layer model.
                //
                // NOT a `continue`: the loop tail pushes `devp`/`names` for EVERY
                // tensor, and skipping it would shift every later tensor's index
                // against the table the packet was compiled with.
                let folded = match src_ckpt {
                    Some(c) => fold_res_score(c, &td.name).transpose()?,
                    None => None,
                };
                if let Some(folded) = folded {
                    if folded.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "{}: folded score weight is {} B, blob declares {}",
                            td.name,
                            folded.len(),
                            td.bytes
                        )));
                    }
                    push(&mut ring, &folded, false)?;
                    wbytes += td.bytes;
                } else if let Some(c) = src_ckpt {
                    // BOTH spellings, because the twin checkpoints disagree
                    // with each other. `/home/lava/models/g31b-fp8w` KEEPS the
                    // `fp8/` prefix in its tensor names; the C reference strips
                    // it. Trying the packet's name first and the stripped form
                    // second costs one hash lookup and works with either
                    // convention, which is better than encoding a guess about
                    // which artifact someone hands us.
                    let stripped = td.name.strip_prefix("fp8/").unwrap_or(&td.name);
                    let (resolved, (src, shape)) = c
                        .tensor_ex(&td.name)
                        .map(|v| (td.name.as_str(), v))
                        .or_else(|| c.tensor_ex(stripped).map(|v| (stripped, v)))
                        .ok_or_else(|| {
                            RuntimeError::Device(format!(
                                "MISSING WEIGHT: {} (tried that name and the \
                                 `fp8/`-stripped form{})",
                                td.name,
                                if is_fp8 { " in PLOW_FP8_DIR" } else { "" }
                            ))
                        })?;
                    // THE DSA LIGHTNING INDEXER'S TWO PROJECTIONS: block-fp8 on
                    // disk, bf16 in the blob. Resolved before `slice_for` for the
                    // same reason the fold above is resolved before the ordinary
                    // lookup — handed an fp8 byte count against a bf16
                    // declaration, `slice_for` refuses a checkpoint that is
                    // perfectly bindable, and it refuses it 200 GiB in. See
                    // `asset::dsa_indexer`: a shim for two named tensors, not a
                    // dtype-coercion layer. `preflight_weights` has already taken
                    // this decision at t=0, so the `?` here cannot fire.
                    let upcast = match crate::asset::dsa_indexer::plan(c, stripped, td.bytes) {
                        Some(p) => {
                            let t = Instant::now();
                            let bf16 = p?.dequantise();
                            LoadProf::add(&prof.gather_ns, t);
                            bf16
                        }
                        None => None,
                    };
                    if let Some(bf16) = upcast {
                        if bf16.len() as u64 != td.bytes {
                            return Err(RuntimeError::Device(format!(
                                "{}: upcast produced {} B, blob declares {}",
                                td.name,
                                bf16.len(),
                                td.bytes
                            )));
                        }
                        // `scrub = false`, and it MATTERS. `is_fp8_e4m3(resolved)`
                        // is TRUE for the source name, and pushing dequantised
                        // BF16 through the 0x80 scrub would zero the high byte of
                        // every bf16 that happens to be a small negative — a
                        // silent corruption of the tensor that picks which KV rows
                        // attention sees, and one that still answers fluently.
                        push(&mut ring, &bf16, false)?;
                        wbytes += td.bytes;
                        nweights += 1;
                    } else {
                        // At n_gpu == 1 this borrows the whole mmap range and the
                        // size check inside is the old `SIZE MISMATCH`. Above 1 it
                        // is the rank's shard — classified by the CHECKPOINT name
                        // (`stripped`), so an fp8 twin shards exactly like its bf16
                        // counterpart instead of falling through as replicated.
                        if do_prefault {
                            if let Some(s) = touched(c, stripped, td.bytes, rank, n_gpu) {
                                prefault(s, &prof);
                            }
                        }
                        let t = Instant::now();
                        let slice = crate::asset::shard::slice_for(
                            stripped, src, shape, td.bytes, rank, n_gpu,
                        )?;
                        LoadProf::add(&prof.gather_ns, t);
                        push(&mut ring, &slice, c.is_fp8_e4m3(resolved))?;
                        wbytes += td.bytes;
                        nweights += 1;
                    }
                } else if td.name.starts_with("fp8/") {
                    return Err(RuntimeError::Device(format!(
                        "packet declares fp8 weights ({}) but PLOW_FP8_DIR is not set",
                        td.name
                    )));
                }
            } else if let Some(r) = &td.init {
                push(&mut ring, &blob.init[r.clone()], false)?;
            } else if let Some(g) = gen_by_tensor.get(&(i as u32)) {
                let data = g.generate().ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "devblob: gen recipe for `{}` has unknown kind {}",
                        td.name, g.kind
                    ))
                })?;
                if data.len() as u64 != td.bytes {
                    return Err(RuntimeError::Device(format!(
                        "devblob: gen recipe for `{}` produced {} B, decl says {}",
                        td.name,
                        data.len(),
                        td.bytes
                    )));
                }
                push(&mut ring, &data, false)?;
            } else if vmm_va.is_none() && !kv_skips_zeroing(&td.name) {
                // A VMM window is (mostly) UNMAPPED VA — a memset would fault,
                // not merely waste time. The `kv.` clause below is the older
                // and independent reason to skip.
                //
                // Attention reads only [0, kvlen), every row of which is
                // written before it is read, so the KV cache needs no zeroing —
                // 11.5 GiB of memset skipped on this model. Other scratch stays
                // zeroed: cheap, and conservative where the argument is less
                // obviously airtight.
                let t = Instant::now();
                EngineDevice::memset_d8(&*be, mem.base, 0, td.bytes as usize)?;
                LoadProf::add(&prof.memset_ns, t);
            }
            devp.push(mem);
            names.push(td.name.clone());
        }
        // EVERY named tensor is bound only once its copy has retired. The ring
        // leaves copies in flight by design, so this is the line that makes
        // "uploaded" mean uploaded.
        ring.drain()?;
        // Slab tail join: the carve-site waits covered every tensor that was
        // carved, but writes AFTER this loop (`bind_packed_experts` fills the
        // expert pointer tables in place) need the WHOLE span committed —
        // VMM has no demand paging. ~0 in practice: the mapper finished while
        // the upload ran.
        if let WeightSlab::Vmm(slab) = &weight_slab {
            let t = Instant::now();
            slab.wait_mapped(slab_bytes)?;
            LoadProf::add(&prof.alloc_ns, t);
        }
        // The MoE half of the bind, and it has to be here: it needs the tensor
        // table (to find each layer's two pointer slots) and the staging ring
        // (which must outlive it — the C reference records that gathering a
        // row-parallel slice into a MALLOC'd buffer faults the SDMA engine,
        // because the copy does not pin its source).
        prof.report("named tensors", t_tensors.elapsed(), wbytes);
        if !matches!(weight_slab, WeightSlab::PerTensor) {
            // The sizing pass and the carve walked the same list with the same
            // filter, so the cursor must land exactly on the total: short wastes
            // the tail, long means two tensors were aliased onto the same bytes
            // and the weights are quietly wrong. The loop above cannot exit
            // early — an error returns from the function — so unlike `exec::gpu`
            // this needs no "did it finish" guard.
            debug_assert_eq!(
                slab_off, slab_bytes,
                "weight slab carve did not consume exactly the sized span"
            );
            // `carved` is what the pool would have been asked for per tensor;
            // the rounding it no longer pays is invisible from here (ROCr never
            // reports it), so the honest thing to log is the request, and the
            // saving is read off `MEMORY_AVAIL` by whoever is measuring.
            tracing::info!(
                slab_mib = slab_bytes / (1 << 20),
                carved = devp.len() - n_view,
                views = n_view,
                "weights carved from one allocation"
            );
        }
        let mut expert_bufs = Vec::new();
        if let Some(c) = ckpt.as_ref() {
            let eprof = LoadProf::default();
            let t_exp = Instant::now();
            let (bufs, bytes, resident_weights) = bind_packed_experts(
                &be,
                &blob,
                c,
                &devp,
                &names,
                &mut ring,
                rank,
                n_gpu,
                &eprof,
                do_prefault,
                // Workers populate their own spans; `prefetch_threads() == 0`
                // (the pool disabled) also turns that readahead off.
                prefetch.is_some(),
                &resident_tables,
            )?;
            if let Some(moe) = moe_aiter.as_mut() {
                moe.bind_resident(resident_weights);
            }
            eprof.report("packed experts", t_exp.elapsed(), bytes);
            expert_bufs = bufs;
            wbytes += bytes;
        }
        // Dense-FFN prefill tables. Must run AFTER the named-weight upload above
        // (it is those uploads that give the projections their device addresses)
        // and is a no-op on a decode-only blob, which declares no such table.
        let n_dense_tab = bind_dense_ffn_tables(&be, &blob, &devp, &names)?;
        if n_dense_tab > 0 {
            tracing::info!(
                tables = n_dense_tab,
                "dense-FFN prefill pointer tables bound (grouped-arm 1-expert path)"
            );
        }
        drop(ring);
        if ckpt.is_some() {
            let s = t_rank.elapsed().as_secs_f64();
            let gib = wbytes as f64 / (1u64 << 30) as f64;
            tracing::info!(
                rank,
                gib = format!("{gib:.2}").as_str(),
                tensors = nweights,
                secs = format!("{s:.1}").as_str(),
                gib_s = format!("{:.2}", if s > 0.0 { gib / s } else { 0.0 }).as_str(),
                "checkpoint weights uploaded"
            );
        } else {
            tracing::warn!(
                "NO CHECKPOINT — weights are uninitialised; timings are real, tokens are not"
            );
        }
        let table: Vec<u8> = devp.iter().flat_map(|m| m.base.to_le_bytes()).collect();
        if let Some(cache) = &mut shared_prefix {
            cache.bind(&devp.iter().map(|m| m.base).collect::<Vec<_>>());
        }
        let d_tens = EngineDevice::alloc(&*be, table.len().max(1) as u64)?;
        EngineDevice::upload(&*be, &d_tens, 0, &table)?;
        let kda_key_factor_half =
            if k_kda_key_factor_wu.is_some() && k_kda_key_factor_carry.is_some() {
                kda_key_factor_scratch_half_bytes(&blob.progs[..dec_ix])?
            } else {
                0
            };
        let d_kda_key_factor_scratch = if kda_key_factor_half != 0 {
            let bytes = kda_key_factor_half.checked_mul(2).ok_or_else(|| {
                RuntimeError::Device("KDA key-factor scratch pair size overflows".into())
            })?;
            tracing::info!(bytes, "allocating one reusable KDA key-factor scratch pair");
            Some(EngineDevice::alloc(&*be, bytes)?)
        } else {
            None
        };
        let kda_key_factor_scratch = d_kda_key_factor_scratch
            .as_ref()
            .map(|m| (m.base, kda_key_factor_half));
        let kda_keyfeed_half =
            if k_kda_chunk_wu_lean_keys.is_some() && k_kda_chunk_carry_keyfeed.is_some() {
                kda_keyfeed_scratch_half_bytes(&blob.progs[..dec_ix])?
            } else {
                0
            };
        let d_kda_keyfeed_scratch = if kda_keyfeed_half != 0 {
            let bytes = kda_keyfeed_half.checked_mul(2).ok_or_else(|| {
                RuntimeError::Device("KDA keyfeed scratch pair size overflows".into())
            })?;
            tracing::info!(bytes, "allocating one reusable KDA keyfeed scratch pair");
            Some(EngineDevice::alloc(&*be, bytes)?)
        } else {
            None
        };
        let kda_keyfeed_scratch = d_kda_keyfeed_scratch
            .as_ref()
            .map(|m| (m.base, kda_keyfeed_half));
        let (stage1_a4_payload, stage1_a4_scales) =
            if k_moe_stage1_a4_quant.is_some() && k_moe_stage1_a4_reuse.is_some() {
                moe_stage1_a4_scratch_bytes(&blob.progs[..dec_ix], &blob.tensors)?
            } else {
                (0, 0)
            };
        let d_moe_stage1_a4_scratch = if stage1_a4_payload != 0 {
            let bytes = stage1_a4_payload
                .checked_add(stage1_a4_scales)
                .ok_or_else(|| {
                    RuntimeError::Device("lean MoE stage-1 scratch size overflows".into())
                })?;
            tracing::info!(bytes, "allocating reusable sorted A4 MoE stage-1 scratch");
            Some(EngineDevice::alloc(&*be, bytes)?)
        } else {
            None
        };
        let stage1_a4_scratch = d_moe_stage1_a4_scratch
            .as_ref()
            .map(|m| (m.base, m.base + stage1_a4_payload));

        let mla_fold = if use_mla_fold {
            Some(amd_mla_fold::MlaFold::load(
                &be, hsaco_dir, &blob.progs[..dec_ix], &blob.tensors, &devp, &mut modules,
            )?)
        } else {
            None
        };

        // OPTIONAL, because a BLOCK asset is not a model. A block takes
        // `act.x` in and gives `act.x` out — it has no embedding, no lm_head,
        // no argmax, and therefore none of `in.ids`/`in.pos`/`in.kvlen`/
        // `act.logits`. Requiring them refused the single most useful A/B
        // vehicle in the tree (one layer, one precision difference) for want of
        // tensors that layer has no reason to own.
        let find = |n: &str| names.iter().position(|x| x == n);
        let t_ids = find("in.ids");
        let t_pos = find("in.pos");
        let t_kvlen = find("in.kvlen");
        let t_active = find("in.parked");
        let t_logits = find("act.logits");
        // The context bound is carried by in.pos, not by any prefill bucket.
        let max_ctx = t_pos.map_or(0, |t| (blob.tensors[t].bytes / 4) as usize);

        // --- per-program tables ---------------------------------------------
        let ctr_banks: u64 = if ctr_dbuf() { 2 } else { 1 };
        let mut progs = Vec::with_capacity(blob.progs.len());
        for (prog_ix, p) in blob.progs.iter().enumerate() {
            let seg_class = derive_segments(p)?;
            let decode_routes = if prog_ix >= dec_ix {
                decode_segment_routes(p, &blob.tensors, &blob.init, &devp)?
            } else {
                vec![DecodeSegmentRoute::Interpreter; seg_class.len()]
            };
            let mut prefill_routes = if prog_ix < dec_ix {
                moe_mxfp4_routes_with_scratch(
                    p,
                    &blob.tensors,
                    &devp,
                    stage1_a4_scratch,
                    tp.map(|b| (b.rank, b.n_gpu)),
                )?
            } else {
                vec![PrefillSegmentRoute::Interpreter; seg_class.len()]
            };
            if prog_ix < dec_ix {
                add_kda_key_factor_routes(p, &devp, &mut prefill_routes, kda_key_factor_scratch)?;
                mla_materialized_routes(p, &devp, &mut prefill_routes)?;
                if use_sparse_mla {
                    for (seg, route) in amd_sparse_mla::routes(p, &blob.tensors, seg_class.len())?
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(route) = route {
                            if !matches!(prefill_routes[seg], PrefillSegmentRoute::Interpreter) {
                                return Err(RuntimeError::Device(
                                    "sparse MLA overlaps another native route".into(),
                                ));
                            }
                            prefill_routes[seg] = PrefillSegmentRoute::SparseMla(route);
                        }
                    }
                }
                if use_mla_fold {
                    for (seg, route) in amd_mla_fold::routes(p, &blob.tensors, seg_class.len())?
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(route) = route {
                            if !matches!(prefill_routes[seg], PrefillSegmentRoute::Interpreter) {
                                return Err(RuntimeError::Device(
                                    "native MLA fold overlaps another route".into(),
                                ));
                            }
                            prefill_routes[seg] = PrefillSegmentRoute::MlaFold(route);
                        }
                    }
                }
                if use_gemm_lt {
                    for (seg, route) in amd_gemm_lt::routes(p, &blob.tensors, seg_class.len())?
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(route) = route {
                            if !matches!(prefill_routes[seg], PrefillSegmentRoute::Interpreter) {
                                return Err(RuntimeError::Device(
                                    "hipBLASLt projection overlaps another native route".into(),
                                ));
                            }
                            prefill_routes[seg] = PrefillSegmentRoute::GemmLt(route);
                        }
                    }
                }
                if use_index_tp {
                    for (seg, route) in
                        amd_index_tp::routes(p, &blob.tensors, seg_class.len(), tp.unwrap())?
                            .into_iter()
                            .enumerate()
                    {
                        if let Some(route) = route {
                            if !matches!(prefill_routes[seg], PrefillSegmentRoute::Interpreter) {
                                return Err(RuntimeError::Device(
                                    "TP indexer overlaps another native route".into(),
                                ));
                            }
                            prefill_routes[seg] = PrefillSegmentRoute::IndexTp(route);
                        }
                    }
                }
                if use_moe_aiter {
                    for (seg, route) in amd_moe_aiter::routes(p, &blob.tensors, seg_class.len())?
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(route) = route {
                            if !matches!(prefill_routes[seg], PrefillSegmentRoute::Interpreter) {
                                return Err(RuntimeError::Device(
                                    "AITER MoE overlaps another native route".into(),
                                ));
                            }
                            prefill_routes[seg] = PrefillSegmentRoute::MoeAiter(route);
                        }
                    }
                }
                promote_kda_intra_wave_items_routes(p, &seg_class, &mut prefill_routes)?;
                promote_kda_carry_regstate_routes(p, &seg_class, &devp, &mut prefill_routes)?;
                promote_kda_wu_lean_routes(
                    p,
                    &seg_class,
                    &devp,
                    &mut prefill_routes,
                    kda_keyfeed_scratch,
                )?;
                for (seg, &class) in seg_class.iter().enumerate() {
                    let graph_phase = graph_phase_segments[prog_ix].contains(&seg);
                    if class == 19 || graph_phase {
                        if !matches!(
                            prefill_routes[seg],
                            PrefillSegmentRoute::Interpreter
                                | PrefillSegmentRoute::XReduceAttnRes { .. }
                        ) {
                            return Err(RuntimeError::Device(format!(
                                "XReduce wave-RS segment {seg} overlaps another specialist route"
                            )));
                        }
                        let dispatch = ProgramDispatch::classify(
                            p.l2_domains,
                            seg_class.len(),
                            p.gq_seg_ofs.len().saturating_sub(1),
                        );
                        if graph_phase && !prefill_segment_specialization_allowed(dispatch) {
                            return Err(RuntimeError::Device(format!(
                                "graph phase segment {seg} in program {prog_ix} has legacy L2-domain topology"
                            )));
                        }
                        prefill_routes[seg] = if graph_phase {
                            PrefillSegmentRoute::GraphPhaseXReduceWaveRs
                        } else {
                            PrefillSegmentRoute::XReduceWaveRs
                        };
                    }
                }
                let has_materialized = prefill_routes.iter().any(|r| {
                    matches!(
                        r,
                        PrefillSegmentRoute::MlaMaterializePack { .. }
                            | PrefillSegmentRoute::MlaMaterializedPrefill { .. }
                    )
                });
                let has_ep = prefill_routes.iter().any(|r| {
                    matches!(
                        r,
                        PrefillSegmentRoute::MoeEpAlign(_)
                            | PrefillSegmentRoute::MoeStage1A4Reuse(_)
                            | PrefillSegmentRoute::MoeEpStage2(_)
                            | PrefillSegmentRoute::MoeEpCombine(_)
                    )
                }) && has_moe_prefill_ep(p);
                let dispatch = ProgramDispatch::classify(
                    p.l2_domains,
                    seg_class.len(),
                    p.gq_seg_ofs.len().saturating_sub(1),
                );
                if has_materialized && !prefill_segment_specialization_allowed(dispatch) {
                    return Err(RuntimeError::Device(format!(
                        "prefill program {prog_ix} (T={}) places materialized MLA raw opcodes in L2-domain windows; standalone objects require ordered wave segments",
                        p.t
                    )));
                }
                if (use_sparse_mla || use_moe_aiter || use_index_tp || use_gemm_lt || use_mla_fold)
                    && !prefill_segment_specialization_allowed(dispatch)
                {
                    return Err(RuntimeError::Device(
                        "native AITER kernels require ordered segment dispatch".into(),
                    ));
                }
                if has_ep && !prefill_segment_specialization_allowed(dispatch) {
                    return Err(RuntimeError::Device(format!(
                        "prefill program {prog_ix} (T={}) places expert-parallel raw opcodes in L2-domain windows; standalone objects require ordered wave segments",
                        p.t
                    )));
                }
            }
            let packed_seg_family = derive_packed_segment_families(p)?;
            let raw_mla_v2_segment = derive_raw_mla_v2_segments(p)?;
            let packed_mla_segmented = packed_family_segments_cover(p, &packed_seg_family, &[5, 6]);
            let packed_kda_segmented = packed_family_segments_cover(p, &packed_seg_family, &[7]);
            let up = |bytes: &[u8]| -> Result<DeviceMem> {
                let m = EngineDevice::alloc(&*be, bytes.len().max(1) as u64)?;
                if !bytes.is_empty() {
                    EngineDevice::upload(&*be, &m, 0, bytes)?;
                }
                Ok(m)
            };
            let mut xreduce_attnres_args = Vec::new();
            for route in &mut prefill_routes {
                let PrefillSegmentRoute::XReduceAttnRes {
                    args, device_args, ..
                } = route
                else {
                    continue;
                };
                let bind = tp.ok_or_else(|| {
                    RuntimeError::Device(
                        "fused XReduceTwoShot+AttnRes requires a tensor-parallel binding".into(),
                    )
                })?;
                let status_id = args.status as u32;
                args.peer_scratch = bind.peer_table;
                args.xctr = bind.xctr;
                args.rank = bind.rank;
                args.nranks = bind.n_gpu;
                args.status = status_id.checked_sub(1).map_or(0, |id| {
                    bind.xctr + u64::from(id) * (CTR_STRIDE_U32 * 4) as u64
                });
                let mem = up(as_bytes(std::slice::from_ref(args)))?;
                *device_args = mem.base;
                xreduce_attnres_args.push(mem);
            }
            let d_inst = up(as_bytes(&p.insts))?;
            let d_stream = up(as_bytes(&p.stream))?;
            let d_sofs = up(as_bytes(&p.stream_ofs))?;
            let d_slen = up(as_bytes(&p.stream_len))?;
            let d_waits = up(as_bytes(&p.waits))?;
            let d_succs = up(as_bytes(&p.succs))?;
            // Counters are allocated, never uploaded — they are re-armed per
            // dispatch group.
            //
            // TWO BANKS when `ctr_dbuf` is on, and then they ARE zeroed here:
            // the double-buffered `run` clears the STALE bank after enqueueing,
            // so the bank a dispatch actually reads was cleared one dispatch
            // ago and the very first dispatch has no such predecessor. Without
            // this memset it would run out of whatever the allocator handed
            // back. (The single-bank path still clears synchronously in
            // `rearm`, so its memset is redundant but harmless and costs one
            // memset at load.)
            let ctr_span = (p.n_counter as usize * CTR_STRIDE_U32 * 4).max(1) as u64;
            let d_ctr = EngineDevice::alloc(&*be, ctr_span * ctr_banks)?;
            EngineDevice::memset_d8(&*be, d_ctr.base, 0, (ctr_span * ctr_banks) as usize)?;
            // The VRAM price of the second bank, per program, so it is a number
            // in the log rather than an argument in a comment.
            tracing::debug!(
                prog = progs.len(),
                n_counter = p.n_counter,
                bank_kib = ctr_span / 1024,
                banks = ctr_banks,
                "counter banks"
            );
            // Per-(CU, segment) windows. DERIVED from the stream we just
            // uploaded, so it cannot disagree with it — the standing failure of
            // a precomputed window table (`PLOW_SEG_OFF` rewrote `stream[].seg`
            // and left `gq_seg_ofs` describing a stream that no longer existed,
            // which ran one segment and reported it as the whole prefill).
            // `n_seg` comes from `derive_segments`, which is `max(seg)+1` — the
            // same count `run_segmented` launches.
            let seg_ofs = static_seg_ofs(
                &p.stream,
                &p.stream_ofs,
                &p.stream_len,
                seg_class.len() as u32,
            )
            .map_err(RuntimeError::Device)?;
            let d_seg_ofs = up(as_bytes(&seg_ofs))?;

            let gq = if p.gq_stream.is_empty() {
                None
            } else {
                let n_seg = p.gq_seg_ofs.len().saturating_sub(1) as u32;
                // DOUBLE-BUFFERED TOO. The GQ cursor is re-armed in the same
                // breath as the counters and is just as much a dispatch's live
                // state, so banking one without the other would have dispatch
                // N+1 resume from dispatch N's cursor.
                let cur_span = (n_seg.max(1) as usize * CTR_STRIDE_U32 * 4) as u64;
                let d_cursor = EngineDevice::alloc(&*be, cur_span * ctr_banks)?;
                EngineDevice::memset_d8(&*be, d_cursor.base, 0, (cur_span * ctr_banks) as usize)?;
                Some(AmdGq {
                    d_stream: up(as_bytes(&p.gq_stream))?,
                    d_seg_ofs: up(as_bytes(&p.gq_seg_ofs))?,
                    d_cursor,
                    n_seg,
                    cur_span,
                })
            };
            progs.push(AmdProg {
                t: p.t,
                packed_prefill_only: p.packed_prefill_only,
                packed_dense_error: check_packed_dense_program(&p.insts)
                    .err()
                    .map(|e| e.to_string()),
                packed_dense: p.insts.iter().any(|d| d.op == DevOp::FlashPrefill as u16),
                packed_needs_mla: p.insts.iter().any(|d| {
                    d.op == DevOp::RmsNorm as u16
                        || d.op == DevOp::HeadNormRope as u16
                        || d.op == DevOp::HeadNormRopeFp8 as u16
                        || d.op == DevOp::FlashMlaPrefill as u16
                        || d.op == DevOp::FlashMlaPrefillFp8 as u16
                }),
                packed_mla_compatible: packed_mla_compatible(p),
                packed_mla_segmented,
                packed_needs_kda: p.insts.iter().any(|d| {
                    d.op == DevOp::KdaConv3 as u16
                        || d.op == DevOp::KdaStateStep as u16
                        || d.op == DevOp::KdaStateStepG as u16
                        || matches!(
                            DevOp::from_u16(d.op),
                            Some(
                                DevOp::KdaConv
                                    | DevOp::KdaChunkPrepare
                                    | DevOp::KdaChunkIntra
                                    | DevOp::KdaChunkWu
                                    | DevOp::KdaChunkCarry
                                    | DevOp::KdaConvStateStepG
                            )
                        )
                }),
                packed_kda_compatible: packed_kda_compatible(p),
                packed_kda_segmented,
                packed_recurrent_spans: crate::exec::amd::packed::recurrent_span_limit(
                    p.insts.iter().map(|d| d.op),
                ),
                decode_routes,
                prefill_routes,
                _xreduce_attnres_args: xreduce_attnres_args,
                n_inst: p.insts.len() as u32,
                trace_records: p.stream.len(),
                n_counter: p.n_counter,
                d_inst,
                d_stream,
                d_sofs,
                d_slen,
                d_waits,
                d_succs,
                d_ctr,
                d_seg_ofs,
                seg_class,
                packed_seg_family,
                raw_mla_v2_segment,
                gq,
                l2_domains: p.l2_domains,
                // DERIVED, not carried in the blob. The emitter appends the two-level
                // maintenance scratch to the tail of the counter region, three u32 per
                // (packet, domain), so its base is implied by fields the header already has.
                // That keeps `PlowProgHeader` at 24 bytes and the blob format unchanged.
                //
                // Zero unless the program is L2-placed: without per-domain windows there is no
                // `nper`, the emitter allocates no scratch, and the interpreter reads 0 as
                // "no hierarchy".
                hier_base: if p.l2_domains != 0 {
                    (p.n_counter).saturating_sub(3 * p.insts.len() as u32 * p.l2_domains)
                } else {
                    0
                },
                ctr_span,
                bank: CounterBankState::new(),
            });
        }
        let decode = progs.len() - 1;
        // THE DECODE BATCH LADDER: programs `[dec_lo, decode]` are decode rungs at ascending
        // widths, `[0, dec_lo)` the prefill bucket ladder. Same value the per-phase object
        // requirements were split at above — one rule, computed once.
        let dec_lo = dec_ix;
        log_gate_hier_status(
            &blob.progs,
            dec_lo,
            decode_objects != 0 && decode_objects_gate_hier == decode_objects,
        );
        // Packet trace (`PLOW_TRACE_RAW=<path>`). Zeroed once at allocation so
        // an entry the run never reaches reads as a zero record rather than as
        // whatever the allocator handed back; every executed slot is rewritten
        // each step, so the buffer always holds the LAST step's timeline.
        let (d_trace, trace_bytes) = if crate::config::RuntimeConfig::get().amd.trace_raw.is_some()
        {
            // Sized for the WIDEST program, not the decode one. The pointer used to be handed
            // only to `decode` (see the kernarg builder), so a prefill dispatch got a null trace
            // and recorded nothing — prefill was untraceable BY CONSTRUCTION, which is why the
            // first prefill trace ever taken came back with 0 packets. K3's prefill buckets carry
            // 2942 stream entries against decode's 2459, so sizing by `decode` alone would have
            // overflowed the buffer the moment the pointer was handed over.
            let bytes =
                blob.progs.iter().map(|g| g.stream.len()).max().unwrap_or(0) * TRACE_REC_BYTES;
            let m = EngineDevice::alloc(&*be, bytes.max(1) as u64)?;
            EngineDevice::upload(&*be, &m, 0, &vec![0u8; bytes])?;
            (Some(m), bytes)
        } else {
            (None, 0)
        };
        let metadata_rows = t_kvlen.map_or(1, |t| (blob.tensors[t].bytes / 4).max(1) as usize);
        let batch = if t_kvlen.is_none() {
            1
        } else {
            max_decode_batch as usize
        };
        if metadata_rows != batch {
            return Err(RuntimeError::Rejected(format!(
                "in.kvlen has {metadata_rows} rows incompatible with decode capacity {batch}"
            )));
        }

        // KV slot geometry, for the per-slot prefill rebase. Only meaningful at
        // batch > 1; at batch 1 the list is empty and every rebase is a no-op,
        // so a single-sequence engine is byte-identical to before.
        // THE DECODE GEMV IS A COMPILE-TIME ROW BUCKET, CAPPED AT 16.
        // `runtime/amd/op_gemm.h`: `PLOW_GEMV_MAXM 16`, and `gemv_rows<MM>`
        // carries `float acc[MM]` and loops `m < MM` — it has NO outer loop
        // over M > MM. `scripts/build_gfx950.sh` clamps `PLOW_GEMV_MM` to 16
        // to satisfy the static assert, so a blob emitted at
        // PLOW_DECODE_BATCH=32 loads against an MM=16 object and every
        // sequence from 16 up gets a ZERO logit row and samples token 0 —
        // §4's bug shape, with no fault anywhere. `plowc` does not refuse it
        // (it wrote `gv_mm_max: 32` into build.json), so refuse it here, where
        // the packet finally meets the code objects.
        //
        // THIS IS THE CEILING, NOT THE PAIRING. It compares the blob against a
        // constant no object can exceed; it says nothing about the bucket the
        // object in `hsaco_dir` was actually compiled at, so a B=8 blob on an
        // MM=1 object passes it (8 <= 16) and produces one correct row out of
        // eight. `check_gemv_capacity` is the half that compares blob against
        // OBJECT, and it runs above in `load_one`. Both are needed: this one
        // still catches a blob no build can serve, and it does so before any
        // object is opened.
        // NO LONGER A HARD CEILING: a WALKING object (`plow_gemv_walk_1`,
        // PLOW_GEMV_WALK=1) serves any M in ceil(M/MM) row-block passes, so a
        // batch above the bucket cap is servable and `check_gemv_capacity` —
        // which sees the actual object — is the gate that refuses a NON-walking
        // object at need > cap, with the correct message. What remains here is
        // a sanity bound: XArgmaxFin's bounded fold caps sequences at 128
        // (PLOW_XAMAX_MAX_BATCH), and nothing above it has ever been emitted.
        if batch > packet::devbuild::XARGMAX_MAX_BATCH as usize {
            return Err(RuntimeError::Device(format!(
                "blob is compiled PLOW_DECODE_BATCH={batch}, past the XArgmaxFin fold's \
                 {}-sequence ceiling (PLOW_XAMAX_MAX_BATCH, runtime/amd/op_collective.h). \
                 Re-emit at PLOW_DECODE_BATCH <= {}.",
                packet::devbuild::XARGMAX_MAX_BATCH,
                packet::devbuild::XARGMAX_MAX_BATCH
            )));
        }

        // The CARRIED recurrent state, per slot. Unlike `kv_slot_stride` this is built at
        // EVERY batch (a batch-1 slot has one region spanning the whole tensor), because the
        // prefix snapshot below is not a batching feature.
        //
        // `blkres` is excluded for the same reason `begin_slot` excludes it: it is sized at the
        // widest PREFILL bucket rather than at `batch`, so `bytes / batch` is not its stride —
        // and it carries nothing across passes, so a snapshot has nothing to capture.
        let carried_slot: Vec<(usize, u64)> = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| is_carried_state(&t.name) && !t.name.contains("blkres"))
            .map(|(i, t)| (i, t.bytes / batch as u64))
            .collect();

        let mut kda_conv_bank_pairs = Vec::new();
        for (alt_i, alt) in blob.tensors.iter().enumerate() {
            if !alt.name.contains(".conv_state_alt.") {
                continue;
            }
            if batch != 1 {
                return Err(RuntimeError::Device(format!(
                    "{} is a double-buffered KDA conv window, but the packet batch is {batch}; \
                     KdaConvStateStepG is B1-only",
                    alt.name
                )));
            }
            let base_name = alt.name.replacen(".conv_state_alt.", ".conv_state.", 1);
            let base_i = blob
                .tensors
                .iter()
                .position(|t| t.name == base_name)
                .ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "{} has no matching legacy KDA conv window {base_name}",
                        alt.name
                    ))
                })?;
            if devp[alt_i].len != devp[base_i].len {
                return Err(RuntimeError::Device(format!(
                    "KDA conv banks disagree: {} is {} bytes, {base_name} is {} bytes",
                    alt.name, devp[alt_i].len, devp[base_i].len
                )));
            }
            kda_conv_bank_pairs.push((devp[alt_i].base, devp[base_i].base, devp[alt_i].len));
        }
        if need_kda_conv_step_db.is_some() && kda_conv_bank_pairs.is_empty() {
            return Err(RuntimeError::Device(
                "KdaConvStateStepG packet has no alternate KDA conv-window tensors".into(),
            ));
        }

        let clear_ranges = if state_clear_device {
            state_clear_ranges(&devp, &carried_slot)?
        } else {
            Vec::new()
        };
        let n_state_clear = u32::try_from(clear_ranges.len())
            .map_err(|_| RuntimeError::Device("too many recurrent-state clear ranges".into()))?;
        let d_state_clear = if clear_ranges.is_empty() {
            None
        } else {
            let bytes = std::mem::size_of_val(clear_ranges.as_slice()) as u64;
            let d = EngineDevice::alloc(&*be, bytes)?;
            EngineDevice::upload(&*be, &d, 0, as_bytes(&clear_ranges))?;
            Some(d)
        };

        if !carried_slot.is_empty() {
            tracing::info!(
                tensors = carried_slot.len(),
                per_slot_mib = carried_slot.iter().map(|&(_, n)| n).sum::<u64>() / (1024 * 1024),
                "carried recurrent state (prefix-snapshot size per slot)"
            );
        }

        let mut kv_slot_stride: Vec<(usize, u64)> = Vec::new();
        if batch > 1 {
            // A KDA recurrent state is compiler-owned per-sequence state, so it
            // lives under `kv.` like the KV cache — but it is NOT shaped
            // `[batch][...]`. It is one `[heads, head_dim, head_dim]` f32 block
            // read-modify-written in place, with no token axis and no batch
            // axis (`devgen::kda::declare_kda_state`). Dividing its bytes by
            // `batch` below would hand slot 1 a pointer 1/batch of the way into
            // slot 0's state: no fault, no missing weight, just every sequence
            // corrupting every other one's recurrence.
            //
            // A blob emitted at `RowKind::Sequences` DOES have that axis
            // (`declare_kda_state(.., slots)`), and the carrier that tells the
            // kernel to use it is `PLOW_KDA_F_SEQ_ROWS` in the state step's
            // flags word. So the question is not "is there a recurrent state"
            // but "was this state emitted per-slot" — and only the blob can
            // answer it.
            //
            // CHECK THE CARRIER, NOT THE ENV. `batch` is itself derived from
            // `in.kvlen`, so it agrees with the emitter by construction and
            // cannot discriminate. The flag cannot be faked into being: it is
            // set only where the emitter also sized the state `slots` wide, so
            // its absence at batch > 1 means the state is one block and the
            // stride below would alias every sequence onto every other's — no
            // fault, no missing weight, just fluent wrong output.
            const KDA_F_SEQ_ROWS: u32 = 2;
            let unbatched = blob.progs[dec_ix..].iter().find_map(|p| {
                p.insts
                    .iter()
                    .any(|d| {
                        (d.op == DevOp::KdaStateStep as u16 || d.op == DevOp::KdaStateStepG as u16)
                            && d.i[4] & KDA_F_SEQ_ROWS == 0
                    })
                    .then_some(p.t)
            });
            if let (Some(t), Some(rung)) = (
                blob.tensors
                    .iter()
                    .find(|t| t.name.starts_with("kv.") && t.name.contains("state")),
                unbatched,
            ) {
                return Err(RuntimeError::Device(format!(
                    "PLOW_DECODE_BATCH = {batch} with a recurrent-state tensor `{}` whose decode \
                     rung T={rung} does NOT carry PLOW_KDA_F_SEQ_ROWS. The state is one block with no \
                     slot axis, so the per-slot stride below would alias every sequence's state \
                     onto every other's. Re-emit with PLOW_DECODE_BATCH = {batch} so the emitter \
                     sizes the state per slot and sets the flag, or run at batch 1.",
                    t.name
                )));
            }
            // `kv.blkres` IS EXCLUDED, and leaving it in was an OUT-OF-BOUNDS GPU WRITE.
            //
            // It matches `kv.` but it is not a per-sequence cache. It is K3's snapshot ring,
            // `[t][nb_cap][hidden]` (`devgen::k3`, "kv.blkres"), sized at the LARGEST `t` in the
            // blob — which is the widest PREFILL bucket, not `batch`. Dividing its bytes by
            // `batch` therefore invents a stride that has nothing to do with its layout, and
            // rebasing slot `s` onto `s * bytes/batch` walks off the end:
            //
            //   T_max 8192, batch 16  ->  stride 512 rows;  slot 15 starts at row 7680,
            //   and a 1024-row prefill chunk then writes to row 8704 of an 8192-row tensor.
            //
            // MEASURED: `Memory access fault by GPU node-7` serving the B=16 packet at
            // concurrency 16 with 1038-token prompts (chunks [1024, 512]). It never fired at
            // B=4 because the stride is 2048 there and slot 3 tops out at row 7168.
            //
            // Nothing is lost by not rebasing it: prefill and decode ALTERNATE rather than
            // overlap, and layer 0 resets the ring at the head of every forward pass, so both
            // phases can use rows `[0, t)` as scratch. It carries nothing between passes —
            // which is the same property that lets `begin_slot` skip clearing it.
            kv_slot_stride = blob
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.name.starts_with("kv.") && !t.name.contains("blkres"))
                .map(|(i, t)| (i, t.bytes / batch as u64))
                .collect();
            tracing::info!(
                batch,
                kv_buffers = kv_slot_stride.len(),
                slot_bytes = kv_slot_stride.first().map(|&(_, s)| s).unwrap_or(0),
                "KV slot geometry for per-slot prefill"
            );
        }

        // --- pinned staging --------------------------------------------------
        let n_dec_inst = blob.progs[decode].insts.len();
        let mut h_inst =
            EngineDevice::host_alloc_pinned(&*be, n_dec_inst * std::mem::size_of::<DevInst64>())?;
        h_inst
            .as_mut_slice()
            .copy_from_slice(as_bytes(&blob.progs[decode].insts));
        // Prefill stages ids AND pos for a whole chunk, so this must hold
        // 2 * max_bucket_T * 4 bytes. Sizing it at a fixed 64 KiB silently
        // truncated a T=8192 chunk's position array.
        let max_t = blob.progs.iter().map(|g| g.t as usize).max().unwrap_or(1);
        let h_scalar = EngineDevice::host_alloc_pinned(&*be, (max_t * 4 * 2).max(64 * 1024))?;
        let max_pf_inst = blob.progs[..decode]
            .iter()
            .map(|g| g.insts.len())
            .max()
            .unwrap_or(0);
        let h_pf_inst = EngineDevice::host_alloc_pinned(
            &*be,
            (max_pf_inst * std::mem::size_of::<DevInst64>()).max(64),
        )?;
        let max_pf_rows = blob.progs[..dec_lo]
            .iter()
            .map(|g| g.t as usize)
            .max()
            .unwrap_or(0);
        let prefill_span_capacity = batch.max(1);
        let prefill_row_capacity = max_pf_rows.max(1);
        let span_bytes = prefill_span_capacity * std::mem::size_of::<PrefillSpan>();
        let parked_bytes = prefill_row_capacity * std::mem::size_of::<u32>();
        let d_prefill_spans = EngineDevice::alloc(&*be, span_bytes as u64)?;
        let d_prefill_parked = EngineDevice::alloc(&*be, parked_bytes as u64)?;
        let h_prefill_meta = EngineDevice::host_alloc_pinned(&*be, span_bytes + parked_bytes)?;
        let pf_src: Vec<Vec<DevInst64>> = blob.progs.iter().map(|g| g.insts.clone()).collect();
        let max_ctr = progs
            .iter()
            .map(|p| p.n_counter as usize * CTR_STRIDE_U32 * 4)
            .max()
            .unwrap_or(4)
            .max(
                progs
                    .iter()
                    .filter_map(|p| p.gq.as_ref())
                    .map(|g| g.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4)
                    .max()
                    .unwrap_or(4),
            );
        let mut h_zero = EngineDevice::host_alloc_pinned(&*be, max_ctr.max(4))?;
        h_zero.as_mut_slice().fill(0);
        let d_token_ring = k_token_capture
            .map(|_| EngineDevice::alloc(&*be, (batch * DEFERRED_TOKEN_MAX_STEPS * 4) as u64))
            .transpose()?;

        let (kvrow, kvrow_i2) = if blob.kvrow.is_empty() {
            derive_kvrow(&blob.progs[decode], &names)
        } else {
            (blob.kvrow.clone(), Vec::new())
        };
        let kvrow_span = kvrow_span(&kvrow.iter().chain(&kvrow_i2).copied().collect::<Vec<_>>());

        // LIVE-`kv_len` MLA split policy. Opt-in: the baked count is the shipped
        // behaviour and a different split reassociates the merge, so this must be
        // asked for rather than inherited.
        //
        // EVERY decode rung, not just `decode`: with a batch ladder the mux
        // dispatches a narrower rung at low load, and patching only the widest
        // would leave the running program on its baked count.
        let mla_nsplit = crate::config::RuntimeConfig::get()
            .amd
            .mla_ns_live
            .then(|| {
                let mut baked = None;
                let mut rungs = Vec::new();
                for p in dec_lo..=decode {
                    let (sites, b) = derive_mla_nsplit(&blob.progs[p].insts)?;
                    if *baked.get_or_insert(b) != b {
                        return None; // rungs disagree; leave every one of them alone
                    }
                    let (lo, hi) = crate::exec::kvrow::kvrow_span(&sites)?;
                    rungs.push(MlaNsplitProg {
                        prog: p,
                        sites: sites.iter().map(|&i| i as usize - lo).collect(),
                        lo,
                        image: blob.progs[p].insts[lo..=hi].to_vec(),
                    });
                }
                let baked = baked?;
                tracing::info!(
                    rungs = rungs.len(), baked,
                    sites = rungs.iter().map(|r| r.sites.len()).sum::<usize>(),
                    "PLOW_MLA_NS_LIVE: MLA decode split count tracks the live kv_len"
                );
                Some(MlaNsplit { progs: rungs, baked, cur: baked })
            })
            .flatten();

        // EVERY slot's row 0 must be mapped before any batched decode runs.
        // At batch > 1 all B rows compute whether or not their slot is fed, and
        // an unfed row still writes K/V at its own `pos` — which is 0. Under
        // the flat allocation that wrote garbage into a block nobody read;
        // under VMM it would fault on unmapped VA.
        if let Some(v) = &vmm {
            for b in 0..batch {
                v.ensure_rows(b, 1)?;
            }
        }
        if let Some(v) = &shared_prefix {
            for b in 0..batch {
                v.ensure_rows(b, 1)?;
            }
        }

        tracing::info!(
            arch = %arch, n_cu = blob.n_cu, progs = progs.len(),
            variant = ?variant, prefill = ?sched_prefill, decode = ?sched_decode,
            n_kvrow = kvrow.len() + kvrow_i2.len(), max_ctx,
            vmm = vmm.is_some(),
            shared_prefix = shared_prefix.is_some(),
            "AMD engine ready"
        );
        // P3 (coalesce the two decode scalar H2Ds into one) is only legal when
        // `in.kvlen` sits immediately after `in.pos` IN THE DEVICE LAYOUT, so
        // log the two bases: the answer is a property of this blob's slab, not
        // of the code.
        tracing::debug!(
            pos = t_pos.map(|t| format!("{:#x}+{}", devp[t].base, devp[t].len)),
            kvlen = t_kvlen.map(|t| format!("{:#x}+{}", devp[t].base, devp[t].len)),
            "decode scalar tensors"
        );

        let mixed_step = if crate::config::RuntimeConfig::get().fusion && tp.is_none() && batch > 1
        {
            match mixed_step::MixedAmdStep::load(
                &be,
                &blob,
                &devp,
                hsaco_dir,
                batch,
                mixed_step::StepRoute::Mixed,
            ) {
                Ok(mixed) => {
                    tracing::info!("runtime prefill/decode fusion enabled");
                    Some(mixed)
                }
                Err(error) => {
                    tracing::warn!(%error, "runtime fusion unavailable; using ordinary execution");
                    None
                }
            }
        } else {
            None
        };
        // UNIFIED TOKEN BATCH. Loaded on the same terms as fusion and never beside it: the two
        // are alternatives for the (gfx942, dense-GQA) pair. A refusal here is by capability
        // name and leaves the ordinary route in place; it never falls back to a token-batch
        // packet on an object without the arms, because AMD's dispatch `default:` writes
        // nothing and does not trap.
        let cfg = crate::config::RuntimeConfig::get();
        let mut token_batch_refusal = if !cfg.token_batch {
            Some(token_batch::TokenBatchRefusal::NotRequested.to_string())
        } else if cfg.fusion {
            Some("explicit fusion takes precedence".to_owned())
        } else if tp.is_some() {
            Some("token batching does not support tensor parallelism".to_owned())
        } else if batch <= 1 {
            Some("token batching requires multiple slots".to_owned())
        } else if arch != "gfx942" {
            Some(format!("token batching is not qualified for {arch}"))
        } else {
            None
        };
        let token_batch_step = if token_batch_refusal.is_none() {
            match mixed_step::MixedAmdStep::load(
                &be,
                &blob,
                &devp,
                hsaco_dir,
                batch,
                mixed_step::StepRoute::TokenBatch,
            ) {
                Ok(step) => Some(step),
                Err(error) => {
                    token_batch_refusal = Some(error.to_string());
                    None
                }
            }
        } else {
            None
        };
        // Loading an executor does not establish that the scheduler ever dispatched it.
        {
            let cap = token_batch::probe_token_batch(
                hsaco_dir,
                &arch,
                |p| std::fs::read(p),
                elf_symbol_u32,
            );
            token_batch::log_route(
                &cap,
                token_batch_step.is_some(),
                token_batch_refusal.as_deref(),
            );
        }
        let engine = AmdEngine {
            mixed_step,
            token_batch_step,
            be,
            arch,
            n_cu: blob.n_cu,
            progs,
            decode,
            dec_lo,
            devp,
            _weight_slab: weight_slab,
            d_tens,
            tensor_names: names,
            _expert_bufs: expert_bufs,
            d_trace,
            trace_bytes,
            trace_write_bytes: Cell::new(0),
            k_prefill,
            k_decode,
            k_decode_mla,
            k_kda_decode_fused,
            k_grouped_moe_glu,
            k_grouped_moe_down,
            k_kda_chunk_intra_cached,
            k_kda_chunk_intra_wave_items,
            k_kda_chunk_carry_regstate,
            k_kda_key_factor_wu,
            k_kda_key_factor_carry,
            _kda_key_factor_scratch: d_kda_key_factor_scratch,
            k_kda_chunk_wu_lean,
            k_kda_chunk_wu_lean_keys,
            k_kda_chunk_carry_keyfeed,
            _kda_keyfeed_scratch: d_kda_keyfeed_scratch,
            k_moe_stage1_mxfp4,
            k_moe_stage1_a4_quant,
            k_moe_stage1_a4_reuse,
            _moe_stage1_a4_scratch: d_moe_stage1_a4_scratch,
            k_moe_stage2_mxfp4,
            k_moe_combine,
            k_attn_res_f32mix,
            k_moe_ep_align,
            k_moe_ep_stage2,
            k_moe_ep_combine,
            k_xreduce_wave_rs,
            k_mla_materialize_pack,
            k_mla_materialized_prefill,
            sparse_mla,
            moe_aiter,
            index_tp,
            gemm_lt,
            mla_fold,
            k_xaudit,
            k_state_clear,
            k_token_capture,
            d_token_ring,
            decode_tiers,
            k_flash,
            k_mla_v2_sv_raw,
            k_packed_mla_norm,
            k_packed_mla_flash,
            k_packed_kda,
            k_xr_attnres,
            packed_prefill_dense: dense_prefill_object && (k_flash.is_none() || dense_flash_object),
            packed_prefill_prefill_abi,
            sched_prefill,
            sched_decode,
            _modules: modules,
            h_inst,
            h_scalar,
            h_zero,
            h_pf_inst,
            d_prefill_spans,
            d_prefill_parked,
            h_prefill_meta,
            prefill_span_capacity,
            prefill_row_capacity,
            packed_prefill: None,
            d_state_clear,
            n_state_clear,
            pf_src,
            kvrow,
            kvrow_i2,
            kvrow_span,
            mla_nsplit,
            t_ids,
            t_pos,
            t_kvlen,
            t_active,
            t_logits,
            max_ctx,
            weights_bound: ckpt.is_some(),
            batch,
            tens_table: table,
            kv_slot_stride,
            kv_slot: 0,
            carried_slot,
            prefix_snap: (0..batch).map(|_| None).collect(),
            prefix_regions: prefix::snapshot_regions(&blob.tensors,
                blob.progs.iter().flat_map(|p| &p.insts), batch, max_ctx),
            prefix_used: vec![0; batch],
            prefix_rows: vec![0; batch],
            prefix_tick: 0,
            kda_conv_bank_pairs,
            kda_conv_alt_stale: vec![false; batch],
            vmm,
            shared_prefix,
            lm_detail: std::cell::RefCell::new(None),
            pf_stream: blob.progs.iter().map(|g| g.stream.clone()).collect(),
            tp,
            seg_enq_us: 0.0,
            seg_drain_us: 0.0,
            seg_launches: 0,
            seg_window: crate::config::RuntimeConfig::get().amd.seg_window,
        };
        engine.report_packed_prefill_route(hsaco_dir);
        Ok(engine)
    }

    /// Say once, at load, whether the packed-prefill route can actually fire.
    ///
    /// It has two silent doors and this closes both. `load_packed_family` returns `Ok(None)`
    /// for a family object that is simply absent from the hsaco directory, and
    /// `packable_prefill_span` then finds no capable rung and packs nothing — no error, no
    /// log, an unchanged number. That is exactly what `PLOW_PACKED_PREFILL_ROUTE=1` did on
    /// gfx942, where no build recipe emitted `interp_packed_mla_*` at all. The second door is
    /// the route being off on a blob that needs it: a dense packet packs on the ordinary
    /// span-aware objects, but an MLA one is refused by `check_packed_prefill_program`
    /// wherever the family objects are not loaded, so `--pf-batch` alone reads as a null.
    fn report_packed_prefill_route(&self, hsaco_dir: &Path) {
        let cfg = crate::config::RuntimeConfig::get();
        let route = cfg.amd.packed_prefill_route;
        let pf_batch = cfg.pf_batch;
        if !route && !pf_batch {
            return;
        }
        let suffix = self.sched_prefill.suffix();
        let missing: Vec<String> = [
            ("interp_packed_mla_norm", self.k_packed_mla_norm.is_some()),
            ("interp_packed_mla_flash", self.k_packed_mla_flash.is_some()),
            ("interp_packed_kda", self.k_packed_kda.is_some()),
        ]
        .into_iter()
        .filter(|&(_, loaded)| !loaded)
        .map(|(stem, _)| format!("{stem}{suffix}.elf"))
        .collect();
        let mut capable: Vec<u32> = Vec::new();
        let mut refusal: Option<String> = None;
        for (prog, width) in self.prefill_rungs().collect::<Vec<_>>() {
            let sibling = packed_prefill_topology_index(prog, self.dec_lo, |candidate| {
                self.progs
                    .get(candidate)
                    .map(|p| (p.t, p.packed_prefill_only))
            });
            match sibling {
                Some(candidate) => match self.check_packed_prefill_program(candidate) {
                    Ok(_) => capable.push(width),
                    Err(e) => {
                        refusal.get_or_insert_with(|| e.to_string());
                    }
                },
                None => {
                    refusal
                        .get_or_insert_with(|| "no packed-prefill topology sibling".to_string());
                }
            }
        }
        if capable.is_empty() {
            tracing::warn!(
                route,
                pf_batch,
                dense_consumers = self.packed_prefill_dense,
                packet_abi = self.packed_prefill_prefill_abi,
                hsaco = %hsaco_dir.display(),
                objects_missing = ?missing,
                reason = refusal
                    .as_deref()
                    .unwrap_or("this blob declares no prefill rung"),
                "packed prefill cannot fire on this blob — every prefill rung refuses"
            );
        } else if !pf_batch {
            tracing::warn!(
                route,
                rungs = ?capable,
                "packed prefill is capable but --pf-batch is off — co-packing will never be attempted"
            );
        } else {
            tracing::info!(
                route,
                rungs = ?capable,
                dense_consumers = self.packed_prefill_dense,
                packet_abi = self.packed_prefill_prefill_abi,
                objects_missing = ?missing,
                "packed prefill armed"
            );
        }    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn n_programs(&self) -> usize {
        self.progs.len()
    }

    /// Device memory backing tensor `name`, for a weight loader to fill.
    pub fn tensor_slot(&self, name: &str) -> Option<&DeviceMem> {
        self.tensor_names
            .iter()
            .position(|x| x == name)
            .map(|i| &self.devp[i])
    }

    /// A model-only tensor handle, or a clear error naming what this asset is.
    fn need(&self, h: Option<usize>, what: &str) -> Result<usize> {
        h.ok_or_else(|| {
            RuntimeError::Device(format!(
                "this blob has no `{what}` — it is a BLOCK asset (act.x in, act.x out), \
                 not a model, so token-level entry points do not apply to it"
            ))
        })
    }

    /// The lm_head instruction's tensor operands, as `(handle, name)`.
    ///
    /// The diagnostic for an all-zero `act.logits` with a healthy `act.hn`: a
    /// matmul whose WEIGHT operand resolved to nothing is memset to zero, and
    /// zero times a healthy activation is exactly all-zero logits with no error
    /// anywhere. Naming the operands is the difference between "the lm_head is
    /// wrong" and "the lm_head's B operand is a tensor nobody filled".
    pub fn lm_head_operands(&self, prog: usize) -> Option<(usize, u16, Vec<(usize, String)>)> {
        let insts = self.pf_src.get(prog)?;
        let t_logits = self.t_logits?;
        for (i, d) in insts.iter().enumerate() {
            let is_matmul = is_lm_head_matmul(d.op);
            if is_matmul && d.t[0] as usize == t_logits {
                let ops =
                    d.t.iter()
                        .enumerate()
                        .filter(|(_, &h)| (h as usize) < self.tensor_names.len())
                        .map(|(k, &h)| (k, self.tensor_names[h as usize].clone()))
                        .collect();
                // How many STREAM entries reference it, and in which segments.
                // An instruction the compiler emitted but no stream entry
                // schedules never runs, and its output tensor stays exactly as
                // the loader left it — zero. That is indistinguishable from a
                // broken kernel unless you look here.
                let mut n_ent = 0usize;
                let mut segs: Vec<u16> = Vec::new();
                for e in &self.pf_stream[prog] {
                    if e.inst as usize == i {
                        n_ent += 1;
                        if !segs.contains(&e.seg) {
                            segs.push(e.seg);
                        }
                    }
                }
                segs.sort_unstable();
                *self.lm_detail.borrow_mut() = Some((d.blocks, d.i, n_ent, segs));
                return Some((i, d.op, ops));
            }
        }
        None
    }

    /// `(blocks, i[8])` of the lm_head, valid after [`AmdEngine::lm_head_operands`].
    pub fn lm_head_detail(&self) -> Option<(u16, [u32; 8], usize, Vec<u16>)> {
        self.lm_detail.borrow().clone()
    }

    /// Every tensor the blob declares, in handle order.
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    fn check_packed_prefill_program(&self, prog: usize) -> Result<u32> {
        let program = self.progs.get(prog).ok_or_else(|| {
            RuntimeError::Device(format!("packed-prefill program {prog} is out of range"))
        })?;
        if prog >= self.dec_lo {
            return Err(RuntimeError::Device(format!(
                "program {prog} is a decode rung, not a packed-prefill program"
            )));
        }
        check_packed_prefill_abi(self.packed_prefill_prefill_abi, false, false)?;
        if program.prefill_routes.iter().any(|r| {
            matches!(
                r,
                PrefillSegmentRoute::MlaMaterializePack { .. }
                    | PrefillSegmentRoute::MlaMaterializedPrefill { .. }
            )
        }) {
            return Err(RuntimeError::Device(
                "materialized MLA prefill currently supports one exact initial sequence, not packed spans"
                    .into(),
            ));
        }
        if matches!(self.prog_dispatch(prog), ProgramDispatch::L2Domains(_)) {
            return Err(RuntimeError::Device(
                "packed-prefill family routing requires independent ordered segments; this legacy \
                 packet stores L2 domains in the segment field"
                    .into(),
            ));
        }
        if program.packed_dense {
            if !self.packed_prefill_dense {
                return Err(RuntimeError::Device(
                    "packed dense prefill requires plow_packed_prefill_dense_consumers_1 in every routed object".into(),
                ));
            }
            if let Some(error) = &program.packed_dense_error {
                return Err(RuntimeError::Device(error.clone()));
            }
        }
        if program.packed_needs_mla && !program.packed_dense {
            if !program.packed_mla_compatible {
                return Err(RuntimeError::Device(
                    "packed-prefill MLA does not support NoPE or gathered/per-query selector packets"
                        .into(),
                ));
            }
            if !program.packed_mla_segmented {
                return Err(RuntimeError::Device(
                    "packed-prefill MLA consumer is in a mixed segment; re-emit with \
                     PLOW_EMIT_PACKED_PREFILL=1"
                        .into(),
                ));
            }
            if program.packed_seg_family.contains(&5) && self.k_packed_mla_norm.is_none() {
                return Err(RuntimeError::Device(
                    "packed-prefill MLA norm/cache segment requires interp_packed_mla_norm".into(),
                ));
            }
            if program.packed_seg_family.contains(&6) && self.k_packed_mla_flash.is_none() {
                return Err(RuntimeError::Device(
                    "packed-prefill MLA flash segment requires interp_packed_mla_flash".into(),
                ));
            }
        }
        if program.packed_needs_kda {
            if !program.packed_kda_compatible {
                return Err(RuntimeError::Device(
                    "packed-prefill KDA supports serial Conv3/state or complete BT64/BC16 chunk \
                     operator groups; gathered or double-buffered layouts remain disabled"
                        .into(),
                ));
            }
            if !program.packed_kda_segmented || self.k_packed_kda.is_none() {
                return Err(RuntimeError::Device(
                    "packed-prefill KDA requires pure family segments and \
                     interp_packed_kda; re-emit with PLOW_EMIT_PACKED_PREFILL=1 and build the \
                     optional family objects"
                        .into(),
                ));
            }
        }
        // The recurrent (D-class) audit. It is LAST because the family-specific refusals above
        // give a more useful message when they apply; it is a DEFAULT-DENY because everything
        // above is a series of "if this family, check that", and an operator family none of them
        // recognises used to fall straight through to `Ok`. On AMD that is not a slow path:
        // the interpreter's dispatch `default:` writes nothing and does not trap.
        program
            .packed_recurrent_spans
            .as_ref()
            .map_err(|error| RuntimeError::Device(error.clone()))?;
        Ok(program.t)
    }

    /// The most request spans one packed launch of this program may carry (§5.4's D-class limit),
    /// or `None` when the program cannot take the packed route at all. The scheduler asks BEFORE
    /// it touches cursors, so a legal plan is chosen rather than an illegal one refused.
    pub fn packed_prefill_span_limit(&self, prog: usize) -> Option<u32> {
        self.progs
            .get(prog)?
            .packed_recurrent_spans
            .as_ref()
            .ok()
            .copied()
    }

    /// Resolve an ordinary prefill rung to its packed-only sibling. Legacy
    /// single-topology blobs continue to validate the requested program itself.
    pub fn packed_prefill_prog_for(&self, prog: usize) -> Option<usize> {
        packed_prefill_topology_index(prog, self.dec_lo, |candidate| {
            self.progs
                .get(candidate)
                .map(|p| (p.t, p.packed_prefill_only))
        })
        .filter(|&candidate| self.check_packed_prefill_program(candidate).is_ok())
    }

    /// Whether this exact prefill program has the packet ABI and every operator-family object
    /// needed by the packed route.
    pub fn packed_prefill_prog_capable(&self, prog: usize) -> bool {
        self.packed_prefill_prog_for(prog).is_some()
    }

    pub(crate) fn prefill_rungs(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.progs[..self.dec_lo]
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.packed_prefill_only)
            .map(|(index, p)| (index, p.t))
    }

    /// Validate and upload one ragged packed-prefill descriptor. The binding is program-exact:
    /// every other program continues to receive null metadata in its kernarg.
    pub fn stage_packed_prefill(
        &mut self,
        prog: usize,
        spans: &[PrefillSpan],
        parked: &[u32],
    ) -> Result<()> {
        self.packed_prefill = None;
        let rung = self.check_packed_prefill_program(prog)?;
        // REFUSED, NOT TRUNCATED (plans/unified-token-batch.md §9, "Recurrent"). Executing the
        // first `limit` spans of a plan that asked for more is a silently short answer: the
        // requests whose spans were dropped keep their cursors and their KV frontiers are
        // committed by the caller on the strength of a launch that never covered them.
        let limit = self
            .progs
            .get(prog)
            .and_then(|p| p.packed_recurrent_spans.as_ref().ok().copied())
            .unwrap_or(u32::MAX);
        if spans.len() as u64 > u64::from(limit) {
            return Err(RuntimeError::Device(format!(
                "packed prefill plan has {} spans but program {prog} carries a recurrent operator                  limited to {limit} span(s) per launch (plans/unified-token-batch.md §5.4). The                  plan is refused, not truncated.",
                spans.len()
            )));
        }
        let binding = validate_packed_prefill(prog, rung, self.batch, spans, parked)?;
        if spans.len() > self.prefill_span_capacity || parked.len() > self.prefill_row_capacity {
            return Err(RuntimeError::Device(format!(
                "packed-prefill metadata exceeds staging capacity: spans {}/{}, rows {}/{}",
                spans.len(),
                self.prefill_span_capacity,
                parked.len(),
                self.prefill_row_capacity
            )));
        }

        let span_bytes = std::mem::size_of_val(spans);
        let parked_bytes = std::mem::size_of_val(parked);
        let parked_off = self.prefill_span_capacity * std::mem::size_of::<PrefillSpan>();
        self.h_prefill_meta.as_mut_slice()[..span_bytes].copy_from_slice(as_bytes(spans));
        self.h_prefill_meta.as_mut_slice()[parked_off..parked_off + parked_bytes]
            .copy_from_slice(as_bytes(parked));
        self.be.memcpy_htod_pinned_batch(&[
            (
                self.d_prefill_spans.base,
                &self.h_prefill_meta.as_slice()[..span_bytes],
            ),
            (
                self.d_prefill_parked.base,
                &self.h_prefill_meta.as_slice()[parked_off..parked_off + parked_bytes],
            ),
        ])?;
        self.packed_prefill = Some(binding);
        Ok(())
    }

    /// Remove the current packed-prefill binding. Device buffers remain allocated for reuse.
    pub fn clear_packed_prefill(&mut self) {
        self.packed_prefill = None;
    }

    /// Upload bytes into a named tensor (block I/O, and weight loaders).
    pub fn write_tensor(&mut self, name: &str, src: &[u8]) -> Result<()> {
        let i = self
            .tensor_names
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| RuntimeError::Device(format!("no tensor {name:?}")))?;
        EngineDevice::upload(&*self.be, &self.devp[i], 0, src)
    }

    /// Dump the decode program's `PlowTraceRec[n_stream]` to `path`.
    ///
    /// A no-op unless `PLOW_TRACE_RAW` was set when the engine was built — the
    /// buffer is not allocated otherwise. The records are the LAST launch's, so
    /// call this after a steady-state step, never after the warmup.
    ///
    /// The file is the raw device buffer: `n_stream` 40-byte records, slot
    /// `stream_ofs[cu] + pc`, each carrying `(cu, pc, inst, op, slice,
    /// t_arrive, t_ready, t_end)` in `s_memrealtime` ticks (100 MHz). An entry
    /// no workgroup reached is all-zero. `scripts/k3_trace_report.py` reads it.
    pub fn trace_write(&self, path: &Path) -> Result<()> {
        let Some(m) = &self.d_trace else {
            return Err(RuntimeError::Device(
                "trace buffer was not allocated — set PLOW_TRACE_RAW before loading".into(),
            ));
        };
        let bytes = self.trace_write_bytes.get();
        if bytes == 0 || bytes > self.trace_bytes {
            return Err(RuntimeError::Device(
                "trace buffer has no completed program extent".into(),
            ));
        }
        let mut buf = vec![0u8; bytes];
        EngineDevice::download(&*self.be, m, 0, &mut buf)?;
        std::fs::write(path, &buf)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))
    }

    /// Read a named tensor back.
    pub fn read_tensor(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let i = self
            .tensor_names
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| RuntimeError::Device(format!("no tensor {name:?}")))?;
        EngineDevice::download(&*self.be, &self.devp[i], 0, dst)
    }

    /// Byte size the blob declares for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<u64> {
        self.tensor_names
            .iter()
            .position(|x| x == name)
            .map(|i| self.devp[i].len)
    }

    /// Build the kernarg block for program `p` at segment `seg`.
    fn kernarg(&self, p: usize, seg: u32) -> DevProgram {
        let g = &self.progs[p];
        let (prefill_spans, prefill_parked, n_prefill_spans, n_prefill_rows) =
            packed_prefill_kernarg(
                self.packed_prefill,
                p,
                self.d_prefill_spans.base,
                self.d_prefill_parked.base,
            );
        DevProgram {
            insts: g.d_inst.base,
            stream: g.d_stream.base,
            stream_ofs: g.d_sofs.base,
            stream_len: g.d_slen.base,
            waits: g.d_waits.base,
            succs: g.d_succs.base,
            // BANKED. `bank` is 0 for the whole life of a single-buffered
            // program, so this is the old expression there.
            counters: g.d_ctr.base + g.bank.current() as u64 * g.ctr_span,
            tensors: self.d_tens.base,
            // EVERY program traces, not just decode. The buffer is sized for the widest one
            // (see `d_trace`'s allocation), and each dispatch writes slot `base + ix` of its own
            // stream, so a run that prefills and then decodes leaves DECODE records behind —
            // which is why `amd-bench` dumps the prefill trace before the decode loop starts.
            trace: self.d_trace.as_ref().map_or(0, |m| m.base),
            cur_seg: seg,
            l2_domains: g.l2_domains,
            // Two-level cache maintenance. Handed to the device only when the OBJECT was built
            // for it; a stale object reads the field as ordinary padding, so an old cubin on a
            // new blob is inert rather than wrong.
            hier_base: g.hier_base,
            n_seg: g.seg_class.len() as u32,
            // Static-path segment windows. Set for every program: an unsegmented
            // one has a single window covering the whole stream, so the decode
            // path does exactly what the old full scan did.
            seg_ofs: if self.seg_window { g.d_seg_ofs.base } else { 0 },
            // Set unconditionally: they are 0 without a GQ appendix, and the
            // static kernel never reads them, so one path serves both.
            gq_stream: g.gq.as_ref().map_or(0, |q| q.d_stream.base),
            gq_seg_ofs: g.gq.as_ref().map_or(0, |q| q.d_seg_ofs.base),
            gq_cursor: g.gq.as_ref().map_or(0, |q| {
                q.d_cursor.base + g.bank.current() as u64 * q.cur_span
            }),
            // Single-GPU leaves all four at zero, and `n_gpu == 0` is the
            // interpreter's "not a group" convention (`tp_decode.c:551` fills
            // them only when `n_gpu > 1`). The collective opcodes never appear
            // in an unsharded program, so nothing reads them there.
            xctr: self.tp.map_or(0, |t| t.xctr),
            peer_scratch: self.tp.map_or(0, |t| t.peer_table),
            rank: self.tp.map_or(0, |t| t.rank),
            n_gpu: self.tp.map_or(0, |t| t.n_gpu),
            prefill_spans,
            prefill_parked,
            n_prefill_spans,
            n_prefill_rows,
            // Unified token batch: this engine does not build one yet, and NULL is the
            // documented "every existing path, bit for bit" value.
            token_batch: 0,
        }
    }

    /// Re-arm program `p`'s counters and GQ cursor.
    ///
    /// ONCE per dispatch group, NEVER per segment. A segment's producers ran in
    /// an earlier launch, so zeroing between segments unsatisfies them and the
    /// next segment waits on a count that will never come again.
    fn rearm(&self, p: usize) -> Result<()> {
        self.rearm_bank(p, self.progs[p].bank.current())
    }

    /// Zero ONE bank of program `p`'s counters and GQ cursor.
    ///
    /// Split out of [`AmdEngine::rearm`] for the double-buffered [`AmdEngine::run`],
    /// which clears the bank the PREVIOUS dispatch dirtied rather than the one
    /// the next dispatch will read.
    fn rearm_bank(&self, p: usize, bank: u32) -> Result<()> {
        let g = &self.progs[p];
        let n = g.n_counter as usize * CTR_STRIDE_U32 * 4;
        let mut zeroing: Vec<(u64, &[u8])> = Vec::with_capacity(2);
        if n > 0 {
            zeroing.push((
                g.d_ctr.base + bank as u64 * g.ctr_span,
                &self.h_zero.as_slice()[..n],
            ));
        }
        if let Some(q) = &g.gq {
            let n = q.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4;
            zeroing.push((
                q.d_cursor.base + bank as u64 * q.cur_span,
                &self.h_zero.as_slice()[..n],
            ));
        }
        self.be.memcpy_htod_pinned_batch(&zeroing)?;
        Ok(())
    }

    /// Re-arm program `p`'s counters and cursor. Used by prefill and by TP's
    /// synchronous single-bank fallback.
    pub fn rearm_prog(&self, p: usize) -> Result<()> {
        self.rearm(p)
    }

    /// Whether TP may select this program's inactive local-counter bank.
    ///
    /// The preflight is separate so a group checks EVERY rank before changing
    /// any rank's bank selection.
    pub fn tp_counter_bank_ready(&self, p: usize) -> bool {
        !ctr_dbuf() || self.progs[p].bank.inactive_ready()
    }

    /// Whether TP should use the post-launch inactive-bank re-arm path.
    pub fn tp_counter_double_buffered(&self) -> bool {
        ctr_dbuf()
    }

    /// Select a clean counter bank for the next TP dispatch.
    ///
    /// With double-buffering disabled this performs the original synchronous
    /// current-bank re-arm. The caller must invoke it on every rank only after
    /// all rank preparation has succeeded.
    pub fn tp_begin_counter_bank(&self, p: usize) -> Result<()> {
        match self.progs[p].bank.begin_tp(ctr_dbuf()) {
            Ok(true) => Ok(()),
            Ok(false) => self.rearm(p),
            Err(()) => Err(RuntimeError::Device(format!(
                "program {p} inactive counter bank is stale: its previous post-launch re-arm \
                 did not complete; refusing to dispatch with uncleared local counters"
            ))),
        }
    }

    /// Re-arm the inactive TP counter/cursor bank after every rank has launched.
    ///
    /// The blocking SDMA copy overlaps the resident megakernels. The bank is
    /// marked reusable only after both its counter and optional GQ-cursor clears
    /// complete successfully.
    pub fn tp_rearm_inactive_counter_bank(&self, p: usize) -> Result<()> {
        if !ctr_dbuf() {
            return Ok(());
        }
        let bank = self.progs[p].bank.inactive();
        self.rearm_bank(p, bank)?;
        self.progs[p].bank.mark_inactive_ready();
        Ok(())
    }

    /// Number of segments in program `p`, and whether segment `seg` is class 4.
    pub fn segment_class(&self, p: usize, seg: usize) -> u8 {
        self.progs[p].seg_class[seg]
    }

    /// Enqueue ONE segment of program `p`. No re-arm, no drain.
    ///
    /// The building block of both the single-GPU segmented run and the TP
    /// per-segment rendezvous. Each launch memcpy's its own kernarg slot, so
    /// mutating `cur_seg` between launches is safe — every packet has already
    /// captured its own copy. An L2-placed program launches each ordered host
    /// segment once; all per-XCD queues for that segment drain concurrently.
    pub fn enqueue_segment(&mut self, p: usize, seg: usize) -> Result<()> {
        check_packed_prefill_dispatch(self.packed_prefill, p)?;
        self.trace_write_bytes
            .set(self.progs[p].trace_records * TRACE_REC_BYTES);
        if let Some(PrefillSegmentRoute::XReduceAttnRes {
            device_args, grid, ..
        }) = self.progs[p].prefill_routes.get(seg).copied()
        {
            let kernel = self.k_xr_attnres.ok_or_else(|| {
                RuntimeError::Device(
                    "XReduceTwoShot+AttnRes segment has no exact capability object".into(),
                )
            })?;
            if device_args == 0 {
                return Err(RuntimeError::Device(
                    "XReduceTwoShot+AttnRes route has no device argument block".into(),
                ));
            }
            EngineDevice::launch_cooperative(
                &*self.be,
                kernel,
                grid,
                WG_THREADS_8,
                0,
                as_bytes(std::slice::from_ref(&device_args)),
                None,
            )?;
            self.seg_launches += 1;
            return Ok(());
        }
        if matches!(
            self.progs[p].prefill_routes.get(seg),
            Some(PrefillSegmentRoute::XReduceWaveRs | PrefillSegmentRoute::GraphPhaseXReduceWaveRs)
        ) {
            let kernel = self.k_xreduce_wave_rs.ok_or_else(|| {
                RuntimeError::Device(
                    "marked XReduceTwoShot segment has no packet-paired wave-RS object".into(),
                )
            })?;
            let arg = self.kernarg(p, seg as u32);
            EngineDevice::launch_cooperative(
                &*self.be,
                kernel,
                self.n_cu,
                WG_THREADS_8,
                0,
                kernarg_bytes(&arg),
                None,
            )?;
            self.seg_launches += 1;
            return Ok(());
        }
        if !prefill_segment_specialization_allowed(self.prog_dispatch(p)) {
            let arg = self.kernarg(p, seg as u32);
            EngineDevice::launch_cooperative(
                &*self.be,
                self.k_prefill,
                self.n_cu,
                WG_THREADS_8,
                0,
                kernarg_bytes(&arg),
                None,
            )?;
            self.seg_launches += 1;
            return Ok(());
        }
        let active = self.packed_prefill.is_some_and(|b| b.prog == p);
        if !active {
            if let Some(PrefillSegmentRoute::MlaFold(route)) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.mla_fold.as_ref().ok_or_else(|| {
                    RuntimeError::Device("native MLA fold has no loaded kernels".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table)?;
                self.seg_launches += 3;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::GemmLt(route)) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.gemm_lt.as_ref().ok_or_else(|| {
                    RuntimeError::Device("hipBLASLt projection route has no loaded kernel".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table)?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::IndexTp(route)) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.index_tp.as_ref().ok_or_else(|| {
                    RuntimeError::Device("TP indexer route has no loaded kernels".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table, self.tp.unwrap())?;
                self.seg_launches += 4;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::MoeAiter(route)) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.moe_aiter.as_ref().ok_or_else(|| {
                    RuntimeError::Device("AITER MoE route has no loaded kernels".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table)?;
                self.seg_launches += route.launches();
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::SparseMla(route)) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                if route.active {
                    let sparse = self.sparse_mla.as_ref().ok_or_else(|| {
                        RuntimeError::Device("sparse MLA route has no loaded kernels".into())
                    })?;
                    sparse.enqueue(&self.be, route, &self.tens_table)?;
                    self.seg_launches += 3;
                    return Ok(());
                }
            }
            if let Some(PrefillSegmentRoute::MlaMaterializePack {
                args,
                grid,
                k_rope_ten,
                ..
            }) = self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.k_mla_materialize_pack.ok_or_else(|| {
                    RuntimeError::Device(
                        "materialized MLA pack segment has no validated object".into(),
                    )
                })?;
                let mut args = args;
                if self.kv_slot != 0 {
                    if let Some(&(_, stride)) = self
                        .kv_slot_stride
                        .iter()
                        .find(|(i, _)| *i == k_rope_ten as usize)
                    {
                        args.k_rope += stride * self.kv_slot as u64;
                    }
                }
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::MlaMaterializedPrefill { args, grid }) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.k_mla_materialized_prefill.ok_or_else(|| {
                    RuntimeError::Device(
                        "materialized MLA prefill segment has no validated object".into(),
                    )
                })?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    512,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::KdaChunkIntraCached { args, grid })) = (
                self.k_kda_chunk_intra_cached,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    WG_THREADS_8,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::KdaChunkIntraWaveItems { args, grid }) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = self.k_kda_chunk_intra_wave_items.ok_or_else(|| {
                    RuntimeError::Device(
                        "marked KDA-intra wave-item segment has no validated object".into(),
                    )
                })?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    WG_THREADS_8,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            // The regstate object only covers full BT64 rungs (`t >= 512`, mirrored by the
            // kernel's early return); a shorter ragged tail chunk stays on the interpreter.
            if let Some(PrefillSegmentRoute::KdaChunkCarryRegstate { args, grid, tens }) = self
                .progs[p]
                .prefill_routes
                .get(seg)
                .copied()
                .filter(|r| {
                    matches!(r, PrefillSegmentRoute::KdaChunkCarryRegstate { args, .. } if args.t >= 512)
                })
            {
                let kernel = self.k_kda_chunk_carry_regstate.ok_or_else(|| {
                    RuntimeError::Device(
                        "marked KDA carry regstate segment has no validated object".into(),
                    )
                })?;
                let mut args = args;
                if self.kv_slot != 0 {
                    let fields: [&mut u64; 8] = [
                        &mut args.out,
                        &mut args.state,
                        &mut args.q,
                        &mut args.k,
                        &mut args.w,
                        &mut args.u,
                        &mut args.aqk,
                        &mut args.g,
                    ];
                    for (id, field) in tens.iter().zip(fields) {
                        if let Some(&(_, stride)) =
                            self.kv_slot_stride.iter().find(|(i, _)| *i == *id as usize)
                        {
                            *field += stride * self.kv_slot as u64;
                        }
                    }
                }
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::KdaChunkCarryKeyfeed { args, grid, tens }) = self
                .progs[p]
                .prefill_routes
                .get(seg)
                .copied()
                .filter(|r| {
                    matches!(r, PrefillSegmentRoute::KdaChunkCarryKeyfeed { args, .. } if args.t >= 512)
                })
            {
                let kernel = self.k_kda_chunk_carry_keyfeed.ok_or_else(|| {
                    RuntimeError::Device(
                        "marked KDA carry keyfeed segment has no validated object".into(),
                    )
                })?;
                let mut args = args;
                if self.kv_slot != 0 {
                    let fields: [&mut u64; 8] = [
                        &mut args.out,
                        &mut args.state,
                        &mut args.q,
                        &mut args.k,
                        &mut args.w,
                        &mut args.u,
                        &mut args.aqk,
                        &mut args.g,
                    ];
                    for (id, field) in tens.iter().zip(fields) {
                        if let Some(&(_, stride)) =
                            self.kv_slot_stride.iter().find(|(i, _)| *i == *id as usize)
                        {
                            *field += stride * self.kv_slot as u64;
                        }
                    }
                }
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::KdaChunkWuLean { args, grid, tens }) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let kernel = if args.key_hi != 0 {
                    self.k_kda_chunk_wu_lean_keys
                } else {
                    self.k_kda_chunk_wu_lean
                }
                .ok_or_else(|| {
                    RuntimeError::Device(
                        "marked KDA Wu lean segment has no validated object".into(),
                    )
                })?;
                let mut args = args;
                if self.kv_slot != 0 {
                    let fields: [&mut u64; 8] = [
                        &mut args.w,
                        &mut args.u,
                        &mut args.q,
                        &mut args.ainv,
                        &mut args.k,
                        &mut args.v,
                        &mut args.g,
                        &mut args.beta,
                    ];
                    for (id, field) in tens.iter().zip(fields) {
                        if let Some(&(_, stride)) =
                            self.kv_slot_stride.iter().find(|(i, _)| *i == *id as usize)
                        {
                            *field += stride * self.kv_slot as u64;
                        }
                    }
                }
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::KdaChunkKeyFactorWu { args, grid })) = (
                self.k_kda_key_factor_wu,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (
                Some(kernel),
                Some(PrefillSegmentRoute::KdaChunkKeyFactorCarry { args, grid }),
            ) = (
                self.k_kda_key_factor_carry,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeStage1Mxfp4(route))) = (
                self.k_moe_stage1_mxfp4,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    WG_THREADS_8,
                    119_808,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeEpAlign(route))) = (
                self.k_moe_ep_align,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                for (phase, grid) in [(1, 64), (2, 1), (3, 64), (4, 64)] {
                    let mut args = route.args;
                    args.phase = phase;
                    EngineDevice::launch_kernel(
                        &*self.be,
                        kernel,
                        grid,
                        256,
                        0,
                        as_bytes(std::slice::from_ref(&args)),
                        None,
                    )?;
                }
                self.seg_launches += 4;
                return Ok(());
            }
            if let (Some(quant), Some(kernel), Some(PrefillSegmentRoute::MoeStage1A4Reuse(route))) = (
                self.k_moe_stage1_a4_quant,
                self.k_moe_stage1_a4_reuse,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    quant,
                    route.quant_grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&route.quant_args)),
                    None,
                )?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    256,
                    32_768,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 2;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeStage2Mxfp4(route))) = (
                self.k_moe_stage2_mxfp4,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    256,
                    4_352,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeEpStage2(route))) = (
                self.k_moe_ep_stage2,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    256,
                    4_352,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeCombine(route))) = (
                self.k_moe_combine,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let Some(PrefillSegmentRoute::AttnResF32Mix { args, grid }) =
                self.progs[p].prefill_routes.get(seg).copied()
            {
                let (kernel, threads) = self.k_attn_res_f32mix.ok_or_else(|| {
                    RuntimeError::Device("f32-mix AttnRes segment has no validated object".into())
                })?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    grid,
                    threads,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
            if let (Some(kernel), Some(PrefillSegmentRoute::MoeEpCombine(route))) = (
                self.k_moe_ep_combine,
                self.progs[p].prefill_routes.get(seg).copied(),
            ) {
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    route.grid,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&route.args)),
                    None,
                )?;
                self.seg_launches += 1;
                return Ok(());
            }
        }
        let (k, threads, _) = self.prefill_interpreter_kernel(p, seg)?;
        let arg = self.kernarg(p, seg as u32);
        EngineDevice::launch_cooperative(
            &*self.be,
            k,
            self.n_cu,
            threads,
            0,
            kernarg_bytes(&arg),
            None,
        )?;
        self.seg_launches += 1;
        Ok(())
    }

    /// Number of ordered segment steps in a decode program. A grouped-MoE step owns two
    /// back-to-back raw launches; other raw boundaries and interpreter segments own one.
    pub(crate) fn decode_launches(&self, p: usize) -> usize {
        self.prog_dispatch(p).launches()
    }

    pub(crate) fn graph_phase_replay(&self, p: usize) -> bool {
        self.progs[p]
            .prefill_routes
            .iter()
            .any(|r| matches!(r, PrefillSegmentRoute::GraphPhaseXReduceWaveRs))
    }

    pub(crate) fn begin_graph_phase_replay(&self, p: usize) -> Result<()> {
        if !self.graph_phase_replay(p) {
            return Err(RuntimeError::Device(format!(
                "program {p} has no graph-derived phase-object route"
            )));
        }
        let packets = (0..self.prog_dispatch(p).launches())
            .map(|seg| self.prefill_segment_launches(p, seg))
            .sum();
        self.be.begin_dispatch_chain(packets)
    }

    /// Exact AQL packets `enqueue_segment` emits for one prefill segment. A phase
    /// chain reserves this many before ringing, so the count must track the
    /// multi-launch route branches in `enqueue_segment`; the commit check refuses
    /// a chain whose emission disagrees, so a drift fails closed rather than
    /// overrunning the queue.
    fn prefill_segment_launches(&self, p: usize, seg: usize) -> usize {
        let active = self.packed_prefill.is_some_and(|b| b.prog == p);
        if active || !prefill_segment_specialization_allowed(self.prog_dispatch(p)) {
            return 1;
        }
        match self.progs[p].prefill_routes.get(seg) {
            Some(PrefillSegmentRoute::SparseMla(route)) if route.active => 3,
            Some(PrefillSegmentRoute::GemmLt(_)) => 1,
            Some(PrefillSegmentRoute::MlaFold(_)) => 3,
            Some(PrefillSegmentRoute::MoeAiter(route)) => route.launches() as usize,
            Some(PrefillSegmentRoute::IndexTp(_)) => 4,
            Some(PrefillSegmentRoute::MoeEpAlign(_)) if self.k_moe_ep_align.is_some() => 4,
            Some(PrefillSegmentRoute::MoeStage1A4Reuse(_))
                if self.k_moe_stage1_a4_quant.is_some() && self.k_moe_stage1_a4_reuse.is_some() =>
            {
                2
            }
            _ => 1,
        }
    }

    pub(crate) fn commit_graph_phase_replay(&self) -> Result<()> {
        self.be.commit_dispatch_chain()
    }

    /// Enqueue one ordered decode segment without rearming or draining.
    pub(crate) fn enqueue_decode_segment(
        &mut self,
        p: usize,
        seg: usize,
        interpreter: HsaKernel,
    ) -> Result<()> {
        self.trace_write_bytes
            .set(self.progs[p].trace_records * TRACE_REC_BYTES);
        let route = *self.progs[p].decode_routes.get(seg).ok_or_else(|| {
            RuntimeError::Device(format!(
                "decode segment {seg} is outside {} segments for program {p}",
                self.progs[p].decode_routes.len()
            ))
        })?;
        match route {
            DecodeSegmentRoute::Interpreter => {
                let arg = self.kernarg(p, seg as u32);
                EngineDevice::launch_cooperative(
                    &*self.be,
                    interpreter,
                    self.n_cu,
                    WG_THREADS_8,
                    0,
                    kernarg_bytes(&arg),
                    None,
                )?;
            }
            DecodeSegmentRoute::MlaAttention => {
                let kernel = self.k_decode_mla.ok_or_else(|| {
                    RuntimeError::Device(
                        "decode MLA segment has no validated packet-paired object".into(),
                    )
                })?;
                let arg = self.kernarg(p, seg as u32);
                EngineDevice::launch_cooperative(
                    &*self.be,
                    kernel,
                    self.n_cu,
                    WG_THREADS_8,
                    0,
                    kernarg_bytes(&arg),
                    None,
                )?;
            }
            DecodeSegmentRoute::MoeAiter(route) => {
                let kernel = self.moe_aiter.as_ref().ok_or_else(|| {
                    RuntimeError::Device("AITER decode route has no loaded kernels".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table)?;
                self.seg_launches += route.launches() - 1;
            }
            DecodeSegmentRoute::GemmLt(route) => {
                let kernel = self.gemm_lt.as_ref().ok_or_else(|| {
                    RuntimeError::Device("native decode GEMM has no loaded kernels".into())
                })?;
                kernel.enqueue(&self.be, route, &self.tens_table)?;
            }
            DecodeSegmentRoute::KdaDecodeFused(args) => {
                let kernel = self.k_kda_decode_fused.ok_or_else(|| {
                    RuntimeError::Device(
                        "KdaDecodeFused segment has no validated standalone object".into(),
                    )
                })?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    kernel,
                    args.rows * args.heads,
                    256,
                    0,
                    as_bytes(std::slice::from_ref(&args)),
                    None,
                )?;
            }
            DecodeSegmentRoute::GroupedMoeMxfp4 { glu, down, grid } => {
                let glu_kernel = self.k_grouped_moe_glu.ok_or_else(|| {
                    RuntimeError::Device(
                        "grouped MoE segment has no validated standalone GLU object".into(),
                    )
                })?;
                let down_kernel = self.k_grouped_moe_down.ok_or_else(|| {
                    RuntimeError::Device(
                        "grouped MoE segment has no validated standalone DOWN object".into(),
                    )
                })?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    glu_kernel,
                    grid,
                    WG_THREADS_8,
                    0,
                    as_bytes(std::slice::from_ref(&glu)),
                    None,
                )?;
                EngineDevice::launch_kernel(
                    &*self.be,
                    down_kernel,
                    grid,
                    WG_THREADS_8,
                    0,
                    as_bytes(std::slice::from_ref(&down)),
                    None,
                )?;
                self.seg_launches += 1;
            }
        }
        self.seg_launches += 1;
        Ok(())
    }

    /// Enqueue the single-launch (decode) dispatch of program `p`. No drain.
    ///
    /// Split out of [`AmdEngine::run`] so a TP driver can launch EVERY rank
    /// before waiting on any: the ranks rendezvous on the device through their
    /// cross-GPU counters, inside their own dispatches, so a host wait between
    /// two ranks' launches would make rank 0 spin on a partial rank 1 has not
    /// been dispatched to produce — reintroducing exactly the launched-collective
    /// latency the inline design exists to avoid.
    pub fn enqueue(&mut self, p: usize, k: HsaKernel) -> Result<()> {
        check_packed_prefill_dispatch(self.packed_prefill, p)?;
        self.trace_write_bytes
            .set(self.progs[p].trace_records * TRACE_REC_BYTES);
        let arg = self.kernarg(p, 0);
        EngineDevice::launch_cooperative(
            &*self.be,
            k,
            self.n_cu,
            WG_THREADS_8,
            0,
            kernarg_bytes(&arg),
            None,
        )?;
        self.seg_launches += 1;
        Ok(())
    }

    pub(crate) fn deferred_token_capture_available(&self) -> bool {
        self.k_token_capture.is_some() && self.d_token_ring.is_some()
    }

    /// Queue a copy of the sampled ids after the current decode dispatch.
    /// No drain: the caller owns the existing per-token drain/audit boundary.
    pub(crate) fn enqueue_token_capture(
        &self,
        step: usize,
        quantum: usize,
        batch: usize,
    ) -> Result<()> {
        if step >= quantum || quantum > DEFERRED_TOKEN_MAX_STEPS || batch > self.batch || batch == 0
        {
            return Err(RuntimeError::Device(format!(
                "invalid deferred token capture step={step} quantum={quantum} batch={batch}"
            )));
        }
        let kernel = self.k_token_capture.ok_or_else(|| {
            RuntimeError::Rejected("decode object has no plow_token_capture helper".into())
        })?;
        let ring = self
            .d_token_ring
            .as_ref()
            .ok_or_else(|| RuntimeError::Device("deferred token ring is absent".into()))?;
        let args = TokenCaptureArgs {
            ids: self.devp[self.need(self.t_ids, "in.ids")?].base,
            ring: ring.base,
            step: step as u32,
            quantum: quantum as u32,
            batch: batch as u32,
        };
        EngineDevice::launch_kernel(
            &*self.be,
            kernel,
            (batch as u32).div_ceil(256),
            256,
            0,
            as_bytes(std::slice::from_ref(&args)),
            None,
        )
    }

    /// Read one completed row-major `[batch][quantum]` capture into `out`.
    pub(crate) fn read_token_capture(
        &mut self,
        batch: usize,
        quantum: usize,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if quantum == 0 || quantum > DEFERRED_TOKEN_MAX_STEPS || batch > self.batch {
            return Err(RuntimeError::Device(format!(
                "invalid deferred token read quantum={quantum} batch={batch}"
            )));
        }
        let bytes = batch * quantum * 4;
        let src = self
            .d_token_ring
            .as_ref()
            .ok_or_else(|| RuntimeError::Device("deferred token ring is absent".into()))?
            .base;
        self.be
            .memcpy_dtoh_pinned(&mut self.h_scalar.as_mut_slice()[..bytes], src)?;
        out.clear();
        out.extend(
            self.h_scalar.as_slice()[..bytes]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("4"))),
        );
        Ok(())
    }

    /// Wait for everything this rank has enqueued.
    pub fn drain(&self) -> Result<()> {
        EngineDevice::synchronize(&*self.be)
    }

    /// Enqueue the post-drain exact xctr scan used by compact TP audit.
    pub fn enqueue_xaudit(
        &self,
        p: usize,
        xctr: u64,
        n_xctr: u32,
        n_gpu: u32,
        status: u64,
    ) -> Result<()> {
        let k = self
            .k_xaudit
            .ok_or_else(|| RuntimeError::Device("compact TP audit kernel was not loaded".into()))?;
        let prog = &self.progs[p];
        let arg = XAuditArgs {
            insts: prog.d_inst.base,
            n_inst: prog.n_inst,
            _pad: 0,
            xctr,
            n_xctr,
            n_gpu,
            status,
        };
        EngineDevice::launch_kernel(
            &*self.be,
            k,
            1,
            256,
            0,
            as_bytes(std::slice::from_ref(&arg)),
            None,
        )
    }

    /// Run every segment of program `p`, then drain ONCE.
    ///
    /// The single drain is correct only because each dispatch carries the AQL
    /// barrier bit, which chains segment k+1 behind segment k on the packet
    /// processor with no host round-trip.
    ///
    /// # This shape is WRONG under TP, and that is not a performance opinion
    ///
    /// `tp_decode.c` records the failure: enqueueing all of one rank's segments
    /// and only then moving to the next rank let the ranks **desync — a lagging
    /// rank made peers time out and bail, giving a WRONG, 100x-slow reduction at
    /// TP>=4.** A class-8 segment holds both of a layer's all-reduces, and the
    /// inline system-scope gate only rendezvouses cheaply if every rank is
    /// inside that segment at the same time. So a TP prefill goes PER-SEGMENT,
    /// ALL-RANKS, with a host barrier between segments — see
    /// [`crate::exec::amd_tp::AmdTpGroup::prefill`]. This method stays as it is
    /// because on ONE GPU there is no peer to desync from.
    ///
    pub fn run_segmented(&mut self, p: usize) -> Result<()> {
        self.rearm(p)?;
        let n_seg = self.prog_dispatch(p).launches();
        let t0 = std::time::Instant::now();
        let replay = self.graph_phase_replay(p);
        if replay {
            self.begin_graph_phase_replay(p)?;
        }
        for seg in 0..n_seg {
            self.enqueue_segment(p, seg)?;
        }
        if replay {
            self.commit_graph_phase_replay()?;
        }
        let t1 = std::time::Instant::now();
        if let Err(e) = self.drain() {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                segments = n_seg,
                grid = self.n_cu,
                "segmented run failed at drain"
            );
            return Err(e);
        }
        let t2 = std::time::Instant::now();
        self.seg_enq_us += (t1 - t0).as_secs_f64() * 1e6;
        self.seg_drain_us += (t2 - t1).as_secs_f64() * 1e6;
        crate::obs::ttft::PF_SEGMENTS.tally(n_seg as u64);
        crate::obs::ttft::PF_ENQUEUE.add((t1 - t0).as_nanos() as u64);
        crate::obs::ttft::PF_DRAIN.add((t2 - t1).as_nanos() as u64);
        Ok(())
    }

    /// Single-launch run — the decode path, which is not segmented.
    /// # The re-arm is BEHIND the enqueue, not in front of it
    ///
    /// With [`ctr_dbuf`] on, the bank this dispatch reads was already zeroed —
    /// by the PREVIOUS dispatch, while the GPU was busy. So the order is
    /// enqueue, then clear the stale bank, then drain: the clear's two blocking
    /// SDMA round trips (56 µs measured, §`ctr_dbuf`) overlap 11.6 ms of
    /// megakernel instead of delaying its start. Flipping `bank` after the
    /// enqueue is safe because `enqueue` memcpy'd its own kernarg slot, so the
    /// launch already captured the address it will use.
    ///
    /// Correctness rests on one thing: the bank being cleared is not the one
    /// the in-flight dispatch is reading. That holds because `run` drains
    /// before it returns, so dispatch N-1 (which dirtied `stale`) has retired
    /// before dispatch N is even staged.
    pub fn run(&mut self, p: usize, k: HsaKernel) -> Result<()> {
        self.run_with_capture(p, k, None)
    }

    fn run_with_capture(
        &mut self,
        p: usize,
        k: HsaKernel,
        capture: Option<(usize, usize)>,
    ) -> Result<()> {
        use crate::obs::dstep;
        if requires_segmented_decode(&self.progs[p].decode_routes) {
            if !ctr_dbuf() {
                dstep::timed(&dstep::REARM, || self.rearm(p))?;
            }
            let t0 = std::time::Instant::now();
            let n = self.decode_launches(p);
            dstep::timed(&dstep::ENQUEUE, || {
                for seg in 0..n {
                    self.enqueue_decode_segment(p, seg, k)?;
                }
                Ok(())
            })?;
            if ctr_dbuf() {
                let cur = self.progs[p].bank.current();
                dstep::timed(&dstep::REARM, || self.rearm_bank(p, 1 - cur))?;
                self.progs[p].bank.select_rearmed_inactive();
            }
            if let Some((step, quantum)) = capture {
                self.enqueue_token_capture(step, quantum, self.progs[p].t as usize)?;
            }
            dstep::timed(&dstep::DRAIN, || self.drain())?;
            self.seg_drain_us += t0.elapsed().as_secs_f64() * 1e6;
            return Ok(());
        }
        if !ctr_dbuf() {
            dstep::timed(&dstep::REARM, || self.rearm(p))?;
        }
        let t0 = std::time::Instant::now();
        if let Err(e) = dstep::timed(&dstep::ENQUEUE, || self.enqueue(p, k)) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                grid = self.n_cu,
                block = WG_THREADS_8,
                "program dispatch failed at enqueue"
            );
            return Err(e);
        }
        if ctr_dbuf() {
            let cur = self.progs[p].bank.current();
            dstep::timed(&dstep::REARM, || self.rearm_bank(p, 1 - cur))?;
            self.progs[p].bank.select_rearmed_inactive();
        }
        if let Some((step, quantum)) = capture {
            self.enqueue_token_capture(step, quantum, self.progs[p].t as usize)?;
        }
        // The drain is where an async kernel trap surfaces — capture the
        // dispatch shape at the site before propagating.
        if let Err(e) = dstep::timed(&dstep::DRAIN, || self.drain()) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                grid = self.n_cu,
                block = WG_THREADS_8,
                "program dispatch failed at drain"
            );
            return Err(e);
        }
        self.seg_drain_us += t0.elapsed().as_secs_f64() * 1e6;
        Ok(())
    }

    /// Patch the KV-append row into every `kvrow` site and push ONE contiguous
    /// slice of the instruction stream.
    ///
    /// The sites are scattered in k/v pairs across all layers (Gemma-31B:
    /// `[4,664]` of 676), so one contiguous slice beats a per-site scatter:
    /// fewer bytes than the whole stream and, more importantly, ONE h2d
    /// submission instead of `n_kvrow` of them. Submission overhead, not bytes,
    /// is what costs here. Patched in the PINNED copy so the slice is
    /// contiguous in pinned memory and needs no per-call page pin.
    fn patch_kvrow(&mut self, dp: usize, pos: u32) -> Result<()> {
        let Some((lo, hi)) = self.kvrow_span else {
            return Ok(());
        };
        let sz = std::mem::size_of::<DevInst64>();
        {
            let slab = self.h_inst.as_mut_slice();
            let n = slab.len() / sz;
            // SAFETY: the slab was allocated as `n_inst * size_of::<DevInst64>()`
            // and seeded from a `&[DevInst64]`, so it is exactly `n` live,
            // aligned, initialised records. `DevInst64` is `#[repr(C)]` POD.
            let insts: &mut [DevInst64] =
                unsafe { std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut DevInst64, n) };
            for (sites, field) in [(&self.kvrow, 3usize), (&self.kvrow_i2, 2)] {
                for &idx in sites {
                    let i = idx as usize;
                    if i >= n {
                        return Err(RuntimeError::Device(format!(
                            "kvrow site {i} past the decode program's {n} instructions"
                        )));
                    }
                    insts[i].i[field] = pos;
                }
            }
        }
        let src = &self.h_inst.as_slice()[lo * sz..(hi + 1) * sz];
        // NOT `upload`: that pins its source, and pinning an already
        // device-visible pinned slab is invalid (HSA 4096) as well as
        // syscall-class. This is the whole reason `h_inst` is pinned.
        self.be
            .memcpy_htod_pinned(self.progs[dp].d_inst.base + (lo * sz) as u64, src)
    }

    /// Re-point the MLA decode's KV-split count at the LIVE `kv_len`
    /// (`PLOW_MLA_NS_LIVE`), and push the sites if it moved.
    ///
    /// The emitter sizes `nsplit` for `max_ctx`, which over-splits every shorter
    /// context it serves; the kernel reads it as a runtime argument, so the fix
    /// is a field, not a rebuild. Both the flash and its merge carry the count in
    /// `i[4]` and are patched together — the partials are strided by it.
    ///
    /// No dispatch or counter change: `blocks` stays at the emitted value and the
    /// flash grid-strides `n_work = n_batch*n_tok*n_grp*nsplit` from `slice` by
    /// `nblk`, so the workgroups a smaller split count no longer needs simply find
    /// no work item and retire. Buffers keep their `baked` sizing and the live
    /// count only ever uses a prefix of them.
    ///
    /// EVERY decode rung is written, not just the dispatched one: which rung the
    /// mux picks is a per-step decision this function does not see, and a rung left
    /// on its baked count would be a silent no-op. See [`MlaNsplitProg`].
    fn patch_mla_nsplit(&mut self, kvlen: u32) -> Result<()> {
        let Some(ns) = &mut self.mla_nsplit else {
            return Ok(());
        };
        let (baked, from) = (ns.baked, ns.cur);
        let want = mla_live_nsplit(baked, kvlen);
        if want == from {
            return Ok(());
        }
        for r in &mut ns.progs {
            for &i in &r.sites {
                r.image[i].i[4] = want;
            }
        }
        // A policy change the operator asked for should be VISIBLE, and this fires
        // at most a handful of times per generation (the live count is a step
        // function of `kv_len`), so it is not a hot-path log.
        tracing::info!(
            baked, from, to = want, kvlen,
            "MLA split count re-pointed at the live kv_len"
        );
        let sz = std::mem::size_of::<DevInst64>();
        // MIRROR THE EDIT INTO `h_inst` for the widest decode program.
        //
        // `image` above and `h_inst` are DIFFERENT host buffers for the same device
        // instructions, and `decode_prepare` calls `patch_kvrow` immediately after this.
        // That function uploads `insts[lo ..= hi]` of the widest decode program FROM
        // `h_inst`, so every site the two ranges share is written back at its BAKED split
        // count microseconds after this one uploaded the live count — the rung actually
        // dispatched then runs a policy this function has already logged it out of, which
        // is worse than not patching at all because the log says otherwise.
        let dp = self.decode;
        {
            let slab = self.h_inst.as_mut_slice();
            let n = slab.len() / sz;
            // SAFETY: as in `patch_kvrow` — the slab was allocated as
            // `n_dec_inst * size_of::<DevInst64>()` and seeded from a `&[DevInst64]`, so it
            // is exactly `n` live, aligned, initialised `#[repr(C)]` POD records.
            let insts: &mut [DevInst64] =
                unsafe { std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut DevInst64, n) };
            for r in self
                .mla_nsplit
                .as_ref()
                .expect("checked")
                .progs
                .iter()
                .filter(|r| r.prog == dp)
            {
                for &site in &r.sites {
                    let i = r.lo + site;
                    if i < n {
                        insts[i].i[4] = want;
                    }
                }
            }
        }
        for r in &self.mla_nsplit.as_ref().expect("checked").progs {
            self.be.upload(
                &self.progs[r.prog].d_inst,
                (r.lo * sz) as u64,
                as_bytes(&r.image),
            )?;
        }
        // AFTER the uploads: `cur` is what the DEVICE holds, so a failed push must
        // leave it stale rather than claim a value that never landed.
        self.mla_nsplit.as_mut().expect("checked").cur = want;
        Ok(())
    }

    /// One decode step at absolute position `pos`, with `kvlen` valid KV rows
    /// after it. Returns the token id the DEVICE sampled.
    ///
    /// Per-step host work is deliberately tiny: patch the KV-append row into the
    /// instructions, push one contiguous slice of them, push two 4-byte scalars,
    /// launch, wait, read 4 bytes back.
    ///
    /// `in.ids` is NOT uploaded. The device's argmax wrote the sampled token
    /// there at the end of the previous launch, which is exactly where this
    /// step's embed reads it; writing a host copy over it would feed the model
    /// last step's token twice.
    pub fn decode_step(&mut self, pos: u32, kvlen: u32) -> Result<u32> {
        use crate::obs::dstep;
        dstep::timed(&dstep::PREPARE, || self.decode_prepare(pos, kvlen))?;
        self.run(self.decode, self.decode_kernel_for(self.decode))?;
        let id = dstep::timed(&dstep::READ, || self.read_sampled())?;
        if let Some(v) = &self.vmm {
            v.advise(self.kv_slot, pos + 1);
        }
        Ok(id)
    }

    /// Everything a decode step does BEFORE the dispatch: patch the KV-append
    /// row and push the two scalars. No launch, no wait.
    ///
    /// Separate because a TP group must prepare and re-arm every rank before
    /// dispatching any of them — see [`AmdEngine::enqueue`].
    pub fn decode_prepare(&mut self, pos: u32, kvlen: u32) -> Result<()> {
        if pos as usize >= self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "position {pos} past max_ctx {}",
                self.max_ctx
            )));
        }
        self.vmm_ensure(self.kv_slot, pos + 1)?;
        self.sync_kda_conv_alt(self.kv_slot)?;
        let dp = self.decode;
        self.patch_mla_nsplit(kvlen)?;
        self.patch_kvrow(dp, pos)?;

        // Stage both scalars in pinned memory for the same reason.
        {
            let s = self.h_scalar.as_mut_slice();
            s[..4].copy_from_slice(&pos.to_le_bytes());
            s[4..8].copy_from_slice(&kvlen.to_le_bytes());
        }
        let ptr_pos = self.devp[self.need(self.t_pos, "in.pos")?].base;
        let ptr_kvlen = self.devp[self.need(self.t_kvlen, "in.kvlen")?].base;
        self.be.memcpy_htod_pinned_batch(&[
            (ptr_pos, &self.h_scalar.as_slice()[..4]),
            (ptr_kvlen, &self.h_scalar.as_slice()[4..8]),
        ])?;
        Ok(())
    }

    /// The token the DEVICE sampled into `in.ids`, read after a drain.
    ///
    /// 4 bytes, not the logit row — and read back through the PINNED slab.
    /// `download` pins its destination per call, so a 4-byte readback paid a
    /// page-lock syscall every single decode step.
    pub fn read_sampled(&mut self) -> Result<u32> {
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..4], src)?;
        Ok(u32::from_le_bytes(
            self.h_scalar.as_slice()[..4].try_into().expect("4 bytes"),
        ))
    }

    /// The bucket width to shrink prefill row counts against, or `None` when
    /// `PLOW_RAGGED_CHUNK` is off (or this is the decode program).
    ///
    /// ONE function, read by both `patch_prefill` and `prefill_prepare`, because
    /// the instruction shrink and the `in.kvlen` upload are the same decision
    /// taken in two places — see `rebase_chunk_rows` for what happens when they
    /// disagree.
    fn ragged_bucket(&self, prog: usize) -> Option<u32> {
        (crate::config::RuntimeConfig::get().amd.ragged_chunk && prog != self.decode)
            .then(|| self.progs[prog].t)
    }

    /// Refuse `PLOW_RAGGED_CHUNK` on a packet whose prefill is ROW-BANDED.
    ///
    /// `PLOW_GLM_XR_BAND=K` splits each `[T, hidden]` collective into K row bands,
    /// giving the producing `Gemm` `M = T/K` at `a_row0 = i*T/K` and the
    /// `XReduce` `n = (T/K)*hidden` at its own offset. The row shrink's guard
    /// then declines the `Gemm` (its M is not `T`) but ACCEPTS the `XReduce` (its
    /// n IS a multiple of `T`), which would reduce the wrong element range —
    /// exactly the silent half-application the guard exists to prevent.
    ///
    /// Detected by the signature banding leaves and nothing else does: a non-lm_head
    /// matmul with `a_row0 != 0`, or a `MoeCombinePf` with `t_row0 != 0`. The
    /// shipped GLM-5.2 blob is unbanded (the axis is emit-time, default OFF and
    /// measured net-negative), so this is a guard, not a limitation in practice.
    fn refuse_unraggable(&self) -> Result<()> {
        if !crate::config::RuntimeConfig::get().amd.ragged_chunk {
            return Ok(());
        }
        // Say so ONCE, at the first prefill. An A/B whose two arms differ only by
        // an environment variable needs a positive signal in the log that the
        // variable reached the process; "the number moved" is not that signal.
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                buckets = ?(0..self.decode).map(|p| self.progs[p].t).collect::<Vec<_>>(),
                "PLOW_RAGGED_CHUNK: fewest-launch cover, last chunk runs at its real row count"
            )
        });
        for p in 0..self.decode {
            for (i, d) in self.pf_src[p].iter().enumerate() {
                let banded_gemm = is_lm_head_matmul(d.op)
                    && d.i[4] != 0
                    && Some(d.t[0] as usize) != self.t_logits;
                let banded_combine = d.op == DevOp::MoeCombinePf as u16 && d.i[3] != 0;
                if banded_gemm || banded_combine {
                    return Err(RuntimeError::Device(format!(
                        "PLOW_RAGGED_CHUNK cannot serve this packet: prefill bucket T={} \
                         instruction #{i} (op {}) carries a non-zero row-band offset, which is \
                         the PLOW_GLM_XR_BAND layout. The ragged row shrink would rescale the \
                         banded collective without shrinking the band's GEMM. Re-emit without \
                         PLOW_GLM_XR_BAND, or serve without PLOW_RAGGED_CHUNK.",
                        self.progs[p].t, d.op
                    )));
                }
            }
        }
        Ok(())
    }

    /// Patch a prefill program's instructions for the chunk at `[c0, c0+clen)`.
    ///
    /// The row/window families live in [`rebase_chunk`] (which is where their
    /// identities are argued and where the unit test drives them). What stays
    /// here is the one site that needs the ENGINE's state:
    ///
    /// * lm_head — the FIRST matmul writing `act.logits` → `i[4] = clen - 1`,
    ///   the chunk's last REAL row, so the sampled logits come from the last
    ///   prompt token and not from a padded one.
    fn patch_prefill_rows(
        &mut self,
        prog: usize,
        c0: u32,
        clen: u32,
        bucket: Option<u32>,
    ) -> Result<()> {
        let sz = std::mem::size_of::<DevInst64>();
        let n = self.pf_src[prog].len();
        // Rebuild from the pristine copy every chunk: patches must not
        // accumulate across chunks, and c0 changes every time.
        self.h_pf_inst.as_mut_slice()[..n * sz].copy_from_slice(as_bytes(&self.pf_src[prog]));
        // SAFETY: the slab holds exactly `n` live `DevInst64` records, just
        // written from a `&[DevInst64]`. `DevInst64` is `#[repr(C)]` POD.
        let insts: &mut [DevInst64] = unsafe {
            std::slice::from_raw_parts_mut(self.h_pf_inst.as_mut_ptr() as *mut DevInst64, n)
        };

        rebase_chunk_rows(
            insts,
            &self.tensor_names,
            c0,
            clen,
            self.progs[prog].t,
            bucket.or_else(|| self.ragged_bucket(prog)),
        );
        rebase_kda_key_factor_routes(&mut self.progs[prog].prefill_routes, clen);
        // The same `rows`/`kv_len` pair [`AmdEngine::prefill_prepare`] uploads to `in.kvlen`.
        let rows = if bucket.is_some_and(|t| clen < t) {
            clen
        } else {
            self.progs[prog].t
        };
        for route in &mut self.progs[prog].prefill_routes {
            if let PrefillSegmentRoute::SparseMla(route) = route {
                route.rebase(rows, c0)?;
            }
            if let PrefillSegmentRoute::MoeAiter(route) = route {
                route.rebase(rows)?;
            }
            if let PrefillSegmentRoute::IndexTp(route) = route {
                route.rebase(rows)?;
            }
            if let PrefillSegmentRoute::GemmLt(route) = route {
                route.rebase(rows)?;
            }
            if let PrefillSegmentRoute::MlaFold(route) = route {
                route.rebase(rows)?;
            }
        }
        rebase_mla_materialized_routes(
            insts,
            &self.tensor_names,
            &mut self.progs[prog].prefill_routes,
            rows,
            c0 + rows,
        )?;

        let mut lm = None;
        for (i, d) in insts.iter().enumerate() {
            let is_matmul = is_lm_head_matmul(d.op);
            if Some(d.t[0] as usize) == self.t_logits && is_matmul {
                lm = Some(i);
                break;
            }
        }
        // A BLOCK has no lm_head; there is no a_row0 to place and that is not
        // an error. A MODEL without one is, because the sampled logits would
        // then come from whatever row the compiler baked in.
        match (lm, self.t_logits) {
            // DIAGNOSTIC: `--amd-lm-row0` / `PLOW_LM_ROW0=1` leaves a_row0 at
            // 0 instead of the
            // chunk's last real row. It samples the WRONG row, so it is not a
            // serving mode — it exists to answer one question. The lm_head is
            // the ONLY op whose a_row0 the host patches to a non-zero value at
            // runtime, so a bug in the a_row0 path is invisible to any check
            // that inspects the packet statically (where all fp8 GEMMs carry
            // a_row0 == 0) and shows up only here.
            (Some(lm), _) if crate::config::RuntimeConfig::get().amd.lm_row0 => {
                tracing::warn!(
                    lm,
                    "PLOW_LM_ROW0=1: a_row0 left at 0 — DIAGNOSTIC, wrong row"
                );
                insts[lm].i[4] = 0;
            }
            (Some(lm), _) => insts[lm].i[4] = clen - 1,
            (None, Some(_)) => {
                // A BLOCK can DECLARE act.logits and never write it — the
                // tensor table is emitted from the model's vocabulary of names,
                // not from what this program actually produces. Refusing here
                // rejected the layer-0 A/B asset outright. Warn, because on a
                // real model the same shape means the sampled logits come from
                // whatever row the compiler baked in.
                tracing::warn!(
                    prog,
                    "act.logits is declared but no matmul writes it — no a_row0 to place \
                     (expected for a block asset, WRONG for a model)"
                );
            }
            (None, None) => {}
        }

        self.be.memcpy_htod_pinned(
            self.progs[prog].d_inst.base,
            &self.h_pf_inst.as_slice()[..n * sz],
        )
    }

    fn patch_prefill(&mut self, prog: usize, c0: u32, clen: u32) -> Result<()> {
        self.patch_prefill_rows(prog, c0, clen, self.ragged_bucket(prog))
    }

    /// Resolve a chunk plan to the programs and ranges that run it.
    ///
    /// Shared by the single-GPU prefill and the TP one, so both walk the prompt
    /// identically — a TP prefill that chunked differently from tp=1 would not
    /// be comparable token-for-token, which is the whole acceptance test.
    pub fn chunk_steps(&self, chunks: &[u32], n_prompt: u32) -> Result<Vec<ChunkStep>> {
        let mut out = Vec::with_capacity(chunks.len());
        let mut c0 = 0u32;
        for &ch in chunks {
            // The DP's cover is >= the prompt (buckets are a ladder, so the
            // last chunk usually overshoots), and it can overshoot by a whole
            // bucket. A chunk starting past the end has clen == 0, and
            // `a_row0 = clen - 1` would then wrap to u32::MAX and index the
            // logits off a row that does not exist. Stop instead.
            if c0 >= n_prompt {
                break;
            }
            let prog = (0..self.dec_lo)
                .find(|&p| self.progs[p].t == ch)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("no compiled bucket for chunk T={ch}"))
                })?;
            out.push(ChunkStep {
                prog,
                c0,
                clen: (n_prompt - c0).min(ch),
            });
            c0 += ch;
        }
        Ok(out)
    }

    /// Upload one chunk's `ids`/`pos`/`kvlen` and patch its bucket program. No
    /// dispatch.
    pub fn prefill_prepare(&mut self, prompt: &[u32], step: ChunkStep) -> Result<()> {
        let rows = if self.ragged_bucket(step.prog).is_some() {
            step.clen
        } else {
            self.progs[step.prog].t
        };
        if step.c0 as usize + rows as usize > self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "prefill chunk at {} writes {rows} rows past max_ctx {}",
                step.c0, self.max_ctx
            )));
        }
        self.vmm_ensure(self.kv_slot, step.c0 + rows)?;
        if !self.kda_conv_bank_pairs.is_empty() {
            self.kda_conv_alt_stale[self.kv_slot] = true;
        }
        let ch = self.progs[step.prog].t;
        // in.kvlen FIRST, so it can borrow the head of the staging slab before
        // ids/pos fill it — the slab is sized for exactly `ids + pos` at the
        // widest bucket and has no spare word past them.
        //
        // This is the MLA prefill's QUERY BASE, and the only place it comes
        // from. `d_flash_mla_decode` (the body `d_flash_mla_prefill` wraps)
        // computes `qpos = kv_len[b] - n_tok + t` with `n_tok = i[4]`, which
        // devgen bakes at the BUCKET width — so `kv_len` must be `c0 + ch`, not
        // `c0 + clen`, or every query in the chunk shifts down by the padding.
        // The pad rows really are part of the cache: they write `ckv`/`krot` at
        // rows `c0+clen .. c0+ch`, and a real row `i` is causally bounded at
        // `c0+i`, so it never reads one.
        //
        // Nothing wrote this during prefill before. The dense-GQA path does not
        // need it (`FlashPrefill` takes n_kv as the `i[1]` immediate that
        // [`rebase_chunk`] patches), so the omission was invisible until MLA
        // prefill landed — at which point `qpos` came out of an uninitialised
        // device word. The CUDA engine has uploaded it since it gained MLA
        // prefill ([`super::gpu`], `run_one_prefill_chunk`).
        // Under RAGGED-M the flash's `n_tok` was shrunk to `clen`, so the query
        // base `qpos = kv_len - n_tok + t` needs `kv_len = c0 + clen` to land on
        // the same absolute positions. The two are one decision, taken once by
        // `ragged_bucket`; see `rebase_chunk_rows`.
        let kv_rows = if self.ragged_bucket(step.prog).is_some() {
            step.c0 + step.clen
        } else {
            step.c0 + ch
        };
        if let Some(t) = self.t_kvlen {
            let d_kvlen = self.devp[t].base;
            self.h_scalar.as_mut_slice()[..4].copy_from_slice(&kv_rows.to_le_bytes());
            self.be
                .memcpy_htod_pinned(d_kvlen, &self.h_scalar.as_slice()[..4])?;
        }
        // ids: the chunk's tokens, ZERO-PADDED past clen. Padded rows write
        // KV nothing reads — `n_kv` bounds every later read at c0+clen.
        // pos: ABSOLUTE positions, so RoPE and the KV row agree with what
        // the decode steps will later assume.
        {
            let s = self.h_scalar.as_mut_slice();
            for i in 0..ch as usize {
                let id = if (i as u32) < step.clen {
                    prompt[(step.c0 + i as u32) as usize]
                } else {
                    0
                };
                s[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            let off = ch as usize * 4;
            for i in 0..ch as usize {
                let p = step.c0 + i as u32;
                s[off + i * 4..off + i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
        }
        let (d_ids, d_pos) = (
            self.devp[self.need(self.t_ids, "in.ids")?].base,
            self.devp[self.need(self.t_pos, "in.pos")?].base,
        );
        let nb = ch as usize * 4;
        self.be.memcpy_htod_pinned_batch(&[
            (d_ids, &self.h_scalar.as_slice()[..nb]),
            (d_pos, &self.h_scalar.as_slice()[nb..nb * 2]),
        ])?;
        self.patch_prefill(step.prog, step.c0, step.clen)
    }

    /// Stage one dense packed-prefill row set. Dispatch remains the TP wrapper's job so no rank
    /// can launch until every rank has accepted the same descriptor and input rows.
    pub fn packed_prefill_prepare(
        &mut self,
        prog: usize,
        spans: &[PrefillSpan],
        prompt_slices: &[&[u32]],
        parked: &[u32],
    ) -> Result<()> {
        if self.kv_slot != 0 {
            return Err(RuntimeError::Device(format!(
                "packed prefill requires the shared KV base, currently rebased to slot {}",
                self.kv_slot
            )));
        }
        let rung = self.check_packed_prefill_program(prog)?;
        let rows = validate_packed_prompt_slices(rung, spans, prompt_slices)?;
        self.stage_packed_prefill(prog, spans, parked)?;

        for span in spans {
            self.vmm_ensure(span.slot as usize, span.kv_len)?;
            if !self.kda_conv_bank_pairs.is_empty() {
                self.kda_conv_alt_stale[span.slot as usize] = true;
            }
        }

        if let Some(t) = self.t_kvlen {
            let stage = self.h_scalar.as_mut_slice();
            for slot in 0..self.batch {
                stage[slot * 4..slot * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
            }
            for span in spans {
                let slot = span.slot as usize;
                stage[slot * 4..slot * 4 + 4].copy_from_slice(&span.kv_len.to_le_bytes());
            }
            self.be.memcpy_htod_pinned(
                self.devp[t].base,
                &self.h_scalar.as_slice()[..self.batch * 4],
            )?;
        }

        stage_packed_prompt_rows(self.h_scalar.as_mut_slice(), rung, spans, prompt_slices)?;
        let bytes = rung as usize * 4;
        let d_ids = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let d_pos = self.devp[self.need(self.t_pos, "in.pos")?].base;
        self.be.memcpy_htod_pinned_batch(&[
            (d_ids, &self.h_scalar.as_slice()[..bytes]),
            (d_pos, &self.h_scalar.as_slice()[bytes..bytes * 2]),
        ])?;
        self.patch_prefill_rows(prog, 0, rows, Some(rung))
    }

    /// The chunk plan for a prompt, from the compiled bucket ladder.
    pub fn plan_for(&self, n_prompt: u32) -> Result<Vec<u32>> {
        self.plan_for_at_most(n_prompt, u32::MAX)
    }

    /// The chunk plan using only compiled rungs no wider than `max_bucket`.
    pub fn plan_for_at_most(&self, n_prompt: u32, max_bucket: u32) -> Result<Vec<u32>> {
        // Here rather than at load: it is the ONE call every prefill path goes
        // through (`prefill`, `prefill_span`, `AmdTpGroup::plan_for`), so a
        // banded packet cannot reach the row shrink by some other door.
        self.refuse_unraggable()?;
        // `dec_lo`, not `decode`: with a DECODE BATCH LADDER the trailing programs are decode
        // rungs, and offering one to `plan_chunks` as a prefill bucket would cover a prompt
        // with a decode program. Without a ladder the two are the same index.
        let buckets: Vec<u32> = (0..self.dec_lo).map(|p| self.progs[p].t).collect();
        // Same reason `refuse_unraggable` announces itself: an A/B whose arms
        // differ only by `PLOW_LAUNCH_ROWS` needs a positive signal that the
        // variable reached the process. "The number moved" is not that signal,
        // and neither is "the number did not move".
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            let cfg = &crate::config::RuntimeConfig::get().amd;
            tracing::info!(
                launch_rows = cfg.launch_rows.unwrap_or(LAUNCH_ROWS),
                overridden = cfg.launch_rows.is_some(),
                ragged = cfg.ragged_chunk,
                // The packet's own cap, not a constant: this line is the only
                // positive signal that a blob's widest rung reached the planner.
                max_chunk = buckets.iter().copied().max().unwrap_or(0),
                requested_max_chunk = max_bucket,
                ?buckets,
                "prefill chunk policy"
            )
        });
        plan_chunks_capped(&buckets, n_prompt, max_bucket)
    }

    pub(crate) fn prefill_chunk(&mut self, prompt: &[u32], step: ChunkStep) -> Result<()> {
        self.prefill_prepare(prompt, step)?;
        self.run_segmented(step.prog)
    }

    pub(crate) fn prefill_packed_chunk(
        &mut self,
        spans: &[PrefillSpan],
        prompts: &[&[u32]],
        parked: &[u32],
    ) -> Result<()> {
        self.clear_packed_prefill();
        let result = (|| {
            let first = spans.first().ok_or_else(|| {
                RuntimeError::Rejected("packed prefill requires at least one span".into())
            })?;
            if spans.iter().any(|s| s.program != first.program) {
                return Err(RuntimeError::Rejected(
                    "packed prefill spans do not share one compiled program".into(),
                ));
            }
            let prog = self
                .packed_prefill_prog_for(first.program as usize)
                .ok_or_else(|| {
                    RuntimeError::Rejected("prefill program has no packed topology".into())
                })?;
            let mut routed = spans.to_vec();
            for span in &mut routed {
                span.program = prog as u32;
            }
            self.packed_prefill_prepare(prog, &routed, prompts, parked)?;
            self.run_segmented(prog)
        })();
        self.clear_packed_prefill();
        result
    }

    /// Prefill `prompt`, leaving the KV cache populated for `[0, prompt.len())`
    /// and the first sampled token in `in.ids`.
    ///
    /// Returns the token the device sampled from the last real prompt row.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            return Err(RuntimeError::Device("prefill of an empty prompt".into()));
        }
        if prompt.len() > self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "prompt of {} tokens exceeds max_ctx {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        // The padded cover, not the prompt length, is what the kernels write —
        // refuse before allocating rather than clamping and writing past it.
        self.refuse_overlong_cover(prompt.len() as u32)?;
        // Back the rows this prefill writes, in whichever slot the KV base is
        // rebased onto. Here rather than in `prefill_slot` so the direct
        // single-sequence path (`amd-bench`, `AmdServe` at batch 1) is covered
        // by the same line.
        self.vmm_ensure(self.kv_slot, self.prefill_rows(prompt.len() as u32))?;

        let t_plan = std::time::Instant::now();
        let chunks = self.plan_for(prompt.len() as u32)?;
        let steps = self.chunk_steps(&chunks, prompt.len() as u32)?;
        crate::obs::ttft::PF_PLAN.add(t_plan.elapsed().as_nanos() as u64);
        crate::obs::ttft::set_cover(&chunks);
        tracing::info!(
            tokens = prompt.len(),
            chunks = ?chunks,
            "prefill plan"
        );

        for step in steps {
            let t = std::time::Instant::now();
            self.prefill_prepare(prompt, step)?;
            crate::obs::ttft::PF_PREPARE.add(t.elapsed().as_nanos() as u64);
            self.run_segmented(step.prog)?;
        }

        // The device sampled into in.ids itself; the first decode step will
        // embed it from there without the host touching it.
        let t_read = std::time::Instant::now();
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..4], src)?;
        crate::obs::ttft::PF_READ.add(t_read.elapsed().as_nanos() as u64);
        Ok(u32::from_le_bytes(
            self.h_scalar.as_slice()[..4].try_into().expect("4 bytes"),
        ))
    }

    /// Sequences one decode dispatch advances.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Point every `kv.*` tensor at sequence `slot`'s block of the cache.
    ///
    /// THE PREFILL PROGRAM IS SINGLE-SEQUENCE and always will be: its
    /// `HeadNormRope` runs with `n_batch_kv == 0`, so it writes at
    /// `hh * out_stride + row` — the *first* sequence's block relative to
    /// whatever base the pointer table hands it. Rebasing the pointer is
    /// therefore the whole of "prefill into slot s": exact, one 8-byte edit per
    /// KV buffer, and it needs no second prefill program and no kernel change.
    ///
    /// The decode program must always run at slot 0 (it derives each
    /// sequence's block itself from `n_batch_kv`), so every prefill restores
    /// the base before returning. A stale rebase would put ALL sequences'
    /// decode KV inside one slot's block — hence the invariant is enforced in
    /// [`AmdEngine::decode_step_batched`] rather than left to callers.
    pub fn kv_rebase(&mut self, slot: usize) -> Result<()> {
        if self.kv_slot == slot || self.kv_slot_stride.is_empty() {
            return Ok(());
        }
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "kv_rebase to slot {slot} past batch {}",
                self.batch
            )));
        }
        for &(i, stride) in &self.kv_slot_stride {
            let base = self.devp[i].base + stride * slot as u64;
            self.tens_table[i * 8..i * 8 + 8].copy_from_slice(&base.to_le_bytes());
        }
        // One upload of the whole table (a few KiB) beats one per KV buffer:
        // there are 2-4 per layer and the submission, not the bytes, is the
        // cost. This is off the per-token path — it happens once per prefill.
        EngineDevice::upload(&*self.be, &self.d_tens, 0, &self.tens_table)?;
        self.kv_slot = slot;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let (i, stride) = self.kv_slot_stride[0];
            let mut back = [0u8; 8];
            EngineDevice::download(&*self.be, &self.d_tens, i as u64 * 8, &mut back)?;
            tracing::debug!(
                slot,
                tensor = %self.tensor_names[i],
                want = format_args!("{:#x}", self.devp[i].base + stride * slot as u64),
                got = format_args!("{:#x}", u64::from_le_bytes(back)),
                "kv rebase readback"
            );
        }
        Ok(())
    }

    /// Prefill `prompt` into sequence slot `slot`, restoring the decode base.
    pub fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.kv_rebase(slot)?;
        let r = self.prefill(prompt);
        // Restore even on failure: a half-prefilled slot is recoverable, a
        // pointer table left pointing at slot s is not.
        self.kv_rebase(0)?;
        r
    }

    pub(crate) fn prefill_slot_cached(
        &mut self, slot: usize, prompt: &[u32], resume: u32, arm: u32,
    ) -> Result<u32> {
        self.kv_rebase(slot)?;
        let result = (|| {
            let mut from = resume;
            if resume > 0 {
                self.restore_carried(slot)?;
            } else if arm > 0 {
                let chunks = self.plan_for(arm)?;
                for step in self.chunk_steps_from(&chunks, 0, arm)? {
                    self.prefill_chunk(prompt, step)?;
                }
                self.snapshot_carried(slot, arm)?;
                from = arm;
            }
            let end = prompt.len() as u32;
            let chunks = self.plan_for(end - from)?;
            for step in self.chunk_steps_from(&chunks, from, end)? {
                self.prefill_chunk(prompt, step)?;
            }
            self.read_sampled()
        })();
        self.kv_rebase(0)?;
        result
    }

    /// Rows a prefill of `n_prompt` tokens WRITES, which is the padded bucket
    /// cover, not `n_prompt`. `prefill_prepare` zero-pads the last chunk out to
    /// its bucket width and those pad rows write KV (nothing reads them —
    /// `n_kv` bounds every later read at `c0 + clen`), so the backing has to
    /// cover them or the pad write faults.
    ///
    /// NOT CLAMPED TO `max_ctx`, and it used to be. `.min(self.max_ctx)` made this
    /// function contradict its own doc comment: the cover is exactly the count of
    /// rows the kernels WILL write, so clamping it returns a number that is not
    /// that, and the caller then backs fewer rows than the hardware touches. See
    /// [`AmdEngine::refuse_overlong_cover`] for what the caller does about it.
    fn prefill_rows(&self, n_prompt: u32) -> u32 {
        let cover: u32 = self
            .plan_for(n_prompt)
            .map(|c| c.iter().sum())
            .unwrap_or(n_prompt);
        cover.max(n_prompt)
    }

    /// Refuse a prompt whose PADDED cover runs past the compiled context.
    ///
    /// `plan_chunks` covers a prompt with compiled bucket widths, so the last
    /// chunk is rounded UP: at `max_ctx = 1500` a 1499-token prompt plans as
    /// `1024 + 512 = 1536`. Every admission check upstream tests `n_prompt`
    /// (`prefill` here, `EngineServe::prefill` at `>= max_ctx`), and 1499 passes
    /// all of them — but the kernels execute the full bucket and write padded KV
    /// rows through 1535, into a cache whose geometry is `max_ctx` rows.
    ///
    /// The old code hid this by clamping [`AmdEngine::prefill_rows`] to `max_ctx`,
    /// which does not make the writes stop — it only stops the ROWS FROM BEING
    /// BACKED. Under `PLOW_VMM_KV` that is a fault on unmapped VA; without VMM it
    /// is a silent write past the end of the KV tensor into whatever the allocator
    /// placed next. The second is the one worth refusing for: it corrupts a
    /// neighbour and reports nothing.
    ///
    /// A refusal rather than a bigger allocation, because the allocation is not
    /// this layer's to grow: `max_ctx` is read out of the compiled `in.pos` tensor
    /// and `VmmPool::ensure_rows` clamps to `geo.max_ctx` independently, so both
    /// the reservation and the mapping are sized by the PACKET. Making the padded
    /// cover fit is an emitter-side decision (pad the KV geometry to the worst
    /// bucket overshoot, or compile a terminal context-sized bucket); until it is
    /// made, this is the boundary that says so instead of writing past it.
    ///
    /// The refused band is narrow — only prompts within one bucket-rounding of
    /// `max_ctx` — and the message names all four numbers so the fix is obvious.
    fn refuse_overlong_cover(&self, n_prompt: u32) -> Result<()> {
        let cover = self.prefill_rows(n_prompt);
        if cover as usize > self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "prompt of {n_prompt} tokens plans as {:?} = {cover} padded rows, past max_ctx \
                 {}. The kernels write every row of the last bucket, so this would write KV rows \
                 [{}, {cover}) outside the cache. Shorten the prompt, or recompile with a \
                 prefill bucket that lands on {} without overshooting.",
                self.plan_for(n_prompt).unwrap_or_default(),
                self.max_ctx,
                self.max_ctx,
                self.max_ctx,
            )));
        }
        Ok(())
    }

    /// Map physical backing for `seq` out to `rows`. No-op without VMM.
    fn vmm_ensure(&self, seq: usize, rows: u32) -> Result<()> {
        if let Some(v) = &self.vmm {
            v.ensure_rows(seq, rows)?;
        }
        if let Some(v) = &self.shared_prefix {
            v.ensure_rows(seq, rows)?;
        }
        Ok(())
    }

    /// Release slot `seq`'s physical backing, remap its row 0, and CLEAR any
    /// carried recurrent state.
    ///
    /// Called when a slot is handed to a NEW sequence: the outgoing sequence's
    /// blocks are what a growable pool exists to reclaim. Row 0 goes straight
    /// back because an idle row still writes KV at `pos = 0`.
    ///
    /// # The clear, and why the KV cache does not need one but KDA does
    ///
    /// An append-only KV cache carries nothing between sequences: `kvlen` returns
    /// to 0, every row the new sequence reads is a row it wrote, and the stale
    /// bytes underneath are unreachable. That is the whole argument behind
    /// [`kv_skips_zeroing`], and it is why handing over a slot costs a pointer
    /// remap and no memset.
    ///
    /// [`is_carried_state`] tensors break that argument: the KDA recurrence READS
    /// `state` on its very first token, and the conv arms read a window that is
    /// supposed to hold the `W - 1` tokens before the sequence began. With no
    /// clear, "the tokens before this sequence began" were the previous REQUEST's
    /// — so a second prompt started from the first one's accumulated state.
    ///
    /// THE CLEAR IS WHOLE-TENSOR, AND AT `batch > 1` THAT IS NOW WRONG. This comment
    /// used to say the state was not per-slot; it is, since `declare_kda_state` gained
    /// `slots` and `k3.rs` passes `slots = t` for a sequence-rows program
    /// (`crates/devgen/src/k3.rs`, `RowKind::Sequences`). So `kv.{layer}.state` and the
    /// conv states hold B INDEPENDENT recurrences, and the `memset` below zeroes ALL of
    /// them — admitting into slot 2 would wipe slots 0/1/3 mid-stream.
    ///
    /// That is latent rather than live only because this function used to be called ONLY on
    /// the single-GPU path, and the shipped K3 config is TP8 — so on TP nothing cleared
    /// carried state at all and every request after the first on a slot inherited the
    /// previous one's recurrence across 69 of K3's 93 layers.
    ///
    /// # BOTH HALVES ARE FIXED HERE
    ///
    /// The clear is now PER SLOT, and [`AmdTpGroup::begin_slot`] calls it on every rank.
    ///
    /// The stride is `len / batch`, which is right because these tensors are SLOT-MAJOR by
    /// construction: `declare_kda_state` sizes `state` as `state_elems * 4 * slots` and
    /// `conv_state` as `proj * conv_w * 4 * slots`, and the kernels index them as
    /// `st_h + t*bstride` with `bstride = H*D*D` (state) and `C*W` (conv) — the same
    /// `[slot][...]` layout the memset now assumes. `slots` is `t` for a sequence-rows
    /// program and 1 otherwise, so at `batch == 1` this is byte-identical to the old
    /// whole-tensor clear.
    ///
    /// `kv.blkres` is EXCLUDED at `batch > 1`, and that is deliberate rather than an
    /// oversight. It is `[T][nb_cap][hidden]` sized at `max(T_max, B)` rows — T_max being
    /// the widest PREFILL bucket — so `len / batch` is NOT its row stride and a per-slot
    /// memset would clear the wrong bytes. It also carries nothing across steps: layer 0
    /// is a snapshot layer that resets the ring every forward pass, so each pass
    /// re-establishes it. Clearing it was always belt-and-braces; skipping it at `batch > 1`
    /// is strictly safer than clearing every live slot's rows.
    pub fn begin_slot(&mut self, seq: usize) -> Result<()> {
        if self.k_state_clear.is_some() {
            self.prepare_device_state_clear(seq)?;
            self.enqueue_state_clear(seq)?;
            return self.drain();
        }
        self.clear_state_serial(seq)
    }

    pub fn device_state_clear_enabled(&self) -> bool {
        self.k_state_clear.is_some()
    }

    pub fn prepare_device_state_clear(&mut self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "prepare_device_state_clear {seq} past batch {}",
                self.batch
            )));
        }
        if let Some(v) = &self.vmm {
            v.begin_seq(seq);
            v.ensure_rows(seq, 1)?;
        }
        self.reset_shared_prefix(seq)?;
        self.kda_conv_alt_stale[seq] = false;
        Ok(())
    }

    pub fn enqueue_state_clear(&self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "enqueue_state_clear {seq} past batch {}",
                self.batch
            )));
        }
        let Some(k) = self.k_state_clear else {
            return Err(RuntimeError::Device(
                "device recurrent-state clear kernel was not loaded".into(),
            ));
        };
        let Some(ranges) = &self.d_state_clear else {
            return Ok(());
        };
        let arg = StateClearArgs {
            ranges: ranges.base,
            n_ranges: self.n_state_clear,
            slot: seq as u32,
        };
        EngineDevice::launch_kernel(
            &*self.be,
            k,
            self.n_state_clear,
            256,
            0,
            as_bytes(std::slice::from_ref(&arg)),
            None,
        )
    }

    fn clear_state_serial(&mut self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "begin_slot {seq} past batch {}",
                self.batch
            )));
        }
        if let Some(v) = &self.vmm {
            v.begin_seq(seq);
            v.ensure_rows(seq, 1)?;
        }
        self.reset_shared_prefix(seq)?;
        for (i, name) in self.tensor_names.iter().enumerate() {
            if !is_carried_state(name) {
                continue;
            }
            let m = &self.devp[i];
            if m.base == 0 || m.len == 0 {
                continue;
            }
            if self.batch == 1 {
                EngineDevice::memset_d8(&*self.be, m.base, 0, m.len as usize)?;
                continue;
            }
            // See the doc above: only the slot-major carried tensors can be strided.
            if name.contains("blkres") {
                continue;
            }
            let b = self.batch as u64;
            if m.len % b != 0 {
                return Err(RuntimeError::Device(format!(
                    "carried-state tensor {name} is {} bytes, not divisible by batch {b} — \
                     its slot stride is unknown, so clearing it would corrupt live slots",
                    m.len
                )));
            }
            let stride = m.len / b;
            EngineDevice::memset_d8(&*self.be, m.base + stride * seq as u64, 0, stride as usize)?;
        }
        self.kda_conv_alt_stale[seq] = false;
        Ok(())
    }

    fn sync_kda_conv_alt(&mut self, slot: usize) -> Result<()> {
        if !self.kda_conv_alt_stale.get(slot).copied().unwrap_or(false) {
            return Ok(());
        }
        self.be.memcpy_dtod_batch(&self.kda_conv_bank_pairs)?;
        self.kda_conv_alt_stale[slot] = false;
        Ok(())
    }

    /// Bytes one slot's carried recurrent state occupies.
    pub fn carried_bytes(&self) -> u64 {
        self.carried_slot.iter().map(|&(_, n)| n).sum()
    }

    /// Has slot `slot` got a prefix snapshot armed?
    pub fn has_snapshot(&self, slot: usize) -> bool {
        self.prefix_rows.get(slot).copied().unwrap_or(0) > 0
            && (self.prefix_regions.as_ref().is_some_and(Vec::is_empty)
                || self.prefix_snap.get(slot).is_some_and(Option::is_some))
    }

    pub fn prefix_cache_capable(&self) -> bool {
        let cap = (crate::config::RuntimeConfig::get().prefix_cache_mib() as u64) << 20;
        self.shared_prefix.is_some() || (self.vmm.is_none()
            && self.prefix_regions.as_ref().is_some_and(|regions| {
                cap == 0 || regions.iter().map(prefix::Region::bytes).sum::<u64>() <= cap
            }))
    }

    pub fn shared_prefix_enabled(&self) -> bool {
        self.shared_prefix.is_some()
    }

    pub fn attach_shared_prefixes(ranks: &mut [Self], slot: usize, prompt: &[u32]) -> Result<u32> {
        if ranks.is_empty() || ranks.iter().any(|e| !e.shared_prefix_enabled() || slot >= e.batch) {
            return Err(RuntimeError::Device("shared prefix attachment requires every rank and a valid slot".into()));
        }
        shared_prefix::attach_ranks(ranks, slot, prompt, |rank| {
            rank.shared_prefix.as_mut().expect("validated above")
        })
    }

    pub fn reset_shared_prefix(&mut self, slot: usize) -> Result<()> {
        if let Some(cache) = &mut self.shared_prefix {
            cache.begin_slot(slot)?;
        }
        Ok(())
    }

    pub fn publish_shared_prefix(&self, slot: usize, prompt: &[u32], frontier: u32) -> Result<()> {
        if let Some(cache) = &self.shared_prefix {
            cache.publish_completed_chunk(slot, prompt, frontier)?;
        }
        Ok(())
    }

    pub fn release_shared_prefix(&self, slot: usize) {
        if let Some(cache) = &self.shared_prefix {
            cache.release(slot);
        }
    }

    fn evict_prefix_snapshot(&mut self, keep: usize) -> bool {
        let victim = (0..self.batch)
            .filter(|&slot| slot != keep && self.prefix_snap[slot].is_some())
            .min_by_key(|&slot| self.prefix_used[slot]);
        if let Some(slot) = victim {
            self.prefix_snap[slot] = None;
            self.prefix_rows[slot] = 0;
            true
        } else {
            false
        }
    }

    /// Capture slot `slot`'s recurrent state and sliding KV, so a later prompt sharing the prefix that
    /// produced it can resume from here instead of re-prefilling those tokens.
    ///
    /// This is the half of prefix caching that a KV cache alone cannot provide. Reusing KV rows
    /// `[0, P)` is positional and free — identical tokens at identical positions give identical
    /// K/V. The KDA recurrence is not positional: resuming at `P` requires the STATE at `P`, and
    /// there is no way to rewind it. So the state at `P` is copied out and copied back.
    ///
    /// The snapshot is exact rather than approximate because `rebase_chunk` sets every KDA op's
    /// row count to `clen`, not to the padded bucket width — so a chunk with `clen == P` leaves
    /// the recurrence at exactly `P` and the split point needs no bucket alignment.
    pub fn snapshot_carried(&mut self, slot: usize, rows: u32) -> Result<()> {
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "snapshot_carried {slot} past batch {}",
                self.batch
            )));
        }
        self.sync_kda_conv_alt(slot)?;
        let total = self.prefix_regions.as_ref().map_or_else(|| self.carried_bytes(),
            |regions| regions.iter().map(prefix::Region::bytes).sum());
        if total == 0 {
            self.prefix_rows[slot] = rows;
            return Ok(());
        }
        if self.prefix_snap[slot].is_none() {
            let cap = (crate::config::RuntimeConfig::get().prefix_cache_mib() as u64) << 20;
            while cap > 0 && self.prefix_snap.iter().flatten().map(|m| m.len).sum::<u64>() + total > cap {
                if !self.evict_prefix_snapshot(slot) { return Ok(()); }
            }
            loop {
                match EngineDevice::alloc(&*self.be, total) {
                    Ok(mem) => { self.prefix_snap[slot] = Some(mem); break; }
                    Err(RuntimeError::Oom(_)) => {
                        if !self.evict_prefix_snapshot(slot) { return Ok(()); }
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        self.prefix_tick += 1;
        self.prefix_used[slot] = self.prefix_tick;
        let dst_base = self.prefix_snap[slot]
            .as_ref()
            .expect("just allocated")
            .base;
        let pairs = self.prefix_copy_pairs(slot, rows, dst_base, false);
        // ONE completion wait for all 276 tensors. Per-copy `memcpy_dtod` blocks the host on its
        // own signal, and at this count that synchronisation — not the 56 MiB — is the cost.
        let t = std::time::Instant::now();
        self.be.memcpy_dtod_batch(&pairs)?;
        self.prefix_rows[slot] = rows;
        crate::obs::pfx::SNAP.add(t.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn prefix_copy_pairs(&self, slot: usize, rows: u32, buffer: u64, restore: bool) -> Vec<(u64, u64, u64)> {
        let capacity = self.prefix_regions.as_ref().map_or(self.carried_slot.len(),
            |regions| regions.iter().map(prefix::Region::max_copies).sum());
        let mut pairs = Vec::with_capacity(capacity);
        let mut offset = 0;
        if let Some(regions) = &self.prefix_regions {
            for region in regions {
                let base = self.devp[region.tensor].base + region.slot_bytes * slot as u64;
                region.copy_spans(rows, |dst, src, bytes| {
                    let (snapshot, live) = (buffer + offset + dst, base + src);
                    pairs.push(if restore { (live, snapshot, bytes) } else { (snapshot, live, bytes) });
                });
                offset += region.bytes();
            }
        } else {
            for &(index, stride) in &self.carried_slot {
                let (snapshot, live) = (buffer + offset, self.devp[index].base + stride * slot as u64);
                pairs.push(if restore { (live, snapshot, stride) } else { (snapshot, live, stride) });
                offset += stride;
            }
        }
        pairs
    }

    /// Put slot `slot`'s carried state back to its snapshot. The inverse of
    /// [`AmdEngine::snapshot_carried`]; a no-op refusal if nothing is armed.
    pub fn restore_carried(&mut self, slot: usize) -> Result<()> {
        if !self.has_snapshot(slot) {
            return Err(RuntimeError::Device(format!(
                "restore_carried: slot {slot} has no snapshot"
            )));
        }
        if self.prefix_regions.as_ref().is_some_and(Vec::is_empty) {
            return Ok(());
        }
        self.prefix_tick += 1;
        self.prefix_used[slot] = self.prefix_tick;
        let src_base = self.prefix_snap[slot].as_ref().expect("checked").base;
        let pairs = self.prefix_copy_pairs(slot, self.prefix_rows[slot], src_base, true);
        let t = std::time::Instant::now();
        self.be.memcpy_dtod_batch(&pairs)?;
        self.kda_conv_alt_stale[slot] = false;
        crate::obs::pfx::RESTORE.add(t.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// [`AmdEngine::chunk_steps`] starting at an arbitrary token offset.
    ///
    /// `from > 0` is a prefix-cache resume: the KV for `[0, from)` is already resident and this
    /// covers `[from, n_prompt)`. Nothing about the chunk itself is special — an ordinary second
    /// chunk is already in exactly this position, attending over KV it did not write.
    pub fn chunk_steps_from(
        &self,
        chunks: &[u32],
        from: u32,
        n_prompt: u32,
    ) -> Result<Vec<ChunkStep>> {
        let mut out = Vec::with_capacity(chunks.len());
        let mut c0 = from;
        for &ch in chunks {
            if c0 >= n_prompt {
                break;
            }
            let prog = (0..self.dec_lo)
                .find(|&p| self.progs[p].t == ch)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("no compiled bucket for chunk T={ch}"))
                })?;
            out.push(ChunkStep {
                prog,
                c0,
                clen: (n_prompt - c0).min(ch),
            });
            c0 += ch;
        }
        Ok(out)
    }

    /// Pool counters (`blocks_live` is the HBM the KV cache actually holds).
    pub fn vmm_stats(&self) -> Option<crate::memory::vmm::VmmStats> {
        self.vmm.as_ref().map(|v| v.stats())
    }

    /// Stage the per-sequence `pos` and `kvlen` for a batched decode step.
    ///
    /// Factored out of [`Self::decode_step_batched`] because the TENSOR-PARALLEL path needs the
    /// same staging without the launch: `AmdTpGroup::submit_decode` owns
    /// zero-all-then-launch-all across ranks, so it must prepare every rank first and launch them
    /// together. Duplicating this there is how the two would drift — and the failure would be a
    /// rank feeding one sequence a stale position, which is silent.
    ///
    /// `patch_kvrow` runs only at `batch == 1`. Above it, `i[3]` is dead: `devgen` arms
    /// `i[6] = n_batch_kv` on the decode `HeadNormRope` and the kernel takes BOTH the write row
    /// and the RoPE angle from `pos[t]`, so the host must not patch a single write row it no
    /// longer owns.
    pub fn decode_prepare_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<()> {
        let b = self.batch;
        // The bound lives HERE, not only in `decode_step_batched`: the TP path
        // (`amd_tp::submit_decode_batched`) calls this directly, so a guard one
        // level up left tensor-parallel decode with no refusal at all — an
        // over-long `pos` walked past the KV geometry. Every other decode entry
        // (B=1 `decode_prepare`, batched, both CUDA paths) checks it.
        if let Some(&p) = pos.iter().find(|&&p| p as usize >= self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "position {p} past max_ctx {}",
                self.max_ctx
            )));
        }
        if self.vmm.is_some() || self.shared_prefix.is_some() {
            for (slot, &position) in pos.iter().enumerate() {
                self.vmm_ensure(slot, position + 1)?;
            }
        }
        self.sync_kda_conv_alt(self.kv_slot)?;
        // ONE split count covers every row of the dispatch. Any count is CORRECT
        // for any row (a split is a partition of that row's own KV window), so
        // this is purely policy: size it for the LONGEST sequence, the one whose
        // latent stream the splits exist to parallelise.
        if let Some(&k) = kvlen.iter().max() {
            self.patch_mla_nsplit(k)?;
        }
        if b == 1 {
            self.patch_kvrow(self.decode, pos[0])?;
        }
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, p) in pos.iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
            for (i, k) in kvlen.iter().enumerate() {
                s[(b + i) * 4..(b + i) * 4 + 4].copy_from_slice(&k.to_le_bytes());
            }
        }
        let d_pos = self.devp[self.need(self.t_pos, "in.pos")?].base;
        let d_kvlen = self.devp[self.need(self.t_kvlen, "in.kvlen")?].base;
        self.be.memcpy_htod_pinned_batch(&[
            (d_pos, &self.h_scalar.as_slice()[..b * 4]),
            (d_kvlen, &self.h_scalar.as_slice()[b * 4..b * 8]),
        ])?;
        Ok(())
    }

    /// One decode step for ALL `batch` sequences, returning each one's sampled
    /// token.
    ///
    /// `pos` and `kvlen` are per-sequence and may be RAGGED at `batch > 1`.
    ///
    /// Ragged used to be refused here on the grounds that the KV write row is
    /// one host-patched immediate (`i[3]`). That is only true of a `batch == 1`
    /// program. `devgen` arms `i[6] = n_batch_kv` on every decode `HeadNormRope`
    /// when `t > 1`, and the kernel then takes BOTH the write row and the RoPE
    /// angle from `pos[t]` — `op_norm.h`:
    ///   `obase = (t*nhead + hh) * out_stride + (pos[t] & kv_mask)`, `p = pos[t] * H2`
    /// — while `flash_decode` reads `kv_len[b]` and bases K/V/Q at
    /// `b * n_kv_head`. So every position-dependent term is already
    /// per-sequence; nothing about a common `pos` was load-bearing. `i[3]` is
    /// dead on this arm, and `patch_kvrow` is skipped rather than fed a lie.
    ///
    /// At `batch == 1` the legacy single-ring formula still applies and `i[3]`
    /// is still the write row, so that path is unchanged.
    pub fn decode_step_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<Vec<u32>> {
        self.decode_step_batched_at(pos, kvlen, self.decode)
    }

    /// [`Self::decode_step_batched`] on a NAMED decode rung (`decode_prog_for`).
    ///
    /// `pos`/`kvlen` still carry all `batch` slots — the tensors are sized at the widest rung
    /// and a narrow rung simply reads the prefix it advances. Rows the rung does not cover are
    /// NOT stepped on the device, which is exactly the wasted work the ladder exists to skip;
    /// the caller guarantees they hold no live sequence.
    pub fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        self.run_decode_batched_at(pos, kvlen, dp, None)?;
        let rows = (self.progs[dp].t as usize).min(self.batch);
        let mut ids = self.read_sampled_batched(rows)?;
        ids.resize(self.batch, 0);
        Ok(ids)
    }

    pub(crate) fn decode_batched_deferred_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
        step: usize,
        quantum: usize,
    ) -> Result<()> {
        if !self.deferred_token_capture_available()
            || step >= quantum
            || quantum > DEFERRED_TOKEN_MAX_STEPS
        {
            return Err(RuntimeError::Rejected(
                "invalid deferred decode quantum".into(),
            ));
        }
        self.run_decode_batched_at(pos, kvlen, dp, Some((step, quantum)))
    }

    fn run_decode_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
        capture: Option<(usize, usize)>,
    ) -> Result<()> {
        let b = self.batch;
        if pos.len() != b || kvlen.len() != b {
            return Err(RuntimeError::Device(format!(
                "decode_step_batched wants {b} positions and {b} kvlens, got {} and {}",
                pos.len(),
                kvlen.len()
            )));
        }
        if let Some(&p) = pos.iter().find(|&&p| p as usize >= self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "position {p} past max_ctx {}",
                self.max_ctx
            )));
        }
        // Decode derives each sequence's own block; a base left rebased by a
        // prefill would funnel all B into one slot's cache.
        if self.kv_slot != 0 {
            return Err(RuntimeError::Device(format!(
                "decode with the KV base rebased onto slot {} — prefill_slot must \
                 restore it",
                self.kv_slot
            )));
        }
        self.decode_prepare_batched(pos, kvlen)?;

        self.run_with_capture(dp, self.decode_kernel_for(dp), capture)?;

        // Hand the pre-mapper the new frontier so the next block is mapped
        // BEFORE a step needs it. Never blocks; `vmm_ensure` above is the
        // correctness backstop if it falls behind.
        if let Some(v) = &self.vmm {
            for (i, &p) in pos.iter().enumerate() {
                v.advise(i, p + 1);
            }
        }
        Ok(())
    }

    /// The `b` tokens the DEVICE sampled into `in.ids`, one per sequence slot.
    ///
    /// The batched twin of [`AmdEngine::read_sampled`], and factored out of `decode_step_batched`
    /// so the TP group can read per-slot ids too: `AmdTpGroup::complete_decode` returns one id per
    /// RANK (it is an agreement check across shards), which at B>1 collapses B sequences to one.
    pub fn read_sampled_batched(&mut self, b: usize) -> Result<Vec<u32>> {
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..b * 4], src)?;
        Ok(self.h_scalar.as_slice()[..b * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4")))
            .collect())
    }

    /// Seed `in.ids` with a starting token.
    ///
    /// Needed exactly once, before the first decode step. After that the device
    /// writes its own sampled token there and the host must NOT touch it — see
    /// the module note on why `in.ids` is absent from the per-step uploads.
    ///
    /// Decoding from position 0 with a seeded id is a genuinely self-contained
    /// forward pass: the step writes KV row 0 and attends over exactly `[0,1)`,
    /// so nothing is read that was not written. Starting mid-context without a
    /// prefill would attend over KV rows nobody ever wrote, which is how a run
    /// samples the same id every step and looks like a working decoder.
    pub fn seed_ids(&mut self, ids: &[u32]) -> Result<()> {
        let n = ids.len().min(self.h_scalar.len() / 4);
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, id) in ids[..n].iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
        }
        let dst = self.devp[self.need(self.t_ids, "in.ids")?].base;
        self.be
            .memcpy_htod_pinned(dst, &self.h_scalar.as_slice()[..n * 4])
    }

    /// Publish the per-row participation mask for the next decode dispatch.
    ///
    /// `parked[s] != 0` parks row `s`: its KDA recurrence and conv window are left ALONE for that
    /// dispatch. The sense is inverted deliberately — an all-zero (or never-written) mask means
    /// every row participates, so a caller that does not know about the mask cannot break the
    /// model by omitting it. `amd-bench` is exactly such a caller. Everything else about the row still runs — `t` is compiled, so the GEMVs and the
    /// KV write happen regardless — and that is fine, because those are the parts an idle or
    /// mid-prefill row can safely redo. The recurrence is the part it cannot.
    ///
    /// A blob without `in.parked` (anything not emitted at `RowKind::Sequences`) ignores this.
    pub fn upload_parked(&mut self, parked: &[u32]) -> Result<()> {
        let Some(t) = self.t_active else {
            return Ok(());
        };
        let n = parked.len().min(self.batch);
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, a) in parked[..n].iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&a.to_le_bytes());
            }
        }
        let dst = self.devp[t].base;
        self.be
            .memcpy_htod_pinned(dst, &self.h_scalar.as_slice()[..n * 4])
    }

    /// Whether model weights were bound at load. A `false` here means the
    /// timings are real and the tokens are not.
    pub fn weights_bound(&self) -> bool {
        self.weights_bound
    }

    /// The decode program's index, for callers that want [`AmdEngine::run`]. This is the
    /// WIDEST rung, which is the one whose `t` matches `in.kvlen` and therefore the only
    /// safe answer for a caller that does not know the ladder exists (`amd-bench`, the TP
    /// audit, `patch_kvrow`).
    pub fn decode_prog(&self) -> usize {
        self.decode
    }

    /// Does this blob carry a prefill bucket ladder? `false` means the prompt has to be
    /// walked through a decode program one token at a time.
    ///
    /// Was `n_programs() == 1` at the call site, which a DECODE LADDER breaks: five rungs
    /// and no prefill is five programs and still decode-only.
    pub fn has_prefill(&self) -> bool {
        self.dec_lo > 0
    }

    /// Compiled row count for a prefill program. Decode program indices are rejected.
    pub fn prefill_prog_t(&self, prog: usize) -> Option<u32> {
        (prog < self.dec_lo && !self.progs[prog].packed_prefill_only).then(|| self.progs[prog].t)
    }

    /// The decode rung widths, ascending. One entry without a ladder.
    pub fn decode_rungs(&self) -> Vec<u32> {
        (self.dec_lo..=self.decode)
            .map(|p| self.progs[p].t)
            .collect()
    }

    /// THE LADDER SELECTION: the program index of the NARROWEST decode rung that advances
    /// at least `rows` sequence slots.
    ///
    /// `rows` is a SLOT COUNT, not a live-request count, and the difference is the whole
    /// correctness argument: slot `s` is only advanced by a rung whose width exceeds `s`, so
    /// the caller must pass `highest_live_slot + 1`. A sequence parked in slot 5 while rung 4
    /// runs would have its position stepped on the host and never on the device.
    ///
    /// Saturates at the widest rung, so an out-of-range `rows` degrades to today's behaviour
    /// rather than refusing — the slot itself is bounded by `batch` elsewhere.
    pub fn decode_prog_for(&self, rows: usize) -> usize {
        (self.dec_lo..=self.decode)
            .find(|&p| self.progs[p].t as usize >= rows)
            .unwrap_or(self.decode)
    }

    /// The decode kernel handle.
    pub fn decode_kernel(&self) -> HsaKernel {
        self.k_decode
    }

    /// Task-13 per-rung object selection: the tier ladder from
    /// PLOW_HSACO_LOWRUNG (each entry pairing-checked at its own max) serves a
    /// rung on the NARROWEST object that fits it — the dead-lane cost of a
    /// GEMV bucket is paid per compiled MM, not per live row (r14/r15: rung 1
    /// on MM16 measured +35% TPOT over MM1). Everything wider than the last
    /// tier runs the primary object. Without tiers this is `decode_kernel()`.
    pub fn decode_kernel_for(&self, dp: usize) -> HsaKernel {
        let t = self.progs[dp].t;
        for &(max, k) in &self.decode_tiers {
            if t <= max {
                return k;
            }
        }
        self.k_decode
    }

    /// Task-9 round-7 data audit: the head of sequence slot `slot`'s region of
    /// every layer-0 KV ring buffer (via the same `kv_slot_stride` table the
    /// rebase uses, so the addressing under test is the addressing measured),
    /// plus any tensor by name via [`AmdEngine::snapshot_tensor`]. Debug
    /// instrument, only reachable under PLOW_TENS_SNAP.
    pub fn snapshot_kv_slot(
        &mut self,
        slot: usize,
        bytes: usize,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "snapshot slot {slot} past engine batch {}",
                self.batch
            )));
        }
        let mut out = Vec::new();
        let picks: Vec<(usize, u64)> = self
            .kv_slot_stride
            .iter()
            .filter(|&&(i, _)| self.tensor_names[i].starts_with("kv.0."))
            .copied()
            .collect();
        for (i, stride) in picks {
            let len = bytes.min(stride as usize);
            let mut buf = vec![0u8; len];
            let name = self.tensor_names[i].clone();
            EngineDevice::download(&*self.be, &self.devp[i], stride * slot as u64, &mut buf)?;
            out.push((name, buf));
        }
        Ok(out)
    }

    /// Full download of a named tensor. Production callers validate the
    /// aggregate selection against a bounded byte budget before dispatch.
    pub fn snapshot_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let i = self.need(self.tensor_names.iter().position(|x| x == name), name)?;
        let len = self.devp[i].len as usize;
        let mut buf = vec![0u8; len];
        EngineDevice::download(&*self.be, &self.devp[i], 0, &mut buf)?;
        Ok(buf)
    }

    /// Word 0 of every counter line of program `p`'s current bank — the task-9
    /// differential audit. A deterministic program must leave the IDENTICAL
    /// counter end-state every step; a per-tick diff on the same rung names the
    /// corrupted counter and therefore the packet. Debug instrument, off the
    /// hot path (only called under PLOW_CTR_SNAP).
    pub fn ctr_word0_snapshot(&mut self, p: usize) -> Result<Vec<u32>> {
        let g = &self.progs[p];
        let span = g.n_counter as usize * CTR_STRIDE_U32 * 4;
        let mut buf = vec![0u8; span];
        EngineDevice::download(
            &*self.be,
            &g.d_ctr,
            g.bank.current() as u64 * g.ctr_span,
            &mut buf,
        )?;
        Ok(buf
            .chunks_exact(CTR_STRIDE_U32 * 4)
            .map(|c| u32::from_le_bytes(c[..4].try_into().expect("4")))
            .collect())
    }

    /// Per-program compiled `T` (decode is 1).
    pub fn prog_t(&self, p: usize) -> u32 {
        self.progs[p].t
    }

    /// Segment count for program `p`.
    pub fn prog_segments(&self, p: usize) -> usize {
        self.progs[p].seg_class.len()
    }

    /// Ordered host launches and their device-side XCD queue layout.
    pub(crate) fn prog_dispatch(&self, p: usize) -> ProgramDispatch {
        let g = &self.progs[p];
        ProgramDispatch::classify(
            g.l2_domains,
            g.seg_class.len(),
            g.gq.as_ref().map_or(0, |q| q.n_seg as usize),
        )
    }

    pub fn schedulers(&self) -> (Sched, Sched) {
        (self.sched_prefill, self.sched_decode)
    }
}

/// Reinterpret a POD slice as bytes. The blob's tables are `#[repr(C)]` mirrors
/// of the C structs, so their in-memory form IS the wire form.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a `#[repr(C)]` POD mirror (`DevInst64`, `StreamEnt`,
    // `Wait`, `u32`) whose every bit pattern is valid, and the slice is live.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The kernarg block is the `DevProgram` struct's own bytes — which is what the
/// kernarg-ring memcpy copies. `dev_isa.h` static-asserts the size and
/// `packet::dev_abi` pins the Rust mirror against the C header.
fn kernarg_bytes(p: &DevProgram) -> &[u8] {
    // `size_of`, NEVER a literal. This was hard-coded `128` and the struct grew
    // to 136 when `seg_ofs` was appended: the launcher then copied 128 bytes and
    // wrote the COv5 implicit block at `(args_size+7)&!7 == 128` — i.e. ON TOP of
    // `seg_ofs`, which the interpreter read as a device pointer. Every static
    // prefill died with `Memory access fault ... Reason: Unknown`, in BOTH arms of
    // the window A/B, because the implicit block's first word (grid dim) is
    // non-zero so the NULL fallback never triggered either.
    //
    // SAFETY: `DevProgram` is `repr(C)` and POD (u64/u32 only).
    unsafe {
        std::slice::from_raw_parts(
            p as *const DevProgram as *const u8,
            std::mem::size_of::<DevProgram>(),
        )
    }
}

#[cfg(test)]
mod tests;

/// The routed-expert NAME RESOLUTION, against synthetic checkpoints.
///
/// Three shipping spellings reach [`resolve_expert_names`], and the wrong answer
/// is SILENT: picking the block-fp8 arm for an mxfp4 checkpoint reads an E8M0
/// row as an f32 grid and gets a plausible count of plausible bytes. So each
/// spelling is pinned as the bytes a checkpoint would actually have — the tensor
/// names, the dtypes, and the shapes, all three taken from the real artifacts.
#[cfg(test)]
mod expert_name_tests;

#[cfg(test)]
mod slab_tests;
