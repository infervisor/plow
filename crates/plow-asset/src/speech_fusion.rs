//! The `speech_fusion.v1` obligation (lean-plow `Plow/SpeechFusion.lean`): the emitted operands of
//! every fused speech site, which `Plow.Speech`'s equivalences assume.
//!
//! * `DenseGemmF32` with the LayerNorm prologue (flag bit 5): the last writer of its `stats`
//!   tensor before it, in (program, pc) order, and every write of its A tensor or `stats` from
//!   that writer's program through the GEMM's program.
//! * `Conv1dF32` with `row_scale`: the raw geometry fields and operand sizes.
//!
//! Programs are taken to run in index order, one after another (a forward pipeline's sequence);
//! a write inside the GEMM's own program counts as intervening whatever its pc, since ops of one
//! program may overlap. An instruction whose writes are not audited (`logical_effects`) rejects.

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde_json::{json, Value};

use crate::logical_effects::outputs;
use crate::program::Packet;

pub const ENDPOINT: &str = "speech_fusion.v1";
const LN_PROLOGUE: u32 = 1 << 5;

fn op(d: &DevInst64) -> Option<DevOp> {
    DevOp::from_u16(d.op)
}

fn writes(d: &DevInst64, handle: u16) -> Result<bool, String> {
    Ok(outputs(d)?.iter().any(|&s| d.t[s] == handle))
}

fn bytes(p: &Packet<'_>, handle: u16) -> Result<u64, String> {
    if handle == TENSOR_NONE16 {
        return Ok(0);
    }
    p.tensors.get(handle as usize).map(|t| t.bytes).ok_or_else(|| format!("tensor handle {handle} out of range"))
}

fn ln_site(p: &Packet<'_>, gp: usize, gpc: usize) -> Result<Value, String> {
    let g = &p.programs[gp].insts[gpc];
    let (a, stats) = (g.t[1], g.t[5]);
    if stats == TENSOR_NONE16 {
        return Err(format!("program {gp} pc {gpc}: LayerNorm prologue without stats"));
    }
    let order = p.programs.iter().enumerate().flat_map(|(i, prog)| (0..prog.insts.len()).map(move |pc| (i, pc)));
    let mut writer = None;
    for (i, pc) in order.take_while(|&(i, pc)| (i, pc) < (gp, gpc)) {
        if writes(&p.programs[i].insts[pc], stats)? {
            writer = Some((i, pc));
        }
    }
    let (wp, wpc) = writer.ok_or_else(|| format!("program {gp} pc {gpc}: stats tensor {stats} has no earlier writer"))?;
    let w = &p.programs[wp].insts[wpc];
    let (mut a_writes, mut stats_writes) = (0u32, 0u32);
    for i in wp..=gp {
        for (pc, d) in p.programs[i].insts.iter().enumerate() {
            if (i, pc) == (wp, wpc) || (i, pc) == (gp, gpc) {
                continue;
            }
            a_writes += u32::from(writes(d, a)?);
            stats_writes += u32::from(writes(d, stats)?);
        }
    }
    Ok(json!({
        "kind": "ln_prologue",
        "writer_program": wp, "gemm_program": gp,
        "writer_row_stats": op(w) == Some(DevOp::RowStatsF32),
        "writer_x": w.t[1], "gemm_a": a,
        "writer_rows": w.i[0], "writer_feat": w.i[1],
        "gemm_k": g.i[2], "a_row0": g.i[4], "m": g.i[0],
        "stats_bytes": bytes(p, stats)?,
        "gamma_bytes": bytes(p, g.t[6])?, "beta_bytes": bytes(p, g.t[7])?,
        "a_writes_between": a_writes, "stats_writes_between": stats_writes,
    }))
}

fn conv_site(p: &Packet<'_>, d: &DevInst64) -> Result<Value, String> {
    let pads = d.fj[1];
    Ok(json!({
        "kind": "conv_row_scale",
        "batch": d.i[0], "in_rows": d.i[1], "out_channels": d.i[3], "kernel": d.i[4],
        "stride": d.i[5], "dilation": d.i[6], "pad_before": pads & 0xffff, "pad_after": pads >> 16,
        "out_bytes": bytes(p, d.t[0])?, "residual_bytes": bytes(p, d.t[5])?,
        "row_scale_bytes": bytes(p, d.t[7])?,
        "row_scale_is_out": d.t[7] == d.t[0], "row_scale_is_residual": d.t[7] == d.t[5],
    }))
}

/// The `speech_fusion.v1` request for a packet; `None` when it has no fused speech site.
pub fn request(p: &Packet<'_>) -> Result<Option<Value>, String> {
    let mut sites = Vec::new();
    for (gp, prog) in p.programs.iter().enumerate() {
        for (pc, d) in prog.insts.iter().enumerate() {
            match op(d) {
                Some(DevOp::DenseGemmF32) if d.i[7] & LN_PROLOGUE != 0 => sites.push(ln_site(p, gp, pc)?),
                Some(DevOp::Conv1dF32) if d.t[7] != TENSOR_NONE16 => sites.push(conv_site(p, d)?),
                _ => {}
            }
        }
    }
    Ok((!sites.is_empty()).then(|| json!({"schema": 1, "sites": sites})))
}
