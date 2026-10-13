//! The compact terminal's tail recognition: the last instructions of a prefill program that
//! compute one token from one row (norm, head GEMM, optional softcap, argmax, fin). Shared by the
//! packed terminal (CUDA) and the offline capability inventory, so it stays device-free.
use packet::dev::{DevInst64, DevOp};

pub(crate) fn is_head(op: u16) -> bool {
    [DevOp::Gemm, DevOp::GemmSmall, DevOp::GemmMed]
        .iter()
        .any(|&o| o as u16 == op)
}

/// The compact tail's five slots: [norm, head, softcap, argmax, fin]. A packet whose final
/// norm is not the generic RmsNorm (its tail ends [head, argmax, fin]) keeps the norm in the
/// body and the terminal gathers the head's already-normalized input rows: norm = Nop.
pub(crate) fn layout(
    insts: &[DevInst64],
    rows: u32,
    logits: usize,
    ids: usize,
) -> Option<[DevInst64; 5]> {
    let nop = DevInst64 {
        op: DevOp::Nop as u16,
        blocks: 1,
        ..Default::default()
    };
    let last = insts.get(insts.len().checked_sub(4)?..)?;
    let tail: [DevInst64; 5] = if last[0].op == DevOp::RmsNorm as u16 {
        [last[0], last[1], nop, last[2], last[3]]
    } else if is_head(last[1].op) && last[2].op == DevOp::Argmax as u16 {
        [nop, last[1], nop, last[2], last[3]]
    } else {
        insts.get(insts.len().checked_sub(5)?..)?.try_into().ok()?
    };
    let [norm, head, cap, max, fin] = tail;
    let softcap = cap.op == DevOp::SoftCap as u16;
    let post_norm = norm.op == DevOp::Nop as u16;
    if rows == 0
        || !(post_norm || norm.op == DevOp::RmsNorm as u16)
        || !(head.op == DevOp::Gemm as u16 || (post_norm && is_head(head.op)))
        || [cap.op, max.op, fin.op]
            != [
                if softcap { DevOp::SoftCap } else { DevOp::Nop },
                DevOp::Argmax,
                DevOp::ArgmaxFin,
            ]
            .map(|op| op as u16)
        || (!post_norm && (norm.i[0] != rows || norm.i[1] == 0 || head.i[2] != norm.i[1]))
        || head.i[0] != 1
        || head.i[1] == 0
        || head.i[2] == 0
        || head.i[4] != rows - 1
        || (head.i[6] == 0) != (head.i[7] == 0)
        || (!post_norm && head.t[1] != norm.t[0])
        || (post_norm && head.t[1] == head.t[0])
        || head.t[0] as usize != logits
        || (softcap && (cap.t[0] != head.t[0] || cap.t[1] != head.t[0] || cap.i[0] != head.i[1]))
        || max.t[1] != head.t[0]
        || max.i[0] != head.i[1]
        || max.i[1] > 1
        || fin.t[1] != max.t[0]
        || fin.t[0] as usize != ids
        || fin.i[0] != max.blocks as u32
        || fin.i[1] > 1
        || (!post_norm
            && (norm.t[1] == norm.t[0] || norm.t[1] == head.t[0] || norm.t[1] == max.t[0]))
    {
        return None;
    }
    Some(tail)
}
