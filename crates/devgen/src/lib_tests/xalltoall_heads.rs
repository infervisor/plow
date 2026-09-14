// `emit_xalltoall_heads`'s instruction shape (`packet::dev::DevOp::XAllToAllHeads`,
// rowsplit-attention-design.md §3.1 steps 2 and 4) — the tier-1 gate for the op, pinned in
// isolation. Not wired into any recipe yet: the emitter arm (`PLOW_GLM_ROWSPLIT_ATTN`) lands
// separately, on the 8192 rung only.
use super::*;

#[test]
fn q_form_shape() {
    let mut b = Builder::new(8);
    let dst = b.tensor("act.q_a2a", 1024 * 64 * 576 * 2);
    let mut xgate = 5;
    let c = emit_xalltoall_heads(
        &mut b,
        &mut xgate,
        &[0, 1, 2, 3],
        &[],
        dst,
        1024,
        8,
        576,
        64,
        8,
        0,
        0,
    );
    let p = b.finish();
    let inst = &p.insts[c as usize];
    assert_eq!(inst.op, DevOp::XAllToAllHeads as u16);
    assert_eq!(inst.t[0], dst);
    // i0=rpr i1=nh_l i2=d i3=nh_total i4=gate i5=n_gpu i6=slot_bytes i7=dir
    assert_eq!(inst.i, [1024, 8, 576, 64, 5, 8, 0, 0]);
    assert_eq!(xgate, 6, "gate id consumed exactly once");
}

#[test]
fn o_form_carries_slot_bytes_and_dir() {
    let mut b = Builder::new(8);
    let dst = b.tensor("act.o_a2a", 8192 * 8 * 512 * 2);
    let mut xgate = 0;
    let c = emit_xalltoall_heads(
        &mut b,
        &mut xgate,
        &[0, 1, 2, 3],
        &[],
        dst,
        1024,
        8,
        512,
        64,
        8,
        75_497_472,
        1,
    );
    let p = b.finish();
    let inst = &p.insts[c as usize];
    assert_eq!(inst.i, [1024, 8, 512, 64, 0, 8, 75_497_472, 1]);
}

#[test]
#[should_panic(expected = "nh_total must be nh_l * tp")]
fn rejects_inconsistent_head_count() {
    let mut b = Builder::new(8);
    let dst = b.tensor("act.bad", 4);
    let mut xgate = 0;
    emit_xalltoall_heads(&mut b, &mut xgate, &[0], &[], dst, 1024, 8, 576, 63, 8, 0, 0);
}

#[test]
#[should_panic(expected = "dir must be 0 (Q) or 1 (O)")]
fn rejects_bad_dir() {
    let mut b = Builder::new(8);
    let dst = b.tensor("act.bad", 4);
    let mut xgate = 0;
    emit_xalltoall_heads(&mut b, &mut xgate, &[0], &[], dst, 1024, 8, 576, 64, 8, 0, 2);
}
