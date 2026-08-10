use super::*;

fn router_program(split_plan: Option<(u32, DevOp)>) -> packet::devbuild::Program {
    router_program_b(split_plan, 1)
}

fn router_program_b(split_plan: Option<(u32, DevOp)>, nrow: u32) -> packet::devbuild::Program {
    let mut b = Builder::new(188);
    let dep = b.emit(DevOp::Nop, vec![0], &[], |_| {});
    let _ = emit_gemma_moe_router(
        &mut b,
        dep,
        10, // table
        11, // residual
        12, // projection
        13, // channel scale
        14, // per-expert scale
        if split_plan.is_some() {
            15
        } else {
            TENSOR_NONE
        },
        2816,
        128,
        8,
        (2816.0f32).powf(-0.5),
        1e-6,
        split_plan,
        nrow,
    );
    b.finish()
}

/// BATCH B>1: the batch row count reaches every router op, top-k gets one CTA per row, and
/// the score op's CTA count scales with the (row, expert) pair space.
#[test]
fn batched_router_emit_carries_b() {
    let plan = gemma_moe_router_split_plan(188, 128, 8);
    let p = router_program_b(plan, 8);
    let score = &p.insts[1];
    let topk = &p.insts[2];
    assert_eq!(score.i[2], 8, "score op carries B");
    assert_eq!(score.blocks, 128, "8 rows x 128 experts / 8 per CTA");
    assert_eq!(topk.i[3], 8, "top-k carries B");
    assert_eq!(topk.blocks, 8, "one top-k CTA per row");
    // B=1 must leave the immediate at 0 so the packet bytes never move.
    let p1 = router_program(gemma_moe_router_split_plan(188, 128, 1));
    assert_eq!(p1.insts[1].i[2], 0);
    assert_eq!(p1.insts[2].i[3], 0);
    assert_eq!(p1.insts[2].blocks, 1);
}

#[test]
fn legacy_router_emit_stays_one_original_opcode() {
    let p = router_program(None);
    assert_eq!(p.insts.len(), 2);
    let r = &p.insts[1];
    assert_eq!(r.op, DevOp::MoeRouterGemma as u16);
    assert_eq!(r.blocks, 1);
    assert_eq!(r.t[..5], [10, 11, 12, 13, 14]);
    assert_eq!(r.i[..3], [2816, 128, 8]);
}

#[test]
fn split_router_emits_parallel_score_then_one_cta_tail() {
    let p = router_program(Some((16, DevOp::MoeRouterGemmaScore)));
    assert_eq!(p.insts.len(), 3);
    let score = &p.insts[1];
    let tail = &p.insts[2];
    assert_eq!(score.op, DevOp::MoeRouterGemmaScore as u16);
    assert_eq!(score.blocks, 16);
    assert_eq!(score.t[..4], [15, 11, 12, 13]);
    assert_eq!(tail.op, DevOp::MoeRouterGemmaTopk as u16);
    assert_eq!(tail.blocks, 1);
    assert_eq!(tail.t[..3], [10, 15, 14]);
    assert_eq!(tail.wait_len, 1);
    assert_eq!(p.waits[tail.wait_ofs as usize].threshold, 16);
}

#[test]
fn fast_router_is_a_distinct_default_off_score_opcode() {
    let p = router_program(Some((16, DevOp::MoeRouterGemmaScoreFast)));
    assert_eq!(p.insts[1].op, DevOp::MoeRouterGemmaScoreFast as u16);
    assert_eq!(p.insts[1].blocks, 16);
    assert_eq!(p.insts[2].op, DevOp::MoeRouterGemmaTopk as u16);
}
