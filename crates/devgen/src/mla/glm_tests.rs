//! The GLM-5.2 (GlmMoeDsa) single-layer emit is the FIRST milestone-1 gate: the emitted op
//! sequence must be identical to the 34-op MoE block that runtime/tests/
//! glm52_real_block_gfx950_test.c validated on gfx950 against the HF oracle (real 256 experts,
//! real [128,128] block-fp8 scales — see the design notes, "B4-CORE DONE"). Asserting op-for-op
//! equality here, offline, means the emitted layer inherits that passing GPU result. No GPU, no
//! weights — a pure structural equivalence proof, exactly as the Gemma pick_tile tests lock in
//! the tile choice offline.
use super::*;

#[test]
fn single_row_prefill_gemv_preserves_bf16_projection_layout() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_FP8_KV", "1"), ("PLOW_UNISEG", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    for rows in [1, 2, 8, 128] {
        let mut decl = Builder::new(304);
        let n = declare_glm_rows_batched(&mut decl, &c, 81920, &[0], rows, 20, MoeEnc::Fp8Blk);
        let mut b = Builder::new(304);
        b.adopt_tensors(decl.tensors());
        let all = b.all();
        emit_glm_mla_prefill(&mut b, &c, &n, 0, 81920, rows, MoeEnc::Fp8Blk,
            n.x, &[], false, &mut 0, &all, None);
        let p = b.finish();
        let projections: Vec<_> = p.insts.iter().filter(|d|
            d.t[0] == n.qlr && d.t[1] == n.xn && d.t[2] == n.lw[0].qad
        ).collect();
        assert_eq!(projections.len(), 1);
        let d = projections[0];
        assert_eq!(d.op, if rows == 1 { DevOp::Gemv } else { DevOp::GemmSmall } as u16);
        assert_eq!(d.i[..4], [rows, 2048, 6144, 0]);
        assert_eq!(d.t[3..], [TENSOR_NONE; 5]);
    }
    let quant = kernelcaps::QuantScheme::None;
    crate::with_emit_target_amd(false, || {
        assert_eq!(glm_prefill_projection_op(&c, 1, 2048, 6144, 304, quant),
            pick_tile(1, 2048, 6144, 304, quant));
    });
    assert_eq!(glm_prefill_projection_op(&c, 1, 6144, 256, 304, quant),
        pick_tile(1, 6144, 256, 304, quant));
    assert_eq!(glm_prefill_projection_op(&c, 1, 2048, 6144, 256, quant),
        pick_tile(1, 2048, 6144, 256, quant));
    let fp4 = mxfp4_quant(MoeEnc::Mxfp4);
    assert_eq!(glm_prefill_projection_op(&c, 1, 2048, 6144, 304, fp4),
        pick_tile(1, 2048, 6144, 304, fp4));
    c.tp = 4;
    assert_eq!(glm_prefill_projection_op(&c, 1, 2048, 6144, 304, quant),
        pick_tile(1, 2048, 6144, 304, quant));
}

#[test]
fn small_prefill_splits_allocate_and_emit_matching_partial_layouts() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_FP8_KV", "1"), ("PLOW_MLA_PF_V2", ""), ("PLOW_UNISEG", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    for (rows, expected) in [(1,32), (2,32), (4,32), (8,32), (16,32), (20,32),
        (32,32), (64,32), (128,16), (256,8), (512,4), (1024,2),
        (2048,1), (4096,1), (8192,1)] {
        assert_eq!(glm_small_pf_split_cap(&c, 304, rows, false), expected);
        // A packed-topology program has no KV-split axis in its flash arm: one partial per row.
        assert_eq!(glm_small_pf_split_cap(&c, 304, rows, true), 1);
        assert_eq!(crate::with_emit_target_amd(false, || glm_small_pf_split_cap(&c, 304, rows, false)), 1);
        assert_eq!(glm_small_pf_split_cap(&c, 256, rows, false), 1);
        let mut decl = Builder::new(304);
        let n = declare_glm_rows_batched(&mut decl, &c, 81920, &[0], rows, 20, MoeEnc::Fp8Blk);
        let mut b = Builder::new(304);
        b.adopt_tensors(decl.tensors());
        let all = b.all();
        emit_glm_mla_prefill(&mut b, &c, &n, 0, 81920, rows, MoeEnc::Fp8Blk,
            n.x, &[], false, &mut 0, &all, None);
        let prog = b.finish();
        let ix = prog.insts.iter().position(|d| d.op == DevOp::FlashMlaPrefillFp8 as u16).unwrap();
        let flash = &prog.insts[ix];
        let merge = &prog.insts[ix + 1];
        assert_eq!(flash.j[1], if expected > 1 { expected } else { 0 });
        if expected > 1 {
            let segment = prog.stream.iter().find(|e| e.inst as usize == ix).unwrap().seg;
            assert!(prog.stream.iter().filter(|e| e.seg == segment).all(|e| e.inst as usize == ix));
        }
        assert_eq!(merge.op, DevOp::MlaMergeFold as u16);
        assert_eq!((merge.t[1], merge.t[2], merge.i[4]), (n.opart, n.mlpart, expected));
        assert!(decl.tensors()[n.opart as usize].bytes >= u64::from(rows) * 8 * u64::from(expected) * 512 * 4);
        assert!(decl.tensors()[n.mlpart as usize].bytes >= u64::from(rows) * 8 * u64::from(expected) * 2 * 4);
    }
}

#[test]
fn dense_prefill_rungs_populate_keys_for_sparse_decode() {
    let _guard = crate::test_env::env_guard();
    let mut c = glm_ref_cfg();
    c.tp = 8;
    let ctx = 81920;
    for sparse_prefill in ["0", "1"] {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_GLM_DSA", "1"),
            ("PLOW_GLM_DSA_PF", sparse_prefill),
            ("PLOW_GLM_FP8_KV", "1"),
        ]);
        for t in [128, 512, 2048, 8192] {
            let mut decl = Builder::new(304);
            let n = declare_glm_rows_batched(&mut decl, &c, ctx, &[0], t, 20, MoeEnc::Fp8Blk);
            assert_ne!(n.kidx_pf, TENSOR_NONE);
            let mut b = Builder::new(304);
            b.adopt_tensors(decl.tensors());
            let all = b.all();
            emit_glm_mla_prefill(
                &mut b,
                &c,
                &n,
                0,
                ctx,
                t,
                MoeEnc::Fp8Blk,
                n.x,
                &[],
                false,
                &mut 0,
                &all,
                None,
            );
            let p = b.finish();
            let writers: Vec<_> = p.insts.iter().filter(|d| d.t[0] == n.kidx[0]).collect();
            assert_eq!(
                writers.len(),
                1,
                "DSA key rows missing or duplicated at T={t}"
            );
            let writer = writers[0];
            assert_eq!(writer.op, DevOp::HeadNormRope as u16);
            assert_eq!(
                (writer.t[1], writer.i[0], writer.i[1], writer.i[2]),
                (n.kidx_pf, t, 1, 128)
            );
            assert_eq!(writer.j[1], KV_MASK_NONE);
            assert!(p
                .insts
                .iter()
                .any(|d| d.t[0] == n.kidx_pf && d.t[1] == n.xn && d.t[2] == n.lw[0].iwk));
            assert!(p.insts.iter().any(|d| d.op == DevOp::LayerNorm as u16
                && d.t[0] == n.kidx_pf
                && d.t[2] == n.lw[0].iknw
                && d.t[3] == n.lw[0].iknb));
            if sparse_prefill == "0" || t <= 2048 {
                assert!(!p.insts.iter().any(|d| matches!(
                    DevOp::from_u16(d.op),
                    Some(DevOp::IndexScorePf | DevOp::IndexTpPf | DevOp::IndexSelectPf)
                )));
                assert!(!p.insts.iter().any(|d| d.t[2] == n.lw[0].iwqb));
            }
        }
    }
}

/// The real GLM-5.2-FP8 config dims. `layers` is trimmed — the single
/// block only touches one layer.
fn glm_ref_cfg() -> GlmCfg {
    GlmCfg {
        layers: 4,
        hidden: 6144,
        heads: 64,
        kv_lora: 512,
        q_lora: 2048,
        qk_nope: 192,
        qk_rope: 64,
        v_head: 256,
        vocab: 154880,
        eps: 1e-5,
        n_exp: 256,
        top_k: 8,
        n_group: 1, // GLM-5.2 does not group-limit (why the flat top-k matched its oracle)
        topk_group: 1,
        moe_inter: 2048,
        dense_inter: 12288,
        first_k_dense: 3,
        route_scale: 2.5,
        attn_scale: (256f32).powf(-0.5),
        rope_theta: Some(8_000_000.0),
        rope_scale: packet::rope::RopeScale::None,
        prefix: "model.".into(),
        tp: 1,
        ep: false,
        group: false,
        index_heads: 32,
        index_dim: 128,
        index_topk: 2048,
        index_kpool: 1,
        // indexer_types[0..4] = full,full,full,shared (real GLM-5.2 pattern); irrelevant to these
        // ctx=512 offline tests (DSA is gated OFF at ctx<=2048) but set for completeness.
        indexer_full: vec![true, true, true, false],
        softmax_layers: vec![],
        has_dsa: true,
    }
}

fn emitted_ops(use_fp8: bool) -> Vec<u16> {
    let c = glm_ref_cfg();
    let mut b = Builder::new(256);
    // Emit MoE layer 3 (the B4 oracle's layer), matching the harness.
    let tn = declare_glm(&mut b, &c, 512, &[3]);
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors);
    let mut xgate = 0u32;
    emit_glm_block(
        &mut b2,
        &c,
        &tn,
        0,
        512,
        1,
        1,
        MoeEnc::from_flags(use_fp8, false),
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    b2.finish().insts.iter().map(|d| d.op).collect()
}

/// The reference MoE-block op sequence, in emission order. This is the B4 harness sequence
/// (glm52_real_block_gfx950_test.c) with the two rope-slice GEMVs each followed by a dynamic
/// interleaved HeadNormRope (HD=64) instead of a position-FOLDED GEMV — the production form that
/// runtime/tests/glm52_run.c validates on gfx950 (dynamic rope at a fixed position reproduces the
/// folded B4 numbers). The folded B4 result is inherited by transitivity: dynamic-at-fixed-pos ==
/// the fold, proven numerically by the glm52_run ms1 gate.
fn ref_sequence(use_fp8: bool) -> Vec<u16> {
    use DevOp::*;
    let (glu, down) = if use_fp8 {
        (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
    } else {
        (MoeExpertGlu, MoeExpertDown)
    };
    let mut ops = vec![
        RmsNorm,        // input_layernorm
        GemvQkv,        // FUSED A: q_a + kv_a + k_rope input projections (share xn) -> one GemvQkv
        RmsNorm,        // q_a_layernorm
        GemvQkv,        // FUSED G: Wqa (absorbed q_nope) + Wqr (q_rope) -> one GemvQkv
        HeadNormRope,   // q_rope dynamic interleaved RoPE (HD=64)
        RmsNorm,        // kv_a_layernorm -> latent cache
        HeadNormRope,   // k_rope dynamic interleaved RoPE -> rope cache
        FlashMlaDecode, // MLA flash
        MlaMergeFold,   // fused latent merge + W_uv fold (was FlashMerge + OUvFold)
        Gemv,           // o_proj
        Residual,       // x_mid
        RmsNorm,        // post_attention_layernorm
        Gemv,           // router SCORE GEMV (multi-CU wave-cooperative; the router split)
        MoeRouterTopk,  // router tail: sigmoid+bias+norm_topk+scale (1-CU bit-exact selection)
        GemvGlu,        // shared expert gate|up
        Gemv,           // shared expert down
    ];
    for _ in 0..8 {
        ops.push(glu);
        ops.push(down);
    }
    ops.push(MoeCombine);
    ops.into_iter().map(|o| o as u16).collect()
}

#[test]
fn glm_block_matches_reference_bf16() {
    let _guard = crate::test_env::env_guard();
    assert_eq!(
        emitted_ops(false),
        ref_sequence(false),
        "bf16 op sequence != reference"
    );
}

#[test]
fn glm_block_matches_reference_fp8() {
    let _guard = crate::test_env::env_guard();
    assert_eq!(
        emitted_ops(true),
        ref_sequence(true),
        "block-fp8 op sequence != reference"
    );
}

/// The dense (layers 0-2) block op sequence: shared MLA (16 ops) + block-fp8 SwiGLU (dense GLU
/// op 47, dense down GEMV_FP8_BLK op 44) + residual = 19 ops.
fn emitted_dense_ops() -> Vec<u16> {
    let c = glm_ref_cfg();
    let mut b = Builder::new(256);
    let tn = declare_glm(&mut b, &c, 512, &[0]); // layer 0 is dense (first_k_dense_replace=3)
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors);
    let mut xgate = 0u32;
    emit_glm_dense_block(
        &mut b2,
        &c,
        &tn,
        0,
        512,
        1,
        1,
        MoeEnc::Fp8Blk,
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    b2.finish().insts.iter().map(|d| d.op).collect()
}

/// Emit ONE MoE layer (slot layer 3) at `ctx`, with the indexer 'full'/'shared'/off, and return
/// the op sequence. `full` binds an indexer on layer 3; `ctx>2048` arms the DSA gate.
fn emitted_ops_dsa(ctx: u32, full: bool) -> Vec<u16> {
    let mut c = glm_ref_cfg();
    c.indexer_full = vec![false, false, false, full]; // layer 3 = MoE; full toggles its indexer
    let mut b = Builder::new(256);
    let tn = declare_glm(&mut b, &c, ctx, &[3]);
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors);
    let mut xgate = 0u32;
    emit_glm_block(
        &mut b2,
        &c,
        &tn,
        0,
        ctx,
        1,
        1,
        MoeEnc::Fp8Blk,
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    b2.finish().insts.iter().map(|d| d.op).collect()
}

#[test]
fn glm_dsa_gate_off_below_cutover() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // ctx<=CROSSOVER (65536): NO DSA ops, dense FlashMlaDecode — byte-identical to the non-DSA MoE
    // block. 32768 is in the mid-ctx band, where the measured full-model TP4 winner is dense.
    let ops = emitted_ops_dsa(32768, true);
    assert!(
        ops.contains(&(FlashMlaDecode as u16)),
        "dense flash below cutover"
    );
    assert!(
        !ops.contains(&(FlashGatherDecode as u16)),
        "no gather below cutover"
    );
    assert!(
        !ops.contains(&(IndexScore as u16)),
        "no indexer below cutover"
    );
    assert_eq!(
        ops,
        ref_sequence(true),
        "ctx<=2048 == plain MoE block (DSA off)"
    );
}

/// Emit one MoE layer at `ctx`/`tp` and hand back the MLA flash-decode packet itself, so a
/// test can read the fields the INTERPRETER dispatches on rather than just the opcode. Either
/// opcode counts: `FlashGatherDecode` (the DSA arm, above the 64k cutover) and
/// `FlashMlaDecode` are two instantiations of ONE wrapper, `exec_flash_mla_decode`, and both
/// read GF from `i[7]` — so both had the missing GF=8 arm and both have it now.
fn glm_flash_pkt(ctx: u32, tp: u32) -> crate::DevInst {
    let mut c = glm_ref_cfg();
    c.tp = tp;
    c.indexer_full = vec![false, false, false, false]; // keep the dense flash arm, not GATHER
    let mut b = Builder::new(256);
    let tn = declare_glm(&mut b, &c, ctx, &[3]);
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors);
    let mut xgate = 0u32;
    emit_glm_block(
        &mut b2,
        &c,
        &tn,
        0,
        ctx,
        1,
        1,
        MoeEnc::Fp8Blk,
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    b2.finish()
        .insts
        .into_iter()
        .find(|d| d.op == DevOp::FlashMlaDecode as u16 || d.op == DevOp::FlashGatherDecode as u16)
        .expect("an MLA flash-decode packet")
}

/// THE REVERSE COVERAGE CHECK for the GF=8 arm (knob-contract §4, read in the direction that
/// guard does NOT cover: *an arm exists — does anything route to it, and does the packet the
/// emitter builds match the body the interpreter will pick?*).
///
/// This is the test that would have caught the original bug. `glm_gf` returned 8 on every
/// long-context GLM blob for as long as the crossover has existed, `exec_flash_mla_decode`
/// dispatched `if (gf == 2) <2> else <4>`, and the GF=8 body did not exist — so `i[7] = 8`
/// selected the GF=4 arm silently. Nothing failed, nothing warned, and `flash_mla_cus` was
/// written to MIRROR the wrong dispatch so the workgroup count stayed self-consistent with it.
///
/// `blocks` is the load-bearing half: the kernel grid-strides `w = slice; w < n_work` over
/// `n_batch*n_tok*(nh_l/GF)*nsplit`, so the packet's width has to be derived from the SAME GF
/// the interpreter will instantiate. If these two ever disagree again, either work is dropped
/// (width > n_work is only wasteful; width derived from a LARGER GF than the body uses drops
/// items) or the chip is under-filled without anyone noticing.
#[test]
fn glm_flash_decode_packet_matches_the_arm_the_interpreter_dispatches() {
    // Every assertion below reads `i[7]`, which `glm_gf` resolves from a LIVE `PLOW_GLM_GF`
    // read — so a sibling test holding the `=8` pin makes this one fail with `left: 8`
    // against a change that is entirely innocent. Observed once in a full-suite run while
    // migrating the knobs to EmitConfig; the guard is the discipline `test_env`'s header
    // already states for any test whose emitted shape depends on a live knob.
    let _g = crate::test_env::env_guard();
    // The packet-level harness is TP1 only: `emit_glm_mla` at tp>1 hands some collective an
    // empty CU list in this single-block fixture ("an op must run at least one CU"), at every
    // ctx and on both sides of this change. The TP4 shape is asserted arithmetically instead,
    // in `mla_fold_is_sized_to_its_work_items_and_never_flips_vt`.
    let fl = glm_flash_pkt(32768, 1);
    // GF=4, NOT 8, and this assertion is the whole point of the test now. `PLOW_GLM_GF8_ARM`
    // defaults to 0 (op_attention.h — the arm is a +32% decode regression by mere presence),
    // so the interpreter instantiates {2,4} and an emitted 8 would run the GF=4 body. It would
    // ALSO narrow `blocks` to (nh_l/8)*nsplit, because 9dc27bb made `flash_mla_cus` read i[7]
    // literally: the emitter would hand HALF the workgroups to a body that has full-GF work to
    // do. Measured cost of that mismatch on GLM-5.2 TP4 (arm-absent object, per-layer chain):
    // 97.6 -> 83.7 us at ctx 8192 and 168.1 -> 135.9 us at 32768; end-to-end median ITL over
    // 78 layers 28.58 -> 27.45 ms and 34.81 -> 31.49 ms, token-identical.
    assert_eq!(
        fl.i[7], 4,
        "long ctx bakes the GF the default object actually instantiates"
    );
    // nsplit is capped for GLM_MLA_GF=4: fill = ceil(256/(64/4)) = 16, below ctx/NS_PER = 128.
    assert_eq!(fl.i[4], 16, "nsplit");
    // ... so the work-item count is (64/4)*16 = 256, the whole chip, and `blocks` matches it.
    assert_eq!(
        fl.blocks, 256,
        "GF=4 => (nh_l/4)*nsplit workgroups, chip-wide"
    );

    // Short ctx stays on the GF=2 arm, which has always existed and always been dispatched.
    let sh = glm_flash_pkt(1024, 1);
    assert_eq!(
        (sh.i[7], sh.blocks),
        (2, 256),
        "max_ctx <= 4096 keeps GF=2, chip-wide"
    );

    // Every GF the emitter can bake MUST be one the interpreter instantiates. The set is
    // {2,4,8} and `exec_flash_mla_decode` dispatches exactly those three; anything else lands
    // in the `else` and silently runs GF=4, which is the bug this test exists to prevent.
    for &ctx in &[512u32, 1024, 4096, 8192, 32768, 131072] {
        let g = glm_flash_pkt(ctx, 1).i[7];
        assert!(matches!(g, 2 | 4 | 8), "ctx={ctx}: uninstantiated GF {g}");
    }
}

/// THE SELECTOR MUST BE TOLD THE LIVE KV LENGTH, AND THE INDEXER MUST DECLARE ITS GEOMETRY.
///
/// Both halves are field-level and therefore invisible to every op-sequence test in this file:
/// the ops were all present and in the right order the whole time.
///
/// 1. `IndexSelect.t[4] = in.kvlen`. `i[0]` is the packet's MAX ctx, but `INDEX_SCORE` writes
///    `iscore[pos]` only for `pos < kvlen`. Without the operand the radix ranked `ctx - kvlen`
///    never-written words — and since DSA arms only above a 64k crossover, that was nearly the
///    whole array on any real decode step. The selector then handed the gather positions past
///    the end of the cache and `d_flash_mla_decode<...,GATHER=true>` applies NO mask, so those
///    rows were read as if they were real. `runtime/nvidia/op_dsa.cuh` records the same class of
///    defect against this kernel as `[RAG]`.
/// 2. `IndexScore.i[1]/i[3]` = the indexer geometry the ISA contract has always specified. They
///    were left at ZERO while `interp.hip` hardcoded `DI_=128, HI_=32`, so a checkpoint with a
///    different geometry parsed cleanly and was silently strided wrong. `GlmCfg::dsa` now
///    refuses that outright; these fields make the packet self-describing so the two cannot
///    drift again without the assert catching it.
#[test]
fn glm_dsa_selector_is_bound_to_the_live_kv_length_and_declares_its_geometry() {
    let _guard = crate::test_env::env_guard();
    let mut c = glm_ref_cfg();
    c.indexer_full = vec![false, false, false, true];
    let ctx = 131072; // above CROSSOVER, so the DSA arm is live
    let mut b = Builder::new(256);
    let tn = declare_glm(&mut b, &c, ctx, &[3]);
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors.clone());
    let mut xgate = 0u32;
    emit_glm_block(
        &mut b2,
        &c,
        &tn,
        0,
        ctx,
        1,
        1,
        MoeEnc::Fp8Blk,
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    let insts = b2.finish().insts;
    let kvlen = tensors
        .iter()
        .position(|t| t.name == "in.kvlen")
        .expect("in.kvlen declared") as u32;

    let sel = insts
        .iter()
        .find(|d| d.op == DevOp::IndexSelect as u16)
        .expect("an IndexSelect packet");
    assert_eq!(
        sel.t[4], kvlen,
        "IndexSelect must read the LIVE kv length; i[0]={} is only the max ctx, and the score \
             kernel writes nothing past kvlen",
        sel.i[0]
    );
    assert_eq!(
        sel.i[0], ctx,
        "i[0] stays the max ctx (the scan upper bound)"
    );
    assert_eq!(sel.i[1], c.index_topk, "i[1] is the top_k ceiling");

    let sc = insts
        .iter()
        .find(|d| d.op == DevOp::IndexScore as u16)
        .expect("an IndexScore packet");
    assert_eq!(sc.i[1], c.index_heads, "i[1] = index_heads, per dev_isa.h");
    assert_eq!(sc.i[3], c.index_dim, "i[3] = index_head_dim, per dev_isa.h");
    // ...and those are the ONLY values the kernel can execute, so the emitter must not be able
    // to produce anything else. `d_index_score_mfma` static_asserts HIc == 32.
    assert_eq!(
        (sc.i[1], sc.i[3]),
        (32, 128),
        "the geometry interp.hip hardcodes"
    );
}

#[test]
fn glm_sparse_fp8_cache_writer_and_attention_operands() {
    let _g = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_FP8_KV", "1"),
        ("PLOW_GLM_DSA", "1"),
        ("PLOW_GLM_FUSE_ROPE", "0"),
        ("PLOW_GLM_DSA_PF", "1"),
    ]);
    let mut c = glm_ref_cfg();
    c.heads = 8;
    c.indexer_full[3] = true;
    let ctx = 81920;
    let mut declarations = Builder::new(256);
    let n = declare_glm_rows_batched(&mut declarations, &c, ctx, &[3], 8192, 16, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    assert_eq!(tensors[n.ckv[0] as usize].bytes, 16 * u64::from(ctx) * 512);
    assert_eq!(
        tensors[n.kv_scale[0] as usize].bytes,
        16 * u64::from(ctx) * 4
    );
    for rows in [1, 8, 16] {
        let mut b = Builder::new(256);
        b.adopt_tensors(tensors.clone());
        emit_glm_mla(
            &mut b,
            &c,
            &n,
            0,
            ctx,
            rows,
            16,
            MoeEnc::Fp8Blk,
            n.x,
            &[],
            false,
            &mut 0,
            &[],
        );
        let p = b.finish();
        let writer = p.insts.iter().find(|d| d.t[0] == n.ckv[0]).unwrap();
        assert_eq!(writer.op, DevOp::HeadNormRopeFp8 as u16);
        assert_eq!((writer.i[0], writer.i[6], writer.j[0]), (rows, rows, ctx));
        assert_eq!(writer.t[6], n.kv_scale[0]);
        let flash = p
            .insts
            .iter()
            .find(|d| d.op == DevOp::FlashMlaDecodeFp8 as u16)
            .unwrap();
        assert_eq!(
            (flash.t[7], flash.j[0], flash.i[6]),
            (n.kv_scale[0], n.iidx + 1, 2048)
        );
    }
    let mut b = Builder::new(256);
    b.adopt_tensors(tensors);
    emit_glm_mla_prefill(
        &mut b,
        &c,
        &n,
        0,
        ctx,
        8192,
        MoeEnc::Fp8Blk,
        n.x,
        &[],
        false,
        &mut 0,
        &[],
        None,
    );
    let p = b.finish();
    let flash = p
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashMlaPrefillFp8 as u16)
        .unwrap();
    assert_eq!(
        (flash.t[7], flash.j[0], flash.i[6]),
        (n.kv_scale[0], n.iuni + 1, 16384)
    );
}

#[test]
fn glm_dsa_local_selection_keeps_one_completion_for_independent_rows() {
    let _g = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[("PLOW_GLM_SELECT_LOCAL", "1")]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    c.indexer_full[3] = true;
    let ctx = 81920;
    let mut declarations = Builder::new(304);
    let n = declare_glm_rows_batched(&mut declarations, &c, ctx, &[3], 8192, 20, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    for rows in [1, 2, 4, 8, 16, 20] {
        let mut b = Builder::new(304);
        b.adopt_tensors(tensors.clone());
        let ready = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let complete = emit_glm_dsa_decode_select(
            &mut b,
            &c,
            &n,
            &n.lw[0],
            0,
            ctx,
            rows,
            20,
            MoeEnc::Fp8Blk,
            &(0..304).collect::<Vec<_>>(),
            c.eps as f32,
            c.q_lora,
            c.hidden,
            ready,
            ready,
            &(0..32).collect::<Vec<_>>(),
            &[0],
            None,
        );
        let p = b.finish();
        let selects: Vec<_> = p
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::IndexSelect as u16)
            .collect();
        assert_eq!(selects.len(), 1);
        let (ix, d) = selects[0];
        assert_eq!(complete as usize, ix);
        assert_eq!((d.i[3], d.i[4]), (0, u32::from(rows > 1)));
        assert_eq!(u32::from(d.blocks), if rows == 1 { 32 } else { rows });
        if rows > 1 {
            assert_eq!([d.t[2], d.t[3]], [TENSOR_NONE; 2]);
        }
        let mut slices: Vec<_> = p
            .stream
            .iter()
            .filter(|e| e.inst as usize == ix)
            .map(|e| e.slice)
            .collect();
        slices.sort_unstable();
        assert_eq!(slices, (0..u32::from(d.blocks)).collect::<Vec<_>>());
    }
}

#[test]
fn glm_decode_glue_cus_gives_the_key_norm_one_workgroup_per_row() {
    let _g = crate::test_env::env_guard();
    let mut c = glm_ref_cfg();
    c.tp = 8;
    c.indexer_full[3] = true;
    let ctx = 81920;
    let mut declarations = Builder::new(304);
    let n = declare_glm_rows_batched(&mut declarations, &c, ctx, &[3], 8192, 20, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    for glue in ["0", "1"] {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_GLM_SELECT_LOCAL", "1"),
            ("PLOW_GLM_DECODE_GLUE_CUS", glue),
        ]);
        for rows in [2, 8, 20] {
            let mut b = Builder::new(304);
            b.adopt_tensors(tensors.clone());
            let ready = b.emit(DevOp::Nop, vec![0], &[], |_| {});
            let rq: Vec<u32> = (0..3).collect();
            let rk: Vec<u32> = vec![3];
            emit_glm_dsa_decode_select(
                &mut b,
                &c,
                &n,
                &n.lw[0],
                0,
                ctx,
                rows,
                20,
                MoeEnc::Fp8Blk,
                &(0..304).collect::<Vec<_>>(),
                c.eps as f32,
                c.q_lora,
                c.hidden,
                ready,
                ready,
                &rq,
                &rk,
                None,
            );
            let p = b.finish();
            let blocks = |op: DevOp| -> Vec<u32> {
                let ix: Vec<usize> = p
                    .insts
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.op == op as u16)
                    .map(|(i, _)| i)
                    .collect();
                ix.iter().map(|&i| u32::from(p.insts[i].blocks)).collect()
            };
            let norm = blocks(DevOp::LayerNorm);
            assert_eq!(norm, [if glue == "1" { rows } else { 1 }], "glue={glue} rows={rows}");
        }
    }
}

#[test]
fn glm_dsa_split_selection_gives_each_row_its_own_group_and_strips() {
    let _g = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_SELECT_LOCAL", "1"),
        ("PLOW_GLM_SELECT_SPLIT", "15"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    c.indexer_full[3] = true;
    let ctx = 81920;
    let mut declarations = Builder::new(304);
    let n = declare_glm_rows_batched(&mut declarations, &c, ctx, &[3], 8192, 20, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    for rows in [1, 2, 4, 8, 16, 20] {
        let mut b = Builder::new(304);
        b.adopt_tensors(tensors.clone());
        let ready = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let complete = emit_glm_dsa_decode_select(
            &mut b,
            &c,
            &n,
            &n.lw[0],
            0,
            ctx,
            rows,
            20,
            MoeEnc::Fp8Blk,
            &(0..304).collect::<Vec<_>>(),
            c.eps as f32,
            c.q_lora,
            c.hidden,
            ready,
            ready,
            &(0..32).collect::<Vec<_>>(),
            &[0],
            None,
        );
        let p = b.finish();
        let selects: Vec<_> = p
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::IndexSelect as u16)
            .collect();
        assert_eq!(selects.len(), 1);
        let (ix, d) = selects[0];
        assert_eq!(complete as usize, ix);
        if rows == 1 {
            // One row keeps the serialized cooperative form.
            assert_eq!((d.i[4], u32::from(d.blocks)), (0, 32));
            continue;
        }
        let g = 15.min(304 / rows);
        assert_eq!((d.i[3], d.i[4], d.i[5]), (0, 2, g));
        assert_eq!(u32::from(d.blocks), rows * g);
        assert_eq!(
            [d.t[0], d.t[1], d.t[2], d.t[3], d.t[4]],
            [n.iidx, n.iscore, n.ighist, n.igctl, n.kvlen]
        );
        let mut slices: Vec<_> = p
            .stream
            .iter()
            .filter(|e| e.inst as usize == ix)
            .map(|e| e.slice)
            .collect();
        slices.sort_unstable();
        assert_eq!(slices, (0..rows * g).collect::<Vec<_>>());
    }
}

#[test]
fn glm_dsa_decode_batch_strides_producers_and_serializes_selection() {
    let _g = crate::test_env::env_guard();
    let mut c = glm_ref_cfg();
    c.indexer_full[3] = true;
    let ctx = 81920;
    let mut tb = Builder::new(256);
    let n = declare_glm_rows_batched(&mut tb, &c, ctx, &[3], 8192, 8, MoeEnc::Fp8Blk);
    let tensors = tb.tensors();
    for (handle, bytes) in [
        (n.qidx, 8 * 32 * 128 * 2),
        (n.kidx_raw, 8 * 128 * 2),
        (n.kidx_normed, 8 * 128 * 2),
        (n.widx, 8 * 32 * 2),
        (n.iscore, 8 * ctx * 4),
        (n.iidx, 8 * 2048 * 4),
    ] {
        assert_eq!(tensors[handle as usize].bytes, bytes as u64);
    }
    for rows in [1, 2, 8] {
        let mut b = Builder::new(256);
        b.adopt_tensors(tensors.clone());
        emit_glm_block(
            &mut b,
            &c,
            &n,
            0,
            ctx,
            rows,
            8,
            MoeEnc::Fp8Blk,
            n.x,
            n.xnext,
            &[],
            &mut 0,
            &[],
        );
        let prog = b.finish();
        for output in [n.qidx, n.kidx_raw, n.kidx_normed, n.widx, n.iscore] {
            for d in prog.insts.iter().filter(|d| d.t[0] == output) {
                assert_eq!(d.i[0], rows, "producer op {}", d.op);
            }
        }
        let writer = prog.insts.iter().find(|d| d.t[0] == n.kidx[0]).unwrap();
        assert_eq!((writer.i[0], writer.i[6], writer.j[0]), (rows, rows, ctx));
        let select: Vec<_> = prog
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::IndexSelect as u16)
            .collect();
        assert_eq!(select.len(), rows as usize);
        for (row, (_, d)) in select.iter().enumerate() {
            assert_eq!((d.i[0], d.i[1], d.i[3]), (ctx, 2048, row as u32));
            assert_eq!(d.t[4], n.kvlen);
        }
        for pair in select.windows(2) {
            let before = prog
                .stream
                .iter()
                .find(|e| e.inst as usize == pair[0].0)
                .unwrap();
            let after = prog
                .stream
                .iter()
                .find(|e| e.inst as usize == pair[1].0)
                .unwrap();
            let succs = &prog.succs[before.succ_ofs as usize..][..before.succ_len as usize];
            let waits = &prog.waits[after.wait_ofs as usize..][..after.wait_len as usize];
            assert!(waits.iter().any(|w| succs.contains(&w.id)));
        }
    }
}

#[test]
fn glm_dsa_full_layer_emits_indexer() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // ctx>CROSSOVER, 'full': indexer (2 fp8 projections + LayerNorm + 2 rope + weights_proj GEMV +
    // score + select) then FLASH_GATHER (not dense).
    let ops = emitted_ops_dsa(131072, true);
    assert!(
        ops.contains(&(IndexScore as u16)),
        "full layer scores the indexer"
    );
    assert!(
        ops.contains(&(IndexSelect as u16)),
        "full layer selects top-k"
    );
    assert!(
        ops.contains(&(LayerNorm as u16)),
        "full layer k_norm LayerNorm"
    );
    assert!(ops.contains(&(FlashGatherDecode as u16)), "gather flash");
    assert!(
        !ops.contains(&(FlashMlaDecode as u16)),
        "no dense flash under DSA"
    );
}

#[test]
fn glm_dsa_shared_layer_reuses_idx() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // ctx>CROSSOVER, 'shared': NO indexer ops (reuses the last full layer's idx) but still GATHERs.
    let ops = emitted_ops_dsa(131072, false);
    assert!(
        !ops.contains(&(IndexScore as u16)),
        "shared layer emits no score"
    );
    assert!(
        !ops.contains(&(IndexSelect as u16)),
        "shared layer emits no select"
    );
    assert!(
        !ops.contains(&(LayerNorm as u16)),
        "shared layer emits no k_norm"
    );
    assert!(
        ops.contains(&(FlashGatherDecode as u16)),
        "shared layer still gathers"
    );
}

fn emitted_dsa_pooled(ctx: u32, full: bool, index_kpool: u32) -> packet::devbuild::Program {
    let mut c = glm_ref_cfg();
    c.index_kpool = index_kpool;
    c.indexer_full = vec![false, false, false, full];
    let mut b = Builder::new(256);
    let tn = declare_glm(&mut b, &c, ctx, &[3]);
    let tensors = b.tensors();
    let mut b2 = Builder::new(256);
    b2.adopt_tensors(tensors);
    let mut xgate = 0u32;
    emit_glm_block(
        &mut b2,
        &c,
        &tn,
        0,
        ctx,
        1,
        1,
        MoeEnc::Fp8Blk,
        tn.x,
        tn.xnext,
        &[],
        &mut xgate,
        &[],
    );
    b2.finish()
}

fn emitted_ops_dsa_pooled(ctx: u32, full: bool, index_kpool: u32) -> Vec<u16> {
    emitted_dsa_pooled(ctx, full, index_kpool)
        .insts
        .iter()
        .map(|d| d.op)
        .collect()
}

#[test]
fn glm_dsa_pool_size_one_is_byte_identical_to_dense() {
    let _guard = crate::test_env::env_guard();
    // index_kpool=1 is the explicit no-op path — same op multiset as the ordinary
    // (unspecified-index_kpool, defaults to 1 in glm_ref_cfg) dense-indexer test above,
    // for BOTH full and shared layers. This is the hard regression bar for the whole
    // pool_size>1 composition: GLM-5.2 (which never sets index_kpool) must never reach
    // the pooled branch at all.
    let mut full_dense = emitted_ops_dsa(131072, true);
    let mut full_pooled_1 = emitted_ops_dsa_pooled(131072, true, 1);
    full_dense.sort_unstable();
    full_pooled_1.sort_unstable();
    assert_eq!(
        full_dense, full_pooled_1,
        "full layer, index_kpool=1 vs dense"
    );

    let mut shared_dense = emitted_ops_dsa(131072, false);
    let mut shared_pooled_1 = emitted_ops_dsa_pooled(131072, false, 1);
    shared_dense.sort_unstable();
    shared_pooled_1.sort_unstable();
    assert_eq!(
        shared_dense, shared_pooled_1,
        "shared layer, index_kpool=1 vs dense"
    );
}

#[test]
fn glm_dsa_pooled_full_layer_emits_the_kpool_chain_in_order() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // ctx>CROSSOVER, 'full', index_kpool=4: the pooled indexer chain, not plain IndexScore/
    // IndexSelect — gate-score Gemv, iwp_f32 GemvF32, DsaPoolStash, DsaPoolCompress,
    // DsaQQuant, IndexScoreKpool, IndexSelect (now pool_size-aware), DsaPoolExpand, in
    // that relative order (not necessarily adjacent — MLA-side ops interleave).
    let ops = emitted_ops_dsa_pooled(131072, true, 4);
    assert!(
        !ops.contains(&(IndexScore as u16)),
        "pooled full layer does not use plain IndexScore"
    );
    assert!(
        ops.contains(&(IndexScoreKpool as u16)),
        "pooled full layer scores via the fp8 pooled variant"
    );
    assert!(
        ops.contains(&(IndexSelect as u16)),
        "pooled full layer still uses IndexSelect (now pool_size-aware)"
    );
    assert!(
        ops.contains(&(DsaPoolStash as u16)),
        "pooled full layer stashes into the decode ring"
    );
    assert!(
        ops.contains(&(DsaPoolCompress as u16)),
        "pooled full layer compresses at pool boundaries"
    );
    assert!(
        ops.contains(&(DsaQQuant as u16)),
        "pooled full layer quantizes q_idx"
    );
    assert!(
        ops.contains(&(DsaPoolExpand as u16)),
        "pooled full layer expands pool ids back to token ids"
    );
    assert!(
        ops.contains(&(GemvF32 as u16)),
        "pooled full layer projects weights_proj at fp32"
    );
    assert!(ops.contains(&(FlashGatherDecode as u16)), "gather flash");

    let idx_of = |op: DevOp| ops.iter().position(|&o| o == op as u16);
    let (i_stash, i_compress, i_qquant, i_score, i_select, i_expand) = (
        idx_of(DsaPoolStash).expect("stash present"),
        idx_of(DsaPoolCompress).expect("compress present"),
        idx_of(DsaQQuant).expect("qquant present"),
        idx_of(IndexScoreKpool).expect("score present"),
        idx_of(IndexSelect).expect("select present"),
        idx_of(DsaPoolExpand).expect("expand present"),
    );
    assert!(
        i_stash < i_compress,
        "stash before compress (compress reads the ring)"
    );
    assert!(
        i_compress < i_score && i_qquant < i_score,
        "compress and q-quant both feed the score"
    );
    assert!(i_score < i_select, "score before select");
    assert!(i_select < i_expand, "select before expand");
}

#[test]
fn glm_dsa_pooled_selection_width_matches_decode_and_prefill_geometry() {
    let _guard = crate::test_env::env_guard();
    const CTX: u32 = 131072;
    const POOL: u32 = 4;
    const WIDTH: u32 = 2051;

    let decode = emitted_dsa_pooled(CTX, true, POOL);
    let gather = decode
        .insts
        .iter()
        .find(|d| d.op == DevOp::FlashGatherDecode as u16)
        .expect("decode gather");
    let expand = decode
        .insts
        .iter()
        .find(|d| d.op == DevOp::DsaPoolExpand as u16)
        .expect("decode pool expand");
    assert_eq!(expand.i[1] * expand.i[2] + expand.i[2] - 1, WIDTH);
    assert_eq!(
        gather.i[6], WIDTH,
        "decode gather must consume the appended tail"
    );
    let iidx = decode
        .tensors
        .iter()
        .find(|t| t.name == "act.iidx")
        .expect("decode index tensor");
    assert_eq!(iidx.bytes, WIDTH as u64 * I32);

    let mut c = glm_ref_cfg();
    c.index_kpool = POOL;
    assert_eq!(glm_dsa_select_width(&c, CTX), WIDTH);
    assert_eq!(glm_dsa_pf_cap(&c, CTX), GLM_DSA_PF_PACK * WIDTH);
}

#[test]
fn glm_dsa_pooled_shared_layer_reuses_the_pool_cache() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // 'shared' pooled layers reuse the last full layer's selection — no indexer ops of
    // ANY kind (dense or pooled), same as the dense-indexer shared-layer case.
    let ops = emitted_ops_dsa_pooled(131072, false, 4);
    for op in [
        IndexScore,
        IndexScoreKpool,
        DsaPoolStash,
        DsaPoolCompress,
        DsaQQuant,
        DsaPoolExpand,
    ] {
        assert!(
            !ops.contains(&(op as u16)),
            "shared pooled layer emits no {op:?}"
        );
    }
    assert!(
        ops.contains(&(FlashGatherDecode as u16)),
        "shared pooled layer still gathers"
    );
}

#[test]
fn glm_dense_block_sequence() {
    let _guard = crate::test_env::env_guard();
    use DevOp::*;
    // Fused MLA (A+G): the 3 input GEMVs (q_a/kv_a/k_rope) -> one GemvQkv, and Wqa+Wqr -> one GemvQkv.
    let mla = vec![
        RmsNorm,
        GemvQkv,
        RmsNorm,
        GemvQkv,
        HeadNormRope,
        RmsNorm,
        HeadNormRope,
        FlashMlaDecode,
        MlaMergeFold,
        Gemv,
        Residual,
        RmsNorm,
    ];
    let mut want: Vec<u16> = mla.into_iter().map(|o| o as u16).collect();
    want.extend([DenseGluFp8Blk as u16, GemvFp8Blk as u16, Residual as u16]);
    assert_eq!(emitted_dense_ops(), want, "dense block op sequence");
    assert_eq!(emitted_dense_ops().len(), 15);
}

#[test]
fn glm_block_op_count() {
    let _guard = crate::test_env::env_guard();
    // 16 attention/pre-MoE ops after the A/G fusion (input q_a/kv_a/k_rope -> 1 GemvQkv, Wqa/Wqr
    // -> 1 GemvQkv; 2 dynamic-rope HeadNormRope + the 2-op router split + fused MlaMergeFold)
    // + 8*(glu+down) + 1 combine = 33 (was 36 pre-fusion).
    assert_eq!(emitted_ops(false).len(), 33);
}

// --- `--block` extraction path (M2, glm_build_block) ---------------------------
// These exercise the actual single-block emit + descriptor build on the CPU with the
// synthetic ref cfg (no checkpoint, no GPU): the block path must add NOTHING beyond
// the validated per-layer block (no embed/tail), and the descriptor must reflect the
// DSA IndexShare role + carried state.

fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>) -> Vec<u16> {
    let (m, _desc) = glm_build_block(c, ctx, 256, block, true, "glm-ref", MlaArch::Glm);
    m.progs[0].insts.iter().map(|d| d.op).collect()
}

/// A single MoE-layer `--block 3` extraction emits EXACTLY the validated MoE block
/// op sequence — no embed, no final-norm/lm_head/argmax tail. This is the numeric
/// coverage lever: the block inherits glm_block_matches_reference_*'s GPU parity.
#[test]
fn glm_block_extract_matches_reference() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg();
    assert_eq!(
        block_ops(&c, 512, 3..4),
        ref_sequence(true),
        "single-block --block 3 op sequence != validated MoE block"
    );
}

/// The weight namespace is a CFG PROPERTY, and it is the only thing a wrapper prefix moves.
///
/// Kimi-K3's tower is `language_model.model.layers.{L}.…` — 497 052 of its 497 220 tensors,
/// and NOT ONE under `model.`. Two properties are asserted, both against `c.prefix` rather
/// than against a spelled-out string, so changing the prefix cannot leave the test agreeing
/// with itself for the wrong reason:
///
///  * every checkpoint-bound tensor moves with the prefix, and
///  * no compiler-owned tensor moves at all (`kv.`/`act.`/`in.` are plow's, not the model's).
///
/// Third property, and the one that guards the shipping models: switching the prefix changes
/// only the SPELLING, so the two declarations are the same tensors in the same order with the
/// same byte sizes.
#[test]
fn the_weight_prefix_is_cfg_data_and_moves_only_the_weights() {
    let _guard = crate::test_env::env_guard();
    let decl = |pfx: &str| {
        let mut c = glm_ref_cfg();
        c.prefix = pfx.to_string();
        let mut b = Builder::new(256);
        let _ = declare_glm(&mut b, &c, 512, &[3]);
        b.tensors()
            .iter()
            .map(|t| (t.name.clone(), t.bytes))
            .collect::<Vec<_>>()
    };
    let flat = decl("model.");
    let nested = decl("language_model.model.");
    assert_eq!(
        flat.len(),
        nested.len(),
        "a prefix must not add or drop tensors"
    );

    let (mut moved, mut fixed) = (0usize, 0usize);
    for ((fname, fbytes), (nname, nbytes)) in flat.iter().zip(nested.iter()) {
        assert_eq!(
            fbytes, nbytes,
            "{fname}: a prefix must not change a byte size"
        );
        match fname.strip_prefix("model.") {
            // Anything the checkpoint names — weights AND the pointer tables declared beside
            // them, which `bind_packed_experts` resolves by that same prefix.
            Some(tail) => {
                assert_eq!(
                    *nname,
                    format!("language_model.model.{tail}"),
                    "a name under the model prefix did not follow the cfg prefix"
                );
                moved += 1;
            }
            // `lm_head.weight` and every compiler-owned tensor are outside the prefix by
            // construction and must NOT move: `kv.3.krot` is plow's, not the model's.
            None => {
                assert_eq!(fname, nname, "{fname} is not the checkpoint's to rename");
                fixed += 1;
            }
        }
    }
    // Both sides non-trivial, so neither arm can pass by being empty.
    assert!(moved >= 15 && fixed >= 5, "moved {moved}, fixed {fixed}");

    // And the compiler-owned namespaces really are compiler-owned: none of them is ever
    // demanded of a checkpoint, under either spelling.
    for (n, _) in flat.iter().chain(nested.iter()) {
        if packet::names::is_runtime_tensor(n) {
            assert!(!packet::names::is_checkpoint_weight(n), "{n}");
        }
    }
}

/// The loaders' weight predicate must be a SUPERSET of the prefix allowlist it replaced —
/// otherwise a shipping model would stop binding something it used to bind.
///
/// The old rule was `starts_with("model.") || starts_with("fp8/")` (`exec/gpu.rs`,
/// `serve/manager.rs`) plus `|| starts_with("lm_head")` on the AMD loader only. Asserted over
/// a real GLM declaration, so it covers the expert POINTER tables — the one family that looks
/// like a weight, lives under the model prefix, and must NOT be demanded of a checkpoint.
#[test]
fn the_new_weight_predicate_binds_everything_the_old_one_did() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg();
    let mut b = Builder::new(256);
    let _ = declare_glm(&mut b, &c, 512, &[3]);
    let (mut n_old, mut n_tables) = (0usize, 0usize);
    for t in b.tensors() {
        let n = t.name.as_str();
        let old = n.starts_with("model.") || n.starts_with("fp8/") || n.starts_with("lm_head");
        let table = packet::names::is_host_filled_table(n);
        if table {
            n_tables += 1;
            assert!(
                !packet::names::is_checkpoint_weight(n),
                "{n}: host-filled, not a weight"
            );
            continue;
        }
        if old {
            n_old += 1;
            assert!(
                packet::names::is_checkpoint_weight(n),
                "{n} used to bind from the checkpoint and no longer would"
            );
        }
    }
    assert!(
        n_old >= 15 && n_tables > 0,
        "weights {n_old}, tables {n_tables}"
    );
}

/// A multi-layer `--block 2..4` extraction is the per-layer blocks concatenated
/// (dense layer 2 then MoE layer 3), and the residual ping-pong lands the output in
/// `act.x` after an even layer count.
#[test]
fn glm_block_extract_multi_layer_chains() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg();
    let mut want = emitted_dense_ops(); // layer 2 (dense)
    want.extend(ref_sequence(true)); // layer 3 (MoE)
    assert_eq!(
        block_ops(&c, 512, 2..4),
        want,
        "2-layer block != dense++moe"
    );
    let (_, desc) = glm_build_block(&c, 512, 256, 2..4, true, "glm-ref", MlaArch::Glm);
    assert_eq!(
        desc.outputs[0].name, "act.x",
        "even layer count -> act.x out"
    );
    assert_eq!(desc.layer, 2, "descriptor.layer = block start");
}

/// Descriptor for a single MoE block: arch/kind/dims + `act.xnext` output (odd
/// layer count) + kv carried state, DSA gate OFF at this ctx (no dsa_indices).
#[test]
fn glm_block_descriptor_moe() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg(); // indexer_full[3] = false (reuse)
    let (_, d) = glm_build_block(&c, 512, 256, 3..4, true, "glm-ref", MlaArch::Glm);
    assert_eq!(d.arch, "glm_mla_dsa");
    assert_eq!(d.kind, vec!["mla_dsa", "moe_ffn"]);
    assert_eq!(d.dtype, "fp8");
    assert_eq!(d.dims.kv_lora, Some(512));
    assert_eq!(d.dims.q_lora, Some(2048));
    assert_eq!(d.dims.n_exp, Some(256));
    assert_eq!(d.dims.top_k, Some(8));
    assert_eq!(d.dims.shared_exp, Some(1));
    assert_eq!(d.dims.moe_inter, Some(2048));
    assert_eq!(d.dims.index_topk, Some(2048));
    assert_eq!(
        d.outputs[0].name, "act.xnext",
        "odd layer count -> act.xnext"
    );
    assert_eq!(d.weights.prefix, "model.layers.3.");
    // Prefill is OPT-IN (`PLOW_MLA_PREFILL`), so the default block emit is still decode-only —
    // and must stay so, or every existing GLM asset gains buckets whose FFN half does not exist.
    assert!(
        d.programs.prefill_buckets.is_empty(),
        "GLM block emit is decode-only unless prefill buckets are requested"
    );
    // DSA gate off (ctx <= CROSSOVER): reuse role, but NO dsa_indices carried.
    assert_eq!(d.dsa_role.as_deref(), Some("reuse"));
    assert_eq!(d.carried_state.len(), 1);
    assert_eq!(d.carried_state[0].role, "kv");
    assert_eq!(d.carried_state[0].tensors, vec!["kv.3.ckv", "kv.3.krot"]);
}

/// Descriptor for a DENSE block (`--block 0`): no MoE dims, dense_ffn kind.
#[test]
fn glm_block_descriptor_dense() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg();
    let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "glm-ref", MlaArch::Glm);
    assert_eq!(d.kind, vec!["mla_dsa", "dense_ffn"]);
    assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
    assert_eq!(d.dims.moe_inter, None);
    assert_eq!(d.dims.kv_lora, Some(512), "MLA dims still present");
}

/// IndexShare (§7): under an ARMED DSA gate (ctx > CROSSOVER=65536), a 'reuse'
/// layer carries `dsa_indices` in (it does not recompute the top-k), while an
/// 'indexer' layer computes them in-block (kv carries its kidx cache instead).
#[test]
fn glm_block_dsa_indexshare_carried_state() {
    let _guard = crate::test_env::env_guard();
    // 'reuse' layer 3 (indexer_types[3] = shared).
    let mut c = glm_ref_cfg();
    c.indexer_full = vec![false, false, false, false];
    let (_, reuse) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
    assert_eq!(reuse.dsa_role.as_deref(), Some("reuse"));
    let dsa = reuse
        .carried_state
        .iter()
        .find(|s| s.role == "dsa_indices")
        .expect("reuse layer carries dsa_indices");
    assert_eq!(dsa.tensors, vec!["act.iidx"]);

    // 'indexer' layer 3 (indexer_types[3] = full): computes indices in-block, so
    // no dsa_indices carry; its kidx key cache joins the kv carried state.
    c.indexer_full = vec![false, false, false, true];
    let (_, idx) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
    assert_eq!(idx.dsa_role.as_deref(), Some("indexer"));
    assert!(
        idx.carried_state.iter().all(|s| s.role != "dsa_indices"),
        "indexer layer does not carry dsa_indices in"
    );
    assert!(
        idx.carried_state[0]
            .tensors
            .contains(&"kv.3.kidx".to_string()),
        "indexer layer carries its kidx cache"
    );
}

/// `MlaMergeFold` is sized to its OWN work-item count, never to `n_cu`, and the narrowing must
/// leave the interpreter's VT branch exactly where it found it — a different VT is a different
/// fold map and therefore different arithmetic (`exec_mla_merge_fold`, op_attention.h).
#[test]
fn mla_fold_is_sized_to_its_work_items_and_never_flips_vt() {
    // Sets PLOW_GLM_GF / PLOW_GLM_WGFIT below, which `glm_gf`/`wgfit` read
    // live — hold the lock so no sibling test emits under them.
    let _env = crate::test_env::env_guard();
    let all: Vec<u32> = (0..256u32).collect();
    // GLM-5.2, v_head 256: bh*ceil(256/32) = bh*8, and bh*8 <= nblk keeps VT at 32.
    for &(nh_l, want) in &[(16u32, 128usize), (8, 64), (4, 32), (32, 256)] {
        let got = mla_fold_cus(&all, nh_l, 256);
        assert_eq!(got.len(), want, "GLM nh_l={nh_l}");
        assert_eq!(got[0], 0, "the narrowing keeps slice 0 == workgroup 0");
        // The width IS the work-item count, so no workgroup is left without an item.
        let vt = mla_fold_vt(nh_l, got.len() as u32, 256);
        assert_eq!(vt, mla_fold_vt(nh_l, 256, 256), "VT branch must not move");
        assert_eq!(
            nh_l * 256u32.div_ceil(vt),
            got.len() as u32,
            "sized to n_work"
        );
    }
    // Kimi-K3, v_head 128: bh=16 would pick VT=32 at 256 wgs and VT=128 at the narrowed 64,
    // which reassociates the fold. The rule must REFUSE rather than narrow.
    assert_eq!(
        mla_fold_cus(&all, 16, 128).len(),
        256,
        "v=128 narrowing flips VT — refuse"
    );
    // ... but a bh too large for VT=32 in the first place narrows safely.
    assert_eq!(
        mla_fold_cus(&all, 96, 128).len(),
        96,
        "v=128, VT=256 both sides"
    );
    // Prefill (n_batch = t folded into bh) hands the whole machine back.
    assert_eq!(
        mla_fold_cus(&all, 128 * 16, 256).len(),
        256,
        "prefill bucket is inert"
    );
    // The flash-decode rule cancels EXACTLY at GF=4 — `glm_nsplit`'s fill cap uses the same
    // `GLM_MLA_GF`, so `(nh_l/4)*fill == n_cu` and the long-ctx blob is chip-wide. THIS IS THE
    // REGRESSION GUARD FOR THE HALF-WIDTH DEFECT: while `glm_gf` returned 8 against an object
    // built without `-DPLOW_GLM_GF8_ARM=1`, these packets carried 128 workgroups for 256 work
    // items — correct output (the body grid-strides) at half the parallelism, worth a measured
    // -3.35 ms/token at ctx 32768 end-to-end. If this ever reads 128 again, either `glm_gf`
    // went back to 8 or `flash_mla_cus` stopped agreeing with the body.
    // tp4 (nh_l=16) and tp2 (nh_l=32) land exactly on the chip. tp8 (nh_l=8) does NOT, and
    // that is pre-existing and deliberate: its `fill` is 128, but `NS_CEIL_MEASURED` holds
    // nsplit at 64 because the ladder behind the ceiling is a tp4 ladder. tp8's long-ctx flash
    // therefore runs 2*64 = 128 items on 256 CUs — HALF THE CHIP — and whether raising it pays
    // is an open, measurable question, not an assumption to bake in. See `glm_nsplit`.
    for &(nh_l, ctx, want) in &[
        (16u32, 65536u32, 256usize),
        (32, 65536, 256),
        (8, 65536, 128),
    ] {
        let got = flash_mla_cus(&all, 1, 1, nh_l, glm_gf(ctx, nh_l), glm_nsplit(ctx, nh_l));
        assert_eq!(got.len(), want, "nh_l={nh_l} ctx={ctx}: GF=4 work items");
    }
    // The GF=8 arm, when someone builds it (-DPLOW_GLM_GF8_ARM=1) and pins PLOW_GLM_GF=8,
    // halves the work items and needs 2x nsplit to be chip-wide again — at 2x the merge
    // inputs, and the merge is a function of nsplit ALONE (measured: gf4/ns64 26.5 us vs
    // gf8/ns64 26.5; gf4/ns128 41.7 vs gf8/ns128 47.2). That is why matching WORK ITEMS across
    // GF is not a fair trade and GF=8 lost the matched-item A/B at both ctx.
    assert_eq!(
        flash_mla_cus(&all, 1, 1, 16, 8, glm_nsplit(65536, 16)).len(),
        128,
        "GF=8 at the GF=4 nsplit is half the chip"
    );
    assert_eq!(
        flash_mla_cus(&all, 1, 1, 16, 8, 2 * glm_nsplit(65536, 16)).len(),
        256,
        "GF=8 at 2x nsplit restores full fill"
    );
    assert_eq!(
        flash_mla_cus(&all, 1, 1, 16, glm_gf(1024, 16), glm_nsplit(1024, 16)).len(),
        128,
        "max_ctx 1024 is GF=2 with a GF=4-sized nsplit: 128 items, not 256"
    );
    // i[7] is read LITERALLY by `flash_mla_cus`, so the value the emitter bakes MUST be the
    // one the interpreter instantiates — and with `PLOW_GLM_GF8_ARM=0` (the default) that set
    // is {2,4}. This pair is the invariant: the emitted GF and the dispatch width agree.
    assert_eq!(
        glm_gf(65536, 16),
        4,
        "long ctx bakes the GF the default object runs"
    );
    assert_eq!(
        flash_mla_cus(&all, 1, 1, 16, 4, 64).len(),
        256,
        "GF=4 => nh_l/4 groups"
    );
    // `n_grp = nh_l / GF` is integer: a GF larger than this rank's head shard makes the flash
    // do NOTHING. GLM-5.2 n_head=64 reaches nh_l=4 at tp16, so the clamp is live, not
    // hypothetical.
    assert_eq!(glm_gf(65536, 8), 4, "tp8 (nh_l=8): 4, the default arm");
    assert_eq!(
        glm_gf(65536, 4),
        4,
        "tp16 (nh_l=4) must clamp to 4, not divide to zero"
    );
    assert_eq!(glm_gf(65536, 2), 2, "nh_l=2 clamps all the way to 2");
    // `PLOW_GLM_GF=8` still reaches the arm — it is the only way to run its A/B — and it is
    // still clamped by divisibility. Asserted at the end of this test, where the env var is
    // set (§6g-GF8: a pinned 8 on a tp16 blob would otherwise emit all-zero attention).
    // GF MUST DIVIDE nh_l, NOT MERELY FIT IN IT. `n_grp = nh_l / GF` truncates and the only
    // head cursor is `h0 = hg*GF`, so the `nh_l % GF` tail is never visited and its opart /
    // mlpart rows are read back by the merge uninitialised. Kimi-K3 is the first model in the
    // tree with a non-power-of-two head count (96), and it is the reference TP that breaks:
    //   tp8  -> nh_l=12: the old `g <= nh_l` rule took GF=8, n_grp=1, heads 8..11 DROPPED
    //   tp16 -> nh_l=6 : took GF=4, n_grp=1, heads 4..5 DROPPED
    assert_eq!(
        glm_gf(65536, 12),
        4,
        "K3 tp8 (nh_l=12): 8 does not divide 12, 4 does"
    );
    assert_eq!(
        glm_gf(65536, 6),
        2,
        "K3 tp16 (nh_l=6): 4 does not divide 6, 2 does"
    );
    assert_eq!(
        glm_gf(65536, 24),
        4,
        "K3 tp4 (nh_l=24): 4, the default arm (8 is pin-only)"
    );
    assert_eq!(glm_gf_prefill(65536, 12), 4, "prefill twin: 12 % 4 == 0");
    assert_eq!(
        glm_gf_prefill(65536, 6),
        2,
        "prefill twin: 6 % 4 != 0, fall to 2"
    );
    // Every nh_l a shipping model produces is a power of two, and every power of two is
    // divisible by 8, 4 and 2 — so this change moves NO emitted packet for GLM-5.2, Kimi-K2.7
    // or DeepSeek-V3. Pin that, because "it is byte-identical" is the claim that makes this
    // safe to land without re-validating those blobs.
    for nh_l in [2u32, 4, 8, 16, 32, 64, 128] {
        for ctx in [1024u32, 65536] {
            assert_eq!(
                glm_gf(ctx, nh_l),
                [8u32, 4, 2]
                    .into_iter()
                    .find(|&g| g <= if ctx <= GLM_GF_CROSSOVER { 2 } else { 4 } && g <= nh_l)
                    .unwrap_or(2),
                "power-of-two nh_l={nh_l} ctx={ctx} must be unchanged by the divisibility rule"
            );
        }
    }
    // The pin is clamped too: a sweep must not be able to emit all-zero attention.
    {
        let _p = crate::test_env::EnvScope::set(&[("PLOW_GLM_GF", "8")]);
        assert_eq!(glm_gf(1024, 16), 8, "the pin overrides the crossover");
        assert_eq!(glm_gf(1024, 4), 4, "the pin is still clamped by nh_l");
    }
    // The knob restores the control arm exactly. Scoped, so a failing
    // assert restores the knob on unwind instead of leaking it.
    {
        let _p = crate::test_env::EnvScope::set(&[("PLOW_GLM_WGFIT", "0")]);
        assert_eq!(mla_fold_cus(&all, 16, 256).len(), 256);
        assert_eq!(blocked_gemv_cus(&all, 2624).len(), 256);
    }
}

/// An ODD head shard cannot be expressed by ANY instantiated GF, and the emit must say so.
///
/// The interpreter instantiates GF in {2,4,8}; none divides an odd `nh_l`, so there is no
/// correct packet to emit and the only honest outcome is a refusal at compile time. The
/// runtime cannot catch this — unvisited heads are not an error condition anywhere, they are
/// memory nobody wrote that the merge consumes as if it were a partial. Reachable on Kimi-K3
/// (96 heads) at tp32: 96/32 = 3.
#[test]
#[should_panic(expected = "does not divide this rank's head shard")]
fn an_odd_head_shard_is_refused_rather_than_silently_truncated() {
    // Reads PLOW_GLM_GF live: a sibling test's pin decides which branch
    // this reaches, so the refusal it asserts must not be a race.
    let _env = crate::test_env::env_guard();
    glm_gf(65536, 3);
}

/// A GV_BLOCKED gemv packet owns columns in runs of `per = ceil(n/nblk)`; the narrowing drops
/// only the ceiling tail that owns none, and it is a FIXED POINT of that arithmetic, so every
/// surviving workgroup's column run is byte-for-byte the one it had before.
#[test]
fn blocked_gemv_drops_only_the_empty_ceiling_tail() {
    // `blocked_gemv_cus` early-returns the un-narrowed list under
    // PLOW_GLM_WGFIT=0, which a sibling test sets.
    let _env = crate::test_env::env_guard();
    let all: Vec<u32> = (0..256u32).collect();
    for &n in &[1u32, 63, 255, 256, 257, 512, 2624, 6144, 9216, 154880] {
        let got = blocked_gemv_cus(&all, n);
        let per = n.div_ceil(256);
        let per_after = n.div_ceil(got.len() as u32);
        assert_eq!(
            per_after, per,
            "n={n}: `per` moved, the column map is not preserved"
        );
        // every surviving workgroup owns at least one column ...
        assert!(
            (got.len() as u32 - 1) * per < n,
            "n={n}: kept an empty workgroup"
        );
        // ... and no column is dropped.
        assert!(got.len() as u32 * per >= n, "n={n}: dropped columns");
    }
    // GLM-5.2 TP4: fusion A is 2048+512+64 over 256 workgroups.
    assert_eq!(blocked_gemv_cus(&all, 2048 + 512 + 64).len(), 239);
    // fusion G (16*512 + 16*64) already divides evenly.
    assert_eq!(blocked_gemv_cus(&all, 16 * 512 + 16 * 64).len(), 256);
}

#[test]
fn glm_decode_gemv_tuning_preserves_live_column_ownership() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_WGFIT", "1"),
        ("PLOW_GEMV_WG_TUNING", ""),
        ("PLOW_GLM_DSA", "1"),
        ("PLOW_GLM_FUSE_ROPE", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    let all: Vec<u32> = (0..304).collect();
    let mut declarations = Builder::new(304);
    let n = declare_glm_rows_batched(&mut declarations, &c, 81920, &[3], 8192, 8, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    for rows in [1, 2, 4, 8] {
        let emit = || {
            let mut b = Builder::new(304);
            b.adopt_tensors(tensors.clone());
            emit_glm_block(
                &mut b,
                &c,
                &n,
                0,
                81920,
                rows,
                8,
                MoeEnc::Fp8Blk,
                n.x,
                n.xnext,
                &[],
                &mut 0,
                &all,
            );
            b.finish()
        };
        let baseline = emit();
        let _tuning =
            crate::test_env::EnvScope::set(&[("PLOW_GEMV_WG_TUNING", "64x6144=304,256x6144=304")]);
        let tuned = emit();
        assert_eq!(baseline.insts.len(), tuned.insts.len());
        let mut changed = 0;
        for (before, after) in baseline.insts.iter().zip(&tuned.insts) {
            assert_eq!(
                (before.op, before.t, before.i, before.j, before.f),
                (after.op, after.t, after.i, after.j, after.f)
            );
            if before.blocks != after.blocks {
                changed += 1;
                assert_eq!(after.op, DevOp::Gemv as u16);
                assert!(matches!(after.i[1], 64 | 256));
                assert_eq!(after.i[2], 6144);
                let n = after.i[1];
                let per = n.div_ceil(u32::from(before.blocks));
                assert_eq!(n.div_ceil(u32::from(after.blocks)), per);
                for column in 0..n {
                    assert!(column / per < u32::from(after.blocks));
                }
            }
        }
        if rows > 1 {
            assert!(changed > 0, "B{rows} tuning was ignored");
        }
    }
    let all: Vec<_> = (0..304).rev().collect();
    let _tuning = crate::test_env::EnvScope::set(&[("PLOW_GEMV_WG_TUNING", "64x6144=304")]);
    assert_eq!(glm_decode_gemv_cus(&all, DevOp::Gemv, 64, 6144), all[..64]);
    for op in [DevOp::GemvMxfp4, DevOp::GemvFp8Blk, DevOp::GemvQkv] {
        assert_eq!(glm_decode_gemv_cus(&all, op, 64, 6144), all);
    }
    assert_eq!(glm_decode_gemv_cus(&all, DevOp::Gemv, 64, 2048), all);
    let _disabled = crate::test_env::EnvScope::set(&[("PLOW_GLM_WGFIT", "0")]);
    assert_eq!(glm_decode_gemv_cus(&all, DevOp::Gemv, 64, 6144), all);
}

#[test]
fn glm_decode_norm_rows_preserves_arithmetic_and_completion() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_DECODE_NORM_ROWS", "0"),
        ("PLOW_GLM_FUSE_B1", "1"),
        ("PLOW_GLM_FUSE_SEAM", "1"),
        ("PLOW_GLM_FUSE_QNORM", "0"),
        ("GLM_FUSE_XRN", "0"),
        ("PLOW_GLM_MOE_RESIDENT", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    c.layers = 5;
    c.indexer_full.push(false);
    let layers: Vec<_> = (0..c.layers).collect();
    let mut declarations = Builder::new(304);
    let n = declare_glm_rows_batched(&mut declarations, &c, 64, &layers, 128, 20, MoeEnc::Fp8Blk);
    let tensors = declarations.tensors();
    for rows in [1, 2, 3, 4, 8, 16, 20] {
        let emit = || {
            let mut b = Builder::new(304);
            b.adopt_tensors(tensors.clone());
            let all = b.all();
            let mut dep = Vec::new();
            let mut xgate = 0;
            let mut cur = n.x;
            for (slot, &layer) in layers.iter().enumerate() {
                let next = if cur == n.x { n.xnext } else { n.x };
                let emit_block = if c.is_dense(layer) {
                    emit_glm_dense_block
                } else {
                    emit_glm_block
                };
                dep = vec![emit_block(
                    &mut b,
                    &c,
                    &n,
                    slot,
                    64,
                    rows,
                    20,
                    MoeEnc::Fp8Blk,
                    cur,
                    next,
                    &dep,
                    &mut xgate,
                    &all,
                )];
                cur = next;
            }
            b.finish()
        };
        let baseline = emit();
        let _wide = crate::test_env::EnvScope::set(&[("PLOW_GLM_DECODE_NORM_ROWS", "1")]);
        let wide = emit();
        assert_eq!(baseline.insts.len(), wide.insts.len());
        let mut changed = 0;
        for (ix, (before, after)) in baseline.insts.iter().zip(&wide.insts).enumerate() {
            assert_eq!(
                (before.op, before.t, before.i, before.j, before.f),
                (after.op, after.t, after.i, after.j, after.f)
            );
            if before.blocks != after.blocks {
                changed += 1;
                assert!(matches!(
                    DevOp::from_u16(after.op),
                    Some(DevOp::RmsNorm | DevOp::AddNorm)
                ));
                assert_eq!(after.blocks as u32, rows);
                let slices: std::collections::BTreeSet<_> = wide
                    .stream
                    .iter()
                    .filter(|e| e.inst as usize == ix)
                    .map(|e| e.slice)
                    .collect();
                assert_eq!(slices, (0..rows).collect());
                for wait in wide.waits.iter().filter(|w| w.id as usize == ix) {
                    assert_eq!(wait.threshold, rows);
                }
            }
        }
        if rows == 1 {
            assert_eq!(baseline.to_blob(), wide.to_blob());
        } else {
            assert_eq!(changed, 15);
        }
    }
}

/// The MLA flash-decode split factor is the ctx-scaled cost optimum, capped by the ACTUAL
/// per-rank chip-fill `fill = ceil(n_cu / (nh_l/GF))` and the KV-tile count. `glm_nsplit` takes
/// nh_l (= n_head/tp) so the cap is correct under TP/EP — the pre-fix bug sized it from the
/// global n_head=64, pinning the cap to tp=1's fill regardless of TP. Asserts the caps and the
/// measured (MI350X mla_perf) chain optima: ns~16 up to 8k, ns~64 at 32k.
#[test]
fn glm_nsplit_is_ctx_scaled_and_capped_per_rank() {
    let _guard = crate::test_env::env_guard();
    let n_cu = 256u32;
    for &(_tp, nh_l) in &[(1u32, 64u32), (2, 32), (4, 16), (8, 8)] {
        let n_grp = (nh_l / GLM_MLA_GF).max(1);
        let fill = (n_cu + n_grp - 1) / n_grp;
        let mut prev = 0u32;
        for &ctx in &[1024u32, 4096, 8192, 16384, 32768, 65536, 131072] {
            let ns = glm_nsplit(ctx, nh_l);
            let kv_tiles = ctx.div_ceil(32);
            // Cap 1 — never over-split past the chip (the nh_l-aware fill).
            assert!(
                ns <= fill,
                "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds chip-fill {fill}"
            );
            // Cap 2 — never split finer than there are KV tiles (no empty splits).
            assert!(
                ns <= kv_tiles,
                "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds {kv_tiles} KV tiles"
            );
            // Monotone non-decreasing in ctx (more latent => more useful splits).
            assert!(
                ns >= prev,
                "nh_l={nh_l} ctx={ctx}: ns={ns} < prev {prev} (not ctx-monotone)"
            );
            prev = ns;
        }
    }
    // MEASURED chain optima locked in, one assert per rung of the ladder in the header
    // (GLM-5.2 TP4, arm-absent object, per-layer chain us; the whole table is there). These
    // are the rows this rule exists to reproduce, so they are pinned individually rather than
    // as "ns grows with ctx" — the previous constant satisfied that and still missed two.
    for &nh_l in &[8u32, 16] {
        for &(ctx, want, why) in &[
            (1024u32, 16u32, "61.1 vs ns32's 65.8"),
            (4096, 16, "66.6 vs ns32's 67.2 — still the floor's rung"),
            (8192, 32, "73.3 vs ns16's 90.1 — the rung ctx/512 got WRONG"),
            (
                16384,
                64,
                "88.1 vs ns32's 103.7 — the other rung ctx/512 got wrong",
            ),
            (32768, 64, "135.9, and 128 regresses to 141.3"),
            (65536, 64, "183.0, fill-capped anyway"),
        ] {
            assert_eq!(
                glm_nsplit(ctx, nh_l),
                want,
                "nh_l={nh_l} ctx={ctx}: measured optimum is ns={want} ({why})"
            );
        }
    }
    // tp=1 is chip-full at ns=16 (n_grp=16), so the fill cap pins every ctx to 16 — byte-identical
    // to the pre-fix path (no regression on single-GPU decode).
    for &ctx in &[1024u32, 8192, 32768, 131072] {
        assert_eq!(
            glm_nsplit(ctx, 64),
            16,
            "tp=1 ctx={ctx}: fill-capped to 16 (unchanged)"
        );
    }
    // The refined rule must NOT full-fill mid ctx (the measured 8k regression at ns=128): at tp=8
    // 8k it stays at the floor, not fill=128.
    assert!(
        glm_nsplit(8192, 8) < ((256 + 1) / 2),
        "tp=8 8k must not full-fill (mid-ctx merge regression)"
    );
}

#[test]
fn glm_cfg_qk_scale() {
    let _guard = crate::test_env::env_guard();
    let c = glm_ref_cfg();
    assert_eq!(c.qk_head(), 256);
    assert!(
        (c.attn_scale - 0.0625).abs() < 1e-6,
        "MLA scale = 1/sqrt(256)"
    );
    assert!(
        c.is_dense(0) && c.is_dense(2) && !c.is_dense(3),
        "first_k_dense_replace=3"
    );
}

/// `GLM_LINEAR_FP8` re-declares four tensors per layer at HALF their bf16 size, and the
/// PREFILL emitters have to be told. They were not, for three interpreters: `declare_glm_rows`
/// REFUSED a stacked emit (`require_lin_fp8_decode_only`) rather than put a bf16 `Gemm` on fp8
/// bytes, because no dense T-row block-fp8 GEMM existed. `GemmFp8Blk` (107) is that GEMM, so
/// what is pinned here is the ROUTE, not the refusal.
///
/// Pinned through `emit_pf_gemm_fp8_blk` rather than by setting `GLM_LINEAR_FP8` and calling
/// the emitters: the knob is process-global env state, cargo runs tests in parallel threads,
/// and a sibling test that counts tensors sees the four extra `weight_scale_inv` handles appear
/// under it. That is not hypothetical — it broke
/// `the_weight_prefix_is_cfg_data_and_moves_only_the_weights` (58 vs 54) when the old version of
/// this test set the var. Test the pure part as a pure function.
#[test]
fn glm_linear_fp8_prefill_routes_to_the_block_fp8_gemm() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(256);
    let w = b.tensor("w.weight_fp8", 6144 * 4096);
    let s = b.tensor("w.weight_scale_inv", 48 * 32 * F32);
    let x = b.tensor("act.x", 512 * 4096 * BF16);
    let o = b.tensor("act.o", 512 * 6144 * BF16);
    let all: Vec<u32> = (0..256u32).collect();
    emit_pf_gemm_fp8_blk(&mut b, &all, o, x, w, s, 512, 6144, 4096, &[]);
    let p = b.finish();
    assert_eq!(p.insts.len(), 1);
    let d = &p.insts[0];
    assert_eq!(
        d.op,
        DevOp::GemmFp8Blk as u16,
        "the prefill arm must be the block-fp8 GEMM, never a bf16 Gemm on fp8 bytes"
    );
    // The scale grid is NOT optional and must ride t[3]: a null there is a wrong number, not a
    // fault, because the kernel's promotion multiplies by whatever it reads.
    assert_eq!([d.t[0], d.t[1], d.t[2], d.t[3]], [o, x, w, s]);
    assert_eq!([d.i[0], d.i[1], d.i[2]], [512, 6144, 4096]);
}

/// A block-fp8 weight without its scale grid is a NULL pointer inside the kernel's promotion.
/// The two handles are declared as a pair; refuse rather than emit half of one.
#[test]
fn glm_linear_fp8_prefill_refuses_a_weight_with_no_scale_grid() {
    let _guard = crate::test_env::env_guard();
    let mut b = Builder::new(256);
    let w = b.tensor("w.weight_fp8", 64);
    let x = b.tensor("act.x", 64);
    let o = b.tensor("act.o", 64);
    let all: Vec<u32> = (0..256u32).collect();
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        emit_pf_gemm_fp8_blk(&mut b, &all, o, x, w, TENSOR_NONE, 8, 8, 8, &[]);
    }))
    .err()
    .expect("a scale-less block-fp8 GEMM must be refused, not emitted");
    let msg = e
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("weight_scale_inv"),
        "name the missing handle; got: {msg}"
    );
}

/// A synthetic GLM-5.3 config, trimmed to two all-KDA layers and dense FFN — the smallest
/// shape that still exercises `emit_kda_mixer_ex`'s per-call scratch-tensor declarations
/// (`x`, `q_raw`/`k_raw`/`v_raw`, ...), which is what the next test is about.
fn glm53_ref_cfg() -> GlmCfg {
    GlmCfg {
        layers: 2,
        hidden: 4096,
        heads: 64,
        kv_lora: 512,
        q_lora: 1536,
        qk_nope: 256,
        qk_rope: 0,
        v_head: 256,
        vocab: 154880,
        eps: 1e-6,
        n_exp: 1,
        top_k: 1,
        n_group: 1,
        topk_group: 1,
        moe_inter: 64,
        dense_inter: 512,
        first_k_dense: 2, // both layers dense-FFN — no MoE routing to set up for this test
        route_scale: 1.0,
        attn_scale: (512f32).powf(-0.5),
        rope_theta: None,
        rope_scale: packet::rope::RopeScale::None,
        prefix: "model.language_model.".into(),
        tp: 1,
        ep: false,
        group: false,
        index_heads: 32,
        index_dim: 128,
        index_topk: 2048,
        index_kpool: 1,
        indexer_full: vec![true, true],
        softmax_layers: vec![false, false], // both layers KDA, no MLA layer needed here
        has_dsa: false,
    }
}

/// **Regression for the EIGHTEENTH-PASS bug** (see `status.md`): `glm53_emit_full` used to
/// capture the model's `tensors` list ONCE, before any program was built, then reuse that
/// same pre-emission snapshot for every program AND for the final `Model`. But
/// `emit_kda_mixer_ex` declares its own fresh scratch tensors (`x`, `q_raw`/`k_raw`/`v_raw`,
/// ...) on the per-program `Builder` during emission — so those handles existed in the
/// program's own instruction stream but never made it into the blob's tensor table. On real
/// hardware this read past the end of the on-device tensor-pointer table and faulted with a
/// null address, misreported by the async fault handler as a crash in whatever op happened
/// to be dispatching next (`GemvQkv`) rather than the real culprit (`RmsNorm`, one op
/// earlier). The fix threads `tensors` forward from each finished program's OWN tensor list
/// (`prog.tensors`), mirroring `k3_build_model`'s identical pattern in `kimi_k3.rs` for the
/// same emitter. This test rebuilds `glm53_emit_full`'s tensor/program assembly in
/// miniature and asserts the invariant directly: every tensor handle any instruction
/// references must be within the final table's bounds.
#[test]
fn glm53_program_assembly_keeps_every_tensor_handle_in_the_final_table() {
    let _guard = crate::test_env::env_guard();
    let c = glm53_ref_cfg();
    let layers: Vec<u32> = (0..c.layers).collect();
    let enc = MoeEnc::from_flags(false, false);
    let mut tb = Builder::new(256);
    let n = declare_glm_rows_batched(&mut tb, &c, 64, &layers, 1, 1, enc);
    let s = declare_glm53(&mut tb, &c, 1, 1, n.pos);
    let mut tensors = tb.tensors();

    let mut b = Builder::new(256);
    b.set_tensor_dedup(true);
    b.adopt_tensors(tensors.clone());
    emit_glm53_program(&mut b, &c, &n, &s, 64, 1, 1, enc, false);
    let prog = b.finish();
    tensors = prog.tensors.clone();

    for inst in &prog.insts {
        for &h in &inst.t {
            assert!(
                h == TENSOR_NONE || (h as usize) < tensors.len(),
                "op {} references tensor handle {h}, but the final table has only {} entries",
                inst.op,
                tensors.len()
            );
        }
    }
}

#[test]
fn glm_placed_prefill_preserves_native_segment_boundaries() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", "full:128"),
        ("PLOW_GLM_PLACE_PF", "1"),
        ("PLOW_GLM_MOE_AITER", "1"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "0"),
    ]);
    let dir = std::env::temp_dir().join(format!("plow-glm-placed-prefill-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(|model| {
        let prefill = &model.progs[0];
        assert_eq!(prefill.l2_domains, 8);
        let native = prefill
            .insts
            .iter()
            .position(|d| d.op == DevOp::MoeAiterFp8Pf as u16)
            .unwrap();
        let segment = prefill
            .stream
            .iter()
            .find(|e| e.inst as usize == native)
            .unwrap()
            .seg;
        assert!(segment > 0);
        for entries in [&prefill.stream, &prefill.gq_stream] {
            assert!(entries
                .iter()
                .filter(|e| e.seg == segment)
                .all(|e| e.inst as usize == native));
            assert!(entries.iter().any(|e| e.seg > segment));
        }
        let segments = prefill
            .stream
            .iter()
            .map(|e| usize::from(e.seg) + 1)
            .max()
            .unwrap();
        assert_eq!(prefill.gq_seg_ofs.len(), segments * 8 + 1);
        Ok(crate::LeanReport::skipped("structural regression test"))
    });
    glm_emit_full(
        &dir,
        512,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        Some(packet::devbuild::L2Layout {
            sms: 38,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        }),
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn glm_flat_decode_preserves_xcd_native_segment_boundaries() {
    check_glm_flat_segments(false);
}

#[test]
fn glm_resident_moe_preserves_all_native_segment_boundaries() {
    check_glm_flat_segments(true);
}

fn check_glm_flat_segments(resident: bool) {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", if resident { "full:1,2,4,8,16,20,32,64,128,256,512,1024,2048,4096,8192" } else { "full:128" }),
        ("PLOW_GLM_PLACE_PF", "0"),
        ("PLOW_GLM_MOE_AITER", "0"),
        ("PLOW_GLM_MOE_FLAT_DECODE", if resident { "0" } else { "1" }),
        ("PLOW_GLM_MOE_RESIDENT", if resident { "1" } else { "0" }),
        (
            "PLOW_DECODE_BATCH_LADDER",
            if resident { "1,2,4,8,16,20" } else { "1,2,4,8" },
        ),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "0"),
    ]);
    let dir = std::env::temp_dir().join(format!("plow-glm-flat-decode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(move |model| {
        let mut checked = 0;
        let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
        assert_eq!(prefill_count, if resident { 15 } else { 1 });
        for (p, (prog, &rows)) in model.progs.iter().zip(&model.prog_t).enumerate() {
            let decode = p >= prefill_count;
            let native = prog
                .insts
                .iter()
                .enumerate()
                .filter(|(_, d)| d.op == DevOp::MoeAiterFp8Pf as u16)
                .collect::<Vec<_>>();
            if !resident && !matches!(rows, 2 | 4 | 8) {
                assert!(native.is_empty());
                continue;
            }
            checked += 1;
            if decode {
                assert_eq!(prog.l2_domains, 8);
            }
            assert_eq!(native.len(), 1);
            let (ix, inst) = native[0];
            if !decode {
                assert_eq!(inst.i, [rows, 6144, 256, 256, 8, 64, 2, 1]);
                let combine = prog.insts[ix + 1..]
                    .iter()
                    .find(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == inst.t[0])
                    .unwrap();
                assert_eq!(combine.i, [6144, 1, rows, 0, 0, 0, 0, 1]);
            } else {
                assert_eq!(inst.i, [rows, 6144, 256, 256, 8, 0, 1, u32::from(resident)]);
                assert_eq!(&inst.t[5..], &[TENSOR_NONE; 3]);
                let router = prog.insts[..ix]
                    .iter()
                    .rposition(|d| d.op == DevOp::MoeRouterTopkPf as u16 && d.t[0] == inst.t[4])
                    .unwrap();
                assert!(!prog.insts[router..ix]
                    .iter()
                    .any(|d| d.op == DevOp::MoeAlignPf as u16));
                let combine = prog.insts[ix + 1..]
                    .iter()
                    .find(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == inst.t[0])
                    .unwrap();
                assert_eq!(combine.i, [6144, 1, rows, 0, 0, 0, 0, 1]);
            }
            let segment = prog
                .stream
                .iter()
                .find(|e| e.inst as usize == ix)
                .unwrap()
                .seg;
            assert!(segment > 0);
            for entries in [&prog.stream, &prog.gq_stream] {
                for e in entries.iter().filter(|e| e.seg == segment) {
                    assert_eq!(e.inst as usize, ix);
                    assert_eq!(
                        (e.wait_len, e.succ_len, e.flags & packet::dev::SE_XCTR),
                        (0, 0, 0)
                    );
                }
                assert!(entries.iter().any(|e| e.seg > segment));
            }
            let segments = prog
                .stream
                .iter()
                .map(|e| usize::from(e.seg) + 1)
                .max()
                .unwrap();
            if decode {
                assert_eq!(prog.gq_seg_ofs.len(), segments * 8 + 1);
            }
        }
        assert_eq!(checked, if resident { 21 } else { 3 });
        Ok(crate::LeanReport::skipped(
            "flat decode segment regression test",
        ))
    });
    glm_emit_full(
        &dir,
        81920,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        Some(packet::devbuild::L2Layout {
            sms: 38,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        }),
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn glm_native_decode_gemm_preserves_xcd_boundaries() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", "full:128"),
        ("PLOW_GLM_PLACE_PF", "0"),
        ("PLOW_GLM_MOE_AITER", "0"),
        ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
        ("PLOW_GLM_GEMM_LT_DECODE", "1"),
        ("PLOW_GLM_GEMM_LT", "0"),
        ("PLOW_DECODE_BATCH_LADDER", "1,16,20"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "0"),
    ]);
    let dir = std::env::temp_dir().join(format!("plow-glm-native-gemm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(|model| {
        let mut checked = 0;
        for (prog, &rows) in model.progs.iter().zip(&model.prog_t) {
            let native = prog
                .insts
                .iter()
                .enumerate()
                .filter(|(_, d)| d.op == DevOp::GemmLtPf as u16)
                .collect::<Vec<_>>();
            if !matches!(rows, 16 | 20) {
                assert!(native.is_empty());
                continue;
            }
            checked += 1;
            assert_eq!(prog.l2_domains, 8);
            assert_eq!(native.len(), 16);
            assert_eq!(
                native
                    .iter()
                    .filter(|(_, d)| (d.i[1], d.i[2]) == (256, 6144))
                    .count(),
                3
            );
            assert_eq!(
                native
                    .iter()
                    .filter(|(_, d)| (d.i[1], d.i[2]) == (6144, 256))
                    .count(),
                1
            );
            for (ix, inst) in native {
                assert_eq!(inst.i[0], rows);
                assert_eq!(inst.i[3], 1);
                assert!(matches!(
                    (inst.i[1], inst.i[2]),
                    (2048, 6144)
                        | (512, 6144)
                        | (4096, 2048)
                        | (6144, 2048)
                        | (256, 6144)
                        | (6144, 256)
                ));
                let segment = prog
                    .stream
                    .iter()
                    .find(|e| e.inst as usize == ix)
                    .unwrap()
                    .seg;
                assert!(segment > 0);
                for entries in [&prog.stream, &prog.gq_stream] {
                    for e in entries.iter().filter(|e| e.seg == segment) {
                        assert_eq!(e.inst as usize, ix);
                        assert_eq!(
                            (e.wait_len, e.succ_len, e.flags & packet::dev::SE_XCTR),
                            (0, 0, 0)
                        );
                    }
                    assert!(entries.iter().any(|e| e.seg > segment));
                }
            }
            let segments = prog
                .stream
                .iter()
                .map(|e| usize::from(e.seg) + 1)
                .max()
                .unwrap();
            assert_eq!(prog.gq_seg_ofs.len(), segments * 8 + 1);
        }
        assert_eq!(checked, 2);
        Ok(crate::LeanReport::skipped(
            "native decode GEMM segment regression test",
        ))
    });
    glm_emit_full(
        &dir,
        512,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        Some(packet::devbuild::L2Layout {
            sms: 38,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        }),
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn glm_native_decode_gemm_ext_covers_rung8_and_narrow_projections() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", "full:128"),
        ("PLOW_GLM_PLACE_PF", "0"),
        ("PLOW_GLM_MOE_AITER", "0"),
        ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
        ("PLOW_GLM_GEMM_LT_DECODE", "1"),
        ("PLOW_GLM_GEMM_LT_DECODE_EXT", "1"),
        ("PLOW_GLM_GEMM_LT", "0"),
        ("GLM_SHARD_HEAD", "1"),
        ("PLOW_DECODE_BATCH_LADDER", "1,4,8,16,20"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "0"),
    ]);
    let dir =
        std::env::temp_dir().join(format!("plow-glm-native-gemm-ext-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // GLM-5.3 geometry with the real vocab: the lm_head shard is the whitelisted [19360, 6144].
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 154880, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(|model| {
        let mut checked = 0;
        for (prog, &rows) in model.progs.iter().zip(&model.prog_t) {
            let native = prog
                .insts
                .iter()
                .enumerate()
                .filter(|(_, d)| d.op == DevOp::GemmLtPf as u16)
                .collect::<Vec<_>>();
            // Prefill buckets and rungs 1/4 stay on the interpreter GEMV.
            if !matches!(rows, 8 | 16 | 20) {
                assert!(native.is_empty(), "rows={rows}");
                continue;
            }
            checked += 1;
            let count = |shape: (u32, u32)| {
                native
                    .iter()
                    .filter(|(_, d)| (d.i[1], d.i[2]) == shape)
                    .count()
            };
            // The EXT shapes: k_rope and q_rope once per layer wherever the rung emits them
            // unfused (fusions A/G fold them into a GemvQkv below their LDS fit), lm_head once.
            let fused = |nq: u32, k: u32| {
                prog.insts
                    .iter()
                    .filter(|d| d.op == DevOp::GemvQkv as u16 && (d.i[1], d.i[2]) == (nq, k))
                    .count()
            };
            assert_eq!(count((64, 6144)) + fused(2048, 6144), 4, "k_rope rows={rows}");
            assert_eq!(count((512, 2048)) + fused(4096, 2048), 4, "q_rope rows={rows}");
            assert_eq!(count((19360, 6144)), 1, "lm_head rows={rows}");
            // The base set the plain knob already routes is still routed where it is unfused
            // (rung 8 folds q_a|kv_a into fusion A in this 4-layer model).
            for shape in [
                (2048, 6144),
                (512, 6144),
                (4096, 2048),
                (6144, 2048),
                (256, 6144),
                (6144, 256),
            ] {
                let fused_form = match shape {
                    (2048, 6144) | (512, 6144) => fused(2048, 6144),
                    (4096, 2048) => fused(4096, 2048),
                    _ => 0,
                };
                assert!(count(shape) + fused_form > 0, "{shape:?} rows={rows}");
            }
            assert!(
                prog.insts.iter().all(|d| d.op != DevOp::Gemv as u16
                    || !matches!((d.i[1], d.i[2]), (64, 6144) | (512, 2048) | (19360, 6144))),
                "an EXT shape stayed on the interpreter GEMV at rows={rows}"
            );
            for (ix, inst) in native {
                assert_eq!(inst.i[0], rows);
                assert_eq!(inst.i[3], 1);
                assert_eq!(inst.i[4..], [0; 4]);
                let segment = prog
                    .stream
                    .iter()
                    .find(|e| e.inst as usize == ix)
                    .unwrap()
                    .seg;
                for e in prog.stream.iter().filter(|e| e.seg == segment) {
                    assert_eq!(e.inst as usize, ix);
                    assert_eq!(
                        (e.wait_len, e.succ_len, e.flags & packet::dev::SE_XCTR),
                        (0, 0, 0)
                    );
                }
            }
        }
        assert_eq!(checked, 3);
        Ok(crate::LeanReport::skipped(
            "native decode GEMM EXT regression test",
        ))
    });
    glm_emit_full(
        &dir,
        512,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        None,
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// `PLOW_GLM_DECODE_GEMM_GROUP` only reorders: every program keeps the same instructions, rungs
/// without native decode GEMMs keep their order, and on the native rungs the MoE layer's router and
/// shared gate/up GEMMs are adjacent while top-k and Glu share one segment.
#[test]
fn glm_decode_gemm_group_reorders_native_gemms_without_changing_work() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    type Inst = (u16, u32, u32, u16);
    type Prog = (u32, Vec<Inst>, Vec<String>);
    let emit = |group: &str| -> Vec<Prog> {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_MLA_PREFILL", "full:128"),
            ("PLOW_GLM_PLACE_PF", "0"),
            ("PLOW_GLM_MOE_AITER", "0"),
            ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
            ("PLOW_GLM_GEMM_LT_DECODE", "1"),
            ("PLOW_GLM_GEMM_LT_DECODE_EXT", "1"),
            ("PLOW_GLM_GEMM_LT", "0"),
            ("GLM_SHARD_HEAD", "1"),
            ("PLOW_DECODE_BATCH_LADDER", "1,4,8,16,20"),
            ("PLOW_EMIT_PACKED_PREFILL", "0"),
            ("PLOW_UNISEG", "0"),
            ("PLOW_GLM_DECODE_GEMM_GROUP", group),
        ]);
        let dir = std::env::temp_dir().join(format!(
            "plow-glm-gemm-group-{group}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
            "hidden_size": 6144, "num_attention_heads": 64,
            "kv_lora_rank": 512, "q_lora_rank": 2048,
            "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
            "vocab_size": 154880, "rms_norm_eps": 1e-5,
            "n_routed_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048, "intermediate_size": 12288,
            "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
            "rope_theta": 8000000.0
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let out = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = out.clone();
        let verify: crate::VerifyHook = Box::new(move |model| {
            for (prog, &rows) in model.progs.iter().zip(&model.prog_t) {
                let seg = |ix: usize| {
                    prog.stream
                        .iter()
                        .find(|e| e.inst as usize == ix)
                        .map_or(u16::MAX, |e| e.seg)
                };
                let insts = prog
                    .insts
                    .iter()
                    .enumerate()
                    .map(|(ix, d)| (d.op, d.i[1], d.i[2], seg(ix)))
                    .collect();
                let payload = prog
                    .insts
                    .iter()
                    .map(|d| format!("{:?} {} {:?} {:?} {:?}", d.op, d.blocks, d.t, d.i, d.j))
                    .collect();
                sink.lock().unwrap().push((rows, insts, payload));
            }
            Ok(crate::LeanReport::skipped("decode GEMM group test"))
        });
        glm_emit_full(
            &dir,
            512,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            None,
            Some(&verify),
        );
        std::fs::remove_dir_all(dir).unwrap();
        let progs = out.lock().unwrap().clone();
        progs
    };
    let (off, on) = (emit("0"), emit("1"));
    assert_eq!(off.len(), on.len());
    let lt = DevOp::GemmLtPf as u16;
    let mut native_rungs = 0;
    for ((rows, a, pa), (_, b, pb)) in off.iter().zip(&on) {
        let (mut x, mut y) = (pa.clone(), pb.clone());
        x.sort();
        y.sort();
        assert_eq!(x, y, "rows={rows}: the reorder must keep the same instructions");
        // The shared expert splits into native gate/up halves + Glu only past its LDS fit (rows
        // 16/20); below it, or with no native GEMM, there is nothing to move.
        if !a.iter().any(|i| i.0 == lt) || !a.iter().any(|i| i.0 == DevOp::Glu as u16) {
            assert_eq!(pa, pb, "rows={rows}: nothing to group, no reorder");
            continue;
        }
        native_rungs += 1;
        let n_seg = |p: &[Inst]| p.iter().map(|i| i.3).max().unwrap();
        // One MoE layer in this model: top-k and Glu now share a segment.
        assert_eq!(n_seg(b) + 1, n_seg(a), "rows={rows}");
        let topk = b
            .iter()
            .position(|i| i.0 == DevOp::MoeRouterTopkPf as u16)
            .unwrap();
        assert!(
            b[topk - 3..topk]
                .iter()
                .all(|i| (i.0, i.1, i.2) == (lt, 256, 6144)),
            "rows={rows}: router, shared gate and shared up must precede top-k back to back"
        );
        let glu = b.iter().position(|i| i.0 == DevOp::Glu as u16).unwrap();
        assert_eq!(
            b[topk].3, b[glu].3,
            "rows={rows}: top-k and Glu share a segment"
        );
    }
    assert_eq!(native_rungs, 2);
}

#[test]
fn glm_native_prefill_fold_preserves_xcd_boundaries() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", "full:128,2048"),
        ("PLOW_GLM_PLACE_PF", "1"),
        ("PLOW_GLM_MOE_AITER", "0"),
        ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
        ("PLOW_GLM_GEMM_LT_DECODE", "0"),
        ("PLOW_GLM_FOLD_LT", "1"),
        ("PLOW_GLM_GEMM_LT", "0"),
        ("PLOW_DECODE_BATCH_LADDER", "1"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "1"),
    ]);
    let dir = std::env::temp_dir().join(format!("plow-glm-native-fold-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(|model| {
        let mut checked = 0;
        for (prog, &rows) in model.progs.iter().zip(&model.prog_t) {
            let native = prog
                .insts
                .iter()
                .enumerate()
                .filter(|(_, d)| d.op == DevOp::MlaMergeFold as u16 && d.i[5] != 0)
                .collect::<Vec<_>>();
            if rows != 2048 {
                assert!(native.is_empty());
                continue;
            }
            checked += 1;
            assert_eq!(prog.l2_domains, 8);
            assert_eq!(native.len(), 4);
            for (ix, inst) in native {
                assert_eq!(inst.i[0], rows);
                assert_eq!(inst.i[5], 1);
                assert_eq!((inst.i[1], inst.i[2]), (8, 256));
                let segment = prog
                    .stream
                    .iter()
                    .find(|e| e.inst as usize == ix)
                    .unwrap()
                    .seg;
                assert!(segment > 0);
                for entries in [&prog.stream, &prog.gq_stream] {
                    for e in entries.iter().filter(|e| e.seg == segment) {
                        assert_eq!(e.inst as usize, ix);
                        assert_eq!(
                            (e.wait_len, e.succ_len, e.flags & packet::dev::SE_XCTR),
                            (0, 0, 0)
                        );
                    }
                    assert!(entries.iter().any(|e| e.seg > segment));
                }
            }
            let segments = prog
                .stream
                .iter()
                .map(|e| usize::from(e.seg) + 1)
                .max()
                .unwrap();
            assert_eq!(prog.gq_seg_ofs.len(), segments * 8 + 1);
        }
        assert_eq!(checked, 1);
        Ok(crate::LeanReport::skipped(
            "native prefill fold segment regression test",
        ))
    });
    glm_emit_full(
        &dir,
        4096,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        Some(packet::devbuild::L2Layout {
            sms: 38,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        }),
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// Token-batch body (`plans/unified-token-batch.md`, "AMD TP8 lowering decision"): the band rows
/// take the batched decode attention chain BEHIND the prefill fold, on the prefill projections;
/// the tail samples the whole band through a tiled GEMM; the band packets sit in their own
/// segment class. The plain emit (`band = None`) is what every other test in this module pins.
#[test]
fn token_batch_body_band_rides_the_prefill_program() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[("PLOW_GLM_FP8_KV", "1"), ("PLOW_UNISEG", "0")]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    let (ctx, rows, band) = (81920u32, 128u32, 20u32);
    let mut decl = Builder::new(304);
    let n = declare_glm_rows_batched(&mut decl, &c, ctx, &[0], rows, band, MoeEnc::Fp8Blk);
    let mut b = Builder::new(304);
    b.adopt_tensors(decl.tensors());
    b.set_packed_prefill_segments(true);
    b.set_token_batch_band(band);
    let all = b.all();
    emit_glm_mla_prefill(&mut b, &c, &n, 0, ctx, rows, MoeEnc::Fp8Blk, n.x, &[], false, &mut 0, &all,
        Some(band));
    emit_glm_tail(&mut b, &c, &n, n.x, &[], rows, false, &mut 0, Some(band));
    let p = b.finish();
    let ops: Vec<u16> = p.insts.iter().map(|d| d.op).collect();
    let pf = ops.iter().position(|&o| o == DevOp::FlashMlaPrefillFp8 as u16).unwrap();
    let dec = ops.iter().position(|&o| o == DevOp::FlashMlaDecodeFp8 as u16).unwrap();
    let folds: Vec<usize> = ops.iter().enumerate()
        .filter(|(_, &o)| o == DevOp::MlaMergeFold as u16).map(|(i, _)| i).collect();
    assert_eq!(folds.len(), 2, "one prefill fold over T, one band fold over the band");
    assert!(pf < folds[0] && folds[0] < dec && dec < folds[1],
        "order: prefill flash {pf}, prefill fold {}, band flash {dec}, band fold {}", folds[0], folds[1]);
    assert_eq!(p.insts[folds[0]].i[0], rows);
    assert_eq!(p.insts[folds[1]].i[0], band);
    let fl = &p.insts[dec];
    assert_eq!((fl.i[0], fl.i[1], fl.t[6], fl.t[4]), (band, c.heads / c.tp, n.kvlen, n.ckv[0]));
    // The prefill flash still covers every row: band rows are ordinary rows of the T-row packets.
    assert_eq!(p.insts[pf].i[4], rows);
    if c.dsa(ctx) && n.lw[0].iwqb != TENSOR_NONE {
        let sc = p.insts.iter().find(|d| d.op == DevOp::IndexScore as u16).unwrap();
        assert_eq!((sc.i[0], sc.t[1], sc.t[3]), (band, n.qidx, n.widx));
        assert!(p.insts.iter().any(|d| d.op == DevOp::IndexSelect as u16));
        assert_eq!(fl.j[0], n.iidx + 1, "sparse gather over the band's own selection");
        // The band's indexer query projection is a tiled GEMM over the band, not a Gemv.
        let qi = p.insts.iter().find(|d| d.t[0] == n.qidx && d.t[1] == n.qlat).unwrap();
        assert_ne!(qi.op, DevOp::Gemv as u16);
        assert_eq!(qi.i[0], band);
    }
    // Tail: the head is a tiled GEMM over the band and every band row samples.
    let am = p.insts.iter().find(|d| d.op == DevOp::Argmax as u16).unwrap();
    assert_eq!(am.i[1], band);
    let head = p.insts.iter().find(|d| d.t[0] == n.logits).unwrap();
    assert_ne!(head.op, DevOp::Gemv as u16);
    assert_eq!((head.i[0], head.i[1]), (band, glm_vocab_l(&c)));
    let fin = p.insts.iter().find(|d| d.op == DevOp::XArgmaxFin as u16 || d.op == DevOp::ArgmaxFin as u16).unwrap();
    assert_eq!(fin.i[1], band);
    // Segment isolation: the band attention is not in the prefill flash's or fold's segment.
    let seg_of = |ix: usize| p.stream.iter().find(|e| e.inst as usize == ix).unwrap().seg;
    assert_ne!(seg_of(dec), seg_of(pf));
    assert_ne!(seg_of(folds[1]), seg_of(folds[0]));
    assert_eq!(seg_of(dec), seg_of(folds[1]), "the band flash and its fold share one segment");
}

#[test]
fn dsa_decode_nsplit_fills_one_item_per_cu() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[("PLOW_MLA_NS", "")]);
    // TP8 (nh_l=8, GF=4): 2 head-groups per row on 304 CUs.
    for (rows, expected) in [(1, 16), (2, 16), (4, 16), (8, 16), (16, 8), (20, 4), (64, 4)] {
        assert_eq!(glm_dsa_decode_nsplit(rows, 8, 4, 304), expected, "rows {rows}");
    }
    // TP4 (nh_l=16): 4 head-groups per row.
    assert_eq!(glm_dsa_decode_nsplit(1, 16, 4, 304), 16);
    assert_eq!(glm_dsa_decode_nsplit(20, 16, 4, 304), 4);
    // The pin wins, as it does for the dense rule.
    let _pin = crate::test_env::EnvScope::set(&[("PLOW_MLA_NS", "8")]);
    assert_eq!(glm_dsa_decode_nsplit(20, 8, 4, 304), 8);
}

/// The sparse (DSA) 8192 rung joins the packed-sibling and token-batch-body passes only under
/// `PLOW_PACKED_SPARSE_PF=1`, and then with the span-aware chain: the TP indexer feeds the
/// sparse FP8 flash directly (`fj[1] = iidx_pf + 1`), no per-8-query union. Every ordinary
/// program is byte-identical either way — `PLOW_PACKED_SPARSE_PF=0` is the rollback.
#[test]
fn sparse_rung_joins_the_packed_passes_only_under_packed_sparse_pf() {
    use std::sync::{Arc, Mutex};
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let dir = std::env::temp_dir().join(format!("plow-glm-packed-sparse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // `indexer_types`: a "full" layer owns the DSA indexer whose prefill selection chain is
    // the class-C site under test; "shared" layers reuse the last full layer's selection.
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0,
        "indexer_types": ["full", "shared", "full", "shared"]
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    type Snapshot = Vec<(u32, Vec<packet::dev::DevInst>, Vec<packet::dev::StreamEnt>, Vec<packet::dev::StreamEnt>)>;
    let emit = |sparse_pf: &str| -> Snapshot {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_MLA_PREFILL", "full:2048,8192"),
            ("PLOW_DECODE_BATCH", "20"),
            ("PLOW_GLM_DSA_PF", "1"),
            ("PLOW_GLM_INDEX_TP", "1"),
            ("PLOW_GLM_FP8_KV", "1"),
            ("PLOW_GLM_MOE_AITER", "1"),
            ("PLOW_GLM_GEMM_LT", "1"),
            ("PLOW_MLA_PF_V2", "1"),
            ("PLOW_MLA_PF_AITER", "1"),
            ("PLOW_EMIT_PACKED_PREFILL", "1"),
            ("PLOW_TOKEN_BATCH_TP", "1"),
            ("PLOW_PACKED_SPARSE_PF", sparse_pf),
            ("PLOW_UNISEG", "0"),
        ]);
        let seen: Arc<Mutex<Snapshot>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let verify: crate::VerifyHook = Box::new(move |model| {
            *sink.lock().unwrap() = model
                .progs
                .iter()
                .zip(&model.prog_t)
                .map(|(p, &t)| (t, p.insts.clone(), p.stream.clone(), p.gq_stream.clone()))
                .collect();
            Ok(crate::LeanReport::skipped("structural regression test"))
        });
        // The production context: a batched FP8-KV decode ladder needs the DSA decode arm,
        // which arms above the 64K crossover.
        glm_emit_full(
            &dir,
            81920,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            None,
            Some(&verify),
        );
        let out = seen.lock().unwrap().clone();
        out
    };
    let dense_only = emit("0");
    let with_sparse = emit("1");
    std::fs::remove_dir_all(dir).unwrap();

    type Entry = (u32, Vec<packet::dev::DevInst>, Vec<packet::dev::StreamEnt>, Vec<packet::dev::StreamEnt>);
    fn ordinary(s: &[Entry]) -> Vec<&Entry> {
        s.iter()
            .filter(|e| !packet::devbuild::is_packed_prefill_program(e.0) && !packet::devbuild::is_token_batch_program(e.0))
            .collect()
    }
    for entry in ordinary(&dense_only) {
        let twin = with_sparse
            .iter()
            .find(|e| e.0 == entry.0)
            .unwrap_or_else(|| panic!("program t={} vanished under PLOW_PACKED_SPARSE_PF", entry.0));
        assert!(entry == twin, "ordinary program t={} changed under PLOW_PACKED_SPARSE_PF", entry.0);
    }
    let has = |insts: &[packet::dev::DevInst], op: DevOp| insts.iter().any(|d| d.op == op as u16);
    // The ordinary 8192 rung is sparse and keeps the union: its flash names `iuni`.
    let plain_8192 = dense_only.iter().find(|e| e.0 == 8192).expect("8192 rung");
    assert!(has(&plain_8192.1, DevOp::IndexTpPf) && has(&plain_8192.1, DevOp::IndexUnionPf));
    let plain_flash = plain_8192.1.iter().find(|d| d.op == DevOp::FlashMlaPrefillFp8 as u16 && d.j[0] != 0).unwrap();
    let plain_union = plain_8192.1.iter().find(|d| d.op == DevOp::IndexUnionPf as u16).unwrap();
    assert_eq!(plain_flash.j[0], plain_union.t[0] + 1);
    // Dense-only: the sparse bucket gets neither a sibling nor a body; the dense 2048 gets both.
    fn tagged(s: &[Entry], t: u32) -> Vec<&Vec<packet::dev::DevInst>> {
        s.iter()
            .filter(|e| e.0 == packet::devbuild::packed_prefill_program_t(t) || e.0 == packet::devbuild::token_batch_program_t(t))
            .map(|e| &e.1)
            .collect()
    }
    assert_eq!(tagged(&dense_only, 8192).len(), 0);
    assert_eq!(tagged(&dense_only, 2048).len(), 2);
    // With the knob: one sibling and one body at 8192, carrying the span-aware chain.
    let sparse_packed = tagged(&with_sparse, 8192);
    assert_eq!(sparse_packed.len(), 2, "one packed sibling and one token-batch body at 8192");
    for insts in sparse_packed {
        assert!(has(insts, DevOp::IndexTpPf), "the TP indexer (PlowKvSpan table at the runtime)");
        assert!(!has(insts, DevOp::IndexUnionPf), "no per-8-query union under the packed topology");
        assert!(!has(insts, DevOp::IndexScorePf) && !has(insts, DevOp::IndexSelectPf));
        let tp = insts.iter().find(|d| d.op == DevOp::IndexTpPf as u16).unwrap();
        let flash = insts.iter().find(|d| d.op == DevOp::FlashMlaPrefillFp8 as u16 && d.j[0] != 0).unwrap();
        assert_eq!(flash.j[0], tp.t[0] + 1, "the sparse flash gathers the TP indexer's per-row selection");
        assert_eq!((flash.i[4], tp.i[0]), (8192, 8192));
    }
    assert_eq!(tagged(&with_sparse, 2048).len(), 2);
}

/// PLOW_GLM_MOE_SHARED_FOLD: the two-emit comparison. Same config and knobs, fold off vs on.
///
/// Off is the default-off contract (the knob's first emit IS the ordinary packet). On, every
/// native prefill MoE chain moves as one: the router appends its constant slot (i5=1) over the
/// same 256/8 selection, the align and the fused call both run 257/9, the shared expert's
/// GLU/down packets disappear, and every combine band gives up its `shared` operand while still
/// covering all T rows. Decode keeps top-8 over the 257-entry table and its own shared GEMVs.
#[test]
fn shared_fold_rewrites_only_the_prefill_moe_chain() {
    use std::sync::{Arc, Mutex};
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let dir = std::env::temp_dir().join(format!("plow-glm-shared-fold-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    type Snapshot = (Vec<(u32, Vec<packet::dev::DevInst>)>, Vec<(String, u64)>);
    let emit = |fold: &str| -> Snapshot {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_MLA_PREFILL", "full:1,2,4,8,16,20,32,64,128,256,512,1024,2048,4096,8192"),
            ("PLOW_GLM_PLACE_PF", "0"),
            ("PLOW_GLM_MOE_AITER", "0"),
            ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
            ("PLOW_GLM_MOE_RESIDENT", "1"),
            ("PLOW_GLM_MOE_SHARED_FOLD", fold),
            ("PLOW_DECODE_BATCH_LADDER", "1,2,4,8,16,20"),
            ("PLOW_EMIT_PACKED_PREFILL", "0"),
            ("PLOW_UNISEG", "0"),
        ]);
        let seen: Arc<Mutex<Snapshot>> = Arc::new(Mutex::new((Vec::new(), Vec::new())));
        let sink = Arc::clone(&seen);
        let verify: crate::VerifyHook = Box::new(move |model| {
            *sink.lock().unwrap() = (
                model.progs.iter().zip(&model.prog_t).map(|(p, &t)| (t, p.insts.clone())).collect(),
                model.tensors.iter().map(|t| (t.name.clone(), t.bytes)).collect(),
            );
            Ok(crate::LeanReport::skipped("shared-fold structural test"))
        });
        glm_emit_full(
            &dir,
            81920,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            Some(packet::devbuild::L2Layout {
                sms: 38,
                domains: 8,
                map: packet::devbuild::L2Map::RoundRobin,
            }),
            Some(&verify),
        );
        let out = seen.lock().unwrap().clone();
        out
    };
    let (plain, plain_tensors) = emit("0");
    let (folded, folded_tensors) = emit("1");
    std::fs::remove_dir_all(dir).unwrap();

    let prefill_count =
        packet::devbuild::decode_rung_lo(&plain.iter().map(|(t, _)| *t).collect::<Vec<_>>());
    assert_eq!(plain.len(), folded.len());
    let mut native_prefill = 0;
    for (p, ((t, off), (t2, on))) in plain.iter().zip(&folded).enumerate() {
        assert_eq!(t, t2);
        let decode = p >= prefill_count;
        let natives: Vec<usize> = on
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::MoeAiterFp8Pf as u16)
            .map(|(i, _)| i)
            .collect();
        let plain_natives = off.iter().filter(|d| d.op == DevOp::MoeAiterFp8Pf as u16).count();
        assert_eq!(natives.len(), plain_natives, "t={t}: a MoE layer gained or lost its call");
        for &ix in &natives {
            let call = &on[ix];
            if decode {
                assert_eq!(call.i[3..5], [257, 8], "t={t}: decode keeps top-8 over 257 entries");
                let router = on[..ix]
                    .iter()
                    .rev()
                    .find(|d| d.op == DevOp::MoeRouterTopkPf as u16 && d.t[0] == call.t[4])
                    .unwrap();
                assert_eq!((router.i[1], router.i[2], router.i[5]), (256, 8, 0));
                continue;
            }
            native_prefill += 1;
            assert_eq!(call.i, [*t, 6144, 256, 257, 9, 64, 2, 1], "t={t}");
            let align = on[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeAlignPf as u16 && d.t[0] == call.t[4])
                .unwrap();
            assert_eq!(align.i[..3], [*t, 257, 9], "t={t}");
            let router = on[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeRouterTopkPf as u16 && d.t[0] == align.t[1])
                .unwrap();
            assert_eq!((router.i[1], router.i[2], router.i[5]), (256, 8, 1), "t={t}");
            let mut covered = 0;
            for c in on[ix + 1..]
                .iter()
                .take_while(|d| d.t[0] != call.t[0])
                .filter(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == call.t[0])
            {
                assert_eq!(c.t[2], TENSOR_NONE, "t={t}: the combine still reads `shared`");
                assert_eq!(c.i[3], covered, "t={t}: combine bands are not contiguous");
                covered += c.i[2];
            }
            assert_eq!(covered, *t, "t={t}: combine bands do not cover every row");
        }
        let mut a: Vec<u16> = off.iter().map(|d| d.op).collect();
        let mut b: Vec<u16> = on.iter().map(|d| d.op).collect();
        a.sort_unstable();
        b.sort_unstable();
        if decode {
            assert_eq!(a, b, "t={t}: the fold changed a decode program's ops");
            continue;
        }
        // Prefill: what goes away is EXACTLY the packets that touched a `shared_experts.*`
        // weight in the unfolded program (its gate|up and down, whatever tile op the rung picked),
        // two per native MoE layer, and nothing is added.
        for o in &b {
            let at = a.iter().position(|x| x == o).expect("the fold added an op");
            a.remove(at);
        }
        let shared_weight = |h: u32| {
            plain_tensors
                .get(h as usize)
                .is_some_and(|(n, _)| n.contains("shared_experts."))
        };
        let mut shared_ops: Vec<u16> =
            off.iter().filter(|d| d.t.iter().any(|&h| shared_weight(h))).map(|d| d.op).collect();
        shared_ops.sort_unstable();
        assert_eq!(a, shared_ops, "t={t}: the fold removed something other than the shared GEMMs");
        assert_eq!(a.len(), 2 * natives.len(), "t={t}: removed {a:?}");
    }
    assert!(native_prefill > 0, "no prefill program carried a native MoE call");
    // The pointer tables switch to the `_sf` spelling at 257 entries; nothing else is added or
    // removed from the tensor table.
    let sf: Vec<_> = folded_tensors.iter().filter(|(n, _)| n.ends_with("_table_sf")).collect();
    assert!(!sf.is_empty() && sf.iter().all(|(_, b)| *b == 257 * 3 * 8));
    assert!(!folded_tensors.iter().any(|(n, _)| n.ends_with("expert_weight_table")));
    assert_eq!(plain_tensors.len(), folded_tensors.len());
}

/// The packed siblings ride next to the native AITER MoE and hipBLASLt segments — the production
/// gfx942 TP8 recipe — and emitting them leaves every ordinary program byte-identical. This is
/// the emit half of what lets the serve mux pack several requests' spans into one rung on that
/// recipe; the runtime half is `exec/amd.rs`'s `packed_mla_compatible`.
#[test]
fn packed_siblings_carry_native_moe_and_leave_the_plain_programs_byte_identical() {
    use std::sync::{Arc, Mutex};
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let dir = std::env::temp_dir().join(format!("plow-glm-packed-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    type Snapshot = Vec<(u32, Vec<packet::dev::DevInst>, Vec<packet::dev::StreamEnt>, Vec<packet::dev::StreamEnt>)>;
    let emit = |packed: &str| -> Snapshot {
        let _env = crate::test_env::EnvScope::set(&[
            ("PLOW_MLA_PREFILL", "full:128,2048"),
            ("PLOW_GLM_MOE_AITER", "1"),
            ("PLOW_GLM_GEMM_LT", "1"),
            ("PLOW_EMIT_PACKED_PREFILL", packed),
            ("PLOW_UNISEG", "0"),
        ]);
        let seen: Arc<Mutex<Snapshot>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let verify: crate::VerifyHook = Box::new(move |model| {
            *sink.lock().unwrap() = model
                .progs
                .iter()
                .zip(&model.prog_t)
                .map(|(p, &t)| (t, p.insts.clone(), p.stream.clone(), p.gq_stream.clone()))
                .collect();
            Ok(crate::LeanReport::skipped("structural regression test"))
        });
        glm_emit_full(
            &dir,
            4096,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            None,
            Some(&verify),
        );
        let out = seen.lock().unwrap().clone();
        out
    };
    let plain = emit("0");
    let with_siblings = emit("1");
    std::fs::remove_dir_all(dir).unwrap();

    assert!(plain.iter().all(|(t, ..)| !packet::devbuild::is_packed_prefill_program(*t)));
    // Every ordinary program — both prefill buckets and the decode rung — is byte-identical.
    for entry in &plain {
        let twin = with_siblings
            .iter()
            .find(|e| e.0 == entry.0)
            .unwrap_or_else(|| panic!("program t={} vanished under the packed emit", entry.0));
        assert!(entry == twin, "program t={} changed under the packed emit", entry.0);
    }
    // Each dense bucket gained exactly one sibling, and the sibling carries the native segments.
    for rows in [128u32, 2048] {
        let siblings: Vec<_> = with_siblings
            .iter()
            .filter(|e| e.0 == packet::devbuild::packed_prefill_program_t(rows))
            .collect();
        assert_eq!(siblings.len(), 1, "rows={rows}");
        let (_, insts, stream, _) = siblings[0];
        let has = |op: DevOp| insts.iter().any(|d| d.op == op as u16);
        assert!(has(DevOp::MoeAiterFp8Pf), "rows={rows}: native MoE");
        assert_eq!(has(DevOp::GemmLtPf), rows >= 2048, "rows={rows}: hipBLASLt");
        assert!(!has(DevOp::IndexTpPf) && !has(DevOp::FlashGatherPrefill));
        // The MLA family segments are pure: a norm/flash op never shares its segment with a
        // Gemm or the MoE chain, which is what `check_packed_prefill_program` routes on.
        let family = |op: u16| -> u8 {
            if op == DevOp::RmsNorm as u16
                || op == DevOp::HeadNormRope as u16
                || op == DevOp::HeadNormRopeFp8 as u16
            {
                5
            } else if op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16 {
                6
            } else {
                0
            }
        };
        let seg_family = |seg: u16| -> Vec<u8> {
            stream
                .iter()
                .filter(|e| e.seg == seg)
                .map(|e| family(insts[e.inst as usize].op))
                .collect()
        };
        for e in stream {
            let f = family(insts[e.inst as usize].op);
            if f != 0 {
                assert!(seg_family(e.seg).iter().all(|&g| g == f), "rows={rows} seg={}", e.seg);
            }
        }
    }
}


/// THE QUALIFIED RECIPE IS WHAT AN UNFLAGGED GLM gfx942 TP8 EMIT PRODUCES.
///
/// The packet serving GLM-5.3 on 8x MI300X (47-50 out tok/s, 18/18 on the retrieval screen) was
/// emitted by naming the `glm_*` recipe flags. Every one of them used to be `default_value_t = false`,
/// so the qualified configuration was reachable only by typing the whole incantation and dropping
/// one emitted a slower packet that still loaded and still served. This is the gate on that: the
/// explicit recipe and the unflagged emit produce the SAME programs, and an arm with the recipe
/// explicitly OFF produces different ones — which is what makes the first assertion mean
/// something rather than pass vacuously.
///
/// `PLOW_GLM_DSA_PF` is set in every arm: it is part of the frozen recipe but is NOT one of the
/// defaulted knobs (sparse prefill is a separate qualification), so it has to be named on both
/// sides for the comparison to be about the recipe.
#[test]
fn the_qualified_glm_recipe_is_what_an_unflagged_gfx942_tp8_emit_produces() {
    use std::sync::{Arc, Mutex};
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let dir = std::env::temp_dir().join(format!("plow-glm-defaults-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0,
        "indexer_types": ["full", "shared", "full", "shared"]
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    type Snapshot = Vec<(u32, Vec<packet::dev::DevInst>, Vec<packet::dev::StreamEnt>, Vec<packet::dev::StreamEnt>)>;
    const RECIPE: [&str; 12] = [
        "PLOW_GLM_FP8_KV",
        "PLOW_GLM_MOE_AITER",
        "PLOW_GLM_MOE_RESIDENT",
        "PLOW_GLM_INDEX_TP",
        "PLOW_GLM_SELECT_LOCAL",
        "PLOW_GLM_DECODE_NORM_ROWS",
        "PLOW_GLM_GEMM_LT",
        "PLOW_GLM_GEMM_LT_DECODE",
        "PLOW_GLM_GEMM_LT_DECODE_EXT",
        "PLOW_GLM_FOLD_LT",
        "PLOW_GLM_SEQ_PAR",
        "PLOW_GLM_SEQ_PAR_PROJ",
    ];
    let emit = |glm: &[(&str, &str)]| -> Snapshot {
        let mut env: Vec<(&str, &str)> = vec![
            ("PLOW_MLA_PREFILL", "full:2048,8192"),
            ("PLOW_DECODE_BATCH", "20"),
            ("PLOW_GLM_DSA", "1"),
            ("PLOW_GLM_DSA_PF", "1"),
            ("PLOW_MLA_PF_V2", "1"),
            ("PLOW_MLA_PF_AITER", "1"),
            ("PLOW_UNISEG", "0"),
        ];
        env.extend_from_slice(glm);
        let _env = crate::test_env::EnvScope::set(&env);
        // Exactly what `run_verified` does between parsing and emitting. `EnvScope` has already
        // installed the parsed config; this replaces it with the resolved one.
        let mut cfg = crate::emit_config::EmitConfig::from_env();
        crate::apply_production_defaults(
            &mut cfg,
            crate::emit_capabilities("glm_moe_dsa"),
            "gfx942",
            8,
            304,
        );
        crate::emit_config::install(cfg);
        let seen: Arc<Mutex<Snapshot>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let verify: crate::VerifyHook = Box::new(move |model| {
            *sink.lock().unwrap() = model
                .progs
                .iter()
                .zip(&model.prog_t)
                .map(|(p, &t)| (t, p.insts.clone(), p.stream.clone(), p.gq_stream.clone()))
                .collect();
            Ok(crate::LeanReport::skipped("structural regression test"))
        });
        glm_emit_full(
            &dir,
            81920,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            None,
            Some(&verify),
        );
        let out = seen.lock().unwrap().clone();
        out
    };

    let on: Vec<(&str, &str)> = RECIPE.iter().map(|k| (*k, "1")).collect();
    let off: Vec<(&str, &str)> = RECIPE.iter().map(|k| (*k, "0")).collect();
    let explicit = emit(&on);
    let unflagged = emit(&[]);
    let rolled_back = emit(&off);
    assert!(!explicit.is_empty());
    assert_eq!(
        explicit, unflagged,
        "an unflagged gfx942 TP8 GLM emit must be the qualified recipe"
    );
    assert_ne!(
        explicit, rolled_back,
        "the recipe must still be rollable back, and must still change the packet"
    );

    // And each knob individually: from the fully rolled-back arm, turning ONE on moves the
    // packet, so none of them is riding along inert.
    //
    // Probed from OFF rather than from ON because two of them overlap: `native_moe` is
    // `glm_moe_aiter || glm_moe_resident`, so dropping AITER out of the full recipe changes
    // nothing while RESIDENT still holds the native arm. Off-plus-one has no such shadow.
    //
    // EXT is the one exception to "from OFF": it only widens GEMM_LT_DECODE and is inert without
    // it by construction, so it is probed on top of its parent, which is the question that matters.
    for knob in RECIPE {
        let parent = match knob {
            "PLOW_GLM_GEMM_LT_DECODE_EXT" => Some("PLOW_GLM_GEMM_LT_DECODE"),
            "PLOW_GLM_SEQ_PAR_PROJ" => Some("PLOW_GLM_SEQ_PAR"),
            _ => None,
        };
        let with = |extra: Option<&str>| -> Vec<(&'static str, &'static str)> {
            RECIPE
                .iter()
                .map(|k| (*k, if Some(*k) == parent || Some(*k) == extra { "1" } else { "0" }))
                .collect()
        };
        let base = if parent.is_some() { emit(&with(None)) } else { rolled_back.clone() };
        assert_ne!(base, emit(&with(Some(knob))), "{knob}=1 changed nothing");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// `PLOW_GLM_GEMM_BLK`: at a >=2048-row bucket, q_a / kv_a / o_proj each sit alone in a segment
/// as `GemmBlkPf`; q_a quantizes `xn` and kv_a reuses it; q_absorb (a prep product) stays on
/// `GemmLtPf`; the 128-row bucket keeps the bf16 arms.
#[test]
fn glm_gemm_blk_isolates_the_checkpoint_fp8_projections() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_MLA_PREFILL", "full:128,2048"),
        ("PLOW_GLM_PLACE_PF", "1"),
        ("PLOW_GLM_MOE_AITER", "0"),
        ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
        ("PLOW_GLM_GEMM_LT_DECODE", "0"),
        ("PLOW_GLM_FOLD_LT", "0"),
        ("PLOW_GLM_GEMM_LT", "1"),
        ("PLOW_GLM_GEMM_BLK", "1"),
        ("PLOW_DECODE_BATCH_LADDER", "1"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "1"),
    ]);
    let dir = std::env::temp_dir().join(format!("plow-glm-gemm-blk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let verify: crate::VerifyHook = Box::new(|model| {
        let mut checked = 0;
        for (prog, &rows) in model.progs.iter().zip(&model.prog_t) {
            let blk = prog
                .insts
                .iter()
                .enumerate()
                .filter(|(_, d)| d.op == DevOp::GemmBlkPf as u16)
                .collect::<Vec<_>>();
            if rows != 2048 {
                assert!(blk.is_empty());
                continue;
            }
            checked += 1;
            let mut shapes = blk
                .iter()
                .map(|(_, d)| (d.i[1], d.i[2], d.i[3]))
                .collect::<Vec<_>>();
            shapes.sort_unstable();
            let mut want = [(2048, 6144, 1), (512, 6144, 0), (6144, 2048, 1)].repeat(4);
            want.sort_unstable();
            assert_eq!(shapes, want);
            let lt = prog
                .insts
                .iter()
                .filter(|d| d.op == DevOp::GemmLtPf as u16)
                .map(|d| (d.i[1], d.i[2]))
                .collect::<Vec<_>>();
            assert_eq!(lt, [(4096, 2048); 4]);
            let seg_of = |ix: usize| {
                prog.stream
                    .iter()
                    .find(|e| e.inst as usize == ix)
                    .unwrap()
                    .seg
            };
            let mut last_quant = None;
            let mut order = blk
                .iter()
                .map(|(ix, d)| (seg_of(*ix), *ix, *d))
                .collect::<Vec<_>>();
            order.sort_unstable_by_key(|(seg, ix, _)| (*seg, *ix));
            for (segment, ix, inst) in order {
                assert_eq!(inst.i[0], rows);
                assert!(segment > 0);
                if inst.i[3] == 1 {
                    last_quant = Some(inst.t[1]);
                } else {
                    assert_eq!(last_quant, Some(inst.t[1]), "reuse without its quantizer");
                }
                for entries in [&prog.stream, &prog.gq_stream] {
                    for e in entries.iter().filter(|e| e.seg == segment) {
                        assert_eq!(e.inst as usize, ix);
                        assert_eq!(
                            (e.wait_len, e.succ_len, e.flags & packet::dev::SE_XCTR),
                            (0, 0, 0)
                        );
                    }
                }
            }
        }
        assert_eq!(checked, 1);
        Ok(crate::LeanReport::skipped(
            "block-scale FP8 projection segment regression test",
        ))
    });
    glm_emit_full(
        &dir,
        4096,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        Some(packet::devbuild::L2Layout {
            sms: 38,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        }),
        Some(&verify),
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// `PLOW_GLM_SEQ_PAR`: every TP seam of a >= 2048-row prefill program is a reduce-scatter, band
/// work (residual, norm) and an all-gather of the normed rows; the 128-row rung and decode keep
/// their collectives. Off, no result slot and no split collective is emitted.
#[test]
fn glm_seq_par_splits_every_prefill_seam_on_the_owned_band() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let dir = std::env::temp_dir().join(format!("plow-glm-seq-par-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    fn is(d: &crate::DevInst, op: DevOp) -> bool {
        d.op == op as u16
    }
    let emit = |verify: crate::VerifyHook| {
        glm_emit_full(
            &dir,
            4096,
            dir.join("model.pkt").to_str().unwrap(),
            304,
            8,
            true,
            true,
            "gfx942",
            None,
            Some(&verify),
        )
    };
    let base_env = [
        ("PLOW_MLA_PREFILL", "full:128,2048"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_UNISEG", "0"),
    ];

    let ran = std::sync::Arc::new(AtomicBool::new(false));
    {
        let _env = crate::test_env::EnvScope::set(&base_env);
        let ran = ran.clone();
        emit(Box::new(move |model| {
            assert!(model.tensors.iter().all(|t| t.name != "act.h2_tp"));
            assert!(model
                .progs
                .iter()
                .flat_map(|p| &p.insts)
                .all(|d| !is(d, DevOp::XReduceScatter) && !is(d, DevOp::XAllGather)));
            ran.store(true, Ordering::SeqCst);
            Ok(crate::LeanReport::skipped("structural regression test"))
        }));
    }
    assert!(ran.swap(false, Ordering::SeqCst));

    let mut env = base_env.to_vec();
    env.push(("PLOW_GLM_SEQ_PAR", "1"));
    let _env = crate::test_env::EnvScope::set(&env);
    let ran2 = ran.clone();
    emit(Box::new(move |model| {
        let (t, h, tp) = (2048u32, 6144u32, 8u32);
        let slot_b = t * h * 2;
        let tn = |ix: u32| model.tensors[ix as usize].name.as_str();
        let count = |p: &packet::devbuild::Program, op: DevOp| {
            p.insts.iter().filter(|d| is(d, op)).count()
        };
        // The 128-row rung keeps its two-shots; decode keeps its one-shots.
        assert_eq!(count(&model.progs[0], DevOp::XReduceTwoShot), 8);
        for p in model.progs.iter().filter(|p| p.insts.iter().any(|d| is(d, DevOp::XReduce))) {
            assert_eq!(count(p, DevOp::XReduceScatter) + count(p, DevOp::XAllGather), 0);
        }
        let p = &model.progs[1];
        assert_eq!(count(p, DevOp::XReduceTwoShot), 0);
        assert_eq!(count(p, DevOp::XReduceScatter), 8, "4 attention + 3 dense + 1 MoE seams");
        assert_eq!(count(p, DevOp::XAllGather), 9, "4 input norms + 4 post-attn norms + final");
        let mut gates = std::collections::BTreeSet::new();
        for d in &p.insts {
            if is(d, DevOp::XReduceScatter) {
                assert_eq!((d.i[0], d.i[1]), (t * h, tp));
                assert!(d.i[2] == 0 || d.i[2] == slot_b);
                assert!(gates.insert(d.i[3]), "gate {} reused", d.i[3]);
            } else if is(d, DevOp::XAllGather) {
                assert_eq!((d.i[0], d.i[4], d.i[5]), (t * h, tp, 3 * slot_b));
                assert!(["act.xn", "act.xn2"].contains(&tn(d.t[0])), "gathers {}", tn(d.t[0]));
                assert!(gates.insert(d.i[3]), "gate {} reused", d.i[3]);
            }
        }
        // Every norm feeding a gather and every seam residual runs on the band.
        let band_norms: Vec<_> = p
            .insts
            .iter()
            .filter(|d| is(d, DevOp::RmsNorm) && tn(d.t[0]) == "act.h2_tp@band2048")
            .collect();
        assert_eq!(band_norms.len(), 9);
        assert!(band_norms
            .iter()
            .all(|d| d.i[0] == t / tp && tn(d.t[1]).ends_with("@band2048")));
        let res: Vec<_> = p.insts.iter().filter(|d| is(d, DevOp::Residual)).collect();
        assert_eq!(res.len(), 8);
        for d in res {
            assert_eq!(d.i[0], t / tp * h);
            assert!((0..3).all(|k| tn(d.t[k]).ends_with("@band2048")));
        }
        // Band views: an eighth of a base bound before them, all in the blob's table.
        let mut views = 0;
        for (ix, v) in model.tensors.iter().enumerate() {
            let Some(base) = v.name.strip_suffix("@band2048") else {
                continue;
            };
            let bix = model.tensors.iter().position(|x| x.name == base).expect("band base");
            assert!(bix < ix);
            assert_eq!(model.tensors[bix].bytes, tp as u64 * v.bytes, "{}", v.name);
            views += 1;
        }
        assert_eq!(views, 6, "x, xnext, xmid, og_tp, dg_tp, h2_tp");
        for name in ["act.h2_tp", "act.xe_tp", "act.rt_tp"] {
            assert!(model.tensors.iter().any(|x| x.name == name), "{name} missing");
        }
        for q in &model.progs {
            for d in &q.insts {
                assert!(d
                    .t
                    .iter()
                    .all(|&h| h == TENSOR_NONE || (h as usize) < model.tensors.len()));
            }
        }
        ran2.store(true, Ordering::SeqCst);
        Ok(crate::LeanReport::skipped("structural regression test"))
    }));
    assert!(ran.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

/// `PLOW_GLM_SEQ_PAR_PROJ`: on a full-indexer layer the input norm, q_a (+ its norm), kv_a,
/// k_rope and the indexer's k / weights projections all run on the owned band; one gather
/// carries q_a-normed / kv_a / k_rope, a second the indexer k / weights. No packet reads the
/// T-row `xn`.
#[test]
fn glm_seq_par_proj_moves_the_entry_projections_onto_the_band() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_SEQ_PAR", "1"),
        ("PLOW_GLM_SEQ_PAR_PROJ", "1"),
        ("PLOW_GLM_DSA_PF", "1"),
        ("PLOW_GLM_INDEX_TP", "1"),
        ("PLOW_GLM_FP8_KV", "1"),
        ("PLOW_UNISEG", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    let (ctx, t, tp) = (81920u32, 8192u32, 8u32);
    let tb = t / tp;
    let mut decl = Builder::new(304);
    let n = declare_glm_rows_batched(&mut decl, &c, ctx, &[0], t, 1, MoeEnc::Fp8Blk);
    assert_ne!(n.ug_tp, TENSOR_NONE);
    let mut b = Builder::new(304);
    b.adopt_tensors(decl.tensors());
    let all = b.all();
    let mut xgate = 0;
    emit_glm_mla_prefill(&mut b, &c, &n, 0, ctx, t, MoeEnc::Fp8Blk, n.x, &[], false, &mut xgate,
        &all, None);
    let p = b.finish();
    let name = |h: u32| p.tensors[h as usize].name.as_str();
    let is = |d: &crate::DevInst, op: DevOp| d.op == op as u16;
    assert!(p.insts.iter().all(|d| d.t[1] != n.xn), "a packet reads the T-row xn");
    assert!(p.insts.iter().any(|d| is(d, DevOp::IndexTpPf)));
    let ag: Vec<_> = p.insts.iter().filter(|d| is(d, DevOp::XAllGather)).collect();
    assert_eq!(ag.len(), 3, "entry projections, indexer k/w, post-attention norm");
    let slot_b = n.slot_b;
    assert_eq!(
        (ag[0].t[0], ag[0].t[1], ag[0].t[2]),
        (n.qlat, n.ckvraw, n.krr)
    );
    assert_eq!(
        [ag[0].i[0], ag[0].i[1], ag[0].i[2], ag[0].i[5], ag[0].i[6], ag[0].i[7]],
        [t * c.q_lora, t * c.kv_lora, t * c.qk_rope, 3 * slot_b, 4 * slot_b, 5 * slot_b]
    );
    assert_eq!((ag[1].t[0], ag[1].t[1]), (n.kidx_pf, n.widx_pf));
    assert_eq!(
        [ag[1].i[0], ag[1].i[1], ag[1].i[5], ag[1].i[6]],
        [t * c.index_dim, t * c.index_heads, 2 * slot_b, 0]
    );
    assert_eq!(ag[1].t[2], TENSOR_NONE);
    assert_eq!(ag[2].t[0], n.xn2);
    // The five band projections and the two band norms carry the band's rows.
    let fam = crate::gemm_family_ops();
    let band_gemms: Vec<_> = p
        .insts
        .iter()
        .filter(|d| fam.contains(&d.op) && name(d.t[1]).ends_with("@band8192"))
        .collect();
    assert_eq!(band_gemms.len(), 5);
    assert!(band_gemms.iter().all(|d| d.i[0] == tb && name(d.t[0]).contains("@band8192")));
    let norms: Vec<_> = p
        .insts
        .iter()
        .filter(|d| is(d, DevOp::RmsNorm) && name(d.t[0]).contains("@band8192"))
        .collect();
    assert_eq!(norms.len(), 3, "input norm, q_a norm, post-attention norm");
    assert!(norms.iter().all(|d| d.i[0] == tb));
    // Every slot-backed view is an eighth of the gathered array, rank-strided from its slot.
    for (h, rows) in [
        (".q", c.q_lora),
        (".kv", c.kv_lora),
        (".kr", c.qk_rope),
        (".ki", c.index_dim),
        (".w", c.index_heads),
    ] {
        let v = p.tensors.iter().find(|x| x.name.ends_with(&format!("@band8192{h}"))).unwrap();
        assert_eq!(v.bytes, tb as u64 * rows as u64 * 2, "{}", v.name);
    }
}

/// `PLOW_GLM_SEQ_PAR_PROJ` on a MoE layer: the router score and top-k run on the band and the
/// route table is gathered (slot 5) before the full-T align.
#[test]
fn glm_seq_par_proj_routes_on_the_band() {
    let _guard = crate::test_env::env_guard();
    let _target = crate::EmitAmdGuard::set(true);
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_SEQ_PAR", "1"),
        ("PLOW_GLM_SEQ_PAR_PROJ", "1"),
        ("PLOW_GLM_MOE_AITER", "1"),
        ("PLOW_GLM_FP8_KV", "1"),
        ("PLOW_UNISEG", "0"),
    ]);
    let mut c = glm_ref_cfg();
    c.tp = 8;
    let (ctx, t, tp) = (81920u32, 8192u32, 8u32);
    let tb = t / tp;
    let mut decl = Builder::new(304);
    let n = declare_glm_rows_batched(&mut decl, &c, ctx, &[3], t, 1, MoeEnc::Fp8Blk);
    let mut b = Builder::new(304);
    b.adopt_tensors(decl.tensors());
    let all = b.all();
    let mut xgate = 0;
    let c_rn2 = emit_glm_mla_prefill(&mut b, &c, &n, 0, ctx, t, MoeEnc::Fp8Blk, n.x, &[], false,
        &mut xgate, &all, None);
    emit_glm_moe_ffn_prefill(&mut b, &c, &n, 0, t, MoeEnc::Fp8Blk, n.xnext, c_rn2, &mut xgate,
        &all, false);
    let p = b.finish();
    let name = |h: u32| p.tensors[h as usize].name.as_str();
    let is = |d: &crate::DevInst, op: DevOp| d.op == op as u16;
    let router = p.insts.iter().find(|d| is(d, DevOp::MoeRouterTopkPf)).unwrap();
    assert_eq!(router.i[4], tb);
    assert!(name(router.t[0]).ends_with("@band8192.rt"));
    assert!(name(router.t[1]).ends_with("@band8192"));
    let score = p.insts.iter().find(|d| d.t[0] == router.t[1]).unwrap();
    assert_eq!(score.i[0], tb);
    assert_eq!(name(score.t[1]), "act.xn2@band8192");
    let ag: Vec<_> = p.insts.iter().filter(|d| is(d, DevOp::XAllGather)).collect();
    assert_eq!(ag.len(), 3, "entry projections, post-attention norm, route table");
    assert_eq!((ag[2].t[0], ag[2].i[0], ag[2].i[5]), (n.tab, t * c.top_k * 4, 5 * n.slot_b));
    assert!(p.insts.iter().filter(|d| is(d, DevOp::MoeAlignPf)).all(|d| d.i[0] == t));
    assert_eq!(p.insts.iter().filter(|d| is(d, DevOp::XReduceScatter)).count(), 2);
}
