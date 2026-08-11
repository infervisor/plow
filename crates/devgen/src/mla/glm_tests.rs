//! The GLM-5.2 (GlmMoeDsa) single-layer emit is the FIRST milestone-1 gate: the emitted op
//! sequence must be identical to the 34-op MoE block that runtime/tests/
//! glm52_real_block_gfx950_test.c validated on gfx950 against the HF oracle (real 256 experts,
//! real [128,128] block-fp8 scales — see the design notes, "B4-CORE DONE"). Asserting op-for-op
//! equality here, offline, means the emitted layer inherits that passing GPU result. No GPU, no
//! weights — a pure structural equivalence proof, exactly as the Gemma pick_tile tests lock in
//! the tile choice offline.
use super::*;

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
        rope_scale: RopeScale::None,
        prefix: "model.".into(),
        tp: 1,
        ep: false,
        group: false,
        index_heads: 32,
        index_dim: 128,
        index_topk: 2048,
        // indexer_types[0..4] = full,full,full,shared (real GLM-5.2 pattern); irrelevant to these
        // ctx=512 offline tests (DSA is gated OFF at ctx<=2048) but set for completeness.
        indexer_full: vec![true, true, true, false],
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
    assert_eq!(
        emitted_ops(false),
        ref_sequence(false),
        "bf16 op sequence != reference"
    );
}

#[test]
fn glm_block_matches_reference_fp8() {
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
fn glm_dsa_full_layer_emits_indexer() {
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

#[test]
fn glm_dense_block_sequence() {
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

/// The MLA flash-decode split factor is the ctx-scaled cost optimum, capped by the ACTUAL
/// per-rank chip-fill `fill = ceil(n_cu / (nh_l/GF))` and the KV-tile count. `glm_nsplit` takes
/// nh_l (= n_head/tp) so the cap is correct under TP/EP — the pre-fix bug sized it from the
/// global n_head=64, pinning the cap to tp=1's fill regardless of TP. Asserts the caps and the
/// measured (MI350X mla_perf) chain optima: ns~16 up to 8k, ns~64 at 32k.
#[test]
fn glm_nsplit_is_ctx_scaled_and_capped_per_rank() {
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
