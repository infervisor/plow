//! Kimi K2.7 / DeepSeek MLA+MoE `--block` extraction (M3).
//! Kimi REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a cfg that holds
//! the DSA gate off (`has_dsa=false`) — so a Kimi block is the SAME op sequence as a GLM block
//! BELOW the DSA crossover, minus every indexer artifact: no DSA scratch, FlashMlaDecode (never
//! FlashGatherDecode) at ANY ctx, and a descriptor with no dsa_role / no index_* dims. These
//! synthetic-CPU tests are the only verification available on this box (no Kimi checkpoint, no
//! transformers → no real blob, no GPU parity). They lock in the op sequence + descriptor exactly
//! as glm_tests does for GLM.
use super::*;

/// Synthetic small Kimi cfg (structurally faithful: DeepSeek-schema MLA + MoE, first_k_dense=1
/// so layer 0 is dense and 1+ are MoE, has_dsa=false). Real K2.7 geometry is hidden 7168 / 64
/// heads / kv_lora 512 / q_lora 1536 / qk_nope 128 / qk_rope 64 / v_head 128 / 384 exp / top_k 8
/// / moe_inter 2048; the shape logic is dim-agnostic, so small dims exercise the same emit.
fn kimi_ref_cfg() -> GlmCfg {
    GlmCfg {
        layers: 4,
        hidden: 256,
        heads: 4,
        kv_lora: 64,
        q_lora: 96,
        qk_nope: 32,
        qk_rope: 16,
        v_head: 32,
        vocab: 1000,
        eps: 1e-5,
        n_exp: 16,
        top_k: 4,
        n_group: 1,
        topk_group: 1,
        moe_inter: 128,
        dense_inter: 256,
        first_k_dense: 1,
        route_scale: 2.5,
        attn_scale: (48f32).powf(-0.5), // 1/sqrt(qk_nope+qk_rope = 48)
        rope_theta: Some(50_000.0),
        prefix: "model.".into(),
        tp: 1,
        ep: false,
        group: false,
        // Indexer fields are inert under has_dsa=false (never read); set placeholders.
        index_heads: 8,
        index_dim: 32,
        index_topk: 64,
        index_kpool: 1,
        indexer_full: Vec::new(), // Kimi/DeepSeek config has no `indexer_types`
        softmax_layers: vec![],
        has_dsa: false,
    }
}

fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>, arch: MlaArch) -> Vec<u16> {
    let (m, _d) = glm_build_block(c, ctx, 256, block, true, "kimi-ref", arch);
    m.progs[0].insts.iter().map(|d| d.op).collect()
}

/// Expected MoE-block op sequence: shared MLA (12 ops) + router split (2) + shared expert (2) +
/// top_k×(glu, down) + MoeCombine. IDENTICAL shape to glm_tests::ref_sequence but parameterized
/// on top_k — the reuse the arch is built on.
fn kimi_moe_sequence(use_fp8: bool, top_k: usize) -> Vec<u16> {
    use DevOp::*;
    let (glu, down) = if use_fp8 {
        (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
    } else {
        (MoeExpertGlu, MoeExpertDown)
    };
    let mut ops = vec![
        RmsNorm,        // input_layernorm
        GemvQkv,        // FUSED A: q_a + kv_a + k_rope down-projections
        RmsNorm,        // q_a_layernorm
        GemvQkv,        // FUSED G: q_absorb + q_rope down
        HeadNormRope,   // q_rope dynamic interleaved RoPE
        RmsNorm,        // kv_a_layernorm -> latent cache
        HeadNormRope,   // k_rope dynamic RoPE -> rope cache
        FlashMlaDecode, // MLA flash (NO DSA gather)
        MlaMergeFold,   // fused latent merge + W_uv fold
        Gemv,           // o_proj
        Residual,       // post-attn residual
        RmsNorm,        // post_attention_layernorm
        Gemv,           // router score GEMV
        MoeRouterTopk,  // router top-k select
        GemvGlu,        // shared expert gate|up
        Gemv,           // shared expert down
    ];
    for _ in 0..top_k {
        ops.push(glu);
        ops.push(down);
    }
    ops.push(MoeCombine);
    ops.into_iter().map(|o| o as u16).collect()
}

/// Expected DENSE-block op sequence: shared MLA (12) + block-fp8 SwiGLU (gate/up + down) +
/// residual. The GLM emitter's dense FFN is block-fp8 regardless of `use_fp8`, so Kimi's dense
/// layer (layer 0) inherits those opcodes.
fn kimi_dense_sequence() -> Vec<u16> {
    use DevOp::*;
    vec![
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
        DenseGluFp8Blk,
        GemvFp8Blk,
        Residual,
    ]
    .into_iter()
    .map(|o| o as u16)
    .collect()
}

/// A single MoE-layer `--block 1` extraction emits EXACTLY the MLA+MoE block — no embed, no
/// final-norm/lm_head/argmax tail, `act.x` in and out.
#[test]
fn kimi_block_extract_matches_mla_moe_sequence() {
    let c = kimi_ref_cfg();
    assert_eq!(
        block_ops(&c, 512, 1..2, MlaArch::Kimi),
        kimi_moe_sequence(true, 4),
        "single-block --block 1 op sequence != MLA+MoE block (fp8)"
    );
    assert_eq!(
        {
            let (m, _) = glm_build_block(&c, 512, 256, 1..2, false, "kimi-ref", MlaArch::Kimi);
            m.progs[0].insts.iter().map(|d| d.op).collect::<Vec<_>>()
        },
        kimi_moe_sequence(false, 4),
        "bf16 op sequence != MLA+MoE block"
    );
}

/// Descriptor for a Kimi MoE block: arch tag, mla_attn+moe_ffn kind, NO dsa_role, MLA+MoE dims,
/// NO index_* dims, KV latent (ckv/krot) carried state only, decode-only programs.
#[test]
fn kimi_block_descriptor_moe() {
    let c = kimi_ref_cfg();
    let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "kimi-k2.7", MlaArch::Kimi);
    assert_eq!(d.arch, "kimi_mla_moe");
    assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
    assert_eq!(d.dtype, "fp8");
    assert_eq!(d.dsa_role, None, "plain MLA has no DSA indexer role");
    assert_eq!(d.dims.heads, Some(4));
    assert_eq!(d.dims.kv_lora, Some(64));
    assert_eq!(d.dims.q_lora, Some(96));
    assert_eq!(d.dims.n_exp, Some(16));
    assert_eq!(d.dims.top_k, Some(4));
    assert_eq!(d.dims.shared_exp, Some(1));
    assert_eq!(d.dims.moe_inter, Some(128));
    assert_eq!(d.dims.index_heads, None, "no DSA => no index dims");
    assert_eq!(d.dims.index_dim, None);
    assert_eq!(d.dims.index_topk, None);
    assert_eq!(d.layer, 1);
    assert_eq!(d.weights.prefix, "model.layers.1.");
    assert_eq!(
        d.outputs[0].name, "act.xnext",
        "odd layer count -> act.xnext"
    );
    // Decode-only unless prefill buckets are asked for; `kimi_prefill_descriptor_lists_buckets`
    // covers the opted-in shape.
    assert!(
        d.programs.prefill_buckets.is_empty(),
        "GLM/Kimi block emit is decode-only unless prefill buckets are requested"
    );
    assert_eq!(d.programs.decode_t, 1);
    // KV latent carried state only — no kidx, no dsa_indices.
    assert_eq!(d.carried_state.len(), 1);
    assert_eq!(d.carried_state[0].role, "kv");
    assert_eq!(d.carried_state[0].layout, "mla_latent");
    assert_eq!(
        d.carried_state[0].tensors,
        vec!["kv.1.ckv", "kv.1.krot"],
        "MLA latent caches only (no indexer kidx)"
    );
}

/// Descriptor for a Kimi DENSE block (layer 0, first_k_dense=1): dense_ffn kind, no MoE dims,
/// MLA dims still present.
#[test]
fn kimi_block_descriptor_dense() {
    let c = kimi_ref_cfg();
    let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "kimi-ref", MlaArch::Kimi);
    assert_eq!(d.kind, vec!["mla_attn", "dense_ffn"]);
    assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
    assert_eq!(d.dims.moe_inter, None);
    assert_eq!(d.dims.kv_lora, Some(64), "MLA dims still present");
    assert_eq!(d.dsa_role, None);
}

/// A multi-layer `--block 0..2` extraction chains dense layer 0 then MoE layer 1, and the
/// residual ping-pong lands the output back in `act.x` after an even layer count.
#[test]
fn kimi_block_multi_layer_chains() {
    let c = kimi_ref_cfg();
    let mut want = kimi_dense_sequence(); // layer 0 (dense)
    want.extend(kimi_moe_sequence(true, 4)); // layer 1 (MoE)
    assert_eq!(
        block_ops(&c, 512, 0..2, MlaArch::Kimi),
        want,
        "2-layer block != dense++moe"
    );
    let (_, d) = glm_build_block(&c, 512, 256, 0..2, true, "kimi-ref", MlaArch::Kimi);
    assert_eq!(d.outputs[0].name, "act.x", "even layer count -> act.x out");
    assert_eq!(d.layer, 0, "descriptor.layer = block start");
}

/// The DSA gate is held OFF at EVERY ctx (has_dsa=false): even at 131072 (well past GLM's 65536
/// crossover) the block emits FlashMlaDecode — never FlashGatherDecode — and carries no
/// dsa_indices / no kidx. This is what "reuse GLM MLA without DSA" means structurally.
#[test]
fn kimi_no_dsa_at_long_ctx() {
    let c = kimi_ref_cfg();
    let ops = block_ops(&c, 131072, 1..2, MlaArch::Kimi);
    assert!(
        ops.contains(&(DevOp::FlashMlaDecode as u16)),
        "dense MLA flash present"
    );
    assert!(
        !ops.contains(&(DevOp::FlashGatherDecode as u16)),
        "no DSA gather flash for Kimi"
    );
    let (_, d) = glm_build_block(&c, 131072, 256, 1..2, true, "kimi-ref", MlaArch::Kimi);
    assert_eq!(d.dsa_role, None);
    assert!(
        d.carried_state.iter().all(|s| s.role != "dsa_indices"),
        "no dsa_indices carried"
    );
    assert!(
        d.carried_state[0]
            .tensors
            .iter()
            .all(|t| !t.contains("kidx")),
        "no indexer kidx cache"
    );
}

// ===== PREFILL buckets (FLASH_MLA_PREFILL + MLA_MERGE_FOLD at T rows) ======================
// These are the offline gate on the arm whose ABSENCE meant Kimi K2.7 / DeepSeek / GLM-5.2
// could decode on gfx950 but could not prefill through their own attention. Same discipline as
// the decode tests above: no GPU, no weights — pin the emitted stream and the operand fields
// that select the kernel body, so a hardware run inherits a checked packet rather than a guess.

/// A cfg with 8 heads, so tp ∈ {1,2,4,8} all divide the head count (the real models have 64).
/// A TP sweep fixture whose head shard stays EMITTABLE at every `tp` it is used with.
///
/// This was `heads = 8`, and the sweeps below run `tp` up to 8 — so the tp=8 arm had
/// `nh_l = 1`. The smallest GF the interpreter instantiates is 2, so that arm was emitting a
/// `FlashMlaPrefill` with `n_grp = nh_l / GF = 0`: `n_work = 0`, the kernel's work loop never
/// executes, no partial is ever written, and `MlaMergeFold` consumes uninitialised `opart`.
/// The test asserted the packet's OPERANDS and they were all correct, so it passed for a
/// packet that computes nothing — the assertion was on `i[1]`, never on whether the shape was
/// runnable.
///
/// `require_gf_divides` now refuses that shape at emit. 16 heads keeps the sweep's four `tp`
/// points (nh_l 16/8/4/2) inside what the kernel can express; `nh_l = 1` has its own
/// `#[should_panic]` test below, because it is a REAL limitation of the current arm set and
/// not a fixture detail.
fn kimi_tp_cfg(tp: u32) -> GlmCfg {
    let mut c = kimi_ref_cfg();
    c.heads = 16;
    c.tp = tp;
    c
}

/// `heads == tp` gives `nh_l = 1`, and NO instantiated GF can express it.
///
/// Found by `require_gf_divides` firing on a test fixture that had been passing. Both
/// selectors fall back to GF=2 when nothing fits (`glm_gf`'s `unwrap_or(2)`, `glm_gf_prefill`'s
/// `else` branch), and 2 > 1, so this was a silent divide-to-zero on BOTH the decode and
/// prefill paths — not a new restriction, a newly-visible one. A model sharded until each rank
/// owns a single MLA head must be refused at emit or it produces attention from nothing.
#[test]
#[should_panic(expected = "does not divide this rank's head shard")]
fn a_single_head_per_rank_cannot_be_expressed_by_any_gf() {
    let mut c = kimi_ref_cfg();
    c.heads = 8;
    c.tp = 8; // nh_l = 1
    pf_block(&c, 512, &[128]);
}

/// Build one MLA block with prefill buckets and return (model, descriptor).
fn pf_block(c: &GlmCfg, ctx: u32, pf: &[u32]) -> (Model, plow_asset::BlockDescriptor) {
    glm_build_block_pf(
        c,
        ctx,
        256,
        1..2,
        true,
        "kimi-ref",
        MlaArch::Kimi,
        pf,
        PrefillScope::Attn,
        MoeEnc::Fp8Blk,
    )
}

fn find_op(p: &packet::devbuild::Program, op: DevOp) -> &packet::dev::DevInst {
    p.insts
        .iter()
        .find(|d| d.op == op as u16)
        .unwrap_or_else(|| panic!("{op:?} not emitted"))
}

/// The co-resident shared gate/up halves must be DISJOINT and cover the slice. Overlap is
/// silent: the packets still compute the right numbers, they just run one after the other on
/// the shared workgroups, which is the entire cost `glm_shared_glu_split` exists to remove.
#[test]
fn glm_shared_glu_halves_are_disjoint_and_total() {
    for n in [2usize, 3, 8, 32, 224, 256] {
        let cus: Vec<u32> = (0..n as u32).collect();
        let (g, u) = glm_glu_halves(&cus);
        assert!(
            !g.is_empty() && !u.is_empty(),
            "n={n}: an empty CU set is not emittable"
        );
        assert!(g.iter().all(|c| !u.contains(c)), "n={n}: halves overlap");
        let mut all: Vec<u32> = g.iter().chain(u.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, cus, "n={n}: the halves must cover the slice exactly");
    }
    // A 1-CU slice cannot be split; the fallback is the serial arrangement, not an empty set.
    assert_eq!(glm_glu_halves(&[7]), (vec![7], vec![7]));
}

/// The prefill program IS the MLA attention sub-block at T rows, and it is built from the GEMM
/// family — not one decode-shaped op survives into it.
#[test]
fn mla_prefill_bucket_op_sequence() {
    use DevOp::*;
    let c = kimi_ref_cfg();
    let (m, _) = pf_block(&c, 512, &[128]);
    // Buckets FIRST, decode LAST — manifest.rs and plowrt both key off that order.
    assert_eq!(m.progs.len(), 2, "one prefill bucket + decode");
    assert_eq!(m.prog_t, vec![128, 1]);
    let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
    assert_eq!(
        ops.len(),
        15,
        "norm + 3 down GEMMs + norm + 2 q GEMMs + 2 rope + norm + \
                                   flash + merge-fold + o_proj + residual + norm"
    );
    let tiled = [Gemm as u16, GemmMed as u16, GemmSmall as u16];
    // Positions 1,2,3 (q_a/kv_a/k_rope), 5,6 (q_absorb/q_rope) and 12 (o_proj) are the tiled
    // GEMMs that replace decode's GemvQkv fusions and its o_proj Gemv.
    for i in [1usize, 2, 3, 5, 6, 12] {
        assert!(
            tiled.contains(&ops[i]),
            "op {i} = {:?} is not a tiled GEMM arm",
            ops[i]
        );
    }
    assert_eq!(
        [ops[0], ops[4], ops[7], ops[8], ops[9], ops[10], ops[11], ops[13], ops[14]],
        [
            RmsNorm as u16,
            RmsNorm as u16,
            HeadNormRope as u16,
            RmsNorm as u16,
            HeadNormRope as u16,
            FlashMlaPrefill as u16,
            MlaMergeFold as u16,
            Residual as u16,
            RmsNorm as u16
        ]
    );
    // No decode-family op may leak into a prefill bucket: those bodies are compiled into the
    // DECODE object, and the AMD dispatch default silently no-ops an opcode with no arm.
    for bad in [
        Gemv,
        GemvQkv,
        GemvGlu,
        GemvFp8Blk,
        FlashMlaDecode,
        FlashGatherDecode,
        MoeRouterTopk,
        MoeExpertGluFp8Blk,
        MoeCombine,
        DenseGluFp8Blk,
    ] {
        assert!(
            !ops.contains(&(bad as u16)),
            "{bad:?} leaked into the prefill bucket"
        );
    }
}

/// The flash + fold operand fields, which are what select the kernel body and its work
/// decomposition. `i[4] = n_tok` (not nsplit) and `nsplit = 1` are the prefill PRECONDITION:
/// under a per-token causal bound an early token's later splits are empty and an empty split
/// emits l=0 for the merge to divide by.
#[test]
fn mla_prefill_flash_and_fold_operands() {
    let c = kimi_ref_cfg();
    let (m, _) = pf_block(&c, 512, &[128]);
    let p = &m.progs[0];
    let fl = find_op(p, DevOp::FlashMlaPrefill);
    assert_eq!(fl.i[0], 1, "n_batch: one sequence per prefill chunk");
    assert_eq!(
        fl.i[1], c.heads,
        "n_head = per-rank head count (tp=1 => all)"
    );
    assert_eq!(fl.i[2], 512, "kv_stride = ctx");
    assert_eq!(fl.i[3], 0, "window 0 = full causal");
    assert_eq!(
        fl.i[4], 128,
        "i[4] carries n_tok on the prefill arm, not nsplit"
    );
    assert_eq!(fl.i[5], KV_MASK_NONE);
    assert!(
        matches!(fl.i[7], 2 | 4),
        "GF must be an instantiated prefill body, got {}",
        fl.i[7]
    );
    assert_eq!(fl.t[6], c_kvlen(&m), "kv_len operand bound");
    let fold = find_op(p, DevOp::MlaMergeFold);
    assert_eq!(
        fold.i[0], 128,
        "the token axis folds into n_batch: partials are (b*n_tok+t)"
    );
    assert_eq!(fold.i[1], c.heads);
    assert_eq!(fold.i[2], c.v_head);
    assert_eq!(fold.i[4], 1, "nsplit MUST be 1 on the prefill arm");
}

/// `in.kvlen`'s handle, so the operand check above is against the real tensor, not an index.
fn c_kvlen(m: &Model) -> u32 {
    m.tensors
        .iter()
        .position(|t| t.name == "in.kvlen")
        .expect("in.kvlen declared") as u32
}

/// EVERY head-dimensioned prefill field is the PER-RANK count nh_l = n_head/tp. Sizing any of
/// them from the global head count is the measured tp=8 bug the `glm_nsplit` header records —
/// the flash ran on 32 of 256 CUs — and prefill has strictly more work items than decode, so it
/// would be just as invisible here.
#[test]
fn mla_prefill_tp_shapes_scale_with_tp() {
    for tp in [1u32, 2, 4, 8] {
        let c = kimi_tp_cfg(tp);
        let nh_l = c.heads / tp;
        let (m, _) = pf_block(&c, 512, &[128]);
        let p = &m.progs[0];
        assert_eq!(
            find_op(p, DevOp::FlashMlaPrefill).i[1],
            nh_l,
            "tp={tp} flash n_head"
        );
        assert_eq!(
            find_op(p, DevOp::MlaMergeFold).i[1],
            nh_l,
            "tp={tp} fold n_head"
        );
        // o_proj is row-parallel: K = this rank's nh_l*v_head lanes, N = full hidden.
        let o = p
            .insts
            .iter()
            .rev()
            .find(|d| {
                matches!(d.op, x if x == DevOp::Gemm as u16
                                        || x == DevOp::GemmMed as u16
                                        || x == DevOp::GemmSmall as u16)
            })
            .expect("o_proj GEMM");
        assert_eq!(o.i[1], c.hidden, "tp={tp} o_proj N = full hidden");
        assert_eq!(
            o.i[2],
            nh_l * c.v_head,
            "tp={tp} o_proj K = per-rank head lanes"
        );
        // The q projections are column-parallel by head.
        let qa = &p.insts[5];
        assert_eq!(qa.i[1], nh_l * c.kv_lora, "tp={tp} q_absorb N");
    }
}

/// PREFILL all-reduces the [T,hidden] o_proj partial with the TWO-SHOT collective, not decode's
/// one-shot: the partial is bandwidth-bound at T rows, so the two-shot moves ~tp/2x less over
/// the fabric. tp=1 emits no collective at all.
#[test]
fn mla_prefill_tp_emits_two_shot_allreduce() {
    let (m1, _) = pf_block(&kimi_tp_cfg(1), 512, &[128]);
    assert!(
        !m1.progs[0]
            .insts
            .iter()
            .any(|d| d.op == DevOp::XReduceTwoShot as u16 || d.op == DevOp::XReduce as u16),
        "tp=1 emits no collective"
    );
    for tp in [2u32, 4, 8] {
        let c = kimi_tp_cfg(tp);
        let (m, _) = pf_block(&c, 512, &[128]);
        let p = &m.progs[0];
        let xr = find_op(p, DevOp::XReduceTwoShot);
        assert_eq!(xr.i[0], 128 * c.hidden, "tp={tp} reduces t*hidden elements");
        assert_eq!(xr.i[1], tp, "tp={tp} n_gpu");
        assert!(
            !p.insts.iter().any(|d| d.op == DevOp::XReduce as u16),
            "tp={tp}: prefill must not use decode's one-shot all-reduce"
        );
    }
}

/// EP (expert-parallel) survives on the DECODE half that the prefill buckets sit alongside:
/// attention stays TP-sharded (nh_l above) while the ROUTED experts are distributed WHOLE
/// across ranks — full `moe_inter` per expert, not the TP slice — so a rank never runs a
/// CU-starved fragment of an expert.
#[test]
fn mla_ep_keeps_routed_experts_whole_beside_prefill() {
    let mut c = kimi_tp_cfg(4);
    c.ep = true;
    let (m, _) = pf_block(&c, 512, &[128]);
    let dec = m.progs.last().unwrap();
    let glu = find_op(dec, DevOp::MoeExpertGluFp8Blk);
    assert_eq!(
        glu.i[1], c.moe_inter,
        "EP: routed expert keeps the FULL moe_inter"
    );
    // The SHARED expert stays TP-sharded — that is the floor EP deliberately does not touch.
    let sh = find_op(dec, DevOp::GemvGlu);
    assert_eq!(
        sh.i[1],
        c.moe_inter / 4,
        "shared expert stays TP-sharded under EP"
    );
    // And the attention half of the SAME asset is still per-rank sharded.
    assert_eq!(
        find_op(&m.progs[0], DevOp::FlashMlaPrefill).i[1],
        c.heads / 4
    );

    let mut c_tp = c.clone();
    c_tp.ep = false;
    let (m2, _) = pf_block(&c_tp, 512, &[128]);
    assert_eq!(
        find_op(m2.progs.last().unwrap(), DevOp::MoeExpertGluFp8Blk).i[1],
        c.moe_inter / 4,
        "without EP the routed expert IS TP-sliced"
    );
}

/// One tensor table serves every program, so the row-dimensioned activations are sized for the
/// widest bucket. Under-sizing them is an out-of-bounds DEVICE write, not a slowdown.
#[test]
fn mla_prefill_widens_the_shared_tensor_table() {
    let c = kimi_ref_cfg();
    let bytes = |m: &Model, n: &str| {
        m.tensors
            .iter()
            .find(|t| t.name == n)
            .unwrap_or_else(|| panic!("{n}"))
            .bytes
    };
    let (dec_only, _) = pf_block(&c, 512, &[]);
    let (with_pf, _) = pf_block(&c, 512, &[128, 512]);
    let h = c.hidden as u64;
    assert_eq!(bytes(&dec_only, "act.x"), h * 2, "decode-only: one row");
    assert_eq!(
        bytes(&with_pf, "act.x"),
        512 * h * 2,
        "sized for the WIDEST bucket"
    );
    assert_eq!(bytes(&with_pf, "act.xn2"), 512 * h * 2);
    assert_eq!(
        bytes(&with_pf, "act.oat"),
        512 * (c.heads * c.v_head) as u64 * 2
    );
    // The flash partials are [t][head][nsplit][DK] with nsplit=1 at prefill and ns at decode —
    // the MAX of the two, not their product.
    let ns = glm_nsplit(512, c.heads);
    assert_eq!(
        bytes(&with_pf, "act.opart"),
        (c.heads * 512.max(ns) * c.kv_lora) as u64 * 4
    );
    // The MoE partials are [T*k, H] f32 — the grouped prefill FFN scatters into them.
    assert_eq!(
        bytes(&with_pf, "act.part"),
        512 * (c.top_k * c.hidden) as u64 * 4
    );
    assert_eq!(
        bytes(&dec_only, "act.part"),
        (c.top_k * c.hidden) as u64 * 4,
        "decode: one row"
    );
    // The DECODE per-slot gate/up buffer stays one row — the grouped path uses moe_fug instead.
    assert_eq!(
        bytes(&with_pf, "act.fu"),
        (c.top_k * c.moe_inter) as u64 * 2
    );
}

/// The descriptor reports the buckets it actually emitted.
#[test]
fn kimi_prefill_descriptor_lists_buckets() {
    let c = kimi_ref_cfg();
    let (_, d) = pf_block(&c, 512, &[128, 512]);
    assert_eq!(d.programs.prefill_buckets, vec![128, 512]);
    assert_eq!(d.programs.decode_t, 1);
}

/// The bucket ladder is capped at ctx — a rung above the compiled context can never be invoked.
#[test]
fn mla_prefill_bucket_ladder_is_ctx_capped() {
    assert_eq!(glm_prefill_buckets(512), vec![128, 512]);
    assert_eq!(glm_prefill_buckets(100), Vec::<u32>::new());
    assert_eq!(
        glm_prefill_buckets(1 << 20),
        vec![128, 512, 1024, 2048, 4096, 8192]
    );
}

/// A multi-layer extraction cannot carry prefill: the program ends at the post-attention norm,
/// so there is no residual stream for layer l+1 to read. It must refuse, not emit a broken chain.
#[test]
#[should_panic(expected = "single-layer")]
fn mla_prefill_refuses_a_multi_layer_block() {
    let c = kimi_ref_cfg();
    glm_build_block_pf(
        &c,
        512,
        256,
        0..2,
        true,
        "kimi-ref",
        MlaArch::Kimi,
        &[128],
        PrefillScope::Attn,
        MoeEnc::Fp8Blk,
    );
}

/// The attention-only scope emits the verified flash-prefill arm and stops there.
#[test]
fn mla_prefill_attn_scope_still_emits() {
    let (m, _) = pf_block(&kimi_ref_cfg(), 512, &[128]);
    assert!(m.progs[0]
        .insts
        .iter()
        .any(|d| d.op == DevOp::FlashMlaPrefill as u16));
}

// ===== WHOLE-LAYER prefill: MLA attention + token-sorted grouped MoE FFN (ops 83-87) ========

fn pf_full_enc(
    c: &GlmCfg,
    ctx: u32,
    pf: &[u32],
    block: std::ops::Range<usize>,
    enc: MoeEnc,
) -> Model {
    glm_build_block_pf(
        c,
        ctx,
        256,
        block,
        true,
        "kimi-ref",
        MlaArch::Kimi,
        pf,
        PrefillScope::Full,
        enc,
    )
    .0
}

fn pf_full(c: &GlmCfg, ctx: u32, pf: &[u32], block: std::ops::Range<usize>) -> Model {
    pf_full_enc(c, ctx, pf, block, MoeEnc::Fp8Blk)
}

/// The whole-layer prefill bucket, op for op. The FFN is the GROUPED path — not the decode
/// per-slot ops with a row loop — so no `MoeExpertGlu*`/`MoeCombine` may appear.
#[test]
fn mla_full_prefill_bucket_op_sequence() {
    use DevOp::*;
    let m = pf_full(&kimi_ref_cfg(), 512, &[128], 1..2);
    let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
    // 15 attention ops, then: router score GEMM, router top-k tail, align, shared GLU,
    // shared down, grouped glu, grouped down, combine = 23.
    assert_eq!(ops.len(), 23, "attention (15) + MoE FFN (8)");
    assert_eq!(
        [ops[16], ops[17], ops[18], ops[20], ops[21], ops[22]],
        [
            MoeRouterTopkPf as u16, // router top-k tail (15 = the [T,n_exp] score GEMM)
            MoeAlignPf as u16,      // token-sort / MPF_BM-padded prefix
            GemmGlu as u16,         // shared expert gate|up (19 = its down GEMM)
            MoeGroupGluPf as u16,   // grouped gate/up over the sorted rows
            MoeGroupDownPf as u16,
            MoeCombinePf as u16
        ]
    );
    assert!(
        ops.contains(&(FlashMlaPrefill as u16)),
        "attention half still present"
    );
    for bad in [
        MoeExpertGluFp8Blk,
        MoeExpertDownFp8Blk,
        MoeCombine,
        MoeRouterTopk,
        GemvGlu,
        Gemv,
    ] {
        assert!(
            !ops.contains(&(bad as u16)),
            "{bad:?} is a DECODE op; it must not appear"
        );
    }
}

/// The grouped-FFN operand fields, including the ones a wrong value would make silently wrong:
/// `n_exp` on both grouped GEMMs (the table indirection), `T` on the router tail and combine.
#[test]
fn mla_full_prefill_moe_operands() {
    let c = kimi_k27_code_cfg(4);
    let m = pf_full(&c, 1024, &[256], 1..2);
    let p = &m.progs[0];
    let rt = find_op(p, DevOp::MoeRouterTopkPf);
    assert_eq!((rt.i[1], rt.i[2], rt.i[4]), (384, 8, 256), "n_exp, k, T");
    assert_eq!(rt.i[3], GLM_ROUTER_FLAGS);
    let al = find_op(p, DevOp::MoeAlignPf);
    assert_eq!((al.i[0], al.i[1], al.i[2]), (256, 384, 8), "T, n_exp, k");
    assert_eq!(al.blocks, 1, "align is a single-workgroup global scan");
    let g = find_op(p, DevOp::MoeGroupGluPf);
    assert_eq!(
        (g.i[0], g.i[1], g.i[2], g.i[3]),
        (c.moe_inter, c.hidden, 384, 1),
        "I_moe (EP: whole), H, n_exp, fp8"
    );
    let dn = find_op(p, DevOp::MoeGroupDownPf);
    assert_eq!((dn.i[0], dn.i[1], dn.i[2]), (c.hidden, c.moe_inter, 384));
    let cb = find_op(p, DevOp::MoeCombinePf);
    assert_eq!((cb.i[0], cb.i[1], cb.i[2]), (c.hidden, 8, 256), "H, k, T");
}

/// The gathered arrays are sized on the MPF_BM-PADDED bound `T*k + n_exp*(MPF_BM-1)`, not `T*k`.
/// Sizing them from `T*k` is an out-of-bounds device write that hides at small expert counts and
/// is guaranteed at 384: the padding alone is 384*127 = 48768 rows.
#[test]
fn mla_full_prefill_pads_the_gathered_rows() {
    let c = kimi_k27_code_cfg(4);
    let m = pf_full(&c, 1024, &[256], 1..2);
    let bytes = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
    let pad = 256u64 * 8 + 384 * (MPF_BM as u64 - 1);
    assert_eq!(pad, 2048 + 48768);
    assert_eq!(bytes("act.moe_rowtok"), pad * 4);
    assert_eq!(bytes("act.moe_rowpart"), pad * 4);
    assert_eq!(bytes("act.moe_rowgate"), pad * 4);
    assert_eq!(
        bytes("act.moe_fug"),
        pad * c.moe_inter as u64 * 2,
        "EP: full moe_inter"
    );
    assert_eq!(bytes("act.moe_meta"), (3 * 384 + 1) as u64 * 4);
    // part is [T*k, H] f32 — no padding, the down op scatters by row_partidx.
    assert_eq!(bytes("act.part"), 256 * (8 * c.hidden) as u64 * 4);
}

/// Whole-layer prefill CHAINS across layers (unlike the attention-only scope), because the
/// combine produces a real residual stream.
#[test]
fn mla_full_prefill_chains_multiple_layers() {
    let mut c = kimi_ref_cfg();
    c.first_k_dense = 0; // every layer MoE, so a 2-layer block is all-prefillable
    let m = pf_full(&c, 512, &[128], 1..3);
    let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
    assert_eq!(ops.len(), 46, "two whole layers");
    assert_eq!(
        ops.iter()
            .filter(|&&o| o == DevOp::FlashMlaPrefill as u16)
            .count(),
        2
    );
    assert_eq!(
        ops.iter()
            .filter(|&&o| o == DevOp::MoeCombinePf as u16)
            .count(),
        2
    );
}

/// TP: the shared expert and the routed partials all-reduce through the TWO-SHOT collective at
/// `t*hidden`, and the combine writes into the peer slot rather than onto the residual (which
/// XReduce would otherwise sum tp times).
#[test]
fn mla_full_prefill_tp_combine_is_a_partial() {
    let c = kimi_k27_code_cfg(4);
    let m = pf_full(&c, 1024, &[256], 1..2);
    let p = &m.progs[0];
    let xrs: Vec<_> = p
        .insts
        .iter()
        .filter(|d| d.op == DevOp::XReduceTwoShot as u16)
        .collect();
    assert_eq!(xrs.len(), 2, "one for o_proj, one for the FFN combine");
    for x in &xrs {
        assert_eq!(x.i[0], 256 * c.hidden, "reduces t*hidden");
        assert_eq!(x.i[1], 4);
    }
    assert_ne!(
        xrs[0].i[2], xrs[1].i[2],
        "the two partials occupy DIFFERENT peer slots"
    );
    assert_eq!(
        xrs[1].i[2],
        256 * c.hidden * 2,
        "FFN partial sits past the o_proj partial"
    );
    // The combine's residual is ZERO, not xmid — xmid is added after the all-reduce. It is
    // spelled TENSOR_NONE rather than `act.zero_h`: the kernel's `residual ? ... : 0.0f`
    // makes a null pointer the zero residual, and naming a materialised [T,H] zero buffer
    // cost the op a whole extra input stream for no arithmetic. `act.zero_h` still exists —
    // the DECODE MoeCombine (op 82) names it — so this asserts the PREFILL op only.
    assert_eq!(find_op(p, DevOp::MoeCombinePf).t[1], TENSOR_NONE);
}

/// The manifest is what pairs a packet with an object, so a whole-layer bucket must declare the
/// MoE prefill axis — `interp_prefill_mla` alone would hit `default:` on ops 83-87 and write
/// nothing.
#[test]
fn mla_full_prefill_declares_the_moe_prefill_axis() {
    let m = pf_full(&kimi_k27_code_cfg(4), 1024, &[256], 1..2);
    let man = crate::manifest::build(
        &m,
        "gfx950",
        &crate::LeanReport::skipped("test: gate not run"),
    );
    assert_eq!(man["features"]["moe_prefill"], true);
    assert_eq!(man["features"]["prefill"], true);
    let ops: Vec<&str> = man["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    for o in [
        "MoeRouterTopkPf",
        "MoeAlignPf",
        "MoeGroupGluPf",
        "MoeGroupDownPf",
        "MoeCombinePf",
    ] {
        assert!(ops.contains(&o), "{o} missing from the manifest");
    }
}

// ===== GROUP-LIMITED ROUTING (DeepSeek noaux_tc) ===========================================

/// Both router tails must carry the group operands, and they must AGREE. The prefill tail is the
/// decode tail under a token loop, so an emitter that set them on one and not the other would
/// route the same token to different experts depending on which program ran it — the exact
/// class of prefill/decode divergence that reads as a model bug rather than a compiler one.
#[test]
fn mla_group_routing_reaches_both_router_tails() {
    let mut c = kimi_k27_code_cfg(4);
    c.n_group = 8;
    c.topk_group = 4;
    let m = pf_full(&c, 1024, &[256], 1..2);
    let pf = find_op(&m.progs[0], DevOp::MoeRouterTopkPf);
    assert_eq!(
        (pf.i[6], pf.i[7]),
        (8, 4),
        "prefill tail carries n_group/topk_group"
    );
    let dec = find_op(m.progs.last().unwrap(), DevOp::MoeRouterTopk);
    assert_eq!(
        (dec.i[6], dec.i[7]),
        (8, 4),
        "decode tail carries the SAME pair"
    );
    assert_eq!(
        (pf.i[1], pf.i[2]),
        (dec.i[1], dec.i[2]),
        "n_exp/k agree too"
    );
}

/// At `n_group <= 1` the rule is the identity, and the emitter must still say so explicitly —
/// the kernel treats 1 as inert, so every GLM / Qwen / Mixtral packet stays bit-identical.
#[test]
fn mla_group_routing_is_inert_for_ungrouped_models() {
    let c = kimi_ref_cfg(); // n_group = 1
    let (m, _) = pf_block(&c, 512, &[]);
    let dec = find_op(m.progs.last().unwrap(), DevOp::MoeRouterTopk);
    assert_eq!(
        (dec.i[6], dec.i[7]),
        (1, 1),
        "ungrouped => identity operands"
    );
}

// ===== A4W4 (MXFP4 on both operands) for the grouped expert path ===========================

/// `i[3]` selects the encoding, and A4W4 binds the two extra operands the fused bridge needs:
/// `t7` on GLU is the E8M0 rows it WRITES (the bridge is its epilogue), `t5` on DOWN is the same
/// rows READ back. Getting either wrong means the intermediate silently loses its scales.
#[test]
fn mla_a4w4_expert_path_binds_the_bridge_operands() {
    let c = kimi_k27_code_cfg(4);
    let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let p = &m.progs[0];
    let scale = m
        .tensors
        .iter()
        .position(|t| t.name == "act.moe_fuscale")
        .unwrap() as u32;
    let rowpart = m
        .tensors
        .iter()
        .position(|t| t.name == "act.moe_rowpart")
        .unwrap() as u32;
    let g = find_op(p, DevOp::MoeGroupGluPf);
    assert_eq!(g.i[3], 2, "i[3]=2 selects the MXFP4 body");
    assert_eq!(
        g.t[7], scale,
        "GLU writes the E8M0 rows (its epilogue IS the bridge)"
    );
    assert_eq!(
        g.t[6], rowpart,
        "GLU needs row_partidx so the bridge skips PAD rows"
    );
    let d = find_op(p, DevOp::MoeGroupDownPf);
    assert_eq!(d.i[3], 2);
    assert_eq!(d.t[5], scale, "DOWN reads the same E8M0 rows back");
    assert_eq!(
        d.t[1], g.t[0],
        "DOWN's A operand IS the bridge's fp4 output"
    );
}

/// THE ENCODING SLOT IS NOT THE SAME ON BOTH PHASES, and this test exists because getting it
/// wrong is silent.
///
/// Prefill ops 85/86 carry `n_exp` in `i[2]`, so the encoding took `i[3]`. Decode ops
/// 45/46/48/49 predate the field and already use `i[3]` for `n_exp`, so theirs is `i[6]`.
/// Writing the encoding into `i[3]` on a DECODE op sets `n_exp = 2`; every expert id >= 2 then
/// hits `if (eid >= n_exp) return;` and the op writes nothing at all. Combined with the AMD
/// dispatch default, which also writes nothing, the result is a layer that emits ZEROS with no
/// fault and no diagnostic — a dead MoE behind fluent-looking output.
///
/// So: pin the slot per op, and pin that `n_exp` still lands where the kernel reads it. Either
/// assertion alone would miss the failure; the pair is what makes it impossible.
#[test]
fn mla_encoding_slot_differs_between_decode_and_prefill() {
    let c = kimi_k27_code_cfg(4);
    let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let (pf, dec) = (&m.progs[0], m.progs.last().unwrap());

    // PREFILL: encoding in i[3], n_exp in i[2].
    for op in [DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf] {
        let x = find_op(pf, op);
        assert_eq!(
            x.i[MoeEnc::PREFILL_SLOT],
            2,
            "{op:?}: encoding is i[3] on prefill"
        );
        assert_eq!(x.i[2], c.n_exp, "{op:?}: n_exp stays in i[2]");
    }
    // DECODE: encoding in i[6], n_exp in i[3]. If these two ever swap, n_exp becomes 2.
    for op in [DevOp::MoeExpertGluFp8Blk, DevOp::MoeExpertDownFp8Blk] {
        let x = find_op(dec, op);
        assert_eq!(
            x.i[MoeEnc::DECODE_SLOT],
            2,
            "{op:?}: encoding is i[6] on decode"
        );
        assert_eq!(
            x.i[3], c.n_exp,
            "{op:?}: i[3] is n_exp — writing the encoding here kills it"
        );
        assert_ne!(x.i[3], 2, "n_exp=2 is the silent-zeros signature");
    }
    assert_ne!(
        MoeEnc::PREFILL_SLOT,
        MoeEnc::DECODE_SLOT,
        "the slots differ, deliberately"
    );
}

/// The grouped decode pair (48/49) takes the encoding in the same slot as the per-slot pair.
#[test]
fn mla_encoding_slot_on_the_grouped_decode_ops() {
    let mut c = kimi_k27_code_cfg(4);
    c.group = true; // collapses the 2*top_k per-slot packets into ops 48/49
    let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let dec = m.progs.last().unwrap();
    for op in [DevOp::MoeGroupGluFp8Blk, DevOp::MoeGroupDownFp8Blk] {
        let x = find_op(dec, op);
        assert_eq!(x.i[MoeEnc::DECODE_SLOT], 2, "{op:?}: encoding is i[6]");
        assert_eq!(x.i[3], c.n_exp, "{op:?}: i[3] is n_exp");
    }
}

/// On the EXPERT path a precision change is a field change: the same opcodes, one operand
/// different. That is the property the kernel side bought by making the encoding `i[3]`/`i[6]`
/// instead of new opcodes, and it is worth pinning because it is what makes an A/B across
/// encodings a controlled comparison rather than two different programs.
///
/// It does NOT extend to the PROJECTIONS, and the test says so rather than pretending: w4a16 is
/// a genuinely different kernel family (`GemmMxfp4` reuses the bf16 wide-K MFMA but fetches fp4
/// with the MX scale folded into the convert), so those change opcode. Precision is a field on
/// the experts and an opcode on the projections; asserting the stronger claim everywhere would
/// have been false.
#[test]
fn mla_encoding_is_a_field_on_the_expert_path() {
    let c = kimi_k27_code_cfg(4);
    let ops = |enc| -> Vec<u16> {
        pf_full_enc(&c, 1024, &[256], 1..2, enc).progs[0]
            .insts
            .iter()
            .map(|d| d.op)
            .collect()
    };
    assert_eq!(
        ops(MoeEnc::Bf16),
        ops(MoeEnc::Fp8Blk),
        "bf16 vs block-fp8: same stream"
    );
    // Same expert opcodes under all three; only i[3] moves.
    for (enc, code) in [(MoeEnc::Bf16, 0), (MoeEnc::Fp8Blk, 1), (MoeEnc::Mxfp4, 2)] {
        let m = pf_full_enc(&c, 1024, &[256], 1..2, enc);
        assert_eq!(find_op(&m.progs[0], DevOp::MoeGroupGluPf).i[3], code);
        assert_eq!(find_op(&m.progs[0], DevOp::MoeGroupDownPf).i[3], code);
        // ... and on decode, in the OTHER slot. bf16 rides its own opcodes (41/42) rather
        // than the scale-table-carrying pair, so look at whichever the encoding selects.
        let dop = if enc == MoeEnc::Bf16 {
            DevOp::MoeExpertGlu
        } else {
            DevOp::MoeExpertGluFp8Blk
        };
        assert_eq!(
            find_op(m.progs.last().unwrap(), dop).i[6],
            code,
            "decode slot i[6]"
        );
    }
}

/// An all-MXFP4 packet: EVERY matmul weight consumer, in BOTH programs, on an MXFP4 arm — and
/// nothing left on a block-fp8 one. This is the whole point of the encoding work, and the check
/// is by absence as much as by presence: a single surviving `GemvFp8Blk` or `Gemv` would be a
/// mixed run reported as an MXFP4 one.
#[test]
fn mla_all_mxfp4_packet_has_no_other_encoding() {
    use DevOp::*;
    let c = kimi_k27_code_cfg(4);
    let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let pf: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
    let dec: Vec<u16> = m.progs.last().unwrap().insts.iter().map(|d| d.op).collect();

    // PREFILL: projections are on SOME mxfp4 tile rung, the shared GLU unfused into two of
    // them plus a Glu. Which rung is a per-shape decision now — pinning `GemmMxfp4` here
    // would re-assert the very thing the T3 fix removed, namely that every fp4 prefill GEMM
    // takes the 256x256 tile whatever its shape. What must hold is that the ENCODING is
    // uniform, which is what the absence check below states.
    const MXFP4_TILES: [DevOp; 5] = [
        GemmMxfp4,
        GemmMedMxfp4,
        GemmSmallMxfp4,
        GemmWideMxfp4,
        GemmC5Mxfp4,
    ];
    assert!(
        MXFP4_TILES.iter().any(|t| pf.contains(&(*t as u16))),
        "no mxfp4 prefill GEMM at all; stream = {pf:?}"
    );
    assert!(
        pf.contains(&(Glu as u16)),
        "no GemmGluMxfp4 => explicit Glu"
    );
    // DECODE: projections are GemvMxfp4, the shared GLU IS fused (op 92 exists at decode).
    assert!(dec.contains(&(GemvMxfp4 as u16)));
    assert!(dec.contains(&(GemvGluMxfp4 as u16)));

    // Nothing may remain on a bf16 or block-fp8 matmul arm, in EITHER program.
    for (name, ops) in [("prefill", &pf), ("decode", &dec)] {
        for bad in [
            Gemv,
            GemvGlu,
            GemvQkv,
            GemvFp8Blk,
            DenseGluFp8Blk,
            Gemm,
            GemmMed,
            GemmSmall,
            GemmWide,
            GemmC5,
            GemmGlu,
            GemmFp8,
            GemmMedFp8,
            GemmSmallFp8,
            GemmWideFp8,
            GemmC5Fp8,
            GemmGluFp8,
        ] {
            assert!(
                !ops.contains(&(bad as u16)),
                "{name}: {bad:?} survived into an all-MXFP4 packet"
            );
        }
    }
    // The expert path carries the encoding in its two phase-dependent slots.
    assert_eq!(
        find_op(&m.progs[0], MoeGroupGluPf).i[MoeEnc::PREFILL_SLOT],
        2
    );
    assert_eq!(
        find_op(m.progs.last().unwrap(), MoeExpertGluFp8Blk).i[MoeEnc::DECODE_SLOT],
        2
    );
}

/// MXFP4 weights are packed at half a byte with one E8M0 byte per 32 — and the scale handle
/// must be BOUND, not merely declared. A packed weight whose scale operand is TENSOR_NONE is a
/// null pointer in the kernel.
#[test]
fn mla_mxfp4_weights_are_packed_and_their_scales_bound() {
    let c = kimi_k27_code_cfg(4);
    let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let bf = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk);
    let bytes = |m: &Model, n: &str| m.tensors.iter().find(|t| t.name == n).map(|t| t.bytes);
    let nm = "model.layers.1.self_attn.o_proj.weight";
    let (n, k) = (c.hidden as u64, (c.heads / 4 * c.v_head) as u64);
    assert_eq!(bytes(&bf, nm), Some(n * k * 2), "bf16: 2 B/elt");
    assert_eq!(bytes(&mx, nm), Some(n * k / 2), "packed fp4: half a byte");
    assert_eq!(
        bytes(&mx, &format!("{nm}_scale")),
        Some(n * k / MX_BLOCK as u64)
    );
    assert_eq!(
        bytes(&bf, &format!("{nm}_scale")),
        None,
        "no E8M0 rows off the MXFP4 arm"
    );
    // Every MXFP4 projection op must carry a real scale handle in t3.
    for p in &mx.progs {
        for i in p
            .insts
            .iter()
            .filter(|d| d.op == DevOp::GemvMxfp4 as u16 || d.op == DevOp::GemmMxfp4 as u16)
        {
            assert_ne!(
                i.t[3], TENSOR_NONE,
                "MXFP4 projection with an unbound E8M0 scale"
            );
        }
    }
}

/// The ONE bf16 tensor left under MXFP4, declared as an exception rather than hidden. It is
/// safe precisely because it is DERIVED (weight-prep folds kv_b_proj, so a bf16 copy exists
/// whatever the checkpoint stores) — unlike the expert weights, where fp4 bytes read as bf16
/// would be noise.
#[test]
fn mla_mxfp4_wuv_is_the_declared_exception() {
    let c = kimi_k27_code_cfg(4);
    let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    let nm = "model.layers.1.self_attn.derived.v_absorb.weight";
    let w = mx.tensors.iter().find(|t| t.name == nm).unwrap();
    let (nh_l, dk, vd) = (c.heads / 4, c.kv_lora, c.v_head);
    assert_eq!(
        w.bytes,
        (nh_l * dk * vd) as u64 * 2,
        "W_uv stays bf16 under MXFP4"
    );
    assert!(mx.tensors.iter().all(|t| t.name != format!("{nm}_scale")));
    assert!(mxfp4_bf16_exceptions().contains(&"MlaMergeFold/Wuv"));
    // How much of the model this is, at the REAL Kimi K2.7 geometry rather than the scaled
    // fixture — the number a dtype comparison has to be able to quote instead of guess.
    // W_uv = n_head*kv_lora*v_head; experts = n_exp*3*moe_inter*hidden.
    let wuv_real: u64 = 64 * 512 * 128;
    let experts_real: u64 = 384 * 3 * 2048 * 7168;
    assert_eq!(wuv_real, 4_194_304);
    assert!(
        wuv_real * 4000 < experts_real,
        "W_uv is {:.4}% of one layer's expert weights",
        wuv_real as f64 * 100.0 / experts_real as f64
    );
}

/// The manifest must not call an MXFP4 packet fp8 just because the expert opcodes still carry
/// fp8-era NAMES. Once the encoding became a runtime field, `MoeExpertGluFp8Blk` with `i[6]=2`
/// is an MXFP4 instruction — the name stopped being a fact and became a label.
#[test]
fn mla_manifest_does_not_call_an_mxfp4_packet_fp8() {
    let c = kimi_k27_code_cfg(4);
    let mx = crate::manifest::build(
        &pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4),
        "gfx950",
        &crate::LeanReport::skipped("test: gate not run"),
    );
    assert_eq!(
        mx["shapes"]["moe_enc"],
        serde_json::json!([2]),
        "ONE encoding"
    );
    assert_eq!(mx["features"]["moe_enc_mixed"], false);
    assert_eq!(mx["features"]["a4w4"], true);
    assert_eq!(mx["features"]["mxfp4_weights"], true);
    assert_eq!(
        mx["features"]["fp8_weights"], false,
        "no fp8 weight anywhere in this packet"
    );
    // The block-fp8 packet is still reported as fp8 — the correction must not overreach.
    let fp8 = crate::manifest::build(
        &pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk),
        "gfx950",
        &crate::LeanReport::skipped("test: gate not run"),
    );
    assert_eq!(fp8["shapes"]["moe_enc"], serde_json::json!([1]));
    assert_eq!(fp8["features"]["fp8_weights"], true);
    assert_eq!(fp8["features"]["a4w4"], false);
}

/// PLOW_MXFP4=1 and PLOW_FP8=1 together ask for two encodings in one packet.
#[test]
fn mla_two_encodings_at_once_is_refused() {
    assert_eq!(
        MoeEnc::from_flags(true, true),
        MoeEnc::Mxfp4,
        "mxfp4 wins the enum"
    );
    // The env-level guard is what actually refuses; exercised through the CLI.
}

/// A4W4 halves the gathered intermediate and adds one E8M0 byte per 32 values. The scale rows
/// have no bf16 counterpart, so they must be DECLARED or the bridge writes to a null handle.
#[test]
fn mla_a4w4_sizes_the_packed_intermediate() {
    let c = kimi_k27_code_cfg(4);
    let pad = 256u64 * 8 + 384 * (MPF_BM as u64 - 1);
    let bytes = |m: &Model, n: &str| m.tensors.iter().find(|t| t.name == n).map(|t| t.bytes);
    let fp8 = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk);
    let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
    assert_eq!(
        bytes(&fp8, "act.moe_fug"),
        Some(pad * c.moe_inter as u64 * 2)
    );
    assert_eq!(
        bytes(&mx, "act.moe_fug"),
        Some(pad * (c.moe_inter / 2) as u64),
        "packed fp4"
    );
    assert_eq!(
        bytes(&mx, "act.moe_fuscale"),
        Some(pad * (c.moe_inter / MX_BLOCK) as u64)
    );
    assert_eq!(
        bytes(&fp8, "act.moe_fuscale"),
        None,
        "no E8M0 rows on the block-fp8 arm"
    );
}

/// bf16 and block-fp8 emission must be bit-identical to before the encoding field existed.
#[test]
fn mla_default_encoding_is_block_fp8_and_unchanged() {
    let c = kimi_ref_cfg();
    let (m, _) = pf_block(&c, 512, &[128]);
    let dec = m.progs.last().unwrap();
    // The decode expert ops are untouched by the encoding work — they have no such field.
    assert!(dec
        .insts
        .iter()
        .any(|d| d.op == DevOp::MoeExpertGluFp8Blk as u16));
    assert_eq!(MoeEnc::from_flags(true, false), MoeEnc::Fp8Blk);
    assert_eq!(MoeEnc::from_flags(false, false), MoeEnc::Bf16);
    assert_eq!(MoeEnc::from_flags(false, true), MoeEnc::Mxfp4);
    assert_eq!(
        (
            MoeEnc::Bf16.code(),
            MoeEnc::Fp8Blk.code(),
            MoeEnc::Mxfp4.code()
        ),
        (0, 1, 2)
    );
}

/// 384 experts is inside the grouped prefill's LDS bound; past it the align histogram and the
/// router key array would overrun the shared arena with nothing on device to notice.
#[test]
#[should_panic(expected = "exceeds the grouped MoE prefill LDS bound")]
fn mla_full_prefill_bounds_the_expert_count() {
    let mut c = kimi_k27_code_cfg(4);
    assert_eq!(c.n_exp, 384, "Kimi K2.7 routes 384 — inside the bound");
    c.n_exp = 1024;
    pf_full(&c, 512, &[128], 1..2);
}

/// A DENSE layer prefills on the GROUPED EXPERT ARMS with degenerate 1-expert routing, because
/// there is no block-fp8 T-row GEMM opcode and ops 85/86 already are one. This pins the whole
/// construction: the align op gets NO routing table (that is what makes it synthesise
/// "every token -> expert 0, gate 1"), the two grouped GEMMs carry `n_exp = 1` and the dense
/// weight-pointer tables rather than the expert ones, and there is no router and no shared
/// expert. Getting any of these wrong produces a packet that RUNS and is wrong — the AMD
/// dispatch `default:` leaves outputs untouched rather than trapping.
#[test]
fn mla_full_prefill_dense_layer_uses_synthetic_single_expert_routing() {
    let c = kimi_ref_cfg(); // first_k_dense = 1, so layer 0 is dense
    let m = pf_full(&c, 512, &[128], 0..1);
    let pf = &m.progs[0];
    let ops: Vec<u16> = pf.insts.iter().map(|d| d.op).collect();

    // No router on a dense layer: nothing to score, and `mlp.gate.weight` does not exist.
    assert!(
        !ops.contains(&(DevOp::MoeRouterTopkPf as u16)),
        "a dense layer has no router — its routing is synthesised by the align op"
    );

    let align = pf
        .insts
        .iter()
        .find(|d| d.op == DevOp::MoeAlignPf as u16)
        .expect("dense prefill still aligns: the grouped GEMMs read its meta/row maps");
    assert_eq!(
        align.t[1], TENSOR_NONE,
        "the routing table operand MUST be TENSOR_NONE — that is the signal d_moe_align_pf \
             reads to synthesise single-expert routing. Binding a real table here would route the \
             dense FFN through whatever the previous MoE layer left in `tab`."
    );
    assert_eq!((align.i[1], align.i[2]), (1, 1), "n_exp = 1, top_k = 1");

    for (op, name) in [
        (DevOp::MoeGroupGluPf, "gate/up"),
        (DevOp::MoeGroupDownPf, "down"),
    ] {
        let d = pf
            .insts
            .iter()
            .find(|d| d.op == op as u16)
            .unwrap_or_else(|| panic!("dense prefill must emit the grouped {name} arm"));
        assert_eq!(d.i[2], 1, "{name}: exactly one 'expert'");
        assert_eq!(
            d.i[3],
            MoeEnc::Fp8Blk.code(),
            "{name}: block-fp8 goes in the PREFILL encoding slot i[3], not decode's i[6]"
        );
        assert_ne!(
            d.t[2], TENSOR_NONE,
            "{name}: dense weight-pointer table must be bound"
        );
        assert_ne!(
            d.t[3], TENSOR_NONE,
            "{name}: dense scale-pointer table must be bound"
        );
    }

    // The combine takes no shared expert — a dense layer has none, and d_moe_combine_pf
    // already honours a null `shared`.
    let cmb = pf
        .insts
        .iter()
        .find(|d| d.op == DevOp::MoeCombinePf as u16)
        .expect("dense prefill combines part into the residual");
    assert_eq!(cmb.t[2], TENSOR_NONE, "no shared expert on a dense layer");
    assert_eq!(cmb.i[1], 1, "k = 1: one part slot per token");
}

/// The dense prefill borrows the MoE arms, so it must NOT borrow the MoE weight tables. Binding
/// `ewt`/`est` there would read 256 routed experts that a dense layer does not have.
#[test]
fn mla_full_prefill_dense_binds_dense_tables_not_expert_tables() {
    let c = kimi_ref_cfg();
    let m = pf_full(&c, 512, &[128], 0..1);
    let names = m
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>();
    assert!(
        names.iter().any(|n| n.contains("mlp.dense_weight_table")),
        "dense prefill declares its own [3] u64 pointer table; got {names:?}"
    );
    let glu = m.progs[0]
        .insts
        .iter()
        .find(|d| d.op == DevOp::MoeGroupGluPf as u16)
        .unwrap();
    let bound = m.tensors[glu.t[2] as usize].name.as_str();
    assert!(
        bound.contains("dense_weight_table"),
        "the grouped GLU on a DENSE layer must read the dense table, got `{bound}`"
    );
}

/// EVERY program in one blob must name the SAME peer partial-slot-B offset.
///
/// The host binds `act.dg_tp` ONCE, at `scratch_base + DevBlob::tp.slot_bytes`, and recovers
/// `slot_bytes` as `max(i[2])` over every `XReduce`/`XReduceTwoShot` in the blob
/// (`plowrt/src/asset/devblob.rs`). So a blob that bakes a different `i[2]` per program has, by
/// construction, at most ONE program whose FFN all-reduce reads the buffer its own combine
/// wrote — the others reduce untouched peer memory and their FFN contribution silently
/// disappears. That is exactly what a bucket ladder + decode bundle did: `t*h*2` per bucket and
/// `h*2` for decode, four values, and the decode program (healthy on its own, where its value
/// IS the max) started emitting a constant token the moment prefill buckets joined it.
#[test]
fn every_program_shares_one_peer_partial_slot_offset() {
    let mut c = kimi_ref_cfg();
    c.tp = 2; // TP is what puts XReduce in the program at all
    let m = glm_build_block_pf(
        &c,
        2048,
        256,
        0..2,
        true,
        "kimi-ref",
        MlaArch::Kimi,
        &[128, 512],
        PrefillScope::Full,
        MoeEnc::Fp8Blk,
    )
    .0;
    assert!(
        m.progs.len() >= 3,
        "two buckets + decode; got {}",
        m.progs.len()
    );
    let want = 512 * c.hidden * 2; // rows_max * hidden * 2, the widest bucket
    let mut slots: Vec<u32> = m
        .progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .filter(|d| d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceTwoShot as u16)
        .map(|d| d.i[2])
        .collect();
    assert!(!slots.is_empty(), "tp=2 must emit collectives");
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(
        slots,
        vec![0, want],
        "slot A is 0 and slot B is rows_max*hidden*2 in EVERY program; a per-bucket offset \
             here means the host binds act.dg_tp where most programs do not look"
    );
    // ...and the declared buffer must actually be that wide, since the prefill combine writes
    // [T, hidden] into it.
    let dg = m
        .tensors
        .iter()
        .find(|t| t.name == "act.dg_tp")
        .expect("tp>1 declares act.dg_tp");
    assert_eq!(
        dg.bytes, want as u64,
        "dg_tp is row-dimensioned, like og_tp"
    );
}

/// MXFP4 is the one encoding with no dense prefill arm: its grouped path is the A4W4
/// fused-bridge, whose scale rows the dense emit does not declare. Refuse rather than emit an
/// encoding field pointing at an arm nothing bound operands for.
#[test]
#[should_panic(expected = "MXFP4 prefill is not implemented for DENSE layer")]
fn mla_full_prefill_refuses_a_dense_mxfp4_layer() {
    let c = kimi_ref_cfg();
    glm_build_block_pf(
        &c,
        512,
        256,
        0..1,
        false,
        "glm-ref",
        MlaArch::Glm,
        &[128],
        PrefillScope::Full,
        MoeEnc::Mxfp4,
    );
}

/// The DSA gate does NOT reach for `FlashGatherPrefill`, even armed. A gathered prefill wants
/// one top_k row PER QUERY (`idx[b][t][top_k]`) and the selector produces a single row, so
/// emitting the gather would give every query token the last token's selection.
#[test]
fn mla_prefill_stays_dense_under_an_armed_dsa_gate() {
    let mut c = kimi_ref_cfg();
    c.has_dsa = true;
    // `kimi_ref_cfg` carries a SYNTHETIC indexer geometry (index_heads 8, index_topk 64) that
    // no AMD kernel can execute — `d_index_score_mfma` static_asserts HIc==32 and interp.hip
    // hardcodes DI_=128 — and this test arms DSA only to check that PREFILL stays dense while
    // DECODE gathers. Give it the real GLM geometry so the fixture describes an emittable blob;
    // the routing assertions below are unchanged and are what the test is actually about.
    c.index_heads = 32;
    c.index_dim = 128;
    c.indexer_full = vec![false, true, false, false];
    let (m, _) = glm_build_block_pf(
        &c,
        131072,
        256,
        1..2,
        true,
        "glm-ref",
        MlaArch::Glm,
        &[128],
        PrefillScope::Attn,
        MoeEnc::Fp8Blk,
    );
    let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
    assert!(
        ops.contains(&(DevOp::FlashMlaPrefill as u16)),
        "dense MLA prefill"
    );
    assert!(
        !ops.contains(&(DevOp::FlashGatherPrefill as u16)),
        "no per-query selector exists"
    );
    assert!(
        !ops.contains(&(DevOp::IndexScore as u16)),
        "the indexer is decode-shaped"
    );
    // The DECODE program of the same asset still gathers — the gate is armed, only prefill opts out.
    let dec: Vec<u16> = m.progs.last().unwrap().insts.iter().map(|d| d.op).collect();
    assert!(
        dec.contains(&(DevOp::FlashGatherDecode as u16)),
        "decode still gathers at 128k"
    );
}

/// The manifest must SEE the prefill buckets, since that is what tells an object builder it
/// needs the `PLOW_MLA_PREFILL=1` arms. Derived from the instruction stream, not from intent.
#[test]
fn mla_prefill_shows_up_in_the_build_manifest() {
    let c = kimi_ref_cfg();
    let (m, _) = pf_block(&c, 512, &[128, 512]);
    let man = crate::manifest::build(
        &m,
        "gfx950",
        &crate::LeanReport::skipped("test: gate not run"),
    );
    let ops: Vec<&str> = man["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(ops.contains(&"FlashMlaPrefill"));
    assert!(ops.contains(&"MlaMergeFold"));
    assert_eq!(man["features"]["mla"], true);
    // `prefill` must be true for an MLA packet: its prefill flash is the LATENT one, and this
    // flag is what tells the object builder it needs PLOW_MLA_PREFILL=1.
    assert_eq!(man["features"]["prefill"], true);
    assert_eq!(
        man["features"]["moe_prefill"], false,
        "attention-only scope"
    );
    // block-fp8 experts ARE fp8 weights, [128,128] scale grid or not.
    assert_eq!(man["features"]["fp8_weights"], true);
    assert_eq!(
        man["shapes"]["prefill_buckets"],
        serde_json::json!([128, 512])
    );
    let progs = man["programs"].as_array().unwrap();
    assert_eq!(progs[0]["kind"], "prefill");
    assert_eq!(progs[0]["bucket"], 128);
    assert_eq!(progs.last().unwrap()["kind"], "decode");
}

// ===== Kimi K2.7-Code at the ATOM/AITER comparison point: 384 experts, top-8, TP=4 ==========

/// Kimi K2.7-Code routing geometry. The GLM-5.2 numbers this emitter was first written against
/// are 256 experts / top-8; K2.7-Code is 384 / top-8, so anything sized for 256 is a Kimi bug.
fn kimi_k27_code_cfg(tp: u32) -> GlmCfg {
    let mut c = kimi_ref_cfg();
    c.heads = 64; // real K2.7 head count — the shape that has to divide by tp=4
    c.n_exp = 384;
    c.top_k = 8;
    c.tp = tp;
    c.ep = true; // 384/4 = 96 WHOLE experts per rank
    c
}

/// Nothing in the emit is sized for 256 experts: every expert-dimensioned field follows
/// `c.n_exp`, and the co-resident CU partition follows `top_k`, not the expert count.
#[test]
fn kimi_k27_code_384_experts_top8_tp4() {
    let c = kimi_k27_code_cfg(4);
    let (m, d) = pf_block(&c, 1024, &[128]);
    let dec = m.progs.last().unwrap();
    // Router: the score GEMV's N is the expert count, and the top-k tail carries it too.
    let topk = find_op(dec, DevOp::MoeRouterTopk);
    assert_eq!(topk.i[1], 384, "router top-k must see all 384 experts");
    assert_eq!(topk.i[2], 8, "top_k = 8");
    // One (glu, down) pair per top_k slot — 8, not 2 and not 256.
    let n_glu = dec
        .insts
        .iter()
        .filter(|x| x.op == DevOp::MoeExpertGluFp8Blk as u16)
        .count();
    assert_eq!(n_glu, 8, "one expert packet per top_k slot");
    for g in dec
        .insts
        .iter()
        .filter(|x| x.op == DevOp::MoeExpertGluFp8Blk as u16)
    {
        assert_eq!(g.i[3], 384, "expert op carries n_exp=384 (table bound)");
        // EP at tp=4: 384/4 = 96 WHOLE experts per rank, so each keeps the FULL moe_inter.
        assert_eq!(
            g.i[1], c.moe_inter,
            "EP: whole expert, full moe_inter — not the TP slice"
        );
    }
    assert_eq!(d.dims.n_exp, Some(384));
    assert_eq!(d.dims.top_k, Some(8));
    // 384 divides by every TP degree we care about: 4 -> 96 whole experts per rank.
    assert_eq!(384 % 4, 0);
    // And the attention half of the SAME asset is per-rank sharded at 64/4 = 16 heads.
    assert_eq!(find_op(&m.progs[0], DevOp::FlashMlaPrefill).i[1], 16);
    assert_eq!(find_op(&m.progs[0], DevOp::MlaMergeFold).i[1], 16);
}

/// TP=4 is the primary serving degree for this comparison, so pin the whole prefill bucket at
/// it: real head count, real expert count, both programs present, no decode op in the bucket.
#[test]
fn kimi_k27_code_tp4_prefill_bucket_is_complete_attention() {
    let c = kimi_k27_code_cfg(4);
    let (m, _) = pf_block(&c, 8192, &[128, 1024, 8192]);
    assert_eq!(m.prog_t, vec![128, 1024, 8192, 1], "3 buckets + decode");
    for (i, &t) in [128u32, 1024, 8192].iter().enumerate() {
        let p = &m.progs[i];
        assert_eq!(find_op(p, DevOp::FlashMlaPrefill).i[4], t, "n_tok = bucket");
        assert_eq!(find_op(p, DevOp::FlashMlaPrefill).i[1], 16, "nh_l = 64/4");
        assert_eq!(
            find_op(p, DevOp::MlaMergeFold).i[0],
            t,
            "fold n_batch = tokens"
        );
        assert_eq!(find_op(p, DevOp::MlaMergeFold).i[4], 1, "nsplit = 1");
        assert_eq!(find_op(p, DevOp::XReduceTwoShot).i[1], 4, "tp=4 all-reduce");
    }
}

/// The DeepSeek flavor differs only in the descriptor arch tag; the emit + kind + no-DSA are
/// identical to Kimi.
#[test]
fn deepseek_arch_tag() {
    let c = kimi_ref_cfg();
    let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "deepseek-v3", MlaArch::DeepSeek);
    assert_eq!(d.arch, "deepseek_mla_moe");
    assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
    assert_eq!(d.dsa_role, None);
}
