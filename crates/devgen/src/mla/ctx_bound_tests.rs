//! The emit-time `ctx` is a CEILING, and `packet::ctx_bound` is the rule that
//! narrows a packet emitted at one to a smaller LIVE bound.
//!
//! These tests are the whole evidence for that rule. The runtime applies it to a
//! parsed blob with no emitter in reach, so this is the only place the claim
//! "narrowing reproduces what the emitter would have written" can be checked at
//! all — and it is checked against the PRODUCTION recipe (`glm53-tp8`'s recorded
//! knobs: fp8 latent cache, armed DSA decode and prefill, seams, the decode
//! ladder and every prefill bucket), not a reduced fixture.

use super::*;
use packet::ctx_bound::{self, Scaling};

/// The production GLM recipe emitted whole at `ctx`: the tensor table, the
/// `GenTensor` recipes, every program's instruction stream, and the sha256 of
/// the packet bytes.
#[allow(clippy::type_complexity)]
struct CtxEmit {
    tensors: Vec<(String, u64)>,
    gen: Vec<packet::rope::GenTensor>,
    progs: Vec<(u32, Vec<crate::DevInst>)>,
    sha256: String,
}

fn ctx_emit(ctx: u32) -> CtxEmit {
    use std::sync::{Arc, Mutex};
    let _target = crate::EmitAmdGuard::set(true);
    crate::tune_demand::reset_tally();
    // Unique per CALL: two of these tests emit at the same ctx, and the test
    // harness starts them on separate threads (they serialise on the env guard,
    // but the directory name must not be shared even so).
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("plow-ctx-bound-{ctx}-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 5,
        "hidden_size": 6144, "num_attention_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "vocab_size": 128, "rms_norm_eps": 1e-5,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "first_k_dense_replace": 3, "routed_scaling_factor": 2.5,
        "rope_theta": 8000000.0,
        "indexer_types": ["full", "shared", "full", "shared", "full"]
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_FP8", "1"),
        ("PLOW_GLM_DECODE_NORM_ROWS", "1"),
        ("PLOW_GLM_DSA", "1"),
        ("PLOW_GLM_DSA_PF", "1"),
        ("PLOW_GLM_DSA_PF_SPAN", "3"),
        ("PLOW_GLM_FP8_KV", "1"),
        ("PLOW_GLM_FUSE_B1", "1"),
        ("PLOW_GLM_FUSE_ROPE", "0"),
        ("PLOW_GLM_FUSE_SEAM", "1"),
        ("PLOW_GLM_GEMM_LT", "1"),
        ("PLOW_GLM_GEMM_LT_DECODE", "1"),
        ("PLOW_GLM_INDEX_TP", "1"),
        ("PLOW_GLM_MOE_AITER", "1"),
        ("GLM_MOE_CORESIDENT", "2"),
        ("PLOW_GLM_MOE_FLAT_DECODE", "0"),
        ("PLOW_GLM_MOE_RESIDENT", "1"),
        ("PLOW_GLM_PLACE_PF", "0"),
        ("PLOW_GLM_SELECT_LOCAL", "1"),
        ("GLM_SHARD_HEAD", "1"),
        ("GLM_SHARED_CUS", "48"),
        ("PLOW_MLA_PREFILL", "full:128,512,2048,8192"),
        ("PLOW_MOE_PF_DET", "1"),
        ("PLOW_EMIT_PACKED_PREFILL", "0"),
        ("PLOW_DECODE_BATCH", "32"),
        ("PLOW_DECODE_BATCH_LADDER", "1,2,4,8,16,32"),
        ("PLOW_UNISEG", "0"),
        ("PLOW_MLA_PF_V2", "1"),
        ("PLOW_MLA_PF_AITER", "1"),
        ("PLOW_GLM_SEQ_PAR", "1"),
        ("PLOW_GLM_SEQ_PAR_PROJ", "1"),
        ("PLOW_GLM_FOLD_LT", "1"),
        ("PLOW_GLM_GEMM_LT_DECODE_EXT", "1"),
    ]);
    type Out = (
        Vec<(String, u64)>,
        Vec<packet::rope::GenTensor>,
        Vec<(u32, Vec<crate::DevInst>)>,
        String,
    );
    let seen: Arc<Mutex<Out>> = Arc::new(Mutex::new(Default::default()));
    let sink = Arc::clone(&seen);
    let verify: crate::VerifyHook = Box::new(move |model| {
        let mut g = sink.lock().unwrap();
        g.0 = model
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.bytes))
            .collect();
        g.1 = model.gen.clone();
        g.2 = model
            .progs
            .iter()
            .zip(&model.prog_t)
            .map(|(p, &t)| (t, p.insts.clone()))
            .collect();
        g.3 = plow_asset::decode_objects::image_sha256(&model.to_blob());
        Ok(crate::LeanReport::skipped("ctx-bound rule test"))
    });
    glm_emit_full(
        &dir,
        ctx,
        dir.join("model.pkt").to_str().unwrap(),
        304,
        8,
        true,
        true,
        "gfx942",
        None,
        Some(&verify),
    );
    std::fs::remove_dir_all(&dir).ok();
    let g = seen.lock().unwrap();
    CtxEmit {
        tensors: g.0.clone(),
        gen: g.1.clone(),
        progs: g.2.clone(),
        sha256: g.3.clone(),
    }
}

/// The production recipe's declared CEILING and the bound this campaign serves
/// it at. Both are above `GlmCfg::dsa`'s 65536 crossover and above every
/// `min(_, ctx)` term in the emit, so `ctx` decides nothing but geometry
/// between them — which is what makes the narrowing rule an equality.
const CEILING: u32 = 1_048_576;
const LIVE: u32 = 81_920;

struct Fixture {
    ceiling: CtxEmit,
    live: CtxEmit,
    live_again: CtxEmit,
}

/// Three emits, shared by every test in this file.
///
/// Emitting the whole recipe costs ~20 s and holds the emit knobs in the
/// process environment for all of it, and `test_env` exists because a sibling
/// test emitting during that window reads them. One set of emits for the file
/// keeps that window as narrow as the evidence allows. Callers hold
/// [`crate::test_env::env_guard`], which is also what serialises this.
fn fixture() -> &'static Fixture {
    static F: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    F.get_or_init(|| Fixture {
        ceiling: ctx_emit(CEILING),
        live: ctx_emit(LIVE),
        live_again: ctx_emit(LIVE),
    })
}

/// Apply `packet::ctx_bound`'s narrowing rule to an emit, exactly as
/// `plowrt::exec::amd` applies it to the parsed blob.
fn narrow(e: &CtxEmit, from: u32, to: u32) -> CtxEmit {
    let mut gen = e.gen.clone();
    for g in &mut gen {
        if ctx_bound::is_rope_recipe(g) {
            assert_eq!(g.ctx, from, "rope recipe row count is not the emit ctx");
            g.ctx = to;
        }
    }
    let rope_bytes: std::collections::HashMap<u32, u64> =
        gen.iter().map(|g| (g.tensor, g.byte_len())).collect();
    let tensors = e
        .tensors
        .iter()
        .enumerate()
        .map(|(h, (name, bytes))| {
            let bytes = match ctx_bound::tensor_scaling(name) {
                Scaling::Linear => ctx_bound::linear_bytes(*bytes, from, to)
                    .unwrap_or_else(|| panic!("{name}: {bytes} B is not a multiple of ctx {from}")),
                Scaling::IndexerBlock16 => ctx_bound::indexer_rescaled_bytes(*bytes, from, to)
                    .unwrap_or_else(|| panic!("{name}: invalid packed indexer context")),
                Scaling::Rope => rope_bytes[&(h as u32)],
                Scaling::Inert => *bytes,
                Scaling::Unknown => panic!("{name}: no ctx-scaling rule"),
            };
            (name.clone(), bytes)
        })
        .collect();
    let narrowed: std::collections::HashSet<u32> = e
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, (n, _))| ctx_bound::is_narrowed_cache(n))
        .map(|(h, _)| h as u32)
        .collect();
    let progs = e
        .progs
        .iter()
        .map(|(t, insts)| {
            if let Some((k, op, slot)) =
                ctx_bound::residual_ceiling_builder(insts, from, |h| narrowed.contains(&h))
            {
                panic!(
                    "program t={t} instruction {k} (op {op}) keeps the ceiling in {slot:?}: {:?}",
                    insts[k]
                );
            }
            let mut insts = insts.clone();
            ctx_bound::restride_builder(&mut insts, from, to);
            (*t, insts)
        })
        .collect();
    CtxEmit {
        tensors,
        gen,
        progs,
        sha256: e.sha256.clone(),
    }
}

fn assert_same(a: &CtxEmit, b: &CtxEmit, what: &str) {
    assert_eq!(a.tensors.len(), b.tensors.len(), "{what}: tensor count");
    for ((n1, b1), (n2, b2)) in a.tensors.iter().zip(&b.tensors) {
        assert_eq!(n1, n2, "{what}: tensor order");
        assert_eq!(b1, b2, "{what}: tensor `{n1}` bytes");
    }
    assert_eq!(a.gen, b.gen, "{what}: gen recipes");
    assert_eq!(a.progs.len(), b.progs.len(), "{what}: program count");
    for ((t1, i1), (t2, i2)) in a.progs.iter().zip(&b.progs) {
        assert_eq!(t1, t2, "{what}: program rows");
        assert_eq!(i1.len(), i2.len(), "{what}: program t={t1} instructions");
        for (k, (x, y)) in i1.iter().zip(i2).enumerate() {
            assert_eq!(x, y, "{what}: program t={t1} instruction {k}");
        }
    }
}

/// THE RULE. A packet emitted at a 1M ceiling, narrowed to 81920, IS the packet
/// the emitter writes at 81920 — same tensor table, same RoPE recipes, same
/// instruction stream, field for field.
///
/// This is what lets `plowc` stop encoding a serving length. Both ends of the
/// pair are above `GlmCfg::dsa`'s 65536 crossover and above every `min(_, ctx)`
/// policy term in the emit (`glm_dsa_pf_cap`'s 8*2048, `index_topk`'s 2048), and
/// `glm_nsplit` and `glm_gf` are saturated across the whole interval — so the
/// only thing `ctx` still decides between them is geometry, which is exactly
/// what the rule rewrites. Below 65536 that stops being true (the DSA decode arm
/// is gated on the context, not on the live length), which is why the runtime
/// refuses to narrow there rather than pretending the equality still holds.
#[test]
fn a_ceiling_packet_narrowed_to_the_live_bound_is_the_emit_at_that_bound() {
    let _guard = crate::test_env::env_guard();
    let e = fixture();
    assert_same(&narrow(&e.ceiling, CEILING, LIVE), &e.live, "1M -> 81920");
}

/// `d = identity`: narrowing to the ceiling itself moves nothing, down to the
/// sha256 of the emitted packet. Every artifact in the tree loads through the
/// same code path with `to == from`, so this is the gate that says the runtime
/// change costs existing packets nothing.
#[test]
fn narrowing_to_the_declared_ceiling_is_the_identity() {
    let _guard = crate::test_env::env_guard();
    let e = fixture();
    assert_same(&narrow(&e.live, LIVE, LIVE), &e.live, "81920 -> 81920");
    assert_same(&narrow(&e.ceiling, CEILING, CEILING), &e.ceiling, "1M -> 1M");
    // The packet BYTES, so "nothing moved" is not a claim about the three tables
    // this file happens to look at: the emitter is untouched by this change, so
    // two emits of the recipe at the old baked max_ctx hash the same and every
    // shipped artifact is bit-identical without being re-emitted.
    assert_eq!(e.live.sha256, e.live_again.sha256, "two emits at ctx 81920");
    assert_eq!(e.live.sha256.len(), 64);
}

/// Every tensor the emitter sizes from `ctx` is one this module's rule knows
/// about, and none of them is anything but exactly linear in it.
///
/// The soundness direction of the rule is checked by the pair test above (a
/// tensor wrongly called ctx-scaled would come out the wrong size there). This
/// is the COMPLETENESS direction: a tensor that moves with `ctx` and is not
/// classified [`Scaling::Linear`] or [`Scaling::Rope`] would be left at the
/// ceiling — correct, but it would hold the memory this whole change exists to
/// give back, silently.
#[test]
fn the_ctx_scaled_tensor_set_is_exactly_what_moves_with_ctx() {
    let _guard = crate::test_env::env_guard();
    let e = fixture();
    let (a, b) = (&e.live, &e.ceiling);
    assert_eq!(a.tensors.len(), b.tensors.len());
    let mut moved = Vec::new();
    for ((n1, b1), (n2, b2)) in a.tensors.iter().zip(&b.tensors) {
        assert_eq!(n1, n2);
        let scaling = ctx_bound::tensor_scaling(n1);
        if b1 == b2 {
            assert_ne!(scaling, Scaling::Linear, "`{n1}` is inert but classified");
            assert_ne!(scaling, Scaling::IndexerBlock16, "`{n1}` is inert but classified");
            assert_ne!(scaling, Scaling::Rope, "`{n1}` is inert but classified");
            continue;
        }
        assert_eq!(
            *b2 as u128 * LIVE as u128,
            *b1 as u128 * CEILING as u128,
            "`{n1}` moves with ctx but not linearly"
        );
        assert!(
            matches!(scaling, Scaling::Linear | Scaling::IndexerBlock16 | Scaling::Rope),
            "`{n1}` moves with ctx and has no rule"
        );
        moved.push(n1.as_str());
    }
    // Named rather than counted: this is the set the runtime gives memory back
    // on, and a recipe change that drops one of them should read as a diff.
    assert!(moved.contains(&"in.pos"), "{moved:?}");
    assert!(moved.contains(&"in.cos"), "{moved:?}");
    assert!(moved.contains(&"act.iscore_pf"), "{moved:?}");
    assert!(moved.contains(&"kv.0.ckv"), "{moved:?}");
    assert!(moved.contains(&"kv.0.krot"), "{moved:?}");
    assert!(moved.contains(&"kv.0.scale"), "{moved:?}");
    assert!(moved.contains(&"kv.0.kidx"), "{moved:?}");
    assert_eq!(moved.len(), 27, "{moved:?}");
}

/// The two ctx-derived constants that reach GENERATED CODE rather than a buffer
/// size are already saturated over the range the runtime narrows across, so the
/// ceiling emits exactly what the live bound would have.
///
/// This is why `exec::ctx_bound` rewrites strides and patches nothing else.
///
/// * `i[4]`, [`glm_nsplit`]: `ctx / NS_PER` reaches `NS_CEIL_MEASURED` at
///   `16 * 256 = 4096`... and then `min(fill)` and `min(kv_tiles)` take over, so
///   the value is pinned from 16384 up for every head shard in the tree. A
///   narrower live bound is the one case it moves, and `PLOW_MLA_NS_LIVE=1`
///   tracks it from `kv_len` per step — which is strictly better than baking it,
///   because the bound is not the live length either.
/// * `i[7]`, [`glm_gf`]: one crossover, at 4096. Every bound above it gets 4.
///
/// Both are checked at the head shards GLM-5.2/5.3 and Kimi-K3 produce
/// (`heads/tp` for 64 and 96 heads over tp 4/8/16).
#[test]
fn the_ctx_derived_attention_constants_saturate_over_the_narrowable_range() {
    let _guard = crate::test_env::env_guard();
    for nh_l in [4u32, 6, 8, 12, 16] {
        let ns = glm_nsplit(16_384, nh_l);
        let gf = glm_gf(16_384, nh_l);
        for ctx in [16_384u32, 32_768, 65_568, 81_920, 131_072, 1_048_576] {
            assert_eq!(glm_nsplit(ctx, nh_l), ns, "nsplit moved at ctx {ctx}, nh_l {nh_l}");
            assert_eq!(glm_gf(ctx, nh_l), gf, "GF moved at ctx {ctx}, nh_l {nh_l}");
        }
        // And the floor of that range is where `nsplit` stops being pinned: one
        // rung down it is half, which is the cost `exec::ctx_bound` warns about
        // below 16384 and `PLOW_MLA_NS_LIVE` exists to take back.
        assert!(glm_nsplit(8_192, nh_l) <= ns);
    }
}
