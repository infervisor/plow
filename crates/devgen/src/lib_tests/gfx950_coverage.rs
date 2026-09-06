//! The AMD opcode-coverage gate, and the drift test that keeps its list honest.
use super::*;

/// `MOE_MAX_TOPK` must equal `PLOW_MOE_MAX_TOPK` — PARSED out of `op_moe.h`, not restated.
///
/// The two halves of this bound live in different languages and different repos-worth of
/// build system, and only one of them refuses. If the Rust constant drifts HIGH the emit
/// happily produces a packet the kernel truncates silently; if it drifts LOW the emit refuses
/// a model that would have run. Neither shows up in any other test.
#[test]
fn moe_topk_matches_the_amd_kernel() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("runtime/amd/op_moe.h"));
    let Some(path) = root.filter(|p| p.exists()) else {
        eprintln!("op_moe.h not found — skipping (source checkout only)");
        return;
    };
    let src = std::fs::read_to_string(&path).unwrap();
    let def = src
        .lines()
        .find_map(|l| l.trim().strip_prefix("#define PLOW_MOE_MAX_TOPK "))
        .expect("op_moe.h has no `#define PLOW_MOE_MAX_TOPK`");
    let got: u32 = def
        .trim()
        .trim_end_matches('u')
        .parse()
        .unwrap_or_else(|e| panic!("cannot parse PLOW_MOE_MAX_TOPK {def:?}: {e}"));
    assert_eq!(
        got, MOE_MAX_TOPK,
        "devgen::MOE_MAX_TOPK ({MOE_MAX_TOPK}) disagrees with op_moe.h's \
             PLOW_MOE_MAX_TOPK ({got}). Raise them together: the kernel bound is what the routers \
             can select into, the Rust one is what refuses to emit past it."
    );
}

/// The refusal must name the model AND the limit, and must not fire at or below the bound.
/// A gate that refuses everything is as useless as one that refuses nothing.
#[test]
fn moe_topk_refusal_is_a_threshold_and_names_the_model() {
    for k in 1..=MOE_MAX_TOPK {
        require_moe_topk(k, "in-bounds");
    }
    // Expressed against the constant, not a literal: this test must keep meaning the same
    // thing the next time the bound moves.
    let over = MOE_MAX_TOPK + 1;
    let err = std::panic::catch_unwind(|| require_moe_topk(over, "kimi_k3")).unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or("");
    assert!(
        msg.contains("kimi_k3"),
        "refusal must name the model: {msg}"
    );
    assert!(
        msg.contains("PLOW_MOE_MAX_TOPK"),
        "must name the limit: {msg}"
    );
    assert!(
        msg.contains("uninitialised"),
        "must say what goes wrong: {msg}"
    );
}

/// The list must equal what `interp.hip` actually dispatches — PARSED, not restated. A
/// hand-maintained copy of a fact in another file is the drift `manifest.rs` was written to
/// stop; this is the same discipline `packet`'s `dev_abi` test applies to `dev_isa.h`.
///
/// Two dispatch FORMS, and missing the second would produce false refusals: most opcodes are
/// `case PLOW_DOP_X:` in the switch, but the TP collectives are handled by `if (in->op == ...)`
/// before it. A `case`-only parse undercounts them — which `docs/amd/model-op-coverage.md`
/// warns about in its opening lines.
#[test]
fn dispatched_list_matches_the_amd_interpreter() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("runtime/amd/interp.hip"));
    let Some(path) = root.filter(|p| p.exists()) else {
        eprintln!("interp.hip not found — skipping (source checkout only)");
        return;
    };
    let src = std::fs::read_to_string(&path).unwrap();
    let mut found: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        // `case PLOW_DOP_X:` — the switch arms.
        if let Some(r) = t.strip_prefix("case PLOW_DOP_") {
            let name: String = r
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            found.push(format!("PLOW_DOP_{name}"));
        }
        // `in->op == PLOW_DOP_X` — the collectives, dispatched ahead of the switch.
        let mut rest = t;
        while let Some(i) = rest.find("op == PLOW_DOP_") {
            let r = &rest[i + "op == PLOW_DOP_".len()..];
            let name: String = r
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            found.push(format!("PLOW_DOP_{name}"));
            rest = &r[name.len()..];
        }
    }
    found.sort();
    found.dedup();
    let mut want: Vec<String> = GFX950_DISPATCHED.iter().map(|s| s.to_string()).collect();
    want.sort();
    let missing_here: Vec<&String> = found.iter().filter(|n| !want.contains(n)).collect();
    let stale: Vec<&String> = want.iter().filter(|n| !found.contains(n)).collect();
    assert!(
            missing_here.is_empty() && stale.is_empty(),
            "GFX950_DISPATCHED disagrees with interp.hip.\n  interp has, list lacks: {missing_here:?}\n  \
             list has, interp lacks: {stale:?}\nAn over-long list lets a packet through that will \
             silently write nothing; a short one refuses a packet that would have run."
        );
}

/// The gate refuses an opcode with no AMD arm. `GemvArgmax` is the real instance: it has no
/// `case` in interp.hip, and `PLOW_FUSE_ARGMAX=1` emits it — a decode would have argmaxed over
/// an untouched buffer and returned token 0 every step, with no fault anywhere.
#[test]
#[should_panic(expected = "gfx950_opcode_arm")]
fn opcode_with_no_amd_arm_is_refused() {
    assert!(
        !GFX950_DISPATCHED.contains(&DevOp::GemvArgmax.c_name()),
        "if AMD ever gains a GEMV_ARGMAX arm this test should be re-pointed, not deleted"
    );
    let i = packet::dev::DevInst {
        op: DevOp::GemvArgmax as u16,
        blocks: 1,
        ..Default::default()
    };
    let p = packet::devbuild::Program {
        hier_base: 0,
        n_cu: 4,
        n_counter: 0,
        insts: vec![i],
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        tensors: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
        l2_sms: 0,
        l2_domains: 0,
    };
    let m = Model {
        n_cu: 256,
        target: 0,
        tensors: vec![],
        progs: vec![p],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    check_gfx950_opcode_coverage(&m, true);
}

/// …and lets an ordinary packet through, on both targets.
#[test]
fn covered_opcodes_pass_and_nvidia_is_never_checked() {
    let p = packet::devbuild::Program {
        hier_base: 0,
        n_cu: 4,
        n_counter: 0,
        insts: vec![
            packet::dev::DevInst {
                op: DevOp::Gemv as u16,
                blocks: 1,
                ..Default::default()
            },
            packet::dev::DevInst {
                op: DevOp::XReduce as u16,
                blocks: 1,
                ..Default::default()
            },
        ],
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        tensors: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
        l2_sms: 0,
        l2_domains: 0,
    };
    let m = Model {
        n_cu: 256,
        target: 0,
        tensors: vec![],
        progs: vec![p],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    check_gfx950_opcode_coverage(&m, true);
    // The Gemma-MoE family has no AMD arm at all; on an NVIDIA target that must not be checked.
    let p2 = packet::devbuild::Program {
        hier_base: 0,
        n_cu: 4,
        n_counter: 0,
        insts: vec![packet::dev::DevInst {
            op: DevOp::MoeRouterGemma as u16,
            blocks: 1,
            ..Default::default()
        }],
        stream: vec![],
        stream_ofs: vec![],
        stream_len: vec![],
        waits: vec![],
        succs: vec![],
        tensors: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![],
        l2_sms: 0,
        l2_domains: 0,
    };
    let m2 = Model {
        n_cu: 170,
        target: 0,
        tensors: vec![],
        progs: vec![p2],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    check_gfx950_opcode_coverage(&m2, false);
}

// ===== THE REVERSE DIRECTION ==================================================================
//
// Everything above asks "the packet carries opcode X — does gfx950 have an arm?". Neither of
// these does, and that asymmetry is the single most-repeated bug in this tree (~10 instances):
//
//     an arm exists, is correct, is register-gated, and NOTHING ROUTES TO IT.
//
// The runtime gate is structurally unable to see it. It inspects one emitted packet, built
// under one flag combination, so "no instruction selected this arm" and "this program did not
// need this arm" are the same observation. The reverse question is about the SOURCE — is there
// any reachable emit path at all — so it is asked here, against the source, exactly as
// `dispatched_list_matches_the_amd_interpreter` asks its question against `interp.hip`.
//
// Two checks, because "unreachable" has two shapes, and the expensive one is the second:
//   A. NO emit site anywhere. `every_dispatched_arm_has_an_emit_site`.
//   B. an emit site exists on ONE emitter family, and the flag that selects it is not read by
//      the others, so those silently emit the default precision. `PLOW_MXFP4` was this:
//      `DevOp::GemvMxfp4` is emitted (by `mla.rs`), so check A is green, and a dense model with
//      `PLOW_MXFP4=1` still produced a byte-identical-to-bf16 packet.
//      `precision_knob_table_matches_the_emitters`.

/// The emitter files. `manifest.rs` is EXCLUDED even though it names opcodes: it classifies a
/// finished stream (`DevOp::GemvMxfp4 => s.mxfp4_proj = true`), so counting it as an emit site
/// would let a reporter vouch for an arm no emitter reaches — the exact confusion this checks.
const EMITTER_SRC: &[&str] = &[
    "lib.rs",
    "mla.rs",
    "block.rs",
    "ladder.rs",
    "kda.rs",
    "k3.rs",
    "mla/kimi_k3.rs",
];

/// Arms gfx950 dispatches that NOTHING emits, each with why that is deliberate.
///
/// An allowlist, not a suppression: the justification is required, and an arm that leaves this
/// list without gaining an emit site fails the test. Adding a row is the moment to ask §4's
/// question — "what selects this, and is that selector complete over precisions?"
const GFX950_UNEMITTED: &[(&str, &str)] = &[
    // PLOW_DOP_HYPER_CONN_PRE/_POST: removed from this list — mla.rs now emits both
    // (the hyper-connections pre/post wiring landed ahead of glm53.rs itself).
    // PLOW_DOP_DSA_POOL_COMPRESS/_EXPAND/_STASH, PLOW_DOP_DSA_Q_QUANT,
    // PLOW_DOP_INDEX_SCORE_KPOOL, PLOW_DOP_GEMV_F32: removed from this list —
    // `emit_glm_dsa_decode_select`'s `index_kpool>1` branch (mla.rs) now emits all six.
    // The prefill twin (`emit_glm_dsa_prefill_select`) and `glm53.rs` itself are still
    // pending; HyperConnPre/HyperConnPost below remain unrouted until glm53.rs exists.
    (
        "PLOW_DOP_FLASH_GATHER_PREFILL",
        "Sparse MLA prefill needs one causal top-k index row per query token. IndexScore and \
          IndexSelect currently produce only a single-query index, so mla.rs deliberately emits \
          dense FlashMlaPrefill until those selectors gain a token axis.",
    ),
    (
        "PLOW_DOP_ATTN_SELECT",
        "DeepSeek DSA on-device top-k KV selection. The DSA path that ships emits IndexScore(58) \
          + IndexSelect(59) instead; ATTN_SELECT is the single-kernel alternative and is kept \
          because the two are being compared. Not a precision arm — no silent-wrong risk.",
    ),
    (
        "PLOW_DOP_O_UV_FOLD",
        "SUPERSEDED by MlaMergeFold(60), which fuses FlashMerge<512> + O_UV_FOLD into one packet \
          (mla.rs:1595 states the substitution). The arm stays for the unfused A/B.",
    ),
    (
        "PLOW_DOP_ROWRMS",
        "Precomputed row-RMS feeding GemmNorm's norm=1 mode (op_gemm.h:1086). Every emitter uses \
          the fused norm path, which needs no separate RMS packet.",
    ),
    (
        "PLOW_DOP_MOE_COMBINE_GEMMA",
        "SUPERSEDED on the Gemma-4 MoE path by MoeCombineNormGemma(70), which fuses \
          combine + RMSNorm + residual into one packet and is what lib.rs:3675 emits. Op 64 is \
          the unfused combine the sm_120 interpreter still carries; the AMD arm exists so the \
          two can be A/B'd and so a packet built before the fuse still runs. Not a precision \
          arm — no silent-wrong risk.",
    ),
    (
        "PLOW_DOP_MOE_EXPERT_GLU_GEMMA",
        "SUPERSEDED on the Gemma-4 MoE path by MoeExpertGluNormGemma(71), which folds the \
          pre-FFN RMSNorm into the expert gate/up dots and is what lib.rs:3599 emits. Op 62 is \
          the unfused twin (it takes an already-normed x); the AMD arm exists for the A/B and \
          for a pre-fuse packet. Not a precision arm.",
    ),
    (
        "PLOW_DOP_XFLASHMERGE",
        "CONTEXT-PARALLEL cross-rank LSE merge. TP shards attention by \
          whole heads, so no rank holds a partial for another rank's head and there is nothing to \
          merge. Emitted only once CP exists; blocked on S4.",
    ),
];

/// CHECK A — every arm gfx950 dispatches is either emitted by some emitter, or allowlisted with
/// a reason.
///
/// Deliberately coarse in the same way its forward twin is: "does any emitter file name this
/// `DevOp`", not "is that site reachable under the flags this build sets". A reachability
/// analysis would need the flag cross-product, and a wrong one would fail builds that work.
/// Naming is the cheap 90%: it catches an opcode added to `dev_isa.h` + `interp.hip` +
/// `GFX950_DISPATCHED` and then never wired, which is how five of the ~10 instances happened.
#[test]
fn every_dispatched_arm_has_an_emit_site() {
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut named: std::collections::BTreeSet<String> = Default::default();
    for f in EMITTER_SRC {
        let Ok(text) = std::fs::read_to_string(src_dir.join(f)) else {
            continue;
        };
        for line in text.lines() {
            // Comments are not emit sites. Without this, the paragraph explaining WHY an arm is
            // unwired would itself satisfy the check.
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            let mut rest = line;
            while let Some(i) = rest.find("DevOp::") {
                let r = &rest[i + "DevOp::".len()..];
                let n: String = r.chars().take_while(char::is_ascii_alphanumeric).collect();
                if !n.is_empty() {
                    named.insert(n.clone());
                }
                rest = &r[n.len()..];
            }
        }
    }
    assert!(
        named.len() > 20,
        "parsed only {} DevOp:: references from {EMITTER_SRC:?} — the parse broke, and a broken \
             parse here reports every arm as unemitted",
        named.len()
    );
    let allow: std::collections::BTreeMap<&str, &str> = GFX950_UNEMITTED.iter().copied().collect();
    let mut unemitted: Vec<&str> = Vec::new();
    let mut stale: Vec<&str> = Vec::new();
    for c in GFX950_DISPATCHED {
        let op = DevOp::ALL.iter().copied().find(|o| o.c_name() == *c);
        let emitted = op.is_some_and(|o| named.contains(&format!("{o:?}")));
        match (emitted, allow.contains_key(c)) {
            (false, false) => unemitted.push(c),
            (true, true) => stale.push(c),
            _ => {}
        }
    }
    assert!(
        unemitted.is_empty(),
        "gfx950 dispatches {} arm(s) NO emitter routes to: {unemitted:?}.\nThis is the recurring \
             shape: the arm exists, is correct, is register-gated, and nothing selects it — so it \
             ships in the object, is paid for in the register budget, and never runs. The runtime \
             coverage gate CANNOT see this (it only checks emitted opcode => arm exists).\nEither \
             wire an emit path, or add the opcode to GFX950_UNEMITTED with the reason it is \
             deliberately unrouted.",
        unemitted.len()
    );
    assert!(
        stale.is_empty(),
        "GFX950_UNEMITTED claims {stale:?} is deliberately unrouted, but an emitter now names \
             it. Drop the row — a stale allowlist entry re-hides the next real instance."
    );
}

/// How each emitter family treats a precision knob. The states are exhaustive on purpose:
/// there is no "not applicable", because a knob a family neither honours nor refuses is
/// SILENTLY IGNORED, which is the defect.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Knob {
    /// Read, and it selects a different arm here.
    Wired,
    /// Read, and refused with a message — the family has no arm for it.
    Refused,
    /// Not read at all. **This is the bug state.** A row may only be `Ignored` with a written
    /// justification, and every justification here should be read as a debt.
    Ignored(&'static str),
}

/// The precision axes × the two emitter families, and the file each family lives in.
///
/// WHY A TABLE AND NOT A GREP. The failure is not "a knob is unread"; it is "a knob is unread
/// *by one family* while another family honours it, so the same env var means fp8 over here
/// and nothing at all over there". Only a per-family statement can express that, and writing
/// the statement down is what makes the asymmetry visible at review time instead of at
/// benchmark time.
///
/// Scope is the PRECISION knobs specifically, because those are the ones whose silent
/// no-op yields an asset that runs, produces correct-looking output, and is wrong about what
/// it measured — §0's apples-to-apples rule.
///
/// This used to add "shape/scheduling knobs (`PLOW_GEMV_MM`, `PLOW_GQ_BATCH`, …) fail visibly
/// or not at all". THAT WAS FALSE, and `PLOW_GEMV_MM` is the counterexample: it is a KERNEL
/// knob, so the failure was not in either emitter family this table covers — no AMD build
/// input defined it at all, every gfx950 decode object compiled at op_gemm.h's default of 1,
/// and a B=4 asset with a correct B-wide program, 4× KV cache and `gv_mm_max: 4` in its
/// build.json produced ONE non-zero logits row. Sequences 1..B-1 sampled token 0 forever.
/// That is silent, and it survived because the two backends NAME the knob differently
/// (`GV_MM_MAX` on NVIDIA, `PLOW_GEMV_MM` on AMD) so a grep for either one looked wired.
///
/// The uncovered axis is therefore "devgen emits a program shaped by knob K, but the kernel
/// build for some backend never receives K". This table cannot see it — both emitter families
/// were correct here. Routed now in `scripts/build_gfx950.sh` and `runtime/CMakeLists.txt`.
///
/// THE GUARD OVER KERNEL-BUILD INPUTS NOW EXISTS, for this knob. Routing it was necessary and
/// not sufficient: routing fixes the objects someone remembers to rebuild, and says nothing
/// about the pairing of a given packet with a given object directory. `PLOW_GEMV_MM` is now
/// EMITTED INTO the object as `plow_gemv_mm_cap_<N>` (`runtime/amd/op_gemm.h`, named for the
/// macro itself so it cannot disagree with what was compiled), and `check_gemv_capacity`
/// (`crates/plowrt/src/exec/amd.rs`) refuses at load when the packet's widest GEMV asks for
/// more rows than the object advertises — or when it advertises nothing at all. That is the
/// shape any future entry on this axis should take: make the kernel build input OBSERVABLE in
/// the object, then compare it against the packet where the two finally meet.
/// Maps env-var knob name → EmitConfig field name(s) for the source grep.
/// After Phase 3b migration, emitters access `emit_config::active().field`
/// rather than `std::env::var("KNOB")`. Both patterns count as "reads".
fn knob_field_names(knob: &str) -> &'static [&'static str] {
    match knob {
        "PLOW_FP8" => &[".fp8"],
        "PLOW_W8A16" => &[".w8a16"],
        "PLOW_W8A8" => &[".w8a8"],
        "PLOW_MXFP4" => &[".mxfp4"],
        "PLOW_FP8_KV" => &[".fp8_kv"],
        "PLOW_KV_FP8" => &[".fp8_kv", "PLOW_KV_FP8"],
        _ => &[],
    }
}

const PRECISION_KNOBS: &[(&str, Knob, Knob)] = &[
    // knob            dense-GQA (lib.rs)                    MLA/MoE (mla.rs)
    ("PLOW_FP8", Knob::Wired, Knob::Wired),
    ("PLOW_W8A16", Knob::Wired, Knob::Wired),
    ("PLOW_W8A8", Knob::Wired, Knob::Refused),
    ("PLOW_MXFP4", Knob::Refused, Knob::Wired),
    // The K3 full-model path in mla.rs now emits the compressed-latent fp8 twins. Other MLA
    // entry points still need the same wiring, but the family no longer silently ignores the
    // knob universally; K3's structural tests pin the allocation and opcode swap.
    ("PLOW_FP8_KV", Knob::Wired, Knob::Wired),
    ("PLOW_KV_FP8", Knob::Wired, Knob::Wired),
];

/// CHECK B — the table above is true of the sources.
///
/// Catches bug shape B directly: a precision knob wired on one family and unread on another.
/// Before `PLOW_MXFP4` was refused on the dense path, its dense column was `Ignored`, and the
/// only way to make this test pass was to WRITE DOWN that a dense `PLOW_MXFP4=1` build emits
/// bf16 — which nobody would have written down and left.
///
/// The evidence is `env::var("KNOB")`, not a mention: a comment naming the flag is exactly what
/// the dense path had, and it is worth nothing at runtime.
#[test]
fn precision_knob_table_matches_the_emitters() {
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let read = |f: &str| std::fs::read_to_string(src_dir.join(f)).unwrap_or_default();
    let dense = read("lib.rs");
    let mla = read("mla.rs") + &read("mla/kimi_k3.rs");
    assert!(
        !dense.is_empty() && !mla.is_empty(),
        "emitter sources not readable"
    );
    for (knob, d, m) in PRECISION_KNOBS {
        for (family, file, src, state) in [
            ("dense-GQA", "lib.rs", &dense, d),
            ("MLA/MoE", "mla.rs", &mla, m),
        ] {
            // Post-migration: emitters access `emit_config::active().field` instead of
            // `std::env::var("KNOB")`. Both patterns count as "reading" the knob.
            let reads = src.contains(&format!("env::var(\"{knob}\")"))
                || knob_field_names(knob).iter().any(|f| src.contains(f));
            match state {
                    Knob::Wired | Knob::Refused => assert!(
                        reads,
                        "PRECISION_KNOBS says the {family} emitter handles {knob} as {state:?}, but \
                         {file} contains no `env::var(\"{knob}\")`. Either it never did and the \
                         table is wrong, or a refactor dropped the read — in which case {knob}=1 is \
                         now SILENTLY IGNORED on {family} and that build emits the default \
                         precision under a flag that named another one."
                    ),
                    Knob::Ignored(why) => {
                        assert!(
                            !why.is_empty(),
                            "{knob} is Ignored on {family} with no justification. An unread \
                             precision knob is a silently-wrong asset; say why in the table."
                        );
                        assert!(
                            !reads,
                            "PRECISION_KNOBS says {knob} is IGNORED on {family}, but {file} now \
                             reads it. Good — update the row to Wired or Refused so the next reader \
                             is not told a fixed hole is still open."
                        );
                    }
                }
        }
    }
}

/// The instance this whole section exists for, pinned as behaviour rather than as a table row:
/// `PLOW_MXFP4=1` on a dense model must not hand back a bf16 packet.
///
/// This used to assert a REFUSAL, which was correct while gfx950 had no arm for
/// `GEMV_MXFP4`/`GEMV_GLU_MXFP4` — emitting then would have produced a packet byte-identical to
/// the bf16 one and benchmarked it as mxfp4. Both opcodes are in `GFX950_DISPATCHED` now, so the
/// emit legitimately succeeds and the invariant is checked directly instead: the decode program
/// has to carry the MXFP4 opcodes.
///
/// Emission reads process-global env, so the variable is restored on the way out whether or
/// not the assert fires — leaving it set would change every blob a later test in this binary
/// emits. (`tests/golden_blob.rs` runs in a separate process and takes its own `EMIT_LOCK`.)
#[test]
fn dense_mxfp4_is_not_silently_bf16() {
    let dir = std::env::temp_dir().join("devgen_dense_mxfp4_refusal");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"qwen3","hidden_size":512,"intermediate_size":1024,
                "num_hidden_layers":2,"num_attention_heads":8,"head_dim":64,
                "num_key_value_heads":2,"rms_norm_eps":1e-6,"vocab_size":4096,
                "rope_theta":1000000.0,"rope_scaling":null,"tie_word_embeddings":true}"#,
    )
    .unwrap();
    let out = dir.join("model.pkt");
    let args = || EmitArgs {
        dir: dir.clone(),
        ctx: 256,
        out: out.to_str().unwrap().to_string(),
        n_cu: 256,
        tp: 1,
        block_spec: None,
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen: true,
        l2_layout: None,
        gpu: "MI355X".into(),
        arch: "gfx950".into(),
        emit_cfg: None,
        whole_graph_fusions: WholeGraphFusionDecisions::default(),
    };
    // The emit now runs to completion instead of refusing immediately, so PLOW_MXFP4 is live for
    // far longer; without the shared lock a concurrent emitter test in this binary sees it and
    // emits MXFP4 opcodes its own contract asserts reject.
    let _env = crate::test_env::env_guard();
    let _scope = crate::test_env::EnvScope::set(&[("PLOW_MXFP4", "1")]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(args())));
    r.expect("PLOW_MXFP4=1 must emit on a target whose interpreter dispatches the MXFP4 opcodes");
    // The emitted file is the artifact under test; scan its raw bytes for the MXFP4 opcodes
    // rather than pulling in plowrt's blob parser (devgen cannot depend on it).
    let raw = std::fs::read(&out).expect("emitted packet");
    let mxfp4 = [DevOp::GemvMxfp4 as u16, DevOp::GemvGluMxfp4 as u16]
        .iter()
        .any(|op| raw.windows(2).any(|w| u16::from_le_bytes([w[0], w[1]]) == *op));
    assert!(
        mxfp4,
        "PLOW_MXFP4=1 emitted a packet with no MXFP4 opcode. It is byte-identical to the bf16 one \
         and would be benchmarked as mxfp4."
    );
}
