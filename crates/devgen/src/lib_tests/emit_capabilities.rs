use super::*;
use clap::Parser;

#[derive(Parser)]
struct EmitArgsForTest {
    #[command(flatten)]
    emit: emit_config::EmitConfig,
}

#[test]
fn packet_capabilities_are_explicit() {
    for model_type in [
        "gemma3",
        "gemma3_text",
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_unified_text",
        "llama",
        "qwen3",
    ] {
        let capabilities = emit_capabilities(model_type);
        assert!(capabilities.dense_packet_contracts);
        assert_eq!(capabilities.gemma, model_type.starts_with("gemma"));
        assert!(capabilities.decode_objects);
        assert_eq!(
            capabilities.cublaslt_decode,
            model_type.starts_with("gemma")
        );
        assert!(capabilities.decode_ladder);
    }
    let qwen = emit_capabilities("qwen3_5");
    assert!(!qwen.dense_packet_contracts);
    assert!(!qwen.gemma);
    assert!(qwen.decode_objects);
    assert!(qwen.cublaslt_decode);
    assert!(!qwen.decode_ladder);
    assert!(emit_capabilities("gpt_oss").decode_ladder);
    for model_type in ["kimi_k3", "glm5_next", "unknown"] {
        let capabilities = emit_capabilities(model_type);
        assert!(!capabilities.dense_packet_contracts);
        assert!(!capabilities.decode_objects);
        assert!(!capabilities.cublaslt_decode);
        assert!(!capabilities.decode_ladder);
    }
}

#[test]
fn production_defaults_are_capability_and_target_driven() {
    for model_type in ["gemma3", "gemma3_text", "gemma4"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model_type), "sm_90a", 1, 304);
        assert_eq!(cfg.decode_rungs(), [1, 2, 4, 8, 16]);
        assert!(cfg.decode_ladder_default);
        assert!(cfg.packed_prefill_on());

        let mut disabled = EmitArgsForTest::try_parse_from([
            "test",
            "--emit-decode-batch-ladder=1",
            "--emit-packed-prefill=false",
        ])
        .unwrap()
        .emit;
        apply_production_defaults(&mut disabled, emit_capabilities(model_type), "sm_90a", 1, 304);
        assert_eq!(disabled.decode_rungs(), [1]);
        assert!(!disabled.decode_ladder_default);
        assert!(!disabled.packed_prefill_on());
    }

    let mut unsupported = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut unsupported, emit_capabilities("qwen3_5"), "sm_90a", 1, 304);
    assert_eq!(unsupported.decode_rungs(), [1]);
    assert!(!unsupported.packed_prefill_on());

    // gfx942 STOPS AT 8, and NOT for the reason this comment used to give. The 16 rung is a
    // 25% device-throughput WIN there (130.7 -> 163.3 tok/s, `amd-bench --batched`, n_cu 304);
    // what it costs is the NARROWER rungs, because a ladder has one decode object and the
    // MM=16 that rung needs takes served c=8 from 117.3 to 85.0 tok/s. See
    // `apply_production_defaults`. Pin the width so a later "make the ladders match" tidy-up
    // has to argue with that instead of silently adding it.
    let mut amd = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut amd, emit_capabilities("gemma4"), "gfx942", 1, 304);
    assert_eq!(amd.decode_rungs(), [1, 2, 4, 8]);
    assert!(amd.decode_ladder_default);
    assert!(amd.packed_prefill_on());
    amd.emit_packed_prefill = Some(false);
    assert!(!amd.packed_prefill_on());

    let mut other_target = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut other_target, emit_capabilities("gemma4"), "gfx950", 1, 304);
    assert_eq!(other_target.decode_rungs(), [1]);
    assert!(!other_target.packed_prefill_on());
}

#[test]
fn pure_gemm_default_is_scoped_to_sm90a_gemma_w8a8() {
    let _guard = crate::test_env::env_guard();
    let mut qualified =
        EmitArgsForTest::try_parse_from(["test", "--w8a8"]).unwrap().emit;
    apply_production_defaults(
        &mut qualified,
        emit_capabilities("gemma4"),
        "sm_90a",
        1,
        132,
    );
    assert_eq!(qualified.seg_pure_gemm.as_deref(), Some("1"));
    emit_config::install(qualified.clone());
    assert_eq!(
        packet::devbuild::knobs().seg_pure_gemm.as_deref(),
        Some("1")
    );

    let mut disabled = EmitArgsForTest::try_parse_from([
        "test",
        "--w8a8",
        "--emit-pure-gemm-segments=0",
    ])
    .unwrap()
    .emit;
    apply_production_defaults(
        &mut disabled,
        emit_capabilities("gemma4"),
        "sm_90a",
        1,
        132,
    );
    assert_eq!(disabled.seg_pure_gemm.as_deref(), Some("0"));

    for (model, arch, tp, w8a8) in [
        ("gemma4", "sm_90a", 1, false),
        ("gemma4", "gfx942", 1, true),
        ("gemma4", "sm_90a", 2, true),
        ("qwen3", "sm_90a", 1, true),
    ] {
        let argv = if w8a8 { vec!["test", "--w8a8"] } else { vec!["test"] };
        let mut cfg = EmitArgsForTest::try_parse_from(argv).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), arch, tp, 132);
        assert_eq!(cfg.seg_pure_gemm, None, "{model} {arch} tp={tp}");
    }
}

#[test]
fn amd_packed_defaults_require_dense_bf16_single_gpu() {
    for flag in ["--fp8", "--w8a8", "--w8a16", "--fp8-kv"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test", flag])
            .unwrap()
            .emit;
        apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "gfx942", 1, 304);
        assert!(!cfg.packed_prefill_on(), "{flag}");
    }
    for (model, tp) in [("gemma4", 2), ("qwen3_5", 1), ("kimi_k3", 1)] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), "gfx942", tp, 304);
        assert!(!cfg.packed_prefill_on(), "{model} tp={tp}");
    }
}

#[test]
fn qualified_fp8_weight_metadata_follows_production_defaults() {
    for flag in ["--fp8", "--w8a16"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test", flag])
            .unwrap()
            .emit;
        apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1, 304);
        assert!(cfg.packed_prefill_on());
        assert!(cfg.packed_prefill_metadata_on());
        cfg.emit_packed_prefill = Some(true);
        assert!(cfg.packed_prefill_metadata_on());
        cfg.emit_packed_prefill = Some(false);
        assert!(!cfg.packed_prefill_on());
        assert!(!cfg.packed_prefill_metadata_on());
    }
    let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1, 304);
    assert!(cfg.packed_prefill_metadata_on());
}

#[test]
fn activation_fp8_packing_still_requires_explicit_selection() {
    let mut cfg = EmitArgsForTest::try_parse_from(["test", "--w8a8"])
        .unwrap()
        .emit;
    apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1, 304);
    assert!(cfg.packed_prefill_on());
    assert!(!cfg.packed_prefill_metadata_on());
    cfg.emit_packed_prefill = Some(true);
    assert!(cfg.packed_prefill_metadata_on());
}

#[test]
fn cublaslt_emission_rejects_unloadable_combinations() {
    let qwen = emit_capabilities("qwen3_5");
    assert!(cublaslt_emit_supported(qwen, "sm_90a", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_120", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "gfx950", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_90a", 2, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_90a", 1, true));
    for model_type in ["gemma3", "gemma3_text", "gemma4"] {
        assert!(cublaslt_emit_supported(
            emit_capabilities(model_type),
            "sm_90a",
            1,
            false,
        ));
    }
}


/// The nine `glm_*` knobs of the qualified gfx942 TP8 recipe are ON by default for GLM on that
/// target and OFF everywhere else.
///
/// The target predicate is not decoration: `glm_gemm_lt`, `glm_index_tp`, `glm_moe_aiter` and
/// `glm_moe_resident` each assert gfx942 / `tp == 8` / `n_cu == 304` at their emit site, so a
/// default that fired more widely would turn a working TP1 or MI300A GLM emit into a panic.
#[test]
/// The two knobs the combined tier 4 qualified (job `1789228962-combined-t4`, +7.5% out tok/s):
/// the prefill hipBLASLt groups `o_proj,band,shared` (NOT `router`, review log #65) and the decode
/// GEMM grouping. Both are recipe defaults on the qualified target only, and both roll back.
#[test]
fn glm_qualified_recipe_defaults_the_lt_groups_and_decode_gemm_group() {
    let _guard = crate::test_env::env_guard();
    let qualified = |args: &[&str]| {
        let mut cfg = EmitArgsForTest::try_parse_from(args).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities("glm_moe_dsa"), "gfx942", 8, 304);
        cfg
    };
    let cfg = qualified(&["test"]);
    assert!(cfg.glm_decode_gemm_group());
    assert_eq!(cfg.glm_gemm_lt_pf_ext_spec(), emit_config::GLM_GEMM_LT_PF_EXT_QUALIFIED);
    assert!(!cfg.glm_gemm_lt_pf_ext_spec().contains("router"));

    // Not the qualified target: both stay off, so a TP1 or gfx950 emit is unchanged.
    let mut tp1 = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut tp1, emit_capabilities("glm_moe_dsa"), "gfx942", 1, 304);
    assert!(!tp1.glm_decode_gemm_group());
    assert_eq!(tp1.glm_gemm_lt_pf_ext_spec(), "");

    // Per-knob rollback.
    let off = qualified(&["test", "--glm-decode-gemm-group=false", "--glm-gemm-lt-pf-ext", ""]);
    assert!(!off.glm_decode_gemm_group());
    assert_eq!(off.glm_gemm_lt_pf_ext_spec(), "");
}

#[test]
fn glm_production_recipe_defaults_on_only_for_the_qualified_target() {
    let _guard = crate::test_env::env_guard();
    let resolved = |cfg: &emit_config::EmitConfig| {
        [
            cfg.glm_fp8_kv(),
            cfg.glm_moe_aiter(),
            cfg.glm_moe_resident(),
            cfg.glm_index_tp(),
            cfg.glm_select_local(),
            cfg.glm_decode_norm_rows(),
            cfg.glm_gemm_lt(),
            cfg.glm_gemm_lt_decode(),
            cfg.glm_gemm_lt_decode_ext(),
        ]
    };
    for model in ["glm_moe_dsa", "glm5_next"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), "gfx942", 8, 304);
        assert_eq!(resolved(&cfg), [true; 9], "{model}");
        assert!(cfg.packed_prefill_on(), "{model}");
    }
    for (model, arch, tp, n_cu, flag) in [
        ("glm_moe_dsa", "sm_90a", 8u32, 304u32, None),
        ("glm_moe_dsa", "gfx950", 8, 304, None),
        ("glm_moe_dsa", "gfx942", 1, 304, None),
        ("glm_moe_dsa", "gfx942", 4, 304, None),
        // MI300A is gfx942 with 228 CUs; the native MoE and hipBLASLt arms are 304-CU only.
        ("glm_moe_dsa", "gfx942", 8, 228, None),
        // A4W4 has no block-fp8 expert arm for the native MoE to bind.
        ("glm_moe_dsa", "gfx942", 8, 304, Some("--mxfp4")),
        ("gemma4", "gfx942", 8, 304, None),
        ("kimi_k3", "gfx942", 8, 304, None),
        ("qwen3_5", "gfx942", 8, 304, None),
    ] {
        let argv: Vec<&str> = ["test"].into_iter().chain(flag).collect();
        let mut cfg = EmitArgsForTest::try_parse_from(argv).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), arch, tp, n_cu);
        assert_eq!(
            resolved(&cfg),
            [false; 9],
            "{model} {arch} tp={tp} n_cu={n_cu} {flag:?}"
        );
    }
}

/// Build one `EmitConfig` the way `plowc` does — clap parse, knob record, production defaults —
/// and hand back the resolved config plus a `build.json`-style `id -> (value, source)` lookup.
fn resolve_like_plowc(
    argv: &[&str],
    model_type: &str,
    arch: &str,
    tp: u32,
    n_cu: u32,
) -> (
    emit_config::EmitConfig,
    std::collections::BTreeMap<String, (String, &'static str)>,
) {
    use clap::{Args, FromArgMatches};
    let matches = emit_config::EmitConfig::augment_args(clap::Command::new("test"))
        .try_get_matches_from(argv)
        .unwrap();
    let mut cfg = emit_config::EmitConfig::from_arg_matches(&matches).unwrap();
    emit_config::record_knobs(Some(&matches));
    apply_production_defaults(&mut cfg, emit_capabilities(model_type), arch, tp, n_cu);
    let recorded = emit_config::knobs_or_env()
        .into_iter()
        .map(|k| (k.id, (k.value, k.source)))
        .collect();
    (cfg, recorded)
}

/// PRECEDENCE, and `build.json` naming the winner.
///
/// `--replay-knobs` is deliberately not a fourth source: `plowc` applies a replayed recipe by
/// `set_var`-ing the recorded `PLOW_*` assignments BEFORE clap parses, and only where the
/// variable is not already set. So a replayed value arrives as clap's `EnvVariable` — which is
/// what the env arm below pins — and a replayed `false` therefore survives the production
/// default, which is what makes re-emitting a frozen recipe reproduce it.
///
/// One record per process in production; one per test thread here. Each source is asserted on a
/// DIFFERENT knob so the three coexist in a single record rather than overwriting each other.
#[test]
fn glm_recipe_precedence_is_cli_then_env_then_production_default() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[("PLOW_GLM_SELECT_LOCAL", "0")]);
    let (cfg, rec) = resolve_like_plowc(
        &["test", "--glm-gemm-lt=false"],
        "glm_moe_dsa",
        "gfx942",
        8,
        304,
    );
    // Explicit CLI false beats the production default, and is recorded as the user's.
    assert!(!cfg.glm_gemm_lt());
    assert_eq!(rec["glm_gemm_lt"], ("false".into(), "cli"));
    // Env (this is also where a replayed recipe lands) beats the production default.
    assert!(!cfg.glm_select_local());
    assert_eq!(rec["glm_select_local"], ("0".into(), "env"));
    // Everything nobody spoke for is the qualified recipe, recorded as such — which is what
    // keeps it OUT of `emit_config.replay`, whose filter is `cli | env`.
    for id in [
        "glm_fp8_kv",
        "glm_moe_aiter",
        "glm_moe_resident",
        "glm_index_tp",
        "glm_decode_norm_rows",
        "glm_gemm_lt_decode",
        "glm_gemm_lt_decode_ext",
        "glm_fold_lt",
        "glm_seq_par",
        "glm_seq_par_proj",
    ] {
        assert_eq!(rec[id], ("true".into(), "production_default"), "{id}");
    }
    assert!(cfg.glm_fold_lt() && cfg.glm_seq_par() && cfg.glm_seq_par_proj());
    assert!(
        cfg.glm_fp8_kv()
            && cfg.glm_moe_aiter()
            && cfg.glm_moe_resident()
            && cfg.glm_index_tp()
            && cfg.glm_decode_norm_rows()
            && cfg.glm_gemm_lt_decode()
            && cfg.glm_gemm_lt_decode_ext()
    );
}

/// A flag beats the env var it shares a knob with, production default or not.
#[test]
fn glm_recipe_cli_beats_env() {
    let _guard = crate::test_env::env_guard();
    let _env = crate::test_env::EnvScope::set(&[
        ("PLOW_GLM_GEMM_LT", "0"),
        ("PLOW_GLM_MOE_AITER", "1"),
    ]);
    let (cfg, rec) = resolve_like_plowc(
        &["test", "--glm-gemm-lt=true", "--glm-moe-aiter=false"],
        "glm_moe_dsa",
        "gfx942",
        8,
        304,
    );
    assert!(cfg.glm_gemm_lt() && !cfg.glm_moe_aiter());
    assert_eq!(rec["glm_gemm_lt"], ("true".into(), "cli"));
    assert_eq!(rec["glm_moe_aiter"], ("false".into(), "cli"));
}

/// Off the qualified target an unset knob stays off and stays `default`, so nothing about a
/// Gemma, Kimi or sm_90a GLM emit moves.
#[test]
fn non_qualified_targets_record_no_glm_production_default() {
    let _guard = crate::test_env::env_guard();
    let (cfg, rec) = resolve_like_plowc(&["test"], "glm_moe_dsa", "sm_90a", 8, 304);
    assert!(!cfg.glm_gemm_lt() && !cfg.glm_fp8_kv() && !cfg.glm_moe_resident());
    for (id, _, _) in cfg.glm_recipe_unset() {
        assert_eq!(rec[id].1, "default", "{id}");
    }
    assert!(!cfg.glm_fold_lt() && !cfg.glm_seq_par() && !cfg.glm_seq_par_proj());
}

/// The three tier-4 knobs (`PLOW_GLM_SEQ_PAR`, `_PROJ`, `PLOW_GLM_FOLD_LT`): each rolls back on
/// its own flag; `_PROJ`'s default follows `SEQ_PAR`, so rolling back the seams alone takes it
/// along; the seams' default stands aside for the two-shot seam knobs it replaces; and nothing
/// turns on for a non-GLM gfx942 TP8 emit.
#[test]
fn glm_seq_par_and_fold_defaults_roll_back_per_knob() {
    let _guard = crate::test_env::env_guard();
    type Rec = std::collections::BTreeMap<String, (String, &'static str)>;
    // One knob record per thread (production has one per process): resolve each case fresh.
    let resolve = |argv: &[&str], model: &str| -> ((bool, bool, bool), Rec) {
        let argv: Vec<String> = argv.iter().map(|a| a.to_string()).collect();
        let model = model.to_string();
        std::thread::spawn(move || {
            let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
            let (cfg, rec) = resolve_like_plowc(&argv, &model, "gfx942", 8, 304);
            ((cfg.glm_seq_par(), cfg.glm_seq_par_proj(), cfg.glm_fold_lt()), rec)
        })
        .join()
        .unwrap()
    };
    let glm = |argv: &[&str]| resolve(argv, "glm_moe_dsa");

    assert_eq!(glm(&["test"]).0, (true, true, true));

    let (on, rec) = glm(&["test", "--glm-fold-lt=false"]);
    assert_eq!(on, (true, true, false));
    assert_eq!(rec["glm_fold_lt"], ("false".into(), "cli"));

    let (on, rec) = glm(&["test", "--glm-seq-par-proj=false"]);
    assert_eq!(on, (true, false, true));
    assert_eq!(rec["glm_seq_par_proj"], ("false".into(), "cli"));

    let (on, rec) = glm(&["test", "--glm-seq-par=false"]);
    assert_eq!(on, (false, false, true));
    assert_eq!(rec["glm_seq_par"], ("false".into(), "cli"));
    assert_eq!(rec["glm_seq_par_proj"], ("false".into(), "production_default"));

    for argv in [&["test", "--glm-xr-res=true"][..], &["test", "--glm-xr-band", "4"][..]] {
        let (on, rec) = glm(argv);
        assert_eq!(on, (false, false, true), "{argv:?}");
        assert_eq!(rec["glm_seq_par"], ("false".into(), "production_default"), "{argv:?}");
    }

    let (on, rec) = resolve(&["test"], "gemma4");
    assert_eq!(on, (false, false, false));
    for id in ["glm_fold_lt", "glm_seq_par", "glm_seq_par_proj"] {
        assert_eq!(rec[id].1, "default", "{id}");
    }
}
