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
        assert!(capabilities.decode_objects);
        assert_eq!(
            capabilities.cublaslt_decode,
            model_type.starts_with("gemma")
        );
        assert!(capabilities.decode_ladder);
    }
    let qwen = emit_capabilities("qwen3_5");
    assert!(!qwen.dense_packet_contracts);
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


/// The eight `glm_*` knobs of the qualified gfx942 TP8 recipe are ON by default for GLM on that
/// target and OFF everywhere else.
///
/// The target predicate is not decoration: `glm_gemm_lt`, `glm_index_tp`, `glm_moe_aiter` and
/// `glm_moe_resident` each assert gfx942 / `tp == 8` / `n_cu == 304` at their emit site, so a
/// default that fired more widely would turn a working TP1 or MI300A GLM emit into a panic.
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
        ]
    };
    for model in ["glm_moe_dsa", "glm5_next"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), "gfx942", 8, 304);
        assert_eq!(resolved(&cfg), [true; 8], "{model}");
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
            [false; 8],
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
    ] {
        assert_eq!(rec[id], ("true".into(), "production_default"), "{id}");
    }
    assert!(
        cfg.glm_fp8_kv()
            && cfg.glm_moe_aiter()
            && cfg.glm_moe_resident()
            && cfg.glm_index_tp()
            && cfg.glm_decode_norm_rows()
            && cfg.glm_gemm_lt_decode()
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
    for (id, _) in cfg.glm_recipe_unset() {
        assert_eq!(rec[id].1, "default", "{id}");
    }
}
