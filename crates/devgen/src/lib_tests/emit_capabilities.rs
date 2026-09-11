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
        apply_production_defaults(&mut cfg, emit_capabilities(model_type), "sm_90a", 1);
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
        apply_production_defaults(&mut disabled, emit_capabilities(model_type), "sm_90a", 1);
        assert_eq!(disabled.decode_rungs(), [1]);
        assert!(!disabled.decode_ladder_default);
        assert!(!disabled.packed_prefill_on());
    }

    let mut unsupported = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut unsupported, emit_capabilities("qwen3_5"), "sm_90a", 1);
    assert_eq!(unsupported.decode_rungs(), [1]);
    assert!(!unsupported.packed_prefill_on());

    // gfx942 STOPS AT 8, and NOT for the reason this comment used to give. The 16 rung is a
    // 25% device-throughput WIN there (130.7 -> 163.3 tok/s, `amd-bench --batched`, n_cu 304);
    // what it costs is the NARROWER rungs, because a ladder has one decode object and the
    // MM=16 that rung needs takes served c=8 from 117.3 to 85.0 tok/s. See
    // `apply_production_defaults`. Pin the width so a later "make the ladders match" tidy-up
    // has to argue with that instead of silently adding it.
    let mut amd = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut amd, emit_capabilities("gemma4"), "gfx942", 1);
    assert_eq!(amd.decode_rungs(), [1, 2, 4, 8]);
    assert!(amd.decode_ladder_default);
    assert!(amd.packed_prefill_on());
    amd.emit_packed_prefill = Some(false);
    assert!(!amd.packed_prefill_on());

    let mut other_target = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut other_target, emit_capabilities("gemma4"), "gfx950", 1);
    assert_eq!(other_target.decode_rungs(), [1]);
    assert!(!other_target.packed_prefill_on());
}

#[test]
fn amd_packed_defaults_require_dense_bf16_single_gpu() {
    for flag in ["--fp8", "--w8a8", "--w8a16", "--fp8-kv"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test", flag])
            .unwrap()
            .emit;
        apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "gfx942", 1);
        assert!(!cfg.packed_prefill_on(), "{flag}");
    }
    for (model, tp) in [("gemma4", 2), ("qwen3_5", 1), ("kimi_k3", 1)] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
        apply_production_defaults(&mut cfg, emit_capabilities(model), "gfx942", tp);
        assert!(!cfg.packed_prefill_on(), "{model} tp={tp}");
    }
}

#[test]
fn qualified_fp8_weight_metadata_follows_production_defaults() {
    for flag in ["--fp8", "--w8a16"] {
        let mut cfg = EmitArgsForTest::try_parse_from(["test", flag])
            .unwrap()
            .emit;
        apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1);
        assert!(cfg.packed_prefill_on());
        assert!(cfg.packed_prefill_metadata_on());
        cfg.emit_packed_prefill = Some(true);
        assert!(cfg.packed_prefill_metadata_on());
        cfg.emit_packed_prefill = Some(false);
        assert!(!cfg.packed_prefill_on());
        assert!(!cfg.packed_prefill_metadata_on());
    }
    let mut cfg = EmitArgsForTest::try_parse_from(["test"]).unwrap().emit;
    apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1);
    assert!(cfg.packed_prefill_metadata_on());
}

#[test]
fn activation_fp8_packing_still_requires_explicit_selection() {
    let mut cfg = EmitArgsForTest::try_parse_from(["test", "--w8a8"])
        .unwrap()
        .emit;
    apply_production_defaults(&mut cfg, emit_capabilities("gemma4"), "sm_90a", 1);
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
