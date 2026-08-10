//! Reading the weight encoding off the CHECKPOINT rather than off a flag.
//!
//! The first real GLM-5.2 emit produced a bf16 block from a checkpoint that is block-fp8 on
//! disk — asking the loader to bind bf16 weights that do not exist, and never reaching the
//! block-fp8 expert arms built for that exact model family. These pin the parse against the
//! shapes that actually appear in `zai-org/GLM-5.2-FP8`'s `config.json`.
use super::*;

fn cfg_dir(name: &str, body: &str) -> std::path::PathBuf {
    // CARGO_TARGET_TMPDIR is only defined for integration tests, not unit tests.
    let d = std::env::temp_dir().join(format!("plow_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("config.json"), body).unwrap();
    d
}

/// The real GLM-5.2-FP8 shape, including the two things that would fool a dtype-keyed probe:
/// the key is `dtype` and not `torch_dtype`, and its value is "bfloat16" — the COMPUTE dtype —
/// on a checkpoint whose weights are e4m3.
#[test]
fn block_fp8_checkpoint_is_detected_despite_a_bfloat16_dtype_field() {
    let d = cfg_dir(
        "ckpt_fp8",
        r#"{"model_type":"glm_moe_dsa","dtype":"bfloat16",
                "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
                "quant_method":"fp8","weight_block_size":[128,128]}}"#,
    );
    assert_eq!(mla_ckpt_enc(&d), Some(MoeEnc::Fp8Blk));
}

/// No `quantization_config` => the historical path, where the env flags decide and nothing
/// about an existing workflow changes.
#[test]
fn unquantized_checkpoint_leaves_the_decision_to_the_flags() {
    let d = cfg_dir(
        "ckpt_plain",
        r#"{"model_type":"kimi_k2","dtype":"bfloat16"}"#,
    );
    assert_eq!(mla_ckpt_enc(&d), None);
}

/// 128 is not a parameter anywhere in this emitter — every scale-grid size is written as
/// `div_ceil(128)` — so a checkpoint quantized at another block size would bind grids of the
/// wrong shape against weights that look perfectly fine. The field exists because it can vary.
#[test]
#[should_panic(expected = "fp8_block_size")]
fn a_different_block_size_is_refused() {
    let d = cfg_dir(
        "ckpt_blk64",
        r#"{"quantization_config":{"quant_method":"fp8","fmt":"e4m3",
                "weight_block_size":[64,64]}}"#,
    );
    mla_ckpt_enc(&d);
}

/// A quantization this emitter has no arms for must REFUSE, not fall back to bf16: the weights
/// on disk are not bf16, so a bf16 packet is a WRONG packet rather than an unoptimised one.
/// Same rule as w8a16-on-gfx950.
#[test]
#[should_panic(expected = "ckpt_quant_awq")]
fn an_unsupported_quantization_is_refused_rather_than_downgraded() {
    let d = cfg_dir(
        "ckpt_awq",
        r#"{"quantization_config":{"quant_method":"awq"}}"#,
    );
    mla_ckpt_enc(&d);
}

/// A non-e4m3 fp8 flavour is refused too — `fmt` is checked, not assumed.
#[test]
#[should_panic(expected = "fp8_fmt_e5m2")]
fn a_non_e4m3_fp8_format_is_refused() {
    let d = cfg_dir(
        "ckpt_e5m2",
        r#"{"quantization_config":{"quant_method":"fp8","fmt":"e5m2",
                "weight_block_size":[128,128]}}"#,
    );
    mla_ckpt_enc(&d);
}
