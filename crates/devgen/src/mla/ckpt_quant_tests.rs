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

/// The REAL DeepSeek-V4.1-Flash `quantization_config`, copied field for field off the
/// checkpoint. Its five fields say two different things at once: `weight_block_size [32,32]`
/// with `scale_fmt: "ue8m0"` describes the DENSE projections, and `expert_dtype: "fp4"` the
/// routed experts.
///
/// The PARSE can state that, so it does. What cannot is the collapse to one `MoeEnc`, and the
/// two tests below are the pair: a checkpoint this reads correctly, and an emitter that refuses
/// to pretend it is uniform.
#[test]
fn the_v41_checkpoint_parses_as_mixed_rather_than_being_refused_outright() {
    let d = cfg_dir(
        "ckpt_v41",
        r#"{"model_type":"deepseek_v41","dtype":"bfloat16",
                "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic",
                "weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}}"#,
    );
    let ck = mla_ckpt_enc_full(&d).expect("V4.1 carries a quantization_config");
    assert_eq!(ck.dense, DenseEnc::Fp8Mx32, "dense projections are the [32,32] ue8m0 grid");
    assert_eq!(ck.expert, MoeEnc::Mxfp4, "routed experts are fp4, per expert_dtype");
    assert!(!ck.is_uniform(), "the whole point: one MoeEnc cannot describe this checkpoint");
}

/// And the collapse refuses, naming the capability rather than a constant. Answering with EITHER
/// side would be a lie about the other: the expert encoding would declare fp4 scale grids for
/// projections that are block-fp8 on disk, and the dense one would do the reverse to the experts.
///
/// The message must NOT read as a kernel gap, because it is not one — op 198 and
/// `d_gemm_t<WFP8MX>` exist and pass on gfx942. What is missing is an emitter that can thread two
/// encodings through one run.
#[test]
#[should_panic(expected = "emit_mixed_dense_expert_encoding")]
fn collapsing_a_mixed_checkpoint_to_one_encoding_is_refused() {
    let d = cfg_dir(
        "ckpt_v41_collapse",
        r#"{"model_type":"deepseek_v41","dtype":"bfloat16",
                "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic",
                "weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}}"#,
    );
    mla_ckpt_enc(&d);
}

/// The uniform families must keep collapsing exactly as before — this split is meant to change
/// nothing for them, and `is_uniform` is what decides.
#[test]
fn the_uniform_families_still_collapse_to_one_encoding() {
    let d = cfg_dir(
        "ckpt_uniform_fp8",
        r#"{"model_type":"glm_moe_dsa","quantization_config":{"quant_method":"fp8",
                "fmt":"e4m3","weight_block_size":[128,128]}}"#,
    );
    let ck = mla_ckpt_enc_full(&d).unwrap();
    assert_eq!(ck.dense, DenseEnc::Fp8Blk128);
    assert!(ck.is_uniform());
    assert_eq!(mla_ckpt_enc(&d), Some(MoeEnc::Fp8Blk));
}

/// The V4.1 refusal is keyed on BOTH the block size and the scale format, so it must not
/// swallow a plain `[32,32]` fp8 checkpoint with f32 scales — a shape no V4.1 has, and which
/// still belongs to the generic block-size refusal below it. Pinning this keeps the specific
/// branch from widening into the general one on a later edit.
#[test]
#[should_panic(expected = "fp8_block_size")]
fn a_32_block_without_ue8m0_scales_still_takes_the_generic_refusal() {
    let d = cfg_dir(
        "ckpt_blk32_f32",
        r#"{"quantization_config":{"quant_method":"fp8","fmt":"e4m3",
                "weight_block_size":[32,32]}}"#,
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
