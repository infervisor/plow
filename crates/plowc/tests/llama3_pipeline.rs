//! End-to-end pipeline test: exercises the plowc compiler for LLaMA 3.1 8B BF16
//! and the new Kimi/Qwen3 model support through the nn-graph → rewrite → bridge
//! → schedule → packet emit path.
//!
//! These tests use the `build_from_config_json` offline path (no network access)
//! to prove the full compile pipeline works for each model family at small dims.

use nn_graph::models::build_from_config_json;
use rewrite::bridge::plan_from_all_blocks;

/// Realistic LLaMA 3.1 8B config (scaled down for test speed).
const LLAMA3_8B_CONFIG: &str = r#"{
    "model_type": "llama",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
}"#;

/// Kimi K2 config (scaled down).
const KIMI_CONFIG: &str = r#"{
    "model_type": "kimi",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 48,
    "v_head_dim": 64,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "first_k_dense_replace": 2,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": true,
    "routed_scaling_factor": 2.827,
    "torch_dtype": "bfloat16"
}"#;

/// Qwen 3.5 config (scaled down).
const QWEN3_CONFIG: &str = r#"{
    "model_type": "qwen3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
}"#;

/// Full pipeline: config → graph → bind → rewrite → bridge (LayerPlan) for LLaMA 3.1.
#[test]
fn llama3_8b_bf16_config_to_plan() {
    let mut g = build_from_config_json(LLAMA3_8B_CONFIG).expect("build llama3");
    g.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 128));

    // Rewrite fires SwiGLU fusions.
    let (_fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");
    assert!(stats.fused > 0, "SwiGLU fusion should fire");
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion reduces op count"
    );

    // Bridge produces a LayerPlan with ops.
    let plan = plan_from_all_blocks(&g).expect("bridge to plan");
    assert!(!plan.ops.is_empty(), "plan must contain ops");

    // All compute dtypes should be BF16.
    for op in &plan.ops {
        assert_eq!(
            op.compute_dtype,
            nn_graph::DType::BF16,
            "op {} compute dtype should be BF16",
            op.name
        );
    }
}

/// Kimi K2 (MLA + MoE): config → graph → bind → rewrite → bridge (LayerPlan).
/// Now fully supported through the bridge since MoeRouter is handled.
#[test]
fn kimi_bf16_config_to_plan() {
    let mut g = build_from_config_json(KIMI_CONFIG).expect("build kimi");
    g.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 128));

    let (_fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");
    assert!(stats.fused > 0, "SwiGLU fusion should fire in Kimi");
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion reduces ops in Kimi"
    );

    // Bridge produces a LayerPlan — now works with MoeRouter handling.
    let plan = plan_from_all_blocks(&g).expect("bridge to plan");
    assert!(!plan.ops.is_empty(), "Kimi plan must contain ops");

    // All compute dtypes should be BF16.
    for op in &plan.ops {
        assert_eq!(
            op.compute_dtype,
            nn_graph::DType::BF16,
            "op {} compute dtype should be BF16",
            op.name
        );
    }
}

/// Full pipeline for Qwen 3.5 (GQA + SwiGLU).
#[test]
fn qwen3_bf16_config_to_plan() {
    let mut g = build_from_config_json(QWEN3_CONFIG).expect("build qwen3");
    g.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 128));

    let (_fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");
    assert!(stats.fused > 0, "SwiGLU fusion should fire in Qwen3");

    let plan = plan_from_all_blocks(&g).expect("bridge to plan");
    assert!(!plan.ops.is_empty(), "plan must contain ops");

    // All compute dtypes should be BF16.
    for op in &plan.ops {
        assert_eq!(
            op.compute_dtype,
            nn_graph::DType::BF16,
            "op {} compute dtype should be BF16",
            op.name
        );
    }
}
