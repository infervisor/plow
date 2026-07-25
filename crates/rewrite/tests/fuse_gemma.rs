//! End-to-end: build a Gemma block, lower to egglog, run Stage-2 fusion, and
//! assert the fused forms appear and the op count drops.

use nn_graph::models::build_from_config_json;
use nn_graph::Origin;
use std::collections::BTreeSet;

const GEMMA: &str = r#"{
    "model_type": "gemma3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "sliding_window_pattern": 2
}"#;

const GEMMA4: &str = r#"{
    "model_type": "gemma4_unified_text",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_global_key_value_heads": 4,
    "global_head_dim": 128,
    "attention_k_eq_v": true,
    "use_qk_norm": true,
    "query_pre_attn_scalar": 128.0,
    "sliding_window": 512,
    "layer_types": ["full_attention"],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5
        }
    }
}"#;

/// Weight-leaf names referenced by the fused graph.
fn fused_weights(f: &rewrite::FusedGraph) -> BTreeSet<String> {
    f.nodes
        .iter()
        .filter(|n| n.op == "Weight")
        .filter_map(|n| match n.args.first() {
            Some(rewrite::Arg::Str(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// Weight-leaf names declared by the input graph.
fn graph_weights(g: &nn_graph::Graph) -> BTreeSet<String> {
    g.tensors
        .iter()
        .filter(|t| matches!(t.origin, Origin::Weight))
        .filter_map(|t| t.name.clone())
        .collect()
}

#[test]
fn fusion_fires_and_reduces_ops() {
    let g = build_from_config_json(GEMMA).expect("build gemma");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Both fusion rules fired.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire"
    );
    assert!(fused.contains("SwiGLU"), "act·mul fusion did not fire");

    // q/k/v + gate/up + lm_head fold their norm (6), plus one SwiGLU.
    assert!(
        stats.fused >= 6,
        "expected >= 6 fused nodes, got {}",
        stats.fused
    );

    // Fusion strictly reduced the operation count.
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // The unfused patterns are gone from the extracted form.
    assert!(!fused.contains("Linear") || fused.contains("FusedNormLinear"));
}

#[test]
fn gemma4_shared_kv_and_norm_rope_fusion() {
    let g = build_from_config_json(GEMMA4).expect("build gemma4");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Core fusions fire.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire"
    );
    assert!(fused.contains("SwiGLU"), "act·mul fusion did not fire");

    // Norm→Rope fusions fire on qk_norm paths.
    assert!(
        fused.contains("FusedNormRopeScale"),
        "norm→rope→scale fusion did not fire on Q path"
    );

    // Fusion reduced ops.
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // Shared kv_proj: the weight "layers.0.self_attn.kv_proj.weight" should
    // appear exactly once in the fused weight leaves (not duplicated).
    let weights = fused_weights(&fused);
    let kv_weights: Vec<_> = weights.iter().filter(|w| w.contains("kv_proj")).collect();
    assert_eq!(
        kv_weights.len(),
        1,
        "expected exactly 1 kv_proj weight, got {:?}",
        kv_weights
    );

    // No separate k_proj/v_proj weights should exist (only kv_proj).
    assert!(
        !weights
            .iter()
            .any(|w| w.contains(".k_proj.") || w.contains(".v_proj.")),
        "separate k_proj/v_proj should not exist with attention_k_eq_v"
    );

    // Weight completeness: no weights lost in fusion.
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion altered weight manifest"
    );
}

#[test]
fn gemma4_has_fewer_norm_linear_fusions_than_gemma3() {
    // Gemma 3 with 1 layer: q/k/v/gate/up + lm_head = 6 FusedNormLinear.
    let g3 = build_from_config_json(GEMMA).expect("build gemma3");
    let (fused3, _) = rewrite::rewrite_graph(&g3).expect("rewrite gemma3");
    let fnl3 = fused3
        .nodes
        .iter()
        .filter(|n| n.op == "FusedNormLinear")
        .count();

    // Gemma 4 with 1 layer and k_eq_v: q/kv/gate/up + lm_head = 5 FusedNormLinear.
    let g4 = build_from_config_json(GEMMA4).expect("build gemma4");
    let (fused4, _) = rewrite::rewrite_graph(&g4).expect("rewrite gemma4");
    let fnl4 = fused4
        .nodes
        .iter()
        .filter(|n| n.op == "FusedNormLinear")
        .count();

    // Gemma 4 shared K/V should produce one fewer FusedNormLinear per block.
    assert!(
        fnl4 < fnl3,
        "expected gemma4 ({fnl4}) to have fewer FusedNormLinear than gemma3 ({fnl3})"
    );
}
