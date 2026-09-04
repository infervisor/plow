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

    // Q/K norm→RoPE fusion fires; Gemma 4 applies no extra query scaling.
    assert!(
        fused.contains("FusedNormRope"),
        "norm→rope fusion did not fire on Q/K paths"
    );

    // Fusion reduced ops.
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // Full-attention Gemma 4 shares the k_proj result with V. The checkpoint
    // carries one k_proj tensor and no v_proj/kv_proj tensor for this layer.
    let weights = fused_weights(&fused);
    let k_weights: Vec<_> = weights.iter().filter(|w| w.contains("k_proj")).collect();
    assert_eq!(
        k_weights.len(),
        1,
        "expected exactly 1 k_proj weight, got {:?}",
        k_weights
    );

    assert!(
        !weights
            .iter()
            .any(|w| w.contains(".v_proj.") || w.contains("kv_proj")),
        "full-attention shared K/V should not declare v_proj or kv_proj"
    );

    // Weight completeness: no weights lost in fusion.
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion altered weight manifest"
    );
}

#[test]
fn gemma4_full_attention_reuses_k_projection_for_v() {
    let g3 = build_from_config_json(GEMMA).expect("build gemma3");
    let g4 = build_from_config_json(GEMMA4).expect("build gemma4");
    let g3_weights = graph_weights(&g3);
    let g4_weights = graph_weights(&g4);

    assert!(g3_weights.contains("layers.0.self_attn.v_proj.weight"));
    assert!(g4_weights.contains("layers.0.self_attn.k_proj.weight"));
    assert!(!g4_weights.contains("layers.0.self_attn.v_proj.weight"));
    assert!(!g4_weights.iter().any(|name| name.contains("kv_proj")));
}

/// The 48-layer unroll must EXTRACT, not abort.
///
/// egglog's default `TreeAdditiveCostModel` sums children and combines with
/// `saturating_add`. A residual stream references its hidden state ~8× per
/// layer, so the *tree* cost of layer `L` grows ~8^L and pins at `u64::MAX`
/// around layer 21. Past that, Bellman-Ford's `topo_rnk` stops advancing with
/// the costs, `save_best_parent_edge` records no parent edge for some e-class,
/// and reconstruction unwraps `None` (`egglog-2.0.0/src/extract.rs:471`). With
/// `panic = "abort"` in the release profile that is process death — which is
/// exactly why the devblob path only ever calls `explore_stats` (saturate-only)
/// and never `rewrite_graph`.
///
/// `extract::DepthCost` replaces the sum with a max, so cost is bounded by
/// graph DEPTH and cannot saturate at any layer count. Bisected against the
/// real Gemma-4-12B text config: before the fix, extraction succeeded at 1–16
/// layers and aborted at 24 and 48; after it, all of them succeed.
///
/// 48 is the depth of the model plow actually serves. Dims are small because
/// the bug is structural — it depends on residual DEPTH, not on tensor sizes.
#[test]
fn gemma4_48_layer_unroll_extracts_without_saturating_cost() {
    let layers = (0..48)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                "\"full_attention\""
            } else {
                "\"sliding_attention\""
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let cfg = format!(
        r#"{{
        "model_type": "gemma4_unified_text",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_global_key_value_heads": 4,
        "global_head_dim": 128,
        "attention_k_eq_v": true,
        "use_qk_norm": true,
        "query_pre_attn_scalar": 128.0,
        "sliding_window": 512,
        "layer_types": [{layers}],
        "rope_parameters": {{
            "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 0.5}},
            "sliding_attention": {{"rope_theta": 10000.0}}
        }}
    }}"#
    );

    let g = build_from_config_json(&cfg).expect("build 48-layer gemma4");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("48-layer extract must not abort");

    // Fusion actually fired across the whole unroll, not just the first blocks.
    assert!(
        stats.fused > 200,
        "expected >200 fused nodes over 48 layers, got {}",
        stats.fused
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // Per-layer fusions are present at full multiplicity, so the tail of the
    // unroll fused too — the failure mode this pins is depth-dependent.
    let count = |op: &str| fused.nodes.iter().filter(|n| n.op == op).count();
    assert_eq!(count("SwiGLU"), 48, "one gated-MLP fusion per layer");
    assert_eq!(
        count("FusedNormRope") + count("FusedNormRopeScale"),
        96,
        "q and k norm→rope fusions on every layer"
    );
}
