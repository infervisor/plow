//! Negative fusion tests: patterns that structurally resemble fusion targets but
//! should NOT produce semantically incorrect results. In equality saturation both
//! fused and unfused forms coexist — these tests verify that fusion preserves
//! graph semantics and that extraction picks the correct form.

use nn_graph::models::build_from_config_json;

/// A norm whose input is NOT an add should NOT produce FusedResidualNorm.
/// The FusedResidualNorm rule only matches `RmsNorm(Ew("add", a, b), w, eps)`,
/// so a standalone norm on a non-add input stays unfused.
#[test]
fn standalone_norm_does_not_produce_residual_norm() {
    // Single-layer Llama with 1 layer: the very first input_layernorm reads the
    // embedding output (no preceding add). That norm must NOT be FusedResidualNorm.
    let json = r#"{
        "model_type": "llama",
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;
    let g = build_from_config_json(json).expect("build");
    let (fused, _) = rewrite::rewrite_graph(&g).expect("rewrite");

    // With only 1 layer: the first block's input_layernorm reads the embedding
    // (which may be fused from add in post_attention of a prior block — but there
    // IS no prior block). Still, the FusedResidualNorm should fire on the
    // post_attention_layernorm (which DOES read an add). Verify: at most as many
    // FusedResidualNorm as there are residual adds feeding norms (= 1 per block × 2
    // sub-blocks = 2, but the first block's input_layernorm reads embedding, not add).
    let residual_norms = fused
        .nodes
        .iter()
        .filter(|n| n.op == "FusedResidualNorm")
        .count();
    // 1 layer: attn residual→post_attention_layernorm is 1 FusedResidualNorm,
    // MLP residual→final norm is 1 FusedResidualNorm (if >1 layer it'd be
    // input_layernorm of the next block). With 1 layer, the final `norm` is fed
    // by add(residual, mlp_output).
    assert!(
        residual_norms <= 2,
        "too many FusedResidualNorm: got {residual_norms}, expected <= 2 for 1-layer model"
    );
}

/// RmsNorm output consumed by multiple Linears: both should fuse independently.
/// This is the q/k/v fan-out pattern — each projection gets its own
/// FusedNormLinear and the norm weight is shared (not duplicated).
#[test]
fn norm_fanout_fuses_all_consumers() {
    let json = r#"{
        "model_type": "llama",
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;
    let g = build_from_config_json(json).expect("build");
    let (fused, _) = rewrite::rewrite_graph(&g).expect("rewrite");

    // input_layernorm (not from an add) feeds q/k/v → 3 FusedNormLinear.
    // post_attention_layernorm and final norm consume an add → extracted as
    // FusedResidualNorm, so gate/up/lm_head appear as plain Linear (the norm is
    // fused into the residual-add, not into the linear). All three q/k/v fanout
    // from the same norm demonstrates the fan-out property.
    let fnl_count = fused
        .nodes
        .iter()
        .filter(|n| n.op == "FusedNormLinear")
        .count();
    assert!(
        fnl_count >= 3,
        "expected >= 3 FusedNormLinear (q/k/v from input_layernorm), got {fnl_count}"
    );

    // The norm weight appears exactly once per norm (not duplicated per consumer).
    let norm_weights: Vec<_> = fused
        .nodes
        .iter()
        .filter(|n| n.op == "Weight")
        .filter_map(|n| match n.args.first() {
            Some(rewrite::Arg::Str(s)) if s.contains("layernorm") => Some(s.clone()),
            _ => None,
        })
        .collect();
    // With hash-consing, each weight name should appear exactly once in the DAG.
    let mut seen = std::collections::BTreeSet::new();
    for w in &norm_weights {
        assert!(
            seen.insert(w.clone()),
            "weight {w} duplicated in fused graph"
        );
    }
}

/// The SwiGLU rule fires on act(gate)*up but should not break the graph when
/// the activation function varies (SiLU vs GELU). Both produce a SwiGLU node
/// carrying the act kind as its first argument, so a GeGLU (gelu_tanh) never
/// aliases with a SwiGLU (silu) over the same gate/up.
#[test]
fn swiglu_fires_on_gelu_gated_mlp() {
    // Gemma uses GeluTanh in its MLP (GeGLU), which should still fuse to SwiGLU.
    let json = r#"{
        "model_type": "gemma3",
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "sliding_window_pattern": 2
    }"#;
    let g = build_from_config_json(json).expect("build");
    let (fused, _) = rewrite::rewrite_graph(&g).expect("rewrite");

    assert!(
        fused.contains("SwiGLU"),
        "GeGLU pattern should fuse to SwiGLU"
    );
    // The activation kind must survive fusion: Gemma's MLP is GeGLU (gelu_tanh).
    let swiglu = fused
        .nodes
        .iter()
        .find(|n| n.op == "SwiGLU")
        .expect("SwiGLU node");
    assert_eq!(
        swiglu.args.first(),
        Some(&rewrite::Arg::Str("gelu_tanh".into())),
        "SwiGLU must carry the activation kind as its first argument"
    );
}

/// Weight-manifest completeness must hold even with all the new fusions active.
/// Tests the most aggressive fusion (Llama 2-layer with every rule firing).
#[test]
fn full_fusion_preserves_all_weights() {
    let json = r#"{
        "model_type": "llama",
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;
    let g = build_from_config_json(json).expect("build");
    let (fused, _) = rewrite::rewrite_graph(&g).expect("rewrite");

    let fused_w: std::collections::BTreeSet<String> = fused
        .nodes
        .iter()
        .filter(|n| n.op == "Weight")
        .filter_map(|n| match n.args.first() {
            Some(rewrite::Arg::Str(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let graph_w: std::collections::BTreeSet<String> = g
        .tensors
        .iter()
        .filter(|t| matches!(t.origin, nn_graph::Origin::Weight))
        .filter_map(|t| t.name.clone())
        .collect();

    assert_eq!(fused_w, graph_w, "fusion altered the weight manifest");
}

/// A K3 MLA layer without an output gate must not acquire one during rewriting.
#[test]
fn kimi_k3_ungated_mla_does_not_fuse_an_output_gate() {
    let json = r#"{
        "model_type": "kimi_k3",
        "text_config": {
          "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
          "num_hidden_layers": 2, "num_attention_heads": 2,
          "q_lora_rank": 16, "kv_lora_rank": 16,
          "qk_rope_head_dim": 8, "qk_nope_head_dim": 16, "v_head_dim": 16,
          "num_experts": 4, "num_experts_per_token": 2, "num_shared_experts": 1,
          "moe_intermediate_size": 32, "routed_expert_hidden_size": 32,
          "first_k_dense_replace": 2, "attn_res_block_size": 2,
          "linear_attn_config": {
            "num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4,
            "use_full_rank_gate": true,
            "full_attn_layers": [1], "kda_layers": [2]
          }
        }
    }"#;
    let g = build_from_config_json(json).expect("build K3");
    let (fused, _) = rewrite::rewrite_graph(&g).expect("rewrite");

    assert!(fused.contains("FusedKdaGatedNorm"));
    assert!(!fused.contains("FusedMlaOutGate"));
}
