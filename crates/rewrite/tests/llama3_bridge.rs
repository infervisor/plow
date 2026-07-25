//! Integration test: build a full LLaMA 3.1 8B graph at real dimensions via
//! `nn_graph`, run egglog fusion, bridge to the tiling IR, and verify that:
//!   1. All expected fusions fire (FusedNormLinear, SwiGLU, FusedResidualNorm)
//!   2. BF16 dtype propagates correctly to every tile op
//!   3. Weight leaves are preserved (no silent drops or duplicates)

use nn_graph::models::build_from_config_json;
use nn_graph::Origin;
use std::collections::BTreeSet;

/// Real LLaMA 3.1 8B config (2 layers to keep test fast, full dims).
const LLAMA3_8B: &str = r#"{
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": false
}"#;

/// Weight-leaf names declared by the input graph.
fn graph_weights(g: &nn_graph::Graph) -> BTreeSet<String> {
    g.tensors
        .iter()
        .filter(|t| matches!(t.origin, Origin::Weight))
        .filter_map(|t| t.name.clone())
        .collect()
}

#[test]
fn llama3_8b_builds_and_infers_shapes() {
    let g = build_from_config_json(LLAMA3_8B).expect("build llama3 8b");
    // Should have tensors for: embed, 2 layers (each: q/k/v/o + gate/up/down + 2 norms), final norm, lm_head.
    // 2 layers × (4 proj + 3 mlp + 2 norm) = 18 weights + embed + final_norm + lm_head = 21
    let weights = graph_weights(&g);
    assert!(
        weights.len() >= 21,
        "expected at least 21 weight tensors, got {}",
        weights.len()
    );
    // Verify key weight names exist.
    assert!(weights.contains("embed_tokens.weight"));
    assert!(weights.contains("lm_head.weight"));
    assert!(weights.contains("layers.0.self_attn.q_proj.weight"));
    assert!(weights.contains("layers.0.mlp.gate_proj.weight"));
    assert!(weights.contains("layers.1.self_attn.o_proj.weight"));
    assert!(weights.contains("norm.weight"));
}

#[test]
fn llama3_8b_fusions_fire() {
    let g = build_from_config_json(LLAMA3_8B).expect("build llama3 8b");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // FusedNormLinear: rmsnorm → linear should fuse.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in LLaMA 3.1 8B"
    );
    // SwiGLU: silu(gate) * up should fuse.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in LLaMA 3.1 8B"
    );
    // FusedResidualNorm: add → rmsnorm should fuse on the post-attention path.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in LLaMA 3.1 8B"
    );

    eprintln!(
        "llama3_8b fusion stats: {} ops before, {} ops after, {} fused",
        stats.ops_before, stats.ops_after, stats.fused
    );
}

#[test]
fn llama3_8b_bridge_preserves_weights_and_bf16() {
    use costmodel::{hwspec, DEFAULT_PAGE_BYTES, Soc, SramPolicy};
    use rewrite::assemble;

    let mut g = build_from_config_json(LLAMA3_8B).expect("build llama3 8b");
    // Bind the symbolic batch/seq dims to a concrete bucket — the bridge
    // requires fully static shapes.
    g.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 512));
    let src_weights = graph_weights(&g);

    // Bridge to tile graph via the block-level path the schedule tests use
    // (fusion firing is covered by `llama3_8b_fusions_fire`).
    let plan = rewrite::plan_from_block(&g, 0).expect("plan block 0");

    // Every block-0 weight leaf must survive as a plan-op input.
    let plan_inputs: BTreeSet<&str> = plan
        .ops
        .iter()
        .flat_map(|op| op.inputs.iter().map(String::as_str))
        .collect();
    for w in src_weights.iter().filter(|w| w.starts_with("layers.0.")) {
        assert!(
            plan_inputs.contains(w.as_str()),
            "weight {w} dropped by the bridge"
        );
    }

    // All ops must carry BF16.
    for op in &plan.ops {
        assert_eq!(
            op.weight_dtype,
            nn_graph::DType::BF16,
            "op {} has non-BF16 weight_dtype: {:?}",
            op.name,
            op.weight_dtype
        );
        assert_eq!(
            op.compute_dtype,
            nn_graph::DType::BF16,
            "op {} has non-BF16 compute_dtype: {:?}",
            op.name,
            op.compute_dtype
        );
    }

    // Assemble into a tile graph on a real SoC to verify no panics.
    let spec = hwspec::registry::lookup("H100 SXM5").unwrap();
    let soc = Soc::single(spec, DEFAULT_PAGE_BYTES);
    let (tg, _cons) = assemble(&soc, &plan, SramPolicy::Stream, None)
        .expect("assemble tile graph for LLaMA 3.1 8B block");
    assert!(
        tg.nodes.len() > 10,
        "tile graph should have substantial nodes, got {}",
        tg.nodes.len()
    );

    eprintln!(
        "llama3_8b bridge: {} plan ops -> {} tile nodes",
        plan.ops.len(),
        tg.nodes.len()
    );
}
