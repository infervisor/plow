//! Binding symbolic shapes to concrete batch/sequence params, and the per-op
//! weight manifest used by later loading stages.

use nn_graph::models::build_from_config_json;
use nn_graph::{Bindings, DType};

const GEMMA: &str = r#"{
    "model_type": "gemma3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "sliding_window_pattern": 2
}"#;

#[test]
fn bind_batch_and_sequence() {
    let mut g = build_from_config_json(GEMMA).expect("build");

    // Symbolic before binding.
    let out = *g.outputs.last().unwrap();
    assert_eq!(
        g.tensor(out).shape.as_ref().unwrap().display_with(&g.syms),
        "[B, S, 1000]"
    );
    assert!(!g.tensor(out).shape.as_ref().unwrap().is_fully_static());

    // Bind concrete params ⇒ fully static shapes.
    g.bind(&Bindings::new().set("B", 2).set("S", 128));
    let shape = g.tensor(out).shape.as_ref().unwrap();
    assert_eq!(shape.display_with(&g.syms), "[2, 128, 1000]");
    assert!(shape.is_fully_static());
}

#[test]
fn partial_binding_stays_symbolic() {
    let mut g = build_from_config_json(GEMMA).expect("build");
    // Bind only batch; sequence stays symbolic.
    g.bind(&Bindings::new().set("B", 4));
    let out = *g.outputs.last().unwrap();
    assert_eq!(
        g.tensor(out).shape.as_ref().unwrap().display_with(&g.syms),
        "[4, S, 1000]"
    );
}

#[test]
fn weight_manifest_lists_per_op_weights() {
    let g = build_from_config_json(GEMMA).expect("build");
    let manifest = g.weight_manifest();

    // Names a loader needs are present, tied to their op.
    let names: Vec<&str> = manifest.iter().map(|w| w.name).collect();
    assert!(names.contains(&"model.embed_tokens.weight"));
    assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight"));
    assert!(names.iter().any(|n| n.ends_with("mlp.gate_proj.weight")));

    // Every linear weight is a rank-2 [out, in] tensor with a known dtype.
    let qproj = manifest
        .iter()
        .find(|w| w.name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(qproj.op, "linear");
    assert_eq!(qproj.shape.unwrap().rank(), 2);
    // Every weight carries a dtype.
    assert!(manifest.iter().all(|w| w.dtype.byte_size().is_some()));
}

#[test]
fn blocks_group_per_layer() {
    let g = build_from_config_json(GEMMA).expect("build"); // num_hidden_layers = 2
    assert_eq!(g.blocks.len(), 2);
    assert_eq!(g.block_label(0), Some("model.layers.0"));

    // Every node in block 0 is tagged with block 0 and names that layer.
    let nodes: Vec<_> = g.block_nodes(0).collect();
    assert!(!nodes.is_empty());
    assert!(nodes.iter().all(|(_, n)| n.block == Some(0)));

    // Pre/post nodes (embedding, final norm, lm_head) are block-less.
    let embed = &g.nodes[0];
    assert_eq!(embed.op.name(), "embedding");
    assert_eq!(embed.block, None);
    assert_eq!(g.nodes.last().unwrap().block, None); // lm_head
}

#[test]
fn weight_dtype_inherited_from_top_level() {
    // Newer multimodal configs put `dtype` at the top level (no `torch_dtype`,
    // none in text_config). It must flow down to the weights.
    let cfg = r#"{
        "model_type": "gemma3",
        "dtype": "float16",
        "text_config": {
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 32
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build");
    let w = g.weight_manifest();
    assert!(!w.is_empty());
    assert!(
        w.iter().all(|s| s.dtype == DType::F16),
        "weights should be f16"
    );
}
