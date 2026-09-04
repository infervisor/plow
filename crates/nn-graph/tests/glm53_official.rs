#![recursion_limit = "256"]

use std::collections::BTreeSet;

use nn_graph::models::build_from_config_json;
use nn_graph::Op;

fn official_config() -> String {
    let mlp_layer_types = (0..78)
        .map(|layer| if layer < 3 { "dense" } else { "sparse" })
        .collect::<Vec<_>>();
    let indexer_types = (0..78)
        .map(|layer| {
            if layer < 3 || (layer >= 6 && (layer - 6) % 4 == 0) {
                "full"
            } else {
                "shared"
            }
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&serde_json::json!({
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "rms_norm_eps": 1e-5,
        "attention_bias": false,
        "hidden_act": "silu",
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "rope_interleave": true,
        "rope_parameters": {"rope_theta": 8000000.0, "rope_type": "default"},
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "mlp_layer_types": mlp_layer_types,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_dtype": "float32",
        "indexer_types": indexer_types,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "indexer_rope_interleave": true,
        "index_skip_topk_offset": 3,
        "num_nextn_predict_layers": 1,
        "index_share_for_mtp_iteration": true,
        "dtype": "bfloat16",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        }
    }))
    .unwrap()
}

#[test]
fn official_glm53_manifest_and_semantic_graph_are_exact() {
    let graph = build_from_config_json(&official_config()).expect("official GLM-5.3 graph");
    let manifest = graph.checkpoint_manifest();
    let names = manifest
        .iter()
        .map(|weight| weight.name)
        .collect::<BTreeSet<_>>();

    assert_eq!(manifest.len(), 118_629);
    assert_eq!(graph.fp8_scale_bindings.len(), 59_044);
    assert_eq!(graph.expert_bindings.len(), 76);
    assert!(graph.expert_bindings.iter().all(|binding| {
        binding.num_experts == 256 && binding.top_k == 8 && binding.routed_experts.len() == 256
    }));
    assert_eq!(
        graph.count_ops(|op| matches!(op, Op::DsaIndexer { .. })),
        22
    );
    assert_eq!(
        graph.count_ops(|op| matches!(op, Op::DsaAttention { .. })),
        79
    );
    assert_eq!(graph.checkpoint_storage_bytes(), Some(755_617_140_416));

    for name in [
        "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
        "model.layers.3.mlp.experts.255.up_proj.weight_scale_inv",
        "model.layers.78.eh_proj.weight",
        "model.layers.78.self_attn.indexer.weights_proj.weight",
        "model.layers.78.shared_head.norm.weight",
    ] {
        assert!(names.contains(name), "missing official tensor {name}");
    }
    assert!(!names.iter().any(|name| name.starts_with("mtp_heads.")));
    assert!(graph
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            Op::Rope {
                dim,
                frequency_dim,
                interleave,
                ..
            } => Some(dim == 64 && frequency_dim == 192 && interleave),
            _ => None,
        })
        .all(|exact| exact));
}
