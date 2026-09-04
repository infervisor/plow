use nn_graph::models::build_from_config_json;
use nn_graph::{MoeScoring, Op};

const DEEPSEEK_V3: &str = r#"{
  "model_type":"deepseek_v3", "vocab_size":129280, "hidden_size":7168,
  "intermediate_size":18432, "num_hidden_layers":61, "num_attention_heads":128,
  "rms_norm_eps":1e-6, "rope_theta":10000.0, "q_lora_rank":1536,
  "kv_lora_rank":512, "qk_rope_head_dim":64, "qk_nope_head_dim":128,
  "v_head_dim":128, "n_routed_experts":256, "n_shared_experts":1,
  "num_experts_per_tok":8, "moe_intermediate_size":2048,
  "first_k_dense_replace":3, "scoring_func":"sigmoid", "topk_method":"noaux_tc",
  "n_group":8, "topk_group":4, "norm_topk_prob":true,
  "routed_scaling_factor":2.5, "num_nextn_predict_layers":1,
  "torch_dtype":"bfloat16",
  "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
    "quant_method":"fp8","weight_block_size":[128,128]}
}"#;

const KIMI_K2: &str = r#"{
  "model_type":"kimi_k2", "vocab_size":163840, "hidden_size":7168,
  "intermediate_size":18432, "num_hidden_layers":61, "num_attention_heads":64,
  "rms_norm_eps":1e-6, "rope_theta":10000.0, "q_lora_rank":1536,
  "kv_lora_rank":512, "qk_rope_head_dim":64, "qk_nope_head_dim":128,
  "v_head_dim":128, "n_routed_experts":384, "n_shared_experts":1,
  "num_experts_per_tok":8, "moe_intermediate_size":2048,
  "first_k_dense_replace":1, "scoring_func":"sigmoid", "topk_method":"noaux_tc",
  "n_group":1, "topk_group":1, "norm_topk_prob":true,
  "routed_scaling_factor":2.827, "torch_dtype":"bfloat16",
  "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
    "quant_method":"fp8","weight_block_size":[128,128]}
}"#;

#[test]
fn official_deepseek_v3_manifest_and_routing_are_exhaustive() {
    let graph = build_from_config_json(DEEPSEEK_V3).unwrap();
    assert_eq!(graph.expert_bindings.len(), 58);
    assert!(graph
        .expert_bindings
        .iter()
        .all(|layer| layer.routed_experts.len() == 256));
    assert_eq!(graph.checkpoint_manifest().len(), 90_427);
    assert!(
        graph.nodes.len() < 3_000,
        "experts exploded the compute DAG"
    );
    assert_eq!(
        graph.count_ops(|op| matches!(op, Op::MoeExperts { .. })),
        58
    );
    assert!(graph
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            Op::MoeRouter {
                scoring,
                norm_topk,
                route_scale,
                correction_bias,
                group,
                ..
            } => Some((scoring, norm_topk, route_scale, correction_bias, group)),
            _ => None,
        })
        .all(|(scoring, norm, scale, bias, group)| {
            scoring == MoeScoring::Sigmoid
                && norm
                && scale == 2.5
                && bias
                && group.is_some_and(|g| g.n_group == 8 && g.topk_group == 4)
        }));
    assert_eq!(graph.fp8_scale_bindings.len(), 45_032);
}

#[test]
fn official_kimi_k2_manifest_and_routing_are_exhaustive() {
    let graph = build_from_config_json(KIMI_K2).unwrap();
    assert_eq!(graph.expert_bindings.len(), 60);
    assert!(graph
        .expert_bindings
        .iter()
        .all(|layer| layer.routed_experts.len() == 384));
    assert_eq!(graph.checkpoint_manifest().len(), 139_644);
    assert!(
        graph.nodes.len() < 3_000,
        "experts exploded the compute DAG"
    );
    assert_eq!(
        graph.count_ops(|op| matches!(op, Op::MoeExperts { .. })),
        60
    );
    assert_eq!(graph.fp8_scale_bindings.len(), 69_608);
    assert_eq!(graph.checkpoint_storage_bytes(), Some(1_029_173_257_696));
}
