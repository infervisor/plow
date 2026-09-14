//! The fused sites devgen lowers under `PLOW_EMIT_REWRITE`: every graph output extracted, each
//! fused node keyed by checkpoint weight name.

use nn_graph::models::build_from_config_json;
use rewrite::FusedSites;

const GEMMA4: &str = r#"{
    "model_type": "gemma4_unified_text",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_global_key_value_heads": 4,
    "global_head_dim": 128,
    "attention_k_eq_v": true,
    "use_qk_norm": true,
    "query_pre_attn_scalar": 128.0,
    "sliding_window": 512,
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0}
    }
}"#;

const LLAMA: &str = r#"{
    "model_type": "llama",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": false
}"#;

const GLM5: &str = r#"{
    "model_type": "glm_moe_dsa",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "attention_bias": false,
    "hidden_act": "silu",
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_head_dim": 64,
    "qk_nope_head_dim": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "rope_interleave": true,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "first_k_dense_replace": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
    "scoring_func": "sigmoid",
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": true,
    "n_group": 2,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "moe_router_dtype": "float32",
    "indexer_types": ["full", "full", "full", "shared"],
    "index_head_dim": 32,
    "index_n_heads": 4,
    "index_topk": 64,
    "index_topk_freq": 1,
    "indexer_rope_interleave": true,
    "index_skip_topk_offset": 2,
    "num_nextn_predict_layers": 1,
    "index_share_for_mtp_iteration": true,
    "torch_dtype": "bfloat16",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128]
    }
}"#;

fn anchored(sites: &FusedSites, kind: &str, weight_suffix: &str) -> bool {
    sites
        .get(kind)
        .is_some_and(|w| w.iter().any(|name| name.ends_with(weight_suffix)))
}

#[test]
fn gemma4_seams_are_sandwich_sites_keyed_by_the_following_norm() {
    let g = build_from_config_json(GEMMA4).unwrap();
    let (fused, _) = rewrite::rewrite_graph_outputs(&g).unwrap();
    let sites = fused.sites();
    for l in 0..2 {
        assert!(
            anchored(
                &sites,
                "FusedNormResidualNorm",
                &format!("layers.{l}.pre_feedforward_layernorm.weight")
            ),
            "post-attention seam of layer {l}: {sites:?}"
        );
    }
    assert!(anchored(
        &sites,
        "FusedNormResidualScaleNorm",
        "layers.1.input_layernorm.weight"
    ));
    assert!(anchored(
        &sites,
        "FusedNormResidualScaleNorm",
        "layers.0.layer_scalar"
    ));
    assert!(anchored(
        &sites,
        "FusedNormResidualScaleNorm",
        "norm.weight"
    ));
    assert!(!sites.contains_key("FusedResidualNorm"), "{sites:?}");
}

#[test]
fn swiglu_is_keyed_by_its_projection_weights() {
    let sites = rewrite::fused_sites_for_config(LLAMA).unwrap();
    for l in 0..2 {
        for proj in ["gate_proj", "up_proj"] {
            assert!(anchored(
                &sites,
                "SwiGLU",
                &format!("layers.{l}.mlp.{proj}.weight")
            ));
        }
        assert!(anchored(
            &sites,
            "FusedResidualNorm",
            &format!("layers.{l}.post_attention_layernorm.weight")
        ));
    }
    assert!(anchored(
        &sites,
        "FusedResidualNorm",
        "layers.1.input_layernorm.weight"
    ));
    assert!(anchored(&sites, "FusedResidualNorm", "model.norm.weight"));
}

#[test]
fn every_output_is_extracted_and_shared_paths_extract_identically() {
    let g = build_from_config_json(GLM5).unwrap();
    assert_eq!(g.outputs.len(), 2, "base head + MTP head");
    let (last, _) = rewrite::rewrite_graph(&g).unwrap();
    let (all, _) = rewrite::rewrite_graph_outputs(&g).unwrap();
    let (last, all) = (last.sites(), all.sites());

    assert!(!last.values().flatten().any(|w| w == "model.norm.weight"));
    assert!(
        anchored(&all, "FusedResidual3Norm", "model.norm.weight"),
        "{all:?}"
    );
    for (kind, weights) in &last {
        let extra = weights.difference(&all[kind]).collect::<Vec<_>>();
        assert!(
            extra.is_empty(),
            "{kind}: sites lost when extracting every output: {extra:?}"
        );
    }
}
