//! End-to-end: parse a `config.json`, build the symbolic operator graph, run
//! shape inference, and assert the output shapes for each model family.
//!
//! Configs are scaled-down (few layers) but structurally faithful so the tests
//! run fast while still exercising every op path.

use nn_graph::graph::Origin;
use nn_graph::models::{build_from_config_json, build_from_config_json_at, ShapeBucket};
use nn_graph::Graph;

/// All node-produced tensors must have an inferred shape.
fn assert_fully_inferred(g: &Graph) {
    for (i, t) in g.tensors.iter().enumerate() {
        if matches!(t.origin, Origin::Node(_)) {
            assert!(
                t.shape.is_some(),
                "tensor {i} (node output) has no inferred shape"
            );
        }
    }
}

fn output_shape_str(g: &Graph) -> String {
    let out = *g.outputs.last().expect("graph has an output");
    let shape = g.tensor(out).shape.clone().expect("output shape inferred");
    shape.display_with(&g.syms)
}

#[test]
fn gemma_decoder() {
    let cfg = r#"{
        "model_type": "gemma3",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 512,
        "sliding_window_pattern": 2,
        "query_pre_attn_scalar": 256.0,
        "use_qk_norm": true,
        "torch_dtype": "bfloat16"
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma");
    assert_fully_inferred(&g);

    // Output: logits [B, S, vocab].
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // One attention op per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        4
    );

    // Alternating local/global with pattern=2: layers 1,3 global (no window),
    // layers 0,2 local (window). Check at least one of each appears.
    let windows: Vec<Option<u32>> = g
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            nn_graph::Op::Attention { sliding_window, .. } => Some(sliding_window),
            _ => None,
        })
        .collect();
    assert!(
        windows.contains(&Some(512)),
        "expected a local (windowed) layer"
    );
    assert!(windows.contains(&None), "expected a global layer");
}

#[test]
fn deepseek_mla_moe() {
    let cfg = r#"{
        "model_type": "deepseek_v3",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "q_lora_rank": 96,
        "kv_lora_rank": 64,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 32,
        "v_head_dim": 32,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "first_k_dense_replace": 1,
        "torch_dtype": "bfloat16"
    }"#;
    let g = build_from_config_json(cfg).expect("build deepseek");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // first_k_dense_replace=1 ⇒ layers 1,2 are MoE ⇒ 2 routers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        2
    );
    // MLA broadcast of the shared rotary key appears once per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Broadcast { .. })),
        3
    );
}

#[test]
fn deepseek_without_q_lora() {
    // q_lora_rank = 0 exercises the direct q_proj path.
    let cfg = r#"{
        "model_type": "deepseek_v2",
        "vocab_size": 500,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "q_lora_rank": 0,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 0,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1
    }"#;
    let g = build_from_config_json(cfg).expect("build deepseek-lite");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 500]");
}

#[test]
fn siglip_vision() {
    let cfg = r#"{
        "model_type": "siglip_vision_model",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_channels": 3,
        "image_size": 32,
        "patch_size": 16,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh"
    }"#;
    let g = build_from_config_json(cfg).expect("build siglip");
    assert_fully_inferred(&g);

    // Pooled image embedding: [B, hidden].
    assert_eq!(output_shape_str(&g), "[B, 64]");
    // 2x2 = 4 patches; one conv patch-embed.
    assert_eq!(g.count_ops(|o| matches!(o, nn_graph::Op::Conv2d { .. })), 1);
}

#[test]
fn qwen_vl_vision() {
    let cfg = r#"{
        "model_type": "qwen2_5_vl",
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 128
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build qwen-vl");
    assert_fully_inferred(&g);

    // Merged-token output: [Pm, out_hidden].
    assert_eq!(output_shape_str(&g), "[Pm, 128]");
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        2
    );
}

#[test]
fn gemma4_unified_per_layer_attention() {
    // Gemma 4 specifics: layer_types drive local/global; global layers use a
    // larger head_dim and fewer KV heads; K==V projection; partial RoPE.
    let cfg = r#"{
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 64,
            "global_head_dim": 128,
            "attention_k_eq_v": true,
            "rms_norm_eps": 1e-6,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // Per-layer head dims: global layer uses 128, sliding layers use 64.
    let head_dims: Vec<u32> = g
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            nn_graph::Op::Attention { head_dim, .. } => Some(head_dim),
            _ => None,
        })
        .collect();
    assert_eq!(head_dims, vec![64, 64, 128, 64]);
}

#[test]
fn qwen_image_vae_encoder() {
    let cfg = r#"{
        "_class_name": "AutoencoderKLQwenImage",
        "base_dim": 32,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 1,
        "z_dim": 16,
        "temperal_downsample": [false, true, true]
    }"#;
    // Compile at a 256x256 bucket: 256 → 8x downsample → 32 latent.
    let g =
        build_from_config_json_at(cfg, &ShapeBucket::square(256)).expect("build qwen-image vae");
    assert_fully_inferred(&g);

    // output 2*z_dim = 32 channels; T stays 1.
    assert_eq!(output_shape_str(&g), "[B, 32, 1, 32, 32]");
    assert!(g.count_ops(|o| matches!(o, nn_graph::Op::Conv3d { .. })) >= 4);

    // A different resolution bucket compiles a different latent grid.
    let g512 = build_from_config_json_at(cfg, &ShapeBucket::square(512)).expect("build vae 512");
    assert_eq!(output_shape_str(&g512), "[B, 32, 1, 64, 64]");
}

#[test]
fn qwen_image_dit_mmdit() {
    let cfg = r#"{
        "_class_name": "QwenImageTransformer2DModel",
        "num_layers": 2,
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "in_channels": 64,
        "out_channels": 16,
        "patch_size": 2,
        "joint_attention_dim": 128,
        "pooled_projection_dim": 64,
        "axes_dims_rope": [8, 12, 12],
        "guidance_embeds": false
    }"#;
    // Compile at 512x512: latent 512/8 = 64, patches 64/2 = 32/side ⇒ N = 1024.
    let g =
        build_from_config_json_at(cfg, &ShapeBucket::square(512)).expect("build qwen-image dit");
    assert_fully_inferred(&g);

    // proj_out → patch² * out_channels = 4 * 16 = 64.
    assert_eq!(output_shape_str(&g), "[B, 1024, 64]");
    // Joint attention: one attention op per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        2
    );

    // Different bucket ⇒ different token count.
    let g256 = build_from_config_json_at(cfg, &ShapeBucket::square(256)).expect("dit 256");
    assert_eq!(output_shape_str(&g256), "[B, 256, 64]");
}

#[test]
fn gemma4_multimodal_dense() {
    // Full multimodal Gemma 4: vision encoder + projector + text decoder.
    let cfg = r#"{
        "model_type": "gemma4_unified",
        "vision_config": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 56,
            "patch_size": 14,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh"
        },
        "mm_input_projection_config": {
            "input_size": 64,
            "output_size": 128
        },
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": 500,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attention_k_eq_v": true,
            "use_qk_norm": true,
            "query_pre_attn_scalar": 32.0,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"]
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4 multimodal");
    assert_fully_inferred(&g);

    // Vision: 56/14 = 4 patches per side → 16 patches total.
    // Combined output: logits [B, 16+S, 500] (image tokens + text tokens).
    let out_str = output_shape_str(&g);
    assert!(
        out_str.contains("500"),
        "expected vocab dim 500 in output, got: {out_str}"
    );

    // Vision encoder has 2 attention layers (non-causal).
    // Text decoder has 2 attention layers (causal).
    let attn_count = g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. }));
    assert_eq!(
        attn_count, 4,
        "expected 4 attention ops (2 vision + 2 text)"
    );

    // One conv2d for patch embedding.
    assert_eq!(g.count_ops(|o| matches!(o, nn_graph::Op::Conv2d { .. })), 1);

    // No MoE routers (dense model).
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        0
    );
}

#[test]
fn gemma4_multimodal_moe() {
    // Multimodal with MoE text decoder.
    let cfg = r#"{
        "model_type": "gemma4_unified",
        "vision_config": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 28,
            "patch_size": 14,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh"
        },
        "mm_input_projection_config": {
            "input_size": 64,
            "output_size": 128
        },
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": 500,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attention_k_eq_v": true,
            "use_qk_norm": true,
            "num_local_experts": 4,
            "num_experts_per_tok": 1,
            "layer_types": ["moe_sliding_attention", "full_attention"],
            "sliding_window": 256
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4 multimodal moe");
    assert_fully_inferred(&g);

    // 1 MoE layer ⇒ 1 router.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        1
    );
    // Vision (1 layer) + text (2 layers) = 3 attention ops.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        3
    );
}

#[test]
fn gemma4_moe_sparse() {
    // Gemma 4 MoE (gemma-4-26B-A4B style): some layers use MoE FFN, others dense.
    // Uses `moe_sliding_attention` / `moe_full_attention` in layer_types.
    let cfg = r#"{
        "model_type": "gemma4_unified_text",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_global_key_value_heads": 4,
        "global_head_dim": 128,
        "attention_k_eq_v": true,
        "use_qk_norm": true,
        "query_pre_attn_scalar": 128.0,
        "sliding_window": 512,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "layer_types": [
            "sliding_attention",
            "moe_sliding_attention",
            "full_attention",
            "moe_full_attention"
        ],
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_theta": 10000.0}
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4 moe");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // 2 out of 4 layers are MoE ⇒ 2 routers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        2
    );
    // All 4 layers still have attention.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        4
    );
    // Per-layer head dims: sliding layers use 64, global layers use 128.
    let head_dims: Vec<u32> = g
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            nn_graph::Op::Attention { head_dim, .. } => Some(head_dim),
            _ => None,
        })
        .collect();
    assert_eq!(head_dims, vec![64, 64, 128, 128]);
}

#[test]
fn gemma4_moe_all_layers() {
    // All layers MoE (simpler variant: no layer_types, all layers become MoE).
    let cfg = r#"{
        "model_type": "gemma4_unified_text",
        "vocab_size": 500,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "attention_k_eq_v": true,
        "use_qk_norm": true,
        "num_local_experts": 4,
        "num_experts_per_tok": 1,
        "sliding_window_pattern": 1
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4 all-moe");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 500]");

    // Both layers are MoE ⇒ 2 routers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        2
    );
}

#[test]
fn gemma4_31b_dense_structure() {
    // Scaled-down version of gemma-4-31B-it (dense, multimodal top-level with text_config).
    let cfg = r#"{
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": 1000,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "num_global_key_value_heads": 8,
            "global_head_dim": 128,
            "attention_k_eq_v": true,
            "use_qk_norm": true,
            "query_pre_attn_scalar": 128.0,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4-31b-like");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // No MoE routers (dense model).
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        0
    );
    // 4 attention layers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        4
    );
}

#[test]
fn gemma4_12b_dense_structure() {
    // Scaled-down version of gemma-4-12B-it.
    let cfg = r#"{
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": 800,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_hidden_layers": 3,
            "num_attention_heads": 6,
            "num_key_value_heads": 3,
            "head_dim": 64,
            "num_global_key_value_heads": 6,
            "global_head_dim": 128,
            "attention_k_eq_v": true,
            "use_qk_norm": true,
            "query_pre_attn_scalar": 128.0,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build gemma4-12b-like");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 800]");

    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        0
    );
}

#[test]
fn unknown_architecture_errors() {
    let cfg = r#"{ "model_type": "llama" }"#;
    assert!(build_from_config_json(cfg).is_err());
}

#[test]
fn gemma1_rejected() {
    let cfg = r#"{ "model_type": "gemma", "vocab_size": 1000 }"#;
    let err = build_from_config_json(cfg).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("only Gemma 3/4"),
        "expected Gemma1 rejection, got: {msg}"
    );
}

#[test]
fn gemma2_rejected() {
    let cfg = r#"{ "model_type": "gemma2", "vocab_size": 1000 }"#;
    let err = build_from_config_json(cfg).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("only Gemma 3/4"),
        "expected Gemma2 rejection, got: {msg}"
    );
}

#[test]
fn qwen2_vl_rejected() {
    let cfg = r#"{ "model_type": "qwen2_vl", "vision_config": {} }"#;
    let err = build_from_config_json(cfg).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("only Qwen2.5-VL"),
        "expected Qwen2-VL rejection, got: {msg}"
    );
}

#[test]
fn qwen_vl_merger_uses_gelu() {
    // Verify the merger activation is Gelu (not GeluTanh).
    let cfg = r#"{
        "model_type": "qwen2_5_vl",
        "vision_config": {
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 128
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build qwen2.5-vl");
    assert_fully_inferred(&g);

    // The merger MLP uses one Gelu activation.
    let gelu_count = g.count_ops(|o| matches!(o, nn_graph::Op::Act(nn_graph::ActKind::Gelu)));
    assert!(
        gelu_count >= 1,
        "expected at least one Gelu op in merger, found {gelu_count}"
    );
    // No GeluTanh should appear (the vision blocks use Silu, not GeluTanh).
    let gelu_tanh_count =
        g.count_ops(|o| matches!(o, nn_graph::Op::Act(nn_graph::ActKind::GeluTanh)));
    assert_eq!(gelu_tanh_count, 0, "unexpected GeluTanh op found");
}

#[test]
fn glm_moe_dsa() {
    // Scaled-down GLM-5.2: MLA + MoE + DSA indexer.
    let cfg = r#"{
        "model_type": "glm_moe_dsa",
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 48,
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 96,
        "kv_lora_rank": 64,
        "qk_head_dim": 64,
        "qk_nope_head_dim": 48,
        "qk_rope_head_dim": 16,
        "v_head_dim": 64,
        "rope_interleave": true,
        "rope_parameters": { "rope_theta": 8000000.0, "rope_type": "default" },
        "first_k_dense_replace": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "indexer_types": ["full", "full", "full", "shared"],
        "index_head_dim": 32,
        "index_n_heads": 4,
        "index_topk": 64,
        "index_skip_topk_offset": 3,
        "num_nextn_predict_layers": 1,
        "scoring_func": "sigmoid",
        "torch_dtype": "bfloat16"
    }"#;
    let g = build_from_config_json(cfg).expect("build glm");
    assert_fully_inferred(&g);

    // Primary output: logits [B, S, vocab].
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // MoE layers 1,2,3 ⇒ 3 routers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        3
    );
    // MLA broadcast of the shared rotary key: once per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Broadcast { .. })),
        4
    );
    // Interleaved RoPE: 2 per layer (q + k) = 8 total.
    let rope_count = g.count_ops(|o| {
        matches!(
            o,
            nn_graph::Op::Rope {
                interleave: true,
                ..
            }
        )
    });
    assert_eq!(rope_count, 8);

    // Multi-token prediction: 2 outputs total (main + 1 MTP head).
    assert_eq!(g.outputs.len(), 2);
}

#[test]
fn glm_architecture_detection() {
    // Ensure architecture fallback works.
    let cfg = r#"{
        "architectures": ["GlmMoeDsaForCausalLM"],
        "vocab_size": 500,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "q_lora_rank": 48,
        "kv_lora_rank": 32,
        "qk_head_dim": 40,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 0,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1
    }"#;
    let g = build_from_config_json(cfg).expect("build glm via architectures[]");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 500]");
}

// ── Kimi: K2 builds, K3 is refused, and a partial config no longer fabricates ──
//
// `KimiConfig` had no test at all when it landed, and it carried a blanket
// `#[serde(default)]` over a full K2 geometry. These three lock down the fix.

/// A complete, scaled-down K2 config still builds.
#[test]
fn kimi_k2_mla_moe() {
    let cfg = r#"{
        "model_type": "kimi",
        "vocab_size": 500,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "q_lora_rank": 48,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 32,
        "v_head_dim": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1
    }"#;
    let g = build_from_config_json(cfg).expect("build kimi k2");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 500]");
    // 3 layers, one MLA attention each.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        3
    );
    // first_k_dense_replace = 1 ⇒ layer 0 dense, layers 1-2 MoE.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        2
    );
}

/// A K2 config missing required geometry must ERROR, not silently fall back to
/// the published 61-layer defaults. This is the whole point of dropping the
/// blanket `#[serde(default)]`.
#[test]
fn kimi_partial_config_is_rejected_not_defaulted() {
    // Everything present except `v_head_dim` — previously this yielded 128.
    let cfg = r#"{
        "model_type": "kimi",
        "vocab_size": 500,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "q_lora_rank": 48,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1
    }"#;
    let err = build_from_config_json(cfg).expect_err("must not build from defaults");
    let msg = err.to_string();
    assert!(
        msg.contains("v_head_dim"),
        "error should name the missing field, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Kimi-K3
// ---------------------------------------------------------------------------

/// A complete, small K3 config. Four layers: 0 and 2 are MLA, 1 and 3 are KDA
/// (the config lists them 1-BASED, so `[1,3]` and `[2,4]`).
fn k3_json(extra_text: &str) -> String {
    format!(
        r#"{{
          "model_type": "kimi_k3",
          "text_config": {{
            "model_type": "kimi_linear",
            "vocab_size": 1000, "hidden_size": 256, "intermediate_size": 512,
            "num_hidden_layers": 4, "num_attention_heads": 4,
            "q_lora_rank": 64, "kv_lora_rank": 32,
            "qk_rope_head_dim": 16, "qk_nope_head_dim": 32, "v_head_dim": 32,
            "mla_use_output_gate": true,
            "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1,
            "moe_intermediate_size": 128, "routed_expert_hidden_size": 192,
            "first_k_dense_replace": 1,
            "attn_res_block_size": 2,
            "linear_attn_config": {{
              "num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4,
              "use_full_rank_gate": true,
              "full_attn_layers": [1, 3], "kda_layers": [2, 4]
            }}
            {extra_text}
          }}
        }}"#
    )
}

/// K3 builds, and builds as a HYBRID: the mixer is chosen per layer from the
/// config's list, never from a stride.
#[test]
fn kimi_k3_builds_a_hybrid_of_mla_and_kda_layers() {
    let g = build_from_config_json(&k3_json("")).expect("K3 must build");

    // Two softmax-attention layers and two linear-attention layers.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        2,
        "layers 0 and 2 (1-based 1 and 3) are the MLA layers"
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::LinearAttention { .. })),
        2,
        "layers 1 and 3 (1-based 2 and 4) are the KDA layers"
    );
    // Three depthwise convs per KDA layer — one each on q, k and v.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Conv1dDepthwise { kernel: 4 })),
        6,
        "2 KDA layers x q/k/v"
    );
    // `situ` on every GLU: one dense layer + three MoE layers, each of which
    // has a routed expert and a shared expert.
    assert!(
        g.count_ops(|o| matches!(o, nn_graph::Op::SituGlu { .. })) >= 4,
        "K3 uses situ on every GLU, not SiLU"
    );
}

/// K3's manifest uses the released checkpoint names, shapes, and F32 state scalars.
#[test]
fn kimi_k3_weight_manifest_matches_checkpoint_contract() {
    let g = build_from_config_json(&k3_json("")).expect("K3 must build");
    let manifest = g.weight_manifest();
    let find = |name: &str| {
        manifest
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing K3 weight {name}"))
    };

    for (name, shape) in [
        ("layers.0.self_attn.g_proj.weight", "[128, 256]"),
        ("layers.1.self_attn.q_proj.weight", "[128, 256]"),
        ("layers.1.self_attn.q_conv1d.weight", "[128, 1, 4]"),
        ("layers.1.self_attn.A_log", "[4]"),
        ("layers.1.self_attn.dt_bias", "[128]"),
        ("layers.1.self_attn.o_norm.weight", "[32]"),
        ("layers.1.self_attention_res_norm.weight", "[256]"),
        ("layers.1.self_attention_res_proj.weight", "[1, 256]"),
        ("layers.1.mlp_res_norm.weight", "[256]"),
        ("layers.1.mlp_res_proj.weight", "[1, 256]"),
    ] {
        let w = find(name);
        assert_eq!(w.shape.unwrap().display_with(&g.syms), shape, "{name}");
    }

    assert_eq!(find("layers.1.self_attn.A_log").dtype, nn_graph::DType::F32);
    for name in [
        "layers.1.self_attn.dt_bias",
        "layers.1.self_attn.q_conv1d.weight",
        "layers.1.self_attn.o_norm.weight",
    ] {
        assert_eq!(find(name).dtype, nn_graph::DType::F32, "{name}");
    }
    assert!(manifest.iter().all(|w| !w.name.contains(".linear_attn.")));
    assert!(manifest.iter().all(|w| !w.name.ends_with("_res.weight")));
}

/// The block residual is present and is NOT a plain add.
///
/// This is the one K3 departure that is numerically invisible at a
/// non-snapshot layer (measured 3.0e-3 at the block output against 8.1e-1 at
/// the mix itself), so a graph that silently used `Add` here would look right.
#[test]
fn kimi_k3_carries_the_attn_res_block_residual() {
    let g = build_from_config_json(&k3_json("")).expect("K3 must build");
    assert!(
        g.count_ops(|o| matches!(o, nn_graph::Op::BlockResidual { .. })) > 0,
        "AttnRes must appear; a plain residual add is the bug this models away"
    );
}

/// Group-limited routing reaches the router op. Flat top-k over the same expert
/// count is a DIFFERENT model, so this must not be defaulted away.
#[test]
fn kimi_k3_group_limited_routing_is_carried_to_the_router() {
    let g = build_from_config_json(&k3_json(r#", "n_group": 2, "topk_group": 1"#))
        .expect("K3 with groups must build");
    assert!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::MoeRouter {
                group: Some(nn_graph::MoeGroups {
                    n_group: 2,
                    topk_group: 1
                }),
                ..
            }
        )) > 0,
        "n_group/topk_group must reach the router op"
    );

    // Half a pair is refused rather than resolved by defaulting.
    let err = build_from_config_json(&k3_json(r#", "n_group": 2"#))
        .expect_err("n_group without topk_group must not build");
    assert!(
        err.to_string().contains("topk_group"),
        "the refusal must name the missing half, got: {err}"
    );

    // K3's own spelling. The released checkpoint says `num_expert_group`, not
    // `n_group` — reading only the DeepSeek name made the real config look
    // like half a pair and refuse to build.
    let g = build_from_config_json(&k3_json(r#", "num_expert_group": 1, "topk_group": 1"#))
        .expect("K3 with the num_expert_group spelling must build");
    assert!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::MoeRouter {
                group: Some(nn_graph::MoeGroups {
                    n_group: 1,
                    topk_group: 1
                }),
                ..
            }
        )) > 0,
        "num_expert_group must alias n_group and reach the router op"
    );
}

/// The layer partition must cover every layer exactly once. A gap or an overlap
/// would otherwise bind a whole layer's tensors from the wrong mixer.
#[test]
fn kimi_k3_layer_partition_must_be_disjoint_and_complete() {
    // Layer 4 (1-based) listed in neither list.
    let gap = k3_json("").replace(r#""kda_layers": [2, 4]"#, r#""kda_layers": [2]"#);
    let err = build_from_config_json(&gap).expect_err("an uncovered layer must not build");
    assert!(
        err.to_string().contains("neither"),
        "must say the layer is in neither list, got: {err}"
    );

    // Layer 2 (1-based) in both.
    let dup = k3_json("").replace(
        r#""full_attn_layers": [1, 3]"#,
        r#""full_attn_layers": [1, 2, 3]"#,
    );
    let err = build_from_config_json(&dup).expect_err("a doubly-assigned layer must not build");
    assert!(
        err.to_string().contains("twice"),
        "must say the layer was assigned twice, got: {err}"
    );
}

/// A multimodal K3 checkpoint is refused: this builds the text tower only, and
/// a text-only graph would be silently wrong on every image prompt.
#[test]
fn kimi_k3_multimodal_is_refused() {
    let mm = k3_json("").replace(
        r#""model_type": "kimi_k3","#,
        r#""model_type": "kimi_k3", "vision_config": {"hidden_size": 64},"#,
    );
    let err = build_from_config_json(&mm).expect_err("multimodal K3 must not build");
    assert!(
        err.to_string().contains("vision_config"),
        "must name vision_config, got: {err}"
    );
}

/// K3 must never build as K2 — the failure this guards is silent SUCCESS.
///
/// Its `architectures` entry starts with "Kimi", which the arch fallback used
/// to map onto the K2 builder: a 93-layer hybrid model standing in for a
/// 61-layer MLA+MoE one, with three MoE fields silently defaulted because K3
/// spells them differently.
#[test]
fn kimi_k3_never_builds_as_k2() {
    // Reaches the K3 config by every dispatch route, including the
    // `architectures` fallback with no model_type at all.
    for cfg in [
        r#"{"model_type": "kimi_k3", "hidden_size": 7168}"#,
        r#"{"model_type": "kimi_linear", "hidden_size": 7168}"#,
        r#"{"architectures": ["KimiLinearForCausalLM"], "hidden_size": 7168}"#,
    ] {
        let err =
            build_from_config_json(cfg).expect_err("a K3 config with no geometry must not build");
        let msg = err.to_string();
        // A serde "missing field" error — NOT a K2 graph, and not a default.
        assert!(
            msg.contains("missing field"),
            "K3 must fail on the missing geometry rather than defaulting, got: {msg}"
        );
    }

    // And the complete config produces a K3 graph, not K2's: K2 has no linear
    // attention at all.
    let g = build_from_config_json(&k3_json("")).unwrap();
    assert!(
        g.count_ops(|o| matches!(o, nn_graph::Op::LinearAttention { .. })) > 0,
        "a K3 config must not produce an all-softmax (K2-shaped) graph"
    );
}

// ---------------------------------------------------------------------------
// DeepSeek-V4
// ---------------------------------------------------------------------------

/// A structurally faithful, scaled-down V4: hyper-connections on every
/// sub-layer, one KV head, per-layer compress ratios (0 / 4 / 128), a sparse
/// indexer on the ratio-4 layers, and hash-routed leading layers.
///
/// `drop_field`, when non-empty, removes one required key so the config must
/// error instead of defaulting.
fn v4_json(drop_field: &str) -> String {
    let base = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 64,
        "o_groups": 2,
        "o_lora_rank": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": 8,
        "compress_ratios": [0, 0, 4, 128, 4, 0],
        "compress_rope_theta": 160000.0,
        "index_n_heads": 4,
        "index_head_dim": 16,
        "index_topk": 32,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "num_hash_layers": 2,
        "swiglu_limit": 10.0,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "torch_dtype": "bfloat16"
    }"#;
    if drop_field.is_empty() {
        return base.to_string();
    }
    base.lines()
        .filter(|l| !l.trim_start().starts_with(&format!("\"{drop_field}\"")))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn deepseek_v4_hyper_connections_and_compressed_kv() {
    let g = build_from_config_json(&v4_json("")).expect("build deepseek-v4");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // Two hyper-connection sub-layers per layer, plus the final head reduce;
    // every reduce pairs with an expand except that head one.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::HcReduce { .. })),
        5 * 2 + 1
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::HcExpand { .. })),
        5 * 2
    );
    // There is no residual add anywhere in this model: the only Add is the
    // shared expert joining the routed one, once per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Elementwise(nn_graph::op::EwKind::Add))),
        5
    );

    // compress_ratios = [0,0,4,128,4] over 5 layers ⇒ 3 compressed layers, and
    // each ratio-4 layer adds a second compressor inside its indexer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::KvCompress { .. })),
        3 + 2
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::KvCompress { overlap: true, .. })),
        2 + 2
    );

    // Every layer attends with a sink, projects out block-diagonally, and runs
    // clamped SwiGLU in both its routed and its shared expert.
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::Attention {
                attn_sink: true,
                ..
            }
        )),
        5
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::GroupedLinear { .. })),
        5
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::ClampedSwiGlu { limit } if *limit == 10.0)),
        5 * 2
    );

    // num_hash_layers = 2 ⇒ layers 0,1 route from the token-id table and the
    // other three from the biased scores. Getting this split wrong routes every
    // token to a different expert set and still builds.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { hash: true, .. })),
        2
    );
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::MoeRouter {
                hash: false,
                select_bias: true,
                ..
            }
        )),
        3
    );

    // The output de-rotation is a distinct rotation, once per layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Rope { inverse: true, .. })),
        5
    );
}

/// The manifest is the loader's contract, and V4 spells its tensors its own
/// way — `layers.N.attn.wq_a`, not V2/V3's `model.layers.N.self_attn.*`.
#[test]
fn deepseek_v4_weight_manifest_matches_checkpoint_contract() {
    let g = build_from_config_json(&v4_json("")).expect("build deepseek-v4");
    let manifest = g.weight_manifest();
    let names: std::collections::HashSet<&str> = manifest.iter().map(|w| w.name).collect();

    for want in [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_fn",
        "hc_head_scale",
        "hc_head_base",
        "layers.0.hc_attn_fn",
        "layers.0.hc_ffn_base",
        "layers.0.attn_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.q_norm.weight",
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.kv_norm.weight",
        "layers.0.attn.attn_sink",
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wo_b.weight",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.gate.tid2eid",
        "layers.2.ffn.gate.bias",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w2.weight",
        "layers.0.ffn.experts.0.w3.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
        "layers.2.attn.compressor.ape",
        "layers.2.attn.compressor.norm.weight",
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.weights_proj.weight",
        "layers.2.attn.indexer.compressor.ape",
    ] {
        assert!(names.contains(want), "manifest is missing {want}");
    }

    // A hash layer has no selection bias; a scored layer has no lookup table.
    assert!(!names.contains("layers.0.ffn.gate.bias"));
    assert!(!names.contains("layers.2.ffn.gate.tid2eid"));
    // Sliding-window-only layers have neither compressor nor indexer, and a
    // ratio-128 layer compresses without indexing.
    assert!(!names.contains("layers.0.attn.compressor.wkv.weight"));
    assert!(names.contains("layers.3.attn.compressor.wkv.weight"));
    assert!(!names.contains("layers.3.attn.indexer.wq_b.weight"));

    let shape_of = |n: &str| {
        manifest
            .iter()
            .find(|w| w.name == n)
            .unwrap_or_else(|| panic!("missing V4 weight {n}"))
            .shape
            .map(|s| format!("{s}"))
            .expect("weight shape")
    };
    // wo_a is ONE stacked [groups*rank, group_width] tensor, not `groups`
    // separate projections, and the overlapped compressor projects twice its
    // output width while the plain one does not.
    assert_eq!(shape_of("layers.0.attn.wo_a.weight"), "[32, 64]");
    assert_eq!(shape_of("layers.0.attn.wq_b.weight"), "[128, 64]");
    assert_eq!(shape_of("layers.0.attn.wkv.weight"), "[32, 128]");
    assert_eq!(shape_of("layers.2.attn.compressor.wkv.weight"), "[64, 128]");
    assert_eq!(shape_of("layers.3.attn.compressor.wkv.weight"), "[32, 128]");
    assert_eq!(shape_of("layers.0.hc_attn_fn"), "[24, 512]");
    assert_eq!(shape_of("hc_head_fn"), "[4, 512]");
    assert_eq!(shape_of("layers.0.ffn.gate.tid2eid"), "[1000, 2]");
}

#[test]
fn deepseek_v4_partial_config_is_rejected_not_defaulted() {
    // Geometry with no honest default: without it the graph would model a
    // different attention or residual shape and still build.
    for field in ["head_dim", "o_groups", "hc_mult", "compress_ratios"] {
        let err = build_from_config_json(&v4_json(field))
            .expect_err("a V4 config missing required geometry must not build");
        let msg = err.to_string();
        assert!(
            msg.contains("missing field"),
            "V4 must fail on the missing {field} rather than defaulting, got: {msg}"
        );
    }
}

/// `DeepseekV4ForCausalLM` must not fall through to the V2/V3 prefix arms: a V3
/// build would model MLA with a `kv_lora_rank` this checkpoint has no tensor for.
#[test]
fn deepseek_v4_architectures_fallback_does_not_reach_v3() {
    let cfg = r#"{"architectures": ["DeepseekV4ForCausalLM"], "hidden_size": 128}"#;
    let err = build_from_config_json(cfg).expect_err("no geometry ⇒ must not build");
    assert!(
        err.to_string().contains("missing field"),
        "V4 must be claimed by its own arm, got: {err}"
    );
}
