//! End-to-end: parse a `config.json`, build the symbolic operator graph, run
//! shape inference, and assert the output shapes for each model family.
//!
//! Configs are scaled-down (few layers) but structurally faithful so the tests
//! run fast while still exercising every op path.

use nn_graph::graph::Origin;
use nn_graph::models::{
    build_from_config_json, build_from_config_json_at, build_text_generation_from_config_json_at,
    ShapeBucket,
};
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

fn qwen35_json() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 500,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "layer_types": [
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_value_head_dim": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
                "mrope_interleaved": true
            },
            "attention_bias": false,
            "attn_output_gate": true,
            "hidden_act": "silu",
            "mamba_ssm_dtype": "float32",
            "output_gate_type": "swish",
            "tie_word_embeddings": false,
            "dtype": "bfloat16"
        }
    }"#
}

#[test]
fn qwen35_hybrid_decoder() {
    let g = build_text_generation_from_config_json_at(qwen35_json(), &ShapeBucket::default())
        .expect("build Qwen3.5/3.8 text decoder");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 500]");
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        1
    );
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::LinearAttention {
                kind: nn_graph::LinearAttnKind::QwenGatedDelta,
                ..
            }
        )),
        3
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Conv1dDepthwise { kernel: 4 })),
        3
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::RmsNormZeroCentered { .. })),
        11
    );

    let rotary_dims: Vec<u32> = g
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            nn_graph::Op::Rope { dim, .. } => Some(dim),
            _ => None,
        })
        .collect();
    assert_eq!(rotary_dims, vec![8, 8]);
}

#[test]
fn qwen25_uses_qkv_bias_without_qk_norm() {
    let cfg = r#"{
        "architectures":["Qwen2ForCausalLM"], "model_type":"qwen2",
        "vocab_size":64, "hidden_size":32, "intermediate_size":64,
        "num_hidden_layers":1, "num_attention_heads":4,
        "num_key_value_heads":2, "rms_norm_eps":1e-6,
        "rope_theta":1000000.0, "use_sliding_window":false,
        "tie_word_embeddings":false, "torch_dtype":"bfloat16"
    }"#;
    let graph = build_text_generation_from_config_json_at(cfg, &ShapeBucket::default())
        .expect("build Qwen2.5 decoder");
    assert_fully_inferred(&graph);
    let weights = graph
        .checkpoint_manifest()
        .into_iter()
        .map(|weight| weight.name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(weights.len(), 15);
    for name in [
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
    ] {
        assert!(weights.contains(name), "missing Qwen2.5 tensor {name}");
    }
    assert!(!weights.iter().any(|name| name.ends_with("q_norm.weight")));
    assert!(!weights.iter().any(|name| name.ends_with("k_norm.weight")));
}

#[test]
fn qwen2_sliding_window_fails_closed() {
    let cfg = r#"{
        "model_type":"qwen2", "use_sliding_window":true,
        "vocab_size":64, "hidden_size":32, "intermediate_size":64,
        "num_hidden_layers":1, "num_attention_heads":4,
        "num_key_value_heads":2
    }"#;
    let error = build_text_generation_from_config_json_at(cfg, &ShapeBucket::default())
        .expect_err("unsupported Qwen2 sliding attention must not become full attention");
    assert!(error.to_string().contains("sliding-window attention"));
}

#[test]
fn qwen35_weight_manifest_matches_target_checkpoint_layout() {
    let g = build_text_generation_from_config_json_at(qwen35_json(), &ShapeBucket::default())
        .expect("build Qwen3.5/3.8 text decoder");
    let manifest = g.weight_manifest();
    assert_eq!(manifest.len(), 56);
    let find = |name: &str| {
        manifest
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing Qwen3.5 weight {name}"))
    };

    for (name, shape) in [
        ("model.language_model.embed_tokens.weight", "[500, 128]"),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "[160, 128]",
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "[160, 1, 4]",
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "[96, 128]",
        ),
        ("model.language_model.layers.0.linear_attn.A_log", "[6]"),
        ("model.language_model.layers.0.linear_attn.dt_bias", "[6]"),
        (
            "model.language_model.layers.0.linear_attn.norm.weight",
            "[16]",
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "[256, 128]",
        ),
        (
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "[32, 128]",
        ),
        ("lm_head.weight", "[500, 128]"),
    ] {
        let w = find(name);
        assert_eq!(w.shape.unwrap().display_with(&g.syms), shape, "{name}");
        assert_eq!(w.dtype, nn_graph::DType::BF16, "{name}");
    }
}

#[test]
fn qwen35_missing_geometry_is_rejected() {
    let cfg = qwen35_json().replace("\"linear_value_head_dim\": 16,", "");
    let err = build_text_generation_from_config_json_at(&cfg, &ShapeBucket::default())
        .expect_err("required geometry must not default");
    assert!(
        err.to_string().contains("linear_value_head_dim"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen35_multimodal_graph_is_not_silently_reduced_to_text() {
    let with_vision = qwen35_json().replace(
        "\"text_config\": {",
        "\"vision_config\": {\"model_type\": \"qwen3_5\"}, \"text_config\": {",
    );
    let err = build_from_config_json(&with_vision).expect_err("vision tower is not implemented");
    assert!(err.to_string().contains("multimodal graph"), "{err}");
}

fn qwen38_27b_config() -> String {
    let layer_types = (0..64)
        .map(|layer| {
            if layer % 4 == 3 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "language_model_only": false,
          "model_type": "qwen3_5",
          "text_config": {{
            "model_type": "qwen3_5_text",
            "vocab_size": 248320, "hidden_size": 5120,
            "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "layer_types": [{layer_types}],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128, "linear_num_key_heads": 16,
            "linear_num_value_heads": 48, "linear_value_head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {{
              "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
              "rope_type": "default", "mrope_interleaved": true
            }},
            "attention_bias": false, "attn_output_gate": true,
            "hidden_act": "silu", "mamba_ssm_dtype": "float32",
            "output_gate_type": "swish", "tie_word_embeddings": false,
            "dtype": "bfloat16"
          }},
          "vision_config": {{"model_type": "qwen3_5"}}
        }}"#
    )
}

#[test]
fn qwen38_27b_exact_text_manifest_contract() {
    let cfg = qwen38_27b_config();
    let g = build_text_generation_from_config_json_at(&cfg, &ShapeBucket::default())
        .expect("build exact Qwen3.8-27B text endpoint");
    assert_fully_inferred(&g);
    assert_eq!(output_shape_str(&g), "[B, S, 248320]");
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        16
    );
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            nn_graph::Op::LinearAttention {
                kind: nn_graph::LinearAttnKind::QwenGatedDelta,
                ..
            }
        )),
        48
    );

    let manifest = g.weight_manifest();
    assert_eq!(manifest.len(), 851);
    assert_eq!(
        manifest
            .iter()
            .filter(|w| w.name.starts_with("model.language_model."))
            .count(),
        850
    );
    assert_eq!(
        manifest
            .iter()
            .filter(|w| w.name == "lm_head.weight")
            .count(),
        1
    );
    assert!(manifest
        .iter()
        .all(|w| !w.name.contains("model.visual") && !w.name.contains(".mtp.")));
}

#[test]
fn qwen38_27b_fp8_outer_metadata_and_scale_manifest() {
    let mut cfg: serde_json::Value = serde_json::from_str(&qwen38_27b_config()).unwrap();
    let mut ignored = vec![serde_json::Value::String("lm_head".into())];
    for layer in 0..64 {
        if layer % 4 != 3 {
            for projection in ["in_proj_a", "in_proj_b"] {
                ignored.push(serde_json::Value::String(format!(
                    "model.language_model.layers.{layer}.linear_attn.{projection}"
                )));
            }
        }
    }
    cfg["quantization_config"] = serde_json::json!({
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "modules_to_not_convert": ignored,
        "weight_block_size": [128, 128]
    });

    let g = build_text_generation_from_config_json_at(&cfg.to_string(), &ShapeBucket::default())
        .expect("build official Qwen3.8-27B-FP8 text endpoint metadata");
    assert_eq!(g.weight_manifest().len(), 851);

    let checkpoint = g.checkpoint_manifest();
    assert_eq!(checkpoint.len(), 1251);
    assert_eq!(g.fp8_scale_bindings.len(), 400);
    assert!(g.fp8_scale_bindings.iter().all(|binding| {
        binding.block_shape == [128, 128]
            && binding.scale == binding.weight.replace(".weight", ".weight_scale_inv")
    }));
    assert_eq!(
        checkpoint
            .iter()
            .filter(|w| w.dtype == nn_graph::DType::F8E4M3)
            .count(),
        400
    );
    assert_eq!(
        checkpoint
            .iter()
            .filter(|w| w.dtype == nn_graph::DType::BF16)
            .count(),
        451
    );
    assert_eq!(
        checkpoint
            .iter()
            .filter(|w| w.name.ends_with(".weight_scale_inv"))
            .count(),
        400
    );
    assert!(checkpoint
        .iter()
        .filter(|w| w.name.ends_with(".weight_scale_inv"))
        .all(|w| w.dtype == nn_graph::DType::F32));

    let find = |name: &str| {
        checkpoint
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing FP8 checkpoint tensor {name}"))
    };
    assert_eq!(find("lm_head.weight").dtype, nn_graph::DType::BF16);
    assert_eq!(
        find("model.language_model.layers.0.linear_attn.in_proj_a.weight").dtype,
        nn_graph::DType::BF16
    );
    assert_eq!(
        find("model.language_model.layers.0.linear_attn.in_proj_qkv.weight").dtype,
        nn_graph::DType::F8E4M3
    );
    assert_eq!(
        find("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv")
            .shape
            .unwrap()
            .display_with(&g.syms),
        "[80, 40]"
    );
    assert_eq!(
        find("model.language_model.layers.3.self_attn.q_proj.weight_scale_inv")
            .shape
            .unwrap()
            .display_with(&g.syms),
        "[96, 40]"
    );
}

#[test]
fn qwen35_semantic_near_variants_are_rejected() {
    for (from, to, expected) in [
        (
            "\"hidden_act\": \"silu\"",
            "\"hidden_act\": \"gelu\"",
            "hidden_act",
        ),
        (
            "\"mamba_ssm_dtype\": \"float32\"",
            "\"mamba_ssm_dtype\": \"bfloat16\"",
            "mamba_ssm_dtype",
        ),
        (
            "\"attn_output_gate\": true",
            "\"attn_output_gate\": false",
            "attn_output_gate",
        ),
        (
            "\"linear_num_value_heads\": 6",
            "\"linear_num_value_heads\": 5",
            "divisible",
        ),
    ] {
        let cfg = qwen35_json().replace(from, to);
        let err = build_text_generation_from_config_json_at(&cfg, &ShapeBucket::default())
            .expect_err("unsupported semantic variant must fail closed");
        assert!(
            err.to_string().contains(expected),
            "unexpected error: {err}"
        );
    }
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
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc",
        "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        "routed_scaling_factor": 2.827
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
fn gemma4_text_manifest_matches_official_projection_layout() {
    let cfg = r#"{
        "model_type": "gemma4",
        "dtype": "bfloat16",
        "text_config": {
            "model_type": "gemma4_text",
            "vocab_size": 1000,
            "hidden_size": 5376,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 64,
            "global_head_dim": 128,
            "attention_k_eq_v": true,
            "tie_word_embeddings": true,
            "final_logit_softcapping": 30.0,
            "use_qk_norm": true,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }
    }"#;
    let g = build_text_generation_from_config_json_at(cfg, &ShapeBucket::default())
        .expect("build Gemma 4 text tower");
    let weights = g
        .checkpoint_manifest()
        .into_iter()
        .map(|weight| weight.name)
        .collect::<std::collections::BTreeSet<_>>();

    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.layer_scalar",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.layers.1.layer_scalar",
        "model.language_model.norm.weight",
    ] {
        assert!(weights.contains(name), "missing checkpoint tensor {name}");
    }
    assert!(!weights.contains("model.language_model.layers.1.self_attn.v_proj.weight"));
    assert!(!weights.iter().any(|name| name.contains("kv_proj")));
    assert!(!weights.contains("lm_head.weight"));

    let weightless_norms = g
        .nodes
        .iter()
        .filter(|node| matches!(node.op, nn_graph::Op::RmsNorm { .. }) && node.inputs.len() == 1)
        .count();
    assert_eq!(weightless_norms, 2, "Gemma 4 normalizes V without a weight");

    let rope_dims = g
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            nn_graph::Op::Rope {
                dim, frequency_dim, ..
            } => Some((dim, frequency_dim)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(rope_dims, vec![(64, 64), (64, 64), (32, 128), (32, 128)]);

    let scales = g
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            nn_graph::Op::Scale(scale) => Some(scale),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(scales, vec![73.5, 1.0 / 30.0, 30.0]);
    assert_eq!(
        g.count_ops(|op| matches!(op, nn_graph::Op::Act(nn_graph::ActKind::Tanh))),
        1
    );
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
    // Full multimodal Gemma 4 is refused until its exact vision checkpoint
    // contract is implemented. The endpoint-specific text frontend remains valid.
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
    let error = build_from_config_json(cfg).unwrap_err().to_string();
    assert!(error.contains("vision graph is not implemented"), "{error}");

    let text = build_text_generation_from_config_json_at(cfg, &ShapeBucket::default())
        .expect("build Gemma 4 text-generation graph");
    assert_fully_inferred(&text);
    assert_eq!(
        text.count_ops(|o| matches!(o, nn_graph::Op::Attention { .. })),
        2
    );
    assert_eq!(
        text.count_ops(|o| matches!(o, nn_graph::Op::Conv2d { .. })),
        0
    );
}

#[test]
fn gemma4_multimodal_moe_fails_closed() {
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
    let error = build_from_config_json(cfg).unwrap_err().to_string();
    assert!(error.contains("Gemma 4 MoE"), "{error}");
    assert!(error.contains("refusing"), "{error}");
}

#[test]
fn gemma4_moe_sparse_fails_closed() {
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
    let error = build_from_config_json(cfg).unwrap_err().to_string();
    assert!(error.contains("Gemma 4 MoE"), "{error}");
}

#[test]
fn gemma4_moe_all_layers_fails_closed() {
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
    let error = build_from_config_json(cfg).unwrap_err().to_string();
    assert!(error.contains("Gemma 4 MoE"), "{error}");
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
        "attention_bias": false,
        "hidden_act": "silu",
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
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_dtype": "float32",
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "indexer_types": ["full", "full", "full", "shared"],
        "index_head_dim": 32,
        "index_n_heads": 4,
        "index_topk": 64,
        "index_topk_freq": 4,
        "indexer_rope_interleave": true,
        "index_skip_topk_offset": 3,
        "num_nextn_predict_layers": 1,
        "index_share_for_mtp_iteration": true,
        "scoring_func": "sigmoid",
        "torch_dtype": "bfloat16",
        "quantization_config": {
          "activation_scheme":"dynamic", "fmt":"e4m3", "quant_method":"fp8",
          "weight_block_size":[128,128]
        }
    }"#;
    let g = build_from_config_json(cfg).expect("build glm");
    assert_fully_inferred(&g);

    // Primary output: logits [B, S, vocab].
    assert_eq!(output_shape_str(&g), "[B, S, 1000]");

    // Base MoE layers 1,2,3 plus sparse MTP layer 4.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::MoeRouter { .. })),
        4
    );
    // MLA broadcast of the shared rotary key: once per base/MTP layer.
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::Broadcast { .. })),
        5
    );
    // Interleaved main RoPE: 2 per base/MTP layer = 10 total.
    let rope_count = g.count_ops(|o| {
        matches!(
            o,
            nn_graph::Op::Rope {
                interleave: true,
                ..
            }
        )
    });
    assert_eq!(rope_count, 10);

    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::DsaIndexer { .. })),
        4
    );
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::DsaAttention { .. })),
        5
    );
    assert_eq!(g.expert_bindings.len(), 4);
    assert!(g
        .expert_bindings
        .iter()
        .all(|binding| binding.routed_experts.len() == 8));
    assert_eq!(g.fp8_scale_bindings.len(), 144);
    assert_eq!(g.checkpoint_manifest().len(), 335);
    let names: std::collections::BTreeSet<_> = g
        .checkpoint_manifest()
        .into_iter()
        .map(|weight| weight.name)
        .collect();
    assert!(names.contains("model.layers.0.self_attn.indexer.wq_b.weight"));
    assert!(names.contains("model.layers.3.mlp.experts.7.down_proj.weight_scale_inv"));
    assert!(names.contains("model.layers.4.eh_proj.weight"));
    assert!(names.contains("model.layers.4.shared_head.norm.weight"));
    assert!(!names.iter().any(|name| name.starts_with("mtp_heads.")));

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
        "rms_norm_eps": 1e-5,
        "attention_bias": false,
        "hidden_act": "silu",
        "q_lora_rank": 48,
        "kv_lora_rank": 32,
        "qk_head_dim": 40,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "rope_interleave": true,
        "rope_parameters": {"rope_theta":8000000.0,"rope_type":"default"},
        "n_routed_experts": 4,
        "n_shared_experts": 0,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1,
        "mlp_layer_types":["dense"],
        "scoring_func":"sigmoid", "routed_scaling_factor":2.5,
        "norm_topk_prob":true, "n_group":1, "topk_group":1,
        "topk_method":"noaux_tc", "moe_router_dtype":"float32",
        "indexer_types":["full"], "index_head_dim":16,
        "index_n_heads":2, "index_topk":16, "index_topk_freq":4,
        "indexer_rope_interleave":true, "index_skip_topk_offset":1,
        "num_nextn_predict_layers":0, "index_share_for_mtp_iteration":true,
        "dtype":"bfloat16",
        "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
          "quant_method":"fp8","weight_block_size":[128,128]}
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
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc",
        "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        "routed_scaling_factor": 2.827
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

#[test]
fn text_generation_graph_selects_nested_text_tower_without_architecture_cases() {
    let json = k3_json(r##", "dtype": "bfloat16""##).replacen(
        "\"text_config\":",
        "\"vision_config\": {\"model_type\": \"moonvit\"}, \"text_config\":",
        1,
    );
    assert!(
        build_from_config_json(&json).is_err(),
        "the full multimodal graph must still refuse an unimplemented vision tower"
    );
    let g = build_text_generation_from_config_json_at(&json, &ShapeBucket::default())
        .expect("text endpoint must construct its complete nested decoder graph");
    assert_eq!(g.blocks.len(), 4);
    assert_eq!(
        g.count_ops(|o| matches!(o, nn_graph::Op::LinearAttention { .. })),
        2
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
