use nn_graph::models::build_from_config_json;
use nn_graph::{DType, Op};

/// `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` text geometry and quant exclusions.
fn official_config() -> String {
    let n = 62;
    let full = |l: usize| (n - 1 - l).is_multiple_of(4);
    let layer_types: Vec<_> = (0..n)
        .map(|l| {
            if full(l) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let layer_rope_theta: Vec<f64> = (0..n).map(|l| if full(l) { 0.0 } else { 5e5 }).collect();
    let p = "model.language_model.layers";
    let mut exclude = vec!["lm_head".to_string()];
    for l in 0..n {
        for m in ["mlp.gate", "mlp.pre_expert_proj", "mlp.post_expert_proj"] {
            exclude.push(format!("{p}.{l}.{m}"));
        }
        for m in ["gate_proj", "k_proj", "o_proj", "q_proj", "v_proj"] {
            exclude.push(format!("{p}.{l}.self_attn.{m}"));
        }
    }
    for l in [0, 61] {
        for e in 0..256 {
            for m in ["down_proj", "gate_proj", "up_proj"] {
                exclude.push(format!("{p}.{l}.mlp.experts.{e}.{m}"));
            }
        }
    }
    serde_json::to_string(&serde_json::json!({
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "vision_config": {"model_type": "muse_spark_vision"},
        "audio_config": {"model_type": "muse_spark_audio"},
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": {
                "input_tensors": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"},
                "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
            },
            "exclude": exclude,
            "layer_quant_config": {},
            "layer_type_quant_config": {}
        },
        "text_config": {
            "model_type": "muse_spark_text",
            "attention_bias": false,
            "final_logit_softcapping": 20.0,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 8192,
            "intermediate_size": 16384,
            "layer_rope_theta": layer_rope_theta,
            "layer_types": layer_types,
            "max_position_embeddings": 1048576,
            "mlp_layer_types": vec!["sparse"; n],
            "moe_hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_attention_heads": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": n,
            "num_key_value_heads": 16,
            "num_local_experts": 256,
            "num_shared_experts": 0,
            "output_multiplier": 0.17677669529663687,
            "post_norm_eps": 1e-08,
            "qk_scale_factor": 5.4730064863875,
            "residual_gate_temperature": 0.3,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "routed_scaling_factor": 1.0,
            "sliding_window": 2048,
            "tie_word_embeddings": false,
            "vocab_size": 202048
        }
    }))
    .unwrap()
}

#[test]
fn official_muse_spark_text_manifest_is_exact() {
    let g = build_from_config_json(&official_config()).expect("official Muse Spark graph");
    let manifest = g.checkpoint_manifest();
    assert_eq!(manifest.len(), 48_611);
    assert_eq!(g.expert_bindings.len(), 62);
    let elements: u64 = manifest
        .iter()
        .map(|w| {
            w.shape
                .unwrap()
                .dims()
                .iter()
                .map(|d| d.as_static().unwrap() as u64)
                .product::<u64>()
        })
        .sum();
    assert_eq!(elements, 819_990_552_064);
    assert_eq!(g.checkpoint_storage_bytes(), Some(504_759_203_840));

    let window = |w: Option<u32>| {
        g.count_ops(|o| matches!(o, Op::Attention { sliding_window, .. } if *sliding_window == w))
    };
    assert_eq!((window(Some(2048)), window(None)), (46, 16));
    assert_eq!(
        g.count_ops(|o| matches!(o, Op::Rope { theta, .. } if *theta == 5e5)),
        92
    );

    let dtype = |name: &str| manifest.iter().find(|w| w.name == name).unwrap().dtype;
    let p = "model.language_model.layers";
    assert_eq!(
        dtype(&format!("{p}.0.mlp.experts.7.up_proj.weight")),
        DType::BF16
    );
    assert_eq!(
        dtype(&format!("{p}.61.mlp.experts.7.up_proj.weight")),
        DType::BF16
    );
    assert_eq!(
        dtype(&format!("{p}.30.mlp.experts.7.up_proj.weight")),
        DType::F4
    );
    assert_eq!(
        dtype(&format!("{p}.30.self_attn.q_proj.weight")),
        DType::BF16
    );
    assert_eq!(dtype(&format!("{p}.30.mlp.gate.weight")), DType::F32);
    let scaled = g
        .expert_bindings
        .iter()
        .flat_map(|l| &l.routed_experts)
        .filter(|e| e.gate.scale.is_some())
        .count();
    assert_eq!(scaled, 60 * 256);
}
