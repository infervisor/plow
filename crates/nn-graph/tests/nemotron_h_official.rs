//! nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4: the released `llm_config`
//! geometry and ModelOpt `quantized_layers`, against its safetensors headers.
#![recursion_limit = "256"]

use std::collections::BTreeMap;

use nn_graph::models::{build_from_config_json, ModelConfig};
use nn_graph::{DType, LinearQuantFormat, MoeExpertAct, Op};

/// `layers_block_type`: M = mamba, E = moe, * = attention.
const PATTERN: &str =
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";

/// Every `language_model.` tensor of the checkpoint outside `mtp.`, as
/// `count pattern dtype shape` with layer/expert indices folded to `N`
/// (generated from the 88 shard headers).
const TEXT_TENSORS: &str = "\
1 language_model.backbone.embeddings.weight BF16 131072x4096
40 language_model.backbone.layers.N.mixer.A_log BF16 128
40 language_model.backbone.layers.N.mixer.D BF16 128
40 language_model.backbone.layers.N.mixer.conv1d.bias BF16 10240
40 language_model.backbone.layers.N.mixer.conv1d.weight BF16 10240x1x4
40 language_model.backbone.layers.N.mixer.dt_bias BF16 128
20480 language_model.backbone.layers.N.mixer.experts.N.down_proj.input_scale F32 scalar
20480 language_model.backbone.layers.N.mixer.experts.N.down_proj.weight U8 1024x1344
20480 language_model.backbone.layers.N.mixer.experts.N.down_proj.weight_scale F8_E4M3 1024x168
20480 language_model.backbone.layers.N.mixer.experts.N.down_proj.weight_scale_2 F32 scalar
20480 language_model.backbone.layers.N.mixer.experts.N.up_proj.input_scale F32 scalar
20480 language_model.backbone.layers.N.mixer.experts.N.up_proj.weight U8 2688x512
20480 language_model.backbone.layers.N.mixer.experts.N.up_proj.weight_scale F8_E4M3 2688x64
20480 language_model.backbone.layers.N.mixer.experts.N.up_proj.weight_scale_2 F32 scalar
40 language_model.backbone.layers.N.mixer.fc1_latent_proj.weight BF16 1024x4096
40 language_model.backbone.layers.N.mixer.fc2_latent_proj.weight BF16 4096x1024
40 language_model.backbone.layers.N.mixer.gate.e_score_correction_bias F32 512
40 language_model.backbone.layers.N.mixer.gate.weight BF16 512x4096
40 language_model.backbone.layers.N.mixer.in_proj.input_scale F32 1
40 language_model.backbone.layers.N.mixer.in_proj.weight F8_E4M3 18560x4096
40 language_model.backbone.layers.N.mixer.in_proj.weight_scale F32 scalar
8 language_model.backbone.layers.N.mixer.k_proj.input_scale F32 1
8 language_model.backbone.layers.N.mixer.k_proj.weight F8_E4M3 256x4096
8 language_model.backbone.layers.N.mixer.k_proj.weight_scale F32 scalar
40 language_model.backbone.layers.N.mixer.norm.weight BF16 8192
8 language_model.backbone.layers.N.mixer.o_proj.input_scale F32 1
8 language_model.backbone.layers.N.mixer.o_proj.weight F8_E4M3 4096x4096
8 language_model.backbone.layers.N.mixer.o_proj.weight_scale F32 scalar
40 language_model.backbone.layers.N.mixer.out_proj.input_scale F32 1
40 language_model.backbone.layers.N.mixer.out_proj.weight F8_E4M3 4096x8192
40 language_model.backbone.layers.N.mixer.out_proj.weight_scale F32 scalar
8 language_model.backbone.layers.N.mixer.q_proj.input_scale F32 1
8 language_model.backbone.layers.N.mixer.q_proj.weight F8_E4M3 4096x4096
8 language_model.backbone.layers.N.mixer.q_proj.weight_scale F32 scalar
40 language_model.backbone.layers.N.mixer.shared_experts.down_proj.input_scale F32 scalar
40 language_model.backbone.layers.N.mixer.shared_experts.down_proj.weight U8 4096x2688
40 language_model.backbone.layers.N.mixer.shared_experts.down_proj.weight_scale F8_E4M3 4096x336
40 language_model.backbone.layers.N.mixer.shared_experts.down_proj.weight_scale_2 F32 scalar
40 language_model.backbone.layers.N.mixer.shared_experts.up_proj.input_scale F32 scalar
40 language_model.backbone.layers.N.mixer.shared_experts.up_proj.weight U8 5376x2048
40 language_model.backbone.layers.N.mixer.shared_experts.up_proj.weight_scale F8_E4M3 5376x256
40 language_model.backbone.layers.N.mixer.shared_experts.up_proj.weight_scale_2 F32 scalar
8 language_model.backbone.layers.N.mixer.v_proj.input_scale F32 1
8 language_model.backbone.layers.N.mixer.v_proj.weight F8_E4M3 256x4096
8 language_model.backbone.layers.N.mixer.v_proj.weight_scale F32 scalar
88 language_model.backbone.layers.N.norm.weight BF16 4096
1 language_model.backbone.norm_f.weight BF16 4096
1 language_model.lm_head.input_scale F32 1
1 language_model.lm_head.weight F8_E4M3 131072x4096
1 language_model.lm_head.weight_scale F32 scalar
";

fn layer_types() -> Vec<&'static str> {
    PATTERN
        .chars()
        .map(|c| match c {
            'M' => "mamba",
            'E' => "moe",
            _ => "attention",
        })
        .collect()
}

fn official_config() -> String {
    let p = "language_model.backbone.layers";
    let mut quantized = serde_json::Map::new();
    let fp8 = serde_json::json!({"quant_algo": "FP8"});
    let nvfp4 = serde_json::json!({"quant_algo": "NVFP4", "group_size": 16});
    for (l, t) in layer_types().into_iter().enumerate() {
        let mut put = |m: String, q: &serde_json::Value| {
            quantized.insert(format!("{p}.{l}.mixer.{m}"), q.clone());
        };
        match t {
            "mamba" => {
                for m in ["in_proj", "out_proj"] {
                    put(m.into(), &fp8);
                }
            }
            "attention" => {
                for m in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(m.into(), &fp8);
                }
            }
            _ => {
                for m in ["shared_experts.up_proj", "shared_experts.down_proj"] {
                    put(m.into(), &nvfp4);
                }
                for e in 0..512 {
                    put(format!("experts.{e}.up_proj"), &nvfp4);
                    put(format!("experts.{e}.down_proj"), &nvfp4);
                }
            }
        }
    }
    quantized.insert("language_model.lm_head".into(), fp8);
    serde_json::to_string(&serde_json::json!({
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "dtype": "bfloat16",
        "sound_config": null,
        "vision_config": {"model_type": "radio", "hidden_size": 1280, "num_hidden_layers": 32},
        "quantization_config": {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
            "producer": {"name": "modelopt", "version": "0.37.0"},
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
            "quantized_layers": quantized,
        },
        "llm_config": {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "attention_bias": false,
            "chunk_size": 128,
            "conv_kernel": 4,
            "dtype": "bfloat16",
            "expand": 2,
            "head_dim": 128,
            "hidden_size": 4096,
            "intermediate_size": 2688,
            "layer_norm_epsilon": 1e-05,
            "layers_block_type": layer_types(),
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 128,
            "mamba_proj_bias": false,
            "max_position_embeddings": 1048576,
            "mlp_bias": false,
            "mlp_hidden_act": "relu2",
            "moe_intermediate_size": 2688,
            "moe_latent_size": 1024,
            "moe_shared_expert_intermediate_size": 5376,
            "mtp_layers_block_type": ["attention", "moe"],
            "n_group": 1,
            "n_groups": 8,
            "n_routed_experts": 512,
            "n_shared_experts": 1,
            "norm_topk_prob": true,
            "num_attention_heads": 32,
            "num_experts_per_tok": 22,
            "num_key_value_heads": 2,
            "num_nextn_predict_layers": 1,
            "routed_scaling_factor": 5.0,
            "sliding_window": null,
            "ssm_state_size": 128,
            "tie_word_embeddings": false,
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "time_step_min": 0.001,
            "topk_group": 1,
            "use_bias": false,
            "use_conv_bias": true,
            "vocab_size": 131072
        }
    }))
    .unwrap()
}

fn fold(name: &str) -> String {
    name.split('.')
        .map(|part| {
            if part.parse::<u32>().is_ok() {
                "N"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

#[test]
fn official_text_manifest_matches_checkpoint_headers() {
    let json = official_config();
    let ModelConfig::NemotronH(cfg) = ModelConfig::from_json(&json).unwrap() else {
        panic!("nemotron_h_omni must parse as NemotronH");
    };
    assert_eq!(cfg.weight_prefix, "language_model.");
    assert_eq!(cfg.time_step_limit, (0.0, f32::INFINITY));
    assert!(cfg.kv_cache_fp8());
    assert_eq!(
        cfg.omitted,
        [
            "vision_config (RADIO vision tower + mlp1 projector)",
            "mtp (1 draft layer(s))"
        ]
    );

    let g = build_from_config_json(&json).expect("official Nemotron-H graph");
    let manifest = g.checkpoint_manifest();
    let mut got: BTreeMap<String, usize> = BTreeMap::new();
    for w in &manifest {
        let dims = w
            .shape
            .unwrap()
            .dims()
            .iter()
            .map(|d| d.as_static().unwrap().to_string())
            .collect::<Vec<_>>();
        let shape = if dims.is_empty() {
            "scalar".to_string()
        } else {
            dims.join("x")
        };
        let dtype = w.dtype.safetensors_name().unwrap();
        *got.entry(format!("{} {dtype} {shape}", fold(w.name)))
            .or_default() += 1;
    }
    let want: BTreeMap<String, usize> = TEXT_TENSORS
        .lines()
        .map(|l| {
            let (count, rest) = l.split_once(' ').unwrap();
            (rest.to_string(), count.parse().unwrap())
        })
        .collect();
    assert_eq!(got, want);
    assert_eq!(manifest.len(), 164_989);
    assert_eq!(g.checkpoint_storage_bytes(), Some(71_532_437_000));
}

#[test]
fn official_graph_structure() {
    let g = build_from_config_json(&official_config()).unwrap();
    assert_eq!(g.blocks.len(), 88);
    assert_eq!(g.count_ops(|o| matches!(o, Op::Mamba2Scan { .. })), 40);
    assert_eq!(g.count_ops(|o| matches!(o, Op::Rope { .. })), 0);
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            Op::Attention {
                num_heads: 32,
                num_kv_heads: 2,
                head_dim: 128,
                causal: true,
                sliding_window: None,
                ..
            }
        )),
        8
    );
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            Op::MoeRouter {
                num_experts: 512,
                top_k: 22,
                norm_topk: true,
                correction_bias: true,
                ..
            }
        )),
        40
    );
    assert_eq!(
        g.count_ops(|o| matches!(
            o,
            Op::MoeExperts {
                act: MoeExpertAct::Relu2,
                intermediate_size: 2688,
                ..
            }
        )),
        40
    );
    assert!(g.nodes.iter().all(|n| match &n.op {
        Op::Mamba2Scan {
            num_heads,
            head_dim,
            n_groups,
            state_size,
            chunk_size,
            dt_limit,
        } =>
            (*num_heads, *head_dim, *n_groups, *state_size, *chunk_size) == (128, 64, 8, 128, 128)
                && *dt_limit == (0.0, f32::INFINITY),
        Op::MoeRouter { route_scale, .. } => *route_scale == 5.0,
        _ => true,
    }));

    assert_eq!(g.expert_bindings.len(), 40);
    for layer in &g.expert_bindings {
        assert_eq!(layer.routed_experts.len(), 512);
        assert_eq!((layer.n_group, layer.topk_group), (1, 1));
        for e in &layer.routed_experts {
            assert!(e.gate.is_none());
            for proj in [&e.up, &e.down] {
                assert!(proj.scale.is_some() && proj.scale_2.is_some());
                assert!(proj.input_scale.is_some());
            }
        }
    }
    let formats = |f: LinearQuantFormat| {
        g.quant_linear_bindings
            .iter()
            .filter(|b| b.format == f)
            .count()
    };
    // in/out_proj x 40, q/k/v/o x 8, lm_head; shared up/down x 40.
    assert_eq!(formats(LinearQuantFormat::Fp8PerTensor), 80 + 32 + 1);
    assert_eq!(formats(LinearQuantFormat::Nvfp4 { group: 16 }), 80);
    let dtype = |name: &str| {
        g.checkpoint_manifest()
            .iter()
            .find(|w| w.name == name)
            .unwrap()
            .dtype
    };
    assert_eq!(
        dtype("language_model.backbone.layers.1.mixer.gate.weight"),
        DType::BF16
    );
}
