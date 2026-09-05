//! Hermetic `--model` compilation through a metadata-only HF snapshot.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use nn_graph::models::{build_text_generation_from_config_json_at, ShapeBucket};
use plowc::{compile, Options, Parallel, Source};
use schedule::Phase;

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn tempdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "plowc-model-metadata-{name}-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_metadata(dir: &Path, config: &str) {
    std::fs::write(dir.join("config.json"), config).unwrap();
    let outer: serde_json::Value = serde_json::from_str(config).unwrap();
    let model_type = outer["model_type"].as_str().unwrap_or_default();
    let graph = build_text_generation_from_config_json_at(config, &ShapeBucket::default())
        .expect("fixture graph");
    let mut weight_map = BTreeMap::new();
    for weight in graph.checkpoint_manifest() {
        let name = if weight.name.starts_with("model.") || weight.name.starts_with("lm_head.") {
            weight.name.to_string()
        } else if model_type == "kimi_k3" {
            format!("language_model.model.{}", weight.name)
        } else if model_type == "gemma4" {
            format!("model.language_model.{}", weight.name)
        } else {
            format!("model.{}", weight.name)
        };
        weight_map.insert(name, "model-00001-of-00001.safetensors");
    }
    // Non-text tensors are legal index entries and must not be mistaken for
    // missing compiler coverage.
    weight_map.insert(
        "model.visual.patch_embed.weight".to_string(),
        "model-00001-of-00001.safetensors",
    );
    weight_map.insert(
        "model.embed_vision.embedding_projection.weight".to_string(),
        "model-00001-of-00001.safetensors",
    );
    weight_map.insert(
        "mtp.layers.0.proj.weight".to_string(),
        "model-00001-of-00001.safetensors",
    );
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 1234},
            "weight_map": weight_map,
        }))
        .unwrap(),
    )
    .unwrap();
    for (name, contents) in [
        ("tokenizer.json", "{}"),
        ("tokenizer_config.json", r#"{"chat_template":"fixture"}"#),
        ("vocab.json", "{}"),
        ("merges.txt", "#version: 0.2"),
        ("chat_template.jinja", "{{ messages }}"),
        ("generation_config.json", "{}"),
    ] {
        std::fs::write(dir.join(name), contents).unwrap();
    }
    // The resolver must neither read nor copy this shard.
    std::fs::write(
        dir.join("model-00001-of-00001.safetensors"),
        "not safetensors: compilation must not inspect this file",
    )
    .unwrap();
}

fn replace_index_with_monolithic_safetensors(dir: &Path) {
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let model_type = serde_json::from_str::<serde_json::Value>(&config).unwrap()["model_type"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let graph = build_text_generation_from_config_json_at(&config, &ShapeBucket::default())
        .expect("fixture graph");
    let manifest = graph
        .checkpoint_manifest()
        .into_iter()
        .map(|weight| {
            let name = if weight.name.starts_with("model.") || weight.name.starts_with("lm_head.") {
                weight.name.to_string()
            } else if model_type == "kimi_k3" {
                format!("language_model.model.{}", weight.name)
            } else if model_type == "gemma4" {
                format!("model.language_model.{}", weight.name)
            } else {
                format!("model.{}", weight.name)
            };
            let shape = weight
                .shape
                .unwrap()
                .dims()
                .iter()
                .map(|dim| dim.as_static().unwrap() as u64)
                .collect::<Vec<_>>();
            (name, (weight.dtype, shape))
        })
        .collect::<BTreeMap<_, _>>();
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for name in index["weight_map"].as_object().unwrap().keys() {
        let (dtype, shape) = manifest
            .get(name)
            .map(|(dtype, shape)| (dtype.safetensors_name().unwrap(), shape.clone()))
            .unwrap_or(("U8", vec![1]));
        let elements = shape.iter().product::<u64>();
        let bytes = elements
            * nn_graph::DType::from_safetensors_name(dtype)
                .unwrap()
                .byte_size()
                .unwrap() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let path = dir.join("model.safetensors");
    let mut file = std::fs::File::create(&path).unwrap();
    use std::io::Write;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + offset).unwrap();
    std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
}

fn options(out: PathBuf, gpu: &str) -> Options {
    Options {
        no_tuning: true,
        tuning_db: None,
        gpu: gpu.into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![1],
        phases: vec![Phase::Decode],
        page_kib: 16,
        out,
        lean_verify: false,
        counter_elim: false,
        scope_narrow: false,
        prefetch: false,
        sram_fit: false,
        lean_oracle: false,
        emit_sample: false,
        emit_tokenize: false,
        emit_trace: false,
        kv: Default::default(),
        weight_dtype_override: None,
    }
}

const LLAMA: &str = r#"{
  "model_type":"llama", "vocab_size":64, "hidden_size":32,
  "intermediate_size":64, "num_hidden_layers":1,
  "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
  "tie_word_embeddings":false, "torch_dtype":"bfloat16"
}"#;

const QWEN3: &str = r#"{
  "model_type":"qwen3", "vocab_size":64, "hidden_size":32,
  "intermediate_size":64, "num_hidden_layers":1,
  "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
  "tie_word_embeddings":false, "torch_dtype":"bfloat16"
}"#;

const QWEN25: &str = r#"{
  "architectures":["Qwen2ForCausalLM"], "model_type":"qwen2",
  "vocab_size":64, "hidden_size":32, "intermediate_size":64,
  "num_hidden_layers":1, "num_attention_heads":4,
  "num_key_value_heads":2, "rms_norm_eps":1e-6,
  "rope_theta":1000000.0, "use_sliding_window":false,
  "tie_word_embeddings":false, "torch_dtype":"bfloat16"
}"#;

const GEMMA3: &str = r#"{
  "model_type":"gemma3", "vocab_size":64, "hidden_size":32,
  "intermediate_size":64, "num_hidden_layers":1,
  "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
  "sliding_window":16, "sliding_window_pattern":1,
  "query_pre_attn_scalar":8.0, "use_qk_norm":true,
  "torch_dtype":"bfloat16"
}"#;

const QWEN35: &str = r#"{
  "architectures":["Qwen3_5ForConditionalGeneration"],
  "model_type":"qwen3_5",
  "text_config": {
    "model_type":"qwen3_5_text", "vocab_size":64, "hidden_size":32,
    "intermediate_size":64, "num_hidden_layers":4,
    "num_attention_heads":4, "num_key_value_heads":1, "head_dim":8,
    "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    "linear_conv_kernel_dim":4, "linear_key_head_dim":8,
    "linear_num_key_heads":1, "linear_num_value_heads":3,
    "linear_value_head_dim":8, "rms_norm_eps":1e-6,
    "rope_parameters": {"rope_theta":10000000.0,"partial_rotary_factor":0.25,
      "rope_type":"default","mrope_interleaved":true},
    "attention_bias":false, "attn_output_gate":true, "hidden_act":"silu",
    "mamba_ssm_dtype":"float32", "output_gate_type":"swish",
    "tie_word_embeddings":false, "dtype":"bfloat16"
  }
}"#;

const DEEPSEEK_V3: &str = r#"{
  "architectures":["DeepseekV3ForCausalLM"], "model_type":"deepseek_v3",
  "vocab_size":64, "hidden_size":32, "intermediate_size":64,
  "num_hidden_layers":2, "num_attention_heads":4, "rms_norm_eps":1e-6,
  "rope_theta":10000.0, "q_lora_rank":16, "kv_lora_rank":8,
  "qk_rope_head_dim":4, "qk_nope_head_dim":8, "v_head_dim":8,
  "n_routed_experts":4, "n_shared_experts":1, "num_experts_per_tok":2,
  "moe_intermediate_size":16, "first_k_dense_replace":1,
  "scoring_func":"sigmoid", "topk_method":"noaux_tc",
  "n_group":1, "topk_group":1, "norm_topk_prob":true,
  "routed_scaling_factor":2.5,
  "torch_dtype":"bfloat16"
}"#;

const KIMI_K2: &str = r#"{
  "architectures":["DeepseekV3ForCausalLM"], "model_type":"kimi_k2",
  "vocab_size":64, "hidden_size":32, "intermediate_size":64,
  "num_hidden_layers":2, "num_attention_heads":4, "rms_norm_eps":1e-6,
  "rope_theta":10000.0, "q_lora_rank":16, "kv_lora_rank":8,
  "qk_rope_head_dim":4, "qk_nope_head_dim":8, "v_head_dim":8,
  "n_routed_experts":4, "n_shared_experts":1, "num_experts_per_tok":2,
  "moe_intermediate_size":16, "first_k_dense_replace":1,
  "scoring_func":"sigmoid", "topk_method":"noaux_tc",
  "n_group":1, "topk_group":1, "norm_topk_prob":true,
  "routed_scaling_factor":2.827,
  "torch_dtype":"bfloat16"
}"#;

// Scaled geometry with the same architecture fields as zai-org/GLM-5.3.
const GLM53: &str = r#"{
  "architectures":["GlmMoeDsaForCausalLM"], "model_type":"glm_moe_dsa",
  "vocab_size":64, "hidden_size":32, "intermediate_size":64,
  "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":4,
  "head_dim":8, "rms_norm_eps":1e-5, "attention_bias":false,
  "hidden_act":"silu", "q_lora_rank":16, "kv_lora_rank":8,
  "qk_head_dim":12, "qk_nope_head_dim":8, "qk_rope_head_dim":4,
  "v_head_dim":8, "rope_interleave":true,
  "rope_parameters":{"rope_theta":8000000.0,"rope_type":"default"},
  "first_k_dense_replace":1, "n_routed_experts":4, "n_shared_experts":1,
  "num_experts_per_tok":2, "moe_intermediate_size":16,
  "mlp_layer_types":["dense","sparse"], "indexer_types":["full","shared"],
  "index_head_dim":8, "index_n_heads":2, "index_topk":4,
  "index_topk_freq":4, "indexer_rope_interleave":true,
  "index_skip_topk_offset":1, "num_nextn_predict_layers":1,
  "index_share_for_mtp_iteration":true,
  "scoring_func":"sigmoid", "norm_topk_prob":true,
  "routed_scaling_factor":2.5, "n_group":1, "topk_group":1,
  "topk_method":"noaux_tc", "moe_router_dtype":"float32",
  "dtype":"bfloat16",
  "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
    "quant_method":"fp8","weight_block_size":[128,128]}
}"#;

// Scaled google/gemma-4-31B text tower; the official wrapper is model_type gemma4.
const GEMMA4: &str = r#"{
  "architectures":["Gemma4ForConditionalGeneration"], "model_type":"gemma4",
  "dtype":"bfloat16",
  "text_config": {
    "model_type":"gemma4_text", "vocab_size":64, "hidden_size":32,
    "intermediate_size":64, "num_hidden_layers":2,
    "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
    "num_global_key_value_heads":1, "global_head_dim":16,
    "attention_k_eq_v":true, "tie_word_embeddings":true, "use_qk_norm":true,
    "final_logit_softcapping":30.0,
    "query_pre_attn_scalar":16.0, "sliding_window":16,
    "layer_types":["sliding_attention","full_attention"],
    "rope_parameters": {
      "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.5,"rope_type":"proportional"},
      "sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0,"rope_type":"default"}
    }
  }
}"#;

// Scaled moonshotai/Kimi-K3 text tower with both official mixer types.
const KIMI_K3: &str = r#"{
  "architectures":["KimiK3ForConditionalGeneration"], "model_type":"kimi_k3",
  "dtype":"bfloat16",
  "text_config": {
    "model_type":"kimi_linear", "vocab_size":64, "hidden_size":32,
    "intermediate_size":64, "num_hidden_layers":2, "num_attention_heads":4,
    "q_lora_rank":16, "kv_lora_rank":8, "qk_rope_head_dim":4,
    "qk_nope_head_dim":8, "v_head_dim":8, "mla_use_output_gate":true,
    "num_experts":4, "num_experts_per_token":2, "num_shared_experts":1,
    "moe_intermediate_size":16, "routed_expert_hidden_size":24,
    "first_k_dense_replace":1, "attn_res_block_size":2,
    "linear_attn_config": {
      "num_heads":4, "head_dim":8, "short_conv_kernel_size":4,
      "use_full_rank_gate":true, "full_attn_layers":[1], "kda_layers":[2]
    }
  }
}"#;

const MINIMAX_M2: &str = r#"{
  "architectures":["MiniMaxM2ForCausalLM"], "model_type":"minimax_m2",
  "hidden_size":3072, "num_hidden_layers":62, "attn_type_list":[1],
  "num_local_experts":256, "num_experts_per_tok":8
}"#;

const NEMOTRON3: &str = r#"{
  "architectures":["NemotronHForCausalLM"], "model_type":"nemotron_h",
  "hidden_size":2688, "num_hidden_layers":52,
  "hybrid_override_pattern":"MEMEM*", "conv_kernel":4,
  "mamba_head_dim":64, "mamba_num_heads":64, "ssm_state_size":128
}"#;

// Distinctive fields from deepseek-ai/DeepSeek-V4-Flash-0731. This is not a
// DeepSeek-V3 MLA configuration: it adds CSA/HCA compression, mHC, mixed
// FP4/FP8 storage, and an attached DSpark speculative module.
const DEEPSEEK_V4_FLASH_0731: &str = r#"{
  "architectures":["DeepseekV4ForCausalLM"], "model_type":"deepseek_v4",
  "hidden_size":4096, "num_hidden_layers":43, "num_attention_heads":64,
  "num_key_value_heads":1, "head_dim":512, "q_lora_rank":1024,
  "qk_rope_head_dim":64, "o_lora_rank":1024, "o_groups":8,
  "num_hash_layers":3, "compress_ratios":[0,0,4,128,4,0],
  "hc_mult":4, "hc_sinkhorn_iters":20, "n_routed_experts":256,
  "num_experts_per_tok":6, "expert_dtype":"fp4",
  "quantization_config":{"quant_method":"fp8","fmt":"e4m3",
    "scale_fmt":"ue8m0","weight_block_size":[128,128]},
  "dspark_block_size":5, "dspark_target_layer_ids":[40,41,42]
}"#;

// Scaled meta-models/Muse-Glimmer-30B wrapper. The text and vision networks
// both require dedicated semantics; neither may fall through to a generic graph.
const MUSE_GLIMMER: &str = r#"{
  "architectures":["MuseGlimmerForConditionalGeneration"],
  "model_type":"muse_glimmer", "dtype":"bfloat16",
  "text_config": {
    "model_type":"muse_glimmer_text", "vocab_size":64, "hidden_size":32,
    "intermediate_size":64, "num_hidden_layers":4,
    "num_attention_heads":4, "num_key_value_heads":1, "head_dim":8,
    "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
    "sliding_window":16, "final_logit_softcapping":20.0,
    "qk_scale_factor":3.87, "output_multiplier":0.196116,
    "rope_parameters":{"sliding_attention":{"rope_theta":500000.0},"full_attention":{"rope_theta":0.0}}
  },
  "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":16,"patch_size":14},
  "multimodal_projector_config":{"hidden_size":16,"out_hidden_size":32}
}"#;

#[test]
fn representative_hf_models_compile_from_metadata_only_for_selected_gpu() {
    for (family, config, gpu, hbm_capacity) in [
        ("llama", LLAMA, "h100", 80_u64 << 30),
        ("qwen3", QWEN3, "rtx6000", 96_u64 << 30),
        ("qwen25", QWEN25, "h100", 80_u64 << 30),
        ("gemma3", GEMMA3, "h100", 80_u64 << 30),
        ("qwen35", QWEN35, "rtx6000", 96_u64 << 30),
        ("gemma4", GEMMA4, "rtx6000", 96_u64 << 30),
        ("deepseek-v3", DEEPSEEK_V3, "h100", 80_u64 << 30),
        ("kimi-k2", KIMI_K2, "rtx6000", 96_u64 << 30),
        ("glm53", GLM53, "rtx6000", 96_u64 << 30),
    ] {
        let metadata = tempdir(&format!("{family}-source"));
        let out = tempdir(&format!("{family}-out"));
        write_metadata(&metadata, config);

        let report = compile(
            &Source::Model(metadata.to_string_lossy().into_owned()),
            &options(out.clone(), gpu),
        )
        .unwrap_or_else(|error| panic!("{family}: {error}"));
        let expected_weight_bytes =
            build_text_generation_from_config_json_at(config, &ShapeBucket::default())
                .unwrap()
                .checkpoint_storage_bytes()
                .unwrap();
        assert_eq!(report.gpu, gpu);
        assert_eq!(
            report.assets.as_ref().unwrap().regions.weights,
            expected_weight_bytes,
            "{family}: HBM accounting omitted checkpoint tensors"
        );
        assert_eq!(
            report.assets.as_ref().unwrap().regions.hbm_capacity,
            hbm_capacity,
            "{family}: selected hardware spec did not reach asset sizing"
        );
        assert!(
            report.assets.as_ref().unwrap().on_disk.hf_metadata_total > 0,
            "{family}: metadata files omitted from on-disk asset accounting"
        );
        assert_eq!(report.buckets.len(), 1);
        assert!(report.buckets[0].packet_bytes > 0);
        for name in [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            assert!(out.join(name).is_file(), "{family}: missing copied {name}");
        }
        assert!(out.join("decode_b1_s1.pkt").is_file());
        let metadata_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("hf_metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata_manifest["weight_shards_downloaded"], false);
        assert!(metadata_manifest["safetensors_index"]["tensors"]
            .as_u64()
            .is_some_and(|count| count > 0));
        assert!(!out.join("model-00001-of-00001.safetensors").exists());

        std::fs::remove_dir_all(metadata).ok();
        std::fs::remove_dir_all(out).ok();
    }
}

#[test]
fn tensor_parallel_hbm_reports_per_device_expert_weights() {
    let metadata = tempdir("deepseek-tp-hbm-source");
    let one_out = tempdir("deepseek-tp-hbm-one");
    let two_out = tempdir("deepseek-tp-hbm-two");
    write_metadata(&metadata, DEEPSEEK_V3);

    let one = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &options(one_out.clone(), "h100"),
    )
    .unwrap();
    let mut two_opts = options(two_out.clone(), "h100");
    two_opts.num_gpus = 2;
    let two = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &two_opts,
    )
    .unwrap();

    let one_weights = one.assets.unwrap().regions.weights;
    let two_assets = two.assets.unwrap();
    assert!(two_assets.regions.weights < one_weights);
    assert!(two_assets.regions.weights > one_weights.div_ceil(2));
    assert_eq!(
        two_assets.regions.total_hbm_peak,
        two_assets.regions.arena_peak + two_assets.regions.weights
    );

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(one_out).ok();
    std::fs::remove_dir_all(two_out).ok();
}

#[test]
fn qwen35_fp8_emits_explicit_scale_bindings_without_weight_shards() {
    let config = QWEN35.replace(
        r#""text_config": {"#,
        r#""quantization_config": {
    "activation_scheme":"dynamic", "fmt":"e4m3", "quant_method":"fp8",
    "modules_to_not_convert":[
      "lm_head",
      "model.language_model.layers.0.linear_attn.in_proj_a",
      "model.language_model.layers.0.linear_attn.in_proj_b",
      "model.language_model.layers.1.linear_attn.in_proj_a",
      "model.language_model.layers.1.linear_attn.in_proj_b",
      "model.language_model.layers.2.linear_attn.in_proj_a",
      "model.language_model.layers.2.linear_attn.in_proj_b"
    ],
    "weight_block_size":[128,128]
  },
  "text_config": {"#,
    );
    let metadata = tempdir("qwen35-fp8-source");
    let out = tempdir("qwen35-fp8-out");
    write_metadata(&metadata, &config);

    let report = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &options(out.clone(), "h100"),
    )
    .expect("metadata-only FP8 compile");
    assert!(report.buckets[0].packet_bytes > 0);
    assert!(
        report.weight_tiling.is_none(),
        "mixed BF16/FP8 weights must not be labeled as one homogeneous tiling"
    );
    let fp8: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("fp8_weights.json")).unwrap()).unwrap();
    assert_eq!(fp8["format"], "e4m3");
    assert_eq!(fp8["scale_dtype"], "f32");
    assert_eq!(fp8["total_scale_bytes"], 100);
    assert_eq!(fp8["bindings"].as_array().unwrap().len(), 25);
    assert!(fp8["bindings"].as_array().unwrap().iter().all(|binding| {
        binding["block_shape"] == serde_json::json!([128, 128])
            && binding["scale"]
                .as_str()
                .is_some_and(|name| name.ends_with(".weight_scale_inv"))
    }));
    let map: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("decode_b1_s1.map.json")).unwrap()).unwrap();
    let kv_layers = map["kv_paging"]["per_layer"].as_array().unwrap();
    assert_eq!(kv_layers.len(), 1);
    assert_eq!(kv_layers[0]["layer_idx"], 3);
    assert!(!out.join("model-00001-of-00001.safetensors").exists());

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn index_mismatch_fails_before_packet_emission() {
    let metadata = tempdir("bad-index-source");
    let out = tempdir("bad-index-out");
    write_metadata(&metadata, LLAMA);
    std::fs::write(
        metadata.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.not_a_real_weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();

    let error = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &options(out.clone(), "h100"),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("does not match the compiled text graph"));
    assert!(!out.join("decode_b1_s1.pkt").exists());

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn model_and_indexed_hf_dir_emit_identical_packets_and_maps() {
    for (family, config) in [("llama", LLAMA), ("qwen35", QWEN35), ("gemma4", GEMMA4)] {
        let metadata = tempdir(&format!("source-parity-{family}"));
        let model_out = tempdir(&format!("source-parity-{family}-model-out"));
        let hf_dir_out = tempdir(&format!("source-parity-{family}-hf-dir-out"));
        write_metadata(&metadata, config);

        let mut model_opts = options(model_out.clone(), "h100");
        model_opts.emit_sample = true;
        let mut hf_dir_opts = options(hf_dir_out.clone(), "h100");
        hf_dir_opts.emit_sample = true;

        compile(
            &Source::Model(metadata.to_string_lossy().into_owned()),
            &model_opts,
        )
        .unwrap_or_else(|error| panic!("{family} metadata model compile: {error}"));
        compile(&Source::HfDir(metadata.clone()), &hf_dir_opts)
            .unwrap_or_else(|error| panic!("{family} indexed hf-dir compile: {error}"));

        for file in ["decode_b1_s1.pkt", "decode_b1_s1.map.json"] {
            assert_eq!(
                std::fs::read(model_out.join(file)).unwrap(),
                std::fs::read(hf_dir_out.join(file)).unwrap(),
                "{family}: source access changed {file}"
            );
        }

        std::fs::remove_dir_all(metadata).ok();
        std::fs::remove_dir_all(model_out).ok();
        std::fs::remove_dir_all(hf_dir_out).ok();
    }
}

#[test]
fn model_and_monolithic_hf_dir_emit_identical_packets_and_maps() {
    let metadata = tempdir("source-parity-monolithic");
    let model_out = tempdir("source-parity-monolithic-model-out");
    let hf_dir_out = tempdir("source-parity-monolithic-hf-dir-out");
    write_metadata(&metadata, LLAMA);
    replace_index_with_monolithic_safetensors(&metadata);

    let mut model_opts = options(model_out.clone(), "h100");
    model_opts.emit_sample = true;
    let mut hf_dir_opts = options(hf_dir_out.clone(), "h100");
    hf_dir_opts.emit_sample = true;

    compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &model_opts,
    )
    .expect("monolithic metadata model compile");
    compile(&Source::HfDir(metadata.clone()), &hf_dir_opts).expect("monolithic hf-dir compile");

    for file in ["decode_b1_s1.pkt", "decode_b1_s1.map.json"] {
        assert_eq!(
            std::fs::read(model_out.join(file)).unwrap(),
            std::fs::read(hf_dir_out.join(file)).unwrap(),
            "monolithic source access changed {file}"
        );
    }

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(model_out).ok();
    std::fs::remove_dir_all(hf_dir_out).ok();
}

#[test]
fn metadata_weight_dtype_override_is_applied_for_both_source_forms() {
    let metadata = tempdir("weight-override-source");
    let auto_out = tempdir("weight-override-auto-out");
    let model_out = tempdir("weight-override-model-out");
    let hf_dir_out = tempdir("weight-override-hf-dir-out");
    write_metadata(&metadata, LLAMA);

    let auto = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &options(auto_out.clone(), "h100"),
    )
    .expect("automatic dtype compile");
    let mut model_opts = options(model_out.clone(), "h100");
    model_opts.weight_dtype_override = Some(nn_graph::DType::F8E4M3);
    let model = compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &model_opts,
    )
    .expect("model override compile");
    let mut hf_dir_opts = options(hf_dir_out.clone(), "h100");
    hf_dir_opts.weight_dtype_override = Some(nn_graph::DType::F8E4M3);
    let hf_dir =
        compile(&Source::HfDir(metadata.clone()), &hf_dir_opts).expect("hf-dir override compile");

    assert_ne!(
        std::fs::read(auto_out.join("decode_b1_s1.pkt")).unwrap(),
        std::fs::read(model_out.join("decode_b1_s1.pkt")).unwrap(),
        "weight dtype override did not change GEMM packet variants"
    );
    assert_eq!(
        std::fs::read(model_out.join("decode_b1_s1.pkt")).unwrap(),
        std::fs::read(hf_dir_out.join("decode_b1_s1.pkt")).unwrap(),
        "source form changed overridden packet"
    );
    let model_weights = model.assets.unwrap().regions.weights;
    assert!(
        model_weights < auto.assets.unwrap().regions.weights,
        "FP8 projection override did not reduce reported weight storage"
    );
    assert_eq!(hf_dir.assets.unwrap().regions.weights, model_weights);

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(auto_out).ok();
    std::fs::remove_dir_all(model_out).ok();
    std::fs::remove_dir_all(hf_dir_out).ok();
}

#[test]
fn packet_routes_missing_from_model_source_fail_closed_instead_of_generic_lowering() {
    for (model_id, config, reason) in [
        (
            "moonshotai/Kimi-K3",
            KIMI_K3,
            "Kimi-K3 is supported by the dedicated MI355X devblob emitter",
        ),
        (
            "MiniMaxAI/MiniMax-M2",
            MINIMAX_M2,
            "MiniMax-M2 hybrid linear-attention MoE is not implemented",
        ),
        (
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            NEMOTRON3,
            "Nemotron Mamba-2 hybrid is not implemented in the nn-graph packet path",
        ),
        (
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            DEEPSEEK_V4_FLASH_0731,
            "deepseek_v4 (CSA/HCA hybrid attention, mHC residuals",
        ),
        (
            "meta-models/Muse-Glimmer-30B",
            MUSE_GLIMMER,
            "muse_glimmer (Muse Glimmer text uses alternating sliding/NoPE attention",
        ),
    ] {
        let metadata = tempdir(&model_id.replace('/', "-"));
        let out = tempdir("unsupported-out");
        std::fs::write(metadata.join("config.json"), config).unwrap();
        std::fs::write(
            metadata.join("model.safetensors.index.json"),
            r#"{"weight_map":{}}"#,
        )
        .unwrap();

        let error = compile(
            &Source::Model(metadata.to_string_lossy().into_owned()),
            &options(out.clone(), "h100"),
        )
        .expect_err("unsupported architecture must fail before packet lowering");
        assert!(
            error.to_string().contains(reason),
            "{model_id}: unexpected error: {error}"
        );
        assert!(!out.join("decode_b1_s1.pkt").exists(), "{model_id}");

        std::fs::remove_dir_all(metadata).ok();
        std::fs::remove_dir_all(out).ok();
    }
}

#[test]
fn qwen35_hybrid_kv_manifest_uses_the_16_full_attention_layer_ids() {
    let mut config: serde_json::Value = serde_json::from_str(QWEN35).unwrap();
    let text = config["text_config"].as_object_mut().unwrap();
    text.insert("num_hidden_layers".into(), 64.into());
    text.insert(
        "layer_types".into(),
        (0..64)
            .map(|layer| {
                serde_json::Value::String(
                    if layer % 4 == 3 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                    .into(),
                )
            })
            .collect::<Vec<_>>()
            .into(),
    );
    let config = serde_json::to_string(&config).unwrap();
    let metadata = tempdir("qwen35-kv-source");
    let out = tempdir("qwen35-kv-out");
    write_metadata(&metadata, &config);

    compile(
        &Source::Model(metadata.to_string_lossy().into_owned()),
        &options(out.clone(), "h100"),
    )
    .expect("64-layer metadata-only Qwen compile");

    let map: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("decode_b1_s1.map.json")).unwrap()).unwrap();
    let layers = map["kv_paging"]["per_layer"].as_array().unwrap();
    let ids: Vec<u64> = layers
        .iter()
        .map(|layer| layer["layer_idx"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, (3..64).step_by(4).collect::<Vec<_>>());
    assert_eq!(layers.len(), 16);

    std::fs::remove_dir_all(metadata).ok();
    std::fs::remove_dir_all(out).ok();
}
