use clap::Parser;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

#[test]
fn gemma31_nvidia_ladder_keeps_fused_projection_shape_through_b16() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("gemma31_cuda_ladder");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "gemma4_text",
          "hidden_size": 5376,
          "intermediate_size": 21504,
          "num_hidden_layers": 2,
          "num_attention_heads": 32,
          "head_dim": 256,
          "global_head_dim": 512,
          "num_key_value_heads": 16,
          "num_global_key_value_heads": 4,
          "attention_k_eq_v": true,
          "sliding_window": 1024,
          "rms_norm_eps": 1e-6,
          "vocab_size": 262144,
          "final_logit_softcapping": 30.0,
          "tie_word_embeddings": true,
          "layer_types": ["sliding_attention", "full_attention"],
          "rope_parameters": {
            "sliding_attention": { "rope_theta": 10000.0, "partial_rotary_factor": 1.0 },
            "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25 }
          }
        }"#,
    )
    .unwrap();

    let emit = Args::try_parse_from([
        "test",
        "--emit-decode-batch",
        "16",
        "--emit-decode-batch-ladder",
        "1,2,4,8,16",
    ])
    .unwrap()
    .emit;
    let out = root.join("model.pkt");
    devgen::run(devgen::EmitArgs {
        dir: root.clone(),
        ctx: 2048,
        out: out.to_string_lossy().into_owned(),
        n_cu: 132,
        tp: 1,
        block_spec: Some("0..2".into()),
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen: true,
        l2_layout: None,
        gpu: "H100 SXM5".into(),
        arch: "sm_90a".into(),
        emit_cfg: Some(emit),
        whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
    });

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("build.json")).unwrap()).unwrap();
    let decode: Vec<_> = manifest["programs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|program| program["kind"] == "decode")
        .collect();
    assert_eq!(decode.len(), 5);
    let expected_insts = decode[0]["insts"].as_u64().unwrap();
    for program in decode {
        let batch = program["batch"].as_u64().unwrap();
        let arms = program["arms"].as_array().unwrap();
        assert_eq!(program["insts"], expected_insts, "B{batch} changed shape");
        assert!(arms.iter().any(|arm| arm == "GemvQkv"), "B{batch}");
        assert!(arms.iter().any(|arm| arm == "GemvGlu"), "B{batch}");
        assert!(!arms.iter().any(|arm| arm == "Glu"), "B{batch}");
    }
}
