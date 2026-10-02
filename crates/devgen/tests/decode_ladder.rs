use clap::Parser;

static EMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

#[test]
fn gemma31_nvidia_ladder_keeps_fused_projection_shape_through_b16() {
    let _guard = EMIT_LOCK.lock().unwrap();
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
        emit_cfg: Some(emit.clone()),
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

#[test]
fn gemma_fp8_lt_decode_quantizes_only_selected_rungs() {
    let _guard = EMIT_LOCK.lock().unwrap();
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("gemma_fp8_lt_decode");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), r#"{
        "model_type":"gemma4_text", "hidden_size":3840, "intermediate_size":15360,
        "num_hidden_layers":1, "num_attention_heads":16, "head_dim":256,
        "global_head_dim":512, "num_key_value_heads":8, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "sliding_window":1024, "rms_norm_eps":1e-6,
        "vocab_size":1024, "tie_word_embeddings":true, "final_logit_softcapping":30.0,
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25}},
        "layer_types":["sliding_attention"]
    }"#).unwrap();
    let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
    cfg.fp8 = true;
    cfg.w8a8 = true;
    cfg.decode_cublaslt = true;
    cfg.decode_cublaslt_min_rows = Some(128);
    cfg.decode_batch = 128;
    cfg.decode_ladder = Some("32,64,128".into());
    cfg.fp8_decode_tc64 = true;
    cfg.max_chunk = Some(256);
    cfg.emit_packed_prefill = Some(false);
    cfg.fuse_argmax = true;
    for head in [false, true] {
        cfg.decode_cublaslt_head = head;
        devgen::run_verified(devgen::EmitArgs {
            dir: root.clone(), ctx: 256, out: root.join("model.pkt").display().to_string(),
            n_cu: 132, tp: 1, block_spec: None, embed_cubin: None,
            embed_hsaco: None, rope_gen: true, l2_layout: None, gpu: "H100 SXM5".into(),
            arch: "sm_90a".into(), emit_cfg: Some(cfg.clone()),
            whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
        }, Some(Box::new(move |model| {
            use packet::dev::{DevOp, TENSOR_NONE};
            let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
            assert_eq!(&model.prog_t[lo..], &[32, 64, 128]);
            let narrow = &model.progs[lo];
            assert!(narrow.insts.iter().any(|i| i.op == DevOp::GemvFp8 as u16));
            assert!(!narrow.insts.iter().any(|i| i.op == DevOp::QuantFp8 as u16));
            let middle = &model.progs[lo + 1];
            assert!(middle.insts.iter().any(|i| i.op == DevOp::GemvFp8 as u16 && i.i[0] == 64));
            assert!(!middle.insts.iter().any(|i| i.op == DevOp::QuantFp8 as u16));
            let wide = &model.progs[lo + 2];
            assert!(model.progs[lo].insts.iter().any(|i| i.op == DevOp::GemvArgmax as u16));
            assert_eq!(wide.insts.iter().any(|i| i.op == DevOp::GemvArgmax as u16), !head);
            if head {
                let (pc, lm) = wide.insts.iter().enumerate().find(|(_, i)| {
                    i.op == DevOp::Gemv as u16
                        && model.tensors[i.t[2] as usize].name.ends_with("embed_tokens.weight")
                }).expect("selected head must remain BF16 Gemv for the Lt route");
                assert_eq!(&lm.i[..3], &[128, 1024, 3840]);
                let segment = wide.stream.iter().find(|e| e.inst as usize == pc).unwrap().seg;
                assert!(wide.stream.iter().filter(|e| e.seg == segment).all(|e| e.inst as usize == pc));
                assert!(wide.insts.iter().any(|i| i.op == DevOp::SoftCap as u16 && i.i[0] == 128 * 1024));
                assert!(wide.insts.iter().any(|i| i.op == DevOp::Argmax as u16 && i.i[1] == 128));
            }
            assert!(!wide.insts.iter().any(|i| matches!(DevOp::from_u16(i.op),
                Some(DevOp::GemvFp8 | DevOp::GemvGluFp8 | DevOp::GemvQkvFp8))));
            let projections: Vec<_> = wide.insts.iter().filter(|i| i.op == DevOp::GemmFp8 as u16).collect();
            assert!(projections.len() >= 6);
            for op in projections {
                assert_eq!(op.i[0], 128);
                assert!(!op.t[..5].contains(&TENSOR_NONE));
                assert!(op.i[3..].iter().all(|&i| i == 0));
                assert!(wide.insts.iter().any(|q| q.op == DevOp::QuantFp8 as u16
                    && q.t[0] == op.t[1] && q.t[2] == op.t[3]));
            }
            Ok(devgen::LeanReport { reason: Some("structural FP8 decode test".into()), ..Default::default() })
        })));
        let config = std::fs::read_to_string(root.join("plow_config.h")).unwrap();
        assert!(config.contains("#define PLOW_NV_FP8_LT_DECODE 1"));
        assert!(config.contains("#define PLOW_NV_FP8_DECODE_TC64 1"));
    }
}
