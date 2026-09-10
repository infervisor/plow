use clap::Parser;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

#[test]
fn aggregate_ladders_keep_request_sized_rings_and_require_masked_objects() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("request_chunk_limit");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{
        "model_type":"gemma4_text", "hidden_size":3840, "intermediate_size":15360,
        "num_hidden_layers":2, "num_attention_heads":16, "head_dim":256,
        "global_head_dim":512, "num_key_value_heads":8, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "sliding_window":1024, "rms_norm_eps":1e-6,
        "vocab_size":262144, "final_logit_softcapping":30.0, "tie_word_embeddings":true,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25}}
    }"#,
    )
    .unwrap();
    for (arch, limit, precision, valid) in [
        ("sm_90a", "1024", None, true),
        ("sm_90a", "256", None, false),
        ("sm_90a", "16384", None, false),
        ("gfx942", "1024", None, false),
        ("sm_90a", "1024", Some("--fp8-kv"), false),
        ("sm_90a", "1024", Some("--fp8"), false),
    ] {
        let mut argv = vec!["test", "--emit-max-request-chunk", limit];
        argv.extend(precision);
        let mut cfg = Args::try_parse_from(argv).unwrap().emit;
        cfg.max_chunk = Some(8192);
        let out = root.join("model.pkt");
        let result = std::panic::catch_unwind(|| {
            devgen::run_verified(
                devgen::EmitArgs {
                    dir: root.clone(),
                    ctx: 20480,
                    out: out.display().to_string(),
                    n_cu: 132,
                    tp: 1,
                    block_spec: None,
                    embed_cubin: None,
                    embed_hsaco: None,
                    rope_gen: true,
                    l2_layout: None,
                    gpu: "h100".into(),
                    arch: arch.into(),
                    emit_cfg: Some(cfg),
                    whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
                },
                Some(Box::new(move |model| {
                    if valid {
                        plow_asset::program::with_model(model, |p| {
                            let live = plow_asset::live_kv::emit(p).unwrap();
                            assert_eq!(
                                live.caches
                                    .iter()
                                    .filter(|c| c.window != 0)
                                    .map(|c| c.stride)
                                    .collect::<Vec<_>>(),
                                [2048]
                            );
                            assert_eq!(
                                p.programs[..p.prefill_count]
                                    .iter()
                                    .map(|g| g.rows)
                                    .collect::<Vec<_>>(),
                                [128, 512, 1024, 2048, 4096, 8192]
                            );
                        });
                    }
                    Ok(devgen::LeanReport::skipped("structural contract test"))
                })),
            )
        });
        assert_eq!(result.is_ok(), valid, "{arch} {limit} {precision:?}");
        if valid {
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("build.json")).unwrap()).unwrap();
            let packed = &manifest["objects"]["packed_prefill"];
            assert_eq!(packed["max_request_rows"], 1024);
            assert_eq!(
                packed["masked_padding_capability"]["symbol"],
                plow_asset::packed_prefill::MASKED_PADDING_CAPABILITY
            );
            assert!(devgen::manifest::config_header(&manifest)
                .contains("#define PLOW_NV_MASKED_PADDING 1"));
            assert_eq!(
                manifest["pairing"]["hash"],
                format!("0x{:016x}", devgen::manifest::pairing_hash(&manifest))
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
