use clap::Parser;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

#[test]
fn emitted_bf16_fp8_and_mixed_kv_contracts_preserve_both_backend_ladders() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("live_kv_fp8");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{
        "model_type":"gemma4_text", "hidden_size":5376, "intermediate_size":21504,
        "num_hidden_layers":2, "num_attention_heads":32, "head_dim":256,
        "global_head_dim":512, "num_key_value_heads":16, "num_global_key_value_heads":4,
        "attention_k_eq_v":true, "sliding_window":1024, "rms_norm_eps":1e-6,
        "vocab_size":262144, "final_logit_softcapping":30.0, "tie_word_embeddings":true,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25}}
    }"#,
    )
    .unwrap();
    for arch in ["gfx942", "sm_90a"] {
        for (fp8, full_only) in [(false, false), (true, false), (true, true)] {
            let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
            cfg.fp8_kv = fp8;
            cfg.fp8_kv_full = full_only;
            cfg.max_chunk = Some(1024);
            cfg.emit_packed_prefill = Some(false);
            devgen::run_verified(
                devgen::EmitArgs {
                    dir: root.clone(),
                    ctx: 2048,
                    out: root.join("model.pkt").display().to_string(),
                    n_cu: if arch == "gfx942" { 304 } else { 132 },
                    tp: 1,
                    block_spec: None,
                    embed_cubin: None,
                    embed_hsaco: None,
                    rope_gen: true,
                    l2_layout: None,
                    gpu: String::new(),
                    arch: arch.into(),
                    emit_cfg: Some(cfg),
                    whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
                },
                Some(Box::new(move |m| {
                    plow_asset::program::with_model(m, |p| {
                        let live = plow_asset::live_kv::emit(p).unwrap_or_else(|e| {
                            panic!("{arch} fp8={fp8} full_only={full_only}: {e}")
                        });
                        assert_eq!(live.version, if fp8 { 2 } else { 1 });
                        assert_eq!(live.caches.len(), 2);
                        assert_eq!(
                            live.caches.iter().filter(|c| c.scales.is_some()).count(),
                            if !fp8 {
                                0
                            } else if full_only {
                                1
                            } else {
                                2
                            }
                        );
                        assert_eq!(
                            p.programs[..p.prefill_count]
                                .iter()
                                .map(|p| p.rows)
                                .collect::<Vec<_>>(),
                            [128, 512, 1024]
                        );
                        assert_eq!(
                            p.programs[p.prefill_count..]
                                .iter()
                                .map(|p| p.rows)
                                .collect::<Vec<_>>(),
                            if arch == "gfx942" {
                                vec![1, 2, 4, 8]
                            } else {
                                vec![1, 2, 4, 8, 16]
                            }
                        );
                        let encoded = serde_json::to_string(&live).unwrap();
                        assert_eq!(encoded.contains("\"scales\""), fp8);
                        let roundtrip: plow_asset::live_kv::Manifest =
                            serde_json::from_str(&encoded).unwrap();
                        assert_eq!(roundtrip, live);
                        roundtrip.validate(p).unwrap();
                        if fp8 {
                            let mut bad = live.clone();
                            bad.version = 1;
                            assert!(bad.validate(p).is_err());
                            for ci in 0..live.caches.len() {
                                if live.caches[ci].scales.is_some() {
                                    let mut bad = live.clone();
                                    bad.caches[ci].scales.as_mut().unwrap().swap(0, 1);
                                    assert!(bad.validate(p).is_err());
                                }
                            }
                        }
                    });
                    Ok(devgen::LeanReport {
                        verified: false,
                        oracle: false,
                        reason: Some("structural contract test".into()),
                    })
                })),
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
