use clap::Parser;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

/// A request chunk above the window stages the sliding layers instead of growing their ring:
/// window 1024 at request chunk 4096 keeps the 2048-row ring, every bucket wider than a stage
/// writes and attends each sliding cache once per stage, and `PLOW_STAGE_ROWS=0` restores the
/// ring sized for the whole request chunk.
#[test]
fn request_chunks_above_the_window_stage_the_sliding_ring() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("request_chunk_stage");
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
    for (precision, stage, ring) in [
        (None, None, 2048u32),
        (Some("--fp8-kv"), None, 2048),
        (Some("--w8a8"), None, 2048),
        (None, Some(0), 8192),
    ] {
        let mut argv = vec!["test", "--emit-max-request-chunk", "4096"];
        argv.extend(precision);
        let mut cfg = Args::try_parse_from(argv).unwrap().emit;
        cfg.max_chunk = Some(4096);
        cfg.stage_rows = stage;
        cfg.pf_ladder_append = Some("4160,4224".into());
        cfg.emit_packed_prefill = Some(true);
        let out = root.join("model.pkt");
        devgen::run_verified(
            devgen::EmitArgs {
                dir: root.clone(),
                ctx: 16384,
                out: out.display().to_string(),
                n_cu: 132,
                tp: 1,
                block_spec: None,
                embed_cubin: None,
                embed_hsaco: None,
                rope_gen: true,
                l2_layout: None,
                gpu: "h100".into(),
                arch: "sm_90a".into(),
                emit_cfg: Some(cfg),
                whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
            },
            Some(Box::new(move |model| {
                plow_asset::program::with_model(model, |p| {
                    let live = plow_asset::live_kv::emit(p).unwrap();
                    let sliding: Vec<_> = live.caches.iter().filter(|c| c.window != 0).collect();
                    assert_eq!(sliding.iter().map(|c| c.stride).collect::<Vec<_>>(), [ring]);
                    let rows: Vec<u32> =
                        p.programs[..p.prefill_count].iter().map(|g| g.rows).collect();
                    assert_eq!(rows, [128, 512, 1024, 2048, 4096, 4160, 4224]);
                    for g in &p.programs[..p.prefill_count] {
                        let map = plow_asset::packed_prefill::stage_map(g.insts);
                        let stages = map.iter().flatten().map(|&k| k as u32 + 1).max().unwrap_or(1);
                        let want =
                            if stage == Some(0) { 1 } else { g.rows.min(4096).div_ceil(1024) };
                        assert_eq!(stages, want, "rows {}", g.rows);
                    }
                });
                Ok(devgen::LeanReport::skipped("structural contract test"))
            })),
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("build.json")).unwrap()).unwrap();
        let packed = &manifest["objects"]["packed_prefill"];
        assert_eq!(packed["max_request_rows"], 4096);
        if stage == Some(0) {
            assert!(packed.get("stage_rows").is_none_or(|v| v.is_null()));
        } else {
            assert_eq!(packed["stage_rows"], 1024, "{precision:?}");
            assert_eq!(packed["stages"], 4);
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
