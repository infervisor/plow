use clap::Parser;

const CONFIG: &str = r#"{
        "model_type":"gemma4_text", "hidden_size":3840, "intermediate_size":15360,
        "num_hidden_layers":2, "num_attention_heads":16, "head_dim":256,
        "global_head_dim":512, "num_key_value_heads":8, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "sliding_window":1024, "rms_norm_eps":1e-6,
        "vocab_size":262144, "final_logit_softcapping":30.0, "tie_word_embeddings":true,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25}}
    }"#;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

/// Without a request cap one request may fill the 8192 rung, so the sliding ring follows it; a
/// context below 8192 keeps the `PLOW_MAX_CHUNK` ladder and ring.
#[test]
fn uncapped_ladders_ship_8192_and_ring_it_when_ctx_reaches_it() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("uncapped_8192_rung");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), CONFIG).unwrap();
    for (ctx, rows, ring) in [
        (
            16384u32,
            vec![128u32, 512, 1024, 2048, 4096, 8192],
            16384u32,
        ),
        (4096, vec![128, 512, 1024], 2048),
    ] {
        let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
        cfg.max_chunk = Some(1024);
        cfg.emit_packed_prefill = Some(false);
        let out = root.join("model.pkt");
        devgen::run_verified(
            devgen::EmitArgs {
                dir: root.clone(),
                ctx,
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
                    let got: Vec<u32> = p.programs[..p.prefill_count]
                        .iter()
                        .map(|g| g.rows)
                        .collect();
                    assert_eq!(got, rows, "ctx {ctx}");
                });
                Ok(devgen::LeanReport::skipped("structural contract test"))
            })),
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
