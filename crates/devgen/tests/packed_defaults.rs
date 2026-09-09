use clap::Parser;
use std::collections::BTreeSet;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

#[test]
fn packed_defaults_preserve_backend_contracts_and_ladders() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("packed_defaults");
    let _ = std::fs::remove_dir_all(&root);
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
        let precisions: &[&[&str]] = if arch == "gfx942" {
            &[&[], &["--w8a8"], &["--fp8-kv"]]
        } else {
            &[
                &[],
                &["--fp8"],
                &["--w8a16"],
                &["--fp8-kv"],
                &["--fp8-kv", "--fp8-kv-full"],
                &["--fp8", "--fp8-kv"],
            ]
        };
        for precision in precisions {
            let mut argv = vec!["test"];
            argv.extend_from_slice(precision);
            let mut cfg = Args::try_parse_from(argv).unwrap().emit;
            cfg.max_chunk = Some(1024);
            let mut automatic = None;
            for selection in [None, Some(true), Some(false)] {
                cfg.emit_packed_prefill = selection;
                let out = root.join("model.pkt");
                devgen::run(devgen::EmitArgs {
                    dir: root.clone(),
                    ctx: 2048,
                    out: out.display().to_string(),
                    n_cu: if arch == "gfx942" { 304 } else { 132 },
                    tp: 1,
                    block_spec: None,
                    embed_cubin: None,
                    embed_hsaco: None,
                    rope_gen: true,
                    l2_layout: None,
                    gpu: String::new(),
                    arch: arch.into(),
                    emit_cfg: Some(cfg.clone()),
                    whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
                });
                let blob = std::fs::read(out).unwrap();
                let manifest: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(root.join("build.json")).unwrap())
                        .unwrap();
                let programs = manifest["programs"].as_array().unwrap();
                let decode: BTreeSet<_> = programs
                    .iter()
                    .filter(|p| p["kind"] == "decode")
                    .map(|p| p["batch"].as_u64().unwrap())
                    .collect();
                let prefill: BTreeSet<_> = programs
                    .iter()
                    .filter(|p| p["kind"] == "prefill")
                    .map(|p| p["bucket"].as_u64().unwrap())
                    .collect();
                assert_eq!(
                    decode,
                    if arch == "gfx942" {
                        BTreeSet::from([1, 2, 4, 8])
                    } else {
                        BTreeSet::from([1, 2, 4, 8, 16])
                    }
                );
                assert_eq!(prefill, BTreeSet::from([128, 512, 1024]));
                let request_metadata = arch == "sm_90a"
                    && (selection == Some(true) || (selection.is_none() && !cfg.fp8_kv));
                assert_eq!(
                    blob.windows(b"pf.request.slot".len())
                        .any(|w| w == b"pf.request.slot"),
                    request_metadata,
                    "{arch} {precision:?} {selection:?}"
                );
                if arch == "sm_90a" {
                    assert_eq!(
                        manifest["objects"]["packed_prefill"]["required"],
                        request_metadata
                    );
                    let fp8_cap = &manifest["objects"]["packed_prefill"]["fp8_capability"];
                    if request_metadata && cfg.fp8_kv {
                        assert_eq!(
                            fp8_cap["symbol"],
                            plow_asset::packed_prefill::FP8_CAPABILITY
                        );
                        assert_eq!(
                            fp8_cap["value"],
                            plow_asset::packed_prefill::FP8_CAPABILITY_VALUE
                        );
                    } else {
                        assert!(fp8_cap.is_null());
                    }
                    if selection.is_none() {
                        automatic = Some(blob);
                    } else if selection == Some(true) && !cfg.fp8_kv {
                        assert!(
                            automatic.as_ref().unwrap() == &blob,
                            "automatic vs explicit packet differs: {precision:?}"
                        );
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
