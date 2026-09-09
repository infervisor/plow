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
                        if arch == "sm_90a" {
                            packed_contract(p, &live);
                        }
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

fn packed_contract(p: &plow_asset::program::Packet<'_>, live: &plow_asset::live_kv::Manifest) {
    use packet::dev::{DevOp, TENSOR_NONE16};
    use plow_asset::packed_prefill::{Manifest, Map, FP8_REQUEST_TAG};
    use plow_asset::program::{Packet, Tensor};

    let mut tensors = p.tensors.to_vec();
    let slot = tensors.len() as u16;
    tensors.push(Tensor {
        name: "pf.request.slot",
        bytes: 1024 * 4,
        initialized: false,
    });
    let request = tensors.len() as u16;
    tensors.push(Tensor {
        name: "pf.request.table",
        bytes: (1 + 4 * u64::from(live.batch)) * 4,
        initialized: false,
    });
    let names: Vec<_> = live
        .maps
        .iter()
        .map(|m| format!("pf.request.maps.{}", m.handle))
        .collect();
    let mut maps = Vec::new();
    for (m, name) in live.maps.iter().zip(&names) {
        maps.push(Map {
            original: m.handle as u16,
            slots: tensors.len() as u16,
        });
        tensors.push(Tensor {
            name,
            bytes: 8 * u64::from(live.batch),
            initialized: false,
        });
    }
    let packet = Packet {
        tensors: &tensors,
        ..*p
    };
    let manifest = Manifest {
        version: live.version,
        slot,
        request,
        maps,
        programs: p.programs[..p.prefill_count]
            .iter()
            .map(plow_asset::live_kv::program_digest)
            .collect(),
    };
    manifest.validate(&packet, live).unwrap();
    let roundtrip: Manifest =
        serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
    assert_eq!(manifest, roundtrip);
    let mut bad = manifest.clone();
    bad.version = if live.version == 1 { 2 } else { 1 };
    assert!(bad.validate(&packet, live).is_err());
    bad = manifest.clone();
    bad.request = bad.slot;
    assert!(bad.validate(&packet, live).is_err());
    for (pi, program) in p.programs[..p.prefill_count].iter().enumerate() {
        let pc = program
            .insts
            .iter()
            .position(|d| {
                matches!(
                    DevOp::from_u16(d.op),
                    Some(DevOp::FlashPrefill | DevOp::FlashPrefillFp8)
                )
            })
            .unwrap();
        for mutation in 0..3 {
            let mut insts = program.insts.to_vec();
            let mut entries = program.gq_stream.to_vec();
            let mut tensors = tensors.clone();
            match mutation {
                0 => insts[pc].i[4] = FP8_REQUEST_TAG | u32::from(request),
                1 => tensors[insts[pc].t[2] as usize].bytes = 1,
                2 => {
                    let entry = entries.iter_mut().find(|e| e.inst as usize == pc).unwrap();
                    entry.slice = u32::from(insts[pc].blocks);
                }
                _ => unreachable!(),
            }
            let mut programs = p.programs.to_vec();
            programs[pi].insts = &insts;
            programs[pi].gq_stream = &entries;
            let changed = Packet {
                programs: &programs,
                tensors: &tensors,
                ..packet
            };
            if let Ok(live) = plow_asset::live_kv::emit(&changed) {
                let mut bad = manifest.clone();
                bad.programs = programs[..p.prefill_count]
                    .iter()
                    .map(plow_asset::live_kv::program_digest)
                    .collect();
                assert!(
                    bad.validate(&changed, &live).is_err(),
                    "bucket {} mutation {mutation}",
                    program.rows
                );
            }
        }
    }
    for program in &p.programs[..p.prefill_count] {
        for baseline in program.insts {
            let mut d = *baseline;
            for _ in 0..2 {
                manifest.bind_request(&mut d, true);
                match DevOp::from_u16(d.op) {
                    Some(DevOp::HeadNormRopeFp8) => {
                        assert_eq!(d.t[6], baseline.t[6]);
                        assert_eq!(d.t[7], slot);
                    }
                    Some(DevOp::FlashPrefillFp8) => {
                        assert_eq!(d.t, baseline.t);
                        assert_eq!(d.i[4], FP8_REQUEST_TAG | u32::from(request));
                    }
                    Some(DevOp::HeadNormRope) => assert_eq!(d.t[6], slot),
                    Some(DevOp::FlashPrefill | DevOp::FlashMerge) => assert_eq!(
                        d.t[if d.op == DevOp::FlashMerge as u16 {
                            7
                        } else {
                            6
                        }],
                        request
                    ),
                    _ => {}
                }
                let bound = d;
                manifest.bind_request(&mut d, true);
                assert_eq!((d.t, d.i, d.fj), (bound.t, bound.i, bound.fj));
                manifest.bind_request(&mut d, false);
                if d.op == DevOp::FlashMerge as u16 {
                    assert_eq!(d.t[7], TENSOR_NONE16);
                }
                assert_eq!(
                    (d.op, d.blocks, d.t, d.i, d.fj),
                    (
                        baseline.op,
                        baseline.blocks,
                        baseline.t,
                        baseline.i,
                        baseline.fj
                    )
                );
            }
        }
    }
}
