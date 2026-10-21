use clap::Parser;
use packet::dev::{DevOp, TENSOR_NONE};

const CONFIG: &str = r#"{
        "model_type":"qwen3", "hidden_size":1024, "intermediate_size":3072,
        "num_hidden_layers":2, "num_attention_heads":16, "num_key_value_heads":8, "head_dim":128,
        "rms_norm_eps":1e-6, "rope_theta":1000000.0, "vocab_size":2048,
        "tie_word_embeddings":true, "hidden_act":"silu", "max_position_embeddings":4096
    }"#;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

/// PLOW_DECODE_HNR_FOLD: no decode rung carries a HeadNormRope; every FlashDecode's t7 names its
/// layer's descriptor, whose position word is `in.pos`.
#[test]
fn qwen3_hnr_fold_drops_the_headnorm_packets() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("qwen3_hnr_fold");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), CONFIG).unwrap();
    for fold in [false, true] {
        let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
        cfg.decode_batch = 32;
        cfg.decode_ladder = Some("1,16,32".into());
        cfg.decode_hnr_fold = fold;
        cfg.emit_packed_prefill = Some(true);
        cfg.packed_prefill_small_hd = true;
        devgen::run_verified(
            devgen::EmitArgs {
                dir: root.clone(),
                ctx: 1024,
                out: root.join("model.pkt").display().to_string(),
                n_cu: 170,
                tp: 1,
                block_spec: None,
                embed_cubin: None,
                embed_hsaco: None,
                rope_gen: true,
                l2_layout: None,
                gpu: "RTX 5090".into(),
                arch: "sm_120a".into(),
                emit_cfg: Some(cfg),
                whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
            },
            Some(Box::new(move |model| {
                let pos = model.tensors.iter().position(|t| t.name == "in.pos").unwrap() as u32;
                let descs: Vec<u32> = (0..2)
                    .filter_map(|l| {
                        let name = format!("in.hnr_fold.desc.{l}");
                        model.tensors.iter().position(|t| t.name == name).map(|i| i as u32)
                    })
                    .collect();
                assert_eq!(descs.len(), if fold { 2 } else { 0 });
                for &d in &descs {
                    let w = model.tensors[d as usize].init.as_deref().unwrap();
                    assert_eq!(w.len(), 44);
                    assert_eq!(w[28..32], pos.to_le_bytes());
                }
                let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
                assert_eq!(&model.prog_t[lo..], &[1, 16, 32]);
                for p in &model.progs[lo..] {
                    let hnr = p.insts.iter().filter(|i| i.op == DevOp::HeadNormRope as u16).count();
                    assert_eq!(hnr == 0, fold);
                    let fd: Vec<u32> = p
                        .insts
                        .iter()
                        .filter(|i| i.op == DevOp::FlashDecode as u16)
                        .map(|i| i.t[7] as u32)
                        .collect();
                    assert_eq!(fd, if fold { descs.clone() } else { vec![TENSOR_NONE; 2] });
                }
                Ok(devgen::LeanReport::skipped("structural contract test"))
            })),
        );
        let config = std::fs::read_to_string(root.join("plow_config.h")).unwrap();
        assert_eq!(config.contains("#define PLOW_NV_HNR_FOLD 1"), fold);
    }
    std::fs::remove_dir_all(root).unwrap();
}
