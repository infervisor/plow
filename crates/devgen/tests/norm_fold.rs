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

/// PLOW_DECODE_NORM_FOLD: on every decode rung the o_projs and the first layer's down carry the
/// fold (i3 = 8, partials in t5), the GLUs and the second layer's q|k|v read the partials (t7),
/// and the one AddNorm left is the last layer's, which feeds the head.
#[test]
fn qwen3_norm_fold_drops_the_addnorm_packets() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("qwen3_norm_fold");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), CONFIG).unwrap();
    for fold in [false, true] {
        let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
        cfg.decode_batch = 32;
        cfg.decode_ladder = Some("1,16,32".into());
        cfg.decode_norm_fold = fold;
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
                let nfp = model
                    .tensors
                    .iter()
                    .position(|t| t.name == "act.nfp")
                    .map_or(TENSOR_NONE, |i| i as u32);
                assert_eq!(nfp != TENSOR_NONE, fold);
                let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
                assert_eq!(&model.prog_t[lo..], &[1, 16, 32]);
                for p in &model.progs[lo..] {
                    let n = |f: &dyn Fn(&packet::dev::DevInst) -> bool| p.insts.iter().filter(|i| f(i)).count();
                    let op = |i: &packet::dev::DevInst, o: DevOp| i.op == o as u16;
                    let reads = |i: &packet::dev::DevInst| i.t[7] as u32 == nfp && nfp != TENSOR_NONE;
                    assert_eq!(n(&|i| op(i, DevOp::AddNorm)), if fold { 1 } else { 4 });
                    assert_eq!(
                        n(&|i| op(i, DevOp::Gemv) && i.i[3] == 8 && i.t[5] as u32 == nfp),
                        if fold { 3 } else { 0 }
                    );
                    assert_eq!(n(&|i| op(i, DevOp::GemvGlu) && reads(i)), if fold { 2 } else { 0 });
                    assert_eq!(n(&|i| op(i, DevOp::GemvQkv) && reads(i)), if fold { 1 } else { 0 });
                }
                Ok(devgen::LeanReport::skipped("structural contract test"))
            })),
        );
        let config = std::fs::read_to_string(root.join("plow_config.h")).unwrap();
        assert_eq!(config.contains("#define PLOW_NV_NORM_FOLD 1"), fold);
    }
    std::fs::remove_dir_all(root).unwrap();
}
