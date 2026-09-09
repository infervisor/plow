use plow_asset::hetero::HeteroPlan;
use plow_asset::hetero_channel::{
    ChannelPlan, Layer, Mode, PartialDtype, Partials, ProgPlan, Slice, Span, Weight, SCHEMA,
};

pub fn program(prog: u32, rows: u32, hidden: u32, spans: Vec<Span>) -> ProgPlan {
    ProgPlan {
        prog,
        rows,
        min_rows: 64,
        max_rows: 128,
        call_rows: 128,
        original_sha256: String::new(),
        partials: Partials {
            gpu: format!("channel.{prog}.gpu_partial"),
            ane: format!("channel.{prog}.ane_partial"),
            rows,
            cols: hidden,
            dtype: PartialDtype::F32,
        },
        spans,
    }
}

pub fn plan(
    model: &packet::devbuild::Model,
    channels: u32,
    weights: HeteroPlan,
    programs: Vec<ProgPlan>,
) -> Result<ChannelPlan, String> {
    let (h, i) = (weights.hidden, weights.inter);
    let weight = |tensor, scale, down| {
        let (rows, cols, ane, gpu) = if down {
            (
                h,
                i,
                Slice {
                    rows: [0, h],
                    cols: [0, channels],
                },
                Slice {
                    rows: [0, h],
                    cols: [channels, i],
                },
            )
        } else {
            (
                i,
                h,
                Slice {
                    rows: [0, channels],
                    cols: [0, h],
                },
                Slice {
                    rows: [channels, i],
                    cols: [0, h],
                },
            )
        };
        Weight {
            tensor,
            scale,
            rows,
            cols,
            gpu,
            ane,
        }
    };
    let mut plan = ChannelPlan {
        schema: SCHEMA.into(),
        mode: Mode::ChannelMlp,
        arch: "metal3".into(),
        hidden: h,
        inter: i,
        ane_channels: channels,
        weight_encoding: weights.weight_encoding,
        layers: weights
            .layers
            .into_iter()
            .enumerate()
            .map(|(l, w)| Layer {
                layer: l as u32,
                gate: weight(w.wg, w.sg, false),
                up: weight(w.wu, w.su, false),
                down: weight(w.wd, w.sd, true),
            })
            .collect(),
        programs,
    };
    plow_asset::program::with_model(model, |packet| {
        for pp in &mut plan.programs {
            pp.original_sha256 =
                plow_asset::live_kv::program_digest(&packet.programs[pp.prog as usize]);
        }
        plan.validate(packet)
    })?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use crate::{emit_config::EmitConfig, EmitArgs, WholeGraphFusionDecisions};
    use plow_asset::hetero::{Plan, WeightEncoding};

    #[test]
    fn channel_plans_preserve_unsplit_packets_for_all_encodings() {
        let _env = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_MAX_CHUNK", "256"),
            ("PLOW_ROW_SPLIT", ""),
            ("PLOW_FP8", "0"),
            ("PLOW_MXFP4", "0"),
            ("PLOW_W8A8", "0"),
        ]);
        let dir = std::env::temp_dir().join(format!("plow-channel-emit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (hidden, inter) in [(512, 1024), (3072, 8192)] {
            std::fs::write(
                dir.join("config.json"),
                serde_json::to_vec(&serde_json::json!({
                    "model_type":"qwen3","hidden_size":hidden,"intermediate_size":inter,
                    "num_hidden_layers":2,"num_attention_heads":hidden / 128,"head_dim":128,
                    "num_key_value_heads":hidden / 512,"rms_norm_eps":1e-6,"vocab_size":4096,
                    "rope_theta":1000000.0,"tie_word_embeddings":true
                }))
                .unwrap(),
            )
            .unwrap();
            for encoding in [
                WeightEncoding::Bf16,
                WeightEncoding::Fp8,
                WeightEncoding::Mxfp4,
            ] {
                let root = dir.join(format!("{hidden}-{encoding:?}"));
                let emit = |label: &str, channels: Option<u32>| {
                    let outdir = root.join(label);
                    std::fs::create_dir_all(&outdir).unwrap();
                    let mut cfg = EmitConfig::from_env();
                    cfg.fp8 = encoding == WeightEncoding::Fp8;
                    cfg.mxfp4 = encoding == WeightEncoding::Mxfp4;
                    cfg.ane_mlp_channels = channels;
                    cfg.no_glu_fuse = hidden == 512;
                    crate::run(EmitArgs {
                        dir: dir.clone(),
                        ctx: 512,
                        out: outdir.join("model.pkt").to_str().unwrap().into(),
                        n_cu: 16,
                        tp: 1,
                        block_spec: None,
                        embed_cubin: None,
                        embed_hsaco: None,
                        rope_gen: true,
                        l2_layout: None,
                        gpu: "m4pro".into(),
                        arch: "metal3".into(),
                        emit_cfg: Some(cfg),
                        whole_graph_fusions: WholeGraphFusionDecisions::default(),
                    });
                    outdir
                };
                let baseline = emit("gpu", None);
                let channel = emit("channel", Some(512));
                assert_eq!(
                    std::fs::read(baseline.join("model.pkt")).unwrap(),
                    std::fs::read(channel.join("model.pkt")).unwrap()
                );
                let bytes = std::fs::read(channel.join(plow_asset::hetero::FILE)).unwrap();
                let Plan::Channel(plan) = plow_asset::hetero::parse(&bytes).unwrap() else {
                    panic!("not channel mode")
                };
                assert_eq!(plan.weight_encoding, encoding);
                assert_eq!(plan.layers.len(), 2);
                assert_eq!(plan.programs.len(), 1);
                assert_eq!(plan.programs[0].rows, 128);
                assert_eq!(plan.programs[0].spans.len(), 2);
                for layer in &plan.layers {
                    assert_eq!(layer.gate.ane.rows, [0, 512]);
                    assert_eq!(layer.gate.gpu.rows, [512, inter]);
                    assert_eq!(layer.down.ane.cols, [0, 512]);
                    assert_eq!(layer.down.gpu.cols, [512, inter]);
                }
                assert_eq!(
                    plan,
                    serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap()
                );
            }
        }
    }
}
