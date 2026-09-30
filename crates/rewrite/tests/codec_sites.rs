//! The fused sites `devgen::codec` / `devgen::s3gen` lower: the SNAC and S3Gen export configs
//! built into whole graphs and rewritten.

use rewrite::{fused_sites_for_codec_config, FusedSites};

const SNAC: &str = r#"{"model_type": "snac", "sampling_rate": 24000, "codebook_size": 4096,
  "vq_strides": [4, 2, 1], "latent_dim": 768,
  "blocks": [
    {"stride": 8, "kernel": 16, "padding": 4, "output_padding": 0, "cin": 1024, "cout": 512, "dilations": [1, 3, 9]},
    {"stride": 8, "kernel": 16, "padding": 4, "output_padding": 0, "cin": 512, "cout": 256, "dilations": [1, 3, 9]},
    {"stride": 4, "kernel": 8, "padding": 2, "output_padding": 0, "cin": 256, "cout": 128, "dilations": [1, 3, 9]},
    {"stride": 2, "kernel": 4, "padding": 1, "output_padding": 0, "cin": 128, "cout": 64, "dilations": [1, 3, 9]}],
  "frame_codes": 7, "frame_samples": 2048}"#;

const S3GEN: &str = r#"{"model_type": "chatterbox_s3gen", "vocab": 6561,
  "upsample": [[8, 16, 4], [5, 11, 3], [3, 7, 2]], "source_downs": [[30, 15, 7], [6, 3, 1], [1, 1, 0]],
  "source_resblocks": [[7, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
  "resblocks": [[3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]],
    [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
  "n_fft": 16, "harmonics": 9}"#;

fn has(s: &FusedSites, kind: &str, weight: &str) -> bool {
    s.get(kind).is_some_and(|w| w.contains(weight))
}

fn any(s: &FusedSites, kinds: &[&str], weight: &str) -> bool {
    kinds.iter().any(|k| has(s, k, weight))
}

const INPUT_ACT: &[&str] = &[
    "FusedActConv1d",
    "FusedParamActConv1d",
    "FusedActConv1dResidual",
    "FusedParamActConv1dResidual",
    "FusedParamActConv1dAct",
];

#[test]
fn snac_fuses_every_snake_residual_and_the_tanh_head() {
    let s = fused_sites_for_codec_config(SNAC).unwrap();
    let kinds: Vec<&str> = s.keys().map(String::as_str).collect();
    assert_eq!(
        kinds,
        [
            "FusedGatedResidual",
            "FusedParamActConv1d",
            "FusedParamActConv1dAct",
            "FusedParamActConv1dResidual"
        ]
    );
    for i in 0..4 {
        assert!(has(&s, "FusedParamActConv1d", &format!("blk{i}.up.w")));
        // The noise block `x + n * (x·Wn)` is the generic gated residual.
        assert!(has(&s, "FusedGatedResidual", &format!("blk{i}.noise.w")));
        for j in 0..3 {
            assert!(has(&s, "FusedParamActConv1d", &format!("blk{i}.ru{j}.dw.w")));
            assert!(has(&s, "FusedParamActConv1dResidual", &format!("blk{i}.ru{j}.pw.w")));
        }
    }
    assert!(has(&s, "FusedParamActConv1dAct", "out.w"));
    assert!(!any(&s, INPUT_ACT, "pre.pw.w"));
}

#[test]
fn s3gen_fuses_conv_and_projection_epilogues() {
    let s = fused_sites_for_codec_config(S3GEN).unwrap();
    for l in 0..10 {
        for w in ["out", "ff2"] {
            assert!(has(&s, "FusedLinearBiasResidual", &format!("enc.L{l}.{w}.w")));
        }
        assert!(has(&s, "FusedLayerNormLinearBiasAct", &format!("enc.L{l}.ff1.w")));
    }
    for k in 0..56 {
        for w in ["out", "ff2"] {
            assert!(has(&s, "FusedLinearBiasResidual", &format!("cfm.tb{k}.{w}.w")));
        }
        assert!(has(&s, "FusedLayerNormLinearBiasAct", &format!("cfm.tb{k}.ff1.w")));
    }
    for j in 0..14 {
        assert!(has(&s, "FusedConv1dResidual", &format!("cfm.r{j}.res.w")));
    }
    assert!(has(&s, "FusedConv1dResidual", "enc.la2.w"));
    for pre in (0..9).map(|i| format!("hift.rb{i}")).chain((0..3).map(|i| format!("hift.sr{i}"))) {
        for d in 0..3 {
            assert!(has(&s, "FusedParamActConv1d", &format!("{pre}.d{d}.c1.w")));
            assert!(has(&s, "FusedParamActConv1dResidual", &format!("{pre}.d{d}.c2.w")));
        }
    }
    assert!(has(&s, "FusedActConv1dResidual", "hift.up1.w"));
    assert!(has(&s, "FusedActConv1d", "hift.up2.w"));
    assert!(has(&s, "FusedActConv1d", "hift.post.w"));
    assert!(has(&s, "FusedConv1dAct", "hift.f0.c4.w"));
    assert!(has(&s, "FusedLinearBiasAct", "hift.f0.cls.w"));
    assert!(has(&s, "FusedLinearBiasAct", "hift.src.w"));
    // An activation between two convs is fused into exactly one of them.
    for (producer, consumer) in [("enc.la1.w", "enc.la2.w"), ("hift.pre.w", "hift.up0.w")]
        .into_iter()
        .chain((0..4).map(|i| (["hift.f0.c0.w", "hift.f0.c1.w", "hift.f0.c2.w", "hift.f0.c3.w"][i], ["hift.f0.c1.w", "hift.f0.c2.w", "hift.f0.c3.w", "hift.f0.c4.w"][i])))
    {
        let at_producer = has(&s, "FusedConv1dAct", producer);
        let at_consumer = any(&s, INPUT_ACT, consumer);
        assert!(at_producer != at_consumer, "{producer} -> {consumer}");
    }
}
