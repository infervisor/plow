//! Qwen3-ASR composite & sub-network graph definitions:
//! Audio front-end (three strided 2D convolutions) + Audio Transformer Encoder +
//! Projection + Qwen3 Thinker Text Decoder.

use super::config::{parse_dtype, Qwen3AsrConfig, Qwen3AudioConfig, Qwen3Config};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

/// Build the standalone audio encoder graph:
/// Padded log-mel `[B, 1, mel_bins, 8*T_sub]` -> `[B, T_sub, output_dim]`.
/// Chunk packing and attention-window partitioning are outside this graph;
/// sinusoidal position values are inputs, not checkpoint weights.
pub fn build_audio_encoder(cfg: &Qwen3AsrConfig) -> Graph {
    let acfg = &cfg.audio_config;
    let dt = parse_dtype(acfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let b = nn.sym("B");
    let t_sub = nn.sym("T_sub");
    let t_mel = t_sub.mul(&Dim::stat(8));
    let mel_bins = acfg.num_mel_bins;

    let mel = nn.input(
        "audio_features",
        nn.shape([b.clone(), Dim::stat(1), Dim::stat(mel_bins), t_mel]),
        dt,
    );

    let out = audio_encoder_core(&mut nn, acfg, mel, &b, &t_sub);
    nn.mark_output(out);
    nn.finish()
}

/// Build the standalone Qwen3 text thinker decoder.
pub fn build_text_decoder(cfg: &Qwen3AsrConfig) -> Graph {
    let mut g = super::qwen3::build(&cfg.text_config);
    for t in &mut g.tensors {
        if matches!(t.origin, crate::Origin::Weight) {
            if let Some(name) = &mut t.name {
                *name = format!("thinker.{name}");
            }
        }
    }
    for block in &mut g.blocks {
        *block = format!("thinker.{block}");
    }
    g
}

/// Build the end-to-end composite graph:
/// Audio encoder features + text prompt token embeddings -> Thinker decoder -> Logits.
pub fn build_composite(cfg: &Qwen3AsrConfig) -> Graph {
    let acfg = &cfg.audio_config;
    let tcfg = &cfg.text_config;
    let dt = parse_dtype(acfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let b = nn.sym("B");
    let t_sub = nn.sym("T_sub");
    let t_mel = t_sub.mul(&Dim::stat(8));
    let s_text = nn.sym("S");

    let mel_bins = acfg.num_mel_bins;
    let h_text = tcfg.hidden_size;
    let eps = tcfg.rms_norm_eps;

    // --- Audio encoder tower ---
    let mel = nn.input(
        "audio_features",
        nn.shape([b.clone(), Dim::stat(1), Dim::stat(mel_bins), t_mel]),
        dt,
    );
    let projected_audio = audio_encoder_core(&mut nn, acfg, mel, &b, &t_sub);

    // --- Text decoder input ---
    let ids = nn.input(
        "input_ids",
        nn.shape([b.clone(), s_text.clone()]),
        DType::I32,
    );
    let text_embed = nn.embedding("thinker.model.embed_tokens", ids, tcfg.vocab_size, h_text);

    // --- Interleave: concat audio representations with text prompt tokens ---
    let combined = nn.concat(1, vec![projected_audio, text_embed]);
    let s_combined = t_sub.add(&s_text);

    // --- Qwen3 Thinker Decoder Layers ---
    let mut x = combined;
    for layer in 0..tcfg.num_hidden_layers {
        let p = format!("thinker.model.layers.{layer}");
        nn.begin_block(&p);

        // Self-Attention
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), x, h_text, eps);
        let attn = text_attention(&mut nn, tcfg, &p, normed, &b, &s_combined);
        x = nn.add(residual, attn);

        // SwiGLU MLP
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), x, h_text, eps);
        let mlp = swiglu_mlp(&mut nn, tcfg, &p, normed, h_text);
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    x = nn.rmsnorm("thinker.model.norm", x, h_text, eps);

    // Tied or explicit LM head
    let head_name = if tcfg.tie_word_embeddings {
        "thinker.model.embed_tokens"
    } else {
        "thinker.lm_head"
    };
    let logits = nn.linear(head_name, x, h_text, tcfg.vocab_size, false);
    nn.mark_output(logits);
    nn.finish()
}

pub fn build(cfg: &Qwen3AsrConfig) -> Graph {
    build_composite(cfg)
}

/// Core audio front-end & transformer encoder.
fn audio_encoder_core(
    nn: &mut Nn,
    cfg: &Qwen3AudioConfig,
    mel: TensorId,
    b: &Dim,
    t_sub: &Dim,
) -> TensorId {
    let d = cfg.d_model;
    let nh = cfg.encoder_attention_heads;
    let head_dim = cfg.head_dim();
    let ffn_dim = cfg.encoder_ffn_dim;
    let eps = 1e-5f32;
    let ch = cfg.downsample_hidden_size;

    // Conv2d front-end: 2x downsampling stage 1
    let conv1 = nn.conv2d(
        "thinker.audio_tower.conv2d1",
        mel,
        1,
        ch,
        (3, 3),
        (2, 2),
        (1, 1),
        true,
    );
    let conv1 = nn.act(ActKind::Gelu, conv1);

    // Conv2d front-end: 2x downsampling stage 2
    let conv2 = nn.conv2d(
        "thinker.audio_tower.conv2d2",
        conv1,
        ch,
        ch,
        (3, 3),
        (2, 2),
        (1, 1),
        true,
    );
    let conv2 = nn.act(ActKind::Gelu, conv2);

    let conv3 = nn.conv2d(
        "thinker.audio_tower.conv2d3",
        conv2,
        ch,
        ch,
        (3, 3),
        (2, 2),
        (1, 1),
        true,
    );
    let conv3 = nn.act(ActKind::Gelu, conv3);
    let flat_dim = ch * ((cfg.num_mel_bins + 7) / 8);
    let time_first = nn.transpose(conv3, vec![0, 3, 1, 2]);
    let reshaped = nn.reshape(time_first, [b.clone(), t_sub.clone(), Dim::stat(flat_dim)]);
    let mut x = nn.linear("thinker.audio_tower.conv_out", reshaped, flat_dim, d, false);

    let pos = nn.input(
        "audio_position",
        nn.shape([t_sub.clone(), Dim::stat(d)]),
        parse_dtype(cfg.torch_dtype.as_deref()),
    );
    x = nn.add(x, pos);

    // Audio Transformer Encoder Layers
    for layer in 0..cfg.encoder_layers {
        let p = format!("thinker.audio_tower.layers.{layer}");
        nn.begin_block(&p);

        // Attention block
        let residual = x;
        let normed = nn.layernorm(&format!("{p}.self_attn_layer_norm"), x, d, eps);
        let q = nn.linear(&format!("{p}.self_attn.q_proj"), normed, d, d, true);
        let k = nn.linear(&format!("{p}.self_attn.k_proj"), normed, d, d, true);
        let v = nn.linear(&format!("{p}.self_attn.v_proj"), normed, d, d, true);

        let q = nn.reshape(
            q,
            [
                b.clone(),
                t_sub.clone(),
                Dim::stat(i64::from(nh)),
                Dim::stat(head_dim),
            ],
        );
        let k = nn.reshape(
            k,
            [
                b.clone(),
                t_sub.clone(),
                Dim::stat(i64::from(nh)),
                Dim::stat(head_dim),
            ],
        );
        let v = nn.reshape(
            v,
            [
                b.clone(),
                t_sub.clone(),
                Dim::stat(i64::from(nh)),
                Dim::stat(head_dim),
            ],
        );

        let attn = nn.attention(q, k, v, nh, nh, head_dim as u32, false, None, None);
        let attn = nn.reshape(attn, [b.clone(), t_sub.clone(), Dim::stat(d)]);
        let out = nn.linear(&format!("{p}.self_attn.out_proj"), attn, d, d, true);
        x = nn.add(residual, out);

        // FFN block
        let residual = x;
        let normed = nn.layernorm(&format!("{p}.final_layer_norm"), x, d, eps);
        let h = nn.linear(&format!("{p}.fc1"), normed, d, ffn_dim, true);
        let h = nn.act(ActKind::Gelu, h);
        let h = nn.linear(&format!("{p}.fc2"), h, ffn_dim, d, true);
        x = nn.add(residual, h);
    }
    nn.end_block();

    let x = nn.layernorm("thinker.audio_tower.ln_post", x, d, eps);
    let x = nn.linear("thinker.audio_tower.proj1", x, d, d, true);
    let x = nn.act(ActKind::Gelu, x);
    nn.linear("thinker.audio_tower.proj2", x, d, cfg.output_dim, true)
}

fn text_attention(
    nn: &mut Nn,
    cfg: &Qwen3Config,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim();

    let q = nn.linear(
        &format!("{p}.self_attn.q_proj"),
        x,
        h,
        nh as i64 * head_dim,
        cfg.qkv_bias,
    );
    let k = nn.linear(
        &format!("{p}.self_attn.k_proj"),
        x,
        h,
        nkv as i64 * head_dim,
        cfg.qkv_bias,
    );
    let v = nn.linear(
        &format!("{p}.self_attn.v_proj"),
        x,
        h,
        nkv as i64 * head_dim,
        cfg.qkv_bias,
    );

    let q = nn.reshape(
        q,
        [
            b.clone(),
            s.clone(),
            Dim::stat(i64::from(nh)),
            Dim::stat(head_dim),
        ],
    );
    let k = nn.reshape(
        k,
        [
            b.clone(),
            s.clone(),
            Dim::stat(i64::from(nkv)),
            Dim::stat(head_dim),
        ],
    );
    let v = nn.reshape(
        v,
        [
            b.clone(),
            s.clone(),
            Dim::stat(i64::from(nkv)),
            Dim::stat(head_dim),
        ],
    );

    let q = if cfg.use_qk_norm {
        nn.rmsnorm(
            &format!("{p}.self_attn.q_norm"),
            q,
            head_dim,
            cfg.rms_norm_eps,
        )
    } else {
        q
    };
    let k = if cfg.use_qk_norm {
        nn.rmsnorm(
            &format!("{p}.self_attn.k_norm"),
            k,
            head_dim,
            cfg.rms_norm_eps,
        )
    } else {
        k
    };

    let q = nn.rope(q, head_dim as u32, cfg.rope_theta);
    let k = nn.rope(k, head_dim as u32, cfg.rope_theta);

    let attn = nn.attention(q, k, v, nh, nkv, head_dim as u32, true, None, None);
    let width = i64::from(nh) * head_dim;
    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(width)]);
    nn.linear(&format!("{p}.self_attn.o_proj"), merged, width, h, false)
}

fn swiglu_mlp(nn: &mut Nn, cfg: &Qwen3Config, p: &str, x: TensorId, h: i64) -> TensorId {
    let gate = nn.linear(
        &format!("{p}.mlp.gate_proj"),
        x,
        h,
        cfg.intermediate_size,
        false,
    );
    let up = nn.linear(
        &format!("{p}.mlp.up_proj"),
        x,
        h,
        cfg.intermediate_size,
        false,
    );
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(
        &format!("{p}.mlp.down_proj"),
        hidden,
        cfg.intermediate_size,
        h,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer_shapes;

    #[test]
    fn audio_encoder_matches_checkpoint_weights_and_layout() {
        use crate::{Bindings, Origin};
        use std::collections::BTreeMap;

        for mel_bins in [128, 129] {
            let mut cfg = Qwen3AsrConfig::default();
            cfg.audio_config.encoder_layers = 2;
            cfg.audio_config.num_mel_bins = mel_bins;
            let a = &cfg.audio_config;
            let (d, ch, ffn) = (a.d_model, a.downsample_hidden_size, a.encoder_ffn_dim);
            let mut expected = BTreeMap::new();
            let mut weight = |name: &str, shape: &[i64]| {
                expected.insert(format!("thinker.audio_tower.{name}"), shape.to_vec());
            };
            for i in 1..=3 {
                weight(
                    &format!("conv2d{i}.weight"),
                    &[ch, if i == 1 { 1 } else { ch }, 3, 3],
                );
                weight(&format!("conv2d{i}.bias"), &[ch]);
            }
            weight("conv_out.weight", &[d, ch * ((mel_bins + 7) / 8)]);
            for layer in 0..a.encoder_layers {
                for norm in ["self_attn_layer_norm", "final_layer_norm"] {
                    weight(&format!("layers.{layer}.{norm}.weight"), &[d]);
                    weight(&format!("layers.{layer}.{norm}.bias"), &[d]);
                }
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    weight(&format!("layers.{layer}.self_attn.{proj}.weight"), &[d, d]);
                    weight(&format!("layers.{layer}.self_attn.{proj}.bias"), &[d]);
                }
                weight(&format!("layers.{layer}.fc1.weight"), &[ffn, d]);
                weight(&format!("layers.{layer}.fc1.bias"), &[ffn]);
                weight(&format!("layers.{layer}.fc2.weight"), &[d, ffn]);
                weight(&format!("layers.{layer}.fc2.bias"), &[d]);
            }
            weight("ln_post.weight", &[d]);
            weight("ln_post.bias", &[d]);
            weight("proj1.weight", &[d, d]);
            weight("proj1.bias", &[d]);
            weight("proj2.weight", &[a.output_dim, d]);
            weight("proj2.bias", &[a.output_dim]);

            let mut g = build_audio_encoder(&cfg);
            infer_shapes(&mut g).unwrap();
            let actual: BTreeMap<_, _> = g
                .tensors
                .iter()
                .filter(|t| matches!(t.origin, Origin::Weight))
                .map(|t| {
                    (
                        t.name.clone().unwrap(),
                        t.shape
                            .as_ref()
                            .unwrap()
                            .dims()
                            .iter()
                            .map(|d| d.as_static().expect("checkpoint weight is static"))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            assert_eq!(actual, expected);
            g.bind(&Bindings::new().set("B", 2).set("T_sub", 13));
            assert_eq!(
                g.tensor(g.inputs[0]).shape.as_ref().unwrap().to_string(),
                format!("[2, 1, {mel_bins}, 104]")
            );
            assert_eq!(
                g.tensor(g.outputs[0]).shape.as_ref().unwrap().to_string(),
                format!("[2, 13, {}]", a.output_dim)
            );
            let transpose = g
                .nodes
                .iter()
                .find(|n| matches!(n.op, crate::Op::Transpose { .. }))
                .unwrap();
            assert!(
                matches!(&transpose.op, crate::Op::Transpose { perm } if perm == &[0, 3, 1, 2])
            );
            assert_eq!(
                g.tensor(transpose.output)
                    .shape
                    .as_ref()
                    .unwrap()
                    .to_string(),
                format!("[2, 13, {ch}, {}]", (mel_bins + 7) / 8)
            );
            let proj2 = &g.nodes[g.nodes.len() - 1];
            let gelu = &g.nodes[g.nodes.len() - 2];
            assert_eq!(proj2.inputs[0], gelu.output);
            assert!(matches!(gelu.op, crate::Op::Act(ActKind::Gelu)));
        }
    }

    #[test]
    fn test_qwen3_asr_audio_encoder() {
        let mut cfg = Qwen3AsrConfig::default();
        cfg.audio_config.encoder_layers = 2;
        let mut g = build_audio_encoder(&cfg);
        infer_shapes(&mut g).expect("shape inference must succeed");
        assert!(!g.tensors.is_empty());
        assert!(!g.nodes.is_empty());
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn test_qwen3_asr_composite_pipeline() {
        let mut cfg = Qwen3AsrConfig::default();
        cfg.audio_config.encoder_layers = 2;
        cfg.text_config.num_hidden_layers = 2;
        let mut g = build_composite(&cfg);
        infer_shapes(&mut g).expect("shape inference must succeed");
        assert!(!g.tensors.is_empty());
        assert!(!g.nodes.is_empty());
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn test_qwen3_asr_from_json() {
        let json_str = r#"{
            "model_type": "qwen3_asr",
            "thinker_config": {
                "audio_config": {
                    "d_model": 512,
                    "encoder_layers": 2,
                    "encoder_attention_heads": 8,
                    "encoder_ffn_dim": 2048,
                    "output_dim": 1024
                },
                "text_config": {
                    "model_type": "qwen3",
                    "hidden_size": 1024,
                    "intermediate_size": 2816,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "vocab_size": 151936
                }
            }
        }"#;

        let model_cfg = crate::models::ModelConfig::from_json(json_str).expect("parse json");
        assert!(matches!(model_cfg, crate::models::ModelConfig::Qwen3Asr(_)));
        let g = crate::models::build_graph(&model_cfg, &crate::models::ShapeBucket::default())
            .expect("build graph");
        assert!(!g.nodes.is_empty());
    }
}
