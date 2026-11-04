//! Qwen3-ASR composite & sub-network graph definitions:
//! Audio front-end (2D Conv 4x subsampling) + Audio Transformer Encoder +
//! Projection + Qwen3 Thinker Text Decoder.

use super::config::{parse_dtype, Qwen3AsrConfig, Qwen3AudioConfig, Qwen3Config};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

/// Build the standalone audio encoder graph:
/// `[B, T_frames, 128]` log-mel -> 2D Conv subsampling -> Transformer layers -> `[B, T_sub, output_dim]`
pub fn build_audio_encoder(cfg: &Qwen3AsrConfig) -> Graph {
    let acfg = &cfg.audio_config;
    let dt = parse_dtype(acfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let b = nn.sym("B");
    let t_sub = nn.sym("T_sub");
    let t_mel = t_sub.mul(&Dim::stat(4));
    let mel_bins = acfg.num_mel_bins;
    let d_model = acfg.d_model;
    let out_dim = acfg.output_dim;

    // Audio input: [B, 1, T_mel, mel_bins]
    let mel = nn.input(
        "audio_features",
        nn.shape([b.clone(), Dim::stat(1), t_mel.clone(), Dim::stat(mel_bins)]),
        dt,
    );

    let projected_audio = audio_encoder_core(&mut nn, acfg, mel, &b, &t_sub);
    let out = nn.linear(
        "thinker.audio_tower.proj",
        projected_audio,
        d_model,
        out_dim,
        false,
    );
    nn.mark_output(out);
    nn.finish()
}

/// Build the standalone Qwen3 text thinker decoder.
pub fn build_text_decoder(cfg: &Qwen3AsrConfig) -> Graph {
    super::qwen3::build(&cfg.text_config)
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
    let t_mel = t_sub.mul(&Dim::stat(4));
    let s_text = nn.sym("S");

    let mel_bins = acfg.num_mel_bins;
    let d_model = acfg.d_model;
    let h_text = tcfg.hidden_size;
    let eps = tcfg.rms_norm_eps;

    // --- Audio encoder tower ---
    let mel = nn.input(
        "audio_features",
        nn.shape([b.clone(), Dim::stat(1), t_mel.clone(), Dim::stat(mel_bins)]),
        dt,
    );
    let audio_enc = audio_encoder_core(&mut nn, acfg, mel, &b, &t_sub);
    let projected_audio = nn.linear(
        "thinker.audio_tower.proj",
        audio_enc,
        d_model,
        h_text,
        false,
    );

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
        "thinker.audio_tower.conv1",
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
        "thinker.audio_tower.conv2",
        conv1,
        ch,
        ch,
        (3, 3),
        (2, 2),
        (1, 1),
        true,
    );
    let conv2 = nn.act(ActKind::Gelu, conv2);

    let flat_dim = ch * (cfg.num_mel_bins / 4);
    let reshaped = nn.reshape(conv2, [b.clone(), t_sub.clone(), Dim::stat(flat_dim)]);
    let mut x = nn.linear("thinker.audio_tower.conv_out", reshaped, flat_dim, d, false);

    // Positional embedding addition
    let pos = nn.param(
        "thinker.audio_tower.positional_embedding",
        [t_sub.clone(), Dim::stat(d)],
    );
    x = nn.add(x, pos);

    // Audio Transformer Encoder Layers
    for layer in 0..cfg.encoder_layers {
        let p = format!("thinker.audio_tower.layers.{layer}");
        nn.begin_block(&p);

        // Attention block
        let residual = x;
        let normed = nn.layernorm(&format!("{p}.ln1"), x, d, eps);
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
        let normed = nn.layernorm(&format!("{p}.ln2"), x, d, eps);
        let h = nn.linear(&format!("{p}.mlp.fc1"), normed, d, ffn_dim, true);
        let h = nn.act(ActKind::Gelu, h);
        let h = nn.linear(&format!("{p}.mlp.fc2"), h, ffn_dim, d, true);
        x = nn.add(residual, h);
    }
    nn.end_block();

    x
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
