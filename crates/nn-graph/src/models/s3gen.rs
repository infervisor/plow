//! Chatterbox S3Gen → symbolic operator graph, channels-last `[B, T, C]`: the three networks
//! `devgen::s3gen` lowers to program kinds, each with its own inputs and output.
//!
//! * Token encoder (`S` tokens incl. the prompt): embedding, pre-lookahead convolutions, six
//!   conformer layers, nearest ×2 upsampling and a causal conv, four more conformer layers and
//!   the projection to mel.
//! * CFM estimator (one Euler step, `T` mel frames): the ConditionalDecoder's ResNet1D blocks
//!   and BasicTransformerBlocks. Its input is the concatenated `[x | mu | spks | cond]` and the
//!   per-block time embeddings, which are inputs (a function of the step only).
//! * HiFT vocoder (`T` mel frames): F0 predictor, the source merge, STFT, transposed-conv
//!   upsampling with snake ResBlocks, and the post conv. The NSF harmonic source (a phase
//!   integral and noise) is an input, as is the iSTFT it would feed: neither has a fusion.
//!
//! Not modeled: the conformer's relative-position score bias (`relpos`/`pos_vu`; an
//! `Attention` bias input would be dropped by the egglog lowering anyway) and sequence masks
//! (per-item lengths are a runtime resource). Tensor names are the export's
//! (`scripts/tts/s3gen_export.py`), so fused sites anchor on the names `devgen::s3gen` loads.

use super::config::S3GenConfig;
use crate::op::{ActKind, Op, PadMode};
use crate::{DType, Dim, Graph, Nn, TensorId};

fn conv(stride: u32, dilation: u32, pad: (u32, u32), pad_mode: PadMode) -> Op {
    Op::Conv1d {
        stride,
        dilation,
        groups: 1,
        padding: pad,
        pad_mode,
    }
}

fn same(k: u32, dilation: u32) -> Op {
    let p = dilation * (k - 1) / 2;
    conv(1, dilation, (p, p), PadMode::Zero)
}

fn causal(k: u32) -> Op {
    conv(1, 1, (k - 1, 0), PadMode::Zero)
}

pub fn build(cfg: &S3GenConfig) -> Graph {
    let mut nn = Nn::new(DType::F32, DType::F32);
    let b = nn.sym("B");
    encoder(&mut nn, cfg, &b);
    estimator(&mut nn, cfg, &b);
    vocoder(&mut nn, cfg, &b);
    nn.finish()
}

fn heads(nn: &mut Nn, x: TensorId, b: &Dim, t: &Dim, heads: u32, width: i64) -> TensorId {
    nn.reshape(x, [b.clone(), t.clone(), Dim::stat(i64::from(heads)), Dim::stat(width / i64::from(heads))])
}

/// ESPnet conformer layer (macaron and conv module off): `x += MHA(LN(x))`, `x += FF(LN(x))`.
fn conformer(nn: &mut Nn, cfg: &S3GenConfig, i: u32, x: TensorId, b: &Dim, t: &Dim) -> TensorId {
    let d = cfg.encoder_dim;
    let p = format!("enc.L{i}");
    nn.begin_block(&p);
    let a = nn.layernorm_named(&format!("{p}.ln_mha.g"), &format!("{p}.ln_mha.b"), x, d, 1e-12);
    // The export folds pos_bias_u into the query bias.
    let q = nn.linear_named(&format!("{p}.q.w"), Some(&format!("{p}.q_u.b")), a, d, d);
    let k = nn.linear_named(&format!("{p}.k.w"), Some(&format!("{p}.k.b")), a, d, d);
    let v = nn.linear_named(&format!("{p}.v.w"), Some(&format!("{p}.v.b")), a, d, d);
    let (q, k, v) = (heads(nn, q, b, t, cfg.heads, d), heads(nn, k, b, t, cfg.heads, d), heads(nn, v, b, t, cfg.heads, d));
    let hd = (d / i64::from(cfg.heads)) as u32;
    let att = nn.attention(q, k, v, cfg.heads, cfg.heads, hd, false, None, None);
    let att = nn.reshape(att, [b.clone(), t.clone(), Dim::stat(d)]);
    let o = nn.linear_named(&format!("{p}.out.w"), Some(&format!("{p}.out.b")), att, d, d);
    let x = nn.add(x, o);
    let a = nn.layernorm_named(&format!("{p}.ln_ff.g"), &format!("{p}.ln_ff.b"), x, d, 1e-12);
    let h = nn.linear_named(&format!("{p}.ff1.w"), Some(&format!("{p}.ff1.b")), a, d, 4 * d);
    let h = nn.act(ActKind::Silu, h);
    let h = nn.linear_named(&format!("{p}.ff2.w"), Some(&format!("{p}.ff2.b")), h, 4 * d, d);
    let x = nn.add(x, h);
    nn.end_block();
    x
}

fn encoder(nn: &mut Nn, cfg: &S3GenConfig, b: &Dim) {
    let d = cfg.encoder_dim;
    let s = nn.sym("S");
    let ids = nn.input("tokens", nn.shape([b.clone(), s.clone()]), DType::I32);
    let e = nn.embedding_named("emb", ids, cfg.vocab, d);
    let e = nn.linear_named("enc.embed.w", Some("enc.embed.b"), e, d, d);
    let e = nn.layernorm_named("enc.embed.ln.g", "enc.embed.ln.b", e, d, 1e-5);
    // Pre-lookahead: leaky(conv k4 looking 3 ahead), causal conv k3, + input.
    let h = nn.conv1d("enc.la1.w", Some("enc.la1.b"), e, d, d, 4, conv(1, 1, (0, 3), PadMode::Zero));
    let h = nn.act(ActKind::LeakyRelu(0.01), h);
    let h = nn.conv1d("enc.la2.w", Some("enc.la2.b"), h, d, d, 3, causal(3));
    let mut x = nn.add(e, h);
    for i in 0..cfg.encoder_layers.0 {
        x = conformer(nn, cfg, i, x, b, &s);
    }
    // Nearest ×2, then a causal k5 conv.
    let t = s.mul(&Dim::stat(2));
    let x2 = nn.reshape(x, [b.clone(), s.clone(), Dim::stat(1), Dim::stat(d)]);
    let x2 = nn.broadcast(x2, [b.clone(), s.clone(), Dim::stat(2), Dim::stat(d)]);
    let x2 = nn.reshape(x2, [b.clone(), t.clone(), Dim::stat(d)]);
    let u = nn.conv1d("enc.up.w", Some("enc.up.b"), x2, d, d, 5, causal(5));
    let u = nn.linear_named("enc.up_embed.w", Some("enc.up_embed.b"), u, d, d);
    let mut x = nn.layernorm_named("enc.up_embed.ln.g", "enc.up_embed.ln.b", u, d, 1e-5);
    for i in 0..cfg.encoder_layers.1 {
        x = conformer(nn, cfg, cfg.encoder_layers.0 + i, x, b, &t);
    }
    let a = nn.layernorm_named("enc.after_ln.g", "enc.after_ln.b", x, d, 1e-5);
    let mu = nn.linear_named("enc.proj.w", Some("enc.proj.b"), a, d, cfg.mel_bins);
    nn.mark_output(mu);
}

/// CausalResnetBlock1D: `mish(LN(conv(x)))`, `+ t`, `mish(LN(conv(.)))`, `+ res(x)`.
fn resnet(nn: &mut Nn, cfg: &S3GenConfig, j: u32, x: TensorId, cin: i64, b: &Dim) -> TensorId {
    let d = cfg.estimator_dim;
    let p = format!("cfm.r{j}");
    nn.begin_block(&p);
    let y = nn.conv1d(&format!("{p}.c1.w"), Some(&format!("{p}.c1.b")), x, cin, d, 3, causal(3));
    let y = nn.layernorm_named(&format!("{p}.ln1.g"), &format!("{p}.ln1.b"), y, d, 1e-5);
    let y = nn.act(ActKind::Mish, y);
    let temb = nn.input(&format!("cfm.t.r{j}"), nn.shape([b.clone(), Dim::stat(1), Dim::stat(d)]), DType::F32);
    let y = nn.add(y, temb);
    let y = nn.conv1d(&format!("{p}.c2.w"), Some(&format!("{p}.c2.b")), y, d, d, 3, causal(3));
    let y = nn.layernorm_named(&format!("{p}.ln2.g"), &format!("{p}.ln2.b"), y, d, 1e-5);
    let y = nn.act(ActKind::Mish, y);
    let r = nn.conv1d(&format!("{p}.res.w"), Some(&format!("{p}.res.b")), x, cin, d, 1, same(1, 1));
    let out = nn.add(y, r);
    nn.end_block();
    out
}

/// BasicTransformerBlock: `h += Attn(LN(h))`, `h += FF_gelu(LN(h))`.
fn tblock(nn: &mut Nn, cfg: &S3GenConfig, k: u32, h: TensorId, b: &Dim, t: &Dim) -> TensorId {
    let d = cfg.estimator_dim;
    let inner = i64::from(cfg.heads) * 64;
    let p = format!("cfm.tb{k}");
    nn.begin_block(&p);
    let a = nn.layernorm_named(&format!("{p}.ln1.g"), &format!("{p}.ln1.b"), h, d, 1e-5);
    let qkv = nn.linear_named(&format!("{p}.qkv.w"), None, a, d, 3 * inner);
    let part = |nn: &mut Nn, n: i64| {
        let s = nn.slice(qkv, 2, n * inner, inner);
        heads(nn, s, b, t, cfg.heads, inner)
    };
    let (q, kk, v) = (part(nn, 0), part(nn, 1), part(nn, 2));
    let att = nn.attention(q, kk, v, cfg.heads, cfg.heads, 64, false, None, None);
    let att = nn.reshape(att, [b.clone(), t.clone(), Dim::stat(inner)]);
    let o = nn.linear_named(&format!("{p}.out.w"), Some(&format!("{p}.out.b")), att, inner, d);
    let h = nn.add(h, o);
    let a = nn.layernorm_named(&format!("{p}.ln3.g"), &format!("{p}.ln3.b"), h, d, 1e-5);
    let f = nn.linear_named(&format!("{p}.ff1.w"), Some(&format!("{p}.ff1.b")), a, d, 4 * d);
    let f = nn.act(ActKind::Gelu, f);
    let f = nn.linear_named(&format!("{p}.ff2.w"), Some(&format!("{p}.ff2.b")), f, 4 * d, d);
    let h = nn.add(h, f);
    nn.end_block();
    h
}

fn estimator(nn: &mut Nn, cfg: &S3GenConfig, b: &Dim) {
    let d = cfg.estimator_dim;
    let t = nn.sym("T");
    let cin = 4 * cfg.mel_bins;
    let x = nn.input("cfm.x", nn.shape([b.clone(), t.clone(), Dim::stat(cin)]), DType::F32);
    let tb = cfg.transformer_blocks;
    let mut h = resnet(nn, cfg, 0, x, cin, b);
    for i in 0..tb {
        h = tblock(nn, cfg, i, h, b, &t);
    }
    let skip = h;
    h = nn.conv1d("cfm.down.w", Some("cfm.down.b"), h, d, d, 3, causal(3));
    for j in 1..=cfg.mid_blocks {
        h = resnet(nn, cfg, j, h, d, b);
        for i in 0..tb {
            h = tblock(nn, cfg, tb * j + i, h, b, &t);
        }
    }
    let cat = nn.concat(2, vec![h, skip]);
    let last = cfg.mid_blocks + 1;
    h = resnet(nn, cfg, last, cat, 2 * d, b);
    for i in 0..tb {
        h = tblock(nn, cfg, tb * last + i, h, b, &t);
    }
    h = nn.conv1d("cfm.upc.w", Some("cfm.upc.b"), h, d, d, 3, causal(3));
    let y = nn.conv1d("cfm.fin.w", Some("cfm.fin.b"), h, d, d, 3, causal(3));
    let y = nn.layernorm_named("cfm.fin.ln.g", "cfm.fin.ln.b", y, d, 1e-5);
    let y = nn.act(ActKind::Mish, y);
    let v = nn.conv1d("cfm.proj.w", Some("cfm.proj.b"), y, d, cfg.mel_bins, 1, same(1, 1));
    nn.mark_output(v);
}

/// HiFT ResBlock: per dilation `x += c2(snake(c1(snake(x))))`.
fn resblock(nn: &mut Nn, pre: &str, mut x: TensorId, k: u32, dils: &[u32], ch: i64) -> TensorId {
    nn.begin_block(pre);
    for (j, &dil) in dils.iter().enumerate() {
        let p = format!("{pre}.d{j}");
        let y = nn.snake(&format!("{p}.a1"), x, ch);
        let y = nn.conv1d(&format!("{p}.c1.w"), Some(&format!("{p}.c1.b")), y, ch, ch, i64::from(k), same(k, dil));
        let y = nn.snake(&format!("{p}.a2"), y, ch);
        let y = nn.conv1d(&format!("{p}.c2.w"), Some(&format!("{p}.c2.b")), y, ch, ch, i64::from(k), same(k, 1));
        x = nn.add(x, y);
    }
    nn.end_block();
    x
}

fn vocoder(nn: &mut Nn, cfg: &S3GenConfig, b: &Dim) {
    let t = nn.sym("T");
    let mel = nn.input("hift.mel", nn.shape([b.clone(), t.clone(), Dim::stat(cfg.mel_bins)]), DType::F32);
    // F0 predictor.
    let fc = cfg.f0_channels;
    let (mut v, mut cin) = (mel, cfg.mel_bins);
    for i in 0..cfg.f0_convs {
        v = nn.conv1d(&format!("hift.f0.c{i}.w"), Some(&format!("hift.f0.c{i}.b")), v, cin, fc, 3, same(3, 1));
        v = nn.act(ActKind::Elu, v);
        cin = fc;
    }
    let f0 = nn.linear_named("hift.f0.cls.w", Some("hift.f0.cls.b"), v, fc, 1);
    let f0 = nn.act(ActKind::Abs, f0);
    nn.mark_output(f0);
    // Source: tanh(Linear(harmonics)), then its STFT (real | imaginary bins).
    let hop = cfg.n_fft / 4;
    let samples: i64 = cfg.upsample.iter().map(|u| i64::from(u[0])).product::<i64>() * i64::from(hop);
    let wav = t.mul(&Dim::stat(samples));
    let har = nn.input("hift.harmonics", nn.shape([b.clone(), wav, Dim::stat(cfg.harmonics)]), DType::F32);
    let src = nn.linear_named("hift.src.w", Some("hift.src.b"), har, cfg.harmonics, 1);
    let src = nn.act(ActKind::Tanh, src);
    let spec = 2 * (i64::from(cfg.n_fft) / 2 + 1);
    let half = cfg.n_fft / 2;
    let stft = nn.conv1d("hift.stft.w", None, src, 1, spec, i64::from(cfg.n_fft), conv(hop, 1, (half, half), PadMode::Reflect));
    // Mel → waveform features.
    let mut c = cfg.hift_channels;
    let mut x = nn.conv1d("hift.pre.w", Some("hift.pre.b"), mel, cfg.mel_bins, c, 7, same(7, 1));
    let per_stage = cfg.resblocks.len() / cfg.upsample.len().max(1);
    for (i, &[u, k, pad]) in cfg.upsample.iter().enumerate() {
        let co = c / 2;
        let last = i + 1 == cfg.upsample.len();
        let xl = nn.act(ActKind::LeakyRelu(0.1), x);
        let up = Op::ConvTranspose1d { stride: u, groups: 1, crop: (pad, pad), output_padding: 0 };
        let mut xu = nn.conv1d(&format!("hift.up{i}.w"), Some(&format!("hift.up{i}.b")), xl, c, co, i64::from(k), up);
        if last {
            // ReflectionPad1d((1, 0)).
            let first = nn.slice(xu, 1, 1, 1);
            xu = nn.concat(1, vec![first, xu]);
        }
        let [sk, ss, sp] = cfg.source_downs[i];
        let sd = nn.conv1d(&format!("hift.sd{i}.w"), Some(&format!("hift.sd{i}.b")), stft, spec, co, i64::from(sk), conv(ss, 1, (sp, sp), PadMode::Zero));
        let (rk, dils) = &cfg.source_resblocks[i];
        let si = resblock(nn, &format!("hift.sr{i}"), sd, *rk, dils, co);
        let xs = nn.add(si, xu);
        let mut acc: Option<TensorId> = None;
        for j in 0..per_stage {
            let n = i * per_stage + j;
            let (rk, dils) = &cfg.resblocks[n];
            let y = resblock(nn, &format!("hift.rb{n}"), xs, *rk, dils, co);
            acc = Some(match acc {
                Some(a) => nn.add(a, y),
                None => y,
            });
        }
        x = nn.scale(acc.expect("HiFT stage has ResBlocks"), 1.0 / per_stage as f32);
        c = co;
    }
    let xl = nn.act(ActKind::LeakyRelu(0.01), x);
    let post = nn.conv1d("hift.post.w", Some("hift.post.b"), xl, c, spec, 7, same(7, 1));
    nn.mark_output(post);
}
