//! Qwen-Image DiT (`QwenImageTransformer2DModel`) → symbolic operator graph.
//!
//! An MMDiT (FLUX-style multimodal diffusion transformer): two streams — image
//! latent patches and text-encoder hidden states — each modulated by an AdaLN
//! conditioning vector (timestep/guidance), interacting through **joint
//! attention** (the two streams' Q/K/V are concatenated along the sequence,
//! attended together, then split back). RoPE is 3D (axes [t, h, w]).
//!
//! The image grid is compiled per shape bucket (static token count `N`); the
//! text length `L` and batch `B` stay symbolic.

use super::config::{parse_dtype, QwenImageDitConfig};
use super::ShapeBucket;
use crate::op::ActKind;
use crate::Nn;
use crate::{Dim, Graph, TensorId};

/// Qwen-Image VAE spatial downsample factor (image → latent).
const VAE_SCALE_FACTOR: i64 = 8;

pub fn build(cfg: &QwenImageDitConfig, bucket: &ShapeBucket) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let hidden = cfg.hidden_size();
    let heads = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let mlp_dim = hidden * 4;

    // Image-token count derives from the bucket resolution: image → latent
    // (÷ VAE scale) → patches (÷ patch_size), per spatial axis. Text tokens
    // stay symbolic.
    let lat_h = bucket.image_height / VAE_SCALE_FACTOR / cfg.patch_size;
    let lat_w = bucket.image_width / VAE_SCALE_FACTOR / cfg.patch_size;
    let n_img = lat_h * lat_w;
    let b = nn.sym("B");
    let l = nn.sym("L");
    let n = Dim::stat(n_img);

    let img_in = nn.input(
        "hidden_states",
        nn.shape([b.clone(), n.clone(), Dim::stat(cfg.in_channels)]),
        dt,
    );
    let txt_in = nn.input(
        "encoder_hidden_states",
        nn.shape([b.clone(), l.clone(), Dim::stat(cfg.joint_attention_dim)]),
        dt,
    );
    // Pooled timestep/guidance conditioning vector.
    let temb = nn.input("temb", nn.shape([b.clone(), Dim::stat(hidden)]), dt);

    // Stream embeddings.
    let mut img = nn.linear("img_in", img_in, cfg.in_channels, hidden, true);
    let txt = nn.rmsnorm("txt_norm", txt_in, cfg.joint_attention_dim, 1e-6);
    let mut txt = nn.linear("txt_in", txt, cfg.joint_attention_dim, hidden, true);

    for layer in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{layer}");
        nn.begin_block(&p);
        let (ni, nx) = block(
            &mut nn, cfg, &p, img, txt, temb, &b, &l, &n, hidden, heads, hd, mlp_dim,
        );
        img = ni;
        txt = nx;
    }
    nn.end_block();

    // Final AdaLN + projection to patch outputs (patch² * out_channels).
    let smod = nn.act(ActKind::Silu, temb);
    let params = nn.linear("norm_out.linear", smod, hidden, 2 * hidden, true);
    let parts = chunk_mod(&mut nn, params, 2, hidden, &b);
    let img_n = nn.layernorm("norm_out.norm", img, hidden, 1e-6);
    let img = modulate(&mut nn, img_n, parts[1], parts[0]);
    let patch_out = cfg.patch_size * cfg.patch_size * cfg.out_channels;
    let out = nn.linear("proj_out", img, hidden, patch_out, true);

    nn.mark_output(out);
    nn.finish()
}

#[allow(clippy::too_many_arguments)]
fn block(
    nn: &mut Nn,
    cfg: &QwenImageDitConfig,
    p: &str,
    img: TensorId,
    txt: TensorId,
    temb: TensorId,
    b: &Dim,
    l: &Dim,
    n: &Dim,
    hidden: i64,
    heads: u32,
    hd: i64,
    mlp_dim: i64,
) -> (TensorId, TensorId) {
    // Per-stream modulation params: 6 chunks each (shift/scale/gate ×2).
    let smod = nn.act(ActKind::Silu, temb);
    let img_mod = nn.linear(&format!("{p}.img_mod.1"), smod, hidden, 6 * hidden, true);
    let txt_mod = nn.linear(&format!("{p}.txt_mod.1"), smod, hidden, 6 * hidden, true);
    let im = chunk_mod(nn, img_mod, 6, hidden, b);
    let tm = chunk_mod(nn, txt_mod, 6, hidden, b);

    // ---- joint attention ----
    let img_n = nn.layernorm(&format!("{p}.img_norm1"), img, hidden, 1e-6);
    let img_a = modulate(nn, img_n, im[1], im[0]);
    let txt_n = nn.layernorm(&format!("{p}.txt_norm1"), txt, hidden, 1e-6);
    let txt_a = modulate(nn, txt_n, tm[1], tm[0]);

    let (img_attn, txt_attn) =
        joint_attention(nn, cfg, p, img_a, txt_a, b, l, n, hidden, heads, hd);

    let img_attn = nn.linear(&format!("{p}.attn.to_out"), img_attn, hidden, hidden, true);
    let txt_attn = nn.linear(
        &format!("{p}.attn.to_add_out"),
        txt_attn,
        hidden,
        hidden,
        true,
    );
    let img = gate_residual(nn, img, img_attn, im[2]);
    let txt = gate_residual(nn, txt, txt_attn, tm[2]);

    // ---- per-stream MLP ----
    let img_n2 = nn.layernorm(&format!("{p}.img_norm2"), img, hidden, 1e-6);
    let img_m = modulate(nn, img_n2, im[4], im[3]);
    let img_mlp = mlp(nn, &format!("{p}.img_mlp"), img_m, hidden, mlp_dim);
    let img = gate_residual(nn, img, img_mlp, im[5]);

    let txt_n2 = nn.layernorm(&format!("{p}.txt_norm2"), txt, hidden, 1e-6);
    let txt_m = modulate(nn, txt_n2, tm[4], tm[3]);
    let txt_mlp = mlp(nn, &format!("{p}.txt_mlp"), txt_m, hidden, mlp_dim);
    let txt = gate_residual(nn, txt, txt_mlp, tm[5]);

    (img, txt)
}

#[allow(clippy::too_many_arguments)]
fn joint_attention(
    nn: &mut Nn,
    cfg: &QwenImageDitConfig,
    p: &str,
    img: TensorId,
    txt: TensorId,
    b: &Dim,
    l: &Dim,
    n: &Dim,
    hidden: i64,
    heads: u32,
    hd: i64,
) -> (TensorId, TensorId) {
    let theta = 10_000.0;
    let to_heads = |nn: &mut Nn, x: TensorId, seq: &Dim| {
        nn.reshape(
            x,
            [
                b.clone(),
                seq.clone(),
                Dim::stat(heads as i64),
                Dim::stat(hd),
            ],
        )
    };

    // Image stream Q/K/V.
    let iq = nn.linear(&format!("{p}.attn.to_q"), img, hidden, hidden, true);
    let ik = nn.linear(&format!("{p}.attn.to_k"), img, hidden, hidden, true);
    let iv = nn.linear(&format!("{p}.attn.to_v"), img, hidden, hidden, true);
    let mut iq = to_heads(nn, iq, n);
    let mut ik = to_heads(nn, ik, n);
    let iv = to_heads(nn, iv, n);
    iq = nn.rmsnorm(&format!("{p}.attn.norm_q"), iq, hd, 1e-6);
    ik = nn.rmsnorm(&format!("{p}.attn.norm_k"), ik, hd, 1e-6);
    iq = nn.rope(iq, hd as u32, theta);
    ik = nn.rope(ik, hd as u32, theta);

    // Text stream Q/K/V.
    let tq = nn.linear(&format!("{p}.attn.add_q_proj"), txt, hidden, hidden, true);
    let tk = nn.linear(&format!("{p}.attn.add_k_proj"), txt, hidden, hidden, true);
    let tv = nn.linear(&format!("{p}.attn.add_v_proj"), txt, hidden, hidden, true);
    let mut tq = to_heads(nn, tq, l);
    let mut tk = to_heads(nn, tk, l);
    let tv = to_heads(nn, tv, l);
    tq = nn.rmsnorm(&format!("{p}.attn.norm_added_q"), tq, hd, 1e-6);
    tk = nn.rmsnorm(&format!("{p}.attn.norm_added_k"), tk, hd, 1e-6);
    tq = nn.rope(tq, hd as u32, theta);
    tk = nn.rope(tk, hd as u32, theta);

    // Concatenate streams along the sequence axis: [text ; image].
    let q = nn.concat(1, vec![tq, iq]);
    let k = nn.concat(1, vec![tk, ik]);
    let v = nn.concat(1, vec![tv, iv]);

    let attn = nn.attention(q, k, v, heads, heads, hd as u32, false, None, None);
    let total = l.add(n);
    let attn = nn.reshape(attn, [b.clone(), total, Dim::stat(hidden)]);

    // Split back: first L tokens are text, next N are image.
    let txt_attn = nn.slice_dim(attn, 1, Dim::stat(0), l.clone());
    let img_attn = nn.slice_dim(attn, 1, l.clone(), n.clone());
    let _ = cfg;
    (img_attn, txt_attn)
}

fn mlp(nn: &mut Nn, p: &str, x: TensorId, hidden: i64, mlp_dim: i64) -> TensorId {
    let up = nn.linear(&format!("{p}.0"), x, hidden, mlp_dim, true);
    let up = nn.act(ActKind::GeluTanh, up);
    nn.linear(&format!("{p}.2"), up, mlp_dim, hidden, true)
}

/// Modulate: `x * (1 + scale) + shift` (scale/shift are [B,1,H]).
fn modulate(nn: &mut Nn, x: TensorId, scale: TensorId, shift: TensorId) -> TensorId {
    let scaled = nn.mul(x, scale);
    let with_x = nn.add(scaled, x);
    nn.add(with_x, shift)
}

/// Gated residual: `x + gate * y` (gate is [B,1,H]).
fn gate_residual(nn: &mut Nn, x: TensorId, y: TensorId, gate: TensorId) -> TensorId {
    let gated = nn.mul(y, gate);
    nn.add(x, gated)
}

/// Split a `[B, n*H]` modulation projection into `n` chunks of `[B, 1, H]`.
fn chunk_mod(nn: &mut Nn, params: TensorId, n: usize, hidden: i64, b: &Dim) -> Vec<TensorId> {
    (0..n)
        .map(|i| {
            let piece = nn.slice(params, -1, i as i64 * hidden, hidden);
            nn.reshape(piece, [b.clone(), Dim::stat(1), Dim::stat(hidden)])
        })
        .collect()
}
