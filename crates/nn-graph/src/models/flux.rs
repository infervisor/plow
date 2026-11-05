//! FLUX.1 DiT (`FluxTransformer2DModel`) → symbolic operator graph.
//!
//! `transformer_blocks.*` are double-stream MMDiT blocks (separate image/text weights, joint
//! attention), `single_transformer_blocks.*` run one stream over `[text ; image]` with a fused
//! attention + MLP output projection. Weight names follow the diffusers checkpoint. The
//! sinusoidal timestep/guidance projections are graph inputs (`timestep_proj`,
//! `guidance_proj`, 256 wide); everything after them is modelled. LayerNorms carry weights as in
//! the Qwen-Image DiT builder.

use super::config::{parse_dtype, FluxConfig};
use super::ShapeBucket;
use crate::op::ActKind;
use crate::Nn;
use crate::{Dim, Graph, TensorId};

/// VAE downsample (8) times the pipeline's 2x2 latent packing.
const PIXELS_PER_TOKEN: i64 = 16;
const TIME_PROJ_DIM: i64 = 256;

pub fn build(cfg: &FluxConfig, bucket: &ShapeBucket) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);
    let hidden = cfg.hidden_size();
    let heads = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let mlp_dim = hidden * 4;

    let n_img = (bucket.image_height / PIXELS_PER_TOKEN / cfg.patch_size)
        * (bucket.image_width / PIXELS_PER_TOKEN / cfg.patch_size);
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
    let pooled = nn.input(
        "pooled_projections",
        nn.shape([b.clone(), Dim::stat(cfg.pooled_projection_dim)]),
        dt,
    );
    let t_proj = nn.input(
        "timestep_proj",
        nn.shape([b.clone(), Dim::stat(TIME_PROJ_DIM)]),
        dt,
    );

    let mut temb = mlp_embedder(
        &mut nn,
        "time_text_embed.timestep_embedder",
        t_proj,
        TIME_PROJ_DIM,
        hidden,
    );
    if cfg.guidance_embeds {
        let g_proj = nn.input(
            "guidance_proj",
            nn.shape([b.clone(), Dim::stat(TIME_PROJ_DIM)]),
            dt,
        );
        let g = mlp_embedder(
            &mut nn,
            "time_text_embed.guidance_embedder",
            g_proj,
            TIME_PROJ_DIM,
            hidden,
        );
        temb = nn.add(temb, g);
    }
    let pooled = mlp_embedder(
        &mut nn,
        "time_text_embed.text_embedder",
        pooled,
        cfg.pooled_projection_dim,
        hidden,
    );
    let temb = nn.add(temb, pooled);

    let mut img = nn.linear("x_embedder", img_in, cfg.in_channels, hidden, true);
    let mut txt = nn.linear(
        "context_embedder",
        txt_in,
        cfg.joint_attention_dim,
        hidden,
        true,
    );

    let dims = Dims {
        b: &b,
        l: &l,
        n: &n,
        hidden,
        heads,
        hd,
        mlp_dim,
    };
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        nn.begin_block(&p);
        (img, txt) = double_block(&mut nn, &p, img, txt, temb, &dims);
    }
    nn.end_block();

    let total = l.add(&n);
    let mut x = nn.concat(1, vec![txt, img]);
    for i in 0..cfg.num_single_layers {
        let p = format!("single_transformer_blocks.{i}");
        nn.begin_block(&p);
        x = single_block(&mut nn, &p, x, temb, &total, &dims);
    }
    nn.end_block();
    let img = nn.slice_dim(x, 1, l.clone(), n.clone());

    let smod = nn.act(ActKind::Silu, temb);
    let params = nn.linear("norm_out.linear", smod, hidden, 2 * hidden, true);
    let m = chunk_mod(&mut nn, params, 2, hidden, &b);
    let img_n = nn.layernorm("norm_out.norm", img, hidden, 1e-6);
    let img = modulate(&mut nn, img_n, m[0], m[1]);
    let out_ch = cfg.out_channels.unwrap_or(cfg.in_channels);
    let out = nn.linear(
        "proj_out",
        img,
        hidden,
        cfg.patch_size * cfg.patch_size * out_ch,
        true,
    );
    nn.mark_output(out);
    nn.finish()
}

struct Dims<'a> {
    b: &'a Dim,
    l: &'a Dim,
    n: &'a Dim,
    hidden: i64,
    heads: u32,
    hd: i64,
    mlp_dim: i64,
}

fn mlp_embedder(nn: &mut Nn, p: &str, x: TensorId, cin: i64, hidden: i64) -> TensorId {
    let h = nn.linear(&format!("{p}.linear_1"), x, cin, hidden, true);
    let h = nn.act(ActKind::Silu, h);
    nn.linear(&format!("{p}.linear_2"), h, hidden, hidden, true)
}

/// Q/K/V with per-head RMSNorm and RoPE, as `[B, seq, heads, hd]`.
fn qkv(nn: &mut Nn, p: &str, names: [&str; 5], x: TensorId, seq: &Dim, d: &Dims) -> [TensorId; 3] {
    let [q, k, v, nq, nk] = names;
    let heads = |nn: &mut Nn, t: TensorId| {
        nn.reshape(
            t,
            [
                d.b.clone(),
                seq.clone(),
                Dim::stat(d.heads as i64),
                Dim::stat(d.hd),
            ],
        )
    };
    let q = nn.linear(&format!("{p}.{q}"), x, d.hidden, d.hidden, true);
    let k = nn.linear(&format!("{p}.{k}"), x, d.hidden, d.hidden, true);
    let v = nn.linear(&format!("{p}.{v}"), x, d.hidden, d.hidden, true);
    let (q, k, v) = (heads(nn, q), heads(nn, k), heads(nn, v));
    let q = nn.rmsnorm(&format!("{p}.{nq}"), q, d.hd, 1e-6);
    let k = nn.rmsnorm(&format!("{p}.{nk}"), k, d.hd, 1e-6);
    let q = nn.rope(q, d.hd as u32, 10_000.0);
    let k = nn.rope(k, d.hd as u32, 10_000.0);
    [q, k, v]
}

fn double_block(
    nn: &mut Nn,
    p: &str,
    img: TensorId,
    txt: TensorId,
    temb: TensorId,
    d: &Dims,
) -> (TensorId, TensorId) {
    let smod = nn.act(ActKind::Silu, temb);
    let im = nn.linear(
        &format!("{p}.norm1.linear"),
        smod,
        d.hidden,
        6 * d.hidden,
        true,
    );
    let tm = nn.linear(
        &format!("{p}.norm1_context.linear"),
        smod,
        d.hidden,
        6 * d.hidden,
        true,
    );
    let im = chunk_mod(nn, im, 6, d.hidden, d.b);
    let tm = chunk_mod(nn, tm, 6, d.hidden, d.b);

    let img_n = nn.layernorm(&format!("{p}.norm1.norm"), img, d.hidden, 1e-6);
    let img_a = modulate(nn, img_n, im[0], im[1]);
    let txt_n = nn.layernorm(&format!("{p}.norm1_context.norm"), txt, d.hidden, 1e-6);
    let txt_a = modulate(nn, txt_n, tm[0], tm[1]);

    let attn = format!("{p}.attn");
    let [iq, ik, iv] = qkv(
        nn,
        &attn,
        ["to_q", "to_k", "to_v", "norm_q", "norm_k"],
        img_a,
        d.n,
        d,
    );
    let [tq, tk, tv] = qkv(
        nn,
        &attn,
        [
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "norm_added_q",
            "norm_added_k",
        ],
        txt_a,
        d.l,
        d,
    );
    let q = nn.concat(1, vec![tq, iq]);
    let k = nn.concat(1, vec![tk, ik]);
    let v = nn.concat(1, vec![tv, iv]);
    let a = nn.attention(q, k, v, d.heads, d.heads, d.hd as u32, false, None, None);
    let a = nn.reshape(a, [d.b.clone(), d.l.add(d.n), Dim::stat(d.hidden)]);
    let txt_a = nn.slice_dim(a, 1, Dim::stat(0), d.l.clone());
    let img_a = nn.slice_dim(a, 1, d.l.clone(), d.n.clone());
    let img_a = nn.linear(&format!("{attn}.to_out.0"), img_a, d.hidden, d.hidden, true);
    let txt_a = nn.linear(
        &format!("{attn}.to_add_out"),
        txt_a,
        d.hidden,
        d.hidden,
        true,
    );
    let img = gate_residual(nn, img, img_a, im[2]);
    let txt = gate_residual(nn, txt, txt_a, tm[2]);

    let img_n = nn.layernorm(&format!("{p}.norm2"), img, d.hidden, 1e-6);
    let img_m = modulate(nn, img_n, im[3], im[4]);
    let img_m = feed_forward(nn, &format!("{p}.ff"), img_m, d);
    let img = gate_residual(nn, img, img_m, im[5]);

    let txt_n = nn.layernorm(&format!("{p}.norm2_context"), txt, d.hidden, 1e-6);
    let txt_m = modulate(nn, txt_n, tm[3], tm[4]);
    let txt_m = feed_forward(nn, &format!("{p}.ff_context"), txt_m, d);
    let txt = gate_residual(nn, txt, txt_m, tm[5]);
    (img, txt)
}

fn single_block(
    nn: &mut Nn,
    p: &str,
    x: TensorId,
    temb: TensorId,
    seq: &Dim,
    d: &Dims,
) -> TensorId {
    let smod = nn.act(ActKind::Silu, temb);
    let m = nn.linear(
        &format!("{p}.norm.linear"),
        smod,
        d.hidden,
        3 * d.hidden,
        true,
    );
    let m = chunk_mod(nn, m, 3, d.hidden, d.b);
    let xn = nn.layernorm(&format!("{p}.norm.norm"), x, d.hidden, 1e-6);
    let xm = modulate(nn, xn, m[0], m[1]);

    let mlp = nn.linear(&format!("{p}.proj_mlp"), xm, d.hidden, d.mlp_dim, true);
    let mlp = nn.act(ActKind::GeluTanh, mlp);

    let [q, k, v] = qkv(
        nn,
        &format!("{p}.attn"),
        ["to_q", "to_k", "to_v", "norm_q", "norm_k"],
        xm,
        seq,
        d,
    );
    let a = nn.attention(q, k, v, d.heads, d.heads, d.hd as u32, false, None, None);
    let a = nn.reshape(a, [d.b.clone(), seq.clone(), Dim::stat(d.hidden)]);

    let fused = nn.concat(2, vec![a, mlp]);
    let y = nn.linear(
        &format!("{p}.proj_out"),
        fused,
        d.hidden + d.mlp_dim,
        d.hidden,
        true,
    );
    gate_residual(nn, x, y, m[2])
}

fn feed_forward(nn: &mut Nn, p: &str, x: TensorId, d: &Dims) -> TensorId {
    let up = nn.linear(&format!("{p}.net.0.proj"), x, d.hidden, d.mlp_dim, true);
    let up = nn.act(ActKind::GeluTanh, up);
    nn.linear(&format!("{p}.net.2"), up, d.mlp_dim, d.hidden, true)
}

/// `x * (1 + scale) + shift`; diffusers chunks AdaLN-Zero as (shift, scale, gate, ...).
fn modulate(nn: &mut Nn, x: TensorId, shift: TensorId, scale: TensorId) -> TensorId {
    let scaled = nn.mul(x, scale);
    let with_x = nn.add(scaled, x);
    nn.add(with_x, shift)
}

fn gate_residual(nn: &mut Nn, x: TensorId, y: TensorId, gate: TensorId) -> TensorId {
    let gated = nn.mul(y, gate);
    nn.add(x, gated)
}

/// `[B, n*H]` → `n` chunks of `[B, 1, H]`.
fn chunk_mod(nn: &mut Nn, params: TensorId, n: usize, hidden: i64, b: &Dim) -> Vec<TensorId> {
    (0..n)
        .map(|i| {
            let piece = nn.slice(params, -1, i as i64 * hidden, hidden);
            nn.reshape(piece, [b.clone(), Dim::stat(1), Dim::stat(hidden)])
        })
        .collect()
}
