//! SNAC decoder (codes → 24 kHz PCM) → symbolic operator graph, channels-last `[B, T, C]`.
//!
//! Tensor names are the export's (`scripts/tts/snac_export.py`), so fused sites anchor on the
//! names `devgen::codec` loads. The export has already folded weight norm and projected the
//! codebooks (`q.P{level}`: code → latent row). The per-block noise `n ~ N(0, 1)` is a graph
//! input (`noise.blk{i}`, `[B, T_i, 1]`): it is drawn at run time, not a weight.

use super::config::SnacConfig;
use crate::op::{ActKind, Op, PadMode};
use crate::{DType, Dim, Graph, Nn, TensorId};

fn conv(stride: u32, dilation: u32, groups: u32, pad: u32) -> Op {
    Op::Conv1d {
        stride,
        dilation,
        groups,
        padding: (pad, pad),
        pad_mode: PadMode::Zero,
    }
}

pub fn build(cfg: &SnacConfig) -> Graph {
    let mut nn = Nn::new(DType::F32, DType::F32);
    let b = nn.sym("B");
    // Frames; one frame is `vq_strides[0]` latent rows.
    let f = nn.sym("F");
    let rows = cfg.vq_strides.first().copied().unwrap_or(1);
    let latent = cfg.latent_dim;

    // Latent: the sum over levels (coarse first) of each level's projected code, repeated over
    // the latent rows it covers.
    let mut z: Option<TensorId> = None;
    for (level, &stride) in cfg.vq_strides.iter().enumerate() {
        let codes = f.mul(&Dim::stat(rows / stride));
        let ids = nn.input(&format!("codes.{level}"), nn.shape([b.clone(), codes.clone()]), DType::I32);
        let e = nn.embedding_named(&format!("q.P{level}"), ids, cfg.codebook_size, latent);
        let e = nn.reshape(e, [b.clone(), codes.clone(), Dim::stat(1), Dim::stat(latent)]);
        let e = nn.broadcast(e, [b.clone(), codes, Dim::stat(stride), Dim::stat(latent)]);
        let e = nn.reshape(e, [b.clone(), f.mul(&Dim::stat(rows)), Dim::stat(latent)]);
        z = Some(match z {
            Some(z) => nn.add(z, e),
            None => e,
        });
    }
    let z = z.expect("SNAC has at least one codebook level");

    let x = nn.conv1d("pre.dw.w", Some("pre.dw.b"), z, latent, latent, 7, conv(1, 1, latent as u32, 3));
    let mut c = cfg.blocks.first().map_or(latent, |blk| blk.cin);
    let mut x = nn.conv1d("pre.pw.w", Some("pre.pw.b"), x, latent, c, 1, conv(1, 1, 1, 0));
    let mut t = f.mul(&Dim::stat(rows));
    for (i, blk) in cfg.blocks.iter().enumerate() {
        let p = format!("blk{i}");
        nn.begin_block(&p);
        let s = nn.snake(&format!("{p}.snake"), x, blk.cin);
        let up = Op::ConvTranspose1d {
            stride: blk.stride,
            groups: 1,
            crop: (blk.padding, blk.padding),
            output_padding: blk.output_padding,
        };
        x = nn.conv1d(&format!("{p}.up.w"), Some(&format!("{p}.up.b")), s, blk.cin, blk.cout, i64::from(blk.kernel), up);
        c = blk.cout;
        t = t
            .sub(&Dim::stat(1))
            .mul(&Dim::stat(i64::from(blk.stride)))
            .add(&Dim::stat(i64::from(blk.kernel + blk.output_padding) - 2 * i64::from(blk.padding)));
        // Noise injection: x + n * (x · Wn).
        let noise = nn.input(&format!("noise.{p}"), nn.shape([b.clone(), t.clone(), Dim::stat(1)]), DType::F32);
        let h = nn.conv1d(&format!("{p}.noise.w"), None, x, c, c, 1, conv(1, 1, 1, 0));
        let n = nn.mul(h, noise);
        x = nn.add(x, n);
        for (j, &d) in blk.dilations.iter().enumerate() {
            let r = format!("{p}.ru{j}");
            let y = nn.snake(&format!("{r}.a1"), x, c);
            let y = nn.conv1d(&format!("{r}.dw.w"), Some(&format!("{r}.dw.b")), y, c, c, 7, conv(1, d, c as u32, 3 * d));
            let y = nn.snake(&format!("{r}.a2"), y, c);
            let y = nn.conv1d(&format!("{r}.pw.w"), Some(&format!("{r}.pw.b")), y, c, c, 1, conv(1, 1, 1, 0));
            x = nn.add(x, y);
        }
        nn.end_block();
    }
    let y = nn.snake("out.snake", x, c);
    let y = nn.conv1d("out.w", Some("out.b"), y, c, 1, 7, conv(1, 1, 1, 3));
    let pcm = nn.act(ActKind::Tanh, y);
    nn.mark_output(pcm);
    nn.finish()
}
