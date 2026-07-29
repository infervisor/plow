//! SigLIP vision tower → symbolic operator graph.
//!
//! Standard ViT encoder used as an image embedder: Conv2d patch embedding +
//! learned position embedding, a stack of full-attention pre-norm transformer
//! blocks (LayerNorm, GELU-tanh MLP), a final LayerNorm, and mean pooling to a
//! single image embedding. (The real SigLIP head is attention pooling; mean
//! pooling is a faithful-shape stand-in for the embedding output.)

use super::config::{parse_dtype, SiglipConfig};
use crate::op::{ActKind, ReduceKind};
use crate::Nn;
use crate::{Dim, Graph, TensorId};

pub fn build(cfg: &SiglipConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let head_dim = h / nh as i64;
    let eps = cfg.layer_norm_eps;
    let n_patch = cfg.num_patches();

    let b = nn.sym("B");
    let img = nn.input(
        "pixel_values",
        nn.shape([
            b.clone(),
            Dim::stat(cfg.num_channels),
            Dim::stat(cfg.image_size),
            Dim::stat(cfg.image_size),
        ]),
        dt,
    );

    // Patch embedding via strided conv: [B, C, I, I] -> [B, H, P, P].
    let patches = nn.conv2d(
        "embeddings.patch_embedding",
        img,
        cfg.num_channels,
        h,
        (cfg.patch_size, cfg.patch_size),
        (cfg.patch_size as u32, cfg.patch_size as u32),
        (0, 0),
        true,
    );
    // Flatten spatial and move channels last: [B, H, P*P] -> [B, P*P, H].
    let patches = nn.reshape(patches, [b.clone(), Dim::stat(h), Dim::stat(n_patch)]);
    let mut x = nn.transpose(patches, vec![0, 2, 1]);

    // Learned position embedding (broadcast over batch).
    let pos = nn.param(
        "embeddings.position_embedding.weight",
        [Dim::stat(n_patch), Dim::stat(h)],
    );
    x = nn.add(x, pos);

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("encoder.layers.{layer}");
        nn.begin_block(&p);

        let residual = x;
        let normed = nn.layernorm(&format!("{p}.layer_norm1"), x, h, eps);
        let attn = self_attention(&mut nn, &p, normed, &b, n_patch, h, nh, head_dim);
        x = nn.add(residual, attn);

        let residual = x;
        let normed = nn.layernorm(&format!("{p}.layer_norm2"), x, h, eps);
        let mlp = mlp(
            &mut nn,
            &p,
            normed,
            h,
            cfg.intermediate_size,
            &cfg.hidden_act,
        );
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    x = nn.layernorm("post_layernorm", x, h, eps);

    // Pooled image embedding: mean over the patch axis -> [B, H].
    let pooled = nn.reduce(x, ReduceKind::Mean, 1, false);
    nn.mark_output(pooled);
    nn.finish()
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    nn: &mut Nn,
    p: &str,
    x: TensorId,
    b: &Dim,
    seq: i64,
    h: i64,
    nh: u32,
    head_dim: i64,
) -> TensorId {
    let q = nn.linear(&format!("{p}.self_attn.q_proj"), x, h, h, true);
    let k = nn.linear(&format!("{p}.self_attn.k_proj"), x, h, h, true);
    let v = nn.linear(&format!("{p}.self_attn.v_proj"), x, h, h, true);

    let q = nn.reshape(
        q,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );
    let k = nn.reshape(
        k,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );
    let v = nn.reshape(
        v,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );

    // Bidirectional (non-causal) full attention.
    let attn = nn.attention(q, k, v, nh, nh, head_dim as u32, false, None, None);
    let merged = nn.reshape(attn, [b.clone(), Dim::stat(seq), Dim::stat(h)]);
    nn.linear(&format!("{p}.self_attn.out_proj"), merged, h, h, true)
}

fn mlp(nn: &mut Nn, p: &str, x: TensorId, h: i64, inter: i64, act: &str) -> TensorId {
    let up = nn.linear(&format!("{p}.mlp.fc1"), x, h, inter, true);
    let up = nn.act(act_kind(act), up);
    nn.linear(&format!("{p}.mlp.fc2"), up, inter, h, true)
}

fn act_kind(s: &str) -> ActKind {
    match s {
        "gelu" => ActKind::Gelu,
        "gelu_pytorch_tanh" | "gelu_tanh" => ActKind::GeluTanh,
        "relu" => ActKind::Relu,
        "quick_gelu" => ActKind::QuickGelu,
        _ => ActKind::GeluTanh,
    }
}
