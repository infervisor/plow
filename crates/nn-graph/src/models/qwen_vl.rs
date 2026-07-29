//! Qwen2-VL / Qwen2.5-VL vision encoder → symbolic operator graph.
//!
//! Processes an image as a flat sequence of patches (`pixel_values` is already
//! patchified by the processor). A linear patch embedding, a stack of
//! RMSNorm/2D-RoPE full-attention blocks with gated-SiLU MLPs, and a spatial
//! patch merger that fuses `spatial_merge_size²` patches and projects into the
//! LLM hidden size.
//!
//! The pre-merge token count is modeled as `Pm * merge²` (Pm = merged-token
//! count) so the merger's reshape divides exactly while staying symbolic.

use super::config::{parse_dtype, QwenVlVisionConfig};
use crate::op::ActKind;
use crate::Nn;
use crate::{Dim, Graph, TensorId};

pub fn build(cfg: &QwenVlVisionConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let head_dim = cfg.head_dim() as i64;
    let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
    // RMSNorm eps isn't in the vision sub-config; Qwen uses 1e-6.
    let eps = 1e-6f32;

    // Symbolic merged-token count; pre-merge tokens = Pm * merge².
    let pm = nn.sym("Pm");
    let tokens = pm.mul(&Dim::stat(merge2));

    let pixels = nn.input(
        "pixel_values",
        nn.shape([tokens.clone(), Dim::stat(cfg.patch_input_dim())]),
        dt,
    );

    // Linear patch embedding (Conv3d over a patch == linear over flattened patch).
    let mut x = nn.linear("patch_embed.proj", pixels, cfg.patch_input_dim(), h, true);

    for layer in 0..cfg.depth {
        let p = format!("blocks.{layer}");
        nn.begin_block(&p);

        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.norm1"), x, h, eps);
        let attn = attention(
            &mut nn,
            &p,
            normed,
            &tokens,
            h,
            nh,
            head_dim,
            cfg.rope_theta(),
        );
        x = nn.add(residual, attn);

        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.norm2"), x, h, eps);
        let mlp = gated_mlp(&mut nn, &p, normed, h, cfg.intermediate_size);
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    // Patch merger: norm, fuse merge² neighbors, project to LLM hidden size.
    let x = nn.rmsnorm("merger.ln_q", x, h, eps);
    let merged = nn.reshape(x, [pm.clone(), Dim::stat(cfg.merged_dim())]);
    let m = nn.linear(
        "merger.mlp.0",
        merged,
        cfg.merged_dim(),
        cfg.merged_dim(),
        true,
    );
    let m = nn.act(ActKind::Gelu, m);
    let out = nn.linear(
        "merger.mlp.2",
        m,
        cfg.merged_dim(),
        cfg.out_hidden_size,
        true,
    );

    nn.mark_output(out);
    nn.finish()
}

#[allow(clippy::too_many_arguments)]
fn attention(
    nn: &mut Nn,
    p: &str,
    x: TensorId,
    tokens: &Dim,
    h: i64,
    nh: u32,
    head_dim: i64,
    theta: f32,
) -> TensorId {
    // Fused QKV projection, then split.
    let qkv = nn.linear(&format!("{p}.attn.qkv"), x, h, 3 * h, true);
    let q = nn.slice(qkv, -1, 0, h);
    let k = nn.slice(qkv, -1, h, h);
    let v = nn.slice(qkv, -1, 2 * h, h);

    let q = nn.reshape(
        q,
        [tokens.clone(), Dim::stat(nh as i64), Dim::stat(head_dim)],
    );
    let k = nn.reshape(
        k,
        [tokens.clone(), Dim::stat(nh as i64), Dim::stat(head_dim)],
    );
    let v = nn.reshape(
        v,
        [tokens.clone(), Dim::stat(nh as i64), Dim::stat(head_dim)],
    );

    // 2D rotary position embedding (over head_dim).
    let q = nn.rope(q, head_dim as u32, theta);
    let k = nn.rope(k, head_dim as u32, theta);

    let attn = nn.attention(q, k, v, nh, nh, head_dim as u32, false, None, None);
    let merged = nn.reshape(attn, [tokens.clone(), Dim::stat(h)]);
    nn.linear(&format!("{p}.attn.proj"), merged, h, h, true)
}

fn gated_mlp(nn: &mut Nn, p: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{p}.mlp.gate_proj"), x, h, inter, true);
    let up = nn.linear(&format!("{p}.mlp.up_proj"), x, h, inter, true);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{p}.mlp.down_proj"), hidden, inter, h, true)
}
