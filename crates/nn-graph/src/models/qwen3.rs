//! Qwen3 / Qwen2.5 text decoder → symbolic operator graph.
//!
//! Standard pre-norm decoder: RMSNorm, GQA with RoPE, SwiGLU MLP. Differs
//! from the LLaMA family in two ways: per-head qk-norm before RoPE, and an
//! explicit `head_dim` decoupled from hidden/heads (Qwen3-4B: 2560/32 but
//! head_dim 128).

use super::config::{parse_dtype, Qwen3Config};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &Qwen3Config) -> Graph {
    build_inner(cfg, None)
}

/// Build as a text ENCODER: no lm_head; the final RMSNorm'd hidden states
/// `[B, S, H]` are the last output. `taps` lists layer indices whose
/// residual-stream output is additionally marked as an output (in the given
/// order, before the final hidden states). Pass `&[]` for last-hidden-state
/// only — the Z-Image text-encoder configuration.
pub fn build_encoder(cfg: &Qwen3Config, taps: &[u32]) -> Graph {
    build_inner(cfg, Some(taps))
}

fn build_inner(cfg: &Qwen3Config, encoder_taps: Option<&[u32]>) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    let b = nn.sym("B");
    let s = nn.sym("S");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let mut x = nn.embedding("embed_tokens", ids, cfg.vocab_size, h);

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("layers.{layer}");
        nn.begin_block(&p);

        // --- attention block (pre-norm residual) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), x, h, eps);
        let attn = attention(&mut nn, cfg, &p, normed, &b, &s);
        x = nn.add(residual, attn);

        // --- MLP block (pre-norm residual, SwiGLU) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), x, h, eps);
        let mlp = swiglu_mlp(&mut nn, &p, normed, h, cfg.intermediate_size);
        x = nn.add(residual, mlp);

        if let Some(taps) = encoder_taps {
            if taps.contains(&layer) {
                nn.mark_output(x);
            }
        }
    }
    nn.end_block();

    x = nn.rmsnorm("norm", x, h, eps);
    if encoder_taps.is_some() {
        nn.mark_output(x);
        return nn.finish();
    }
    let logits = if cfg.tie_word_embeddings {
        nn.linear("embed_tokens", x, h, cfg.vocab_size, false)
    } else {
        nn.linear("lm_head", x, h, cfg.vocab_size, false)
    };
    nn.mark_output(logits);
    nn.finish()
}

fn attention(
    nn: &mut Nn,
    cfg: &Qwen3Config,
    prefix: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.kv_heads();
    let hd = cfg.head_dim();
    let q_dim = nh as i64 * hd;
    let kv_dim = nkv as i64 * hd;

    let q = nn.linear(
        &format!("{prefix}.self_attn.q_proj"),
        x,
        h,
        q_dim,
        cfg.qkv_bias,
    );
    let k = nn.linear(
        &format!("{prefix}.self_attn.k_proj"),
        x,
        h,
        kv_dim,
        cfg.qkv_bias,
    );
    let v = nn.linear(
        &format!("{prefix}.self_attn.v_proj"),
        x,
        h,
        kv_dim,
        cfg.qkv_bias,
    );

    // Split heads: [B, S, n*hd] -> [B, S, n, hd].
    let q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );
    let k = nn.reshape(
        k,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );
    let v = nn.reshape(
        v,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );

    // Qwen3 adds per-head Q/K RMSNorm. Qwen2/Qwen2.5 goes directly to RoPE.
    let (q, k) = if cfg.use_qk_norm {
        (
            nn.rmsnorm(
                &format!("{prefix}.self_attn.q_norm"),
                q,
                hd,
                cfg.rms_norm_eps,
            ),
            nn.rmsnorm(
                &format!("{prefix}.self_attn.k_norm"),
                k,
                hd,
                cfg.rms_norm_eps,
            ),
        )
    } else {
        (q, k)
    };
    let q = nn.rope(q, hd as u32, cfg.rope_theta);
    let k = nn.rope(k, hd as u32, cfg.rope_theta);

    let attn = nn.attention(q, k, v, nh, nkv, hd as u32, true, None, None);

    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    nn.linear(
        &format!("{prefix}.self_attn.o_proj"),
        merged,
        q_dim,
        h,
        false,
    )
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
fn swiglu_mlp(nn: &mut Nn, prefix: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{prefix}.mlp.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{prefix}.mlp.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{prefix}.mlp.down_proj"), hidden, inter, h, false)
}
