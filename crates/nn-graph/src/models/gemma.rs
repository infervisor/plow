//! Gemma 3/4 text decoder → symbolic operator graph.
//!
//! Captures the Gemma3/4-family specifics: embedding scaled by `sqrt(hidden)`,
//! GQA, query/key RMSNorm, RoPE (per-layer-type theta/partial), alternating
//! sliding-window/global attention, `query_pre_attn_scalar` query scaling,
//! GeGLU MLP, and the four-norm (pre+post) residual block layout.
//!
//! Gemma 4 MoE variants (gemma-4-26B-A4B) replace the dense GeGLU MLP with a
//! routed expert layer on designated MoE layers. The router + representative
//! expert pattern matches the existing DeepSeek MoE infra.
//!
//! Gemma 1/2 are not supported (they use a different 2-norm block layout and
//! attention logit softcapping that Gemma 3+ removed).

use super::config::{parse_dtype, GemmaConfig};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &GemmaConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let model_prefix = cfg.weight_prefix.as_deref().unwrap_or("model");
    let name = |suffix: &str| {
        if model_prefix.is_empty() {
            suffix.to_string()
        } else {
            format!("{model_prefix}.{suffix}")
        }
    };

    // Symbolic batch / sequence.
    let b = nn.sym("B");
    let s = nn.sym("S");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);

    // Embedding, scaled by sqrt(hidden) (Gemma normalizer).
    let embed_name = name("embed_tokens");
    let mut x = nn.embedding(&embed_name, ids, cfg.vocab_size, h);
    let embedding_scale = if cfg.is_gemma4() {
        round_to_bf16((h as f32).sqrt())
    } else {
        (h as f32).sqrt()
    };
    x = nn.scale(x, embedding_scale);

    for layer in 0..cfg.num_hidden_layers {
        let p = name(&format!("layers.{layer}"));
        nn.begin_block(&p);
        let is_global = cfg.layer_is_global(layer);
        let sliding = if is_global {
            None
        } else {
            Some(cfg.sliding_window)
        };

        // --- attention block (pre-norm, post-norm, residual) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), x, h, eps);
        let attn = attention(&mut nn, cfg, &p, normed, &b, &s, is_global, sliding);
        let attn = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), attn, h, eps);
        x = nn.add(residual, attn);

        // --- MLP block (pre-norm, post-norm, residual) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.pre_feedforward_layernorm"), x, h, eps);
        let mlp = if cfg.layer_is_moe(layer) {
            moe_ffn(&mut nn, cfg, &p, normed, h)
        } else {
            geglu_mlp(&mut nn, cfg, &p, normed, h)
        };
        let mlp = nn.rmsnorm(&format!("{p}.post_feedforward_layernorm"), mlp, h, eps);
        x = nn.add(residual, mlp);
        if cfg.is_gemma4() {
            let layer_scalar = nn.param(&format!("{p}.layer_scalar"), [Dim::stat(1)]);
            x = nn.mul(x, layer_scalar);
        }
    }
    nn.end_block();

    x = nn.rmsnorm(&name("norm"), x, h, eps);

    let mut logits = if cfg.tie_word_embeddings {
        nn.linear(&embed_name, x, h, cfg.vocab_size, false)
    } else {
        nn.linear("lm_head", x, h, cfg.vocab_size, false)
    };
    if let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) {
        logits = nn.scale(logits, 1.0 / cap);
        logits = nn.act(ActKind::Tanh, logits);
        logits = nn.scale(logits, cap);
    }
    nn.mark_output(logits);
    nn.finish()
}

#[allow(clippy::too_many_arguments)]
fn attention(
    nn: &mut Nn,
    cfg: &GemmaConfig,
    prefix: &str,
    x: crate::TensorId,
    b: &Dim,
    s: &Dim,
    is_global: bool,
    sliding_window: Option<u32>,
) -> crate::TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    // Gemma 4: head_dim and KV-head count depend on the layer type.
    let nkv = cfg.kv_heads_for(is_global);
    let hd = cfg.head_dim_for(is_global);
    let eps = cfg.rms_norm_eps;
    let q_dim = nh as i64 * hd;
    let kv_dim = nkv as i64 * hd;

    // Full-attention Gemma 4 layers derive both K and V from `k_proj`. Sliding
    // layers retain an independent `v_proj` even when `attention_k_eq_v=true`.
    let q = nn.linear(&format!("{prefix}.self_attn.q_proj"), x, h, q_dim, false);
    let k = nn.linear(&format!("{prefix}.self_attn.k_proj"), x, h, kv_dim, false);
    let (k_lin, v_lin) = if cfg.attention_k_eq_v && is_global {
        (k, k)
    } else {
        let v = nn.linear(&format!("{prefix}.self_attn.v_proj"), x, h, kv_dim, false);
        (k, v)
    };

    // Split heads: [B, S, n*hd] -> [B, S, n, hd].
    let mut q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );
    let mut k = nn.reshape(
        k_lin,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );
    let mut v = nn.reshape(
        v_lin,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );

    // Gemma3/4 query/key RMSNorm over head_dim.
    if cfg.use_qk_norm {
        q = nn.rmsnorm(&format!("{prefix}.self_attn.q_norm"), q, hd, eps);
        k = nn.rmsnorm(&format!("{prefix}.self_attn.k_norm"), k, hd, eps);
    }
    if cfg.is_gemma4() {
        v = nn.rmsnorm_weightless(v, eps);
    }

    // RoPE over the (possibly partial) head_dim, with per-layer-type theta.
    let (theta, rotary_factor) = cfg.rope_for(is_global);
    let rotary_dim = ((hd as f32) * rotary_factor).round() as u32;
    let frequency_dim = cfg.rope_frequency_dim(is_global, rotary_dim, hd as u32);
    q = nn.rope_with_frequency_dim(q, rotary_dim, theta, frequency_dim);
    k = nn.rope_with_frequency_dim(k, rotary_dim, theta, frequency_dim);

    // Gemma query scaling: 1/sqrt(query_pre_attn_scalar) (defaults to head_dim).
    let scalar = if cfg.query_pre_attn_scalar > 0.0 {
        cfg.query_pre_attn_scalar
    } else {
        hd as f32
    };
    if !cfg.is_gemma4() {
        q = nn.scale(q, 1.0 / scalar.sqrt());
    }

    let attn = nn.attention(q, k, v, nh, nkv, hd as u32, true, sliding_window, None);

    // Merge heads back: [B, S, n, hd] -> [B, S, n*hd].
    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    nn.linear(
        &format!("{prefix}.self_attn.o_proj"),
        merged,
        q_dim,
        h,
        false,
    )
}

fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

/// GeGLU MLP: `down(act(gate(x)) * up(x))`.
fn geglu_mlp(nn: &mut Nn, cfg: &GemmaConfig, prefix: &str, x: TensorId, h: i64) -> TensorId {
    let inter = cfg.intermediate_size;
    let gate = nn.linear(&format!("{prefix}.mlp.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{prefix}.mlp.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::GeluTanh, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{prefix}.mlp.down_proj"), hidden, inter, h, false)
}

/// Single expert GeGLU FFN (same activation as the dense MLP).
fn expert_geglu(nn: &mut Nn, prefix: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{prefix}.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{prefix}.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::GeluTanh, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{prefix}.down_proj"), hidden, inter, h, false)
}

/// MoE FFN: router logits + one representative routed expert GeGLU. Per-token
/// expert selection is a runtime indirection; the graph carries the router and
/// the FFN shapes (same pattern as DeepSeek MoE).
fn moe_ffn(nn: &mut Nn, cfg: &GemmaConfig, prefix: &str, x: TensorId, h: i64) -> TensorId {
    // Router: [.., H] -> [.., num_local_experts] logits.
    let _logits = nn.moe_router(
        &format!("{prefix}.mlp.gate"),
        x,
        h,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
    );

    // One representative routed expert FFN (replicated `num_local_experts` times
    // at load time; shape-identical, so a single node stands in for the graph).
    expert_geglu(
        nn,
        &format!("{prefix}.mlp.experts.0"),
        x,
        h,
        cfg.intermediate_size,
    )
}
