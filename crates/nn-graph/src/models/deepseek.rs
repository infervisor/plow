//! DeepSeek-V2 / V3 decoder → symbolic operator graph.
//!
//! The two distinctive pieces:
//!
//! * **MLA (multi-head latent attention).** Q and KV are produced through
//!   low-rank down/up projections. The per-head key/query dim splits into a
//!   non-RoPE ("nope") content part and a RoPE part; the RoPE key is shared
//!   across heads (compressed) and broadcast. Value carries its own head dim.
//! * **DeepSeekMoE.** The first `first_k_dense_replace` layers use a dense MLP;
//!   the rest use a router + shared experts + routed experts. Expert dispatch
//!   is data-dependent (runtime indirection), so the static graph models the
//!   router logits and one representative expert/shared FFN, both shape-equal
//!   to their input.

use super::config::{parse_dtype, DeepSeekConfig};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &DeepSeekConfig) -> Graph {
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

        // Attention sub-block (pre-norm residual).
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), x, h, eps);
        let attn = mla_attention(&mut nn, cfg, &p, normed, &b, &s);
        x = nn.add(residual, attn);

        // FFN sub-block (pre-norm residual): dense for the first K layers, MoE after.
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), x, h, eps);
        let ffn = if layer < cfg.first_k_dense_replace {
            swiglu_mlp(
                &mut nn,
                &format!("{p}.mlp"),
                normed,
                h,
                cfg.intermediate_size,
            )
        } else {
            moe(&mut nn, cfg, &format!("{p}.mlp"), normed, h)
        };
        x = nn.add(residual, ffn);
    }
    nn.end_block();

    x = nn.rmsnorm("norm", x, h, eps);
    let logits = nn.linear("lm_head", x, h, cfg.vocab_size, false);
    nn.mark_output(logits);
    nn.finish()
}

fn mla_attention(
    nn: &mut Nn,
    cfg: &DeepSeekConfig,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps;
    let qk_nope = cfg.qk_nope_head_dim as i64;
    let qk_rope = cfg.qk_rope_head_dim as i64;
    let qk_head = cfg.qk_head_dim() as i64;
    let v_head = cfg.v_head_dim as i64;
    let kv_lora = cfg.kv_lora_rank as i64;

    // ---- query path (optionally low-rank) ----
    let q = if cfg.q_lora_rank > 0 {
        let q_lora = cfg.q_lora_rank as i64;
        let qa = nn.linear(&format!("{p}.self_attn.q_a_proj"), x, h, q_lora, false);
        let qa = nn.rmsnorm(&format!("{p}.self_attn.q_a_layernorm"), qa, q_lora, eps);
        nn.linear(
            &format!("{p}.self_attn.q_b_proj"),
            qa,
            q_lora,
            nh as i64 * qk_head,
            false,
        )
    } else {
        nn.linear(
            &format!("{p}.self_attn.q_proj"),
            x,
            h,
            nh as i64 * qk_head,
            false,
        )
    };
    let q = nn.reshape(
        q,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(qk_head),
        ],
    );
    let q_nope = nn.slice(q, -1, 0, qk_nope);
    let mut q_pe = nn.slice(q, -1, qk_nope, qk_rope);
    q_pe = nn.rope(q_pe, qk_rope as u32, cfg.rope_theta);
    let q = nn.concat(-1, vec![q_nope, q_pe]); // [B,S,nh,qk_head]

    // ---- key/value path (shared compressed latent) ----
    let kv_a = nn.linear(
        &format!("{p}.self_attn.kv_a_proj_with_mqa"),
        x,
        h,
        kv_lora + qk_rope,
        false,
    );
    let compressed = nn.slice(kv_a, -1, 0, kv_lora);
    let mut k_pe = nn.slice(kv_a, -1, kv_lora, qk_rope); // [B,S,qk_rope]
    let compressed = nn.rmsnorm(
        &format!("{p}.self_attn.kv_a_layernorm"),
        compressed,
        kv_lora,
        eps,
    );
    let kv = nn.linear(
        &format!("{p}.self_attn.kv_b_proj"),
        compressed,
        kv_lora,
        nh as i64 * (qk_nope + v_head),
        false,
    );
    let kv = nn.reshape(
        kv,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(qk_nope + v_head),
        ],
    );
    let k_nope = nn.slice(kv, -1, 0, qk_nope);
    let value = nn.slice(kv, -1, qk_nope, v_head); // [B,S,nh,v_head]

    // Shared rotary key: rope on [B,S,1,qk_rope], broadcast across heads.
    k_pe = nn.reshape(
        k_pe,
        [b.clone(), s.clone(), Dim::stat(1), Dim::stat(qk_rope)],
    );
    k_pe = nn.rope(k_pe, qk_rope as u32, cfg.rope_theta);
    k_pe = nn.broadcast(
        k_pe,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(qk_rope),
        ],
    );
    let k = nn.concat(-1, vec![k_nope, k_pe]); // [B,S,nh,qk_head]

    // Attention. Q/K head dim is qk_head; V head dim is v_head ⇒ output [B,S,nh,v_head].
    let attn = nn.attention(q, k, value, nh, nh, qk_head as u32, true, None, None);
    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(nh as i64 * v_head)]);
    nn.linear(
        &format!("{p}.self_attn.o_proj"),
        merged,
        nh as i64 * v_head,
        h,
        false,
    )
}

/// SwiGLU dense MLP: `down(silu(gate(x)) * up(x))`.
fn swiglu_mlp(nn: &mut Nn, p: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{p}.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{p}.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{p}.down_proj"), hidden, inter, h, false)
}

/// DeepSeekMoE: router logits + shared expert(s) + one representative routed
/// expert, recombined. Per-token expert selection is a runtime indirection;
/// the graph carries the router and the FFN shapes.
fn moe(nn: &mut Nn, cfg: &DeepSeekConfig, p: &str, x: TensorId, h: i64) -> TensorId {
    // Router: [.., H] -> [.., n_routed_experts] logits.
    let _logits = nn.moe_router(
        &format!("{p}.gate"),
        x,
        h,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );

    // One representative routed expert FFN (replicated `n_routed_experts` times
    // at load time; shape-identical, so a single node stands in for the graph).
    let routed = swiglu_mlp(
        nn,
        &format!("{p}.experts.0"),
        x,
        h,
        cfg.moe_intermediate_size,
    );

    if cfg.n_shared_experts > 0 {
        let shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts as i64;
        let shared = swiglu_mlp(nn, &format!("{p}.shared_experts"), x, h, shared_inter);
        nn.add(routed, shared)
    } else {
        routed
    }
}
