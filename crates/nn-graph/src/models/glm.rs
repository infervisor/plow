//! GLM-5 MoE-DSA decoder → symbolic operator graph.
//!
//! Architecturally a close relative of DeepSeek-V3:
//!
//! * **MLA (multi-head latent attention).** Same low-rank Q/KV decomposition:
//!   q_a → q_norm → q_b (or direct q_proj), kv_a → kv_norm → kv_b, with a
//!   shared rotary key broadcast across heads. Uses interleaved RoPE.
//! * **DeepSeekMoE.** Dense MLP for the first `first_k_dense_replace` layers,
//!   then router + shared expert + routed experts. Per-layer dispatch driven by
//!   `mlp_layer_types[]`.
//! * **DSA (Dense-Sparse Attention).** Novel indexer mechanism: "full" layers
//!   attend over the entire KV cache; "shared" layers compute a scoring
//!   projection, select top-k KV positions, and attend only over those. Modeled
//!   as a standard attention with `seq_kv = index_topk` for the sparse layers.
//! * **Multi-token prediction.** Extra linear head(s) that predict token+N.
//!   Compile-time: just additional GEMMs off the final norm output.

use super::config::{parse_dtype, GlmConfig};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &GlmConfig) -> Graph {
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
        let attn = mla_attention(&mut nn, cfg, &p, normed, &b, &s, layer);
        x = nn.add(residual, attn);

        // FFN sub-block (pre-norm residual): dense for early layers, MoE after.
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), x, h, eps);
        let ffn = if cfg.layer_is_dense(layer) {
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

    // Multi-token prediction heads (compile-time: just extra linears).
    for mtp in 0..cfg.num_nextn_predict_layers {
        let mtp_logits = nn.linear(
            &format!("mtp_heads.{mtp}.lm_head"),
            x,
            h,
            cfg.vocab_size,
            false,
        );
        nn.mark_output(mtp_logits);
    }

    nn.finish()
}

fn mla_attention(
    nn: &mut Nn,
    cfg: &GlmConfig,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
    layer: u32,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps;
    let qk_nope = cfg.qk_nope_head_dim as i64;
    let qk_rope = cfg.qk_rope_head_dim as i64;
    let qk_head = cfg.qk_head_dim as i64;
    let v_head = cfg.v_head_dim as i64;
    let kv_lora = cfg.kv_lora_rank as i64;
    let theta = cfg.rope_theta();

    // ---- query path (low-rank) ----
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
    // Interleaved RoPE for GLM.
    q_pe = nn.rope_interleaved(q_pe, qk_rope as u32, theta);
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
    k_pe = nn.rope_interleaved(k_pe, qk_rope as u32, theta);
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

    // DSA: "full" layers use standard attention; "shared" layers attend only
    // over top-k indices (modeled as attention with seq_kv = index_topk). At
    // compile time the shape difference is seq_kv; at runtime the gather op
    // fills the KV buffer with the selected positions.
    let attn = if cfg.layer_is_full_attn(layer) {
        nn.attention(q, k, value, nh, nh, qk_head as u32, true, None, None)
    } else {
        // DSA sparse layer: index scoring + top-k selection + reduced attention.
        // The indexer scores are a separate projection producing per-position
        // relevance. We model this as a linear projection (scoring head), then
        // the attention itself has seq_kv = index_topk (the runtime fills KV
        // with only the top-k entries). Shape-wise this is correct: the
        // attention output is [B,S,nh,v_head] regardless of seq_kv.
        let _index_score = nn.linear(
            &format!("{p}.self_attn.index_head"),
            x,
            h,
            cfg.index_n_heads as i64 * cfg.index_head_dim as i64,
            false,
        );
        // After top-k selection, attention runs on a reduced KV set.
        // The Attention op's shape inference uses Q's sequence for the output
        // and K's first axis for seq_kv — both are symbolic here (S), so the
        // cost model will see the full sequence. For accurate cost modeling of
        // DSA layers, we'd need a concrete `index_topk` dim. For now, emit
        // standard attention — the runtime handles the sparse dispatch.
        nn.attention(q, k, value, nh, nh, qk_head as u32, true, None, None)
    };
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

/// GLM MoE: router logits + shared expert(s) + one representative routed
/// expert, recombined. Same pattern as DeepSeek.
fn moe(nn: &mut Nn, cfg: &GlmConfig, p: &str, x: TensorId, h: i64) -> TensorId {
    // Router: [.., H] -> [.., n_routed_experts] logits.
    let _logits = nn.moe_router(
        &format!("{p}.gate"),
        x,
        h,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );

    // One representative routed expert FFN (shape-identical for all experts).
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
