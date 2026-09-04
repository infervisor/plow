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
use crate::op::{ActKind, MoeGroups};
use crate::Nn;
use crate::{
    DType, Dim, ExpertLayerBinding, ExpertProjectionBinding, Graph, RoutedExpertBinding, TensorId,
};

pub fn build(cfg: &DeepSeekConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, DType::BF16);

    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    let b = nn.sym("B");
    let s = nn.sym("S");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let mut x = nn.embedding("model.embed_tokens", ids, cfg.vocab_size, h);

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer}");
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
                cfg,
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

    x = nn.rmsnorm("model.norm", x, h, eps);
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
        let qa = linear(nn, cfg, &format!("{p}.self_attn.q_a_proj"), x, h, q_lora);
        let qa = nn.rmsnorm(&format!("{p}.self_attn.q_a_layernorm"), qa, q_lora, eps);
        linear(
            nn,
            cfg,
            &format!("{p}.self_attn.q_b_proj"),
            qa,
            q_lora,
            nh as i64 * qk_head,
        )
    } else {
        linear(
            nn,
            cfg,
            &format!("{p}.self_attn.q_proj"),
            x,
            h,
            nh as i64 * qk_head,
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
    let kv_a = linear(
        nn,
        cfg,
        &format!("{p}.self_attn.kv_a_proj_with_mqa"),
        x,
        h,
        kv_lora + qk_rope,
    );
    let compressed = nn.slice(kv_a, -1, 0, kv_lora);
    let mut k_pe = nn.slice(kv_a, -1, kv_lora, qk_rope); // [B,S,qk_rope]
    let compressed = nn.rmsnorm(
        &format!("{p}.self_attn.kv_a_layernorm"),
        compressed,
        kv_lora,
        eps,
    );
    let kv = linear(
        nn,
        cfg,
        &format!("{p}.self_attn.kv_b_proj"),
        compressed,
        kv_lora,
        nh as i64 * (qk_nope + v_head),
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
    linear(
        nn,
        cfg,
        &format!("{p}.self_attn.o_proj"),
        merged,
        nh as i64 * v_head,
        h,
    )
}

/// SwiGLU dense MLP: `down(silu(gate(x)) * up(x))`.
fn swiglu_mlp(
    nn: &mut Nn,
    cfg: &DeepSeekConfig,
    p: &str,
    x: TensorId,
    h: i64,
    inter: i64,
) -> TensorId {
    let gate = linear(nn, cfg, &format!("{p}.gate_proj"), x, h, inter);
    let up = linear(nn, cfg, &format!("{p}.up_proj"), x, h, inter);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    linear(nn, cfg, &format!("{p}.down_proj"), hidden, inter, h)
}

/// DeepSeekMoE: router logits + shared expert(s) + one representative routed
/// expert, recombined. Per-token expert selection is a runtime indirection;
/// the graph carries the router and the FFN shapes.
fn moe(nn: &mut Nn, cfg: &DeepSeekConfig, p: &str, x: TensorId, h: i64) -> TensorId {
    // Router: [.., H] -> [.., n_routed_experts] logits.
    let routes = nn.moe_router_noaux(
        &format!("{p}.gate"),
        x,
        h,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        MoeGroups {
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
        },
        cfg.norm_topk_prob,
        cfg.routed_scaling_factor,
    );
    let routed_experts = register_experts(nn, cfg, p, h);
    nn.expert_binding(ExpertLayerBinding {
        block: p
            .strip_prefix("model.layers.")
            .and_then(|p| p.split('.').next())
            .and_then(|p| p.parse().ok())
            .unwrap_or(0),
        layer_label: p.to_string(),
        num_experts: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        scoring_func: cfg.scoring_func.clone(),
        norm_topk: cfg.norm_topk_prob,
        route_scale: cfg.routed_scaling_factor,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        correction_bias: Some(format!("{p}.gate.e_score_correction_bias")),
        routed_experts,
    });
    let routed = nn.moe_experts(
        x,
        routes,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size as u32,
        cfg.quantization_config.is_some(),
    );

    if cfg.n_shared_experts > 0 {
        let shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts as i64;
        let shared = swiglu_mlp(nn, cfg, &format!("{p}.shared_experts"), x, h, shared_inter);
        nn.add(routed, shared)
    } else {
        routed
    }
}

fn linear(
    nn: &mut Nn,
    cfg: &DeepSeekConfig,
    name: &str,
    x: TensorId,
    input: i64,
    output: i64,
) -> TensorId {
    let dtype = cfg.projection_weight_dtype();
    let out = nn.linear_dtype(name, x, input, output, false, dtype);
    if let (Some(shape), Some(quant)) = (
        cfg.fp8_scale_shape(output, input),
        cfg.quantization_config.as_ref(),
    ) {
        nn.fp8_scale_binding(
            &format!("{name}.weight"),
            &format!("{name}.weight_scale_inv"),
            shape.map(Dim::stat),
            quant.weight_block_size,
        );
    }
    out
}

fn register_experts(
    nn: &mut Nn,
    cfg: &DeepSeekConfig,
    p: &str,
    hidden: i64,
) -> Vec<RoutedExpertBinding> {
    (0..cfg.n_routed_experts)
        .map(|expert| {
            let p = format!("{p}.experts.{expert}");
            RoutedExpertBinding {
                gate: register_projection(
                    nn,
                    cfg,
                    &format!("{p}.gate_proj"),
                    hidden,
                    cfg.moe_intermediate_size,
                ),
                up: register_projection(
                    nn,
                    cfg,
                    &format!("{p}.up_proj"),
                    hidden,
                    cfg.moe_intermediate_size,
                ),
                down: register_projection(
                    nn,
                    cfg,
                    &format!("{p}.down_proj"),
                    cfg.moe_intermediate_size,
                    hidden,
                ),
            }
        })
        .collect()
}

fn register_projection(
    nn: &mut Nn,
    cfg: &DeepSeekConfig,
    name: &str,
    input: i64,
    output: i64,
) -> ExpertProjectionBinding {
    let weight = format!("{name}.weight");
    nn.param_dtype(
        &weight,
        [Dim::stat(output), Dim::stat(input)],
        cfg.projection_weight_dtype(),
    );
    let scale = cfg.fp8_scale_shape(output, input).map(|shape| {
        let scale = format!("{name}.weight_scale_inv");
        nn.fp8_scale_binding(
            &weight,
            &scale,
            shape.map(Dim::stat),
            cfg.quantization_config.as_ref().unwrap().weight_block_size,
        );
        scale
    });
    ExpertProjectionBinding { weight, scale }
}
