//! Muse Spark text decoder → symbolic operator graph (language model only; the
//! vision/audio towers are not modeled).
//!
//! Per layer: centered-RMSNorm pre-norms, GQA with weightless QK-norm (q also
//! scaled by `qk_scale_factor`), RoPE on sliding layers and NoPE on full ones,
//! a weightless per-head output norm gated by `sigmoid(gate_proj(x))`, a
//! weightless post-attention norm, an input-scaled post-FFN norm, and gated
//! residuals in place of adds. Sparse layers pool `hidden -> moe_hidden`,
//! route on the un-pooled input, and normalize every expert output before the
//! weighted combine. Logits are `T * tanh(logits * output_multiplier / T)`.

use super::config::{parse_dtype, MuseSparkConfig, MUSE_SPARK_RESIDUAL_GATE_EPS};
use crate::op::ActKind;
use crate::{
    DType, Dim, ExpertLayerBinding, ExpertProjectionBinding, Graph, Nn, RoutedExpertBinding,
    TensorId,
};

const PREFIX: &str = "model.language_model";

pub fn build(cfg: &MuseSparkConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, DType::BF16);
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let post_eps = cfg.post_norm_eps;
    let temp = cfg.residual_gate_temperature;
    let gate_eps = MUSE_SPARK_RESIDUAL_GATE_EPS;

    let b = nn.sym("B");
    let s = nn.sym("S");
    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let embed = format!("{PREFIX}.embed_tokens");
    let mut x = nn.embedding(&embed, ids, cfg.vocab_size, h);
    x = nn.rmsnorm_weightless(x, eps);

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("{PREFIX}.layers.{layer}");
        nn.begin_block(&p);

        let residual = x;
        let normed = nn.rmsnorm_zero_centered(&format!("{p}.input_layernorm"), x, h, eps);
        let attn = attention(&mut nn, cfg, &p, layer, normed, &b, &s);
        let attn = nn.rmsnorm_weightless(attn, post_eps);
        x = nn.residual_gate(
            &format!("{p}.post_attention_residual_gate"),
            residual,
            attn,
            h,
            temp,
            gate_eps,
        );

        let residual = x;
        let normed = nn.rmsnorm_zero_centered(&format!("{p}.pre_feedforward_layernorm"), x, h, eps);
        let mlp = if cfg.is_sparse(layer) {
            moe(&mut nn, cfg, &p, layer, normed)
        } else {
            swiglu_mlp(
                &mut nn,
                &format!("{p}.mlp"),
                normed,
                h,
                cfg.intermediate_size,
            )
        };
        let mlp =
            nn.rmsnorm_input_scaled(&format!("{p}.post_feedforward_layernorm"), mlp, h, post_eps);
        x = nn.residual_gate(
            &format!("{p}.post_feedforward_residual_gate"),
            residual,
            mlp,
            h,
            temp,
            gate_eps,
        );
    }
    nn.end_block();

    x = nn.rmsnorm(&format!("{PREFIX}.norm"), x, h, eps);
    let head = if cfg.tie_word_embeddings {
        embed.as_str()
    } else {
        "lm_head"
    };
    let mut logits = nn.linear(head, x, h, cfg.vocab_size, false);
    let cap = cfg.final_logit_softcapping;
    logits = nn.scale(logits, cfg.output_multiplier);
    logits = nn.scale(logits, 1.0 / cap);
    logits = nn.act(ActKind::Tanh, logits);
    logits = nn.scale(logits, cap);
    nn.mark_output(logits);
    nn.finish()
}

fn attention(
    nn: &mut Nn,
    cfg: &MuseSparkConfig,
    p: &str,
    layer: u32,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let bias = cfg.attention_bias;
    let q_dim = nh as i64 * hd;
    let kv_dim = nkv as i64 * hd;
    let heads = |n: u32| [b.clone(), s.clone(), Dim::stat(n as i64), Dim::stat(hd)];

    let q = nn.linear(&format!("{p}.self_attn.q_proj"), x, h, q_dim, bias);
    let k = nn.linear(&format!("{p}.self_attn.k_proj"), x, h, kv_dim, bias);
    let v = nn.linear(&format!("{p}.self_attn.v_proj"), x, h, kv_dim, bias);
    let gate = nn.linear(&format!("{p}.self_attn.gate_proj"), x, h, q_dim, false);
    let q = nn.reshape(q, heads(nh));
    let k = nn.reshape(k, heads(nkv));
    let v = nn.reshape(v, heads(nkv));

    let q = nn.rmsnorm_weightless(q, eps);
    let mut q = nn.scale(q, cfg.qk_scale_factor);
    let mut k = nn.rmsnorm_weightless(k, eps);
    if let Some(theta) = cfg.rope_theta(layer) {
        q = nn.rope(q, hd as u32, theta);
        k = nn.rope(k, hd as u32, theta);
    }
    let window = cfg.is_sliding(layer).then_some(cfg.sliding_window);
    let attn = nn.attention(q, k, v, nh, nkv, hd as u32, true, window, None);

    let attn = nn.rmsnorm_weightless(attn, eps);
    let attn = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    let gate = nn.act(ActKind::Sigmoid, gate);
    let attn = nn.mul(attn, gate);
    nn.linear(&format!("{p}.self_attn.o_proj"), attn, q_dim, h, bias)
}

fn swiglu_mlp(nn: &mut Nn, p: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{p}.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{p}.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{p}.down_proj"), hidden, inter, h, false)
}

fn moe(nn: &mut Nn, cfg: &MuseSparkConfig, p: &str, layer: u32, x: TensorId) -> TensorId {
    let h = cfg.hidden_size;
    let m = cfg.moe_hidden_size;
    let p = format!("{p}.mlp");
    let routes = nn.moe_router_sigmoid_f32(
        &format!("{p}.gate"),
        x,
        h,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.routed_scaling_factor,
        cfg.output_multiplier,
    );
    let routed_experts = (0..cfg.num_local_experts)
        .map(|e| {
            let e = format!("{p}.experts.{e}");
            let inter = cfg.moe_intermediate_size;
            RoutedExpertBinding {
                gate: expert_projection(nn, cfg, &format!("{e}.gate_proj"), m, inter),
                up: expert_projection(nn, cfg, &format!("{e}.up_proj"), m, inter),
                down: expert_projection(nn, cfg, &format!("{e}.down_proj"), inter, m),
            }
        })
        .collect();
    nn.expert_binding(ExpertLayerBinding {
        block: layer,
        layer_label: p.clone(),
        num_experts: cfg.num_local_experts,
        top_k: cfg.num_experts_per_tok,
        scoring_func: "sigmoid".to_string(),
        norm_topk: true,
        route_scale: cfg.routed_scaling_factor,
        n_group: 1,
        topk_group: 1,
        correction_bias: Some(format!("{p}.gate.e_score_correction_bias")),
        routed_experts,
    });

    let pooled = nn.linear(&format!("{p}.pre_expert_proj"), x, h, m, false);
    let pooled =
        nn.rmsnorm_zero_centered(&format!("{p}.pre_expert_norm"), pooled, m, cfg.rms_norm_eps);
    let routed = nn.moe_experts_post_norm(
        &format!("{p}.experts.post_expert_norm"),
        pooled,
        routes,
        m,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size as u32,
        cfg.post_norm_eps,
    );
    nn.linear(&format!("{p}.post_expert_proj"), routed, m, h, false)
}

/// Quark MXFP4 stores `weight` as packed fp4 pairs and `weight_scale` as e8m0
/// per 32-wide group; the F4 logical tensor accounts for both, so the scale is
/// named in the binding but not declared as a second tensor.
fn expert_projection(
    nn: &mut Nn,
    cfg: &MuseSparkConfig,
    module: &str,
    input: i64,
    output: i64,
) -> ExpertProjectionBinding {
    let weight = format!("{module}.weight");
    let dtype = cfg.module_weight_dtype(module);
    nn.param_dtype(&weight, [Dim::stat(output), Dim::stat(input)], dtype);
    ExpertProjectionBinding {
        scale: (dtype == DType::F4).then(|| format!("{module}.weight_scale")),
        weight,
    }
}
