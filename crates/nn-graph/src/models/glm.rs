//! GLM-MoE-DSA decoder graph.

use super::config::{parse_dtype, GlmConfig};
use crate::op::{ActKind, MoeGroups};
use crate::{
    DType, Dim, ExpertLayerBinding, ExpertProjectionBinding, Graph, Nn, RoutedExpertBinding,
    TensorId,
};

pub fn build(cfg: &GlmConfig) -> Graph {
    let mut nn = Nn::new(parse_dtype(cfg.torch_dtype.as_deref()), DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let embedding = nn.embedding("model.embed_tokens", ids, cfg.vocab_size, cfg.hidden_size);
    let mut x = embedding;
    let mut topk = None;

    for layer in 0..cfg.num_hidden_layers {
        let (next, next_topk) = decoder_layer(&mut nn, cfg, layer, x, topk, &b, &s);
        x = next;
        topk = Some(next_topk);
    }

    let base_hidden = x;
    let normed = nn.rmsnorm("model.norm", x, cfg.hidden_size, cfg.rms_norm_eps);
    let logits = nn.linear("lm_head", normed, cfg.hidden_size, cfg.vocab_size, false);
    nn.mark_output(logits);

    if cfg.num_nextn_predict_layers == 1 {
        let layer = cfg.num_hidden_layers;
        let p = format!("model.layers.{layer}");
        let e = nn.rmsnorm(
            &format!("{p}.enorm"),
            embedding,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        );
        let h = nn.rmsnorm(
            &format!("{p}.hnorm"),
            base_hidden,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        );
        let eh = nn.concat(-1, vec![e, h]);
        // The released layer-78 eh projection is BF16 (there is no scale tensor).
        let x = nn.linear_dtype(
            &format!("{p}.eh_proj"),
            eh,
            cfg.hidden_size * 2,
            cfg.hidden_size,
            false,
            DType::BF16,
        );
        let (x, _) = decoder_layer(&mut nn, cfg, layer, x, None, &b, &s);
        let x = nn.rmsnorm(
            &format!("{p}.shared_head.norm"),
            x,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        );
        let mtp_logits = nn.linear("lm_head", x, cfg.hidden_size, cfg.vocab_size, false);
        nn.mark_output(mtp_logits);
    }

    nn.finish()
}

fn decoder_layer(
    nn: &mut Nn,
    cfg: &GlmConfig,
    layer: u32,
    x: TensorId,
    previous_topk: Option<TensorId>,
    b: &Dim,
    s: &Dim,
) -> (TensorId, TensorId) {
    let p = format!("model.layers.{layer}");
    nn.begin_block(&p);
    let residual = x;
    let normed = nn.rmsnorm(
        &format!("{p}.input_layernorm"),
        x,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    );
    let (attn, topk) = mla_dsa(nn, cfg, &p, normed, previous_topk, b, s, layer);
    let x = nn.add(residual, attn);

    let residual = x;
    let normed = nn.rmsnorm(
        &format!("{p}.post_attention_layernorm"),
        x,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    );
    let ffn = if layer < cfg.num_hidden_layers && cfg.layer_is_dense(layer) {
        swiglu_fp8(nn, cfg, &format!("{p}.mlp"), normed, cfg.intermediate_size)
    } else {
        moe(nn, cfg, layer, &format!("{p}.mlp"), normed)
    };
    let x = nn.add(residual, ffn);
    nn.end_block();
    (x, topk)
}

#[allow(clippy::too_many_arguments)]
fn mla_dsa(
    nn: &mut Nn,
    cfg: &GlmConfig,
    p: &str,
    x: TensorId,
    previous_topk: Option<TensorId>,
    b: &Dim,
    s: &Dim,
    layer: u32,
) -> (TensorId, TensorId) {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let ql = cfg.q_lora_rank as i64;
    let kl = cfg.kv_lora_rank as i64;
    let qk_nope = cfg.qk_nope_head_dim as i64;
    let qk_rope = cfg.qk_rope_head_dim as i64;
    let qk = cfg.qk_head_dim as i64;
    let vd = cfg.v_head_dim as i64;

    let qa = fp8_linear(nn, cfg, &format!("{p}.self_attn.q_a_proj"), x, h, ql);
    let q_resid = nn.rmsnorm(
        &format!("{p}.self_attn.q_a_layernorm"),
        qa,
        ql,
        cfg.rms_norm_eps,
    );
    let q = fp8_linear(
        nn,
        cfg,
        &format!("{p}.self_attn.q_b_proj"),
        q_resid,
        ql,
        nh as i64 * qk,
    );
    let q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(qk)],
    );
    let q_nope = nn.slice(q, -1, 0, qk_nope);
    let q_rope = nn.slice(q, -1, qk_nope, qk_rope);
    let q_rope = nn.rope_interleaved_with_frequency_dim(
        q_rope,
        qk_rope as u32,
        cfg.rope_theta(),
        cfg.head_dim,
    );
    let q = nn.concat(-1, vec![q_nope, q_rope]);

    let kva = fp8_linear(
        nn,
        cfg,
        &format!("{p}.self_attn.kv_a_proj_with_mqa"),
        x,
        h,
        kl + qk_rope,
    );
    let compressed = nn.slice(kva, -1, 0, kl);
    let k_rope = nn.slice(kva, -1, kl, qk_rope);
    let compressed = nn.rmsnorm(
        &format!("{p}.self_attn.kv_a_layernorm"),
        compressed,
        kl,
        cfg.rms_norm_eps,
    );
    let kv = fp8_linear(
        nn,
        cfg,
        &format!("{p}.self_attn.kv_b_proj"),
        compressed,
        kl,
        nh as i64 * (qk_nope + vd),
    );
    let kv = nn.reshape(
        kv,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(qk_nope + vd),
        ],
    );
    let k_nope = nn.slice(kv, -1, 0, qk_nope);
    let v = nn.slice(kv, -1, qk_nope, vd);
    let k_rope = nn.reshape(
        k_rope,
        [b.clone(), s.clone(), Dim::stat(1), Dim::stat(qk_rope)],
    );
    let k_rope = nn.rope_interleaved_with_frequency_dim(
        k_rope,
        qk_rope as u32,
        cfg.rope_theta(),
        cfg.head_dim,
    );
    let k_rope = nn.broadcast(
        k_rope,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(qk_rope),
        ],
    );
    let k = nn.concat(-1, vec![k_nope, k_rope]);

    let computes_index = layer == cfg.num_hidden_layers || cfg.layer_computes_index(layer);
    let topk = if computes_index {
        dsa_indexer(nn, cfg, p, x, q_resid)
    } else {
        previous_topk.expect("validated GLM shared indexer has a prior full layer")
    };
    let attn = nn.dsa_attention(
        q,
        k,
        v,
        topk,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.qk_head_dim,
        cfg.index_topk,
    );
    let attn = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(nh as i64 * vd)]);
    let out = fp8_linear(
        nn,
        cfg,
        &format!("{p}.self_attn.o_proj"),
        attn,
        nh as i64 * vd,
        h,
    );
    (out, topk)
}

fn dsa_indexer(nn: &mut Nn, cfg: &GlmConfig, p: &str, x: TensorId, q: TensorId) -> TensorId {
    let p = format!("{p}.self_attn.indexer");
    let wq_name = format!("{p}.wq_b.weight");
    let wk_name = format!("{p}.wk.weight");
    let wq = fp8_param(
        nn,
        cfg,
        &wq_name,
        cfg.index_n_heads as i64 * cfg.index_head_dim as i64,
        cfg.q_lora_rank as i64,
    );
    let wk = fp8_param(
        nn,
        cfg,
        &wk_name,
        cfg.index_head_dim as i64,
        cfg.hidden_size,
    );
    let k_norm_w = nn.param_dtype(
        &format!("{p}.k_norm.weight"),
        [Dim::stat(cfg.index_head_dim as i64)],
        DType::BF16,
    );
    let k_norm_b = nn.param_dtype(
        &format!("{p}.k_norm.bias"),
        [Dim::stat(cfg.index_head_dim as i64)],
        DType::BF16,
    );
    let weights = nn.param_dtype(
        &format!("{p}.weights_proj.weight"),
        [
            Dim::stat(cfg.index_n_heads as i64),
            Dim::stat(cfg.hidden_size),
        ],
        DType::BF16,
    );
    nn.dsa_indexer(
        x,
        q,
        wq,
        wk,
        k_norm_w,
        k_norm_b,
        weights,
        cfg.index_n_heads,
        cfg.index_head_dim,
        cfg.qk_rope_head_dim,
        cfg.index_topk,
        cfg.rope_theta(),
    )
}

fn moe(nn: &mut Nn, cfg: &GlmConfig, layer: u32, p: &str, x: TensorId) -> TensorId {
    let routes = nn.moe_router_noaux(
        &format!("{p}.gate"),
        x,
        cfg.hidden_size,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        MoeGroups {
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
        },
        cfg.norm_topk_prob,
        cfg.routed_scaling_factor,
    );

    let mut routed_experts = Vec::with_capacity(cfg.n_routed_experts as usize);
    for expert in 0..cfg.n_routed_experts {
        let ep = format!("{p}.experts.{expert}");
        let gate = expert_projection(
            nn,
            cfg,
            &format!("{ep}.gate_proj.weight"),
            cfg.moe_intermediate_size,
            cfg.hidden_size,
        );
        let up = expert_projection(
            nn,
            cfg,
            &format!("{ep}.up_proj.weight"),
            cfg.moe_intermediate_size,
            cfg.hidden_size,
        );
        let down = expert_projection(
            nn,
            cfg,
            &format!("{ep}.down_proj.weight"),
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        );
        routed_experts.push(RoutedExpertBinding { gate, up, down });
    }
    nn.expert_binding(ExpertLayerBinding {
        block: layer,
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
        true,
    );
    let shared = swiglu_fp8(
        nn,
        cfg,
        &format!("{p}.shared_experts"),
        x,
        cfg.moe_intermediate_size * cfg.n_shared_experts as i64,
    );
    nn.add(routed, shared)
}

fn swiglu_fp8(nn: &mut Nn, cfg: &GlmConfig, p: &str, x: TensorId, inter: i64) -> TensorId {
    let gate = fp8_linear(
        nn,
        cfg,
        &format!("{p}.gate_proj"),
        x,
        cfg.hidden_size,
        inter,
    );
    let up = fp8_linear(nn, cfg, &format!("{p}.up_proj"), x, cfg.hidden_size, inter);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    fp8_linear(
        nn,
        cfg,
        &format!("{p}.down_proj"),
        hidden,
        inter,
        cfg.hidden_size,
    )
}

fn fp8_linear(
    nn: &mut Nn,
    cfg: &GlmConfig,
    name: &str,
    x: TensorId,
    in_features: i64,
    out_features: i64,
) -> TensorId {
    let out = nn.linear_dtype(name, x, in_features, out_features, false, DType::F8E4M3);
    nn.fp8_scale_binding(
        &format!("{name}.weight"),
        &format!("{name}.weight_scale_inv"),
        cfg.fp8_scale_shape(out_features, in_features)
            .map(Dim::stat),
        cfg.quantization_config.weight_block_size,
    );
    out
}

fn fp8_param(
    nn: &mut Nn,
    cfg: &GlmConfig,
    name: &str,
    out_features: i64,
    in_features: i64,
) -> TensorId {
    let weight = nn.param_dtype(
        name,
        [Dim::stat(out_features), Dim::stat(in_features)],
        DType::F8E4M3,
    );
    nn.fp8_scale_binding(
        name,
        &format!("{name}_scale_inv"),
        cfg.fp8_scale_shape(out_features, in_features)
            .map(Dim::stat),
        cfg.quantization_config.weight_block_size,
    );
    weight
}

fn expert_projection(
    nn: &mut Nn,
    cfg: &GlmConfig,
    weight: &str,
    out_features: i64,
    in_features: i64,
) -> ExpertProjectionBinding {
    nn.param_dtype(
        weight,
        [Dim::stat(out_features), Dim::stat(in_features)],
        DType::F8E4M3,
    );
    let scale = format!("{weight}_scale_inv");
    nn.fp8_scale_binding(
        weight,
        &scale,
        cfg.fp8_scale_shape(out_features, in_features)
            .map(Dim::stat),
        cfg.quantization_config.weight_block_size,
    );
    ExpertProjectionBinding {
        weight: weight.to_string(),
        scale: Some(scale),
    }
}
