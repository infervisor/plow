//! Nemotron-H language model → symbolic operator graph (`modeling_nemotron_h.py`).
//!
//! Every layer is `x + mixer(rmsnorm(x))` with one mixer per `layers_block_type`
//! entry:
//!
//! * Mamba-2: `in_proj → [z | xBC | dt]`, causal depthwise conv + SiLU on `xBC`,
//!   the SSD scan, then the gated grouped RMSNorm `w * rms_g(y * silu(z))` (gate
//!   BEFORE the norm, groups of `d_inner / n_groups`) and `out_proj`.
//! * NoPE causal GQA (`apply_rotary_pos_emb` is never called).
//! * MoE: sigmoid + correction-bias router on the normed hidden state; routed
//!   relu² experts run in the latent space between `fc1_latent_proj` and
//!   `fc2_latent_proj`; the shared relu² expert reads the hidden state.
//!
//! ModelOpt FP8 / NVFP4 linears bind their real checkpoint tensors: the weight is
//! the graph operand, the scales are in `Graph::quant_linear_bindings` (dense) or
//! the expert bindings (routed).

use super::config::{parse_dtype, NemotronHConfig, NemotronHLayer, NemotronHLinearQuant};
use crate::op::{ActKind, MoeGroups, Op};
use crate::{
    DType, Dim, ExpertLayerBinding, ExpertProjectionBinding, Graph, LinearQuantFormat, Nn,
    QuantLinearBinding, RoutedExpertBinding, TensorId,
};

pub fn build(cfg: &NemotronHConfig) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, DType::BF16);
    let p = &cfg.weight_prefix;
    let h = cfg.hidden_size;
    let eps = cfg.layer_norm_epsilon;

    let b = nn.sym("B");
    let s = nn.sym("S");
    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let embed = format!("{p}backbone.embeddings");
    let mut x = nn.embedding(&embed, ids, cfg.vocab_size, h);

    for (i, layer) in cfg.layers.iter().enumerate() {
        let lp = format!("{p}backbone.layers.{i}");
        nn.begin_block(&lp);
        let normed = nn.rmsnorm(&format!("{lp}.norm"), x, h, eps);
        let m = format!("{lp}.mixer");
        let mixed = match layer {
            NemotronHLayer::Mamba => mamba(&mut nn, cfg, &m, normed, &b, &s),
            NemotronHLayer::Attention => attention(&mut nn, cfg, &m, normed, &b, &s),
            NemotronHLayer::Moe => moe(&mut nn, cfg, &m, i as u32, normed),
        };
        x = nn.add(x, mixed);
    }
    nn.end_block();

    x = nn.rmsnorm(&format!("{p}backbone.norm_f"), x, h, eps);
    let head = if cfg.tie_word_embeddings {
        embed
    } else {
        format!("{p}lm_head")
    };
    let logits = linear(&mut nn, cfg, &head, x, h, cfg.vocab_size);
    nn.mark_output(logits);
    nn.finish()
}

fn mamba(nn: &mut Nn, cfg: &NemotronHConfig, m: &str, x: TensorId, b: &Dim, s: &Dim) -> TensorId {
    let nh = cfg.mamba_num_heads;
    let hd = cfg.mamba_head_dim;
    let (ng, n) = (cfg.n_groups, cfg.ssm_state_size);
    let d_inner = cfg.d_inner();
    let bc = cfg.bc_width();
    let conv_dim = cfg.conv_dim();
    let dt_dtype = parse_dtype(cfg.torch_dtype.as_deref());
    let bsd = |dims: [i64; 2]| [b.clone(), s.clone(), Dim::stat(dims[0]), Dim::stat(dims[1])];

    let proj = linear(
        nn,
        cfg,
        &format!("{m}.in_proj"),
        x,
        cfg.hidden_size,
        cfg.in_proj_width(),
    );
    let z = nn.slice(proj, 2, 0, d_inner);
    let xbc = nn.slice(proj, 2, d_inner, conv_dim);
    let dt = nn.slice(proj, 2, d_inner + conv_dim, nh as i64);

    let conv = format!("{m}.conv1d");
    let xbc = if cfg.use_conv_bias {
        nn.conv1d_depthwise_bias(&conv, xbc, conv_dim, cfg.conv_kernel, dt_dtype)
    } else {
        nn.conv1d_depthwise(&conv, xbc, conv_dim, cfg.conv_kernel, dt_dtype)
    };
    let xbc = nn.act(ActKind::Silu, xbc);
    let xs = nn.slice(xbc, 2, 0, d_inner);
    let bm = nn.slice(xbc, 2, d_inner, bc);
    let cm = nn.slice(xbc, 2, d_inner + bc, bc);
    let xs = nn.reshape(xs, bsd([nh as i64, hd]));
    let bm = nn.reshape(bm, bsd([ng as i64, n]));
    let cm = nn.reshape(cm, bsd([ng as i64, n]));

    let heads = [Dim::stat(nh as i64)];
    let a_log = nn.param(&format!("{m}.A_log"), heads.clone());
    let d = nn.param(&format!("{m}.D"), heads.clone());
    let dt_bias = nn.param(&format!("{m}.dt_bias"), heads);
    let y = nn.mamba2_scan(
        xs,
        dt,
        a_log,
        bm,
        cm,
        d,
        dt_bias,
        Op::Mamba2Scan {
            num_heads: nh,
            head_dim: hd as u32,
            n_groups: ng,
            state_size: n as u32,
            chunk_size: cfg.chunk_size,
            dt_limit: cfg.time_step_limit,
        },
    );
    let y = nn.reshape(y, [b.clone(), s.clone(), Dim::stat(d_inner)]);

    let gate = nn.act(ActKind::Silu, z);
    let y = nn.mul(y, gate);
    let y = nn.reshape(y, bsd([ng as i64, d_inner / ng as i64]));
    let y = nn.rmsnorm_weightless(y, cfg.layer_norm_epsilon);
    let y = nn.reshape(y, [b.clone(), s.clone(), Dim::stat(d_inner)]);
    let norm = nn.param(&format!("{m}.norm.weight"), [Dim::stat(d_inner)]);
    let y = nn.mul(y, norm);
    linear(
        nn,
        cfg,
        &format!("{m}.out_proj"),
        y,
        d_inner,
        cfg.hidden_size,
    )
}

fn attention(
    nn: &mut Nn,
    cfg: &NemotronHConfig,
    m: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let (nh, nkv, hd) = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let (q_dim, kv_dim) = (nh as i64 * hd, nkv as i64 * hd);
    let heads = |n: u32| [b.clone(), s.clone(), Dim::stat(n as i64), Dim::stat(hd)];

    let q = linear(nn, cfg, &format!("{m}.q_proj"), x, h, q_dim);
    let k = linear(nn, cfg, &format!("{m}.k_proj"), x, h, kv_dim);
    let v = linear(nn, cfg, &format!("{m}.v_proj"), x, h, kv_dim);
    let q = nn.reshape(q, heads(nh));
    let k = nn.reshape(k, heads(nkv));
    let v = nn.reshape(v, heads(nkv));
    let attn = nn.attention(q, k, v, nh, nkv, hd as u32, true, None, None);
    let attn = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    linear(nn, cfg, &format!("{m}.o_proj"), attn, q_dim, h)
}

fn moe(nn: &mut Nn, cfg: &NemotronHConfig, m: &str, layer: u32, x: TensorId) -> TensorId {
    let h = cfg.hidden_size;
    let e_in = cfg.moe_latent_size.unwrap_or(h);
    let inter = cfg.moe_intermediate_size;
    let routes = nn.moe_router_noaux(
        &format!("{m}.gate"),
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
    let routed_experts = (0..cfg.n_routed_experts)
        .map(|e| {
            let e = format!("{m}.experts.{e}");
            RoutedExpertBinding {
                gate: None,
                up: expert_projection(nn, cfg, &format!("{e}.up_proj"), e_in, inter),
                down: expert_projection(nn, cfg, &format!("{e}.down_proj"), inter, e_in),
            }
        })
        .collect();
    nn.expert_binding(ExpertLayerBinding {
        block: layer,
        layer_label: m.to_string(),
        num_experts: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        scoring_func: "sigmoid".to_string(),
        norm_topk: cfg.norm_topk_prob,
        route_scale: cfg.routed_scaling_factor,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        correction_bias: Some(format!("{m}.gate.e_score_correction_bias")),
        routed_experts,
    });

    let latent = match cfg.moe_latent_size {
        Some(l) => linear(nn, cfg, &format!("{m}.fc1_latent_proj"), x, h, l),
        None => x,
    };
    let routed = nn.moe_experts_relu2(
        latent,
        routes,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        inter as u32,
    );
    let routed = match cfg.moe_latent_size {
        Some(l) => linear(nn, cfg, &format!("{m}.fc2_latent_proj"), routed, l, h),
        None => routed,
    };
    let si = cfg.moe_shared_expert_intermediate_size;
    let up = linear(nn, cfg, &format!("{m}.shared_experts.up_proj"), x, h, si);
    let up = nn.act(ActKind::Relu, up);
    let up = nn.mul(up, up);
    let shared = linear(nn, cfg, &format!("{m}.shared_experts.down_proj"), up, si, h);
    nn.add(routed, shared)
}

/// The weight tensor of `module` in its checkpoint storage, plus its scales.
fn quant_weight(
    nn: &mut Nn,
    cfg: &NemotronHConfig,
    module: &str,
    input: i64,
    output: i64,
) -> (TensorId, Option<QuantLinearBinding>) {
    let weight = format!("{module}.weight");
    let (o, i) = (Dim::stat(output), Dim::stat(input));
    // ModelOpt 0.37 stores the FP8 `input_scale` as [1] and every other scalar
    // scale as rank 0 (checked against the safetensors headers).
    let scalar = |nn: &mut Nn, suffix: &str, dims: Vec<Dim>| {
        let name = format!("{module}.{suffix}");
        nn.param_dtype(&name, dims, DType::F32);
        name
    };
    match cfg.linear_quant(module) {
        NemotronHLinearQuant::Bf16 => (nn.param(&weight, [o, i]), None),
        NemotronHLinearQuant::Fp8 => {
            let w = nn.param_dtype(&weight, [o, i], DType::F8E4M3);
            let binding = QuantLinearBinding {
                format: LinearQuantFormat::Fp8PerTensor,
                weight_scale: scalar(nn, "weight_scale", vec![]),
                weight_scale_2: None,
                input_scale: Some(scalar(nn, "input_scale", vec![Dim::stat(1)])),
                weight,
            };
            (w, Some(binding))
        }
        NemotronHLinearQuant::Nvfp4 { group } => {
            let w = nn.param_dtype(&weight, [o.clone(), Dim::stat(input / 2)], DType::U8);
            let weight_scale = format!("{module}.weight_scale");
            nn.param_dtype(
                &weight_scale,
                [o, Dim::stat(input / group as i64)],
                DType::F8E4M3,
            );
            let binding = QuantLinearBinding {
                format: LinearQuantFormat::Nvfp4 { group },
                weight_scale,
                weight_scale_2: Some(scalar(nn, "weight_scale_2", vec![])),
                input_scale: Some(scalar(nn, "input_scale", vec![])),
                weight,
            };
            (w, Some(binding))
        }
    }
}

fn linear(
    nn: &mut Nn,
    cfg: &NemotronHConfig,
    module: &str,
    x: TensorId,
    input: i64,
    output: i64,
) -> TensorId {
    let (w, binding) = quant_weight(nn, cfg, module, input, output);
    if let Some(binding) = binding {
        nn.quant_linear_binding(binding);
    }
    nn.linear_weight(x, w, output)
}

fn expert_projection(
    nn: &mut Nn,
    cfg: &NemotronHConfig,
    module: &str,
    input: i64,
    output: i64,
) -> ExpertProjectionBinding {
    let (_, binding) = quant_weight(nn, cfg, module, input, output);
    match binding {
        None => ExpertProjectionBinding {
            weight: format!("{module}.weight"),
            scale: None,
            scale_2: None,
            input_scale: None,
        },
        Some(q) => ExpertProjectionBinding {
            weight: q.weight,
            scale: Some(q.weight_scale),
            scale_2: q.weight_scale_2,
            input_scale: q.input_scale,
        },
    }
}
