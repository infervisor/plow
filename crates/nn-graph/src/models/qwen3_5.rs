use super::config::{parse_dtype, Qwen35Config};
use crate::op::{ActKind, LinearAttnKind};
use crate::{DType, Dim, Graph, Nn, TensorId};

pub fn build(cfg: &Qwen35Config) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let h = cfg.hidden_size;
    let model_prefix = cfg.weight_prefix.as_deref().unwrap_or("model");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let mut x = nn.embedding(
        &format!("{model_prefix}.embed_tokens"),
        ids,
        cfg.vocab_size,
        h,
    );

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("{model_prefix}.layers.{layer}");
        nn.begin_block(&p);

        let residual = x;
        let normed =
            nn.rmsnorm_zero_centered(&format!("{p}.input_layernorm"), x, h, cfg.rms_norm_eps);
        let mixed = match cfg.layer_types.get(layer as usize).map(String::as_str) {
            Some("linear_attention") => linear_attention(&mut nn, cfg, &p, normed, &b, &s),
            Some("full_attention") => full_attention(&mut nn, cfg, &p, normed, &b, &s),
            other => panic!("unsupported Qwen3.5 layer type at layer {layer}: {other:?}"),
        };
        x = nn.add(residual, mixed);

        let residual = x;
        let normed = nn.rmsnorm_zero_centered(
            &format!("{p}.post_attention_layernorm"),
            x,
            h,
            cfg.rms_norm_eps,
        );
        let mlp = swiglu_mlp(&mut nn, cfg, &p, normed, h, cfg.intermediate_size);
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    x = nn.rmsnorm_zero_centered(&format!("{model_prefix}.norm"), x, h, cfg.rms_norm_eps);
    let logits = if cfg.tie_word_embeddings {
        nn.linear(
            &format!("{model_prefix}.embed_tokens"),
            x,
            h,
            cfg.vocab_size,
            false,
        )
    } else {
        linear(&mut nn, cfg, "lm_head", x, h, cfg.vocab_size, false)
    };
    nn.mark_output(logits);
    nn.finish()
}

fn full_attention(
    nn: &mut Nn,
    cfg: &Qwen35Config,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let q_dim = nh as i64 * hd;
    let kv_dim = nkv as i64 * hd;

    let q_and_gate = linear(
        nn,
        cfg,
        &format!("{p}.self_attn.q_proj"),
        x,
        h,
        2 * q_dim,
        cfg.attention_bias,
    );
    let q_and_gate = nn.reshape(
        q_and_gate,
        [
            b.clone(),
            s.clone(),
            Dim::stat(nh as i64),
            Dim::stat(2 * hd),
        ],
    );
    let q = nn.slice(q_and_gate, 3, 0, hd);
    let gate = nn.slice(q_and_gate, 3, hd, hd);
    let k = linear(
        nn,
        cfg,
        &format!("{p}.self_attn.k_proj"),
        x,
        h,
        kv_dim,
        cfg.attention_bias,
    );
    let v = linear(
        nn,
        cfg,
        &format!("{p}.self_attn.v_proj"),
        x,
        h,
        kv_dim,
        cfg.attention_bias,
    );
    let k = nn.reshape(
        k,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );
    let v = nn.reshape(
        v,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );

    let q = nn.rmsnorm_zero_centered(&format!("{p}.self_attn.q_norm"), q, hd, cfg.rms_norm_eps);
    let k = nn.rmsnorm_zero_centered(&format!("{p}.self_attn.k_norm"), k, hd, cfg.rms_norm_eps);
    let q = nn.rope(q, cfg.rotary_dim(), cfg.rope_parameters.rope_theta);
    let k = nn.rope(k, cfg.rotary_dim(), cfg.rope_parameters.rope_theta);
    let mixed = nn.attention(q, k, v, nh, nkv, hd as u32, true, None, None);
    let mixed = nn.reshape(mixed, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    let gate = nn.reshape(gate, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    let gate = nn.act(ActKind::Sigmoid, gate);
    let mixed = if cfg.attn_output_gate {
        nn.mul(mixed, gate)
    } else {
        mixed
    };
    linear(
        nn,
        cfg,
        &format!("{p}.self_attn.o_proj"),
        mixed,
        q_dim,
        h,
        cfg.attention_bias,
    )
}

fn linear_attention(
    nn: &mut Nn,
    cfg: &Qwen35Config,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let nkh = cfg.linear_num_key_heads;
    let nvh = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let key_dim = nkh as i64 * khd;
    let value_dim = nvh as i64 * vhd;
    let conv_dim = 2 * key_dim + value_dim;
    let p = format!("{p}.linear_attn");

    let qkv = linear(nn, cfg, &format!("{p}.in_proj_qkv"), x, h, conv_dim, false);
    let qkv = nn.conv1d_depthwise(
        &format!("{p}.conv1d"),
        qkv,
        conv_dim,
        cfg.linear_conv_kernel_dim,
        parse_dtype(cfg.torch_dtype.as_deref()),
    );
    let qkv = nn.act(ActKind::Silu, qkv);
    let q = nn.slice(qkv, 2, 0, key_dim);
    let k = nn.slice(qkv, 2, key_dim, key_dim);
    let v = nn.slice(qkv, 2, 2 * key_dim, value_dim);
    let q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nkh as i64), Dim::stat(khd)],
    );
    let k = nn.reshape(
        k,
        [b.clone(), s.clone(), Dim::stat(nkh as i64), Dim::stat(khd)],
    );
    let v = nn.reshape(
        v,
        [b.clone(), s.clone(), Dim::stat(nvh as i64), Dim::stat(vhd)],
    );

    let repeats = nvh / nkh;
    let repeat_qk = |nn: &mut Nn, tensor| {
        let tensor = nn.reshape(
            tensor,
            [
                b.clone(),
                s.clone(),
                Dim::stat(nkh as i64),
                Dim::stat(1),
                Dim::stat(khd),
            ],
        );
        let tensor = nn.broadcast(
            tensor,
            [
                b.clone(),
                s.clone(),
                Dim::stat(nkh as i64),
                Dim::stat(repeats as i64),
                Dim::stat(khd),
            ],
        );
        nn.reshape(
            tensor,
            [b.clone(), s.clone(), Dim::stat(nvh as i64), Dim::stat(khd)],
        )
    };
    let q = repeat_qk(nn, q);
    let k = repeat_qk(nn, k);

    let z = linear(nn, cfg, &format!("{p}.in_proj_z"), x, h, value_dim, false);
    let z = nn.reshape(
        z,
        [b.clone(), s.clone(), Dim::stat(nvh as i64), Dim::stat(vhd)],
    );
    let beta = linear(nn, cfg, &format!("{p}.in_proj_b"), x, h, nvh as i64, false);
    let beta = nn.act(ActKind::Sigmoid, beta);
    let gate = linear(nn, cfg, &format!("{p}.in_proj_a"), x, h, nvh as i64, false);
    let a_log = nn.param(&format!("{p}.A_log"), [Dim::stat(nvh as i64)]);
    let dt_bias = nn.param(&format!("{p}.dt_bias"), [Dim::stat(nvh as i64)]);

    let mixed = nn.linear_attention(
        LinearAttnKind::QwenGatedDelta,
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        nvh,
        khd as u32,
    );
    let mixed = nn.rmsnorm(&format!("{p}.norm"), mixed, vhd, cfg.rms_norm_eps);
    let z = nn.act(ActKind::Silu, z);
    let mixed = nn.mul(mixed, z);
    let mixed = nn.reshape(mixed, [b.clone(), s.clone(), Dim::stat(value_dim)]);
    linear(
        nn,
        cfg,
        &format!("{p}.out_proj"),
        mixed,
        value_dim,
        h,
        false,
    )
}

fn swiglu_mlp(
    nn: &mut Nn,
    cfg: &Qwen35Config,
    p: &str,
    x: TensorId,
    h: i64,
    inter: i64,
) -> TensorId {
    let gate = linear(nn, cfg, &format!("{p}.mlp.gate_proj"), x, h, inter, false);
    let up = linear(nn, cfg, &format!("{p}.mlp.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::Silu, gate);
    let hidden = nn.mul(gate, up);
    linear(
        nn,
        cfg,
        &format!("{p}.mlp.down_proj"),
        hidden,
        inter,
        h,
        false,
    )
}

fn linear(
    nn: &mut Nn,
    cfg: &Qwen35Config,
    name: &str,
    x: TensorId,
    in_features: i64,
    out_features: i64,
    bias: bool,
) -> TensorId {
    let dtype = cfg.projection_weight_dtype(name);
    let out = nn.linear_dtype(name, x, in_features, out_features, bias, dtype);
    if dtype == DType::F8E4M3 {
        let shape = cfg
            .fp8_scale_shape(out_features, in_features)
            .expect("FP8 projection has a block scale shape");
        nn.fp8_scale_binding(
            &format!("{name}.weight"),
            &format!("{name}.weight_scale_inv"),
            shape.map(Dim::stat),
            cfg.quantization_config
                .as_ref()
                .expect("FP8 projection has quantization metadata")
                .weight_block_size,
        );
    }
    out
}
