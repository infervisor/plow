//! Kimi-K3 (Moonshot) decoder → symbolic operator graph.
//!
//! K3 shares K2's MLA geometry and almost nothing else. Three departures shape
//! this file, and each is the kind that produces a *plausible* wrong graph if
//! it is glossed:
//!
//! 1. **Hybrid mixer.** 69 of 93 layers are KDA (linear attention with a
//!    carried recurrent state); the other 24 are MLA. The partition comes from
//!    `linear_attn_config`, per layer, never from a stride.
//! 2. **`AttnRes` block residual.** A K3 layer is not
//!    `residual + attn; residual + mlp`. Every layer mixes over a running
//!    prefix sum and a stack of snapshots, twice, with different weights each
//!    time; every `attn_res_block_size`-th layer pushes a snapshot and resets
//!    the prefix.
//! 3. **Latent MoE.** The routed experts read a projected latent, not the
//!    hidden state.
//!
//! # What this graph does and does not claim
//!
//! This is a *shape-level* frontend IR, so — exactly as [`crate::Op::Attention`]
//! does not model the KV cache — [`crate::Op::LinearAttention`] does not model
//! KDA's recurrent state and [`crate::Op::Conv1dDepthwise`] does not model the
//! short conv's carried window. Both are per-sequence runtime resources, not
//! symbolic dataflow. Modeling them as edges would also force multi-output
//! nodes, which the whole inference pass is built on not having.
//!
//! Like every other MoE builder here, the routed experts are represented by
//! ONE expert FFN: the dispatch is data-dependent, and a graph with 896 expert
//! subgraphs would describe a computation that never runs.

use super::config::ConfigError;
use super::config::{parse_dtype, K3Layer, KimiK3Config, KimiK3TextConfig};
use crate::op::{ActKind, LinearAttnKind, MoeGroups};
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

/// Build the K3 text tower.
///
/// Returns an error rather than a graph for the cases where a graph would be
/// confidently wrong: a multimodal checkpoint (this builds the text tower only)
/// and any layer partition that does not cover every layer exactly once.
pub fn build(cfg: &KimiK3Config) -> Result<Graph, ConfigError> {
    if cfg.vision_config.is_some() {
        return Err(ConfigError::Unsupported(
            "kimi_k3 carries a vision_config (MoonViT tower + projector) and this builds \
             the TEXT tower only. A text-only graph would load, run, and be silently \
             wrong on every image prompt, so it is refused rather than ignored."
                .into(),
        ));
    }
    let t = &cfg.text_config;
    let kinds = t.attn_kinds()?;
    let groups = t.moe_groups()?;

    let dt = parse_dtype(t.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = t.hidden_size;
    let eps = t.rms_norm_eps;
    let b = nn.sym("B");
    let s = nn.sym("S");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let mut x = nn.embedding("embed_tokens", ids, t.vocab_size, h);

    // THE BLOCK-RESIDUAL STATE, and why it is two variables.
    //
    // `prefix` is the running sum; `snapshots` is the stack `AttnRes` mixes
    // over. A snapshot layer pushes the layer input and RESETS the prefix, so
    // its output is `attn + ffn` rather than `hidden + attn + ffn` — a 1.0
    // relative difference from the plain wiring, which is why the reset cannot
    // be skipped. Layer 0 is a snapshot layer with an empty stack, so it skips
    // the first mix entirely.
    let mut snapshots: Vec<TensorId> = Vec::new();
    let max_snap = 8;

    for layer in 0..t.num_hidden_layers {
        let p = format!("layers.{layer}");
        nn.begin_block(&p);
        let prefix_in = x;

        // First mix: over the incoming prefix and every snapshot so far.
        let mut h_in = prefix_in;
        if !snapshots.is_empty() {
            h_in = nn.block_residual(
                &format!("{p}.self_attention_res"),
                prefix_in,
                &snapshots,
                h,
                max_snap,
            );
        }

        let is_snapshot = layer % t.attn_res_block_size == 0;
        if is_snapshot {
            snapshots.push(prefix_in);
            if snapshots.len() > max_snap as usize {
                snapshots.remove(0);
            }
        }

        // Mixer: MLA on the full-attention layers, KDA on the rest.
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), h_in, h, eps);
        let attn = match kinds[layer as usize] {
            K3Layer::Mla => mla_attention(&mut nn, t, &p, normed, &b, &s),
            K3Layer::Kda => kda_attention(&mut nn, t, &p, normed, &b, &s),
        };

        // A snapshot layer's prefix restarts at the mixer output.
        let prefix = if is_snapshot {
            attn
        } else {
            nn.add(prefix_in, attn)
        };

        // Second mix, with its OWN weights — layer 0 never reads the first pair.
        let h2 = if snapshots.is_empty() {
            prefix
        } else {
            nn.block_residual(&format!("{p}.mlp_res"), prefix, &snapshots, h, max_snap)
        };

        let normed = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), h2, h, eps);
        let ffn = if layer < t.first_k_dense_replace {
            situ_mlp(
                &mut nn,
                t,
                &format!("{p}.mlp"),
                normed,
                h,
                t.intermediate_size,
            )
        } else {
            latent_moe(&mut nn, t, &format!("{p}.mlp"), normed, h, groups)
        };
        x = nn.add(prefix, ffn);
    }
    nn.end_block();

    x = nn.rmsnorm("norm", x, h, eps);
    let logits = nn.linear("lm_head", x, h, t.vocab_size, false);
    nn.mark_output(logits);
    Ok(nn.finish())
}

/// MLA, on the 24 full-attention layers. Geometrically identical to K2's, plus
/// K3's output gate.
fn mla_attention(
    nn: &mut Nn,
    cfg: &KimiK3TextConfig,
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

    // Query, low-rank.
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
    let q = nn.concat(-1, vec![q_nope, q_pe]);

    // Key/value, shared compressed latent.
    let kv_a = nn.linear(
        &format!("{p}.self_attn.kv_a_proj_with_mqa"),
        x,
        h,
        kv_lora + qk_rope,
        false,
    );
    let compressed = nn.slice(kv_a, -1, 0, kv_lora);
    let mut k_pe = nn.slice(kv_a, -1, kv_lora, qk_rope);
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
    let value = nn.slice(kv, -1, qk_nope, v_head);

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
    let k = nn.concat(-1, vec![k_nope, k_pe]);

    let attn = nn.attention(q, k, value, nh, nh, qk_head as u32, true, None, None);
    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(nh as i64 * v_head)]);

    // K3's optional MLA output gate, before o_proj.
    let gated = if cfg.mla_use_output_gate {
        let gate = nn.linear(
            &format!("{p}.self_attn.g_proj"),
            x,
            h,
            nh as i64 * v_head,
            false,
        );
        let gate = nn.act(ActKind::Sigmoid, gate);
        nn.mul(merged, gate)
    } else {
        merged
    };

    nn.linear(
        &format!("{p}.self_attn.o_proj"),
        gated,
        nh as i64 * v_head,
        h,
        false,
    )
}

/// KDA — Kimi Delta Attention, on 69 of 93 layers.
///
/// Six independent projections off the layer norm, a depthwise causal conv on
/// each of q/k/v, a low-rank forget gate, and the gated delta-rule recurrence.
/// Mirrors `crates/devgen/src/kda.rs`, which emits the same layer as packets.
fn kda_attention(
    nn: &mut Nn,
    cfg: &KimiK3TextConfig,
    p: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let lac = &cfg.linear_attn_config;
    let h = cfg.hidden_size;
    let nh = lac.num_heads;
    let hd = lac.head_dim as i64;
    let inner = nh as i64 * hd; // 96 * 128 = 12288 on the shipped config
    let conv_k = lac.short_conv_kernel_size;
    let eps = cfg.rms_norm_eps;

    // q / k / v, each [B, S, inner], each through its own depthwise causal conv.
    let mut qkv = Vec::with_capacity(3);
    for which in ["q", "k", "v"] {
        let proj = nn.linear(&format!("{p}.self_attn.{which}_proj"), x, h, inner, false);
        let conv = nn.conv1d_depthwise(
            &format!("{p}.self_attn.{which}_conv1d"),
            proj,
            inner,
            conv_k,
            DType::F32,
        );
        qkv.push(nn.reshape(
            conv,
            [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
        ));
    }
    let (q, k, v) = (qkv[0], qkv[1], qkv[2]);

    // Forget gate, low-rank: hidden -> f_a_proj (rank r) -> f_b_proj (inner).
    // The rank is read off the checkpoint dims rather than named in config, so
    // it is derived from the declared f_a width.
    let gate_rank = lac.head_dim as i64; // f_a_proj is [rank=head_dim, hidden]
    let fa = nn.linear(&format!("{p}.self_attn.f_a_proj"), x, h, gate_rank, false);
    let fb = nn.linear(
        &format!("{p}.self_attn.f_b_proj"),
        fa,
        gate_rank,
        inner,
        false,
    );
    let gate = nn.reshape(
        fb,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );

    // beta — ONE write-strength scalar per (token, head), not per channel.
    let beta = nn.linear(&format!("{p}.self_attn.b_proj"), x, h, nh as i64, false);
    let beta = nn.act(ActKind::Sigmoid, beta);

    let a_log = nn.param_dtype(
        &format!("{p}.self_attn.A_log"),
        [Dim::stat(nh as i64)],
        DType::F32,
    );
    let dt_bias = nn.param_dtype(
        &format!("{p}.self_attn.dt_bias"),
        [Dim::stat(inner)],
        DType::F32,
    );

    // The recurrence. The [heads, head_dim, head_dim] state is a runtime
    // resource (see the module note), so it is not an input here.
    let o = nn.linear_attention(
        LinearAttnKind::KimiDelta,
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        nh,
        lac.head_dim,
    );

    // Output gate + norm, then project back to hidden.
    let o = nn.rmsnorm_dtype(&format!("{p}.self_attn.o_norm"), o, hd, eps, DType::F32);
    let g = if lac.use_full_rank_gate {
        nn.linear(&format!("{p}.self_attn.g_proj"), x, h, inner, false)
    } else {
        let ga = nn.linear(&format!("{p}.self_attn.g_a_proj"), x, h, gate_rank, false);
        nn.linear(
            &format!("{p}.self_attn.g_b_proj"),
            ga,
            gate_rank,
            inner,
            false,
        )
    };
    let g = nn.reshape(
        g,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );
    let g = nn.act(ActKind::Sigmoid, g);
    let y = nn.mul(o, g);
    let y = nn.reshape(y, [b.clone(), s.clone(), Dim::stat(inner)]);

    nn.linear(&format!("{p}.self_attn.o_proj"), y, inner, h, false)
}

/// Dense FFN with K3's `situ` GLU.
fn situ_mlp(
    nn: &mut Nn,
    cfg: &KimiK3TextConfig,
    p: &str,
    x: TensorId,
    h: i64,
    inter: i64,
) -> TensorId {
    let gate = nn.linear(&format!("{p}.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{p}.up_proj"), x, h, inter, false);
    let hidden = nn.situ_glu(
        gate,
        up,
        cfg.activation_situ_beta,
        cfg.activation_situ_linear_beta,
    );
    nn.linear(&format!("{p}.down_proj"), hidden, inter, h, false)
}

/// LATENT MoE: the routed experts run at `routed_expert_hidden_size`, not at
/// `hidden_size`.
///
/// The block projects down once, runs every routed expert at the latent width,
/// and projects back. Building the experts at `hidden_size` instead would give
/// a graph whose expert GEMMs are 2x the real K — right shape family, wrong
/// arithmetic, and it matches no checkpoint tensor.
fn latent_moe(
    nn: &mut Nn,
    cfg: &KimiK3TextConfig,
    p: &str,
    x: TensorId,
    h: i64,
    groups: Option<MoeGroups>,
) -> TensorId {
    let latent = cfg.routed_expert_hidden_size;

    // Router reads the HIDDEN state, not the latent.
    let _logits = match groups {
        Some(g) => nn.moe_router_grouped(
            &format!("{p}.gate"),
            x,
            h,
            cfg.num_experts,
            cfg.num_experts_per_token,
            g,
        ),
        None => nn.moe_router(
            &format!("{p}.gate"),
            x,
            h,
            cfg.num_experts,
            cfg.num_experts_per_token,
        ),
    };

    let down = nn.linear(&format!("{p}.routed_expert_down_proj"), x, h, latent, false);
    let routed = situ_mlp(
        nn,
        cfg,
        &format!("{p}.experts.0"),
        down,
        latent,
        cfg.moe_intermediate_size,
    );
    let routed = nn.linear(
        &format!("{p}.routed_expert_up_proj"),
        routed,
        latent,
        h,
        false,
    );

    if cfg.num_shared_experts > 0 {
        let shared_inter = cfg.moe_intermediate_size * cfg.num_shared_experts as i64;
        let shared = situ_mlp(nn, cfg, &format!("{p}.shared_experts"), x, h, shared_inter);
        nn.add(routed, shared)
    } else {
        routed
    }
}
