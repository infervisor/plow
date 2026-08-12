//! DeepSeek-V4 decoder → symbolic operator graph.
//!
//! Ground truth is the `inference/model.py` shipped with the checkpoint, and
//! the tensor names are the checkpoint's own (`layers.N.attn.wq_a`, not the
//! `model.layers.N.self_attn.*` spelling V2/V3 use). Four pieces make this a
//! different architecture from V3 rather than a variant of it:
//!
//! * **Hyper-connections.** The hidden state is `hc_mult` parallel residual
//!   streams. Each sub-layer reduces them to one vector, runs, and writes back
//!   through a learned Sinkhorn-normalized mix — see [`Op::HcReduce`] /
//!   [`Op::HcExpand`]. There is no residual add anywhere in the model.
//! * **One KV head, sliding window, compressed history.** `wkv` projects a
//!   single `head_dim` KV vector shared by all 64 query heads. Attention sees
//!   the last `sliding_window` tokens plus, on the compressed layers, a learned
//!   pooling of everything older at rate `compress_ratios[layer]`.
//! * **A sparse indexer** on the `ratio == 4` layers, which scores the
//!   compressed entries and keeps `index_topk` of them. The scoring is modeled;
//!   the selection is data-dependent and handled at runtime, exactly as GLM's
//!   DSA layers are.
//! * **Hash-routed leading layers.** The first `num_hash_layers` MoE gates read
//!   their expert set from a frozen `[vocab, top_k]` table.
//!
//! Not modeled here: the DSpark/MTP stages under the checkpoint's `mtp.*`
//! prefix. They are a separate draft network with their own attention variant
//! and heads, and they build as their own graph — the same split vLLM makes
//! between a decoder and its MTP model. The `.scale` companions of the FP8/FP4
//! tensors are likewise absent: they are a quantization detail of one weight,
//! not separate operands. As in the V2/V3 builder, expert dispatch is runtime
//! indirection, so only `experts.0` of each layer stands in the graph.
//!
//! One thing a loader must not read literally: an [`DType::F4`] weight is
//! listed at its LOGICAL `[out, in]` width, while the checkpoint stores it
//! nibble-packed as `[out, in / 2]` — that packing is what the dtype means.
//! Against the released 43-layer checkpoint the manifest is name-exact: 1328
//! tensors in scope, none missing and none invented.

use super::config::{parse_dtype, DeepSeekV4Config};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &DeepSeekV4Config) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let hc = cfg.hc_mult;

    let b = nn.sym("B");
    let s = nn.sym("S");

    let ids = nn.input("input_ids", nn.shape([b.clone(), s.clone()]), DType::I32);
    let embed = nn.embedding("embed", ids, cfg.vocab_size, h);

    // Every stream starts as a copy of the embedding: [B,S,D] -> [B,S,hc,D].
    let stacked = nn.reshape(embed, [b.clone(), s.clone(), Dim::stat(1), Dim::stat(h)]);
    let mut x = nn.broadcast(
        stacked,
        [b.clone(), s.clone(), Dim::stat(hc as i64), Dim::stat(h)],
    );

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("layers.{layer}");
        nn.begin_block(&p);

        // --- attention sub-layer -------------------------------------------
        let residual = x;
        let reduced = nn.hc_reduce(
            &format!("{p}.hc_attn"),
            x,
            hc,
            h,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
        );
        let normed = nn.rmsnorm(&format!("{p}.attn_norm"), reduced, h, eps);
        let attn = attention(&mut nn, cfg, &p, layer, normed, &b, &s);
        x = nn.hc_expand(
            &format!("{p}.hc_attn"),
            attn,
            residual,
            hc,
            h,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
        );

        // --- FFN sub-layer --------------------------------------------------
        let residual = x;
        let reduced = nn.hc_reduce(
            &format!("{p}.hc_ffn"),
            x,
            hc,
            h,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
        );
        let normed = nn.rmsnorm(&format!("{p}.ffn_norm"), reduced, h, eps);
        let ffn = moe(&mut nn, cfg, &format!("{p}.ffn"), layer, normed, ids);
        x = nn.hc_expand(
            &format!("{p}.hc_ffn"),
            ffn,
            residual,
            hc,
            h,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
        );
    }
    nn.end_block();

    // The head reduce is a plain sigmoid gate, not a Sinkhorn mix.
    let x = nn.hc_reduce_head("hc_head", x, hc, h, cfg.hc_eps);
    let x = nn.rmsnorm("norm", x, h, eps);
    let logits = nn.linear("head", x, h, cfg.vocab_size, false);
    nn.mark_output(logits);
    nn.finish()
}

/// The compressed KV length at `ratio`, as its own size variable.
///
/// `Dim` is a polynomial over size variables with integer coefficients, so
/// `S / 4` is not something it can hold. Naming the rate gives every consumer
/// at that ratio — a layer's compressor and the indexer's — the same dim, which
/// is what makes their outputs concatenable.
fn compressed_seq(nn: &mut Nn, ratio: u32) -> Dim {
    nn.sym(&format!("Sc{ratio}"))
}

/// Rotate only the trailing `rope_dim` lanes, leaving the content lanes alone.
fn rope_tail(nn: &mut Nn, x: TensorId, head_dim: i64, rope_dim: u32, theta: f32) -> TensorId {
    let nope = head_dim - rope_dim as i64;
    let content = nn.slice(x, -1, 0, nope);
    let pe = nn.slice(x, -1, nope, rope_dim as i64);
    let pe = nn.rope(pe, rope_dim, theta);
    nn.concat(-1, vec![content, pe])
}

/// The inverse rotation applied to the attention output's rope lanes.
fn rope_tail_inverse(
    nn: &mut Nn,
    x: TensorId,
    head_dim: i64,
    rope_dim: u32,
    theta: f32,
) -> TensorId {
    let nope = head_dim - rope_dim as i64;
    let content = nn.slice(x, -1, 0, nope);
    let pe = nn.slice(x, -1, nope, rope_dim as i64);
    let pe = nn.rope_inverse(pe, rope_dim, theta);
    nn.concat(-1, vec![content, pe])
}

fn attention(
    nn: &mut Nn,
    cfg: &DeepSeekV4Config,
    p: &str,
    layer: u32,
    x: TensorId,
    b: &Dim,
    s: &Dim,
) -> TensorId {
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim as i64;
    let rd = cfg.qk_rope_head_dim;
    let q_lora = cfg.q_lora_rank as i64;
    let theta = cfg.rope_theta_for(layer);
    let ratio = cfg.compress_ratio(layer);

    // ---- query: low-rank, then per-head RMS rescale with no learned gain ----
    let qr = nn.linear_dtype(
        &format!("{p}.attn.wq_a"),
        x,
        h,
        q_lora,
        false,
        DType::F8E4M3,
    );
    let qr = nn.rmsnorm(&format!("{p}.attn.q_norm"), qr, q_lora, eps);
    let q = nn.linear_dtype(
        &format!("{p}.attn.wq_b"),
        qr,
        q_lora,
        nh as i64 * hd,
        false,
        DType::F8E4M3,
    );
    let q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );
    let q = nn.rmsnorm_weightless(q, eps);
    let q = rope_tail(nn, q, hd, rd, theta);

    // ---- the single shared KV head ----------------------------------------
    let kv = nn.linear_dtype(&format!("{p}.attn.wkv"), x, h, hd, false, DType::F8E4M3);
    let kv = nn.rmsnorm(&format!("{p}.attn.kv_norm"), kv, hd, eps);
    let kv = rope_tail(nn, kv, hd, rd, theta);
    let kv = nn.reshape(kv, [b.clone(), s.clone(), Dim::stat(1), Dim::stat(hd)]);

    // ---- compressed history, and the indexer that scores it ----------------
    let mut index_scores = None;
    let kv = match ratio {
        None => kv,
        Some(r) => {
            let overlap = DeepSeekV4Config::overlaps(r);
            let seq_c = compressed_seq(nn, r);
            let c = nn.kv_compress(
                &format!("{p}.attn.compressor"),
                x,
                h,
                r,
                hd,
                overlap,
                seq_c.clone(),
            );
            let c = rope_tail(nn, c, hd, rd, theta);
            let c = nn.reshape(c, [b.clone(), seq_c, Dim::stat(1), Dim::stat(hd)]);
            if overlap {
                // The scores decide which compressed entries the layer may
                // attend to, so they enter attention as its score mask. Only
                // the top-k selection over them is data-dependent.
                index_scores = Some(indexer(nn, cfg, p, qr, x, b, s, r, theta));
            }
            // Attention reads the window and the compressed history as one
            // key/value sequence.
            nn.concat(1, vec![kv, c])
        }
    };

    let o = nn.attention_sink(
        &format!("{p}.attn.attn_sink"),
        q,
        kv,
        kv,
        index_scores,
        nh,
        1,
        cfg.head_dim,
        true,
        Some(cfg.sliding_window),
    );
    let o = rope_tail_inverse(nn, o, hd, rd, theta);

    // ---- block-diagonal output projection ----------------------------------
    let width = cfg.o_group_width();
    let o = nn.reshape(
        o,
        [
            b.clone(),
            s.clone(),
            Dim::stat(cfg.o_groups as i64),
            Dim::stat(width),
        ],
    );
    let o = nn.grouped_linear(
        &format!("{p}.attn.wo_a"),
        o,
        cfg.o_groups,
        width,
        cfg.o_lora_rank as i64,
        DType::F8E4M3,
    );
    let flat = cfg.o_groups as i64 * cfg.o_lora_rank as i64;
    let o = nn.reshape(o, [b.clone(), s.clone(), Dim::stat(flat)]);
    nn.linear_dtype(&format!("{p}.attn.wo_b"), o, flat, h, false, DType::F8E4M3)
}

/// Scores the compressed KV entries so the runtime can keep the top
/// `index_topk`. The selection itself is data-dependent and is not a graph
/// edge, the same convention the KV cache and MoE dispatch already use.
#[allow(clippy::too_many_arguments)]
fn indexer(
    nn: &mut Nn,
    cfg: &DeepSeekV4Config,
    p: &str,
    qr: TensorId,
    x: TensorId,
    b: &Dim,
    s: &Dim,
    ratio: u32,
    theta: f32,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.index_n_heads as i64;
    let hd = cfg.index_head_dim as i64;
    let rd = cfg.qk_rope_head_dim;
    let seq_c = compressed_seq(nn, ratio);

    let q = nn.linear_dtype(
        &format!("{p}.attn.indexer.wq_b"),
        qr,
        cfg.q_lora_rank as i64,
        nh * hd,
        false,
        DType::F8E4M3,
    );
    let q = nn.reshape(q, [b.clone(), s.clone(), Dim::stat(nh), Dim::stat(hd)]);
    let q = rope_tail(nn, q, hd, rd, theta);

    // The indexer keeps its own compressor, at the index head width.
    let kv = nn.kv_compress(
        &format!("{p}.attn.indexer.compressor"),
        x,
        h,
        ratio,
        hd,
        true,
        seq_c.clone(),
    );

    // einsum("bshd,btd->bsht"): fold the head axis into the rows so this is one
    // batched matmul, then unfold it again.
    let qf = nn.reshape(q, [b.clone(), s.mul(&Dim::stat(nh)), Dim::stat(hd)]);
    let kvt = nn.transpose(kv, vec![0, 2, 1]);
    let scores = nn.matmul(qf, kvt);
    let scores = nn.reshape(scores, [b.clone(), s.clone(), Dim::stat(nh), seq_c.clone()]);
    let scores = nn.act(ActKind::Relu, scores);

    // Per-head weights fold the heads away, leaving one score per entry.
    let w = nn.linear(&format!("{p}.attn.indexer.weights_proj"), x, h, nh, false);
    let w = nn.reshape(w, [b.clone(), s.clone(), Dim::stat(nh), Dim::stat(1)]);
    let scores = nn.mul(scores, w);
    nn.reduce(scores, crate::op::ReduceKind::Sum, -2, false)
}

fn moe(
    nn: &mut Nn,
    cfg: &DeepSeekV4Config,
    p: &str,
    layer: u32,
    x: TensorId,
    ids: TensorId,
) -> TensorId {
    let h = cfg.hidden_size;
    let e = cfg.n_routed_experts;
    let k = cfg.num_experts_per_tok;

    // The leading layers do not select from the scores at all.
    let scores = if layer < cfg.num_hash_layers {
        nn.moe_router_hashed(&format!("{p}.gate"), x, ids, h, cfg.vocab_size, e, k)
    } else {
        nn.moe_router_select_bias(&format!("{p}.gate"), x, h, e, k)
    };

    // Dispatch is runtime indirection; the static graph carries one
    // representative routed expert, as the V2/V3 builder does. Its combine
    // weight is that expert's own routing score — modeled, rather than the
    // router being left dangling, so the gate weights stay reachable.
    let routed = expert(nn, cfg, &format!("{p}.experts.0"), x, DType::F4);
    let weight = nn.slice(scores, -1, 0, 1);
    let routed = nn.mul(routed, weight);
    let shared = expert(nn, cfg, &format!("{p}.shared_experts"), x, DType::F8E4M3);
    nn.add(routed, shared)
}

fn expert(nn: &mut Nn, cfg: &DeepSeekV4Config, p: &str, x: TensorId, dtype: DType) -> TensorId {
    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let gate = nn.linear_dtype(&format!("{p}.w1"), x, h, inter, false, dtype);
    let up = nn.linear_dtype(&format!("{p}.w3"), x, h, inter, false, dtype);
    let act = nn.clamped_swiglu(gate, up, cfg.swiglu_limit);
    nn.linear_dtype(&format!("{p}.w2"), act, inter, h, false, dtype)
}
