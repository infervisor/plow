//! DeepSeek-V4 (`deepseek_v4`) front-end claim and capability report.
//!
//! V4 has no emitter. This module exists so that saying so is *accurate and
//! actionable* rather than a panic naming the wrong architecture, and so the
//! claim happens before any arm that would parse it as something else.
//!
//! That risk is not hypothetical. V4 spells `q_lora_rank`, `n_routed_experts`,
//! `num_experts_per_tok` and `moe_intermediate_size` exactly as DeepSeek-V3
//! does, so a `starts_with("deepseek")` arm — or anyone adding one — parses
//! this config, finds every key it looks for, and emits a blob for a model that
//! shares none of V4's dataflow: no `kv_lora_rank`, no residual add, one KV
//! head instead of MLA's latent, a learned KV compressor, and FP4 experts. That
//! is the Kimi-K3-as-K2 failure with a different pair of names, and the fix is
//! the same one: claim the `model_type` explicitly and refuse loudly.
//!
//! The report follows `kimi_k3::kimi_k3_emit`: geometry, config-vs-tensor
//! agreement (tensors win), what already exists so it is not rebuilt, and a
//! ranked gap list.

use super::kimi_k3::{k3_shard_headers, textwrap72};
use serde_json::Value;
use std::path::Path;

pub(crate) struct V4Cfg {
    pub layers: u32,
    pub hidden: i64,
    pub heads: u32,
    pub head_dim: u32,
    pub rope_head_dim: u32,
    pub q_lora: i64,
    pub o_groups: u32,
    pub o_lora: i64,
    pub vocab: i64,
    pub window: u32,
    pub compress_ratios: Vec<u32>,
    pub rope_theta: f64,
    pub compress_rope_theta: f64,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub n_exp: u32,
    pub shared_exp: u32,
    pub top_k: u32,
    pub moe_inter: i64,
    pub hash_layers: u32,
    pub swiglu_limit: f64,
    pub score_func: String,
    pub route_scale: f64,
    pub hc_mult: u32,
    pub hc_iters: u32,
    pub expert_dtype: String,
    pub quant_method: String,
    pub quant_block: Vec<i64>,
    pub scale_fmt: String,
    pub mtp_layers: u32,
    pub dspark_block: u32,
    pub dspark_targets: Vec<u32>,
}

impl V4Cfg {
    /// Layers whose KV history is compressed at all.
    pub fn n_compressed(&self) -> usize {
        self.compress_ratios
            .iter()
            .take(self.layers as usize)
            .filter(|&&r| r != 0)
            .count()
    }

    /// Layers that additionally run the sparse indexer (`ratio == 4`).
    pub fn n_indexed(&self) -> usize {
        self.compress_ratios
            .iter()
            .take(self.layers as usize)
            .filter(|&&r| r == 4)
            .count()
    }

    pub fn nope_head_dim(&self) -> u32 {
        self.head_dim - self.rope_head_dim
    }

    /// Rows of a per-layer hyper-connection projection.
    pub fn hc_rows(&self) -> i64 {
        (2 + self.hc_mult as i64) * self.hc_mult as i64
    }
}

pub(crate) fn cfg_deepseek_v4(dir: &Path) -> V4Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config.json is not valid JSON");
    let u = |k: &str| -> u32 {
        v[k].as_u64()
            .unwrap_or_else(|| panic!("deepseek_v4: config.json is missing `{k}`")) as u32
    };
    let i = |k: &str| -> i64 {
        v[k].as_i64()
            .unwrap_or_else(|| panic!("deepseek_v4: config.json is missing `{k}`"))
    };
    let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
    let s = |k: &str, d: &str| v[k].as_str().unwrap_or(d).to_string();
    let q = &v["quantization_config"];

    V4Cfg {
        layers: u("num_hidden_layers"),
        hidden: i("hidden_size"),
        heads: u("num_attention_heads"),
        head_dim: u("head_dim"),
        rope_head_dim: u("qk_rope_head_dim"),
        q_lora: i("q_lora_rank"),
        o_groups: u("o_groups"),
        o_lora: i("o_lora_rank"),
        vocab: i("vocab_size"),
        window: u("sliding_window"),
        compress_ratios: v["compress_ratios"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect()
            })
            .unwrap_or_default(),
        rope_theta: f("rope_theta", 10_000.0),
        compress_rope_theta: f("compress_rope_theta", 0.0),
        index_heads: u("index_n_heads"),
        index_head_dim: u("index_head_dim"),
        index_topk: u("index_topk"),
        n_exp: u("n_routed_experts"),
        shared_exp: u("n_shared_experts"),
        top_k: u("num_experts_per_tok"),
        moe_inter: i("moe_intermediate_size"),
        hash_layers: u("num_hash_layers"),
        swiglu_limit: f("swiglu_limit", 0.0),
        score_func: s("scoring_func", "?"),
        route_scale: f("routed_scaling_factor", 1.0),
        hc_mult: u("hc_mult"),
        hc_iters: u("hc_sinkhorn_iters"),
        expert_dtype: s("expert_dtype", "bf16"),
        quant_method: q["quant_method"].as_str().unwrap_or("none").to_string(),
        quant_block: q["weight_block_size"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
            .unwrap_or_default(),
        scale_fmt: q["scale_fmt"].as_str().unwrap_or("?").to_string(),
        mtp_layers: v["num_nextn_predict_layers"].as_u64().unwrap_or(0) as u32,
        dspark_block: v["dspark_block_size"].as_u64().unwrap_or(0) as u32,
        dspark_targets: v["dspark_target_layer_ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Cross-check config-derived shapes against the shard headers. **Tensors win.**
///
/// Same discipline as `k3_config_vs_tensors`, and it has already earned its keep
/// here: `num_nextn_predict_layers` in this checkpoint's `config.json` says 1
/// while the tensors carry three `mtp.*` stages.
fn v4_config_vs_tensors(
    c: &V4Cfg,
    h: &std::collections::BTreeMap<String, (String, Vec<i64>)>,
) -> Vec<String> {
    let mut errs = Vec::new();
    let mut check = |name: String, want: Vec<i64>| {
        if let Some((_, got)) = h.get(&name) {
            if *got != want {
                errs.push(format!(
                    "{name}: config implies {want:?}, tensor is {got:?}"
                ));
            }
        }
    };

    let d = c.hidden;
    check("embed.weight".into(), vec![c.vocab, d]);
    check("head.weight".into(), vec![c.vocab, d]);
    check("norm.weight".into(), vec![d]);
    check(
        "hc_head_fn".into(),
        vec![c.hc_mult as i64, c.hc_mult as i64 * d],
    );

    for l in 0..c.layers {
        check(format!("layers.{l}.attn.wq_a.weight"), vec![c.q_lora, d]);
        check(
            format!("layers.{l}.attn.wq_b.weight"),
            vec![c.heads as i64 * c.head_dim as i64, c.q_lora],
        );
        check(
            format!("layers.{l}.attn.wkv.weight"),
            vec![c.head_dim as i64, d],
        );
        check(
            format!("layers.{l}.attn.wo_a.weight"),
            vec![
                c.o_groups as i64 * c.o_lora,
                c.heads as i64 * c.head_dim as i64 / c.o_groups as i64,
            ],
        );
        check(
            format!("layers.{l}.attn.wo_b.weight"),
            vec![d, c.o_groups as i64 * c.o_lora],
        );
        check(format!("layers.{l}.attn.attn_sink"), vec![c.heads as i64]);
        check(
            format!("layers.{l}.hc_attn_fn"),
            vec![c.hc_rows(), c.hc_mult as i64 * d],
        );
        check(format!("layers.{l}.hc_ffn_base"), vec![c.hc_rows()]);
        check(
            format!("layers.{l}.ffn.gate.weight"),
            vec![c.n_exp as i64, d],
        );

        // The compressor projects twice its output width exactly on the
        // overlapping (ratio == 4) layers.
        if let Some(&r) = c.compress_ratios.get(l as usize) {
            if r != 0 {
                let wide = if r == 4 { 2 } else { 1 } * c.head_dim as i64;
                check(
                    format!("layers.{l}.attn.compressor.wkv.weight"),
                    vec![wide, d],
                );
                check(
                    format!("layers.{l}.attn.compressor.wgate.weight"),
                    vec![wide, d],
                );
                check(
                    format!("layers.{l}.attn.compressor.ape"),
                    vec![r as i64, wide],
                );
                check(
                    format!("layers.{l}.attn.compressor.norm.weight"),
                    vec![c.head_dim as i64],
                );
            }
            if r == 4 {
                check(
                    format!("layers.{l}.attn.indexer.wq_b.weight"),
                    vec![c.index_heads as i64 * c.index_head_dim as i64, c.q_lora],
                );
                check(
                    format!("layers.{l}.attn.indexer.weights_proj.weight"),
                    vec![c.index_heads as i64, d],
                );
                check(
                    format!("layers.{l}.attn.indexer.compressor.wkv.weight"),
                    vec![2 * c.index_head_dim as i64, d],
                );
            }
        }
    }

    // Hash layers carry a token-id table and no selection bias; scored layers
    // are the other way round. Presence, not shape, is the thing to check.
    for l in 0..c.layers {
        let table = h.contains_key(&format!("layers.{l}.ffn.gate.tid2eid"));
        let bias = h.contains_key(&format!("layers.{l}.ffn.gate.bias"));
        let want_hash = l < c.hash_layers;
        if !h.is_empty() && h.contains_key(&format!("layers.{l}.ffn.gate.weight")) {
            if want_hash && !table {
                errs.push(format!(
                    "layers.{l}.ffn.gate: config says hash-routed (num_hash_layers={}) but the \
                     checkpoint has no tid2eid table",
                    c.hash_layers
                ));
            }
            if !want_hash && !bias {
                errs.push(format!(
                    "layers.{l}.ffn.gate: config says score-routed but the checkpoint has no \
                     selection bias"
                ));
            }
        }
    }

    // `num_nextn_predict_layers` vs the mtp stacks actually shipped.
    let mtp_seen = (0..8)
        .filter(|i| h.contains_key(&format!("mtp.{i}.attn.wkv.weight")))
        .count() as u32;
    if mtp_seen > 0 && mtp_seen != c.mtp_layers {
        errs.push(format!(
            "mtp stages: config num_nextn_predict_layers={} but the checkpoint ships {mtp_seen} \
             (inference/config.json says n_mtp_layers=3 — the HF config is the one that is wrong)",
            c.mtp_layers
        ));
    }
    errs
}

struct V4Gap {
    what: &'static str,
    scope: String,
    why: String,
    fix: &'static str,
    /// `Some(evidence)` once a kernel exists AND passes a real numeric gate.
    /// Printed verbatim in the CLOSED section, and it is the reason not to
    /// rebuild the thing — K3's report has actually caused that.
    done: Option<&'static str>,
}

/// Ranked missing-capability list, blocker first.
///
/// Ordering rule, as for K3: a gap that blocks EVERY layer outranks one that
/// blocks a subset.
fn v4_gaps(c: &V4Cfg) -> Vec<V4Gap> {
    vec![
        V4Gap {
            what: "hyper-connections (hc_pre / hc_post) in place of the residual",
            scope: format!("{}/{} layers, both sub-layers", c.layers, c.layers),
            why: format!(
                "the hidden state is {} PARALLEL residual streams, not one. Each sub-layer \
                 RMS-scales the flattened streams, projects them through hc_*_fn to {} mixing \
                 coefficients, runs {} Sinkhorn row/column normalization iterations, reduces the \
                 streams to one vector, and afterwards writes the branch output back as \
                 `post (x) branch + comb . residual`. There is no residual add anywhere in this \
                 model: emitting one is the hc_mult=1, post=comb=1 special case and is a \
                 different network. The Sinkhorn loop is a per-token {}x{} normalization, which \
                 is a new kernel shape for this tree - nothing in devgen iterates like it.",
                c.hc_mult,
                c.hc_rows(),
                c.hc_iters,
                c.hc_mult,
                c.hc_mult
            ),
            fix: "one fused op per sub-layer boundary (reduce and expand share the same three \
                  weights and the same mixes, so they are two halves of one kernel, not two \
                  independent ones). nn-graph models them as Op::HcReduce / Op::HcExpand; the \
                  Lean-side obligation is that expand's output equals the reference composition.",
            done: Some(
                "d_hc_reduce / d_hc_expand / d_hc_reduce_head in runtime/amd/op_deepseek_v4.h.                  Gated by runtime/tests/v4_hc_oracle_gfx942.hip against a host transcription in                  double at the shipped HC=4/iters=20/D=4096, at T>1 across workgroups, and with                  the expand output ALIASING the residual: max rel 6.0e-3..1.1e-2 vs a 2e-2 bf16                  floor. Two negative controls, both loud: dropping post's factor of 2 gives 1.15,                  normalizing over D instead of HC*D gives 2.8e-1. The reduce STASHES post/comb to                  a [T, HC+HC*HC] fp32 scratch that the expand reads - do not re-run the                  projection there.",
            ),
        },
        V4Gap {
            what: "single-KV-head attention at head_dim 512 with a learned sink",
            scope: format!("{}/{} layers", c.layers, c.layers),
            why: format!(
                "one KV vector of {} lanes ({} content + {} rope) is shared by all {} query \
                 heads - a kvh=1, hd={} flash shape, which is its own kernel instantiation and \
                 not a short loop over an existing one. Queries are RMS-rescaled per head with NO \
                 learned gain, the last {} lanes carry rope, and the OUTPUT is de-rotated by the \
                 conjugate angle before the projection. The sink is a learned per-head logit that \
                 joins the softmax denominator without a value row, so it shifts every \
                 probability in the row.",
                c.head_dim,
                c.nope_head_dim(),
                c.rope_head_dim,
                c.heads,
                c.head_dim,
                c.rope_head_dim
            ),
            fix: "a kvh=1/hd=512 flash arm with a sink term in the running denominator, plus the \
                  inverse rope on the output. The sliding window is only 128, so the window part \
                  is small and dense; the size is all in the compressed history.",
            done: Some(
                "d_v4_sparse_attn. NOTE this op also covers the CONSUMPTION side of the                  compressed history and the indexer: Attention.forward builds ONE index list per                  query (window ++ compressed) and there is no dense arm. Gated by                  runtime/tests/v4_attn_oracle_gfx942.hip at H=64/D=512/TOPK=640, with a masked                  prefix, at T>1, and at the indexer's D=128: max rel 7.1e-3..1.5e-2. Controls: a                  -1 index reading row 0 gives 9.7; dropping the sink gives only 2.2e-2 at 640                  keys, so the gate carries a sequence-start row (8 keys) where it is 1.56.                  CORRECTNESS REFERENCE, NOT TUNED: keys are consumed one at a time, so each costs                  a six-shuffle reduction against 8 MACs - Stage 4 owes it an MFMA key tile.",
            ),
        },
        V4Gap {
            what: "learned KV compressor (gated pooling, sequence-rate changing)",
            scope: format!(
                "{}/{} layers ({} overlapping at ratio 4, the rest at their own ratio)",
                c.n_compressed(),
                c.layers,
                c.n_indexed()
            ),
            why: "every `ratio` consecutive tokens are pooled into ONE compressed KV entry by a \
                  softmax over learned gate scores plus a per-offset positional bias, then \
                  RMS-normed and roped at the compressed position. This is the only rate-changing \
                  op in the model. The ratio-4 layers additionally OVERLAP their windows - each \
                  window carries the previous window's half, which is why their wkv/wgate/ape are \
                  twice as wide - and the overlapped form builds its windows by value-padding (0 \
                  for KV, -inf for the scores). Attention then reads the sliding window and the \
                  compressed history as one KV sequence."
                .to_string(),
            fix: "a compressor kernel writing into the tail of the same KV ring the window uses, \
                  with a decode-time incremental form (the reference keeps a per-sequence window \
                  state and only emits an entry every `ratio` steps, so the state is a runtime \
                  resource exactly like the KV cache).",
            done: Some(
                "d_v4_kv_compress - PREFILL FORM ONLY. Gated by                  runtime/tests/v4_compress_oracle_gfx942.hip at ratio 4 overlapping, ratio 128                  plain, and the indexer's D=128, over several groups so group 0's absent                  predecessor is exercised: max rel 7.4e-3..1.1e-2. Controls: collapsing the                  PER-OUTPUT-DIM softmax to a per-row one gives 2.9-5.5; swapping which half of                  the projection each window half reads gives 4.3-5.8 on the overlapping cases and                  correctly leaves the plain one alone. STILL OPEN: the decode-incremental form                  (start_pos > 0) and the prefill remainder that seeds its state, and a T that is                  not a multiple of ratio has its tail DROPPED.",
            ),
        },
        V4Gap {
            what: "sparse indexer (top-k selection over the compressed history)",
            scope: format!("{}/{} layers", c.n_indexed(), c.layers),
            why: format!(
                "a second, independent compressor at width {} plus a {}-head scorer: score = \
                 sum_h relu(q_h . kv) * w_h, then keep the top {} compressed entries. Under TP \
                 the score is all-reduced BEFORE the top-k, so the selection is a collective, not \
                 a per-rank decision - ranks that disagree on the selected set produce different \
                 tokens.",
                c.index_head_dim, c.index_heads, c.index_topk
            ),
            fix: "the DSA machinery for GLM-5.2 is the closest existing arm and is the thing to \
                  read first (crates/devgen/src/mla.rs, the glm_* indexer path). What is new here \
                  is that the indexer scores a COMPRESSED sequence it computes itself, rather \
                  than the raw KV.",
            done: Some(
                "STRUCTURE ONLY - the fp4 activation simulation is NOT modeled, see below.                  d_v4_index_score + d_v4_index_topk; the second compressor is d_v4_kv_compress at                  index width, already closed above. Gated by                  runtime/tests/v4_index_oracle_gfx942.hip: scores match to 7.7e-7 (they are fp32)                  and the SELECTED SET is compared EXACTLY, including the prefill causal limit and                  the ring offset. Controls: moving the relu outside the head sum gives 3.4-4.4                  and a different set; using t/ratio instead of (t+1)/ratio for the causal limit                  leaves the scores untouched and breaks only the prefill selection. Ties resolve                  to the LOWER index so the choice is deterministic across ranks. NOT MODELED: the                  reference Hadamard-rotates and fp4 quantize-dequantizes both q and the                  compressed KV. The Hadamard is orthogonal and applied to BOTH sides of the dot,                  so it cancels exactly; the fp4 rounding does not, and near a tie it can flip a                  selection. TP: the score is all-reduced BEFORE the top-k - the emitter must                  place that reduction, and it is not optional.",
            ),
        },
        V4Gap {
            what: "FP4 routed experts with hash-routed leading layers",
            scope: format!(
                "{} experts/layer (top-{}) + {} shared, {} hash-routed layers",
                c.n_exp, c.top_k, c.shared_exp, c.hash_layers
            ),
            why: format!(
                "routed experts are {} ({} nibble-packed, one {}-element scale group per row \
                 chunk) while the shared expert and the attention projections stay {}. Routing is \
                 `{}` scoring - NOT softmax and NOT sigmoid - with a selection bias that shifts \
                 the top-k comparison but not the combine weights, renormalization over the \
                 selected set, and a route scale of {}. The first {} layers do not score at all: \
                 the expert set is read from a [vocab, top_k] token-id table. The expert SwiGLU \
                 clamps both branches at {}.",
                c.expert_dtype,
                if c.expert_dtype == "fp4" { "e2m1" } else { "-" },
                32,
                c.quant_method,
                c.score_func,
                c.route_scale,
                c.hash_layers,
                c.swiglu_limit
            ),
            fix: "the mxfp4 expert path (MoeEnc::Mxfp4, runtime/amd/op_moe.h wave_dot_mxfp4) is \
                  w4a16 and already exists - see the K3 report's ALREADY-COVERED section. What is \
                  new is sqrtsoftplus scoring, the clamped SwiGLU, and the hash table (a gather, \
                  not a top-k, and it makes the router's score GEMM dead weight on those layers).",
            done: Some(
                "ROUTING AND ACTIVATION ONLY; the FP4 expert GEMM is the existing mxfp4 path and                  was never missing. d_v4_moe_route + d_v4_clamped_swiglu, gated by                  runtime/tests/v4_moe_oracle_gfx942.hip at 256 experts / top-6 / scale 1.5 /                  limit 10: the router reproduces the reference's expert SET exactly and its                  weights to 1.3e-7, the SwiGLU to 4.5e-3. Controls: letting the selection bias                  reach the combine weight gives 1.4e-1 (and correctly leaves the hash layers,                  which have no bias, untouched); exp() scoring picks DIFFERENT experts; dropping                  the clamp gives 14.4. The clamp control only bites with inputs that reach the                  limit.",
            ),
        },
        V4Gap {
            what: "block-diagonal output projection (wo_a)",
            scope: format!("{}/{} layers", c.layers, c.layers),
            why: format!(
                "{} independent [{}, {}] blocks stored as ONE stacked tensor, applied to the \
                 matching slice of the head axis. A dense linear of the same total size mixes \
                 groups the reference keeps separate.",
                c.o_groups,
                c.o_lora,
                c.heads as i64 * c.head_dim as i64 / c.o_groups as i64
            ),
            fix: "a grouped GEMM over the head axis, then the ordinary wo_b projection. Cheap \
                  next to the rest of this list, but it is a shape no existing arm emits.",
            done: Some(
                "d_v4_grouped_linear, gated by runtime/tests/v4_moe_oracle_gfx942.hip at the                  shipped 8 groups x rank 1024 over 4096-wide slices: max rel 9.0e-3. Control: a                  dense linear of the same element count gives 4.63. Applied in bf16, as the                  reference does explicitly ('wo_a is FP8 in checkpoint; using BF16 for                  simplicity') - the fp8 arm is a Stage-4 question, not a correctness one.",
            ),
        },
        V4Gap {
            what: "DSpark / MTP draft stages",
            scope: format!(
                "{} mtp stage(s), block size {}, target layers {:?}",
                c.mtp_layers, c.dspark_block, c.dspark_targets
            ),
            why: "the draft network reads the mean-pooled hidden state of the target layers, runs \
                  its own attention variant over a noise-token block, and scores candidates with \
                  a Markov head plus a confidence head. It is a separate network sharing the \
                  embedding and the lm_head."
                .to_string(),
            fix: "out of scope until the main tower runs. Speculative decoding is a throughput \
                  multiplier on top of a correct model, never a prerequisite for one.",
            done: None,
        },
    ]
}

/// Report what V4 is and what it needs, then refuse. Never returns.
pub(crate) fn deepseek_v4_emit(dir: &Path, ctx: u32, tp: u32) -> ! {
    let c = cfg_deepseek_v4(dir);
    let (hdrs, have, total) = k3_shard_headers(dir);
    let mismatches = v4_config_vs_tensors(&c, &hdrs);

    eprintln!("deepseek_v4: config ACCEPTED, emission REFUSED.\n");
    eprintln!("  checkpoint  {}", dir.display());
    if total == 0 {
        eprintln!("  shards      none on disk — every dimension below is CONFIG-ONLY, unverified");
    } else {
        eprintln!(
            "  shards      {have}/{total} readable, {} tensors{}",
            hdrs.len(),
            if have < total {
                "  (PARTIAL: a tensor's absence proves nothing)"
            } else {
                ""
            }
        );
    }
    eprintln!(
        "  tower       {} layers | hidden {} | heads {} | vocab {} | ctx {ctx} | tp {tp}",
        c.layers, c.hidden, c.heads, c.vocab
    );
    eprintln!(
        "  attention   kvh=1 hd={} ({}+{} rope) | q_lora {} | window {} | sink=yes | \
         out {}x{} block-diag",
        c.head_dim,
        c.nope_head_dim(),
        c.rope_head_dim,
        c.q_lora,
        c.window,
        c.o_groups,
        c.o_lora
    );
    eprintln!(
        "  kv history  {} of {} layers compressed ({} of them overlapping at ratio 4) | \
         rope_theta {} window / {} compressed",
        c.n_compressed(),
        c.layers,
        c.n_indexed(),
        c.rope_theta,
        c.compress_rope_theta
    );
    eprintln!(
        "  indexer     {} layers | {} heads x {} dim | top-{}",
        c.n_indexed(),
        c.index_heads,
        c.index_head_dim,
        c.index_topk
    );
    eprintln!(
        "  MoE         {} routed (top-{}) + {} shared | inter {} | experts {} | scoring {} \
         (scale {}) | swiglu clamp {} | hash-routed L<{}",
        c.n_exp,
        c.top_k,
        c.shared_exp,
        c.moe_inter,
        c.expert_dtype,
        c.score_func,
        c.route_scale,
        c.swiglu_limit,
        c.hash_layers
    );
    eprintln!(
        "  residual    HYPER-CONNECTIONS: {} streams, {} Sinkhorn iters, {}-row mix — \
         there is NO residual add in this model",
        c.hc_mult,
        c.hc_iters,
        c.hc_rows()
    );
    eprintln!(
        "  quant       {} | block {:?} | scale_fmt {} | routed experts {} (attn/shared stay fp8)",
        c.quant_method, c.quant_block, c.scale_fmt, c.expert_dtype
    );
    eprintln!(
        "  draft       {} mtp stage(s), dspark block {}, targets {:?}",
        c.mtp_layers, c.dspark_block, c.dspark_targets
    );

    if mismatches.is_empty() {
        eprintln!(
            "\n  config vs tensors: AGREE on every dimension the {} readable tensors can speak to.",
            hdrs.len()
        );
    } else {
        eprintln!(
            "\n  config vs tensors: {} DISAGREEMENT(S). TRUST THE TENSORS (GLM-5.2 lesson):",
            mismatches.len()
        );
        for m in &mismatches {
            eprintln!("    - {m}");
        }
    }

    eprintln!("\nALREADY COVERED — do not rebuild these:");
    eprintln!(
        "  operator IR           `deepseek_v4` parses, builds and shape-infers in nn-graph, and \
         its weight\n                        manifest is name-exact against this checkpoint (1328 \
         tensors in scope,\n                        none missing, none invented). Read \
         crates/nn-graph/src/models/deepseek_v4.rs\n                        before deciding what \
         any kernel has to compute — the op order there is the\n                        reference \
         forward pass, already checked."
    );
    eprintln!(
        "  egglog vocabulary     every V4 op lowers (HcReduce/HcExpand, KvCompress, \
         GroupedLinear,\n                        ClampedSwiGlu, RopeInverse, AttentionSinkMask, \
         RmsNormNoGain, and the two\n                        router variants). No new rewrite rule \
         was needed, so Checkpoint A carries\n                        no new obligation."
    );
    eprintln!(
        "  fp4 expert weights    MoeEnc::Mxfp4 + `wave_dot_mxfp4` (runtime/amd/op_moe.h) is w4a16 \
         against\n                        nibble-packed fp4 — this checkpoint's exact scheme. The \
         on-disk layout is\n                        [N, K/2] packed with an e8m0 scale per 32 \
         elements. Nothing to convert."
    );
    eprintln!(
        "  block-fp8 attention   the fp8 e4m3 / ue8m0 128x128 block scheme on the attention \
         projections is\n                        the same one GLM-5.2 already ships on this ISA."
    );

    let gaps = v4_gaps(&c);
    let (closed, open): (Vec<_>, Vec<_>) = gaps.iter().partition(|g| g.done.is_some());

    // CLOSED FIRST, and in full — the point of this section is to stop the next
    // reader rebuilding what already passes a gate, which is a failure K3's
    // report has actually caused rather than a hypothetical one.
    if !closed.is_empty() {
        eprintln!(
            "\nCLOSED — {} capabilities that WERE on this list and now have a kernel WITH a \
             passing\nnumeric gate. DO NOT REBUILD THESE. Read the `done:` line first; several \
             say what is\nstill missing inside an otherwise-closed item.\n",
            closed.len()
        );
        for (i, g) in closed.iter().enumerate() {
            eprintln!("C{:<2} {}  [{}]", i + 1, g.what, g.scope);
            for (n, line) in textwrap72(g.done.unwrap()).into_iter().enumerate() {
                eprintln!("      {} {line}", if n == 0 { "done:" } else { "     " });
            }
            eprintln!();
        }
    }

    if !open.is_empty() {
        eprintln!(
            "OPEN — {} capabilities this checkpoint needs and plow does not have.\n",
            open.len()
        );
        for (i, g) in open.iter().enumerate() {
            eprintln!("G{:<2} {}  [{}]", i + 1, g.what, g.scope);
            for line in textwrap72(&g.why) {
                eprintln!("      {line}");
            }
            for (n, line) in textwrap72(g.fix).into_iter().enumerate() {
                eprintln!("      {} {line}", if n == 0 { "fix:" } else { "    " });
            }
            eprintln!();
        }
    }

    eprintln!(
        "STILL NOT EMITTABLE. Every kernel above is gated in ISOLATION against the reference; \
         none\nof them is wired into a DevOp, an emitter or the interpreter, so there is no blob \
         and no\n`--block` extraction yet. That wiring — plus the indexer (G4) and the decode \
         side of the\ncompressor — is what stands between this and a single V4 layer running.\n\
         \n\
         NO PERFORMANCE CLAIM IS MADE ANYWHERE IN THIS REPORT. The closed kernels are \
         correctness\nreferences: the attention consumes keys one at a time and the compressor \
         re-streams its\nprojection per pooled row. Both are Stage-4 work and neither has been \
         timed on any part."
    );
    std::process::exit(2);
}

#[cfg(test)]
/// A small but structurally faithful V4 config for emitter tests: the real
/// hc/head geometry, four layers, one of each compress kind.
pub(crate) fn cfg_deepseek_v4_for_test() -> V4Cfg {
    V4Cfg {
        layers: 4,
        hidden: 4096,
        heads: 64,
        head_dim: 512,
        rope_head_dim: 64,
        q_lora: 1024,
        o_groups: 8,
        o_lora: 1024,
        vocab: 129280,
        window: 128,
        compress_ratios: vec![0, 0, 4, 128],
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        index_heads: 64,
        index_head_dim: 128,
        index_topk: 512,
        n_exp: 256,
        shared_exp: 1,
        top_k: 6,
        moe_inter: 2048,
        hash_layers: 3,
        swiglu_limit: 10.0,
        score_func: "sqrtsoftplus".into(),
        route_scale: 1.5,
        hc_mult: 4,
        hc_iters: 20,
        expert_dtype: "fp4".into(),
        quant_method: "fp8".into(),
        quant_block: vec![128, 128],
        scale_fmt: "ue8m0".into(),
        mtp_layers: 1,
        dspark_block: 5,
        dspark_targets: vec![],
    }
}
