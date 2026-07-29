//! KDA — Kimi Delta Attention. The mixer in **69 of Kimi-K3's 93 layers**.
//!
//! Spec: `docs/kimi-k3-kda.md`. Implementation notes: `plans/kimi-k3-kda-impl.md`.
//!
//! This module emits ONE KDA layer. That is deliberate and it is the milestone: the GLM B4 gate
//! de-risked one real-weight layer before anyone tried 78, and that discipline is why GLM's bugs
//! were findable. A 93-layer K3 emit needs the AttnRes block, `situ`, LatentMoE, the MLA output
//! gate and NoPE, all owned elsewhere (`plans/kimi-k3-frontend.md` §4).
//!
//! # The shape of the layer, and why it is thirteen packets
//!
//! ```text
//!   P0  pre-norm            RmsNorm        hidden, ln_w        -> x[7168]
//!   P1  q proj              Gemv           x, W_q              -> q~[12288]      | gated only
//!   P2  k proj              Gemv           x, W_k              -> k~[12288]      | on P0 —
//!   P3  v proj              Gemv           x, W_v              -> v~[12288]      | six
//!   P4  output gate         Gemv           x, W_g              -> g^[12288]      | independent
//!   P5  forget-gate down    Gemv           x, W_fa             -> r[128]         | GEMVs,
//!   P6  beta logits         Gemv           x, W_b              -> b~[96]         | all ready
//!   P7  forget-gate up      Gemv           r, W_fb             -> g~[12288]
//!   P8  short conv          KdaConv        q~,k~,v~, conv_w, conv_state
//!   P9  gate + beta         KdaGate        g~, b~, A_log, dt_bias
//!   P10 state step          KdaStateStep   q,k,v,g,beta, STATE -> o; STATE'
//!   P11 gated norm          KdaGatedNorm   o, o_norm_w, g^     -> y[12288]
//!   P12 out proj            Gemv           y, W_o              -> attn[7168]
//!   P13 residual            Residual       hidden, attn        -> hidden'
//! ```
//!
//! **Do not collapse P1–P6.** `plans/knob-contract.md` §6g-KNOBS measured `GLM_GROUP=1` removing
//! **38% of the ops for +2.88 ms**, because collapsing work that ran on disjoint CU slices into a
//! loop inside one packet destroys concurrency. Op count is not the objective function. The merge
//! that IS safe is along the OUTPUT dimension (`GemvQkv = 22` already does q/k/v in one GEMV, and
//! a `qkvg` extension would remove 207 packets/token) — that keeps the op wide and still spread
//! over 256 CUs. Merging along a LOOP dimension is the fatal one.
//!
//! **`Dep::Coarse` everywhere, and that is provable rather than assumed.** `devbuild.rs:278-300`
//! cites `lean-plow/Plow/CounterGranularity.lean`'s `collapse`: if the work is UNIFORM across each
//! stage's slices then the fine schedule's makespan is *identical* to the coarse one, for any
//! producer maps whatsoever. Every KDA head is identical and every column tile is identical, so
//! `Dep::Fine` provably buys nothing here.
//!
//! # Three things that silently corrupt, all verified against the checkpoint
//!
//! `scripts/kda_verify_ckpt.py` checks these across all 69 KDA layers, not a sample:
//!
//! 1. **Layer lists are 1-BASED** — `is_kda_layer` tests `(idx + 1) in kda_layers`. The tail is
//!    `KKK MM`, so an `i % 4 == 3` rule gets 0-based layer 92 wrong. [`is_kda_layer_0based`]
//!    takes the config list; there is no modulus anywhere in this file.
//! 2. **`A_log` ships `[128]`, is a per-head `[96]` ZERO-PADDED** to `head_dim`. The declared
//!    handle is `[H]` and [`KDA_A_LOG_CKPT_LEN`] records the checkpoint length so a loader that
//!    forgets the slice fails on byte size rather than computing the wrong decay for every token.
//! 3. **State is V-FIRST `[h][v][k]`.** Transposing it gives garbage with exactly the right norm.
//!    [`KDA_STATE_LAYOUT`] is carried into the block descriptor for that reason.

use packet::dev::DevOp;
use packet::devbuild::Builder;

/// K3's KDA geometry (`text_config.linear_attn_config` plus `hidden_size`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdaCfg {
    /// `hidden_size` — 7168.
    pub hidden: u32,
    /// `linear_attn_config.num_heads` — 96. K3 is not GVA: `num_k_heads == num_heads`.
    pub heads: u32,
    /// `linear_attn_config.head_dim` — 128, and it is BOTH the key dim and the value dim, so the
    /// state is square.
    pub head_dim: u32,
    /// `linear_attn_config.short_conv_kernel_size` — 4.
    pub conv_w: u32,
    /// `linear_attn_config.gate_lower_bound` — `Some(-5.0)` for K3. `None` selects the older
    /// unbounded `-exp(A_log) * softplus(.)` branch, which is what every released vLLM has and
    /// what K3 does NOT use.
    pub gate_lower_bound: Option<f32>,
    /// `rms_norm_eps` — 1e-5. Used by the pre-norm and by the per-head output norm.
    pub eps: f32,
    /// Value columns per workgroup in [`DevOp::KdaStateStep`]. See [`KdaCfg::state_step_blocks`].
    pub bv: u32,
}

impl KdaCfg {
    /// The projection width, `H * D` = 12288.
    pub fn proj(&self) -> u32 {
        self.heads * self.head_dim
    }
    /// The short conv's channel count: q, k and v concatenated. 36864.
    pub fn conv_dim(&self) -> u32 {
        3 * self.proj()
    }
    /// Elements in one sequence's recurrent state for this layer: `[H, D, D]`, **f32**.
    ///
    /// f32 is settled from the reference implementation, not from prose. AMD's day-0 post says
    /// "FP32 KDA SSM states" but its own formula uses 2 bytes per element
    /// (`docs/amd/kimi-k3-atom-day0.md` §4), so the two halves of that document disagree.
    /// `fla/ops/kda/fused_recurrent.py` allocates `dtype=torch.float32` in both layout branches
    /// and accumulates in `tl.float32`; `naive_recurrent_kda` casts every input to `torch.float`;
    /// vLLM's `mamba_utils.py` hardcodes fp32 independently. The state is a running accumulator
    /// over up to 10^6 rank-1 updates, which is not the same risk class as a KV ring where each
    /// entry is written once. bf16 halves it to 245.8 MiB/seq and is UNMEASURED.
    pub fn state_elems(&self) -> u64 {
        self.heads as u64 * self.head_dim as u64 * self.head_dim as u64
    }
    /// Elements in one sequence's conv window for this layer: `[3*H*D, W]`, f32.
    ///
    /// `W` slots, not `W-1`: `[fla]` keeps the current token in the buffer and `[vllm]` prepends
    /// it. Both are correct and they differ by 36864 elements per layer. This is the `[fla]`
    /// convention because `[fla]` is what the numeric gate runs against.
    pub fn conv_state_elems(&self) -> u64 {
        self.conv_dim() as u64 * self.conv_w as u64
    }

    /// Workgroups for [`DevOp::KdaStateStep`] — `H * D / BV` work items, capped at the CU count.
    ///
    /// `docs/kimi-k3-kda.md` §7.3 requires every proposal to be checked against 256 explicitly,
    /// because head-parallelism alone reproduces the `MlaMergeFold` defect: one workgroup per head
    /// is 96/256 = **37.5%** at TP1 and 24/256 = **9.4%** at TP4, and TP divides the head count so
    /// it gets worse exactly where K3 has to run.
    ///
    /// At `H=96, D=128, BV=16` this is 768 items, so `blocks = 256` — **100%**. Note the honest
    /// caveat §7.3 does not state: 100% at every TP degree requires `BV` to SHRINK with the head
    /// count. At TP8 (12 heads) a fixed `BV=16` gives 96 items and 37.5%; `BV=8` restores 192.
    /// `BV` is a loop bound, not a register constraint, which is why it is an immediate.
    pub fn state_step_blocks(&self, n_cu: u32) -> u32 {
        let items = self.proj() / self.bv;
        items.min(n_cu).max(1)
    }
}

/// The checkpoint's `A_log` length. It ships `[128]` = `head_dim`, but only `[:96]` = `num_heads`
/// is non-zero and only `[:96]` is ever read (the kernel indexes `A_log + i_hv`).
///
/// Verified in **69/69** KDA layers by `scripts/kda_verify_ckpt.py`: exactly indices 0..95 are
/// non-zero, 96..127 are exactly 0.0. Both public K3 implementations narrow the same way (SGLang's
/// custom `weight_loader`, vLLM PR #50089's `a_log_weight_loader`). A loader that consumes all 128
/// values as if per-head-dim silently computes the wrong decay for every token of every KDA layer;
/// one that asserts `numel() == num_heads` rejects a valid checkpoint. The HF-shipped
/// `modeling_kimi_linear.py` does the latter, so "it loads in transformers" is NOT available as a
/// correctness signal for K3.
pub const KDA_A_LOG_CKPT_LEN: u32 = 128;

/// Layout tag for the recurrent state in the block descriptor. **V-first**, `[h][v][k]`.
///
/// K3 passes `transpose_state_layout=True`, a deprecated alias for `state_v_first`. Since
/// `V == K == 128` the byte count is unchanged, so getting it backwards transposes the state and
/// produces garbage that still has the right norm. It is recorded here so a consumer that reads
/// the descriptor cannot guess.
pub const KDA_STATE_LAYOUT: &str = "head_v_k";

/// Is 0-based layer `l` a KDA layer, given the config's **1-BASED** `kda_layers` list?
///
/// `configuration_kimi_k3.py::is_kda_layer` tests `(layer_idx + 1) in kda_layers`. Drive this from
/// the list, never from a modulus: K3's run-length pattern over 0..92 is `(KKK M) x 22` then
/// `KKK MM`, so layers 91 AND 92 are both MLA and an `i % 4 == 3` rule gets 92 wrong. AMD's own
/// day-0 post independently says "with one additional MLA layer at the end".
pub fn is_kda_layer_0based(kda_layers_1based: &[u32], l: u32) -> bool {
    kda_layers_1based.contains(&(l + 1))
}

/// Handles for one KDA layer's checkpoint weights, in the order [`declare_kda_weights`] declares
/// them. Names are the raw safetensors keys — plow has no HF-name translation table, the packet
/// tensor name IS the checkpoint key.
#[derive(Clone, Copy, Debug)]
pub struct KdaWeights {
    pub q_proj: u32,
    pub k_proj: u32,
    pub v_proj: u32,
    pub g_proj: u32,
    pub o_proj: u32,
    pub f_a_proj: u32,
    pub f_b_proj: u32,
    pub b_proj: u32,
    /// `[H*D, W]` f32 conv taps, one handle per stream. Three separate [`DevOp::KdaConv`] packets
    /// rather than one over a concatenated `3*H*D` axis: they are independent work, so three
    /// packets is three times the concurrency and zero times the loader-side surgery. Each still
    /// spans all 256 CUs at `H*D = 12288` channels.
    pub conv_w: [u32; 3],
    /// `[H]` f32 — the `[:96]` slice of the checkpoint's `[128]`. See [`KDA_A_LOG_CKPT_LEN`].
    pub a_log: u32,
    /// `[H*D]` f32, laid out `[H, D]` row-major. No padding (verified, 69/69 layers).
    pub dt_bias: u32,
    /// `[D]` f32, SHARED by all `H` heads.
    pub o_norm: u32,
    pub ln_w: u32,
}

/// Declare one KDA layer's 14 checkpoint tensors under `prefix`.
///
/// `prefix` must be the checkpoint's own, e.g.
/// `language_model.model.layers.{l}.self_attn.` — note **`language_model.`**: K3 nests the text
/// tower under a multimodal wrapper, and of the checkpoint's 497 220 tensors, **zero** start with
/// `model.`. Commit `6067014` inverted plow's binder to classify by EXCLUSION for exactly this
/// reason, so an unknown name is now demanded rather than silently skipped.
///
/// Every one of the 14 tensors keeps its checkpoint name and shape. Nothing is concatenated,
/// packed or folded — the `Mamba2Scan` precedent crams `A_log | D | dt_bias | conv_b | norm_w`
/// into one f32 handle because it ran out of operand slots, and that packing is a symptom of
/// over-fusion, not a constraint. Decomposed, no op here is at the slot ceiling and the
/// `A_log[:96]` narrow happens in the ordinary loader against its own declared handle.
pub fn declare_kda_weights(b: &mut Builder, c: &KdaCfg, prefix: &str, ln_prefix: &str) -> KdaWeights {
    let bf = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 2);
    let f32t = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 4);
    let (h, hd, p, w) = (
        c.hidden as u64,
        c.head_dim as u64,
        c.proj() as u64,
        c.conv_w as u64,
    );
    KdaWeights {
        q_proj: bf(b, format!("{prefix}q_proj.weight"), p * h),
        k_proj: bf(b, format!("{prefix}k_proj.weight"), p * h),
        v_proj: bf(b, format!("{prefix}v_proj.weight"), p * h),
        // Full-rank output gate. `use_full_rank_gate: true` is a K3 change relative to the Kimi
        // Linear paper, which parameterises the OUTPUT gate low-rank. The checkpoint has
        // `g_proj [12288,7168]` and no `g_a_proj`/`g_b_proj` anywhere. Note the asymmetry that is
        // easy to get backwards: the output gate is full-rank, the FORGET gate is still low-rank
        // 128 (`f_a_proj` -> `f_b_proj`) regardless of the flag.
        g_proj: bf(b, format!("{prefix}g_proj.weight"), p * h),
        o_proj: bf(b, format!("{prefix}o_proj.weight"), h * p),
        f_a_proj: bf(b, format!("{prefix}f_a_proj.weight"), hd * h),
        f_b_proj: bf(b, format!("{prefix}f_b_proj.weight"), p * hd),
        // beta logits: ONE SCALAR PER HEAD, [96,7168]. Not per channel.
        b_proj: bf(b, format!("{prefix}b_proj.weight"), c.heads as u64 * h),
        // F32 in the checkpoint, not bf16. Shape `[H*D, 1, W]`; the middle axis is the depthwise
        // group of 1 and carries no bytes.
        conv_w: [
            f32t(b, format!("{prefix}q_conv1d.weight"), p * w),
            f32t(b, format!("{prefix}k_conv1d.weight"), p * w),
            f32t(b, format!("{prefix}v_conv1d.weight"), p * w),
        ],
        a_log: f32t(b, format!("{prefix}A_log"), c.heads as u64),
        dt_bias: f32t(b, format!("{prefix}dt_bias"), p),
        o_norm: f32t(b, format!("{prefix}o_norm.weight"), hd),
        ln_w: bf(b, format!("{ln_prefix}input_layernorm.weight"), h),
    }
}

/// The two per-sequence state tensors a KDA layer carries. Both are f32 and both are
/// **read-modify-write**; neither is addressable by token.
///
/// A KDA layer's cache is not a ring. There is one slot per (sequence, layer), 6.5625 MiB, and it
/// is CONSTANT in context length — which is the whole architectural argument (69 KDA layers cost
/// 0.44 GiB at 1M tokens where 24 MLA layers cost 27 GiB) and also a real scheduling constraint,
/// because it is a hard cost at sequence ADMISSION rather than one that grows.
#[derive(Clone, Copy, Debug)]
pub struct KdaState {
    /// `[H, D, D]` f32, **V-FIRST** — see [`KDA_STATE_LAYOUT`].
    pub state: u32,
    /// `[H*D, W]` f32 per stream (q, k, v), current token at slot `W-1`.
    pub conv_state: [u32; 3],
}

/// Declare the carried state for one KDA layer. `prefix` is a COMPILER-owned namespace
/// (`kv.`-style), not a checkpoint one: these are runtime buffers, not weights.
pub fn declare_kda_state(b: &mut Builder, c: &KdaCfg, prefix: &str) -> KdaState {
    let cw = c.proj() as u64 * c.conv_w as u64 * 4;
    KdaState {
        state: b.tensor(&format!("{prefix}state"), c.state_elems() * 4),
        conv_state: [
            b.tensor(&format!("{prefix}conv_state.q"), cw),
            b.tensor(&format!("{prefix}conv_state.k"), cw),
            b.tensor(&format!("{prefix}conv_state.v"), cw),
        ],
    }
}

/// Emit one KDA layer for `t` tokens, reading `hidden` and writing `next`.
///
/// Returns the counter of the final residual, for the next block to gate on.
///
/// `t == 1` is decode; `t > 1` runs the identical op graph, because
/// [`DevOp::KdaStateStep`]'s serial-`T` loop is the reference `fused_recurrent` algorithm and is
/// exact at any `T`. The chunked scan of the spec's §7.5 is a matmul-bound REWRITE of a path that
/// then already works; it is not needed for correctness and its opcode is deliberately not
/// declared until its kernel exists.
#[allow(clippy::too_many_arguments)]
pub fn emit_kda_layer(
    b: &mut Builder,
    c: &KdaCfg,
    w: &KdaWeights,
    st: &KdaState,
    act_prefix: &str,
    t: u32,
    hidden: u32,
    next: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    assert_eq!(c.head_dim % 64, 0, "KDA: head_dim must be a multiple of the 64-lane wave");
    assert_eq!(
        c.proj() % c.bv,
        0,
        "KDA: BV must divide H*D so the column tiles partition the state"
    );
    let all: Vec<u32> = (0..n_cu).collect();
    let bft = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 2);
    let f32t = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 4);
    let (hi, p, hd, nh) = (c.hidden, c.proj(), c.head_dim, c.heads);
    let (tt, pu, hiu) = (t as u64, p as u64, hi as u64);
    let a = act_prefix;

    // Activations. q, k and v stay three separate [T, H*D] buffers all the way through: the three
    // projections are independent packets, the three convs are independent packets, and the state
    // step reads all three. Nothing here is concatenated, so no operand carries an offset the
    // kernel would have to be told about — an immediate that is emitted but not read is the
    // contract's §3 bug shape verbatim, and it fails silently because the output stays finite.
    let x = bft(b, format!("{a}x"), tt * hiu);
    let raw = [
        bft(b, format!("{a}q_raw"), tt * pu),
        bft(b, format!("{a}k_raw"), tt * pu),
        bft(b, format!("{a}v_raw"), tt * pu),
    ];
    let mix = [
        bft(b, format!("{a}q"), tt * pu),
        bft(b, format!("{a}k"), tt * pu),
        bft(b, format!("{a}v"), tt * pu),
    ];
    let g_raw = bft(b, format!("{a}g_raw"), tt * pu);
    let fa = bft(b, format!("{a}f_a"), tt * hd as u64);
    let f_raw = bft(b, format!("{a}f_raw"), tt * pu);
    let b_raw = bft(b, format!("{a}b_raw"), tt * nh as u64);
    let gate = f32t(b, format!("{a}gate"), tt * pu);
    let beta = f32t(b, format!("{a}beta"), tt * nh as u64);
    let o = bft(b, format!("{a}o"), tt * pu);
    let y = bft(b, format!("{a}y"), tt * pu);
    let attn = bft(b, format!("{a}attn"), tt * hiu);

    // P0 — pre-norm.
    let c_ln = b.emit(DevOp::RmsNorm, all.clone(), deps, |d| {
        d.t[0] = x;
        d.t[1] = hidden;
        d.t[2] = w.ln_w;
        d.i[0] = t;
        d.i[1] = hi;
        d.f[0] = c.eps;
    });

    // P1-P6 — six INDEPENDENT GEMVs gated only on P0. They read the same `x` and write disjoint
    // outputs, so all six are ready at once and the scheduler overlaps them across 256 CUs. A
    // monolithic KDA op would have serialized them behind one packet, which is the `GLM_GROUP=1`
    // mistake exactly.
    //
    // `Gemm`/`Gemv` compute `C[M,N] = A[M,K] . B[N,K]^T` with `B` stored `[out_features,
    // in_features]` (dev.rs:87-89) — which is precisely how HF stores an nn.Linear weight. No
    // transpose at load, for any of the eight.
    let gemv = |b: &mut Builder, out: u32, row: u32, wt: u32, n: u32, k: u32, dep: u32| {
        b.emit(DevOp::Gemv, all.clone(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = row;
            d.t[2] = wt;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = k;
        })
    };
    let c_q = gemv(b, raw[0], x, w.q_proj, p, hi, c_ln);
    let c_k = gemv(b, raw[1], x, w.k_proj, p, hi, c_ln);
    let c_v = gemv(b, raw[2], x, w.v_proj, p, hi, c_ln);
    let c_g = gemv(b, g_raw, x, w.g_proj, p, hi, c_ln);
    let c_fa = gemv(b, fa, x, w.f_a_proj, hd, hi, c_ln);
    let c_bb = gemv(b, b_raw, x, w.b_proj, nh, hi, c_ln);
    // P7 — forget-gate up-projection. The forget gate stays LOW RANK 128 even though the output
    // gate is full rank; that asymmetry is `use_full_rank_gate`'s and it is easy to get backwards.
    let c_fb = gemv(b, f_raw, fa, w.f_b_proj, p, hd, c_fa);

    // P8a-c — the three short convs, one per stream, each over H*D = 12288 channels and each
    // spanning all 256 CUs. Coarse deps: every channel is identical work, so `CounterGranularity`'s
    // `collapse` says a fine schedule has an identical makespan.
    let mut c_conv = [0u32; 3];
    for s in 0..3usize {
        c_conv[s] = b.emit(DevOp::KdaConv, all.clone(), &[[c_q, c_k, c_v][s]], |d| {
            d.t[0] = mix[s];
            d.t[1] = raw[s];
            d.t[2] = w.conv_w[s];
            d.t[3] = st.conv_state[s];
            d.i[0] = t;
            d.i[1] = p;
            d.i[2] = c.conv_w;
            d.i[3] = 1; // silu, applied AFTER the convolution
        });
    }

    // P9 — gate + beta. Independent of P8.
    let c_gate = b.emit(DevOp::KdaGate, all.clone(), &[c_fb, c_bb], |d| {
        d.t[0] = gate;
        d.t[1] = beta;
        d.t[2] = f_raw;
        d.t[3] = b_raw;
        d.t[4] = w.a_log;
        d.t[5] = w.dt_bias;
        d.i[0] = t;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[3] = u32::from(c.gate_lower_bound.is_some());
        d.f[0] = c.gate_lower_bound.unwrap_or(0.0);
    });

    // P10 — the recurrence. `blocks` is checked against the CU count here rather than left to
    // whatever the work happens to produce, because head-parallelism alone is the pathology.
    let nb = c.state_step_blocks(n_cu);
    let cus: Vec<u32> = (0..nb).collect();
    let c_step = b.emit(DevOp::KdaStateStep, cus, &[c_conv[0], c_conv[1], c_conv[2], c_gate], |d| {
        d.t[0] = o;
        d.t[1] = mix[0];
        d.t[2] = mix[1];
        d.t[3] = mix[2];
        d.t[4] = gate;
        d.t[5] = beta;
        d.t[6] = st.state;
        d.i[0] = t;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[3] = c.bv;
        d.i[4] = 1; // flags bit0: L2-normalize q and k in kernel, eps INSIDE the sqrt
        // scale = D^-0.5, applied to q AFTER the L2 norm; k is NOT scaled.
        d.f[0] = (c.head_dim as f32).powf(-0.5);
    });

    // P11 — output gate. Gated on P4 (whose GEMV has had the whole conv+gate+state chain to hide
    // under) and P10.
    let c_norm = b.emit(DevOp::KdaGatedNorm, all.clone(), &[c_g, c_step], |d| {
        d.t[0] = y;
        d.t[1] = o;
        d.t[2] = w.o_norm;
        d.t[3] = g_raw;
        d.i[0] = t;
        d.i[1] = nh;
        d.i[2] = hd;
        d.f[0] = c.eps;
    });

    // P12/P13 — out projection and residual.
    let c_o = gemv(b, attn, y, w.o_proj, hi, p, c_norm);
    b.emit(DevOp::Residual, all, &[c_o], |d| {
        d.t[0] = next;
        d.t[1] = hidden;
        d.t[2] = attn;
        d.i[0] = t * hi;
        d.f[0] = 1.0;
    })
}

/// Bidirectional coverage for one KDA layer's checkpoint names.
///
/// `checkpoint::validate_coverage` is bidirectional — every declared name must exist and every
/// checkpoint tensor must be covered — which is what stops the `Mamba2Scan` failure mode where an
/// emitter declares synthetic handles (`mamba.{l}.conv1d.w`, …) that no loader ever binds and the
/// op has therefore only ever run against a zero-filled table.
///
/// The three `*_conv1d.weight` tensors are consumed into one concatenated handle, so they are
/// listed here explicitly rather than derived from the declared names.
pub fn kda_checkpoint_names(prefix: &str) -> Vec<String> {
    [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "g_proj.weight",
        "o_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "b_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
    ]
    .iter()
    .map(|n| format!("{prefix}{n}"))
    .collect()
}

/// TP sharding class for a KDA tensor, by suffix.
///
/// `crates/plowrt/src/asset/shard.rs` classifies by substring and **defaults to replicate**, so a
/// name it does not recognise is not a crash — it is wrong math on more than one GPU. The two
/// entries that look wrong and are right: `f_a_proj` and `o_norm` are REPLICATED while everything
/// around them is column-parallel, because `f_a_proj`'s output is the rank-128 bottleneck (not
/// per-head) and `o_norm` is a `[D]` vector shared by every head. `[vllm]` uses `ReplicatedLinear`
/// for `f_a_proj` explicitly, and AMD's day-0 post lists "KDA `f_a`" under replicated.
///
/// `A_log` splits over `H` **after** the `[:96]` narrow, never before.
pub fn kda_shard_class(suffix: &str) -> &'static str {
    match suffix {
        "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "g_proj.weight"
        | "f_b_proj.weight" | "b_proj.weight" | "q_conv1d.weight" | "k_conv1d.weight"
        | "v_conv1d.weight" | "A_log" | "dt_bias" => "column",
        "o_proj.weight" => "row",
        "f_a_proj.weight" | "o_norm.weight" => "replicated",
        _ => panic!("KDA: unclassified tensor `{suffix}` — shard.rs defaults to REPLICATE, which \
                     for KDA is not a crash, just wrong math on >1 GPU"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3() -> KdaCfg {
        KdaCfg {
            hidden: 7168,
            heads: 96,
            head_dim: 128,
            conv_w: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
            bv: 16,
        }
    }

    /// The 1-based off-by-one, and the modulus rule it invalidates. K3's real list.
    #[test]
    fn layer_lists_are_one_based_and_the_tail_is_kkk_mm() {
        // (KKK M) x 22 then KKK MM, 1-based.
        let mut kda: Vec<u32> = Vec::new();
        for blk in 0..23u32 {
            for j in 0..3 {
                kda.push(blk * 4 + 1 + j);
            }
        }
        assert_eq!(kda.len(), 69);
        assert_eq!(*kda.last().unwrap(), 91);
        // 0-based 88, 89, 90 are KDA; 91 and 92 are BOTH MLA.
        assert!(is_kda_layer_0based(&kda, 88));
        assert!(is_kda_layer_0based(&kda, 90));
        assert!(!is_kda_layer_0based(&kda, 91));
        assert!(!is_kda_layer_0based(&kda, 92));
        // The modulus rule agrees everywhere EXCEPT the last layer, which is exactly why it is a
        // silent bug rather than an obvious one.
        let disagree: Vec<u32> =
            (0..93).filter(|&l| (l % 4 == 3) == is_kda_layer_0based(&kda, l)).collect();
        assert_eq!(disagree, vec![92]);
    }

    /// §7.3's discipline as a test: no proposal ships without its workgroup count.
    #[test]
    fn state_step_fills_the_chip_and_head_parallelism_does_not() {
        let c = k3();
        assert_eq!(c.proj(), 12288);
        assert_eq!(c.proj() / c.bv, 768, "work items");
        assert_eq!(c.state_step_blocks(256), 256, "column-tiled: 100% of 256 CUs");
        // One workgroup per head is the MlaMergeFold defect: 96/256 at TP1, 24/256 at TP4.
        assert_eq!(c.heads.min(256), 96);
        // TP8 with a FIXED BV does NOT hold 100% — the spec's table quietly assumes BV shrinks.
        let tp8 = KdaCfg { heads: 12, ..c };
        assert_eq!(tp8.state_step_blocks(256), 96, "12*128/16 = 96 items, 37.5%");
        assert_eq!(KdaCfg { bv: 8, ..tp8 }.state_step_blocks(256), 192, "BV=8 restores 192");
    }

    /// The state is f32 and CONSTANT in context length. 6.00 MiB + 0.5625 MiB per layer per seq.
    #[test]
    fn state_bytes_match_the_spec() {
        let c = k3();
        assert_eq!(c.state_elems(), 96 * 128 * 128);
        assert_eq!(c.state_elems() * 4, 6 * 1024 * 1024, "6.000 MiB f32");
        assert_eq!(c.conv_state_elems(), 3 * 12288 * 4);
        assert_eq!(c.conv_state_elems() * 4, 589_824, "0.5625 MiB f32");
        let per_layer = (c.state_elems() + c.conv_state_elems()) * 4;
        assert_eq!(per_layer, 6_881_280, "6.5625 MiB per layer per sequence");
        assert_eq!(per_layer * 69, 474_808_320, "452.8 MiB/seq — CONSTANT in context length");
    }

    #[test]
    fn a_log_is_narrowed_and_the_padding_is_recorded() {
        let c = k3();
        assert_eq!(KDA_A_LOG_CKPT_LEN, c.head_dim, "ships [128] = head_dim");
        assert_eq!(c.heads, 96, "only [:96] is non-zero and only [:96] is read");
        assert_ne!(KDA_A_LOG_CKPT_LEN, c.heads, "the whole point: they differ");
    }

    /// The two easy-to-get-backwards shard classes.
    #[test]
    fn f_a_proj_and_o_norm_are_replicated_amid_column_parallel_neighbours() {
        assert_eq!(kda_shard_class("f_a_proj.weight"), "replicated");
        assert_eq!(kda_shard_class("o_norm.weight"), "replicated");
        assert_eq!(kda_shard_class("f_b_proj.weight"), "column");
        assert_eq!(kda_shard_class("o_proj.weight"), "row");
        for n in kda_checkpoint_names("") {
            let _ = kda_shard_class(&n); // panics on anything unclassified
        }
    }

    #[test]
    fn coverage_lists_all_fourteen_checkpoint_tensors() {
        let n = kda_checkpoint_names("language_model.model.layers.0.self_attn.");
        assert_eq!(n.len(), 14);
        assert!(n.iter().all(|s| s.starts_with("language_model.")), "K3 has ZERO `model.` tensors");
        assert!(n.contains(&"language_model.model.layers.0.self_attn.A_log".to_string()));
    }

    /// The op graph: thirteen packets, six of them concurrent, and every new opcode reachable.
    #[test]
    fn one_layer_emits_the_expected_graph() {
        let c = k3();
        let mut b = Builder::new(256);
        let all: Vec<u32> = (0..256).collect();
        let hidden = b.tensor("in.hidden", 7168 * 2);
        let next = b.tensor("act.next", 7168 * 2);
        let w = declare_kda_weights(&mut b, &c, "language_model.model.layers.0.self_attn.", "language_model.model.layers.0.");
        let st = declare_kda_state(&mut b, &c, "kda.0.");
        let seed = b.emit(DevOp::Nop, all, &[], |_| {});
        emit_kda_layer(&mut b, &c, &w, &st, "act.kda0.", 1, hidden, next, 256, &[seed]);
        let p = b.finish();
        let ops: Vec<u16> = p.insts.iter().map(|i| i.op).collect();
        assert_eq!(ops.iter().filter(|&&o| o == DevOp::Gemv as u16).count(), 8);
        for op in [DevOp::KdaConv, DevOp::KdaGate, DevOp::KdaStateStep, DevOp::KdaGatedNorm] {
            assert!(ops.contains(&(op as u16)), "{op:?} is not emitted — an opcode nothing \
                    reaches is how Mamba2Scan became dead code");
        }
        // The state step must not be stranded on a handful of CUs.
        let step = p.insts.iter().find(|i| i.op == DevOp::KdaStateStep as u16).unwrap();
        assert_eq!(step.blocks, 256, "column tiling, not head parallelism");
        // Every other KDA op spans the chip too.
        for op in [DevOp::KdaConv, DevOp::KdaGate] {
            let i = p.insts.iter().find(|i| i.op == op as u16).unwrap();
            assert_eq!(i.blocks, 256, "{op:?}");
        }
    }
}
