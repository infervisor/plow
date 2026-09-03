//! KDA — Kimi Delta Attention. The mixer in **69 of Kimi-K3's 93 layers**.
//!
//! Spec: `docs/kimi-k3-kda.md`. Implementation notes: the design notes.
//!
//! This module emits ONE KDA layer. That is deliberate and it is the milestone: the GLM B4 gate
//! de-risked one real-weight layer before anyone tried 78, and that discipline is why GLM's bugs
//! were findable. A 93-layer K3 emit needs the AttnRes block, `situ`, LatentMoE, the MLA output
//! gate and NoPE, all owned elsewhere.
//!
//! # The shape of the layer, and why it is thirteen packets
//!
//! ```text
//!   P0  pre-norm            RmsNorm        hidden, ln_w        -> x[7168]
//!   P1  q proj              \                                  -> q~[12288]      \ ONE packet:
//!   P2  k proj               |  GemvQkvg   x, W_q|W_k|         -> k~[12288]       | 49152 cols
//!   P3  v proj               |             W_v|W_g             -> v~[12288]       | over 256 CUs
//!   P4  output gate         /                                  -> g^[12288]      / (`fuse_qkvg`)
//!   P5  forget-gate down    Gemv           x, W_fa             -> r[128]         | gated only
//!   P6  beta logits         Gemv           x, W_b              -> b~[96]         | on P0
//!   P7  forget-gate up      Gemv           r, W_fb             -> g~[12288]
//!   P8  short conv         \ KdaConv3       q~,k~,v~, conv_w, conv_state  \ ONE packet:
//!                           |                                             | 36864 channels
//!   P9  gate + beta         | KdaStateStepG q,k,v,g~,b~,A_log,dt_bias,    | over 256 CUs
//!   P10 state step         /                STATE -> o; STATE'            / (`fuse_kda`)
//!   P11 gated norm          KdaGatedNorm   o, o_norm_w, g^     -> y[12288]
//!   P12 out proj            Gemv           y, W_o              -> attn[7168]
//!   P13 residual            Residual       hidden, attn        -> hidden'
//! ```
//!
//! **Do not collapse P1–P6 along a LOOP axis.** The design notes §6g-KNOBS measured
//! `GLM_GROUP=1` removing **38% of the ops for +2.88 ms**, because collapsing work that ran on
//! disjoint CU slices into a loop inside one packet destroys concurrency. Op count is not the
//! objective function. The merge that IS safe is along the OUTPUT dimension, and P1–P4 now take
//! it: [`DevOp::GemvQkvg`] extends `GemvQkv = 22` with a fourth output stream, removing **207
//! packets/token** while making the op WIDER (48 -> 192 columns per CU, still all 256 CUs).
//! [`fuse_qkvg`] carries the argument and the LDS bound. P5/P6 stay separate — their weights are
//! 1/128th and 1/96th of a projection, so folding them in would buy two gates and hand two CUs a
//! ragged tail. Merging along a LOOP dimension is still the fatal one.
//!
//! **P8-P10 take the same merge, twice, and it is worth 3 packets/layer.** The four K3-specific
//! opcodes were SIX packets — three convs, a gate, the step, the gated norm — and are now THREE.
//! `runtime/tests/kda_fuse_bench_gfx950.c` is the instrument: at TP8 the six-packet chain costs
//! **5.03 ms** over 69 layers against 108 MiB of state traffic (~17 us at roofline), and the cost
//! is linear in the packet count at **12.08 us/packet**. [`fuse_kda`] argues each merge, including
//! the one that is REFUSED — folding the conv into the state step is a genuine cross-workgroup
//! RACE on `conv_state`, not merely a concurrency loss.
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

use packet::dev::{DevOp, WG_WAVES};
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

    /// Whether the single-row double-buffered Conv3+recurrence body has a compiled shape rung.
    ///
    /// This is deliberately geometry-based. The fused body gives one value column to each wave,
    /// supports the same lane-depth rungs as the serial recurrence, and keeps at most eight
    /// convolution taps in registers. Other shapes retain the unfused oracle.
    pub fn supports_conv_state_step_db(&self) -> bool {
        self.heads > 0
            && matches!(self.head_dim, 64 | 128 | 256)
            && self.bv == WG_WAVES
            && (1..=8).contains(&self.conv_w)
    }

    /// Whether the standalone 256-thread fused decode boundary has a validated shape rung.
    pub fn supports_decode_fused(&self, rows: u32, n_cu: u32, seq_rows: bool) -> bool {
        matches!(rows, 1 | 8)
            && (rows == 1 || seq_rows)
            && self.heads > 0
            && rows.saturating_mul(self.heads) <= n_cu
            && self.head_dim == 128
            && self.bv == 8
            && self.conv_w == 4
            && self.gate_lower_bound.is_some_and(f32::is_finite)
            && self.eps.is_finite()
            && self.eps > 0.0
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
        self.state_step_blocks_rows(n_cu, 1)
    }

    /// [`KdaCfg::state_step_blocks`] when the program's `rows` are INDEPENDENT SEQUENCES.
    ///
    /// The kernel folds that row axis into its work-item map (`bstride != 0` in
    /// `d_kda_state_step_t`), because nothing in row `t`'s recurrence reads row `t-1`'s. So the
    /// item count is `rows * H * ntile`, not `H * ntile`, and the packet has to be WIDE enough to
    /// expose it — at TP8/BV=8 the un-scaled count is 192 of 256 CUs, and a B=16 decode was
    /// running 16 rows serially inside those same 192 workgroups on 69 of 93 layers.
    ///
    /// `rows = 1` reproduces the old count exactly, so every prefill bucket and every B=1 decode
    /// is byte-identical.
    pub fn state_step_blocks_rows(&self, n_cu: u32, rows: u32) -> u32 {
        let items = (self.proj() / self.bv).saturating_mul(rows.max(1));
        items.min(n_cu).max(1)
    }

    /// Workgroups for [`DevOp::KdaGatedNorm`]: one wave per `(token, head)` row.
    pub fn gated_norm_blocks_rows(&self, n_cu: u32, rows: u32) -> u32 {
        rows.max(1)
            .saturating_mul(self.heads)
            .div_ceil(WG_WAVES)
            .min(n_cu)
            .max(1)
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
pub fn declare_kda_weights(
    b: &mut Builder,
    c: &KdaCfg,
    prefix: &str,
    ln_prefix: &str,
) -> KdaWeights {
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
    /// Alternate q/k/v window bank for the B1 Conv3+state experiment.
    pub conv_state_alt: Option<[u32; 3]>,
    /// `in.pos`, whose parity selects the source bank without another host upload.
    pub bank_pos: Option<u32>,
}

/// Declare the carried state for one KDA layer. `prefix` is a COMPILER-owned namespace
/// (`kv.`-style), not a checkpoint one: these are runtime buffers, not weights.
/// `slots` is the blob-wide capacity for INDEPENDENT SEQUENCES. A prefill program uses slot 0
/// while a batched decode program uses its first `B` slots.
///
/// It is not the same thing as the program's `t`. A prefill program has `t` rows and ONE slot,
/// because its rows are consecutive tokens of one sequence threading through one carried state.
/// A batched decode program has `t == B` rows and `B` slots. Conflating the two is the failure
/// `RowKind` exists to name (see `crate::k3::RowKind`), and it is silent: the wrong one still
/// allocates, still runs, and still produces fluent text.
///
/// The cost is real and worth stating at the declaration: the state is **6.5625 MiB per
/// (sequence, layer)** and CONSTANT in context, so 69 KDA layers are 0.44 GiB per slot. That is
/// a hard cost at ADMISSION rather than one that grows — which is the architectural argument for
/// KDA and also, once slots exist, the thing that bounds how many sequences fit.
pub fn declare_kda_state(b: &mut Builder, c: &KdaCfg, prefix: &str, slots: u64) -> KdaState {
    assert!(slots >= 1, "declare_kda_state: slots must be >= 1");
    let cw = c.proj() as u64 * c.conv_w as u64 * 4 * slots;
    KdaState {
        state: b.tensor(&format!("{prefix}state"), c.state_elems() * 4 * slots),
        conv_state: [
            b.tensor(&format!("{prefix}conv_state.q"), cw),
            b.tensor(&format!("{prefix}conv_state.k"), cw),
            b.tensor(&format!("{prefix}conv_state.v"), cw),
        ],
        conv_state_alt: None,
        bank_pos: None,
    }
}

pub fn declare_kda_state_db(
    b: &mut Builder,
    c: &KdaCfg,
    prefix: &str,
    slots: u64,
    pos: u32,
) -> KdaState {
    assert_eq!(slots, 1, "KDA Conv3/state fusion is currently B1-only");
    let mut state = declare_kda_state(b, c, prefix, slots);
    let cw = c.proj() as u64 * c.conv_w as u64 * 4;
    state.conv_state_alt = Some([
        b.tensor(&format!("{prefix}conv_state_alt.q"), cw),
        b.tensor(&format!("{prefix}conv_state_alt.k"), cw),
        b.tensor(&format!("{prefix}conv_state_alt.v"), cw),
    ]);
    state.bank_pos = Some(pos);
    state
}

/// May P1-P4 collapse into one [`DevOp::GemvQkvg`] packet for `t` rows of width `hidden`?
///
/// # Why this merge is safe when `GLM_GROUP=1` was not
///
/// The design notes §6g-KNOBS measured `GLM_GROUP=1` removing **38% of the ops for
/// +2.88 ms**. It merged along a LOOP dimension: work that had been running on DISJOINT CU slices
/// became a loop inside one packet, and the concurrency went with it. This merges along the
/// OUTPUT dimension. Each of q/k/v/g is already `all` = 256 CUs, so per CU they were 4 x 48
/// columns run back-to-back down that CU's own stream; fused they are 192 columns of one sweep on
/// the same 256 CUs. Nothing that ran in parallel starts running in sequence — the op gets WIDER
/// (48 -> 192 columns per CU), which is the direction the knob contract asks for.
///
/// What is actually deleted: three counter gates per layer (**207 packets per token** over 69 KDA
/// layers) and three redundant LDS stagings of the same `x[7168]`.
///
/// And it costs no conv concurrency, which is the objection to check rather than assume.
/// `emit_kda_mixer` appends P1..P7 before P8a, and `devbuild` writes every CU's stream in
/// topological order — so a CU reaches `KdaConv(q)` only after it has already executed its share
/// of the k, v and g projections. Gating the convs on one fused counter instead of three cannot
/// delay them.
///
/// # The bound, and why it is a REFUSAL to fuse rather than a runtime fallback
///
/// `gemv_qkvg_rows` reads `x` only through LDS (`op_gemm.h`: "x is ALWAYS staged in LDS here"),
/// so the staged rows must fit [`crate::gm_lds_halves()`]. At `hidden = 7168` that is 10 rows —
/// decode's `t = 1` always fits, and the 4-stream op is decode-only anyway. Mis-gating this is
/// the §6g-BATCH silent corruption verbatim (rows past the arena fluent-but-wrong), so the check
/// is here, at emit time, where it produces a DIFFERENT PACKET. There is no device-side fallback:
/// `interp.hip`'s arm traps on a malformed 4-stream packet rather than degrading to three
/// streams, because this interpreter's dispatch `default:` writes nothing and a degraded sweep
/// would leave `g_raw` untouched and finite.
///
/// `PLOW_KDA_FUSE_QKVG=0` emits P1-P4 as four plain [`DevOp::Gemv`] packets instead. It exists so
/// the fusion stays falsifiable against the same program, not as a shape guard.
fn fuse_qkvg(t: u32, hidden: u32) -> bool {
    // Hardcoded ON (was `PLOW_KDA_FUSE_QKVG`, never disabled in production).
    // DECODE ONLY, and now stated as `t == 1` rather than left to the LDS bound to imply.
    //
    // The bound alone declines above ~10 rows at `hidden = 7168` (`10 * 7168 = 71680 <= 73728`,
    // `11 * 7168 = 78848` does not), so it already refuses every rung of the prefill ladder, whose
    // floor is 128. But it ACCEPTS t = 2..10, and `GemvQkvg` (op 100) is compiled into the DECODE
    // interpreter only — the prefill bucket has no `case PLOW_DOP_GEMV_QKVG`, and this
    // interpreter's dispatch `default:` writes nothing. A `K3_PREFILL=8` bucket would therefore
    // have emitted a four-stream packet that silently produced NOTHING for q, k, v and g, and the
    // layer would run on whatever the arena held. The LDS bound is a real constraint and is kept;
    // it is just not the one that makes this op safe.
    if t != 1 {
        return false;
    }
    crate::gemv_staged_rows(t) as u64 * hidden as u64 <= crate::gm_lds_halves()
}

/// May the K3-specific chain collapse from SIX packets to THREE?
///
/// # The measurement that decides it
///
/// `runtime/tests/kda_fuse_bench_gfx950.c` builds the six-op chain over 69 chained layers and
/// times it on gfx950. At TP8 — the shape K3 decode actually runs — 414 packets cost **5.03 ms**
/// while the arithmetic underneath them is 108 MiB of state traffic, about 17 us at roofline. The
/// cost is LINEAR in the packet count: 0.096 ms at L=1, 0.413 at L=5, 1.28 at L=17, 5.03 at L=69,
/// a slope of **12.08 us per packet** and an intercept of 0.02 ms. A KDA decode layer is not
/// bandwidth bound at batch 1; it is protocol, and the only lever is the packet count.
///
/// # Which merges are safe, and the one that is NOT
///
/// Two of the three are taken here:
///
/// - **[`DevOp::KdaConv3`]** merges the three convs along the CHANNEL axis, which is their OUTPUT
///   axis. Each conv already spanned all 256 CUs at `ceil(H*D/256)` channels; fused they span the
///   same 256 CUs at `ceil(3*H*D/256)`. Per CU the work RISES 48 -> 144 at TP1 and 6 -> 18 at TP8.
/// - **[`DevOp::KdaStateStepG`]** folds the gate into the recurrence's LDS staging. `blocks` is
///   still [`KdaCfg::state_step_blocks`] and the item map is untouched — the gate is evaluated
///   where its consumer already is, not looped over — so nothing narrows. It is bit-identical to
///   the pair it replaces, because the deleted `g` was an f32 HBM round trip.
///
/// The third is REFUSED, and the reason is worth keeping. Folding the conv into
/// [`DevOp::KdaStateStepG`] would delete a third packet, and the arithmetic works: item
/// `(head, tile)` needs `q[h,:]`, `k[h,:]` and `v[h, tile]` convolved, all computable locally. It
/// is a RACE. A head's `D/BV` tiles land on different workgroups, and every one of them needs the
/// pre-update `conv_state` window for that head's q and k channels while exactly one of them must
/// write the post-update window. Read and write are then unordered across workgroups and the
/// answer depends on which lands first — silent, nondeterministic, and with the right norm.
/// Double-buffering the conv state would fix it and would change the state contract; that is a
/// separate decision, not a free one.
///
/// [`DevOp::KdaGatedNorm`] stays its own packet for the reason its own doc gives: the norm reduces
/// over a whole head, whose `D` outputs are spread across `D/BV` workgroups under the step's slice
/// map, so folding it in needs a grid-wide barrier the interpreter does not provide.
///
/// `PLOW_KDA_FUSE=0` emits the six-packet chain instead, so the fusion stays falsifiable against
/// the same program — the same escape [`fuse_qkvg`] carries, and the reason
/// `runtime/tests/k3_block_gfx950_test.c` can score both spellings against one fixture.
/// Always fuse KDA chain (hardcoded — was `PLOW_KDA_FUSE`, never disabled).
fn fuse_kda() -> bool {
    true
}

/// Emit one KDA MIXER for `t` tokens: P0-P12, pre-norm through `o_proj`.
///
/// Returns `(counter, attn)` — the counter of the `o_proj` GEMV, and the handle
/// holding the un-residualled attention output `[t, hidden]`.
///
/// The residual is deliberately NOT here. A plain block adds the layer input;
/// K3 adds its AttnRes prefix sum, and at a snapshot layer it does not add at
/// all. Baking one in would force every K3 caller to emit a compensating op, so
/// the choice belongs to the caller. [`emit_kda_layer`] is the plain-residual
/// wrapper for everyone who wants the GLM-shaped boundary.
///
/// `t == 1` is decode; `t > 1` runs the identical op graph, because
/// [`DevOp::KdaStateStep`]'s serial-`T` loop is the reference `fused_recurrent` algorithm and is
/// exact at any `T`. The chunked scan of the spec's §7.5 is a matmul-bound REWRITE of a path that
/// then already works; it is not needed for correctness and its opcode is deliberately not
/// declared until its kernel exists.
#[allow(clippy::too_many_arguments)]
pub fn emit_kda_mixer(
    b: &mut Builder,
    c: &KdaCfg,
    w: &KdaWeights,
    st: &KdaState,
    act_prefix: &str,
    t: u32,
    hidden: u32,
    // Where `o_proj` writes. `None` allocates a local `{act_prefix}attn`. Under
    // TP this is the PEER-VISIBLE partial slot, because `o_proj` is row-parallel
    // and its output is a partial sum until the all-reduce.
    attn_dst: Option<u32>,
    n_cu: u32,
    // `true` when the CALLER has already normed `hidden` — K3's `AttnRes` absorbs P0 (see
    // `crate::k3::fuse_attnres_norm`), so this mixer must not norm it a second time.
    prenormed: bool,
    deps: &[u32],
    // See `emit_kda_mixer_ex`'s parameter of the same name: `true` means the `t` rows are
    // independent sequences (batched decode), each carrying its own state.
    seq_rows: bool,
) -> (u32, u32) {
    emit_kda_mixer_ex(
        b,
        c,
        w,
        st,
        act_prefix,
        t,
        hidden,
        attn_dst,
        n_cu,
        prenormed,
        deps,
        fuse_kda(),
        crate::emit_config::active().kda_chunk.unwrap_or_else(|| {
            crate::emit_is_amd() && crate::amd_target::active().1 == hwspec::IsaLevel::Gfx950
        }),
        crate::emit_config::active().kda_decode_fused
            && crate::amd_target::active().1 == hwspec::IsaLevel::Gfx950,
        seq_rows,
    )
}

/// [`emit_kda_mixer`] with the P8-P10 fusion decided by the CALLER rather than by the environment.
///
/// The knob is a parameter and not a second read of `PLOW_KDA_FUSE` because a process-global that
/// two emissions in one process disagree about is a bug generator, and because a test that flips an
/// env var races every other test in the binary — which it did, exactly once, before this seam
/// existed.
#[allow(clippy::too_many_arguments)]
fn emit_kda_mixer_ex(
    b: &mut Builder,
    c: &KdaCfg,
    w: &KdaWeights,
    st: &KdaState,
    act_prefix: &str,
    t: u32,
    hidden: u32,
    attn_dst: Option<u32>,
    n_cu: u32,
    prenormed: bool,
    deps: &[u32],
    fuse: bool,
    chunk_enable: bool,
    fused_decode_enable: bool,
    // Do this mixer's `t` rows carry one state between them, or one state EACH? `true` is a
    // batched decode program (independent sequences); everything today is `false`.
    seq_rows: bool,
) -> (u32, u32) {
    assert_eq!(
        c.head_dim % 64,
        0,
        "KDA: head_dim must be a multiple of the 64-lane wave"
    );
    assert_eq!(
        c.proj() % c.bv,
        0,
        "KDA: BV must divide H*D so the column tiles partition the state"
    );
    if t == 1 && st.conv_state_alt.is_some() {
        assert!(
            c.supports_conv_state_step_db(),
            "KDA Conv3+state-step DB requires H>0, D in {{64,128,256}}, BV={WG_WAVES}, and W in 1..=8"
        );
    }
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
    // `prenormed`: the caller's AttnRes already wrote the NORMED activation into `hidden`, so
    // there is no second buffer and no P0 packet. A declared handle nothing writes is the
    // `Mamba2Scan` smell this file names below, so `x` is not allocated on that path either.
    let x = if prenormed {
        hidden
    } else {
        bft(b, format!("{a}x"), tt * hiu)
    };
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
    // `gate`/`beta` exist ONLY on the unfused path. A declared handle nothing writes is the
    // `Mamba2Scan` smell — 69 layers x 49 KiB of arena that no op touches — and an emitter that
    // allocates it anyway is one refactor away from an op reading it.
    let chunk = chunk_enable && t >= 512 && !seq_rows && matches!(hd, 64 | 128);
    let (gate, beta) = if fuse && !chunk {
        (u32::MAX, u32::MAX)
    } else {
        (
            f32t(b, format!("{a}gate"), tt * pu),
            f32t(b, format!("{a}beta"), tt * nh as u64),
        )
    };
    let o = bft(b, format!("{a}o"), tt * pu);
    let y = bft(b, format!("{a}y"), tt * pu);
    let attn = match attn_dst {
        Some(h) => h,
        None => bft(b, format!("{a}attn"), tt * hiu),
    };
    let chunk_tmp = chunk.then(|| {
        (
            f32t(b, format!("{a}chunk.aqk"), tt * nh as u64 * 64),
            f32t(b, format!("{a}chunk.ainv"), tt * nh as u64 * 64),
            bft(b, format!("{a}chunk.w"), tt * pu),
            bft(b, format!("{a}chunk.u"), tt * pu),
        )
    });

    // P0 — pre-norm, unless the caller's AttnRes absorbed it (`crate::k3::fuse_attnres_norm`).
    let c_ln = if prenormed {
        deps[0]
    } else {
        b.emit(DevOp::RmsNorm, crate::k3::norm_cus(&all, t), deps, |d| {
            d.t[0] = x;
            d.t[1] = hidden;
            d.t[2] = w.ln_w;
            d.i[0] = t;
            d.i[1] = hi;
            d.f[0] = c.eps;
        })
    };

    // P1-P6 — INDEPENDENT GEMVs gated only on P0. They read the same `x` and write disjoint
    // outputs, so all are ready at once and the scheduler overlaps them across 256 CUs. A
    // monolithic KDA op would have serialized them behind one packet, which is the `GLM_GROUP=1`
    // mistake exactly. The one merge taken is P1-P4's, and it is along the OUTPUT axis, not a
    // loop axis — see [`fuse_qkvg`], which is where that distinction is argued.
    //
    // `Gemm`/`Gemv` compute `C[M,N] = A[M,K] . B[N,K]^T` with `B` stored `[out_features,
    // in_features]` (dev.rs:87-89) — which is precisely how HF stores an nn.Linear weight. No
    // transpose at load, for any of the eight.
    // `Gemv` at t == 1, a tiled GEMM above it — see [`crate::k3::emit_k3_linear`]. The seam is
    // here rather than at the call sites because all eight projections take it.
    let gemv = |b: &mut Builder, out: u32, row: u32, wt: u32, n: u32, k: u32, dep: u32| {
        crate::k3::emit_k3_linear(b, out, row, wt, t, n, k, n_cu, seq_rows, &[dep])
    };
    // P1-P4 collapse into ONE packet along the OUTPUT axis. See [`fuse_qkvg`] for why that is the
    // safe direction and the LDS bound that decides it; P5/P6 stay separate because their weights
    // (`[128,7168]` and `[96,7168]`) are 1/128th and 1/96th of a projection each — concatenating
    // them onto a 49152-wide sweep would buy two gates and hand two of the 256 CUs a ragged tail.
    let (c_q, c_k, c_v, c_g) = if fuse_qkvg(t, hi) {
        let f = b.emit(DevOp::GemvQkvg, all.clone(), &[c_ln], |d| {
            d.t[0] = raw[0];
            d.t[1] = x;
            d.t[2] = w.q_proj;
            d.t[3] = raw[1];
            d.t[4] = w.k_proj;
            d.t[5] = raw[2];
            d.t[6] = w.v_proj;
            d.t[7] = g_raw;
            d.i[0] = t;
            d.i[1] = p;
            d.i[2] = hi;
            d.i[3] = p;
            d.i[4] = p;
            d.i[5] = p;
            // The ninth pointer. `t[8]` is full; `DevOp::GemvQkvg` states why the demoted
            // operand is a weight and not an output.
            d.i[6] = w.g_proj;
        });
        (f, f, f, f)
    } else {
        (
            gemv(b, raw[0], x, w.q_proj, p, hi, c_ln),
            gemv(b, raw[1], x, w.k_proj, p, hi, c_ln),
            gemv(b, raw[2], x, w.v_proj, p, hi, c_ln),
            gemv(b, g_raw, x, w.g_proj, p, hi, c_ln),
        )
    };
    let c_fa = gemv(b, fa, x, w.f_a_proj, hd, hi, c_ln);
    let c_bb = gemv(b, b_raw, x, w.b_proj, nh, hi, c_ln);
    // P7 — forget-gate up-projection. The forget gate stays LOW RANK 128 even though the output
    // gate is full rank; that asymmetry is `use_full_rank_gate`'s and it is easy to get backwards.
    let c_fb = gemv(b, f_raw, fa, w.f_b_proj, p, hd, c_fa);

    // P8/P9/P10 — the K3-specific chain, SIX packets decomposed and THREE fused. `fuse_kda`
    // argues the direction; both spellings of the graph are emitted from here so the fusion stays
    // falsifiable against the same program.
    // Rows are a parallel axis only when they are independent sequences; a prefill bucket's rows
    // thread through one state and must stay serial, so it keeps the un-scaled count.
    let nb = c.state_step_blocks_rows(n_cu, if seq_rows { t } else { 1 });
    let cus: Vec<u32> = (0..nb).collect();
    let gate_mode = u32::from(c.gate_lower_bound.is_some());
    let lower_bound = c.gate_lower_bound.unwrap_or(0.0);
    let parked = seq_rows.then(|| b.tensor("in.parked", 4 * t as u64));
    // scale = D^-0.5, applied to q AFTER the L2 norm; k is NOT scaled.
    let scale = (c.head_dim as f32).powf(-0.5);

    if fused_decode_enable
        && fuse
        && chunk_tmp.is_none()
        && st.conv_state_alt.is_none()
        && c.supports_decode_fused(t, n_cu, seq_rows)
    {
        let handles = [
            w.conv_w[0],
            w.conv_w[1],
            w.conv_w[2],
            st.conv_state[0],
            st.conv_state[1],
            st.conv_state[2],
            w.a_log,
            w.dt_bias,
            w.o_norm,
            g_raw,
            parked.unwrap_or(packet::dev::TENSOR_NONE_I),
        ];
        let mut descriptor = Vec::with_capacity(handles.len() * 4);
        for handle in handles {
            descriptor.extend_from_slice(&handle.to_le_bytes());
        }
        let desc = b.tensor_init(&format!("{a}decode_fused.desc"), descriptor);
        let mut fused_deps = vec![c_q, c_k, c_v, c_g, c_fb, c_bb];
        fused_deps.sort_unstable();
        fused_deps.dedup();
        let blocks = t * nh;
        let fused_cus: Vec<u32> = (0..blocks).collect();
        let c_fused = b.emit(DevOp::KdaDecodeFused, fused_cus, &fused_deps, |d| {
            d.t[0] = y;
            d.t[1] = raw[0];
            d.t[2] = raw[1];
            d.t[3] = raw[2];
            d.t[4] = f_raw;
            d.t[5] = b_raw;
            d.t[6] = st.state;
            d.t[7] = desc;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = hd;
            d.i[3] = c.bv;
            d.i[4] = c.conv_w;
            d.i[5] = 1 | if seq_rows { 2 } else { 0 };
            d.i[6] = gate_mode;
            d.i[7] = 2;
            d.f[0] = scale;
            d.f[1] = lower_bound;
            d.j[1] = c.eps.to_bits();
        });
        let c_o = gemv(b, attn, y, w.o_proj, hi, p, c_fused);
        return (c_o, attn);
    }

    let c_step = if let (1, Some(alt), Some(bank_pos)) = (t, st.conv_state_alt, st.bank_pos) {
        let handles = [
            w.conv_w[0],
            w.conv_w[1],
            w.conv_w[2],
            st.conv_state[0],
            st.conv_state[1],
            st.conv_state[2],
            alt[0],
            alt[1],
            alt[2],
            w.a_log,
            w.dt_bias,
            bank_pos,
            parked.unwrap_or(packet::dev::TENSOR_NONE_I),
        ];
        let mut descriptor = Vec::with_capacity(handles.len() * 4);
        for handle in handles {
            descriptor.extend_from_slice(&handle.to_le_bytes());
        }
        let desc = b.tensor_init(&format!("{a}conv_step_db.desc"), descriptor);
        b.emit(
            DevOp::KdaConvStateStepG,
            cus,
            &[c_q, c_k, c_v, c_fb, c_bb],
            |d| {
                d.t[0] = o;
                d.t[1] = raw[0];
                d.t[2] = raw[1];
                d.t[3] = raw[2];
                d.t[4] = f_raw;
                d.t[5] = b_raw;
                d.t[6] = st.state;
                d.t[7] = desc;
                d.i[0] = t;
                d.i[1] = nh;
                d.i[2] = hd;
                d.i[3] = c.bv;
                d.i[4] = 1 | if seq_rows { 2 } else { 0 };
                d.i[5] = c.conv_w;
                d.i[6] = gate_mode;
                d.f[0] = scale;
                d.f[1] = lower_bound;
            },
        )
    } else if let Some((aqk, ainv, cw, cu)) = chunk_tmp {
        let c_conv = b.emit(DevOp::KdaConv3, all.clone(), &[c_q, c_k, c_v], |d| {
            d.t[0] = mix[0];
            d.t[1] = mix[1];
            d.t[2] = mix[2];
            d.t[3] = raw[0];
            d.t[4] = raw[1];
            d.t[5] = raw[2];
            d.t[6] = w.conv_w[0];
            d.t[7] = w.conv_w[1];
            d.i[0] = t;
            d.i[1] = p;
            d.i[2] = c.conv_w;
            d.i[3] = 1;
            d.i[4] = w.conv_w[2];
            d.i[5] = st.conv_state[0];
            d.i[6] = st.conv_state[1];
            d.i[7] = st.conv_state[2];
        });
        let c_prepare = b.emit(
            DevOp::KdaChunkPrepare,
            all.clone(),
            &[c_conv, c_fb, c_bb],
            |d| {
                d.t[0] = mix[0];
                d.t[1] = mix[1];
                d.t[2] = gate;
                d.t[3] = beta;
                d.t[4] = f_raw;
                d.t[5] = b_raw;
                d.t[6] = w.a_log;
                d.t[7] = w.dt_bias;
                d.i[0] = t;
                d.i[1] = nh;
                d.i[2] = hd;
                d.i[3] = gate_mode;
                d.f[0] = lower_bound;
            },
        );
        let intra_cus: Vec<u32> = (0..n_cu.min(t.div_ceil(64) * nh)).collect();
        let c_intra = b.emit(DevOp::KdaChunkIntra, intra_cus, &[c_prepare], |d| {
            d.t[0] = aqk;
            d.t[1] = ainv;
            d.t[2] = mix[0];
            d.t[3] = mix[1];
            d.t[4] = gate;
            d.t[5] = beta;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = hd;
            d.f[0] = scale;
        });
        let c_wu = b.emit(DevOp::KdaChunkWu, all.clone(), &[c_intra], |d| {
            d.t[0] = cw;
            d.t[1] = cu;
            d.t[2] = ainv;
            d.t[3] = mix[1];
            d.t[4] = mix[2];
            d.t[5] = gate;
            d.t[6] = beta;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = hd;
            d.i[3] = hd;
        });
        let carry_cus: Vec<u32> = (0..n_cu.min(nh * (hd / 16))).collect();
        b.emit(DevOp::KdaChunkCarry, carry_cus, &[c_wu], |d| {
            d.t[0] = o;
            d.t[1] = st.state;
            d.t[2] = mix[0];
            d.t[3] = mix[1];
            d.t[4] = cw;
            d.t[5] = cu;
            d.t[6] = aqk;
            d.t[7] = gate;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = hd;
            d.i[3] = hd;
            d.f[0] = scale;
        })
    } else if fuse {
        // P8 — ONE conv over the 3*H*D concatenated channel axis, still all 256 CUs, with the
        // per-CU channel count RISING 3x. The three streams keep separate buffers; the op takes
        // twelve pointers and four of them ride in `i[]`.
        // PER-ROW PARKED MASK, `[t]` u32, NON-ZERO = skip this row's recurrence.
        //
        // The sense is "parked", not "active", and that is a safety property rather than a
        // preference: an unwritten or zeroed tensor then means EVERY ROW PARTICIPATES, which is
        // the pre-mask behaviour. With the opposite sense a caller that forgot to upload the
        // mask -- `amd-bench` drives the engine directly and has no reason to know about it --
        // would silently park every row and produce fluent garbage.
        //
        // Only a sequence-rows program has one: its `t` rows are INDEPENDENT sequences, and a
        // decode dispatch advances every row whether or not the server has work for it. That is
        // harmless for an append-only KV cache (an idle row rewrites a row nothing reads) and NOT
        // harmless for the recurrence, which reads and writes `state[row]` unconditionally — so a
        // slot in the middle of a chunked prefill would have its recurrence advanced by a garbage
        // token the moment any other slot decoded. The mask is what lets the host freeze a row.
        //
        // Declared here rather than threaded through the signature: the builder dedups by name,
        // so this returns the one `in.active` no matter how many layers ask for it.
        let parked = parked.unwrap_or(0);
        let c_conv = b.emit(DevOp::KdaConv3, all.clone(), &[c_q, c_k, c_v], |d| {
            d.t[0] = mix[0];
            d.t[1] = mix[1];
            d.t[2] = mix[2];
            d.t[3] = raw[0];
            d.t[4] = raw[1];
            d.t[5] = raw[2];
            d.t[6] = w.conv_w[0];
            d.t[7] = w.conv_w[1];
            d.i[0] = t;
            d.i[1] = p; // channels PER STREAM, not the concatenated 3*H*D
            d.i[2] = c.conv_w;
            d.i[3] = 1; // silu, applied AFTER the convolution
            d.i[4] = w.conv_w[2];
            // Per-row conv-window stride, in floats. NOT a flag and NOT in `i[]`: every integer
            // slot on this op is an operand (`T C W act w_v cs_q cs_k cs_v`), and reading one of
            // them as flags faulted the GPU. `fj[1].u` is untouched by this emitter.
            d.j[0] = if seq_rows { p * c.conv_w } else { 0 };
            d.i[5] = st.conv_state[0];
            d.i[6] = st.conv_state[1];
            d.i[7] = st.conv_state[2];
            // `j[1]` (= `fj[2]`), the last free slot on this op. Read only when `j[0] != 0`,
            // which is exactly the seq-rows condition, so a non-batched blob is unchanged.
            d.j[1] = parked;
        });
        // P9+P10 — the gate is computed inside the recurrence's LDS staging, where its only
        // consumer already is. `blocks` is the SAME `cus` the unfused step takes; that is the
        // constraint a fusion is most likely to break silently, and
        // `the_fusion_widens_the_op_and_never_narrows_the_state_step` compares the two directly.
        b.emit(DevOp::KdaStateStepG, cus, &[c_conv, c_fb, c_bb], |d| {
            d.t[0] = o;
            d.t[1] = mix[0];
            d.t[2] = mix[1];
            d.t[3] = mix[2];
            d.t[4] = f_raw;
            d.t[5] = b_raw;
            d.t[6] = st.state;
            d.t[7] = w.a_log;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = hd;
            d.i[3] = c.bv;
            // bit0: L2-normalize q and k in kernel, eps INSIDE the sqrt.
            // bit1 (PLOW_KDA_F_SEQ_ROWS): the rows are INDEPENDENT SEQUENCES, so the recurrence
            // strides its carried state per row instead of threading them all through one.
            d.i[4] = 1 | if seq_rows { 2 } else { 0 };
            d.i[5] = w.dt_bias;
            d.i[6] = gate_mode;
            // `i[7]`, the one free integer slot. Read only when the flags word carries
            // PLOW_KDA_F_SEQ_ROWS, so leaving it 0 on a non-batched blob keeps that blob
            // byte-identical (0 is a valid tensor index, hence the flag rather than a sentinel).
            d.i[7] = parked;
            d.f[0] = scale;
            d.f[1] = lower_bound;
        })
    } else {
        // P8a-c — the three short convs, one per stream, each over H*D = 12288 channels and each
        // spanning all 256 CUs. Coarse deps: every channel is identical work, so
        // `CounterGranularity`'s `collapse` says a fine schedule has an identical makespan.
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
            d.i[3] = gate_mode;
            d.f[0] = lower_bound;
        });

        // P10 — the recurrence. `blocks` is checked against the CU count here rather than left to
        // whatever the work happens to produce, because head-parallelism alone is the pathology.
        b.emit(
            DevOp::KdaStateStep,
            cus,
            &[c_conv[0], c_conv[1], c_conv[2], c_gate],
            |d| {
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
                // bit0: L2-normalize q and k in kernel, eps INSIDE the sqrt.
                // bit1 (PLOW_KDA_F_SEQ_ROWS): the rows are INDEPENDENT SEQUENCES, so the recurrence
                // strides its carried state per row instead of threading them all through one.
                d.i[4] = 1 | if seq_rows { 2 } else { 0 };
                d.f[0] = scale;
            },
        )
    };

    // P11 — output gate. Gated on P4 (whose GEMV has had the whole conv+gate+state chain to hide
    // under) and P10.
    let norm_cus: Vec<u32> = (0..c.gated_norm_blocks_rows(n_cu, t)).collect();
    let c_norm = b.emit(DevOp::KdaGatedNorm, norm_cus, &[c_g, c_step], |d| {
        d.t[0] = y;
        d.t[1] = o;
        d.t[2] = w.o_norm;
        d.t[3] = g_raw;
        d.i[0] = t;
        d.i[1] = nh;
        d.i[2] = hd;
        d.f[0] = c.eps;
    });

    // P12 — out projection. The residual is NOT here: see `emit_kda_layer`.
    let _ = all;
    let c_o = gemv(b, attn, y, w.o_proj, hi, p, c_norm);
    (c_o, attn)
}

/// Emit one KDA layer with the plain `next = hidden + attn` residual (P13).
///
/// This is the GLM-shaped block boundary and it is what every pre-K3 caller
/// wants. **Kimi-K3 must not use it**: its accumulator is the AttnRes prefix
/// sum, so the residual's left operand is `prefix_in` — the layer's input
/// *before* the attention-side AttnRes mix — not the `hidden` this reads. At a
/// snapshot layer there is no add at all, because the prefix RESTARTS at the
/// mixer output. Both differ from `hidden + attn` by the whole embedding state,
/// which is why `k3_emit_block` calls [`emit_kda_mixer`] and adds its own.
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
    let (c_o, attn) = emit_kda_mixer(
        b, c, w, st, act_prefix, t, hidden, None, n_cu, false, deps, false,
    );
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(
        DevOp::Residual,
        crate::k3::vec8_cus(&all, t * c.hidden),
        &[c_o],
        |d| {
            d.t[0] = next;
            d.t[1] = hidden;
            d.t[2] = attn;
            d.i[0] = t * c.hidden;
            d.f[0] = 1.0;
        },
    )
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
        _ => panic!(
            "KDA: unclassified tensor `{suffix}` — shard.rs defaults to REPLICATE, which \
                     for KDA is not a crash, just wrong math on >1 GPU"
        ),
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
        let disagree: Vec<u32> = (0..93)
            .filter(|&l| (l % 4 == 3) == is_kda_layer_0based(&kda, l))
            .collect();
        assert_eq!(disagree, vec![92]);
    }

    /// §7.3's discipline as a test: no proposal ships without its workgroup count.
    #[test]
    fn state_step_fills_the_chip_and_head_parallelism_does_not() {
        let c = k3();
        assert_eq!(c.proj(), 12288);
        assert_eq!(c.proj() / c.bv, 768, "work items");
        assert_eq!(
            c.state_step_blocks(256),
            256,
            "column-tiled: 100% of 256 CUs"
        );
        // One workgroup per head is the MlaMergeFold defect: 96/256 at TP1, 24/256 at TP4.
        assert_eq!(c.heads.min(256), 96);
        // TP8 with a FIXED BV does NOT hold 100% — the spec's table quietly assumes BV shrinks.
        let tp8 = KdaCfg { heads: 12, ..c };
        assert_eq!(
            tp8.state_step_blocks(256),
            96,
            "12*128/16 = 96 items, 37.5%"
        );
        assert_eq!(
            KdaCfg { bv: 8, ..tp8 }.state_step_blocks(256),
            192,
            "BV=8 restores 192"
        );
        assert_eq!(c.gated_norm_blocks_rows(256, 1), 12);
        assert_eq!(tp8.gated_norm_blocks_rows(256, 1), 2);
        assert_eq!(tp8.gated_norm_blocks_rows(256, 32), 48);
        assert_eq!(tp8.gated_norm_blocks_rows(256, 128), 192);
        assert_eq!(tp8.gated_norm_blocks_rows(256, 512), 256);
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
        assert_eq!(
            per_layer * 69,
            474_808_320,
            "452.8 MiB/seq — CONSTANT in context length"
        );
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
        assert!(
            n.iter().all(|s| s.starts_with("language_model.")),
            "K3 has ZERO `model.` tensors"
        );
        assert!(n.contains(&"language_model.model.layers.0.self_attn.A_log".to_string()));
    }

    /// The fusion gate is a REFUSAL at emit time, not a runtime fallback: past the LDS arena the
    /// fused op stages garbage rows and stays finite (§6g-BATCH), so the bound is checked here.
    ///
    /// TWO independent refusals now, and the second is the one that makes a prefill bucket safe.
    /// The LDS arena declines above 10 rows at `hidden = 7168` — which already covers every rung
    /// of the ladder, whose floor is 128 — but it ACCEPTS 2..10, and `GemvQkvg` is compiled into
    /// the DECODE interpreter only. A `K3_PREFILL=8` bucket would have emitted a packet the
    /// prefill object has no `case` for, and this interpreter's `default:` writes nothing: q, k, v
    /// and g all silently unwritten. So the gate is `t == 1` AND the arena, not the arena alone.
    #[test]
    fn qkvg_fusion_is_bounded_by_the_lds_arena_and_by_the_decode_bucket() {
        assert!(fuse_qkvg(1, 7168), "decode must fuse");
        // The arena bound, stated on the numbers it is derived from. It is no longer what DECIDES
        // t = 10 — `t != 1` gets there first — but it is still the reason the fused op cannot be
        // widened to prefill by simply deleting that check.
        assert!(crate::gemv_staged_rows(10) as u64 * 7168 <= crate::gm_lds_halves());
        assert!(crate::gemv_staged_rows(11) as u64 * 7168 > crate::gm_lds_halves());
        // Every T above decode refuses, INCLUDING the ones the arena would have allowed.
        for t in [2u32, 8, 10, 11, 128, 8192] {
            assert!(
                !fuse_qkvg(t, 7168),
                "T={t}: GemvQkvg exists only in the decode object"
            );
        }
    }

    /// A T-row KDA mixer emits NO decode-only opcode.
    ///
    /// `GemvQkvg` (100) and `Gemv` (2) are the two, and they fail differently: the prefill
    /// interpreter has no `case` for `GEMV_QKVG` at all, so it writes NOTHING, while `Gemv` DOES
    /// have a prefill arm but is compiled at a fixed row bucket (`PLOW_GEMV_MM <= 16`) and would
    /// process the first few rows of a 128-row packet and leave the rest holding the arena. Both
    /// are finite, plausible and wrong; neither faults.
    #[test]
    fn a_t_row_kda_mixer_emits_gemms_and_only_supported_prefill_kda() {
        let c = k3();
        for t in [128u32, 1024] {
            let mut b = Builder::new(256);
            let all: Vec<u32> = (0..256).collect();
            let hidden = b.tensor("in.hidden", t as u64 * 7168 * 2);
            let next = b.tensor("act.next", t as u64 * 7168 * 2);
            let w = declare_kda_weights(&mut b, &c, "l.self_attn.", "l.");
            let st = declare_kda_state(&mut b, &c, "kda.0.", 1);
            let seed = b.emit(DevOp::Nop, all, &[], |_| {});
            emit_kda_layer(
                &mut b,
                &c,
                &w,
                &st,
                "act.pf.",
                t,
                hidden,
                next,
                256,
                &[seed],
            );
            let p = b.finish();
            let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
            assert_eq!(n(DevOp::Gemv), 0, "T={t}: no decode GEMV survives");
            assert_eq!(
                n(DevOp::GemvQkvg),
                0,
                "T={t}: the 4-stream op is decode-only"
            );
            // EIGHT projections at T rows: q, k, v, g, f_a, beta, f_b, o_proj — all tiled GEMMs,
            // and every one of them carries the real row count.
            let gemm: Vec<_> = p
                .insts
                .iter()
                .filter(|i| crate::gemm_family_ops().contains(&i.op))
                .collect();
            assert_eq!(gemm.len(), 8, "T={t}: eight projections, none fused");
            assert!(
                gemm.iter().all(|i| i.i[0] == t),
                "T={t}: M must be the row count"
            );
            assert_eq!(n(DevOp::KdaConv3), 1);
            assert_eq!(n(DevOp::KdaGatedNorm), 1);
            if t >= 512 {
                assert_eq!(n(DevOp::KdaStateStepG), 0);
                for op in [
                    DevOp::KdaChunkPrepare,
                    DevOp::KdaChunkIntra,
                    DevOp::KdaChunkWu,
                    DevOp::KdaChunkCarry,
                ] {
                    assert_eq!(n(op), 1, "T={t}: missing default gfx950 {op:?}");
                }
            } else {
                assert_eq!(n(DevOp::KdaStateStepG), 1);
                assert_eq!(n(DevOp::KdaChunkPrepare), 0);
            }
        }
    }

    #[test]
    fn chunk_kda_override_is_shape_gated() {
        let build = |t: u32, seq_rows: bool, enabled: bool| {
            let c = KdaCfg {
                heads: 12,
                bv: 8,
                ..k3()
            };
            let mut b = Builder::new(256);
            let hidden = b.tensor("in.hidden", t as u64 * c.hidden as u64 * 2);
            let w = declare_kda_weights(&mut b, &c, "p.", "l.");
            let st = declare_kda_state(
                &mut b,
                &c,
                "kv.0.",
                u64::from(seq_rows.then_some(t).unwrap_or(1)),
            );
            let seed = b.emit(DevOp::Nop, (0..256).collect(), &[], |_| {});
            emit_kda_mixer_ex(
                &mut b,
                &c,
                &w,
                &st,
                "act.pf.",
                t,
                hidden,
                None,
                256,
                false,
                &[seed],
                true,
                enabled,
                false,
                seq_rows,
            );
            b.finish()
        };
        let count = |p: &packet::devbuild::Program, op: DevOp| {
            p.insts.iter().filter(|i| i.op == op as u16).count()
        };

        let chunk = build(512, false, true);
        for op in [
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ] {
            assert_eq!(count(&chunk, op), 1, "missing {op:?}");
        }
        assert_eq!(count(&chunk, DevOp::KdaStateStepG), 0);

        for legacy in [
            build(512, false, false),
            build(128, false, true),
            build(576, true, true),
        ] {
            assert_eq!(count(&legacy, DevOp::KdaStateStepG), 1);
            assert_eq!(count(&legacy, DevOp::KdaChunkPrepare), 0);
        }
        assert_eq!(count(&build(511, false, true), DevOp::KdaStateStepG), 1);
        for t in [513, 8191] {
            let ragged = build(t, false, true);
            assert_eq!(count(&ragged, DevOp::KdaChunkPrepare), 1, "T={t}");
            assert_eq!(count(&ragged, DevOp::KdaStateStepG), 0, "T={t}");
        }
    }

    /// The op graph: and every new opcode reachable.
    #[test]
    fn one_layer_emits_the_expected_graph() {
        let c = k3();
        let mut b = Builder::new(256);
        let all: Vec<u32> = (0..256).collect();
        let hidden = b.tensor("in.hidden", 7168 * 2);
        let next = b.tensor("act.next", 7168 * 2);
        let w = declare_kda_weights(
            &mut b,
            &c,
            "language_model.model.layers.0.self_attn.",
            "language_model.model.layers.0.",
        );
        let st = declare_kda_state(&mut b, &c, "kda.0.", 1);
        let seed = b.emit(DevOp::Nop, all, &[], |_| {});
        emit_kda_layer(
            &mut b,
            &c,
            &w,
            &st,
            "act.kda0.",
            1,
            hidden,
            next,
            256,
            &[seed],
        );
        let p = b.finish();
        let ops: Vec<u16> = p.insts.iter().map(|i| i.op).collect();
        // The DEFAULT spelling, whatever the environment says — `emit_kda_layer` reads the knob
        // once and this reads the same call. Both spellings are exercised by
        // `the_fusion_widens_the_op_and_never_narrows_the_state_step`, which drives the seam
        // directly rather than racing an env var against every other test in the binary.
        let fused = fuse_kda();
        // FOUR plain GEMVs (f_a, beta, f_b, o_proj) — q/k/v/g are one fused packet, which is
        // where the 207-packets-per-token saving comes from.
        assert_eq!(ops.iter().filter(|&&o| o == DevOp::Gemv as u16).count(), 4);
        let f = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::GemvQkvg as u16)
            .expect("P1-P4 must fuse at t=1");
        assert_eq!(
            f.blocks, 256,
            "the fused projection must stay chip-wide, not narrow"
        );
        assert_eq!(
            [f.i[1], f.i[3], f.i[4], f.i[5]],
            [c.proj(); 4],
            "Nq/Nk/Nv/Ng"
        );
        assert_eq!(f.i[2], c.hidden, "K");
        assert_eq!(
            [f.t[2], f.t[4], f.t[6]],
            [w.q_proj, w.k_proj, w.v_proj],
            "W_q/W_k/W_v"
        );
        assert_eq!(
            f.i[6], w.g_proj,
            "the ninth pointer is the W_g HANDLE, not an integer"
        );
        assert_ne!(
            f.i[6],
            packet::dev::TENSOR_NONE,
            "the arm traps on an absent W_g"
        );
        let chain: &[DevOp] = if fused {
            &[DevOp::KdaConv3, DevOp::KdaStateStepG, DevOp::KdaGatedNorm]
        } else {
            &[
                DevOp::KdaConv,
                DevOp::KdaGate,
                DevOp::KdaStateStep,
                DevOp::KdaGatedNorm,
            ]
        };
        for op in chain {
            assert!(
                ops.contains(&(*op as u16)),
                "{op:?} is not emitted — an opcode nothing \
                    reaches is how Mamba2Scan became dead code"
            );
        }
        // ...and NOTHING from the other spelling, or the layer would run the chain twice.
        let other: &[DevOp] = if fused {
            &[DevOp::KdaConv, DevOp::KdaGate, DevOp::KdaStateStep]
        } else {
            &[DevOp::KdaConv3, DevOp::KdaStateStepG]
        };
        for op in other {
            assert!(
                !ops.contains(&(*op as u16)),
                "{op:?} escaped the other branch"
            );
        }
        // The state step must not be stranded on a handful of CUs — and the FUSED one must not
        // narrow it, which is the constraint a fusion is most likely to break silently.
        let step_op = if fused {
            DevOp::KdaStateStepG
        } else {
            DevOp::KdaStateStep
        };
        let step = p.insts.iter().find(|i| i.op == step_op as u16).unwrap();
        assert_eq!(step.blocks, 256, "column tiling, not head parallelism");
        // Every other KDA op spans the chip too.
        for op in if fused {
            vec![DevOp::KdaConv3]
        } else {
            vec![DevOp::KdaConv, DevOp::KdaGate]
        } {
            let i = p.insts.iter().find(|i| i.op == op as u16).unwrap();
            assert_eq!(i.blocks, 256, "{op:?}");
        }
    }

    /// The fusion moves PACKETS, not work, and both halves of that are checkable here.
    ///
    /// Per-CU channel count is the number the knob contract asks for: `GLM_GROUP=1` removed 38% of
    /// the ops for +2.88 ms by collapsing disjoint CU slices into a loop, so a merge is only safe
    /// if the op gets WIDER on the same CUs. It does — 3x, exactly the `GemvQkvg` shape.
    #[test]
    fn the_fusion_widens_the_op_and_never_narrows_the_state_step() {
        let c = k3();
        let build = |fuse: bool| {
            let mut b = Builder::new(256);
            let all: Vec<u32> = (0..256).collect();
            let hidden = b.tensor("in.hidden", 7168 * 2);
            let w = declare_kda_weights(&mut b, &c, "p.", "l.");
            let st = declare_kda_state(&mut b, &c, "kda.0.", 1);
            let seed = b.emit(DevOp::Nop, all, &[], |_| {});
            emit_kda_mixer_ex(
                &mut b,
                &c,
                &w,
                &st,
                "act.kda0.",
                1,
                hidden,
                None,
                256,
                false,
                &[seed],
                fuse,
                false,
                false,
                false,
            );
            b.finish()
        };
        let k3_ops = [
            DevOp::KdaConv,
            DevOp::KdaGate,
            DevOp::KdaStateStep,
            DevOp::KdaGatedNorm,
            DevOp::KdaConv3,
            DevOp::KdaStateStepG,
        ];
        let count = |p: &packet::devbuild::Program| {
            p.insts
                .iter()
                .filter(|i| k3_ops.iter().any(|o| *o as u16 == i.op))
                .count()
        };
        let (un, fu) = (build(false), build(true));
        assert_eq!(
            count(&un),
            6,
            "three convs, a gate, the step, the gated norm"
        );
        assert_eq!(count(&fu), 3, "one conv, the gated step, the gated norm");

        // PER-CU WORK. The conv's channel count per CU RISES 3x on the same 256 CUs; the state
        // step's item count per CU is IDENTICAL. Neither is a loop-axis collapse.
        let one = un
            .insts
            .iter()
            .find(|i| i.op == DevOp::KdaConv as u16)
            .unwrap();
        let three = fu
            .insts
            .iter()
            .find(|i| i.op == DevOp::KdaConv3 as u16)
            .unwrap();
        assert_eq!(
            one.blocks, three.blocks,
            "the same 256 CUs, before and after"
        );
        let per_cu = |chans: u32, blocks: u16| chans.div_ceil(blocks as u32);
        assert_eq!(
            per_cu(one.i[1], one.blocks),
            48,
            "12288 channels over 256 CUs"
        );
        assert_eq!(
            per_cu(3 * three.i[1], three.blocks),
            144,
            "36864 channels over 256 CUs"
        );

        let s0 = un
            .insts
            .iter()
            .find(|i| i.op == DevOp::KdaStateStep as u16)
            .unwrap();
        let s1 = fu
            .insts
            .iter()
            .find(|i| i.op == DevOp::KdaStateStepG as u16)
            .unwrap();
        assert_eq!(
            s0.blocks, s1.blocks,
            "the fused step must not narrow the slice map"
        );
        assert_eq!(
            [s1.i[1], s1.i[2], s1.i[3]],
            [s0.i[1], s0.i[2], s0.i[3]],
            "H, D and BV"
        );
        // The demoted handles are HANDLES and all four are present — the arm traps otherwise.
        for k in 4..8 {
            assert_ne!(three.i[k], packet::dev::TENSOR_NONE, "KdaConv3 i{k}");
        }
        assert_ne!(s1.i[5], packet::dev::TENSOR_NONE, "dt_bias rides in i5");
        assert_eq!(s1.f[1], -5.0, "gate_lower_bound rides in f1");
    }

    #[test]
    fn b1_double_buffered_state_emits_one_typed_packet_and_both_conv_banks() {
        for seq_rows in [false, true] {
            let c = KdaCfg {
                heads: 12,
                bv: 8,
                ..k3()
            };
            let mut b = Builder::new(304);
            let hidden = b.tensor("in.hidden", 7168 * 2);
            let pos = b.tensor("in.pos", 4);
            let w = declare_kda_weights(&mut b, &c, "p.", "l.");
            let st = declare_kda_state_db(&mut b, &c, "kv.0.", 1, pos);
            let seed = b.emit(DevOp::Nop, (0..304).collect(), &[], |_| {});
            emit_kda_mixer_ex(
                &mut b,
                &c,
                &w,
                &st,
                "act.kda0.",
                1,
                hidden,
                None,
                304,
                false,
                &[seed],
                true,
                false,
                false,
                seq_rows,
            );
            let p = b.finish();
            let fused: Vec<_> = p
                .insts
                .iter()
                .filter(|i| i.op == DevOp::KdaConvStateStepG as u16)
                .collect();
            assert_eq!(fused.len(), 1);
            assert_eq!(
                fused[0].i[..7],
                [1, 12, 128, 8, 1 | if seq_rows { 2 } else { 0 }, 4, 1]
            );
            assert_eq!(fused[0].blocks, 192, "one BV8 state column per block");
            for old in [DevOp::KdaConv3, DevOp::KdaStateStepG] {
                assert!(p.insts.iter().all(|i| i.op != old as u16), "{old:?}");
            }
            for stream in ["q", "k", "v"] {
                assert!(p
                    .tensors
                    .iter()
                    .any(|t| t.name == format!("kv.0.conv_state.{stream}")));
                assert!(p
                    .tensors
                    .iter()
                    .any(|t| t.name == format!("kv.0.conv_state_alt.{stream}")));
            }
            let desc = &p.tensors[fused[0].t[7] as usize];
            assert_eq!(desc.bytes, 13 * 4);
            let init = desc.init.as_ref().expect("descriptor initializer");
            assert_eq!(init.len(), 13 * 4);
            let parked = u32::from_le_bytes(init[48..52].try_into().unwrap());
            if seq_rows {
                assert_eq!(p.tensors[parked as usize].name, "in.parked");
            } else {
                assert_eq!(parked, packet::dev::TENSOR_NONE_I);
            }
        }
    }

    #[test]
    fn double_buffered_fusion_is_shape_gated_not_model_gated() {
        for (heads, head_dim, conv_w, n_cu, blocks) in [
            (1, 64, 1, 304, 8),
            (12, 128, 4, 304, 192),
            (16, 256, 8, 304, 304),
        ] {
            let c = KdaCfg {
                heads,
                head_dim,
                conv_w,
                bv: WG_WAVES,
                ..k3()
            };
            assert!(c.supports_conv_state_step_db());
            let mut b = Builder::new(n_cu);
            let hidden = b.tensor("in.hidden", c.hidden as u64 * 2);
            let pos = b.tensor("in.pos", 4);
            let w = declare_kda_weights(&mut b, &c, "p.", "l.");
            let st = declare_kda_state_db(&mut b, &c, "kv.0.", 1, pos);
            let seed = b.emit(DevOp::Nop, (0..n_cu).collect(), &[], |_| {});
            emit_kda_mixer_ex(
                &mut b,
                &c,
                &w,
                &st,
                "act.kda0.",
                1,
                hidden,
                None,
                n_cu,
                false,
                &[seed],
                true,
                false,
                false,
                false,
            );
            let p = b.finish();
            let fused = p
                .insts
                .iter()
                .find(|i| i.op == DevOp::KdaConvStateStepG as u16)
                .expect("shape must emit the fused recurrent packet");
            assert_eq!(fused.i[1], heads);
            assert_eq!(fused.i[2], head_dim);
            assert_eq!(fused.i[5], conv_w);
            assert_eq!(fused.blocks, blocks);
        }

        for c in [
            KdaCfg { heads: 0, ..k3() },
            KdaCfg {
                head_dim: 32,
                ..k3()
            },
            KdaCfg { bv: 16, ..k3() },
            KdaCfg { conv_w: 0, ..k3() },
            KdaCfg { conv_w: 9, ..k3() },
        ] {
            assert!(!c.supports_conv_state_step_db());
        }
    }

    fn build_decode_fused_probe(
        c: &KdaCfg,
        rows: u32,
        seq_rows: bool,
        enabled: bool,
    ) -> packet::devbuild::Program {
        let mut b = Builder::new(256);
        let hidden = b.tensor("in.hidden", rows as u64 * c.hidden as u64 * 2);
        let w = declare_kda_weights(&mut b, c, "p.", "l.");
        let st = declare_kda_state(
            &mut b,
            c,
            "kv.0.",
            u64::from(seq_rows.then_some(rows).unwrap_or(1)),
        );
        let seed = b.emit(DevOp::Nop, (0..256).collect(), &[], |_| {});
        emit_kda_mixer_ex(
            &mut b,
            c,
            &w,
            &st,
            "act.kda0.",
            rows,
            hidden,
            None,
            256,
            false,
            &[seed],
            true,
            false,
            enabled,
            seq_rows,
        );
        b.finish()
    }

    #[test]
    fn standalone_decode_fusion_is_shape_gated_and_falls_back_byte_identically() {
        let supported = KdaCfg {
            heads: 12,
            head_dim: 128,
            conv_w: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1.0e-5,
            bv: 8,
            ..k3()
        };
        assert!(supported.supports_decode_fused(1, 256, false));
        assert!(supported.supports_decode_fused(8, 256, true));
        assert!(!supported.supports_decode_fused(8, 256, false));
        assert!(!supported.supports_decode_fused(2, 256, true));

        let unsupported = [
            KdaCfg {
                head_dim: 64,
                ..supported.clone()
            },
            KdaCfg {
                bv: 16,
                ..supported.clone()
            },
            KdaCfg {
                conv_w: 3,
                ..supported.clone()
            },
            KdaCfg {
                gate_lower_bound: None,
                ..supported.clone()
            },
        ];
        for c in &unsupported {
            let legacy = build_decode_fused_probe(c, 1, false, false);
            let requested = build_decode_fused_probe(c, 1, false, true);
            assert_eq!(requested.to_blob(), legacy.to_blob());
            assert!(requested
                .insts
                .iter()
                .all(|i| i.op != DevOp::KdaDecodeFused as u16));
        }
        let legacy_b8 = build_decode_fused_probe(&supported, 8, false, false);
        let invalid_b8 = build_decode_fused_probe(&supported, 8, false, true);
        assert_eq!(invalid_b8.to_blob(), legacy_b8.to_blob());
    }

    #[test]
    fn standalone_decode_fusion_has_exact_slots_rungs_deps_and_a_pure_segment() {
        for (rows, seq_rows, expected_blocks) in [(1, false, 12), (8, true, 96)] {
            let c = KdaCfg {
                heads: 12,
                bv: 8,
                ..k3()
            };
            let p = build_decode_fused_probe(&c, rows, seq_rows, true);
            let fused_index = p
                .insts
                .iter()
                .position(|i| i.op == DevOp::KdaDecodeFused as u16)
                .expect("validated rung must emit one standalone boundary");
            assert_eq!(
                p.insts
                    .iter()
                    .filter(|i| i.op == DevOp::KdaDecodeFused as u16)
                    .count(),
                1
            );
            for old in [DevOp::KdaConv3, DevOp::KdaStateStepG, DevOp::KdaGatedNorm] {
                assert!(p.insts.iter().all(|i| i.op != old as u16), "{old:?}");
            }
            let fused = &p.insts[fused_index];
            assert_eq!(fused.blocks, expected_blocks);
            assert_eq!(
                fused.i,
                [rows, 12, 128, 8, 4, 1 | if seq_rows { 2 } else { 0 }, 1, 2]
            );
            assert_eq!(fused.f, [1.0 / (128.0f32).sqrt(), -5.0]);
            assert_eq!(fused.j, [0, 1.0e-5f32.to_bits()]);
            let desc = &p.tensors[fused.t[7] as usize];
            assert_eq!(desc.bytes, 11 * 4);
            assert_eq!(desc.init.as_ref().unwrap().len(), 11 * 4);

            let producer_for = |slot: usize, handle: u32| {
                p.insts[..fused_index].iter().position(|i| {
                    i.t[0] == handle || (i.op == DevOp::GemvQkvg as u16 && i.t[slot] == handle)
                })
            };
            for (slot, handle) in [
                (0, fused.t[1]),
                (3, fused.t[2]),
                (5, fused.t[3]),
                (7, fused.t[4]),
                (0, fused.t[5]),
            ] {
                assert!(
                    producer_for(slot, handle).is_some(),
                    "missing producer for t{slot}"
                );
            }

            let output_index = fused_index + 1;
            let output = &p.insts[output_index];
            assert_eq!(output.t[1], fused.t[0], "o_proj must consume fused y");
            assert_eq!(output.op, DevOp::Gemv as u16);

            let fused_segs: std::collections::BTreeSet<u16> = p
                .stream
                .iter()
                .filter(|e| e.inst as usize == fused_index)
                .map(|e| e.seg)
                .collect();
            assert_eq!(fused_segs.len(), 1);
            let seg = *fused_segs.iter().next().unwrap();
            assert!(p
                .stream
                .iter()
                .filter(|e| e.seg == seg)
                .all(|e| { p.insts[e.inst as usize].op == DevOp::KdaDecodeFused as u16 }));
            assert!(p
                .stream
                .iter()
                .filter(|e| (e.inst as usize) < fused_index)
                .all(|e| e.seg < seg));
            assert!(p
                .stream
                .iter()
                .filter(|e| e.inst as usize == output_index)
                .all(|e| e.seg > seg));
            assert_eq!((fused.wait_len, fused.succ_len), (0, 0));
            for e in p.stream.iter().filter(|e| e.inst as usize == fused_index) {
                assert_eq!((e.wait_len, e.succ_len), (0, 0));
                assert_eq!(e.flags & packet::dev::SE_XCTR, 0);
            }
            for e in &p.stream {
                for wait in &p.waits[e.wait_ofs as usize..][..e.wait_len as usize] {
                    let producer_seg = p
                        .stream
                        .iter()
                        .find(|producer| producer.inst == wait.id)
                        .expect("coarse wait producer must have a stream entry")
                        .seg;
                    assert_eq!(
                        producer_seg, e.seg,
                        "host-ordered cross-segment edge leaked into the interpreter protocol"
                    );
                }
            }
        }
    }

    /// TP8 is the shape K3 decode runs, and it is where a fixed `BV` strands the step on 96 of 256
    /// CUs. The fused step inherits `state_step_blocks`, so it inherits the shrink too.
    #[test]
    fn the_fused_step_tracks_bv_down_at_tp8() {
        let tp8 = KdaCfg {
            heads: 12,
            bv: 8,
            ..k3()
        };
        assert_eq!(tp8.state_step_blocks(256), 192, "12*128/8 = 192 items");
        assert_eq!(
            KdaCfg { bv: 16, ..tp8 }.state_step_blocks(256),
            96,
            "a fixed BV strands it"
        );
    }
}
