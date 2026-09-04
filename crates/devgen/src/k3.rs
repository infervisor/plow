//! Kimi-K3 BLOCK structure — AttnRes and the `situ` GLU.
//!
//! `crates/devgen/src/kda.rs` emits K3's MIXER. This emits the BLOCK AROUND IT, and that is where
//! K3 stops looking like every other model in this tree:
//!
//! ```text
//!   a K3 layer is NOT  `residual + attn` then `residual + mlp`.
//! ```
//!
//! `attn_res_block_size: 12` routes every layer through `_forward_attn_residual`
//! (`modeling_kimi_linear.py:973`). The plain residual ADD is replaced by **AttnRes** — a softmax
//! mix over the running prefix sum and up to 8 snapshots of it — applied TWICE per layer, and the
//! block's OUTPUT is the prefix sum. AMD's day-0 post names the structure and confirms the period:
//! *"stores one block residual every 12 layers"*.
//!
//! It is applied a THIRD time, ONCE PER MODEL, after the layer loop and before `model.norm`
//! (`KimiLinearModel.forward` -> `_apply_output_attn_res`, `modeling_kimi_linear.py:1215`), with
//! the model's own `output_attn_res_norm`/`output_attn_res_proj` pair. That site is not decorative:
//! the prefix sum RESTARTS at every snapshot layer, so what the loop leaves behind is only the
//! partial sum since the LAST snapshot — layers 84..92 of 93. Everything before it lives in the
//! ring and the output mix is the only reader that puts it back.
//!
//! # The control flow, and the two shapes it produces
//!
//! ```text
//!   prefix_in                                        (the layer input)
//!   if nb > 0:  h = AttnRes(prefix_in, blkres, attn_score_w)   else  h = prefix_in
//!   if l % B == 0:  blkres.push(prefix_in) ; prefix = None     (a SNAPSHOT layer)
//!   attn = KDA_or_MLA(input_layernorm(h))
//!   prefix = (prefix_in + attn) if prefix is not None else attn
//!   h2  = AttnRes(prefix, blkres, mlp_score_w)
//!   ffn = MLP_or_MoE(post_attention_layernorm(h2))
//!   out = prefix + ffn
//!
//!   # once, after the last layer:
//!   logits = lm_head(model.norm(AttnRes(out, blkres, output_score_w)))
//! ```
//!
//! Layer 0 is a snapshot layer with an EMPTY `blkres`, so it skips the first AttnRes AND resets
//! `prefix` — its output is `attn + ffn`, and the layer input reaches it only through the mix.
//! Layer 1 is neither, so both AttnRes calls are live and `prefix` accumulates.
//!
//! **The two calls use DIFFERENT weights** (`self_attention_res_*` vs `mlp_res_*`). Layer 0 never
//! reads the first pair at all, so a loader that derives coverage from the tensor set will report
//! two dead tensors on that layer and be right.
//!
//! # Where a wrong wiring hides, measured
//!
//! At a SNAPSHOT layer the block output is `attn + ffn` while the plain wiring gives
//! `hidden + attn + ffn`: a **1.0** relative difference, which any output check catches.
//! At a NON-snapshot layer `prefix = prefix_in + attn`, so the block output is
//! `prefix_in + attn + ffn` — **exactly what the plain wiring produces**. Measured on real layer-1
//! weights: **3.0e-3** at the block output, against **8.1e-1** and **7.6e-1** at the two AttnRes
//! outputs themselves.
//!
//! > A block-output-only gate does not see AttnRes at 68 of K3's 93 layers. The gate has to diff
//! > the sub-layer INPUTS. `runtime/tests/k3_moe_block_gfx950_test.c` does, and says so.
//!
//! # `situ`, and why it is an opcode and not an `act` code
//!
//! Every GLU site in this tree computes `act(g) * u` and selects the activation with a two-value
//! ternary. `situ` transforms the UP branch too — `A(g) * B(u)` — so the EXPRESSION SHAPE changes,
//! not just the function. A third act code alone would apply the gate transform and leave `up`
//! un-clipped: a small error at `|u| < 25` that grows with the tail, i.e. plausible output and the
//! wrong model. [`DevOp::SituGlu`] makes the omission impossible.
//!
//! The routed experts cannot use it — their GLU is inside the expert kernel — so `op_moe.h` grew
//! `PLOW_MOE_ACT_SITU = 2` and a PAIR-form epilogue (`moe_glu`), with the betas riding in `f0`/`f1`
//! (free on every GLU-family op). `moe_act` itself returns **NaN** for that code, on purpose: any
//! epilogue not converted to the pair form poisons its output instead of silently computing
//! `gelu_tanh(g) * u`.

use crate::emit_config;
use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::Builder;
use packet::rope::{GenTensor, RopeScale};

/// K3's block-structure constants (`text_config`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct K3BlockCfg {
    /// `hidden_size` — 7168.
    pub hidden: u32,
    /// `rms_norm_eps` — 1e-5, used by both norms and by AttnRes's per-row normalization.
    pub eps: f32,
    /// `attn_res_block_size` — 12. A snapshot is taken when `layer_idx % this == 0`.
    pub attn_res_block_size: u32,
    /// `activation_situ_beta` — 4.0. Clips the GATE branch to +-beta.
    pub situ_beta: f32,
    /// `activation_situ_linear_beta` — 25.0. Clips the UP branch. `<= 0` disables it, which is
    /// what `linear_beta is None` means in the reference.
    pub situ_linear_beta: f32,
}

/// The AttnRes kernel's compile-time bound on `nb`, mirroring `PLOW_ATTNRES_MAXB` in
/// `runtime/amd/op_k3.h`.
///
/// K3 needs at most `ceil(93/12) = 8`. The arm RETURNS on a larger `nb` rather than indexing off
/// the end of its LDS carve, so the emitter has to refuse first — a silent no-write is the failure
/// this tree keeps finding.
pub const K3_ATTNRES_MAXB: u32 = 16;

/// `PLOW_MOE_ACT_SITU` (`runtime/amd/op_moe.h`) — the act code every K3 expert GLU carries.
///
/// It is NOT an ordinary `moe_act` value: `moe_act` returns the GATE transform alone and situ
/// transforms the UP branch too, so only the epilogues converted to the PAIR form `moe_glu` may
/// see it. `moe_act` returns NaN for it, on purpose, so an unconverted epilogue poisons instead of
/// silently computing `gelu_tanh(g) * u`.
pub const K3_MOE_ACT_SITU: u32 = 2;

/// `MoeEnc::Mxfp4` as it travels in a packet's encoding slot. K3 ships mxfp4 experts.
pub const K3_MOE_ENC_MXFP4: u32 = 2;

/// `PLOW_THREADS` (`runtime/amd/amd_common.h`) — 512, i.e. 8 waves of 64.
const K3_WG_THREADS: u32 = 512;

/// Elements each thread of a `bf16v8` elementwise body owns. `d_residual`, `d_situ_glu` and
/// `d_mla_out_gate` all step `(slice * PLOW_THREADS + threadIdx.x) * 8`, so one workgroup covers
/// `PLOW_THREADS * 8` elements, not `PLOW_THREADS`.
const K3_ELEM_PER_THREAD: u32 = 8;

/// `PLOW_K3_WGFIT=0` restores the un-narrowed `(0..n_cu)` widths, so the control arm of an A/B
/// comes out of the SAME `plowc` binary as the narrowed one. Same shape and same reason as GLM's
/// Workgroup-fit: always enabled (hardcoded — was `PLOW_K3_WGFIT`, never disabled).
pub(crate) fn k3_wgfit() -> bool {
    true
}

/// The workgroups a **row-partitioned norm** can actually use.
///
/// `d_rmsnorm` (`runtime/amd/op_norm.h:65`) walks `for (row = slice; row < rows; row += nblk)` and
/// reduces each row inside ONE workgroup via `block_sum`. So a packet with `rows` rows saturates
/// at `rows` workgroups: at K3's decode `rows = 1`, slice 0 does the whole norm and slices 1..255
/// arrive, take the acquire, find nothing to do, and signal. That is not free — the measured
/// per-packet floor is 6.5-6.9 us of which **~3.0 us is 256-way cross-XCD producer convergence**
/// (`runtime/bench/interp_dispatch_floor.hip:18`), and it is paid 302 times per K3 token.
///
/// Pure narrowing and BIT-IDENTICAL: slices `0..need` own exactly the rows they own today.
/// Prefill is inert — `rows = T` saturates the machine at every bucket K3 emits.
pub(crate) fn norm_cus(cus: &[u32], rows: u32) -> Vec<u32> {
    let need = (rows.max(1) as usize).min(cus.len());
    if !k3_wgfit() {
        return cus.to_vec();
    }
    cus[..need].to_vec()
}

/// The workgroups a flat `bf16v8` elementwise `[n]` packet can actually use.
///
/// Same rule as GLM's `elem_cus`, with the ×8 the vector bodies actually stride by — `elem_cus`
/// is written for `d_moe_combine`'s one-element-per-thread map and would over-dispatch these
/// eight-fold. At K3's hidden 7168 this is 2 workgroups against the 256 handed in today, and at
/// the latent 3584 it is 1.
///
/// Pure narrowing, BIT-IDENTICAL, and an EMIT-TIME change: an already-built `.pkt` keeps its old
/// width until re-emitted.
pub(crate) fn vec8_cus(cus: &[u32], n: u32) -> Vec<u32> {
    let per_wg = K3_WG_THREADS * K3_ELEM_PER_THREAD;
    let need = (n.div_ceil(per_wg).max(1) as usize).min(cus.len());
    if !k3_wgfit() {
        return cus.to_vec();
    }
    cus[..need].to_vec()
}

/// The workgroups a `d_moe_combine` packet can actually use — one element per thread, so
/// `ceil(n / PLOW_THREADS)`. This is GLM's `elem_cus` rule; K3's combine runs at the LATENT
/// width, which is 7 workgroups at 3584 rather than 256.
pub(crate) fn combine_cus(cus: &[u32], n: u32) -> Vec<u32> {
    let need = (n.div_ceil(K3_WG_THREADS).max(1) as usize).min(cus.len());
    if !k3_wgfit() {
        return cus.to_vec();
    }
    cus[..need].to_vec()
}

/// The workgroups a `HeadNormRope`-family packet can actually use: one wave per `(token, head)`
/// pair, so `ceil(ntok * nhead / PLOW_WAVES)`.
///
/// `start` keeps a concurrent q/k pair on DISJOINT slices — the q-side and k-side ropes are gated
/// on the same producer and run together, so narrowing both to `0..need` would serialise them on
/// the same workgroups. Same rule, same reason and same constant as GLM's `rope_cus`, which
/// measured **-1.148 ms/token (-4.3%)** on its own, token-identical.
pub(crate) fn k3_rope_cus(cus: &[u32], start: usize, ntok: u32, nhead: u32) -> Vec<u32> {
    if !k3_wgfit() {
        return cus.to_vec();
    }
    let waves = K3_WG_THREADS / 64;
    let items = ntok as u64 * nhead as u64;
    let need = (items.div_ceil(waves as u64).max(1) as usize).min(cus.len());
    let start = start.min(cus.len() - need);
    cus[start..start + need].to_vec()
}

/// May the MLA mixer's THREE flash-feeding down-projections off the pre-norm collapse into one
/// [`DevOp::GemvQkv`] packet?
///
/// `q_a`, `kv_a` and `k_rope_d` all read the SAME `x` with `K = hidden`, so their output columns
/// concatenate and the three packets become one — byte-exact, because a concatenated GEMV computes
/// each column exactly as the separate one did. This is the same merge `PLOW_KDA_FUSE_QKVG` makes on
/// the 69 KDA layers, and that one measured **-22..27%**, far more than its packet-count arithmetic
/// predicts: the win is that the fused op is WIDER, so a wave owns more columns of the row and issues
/// enough loads to stop being latency-bound. K3's MLA down-projections are the narrowest ops in the
/// model — `k_rope_d` is `N = 64` and `kv_a` is `N = 512` against hidden 7168 — exactly the shape
/// that starves.
///
/// # DEFAULT OFF, because it LOSES at the context this model is tuned for
///
/// Built, measured, and it is a ctx-dependent trade with the crossover inside the range we serve.
/// Interleaved repeats on the real TP8 asset, ms/token:
///
/// | ctx | unfused (3 packets) | 3-stream fused | 4-stream (+`o_gate`) |
/// |---|---|---|---|
/// | 8000 | 39.542 | **38.858** | **38.563** |
/// | 32000 | **40.593** | 41.468 | 41.186 |
///
/// So it wins ~0.7-1.0 ms at 8k and loses ~0.9-1.2 ms at 32k, and folding `o_gate` in as a fourth
/// stream moves both arms the same direction — the effect is not about `o_gate`.
///
/// WHY, and it is the counter-argument to "fewer packets is better": `q_a`, `kv_a` and `k_rope_d` are
/// all gated on the SAME pre-norm and are mutually independent, so the global queue runs them
/// CONCURRENTLY across 256 CUs and their cost is the MAX of the three. Fusing makes the work serial
/// inside one packet, so the cost becomes the SUM. That is a win only while the projections are
/// latency-starved enough that a wider op more than repays the lost concurrency — true at 8k, false
/// at 32k, where flash is the long pole and anything that delays its start is paid in full.
///
/// This is `GLM_GROUP=1`'s failure mode (38% fewer ops, +2.88 ms) reproduced on a smaller merge, and
/// it is why `PLOW_KDA_FUSE_QKVG`'s -22..27% does NOT transfer here by analogy: KDA's four
/// projections are each 1536 wide against these three summing to 2112, and KDA has no flash behind
/// them waiting.
///
/// `PLOW_K3_FUSE_A=1` opts in — worth it for a short-context deployment, and the knob keeps the
/// measurement reproducible from the same binary.
///
/// DECODE ONLY, for the reason `kda::fuse_qkvg` states: op 108 is compiled into the decode
/// interpreter only, and this interpreter's dispatch `default:` WRITES NOTHING, so a prefill bucket
/// emitting it would silently produce no q, kv, k_rope or gate at all. The LDS bound is kept because
/// it is a real constraint, but `t == 1` is what makes the op safe.
fn fuse_mla_a(t: u32, hidden: u32) -> bool {
    if !crate::emit_config::active().k3_fuse_a {
        return false;
    }
    if t != 1 {
        return false;
    }
    crate::gemv_staged_rows(t) as u64 * hidden as u64 <= crate::gm_lds_halves()
}

/// May an [`DevOp::AttnRes`] absorb the RMSNorm that always follows it?
///
/// # Every one of them has exactly one consumer, and it is a norm
///
/// A K3 layer runs `AttnRes -> RmsNorm` twice: the attention-side mix feeds the mixer's pre-norm
/// (`input_layernorm`), the MLP-side mix feeds `post_attention_layernorm`. `plowrt disasm
/// --program 1` over the 93-layer TP8 decode blob confirms it exhaustively — **186 of 186** AttnRes
/// outputs are read by the RMSNorm on the very next instruction and by nothing else in the program.
/// The model-level mix added since makes it **187 of 187**: its sole consumer is `model.norm`, the
/// same shape one level up.
///
/// # Why it is worth a fusion and the packet-count arithmetic understates it
///
/// Both packets are ONE workgroup at `T = 1`: the mix because its two reductions span the 7168-wide
/// row and the softmax couples the rows ([`emit_attn_res`]), the norm because `rows = 1`
/// ([`norm_cus`]). So this is two SERIAL narrow packets, back to back, on a decode chain with
/// nothing ready behind either. `d_norm_residual_norm`'s note in `runtime/amd/op_norm.h` priced
/// exactly this shape from the other direction: splitting one narrow op into three serial ones,
/// with the arithmetic proven bit-identical, cost **+1.28 ms/token over 120 sites — ~5.3 us per
/// added packet**, nearly twice what the op itself cost. K3 pays that 186 times.
///
/// # Bit-exact, and the kernel is written to keep it that way
///
/// The mix is stored to `out` as the bf16-ROUNDED value the unfused arm stored, and the norm's
/// sum-of-squares runs over the value re-READ from `out` — so the reduction sees precisely what the
/// separate RMSNORM packet would have re-read from HBM, in the same per-thread order, folded by the
/// same `block_sum`. The norm is IN PLACE, which is what lets it need no second output tensor: the
/// raw mix has one consumer and this is it.
///
/// `eps` is shared. Both packets took it from the same `K3BlockCfg`/mixer config field, and the
/// call sites assert that rather than assume it.
///
/// `PLOW_K3_FUSE_ARNORM=0` restores the two-packet form from the SAME binary, which is what makes
/// the A/B a control rather than a rebuild.
/// Always fuse attnres+norm (hardcoded — was `PLOW_K3_FUSE_ARNORM`, never disabled).
pub(crate) fn fuse_attnres_norm() -> bool {
    true
}

/// May the LatentMoE tail's residual add absorb the BLOCK-OUTPUT residual that follows it?
///
/// A K3 MoE layer ends in two [`DevOp::Residual`] packets that only each other read:
///
/// ```text
///   ffn = up_latent + shared_down      (emit_k3_latent_moe's tail)
///   x   = prefix    + ffn              (emit_k3_block_out)
/// ```
///
/// `plowrt disasm --program 1` finds **92 of these chained pairs** in the 93-layer decode blob —
/// one per MoE layer — and `{a}ffn` is read exactly once, by the second of the pair. Both packets
/// are `vec8_cus(7168)` = TWO workgroups, so this is two serial 2-CU packets on a chain with
/// nothing ready behind either: the same shape [`fuse_attnres_norm`] prices, and the same fix.
/// Folding them deletes a packet AND a full 7168-wide HBM round trip (`ffn` is written and
/// immediately re-read).
///
/// BIT-EXACT: `d_residual`'s three-input form rounds the inner sum to bf16 before the outer add,
/// which is precisely the value the deleted packet stored. Both scales were 1.0f and the call site
/// asserts it.
///
/// Latent-MoE layers only. Layer 0's dense MLP is one site out of 93 and would need the same
/// plumbing through a second emitter for 1/92nd of the win.
///
/// `PLOW_K3_FUSE_BRESID=0` restores the two-packet form from the SAME binary.
/// The prefix-accumulate `Residual` (`prefix = prefix_in + attn`) does NOT fold into the
/// [`DevOp::AttnRes`] that follows it. TRIED, and it produced a fluent wrong model — prefill token
/// 261 instead of 17374, all 8 ranks agreeing on the garbage.
///
/// The trap is that `prefix` has TWO consumers, not one: the mix on the next line, and the
/// BLOCK-OUTPUT residual further down (`fuse_bo.then_some(prefix)` — `x = prefix + ffn`). Folding
/// the accumulate into the mix and handing the mix `prefix_in + attn` leaves the block output
/// reading `prefix_in`, silently dropping the whole attention contribution of every non-snapshot
/// layer.
///
/// Two ways I convinced myself it had one consumer, both wrong, both worth naming:
///   * `plowrt disasm | grep act.lN.prefix` piped through `cut -c1-120`, which TRUNCATES the
///     operand list — the third reference sits past column 120. Never truncate a disasm you are
///     about to draw a dependency conclusion from.
///   * the counter fan-out census said all 178 `Residual` counters have fan-out 1. That is a
///     statement about the COUNTER graph after `fuse_block_resid` has already folded the
///     block-output add into the MoE tail — the tensor is still read, just not through an edge
///     that census counts.
///
/// It was also worth less than it looked: 85 packets, not 178, because the other `Residual`s are
/// the MoE tail and block-output adds. ~0.3 ms against a 36 ms token.
///
/// If someone retries it: the mix must write `prefix` as well as consuming it, or the block-output
/// residual must take `prefix_in` and `attn` as separate addends.
/// Always fuse block-residual (hardcoded — was `PLOW_K3_FUSE_BRESID`, never disabled).
pub(crate) fn fuse_block_resid() -> bool {
    true
}

/// May the shared expert's `gate`, `up` and `SituGlu` collapse into one [`DevOp::GemvGlu`]?
///
/// DECODE ONLY: the fused GEMV arm is a `d_gemv_*` body, and a prefill bucket runs tiled GEMM
/// instead — `gfx950_prefill_tile` would pick a `Gemm*` opcode for which this three-way merge has no
/// arm, and AMD's dispatch `default:` writes NOTHING. The LDS bound is the same one
/// `kda::fuse_qkvg` uses and is kept for the same reason.
///
/// `PLOW_K3_FUSE_SHGLU=0` restores the three-packet form from the same binary.
fn fuse_shared_glu(t: u32, hidden: u32) -> bool {
    // Hardcoded ON (was `PLOW_K3_FUSE_SHGLU`, never disabled in production).
    if t != 1 {
        return false;
    }
    crate::gemv_staged_rows(t) as u64 * hidden as u64 <= crate::gm_lds_halves()
}

/// May a `b=1` [`DevOp::RmsNorm`] be folded into the `b=256` [`DevOp::Gemv`] that reads it?
///
/// This is recommendation 2 of `perf-data/archive/k3/k3-decode-counter-graph.md`, taken for the ONE of its
/// three named ops that survives the check `op_gemm.h` demands before any such fusion.
///
/// # The lever, and why a narrow gate in front of a wide consumer is worth more than it looks
///
/// A `b=1` packet bumps its counter once, but a `b=256` consumer polls that counter from all 256
/// of its workgroups. The census over the 93-layer TP8 decode blob: 303 `AttnRes`/`RmsNorm`
/// packets contribute 303 bumps and **151,598 polls** — 19% of all counter traffic from 12% of
/// the packets. Worse, each is its own level of a dependency chain that is **1739 deep at mean
/// width 1.41**, i.e. there is essentially nothing to overlap a narrow packet with. So deleting
/// one deletes a packet, an edge and a CHAIN LEVEL together, at ~5.7 us of measured per-packet
/// protocol cost.
///
/// # `op_gemm.h` already priced this fusion as a LOSS, and the difference is fan-out
///
/// Its `norm` mode 2 — compute the row RMS inside the GEMV from the `x` it already stages in LDS
/// — was implemented, measured at **22.4 -> 24.4 ms/token on Gemma**, and deleted. The cause was
/// not the arithmetic (which is nearly free) but N: Gemma's attention norm has FIVE consumers, so
/// folding turned one shared reduction into five redundant ones. The rule it left behind is the
/// gate this function implements: *fusion that duplicates a reduction across N consumers costs
/// (N-1) extra reductions — check N first.*
///
/// K3 answers that check differently, and the answer was read off the emitted blob rather than
/// assumed (`plowrt disasm --program 1 --counters`, full operand lists, NOT truncated — see
/// [`fuse_block_resid`] for what truncating one costs):
///
/// ```text
///   RmsNorm  116 packets   fan=1: 92   fan=2: 24      <- fusable, and 92 of them are FREE
///   AttnRes  187 packets   fan=3: 161  fan=4: 24      <- REFUSED, see below
///   MoeRouterTopk 92       fan=2: 92                  <- REFUSED, see below
/// ```
///
/// The 92 `fan=1` sites are `routed_expert_norm` (`feat=3584`) feeding the latent up-projection:
/// N=1, so the fusion costs ZERO extra reductions and is pure profit. The 24 `fan=2` sites are
/// `q_a_layernorm` (`feat=1536`) feeding `q_absorb` and `q_rope`: one extra reduction each, over
/// 1536 elements already resident in LDS, against a whole chain level.
///
/// # Why the other two ops in that recommendation are NOT taken here
///
/// * **`AttnRes` (187 levels, the biggest single prize) is a LOSS by arithmetic.** Its mix spans
///   `nb+1` rows of 7168 (`nb` runs 0->8; mean 4.36, so mean 5.36 rows), and a fused consumer
///   workgroup would have to re-read every one of them — a GEMV stages ONE row. Summed over the
///   real fan-out that is 619 consumer workgroups per site x 5.36 x 7168 x 2 B = **47.6 MB per
///   site, 8.9 GB per token**, against a token that already streams 57 GiB in 36 ms. It buys 187
///   levels = 1.07 ms and spends several times that. It would also stage 126 KB of LDS per
///   workgroup, which alone costs the occupancy the GEMV depends on.
/// * **`MoeRouterTopk` needs a measurement this change does not make.** The census in that doc
///   lists one consumer (`MoeGroupGluFp8Blk`); the blob has **two** — `MoeGroupDownFp8Blk` reads
///   `route_tab` as well — so it is fan=2, and both consumers are `b=256`. Folding replicates a
///   top-16-of-896 selection into 512 workgroups per site and puts it on the critical path of
///   both. The re-read is cheap (896 f32 logits + bias), so this one is not obviously a loss, but
///   it is not obviously a win either and it is a different kernel; it wants its own A/B.
///
/// # The trap that cost a day: `plow_smem` is a UNION
///
/// `interp.hip`'s shared memory overlaps `part` (block_sum's scratch) with `gm` (the GEMV arena) —
/// ONE address. The first version of this fold took `part` from the caller, so its `block_sum`
/// wrote through the first 16 halves of the row it was reducing. It did not fail loudly: the
/// write-back repairs those halves from registers, so the corruption is partial and
/// data-dependent, the token stream stayed IDENTICAL over 32 steps, and only the logits drifted
/// ~1 ULP per layer — the exact scale `op_collective.h` prices at ~0.03 logits over 92 MoE layers.
/// The standalone kernel test PASSED throughout, because it declared two separate `__shared__`
/// arrays and therefore did not alias. See `perf-data/archive/k3/k3-narrow-gate-fusion.md` §4.
///
/// # STATUS: BIT-EXACT END-TO-END. DEFAULT ON.
///
/// The design below is bit-exact and the isolated hardware gate agrees — `norm = 2` reproduces
/// the RMSNORM+GEMV pair to ZERO differing halves at every shape the emitter folds, including the
/// tp8-sharded `N=896 K=3584 b=224` geometry (`runtime/tests/gemv_fusednorm_gfx950_test.hip`).
///
/// The current gfx942 TP8 gate compares rank-0's complete BF16 logit row after prefill and all 32
/// decode steps at `ctx=5`. Every vector is byte-identical (`maxabs=0`), all eight ranks emit the
/// same token stream, and two matched `plowrt serve` pairs improve TPOT by 1.33--1.45 ms. The
/// earlier divergent implementation reduced through storage aliased by `plow_smem`; the derived
/// arena-top scratch below makes that pointer unreachable. `PLOW_K3_FUSE_NGEMV=0` retains the
/// unfused control packet.
///
/// # Why it is bit-exact by construction, when the arena is not contended
///
/// The deleted packet stored `f2bf(x*inv*gamma)` to HBM and the GEMV re-read those bf16 values.
/// So the fused arm does NOT multiply through the k-loop the way mode 1 does — it normalizes the
/// LDS-staged copy IN PLACE, rounding to bf16, walking the row with `d_rmsnorm`'s exact
/// per-thread element map and `block_sum`, and then runs the ordinary un-normed hot loop. The
/// bytes the k-loop reads are the bytes the unfused pair wrote. See `gemv_norm_lds`.
///
/// # Preconditions, all of which the arm re-checks
///
/// DECODE ONLY (`t == 1`), for the reason every other gate in this file states: a prefill bucket
/// picks a `Gemm*` opcode, which has no mode-2 arm, and this interpreter's dispatch `default:`
/// WRITES NOTHING — a silent no-op, not a crash. The row must also fit the staged arm
/// (`GM_LDS_HALVES`) and `d_rmsnorm`'s register path (`feat <= RN_REG*PLOW_THREADS`, `feat % 8`).
///
/// `PLOW_K3_FUSE_NGEMV=0` restores the unfused control; `=lat` and `=q` enable ONE site each,
/// which is how the original divergence was bisected.
/// Which fold site a [`fuse_norm_gemv`] call is asking about. The knob can name ONE of them, which
/// is what makes an end-to-end divergence bisectable: `lat` and `q` are independent rewrites and a
/// whole-model A/B that moves cannot say which one moved it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NormSite {
    /// `routed_expert_norm` -> the latent up projection. 92 sites, fan-out 1.
    Lat,
    /// `q_a_layernorm` -> `q_absorb` and `q_rope`. 24 sites, fan-out 2.
    Q,
}

pub(crate) fn fuse_norm_gemv(t: u32, feat: u32, site: NormSite) -> bool {
    // Default on after the full-model BF16-logit gate above. `=0` is the measured control;
    // `=lat`/`=q` keep the two sites independently bisectable.
    match crate::emit_config::active().k3_fuse_ngemv.as_deref() {
        None | Some("1") => {}
        Some("lat") if site == NormSite::Lat => {}
        Some("q") if site == NormSite::Q => {}
        _ => return false,
    }
    if t != 1 {
        return false;
    }
    // Mirrors `d_gemv_t`'s `fold` test and `d_rmsnorm`'s `fits` test. Both are re-checked on the
    // device; this is the half that must not emit the opcode in the first place.
    // + GV_NORM_SCRATCH: the fold takes its cross-wave reduction scratch from the TOP of the
    // arena, because `plow_smem` is a union and the interpreter's `part` aliases the staged row.
    crate::gemv_staged_rows(t) as u64 * feat as u64 + crate::GV_NORM_SCRATCH
        <= crate::gm_lds_halves()
        && feat <= crate::RN_REG * crate::PLOW_THREADS
        && feat % 8 == 0
}

/// Is `routed_expert_up_proj` COLUMN-parallel? `PLOW_K3_SHARD_UP=0` restores the replicated
/// emit from the SAME binary, which is what makes the A/B a control rather than a rebuild.
///
/// Both halves must answer this identically — [`declare_k3_moe_weights`] sizes the tensor and
/// [`emit_k3_latent_moe`] sizes the GEMV and the gather — so it is one function and not two
/// reads of the environment. A disagreement is not a crash: a full declaration with a sliced
/// GEMV computes an eighth of the FFN and adds it to seven zeros.
/// Always shard the up-projection under TP (hardcoded — was `PLOW_K3_SHARD_UP`, never disabled).
pub(crate) fn shard_up_proj(tp: u32) -> bool {
    tp > 1
}

/// BISECTION INSTRUMENTS for the column-parallel up projection. Neither is a serving mode.
///
/// The shard has two independent halves — a SLICED WEIGHT and a GATHERED partial — and when
/// the output is wrong, "which half" is the only question worth asking. Each knob keeps one
/// half and removes the other, so the answer is one run each:
///
/// * `PLOW_K3_UP_NOGATHER=1` — slice the weight, write the peer slot, and DO NOT gather.
///   `ffn` then loses the routed-expert path entirely. If the output equals the sharded
///   one, the gather is contributing nothing at runtime.
/// * `PLOW_K3_UP_GATHER_ONLY=1` — keep the weight REPLICATED (so every rank computes the
///   WHOLE `yh`) but still route it through the peer slot and the gather. The gather indexes
///   each owner's slot COMPACTLY, so what it assembles is `yh[e % gcols]` tiled — wrong by
///   construction, and that is the point: it is nonzero and structurally unrelated to the
///   sharded answer, so it separates "the gather never ran" from "the gather ran".
///
/// HOW THE TWO READ, and this is the reason to write them down rather than re-derive them:
/// on the real checkpoint (24 steps, `1008,10484,318,15383,387`) the replicated emit answers
/// ". The population is approximately 67 million people. …", `nogather` answers
/// "6. 2. 2. 2. …" and `gather_only` answers "\n The \n The …" — both controls collapse
/// completely, while the SHARDED emit stayed grammatical. That gap is what said the peer
/// path was intact and the error was arithmetic, which is how the missing bf16 round in
/// `d_xreduce`'s gather arm was found.
///
/// Neither knob is a serving mode. They exist because the alternative was guessing.
fn up_nogather() -> bool {
    crate::emit_config::active().k3_up_nogather
}
fn up_gather_only() -> bool {
    crate::emit_config::active().k3_up_gather_only
}

/// `PLOW_K3_SHARD_HEAD=1` — vocab-column-parallel `lm_head`, and this rank's slice of the vocab.
///
/// The replicated default streams the FULL `vocab * hidden` bf16 table on EVERY rank EVERY step:
/// at K3's 163840 x 7168 that is **2.35 GB per rank per token**, ~0.37 ms at the 6.4 TB/s measured
/// streaming ceiling, of which 7/8 is redundant at TP8. GLM measured its own replicated head — a
/// SMALLER table, 1.90 GB at TP4 — at **106% of the HBM ceiling**, i.e. entirely a sharding gap and
/// not a kernel one, and sharding it was worth -0.26 ms/token bit-identically over 256 tokens
/// including 11 whose ids fall outside rank 0's shard (`mla.rs:613-631`).
///
/// The two halves MUST move together, which is why one knob gates both: a column-parallel head
/// leaves each rank holding only its own slice of the logits, so `ArgmaxFin` would have every rank
/// confidently sample its LOCAL winner and the ranks would disagree on the first token.
/// `XArgmaxFin` is what folds the per-rank maxima, rebasing the winning index by `rank * vocab_l`
/// and taking the cross-rank max; it subsumes `ArgmaxFin` rather than following it.
///
/// Refuses a vocab that does not divide `tp`: one blob serves every rank, so a ragged last shard
/// has nowhere to record its own width. 163840 / 8 = 20480, so K3 is clean.
///
/// OFF by default until it has been measured on a real K3 load, the same standing GLM's arm had.
fn k3_shard_head(c: &K3ModelCfg) -> bool {
    let on = c.tp > 1 && crate::emit_config::active().k3_shard_head;
    assert!(
        !on || c.vocab % c.tp == 0,
        "PLOW_K3_SHARD_HEAD needs vocab ({}) divisible by tp ({})",
        c.vocab,
        c.tp
    );
    on
}

/// This rank's lm_head vocab shard (`c.vocab` when replicated).
fn k3_vocab_l(c: &K3ModelCfg) -> u32 {
    if k3_shard_head(c) {
        c.vocab / c.tp
    } else {
        c.vocab
    }
}

/// The workgroups a `FlashMlaDecode`-family packet can actually use.
///
/// `exec_flash_mla` gives one work item per `(batch, token, head-group, split)`, so the count is
/// `n_batch * ntok * (nh_l / gf) * nsplit`. At K3's TP8 decode that is `1 * 1 * (12/4) * 4 = 12`
/// against the 256 workgroups the emitter hands it — 244 empty slices on all 24 MLA layers.
pub(crate) fn k3_flash_cus(cus: &[u32], ntok: u32, nh_l: u32, gf: u32, nsplit: u32) -> Vec<u32> {
    if !k3_wgfit() {
        return cus.to_vec();
    }
    let groups = (nh_l / gf.max(1)).max(1);
    let items = ntok as u64 * groups as u64 * nsplit.max(1) as u64;
    let need = (items.max(1) as usize).min(cus.len());
    cus[..need].to_vec()
}

impl K3BlockCfg {
    /// How many block-residual rows layer `l` (0-based) sees on ENTRY.
    ///
    /// A snapshot is pushed at the TOP of every layer with `l % B == 0`, before the attention, so
    /// layer `l` enters with one row per earlier snapshot: `ceil(l / B)`. Layer 0 enters with
    /// **zero**, which is why its attn-side AttnRes is skipped entirely.
    pub fn blocks_at(&self, l: u32) -> u32 {
        l.div_ceil(self.attn_res_block_size)
    }
    /// Does layer `l` push a snapshot and therefore RESET the prefix sum?
    ///
    /// Driven by the modulus because the reference is (`layer_idx % attn_res_block_size == 0`) —
    /// unlike the KDA/MLA layer split, which is a config LIST and where a modulus is wrong at
    /// 0-based layer 92.
    pub fn snapshots(&self, l: u32) -> bool {
        l % self.attn_res_block_size == 0
    }
    /// The largest `nb` any layer of a `n_layers`-deep model sees. Must be `<= K3_ATTNRES_MAXB`.
    pub fn max_blocks(&self, n_layers: u32) -> u32 {
        self.blocks_at(n_layers - 1) + u32::from(self.snapshots(n_layers - 1))
    }
}

/// Emit AttnRes: `out = softmax_r(rms(v_r) . score_w) @ v`, over `blkres` rows then `prefix`.
///
/// `score_w` is `norm.weight * proj.weight`, **folded at prep time** into one `[hidden]` f32 —
/// both factors are parameters, so the product is constant and neither is needed separately on
/// device.
///
/// `nb == 0` is an exact copy (softmax over one element). The reference SKIPS the call in that
/// case; the caller should too, and [`emit_k3_block_out`] does. The arm handles it so that
/// emitting it anyway cannot produce a zero-filled buffer.
///
/// **One workgroup per token.** Both reductions span the full 7168-wide row and the softmax couples
/// the rows, so `blocks = min(T, n_cu)` — at `T = 1` that is 1 of 256. It stays ONE packet:
/// `perf-data/archive/k3/kimi-k3-kernel-gap.md` §10 item 7 measured "three packets x 186 is 3.3 ms/token of
/// pure protocol", which rules out finishing the reduction in a second packet.
///
/// The launch geometry is still one workgroup, but the BODY is no longer what that note assumed.
/// §10 item 7 called this op bandwidth-trivial; it was not, it was instruction-bound — scalar
/// 2-byte loads, and one full L2 latency per row because the row-outer loop closed each row with a
/// `block_sum` whose `s_waitcnt vmcnt(0)` blocked the next row's loads. Vectorised loads plus a
/// row-inner sweep took it **3.33x** (4.09 -> 1.23 ms/token, weighted over `nb = ceil(layer/12)`
/// across 93 layers x 2 sites), with the real-weight gate reporting identical digits.
///
/// What is left is the MULTI-CU split, and it is left deliberately. The bench measured the
/// cross-workgroup barrier it needs at 0.5-0.9 us for 2-16 blocks, so an 8-16 block split would pay
/// ~1.7 us to save ~8 us at nb=8 — a further ~2x. It is not a kernel change: a workgroup spinning
/// in an intra-packet barrier has stopped draining its own stream, so it can no longer signal a
/// counter a PEER workgroup is gated on, and nothing inside `op_k3.h` can establish that the
/// emitter will not order the per-CU streams into that cycle. It needs a scheduling invariant here
/// and a deadlock argument, not a faster loop.
#[allow(clippy::too_many_arguments)]
pub fn emit_attn_res(
    b: &mut Builder,
    c: &K3BlockCfg,
    out: u32,
    prefix: u32,
    blkres: u32,
    score_w: u32,
    t: u32,
    nb: u32,
    // The ring's ALLOCATED row count. `blkres` is `[T][nb_cap][hidden]` and the kernel strides by
    // this, NOT by `nb` — `nb` grows with depth, `nb_cap` is a property of the program. The two
    // are indistinguishable at `T == 1` (the token index is 0 so the stride never multiplies),
    // which is why this operand did not exist until prefill needed it.
    nb_cap: u32,
    n_cu: u32,
    // `Some(src)` pushes `src[t]` onto ring row `nb` of every token's slice before the mix — the
    // snapshot a layer with `l % attn_res_block_size == 0` takes. Legal at any `T`: the workgroups
    // partition the tokens, the ring is per token, and the mix reads rows `[0, nb)` while this
    // writes row `nb`, one past. See [`DevOp::AttnRes`] for the full argument.
    push: Option<u32>,
    // `Some(w)` FUSES the RMSNorm that always follows, in place over `out`. See
    // [`fuse_attnres_norm`] for why every AttnRes in a K3 program has exactly one such consumer,
    // and `d_attn_res`'s note for why it is bit-exact. `None` leaves the raw mix.
    post_norm: Option<u32>,
    deps: &[u32],
) -> u32 {
    assert!(
        nb <= K3_ATTNRES_MAXB,
        "AttnRes: nb = {nb} exceeds PLOW_ATTNRES_MAXB = {K3_ATTNRES_MAXB}; the arm POISONS rather \
         than overrunning its LDS carve, so this would fill `out` with qNaN — loud, but still a \
         program that cannot run"
    );
    assert!(
        nb_cap >= nb && (push.is_none() || nb_cap > nb),
        "AttnRes: ring capacity {nb_cap} cannot hold {nb} live rows{}; the ring strides by the \
         CAPACITY, so a short one aliases token slices onto each other — no fault, no NaN, a \
         fluent wrong model",
        if push.is_some() {
            " plus the pushed one"
        } else {
            ""
        }
    );
    let blocks: Vec<u32> = (0..t.min(n_cu).max(1)).collect();
    b.emit(DevOp::AttnRes, blocks, deps, |d| {
        d.t[0] = out;
        d.t[1] = prefix;
        d.t[2] = blkres;
        d.t[3] = score_w;
        d.t[4] = push.unwrap_or(packet::dev::TENSOR_NONE);
        d.t[5] = post_norm.unwrap_or(packet::dev::TENSOR_NONE);
        d.i[0] = t;
        d.i[1] = c.hidden;
        d.i[2] = nb;
        d.i[3] = nb; // push_row: the snapshot lands one past the live count
        d.i[4] = nb_cap;
        d.i[5] = packet::dev::TENSOR_NONE_I;
        d.f[0] = c.eps;
    })
}

/// One bf16 linear projection `C[t, n] = A[t, k] · W[n, k]^T`, at whichever row count the
/// program is being emitted for.
///
/// **This is the GEMV/GEMM seam, and it is the whole reason a K3 prefill bucket is not just
/// `emit_k3_model` with a bigger `t`.** Every projection in this family was written as
/// [`DevOp::Gemv`] because K3 was decode-only. That op is compiled with a COMPILE-TIME row bucket
/// (`PLOW_GEMV_MM = next_pow2(PLOW_DECODE_BATCH)`, clamped to 16), so at `t = 128` it would
/// process the first 16 rows and leave the rest holding whatever the arena held — finite, plausible
/// and wrong. Its fp4 sibling [`DevOp::GemvMxfp4`] is worse: the prefill interpreter has no arm for
/// it at all, so the packet falls through the dispatch `default:` and writes nothing.
///
/// At `t == 1` this emits `Gemv` with byte-identical operands, so a decode program is unchanged.
/// Above it, [`crate::gfx950_prefill_tile`] picks the rung from the same inventory (and the same
/// tunedb records) every other prefill emitter in the tree selects from — the shapes here are
/// ordinary `[t, n] = [t, k] · [n, k]` and there is nothing K3-specific to rank.
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_linear(
    b: &mut Builder,
    out: u32,
    x: u32,
    wt: u32,
    t: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    seq_rows: bool,
    deps: &[u32],
) -> u32 {
    emit_k3_linear_norm(b, out, x, wt, t, n, k, n_cu, seq_rows, None, deps)
}

/// [`emit_k3_linear`] with the option of ABSORBING the [`DevOp::RmsNorm`] that produced `x`.
///
/// `fold = Some((gamma, eps))` sets `norm = 2`, which makes the GEMV normalize the activation it
/// already stages in LDS and lets the caller skip emitting the RMSNORM packet entirely — one
/// packet, one edge and one chain level, all three at once. `x` then names the norm's INPUT, not
/// its output. See [`fuse_norm_gemv`] for when this is legal and why it is bit-exact.
///
/// `eps` rides in `f[0]`, which the GEMV already carries (`exec_gemv` passes `in->fj[0].f`) and
/// which no un-normed emit sets — so this costs no operand slot.
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_linear_norm(
    b: &mut Builder,
    out: u32,
    x: u32,
    wt: u32,
    t: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    seq_rows: bool,
    fold: Option<(u32, f32)>,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    // A BATCHED DECODE TAKES THE GEMV ARM, and `t == 1` alone does not say so.
    //
    // `PLOW_GEMV_MM = next_pow2(PLOW_DECODE_BATCH)` clamped to 16 is exactly the compile-time row
    // bucket that makes `Gemv` carry B rows — it exists FOR this case, which is also why the
    // hsaco's PLOW_DECODE_BATCH must match the emit's. Falling through to a tiled prefill rung at
    // `t = B` instead is wrong twice over: those opcodes are PREFILL arms, and a decode program
    // runs on the DECODE object, where AMD's dispatch `default:` writes NOTHING rather than
    // trapping. The packet then leaves its output holding whatever the arena held — the run
    // completes, every row is finite and plausible, and every row is wrong.
    let gemv_arm = t == 1 || seq_rows;
    let (op, gemm_blocks, gemm_variant) = if gemv_arm {
        (DevOp::Gemv, n_cu, 0)
    } else {
        crate::pick_gemm_emit_plan(t, n, k, n_cu, kernelcaps::QuantScheme::None)
    };
    // A tiled prefill rung has no mode-2 arm, and AMD's dispatch `default:` writes NOTHING. The
    // caller's gate is `t == 1`; this is the assert that a future caller cannot get past it.
    assert!(
        fold.is_none() || op == DevOp::Gemv,
        "fused-norm GEMV is decode-only: op {op:?} has no `norm == 2` arm and would silently no-op"
    );
    // GV_BLOCKED owns output columns in contiguous runs, so a packet with fewer columns than
    // workgroups leaves the tail owning nothing: K3's `b_proj` is N=12 and `f_a_proj` N=128,
    // handed 256 workgroups each on all 69 KDA layers. The narrowing is a fixed point of the
    // kernel's own `per = ceil(n/nblk)` arithmetic, so every surviving workgroup's column range is
    // unchanged and the packet is bit-identical. Tiled prefill ops own TILES, not column runs, so
    // the rule applies to the GEMV arm only.
    let blocks = if gemv_arm {
        crate::mla::blocked_gemv_cus_tuned(&all, n, k)
    } else {
        (0..gemm_blocks).collect()
    };
    b.emit(op, blocks, deps, |d| {
        d.t[0] = out;
        d.t[1] = x;
        d.t[2] = wt;
        d.i[0] = t;
        d.i[1] = n;
        d.i[2] = k;
        d.i[7] = gemm_variant;
        if let Some((gamma, eps)) = fold {
            // t[3] is the PRECOMPUTED-rms operand of mode 1 and stays absent: mode 2 computes
            // the scalar itself, which is the whole point.
            d.t[4] = gamma;
            d.i[3] = 2;
            d.f[0] = eps;
        }
    })
}

/// Emit the `situ` GLU over `n` elements: `A(gate) * B(up)`.
///
/// The two betas ride in `f0`/`f1`, which are free on every GLU-family op — so this consumes no
/// `i` slot and no existing emitter changes.
#[allow(clippy::too_many_arguments)]
pub fn emit_situ_glu(
    b: &mut Builder,
    c: &K3BlockCfg,
    out: u32,
    gate: u32,
    up: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::SituGlu, vec8_cus(&all, n), deps, |d| {
        d.t[0] = out;
        d.t[1] = gate;
        d.t[2] = up;
        d.i[0] = n;
        d.f[0] = c.situ_beta;
        d.f[1] = c.situ_linear_beta;
    })
}

/// Emit the MLA **output gate**: `out = attn * sigmoid(g)`, applied BEFORE `o_proj`.
///
/// `mla_use_output_gate: true` on all 24 of K3's MLA layers. `g` is the raw `g_proj` output of the
/// MLA sub-layer INPUT (the post-`input_layernorm` hidden), NOT of the attention output — the
/// reference reads `hidden_states`, which at that point is the norm output
/// (`modeling_kimi_linear.py:471`). Feeding it the attention output has the right shape and the
/// wrong model.
///
/// `n` is `n_head_local * v_head_dim` and both operands are **head-major**, which is exactly what
/// [`DevOp::MlaMergeFold`] writes and exactly what the reference's `.reshape(batch, seq, -1)`
/// produces, so no permute is implied.
///
/// It is `sigmoid(g)`, not `silu(g)` — which is why it is [`DevOp::MlaOutGate`] rather than a third
/// `act` code on [`DevOp::Glu`], whose `act=1` IS silu. The two differ by a factor of the logit:
/// finite, correctly-shaped, and the wrong model on 24 layers of every token.
pub fn emit_mla_out_gate(
    b: &mut Builder,
    out: u32,
    attn: u32,
    g: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::MlaOutGate, vec8_cus(&all, n), deps, |d| {
        d.t[0] = out;
        d.t[1] = attn;
        d.t[2] = g;
        d.i[0] = n;
    })
}

/// The `(cos, sin)` recipe pair for K3's **NoPE** MLA: an all-identity table, `cos = 1, sin = 0`.
///
/// `mla_use_nope: true` and `text_config` carries no `rope_theta`
/// (`configuration_kimi_k3.py`; `KimiMLAAttention.__init__` ends `self.rotary_emb = None` and
/// `forward` concatenates `q_rot`/`k_rot` back UNROTATED). The 64 "rope" dims are extra CONTENT
/// channels of the 192-wide key.
///
/// # THIS IS NOT A REMOVAL, and that is the whole point
///
/// `perf-data/archive/k3/kimi-k3-kernel-gap.md` §8c calls the fix "skip both `HeadNormRope` emits — a removal,
/// effort XS". It is not. The k-side [`DevOp::HeadNormRope`] is the **only writer of the
/// `kv.{l}.krot` cache row**, AND it is the instruction that
/// `plowrt::exec::amd::kv_write_row_field` and `runtime/tests/glm52_decode.c:419` both SCAN FOR in
/// order to patch that row's position each step. Delete it and (a) the rope half of every cached
/// key is never written while `FlashMlaDecode` keeps reading it at `i[5]`, and (b) the layer
/// silently drops out of the KV-row-writer list — the scan just finds fewer, with **no count
/// check**. In `glm52_decode.c` the two arrays are index-paired and `nlk` is incremented only on
/// the krot match, so losing the krot writer silently loses the ckv patch too.
///
/// So: **keep the WRITE, remove the ROTATION.** With `cos = 1, sin = 0` the rotation is
/// `v*1 - partner*0 = v` — both constants exact in f32, `gamma` already `TENSOR_NONE` and
/// `skip_norm` already 1 on this emit, so `HeadNormRope` becomes a bit-exact bf16 copy that still
/// writes the cache row and still answers the scan.
///
/// # No new generator kind was needed
///
/// [`packet::rope::rope_tables`] already emits `(1.0, 0.0)` for every angle past
/// `rope_angles = frac * hd / 2`, because GLM's partial rotary needs exactly that for its NoPE
/// tail. `frac = 0.0` puts EVERY angle past it. `theta` is then never read, so it is pinned to
/// `1.0` rather than left to a caller's `unwrap_or` default — a defaulted theta reaching a table
/// that ignores it is the shape of bug this tree keeps finding, and the assertion in
/// [`k3_nope_table_is_exactly_identity`](self) is what keeps it that way.
pub fn k3_nope_rope_pair(ctx: u32, qk_rope: u32) -> [GenTensor; 2] {
    GenTensor::rope_pair(ctx, qk_rope, 1.0, 0.0, RopeScale::None)
}

/// The block's final combine: `out = prefix + ffn`.
///
/// Its own function because the operand that is NOT here is the point. A plain block would add the
/// LAYER INPUT; K3 adds the PREFIX SUM, and at a snapshot layer those differ by the whole embedding
/// state.
pub fn emit_k3_block_out(
    b: &mut Builder,
    out: u32,
    prefix: u32,
    ffn: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::Residual, vec8_cus(&all, n), deps, |d| {
        d.t[0] = out;
        d.t[1] = prefix;
        d.t[2] = ffn;
        d.i[0] = n;
        d.f[0] = 1.0;
    })
}

/// Geometry of one K3 **LatentMoE** block.
///
/// The name is the whole point: the routed experts run at `latent`
/// (`routed_expert_hidden_size` = 3584), not at `hidden` (7168). The kernels
/// take the width as a runtime operand and need no change — it is the GRAPH
/// that has to know there are two widths. Sizing the experts at `hidden` is a
/// silent 2x error, which is why `latent` is a separate field rather than
/// derived.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct K3MoeCfg {
    pub hidden: u32,
    /// `routed_expert_hidden_size` — the width every routed expert runs at.
    pub latent: u32,
    /// `moe_intermediate_size` — one routed expert's FFN width.
    pub moe_inter: u32,
    /// `num_shared_experts * moe_intermediate_size`, the shared FFN's width.
    pub shared_inter: u32,
    pub n_exp: u32,
    pub top_k: u32,
    /// Router flag word (sigmoid / renormalize), as `MoeRouterTopk` `i[3]`.
    pub route_flags: u32,
    pub route_scale: f32,
    /// `num_expert_group` / `topk_group`. Both 1 is the identity.
    pub n_group: u32,
    pub topk_group: u32,
    /// Expert weight encoding, matching `MoeEnc`: 0 bf16, 1 block-fp8, 2 mxfp4.
    /// K3 ships mxfp4. Travels in `i[6]` on the decode expert ops — NOT `i[3]`,
    /// which is `n_exp` there.
    pub enc: u32,
    /// Collapse decode's top-k slots into one grouped GLU packet and one grouped DOWN packet.
    /// The full real-weight TP8 4K+16 gate measured 103.161 -> 62.893 ms/token with bit-identical
    /// logits. `K3_MOE_GROUP=0` retains the per-slot baseline in the full-model emitter.
    pub group_decode: bool,
}

/// The LatentMoE's weight handles for one layer.
pub struct K3MoeWeights {
    pub router: u32,
    /// Router bias (`e_score_correction_bias`), or `TENSOR_NONE`.
    pub router_bias: u32,
    /// `routed_expert_down_proj` — `[latent, hidden]`, run ONCE for the block.
    pub down_latent: u32,
    /// `routed_expert_norm` — `[latent]`.
    pub latent_norm: u32,
    /// `routed_expert_up_proj` — `[hidden, latent]`.
    pub up_latent: u32,
    /// Host-filled `[n_exp*3]` u64 pointer tables.
    pub expert_w_table: u32,
    pub expert_s_table: u32,
    pub shared_gate: u32,
    pub shared_up: u32,
    pub shared_down: u32,
}

/// Emit one K3 LatentMoE block: `out = up(norm(combine(experts(down(x))))) + shared(x)`.
///
/// `x` is the POST-`post_attention_layernorm` hidden. Returns the counter of
/// the final add.
///
/// Three operand facts that a code read would not pin, all measured by the
/// rung-2 gate (`runtime/tests/k3_moe_block_gfx950_test.c`):
///
/// * The ROUTER scores the hidden state, at `hidden`; only the experts see the
///   latent. Routing on the latent is the same dtype and the wrong model.
/// * The SHARED expert reads the PRE-down hidden — `identity` in
///   `KimiSparseMoeBlock.forward`. Feeding it the latent fails loudly on width;
///   feeding it `h2` instead of `h3` would fail QUIETLY.
/// * `MoeCombine` runs at LATENT width with NO residual and NO shared operand.
///   At hidden width the residual add happens in the combine; here there is
///   nothing 3584 wide to add, so it happens after the UP projection.
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_latent_moe(
    b: &mut Builder,
    cb: &K3BlockCfg,
    c: &K3MoeCfg,
    w: &K3MoeWeights,
    act_prefix: &str,
    out: u32,
    x: u32,
    t: u32,
    seq_rows: bool,
    n_cu: u32,
    tp: &mut K3Tp,
    // `Some(p)` FOLDS the block-output residual into this block's own tail add, so `out` is the
    // layer output `p + (up_latent + shared_down)` and there is no `{a}ffn` buffer and no second
    // packet. See [`fuse_block_resid`].
    resid_in: Option<u32>,
    deps: &[u32],
) -> u32 {
    assert!(
        c.latent > 0 && c.latent != c.hidden,
        "K3 LatentMoE: latent width {} is absent or equal to hidden {} — the whole point of this \
         block is that the routed experts run NARROWER. Sizing them at hidden is a silent 2x.",
        c.latent,
        c.hidden
    );
    let all: Vec<u32> = (0..n_cu).collect();
    let a = act_prefix;
    let (tt, hid, lat) = (t as u64, c.hidden, c.latent);
    let bft = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 2);
    let f32t = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 4);

    let logit = bft(b, format!("{a}logit"), tt * c.n_exp as u64);
    let tab = f32t(b, format!("{a}route_tab"), tt * c.top_k as u64 * 2);
    let xe = bft(b, format!("{a}xe"), tt * lat as u64);
    // The expert INTERMEDIATE shards across ranks; the latent and hidden widths
    // do not — the two latent projections are shared by every expert, so there
    // is no expert axis to cut.
    let imoe = tp.local(c.moe_inter);
    // The per-SLOT gate/up buffer exists only on the decode arm — the grouped prefill chain writes
    // its GLU output into a row-sorted `fu_g` sized on the padded row bound instead. Declaring it
    // anyway would be `T*k*imoe` bf16 of arena that no op in the program touches, which is the
    // `Mamba2Scan` smell this tree keeps finding.
    let fu = if t == 1 {
        bft(b, format!("{a}fu"), tt * c.top_k as u64 * imoe as u64)
    } else {
        packet::dev::TENSOR_NONE
    };
    let pf_fuse = if t > 1 {
        crate::mla::moe_pf_fuse(c.top_k)
    } else {
        crate::mla::MoePfFuse::None
    };
    let part_pf = match pf_fuse {
        crate::mla::MoePfFuse::Atomic => tt * lat as u64 * 4,
        crate::mla::MoePfFuse::Det => tt * lat as u64 * 8,
        crate::mla::MoePfFuse::None => tt * c.top_k as u64 * lat as u64 * 4,
    };
    let part = b.tensor(
        &format!("{a}part"),
        part_pf.max(c.top_k as u64 * lat as u64 * 4),
    );
    let ylat = bft(b, format!("{a}ylat"), tt * lat as u64);
    // Declared even when [`fuse_norm_gemv`] folds the norm away and nothing writes it. Keeping the
    // tensor table IDENTICAL across both arms is what makes `PLOW_K3_FUSE_NGEMV` an A/B control:
    // the only difference between the two blobs is then the packet stream. It costs `lat` halves.
    let yn = bft(b, format!("{a}yn"), tt * lat as u64);
    // The up projection's output. Under TP it is COLUMN-parallel and lives in the peer
    // GATHER slot instead (`tp.ug`), and declaring an arena tensor nothing writes is the
    // `Mamba2Scan` smell — so it exists only on the tp=1 arm.
    let yh = if shard_up_proj(tp.tp) {
        packet::dev::TENSOR_NONE
    } else {
        bft(b, format!("{a}yh"), tt * hid as u64)
    };
    let shg = bft(b, format!("{a}sh_gate"), tt * c.shared_inter as u64);
    let shu = bft(b, format!("{a}sh_up"), tt * c.shared_inter as u64);
    let sha = bft(b, format!("{a}sh_act"), tt * c.shared_inter as u64);
    let shd = bft(b, format!("{a}sh_down"), tt * hid as u64);

    let gemv = |b: &mut Builder, out: u32, row: u32, wt: u32, n: u32, k: u32, dep: u32| {
        emit_k3_linear(b, out, row, wt, t, n, k, n_cu, seq_rows, &[dep])
    };

    // Router — scores the HIDDEN state. `logit` is [T, n_exp] on both phases; only the top-k TAIL
    // differs, and the prefill one is literally the decode kernel under a token loop.
    let c_rl = gemv(b, logit, x, w.router, c.n_exp, hid, deps[0]);

    // The DOWN projection is what makes this a LatentMoE. Independent of the
    // router, so it overlaps the rank pass.
    let c_xe = gemv(b, xe, x, w.down_latent, lat, hid, deps[0]);

    let si_l = tp.local(c.shared_inter);
    // Only independent decode sequences take this overlap. Prefill token rows keep their proven
    // recurrent schedule, and B1 keeps the fused packet below.
    let early_shared = if seq_rows && !fuse_shared_glu(t, c.hidden) {
        let c_sg = gemv(b, shg, x, w.shared_gate, si_l, hid, deps[0]);
        let c_su = gemv(b, shu, x, w.shared_up, si_l, hid, deps[0]);
        Some((c_sg, c_su))
    } else {
        None
    };

    // Under TP the combine lands in the peer slot and is all-reduced BEFORE the
    // norm: `routed_expert_norm` is nonlinear, so normalising a partial sum
    // would be finite, plausible and wrong.
    let cmb_dst = if tp.on() { tp.dg } else { ylat };

    // THE EXPERT CHAIN, AND IT IS TWO CHAINS. Decode's grouped ops loop the `top_k` routing-table
    // slots inside ONE gate/up packet and ONE down packet. They do not reuse weights across
    // tokens—the input is still one row—but they remove 30 counter-gated packets per K3 layer.
    // `group_decode=false` retains the old one-pair-per-slot graph as an A/B control. Prefill is a
    // different grouping: it SORTS the T*k (token, expert) pairs by expert and runs one grouped
    // GEMM per phase, so an expert's weights cross HBM once for every token that chose it.
    let mut c_cmb = if t == 1 {
        let c_rt = b.emit(DevOp::MoeRouterTopk, vec![0], &[c_rl], |d| {
            d.t[0] = tab;
            d.t[1] = logit;
            d.t[3] = w.router_bias;
            d.i[1] = c.n_exp;
            d.i[2] = c.top_k;
            d.i[3] = c.route_flags;
            d.i[6] = c.n_group;
            d.i[7] = c.topk_group;
            d.f[0] = c.route_scale;
        });
        if c.group_decode {
            let c_g = b.emit(DevOp::MoeGroupGluFp8Blk, all.clone(), &[c_rt, c_xe], |d| {
                d.t[0] = fu;
                d.t[1] = xe;
                d.t[2] = tab;
                d.t[3] = w.expert_w_table;
                d.t[4] = w.expert_s_table;
                d.i[0] = c.top_k;
                d.i[1] = imoe;
                d.i[2] = lat;
                d.i[3] = c.n_exp;
                d.i[5] = K3_MOE_ACT_SITU;
                d.i[6] = c.enc;
                d.f[0] = cb.situ_beta;
                d.f[1] = cb.situ_linear_beta;
            });
            let c_d = b.emit(DevOp::MoeGroupDownFp8Blk, all.clone(), &[c_g], |d| {
                d.t[0] = part;
                d.t[1] = fu;
                d.t[2] = tab;
                d.t[3] = w.expert_w_table;
                d.t[4] = w.expert_s_table;
                d.i[0] = c.top_k;
                d.i[1] = lat;
                d.i[2] = imoe;
                d.i[3] = c.n_exp;
                d.i[6] = c.enc;
            });
            b.emit(DevOp::MoeCombine, combine_cus(&all, t * lat), &[c_d], |d| {
                d.t[0] = cmb_dst;
                d.t[3] = part;
                d.i[0] = lat;
                d.i[1] = c.top_k;
            })
        } else {
            // Baseline: one gate/up + down pair per selected slot.
            let mut c_down = Vec::with_capacity(c.top_k as usize);
            for slot in 0..c.top_k {
                let c_g = b.emit(DevOp::MoeExpertGluFp8Blk, all.clone(), &[c_rt, c_xe], |d| {
                    d.t[0] = fu;
                    d.t[1] = xe;
                    d.t[2] = tab;
                    d.t[3] = w.expert_w_table;
                    d.t[4] = w.expert_s_table;
                    d.i[0] = slot;
                    d.i[1] = imoe;
                    d.i[2] = lat;
                    d.i[3] = c.n_exp;
                    d.i[5] = K3_MOE_ACT_SITU;
                    d.i[6] = c.enc;
                    d.f[0] = cb.situ_beta;
                    d.f[1] = cb.situ_linear_beta;
                });
                c_down.push(
                    b.emit(DevOp::MoeExpertDownFp8Blk, all.clone(), &[c_g], |d| {
                        d.t[0] = part;
                        d.t[1] = fu;
                        d.t[2] = tab;
                        d.t[3] = w.expert_w_table;
                        d.t[4] = w.expert_s_table;
                        d.i[0] = slot;
                        d.i[1] = lat;
                        d.i[2] = imoe;
                        d.i[3] = c.n_exp;
                        d.i[6] = c.enc;
                    }),
                );
            }
            // Combine at LATENT width. `t[1]` (residual) and `t[2]` (shared) stay
            // TENSOR_NONE — see the doc comment.
            b.emit(
                DevOp::MoeCombine,
                combine_cus(&all, t * lat),
                &c_down,
                |d| {
                    d.t[0] = cmb_dst;
                    d.t[3] = part;
                    d.i[0] = lat;
                    d.i[1] = c.top_k;
                },
            )
        }
    } else {
        let atom = pf_fuse == crate::mla::MoePfFuse::Atomic;
        let det = pf_fuse == crate::mla::MoePfFuse::Det;
        // The grouped arrays are sized on the MPF_BM-PADDED row bound: the align op rounds each
        // expert's row range up to a whole tile, so every expert can waste up to MPF_BM-1 rows.
        // Sizing them from `T*k` alone is an out-of-bounds device write with no symptom at small
        // expert counts and a guaranteed one at 896.
        let pad_rows = t as u64 * c.top_k as u64 + (c.n_exp * (crate::mla::MPF_BM - 1)) as u64;
        let mx = c.enc == K3_MOE_ENC_MXFP4;
        const ALIGN_BLOCKS: u32 = 64;
        let align_par = crate::emit_config::active().moe_align_par && t >= 1024;
        let meta_ints = 3 * c.n_exp + 1 + u32::from(align_par) * ALIGN_BLOCKS * c.n_exp;
        let meta = b.tensor(&format!("{a}moe_meta"), meta_ints as u64 * 4);
        let row_token = b.tensor(&format!("{a}moe_rowtok"), pad_rows * 4);
        let row_partidx = b.tensor(&format!("{a}moe_rowpart"), pad_rows * 4);
        let row_gate = b.tensor(&format!("{a}moe_rowgate"), pad_rows * 4);
        // The gathered GLU output is bf16 on the bf16/block-fp8 arms and PACKED fp4 under A4W4 —
        // half a byte per value, plus one E8M0 byte per 32. The E8M0 rows have no bf16 counterpart
        // and must be declared, or the fused bridge writes through a null handle.
        let fu_g = b.tensor(
            &format!("{a}moe_fug"),
            if mx {
                pad_rows * (imoe / 2) as u64
            } else {
                pad_rows * imoe as u64 * 2
            },
        );
        let fu_scale = if mx {
            b.tensor(
                &format!("{a}moe_fuscale"),
                pad_rows * (imoe / crate::mla::MX_BLOCK) as u64,
            )
        } else {
            packet::dev::TENSOR_NONE
        };
        // Top-k tail, block per token. Bit-identical PER TOKEN to the decode tail (the kernel is
        // that kernel under a token loop), so a prefill chunk makes the selection decode would
        // have made for the same row — which is what makes the two phases the same model.
        // One workgroup owns one token at a time. Launching more than T only creates empty
        // interpreter entries; launching fewer keeps the kernel's existing strided token loop.
        let router_blocks: Vec<u32> = (0..t.min(n_cu)).collect();
        let c_rt = b.emit(DevOp::MoeRouterTopkPf, router_blocks, &[c_rl], |d| {
            d.t[0] = tab;
            d.t[1] = logit;
            d.t[3] = w.router_bias;
            if atom || det {
                d.t[2] = part;
                d.i[0] = lat;
            }
            d.i[1] = c.n_exp;
            d.i[2] = c.top_k;
            d.i[3] = c.route_flags;
            d.i[4] = t;
            d.i[6] = c.n_group;
            d.i[7] = c.topk_group;
            d.f[0] = c.route_scale;
        });
        let align = |b: &mut Builder, blocks: Vec<u32>, deps: &[u32], phase: u32| {
            b.emit(DevOp::MoeAlignPf, blocks, deps, |d| {
                d.t[0] = meta;
                d.t[1] = tab;
                d.t[2] = row_token;
                d.t[3] = row_partidx;
                d.t[4] = row_gate;
                d.i[0] = t;
                d.i[1] = c.n_exp;
                d.i[2] = c.top_k;
                d.i[3] = phase;
                d.i[4] = u32::from(phase != 0) * ALIGN_BLOCKS;
            })
        };
        let c_align = if align_par {
            let par_blocks: Vec<u32> = all.iter().copied().take(ALIGN_BLOCKS as usize).collect();
            let c_count = align(b, par_blocks.clone(), &[c_rt], 1);
            let c_prefix = align(b, vec![0u32], &[c_count], 2);
            let c_init = align(b, par_blocks.clone(), &[c_prefix], 3);
            align(b, par_blocks, &[c_init], 4)
        } else {
            align(b, vec![0u32], &[c_rt], 0)
        };
        // Grouped gate/up + situ over the sorted rows. `i[1]` is the LATENT, not the hidden: the
        // A operand is gathered from `xe`, which is what makes this a LatentMoE.
        let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_xe], |d| {
            d.t[0] = fu_g;
            d.t[1] = xe;
            d.t[2] = w.expert_w_table;
            d.t[3] = w.expert_s_table;
            d.t[4] = meta;
            d.t[5] = row_token;
            if mx {
                // A4W4 binds two more: t6 = row_partidx so the fused bridge can tell a PAD row
                // from a live one, t7 = the E8M0 rows it WRITES (the bridge is this op's epilogue,
                // not a separate op).
                d.t[6] = row_partidx;
                d.t[7] = fu_scale;
            }
            d.i[0] = imoe;
            d.i[1] = lat;
            d.i[2] = c.n_exp;
            d.i[crate::mla::MoeEnc::PREFILL_SLOT] = c.enc;
            d.i[5] = K3_MOE_ACT_SITU;
            // The betas ride f0/f1 on both grouped epilogues, which call the PAIR form `moe_glu`.
            // `moe_act` returns NaN for the situ code on purpose, so an epilogue that had not been
            // converted would poison rather than silently compute `gelu_tanh(g) * u`.
            d.f[0] = cb.situ_beta;
            d.f[1] = cb.situ_linear_beta;
        });
        // Grouped down + gate-scale + SCATTER into part[T*k, latent]. `row_partidx` carries each
        // gathered row's FIXED destination, so the align op's nondeterministic within-expert row
        // order never reaches the output.
        let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
            d.t[0] = part;
            d.t[1] = fu_g;
            d.t[2] = w.expert_w_table;
            d.t[3] = w.expert_s_table;
            d.t[4] = meta;
            if mx {
                d.t[5] = fu_scale;
            }
            d.t[6] = row_partidx;
            d.t[7] = row_gate;
            d.i[0] = lat;
            d.i[1] = imoe;
            d.i[2] = c.n_exp;
            d.i[crate::mla::MoeEnc::PREFILL_SLOT] = c.enc;
            if atom {
                d.i[4] = c.top_k.trailing_zeros() + 1;
            }
            if det {
                d.i[5] = c.top_k.trailing_zeros() + 1;
            }
        });
        // T-token combine at LATENT width, still with NO residual and NO shared operand: there is
        // nothing 3584 wide to add, so both of those happen after the up-projection.
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = cmb_dst;
            d.t[3] = part;
            d.i[0] = lat;
            d.i[1] = if atom || det { 1 } else { c.top_k };
            d.i[2] = t;
            d.i[4] = u32::from(det);
        })
    };
    if tp.on() {
        c_cmb = crate::emit_xreduce(
            b,
            &mut tp.xgate,
            t == 1,
            &tp.xr_cus,
            c_cmb,
            ylat,
            t * lat,
            tp.tp,
            tp.slot_b,
        );
    }
    // THE LATENT NORM FOLDS INTO THE UP-PROJECTION THAT READS IT — 92 packets, 92 chain levels.
    //
    // This is the `fan=1` half of [`fuse_norm_gemv`]: `yn` is read by the up projection below and
    // by nothing else in the program, so folding costs ZERO redundant reductions (the trap that
    // deleted `op_gemm.h`'s norm mode 2 for Gemma) and deletes a `b=1` packet standing in front
    // of a `b=256` consumer — 256 polls, one edge, and a level of a 1739-deep chain.
    let fold_ln = fuse_norm_gemv(t, lat, NormSite::Lat);
    let c_ln = if fold_ln {
        c_cmb
    } else {
        b.emit(DevOp::RmsNorm, norm_cus(&all, t), &[c_cmb], |d| {
            d.t[0] = yn;
            d.t[1] = ylat;
            d.t[2] = w.latent_norm;
            d.i[0] = t;
            d.i[1] = lat;
            d.f[0] = cb.eps;
        })
    };
    // Folded, the GEMV reads the norm's INPUT and normalizes it on the way in.
    let up_src = if fold_ln { ylat } else { yn };
    // THE UP PROJECTION IS COLUMN-PARALLEL, and it is the largest single weight in the
    // decode stream after the experts themselves.
    //
    // `routed_expert_up_proj` is [hidden, latent] = 7168 x 3584 bf16 = 51.4 MB, and it used
    // to be REPLICATED: 92 MoE layers x 51.4 MB is **4.73 GB streamed per rank per token**,
    // ~0.74 ms at the 6.4 TB/s measured streaming ceiling, seven eighths of it redundant at
    // TP8. It is the same line item the shared expert was (`declare_k3_moe_weights`), and
    // the same fix — the reason it was NOT taken with the shared expert is that the shared
    // expert's `down_proj` is ROW-parallel, so its partial rides an all-reduce this block
    // already has, while this one has no reduction to ride.
    //
    // COLUMN, not ROW, and the choice is forced twice over:
    //
    //  * `yn` is REPLICATED (the latent norm is nonlinear, so the expert combine is already
    //    reduced before it). A row-parallel up would need rank r to read `yn[r*lat/tp ..]` —
    //    a rank-dependent offset into an ACTIVATION, which one blob shared by eight ranks
    //    cannot express, and which no `Gemm*` prefill arm can take anyway (its A operand's
    //    row stride is its K).
    //  * Column splits the OUTPUT, so every element of `yh` is still one dot product over
    //    the FULL 3584-wide latent, in the same order, at the same precision. The values
    //    are BIT-IDENTICAL to the replicated emit; only their assembly changes.
    //
    // What changes is that `yh` is now a CONCATENATION across ranks, not a sum, so it needs
    // an all-gather. That gather is folded into the shared expert's existing all-reduce
    // below — one packet, one rendezvous, for both — because the two results are ADDED.
    let shard_up = shard_up_proj(tp.tp);
    // `hid_l` is the GATHER's per-rank column count; `up_n` is what the GEMV computes. They
    // differ only under `PLOW_K3_UP_GATHER_ONLY`, which keeps the weight whole on purpose.
    let hid_l = if shard_up { tp.local(hid) } else { hid };
    let up_n = if shard_up && !up_gather_only() {
        hid_l
    } else {
        hid
    };
    let up_dst = if shard_up { tp.ug } else { yh };
    let c_up = emit_k3_linear_norm(
        b,
        up_dst,
        up_src,
        w.up_latent,
        t,
        up_n,
        lat,
        n_cu,
        seq_rows,
        fold_ln.then_some((w.latent_norm, cb.eps)),
        &[c_ln],
    );

    // Shared expert, off the PRE-down hidden. THREE packets on both phases, and the fused
    // gate|up GEMM is deliberately not taken even at T rows: every fused GLU epilogue in
    // `op_gemm.h` — `GemmGlu`, `GemmGluMxfp4`, all five rungs — calls `act_gate_only`, which by
    // construction computes `act(g) * u` and returns NaN for the situ code. situ transforms the UP
    // branch too, so a fused emit here would poison (loudly, by design) rather than run. The
    // fusion is a real win on GLM's silu shared expert and is not available to this activation
    // until an epilogue takes the betas the way `moe_glu` does.
    // FUSED gate|up|situ when the epilogue can take the betas, three packets otherwise.
    //
    // This is the fusion the comment above used to say was unavailable, and the blocker was real:
    // every fused GLU epilogue called `act_gate_only`, which returns NaN for the situ code BY
    // DESIGN because situ transforms the UP branch too. `gemv_glu_rows` now open-codes the pair
    // form (`k3_situ_gate * k3_situ_up`, the same split `moe_glu` makes) and takes the betas in
    // f[0]/f[1] exactly as `SituGlu` does, so the arm exists for the first time.
    //
    // Why this merge is safe where the MLA down-projection merge was NOT (see `fuse_mla_a`): that
    // one concatenated three INDEPENDENT ops that the global queue was already running
    // concurrently, so it traded a max for a sum. This one is output-stationary — one wave owns
    // column n and streams BOTH weight rows for it — so the byte count is unchanged, two packets
    // and one whole activation round-trip of `sha` disappear, and each wave now has twice the loads
    // in flight, which is the starvation `op_gemm.h`'s own GEMV sweep names as the sub-ceiling cause.
    let fused_sh = fuse_shared_glu(t, c.hidden);
    let c_sa = if fused_sh {
        b.emit(
            DevOp::GemvGlu,
            crate::mla::blocked_gemv_cus(&all, si_l),
            deps,
            |d| {
                d.t[0] = sha;
                d.t[1] = x;
                d.t[2] = w.shared_gate;
                d.t[5] = w.shared_up;
                d.i[0] = t;
                d.i[1] = si_l;
                d.i[2] = hid;
                d.i[5] = K3_MOE_ACT_SITU;
                d.f[0] = cb.situ_beta;
                d.f[1] = cb.situ_linear_beta;
            },
        )
    } else {
        let (c_sg, c_su) = if let Some(pair) = early_shared {
            pair
        } else {
            (
                gemv(b, shg, x, w.shared_gate, si_l, hid, deps[0]),
                gemv(b, shu, x, w.shared_up, si_l, hid, deps[0]),
            )
        };
        emit_situ_glu(b, cb, sha, shg, shu, t * si_l, n_cu, &[c_sg, c_su])
    };

    // THE DOWN PROJECTION WRITES THE PEER SLOT, NOT A LOCAL BUFFER — and that this was ever
    // otherwise is the bug that made every TP>1 K3 emit the wrong model.
    //
    // `d_xreduce` (runtime/amd/op_collective.h:132) sums `peer_scratch[r] + slot` over every rank
    // and writes `out` WITHOUT READING IT. So a rank's own partial reaches the sum only by being
    // in ITS OWN peer region at that offset. Every other row-parallel site in this emitter obeys
    // that — `o_proj` writes `act.og_tp` and reduces into `attn_full`, `MoeCombine` writes
    // `act.dg_tp` and reduces into `ylat` — but the shared expert's `down_proj` wrote the ordinary
    // arena tensor `shd` and then reduced slot 0 INTO the same `shd`.
    //
    // Nothing faulted. Slot 0 is `act.og_tp`, which this layer's `o_proj` filled a few packets
    // earlier, so the reduce returned `sum_r(o_proj partial_r)` — the ATTENTION OUTPUT, recomputed
    // — and every rank's shared-expert result was overwritten by it. The layer then computed
    //
    //     ffn = up_latent + attn        instead of      ffn = up_latent + shared_expert
    //
    // on 92 of 93 layers, with the shared expert (2 x 3072 intermediate, the FFN's dense half)
    // discarded entirely. Finite, stable, plausible logits and a model that predicts punctuation.
    // Invisible at tp=1, where `tp.on()` is false and `shd` is simply the answer — which is why a
    // tp1-vs-tp8 A/B on the SAME asset is what found it: `K3_NLAYERS=1` (dense MLP, no shared
    // expert) agrees to the token, `K3_NLAYERS=2` (the first latent MoE) does not.
    //
    // ORDERED AFTER THE EXPERT-COMBINE COLLECTIVE, which is new and is load-bearing. The reuse of
    // slot A is only safe because a rank cannot overwrite its slot-A partial until every peer has
    // finished READING it, and the one-shot's gate does not say that on its own — it says every
    // peer ARRIVED. What does say it is the intervening collective: this rank cannot pass the
    // expert-combine reduce until every peer has signalled it, and a peer signals it only after
    // leaving the attention reduce. Before this dependency the shared-expert chain hung off `deps`
    // alone and could reach its slot-A write while a peer was still reading slot A for the
    // attention reduce. It costs the overlap between the shared expert's down GEMV and the routed
    // chain; a third `act.*_tp` slot would buy it back, at the price of a host change (the peer
    // slots are bound by literal name, `plowrt::exec::amd:2861-2862`) and a peer-region resize.
    let sh_dst = if tp.on() { tp.og } else { shd };
    let sh_deps: Vec<u32> = if tp.on() {
        vec![c_sa, c_cmb]
    } else {
        vec![c_sa]
    };
    let mut c_sd = emit_k3_linear(
        b,
        sh_dst,
        sha,
        w.shared_down,
        t,
        hid,
        si_l,
        n_cu,
        seq_rows,
        &sh_deps,
    );

    // ONE COLLECTIVE FOR BOTH FFN HALVES, and the packet count is why it is one and not two.
    //
    // The two halves are shaped differently and neither can be summed locally first:
    // `shd` is a ROW-parallel partial (`sum_r shd_r` is the answer) and `yh` is now a
    // COLUMN-parallel slice (`concat_r yh_r` is the answer). But they are ADDED, so
    //
    //     ffn = sum_r shd_r + concat_r yh_r
    //
    // is one reduction with a gathered addend, and `d_xreduce`'s `gcols` operand does
    // exactly that: slot 0 is reduced over every rank, slot 2 is read from the rank that
    // OWNS each output column. Gathering in its own packet instead would cost a packet AND
    // a rendezvous per MoE layer — ~5.3 us x 92 = 0.49 ms/token by this tree's own
    // measurement (`runtime/amd/op_norm.h`), against the 0.74 ms the sharding saves. The
    // fold makes the sharding free: the packet count is IDENTICAL to the replicated emit.
    //
    // BIT-EXACT, and the bf16 round inside `d_xreduce`'s gather arm is what makes it so:
    // unfolded, the collective STORED `f2bf(sum_r shd_r)` and the residual re-read it, so the
    // gather add must see the ROUNDED reduction, not the f32 one. Skipping that round is a
    // 1-ULP difference per element per layer, and 92 layers of it flips a token — measured,
    // ". The capital of X is Y. The capital of ..." instead of the real continuation. The
    // kernel's comment carries the arithmetic.
    //
    // Ordered after the shared down GEMV *and* the up GEMV: both partials must be
    // agent-visible before this rank signals, and the up GEMV is the only writer of slot 2.
    //
    // Slot 2 is safe to reuse next layer for the same reason slot 0 is
    // (`perf-data/archive/k3/kimi-k3-tp-peer-slots.md`): a peer signals the NEXT layer's attention
    // reduce only after leaving this one, and two collectives separate the two writes.
    if shard_up {
        // With the outer residual folded in, one add is still owed (`out = ffn + prefix`) and
        // the collective lands in `shd`. Without it, the collective already IS the answer and
        // writes `out` directly — one packet fewer than the replicated emit, not one more.
        let ffn = if resid_in.is_some() { shd } else { out };
        let gather = if up_nogather() {
            None
        } else {
            Some((tp.slot_c(), hid_l, hid))
        };
        c_sd = crate::emit_xreduce_gather(
            b,
            &mut tp.xgate,
            t == 1,
            &tp.xr_cus,
            &[c_sd, c_up],
            ffn,
            t * hid,
            tp.tp,
            0,
            gather,
        );
        return match resid_in {
            Some(p) => b.emit(DevOp::Residual, vec8_cus(&all, t * hid), &[c_sd], |d| {
                d.t[0] = out;
                d.t[1] = ffn;
                d.t[2] = p;
                d.i[0] = t * hid;
                d.f[0] = 1.0;
            }),
            None => c_sd,
        };
    }

    // `shd` is a row-parallel PARTIAL while `yh` (up_latent) is replicated and therefore WHOLE, so
    // they cannot be summed locally first: `sum_r(yh + shd_r)` is `tp*yh + sum_r(shd_r)`, which is
    // finite, plausible and wrong. Reduce to whole, THEN add.
    if tp.on() {
        c_sd = crate::emit_xreduce(
            b,
            &mut tp.xgate,
            t == 1,
            &tp.xr_cus,
            c_sd,
            shd,
            t * hid,
            tp.tp,
            0,
        );
    }

    b.emit(
        DevOp::Residual,
        vec8_cus(&all, t * hid),
        &[c_up, c_sd],
        |d| {
            d.t[0] = out;
            d.t[1] = yh;
            d.t[2] = shd;
            // The OUTER residual, folded. `d_residual` rounds the inner sum to bf16 before adding it,
            // which is exactly what the deleted `{a}ffn` store did, so this is bit-exact.
            d.t[3] = resid_in.unwrap_or(packet::dev::TENSOR_NONE);
            d.i[0] = t * hid;
            d.f[0] = 1.0;
        },
    )
}

/// K3's MLA geometry. Shares K2's shape; the two departures are the OUTPUT GATE
/// and NoPE (`mla_use_nope`, so `rope_theta` is absent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct K3MlaCfg {
    pub hidden: u32,
    /// `num_attention_heads` — the LOCAL count under TP.
    pub heads: u32,
    /// `q_lora_rank`.
    pub q_lora: u32,
    /// `kv_lora_rank` — the latent width every head reads.
    pub kv_lora: u32,
    /// Unabsorbed NoPE query/key width. Used only by the default-off materialized prefill path.
    pub qk_nope: u32,
    /// `qk_rope_head_dim`.
    pub qk_rope: u32,
    /// `v_head_dim`.
    pub v_head: u32,
    pub eps: f32,
    /// `1 / sqrt(qk_nope + qk_rope)`, folded at prep time.
    pub scale: f32,
    /// FlashMLA split count, and the merge-fold's `i[4]`.
    pub n_split: u32,
    /// FlashMLA's group factor, `i[7]`. Must divide the LOCAL head count — at
    /// TP8 K3 gives `nh_l = 12`, which 8 does not divide and 4 does.
    pub gf: u32,
    /// Store the 512-wide compressed latent KV in e4m3 with one f32 scale per row.
    /// The 64-wide NoPE/rope cache stays bf16: quantizing it saves only another 5.2% of the
    /// original cache while adding a second source of attention error.
    pub fp8_kv: bool,
}

impl K3MlaCfg {
    /// `n_head_local * v_head_dim` — the width the output gate and `o_proj` see.
    pub fn nhvd(&self) -> u32 {
        self.heads * self.v_head
    }
}

/// One MLA layer's weight and cache handles.
///
/// The three `derived.*` entries are ABSORBED weights written by the prep, not
/// raw checkpoint tensors: the kernel never sees `q_pass`, `k_pass` or
/// `qk_nope` at all, it works entirely in the latent. They are also the three
/// MLA names `plowrt`'s shard classifier knows as column-parallel.
pub struct K3MlaWeights {
    /// `input_layernorm.weight`.
    pub ln_w: u32,
    pub q_a: u32,
    pub q_a_norm: u32,
    /// `derived.q_absorb` — `[heads*kv_lora, q_lora]`.
    pub q_absorb: u32,
    /// `derived.q_rope` — `[heads*qk_rope, q_lora]`.
    pub q_rope: u32,
    /// Raw checkpoint projections retained only when materialized prefill is requested.
    pub q_b: u32,
    pub kv_b: u32,
    /// `kv_a_proj_with_mqa`, the shared latent down-projection.
    pub kv_a: u32,
    /// The k-rope down-projection, `[qk_rope, hidden]`.
    pub k_rope_d: u32,
    pub kv_a_norm: u32,
    /// `derived.v_absorb`, the merge-fold's weight.
    pub v_absorb: u32,
    /// `o_gate_proj` — K3 only.
    pub o_gate: u32,
    pub o_proj: u32,
    /// The NoPE identity table (see [`k3_nope_rope_pair`]) and the position row.
    pub cos: u32,
    pub sin: u32,
    pub pos: u32,
    /// `kv.{l}.ckv` / `kv.{l}.krot` cache rows, and `in.kvlen`.
    pub ckv: u32,
    pub krot: u32,
    /// One f32 dequantization scale per latent-cache row, or `TENSOR_NONE` for bf16 KV.
    pub kv_scale: u32,
    pub kvlen: u32,
}

/// Emit one K3 MLA MIXER: pre-norm through `o_proj`, returning `(counter, attn)`.
///
/// Mirrors [`crate::kda::emit_kda_mixer`] — no residual, for the same reason.
///
/// Three things here are K3-specific and each is a way to build a model that
/// runs and is wrong:
///
/// * **The output gate** sits between `MlaMergeFold` and `o_proj`, and its
///   `g_proj` reads the POST-`input_layernorm` hidden — `x` below — not the
///   attention output. HF gates on `hidden_states`. Here the two have different
///   widths (hidden vs heads*v_head) so that mistake fails loudly, but gating
///   on the post-attention norm instead would not.
/// * **NoPE.** Both `HeadNormRope` ops are still emitted, with an identity
///   cos=1/sin=0 table, so each is a bit-exact copy. They are NOT dropped: the
///   k-side one is the only writer of the `krot` cache row AND the instruction
///   the runtime scans for to patch the row position each step. Delete it and
///   the rope half of every cached key stays uninitialised while FlashMLA keeps
///   reading it — garbage that grows with context and never faults.
/// * **The absorption.** `q_absorb` outputs the query already in latent space,
///   so `FlashMlaDecode` reads one 512-wide latent plus the shared 64-wide rope
///   row for every head.
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_mla_mixer(
    b: &mut Builder,
    c: &K3MlaCfg,
    w: &K3MlaWeights,
    act_prefix: &str,
    t: u32,
    ctx: u32,
    hidden: u32,
    // Where `o_proj` writes; under TP this is the peer-visible partial slot.
    attn_dst: Option<u32>,
    n_cu: u32,
    // `true` when the CALLER has already normed `hidden` — K3's `AttnRes` absorbs the pre-norm
    // (see [`fuse_attnres_norm`]), so this mixer must not norm it a second time.
    prenormed: bool,
    deps: &[u32],
    // `true` when the `t` rows are INDEPENDENT SEQUENCES (batched decode) rather than
    // consecutive tokens of one. Selects the DECODE flash arm at t > 1 and sets n_batch.
    seq_rows: bool,
) -> (u32, u32) {
    let all: Vec<u32> = (0..n_cu).collect();
    let a = act_prefix;
    let (nh, dk, dn, dr, vd, ql) = (c.heads, c.kv_lora, c.qk_nope, c.qk_rope, c.v_head, c.q_lora);
    let decode_arm = t == 1 || seq_rows;
    let materialized = emit_config::active().mla_materialized_prefill
        && !decode_arm
        && !c.fp8_kv
        && dn + dr == 192
        && vd == 128
        && w.q_b != TENSOR_NONE
        && w.kv_b != TENSOR_NONE;
    let nhvd = c.nhvd();
    let tt = t as u64;
    let bft = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 2);
    let f32t = |b: &mut Builder, n: String, e: u64| b.tensor(&n, e * 4);

    // `prenormed` makes the caller's AttnRes output the pre-normed `x` itself: no second buffer,
    // no P0 packet.
    let x = if prenormed {
        hidden
    } else {
        bft(b, format!("{a}x"), tt * c.hidden as u64)
    };
    let qlr = bft(b, format!("{a}q_lora"), tt * ql as u64);
    // Unwritten when the q-side norm folds; declared anyway, for the reason `{a}yn` is.
    let qlat = bft(b, format!("{a}q_lat"), tt * ql as u64);
    let qa = bft(
        b,
        format!("{a}q_absorbed"),
        tt * nh as u64
            * if materialized {
                (dn + dr) as u64
            } else {
                dk as u64
            },
    );
    let qrr = bft(b, format!("{a}q_rope_raw"), tt * nh as u64 * dr as u64);
    let qr = bft(b, format!("{a}q_rope"), tt * nh as u64 * dr as u64);
    let ckvraw = bft(b, format!("{a}ckv_raw"), tt * dk as u64);
    let krr = bft(b, format!("{a}krot_raw"), tt * dr as u64);
    let kvmat = materialized
        .then(|| {
            bft(
                b,
                format!("{a}kv_materialized"),
                tt * nh as u64 * (dn + vd) as u64,
            )
        })
        .unwrap_or(TENSOR_NONE);
    let kmat = materialized
        .then(|| {
            bft(
                b,
                format!("{a}k_materialized"),
                tt * nh as u64 * (dn + dr) as u64,
            )
        })
        .unwrap_or(TENSOR_NONE);
    let vmat = materialized
        .then(|| bft(b, format!("{a}v_materialized"), tt * nh as u64 * vd as u64))
        .unwrap_or(TENSOR_NONE);
    // NSPLIT IS 1 ON THE PREFILL ARM, and it is forced rather than chosen: under a per-token
    // causal bound an early token's later splits cover nothing, and an empty split emits `l = 0`
    // for the merge to divide by (`d_flash_mla_prefill`'s header). Prefill already has
    // `n_tok * n_grp` work items and does not need the split to fill the machine.
    //
    // IT MUST TRACK `decode_arm` BELOW, not `t == 1`. A batched decode has `t > 1` and still takes
    // the DECODE arm, which writes `n_split` splits per row — sizing these two buffers at
    // `nsplit = 1` while the kernel writes `c.n_split` is a silent overrun, and it is silent
    // twice over: the arena is contiguous, so the writes land in whatever tensor follows and the
    // run completes with plausible garbage rather than faulting. This is the same
    // "`t > 1` does not mean prefill" trap the arm selection below spells out, one buffer earlier.
    let nsplit = if t == 1 || seq_rows { c.n_split } else { 1 };
    let opart = f32t(
        b,
        format!("{a}o_part"),
        tt * nh as u64 * dk as u64 * nsplit as u64,
    );
    let mlpart = f32t(b, format!("{a}ml_part"), tt * nh as u64 * 2 * nsplit as u64);
    let oat = bft(b, format!("{a}o_attn"), tt * nhvd as u64);
    let gl = bft(b, format!("{a}o_gate_raw"), tt * nhvd as u64);
    let oatg = bft(b, format!("{a}o_gated"), tt * nhvd as u64);
    let attn = match attn_dst {
        Some(h) => h,
        None => bft(b, format!("{a}attn"), tt * c.hidden as u64),
    };

    let gemv = |b: &mut Builder, out: u32, row: u32, wt: u32, n: u32, k: u32, dep: &[u32]| {
        emit_k3_linear(b, out, row, wt, t, n, k, n_cu, seq_rows, dep)
    };
    let rms = |b: &mut Builder, out: u32, inp: u32, gw: u32, n: u32, dep: &[u32]| {
        b.emit(DevOp::RmsNorm, norm_cus(&all, t), dep, |d| {
            d.t[0] = out;
            d.t[1] = inp;
            d.t[2] = gw;
            d.i[0] = t;
            d.i[1] = n;
            d.f[0] = c.eps;
        })
    };

    // Pre-norm. `x` feeds the q down-proj, the kv down-proj, the k-rope
    // down-proj AND g_proj — four consumers, which is why the gate's dependency
    // is on this and not on the attention output.
    let c_ln = if prenormed {
        deps[0]
    } else {
        rms(b, x, hidden, w.ln_w, c.hidden, deps)
    };

    // FUSION A. Four down-projections read `x` at `K = hidden`; their columns concatenate into one
    // four-stream packet. `W_g` is the ninth pointer and rides in `i[6]` as a handle — `t[8]` is
    // full, and the demoted operand is a WEIGHT, never an output, which is the rule `GemvQkvg`
    // states. Emitted BEFORE the rope pair so `gl` is ready early; it only ever depended on `x`.
    let (c_qad, c_ckvd, c_krr) = if fuse_mla_a(t, c.hidden) {
        let f = b.emit(
            DevOp::GemvQkv,
            crate::mla::blocked_gemv_cus(&all, ql + dk + dr),
            &[c_ln],
            |d| {
                d.t[0] = qlr;
                d.t[1] = x;
                d.t[2] = w.q_a;
                d.t[3] = ckvraw;
                d.t[4] = w.kv_a;
                d.t[5] = krr;
                d.t[6] = w.k_rope_d;
                d.i[0] = t;
                d.i[1] = ql;
                d.i[2] = c.hidden;
                d.i[3] = dk;
                d.i[4] = dr;
            },
        );
        (f, f, f)
    } else {
        (
            gemv(b, qlr, x, w.q_a, ql, c.hidden, &[c_ln]),
            gemv(b, ckvraw, x, w.kv_a, dk, c.hidden, &[c_ln]),
            gemv(b, krr, x, w.k_rope_d, dr, c.hidden, &[c_ln]),
        )
    };
    // `q_a_layernorm` folds into BOTH GEMVs that read it — the `fan=2` half of
    // [`fuse_norm_gemv`]. Unlike the latent norm this one is not free: `q_absorb` and `q_rope`
    // each redo the reduction, so it costs ONE extra reduction over `ql`=1536 elements that are
    // already staged in LDS. That buys a `b=1` packet in front of two `b=256` consumers — 512
    // polls and a level of the chain. Both consumers must fold or neither can: leaving one
    // reading `qlat` would keep the RMSNORM alive and the fusion would buy nothing.
    let fold_q = fuse_norm_gemv(t, ql, NormSite::Q);
    let c_rnq = if fold_q {
        c_qad
    } else {
        rms(b, qlat, qlr, w.q_a_norm, ql, &[c_qad])
    };
    let q_src = if fold_q { qlr } else { qlat };
    let qn = fold_q.then_some((w.q_a_norm, c.eps));
    let (c_qa, c_qrr) = if materialized {
        let c_q = emit_k3_linear_norm(
            b,
            qa,
            q_src,
            w.q_b,
            t,
            nh * (dn + dr),
            ql,
            n_cu,
            seq_rows,
            qn,
            &[c_rnq],
        );
        (c_q, c_q)
    } else {
        (
            emit_k3_linear_norm(
                b,
                qa,
                q_src,
                w.q_absorb,
                t,
                nh * dk,
                ql,
                n_cu,
                seq_rows,
                qn,
                &[c_rnq],
            ),
            emit_k3_linear_norm(
                b,
                qrr,
                q_src,
                w.q_rope,
                t,
                nh * dr,
                ql,
                n_cu,
                seq_rows,
                qn,
                &[c_rnq],
            ),
        )
    };

    // q-side HeadNormRope: identity table, gamma absent, skip_norm — a bit-exact
    // copy. Emitted rather than skipped so it stays checkable.
    // The three rope-family packets below are gated on three independent GEMVs and therefore run
    // CONCURRENTLY, so each takes a DISJOINT slice run — narrowing all three to `0..need` would
    // serialise them onto the same workgroups. Same rule as GLM's `rope_cus` pair.
    let rq = k3_rope_cus(&all, 0, t, nh);
    let rkv = k3_rope_cus(&all, rq.len(), t, 1);
    let rk = k3_rope_cus(&all, rq.len() + rkv.len(), t, 1);
    let c_qr = if materialized {
        c_qrr
    } else {
        b.emit(DevOp::HeadNormRope, rq.clone(), &[c_qrr], |d| {
            d.t[0] = qr;
            d.t[1] = qrr;
            d.t[3] = w.cos;
            d.t[4] = w.sin;
            d.t[5] = w.pos;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = dr;
            d.i[3] = 0; // out_row0: q is not cached.
            d.i[4] = 1; // skip_norm
            d.f[0] = c.eps;
            d.j[1] = crate::KV_MASK_NONE;
        })
    };

    // kv_a_layernorm, writing the LATENT cache row. The fp8 spelling reuses
    // HeadNormRopeFp8 with no trig tables: it computes the same RMSNorm, quantizes the row, and
    // records its scale in one pass. The runtime already recognizes that opcode as a KV writer
    // and patches `i[3]`; the bf16 RmsNorm writer uses `i[2]`.
    let c_rnkv = if c.fp8_kv {
        b.emit(DevOp::HeadNormRopeFp8, rkv.clone(), &[c_ckvd], |d| {
            d.t[0] = w.ckv;
            d.t[1] = ckvraw;
            d.t[2] = w.kv_a_norm;
            d.t[5] = w.pos;
            d.t[6] = w.kv_scale;
            d.i[0] = t;
            d.i[1] = 1;
            d.i[2] = dk;
            d.i[3] = 0; // row, patched per step
            d.i[4] = 0; // apply RMSNorm before quantizing
            d.i[7] = ctx; // packed-prefill per-slot cache row stride
            d.f[0] = c.eps;
            d.j[1] = crate::KV_MASK_NONE;
            // BATCHED DECODE WRITES ITS OWN RING AT ITS OWN POSITION. `i[6]` is the kernel's
            // `n_batch_kv` and `j[0]` its `out_stride` (op_norm.h): together they select
            //     obase = ((t*nhead + hh)*out_stride + pos[t]) * hd
            // instead of the legacy `(out_row0 + t)`. The legacy form CANNOT express a batch:
            // `out_row0` is ONE operand the host patches with ONE position per step, so four
            // sequences at four different positions have nowhere to go — rows 1..B-1 land on
            // sequence 0's ring as if they were its next tokens, and sequence 0 itself is written
            // at the step's patched row while flash reads `kv_len[0]-1`.
            //
            // At `nhead == 1` (the latent is one head) `out_stride = ctx` is ALSO what the
            // legacy branch computes, so this is only reachable-path selection, not a layout
            // change — and it is gated on `seq_rows` regardless, so every prefill packet and
            // every B=1 decode is byte-identical.
            if seq_rows {
                d.i[6] = t;
                d.j[0] = ctx;
            }
        })
    } else {
        b.emit(DevOp::RmsNorm, norm_cus(&all, t), &[c_ckvd], |d| {
            d.t[0] = w.ckv;
            d.t[1] = ckvraw;
            d.t[2] = w.kv_a_norm;
            d.i[0] = t;
            d.i[1] = dk;
            d.i[2] = 0; // row, patched per step
            d.i[7] = ctx; // packed-prefill per-slot cache row stride
            d.f[0] = c.eps;
        })
    };

    // k-side HeadNormRope — THE ONE THAT MUST NOT BE DELETED. Only writer of the
    // krot row, and the instruction the runtime scans for to patch `i[3]`.
    let c_krd = b.emit(DevOp::HeadNormRope, rk.clone(), &[c_krr], |d| {
        d.t[0] = w.krot;
        d.t[1] = krr;
        d.t[3] = w.cos;
        d.t[4] = w.sin;
        d.t[5] = w.pos;
        d.i[0] = t;
        d.i[1] = 1;
        d.i[2] = dr;
        d.i[3] = 0; // row, patched per step
        d.i[4] = 1;
        d.i[7] = ctx; // packed-prefill per-slot cache row stride
        d.f[0] = c.eps;
        d.j[1] = crate::KV_MASK_NONE;
        // Same batch-major selection as the latent writer above; see its note.
        if seq_rows {
            d.i[6] = t;
            d.j[0] = ctx;
        }
    });

    // THE ONE OPCODE THAT DIFFERS BETWEEN THE PHASES, and the operand slot that is REINTERPRETED
    // rather than added: `i[4]` carries `nsplit` on the decode arm and `n_tok` on the prefill one.
    // Everything else is the same packet. `FlashGatherPrefill` is not an option here — K3 has no
    // sparse selector at all (`mla_use_nope`, dense full-causal attention on all 24 MLA layers).
    // WHICH ARM `t > 1` MEANS IS NOT DECIDABLE FROM `t`. A prefill program has `t` consecutive
    // tokens of ONE sequence and wants the PREFILL arm with `i[4] = n_tok`; a batched decode
    // program has `t` INDEPENDENT sequences of one token each and wants the DECODE arm with
    // `i[4] = n_split` and `n_batch = t`. Selecting on `t == 1` alone silently routes a batched
    // decode to the prefill kernel, which would then read the `t` rows as one sequence's history.
    let c_fl = if materialized {
        let c_kvmat = emit_k3_linear(
            b,
            kvmat,
            w.ckv,
            w.kv_b,
            t,
            nh * (dn + vd),
            dk,
            n_cu,
            false,
            &[c_rnkv],
        );
        let c_pack = b.emit(
            DevOp::MlaMaterializePack,
            all.clone(),
            &[c_kvmat, c_krr],
            |d| {
                d.t[0] = kmat;
                d.t[1] = vmat;
                d.t[2] = kvmat;
                d.t[3] = krr;
                d.i[0] = t;
                d.i[1] = nh;
                d.i[2] = dn;
                d.i[3] = dr;
                d.i[4] = vd;
            },
        );
        b.emit(
            DevOp::FlashMlaMaterializedPrefill,
            all.clone(),
            &[c_qa, c_pack],
            |d| {
                d.t[0] = oat;
                d.t[1] = qa;
                d.t[2] = kmat;
                d.t[3] = vmat;
                d.i[0] = t;
                d.i[1] = nh;
                d.i[2] = nh;
                d.i[3] = dn + dr;
                d.i[4] = vd;
                d.i[5] = 1;
                d.f[0] = c.scale;
            },
        )
    } else {
        b.emit(
            match (decode_arm, c.fp8_kv) {
                (true, false) => DevOp::FlashMlaDecode,
                (false, false) => DevOp::FlashMlaPrefill,
                (true, true) => DevOp::FlashMlaDecodeFp8,
                (false, true) => DevOp::FlashMlaPrefillFp8,
            },
            // Decode's work-item count is `(nh/gf) * nsplit`; prefill's `i[4]` is `n_tok`, so the
            // count saturates the machine at every bucket and the narrowing is inert there.
            if decode_arm {
                k3_flash_cus(&all, t, nh, c.gf, c.n_split)
            } else {
                all.clone()
            },
            &[c_qa, c_qr, c_rnkv, c_krd],
            |d| {
                d.t[0] = opart;
                d.t[1] = mlpart;
                d.t[2] = qa;
                d.t[3] = qr;
                d.t[4] = w.ckv;
                d.t[5] = w.krot;
                d.t[6] = w.kvlen;
                if c.fp8_kv {
                    d.t[7] = w.kv_scale;
                }
                // n_batch. One sequence unless the rows ARE sequences, in which case each owns its
                // own KV region and the kernel strides them by this axis.
                d.i[0] = if seq_rows { t } else { 1 };
                d.i[1] = nh;
                d.i[2] = ctx;
                d.i[3] = 0; // window: dense, full causal
                d.i[4] = if decode_arm { c.n_split } else { t };
                d.i[5] = crate::KV_MASK_NONE;
                d.i[6] = 0; // keep the 64-wide NoPE/rope cache in bf16
                d.i[7] = c.gf;
                d.f[0] = c.scale;
            },
        )
    };
    // The partials are `[b][t][head][nsplit][DK]` and the fold indexes them as `(b*n_head + h)`,
    // so the TOKEN axis folds into `i[0]`: `n_batch := 1*t`. Same identity the flash uses
    // (`qrow = (b*n_tok + t)*n_head`), not a trick. At nsplit=1 the online-softmax merge is a
    // pass-through and this op is purely the W_uv fold, which is why no separate `OUvFold` is
    // emitted — `MlaMergeFold` subsumes it and both objects carry it.
    let c_uv = if materialized {
        c_fl
    } else {
        b.emit(DevOp::MlaMergeFold, all.clone(), &[c_fl], |d| {
            d.t[0] = oat;
            d.t[1] = opart;
            d.t[2] = mlpart;
            d.t[3] = w.v_absorb;
            d.i[0] = t;
            d.i[1] = nh;
            d.i[2] = vd;
            d.i[4] = nsplit;
        })
    };

    // The output gate, off the PRE-attention normed hidden.
    // Its own packet ON PURPOSE — see `fuse_mla_a`: folding it into the fused down-projection is
    // legal, removes a packet, and LOSES 1.19 ms at ctx 32000 because it stops overlapping flash.
    let c_gl = gemv(b, gl, x, w.o_gate, nhvd, c.hidden, &[c_ln]);
    let c_gt = emit_mla_out_gate(b, oatg, oat, gl, t * nhvd, n_cu, &[c_uv, c_gl]);
    let c_o = gemv(b, attn, oatg, w.o_proj, c.hidden, nhvd, &[c_gt]);
    (c_o, attn)
}

/// Tensor-parallel state threaded through a K3 emit.
///
/// `tp == 1` is the identity: no collective is emitted and every handle stays
/// local, so a TP1 blob is byte-identical to one built without this.
///
/// TWO all-reduces per layer, the same count GLM runs, and their POSITIONS are
/// the part worth stating:
///
/// * after `o_proj`, which is row-parallel over the head axis;
/// * after the expert DOWN, at the **latent** width — not at the FFN output.
///   That is forced by the LatentMoE: `routed_expert_norm` sits between the
///   combine and the up-projection, and a norm is nonlinear, so the partial
///   sums must already be reduced when it runs. Reducing after the up-projection
///   instead would normalise 1/tp of the activation and is finite, plausible and
///   wrong.
///
/// A third collective would have been the price of the up-projection's shard, so it is
/// FOLDED into the second: `routed_expert_up_proj` is column-parallel and its all-gather
/// rides `d_xreduce`'s `gcols` operand inside the shared expert's all-reduce, out of the
/// GATHER slot (`K3Tp::slot_c`). Two collectives, three row/column-parallel producers.
///
/// What is still REPLICATED: `routed_expert_down_proj` and `routed_expert_norm`. The down
/// projection's output feeds experts sharded on their INTERMEDIATE, so every rank needs
/// the whole latent vector and any shard of it needs a gather with no existing collective
/// at that point to ride — ~0.74 ms/token of streaming saved against at least a packet
/// (~5.3 us) x 92 layers, which is not a trade worth taking. The norm is [latent].
/// This matches the classification `plowrt`'s `shard_of` gives those names.
pub struct K3Tp {
    pub tp: u32,
    /// Peer-visible partial slot for the attention output.
    pub og: u32,
    /// Peer-visible partial slot for the expert-combine output.
    pub dg: u32,
    /// Peer-visible GATHER slot for the column-parallel `routed_expert_up_proj`
    /// partial. Slot 2; folded into slot 0's reduce, see [`K3Tp::slot_c`].
    pub ug: u32,
    /// Byte offset of the second peer slot.
    pub slot_b: u32,
    /// Running xctr gate id allocator.
    pub xgate: u32,
    /// CUs the collective may use.
    pub xr_cus: Vec<u32>,
}

impl K3Tp {
    /// A TP1 context: no collectives, no peer slots.
    pub fn none() -> Self {
        K3Tp {
            tp: 1,
            og: 0,
            dg: 0,
            ug: 0,
            slot_b: 0,
            xgate: 0,
            xr_cus: Vec::new(),
        }
    }
    pub fn on(&self) -> bool {
        self.tp > 1
    }
    /// Byte offset of the GATHER slot (partial slot 2). The host binds `act.ug_tp` at
    /// `scratch_base + 2 * slot_b` (`plowrt::exec::amd`, `PARTIAL_SLOTS`), so this
    /// arithmetic is the emitter's half of that literal-name contract.
    pub fn slot_c(&self) -> u32 {
        2 * self.slot_b
    }
    /// Local share of a width that shards across ranks.
    pub fn local(&self, w: u32) -> u32 {
        assert_eq!(
            w % self.tp,
            0,
            "K3: --num-gpus {} must divide {w}; a remainder would leave the tail columns \
             unowned and every rank would silently compute a short row",
            self.tp
        );
        w / self.tp
    }
}

/// Which mixer a K3 layer carries. The partition comes from the config's
/// `full_attn_layers` / `kda_layers` LISTS, never a stride — at 0-based layer 92
/// a modulus rule disagrees with the list.
pub enum K3Mixer<'a> {
    Kda {
        cfg: &'a crate::kda::KdaCfg,
        w: &'a crate::kda::KdaWeights,
        st: &'a crate::kda::KdaState,
    },
    Mla {
        cfg: &'a K3MlaCfg,
        w: &'a K3MlaWeights,
    },
}

/// Which FFN a K3 layer carries.
///
/// `layer < first_k_dense_replace` (1 on the shipped config, so layer 0 alone)
/// is a plain dense FFN; every other layer is the LatentMoE. Both use `situ`,
/// and both are `[hidden] -> [inter] -> [hidden]` from the outside — the
/// difference is entirely inside.
pub enum K3Ffn<'a> {
    Dense {
        gate: u32,
        up: u32,
        down: u32,
        /// `intermediate_size` — the DENSE width, not `moe_intermediate_size`.
        inter: u32,
    },
    Latent(&'a K3MoeWeights),
}

/// Emit K3's dense FFN: `down(situ(gate(x), up(x)))`.
///
/// Layer 0 only, and the two shared experts reuse the same shape. `SituGlu` is
/// a separate packet here rather than an `act` code on a fused GEMM, because
/// every fused gate/up path in this tree is gate-only and now REFUSES situ with
/// a NaN (`act_gate_only`, op_elementwise.h).
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_dense_mlp(
    b: &mut Builder,
    cb: &K3BlockCfg,
    act_prefix: &str,
    out: u32,
    x: u32,
    gate_w: u32,
    up_w: u32,
    down_w: u32,
    inter: u32,
    t: u32,
    seq_rows: bool,
    n_cu: u32,
    tp: &mut K3Tp,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    let a = act_prefix;
    // gate/up are column-parallel over the intermediate; down is row-parallel
    // over it, so its output is a partial sum until the all-reduce below.
    let inter = tp.local(inter);
    let sz = t as u64 * inter as u64 * 2;
    let g = b.tensor(&format!("{a}mlp_gate"), sz);
    let u = b.tensor(&format!("{a}mlp_up"), sz);
    let act = b.tensor(&format!("{a}mlp_act"), sz);
    let _ = &all;
    let gemv = |b: &mut Builder, out: u32, row: u32, wt: u32, n: u32, k: u32, dep: &[u32]| {
        emit_k3_linear(b, out, row, wt, t, n, k, n_cu, seq_rows, dep)
    };
    let c_g = gemv(b, g, x, gate_w, inter, cb.hidden, deps);
    let c_u = gemv(b, u, x, up_w, inter, cb.hidden, deps);
    let c_a = emit_situ_glu(b, cb, act, g, u, t * inter, n_cu, &[c_g, c_u]);
    let dst = if tp.on() { tp.dg } else { out };
    let c_d = gemv(b, dst, act, down_w, cb.hidden, inter, &[c_a]);
    if tp.on() {
        return crate::emit_xreduce(
            b,
            &mut tp.xgate,
            t == 1,
            &tp.xr_cus,
            c_d,
            out,
            t * cb.hidden,
            tp.tp,
            tp.slot_b,
        );
    }
    c_d
}

/// Everything one K3 KDA layer binds, beyond the mixer's own 14 tensors.
pub struct K3LayerWeights<'a> {
    pub mixer: K3Mixer<'a>,
    pub ffn: K3Ffn<'a>,
    /// `post_attention_layernorm.weight`.
    pub post_ln: u32,
    /// The two FOLDED AttnRes score weights — `norm.weight * proj.weight`, f32
    /// `[hidden]`. They are DIFFERENT tensors: `self_attention_res_*` on the
    /// attention side, `mlp_res_*` on the MLP side.
    pub sa_score_w: u32,
    pub mlp_score_w: u32,
}

/// Emit one complete K3 **KDA** layer: both AttnRes mixes, the KDA mixer, the
/// LatentMoE, and the block output. Returns the counter of the final add.
///
/// This is the composition the tree did not have — `kda.rs` emitted a mixer and
/// `k3.rs` emitted single packets, and nothing called them together.
///
/// The residual structure is the part that is not a normal block:
///
/// * `prefix_in` is the running PREFIX SUM, not the layer input.
/// * At a snapshot layer (`l % attn_res_block_size == 0`) the prefix RESTARTS at
///   the mixer output — no add at all — and the caller must push `prefix_in`
///   onto the `blkres` ring BEFORE calling. Layer 0 is a snapshot layer with an
///   empty ring, so both mixes are skipped.
/// * Both AttnRes mixes read the SAME ring but with different folded weights.
///
/// A gate on the block output alone is blind to almost all of this: at a
/// non-snapshot layer the output equals `prefix_in + attn + moe`, which is a
/// plain residual to 3.0e-3, while the two AttnRes outputs differ from the
/// plain wiring by 8.1e-1 and 7.7e-1. Check `h_a` and `h2`.
#[allow(clippy::too_many_arguments)]
pub fn emit_k3_block(
    b: &mut Builder,
    cb: &K3BlockCfg,
    cm: &K3MoeCfg,
    w: &K3LayerWeights,
    act_prefix: &str,
    layer: u32,
    ctx: u32,
    out: u32,
    prefix_in: u32,
    blkres: u32,
    // The snapshot ring's ALLOCATED row count — `blkres` is `[t][nb_cap][hidden]`. A property of
    // the MODEL (`max over layers of blocks_at(l) + snapshots(l)`), not of this layer, and the
    // stride the kernel walks. See [`emit_attn_res`].
    nb_cap: u32,
    t: u32,
    n_cu: u32,
    tp: &mut K3Tp,
    deps: &[u32],
    // `true` when this block's `t` rows are INDEPENDENT SEQUENCES (batched decode) rather
    // than consecutive tokens of one. Only the KDA mixer cares: its recurrence and its conv
    // window carry state between rows, and sharing that across independent sequences is silent.
    seq_rows: bool,
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    let a = act_prefix;
    let hid = cb.hidden;
    // TWO ring counts, not one. The reference pushes the snapshot BETWEEN the two
    // mixes, so at a snapshot layer the MLP-side mix sees one more row than the
    // attention-side mix did. `nb_out == blocks_at(layer + 1)`, which is what
    // makes the next layer's entry count agree.
    let nb_in = cb.blocks_at(layer);
    let nb_out = nb_in + u32::from(cb.snapshots(layer));
    let bft = |b: &mut Builder, n: String| b.tensor(&n, t as u64 * hid as u64 * 2);

    // A1 — the attention-side mix, and the snapshot push that rides with it.
    //
    // Emitted even at nb_in == 0 when this layer snapshots (layer 0), where the
    // mix is an exact copy: the packet is there for the PUSH. Skipping it would
    // leave the ring holding whatever it was allocated with, and every later
    // AttnRes would mix against it — no fault, no missing weight, just wrong.
    // The mixer's own PRE-NORM is the sole consumer of this mix, so the mix absorbs it and the
    // mixer is told to skip it. See [`fuse_attnres_norm`]; `eps` is asserted equal rather than
    // assumed, because the fused op has one `f[0]` and uses it for both reductions.
    let (mixer_ln, mixer_eps) = match &w.mixer {
        K3Mixer::Kda { cfg, w: kw, .. } => (kw.ln_w, cfg.eps),
        K3Mixer::Mla { cfg, w: mw } => (mw.ln_w, cfg.eps),
    };
    let push = cb.snapshots(layer).then_some(prefix_in);
    let fuse_pre = fuse_attnres_norm() && (nb_in > 0 || push.is_some()) && mixer_eps == cb.eps;
    let (h_a, mut dep) = if nb_in > 0 || push.is_some() {
        let h = bft(b, format!("{a}h_a"));
        let c = emit_attn_res(
            b,
            cb,
            h,
            prefix_in,
            blkres,
            w.sa_score_w,
            t,
            nb_in,
            nb_cap,
            n_cu,
            push,
            fuse_pre.then_some(mixer_ln),
            deps,
        );
        (h, c)
    } else {
        (prefix_in, deps[0])
    };

    // Under TP, `o_proj` is row-parallel: the mixer writes the peer-visible
    // partial slot and the all-reduce below produces the real attention output.
    let attn_dst = if tp.on() { Some(tp.og) } else { None };
    let (c_o, attn_partial) = match &w.mixer {
        K3Mixer::Kda { cfg, w: kw, st } => crate::kda::emit_kda_mixer(
            b,
            cfg,
            kw,
            st,
            a,
            t,
            h_a,
            attn_dst,
            n_cu,
            fuse_pre,
            &[dep],
            seq_rows,
        ),
        K3Mixer::Mla { cfg, w: mw } => emit_k3_mla_mixer(
            b,
            cfg,
            mw,
            a,
            t,
            ctx,
            h_a,
            attn_dst,
            n_cu,
            fuse_pre,
            &[dep],
            seq_rows,
        ),
    };
    let (c_o, attn) = if tp.on() {
        let full = bft(b, format!("{a}attn_full"));
        (
            crate::emit_xreduce(
                b,
                &mut tp.xgate,
                t == 1,
                &tp.xr_cus,
                c_o,
                full,
                t * hid,
                tp.tp,
                0,
            ),
            full,
        )
    } else {
        (c_o, attn_partial)
    };

    // The prefix accumulate — or, at a snapshot layer, the RESTART.
    let prefix = if cb.snapshots(layer) {
        dep = c_o;
        attn
    } else {
        let p = bft(b, format!("{a}prefix"));
        dep = b.emit(DevOp::Residual, vec8_cus(&all, t * hid), &[c_o], |d| {
            d.t[0] = p;
            d.t[1] = prefix_in;
            d.t[2] = attn;
            d.i[0] = t * hid;
            d.f[0] = 1.0;
        });
        p
    };

    // A2 — the MLP-side mix, with the OTHER fold, at the POST-push count. `post_attention_layernorm`
    // is its sole consumer, so the mix absorbs it too (see [`fuse_attnres_norm`]) and `h2` IS the
    // normed activation. At `nb_out == 0` there is no mix to fuse into and the norm stays its own
    // packet.
    let fuse_post = fuse_attnres_norm() && nb_out > 0;
    let h2 = if nb_out > 0 {
        let h = bft(b, format!("{a}h2"));
        dep = emit_attn_res(
            b,
            cb,
            h,
            prefix,
            blkres,
            w.mlp_score_w,
            t,
            nb_out,
            nb_cap,
            n_cu,
            None,
            fuse_post.then_some(w.post_ln),
            &[dep],
        );
        h
    } else {
        prefix
    };

    let h3 = if fuse_post {
        h2
    } else {
        let h3 = bft(b, format!("{a}h3"));
        dep = b.emit(DevOp::RmsNorm, norm_cus(&all, t), &[dep], |d| {
            d.t[0] = h3;
            d.t[1] = h2;
            d.t[2] = w.post_ln;
            d.i[0] = t;
            d.i[1] = hid;
            d.f[0] = cb.eps;
        });
        h3
    };

    // The block-output residual, folded into the MoE tail's own add when there is one to fold into
    // (see [`fuse_block_resid`]). `out` then IS the layer output and `{a}ffn` never exists.
    let fuse_bo = fuse_block_resid() && matches!(w.ffn, K3Ffn::Latent(_));
    let ffn = if fuse_bo {
        out
    } else {
        bft(b, format!("{a}ffn"))
    };
    let c_ffn = match &w.ffn {
        K3Ffn::Latent(mw) => emit_k3_latent_moe(
            b,
            cb,
            cm,
            mw,
            &format!("{a}moe."),
            ffn,
            h3,
            t,
            seq_rows,
            n_cu,
            tp,
            fuse_bo.then_some(prefix),
            &[dep],
        ),
        K3Ffn::Dense {
            gate,
            up,
            down,
            inter,
        } => emit_k3_dense_mlp(
            b,
            cb,
            a,
            ffn,
            h3,
            *gate,
            *up,
            *down,
            *inter,
            t,
            seq_rows,
            n_cu,
            tp,
            &[dep],
        ),
    };
    if fuse_bo {
        c_ffn
    } else {
        emit_k3_block_out(b, out, prefix, ffn, t * hid, n_cu, &[c_ffn])
    }
}

/// Everything `emit_k3_model` needs that is not per-layer.
pub struct K3ModelCfg {
    pub block: K3BlockCfg,
    pub kda: crate::kda::KdaCfg,
    pub mla: K3MlaCfg,
    pub moe: K3MoeCfg,
    pub vocab: u32,
    /// `first_k_dense_replace` — layers below this carry a dense FFN.
    pub first_k_dense: u32,
    /// `intermediate_size`, the DENSE FFN width.
    pub dense_inter: u32,
    /// Checkpoint prefix. K3 nests its text tower under a multimodal wrapper,
    /// so this is `language_model.model.` — of the checkpoint's 497 220
    /// tensors, ZERO start with `model.`.
    pub prefix: String,
    /// Tensor-parallel degree (`plowc --num-gpus`). 1 emits no collectives.
    pub tp: u32,
}

/// Emit a truncated K3 decode program: embed, `layers.len()` KDA layers, tail.
///
/// **This is the layer LOOP the tree did not have.** It is deliberately not yet
/// `glm_emit_full`'s equal — it refuses MLA layers, because the K3 MLA block
/// (output gate + NoPE + no built-in residual) is not written. Layers 0..2 of
/// the shipped config are all KDA, so a 3-layer truncation is emittable today
/// and exercises everything structural: the snapshot ring, the layer-0 special
/// case, the prefix accumulate, and both FFN kinds.
///
/// `layers` is the 0-based list to emit, so truncation shrinks the tensor table
/// too. This is the `GLM_NLAYERS` role: a truncated model loads in seconds
/// instead of paying a full-checkpoint load per iteration.
///
/// # The snapshot ring
///
/// `blkres` holds up to `max_blocks` H-wide rows. A snapshot layer pushes its
/// INPUT (the prefix on entry) and restarts the prefix. The push is a host-side
/// concern in this emit — the ring rows are declared here and written by the
/// block's own AttnRes reads — which is the piece a full emit must still make
/// explicit per layer.
/// How the `t` rows of a program relate to each other. **This is not cosmetic**: a KDA layer's
/// recurrence threads its rows through a carried state, and the two answers need opposite
/// behaviour from the same kernel.
///
/// * [`RowKind::Tokens`] — `t` CONSECUTIVE TOKENS OF ONE SEQUENCE (every program today, prefill
///   and single-stream decode alike). The recurrence threads them through ONE state and the conv
///   window is loaded once, rolled across all of them, and stored once. That sharing IS the
///   carried state.
/// * [`RowKind::Sequences`] — `t` INDEPENDENT SEQUENCES, one token each (batched decode). Each
///   row owns its own state and its own conv window.
///
/// Sharing a state across the second kind runs sequence 1's token into sequence 0's and produces
/// fluent, plausible, WRONG output — no crash, no NaN — which is why the distinction is a type
/// and not a `bool` parameter that reads as `false` at the call site.
///
/// See `perf-data/archive/k3/k3-batched-decode-design.md`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowKind {
    /// Consecutive tokens of one sequence.
    Tokens,
    /// Independent sequences, one token each.
    Sequences,
}

#[allow(clippy::too_many_arguments)]
pub fn emit_k3_model(
    b: &mut Builder,
    c: &K3ModelCfg,
    is_kda: &dyn Fn(u32) -> bool,
    layers: &[u32],
    ctx: u32,
    t: u32,
    scratch_rows: u32,
    sequence_slots: u32,
    n_cu: u32,
    rows: RowKind,
) {
    // These are blob-wide capacities, not properties of this program. Prefill token rows need
    // transient activation/peer space; only decode rungs add independently carried sequences.
    let seq_rows = matches!(rows, RowKind::Sequences);
    assert!(
        scratch_rows >= t,
        "K3: scratch has {scratch_rows} rows but program needs {t}"
    );
    assert!(
        sequence_slots >= if seq_rows { t } else { 1 },
        "K3: carried state has {sequence_slots} slots but program needs {}",
        if seq_rows { t } else { 1 }
    );
    let slots = sequence_slots as u64;
    // SHARING IS EXPRESSED THROUGH THE BUILDER, not through the naming. Re-declaring a name now
    // returns the existing handle and grows it to the larger size, which is what makes the shared
    // prefill prefix below actually share one buffer instead of allocating 93 identically-named
    // ones. It is a NO-OP for decode — nothing there declares a name twice, and
    // `no_two_tensors_share_a_name_on_either_phase` is what keeps that true — so a decode blob is
    // byte-identical with and without it. It is also what lets `k3_build_model` build several
    // programs against ONE table by adopting the previous one.
    b.set_tensor_dedup(true);
    let all = b.all();
    let (hid, cb) = (c.block.hidden, &c.block);
    let pfx = c.prefix.as_str();
    let nb_max = layers
        .iter()
        .map(|&l| cb.blocks_at(l) + u32::from(cb.snapshots(l)))
        .max()
        .unwrap_or(0)
        .max(1);
    assert!(
        nb_max <= K3_ATTNRES_MAXB,
        "K3: the snapshot ring needs {nb_max} rows, past PLOW_ATTNRES_MAXB = {K3_ATTNRES_MAXB}"
    );

    // WHERE THE PER-LAYER SCRATCH LIVES, and it is not the same answer on the two phases.
    //
    // Decode keeps `act.l{l}.…`: one private set of activation buffers per layer, ~50 KiB each at
    // one token, and every existing K3 decode blob is byte-identical because of it.
    //
    // A prefill bucket CANNOT. The same naming at T = 8192 asks for `act.l{l}.moe.part` =
    // `T * top_k * latent` f32 = **1.9 GiB per layer** across 92 MoE layers, plus five [T, hidden]
    // buffers per block at 117 MiB each — tens of terabytes. So a bucket shares ONE set across the
    // layer loop, which is exactly what GLM's `GlmTn` does and is safe for the same reason: the
    // layers are strictly serialized by the counter DAG (layer l+1's first op waits on layer l's
    // block output), and every scratch buffer is written before it is read inside its own layer.
    // What stays per-layer is what must: the KDA recurrent state and the MLA `kv.{l}.*` cache rows,
    // which carry ACROSS layers and are `kv.`-prefixed, not `act.`-prefixed.
    let act_pfx = |l: u32| {
        if t == 1 {
            format!("act.l{l}.")
        } else {
            "act.pf.".to_string()
        }
    };

    // Globals.
    let ids = b.tensor("in.ids", scratch_rows as u64 * 4);
    let emb = b.tensor(
        &format!("{pfx}embed_tokens.weight"),
        c.vocab as u64 * hid as u64 * 2,
    );
    let fin = b.tensor(&format!("{pfx}norm.weight"), hid as u64 * 2);
    // lm_head SITS BESIDE `model.`, NOT INSIDE IT. The checkpoint has
    // `language_model.lm_head.weight` while every layer is under
    // `language_model.model.layers.`, so the head takes the TOWER prefix — `prefix`
    // with the trailing `model.` removed — and a bare "lm_head.weight" (what this
    // was) matches nothing in a K3 checkpoint and loads a zeroed head, which
    // samples token 0 from all-zero logits. `crates/devgen/src/checkpoint.rs:405`
    // already records the real spelling.
    let tower = pfx.strip_suffix("model.").unwrap_or(pfx);
    // Declared size IS the sharding request: `asset::shard::slice_for` promotes lm_head to Column
    // purely from `want == full / tp`, so this one line is the whole host-side change.
    let vocab_l = k3_vocab_l(c);
    let head = b.tensor(
        &format!("{tower}lm_head.weight"),
        vocab_l as u64 * hid as u64 * 2,
    );
    let x = b.tensor("act.x", scratch_rows as u64 * hid as u64 * 2);
    let xnext = b.tensor("act.xnext", scratch_rows as u64 * hid as u64 * 2);
    let xn = b.tensor("act.xn", scratch_rows as u64 * hid as u64 * 2);
    // ROWS THE TAIL SAMPLES. A prefill bucket samples ONE row (the last real one) however wide
    // the bucket is — a [T, vocab] logit matrix would cost a 163k-wide GEMM per prompt token to
    // throw all but one row away. A batched DECODE samples ALL of them, because its rows are
    // independent sequences and each one needs its own token.
    let s_rows = if seq_rows { t } else { 1 };
    let logits = b.tensor("act.logits", sequence_slots as u64 * vocab_l as u64 * 2);
    let amax = b.tensor(
        "act.amax",
        sequence_slots as u64 * crate::AMAX_BLOCKS as u64 * 8,
    );
    // The block-residual ring: compiler-owned, per-sequence, `kv.`-prefixed so
    // the loader classifies it by exclusion rather than demanding it of the
    // checkpoint.
    //
    // `[t][nb_max][hidden]`, and the t axis is the one that did not exist. A decode ring is one
    // token's worth of rows; a prefill bucket needs a private ring PER TOKEN, because every token
    // has its own prefix sum and its own snapshots. `nb_max` is the stride and travels to the
    // kernel as an operand — see [`emit_attn_res`].
    let blkres = b.tensor(
        "kv.blkres",
        scratch_rows as u64 * nb_max as u64 * hid as u64 * 2,
    );

    // Tensor parallelism. The two peer slots are the partial buffers the collectives reduce out
    // of: one for the attention output, one for the expert combine.  `slot_b` MUST be identical
    // in every program sharing a blob.  The host binds `act.dg_tp` at the blob-wide maximum slot
    // offset; deriving it from this program's `t` made decode read at `H*2` while the host placed
    // the partial after the widest prefill bucket (`Tmax*H*2`).
    let mut tp = if c.tp > 1 {
        assert!(
            scratch_rows >= t,
            "K3: peer slot has {scratch_rows} rows but program needs {t}"
        );
        let slot_b = scratch_rows * hid * 2;
        K3Tp {
            tp: c.tp,
            og: b.tensor("act.og_tp", slot_b as u64),
            dg: b.tensor("act.dg_tp", slot_b as u64),
            ug: b.tensor("act.ug_tp", slot_b as u64),
            slot_b,
            xgate: 0,
            xr_cus: (0..crate::emit_config::active()
                .xr_cus
                .unwrap_or(n_cu)
                .clamp(1, n_cu))
                .collect(),
        }
    } else {
        K3Tp::none()
    };
    // The head axis is what TP cuts on both mixers. K3 at tp8 gives nh_l = 12,
    // and FlashMLA's group factor must divide it — 8 does not, 4 does.
    let mut kda_l = c.kda;
    kda_l.heads = tp.local(c.kda.heads);
    let mut mla_l = c.mla;
    mla_l.heads = tp.local(c.mla.heads);
    assert!(
        mla_l.heads > 1 || c.tp == 1,
        "K3: nh_l = 1 at --num-gpus {}; no instantiated FlashMLA group factor can express a \
         single local head",
        c.tp
    );
    assert_eq!(
        mla_l.heads % mla_l.gf,
        0,
        "K3: FlashMLA group factor {} must divide the LOCAL head count {}",
        mla_l.gf,
        mla_l.heads
    );
    // The NoPE identity table: cos=1, sin=0, both exact in bf16. Declared even
    // though K3 has no rotation, because both `HeadNormRope` ops stay emitted —
    // see `emit_k3_mla_mixer`.
    let [cos_t, sin_t] = k3_nope_rope_pair(ctx, c.mla.qk_rope);
    let cos = b.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = b.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    // `in.pos` is CTX-sized, not `t`-sized, and that is not slack: the runtime recovers
    // `max_ctx` as `bytes_of("in.pos") / 4` (`plowrt::exec::amd:2399`). Sizing it by `t` made a
    // blob compiled `--max-ctx 32768` report `max_ctx = 8192` — the widest prefill bucket, because
    // tensor dedup keeps the largest declaration — and every decode step past 8192 was refused.
    // Every sibling emitter already declares it at `ctx` (`lib.rs:1016`, `mla.rs:1087`, `:5175`).
    let pos = b.tensor("in.pos", ctx as u64 * 4);
    // ONE kvlen PER SLOT. The runtime derives the decode batch from this tensor's SIZE
    // (`exec/amd.rs`: `batch = in.kvlen.bytes / 4`) and refuses a blob whose decode program's `t`
    // disagrees, so this is not merely storage -- it is how the host is told how many independent
    // sequences the program carries. `in.pos` needs no change: it is already `ctx * 4`, which has
    // room for `slots` positions and is separately the tensor `max_ctx` is read from.
    let kvlen = b.tensor("in.kvlen", 4 * slots);

    let mut dep = b.emit(DevOp::Embed, all.clone(), &[], |d| {
        d.t[0] = x;
        d.t[1] = emb;
        d.t[2] = ids;
        d.i[0] = t;
        d.i[1] = hid;
        d.f[0] = 1.0; // K3 has no embedding scale.
    });

    let mut cur = x;
    for &l in layers {
        let nxt = if cur == x { xnext } else { x };
        let lp = format!("{pfx}layers.{l}.");
        // Per-layer mixer, chosen from the config LIST. A KDA layer binds
        // `q_proj`; an MLA layer binds `q_a_proj`. Getting the partition wrong
        // mis-binds 69 of 93 layers, which is why it is never a stride.
        let kda_pair;
        let mla_w;
        let mixer = if is_kda(l) {
            let state = if crate::emit_config::active().k3_kda_conv_step_db {
                crate::kda::declare_kda_state_db(b, &kda_l, &format!("kv.{l}."), slots, pos)
            } else {
                crate::kda::declare_kda_state(b, &kda_l, &format!("kv.{l}."), slots)
            };
            kda_pair = Some((
                crate::kda::declare_kda_weights(b, &kda_l, &format!("{lp}self_attn."), &lp),
                state,
            ));
            let (kw, ks) = kda_pair.as_ref().unwrap();
            K3Mixer::Kda {
                cfg: &kda_l,
                w: kw,
                st: ks,
            }
        } else {
            mla_w = Some(declare_k3_mla_weights(
                b, &mla_l, &lp, l, ctx, cos, sin, pos, kvlen, slots,
            ));
            K3Mixer::Mla {
                cfg: &mla_l,
                w: mla_w.as_ref().unwrap(),
            }
        };
        let mw;
        let ffn = if l < c.first_k_dense {
            // Dense gate/up are column-parallel and down is row-parallel.  All three checkpoint
            // matrices therefore contribute exactly 1/tp of their bytes to this rank:
            //
            //   gate/up [I/tp, H]     down [H, I/tp]
            //
            // The GEMVs below have always used `tp.local(I)`.  Declaring the weights at the full
            // `I` made `slice_for` interpret `want == full` as an explicit request to replicate
            // the matrix. The gate/up ranks then all read rank 0's rows, and down walked the
            // full-stride `[H,I]` allocation as `[H,I/tp]`: bounded memory accesses, but the wrong
            // matrices on every rank.
            let inter_l = tp.local(c.dense_inter);
            K3Ffn::Dense {
                gate: b.tensor(
                    &format!("{lp}mlp.gate_proj.weight"),
                    inter_l as u64 * hid as u64 * 2,
                ),
                up: b.tensor(
                    &format!("{lp}mlp.up_proj.weight"),
                    inter_l as u64 * hid as u64 * 2,
                ),
                down: b.tensor(
                    &format!("{lp}mlp.down_proj.weight"),
                    hid as u64 * inter_l as u64 * 2,
                ),
                inter: c.dense_inter,
            }
        } else {
            mw = declare_k3_moe_weights(b, &c.moe, &lp, c.tp);
            K3Ffn::Latent(&mw)
        };
        let w = K3LayerWeights {
            mixer,
            ffn,
            post_ln: b.tensor(
                &format!("{lp}post_attention_layernorm.weight"),
                hid as u64 * 2,
            ),
            sa_score_w: b.tensor(
                &format!("{lp}self_attention_res_score.weight"),
                hid as u64 * 4,
            ),
            mlp_score_w: b.tensor(&format!("{lp}mlp_res_score.weight"), hid as u64 * 4),
        };
        dep = emit_k3_block(
            b,
            cb,
            &c.moe,
            &w,
            &act_pfx(l),
            l,
            ctx,
            nxt,
            cur,
            blkres,
            nb_max,
            t,
            n_cu,
            &mut tp,
            &[dep],
            rows == RowKind::Sequences,
        );
        cur = nxt;
    }

    // THE MODEL-LEVEL AttnRes — the THIRD site, and it is not optional.
    //
    // `KimiLinearModel.forward` ends with `_apply_output_attn_res(hidden_states, block_residual)`
    // BEFORE `self.norm` (`modeling_kimi_linear.py:1213`, `:1226`), using the model's own
    // `output_attn_res_norm` [H] / `output_attn_res_proj` [1, H] pair — both present in the
    // checkpoint (`language_model.model.output_attn_res_{norm,proj}.weight`) and both folded to
    // one f32 [H] by `plowrt`'s generic `fold_res_score`, which keys on the `_res_score.weight`
    // SUFFIX and therefore already handles this pair with no host change.
    //
    // WHY ITS ABSENCE WAS A DEPTH-DEPENDENT COLLAPSE rather than a small error. The prefix sum
    // RESTARTS at every snapshot layer, so what the layer loop leaves in `cur` is NOT the model's
    // residual stream — it is only the partial sum since the LAST snapshot. At 93 layers the last
    // snapshot is layer 84, so `cur` carries layers 84..92 and nothing else: no embedding, and
    // none of the seven earlier blocks. The output mix is the only thing that puts them back.
    // At `K3_NLAYERS <= 12` the sole snapshot is layer 0's (the embedding) and the missing mix is
    // a plausible-looking perturbation — varied tokens, wrong model; at 93 it is a hidden state
    // with 90% of the network missing from it, which greedily decodes one constant token forever.
    //
    // `nb_fin` is the ring's live count after the loop, which is the last layer's POST-push count
    // — `blocks_at(l) + snapshots(l)` — the same expression `nb_max` maximises over.
    let nb_fin = layers
        .last()
        .map(|&l| cb.blocks_at(l) + u32::from(cb.snapshots(l)))
        .unwrap_or(0);
    // `self.norm` is this mix's sole consumer, exactly as the two per-layer norms are of theirs,
    // so it absorbs it under the same knob ([`fuse_attnres_norm`]) and `xn` IS the normed row.
    let fuse_fin = fuse_attnres_norm() && nb_fin > 0;
    if nb_fin > 0 {
        let ores = b.tensor(
            &format!("{pfx}output_attn_res_score.weight"),
            hid as u64 * 4,
        );
        let dst = if fuse_fin {
            xn
        } else {
            b.tensor("act.xres", t as u64 * hid as u64 * 2)
        };
        dep = emit_attn_res(
            b,
            cb,
            dst,
            cur,
            blkres,
            ores,
            t,
            nb_fin,
            nb_max,
            n_cu,
            None,
            fuse_fin.then_some(fin),
            &[dep],
        );
        cur = dst;
    }

    // Tail: final norm, lm_head, argmax.
    let c_f = if fuse_fin {
        dep
    } else {
        b.emit(DevOp::RmsNorm, vec![0u32], &[dep], |d| {
            d.t[0] = xn;
            d.t[1] = cur;
            d.t[2] = fin;
            d.i[0] = t;
            d.i[1] = hid;
            d.f[0] = cb.eps;
        })
    };
    let c_lm = b.emit(DevOp::Gemv, all, &[c_f], |d| {
        d.t[0] = logits;
        d.t[1] = xn;
        d.t[2] = head;
        d.i[0] = s_rows;
        d.i[1] = vocab_l;
        d.i[2] = hid;
        // a_row0: the last real row on a prefill bucket, row 0 on a batched decode (which wants
        // every row, so it starts at the first). The host re-patches this per prefill chunk.
        d.i[4] = if seq_rows { 0 } else { t - 1 };
    });
    let c_am = b.emit(
        DevOp::Argmax,
        (0..crate::AMAX_BLOCKS).collect(),
        &[c_lm],
        |d| {
            d.t[0] = amax;
            d.t[1] = logits;
            d.i[0] = vocab_l;
            // `d_argmax`'s n_batch, which it has always taken and no emitter has ever set:
            // "i1 = n_batch (0/1 => single sequence, byte-identical)" (runtime/amd/interp.hip:1299).
            d.i[1] = if seq_rows { t } else { 0 };
        },
    );
    if k3_shard_head(c) {
        // XArgmaxFin SUBSUMES ArgmaxFin — it folds the AMAX_BLOCKS partials itself, rebases the
        // winning index by `rank * vocab_l` and takes the cross-rank max. Emitting both would fold
        // twice and write the LOCAL winner's id first. One arrival gate plus peer-visible value
        // lines, distinct because one is an atomic counter and the others are data. Each
        // 128-byte line carries sixteen u64 winners.
        let value_lines = packet::devbuild::xargmax_value_lines(s_rows).unwrap_or_else(|| {
            panic!(
                "XArgmaxFin carries at most {} sequences, got {s_rows}",
                packet::devbuild::XARGMAX_MAX_BATCH
            )
        });
        let gate = tp.xgate;
        tp.xgate = tp
            .xgate
            .checked_add(1 + value_lines)
            .expect("XArgmaxFin counter id overflow");
        b.emit(DevOp::XArgmaxFin, vec![0u32], &[c_am], |d| {
            d.t[0] = ids;
            d.t[1] = amax;
            d.i[0] = crate::AMAX_BLOCKS;
            // n_batch. The fold publishes one u64 per sequence into consecutive 128-byte peer
            // lines, matching PLOW_XAMAX_MAX_BATCH in op_collective.h.
            d.i[1] = s_rows;
            d.i[2] = vocab_l;
            d.i[3] = gate;
            d.i[4] = gate + 1;
        });
    } else {
        b.emit(DevOp::ArgmaxFin, vec![0u32], &[c_am], |d| {
            d.t[0] = ids;
            d.t[1] = amax;
            d.i[0] = crate::AMAX_BLOCKS;
            d.i[1] = if seq_rows { t } else { 0 }; // n_batch, as on Argmax above
        });
    }
}

/// Declare one MLA layer's weights and its two KV cache rows.
///
/// The `derived.*` trio are ABSORBED weights the prep writes, not raw checkpoint
/// tensors — and they are exactly the three MLA names `plowrt`'s shard
/// classifier already knows as column-parallel. `derived.kv_a_latent` and
/// `derived.k_rope` are deliberately absent: the latent KV path is shared by
/// every head, so there is no head axis to cut and every rank binds it whole.
#[allow(clippy::too_many_arguments)]
pub fn declare_k3_mla_weights(
    b: &mut Builder,
    c: &K3MlaCfg,
    lp: &str,
    layer: u32,
    ctx: u32,
    cos: u32,
    sin: u32,
    pos: u32,
    kvlen: u32,
    // Blob-wide INDEPENDENT-SEQUENCE capacity. Prefill uses slot 0; batched decode indexes the
    // first `B` slots through `d_flash_mla_decode`'s `n_batch` axis.
    slots: u64,
) -> K3MlaWeights {
    let (h, nh) = (c.hidden as u64, c.heads as u64);
    let (dk, dr, vd, ql) = (
        c.kv_lora as u64,
        c.qk_rope as u64,
        c.v_head as u64,
        c.q_lora as u64,
    );
    K3MlaWeights {
        ln_w: b.tensor(&format!("{lp}input_layernorm.weight"), h * 2),
        q_a: b.tensor(&format!("{lp}self_attn.q_a_proj.weight"), ql * h * 2),
        q_a_norm: b.tensor(&format!("{lp}self_attn.q_a_layernorm.weight"), ql * 2),
        q_absorb: b.tensor(
            &format!("{lp}self_attn.derived.q_absorb.weight"),
            nh * dk * ql * 2,
        ),
        q_rope: b.tensor(
            &format!("{lp}self_attn.derived.q_rope.weight"),
            nh * dr * ql * 2,
        ),
        q_b: if emit_config::active().mla_materialized_prefill
            && c.qk_nope + c.qk_rope == 192
            && c.v_head == 128
            && !c.fp8_kv
        {
            b.tensor(
                &format!("{lp}self_attn.q_b_proj.weight"),
                nh * (c.qk_nope + c.qk_rope) as u64 * ql * 2,
            )
        } else {
            TENSOR_NONE
        },
        kv_b: if emit_config::active().mla_materialized_prefill
            && c.qk_nope + c.qk_rope == 192
            && c.v_head == 128
            && !c.fp8_kv
        {
            b.tensor(
                &format!("{lp}self_attn.kv_b_proj.weight"),
                nh * (c.qk_nope + c.v_head) as u64 * dk * 2,
            )
        } else {
            TENSOR_NONE
        },
        kv_a: b.tensor(
            &format!("{lp}self_attn.kv_a_proj_with_mqa.weight"),
            dk * h * 2,
        ),
        k_rope_d: b.tensor(
            &format!("{lp}self_attn.derived.k_rope_down.weight"),
            dr * h * 2,
        ),
        kv_a_norm: b.tensor(&format!("{lp}self_attn.kv_a_layernorm.weight"), dk * 2),
        v_absorb: b.tensor(
            &format!("{lp}self_attn.derived.v_absorb.weight"),
            nh * dk * vd * 2,
        ),
        // THE CHECKPOINT CALLS THE MLA OUTPUT GATE `g_proj`, not `o_gate_proj` —
        // the same spelling the KDA layers use for their gate, which is consistent
        // once you notice both are "the gate on this mixer's output". Verified
        // against the index: an MLA layer ships exactly
        // {q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm,
        //  kv_b_proj, o_proj, g_proj} and no `o_gate_proj` anywhere in 497,220
        // tensors. There is no collision: a layer is KDA or MLA, never both.
        o_gate: b.tensor(&format!("{lp}self_attn.g_proj.weight"), nh * vd * h * 2),
        o_proj: b.tensor(&format!("{lp}self_attn.o_proj.weight"), h * nh * vd * 2),
        cos,
        sin,
        pos,
        // The two cache rows. `kv.`-prefixed so the loader classifies them by
        // exclusion, and so its `kv_row_writer` scan finds the two instructions
        // that write them.
        ckv: b.tensor(
            &format!("kv.{layer}.ckv"),
            slots * ctx as u64 * dk * if c.fp8_kv { 1 } else { 2 },
        ),
        krot: b.tensor(&format!("kv.{layer}.krot"), slots * ctx as u64 * dr * 2),
        kv_scale: if c.fp8_kv {
            b.tensor(&format!("kv.{layer}.scale"), slots * ctx as u64 * 4)
        } else {
            packet::dev::TENSOR_NONE
        },
        kvlen,
    }
}

/// Declare one layer's LatentMoE tensors under `lp` (`…layers.{l}.`).
///
/// The expert weights themselves are NOT declared: they are bound by name
/// pattern into the two host-filled pointer tables, as on every other MoE path.
/// The MoE sub-namespace Kimi-K3 actually ships, and it is NOT `mlp.`.
///
/// Verified against `models--moonshotai--Kimi-K3`'s own index (497,220 tensors):
/// layer 0 is the DENSE layer and owns `mlp.{gate,up,down}_proj.weight`; layers
/// 1..=92 are MoE and put everything under `block_sparse_moe.` —
///
/// ```text
/// block_sparse_moe.gate.weight                       92 (one per MoE layer)
/// block_sparse_moe.gate.e_score_correction_bias      92
/// block_sparse_moe.routed_expert_{down,up}_proj      92 each
/// block_sparse_moe.routed_expert_norm.weight         92
/// block_sparse_moe.shared_experts.*_proj.weight     276 (92 x 3)
/// block_sparse_moe.experts.{e}.w{1,2,3}.*       494,592
/// ```
///
/// Declaring these under `mlp.` did not fail loudly: `mlp.gate.weight` collides
/// with nothing, so the loader simply reported five MISSING WEIGHT tensors per
/// layer and matmuls against a zeroed buffer. The dense layer keeps `mlp.`, which
/// is why the two namespaces have to be spelled separately rather than unified.
const K3_MOE_NS: &str = "block_sparse_moe.";

pub fn declare_k3_moe_weights(b: &mut Builder, c: &K3MoeCfg, lp: &str, tp: u32) -> K3MoeWeights {
    let (h, lat) = (c.hidden as u64, c.latent as u64);
    // Same rule as `K3Tp::local`: an exact division, because every K3 width divides 8.
    let tp_local = |n: u32| (n / tp.max(1)) as u64;
    let n = |s: &str| format!("{lp}{K3_MOE_NS}{s}");
    if c.enc == K3_MOE_ENC_MXFP4
        && tp_local(c.moe_inter) == 384
        && c.latent.is_multiple_of(16)
        && crate::emit_config::active().moe_stage2_lean
    {
        b.tensor(
            &format!("moe.{lp}expert_weight_table_moe2"),
            c.n_exp as u64 * 3 * 8,
        );
        b.tensor(
            &format!("moe.{lp}expert_scale_table_moe2"),
            c.n_exp as u64 * 3 * 8,
        );
    }
    K3MoeWeights {
        router: b.tensor(&n("gate.weight"), c.n_exp as u64 * h * 2),
        router_bias: b.tensor(&n("gate.e_score_correction_bias"), c.n_exp as u64 * 4),
        down_latent: b.tensor(&n("routed_expert_down_proj.weight"), lat * h * 2),
        // bf16, NOT f32. It is an ordinary RMSNorm gain and reaches the SAME
        // `DevOp::RmsNorm` arm as `post_attention_layernorm.weight`, which is
        // declared `hidden * 2` two fields down — one op cannot read two dtypes for
        // the same operand. The checkpoint agrees: 7168 B for [3584], i.e. 2 B each.
        // Declared at `lat * 4` the load failed with "the checkpoint has 7168 B and
        // the blob declares 14336" — the good outcome; had the sizes happened to
        // line up, the norm would have read pairs of bf16 as one f32.
        latent_norm: b.tensor(&n("routed_expert_norm.weight"), lat * 2),
        // COLUMN-PARALLEL, and the declared size is the whole request: `slice_for` reads
        // `up_proj.weight` out of this name, sees `want == full / tp`, and hands rank r the
        // contiguous row range `[r*H/tp, (r+1)*H/tp)` of [hidden, latent] — a borrow of the
        // mmap, no gather. Declared at FULL width it stays REPLICATED (that same function
        // demotes `full == want` back to Replicated), which is the tp=1 emit and was the
        // TP8 one: 92 layers x 7168 x 3584 x 2 B = **4.73 GB streamed per rank per token**,
        // ~0.74 ms at the 6.4 TB/s ceiling, seven eighths of it redundant.
        //
        // The matching gather lives in `emit_k3_latent_moe`'s tail collective; see there
        // for why COLUMN is the only axis available and why the gather is free.
        up_latent: b.tensor(
            &n("routed_expert_up_proj.weight"),
            if shard_up_proj(tp) && !up_gather_only() {
                tp_local(c.hidden)
            } else {
                c.hidden as u64
            } * lat
                * 2,
        ),
        expert_w_table: b.tensor(
            &format!("moe.{lp}expert_weight_table"),
            c.n_exp as u64 * 3 * 8,
        ),
        expert_s_table: b.tensor(
            &format!("moe.{lp}expert_scale_table"),
            c.n_exp as u64 * 3 * 8,
        ),
        // SHARDED, and this was the single largest line item in the decode stream. `shared_inter`
        // is `n_shared_experts * moe_inter` = 2 * 3072 = 6144, so the three matrices are
        // 3 * 6144 * 7168 * 2 B = 264 MB per layer. Declared at FULL width they were REPLICATED,
        // and 92 MoE layers x 264 MB is **24.3 GB streamed per rank per token** — eight times the
        // whole routed-expert stream (3.0 GB at top-16 mxfp4) and ~3.8 ms at the 6.4 TB/s measured
        // streaming ceiling, paid identically and redundantly on all eight ranks.
        //
        // gate/up are column-parallel off the full `x`; `down` is row-parallel over the rank's OWN
        // `sha`, so no input offset is needed anywhere. `slice_for` infers the axis from the name
        // once `want == full/tp` (`gate_proj`/`up_proj` -> Column, `down_proj` -> Row), so the
        // loader needs no change. `down`'s output is a hidden-width PARTIAL and is reduced below.
        shared_gate: b.tensor(
            &n("shared_experts.gate_proj.weight"),
            tp_local(c.shared_inter) * h * 2,
        ),
        shared_up: b.tensor(
            &n("shared_experts.up_proj.weight"),
            tp_local(c.shared_inter) * h * 2,
        ),
        shared_down: b.tensor(
            &n("shared_experts.down_proj.weight"),
            h * tp_local(c.shared_inter) * 2,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3() -> K3BlockCfg {
        K3BlockCfg {
            hidden: 7168,
            eps: 1e-5,
            attn_res_block_size: 12,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
        }
    }

    fn k3_moe() -> K3MoeCfg {
        K3MoeCfg {
            hidden: 7168,
            latent: 3584,
            moe_inter: 3072,
            shared_inter: 6144,
            n_exp: 896,
            top_k: 16,
            route_flags: 3,
            route_scale: 2.5,
            n_group: 1,
            topk_group: 1,
            enc: 2, // mxfp4
            group_decode: false,
        }
    }

    /// Build the CRITICAL BLOCK — layer 1, KDA mixer + LatentMoE — and check it
    /// against the op sequence the rung-2 gate
    /// (`runtime/tests/k3_moe_block_gfx950_test.c`) hand-built and measured.
    /// That gate and this emitter are otherwise two independent transcriptions
    /// of the same graph, which is the drift this test exists to catch.
    fn build_layer1_with_group(group_decode: bool) -> packet::devbuild::Program {
        let mut cm = k3_moe();
        cm.group_decode = group_decode;
        let (cb, ck) = (
            k3(),
            crate::kda::KdaCfg {
                hidden: 7168,
                heads: 96,
                head_dim: 128,
                conv_w: 4,
                gate_lower_bound: Some(-5.0),
                eps: 1e-5,
                bv: 16,
            },
        );
        let mut b = Builder::new(256);
        let p = "language_model.model.layers.1.";
        let kw = crate::kda::declare_kda_weights(&mut b, &ck, &format!("{p}self_attn."), p);
        let ks = crate::kda::declare_kda_state(&mut b, &ck, "kv.1.", 1);
        let mw = K3MoeWeights {
            router: b.tensor(&format!("{p}mlp.gate.weight"), 896 * 7168 * 2),
            router_bias: b.tensor(&format!("{p}mlp.gate.e_score_correction_bias"), 896 * 4),
            down_latent: b.tensor(
                &format!("{p}mlp.routed_expert_down_proj.weight"),
                3584 * 7168 * 2,
            ),
            latent_norm: b.tensor(&format!("{p}mlp.routed_expert_norm.weight"), 3584 * 4),
            up_latent: b.tensor(
                &format!("{p}mlp.routed_expert_up_proj.weight"),
                7168 * 3584 * 2,
            ),
            expert_w_table: b.tensor("moe.expert_weight_table", 896 * 3 * 8),
            expert_s_table: b.tensor("moe.expert_scale_table", 896 * 3 * 8),
            shared_gate: b.tensor(
                &format!("{p}mlp.shared_experts.gate_proj.weight"),
                2048 * 7168 * 2,
            ),
            shared_up: b.tensor(
                &format!("{p}mlp.shared_experts.up_proj.weight"),
                2048 * 7168 * 2,
            ),
            shared_down: b.tensor(
                &format!("{p}mlp.shared_experts.down_proj.weight"),
                7168 * 2048 * 2,
            ),
        };
        let w = K3LayerWeights {
            mixer: K3Mixer::Kda {
                cfg: &ck,
                w: &kw,
                st: &ks,
            },
            ffn: K3Ffn::Latent(&mw),
            post_ln: b.tensor(&format!("{p}post_attention_layernorm.weight"), 7168 * 2),
            sa_score_w: b.tensor(&format!("{p}self_attention_res_score.weight"), 7168 * 4),
            mlp_score_w: b.tensor(&format!("{p}mlp_res_score.weight"), 7168 * 4),
        };
        let prefix_in = b.tensor("act.x", 7168 * 2);
        let blkres = b.tensor("kv.blkres", 8 * 7168 * 2);
        let out = b.tensor("act.xnext", 7168 * 2);
        let seed = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        emit_k3_block(
            &mut b,
            &cb,
            &cm,
            &w,
            "act.l1.",
            1,
            4096,
            out,
            prefix_in,
            blkres,
            8,
            1,
            256,
            &mut K3Tp::none(),
            &[seed],
            false,
        );
        b.finish()
    }

    fn build_layer1() -> packet::devbuild::Program {
        build_layer1_with_group(false)
    }

    #[test]
    fn grouped_decode_collapses_topk_packets_and_preserves_operands() {
        let p = build_layer1_with_group(true);
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        assert_eq!(n(DevOp::MoeExpertGluFp8Blk), 0);
        assert_eq!(n(DevOp::MoeExpertDownFp8Blk), 0);
        assert_eq!(n(DevOp::MoeGroupGluFp8Blk), 1);
        assert_eq!(n(DevOp::MoeGroupDownFp8Blk), 1);
        let g = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupGluFp8Blk as u16)
            .unwrap();
        assert_eq!(g.i[0], 16, "grouped packet loops every selected slot");
        assert_eq!(
            g.i[1], 3072,
            "single-rank block keeps the full expert intermediate width"
        );
        assert_eq!(g.i[2], 3584, "experts consume the latent width");
        assert_eq!(g.i[5], K3_MOE_ACT_SITU);
        assert_eq!(g.i[6], K3_MOE_ENC_MXFP4);
        let d = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupDownFp8Blk as u16)
            .unwrap();
        assert_eq!(d.i[0], 16);
        assert_eq!(d.i[1], 3584);
        assert_eq!(d.i[2], 3072);
        assert_eq!(d.i[6], K3_MOE_ENC_MXFP4);
    }

    /// Every op the critical block needs is emitted, in the counts the gate used.
    #[test]
    fn critical_block_emits_the_rung2_op_census() {
        let p = build_layer1();
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        // Two AttnRes with DIFFERENT folded weights — the single most K3-specific
        // thing about the block, and invisible at the block output.
        assert_eq!(n(DevOp::AttnRes), 2, "both AttnRes mixes must be emitted");
        let ars: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::AttnRes as u16)
            .collect();
        assert_ne!(
            ars[0].t[3], ars[1].t[3],
            "the two mixes must use DIFFERENT score weights"
        );
        // The KDA mixer, whole. THREE packets, not six: `kda::fuse_kda` merges the three convs
        // along the channel axis and folds the gate into the recurrence's LDS staging, which is
        // worth 207 packets per token against a measured ~12 us per packet.
        for o in [DevOp::KdaConv3, DevOp::KdaStateStepG, DevOp::KdaGatedNorm] {
            assert!(p.insts.iter().any(|i| i.op == o as u16), "{o:?} missing");
        }
        assert_eq!(
            n(DevOp::KdaConv3),
            1,
            "ONE conv packet over all three streams"
        );
        for o in [DevOp::KdaConv, DevOp::KdaGate, DevOp::KdaStateStep] {
            assert_eq!(
                n(o),
                0,
                "{o:?} is the unfused spelling and must not be emitted"
            );
        }
        // The LatentMoE: router, top_k expert pairs, one combine.
        assert_eq!(n(DevOp::MoeRouterTopk), 1);
        assert_eq!(
            n(DevOp::MoeExpertGluFp8Blk),
            16,
            "one gate/up per selected slot"
        );
        assert_eq!(n(DevOp::MoeExpertDownFp8Blk), 16);
        assert_eq!(n(DevOp::MoeCombine), 1);
        // ZERO standalone situ GLUs at decode: the shared expert's gate|up|situ is now ONE fused
        // `GemvGlu` carrying the betas in f[0]/f[1] (`fuse_shared_glu`), and the routed experts
        // carry situ as an `act` code inside their own GEMM. `PLOW_K3_FUSE_SHGLU=0` restores the
        // three-packet form and this count to 1.
        assert_eq!(n(DevOp::SituGlu), 0);
        assert_eq!(n(DevOp::GemvGlu), 1, "the shared expert, fused");
    }

    /// The three operand facts the rung-2 gate pins, which a code read would not.
    #[test]
    fn critical_block_pins_the_latent_widths_and_the_shared_expert_input() {
        let p = build_layer1();
        // Every routed expert runs at the LATENT width, not at hidden. Sizing
        // them at hidden is a silent 2x error.
        for i in p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::MoeExpertGluFp8Blk as u16)
        {
            assert_eq!(i.i[2], 3584, "expert GLU K must be the latent width");
            assert_eq!(i.i[5], 2, "routed experts must carry the situ act code");
            assert_eq!(
                i.i[6], 2,
                "mxfp4 encoding must ride i[6], not i[3] (= n_exp)"
            );
        }
        for i in p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::MoeExpertDownFp8Blk as u16)
        {
            assert_eq!(i.i[1], 3584, "expert DOWN N must be the latent width");
        }
        // The combine runs at latent width with NO residual and NO shared operand.
        let cmb = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeCombine as u16)
            .unwrap();
        assert_eq!(
            cmb.i[0], 3584,
            "combine must accumulate at the latent width"
        );
        assert_eq!(
            cmb.t[1],
            packet::dev::TENSOR_NONE,
            "no residual at latent width"
        );
        assert_eq!(
            cmb.t[2],
            packet::dev::TENSOR_NONE,
            "shared expert is added AFTER the up-proj"
        );
        // The router scores the HIDDEN state.
        let rl = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeRouterTopk as u16)
            .unwrap();
        assert_eq!(rl.i[1], 896);
        assert_eq!(rl.i[2], 16);
    }

    /// A snapshot layer RESTARTS the prefix — no residual add — and layer 0
    /// additionally skips both mixes because its ring is empty. Getting this
    /// wrong is a 1.0 relative difference, not a rounding one.
    #[test]
    fn snapshot_layers_restart_the_prefix_instead_of_adding() {
        let p = build_layer1();
        // Layer 1 is NOT a snapshot layer: it adds.
        let res = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::Residual as u16)
            .count();
        assert!(res >= 2, "layer 1 needs the prefix add and the block out");
        assert_eq!(k3().blocks_at(0), 0, "layer 0 enters with an empty ring");
        assert!(k3().snapshots(0) && !k3().snapshots(1));
    }

    fn model_cfg() -> K3ModelCfg {
        K3ModelCfg {
            block: k3(),
            kda: crate::kda::KdaCfg {
                hidden: 7168,
                heads: 96,
                head_dim: 128,
                conv_w: 4,
                gate_lower_bound: Some(-5.0),
                eps: 1e-5,
                bv: 16,
            },
            mla: K3MlaCfg {
                hidden: 7168,
                heads: 96,
                q_lora: 1536,
                kv_lora: 512,
                qk_nope: 128,
                qk_rope: 64,
                v_head: 128,
                eps: 1e-5,
                scale: 0.0883883,
                n_split: 4,
                gf: 4,
                fp8_kv: false,
            },
            moe: k3_moe(),
            vocab: 163840,
            first_k_dense: 1,
            dense_inter: 33792,
            prefix: "language_model.model.".into(),
            tp: 1,
        }
    }

    /// A THREE-LAYER K3 decode program — embed, layers 0/1/2, tail. Layers 0-2
    /// of the shipped config are all KDA, so this is the deepest truncation that
    /// needs no MLA block, and it exercises every structural case: layer 0's
    /// empty ring and dense FFN, and the prefix accumulate + LatentMoE on 1/2.
    #[test]
    fn a_truncated_model_emits_embed_layers_and_tail() {
        let c = model_cfg();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|_| true,
            &[0, 1, 2],
            4096,
            1,
            1,
            1,
            256,
            RowKind::Tokens,
        );
        let p = b.finish();
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();

        assert_eq!(n(DevOp::Embed), 1, "one embed prologue");
        assert_eq!(n(DevOp::KdaStateStepG), 3, "one recurrence per layer");
        assert_eq!(
            n(DevOp::KdaConv3),
            3,
            "ONE fused short conv per layer, not three"
        );
        // Layer 0 is dense, layers 1 and 2 are LatentMoE.
        assert_eq!(n(DevOp::MoeRouterTopk), 2, "only the two MoE layers route");
        assert_eq!(n(DevOp::MoeCombine), 2);
        // situ: 1 packet for layer 0's DENSE FFN only. The two MoE layers' shared experts are now
        // fused into `GemvGlu`, so they no longer emit a standalone situ packet.
        assert_eq!(n(DevOp::SituGlu), 1, "layer 0's dense FFN");
        assert_eq!(
            n(DevOp::GemvGlu),
            2,
            "one fused shared expert per MoE layer"
        );
        // Two AttnRes per layer, including layer 0 — which enters with an EMPTY
        // ring but is a SNAPSHOT layer, so its attention-side packet exists to
        // carry the push (the mix itself is an exact copy at nb = 0), and its
        // MLP-side mix then runs at the POST-push count of 1.
        // ... plus the MODEL-LEVEL mix before the final norm, which is one per PROGRAM and not
        // per layer. `2 * layers + 1`.
        assert_eq!(
            n(DevOp::AttnRes),
            7,
            "two mixes per layer, push included, plus the output mix"
        );
        let ars: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::AttnRes as u16)
            .collect();
        // Exactly one push across layers 0..2: only layer 0 snapshots at B = 12.
        let pushes = ars
            .iter()
            .filter(|i| i.t[4] != packet::dev::TENSOR_NONE)
            .count();
        assert_eq!(pushes, 1, "only layer 0 snapshots in 0..2");
        // Layer 0: mix reads 0 rows, then the push lands on row 0.
        assert_eq!(ars[0].i[2], 0, "layer 0 mixes over an empty ring");
        assert_eq!(ars[0].i[3], 0, "and pushes onto row 0");
        // Its MLP-side mix sees the pushed row.
        assert_eq!(ars[1].i[2], 1, "post-push count");
        // The tail.
        assert_eq!(n(DevOp::Argmax), 1);
        assert_eq!(n(DevOp::ArgmaxFin), 1);
        let lm = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::Gemv as u16)
            .last()
            .unwrap();
        assert_eq!(
            lm.i[1], 163840,
            "the last GEMV is lm_head over the full vocab"
        );
    }

    /// Truncation shrinks the TENSOR TABLE, which is the whole point of the
    /// knob: a truncated model must load in seconds, not pay a full-checkpoint
    /// load per iteration.
    #[test]
    fn truncation_shrinks_the_tensor_table() {
        let c = model_cfg();
        let count = |layers: &[u32]| {
            let mut b = Builder::new(256);
            emit_k3_model(
                &mut b,
                &c,
                &|_| true,
                layers,
                4096,
                1,
                1,
                1,
                256,
                RowKind::Tokens,
            );
            b.finish().tensors.len()
        };
        assert!(count(&[0]) < count(&[0, 1]) && count(&[0, 1]) < count(&[0, 1, 2]));
    }

    /// The HYBRID: a four-layer span where layer 3 is MLA and 0/1/2 are KDA.
    /// This is the minimum honest truncation — anything shorter does not
    /// exercise the mixer partition at all, and the partition is the single
    /// most K3-specific thing about the model.
    #[test]
    fn a_hybrid_span_emits_both_mixers() {
        let c = model_cfg();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|l| l != 3,
            &[0, 1, 2, 3],
            4096,
            1,
            1,
            1,
            256,
            RowKind::Tokens,
        );
        let p = b.finish();
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        // Three KDA recurrences and exactly one MLA attention.
        assert_eq!(n(DevOp::KdaStateStepG), 3, "layers 0/1/2 are KDA");
        assert_eq!(n(DevOp::FlashMlaDecode), 1, "layer 3 is MLA");
        assert_eq!(n(DevOp::MlaOutGate), 1, "K3's MLA carries an output gate");
        assert_eq!(n(DevOp::MlaMergeFold), 1);
        // BOTH HeadNormRope ops survive on the MLA layer. The k-side one is the
        // only writer of the krot cache row and the instruction the runtime
        // scans for; dropping it is garbage that grows with context.
        assert_eq!(
            n(DevOp::HeadNormRope),
            2,
            "NoPE keeps both rope ops, neutralized"
        );
        // The MLA layer binds q_a_proj, never q_proj.
        let names: Vec<&str> = p.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names
            .iter()
            .any(|n| n.contains("layers.3.self_attn.q_a_proj")));
        assert!(!names
            .iter()
            .any(|n| n.contains("layers.3.self_attn.q_proj")));
        assert!(names
            .iter()
            .any(|n| n.contains("layers.2.self_attn.q_proj")));
        // The MLA layer declares KV cache rows; KDA layers declare recurrent state.
        assert!(names.contains(&"kv.3.ckv") && names.contains(&"kv.3.krot"));
        assert!(names.iter().any(|n| *n == "kv.2.state"));
        assert!(!names.contains(&"kv.2.ckv"), "a KDA layer has no KV cache");
    }

    /// THE WHOLE MODEL: all 93 layers, 69 KDA and 24 MLA, at the shipped
    /// geometry. This is the emit the tree called "THE ONE REMAINING BLOCKER".
    #[test]
    fn the_full_93_layer_hybrid_emits() {
        let c = model_cfg();
        // The real partition shape: 24 of 93 are full-attention. Spread them the
        // way the checkpoint does — roughly every fourth layer — and take the
        // membership from a LIST, never a stride, which is the rule that matters.
        let mla: std::collections::BTreeSet<u32> =
            (0..93).filter(|l| l % 4 == 3).take(24).collect();
        assert_eq!(mla.len(), 23); // 93/4 rounds to 23; the 24th rides layer 92.
        let mut mla = mla;
        mla.insert(92);
        assert_eq!(mla.len(), 24, "24 MLA layers, 69 KDA");

        let layers: Vec<u32> = (0..93).collect();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|l| !mla.contains(&l),
            &layers,
            4096,
            1,
            1,
            1,
            256,
            RowKind::Tokens,
        );
        let p = b.finish();
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();

        assert_eq!(n(DevOp::KdaStateStepG), 69, "69 KDA layers");
        assert_eq!(n(DevOp::FlashMlaDecode), 24, "24 MLA layers");
        assert_eq!(n(DevOp::MlaOutGate), 24, "every MLA layer is gated");
        assert_eq!(
            n(DevOp::HeadNormRope),
            48,
            "both rope ops on every MLA layer"
        );
        // 92 MoE layers (layer 0 is dense), each routing once and combining once.
        assert_eq!(n(DevOp::MoeRouterTopk), 92);
        assert_eq!(n(DevOp::MoeCombine), 92);
        // One embed, one tail.
        assert_eq!(n(DevOp::Embed), 1);
        assert_eq!(n(DevOp::ArgmaxFin), 1);
        // The snapshot ring never exceeds the kernel's bound: ceil(92/12) = 8.
        let worst = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::AttnRes as u16)
            .map(|i| i.i[2])
            .max()
            .unwrap();
        assert_eq!(worst, 8, "8 live snapshots at the deepest layer");
        assert!(worst <= K3_ATTNRES_MAXB);
        // Layer 92 is MLA on this partition, so the LAST layer must bind
        // q_a_proj. A modulus rule would disagree here — that is the documented
        // off-by-one the config LIST exists to prevent.
        let names: Vec<&str> = p.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names
            .iter()
            .any(|n| n.contains("layers.92.self_attn.q_a_proj")));
    }

    /// Build the full 93-layer hybrid at a given TP degree.
    fn build_full(tp: u32) -> packet::devbuild::Program {
        let mut c = model_cfg();
        c.tp = tp;
        let mut mla: std::collections::BTreeSet<u32> =
            (0..93).filter(|l| l % 4 == 3).take(24).collect();
        mla.insert(92);
        let layers: Vec<u32> = (0..93).collect();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|l| !mla.contains(&l),
            &layers,
            4096,
            1,
            1,
            1,
            256,
            RowKind::Tokens,
        );
        b.finish()
    }

    #[test]
    fn fp8_latent_kv_swaps_the_writer_flash_and_storage() {
        let mut c = model_cfg();
        c.tp = 8;
        c.mla.fp8_kv = true;
        let mut mla: std::collections::BTreeSet<u32> =
            (0..93).filter(|l| l % 4 == 3).take(24).collect();
        mla.insert(92);
        let layers: Vec<u32> = (0..93).collect();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|l| !mla.contains(&l),
            &layers,
            4096,
            1,
            1,
            1,
            256,
            RowKind::Tokens,
        );
        let p = b.finish();
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        assert_eq!(n(DevOp::FlashMlaDecode), 0);
        assert_eq!(n(DevOp::FlashMlaDecodeFp8), 24);
        assert_eq!(
            n(DevOp::HeadNormRopeFp8),
            24,
            "one latent-cache quantizer per MLA layer"
        );
        let flash = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::FlashMlaDecodeFp8 as u16)
            .unwrap();
        assert_ne!(
            flash.t[7],
            packet::dev::TENSOR_NONE,
            "flash needs the row-scale strip"
        );
        assert_eq!(flash.i[6], 0, "the 64-wide NoPE cache remains bf16");
        let bytes = |name: &str| p.tensors.iter().find(|t| t.name == name).unwrap().bytes;
        assert_eq!(
            bytes("kv.3.ckv"),
            4096 * 512,
            "latent cache is one e4m3 byte/value"
        );
        assert_eq!(bytes("kv.3.krot"), 4096 * 64 * 2, "NoPE cache remains bf16");
        assert_eq!(
            bytes("kv.3.scale"),
            4096 * 4,
            "one f32 scale per latent row"
        );
    }

    /// **THE FULL MODEL AT TP8.** Two all-reduces per layer, the same count GLM
    /// runs, and the second one lands at the LATENT width rather than at the FFN
    /// output — forced by `routed_expert_norm` sitting between the combine and
    /// the up-projection. A norm applied to a partial sum is finite, plausible
    /// and wrong.
    #[test]
    fn the_full_model_emits_at_tp8() {
        let p = build_full(8);
        let xr: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16)
            .collect();
        // 93 attention (hidden) + 92 MoE-combine (latent) + 1 dense-FFN (hidden)
        // + 92 SHARED-EXPERT (hidden). The third per-MoE-layer reduce is what pays for sharding
        // `shared_experts.*`, whose replicated form streamed 24.3 GB/rank/token — 49% of the whole
        // decode weight stream. It reuses peer slot A rather than adding a name the host would have
        // to bind; see the emit site for the barrier argument that makes the reuse safe.
        assert_eq!(
            xr.len(),
            278,
            "three all-reduces per MoE layer, two per attention-only layer"
        );
        for i in &xr {
            assert_eq!(i.i[1], 8, "every collective must carry the world size");
        }
        // Every gate id is UNIQUE. A collision is a rendezvous two collectives
        // both wait on, which deadlocks or, worse, releases early.
        let gates: std::collections::BTreeSet<u32> = xr.iter().map(|i| i.i[3]).collect();
        assert_eq!(gates.len(), xr.len(), "xctr gate ids must not collide");
        // The two reductions in a layer use DIFFERENT peer slots, or they alias.
        let slots: std::collections::BTreeSet<u32> = xr.iter().map(|i| i.i[2]).collect();
        assert_eq!(
            slots.len(),
            2,
            "attention and FFN partials need distinct slots"
        );
        // The attention reduce is hidden-wide; the MoE one is LATENT-wide.
        let widths: std::collections::BTreeSet<u32> = xr.iter().map(|i| i.i[0]).collect();
        assert!(widths.contains(&7168), "attention reduces at hidden width");
        assert!(
            widths.contains(&3584),
            "the expert combine reduces at LATENT width"
        );
    }

    /// TP1 emits NO collective at all — the identity, so a TP1 blob is
    /// byte-identical to one built before tensor parallelism existed.
    #[test]
    fn tp1_emits_no_collectives() {
        let p = build_full(1);
        assert_eq!(
            p.insts
                .iter()
                .filter(|i| i.op == DevOp::XReduce as u16)
                .count(),
            0
        );
    }

    /// TP shards the HEAD axis and the expert INTERMEDIATE, and leaves the
    /// latent projections whole. At tp8 K3's 96 KDA heads become 12 and its 96
    /// MLA heads become 12 — which is why the group factor is 4 and not 8.
    #[test]
    fn tp8_shards_the_head_axis_and_the_expert_width() {
        let p = build_full(8);
        let bytes = |suffix: &str| {
            p.tensors
                .iter()
                .find(|t| t.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing tensor ending in {suffix}"))
                .bytes
        };
        let c = model_cfg();
        let dense_shard = c.dense_inter as u64 / 8 * c.block.hidden as u64 * 2;
        assert_eq!(
            bytes("layers.0.mlp.gate_proj.weight"),
            dense_shard,
            "dense gate must declare its column-parallel rank slice"
        );
        assert_eq!(
            bytes("layers.0.mlp.up_proj.weight"),
            dense_shard,
            "dense up must declare its column-parallel rank slice"
        );
        assert_eq!(
            bytes("layers.0.mlp.down_proj.weight"),
            dense_shard,
            "dense down must declare its row-parallel rank slice"
        );
        // Expert GLU N is the LOCAL intermediate: 3072/8.
        let g = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeExpertGluFp8Blk as u16)
            .unwrap();
        assert_eq!(g.i[1], 384, "expert intermediate shards");
        assert_eq!(
            g.i[2], 3584,
            "but the latent width does NOT — it is shared by every expert"
        );
        // FlashMLA runs the local head count.
        let f = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::FlashMlaDecode as u16)
            .unwrap();
        assert_eq!(f.i[1], 12, "96 heads / tp8");
        assert_eq!(
            f.i[7], 4,
            "the group factor must divide nh_l = 12; 8 does not"
        );
    }

    /// A TP degree that does not divide the head count is REFUSED at emit.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_tp_degree_that_does_not_divide_is_refused() {
        build_full(7);
    }

    /// **`routed_expert_up_proj` IS COLUMN-PARALLEL, AND ITS GATHER COSTS NO PACKET.**
    ///
    /// All four halves have to agree or the model is silently wrong, so all four are
    /// pinned here: the DECLARED size (which is what `plowrt::asset::shard` reads as the
    /// sharding request), the GEMV's N, the peer slot it writes, and the collective that
    /// gathers it. A declared slice with a full-width GEMV writes off the end of the peer
    /// slot; a full declaration with a sliced GEMV computes 1/8 of the FFN and adds it to
    /// seven zeros.
    ///
    /// Replicated this tensor was 92 x 7168 x 3584 x 2 B = **4.73 GB per rank per token**,
    /// ~0.74 ms at the 6.4 TB/s streaming ceiling, seven eighths redundant at TP8.
    #[test]
    fn tp8_shards_the_up_projection_and_folds_its_gather_into_the_shared_reduce() {
        // This pins the DEFAULT. `PLOW_K3_SHARD_UP=0` is the A/B control and deliberately
        // emits the replicated form; asserting the default under it would fail the suite
        // for the one run that is supposed to produce the other blob.
        if !shard_up_proj(8) {
            return;
        }
        let p = build_full(8);
        let c = model_cfg();
        let (hid, lat) = (c.block.hidden, c.moe.latent);
        let hid_l = hid / 8;

        // 1. The DECLARED size is the request: 1/tp of [hidden, latent].
        let up = p
            .tensors
            .iter()
            .find(|t| {
                t.name
                    .ends_with("block_sparse_moe.routed_expert_up_proj.weight")
            })
            .expect("no up projection declared");
        assert_eq!(
            up.bytes,
            hid_l as u64 * lat as u64 * 2,
            "declared 1/tp of [hidden, latent]"
        );
        // The DOWN projection stays whole — the experts each need the entire latent.
        let down = p
            .tensors
            .iter()
            .find(|t| {
                t.name
                    .ends_with("block_sparse_moe.routed_expert_down_proj.weight")
            })
            .unwrap();
        assert_eq!(
            down.bytes,
            lat as u64 * hid as u64 * 2,
            "down stays replicated"
        );

        // 2. The GEMV computes the local column count, over the FULL latent — which is
        //    what makes the shard bit-neutral.
        let ug = p
            .tensors
            .iter()
            .position(|t| t.name == "act.ug_tp")
            .expect("no gather slot");
        // The three slots are the same stride, which is the host's `PARTIAL_SLOTS` contract.
        let slot_b = p.tensors[ug].bytes as u32;
        assert_eq!(
            slot_b,
            p.tensors
                .iter()
                .find(|t| t.name == "act.og_tp")
                .unwrap()
                .bytes as u32,
            "the gather slot must be the same stride the host lays slots out at"
        );
        let up_gemv = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::Gemv as u16 && i.t[2] as usize == up_h(&p))
            .expect("no GEMV reads the up projection");
        assert_eq!(up_gemv.i[1], hid_l, "N is hidden/tp");
        assert_eq!(
            up_gemv.i[2], lat,
            "K is the WHOLE latent: column-parallel splits the output"
        );
        assert_eq!(
            up_gemv.t[0] as usize, ug,
            "the partial must land in the peer gather slot"
        );

        // 3. Exactly one collective per MoE layer carries the gather, and it is the
        //    shared expert's reduce out of slot 0 — no packet was added for it.
        let gathering: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16 && i.i[5] != 0)
            .collect();
        assert_eq!(gathering.len(), 92, "one per MoE layer");
        for g in &gathering {
            assert_eq!(
                g.i[2], 0,
                "reduces slot 0 — the shared expert's row-parallel partial"
            );
            assert_eq!(
                g.i[4],
                2 * slot_b,
                "gathers slot 2, at 2 x the host's slot stride"
            );
            assert_eq!(g.i[5], hid_l, "columns per rank");
            assert_eq!(g.i[6], hid, "the full row width");
            assert_eq!(g.i[0], hid, "and it still writes the whole hidden vector");
        }
        // THE COLLECTIVE COUNT IS UNCHANGED BY THE SHARD, which is the point: 93 attention
        // reduces, layer 0's dense down, and the combine + shared pair on each of the 92
        // MoE layers. The gather added none of them.
        let xr = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16)
            .count();
        assert_eq!(xr, 93 + 1 + 92 * 2, "the shard must not add a collective");
    }

    /// Handle of the first layer's `routed_expert_up_proj`, by name.
    fn up_h(p: &packet::devbuild::Program) -> usize {
        p.tensors
            .iter()
            .position(|t| {
                t.name
                    .ends_with("block_sparse_moe.routed_expert_up_proj.weight")
            })
            .unwrap()
    }

    /// The block-residual count per layer, and the two layers the gates were run on.
    #[test]
    fn snapshot_schedule_matches_the_reference_control_flow() {
        let c = k3();
        // Layer 0: snapshot, and NOTHING to mix with yet — so the attn-side AttnRes is skipped and
        // the prefix sum is reset. Its block output is `attn + ffn`.
        assert_eq!(c.blocks_at(0), 0);
        assert!(c.snapshots(0));
        // Layer 1: no snapshot, one row available — BOTH AttnRes calls live, prefix accumulates.
        assert_eq!(c.blocks_at(1), 1);
        assert!(!c.snapshots(1));
        // Layer 12 snapshots again; layer 13 sees two rows.
        assert!(c.snapshots(12));
        assert_eq!(c.blocks_at(12), 1);
        assert_eq!(c.blocks_at(13), 2);
        // 93 layers -> snapshots at 0,12,...,84 = 8 of them, so nb never exceeds 8.
        let snaps: Vec<u32> = (0..93).filter(|&l| c.snapshots(l)).collect();
        assert_eq!(snaps, vec![0, 12, 24, 36, 48, 60, 72, 84]);
        assert_eq!(c.max_blocks(93), 8);
        assert!(c.max_blocks(93) <= K3_ATTNRES_MAXB);
    }

    /// `emit_attn_res` must refuse an `nb` the kernel would silently drop.
    #[test]
    #[should_panic(expected = "PLOW_ATTNRES_MAXB")]
    fn attn_res_refuses_an_nb_the_arm_would_silently_ignore() {
        let c = k3();
        let mut b = Builder::new(256);
        let o = b.tensor("act.o", 7168 * 2);
        let p = b.tensor("act.p", 7168 * 2);
        let r = b.tensor("act.r", 7168 * 2 * 17);
        let w = b.tensor("act.w", 7168 * 4);
        emit_attn_res(
            &mut b,
            &c,
            o,
            p,
            r,
            w,
            1,
            K3_ATTNRES_MAXB + 1,
            K3_ATTNRES_MAXB + 1,
            256,
            None,
            None,
            &[],
        );
    }

    /// Both new opcodes are reachable from an emitter, and AttnRes's honest occupancy is asserted
    /// rather than left to be discovered.
    #[test]
    fn the_two_new_opcodes_are_emitted_and_their_slice_maps_are_pinned() {
        let c = k3();
        let mut b = Builder::new(256);
        let out = b.tensor("act.out", 7168 * 2);
        let prefix = b.tensor("act.prefix", 7168 * 2);
        let blkres = b.tensor("act.blkres", 7168 * 2);
        let sw = b.tensor("act.score_w", 7168 * 4);
        let g = b.tensor("act.g", 33792 * 2);
        let u = b.tensor("act.u", 33792 * 2);
        let a = b.tensor("act.a", 33792 * 2);
        let seed = b.emit(DevOp::Nop, (0..256).collect::<Vec<u32>>(), &[], |_| {});
        let c_ar = emit_attn_res(
            &mut b,
            &c,
            out,
            prefix,
            blkres,
            sw,
            1,
            1,
            1,
            256,
            None,
            None,
            &[seed],
        );
        let c_su = emit_situ_glu(&mut b, &c, a, g, u, 33792, 256, &[c_ar]);
        emit_k3_block_out(&mut b, out, prefix, a, 7168, 256, &[c_su]);
        let p = b.finish();
        let ops: Vec<u16> = p.insts.iter().map(|i| i.op).collect();
        for op in [DevOp::AttnRes, DevOp::SituGlu] {
            assert!(
                ops.contains(&(op as u16)),
                "{op:?} is not emitted — an opcode nothing reaches is how Mamba2Scan became dead \
                 code"
            );
        }
        let ar = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::AttnRes as u16)
            .unwrap();
        assert_eq!(
            ar.blocks, 1,
            "one workgroup per token: 1 of 256 at T=1, a KNOWN perf gap"
        );
        assert_eq!(ar.i[2], 1, "nb");
        let su = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::SituGlu as u16)
            .unwrap();
        // NOT "the whole chip": `d_situ_glu` strides `(slice * PLOW_THREADS + tid) * 8`, so one
        // workgroup covers 512*8 = 4096 elements and 33792 of them need 9. The 247 extra slices
        // this used to pin arrived, took the acquire, found no elements and signalled — ~3 us of
        // 256-way cross-XCD convergence per packet, 93 times a token.
        assert_eq!(su.blocks, 9, "ceil(33792 / (PLOW_THREADS * 8))");
        assert_eq!(su.i[0], 33792);
        // The betas must actually reach the packet. An immediate that is emitted but not read is
        // the contract's SS3 bug shape; one that is READ but never emitted is this one.
        assert_eq!(su.f[0], 4.0);
        assert_eq!(su.f[1], 25.0);
    }

    /// K3's NoPE table must be EXACTLY identity at every position and every angle.
    ///
    /// Not "close to" identity: `HeadNormRope` computes `v*cos ± partner*sin`, and the claim this
    /// gate rests on is that the op becomes a BIT-EXACT copy. `1.0` and `0.0` are exact in f32 and
    /// `x*1 - y*0 == x` for every finite `x`, so the table has to be those two constants and
    /// nothing else. A table that were merely tiny-angled would round-trip through bf16 and pass a
    /// 1.5e-2 residual check while quietly rotating.
    #[test]
    fn k3_nope_table_is_exactly_identity() {
        let ctx = 64u32;
        let hd = 64u32; // qk_rope_head_dim
        let [gc, gs] = k3_nope_rope_pair(ctx, hd);
        let cos = gc.generate().expect("cos recipe must materialize");
        let sin = gs.generate().expect("sin recipe must materialize");
        let n = (ctx * hd / 2) as usize;
        assert_eq!(cos.len(), n * 4);
        assert_eq!(sin.len(), n * 4);
        for i in 0..n {
            let c = f32::from_le_bytes(cos[i * 4..i * 4 + 4].try_into().unwrap());
            let s = f32::from_le_bytes(sin[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(
                c.to_bits(),
                1.0f32.to_bits(),
                "cos[{i}] must be exactly 1.0, got {c}"
            );
            assert_eq!(
                s.to_bits(),
                0.0f32.to_bits(),
                "sin[{i}] must be exactly 0.0, got {s}"
            );
        }
        // And it must NOT coincide with a real table — otherwise the test proves nothing about the
        // emit having actually selected the NoPE recipe.
        let [rc, _] = GenTensor::rope_pair(ctx, hd, 8_000_000.0, 1.0, RopeScale::None);
        assert_ne!(
            rc.generate().unwrap(),
            cos,
            "the NoPE recipe produced GLM's rotating table"
        );
    }

    /// `MlaOutGate` reaches a packet, with the right width and the right operand ORDER.
    ///
    /// The operand order is the load-bearing part: `t1` is the attention output and `t2` is the
    /// g_proj logits, and the kernel applies the sigmoid to `t2` only. Swapping them yields
    /// `sigmoid(attn) * g` — finite, same shape, wrong model — which no shape or symbol check sees.
    #[test]
    fn mla_out_gate_is_emitted_with_the_gated_operand_in_t2() {
        let mut b = Builder::new(256);
        let out = b.tensor("act.oatg", 12288 * 2);
        let attn = b.tensor("act.oat", 12288 * 2);
        let g = b.tensor("act.gproj", 12288 * 2);
        emit_mla_out_gate(&mut b, out, attn, g, 96 * 128, 256, &[]);
        let p = b.finish();
        let i = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MlaOutGate as u16)
            .expect("MlaOutGate is not emitted — an opcode nothing reaches is how Mamba2Scan died");
        assert_eq!(
            i.blocks, 3,
            "ceil(12288 / (PLOW_THREADS * 8)) — d_mla_out_gate is bf16v8 too"
        );
        assert_eq!(i.i[0], 12288, "n_head * v_head_dim = 96 * 128");
        assert_eq!(i.t[0], out);
        assert_eq!(i.t[1], attn, "t1 is the UNGATED attention output");
        assert_eq!(
            i.t[2], g,
            "t2 is the g_proj logits — the operand the sigmoid is applied to"
        );
    }

    /// At T > 1 AttnRes gets more workgroups, capped by the CU count.
    #[test]
    fn attn_res_blocks_track_the_token_count() {
        let c = k3();
        for (t, want) in [(1u32, 1u32), (4, 4), (512, 256)] {
            let mut b = Builder::new(256);
            let o = b.tensor("act.o", 1);
            let p = b.tensor("act.p", 1);
            let r = b.tensor("act.r", 1);
            let w = b.tensor("act.w", 1);
            emit_attn_res(&mut b, &c, o, p, r, w, t, 1, 1, 256, None, None, &[]);
            assert_eq!(b.finish().insts[0].blocks, want as u16, "T={t}");
        }
    }

    #[test]
    fn full_graph_xreduce_attnres_fusion_is_prefill_only_and_keeps_token_ownership() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_FUSE_RESIDUAL_INPUT", "1"),
            ("PLOW_FUSE_XR_ATTNRES", "1"),
        ]);

        let decode = build_full_t(8, 1);
        assert_eq!(
            decode
                .insts
                .iter()
                .filter(|i| {
                    i.op == DevOp::XReduceTwoShot as u16 && i.t[3] != packet::dev::TENSOR_NONE
                })
                .count(),
            0,
            "single-row collectives use the one-shot path"
        );

        let prefill = build_full_t(8, 8192);
        let fused: Vec<_> = prefill
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduceTwoShot as u16 && i.t[3] != packet::dev::TENSOR_NONE)
            .collect();
        assert_eq!(
            fused.len(),
            94,
            "93 attention collectives plus the eligible dense-layer output seam"
        );
        for i in fused {
            assert_eq!(
                i.blocks, 256,
                "phase 2 and AttnRes share the token-owner grid"
            );
            assert_eq!(i.i[0], 8192 * 7168);
            assert_eq!(i.i[1], 8);
            assert_eq!(i.i[5], 7168);
            assert_eq!(i.i[0] / i.i[5], 8192);
            assert_eq!(
                (i.i[0] / i.i[5]) % i.i[1],
                0,
                "rank slices end on row boundaries"
            );
        }
    }

    // ================================================================================
    // PREFILL
    //
    // There is NO K3 CHECKPOINT on this machine, so nothing below is a numeric gate and none of it
    // claims to be. What it pins is STRUCTURE: which opcodes a bucket emits, which it must not
    // emit, what the ring strides by, and that the decode program did not move. Every one of these
    // is a place a wrong answer is silent — the AMD interpreter's dispatch `default:` writes
    // nothing, so an op the prefill object was not built with produces a finite, plausible,
    // completely wrong layer.
    // ================================================================================

    /// Build the 93-layer hybrid at `t` rows, at a given TP degree.
    fn build_full_t(tp: u32, t: u32) -> packet::devbuild::Program {
        build_full_rows(tp, t, RowKind::Tokens)
    }

    fn build_full_rows(tp: u32, t: u32, rows: RowKind) -> packet::devbuild::Program {
        let mut c = model_cfg();
        c.tp = tp;
        let mut mla: std::collections::BTreeSet<u32> =
            (0..93).filter(|l| l % 4 == 3).take(24).collect();
        mla.insert(92);
        let layers: Vec<u32> = (0..93).collect();
        let mut b = Builder::new(256);
        emit_k3_model(
            &mut b,
            &c,
            &|l| !mla.contains(&l),
            &layers,
            4096,
            t,
            t,
            if rows == RowKind::Sequences { t } else { 1 },
            256,
            rows,
        );
        b.finish()
    }

    /// Opcodes that exist ONLY in the decode interpreter object. A prefill bucket carrying any of
    /// them is a silent no-write, not a slow path.
    const DECODE_ONLY: [DevOp; 6] = [
        DevOp::Gemv,
        DevOp::GemvQkvg,
        DevOp::FlashMlaDecode,
        DevOp::MoeRouterTopk,
        DevOp::MoeExpertGluFp8Blk,
        DevOp::MoeExpertDownFp8Blk,
    ];

    /// A prefill bucket emits the GEMM family and the GROUPED MoE chain, and NOTHING decode-only.
    ///
    /// The one deliberate exception is the lm_head, which stays a `Gemv` with `M = 1` and
    /// `a_row0 = t-1`: it reduces the LAST row only, the prefill object carries an unconditional
    /// `case PLOW_DOP_GEMV`, and GLM's `emit_glm_tail` makes the identical choice for the identical
    /// reason. So the check is "one Gemv, and it is the tail".
    #[test]
    fn a_prefill_bucket_emits_gemms_and_the_grouped_moe_chain() {
        let p = build_full_t(1, 512);
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        for o in DECODE_ONLY {
            let want = usize::from(o == DevOp::Gemv); // the lm_head, and only it
            assert_eq!(n(o), want, "{o:?} must not appear in a T-row program");
        }
        let lm = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::Gemv as u16)
            .last()
            .unwrap();
        assert_eq!(lm.i[0], 1, "the surviving GEMV is the lm_head: one row");
        assert_eq!(lm.i[1], 163840, "over the full vocab");
        assert_eq!(lm.i[4], 511, "a_row0 = t-1, the last real row");
        // The grouped chain, once per MoE layer (92 of 93 — layer 0 is dense).
        assert_eq!(n(DevOp::MoeRouterTopkPf), 92);
        assert_eq!(n(DevOp::MoeAlignPf), 92);
        assert_eq!(n(DevOp::MoeGroupGluPf), 92);
        assert_eq!(n(DevOp::MoeGroupDownPf), 92);
        assert_eq!(n(DevOp::MoeCombinePf), 92);
        // The MLA half.
        assert_eq!(n(DevOp::FlashMlaPrefill), 24);
        assert_eq!(n(DevOp::MlaMergeFold), 24);
        assert_eq!(
            n(DevOp::MlaOutGate),
            24,
            "the output gate is not a decode-only thing"
        );
        // The block structure is UNCHANGED — that is the point of AttnRes being in both buckets.
        assert_eq!(
            n(DevOp::AttnRes),
            187,
            "two mixes per layer plus the output mix, at T rows too"
        );
        for o in [
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ] {
            assert_eq!(n(o), 69, "one {o:?} packet per KDA layer");
        }
        assert_eq!(n(DevOp::KdaStateStepG), 0);
        assert_eq!(n(DevOp::KdaConv3), 69);
        assert_eq!(
            n(DevOp::SituGlu),
            93,
            "layer 0's dense FFN + one shared expert per MoE layer"
        );
        assert_eq!(n(DevOp::Embed), 1);
        assert_eq!(n(DevOp::ArgmaxFin), 1);
    }

    #[test]
    fn materialized_mla_prefill_is_generic_opt_in_and_has_pure_raw_boundaries() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_MLA_MATERIALIZED_PREFILL", "1")]);
        let p = build_full_t(1, 512);
        let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
        assert_eq!(n(DevOp::MlaMaterializePack), 24);
        assert_eq!(n(DevOp::FlashMlaMaterializedPrefill), 24);
        assert_eq!(n(DevOp::FlashMlaPrefill), 0);
        assert_eq!(n(DevOp::MlaMergeFold), 0);
        assert_eq!(
            p.tensors
                .iter()
                .filter(|t| t.name.ends_with("self_attn.q_b_proj.weight"))
                .count(),
            24
        );
        assert_eq!(
            p.tensors
                .iter()
                .filter(|t| t.name.ends_with("self_attn.kv_b_proj.weight"))
                .count(),
            24
        );

        for (ix, inst) in p.insts.iter().enumerate().filter(|(_, i)| {
            matches!(
                DevOp::from_u16(i.op),
                Some(DevOp::MlaMaterializePack | DevOp::FlashMlaMaterializedPrefill)
            )
        }) {
            let entries: Vec<_> = p.stream.iter().filter(|e| e.inst as usize == ix).collect();
            assert!(!entries.is_empty(), "raw instruction {ix} is unscheduled");
            let seg = entries[0].seg;
            assert!(entries.iter().all(|e| {
                e.seg == seg
                    && e.wait_len == 0
                    && e.succ_len == 0
                    && e.flags & packet::dev::SE_XCTR == 0
            }));
            assert!(p
                .stream
                .iter()
                .filter(|e| e.seg == seg)
                .all(|e| e.inst as usize == ix));
            if inst.op == DevOp::MlaMaterializePack as u16 {
                assert_eq!(inst.i[..5], [512, 96, 128, 64, 128]);
            } else {
                assert_eq!(inst.i[..6], [512, 96, 96, 192, 128, 1]);
            }
        }
    }

    #[test]
    fn early_shared_gate_up_is_batched_decode_only_and_keeps_tp_slot_order() {
        let locate = |p: &packet::devbuild::Program, suffix: &str| {
            let weight = p
                .tensors
                .iter()
                .position(|t| t.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing tensor ending in {suffix}"))
                as u32;
            p.insts
                .iter()
                .position(|i| i.t[2] == weight)
                .unwrap_or_else(|| panic!("missing instruction for weight handle {weight}"))
        };

        let batched = build_full_rows(8, 32, RowKind::Sequences);
        let prefill = build_full_rows(8, 128, RowKind::Tokens);
        let score_name = "layers.1.block_sparse_moe.gate.weight";
        let latent_name = "layers.1.block_sparse_moe.routed_expert_down_proj.weight";
        let gate_name = "layers.1.block_sparse_moe.shared_experts.gate_proj.weight";
        let up_name = "layers.1.block_sparse_moe.shared_experts.up_proj.weight";
        let down_name = "layers.1.block_sparse_moe.shared_experts.down_proj.weight";

        let score = locate(&batched, score_name);
        let latent = locate(&batched, latent_name);
        let gate = locate(&batched, gate_name);
        let up = locate(&batched, up_name);
        let router = batched
            .insts
            .iter()
            .position(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .expect("missing batched router");
        assert!(score < gate && score < up && gate < router && up < router);
        let h3 = batched.insts[score].t[1];
        assert_eq!(batched.insts[latent].t[1], h3);
        assert_eq!(batched.insts[gate].t[1], h3);
        assert_eq!(batched.insts[up].t[1], h3);

        let prefill_router = prefill
            .insts
            .iter()
            .position(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .expect("missing prefill router");
        assert!(locate(&prefill, gate_name) > prefill_router);
        assert!(locate(&prefill, up_name) > prefill_router);

        let down = locate(&batched, down_name);
        let combine = batched
            .insts
            .iter()
            .position(|i| i.op == DevOp::MoeCombinePf as u16)
            .expect("missing routed combine");
        let routed_reduce = ((combine + 1)..down)
            .find(|&id| {
                let op = batched.insts[id].op;
                op == DevOp::XReduce as u16 || op == DevOp::XReduceTwoShot as u16
            })
            .expect("missing routed TP collective after combine");
        let mut waits = Vec::new();
        for entry in batched.stream.iter().filter(|e| e.inst as usize == down) {
            for k in 0..entry.wait_len as usize {
                waits.push(batched.waits[entry.wait_ofs as usize + k].id as usize);
            }
        }
        assert!(waits.contains(&routed_reduce));
    }

    /// The default decode emit carries the full-model-gated narrow-norm fusion.
    #[test]
    fn the_narrow_norm_fusion_is_on_by_default() {
        let p = build_full(8);
        let fused = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::Gemv as u16 && i.i[3] == 2)
            .count();
        assert!(fused > 0, "the default K3 decode must carry norm=2 GEMVs");
        let norms = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::RmsNorm as u16)
            .count();
        assert!(norms > 0, "non-fusable RMSNORM packets must remain");
        let widths: std::collections::BTreeSet<u32> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::RmsNorm as u16)
            .map(|i| i.i[1])
            .collect();
        assert!(
            widths.iter().all(|&w| w % 8 == 0),
            "every fusable width must satisfy the arm's `feat % 8` precondition: {widths:?}"
        );
    }

    /// Every projection in a T-row program carries the REAL row count.
    ///
    /// `M` is the operand that silently truncates: `Gemv` at `M = 512` runs its compiled row
    /// bucket (<= 16) and leaves 496 rows holding the arena, and a GEMM emitted with `i[0] = 1`
    /// would compute one row of a 512-row activation. Both are finite and neither faults.
    #[test]
    fn every_t_row_gemm_carries_the_row_count() {
        let t = 512u32;
        let p = build_full_t(1, t);
        let fam = crate::gemm_family_ops();
        let gemms: Vec<_> = p.insts.iter().filter(|i| fam.contains(&i.op)).collect();
        assert!(
            gemms.len() > 500,
            "a 93-layer prefill program is mostly GEMMs, got {}",
            gemms.len()
        );
        for g in &gemms {
            assert_eq!(g.i[0], t, "a projection at M={} instead of {t}", g.i[0]);
        }
        // The row-carrying norm ops too.
        for i in p.insts.iter().filter(|i| i.op == DevOp::RmsNorm as u16) {
            assert_eq!(i.i[0], t);
        }
        for i in p.insts.iter().filter(|i| i.op == DevOp::AttnRes as u16) {
            assert_eq!(i.i[0], t);
        }
    }

    #[test]
    fn c8_exact_shape_is_explicit_and_default_off_in_the_packet() {
        let _guard = crate::test_env::env_guard();
        crate::set_amd_target("MI350X");
        let emit = |m: u32, n: u32, k: u32| {
            let mut b = Builder::new(256);
            let out = b.tensor("out", 1);
            let x = b.tensor("x", 1);
            let w = b.tensor("w", 1);
            emit_k3_linear(&mut b, out, x, w, m, n, k, 256, false, &[]);
            b.finish().insts[0]
        };

        {
            let _scope = crate::test_env::EnvScope::set(&[("PLOW_GEMM_WIDE_C8_SHAPE", "")]);
            let inst = emit(8192, 1536, 7168);
            assert_eq!(
                inst.pack(),
                packet::dev::DevInst64 {
                    op: DevOp::GemmWide as u16,
                    blocks: 256,
                    fj: [0; 3],
                    t: [0, 1, 2, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff],
                    i: [8192, 1536, 7168, 0, 0, 0, 0, 0],
                },
                "flag-off must preserve the exact pre-c8 64-byte wire packet"
            );
            assert_eq!(inst.i[7], 0, "flag-off packets must retain the c2 encoding");
            assert_eq!(
                inst.op,
                crate::gfx950_prefill_tile(8192, 1536, 7168, 256, kernelcaps::QuantScheme::None)
                    as u16,
                "flag-off selection must use the unchanged TuneDB/analytical path"
            );
            assert_eq!(inst.blocks, 256);
        }
        {
            let _scope =
                crate::test_env::EnvScope::set(&[("PLOW_GEMM_WIDE_C8_SHAPE", "8192x1536x7168")]);
            let inst = emit(8192, 1536, 7168);
            assert_eq!(inst.op, DevOp::GemmWide as u16);
            assert_eq!(inst.i[7], packet::dev::GEMM_WIDE_C8_TAG);
            assert_eq!(inst.blocks, 256, "64x4 c8 tiles must emit the exact grid");

            let other = emit(4096, 1536, 7168);
            assert_eq!(other.i[7], 0, "the opt-in is exact-shape, not model-wide");
        }
        {
            let _scope =
                crate::test_env::EnvScope::set(&[("PLOW_GEMM_WIDE_C8_SHAPE", "4096x1536x7168")]);
            let inst = emit(4096, 1536, 7168);
            assert_eq!(
                inst.i[7], 0,
                "a 128-tile grid must not claim the 256-CU c8 arm"
            );
        }
    }

    #[test]
    fn c8_full_graph_preserves_the_measured_tune_census() {
        let _guard = crate::test_env::env_guard();
        crate::set_amd_target("MI350X");
        let buckets = [128, 512, 1024, 2048, 4096, 8192];
        let emit = |shape: &str| {
            let _scope = crate::test_env::EnvScope::set(&[("PLOW_GEMM_WIDE_C8_SHAPE", shape)]);
            crate::tune_demand::reset_tally();
            crate::tune_demand::start_recording();
            let programs: Vec<_> = buckets.iter().map(|&t| build_full_t(8, t)).collect();
            let model = packet::devbuild::Model {
                n_cu: 256,
                target: 0,
                tensors: programs.last().unwrap().tensors.clone(),
                progs: programs,
                kv_row_insts: Vec::new(),
                prog_t: buckets.to_vec(),
                gen: Vec::new(),
            };
            let manifest = crate::manifest::build(&model, "gfx950", &crate::LeanReport::default());
            (
                model.progs,
                crate::tune_demand::tally(),
                crate::tune_demand::take(),
                manifest,
            )
        };

        let (control, control_census, control_lookups, control_manifest) = emit("");
        let (candidate, candidate_census, candidate_lookups, candidate_manifest) =
            emit("8192x1536x7168");
        assert_eq!(control_census, (7650, 7650));
        assert_eq!(candidate_census, control_census);
        for manifest in [&control_manifest, &candidate_manifest] {
            assert_eq!(manifest["tuning"]["tile_measured"], 7650);
            assert_eq!(manifest["tuning"]["tile_lookups"], 7650);
            assert_eq!(manifest["tuning"]["tile_source"], "measured");
        }
        assert_eq!(control_lookups.len(), 7650);
        assert_eq!(candidate_lookups.len(), control_lookups.len());
        assert!(control_lookups.iter().all(|d| d.hit));
        assert!(candidate_lookups.iter().all(|d| d.hit));

        let mut tagged = 0usize;
        for (t, (a, b)) in buckets.iter().zip(control.iter().zip(&candidate)) {
            assert_eq!(a.insts.len(), b.insts.len(), "M={t}: packet count moved");
            for (before, after) in a.insts.iter().zip(&b.insts) {
                if before != after {
                    assert_eq!(*t, 8192, "C8 changed a non-qualified bucket");
                    let mut expected = *before;
                    expected.i[7] = packet::dev::GEMM_WIDE_C8_TAG;
                    assert_eq!(*after, expected, "C8 may change only the explicit tile tag");
                    tagged += 1;
                }
            }
        }
        assert_eq!(tagged, 324, "the qualified full-graph shape census moved");
    }

    /// **BLOCKER 1.** The ring strides by its CAPACITY, and the capacity reaches the packet.
    ///
    /// `blkres` is `[T][nb_cap][hidden]`. At `T = 1` the token index is 0, so a kernel striding by
    /// the LIVE count `nb` is indistinguishable from a correct one — which is exactly why this was
    /// invisible until prefill. At `T > 1` every layer would address a differently-strided view of
    /// the same buffer and the rows would shift under it: no fault, no NaN, a fluent wrong model.
    #[test]
    fn the_ring_strides_by_the_capacity_not_the_live_count() {
        for t in [1u32, 128] {
            let mut c = model_cfg();
            c.tp = 1;
            let layers: Vec<u32> = (0..93).collect();
            let mut b = Builder::new(256);
            emit_k3_model(
                &mut b,
                &c,
                &|_| true,
                &layers,
                4096,
                t,
                t,
                1,
                256,
                RowKind::Tokens,
            );
            let p = b.finish();
            // ceil(92/12) = 8 live rows at the deepest layer.
            let cap = 8u32;
            let ring = p.tensors.iter().find(|x| x.name == "kv.blkres").unwrap();
            assert_eq!(
                ring.bytes,
                t as u64 * cap as u64 * 7168 * 2,
                "T={t}: the ring needs one private slice PER TOKEN"
            );
            let ars: Vec<_> = p
                .insts
                .iter()
                .filter(|i| i.op == DevOp::AttnRes as u16)
                .collect();
            assert_eq!(ars.len(), 187, "2 per layer + the model-level output mix");
            let live: std::collections::BTreeSet<u32> = ars.iter().map(|i| i.i[2]).collect();
            assert!(
                live.len() > 1,
                "nb varies with depth — that is the whole hazard"
            );
            for i in &ars {
                assert_eq!(
                    i.i[4], cap,
                    "T={t}: every mix must stride by the SAME capacity"
                );
                assert!(i.i[2] <= i.i[4], "live count past the capacity");
            }
        }
    }

    /// **BLOCKER 2.** The snapshot push survives `T > 1` — it is emitted, not refused.
    ///
    /// The kernel used to return qNaN for a push at `T != 1`, on the reasoning that "a
    /// multi-workgroup push has no barrier between them". Re-derived and wrong: `blocks =
    /// min(T, n_cu)` and the loop is `for (t = slice; t < T; t += nblk)`, so the workgroups
    /// partition the TOKENS; the ring is per token; and within a token the mix reads rows
    /// `[0, nb)` while the push writes row `nb`, one past. No workgroup reads another's pushed row
    /// inside the packet, and both real readers — this layer's MLP-side mix and every later layer
    /// — are separate packets joined by `Dep::Coarse`, which waits on every producer slice.
    #[test]
    fn the_snapshot_push_is_emitted_at_t_above_one() {
        for t in [1u32, 128, 4096] {
            let p = build_full_t(1, t);
            let ars: Vec<_> = p
                .insts
                .iter()
                .filter(|i| i.op == DevOp::AttnRes as u16)
                .collect();
            let pushes: Vec<_> = ars
                .iter()
                .filter(|i| i.t[4] != packet::dev::TENSOR_NONE)
                .collect();
            // Snapshots at layers 0, 12, 24, ..., 84 — eight of them, at every T.
            assert_eq!(
                pushes.len(),
                8,
                "T={t}: the snapshot schedule is not a function of T"
            );
            for i in &pushes {
                assert_eq!(i.i[3], i.i[2], "the push lands ONE PAST the live count");
                assert!(i.i[3] < i.i[4], "and inside the ring");
            }
            // Layer 0 still emits its attention-side packet at nb_in == 0: the mix is an exact
            // copy there, and the packet exists for the PUSH. Skipping it would leave the ring
            // holding whatever it was allocated with, and every later mix would read that.
            assert_eq!(ars[0].i[2], 0, "T={t}: layer 0 mixes over an empty ring");
            assert_eq!(ars[0].i[3], 0, "and pushes onto row 0");
            assert_ne!(
                ars[0].t[4],
                packet::dev::TENSOR_NONE,
                "layer 0's packet carries the push"
            );
            assert_eq!(ars[1].i[2], 1, "its MLP-side mix sees the pushed row");
        }
    }

    /// A ring too short for the rows it is asked to hold is REFUSED at emit.
    #[test]
    #[should_panic(expected = "ring capacity")]
    fn a_ring_shorter_than_the_live_count_is_refused() {
        let c = k3();
        let mut b = Builder::new(256);
        let o = b.tensor("act.o", 1);
        let p = b.tensor("act.p", 1);
        let r = b.tensor("act.r", 1);
        let w = b.tensor("act.w", 1);
        emit_attn_res(&mut b, &c, o, p, r, w, 128, 8, 4, 256, None, None, &[]);
    }

    /// A push needs a row the ring HAS: `nb_cap` must exceed the live count, not merely match it.
    #[test]
    #[should_panic(expected = "ring capacity")]
    fn a_push_onto_a_full_ring_is_refused() {
        let c = k3();
        let mut b = Builder::new(256);
        let o = b.tensor("act.o", 1);
        let p = b.tensor("act.p", 1);
        let r = b.tensor("act.r", 1);
        let w = b.tensor("act.w", 1);
        emit_attn_res(&mut b, &c, o, p, r, w, 128, 8, 8, 256, Some(p), None, &[]);
    }

    /// The grouped MoE operands, which a code read would not pin.
    ///
    /// The LATENT width is the K3-specific one and it appears in two different slots on two ops
    /// (`i[1]` on the gate/up, `i[0]` on the down); putting the hidden width in either is the
    /// silent 2x this block's whole design exists to prevent. `situ` and the mxfp4 encoding travel
    /// in slots that are NOT the ones the decode ops use.
    #[test]
    fn the_grouped_prefill_moe_runs_at_the_latent_width_with_situ() {
        let _guard = crate::test_env::env_guard();
        let t = 512u32;
        let p = build_full_t(1, t);
        let g = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupGluPf as u16)
            .unwrap();
        assert_eq!(g.i[0], 3072, "N is the expert intermediate");
        assert_eq!(
            g.i[1], 3584,
            "K is the LATENT, not the hidden — the A operand is `xe`"
        );
        assert_eq!(g.i[2], 896, "n_exp");
        assert_eq!(g.i[crate::mla::MoeEnc::PREFILL_SLOT], K3_MOE_ENC_MXFP4);
        assert_eq!(g.i[5], K3_MOE_ACT_SITU, "the routed GLU carries situ");
        assert_eq!(g.f[0], 4.0, "and the betas the PAIR-form epilogue reads");
        assert_eq!(g.f[1], 25.0);
        // A4W4 binds the pad-row map and the E8M0 rows the fused bridge WRITES.
        assert_ne!(
            g.t[6],
            packet::dev::TENSOR_NONE,
            "row_partidx: pad rows must be skippable"
        );
        assert_ne!(
            g.t[7],
            packet::dev::TENSOR_NONE,
            "the bridge's E8M0 scale rows"
        );
        let d = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupDownPf as u16)
            .unwrap();
        assert_eq!(d.i[0], 3584, "DOWN's N is the latent");
        assert_eq!(d.i[1], 3072);
        assert_eq!(
            d.t[5], g.t[7],
            "DOWN reads exactly the scale rows the bridge wrote"
        );
        // The combine is at LATENT width with NO residual and NO shared operand: there is nothing
        // 3584 wide to add, so both happen after the up-projection.
        let cb = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeCombinePf as u16)
            .unwrap();
        assert_eq!(cb.i[0], 3584);
        assert_eq!(cb.i[1], 16, "top_k");
        assert_eq!(cb.i[2], t, "the combine is over T tokens");
        assert_eq!(cb.t[1], packet::dev::TENSOR_NONE);
        assert_eq!(cb.t[2], packet::dev::TENSOR_NONE);
        // The router scores the HIDDEN state on both phases, and the prefill tail carries T.
        let rt = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .unwrap();
        assert_eq!(rt.i[1], 896);
        assert_eq!(rt.i[2], 16);
        assert_eq!(
            rt.i[4], t,
            "the prefill tail is the decode kernel under a TOKEN LOOP"
        );
        assert_eq!(
            rt.i[6], 1,
            "group operands must match the decode tail or the two phases route"
        );
        assert_eq!(rt.i[7], 1, "the same token to DIFFERENT experts");
        // The align op is ONE workgroup and must be: the MPF_BM-padded row prefix is a global scan.
        let al = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeAlignPf as u16)
            .unwrap();
        assert_eq!(al.blocks, 1);
        assert_eq!(al.i[0], t);
    }

    #[test]
    fn atomic_k3_grouped_prefill_sets_packets_and_manifest_requirement() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_MOE_PF_ATOMIC", "1"),
            ("PLOW_MOE_PF_DET", "0"),
        ]);
        let p = build_full_t(1, 512);
        let router = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .unwrap();
        let down = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupDownPf as u16)
            .unwrap();
        let combine = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeCombinePf as u16)
            .unwrap();
        assert_eq!(router.t[2], down.t[0], "router zeros DOWN's accumulator");
        assert_eq!(router.i[0], 3584, "the accumulator is at latent width");
        assert_eq!(
            down.i[4],
            model_cfg().moe.top_k.trailing_zeros() + 1,
            "log2(top_k) + 1 arms atomic scatter"
        );
        assert_eq!(down.i[5], 0, "the deterministic arm stays disjoint");
        assert_eq!(combine.i[1], 1, "combine reads one accumulated row");
        assert_eq!(combine.i[4], 0, "the accumulator contains f32");

        let m = packet::devbuild::Model {
            n_cu: 256,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: Vec::new(),
            prog_t: vec![512],
            gen: Vec::new(),
        };
        let manifest = crate::manifest::build(&m, "gfx950", &crate::LeanReport::default());
        assert_eq!(
            manifest.pointer("/features/moe_pf_atomic"),
            Some(&serde_json::Value::Bool(true))
        );
        let requires = manifest
            .pointer("/backends/gfx950/requires")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert!(requires
            .iter()
            .any(|v| v.as_str() == Some("PLOW_MOE_PF_ATOMIC=1")));
    }

    #[test]
    fn deterministic_k3_grouped_prefill_sets_f64_packets_and_manifest_requirement() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_MOE_PF_ATOMIC", "0"),
            ("PLOW_MOE_PF_DET", "1"),
        ]);
        let t = 512u32;
        let p = build_full_t(1, t);
        let router = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .unwrap();
        let down = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupDownPf as u16)
            .unwrap();
        let combine = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeCombinePf as u16)
            .unwrap();
        assert_eq!(router.t[2], down.t[0], "router zeros DOWN's accumulator");
        assert_eq!(router.i[0], 3584, "the accumulator is at latent width");
        assert_eq!(
            p.tensors[down.t[0] as usize].bytes,
            t as u64 * 3584 * 8,
            "deterministic accumulation stores one f64 row per token"
        );
        assert_eq!(down.i[4], 0, "the atomic arm stays disjoint");
        assert_eq!(
            down.i[5],
            model_cfg().moe.top_k.trailing_zeros() + 1,
            "log2(top_k) + 1 arms deterministic scatter"
        );
        assert_eq!(combine.i[1], 1, "combine reads one accumulated row");
        assert_eq!(combine.i[4], 1, "combine decodes the accumulator as f64");

        let m = packet::devbuild::Model {
            n_cu: 256,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: Vec::new(),
            prog_t: vec![t],
            gen: Vec::new(),
        };
        let manifest = crate::manifest::build(&m, "gfx950", &crate::LeanReport::default());
        assert_eq!(
            manifest.pointer("/features/moe_pf_det"),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            manifest.pointer("/features/moe_pf_atomic"),
            Some(&serde_json::Value::Bool(false))
        );
        let requires = manifest
            .pointer("/backends/gfx950/requires")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert!(requires
            .iter()
            .any(|v| v.as_str() == Some("PLOW_MOE_PF_DET=1")));
        assert!(!requires
            .iter()
            .any(|v| v.as_str() == Some("PLOW_MOE_PF_ATOMIC=1")));
    }

    #[test]
    fn grouped_router_uses_one_block_per_token_up_to_the_cu_count() {
        for (t, want) in [(32, 32), (128, 128), (512, 256)] {
            let rows = if t == 32 {
                RowKind::Sequences
            } else {
                RowKind::Tokens
            };
            let p = build_full_rows(8, t, rows);
            let router = p
                .insts
                .iter()
                .find(|i| i.op == DevOp::MoeRouterTopkPf as u16)
                .expect("missing grouped router");
            assert_eq!(router.blocks, want, "T={t}");
        }
    }

    #[test]
    fn parallel_align_emits_ordered_count_prefix_scatter_packets() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_MOE_ALIGN_PAR", "1")]);
        let p = build_full_t(1, 1024);
        let align: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::MoeAlignPf as u16)
            .collect();
        let router = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeRouterTopkPf as u16)
            .unwrap();
        assert_eq!(router.blocks, 256);
        assert_eq!(align.len(), 92 * 4);
        for phase in align.chunks_exact(4) {
            assert_eq!(
                phase.iter().map(|i| i.i[3]).collect::<Vec<_>>(),
                [1, 2, 3, 4]
            );
            assert!(phase[0].blocks > 1);
            assert_eq!(phase[1].blocks, 1);
            assert!(phase[2].blocks > 1);
            assert!(phase[3].blocks > 1);
        }
        let meta = p
            .tensors
            .iter()
            .find(|x| x.name == "act.pf.moe.moe_meta")
            .unwrap();
        assert_eq!(meta.bytes, (3 * 896 + 1 + 64 * 896) * 4);
    }

    /// The gathered row arrays are sized on the MPF_BM-PADDED bound, not on `T*k`.
    ///
    /// `MoeAlignPf` rounds each expert's row range up to a whole `MPF_BM` tile, so every expert can
    /// waste up to `MPF_BM-1` rows. Sizing from `T*k` alone is an out-of-bounds DEVICE WRITE with
    /// no symptom at small expert counts and a guaranteed one at K3's 896.
    #[test]
    fn the_gathered_row_arrays_carry_the_align_padding() {
        let _guard = crate::test_env::env_guard();
        let t = 512u64;
        let p = build_full_t(1, t as u32);
        let bytes = |n: &str| p.tensors.iter().find(|x| x.name == n).unwrap().bytes;
        let pad = t * 16 + 896 * (crate::mla::MPF_BM as u64 - 1);
        assert_eq!(bytes("act.pf.moe.moe_rowtok"), pad * 4);
        assert_eq!(bytes("act.pf.moe.moe_rowpart"), pad * 4);
        assert_eq!(bytes("act.pf.moe.moe_rowgate"), pad * 4);
        assert_eq!(bytes("act.pf.moe.moe_meta"), (3 * 896 + 1) * 4);
        // fp4 is HALF a byte per value, plus one E8M0 byte per 32.
        assert_eq!(bytes("act.pf.moe.moe_fug"), pad * (3072 / 2));
        assert_eq!(
            bytes("act.pf.moe.moe_fuscale"),
            pad * (3072 / crate::mla::MX_BLOCK as u64)
        );
        // `part` is [T*k, latent] f32 — no padding; DOWN scatters by row_partidx.
        assert_eq!(bytes("act.pf.moe.part"), t * 16 * 3584 * 4);
        // The per-SLOT decode buffer must NOT be declared: nothing in a T-row program writes it.
        assert!(!p.tensors.iter().any(|x| x.name == "act.pf.moe.fu"));
    }

    /// A prefill bucket shares ONE set of activation buffers across the layer loop.
    ///
    /// Decode's `act.l{l}.…` naming is 93 private sets, which is ~50 KiB each at one token and is
    /// what every existing K3 blob has. At T = 8192 the same naming asks for `act.l{l}.moe.part` =
    /// 1.9 GiB **per layer**. Sharing is safe for the reason GLM's `GlmTn` is: the layers are
    /// strictly serialized by the counter DAG and every scratch buffer is written before it is
    /// read inside its own layer. What stays per-layer is what must — the KDA recurrent state and
    /// the MLA `kv.{l}.*` cache rows, which carry ACROSS layers.
    #[test]
    fn a_prefill_bucket_shares_its_activation_scratch_across_layers() {
        let p = build_full_t(1, 512);
        let names: Vec<&str> = p.tensors.iter().map(|t| t.name.as_str()).collect();
        // `act.l{digit}` — not `act.l`, which also matches the global `act.logits`.
        let per_layer = |n: &str| {
            n.strip_prefix("act.l")
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
        };
        assert!(
            !names.iter().any(|n| per_layer(n)),
            "no per-layer prefill scratch"
        );
        assert!(names.contains(&"act.pf.h_a") && names.contains(&"act.pf.moe.part"));
        // State and cache stay per layer, or layer 5 would read layer 4's recurrence.
        assert!(names.contains(&"kv.0.state") && names.contains(&"kv.92.ckv"));
        // And decode keeps its per-layer naming, byte for byte.
        let d = build_full_t(1, 1);
        let dn: Vec<&str> = d.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(dn.contains(&"act.l1.h_a") && !dn.iter().any(|n| n.starts_with("act.pf.")));
    }

    /// **THE FULL MODEL AT TP8, AT T ROWS.** Two all-reduces per layer, exactly as decode, and the
    /// second still lands at the LATENT width — `routed_expert_norm` sits between the combine and
    /// the up-projection and a norm applied to a partial sum is finite, plausible and wrong.
    ///
    /// The collectives take the TWO-SHOT form at T rows (reduce-scatter + all-gather): the partial
    /// is bandwidth-bound at T rows, so the two-shot moves ~tp/2x less over the fabric. Decode's
    /// one-shot is the right answer at one row and the wrong one here.
    ///
    /// The explicit rollback keeps the shared-expert reduce carrying the up-projection's
    /// ALL-GATHER on the one-shot path. The production-default test below covers its generic
    /// two-shot folded-gather form.
    #[test]
    fn folded_gather_prefill_can_restore_the_one_shot_path() {
        let _guard = crate::test_env::env_guard();
        let _env = crate::test_env::EnvScope::set(&[("PLOW_XR2_GATHER", "0")]);
        // Pins the explicit one-shot rollback while keeping the sharded up projection.
        if !shard_up_proj(8) {
            return;
        }
        let p = build_full_t(8, 512);
        let xr: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16 || i.op == DevOp::XReduceTwoShot as u16)
            .collect();
        assert_eq!(
            xr.len(),
            278,
            "same count as decode: the shared-expert reduce is phase-agnostic"
        );
        let one: Vec<_> = xr
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16)
            .collect();
        assert_eq!(
            one.len(),
            92,
            "one gathering one-shot per MoE layer, and nothing else"
        );
        assert!(
            one.iter()
                .all(|i| i.i[5] == 896 && i.i[6] == 7168 && i.i[4] != 0),
            "the one-shots are exactly the gathering ones: 7168/tp8 columns per rank"
        );
        assert!(
            xr.iter()
                .filter(|i| i.op == DevOp::XReduceTwoShot as u16)
                .count()
                == 278 - 92,
            "every OTHER T-row partial takes the two-shot path"
        );
        let gates: std::collections::BTreeSet<u32> = xr.iter().map(|i| i.i[3]).collect();
        assert_eq!(gates.len(), xr.len(), "xctr gate ids must not collide");
        let widths: std::collections::BTreeSet<u32> = xr.iter().map(|i| i.i[0]).collect();
        assert!(
            widths.contains(&(512 * 7168)),
            "attention reduces T rows at hidden width"
        );
        assert!(
            widths.contains(&(512 * 3584)),
            "the expert combine reduces at LATENT width"
        );
        // The head axis still shards, and the grouped experts still take the local intermediate.
        let f = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::FlashMlaPrefill as u16)
            .unwrap();
        assert_eq!(f.i[1], 12, "96 heads / tp8");
        assert_eq!(f.i[4], 512, "i4 is n_tok on the prefill arm, NOT nsplit");
        let g = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeGroupGluPf as u16)
            .unwrap();
        assert_eq!(g.i[0], 384, "3072 / tp8");
        assert_eq!(
            g.i[1], 3584,
            "the latent does NOT shard — it is shared by every expert"
        );
    }

    #[test]
    fn folded_gather_prefill_uses_the_generic_twoshot_path_by_default() {
        let _guard = crate::test_env::env_guard();
        let _env = crate::test_env::EnvScope::set(&[("PLOW_XR2_GATHER", "1")]);
        let p = build_full_t(8, 512);
        let xr: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16 || i.op == DevOp::XReduceTwoShot as u16)
            .collect();
        assert_eq!(xr.len(), 278);
        assert!(xr.iter().all(|i| i.op == DevOp::XReduceTwoShot as u16));
        let gathered: Vec<_> = xr.iter().filter(|i| i.i[7] != 0).collect();
        assert_eq!(gathered.len(), 92);
        assert!(gathered
            .iter()
            .all(|i| i.i[6] != 0 && i.i[7] == 896 && i.i[1] * i.i[7] == 7168));
        let gates: std::collections::BTreeSet<u32> =
            xr.iter().flat_map(|i| [i.i[3], i.i[4]]).collect();
        assert_eq!(
            gates.len(),
            2 * xr.len(),
            "both rendezvous gates stay unique"
        );
    }

    #[test]
    fn k3_honors_the_generic_xreduce_cu_cap() {
        let _guard = crate::test_env::env_guard();
        let _env = crate::test_env::EnvScope::set(&[("PLOW_XR_CUS", "32")]);
        let p = build_full_t(8, 512);
        let xr: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::XReduce as u16 || i.op == DevOp::XReduceTwoShot as u16)
            .collect();
        assert_eq!(xr.len(), 278);
        assert!(
            xr.iter().all(|i| i.blocks == 32),
            "PLOW_XR_CUS must cap every K3 collective packet"
        );
    }

    /// The MLA prefill arm forces `nsplit = 1`, and the merge-fold agrees with it.
    ///
    /// Not opportunistic: under a per-token causal bound an early token's later splits cover
    /// nothing, and an empty split emits `l = 0` for the merge to divide by. `i[4]` is the same
    /// slot on both arms and means DIFFERENT THINGS — `nsplit` at decode, `n_tok` at prefill — so
    /// a partial buffer sized for `nsplit = 4` and a packet claiming `n_tok` would agree on
    /// nothing.
    #[test]
    fn the_mla_prefill_arm_forces_one_split() {
        let d = build_full_t(1, 1);
        let dec = d
            .insts
            .iter()
            .find(|i| i.op == DevOp::FlashMlaDecode as u16)
            .unwrap();
        assert_eq!(dec.i[4], 4, "decode keeps its four splits");
        let dfold = d
            .insts
            .iter()
            .find(|i| i.op == DevOp::MlaMergeFold as u16)
            .unwrap();
        assert_eq!(dfold.i[4], 4);
        assert_eq!(dfold.i[0], 1);

        let t = 512u32;
        let p = build_full_t(1, t);
        let fl = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::FlashMlaPrefill as u16)
            .unwrap();
        assert_eq!(fl.i[4], t, "i4 carries n_tok here");
        let fold = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MlaMergeFold as u16)
            .unwrap();
        assert_eq!(
            fold.i[4], 1,
            "nsplit MUST be 1 under a per-token causal bound"
        );
        assert_eq!(
            fold.i[0], t,
            "the token axis folds into n_batch: (b*n_tok + t)*n_head"
        );
        // The partial buffer follows, or the fold reads four splits' worth of stride from one.
        let opart = p
            .tensors
            .iter()
            .find(|x| x.name == "act.pf.o_part")
            .unwrap();
        assert_eq!(opart.bytes, t as u64 * 96 * 512 * 4);
    }

    /// NO NAME IS DECLARED TWICE, on either phase — the precondition `Builder::set_tensor_dedup`
    /// rests on.
    ///
    /// Dedup makes a re-declared name return the EXISTING handle. That is what lets a prefill
    /// bucket share one set of activation buffers across 93 layers, and what lets several programs
    /// share one tensor table. It also means that if two DIFFERENT buffers ever collided on a name,
    /// they would silently become one aliased buffer instead of two. Decode is the case that must
    /// be watched: it declares per-layer scratch under `act.l{l}.`, so a collision there would be a
    /// layer reading its neighbour's activations.
    #[test]
    fn no_two_tensors_share_a_name_on_either_phase() {
        for t in [1u32, 512] {
            let p = build_full_t(1, t);
            let mut seen = std::collections::BTreeSet::new();
            for td in &p.tensors {
                assert!(
                    seen.insert(td.name.clone()),
                    "T={t}: `{}` declared twice",
                    td.name
                );
            }
        }
    }

    /// `K3_NLAYERS` truncation works on BOTH program kinds, and shrinks the tensor table on both.
    #[test]
    fn truncation_works_at_t_rows_too() {
        let c = model_cfg();
        let build = |layers: &[u32], t: u32| {
            let mut b = Builder::new(256);
            emit_k3_model(
                &mut b,
                &c,
                &|l| l != 3,
                layers,
                4096,
                t,
                t,
                1,
                256,
                RowKind::Tokens,
            );
            b.finish()
        };
        for t in [1u32, 128] {
            let short = build(&[0, 1, 2, 3], t);
            let long = build(&(0..12).collect::<Vec<u32>>(), t);
            assert!(
                short.tensors.len() < long.tensors.len(),
                "T={t}: truncation shrinks the table"
            );
            assert!(
                short.insts.len() < long.insts.len(),
                "T={t}: and the program"
            );
            let n = |p: &packet::devbuild::Program, o: DevOp| {
                p.insts.iter().filter(|i| i.op == o as u16).count()
            };
            assert_eq!(
                n(&short, DevOp::AttnRes),
                9,
                "T={t}: two mixes on each of four layers, plus the output mix"
            );
            let flash = if t == 1 {
                DevOp::FlashMlaDecode
            } else {
                DevOp::FlashMlaPrefill
            };
            assert_eq!(
                n(&short, flash),
                1,
                "T={t}: layer 3 is the only MLA layer in 0..3"
            );
            assert_eq!(n(&short, DevOp::KdaStateStepG), 3);
        }
    }

    /// **THE GATHERING COLLECTIVE MUST WAIT FOR THE UP GEMV.**
    ///
    /// `d_xreduce`'s gather reads slot 2 out of every peer WITHOUT reading `out`, so a
    /// collective that is not ordered after the up projection reads whatever slot 2 held —
    /// the PREVIOUS layer's partial. That is finite, plausible and the wrong model, and
    /// nothing on the host sees it: the ranks still agree and the counters still balance.
    ///
    /// Counter ids ARE instruction indices in this emitter (`n_counter == n_inst`), which is
    /// what lets the wait list be read without walking the successor table.
    #[test]
    fn the_gathering_reduce_waits_on_the_up_projection() {
        if !shard_up_proj(8) {
            return;
        }
        let p = build_full(8);
        assert_eq!(
            p.n_counter as usize,
            p.insts.len(),
            "counter id == inst index"
        );
        let ug = p
            .tensors
            .iter()
            .position(|t| t.name == "act.ug_tp")
            .expect("no gather slot");
        let up_ix: Vec<usize> = p
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::Gemv as u16 && d.t[0] as usize == ug)
            .map(|(i, _)| i)
            .collect();
        let xr_ix: Vec<usize> = p
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::XReduce as u16 && d.i[5] != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(up_ix.len(), 92, "one up GEMV per MoE layer");
        assert_eq!(xr_ix.len(), 92);
        for (&xr, &up) in xr_ix.iter().zip(up_ix.iter()) {
            assert!(up < xr, "the up GEMV must precede its reduce");
            let mut waits: Vec<u32> = Vec::new();
            for e in p.stream.iter().filter(|e| e.inst as usize == xr) {
                for k in 0..e.wait_len as usize {
                    waits.push(p.waits[e.wait_ofs as usize + k].id);
                }
            }
            waits.sort_unstable();
            waits.dedup();
            assert!(
                waits.contains(&(up as u32)),
                "the gathering XReduce #{xr} does not wait on the up GEMV #{up} — it would \
                 read the PREVIOUS layer's slot-2 partial. waits: {waits:?}"
            );
        }
    }
}
