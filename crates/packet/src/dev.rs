//! Device ISA — the fixed-stride instruction the persistent on-device
//! interpreter executes. Byte-for-byte mirror of `runtime/common/dev_isa.h`.
//!
//! This is a **different layer** from the wire packet stream in [`crate`]. The
//! wire stream is variable-length and host-decoded; the GPU must not parse it.
//! The host expands the stream ONCE into a `[DevInst]` in device memory, so the
//! interpreter's inner loop is an indexed load and a switch. Nothing is decoded
//! on device.
//!
//! # Execution model
//!
//! One persistent kernel per device, grid == CU count, resident for the life of
//! the model. There is **no per-op launch**. Each workgroup owns one CU and walks
//! its own stream of [`StreamEnt`]:
//!
//! ```text
//! for each (inst, slice) in my stream:
//!     spin until every wait-counter reaches its threshold
//!     execute inst, computing only the `slice`-th share of its work
//!     threadfence (agent scope)
//!     atomically bump every successor counter
//! ```
//!
//! An op spread over N workgroups appears once in `insts` and N times in the
//! streams, with slices `0..N`; its consumers wait on a counter whose threshold
//! is N. "All producers done" therefore falls straight out of the counter
//! protocol — no grid-wide barrier primitive is needed.
//!
//! **Co-residency is the safety condition.** The spin is only sound because
//! grid == CU count, so every workgroup is resident and a producer can never be
//! starved by a spinning consumer. A grid larger than the CU count deadlocks.
//!
//! The layouts here are asserted against the C `_Static_assert`s in `dev_isa.h`
//! by `tests/dev_abi.rs`. Change a field and that test fails — which is the point.

use core::mem::size_of;

/// Sentinel tensor handle: an absent optional operand. Reaches the device op as a
/// null pointer (e.g. `gamma` on Gemma's weightless `v_norm`).
/// Workgroup geometry of the persistent interpreter, mirrored from
/// `runtime/common/dev_isa.h`. The program builder needs this to size dispatches.
pub const WG_WAVES: u32 = 8;
pub const WG_THREADS: u32 = WG_WAVES * 64;

pub const TENSOR_NONE: u32 = 0xFFFF_FFFF;

/// uint32 slots between adjacent counters — one 128-byte cache line.
///
/// A dense counter array puts 32 counters per line, so while 256 workgroups `fetch_add`
/// counter *i*, another 256 poll counter *i+1* on the SAME line and it ping-pongs across 8
/// XCDs on every signal and every poll. Mirrored from `PLOW_CTR_STRIDE` in `dev_isa.h`.
pub const CTR_STRIDE: u32 = 32;

/// Device opcodes. A small closed set: the interpreter's `switch` is the hot
/// path, so this is an ISA, not an extension point.
///
/// The operand slots each op reads are documented on the variant. `t` = tensor
/// handles, `i` = integers, `f` = floats — matching [`DevInst`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DevOp {
    Nop = 0,
    /// `t0=out t1=x t2=gamma?` · `i0=rows i1=feat` · `f0=eps`.
    /// `gamma = TENSOR_NONE` is the weightless RMSNorm (Gemma's `v_norm`).
    RmsNorm = 1,
    /// `t0=rms(f32) t1=x` · `i0=rows i1=feat` · `f0=eps`.
    /// Row RMS scalars only, so [`DevOp::GemmNorm`] can apply the norm in its
    /// prologue and the normalized activation never round-trips through HBM.
    RowRms = 2,
    /// `t0=out t1=x t2=gamma? t3=cos? t4=sin? t5=pos(i32)` ·
    /// `i0=ntok i1=nhead i2=hd i3=out_row0` · `f0=eps`.
    /// `cos = TENSOR_NONE` skips RoPE. `out_row0` lets K/V land directly at a row
    /// offset of the KV cache, so the cache write is not a separate copy.
    HeadNormRope = 3,
    /// `t0=out t1=a t2=b` · `i0=n` · `f0=scale`, computing `(a + b) * scale`.
    /// `scale` absorbs Gemma's per-layer `layer_scalar` on the SECOND residual
    /// add; pass 1.0 for the first.
    Residual = 4,
    /// `t0=out t1=gate t2=up` · `i0=n i1=act` (0 = gelu_tanh, 1 = silu).
    /// Gemma is GeGLU (gelu_tanh), not SwiGLU.
    Glu = 5,
    /// `t0=out t1=table t2=ids(i32)` · `i0=ntok i1=hidden` · `f0=scale`.
    /// `scale` must be the BF16-ROUNDED `sqrt(hidden)`: 73.5 (31B), 62.0 (12B).
    Embed = 6,
    /// `t0=out t1=x` · `i0=n` · `f0=cap`, computing `cap * tanh(x / cap)`.
    SoftCap = 7,
    /// `t0=C t1=A t2=B` · `i0=M i1=N i2=K`, computing `C[M,N] = A[M,K] . B[N,K]^T`.
    /// `B` is `[out_features, in_features]`, as HF stores a Linear weight.
    Gemm = 8,
    /// As [`DevOp::Gemm`], with RMSNorm folded into the A-operand load.
    /// `t3=rms(f32) t4=gamma`.
    GemmNorm = 9,
    /// `t0=C t1=x t2=W t3=rms? t4=gamma?` · `i0=M i1=N i2=K i3=norm`.
    /// Decode path (`M <= 16`): bandwidth-bound, uses no MFMA.
    Gemv = 10,
    /// `t0=O t1=Q t2=K t3=V` ·
    /// `i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd` · `f0=scale`.
    /// `window = 0` is full causal. `hd` must be 256 or 512.
    /// For Gemma `scale = 1.0` — there is NO `1/sqrt(head_dim)`.
    FlashPrefill = 11,
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Q t3=K t4=V t5=kv_len(i32)` ·
    /// `i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window i5=nsplit i6=hd` ·
    /// `f0=scale`.
    FlashDecode = 12,
    /// `t0=O t1=Opart t2=mlpart` · `i0=n_batch i1=n_head i2=nsplit i3=hd`.
    FlashMerge = 13,
    /// As [`DevOp::Gemm`], 64x128 tile.
    GemmSmall = 14,
    /// As [`DevOp::Gemm`], 128x128 tile.
    GemmMed = 15,
    /// `t0=out t1=a t2=b t3=gamma?` · `i0=rows i1=feat` · `f0=eps f1=scale`, computing
    /// `out = (a + RMSNorm(b, gamma)) * scale` — Gemma's sandwich tail in ONE packet instead
    /// of an RMSNORM followed by a RESIDUAL.
    NormResidual = 16,
    /// `t0=out t1=resid t2=a t3=b t4=gamma?` · `i0=rows i1=feat` · `f0=eps`, computing
    /// `resid = a + b` then `out = RMSNorm(resid, gamma)` — the Qwen/Llama PRE-norm tail in ONE
    /// packet: the residual add and the norm that always follows it. `a`/`resid` alias in-place.
    ///
    /// Distinct from [`DevOp::NormResidual`] (Gemma's SANDWICH `a + RMSNorm(b)`): here the norm is
    /// over the SUM, and BOTH the updated residual stream and its norm are written. Merges a
    /// RESIDUAL packet and an RMSNORM packet — two global gates, each a single-workgroup decode op.
    AddNorm = 21,
    /// `t0=part(u64[blocks]) t1=x` · `i0=n`. Per-block partial of a greedy argmax, packed as
    /// `(ordered_bf16_key << 32) | ~index` so a plain unsigned max does the whole reduction.
    Argmax = 17,
    /// `t0=ids(i32) t1=part` · `i0=blocks`. Folds the partials and writes the token id
    /// straight into the tensor the NEXT step's [`DevOp::Embed`] reads — so a sampled token
    /// never leaves the GPU.
    ArgmaxFin = 18,
    /// `t0=fu t1=x t2=W_gate t5=W_up` · `i0=M i1=N i2=K i5=act`, computing
    /// `fu = act(W_gate @ x) * (W_up @ x)` — gate and up in ONE GEMV, with the GLU applied in
    /// the **epilogue**, as every BLAS does it (cuBLASLt/CK/hipBLASLt).
    ///
    /// Output-stationary: the workgroup that owns column `n` computes both halves of it and
    /// applies the GLU exactly once. Nothing is replicated. This deletes the [`DevOp::Glu`]
    /// packet *and* merges two GEMVs into one.
    ///
    /// Fuse into the producer's EPILOGUE, never into the consumer's PROLOGUE: folding the GLU
    /// into the *down* GEMV's staging instead was measured at a **39x loss**, because `fu` is
    /// down's K dimension and all 256 of its workgroups would recompute the whole GLU.
    ///
    /// Legal only when `M*K` fits LDS (`GM_LDS_HALVES`); plowc falls back to the unfused triple.
    GemvGlu = 19,
    /// `t0=fu t1=x t2=W_gate t5=W_up` · `i0=M i1=N i2=K i5=act`, computing
    /// `fu = act(W_gate @ x) * (W_up @ x)` — the prefill twin of [`DevOp::GemvGlu`].
    ///
    /// Same tile, same registers, same MFMA count as a plain [`DevOp::Gemm`]: the accumulator's
    /// `SN` axis selects **gate vs up** instead of a column block, so both halves of an output
    /// element land in the same lane and the epilogue needs no shuffle. A workgroup emits
    /// `BN/2` fused columns for the MFMA count it used to spend on `BN` raw ones — the same
    /// arithmetic, since every output needs both halves anyway.
    ///
    /// `gt`/`ut` never reach HBM, and the [`DevOp::Glu`] packet and its gate disappear.
    GemmGlu = 20,
    /// `t0=q_out t1=x t2=W_q t3=k_out t4=W_k t5=v_out t6=W_v` · `i0=M i1=Nq i2=K i3=Nk i4=Nv`,
    /// computing all three attention projections `q=W_q@x`, `k=W_k@x`, `v=W_v@x` in ONE GEMV.
    ///
    /// The three share the same `x` and `K`, so their output columns concatenate into one
    /// `N=Nq+Nk+Nv` sweep: column `n<Nq` writes `q_out`, `n<Nq+Nk` writes `k_out`, else `v_out`.
    /// Decode-only. Replaces three separate GEMVs on disjoint CU sets (`split3`) with one op that
    /// fills every CU uniformly — deleting two counter gates per layer and the cross-op arrival
    /// imbalance of the 171/42/43 partition. Same math, same bytes as the three GEMVs.
    ///
    /// Structurally a single-stream GEMV (one weight row per column), so it keeps the plain
    /// GEMV's register budget, NOT the doubled one of [`DevOp::GemvGlu`]. Legal only when
    /// `M*K` fits LDS; plowc emits it only in decode (`M=1`), where it always does.
    GemvQkv = 22,
    /// `t0=C t1=x t2=W(fp8) t5=w_scale(f32[N])` · `i0=M i1=N i2=K i4=a_row0`. The fp8 (w8a16) twin
    /// of [`DevOp::Gemv`]: the weight row is `uint8[K]` OCP e4m3, so decode streams HALF the bytes
    /// (~2x the bandwidth-bound decode roofline). Each fp8 is converted to bf16 on load and the
    /// existing bf16 `fdot2` reduction is unchanged; the per-output-channel dequant `w_scale[n]` is
    /// applied ONCE in the epilogue on the wave sum, never per element. Decode-only.
    GemvFp8 = 30,
    /// `t0=fu t1=x t2=W_gate(fp8) t5=W_up(fp8) t3=gate_scale(f32[N]) t4=up_scale(f32[N])` ·
    /// `i0=M i1=N i2=K i5=act`. The fp8 twin of [`DevOp::GemvGlu`]: gate|up in ONE pass with the
    /// GLU applied in the epilogue, both weight streams fp8 e4m3 with their own per-channel scale.
    GemvGluFp8 = 31,
    /// `t0=xq(fp8) t1=x(bf16) t2=a_scale(f32[M])` · `i0=M i1=K`. Per-row (per-token) fp8 activation
    /// quant — the w8a8 prefill's activation half. `a_scale[m] = rowmax|x[m,:]|/448`, `xq[m,k] =
    /// round_e4m3(x[m,k]/a_scale[m])`. Emitted once per activation, reused by every fp8 GEMM.
    QuantFp8 = 32,
    /// `t0=C t1=A(fp8) t2=B(fp8) t3=a_scale(f32[M]) t4=w_scale(f32[N])` · `i0=M i1=N i2=K i4=a_row0`.
    /// The fp8 (w8a8) prefill twin of [`DevOp::Gemm`]: BOTH operands fp8 e4m3. NOTE the 2x-rate MFMA
    /// on gfx950 is the WIDE-K `mfma_scale_f32_32x32x64_f8f6f4` (measured 2x bf16), NOT the K16
    /// `mfma_f32_32x32x16_fp8_fp8` the design first named (measured 1x = bf16). The f32 accumulator is
    /// dequantized `acc*a_scale[m]*w_scale[n]` in the epilogue; its layout is identical to the bf16
    /// 32x32 MFMA, so the epilogue carries over. See op_gemm.h d_gemm_fp8_t.
    GemmFp8 = 33,
    /// 128x128 fp8 tile — the medium-tile twin of [`DevOp::GemmMed`]. Same operands as [`DevOp::GemmFp8`].
    GemmMedFp8 = 34,
    /// 64x128 fp8 tile — the small-tile twin of [`DevOp::GemmSmall`]. Same operands as [`DevOp::GemmFp8`].
    GemmSmallFp8 = 35,
    /// `t0=fu t1=A(fp8) t2=Wg(fp8) t5=Wu(fp8) t3=a_scale t4=g_scale t6=u_scale` · `i0=M i1=N i2=K i5=act`.
    /// The fp8 (w8a8) twin of [`DevOp::GemmGlu`]: fp8 gate|up in ONE pass, act(gate)*up in the epilogue.
    GemmGluFp8 = 36,
    /// `t0=out t1=resid t2=a t3=b t4=gamma_b? t5=gamma_n?` · `i0=rows i1=feat` · `f0=eps f1=scale`,
    /// computing `resid = (a + RMSNorm(b, gamma_b)) * scale` then `out = RMSNorm(resid, gamma_n)` —
    /// Gemma's SANDWICH tail AND the norm that follows it, in ONE packet (Experiment N1).
    ///
    /// The narrow→narrow successor to [`DevOp::AddNorm`]: it fuses a [`DevOp::NormResidual`] and the
    /// [`DevOp::RmsNorm`] that re-reads its output, deleting a global gate and a full HBM round trip
    /// (`resid` is held in registers across the second reduction instead of round-tripping). Both are
    /// single-workgroup decode ops, so the fusion replicates nothing. `a`/`resid` alias in-place.
    ///
    /// Bit-exact to the pair: `resid` is rounded to bf16 before the second reduction, reproducing
    /// NORM_RESIDUAL's store and RMSNORM's reload.
    NormResidualNorm = 23,

    // ===== Cross-GPU (tensor-parallel) tile-graph ops. =========================
    // New opcodes AFTER main's last (23), no collision. Names mirror the generic
    // infervisor RDMA-family variants (see `crate::lib` `RDMA`), so the two ABIs
    // converge. Their wait/succ counters live in the SYSTEM-scope [`crate::dev`]
    // `xctr` region, selected by [`SE_XCTR`] on the stream entry. Only these ops
    // touch peer VRAM; weights/KV/residual never cross the fabric (tp-design §7).
    /// All-reduce, the TP primitive (tp-design §8a). One-shot: each rank's producing
    /// GEMV (o_proj/down) has published its partial H-vector into its own peer_scratch
    /// slot and system-signalled every peer's `xctr`; this is the CONSUME half — waits
    /// on N partials (SE_XCTR gate) then sums the N peer slots into a local full
    /// H-vector, f32 accumulate rounded to bf16.
    /// `t0=out` · `i0=H i1=n_gpu i2=slot(byte offset into peer_scratch)`.
    XReduce = 24,
    /// Reduce-scatter half of the symmetric all-reduce decomposition. Kept defined for
    /// CP / larger worlds; not emitted on the N<=8 decode path (one-shot `XReduce` wins).
    XReduceScatter = 25,
    /// All-gather half of the symmetric decomposition. See [`DevOp::XReduceScatter`].
    XAllGather = 26,
    /// Context-parallel cross-GPU flash LSE-merge (tp-design §8c, §9). Folds N peers'
    /// `(O_partial,m,l)` over their KV-position shards into the replicated attention
    /// output. ABI mirrors [`DevOp::FlashMerge`] with `t1..` in peer_scratch + xctr
    /// gates. STUB (dispatch present, body deferred to the CP phase).
    XFlashMerge = 27,
    /// Sharded lm_head argmax-merge (tp-design §8d). lm_head is vocab-column parallel:
    /// each rank argmaxes its `V/N` logits to a packed `(key,idx)` u64; this reads the N
    /// peer packed maxima and folds them to the global token id, written into every
    /// rank's `in.ids`. `t0=ids(i32) t1=local_part(u64)` · `i0=n_gpu i2=slot`.
    XArgmaxFin = 28,
    /// TWO-SHOT all-reduce (reduce-scatter + all-gather) for the LARGE prefill
    /// [T,hidden] message — bandwidth-optimal where the one-shot [`DevOp::XReduce`] is
    /// fabric-bound (plans/tp-prefill.md §4). Fused + self-contained like `XReduce`:
    /// partitions the flat [n] result into N contiguous slices, reduces THIS rank's
    /// owned slice from every peer's partial (writing it in-place, peer-visible), then
    /// gathers every peer's reduced slice into the local full vector. Two internal xctr
    /// rendezvous bracket the phases. Fabric ≈ 2(N−1)/N·msg/rank vs one-shot's (N−1)·msg.
    /// Bit-identical result to one-shot (same f32-acc, r=0..N−1 order). DECODE keeps the
    /// one-shot (its tiny [1,hidden] message is latency-, not bandwidth-, bound).
    /// `t0=out` · `i0=n(=t·hidden) i1=n_gpu i2=slot(byte offset) i3=gate_rs i4=gate_ag`.
    XReduceTwoShot = 29,
    /// FP8 (e4m3) KV-cache twin of [`DevOp::HeadNormRope`]: writes K/V as `uint8[...]` e4m3 with a
    /// per-(token,kv_head) f32 dequant scale, halving the KV footprint and the decode KV stream.
    /// `t0=out(uint8) t6=scale(f32[kv_head][ctx])`; `t1..t5,i,f,j` as [`DevOp::HeadNormRope`].
    HeadNormRopeFp8 = 37,
    /// FP8 KV twin of [`DevOp::FlashDecode`]: reads the e4m3 cache (HALF the HBM bytes) and applies
    /// the per-row scale. `t3=K(fp8) t4=V(fp8) t6=k_scale t7=v_scale`; else as [`DevOp::FlashDecode`].
    FlashDecodeFp8 = 38,
    /// FP8 KV twin of [`DevOp::FlashPrefill`]: dequantizes the e4m3 cache at the LDS stage, so the
    /// MFMA is unchanged. `t3=K(fp8) t4=V(fp8) t6=k_scale t7=v_scale`; else as [`DevOp::FlashPrefill`].
    FlashPrefillFp8 = 39,

    // ===== MoE data-dependent counter-gate ops (plans/moe-plow-design.md §3, =====
    // plans/moe-ep-kernels.md §2-§3). Opcodes in the HIGH free range 40+ so they do
    // NOT collide with tp's collectives (24-29) or the fp8 merge (30+). These are the
    // FIRST ops whose BODY branches on a runtime buffer (the routing table): the
    // counter DAG stays static (deadlock-free, `executed == total`), and each expert
    // packet ALWAYS signals its completion counter whether it computed or skipped —
    // so the interpreter's gate/signal loop is UNCHANGED (interp.hip:600-720). The
    // conditionality lives entirely inside `plow_exec`.
    /// MoE router (`moe-ep-kernels.md §2b`). One packet, per token: GEMV `x·Wr` over
    /// `n_exp` experts → per-expert `score` (sigmoid or softmax) → k-pass masked argmax
    /// top-k with **lowest-expert-id tie-break** (bit-exactness linchpin) → optional
    /// `norm_topk` (renormalise the k gates to sum 1) → `×route_scale` → writes the
    /// routing table `[k]` of `(u32 expert_id, f32 gate)`, padding unused slots with the
    /// `EXPERT_UNUSED` sentinel. Its completion counter is the data-dependent gate the K
    /// expert slots wait on.
    /// `t0=routing_table(out) t1=x t2=Wr [t3=e_score_correction_bias f32[n_exp]]` ·
    /// `i0=H i1=n_exp i2=k i3=flags` · `f0=route_scale`.
    /// `flags` bit0 = scoring (1 sigmoid, 0 softmax), bit1 = norm_topk, bit2 = apply t3
    /// `e_score_correction_bias` to the top-k SELECTION only (DeepSeek/GLM noaux_tc — the
    /// gate value stays the raw unbiased score).
    MoeRouter = 40,
    /// MoE expert gate/up GEMV — the common expert segment's first half
    /// (`moe-ep-kernels.md §3a`), one instance per top-k slot. Reads
    /// `routing_table[slot].expert_id`; if `>= n_exp` (sentinel) it **skips** (writes
    /// nothing, streams zero weight bytes) and the interpreter still signals its counter;
    /// else it resolves `wbase = expert_weight_table[expert_id]` (a two-level indirection
    /// through a table of device pointers) and runs the fused `act(gate·x)·(up·x)` GEMV
    /// (identical arithmetic to [`DevOp::GemvGlu`]) into per-slot scratch `fu[slot]`.
    /// `t0=fu(out,[k,I_moe]) t1=x t2=routing_table t3=expert_weight_table` ·
    /// `i0=slot i1=I_moe(N) i2=H(K) i3=n_exp i5=act`.
    MoeExpertGlu = 41,
    /// MoE expert down GEMV — the common expert segment's second half
    /// (`moe-ep-kernels.md §3a`). Same sentinel skip; else `wbase.down`, runs the down
    /// projection `W_down·fu[slot]`, multiplies by `routing_table[slot].gate`, and writes
    /// the gate-scaled partial `part[slot]`. On the skip path it zeroes `part[slot]` so the
    /// combine sums a deterministic zero.
    /// `t0=part(out,[k,H]) t1=fu t2=routing_table t3=expert_weight_table` ·
    /// `i0=slot i1=H(N) i2=I_moe(K) i3=n_exp`.
    MoeExpertDown = 42,
    /// MoE combine (`moe-ep-kernels.md §3b`) — the deterministic gather-combine. Waits on
    /// all `k` expert-down slots + the shared expert, then
    /// `out = residual + shared + Σ_{j=0..k-1} part[j]`, **f32 accumulate in fixed slot
    /// order rounded to bf16** (independent of which expert finished first — the MoE
    /// bit-exactness obligation). `shared = TENSOR_NONE` for a 0-shared-expert config.
    /// `t0=out t1=residual t2=shared? t3=part_base([k,H])` · `i0=H i1=k`.
    MoeCombine = 43,

    // ===== Block-fp8 (DeepSeek/GLM weight_block_size [128,128]) weight-stream ops. =====
    // Free band 44-49. The weight is e4m3 and dequant reads a per-[128 out][128 K] f32
    // scale grid (`ceil(N/128) x ceil(K/128)`, row-major) folded into the K-reduction
    // per 128-K block — NOT the per-channel epilogue scale of the 30-39 fp8 ops. A lane's
    // 16 consecutive fp8 lie within one 128-block, so one FMA/chunk folds the block scale
    // (no cross-lane reshuffle). x stays bf16 (w8a16 decode weight-stream path).
    /// Block-fp8 decode GEMV — the block-scale twin of [`DevOp::GemvFp8`]. `t5` is the
    /// `[ceil(N/128)][ceil(K/128)]` f32 scale grid instead of a per-channel `f32[N]`.
    /// `t0=C t1=x t2=W(fp8) t5=w_scale(grid)` · `i0=M i1=N i2=K i4=a_row0`.
    GemvFp8Blk = 44,
    /// Block-fp8 expert gate/up — the block-scale twin of [`DevOp::MoeExpertGlu`]. Weight
    /// bases from `expert_weight_table[eid][0,1]` (fp8 rows); block-scale grid bases from a
    /// parallel `expert_scale_table[eid][0,1]` (`[I_moe/128][H/128]` f32 each). Sentinel skip.
    /// `t0=fu t1=x t2=routing_table t3=expert_weight_table t4=expert_scale_table` ·
    /// `i0=slot i1=I_moe i2=H i3=n_exp i5=act`.
    MoeExpertGluFp8Blk = 45,
    /// Block-fp8 expert down — the block-scale twin of [`DevOp::MoeExpertDown`]. `Wd` base from
    /// `expert_weight_table[eid][2]`, scale grid (`[H/128][I_moe/128]`) from
    /// `expert_scale_table[eid][2]`. Sentinel skip zeroes the partial.
    /// `t0=part t1=fu t2=routing_table t3=expert_weight_table t4=expert_scale_table` ·
    /// `i0=slot i1=H i2=I_moe i3=n_exp`.
    MoeExpertDownFp8Blk = 46,
    /// Block-fp8 DENSE SwiGLU gate/up — the non-routed twin of [`DevOp::MoeExpertGluFp8Blk`] on
    /// NAMED weights (GLM-5.2 dense layers 0-2): no routing table, no sentinel, no gate multiply.
    /// `fu[n] = act(gate_n·x) * (up_n·x)`. `Wg,Wu` e4m3 `[N][K]`; `Sg,Su` block-scale grids
    /// (`[N/128][K/128]` f32, `weight_scale_inv`). The dense DOWN projection reuses [`DevOp::GemvFp8Blk`].
    /// `t0=fu t1=x t2=Wg t5=Wu t3=Sg t4=Su` · `i0=N(intermediate) i1=K(hidden) i5=act`.
    DenseGluFp8Blk = 47,

    /// GROUPED block-fp8 expert gate/up — ONE packet loops all `k` top-k slots that
    /// [`DevOp::MoeExpertGluFp8Blk`] did one-per-packet. Bit-identical `fu` (same per-output
    /// `wave_dot_fp8_blk`, same slot layout); the win is per-op overhead (one counter edge + one
    /// interp dispatch instead of `k`). Sentinel/EP-non-local (null base) slots leave `fu` unwritten.
    /// `t0=fu[k,I_moe] t1=x t2=routing_table t3=expert_weight_table t4=expert_scale_table` ·
    /// `i0=k i1=I_moe i2=H i3=n_exp i5=act`.
    MoeGroupGluFp8Blk = 48,
    /// GROUPED block-fp8 expert down — ONE packet loops all `k` slots that
    /// [`DevOp::MoeExpertDownFp8Blk`] did one-per-packet. Bit-identical gate-scaled `part`; sentinel /
    /// EP-non-local slots zero the partial so the fixed-slot combine sums a deterministic zero.
    /// `t0=part[k,H] t1=fu t2=routing_table t3=expert_weight_table t4=expert_scale_table` ·
    /// `i0=k i1=H i2=I_moe i3=n_exp`.
    MoeGroupDownFp8Blk = 49,

    // ===== DeepSeek MLA (Multi-head Latent Attention) — flash READ path. =====
    // Opcodes 50+, mirroring `runtime/common/dev_isa.h` (sparse-attn-design.md §4
    // renumbered into the free band above MoE). The WRITE path reuses existing ops
    // (GEMV/GEMM for the c_kv/k_rope down-projections, HEADNORM_ROPE for the shared
    // rope key). These carry the latent geometry as compile-time-fixed operands; the
    // interpreter dispatches `d_flash_mla_decode<512,64,8>` (DK=kv_lora_rank=512,
    // DR=qk_rope_dim=64, GF=8 head-fusion). FLASH_MLA_MERGE is not a new opcode — the
    // latent-wide LSE merge reuses [`DevOp::FlashMerge`] at `hd=512`.
    /// MLA latent flash decode (`sparse-attn-design.md §2.3`). Per head g, kv row j:
    /// `score[g] = q_abs[g]·C_kv[j] + q_rope[g]·K_rope[j]`, online softmax, then
    /// `oacc[g] += p[g]·C_kv[j]` (PV accumulates on the latent). Emits latent-wide
    /// `(O_partial,m,l)` for [`DevOp::FlashMerge`] at `hd=512`; [`DevOp::OUvFold`]
    /// folds to `v_head_dim` after the merge.
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len(i32)` ·
    /// `i0=n_batch i1=n_head i2=kv_stride i3=window i4=nsplit i5=kv_mask` · `f0=scale`.
    FlashMlaDecode = 50,
    /// MLA prefill MFMA twin — not yet built (reserved so the ABI is stable).
    FlashMlaPrefill = 51,
    /// Per-head `W_uv` fold (`sparse-attn-design.md §2.5`): `o[b][h][v] =
    /// Σ_l O_latent[b][h][l]·W_uv[h][l][v]` — the O(n_q) query-side epilogue that folds
    /// the merged latent accumulator down to `v_head_dim`. `W_uv` is `[n_head][DK][V]`
    /// (l-major), the reduction runs `l=0..DK` in order (bit-exact to reference).
    /// `t0=O t1=Olat t2=Wuv` · `i0=n_batch i1=n_head i2=V`.
    OUvFold = 52,

    // ===== Sparse top-k DSA (sparse-attn-design.md §3). =====
    /// On-device top-k KV selection (`sparse-attn-design.md §3.2`) — the attention twin
    /// of the MoE routing table. Scores the indexer queries against the index keys,
    /// keeps the top-k KV row indices per query (deterministic lowest-index tie-break),
    /// writes the `idx` table [`DevOp::FlashGatherDecode`] reads. Reserved; body TBD.
    AttnSelect = 53,
    /// Gathered flash decode (`sparse-attn-design.md §3.3`) — the MLA latent flash
    /// reading ONLY the `top_k` selected latent rows via the `idx` table (the
    /// `GATHER=true` instantiation of `d_flash_mla_decode`). Everything else (absorbed
    /// score, online softmax, split, merge) is byte-identical to the dense MLA decode.
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len(i32)
    /// t7=idx(i32)` · `i0=n_batch i1=n_head i2=kv_stride i4=nsplit i5=kv_mask i6=top_k` ·
    /// `f0=scale`.
    FlashGatherDecode = 54,
    /// Gathered flash prefill — reserved, not yet built.
    FlashGatherPrefill = 55,

    /// ROUTER TOP-K tail (the router SPLIT). The score matmul `logit = x·Wr` is now the ordinary
    /// multi-CU wave-cooperative [`DevOp::Gemv`] over the 256 experts (was a 1141us single-CU scalar
    /// dot); this cheap 1-CU op does the score transform (sigmoid|softmax) + `e_score_correction_bias`
    /// on SELECTION only + group-limited top-k with lowest-id tie-break + norm_topk + route_scale over
    /// the precomputed logits. Byte-for-byte the selection logic of [`DevOp::MoeRouter`], minus the GEMV.
    /// `t0=table t1=logit(bf16[n_exp]) t3=bias(f32[n_exp] or 0)` · `i1=n_exp i2=k i3=flags` · `f0=route_scale`.
    MoeRouterTopk = 56,

    /// FUSED MLA merge + W_uv fold (`d_mla_merge_fold`) — replaces the [`DevOp::FlashMerge`] +
    /// [`DevOp::OUvFold`] pair on the MLA decode path. Online-softmax-merges the `nsplit` latent
    /// partials into `olat[DK]` in LDS, then folds `olat @ W_uv[head]` straight to `o[head][V]` —
    /// killing the separate merge pass, its `Olat` HBM round-trip, and one dependency gate.
    /// Validated (rms ~0.004 vs the merge→fold sequence); ~1.1-1.24x on the MLA chain, composing
    /// with the ctx-scaled nsplit to 1.59x at 32k. `t0=O(v_head) t1=Opart(f32) t2=mlpart(f32)
    /// t3=Wuv` · `i0=n_batch i1=n_head i2=V i4=nsplit`.
    MlaMergeFold = 57,

    // ===== DSA lightning indexer (GLM-5.2 GlmMoeDsa; sparse-attn-design.md §3.1, arXiv 2512.02556
    // eq.1). The two ops that PRODUCE the `idx` table [`DevOp::FlashGatherDecode`] gathers over —
    // scoring the pre-projected/pre-RoPE'd indexer q/k, then the top-k radix select. =====
    /// Lightning-indexer SCORE (`d_index_score`/`d_index_score_fast`). For every KV position `t`,
    /// `score[b][t] = Σ_h w[b][h]·ReLU(q_idx[b][h]·k_idx[b][t])` (scale folded in; selection is
    /// scale-invariant). `q_idx` (`[b][HI][DI]`) and `k_idx` (`[b][ctx][DI]`) arrive already
    /// projected (wq_b/wk fp8 GEMV), k_norm'd (LayerNorm+bias) and interleaved-RoPE'd; `w` (`[b][HI]`)
    /// is `weights_proj·x`. Grid-strided over all CUs. `t0=Score(f32) t1=Qidx(bf16) t2=Kidx(bf16)
    /// t3=W(bf16) t4=kv_len(i32)` · `i0=n_batch i1=index_heads i2=kv_stride i3=index_head_dim` · `f0=scale`.
    IndexScore = 58,
    /// Lightning-indexer top-k SELECT (`d_index_select_coop`) — ONE cooperative launch of exactly
    /// `n_cu` co-resident workgroups that grid-sync (fenceless L2-atomic barrier) between 7 radix
    /// passes over the monotone packed key `(ordered_bits(score)<<20)|(len-1-t)`, emitting the exact
    /// `top_k` highest-score positions (lowest-index tie-break) into `idx`. `t0=idx(i32) t1=Score(f32)
    /// t2=gHist(u32[7*256]) t3=gCtl(u32[3])` · `i0=len i1=top_k`. Host zeroes gHist/gCtl once; the
    /// kernel leaves them clean for relaunch.
    IndexSelect = 59,

    /// LayerNorm WITH bias + mean-subtract over `feat` (`d_layernorm_bias`) — the DSA indexer key-norm
    /// (`indexer.k_norm`, nn.LayerNorm(index_head_dim=128, eps=1e-6, bias=True); the only non-RMS norm
    /// in GLM-5.2). `y=(x-μ)·rsqrt(var+eps)·γ+β`. `t0=out t1=x t2=gamma t3=beta` · `i0=rows i1=feat
    /// i3=out_row0` · `f0=eps`. `out_row0` writes the current token's index-key into its [ctx][DI] cache.
    LayerNorm = 60,

    /// Gemma-4 26B-A4B bf16 sparse-MoE DECODE router (`d_moe_router_gemma`,
    /// `plans/rtx-08-gemma4-moe-26b.md`). Weightless-RMS(resid) → `·scale[H]·root` →
    /// softmax(proj@·) → top-k (lowest-id tie) → norm_topk → `·per_expert_scale`. Writes
    /// `routing_table[k]={u32 id, f32 gate}`. ONE block. `t0=table t1=resid t2=proj t3=scale
    /// t4=per_expert_scale` · `i0=H i1=n_exp i2=k` · `f0=root(=H^-0.5) f1=eps`.
    MoeRouterGemma = 61,
    /// Gemma-4 fused-gate_up expert GLU (`d_moe_expert_glu_gemma`): `fu[slot][n] =
    /// gelu_tanh(gate_e·x)·(up_e·x)`, one warp per output. `t0=fu([k,I]) t1=x t2=table t3=ewt`
    /// (ewt = 2 u64/expert {gate_up base, down base}) · `i0=k i1=I_moe i2=H i3=n_exp`.
    MoeExpertGluGemma = 62,
    /// Gemma-4 expert down (`d_moe_expert_down_gemma`): `part[slot][h] = gate·(down_e[h]·fu)`,
    /// f32, one warp per output. `t0=part([k,H],f32) t1=fu t2=table t3=ewt` · `i0=k i1=H
    /// i2=I_moe i3=n_exp`.
    MoeExpertDownGemma = 63,
    /// Gemma-4 combine (`d_moe_combine_gemma`): `moe[h] = Σ_slot part[slot][h]` (f32, fixed
    /// slot order) → bf16. `t0=moe t1=part([k,H])` · `i0=H i1=k`.
    MoeCombineGemma = 64,
    /// Per-output-channel e4m3 twin of [`DevOp::MoeExpertGluGemma`]. `ewt` points at fused
    /// `[2*I,H]` fp8 expert weights and `est` at `[2*I]` f32 row scales.
    /// `t0=fu t1=x t2=table t3=ewt t4=est` · `i0=k i1=I i2=H i3=n_exp`.
    MoeExpertGluGemmaFp8 = 65,
    /// Per-output-channel e4m3 twin of [`DevOp::MoeExpertDownGemma`]. `ewt` points at `[H,I]`
    /// fp8 expert weights and `est` at `[H]` f32 row scales.
    /// `t0=part t1=fu t2=table t3=ewt t4=est` · `i0=k i1=H i2=I i3=n_exp`.
    MoeExpertDownGemmaFp8 = 66,
    /// Gemma-4 router SCORE half. Each workgroup recomputes the small weightless-RMS scalar,
    /// then its eight warps score eight experts with coalesced bf16 loads and a warp reduction.
    /// The separate [`DevOp::MoeRouterGemmaTopk`] consumer waits for every score workgroup.
    /// `t0=score(f32[n_exp]) t1=resid t2=proj t3=scale` · `i0=H i1=n_exp` ·
    /// `f0=root(=H^-0.5) f1=eps`.
    MoeRouterGemmaScore = 67,
    /// Gemma-4 router softmax/top-k tail. Reads the f32 scores produced by
    /// [`DevOp::MoeRouterGemmaScore`], preserves the lowest-id tie break and fixed expert/slot
    /// accumulation order, applies norm_topk and the per-expert scale, and writes the routing
    /// table. ONE block. `t0=table t1=score(f32[n_exp]) t2=per_expert_scale` ·
    /// `i1=n_exp i2=k`.
    MoeRouterGemmaTopk = 68,
    /// Experimental fast twin of [`DevOp::MoeRouterGemmaScore`]. It uses ordinary per-lane f32
    /// accumulation followed by a warp reduction. Loads are coalesced and substantially cheaper,
    /// but the changed reduction association is not bit-identical; packets may opt in only for
    /// whole-model correctness/performance experiments. Operands are identical to opcode 67.
    MoeRouterGemmaScoreFast = 69,
    /// Fused combine + RMSNorm + residual-add for Gemma-4 MoE. Replaces the 3-op tail chain
    /// (MoeCombineGemma → RmsNorm → Residual) with a single counter-gated op.
    /// `t0=out t1=part(f32[k,H]) t2=resid t3=gamma` · `i0=H i1=k` · `f0=eps`.
    MoeCombineNormGemma = 70,
    /// Fused pre-FFN-norm-2 + expert GLU. Same as [`DevOp::MoeExpertGluGemma`] but
    /// takes raw residual + gamma and computes RMSNorm inline, eliminating a separate
    /// RmsNorm packet. `t0=fu t1=resid t2=table t3=ewt t4=gamma` ·
    /// `i0=k i1=I i2=H i3=n_exp` · `f0=eps`.
    MoeExpertGluNormGemma = 71,
    /// Fused MoE layer tail: ([`DevOp::MoeCombineNormGemma`] → NormResidualNorm) in one
    /// counter-gated packet — combine + post_ffn norm, sandwich residual, and the NEXT
    /// sublayer's input norm. Bit-exact to the pair (intermediates rounded to bf16).
    /// `t0=hn t1=x(in/out) t2=part(f32[k,H]) t3=h1 t4=g_pf2 t5=g_po t6=gn` ·
    /// `i0=H i1=k` · `f0=eps f1=layer_scalar`.
    MoeCombineResidNormGemma = 72,

    // ===== Gemma-4 26B-A4B bf16 grouped-MoE PREFILL ops (plans/p9-26b-prefill-moe.md). =====
    // Token-sorted grouped expert GEMM for T>1. Ids 73+ (71/72 free; 72 reserved elsewhere).
    // Built only in the prefill (_pf) interpreter object.
    /// T-token router: block-per-token loop of the exact decode router. Writes
    /// `routing_table[token*k + j] = {u32 eid, f32 gate}`, bit-identical per token to decode.
    /// `t0=table t1=resid([T,H]) t2=proj t3=scale t4=per_expert_scale` ·
    /// `i0=H i1=n_exp i2=k i3=T` · `f0=root(=H^-0.5) f1=eps`.
    MoeRouterGemmaPf = 73,
    /// Align/sort: ONE block. Histogram T*k routing slots by expert, padded prefix to BM=128 tile
    /// boundaries, scatter (token,slot,gate) into expert-contiguous gathered rows (pad rows get
    /// `EXPERT_UNUSED`). `meta` (i32): `[0,n_exp)` rowoff, `[n_exp,2n_exp)` cnt,
    /// `[2n_exp,3n_exp+1)` tile_prefix (`tile_prefix[n_exp]` = total_tiles).
    /// `row_token` = source token per gathered row (UNUSED for pad); `row_partidx` = token*k+slot
    /// (destination row in part[T*k,H]; UNUSED for pad); `row_gate` = the slot gate.
    /// `t0=meta(i32) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)` ·
    /// `i0=T i1=n_exp i2=k`.
    MoeAlignGemmaPf = 74,
    /// Grouped gate/up GEMM + GeGLU. Flat tile list over (workitem→(expert,m_tile))×n_tiles; A
    /// gathered from `xn2` via `row_token`; B = fused `ewt[e*2+0]` (`[2*I,H]`); GeGLU epilogue to
    /// `fu_gathered[row*I_moe+n]`. Reuses the tiled-GEMM body (cp.async, m16n8k16, 128x128).
    /// `t0=fu_g t1=xn2([T,H]) t2=ewt t3=meta t4=row_token` · `i0=I_moe(N) i1=H(K) i2=n_exp i5=act`.
    MoeGroupGluGemmaPf = 75,
    /// Grouped down GEMM + gate-scale + scatter. A = `fu_gathered` (contiguous), B = `ewt[e*2+1]`
    /// (`[H,I_moe]`), N=H K=I_moe; epilogue `*row_gate` and SCATTERS to `part[row_partidx*H+h]`
    /// via `row_partidx`; pad rows skipped.
    /// `t0=part([T,k,H] f32) t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate` ·
    /// `i0=H(N) i1=I_moe(K) i2=n_exp`.
    MoeGroupDownGemmaPf = 76,
    /// T-row combine + sandwich: block-per-token loop of [`DevOp::MoeCombineNormGemma`].
    /// `out[t] = RMSNorm(Σ_slot part[t][slot], gamma) + h1[t]`.
    /// `t0=out t1=part([T,k,H]) t2=h1([T,H]) t3=gamma` · `i0=H i1=k i2=T` · `f0=eps`.
    MoeCombineNormGemmaPf = 77,

    /// SplitZip (bf16 lossless) DECODE GEMV — twin of [`DevOp::Gemv`]. The weight is one
    /// self-describing compressed blob (header {nesc, exp_base} + lo|cd|eoff|epos|eval; see
    /// `op_gemm.cuh` `sz_blob`). Output BIT-IDENTICAL to `Gemv`; only the HBM weight bytes shrink
    /// ~1.33x. `t0=C t1=x t2=blob` · `i0=M i1=N i2=K`.
    GemvSz = 78,
    /// SplitZip fused gate|up DECODE GEMV — twin of [`DevOp::GemvGlu`], two compressed blobs.
    /// `t0=fu t1=x t2=gblob t3=ublob` · `i0=M i1=N i2=K i5=act`.
    GemvGluSz = 79,

    /// E5 (rtx-19): lm_head GEMV with the greedy-argmax epilogue FUSED in (flag PLOW_FUSE_ARGMAX).
    /// Twin of [`DevOp::Gemv`] (bf16, M=1 decode), but each block reduces its owned vocab slice to
    /// one packed-u64 argmax partial `part[block]` IN the GEMV epilogue — reproducing the
    /// [`DevOp::SoftCap`]→[`DevOp::Argmax`] value chain bit-for-bit (`f0=cap`, 0 = none) so the
    /// selected token is BYTE-IDENTICAL — and still writes the (un-softcapped) logits for
    /// diagnostics. Replaces the SoftCap + Argmax packets; [`DevOp::ArgmaxFin`] folds `nblk` parts.
    /// `t0=C(logits) t1=x t2=W t3=part(u64[nblk])` · `i0=1 i1=N i2=K i4=a_row0` · `f0=cap`.
    GemvArgmax = 80,

    /// fp8 (w8a8) GROUPED gate/up GEMM + GeGLU — the native-fp8 twin of [`DevOp::MoeGroupGluGemmaPf`].
    /// BOTH operands e4m3 (`mma.sync.m16n8k32`). A gathered from `xq8` (e4m3, per-token `ascale`);
    /// Wg/Wu from `ewt` (e4m3) + per-output-channel scale from `est`. Pad rows write `fu=0`.
    /// `t0=fu t1=xq8([T,H]) t2=ewt t3=meta t4=row_token t5=ascale(f32[T]) t6=est` ·
    /// `i0=I_moe i1=H i2=n_exp i5=act`. Gated behind `PLOW_NV_W8A8`.
    MoeGroupGluGemmaPfW8a8 = 81,
    /// fp8 (w8a8) GROUPED down GEMM + gate-scale + scatter — twin of [`DevOp::MoeGroupDownGemmaPf`].
    /// A = `fu8` (contiguous e4m3, per-row `fscale`); Wd from `ewt` + per-channel `est`. Epilogue
    /// `acc*row_gate*fscale[row]*dscale[h]`, scatter to `part[row_partidx*H+h]`; pad skipped.
    /// `t0=part t1=fu8([pad,I]) t2=ewt t3=meta t4=row_partidx t5=row_gate t6=est t7=fscale(f32[pad])` ·
    /// `i0=H i1=I_moe i2=n_exp`. Gated behind `PLOW_NV_W8A8`.
    MoeGroupDownGemmaPfW8a8 = 82,

    // ===== Nemotron-3 Mamba-2 SSD mixer (plans/block-asset-harness.md §7 Nemotron, M4). =====
    // A NEW op family (opcode 90, leaving 83-89 as a gap after the MoE-prefill band). This is
    // the FIRST state-space op in the tree — no reuse of any existing kernel. The in_proj /
    // out_proj projections are ordinary GEMV/GEMM; this op is the mixer CORE: causal depthwise
    // conv1d + SiLU over (x,B,C), the selective SSD scan with per-head scalar decay, the D skip,
    // and the gated RMSNorm. It reads AND writes the carried conv_state + ssm_state, so prefill
    // is a full scan and decode is a single-step update through the SAME op. Correctness-first
    // (see runtime/nvidia/op_mamba.cuh) — the CPU golden is the f32 sequential recurrence in
    // `plowc` bin gemma4 `mamba_ref`, checked against an independent quadratic form.
    //
    // Operand layout (kept minimal — the scalar params are PACKED so conv_state and ssm_state
    // each get their own handle):
    //   t0=out([T,d_inner] bf16)  t1=xBC([T,conv_dim] bf16, post-in_proj pre-conv)
    //   t2=dt([T,n_head] bf16, raw)  t3=z([T,d_inner] bf16, gate)  t4=conv_w([conv_dim,d_conv] bf16)
    //   t5=params(f32: A_log[n_head] | D[n_head] | dt_bias[n_head] | conv_b[conv_dim] | norm_w[d_inner])
    //   t6=conv_state(f32[d_conv-1,conv_dim], in/out)  t7=ssm_state(f32[n_head,head_dim,d_state], in/out)
    //   i0=T i1=d_inner i2=n_head i3=head_dim i4=d_state i5=n_groups i6=d_conv i7=conv_dim   f0=eps.
    // conv_dim = d_inner + 2*n_groups*d_state. A[h] = -exp(A_log[h]); dt[t,h] = softplus(dt_raw+dt_bias);
    // dA = exp(dt*A). No time_step_limit clamp (assumption, see mamba_ref).
    Mamba2Scan = 90,

    /// MXFP4 decode GEMV (w4a16): OCP microscaling e2m1 weights + one E8M0 scale per 32 K.
    ///
    /// Decode is weight-bandwidth-bound, so this is the shape where fp4 pays: 4.25 effective
    /// bits/element (half a weight byte plus one scale byte per 32) against fp8's 8, moving the
    /// roofline ~1.88x over [`DevOp::GemvFp8`] and ~3.76x over bf16 [`DevOp::Gemv`].
    ///
    /// The E8M0 scale folds into the fp4 -> bf16 convert and that fold is EXACT — an MX scale is a
    /// power of two by construction, so the hardware's exponent-only `scalef32` operand loses
    /// nothing. (Contrast [`DevOp::GemvFp8Blk`], whose arbitrary-f32 block scale must stay a
    /// separate multiply.) There is therefore no dequant in the epilogue.
    ///
    /// Layout: `W` packed 2 fp4/byte, row stride `K/2` bytes, low nibble = even k; `S` one E8M0
    /// byte per 32-K block, row stride `K/32` bytes. One lane's 16-byte load is exactly one block.
    ///
    /// `t0=C(bf16) t1=x(bf16) t2=W(fp4) t3=S(e8m0)   i0=M i1=N i2=K`
    GemvMxfp4 = 91,

    /// MXFP4 decode fused gate|up GEMV+GLU (w4a16) — the mxfp4 twin of [`DevOp::GemvGluFp8`].
    /// Gate/up are two fp4 weight matrices with their own E8M0 scale rows; the SwiGLU fuses
    /// `act(g)*u` in one packet. `t0=C t1=x t2=Wg(fp4) t5=Wu(fp4) t3=Sg(e8m0) t4=Su(e8m0)
    /// i0=M i1=N i2=K i5=act`. See op_gemm.h `d_gemv_glu_mxfp4`.
    GemvGluMxfp4 = 92,

    /// MXFP4 (w4a16) prefill GEMM — bf16 activations × packed fp4 weights + E8M0 scale rows.
    /// Reuses the bf16 wide-K MFMA; only the weight fetch dequants fp4→bf16 with the MX scale
    /// folded. `t0=C t1=A(bf16) t2=W(fp4) t3=wscale(e8m0)  i0=M i1=N i2=K`. See `d_gemm_mxfp4`.
    GemmMxfp4 = 93,
}

impl DevOp {
    /// Every opcode, in numeric order.
    ///
    /// Hand-maintained alongside the enum. `dev_abi.rs` reparses the enum out of
    /// this file and fails if a variant is missing here, so it cannot silently
    /// fall out of date.
    pub const ALL: &'static [DevOp] = &[
        DevOp::Nop, DevOp::RmsNorm, DevOp::RowRms, DevOp::HeadNormRope,
        DevOp::Residual, DevOp::Glu, DevOp::Embed, DevOp::SoftCap,
        DevOp::Gemm, DevOp::GemmNorm, DevOp::Gemv, DevOp::FlashPrefill,
        DevOp::FlashDecode, DevOp::FlashMerge, DevOp::GemmSmall, DevOp::GemmMed,
        DevOp::NormResidual, DevOp::AddNorm, DevOp::Argmax, DevOp::ArgmaxFin,
        DevOp::GemvGlu, DevOp::GemmGlu, DevOp::GemvQkv, DevOp::GemvFp8,
        DevOp::GemvGluFp8, DevOp::QuantFp8, DevOp::GemmFp8, DevOp::GemmMedFp8,
        DevOp::GemmSmallFp8, DevOp::GemmGluFp8, DevOp::NormResidualNorm, DevOp::XReduce,
        DevOp::XReduceScatter, DevOp::XAllGather, DevOp::XFlashMerge, DevOp::XArgmaxFin,
        DevOp::XReduceTwoShot, DevOp::HeadNormRopeFp8, DevOp::FlashDecodeFp8, DevOp::FlashPrefillFp8,
        DevOp::MoeRouter, DevOp::MoeExpertGlu, DevOp::MoeExpertDown, DevOp::MoeCombine,
        DevOp::GemvFp8Blk, DevOp::MoeExpertGluFp8Blk, DevOp::MoeExpertDownFp8Blk, DevOp::DenseGluFp8Blk,
        DevOp::MoeGroupGluFp8Blk, DevOp::MoeGroupDownFp8Blk, DevOp::FlashMlaDecode, DevOp::FlashMlaPrefill,
        DevOp::OUvFold, DevOp::AttnSelect, DevOp::FlashGatherDecode, DevOp::FlashGatherPrefill,
        DevOp::MoeRouterTopk, DevOp::MlaMergeFold, DevOp::IndexScore, DevOp::IndexSelect,
        DevOp::LayerNorm, DevOp::MoeRouterGemma, DevOp::MoeExpertGluGemma, DevOp::MoeExpertDownGemma,
        DevOp::MoeCombineGemma, DevOp::MoeExpertGluGemmaFp8, DevOp::MoeExpertDownGemmaFp8, DevOp::MoeRouterGemmaScore,
        DevOp::MoeRouterGemmaTopk, DevOp::MoeRouterGemmaScoreFast, DevOp::MoeCombineNormGemma, DevOp::MoeExpertGluNormGemma,
        DevOp::MoeCombineResidNormGemma, DevOp::MoeRouterGemmaPf, DevOp::MoeAlignGemmaPf, DevOp::MoeGroupGluGemmaPf,
        DevOp::MoeGroupDownGemmaPf, DevOp::MoeCombineNormGemmaPf, DevOp::GemvSz, DevOp::GemvGluSz,
        DevOp::GemvArgmax, DevOp::MoeGroupGluGemmaPfW8a8, DevOp::MoeGroupDownGemmaPfW8a8, DevOp::Mamba2Scan,
        DevOp::GemvMxfp4, DevOp::GemvGluMxfp4, DevOp::GemmMxfp4,
    ];

    /// The `dev_isa.h` spelling of this opcode.
    ///
    /// Not derivable from the Rust name: `XReduceTwoShot` is `PLOW_DOP_XREDUCE2`
    /// and `RowRms` is `PLOW_DOP_ROWRMS`, so the map is explicit. The exhaustive
    /// `match` makes a new variant a compile error until it is spelled here.
    pub fn c_name(self) -> &'static str {
        match self {
            DevOp::Nop => "PLOW_DOP_NOP",
            DevOp::RmsNorm => "PLOW_DOP_RMSNORM",
            DevOp::RowRms => "PLOW_DOP_ROWRMS",
            DevOp::HeadNormRope => "PLOW_DOP_HEADNORM_ROPE",
            DevOp::Residual => "PLOW_DOP_RESIDUAL",
            DevOp::Glu => "PLOW_DOP_GLU",
            DevOp::Embed => "PLOW_DOP_EMBED",
            DevOp::SoftCap => "PLOW_DOP_SOFTCAP",
            DevOp::Gemm => "PLOW_DOP_GEMM",
            DevOp::GemmNorm => "PLOW_DOP_GEMM_NORM",
            DevOp::Gemv => "PLOW_DOP_GEMV",
            DevOp::FlashPrefill => "PLOW_DOP_FLASH_PREFILL",
            DevOp::FlashDecode => "PLOW_DOP_FLASH_DECODE",
            DevOp::FlashMerge => "PLOW_DOP_FLASH_MERGE",
            DevOp::GemmSmall => "PLOW_DOP_GEMM_SMALL",
            DevOp::GemmMed => "PLOW_DOP_GEMM_MED",
            DevOp::NormResidual => "PLOW_DOP_NORM_RESIDUAL",
            DevOp::AddNorm => "PLOW_DOP_ADD_NORM",
            DevOp::Argmax => "PLOW_DOP_ARGMAX",
            DevOp::ArgmaxFin => "PLOW_DOP_ARGMAX_FIN",
            DevOp::GemvGlu => "PLOW_DOP_GEMV_GLU",
            DevOp::GemmGlu => "PLOW_DOP_GEMM_GLU",
            DevOp::GemvQkv => "PLOW_DOP_GEMV_QKV",
            DevOp::GemvFp8 => "PLOW_DOP_GEMV_FP8",
            DevOp::GemvGluFp8 => "PLOW_DOP_GEMV_GLU_FP8",
            DevOp::QuantFp8 => "PLOW_DOP_QUANT_FP8",
            DevOp::GemmFp8 => "PLOW_DOP_GEMM_FP8",
            DevOp::GemmMedFp8 => "PLOW_DOP_GEMM_MED_FP8",
            DevOp::GemmSmallFp8 => "PLOW_DOP_GEMM_SMALL_FP8",
            DevOp::GemmGluFp8 => "PLOW_DOP_GEMM_GLU_FP8",
            DevOp::NormResidualNorm => "PLOW_DOP_NORM_RESIDUAL_NORM",
            DevOp::XReduce => "PLOW_DOP_XREDUCE",
            DevOp::XReduceScatter => "PLOW_DOP_XREDUCESCATTER",
            DevOp::XAllGather => "PLOW_DOP_XALLGATHER",
            DevOp::XFlashMerge => "PLOW_DOP_XFLASHMERGE",
            DevOp::XArgmaxFin => "PLOW_DOP_XARGMAX_FIN",
            DevOp::XReduceTwoShot => "PLOW_DOP_XREDUCE2",
            DevOp::HeadNormRopeFp8 => "PLOW_DOP_HEADNORM_ROPE_FP8",
            DevOp::FlashDecodeFp8 => "PLOW_DOP_FLASH_DECODE_FP8",
            DevOp::FlashPrefillFp8 => "PLOW_DOP_FLASH_PREFILL_FP8",
            DevOp::MoeRouter => "PLOW_DOP_MOE_ROUTER",
            DevOp::MoeExpertGlu => "PLOW_DOP_MOE_EXPERT_GLU",
            DevOp::MoeExpertDown => "PLOW_DOP_MOE_EXPERT_DOWN",
            DevOp::MoeCombine => "PLOW_DOP_MOE_COMBINE",
            DevOp::GemvFp8Blk => "PLOW_DOP_GEMV_FP8_BLK",
            DevOp::MoeExpertGluFp8Blk => "PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK",
            DevOp::MoeExpertDownFp8Blk => "PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK",
            DevOp::DenseGluFp8Blk => "PLOW_DOP_DENSE_GLU_FP8_BLK",
            DevOp::MoeGroupGluFp8Blk => "PLOW_DOP_MOE_GROUP_GLU_FP8_BLK",
            DevOp::MoeGroupDownFp8Blk => "PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK",
            DevOp::FlashMlaDecode => "PLOW_DOP_FLASH_MLA_DECODE",
            DevOp::FlashMlaPrefill => "PLOW_DOP_FLASH_MLA_PREFILL",
            DevOp::OUvFold => "PLOW_DOP_O_UV_FOLD",
            DevOp::AttnSelect => "PLOW_DOP_ATTN_SELECT",
            DevOp::FlashGatherDecode => "PLOW_DOP_FLASH_GATHER_DECODE",
            DevOp::FlashGatherPrefill => "PLOW_DOP_FLASH_GATHER_PREFILL",
            DevOp::MoeRouterTopk => "PLOW_DOP_MOE_ROUTER_TOPK",
            DevOp::MlaMergeFold => "PLOW_DOP_MLA_MERGE_FOLD",
            DevOp::IndexScore => "PLOW_DOP_INDEX_SCORE",
            DevOp::IndexSelect => "PLOW_DOP_INDEX_SELECT",
            DevOp::LayerNorm => "PLOW_DOP_LAYERNORM",
            DevOp::MoeRouterGemma => "PLOW_DOP_MOE_ROUTER_GEMMA",
            DevOp::MoeExpertGluGemma => "PLOW_DOP_MOE_EXPERT_GLU_GEMMA",
            DevOp::MoeExpertDownGemma => "PLOW_DOP_MOE_EXPERT_DOWN_GEMMA",
            DevOp::MoeCombineGemma => "PLOW_DOP_MOE_COMBINE_GEMMA",
            DevOp::MoeExpertGluGemmaFp8 => "PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8",
            DevOp::MoeExpertDownGemmaFp8 => "PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8",
            DevOp::MoeRouterGemmaScore => "PLOW_DOP_MOE_ROUTER_GEMMA_SCORE",
            DevOp::MoeRouterGemmaTopk => "PLOW_DOP_MOE_ROUTER_GEMMA_TOPK",
            DevOp::MoeRouterGemmaScoreFast => "PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST",
            DevOp::MoeCombineNormGemma => "PLOW_DOP_MOE_COMBINE_NORM_GEMMA",
            DevOp::MoeExpertGluNormGemma => "PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA",
            DevOp::MoeCombineResidNormGemma => "PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA",
            DevOp::MoeRouterGemmaPf => "PLOW_DOP_MOE_ROUTER_GEMMA_PF",
            DevOp::MoeAlignGemmaPf => "PLOW_DOP_MOE_ALIGN_GEMMA_PF",
            DevOp::MoeGroupGluGemmaPf => "PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF",
            DevOp::MoeGroupDownGemmaPf => "PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF",
            DevOp::MoeCombineNormGemmaPf => "PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF",
            DevOp::GemvSz => "PLOW_DOP_GEMV_SZ",
            DevOp::GemvGluSz => "PLOW_DOP_GEMV_GLU_SZ",
            DevOp::GemvArgmax => "PLOW_DOP_GEMV_ARGMAX",
            DevOp::MoeGroupGluGemmaPfW8a8 => "PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8",
            DevOp::MoeGroupDownGemmaPfW8a8 => "PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8",
            DevOp::Mamba2Scan => "PLOW_DOP_MAMBA2_SCAN",
            DevOp::GemvMxfp4 => "PLOW_DOP_GEMV_MXFP4",
            DevOp::GemvGluMxfp4 => "PLOW_DOP_GEMV_GLU_MXFP4",
            DevOp::GemmMxfp4 => "PLOW_DOP_GEMM_MXFP4",
        }
    }

    /// One past the highest opcode value — mirrors `PLOW_DOP__COUNT`. This is a
    /// dispatch-table bound, *not* the number of opcodes (the range has holes).
    pub const COUNT: u16 = 94;
}

/// Sentinel expert id a router writes for an unused top-k slot; the expert body skips
/// compute (still signals) when `expert_id >= num_experts`. Mirrors
/// `PLOW_EXPERT_UNUSED` in `dev_isa.h` and `Experts.expert_unused_sentinel`.
pub const EXPERT_UNUSED: u32 = u32::MAX;

/// A gate: wait until `counters[id] >= threshold`.
///
/// `threshold` is the number of *producing workgroups*, so it is determined by
/// how the producer was sliced ([`DevInst::blocks`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Wait {
    pub id: u32,
    pub threshold: u32,
}

/// One instruction, BUILDER-side. This is what the model compilers fill in and
/// what [`crate::devbuild::Builder`] owns in memory; it is NOT the wire format.
/// [`DevInst::pack`] converts it to the fixed 64-byte [`DevInst64`] the device
/// consumes. The wait/succ lists computed here are copied onto every
/// [`StreamEnt`] at build time — the wire instruction no longer carries them.
// No `Eq`: `f` is `[f32; 2]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DevInst {
    /// A [`DevOp`] discriminant.
    pub op: u16,
    /// Workgroups this op is sliced across (>= 1). Consumers gate on a counter
    /// with this as the threshold.
    pub blocks: u16,
    pub wait_len: u16,
    pub succ_len: u16,
    /// Index into the [`Wait`] table.
    pub wait_ofs: u32,
    /// Index into the counter-id table.
    pub succ_ofs: u32,
    /// Tensor handles; [`TENSOR_NONE`] for an absent optional operand.
    pub t: [u32; 8],
    pub i: [u32; 8],
    pub f: [f32; 2],
    /// Spare integer operands: `i` is full for some ops. `j[0]` carries the KV-cache stride,
    /// which the HEAD-MAJOR layout needs — see `dev_isa.h`.
    pub j: [u32; 2],
}

/// Wire sentinel for an absent tensor operand in [`DevInst64::t`]. The u16 twin
/// of [`TENSOR_NONE`]; usable handles are `0..0xFFFE`.
pub const TENSOR_NONE16: u16 = 0xFFFF;

/// One instruction, WIRE format. Fixed 64-byte stride — no variable-length
/// records on device. Byte-for-byte mirror of `PlowDevInst` in
/// `runtime/common/dev_isa.h` (asserted by `tests/dev_abi.rs`).
///
/// Layout vs the builder-side [`DevInst`]:
/// - wait/succ metadata lives on [`StreamEnt`] (every entry, not just `SE_FINE`),
/// - tensor handles narrow to u16 ([`TENSOR_NONE16`] = absent),
/// - `f[0..2]` and `j[0..2]` collapse into three 32-bit words: no opcode reads
///   both `f[1]` and `j[0]` (norm/MoE vs attention families — disjoint), so they
///   share `fj[1]`. `fj[0]` = `f[0]` bits, `fj[2]` = `j[1]`.
///
/// Field ORDER puts `t` at byte 16: the record stride is 64, so `t[8]` is
/// 16-byte aligned and the NVIDIA interpreter fetches all eight handles with a
/// single vector load (per-site u16 loads measured ~+0.4% decode TPOT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct DevInst64 {
    pub op: u16,
    pub blocks: u16,
    /// `fj[0]` = `f[0]` bit pattern; `fj[1]` = `f[1]` bit pattern ⊕ `j[0]`
    /// (mutually exclusive, asserted in [`DevInst::pack`]); `fj[2]` = `j[1]`.
    pub fj: [u32; 3],
    /// Tensor handles; [`TENSOR_NONE16`] for an absent optional operand.
    pub t: [u16; 8],
    pub i: [u32; 8],
}

impl DevInst {
    /// Pack to the 64-byte wire format. Panics on a tensor handle that overflows
    /// the u16 wire slot or an op that populates both members of the `fj[1]`
    /// overlay — both are compiler bugs, not runtime conditions.
    pub fn pack(&self) -> DevInst64 {
        let mut t = [TENSOR_NONE16; 8];
        for (k, slot) in t.iter_mut().enumerate() {
            *slot = if self.t[k] == TENSOR_NONE {
                TENSOR_NONE16
            } else {
                assert!(
                    self.t[k] < TENSOR_NONE16 as u32,
                    "tensor handle {} overflows the u16 wire slot (op {})",
                    self.t[k],
                    self.op
                );
                self.t[k] as u16
            };
        }
        assert!(
            self.f[1].to_bits() == 0 || self.j[0] == 0,
            "op {} sets both f[1] ({}) and j[0] ({}) — they share wire slot fj[1]",
            self.op,
            self.f[1],
            self.j[0]
        );
        DevInst64 {
            op: self.op,
            blocks: self.blocks,
            fj: [self.f[0].to_bits(), self.f[1].to_bits() | self.j[0], self.j[1]],
            t,
            i: self.i,
        }
    }
}

/// This entry carries its own wait/succ lists, replacing the instruction's.
pub const SE_FINE: u16 = 1;

/// This entry's wait/succ counters live in the SYSTEM-scope, peer-mapped [`DevProgram::xctr`]
/// region (a cross-GPU collective), not the agent-scope local `counters`. Orthogonal to
/// [`SE_FINE`]. Set by the TP compiler on the collective packets and on a producing GEMV whose
/// successor is a cross-GPU "partial ready" bump. See `plans/tp-design.md` §6a.
pub const SE_XCTR: u16 = 2;

/// One entry in a CU's stream: run `inst`, taking share `slice` of its work.
///
/// # Per-slice gates
///
/// By default the wait/succ lists live on the [`DevInst`], so all `blocks` workgroups
/// running an op block on the same counters — every op is a full N-way barrier, and a
/// consumer waits for the SLOWEST producer workgroup even when it needs the output of only
/// a handful of them.
///
/// That wait is not free. Measured on the real model (decode, ctx=3326): 256 CUs doing
/// *identical* gemv work finish across a 9.6–16.6 µs spread, and summed over the critical
/// path the gate burns **2.63 ms of a 16.9 ms token** waiting for one straggler after half
/// the machine is done. The straggler is **diffuse**, not one structurally slow CU (per-CU
/// mean duration spreads only 5.1%; the top-5 "finished last" CUs account for 8% of
/// instructions against 2% for random) — so it is recoverable.
///
/// With [`SE_FINE`] set, this entry's own lists replace the instruction's, and slice `s`
/// blocks only on the producer slices that actually feed it.
///
/// **Why this cannot deadlock:** [`crate::devbuild::Builder`] appends ops in topological
/// order, so for any dependency A → B every slice of A precedes every slice of B in every
/// CU's stream. A fine list can only *lower* a threshold or *narrow* a wait set; it can
/// never make a workgroup wait on something issued later in its own stream. Add a scheduler
/// that interleaves tiles across ops and that argument dies — see
/// `plans/fine-counter-deadlock-fix.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct StreamEnt {
    pub inst: u32,
    pub slice: u32,
    /// Index into the wait table. Only read when [`SE_FINE`] is set.
    pub wait_ofs: u32,
    /// Index into the successor counter-id table. Only read when [`SE_FINE`] is set.
    pub succ_ofs: u32,
    pub wait_len: u16,
    pub succ_len: u16,
    pub flags: u16,
    /* Which wave-class SEGMENT this entry belongs to. The host relaunches the interpreter once
     * per segment with that segment's wave count; the interp skips entries whose seg != cur_seg.
     * 0 for a single-segment (unsegmented) program. Was `_pad`; same 16-bit slot. */
    pub seg: u16,
}

/// Everything the interpreter needs, passed once as the kernel's args. The
/// pointers are **device** addresses, so this is only meaningful as the kernarg
/// block handed to `plow_interp_*`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DevProgram {
    pub insts: u64,
    pub stream: u64,
    /// `[n_cu]`
    pub stream_ofs: u64,
    /// `[n_cu]`
    pub stream_len: u64,
    pub waits: u64,
    pub succs: u64,
    /// Zeroed by the host before each run.
    pub counters: u64,
    /// Device pointer table, indexed by tensor handle.
    pub tensors: u64,
    /// `[total stream length]`, or 0 to disable tracing. Slot `stream_ofs[cu] + pc`.
    pub trace: u64,
    /// Segmented dispatch: interp runs only entries with `seg == cur_seg`.
    pub cur_seg: u32,
    pub _segpad: u32,
    /// Global-queue interpreter (Experiment E1). Op-major stream, segment bounds, shared cursor.
    pub gq_stream: u64,
    pub gq_seg_ofs: u64,
    pub gq_cursor: u64,
    // ===== Cross-GPU (tensor-parallel) fields. Single-GPU runs leave these 0. =====
    // Appended AFTER `gq_cursor` so every existing field (notably `trace`) keeps its
    // offset — the ABI-lock test only sees the size grow. See `plans/tp-design.md` §6a.
    /// This rank's cross-GPU counter region (SYSTEM-scope, peer-mapped). Points INTO
    /// `peer_scratch[rank]`; the per-rank offset is `xctr - peer_scratch[rank]`.
    pub xctr: u64,
    /// `[n_gpu]` — each rank's peer-mapped reduction region base.
    pub peer_scratch: u64,
    /// This GPU's TP rank.
    pub rank: u32,
    /// TP degree (1 = single-GPU, cross-GPU fields unused).
    pub n_gpu: u32,
}

/// One packet boundary, timestamped by the interpreter.
///
/// plow schedules a NETWORK; packets from different ops overlap on different CUs
/// under counter gates. A single-op benchmark cannot see that, so every packet
/// boundary is timestamped and the timeline is the actual object of study.
/// `t_ready - t_arrive` is the STALL (waiting on producers); `t_end - t_ready` is
/// the WORK. Clocked with `s_memrealtime` — a constant-rate counter coherent across
/// CUs, unlike the shader clock, which moves with DVFS (this GPU drops 2.2 -> 1.58
/// GHz under load, which would silently skew a shader-clock timeline).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceRec {
    pub cu: u32,
    pub pc: u32,
    pub inst: u32,
    pub op: u16,
    pub slice: u16,
    pub t_arrive: u64,
    pub t_ready: u64,
    pub t_end: u64,
}

// --- ABI lock: these must match the _Static_asserts in runtime/common/dev_isa.h.
// (`DevInst` is builder-internal and NOT part of the ABI; `DevInst64` is the wire type.)
const _: () = assert!(size_of::<Wait>() == 8);
const _: () = assert!(size_of::<DevInst64>() == 64);
const _: () = assert!(size_of::<StreamEnt>() == 24);
const _: () = assert!(size_of::<TraceRec>() == 40);
const _: () = assert!(size_of::<DevProgram>() == 128);
