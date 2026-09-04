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

/// `GemmWide.i[7]` tag selecting the gfx950 128x384x64 implementation.
/// Zero remains the 128x256x64 body, preserving existing packet bytes.
pub const GEMM_WIDE_C8_TAG: u32 = (128 << 16) | 384;

/// Sentinel for an ABSENT tensor handle carried in a [`DevInst::i`] slot.
///
/// A handful of ops demote a pointer into `i[]` because ten operands do not fit
/// eight `t` slots ([`DevOp::GemvQkvg`]'s `i6`, [`DevOp::GemvQkvMxfp4`]'s
/// `i5`/`i6`/`i7`). `i[]` is `u32` on the WIRE and does not narrow the way `t[]`
/// does, so [`TENSOR_NONE`] (`0xFFFF_FFFF`) would arrive as itself and never
/// equal the device's `PLOW_TENSOR_NONE`. The i-slot sentinel is therefore the
/// wire one, [`TENSOR_NONE16`], widened — the value `interp.hip` tests against.
///
/// It matters that this is not `0`: `0` is a legal tensor handle, so an emitter
/// that simply left an optional i-slot unset would name tensor 0 rather than
/// nothing, and the arm's absence check would pass on a packet that is missing
/// an operand.
pub const TENSOR_NONE_I: u32 = TENSOR_NONE16 as u32;

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
    /// `t0=out t1=x t2=gamma? t3=xq? t4=ascale?` · `i0=rows i1=feat` · `f0=eps`.
    /// `gamma = TENSOR_NONE` is the weightless RMSNorm (Gemma's `v_norm`).
    /// `t3/t4` (T11, `PLOW_QNORM_FUSE=1`): fused w8a8 activation quant — the normed row is
    /// also written as e4m3 `xq` with per-row `a_scale`, exactly the values a following
    /// [`DevOp::QuantFp8`] would produce (token-identical; needs a t3/t4-aware cubin).
    RmsNorm = 1,
    /// `t0=rms(f32) t1=x` · `i0=rows i1=feat` · `f0=eps`.
    /// Row RMS scalars only, so [`DevOp::GemmNorm`] can apply the norm in its
    /// prologue and the normalized activation never round-trips through HBM.
    RowRms = 2,
    /// `t0=out t1=x t2=gamma? t3=cos? t4=sin? t5=pos(i32)` ·
    /// `i0=ntok i1=nhead i2=hd i3=out_row0 i4=flags i6=n_batch_kv` · `f0=eps` ·
    /// `j0=out_stride j1=kv_mask`.
    ///
    /// `n_batch_kv != 0` makes row `t` sequence `t`: it writes its OWN batch-major ring at its
    /// OWN position, `((t*nhead + hh)*out_stride + pos[t]) * hd`. That is the only addressing a
    /// batched decode can use — the legacy `out_row0 + t` form takes ONE host-patched position
    /// per step and cannot express B sequences at B different positions. Zero keeps the legacy
    /// path byte-identical, so prefill and B=1 decode are unchanged.
    /// `cos = TENSOR_NONE` skips RoPE. `out_row0` lets K/V land directly at a row
    /// offset of the KV cache, so the cache write is not a separate copy.
    HeadNormRope = 3,
    /// `t0=out t1=a t2=b t3=pre?` · `i0=n` · `f0=scale`, computing
    /// `(a + b) * scale`, or `(pre + bf16(a + b)) * scale` when `pre` is present.
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
    /// `t0=C t1=x t2=W t3=rms? t4=gamma?` · `i0=M i1=N i2=K i3=norm` · `f0=eps`.
    /// Decode path (`M <= 16`): bandwidth-bound, uses no MFMA.
    ///
    /// `norm`: 0 = none. 1 = scale the A-operand by a PRECOMPUTED per-row RMS in `t3`, which a
    /// separate [`DevOp::RowRms`] packet produced. 2 = compute that scalar IN THE GEMV, from the
    /// `x` it already stages in LDS, so the producing [`DevOp::RmsNorm`] packet is not emitted at
    /// all — `t1` then names the norm's INPUT and `t4`/`f0` carry its gamma and eps.
    ///
    /// Mode 2 is emitted only by `k3::fuse_norm_gemv`, and only where the norm's output has no
    /// consumer that is not one of these GEMVs: it recomputes the reduction once per CONSUMER
    /// WORKGROUP, so it pays for itself through the deleted chain level and not otherwise. It is
    /// bit-exact — it normalizes the staged copy in place, rounding to bf16 with `d_rmsnorm`'s
    /// element map, so the k-loop reads the bytes the deleted packet would have written. It
    /// requires the LDS-staged arm and `d_rmsnorm`'s register path; the kernel re-checks both and
    /// demotes to `norm = 0` rather than reduce over an arena it never staged.
    Gemv = 10,
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Q t3=K t4=V t5=O_final` ·
    /// `i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit` ·
    /// `f0=scale j0=kv_stride j1=kv_mask`.
    /// `window = 0` is full causal. `hd` must be 256 or 512.
    /// For Gemma `scale = 1.0` — there is NO `1/sqrt(head_dim)`.
    ///
    /// This spec read `t0=O t1=Q t2=K t3=V` until 2026-07-29 and was STALE: it
    /// predates the split-K epilogue that added `Opart`/`mlpart` and moved the
    /// bf16 output to `t5`. Ground truth is `exec_flash_prefill`
    /// (`runtime/amd/interp.hip:272`), which passes
    /// `TEN(0), TEN(1), TEN(5), TEN(2), TEN(3), TEN(4)` into `d_flash_prefill`'s
    /// `(Opart, mlpart, O_final, Q, K, V)` — and [`DevOp::FlashDecode`] below,
    /// whose own spec was updated at the time and already agreed. Caught by
    /// `crate::slots`' drift test; see that module for why the table is checked
    /// against these comments rather than generated from them.
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
    /// `t0=part(u64[blocks]) t1=x` · `i0=n i1=n_batch`. Per-block partial of a greedy argmax,
    /// packed as `(ordered_bf16_key << 32) | ~index` so a plain unsigned max does the whole
    /// reduction. `n_batch` rows: `x` is `[n_batch][n]` and `part` is `[n_batch][blocks]`;
    /// 0 and 1 are byte-identical, which is why no emitter had to set it before batched decode.
    Argmax = 17,
    /// `t0=ids(i32) t1=part` · `i0=blocks i1=n_batch`. Folds the partials and writes the token id
    /// straight into the tensor the NEXT step's [`DevOp::Embed`] reads — so a sampled token
    /// never leaves the GPU.
    ArgmaxFin = 18,
    /// `t0=fu t1=x t2=W_gate t5=W_up` · `i0=M i1=N i2=K i5=act` · `f0=situ_beta
    /// f1=situ_linear_beta`, computing
    /// `fu = act(W_gate @ x) * (W_up @ x)` — gate and up in ONE GEMV, with the GLU applied in
    /// the **epilogue**, as every BLAS does it (cuBLASLt/CK/hipBLASLt).
    ///
    /// The two `f` slots carry Kimi-K3's `situ` betas and are read only at `act = 2`; every other
    /// activation ignores them.
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
    /// `t0=q_out t1=x t2=W_q t3=k_out t4=W_k t5=v_out t6=W_v t7=gamma?` ·
    /// `i0=M i1=Nq i2=K i3=Nk i4=Nv` · `f0=eps`,
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
    ///
    /// **`t7` PRESENT = THE Q-NORM FOLD** (emit knob `PLOW_GLM_FUSE_QNORM`, build axis of the
    /// same name in `op_gemm.h`): `t1` then carries the RAW pre-norm activation and this op
    /// applies the producing `RmsNorm` itself, to the LDS copy of `x` it already stages, with
    /// `f0` as eps — deleting the one-workgroup [`DevOp::RmsNorm`] packet that used to sit
    /// between GLM's two fused MLA GEMVs. Bit-exact to that packet: the staged row is
    /// normalized IN PLACE and ROUNDED to bf16 exactly as `d_rmsnorm`'s `fits` path does, and
    /// the ordinary un-normed hot loop then reads LDS bytes identical to the HBM bytes the
    /// deleted packet would have written (`gemv_norm_lds` in op_gemm.h). It is NOT
    /// [`DevOp::Gemv`]'s `norm == 1`, which multiplies inside the k-loop in f32 and produces a
    /// different number.
    ///
    /// `t7` absent is the pre-fold reading and every blob emitted without the knob is
    /// byte-identical. Op 22 has never carried a `t7`, so the discriminator is unambiguous:
    /// [`DevOp::GemvQkvg`]'s `t7` is `g_out` on a DIFFERENT opcode, and
    /// [`DevOp::GemvQkvMxfp4`]/[`DevOp::GemvQkvFp8`] spend `i5/i6/i7` and leave `t7` alone.
    /// Nothing here is a bitfield — unlike [`DevOp::FlashMlaPrefill`]'s `i6` (low 8 bits =
    /// causal KV-split `ns`, bit 8 = `W_ofold`) — and the two ops never meet: 22 is decode,
    /// 51 is prefill.
    GemvQkv = 22,
    /// `t0=C t1=x t2=W(fp8) t3=resid_out t4=b t5=w_scale(f32[N]) t6=gamma_b t7=gamma_n` ·
    /// `i0=M i1=N i2=K i3=nrn i4=a_row0` · `f0=eps f1=scale`. The fp8 (w8a16) twin
    /// of [`DevOp::Gemv`]: the weight row is `uint8[K]` OCP e4m3, so decode streams HALF the bytes
    /// (~2x the bandwidth-bound decode roofline). Each fp8 is converted to bf16 on load and the
    /// existing bf16 `fdot2` reduction is unchanged; the per-output-channel dequant `w_scale[n]` is
    /// applied ONCE in the epilogue on the wave sum, never per element. Decode-only.
    ///
    /// `i3 != 0` (AMD decode) is the NRN FOLD: [`DevOp::NormResidualNorm`] computed into this
    /// GEMV's LDS staging — `t1` becomes `a` (the residual in), `t3`/`t4`/`t6`/`t7` carry
    /// resid_out/b/gamma_b/gamma_n, `f0`=eps `f1`=layer_scale; bit 1 of `i3` marks the ONE packet
    /// of the q/k/v trio that stores the residual. Bit-exact to op 23 followed by op 30 — see
    /// `gemv_nrn_lds` in op_gemm.h for the mechanism and the ping-pong the concurrency forces.
    GemvFp8 = 30,
    /// `t0=fu t1=x t2=W_gate(fp8) t5=W_up(fp8) t3=gate_scale(f32[N]) t4=up_scale(f32[N])` ·
    /// `i0=M i1=N i2=K i3=resid_out i4=b i5=act i6=gamma_b i7=gamma_n` · `f0=eps f1=scale` ·
    /// `j1=nrn`. The fp8 twin of [`DevOp::GemvGlu`]: gate|up in ONE pass with the
    /// GLU applied in the epilogue, both weight streams fp8 e4m3 with their own per-channel scale.
    ///
    /// `j1 != 0` (AMD decode) is the NRN1 FOLD — the GemvGluFp8 sibling of op 30's `i3` fold:
    /// [`DevOp::NormResidualNorm`] computed into this GEMV's LDS staging. `t1` becomes `a` (the
    /// residual in), and because both free t slots already carry dequant scales, the four fold
    /// operands ride the free INTEGER slots as TENSOR HANDLES: `i3`=resid_out `i4`=b
    /// `i6`=gamma_b `i7`=gamma_n, with `f0`=eps `f1`=layer_scale (always 1.0 for NRN1).
    /// Bit-exact to op 23 followed by op 31 — see `d_gemv_glu_fp8_nrn` in op_gemm.h.
    GemvGluFp8 = 31,
    /// `t0=xq(fp8) t1=x(bf16) t2=a_scale(f32[M]) t3=gate? t4=up?` · `i0=M i1=K i2=act`.
    /// Per-row (per-token) fp8 activation quant — the w8a8 prefill's activation half.
    /// `a_scale[m] = rowmax|x[m,:]|/448`, `xq[m,k] = round_e4m3(x[m,k]/a_scale[m])`.
    /// Emitted once per activation, reused by every fp8 GEMM.
    /// `t3/t4` (T11, `PLOW_QNORM_FUSE=1`): fused GLU producer — `x` becomes an OUTPUT,
    /// the packet computes `fu = act(gate)*up` (bf16-rounded, exactly what [`DevOp::Glu`]
    /// writes) then quantizes it; token-identical to the split form (needs a t3/t4-aware cubin).
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
    ///
    /// `i4`/`i5`/`i6` fold an ALL-GATHER into the same packet: a COLUMN-parallel producer
    /// leaves rank r owning output columns `[r*gcols, (r+1)*gcols)` in a SECOND peer slot,
    /// and the gathered vector is added to the reduced one. `gcols = 0` disables it and the
    /// arm is byte-identical to the reduce-only one. K3's `routed_expert_up_proj` is the
    /// only user: its column-parallel partial and the shared expert's row-parallel one are
    /// summed anyway, so the gather costs one bf16 load per element on a rendezvous that
    /// already happened — instead of its own packet and its own rendezvous. The reduction is
    /// ROUNDED to bf16 before the gathered term is added, which is what makes the fold
    /// bit-exact against the reduce-then-`Residual` pair it replaces.
    /// `t0=out` · `i0=H i1=n_gpu i2=slot(byte offset into peer_scratch) i3=gate i4=gslot?
    /// i5=gcols? i6=row_w?`.
    XReduce = 24,
    /// Reduce-scatter half of the two-shot, ON ITS OWN: rendezvous `gate_rs`, then phase 1 of
    /// [`DevOp::XReduceTwoShot`] — this rank's owned slice `[n*rank/N, n*(rank+1)/N)` of the
    /// flat `[n]` partial is reduced (f32 acc, r = 0..N-1) and written IN PLACE into its own
    /// peer slot. Nothing is gathered; the packets that follow read the owned slice as a
    /// rank-relative band view. Emitted by the sequence-parallel seams (`PLOW_SEQ_PAR_SEAMS`).
    /// `t0=slot_tensor t1=band_copy?` · `i0=n i1=n_gpu i2=slot i3=gate_rs i6=gslot? i7=gcols?`.
    /// `slot_tensor` is the peer slot reduced in place, `slot` its byte offset. With `gcols`,
    /// the owned slice also folds the column-parallel partial at `gslot` for its own rows —
    /// rounded to bf16 BEFORE the add, exactly as the two-shot's phase 2 does. `band_copy` (a
    /// LOCAL `[n/N]` tensor) receives the same owned slice for a reader that outlives the
    /// slot's next writer.
    ///
    /// The kernel arm is not built yet (`op_collective.h`); a packet carrying one on an object
    /// without it must be refused at load, which is what the manifest requirement
    /// `PLOW_SEQ_PAR_SEAMS=1` is for. The number is ABI: it was reserved since the enum was
    /// written, and no blob on disk carries it.
    XReduceScatter = 25,
    /// All-gather ON ITS OWN, of up to THREE row-banded arrays under ONE rendezvous: every
    /// rank has written its band of each array into a peer slot (earlier packets), one
    /// workgroup announces the rank on `gate`, all wait `n_gpu` arrivals, then each rank copies
    /// slice `s` of each array from peer `s`'s slot into the local full tensor (the two-shot's
    /// phase-2 loop, unchanged). A `dst` of `TENSOR_NONE` (with `n = 0`) leaves that pair unused.
    /// `t0=dst0? t1=dst1? t2=dst2?` · `i0=n0? i1=n1? i2=n2? i3=gate i4=n_gpu i5=src_slot0?
    /// i6=src_slot1? i7=src_slot2?`.
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
    /// fabric-bound. Fused + self-contained like `XReduce`:
    /// partitions the flat [n] result into N contiguous slices, reduces THIS rank's
    /// owned slice from every peer's partial (writing it in-place, peer-visible), then
    /// gathers every peer's reduced slice into the local full vector. Two internal xctr
    /// rendezvous bracket the phases. Fabric ≈ 2(N−1)/N·msg/rank vs one-shot's (N−1)·msg.
    /// Bit-identical result to one-shot (same f32-acc, r=0..N−1 order). DECODE keeps the
    /// one-shot (its tiny [1,hidden] message is latency-, not bandwidth-, bound).
    /// `t0=out t1=resid? t2=attnres_out? t3=attnres_ring? t4=attnres_score? t5=attnres_gamma?
    /// t6=prefix_out?` · `i0=n(=t·hidden) i1=n_gpu i2=slot(byte offset) i3=gate_rs i4=gate_ag
    /// i5=e0_or_H i6=gslot_or_nb i7=gcols_or_nbcap` · `f0=attnres_eps?`. Plain: `i5=e0
    /// i6=gslot? i7=gcols?`. With a graph-selected fused AttnRes consumer `t1..t6` are set and
    /// `i5=H i6=nb i7=nb_cap f0=eps`. Phase 2 preserves `out`, materializes the rounded prefix,
    /// then runs AttnRes+RMSNorm on the same token-owned workgroups.
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

    // ===== MoE data-dependent counter-gate ops =====
    // Opcodes in the HIGH free range 40+ so they do
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
    /// **`t1` (residual) is OPTIONAL**, and Kimi-K3's Stable LatentMoE is why: its routed experts
    /// run at `routed_expert_hidden_size` (3584), so their combine has no hidden-width residual to
    /// add — the residual add happens after the up-projection back to 7168. Absent means "add
    /// nothing"; it used to be an unconditional dereference of a null pointer.
    /// `t0=out t1=residual? t2=shared? t3=part_base([k,H])` · `i0=H i1=k`.
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
    /// `i0=slot i1=I_moe i2=H i3=n_exp i5=act i6=enc` · `f0=situ_beta f1=situ_linear_beta`.
    ///
    /// `i5 = act`: 0 gelu_tanh, 1 silu, **2 = Kimi-K3 `situ`**, and only for 2 are `f0`/`f1` read
    /// (`activation_situ_beta` / `activation_situ_linear_beta`). situ transforms the UP branch as
    /// well as the gate, so the epilogue is `A(g)*B(u)` and not `act(g)*u`; the kernel routes it
    /// through `moe_glu`, and `moe_act` returns **NaN** for code 2 so an unconverted epilogue
    /// poisons its output instead of silently computing `gelu_tanh(g)*u`. `f0`/`f1` were free on
    /// every GLU-family op, so this consumes no `i` slot and every pre-K3 packet is unchanged.
    /// `i6 = weight encoding` (0 bf16, 1 block-fp8, 2 mxfp4).
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
    /// `i0=k i1=I_moe i2=H i3=n_exp i5=act i6=enc` · `f0=situ_beta f1=situ_linear_beta`, exactly
    /// as [`DevOp::MoeExpertGluFp8Blk`].
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
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len(i32)
    /// t7=qr_cos?` · `i0=n_batch i1=n_head i2=kv_stride i3=window i4=nsplit i5=kv_mask
    /// i6=qr_sin i7=gf` · `f0=scale`.
    ///
    /// **`t7` PRESENT = THE Q-ROPE FOLD** (emit knob `PLOW_GLM_FUSE_ROPE`, `[HNR-FOLD]` in
    /// `op_attention.h`): `t3` then carries the RAW `q_rope` projection and this op applies the
    /// interleaved RoPE itself, in the query staging that already reads every one of those
    /// `n_head*DR` elements — deleting the [`DevOp::HeadNormRope`] packet that used to sit
    /// between the q GEMV and the flash. Bit-identical to that packet (it runs `gamma=None`,
    /// `skip_norm=1`, so its value is `f2bf(v*cos -/+ partner*sin)` and nothing else).
    ///
    /// `t7` absent is the pre-fold reading and every blob emitted without the knob is
    /// byte-identical. The position operand is free: `qpos = kv_len[b] - 1`, which every decode
    /// entry point makes equal to the `pos[0]` the rope packet read.
    ///
    /// `i6` is the **sin table as a demoted tensor handle** — the [`DevOp::GemvQkvg`] rule for a
    /// read-only operand when the `t` slots are full — and is read ONLY when `t7` is present.
    /// It is a HANDLE, not a bitfield: unlike [`DevOp::FlashMlaPrefill`]'s `i6` (low 8 bits =
    /// causal KV-split `ns`, bit 8 = `W_ofold`) there is nothing packed into it. The two do not
    /// meet — 50 is decode, 51 is prefill.
    ///
    /// The fold is EXCLUSIVE with both sibling arms and always will be, because they spend the
    /// same two slots: [`DevOp::FlashGatherDecode`] puts `idx` in `t7` and `top_k` in `i6`, and
    /// [`DevOp::FlashMlaDecodeFp8`] puts its per-row scale strip in `t7`. plowc refuses the
    /// combinations.
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

    /// Gemma-4 26B-A4B bf16 sparse-MoE DECODE router (`d_moe_router_gemma`).
    /// Weightless-RMS(resid) → `·scale[H]·root` →
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

    // ===== Gemma-4 26B-A4B bf16 grouped-MoE PREFILL ops =====
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

    // ===== Nemotron-3 Mamba-2 SSD mixer =====
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
    // ===== MoE PREFILL (T>1) for the MLA family — Kimi K2.7 / GLM-5.2 / DeepSeek. =====
    // Mirrors `runtime/common/dev_isa.h` ops 83-87 (the AMD side landed first). The prefill bucket
    // had NO MoE arm of any kind — ops 40-49/56 are all decode-only, M=1 wave-per-output — so an MLA
    // prefill was attention-complete and FFN-incomplete, and the AMD dispatch `default:` writes
    // nothing rather than trapping, which would have made that a silent accuracy bug. These are the
    // token-sorted grouped-expert path: rows are padded to `MPF_BM=64` per expert, so the gathered
    // arrays are sized `T*k + n_exp*(MPF_BM-1)`.
    /// T-token router tail — a block-per-token loop of [`DevOp::MoeRouterTopk`], so it is
    /// bit-identical PER TOKEN to the decode router by construction. The `[T,n_exp]` logit matrix is
    /// an ordinary [`DevOp::Gemm`], already in the prefill bucket; only this tail was missing.
    /// `t0=table[T*k] t1=logit(bf16[T,n_exp]) t2=atom_acc? t3=bias` ·
    /// `i0=atom_h i1=n_exp i2=k i3=flags i4=T` · `f0=route_scale`.
    ///
    /// `t2`/`i0` are PLOW_MOE_PF_ATOMIC's fused-MoE accumulator: when set, this packet also zeroes
    /// `atom_acc[T, atom_h]` f32 before the top-k loop, because it is the earliest packet of the
    /// MoE chain (83 -> 84 -> 85 -> 86) and op 86 atomically adds into it. TENSOR_NONE / 0 on
    /// every other blob.
    MoeRouterTopkPf = 83,
    /// ALIGN/SORT, ONE workgroup: histogram the `T*k` routing slots by expert, build an
    /// `MPF_BM`-padded prefix, scatter each live slot into its expert's contiguous gathered-row
    /// range. `meta` is `[3*n_exp+1]` i32 (rowoff | cnt | m-tile prefix).
    /// `t0=meta(i32) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)` ·
    /// `i0=T i1=n_exp i2=k`.
    MoeAlignPf = 84,
    /// Grouped gate/up GEMM + GLU: A gathered from `xn2` by `row_token`, B is the tile's expert's
    /// gate|up staged into one BN tile. `i3` selects the block-fp8 or bf16 weight arm.
    /// `t0=fu_g t1=xn2[T,H] t2=expert_weight_table t3=expert_scale_table t4=meta t5=row_token` ·
    /// `i0=I_moe i1=H i2=n_exp i3=fp8 i5=act`.
    MoeGroupGluPf = 85,
    /// Grouped down GEMM + gate-scale + SCATTER into `part[row_partidx][H]`; pad rows are dropped.
    /// `t0=part(f32[T*k,H]) t1=fu_g t2=expert_weight_table t3=expert_scale_table t4=meta
    /// t6=row_partidx t7=row_gate` · `i0=H i1=I_moe i2=n_exp i3=fp8 i4=atom_ksh i5=det_ksh`.
    ///
    /// `i4 = log2(k)+1` is PLOW_MOE_PF_ATOMIC: `t0` is then a `[T,H]` f32 ACCUMULATOR this op
    /// atomically adds `gate*value` into at row `row_partidx[row] >> log2(k)`, not the `[T*k,H]`
    /// scatter. 0 = the shipped scatter, every existing blob.
    ///
    /// `i5 = log2(k)+1` is PLOW_MOE_PF_DET, the DETERMINISTIC twin: `t0` is a `[T,H]` **f64**
    /// FIXED-POINT accumulator (`rint(gate*value * 2^32)`, summed with an f64 atomic, exact and
    /// therefore order-independent). A separate field from `i4` on purpose — the two arms are
    /// separate build axes and a blob must never be readable as the other one.
    MoeGroupDownPf = 86,
    /// T-token combine — same expression and same FIXED slot order as [`DevOp::MoeCombine`], so at
    /// `T=1` it is bit-identical to it.
    /// `t0=out t1=residual[T,H]|NONE t2=shared[T,H]|NONE t3=part(f32[T*k,H])` ·
    /// `i0=H i1=k i2=T i3=t_row0 i4=det`.
    ///
    /// `t1 = TENSOR_NONE` IS the zero residual — the kernel spells it `residual ? ... : 0.0f`,
    /// which is what the TP path wants (xmid is added after the all-reduce). `i3 = t_row0` bands
    /// the combine over token rows `[i3, i3+i2)`. `i4 = 1` is PLOW_MOE_PF_DET: `t3` is the `[T,H]`
    /// f64 FIXED-POINT accumulator op 86 summed in place, read as one contiguous stream with
    /// `i1 == 1` and scaled by 2^-32.
    MoeCombinePf = 87,

    // ===== KDA — Kimi Delta Attention, 69 of Kimi-K3's 93 layers. =====
    // Spec: `docs/kimi-k3-kda.md`. The recurrence, per head, per token, is
    //   S <- (I - beta k k^T) . diag(exp(g)) . S + beta k v^T ;   o = S^T q
    // i.e. two composed memory mechanisms: an UNTARGETED per-(head, key-channel) forget gate
    // `diag(exp(g))`, and a TARGETED delta rule `(I - beta k k^T)` which — because the kernel L2
    // normalizes `k`, so `||k|| = 1` — is exactly `I` minus `beta` times the orthogonal projector
    // onto `k`. It erases the memory stored at key `k` and leaves everything orthogonal to `k`
    // untouched. Conflating the two is the single easiest way to get this wrong.
    //
    // THE STATE IS A DECLARED HBM TENSOR, NOT REGISTERS. `[H,D,D]` f32 per sequence per layer,
    // 6.00 MiB, constant in context length. A decode step is a read-modify-write tile op over it,
    // exactly like a KV ring, and that is what makes the register budget a knob instead of a veto
    // (`docs/kimi-k3-kda.md` §7.2).
    //
    // Four ops, not one. `Mamba2Scan = 90` is the cautionary tale, not the template: it is
    // monolithic, emitted onto ONE CU, consumes all 16 operand slots, has NO `interp.hip` arm (so
    // on AMD it hits the silent dispatch `default:` and computes nothing), and has never run on a
    // GPU. Decomposed, no op here is at the slot ceiling, six projection GEMVs stay concurrent
    // (`docs/kimi-k3-kda.md` §7.4 — `GLM_GROUP=1` measured +2.88 ms for a 38% op-count cut, so op
    // count is not the objective function), and every one of the four dispatches on gfx950.
    //
    // The chunked prefill scan of §7.6 is deliberately ABSENT. `KdaStateStep` takes a serial-`T`
    // loop, which is the reference `fused_recurrent` algorithm at any `T` and is exact; the
    // chunked form is a matmul-bound rewrite of a path that then already works. An opcode declared
    // before its kernel exists is how op 90 became dead code.
    /// KDA causal depthwise short conv + SiLU over the concatenated `q|k|v` projections.
    ///
    /// Three independent width-`i2` depthwise convs over `H*D` channels each (`groups =
    /// hidden_size`, `padding = W-1`, no bias), which is what gives KDA the local 4-token mixing a
    /// pure linear-attention recurrence cannot express. `t3` is the rolling input window, `[C,W]`
    /// f32, holding the last `W` inputs per channel with the CURRENT token at slot `W-1` (the
    /// `[fla]` convention, `short_conv.py:232-235`); it is read AND written. Activation is applied
    /// AFTER the convolution.
    ///
    /// The conv is a 4-tap stencil, not a scan, so it is fully parallel over `(t, channel)`; only
    /// the window carry is sequential, and that is per channel and lives in registers.
    ///
    /// `t0=out([T,3*H*D] bf16, post-activation) t1=x([T,3*H*D] bf16, pre-conv)
    /// t2=w([3*H*D,W] f32) t3=conv_state([3*H*D,W] f32, IN/OUT)` ·
    /// `i0=T i1=conv_dim(3*H*D) i2=W i3=act(1=silu)`.
    KdaConv = 88,
    /// KDA gate pre-pass — pure elementwise, and factored out of both the decode and the prefill
    /// paths so it is independently testable (`docs/kimi-k3-kda.md` §5.2).
    ///
    /// `i3 == 1` (K3, `gate_lower_bound = -5.0`) selects the BOUNDED branch
    /// `g = f0 * sigmoid(exp(A_log[h]) * (g_raw + dt_bias[h,d]))`, so `g` is strictly in
    /// `[f0, 0)` and the per-step decay `exp(g)` in `(e^f0, 1)` — the state can never be zeroed by
    /// the gate in one step and can never grow. `i3 == 0` is the older unbounded branch
    /// `-exp(A_log[h]) * softplus(g_raw + dt_bias[h,d])`. K3 is the first checkpoint to ship the
    /// bounded gate; no Kimi-Linear-era implementation has it, and neither does vLLM 0.23.0 or
    /// `main` (`docs/kimi-k3-kda.md` §3.4).
    ///
    /// `A_log` is indexed **per head** and `dt_bias` **per `h*D + d`** — they are different ranks,
    /// and the checkpoint ships `A_log` as a `[96]` per-head vector ZERO-PADDED to `[128]`, so the
    /// loader must slice `[:96]` (§3.2). `beta = sigmoid(beta_raw)` is one scalar per head, not
    /// per channel.
    ///
    /// `t0=g([T,H,D] f32) t1=beta([T,H] f32) t2=g_raw([T,H*D] bf16) t3=beta_raw([T,H] bf16)
    /// t4=A_log([H] f32) t5=dt_bias([H*D] f32)` · `i0=T i1=H i2=D i3=gate_mode` · `f0=lower_bound`.
    KdaGate = 89,

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

    /// As [`DevOp::Gemm`], 128×256 tile — the rung that owns the M=1024–2048 serving chunk.
    /// `i7=tile_variant`: zero selects 128×256×64; [`GEMM_WIDE_C8_TAG`] selects the
    /// default-off gfx950 128×384×64 exact-grid experiment.
    ///
    /// Added by the tile-inventory campaign. Between `Gemm` (256×256) and [`DevOp::GemmMed`]
    /// (128×128) the inventory had nothing that both fills 256 CUs and keeps BN=256's A-reuse,
    /// so every M≥1024 prefill shape paid 1.3–1.8×. Measured on Gemma-31B `q_proj` at M=1024:
    /// 926 TF/s here against 684 (`GemmMed`) and 523 (`Gemm`) — and 926 is parity with the
    /// Tensile assembly kernel measured on the same shape.
    GemmWide = 94,
    /// As [`DevOp::Gemm`], 192×256 tile — the rung that owns M≥4096 and every K-heavy shape.
    ///
    /// This is the tile every earlier sweep in the tree calls **c5** (`test_kernels.hip`, the
    /// Qwen `-DGM_BM=192` prefill object, the Tensile A/B). It was plow's own measured-best
    /// serving tile and was not selectable at all. Gemma-31B `down_proj` M=2048: 1033 TF/s
    /// against `Gemm`'s 794.
    GemmC5 = 95,

    /// 128×128 mxfp4 tile — the medium-tile twin of [`DevOp::GemmMxfp4`].
    ///
    /// `GemmMxfp4` hard-coded 256×256 for *every* shape with no selection at all, which is what
    /// put Kimi's mxfp4 `kv_a_proj` at ≈0.4% of peak. These four give the fp4 family the same
    /// rungs, and therefore the same `pick_tile`, as bf16.
    GemmMedMxfp4 = 96,
    /// 64×128 mxfp4 tile — the small-tile twin of [`DevOp::GemmMxfp4`].
    GemmSmallMxfp4 = 97,
    /// 128×256 mxfp4 tile — the twin of [`DevOp::GemmWide`].
    GemmWideMxfp4 = 98,
    /// 192×256 mxfp4 tile — the twin of [`DevOp::GemmC5`].
    GemmC5Mxfp4 = 99,

    /// 128×256 fp8 tile — the w8a8 twin of [`DevOp::GemmWide`].
    ///
    /// fp8 halves the operand bytes without halving the MFMA work, so the tile that balances CU
    /// fill against arithmetic intensity is not the one bf16 picks. With only three fp8 rungs
    /// against five bf16 ones the selector could not express that, whatever it was told about
    /// precision.
    GemmWideFp8 = 100,
    /// 192×256 fp8 tile — the w8a8 twin of [`DevOp::GemmC5`].
    GemmC5Fp8 = 101,

    /// KDA gated delta-rule state update — a READ-MODIFY-WRITE on the `[H,D,D]` f32 state `t6`.
    ///
    /// Per token, per head, per value column `j`, with `S' = diag(exp(g)) S`:
    /// ```text
    ///   S'[k] = S[k] * exp(g[k])          decay FIRST — u is the error against the DECAYED state
    ///   u     = v[j] - sum_k S'[k]*k[k]   delta / prediction error
    ///   S[k]  = S'[k] + beta*u*k[k]       rank-1 write
    ///   o[j]  = sum_k S[k]*q[k]           read the UPDATED state
    /// ```
    ///
    /// **STATE IS V-FIRST, `[h][v][k]`** (`transpose_state_layout=True` in K3's config, renamed
    /// `state_v_first` upstream). Since `V == K == 128` the byte count is identical either way, so
    /// transposing it produces garbage with exactly the right norm — the worst kind of wrong, and
    /// no norm check will catch it. V-first is also what makes the tiling free: a `v`-column is 512
    /// CONTIGUOUS bytes and both reductions run over `k` for fixed `v`.
    ///
    /// Tiling: `i3 = BV` value columns per workgroup, `D/BV` tiles per head, so `H*D/BV` work
    /// items — 768 at `H=96, D=128, BV=16`, hence `blocks = 256` and 100% CU fill. **Never
    /// parallelize over heads alone**: 96 heads is 37.5% of 256 at TP1 and 9.4% at TP4, which is
    /// the `MlaMergeFold` occupancy defect reproduced exactly (`docs/kimi-k3-kda.md` §7.3).
    /// One WAVE owns one column — `D = 64 lanes × 2` — so the state costs **2 f32/lane**, both
    /// reductions are `wave_sum`, and nothing crosses a wave.
    ///
    /// `i4` bit0 = L2-normalize `q` and `k` in kernel, with `eps` INSIDE the sqrt
    /// (`x / sqrt(sum x^2 + 1e-6)`, not `x / (norm + eps)`); `q` is then scaled by `f0` and `k` is
    /// NOT. `||k|| = 1` is load-bearing — it is what makes the delta term an exact projector.
    ///
    /// `T > 1` runs the same recurrence serially. That is exact at any `T`, and it is how the
    /// prefill/decode-agreement gate is expressed without a second algorithm.
    ///
    /// `t0=o([T,H,D] bf16) t1=q t2=k t3=v ([T,H,D] bf16) t4=g([T,H,D] f32) t5=beta([T,H] f32)
    /// t6=state([H,D,D] f32, V-FIRST, IN/OUT)` · `i0=T i1=H i2=D i3=BV i4=flags` ·
    /// `f0=scale(D^-0.5)`.
    KdaStateStep = 102,
    /// KDA output gate — `y[h,d] = RMSNorm_D(o[h,:])[d] * sigmoid(g_raw[h,d])`.
    ///
    /// `FusedRMSNormGated(head_dim, eps, activation='sigmoid')`. Three things are easy to get
    /// backwards and all three produce plausible-but-wrong output: the norm is over `D = 128`
    /// **inside a head**, not over `H*D`; its weight is a single `[D]` f32 vector SHARED by all
    /// `H` heads; and the sigmoid is applied to the RAW `g_proj` output with the gate multiplying
    /// **after** the norm.
    ///
    /// This is its own op rather than [`DevOp::KdaStateStep`]'s epilogue because the norm reduces
    /// over a whole head, whose `D` outputs are spread across `D/BV` workgroups under that op's
    /// slice map — folding it in would need a grid-wide barrier the interpreter does not provide.
    /// One wave per `(token, head)` row: `T*H` items, no cross-wave reduction.
    ///
    /// `t0=y([T,H,D] bf16) t1=o([T,H,D] bf16) t2=norm_w([D] f32) t3=g_raw([T,H*D] bf16)` ·
    /// `i0=T i1=H i2=D` · `f0=eps`.
    KdaGatedNorm = 103,
    /// **AttnRes** — Kimi-K3's residual-attention block. It REPLACES the plain residual add,
    /// twice per layer, in all 93 layers (`attn_res_block_size: 12`; AMD's day-0 post:
    /// *"stores one block residual every 12 layers"*).
    ///
    /// ```text
    ///   v      = cat(block_residual, prefix_sum)      [T, nb+1, H],  nb <= 8
    ///   k      = v * rsqrt(mean(v^2) + eps)           per row, eps INSIDE the rsqrt
    ///   scores = sum_d k[d] * score_w[d]
    ///   out    = softmax(scores) @ v                  the mix is over the RAW rows v, not k
    /// ```
    ///
    /// Three things that look like details and are not:
    /// - `score_w` is `norm.weight * proj.weight`, **constant**, and folds at prep time into one
    ///   `[H]` f32. Neither factor is needed separately.
    /// - The mix is over the **raw** rows. Mixing the normalized rows is a plausible misreading
    ///   that gives the right shape and the wrong per-row magnitude.
    /// - `variance = mean(x^2)`, RMSNorm's variance, not a mean-centred one.
    ///
    /// `nb = 0` degenerates to an exact copy (softmax over one element). The reference skips the
    /// call in that case; the arm handles it so a caller cannot get a zero-filled output by
    /// emitting it anyway.
    ///
    /// Slice map, honestly: **one workgroup per token**, `blocks = min(T, ncu)`, because both
    /// reductions span the full 7168-wide row and the softmax couples the rows. At `T = 1` that
    /// is 1 of 256 CUs. `perf-data/archive/k3/kimi-k3-kernel-gap.md` §10 item 7 requires this to stay ONE
    /// packet ("three packets × 186 is 3.3 ms/token of pure protocol"), which rules out the
    /// obvious fix of splitting the reduction across blocks and finishing it in a second packet.
    ///
    /// `t0=out([T,H] bf16) t1=prefix_sum([T,H] bf16) t2=block_residual([T,nb_cap,H] bf16)
    /// t3=score_w([H] f32) t4=push_src? t5=gamma? t6=res_a? t7=res_b?` ·
    /// `i0=T i1=H i2=nb i3=push_row i4=nb_cap i5=res_pre?` · `f0=eps`.
    ///
    /// `gamma` is the FUSED POST-NORM, `[H] bf16`, and it is what makes the slice map above
    /// affordable. Every AttnRes in a K3 program is read by exactly one consumer and that consumer
    /// is an RMSNorm over its own output — the attention-side mix feeds the mixer's pre-norm, the
    /// MLP-side mix feeds `post_attention_layernorm`, 186 of 186 in the 93-layer decode blob. Both
    /// packets are ONE workgroup, back to back, on a chain with nothing ready behind them. Present,
    /// the mix is RMSNormed IN PLACE over `out` (`out = out * rsqrt(mean(out^2) + eps) * gamma`,
    /// same `eps`) and the packet subsumes the RMSNORM. Bit-exact: the reduction runs over the
    /// bf16-ROUNDED mix re-read from `out`, which is precisely what the separate packet re-read
    /// from HBM. Absent, the raw mix is left in `out` and the arm is byte-identical to before.
    ///
    /// `nb` and `nb_cap` are **different numbers**. `nb` is the count of rows LIVE at this layer
    /// (0 -> 8 with depth); `nb_cap` is the ring's allocated row count, constant for the program,
    /// and it is what the ring strides by: `blkres[t][r]` is at `(t*nb_cap + r)*H`. At `T = 1` the
    /// token index is 0, the stride never multiplies and the two are indistinguishable — which is
    /// why the operand had to exist before prefill did. The arm poisons on `nb_cap < nb`.
    ///
    /// `push_src` is `[T,H] bf16` — the layer input a SNAPSHOT layer pushes onto ring row
    /// `push_row` of EVERY token's slice before the mix. Absent on every non-snapshot layer. It is
    /// safe at any `T`: the workgroups partition the tokens, the ring is per token, and the mix
    /// reads rows `[0, nb)` while the push writes row `nb` — one past — so no workgroup reads
    /// another's pushed row inside the packet, and the readers that follow are gated by the
    /// counter DAG (`Dep::Coarse` waits on every producer slice).
    ///
    /// `res_a`/`res_b` optionally materialize `prefix_sum = bf16(res_a + res_b)` inside this
    /// packet. `res_pre` selects `prefix_sum = bf16(res_pre + bf16(res_a + res_b))`, retaining
    /// the intermediate BF16 rounding and the materialized tensor for all other consumers.
    AttnRes = 104,
    /// **`situ` GLU** — Kimi-K3's activation, on EVERY GLU in the model (dense L0, shared
    /// experts, routed experts).
    ///
    /// `out = beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`, with
    /// `activation_situ_beta = 4.0` and `activation_situ_linear_beta = 25.0`.
    ///
    /// **A distinct opcode, not a third `act` code**, because situ transforms the UP branch as
    /// well: the expression shape is `A(g) * B(u)`, where every existing GLU site in this tree is
    /// `act(g) * u` selected by a two-value ternary (`op_elementwise.h:69` and seven more). A new
    /// act code alone would apply the gate transform and leave `up` un-clipped — a small error at
    /// `|u| < 25` that grows with the tail, i.e. plausible output and the wrong model.
    ///
    /// It is a soft-clipped SiLU: as `beta -> inf`, `beta*tanh(g/beta) -> g` and the gate branch
    /// becomes `silu`. `linear_beta <= 0` disables the up transform (what `linear_beta is None`
    /// means), so a zeroed immediate degrades to "no transform" rather than "clip to zero".
    ///
    /// `t0=out t1=gate t2=up` (all `[n]` bf16) · `i0=n` · `f0=beta f1=linear_beta`.
    SituGlu = 105,
    /// Kimi-K3 **MLA output gate** (`mla_use_output_gate: true`, 24 of 93 layers).
    ///
    /// `out = a * sigmoid(b)`, with `a` the attention output and `b` the RAW `g_proj` logits
    /// (`modeling_kimi_linear.py:470-473`):
    ///
    /// ```text
    /// g           = self.g_proj(hidden_states).sigmoid()
    /// attn_output = attn_output * g          # BEFORE o_proj
    /// ```
    ///
    /// Three things that look like details and are not:
    ///
    /// 1. `b` is `g_proj` of the MLA sub-layer **input** (the post-`input_layernorm` hidden), not
    ///    of the attention output. Feeding it `a` has the right shape and the wrong model.
    /// 2. It is `sigmoid(b)`, not `silu(b)`. This is why it is its own opcode rather than a third
    ///    `act` code on [`DevOp::Glu`], where `act=1` is SiLU: the two differ by a factor of the
    ///    logit, so the substitution is finite, correctly-shaped and wrong on every token.
    /// 3. Both operands are `[n_head * v_head_dim]` **head-major**, which is exactly what
    ///    [`DevOp::MlaMergeFold`] writes (`O + (b*n_head + h)*V`) and exactly what the reference's
    ///    `attn_output.reshape(batch, seq, -1)` produces. No permute is implied.
    ///
    /// Not folded into [`DevOp::MlaMergeFold`]'s epilogue — which
    /// `perf-data/archive/k3/kimi-k3-kernel-gap.md` §10 item 6 suggests and which would be nearly free —
    /// because that op is GLM-5.2's too and is on its critical path. A separate streaming pass
    /// keeps GLM's packet bytes and register table untouched, and keeps the fold and the gate
    /// independently diffable in a stage-by-stage gate.
    ///
    /// `t0=out t1=a t2=b` (all `[n]` bf16) · `i0=n`.
    MlaOutGate = 106,
    /// **DENSE PREFILL block-fp8 GEMM** (w8a16) — `C[M,N] bf16 = A[M,K] bf16 · W[N,K] e4m3`, with
    /// DeepSeek/GLM's `weight_block_size: [128,128]` grid of arbitrary-f32 `weight_scale_inv`.
    ///
    /// The T-row twin of [`DevOp::GemvFp8Blk`] for a plain `[N,K]` weight, and the arm whose
    /// absence made `GLM_LINEAR_FP8` decode-only: `o_proj` and the three `shared_experts.*`
    /// projections had no block-fp8 opcode to lower to at `rows > 1`, so a STACKED (prefill +
    /// decode) blob would have read fp8 bytes as bf16 — `declare_glm_rows` refused rather than
    /// emit one. Same scale convention as ops 44/45/46/48/49 and 85/86:
    /// `S[(n >> 7) * ceil(K/128) + (k >> 7)]`, one convention across the whole family.
    ///
    /// NOT [`DevOp::GemmFp8`] (33): that is the **w8a8** rung — one f32 per output CHANNEL plus a
    /// per-row activation scale from [`DevOp::QuantFp8`] — which can neither address a `[128,128]`
    /// grid nor run without an fp8 A operand. NOT ops 85/86 either: those carry a genuine
    /// block-fp8 prefill body but only under the grouped-MoE contract (expert weight/scale tables,
    /// `MoeAlignPf` meta, gather/scatter row maps, f32 `part` output).
    ///
    /// ONE TILE RUNG (128x128x64), unlike the bf16/w8a8/mxfp4 five-rung families, and it is a
    /// register fact rather than a shortcut: the block scale is arbitrary-f32, so it must be
    /// PROMOTED into a second f32 accumulator every 128 K rather than folded into the fp8->bf16
    /// convert, which doubles a tile's accumulator cost. The 192x256 and 256x256 rungs would need
    /// 192 and 256 accumulator registers and cannot be built at 8 waves at all. See
    /// `d_gemm_fp8_blk` in `runtime/amd/op_gemm.h` for the table. Emitted directly, not through
    /// `pick_tile`.
    ///
    /// `t0=C t1=A t2=W(e4m3) t3=weight_scale_inv(f32)` · `i0=M i1=N i2=K`.
    GemmFp8Blk = 107,
    /// `t0=q_out t1=x t2=W_q t3=k_out t4=W_k t5=v_out t6=W_v t7=g_out` ·
    /// `i0=M i1=Nq i2=K i3=Nk i4=Nv i5=Ng i6=W_g`, computing FOUR projections in ONE GEMV.
    ///
    /// [`DevOp::GemvQkv`] with a fourth output stream. Kimi-K3's KDA block projects the same
    /// pre-normed `x[7168]` four ways — `q_proj`, `k_proj`, `v_proj` and the full-rank output gate
    /// `g_proj`, each `[12288, 7168]` — so all four concatenate into one `N=Nq+Nk+Nv+Ng` sweep
    /// exactly as three did. At 69 KDA layers that is **207 fewer packets per token**.
    ///
    /// # This is the OUTPUT-dimension merge, and the distinction is the whole point
    ///
    /// The design notes measured `GLM_GROUP=1` removing 38% of the ops for
    /// **+2.88 ms**: it collapsed work that ran on disjoint CU slices into a loop inside one
    /// packet, which destroys concurrency. Op count is not the objective function. This merge is
    /// the opposite shape — the per-CU column count RISES from `(Nq+Nk+Nv)/nblk` to
    /// `(Nq+Nk+Nv+Ng)/nblk` (144 to 192 at K3's geometry over 256 CUs), so the op gets WIDER and
    /// nothing that ran in parallel starts running in sequence. What is deleted is three counter
    /// gates and three redundant LDS stagings of the same `x`, not parallelism.
    ///
    /// # `i6` is a TENSOR HANDLE
    ///
    /// Nine pointers (four outputs, four weights, `x`) do not fit [`DevInst`]'s eight `t` slots
    /// and the wire instruction is a fixed 64 bytes. Of the nine, a WEIGHT is the safe one to
    /// demote: a wrong weight handle reads the wrong bytes and the output is visibly garbage,
    /// where a wrong OUTPUT handle would silently overwrite an unrelated tensor. It stays a
    /// handle resolved through the same pointer table as `t[]` — not the packed multi-tensor blob
    /// [`DevOp::Mamba2Scan`] uses, which is a symptom of over-fusion rather than a precedent.
    ///
    /// `i5=Ng` and `i6=W_g` are both REQUIRED. The AMD arm traps on either being absent rather
    /// than degrading to a 3-stream sweep, because that fallback would leave `g_out` exactly as it
    /// found it — finite, fluent and wrong, against a dispatch `default:` that never traps.
    ///
    /// `t[0..6]` is identical to [`DevOp::GemvQkv`]'s, so the two share ONE interpreter body
    /// (op 22 is this with `Ng=0`); the register budget is therefore unchanged, which matters
    /// because the decode object sits at 254 of 256 VGPRs. Decode-only, `M*K` must fit LDS.
    GemvQkvg = 108,

    // ===== FP8 (e4m3) LATENT KV for the MLA family (`PLOW_FP8_KV`). ====
    // The MLA twins of [`DevOp::FlashDecodeFp8`] / [`DevOp::FlashPrefillFp8`], and the
    // reason they had to exist: `PLOW_FP8_KV=1` swapped the DENSE flash only, so every
    // model whose KV is a shared LATENT — DeepSeek, Kimi-K2.7, GLM-5.2, Kimi-K3 — had no
    // fp8-KV path at all. That is also the family with the LARGEST KV: K3's `ckv`(512) +
    // `krot`(64) over 24 MLA layers is 27.0 KiB/token, 3.38 GiB at 128k.
    /// FP8-KV twin of [`DevOp::FlashMlaDecode`]. `t4` is the `uint8[b][ctx][512]` e4m3 latent
    /// cache written by [`DevOp::HeadNormRopeFp8`] at `HD=512` with `cosb`/`sinb` absent (an
    /// RMSNorm plus an fp8 store — the fp8 twin of the `RmsNorm` that writes the bf16 `ckv` row,
    /// and an op the runtime's kv-row-writer scan already matches, so the layer cannot drop out
    /// of the per-step row patch).
    ///
    /// BOTH per-row dequant scales share ONE tensor slot, because the dense MLA decode has
    /// exactly one free one (`t7` is the gather `idx`; every other slot is live):
    /// `kv_scale[b*kv_stride + row]` is the `ckv` scale and
    /// `kv_scale[(n_batch+b)*kv_stride + row]` the `krot` scale, the second half present only
    /// when `krot_fp8 != 0`. `krot_fp8` reuses `i6` (the gather `top_k` slot, dead here) and
    /// selects whether `t5` is the untouched bf16 rope cache or its own e4m3 cache.
    ///
    /// Per-ROW scales, matching [`DevOp::HeadNormRopeFp8`]: a KV row is written once at its own
    /// step and never revisited, so the scale costs one f32 per 512 stored bytes and needs no
    /// second pass — where a per-tensor scale would have to be chosen before the context exists.
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len(i32) t7=kv_scale` ·
    /// `i0=n_batch i1=n_head i2=kv_stride i3=window i4=nsplit i5=kv_mask i6=krot_fp8 i7=gf` ·
    /// `f0=scale`.
    FlashMlaDecodeFp8 = 109,
    /// FP8-KV twin of [`DevOp::FlashMlaPrefill`]. Same operands as [`DevOp::FlashMlaDecodeFp8`]
    /// with `i4 = n_tok` instead of `nsplit` — the same slot reuse the bf16 MLA prefill makes,
    /// and for the same reason (`nsplit` MUST be 1 for prefill). Built only under
    /// `PLOW_MLA_PREFILL`.
    /// `t0=Opart(f32) t1=mlpart(f32) t2=Qabs t3=Qrope t4=Ckv t5=Krope t6=kv_len(i32) t7=kv_scale` ·
    /// `i0=n_batch i1=n_head i2=kv_stride i3=window i4=n_tok i5=kv_mask i6=krot_fp8 i7=gf` ·
    /// `f0=scale`.
    FlashMlaPrefillFp8 = 110,
    /// KDA short conv over all THREE streams in one packet — [`DevOp::KdaConv`] merged along the
    /// CHANNEL axis, which is its output axis.
    ///
    /// The three convs are independent work over `C = H*D` channels each, and they were three
    /// packets for that reason. At batch 1 that reasoning is inverted by measurement: a KDA decode
    /// layer is launch/protocol bound, `runtime/tests/kda_fuse_bench_gfx950.c` puts a packet at
    /// ~12 us against a chain whose entire arithmetic is a rounding error, and three packets of
    /// independent work cost three times one packet of the same work. So they merge.
    ///
    /// This is the [`DevOp::GemvQkvg`] direction, NOT the `GLM_GROUP=1` one. Nothing that ran in
    /// parallel starts running in sequence: each conv already spanned all 256 CUs with
    /// `ceil(C/256)` channels apiece, and fused they span the same 256 CUs with `ceil(3C/256)`.
    /// The op gets WIDER — 48 -> 144 channels per CU at TP1, 6 -> 18 at TP8 — which is what the
    /// knob contract asks of a merge. The body is literally [`DevOp::KdaConv`]'s, called on a
    /// per-stream sub-range, so the two are bit-identical by construction rather than by test.
    ///
    /// TWELVE POINTERS, four per stream, and `t[8]` holds eight. The four demoted into `i[]` are
    /// the `v` TAPS and all three CONV STATES, chosen the way [`DevOp::GemvQkvg`] chose: a wrong
    /// weight or state handle reads visibly wrong bytes, where a wrong OUTPUT handle silently
    /// overwrites an unrelated tensor. All four are REQUIRED — the AMD arm traps on any being
    /// absent rather than convolving a subset, because the dispatch `default:` never traps and a
    /// partial sweep would leave a stream's `mix` finite, fluent and wrong.
    ///
    /// `t0=q_out t1=k_out t2=v_out t3=q_in t4=k_in t5=v_in t6=w_q t7=w_k` ·
    /// `i0=T i1=C i2=W i3=act i4=w_v i5=cs_q i6=cs_k i7=cs_v` · `j0=bstride j1=parked`.
    KdaConv3 = 111,
    /// [`DevOp::KdaStateStep`] with [`DevOp::KdaGate`] folded into its LDS staging.
    ///
    /// The state step already stages this head's `g` into LDS once per (item, token) and
    /// exponentiates it. `KdaGate`'s whole output is that vector plus one scalar per head, and
    /// both are computable from operands the step can read directly, so the separate packet buys
    /// a `[T,H,D]` f32 round trip through HBM and nothing else:
    ///
    /// ```text
    ///   g[h,d] = f1 * sigmoid(exp(A_log[h]) * (g_raw[t,h,d] + dt_bias[h,d]))    i6 == 1
    ///   g[h,d] = -exp(A_log[h]) * softplus(g_raw[t,h,d] + dt_bias[h,d])         i6 == 0
    ///   beta[h] = sigmoid(beta_raw[t,h])
    /// ```
    ///
    /// BIT-IDENTICAL to `KdaGate` followed by [`DevOp::KdaStateStep`], not merely equivalent: the
    /// deleted intermediate was f32 in HBM and an f32 store/load is exact, so the same
    /// expressions produce the same bits. That is what makes the fusion checkable against the
    /// unfused path on the same fixture rather than only against a tolerance.
    ///
    /// SLICE MAP UNCHANGED, which is the constraint that matters. `blocks` is still
    /// `min(H*D/BV, n_cu)` and the item is still `(head, tile of BV value columns)`; the gate is
    /// evaluated where its consumer already is, not looped over. Its arithmetic is recomputed
    /// `D/BV` times per head — `KdaGate` computed each element once — which is a few thousand
    /// transcendentals against a packet that costs ~12 us, and it deletes 12 288 f32 of write plus
    /// `D/BV` x that of read per layer.
    ///
    /// `dt_bias` is demoted to `i5` for the same reason `KdaConv3` demotes taps: nine pointers do
    /// not fit eight slots and a wrong weight handle is visible where a wrong output handle is
    /// not. `i5` and `t7` are REQUIRED and the arm traps on either being absent — without the
    /// gate operands this op cannot fall back to reading a precomputed `g`, because it has no
    /// slot naming one.
    ///
    /// `t0=o t1=q t2=k t3=v t4=g_raw t5=beta_raw t6=state t7=A_log` ·
    /// `i0=T i1=H i2=D i3=BV i4=flags i5=dt_bias i6=gate_mode i7=parked` · `f0=scale f1=lower_bound`.
    KdaStateStepG = 112,
    /// `t0=fu t1=A t2=Wg(fp4) t5=Wu(fp4) t3=Sg(e8m0) t4=Su(e8m0)` · `i0=M i1=N i2=K i5=act`,
    /// computing `fu = act(Wg @ A) * (Wu @ A)` — the MXFP4 twin of [`DevOp::GemmGlu`] and the T-row
    /// twin of [`DevOp::GemvGluMxfp4`].
    ///
    /// # The one place the encoding cost HBM traffic rather than a different weight fetch
    ///
    /// Without this arm the shared-expert prefill unfuses into two [`DevOp::GemmMxfp4`] plus a
    /// [`DevOp::Glu`]: gate and up are each materialised to HBM as `[M, N]` bf16 and read back, ~8
    /// bytes per output element of traffic that the fused epilogue never spends, plus two extra
    /// packets and their counter gates. Everywhere else in an mxfp4 packet the encoding changes
    /// only which bytes a weight fetch reads. Precision and fusion are orthogonal, and this closes
    /// the gap on every MoE layer.
    ///
    /// # It is two existing template flags composed, not a third GEMM body
    ///
    /// `GLU` owns the EPILOGUE — the accumulator's `SN` axis selects gate vs up, so both halves of
    /// an output element land in the same lane and no shuffle is needed — and `WFP4` owns the
    /// B-FETCH, which dequants fp4→bf16 with the E8M0 scale folded exactly. The two are disjoint:
    /// the dequant finishes before the tile reaches LDS and the epilogue never sees a weight. The
    /// only addition is that the SCALE ROW must follow the gate/up select alongside the weight
    /// pointer — a fused arm that switched the weight and kept the gate's scale row would read
    /// `Wu`'s nibbles with `Wg`'s exponents, wrong by a per-block power of two.
    ///
    /// Same tile, same registers, same MFMA count as [`DevOp::GemmMxfp4`]: a workgroup emits `BN/2`
    /// fused columns for the MFMA it used to spend on `BN` raw ones, which is the same arithmetic
    /// because every output needs both halves anyway.
    ///
    /// 256×256 only, as its bf16 twin is (the epilogue needs `SN == 2`), and THAT is what bounds
    /// the win rather than the fusion. Measured at the K3 shared-expert prefill shape over 15
    /// `(T, N)` points: −38.8%…−48.7% against the SAME tile unfused, and −20%…−35% against the BEST
    /// unfused rung where the emit gate fires. The mechanism is the wave count — the fused arm does
    /// the pair's MFMA in ONE kernel over the same tile grid, so it saves a whole pass over the 256
    /// CUs whenever its grid fits one; the round trip is the smaller term. Where the shape does not
    /// fill the machine at 256×256 a 64×128 or 128×128 unfused pair wins by up to 35%, which is why
    /// `glu_fusion_wins_mxfp4` is part of the arm and not a nicety.
    GemmGluMxfp4 = 113,

    /// `t0=q_out t1=x t2=W_q(fp4) t3=k_out t4=W_k(fp4) t5=v_out t6=W_v(fp4)` ·
    /// `i0=M i1=Nq i2=K i3=Nk i4=Nv i5=S_q i6=S_k i7=S_v`, computing up to THREE mxfp4 projections
    /// in ONE decode GEMV — the fp4 twin of [`DevOp::GemvQkv`].
    ///
    /// Kimi-K3's three MLA down-projections (`q_a` → 1536, `kv_a` → 512, `k_rope` → 64) all read
    /// the same pre-normed `x[7168]`, so their output columns concatenate into one
    /// `N = Nq+Nk+Nv` sweep exactly as the bf16 form's do.
    ///
    /// # Why this is worth three packets a layer
    ///
    /// A GEMV census measured every K3 decode GEMV except `lm_head` pinned at a ~0.032 ms LAUNCH
    /// FLOOR, using under 3% of achievable bandwidth. At batch 1 these kernels are launch-bound,
    /// not bandwidth-bound, so deleting packets is the whole win — and `k_rope` is the extreme
    /// case: 64 columns over 256 workgroups gives `per = 1`, so 192 CUs own nothing at all.
    ///
    /// This is the OUTPUT-dimension merge, the same direction [`DevOp::GemvQkvg`] documents. The
    /// per-CU column count RISES (6 / 2 / 1 across the three split sweeps → 9 fused, at K3's
    /// geometry over 256 CUs), so the op gets WIDER and nothing that ran in parallel starts running
    /// in sequence — the opposite of the design notes' `GLM_GROUP=1`, which
    /// removed 38% of the ops for **+2.88 ms** by collapsing disjoint-CU work into a loop.
    ///
    /// # `i5`/`i6`/`i7` are TENSOR HANDLES
    ///
    /// Three outputs, three fp4 weights, three E8M0 scale rows and `x` is TEN pointers against
    /// [`DevInst`]'s eight `t` slots, and the wire instruction is a fixed 64 bytes.
    /// [`DevOp::GemvQkvg`] set the rule for which operand is demoted: a WEIGHT, because a wrong
    /// weight handle reads the wrong bytes and the output is visibly garbage, where a wrong OUTPUT
    /// handle would silently overwrite an unrelated tensor. An MX scale row is that same kind of
    /// operand and strictly SAFER — it is read-only, it is half of the weight (fp4 nibbles are
    /// meaningless without their exponents), and a wrong one is off by a per-block power of two,
    /// which is visible in the first token. So `t[0..6]` stays byte-for-byte [`DevOp::GemvQkv`]'s
    /// and the three scale handles take the three integer slots op 22 leaves empty. No output is
    /// demoted, and they are ordinary handles resolved through the same `T[]` table as `t[]`.
    ///
    /// Rejected: a packed pointer blob like [`DevOp::Mamba2Scan`]'s, which this file already calls
    /// a symptom of over-fusion and which adds an indirection to a launch-bound kernel; and
    /// requiring the three scale rows CONTIGUOUS so one base plus a stride addresses all three —
    /// they are three separate checkpoint tensors, so the arena would have to repack them and a
    /// silent violation reads another tensor's exponents.
    ///
    /// `Nv = 0` is the legal TWO-STREAM form (`t5`/`t6`/`i7` all absent), which is what the
    /// `q_nope`|`q_rope` pair off the q-lora norm needs. Any other absence TRAPS on the AMD arm
    /// rather than degrading to a narrower sweep, because that fallback would leave an output
    /// exactly as it found it — finite, fluent and wrong, against a dispatch `default:` that never
    /// traps.
    ///
    /// BYTE-EXACT to the split [`DevOp::GemvMxfp4`] calls it replaces, and structurally so: a
    /// column's value depends only on its own weight row, its own scale row and the shared `x`,
    /// accumulated over the same chunks in the same order. Concatenating the sweeps changes which
    /// workgroup owns a column, never what that column computes. Verified elementwise on gfx950.
    ///
    /// ONE interpreter body with [`DevOp::GemvMxfp4`], which is this with `Nk = Nv = 0` — the same
    /// register argument [`DevOp::GemvQkvg`] makes. Decode-only, `M*K` must fit LDS.
    GemvQkvMxfp4 = 114,

    /// `t0=q_out t1=x t2=W_q(fp8) t3=k_out t4=W_k(fp8) t5=v_out t6=W_v(fp8)` ·
    /// `i0=M i1=Nq i2=K i3=Nk i4=Nv i5=S_q i6=S_k i7=S_v`, computing up to THREE per-channel-fp8
    /// (w8a16) projections in ONE decode GEMV — the fp8 twin of [`DevOp::GemvQkv`], with
    /// [`DevOp::GemvQkvMxfp4`]'s slot map: `t[0..6]` is byte-for-byte op 22's and the three
    /// `f32[N]` dequant-scale rows are TENSOR HANDLES in the integer slots op 22 leaves empty
    /// (same demotion rule, same [`TENSOR_NONE_I`] sentinel for the absent one).
    ///
    /// Gemma/Qwen/Llama fp8 decode previously ran q/k/v as three `GEMV_FP8` packets on
    /// byte-proportional disjoint CU sets ("opcode 26 deferred"); concatenating their output
    /// columns into one `N = Nq+Nk+Nv` sweep deletes two counter gates per layer and fills every
    /// CU uniformly. BYTE-EXACT to the split calls: a column's value depends only on its own
    /// weight row, its own scale and the shared `x`, accumulated over the same chunks in the same
    /// order — concatenation changes which workgroup owns a column, never what it computes.
    ///
    /// `Nv = 0` is the legal TWO-STREAM form (`t5`/`t6`/`i7` all absent). Any other absence TRAPS
    /// on the AMD arm rather than degrading to a narrower sweep. ONE interpreter body with
    /// [`DevOp::GemvFp8`], which is this with `Nk = Nv = 0`. Decode-only; x staged in LDS when
    /// `M*K` fits, exactly as `d_gemv_fp8`.
    GemvQkvFp8 = 115,

    /// FUSED TP all-reduce + residual + RMSNorm — the decode attention seam
    /// `[XReduce -> AddNorm]` as ONE single-workgroup packet. `t0`=out (normed),
    /// `t1`=resid_out, `t2`=a (residual in), `t3`=gamma; `i0`=feat, `i1`=n_gpu,
    /// `i2`=partial slot byte offset, `i3`=xctr gate id; `f0`=eps. The kernel rounds
    /// the peer reduction to bf16 before the residual add, so the fold is
    /// BIT-IDENTICAL to the pair it replaces (dev_isa.h op 116's note). Decode-only
    /// (one row); feat must fit one workgroup's strided pass (`feat <= 16 * 512`),
    /// both enforced by the emitter.
    /// `t0=out2 t1=xmid_out t2=x t3=gamma` · `i0=feat i1=n_gpu i2=slot i3=gate` · `f0=eps`.
    XReduceAddNorm = 116,
    /// DSA sparse-prefill T-row indexer score (dev_isa.h op 117): score[t][s] =
    /// Σ_h w[t][h]·ReLU(q_idx[t][h]·k_idx[s]) for s <= q_pos0 + t. Score is f32
    /// [n_tok][kv_stride].
    /// `t0=Score t1=Qidx t2=Kidx t3=W t4=kv_len` · `i0=n_tok i1=index_heads i2=kv_stride
    /// i3=index_head_dim` · `f0=scale`.
    IndexScorePf = 117,
    /// Per-query-row EXACT top-k select (op 118): one workgroup per row, the op-59
    /// radix key (score desc, lowest-index tie-break) with LDS-only histograms. idx is
    /// i32 [n_tok][top_k], unused slots -1.
    /// `t0=idx t1=Score t2=kv_len` · `i0=n_tok i1=top_k i2=kv_stride`.
    IndexSelectPf = 118,
    /// Per-64-query-tile UNION build (op 119): membership-mask scatter + ascending
    /// compaction into the table the gathered V2 flash walks (u32 counts header,
    /// then per tile `[cap i32 pos][cap u32 maskLo][cap u32 maskHi]`).
    /// umask scratch is u64 [n_qt][kv_stride].
    /// `t0=union t1=umask t2=idx t3=kv_len` · `i0=n_tok i1=top_k i2=kv_stride i3=cap`.
    IndexUnionPf = 119,
    /// B1 KDA Conv3 + gated state step with double-buffered convolution windows.
    /// `t7` is a u32 tensor-handle descriptor:
    /// `[wq,wk,wv, cs0q,cs0k,cs0v, cs1q,cs1k,cs1v, A_log,dt_bias,in.pos,parked?]`.
    /// `t0=o t1=q_raw t2=k_raw t3=v_raw t4=g_raw t5=beta_raw t6=state t7=descriptor` ·
    /// `i0=T(1) i1=H i2=D i3=BV i4=flags i5=W i6=gate_mode` · `f0=scale f1=lower_bound`.
    KdaConvStateStepG = 120,
    /// Dense single-sequence BT64 chunk-KDA preparation. Normalizes q/k in place and produces
    /// chunk-local log2 gate prefixes plus beta. Emitted for compiled `T>=512`; runtime ragged
    /// rebasing may shorten `T`. `D in {64,128}`.
    /// `t0=q(in/out) t1=k(in/out) t2=g_prefix t3=beta t4=g_raw t5=beta_raw t6=A_log
    /// t7=dt_bias` · `i0=T i1=H i2=D i3=gate_mode` · `f0=lower_bound`.
    KdaChunkPrepare = 121,
    /// Dense BT64 chunk-local QK/KK products and triangular solve.
    /// `t0=Aqk t1=Ainv t2=q t3=k t4=g_prefix t5=beta` · `i0=T i1=H i2=D` · `f0=scale`.
    KdaChunkIntra = 122,
    /// Transform a BT64 inverse into W/U factors.
    /// `t0=W t1=U t2=Ainv t3=k t4=v t5=g_prefix t6=beta t7=q?` ·
    /// `i0=T i1=H i2=D i3=V i4=qpre` · `f0=scale`.
    KdaChunkWu = 123,
    /// Ordered dense single-sequence chunk carry, with V-first f32 recurrent state.
    /// `t0=o t1=state t2=q t3=k t4=W t5=U t6=Aqk t7=g_prefix` ·
    /// `i0=T i1=H i2=D i3=V i4=qpre` · `f0=scale`.
    KdaChunkCarry = 124,
    /// Standalone fused KDA decode boundary: Conv3 + gated recurrent state step + gated RMSNorm.
    /// The raw-argument kernel is selected by opcode capability rather than model identity.
    /// `t7` is a u32 tensor-handle descriptor:
    /// `[wq,wk,wv,csq,csk,csv,A_log,dt_bias,norm_w,output_gate_raw,parked]`.
    /// `t0=y t1=q_raw t2=k_raw t3=v_raw t4=forget_raw t5=beta_raw t6=state t7=descriptor` ·
    /// `i0=rows i1=H i2=D i3=BV i4=W i5=flags i6=gate_mode i7=descriptor_version(2)` ·
    /// `f0=scale f1=lower_bound j1=norm_eps_bits`.
    KdaDecodeFused = 125,
    /// Standalone layout boundary for materialized MLA prefill.
    /// `t0=K[B,T,H,192] t1=V[B,T,H,128] t2=KV[B,T,H,256] t3=K_rope[B,T,64]` ·
    /// `i0=T i1=H i2=qk_nope(128) i3=qk_rope(64) i4=v_head(128)`.
    /// Selected by dimensions, not model identity. The raw kernel copies the first 128 values
    /// of each KV head to K, broadcasts the shared 64-value rope row into every K head, and
    /// copies the final 128 values to V.
    MlaMaterializePack = 126,
    /// Standalone causal bf16 materialized MLA prefill.
    /// `t0=O[B,T,H,128] t1=Q[B,T,H,192] t2=K[B,T,H,192] t3=V[B,T,H,128]` ·
    /// `i0=T i1=H i2=H_KV i3=D_QK(192) i4=D_V(128) i5=abi(1)` · `f0=scale`.
    /// The only production implementation is a capability-checked gfx950 raw object.
    FlashMlaMaterializedPrefill = 127,
}

impl DevOp {
    /// Every opcode, in numeric order.
    ///
    /// Hand-maintained alongside the enum. `dev_abi.rs` reparses the enum out of
    /// this file and fails if a variant is missing here, so it cannot silently
    /// fall out of date.
    pub const ALL: &'static [DevOp] = &[
        DevOp::Nop,
        DevOp::RmsNorm,
        DevOp::RowRms,
        DevOp::HeadNormRope,
        DevOp::Residual,
        DevOp::Glu,
        DevOp::Embed,
        DevOp::SoftCap,
        DevOp::Gemm,
        DevOp::GemmNorm,
        DevOp::Gemv,
        DevOp::FlashPrefill,
        DevOp::FlashDecode,
        DevOp::FlashMerge,
        DevOp::GemmSmall,
        DevOp::GemmMed,
        DevOp::NormResidual,
        DevOp::AddNorm,
        DevOp::Argmax,
        DevOp::ArgmaxFin,
        DevOp::GemvGlu,
        DevOp::GemmGlu,
        DevOp::GemvQkv,
        DevOp::GemvFp8,
        DevOp::GemvGluFp8,
        DevOp::QuantFp8,
        DevOp::GemmFp8,
        DevOp::GemmMedFp8,
        DevOp::GemmSmallFp8,
        DevOp::GemmGluFp8,
        DevOp::NormResidualNorm,
        DevOp::XReduce,
        DevOp::XReduceScatter,
        DevOp::XAllGather,
        DevOp::XFlashMerge,
        DevOp::XArgmaxFin,
        DevOp::XReduceTwoShot,
        DevOp::HeadNormRopeFp8,
        DevOp::FlashDecodeFp8,
        DevOp::FlashPrefillFp8,
        DevOp::MoeRouter,
        DevOp::MoeExpertGlu,
        DevOp::MoeExpertDown,
        DevOp::MoeCombine,
        DevOp::GemvFp8Blk,
        DevOp::MoeExpertGluFp8Blk,
        DevOp::MoeExpertDownFp8Blk,
        DevOp::DenseGluFp8Blk,
        DevOp::MoeGroupGluFp8Blk,
        DevOp::MoeGroupDownFp8Blk,
        DevOp::FlashMlaDecode,
        DevOp::FlashMlaPrefill,
        DevOp::OUvFold,
        DevOp::AttnSelect,
        DevOp::FlashGatherDecode,
        DevOp::FlashGatherPrefill,
        DevOp::MoeRouterTopk,
        DevOp::MlaMergeFold,
        DevOp::IndexScore,
        DevOp::IndexSelect,
        DevOp::LayerNorm,
        DevOp::MoeRouterGemma,
        DevOp::MoeExpertGluGemma,
        DevOp::MoeExpertDownGemma,
        DevOp::MoeCombineGemma,
        DevOp::MoeExpertGluGemmaFp8,
        DevOp::MoeExpertDownGemmaFp8,
        DevOp::MoeRouterGemmaScore,
        DevOp::MoeRouterGemmaTopk,
        DevOp::MoeRouterGemmaScoreFast,
        DevOp::MoeCombineNormGemma,
        DevOp::MoeExpertGluNormGemma,
        DevOp::MoeCombineResidNormGemma,
        DevOp::MoeRouterGemmaPf,
        DevOp::MoeAlignGemmaPf,
        DevOp::MoeGroupGluGemmaPf,
        DevOp::MoeGroupDownGemmaPf,
        DevOp::MoeCombineNormGemmaPf,
        DevOp::GemvSz,
        DevOp::GemvGluSz,
        DevOp::GemvArgmax,
        DevOp::MoeGroupGluGemmaPfW8a8,
        DevOp::MoeGroupDownGemmaPfW8a8,
        DevOp::MoeRouterTopkPf,
        DevOp::MoeAlignPf,
        DevOp::MoeGroupGluPf,
        DevOp::MoeGroupDownPf,
        DevOp::MoeCombinePf,
        DevOp::KdaConv,
        DevOp::KdaGate,
        DevOp::Mamba2Scan,
        DevOp::GemvMxfp4,
        DevOp::GemvGluMxfp4,
        DevOp::GemmMxfp4,
        DevOp::GemmWide,
        DevOp::GemmC5,
        DevOp::GemmMedMxfp4,
        DevOp::GemmSmallMxfp4,
        DevOp::GemmWideMxfp4,
        DevOp::GemmC5Mxfp4,
        DevOp::GemmWideFp8,
        DevOp::GemmC5Fp8,
        DevOp::KdaStateStep,
        DevOp::KdaGatedNorm,
        DevOp::AttnRes,
        DevOp::SituGlu,
        DevOp::MlaOutGate,
        DevOp::GemmFp8Blk,
        DevOp::GemvQkvg,
        DevOp::FlashMlaDecodeFp8,
        DevOp::FlashMlaPrefillFp8,
        DevOp::KdaConv3,
        DevOp::KdaStateStepG,
        DevOp::GemmGluMxfp4,
        DevOp::GemvQkvMxfp4,
        DevOp::GemvQkvFp8,
        DevOp::XReduceAddNorm,
        DevOp::IndexScorePf,
        DevOp::IndexSelectPf,
        DevOp::IndexUnionPf,
        DevOp::KdaConvStateStepG,
        DevOp::KdaChunkPrepare,
        DevOp::KdaChunkIntra,
        DevOp::KdaChunkWu,
        DevOp::KdaChunkCarry,
        DevOp::KdaDecodeFused,
        DevOp::MlaMaterializePack,
        DevOp::FlashMlaMaterializedPrefill,
    ];

    /// Recover the opcode from its wire discriminant, or `None` for a value no
    /// variant claims.
    ///
    /// A linear scan of [`Self::ALL`] rather than a match: `ALL` is already the
    /// list `dev_abi.rs` proves exhaustive, so reusing it means a new variant
    /// cannot be reachable here and missing from there. Callers are disassembly
    /// and diagnostics, never the dispatch path.
    pub fn from_u16(op: u16) -> Option<DevOp> {
        Self::ALL.iter().copied().find(|o| *o as u16 == op)
    }

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
            DevOp::GemvQkvg => "PLOW_DOP_GEMV_QKVG",
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
            DevOp::MoeRouterTopkPf => "PLOW_DOP_MOE_ROUTER_TOPK_PF",
            DevOp::MoeAlignPf => "PLOW_DOP_MOE_ALIGN_PF",
            DevOp::MoeGroupGluPf => "PLOW_DOP_MOE_GROUP_GLU_PF",
            DevOp::MoeGroupDownPf => "PLOW_DOP_MOE_GROUP_DOWN_PF",
            DevOp::MoeCombinePf => "PLOW_DOP_MOE_COMBINE_PF",
            DevOp::KdaConv => "PLOW_DOP_KDA_CONV",
            DevOp::KdaGate => "PLOW_DOP_KDA_GATE",
            DevOp::KdaStateStep => "PLOW_DOP_KDA_STATE_STEP",
            DevOp::KdaGatedNorm => "PLOW_DOP_KDA_GATED_NORM",
            DevOp::AttnRes => "PLOW_DOP_ATTN_RES",
            DevOp::SituGlu => "PLOW_DOP_SITU_GLU",
            DevOp::MlaOutGate => "PLOW_DOP_MLA_OUT_GATE",
            DevOp::Mamba2Scan => "PLOW_DOP_MAMBA2_SCAN",
            DevOp::GemvMxfp4 => "PLOW_DOP_GEMV_MXFP4",
            DevOp::GemvGluMxfp4 => "PLOW_DOP_GEMV_GLU_MXFP4",
            DevOp::GemmGluMxfp4 => "PLOW_DOP_GEMM_GLU_MXFP4",
            DevOp::GemvQkvMxfp4 => "PLOW_DOP_GEMV_QKV_MXFP4",
            DevOp::GemmMxfp4 => "PLOW_DOP_GEMM_MXFP4",
            DevOp::GemmWide => "PLOW_DOP_GEMM_WIDE",
            DevOp::GemmC5 => "PLOW_DOP_GEMM_C5",
            DevOp::GemmMedMxfp4 => "PLOW_DOP_GEMM_MED_MXFP4",
            DevOp::GemmSmallMxfp4 => "PLOW_DOP_GEMM_SMALL_MXFP4",
            DevOp::GemmWideMxfp4 => "PLOW_DOP_GEMM_WIDE_MXFP4",
            DevOp::GemmC5Mxfp4 => "PLOW_DOP_GEMM_C5_MXFP4",
            DevOp::GemmWideFp8 => "PLOW_DOP_GEMM_WIDE_FP8",
            DevOp::GemmC5Fp8 => "PLOW_DOP_GEMM_C5_FP8",
            DevOp::GemmFp8Blk => "PLOW_DOP_GEMM_FP8_BLK",
            DevOp::FlashMlaDecodeFp8 => "PLOW_DOP_FLASH_MLA_DECODE_FP8",
            DevOp::FlashMlaPrefillFp8 => "PLOW_DOP_FLASH_MLA_PREFILL_FP8",
            DevOp::KdaConv3 => "PLOW_DOP_KDA_CONV3",
            DevOp::KdaStateStepG => "PLOW_DOP_KDA_STATE_STEP_G",
            DevOp::GemvQkvFp8 => "PLOW_DOP_GEMV_QKV_FP8",
            DevOp::XReduceAddNorm => "PLOW_DOP_XREDUCE_ADD_NORM",
            DevOp::IndexScorePf => "PLOW_DOP_INDEX_SCORE_PF",
            DevOp::IndexSelectPf => "PLOW_DOP_INDEX_SELECT_PF",
            DevOp::IndexUnionPf => "PLOW_DOP_INDEX_UNION_PF",
            DevOp::KdaConvStateStepG => "PLOW_DOP_KDA_CONV_STATE_STEP_G",
            DevOp::KdaChunkPrepare => "PLOW_DOP_KDA_CHUNK_PREPARE",
            DevOp::KdaChunkIntra => "PLOW_DOP_KDA_CHUNK_INTRA",
            DevOp::KdaChunkWu => "PLOW_DOP_KDA_CHUNK_WU",
            DevOp::KdaChunkCarry => "PLOW_DOP_KDA_CHUNK_CARRY",
            DevOp::KdaDecodeFused => "PLOW_DOP_KDA_DECODE_FUSED",
            DevOp::MlaMaterializePack => "PLOW_DOP_MLA_MATERIALIZE_PACK",
            DevOp::FlashMlaMaterializedPrefill => "PLOW_DOP_FLASH_MLA_MATERIALIZED_PREFILL",
        }
    }

    /// One past the highest opcode value — mirrors `PLOW_DOP__COUNT`. This is a
    /// dispatch-table bound, *not* the number of opcodes (the range has holes).
    ///
    /// 106 -> 107 when `MlaOutGate = 106` was added: C's `PLOW_DOP__COUNT` is an
    /// auto-numbered enum terminator and moved on its own, this one did not. Nothing
    /// indexes either constant today, so there was no runtime consequence — but the
    /// drift went unseen for a different reason worth recording: `cargo test` stops at
    /// the first failing SUITE, devgen's `tuned_tile_selection` fails ahead of packet,
    /// and so `dev_opcodes` never ran. Use `--no-fail-fast` when checking ABI guards.
    ///
    /// 108 -> 109 when `GemvQkvg = 108` was added, and it hid the SAME way twice
    /// over: `tuned_tile_selection` was red again (a stale tuning cell, not this),
    /// and the runs that did pass were filtered to `--lib`, which excludes
    /// `tests/dev_opcodes.rs` entirely. It surfaced only once the MXFP4 tile
    /// campaign made that suite green. Two independent filters, one blind spot —
    /// which is the argument for `--no-fail-fast` above, not against it.
    ///
    /// 109 -> 111 when `FlashMlaDecodeFp8 = 109` / `FlashMlaPrefillFp8 = 110` were
    /// added. Two at once, and the constant is `highest + 1`, not `+ 1` per op.
    /// 111 -> 113 for `KdaConv3 = 111` and `KdaStateStepG = 112`. These were authored as
    /// 109/110 by an agent working in parallel with the one that added the fp8-KV pair, and BOTH
    /// branches picked the same two free values — the collision surfaced only at merge, in
    /// `dev_isa.h`. Renumbering the later pair was the resolution. Two variants, one
    /// bump: this constant is one past the HIGHEST opcode, not a count, and adding a pair moves it
    /// by two whether or not the range has holes.
    /// 116 -> 117 for `XReduceAddNorm = 116` (the fused TP seam).
    pub const COUNT: u16 = 128;

    /// The `(M, N, K, quant)` a decode-GEMV opcode carries, or `None` if this is not one.
    ///
    /// # Why this lives on the opcode and not in each emitter
    ///
    /// The GEMM tuner's shape list was AUTHORED BY HAND, and that is exactly how GLM-5.2's
    /// prefill came to be 100% unmeasured while `tuned_tile_selection` kept passing — some
    /// qualified record existed, just never for GLM's shapes. The fix there was
    /// `PLOW_TUNE_DUMP`, which reads the demand back out of the compiler
    /// (`crates/devgen/src/lib.rs`, `GemmMeasurements::for_shape`).
    ///
    /// The GEMV path has the same exposure and thirty-odd emit sites across `devgen`'s
    /// `lib.rs` / `mla.rs` / `kda.rs`. Instrumenting those by hand would reproduce the
    /// original mistake one level down: a site added later is a shape the census never sees.
    /// So the hook goes at [`crate::devbuild::Builder::emit_dep`], the single choke point every
    /// emitter funnels through, and the layout knowledge lives HERE — beside the doc comments
    /// that define it — rather than being restated at the call site.
    ///
    /// Every `Gemv*` opcode carries `i0=M i1=N i2=K`. The one irregularity is the fused-QKV
    /// pair: [`DevOp::GemvQkv`]'s `i1/i3/i4` are `Nq/Nk/Nv` and [`DevOp::GemvQkvg`] adds
    /// `i5=Ng`, for three (resp. four) concatenated weight streams read against one staged `x`;
    /// the shape that governs their cost is the sum. They are separate FAMILIES for the reason
    /// below — a 3-stream and a 4-stream fusion at one `(M,N,K)` are different operations.
    ///
    /// `quant` is the encoding of the WEIGHT stream, spelled to match
    /// `kernelcaps::QuantScheme`'s `Debug` so the op-case keys of the two families are
    /// directly comparable. Note `GemvFp8Blk` reports `W8A8`: `QuantScheme` has no block-fp8
    /// variant, and the block grid is a scale layout rather than a different operand width.
    ///
    /// # `family` is part of the key, and leaving it out gives WRONG ANSWERS
    ///
    /// It was left out in the first draft, and `tunedb-gemv best` immediately printed
    /// nonsense: `TuneStore::best_for` ranks every `kernel_id` filed under one `op_case` and
    /// returns the fastest, so a plain `Gemv` and a fused `GemvQkv` at the same `(M, N, K)`
    /// were compared as if they were two implementations of one operation. They are not — the
    /// fused arm reads THREE weight streams against one staged `x` and produces three outputs.
    /// Whichever happened to be faster was reported as "the winner" for a shape the other one
    /// is the only legal answer for.
    ///
    /// That is the GEMM cell's structure applied where it does not hold: there, five tiles are
    /// genuinely interchangeable implementations of one GEMM. Here the arms are different ops.
    /// The family therefore goes in the key, which leaves exactly one kernel per case — an
    /// honest representation of a path that has no per-shape rung to select.
    pub fn gemv_case(self, i: &[u32; 8]) -> Option<(&'static str, u32, u32, u32, &'static str)> {
        let (fam, n, q) = match self {
            DevOp::Gemv | DevOp::GemvSz => ("gemv", i[1], "None"),
            DevOp::GemvArgmax => ("gemvargmax", i[1], "None"),
            DevOp::GemvGlu | DevOp::GemvGluSz => ("gemvglu", i[1], "None"),
            DevOp::GemvQkv => ("gemvqkv", i[1] + i[3] + i[4], "None"),
            DevOp::GemvQkvg => ("gemvqkvg", i[1] + i[3] + i[4] + i[5], "None"),
            DevOp::GemvFp8 => ("gemv", i[1], "W8A8"),
            DevOp::GemvFp8Blk => ("gemvblk", i[1], "W8A8"),
            DevOp::GemvGluFp8 => ("gemvglu", i[1], "W8A8"),
            DevOp::GemvMxfp4 => ("gemv", i[1], "Mxfp4"),
            DevOp::GemvGluMxfp4 => ("gemvglu", i[1], "Mxfp4"),
            DevOp::GemvQkvMxfp4 => ("gemvqkv", i[1] + i[3] + i[4], "Mxfp4"),
            DevOp::GemvQkvFp8 => ("gemvqkv", i[1] + i[3] + i[4], "W8A8"),
            _ => return None,
        };
        Some((fam, i[0], n, i[2], q))
    }
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
            fj: [
                self.f[0].to_bits(),
                self.f[1].to_bits() | self.j[0],
                self.j[1],
            ],
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
/// successor is a cross-GPU "partial ready" bump. See the design notes.
pub const SE_XCTR: u16 = 2;

/// This entry belongs to an opt-in pure `XReduceTwoShot` specialist segment.
/// Older runtimes ignore the bit and safely execute the segment on the primary interpreter.
pub const SE_XR_WAVE_RS: u16 = 4;

/// This entry belongs to an opt-in pure `KdaChunkIntra` wave-item segment.
/// Older runtimes ignore the bit and execute the unchanged instruction normally.
pub const SE_KDA_INTRA_WAVE_ITEMS: u16 = 8;

/// The same bit on a pure `KdaChunkCarry` segment selects the register-resident gfx950 carry
/// object. `flags` has no free bit left (4..12 hold [`SE_NPER_MASK`], 13..15 the domain), so
/// the opcode of the entry disambiguates; a runtime that predates this route refuses such a
/// packet as an impure wave-item segment rather than running it.
pub const SE_KDA_CARRY_REGSTATE: u16 = SE_KDA_INTRA_WAVE_ITEMS;

/// Shift of the per-(packet, L2 domain) slice count packed into [`StreamEnt::flags`].
///
/// Mirrors `PLOW_SE_NPER_SHIFT` in `runtime/common/dev_isa.h`; read the note there for why the
/// count is only knowable under `PLOW_L2_PLACE` and what the interpreter does with it. `flags` is
/// only ever read through masks, never compared whole, so bits 4..12 are free and the ABI does
/// not grow. Nine bits holds the 256-workgroup maximum; the emitter asserts rather than truncate.
pub const SE_NPER_SHIFT: u16 = 4;
/// Mask of the field at [`SE_NPER_SHIFT`] — bits 4..12.
pub const SE_NPER_MASK: u16 = 0x1FF0;

/// L2 locality domain carried independently from [`StreamEnt::seg`]. Bits
/// 13..15 hold the eight gfx94x/gfx95x XCDs while `seg` remains the ordered
/// ordered kernel-family segment. This lets a pure lean segment retain dynamic per-XCD GQ
/// windows instead of choosing between segmentation and locality.
pub const SE_DOMAIN_SHIFT: u16 = 13;
pub const SE_DOMAIN_MASK: u16 = 0xE000;

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
/// the design notes.
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
     * per segment with that segment's wave count; the interp runs only that segment's entries
     * (bounded by [`DevProgram::seg_ofs`] when the host builds it, otherwise by skipping every
     * entry whose seg != cur_seg).
     * 0 for a single-segment (unsegmented) program. Was `_pad`; same 16-bit slot. */
    pub seg: u16,
}

/// One request span in a ragged prefill pack.
///
/// Rows are dense in activation scratch (`row0..row0+n_rows`) but retain their
/// request-local KV coordinates and carried-state slot. The descriptor is model
/// independent; consumers that cannot honor it must execute the span through
/// the isolated prefill path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PrefillSpan {
    /// First row in the packed activation tensors.
    pub row0: u32,
    /// Real rows contributed by this request.
    pub n_rows: u32,
    /// Decode/KV slot that owns the request.
    pub slot: u32,
    /// [`PREFILL_SPAN_RESET_STATE`] when this is the request's first span.
    pub flags: u32,
    /// Request-local absolute KV row of `row0`.
    pub kv_row0: u32,
    /// Request-local KV length after this span.
    pub kv_len: u32,
    /// Slot used for recurrent state; explicit so KV and state layouts may diverge.
    pub state_slot: u32,
    /// Compiled prefill program/rung selected for the span.
    pub program: u32,
}

pub const PREFILL_SPAN_RESET_STATE: u32 = 1;

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
    /// Segmented dispatch: interp runs only this segment's entries.
    pub cur_seg: u32,
    /// L2-domain placement (`PLOW_L2_PLACE`): number of physical domains per
    /// ordered kernel-family segment, or `0`. A placed interpreter selects window
    /// `cur_seg * l2_domains + physical_domain`, so all domains drain concurrently
    /// inside each lean or ordinary segment launch.
    pub l2_domains: u32,
    /// Segments [`Self::seg_ofs`] is built for; the row stride there is `n_seg + 1`.
    /// Only read when `seg_ofs != 0`.
    ///
    /// This field and `l2_domains` both once occupied the single spare `_segpad` u32 — added on
    /// independent branches, each claiming it. They are independent axes (`l2_domains` windows
    /// the GLOBAL QUEUE by physical L2 domain; `n_seg` describes the STATIC per-CU `seg_ofs`
    /// table by wave-class segment), so both are kept and the struct grew 136 -> 144.
    pub n_seg: u32,
    /// Base counter ID of the two-level maintenance scratch — three u32 per (packet, L2 domain),
    /// carved out of the tail of the ordinary counter region. Mirrors `hier_base` in
    /// `runtime/common/dev_isa.h`; the layout and the reason it lives in the existing alignment
    /// pad (so `sizeof` stays 144 and no kernarg copy goes short) are documented there.
    /// Zero disables the hierarchy and every workgroup does its own cache maintenance.
    pub hier_base: u32,
    /// Global-queue interpreter (Experiment E1). Op-major stream, segment bounds, shared cursor.
    pub gq_stream: u64,
    pub gq_seg_ofs: u64,
    pub gq_cursor: u64,
    // ===== Cross-GPU (tensor-parallel) fields. Single-GPU runs leave these 0. =====
    // Appended AFTER `gq_cursor` so every existing field (notably `trace`) keeps its
    // offset — the ABI-lock test only sees the size grow. See the design notes.
    /// This rank's cross-GPU counter region (SYSTEM-scope, peer-mapped). Points INTO
    /// `peer_scratch[rank]`; the per-rank offset is `xctr - peer_scratch[rank]`.
    pub xctr: u64,
    /// `[n_gpu]` — each rank's peer-mapped reduction region base.
    pub peer_scratch: u64,
    /// This GPU's TP rank.
    pub rank: u32,
    /// TP degree (1 = single-GPU, cross-GPU fields unused).
    pub n_gpu: u32,
    /// STATIC-path per-(CU, segment) stream windows: `[n_cu][n_seg+1]` u32,
    /// row-major, indices relative to `stream_ofs[cu]`. `0` ⇒ the interpreter
    /// falls back to scanning the whole per-CU stream and filtering on `seg`.
    ///
    /// Built by [`crate::devbuild::static_seg_ofs`] at load time, NOT carried in
    /// the blob — see the field comment in `runtime/common/dev_isa.h` for why.
    pub seg_ofs: u64,
    /// Device `[n_prefill_spans]` ragged packed-prefill metadata, or 0 for the
    /// single-request path.
    pub prefill_spans: u64,
    /// Device `[n_prefill_rows]` parked-row mask, including the compiled rung's padded tail.
    pub prefill_parked: u64,
    pub n_prefill_spans: u32,
    /// Launched compiled row count `T`, including parked padding after the dense real spans.
    pub n_prefill_rows: u32,
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
const _: () = assert!(size_of::<PrefillSpan>() == 32);
const _: () = assert!(size_of::<TraceRec>() == 40);
const _: () = assert!(size_of::<DevProgram>() == 168);
