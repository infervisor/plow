/* op_kda.h — KDA (Kimi Delta Attention), the mixer in 69 of Kimi-K3's 93 layers.
 *
 * Spec: docs/kimi-k3-kda.md. Per head, per token, carrying a state S in [K,V]:
 *
 *     S  <-  (I - beta k k^T) . diag(exp(g)) . S  +  beta k v^T ;      o = S^T q
 *            \___ delta rule ___/  \_ forget gate _/  \_ write _/
 *
 * TWO COMPOSED MEMORY MECHANISMS, and conflating them is the fastest way to a plausible wrong
 * answer:
 *   - the forget gate diag(exp(g)) is UNTARGETED — it decays the whole state, per (head,
 *     key-channel), data-dependent, and bounded to [e^lb, 1) by gate_lower_bound;
 *   - the delta rule (I - beta k k^T) is TARGETED. The kernel L2-normalizes k, so ||k|| = 1 and
 *     this is EXACTLY I minus beta times the orthogonal projector onto k: it erases only the
 *     component of memory stored at key k and leaves everything orthogonal to k untouched.
 *
 * THE STATE IS A DECLARED HBM TENSOR, NOT REGISTERS. [H,D,D] f32 = 6.00 MiB per layer per
 * sequence, CONSTANT in context length (that is the whole architectural argument: 69 KDA layers
 * cost 0.44 GiB at 1M tokens where 24 MLA layers cost 27 GiB). A decode step is a
 * read-modify-write over it, the same kind of object as a KV ring.
 *
 * SLICE MAP FIRST, INNER LOOP SECOND. Every slow kernel in this tree failed the same way:
 * achieved % of ceiling ~= active-workgroup fraction. One workgroup per head is 96/256 = 37.5% at
 * TP1 and 24/256 = 9.4% at TP4 — worse than the MlaMergeFold defect that cost 8.69 ms of a
 * 34.68 ms token. So NOTHING here parallelizes over heads alone.
 *
 * These are the AMD arms. runtime/nvidia/op_mamba.cuh is the precedent to avoid, not to copy: it
 * is monolithic, emitted onto ONE CU, and has no arm in amd/interp.hip at all, so op 90 falls to
 * the silent dispatch default: on gfx950 and computes nothing.
 *
 * SIX ARMS, FOUR ALGORITHMS. Ops 109 and 110 are ops 88 and 102+89 re-sliced into fewer PACKETS,
 * not re-derived: 109 calls 88's own `kda_conv_range` per stream, and 110 is 102's body with one
 * `if constexpr`-shaped branch on where `g` comes from. That is deliberate — a KDA decode layer is
 * launch bound (a packet costs ~12 us in this interpreter, measured; the whole six-op chain's
 * arithmetic is a rounding error against that), so the fusion had to move packets without moving
 * arithmetic. Two bodies computing the same thing is how the transposed-state class of bug gets in.
 *
 * All six ops take (slice, nblk) where a standalone kernel would take (blockIdx.x, gridDim.x) —
 * the interpreter is persistent, grid == CU count, and an op "spread over N workgroups" appears
 * once in the instruction stream and N times in the per-CU streams.
 */
#ifndef PLOW_OP_KDA_H
#define PLOW_OP_KDA_H

#include "amd_common.h"

#ifndef PLOW_KDA_PF_STATE_RESIDENT
#define PLOW_KDA_PF_STATE_RESIDENT 0
#endif
#ifndef PLOW_KDA_CONV_STEP_DB
#define PLOW_KDA_CONV_STEP_DB 0
#endif
#define PLOW_KDA_F_JOURNAL 4u
#define PLOW_KDA_JOURNAL_BANKS 9u

#if PLOW_K3_SPEC_VERIFY
#define PLOW_KDA_JOURNAL_PARAM , const unsigned* __restrict__ journal_commit
#define PLOW_KDA_JOURNAL_ARG , journal_commit
#else
#define PLOW_KDA_JOURNAL_PARAM
#define PLOW_KDA_JOURNAL_ARG
#endif

/* Gate activation modes, mirroring [fla]'s `safe_gate` switch (fla/ops/kda/gate.py:118-124). */
enum { PLOW_KDA_GATE_SOFTPLUS = 0, PLOW_KDA_GATE_LOWER_BOUND = 1 };
/* Flag bits for d_kda_state_step. */
enum {
    PLOW_KDA_F_QK_L2NORM = 1u,
    /* THE ROWS ARE INDEPENDENT SEQUENCES, not consecutive tokens of one.
     *
     * A KDA layer's `T` rows mean two different things and the recurrence cannot tell them apart
     * from `T`: on a PREFILL program they are consecutive tokens of ONE sequence and must thread
     * through ONE state; on a BATCHED DECODE program they are B independent sequences, each of
     * which owns its own state. Sharing the state across the second kind runs sequence 1's token
     * into sequence 0's and produces fluent, plausible, WRONG output — no crash, no NaN.
     *
     * So the distinction is carried explicitly. Clear (every program that exists today) makes the
     * per-row state stride 0, the state pointer does not move, and the emitted code is unchanged.
     * See perf-data/k3-batched-decode-design.md §1. */
    PLOW_KDA_F_SEQ_ROWS = 2u,
};

__device__ __forceinline__ float kda_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

/* softplus(x) = log1p(exp(x)), evaluated in the numerically safe branch. [fla] uses
 * `tl.log(1 + tl.exp(x))` guarded to `x` for large x; matching that guard matters because the
 * unbounded gate branch feeds an exp() straight after. */
__device__ __forceinline__ float kda_softplus(float x) {
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

/* -------------------------------------------------------------------------------------------
 * op 88 — KDA short conv.
 *
 * ONE stream's causal depthwise convolution of width W over `conv_dim` channels, then an
 * activation. KDA has three such streams (q, k, v); this arm takes one, and op 109 takes all three
 * in one packet. `groups = hidden_size` makes it depthwise, `padding = W-1` makes it causal, and
 * there is no bias (the checkpoint ships no *_conv1d.bias). This is what gives KDA local W-token
 * mixing that a pure linear-attention recurrence cannot express — it is 0.03% of the layer's MACs
 * and it is not optional.
 *
 * `state` is the rolling input window, [conv_dim, W] f32, holding the last W inputs per channel
 * with the CURRENT token at slot W-1:
 *      state.roll(-1); state[:, W-1] = x_t; y = sum_j state[:, j] * w[:, j]
 * ([fla] short_conv.py:232-235). [vllm] instead keeps W-1 slots and prepends the current token;
 * both are correct and they differ by 36864 elements per layer. This is the [fla] convention
 * because the reference the numeric gate runs against is [fla].
 *
 * SLICE MAP: the conv is a W-tap STENCIL, not a scan — y_t depends on x_{t-W+1..t}, never on
 * y_{t-1} — so it is fully parallel over (t, channel). Only the window carry is sequential, and
 * that is per channel and lives in 4 registers. Channels are therefore the parallel axis and each
 * thread walks all T tokens of its own channel, which also produces the final `state` for free
 * with no second pass.
 *
 * Each block takes a CONTIGUOUS chunk of channels rather than a strided one: a channel's window is
 * 4 contiguous f32 = one global_load_dwordx4, and contiguous chunks keep the 512-byte-per-16-lane
 * coalescing while spreading the work over all 256 CUs. At conv_dim = 36864 that is 144 channels
 * per block — 144 of 512 lanes busy, which is a wave-level idle, not a CU-level one, and CU spread
 * is what the bandwidth wants.
 */
/* One stream's channels [c0, c1) of the conv, for THIS workgroup. Factored out so op 88 and op 109
 * share ONE body: the fused arm calls this on a per-stream sub-range, so "the fused conv equals
 * three separate convs" is true by construction and not by tolerance. `conv_dim` is the stream's
 * own channel stride, which is what makes the split legal — nothing in the loop couples channels. */
__device__ __forceinline__ void kda_conv_range(bf16* __restrict__ out, const bf16* __restrict__ x,
                                               const float* __restrict__ w,
                                               float* __restrict__ state, unsigned T,
                                               unsigned conv_dim, unsigned W, unsigned act,
                                               unsigned c0, unsigned c1, size_t bstride,
                                               const unsigned* __restrict__ parked
                                                   PLOW_KDA_JOURNAL_PARAM) {
#if PLOW_K3_SPEC_VERIFY
    if (journal_commit != nullptr) {
        const unsigned committed = *journal_commit;
        if (committed >= PLOW_KDA_JOURNAL_BANKS) __builtin_trap();
        const size_t bank_elems = (size_t)conv_dim * W;
        for (unsigned c = c0 + threadIdx.x; c < c1; c += PLOW_THREADS) {
            float win[8], tap[8];
            const unsigned width = W < 8 ? W : 8;
            const float* source = state + (size_t)committed * bank_elems + (size_t)c * W;
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                win[j] = j < width ? source[j] : 0.0f;
                tap[j] = j < width ? w[(size_t)c * W + j] : 0.0f;
            }
            for (unsigned t = 0; t < T; ++t) {
#pragma unroll
                for (unsigned j = 0; j + 1 < 8; ++j) win[j] = win[j + 1];
                win[width - 1] = bf2f(x[(size_t)t * conv_dim + c]);
                float y = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) y += win[j] * tap[j];
                st_act1(&out[(size_t)t * conv_dim + c], f2bf(act == 1u ? act_silu(y) : y));
                const unsigned target_bank = (committed + 1u + t) % PLOW_KDA_JOURNAL_BANKS;
                float* target = state + (size_t)target_bank * bank_elems + (size_t)c * W;
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    if (j < width) target[j] = win[j];
            }
        }
        return;
    }
#endif
    /* INDEPENDENT-SEQUENCE PATH. `bstride != 0` means the T rows are B separate sequences
     * (batched decode), so each row owns its own sliding window: load it, roll ONE token
     * through, store it back. The shared path below is the opposite and is the one every
     * program uses today — it loads the window once, rolls all T consecutive tokens of ONE
     * sequence through it, and stores once, which is the whole point of a conv state.
     *
     * Kept as a separate loop rather than a stride inside the shared one because the LOAD and
     * STORE move, not just the address: hoisting them out of the token loop is exactly what
     * makes the shared path correct, and exactly what makes it wrong for independent rows. */
    if (bstride) {
        for (unsigned c = c0 + threadIdx.x; c < c1; c += PLOW_THREADS) {
            enum { PLOW_KDA_WMAX_B = 8 };
            const unsigned Wc = W < PLOW_KDA_WMAX_B ? W : PLOW_KDA_WMAX_B;
            for (unsigned t = 0; t < T; t++) {
                /* Same contract as the recurrence's mask: a parked row must not have its
                 * convolution window shifted, or the sequence it belongs to resumes against a
                 * window holding tokens from nobody. */
                if (parked && parked[t]) continue;
                float* st = state + (size_t)t * bstride + (size_t)c * W;
                float win[PLOW_KDA_WMAX_B], tap[PLOW_KDA_WMAX_B];
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++) {
                    win[j] = j < Wc ? st[j] : 0.0f;
                    tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
                }
#pragma unroll
                for (unsigned j = 0; j + 1 < PLOW_KDA_WMAX_B; j++) win[j] = win[j + 1];
                win[Wc - 1] = bf2f(x[(size_t)t * conv_dim + c]);
                float y = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++) y += win[j] * tap[j];
                st_act1(&out[(size_t)t * conv_dim + c], f2bf(act == 1u ? act_silu(y) : y));
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++)
                    if (j < Wc) st[j] = win[j];
            }
        }
        return;
    }
    for (unsigned c = c0 + threadIdx.x; c < c1; c += PLOW_THREADS) {
        /* The window and the taps. W is 4 for K3 and the loop bound is a runtime value, so this is
         * written as a small fixed array; PLOW_KDA_WMAX bounds the register cost. */
        enum { PLOW_KDA_WMAX = 8 };
        float win[PLOW_KDA_WMAX], tap[PLOW_KDA_WMAX];
        const unsigned Wc = W < PLOW_KDA_WMAX ? W : PLOW_KDA_WMAX;
#pragma unroll
        for (unsigned j = 0; j < PLOW_KDA_WMAX; j++) {
            win[j] = j < Wc ? state[(size_t)c * W + j] : 0.0f;
            tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
        }
        for (unsigned t = 0; t < T; t++) {
            /* roll left, insert x_t at the newest slot */
#pragma unroll
            for (unsigned j = 0; j + 1 < PLOW_KDA_WMAX; j++) win[j] = win[j + 1];
            win[Wc - 1] = bf2f(x[(size_t)t * conv_dim + c]);
            float y = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < PLOW_KDA_WMAX; j++) y += win[j] * tap[j];
            /* activation AFTER the convolution (short_conv.py:55-72) */
            st_act1(&out[(size_t)t * conv_dim + c], f2bf(act == 1u ? act_silu(y) : y));
        }
#pragma unroll
        for (unsigned j = 0; j < PLOW_KDA_WMAX; j++)
            if (j < Wc) state[(size_t)c * W + j] = win[j];
    }
}

__device__ void d_kda_conv(bf16* __restrict__ out, const bf16* __restrict__ x,
                           const float* __restrict__ w, float* __restrict__ state, unsigned T,
                           unsigned conv_dim, unsigned W, unsigned act, unsigned slice,
                           unsigned nblk, size_t bstride) {
    const unsigned chunk = (conv_dim + nblk - 1) / nblk;
    const unsigned c0 = slice * chunk;
    unsigned c1 = c0 + chunk;
    if (c1 > conv_dim) c1 = conv_dim;
    if (c0 < c1)
        kda_conv_range(out, x, w, state, T, conv_dim, W, act, c0, c1, bstride, nullptr
#if PLOW_K3_SPEC_VERIFY
                       , nullptr
#endif
        );
}

/* -------------------------------------------------------------------------------------------
 * op 109 — the same conv over all THREE streams in one packet.
 *
 * WHY, given that op 88's own note argues three packets is three times the concurrency: because
 * at batch 1 that is not what three packets buys. `runtime/tests/kda_fuse_bench_gfx950.c` measures
 * a packet in this interpreter at ~12 us against a KDA chain whose entire arithmetic is a rounding
 * error — 414 packets of the six-op chain cost 5.03 ms over 69 layers at TP8, and the cost is
 * LINEAR in the packet count with a slope of 12.08 us and an intercept of 0.02 ms. Three
 * independent packets therefore cost three times one packet holding the same work. The
 * concurrency op 88 was protecting is real and is preserved here; what is deleted is two counter
 * gates per layer, 138 per token.
 *
 * THE MERGE IS ALONG THE OUTPUT AXIS. The block still takes a CONTIGUOUS chunk, now of the 3*C
 * concatenated channel axis, so per CU the channel count RISES (48 -> 144 at TP1, 6 -> 18 at TP8)
 * on the same 256 CUs. Nothing that ran in parallel starts running in sequence. That is the
 * `GemvQkvg` direction; `GLM_GROUP=1`, which collapsed disjoint CU slices into a loop for +2.88 ms,
 * is the other one.
 *
 * A chunk of the 3*C axis crosses at most two stream boundaries, so this is a 3-iteration loop
 * over intersections, each delegating to op 88's own body. The streams keep
 * SEPARATE buffers — nothing here assumes q|k|v are contiguous in memory, which they are not:
 * `GemvQkvg` writes three distinct handles.
 */
__device__ void d_kda_conv3(bf16* __restrict__ oq, bf16* __restrict__ ok, bf16* __restrict__ ov,
                            const bf16* __restrict__ xq, const bf16* __restrict__ xk,
                            const bf16* __restrict__ xv, const float* __restrict__ wq,
                            const float* __restrict__ wk, const float* __restrict__ wv,
                            float* __restrict__ sq, float* __restrict__ sk,
                            float* __restrict__ sv, unsigned T, unsigned C, unsigned W,
                            unsigned act, unsigned slice, unsigned nblk, size_t bstride,
                            const unsigned* __restrict__ parked PLOW_KDA_JOURNAL_PARAM) {
    const unsigned total = 3u * C;
    const unsigned chunk = (total + nblk - 1) / nblk;
    const unsigned g0 = slice * chunk;
    unsigned g1 = g0 + chunk;
    if (g1 > total) g1 = total;
    if (g0 >= g1) return;
    /* ROLLED, not unrolled. `kda_conv_range` is force-inlined and holds 2*PLOW_KDA_WMAX floats of
     * window and taps, so unrolling puts three copies of that in the function; rolled there is ONE
     * inline site and the pointer triples become selects. Measured on the K3 decode object it
     * changes NOTHING — 254 VGPR / occ 2 / 4 spill either way — so this is a code-size choice, not
     * a register one, and it is written down that way rather than claimed as a win. (The 4 spilled
     * VGPRs are not this op's: adding EITHER new arm alone produces them, and `noinline` on both
     * does not remove them. The K3=0 object is unchanged at 0.) */
#pragma unroll 1
    for (unsigned s = 0; s < 3u; s++) {
        const unsigned lo = s * C;
        const unsigned a = g0 > lo ? g0 - lo : 0u;         /* stream-local start */
        const unsigned bb = g1 > lo ? g1 - lo : 0u;        /* stream-local end   */
        const unsigned b = bb > C ? C : bb;
        if (a >= b) continue;
        bf16* o = s == 0 ? oq : (s == 1 ? ok : ov);
        const bf16* x = s == 0 ? xq : (s == 1 ? xk : xv);
        const float* w = s == 0 ? wq : (s == 1 ? wk : wv);
        float* st = s == 0 ? sq : (s == 1 ? sk : sv);
        kda_conv_range(o, x, w, st, T, C, W, act, a, b, bstride, parked
                           PLOW_KDA_JOURNAL_ARG);
    }
}

/* -------------------------------------------------------------------------------------------
 * op 89 — KDA gate pre-pass. Pure elementwise over [T,H,D] plus [T,H].
 *
 *   mode 1 (K3, gate_lower_bound = -5.0):
 *       g[t,h,d] = lb * sigmoid( exp(A_log[h]) * (g_raw[t,h,d] + dt_bias[h,d]) )
 *   mode 0 (Kimi-Linear and every released vLLM):
 *       g[t,h,d] = -exp(A_log[h]) * softplus( g_raw[t,h,d] + dt_bias[h,d] )
 *   both:   beta[t,h] = sigmoid( beta_raw[t,h] )
 *
 * The bounded branch clamps g to [lb, 0), so the per-step decay exp(g) lies in (e^lb, 1): the
 * state can never be zeroed by the gate in one step and can never grow. K3 is the FIRST
 * checkpoint to ship it — do not expect the Kimi Linear paper, or any released vLLM, to describe
 * it, and do not use them as an oracle for it.
 *
 * A_log is indexed PER HEAD; dt_bias is [H,D] row-major, per (h,d). They are different ranks and
 * swapping them is silent. The gate is a VECTOR over the key dimension — each of the 128 key
 * channels of each head forgets at its own data-dependent rate — while beta is ONE SCALAR PER
 * HEAD. Making beta per-channel, or A_log per-channel, both produce finite plausible output.
 *
 * This is factored out of both the decode and the prefill paths rather than fused into either.
 * [vllm] factors it out for decode and fuses it for prefill; factoring it out in BOTH costs one
 * [T,H*D] f32 round-trip and buys an independently testable op — which is the point, because a
 * gate bug is invisible downstream.
 */
__device__ void d_kda_gate(float* __restrict__ g, float* __restrict__ beta,
                           const bf16* __restrict__ g_raw, const bf16* __restrict__ beta_raw,
                           const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                           unsigned T, unsigned H, unsigned D, unsigned mode, float lb,
                           unsigned slice, unsigned nblk) {
    const size_t n = (size_t)T * H * D;
    const size_t chunk = (n + nblk - 1) / nblk;
    const size_t i0 = (size_t)slice * chunk;
    size_t i1 = i0 + chunk;
    if (i1 > n) i1 = n;

    for (size_t i = i0 + threadIdx.x; i < i1; i += PLOW_THREADS) {
        const unsigned hd = (unsigned)(i % ((size_t)H * D)); /* h*D + d */
        const unsigned h = hd / D;
        const float a = __expf(A_log[h]);
        const float s = bf2f(g_raw[i]) + dt_bias[hd];
        g[i] = (mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a * s)
                                                   : -a * kda_softplus(s);
    }
    /* beta: T*H elements, its own chunking so it is not stranded on block 0. */
    const size_t nb = (size_t)T * H;
    const size_t cb = (nb + nblk - 1) / nblk;
    const size_t b0 = (size_t)slice * cb;
    size_t b1 = b0 + cb;
    if (b1 > nb) b1 = nb;
    for (size_t i = b0 + threadIdx.x; i < b1; i += PLOW_THREADS)
        beta[i] = kda_sigmoid(bf2f(beta_raw[i]));
}

/* -------------------------------------------------------------------------------------------
 * op 102 — KDA gated delta-rule state update. The core, and a read-modify-write on `state`.
 *
 * THE STATE IS V-FIRST: state[h][v][k], NOT [h][k][v]. K3 passes transpose_state_layout=True
 * (renamed state_v_first upstream). Since V == K == 128 the byte count is identical either way,
 * so a transposed state is garbage WITH EXACTLY THE RIGHT NORM. No magnitude check finds it; the
 * stride arithmetic below IS the assertion.
 *
 * V-first is also what makes the tiling free, via two facts that compose:
 *   1. a v-column (fixed h, fixed v, all K) is 512 CONTIGUOUS bytes;
 *   2. BOTH reductions in the step (S'^T k and S^T q) sum over k for a fixed v, so each output
 *      element is a private, contiguous, 512-byte dot product.
 *
 * LANE MAP. docs/kimi-k3-kda.md §7.2 says "no cross-lane reduction anywhere" AND "4-8 VGPRs/lane";
 * those are inconsistent, because BV*D/512 f32 per lane means 512/BV lanes cooperate on a column
 * and the reduction over k does cross lanes. The resolution keeps every number in that table and
 * drops only the absolute claim:
 *
 *     ONE WAVE OWNS ONE COLUMN. D = 128 = 64 lanes x 2, so a lane holds 2 f32 of state and both
 *     reductions are wave_sum — 6 shuffle steps, no LDS, no __syncthreads, and the whole column is
 *     one wave's private business. Nothing crosses a WAVE, which is the property that matters.
 *
 * Per lane: Sc[2] + q[2] + k[2] + g[2] = 8 f32.
 *
 * SLICE MAP: work item = (head, tile of BV value columns) => H*D/BV items. At H=96, D=128, BV=16
 * that is 768 items over 256 blocks, 3 each — blocks = 256, 100% fill. Blocks stride over items
 * so every block gets the same count. q/k/g for the item's head are staged in LDS once (3*D f32 =
 * 1.5 KiB) and reused by all PLOW_WAVES waves, so the per-head broadcast operands are re-read
 * D/BV times per layer instead of D times.
 *
 * ORDER OF OPERATIONS follows the reference kernel exactly (fla fused_recurrent.py:174-198):
 * decay is applied BEFORE the delta correction is computed, so u is the error against the ALREADY
 * DECAYED state; and o is read off the UPDATED state. The algebraic shortcut
 * o = S'^T q + beta (k.q) u is equivalent but is NOT used — the state is already in registers, so
 * the second pass is free, and matching the reference's association removes a source of fp32
 * divergence rather than trading it for nothing.
 *
 * L2 norm (flags bit 0): eps is INSIDE the sqrt, x / sqrt(sum x^2 + 1e-6), not x / (norm + eps).
 * q is then scaled by `scale` and k is NOT. ||k|| = 1 is load-bearing: it is what makes the delta
 * term an exact rank-1 projector rather than an approximate one.
 *
 * T > 1 runs the same recurrence serially. That is exact at any T — it is the reference's
 * `fused_recurrent` path, which fla uses for q_len == 1 but which is valid for all T — and it is
 * how prefill/decode agreement is tested without a second algorithm.
 */
/* PL = D / PLOW_WAVE, the state elements a lane holds, as a COMPILE-TIME bound. It has to be:
 * `sc[]` is indexed in the inner loop, and a runtime-bounded local array lands in scratch, which
 * is exactly the spill this whole tiling exists to avoid. D=128 => PL=2. */
/* GATE folds op 89 in (op 110). It changes only how `l_g[d]` and `b` are OBTAINED — the slice map,
 * the item map, the LDS layout and every line of the recurrence are shared, which is the point:
 * there is one body, so the fused and unfused paths cannot drift. `g` was an f32 HBM round trip of
 * exactly the expression computed inline here, and an f32 store/load is exact, so the two are
 * BIT-identical rather than merely close. */
template <unsigned PL, bool GATE>
__device__ void d_kda_state_step_t(bf16* __restrict__ o, const bf16* __restrict__ q,
                                   const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   const float* __restrict__ g, const float* __restrict__ beta,
                                   const bf16* __restrict__ g_raw,
                                   const bf16* __restrict__ beta_raw,
                                   const float* __restrict__ a_log,
                                   const float* __restrict__ dt_bias, unsigned gate_mode, float lb,
                                   float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                   unsigned BV, unsigned flags, float scale, unsigned slice,
                                   unsigned nblk, float* __restrict__ lds, size_t bstride,
                                   const unsigned* __restrict__ parked PLOW_KDA_JOURNAL_PARAM) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned ntile = D / BV; /* column tiles per head */
    const unsigned items = H * ntile;
    const unsigned cols_per_wave = BV / PLOW_WAVES; /* 2 at BV=16, PLOW_WAVES=8 */

    /* THE ROW AXIS IS PARALLEL WHEN THE ROWS ARE INDEPENDENT SEQUENCES, AND SERIAL OTHERWISE.
     *
     * `bstride != 0` means the T rows are B separate sequences, each with its own carried state at
     * `state + t*bstride`. Nothing in row t's recurrence reads row t-1's, so t is a WORK-ITEM axis
     * exactly like h and tile. `bstride == 0` is a prefill: the T rows are consecutive tokens of
     * ONE sequence threading through ONE state, and there t MUST stay serial.
     *
     * Folding t in is what makes batched decode scale. The item map used to be `H * ntile` at
     * every batch -- 192 items at TP8 with BV=8 -- so a B=16 decode ran 16 rows SERIALLY inside a
     * workgroup count that did not know B existed, on 69 of K3's 93 layers, while 64 of 256 CUs
     * sat idle. The batched-decode design doc asked for "an OUTER slot dimension"; what shipped
     * first was the per-row STRIDE without the axis. */
    const unsigned trep = bstride ? T : 1u;
    const unsigned nitem = items * trep;
#if PLOW_K3_SPEC_VERIFY
    const bool journal = (flags & PLOW_KDA_F_JOURNAL) != 0;
    if (journal && journal_commit == nullptr) __builtin_trap();
    const unsigned committed = journal ? *journal_commit : 0u;
    if (journal && committed >= PLOW_KDA_JOURNAL_BANKS) __builtin_trap();
    const size_t journal_stride = (size_t)H * D * D;
#endif

    float* l_q = lds;         /* [D] */
    float* l_k = lds + D;     /* [D] */
    float* l_g = lds + 2 * D; /* [D] */

    for (unsigned it = slice; it < nitem; it += nblk) {
        /* Row-major over (row, h, tile), so consecutive slices stay inside one row's state when
         * the row axis is folded in — the same locality the un-folded map had. */
        const unsigned row = bstride ? it / items : 0u;
        const unsigned base = bstride ? it % items : it;
        const unsigned h = base / ntile;
        const unsigned tile = base % ntile;
        float* st_h = state + (size_t)h * D * D;

        /* dt_bias is [H,D] row-major and A_log is PER HEAD — different ranks, and swapping them is
         * silent. Hoisted out of the token loop because neither depends on t. */
        const size_t dtb = (size_t)h * D;
        const float a_h = GATE ? __expf(a_log[h]) : 0.0f;

#if PLOW_KDA_PF_STATE_RESIDENT
        /* TP8 prefill assigns exactly one value column to each wave. Keep that column's two
         * D=128 lane elements live across the serial token recurrence instead of round-tripping
         * them through HBM at every token. Independent decode rows must retain the per-row path. */
        const bool resident = PL == 2 && !bstride && BV == PLOW_WAVES
#if PLOW_K3_SPEC_VERIFY
                              && !journal
#endif
            ;
        float resident_sc[PL];
        float* resident_col = nullptr;
        if (resident) {
            const unsigned j = tile * BV + wave;
            resident_col = st_h + (size_t)j * D;
#pragma unroll
            for (unsigned r = 0; r < PL; r++)
                resident_sc[r] = resident_col[r * PLOW_WAVE + lane];
        }
#endif

        for (unsigned t = bstride ? row : 0u; t < (bstride ? row + 1u : T); t++) {
            /* PER-ROW PARKED MASK (non-zero = skip). Only a sequence-rows program supplies one,
             * and skipping the
             * row here is the whole point: this recurrence reads AND writes `state[row]` on every
             * dispatch, so a row the server has parked -- a slot in the middle of a chunked
             * prefill, or an idle slot -- would otherwise have its carried state advanced by a
             * garbage token. An append-only KV cache tolerates that (an idle row rewrites a row
             * nothing reads); a recurrence does not. `nullptr` OR an all-zero mask = every row
             * participates, so forgetting to publish one is safe by construction. */
            if (parked && parked[t]) continue;
            const size_t hd = (size_t)t * H * D + (size_t)h * D;
            /* Stage this head's q, k, g once per (item, token) and share across the waves. The L2
             * norm is a whole-head reduction, so it happens here, once, not per column. */
            float qs = 0.0f, ks = 0.0f;
            for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
                const float qv = bf2f(q[hd + d]), kv = bf2f(k[hd + d]);
                l_q[d] = qv;
                l_k[d] = kv;
                float gv;
                if (GATE) {
                    /* op 89's body, verbatim, evaluated where its only consumer already is. */
                    const float sgm = bf2f(g_raw[hd + d]) + dt_bias[dtb + d];
                    gv = (gate_mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a_h * sgm)
                                                                  : -a_h * kda_softplus(sgm);
                } else {
                    gv = g[hd + d];
                }
                l_g[d] = __expf(gv);
                qs += qv * qv;
                ks += kv * kv;
            }
            if (flags & PLOW_KDA_F_QK_L2NORM) {
                /* D <= PLOW_THREADS, so each lane held at most one element; the block reduction is
                 * over whatever the loop above accumulated. eps INSIDE the sqrt. */
                qs = block_sum(qs, lds + 3 * D);
                ks = block_sum(ks, lds + 3 * D + PLOW_WAVES);
                const float rq = scale / sqrtf(qs + 1e-6f), rk = 1.0f / sqrtf(ks + 1e-6f);
                for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
                    l_q[d] *= rq;
                    l_k[d] *= rk;
                }
            } else {
                for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) l_q[d] *= scale;
            }
            __syncthreads();

            /* beta is ONE SCALAR PER HEAD, not per channel. Making it per-channel produces finite
             * plausible output, which is why op 89's note says so twice. */
            const float b = GATE ? kda_sigmoid(bf2f(beta_raw[(size_t)t * H + h]))
                                 : beta[(size_t)t * H + h];
            for (unsigned c = 0; c < cols_per_wave; c++) {
                const unsigned j = tile * BV + wave * cols_per_wave + c; /* value column */
                /* PER-ROW STATE. `bstride` is 0 for a PREFILL program, where the T rows are
                 * consecutive tokens of ONE sequence and the recurrence must thread them through
                 * one state — the pointer then does not move and the emitted code is unchanged.
                 * It is `H*D*D` for a BATCHED DECODE program, where the rows are B INDEPENDENT
                 * sequences and sharing a state would run sequence 1's token into sequence 0's.
                 * That distinction is invisible in `T` alone, which is why it is its own
                 * parameter and not inferred (see perf-data/k3-batched-decode-design.md §1). */
                float* col = st_h + (size_t)t * bstride + (size_t)j * D; /* V-FIRST: [v][k] */
                float* out_col = col;
#if PLOW_K3_SPEC_VERIFY
                if (journal) {
                    const unsigned source_bank = (committed + t) % PLOW_KDA_JOURNAL_BANKS;
                    const unsigned target_bank = (committed + 1u + t) % PLOW_KDA_JOURNAL_BANKS;
                    col = state + (size_t)source_bank * journal_stride + (size_t)h * D * D +
                          (size_t)j * D;
                    out_col = state + (size_t)target_bank * journal_stride +
                              (size_t)h * D * D + (size_t)j * D;
                }
#endif

                /* decay, in registers: S' = exp(g) * S */
                float sc[PL];
                float pk = 0.0f;
#pragma unroll
                for (unsigned r = 0; r < PL; r++) {
                    const unsigned d = r * PLOW_WAVE + lane;
#if PLOW_KDA_PF_STATE_RESIDENT
                    sc[r] = (resident ? resident_sc[r] : col[d]) * l_g[d];
#else
                    sc[r] = col[d] * l_g[d];
#endif
                    pk += sc[r] * l_k[d];
                }
                pk = wave_sum(pk); /* S'^T k, over k, inside one wave */

                const float u = bf2f(v[hd + j]) - pk;
                const float bu = b * u;

                float pq = 0.0f;
#pragma unroll
                for (unsigned r = 0; r < PL; r++) {
                    const unsigned d = r * PLOW_WAVE + lane;
                    sc[r] += bu * l_k[d]; /* rank-1 write */
#if PLOW_KDA_PF_STATE_RESIDENT
                    if (resident)
                        resident_sc[r] = sc[r];
                    else
                        out_col[d] = sc[r];
#else
                    out_col[d] = sc[r];
#endif
                    pq += sc[r] * l_q[d]; /* read the UPDATED state */
                }
                pq = wave_sum(pq);
                if (lane == 0) o[hd + j] = f2bf(pq);
            }
            __syncthreads(); /* l_q/l_k/l_g are rewritten by the next token */
        }
#if PLOW_KDA_PF_STATE_RESIDENT
        if (resident) {
#pragma unroll
            for (unsigned r = 0; r < PL; r++)
                resident_col[r * PLOW_WAVE + lane] = resident_sc[r];
        }
#endif
    }
}

/* D is a runtime immediate, so select the compile-time lane depth here. D=128 (K3) is the only
 * shape any KDA checkpoint uses; the other rungs exist so a wrong D refuses loudly instead of
 * running the D=128 template on the wrong stride. */
#define PLOW_KDA_STEP_RUNGS(GATE_)                                                                \
    if (D == 128)                                                                                 \
        d_kda_state_step_t<2, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked PLOW_KDA_JOURNAL_ARG);                           \
    else if (D == 64)                                                                             \
        d_kda_state_step_t<1, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked PLOW_KDA_JOURNAL_ARG);                           \
    else if (D == 256)                                                                            \
        d_kda_state_step_t<4, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked PLOW_KDA_JOURNAL_ARG);

__device__ void d_kda_state_step(bf16* __restrict__ o, const bf16* __restrict__ q,
                                 const bf16* __restrict__ k, const bf16* __restrict__ v,
                                 const float* __restrict__ g, const float* __restrict__ beta,
                                 float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                 unsigned BV, unsigned flags, float scale, unsigned slice,
                                 unsigned nblk, float* __restrict__ lds, size_t bstride,
                                 const unsigned* __restrict__ parked PLOW_KDA_JOURNAL_PARAM) {
    const bf16 *g_raw = nullptr, *beta_raw = nullptr;
    const float *a_log = nullptr, *dt_bias = nullptr;
    const unsigned gate_mode = 0;
    const float lb = 0.0f;
    PLOW_KDA_STEP_RUNGS(false)
}

/* op 110 — the same recurrence with op 89's gate inlined. Same rungs, same body, same slice map;
 * `g`/`beta` are absent by construction rather than by a null check, because this op has no slot
 * that could name them. */
__device__ void d_kda_state_step_g(bf16* __restrict__ o, const bf16* __restrict__ q,
                                   const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   const bf16* __restrict__ g_raw,
                                   const bf16* __restrict__ beta_raw,
                                   const float* __restrict__ a_log,
                                   const float* __restrict__ dt_bias, unsigned gate_mode, float lb,
                                   float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                   unsigned BV, unsigned flags, float scale, unsigned slice,
                                   unsigned nblk, float* __restrict__ lds, size_t bstride,
                                   const unsigned* __restrict__ parked PLOW_KDA_JOURNAL_PARAM) {
    const float *g = nullptr, *beta = nullptr;
    PLOW_KDA_STEP_RUNGS(true)
}
#undef PLOW_KDA_STEP_RUNGS

#if PLOW_KDA_CONV_STEP_DB
__device__ __forceinline__ bf16 kda_conv_db_one(
    const bf16* raw, const float* weight, const float* source, float* target, unsigned channel,
    unsigned W, bool write_state) {
    float win[8], tap[8];
    const unsigned width = W < 8 ? W : 8;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        win[j] = j < width ? source[(size_t)channel * W + j] : 0.0f;
        tap[j] = j < width ? weight[(size_t)channel * W + j] : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j + 1 < 8; ++j) win[j] = win[j + 1];
    win[width - 1] = bf2f(raw[channel]);
    float value = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) value += win[j] * tap[j];
    if (write_state) {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
            if (j < width) target[(size_t)channel * W + j] = win[j];
    }
    return f2bf(act_silu(value));
}

/* B1-only Conv3 + StateStepG candidate. The old and new conv-window banks are distinct for the
 * whole packet, so every value tile can read the old q/k window before tile 0 publishes the next
 * one. This is the cross-workgroup race an in-place fusion cannot avoid. */
__device__ void d_kda_conv_state_step_g(
    bf16* output, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const bf16* gate_raw, const bf16* beta_raw, float* state, const float* wq, const float* wk,
    const float* wv, const float* csq_source, const float* csk_source,
    const float* csv_source, float* csq_target, float* csk_target, float* csv_target,
    const float* a_log, const float* dt_bias, unsigned H, unsigned D, unsigned BV, unsigned W,
    unsigned flags, unsigned gate_mode, float scale, float lb, unsigned slice, unsigned nblk,
    float* lds) {
    if (H != 12 || D != 128 || BV != 8 || W != 4 || !(flags & PLOW_KDA_F_QK_L2NORM))
        __builtin_trap();
    const unsigned items = H * D / BV;
    if (slice >= items) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    const unsigned h = slice / (D / BV);
    const unsigned tile = slice % (D / BV);
    float* l_q = lds;
    float* l_k = lds + D;
    float* l_g = lds + 2u * D;
    float* l_v = lds + 3u * D + 2u * PLOW_WAVES;
    const unsigned hd0 = h * D;
    float qsum = 0.0f, ksum = 0.0f;
    if (tid < D) {
        const unsigned channel = hd0 + tid;
        const bf16 q = kda_conv_db_one(q_raw, wq, csq_source, csq_target, channel, W, tile == 0);
        const bf16 k = kda_conv_db_one(k_raw, wk, csk_source, csk_target, channel, W, tile == 0);
        l_q[tid] = bf2f(q);
        l_k[tid] = bf2f(k);
        float gate;
        const float gate_input = bf2f(gate_raw[channel]) + dt_bias[channel];
        if (gate_mode == PLOW_KDA_GATE_LOWER_BOUND)
            gate = lb * kda_sigmoid(__expf(a_log[h]) * gate_input);
        else
            gate = -__expf(a_log[h]) * kda_softplus(gate_input);
        l_g[tid] = __expf(gate);
        qsum = l_q[tid] * l_q[tid];
        ksum = l_k[tid] * l_k[tid];
    }
    if (tid < BV) {
        const unsigned channel = hd0 + tile * BV + tid;
        const bf16 v = kda_conv_db_one(v_raw, wv, csv_source, csv_target, channel, W, true);
        l_v[tid] = bf2f(v);
    }
    qsum = block_sum(qsum, lds + 3u * D);
    ksum = block_sum(ksum, lds + 3u * D + PLOW_WAVES);
    const float rq = scale / sqrtf(qsum + 1e-6f);
    const float rk = 1.0f / sqrtf(ksum + 1e-6f);
    if (tid < D) {
        l_q[tid] *= rq;
        l_k[tid] *= rk;
    }
    __syncthreads();

    const unsigned j = tile * BV + wave;
    float* column = state + (size_t)h * D * D + (size_t)j * D;
    float sc[2];
    float pk = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned d = r * PLOW_WAVE + lane;
        sc[r] = column[d] * l_g[d];
        pk += sc[r] * l_k[d];
    }
    pk = wave_sum(pk);
    const float update = kda_sigmoid(bf2f(beta_raw[h])) * (l_v[wave] - pk);
    float pq = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
        const unsigned d = r * PLOW_WAVE + lane;
        sc[r] += update * l_k[d];
        column[d] = sc[r];
        pq += sc[r] * l_q[d];
    }
    pq = wave_sum(pq);
    if (lane == 0) output[hd0 + j] = f2bf(pq);
    (void)nblk;
}
#endif

/* -------------------------------------------------------------------------------------------
 * op 103 — KDA output gate. y[h,d] = RMSNorm_D(o[h,:])[d] * sigmoid(g_raw[h,d]).
 *
 * FusedRMSNormGated(head_dim, eps, activation='sigmoid'). Three things here are easy to get
 * backwards and all three yield plausible-but-wrong output:
 *   - the norm is over D = 128 INSIDE a head, not over H*D = 12288;
 *   - its weight is a single [D] f32 vector SHARED by all H heads (the checkpoint ships exactly
 *     one o_norm.weight per layer, not one per head);
 *   - the sigmoid is applied to the RAW g_proj output and the gate multiplies AFTER the norm, not
 *     before.
 *
 * One wave per (token, head) row: T*H items, the reduction is a wave_sum over D/64 elements per
 * lane, and nothing crosses a wave. The packet therefore needs ceil(T*H/PLOW_WAVES) workgroups:
 * 12 at TP1 B1 and 2 at TP8 B1. Folding this into op 102's epilogue instead needs a grid-wide
 * barrier because a head's D outputs are spread over D/BV workgroups there.
 */
__device__ void d_kda_gated_norm(bf16* __restrict__ y, const bf16* __restrict__ o,
                                 const float* __restrict__ norm_w, const bf16* __restrict__ g_raw,
                                 unsigned T, unsigned H, unsigned D, float eps, unsigned slice,
                                 unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned rows = T * H;
    for (unsigned r = slice * PLOW_WAVES + wave; r < rows; r += nblk * PLOW_WAVES) {
        const size_t base = (size_t)r * D;
        float ss = 0.0f;
        for (unsigned d = lane; d < D; d += PLOW_WAVE) {
            const float x = bf2f(o[base + d]);
            ss += x * x;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)D + eps);
        for (unsigned d = lane; d < D; d += PLOW_WAVE)
            y[base + d] = f2bf(bf2f(o[base + d]) * inv * norm_w[d] * kda_sigmoid(bf2f(g_raw[base + d])));
    }
}

#undef PLOW_KDA_JOURNAL_ARG
#undef PLOW_KDA_JOURNAL_PARAM
#endif /* PLOW_OP_KDA_H */
