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
 * All four ops take (slice, nblk) where a standalone kernel would take (blockIdx.x, gridDim.x) —
 * the interpreter is persistent, grid == CU count, and an op "spread over N workgroups" appears
 * once in the instruction stream and N times in the per-CU streams.
 */
#ifndef PLOW_OP_KDA_H
#define PLOW_OP_KDA_H

#include "amd_common.h"

/* Gate activation modes, mirroring [fla]'s `safe_gate` switch (fla/ops/kda/gate.py:118-124). */
enum { PLOW_KDA_GATE_SOFTPLUS = 0, PLOW_KDA_GATE_LOWER_BOUND = 1 };
/* Flag bits for d_kda_state_step. */
enum { PLOW_KDA_F_QK_L2NORM = 1u };

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
 * Three independent causal depthwise convolutions of width W over H*D channels each (q, k and v,
 * concatenated into one `conv_dim = 3*H*D` axis), then an activation. `groups = hidden_size` makes
 * it depthwise, `padding = W-1` makes it causal, and there is no bias (the checkpoint ships no
 * *_conv1d.bias). This is what gives KDA local W-token mixing that a pure linear-attention
 * recurrence cannot express — it is 0.03% of the layer's MACs and it is not optional.
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
__device__ void d_kda_conv(bf16* __restrict__ out, const bf16* __restrict__ x,
                           const float* __restrict__ w, float* __restrict__ state, unsigned T,
                           unsigned conv_dim, unsigned W, unsigned act, unsigned slice,
                           unsigned nblk) {
    const unsigned chunk = (conv_dim + nblk - 1) / nblk;
    const unsigned c0 = slice * chunk;
    unsigned c1 = c0 + chunk;
    if (c1 > conv_dim) c1 = conv_dim;

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
            out[(size_t)t * conv_dim + c] = f2bf(act == 1u ? act_silu(y) : y);
        }
#pragma unroll
        for (unsigned j = 0; j < PLOW_KDA_WMAX; j++)
            if (j < Wc) state[(size_t)c * W + j] = win[j];
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
template <unsigned PL>
__device__ void d_kda_state_step_t(bf16* __restrict__ o, const bf16* __restrict__ q,
                                   const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   const float* __restrict__ g, const float* __restrict__ beta,
                                   float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                   unsigned BV, unsigned flags, float scale, unsigned slice,
                                   unsigned nblk, float* __restrict__ lds) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned ntile = D / BV; /* column tiles per head */
    const unsigned items = H * ntile;
    const unsigned cols_per_wave = BV / PLOW_WAVES; /* 2 at BV=16, PLOW_WAVES=8 */

    float* l_q = lds;         /* [D] */
    float* l_k = lds + D;     /* [D] */
    float* l_g = lds + 2 * D; /* [D] */

    for (unsigned it = slice; it < items; it += nblk) {
        const unsigned h = it / ntile;
        const unsigned tile = it % ntile;
        float* st_h = state + (size_t)h * D * D;

        for (unsigned t = 0; t < T; t++) {
            const size_t hd = (size_t)t * H * D + (size_t)h * D;
            /* Stage this head's q, k, g once per (item, token) and share across the waves. The L2
             * norm is a whole-head reduction, so it happens here, once, not per column. */
            float qs = 0.0f, ks = 0.0f;
            for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
                const float qv = bf2f(q[hd + d]), kv = bf2f(k[hd + d]);
                l_q[d] = qv;
                l_k[d] = kv;
                l_g[d] = __expf(g[hd + d]);
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

            const float b = beta[(size_t)t * H + h];
            for (unsigned c = 0; c < cols_per_wave; c++) {
                const unsigned j = tile * BV + wave * cols_per_wave + c; /* value column */
                float* col = st_h + (size_t)j * D;                       /* V-FIRST: [v][k] */

                /* decay, in registers: S' = exp(g) * S */
                float sc[PL];
                float pk = 0.0f;
#pragma unroll
                for (unsigned r = 0; r < PL; r++) {
                    const unsigned d = r * PLOW_WAVE + lane;
                    sc[r] = col[d] * l_g[d];
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
                    col[d] = sc[r];
                    pq += sc[r] * l_q[d]; /* read the UPDATED state */
                }
                pq = wave_sum(pq);
                if (lane == 0) o[hd + j] = f2bf(pq);
            }
            __syncthreads(); /* l_q/l_k/l_g are rewritten by the next token */
        }
    }
}

/* D is a runtime immediate, so select the compile-time lane depth here. D=128 (K3) is the only
 * shape any KDA checkpoint uses; the other rungs exist so a wrong D refuses loudly instead of
 * running the D=128 template on the wrong stride. */
__device__ void d_kda_state_step(bf16* __restrict__ o, const bf16* __restrict__ q,
                                 const bf16* __restrict__ k, const bf16* __restrict__ v,
                                 const float* __restrict__ g, const float* __restrict__ beta,
                                 float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                 unsigned BV, unsigned flags, float scale, unsigned slice,
                                 unsigned nblk, float* __restrict__ lds) {
    if (D == 128)
        d_kda_state_step_t<2>(o, q, k, v, g, beta, state, T, H, D, BV, flags, scale, slice, nblk,
                              lds);
    else if (D == 64)
        d_kda_state_step_t<1>(o, q, k, v, g, beta, state, T, H, D, BV, flags, scale, slice, nblk,
                              lds);
    else if (D == 256)
        d_kda_state_step_t<4>(o, q, k, v, g, beta, state, T, H, D, BV, flags, scale, slice, nblk,
                              lds);
}

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
 * lane, and nothing crosses a wave. At T=1 that is 96 rows, so blocks = 96 (37.5%) — acceptable
 * on an op that touches 12288 elements, and the alternative (folding it into op 102's epilogue)
 * needs a grid-wide barrier because a head's D outputs are spread over D/BV workgroups there.
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

#endif /* PLOW_OP_KDA_H */
