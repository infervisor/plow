/* op_hyperconn.h — GLM-5.3-Flash's hyper-connections (mHC), ops 121/122.
 *
 * Ported from vLLM's real reference math (`vllm.model_executor.kernels.mhc.torch::
 * mhc_pre_torch`/`mhc_post_torch` in the vendor image), not derived from prose. Verified
 * against a numerical oracle (tiny synthetic shapes, hand-checkable) BEFORE this file was
 * written — see the campaign notes under `perf-data/` for the exact numbers and how they
 * were generated. Full packet contract: `DevOp::HyperConnPre`/`HyperConnPost`'s doc
 * comments in `crates/packet/src/dev.rs`, mirrored in `runtime/common/dev_isa.h`.
 *
 * Every layer expands the single residual stream into `n=4` parallel streams
 * (`hc_mult`), mixed each stage through a Sinkhorn-Knopp-normalized (doubly stochastic)
 * combine matrix — the "manifold-constrained" part. `n` is 4 for every GLM-5.3-Flash
 * shape shipped; the fixed-size scratch below assumes it and the emitter must assert it.
 *
 * ONE WORKGROUP PER TOKEN, and the tiny `n3 = 2n+n^2 = 24`-wide post-processing (split,
 * scale, sigmoid, softmax + the Sinkhorn loop) runs on lane 0 ALONE, scalar. That is
 * deliberate, not an oversight: n=4 makes this a 24-element computation, and
 * parallelizing an alternating row/column normalize loop across the workgroup would
 * trade real synchronization risk in the trickiest math in this file for no measurable
 * throughput — the same tradeoff `op_kda.h`'s `emit_kda_mixer_ex` already takes for P5/P6
 * ("concatenating them onto a 49152-wide sweep would buy two gates and hand two of the
 * 256 CUs a ragged tail"). The two genuinely large per-token reductions — the residual's
 * own sum of squares over `n*hidden`, and the pre-gate-weighted output sum over `hidden`
 * — DO use the full workgroup, via `block_sum`/a direct strided parallel loop.
 *
 * VERIFIED on gfx950 hardware (2026-09-01) against `/home/shaswot/plow-work/
 * mhc_oracle.npz` via `runtime/tests/hyperconn_gfx950_test.hip`: 0 mismatches, max |err|
 * = 1.19e-7 (f32 machine epsilon — bit-exact) across `post_mix`, `comb_mix`,
 * `layer_input`, and the paired `new_residual`. The Sinkhorn loop, softmax, sigmoid, and
 * both weighted-sum directions all check out against vLLM's real reference, not just by
 * inspection.
 */
#ifndef PLOW_OP_HYPERCONN_H
#define PLOW_OP_HYPERCONN_H

#include "amd_common.h"

/* GLM-5.3-Flash's real deployed default (verified against vLLM's `Glm5NextConfig` class,
 * not the more obvious 1.0 — its checkpoint's config.json omits the field entirely). A
 * compile-time constant because `DevInst` has only two `f` slots and both are already
 * spent on `rms_eps`/`hc_eps`. The emitter MUST assert the config's value matches this
 * before ever selecting this kernel — see the opcode's doc comment. */
#define PLOW_HC_POST_MULT 2.0f

/* PLOW_HC_SINKHORN_RCP — the Sinkhorn normalize's divisor is loop-invariant across the row
 * (or column) it normalizes, but an IEEE `f32` divide is NOT strength-reducible to a
 * reciprocal-and-multiply by the compiler, so the shipped body issues one full `v_div_*`
 * expansion per matrix element: at n=4 and sinkhorn_repeat=20 that is 640 of them per token,
 * on lane 0, serially.
 *
 *   0  SHIPPED. One IEEE divide per element. This is the arm the gfx950 oracle run signed off.
 *   1  One IEEE divide for the row/column reciprocal, then multiplies: 640 divides -> 160.
 *   2  As 1, but the reciprocal is `v_rcp_f32` (~1 ULP): 0 divides.
 *
 * 1 and 2 CHANGE ROUNDING and are default-off — the standing policy on this branch for any
 * arm that moves a bit (`PLOW_GEMV_MFMA4` is the precedent). See the measured instruction
 * counts, the oracle error and the null result in `docs/amd/instruction-cost-arms-20260908.md`. */
#ifndef PLOW_HC_SINKHORN_RCP
#define PLOW_HC_SINKHORN_RCP 0
#endif
#if PLOW_HC_SINKHORN_RCP < 0 || PLOW_HC_SINKHORN_RCP > 2
#error "PLOW_HC_SINKHORN_RCP must be 0, 1 or 2"
#endif

/* The reciprocal arms' normalize step. Arm 0 does NOT go through here — its four call sites keep
 * the shipped expression character for character below, so that the default build's codegen is
 * not merely equivalent but IDENTICAL. That is not pedantry: an earlier version of this file
 * routed arm 0 through a `__forceinline__` helper, which produced the same opcode histogram and
 * the same VGPR count and STILL moved 7 register assignments inside the outlined
 * `d_hyperconn_pre` of all 14 K3 objects. Equivalent-but-different codegen is exactly what a
 * default is not allowed to be here.
 *
 * `eps` is added AFTER the multiply for the initial softmax (`x/s + eps`) and folded into the
 * divisor by the caller for every Sinkhorn pass (`x/(s + eps)`), which is why ADD_EPS is a
 * template parameter and not a `0.0f` addend — `x + 0.0f` is not removable under IEEE. */
#if PLOW_HC_SINKHORN_RCP
template <bool ADD_EPS>
__device__ __forceinline__ void hc_normalize(float* v, unsigned base, unsigned stride,
                                             unsigned cnt, float den, float eps) {
#if PLOW_HC_SINKHORN_RCP == 2
    const float r = __builtin_amdgcn_rcpf(den);
#else
    const float r = 1.0f / den;
#endif
    for (unsigned k = 0; k < cnt; k++) {
        const float q = v[base + k * stride] * r;
        v[base + k * stride] = ADD_EPS ? q + eps : q;
    }
}
#endif

/* The n = 4 combine matrix's softmax + Sinkhorn in REGISTERS, one lane, fully unrolled. The
 * LDS form below is the same arithmetic in the same order (plain division, so arm 0 only), but
 * every one of its ~1300 reads and writes is a dependent LDS round trip on a single lane, which
 * made this op ~110 us per token — the longest link of the serial mHC chain at decode (90 per
 * step) and 15% of an 8192-token prefill. Results are bit-identical. */
#if !PLOW_HC_SINKHORN_RCP
__device__ __forceinline__ void hc_sinkhorn4_reg(const float* __restrict__ mrow, float inv,
                                                 float scale, const float* __restrict__ base,
                                                 float hc_eps, unsigned sinkhorn_repeat,
                                                 float* __restrict__ comb_out) {
    float c[16];
#pragma unroll
    for (unsigned i = 0; i < 4; i++) {
        float m = -3.0e38f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const float v = (mrow[8 + i * 4 + j] * inv) * scale + base[i * 4 + j];
            c[i * 4 + j] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const float e = expf(c[i * 4 + j] - m);
            c[i * 4 + j] = e;
            s += e;
        }
#pragma unroll
        for (unsigned j = 0; j < 4; j++) c[i * 4 + j] = c[i * 4 + j] / s + hc_eps;
    }
#pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        float s = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < 4; i++) s += c[i * 4 + j];
#pragma unroll
        for (unsigned i = 0; i < 4; i++) c[i * 4 + j] = c[i * 4 + j] / (s + hc_eps);
    }
    for (unsigned r = 1; r < sinkhorn_repeat; r++) {
#pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            float s = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) s += c[i * 4 + j];
#pragma unroll
            for (unsigned j = 0; j < 4; j++) c[i * 4 + j] = c[i * 4 + j] / (s + hc_eps);
        }
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            float s = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < 4; i++) s += c[i * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; i++) c[i * 4 + j] = c[i * 4 + j] / (s + hc_eps);
        }
    }
#pragma unroll
    for (unsigned k = 0; k < 16; k++) comb_out[k] = c[k];
}
#endif

/* op 121 — hyper-connections pre-block. See the file header for the shape of the whole
 * thing; this computes `(post_mix, comb_mix, layer_input)` from `mixes` (an ordinary
 * Gemv/Gemm's output — the projection itself is NOT this op's job) and `residual`.
 *
 * `logits[24]` doubles as scratch for the split/scale/sigmoid step AND (in its first `n`
 * slots) as the pre-mix gate the second reduction reads — both threads' worth of state
 * live in the SAME shared array on purpose, matching `d_rmsnorm`'s "produce once, consume
 * from registers/LDS, no second HBM round trip" discipline. */
__device__ void d_hyperconn_pre(float* __restrict__ post_mix, float* __restrict__ comb_mix,
                                bf16* __restrict__ layer_input, const float* __restrict__ mixes,
                                const bf16* __restrict__ residual, const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base, unsigned T, unsigned n,
                                unsigned hidden, unsigned sinkhorn_repeat, float rms_eps,
                                float hc_eps, unsigned slice, unsigned nblk, float* part,
                                bool head_only = false) {
    /* HEAD_ONLY — DeepSeek-V4's LEARNED GATED n -> 1 tower exit (`hc_head`,
     * model.py:709-717), which is this op's `layer_input` half and nothing else:
     *
     *     pre[c] = sigmoid(mixes[c] * inv * hc_scale[0] + hc_base[c]) + hc_eps
     *     y[d]   = SUM_c pre[c] * residual[c][d]
     *
     * -- term for term what reduction 2 below already computes. The differences are entirely
     * in the SHAPE of the inputs: the head's projection is `hc_head_fn[n, n*hidden]`, so
     * `mixes` is n wide rather than 2n+n^2, `hc_head_scale` is [1] and broadcasts, and there
     * is no Sinkhorn, no `post` and no `comb` to produce. Reading `mrow[n..n3)` would read off
     * the end of a head `mixes` row, so the flag skips the whole lane-0 block and leaves
     * post_mix/comb_mix untouched (the emitter passes TENSOR_NONE for both).
     *
     * NOT op 122 mode 2. Mode 2's arithmetic mean is the right function for V4's OTHER n -> 1
     * contraction -- the plain `h.mean(dim=2)` DSpark target taps at model.py:920 -- and the
     * WRONG one here; the tower exit is gated and learned. Both survive.  [DSV4-MHC] */
    const unsigned n3 = head_only ? n : (2 * n + n * n);
    const unsigned nh = n * hidden;
    __shared__ float logits[24]; /* n fixed at 4 — see the file header */
    __shared__ float comb_lds[16];

    for (unsigned t = slice; t < T; t += nblk) {
        const float* mrow = mixes + (size_t)t * n3;
        const bf16* rrow = residual + (size_t)t * nh;

        /* Reduction 1 (full workgroup): sum of squares of `residual`, over n*hidden — this
         * scales the LOGITS (`mixes`), not `residual` itself; see mhc_pre_torch. */
        float ss = 0.0f;
        if ((nh & 7u) == 0u) {
            /* 16 B loads: the 2 B walk was 64 dependent-looking scalar loads per thread. */
            for (unsigned i = threadIdx.x * 8u; i < nh; i += PLOW_THREADS * 8u) {
                const bf16v8 v = ld_glob8(rrow + i);
#pragma unroll
                for (unsigned e = 0; e < 8; e++) {
                    const float x = bf2f(v[e]);
                    ss += x * x;
                }
            }
        } else {
            for (unsigned i = threadIdx.x; i < nh; i += PLOW_THREADS) {
                const float x = bf2f(rrow[i]);
                ss += x * x;
            }
        }
        const float inv = rsqrtf(block_sum(ss, part) / (float)nh + rms_eps);

        /* Post-processing (lane 0 only): split n3 logits into pre/post/comb groups, scale,
         * sigmoid the gates, softmax + Sinkhorn-normalize the combine matrix. */
        if (threadIdx.x == 0) {
            /* pre_mix -> logits[0..n), left in place for reduction 2 below. */
            for (unsigned j = 0; j < n; j++) {
                const float v = (mrow[j] * inv) * hc_scale[0] + hc_base[j];
                logits[j] = 1.0f / (1.0f + expf(-v)) + hc_eps;
            }
        }
        if (threadIdx.x == 0 && !head_only) {
            /* post_mix -> output directly, scaled by the compile-time post-mult constant. */
            for (unsigned j = 0; j < n; j++) {
                const float v = (mrow[n + j] * inv) * hc_scale[1] + hc_base[n + j];
                post_mix[(size_t)t * n + j] = (1.0f / (1.0f + expf(-v))) * PLOW_HC_POST_MULT;
            }
            /* comb_mix: softmax over the LAST axis (dim=-1, fixed row i, varying column j),
             * then one dim=-2 (column) normalize, then (sinkhorn_repeat-1) more (dim=-1,
             * dim=-2) pairs — exactly mhc_pre_torch's loop, not a reordering of it. */
#if !PLOW_HC_SINKHORN_RCP
            if (n == 4u) {
                hc_sinkhorn4_reg(mrow, inv, hc_scale[2], hc_base + 2 * n, hc_eps, sinkhorn_repeat,
                                 comb_mix + (size_t)t * n * n);
            } else
#endif
            {
            for (unsigned i = 0; i < n; i++) {
                float m = -3.0e38f;
                for (unsigned j = 0; j < n; j++) {
                    const float v = (mrow[2 * n + i * n + j] * inv) * hc_scale[2] +
                                    hc_base[2 * n + i * n + j];
                    comb_lds[i * n + j] = v;
                    m = fmaxf(m, v);
                }
                float s = 0.0f;
                for (unsigned j = 0; j < n; j++) {
                    const float e = expf(comb_lds[i * n + j] - m);
                    comb_lds[i * n + j] = e;
                    s += e;
                }
#if PLOW_HC_SINKHORN_RCP
                hc_normalize<true>(comb_lds, i * n, 1, n, s, hc_eps);
#else
                for (unsigned j = 0; j < n; j++) comb_lds[i * n + j] = comb_lds[i * n + j] / s + hc_eps;
#endif
            }
            for (unsigned j = 0; j < n; j++) {
                float s = 0.0f;
                for (unsigned i = 0; i < n; i++) s += comb_lds[i * n + j];
#if PLOW_HC_SINKHORN_RCP
                hc_normalize<false>(comb_lds, j, n, n, s + hc_eps, hc_eps);
#else
                for (unsigned i = 0; i < n; i++) comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
#endif
            }
            for (unsigned r = 1; r < sinkhorn_repeat; r++) {
                for (unsigned i = 0; i < n; i++) {
                    float s = 0.0f;
                    for (unsigned j = 0; j < n; j++) s += comb_lds[i * n + j];
#if PLOW_HC_SINKHORN_RCP
                    hc_normalize<false>(comb_lds, i * n, 1, n, s + hc_eps, hc_eps);
#else
                    for (unsigned j = 0; j < n; j++)
                        comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
#endif
                }
                for (unsigned j = 0; j < n; j++) {
                    float s = 0.0f;
                    for (unsigned i = 0; i < n; i++) s += comb_lds[i * n + j];
#if PLOW_HC_SINKHORN_RCP
                    hc_normalize<false>(comb_lds, j, n, n, s + hc_eps, hc_eps);
#else
                    for (unsigned i = 0; i < n; i++)
                        comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
#endif
                }
            }
            for (unsigned k = 0; k < n * n; k++) comb_mix[(size_t)t * n * n + k] = comb_lds[k];
            }
        }
        __syncthreads(); /* logits[0..n) (pre_mix) must be visible before reduction 2 reads it */

        /* Reduction 2 (full workgroup, no cross-thread reduction needed — pre_mix is a
         * scalar per stream, broadcast from LDS): layer_input[d] = sum_i pre_mix[i] *
         * residual[i][d], parallel over d. UNNORMED — the caller chains a plain RmsNorm
         * with the block's real layernorm weight afterward, same as KDA's `prenormed`. */
        if (n == 4u && (hidden & 7u) == 0u) {
            const float l0 = logits[0], l1 = logits[1], l2 = logits[2], l3 = logits[3];
            for (unsigned d = threadIdx.x * 8u; d < hidden; d += PLOW_THREADS * 8u) {
                const bf16v8 r0 = ld_glob8(rrow + d);
                const bf16v8 r1 = ld_glob8(rrow + (size_t)hidden + d);
                const bf16v8 r2 = ld_glob8(rrow + 2 * (size_t)hidden + d);
                const bf16v8 r3 = ld_glob8(rrow + 3 * (size_t)hidden + d);
                bf16v8 o;
#pragma unroll
                for (unsigned e = 0; e < 8; e++) {
                    float acc = 0.0f;
                    acc += l0 * bf2f(r0[e]);
                    acc += l1 * bf2f(r1[e]);
                    acc += l2 * bf2f(r2[e]);
                    acc += l3 * bf2f(r3[e]);
                    o[e] = f2bf(acc);
                }
                st_glob8(layer_input + (size_t)t * hidden + d, o);
            }
        } else {
            for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
                float acc = 0.0f;
                for (unsigned i = 0; i < n; i++)
                    acc += logits[i] * bf2f(rrow[(size_t)i * hidden + d]);
                layer_input[(size_t)t * hidden + d] = f2bf(acc);
            }
        }
        __syncthreads(); /* `part`/`logits`/`comb_lds` reused by the next token this workgroup handles */
    }
}

#if !PLOW_HC_SINKHORN_RCP
/* Prefill form of d_hyperconn_pre (n = 4, hidden % 8 == 0, not head_only): a workgroup's tokens
 * are taken HC_PRE_G at a time; each token's lane-0 tail (gates + softmax + Sinkhorn) runs
 * on 16 lanes, one combine element per lane, and the group's tokens run side by side. Every
 * value is computed with d_hyperconn_pre's operations in its order (block_sum unrolled into
 * `wsum`, row/column sums formed per lane in hc_sinkhorn4_reg's order), so the outputs are
 * bit-identical. 512 rows: 52 -> 13 us standalone. */
#define HC_PRE_G 16u
__device__ void d_hyperconn_pre_rows(float* __restrict__ post_mix, float* __restrict__ comb_mix,
                                     bf16* __restrict__ layer_input, const float* __restrict__ mixes,
                                     const bf16* __restrict__ residual,
                                     const float* __restrict__ hc_scale,
                                     const float* __restrict__ hc_base, unsigned T, unsigned hidden,
                                     unsigned sinkhorn_repeat, float rms_eps, float hc_eps,
                                     unsigned slice, unsigned nblk) {
    const unsigned nh = 4u * hidden;
    const unsigned wave = threadIdx.x / PLOW_WAVE, lane = threadIdx.x % PLOW_WAVE;
    __shared__ float wsum[HC_PRE_G][PLOW_WAVES];
    __shared__ float gate[HC_PRE_G][4];
    for (unsigned t0 = slice; t0 < T; t0 += nblk * HC_PRE_G) {
        const unsigned cnt = min(HC_PRE_G, (T - t0 + nblk - 1) / nblk);
        for (unsigned g = 0; g < cnt; g++) {
            const bf16* rrow = residual + (size_t)(t0 + g * nblk) * nh;
            float ss = 0.0f;
            for (unsigned i = threadIdx.x * 8u; i < nh; i += PLOW_THREADS * 8u) {
                const bf16v8 v = ld_glob8(rrow + i);
#pragma unroll
                for (unsigned e = 0; e < 8; e++) {
                    const float x = bf2f(v[e]);
                    ss += x * x;
                }
            }
            ss = wave_sum(ss);
            if (lane == 0) wsum[g][wave] = ss;
        }
        __syncthreads();
        if (threadIdx.x < cnt * 16u) {
            /* 16 lanes per token, lane e = 4i + j holds comb[i][j]. Every lane forms each row or
             * column sum itself, in hc_sinkhorn4_reg's order, so the values are its values. */
            const unsigned g = threadIdx.x >> 4, e = threadIdx.x & 15u, t = t0 + g * nblk;
            const unsigned rb = (lane & ~15u) | (e & ~3u), cb = (lane & ~15u) | (e & 3u);
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < PLOW_WAVES; i++) s += wsum[g][i];
            const float inv = rsqrtf(s / (float)nh + rms_eps);
            const float* mrow = mixes + (size_t)t * 24u;
            if (e < 4u) {
                const float v = (mrow[e] * inv) * hc_scale[0] + hc_base[e];
                gate[g][e] = 1.0f / (1.0f + expf(-v)) + hc_eps;
                const float w = (mrow[4 + e] * inv) * hc_scale[1] + hc_base[4 + e];
                post_mix[(size_t)t * 4u + e] = (1.0f / (1.0f + expf(-w))) * PLOW_HC_POST_MULT;
            }
            float c = (mrow[8 + e] * inv) * hc_scale[2] + hc_base[8 + e];
            float m = -3.0e38f;
#pragma unroll
            for (unsigned q = 0; q < 4; q++) m = fmaxf(m, __shfl(c, rb + q, PLOW_WAVE));
            c = expf(c - m);
            s = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < 4; q++) s += __shfl(c, rb + q, PLOW_WAVE);
            c = c / s + hc_eps;
            s = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < 4; q++) s += __shfl(c, cb + 4u * q, PLOW_WAVE);
            c = c / (s + hc_eps);
            for (unsigned r = 1; r < sinkhorn_repeat; r++) {
                s = 0.0f;
#pragma unroll
                for (unsigned q = 0; q < 4; q++) s += __shfl(c, rb + q, PLOW_WAVE);
                c = c / (s + hc_eps);
                s = 0.0f;
#pragma unroll
                for (unsigned q = 0; q < 4; q++) s += __shfl(c, cb + 4u * q, PLOW_WAVE);
                c = c / (s + hc_eps);
            }
            comb_mix[(size_t)t * 16u + e] = c;
        }
        __syncthreads();
        for (unsigned g = 0; g < cnt; g++) {
            const size_t t = t0 + g * nblk;
            const bf16* rrow = residual + t * nh;
            const float l0 = gate[g][0], l1 = gate[g][1], l2 = gate[g][2], l3 = gate[g][3];
            for (unsigned d = threadIdx.x * 8u; d < hidden; d += PLOW_THREADS * 8u) {
                const bf16v8 r0 = ld_glob8(rrow + d);
                const bf16v8 r1 = ld_glob8(rrow + (size_t)hidden + d);
                const bf16v8 r2 = ld_glob8(rrow + 2 * (size_t)hidden + d);
                const bf16v8 r3 = ld_glob8(rrow + 3 * (size_t)hidden + d);
                bf16v8 o;
#pragma unroll
                for (unsigned e = 0; e < 8; e++) {
                    float acc = 0.0f;
                    acc += l0 * bf2f(r0[e]);
                    acc += l1 * bf2f(r1[e]);
                    acc += l2 * bf2f(r2[e]);
                    acc += l3 * bf2f(r3[e]);
                    o[e] = f2bf(acc);
                }
                st_glob8(layer_input + t * hidden + d, o);
            }
        }
        __syncthreads(); /* wsum/gate are rewritten by the next group */
    }
}
#endif

/* One 8-channel step of op 122's n = 4 vector path: output stream j at channels d..d+7. Shared
 * with d_gemv_f32_wg_hcpost so the fold computes the residual with this exact code. */
__device__ __forceinline__ bf16v8 hc_post4_8(const float* c, const float* p, unsigned j,
                                            const bf16v8* r, const bf16v8 xv) {
    bf16v8 o;
#pragma unroll
    for (unsigned e = 0; e < 8; e++) {
        float acc = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < 4; i++) acc += c[i * 4 + j] * bf2f(r[i][e]);
        acc += p[j] * bf2f(xv[e]);
        o[e] = f2bf(acc);
    }
    return o;
}

#ifndef PLOW_HC_POST_ROWS
#define PLOW_HC_POST_ROWS 1
#endif
/* op 122 — hyper-connections post-block, paired with op 121.
 * `new_residual[j][d] = sum_i comb_mix[i][j] * residual[i][d]  +  post_mix[j] * x_out[d]`
 * — an n x n combine (n=4) plus a broadcast multiply-add. Fully parallel over `d` with no
 * reduction and no LDS/sync needed: every input each thread reads is either per-token
 * scalar (post_mix, comb_mix, tiny — reread from HBM by every thread, cheap at n=4) or at
 * its own `d`. */
__device__ void d_hyperconn_post(bf16* __restrict__ new_residual, const bf16* __restrict__ x_out,
                                 const bf16* __restrict__ residual, const float* __restrict__ post_mix,
                                 const float* __restrict__ comb_mix, unsigned T, unsigned n,
                                 unsigned hidden, unsigned mode, unsigned slice, unsigned nblk) {
#if PLOW_HC_POST_ROWS > 1
    /* PLOW_HC_POST_ROWS=R (prefill band rows): a workgroup's rows t, t+nblk, ... are one HBM
     * round trip each in the loop below; issue R rows' loads before storing any. Mode 0, n = 4,
     * one 8-channel step per thread per row (hidden = 8 * PLOW_THREADS). Same per-element
     * arithmetic (hc_post4_8), so bit-identical. */
    if (mode == 0u && n == 4u && hidden == 8u * PLOW_THREADS) {
        constexpr unsigned R = PLOW_HC_POST_ROWS;
        const unsigned d = threadIdx.x * 8u;
        unsigned t = slice;
        for (; t + (R - 1) * nblk < T; t += R * nblk) {
            float c[R][16], p[R][4];
            bf16v8 xv[R], r[R][4];
#pragma unroll
            for (unsigned q = 0; q < R; q++) {
                const size_t tq = t + q * nblk;
#pragma unroll
                for (unsigned k = 0; k < 16; k++) c[q][k] = comb_mix[tq * 16u + k];
#pragma unroll
                for (unsigned k = 0; k < 4; k++) p[q][k] = post_mix[tq * 4u + k];
                xv[q] = ld_glob8(x_out + tq * hidden + d);
#pragma unroll
                for (unsigned i = 0; i < 4; i++)
                    r[q][i] = ld_glob8(residual + tq * 4u * hidden + (size_t)i * hidden + d);
            }
#pragma unroll
            for (unsigned q = 0; q < R; q++) {
                bf16* orow = new_residual + (size_t)(t + q * nblk) * 4u * hidden;
#pragma unroll
                for (unsigned j = 0; j < 4; j++)
                    st_glob8(orow + (size_t)j * hidden + d, hc_post4_8(c[q], p[q], j, r[q], xv[q]));
            }
        }
        slice = t;
    }
#endif
    for (unsigned t = slice; t < T; t += nblk) {
        if (mode == 1u) {
            const bf16* xrow = x_out + (size_t)t * hidden;
            bf16* orow = new_residual + (size_t)t * n * hidden;
            for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS)
                for (unsigned j = 0; j < n; j++) orow[(size_t)j * hidden + d] = xrow[d];
            continue;
        }
        if (mode == 2u) {
            const bf16* rrow = residual + (size_t)t * n * hidden;
            bf16* orow = new_residual + (size_t)t * hidden;
            for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
                float acc = 0.0f;
                for (unsigned i = 0; i < n; i++) acc += bf2f(rrow[(size_t)i * hidden + d]);
                orow[d] = f2bf(acc / (float)n);
            }
            continue;
        }
        const bf16* rrow = residual + (size_t)t * n * hidden;
        const bf16* xrow = x_out + (size_t)t * hidden;
        const float* pm = post_mix + (size_t)t * n;
        const float* cm = comb_mix + (size_t)t * n * n;
        bf16* orow = new_residual + (size_t)t * n * hidden;
        if (n == 4u && (hidden & 7u) == 0u) {
            /* n = 4 (every mHC checkpoint): the 20 mix scalars live in registers for the row and
             * each thread moves 8 channels per step with 16 B loads/stores. The scalar walk
             * below re-read the 20 scalars per element and issued 2 B loads: 1.76 ms per packet
             * at T=8192 (92 per prefill), ~40x its byte floor. Same arithmetic, same order. */
            float c[16], p[4];
#pragma unroll
            for (unsigned k = 0; k < 16; k++) c[k] = cm[k];
#pragma unroll
            for (unsigned k = 0; k < 4; k++) p[k] = pm[k];
            for (unsigned d = threadIdx.x * 8u; d < hidden; d += PLOW_THREADS * 8u) {
                const bf16v8 xv = ld_glob8(xrow + d);
                bf16v8 r[4];
#pragma unroll
                for (unsigned i = 0; i < 4; i++) r[i] = ld_glob8(rrow + (size_t)i * hidden + d);
#pragma unroll
                for (unsigned j = 0; j < 4; j++)
                    st_glob8(orow + (size_t)j * hidden + d, hc_post4_8(c, p, j, r, xv));
            }
            continue;
        }
        for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
            const float xv = bf2f(xrow[d]);
            for (unsigned j = 0; j < n; j++) {
                float acc = 0.0f;
                for (unsigned i = 0; i < n; i++)
                    acc += cm[i * n + j] * bf2f(rrow[(size_t)i * hidden + d]);
                acc += pm[j] * xv;
                orow[(size_t)j * hidden + d] = f2bf(acc);
            }
        }
    }
}

/* PLOW_GLM_HC_POST_FOLD: op 135 (GemvF32, the mHC mix) with op 122 (mode 0, n = 4) folded in.
 * x = new_residual is not read but formed per step from the HyperConnPost operands with
 * hc_post4_8, and the workgroups of column 0 store it for the later readers. The dot product
 * is d_gemv_f32_wg's (workgroup per output, per-step terms added in step order, contraction off),
 * so mixes and new_residual are bit-identical to the HyperConnPost + GemvF32 pair.
 * Requires K = 4 * hidden, K % (8 * PLOW_WAVE) == 0. */
__device__ void d_gemv_f32_wg_hcpost(float* __restrict__ C, bf16* __restrict__ new_residual,
                                     const float* __restrict__ W, const bf16* __restrict__ x_out,
                                     const bf16* __restrict__ residual,
                                     const float* __restrict__ post_mix,
                                     const float* __restrict__ comb_mix, unsigned M, unsigned N,
                                     unsigned K, unsigned slice, unsigned nblk,
                                     float* __restrict__ part) {
    const unsigned wave = threadIdx.x / PLOW_WAVE;
    const unsigned lane = threadIdx.x % PLOW_WAVE;
    const unsigned steps = K / (PLOW_WAVE * 8u), hidden = K / 4u;
    for (size_t output = slice; output < (size_t)M * N; output += nblk) {
        const unsigned m = output / N, n = output % N;
        const float* const wn = W + (size_t)n * K;
        const bf16* const rrow = residual + (size_t)m * K;
        const bf16* const xrow = x_out + (size_t)m * hidden;
        float c[16], p[4];
#pragma unroll
        for (unsigned k = 0; k < 16; k++) c[k] = comb_mix[(size_t)m * 16u + k];
#pragma unroll
        for (unsigned k = 0; k < 4; k++) p[k] = post_mix[(size_t)m * 4u + k];
#pragma unroll 2
        for (unsigned s = wave; s < steps; s += PLOW_WAVES) {
            const unsigned k = s * (PLOW_WAVE * 8u) + lane * 8u;
            const unsigned j = k / hidden, d = k % hidden;
            bf16v8 r[4];
#pragma unroll
            for (unsigned i = 0; i < 4; i++) r[i] = ld_glob8(rrow + (size_t)i * hidden + d);
            const bf16v8 xo = ld_glob8(xrow + d);
            const float4 w0 = *(const float4*)(wn + k);
            const float4 w1 = *(const float4*)(wn + k + 4);
            const bf16v8 xv = hc_post4_8(c, p, j, r, xo);
            if (n == 0u) st_glob8(new_residual + (size_t)m * K + k, xv);
            {
#pragma clang fp contract(off)
                part[s * PLOW_WAVE + lane] =
                    bf2f(xv[0]) * w0.x + bf2f(xv[1]) * w0.y + bf2f(xv[2]) * w0.z +
                    bf2f(xv[3]) * w0.w + bf2f(xv[4]) * w1.x + bf2f(xv[5]) * w1.y +
                    bf2f(xv[6]) * w1.z + bf2f(xv[7]) * w1.w;
            }
        }
        __syncthreads();
        if (wave == 0) {
#pragma clang fp contract(off)
            float acc = 0.0f;
            for (unsigned s = 0; s < steps; s++) acc += part[s * PLOW_WAVE + lane];
            acc = wave_sum(acc);
            if (lane == 0) C[output] = acc;
        }
        __syncthreads();
    }
}

/* PLOW_GLM_HC_MIX_SPLIT: d_gemv_f32_wg_hcpost with each output's K steps split over S
 * workgroups (slice = output * S + part). Every (step, lane) term is the one the unsplit body
 * writes to LDS, written instead to `gpart[output][step][lane]` in global memory; the packet's
 * last workgroup adds them in step order (hc_mix_split_reduce), so mixes and new_residual stay
 * bit-identical. Requires (K / (8 * PLOW_WAVE)) % S == 0, K = 4 * hidden. */
__device__ void d_gemv_f32_hcpost_split_part(float* __restrict__ gpart,
                                             bf16* __restrict__ new_residual,
                                             const float* __restrict__ W,
                                             const bf16* __restrict__ x_out,
                                             const bf16* __restrict__ residual,
                                             const float* __restrict__ post_mix,
                                             const float* __restrict__ comb_mix, unsigned N,
                                             unsigned K, unsigned S, unsigned slice) {
    const unsigned wave = threadIdx.x / PLOW_WAVE;
    const unsigned lane = threadIdx.x % PLOW_WAVE;
    const unsigned steps = K / (PLOW_WAVE * 8u), hidden = K / 4u, per = steps / S;
    const unsigned output = slice / S, part = slice % S;
    const unsigned m = output / N, n = output % N;
    const float* const wn = W + (size_t)n * K;
    const bf16* const rrow = residual + (size_t)m * K;
    const bf16* const xrow = x_out + (size_t)m * hidden;
    float* const gp = gpart + (size_t)output * steps * PLOW_WAVE;
    float c[16], p[4];
#pragma unroll
    for (unsigned k = 0; k < 16; k++) c[k] = comb_mix[(size_t)m * 16u + k];
#pragma unroll
    for (unsigned k = 0; k < 4; k++) p[k] = post_mix[(size_t)m * 4u + k];
    for (unsigned s = part * per + wave; s < (part + 1u) * per; s += PLOW_WAVES) {
        const unsigned k = s * (PLOW_WAVE * 8u) + lane * 8u;
        const unsigned j = k / hidden, d = k % hidden;
        bf16v8 r[4];
#pragma unroll
        for (unsigned i = 0; i < 4; i++) r[i] = ld_glob8(rrow + (size_t)i * hidden + d);
        const bf16v8 xo = ld_glob8(xrow + d);
        const float4 w0 = *(const float4*)(wn + k);
        const float4 w1 = *(const float4*)(wn + k + 4);
        const bf16v8 xv = hc_post4_8(c, p, j, r, xo);
        if (n == 0u) st_glob8(new_residual + (size_t)m * K + k, xv);
        {
#pragma clang fp contract(off)
            gp[s * PLOW_WAVE + lane] =
                bf2f(xv[0]) * w0.x + bf2f(xv[1]) * w0.y + bf2f(xv[2]) * w0.z +
                bf2f(xv[3]) * w0.w + bf2f(xv[4]) * w1.x + bf2f(xv[5]) * w1.y +
                bf2f(xv[6]) * w1.z + bf2f(xv[7]) * w1.w;
        }
    }
}

/* The unsplit body's wave-0 reduction, one wave per output: step-order adds, then wave_sum. */
__device__ void hc_mix_split_reduce(float* __restrict__ C, const float* __restrict__ gpart,
                                    unsigned outputs, unsigned K) {
    const unsigned wave = threadIdx.x / PLOW_WAVE;
    const unsigned lane = threadIdx.x % PLOW_WAVE;
    const unsigned steps = K / (PLOW_WAVE * 8u);
    for (unsigned o = wave; o < outputs; o += PLOW_WAVES) {
#pragma clang fp contract(off)
        const float* const gp = gpart + (size_t)o * steps * PLOW_WAVE;
        float acc = 0.0f;
        for (unsigned s = 0; s < steps; s++) acc += gp[s * PLOW_WAVE + lane];
        acc = wave_sum(acc);
        if (lane == 0) C[o] = acc;
    }
}

#endif /* PLOW_OP_HYPERCONN_H */
