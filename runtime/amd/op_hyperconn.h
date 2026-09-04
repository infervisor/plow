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
                                float hc_eps, unsigned slice, unsigned nblk, float* part) {
    const unsigned n3 = 2 * n + n * n;
    const unsigned nh = n * hidden;
    __shared__ float logits[24]; /* n fixed at 4 — see the file header */
    __shared__ float comb_lds[16];

    for (unsigned t = slice; t < T; t += nblk) {
        const float* mrow = mixes + (size_t)t * n3;
        const bf16* rrow = residual + (size_t)t * nh;

        /* Reduction 1 (full workgroup): sum of squares of `residual`, over n*hidden — this
         * scales the LOGITS (`mixes`), not `residual` itself; see mhc_pre_torch. */
        float ss = 0.0f;
        for (unsigned i = threadIdx.x; i < nh; i += PLOW_THREADS) {
            const float x = bf2f(rrow[i]);
            ss += x * x;
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
            /* post_mix -> output directly, scaled by the compile-time post-mult constant. */
            for (unsigned j = 0; j < n; j++) {
                const float v = (mrow[n + j] * inv) * hc_scale[1] + hc_base[n + j];
                post_mix[(size_t)t * n + j] = (1.0f / (1.0f + expf(-v))) * PLOW_HC_POST_MULT;
            }
            /* comb_mix: softmax over the LAST axis (dim=-1, fixed row i, varying column j),
             * then one dim=-2 (column) normalize, then (sinkhorn_repeat-1) more (dim=-1,
             * dim=-2) pairs — exactly mhc_pre_torch's loop, not a reordering of it. */
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
                for (unsigned j = 0; j < n; j++) comb_lds[i * n + j] = comb_lds[i * n + j] / s + hc_eps;
            }
            for (unsigned j = 0; j < n; j++) {
                float s = 0.0f;
                for (unsigned i = 0; i < n; i++) s += comb_lds[i * n + j];
                for (unsigned i = 0; i < n; i++) comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
            }
            for (unsigned r = 1; r < sinkhorn_repeat; r++) {
                for (unsigned i = 0; i < n; i++) {
                    float s = 0.0f;
                    for (unsigned j = 0; j < n; j++) s += comb_lds[i * n + j];
                    for (unsigned j = 0; j < n; j++)
                        comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
                }
                for (unsigned j = 0; j < n; j++) {
                    float s = 0.0f;
                    for (unsigned i = 0; i < n; i++) s += comb_lds[i * n + j];
                    for (unsigned i = 0; i < n; i++)
                        comb_lds[i * n + j] = comb_lds[i * n + j] / (s + hc_eps);
                }
            }
            for (unsigned k = 0; k < n * n; k++) comb_mix[(size_t)t * n * n + k] = comb_lds[k];
        }
        __syncthreads(); /* logits[0..n) (pre_mix) must be visible before reduction 2 reads it */

        /* Reduction 2 (full workgroup, no cross-thread reduction needed — pre_mix is a
         * scalar per stream, broadcast from LDS): layer_input[d] = sum_i pre_mix[i] *
         * residual[i][d], parallel over d. UNNORMED — the caller chains a plain RmsNorm
         * with the block's real layernorm weight afterward, same as KDA's `prenormed`. */
        for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned i = 0; i < n; i++) acc += logits[i] * bf2f(rrow[(size_t)i * hidden + d]);
            layer_input[(size_t)t * hidden + d] = f2bf(acc);
        }
        __syncthreads(); /* `part`/`logits`/`comb_lds` reused by the next token this workgroup handles */
    }
}

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

#endif /* PLOW_OP_HYPERCONN_H */
