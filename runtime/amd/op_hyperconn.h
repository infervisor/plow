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

/* `d_hyperconn_pre`'s wave-per-token arm. On by default: it is what takes the op off its serial
 * section (see the arm's own comment). 0 restores the shipped workgroup-per-token block. */
#ifndef PLOW_HC_WAVE_TOKEN
#define PLOW_HC_WAVE_TOKEN 1
#endif

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

/* `pre_mode` for `d_hyperconn_pre` — see the note inside it. */
#define PLOW_HC_PRE_OWN 0u
#define PLOW_HC_PRE_SEED 1u
#define PLOW_HC_PRE_DEFER 2u

/* op 121 — hyper-connections pre-block. See the file header for the shape of the whole
 * thing; this computes `(post_mix, comb_mix, layer_input)` from `mixes` (an ordinary
 * Gemv/Gemm's output — the projection itself is NOT this op's job) and `residual`.
 *
 * `logits[24]` doubles as scratch for the split/scale/sigmoid step AND (in its first `n`
 * slots) as the pre-mix gate the second reduction reads — both threads' worth of state
 * live in the SAME shared array on purpose, matching `d_rmsnorm`'s "produce once, consume
 * from registers/LDS, no second HBM round trip" discipline. */
/* THE 4x4 SINKHORN, REGISTER-RESIDENT. Lifted verbatim out of `d_hyperconn_pre`'s n==4 branch so
 * that branch and the wave-per-token arm share one copy rather than two: it is ~1300 dependent
 * scalar ops and the single most error-prone block in the file. `c` is a local array indexed only
 * by constants, which is what keeps it in registers instead of scratch.
 *
 * softmax over the LAST axis (dim=-1, fixed row i, varying column j), then one dim=-2 (column)
 * normalize, then (repeat-1) more (dim=-1, dim=-2) pairs -- exactly mhc_pre_torch's loop, not a
 * reordering of it. */
__device__ __forceinline__ void hc_sinkhorn4(float* c, const float* __restrict__ mrow, float inv,
                                             const float* __restrict__ hc_scale,
                                             const float* __restrict__ hc_base, float hc_eps,
                                             unsigned repeat) {
    constexpr unsigned NC = 4u;
    for (unsigned i = 0; i < NC; i++) {
        float m = -3.0e38f;
        for (unsigned j = 0; j < NC; j++) {
            const float v =
                (mrow[2 * NC + i * NC + j] * inv) * hc_scale[2] + hc_base[2 * NC + i * NC + j];
            c[i * NC + j] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (unsigned j = 0; j < NC; j++) {
            const float e = expf(c[i * NC + j] - m);
            c[i * NC + j] = e;
            s += e;
        }
#if PLOW_HC_SINKHORN_RCP
        hc_normalize<true>(c, i * NC, 1, NC, s, hc_eps);
#else
        for (unsigned j = 0; j < NC; j++) c[i * NC + j] = c[i * NC + j] / s + hc_eps;
#endif
    }
    for (unsigned j = 0; j < NC; j++) {
        float s = 0.0f;
        for (unsigned i = 0; i < NC; i++) s += c[i * NC + j];
#if PLOW_HC_SINKHORN_RCP
        hc_normalize<false>(c, j, NC, NC, s + hc_eps, hc_eps);
#else
        for (unsigned i = 0; i < NC; i++) c[i * NC + j] = c[i * NC + j] / (s + hc_eps);
#endif
    }
    for (unsigned r = 1; r < repeat; r++) {
        for (unsigned i = 0; i < NC; i++) {
            float s = 0.0f;
            for (unsigned j = 0; j < NC; j++) s += c[i * NC + j];
#if PLOW_HC_SINKHORN_RCP
            hc_normalize<false>(c, i * NC, 1, NC, s + hc_eps, hc_eps);
#else
            for (unsigned j = 0; j < NC; j++) c[i * NC + j] = c[i * NC + j] / (s + hc_eps);
#endif
        }
        for (unsigned j = 0; j < NC; j++) {
            float s = 0.0f;
            for (unsigned i = 0; i < NC; i++) s += c[i * NC + j];
#if PLOW_HC_SINKHORN_RCP
            hc_normalize<false>(c, j, NC, NC, s + hc_eps, hc_eps);
#else
            for (unsigned i = 0; i < NC; i++) c[i * NC + j] = c[i * NC + j] / (s + hc_eps);
#endif
        }
    }
}

__device__ void d_hyperconn_pre(float* __restrict__ post_mix, float* __restrict__ comb_mix,
                                bf16* __restrict__ layer_input, const float* __restrict__ mixes,
                                const bf16* __restrict__ residual, const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base, unsigned T, unsigned n,
                                unsigned hidden, unsigned sinkhorn_repeat, float rms_eps,
                                float hc_eps, unsigned slice, unsigned nblk, float* part,
                                bool head_only = false, float* __restrict__ pre_pair = nullptr,
                                unsigned pre_in_half = 0, unsigned pre_mode = PLOW_HC_PRE_OWN) {
    /* PRE_MODE -- WHICH SUBLAYER'S `pre` GATES THE COLLAPSE.
     *
     * GLM-5.3 and DeepSeek-V4.1 disagree here, and the disagreement is invisible in the shapes:
     * both derive `pre` from `mixes[0..n)` and both collapse `residual` with an n-vector, so the
     * wrong one runs silently and produces a fluent, wrong model.
     *
     *   PLOW_HC_PRE_OWN (GLM-5.3)  reduction 2 gates with the `pre` THIS call just computed.
     *   PLOW_HC_PRE_SEED (V4.1, first sublayer of the model)  gates with a one-hot on copy 0 --
     *       `make_identity_pre_mix`, model.py:1159-1163 -- and publishes its own `pre` for the
     *       next sublayer.
     *   PLOW_HC_PRE_DEFER (V4.1, everywhere else)  gates with the PREVIOUS sublayer's `pre`, read
     *       from `pre_pair` half `pre_in_half`, and publishes its own into the other half.
     *       V4.1's `Block.forward` (model.py:965-996): attention collapses with the previous
     *       block's `ffn_pre`, the FFN collapses with this block's `attn_pre`. The class docstring
     *       states it outright -- "the coefficients a sublayer computes are used by the *next*
     *       one". Priced at 51.9% relative on layer 0 alone by `scripts/dsv41_mhc_oracle.py`.
     *
     * `pre_pair` is ONE tensor holding both halves, `[2][T][n]` f32, because `HyperConnPre`
     * already spends 7 of the descriptor's 8 tensor slots. A sublayer reads half `pre_in_half`
     * and writes half `pre_in_half ^ 1`, so consecutive sublayers alternate and never alias:
     * within a call the read and the write are to different halves at the same `t`.
     *
     * `post` and `comb` are same-sublayer in BOTH models (the oracle measures 0.000e+00), so
     * nothing below this comment changes for them. */
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

    /* ONE WAVE PER TOKEN, and it is the SERIAL SECTION this buys, not the loads.
     *
     * The workgroup-per-token form below is at 6x its memory roofline while `d_hyperconn_post`,
     * moving the same ~754 MB in the same access pattern, runs at 1.5x -- so the gap is not
     * bandwidth. It is that lane 0 runs ~1300 dependent scalar ops of Sinkhorn with 511 lanes
     * idle, once per token, 27 times per block at T=8192 over 304 blocks, and LDS caps the
     * interpreter at ONE workgroup per CU so there is nothing resident to hide it behind.
     *
     * A wave owning a token runs eight of those concurrently and needs no barrier at all: the
     * sum of squares becomes a `wave_sum` over 320 elements per lane, the gate broadcasts by
     * `__shfl` instead of through LDS, and the collapse gives each lane 80 of `hidden`.
     *
     * NOT bit-identical: `inv` is now a 64-lane reduction of 320-element partials where it was a
     * 512-lane reduction of 40-element ones, so the sum of 20480 squares is differently
     * associated. Everything downstream of `inv` is unchanged, statement for statement.
     *
     * Gated as the register-resident 4x4 it calls -- `n == 4 && T >= nblk`, so decode and any
     * other `n` keep the shipped block -- and ALSO behind `PLOW_HC_WAVE_TOKEN`, because GLM-5.3's
     * mHC is n=4 too and would otherwise take a non-bit-identical arm without anyone choosing it.
     * Build with -DPLOW_HC_WAVE_TOKEN=0 to restore the shipped block for an A/B. */
    if (PLOW_HC_WAVE_TOKEN && n == 4u && T >= nblk) {
        const unsigned wv = threadIdx.x / PLOW_WAVE;
        const unsigned ln = threadIdx.x % PLOW_WAVE;
        for (unsigned t = slice * PLOW_WAVES + wv; t < T; t += nblk * PLOW_WAVES) {
            const float* const mrow = mixes + (size_t)t * n3;
            const bf16* const rrow = residual + (size_t)t * nh;

            float ss = 0.0f;
            const unsigned nfull = nh & ~(PLOW_WAVE * 8u - 1u);
            for (unsigned i = ln * 8u; i < nfull; i += PLOW_WAVE * 8u) {
                const bf16v8 v = ld_glob8(rrow + i);
#pragma unroll
                for (unsigned u = 0; u < 8; u++) {
                    const float x = bf2f(v[u]);
                    ss = __builtin_fmaf(x, x, ss);
                }
            }
            for (unsigned i = nfull + ln; i < nh; i += PLOW_WAVE) {
                const float x = bf2f(rrow[i]);
                ss = __builtin_fmaf(x, x, ss);
            }
            const float inv = rsqrtf(wave_sum(ss) / (float)nh + rms_eps);

            float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}; /* every lane reads them via __shfl */
            if (ln == 0) {
                for (unsigned j = 0; j < 4u; j++) {
                    const float v = (mrow[j] * inv) * hc_scale[0] + hc_base[j];
                    g[j] = 1.0f / (1.0f + expf(-v)) + hc_eps;
                }
                if (pre_mode != PLOW_HC_PRE_OWN) {
                    float* const po =
                        pre_pair + (size_t)(pre_in_half ^ 1u) * T * 4u + (size_t)t * 4u;
                    for (unsigned j = 0; j < 4u; j++) po[j] = g[j];
                }
                if (!head_only) {
                    for (unsigned j = 0; j < 4u; j++) {
                        const float v = (mrow[4u + j] * inv) * hc_scale[1] + hc_base[4u + j];
                        post_mix[(size_t)t * 4u + j] =
                            (1.0f / (1.0f + expf(-v))) * PLOW_HC_POST_MULT;
                    }
                    float c[16];
                    hc_sinkhorn4(c, mrow, inv, hc_scale, hc_base, hc_eps, sinkhorn_repeat);
                    for (unsigned k = 0; k < 16u; k++) comb_mix[(size_t)t * 16u + k] = c[k];
                }
            }
            /* The collapse gate: lane 0's `g` for PRE_OWN, a one-hot for PRE_SEED, the PREVIOUS
             * sublayer's for PRE_DEFER -- the same three cases as below, with `__shfl` where the
             * workgroup form used LDS and a barrier. */
            for (unsigned j = 0; j < 4u; j++) {
                g[j] = pre_mode == PLOW_HC_PRE_OWN ? __shfl(g[j], 0, PLOW_WAVE)
                       : pre_mode == PLOW_HC_PRE_SEED
                           ? (j == 0 ? 1.0f : 0.0f)
                           : pre_pair[(size_t)pre_in_half * T * 4u + (size_t)t * 4u + j];
            }
            for (unsigned d = ln; d < hidden; d += PLOW_WAVE) {
                float acc = 0.0f;
                for (unsigned i = 0; i < 4u; i++)
                    acc = __builtin_fmaf(g[i], bf2f(rrow[(size_t)i * hidden + d]), acc);
                layer_input[(size_t)t * hidden + d] = f2bf(acc);
            }
        }
        return;
    }

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
            if (pre_mode != PLOW_HC_PRE_OWN) {
                float* const po = pre_pair + (size_t)(pre_in_half ^ 1u) * T * n + (size_t)t * n;
                for (unsigned j = 0; j < n; j++) po[j] = logits[j];
            }
        }
        if (threadIdx.x == 0 && !head_only) {
            /* post_mix -> output directly, scaled by the compile-time post-mult constant. */
            for (unsigned j = 0; j < n; j++) {
                const float v = (mrow[n + j] * inv) * hc_scale[1] + hc_base[n + j];
                post_mix[(size_t)t * n + j] = (1.0f / (1.0f + expf(-v))) * PLOW_HC_POST_MULT;
            }
            /* REGISTER-RESIDENT 4x4. The Sinkhorn below is ~1300 dependent scalar ops on lane 0,
             * and with `comb_lds` in LDS every one of them is an LDS round trip -- at 2 waves/SIMD
             * there is nothing to hide that latency behind. A local array indexed only by constants
             * (which is what `NC = 4` makes these loops) is promoted to registers instead.
             *
             * WHY THE FILE HEADER'S "no measurable throughput" NO LONGER HOLDS: it was written for
             * GLM-5.3 DECODE, where this op sees one token. V4.1 PREFILL dispatches it at T=8192, so
             * a block runs the whole serial section 27 times. Measured by ablation on 8x MI300X at
             * 8k -- hc_sinkhorn_iters 20 -> 2 moves the layer 30.34 -> 25.49 ms -- the Sinkhorn is
             * ~5.4 ms of a 30 ms layer. The `T >= nblk` gate keeps the decode path on the shipped
             * block verbatim, which is the arm the gfx950 oracle signed off.
             *
             * BIT-IDENTICAL: the copy below is generated from the shipped block by substitution, so
             * it is the same statements in the same order at the same precision. Only the storage
             * class of the 4x4 changes. */
            if (n == 4u && T >= nblk) {
                float c[16];
                hc_sinkhorn4(c, mrow, inv, hc_scale, hc_base, hc_eps, sinkhorn_repeat);
                for (unsigned k = 0; k < 16u; k++) comb_mix[(size_t)t * 16u + k] = c[k];
            } else {
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
        /* The gate goes to registers once per token rather than being re-read per `d`: for
         * PRE_DEFER it would otherwise be an HBM read inside the inner loop. `n` is 4 here (see
         * the file header -- `logits[24]`/`comb_lds[16]` are sized for it). */
        float g[4];
        for (unsigned i = 0; i < n; i++) {
            g[i] = pre_mode == PLOW_HC_PRE_OWN     ? logits[i]
                   : pre_mode == PLOW_HC_PRE_SEED  ? (i == 0 ? 1.0f : 0.0f)
                                                   : pre_pair[(size_t)pre_in_half * T * n + (size_t)t * n + i];
        }
        for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned i = 0; i < n; i++) acc += g[i] * bf2f(rrow[(size_t)i * hidden + d]);
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
        /* HOIST THE n RESIDUAL READS OUT OF THE j LOOP. Each output stream j sums over all n
         * input streams, so reading `rrow[i][d]` inside the j loop reads every residual element
         * n TIMES -- 16 loads to produce 4 outputs at n=4. At T=8192 that is 1.34 GB of reads
         * against a 335 MB tensor, and this op measured 3477 us where its traffic wants ~126 us.
         *
         * Needs the literal 4 to keep `rv` in registers: with a runtime `n` the loops do not
         * unroll and the array lands in scratch, which is worse than the re-reads. Same
         * `T >= nblk` gate as the pre op, for the same reason -- the decode path keeps the
         * shipped loop, which is the arm the gfx950 oracle signed off.
         *
         * BIT-IDENTICAL: the same products accumulated in the same i order. */
        if (n == 4u && T >= nblk) {
            constexpr unsigned NC = 4u;
            for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
                const float xv = bf2f(xrow[d]);
                float rv[NC];
                for (unsigned i = 0; i < NC; i++) rv[i] = bf2f(rrow[(size_t)i * hidden + d]);
                for (unsigned j = 0; j < NC; j++) {
                    float acc = 0.0f;
                    for (unsigned i = 0; i < NC; i++) acc += cm[i * NC + j] * rv[i];
                    acc += pm[j] * xv;
                    orow[(size_t)j * hidden + d] = f2bf(acc);
                }
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

#endif /* PLOW_OP_HYPERCONN_H */
