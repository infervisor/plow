/* op_deepseek_v4.h — DeepSeek-V4 hyper-connections.
 *
 * V4 HAS NO RESIDUAL ADD. The hidden state is `HC` parallel residual streams
 * (`hc_mult: 4`), and every sub-layer boundary is a learned mix over them
 * instead of `x = x + f(x)`:
 *
 *     reduce:  y      = sum_j pre[j] * x[j]                    [HC,D] -> [D]
 *     <sub-layer runs on y>
 *     expand:  x'[k]  = post[k] * branch + sum_j comb[j][k] * x[j]
 *
 * Getting this wrong is the silent kind, exactly as `AttnRes` is: a plain
 * residual add has the same shapes and is the `HC == 1, pre = post = comb = 1`
 * special case of the above, so it runs, produces fluent text, and is a
 * different network. `nn_graph::Op::HcReduce` / `HcExpand` model the same split.
 *
 * THE COEFFICIENTS, transcribed from `inference/model.py` (`Block.hc_pre`,
 * `Block.hc_post`, `Block.hc_head`) and `inference/kernel.py`
 * (`hc_split_sinkhorn_kernel`), which is the authority for the normalization:
 *
 *     xf     = x.flatten(streams)                          # [HC*D], fp32
 *     rsqrt  = rsqrt(mean(xf^2) + norm_eps)
 *     mixes  = (hc_fn @ xf) * rsqrt                        # [(2+HC)*HC]
 *     pre[j]     = sigmoid(mixes[j]      * scale[0] + base[j])      + hc_eps
 *     post[j]    = sigmoid(mixes[j+HC]   * scale[1] + base[j+HC]) * 2
 *     comb[j][k] =          mixes[2HC + j*HC + k] * scale[2] + base[2HC + j*HC + k]
 *     comb   = softmax(comb, dim=k) + hc_eps
 *     comb   = comb / (colsum_j + hc_eps)
 *     repeat (iters - 1):  comb = comb / (rowsum_k + hc_eps)
 *                          comb = comb / (colsum_j + hc_eps)
 *
 * Three things this pins that a code read of `model.py` alone does not:
 *
 *   1. `post` carries a factor of TWO that `pre` does not, and neither the
 *      model nor the config says so — it is only in the tilelang kernel.
 *   2. the Sinkhorn is row-then-column, and the FIRST row pass is a softmax
 *      (with `+eps` after it) while every later one is a plain sum-divide. An
 *      implementation that ran `iters` symmetric passes would be close and
 *      wrong.
 *   3. the reduce mixes the RAW streams. `rsqrt` scales the MIXES only — it
 *      never touches the value that is summed. (`AttnRes` has the same shape of
 *      trap and the same answer.)
 *
 * WHY THE MIXES ARE STASHED. `post` and `comb` are computed by the reduce and
 * consumed by the expand at the far side of a whole sub-layer. Recomputing them
 * there would mean re-running the `[(2+HC)*HC, HC*D]` projection — 393K MAC per
 * token, against the ~64 the expand itself costs. So the reduce writes them to
 * a small `[T, HC + HC*HC]` fp32 scratch and the expand reads it back. The IR
 * models the expand as recomputing (`Op::HcExpand` takes the weights) because
 * that keeps its single-output invariant; both spellings compute the same
 * numbers from the same inputs, and the scratch is the implementation of the
 * dependency, not a different model.
 *
 * COST. `hc_fn` is `[24, HC*D]` fp32 = 1.5 MiB per sub-layer, and at decode
 * every token streams all of it: 43 layers x 2 sub-layers = 135 MiB/token,
 * ~32 us at MI325X's measured 4164 GB/s. That is ~1.6% of the model's active
 * weight traffic, so it is not a blocker — but it is the reason this op is
 * bandwidth-bound on its WEIGHTS rather than on the activation, which is what a
 * tuning pass should attack first (the checkpoint stores fp32; bf16 would halve
 * it, and that is a numerics question, not a scheduling one).
 */
#pragma once

#include "amd_common.h"

/* `hc_mult` is 4 in every V4 checkpoint. The bound is what lets `comb` and the
 * per-stream vectors live in registers/LDS at a fixed size; a checkpoint above
 * it poisons rather than silently truncating. */
#define PLOW_V4_HC_MAX 8
#define PLOW_V4_MIX_MAX ((2 + PLOW_V4_HC_MAX) * PLOW_V4_HC_MAX)

/* mixes[(2+HC)*HC] -> pre[HC], post[HC], comb[HC][HC]. Serial and thread-0-only:
 * HC is 4, so this is ~20 iterations over a 4x4 — hundreds of flops against the
 * projection's hundreds of thousands. Parallelizing it would cost more in
 * barriers than it saves. */
__device__ __forceinline__ void d_hc_split_sinkhorn(const float* __restrict__ mixes,
                                                    const float* __restrict__ scale,
                                                    const float* __restrict__ base, unsigned HC,
                                                    unsigned iters, float eps,
                                                    float* __restrict__ pre,
                                                    float* __restrict__ post,
                                                    float* __restrict__ comb) {
    for (unsigned j = 0; j < HC; j++) {
        pre[j] = 1.0f / (1.0f + __expf(-(mixes[j] * scale[0] + base[j]))) + eps;
        /* The factor of 2 is `hc_split_sinkhorn_kernel`'s, not the model's. */
        post[j] = 2.0f / (1.0f + __expf(-(mixes[j + HC] * scale[1] + base[j + HC])));
    }
    for (unsigned j = 0; j < HC; j++)
        for (unsigned k = 0; k < HC; k++) {
            const unsigned m = 2u * HC + j * HC + k;
            comb[j * HC + k] = mixes[m] * scale[2] + base[m];
        }

    /* Pass 0 is a row SOFTMAX (then +eps), not a sum-divide. */
    for (unsigned j = 0; j < HC; j++) {
        float mx = comb[j * HC];
        for (unsigned k = 1; k < HC; k++) mx = fmaxf(mx, comb[j * HC + k]);
        float s = 0.0f;
        for (unsigned k = 0; k < HC; k++) {
            const float e = __expf(comb[j * HC + k] - mx);
            comb[j * HC + k] = e;
            s += e;
        }
        for (unsigned k = 0; k < HC; k++) comb[j * HC + k] = comb[j * HC + k] / s + eps;
    }
    /* ...then a column pass, and only then the (iters-1) symmetric rounds. */
    for (unsigned k = 0; k < HC; k++) {
        float c = 0.0f;
        for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
        c += eps;
        for (unsigned j = 0; j < HC; j++) comb[j * HC + k] /= c;
    }
    for (unsigned it = 1; it < iters; it++) {
        for (unsigned j = 0; j < HC; j++) {
            float r = 0.0f;
            for (unsigned k = 0; k < HC; k++) r += comb[j * HC + k];
            r += eps;
            for (unsigned k = 0; k < HC; k++) comb[j * HC + k] /= r;
        }
        for (unsigned k = 0; k < HC; k++) {
            float c = 0.0f;
            for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
            c += eps;
            for (unsigned j = 0; j < HC; j++) comb[j * HC + k] /= c;
        }
    }
}

/* HC-stream reduce, plus the mix stash the paired expand reads.
 *
 * `x`       [T, HC, D] bf16      the residual streams
 * `hc_fn`   [(2+HC)*HC, HC*D] f32
 * `scale`   [3] f32,  `base` [(2+HC)*HC] f32
 * `out`     [T, D]    bf16       the vector the sub-layer consumes
 * `mix_out` [T, HC + HC*HC] f32  post ++ comb, for `d_hc_expand`
 *
 * One workgroup per token slice; workgroups partition tokens, so nothing here
 * is shared across them and no cross-workgroup barrier is needed.
 *
 * `lds` needs `PLOW_WAVES * (1 + MIX)` floats of reduction scratch plus
 * `MIX + 2*HC + HC*HC` for the coefficients. */
__device__ void d_hc_reduce(bf16* __restrict__ out, float* __restrict__ mix_out,
                            const bf16* __restrict__ x, const float* __restrict__ hc_fn,
                            const float* __restrict__ scale, const float* __restrict__ base,
                            unsigned T, unsigned D, unsigned HC, unsigned iters, float norm_eps,
                            float hc_eps, unsigned slice, unsigned nblk, float* __restrict__ lds) {
    const unsigned MIX = (2u + HC) * HC;
    const unsigned N = HC * D; /* flattened stream width */

    /* POISON, do not return. A silent NOP is indistinguishable from a missing
     * opcode — the dispatch `default:` is already one. */
    if (HC > PLOW_V4_HC_MAX || HC == 0) {
        for (unsigned t = slice; t < T; t += nblk)
            for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS)
                st_act1(&out[(size_t)t * D + d], (bf16)0x7fc1u); /* qNaN */
        return;
    }

    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    float* part = lds;                              /* [PLOW_WAVES][1 + MIX] */
    float* mixes = lds + PLOW_WAVES * (1u + MIX);   /* [MIX] */
    float* pre = mixes + MIX;                       /* [HC]  */
    float* post = pre + HC;                         /* [HC]  */
    float* comb = post + HC;                        /* [HC*HC] */

    for (unsigned t = slice; t < T; t += nblk) {
        const bf16* xt = x + (size_t)t * N;

        /* One pass over the flattened streams accumulating sum(x^2) and all MIX
         * dot products at once: the activation is read ONCE and every mix row is
         * a separate stream of weights. */
        float ss = 0.0f;
        float dot[PLOW_V4_MIX_MAX];
        for (unsigned m = 0; m < MIX; m++) dot[m] = 0.0f;
        for (unsigned i = threadIdx.x; i < N; i += PLOW_THREADS) {
            const float v = bf2f(xt[i]);
            ss += v * v;
            for (unsigned m = 0; m < MIX; m++) dot[m] += v * hc_fn[(size_t)m * N + i];
        }
        /* Wave reduce, then a cross-wave pass through LDS. */
        for (int o = 32; o; o >>= 1) {
            ss += __shfl_down(ss, o);
            for (unsigned m = 0; m < MIX; m++) dot[m] += __shfl_down(dot[m], o);
        }
        if (lane == 0) {
            part[wave * (1u + MIX)] = ss;
            for (unsigned m = 0; m < MIX; m++) part[wave * (1u + MIX) + 1u + m] = dot[m];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tss = 0.0f;
            for (unsigned w = 0; w < PLOW_WAVES; w++) tss += part[w * (1u + MIX)];
            /* `mean(x^2)` over the FLATTENED streams, RMSNorm-style — not
             * mean-centred, and not per stream. */
            const float rs = rsqrtf(tss / (float)N + norm_eps);
            for (unsigned m = 0; m < MIX; m++) {
                float d = 0.0f;
                for (unsigned w = 0; w < PLOW_WAVES; w++) d += part[w * (1u + MIX) + 1u + m];
                mixes[m] = d * rs;
            }
            d_hc_split_sinkhorn(mixes, scale, base, HC, iters, hc_eps, pre, post, comb);
            /* Stash post ++ comb for the expand at the other end of the sub-layer. */
            float* mo = mix_out + (size_t)t * (HC + HC * HC);
            for (unsigned j = 0; j < HC; j++) mo[j] = post[j];
            for (unsigned j = 0; j < HC * HC; j++) mo[HC + j] = comb[j];
        }
        __syncthreads();

        /* The mix is over the RAW streams. */
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned j = 0; j < HC; j++) acc += pre[j] * bf2f(xt[(size_t)j * D + d]);
            st_act1(&out[(size_t)t * D + d], f2bf(acc));
        }
        __syncthreads(); /* `pre` is reused next token */
    }
}

/* HC-stream expand: write a sub-layer's output back into the streams.
 *
 * `branch`   [T, D]    bf16   the sub-layer output
 * `residual` [T, HC, D] bf16  the streams the paired reduce consumed
 * `mix_in`   [T, HC + HC*HC] f32  post ++ comb, as written by `d_hc_reduce`
 * `out`      [T, HC, D] bf16  may alias `residual`
 *
 * `out` aliasing `residual` is safe because every output element `k` reads
 * every input element `j` of the SAME token and depth, so the whole
 * `[HC]`-vector at one `(t, d)` is loaded before any of it is stored. */
__device__ void d_hc_expand(bf16* __restrict__ out, const bf16* __restrict__ branch,
                            const bf16* residual, const float* __restrict__ mix_in, unsigned T,
                            unsigned D, unsigned HC, unsigned slice, unsigned nblk) {
    if (HC > PLOW_V4_HC_MAX || HC == 0) {
        for (unsigned t = slice; t < T; t += nblk)
            for (unsigned i = threadIdx.x; i < HC * D; i += PLOW_THREADS)
                st_act1(&out[(size_t)t * HC * D + i], (bf16)0x7fc1u);
        return;
    }
    for (unsigned t = slice; t < T; t += nblk) {
        const float* mi = mix_in + (size_t)t * (HC + HC * HC);
        const bf16* rt = residual + (size_t)t * HC * D;
        const bf16* bt = branch + (size_t)t * D;
        bf16* ot = out + (size_t)t * HC * D;
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float r[PLOW_V4_HC_MAX];
            for (unsigned j = 0; j < HC; j++) r[j] = bf2f(rt[(size_t)j * D + d]);
            const float b = bf2f(bt[d]);
            for (unsigned k = 0; k < HC; k++) {
                float acc = mi[k] * b; /* post[k] * branch */
                for (unsigned j = 0; j < HC; j++) acc += mi[HC + j * HC + k] * r[j];
                st_act1(&ot[(size_t)k * D + d], f2bf(acc));
            }
        }
    }
}

/* The final, un-normalized reduce that feeds the lm_head (`Block.hc_head`).
 *
 * A DIFFERENT formula, not `d_hc_reduce` with `iters = 0`: `hc_fn` is `[HC,
 * HC*D]` rather than `[(2+HC)*HC, HC*D]`, `hc_scale` is a single scalar, and
 * there is no Sinkhorn and no expand partner — just `sigmoid(mix*scale+base) +
 * eps` as the per-stream weight. */
__device__ void d_hc_reduce_head(bf16* __restrict__ out, const bf16* __restrict__ x,
                                 const float* __restrict__ hc_fn, const float* __restrict__ scale,
                                 const float* __restrict__ base, unsigned T, unsigned D,
                                 unsigned HC, float norm_eps, float hc_eps, unsigned slice,
                                 unsigned nblk, float* __restrict__ lds) {
    const unsigned N = HC * D;
    if (HC > PLOW_V4_HC_MAX || HC == 0) {
        for (unsigned t = slice; t < T; t += nblk)
            for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS)
                st_act1(&out[(size_t)t * D + d], (bf16)0x7fc1u);
        return;
    }
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    float* part = lds;                            /* [PLOW_WAVES][1 + HC] */
    float* pre = lds + PLOW_WAVES * (1u + HC);    /* [HC] */

    for (unsigned t = slice; t < T; t += nblk) {
        const bf16* xt = x + (size_t)t * N;
        float ss = 0.0f;
        float dot[PLOW_V4_HC_MAX];
        for (unsigned m = 0; m < HC; m++) dot[m] = 0.0f;
        for (unsigned i = threadIdx.x; i < N; i += PLOW_THREADS) {
            const float v = bf2f(xt[i]);
            ss += v * v;
            for (unsigned m = 0; m < HC; m++) dot[m] += v * hc_fn[(size_t)m * N + i];
        }
        for (int o = 32; o; o >>= 1) {
            ss += __shfl_down(ss, o);
            for (unsigned m = 0; m < HC; m++) dot[m] += __shfl_down(dot[m], o);
        }
        if (lane == 0) {
            part[wave * (1u + HC)] = ss;
            for (unsigned m = 0; m < HC; m++) part[wave * (1u + HC) + 1u + m] = dot[m];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tss = 0.0f;
            for (unsigned w = 0; w < PLOW_WAVES; w++) tss += part[w * (1u + HC)];
            const float rs = rsqrtf(tss / (float)N + norm_eps);
            for (unsigned m = 0; m < HC; m++) {
                float d = 0.0f;
                for (unsigned w = 0; w < PLOW_WAVES; w++) d += part[w * (1u + HC) + 1u + m];
                pre[m] = 1.0f / (1.0f + __expf(-(d * rs * scale[0] + base[m]))) + hc_eps;
            }
        }
        __syncthreads();
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned j = 0; j < HC; j++) acc += pre[j] * bf2f(xt[(size_t)j * D + d]);
            st_act1(&out[(size_t)t * D + d], f2bf(acc));
        }
        __syncthreads();
    }
}
