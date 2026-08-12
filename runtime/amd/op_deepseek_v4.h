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
/* Compacted equal-to-threshold entries the top-k fix-up may hold. Ties at an
 * exact fp32 threshold are rare; this only has to cover the normal case. */
#define PLOW_V4_TIE_MAX 1024u

/* mixes[(2+HC)*HC] -> pre[HC], post[HC], comb[HC][HC]. Serial and thread-0-only:
 * HC is 4, so this is ~20 iterations over a 4x4 — hundreds of flops against the
 * projection's hundreds of thousands. Parallelizing it would cost more in
 * barriers than it saves.
 *
 * THE STATE HAS TO BE IN REGISTERS, AND THAT NEEDS A COMPILE-TIME `HC`.
 *
 * The arithmetic is a strict dependency chain — every element of round `it`
 * reads what round `it - 1` wrote — so nothing hides the latency of whatever
 * memory it lives in. Three spellings, all measured on this part at the shipped
 * HC=4, iters=20, as the tail of the split reduce:
 *
 *   state in LDS                        78.6 us
 *   plain local arrays, runtime HC     174.8 us   <- WORSE
 *   local arrays, compile-time HC        (below)
 *
 * The middle one is the trap: a local array indexed by a runtime-derived
 * `j * HC + k` is not a register file, it is SCRATCH, which is global memory
 * with a private aperture. So the fast path is templated on `HC` — the same
 * reason `d_headnorm_rope` templates `head_dim` rather than passing it — and a
 * checkpoint with any other `hc_mult` keeps the generic LDS path. */
template <unsigned HCT>
__device__ __forceinline__ void d_hc_split_sinkhorn_t(const float* __restrict__ mixes,
                                                      const float* __restrict__ scale,
                                                      const float* __restrict__ base,
                                                      unsigned iters, float eps,
                                                      float* __restrict__ pre_out,
                                                      float* __restrict__ post_out,
                                                      float* __restrict__ comb_out) {
    float pre[HCT], post[HCT], comb[HCT * HCT];
#pragma unroll
    for (unsigned j = 0; j < HCT; j++) {
        pre[j] = 1.0f / (1.0f + __expf(-(mixes[j] * scale[0] + base[j]))) + eps;
        post[j] = 2.0f / (1.0f + __expf(-(mixes[j + HCT] * scale[1] + base[j + HCT])));
    }
#pragma unroll
    for (unsigned j = 0; j < HCT; j++)
#pragma unroll
        for (unsigned k = 0; k < HCT; k++) {
            const unsigned m = 2u * HCT + j * HCT + k;
            comb[j * HCT + k] = mixes[m] * scale[2] + base[m];
        }
#pragma unroll
    for (unsigned j = 0; j < HCT; j++) {
        float mx = comb[j * HCT];
#pragma unroll
        for (unsigned k = 1; k < HCT; k++) mx = fmaxf(mx, comb[j * HCT + k]);
        float s = 0.0f;
#pragma unroll
        for (unsigned k = 0; k < HCT; k++) {
            const float e = __expf(comb[j * HCT + k] - mx);
            comb[j * HCT + k] = e;
            s += e;
        }
        const float is = 1.0f / s;
#pragma unroll
        for (unsigned k = 0; k < HCT; k++) comb[j * HCT + k] = comb[j * HCT + k] * is + eps;
    }
#pragma unroll
    for (unsigned k = 0; k < HCT; k++) {
        float c = eps;
#pragma unroll
        for (unsigned j = 0; j < HCT; j++) c += comb[j * HCT + k];
        const float ic = 1.0f / c;
#pragma unroll
        for (unsigned j = 0; j < HCT; j++) comb[j * HCT + k] *= ic;
    }
    for (unsigned it = 1; it < iters; it++) {
#pragma unroll
        for (unsigned j = 0; j < HCT; j++) {
            float r = eps;
#pragma unroll
            for (unsigned k = 0; k < HCT; k++) r += comb[j * HCT + k];
            const float ir = 1.0f / r;
#pragma unroll
            for (unsigned k = 0; k < HCT; k++) comb[j * HCT + k] *= ir;
        }
#pragma unroll
        for (unsigned k = 0; k < HCT; k++) {
            float c = eps;
#pragma unroll
            for (unsigned j = 0; j < HCT; j++) c += comb[j * HCT + k];
            const float ic = 1.0f / c;
#pragma unroll
            for (unsigned j = 0; j < HCT; j++) comb[j * HCT + k] *= ic;
        }
    }
#pragma unroll
    for (unsigned j = 0; j < HCT; j++) {
        pre_out[j] = pre[j];
        post_out[j] = post[j];
    }
#pragma unroll
    for (unsigned j = 0; j < HCT * HCT; j++) comb_out[j] = comb[j];
}
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
    /* ...then a column pass, and only then the (iters-1) symmetric rounds.
     *
     * ONE RECIPROCAL PER ROW, NOT A DIVIDE PER ELEMENT. `iters` is 20 and each
     * round normalizes `HC` rows and `HC` columns, so the naive spelling issues
     * `2 * iters * HC * HC` = 640 fp32 divides — and an fp32 divide is an
     * instruction SEQUENCE, not one VALU op. Measured, that was the entire cost
     * of the reduce's tail: 91.8 us of a 104.5 us pair, single-threaded and
     * replicated across every block, against 8.0 us for the 1.5 MiB projection
     * it follows. Hoisting the reciprocal makes it `2 * iters * HC` = 160. */
    for (unsigned k = 0; k < HC; k++) {
        float c = eps;
        for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
        const float ic = 1.0f / c;
        for (unsigned j = 0; j < HC; j++) comb[j * HC + k] *= ic;
    }
    for (unsigned it = 1; it < iters; it++) {
        for (unsigned j = 0; j < HC; j++) {
            float r = eps;
            for (unsigned k = 0; k < HC; k++) r += comb[j * HC + k];
            const float ir = 1.0f / r;
            for (unsigned k = 0; k < HC; k++) comb[j * HC + k] *= ir;
        }
        for (unsigned k = 0; k < HC; k++) {
            float c = eps;
            for (unsigned j = 0; j < HC; j++) c += comb[j * HC + k];
            const float ic = 1.0f / c;
            for (unsigned j = 0; j < HC; j++) comb[j * HC + k] *= ic;
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
            if (HC == 4)
                d_hc_split_sinkhorn_t<4>(mixes, scale, base, iters, hc_eps, pre, post, comb);
            else
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
    /* Grid-strided over (token, depth), not over tokens. The expand is
     * embarrassingly parallel — every `(t, d)` is independent — so a
     * token-parallel loop left it on ONE CU at decode, where it measured 25.2 us
     * for 72 KiB of traffic and would have been co-dominant with the reduce it
     * pairs with. Same defect, same fix, and it is worth stating twice because
     * both were written the same way. */
    const size_t work = (size_t)T * D;
    for (size_t w = (size_t)slice * PLOW_THREADS + threadIdx.x; w < work;
         w += (size_t)nblk * PLOW_THREADS) {
        const unsigned t = (unsigned)(w / D), d = (unsigned)(w - (size_t)t * D);
        const float* mi = mix_in + (size_t)t * (HC + HC * HC);
        const bf16* rt = residual + (size_t)t * HC * D;
        bf16* ot = out + (size_t)t * HC * D;
        float r[PLOW_V4_HC_MAX];
        for (unsigned j = 0; j < HC; j++) r[j] = bf2f(rt[(size_t)j * D + d]);
        const float b = bf2f(branch[(size_t)t * D + d]);
        for (unsigned k = 0; k < HC; k++) {
            float acc = mi[k] * b; /* post[k] * branch */
            for (unsigned j = 0; j < HC; j++) acc += mi[HC + j * HC + k] * r[j];
            st_act1(&ot[(size_t)k * D + d], f2bf(acc));
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

/* ---------------------------------------------------------------------------
 * Sparse attention with a learned sink — V4's ONLY attention kernel.
 * ---------------------------------------------------------------------------
 *
 * The window and the compressed history are not two kernels. `Attention.forward`
 * builds ONE index list per query — `cat(window_idxs, compress_idxs)` — and
 * hands it to `sparse_attn`; the sliding window is just the first `window_size`
 * entries and the compressed history the rest. A layer with no compression
 * passes the window indices alone. So this one op covers G2's window attention
 * and the consumption side of G3/G4, and there is no dense arm to write.
 *
 * `q`         [T, H, D] bf16     H query heads
 * `kv`        [NKV, D]  bf16     ONE shared KV head — key and value are the
 *                                same tensor, not two projections
 * `idx`       [T, TOPK] i32      gathered positions; -1 means "no entry"
 * `sink`      [H] f32            learned per-head logit
 * `out`       [T, H, D] bf16
 *
 * TRANSCRIBED FROM `inference/kernel.py::sparse_attn_kernel`, which is the
 * authority for three things the model file does not state:
 *
 *   1. THE SINK NEVER ENTERS THE NUMERATOR. It is added to the denominator
 *      once, after the loop, as `exp(sink[h] - m_final)` — it has no value row.
 *      A sink with a value row is a different function; a sink that only
 *      inflates the denominator is this one.
 *   2. Whether it joins the running max is NOT a correctness question, and the
 *      earlier note here claiming otherwise was wrong. Softmax is shift
 *      invariant, so folding `sink` into the max and rescaling `acc`/`l` by
 *      `exp(m - max(m, sink))` computes the identical value — measured, not
 *      argued: that variant passes this op's oracle unchanged at every swept
 *      shape. It is a numerical-range choice, and the reference's form (max
 *      over real keys only) is kept because it is the reference's.
 *   3. `idx == -1` contributes a `-inf` score AND a zeroed KV row, so a padded
 *      block is inert rather than reading position 0.
 *
 * K AND V ARE THE SAME ROWS. V4 projects one `[D]` KV vector per token and uses
 * it as both, so there is no `v_head_dim != qk_head_dim` split to carry and the
 * output width is D.
 *
 * WORK DECOMPOSITION, and why the first version of this op was 0.0% of roofline.
 *
 * It parallelized over TOKENS only — `for (t = slice; t < T; t += nblk)`. At
 * decode `T == 1`, so exactly ONE workgroup had work and 303 of 304 CUs idled;
 * the measured 8.3 ms against a 1.03 us VALU roof was almost entirely that, not
 * the inner loop. Token-parallel is the right axis for prefill and the wrong one
 * for the shape that decides interactive latency.
 *
 * So the work index is `(t, h)` flattened: one workgroup per (token, head), and
 * the block's waves SPLIT THE KEYS, each carrying its own `(m, l, acc)` and
 * combining through LDS at the end — the standard flash-decoding split. At the
 * shipped shape that is 64 workgroups x 8 key-splits instead of 1 workgroup.
 *
 * Still not the last word: lanes own dims and keys are consumed one at a time,
 * so a key costs a six-shuffle reduction against `D/64` useful MACs. Replacing
 * that with an MFMA tile over a key block (as the tilelang reference does with
 * `block = 64`, `T.gemm`) is the remaining factor and it is a bigger rewrite. */
__device__ void d_v4_sparse_attn(bf16* __restrict__ out, const bf16* __restrict__ q,
                                 const bf16* __restrict__ kv, const int* __restrict__ idx,
                                 const float* __restrict__ sink, unsigned T, unsigned H,
                                 unsigned D, unsigned TOPK, float scale, unsigned slice,
                                 unsigned nblk, float* __restrict__ lds) {
    /* One lane per 64th of the head dim. A D that is not a multiple of 64 would
     * leave a ragged tail this layout cannot address; poison rather than drop
     * the tail silently. V4 uses D=512 (attention) and D=128 (indexer). */
    if ((D & 63u) || H == 0) {
        for (unsigned w = slice; w < T * H; w += nblk) {
            bf16* oh = out + (size_t)w * D;
            for (unsigned i = threadIdx.x; i < D; i += PLOW_THREADS)
                st_act1(&oh[i], (bf16)0x7fc1u);
        }
        return;
    }
    const unsigned DPL = D >> 6; /* dims per lane */
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const unsigned nwave = PLOW_THREADS >> 6;
    float* pacc = lds;                       /* [nwave][D]  partial numerators */
    float* pml = lds + (size_t)nwave * D;    /* [nwave][2]  partial m and l    */

    /* One workgroup per (token, head); the block's waves split the keys. */
    for (unsigned w = slice; w < T * H; w += nblk) {
        const unsigned t = w / H, h = w % H;
        const int* it = idx + (size_t)t * TOPK;
        const bf16* qh = q + ((size_t)t * H + h) * D;
        float qv[16], acc[16]; /* DPL <= 16 given D <= 1024 */
        for (unsigned e = 0; e < DPL; e++) {
            qv[e] = bf2f(qh[lane * DPL + e]);
            acc[e] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;

        /* `DPL == 8` is D=512, the shipped attention width: one `bf16v8` per lane
         * and the score dot is four `v_dot2c_f32_bf16` instead of eight
         * shift-convert-FMA triples. Anything else takes the scalar path.
         *
         * KEYS ARE TAKEN FOUR AT A TIME. The gather is indirect (`kv[idx[j]]`),
         * so the address is not known until `idx` lands and a one-key loop is a
         * chain of DEPENDENT global loads with the online-softmax update between
         * them — at occupancy 2 there is nothing to hide that with. Four
         * independent loads in flight is what turns this from latency-bound
         * into throughput-bound; the softmax chain itself stays strictly
         * sequential, so the arithmetic is unchanged. */
        const unsigned KU = 4;
        if (DPL == 8) {
            const bf16v8 q8 = ld_glob8(qh + lane * 8);
            for (unsigned j0 = wave * KU; j0 < TOPK; j0 += nwave * KU) {
                int p[KU];
                bf16v8 k8[KU];
                float part[KU];
#pragma unroll
                for (unsigned u = 0; u < KU; u++) {
                    const unsigned j = j0 + u;
                    p[u] = (j < TOPK) ? it[j] : -1;
                    if (p[u] >= 0) k8[u] = ld_glob8(kv + (size_t)p[u] * D + lane * 8);
                }
#pragma unroll
                for (unsigned u = 0; u < KU; u++)
                    part[u] = (p[u] >= 0) ? dot8(q8, k8[u], 0.0f) : 0.0f;
#pragma unroll
                for (unsigned u = 0; u < KU; u++)
                    for (int o = 32; o; o >>= 1) part[u] += __shfl_xor(part[u], o);
#pragma unroll
                for (unsigned u = 0; u < KU; u++) {
                    if (p[u] < 0) continue;
                    const float s = part[u] * scale;
                    const float mn = fmaxf(m, s);
                    const float resc = (m == -INFINITY) ? 0.0f : __expf(m - mn);
                    const float pe = __expf(s - mn);
                    for (unsigned e = 0; e < 8; e++)
                        acc[e] = acc[e] * resc + pe * bf2f(k8[u][e]);
                    l = l * resc + pe;
                    m = mn;
                }
            }
        } else {
            for (unsigned j = wave; j < TOPK; j += nwave) {
                const int p = it[j];
                float kvv[16];
                float part = 0.0f;
                if (p >= 0) {
                    const bf16* kr = kv + (size_t)p * D;
                    for (unsigned e = 0; e < DPL; e++) {
                        kvv[e] = bf2f(kr[lane * DPL + e]);
                        part += qv[e] * kvv[e];
                    }
                } else {
                    for (unsigned e = 0; e < DPL; e++) kvv[e] = 0.0f;
                }
                /* Full-wave sum: every lane ends with the same score. */
                for (int o = 32; o; o >>= 1) part += __shfl_xor(part, o);
                const float s = (p >= 0) ? part * scale : -INFINITY;

                const float mn = fmaxf(m, s);
                const float resc = (m == -INFINITY) ? 0.0f : __expf(m - mn);
                const float pe = (s == -INFINITY) ? 0.0f : __expf(s - mn);
                for (unsigned e = 0; e < DPL; e++) acc[e] = acc[e] * resc + pe * kvv[e];
                l = l * resc + pe;
                m = mn;
            }
        }

        /* Flash combine across the key-split waves. Each wave's partial is at
         * its own max, so rescale to the block max before summing. */
        for (unsigned e = 0; e < DPL; e++) pacc[(size_t)wave * D + lane * DPL + e] = acc[e];
        if (lane == 0) {
            pml[wave * 2] = m;
            pml[wave * 2 + 1] = l;
        }
        __syncthreads();

        float mb = -INFINITY;
        for (unsigned v = 0; v < nwave; v++) mb = fmaxf(mb, pml[v * 2]);
        float lb = 0.0f;
        for (unsigned v = 0; v < nwave; v++)
            if (pml[v * 2] != -INFINITY) lb += pml[v * 2 + 1] * __expf(pml[v * 2] - mb);
        /* The sink joins the DENOMINATOR only, at the block max, after every
         * partial has been rescaled onto it. */
        if (mb != -INFINITY) lb += __expf(sink[h] - mb);
        const float inv = (lb > 0.0f) ? 1.0f / lb : 0.0f;

        bf16* oh = out + ((size_t)t * H + h) * D;
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float s = 0.0f;
            for (unsigned v = 0; v < nwave; v++)
                if (pml[v * 2] != -INFINITY)
                    s += pacc[(size_t)v * D + d] * __expf(pml[v * 2] - mb);
            st_act1(&oh[d], f2bf(s * inv));
        }
        __syncthreads(); /* `pacc`/`pml` are reused by the next work item */
    }
}

/* ---------------------------------------------------------------------------
 * Learned KV compressor — prefill form.
 * ---------------------------------------------------------------------------
 *
 * Pools every `ratio` consecutive tokens into ONE compressed KV entry. This is
 * the only op in the model that changes the sequence rate, and 41 of 43 layers
 * have one (21 of those additionally overlapping, at ratio 4).
 *
 * `x`       [T, HID] bf16      the layer input
 * `wkv`     [COFF*D, HID] bf16   COFF = 2 when overlapping, else 1
 * `wgate`   [COFF*D, HID] bf16
 * `ape`     [ratio, COFF*D] f32  per-offset-within-window bias on the scores
 * `norm_w`  [D] bf16
 * `out`     [T/ratio, D] bf16
 *
 * TRANSCRIBED FROM `inference/model.py::Compressor.forward` at `start_pos == 0`.
 * Four things it pins:
 *
 *   1. THE SOFTMAX IS PER OUTPUT DIM. `score` has the same width as `kv`, and
 *      `score.softmax(dim=2)` normalizes over the pooled ROWS independently for
 *      every one of the D columns. It is not one weight per row.
 *   2. `ape` is added to the scores BEFORE the overlap transform, indexed by
 *      the token's position WITHIN ITS OWN window — so an overlapped group's
 *      two halves carry ape rows from two different windows.
 *   3. THE OVERLAPPED FORM POOLS 2*ratio ROWS: the PREVIOUS window's tokens
 *      through the FIRST half of the projection (`[0:D]`), and the CURRENT
 *      window's through the SECOND (`[D:2D]`). That is what the doubled weight
 *      width is for. Group 0 has no previous window, and its first `ratio` rows
 *      are a zero value at a `-inf` score — inert, not zero-weighted-average.
 *   4. the pooled result is RMS-normed with `norm_w`, and only then (outside
 *      this op, as in the IR) roped at the compressed position.
 *
 * NOT IMPLEMENTED HERE: the decode-incremental form (`start_pos > 0`), which
 * keeps a per-sequence window state and emits one entry every `ratio` steps,
 * and the prefill remainder that seeds that state. Both are the KV-ring side of
 * the op and belong with the runtime's cache management; this is the half a
 * single-block correctness sweep needs. `T` not a multiple of `ratio` has its
 * tail DROPPED here rather than silently folded into the last group.
 *
 * PERFORMANCE STATUS: correctness reference. Every pooled row re-streams the
 * whole projection, which is inherent to a per-token GEMV but is exactly what a
 * prefill GEMM would batch away. No performance claim is made. */
__device__ void d_v4_kv_compress(bf16* __restrict__ out, const bf16* __restrict__ x,
                                 const bf16* __restrict__ wkv, const bf16* __restrict__ wgate,
                                 const float* __restrict__ ape, const bf16* __restrict__ norm_w,
                                 unsigned T, unsigned HID, unsigned D, unsigned ratio,
                                 unsigned overlap, float eps, unsigned slice, unsigned nblk,
                                 float* __restrict__ lds) {
    const unsigned G = ratio ? T / ratio : 0u;
    if (ratio == 0 || G == 0) return;
    const unsigned COFF = overlap ? 2u : 1u;
    const unsigned NROW = overlap ? 2u * ratio : ratio;

    for (unsigned g = slice; g < G; g += nblk) {
        /* Online softmax per output dim: one pass over the pooled rows, so the
         * scores never have to be materialized (`ratio` reaches 128). */
        for (unsigned dd = threadIdx.x; dd < D; dd += PLOW_THREADS) {
            float m = -INFINITY, l = 0.0f, acc = 0.0f;
            for (unsigned r = 0; r < NROW; r++) {
                unsigned tok, wrow, arow;
                if (!overlap) {
                    tok = g * ratio + r;
                    wrow = dd;
                    arow = r;
                } else if (r < ratio) {
                    if (g == 0) continue; /* no previous window: -inf score, 0 value */
                    tok = (g - 1u) * ratio + r;
                    wrow = dd; /* FIRST half of the projection */
                    arow = r;
                } else {
                    const unsigned i = r - ratio;
                    tok = g * ratio + i;
                    wrow = D + dd; /* SECOND half */
                    arow = i;
                }
                const bf16* xt = x + (size_t)tok * HID;
                const bf16* wk = wkv + (size_t)wrow * HID;
                const bf16* wg = wgate + (size_t)wrow * HID;
                float kvv = 0.0f, sc = 0.0f;
                for (unsigned i = 0; i < HID; i++) {
                    const float xv = bf2f(xt[i]);
                    kvv += xv * bf2f(wk[i]);
                    sc += xv * bf2f(wg[i]);
                }
                sc += ape[(size_t)arow * (COFF * D) + wrow];

                const float mn = fmaxf(m, sc);
                const float resc = (m == -INFINITY) ? 0.0f : __expf(m - mn);
                const float pe = __expf(sc - mn);
                acc = acc * resc + pe * kvv;
                l = l * resc + pe;
                m = mn;
            }
            lds[dd] = (l > 0.0f) ? acc / l : 0.0f;
        }
        __syncthreads();

        /* RMSNorm over the pooled entry. */
        float ss = 0.0f;
        for (unsigned dd = threadIdx.x; dd < D; dd += PLOW_THREADS) ss += lds[dd] * lds[dd];
        for (int o = 32; o; o >>= 1) ss += __shfl_xor(ss, o);
        float* red = lds + D;
        if ((threadIdx.x & 63u) == 0) red[threadIdx.x >> 6] = ss;
        __syncthreads();
        float tot = 0.0f;
        for (unsigned w = 0; w < (PLOW_THREADS >> 6); w++) tot += red[w];
        const float rs = rsqrtf(tot / (float)D + eps);
        for (unsigned dd = threadIdx.x; dd < D; dd += PLOW_THREADS)
            st_act1(&out[(size_t)g * D + dd], f2bf(lds[dd] * rs * bf2f(norm_w[dd])));
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------
 * Block-diagonal output projection (`wo_a`).
 * ---------------------------------------------------------------------------
 *
 * `o`      [T, H*D] bf16        attention output, head-major
 * `w`      [G*R, H*D/G] bf16    ONE stacked tensor, viewed as [G, R, H*D/G]
 * `out`    [T, G*R] bf16
 *
 * Group `g` projects ONLY its own slice of the head axis. A dense `[G*R, H*D]`
 * linear of the same element count mixes groups the reference keeps apart, and
 * is what a reader who saw one tensor would naturally write — `nn_graph::
 * Op::GroupedLinear` exists to make that impossible to express by accident.
 *
 * The reference spells it `torch.einsum("bsgd,grd->bsgr", o.view(b,s,G,-1),
 * wo_a.view(G, R, -1))`, and notes that `wo_a` is fp8 in the checkpoint but is
 * applied in bf16 "for simplicity"; this follows the reference, so the fp8
 * arm is a Stage-4 question, not a correctness one. */
__device__ void d_v4_grouped_linear(bf16* __restrict__ out, const bf16* __restrict__ o,
                                    const bf16* __restrict__ w, unsigned T, unsigned GRP,
                                    unsigned R, unsigned WIDTH, unsigned slice, unsigned nblk) {
    for (unsigned t = slice; t < T; t += nblk)
        for (unsigned i = threadIdx.x; i < GRP * R; i += PLOW_THREADS) {
            const unsigned g = i / R, r = i % R;
            const bf16* orow = o + (size_t)t * GRP * WIDTH + (size_t)g * WIDTH;
            const bf16* wrow = w + ((size_t)g * R + r) * WIDTH;
            float acc = 0.0f;
            for (unsigned k = 0; k < WIDTH; k++) acc += bf2f(orow[k]) * bf2f(wrow[k]);
            st_act1(&out[(size_t)t * GRP * R + i], f2bf(acc));
        }
}

/* ---------------------------------------------------------------------------
 * MoE routing — scored and hash-routed.
 * ---------------------------------------------------------------------------
 *
 * `logits` [T, E] f32   gate GEMM output (`w_gate @ x`, computed in fp32)
 * `bias`   [E] f32      selection bias, or nullptr on a hash layer
 * `tid2eid`[V, K] i64   token-id -> expert table, or nullptr on a scored layer
 * `ids`    [T] i32      token ids, only read on a hash layer
 * `sel`    [T, K] i32   chosen experts
 * `wts`    [T, K] f32   combine weights
 *
 * TRANSCRIBED FROM `Gate.forward`. Four things it pins:
 *
 *   1. SCORING IS `sqrt(softplus(x))` — not softmax, not sigmoid. Both of the
 *      others are already in this tree, so picking one by habit is the easy
 *      mistake.
 *   2. THE BIAS SHIFTS SELECTION ONLY. `scores + bias` chooses the top-k; the
 *      COMBINE WEIGHTS are gathered from the UNBIASED scores. Using the biased
 *      score as the weight is a plausible reading of the code and a different
 *      model.
 *   3. RENORMALIZATION IS OVER THE SELECTED SET, and it happens for every
 *      scoring function except softmax — so for V4 it always happens.
 *   4. A HASH LAYER DOES NOT SCORE AT ALL: the expert set comes from
 *      `tid2eid[token_id]`, and the scores only supply the (still renormalized)
 *      weights. The gate GEMM is dead weight on those layers, which is a
 *      Stage-4 saving, not a correctness matter.
 *
 * Top-k here is a serial selection over E; E is 256 and K is 6, so this is 6
 * passes of 256 compares per token. */
__device__ void d_v4_moe_route(int* __restrict__ sel, float* __restrict__ wts,
                               unsigned char* __restrict__ table,
                               const float* __restrict__ logits, const float* __restrict__ bias,
                               const long long* __restrict__ tid2eid, const int* __restrict__ ids,
                               unsigned T, unsigned E, unsigned K, float route_scale,
                               unsigned slice, unsigned nblk, float* __restrict__ lds) {
    for (unsigned t = slice; t < T; t += nblk) {
        const float* lg = logits + (size_t)t * E;
        /* `sqrt(softplus(x))`, in fp32 as the reference computes it. */
        for (unsigned e = threadIdx.x; e < E; e += PLOW_THREADS) {
            const float x = lg[e];
            /* log1p(exp(x)) without overflowing for large x. */
            const float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
            lds[e] = sqrtf(sp);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int* s = sel + (size_t)t * K;
            float* w = wts + (size_t)t * K;
            if (tid2eid != nullptr) {
                const long long* row = tid2eid + (size_t)ids[t] * K;
                for (unsigned k = 0; k < K; k++) s[k] = (int)row[k];
            } else {
                /* Top-k on the BIASED score; the bias never reaches `w`. */
                for (unsigned k = 0; k < K; k++) s[k] = -1;
                for (unsigned k = 0; k < K; k++) {
                    float best = -INFINITY;
                    int bi = -1;
                    for (unsigned e = 0; e < E; e++) {
                        bool taken = false;
                        for (unsigned j = 0; j < k; j++) taken |= (s[j] == (int)e);
                        if (taken) continue;
                        const float v = lds[e] + (bias ? bias[e] : 0.0f);
                        if (v > best) {
                            best = v;
                            bi = (int)e;
                        }
                    }
                    s[k] = bi;
                }
            }
            float sum = 0.0f;
            for (unsigned k = 0; k < K; k++) {
                w[k] = lds[s[k]]; /* UNBIASED score */
                sum += w[k];
            }
            /* Renormalize over the selected set, then scale. */
            const float inv = (sum != 0.0f) ? 1.0f / sum : 0.0f;
            for (unsigned k = 0; k < K; k++) w[k] = w[k] * inv * route_scale;
            /* The SAME selection, in the layout the shipped expert arms read:
             * `[k] of {u32 expert_id, f32 gate}` (`d_moe_router`, op_moe.h). V4
             * scores and selects differently from every other family, but what
             * it hands the experts is the ordinary table — so ops 45/46 stream
             * its experts unmodified and `sel`/`wts` stay for the gate that
             * checks this against the reference. */
            if (table != nullptr) {
                unsigned char* tb = table + (size_t)t * K * 8;
                for (unsigned k = 0; k < K; k++) {
                    const unsigned id = (s[k] < 0) ? 0xFFFFFFFFu : (unsigned)s[k];
                    __builtin_memcpy(tb + k * 8, &id, 4);
                    __builtin_memcpy(tb + k * 8 + 4, &w[k], 4);
                }
            }
        }
        __syncthreads();
    }
}

/* Clamped SwiGLU — every expert in the model (`swiglu_limit: 10`).
 *
 *     gate = min(gate, limit)                  # ONE-SIDED
 *     up   = clamp(up, -limit, +limit)         # TWO-SIDED
 *     out  = silu(gate) * up
 *
 * The asymmetry is the reference's (`Expert.forward`), and it is why this is
 * not `Act(Silu)` + `Mul`: the clamp binds only in the tail, so dropping it
 * agrees on almost every token and diverges exactly where the activation blows
 * up. Same reasoning as `SituGlu`. */
__device__ void d_v4_clamped_swiglu(bf16* __restrict__ out, const bf16* __restrict__ gate,
                                    const bf16* __restrict__ up, unsigned n, float limit) {
    for (unsigned i = threadIdx.x + blockIdx.x * PLOW_THREADS; i < n;
         i += PLOW_THREADS * gridDim.x) {
        float g = bf2f(gate[i]), u = bf2f(up[i]);
        if (limit > 0.0f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        const float s = g / (1.0f + __expf(-g)); /* silu */
        st_act1(&out[i], f2bf(s * u));
    }
}

/* ---------------------------------------------------------------------------
 * Sparse indexer — scoring and top-k over the compressed history.
 * ---------------------------------------------------------------------------
 *
 * Runs on the 21 layers whose compress ratio is 4, and decides which compressed
 * entries their attention may read.
 *
 * `q`     [T, H, HD] bf16   indexer queries (wq_b of the shared q-lora, roped)
 * `ckv`   [NC, HD]   bf16   the indexer's OWN compressed KV (d_v4_kv_compress
 *                           at HD width — it does not reuse the layer's)
 * `w`     [T, H]     bf16   weights_proj output, ALREADY scaled by
 *                           `softmax_scale * H^-0.5` as the reference does
 * `score` [T, NC]    f32
 *
 *     score[t][c] = sum_h relu(q[t][h] . ckv[c]) * w[t][h]
 *
 * The relu is INSIDE the head sum, so a head that disagrees contributes zero
 * rather than cancelling another head's evidence — mean-pooling the raw dots
 * would be a different selector.
 *
 * WHAT IS NOT MODELED HERE, and it is a numerics gap rather than a structural
 * one: the reference runs `rotate_activation` (a Hadamard) and then an
 * fp4 quantize-dequantize on BOTH `q` and the compressed KV, as QAT
 * simulation. The Hadamard is orthogonal and applied to both sides of the dot
 * product, so it CANCELS exactly and its absence changes nothing; the fp4
 * rounding does not cancel, and near a tie it can flip which entries the top-k
 * keeps. Closing that means the fp4 activation path, which is Stage-4 work
 * alongside the other quantized arms.
 *
 * UNDER TP the reference all-reduces `score` BEFORE the top-k, so the selection
 * is a collective: ranks that disagree on the selected set decode different
 * tokens. This op computes a rank-local score; the reduction is the emitter's
 * to place, and it is not optional. */
__device__ void d_v4_index_score(float* __restrict__ score, const bf16* __restrict__ q,
                                 const bf16* __restrict__ ckv, const bf16* __restrict__ w,
                                 unsigned T, unsigned H, unsigned HD, unsigned NC, unsigned slice,
                                 unsigned nblk) {
    /* ONE WAVE PER COMPRESSED ENTRY, ONE HEAD PER LANE. `index_n_heads` is 64
     * and a wave is 64 lanes, so the head sum is exactly a wave reduction and
     * every lane has a full `[HD]` dot to itself — no cross-lane traffic until
     * the fold.
     *
     * The first version of this op strided `c` by the block and `t` by the
     * grid, which at decode (T=1) left one workgroup with all NC entries and
     * 303 CUs idle: 34 ms measured against a 1.64 us VALU roof. The work index
     * here is `(t, c)` flattened over WAVES across the whole grid. */
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const unsigned nwave = PLOW_THREADS >> 6;
    const unsigned wid = slice * nwave + wave;    /* global wave id  */
    const unsigned wstride = nblk * nwave;

    if (H != 64) {
        /* The lane==head mapping is what makes this shape work. Anything else
         * falls back to the portable form rather than computing nonsense. */
        for (unsigned x = slice; x < T * NC; x += nblk)
            for (unsigned c = threadIdx.x; c < 1u; c += PLOW_THREADS) {
                const unsigned t = x / NC, cc = x % NC;
                const bf16* kc = ckv + (size_t)cc * HD;
                float acc = 0.0f;
                for (unsigned h = 0; h < H; h++) {
                    const bf16* qh = q + ((size_t)t * H + h) * HD;
                    float d = 0.0f;
                    for (unsigned e = 0; e < HD; e++) d += bf2f(qh[e]) * bf2f(kc[e]);
                    acc += fmaxf(d, 0.0f) * bf2f(w[(size_t)t * H + h]);
                }
                score[(size_t)t * NC + cc] = acc;
            }
        return;
    }

    /* `q` and the head weight depend on the TOKEN and the lane, not on the
     * entry, so they are hoisted out of the entry loop and live in registers:
     * `HD/8` = 16 `bf16v8` per lane. Re-reading them per entry was 134 MB of
     * redundant L1 traffic at NC=8192 for a 16 KiB tensor. */
    const unsigned NV = HD >> 3;
    for (unsigned t = 0; t < T; t++) {
        bf16v8 q8[16]; /* HD <= 128 */
        if (NV <= 16)
            for (unsigned u = 0; u < NV; u++)
                q8[u] = ld_glob8(q + ((size_t)t * H + lane) * HD + u * 8);
        const float wq = bf2f(w[(size_t)t * H + lane]);

        /* FOUR ENTRIES IN FLIGHT. Each wave owns barely two of the NC entries
         * (4096 entries over 304x8 waves), and one entry is 16 dependent
         * `bf16v8` loads of a row that has to come from HBM — so the op sat at
         * ~1.5 TF/s with nothing to overlap the latency against. Issuing four
         * INDEPENDENT rows before reducing any of them is the same fix the
         * attention gather needed, and for the same reason: the arithmetic was
         * never the cost. The reduction chain and the result are unchanged. */
        const unsigned CU4 = 4u;
        unsigned c = wid;
        for (; c + 3u * wstride < NC; c += CU4 * wstride) {
            float d[CU4];
#pragma unroll
            for (unsigned u = 0; u < CU4; u++) d[u] = 0.0f;
            if (NV <= 16 && (HD & 7u) == 0) {
                for (unsigned e = 0; e < NV; e++) {
                    bf16v8 kk[CU4];
#pragma unroll
                    for (unsigned u = 0; u < CU4; u++)
                        kk[u] = ld_glob8(ckv + (size_t)(c + u * wstride) * HD + e * 8);
#pragma unroll
                    for (unsigned u = 0; u < CU4; u++) d[u] = dot8(q8[e], kk[u], d[u]);
                }
            } else {
                const bf16* qh = q + ((size_t)t * H + lane) * HD;
                for (unsigned u = 0; u < CU4; u++) {
                    const bf16* kc = ckv + (size_t)(c + u * wstride) * HD;
                    for (unsigned e = 0; e < HD; e++) d[u] += bf2f(qh[e]) * bf2f(kc[e]);
                }
            }
#pragma unroll
            for (unsigned u = 0; u < CU4; u++) {
                float part = fmaxf(d[u], 0.0f) * wq; /* relu INSIDE the head sum */
                for (int o = 32; o; o >>= 1) part += __shfl_xor(part, o);
                if (lane == 0) score[(size_t)t * NC + c + u * wstride] = part;
            }
        }
        for (; c < NC; c += wstride) {
            const bf16* kc = ckv + (size_t)c * HD;
            float d = 0.0f;
            if (NV <= 16 && (HD & 7u) == 0) {
                for (unsigned u = 0; u < NV; u++) d = dot8(q8[u], ld_glob8(kc + u * 8), d);
            } else {
                const bf16* qh = q + ((size_t)t * H + lane) * HD;
                for (unsigned e = 0; e < HD; e++) d += bf2f(qh[e]) * bf2f(kc[e]);
            }
            float part = fmaxf(d, 0.0f) * wq;
            for (int o = 32; o; o >>= 1) part += __shfl_xor(part, o);
            if (lane == 0) score[(size_t)t * NC + c] = part;
        }
    }
}

/* Top-k selection over the scores, with the prefill causal mask and the KV-ring
 * offset the layer's index list needs.
 *
 * `idx` [T, K] i32 — the selected compressed positions, `+ offset`, or -1.
 *
 * `causal_ratio != 0` selects the PREFILL rule: query `t` may only see
 * compressed entries strictly below `(t + 1) / ratio`, because entry `c` pools
 * tokens `[c*ratio, (c+1)*ratio)` and a query must not see its own window's
 * future. Decode passes 0 and every entry is legal.
 *
 * Fewer legal entries than `K` is normal near the start of a sequence: the tail
 * of `idx` is -1, which `d_v4_sparse_attn` treats as inert.
 *
 * PERFORMANCE STATUS: K serial passes of a block-wide argmax — correct, and
 * O(K*NC). The reference calls `torch.topk`. This is a gate implementation. */
__device__ void d_v4_index_topk(int* __restrict__ idx, const float* __restrict__ score, unsigned T,
                                unsigned NC, unsigned K, unsigned causal_ratio, int offset,
                                unsigned slice, unsigned nblk, float* __restrict__ lds) {
    /* EXACT top-k by RADIX SELECT on a monotone key, four 8-bit passes.
     *
     * The first version of this op ran `K` sequential passes of a block-wide
     * argmax — O(K*NC), correct, and its own comment said it was a gate
     * implementation. Measured in a whole decode step it was 34.8 ms of 51.3,
     * i.e. 68% of the token, because K is 512 and every pass is a full
     * log-reduction over NC with a barrier. Four passes over a 256-bin
     * histogram replaces all of it and is still exact — this is the same shape
     * as `d_index_select_pf` (op 118), the selector the DSA path already uses.
     *
     * The key: a float's IEEE bits are monotone in the float for positives, and
     * reversed for negatives, so `bits ^ (bits>>31 ? ~0 : 0x80000000)` gives an
     * unsigned that orders exactly as the float does. Scores here can be
     * negative (the head weights are signed), so the negative half matters.
     *
     * Ties: entries equal to the threshold are emitted in ASCENDING INDEX until
     * K is full, which is the lower-index tie-break the previous version had and
     * which TP correctness depends on — ranks that break ties differently
     * select different sets and decode different tokens. */
    unsigned* hist = (unsigned*)lds;      /* [256] */
    unsigned* nabove = hist + 256;        /* [1]   */
    unsigned* emitted = nabove + 1;       /* [1]   */

    for (unsigned t = slice; t < T; t += nblk) {
        const unsigned lim = causal_ratio ? ((t + 1u) / causal_ratio) : NC;
        const unsigned n = lim < NC ? lim : NC;
        const float* sc = score + (size_t)t * NC;

        /* Fewer legal entries than K: emit them all, ascending, then pad. */
        if (n <= K) {
            for (unsigned c = threadIdx.x; c < K; c += PLOW_THREADS)
                idx[(size_t)t * K + c] = (c < n) ? (int)c + offset : -1;
            __syncthreads();
            continue;
        }

        unsigned prefix = 0;   /* key bits fixed so far, high to low */
        unsigned rank = K;     /* how many of the top-K remain to place */
        for (int sh = 24; sh >= 0; sh -= 8) {
            for (unsigned b = threadIdx.x; b < 256u; b += PLOW_THREADS) hist[b] = 0u;
            __syncthreads();
            for (unsigned c = threadIdx.x; c < n; c += PLOW_THREADS) {
                unsigned b;
                __builtin_memcpy(&b, &sc[c], 4);
                const unsigned key = b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
                if (sh == 24 || (key >> (unsigned)(sh + 8)) == (prefix >> (unsigned)(sh + 8)))
                    atomicAdd(&hist[(key >> (unsigned)sh) & 0xffu], 1u);
            }
            __syncthreads();
            /* Find the bin that straddles `rank` by a PARALLEL SUFFIX SCAN over
             * the 256 bins, not by walking them on one thread. The serial walk
             * is 256 dependent LDS reads and it runs once per pass; at four
             * passes that was the dominant cost of the whole op once the
             * histogram itself was parallel — the selector spent its time
             * deciding, not counting. Eight scan steps replace 256 reads. */
            unsigned* suf = hist + 256 + 2 + PLOW_V4_TIE_MAX; /* [256] scratch */
            /* Staging the KEYS in LDS as well was tried and measured as a
             * no-op (7.09 -> 7.23 ms/token, i.e. noise), so the five passes
             * read the scores from HBM. Recorded because it is the obvious
             * next idea and it does not pay. */
            if (threadIdx.x < 256u) suf[threadIdx.x] = hist[threadIdx.x];
            __syncthreads();
            for (unsigned d = 1; d < 256u; d <<= 1) {
                unsigned v = 0;
                if (threadIdx.x < 256u && threadIdx.x + d < 256u) v = suf[threadIdx.x + d];
                __syncthreads();
                if (threadIdx.x < 256u) suf[threadIdx.x] += v;
                __syncthreads();
            }
            /* `suf[b]` is now the count of entries in bins >= b. The digit is
             * the largest b with `suf[b] >= rank`, and the entries strictly
             * above it are `suf[b+1]`. */
            if (threadIdx.x < 256u) {
                const unsigned above = (threadIdx.x + 1u < 256u) ? suf[threadIdx.x + 1u] : 0u;
                if (suf[threadIdx.x] >= rank && above < rank) {
                    hist[0] = threadIdx.x;
                    nabove[0] = above;
                }
            }
            __syncthreads();
            const unsigned digit = hist[0];
            const unsigned above = nabove[0];
            prefix |= digit << (unsigned)sh;
            rank -= above;
            __syncthreads();
        }
        /* `prefix` is now the exact key of the K-th largest entry. */
        const unsigned thr = prefix;

        /* EMISSION IS PARALLEL. Doing it on one thread — even after the radix
         * select made the SEARCH cheap — left 649 us per call, because finding
         * the survivors is still a scan of every entry. Both scans below are
         * grid-strided; only the tie fix-up is serial, over a compacted list
         * that is normally one element long.
         *
         * Order does not matter and membership does: the index list feeds
         * `d_v4_sparse_attn`, which sums under an online softmax and is
         * invariant to the order of its entries. The TIE-BREAK does matter,
         * because two TP ranks that keep different entries at equal score
         * decode different tokens — so ties are resolved by lowest index, as
         * `torch.topk` does. */
        unsigned* ntie = emitted + 1;  /* [1]        */
        unsigned* tie = ntie + 1;      /* [PLOW_V4_TIE_MAX] */
        if (threadIdx.x == 0) {
            emitted[0] = 0u;
            ntie[0] = 0u;
        }
        __syncthreads();
        for (unsigned c = threadIdx.x; c < n; c += PLOW_THREADS) {
            unsigned b;
            __builtin_memcpy(&b, &sc[c], 4);
            const unsigned key = b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
            if (key > thr) {
                const unsigned slot = atomicAdd(&emitted[0], 1u);
                if (slot < K) idx[(size_t)t * K + slot] = (int)c + offset;
            } else if (key == thr) {
                const unsigned u = atomicAdd(&ntie[0], 1u);
                if (u < PLOW_V4_TIE_MAX) tie[u] = c;
            }
        }
        __syncthreads();
        /* Fill the remainder from the ties, lowest index first. `need` is K
         * minus the strictly-greater count, which is 1 unless the scores
         * genuinely collide; the loop is over the compacted list, not over n.
         * A tie count past the cap cannot happen without `PLOW_V4_TIE_MAX`
         * equal scores, and it degrades to taking the first ones seen rather
         * than the lowest — recorded rather than silently assumed away. */
        {
            const unsigned have = emitted[0] < K ? emitted[0] : K;
            const unsigned nt = ntie[0] < PLOW_V4_TIE_MAX ? ntie[0] : PLOW_V4_TIE_MAX;
            const unsigned need = K - have;
            /* RANK EACH TIE IN PARALLEL rather than picking the minimum `need`
             * times. The serial spelling is O(need * nt) and that is not a
             * corner case: when many entries share a score — which happens
             * whenever the compressed history is still sparsely filled — `nt`
             * is the whole candidate set and `need` is K. Measured on
             * all-equal scores it was 37 ms for ONE call, against 649 us for
             * the same op on distinct ones. Indices are unique, so a tie's rank
             * among ties is exactly the count of ties below it, and that is
             * O(nt^2 / threads) with no ordering assumption about the order
             * atomics happened to append in. */
            for (unsigned u = threadIdx.x; u < nt; u += PLOW_THREADS) {
                unsigned r = 0;
                for (unsigned v = 0; v < nt; v++) r += (unsigned)(tie[v] < tie[u]);
                if (r < need) idx[(size_t)t * K + have + r] = (int)tie[u] + offset;
            }
            __syncthreads();
            if (threadIdx.x == 0) emitted[0] = have + (need < nt ? need : nt);
        }
        __syncthreads();
        for (unsigned c = threadIdx.x; c < K; c += PLOW_THREADS)
            if (c >= emitted[0]) idx[(size_t)t * K + c] = -1;
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------
 * Hyper-connection reduce, SPLIT across the grid.
 * ---------------------------------------------------------------------------
 *
 * `d_hc_reduce` above does the whole reduce in one workgroup, which is correct
 * and is what the oracle gates. It is also, at decode, the single most
 * expensive thing in the model: `hc_fn` is 1.5 MiB per sub-layer and there are
 * 86 of them, so a token streams 135 MiB through ONE CU. Measured 238 us per
 * site against a 0.39 us bandwidth roof — 0.2%, and 20.5 ms/token, an order of
 * magnitude more than all 43 layers of attention put together.
 *
 * The obstacle is a genuine global dependency: `pre` needs all `MIX` mix values,
 * each of which is a dot over the whole `hc*D` flattened stream, so no part of
 * the output can be produced before the whole reduction is done. One workgroup
 * is the only way to do that in ONE packet.
 *
 * So it becomes two, ordered by the counter DAG exactly as split-K GEMV already
 * is in this tree:
 *
 *   d_v4_hc_dot   grid-parallel partial reduction of `sum(x^2)` and the MIX dot
 *                 products into a `[T][1+MIX]` f32 scratch, one atomicAdd per
 *                 value per block. This is the part that moves the 1.5 MiB.
 *   d_v4_hc_mix   reads the scratch, does the rsqrt / Sinkhorn split (a few
 *                 hundred flops over a 4x4 — recomputed per block rather than
 *                 broadcast, which is cheaper than another dependency), and
 *                 writes `out` parallel over D.
 *
 * The scratch MUST be zeroed before `d_v4_hc_dot` runs.
 *
 * `hc_fn` is fp32 IN THE CHECKPOINT, which is why this moves 1.5 MiB and not
 * 768 KiB. Storing it bf16 would halve the dominant cost of the model at decode;
 * that is a numerics question for a later pass, not a scheduling one, and it is
 * recorded here because the roofline says it is worth asking. */
/* Zero a hyper-connection partial. `d_v4_hc_dot` accumulates into it with one
 * atomicAdd per value per block, so it has to start at zero — its own comment
 * says so, and nothing enforced it.
 *
 * Grid-strided over the whole grid rather than done on block 0, because block 0
 * finishing early does not order the other blocks' atomics; the counter-DAG
 * orders this packet against the dot, and every block participating keeps the
 * zero a property of the PACKET rather than of one workgroup's timing. */
__device__ void d_v4_hc_zero(float* __restrict__ partial, unsigned n, unsigned slice,
                             unsigned nblk) {
    for (unsigned i = slice * PLOW_THREADS + threadIdx.x; i < n; i += nblk * PLOW_THREADS)
        partial[i] = 0.0f;
}

/* Replicate `src[t]` into every hyper-connection stream of `x[t]`.
 *
 * The alternative — leaving streams 1..HC-1 at whatever the allocator last held
 * — is not "approximately right at layer 0": `d_hc_reduce` reads all HC streams
 * on the first reduce, so uninitialised memory enters the Sinkhorn mix and then
 * every layer after it. */
__device__ void d_v4_hc_broadcast(bf16* __restrict__ x, const bf16* __restrict__ src, unsigned T,
                                  unsigned D, unsigned HC, unsigned slice, unsigned nblk) {
    const size_t work = (size_t)T * D;
    for (size_t w = (size_t)slice * PLOW_THREADS + threadIdx.x; w < work;
         w += (size_t)nblk * PLOW_THREADS) {
        const unsigned t = (unsigned)(w / D), d = (unsigned)(w - (size_t)t * D);
        const bf16 v = src[(size_t)t * D + d];
        bf16* xt = x + (size_t)t * HC * D;
        for (unsigned k = 0; k < HC; k++) st_act1(&xt[(size_t)k * D + d], v);
    }
}

__device__ void d_v4_hc_dot(float* __restrict__ partial, const bf16* __restrict__ x,
                            const float* __restrict__ hc_fn, unsigned T, unsigned D, unsigned HC,
                            unsigned slice, unsigned nblk, float* __restrict__ lds) {
    const unsigned MIX = (2u + HC) * HC;
    const unsigned N = HC * D;
    if (HC > PLOW_V4_HC_MAX || HC == 0) return;
    const unsigned nwave = PLOW_THREADS >> 6;

    /* THE WEIGHT IS SWEPT LINEARLY, and that is the whole point of this shape.
     *
     * `hc_fn` is `[MIX][N]`. The obvious loop — own an `i`, walk `m` — reads
     * MIX values `N * 4` bytes apart, i.e. 24 separate cache lines per thread
     * 64 KiB apart. That was measured: it left 272 of 304 blocks with no work
     * at all (the grid stride exceeded `N`) and ran the rest uncoalesced, for
     * 10 GB/s against a 4164 GB/s bound.
     *
     * So the grid is strided over the FLATTENED `[MIX * N]` weight instead.
     * Consecutive lanes read consecutive floats of `hc_fn`, which is one
     * coalesced stream, and the `(m, i)` decode is two integer ops. Each thread
     * may land on several `m`, so its products go into an LDS accumulator per
     * mix row rather than a register per row. */
    /* TWO DECOMPOSITIONS, because decode and prefill want opposite ones.
     *
     * At decode there is ONE token, so the only parallelism is the `MIX * N`
     * projection and every block must take a slice of it. At prefill there are
     * thousands of tokens and the loop below is per token, so the decode form
     * runs `T` sequential grid passes. Splitting TOKENS across blocks instead
     * turns that into `T / nblk` passes, and is the better shape for a batched
     * decode of a few dozen tokens.
     *
     * IT DOES NOT RESCUE PREFILL, AND THE MEASUREMENT SAYS WHY. At T=4096 the
     * two decompositions are 50.3 ms and 48.4 ms — indistinguishable, both 0.1%
     * of roof. The cost is not scheduling: a token-parallel block re-reads the
     * whole 1.5 MiB `hc_fn` for every token it owns, so total weight traffic is
     * `T * 1.5 MiB` = 6.1 GiB either way. Amortizing a weight across tokens is
     * tiling, i.e. a GEMM — the projection at prefill is `[T, N] x [MIX, N]^T`
     * with AI 20.6, just above the ridge point.
     *
     * So the emitter should issue `DevOp::Gemm` for the mixes at prefill and
     * this pair only at decode, where it already sits at the launch floor. Both
     * spellings are available and the choice is the emitter's; no decomposition
     * of a per-token GEMV can substitute for the GEMM. */
    float* bdot = lds; /* [1 + MIX] per block */
    const bool token_par = T >= nblk;
    const unsigned t0 = token_par ? slice : 0;
    const unsigned tstep = token_par ? nblk : 1;
    for (unsigned t = t0; t < T; t += tstep) {
        const bf16* xt = x + (size_t)t * N;
        for (unsigned k = threadIdx.x; k < 1u + MIX; k += PLOW_THREADS) bdot[k] = 0.0f;
        __syncthreads();

        const size_t total = (size_t)MIX * N;
        /* Token-parallel blocks each sweep the WHOLE projection for their own
         * tokens; element-parallel blocks split it between them. */
        const size_t gstride = token_par ? PLOW_THREADS : (size_t)nblk * PLOW_THREADS;
        const size_t e0 = token_par ? threadIdx.x : (size_t)slice * PLOW_THREADS + threadIdx.x;
        const unsigned lane = threadIdx.x & 63u;
        float ss = 0.0f;
        for (size_t e = e0; e < total; e += gstride) {
            const unsigned m = (unsigned)(e / N), i = (unsigned)(e - (size_t)m * N);
            const float v = bf2f(xt[i]);
            /* `sum(x^2)` rides the m == 0 pass, which visits every i exactly once. */
            if (m == 0) ss += v * v;
            const float prod = v * hc_fn[e];
            /* A wave's 64 lanes are consecutive in `e`, so they almost always
             * share `m` — and an LDS atomic from 64 lanes to ONE address is a
             * 64-way conflict, fully serialized. That, not the bandwidth, was
             * what held this at 15 GB/s. Reduce in-wave and let one lane commit.
             * A wave that straddles a row boundary falls back to per-lane
             * atomics rather than silently summing into the wrong row. */
            if ((unsigned)((e - lane) % N) + 63u < N) {
                float s = prod;
                for (int o = 32; o; o >>= 1) s += __shfl_xor(s, o);
                if (lane == 0) atomicAdd(&bdot[1u + m], s);
            } else {
                atomicAdd(&bdot[1u + m], prod);
            }
        }
        for (int o = 32; o; o >>= 1) ss += __shfl_xor(ss, o);
        if ((threadIdx.x & 63u) == 0) atomicAdd(&bdot[0], ss);
        __syncthreads();

        /* One global atomic per value per BLOCK, not per wave or per lane. */
        if (threadIdx.x < 1u + MIX)
            atomicAdd(&partial[(size_t)t * (1u + MIX) + threadIdx.x], bdot[threadIdx.x]);
        __syncthreads();
    }
    (void)nwave;
}

/* The cheap tail: Sinkhorn split from the reduced partials, then the weighted
 * stream mix, parallel over D across the whole grid. */
__device__ void d_v4_hc_mix(bf16* __restrict__ out, float* __restrict__ mix_out,
                            const bf16* __restrict__ x, const float* __restrict__ partial,
                            const float* __restrict__ scale, const float* __restrict__ base,
                            unsigned T, unsigned D, unsigned HC, unsigned iters, float norm_eps,
                            float hc_eps, unsigned slice, unsigned nblk, float* __restrict__ lds) {
    const unsigned MIX = (2u + HC) * HC;
    const unsigned N = HC * D;
    if (HC > PLOW_V4_HC_MAX || HC == 0) return;
    float* mixes = lds;
    float* pre = mixes + MIX;
    float* post = pre + HC;
    float* comb = post + HC;

    for (unsigned t = 0; t < T; t++) {
        const float* pt = partial + (size_t)t * (1u + MIX);
        if (threadIdx.x == 0) {
            /* Registers, then one copy out — see `d_hc_split_sinkhorn`. */
            const float rs = rsqrtf(pt[0] / (float)N + norm_eps);
            for (unsigned m = 0; m < MIX; m++) mixes[m] = pt[1u + m] * rs;
            if (HC == 4)
                d_hc_split_sinkhorn_t<4>(mixes, scale, base, iters, hc_eps, pre, post, comb);
            else
                d_hc_split_sinkhorn(mixes, scale, base, HC, iters, hc_eps, pre, post, comb);
            if (slice == 0) {
                float* mo = mix_out + (size_t)t * (HC + HC * HC);
                for (unsigned j = 0; j < HC; j++) mo[j] = post[j];
                for (unsigned j = 0; j < HC * HC; j++) mo[HC + j] = comb[j];
            }
        }
        __syncthreads();
        const bf16* xt = x + (size_t)t * N;
        for (unsigned d = slice * PLOW_THREADS + threadIdx.x; d < D; d += nblk * PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned j = 0; j < HC; j++) acc += pre[j] * bf2f(xt[(size_t)j * D + d]);
            st_act1(&out[(size_t)t * D + d], f2bf(acc));
        }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------
 * KV compressor — DECODE step (the incremental form).
 * ---------------------------------------------------------------------------
 *
 * `d_v4_kv_compress` above is the prefill form: it pools whole windows out of a
 * sequence that is already there. At decode a token arrives at a time, so the
 * compressor carries a per-sequence window STATE and emits one compressed entry
 * every `ratio` steps. That state is a runtime resource exactly like the KV
 * cache, which is why it is an operand here and not a graph edge.
 *
 * THE PROJECTIONS ARE NOT DONE HERE. `wkv`/`wgate` are `[COFF*D, HID]` and at
 * decode that is an ordinary GEMV, which this tree already has a tuned arm for.
 * Doing it inside this op would put 4.2M MAC per layer on whatever grid the
 * state update wants — the same mistake the first hyper-connection reduce made.
 * So the emitter issues two `Gemv` packets and hands the results in.
 *
 * `kv_p`     [COFF*D] f32   wkv  @ x   for this token
 * `sc_p`     [COFF*D] f32   wgate@ x   for this token
 * `kv_state` [COFF*ratio, COFF*D] f32  persistent window state
 * `sc_state` [COFF*ratio, COFF*D] f32  persistent, holds scores + ape
 * `out`      [D] bf16       the compressed entry, written ONLY on an emit step
 *
 * TRANSCRIBED FROM `Compressor.forward` at `start_pos > 0`:
 *
 *   score += ape[start_pos % ratio]
 *   plain:    state[start_pos % ratio] = (kv, score)
 *             emit when (start_pos+1) % ratio == 0, pooling `ratio` rows
 *   overlap:  state[ratio + start_pos % ratio] = (kv, score)
 *             emit pools 2*ratio rows: the FIRST half's columns [0:D] from
 *             state rows [0,ratio), the SECOND half's columns [D:2D] from rows
 *             [ratio,2*ratio) — then SHIFTS state[:ratio] = state[ratio:]
 *
 * The shift is what makes an overlapped window see the previous one, and it is
 * the piece with no prefill counterpart: prefill has both windows in hand at
 * once. `start_pos` is patched per step by the host, as `out_row0` already is
 * at every headnorm site.
 *
 * NOT AN EMIT STEP => THIS OP WRITES NOTHING to `out`. The caller must not
 * consume `out` on those steps; the emitter knows the cadence statically from
 * `ratio` even though `start_pos` is runtime. */
__device__ void d_v4_kv_compress_step(bf16* __restrict__ out, float* __restrict__ kv_state,
                                      float* __restrict__ sc_state,
                                      const float* __restrict__ kv_p,
                                      const float* __restrict__ sc_p,
                                      const float* __restrict__ ape,
                                      const bf16* __restrict__ norm_w, unsigned D, unsigned ratio,
                                      unsigned overlap, unsigned start_pos, float eps,
                                      unsigned slice, unsigned nblk, float* __restrict__ lds) {
    if (ratio == 0) return;
    const unsigned COFF = overlap ? 2u : 1u;
    const unsigned W = COFF * D;                  /* projection width      */
    const unsigned NROW = overlap ? 2u * ratio : ratio;
    const unsigned phase = start_pos % ratio;
    const unsigned slot = overlap ? (ratio + phase) : phase;
    const bool emit = ((start_pos + 1u) % ratio) == 0u;

    /* Every block writes the same state slot with the same values — the update
     * is idempotent, so no cross-block ordering is needed for it. Only the
     * pooling below is partitioned. */
    for (unsigned i = threadIdx.x; i < W; i += PLOW_THREADS) {
        kv_state[(size_t)slot * W + i] = kv_p[i];
        sc_state[(size_t)slot * W + i] = sc_p[i] + ape[(size_t)phase * W + i];
    }
    __threadfence();
    __syncthreads();
    if (!emit) return;

    /* Pool. Each thread owns output dims; the softmax is per dim over the rows,
     * as in the prefill form. */
    float* red = lds;
    for (unsigned d = slice * PLOW_THREADS + threadIdx.x; d < D; d += nblk * PLOW_THREADS) {
        float m = -INFINITY, l = 0.0f, acc = 0.0f;
        for (unsigned r = 0; r < NROW; r++) {
            /* Overlap: rows [0,ratio) contribute their FIRST half's column d,
             * rows [ratio,2r) their SECOND half's. */
            const unsigned col = (overlap && r >= ratio) ? (D + d) : d;
            const float sc = sc_state[(size_t)r * W + col];
            const float kv = kv_state[(size_t)r * W + col];
            const float mn = fmaxf(m, sc);
            const float resc = (m == -INFINITY) ? 0.0f : __expf(m - mn);
            /* `sc_state` starts at -inf so the FIRST window of an overlapped
             * layer has no predecessor. `exp(-inf - -inf)` is NaN, not 0, and
             * that NaN is the whole difference between this and the prefill
             * form — which never evaluates those rows because it skips group 0's
             * missing predecessor outright. Measured: with this unguarded,
             * groups 1..7 matched prefill EXACTLY and group 0 was 3.2. */
            const float pe = (sc == -INFINITY) ? 0.0f : __expf(sc - mn);
            acc = acc * resc + pe * kv;
            l = l * resc + pe;
            m = mn;
        }
        red[d] = (l > 0.0f) ? acc / l : 0.0f;
    }
    __syncthreads();

    /* RMSNorm over the pooled entry, then publish. */
    float ss = 0.0f;
    for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) ss += red[d] * red[d];
    for (int o = 32; o; o >>= 1) ss += __shfl_xor(ss, o);
    float* wr = lds + D;
    if ((threadIdx.x & 63u) == 0) wr[threadIdx.x >> 6] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (unsigned w = 0; w < (PLOW_THREADS >> 6); w++) tot += wr[w];
    const float rs = rsqrtf(tot / (float)D + eps);
    for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS)
        st_act1(&out[d], f2bf(red[d] * rs * bf2f(norm_w[d])));
    __syncthreads();

    /* The overlap SHIFT: this window becomes the next one's predecessor. */
    if (overlap)
        for (unsigned i = threadIdx.x; i < (size_t)ratio * W; i += PLOW_THREADS) {
            kv_state[i] = kv_state[(size_t)ratio * W + i];
            sc_state[i] = sc_state[(size_t)ratio * W + i];
        }
    __syncthreads();
}

/* ---------------------------------------------------------------------------
 * Sparse attention, SPLIT-K across blocks — the decode shape.
 * ---------------------------------------------------------------------------
 *
 * `d_v4_sparse_attn` indexes work by (token, head), so a T=1 step launches
 * H = 64 blocks on a 304-CU part and leaves 240 CUs idle. That is not a guess:
 * `runtime/tests/v4_attn_scale_gfx942.hip` times the same kernel at T = 1..16
 * and the TOTAL is flat from 64 to 256 blocks —
 *
 *     T=1  64 blocks  39.52 us      T=4  256 blocks  41.08 us
 *     T=2 128 blocks  39.72 us      T=8  512 blocks  84.36 us
 *
 * — four times the work for the same wall clock, then linear once the grid
 * passes the CU count. The op is CU-starved at decode, not slow.
 *
 * So the key range is split `SPLIT` ways across blocks as well: the work index
 * becomes (token, head, key-chunk) and a T=1 step fills the machine. Each block
 * carries its own `(m, l, acc)` over its chunk and writes them as PARTIALS;
 * `d_v4_sparse_attn_merge` rescales them onto a common max and folds in the
 * sink. Same structure as `FlashDecode` + `FlashMerge` already in this tree.
 *
 * `opart`  [T*H*SPLIT, D] f32   per-chunk numerators
 * `mlpart` [T*H*SPLIT, 2] f32   per-chunk (m, l)
 *
 * The sink is NOT applied here — it belongs to the whole row's denominator, and
 * adding it per chunk would count it `SPLIT` times. */
__device__ void d_v4_sparse_attn_split(float* __restrict__ opart, float* __restrict__ mlpart,
                                       const bf16* __restrict__ q, const bf16* __restrict__ kv,
                                       const int* __restrict__ idx, unsigned T, unsigned H,
                                       unsigned D, unsigned TOPK, unsigned SPLIT, float scale,
                                       unsigned slice, unsigned nblk, float* __restrict__ lds) {
    if ((D & 63u) || H == 0 || SPLIT == 0) return;
    const unsigned DPL = D >> 6;
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const unsigned nwave = PLOW_THREADS >> 6;
    float* pacc = lds;
    float* pml = lds + (size_t)nwave * D;

    for (unsigned w = slice; w < T * H * SPLIT; w += nblk) {
        const unsigned t = w / (H * SPLIT);
        const unsigned r = w - t * H * SPLIT;
        const unsigned h = r / SPLIT, sp = r - h * SPLIT;
        /* Contiguous chunk so the index reads stay coalesced. */
        const unsigned per = (TOPK + SPLIT - 1u) / SPLIT;
        const unsigned lo = sp * per, hi = (lo + per) < TOPK ? (lo + per) : TOPK;
        const int* it = idx + (size_t)t * TOPK;
        const bf16* qh = q + ((size_t)t * H + h) * D;

        float qv[16], acc[16];
        bf16v8 q8;
        if (DPL == 8) q8 = ld_glob8(qh + lane * 8);
        for (unsigned e = 0; e < DPL; e++) {
            qv[e] = bf2f(qh[lane * DPL + e]);
            acc[e] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;

        for (unsigned j = lo + wave; j < hi; j += nwave) {
            const int p = it[j];
            float kvv[16];
            float part = 0.0f;
            if (p >= 0) {
                const bf16* kr = kv + (size_t)p * D;
                if (DPL == 8) {
                    const bf16v8 k8 = ld_glob8(kr + lane * 8);
                    part = dot8(q8, k8, 0.0f);
                    for (unsigned e = 0; e < 8; e++) kvv[e] = bf2f(k8[e]);
                } else {
                    for (unsigned e = 0; e < DPL; e++) {
                        kvv[e] = bf2f(kr[lane * DPL + e]);
                        part += qv[e] * kvv[e];
                    }
                }
            } else {
                for (unsigned e = 0; e < DPL; e++) kvv[e] = 0.0f;
            }
            for (int o = 32; o; o >>= 1) part += __shfl_xor(part, o);
            const float sv = (p >= 0) ? part * scale : -INFINITY;
            const float mn = fmaxf(m, sv);
            const float resc = (m == -INFINITY) ? 0.0f : __expf(m - mn);
            const float pe = (sv == -INFINITY) ? 0.0f : __expf(sv - mn);
            for (unsigned e = 0; e < DPL; e++) acc[e] = acc[e] * resc + pe * kvv[e];
            l = l * resc + pe;
            m = mn;
        }

        /* Fold the block's waves, then publish one partial for the chunk. */
        for (unsigned e = 0; e < DPL; e++) pacc[(size_t)wave * D + lane * DPL + e] = acc[e];
        if (lane == 0) {
            pml[wave * 2] = m;
            pml[wave * 2 + 1] = l;
        }
        __syncthreads();
        float mb = -INFINITY;
        for (unsigned v = 0; v < nwave; v++) mb = fmaxf(mb, pml[v * 2]);
        float lb = 0.0f;
        for (unsigned v = 0; v < nwave; v++)
            if (pml[v * 2] != -INFINITY) lb += pml[v * 2 + 1] * __expf(pml[v * 2] - mb);
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float sacc = 0.0f;
            for (unsigned v = 0; v < nwave; v++)
                if (pml[v * 2] != -INFINITY) sacc += pacc[(size_t)v * D + d] * __expf(pml[v * 2] - mb);
            opart[(size_t)w * D + d] = sacc;
        }
        if (threadIdx.x == 0) {
            mlpart[(size_t)w * 2] = mb;
            mlpart[(size_t)w * 2 + 1] = lb;
        }
        __syncthreads();
    }
}

/* Rescale the `SPLIT` partials of each row onto a common max, fold in the sink,
 * and normalize. Grid-strided over (token, head, depth) so this fills the
 * machine too. */
__device__ void d_v4_sparse_attn_merge(bf16* __restrict__ out, const float* __restrict__ opart,
                                       const float* __restrict__ mlpart,
                                       const float* __restrict__ sink, unsigned T, unsigned H,
                                       unsigned D, unsigned SPLIT, unsigned slice,
                                       unsigned nblk) {
    const size_t rows = (size_t)T * H;
    for (size_t r = slice; r < rows; r += nblk) {
        const unsigned h = (unsigned)(r % H);
        float m = -INFINITY;
        for (unsigned sp = 0; sp < SPLIT; sp++) {
            const float ms = mlpart[(r * SPLIT + sp) * 2];
            m = fmaxf(m, ms);
        }
        float l = 0.0f;
        for (unsigned sp = 0; sp < SPLIT; sp++) {
            const float ms = mlpart[(r * SPLIT + sp) * 2];
            if (ms != -INFINITY) l += mlpart[(r * SPLIT + sp) * 2 + 1] * __expf(ms - m);
        }
        /* The sink joins the DENOMINATOR once, for the whole row. */
        if (m != -INFINITY) l += __expf(sink[h] - m);
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned sp = 0; sp < SPLIT; sp++) {
                const float ms = mlpart[(r * SPLIT + sp) * 2];
                if (ms != -INFINITY)
                    acc += opart[((r * SPLIT + sp) * (size_t)D) + d] * __expf(ms - m);
            }
            st_act1(&out[r * D + d], f2bf(acc * inv));
        }
        __syncthreads();
    }
}
