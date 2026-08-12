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
 * PERFORMANCE STATUS: this is a CORRECTNESS reference, not a tuned arm. Lanes
 * own dims and keys are consumed one at a time, so every key costs a six-shuffle
 * wave reduction against 8 useful MACs. That is the right shape to prove the
 * numerics and the wrong shape to ship: Stage 4 replaces the score reduction
 * with an MFMA tile over a key block, exactly as the tilelang reference does
 * (`block = 64`, `T.gemm`). No performance claim is made here and none has been
 * measured. */
__device__ void d_v4_sparse_attn(bf16* __restrict__ out, const bf16* __restrict__ q,
                                 const bf16* __restrict__ kv, const int* __restrict__ idx,
                                 const float* __restrict__ sink, unsigned T, unsigned H,
                                 unsigned D, unsigned TOPK, float scale, unsigned slice,
                                 unsigned nblk) {
    /* One lane per 64th of the head dim. A D that is not a multiple of 64 would
     * leave a ragged tail this layout cannot address; poison rather than drop
     * the tail silently. V4 uses D=512 (attention) and D=128 (indexer). */
    if ((D & 63u) || H == 0) {
        for (unsigned t = slice; t < T; t += nblk)
            for (unsigned i = threadIdx.x; i < H * D; i += PLOW_THREADS)
                st_act1(&out[(size_t)t * H * D + i], (bf16)0x7fc1u);
        return;
    }
    const unsigned DPL = D >> 6; /* dims per lane */
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const unsigned nwave = PLOW_THREADS >> 6;

    for (unsigned t = slice; t < T; t += nblk) {
        const int* it = idx + (size_t)t * TOPK;
        /* Waves partition the heads; nothing is shared between them. */
        for (unsigned h = wave; h < H; h += nwave) {
            const bf16* qh = q + ((size_t)t * H + h) * D;
            float qv[16], acc[16]; /* DPL <= 16 given D <= 1024 */
            for (unsigned e = 0; e < DPL; e++) {
                qv[e] = bf2f(qh[lane * DPL + e]);
                acc[e] = 0.0f;
            }
            float m = -INFINITY, l = 0.0f;

            for (unsigned j = 0; j < TOPK; j++) {
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
            /* The sink joins the DENOMINATOR only, at the final max, after the
             * rescale chain has finished. */
            if (m != -INFINITY) l += __expf(sink[h] - m);
            const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            bf16* oh = out + ((size_t)t * H + h) * D;
            for (unsigned e = 0; e < DPL; e++)
                st_act1(&oh[lane * DPL + e], f2bf(acc[e] * inv));
        }
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
