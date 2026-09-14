/* op_engram.h — DeepSeek-V4.1's Engram conditional memory, gate + mix.  [DSV41-ENGRAM]
 *
 * Ported term for term from the shipped reference, `inference/engram.py` (the hash side) and
 * `inference/model.py:328-366` (`class Engram`), not from prose.
 *
 * WHAT ENGRAM IS. Two layers (`engram_layer_ids`, [1, 14] in the released checkpoint) carry an
 * n-gram lookup table and write it into the residual stream, gated by how well the looked-up key
 * matches that stream. The pipeline per engram layer is three stages, and only the third is new
 * math:
 *
 *   1. HASH. Each position is hashed with the `max_ngram_size - 1` tokens before it, once per
 *      (n-gram size, head) pair, into that pair's own prime-sized bucket range -- so
 *      `n_hash_cols = (max_ngram_size - 1) * n_heads` ids per token (3 * 8 = 24 here). This is
 *      INTEGER WORK OVER TOKEN IDS ALONE: it touches no activation, and its primes, multipliers
 *      and compressed-token map are all fixed at load time from the tokenizer. So it is host
 *      work that arrives as a tensor, exactly like `pos` or `row_token`, and there is no hash
 *      kernel here. (It cannot be folded into the embed op either: the lookback stops at a dead
 *      token, so a position's ids depend on the mask, not only on its own id.)
 *
 *   2. EMBED + PROJECT. The 24 ids fetch 24 fp8 rows of `engram_head_dim`, dequantized by their
 *      block scales, flattened to `n_hash_cols * head_dim` (6144) and run through `wkv`, an
 *      ordinary fp8 block-scale GEMM to `dim * (hc_mult + 1)` (25600). The table is sharded over
 *      its ROWS with an all-reduce, because it is the largest tensor in the checkpoint by a wide
 *      margin -- 384 006 168 rows of 256 fp8 is 98.3 GB, and there are two of them.
 *
 *   3. GATE + MIX -- this file. `wkv`'s output splits into `hc_mult` keys and one shared value;
 *      the gate is a normalized dot of the residual stream against the key, and the value is
 *      added into every hc copy under it.
 *
 * THE GATE, EXACTLY (model.py:352-366). Per token `t` and hc copy `c`, over `dim`:
 *
 *     rstd  = rsqrt(mean(h[c]^2) + eps) * rsqrt(mean(key[c]^2) + eps)
 *     dot   = SUM_d h[c][d] * (q_weight[c][d] * k_weight[c][d]) * key[c][d] * rstd * dim^-0.5
 *     gate  = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
 *     out[c][d] = h[c][d] + gate * value[d]
 *
 * THREE THINGS THAT LOOK LIKE DETAILS AND ARE NOT:
 *
 *  - The normalization is per (token, hc copy) over `dim`, NOT jointly over the copies. Joint
 *    normalization compiles, runs, and is a different model -- the same class of error
 *    `op_moe.h`'s situ note records for the GLU pair form.
 *  - The SIGNED SQRT before the sigmoid ("matching the training kernel", model.py:364) is not a
 *    numerical guard that can be dropped. It is what makes the gate's response to `dot`
 *    square-root-shaped, and `clamp_min` sits INSIDE it, on the absolute value, so a dot of zero
 *    gives sqrt(1e-6) with the sign of zero -- not zero.
 *  - `q_weight` and `k_weight` are only ever used as a product (the reference says so, and the
 *    checkpoint stores both as [hc_mult, dim]). They are kept as two tensors because that is what
 *    the checkpoint holds; folding them at load time is legal but is the loader's business, not
 *    this kernel's.
 *
 * `token_mask` shuts the gate for positions that take no part in an n-gram -- image spans, cached
 * as DEAD by the hash state. A masked position passes through UNTOUCHED (gate 0, not value 0),
 * which is why the mask is applied to the gate and not to the value.
 *
 * ONE WORKGROUP PER TOKEN, like `d_hyperconn_pre`: the three reductions per hc copy are over
 * `dim` and want the whole workgroup, and `hc_mult` is 4, so a copy axis across workgroups would
 * hand three quarters of them a ragged tail.
 */
#ifndef PLOW_OP_ENGRAM_H
#define PLOW_OP_ENGRAM_H

#include "amd_common.h"

/* model.py:342. A constant and not an `f` slot: `DevInst` has two, and the gate already spends
 * one on `norm_eps`. The emitter must assert the config's value matches before selecting this
 * kernel -- the same contract PLOW_HC_POST_MULT carries in op_hyperconn.h. */
#define PLOW_ENGRAM_CLAMP 1e-6f

/* op 182 — Engram gate + mix, in place on `x`.
 *
 *   x        [T][n][hidden]   the residual stream's hc copies, bf16, READ AND WRITTEN
 *   kv       [T][(n+1)*hidden] `wkv`'s output: n keys then one shared value, bf16
 *   qw, kw   [n][hidden]      the gate weights, used only as a product
 *   tmask    [T]              optional; 0 shuts the gate and the token passes through
 *
 * `kv` is ONE tensor rather than a key and a value, because the reference splits a single
 * projection output and slicing it here costs an offset instead of a second packet operand.
 */
__device__ void d_engram_gate(bf16* __restrict__ x, const bf16* __restrict__ kv,
                              const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                              const unsigned char* __restrict__ tmask, unsigned T, unsigned n,
                              unsigned hidden, float eps, unsigned slice, unsigned nblk,
                              float* __restrict__ part) {
    const float inv_dim = 1.0f / (float)hidden;
    const float dim_rsqrt = rsqrtf((float)hidden);

    for (unsigned t = slice; t < T; t += nblk) {
        bf16* xrow = x + (size_t)t * n * hidden;
        const bf16* krow = kv + (size_t)t * (size_t)(n + 1) * hidden;
        const bf16* vrow = krow + (size_t)n * hidden; /* the shared value, after the n keys */
        /* Workgroup-uniform: every thread reads the same byte, so no barrier hazard below. */
        const bool masked = (tmask != nullptr) && (tmask[t] == 0);

        for (unsigned c = 0; c < n; c++) {
            bf16* hc = xrow + (size_t)c * hidden;
            const bf16* kc = krow + (size_t)c * hidden;
            const bf16* qc = qw + (size_t)c * hidden;
            const bf16* wc = kw + (size_t)c * hidden;

            /* One strided pass feeds all three reductions: h^2, key^2 and the weighted dot.
             * Splitting them would re-read `h` and `key` from HBM twice more for nothing --
             * this loop is the whole cost of the op. */
            float ssh = 0.0f, ssk = 0.0f, dot = 0.0f;
            for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS) {
                const float hv = bf2f(hc[d]);
                const float kvv = bf2f(kc[d]);
                ssh += hv * hv;
                ssk += kvv * kvv;
                dot += hv * (bf2f(qc[d]) * bf2f(wc[d])) * kvv;
            }
            ssh = block_sum(ssh, part);
            ssk = block_sum(ssk, part);
            dot = block_sum(dot, part);

            float gate;
            if (masked) {
                gate = 0.0f;
            } else {
                const float rstd =
                    rsqrtf(ssh * inv_dim + eps) * rsqrtf(ssk * inv_dim + eps);
                const float dv = dot * rstd * dim_rsqrt;
                /* Signed sqrt with the clamp INSIDE, on the magnitude -- see the header. */
                const float mag = sqrtf(fmaxf(fabsf(dv), PLOW_ENGRAM_CLAMP));
                /* OCML `expf`, not `__expf`: this is a GATE on a residual write, and the fast
                 * exponential's error rides into the stream at every one of the two engram
                 * layers. op_moe.h's `moe_act` note records the same choice and the measurement
                 * behind it -- the fast spelling is a re-derivation to avoid, not an oversight. */
                gate = 1.0f / (1.0f + expf(-copysignf(mag, dv)));
            }

            if (gate != 0.0f) {
                for (unsigned d = threadIdx.x; d < hidden; d += PLOW_THREADS)
                    hc[d] = f2bf(bf2f(hc[d]) + gate * bf2f(vrow[d]));
            }
            /* `hc` is read by the NEXT copy's reductions only through its own slice, so no
             * barrier is needed between copies beyond the ones block_sum already issues. */
        }
    }
}

#endif /* PLOW_OP_ENGRAM_H */
