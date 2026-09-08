/* op_compress.h — DeepSeek-V4's learned-pooling KV compressor.  [DSV4-COMPRESS]
 *
 * Derived from the shipped reference, `inference/model.py:285-384` (`class Compressor`) and
 * `inference/kernel.py:36-200` (`act_quant` / `fp4_act_quant`), not from prose. Every layer
 * with `compress_ratio != 0` — 41 of V4's 43 — owns one of these, and each ratio-4 layer owns
 * a second, narrower one inside its indexer.
 *
 * WHAT IT IS. A compressed cache entry is a per-feature-channel softmax-weighted pool over
 * `ratio` consecutive tokens:
 *
 *     out[c] = SUM_s  kv[s][c] * softmax_s( score[s][c] + ape[s][c] )
 *
 * The softmax runs over the SLOT axis, independently for each of the `d` channels — `d`
 * separate softmaxes per pool, not one shared attention weight. `ape` is a learned additive
 * position-WITHIN-block bias on the gate logits. `kv`/`score` are the `wkv`/`wgate`
 * projections of the layer input, produced by a GEMM upstream; this op is everything after
 * them.
 *
 * WHY IT IS NOT `d_dsa_pool_compress` (op 130), which does look similar. GLM-5.3's op 130
 * pools the *indexer key* at d=128 with no post-norm, ropes BEFORE pooling, and quantizes to
 * fp8 with one scale for the whole vector. V4 pools a *dedicated projection* at d=512, applies
 * a LEARNED RMSNorm after pooling, ropes at the pool's first-token position AFTER the norm,
 * and fake-quantizes in blocks (64 for the attention compressor, 32 for the indexer's) with a
 * power-of-two per-block scale, round-tripping back to bf16. Different input, different width,
 * different epilogue, different order. Sharing the name would have been the bug.
 *
 * THE OVERLAP, and it is the whole reason `coff` exists. At `ratio == 4` the projections are
 * `2*d` wide and each pool draws from EIGHT slots, not four (`overlap_transform`,
 * `model.py:313-320`):
 *
 *     slots 0..ratio-1     <- the PREVIOUS block's tokens, through channels [0, d)
 *     slots ratio..2ratio-1 <- the CURRENT  block's tokens, through channels [d, 2d)
 *
 * so every source token contributes twice, once through each projection half, to two adjacent
 * compressed entries. Block 0 has no previous block: its first `ratio` slots carry kv = 0 and
 * score = -inf, which is self-masking under the softmax. `ape` is added BEFORE the transform
 * (`model.py:344`), so slot `s` takes `ape[s % ratio][(s / ratio) * d + c]` — one map that
 * serves both halves and, at `coff == 1`, the non-overlapped ratio-128 form unchanged.
 *
 * NO [ratio, d] TILE, at any ratio. The attention roofline
 * (`docs/amd/deepseek-v4-attention-roofline.md` 6.7) notes that a materialized `[128, 512]`
 * f32 pooling tile is 256 KiB, 4x the gfx942 LDS cap, and asks for a streaming form. It is
 * streaming here for a stronger reason than LDS: the softmax is INDEPENDENT PER CHANNEL, so a
 * thread that owns channel `c` owns that channel's entire reduction and needs no tile, no LDS
 * and no cross-lane traffic for the pooling at all — the same structure `d_dsa_pool_compress`
 * uses at pool_size 4, and it costs nothing extra at 128. LDS enters only for the RMSNorm's
 * cross-channel sum, the ROTATE arm's Hadamard, and staging the row between epilogue stages.
 *
 * PRECISION IS PART OF THE CONTRACT, not an implementation detail. The reference rounds to
 * bf16 at three points and the quantizer sees the rounded value each time:
 *   pooled (f32) -> bf16          `kv.to(dtype)`, model.py:369
 *   -> RMSNorm in f32 -> bf16     `RMSNorm.forward` ends `.to(dtype)`, model.py:202
 *   -> RoPE in f32 -> bf16        `apply_rotary_emb` computes in f32, `y.copy_(x)`, model.py:249
 *   -> fake-quant -> bf16         `act_quant(..., inplace=True)`, kernel.py:84-91
 * Skipping any of them is more accurate and NOT what the checkpoint was quantization-aware
 * trained against. Every round trip below is deliberate.
 *
 * DECODE MODE (`pos != nullptr`, `n_pools` MUST be 1) reuses op 130's gate verbatim, which is
 * the right idiom and is why this op has one: the decode packet program is re-emitted every
 * step, so "is this step a pool boundary" can only be answered from the live device-side
 * position. Every thread reads the same `pos[0]`, so the early return is workgroup-uniform and
 * safe against the barriers below. The caller stages a `coff*ratio`-slot ring (op 132's shape,
 * widened) and this op reads slot `s` directly; the ring's score rows MUST be initialized to
 * -inf so a partly-warm ring self-masks, exactly as `score_state` does at model.py:310.
 *
 * NOT IN SCOPE, and deliberately: the ragged prefill tail. `model.py:331-341` stashes the
 * `seqlen % ratio` leftover tokens into `kv_state` for the next call to finish; this op writes
 * COMPLETE pools only, so `n_pools` is the caller's `cutoff / ratio` and the tail is a stash,
 * the same division of labour ops 130/132 already have.
 */
#ifndef PLOW_OP_COMPRESS_H
#define PLOW_OP_COMPRESS_H

#include "amd_common.h"
#include "op_norm.h"     /* block_max, which op_dsa_pool.h uses but does not include */
#include "op_dsa_pool.h" /* dsa_hadamard128_stage — the ROTATE arm's transform, shared */

/* fp4 e2m1's top code and the reference's own absmax floor (`6 * 2**-126`, kernel.py:161).
 * The fp8 arm's floor is 1e-4 (kernel.py:79) and its top code is PLOW_FP8_E4M3_MAX. Both
 * scales are `exp2(ceil(log2(amax / top)))` — `fast_round_scale`, kernel.py:36-37 — so both
 * are exact powers of two and the dequant is a bare exponent add. */
#define PLOW_CMP_FP4_MAX 6.0f
#define PLOW_CMP_FP4_FLOOR 7.052966328760454e-38f /* 6 * 2**-126 */

/* e2m1 nibble -> its value. `quant_fp4` (amd_common.h) produces sign in bit 3 + a 0..7 code on
 * the ladder {0, .5, 1, 1.5, 2, 3, 4, 6}; the fake quant needs the round trip, not the byte. */
__device__ __forceinline__ float cmp_dequant_fp4(unsigned n) {
    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = lut[n & 7u];
    return (n & 8u) ? -v : v;
}

/* The fake quant of kernel.py:84-91 / :156-167, for ONE block of `blk` channels held in `lds`:
 * absmax -> power-of-two scale -> quantize -> dequantize -> store back as bf16. `ROTATE`
 * selects fp4/e2m1 (the indexer compressor, `rotate=True`) over fp8/e4m3 (the attention
 * compressor). The clamp is the reference's, and it is not dead on the fp8 arm: an absmax at
 * the 1e-4 floor makes `x / s` exceed 448 for any channel above the floor. */
template <bool ROTATE>
__device__ __forceinline__ void cmp_fake_quant_block(float* __restrict__ lds, unsigned c0,
                                                     unsigned blk) {
    const float top = ROTATE ? PLOW_CMP_FP4_MAX : PLOW_FP8_E4M3_MAX;
    const float floor_ = ROTATE ? PLOW_CMP_FP4_FLOOR : 1e-4f;
    float amax = 0.0f;
    for (unsigned i = 0; i < blk; i++) amax = fmaxf(amax, fabsf(lds[c0 + i]));
    amax = fmaxf(amax, floor_);
    const float s = exp2f(ceilf(log2f(amax / top)));
    for (unsigned i = 0; i < blk; i++) {
        const float q = fminf(fmaxf(lds[c0 + i] / s, -top), top);
        const float r = ROTATE ? cmp_dequant_fp4(quant_fp4(q)) : dequant_fp8(quant_fp8(q));
        lds[c0 + i] = bf2f(f2bf(r * s));
    }
}

/* ONE WORKGROUP PER COMPRESSED ENTRY, grid-strided over `n_pools`.
 *
 *   out      [*][d]            the destination cache region; row `out_base + pool` is written
 *   kv       [n_src][coff*d]   the `wkv` projection: per token (prefill) or per ring slot
 *   score    [n_src][coff*d]   the `wgate` projection, same rows, WITHOUT `ape` folded in
 *   ape      [ratio][coff*d]   the learned position-in-block gate bias
 *   gamma    [d]               the post-pool RMSNorm gain
 *   cosb/sinb[*][rd/2]         interleaved (GPT-J) RoPE tables; may be null when rd == 0
 *   out_base                   pool index of `pool == 0`; the chunk/append base, op 130's
 *                              `chunk_base / pool_size` one step earlier in the arithmetic
 *   pos                        decode only; gates on `(pos[0] + 1) % ratio == 0` and supplies
 *                              the output slot and the RoPE position itself
 *
 * `lds` must hold max(d, PLOW_THREADS) floats — `d` to stage the row across epilogue stages,
 * PLOW_THREADS because `dsa_hadamard128_stage` indexes it by raw `threadIdx.x` for every
 * thread, not only the 128 that carry data. `part` is the usual PLOW_WAVES reduction buffer. */
template <bool ROTATE>
__device__ void d_compress_pool(bf16* __restrict__ out, const bf16* __restrict__ kv,
                                const bf16* __restrict__ score, const float* __restrict__ ape,
                                const bf16* __restrict__ gamma, const float* __restrict__ cosb,
                                const float* __restrict__ sinb, unsigned n_pools,
                                unsigned ratio, unsigned coff, unsigned d, unsigned rd,
                                unsigned qblk, float eps, unsigned out_base, unsigned slice,
                                unsigned nblk, float* __restrict__ lds, float* __restrict__ part,
                                const int* __restrict__ pos = nullptr) {
    /* Workgroup-uniform: every thread reads the same pos[0], so a divergent return could not
     * happen and the barriers below stay safe. */
    if (pos != nullptr && ((pos[0] + 1) % (int)ratio) != 0) return;
    const unsigned pbase = (pos != nullptr) ? ((unsigned)pos[0] / ratio) : out_base;

    const unsigned nslot = coff * ratio;
    const unsigned kstr = coff * d;

    for (unsigned pool = slice; pool < n_pools; pool += nblk) {
        /* ---- pool: per-channel softmax over the slots, streaming, no tile, no LDS ---- */
        for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS) {
            float mx = -3.0e38f;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / ratio, r = s % ratio;
                /* DECODE reads the ring row `s` as it stands; PREFILL resolves the slot to a
                 * token, and the overlap half of block 0 resolves BELOW zero -- the -inf that
                 * `overlap_transform`'s fill value produces (model.py:319). */
                const int row = (pos != nullptr)
                                    ? (int)s
                                    : (int)(pool * ratio + r) - (int)((coff == 2 && half == 0) ? ratio : 0);
                if (row < 0) continue;
                mx = fmaxf(mx, bf2f(score[(size_t)row * kstr + half * d + c]) +
                                   ape[(size_t)r * kstr + half * d + c]);
            }
            float den = 0.0f, acc = 0.0f;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / ratio, r = s % ratio;
                const int row = (pos != nullptr)
                                    ? (int)s
                                    : (int)(pool * ratio + r) - (int)((coff == 2 && half == 0) ? ratio : 0);
                if (row < 0) continue;
                const float p = expf(bf2f(score[(size_t)row * kstr + half * d + c]) +
                                     ape[(size_t)r * kstr + half * d + c] - mx);
                den += p;
                acc += bf2f(kv[(size_t)row * kstr + half * d + c]) * p;
            }
            /* `kv.to(dtype)` (model.py:369): the RMSNorm below sees the bf16 value, not the
             * f32 accumulator. */
            lds[c] = bf2f(f2bf(den > 0.0f ? acc / den : 0.0f));
        }
        __syncthreads();

        /* ---- learned RMSNorm over the d channels; the one cross-channel reduction ---- */
        float sq = 0.0f;
        for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS) sq += lds[c] * lds[c];
        const float inv = rsqrtf(block_sum(sq, part) / (float)d + eps);
        for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS)
            lds[c] = bf2f(f2bf(lds[c] * inv * bf2f(gamma[c])));
        __syncthreads();

        /* ---- interleaved (GPT-J) RoPE on the last rd channels, at the pool's FIRST token.
         * Prefill: `freqs_cis[:cutoff:ratio]`, i.e. absolute position (pbase + pool) * ratio.
         * Decode: `freqs_cis[start_pos + 1 - ratio]`, which at a boundary step
         * `start_pos = (j+1)*ratio - 1` is the same j*ratio. One expression. ---- */
        if (rd) {
            const size_t tb = (size_t)((pbase + pool) * ratio) * (rd / 2);
            const unsigned c0 = d - rd;
            /* Pairs are adjacent, so one thread owns both halves and no shuffle is needed --
             * unlike d_headnorm_rope, whose lane-strided layout forces the XOR-1 shuffle. */
            for (unsigned i = threadIdx.x; i < rd / 2; i += PLOW_THREADS) {
                const float x0 = lds[c0 + 2 * i], x1 = lds[c0 + 2 * i + 1];
                const float cs = cosb[tb + i], sn = sinb[tb + i];
                lds[c0 + 2 * i] = bf2f(f2bf(x0 * cs - x1 * sn));
                lds[c0 + 2 * i + 1] = bf2f(f2bf(x0 * sn + x1 * cs));
            }
            __syncthreads();
        }

        /* ---- ROTATE (the indexer compressor): Hadamard-128 over ALL d, then fp4. The
         * transform is fixed-width and the emitter must assert d == 128. Accumulated in f32
         * with ONE bf16 round trip after the 1/sqrt(128), matching d_dsa_pool_compress's
         * choice rather than guessing at fast_hadamard_transform's internal width. ---- */
        if constexpr (ROTATE) {
            float x = (threadIdx.x < d) ? lds[threadIdx.x] : 0.0f;
            __syncthreads(); /* every lds[c] read before the butterfly overwrites it */
#pragma unroll
            for (unsigned stage = 0, stride = 1; stage < 7; stage++, stride <<= 1)
                x = dsa_hadamard128_stage(x, threadIdx.x, stride, lds);
            x *= 0.08838834764831845f; /* 1/sqrt(128), the reference's own scale */
            if (threadIdx.x < d) lds[threadIdx.x] = bf2f(f2bf(x));
            __syncthreads();
        }

        /* ---- fake quant: blocks of `qblk` channels, one thread per block. ROTATE covers all
         * d; the attention arm covers [0, d-rd) only -- "rope dims stay bf16 for positional
         * precision" (model.py:510). ---- */
        const unsigned qspan = ROTATE ? d : (d - rd);
        for (unsigned b = threadIdx.x; b < qspan / qblk; b += PLOW_THREADS)
            cmp_fake_quant_block<ROTATE>(lds, b * qblk, qblk);
        __syncthreads();

        bf16* orow = out + (size_t)(pbase + pool) * d;
        for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS) orow[c] = f2bf(lds[c]);
        __syncthreads(); /* lds is reused by the next pool */
    }
}

/* INVERSE RoPE ON THE ATTENTION OUTPUT.  [DSV4-IROPE]
 *
 * `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)` (model.py:539), and it is
 * STRUCTURAL, not cosmetic. `o = SUM_j p_j * kv_j` mixes cached rows whose rope dims were each
 * rotated by their OWN position, so de-rotating by the QUERY's position i leaves
 * `SUM_j p_j R(j-i) kv_j^rope` -- a position-INDEPENDENT latent that the fixed `wo_a` can
 * consume. An implementation that skips it builds a plausible wrong model.
 *
 * Nothing in plow applies a conjugate rotation to an attention output: `HeadNormRope` and its
 * fp8 twin rotate KEYS and QUERIES on the way IN, and neither has an inverse arm or an output
 * layout to write. The conjugate is just `sin -> -sin`, so this is one elementwise pass over
 * the last `rd` of each `[token][head][D]` row -- 64 of 512 lanes touched.
 *
 * Interleaved (GPT-J) pairs, matching `view_as_complex(x.unflatten(-1, (-1,2)))`
 * (model.py:239): pair m is (2m, 2m+1) and the table index is m, NOT the half-split (i, i+H/2)
 * that d_headnorm_rope's default arm uses. Adjacent pairs mean one thread owns both halves and
 * no cross-lane shuffle is needed, unlike d_headnorm_rope's lane-strided layout.
 *
 * `pos0` is the absolute position of token 0 of this call (decode passes the step position and
 * n_tok 1); query t de-rotates by `pos0 + t`. In place: `o` is both source and destination. */
__device__ void d_rope_inverse_o(bf16* __restrict__ o, const float* __restrict__ cosb,
                                 const float* __restrict__ sinb, unsigned n_tok, unsigned n_head,
                                 unsigned D, unsigned rd, unsigned pos0, unsigned slice,
                                 unsigned nblk) {
    if (!rd) return;
    const unsigned h2 = rd / 2, c0 = D - rd;
    const size_t rows = (size_t)n_tok * n_head;
    for (size_t r = (size_t)slice * PLOW_THREADS + threadIdx.x; r < rows * h2;
         r += (size_t)nblk * PLOW_THREADS) {
        const size_t row = r / h2;
        const unsigned m = (unsigned)(r % h2);
        const unsigned t = (unsigned)(row / n_head);
        const size_t p = (size_t)(pos0 + t) * h2 + m;
        bf16* v = o + row * D + c0 + 2 * m;
        const float x0 = bf2f(v[0]), x1 = bf2f(v[1]);
        const float c = cosb[p], sn = -sinb[p]; /* conjugate: the whole of `inverse=True` */
        v[0] = f2bf(x0 * c - x1 * sn);
        v[1] = f2bf(x0 * sn + x1 * c);
    }
}

#endif /* PLOW_OP_COMPRESS_H */
