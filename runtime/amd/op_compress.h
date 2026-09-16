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
/* The E4M3-scale arm floors amax at `6 * 2**-9` instead (kernel.py:160), and the value is chosen,
 * not conservative: `amax / 6` is then exactly 2**-9, e4m3's smallest subnormal, so the rounded
 * scale of an all-zero block is the smallest NONZERO one. The 2**-126 floor would round to 0 and
 * the dequant would divide by it. */
#define PLOW_CMP_FP4_FLOOR_E4M3 0.01171875f /* 6 * 2**-9 */

/* Quantization mode: value format, scale format and amax floor together, because each is one
 * reference call site rather than three independent choices.
 *
 *   FP8_POW2  V4's attention compressor -- `act_quant(..., round_scale=True)`, e4m3 values,
 *             power-of-two scale, and the only arm that leaves the rope tail unquantized.
 *   FP4_POW2  the indexer -- `fp4_act_quant(k, 32, True)`, e2m1 values, E8M0 scale.
 *   FP4_E4M3  V4.1's compressed KV -- `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)`,
 *             e2m1 values over the WHOLE latent, and a scale that is NOT a power of two.
 *             `scripts/dsv41_csa2_oracle.py` prices the scale format at 15.3% of the latent's
 *             amax and the span at 0.406 absolute, so neither is a rounding detail. */
#define PLOW_CMP_Q_FP8_POW2 0u
#define PLOW_CMP_Q_FP4_POW2 1u
#define PLOW_CMP_Q_FP4_E4M3 2u

/* `d_compress_pool` epilogue: V4 fused rope+quant into the pool, V4.1 stops after the norm. */
#define PLOW_CMP_EPI_ROPE_QUANT 0u
#define PLOW_CMP_EPI_NORM 1u

/* e2m1 nibble -> its value. `quant_fp4` (amd_common.h) produces sign in bit 3 + a 0..7 code on
 * the ladder {0, .5, 1, 1.5, 2, 3, 4, 6}; the fake quant needs the round trip, not the byte. */
__device__ __forceinline__ float cmp_dequant_fp4(unsigned n) {
    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = lut[n & 7u];
    return (n & 8u) ? -v : v;
}

/* The fake quant of kernel.py:84-91 / :156-167, for ONE block of `blk` channels held in `lds`:
 * absmax -> scale -> quantize -> dequantize -> store back as bf16. `qmode` is one of
 * `PLOW_CMP_Q_*` above. The clamp is the reference's, and it is not dead on the fp8 arm: an
 * absmax at the 1e-4 floor makes `x / s` exceed 448 for any channel above the floor.
 *
 * The mode is a runtime argument rather than a template parameter because op 185 picks it per
 * CALL -- compressed KV and index keys differ only here -- and this runs once per `blk` channels
 * against a pooling loop of `nslot * d`. */
__device__ __forceinline__ float cmp_block_scale(float amax, unsigned qmode,
                                                 float* __restrict__ inv_s) {
    const float top = qmode != PLOW_CMP_Q_FP8_POW2 ? PLOW_CMP_FP4_MAX : PLOW_FP8_E4M3_MAX;
    const float floor_ = qmode == PLOW_CMP_Q_FP8_POW2   ? 1e-4f
                         : qmode == PLOW_CMP_Q_FP4_POW2 ? PLOW_CMP_FP4_FLOOR
                                                        : PLOW_CMP_FP4_FLOOR_E4M3;
    amax = fmaxf(amax, floor_);
    float s;
    *inv_s = 0.0f;
    /* An E4M3 scale is NOT a power of two, so neither `plow_round_scale` nor
     * `PLOW_QUANT_SCALE_DIV`'s reciprocal form applies: the divide is a real divide and the
     * dequant multiply is a real multiply. */
    if (qmode == PLOW_CMP_Q_FP4_E4M3)
        s = dequant_fp8(quant_fp8(amax / top));
    else
        plow_round_scale(amax, top, &s, inv_s);
    return s;
}

/* One channel's round trip at a block scale already chosen. */
__device__ __forceinline__ float cmp_quant_rt(float x, float s, float inv_s, unsigned qmode) {
    const bool fp4 = qmode != PLOW_CMP_Q_FP8_POW2;
    const float top = fp4 ? PLOW_CMP_FP4_MAX : PLOW_FP8_E4M3_MAX;
    const float q = fminf(
        fmaxf(qmode == PLOW_CMP_Q_FP4_E4M3 ? x / s : PLOW_QUANT_SCALE_DIV(x, s, inv_s), -top),
        top);
    const float r = fp4 ? cmp_dequant_fp4(quant_fp4(q)) : dequant_fp8(quant_fp8(q));
    return bf2f(f2bf(r * s));
}

__device__ __forceinline__ void cmp_fake_quant_block(float* __restrict__ lds, unsigned c0,
                                                     unsigned blk, unsigned qmode) {
    float amax = 0.0f;
    for (unsigned i = 0; i < blk; i++) amax = fmaxf(amax, fabsf(lds[c0 + i]));
    float inv_s;
    const float s = cmp_block_scale(amax, qmode, &inv_s);
    for (unsigned i = 0; i < blk; i++)
        lds[c0 + i] = cmp_quant_rt(lds[c0 + i], s, inv_s, qmode);
}

/* The ROPED value of channel `c` of `srow`, from the two bf16 its pair needs. Interleaved
 * (GPT-J) pairs mean one thread owns both halves, so this is the entire rope for that channel:
 * no staging, no barrier, and therefore no reason for a workgroup to own a whole row. The bf16
 * round is the reference's -- `apply_rotary_emb` writes a bf16 tensor that the quant then
 * reads, so dropping it would quantize a value the reference never sees. */
__device__ __forceinline__ float cmp_rope_at(const bf16* __restrict__ srow, unsigned c,
                                             unsigned c_rope0, const float* __restrict__ cosb,
                                             const float* __restrict__ sinb, size_t tb) {
    const float x = bf2f(srow[c]);
    if (c < c_rope0) return x;
    const unsigned k = c - c_rope0, m = k >> 1;
    const float p = bf2f(srow[c_rope0 + (m << 1) + (1u - (k & 1u))]);
    const float cs = cosb[tb + m], sn = sinb[tb + m];
    return bf2f(f2bf((k & 1u) ? p * sn + x * cs : x * cs - p * sn));
}

/* ONE WORKGROUP PER COMPRESSED ENTRY, grid-strided over `n_pools`.
 *
 *   out      [*][d]            the destination cache region; row `out_base + pool` is written
 *   kv       [n_src][coff*d]   the `wkv` projection: per token (prefill) or per ring slot
 *   score    [n_src][coff*d]   the `wgate` projection, same rows, WITHOUT `ape` folded in
 *   ape      [ratio][coff*d]   the learned position-in-block gate bias, or NULL -- V4.1's
 *                              `Compressor` has no such parameter at all (`model.py:458-485`),
 *                              and a zero-filled buffer would be a tensor the emitter has to
 *                              keep zero rather than an absence the type says is an absence
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
                                const int* __restrict__ pos = nullptr,
                                unsigned epilogue = PLOW_CMP_EPI_ROPE_QUANT) {
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
                                   (ape ? ape[(size_t)r * kstr + half * d + c] : 0.0f));
            }
            float den = 0.0f, acc = 0.0f;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / ratio, r = s % ratio;
                const int row = (pos != nullptr)
                                    ? (int)s
                                    : (int)(pool * ratio + r) - (int)((coff == 2 && half == 0) ? ratio : 0);
                if (row < 0) continue;
                const float p = expf(bf2f(score[(size_t)row * kstr + half * d + c]) +
                                     (ape ? ape[(size_t)r * kstr + half * d + c] : 0.0f) - mx);
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

        /* V4.1 STOPS HERE. `Compressor.forward` returns the post-norm latent BEFORE RoPE
         * (model.py:432-434) -- "Pre-RoPE is deliberate: the indexer needs the unrotated form, so
         * Attention rotates afterwards" -- and `_compress_kv` does the rope and the quant only
         * after the indexer has read it (model.py:751-761). That tail is op 185.
         *
         * `epilogue` is a kernel-wide immediate, so this early `continue` is workgroup-uniform and
         * the barriers below stay safe, exactly as the decode gate at the top is. */
        if (epilogue == PLOW_CMP_EPI_NORM) {
            bf16* const lrow = out + (size_t)(pbase + pool) * d;
            for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS) lrow[c] = f2bf(lds[c]);
            __syncthreads(); /* lds is reused by the next pool */
            continue;
        }

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
         * precision" (model.py:510). V4.1 does neither here; it takes the branch above. ---- */
        const unsigned qspan = ROTATE ? d : (d - rd);
        const unsigned qmode = ROTATE ? PLOW_CMP_Q_FP4_POW2 : PLOW_CMP_Q_FP8_POW2;
        for (unsigned b = threadIdx.x; b < qspan / qblk; b += PLOW_THREADS)
            cmp_fake_quant_block(lds, b * qblk, qblk, qmode);
        __syncthreads();

        bf16* orow = out + (size_t)(pbase + pool) * d;
        for (unsigned c = threadIdx.x; c < d; c += PLOW_THREADS) orow[c] = f2bf(lds[c]);
        __syncthreads(); /* lds is reused by the next pool */
    }
}

/* ROPE + FAKE QUANT ON ONE COMPRESSED ROW.  [DSV41-CQUANT]
 *
 * V4.1 splits what V4 fused. `Compressor.forward` returns the latent before RoPE, the indexer
 * turns that latent into index keys, and only then does `Attention._compress_kv` rope it and
 * quantize it into the shared cache (`model.py:751-761`). This op is that tail, and ONE op
 * covers both consumers because they differ in nothing else:
 *
 *   compressed KV   fp4 e2m1, blocks of 16, E4M3 scale   `fp4_act_quant(latent, 16, True, e4m3)`
 *   index keys      fp4 e2m1, blocks of 32, E8M0 scale   `fp4_act_quant(k, 32, True)`
 *
 * Over the WHOLE row in both cases -- there is no "leave the rope tail in bf16" here, which is
 * the V4-only behaviour `d_compress_pool`'s `qspan` still has.
 *
 * `out` and `src` are SEPARATE for a reason the reference hides by running in-place: the indexer
 * reads `src`, and an in-place rope would be a write-after-read the packet's dependency order
 * does not express. One extra `[n_rows][d]` bf16 buffer buys that.
 *
 * Row `rbase + r` ropes at absolute position `(rbase + r) * ratio`: a latent stands for the FIRST
 * token of its group. Interleaved (GPT-J) pairs, as everywhere in this file. `pos` is the decode
 * override, with the same gate and the same slot arithmetic `d_compress_pool` takes, so a decode
 * packet that emits op 180 can emit this one beside it without patching immediates.
 *
 * `n_head > 1` makes a row `[n_head][d]` and ropes every head at the SAME position -- which is
 * what the indexer's queries are: `apply_rotary_emb(q[..., -rd:], freqs_cis[start:end])` over
 * `[b, s, n_local_heads, index_head_dim]` (`model.py:550-552`), one angle per TOKEN, then
 * `fp4_act_quant(q, 32, True)` over each head's whole 128. The quant block never spans two heads,
 * so the two consumers of this op differ only in `n_head`, `qblk` and `qmode`.
 *
 * No LDS and no barrier: a quant block's scale depends only on its own `qblk` channels and an
 * interleaved rope pair is two adjacent channels, so one THREAD owns a block end to end. The
 * `lds` parameter stays for the dispatch's uniform call shape and is unused.
 */
/* PLOW_CMP_VEC8=0 restores the per-channel scalar reads (the A/B control for the vector lane
 * inside d_compress_rope_quant). */
#ifndef PLOW_CMP_VEC8
#define PLOW_CMP_VEC8 1
#endif
/* PLOW_CMP_TAB4=0 restores the per-pair scalar table reads (the A/B control for the lane below). */
#ifndef PLOW_CMP_TAB4
#define PLOW_CMP_TAB4 1
#endif
__device__ void d_compress_rope_quant(bf16* __restrict__ out, const bf16* __restrict__ src,
                                      const float* __restrict__ cosb,
                                      const float* __restrict__ sinb, unsigned n_rows, unsigned d,
                                      unsigned rd, unsigned qblk, unsigned ratio,
                                      unsigned row_base, unsigned qmode, unsigned slice,
                                      unsigned nblk, float* __restrict__ lds,
                                      const int* __restrict__ pos = nullptr,
                                      unsigned n_head = 1u) {
    if (pos != nullptr && ((pos[0] + 1) % (int)ratio) != 0) return;
    const unsigned rbase = (pos != nullptr) ? ((unsigned)pos[0] / ratio) : row_base;

    const unsigned items = n_rows * n_head;
    (void)lds;
    if (d % qblk) __builtin_trap(); /* a block may not straddle the end of a row */

    /* ONE THREAD PER QUANT BLOCK, not one workgroup per row. A quant block's scale depends on
     * its own `qblk` channels and on nothing else, and `cmp_rope_at` makes a channel's rope
     * self-contained, so the whole op is embarrassingly parallel at block granularity. The
     * row-per-workgroup form this replaces left 508 of 512 lanes idle on the indexer's queries
     * (`d`=128, `qblk`=32 => FOUR blocks) and paid three barriers and four dependent round
     * trips per row: 862 rows per workgroup at 8k, measured 3.2 ms per call against a 67 MB
     * roofline. The two reads of each channel below are one HBM trip and one L1 hit. */
    const unsigned nb = d / qblk;
    const unsigned c_rope0 = d - rd; /* rd == 0 => no channel ropes */
    const size_t total = (size_t)items * nb;
    for (size_t w = (size_t)slice * PLOW_THREADS + threadIdx.x; w < total;
         w += (size_t)nblk * PLOW_THREADS) {
        const unsigned it = (unsigned)(w / nb);
        const unsigned c0 = (unsigned)(w % nb) * qblk;
        /* The POSITION is the row's, shared by all its heads; the OFFSET is the item's. */
        const unsigned r = it / n_head;
        const size_t off = (size_t)(rbase * n_head) + (size_t)it;
        const bf16* const srow = src + off * d;
        const size_t tb = (size_t)((rbase + r) * ratio) * (rd / 2);

        bf16* const orow = out + off * d;
        /* HOLD THE ROPED BLOCK IN REGISTERS. The two passes below need the same `qblk` values --
         * one to find the block's amax, one to quantize against the scale it implies -- and the
         * shipped form recomputes `cmp_rope_at` for the second: per channel that is two bf16
         * loads, two f32 table loads, two fmas and a bf16 round trip, done twice. A quant block
         * is `qblk` floats and nothing else, so it fits in registers whenever qblk is small
         * enough to be worth it; larger blocks keep the recompute.
         *
         * Bit-identical: the same values in the same order through the same fmaxf fold and the
         * same cmp_quant_rt. */
        constexpr unsigned QMAX = 32u;
        /* VECTOR LANE. The register arm above still reads CHANNEL BY CHANNEL, and `cmp_rope_at`
         * touches TWO channels per call -- its own and the rotary partner -- so a qblk=16 block
         * costs 32 scalar bf16 loads where the data is 32 CONTIGUOUS bytes. The indexer's query
         * call is 8192 x 32 heads x 128 at qblk 16, and it measured 508 us moving ~84 MB: 165 GB/s,
         * 3% of HBM, ~40x off roofline and the worst ratio in the layer.
         *
         * The partner is always inside the same 8-element vector, so one `ld_glob8` serves both
         * reads. RoPE here is INTERLEAVED (GPT-J): channel `c_rope0 + 2m` pairs with
         * `c_rope0 + 2m + 1`, adjacent. A pair therefore straddles an 8-aligned boundary only if
         * `c_rope0 + 2m == 7 (mod 8)`, which is odd and so impossible whenever `c_rope0` is even --
         * the guard below. `c0` is a multiple of `qblk` and `qblk % 8 == 0`, so each 8-group is
         * 8-aligned within the row and `srow` is `d`-strided, keeping the 16 B load aligned.
         *
         * BIT-IDENTICAL: the same two bf16 values per channel, through the same f32 fma pair, the
         * same `f2bf` round trip and the same `fmaxf` fold, in the same order. Only the LOADS
         * change -- 32 of them become 2. */
        /* ...and the TABLES are still one scalar f32 load per pair. An 8-group that lies wholly
         * inside the rope region covers exactly FOUR pairs, at consecutive `m`, so `cosb`/`sinb`
         * each want one 16 B load instead of four 4 B ones -- 8 scalar loads per group become 2.
         *
         * Alignment, and why the guard is what it is: `c_rope0 % 8 == 0` makes every 8-group lie
         * wholly on one side of the rope boundary (no straddle, so the group is all-rope or
         * none), and makes `m0 = (c8 - c_rope0)/2` a multiple of 4. `tb` is `pos * (rd/2)`, so
         * `(rd/2) % 4 == 0` makes it a multiple of 4 too, and `tb + m0` is 4-float aligned
         * against a tensor base that hsa_memory_allocate gave 4 KiB. Both hold for the indexer's
         * queries (`d`=128, `rd`=64) and the MLA compressor; anything else takes the scalar path.
         *
         * BIT-IDENTICAL: the same floats, only fewer loads. */
        const bool tab4 = PLOW_CMP_TAB4 && (c_rope0 & 7u) == 0u && ((rd >> 1) & 3u) == 0u;
        if (PLOW_CMP_VEC8 && qblk <= QMAX && (qblk & 7u) == 0u && (c_rope0 & 1u) == 0u) {
            float v[QMAX];
            float amax = 0.0f;
            for (unsigned i0 = 0; i0 < qblk; i0 += 8u) {
                const bf16v8 xv = ld_glob8(srow + c0 + i0);
                const unsigned c8 = c0 + i0;
                f32x4 cv = {0.f, 0.f, 0.f, 0.f}, sv = {0.f, 0.f, 0.f, 0.f};
                if (tab4 && c8 >= c_rope0) {
                    const size_t mb = tb + (size_t)((c8 - c_rope0) >> 1);
                    cv = *(const PLOW_GLOB f32x4*)(const PLOW_GLOB void*)(cosb + mb);
                    sv = *(const PLOW_GLOB f32x4*)(const PLOW_GLOB void*)(sinb + mb);
                }
#pragma unroll
                for (unsigned j = 0; j < 8u; j++) {
                    const unsigned c = c8 + j;
                    float val;
                    if (c < c_rope0) {
                        val = bf2f(xv[j]);
                    } else {
                        const unsigned k = c - c_rope0, m = k >> 1;
                        const float x = bf2f(xv[j]), p = bf2f(xv[j ^ 1u]);
                        const float cs = tab4 ? cv[j >> 1] : cosb[tb + m];
                        const float sn = tab4 ? sv[j >> 1] : sinb[tb + m];
                        val = bf2f(f2bf((k & 1u) ? p * sn + x * cs : x * cs - p * sn));
                    }
                    v[i0 + j] = val;
                    amax = fmaxf(amax, fabsf(val));
                }
            }
            float inv_s;
            const float s = cmp_block_scale(amax, qmode, &inv_s);
            for (unsigned i = 0; i < qblk; i++)
                orow[c0 + i] = f2bf(cmp_quant_rt(v[i], s, inv_s, qmode));
            continue;
        }
        if (qblk <= QMAX) {
            float v[QMAX];
            float amax = 0.0f;
            for (unsigned i = 0; i < qblk; i++) {
                v[i] = cmp_rope_at(srow, c0 + i, c_rope0, cosb, sinb, tb);
                amax = fmaxf(amax, fabsf(v[i]));
            }
            float inv_s;
            const float s = cmp_block_scale(amax, qmode, &inv_s);
            for (unsigned i = 0; i < qblk; i++)
                orow[c0 + i] = f2bf(cmp_quant_rt(v[i], s, inv_s, qmode));
            continue;
        }

        float amax = 0.0f;
        for (unsigned i = 0; i < qblk; i++)
            amax = fmaxf(amax, fabsf(cmp_rope_at(srow, c0 + i, c_rope0, cosb, sinb, tb)));
        float inv_s;
        const float s = cmp_block_scale(amax, qmode, &inv_s);

        for (unsigned i = 0; i < qblk; i++)
            orow[c0 + i] = f2bf(cmp_quant_rt(cmp_rope_at(srow, c0 + i, c_rope0, cosb, sinb, tb),
                                             s, inv_s, qmode));
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
 * n_tok 1); query t de-rotates by `pos0 + t`. In place: `o` is both source and destination.
 *
 * `pos`, when non-null, SUPERSEDES `pos0` and is read as `pos[0]` — the same device-resident
 * step counter `d_headnorm_rope` takes, and the same override `d_compress_pool` above takes.
 * A decode packet cannot carry the step in an immediate (the packet stream is built once and
 * replayed), so the tensor is the only form that works there; prefill passes `pos0` and null. */
__device__ void d_rope_inverse_o(bf16* __restrict__ o, const float* __restrict__ cosb,
                                 const float* __restrict__ sinb, unsigned n_tok, unsigned n_head,
                                 unsigned D, unsigned rd, unsigned pos0, unsigned slice,
                                 unsigned nblk, const int* __restrict__ pos = nullptr) {
    if (!rd) return;
    if (pos != nullptr) pos0 = (unsigned)pos[0];
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
