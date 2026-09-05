/* op_elementwise.cuh — bandwidth-bound pointwise ops for sm_120 (warp32).
 *
 * Ported from runtime/amd/op_elementwise.h. Memory-bound throughout, so these are flat,
 * fully-coalesced strided loops over 16-byte accesses; there is nothing to tile. The AMD
 * as_glob()/buffer-descriptor machinery has no NVIDIA analogue and is simply dropped —
 * a `__nv_bfloat16*` here is already a generic pointer the compiler turns into ld.global.
 */
#pragma once
#include "sm120_common.cuh"

/* Embedding gather + scale.
 *
 * `scale` is 1.0 for Qwen3. (Gemma passes the BF16-ROUNDED sqrt(hidden) — 73.5, not
 * 73.3212 — because HF downcasts the normalizer to the weight dtype before multiplying and
 * the rounding is observable in the logits. Do not "fix" that to the exact sqrt.)
 *
 * TRAP, from the operand contract: t2 = in.ids has tensor handle 0. Handle 0 is a VALID
 * tensor; the only absent sentinel is PLOW_TENSOR_NONE (0xFFFF). Treating 0 as absent
 * makes EMBED read a null pointer. */
static __device__ void d_embed(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                        const int* __restrict__ ids, unsigned ntok, unsigned hidden, float scale,
                        unsigned slice, unsigned nblk) {
    for (unsigned t = slice; t < ntok; t += nblk) {
        const size_t src = (size_t)ids[t] * hidden, dst = (size_t)t * hidden;
        if ((hidden & 7u) == 0) {
            for (unsigned i = threadIdx.x * 8; i < hidden; i += PLOW_NV_THREADS * 8) {
                const bf16v8 v = ld_glob8(table + src + i);
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    o.x[j] = __float2bfloat16(__bfloat162float(v.x[j]) * scale);
                st_glob8(out + dst + i, o);
            }
        } else {
            for (unsigned i = threadIdx.x; i < hidden; i += PLOW_NV_THREADS)
                out[dst + i] = __float2bfloat16(__bfloat162float(table[src + i]) * scale);
        }
    }
}

/* out = (a + b) * scale. Prefill-only on Qwen (decode fuses this into ADD_NORM).
 * i0=n is the FLAT element count (t*hidden), not a row count. */
static __device__ void d_residual(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ a,
                           const __nv_bfloat16* __restrict__ b, unsigned n, float scale,
                           unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_NV_THREADS * 8;
    for (unsigned i = (slice * PLOW_NV_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 va = ld_glob8(a + i), vb = ld_glob8(b + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++)
                vo.x[j] = __float2bfloat16(
                    (__bfloat162float(va.x[j]) + __bfloat162float(vb.x[j])) * scale);
            st_glob8(out + i, vo);
        } else {
            for (unsigned j = i; j < n; j++)
                out[j] = __float2bfloat16(
                    (__bfloat162float(a[j]) + __bfloat162float(b[j])) * scale);
        }
    }
}

/* Final logit softcapping: cap * tanh(x / cap). Gemma 4 uses cap = 30 on the lm_head output;
 * Llama/Qwen have none (the op is not emitted for them). i0=n is the FLAT element count. */
static __device__ void d_softcap(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                          unsigned n, float cap, unsigned slice, unsigned nblk) {
    const float inv = 1.0f / cap;
    const unsigned stride = nblk * PLOW_NV_THREADS * 8;
    for (unsigned i = (slice * PLOW_NV_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 v = ld_glob8(x + i);
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++)
                o.x[j] = __float2bfloat16(cap * tanhf(__bfloat162float(v.x[j]) * inv));
            st_glob8(out + i, o);
        } else {
            for (unsigned j = i; j < n; j++)
                out[j] = __float2bfloat16(cap * tanhf(__bfloat162float(x[j]) * inv));
        }
    }
}

/* Gated MLP: act(gate) * up. i1=act selects SiLU (1, Qwen) vs gelu_tanh (0, Gemma). */
static __device__ void d_glu(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
                      const __nv_bfloat16* __restrict__ up, unsigned n, unsigned act,
                      unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_NV_THREADS * 8;
    for (unsigned i = (slice * PLOW_NV_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 vg = ld_glob8(gate + i), vu = ld_glob8(up + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float g = __bfloat162float(vg.x[j]);
                float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
#if defined(PLOW_NV_GEMMA) && PLOW_NV_GEMMA && defined(PLOW_NV_GEMMA_GLU_BF16) && PLOW_NV_GEMMA_GLU_BF16
                if (act != PLOW_ACT_SILU_) a = __bfloat162float(__float2bfloat16(a));
#endif
                vo.x[j] = __float2bfloat16(a * __bfloat162float(vu.x[j]));
            }
            st_glob8(out + i, vo);
        } else {
            for (unsigned j = i; j < n; j++) {
                const float g = __bfloat162float(gate[j]);
                float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
#if defined(PLOW_NV_GEMMA) && PLOW_NV_GEMMA && defined(PLOW_NV_GEMMA_GLU_BF16) && PLOW_NV_GEMMA_GLU_BF16
                if (act != PLOW_ACT_SILU_) a = __bfloat162float(__float2bfloat16(a));
#endif
                out[j] = __float2bfloat16(a * __bfloat162float(up[j]));
            }
        }
    }
}

/* Greedy argmax, per-block partial. See amax_pack() in sm120_common.cuh for the packed-key
 * contract — it is reproduced BIT-EXACTLY from AMD, so ties break identically.
 * `part` needs no zeroing: every block writes its own slot unconditionally.
 *
 * BATCH>1 (serving pending #4): logits are [n_batch][n] and each sequence gets its OWN argmax —
 * one token per sequence, no cross-sequence bleed. `part` is [n_batch][nblk]; the packed index
 * stays within the sequence's own [0,n) vocab row. n_batch==0/1 is byte-identical (part[slice]). */
static __device__ void d_argmax(unsigned long long* __restrict__ part, const __nv_bfloat16* __restrict__ x,
                         unsigned n, unsigned n_batch, unsigned slice, unsigned nblk,
                         unsigned long long* lds) {
    const unsigned B = n_batch ? n_batch : 1u;
    for (unsigned b = 0; b < B; b++) {
        const __nv_bfloat16* xb = x + (size_t)b * n;
        unsigned long long best = 0;
        /* VECTORIZED scan: 1 LD.E.128 per 8 elements instead of 8 scalar LD.E.U16 each with its
         * own 64-bit address build (~16 -> ~7 slots/element). This changes which block scans
         * which elements (part[slice] partials shift between slots), but the ONLY consumer is
         * ARGMAX_FIN's max-fold over all slots, and max over the same global candidate set with
         * the same packed tie-break key picks the same winner — the token is unchanged. */
        const unsigned nv = n / 8;
        for (unsigned iv = slice * PLOW_NV_THREADS + threadIdx.x; iv < nv;
             iv += nblk * PLOW_NV_THREADS) {
            const bf16v8 v = ld_glob8_cs(xb + (size_t)iv * 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned long long p = amax_pack(v.x[j], iv * 8 + (unsigned)j);
                best = p > best ? p : best;
            }
        }
        /* n % 8 tail (none on any shipped vocab; kept so an unaligned n cannot over-read). */
        if (slice == 0)
            for (unsigned i = nv * 8 + threadIdx.x; i < n; i += PLOW_NV_THREADS) {
                const unsigned long long p = amax_pack(xb[i], i);
                best = p > best ? p : best;
            }
        best = block_max_u64(best, lds);
        if (threadIdx.x == 0) part[(size_t)b * nblk + slice] = best;
    }
}

/* Fold the per-block partials and write each sequence's token id where the next step's EMBED
 * reads it. BATCH>1: ids[b] gets sequence b's token; part is [n_batch][nparts]. n_batch==0/1 is
 * byte-identical (ids[0] from part[0..nparts)). */
static __device__ void d_argmax_fin(int* __restrict__ ids, const unsigned long long* __restrict__ part,
                             unsigned nparts, unsigned n_batch, unsigned slice) {
    if (slice != 0 || threadIdx.x != 0) return;
    const unsigned B = n_batch ? n_batch : 1u;
    for (unsigned b = 0; b < B; b++) {
        const unsigned long long* pb = part + (size_t)b * nparts;
        unsigned long long best = 0;
        for (unsigned i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        ids[b] = (int)~(unsigned)(best & 0xFFFFFFFFull);
    }
}
