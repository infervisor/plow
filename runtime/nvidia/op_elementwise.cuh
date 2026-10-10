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
                        unsigned slice, unsigned nblk, unsigned pad = 0) {
    for (unsigned t = slice; t < ntok; t += nblk) {
        /* A multimodal row id (bit 31, MmRowsBf16) embeds as the pad token. */
        const int id = ids[t] < 0 ? (int)pad : ids[t];
        const size_t src = (size_t)id * hidden, dst = (size_t)t * hidden;
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

/* MmRowsBf16 (op 209): a row whose id has bit 31 set is replaced by its slab row, found in the
 * open-addressed `table` of (id, slab row) pairs (`cap` a power of two, id 0 = empty). A missing
 * id leaves the row as embedded. */
static __device__ void d_mm_rows(__nv_bfloat16* __restrict__ out, const unsigned* __restrict__ ids,
                                 const unsigned* __restrict__ table, const __nv_bfloat16* __restrict__ slab,
                                 unsigned rows, unsigned width, unsigned cap, unsigned slab_rows,
                                 unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned id = ids[r];
        if (!(id & 0x80000000u)) continue;
        unsigned h = id & (cap - 1u), row = 0xFFFFFFFFu;
        for (unsigned n = 0; n < cap; n++, h = (h + 1u) & (cap - 1u)) {
            const unsigned key = table[2u * h];
            if (key == id) {
                row = table[2u * h + 1u];
                break;
            }
            if (key == 0u) break;
        }
        if (row >= slab_rows) continue;
        const __nv_bfloat16* src = slab + (size_t)row * width;
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS) out[(size_t)r * width + i] = src[i];
    }
}

/* MmSpanExtent (op 210): out[r] = rows after r in r's run of span ids (bits 31 and 30), 0 for any
 * other row. Runs are one media item (<= a few hundred rows), so each row scans its own tail. */
static __device__ void d_mm_span_extent(unsigned* __restrict__ out, const unsigned* __restrict__ ids,
                                        unsigned rows, unsigned slice, unsigned nblk) {
    const unsigned span = 0xC0000000u;
    for (unsigned r = slice * PLOW_NV_THREADS + threadIdx.x; r < rows; r += nblk * PLOW_NV_THREADS) {
        unsigned n = 0;
        if ((ids[r] & span) == span)
            while (r + n + 1 < rows && (ids[r + n + 1] & span) == span) n++;
        out[r] = n;
    }
}

/* EmbedOverlayBf16 (op 179): a row is table[tokens[r]] unless overlay_index[r] names an
 * overlay row, which is BF16-rounded in (the CPU golden's plow_f2bf: round-to-nearest-even). */
static __device__ void d_embed_overlay(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                       const unsigned* __restrict__ tokens, const float* __restrict__ overlay,
                                       const unsigned* __restrict__ overlay_index, unsigned rows, unsigned width,
                                       unsigned vocab, unsigned overlay_rows, unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned sel = overlay_index[r];
        if (sel == 0xFFFFFFFFu) {
            if (tokens[r] >= vocab) { __trap(); return; }
            const __nv_bfloat16* src = table + (size_t)tokens[r] * width;
            for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS) out[(size_t)r * width + i] = src[i];
        } else {
            if (sel >= overlay_rows) { __trap(); return; }
            const float* src = overlay + (size_t)sel * width;
            for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
                out[(size_t)r * width + i] = __float2bfloat16_rn(src[i]);
        }
    }
}

/* EmbedPosBf16 (op 194): out[r] = bf16(table[tokens[r]] + pos_table[pos[r] - base[r]]). */
static __device__ void d_embed_pos(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                   const unsigned* __restrict__ tokens, const __nv_bfloat16* __restrict__ pos_table,
                                   const unsigned* __restrict__ pos, const unsigned* __restrict__ base,
                                   unsigned rows, unsigned width, unsigned vocab, unsigned pos_rows,
                                   unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned p = pos[r] - base[r];
        if (tokens[r] >= vocab || pos[r] < base[r] || p >= pos_rows) { __trap(); return; }
        const __nv_bfloat16* a = table + (size_t)tokens[r] * width;
        const __nv_bfloat16* b = pos_table + (size_t)p * width;
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
            out[(size_t)r * width + i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

/* Compact hidden-row gather — the unified token batch's terminal row selection.
 *
 * out[s][h] = x[rows[s]][h], for s in [0, S), over H features.
 *
 * NOT d_embed above: that gathers EMBEDDING-TABLE rows by token id and scales by sqrt(hidden).
 * This gathers rows of a hidden activation by PACKED ROW INDEX and copies them verbatim,
 * because the final RMSNorm that follows must see exactly the bytes the body wrote.
 *
 * `rows[s] >= live` traps. A clamped index selects another request's hidden row, and the token
 * it produces is fluent and wrong. */
static __device__ void d_row_gather(__nv_bfloat16* __restrict__ out,
                                    const __nv_bfloat16* __restrict__ x,
                                    const unsigned* __restrict__ rows, unsigned n_sample,
                                    unsigned hidden, unsigned live, unsigned slice,
                                    unsigned nblk) {
    for (unsigned s = slice; s < n_sample; s += nblk) {
        const unsigned src_row = rows[s];
        if (src_row >= live) { asm volatile("trap;"); return; }
        const size_t src = (size_t)src_row * hidden, dst = (size_t)s * hidden;
        if ((hidden & 7u) == 0) {
            for (unsigned i = threadIdx.x * 8; i < hidden; i += PLOW_NV_THREADS * 8)
                st_glob8(out + dst + i, ld_glob8(x + src + i));
        } else {
            for (unsigned i = threadIdx.x; i < hidden; i += PLOW_NV_THREADS)
                out[dst + i] = x[src + i];
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
    unsigned i = (slice * PLOW_NV_THREADS + threadIdx.x) * 8;
    /* Four 16-byte loads in flight per thread (a batch of logit rows is tens of MB). */
    for (; i + 3u * stride + 8u <= n; i += 4u * stride) {
        bf16v8 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] = ld_glob8(x + i + u * stride);
#pragma unroll
        for (int u = 0; u < 4; u++) {
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++)
                o.x[j] = __float2bfloat16(cap * tanhf(__bfloat162float(v[u].x[j]) * inv));
            st_glob8(out + i + u * stride, o);
        }
    }
    for (; i < n; i += stride) {
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
/* Op 205: out[r][p] = bf16(act(gate[r][p])) * up[r*stride + col0 + p]. */
static __device__ void d_glu_strided(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
                                     const __nv_bfloat16* __restrict__ up, unsigned rows, unsigned width,
                                     unsigned col0, unsigned stride, unsigned act, unsigned slice,
                                     unsigned nblk) {
    const unsigned n = rows * width;
    if (((width | stride | col0) & 7u) == 0) {
        /* 8 elements per thread-step, 16-byte accesses, two steps in flight: the scalar loop
         * below was one dependent 2-byte load pair per step (latency-bound, ~20 us per op at
         * 2k rows). Same per-element math. */
        const unsigned n8 = n >> 3, w8 = width >> 3, step = nblk * PLOW_NV_THREADS;
        auto one = [&](const bf16v8& g, const bf16v8& u) {
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float x = __bfloat162float(g.x[j]);
                const float a = __bfloat162float(
                    __float2bfloat16((act == PLOW_ACT_SILU_) ? act_silu(x) : act_gelu_tanh(x)));
                o.x[j] = __float2bfloat16(a * __bfloat162float(u.x[j]));
            }
            return o;
        };
        unsigned v = slice * PLOW_NV_THREADS + threadIdx.x;
        for (; v + step < n8; v += 2 * step) {
            const unsigned v1 = v + step;
            const unsigned r0 = v / w8, r1 = v1 / w8;
            const bf16v8 g0 = ld_glob8(gate + (size_t)v * 8), g1 = ld_glob8(gate + (size_t)v1 * 8);
            const bf16v8 u0 = ld_glob8(up + (size_t)r0 * stride + col0 + (v - r0 * w8) * 8);
            const bf16v8 u1 = ld_glob8(up + (size_t)r1 * stride + col0 + (v1 - r1 * w8) * 8);
            st_glob8(out + (size_t)v * 8, one(g0, u0));
            st_glob8(out + (size_t)v1 * 8, one(g1, u1));
        }
        if (v < n8) {
            const unsigned r = v / w8;
            st_glob8(out + (size_t)v * 8,
                     one(ld_glob8(gate + (size_t)v * 8),
                         ld_glob8(up + (size_t)r * stride + col0 + (v - r * w8) * 8)));
        }
        return;
    }
    for (unsigned i = slice * PLOW_NV_THREADS + threadIdx.x; i < n; i += nblk * PLOW_NV_THREADS) {
        const unsigned r = i / width, p = i - r * width;
        const float g = __bfloat162float(gate[i]);
        const float a = __bfloat162float(__float2bfloat16((act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g)));
        out[i] = __float2bfloat16(a * __bfloat162float(up[(size_t)r * stride + col0 + p]));
    }
}

static __device__ void d_glu(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
                      const __nv_bfloat16* __restrict__ up, unsigned n, unsigned act,
                      unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_NV_THREADS * 8;
    const unsigned i0 = (slice * PLOW_NV_THREADS + threadIdx.x) * 8;
    const unsigned nfull = n & ~7u;
    /* Four gate/up pairs (128 B per thread) are LOADED before any is consumed. `#pragma unroll`
     * alone changed nothing (h100 seg-time 291 -> 293 us at 4096 rows): ld_glob8/st_glob8 are
     * uint4 punning, so the compiler keeps each iteration's loads behind the previous store.
     * The n % 8 tail is its own loop below. */
    auto glu8 = [&](const bf16v8& vg, const bf16v8& vu) {
        bf16v8 vo;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float g = __bfloat162float(vg.x[j]);
            float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh_pf(g);
#if defined(PLOW_NV_GEMMA) && PLOW_NV_GEMMA && defined(PLOW_NV_GEMMA_GLU_BF16) && PLOW_NV_GEMMA_GLU_BF16
            if (act != PLOW_ACT_SILU_) a = __bfloat162float(__float2bfloat16(a));
#endif
            vo.x[j] = __float2bfloat16(a * __bfloat162float(vu.x[j]));
        }
        return vo;
    };
    unsigned i = i0;
    for (; i + 3u * stride < nfull; i += 4u * stride) {
        bf16v8 vg[4], vu[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            vg[u] = ld_glob8(gate + i + (unsigned)u * stride);
            vu[u] = ld_glob8(up + i + (unsigned)u * stride);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) st_glob8(out + i + (unsigned)u * stride, glu8(vg[u], vu[u]));
    }
    for (; i < nfull; i += stride) st_glob8(out + i, glu8(ld_glob8(gate + i), ld_glob8(up + i)));
    if (nfull < n) {
        for (unsigned i = i0; i < n; i += stride) {
            if (i < nfull) continue;
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
/* BATCH>1: every block used to scan a 1/nblk sliver of EVERY row, one block reduction per row
 * (Veena B=128: 222 us). Here the (row, chunk) items are spread over the blocks: G = nblk / B
 * contiguous chunks per row (one whole row per item once B >= nblk). A row's chunk-0 item
 * zero-fills its unused slots, and 0 is below every packed key, so ARGMAX_FIN's fold picks the
 * same token. */
static __device__ void d_argmax_rows(unsigned long long* __restrict__ part,
                                     const __nv_bfloat16* __restrict__ x, unsigned n, unsigned B,
                                     unsigned slice, unsigned nblk, unsigned long long* lds) {
    const unsigned G = nblk >= B ? nblk / B : 1u;
    for (unsigned w = slice; w < B * G; w += nblk) {
        const unsigned b = w / G, c = w % G;
        const __nv_bfloat16* xb = x + (size_t)b * n;
        const unsigned mis = (unsigned)(((size_t)b * n) & 7u);
        const unsigned h = mis ? min(8u - mis, n) : 0u;
        const unsigned nv = (n - h) / 8;
        const unsigned v0 = (unsigned)(((unsigned long long)c * nv) / G);
        const unsigned v1 = (unsigned)(((unsigned long long)(c + 1u) * nv) / G);
        unsigned long long best = 0;
        unsigned iv = v0 + threadIdx.x;
        /* Four loads in flight; the max over packed keys does not depend on the visit order. */
        for (; iv + 3u * PLOW_NV_THREADS < v1; iv += 4u * PLOW_NV_THREADS) {
            bf16v8 v[4];
#pragma unroll
            for (unsigned u = 0; u < 4; u++) v[u] = ld_glob8_cs(xb + h + (size_t)(iv + u * PLOW_NV_THREADS) * 8);
#pragma unroll
            for (unsigned u = 0; u < 4; u++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const unsigned long long p =
                        amax_pack(v[u].x[j], h + (iv + u * PLOW_NV_THREADS) * 8 + (unsigned)j);
                    best = p > best ? p : best;
                }
        }
        for (; iv < v1; iv += PLOW_NV_THREADS) {
            const bf16v8 v = ld_glob8_cs(xb + h + (size_t)iv * 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned long long p = amax_pack(v.x[j], h + iv * 8 + (unsigned)j);
                best = p > best ? p : best;
            }
        }
        if (c == 0) {
            for (unsigned i = threadIdx.x; i < h; i += PLOW_NV_THREADS) {
                const unsigned long long p = amax_pack(xb[i], i);
                best = p > best ? p : best;
            }
            for (unsigned i = h + nv * 8 + threadIdx.x; i < n; i += PLOW_NV_THREADS) {
                const unsigned long long p = amax_pack(xb[i], i);
                best = p > best ? p : best;
            }
            for (unsigned i = G + threadIdx.x; i < nblk; i += PLOW_NV_THREADS)
                part[(size_t)b * nblk + i] = 0ull;
        }
        best = block_max_u64(best, lds);
        if (threadIdx.x == 0) part[(size_t)b * nblk + c] = best;
    }
}

static __device__ void d_argmax(unsigned long long* __restrict__ part, const __nv_bfloat16* __restrict__ x,
                         unsigned n, unsigned n_batch, unsigned slice, unsigned nblk,
                         unsigned long long* lds) {
    const unsigned B = n_batch ? n_batch : 1u;
    if (B > 1u) {
        d_argmax_rows(part, x, n, B, slice, nblk, lds);
        return;
    }
    for (unsigned b = 0; b < B; b++) {
        const __nv_bfloat16* xb = x + (size_t)b * n;
        unsigned long long best = 0;
        /* An n that is not a multiple of 8 (Veena: 156951) leaves row b>0 off 16-byte alignment;
         * scan `h` scalar elements first so the vector loads start aligned. h = 0 on aligned rows. */
        const unsigned mis = (unsigned)(((size_t)b * n) & 7u);
        const unsigned h = mis ? min(8u - mis, n) : 0u;
        /* VECTORIZED scan: 1 LD.E.128 per 8 elements instead of 8 scalar LD.E.U16 each with its
         * own 64-bit address build (~16 -> ~7 slots/element). This changes which block scans
         * which elements (part[slice] partials shift between slots), but the ONLY consumer is
         * ARGMAX_FIN's max-fold over all slots, and max over the same global candidate set with
         * the same packed tie-break key picks the same winner — the token is unchanged. */
        const unsigned nv = (n - h) / 8;
        for (unsigned iv = slice * PLOW_NV_THREADS + threadIdx.x; iv < nv;
             iv += nblk * PLOW_NV_THREADS) {
            const bf16v8 v = ld_glob8_cs(xb + h + (size_t)iv * 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned long long p = amax_pack(v.x[j], h + iv * 8 + (unsigned)j);
                best = p > best ? p : best;
            }
        }
        /* Unaligned head and n % 8 tail, scalar. */
        if (slice == 0) {
            for (unsigned i = threadIdx.x; i < h; i += PLOW_NV_THREADS) {
                const unsigned long long p = amax_pack(xb[i], i);
                best = p > best ? p : best;
            }
            for (unsigned i = h + nv * 8 + threadIdx.x; i < n; i += PLOW_NV_THREADS) {
                const unsigned long long p = amax_pack(xb[i], i);
                best = p > best ? p : best;
            }
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
    const unsigned B = n_batch ? n_batch : 1u;
    /* One thread per sequence (it was thread 0 for all of them: 128 x 64 dependent loads, 207 us
     * at Veena B=128). */
    if (slice != 0 || threadIdx.x >= B) return;
    for (unsigned b = threadIdx.x; b < B; b += blockDim.x) {
        const unsigned long long* pb = part + (size_t)b * nparts;
        unsigned long long best = 0;
        for (unsigned i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        ids[b] = (int)~(unsigned)(best & 0xFFFFFFFFull);
    }
}
