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
 * overlay row, which is BF16-rounded in (the CPU golden's plow_f2bf: round-to-nearest-even);
 * index 0xFFFFFFFE keeps the row an earlier embed op wrote. */
static __device__ void d_embed_overlay(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                       const unsigned* __restrict__ tokens, const float* __restrict__ overlay,
                                       const unsigned* __restrict__ overlay_index, unsigned rows, unsigned width,
                                       unsigned vocab, unsigned overlay_rows, unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned sel = overlay_index[r];
        if (sel == 0xFFFFFFFEu) continue;
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

/* EmbedFrameBf16 (op 218): a frame token adds its K codebook rows (bf16 sum), then / div; mode bit 0
 * drops the token row, bit 1 RMS-normalises the frame row (fp32 weight `nw`, eps); see dev.rs.
 * Arena (bit 1): PLOW_NV_WARPS f32 reduction slots. */
static __device__ void d_embed_frame(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                     const unsigned* __restrict__ tokens, const __nv_bfloat16* __restrict__ cb,
                                     const unsigned* __restrict__ codes, const float* __restrict__ nw,
                                     unsigned rows, unsigned width, unsigned vocab, unsigned K,
                                     unsigned cb_size, unsigned frame_lo, unsigned frame_rows, unsigned mode,
                                     float div, float eps, unsigned slice, unsigned nblk, float* arena) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned tok = tokens[r];
        if (tok >= vocab) { __trap(); return; }
        const __nv_bfloat16* a = table + (size_t)tok * width;
        __nv_bfloat16* o = out + (size_t)r * width;
        if (tok - frame_lo >= frame_rows) {
            for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS) o[i] = a[i];
            continue;
        }
        const unsigned* c = codes + (size_t)r * K;
        for (unsigned k = 0; k < K; k++)
            if (c[k] >= cb_size) { __trap(); return; }
        float ss = 0.f;
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS) {
            float s = 0.f;
            for (unsigned k = 0; k < K; k++)
                s += __bfloat162float(cb[((size_t)k * cb_size + c[k]) * width + i]);
            const float sb = __bfloat162float(__float2bfloat16_rn(s));
            const float t = (mode & 1u) ? sb
                                        : __bfloat162float(__float2bfloat16_rn(__bfloat162float(a[i]) + sb));
            o[i] = __float2bfloat16_rn(__fdiv_rn(t, div));
            const float y = __bfloat162float(o[i]);
            ss = __fmaf_rn(y, y, ss);
        }
        if (!(mode & 2u)) continue;
        const float inv = rsqrtf(__fadd_rn(__fdiv_rn(block_sum(ss, arena), (float)width), eps));
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
            o[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(o[i]), inv), nw[i]));
        __syncthreads();
    }
}

/* SampleFishBf16 (op 219): Fish S2 sampling, one block per row; see dev.rs. Arena: PLOW_NV_WARPS u64
 * + PLOW_NV_WARPS f32 reduction slots, 64 selected classes and logits, then the row as bf16 bits. */
static __device__ __forceinline__ unsigned long long sf_mix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static __device__ __forceinline__ float sf_neg_log_exp1(unsigned long long seed, unsigned step, unsigned coord,
                                                        unsigned draw, unsigned i) {
    unsigned long long z = sf_mix(seed ^ sf_mix(((unsigned long long)step << 32) | coord));
    z = sf_mix(z ^ (((unsigned long long)draw << 32) | i));
    const float u = ((float)(z >> 40) + 0.5f) * (1.0f / 16777216.0f);
    return -logf(-logf(u));
}
static __device__ void d_sample_fish(unsigned* __restrict__ out, const __nv_bfloat16* __restrict__ logits,
                                     const float* __restrict__ params, const unsigned* __restrict__ topk,
                                     const unsigned long long* __restrict__ seed, const unsigned* __restrict__ step,
                                     unsigned* __restrict__ hist, unsigned* __restrict__ out_flat, unsigned rows,
                                     unsigned vocab, unsigned out_stride,
                                     unsigned out_col0, unsigned coord, unsigned window, unsigned flags,
                                     unsigned lstride, unsigned ras_limit, unsigned slice, unsigned nblk,
                                     float* arena) {
    unsigned long long* part = (unsigned long long*)arena;
    float* fpart = (float*)(part + PLOW_NV_WARPS);
    unsigned* sel = (unsigned*)(fpart + PLOW_NV_WARPS);
    float* sv = (float*)(sel + 64);
    unsigned short* L = (unsigned short*)(sv + 64);
    const unsigned ls = lstride ? lstride : vocab;
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned K = min(min(topk[r], 64u), vocab);
        if (!K) continue;
        const unsigned short* lg = (const unsigned short*)(logits + (size_t)r * ls);
        float m = -INFINITY;
        for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS) {
            const unsigned short b = lg[i];
            L[i] = b;
            m = fmaxf(m, __uint_as_float((unsigned)b << 16));
        }
        m = block_max(m, fpart);
        float z = 0.f;
        for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS)
            z += expf(__uint_as_float((unsigned)L[i] << 16) - m);
        z = block_sum(z, fpart);
        for (unsigned k = 0; k < K; k++) {
            unsigned long long best = 0;
            for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS) {
                unsigned short b = L[i];
                const unsigned long long p = amax_pack(*(const __nv_bfloat16*)&b, i);
                best = p > best ? p : best;
            }
            best = block_max_u64(best, part);
            if (threadIdx.x == 0) {
                const unsigned idx = ~(unsigned)best;
                sel[k] = idx;
                sv[k] = __uint_as_float((unsigned)L[idx] << 16);
                L[idx] = 0xFF80u;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float* p = params + (size_t)r * 4u;
            auto pick = [&](float t, float top_p, unsigned draw) {
                float cum = 0.f;
                unsigned n = 0;
                for (unsigned i = 0; i < K; i++) {
                    cum += expf(sv[i] - m) / z;
                    if (i > 0 && cum > top_p) break;
                    n = i + 1u;
                }
                const float inv = 1.0f / fmaxf(t, 1e-5f);
                unsigned best = 0;
                float bv = -INFINITY;
                for (unsigned i = 0; i < n; i++) {
                    const float v = (sv[i] - sv[0]) * inv + sf_neg_log_exp1(seed[r], step[r], coord, draw, i);
                    if (v > bv) { bv = v; best = i; }
                }
                return sel[best];
            };
            unsigned c = pick(p[0], p[1], 0u);
            if ((flags & 1u) && hist && window) {
                unsigned* h = hist + (size_t)r * window;
                bool seen = false;
                for (unsigned w = 0; w < window; w++) seen |= h[w] == c + 1u;
                if (step[r] > 0 && c < ras_limit && seen) c = pick(p[2], p[3], 1u);
                h[step[r] % window] = c + 1u;
            }
            out[(size_t)r * out_stride + out_col0] = c;
            if (out_flat) out_flat[r] = c;
        }
        __syncthreads();
    }
}

/* GemvF32Out (op 220): C[m][n] = sum_k x[m][k] * W[n][k] in fp32 (exact bf16 products, fp32 fma), stored
 * fp32 — a reference that casts both operands to f32 before its matmul. A warp per output row, rows
 * in blocks of 8 (W re-read once per block of rows). K % 8 == 0. */
static __device__ void d_gemv_f32out(float* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ W, unsigned M, unsigned N, unsigned K,
                                     unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* wr = W + (size_t)n * K;
        for (unsigned m0 = 0; m0 < M; m0 += 8) {
            float acc[8];
#pragma unroll
            for (int m = 0; m < 8; m++) acc[m] = 0.f;
            for (unsigned k = lane * 8u; k < K; k += 32u * 8u) {
                const bf16v8 wv = ld_glob8(wr + k);
#pragma unroll
                for (int m = 0; m < 8; m++)
                    if (m0 + m < M) acc[m] = dot8(wv, ld_glob8(x + (size_t)(m0 + m) * K + k), acc[m]);
            }
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const float t = warp_sum32(acc[m]);
                if (lane == 0 && m0 + m < M) C[(size_t)(m0 + m) * N + n] = t;
            }
        }
    }
}

/* GemvF32Out modes (i7 != 0; Maya 2's depth projections): bit 0 x is f32, bit 1 W is f16, bit 2 adds
 * an f32 bias t3[N], bit 3 adds an f32 residual t4[M][N] (may alias C: each element is read then
 * written by the one lane that owns it), bits 4-7 the activation applied last (1 = GELU erf). */
template <bool XF, bool WH>
static __device__ void gemv_f32out_mode(float* C, const void* xv, const void* Wv, const float* bias,
                                        const float* res, unsigned act, unsigned M, unsigned N, unsigned K,
                                        unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        for (unsigned m0 = 0; m0 < M; m0 += 8) {
            float acc[8];
#pragma unroll
            for (int m = 0; m < 8; m++) acc[m] = 0.f;
            for (unsigned k = lane * 8u; k < K; k += 32u * 8u) {
                float w[8];
                const uint4 raw = *(const uint4*)((const unsigned short*)Wv + (size_t)n * K + k);
                const unsigned short* h = (const unsigned short*)&raw;
#pragma unroll
                for (int e = 0; e < 8; e++)
                    w[e] = WH ? __half2float(__ushort_as_half(h[e])) : __uint_as_float((unsigned)h[e] << 16);
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    if (m0 + m >= M) break;
                    float xs[8];
                    if (XF) {
                        const float4* xp = (const float4*)((const float*)xv + (size_t)(m0 + m) * K + k);
                        const float4 a = xp[0], b = xp[1];
                        xs[0] = a.x; xs[1] = a.y; xs[2] = a.z; xs[3] = a.w;
                        xs[4] = b.x; xs[5] = b.y; xs[6] = b.z; xs[7] = b.w;
                    } else {
                        const uint4 xr = *(const uint4*)((const unsigned short*)xv + (size_t)(m0 + m) * K + k);
                        const unsigned short* xh = (const unsigned short*)&xr;
#pragma unroll
                        for (int e = 0; e < 8; e++) xs[e] = __uint_as_float((unsigned)xh[e] << 16);
                    }
#pragma unroll
                    for (int e = 0; e < 8; e++) acc[m] = __fmaf_rn(xs[e], w[e], acc[m]);
                }
            }
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const float t = warp_sum32(acc[m]);
                if (lane == 0 && m0 + m < M) {
                    const size_t o = (size_t)(m0 + m) * N + n;
                    float v = t;
                    if (bias) v = __fadd_rn(v, bias[n]);
                    if (res) v = __fadd_rn(v, res[o]);
                    if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                    C[o] = v;
                }
            }
        }
    }
}

/* f32 x, f16 W (modes 3): each block stages 8 rows x GEMV_KT columns of x in shared memory once and
 * every warp reuses them for up to GEMV_CPW columns (x read from L2 once per block, not once per
 * column). With the LayerNorm prologue (bit 8, K <= GEMV_KT) the staged rows are normalised first:
 * fp32 two-pass mean / variance, `(x - mean) * rsqrt(var + eps) * gamma + beta`.
 * Arena: 8 * GEMV_KT + 16 floats. */
#define GEMV_KT 1024u
#define GEMV_CPW 8u
static __device__ void gemv_f32x_f16w_staged(float* C, const float* __restrict__ x, const unsigned short* __restrict__ W,
                                             const float* bias, const float* res, const float* gamma,
                                             const float* beta, float eps, unsigned act, unsigned M, unsigned N,
                                             unsigned K, unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);
    float* xs = arena;
    float* st = arena + 8u * GEMV_KT;
    for (unsigned cb = n0; cb < n1; cb += PLOW_NV_WARPS * GEMV_CPW) {
        for (unsigned m0 = 0; m0 < M; m0 += 8) {
            const unsigned mr = min(8u, M - m0);
            float acc[GEMV_CPW][8];
#pragma unroll
            for (unsigned j = 0; j < GEMV_CPW; j++)
#pragma unroll
                for (int m = 0; m < 8; m++) acc[j][m] = 0.f;
            for (unsigned kt = 0; kt < K; kt += GEMV_KT) {
                const unsigned kw = min(GEMV_KT, K - kt);
                __syncthreads();
                for (unsigned i = threadIdx.x; i < 8u * kw; i += PLOW_NV_THREADS) {
                    const unsigned m = i / kw, k = i % kw;
                    xs[m * GEMV_KT + k] = m < mr ? x[(size_t)(m0 + m) * K + kt + k] : 0.f;
                }
                __syncthreads();
                if (gamma) {
                    for (unsigned m = warp; m < mr; m += PLOW_NV_WARPS) {
                        float s = 0.f;
                        for (unsigned k = lane; k < kw; k += 32u) s += xs[m * GEMV_KT + k];
                        const float mean = warp_sum32(s) / (float)kw;
                        float v = 0.f;
                        for (unsigned k = lane; k < kw; k += 32u) {
                            const float d = xs[m * GEMV_KT + k] - mean;
                            v = __fmaf_rn(d, d, v);
                        }
                        const float var = warp_sum32(v) / (float)kw;
                        if (lane == 0) {
                            st[2 * m] = mean;
                            st[2 * m + 1] = rsqrtf(var + eps);
                        }
                    }
                    __syncthreads();
                    for (unsigned i = threadIdx.x; i < mr * kw; i += PLOW_NV_THREADS) {
                        const unsigned m = i / kw, k = i % kw;
                        float* p = xs + m * GEMV_KT + k;
                        *p = __fmaf_rn((*p - st[2 * m]) * st[2 * m + 1], gamma[k], beta[k]);
                    }
                    __syncthreads();
                }
                for (unsigned k = lane * 8u; k < kw; k += 32u * 8u) {
                    /* Every column's 16 B first (GEMV_CPW loads in flight per lane), then the FMAs. */
                    uint4 wr[GEMV_CPW];
#pragma unroll
                    for (unsigned j = 0; j < GEMV_CPW; j++) {
                        const unsigned n = cb + warp + j * PLOW_NV_WARPS;
                        wr[j] = n < n1 ? *(const uint4*)(W + (size_t)n * K + kt + k) : make_uint4(0u, 0u, 0u, 0u);
                    }
#pragma unroll
                    for (int m = 0; m < 8; m++) {
                        if ((unsigned)m >= mr) break;
                        const float4* xp = (const float4*)(xs + m * GEMV_KT + k);
                        const float4 a = xp[0], b = xp[1];
#pragma unroll
                        for (unsigned j = 0; j < GEMV_CPW; j++) {
                            const __half2* h = (const __half2*)&wr[j];
                            const float2 w0 = __half22float2(h[0]), w1 = __half22float2(h[1]);
                            const float2 w2 = __half22float2(h[2]), w3 = __half22float2(h[3]);
                            float t = acc[j][m];
                            t = __fmaf_rn(a.x, w0.x, t); t = __fmaf_rn(a.y, w0.y, t);
                            t = __fmaf_rn(a.z, w1.x, t); t = __fmaf_rn(a.w, w1.y, t);
                            t = __fmaf_rn(b.x, w2.x, t); t = __fmaf_rn(b.y, w2.y, t);
                            t = __fmaf_rn(b.z, w3.x, t); t = __fmaf_rn(b.w, w3.y, t);
                            acc[j][m] = t;
                        }
                    }
                }
            }
#pragma unroll
            for (unsigned j = 0; j < GEMV_CPW; j++) {
                const unsigned n = cb + warp + j * PLOW_NV_WARPS;
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    const float t = warp_sum32(acc[j][m]);
                    if (lane == 0 && n < n1 && (unsigned)m < mr) {
                        const size_t o = (size_t)(m0 + m) * N + n;
                        float v = t;
                        if (bias) v = __fadd_rn(v, bias[n]);
                        if (res) v = __fadd_rn(v, res[o]);
                        if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                        C[o] = v;
                    }
                }
            }
        }
    }
}

static __device__ void d_gemv_f32out_mode(float* C, const void* x, const void* W, const float* bias,
                                          const float* res, const float* gamma, const float* beta, float eps,
                                          unsigned flags, unsigned M, unsigned N, unsigned K, unsigned slice,
                                          unsigned nblk, float* arena) {
    const unsigned act = (flags >> 4) & 15u;
    bias = (flags & 4u) ? bias : nullptr;
    res = (flags & 8u) ? res : nullptr;
    if ((flags & 3u) == 3u) {
        if ((flags & 256u) && K > GEMV_KT) { __trap(); return; }
        gemv_f32x_f16w_staged(C, (const float*)x, (const unsigned short*)W, bias, res,
                              (flags & 256u) ? gamma : nullptr, beta, eps, act, M, N, K, slice, nblk, arena);
        return;
    }
    switch (flags & 3u) {
    case 0: gemv_f32out_mode<false, false>(C, x, W, bias, res, act, M, N, K, slice, nblk); break;
    case 1: gemv_f32out_mode<true, false>(C, x, W, bias, res, act, M, N, K, slice, nblk); break;
    case 2: gemv_f32out_mode<false, true>(C, x, W, bias, res, act, M, N, K, slice, nblk); break;
    default: gemv_f32out_mode<true, true>(C, x, W, bias, res, act, M, N, K, slice, nblk); break;
    }
}

/* SampleSmallF32 (op 221): one block per row of f32 logits; see dev.rs. Arena: PLOW_NV_WARPS u64 +
 * PLOW_NV_WARPS f32 reduction slots, 64 selected classes and values, then the row (f32). */
static __device__ __forceinline__ unsigned long long f32_key(float f, unsigned i) {
    const unsigned b = __float_as_uint(f + 0.f); /* -0 ranks as +0 (the golden's comparison) */
    const unsigned key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~i);
}
static __device__ void d_sample_small(unsigned* __restrict__ out, const float* __restrict__ logits,
                                      const float* __restrict__ params, const unsigned* __restrict__ ctl,
                                      const unsigned long long* __restrict__ seed, unsigned* __restrict__ bitmap,
                                      const unsigned* __restrict__ force, unsigned* __restrict__ eidx,
                                      unsigned rows, unsigned vocab, unsigned out_stride, unsigned out_col0,
                                      unsigned coord, unsigned words, unsigned lstride, unsigned eoff,
                                      unsigned slice, unsigned nblk, float* arena) {
    unsigned long long* part = (unsigned long long*)arena;
    float* fpart = (float*)(part + PLOW_NV_WARPS);
    unsigned* sel = (unsigned*)(fpart + PLOW_NV_WARPS);
    float* sv = (float*)(sel + 64);
    float* L = sv + 64;
    const unsigned ls = lstride ? lstride : vocab;
    for (unsigned r = slice; r < rows; r += nblk) {
        /* A CFG pair (plow_asset::fish_depth GUIDE / FOLLOW bits): the guide row r samples
         * u + s (c - u) over rows (r, r + 1) = (cond, uncond) and writes both; the follower skips. */
        const unsigned cw = ctl[2u * r];
        if (cw & (1u << 30)) continue;
        const bool guide = cw & (1u << 31);
        const unsigned K = min(min(cw & ~(3u << 30), 64u), vocab);
        if (!K) continue;
        const float* p = params + (size_t)r * 4u;
        const float t = p[0], pen = p[2], cs = p[3];
        const unsigned* bm = bitmap ? bitmap + (size_t)r * words : nullptr;
        const float* lg = logits + (size_t)r * ls;
        const float* lu = guide ? lg + ls : nullptr;
        for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS) {
            float v = lg[i];
            if (lu) v = __fadd_rn(lu[i], __fmul_rn(cs, __fsub_rn(v, lu[i])));
            if (bm && ((bm[i >> 5] >> (i & 31u)) & 1u)) v = v > 0.f ? __fdiv_rn(v, pen) : v * pen;
            L[i] = t > 0.f ? __fdiv_rn(v, t) : v;
        }
        __syncthreads();
        for (unsigned k = 0; k < K; k++) {
            unsigned long long best = 0;
            for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS) {
                const unsigned long long q = f32_key(L[i], i);
                best = q > best ? q : best;
            }
            best = block_max_u64(best, part);
            if (threadIdx.x == 0) {
                const unsigned idx = ~(unsigned)best;
                sel[k] = idx;
                sv[k] = L[idx];
                L[idx] = -INFINITY;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const unsigned step = ctl[2u * r + 1u];
            unsigned n = K;
            if (p[1] < 1.f) {
                float z = 0.f;
                for (unsigned i = 0; i < K; i++) z += expf(sv[i] - sv[0]);
                float cum = 0.f;
                for (unsigned i = 0; i < K; i++) {
                    if (i > 0 && cum > p[1]) { n = i; break; }
                    cum += expf(sv[i] - sv[0]) / z;
                }
            }
            unsigned best = 0;
            float bv = -INFINITY;
            for (unsigned i = 0; i < n; i++) {
                const float v = sv[i] + sf_neg_log_exp1(seed[r], step, coord, 0u, i);
                if (v > bv) { bv = v; best = i; }
            }
            unsigned c = sel[best];
            const size_t o = (size_t)r * out_stride + out_col0;
            if (force && force[o] != 0xFFFFFFFFu) c = force[o];
            if (bm) bitmap[(size_t)r * words + (c >> 5)] |= 1u << (c & 31u);
            out[o] = c;
            if (eidx) eidx[r] = c + eoff;
            if (guide) {
                out[o + out_stride] = c;
                if (eidx) eidx[r + 1] = c + eoff;
            }
        }
        __syncthreads();
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
