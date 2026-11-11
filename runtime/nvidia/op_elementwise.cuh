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
 * + PLOW_NV_WARPS f32 reduction slots, 64 selected classes and logits, the row as bf16 bits, then the
 * top-k scratch (block_topk_u64: 64 u64 + 260 u32, 16-byte aligned). */
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
/* The samplers' radix top-k (plow_config.h: packets with depth programs). Inlined into every
 * interpreter it grew the frames of objects that never sample (Gemma 12B FP8 decode 224 -> 240 B,
 * +2.7% step); 0 keeps the K-round selection alone. */
#ifndef PLOW_NV_SAMPLE_RADIX
#define PLOW_NV_SAMPLE_RADIX 1
#endif
/* The depth stage's arena-staged head GEMV input and per-warp Gumbel pick (plow_config.h: packets
 * with depth programs; those launch on the decode, _gw and speech objects, never on a prefill
 * object). Like the radix top-k, inlined everywhere they grew frames of objects that never run
 * them (Qwen3-TTS prefill 824 -> 840 B, segmented GEMM 1232 -> 1264 B); 0 keeps the RC9 bodies,
 * same results. */
#ifndef PLOW_NV_DEPTH_STAGE
#define PLOW_NV_DEPTH_STAGE 1
#endif
#if PLOW_NV_DEPTH_STAGE && !(defined(PLOW_NV_PREFILL) && PLOW_NV_PREFILL)
#define GV_DEPTH_FAST 1
#else
#define GV_DEPTH_FAST 0
#endif
/* d_sample_small's Gumbel pick on a full warp instead of one lane: same pick, +16 B on the
 * Qwen3-TTS decode frame (112 -> 128) and no measured gain yet, so off. */
#ifndef PLOW_NV_GUMBEL_WARP
#define PLOW_NV_GUMBEL_WARP 0
#endif
#if PLOW_NV_GUMBEL_WARP && GV_DEPTH_FAST
#define GV_GUMBEL_WARP 1
#else
#define GV_GUMBEL_WARP 0
#endif
/* Whether shared bytes up to `end` lie inside this launch's shared allocation (static + dynamic):
 * the radix top-k scratch sits past the row, so a large vocabulary or a narrow launch claim takes
 * the K-round selection instead. */
/* Whether [dynamic arena base, end) lies inside the launch's dynamic shared allocation. */
static __device__ __forceinline__ bool smem_fits(const void* end) {
    extern __shared__ __align__(16) unsigned char plow_dyn_base[];
    unsigned dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
    return (size_t)__cvta_generic_to_shared(end) <= (size_t)__cvta_generic_to_shared(plow_dyn_base) + dyn;
}
/* Block-wide top-K (K <= 64, K <= n) of n distinct u64 keys key(i): sel[0..K) = ~(unsigned)key in
 * descending key order, the set and order K rounds of block_max_u64 pick. MSD radix select over 8-bit
 * digits from bits [shift0, shift0 + 8) down, then a rank sort of the K survivors. Scratch: ctl[4] and
 * hist[256] u32, buf[64] u64, pm[2] u64, never __restrict__: other lanes rewrite them between
 * barriers, and a restrict pm let a lane read back its own older store. Ends synchronized. */
template <class KeyF>
static __device__ __forceinline__ void block_topk_u64(KeyF key, unsigned n, unsigned K, int shift0, unsigned* sel,
                                      unsigned* ctl, unsigned* hist, unsigned long long* buf,
                                      unsigned long long* pm) {
    if (threadIdx.x == 0) {
        pm[0] = 0;
        pm[1] = 0;
        ctl[0] = K;
        ctl[1] = 0;
        ctl[2] = 0;
    }
    for (int shift = shift0; shift >= 0; shift -= 8) {
        for (unsigned b = threadIdx.x; b < 256u; b += PLOW_NV_THREADS) hist[b] = 0;
        __syncthreads();
        const unsigned long long pre = pm[0], msk = pm[1];
        for (unsigned i = threadIdx.x; i < n; i += PLOW_NV_THREADS) {
            const unsigned long long q = key(i);
            if ((q & msk) == pre) atomicAdd(&hist[(unsigned)(q >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x < 32u) {
            /* lane l holds bins 255 - 8l .. 248 - 8l: an inclusive scan from the top bin finds the
             * digit holding the need-th largest remaining key. */
            const unsigned lane = threadIdx.x, need = ctl[0];
            const unsigned* h = hist + 248u - 8u * lane; /* h[7 - j] = the lane's j-th bin from the top */
            unsigned sum = 0;
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) sum += h[j];
            unsigned incl = sum;
#pragma unroll
            for (unsigned off = 1; off < 32u; off <<= 1) {
                const unsigned o = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off) incl += o;
            }
            unsigned cum = incl - sum;
            __syncwarp(); /* every lane has read `need` before the selected lane overwrites ctl[0] */
            if (cum < need && need <= incl) {
                for (unsigned j = 0; j < 8u; j++) {
                    const unsigned c = h[7u - j];
                    if (cum + c >= need) {
                        const unsigned left = need - cum;
                        pm[0] = pre | ((unsigned long long)(255u - 8u * lane - j) << shift);
                        pm[1] = msk | (0xffull << shift);
                        ctl[0] = left;
                        ctl[1] = c == left; /* the whole digit is taken: the threshold is final */
                        break;
                    }
                    cum += c;
                }
            }
        }
        __syncthreads();
        if (ctl[1]) break;
    }
    /* Exactly K keys reach the threshold: the ones above its prefix and the `left` sharing it. */
    const unsigned long long T = pm[0];
    for (unsigned i = threadIdx.x; i < n; i += PLOW_NV_THREADS) {
        const unsigned long long q = key(i);
        if (q >= T) {
            const unsigned at = atomicAdd(&ctl[2], 1u);
            if (at < 64u) buf[at] = q;
        }
    }
    __syncthreads();
    for (unsigned t = threadIdx.x; t < K; t += PLOW_NV_THREADS) {
        const unsigned long long q = buf[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < K; j++) rank += buf[j] > q;
        sel[rank] = ~(unsigned)q;
    }
    __syncthreads();
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
    unsigned long long* tk_buf = (unsigned long long*)(((size_t)(L + vocab) + 15u) & ~(size_t)15u);
    unsigned* tk_ctl = (unsigned*)(tk_buf + 64);
    const bool radix = PLOW_NV_SAMPLE_RADIX && smem_fits(tk_ctl + 4 + 256);
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
        if (radix) {
            block_topk_u64(
                [L](unsigned i) {
                    const unsigned short b = L[i];
                    return amax_pack(*(const __nv_bfloat16*)&b, i);
                },
                vocab, K, 40, sel, tk_ctl, tk_ctl + 4, tk_buf, part);
            for (unsigned k = threadIdx.x; k < K; k += PLOW_NV_THREADS)
                sv[k] = __uint_as_float((unsigned)L[sel[k]] << 16);
            __syncthreads();
        } else {
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
 * in blocks of 8 (W re-read once per block of rows). K % 8 == 0. `round_bf16` (flag bit 11): C rounded to bf16.
 * When all of x fits the launch's arena, a block stages it once (every warp re-read it from L2
 * per output row otherwise); the arithmetic is the same either way. */
static __device__ void d_gemv_f32out(float* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ W, unsigned M, unsigned N, unsigned K,
                                     bool round_bf16, unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);
#if GV_DEPTH_FAST
    if (n0 >= n1) return;
    __nv_bfloat16* xs = (__nv_bfloat16*)arena;
    if ((K & 7u) == 0 && ((uintptr_t)x & 15u) == 0 && smem_fits(xs + (size_t)M * K)) {
        for (size_t i = (size_t)threadIdx.x * 8u; i < (size_t)M * K; i += PLOW_NV_THREADS * 8u)
            *(uint4*)(xs + i) = *(const uint4*)(x + i);
        __syncthreads();
        x = xs;
    }
#else
    (void)arena;
#endif
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
                if (lane == 0 && m0 + m < M)
                    C[(size_t)(m0 + m) * N + n] = round_bf16 ? __bfloat162float(__float2bfloat16_rn(t)) : t;
            }
        }
    }
}

/* GemvF32Out modes (i7 != 0; Maya 2's depth projections): bit 0 x is f32, bit 1 W is f16, bit 2 adds
 * an f32 bias t3[N] (bit 9: t3[M / bgroup][N], one row per group of bgroup rows), bit 3 adds an f32
 * residual t4[M][N] (may alias C: each element is read then written by the one lane that owns it),
 * bits 4-7 the activation applied last (1 = GELU erf). */
template <bool XF, bool WH>
static __device__ void gemv_f32out_mode(float* C, const void* xv, const void* Wv, const float* bias,
                                        unsigned bgroup, const float* res, unsigned act, unsigned M,
                                        unsigned N, unsigned K, unsigned slice, unsigned nblk) {
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
                    if (bias) v = __fadd_rn(v, bias[(bgroup ? (m0 + m) / bgroup * N : 0u) + n]);
                    if (res) v = __fadd_rn(v, res[o]);
                    if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                    C[o] = v;
                }
            }
        }
    }
}

/* f32 x, f16 W (modes 3): each block stages 8 rows x GEMV_KT columns of x in shared memory once and
 * every warp reuses them for up to CPW columns (x read from L2 once per block, not once per
 * column). With the LayerNorm prologue (bit 8, K <= GEMV_KT) the staged rows are normalised first:
 * fp32 two-pass mean / variance, `(x - mean) * rsqrt(var + eps) * gamma + beta`.
 * Arena: 8 * GEMV_KT + 16 floats. */
#define GEMV_KT 1024u
template <unsigned CPW>
static __device__ void gemv_f32x_f16w_staged(float* C, const float* __restrict__ x, const unsigned short* __restrict__ W,
                                             const float* bias, unsigned bgroup, const float* res, const float* gamma,
                                             const float* beta, float eps, unsigned act, unsigned M, unsigned N,
                                             unsigned K, unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);
    float* xs = arena;
    float* st = arena + 8u * GEMV_KT;
    for (unsigned cb = n0; cb < n1; cb += PLOW_NV_WARPS * CPW) {
        for (unsigned m0 = 0; m0 < M; m0 += 8) {
            const unsigned mr = min(8u, M - m0);
            float acc[CPW][8];
#pragma unroll
            for (unsigned j = 0; j < CPW; j++)
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
                /* Every column's 16 B of the next k step are in flight while this step's FMAs run. */
                uint4 wn[CPW];
                const auto load = [&](uint4* w, unsigned k) {
#pragma unroll
                    for (unsigned j = 0; j < CPW; j++) {
                        const unsigned n = cb + warp + j * PLOW_NV_WARPS;
                        w[j] = n < n1 && k < kw ? *(const uint4*)(W + (size_t)n * K + kt + k) : make_uint4(0u, 0u, 0u, 0u);
                    }
                };
                load(wn, lane * 8u);
                for (unsigned k = lane * 8u; k < kw; k += 32u * 8u) {
                    uint4 wr[CPW];
#pragma unroll
                    for (unsigned j = 0; j < CPW; j++) wr[j] = wn[j];
                    load(wn, k + 32u * 8u);
#pragma unroll
                    for (int m = 0; m < 8; m++) {
                        if ((unsigned)m >= mr) break;
                        const float4* xp = (const float4*)(xs + m * GEMV_KT + k);
                        const float4 a = xp[0], b = xp[1];
#pragma unroll
                        for (unsigned j = 0; j < CPW; j++) {
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
            for (unsigned j = 0; j < CPW; j++) {
                const unsigned n = cb + warp + j * PLOW_NV_WARPS;
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    if ((unsigned)m >= mr) break;
                    const float t = warp_sum32(acc[j][m]);
                    if (lane == 0 && n < n1) {
                        const size_t o = (size_t)(m0 + m) * N + n;
                        float v = t;
                        if (bias) v = __fadd_rn(v, bias[(bgroup ? (m0 + m) / bgroup * N : 0u) + n]);
                        if (res) v = __fadd_rn(v, res[o]);
                        if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                        C[o] = v;
                    }
                }
            }
        }
    }
}


/* Tensor-core GemvF32Out (modes 3 | bit 10; the depth stage's multi-row passes): C^T = W x^T with
 * mma.m16n8k16 (f16 x f16 -> f32). x is rounded to f16 after the LayerNorm prologue, as the
 * reference serves its depth stage in fp16. A warp owns 16 output columns (W rows) over up to 64
 * x rows (8 n8 tiles), so W streams once per 64 rows; warps of a block that would idle split K and
 * reduce through shared memory. Inside each 64-wide K block, lane t of a quad reads the contiguous
 * halves [16t, 16t + 16) of its W and x rows: logical k (2t + e, 2t + 8 + e) of k16 step s is
 * physical 16t + 4s + (e, 2 + e), one bijection for A and B, so the sum is unchanged.
 * K % 64 == 0. Arena: 64 x MMA_KT halves of x, 128 floats of row statistics, 8 x 32 x 32 floats
 * of split-K partials. */
#define MMA_KT 256u
#define MMA_ROWS 64u
static __device__ __forceinline__ void mma_16816(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/* Small-M arm of gemv_f16w_mma (M <= 8 * NT rows, round_up(M, 8) * K <= MMA_SMALL_X halves, K % 256 == 0):
 * every x row is staged once (LN applied, f16), and a warp streams its K pieces over the whole K with
 * up to three pieces of W in flight instead of one piece per 256-wide chunk behind a block barrier.
 * The piece -> warp chain and the cross-warp reduction are those of the chunked arm (piece p of 64
 * belongs to warp my_ks iff p % min(ks, 4) == my_ks, pieces in increasing order), so the result is
 * bit-identical to it. Arena: MMA_SMALL_X halves of x, 128 floats of statistics, 8 x 32 x 4 NT floats
 * of partials. */
#define MMA_SMALL_X 24576u
template <unsigned NT>
static __device__ void gemv_f16w_mma_small(float* C, const float* __restrict__ x,
                                           const unsigned short* __restrict__ W, const float* bias,
                                           unsigned bgroup, const float* res, const float* gamma,
                                           const float* beta, float eps, unsigned act, unsigned M, unsigned N,
                                           unsigned K, unsigned t0, unsigned tiles, unsigned ks, float* arena) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned my_tile = warp / ks, my_ks = warp % ks;
    const unsigned step = min(ks, 4u);
    const bool active = my_tile < tiles, compute = active && my_ks < step;
    __half* xs = (__half*)arena;
    float* st = arena + MMA_SMALL_X / 2u;
    float* red = st + 2u * MMA_ROWS;
    const unsigned na = (t0 + my_tile) * 16u + g, nb = na + 8u;
    __syncthreads();
    if (gamma) {
        for (unsigned r = warp; r < M; r += PLOW_NV_WARPS) {
            const float* xr = x + (size_t)(r) * K;
            float s = 0.f;
            for (unsigned k = lane; k < K; k += 32u) s += xr[k];
            const float mean = warp_sum32(s) / (float)K;
            float v = 0.f;
            for (unsigned k = lane; k < K; k += 32u) {
                const float d = xr[k] - mean;
                v = __fmaf_rn(d, d, v);
            }
            const float var = warp_sum32(v) / (float)K;
            if (lane == 0) {
                st[2 * r] = mean;
                st[2 * r + 1] = rsqrtf(var + eps);
            }
        }
        __syncthreads();
    }
    for (unsigned r = 0; r < NT * 8u; r++) {
        const float* xr = r < M ? x + (size_t)(r) * K : nullptr;
        for (unsigned k = threadIdx.x; k < K; k += PLOW_NV_THREADS) {
            float v = 0.f;
            if (xr) {
                v = xr[k];
                if (gamma) v = __fmaf_rn((v - st[2 * r]) * st[2 * r + 1], gamma[k], beta[k]);
            }
            xs[r * K + k] = __float2half_rn(v);
        }
    }
    __syncthreads();
    float acc[NT][4];
#pragma unroll
    for (unsigned j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    if (compute) {
        const unsigned npieces = K / 64u;
        for (unsigned p0 = my_ks; p0 < npieces; p0 += 3u * step) {
            uint4 w[3][4];
#pragma unroll
            for (unsigned q = 0; q < 3u; q++) {
                const unsigned p = p0 + q * step;
                const unsigned k0 = p * 64u + 16u * t;
                if (p < npieces && na < N) {
                    w[q][0] = *(const uint4*)(W + (size_t)na * K + k0);
                    w[q][1] = *(const uint4*)(W + (size_t)na * K + k0 + 8u);
                } else {
                    w[q][0] = w[q][1] = make_uint4(0u, 0u, 0u, 0u);
                }
                if (p < npieces && nb < N) {
                    w[q][2] = *(const uint4*)(W + (size_t)nb * K + k0);
                    w[q][3] = *(const uint4*)(W + (size_t)nb * K + k0 + 8u);
                } else {
                    w[q][2] = w[q][3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
#pragma unroll
            for (unsigned q = 0; q < 3u; q++) {
                const unsigned p = p0 + q * step;
                if (p >= npieces) break;
                const unsigned* pa = (const unsigned*)&w[q][0];
                const unsigned* pb = (const unsigned*)&w[q][2];
#pragma unroll
                for (unsigned s = 0; s < 4u; s++) {
                    const unsigned a[4] = {pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1]};
#pragma unroll
                    for (unsigned j = 0; j < NT; j++) {
                        const unsigned* xr = (const unsigned*)(xs + (j * 8u + g) * K + p * 64u + 16u * t + 4u * s);
                        mma_16816(acc[j], a, xr[0], xr[1]);
                    }
                }
            }
        }
    }
    if (ks > 1u) {
        __syncthreads();
        if (active && my_ks) {
#pragma unroll
            for (unsigned j = 0; j < NT; j++)
#pragma unroll
                for (unsigned c = 0; c < 4u; c++) red[(warp * 32u + lane) * (4u * NT) + j * 4u + c] = acc[j][c];
        }
        __syncthreads();
        if (active && !my_ks) {
            for (unsigned q = 1; q < ks; q++)
#pragma unroll
                for (unsigned j = 0; j < NT; j++)
#pragma unroll
                    for (unsigned c = 0; c < 4u; c++)
                        acc[j][c] += red[((warp + q) * 32u + lane) * (4u * NT) + j * 4u + c];
        }
    }
    if (active && !my_ks) {
#pragma unroll
        for (unsigned j = 0; j < NT; j++) {
#pragma unroll
            for (unsigned c = 0; c < 4u; c++) {
                const unsigned m = j * 8u + 2u * t + (c & 1u), n = (c < 2u) ? na : nb;
                if (m >= M || n >= N) continue;
                float v = acc[j][c];
                if (bias) v = __fadd_rn(v, bias[(bgroup ? m / bgroup * N : 0u) + n]);
                if (res) v = __fadd_rn(v, res[(size_t)(m) * N + n]);
                if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                C[(size_t)(m) * N + n] = v;
            }
        }
    }
}

static __device__ void gemv_f16w_mma(float* C, const float* __restrict__ x, const unsigned short* __restrict__ W,
                                     const float* bias, unsigned bgroup, const float* res, const float* gamma,
                                     const float* beta, float eps, unsigned act, unsigned M, unsigned N, unsigned K,
                                     unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned ntiles = (N + 15u) / 16u;
    const unsigned per = (ntiles + nblk - 1) / nblk;
    const unsigned t0 = slice * per;
    const unsigned tiles = t0 < ntiles ? min(per, ntiles - t0) : 0u;
    if (!tiles) return;
    unsigned ks = max(1u, PLOW_NV_WARPS / tiles);
    ks = min(ks, K / 64u);
    if (M && M <= 24u && K % 256u == 0u && (size_t)((M + 7u) / 8u) * 8u * K <= MMA_SMALL_X) {
        if (M <= 8u)
            gemv_f16w_mma_small<1>(C, x, W, bias, bgroup, res, gamma, beta, eps, act, M, N, K, t0, tiles, ks, arena);
        else if (M <= 16u)
            gemv_f16w_mma_small<2>(C, x, W, bias, bgroup, res, gamma, beta, eps, act, M, N, K, t0, tiles, ks, arena);
        else
            gemv_f16w_mma_small<3>(C, x, W, bias, bgroup, res, gamma, beta, eps, act, M, N, K, t0, tiles, ks, arena);
        return;
    }
    const unsigned my_tile = warp / ks, my_ks = warp % ks;
    const bool active = my_tile < tiles;
    __half* xs = (__half*)arena;
    float* st = arena + MMA_ROWS * MMA_KT / 2u;
    float* red = st + 2u * MMA_ROWS;
    const unsigned na = (t0 + my_tile) * 16u + g, nb = na + 8u;
#if PLOW_NV_SPEECH
    /* The quad staging loads float4s: x, gamma and beta bases must be 16-byte aligned (row and chunk offsets keep
     * it); else one element per trip. Speech objects only: every other object keeps its staging text. */
    const bool vec_ok =
        ((((size_t)x) | (gamma ? ((size_t)gamma | (size_t)beta) : (size_t)0)) & (size_t)15) == 0;
#endif
    for (unsigned mc = 0; mc < M; mc += MMA_ROWS) {
        const unsigned mr = min(MMA_ROWS, M - mc), nt8 = (mr + 7u) / 8u;
        __syncthreads();
        if (gamma) {
            for (unsigned r = warp; r < mr; r += PLOW_NV_WARPS) {
                const float* xr = x + (size_t)(mc + r) * K;
                float s = 0.f;
                for (unsigned k = lane; k < K; k += 32u) s += xr[k];
                const float mean = warp_sum32(s) / (float)K;
                float v = 0.f;
                for (unsigned k = lane; k < K; k += 32u) {
                    const float d = xr[k] - mean;
                    v = __fmaf_rn(d, d, v);
                }
                const float var = warp_sum32(v) / (float)K;
                if (lane == 0) {
                    st[2 * r] = mean;
                    st[2 * r + 1] = rsqrtf(var + eps);
                }
            }
        }
        float acc[MMA_ROWS / 8u][4];
#pragma unroll
        for (unsigned j = 0; j < MMA_ROWS / 8u; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        for (unsigned kt = 0; kt < K; kt += MMA_KT) {
            const unsigned kw = min(MMA_KT, K - kt);
            __syncthreads();
#if PLOW_NV_SPEECH
            /* Thread (rg, c) stages the quad c of rows rg, rg + 4, ..: float4 loads, several rows in flight.
             * (One element per thread and trip left the M >= 32 rungs bound by this loop's load latency.) */
            if (vec_ok) {
                const unsigned c = threadIdx.x & 63u, k = 4u * c;
                if (k < kw) {
                    const float4 gq = gamma ? *(const float4*)(gamma + kt + k) : make_float4(0.f, 0.f, 0.f, 0.f);
                    const float4 bq = gamma ? *(const float4*)(beta + kt + k) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
                    for (unsigned r = threadIdx.x >> 6; r < nt8 * 8u; r += PLOW_NV_THREADS / 64u) {
                        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                        if (r < mr) {
                            v = *(const float4*)(x + (size_t)(mc + r) * K + kt + k);
                            if (gamma) {
                                const float mu = st[2 * r], rs = st[2 * r + 1];
                                v.x = __fmaf_rn((v.x - mu) * rs, gq.x, bq.x);
                                v.y = __fmaf_rn((v.y - mu) * rs, gq.y, bq.y);
                                v.z = __fmaf_rn((v.z - mu) * rs, gq.z, bq.z);
                                v.w = __fmaf_rn((v.w - mu) * rs, gq.w, bq.w);
                            }
                        }
                        __half2* d = (__half2*)(xs + r * MMA_KT + k);
                        d[0] = __floats2half2_rn(v.x, v.y);
                        d[1] = __floats2half2_rn(v.z, v.w);
                    }
                }
            } else
#endif
            for (unsigned i = threadIdx.x; i < nt8 * 8u * kw; i += PLOW_NV_THREADS) {
                const unsigned r = i / kw, k = i % kw;
                float v = 0.f;
                if (r < mr) {
                    v = x[(size_t)(mc + r) * K + kt + k];
                    if (gamma) v = __fmaf_rn((v - st[2 * r]) * st[2 * r + 1], gamma[kt + k], beta[kt + k]);
                }
                xs[r * MMA_KT + k] = __float2half_rn(v);
            }
            __syncthreads();
            if (!active) continue;
            for (unsigned kb = my_ks; kb < kw / 64u; kb += ks) {
                const unsigned k0 = kt + kb * 64u + 16u * t;
                uint4 wa[2], wb[2];
                if (na < N) {
                    wa[0] = *(const uint4*)(W + (size_t)na * K + k0);
                    wa[1] = *(const uint4*)(W + (size_t)na * K + k0 + 8u);
                } else {
                    wa[0] = wa[1] = make_uint4(0u, 0u, 0u, 0u);
                }
                if (nb < N) {
                    wb[0] = *(const uint4*)(W + (size_t)nb * K + k0);
                    wb[1] = *(const uint4*)(W + (size_t)nb * K + k0 + 8u);
                } else {
                    wb[0] = wb[1] = make_uint4(0u, 0u, 0u, 0u);
                }
                const unsigned* pa = (const unsigned*)wa;
                const unsigned* pb = (const unsigned*)wb;
#pragma unroll
                for (unsigned s = 0; s < 4u; s++) {
                    const unsigned a[4] = {pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1]};
#pragma unroll
                    for (unsigned j = 0; j < MMA_ROWS / 8u; j++) {
                        if (j >= nt8) break;
                        const unsigned* xr =
                            (const unsigned*)(xs + (j * 8u + g) * MMA_KT + kb * 64u + 16u * t + 4u * s);
                        mma_16816(acc[j], a, xr[0], xr[1]);
                    }
                }
            }
        }
        if (ks > 1u) {
            __syncthreads();
            if (active && my_ks) {
#pragma unroll
                for (unsigned j = 0; j < MMA_ROWS / 8u; j++)
#pragma unroll
                    for (unsigned c = 0; c < 4u; c++) red[(warp * 32u + lane) * 32u + j * 4u + c] = acc[j][c];
            }
            __syncthreads();
            if (active && !my_ks) {
                for (unsigned q = 1; q < ks; q++)
#pragma unroll
                    for (unsigned j = 0; j < MMA_ROWS / 8u; j++)
#pragma unroll
                        for (unsigned c = 0; c < 4u; c++) acc[j][c] += red[((warp + q) * 32u + lane) * 32u + j * 4u + c];
            }
        }
        if (active && !my_ks) {
#pragma unroll
            for (unsigned j = 0; j < MMA_ROWS / 8u; j++) {
                if (j >= nt8) break;
#pragma unroll
                for (unsigned c = 0; c < 4u; c++) {
                    const unsigned m = j * 8u + 2u * t + (c & 1u), n = (c < 2u) ? na : nb;
                    if (m >= mr || n >= N) continue;
                    const size_t o = (size_t)(mc + m) * N + n;
                    float v = acc[j][c];
                    if (bias) v = __fadd_rn(v, bias[(bgroup ? (mc + m) / bgroup * N : 0u) + n]);
                    if (res) v = __fadd_rn(v, res[o]);
                    if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
                    C[o] = v;
                }
            }
        }
    }
}

#if PLOW_NV_SPEECH && defined(PLOW_NV_HOPPER)
#include "op_gemv_f16_sm90.cuh"
#endif

static __device__ void d_gemv_f32out_mode(float* C, const void* x, const void* W, const float* bias,
                                          const float* res, const float* gamma, const float* beta, float eps,
                                          unsigned flags, unsigned bgroup, unsigned M, unsigned N, unsigned K,
                                          unsigned slice, unsigned nblk, float* arena, const unsigned* vd) {
    /* Row views (flag bit 12, descriptor vd) are not compiled in: plow_speech_caps bit 0 is clear and the
     * loader refuses such a packet before launch. */
    (void)vd;
    if (flags & 4096u) { __trap(); return; }
    const unsigned act = (flags >> 4) & 15u;
    bgroup = (flags & 512u) ? bgroup : 0u;
    if ((flags & 512u) && !bgroup) { __trap(); return; }
    bias = (flags & 4u) ? bias : nullptr;
    res = (flags & 8u) ? res : nullptr;
#if PLOW_NV_SPEECH && defined(PLOW_NV_HOPPER)
    if ((flags & 16384u) && !(flags & 8192u)) { __trap(); return; }
    if (flags & 8192u) {
        if ((flags & (3u | 1024u | 256u)) != (3u | 1024u) || M < 16u || N % 64u || K % 64u ||
            ((((size_t)x) | ((size_t)W)) & 15u) || ((flags & 16384u) && (const void*)C == (const void*)res)) {
            __trap(); return;
        }
        const auto* h = (const unsigned short*)x;
        if (flags & 16384u) {
            if (M >= 256u && K % 128u == 0u) gemv_f16w_half<128, true, 128>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if (M >= 256u) gemv_f16w_half<128, true>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if (N <= 1024u) gemv_f16w_half<16, true>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if ((size_t)((M + 63u) / 64u) * (N / 64u) >= 96u)
                gemv_f16w_half<64, true>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else gemv_f16w_half<32, true>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
        } else {
            if (M >= 256u && K % 128u == 0u) gemv_f16w_half<128, false, 128>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if (M >= 256u) gemv_f16w_half<128, false>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if (N <= 1024u) gemv_f16w_half<16, false>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else if ((size_t)((M + 63u) / 64u) * (N / 64u) >= 96u)
                gemv_f16w_half<64, false>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
            else gemv_f16w_half<32, false>(C, h, (const unsigned short*)W, bias, bgroup, res, act, M, N, K, slice, nblk, arena);
        }
        return;
    }
#endif
    if ((flags & 3u) == 3u && (flags & 1024u)) {
        if (K % 64u) { __trap(); return; }
#if PLOW_NV_SPEECH && defined(PLOW_NV_HOPPER)
        if (M >= 16u && N % 64u == 0u && K >= 192u && K <= 4096u &&
            ((((size_t)x) | ((size_t)W) | (gamma ? ((size_t)gamma | (size_t)beta) : 0u)) & 15u) == 0u) {
            if ((size_t)((M + 63u) / 64u) * (N / 64u) >= 96u && (!(flags & 256u) || K <= 1024u)) {
                gemv_f16w_tiled<64>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res,
                                  (flags & 256u) ? gamma : nullptr, beta, eps, act, M, N, K, slice, nblk, arena);
            } else {
                gemv_f16w_tiled<32>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res,
                                  (flags & 256u) ? gamma : nullptr, beta, eps, act, M, N, K, slice, nblk, arena);
            }
            return;
        }
#endif
        gemv_f16w_mma(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res,
                      (flags & 256u) ? gamma : nullptr, beta, eps, act, M, N, K, slice, nblk, arena);
        return;
    }
    if ((flags & 3u) == 3u) {
        if ((flags & 256u) && K > GEMV_KT) { __trap(); return; }
        /* The fewest columns per warp that still give every warp of the block its share: a
         * narrow projection (N / blocks <= 8 per warp) keeps all warps streaming W. */
        const unsigned per = (N + nblk - 1) / nblk, need = (per + PLOW_NV_WARPS - 1) / PLOW_NV_WARPS;
        const float* g = (flags & 256u) ? gamma : nullptr;
        if (need <= 1)
            gemv_f32x_f16w_staged<1>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res, g, beta, eps,
                                     act, M, N, K, slice, nblk, arena);
        else if (need <= 2)
            gemv_f32x_f16w_staged<2>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res, g, beta, eps,
                                     act, M, N, K, slice, nblk, arena);
        else if (need <= 4)
            gemv_f32x_f16w_staged<4>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res, g, beta, eps,
                                     act, M, N, K, slice, nblk, arena);
        else
            gemv_f32x_f16w_staged<8>(C, (const float*)x, (const unsigned short*)W, bias, bgroup, res, g, beta, eps,
                                     act, M, N, K, slice, nblk, arena);
        return;
    }
    switch (flags & 3u) {
    case 0: gemv_f32out_mode<false, false>(C, x, W, bias, bgroup, res, act, M, N, K, slice, nblk); break;
    case 1: gemv_f32out_mode<true, false>(C, x, W, bias, bgroup, res, act, M, N, K, slice, nblk); break;
    case 2: gemv_f32out_mode<false, true>(C, x, W, bias, bgroup, res, act, M, N, K, slice, nblk); break;
    default: gemv_f32out_mode<true, true>(C, x, W, bias, bgroup, res, act, M, N, K, slice, nblk); break;
    }
}

/* SampleSmallF32 (op 221): one block per row of f32 logits; see dev.rs. Arena: PLOW_NV_WARPS u64 +
 * PLOW_NV_WARPS f32 reduction slots, 64 selected classes and values, the row (f32), then the top-k
 * scratch (block_topk_u64). */
static __device__ __forceinline__ unsigned long long f32_key(float f, unsigned i) {
    const unsigned b = __float_as_uint(f + 0.f); /* -0 ranks as +0 (the golden's comparison) */
    const unsigned key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~i);
}
#if GV_GUMBEL_WARP
/* argmax over i < n of sv[i] + sf_neg_log_exp1(seed, step, coord, 0, i) by one full warp: lanes scan
 * i = lane, lane + 32, ... with strict >, then (max score, lowest i) across lanes = a serial strict->
 * scan's pick; nothing above -inf (or NaN) picks 0. */
static __device__ __forceinline__ unsigned gumbel_pick_warp(const float* sv, unsigned n, unsigned long long seed,
                                                            unsigned step, unsigned coord) {
    unsigned best = 0xFFFFFFFFu;
    float bv = -INFINITY;
    for (unsigned i = threadIdx.x & 31u; i < n; i += 32u) {
        const float v = sv[i] + sf_neg_log_exp1(seed, step, coord, 0u, i);
        if (v > bv) { bv = v; best = i; }
    }
#pragma unroll
    for (unsigned d = 16; d; d >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, d);
        const unsigned ob = __shfl_xor_sync(0xffffffffu, best, d);
        if (ov > bv || (ov == bv && ob < best)) { bv = ov; best = ob; }
    }
    return best == 0xFFFFFFFFu ? 0u : best;
}
#endif
static __device__ void d_sample_small(unsigned* __restrict__ out, const float* __restrict__ logits,
                                      const float* __restrict__ params, const unsigned* __restrict__ ctl,
                                      const unsigned long long* __restrict__ seed, unsigned* __restrict__ bitmap,
                                      const unsigned* __restrict__ force, unsigned* __restrict__ eidx,
                                      unsigned rows, unsigned vocab, unsigned out_stride, unsigned out_col0,
                                      unsigned coord, unsigned words, unsigned lstride, unsigned eoff,
                                      unsigned min_step, unsigned slice, unsigned nblk, float* arena) {
    unsigned long long* part = (unsigned long long*)arena;
    float* fpart = (float*)(part + PLOW_NV_WARPS);
    unsigned* sel = (unsigned*)(fpart + PLOW_NV_WARPS);
    float* sv = (float*)(sel + 64);
    float* L = sv + 64;
#if !defined(PLOW_HAS_SAMPLE_SMALL_F32) || PLOW_HAS_SAMPLE_SMALL_F32
    if (!smem_fits(L + vocab)) { asm volatile("trap;"); return; }
#endif
    unsigned long long* tk_buf = (unsigned long long*)(((size_t)(L + vocab) + 15u) & ~(size_t)15u);
    unsigned* tk_ctl = (unsigned*)(tk_buf + 64);
    const bool radix = PLOW_NV_SAMPLE_RADIX && smem_fits(tk_ctl + 4 + 256);
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
        /* HF min_new_tokens: the last class (the stop) is masked before step min_step. */
        const unsigned masked = ctl[2u * r + 1u] < min_step ? vocab - 1u : vocab;
        for (unsigned i = threadIdx.x; i < vocab; i += PLOW_NV_THREADS) {
            float v = lg[i];
            if (lu) v = __fadd_rn(lu[i], __fmul_rn(cs, __fsub_rn(v, lu[i])));
            if (i == masked) v = -INFINITY;
            if (bm && ((bm[i >> 5] >> (i & 31u)) & 1u)) v = v > 0.f ? __fdiv_rn(v, pen) : v * pen;
            L[i] = t > 0.f ? __fdiv_rn(v, t) : v;
        }
        __syncthreads();
        if (radix) {
            block_topk_u64([L](unsigned i) { return f32_key(L[i], i); }, vocab, K, 56, sel, tk_ctl,
                           tk_ctl + 4, tk_buf, part);
            for (unsigned k = threadIdx.x; k < K; k += PLOW_NV_THREADS) sv[k] = L[sel[k]];
            __syncthreads();
        } else {
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
        }
#if GV_GUMBEL_WARP
        if (threadIdx.x < 32u) {
            const unsigned lane = threadIdx.x;
#else
        if (threadIdx.x == 0) {
            const unsigned lane = 0;
#endif
            const unsigned step = ctl[2u * r + 1u];
            unsigned n = K;
            if (p[1] < 1.f && lane == 0) {
                float z = 0.f;
                for (unsigned i = 0; i < K; i++) z += expf(sv[i] - sv[0]);
                float cum = 0.f;
                for (unsigned i = 0; i < K; i++) {
                    if (i > 0 && cum > p[1]) { n = i; break; }
                    cum += expf(sv[i] - sv[0]) / z;
                }
            }
#if GV_GUMBEL_WARP
            n = __shfl_sync(0xffffffffu, n, 0);
            const unsigned best = gumbel_pick_warp(sv, n, seed[r], step, coord);
#else
            unsigned best = 0;
            float bv = -INFINITY;
            for (unsigned i = 0; i < n; i++) {
                const float v = sv[i] + sf_neg_log_exp1(seed[r], step, coord, 0u, i);
                if (v > bv) { bv = v; best = i; }
            }
#endif
            if (lane == 0) {
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
                out[j] = __float2bfloat16(a * __bfloat162float(up[j]));
            }
        }
    }
}

/* Op 230: d_glu over one row-pitched gate|up input (an N-concatenated projection's output),
 * per element d_glu's vector-path math: out[r*width + p] = bf16(act(g) * u),
 * g = gu[r*in_stride + p], u = gu[r*in_stride + up_col + p]. */
/* GluPitched (an N-concatenated gate|up's GLU) compiles only for a packet that emits it: the
 * unexecuted arms cost Gemma-4-E4B decode B1 +2.1% (interp_sm90a.cubin +2.2%, light_pf 72 -> 73
 * registers). Without a packet config (the general objects, the harnesses) it stays. A packet
 * that emits the op with its define at 0 is refused at emit, and the gated-out case traps. */
#ifndef GLU_PITCHED_ARMS
#if defined(PLOW_PACKET_HAS_GLU_PITCHED) && !PLOW_PACKET_HAS_GLU_PITCHED
#define GLU_PITCHED_ARMS 0
#else
#define GLU_PITCHED_ARMS 1
#endif
#endif
#if GLU_PITCHED_ARMS
static __device__ __forceinline__ float glu_pitched_one(float g, float u, unsigned act) {
    float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh_pf(g);
    return a * u;
}
static __device__ void d_glu_pitched(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gu,
                                     unsigned n, unsigned act, unsigned width, unsigned in_stride,
                                     unsigned up_col, unsigned slice, unsigned nblk) {
    if (((width | in_stride | up_col) & 7u) == 0) {
        const unsigned n8 = n >> 3, w8 = width >> 3, step = nblk * PLOW_NV_THREADS;
        auto src = [&](unsigned v) {
            const unsigned r = v / w8;
            return gu + (size_t)r * in_stride + (v - r * w8) * 8;
        };
        auto one = [&](const bf16v8& g, const bf16v8& u) {
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++)
                o.x[j] = __float2bfloat16(glu_pitched_one(__bfloat162float(g.x[j]), __bfloat162float(u.x[j]), act));
            return o;
        };
        /* Four gate/up pairs loaded before any is consumed, as d_glu. */
        unsigned v = slice * PLOW_NV_THREADS + threadIdx.x;
        for (; v + 3u * step < n8; v += 4u * step) {
            bf16v8 g[4], u[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const __nv_bfloat16* p = src(v + (unsigned)k * step);
                g[k] = ld_glob8(p);
                u[k] = ld_glob8(p + up_col);
            }
#pragma unroll
            for (int k = 0; k < 4; k++) st_glob8(out + (size_t)(v + (unsigned)k * step) * 8, one(g[k], u[k]));
        }
        for (; v < n8; v += step) {
            const __nv_bfloat16* p = src(v);
            st_glob8(out + (size_t)v * 8, one(ld_glob8(p), ld_glob8(p + up_col)));
        }
        return;
    }
    for (unsigned i = slice * PLOW_NV_THREADS + threadIdx.x; i < n; i += nblk * PLOW_NV_THREADS) {
        const unsigned r = i / width, p = i - r * width;
        const __nv_bfloat16* src = gu + (size_t)r * in_stride + p;
        out[i] = __float2bfloat16(glu_pitched_one(__bfloat162float(src[0]), __bfloat162float(src[up_col]), act));
    }
}
#endif /* GLU_PITCHED_ARMS */

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
