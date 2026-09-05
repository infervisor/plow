/* op_norm.cuh — RMSNorm family for sm_120 (warp32).
 *
 * Ported from runtime/amd/op_norm.h. The AMD original is MFMA-free, so this is a pure
 * wave64 -> warp32 re-derivation plus the buffer-descriptor removal (NVIDIA has no
 * __amdgpu_buffer_rsrc_t / hardware num_records bound, so every overshoot is an explicit
 * predicate here).
 *
 * NUMERICS (unchanged from AMD): y = x * rsqrt(mean(x^2) + eps) * gamma, eps INSIDE the
 * power, gamma a PLAIN multiply (no 1+w). gamma == nullptr is the weightless variant.
 *
 * d_headnorm_rope IS WRITTEN FRESH, not adapted from gemma_normrope_sm120 — that kernel is
 * silently wrong for head_dim != 128 (it holds the head in a thread-local `float nv[FA_HD]`
 * and the rotate's nv[i+half] is lane-local only at hd 128). See the LANE-LOCALITY note on
 * the template below for why the layout used here is correct at 128/256/512 alike.
 */
#pragma once
#ifndef PLOW_NV_GEMMA_HNR_BF16
#define PLOW_NV_GEMMA_HNR_BF16 0
#endif
#ifndef PLOW_NV_GEMMA_NRN_BF16
#define PLOW_NV_GEMMA_NRN_BF16 0
#endif
#ifndef PLOW_NV_NRN_WPR
#define PLOW_NV_NRN_WPR 0
#endif
#include "sm120_common.cuh"
#include <cuda_fp8.h> /* __nv_fp8_e4m3 — the T11 fused activation quant */

static __device__ __forceinline__ float gemma_postnorm_round(float x) {
#if defined(PLOW_NV_GEMMA) && PLOW_NV_GEMMA && PLOW_NV_GEMMA_NRN_BF16
    return __bfloat162float(__float2bfloat16(x));
#else
    return x;
#endif
}

/* Elements one thread holds when the row fits in registers, as 16-byte vector loads.
 * RN_REG * 256 threads = 6144 covers every decode hidden the family uses: Qwen3 2560,
 * Gemma-12B 3840, Gemma-26B 2816, and Gemma-31B 5376. At the old RN_REG=16 the bound was
 * 4096, so 31B (5376 > 4096) fell to the scalar slow path that reads x/a/b from HBM TWICE
 * per norm — a live decode-latency cost since 31B runs a norm on every sublayer. 5376 is
 * 8-aligned (672*8), so it takes the vector register path here. */
#ifndef RN_REG
#define RN_REG 24
#endif
#define RN_VEC (RN_REG / 8)

/* T17 warp-per-row cut-in (see the note on d_rmsnorm). Above the widest decode batch the
 * emitter mints (`PLOW_DECODE_BATCH` 1..8, clamped to 16), below any prefill chunk — so the
 * new reduction order applies to prefill only and batched decode stays byte-identical. */
#define PLOW_NV_T17_MIN_ROWS 32u

/* RMSNorm over `feat`. One block per row, strided by nblk.
 *
 * PRODUCER-THEN-CONSUMER is preserved from the AMD original and is not cosmetic: a decode
 * norm is ONE row on ONE block while every other block spins on its counter, so the op's
 * cost is pure load latency. Both x and gamma are issued before either is waited on — one
 * round trip, not two.
 *
 * i2=out_row0 offsets the OUTPUT row only (input stays at `base`); 0 on every Qwen path.
 *
 * T17 WARP-PER-ROW (rows >= PLOW_NV_T17_MIN_ROWS && feat%8==0, i.e. every prefill chunk): the
 * block-per-row body is LATENCY bound, not bandwidth bound — 15+ sequential rows per block,
 * each paying two block_sum barriers (measured: the fat class runs ~16x under the HBM
 * roofline). Eight rows in flight per block, warp reductions, zero barriers. The f32
 * accumulation order changes (lane-strided vs thread-strided), so outputs can differ in the
 * last bf16 ulp from the legacy body — DECODE keeps the legacy path and stays byte-identical.
 *
 * That last guarantee is why the threshold is NOT PLOW_NV_WARPS: `rows` is packet i[0], which
 * decode sets to its batch width (`dbatch`), so a gate at 8 silently moved batched decode onto
 * the new reduction order at the shipped PLOW_DECODE_BATCH=8. The threshold sits above the
 * widest decode batch the emitter will mint (documented 1..8, clamped to 16) and far below any
 * prefill chunk (smallest bucket T is hundreds of rows), so it partitions the two phases
 * exactly.
 *
 * T11 w8a8 QUANT FUSION (t3=xq e4m3, t4=ascale f32[rows]; both null on legacy packets):
 * the row is already normed in registers, so the per-row fp8 activation quant that would
 * otherwise re-read it from HBM (a whole extra packet + gate) rides here. The quantized
 * value is computed FROM THE bf16-ROUNDED output — exactly the value d_quant_fp8 would have
 * read back — so fused and unfused paths are token-identical, not merely close. */
static __device__ void d_rmsnorm(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ gamma, unsigned rows, unsigned feat,
                          float eps, unsigned out_row0, unsigned slice, unsigned nblk,
                          float* part, uint8_t* __restrict__ xq = nullptr,
                          float* __restrict__ ascale = nullptr) {
    if (rows >= PLOW_NV_T17_MIN_ROWS && (feat & 7u) == 0) {
        /* T17 warp-per-row (see header comment). Row set of this block is unchanged:
         * {slice + k*nblk}; warp w takes k ≡ w (mod WARPS). */
        const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
        const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
        for (unsigned k = warp;; k += PLOW_NV_WARPS) {
            const unsigned row = slice + k * nblk;
            if (row >= rows) break;
            const size_t base = (size_t)row * feat;
            const size_t obase = (size_t)(out_row0 + row) * feat;
            float ss = 0.0f;
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(x + base + i);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(v.x[j]);
                    ss += f * f;
                }
            }
            const float inv =
                rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)feat) + eps);
            float am = 0.0f;
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(x + base + i);
                const bf16v8 w = gamma ? ld_glob8(gamma + i) : bf16v8_zero();
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gamma ? __bfloat162float(w.x[j]) : 1.0f;
                    o.x[j] = __float2bfloat16(__bfloat162float(v.x[j]) * inv * g);
                    if (xq) am = fmaxf(am, fabsf(__bfloat162float(o.x[j])));
                }
                st_glob8(out + obase + i, o);
            }
            if (xq) {
                const float as = fmaxf(warp_max32(am) * (1.0f / 448.0f), 1e-12f);
                const float qinv = 1.0f / as;
                if (lane == 0) ascale[out_row0 + row] = as;
                for (unsigned i = lane * 8u; i < feat; i += 256u) {
                    const bf16v8 o = ld_glob8(out + obase + i);
                    uint8_t q8[8];
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        __nv_fp8_e4m3 q(__bfloat162float(o.x[j]) * qinv);
                        q8[j] = *(const uint8_t*)&q;
                    }
                    *(uint2*)(xq + obase + i) = *(const uint2*)q8;
                }
            }
        }
        return;
    }
    const bool fits = (feat <= RN_REG * PLOW_NV_THREADS) && ((feat & 7u) == 0);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const size_t obase = (size_t)(out_row0 + row) * feat;
        if (fits) {
            bf16v8 v[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                v[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    v[c] = ld_glob8(x + base + i);
                    if (gamma) w[c] = ld_glob8(gamma + i);
                }
            }
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(v[c].x[j]);
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
            if (!xq) {
#pragma unroll
                for (int c = 0; c < RN_VEC; c++) {
                    const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                    if (i < feat) {
                        bf16v8 o;
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const float g = gamma ? __bfloat162float(w[c].x[j]) : 1.0f;
                            o.x[j] = __float2bfloat16(__bfloat162float(v[c].x[j]) * inv * g);
                        }
                        st_glob8(out + obase + i, o);
                    }
                }
            } else {
                bf16v8 o[RN_VEC];
                float am = 0.0f;
#pragma unroll
                for (int c = 0; c < RN_VEC; c++) {
                    const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                    if (i < feat) {
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const float g = gamma ? __bfloat162float(w[c].x[j]) : 1.0f;
                            o[c].x[j] = __float2bfloat16(__bfloat162float(v[c].x[j]) * inv * g);
                            am = fmaxf(am, fabsf(__bfloat162float(o[c].x[j])));
                        }
                        st_glob8(out + obase + i, o[c]);
                    }
                }
                const float as = fmaxf(block_max(am, part) * (1.0f / 448.0f), 1e-12f);
                const float qinv = 1.0f / as;
                if (threadIdx.x == 0) ascale[out_row0 + row] = as;
#pragma unroll
                for (int c = 0; c < RN_VEC; c++) {
                    const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                    if (i < feat) {
                        uint8_t q8[8];
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            __nv_fp8_e4m3 q(__bfloat162float(o[c].x[j]) * qinv);
                            q8[j] = *(const uint8_t*)&q;
                        }
                        *(uint2*)(xq + obase + i) = *(const uint2*)q8;
                    }
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float v = __bfloat162float(x[base + i]);
                ss += v * v;
            }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
            float am = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float g = gamma ? __bfloat162float(gamma[i]) : 1.0f;
                const __nv_bfloat16 ob = __float2bfloat16(__bfloat162float(x[base + i]) * inv * g);
                out[obase + i] = ob;
                if (xq) am = fmaxf(am, fabsf(__bfloat162float(ob)));
            }
            if (xq) {
                const float as = fmaxf(block_max(am, part) * (1.0f / 448.0f), 1e-12f);
                const float qinv = 1.0f / as;
                if (threadIdx.x == 0) ascale[out_row0 + row] = as;
                for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                    __nv_fp8_e4m3 q(__bfloat162float(out[obase + i]) * qinv);
                    xq[obase + i] = *(const uint8_t*)&q;
                }
            }
        }
    }
}

/* LayerNorm WITH bias + mean-subtract over `feat` (V6, PLOW_DOP_LAYERNORM=60) — the DSA indexer
 * key-norm (indexer.k_norm; nn.LayerNorm(index_head_dim=128, eps=1e-6, bias=True), the only non-RMS
 * norm in GLM-5.2). y = (x-μ)·rsqrt(var+eps)·γ+β, eps INSIDE the rsqrt, var the BIASED mean of
 * (x-μ)² computed as E[x²]-μ² (PyTorch LayerNorm, unbiased=False). gamma AND beta required.
 *
 * One block per row, strided by nblk. i3=out_row0 offsets the OUTPUT row only (writes the current
 * token's index key into its [ctx][DI] cache); input stays at `base`. Matches d_rmsnorm's shape. */
static __device__ void d_layernorm_bias(__nv_bfloat16* __restrict__ out,
                                        const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ gamma,
                                        const __nv_bfloat16* __restrict__ beta, unsigned rows,
                                        unsigned feat, unsigned out_row0, float eps, unsigned slice,
                                        unsigned nblk, float* part) {
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const size_t obase = (size_t)(out_row0 + row) * feat;
        float s = 0.0f, ss = 0.0f;
        for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
            const float v = __bfloat162float(x[base + i]);
            s += v;
            ss += v * v;
        }
        const float invf = __fdividef(1.0f, (float)feat);
        const float mean = block_sum(s, part) * invf;
        const float meansq = block_sum(ss, part) * invf;
        const float inv = rsqrtf(meansq - mean * mean + eps);
        for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
            const float g = __bfloat162float(gamma[i]);
            const float bb = __bfloat162float(beta[i]);
            out[obase + i] = __float2bfloat16((__bfloat162float(x[base + i]) - mean) * inv * g + bb);
        }
    }
}

/* FUSED residual-add + RMSNorm: resid = a + b (written back), out = RMSNorm(resid, gamma).
 *
 * `resid` ALIASES `a` in the caller (both are act.x on every Qwen packet), so neither is
 * __restrict__: every read of `a` is hoisted into registers before any store to resid. */
static __device__ void d_add_norm(__nv_bfloat16* __restrict__ out, __nv_bfloat16* resid,
                           const __nv_bfloat16* a, const __nv_bfloat16* __restrict__ b,
                           const __nv_bfloat16* __restrict__ gamma, unsigned rows, unsigned feat,
                           float eps, unsigned slice, unsigned nblk, float* part) {
    const bool fits = (feat <= RN_REG * PLOW_NV_THREADS) && ((feat & 7u) == 0);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            bf16v8 av[RN_VEC], bv[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                av[c] = bf16v8_zero();
                bv[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    av[c] = ld_glob8(a + base + i);
                    bv[c] = ld_glob8(b + base + i);
                    if (gamma) w[c] = ld_glob8(gamma + i);
                }
            }
            float r[RN_VEC * 8];
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(av[c].x[j]) + __bfloat162float(bv[c].x[j]);
                    r[c * 8 + j] = f;
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                if (i < feat) {
                    bf16v8 ro, no;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gamma ? __bfloat162float(w[c].x[j]) : 1.0f;
                        const float f = r[c * 8 + j];
                        ro.x[j] = __float2bfloat16(f);
                        no.x[j] = __float2bfloat16(f * inv * g);
                    }
                    st_glob8(resid + base + i, ro);
                    st_glob8(out + base + i, no);
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float f = __bfloat162float(a[base + i]) + __bfloat162float(b[base + i]);
                ss += f * f;
            }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float f = __bfloat162float(a[base + i]) + __bfloat162float(b[base + i]);
                const float g = gamma ? __bfloat162float(gamma[i]) : 1.0f;
                resid[base + i] = __float2bfloat16(f);
                out[base + i] = __float2bfloat16(f * inv * g);
            }
        }
    }
}

/* out = (a + RMSNorm(b, gamma)) * scale — Gemma's SANDWICH residual, in one packet.
 *
 * Warp32 port of runtime/amd/op_norm.h d_norm_residual. Numerics identical: the RMS reduction
 * is over `b` alone, the residual `a` is added AFTER scaling b by inv*gamma, and the whole sum
 * is multiplied by `scale` (Gemma's folded layer_scalar; 1.0 at the post-attn site). `out`
 * ALIASES `a` in the caller (in-place residual), so every read of `a` is hoisted into registers
 * before any store to out. Distinct from d_add_norm: there the norm is on the SUM a+b and both
 * resid and normed outputs are written; here the norm is on `b` only and there is one output. */
static __device__ void d_norm_residual(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* a,
                                const __nv_bfloat16* __restrict__ b,
                                const __nv_bfloat16* __restrict__ gamma, unsigned rows,
                                unsigned feat, float eps, float scale, unsigned slice,
                                unsigned nblk, float* part) {
    if (rows >= PLOW_NV_T17_MIN_ROWS && (feat & 7u) == 0) {
        /* T17 warp-per-row — see d_rmsnorm. */
        const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
        const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
        for (unsigned k = warp;; k += PLOW_NV_WARPS) {
            const unsigned row = slice + k * nblk;
            if (row >= rows) break;
            const size_t base = (size_t)row * feat;
            float ss = 0.0f;
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(b + base + i);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(v.x[j]);
                    ss += f * f;
                }
            }
            const float inv =
                rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)feat) + eps);
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(b + base + i);
                const bf16v8 av = ld_glob8(a + base + i);
                const bf16v8 w = gamma ? ld_glob8(gamma + i) : bf16v8_zero();
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gamma ? __bfloat162float(w.x[j]) : 1.0f;
                    o.x[j] = __float2bfloat16(
                        (__bfloat162float(av.x[j]) + gemma_postnorm_round(__bfloat162float(v.x[j]) * inv * g)) * scale);
                }
                st_glob8(out + base + i, o);
            }
        }
        return;
    }
    const bool fits = (feat <= RN_REG * PLOW_NV_THREADS) && ((feat & 7u) == 0);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            bf16v8 v[RN_VEC], av[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                v[c] = bf16v8_zero();
                av[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    v[c] = ld_glob8(b + base + i);
                    av[c] = ld_glob8(a + base + i);
                    if (gamma) w[c] = ld_glob8(gamma + i);
                }
            }
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(v[c].x[j]);
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                if (i < feat) {
                    bf16v8 o;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gamma ? __bfloat162float(w[c].x[j]) : 1.0f;
                        o.x[j] = __float2bfloat16(
                            (__bfloat162float(av[c].x[j]) + gemma_postnorm_round(__bfloat162float(v[c].x[j]) * inv * g)) *
                            scale);
                    }
                    st_glob8(out + base + i, o);
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float x = __bfloat162float(b[base + i]);
                ss += x * x;
            }
            const float inv = rsqrtf(block_sum(ss, part) * __fdividef(1.0f, (float)feat) + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float g = gamma ? __bfloat162float(gamma[i]) : 1.0f;
                out[base + i] = __float2bfloat16(
                    (__bfloat162float(a[base + i]) + gemma_postnorm_round(__bfloat162float(b[base + i]) * inv * g)) *
                    scale);
            }
        }
    }
}

/* Gemma SANDWICH residual AND the norm that follows it, in ONE packet (fused N1):
 *     resid = (a + RMSNorm(b, gb)) * scale     (the running residual stream)
 *     out   = RMSNorm(resid, gn)               (the normed activation the next sublayer reads)
 *
 * Warp32 port of runtime/amd/op_norm.h d_norm_residual_norm. `resid` is rounded to bf16 before
 * the second reduction. PLOW_NV_NRN_WPR matches the prefill pair's warp reduction order;
 * the legacy block reduction can round differently from the prefill pair. `resid`
 * aliases `a` in the caller; `b`/`out` are distinct. gb/gn == nullptr are the weightless variants. */
static __device__ void d_norm_residual_norm(__nv_bfloat16* __restrict__ out, __nv_bfloat16* resid,
                                     const __nv_bfloat16* a, const __nv_bfloat16* __restrict__ b,
                                     const __nv_bfloat16* __restrict__ gb,
                                     const __nv_bfloat16* __restrict__ gn, unsigned rows,
                                     unsigned feat, float eps, float scale, unsigned slice,
                                     unsigned nblk, float* part) {
#if PLOW_NV_NRN_WPR
    if (rows >= PLOW_NV_T17_MIN_ROWS && (feat & 7u) == 0) {
        const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
        const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
        for (unsigned k = warp;; k += PLOW_NV_WARPS) {
            const unsigned row = slice + k * nblk;
            if (row >= rows) break;
            const size_t base = (size_t)row * feat;
            float ssb = 0.0f;
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(b + base + i);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(v.x[j]);
                    ssb += f * f;
                }
            }
            const float invb = rsqrtf(warp_sum32(ssb) * __fdividef(1.0f, (float)feat) + eps);
            float ssr = 0.0f;
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 v = ld_glob8(b + base + i);
                const bf16v8 av = ld_glob8(a + base + i);
                const bf16v8 w = gb ? ld_glob8(gb + i) : bf16v8_zero();
                bf16v8 r;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gb ? __bfloat162float(w.x[j]) : 1.0f;
                    r.x[j] = __float2bfloat16(
                        (__bfloat162float(av.x[j]) + gemma_postnorm_round(__bfloat162float(v.x[j]) * invb * g)) * scale);
                    const float f = __bfloat162float(r.x[j]);
                    ssr += f * f;
                }
                st_glob8(resid + base + i, r);
            }
            const float invr = rsqrtf(warp_sum32(ssr) * __fdividef(1.0f, (float)feat) + eps);
            for (unsigned i = lane * 8u; i < feat; i += 256u) {
                const bf16v8 r = ld_glob8(resid + base + i);
                const bf16v8 w = gn ? ld_glob8(gn + i) : bf16v8_zero();
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gn ? __bfloat162float(w.x[j]) : 1.0f;
                    o.x[j] = __float2bfloat16(__bfloat162float(r.x[j]) * invr * g);
                }
                st_glob8(out + base + i, o);
            }
        }
        return;
    }
#endif
    const bool fits = (feat <= RN_REG * PLOW_NV_THREADS) && ((feat & 7u) == 0);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            bf16v8 av[RN_VEC], bv[RN_VEC], wb[RN_VEC], wn[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                av[c] = bf16v8_zero();
                bv[c] = bf16v8_zero();
                wb[c] = bf16v8_zero();
                wn[c] = bf16v8_zero();
                if (i < feat) {
                    av[c] = ld_glob8(a + base + i);
                    bv[c] = ld_glob8(b + base + i);
                    if (gb) wb[c] = ld_glob8(gb + i);
                    if (gn) wn[c] = ld_glob8(gn + i);
                }
            }
            /* FIRST norm: RMS over the sublayer output b. */
            float ssb = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(bv[c].x[j]);
                    ssb += f * f;
                }
            const float invb = rsqrtf(block_sum(ssb, part) / (float)feat + eps);
            /* resid = (a + norm(b)*gb) * scale, ROUNDED to bf16; the second reduction runs over
             * the rounded value, exactly reproducing NORM_RESIDUAL's store + RMSNORM's reload. */
            bf16v8 rv[RN_VEC];
            float ssr = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                bf16v8 r;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gb ? __bfloat162float(wb[c].x[j]) : 1.0f;
                    const float f =
                        (__bfloat162float(av[c].x[j]) + gemma_postnorm_round(__bfloat162float(bv[c].x[j]) * invb * g)) *
                        scale;
                    r.x[j] = __float2bfloat16(f);
                    const float rf = __bfloat162float(r.x[j]);
                    ssr += rf * rf;
                }
                rv[c] = r;
            }
            const float invr = rsqrtf(block_sum(ssr, part) / (float)feat + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_NV_THREADS) * 8;
                if (i < feat) {
                    bf16v8 no;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gn ? __bfloat162float(wn[c].x[j]) : 1.0f;
                        no.x[j] = __float2bfloat16(__bfloat162float(rv[c].x[j]) * invr * g);
                    }
                    st_glob8(resid + base + i, rv[c]);
                    st_glob8(out + base + i, no);
                }
            }
        } else {
            float ssb = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float f = __bfloat162float(b[base + i]);
                ssb += f * f;
            }
            const float invb = rsqrtf(block_sum(ssb, part) / (float)feat + eps);
            float ssr = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float g = gb ? __bfloat162float(gb[i]) : 1.0f;
                const __nv_bfloat16 rb = __float2bfloat16(
                    (__bfloat162float(a[base + i]) + gemma_postnorm_round(__bfloat162float(b[base + i]) * invb * g)) *
                    scale);
                resid[base + i] = rb;
                const float rf = __bfloat162float(rb);
                ssr += rf * rf;
            }
            const float invr = rsqrtf(block_sum(ssr, part) / (float)feat + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_NV_THREADS) {
                const float g = gn ? __bfloat162float(gn[i]) : 1.0f;
                out[base + i] = __float2bfloat16(__bfloat162float(resid[base + i]) * invr * g);
            }
        }
    }
}

/* ---- per-head norm + RoPE ------------------------------------------------------------
 * WRITTEN FRESH for warp32. One WARP owns one (token, head) pair; the head's `hd` elements
 * are LANE-STRIDED, lane l holding {l, l+32, l+64, ...} — E = HD/32 elements per lane.
 *
 * LANE LOCALITY IS THE WHOLE DESIGN, and it is what gemma_normrope_sm120 got wrong. The
 * half-split rotate pairs element i with element i + H2 (H2 = HD/2). Under this layout
 * element i lives in lane (i % 32) at e = i/32, so i and i+H2 are in the SAME LANE exactly
 * when H2 is a multiple of 32, i.e. when HD % 64 == 0. The static_assert below states that
 * invariant instead of assuming it. HD = 128/256/512 all satisfy it, so the rotate needs NO
 * cross-lane traffic at any of the three head dims — whereas a thread-local `float nv[HD]`
 * indexed by nv[i+half] is only ever correct at one specific hd.
 *
 * The reduction is warp_sum32 over exactly the 32 lanes holding the head — the AMD original
 * reduces over 64. Using the wave64 form here would fold in a NEIGHBOURING HEAD's partial
 * sums and quietly renormalize both heads by the wrong scale.
 *
 * INTERLEAVE (GPT-J style, template flag, default OFF): rotate adjacent (2m, 2m+1) pairs.
 * The partner index is i XOR 1, which under this layout is lane (l XOR 1) at the SAME e
 * (since e*32 is even), so ONE __shfl_xor_sync(.,1,32) fetches it. Qwen uses INTERLEAVE=0.
 *
 * i4=skip_norm passes the head through unnormalized (Qwen's V write). i3=out_row0 is patched
 * per step by the host. j0=out_stride != 0 selects the HEAD-MAJOR KV layout [kv_head][ctx][hd];
 * j1=kv_mask is the sliding-window RING mask (0xFFFFFFFF on a full layer = one AND that does
 * nothing, no branch). */
/* BATCH>1 (serving pending #4): n_batch_kv>1 selects the per-sequence KV base — token t is
 * sequence t and writes into ITS OWN ring: obase = ((t*nhead + hh)*out_stride + (pos[t]&mask))*hd.
 * The write ROW is the sequence's own position pos[t] (each sequence is at a different position),
 * not the single-sequence out_row0+t. n_batch_kv<=1 keeps the legacy out_row0 formula, so decode
 * B=1 and every prefill packet stay byte-identical. Only the KV-write path (out_stride!=0) differs;
 * the Q path (out_stride==0) is unaffected. */
/* BATCHED PREFILL (PX-1): pfslot != nullptr selects the CROSS-REQUEST prefill KV write — row t
 * of the packed chunk belongs to engine seq-slot pfslot[t] and writes that slot's batch-major
 * ring at its own absolute position pos[t]:
 *   obase = ((pfslot[t]*nhead + hh)*out_stride + (pos[t]&mask))*hd.
 * This is the n_batch_kv formula with the per-row slot map replacing the row==sequence identity,
 * so N requests' chunks share ONE launch while each row lands in its own request's cache. The
 * host only patches t6 (the slot map) on prefill KV-write sites in batched mode; pfslot==nullptr
 * keeps every existing packet byte-identical. */
template <int HD, bool INTERLEAVE = false>
static __device__ void d_headnorm_rope(__nv_bfloat16* __restrict__ out,
                                const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ gamma,
                                const float* __restrict__ cosb, const float* __restrict__ sinb,
                                const int* __restrict__ pos, unsigned ntok, unsigned nhead,
                                float eps, unsigned out_row0, unsigned out_stride,
                                unsigned kv_mask, unsigned skip_norm, unsigned slice,
                                unsigned nblk, unsigned n_batch_kv = 0,
                                const int* __restrict__ pfslot = nullptr) {
    static_assert(HD % 64 == 0,
                  "head_dim must be a multiple of 64 so the half-split RoPE partner (i, i+HD/2) "
                  "stays inside one 32-lane warp — otherwise the rotate needs cross-lane traffic "
                  "this kernel does not perform, and the result is silently wrong");
    constexpr unsigned hd = HD;
    constexpr unsigned E = HD / 32; /* elements per lane: 4 (hd=128), 8 (256), 16 (512) */
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp_in_blk = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned total = ntok * nhead;

    for (unsigned w = slice * PLOW_NV_WARPS + warp_in_blk; w < total;
         w += nblk * PLOW_NV_WARPS) {
        /* nhead is a power of two on every shipped model (Qwen3 32/8, Gemma alike); the u32
         * div/mod chain (~35 instrs) was ~23% of this latency-dust body. Uniform branch: pow2
         * path is 3 instrs, div path survives for any future non-pow2 head count. */
        unsigned t, hh;
        if ((nhead & (nhead - 1u)) == 0u) {
            const unsigned nsh = 31u - (unsigned)__clz((int)nhead);
            t = w >> nsh; hh = w & (nhead - 1u);
        } else { t = w / nhead; hh = w % nhead; }
        const size_t ibase = ((size_t)t * nhead + hh) * hd;
        /* KV write (out_stride!=0): per-row slot map (batched prefill), per-batch ring when
         * n_batch_kv!=0 (the row index derives from pos[t] — at n_batch_kv==1 this is the
         * B=1 decode ring with NO host i[3] patch, plan plowrt-gpu-exec-critical-path
         * stage 2; the engine arms it only on cubins carrying plow_dyn_kvrow), else the
         * legacy host-patched single-ring. */
        const size_t obase =
            out_stride
                ? (pfslot
                       ? ((size_t)((unsigned)pfslot[t] * nhead + hh) * out_stride +
                          ((unsigned)pos[t] & kv_mask)) *
                             hd
                       : n_batch_kv != 0
                       ? ((size_t)(t * nhead + hh) * out_stride + ((unsigned)pos[t] & kv_mask)) * hd
                       : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask)) * hd)
                : ((size_t)(out_row0 + t) * nhead + hh) * hd;

        /* PACK-OF-4 lane layout (HD % 256 == 0, half-split rotate): lane l chunk c owns the
         * FOUR CONTIGUOUS elements [4*(l+32c), +4), so x/gamma/out are 8-byte accesses and the
         * cos/sin table read is one float4 -- and the half-split partner i+HD/2 is chunk c+C/2
         * in the SAME lane, so the rotate still needs no cross-lane traffic. */
        if constexpr (!INTERLEAVE && (HD % 256) == 0) {
            constexpr unsigned C = HD / 128, CH = C / 2;
            float v[4 * C], g[4 * C];
#pragma unroll
            for (unsigned c = 0; c < C; c++) {
                const ushort4 xv = *(const ushort4*)(x + ibase + 4u * (lane + 32u * c));
                const unsigned short* xs = (const unsigned short*)&xv;
#pragma unroll
                for (int k = 0; k < 4; k++)
                    v[4 * c + k] = __bfloat162float(*(const __nv_bfloat16*)&xs[k]);
                if (gamma) {
                    const ushort4 gv = *(const ushort4*)(gamma + 4u * (lane + 32u * c));
                    const unsigned short* gs = (const unsigned short*)&gv;
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        g[4 * c + k] = __bfloat162float(*(const __nv_bfloat16*)&gs[k]);
                } else {
#pragma unroll
                    for (int k = 0; k < 4; k++) g[4 * c + k] = 1.0f;
                }
            }
            float inv = 1.0f;
            if (!skip_norm) {
                float ss = 0.0f;
#pragma unroll
                for (unsigned e = 0; e < 4 * C; e++) ss += v[e] * v[e];
                inv = rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)hd) + eps);
            }
#pragma unroll
            for (unsigned e = 0; e < 4 * C; e++) v[e] = v[e] * inv * g[e];
            if (cosb) {
#if PLOW_NV_GEMMA && PLOW_NV_GEMMA_HNR_BF16
                // Match separate BF16 Q/K normalization before the rotary kernel.
#pragma unroll
                for (unsigned e = 0; e < 4 * C; e++)
                    v[e] = __bfloat162float(__float2bfloat16(v[e]));
#endif
                const size_t p = (size_t)pos[t] * (HD / 2);
                float r[4 * C];
#pragma unroll
                for (unsigned c = 0; c < CH; c++) {
                    const size_t j = p + 4u * (lane + 32u * c);
                    const float4 cc = *(const float4*)(cosb + j);
                    const float4 sv = *(const float4*)(sinb + j);
                    const float* cp = (const float*)&cc;
                    const float* sp = (const float*)&sv;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const unsigned lo = 4 * c + k, hi = 4 * (c + CH) + k;
#if PLOW_NV_GEMMA && PLOW_NV_GEMMA_HNR_BF16
                        const float cr = __bfloat162float(__float2bfloat16(cp[k]));
                        const float sr = __bfloat162float(__float2bfloat16(sp[k]));
                        r[lo] = v[lo] * cr - v[hi] * sr;
                        r[hi] = v[hi] * cr + v[lo] * sr;
#else
                        r[lo] = v[lo] * cp[k] - v[hi] * sp[k];
                        r[hi] = v[hi] * cp[k] + v[lo] * sp[k];
#endif
                    }
                }
#pragma unroll
                for (unsigned e = 0; e < 4 * C; e++) v[e] = r[e];
            }
#pragma unroll
            for (unsigned c = 0; c < C; c++) {
                ushort4 o;
                unsigned short* os = (unsigned short*)&o;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const __nv_bfloat16 h = __float2bfloat16(v[4 * c + k]);
                    os[k] = *(const unsigned short*)&h;
                }
                *(ushort4*)(out + obase + 4u * (lane + 32u * c)) = o;
            }
            continue;
        }
        /* PRODUCE: the head AND its weight, all E of each, before anything is waited on. */
        float v[E], g[E];
#pragma unroll
        for (unsigned e = 0; e < E; e++) {
            v[e] = __bfloat162float(x[ibase + lane + e * 32]);
            g[e] = gamma ? __bfloat162float(gamma[lane + e * 32]) : 1.0f;
        }
        float inv = 1.0f;
        if (!skip_norm) {
            float ss = 0.0f;
#pragma unroll
            for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
            inv = rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)hd) + eps);
        }
#pragma unroll
        for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * g[e];

        if (cosb) {
            constexpr unsigned H2 = HD / 2;
            const size_t p = (size_t)pos[t] * H2;
            float r[E];
            if constexpr (!INTERLEAVE) {
                constexpr unsigned EH = H2 / 32; /* lane-local stride to the half-split partner */
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 32;
                    const unsigned j = (i < H2) ? i : (i - H2);
                    const float c = cosb[p + j], s = sinb[p + j];
                    r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s)  /* i in [0, H2)  */
                                    : (v[e] * c + v[e - EH] * s); /* i in [H2, hd) */
                }
            } else {
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 32;
                    const float c = cosb[p + (i >> 1)], s = sinb[p + (i >> 1)];
                    const float partner = __shfl_xor_sync(0xffffffffu, v[e], 1, 32);
                    r[e] = ((i & 1u) == 0u) ? (v[e] * c - partner * s)
                                            : (v[e] * c + partner * s);
                }
            }
#pragma unroll
            for (unsigned e = 0; e < E; e++) v[e] = r[e];
        }

#pragma unroll
        for (unsigned e = 0; e < E; e++)
            out[obase + lane + e * 32] = __float2bfloat16(v[e]);
    }
}

/* FP8 (e4m3) KV-cache twin of d_headnorm_rope (PLOW_FP8_KV). IDENTICAL norm+RoPE math — it produces
 * the SAME v[e] — but the store quantizes to e4m3 with a PER-ROW scale so flash reads HALF the bytes.
 *
 * A "row" is one (token, kv_head) K/V vector of `hd` elements, held across the 32 lanes of this warp
 * (E = HD/32 per lane). Its amax over the whole row picks the scale: scale = amax/448 maps the largest
 * element onto e4m3's full range, and the stored byte is round_e4m3(v/scale). flash-decode multiplies
 * the decoded value back by `scale` (per row), so the only loss is e4m3's 3-mantissa-bit quantization.
 * `out` is the uint8 cache; `scale` is f32[kv_head][ctx], SAME head-major RING layout as the cache, so
 * its row index is obase/hd (matches the flash reader's ksc[kv & kv_mask]). This path is K/V only, so
 * out_stride != 0 always; q stays plain d_headnorm_rope (it is not cached). skip_norm/INTERLEAVE and
 * the batched (n_batch_kv / pfslot) obase formulas carry over unchanged from d_headnorm_rope. */
template <int HD, bool INTERLEAVE = false>
static __device__ void d_headnorm_rope_fp8(uint8_t* __restrict__ out, float* __restrict__ scale,
                                    const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ gamma,
                                    const float* __restrict__ cosb, const float* __restrict__ sinb,
                                    const int* __restrict__ pos, unsigned ntok, unsigned nhead,
                                    float eps, unsigned out_row0, unsigned out_stride,
                                    unsigned kv_mask, unsigned skip_norm, unsigned slice,
                                    unsigned nblk, unsigned n_batch_kv = 0,
                                    const int* __restrict__ pfslot = nullptr) {
    static_assert(HD % 64 == 0,
                  "head_dim must be a multiple of 64 so the half-split RoPE partner stays in-warp");
    constexpr unsigned hd = HD;
    constexpr unsigned E = HD / 32;
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp_in_blk = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned total = ntok * nhead;

    for (unsigned w = slice * PLOW_NV_WARPS + warp_in_blk; w < total;
         w += nblk * PLOW_NV_WARPS) {
        /* nhead is a power of two on every shipped model (Qwen3 32/8, Gemma alike); the u32
         * div/mod chain (~35 instrs) was ~23% of this latency-dust body. Uniform branch: pow2
         * path is 3 instrs, div path survives for any future non-pow2 head count. */
        unsigned t, hh;
        if ((nhead & (nhead - 1u)) == 0u) {
            const unsigned nsh = 31u - (unsigned)__clz((int)nhead);
            t = w >> nsh; hh = w & (nhead - 1u);
        } else { t = w / nhead; hh = w % nhead; }
        const size_t ibase = ((size_t)t * nhead + hh) * hd;
        /* KV write (out_stride!=0 always here): per-row slot map (batched prefill), per-batch ring
         * when n_batch_kv!=0 (pos[t]-derived row; ==1 is the patch-free B=1 decode ring), else
         * legacy single-ring — EXACTLY d_headnorm_rope's obase. */
        const size_t rowidx =
            pfslot ? ((size_t)((unsigned)pfslot[t] * nhead + hh) * out_stride +
                      ((unsigned)pos[t] & kv_mask))
                   : n_batch_kv != 0
                   ? ((size_t)(t * nhead + hh) * out_stride + ((unsigned)pos[t] & kv_mask))
                   : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask));
        const size_t obase = rowidx * hd;

        /* PACK-OF-4 lane layout (HD % 256 == 0, half-split rotate) — the d_headnorm_rope twin's
         * layout: lane l chunk c owns the FOUR CONTIGUOUS elements [4*(l+32c), +4). Loads widen to
         * one ushort4 per chunk (convert INSIDE the loop — staging into ushort4[] costs +25 reg);
         * the e4m3 store widens too: a 4-element group is 4 CONTIGUOUS bytes -> ONE 32-bit uchar4
         * store (8/16 single-byte stores -> 2/4 for hd256/hd512). Same norm/RoPE math.
         * NOT raw-byte-identical to the scalar path (unlike the bf16 twin): the relayout reorders
         * the sum-of-squares reduction, so `inv` shifts by 1 ULP; e4m3's 3-bit mantissa flips at the
         * rounding boundary (~25% of bytes at hd256) where bf16's 8-bit mantissa did not. But amax
         * shifts WITH inv, so the per-row scale compensates: the DEQUANTIZED KV (byte x scale) that
         * flash-decode actually reads is identical to ~5e-8 relL2 (measured), far below e4m3's own
         * ~6% quantization granularity — model behaviour is unchanged. Register-free: megakernel
         * REG/STACK/SHARED = 208/1024/2192, unchanged from the scalar path. */
        if constexpr (!INTERLEAVE && (HD % 256) == 0) {
            constexpr unsigned C = HD / 128, CH = C / 2;
            float v[4 * C], g[4 * C];
#pragma unroll
            for (unsigned c = 0; c < C; c++) {
                const ushort4 xv = *(const ushort4*)(x + ibase + 4u * (lane + 32u * c));
                const unsigned short* xs = (const unsigned short*)&xv;
#pragma unroll
                for (int k = 0; k < 4; k++)
                    v[4 * c + k] = __bfloat162float(*(const __nv_bfloat16*)&xs[k]);
                if (gamma) {
                    const ushort4 gv = *(const ushort4*)(gamma + 4u * (lane + 32u * c));
                    const unsigned short* gs = (const unsigned short*)&gv;
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        g[4 * c + k] = __bfloat162float(*(const __nv_bfloat16*)&gs[k]);
                } else {
#pragma unroll
                    for (int k = 0; k < 4; k++) g[4 * c + k] = 1.0f;
                }
            }
            float inv = 1.0f;
            if (!skip_norm) {
                float ss = 0.0f;
#pragma unroll
                for (unsigned e = 0; e < 4 * C; e++) ss += v[e] * v[e];
                inv = rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)hd) + eps);
            }
#pragma unroll
            for (unsigned e = 0; e < 4 * C; e++) v[e] = v[e] * inv * g[e];
            if (cosb) {
                const size_t p = (size_t)pos[t] * (HD / 2);
                float r[4 * C];
#pragma unroll
                for (unsigned c = 0; c < CH; c++) {
                    const size_t j = p + 4u * (lane + 32u * c);
                    const float4 cc = *(const float4*)(cosb + j);
                    const float4 sv = *(const float4*)(sinb + j);
                    const float* cp = (const float*)&cc;
                    const float* sp = (const float*)&sv;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const unsigned lo = 4 * c + k, hi = 4 * (c + CH) + k;
                        r[lo] = v[lo] * cp[k] - v[hi] * sp[k];
                        r[hi] = v[hi] * cp[k] + v[lo] * sp[k];
                    }
                }
#pragma unroll
                for (unsigned e = 0; e < 4 * C; e++) v[e] = r[e];
            }
            /* PER-ROW SCALE: amax over the whole hd-vector, then across the warp. */
            float amax = 0.0f;
#pragma unroll
            for (unsigned e = 0; e < 4 * C; e++) amax = fmaxf(amax, fabsf(v[e]));
            amax = warp_max32(amax);
            const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
            if (lane == 0) scale[rowidx] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
#pragma unroll
            for (unsigned c = 0; c < C; c++) {
                uchar4 o;
                unsigned char* op = (unsigned char*)&o;
#pragma unroll
                for (int k = 0; k < 4; k++) op[k] = quant_fp8(v[4 * c + k] * qinv);
                *(uchar4*)(out + obase + 4u * (lane + 32u * c)) = o;
            }
            continue;
        }

        float v[E], g[E];
#pragma unroll
        for (unsigned e = 0; e < E; e++) {
            v[e] = __bfloat162float(x[ibase + lane + e * 32]);
            g[e] = gamma ? __bfloat162float(gamma[lane + e * 32]) : 1.0f;
        }
        float inv = 1.0f;
        if (!skip_norm) {
            float ss = 0.0f;
#pragma unroll
            for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
            inv = rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)hd) + eps);
        }
#pragma unroll
        for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * g[e];

        if (cosb) {
            constexpr unsigned H2 = HD / 2;
            const size_t p = (size_t)pos[t] * H2;
            float r[E];
            if constexpr (!INTERLEAVE) {
                constexpr unsigned EH = H2 / 32;
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 32;
                    const unsigned j = (i < H2) ? i : (i - H2);
                    const float c = cosb[p + j], s = sinb[p + j];
                    r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s) : (v[e] * c + v[e - EH] * s);
                }
            } else {
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 32;
                    const float c = cosb[p + (i >> 1)], s = sinb[p + (i >> 1)];
                    const float partner = __shfl_xor_sync(0xffffffffu, v[e], 1, 32);
                    r[e] = ((i & 1u) == 0u) ? (v[e] * c - partner * s) : (v[e] * c + partner * s);
                }
            }
#pragma unroll
            for (unsigned e = 0; e < E; e++) v[e] = r[e];
        }

        /* PER-ROW SCALE: amax over the whole hd-vector (this lane's E elements, then across the
         * warp), map to e4m3's 448. amax==0 (all-zero row) => qinv 0 => stored bytes 0, scale 0. */
        float amax = 0.0f;
#pragma unroll
        for (unsigned e = 0; e < E; e++) amax = fmaxf(amax, fabsf(v[e]));
        amax = warp_max32(amax);
        const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
        if (lane == 0) scale[rowidx] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
#pragma unroll
        for (unsigned e = 0; e < E; e++)
            out[obase + lane + e * 32] = quant_fp8(v[e] * qinv);
    }
}
