/* op_speech_f32.cuh — FP32 speech primitives (ops 163-178, 180) for the NVIDIA interpreter.
 *
 * Reference semantics: runtime/cpu/dev/golden/f32_primitives.c. Wherever the golden sums in
 * float, the order is reproduced and the adds/muls use the _rn intrinsics so nvcc cannot
 * contract them into FMAs (the C source has none; build the golden -ffp-contract=off to compare).
 * Wherever the golden sums in double, the order is free. The GEMM-shaped ops (DenseGemm,
 * GemmF32, Q8Gemm, standard Conv2d) accumulate in fp32 FMA on 128x128 smem tiles instead of
 * the golden's double/serial sums: they match to fp32 rounding, and to at most one bf16 ulp
 * after bf16 rounding.
 */
#pragma once
#include "dev_isa.h"
#include "sm120_common.cuh"
#include <cuda_fp16.h>

static_assert(PLOW_NV_THREADS == 256, "speech tiles assume 256 threads");

#define SPG_TM 8
#define SPG_TN 8
#define SPG_BK 16
#define SPG_BM (16 * SPG_TM)
#define SPG_BN (16 * SPG_TN)
#define SPG_LDA (SPG_BM + 4)
#define SPG_LDB (SPG_BN + 4)
#define SPG_SMEM_FLOATS (2 * SPG_BK * (SPG_LDA + SPG_LDB))
#define SPA_SCORE_FLOATS (PLOW_NV_WARPS * 256)
/* Covers the GEMM ring and GroupedAttention's K stage for group_rows*(head_width+1) <= 10240
 * (Qwen3-ASR: 104*65). Larger groups read K from global instead. */
#define SP_ARENA_FLOATS 12288
static_assert(SP_ARENA_FLOATS >= SPG_SMEM_FLOATS, "speech arena");

__device__ __forceinline__ float sp_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
/* CUDA's expf/tanhf are 2-ulp; the double forms round like glibc's (~0.5 ulp), which keeps
 * bf16-rounded probabilities and GELUs on the golden's side of a rounding boundary. */
__device__ __forceinline__ float sp_expf(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float sp_tanhf(float x) { return (float)tanh((double)x); }
__device__ __forceinline__ float sp_f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float sp_sigmoid(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, sp_expf(-x)));
}
__device__ __forceinline__ float sp_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, sp_expf(-x))); }

/* bf16(x) -> erf-GELU (Abramowitz-Stegun 7.1.26) -> bf16, exactly as the golden spells it. */
__device__ __forceinline__ float sp_gelu_erf_bf16(float value) {
    value = sp_bf16(value);
    const float z = __fmul_rn(value, 0.7071067811865475f);
    const float t = __fdiv_rn(1.0f, __fadd_rn(1.0f, __fmul_rn(0.3275911f, fabsf(z))));
    float p = __fsub_rn(__fmul_rn(1.061405429f, t), 1.453152027f);
    p = __fadd_rn(__fmul_rn(p, t), 1.421413741f);
    p = __fsub_rn(__fmul_rn(p, t), 0.284496736f);
    p = __fadd_rn(__fmul_rn(p, t), 0.254829592f);
    const float r = __fmul_rn(__fmul_rn(p, t), sp_expf(__fmul_rn(-z, z)));
    return sp_bf16(__fmul_rn(__fmul_rn(0.5f, value), z < 0.0f ? r : __fsub_rn(2.0f, r)));
}

__device__ __forceinline__ double sp_warp_sum_d(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sp_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* ---- tiled FP32 GEMM: C[m][n] = sum_k A(m,k) * B(n,k) -------------------------------------
 * 128x128 output tiles strided over the op's blocks, 8x8 per thread, BK=16 double-buffered smem
 * with register prefetch (measured on H100 vs 64x64/4x4: 1.3x at N>=1280, equal at N=1024). Loaders expose row(r) -> per-row state and load4(state, k) ->
 * elements k..k+3 (zero past K or past the last row). */
struct SpRowF32 {
    const float* p; unsigned k; bool vec;
    __device__ const float* row(unsigned r, unsigned rows) const { return r < rows ? p + (size_t)r * k : nullptr; }
    __device__ float4 load4(const float* r, unsigned c) const {
        if (!r) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (vec && c + 3 < k) return __ldg((const float4*)(r + c));
        float4 o;
        o.x = c < k ? r[c] : 0.f;
        o.y = c + 1 < k ? r[c + 1] : 0.f;
        o.z = c + 2 < k ? r[c + 2] : 0.f;
        o.w = c + 3 < k ? r[c + 3] : 0.f;
        return o;
    }
};
/* Row-major bf16 rows of `stride` elements, of which the first k are the GEMM's inner dim. */
struct SpRowBf16 {
    const __nv_bfloat16* p; unsigned k, stride; bool vec;
    __device__ const __nv_bfloat16* row(unsigned r, unsigned rows) const {
        return r < rows ? p + (size_t)r * stride : nullptr;
    }
    __device__ float4 load4(const __nv_bfloat16* r, unsigned c) const {
        if (!r) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (vec && c + 3 < k) {
            const uint2 u = __ldg((const uint2*)(r + c));
            return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                               __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
        }
        float4 o;
        o.x = c < k ? __bfloat162float(r[c]) : 0.f;
        o.y = c + 1 < k ? __bfloat162float(r[c + 1]) : 0.f;
        o.z = c + 2 < k ? __bfloat162float(r[c + 2]) : 0.f;
        o.w = c + 3 < k ? __bfloat162float(r[c + 3]) : 0.f;
        return o;
    }
};
struct SpRowF32S {
    const float* p; unsigned k, stride; bool vec;
    __device__ const float* row(unsigned r, unsigned rows) const { return r < rows ? p + (size_t)r * stride : nullptr; }
    __device__ float4 load4(const float* r, unsigned c) const {
        if (!r) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (vec && c + 3 < k) return __ldg((const float4*)(r + c));
        float4 o;
        o.x = c < k ? r[c] : 0.f;
        o.y = c + 1 < k ? r[c + 1] : 0.f;
        o.z = c + 2 < k ? r[c + 2] : 0.f;
        o.w = c + 3 < k ? r[c + 3] : 0.f;
        return o;
    }
};
struct SpRowF16 {
    const uint16_t* p; unsigned k;
    __device__ const uint16_t* row(unsigned r, unsigned rows) const { return r < rows ? p + (size_t)r * k : nullptr; }
    __device__ float4 load4(const uint16_t* r, unsigned c) const {
        if (!r) return make_float4(0.f, 0.f, 0.f, 0.f);
        float4 o;
        o.x = c < k ? sp_f16(r[c]) : 0.f;
        o.y = c + 1 < k ? sp_f16(r[c + 1]) : 0.f;
        o.z = c + 2 < k ? sp_f16(r[c + 2]) : 0.f;
        o.w = c + 3 < k ? sp_f16(r[c + 3]) : 0.f;
        return o;
    }
};
/* GGUF Q8_0 weight rows: per 32-element K block one fp16 scale + 32 int8 (34 bytes). */
struct SpRowQ8 {
    const uint8_t* p; unsigned k;
    __device__ const uint8_t* row(unsigned r, unsigned rows) const {
        return r < rows ? p + (size_t)r * (k / 32u) * 34u : nullptr;
    }
    __device__ float4 load4(const uint8_t* r, unsigned c) const {
        if (!r || c >= k) return make_float4(0.f, 0.f, 0.f, 0.f);
        const uint8_t* b = r + (c >> 5) * 34u;
        const float s = sp_f16(*(const uint16_t*)b);
        const int8_t* q = (const int8_t*)(b + 2u + (c & 31u));
        return make_float4(__fmul_rn(s, (float)q[0]), __fmul_rn(s, (float)q[1]),
                           __fmul_rn(s, (float)q[2]), __fmul_rn(s, (float)q[3]));
    }
};

struct SpConvGeom {
    unsigned frames, width, ic, kernel, stride, pad, of, ow, layout;
};
struct SpConvPos { int b, iy0, ix0; bool ok; };
/* im2col view of the conv input: row = (batch, out_frame, out_x), col = (in_ch, ky, kx). */
struct SpRowConv {
    const float* x; SpConvGeom g; unsigned k;
    __device__ SpConvPos row(unsigned r, unsigned rows) const {
        SpConvPos s;
        s.ok = r < rows;
        const unsigned ox = r % g.ow, oy = (r / g.ow) % g.of;
        s.b = (int)(r / (g.ow * g.of));
        s.iy0 = (int)(oy * g.stride) - (int)g.pad;
        s.ix0 = (int)(ox * g.stride) - (int)g.pad;
        return s;
    }
    __device__ float at(const SpConvPos& s, unsigned c) const {
        if (!s.ok || c >= k) return 0.f;
        const unsigned kk = g.kernel * g.kernel;
        const unsigned ic = c / kk, rem = c - ic * kk, ky = rem / g.kernel, kx = rem - ky * g.kernel;
        const int iy = s.iy0 + (int)ky, ix = s.ix0 + (int)kx;
        if (iy < 0 || iy >= (int)g.frames || ix < 0 || ix >= (int)g.width) return 0.f;
        size_t xi;
        if (g.layout == 0u)
            xi = (((size_t)s.b * g.frames + iy) * g.width + ix) * g.ic + ic;
        else if (g.layout == 1u)
            xi = (((size_t)s.b * g.frames + iy) * g.ic + ic) * g.width + ix;
        else
            xi = (((size_t)s.b * g.ic + ic) * g.frames + iy) * g.width + ix;
        return x[xi];
    }
    __device__ float4 load4(const SpConvPos& s, unsigned c) const {
        return make_float4(at(s, c), at(s, c + 1), at(s, c + 2), at(s, c + 3));
    }
};

template <class LA, class LB, class EP>
static __device__ __forceinline__ void sp_gemm(unsigned M, unsigned N, unsigned K, const LA& la,
                                               const LB& lb, const EP& ep, unsigned slice,
                                               unsigned nblk, float* arena) {
    constexpr unsigned KT = SPG_BK / 4, RPP = PLOW_NV_THREADS / KT;
    constexpr unsigned AP = SPG_BM / RPP, BP = SPG_BN / RPP;
    static_assert(AP * RPP == SPG_BM && BP * RPP == SPG_BN, "loader coverage");
    float* As = arena;
    float* Bs = arena + 2 * SPG_BK * SPG_LDA;
    const unsigned tid = threadIdx.x, tx = tid & 15u, ty = tid >> 4;
    const unsigned lr = tid / KT, lk = (tid % KT) * 4u;
    const unsigned tn = (N + SPG_BN - 1) / SPG_BN;
    const unsigned ntiles = ((M + SPG_BM - 1) / SPG_BM) * tn;
    const unsigned nk = (K + SPG_BK - 1) / SPG_BK;
    for (unsigned tile = slice; tile < ntiles; tile += nblk) {
        const unsigned m0 = (tile / tn) * SPG_BM, n0 = (tile % tn) * SPG_BN;
        decltype(la.row(0u, 0u)) ra[AP];
        decltype(lb.row(0u, 0u)) rb[BP];
        float4 va[AP], vb[BP];
#pragma unroll
        for (unsigned p = 0; p < AP; p++) { ra[p] = la.row(m0 + lr + p * RPP, M); va[p] = la.load4(ra[p], lk); }
#pragma unroll
        for (unsigned p = 0; p < BP; p++) { rb[p] = lb.row(n0 + lr + p * RPP, N); vb[p] = lb.load4(rb[p], lk); }
        float acc[SPG_TM][SPG_TN];
#pragma unroll
        for (int i = 0; i < SPG_TM; i++)
#pragma unroll
            for (int j = 0; j < SPG_TN; j++) acc[i][j] = 0.f;
        for (unsigned kt = 0; kt < nk; kt++) {
            float* as = As + (kt & 1u) * SPG_BK * SPG_LDA;
            float* bs = Bs + (kt & 1u) * SPG_BK * SPG_LDB;
#pragma unroll
            for (unsigned p = 0; p < AP; p++) {
                float* d = as + lk * SPG_LDA + lr + p * RPP;
                d[0] = va[p].x; d[SPG_LDA] = va[p].y; d[2 * SPG_LDA] = va[p].z; d[3 * SPG_LDA] = va[p].w;
            }
#pragma unroll
            for (unsigned p = 0; p < BP; p++) {
                float* d = bs + lk * SPG_LDB + lr + p * RPP;
                d[0] = vb[p].x; d[SPG_LDB] = vb[p].y; d[2 * SPG_LDB] = vb[p].z; d[3 * SPG_LDB] = vb[p].w;
            }
            __syncthreads();
            if (kt + 1 < nk) {
#pragma unroll
                for (unsigned p = 0; p < AP; p++) va[p] = la.load4(ra[p], (kt + 1) * SPG_BK + lk);
#pragma unroll
                for (unsigned p = 0; p < BP; p++) vb[p] = lb.load4(rb[p], (kt + 1) * SPG_BK + lk);
            }
            const float* a = as + ty * SPG_TM;
            const float* b = bs + tx * SPG_TN;
#pragma unroll
            for (int k = 0; k < SPG_BK; k++) {
                float av[SPG_TM], bv[SPG_TN];
#pragma unroll
                for (int i = 0; i < SPG_TM; i += 4) *(float4*)(av + i) = *(const float4*)(a + k * SPG_LDA + i);
#pragma unroll
                for (int j = 0; j < SPG_TN; j += 4) *(float4*)(bv + j) = *(const float4*)(b + k * SPG_LDB + j);
#pragma unroll
                for (int i = 0; i < SPG_TM; i++)
#pragma unroll
                    for (int j = 0; j < SPG_TN; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            }
        }
        /* The next tile's first store targets buffer 0, which the last odd k-tile may still read. */
        __syncthreads();
#pragma unroll
        for (int i = 0; i < SPG_TM; i++) {
            const unsigned m = m0 + ty * SPG_TM + i;
            if (m >= M) break;
#pragma unroll
            for (int j = 0; j < SPG_TN; j++) {
                const unsigned n = n0 + tx * SPG_TN + j;
                if (n < N) ep(m, n, acc[i][j]);
            }
        }
    }
}

__device__ __forceinline__ bool sp_aligned(const void* p, unsigned bytes) {
    return ((uintptr_t)p & (bytes - 1u)) == 0;
}

/* DenseGemmF32 (170). flags: 1 = bf16-round, 2 = bf16 erf-GELU (implies the round), 4 = bf16
 * weights. i5 != 0: weight row stride, and i6 < i5 names a one-hot weight column added in. */
struct SpDenseEpi {
    float* out; const float* bias; const void* w; unsigned n, stride, onehot, flags, relu;
    __device__ void operator()(unsigned m, unsigned c, float acc) const {
        float value = bias ? __fadd_rn(acc, bias[c]) : acc;
        if (onehot < stride) {
            const size_t wi = (size_t)c * stride + onehot;
            value = __fadd_rn(value, flags & 4u ? __bfloat162float(((const __nv_bfloat16*)w)[wi])
                                                : ((const float*)w)[wi]);
        }
        if (flags & 2u) value = sp_gelu_erf_bf16(value);
        else if (flags & 1u) value = sp_bf16(value);
        out[(size_t)m * n + c] = relu ? fmaxf(value, 0.0f) : value;
    }
};
static __device__ void d_dense_gemm_f32(float* __restrict__ out, const float* __restrict__ x,
                                        const void* __restrict__ w, const float* __restrict__ bias,
                                        unsigned m, unsigned n, unsigned k, unsigned activation,
                                        unsigned a_row0, unsigned wstride_op, unsigned onehot,
                                        unsigned flags, unsigned slice, unsigned nblk, float* arena) {
    const unsigned stride = wstride_op ? wstride_op : k;
    x += (size_t)a_row0 * k;
    const SpRowF32 la{x, k, (k & 3u) == 0 && sp_aligned(x, 16)};
    const SpDenseEpi ep{out, bias, w, n, stride, wstride_op ? onehot : 0xFFFFFFFFu, flags,
                        activation == 1u};
    if (flags & 4u) {
        const __nv_bfloat16* wb = (const __nv_bfloat16*)w;
        sp_gemm(m, n, k, la, SpRowBf16{wb, k, stride, (stride & 3u) == 0 && sp_aligned(wb, 8)}, ep,
                slice, nblk, arena);
    } else {
        const float* wf = (const float*)w;
        sp_gemm(m, n, k, la, SpRowF32S{wf, k, stride, (stride & 3u) == 0 && sp_aligned(wf, 16)}, ep,
                slice, nblk, arena);
    }
}

/* GemmF32 (180): bf16 x bf16 -> f32. */
struct SpPlainEpi {
    float* out; unsigned n;
    __device__ void operator()(unsigned m, unsigned c, float acc) const { out[(size_t)m * n + c] = acc; }
};
static __device__ void d_gemm_f32(float* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ w, unsigned m, unsigned n,
                                  unsigned k, unsigned slice, unsigned nblk, float* arena) {
    const bool vec = (k & 3u) == 0;
    sp_gemm(m, n, k, SpRowBf16{x, k, k, vec && sp_aligned(x, 8)},
            SpRowBf16{w, k, k, vec && sp_aligned(w, 8)}, SpPlainEpi{out, n}, slice, nblk, arena);
}

/* Q8GemmF32 (163). k must be a multiple of 32 (Q8_0 block). */
struct SpQ8Epi {
    float* out; const float* bias; unsigned n, silu;
    __device__ void operator()(unsigned m, unsigned c, float acc) const {
        const float v = bias ? __fadd_rn(acc, bias[c]) : acc;
        out[(size_t)m * n + c] = silu ? sp_silu(v) : v;
    }
};
static __device__ void d_q8_gemm_f32(float* __restrict__ out, const float* __restrict__ x,
                                     const uint8_t* __restrict__ w, const float* __restrict__ bias,
                                     unsigned m, unsigned n, unsigned k, unsigned activation,
                                     unsigned a_row0, unsigned slice, unsigned nblk, float* arena) {
    if (k & 31u) { __trap(); return; }
    x += (size_t)a_row0 * k;
    sp_gemm(m, n, k, SpRowF32{x, k, sp_aligned(x, 16)}, SpRowQ8{w, k},
            SpQ8Epi{out, bias, n, activation == 1u}, slice, nblk, arena);
}

/* Conv2dF32 (176). flags (fj1): 1 depthwise, 2 relu, [3:2] out layout, [5:4] in layout,
 * 64 f32 weights (else f16), 128 bf16 erf-GELU. Layouts: 0 NFWC, 1 NFCW, 2 NCFW. */
struct SpConvEpi {
    float* out; const float* bias; unsigned oc, of, ow, layout, gelu, relu;
    __device__ void store(unsigned pos, unsigned c, float value) const {
        if (gelu) value = sp_gelu_erf_bf16(value);
        if (relu) value = fmaxf(value, 0.0f);
        const unsigned ox = pos % ow, oy = (pos / ow) % of, b = pos / (ow * of);
        size_t oi;
        if (layout == 0u) oi = (((size_t)b * of + oy) * ow + ox) * oc + c;
        else if (layout == 1u) oi = (((size_t)b * of + oy) * oc + c) * ow + ox;
        else oi = (((size_t)b * oc + c) * of + oy) * ow + ox;
        out[oi] = value;
    }
    __device__ void operator()(unsigned pos, unsigned c, float acc) const {
        store(pos, c, bias ? __fadd_rn(acc, bias[c]) : acc);
    }
};
static __device__ void d_conv2d_f32(float* __restrict__ out, const float* __restrict__ x,
                                    const void* __restrict__ w, const float* __restrict__ bias,
                                    unsigned frames, unsigned width, unsigned ic, unsigned oc,
                                    unsigned kernel, unsigned stride, unsigned pad_before,
                                    unsigned pad_after, unsigned flags, unsigned batches,
                                    unsigned slice, unsigned nblk, float* arena) {
    batches = batches ? batches : 1u;
    if (!kernel || !stride || frames + pad_before + pad_after < kernel ||
        width + pad_before + pad_after < kernel) return;
    const unsigned of = (frames + pad_before + pad_after - kernel) / stride + 1u;
    const unsigned ow = (width + pad_before + pad_after - kernel) / stride + 1u;
    const unsigned depthwise = flags & 1u, out_layout = (flags >> 2) & 3u, in_layout = (flags >> 4) & 3u;
    const unsigned weight_f32 = flags & 64u;
    if (out_layout > 2u || in_layout > 2u) return;
    const uint64_t positions64 = (uint64_t)batches * of * ow;
    if (positions64 * oc > 0xFFFFFFFFull) return;
    const unsigned positions = (unsigned)positions64;
    const SpConvEpi ep{out, bias, oc, of, ow, out_layout, flags & 128u, flags & 2u};
    const SpConvGeom g{frames, width, ic, kernel, stride, pad_before, of, ow, in_layout};
    if (!depthwise) {
        const unsigned k = ic * kernel * kernel;
        const SpRowConv la{x, g, k};
        if (weight_f32)
            sp_gemm(positions, oc, k, la, SpRowF32{(const float*)w, k, (k & 3u) == 0 && sp_aligned(w, 16)}, ep,
                    slice, nblk, arena);
        else sp_gemm(positions, oc, k, la, SpRowF16{(const uint16_t*)w, k}, ep, slice, nblk, arena);
        return;
    }
    const unsigned count = positions * oc;
    for (unsigned index = slice * PLOW_NV_THREADS + threadIdx.x; index < count;
         index += nblk * PLOW_NV_THREADS) {
        const unsigned c = index % oc, pos = index / oc;
        const SpConvPos s = SpRowConv{x, g, 0}.row(pos, positions);
        double sum = bias ? bias[c] : 0.0;
        for (unsigned ky = 0; ky < kernel; ky++) {
            const int iy = s.iy0 + (int)ky;
            if (iy < 0 || iy >= (int)frames) continue;
            for (unsigned kx = 0; kx < kernel; kx++) {
                const int ix = s.ix0 + (int)kx;
                if (ix < 0 || ix >= (int)width) continue;
                size_t xi;
                if (in_layout == 0u) xi = (((size_t)s.b * frames + iy) * width + ix) * ic + c;
                else if (in_layout == 1u) xi = (((size_t)s.b * frames + iy) * ic + c) * width + ix;
                else xi = (((size_t)s.b * ic + c) * frames + iy) * width + ix;
                const size_t wi = ((size_t)c * kernel + ky) * kernel + kx;
                const float wv = weight_f32 ? ((const float*)w)[wi] : sp_f16(((const uint16_t*)w)[wi]);
                sum += (double)x[xi] * wv;
            }
        }
        ep.store(pos, c, (float)sum);
    }
}

/* LayerNormF32 (164). flags: 1 bf16-round the output; 2 float statistics summed in row order
 * (else double). The ordered path stages 32 rows x 64 columns through smem and sums each row
 * serially in one lane, which is the only way to reproduce the golden's float rounding. */
static __device__ void d_layernorm_f32(float* __restrict__ out, const float* __restrict__ x,
                                       const float* __restrict__ gamma, const float* __restrict__ beta,
                                       unsigned rows, unsigned feat, unsigned flags, float eps,
                                       unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (flags & 2u) {
        float* xs = arena;
        float* stat = arena + 32 * 65;
        const float ff = (float)feat;
        for (unsigned base = slice * 32u; base < rows; base += nblk * 32u) {
            const unsigned nr = rows - base < 32u ? rows - base : 32u;
            float mean = 0.f, acc = 0.f;
            for (int pass = 0; pass < 2; pass++) {
                acc = 0.f;
                for (unsigned c0 = 0; c0 < feat; c0 += 64u) {
                    const unsigned nc = feat - c0 < 64u ? feat - c0 : 64u;
                    for (unsigned e = threadIdx.x; e < 32u * 64u; e += PLOW_NV_THREADS) {
                        const unsigned r = e >> 6, c = e & 63u;
                        xs[r * 65u + c] = (r < nr && c < nc) ? x[(size_t)(base + r) * feat + c0 + c] : 0.f;
                    }
                    __syncthreads();
                    if (warp == 0 && lane < nr) {
                        const float* xr = xs + lane * 65u;
                        if (pass == 0)
                            for (unsigned c = 0; c < nc; c++) acc = __fadd_rn(acc, xr[c]);
                        else
                            for (unsigned c = 0; c < nc; c++) {
                                const float v = __fsub_rn(xr[c], mean);
                                acc = __fadd_rn(acc, __fmul_rn(v, v));
                            }
                    }
                    __syncthreads();
                }
                if (pass == 0) mean = __fdiv_rn(acc, ff);
            }
            if (warp == 0 && lane < nr) {
                stat[lane] = mean;
                stat[32 + lane] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc, ff), eps)));
            }
            __syncthreads();
            for (unsigned e = threadIdx.x; e < nr * feat; e += PLOW_NV_THREADS) {
                const unsigned r = e / feat, c = e - r * feat;
                const size_t i = (size_t)(base + r) * feat + c;
                float v = __fmul_rn(__fsub_rn(x[i], stat[r]), stat[32 + r]);
                v = __fadd_rn(__fmul_rn(v, gamma ? gamma[c] : 1.0f), beta ? beta[c] : 0.0f);
                out[i] = flags & 1u ? sp_bf16(v) : v;
            }
            __syncthreads();
        }
        return;
    }
    for (unsigned row = slice * PLOW_NV_WARPS + warp; row < rows; row += nblk * PLOW_NV_WARPS) {
        const float* xr = x + (size_t)row * feat;
        float* yr = out + (size_t)row * feat;
        double sum = 0.0;
        for (unsigned i = lane; i < feat; i += 32u) sum += xr[i];
        const float mean = (float)(sp_warp_sum_d(sum) / feat);
        double sq = 0.0;
        for (unsigned i = lane; i < feat; i += 32u) {
            const double v = __fsub_rn(xr[i], mean);
            sq += v * v;
        }
        const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn((float)(sp_warp_sum_d(sq) / feat), eps)));
        for (unsigned i = lane; i < feat; i += 32u) {
            float v = __fmul_rn(__fsub_rn(xr[i], mean), inv);
            v = __fadd_rn(__fmul_rn(v, gamma ? gamma[i] : 1.0f), beta ? beta[i] : 0.0f);
            yr[i] = flags & 1u ? sp_bf16(v) : v;
        }
    }
}

#define SP_FOR_EACH(i, n) \
    for (unsigned i = slice * PLOW_NV_THREADS + threadIdx.x; i < (n); i += nblk * PLOW_NV_THREADS)

/* ScaledAddF32 (165): a + s*b; flags 1 = bf16-round. */
static __device__ void d_scaled_add_f32(float* out, const float* a, const float* b, unsigned n,
                                        float s, unsigned flags, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) {
        const float v = __fadd_rn(a[i], __fmul_rn(s, b[i]));
        out[i] = flags & 1u ? sp_bf16(v) : v;
    }
}

/* GluF32 (166): x is [rows][2*width], out = a * sigmoid(b). */
static __device__ void d_glu_f32(float* out, const float* x, unsigned rows, unsigned width,
                                 unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, rows * width) {
        const unsigned r = i / width, c = i - r * width;
        out[i] = __fmul_rn(x[(size_t)r * 2u * width + c], sp_sigmoid(x[(size_t)r * 2u * width + width + c]));
    }
}

static __device__ void d_silu_f32(float* out, const float* x, unsigned n, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) out[i] = sp_silu(x[i]);
}

static __device__ void d_relu_f32(float* out, const float* x, unsigned n, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) out[i] = fmaxf(x[i], 0.0f);
}

static __device__ void d_broadcast_add_f32(float* out, const float* m, const float* v, unsigned rows,
                                           unsigned width, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, rows * width) out[i] = __fadd_rn(m[i], v[i % width]);
}

/* CausalDepthwiseConv1dF32 (167): f16 weights [channel][kernel], taps summed in order. */
static __device__ void d_causal_dwconv1d_f32(float* out, const float* x, const uint16_t* w,
                                             unsigned rows, unsigned channels, unsigned kernel,
                                             unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, rows * channels) {
        const unsigned r = i / channels, c = i - r * channels;
        float sum = 0.f;
        for (unsigned tap = 0; tap < kernel; tap++) {
            const int src = (int)r + (int)tap + 1 - (int)kernel;
            if (src >= 0)
                sum = __fadd_rn(sum, __fmul_rn(x[(size_t)src * channels + c], sp_f16(w[(size_t)c * kernel + tap])));
        }
        out[i] = sum;
    }
}

/* EmbedF16F32 (171): one fp16 table row selected by a device-resident token. */
static __device__ void d_embed_f16_f32(float* out, const uint16_t* table, const unsigned* token,
                                       unsigned vocab, unsigned width, unsigned slice, unsigned nblk) {
    const unsigned t = *token;
    if (t >= vocab) return;
    SP_FOR_EACH(i, width) out[i] = sp_f16(table[(size_t)t * width + i]);
}

/* LstmCellF32 (172): gates = [i | f | g | o] x width. */
static __device__ void d_lstm_cell_f32(float* h_new, float* c_new, const float* gates,
                                       const float* c_prev, unsigned width, unsigned slice,
                                       unsigned nblk) {
    SP_FOR_EACH(i, width) {
        const float ig = sp_sigmoid(gates[i]);
        const float fg = sp_sigmoid(gates[width + i]);
        const float cell = sp_tanhf(gates[2u * width + i]);
        const float og = sp_sigmoid(gates[3u * width + i]);
        const float c = __fadd_rn(__fmul_rn(fg, c_prev[i]), __fmul_rn(ig, cell));
        c_new[i] = c;
        h_new[i] = __fmul_rn(og, sp_tanhf(c));
    }
}

/* ArgmaxF32 (173): first index of the row maximum; a NaN never wins (golden: strict >). */
static __device__ void d_argmax_f32(unsigned* ids, const float* x, unsigned rows, unsigned width,
                                    unsigned slice, unsigned nblk, float* arena) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    float* pv = arena;
    unsigned* pi = (unsigned*)(arena + PLOW_NV_WARPS);
    for (unsigned row = slice; row < rows; row += nblk) {
        const float* xr = x + (size_t)row * width;
        float bv = -INFINITY;
        unsigned bi = 0xFFFFFFFFu;
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
            if (xr[i] > bv) { bv = xr[i]; bi = i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { pv[warp] = bv; pi[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned w = 1; w < PLOW_NV_WARPS; w++)
                if (pv[w] > bv || (pv[w] == bv && pi[w] < bi)) { bv = pv[w]; bi = pi[w]; }
            ids[row] = (bi == 0xFFFFFFFFu || isnan(xr[0])) ? 0u : bi;
        }
        __syncthreads();
    }
}

/* PackNcfwRowsF32 (177): [batch][ch][frame][width] -> rows (batch*width) of ch*frames. */
static __device__ void d_pack_ncfw_rows_f32(float* out, const float* x, unsigned rows,
                                            unsigned channels, unsigned frames, unsigned width,
                                            unsigned batches, unsigned slice, unsigned nblk) {
    const uint64_t count64 = (uint64_t)rows * channels * frames;
    if (!rows || !channels || !frames || !width || !batches || rows > (uint64_t)batches * width ||
        count64 > 0xFFFFFFFFull) return;
    const unsigned row_width = channels * frames;
    SP_FOR_EACH(i, (unsigned)count64) {
        const unsigned row = i / row_width, col = i - row * row_width;
        const unsigned b = row / width, p = row - b * width;
        const unsigned ch = col / frames, f = col - ch * frames;
        out[i] = x[((size_t)b * channels + ch) * frames * width + (size_t)f * width + p];
    }
}

/* GroupedAttentionF32 (178): block-diagonal attention over groups of `group_rows` rows, only the
 * first *valid_rows rows. One block per (group, head): K for the group is staged transposed-
 * padded in smem; a warp owns one query row, lanes own keys for the float-ordered score dot and
 * columns for the float-ordered P.V sum. flags: 1 bf16 score, 2 bf16 probability, 4 bf16 out. */
static __device__ void d_grouped_attention_f32(float* __restrict__ context, const float* __restrict__ query,
                                               const float* __restrict__ key, const float* __restrict__ value,
                                               const unsigned* valid, unsigned rows, unsigned width,
                                               unsigned hw, unsigned group_rows, unsigned flags,
                                               unsigned slice, unsigned nblk, float* arena) {
    const unsigned valid_rows = valid ? *valid : rows;
    if (hw == 0 || width % hw != 0 || group_rows == 0 || group_rows > 256u || valid_rows == 0 ||
        valid_rows > rows) return;
    const unsigned heads = width / hw, groups = (valid_rows + group_rows - 1) / group_rows;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    float* sc = arena + warp * 256u;
    const bool staged = group_rows * (hw + 1u) + SPA_SCORE_FLOATS <= SP_ARENA_FLOATS;
    float* ks = arena + SPA_SCORE_FLOATS;
    const float scale = __fsqrt_rn((float)hw);
    for (unsigned item = slice; item < groups * heads; item += nblk) {
        const unsigned g = item / heads, head = item - g * heads;
        const unsigned first = g * group_rows;
        const unsigned last = first + group_rows < valid_rows ? first + group_rows : valid_rows;
        const unsigned n = last - first;
        const float* kb = key + (size_t)first * width + head * hw;
        unsigned kstride = width;
        if (staged) {
            for (unsigned e = threadIdx.x; e < n * hw; e += PLOW_NV_THREADS) {
                const unsigned r = e / hw, c = e - r * hw;
                ks[r * (hw + 1u) + c] = kb[(size_t)r * width + c];
            }
            kb = ks;
            kstride = hw + 1u;
            __syncthreads();
        }
        for (unsigned row = first + warp; row < last; row += PLOW_NV_WARPS) {
            const float* q = query + (size_t)row * width + head * hw;
            float mx = -INFINITY;
            for (unsigned j = lane; j < n; j += 32u) {
                const float* kr = kb + (size_t)j * kstride;
                float s = 0.f;
                for (unsigned c = 0; c < hw; c++) s = __fadd_rn(s, __fmul_rn(q[c], kr[c]));
                if (flags & 1u) s = sp_bf16(s);
                s = __fdiv_rn(s, scale);
                sc[j] = s;
                mx = fmaxf(mx, s);
            }
            mx = sp_warp_max(mx);
            for (unsigned j = lane; j < n; j += 32u) sc[j] = sp_expf(__fsub_rn(sc[j], mx));
            __syncwarp();
            float den = 0.f;
            for (unsigned j = 0; j < n; j++) den = __fadd_rn(den, sc[j]);
            __syncwarp();
            for (unsigned j = lane; j < n; j += 32u) {
                const float p = __fdiv_rn(sc[j], den);
                sc[j] = flags & 2u ? sp_bf16(p) : p;
            }
            __syncwarp();
            const float* vb = value + (size_t)first * width + head * hw;
            float* o = context + (size_t)row * width + head * hw;
            for (unsigned c = lane; c < hw; c += 32u) {
                float sum = 0.f;
                for (unsigned j = 0; j < n; j++) sum = __fadd_rn(sum, __fmul_rn(sc[j], vb[(size_t)j * width + c]));
                o[c] = flags & 4u ? sp_bf16(sum) : sum;
            }
            __syncwarp();
        }
        __syncthreads();
    }
}

/* RelativeAttentionF32 (168): Transformer-XL scores (k.(q+u) + p.(q+v)) / sqrt(hw) in double,
 * optional chunked-left window (i4 = left chunks, UINT32_MAX = full). A warp owns one
 * (row, head); scores are recomputed per pass so any row count works. */
__device__ __forceinline__ float sp_rel_score(const float* q, const float* k, const float* p,
                                              const float* u, const float* v, unsigned hw,
                                              unsigned lane) {
    double content = 0.0, relative = 0.0;
    for (unsigned i = lane; i < hw; i += 32u) {
        content += (double)k[i] * __fadd_rn(q[i], u[i]);
        relative += (double)p[i] * __fadd_rn(q[i], v[i]);
    }
    content = sp_warp_sum_d(content);
    relative = sp_warp_sum_d(relative);
    return (float)((content + relative) / sqrt((double)hw));
}
#define SPR_COLS 8
static __device__ void d_relative_attention_f32(float* __restrict__ context, const float* __restrict__ query,
                                                const float* __restrict__ key, const float* __restrict__ value,
                                                const float* __restrict__ position, const float* __restrict__ bias_u,
                                                const float* __restrict__ bias_v, unsigned rows, unsigned width,
                                                unsigned heads, unsigned chunk, unsigned left_chunks,
                                                unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, hw = width / heads;
    for (unsigned item = slice * PLOW_NV_WARPS + warp; item < rows * heads; item += nblk * PLOW_NV_WARPS) {
        const unsigned qr = item / heads, head = item - qr * heads;
        unsigned first = 0, last = rows;
        if (left_chunks != 0xFFFFFFFFu) {
            const unsigned qc = qr / chunk;
            first = (qc > left_chunks ? qc - left_chunks : 0u) * chunk;
            last = (qc + 1u) * chunk;
            if (last > rows) last = rows;
        }
        const size_t ho = (size_t)head * hw;
        const float* q = query + (size_t)qr * width + ho;
        const float* u = bias_u + ho;
        const float* v = bias_v + ho;
#define SPR_SCORE(kr) sp_rel_score(q, key + (size_t)(kr) * width + ho, \
                                   position + (size_t)(rows - 1u + (kr) - qr) * width + ho, u, v, hw, lane)
        float mx = -INFINITY;
        for (unsigned kr = first; kr < last; kr++) mx = fmaxf(mx, SPR_SCORE(kr));
        float den = 0.f;
        for (unsigned kr = first; kr < last; kr++) den = __fadd_rn(den, sp_expf(__fsub_rn(SPR_SCORE(kr), mx)));
        for (unsigned c0 = 0; c0 < hw; c0 += 32u * SPR_COLS) {
            float acc[SPR_COLS];
#pragma unroll
            for (int j = 0; j < SPR_COLS; j++) acc[j] = 0.f;
            for (unsigned kr = first; kr < last; kr++) {
                const float p = __fdiv_rn(sp_expf(__fsub_rn(SPR_SCORE(kr), mx)), den);
                const float* vr = value + (size_t)kr * width + ho;
#pragma unroll
                for (int j = 0; j < SPR_COLS; j++) {
                    const unsigned c = c0 + lane + 32u * j;
                    if (c < hw) acc[j] = __fadd_rn(acc[j], __fmul_rn(p, vr[c]));
                }
            }
            float* o = context + (size_t)qr * width + ho;
#pragma unroll
            for (int j = 0; j < SPR_COLS; j++) {
                const unsigned c = c0 + lane + 32u * j;
                if (c < hw) o[c] = acc[j];
            }
        }
#undef SPR_SCORE
    }
}

/* The interpreter arm for every op above (operand slots per f32_primitives.c). */
#define SP_TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])
static __device__ void d_speech_f32(const PlowDevInst* in, void* const* T, unsigned slice,
                                    unsigned nblk, float* arena) {
    switch (in->op) {
    case PLOW_DOP_Q8_GEMM_F32:
        d_q8_gemm_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const uint8_t*)SP_TEN(2), (const float*)SP_TEN(3),
                      in->i[0], in->i[1], in->i[2], in->i[3], in->i[4], slice, nblk, arena);
        break;
    case PLOW_DOP_LAYERNORM_F32:
        d_layernorm_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const float*)SP_TEN(2), (const float*)SP_TEN(3),
                        in->i[0], in->i[1], in->i[2], in->fj[0].f, slice, nblk, arena);
        break;
    case PLOW_DOP_SCALED_ADD_F32:
        d_scaled_add_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const float*)SP_TEN(2), in->i[0],
                         in->fj[0].f, in->i[1], slice, nblk);
        break;
    case PLOW_DOP_GLU_F32:
        d_glu_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), in->i[0], in->i[1], slice, nblk);
        break;
    case PLOW_DOP_CAUSAL_DEPTHWISE_CONV1D_F32:
        d_causal_dwconv1d_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const uint16_t*)SP_TEN(2), in->i[0],
                              in->i[1], in->i[2], slice, nblk);
        break;
    case PLOW_DOP_RELATIVE_ATTENTION_F32:
        d_relative_attention_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const float*)SP_TEN(2),
                                 (const float*)SP_TEN(3), (const float*)SP_TEN(4), (const float*)SP_TEN(5),
                                 (const float*)SP_TEN(6), in->i[0], in->i[1], in->i[2], in->i[3], in->i[4],
                                 slice, nblk);
        break;
    case PLOW_DOP_SILU_F32:
        d_silu_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), in->i[0], slice, nblk);
        break;
    case PLOW_DOP_DENSE_GEMM_F32:
        d_dense_gemm_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), SP_TEN(2), (const float*)SP_TEN(3), in->i[0],
                         in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[6], in->i[7], slice,
                         nblk, arena);
        break;
    case PLOW_DOP_EMBED_F16_F32:
        d_embed_f16_f32((float*)SP_TEN(0), (const uint16_t*)SP_TEN(1), (const unsigned*)SP_TEN(2), in->i[0],
                        in->i[1], slice, nblk);
        break;
    case PLOW_DOP_LSTM_CELL_F32:
        d_lstm_cell_f32((float*)SP_TEN(0), (float*)SP_TEN(1), (const float*)SP_TEN(2), (const float*)SP_TEN(3),
                        in->i[0], slice, nblk);
        break;
    case PLOW_DOP_ARGMAX_F32:
        d_argmax_f32((unsigned*)SP_TEN(0), (const float*)SP_TEN(1), in->i[0], in->i[1], slice, nblk, arena);
        break;
    case PLOW_DOP_RELU_F32:
        d_relu_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), in->i[0], slice, nblk);
        break;
    case PLOW_DOP_BROADCAST_ADD_F32:
        d_broadcast_add_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const float*)SP_TEN(2), in->i[0],
                            in->i[1], slice, nblk);
        break;
    case PLOW_DOP_CONV2D_F32:
        d_conv2d_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), SP_TEN(2), (const float*)SP_TEN(3), in->i[0],
                     in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[6], in->i[7],
                     in->fj[1].u, in->fj[2].u, slice, nblk, arena);
        break;
    case PLOW_DOP_PACK_NCFW_ROWS_F32:
        d_pack_ncfw_rows_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), in->i[0], in->i[1], in->i[2],
                             in->i[3], in->i[4], slice, nblk);
        break;
    case PLOW_DOP_GROUPED_ATTENTION_F32:
        d_grouped_attention_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), (const float*)SP_TEN(2),
                                (const float*)SP_TEN(3), (const unsigned*)SP_TEN(4), in->i[0], in->i[1],
                                in->i[2], in->i[3], in->i[4], slice, nblk, arena);
        break;
    case PLOW_DOP_GEMM_F32:
        d_gemm_f32((float*)SP_TEN(0), (const __nv_bfloat16*)SP_TEN(1), (const __nv_bfloat16*)SP_TEN(2), in->i[0],
                   in->i[1], in->i[2], slice, nblk, arena);
        break;
    default: __trap(); break;
    }
}
#undef SP_TEN
