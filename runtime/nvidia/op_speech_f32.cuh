/* op_speech_f32.cuh — FP32 speech primitives (ops 163-178, 180, 195-203) for the NVIDIA interpreter.
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
/* Covers the GEMM ring, AttentionF32's Q/K/V stages (SpfShape), and
 * GroupedAttention's K stage for group_rows*(head_width+1) <= SP_ARENA_FLOATS - 2048 (larger
 * groups read K from global instead). */
#define SP_ARENA_FLOATS 25856
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

/* One 128x128 output tile at (m0, n0). */
template <class LA, class LB, class EP>
static __device__ __forceinline__ void sp_gemm_tile(unsigned m0, unsigned n0, unsigned M, unsigned N,
                                                    unsigned K, const LA& la, const LB& lb,
                                                    const EP& ep, float* arena) {
    constexpr unsigned KT = SPG_BK / 4, RPP = PLOW_NV_THREADS / KT;
    constexpr unsigned AP = SPG_BM / RPP, BP = SPG_BN / RPP;
    static_assert(AP * RPP == SPG_BM && BP * RPP == SPG_BN, "loader coverage");
    float* As = arena;
    float* Bs = arena + 2 * SPG_BK * SPG_LDA;
    const unsigned tid = threadIdx.x, tx = tid & 15u, ty = tid >> 4;
    const unsigned lr = tid / KT, lk = (tid % KT) * 4u;
    const unsigned nk = (K + SPG_BK - 1) / SPG_BK;
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

template <class LA, class LB, class EP>
static __device__ __forceinline__ void sp_gemm(unsigned M, unsigned N, unsigned K, const LA& la,
                                               const LB& lb, const EP& ep, unsigned slice,
                                               unsigned nblk, float* arena) {
    const unsigned tn = (N + SPG_BN - 1) / SPG_BN;
    const unsigned ntiles = ((M + SPG_BM - 1) / SPG_BM) * tn;
    for (unsigned tile = slice; tile < ntiles; tile += nblk)
        sp_gemm_tile((tile / tn) * SPG_BM, (tile % tn) * SPG_BN, M, N, K, la, lb, ep, arena);
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
/* ---- generic signal ops (195-203): golden f32_primitives.c, codes packet::dev::ACT_* ---------- */

/* sin(y)^2 has period pi: Cody-Waite reduce by pi to [-pi/2, pi/2], then the SFU sine (the
 * native SNAC / S3Gen spelling; ~4e-7 absolute, well inside the golden gate). */
__device__ __forceinline__ float sp_sin2(float y) {
    const float k = rintf(y * 0.318309886183790672f);
    float r = fmaf(k, -3.14159274101257324f, y);
    r = fmaf(k, 8.74227800037248e-08f, r);
    const float s = __sinf(r);
    return s * s;
}
__device__ __forceinline__ float sp_snake(float x, float a) { return fmaf(__frcp_rn(a + 1e-9f), sp_sin2(x * a), x); }

/* Calls body(f) once with the activation `kind` as a functor f(x, p0) (packet::dev::ACT_*), so a
 * loop dispatches once instead of switching per element. */
template <class B>
__device__ __forceinline__ void sp_with_act(unsigned kind, float p1, const B& body) {
    switch (kind) {
    case 1: body([](float x, float) { return tanhf(x); }); break;
    case 2: body([](float x, float) { return sinf(x); }); break;
    case 3: body([](float x, float) { return cosf(x); }); break;
    case 4: body([](float x, float) { return expf(x); }); break;
    case 5: body([](float x, float) { return fabsf(x); }); break;
    case 6: body([](float x, float) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }); break;
    case 7: body([](float x, float) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }); break;
    case 8: body([](float x, float) { return x > 0.0f ? x : expm1f(x); }); break;
    case 9: body([](float x, float p0) { return x >= 0.0f ? x : __fmul_rn(x, p0); }); break;
    case 10: body([](float x, float) { return __fmul_rn(x, tanhf(log1pf(expf(x)))); }); break;
    case 11:
        body([](float x, float) {
            return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, erff(__fmul_rn(x, 0.70710678118654752f))));
        });
        break;
    case 12: body([](float x, float p0) { return sp_snake(x, p0); }); break;
    case 13: body([p1](float x, float p0) { return fminf(fmaxf(x, p0), p1); }); break;
    case 14: body([p1](float x, float p0) { return __fadd_rn(__fmul_rn(x, p0), p1); }); break;
    case 15: body([](float x, float) { return x > 0.0f ? x : 0.0f; }); break;
    default: body([](float x, float) { return x; }); break;
    }
}
__device__ __forceinline__ float sp_act(unsigned kind, float x, float p0, float p1) {
    switch (kind) {
    case 1: return tanhf(x);
    case 2: return sinf(x);
    case 3: return cosf(x);
    case 4: return expf(x);
    case 5: return fabsf(x);
    case 6: return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
    case 7: return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
    case 8: return x > 0.0f ? x : expm1f(x);
    case 9: return x >= 0.0f ? x : __fmul_rn(x, p0);
    case 10: return __fmul_rn(x, tanhf(log1pf(expf(x))));
    case 11: return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, erff(__fmul_rn(x, 0.70710678118654752f))));
    case 12: return sp_snake(x, p0);
    case 13: return fminf(fmaxf(x, p0), p1);
    case 14: return __fadd_rn(__fmul_rn(x, p0), p1);
    case 15: return x > 0.0f ? x : 0.0f;
    default: return x;
    }
}
/* Grid-stride loop with U loads in flight per thread (the persistent grid runs one block per SM,
 * so memory-level parallelism has to come from registers): load(e) for U elements, then
 * store(e, value) for each. */
template <unsigned U, class L, class S>
__device__ __forceinline__ void sp_batched(unsigned n, unsigned slice, unsigned nblk, const L& load,
                                           const S& store) {
    using V = decltype(load(0u));
    const unsigned step = nblk * PLOW_NV_THREADS;
    for (unsigned base = slice * PLOW_NV_THREADS + threadIdx.x; base < n; base += step * U) {
        V v[U];
#pragma unroll
        for (unsigned u = 0; u < U; u++)
            if (base + u * step < n) v[u] = load(base + u * step);
#pragma unroll
        for (unsigned u = 0; u < U; u++)
            if (base + u * step < n) store(base + u * step, v[u]);
    }
}
__device__ __forceinline__ bool sp_act_conv_ok(unsigned kind) { return kind != 13u && kind != 14u; }
/* Out-of-line form for the convolution operand paths: one call site instead of the switch inlined
 * into every unrolled load and epilogue element keeps the GEMM tile's code compact. */
static __device__ __noinline__ float sp_act_call(unsigned kind, float x, float p0) { return sp_act(kind, x, p0, 0.f); }

__device__ __forceinline__ float4 sp_ld4(const float* p, bool vec) {
    if (vec) return __ldg((const float4*)p);
    return make_float4(p[0], p[1], p[2], p[3]);
}

/* GatherRowsF32 (195). */
static __device__ void d_gather_rows_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const void* table = SP_TEN(1);
    const unsigned* index = (const unsigned*)SP_TEN(2);
    const unsigned rows = in->i[0], width = in->i[1], vocab = in->i[2];
    const unsigned per_item = in->i[3] ? in->i[3] : rows, repeat = in->i[4] ? in->i[4] : 1u;
    const unsigned flags = in->i[7], out_stride = in->fj[1].u ? in->fj[1].u : width, col0 = in->fj[2].u;
    if (!width || !per_item || (uint64_t)rows * width > 0xFFFFFFFFull) return;
    sp_batched<8>(
        rows * width, slice, nblk,
        [&](unsigned e) {
            const unsigned row = e / width, c = e - row * width;
            const unsigned item = row / per_item, local = (row - item * per_item) / repeat;
            const unsigned src = index ? index[(size_t)item * in->i[5] + local] : local;
            if (src >= vocab) return 0.f;
            const size_t ti = ((size_t)item * in->i[6] + src) * width + c;
            return flags & 1u ? sp_f16(((const uint16_t*)table)[ti]) : ((const float*)table)[ti];
        },
        [&](unsigned e, float v) {
            const unsigned row = e / width;
            float* o = out + (size_t)row * out_stride + col0 + (e - row * width);
            *o = flags & 2u ? __fadd_rn(*o, v) : v;
        });
}

/* CopyColsF32 (196). */
static __device__ void d_copy_cols_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const float* x = (const float*)SP_TEN(1);
    const unsigned items = in->i[0], rows = in->i[1], cols = in->i[2];
    if (!rows || !cols || (uint64_t)items * rows * cols > 0xFFFFFFFFull) return;
    sp_batched<8>(
        items * rows * cols, slice, nblk,
        [&](unsigned e) {
            const unsigned rr = e / cols, c = e - rr * cols, item = rr / rows, row = rr - item * rows;
            return x[(size_t)item * in->fj[1].u + (size_t)row * in->i[3] + in->i[4] + c];
        },
        [&](unsigned e, float v) {
            const unsigned rr = e / cols, c = e - rr * cols, item = rr / rows, row = rr - item * rows;
            out[(size_t)item * in->fj[2].u + (size_t)row * in->i[5] + in->i[6] + c] = v;
        });
}

/* UnaryF32 (199). */
static __device__ void d_unary_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const float* x = (const float*)SP_TEN(1);
    const float* param = (const float*)SP_TEN(2);
    const unsigned rows = in->i[0], width = in->i[1], kind = in->i[2];
    const unsigned stride = in->i[3] ? in->i[3] : width, col0 = in->i[4];
    if (!width || (uint64_t)rows * width > 0xFFFFFFFFull) return;
    const float p0 = in->fj[0].f;
    sp_with_act(kind, in->fj[1].f, [&](auto f) {
        sp_batched<4>(
            rows * width, slice, nblk,
            [&](unsigned e) {
                const unsigned row = e / width;
                return x[(size_t)row * stride + col0 + e - row * width];
            },
            [&](unsigned e, float v) {
                const unsigned row = e / width, c = e - row * width;
                out[(size_t)row * stride + col0 + c] = f(v, param ? param[c] : p0);
            });
    });
}

/* BinaryF32 (200). */
static __device__ void d_binary_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const float* a = (const float*)SP_TEN(1);
    const float* b = (const float*)SP_TEN(2);
    const unsigned items = in->i[0], rows = in->i[1], width = in->i[2], op = in->i[3];
    if (!rows || !width || op > 5u || (uint64_t)items * rows * width > 0xFFFFFFFFull) return;
    const bool scaled = in->i[7] & 1u;
    const float scale = in->fj[0].f;
    const auto run = [&](auto f) {
        sp_batched<4>(
            items * rows * width, slice, nblk,
            [&](unsigned e) {
                const unsigned rr = e / width, c = e - rr * width, item = rr / rows, row = rr - item * rows;
                return make_float2(a[e], b[(size_t)item * in->i[4] + (size_t)row * in->i[5] + (size_t)c * in->i[6]]);
            },
            [&](unsigned e, float2 v) {
                const float r = f(v.x, v.y);
                out[e] = scaled ? __fmul_rn(scale, r) : r;
            });
    };
    switch (op) {
    case 0: run([](float x, float y) { return __fadd_rn(x, y); }); break;
    case 1: run([](float x, float y) { return __fsub_rn(x, y); }); break;
    case 2: run([](float x, float y) { return __fmul_rn(x, y); }); break;
    case 3: run([](float x, float y) { return __fdiv_rn(x, y); }); break;
    case 4: run([](float x, float y) { return fmaxf(x, y); }); break;
    default: run([](float x, float y) { return fminf(x, y); }); break;
    }
}

/* CumSumF64 (201): one (item, column) per block; each thread scans a contiguous chunk of rows,
 * chunk offsets come from a serial fp64 scan of the 256 chunk sums. */
static __device__ void d_cumsum_f64(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk,
                                    float* arena) {
    void* out = SP_TEN(0);
    const float* x = (const float*)SP_TEN(1);
    const float* column_scale = (const float*)SP_TEN(2);
    const unsigned* lengths = (const unsigned*)SP_TEN(3);
    const unsigned items = in->i[0], rows = in->i[1], width = in->i[2];
    const unsigned xw = in->i[3] ? in->i[3] : width, flags = in->i[4];
    double* part = (double*)arena;
    const unsigned tid = threadIdx.x, chunk = (rows + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS;
    const unsigned lo = min(rows, tid * chunk), hi = min(rows, lo + chunk);
    for (unsigned w = slice; w < items * width; w += nblk) {
        const unsigned item = w / width, c = w - item * width;
        const unsigned length = lengths && lengths[item] < rows ? lengths[item] : rows;
        const float* xb = x + (size_t)item * rows * xw + c % xw;
        double s = 0.0;
        for (unsigned r = lo; r < min(hi, length); r++) s += (double)xb[(size_t)r * xw];
        part[tid] = s;
        __syncthreads();
        if (tid == 0) {
            double run = 0.0;
            for (unsigned i = 0; i < PLOW_NV_THREADS; i++) {
                const double v = part[i];
                part[i] = run;
                run += v;
            }
        }
        __syncthreads();
        double run = part[tid];
        const double scale = (double)(column_scale ? column_scale[c] : 1.0f) * (double)in->fj[0].f;
        for (unsigned r = lo; r < hi; r++) {
            const double value = r < length ? (double)xb[(size_t)r * xw] : 0.0;
            if (!(flags & 1u)) run += value;
            double v = run * scale;
            if (flags & 2u) v -= floor(v);
            v *= (double)in->fj[1].f;
            const size_t o = ((size_t)item * rows + r) * width + c;
            if (flags & 4u) ((double*)out)[o] = v;
            else ((float*)out)[o] = (float)v;
            if (flags & 1u) run += value;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ unsigned long long sp_mix64(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* RandF32 (202). The normal is spelled exactly as the counter generators it replaces
 * (sqrtf/logf/cospif, no fast math), so their streams are reproduced bit for bit. */
static __device__ void d_rand_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const unsigned long long* seed = (const unsigned long long*)SP_TEN(1);
    const unsigned items = in->i[0], rows = in->i[1], width = in->i[2];
    const unsigned stream = in->i[3], shift = in->i[4], ca = in->i[5] & 3u, cb = (in->i[5] >> 2) & 3u;
    const unsigned flags = in->fj[2].u;
    if (!rows || !width || shift > 63u || ca > 2u || cb > 2u ||
        (uint64_t)items * rows * width > 0xFFFFFFFFull) return;
    const unsigned long long skey = (unsigned long long)stream << shift;
    SP_FOR_EACH(e, items * rows * width) {
        const unsigned rr = e / width, c = e - rr * width, item = rr / rows, row = rr - item * rows;
        const unsigned a = (ca == 0u ? row : ca == 1u ? c : item) + in->i[6];
        const unsigned b = (cb == 0u ? row : cb == 1u ? c : item) + in->i[7];
        const unsigned long long h =
            sp_mix64(seed[flags & 2u ? 0u : item] ^ sp_mix64(skey ^ ((unsigned long long)a << 32) ^ b));
        float v;
        if (flags & 1u) {
            const float u1 = (float)((h >> 40) + 1) * (1.0f / 16777216.0f);
            const float u2 = (float)(sp_mix64(h) >> 40) * (1.0f / 16777216.0f);
            v = sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
        } else {
            v = (float)(h >> 40) * (1.0f / 16777216.0f);
        }
        out[e] = __fmul_rn(__fadd_rn(v, in->fj[1].f), in->fj[0].f);
    }
}

/* ---- Conv1dF32 (197) / ConvTranspose1dF32 (198) -------------------------------------------- */

__device__ __forceinline__ int sp_pad_row(int u, int length, unsigned mode) {
    if (u >= 0 && u < length) return u;
    if (mode == 1u) u = u < 0 ? -u : 2 * (length - 1) - u;
    else if (mode == 2u) u = u < 0 ? 0 : length - 1;
    else return -1;
    return u >= 0 && u < length ? u : -1;
}

struct SpConvArgs {
    float* out; const float* x; const void* w; const float* bias; const float* alpha;
    const float* residual; const unsigned* lengths;
    unsigned batch, in_rows, cin, cout, kernel, stride, dil, groups, cg, ng, before, after, mode, pre,
        post, wf16, out_rows, opad, cg_mul, cg_shift;
    float slope;
    /* c / cg for c < 2^31 (Granlund-Montgomery; the tap index of a GEMM column). */
    __device__ unsigned div_cg(unsigned c) const { return (__umulhi(c, cg_mul) + c) >> cg_shift; }
    __device__ unsigned length(unsigned b) const {
        return lengths && lengths[b] < in_rows ? lengths[b] : in_rows;
    }
    __device__ float w_at(size_t i) const {
        return wf16 ? sp_f16(((const uint16_t*)w)[i]) : ((const float*)w)[i];
    }
    __device__ float pre_at(float v, unsigned c) const {
        if (!pre) return v;
        const float p = alpha ? alpha[c] : slope;
        if (pre == 12u) return sp_snake(v, p);
        if (pre == 9u) return v >= 0.0f ? v : __fmul_rn(v, p);
        return sp_act_call(pre, v, p);
    }
    __device__ void store(size_t oi, unsigned o, float acc) const {
        float v = post ? sp_act_call(post, acc, alpha ? alpha[o] : slope) : acc;
        out[oi] = residual ? __fadd_rn(v, residual[oi]) : v;
    }
    /* Valid output rows of item b (Conv1d). */
    __device__ unsigned conv_len(unsigned b) const {
        const unsigned span = dil * (kernel - 1u) + 1u, padded = length(b) + before + after;
        return padded >= span ? (padded - span) / stride + 1u : 0u;
    }
    /* Valid output rows of item b (ConvTranspose1d). */
    __device__ int convt_len(unsigned b) const {
        const unsigned l = length(b);
        return l ? (int)((l - 1u) * stride + kernel + opad) - (int)(before + after) : 0;
    }
};

__device__ __forceinline__ bool sp_conv_args(const PlowDevInst* in, void* const* T, bool transpose,
                                             SpConvArgs& a) {
    a.out = (float*)SP_TEN(0); a.x = (const float*)SP_TEN(1); a.w = SP_TEN(2);
    a.bias = (const float*)SP_TEN(3); a.alpha = (const float*)SP_TEN(4);
    a.residual = (const float*)SP_TEN(5); a.lengths = (const unsigned*)SP_TEN(6);
    a.batch = in->i[0]; a.in_rows = in->i[1]; a.cin = in->i[2]; a.cout = in->i[3];
    a.kernel = in->i[4]; a.stride = in->i[5]; a.groups = in->i[7];
    a.dil = transpose ? 1u : in->i[6];
    a.opad = transpose ? in->i[6] : 0u;
    a.before = in->fj[1].u & 0xFFFFu; a.after = in->fj[1].u >> 16;
    const unsigned flags = in->fj[2].u;
    a.mode = transpose ? 0u : flags & 3u;
    a.pre = (flags >> 4) & 15u; a.post = (flags >> 8) & 15u; a.wf16 = (flags >> 12) & 1u;
    a.slope = in->fj[0].f;
    if (!a.kernel || !a.stride || !a.dil || !a.groups || a.cin % a.groups || a.cout % a.groups ||
        a.mode > 2u || !sp_act_conv_ok(a.pre) || !sp_act_conv_ok(a.post) || a.post == 12u || !a.in_rows)
        return false;
    a.cg = a.cin / a.groups; a.ng = a.cout / a.groups;
    a.cg_shift = 0;
    while ((1ull << a.cg_shift) < a.cg) a.cg_shift++;
    a.cg_mul = (unsigned)((((1ull << a.cg_shift) - a.cg) << 32) / a.cg + 1ull);
    if (transpose) {
        const long long full = (long long)(a.in_rows - 1u) * a.stride + a.kernel + a.opad;
        if (full <= (long long)(a.before + a.after)) return false;
        a.out_rows = (unsigned)(full - a.before - a.after);
    } else {
        const unsigned span = a.dil * (a.kernel - 1u) + 1u;
        if (a.in_rows + a.before + a.after < span) return false;
        a.out_rows = (a.in_rows + a.before + a.after - span) / a.stride + 1u;
    }
    return (uint64_t)a.batch * a.out_rows * a.cout <= 0xFFFFFFFFull;
}

/* Per-output direct form: depthwise and narrow (few output channels per group) convolutions. */
static __device__ __forceinline__ void sp_conv1d_direct(const SpConvArgs& a, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(e, a.batch * a.out_rows * a.cout) {
        const unsigned m = e / a.cout, o = e - m * a.cout, b = m / a.out_rows, t = m - b * a.out_rows;
        if (t >= a.conv_len(b)) { a.out[e] = 0.f; continue; }
        const unsigned length = a.length(b), c0 = o / a.ng * a.cg;
        float acc = a.bias ? a.bias[o] : 0.f;
        for (unsigned k = 0; k < a.kernel; k++) {
            const int u = sp_pad_row((int)(t * a.stride + k * a.dil) - (int)a.before, (int)length, a.mode);
            if (u < 0) continue;
            const float* xr = a.x + ((size_t)b * a.in_rows + u) * a.cin + c0;
            for (unsigned i = 0; i < a.cg; i++)
                acc = fmaf(a.pre_at(xr[i], c0 + i), a.w_at(((size_t)o * a.cg + i) * a.kernel + k), acc);
        }
        a.store(e, o, acc);
    }
}

static __device__ __forceinline__ void sp_convt1d_direct(const SpConvArgs& a, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(e, a.batch * a.out_rows * a.cout) {
        const unsigned m = e / a.cout, o = e - m * a.cout, b = m / a.out_rows, t = m - b * a.out_rows;
        if ((int)t >= a.convt_len(b)) { a.out[e] = 0.f; continue; }
        const unsigned length = a.length(b), c0 = o / a.ng * a.cg, on = o - o / a.ng * a.ng;
        const unsigned num = t + a.before;
        float acc = a.bias ? a.bias[o] : 0.f;
        for (unsigned k = num % a.stride; k < a.kernel && k <= num; k += a.stride) {
            const unsigned s = (num - k) / a.stride;
            if (s >= length) continue;
            const float* xr = a.x + ((size_t)b * a.in_rows + s) * a.cin + c0;
            for (unsigned i = 0; i < a.cg; i++)
                acc = fmaf(a.pre_at(xr[i], c0 + i), a.w_at(((size_t)(c0 + i) * a.ng + on) * a.kernel + k), acc);
        }
        a.store(e, o, acc);
    }
}

/* Depthwise Conv1d (groups == channels): 64 output rows x 64 channels per tile; the input window
 * is staged once through smem with padding and the input activation applied, then each thread
 * runs one channel down 16 rows. */
#define SPD_TR 64
#define SPD_TC 64
#define SPD_U 8
static __device__ __forceinline__ bool sp_conv1d_depthwise(const SpConvArgs& a, unsigned slice, unsigned nblk,
                                                            float* arena) {
    const unsigned window = (SPD_TR - 1) * a.stride + a.dil * (a.kernel - 1u) + 1u;
    if ((uint64_t)(window + a.kernel) * SPD_TC > SP_ARENA_FLOATS) return false;
    float* ws = arena + window * SPD_TC;
    const unsigned rt = (a.out_rows + SPD_TR - 1) / SPD_TR, ct = (a.cout + SPD_TC - 1) / SPD_TC;
    const unsigned c = threadIdx.x % SPD_TC, rg = threadIdx.x / SPD_TC;
    constexpr unsigned RPT = SPD_TR * SPD_TC / PLOW_NV_THREADS;
    const bool vec = a.cin % 4u == 0 && sp_aligned(a.x, 16);
    for (unsigned tile = slice; tile < a.batch * rt * ct; tile += nblk) {
        const unsigned b = tile / (rt * ct), rem = tile - b * rt * ct, t0 = rem / ct * SPD_TR;
        const unsigned c0 = (rem - rem / ct * ct) * SPD_TC;
        const unsigned length = a.length(b), out_len = a.conv_len(b);
        const int u0 = (int)(t0 * a.stride) - (int)a.before;
        __syncthreads();
        /* Loads first, then stores: the persistent grid has one block per SM, so the window's
         * memory-level parallelism has to come from each thread's registers. A thread keeps one
         * channel quad, so its activation parameters are loaded once per tile. */
        const unsigned cq = (threadIdx.x % (SPD_TC / 4)) * 4u, ch4 = c0 + cq;
        float p0[4];
#pragma unroll
        for (unsigned j = 0; j < 4; j++) p0[j] = a.alpha && ch4 + j < a.cin ? a.alpha[ch4 + j] : a.slope;
        for (unsigned i0 = threadIdx.x / (SPD_TC / 4); i0 < window; i0 += SPD_U * (PLOW_NV_THREADS * 4 / SPD_TC)) {
            float4 v[SPD_U];
#pragma unroll
            for (unsigned q = 0; q < SPD_U; q++) {
                const unsigned i = i0 + q * (PLOW_NV_THREADS * 4 / SPD_TC);
                v[q] = make_float4(0.f, 0.f, 0.f, 0.f);
                const int u = i < window ? sp_pad_row(u0 + (int)i, (int)length, a.mode) : -1;
                if (u < 0) continue;
                const float* xr = a.x + ((size_t)b * a.in_rows + u) * a.cin + ch4;
                if (vec && ch4 + 3u < a.cin) v[q] = __ldg((const float4*)xr);
                else {
                    if (ch4 < a.cin) v[q].x = xr[0];
                    if (ch4 + 1u < a.cin) v[q].y = xr[1];
                    if (ch4 + 2u < a.cin) v[q].z = xr[2];
                    if (ch4 + 3u < a.cin) v[q].w = xr[3];
                }
            }
            /* The activation is dispatched once per batch: a per-element switch costs more than
             * the whole tile at this occupancy. */
            const auto store = [&](auto f) {
#pragma unroll
                for (unsigned q = 0; q < SPD_U; q++) {
                    const unsigned i = i0 + q * (PLOW_NV_THREADS * 4 / SPD_TC);
                    if (i >= window) break;
                    *(float4*)(arena + i * SPD_TC + cq) =
                        make_float4(f(v[q].x, p0[0]), f(v[q].y, p0[1]), f(v[q].z, p0[2]), f(v[q].w, p0[3]));
                }
            };
            const unsigned pre = a.pre;
            switch (pre) {
            case 0: store([](float x, float) { return x; }); break;
            case 9: store([](float x, float p) { return x >= 0.0f ? x : __fmul_rn(x, p); }); break;
            case 12: store([](float x, float p) { return sp_snake(x, p); }); break;
            default: store([pre](float x, float p) { return sp_act(pre, x, p, 0.f); }); break;
            }
        }
        for (unsigned e = threadIdx.x; e < a.kernel * SPD_TC; e += PLOW_NV_THREADS) {
            const unsigned cc = e / a.kernel, k = e - cc * a.kernel;
            ws[k * SPD_TC + cc] = c0 + cc < a.cout ? a.w_at((size_t)(c0 + cc) * a.kernel + k) : 0.f;
        }
        __syncthreads();
        const unsigned ch = c0 + c;
        if (ch >= a.cout) continue;
        float acc[RPT];
        const float bias = a.bias ? a.bias[ch] : 0.f;
#pragma unroll
        for (unsigned r = 0; r < RPT; r++) acc[r] = bias;
        const float* xs = arena + (size_t)(rg * RPT * a.stride) * SPD_TC + c;
        for (unsigned k = 0; k < a.kernel; k++) {
            const float w = ws[k * SPD_TC + c];
            const float* xk = xs + (size_t)(k * a.dil) * SPD_TC;
#pragma unroll
            for (unsigned r = 0; r < RPT; r++) acc[r] = fmaf(xk[(size_t)(r * a.stride) * SPD_TC], w, acc[r]);
        }
#pragma unroll
        for (unsigned r = 0; r < RPT; r++) {
            const unsigned t = t0 + rg * RPT + r;
            if (t >= a.out_rows) break;
            const size_t oi = ((size_t)b * a.out_rows + t) * a.cout + ch;
            if (t >= out_len) a.out[oi] = 0.f;
            else a.store(oi, ch, acc[r]);
        }
    }
    return true;
}

/* Implicit-im2col A operand. Conv1d: GEMM row m = (b, t), column c = (tap, ci) tap-major so four
 * consecutive columns are four channels of one input row. ConvTranspose1d (one output phase):
 * row m = (b, q), t = q*stride + phase - crop_before, column c = (j, ci) reading input row q - j. */
struct SpConvPos1 { const float* xb; int u0, length; bool ok; };
struct SpRowConv1 {
    SpConvArgs a; unsigned c0, K, n_p, q_lo, phase; bool vec, transpose, pointwise;
    __device__ SpConvPos1 row(unsigned m, unsigned M) const {
        SpConvPos1 s{nullptr, 0, 0, false};
        if (m >= M) return s;
        if (transpose) {
            const unsigned b = m / n_p, q = q_lo + (m - b * n_p);
            s.ok = (int)(q * a.stride + phase - a.before) < a.convt_len(b);
            s.xb = a.x + (size_t)b * a.in_rows * a.cin + c0;
            s.u0 = (int)q;
            s.length = (int)a.length(b);
        } else {
            const unsigned b = m / a.out_rows, t = m - b * a.out_rows;
            s.ok = !a.lengths || t < a.conv_len(b);
            s.xb = a.x + (size_t)b * a.in_rows * a.cin + c0;
            s.u0 = (int)(t * a.stride) - (int)a.before;
            s.length = (int)a.length(b);
            if (pointwise) {
                const int u = sp_pad_row(s.u0, s.length, a.mode);
                s.ok = s.ok && u >= 0;
                s.xb += (size_t)(u < 0 ? 0 : u) * a.cin;
            }
        }
        return s;
    }
    __device__ int in_row(const SpConvPos1& s, unsigned tap) const {
        if (transpose) {
            const int u = s.u0 - (int)tap;
            return u >= 0 && u < s.length ? u : -1;
        }
        return sp_pad_row(s.u0 + (int)(tap * a.dil), s.length, a.mode);
    }
    __device__ float at(const SpConvPos1& s, unsigned c) const {
        if (c >= K) return 0.f;
        if (pointwise) return a.pre_at(s.xb[c], c0 + c);
        const unsigned tap = a.div_cg(c), ci = c - tap * a.cg;
        const int u = in_row(s, tap);
        return u < 0 ? 0.f : a.pre_at(s.xb[(size_t)u * a.cin + ci], c0 + ci);
    }
    __device__ float4 load4(const SpConvPos1& s, unsigned c) const {
        if (!s.ok || c >= K) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (vec) {
            unsigned ci = c;
            const float* xr = s.xb;
            if (!pointwise) {
                const unsigned tap = a.div_cg(c);
                ci = c - tap * a.cg;
                const int u = in_row(s, tap);
                if (u < 0) return make_float4(0.f, 0.f, 0.f, 0.f);
                xr += (size_t)u * a.cin;
            }
            float4 v = __ldg((const float4*)(xr + ci));
            if (a.pre) {
                v.x = a.pre_at(v.x, c0 + ci); v.y = a.pre_at(v.y, c0 + ci + 1);
                v.z = a.pre_at(v.z, c0 + ci + 2); v.w = a.pre_at(v.w, c0 + ci + 3);
            }
            return v;
        }
        return make_float4(at(s, c), at(s, c + 1), at(s, c + 2), at(s, c + 3));
    }
};
/* B operand: weight element (n, c) of this group / phase. */
struct SpRowConvW {
    SpConvArgs a; unsigned c0, n0, K, phase; bool vec, transpose;
    __device__ long long row(unsigned n, unsigned N) const { return n < N ? (long long)(n0 + n) : -1ll; }
    __device__ float at(long long n, unsigned c) const {
        if (c >= K) return 0.f;
        const unsigned tap = a.div_cg(c), ci = c - tap * a.cg;
        if (transpose)
            return a.w_at(((size_t)(c0 + ci) * a.ng + (size_t)(n - n0)) * a.kernel + phase + tap * a.stride);
        return a.w_at(((size_t)n * a.cg + ci) * a.kernel + tap);
    }
    __device__ float4 load4(long long n, unsigned c) const {
        if (n < 0 || c >= K) return make_float4(0.f, 0.f, 0.f, 0.f);
        if (vec) return __ldg((const float4*)((const float*)a.w + (size_t)n * K + c));
        return make_float4(at(n, c), at(n, c + 1), at(n, c + 2), at(n, c + 3));
    }
};
struct SpConvEpi1 {
    SpConvArgs a; unsigned n0, n_p, q_lo, phase; bool transpose;
    __device__ void operator()(unsigned m, unsigned n, float acc) const {
        const unsigned o = n0 + n;
        unsigned b, t;
        bool ok;
        if (transpose) {
            b = m / n_p;
            t = (q_lo + (m - b * n_p)) * a.stride + phase - a.before;
            ok = (int)t < a.convt_len(b);
        } else if (!a.lengths) {
            a.store((size_t)m * a.cout + o, o, a.bias ? __fadd_rn(acc, a.bias[o]) : acc);
            return;
        } else {
            b = m / a.out_rows;
            t = m - b * a.out_rows;
            ok = t < a.conv_len(b);
        }
        const size_t oi = ((size_t)b * a.out_rows + t) * a.cout + o;
        if (!ok) { a.out[oi] = 0.f; return; }
        a.store(oi, o, a.bias ? __fadd_rn(acc, a.bias[o]) : acc);
    }
};

static __device__ void d_conv1d_f32(const PlowDevInst* in, void* const* T, bool transpose, unsigned slice,
                                    unsigned nblk, float* arena) {
    SpConvArgs a;
    if (!sp_conv_args(in, T, transpose, a)) return;
    if (!transpose && a.cg == 1u && a.ng == 1u && sp_conv1d_depthwise(a, slice, nblk, arena)) return;
    if (a.ng < 16u || a.cg * a.kernel < 16u) {
        if (transpose) sp_convt1d_direct(a, slice, nblk);
        else sp_conv1d_direct(a, slice, nblk);
        return;
    }
    const bool avec = a.cg % 4u == 0 && a.cin % 4u == 0 && sp_aligned(a.x, 16);
    const unsigned phases = transpose ? a.stride : 1u;
    const unsigned tn = (a.ng + SPG_BN - 1) / SPG_BN;
    unsigned total = 0;
    for (unsigned p = 0; p < phases; p++) {
        unsigned rows = a.out_rows;
        if (transpose) {
            const unsigned q_lo = a.before > p ? (a.before - p + a.stride - 1) / a.stride : 0u;
            const long long qe = ((long long)a.out_rows + a.before - p + a.stride - 1) / a.stride;
            rows = qe > (long long)q_lo ? (unsigned)(qe - q_lo) : 0u;
        }
        total += a.groups * ((a.batch * rows + SPG_BM - 1) / SPG_BM) * tn;
    }
    for (unsigned tile = slice; tile < total; tile += nblk) {
        unsigned rem = tile, p = 0, n_p = a.out_rows, q_lo = 0, taps = a.kernel, mt = 0;
        for (;; p++) {
            if (transpose) {
                q_lo = a.before > p ? (a.before - p + a.stride - 1) / a.stride : 0u;
                const long long qe = ((long long)a.out_rows + a.before - p + a.stride - 1) / a.stride;
                n_p = qe > (long long)q_lo ? (unsigned)(qe - q_lo) : 0u;
                taps = a.kernel > p ? (a.kernel - p + a.stride - 1) / a.stride : 0u;
            }
            mt = (a.batch * n_p + SPG_BM - 1) / SPG_BM;
            if (rem < a.groups * mt * tn) break;
            rem -= a.groups * mt * tn;
        }
        const unsigned g = rem / (mt * tn), r2 = rem - g * mt * tn;
        const unsigned c0 = g * a.cg, n0 = g * a.ng, K = taps * a.cg;
        const bool bvec = !transpose && a.kernel == 1u && !a.wf16 && a.cg % 4u == 0 && sp_aligned(a.w, 16);
        const SpRowConv1 la{a, c0, K, n_p, q_lo, p, avec, transpose, !transpose && a.kernel == 1u};
        const SpRowConvW lb{a, c0, n0, K, p, bvec, transpose};
        const SpConvEpi1 ep{a, n0, n_p, q_lo, p, transpose};
        sp_gemm_tile((r2 / tn) * SPG_BM, (r2 % tn) * SPG_BN, a.batch * n_p, a.ng, K, la, lb, ep, arena);
    }
}

/* ---- AttentionF32 (203): fp32 flash attention. One block per (item, head, 64-query tile); BK-key
 * tiles staged through smem; a thread owns 4 query
 * rows x 4*BK/64 keys of S and 4 rows x head_width/16 columns of O, with the row statistics
 * reduced over its 16-lane group. */
#define SPF_BQ 64
#define SPF_LDP (SPF_BQ + 4)
template <int HW, int BK>
struct SpfShape {
    static constexpr int LDQ = SPF_BQ + 4, LDV = HW + 4;
    /* P ([BK][LDP]) aliases the K stage once S is consumed. */
    static constexpr int LDK = BK + 4 > (BK * SPF_LDP + HW - 1) / HW ? BK + 4 : (BK * SPF_LDP + HW - 1) / HW;
    static constexpr int FLOATS = HW * LDQ + HW * LDK + BK * LDV;
};
static_assert(SP_ARENA_FLOATS >= SpfShape<128, 64>::FLOATS && SP_ARENA_FLOATS >= SpfShape<64, 128>::FLOATS,
              "attention stages");

template <int HW, int BK>
static __device__ void sp_attention_f32(const PlowDevInst* in, void* const* T, unsigned slice,
                                        unsigned nblk, float* arena) {
    using S = SpfShape<HW, BK>;
    constexpr int LDQ = S::LDQ, LDK = S::LDK, LDV = S::LDV, LDP = SPF_LDP, CB = HW / 64, KG = BK / 64;
    float* Qs = arena;
    float* Ks = Qs + HW * LDQ;
    float* Vs = Ks + HW * LDK;
    float* Ps = Ks;
    float* out = (float*)SP_TEN(0);
    const float* query = (const float*)SP_TEN(1);
    const float* key = (const float*)SP_TEN(2);
    const float* value = (const float*)SP_TEN(3);
    const unsigned* lengths = (const unsigned*)SP_TEN(4);
    const float* bias = (const float*)SP_TEN(5);
    const unsigned batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const unsigned width = heads * HW, stride = in->i[5] ? in->i[5] : width;
    const bool causal = in->i[6] & 1u;
    const unsigned bias_hs = in->i[7], k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    const float scale = in->fj[0].f;
    const bool vec = stride % 4u == 0 && k_col0 % 4u == 0 && v_col0 % 4u == 0 && sp_aligned(query, 16) &&
                     sp_aligned(key, 16) && sp_aligned(value, 16);
    const unsigned tid = threadIdx.x, tx = tid & 15u, ty = tid >> 4;
    const unsigned qtiles = (q_rows + SPF_BQ - 1) / SPF_BQ;
    constexpr unsigned D4 = HW / 4, QPER = SPF_BQ * D4 / PLOW_NV_THREADS, PER = BK * D4 / PLOW_NV_THREADS;
    static_assert(QPER * PLOW_NV_THREADS == SPF_BQ * D4 && PER * PLOW_NV_THREADS == BK * D4, "stage coverage");
    for (unsigned item = slice; item < batch * heads * qtiles; item += nblk) {
        const unsigned qt = item % qtiles, bh = item / qtiles, h = bh % heads, b = bh / heads;
        const unsigned q0 = qt * SPF_BQ;
        const unsigned klen = lengths && lengths[b] < kv_rows ? lengths[b] : kv_rows;
        const unsigned kend = causal ? min(klen, q0 + SPF_BQ) : klen;
        /* Every stage issues all of its loads into registers before the first smem store: with
         * one block per SM, a load-store-load loop exposes the full memory latency per element. */
        {
            float4 qv[QPER];
#pragma unroll
            for (unsigned p = 0; p < QPER; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, r = e / D4, d = (e - r * D4) * 4u;
                qv[p] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (q0 + r < q_rows) qv[p] = sp_ld4(query + ((size_t)b * q_rows + q0 + r) * stride + h * HW + d, vec);
            }
            __syncthreads();
#pragma unroll
            for (unsigned p = 0; p < QPER; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, r = e / D4, d = (e - r * D4) * 4u;
                Qs[d * LDQ + r] = qv[p].x; Qs[(d + 1) * LDQ + r] = qv[p].y;
                Qs[(d + 2) * LDQ + r] = qv[p].z; Qs[(d + 3) * LDQ + r] = qv[p].w;
            }
        }
        float o[4][4 * CB], mrow[4], lrow[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            mrow[i] = -INFINITY; lrow[i] = 0.f;
#pragma unroll
            for (int c = 0; c < 4 * CB; c++) o[i][c] = 0.f;
        }
        for (unsigned k0 = 0; k0 < kend; k0 += BK) {
            /* K/V are not prefetched across tiles: holding them in registers through the tile
             * spills in the interpreter object. */
            float4 kr[PER], vr[PER];
#pragma unroll
            for (unsigned p = 0; p < PER; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, j = e / D4, d = (e - j * D4) * 4u;
                kr[p] = vr[p] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (k0 + j < kend) {
                    const size_t base = ((size_t)b * kv_rows + k0 + j) * stride + h * HW + d;
                    kr[p] = sp_ld4(key + base + k_col0, vec);
                    vr[p] = sp_ld4(value + base + v_col0, vec);
                }
            }
            __syncthreads();
#pragma unroll
            for (unsigned p = 0; p < PER; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, j = e / D4, d = (e - j * D4) * 4u;
                Ks[d * LDK + j] = kr[p].x; Ks[(d + 1) * LDK + j] = kr[p].y;
                Ks[(d + 2) * LDK + j] = kr[p].z; Ks[(d + 3) * LDK + j] = kr[p].w;
                *(float4*)(Vs + j * LDV + d) = vr[p];
            }
            __syncthreads();
            float s[4][4 * KG];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4 * KG; j++) s[i][j] = 0.f;
#pragma unroll 8
            for (int d = 0; d < HW; d++) {
                const float4 qa = *(const float4*)(Qs + d * LDQ + ty * 4);
                const float av[4] = {qa.x, qa.y, qa.z, qa.w};
#pragma unroll
                for (int g = 0; g < KG; g++) {
                    const float4 kb = *(const float4*)(Ks + d * LDK + g * 64 + tx * 4);
                    const float bv[4] = {kb.x, kb.y, kb.z, kb.w};
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 4; j++) s[i][g * 4 + j] = fmaf(av[i], bv[j], s[i][g * 4 + j]);
                }
            }
            const bool full = k0 + BK <= kend && !causal && !bias;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned r = q0 + ty * 4 + i;
                float mt = -INFINITY;
#pragma unroll
                for (int j = 0; j < 4 * KG; j++) {
                    const unsigned kj = k0 + (j >> 2) * 64 + tx * 4 + (j & 3);
                    float v = s[i][j] * scale;
                    if (!full) {
                        if (kj < kend && (!causal || kj <= r)) {
                            if (bias && r < q_rows) v += bias[(size_t)h * bias_hs + (size_t)r * kv_rows + kj];
                        } else {
                            v = -INFINITY;
                        }
                    }
                    s[i][j] = v;
                    mt = fmaxf(mt, v);
                }
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
                const float mnew = fmaxf(mrow[i], mt);
                const float corr = mnew == -INFINITY ? 1.f : expf(mrow[i] - mnew);
                float ps = 0.f;
#pragma unroll
                for (int j = 0; j < 4 * KG; j++) {
                    const float p = s[i][j] == -INFINITY ? 0.f : expf(s[i][j] - mnew);
                    s[i][j] = p;
                    ps += p;
                }
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
                lrow[i] = lrow[i] * corr + ps;
                mrow[i] = mnew;
#pragma unroll
                for (int c = 0; c < 4 * CB; c++) o[i][c] *= corr;
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 4 * KG; j++)
                *(float4*)(Ps + ((j >> 2) * 64 + tx * 4 + (j & 3)) * LDP + ty * 4) =
                    make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
            __syncthreads();
#pragma unroll 8
            for (int j = 0; j < BK; j++) {
                const float4 pa = *(const float4*)(Ps + j * LDP + ty * 4);
                const float av[4] = {pa.x, pa.y, pa.z, pa.w};
#pragma unroll
                for (int cb = 0; cb < CB; cb++) {
                    const float4 vb = *(const float4*)(Vs + j * LDV + cb * 64 + tx * 4);
                    const float bv[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int c = 0; c < 4; c++) o[i][cb * 4 + c] = fmaf(av[i], bv[c], o[i][cb * 4 + c]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned r = q0 + ty * 4 + i;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int cb = 0; cb < CB; cb++)
                *(float4*)(orow + cb * 64 + tx * 4) =
                    make_float4(o[i][cb * 4] * inv, o[i][cb * 4 + 1] * inv, o[i][cb * 4 + 2] * inv,
                                o[i][cb * 4 + 3] * inv);
        }
    }
}

static __device__ void d_attention_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk,
                                       float* arena) {
    if (in->i[4] == 64u) sp_attention_f32<64, 128>(in, T, slice, nblk, arena);
    else if (in->i[4] == 128u) sp_attention_f32<128, 64>(in, T, slice, nblk, arena);
}

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
    case PLOW_DOP_GATHER_ROWS_F32: d_gather_rows_f32(in, T, slice, nblk); break;
    case PLOW_DOP_COPY_COLS_F32: d_copy_cols_f32(in, T, slice, nblk); break;
    case PLOW_DOP_CONV1D_F32: d_conv1d_f32(in, T, false, slice, nblk, arena); break;
    case PLOW_DOP_CONV_TRANSPOSE1D_F32: d_conv1d_f32(in, T, true, slice, nblk, arena); break;
    case PLOW_DOP_UNARY_F32: d_unary_f32(in, T, slice, nblk); break;
    case PLOW_DOP_BINARY_F32: d_binary_f32(in, T, slice, nblk); break;
    case PLOW_DOP_CUMSUM_F64: d_cumsum_f64(in, T, slice, nblk, arena); break;
    case PLOW_DOP_RAND_F32: d_rand_f32(in, T, slice, nblk); break;
    case PLOW_DOP_ATTENTION_F32: d_attention_f32(in, T, slice, nblk, arena); break;
    default: __trap(); break;
    }
}
#undef SP_TEN
