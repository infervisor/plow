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

/* The arena is the dynamic shared memory base (interp_sm120.cu's `arena`, the test kernel's). The
 * op functions are __noinline__ and receive it as a generic pointer; rebinding it to this symbol
 * lets the compiler emit LDS/STS instead of generic LD/ST. */
extern __shared__ __align__(16) float sp_smem[];
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
/* DenseGemmF32's LayerNorm prologue: A' = (A - mean) * inv * gamma + beta with LayerNormF32's
 * exact operation order, per-row (mean, inv) from RowStatsF32; optionally rounded to bf16. */
struct SpLn {
    const float* stats; const float* gamma; const float* beta; unsigned row0; bool round;
    __device__ float apply(float v, float mean, float inv, unsigned c) const {
        v = __fmul_rn(__fsub_rn(v, mean), inv);
        v = __fadd_rn(__fmul_rn(v, gamma ? gamma[c] : 1.0f), beta ? beta[c] : 0.0f);
        return round ? sp_bf16(v) : v;
    }
};
struct SpLnRow { const float* r; float mean, inv; };
struct SpRowLnF32 {
    const float* p; unsigned k; bool vec; SpLn ln;
    __device__ SpLnRow row(unsigned r, unsigned rows) const {
        if (r >= rows) return SpLnRow{nullptr, 0.f, 0.f};
        const float2 st = *(const float2*)(ln.stats + 2 * ((size_t)ln.row0 + r));
        return SpLnRow{p + (size_t)r * k, st.x, st.y};
    }
    __device__ float4 load4(const SpLnRow& s, unsigned c) const {
        const float4 v = SpRowF32{p, k, vec}.load4(s.r, c);
        if (!s.r) return v;
        return make_float4(c < k ? ln.apply(v.x, s.mean, s.inv, c) : 0.f,
                           c + 1 < k ? ln.apply(v.y, s.mean, s.inv, c + 1) : 0.f,
                           c + 2 < k ? ln.apply(v.z, s.mean, s.inv, c + 2) : 0.f,
                           c + 3 < k ? ln.apply(v.w, s.mean, s.inv, c + 3) : 0.f);
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

/* SpRowConv with the column decode (ic, ky, kx) precomputed per column in smem: every layout's
 * input index is base(b, iy0, ix0) + off(ic, ky, kx), so a column costs one table read. */
struct SpConvTab { int off; unsigned kyx; };
struct SpConvPosT { long long base; int iy0, ix0; bool ok; };
struct SpRowConvT {
    const float* x; const SpConvTab* tab; SpConvGeom g; unsigned k;
    __device__ SpConvPosT row(unsigned r, unsigned rows) const {
        const SpConvPos p = SpRowConv{x, g, k}.row(r, rows);
        SpConvPosT s{0, p.iy0, p.ix0, p.ok};
        const long long F = g.frames, W = g.width, C = g.ic;
        if (g.layout == 0u) s.base = ((p.b * F + p.iy0) * W + p.ix0) * C;
        else if (g.layout == 1u) s.base = (p.b * F + p.iy0) * C * W + p.ix0;
        else s.base = ((long long)p.b * C * F + p.iy0) * W + p.ix0;
        return s;
    }
    __device__ float at(const SpConvPosT& s, unsigned c) const {
        if (!s.ok || c >= k) return 0.f;
        const SpConvTab e = tab[c];
        const int iy = s.iy0 + (int)(e.kyx >> 16), ix = s.ix0 + (int)(e.kyx & 0xFFFFu);
        if (iy < 0 || iy >= (int)g.frames || ix < 0 || ix >= (int)g.width) return 0.f;
        return x[s.base + e.off];
    }
    __device__ float4 load4(const SpConvPosT& s, unsigned c) const {
        return make_float4(at(s, c), at(s, c + 1), at(s, c + 2), at(s, c + 3));
    }
};
__device__ __forceinline__ void sp_conv_table(SpConvTab* tab, const SpConvGeom& g, unsigned k) {
    const unsigned kk = g.kernel * g.kernel, F = g.frames, W = g.width, C = g.ic;
    for (unsigned c = threadIdx.x; c < k; c += PLOW_NV_THREADS) {
        const unsigned ic = c / kk, rem = c - ic * kk, ky = rem / g.kernel, kx = rem - ky * g.kernel;
        int off;
        if (g.layout == 0u) off = (int)((ky * W + kx) * C + ic);
        else if (g.layout == 1u) off = (int)((ky * C + ic) * W + kx);
        else off = (int)((ic * F + ky) * W + kx);
        tab[c] = SpConvTab{off, ky << 16 | kx};
    }
    __syncthreads();
}

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

/* ---- tensor-core tiles: 64x64 outputs, 8 warps as 2 (M) x 4 (N) of 32x16, fp32 accumulate ----
 * BF16: both operands must be bf16-representable (the caller's flag says so), so every product is
 * exact and only the summation order differs from the golden. m16n8k16, BK=64.
 * TF32X3: x = hi + lo (tf32, round-to-nearest); lo*hi + hi*lo + hi*hi per k8, ~fp32 accurate
 * (the SNAC split). BK=32.
 * Operands go through the same loaders as sp_gemm_tile (register prefetch, double-buffered smem).
 * Split-K (with a scratch tensor): a tile's S slices each write their fragment partial; the last
 * slice to arrive (ticket) sums them in slice order, so the result is deterministic. */
#define SPT_BM 64
#define SPT_BN 64
#define SPT_TICKETS 1024
template <bool TF32>
struct SpTc {
    static constexpr unsigned BK = TF32 ? 32 : 64, LD = TF32 ? BK + 4 : BK + 8, KT = BK / 4;
    static constexpr unsigned RPP = PLOW_NV_THREADS / KT, P = SPT_BM / RPP;
    static constexpr unsigned STAGE_BYTES = SPT_BM * LD * (TF32 ? 4 : 2);
    static_assert(P * RPP == SPT_BM && SPT_BM == SPT_BN, "tc loader coverage");
};
static_assert(4 * SpTc<false>::STAGE_BYTES <= SP_ARENA_FLOATS * 4 && 4 * SpTc<true>::STAGE_BYTES <= SP_ARENA_FLOATS * 4,
              "tc stages");

/* tf32 by truncation: one LOP3. The hi/lo split below keeps x = hi + lo exact before lo's own
 * truncation (3xTF32 keeps ~21 significant bits either way; cvt.rna costs ~8 ALU ops here). */
__device__ __forceinline__ unsigned sp_tf32(float x) { return __float_as_uint(x) & 0xFFFFE000u; }
__device__ __forceinline__ void sp_mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void sp_mma_tf32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void sp_ldsm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}

template <bool TF32>
__device__ __forceinline__ void sp_tc_put(void* base, unsigned idx, float4 v) {
    if (TF32) {
        *(float4*)((float*)base + idx) = v;
    } else {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
        uint2 u;
        u.x = *(const unsigned*)&lo;
        u.y = *(const unsigned*)&hi;
        *(uint2*)((__nv_bfloat16*)base + idx) = u;
    }
}

/* Loaders whose consecutive k are far apart in memory (im2col gathers, ConvTranspose weights)
 * coalesce across rows instead: a warp then loads 32 consecutive rows at the same k. */
template <class L> __device__ __forceinline__ bool sp_rowfast(const L&) { return false; }
__device__ __forceinline__ bool sp_rowfast(const SpRowConvT&) { return true; }
struct SpRowConvW;
__device__ __forceinline__ bool sp_rowfast(const SpRowConvW& l);

/* acc[mi][nj][r] for this thread's fragments of the 64x64 tile at (m0, n0) over k in [kb, ke). */
template <bool TF32, class LA, class LB>
static __device__ __forceinline__ void sp_tc_tile(unsigned m0, unsigned n0, unsigned M, unsigned N,
                                                  unsigned kb, unsigned ke, const LA& la, const LB& lb,
                                                  float* arena, float (&acc)[2][2][4]) {
    using C = SpTc<TF32>;
    char* smem = (char*)arena;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = (warp & 1u) * 32u, wn = (warp >> 1) * 16u;
    const unsigned lr = tid / C::KT, lk = (tid % C::KT) * 4u;
    /* Row-fast operand: thread owns row tid % 64 and k offsets (tid / 64) * 4 + 16 p. */
    const bool rfa = sp_rowfast(la), rfb = sp_rowfast(lb);
    auto o_row = [&](bool rf, unsigned p) { return rf ? tid % SPT_BM : lr + p * C::RPP; };
    auto o_col = [&](bool rf, unsigned p) { return rf ? (tid / SPT_BM) * 4u + p * 16u : lk; };
    const unsigned nk = (ke - kb + C::BK - 1) / C::BK;
    decltype(la.row(0u, 0u)) ra[C::P];
    decltype(lb.row(0u, 0u)) rb[C::P];
    float4 va[C::P], vb[C::P];
    auto fetch = [&](unsigned kt, float4 (&ua)[C::P], float4 (&ub)[C::P]) {
        const unsigned c = kb + kt * C::BK;
#pragma unroll
        for (unsigned p = 0; p < C::P; p++) {
            ua[p] = la.load4(ra[p], c + o_col(rfa, p));
            ub[p] = lb.load4(rb[p], c + o_col(rfb, p));
        }
    };
#pragma unroll
    for (unsigned p = 0; p < C::P; p++) {
        ra[p] = la.row(m0 + o_row(rfa, p), M);
        rb[p] = lb.row(n0 + o_row(rfb, p), N);
    }
    fetch(0u, va, vb);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;
    const unsigned g = lane >> 2, t = lane & 3u;
    auto step = [&](unsigned kt, float4 (&ua)[C::P], float4 (&ub)[C::P]) {
        char* as = smem + (kt & 1u) * 2u * C::STAGE_BYTES;
        char* bs = as + C::STAGE_BYTES;
#pragma unroll
        for (unsigned p = 0; p < C::P; p++) {
            sp_tc_put<TF32>(as, o_row(rfa, p) * C::LD + o_col(rfa, p), ua[p]);
            sp_tc_put<TF32>(bs, o_row(rfb, p) * C::LD + o_col(rfb, p), ub[p]);
        }
        __syncthreads();
        if (kt + 1 < nk) fetch(kt + 1, ua, ub);
        /* Tensor-core accumulation truncates; each k-tile gets a fresh accumulator and is folded
         * into acc with a rounded add, so the bias does not grow with K. */
        float part[2][2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) part[i][j][r] = 0.f;
        if (TF32) {
            const float* af = (const float*)as;
            const float* bf = (const float*)bs;
#pragma unroll
            for (unsigned ks = 0; ks < C::BK; ks += 8) {
                unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const float* a = af + (wm + i * 16 + g) * C::LD + ks + t;
                    const float x[4] = {a[0], a[8 * C::LD], a[4], a[8 * C::LD + 4]};
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        ah[i][r] = sp_tf32(x[r]);
                        al[i][r] = sp_tf32(x[r] - __uint_as_float(ah[i][r]));
                    }
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const float* b = bf + (wn + j * 8 + g) * C::LD + ks + t;
                    const float x[2] = {b[0], b[4]};
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        bh[j][r] = sp_tf32(x[r]);
                        bl[j][r] = sp_tf32(x[r] - __uint_as_float(bh[j][r]));
                    }
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        sp_mma_tf32(part[i][j], al[i], bh[j][0], bh[j][1]);
                        sp_mma_tf32(part[i][j], ah[i], bl[j][0], bl[j][1]);
                        sp_mma_tf32(part[i][j], ah[i], bh[j][0], bh[j][1]);
                    }
            }
        } else {
            const __nv_bfloat16* ab = (const __nv_bfloat16*)as;
            const __nv_bfloat16* bb = (const __nv_bfloat16*)bs;
#pragma unroll
            for (unsigned ks = 0; ks < C::BK; ks += 16) {
                unsigned a[2][4], b[4];
#pragma unroll
                for (int i = 0; i < 2; i++)
                    sp_ldsm4(a[i], ab + (wm + i * 16 + (lane & 15u)) * C::LD + ks + (lane >> 4) * 8u);
                sp_ldsm4(b, bb + (wn + (lane & 7u) + ((lane >> 4) << 3)) * C::LD + ks + ((lane >> 3) & 1u) * 8u);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    sp_mma_bf16(part[i][0], a[i], b[0], b[1]);
                    sp_mma_bf16(part[i][1], a[i], b[2], b[3]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) acc[i][j][r] = __fadd_rn(acc[i][j][r], part[i][j][r]);
    };
    for (unsigned kt = 0; kt < nk; kt++) step(kt, va, vb);
    /* The next tile's first store targets stage 0, which the last odd k-tile may still read. */
    __syncthreads();
}

/* DenseGemm's common case (plain f32 A rows, bf16 W rows, 16B-aligned, K % 8 == 0): a 3-stage
 * cp.async ring instead of register prefetch, so a short split-K chunk has all of its loads in
 * flight at once. A stays f32 in smem and is packed to bf16 pairs at fragment load. */
#define SPA_NS 3
#define SPA_LDA 72
#define SPA_LDB 72
#define SPA_STAGE_BYTES (SPT_BM * SPA_LDA * 4 + SPT_BN * SPA_LDB * 2)
static_assert(SPA_NS * SPA_STAGE_BYTES <= SP_ARENA_FLOATS * 4, "dense cp.async stages");
__device__ __forceinline__ void sp_cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),
                 "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ unsigned sp_pack_bf16(float2 v) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(v.x, v.y);
    return *(const unsigned*)&h;
}
static __device__ __forceinline__ void sp_tc_tile_async(unsigned m0, unsigned n0, unsigned M, unsigned N,
                                                        unsigned kb, unsigned ke, const SpRowF32& la,
                                                        const SpRowBf16& lb, float* arena,
                                                        float (&acc)[2][2][4], const SpLn* ln) {
    char* smem = (char*)arena;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = (warp & 1u) * 32u, wn = (warp >> 1) * 16u, g = lane >> 2, t = lane & 3u;
    const unsigned nk = (ke - kb + 63u) / 64u;
    auto issue = [&](unsigned kt) {
        if (kt < nk) {
            float* as = (float*)(smem + (kt % SPA_NS) * SPA_STAGE_BYTES);
            __nv_bfloat16* bs = (__nv_bfloat16*)(as + SPT_BM * SPA_LDA);
            const unsigned k0 = kb + kt * 64u;
#pragma unroll
            for (unsigned p = 0; p < 4; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, r = e >> 4, c = (e & 15u) * 4u;
                const bool ok = m0 + r < M && k0 + c < ke;
                sp_cp16(as + r * SPA_LDA + c, ok ? la.p + (size_t)(m0 + r) * la.k + k0 + c : la.p, ok);
            }
#pragma unroll
            for (unsigned p = 0; p < 2; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, r = e >> 3, c = (e & 7u) * 8u;
                const bool ok = n0 + r < N && k0 + c < ke;
                sp_cp16(bs + r * SPA_LDB + c, ok ? lb.p + (size_t)(n0 + r) * lb.stride + k0 + c : lb.p, ok);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;
#pragma unroll
    for (unsigned s = 0; s + 1 < SPA_NS; s++) issue(s);
    for (unsigned kt = 0; kt < nk; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(SPA_NS - 2));
        if (ln) {
            /* LayerNorm prologue on the A chunks this thread copied (its own copies are visible). */
            float* af = (float*)(smem + (kt % SPA_NS) * SPA_STAGE_BYTES);
            const unsigned k0 = kb + kt * 64u;
#pragma unroll
            for (unsigned p = 0; p < 4; p++) {
                const unsigned e = tid + p * PLOW_NV_THREADS, r = e >> 4, c = (e & 15u) * 4u;
                if (m0 + r >= M || k0 + c >= ke) continue;
                const float2 st = *(const float2*)(ln->stats + 2 * ((size_t)ln->row0 + m0 + r));
                float4* d = (float4*)(af + r * SPA_LDA + c);
                float4 v = *d;
                v.x = ln->apply(v.x, st.x, st.y, k0 + c); v.y = ln->apply(v.y, st.x, st.y, k0 + c + 1);
                v.z = ln->apply(v.z, st.x, st.y, k0 + c + 2); v.w = ln->apply(v.w, st.x, st.y, k0 + c + 3);
                *d = v;
            }
        }
        __syncthreads();
        issue(kt + SPA_NS - 1);
        const float* as = (const float*)(smem + (kt % SPA_NS) * SPA_STAGE_BYTES);
        const __nv_bfloat16* bs = (const __nv_bfloat16*)(as + SPT_BM * SPA_LDA);
        float part[2][2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) part[i][j][r] = 0.f;
#pragma unroll
        for (unsigned ks = 0; ks < 64u; ks += 16) {
            unsigned a[2][4], b[4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const float* ap = as + (wm + i * 16 + g) * SPA_LDA + ks + 2 * t;
                a[i][0] = sp_pack_bf16(*(const float2*)ap);
                a[i][1] = sp_pack_bf16(*(const float2*)(ap + 8 * SPA_LDA));
                a[i][2] = sp_pack_bf16(*(const float2*)(ap + 8));
                a[i][3] = sp_pack_bf16(*(const float2*)(ap + 8 * SPA_LDA + 8));
            }
            sp_ldsm4(b, bs + (wn + (lane & 7u) + ((lane >> 4) << 3)) * SPA_LDB + ks + ((lane >> 3) & 1u) * 8u);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                sp_mma_bf16(part[i][0], a[i], b[0], b[1]);
                sp_mma_bf16(part[i][1], a[i], b[2], b[3]);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) acc[i][j][r] = __fadd_rn(acc[i][j][r], part[i][j][r]);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}

/* ---- cp.async f32 tile: both operands land as f32 in an SPX_NS-stage smem ring, 16-byte copies
 * for contiguous k runs (sp_seg) or 4-byte gathers (sp_elem, a warp covering 32 rows at one k);
 * fragments are split to tf32 or packed to bf16 pairs at load. The A pre-activation (conv input
 * activations) is applied in place by the thread that issued each copy. BK = 32. */
#define SPX_NS 3
template <bool TF32>
struct SpX {
    static constexpr unsigned LD = TF32 ? 36 : 40, OPB = SPT_BM * LD * 4, STAGE = 2 * OPB;
};
static_assert(SPX_NS * SpX<false>::STAGE <= SP_ARENA_FLOATS * 4 && SPX_NS * SpX<true>::STAGE <= SP_ARENA_FLOATS * 4,
              "async f32 stages");
__device__ __forceinline__ void sp_cp4(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),
                 "l"(src), "r"(ok ? 4 : 0));
}
/* Loader hooks (overloaded per loader below its definition): sp_async(l) the loader can feed the
 * ring; sp_segs(l) 16-byte runs are valid; sp_seg / sp_elem the source of k..k+3 / k (nullptr =
 * zero); sp_pre(l) the loader has an input activation, applied by sp_pre_at. */
template <class L> __device__ __forceinline__ bool sp_async(const L&) { return false; }
template <class L> __device__ __forceinline__ bool sp_pre(const L&) { return false; }
template <class L, class R> __device__ __forceinline__ float sp_pre_at(const L&, const R&, float v, unsigned) {
    return v;
}
template <class L, class R> __device__ __forceinline__ const float* sp_seg(const L&, const R&, unsigned) {
    return nullptr;
}
template <class L, class R> __device__ __forceinline__ const float* sp_elem(const L&, const R&, unsigned) {
    return nullptr;
}
template <class L> __device__ __forceinline__ bool sp_segs(const L&) { return false; }
__device__ __forceinline__ bool sp_async(const SpRowF32&) { return true; }
__device__ __forceinline__ bool sp_segs(const SpRowF32& l) { return l.vec && (l.k & 3u) == 0; }
__device__ __forceinline__ const float* sp_seg(const SpRowF32& l, const float* r, unsigned c) {
    return r && c < l.k ? r + c : nullptr;
}
__device__ __forceinline__ const float* sp_elem(const SpRowF32& l, const float* r, unsigned c) {
    return r && c < l.k ? r + c : nullptr;
}
__device__ __forceinline__ bool sp_async(const SpRowF32S&) { return true; }
__device__ __forceinline__ bool sp_segs(const SpRowF32S& l) { return l.vec && (l.k & 3u) == 0; }
__device__ __forceinline__ const float* sp_seg(const SpRowF32S& l, const float* r, unsigned c) {
    return r && c < l.k ? r + c : nullptr;
}
__device__ __forceinline__ const float* sp_elem(const SpRowF32S& l, const float* r, unsigned c) {
    return r && c < l.k ? r + c : nullptr;
}
__device__ __forceinline__ bool sp_async(const SpRowLnF32&) { return true; }
__device__ __forceinline__ bool sp_segs(const SpRowLnF32& l) { return l.vec && (l.k & 3u) == 0; }
__device__ __forceinline__ const float* sp_seg(const SpRowLnF32& l, const SpLnRow& s, unsigned c) {
    return s.r && c < l.k ? s.r + c : nullptr;
}
__device__ __forceinline__ const float* sp_elem(const SpRowLnF32& l, const SpLnRow& s, unsigned c) {
    return s.r && c < l.k ? s.r + c : nullptr;
}
__device__ __forceinline__ bool sp_pre(const SpRowLnF32&) { return true; }
__device__ __forceinline__ float sp_pre_at(const SpRowLnF32& l, const SpLnRow& s, float v, unsigned c) {
    return l.ln.apply(v, s.mean, s.inv, c);
}
__device__ __forceinline__ bool sp_async(const SpRowConvT&) { return false; } /* 4-byte cp.async gathers measured slower than the register path */
__device__ __forceinline__ const float* sp_elem(const SpRowConvT& l, const SpConvPosT& s, unsigned c) {
    if (!s.ok || c >= l.k) return nullptr;
    const SpConvTab e = l.tab[c];
    const int iy = s.iy0 + (int)(e.kyx >> 16), ix = s.ix0 + (int)(e.kyx & 0xFFFFu);
    if (iy < 0 || iy >= (int)l.g.frames || ix < 0 || ix >= (int)l.g.width) return nullptr;
    return l.x + s.base + e.off;
}

template <bool TF32, class LA, class LB>
static __device__ __forceinline__ void sp_tcx_tile(unsigned m0, unsigned n0, unsigned M, unsigned N, unsigned kb,
                                                   unsigned ke, const LA& la, const LB& lb, float* arena,
                                                   float (&acc)[2][2][4]) {
    using X = SpX<TF32>;
    constexpr unsigned LD = X::LD;
    char* smem = (char*)arena;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = (warp & 1u) * 32u, wn = (warp >> 1) * 16u, g = lane >> 2, t = lane & 3u;
    const unsigned nk = (ke - kb + 31u) / 32u;
    const bool sa = sp_segs(la), sb = sp_segs(lb), pa = sp_pre(la);
    /* seg mapping: rows (tid >> 3) + 32 p, k (tid & 7) * 4; elem mapping: row tid & 63, k (tid >> 6) + 4 i. */
    decltype(la.row(0u, 0u)) ra[2];
    decltype(lb.row(0u, 0u)) rb[2];
#pragma unroll
    for (unsigned p = 0; p < 2; p++) {
        ra[p] = la.row(m0 + (sa ? (tid >> 3) + 32u * p : tid & 63u), M);
        rb[p] = lb.row(n0 + (sb ? (tid >> 3) + 32u * p : tid & 63u), N);
    }
    unsigned amask = 0; /* per stage (8 bits each): which of this thread's A copies were real */
    auto issue = [&](unsigned kt) {
        if (kt < nk) {
            const unsigned st = kt % SPX_NS;
            float* as = (float*)(smem + st * X::STAGE);
            float* bs = as + SPT_BM * LD;
            const unsigned k0 = kb + kt * 32u;
            unsigned m = 0;
            if (sa) {
#pragma unroll
                for (unsigned p = 0; p < 2; p++) {
                    const unsigned r = (tid >> 3) + 32u * p, c = (tid & 7u) * 4u;
                    const float* src = k0 + c < ke ? sp_seg(la, ra[p], k0 + c) : nullptr;
                    sp_cp16(as + r * LD + c, src ? src : (const float*)arena, src != nullptr);
                    m |= (src != nullptr) << p;
                }
            } else {
#pragma unroll
                for (unsigned i = 0; i < 8; i++) {
                    const unsigned c = (tid >> 6) + 4u * i;
                    const float* src = k0 + c < ke ? sp_elem(la, ra[0], k0 + c) : nullptr;
                    sp_cp4(as + (tid & 63u) * LD + c, src ? src : (const float*)arena, src != nullptr);
                    m |= (src != nullptr) << i;
                }
            }
            amask = (amask & ~(0xFFu << (st * 8u))) | (m << (st * 8u));
            if (sb) {
#pragma unroll
                for (unsigned p = 0; p < 2; p++) {
                    const unsigned r = (tid >> 3) + 32u * p, c = (tid & 7u) * 4u;
                    const float* src = k0 + c < ke ? sp_seg(lb, rb[p], k0 + c) : nullptr;
                    sp_cp16(bs + r * LD + c, src ? src : (const float*)arena, src != nullptr);
                }
            } else {
#pragma unroll
                for (unsigned i = 0; i < 8; i++) {
                    const unsigned c = (tid >> 6) + 4u * i;
                    const float* src = k0 + c < ke ? sp_elem(lb, rb[0], k0 + c) : nullptr;
                    sp_cp4(bs + (tid & 63u) * LD + c, src ? src : (const float*)arena, src != nullptr);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;
#pragma unroll
    for (unsigned s = 0; s + 1 < SPX_NS; s++) issue(s);
    for (unsigned kt = 0; kt < nk; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(SPX_NS - 2));
        const unsigned st = kt % SPX_NS;
        float* as = (float*)(smem + st * X::STAGE);
        const float* bs = as + SPT_BM * LD;
        if (pa) {
            const unsigned m = amask >> (st * 8u), k0 = kb + kt * 32u;
            if (sa) {
#pragma unroll
                for (unsigned p = 0; p < 2; p++) {
                    if (!(m >> p & 1u)) continue;
                    const unsigned r = (tid >> 3) + 32u * p, c = (tid & 7u) * 4u;
                    float* d = as + r * LD + c;
#pragma unroll
                    for (unsigned q = 0; q < 4; q++) d[q] = sp_pre_at(la, ra[p], d[q], k0 + c + q);
                }
            } else {
#pragma unroll
                for (unsigned i = 0; i < 8; i++) {
                    if (!(m >> i & 1u)) continue;
                    const unsigned c = (tid >> 6) + 4u * i;
                    float* d = as + (tid & 63u) * LD + c;
                    *d = sp_pre_at(la, ra[0], *d, k0 + c);
                }
            }
        }
        __syncthreads();
        issue(kt + SPX_NS - 1);
        float part[2][2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) part[i][j][r] = 0.f;
        if (TF32) {
#pragma unroll
            for (unsigned ks = 0; ks < 32u; ks += 8) {
                unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const float* a = as + (wm + i * 16 + g) * LD + ks + t;
                    const float x[4] = {a[0], a[8 * LD], a[4], a[8 * LD + 4]};
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        ah[i][r] = sp_tf32(x[r]);
                        al[i][r] = sp_tf32(x[r] - __uint_as_float(ah[i][r]));
                    }
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const float* b = bs + (wn + j * 8 + g) * LD + ks + t;
                    const float x[2] = {b[0], b[4]};
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        bh[j][r] = sp_tf32(x[r]);
                        bl[j][r] = sp_tf32(x[r] - __uint_as_float(bh[j][r]));
                    }
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        sp_mma_tf32(part[i][j], al[i], bh[j][0], bh[j][1]);
                        sp_mma_tf32(part[i][j], ah[i], bl[j][0], bl[j][1]);
                        sp_mma_tf32(part[i][j], ah[i], bh[j][0], bh[j][1]);
                    }
            }
        } else {
#pragma unroll
            for (unsigned ks = 0; ks < 32u; ks += 16) {
                unsigned a[2][4], b[2][2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const float* ap = as + (wm + i * 16 + g) * LD + ks + 2 * t;
                    a[i][0] = sp_pack_bf16(*(const float2*)ap);
                    a[i][1] = sp_pack_bf16(*(const float2*)(ap + 8 * LD));
                    a[i][2] = sp_pack_bf16(*(const float2*)(ap + 8));
                    a[i][3] = sp_pack_bf16(*(const float2*)(ap + 8 * LD + 8));
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const float* bp = bs + (wn + j * 8 + g) * LD + ks + 2 * t;
                    b[j][0] = sp_pack_bf16(*(const float2*)bp);
                    b[j][1] = sp_pack_bf16(*(const float2*)(bp + 8));
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) sp_mma_bf16(part[i][j], a[i], b[j][0], b[j][1]);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) acc[i][j][r] = __fadd_rn(acc[i][j][r], part[i][j][r]);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}

template <class EP> __device__ __forceinline__ bool sp_ep_mfast(const EP&) { return false; }
/* Epilogue hooks over four consecutive columns: sp_ep4_load fetches what the epilogue reads from
 * memory (issued for all of a thread's groups before any store), sp_ep4 finishes them. */
template <class EP>
__device__ __forceinline__ float4 sp_ep4_load(const EP&, unsigned, unsigned, unsigned) {
    return make_float4(0.f, 0.f, 0.f, 0.f);
}
template <class EP>
__device__ __forceinline__ void sp_ep4(const EP& ep, unsigned m, unsigned n, const float* v, unsigned cnt, float4) {
    for (unsigned q = 0; q < cnt; q++) ep(m, n + q, v[q]);
}
/* The fragments go through smem so the epilogue walks the tile row-major: a warp's stores (and
 * any residual/bias loads the epilogue makes) cover 32 consecutive columns. Needs the arena free
 * (every tile function ends with a barrier) and ends with one. */
template <class EP>
static __device__ __forceinline__ void sp_tc_store(unsigned m0, unsigned n0, unsigned M, unsigned N,
                                                   const EP& ep, const float (&acc)[2][2][4], float* arena) {
    constexpr unsigned LDO = SPT_BN + 1;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned mb = (warp & 1u) * 32u + (lane >> 2), nb = (warp >> 1) * 16u + (lane & 3u) * 2u;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                arena[(mb + i * 16 + (r >> 1) * 8) * LDO + nb + j * 8 + (r & 1)] = acc[i][j][r];
    __syncthreads();
    const unsigned rows = min(SPT_BM, M - m0), cols = min(SPT_BN, N - n0);
    if (sp_ep_mfast(ep)) {
        /* Outputs contiguous along m (channel-major layouts): walk the tile column-major. */
        for (unsigned e = threadIdx.x; e < SPT_BM * SPT_BN; e += PLOW_NV_THREADS) {
            const unsigned r = e % SPT_BM, c = e / SPT_BM;
            if (r < rows && c < cols) ep(m0 + r, n0 + c, arena[r * LDO + c]);
        }
        __syncthreads();
        return;
    }
    /* Four consecutive columns per thread: an epilogue's per-row work is paid once per four. */
    constexpr unsigned G = SPT_BM * SPT_BN / 4u / PLOW_NV_THREADS;
    float4 ld[G];
#pragma unroll
    for (unsigned i = 0; i < G; i++) {
        const unsigned e = threadIdx.x + i * PLOW_NV_THREADS, r = e / (SPT_BN / 4u), c = (e % (SPT_BN / 4u)) * 4u;
        if (r < rows && c < cols) ld[i] = sp_ep4_load(ep, m0 + r, n0 + c, min(4u, cols - c));
    }
#pragma unroll
    for (unsigned i = 0; i < G; i++) {
        const unsigned e = threadIdx.x + i * PLOW_NV_THREADS, r = e / (SPT_BN / 4u), c = (e % (SPT_BN / 4u)) * 4u;
        if (r < rows && c < cols) sp_ep4(ep, m0 + r, n0 + c, arena + r * LDO + c, min(4u, cols - c), ld[i]);
    }
    __syncthreads();
}

/* Split count for `tiles` output tiles over nblk slices: >1 only with scratch and when every
 * slice gets at least one k-step. Devgen's `blocks` mirrors this (pipeline.rs dense_blocks). */
__device__ __forceinline__ unsigned sp_tc_splits(unsigned tiles, unsigned ksteps, unsigned nblk, bool scratch,
                                                 unsigned& chunk) {
    unsigned s = scratch && tiles <= SPT_TICKETS && tiles * 2u <= nblk ? nblk / tiles : 1u;
    if (s > ksteps) s = ksteps;
    if (!s) s = 1u;
    chunk = (ksteps + s - 1) / s;
    return chunk ? (ksteps + chunk - 1) / chunk : 1u;
}

/* scratch: SPT_TICKETS u32 tickets (zero at load; the reducer re-arms its own) then
 * nblk * SPT_BM * SPT_BN float partials. */
template <bool TF32, bool ASYNC = false, class LA, class LB, class EP>
static __device__ __forceinline__ void sp_tc_gemm(unsigned M, unsigned N, unsigned K, const LA& la, const LB& lb,
                                               const EP& ep, unsigned slice, unsigned nblk, float* arena,
                                               float* scratch, const SpLn* ln = nullptr) {
    using C = SpTc<TF32>;
    const unsigned tn = (N + SPT_BN - 1) / SPT_BN, tiles = ((M + SPT_BM - 1) / SPT_BM) * tn;
    const unsigned ksteps = (K + C::BK - 1) / C::BK;
    unsigned chunk;
    const unsigned S = sp_tc_splits(tiles, ksteps, nblk, scratch != nullptr, chunk);
    for (unsigned w = slice; w < tiles * S; w += nblk) {
        const unsigned tile = w / S, s = w - tile * S;
        const unsigned m0 = (tile / tn) * SPT_BM, n0 = (tile % tn) * SPT_BN;
        const unsigned kb = s * chunk * C::BK, ke = min(K, kb + chunk * C::BK);
        float acc[2][2][4];
        if constexpr (ASYNC) sp_tc_tile_async(m0, n0, M, N, kb, ke, la, lb, arena, acc, ln);
        else if (sp_async(la) && sp_async(lb)) sp_tcx_tile<TF32>(m0, n0, M, N, kb, ke, la, lb, arena, acc);
        else sp_tc_tile<TF32>(m0, n0, M, N, kb, ke, la, lb, arena, acc);
        if (S == 1u) {
            sp_tc_store(m0, n0, M, N, ep, acc, arena);
            continue;
        }
        unsigned* tickets = (unsigned*)scratch;
        float* part = scratch + SPT_TICKETS;
        float4* mine = (float4*)(part + ((size_t)tile * S + s) * (SPT_BM * SPT_BN)) + threadIdx.x * 4u;
#pragma unroll
        for (int q = 0; q < 4; q++)
            __stcg(mine + q, make_float4(acc[q >> 1][q & 1][0], acc[q >> 1][q & 1][1], acc[q >> 1][q & 1][2],
                                         acc[q >> 1][q & 1][3]));
        __threadfence();
        __syncthreads();
        unsigned* flag = (unsigned*)arena;
        if (threadIdx.x == 0) *flag = atomicAdd(tickets + tile, 1u);
        __syncthreads();
        const bool last = *flag == S - 1u;
        __syncthreads();
        if (!last) continue;
        __threadfence();
        float sum[2][2][4];
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int r = 0; r < 4; r++) sum[q >> 1][q & 1][r] = 0.f;
        for (unsigned o = 0; o < S; o++) {
            const float4* src = (const float4*)(part + ((size_t)tile * S + o) * (SPT_BM * SPT_BN)) + threadIdx.x * 4u;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 v = o == s ? make_float4(acc[q >> 1][q & 1][0], acc[q >> 1][q & 1][1],
                                                      acc[q >> 1][q & 1][2], acc[q >> 1][q & 1][3])
                                        : __ldcg(src + q);
                float* d = sum[q >> 1][q & 1];
                d[0] = __fadd_rn(d[0], v.x); d[1] = __fadd_rn(d[1], v.y);
                d[2] = __fadd_rn(d[2], v.z); d[3] = __fadd_rn(d[3], v.w);
            }
        }
        sp_tc_store(m0, n0, M, N, ep, sum, arena);
        if (threadIdx.x == 0) tickets[tile] = 0u;
    }
}

/* DenseGemmF32 (170). flags: 1 = bf16-round, 2 = bf16 erf-GELU (implies the round), 4 = bf16
 * weights, 8 = A (and f32 W) bf16-representable -> bf16 tensor cores, 16 = 3xTF32 tensor cores.
 * i5 != 0: weight row stride, and i6 < i5 names a one-hot weight column added in. t4: optional
 * split-K scratch (sp_tc_gemm). */
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
template <class LA, class LB>
static __device__ __forceinline__ void sp_dense_run(unsigned m, unsigned n, unsigned k, const LA& la,
                                                    const LB& lb, const SpDenseEpi& ep, unsigned flags,
                                                    unsigned slice, unsigned nblk, float* arena, float* scratch) {
    if (flags & 8u) sp_tc_gemm<false>(m, n, k, la, lb, ep, slice, nblk, arena, scratch);
    else if (flags & 16u) sp_tc_gemm<true>(m, n, k, la, lb, ep, slice, nblk, arena, scratch);
    else sp_gemm(m, n, k, la, lb, ep, slice, nblk, arena);
}
static __device__ __noinline__ void d_dense_gemm_f32(float* __restrict__ out, const float* __restrict__ x,
                                        const void* __restrict__ w, const float* __restrict__ bias,
                                        unsigned m, unsigned n, unsigned k, unsigned activation,
                                        unsigned a_row0, unsigned wstride_op, unsigned onehot,
                                        unsigned flags, unsigned slice, unsigned nblk, float* arena,
                                        float* scratch, const SpLn* ln) {
    arena = sp_smem;
    const unsigned stride = wstride_op ? wstride_op : k;
    x += (size_t)a_row0 * k;
    const SpRowF32 la{x, k, (k & 3u) == 0 && sp_aligned(x, 16)};
    const SpDenseEpi ep{out, bias, w, n, stride, wstride_op ? onehot : 0xFFFFFFFFu, flags,
                        activation == 1u};
    if (flags & 4u) {
        const __nv_bfloat16* wb = (const __nv_bfloat16*)w;
        if ((flags & 8u) && (k & 7u) == 0 && (stride & 7u) == 0 && sp_aligned(x, 16) && sp_aligned(wb, 16)) {
            sp_tc_gemm<false, true>(m, n, k, la, SpRowBf16{wb, k, stride, true}, ep, slice, nblk, arena, scratch, ln);
            return;
        }
        const SpRowBf16 lb{wb, k, stride, (stride & 3u) == 0 && sp_aligned(wb, 8)};
        if (ln) sp_dense_run(m, n, k, SpRowLnF32{x, k, la.vec, *ln}, lb, ep, flags, slice, nblk, arena, scratch);
        else sp_dense_run(m, n, k, la, lb, ep, flags, slice, nblk, arena, scratch);
    } else {
        const float* wf = (const float*)w;
        const SpRowF32S lb{wf, k, stride, (stride & 3u) == 0 && sp_aligned(wf, 16)};
        if (ln) sp_dense_run(m, n, k, SpRowLnF32{x, k, la.vec, *ln}, lb, ep, flags, slice, nblk, arena, scratch);
        else sp_dense_run(m, n, k, la, lb, ep, flags, slice, nblk, arena, scratch);
    }
}

/* GemmF32 (180): bf16 x bf16 -> f32. */
struct SpPlainEpi {
    float* out; unsigned n;
    __device__ void operator()(unsigned m, unsigned c, float acc) const { out[(size_t)m * n + c] = acc; }
};
static __device__ __noinline__ void d_gemm_f32(float* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ w, unsigned m, unsigned n,
                                  unsigned k, unsigned slice, unsigned nblk, float* arena) {
    arena = sp_smem;
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
static __device__ __noinline__ void d_q8_gemm_f32(float* __restrict__ out, const float* __restrict__ x,
                                     const uint8_t* __restrict__ w, const float* __restrict__ bias,
                                     unsigned m, unsigned n, unsigned k, unsigned activation,
                                     unsigned a_row0, unsigned slice, unsigned nblk, float* arena) {
    arena = sp_smem;
    if (k & 31u) { __trap(); return; }
    x += (size_t)a_row0 * k;
    sp_gemm(m, n, k, SpRowF32{x, k, sp_aligned(x, 16)}, SpRowQ8{w, k},
            SpQ8Epi{out, bias, n, activation == 1u}, slice, nblk, arena);
}

/* Conv2dF32 (176). flags (fj1): 1 depthwise, 2 relu, [3:2] out layout, [5:4] in layout,
 * 64 f32 weights (else f16), 128 bf16 erf-GELU, 256 input and f32 weights bf16-representable
 * (bf16 tensor cores), 512 3xTF32 tensor cores. Layouts: 0 NFWC, 1 NFCW, 2 NCFW. */
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
/* Conv2dF32's bf16 tensor-core path over wide output tiles: 64 positions x 256 channels per block,
 * so each im2col gather feeds 4x more MMAs than a 64x64 tile (the gathers bound this op). A: the
 * table-driven gather, two k-tiles in registers; W: f32 rows by cp.async. BK = 16. */
#define SPW_BN 256
#define SPW_LDA 24
#define SPW_LDB 24
#define SPW_A_BYTES (SPT_BM * SPW_LDA * 4)
#define SPW_B_BYTES (SPW_BN * SPW_LDB * 4)
/* The column table outlives every tile: past both the k-stages and the epilogue's output stage. */
#define SPW_EPI_FLOATS (SPT_BM * (SPW_BN + 1))
#define SPW_TAB0 ((2 * (SPW_A_BYTES + SPW_B_BYTES)) / 4 > SPW_EPI_FLOATS ? (2 * (SPW_A_BYTES + SPW_B_BYTES)) / 4 \
                                                                        : SPW_EPI_FLOATS)
template <class EP>
static __device__ __forceinline__ void sp_conv2d_wide(unsigned M, unsigned N, unsigned K, const SpRowConvT& la,
                                                      const float* w, const EP& ep, bool m_fast, unsigned slice,
                                                      unsigned nblk, float* arena) {
    char* smem = (char*)arena;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = (warp & 1u) * 32u, wn = (warp >> 1) * 64u, g = lane >> 2, t = lane & 3u;
    const unsigned tn = (N + SPW_BN - 1) / SPW_BN, tiles = ((M + SPT_BM - 1) / SPT_BM) * tn;
    const unsigned nk = (K + 15u) / 16u;
    for (unsigned tile = slice; tile < tiles; tile += nblk) {
        const unsigned m0 = (tile / tn) * SPT_BM, n0 = (tile % tn) * SPW_BN;
        const SpConvPosT ra = la.row(m0 + (tid & 63u), M);
        float av[2][4];
        auto fetch_a = [&](unsigned kt, float (&v)[4]) {
#pragma unroll
            for (unsigned i = 0; i < 4; i++) v[i] = la.at(ra, kt * 16u + (tid >> 6) + 4u * i);
        };
        auto issue_b = [&](unsigned kt) {
            if (kt < nk) {
                float* bs = (float*)(smem + 2 * SPW_A_BYTES + (kt & 1u) * SPW_B_BYTES);
#pragma unroll
                for (unsigned p = 0; p < SPW_BN * 4u / PLOW_NV_THREADS; p++) {
                    const unsigned e = tid + p * PLOW_NV_THREADS, r = e >> 2, c = (e & 3u) * 4u;
                    const unsigned k = kt * 16u + c;
                    const bool ok = n0 + r < N && k < K;
                    sp_cp16(bs + r * SPW_LDB + c, ok ? w + (size_t)(n0 + r) * K + k : w, ok);
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        float acc[2][8][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;
        issue_b(0);
        fetch_a(0, av[0]);
        if (nk > 1u) fetch_a(1, av[1]);
        auto step = [&](unsigned kt, float (&v)[4]) {
            float* as = (float*)(smem + (kt & 1u) * SPW_A_BYTES);
#pragma unroll
            for (unsigned i = 0; i < 4; i++) as[(tid & 63u) * SPW_LDA + (tid >> 6) + 4u * i] = v[i];
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            issue_b(kt + 1);
            if (kt + 2 < nk) fetch_a(kt + 2, v);
            const float* bs = (const float*)(smem + 2 * SPW_A_BYTES + (kt & 1u) * SPW_B_BYTES);
            unsigned a[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const float* ap = as + (wm + i * 16 + g) * SPW_LDA + 2 * t;
                a[i][0] = sp_pack_bf16(*(const float2*)ap);
                a[i][1] = sp_pack_bf16(*(const float2*)(ap + 8 * SPW_LDA));
                a[i][2] = sp_pack_bf16(*(const float2*)(ap + 8));
                a[i][3] = sp_pack_bf16(*(const float2*)(ap + 8 * SPW_LDA + 8));
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float* bp = bs + (wn + j * 8 + g) * SPW_LDB + 2 * t;
                const unsigned b0 = sp_pack_bf16(*(const float2*)bp), b1 = sp_pack_bf16(*(const float2*)(bp + 8));
#pragma unroll
                for (int i = 0; i < 2; i++) sp_mma_bf16(acc[i][j], a[i], b0, b1);
            }
        };
        /* Tensor-core accumulation truncates: fold into tot with a rounded add every 64 k. */
        float tot[2][8][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) tot[i][j][r] = 0.f;
        for (unsigned kt = 0; kt < nk; kt += 2) {
            step(kt, av[0]);
            if (kt + 1 < nk) step(kt + 1, av[1]);
            if ((kt & 3u) == 2u || kt + 2 >= nk) {
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++)
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            tot[i][j][r] = __fadd_rn(tot[i][j][r], acc[i][j][r]);
                            acc[i][j][r] = 0.f;
                        }
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        constexpr unsigned LDO = SPW_BN + 1;
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++)
                    arena[(wm + i * 16 + g + (r >> 1) * 8) * LDO + wn + j * 8 + 2 * t + (r & 1)] = tot[i][j][r];
        __syncthreads();
        const unsigned rows = min(SPT_BM, M - m0), cols = min((unsigned)SPW_BN, N - n0);
        for (unsigned e = tid; e < SPT_BM * SPW_BN; e += PLOW_NV_THREADS) {
            const unsigned r = m_fast ? e % SPT_BM : e / SPW_BN, c = m_fast ? e / SPT_BM : e % SPW_BN;
            if (r < rows && c < cols) ep(m0 + r, n0 + c, arena[r * LDO + c]);
        }
        __syncthreads();
    }
}

/* NCFW output: consecutive positions are adjacent. */
__device__ __forceinline__ bool sp_ep_mfast(const SpConvEpi& ep) { return ep.layout == 2u; }
static __device__ __noinline__ void d_conv2d_f32(float* __restrict__ out, const float* __restrict__ x,
                                    const void* __restrict__ w, const float* __restrict__ bias,
                                    unsigned frames, unsigned width, unsigned ic, unsigned oc,
                                    unsigned kernel, unsigned stride, unsigned pad_before,
                                    unsigned pad_after, unsigned flags, unsigned batches,
                                    unsigned slice, unsigned nblk, float* arena) {
    arena = sp_smem;
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
        if (weight_f32) {
            const SpRowF32 lb{(const float*)w, k, (k & 3u) == 0 && sp_aligned(w, 16)};
            /* The column table sits past the tensor-core stages (sp_tcx_tile's ring is the larger). */
            constexpr unsigned TAB0 = SPX_NS * SpX<false>::STAGE / 4u;
            static_assert(TAB0 >= 4u * SpTc<false>::STAGE_BYTES / 4u, "conv table placement");
            SpConvTab* tab = (SpConvTab*)(arena + TAB0);
            const bool tabled = TAB0 + 2u * k <= SP_ARENA_FLOATS && (uint64_t)ic * frames * width < (1ull << 31);
            if ((flags & 256u) && SPW_TAB0 + 2u * k <= SP_ARENA_FLOATS && (k & 3u) == 0 && sp_aligned(w, 16) &&
                (uint64_t)ic * frames * width < (1ull << 31)) {
                SpConvTab* wtab = (SpConvTab*)(arena + SPW_TAB0);
                sp_conv_table(wtab, g, k);
                sp_conv2d_wide(positions, oc, k, SpRowConvT{x, wtab, g, k}, (const float*)w, ep, out_layout == 2u,
                               slice, nblk, arena);
            } else if ((flags & 768u) && tabled) {
                sp_conv_table(tab, g, k);
                const SpRowConvT lt{x, tab, g, k};
                if (flags & 256u) sp_tc_gemm<false>(positions, oc, k, lt, lb, ep, slice, nblk, arena, nullptr);
                else sp_tc_gemm<true>(positions, oc, k, lt, lb, ep, slice, nblk, arena, nullptr);
            } else if (flags & 256u) {
                sp_tc_gemm<false>(positions, oc, k, la, lb, ep, slice, nblk, arena, nullptr);
            } else if (flags & 512u) {
                sp_tc_gemm<true>(positions, oc, k, la, lb, ep, slice, nblk, arena, nullptr);
            } else {
                sp_gemm(positions, oc, k, la, lb, ep, slice, nblk, arena);
            }
        } else {
            sp_gemm(positions, oc, k, la, SpRowF16{(const uint16_t*)w, k}, ep, slice, nblk, arena);
        }
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
 * (else double). The ordered path gives each row a warp: the warp stages the row through smem in
 * SPL_COLS chunks (coalesced) and lane 0 runs the golden's serial float chains over smem, which is
 * the only way to reproduce its rounding; the chain (~4 cycles per element) bounds the op. */
#define SPL_COLS 1024
static_assert(PLOW_NV_WARPS * SPL_COLS <= SP_ARENA_FLOATS, "layernorm row stage");
/* With `stats` (RowStatsF32, 204) only [rows][2] = (mean, inverse std) is written. */
static __device__ __noinline__ void d_layernorm_f32(float* __restrict__ out, const float* __restrict__ x,
                                       const float* __restrict__ gamma, const float* __restrict__ beta,
                                       unsigned rows, unsigned feat, unsigned flags, float eps,
                                       unsigned slice, unsigned nblk, float* arena, float* stats = nullptr) {
    arena = sp_smem;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (flags & 2u) {
        float* xs = arena + warp * SPL_COLS;
        const float ff = (float)feat;
        const bool vec = (feat & 3u) == 0 && sp_aligned(x, 16);
        const bool whole = feat <= SPL_COLS;
        /* The fast epilogue's gamma/beta loads issue before the serial chains, which hide them. */
        const bool fast = whole && vec && (!gamma || sp_aligned(gamma, 16)) && (!beta || sp_aligned(beta, 16)) &&
                          out && sp_aligned(out, 16);
        constexpr unsigned U = SPL_COLS / 128u;
        float4 gv[U], bv[U];
#pragma unroll
        for (unsigned u = 0; u < U; u++) {
            const unsigned c = lane * 4u + u * 128u;
            gv[u] = make_float4(1.f, 1.f, 1.f, 1.f);
            bv[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (fast && c < feat && gamma) gv[u] = __ldg((const float4*)(gamma + c));
            if (fast && c < feat && beta) bv[u] = __ldg((const float4*)(beta + c));
        }
        for (unsigned row = slice * PLOW_NV_WARPS + warp; row < rows; row += nblk * PLOW_NV_WARPS) {
            const float* xr = x + (size_t)row * feat;
            float mean = 0.f, acc = 0.f;
            for (int pass = 0; pass < 2; pass++) {
                acc = 0.f;
                for (unsigned c0 = 0; c0 < feat; c0 += SPL_COLS) {
                    const unsigned nc = min(feat - c0, (unsigned)SPL_COLS);
                    if (!whole || pass == 0) {
                        if (vec) {
                            constexpr unsigned U = SPL_COLS / 128u;
                            float4 v[U];
#pragma unroll
                            for (unsigned u = 0; u < U; u++) {
                                const unsigned c = lane * 4u + u * 128u;
                                if (c < nc) v[u] = *(const float4*)(xr + c0 + c);
                            }
#pragma unroll
                            for (unsigned u = 0; u < U; u++) {
                                const unsigned c = lane * 4u + u * 128u;
                                if (c < nc) *(float4*)(xs + c) = v[u];
                            }
                        } else
                            for (unsigned c = lane; c < nc; c += 32u) xs[c] = xr[c0 + c];
                        __syncwarp();
                    }
                    if (lane == 0) {
                        unsigned c = 0;
                        if (pass == 0) {
#pragma unroll 8
                            for (; c + 4 <= nc; c += 4) {
                                const float4 v = *(const float4*)(xs + c);
                                acc = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(acc, v.x), v.y), v.z), v.w);
                            }
                            for (; c < nc; c++) acc = __fadd_rn(acc, xs[c]);
                        } else {
#pragma unroll 8
                            for (; c + 4 <= nc; c += 4) {
                                const float4 v = *(const float4*)(xs + c);
                                const float a = __fsub_rn(v.x, mean), b = __fsub_rn(v.y, mean);
                                const float d = __fsub_rn(v.z, mean), e = __fsub_rn(v.w, mean);
                                acc = __fadd_rn(acc, __fmul_rn(a, a));
                                acc = __fadd_rn(acc, __fmul_rn(b, b));
                                acc = __fadd_rn(acc, __fmul_rn(d, d));
                                acc = __fadd_rn(acc, __fmul_rn(e, e));
                            }
                            for (; c < nc; c++) {
                                const float v = __fsub_rn(xs[c], mean);
                                acc = __fadd_rn(acc, __fmul_rn(v, v));
                            }
                        }
                    }
                    __syncwarp();
                }
                if (pass == 0) mean = __shfl_sync(0xffffffffu, __fdiv_rn(acc, ff), 0);
            }
            const float inv =
                __shfl_sync(0xffffffffu, __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc, ff), eps))), 0);
            if (stats) {
                if (lane == 0) *(float2*)(stats + 2 * (size_t)row) = make_float2(mean, inv);
                __syncwarp();
                continue;
            }
            float* yr = out + (size_t)row * feat;
            if (fast) {
#pragma unroll
                for (unsigned u = 0; u < U; u++) {
                    const unsigned c = lane * 4u + u * 128u;
                    if (c >= feat) continue;
                    const float4 xv = *(const float4*)(xs + c);
                    const float xa[4] = {xv.x, xv.y, xv.z, xv.w}, ga[4] = {gv[u].x, gv[u].y, gv[u].z, gv[u].w},
                                ba[4] = {bv[u].x, bv[u].y, bv[u].z, bv[u].w};
                    float y[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float v = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xa[q], mean), inv), ga[q]), ba[q]);
                        y[q] = flags & 1u ? sp_bf16(v) : v;
                    }
                    *(float4*)(yr + c) = make_float4(y[0], y[1], y[2], y[3]);
                }
            } else {
                for (unsigned c = lane; c < feat; c += 32u) {
                    float v = __fmul_rn(__fsub_rn(whole ? xs[c] : xr[c], mean), inv);
                    v = __fadd_rn(__fmul_rn(v, gamma ? gamma[c] : 1.0f), beta ? beta[c] : 0.0f);
                    yr[c] = flags & 1u ? sp_bf16(v) : v;
                }
            }
            __syncwarp();
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
        if (stats) {
            if (lane == 0) *(float2*)(stats + 2 * (size_t)row) = make_float2(mean, inv);
            continue;
        }
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
static __device__ __noinline__ void d_scaled_add_f32(float* out, const float* a, const float* b, unsigned n,
                                        float s, unsigned flags, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) {
        const float v = __fadd_rn(a[i], __fmul_rn(s, b[i]));
        out[i] = flags & 1u ? sp_bf16(v) : v;
    }
}

/* GluF32 (166): x is [rows][2*width], out = a * sigmoid(b). */
static __device__ __noinline__ void d_glu_f32(float* out, const float* x, unsigned rows, unsigned width,
                                 unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, rows * width) {
        const unsigned r = i / width, c = i - r * width;
        out[i] = __fmul_rn(x[(size_t)r * 2u * width + c], sp_sigmoid(x[(size_t)r * 2u * width + width + c]));
    }
}

static __device__ __noinline__ void d_silu_f32(float* out, const float* x, unsigned n, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) out[i] = sp_silu(x[i]);
}

static __device__ __noinline__ void d_relu_f32(float* out, const float* x, unsigned n, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, n) out[i] = fmaxf(x[i], 0.0f);
}

static __device__ __noinline__ void d_broadcast_add_f32(float* out, const float* m, const float* v, unsigned rows,
                                           unsigned width, unsigned slice, unsigned nblk) {
    SP_FOR_EACH(i, rows * width) out[i] = __fadd_rn(m[i], v[i % width]);
}

/* CausalDepthwiseConv1dF32 (167): f16 weights [channel][kernel], taps summed in order. */
static __device__ __noinline__ void d_causal_dwconv1d_f32(float* out, const float* x, const uint16_t* w,
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
static __device__ __noinline__ void d_embed_f16_f32(float* out, const uint16_t* table, const unsigned* token,
                                       unsigned vocab, unsigned width, unsigned slice, unsigned nblk) {
    const unsigned t = *token;
    if (t >= vocab) return;
    SP_FOR_EACH(i, width) out[i] = sp_f16(table[(size_t)t * width + i]);
}

/* LstmCellF32 (172): gates = [i | f | g | o] x width. */
static __device__ __noinline__ void d_lstm_cell_f32(float* h_new, float* c_new, const float* gates,
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
static __device__ __noinline__ void d_argmax_f32(unsigned* ids, const float* x, unsigned rows, unsigned width,
                                    unsigned slice, unsigned nblk, float* arena) {
    arena = sp_smem;
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
static __device__ __noinline__ void d_pack_ncfw_rows_f32(float* out, const float* x, unsigned rows,
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
 * first *valid_rows rows. One block per (group, head, 8-query chunk), a warp per query row. K (padded
 * rows) and V for the group's head are staged in smem when they fit. Every sum is the golden's
 * serial float chain: lanes own keys for the score dots and columns for P.V, each lane running
 * several independent chains at once. flags: 1 bf16 score, 2 bf16 probability, 4 bf16 out. */
#define SPA_CHAINS 4
/* context[c] = sum_j p[j] * v[j][c] in key order; a lane owns columns lane + 32u, u < CH. */
template <int CH>
static __device__ __forceinline__ void sp_ga_pv(float* o, const float* p, const float* vb, unsigned vstride,
                                                unsigned n, unsigned hw, unsigned flags, unsigned lane) {
    for (unsigned c0 = lane; c0 < hw; c0 += 32u * CH) {
        float acc[CH];
        const float* vc[CH];
#pragma unroll
        for (int u = 0; u < CH; u++) {
            const unsigned c = c0 + 32u * u;
            vc[u] = vb + (c < hw ? c : 0u);
            acc[u] = 0.f;
        }
#pragma unroll 8
        for (unsigned j = 0; j < n; j++) {
            const float pj = p[j];
#pragma unroll
            for (int u = 0; u < CH; u++) acc[u] = __fadd_rn(acc[u], __fmul_rn(pj, vc[u][(size_t)j * vstride]));
        }
#pragma unroll
        for (int u = 0; u < CH; u++) {
            const unsigned c = c0 + 32u * u;
            if (c < hw) o[c] = flags & 4u ? sp_bf16(acc[u]) : acc[u];
        }
    }
}
static __device__ __noinline__ void d_grouped_attention_f32(float* __restrict__ context, const float* __restrict__ query,
                                               const float* __restrict__ key, const float* __restrict__ value,
                                               const unsigned* valid, unsigned rows, unsigned width,
                                               unsigned hw, unsigned group_rows, unsigned flags,
                                               unsigned slice, unsigned nblk, float* arena) {
    arena = sp_smem;
    const unsigned valid_rows = valid ? *valid : rows;
    if (hw == 0 || width % hw != 0 || group_rows == 0 || group_rows > 256u || valid_rows == 0 ||
        valid_rows > rows) return;
    const unsigned heads = width / hw, groups = (valid_rows + group_rows - 1) / group_rows;
    const unsigned chunks = (group_rows + PLOW_NV_WARPS - 1) / PLOW_NV_WARPS;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    float* sc = arena + warp * 256u;
    const unsigned kfl = (group_rows * (hw + 1u) + 3u) & ~3u, vfl = group_rows * hw;
    const bool kstaged = SPA_SCORE_FLOATS + kfl <= SP_ARENA_FLOATS;
    const bool vstaged = kstaged && SPA_SCORE_FLOATS + kfl + vfl <= SP_ARENA_FLOATS;
    const bool qstaged = vstaged && SPA_SCORE_FLOATS + kfl + vfl + PLOW_NV_WARPS * hw <= SP_ARENA_FLOATS;
    float* ks = arena + SPA_SCORE_FLOATS;
    float* vs = ks + kfl;
    float* qs = vs + vfl + warp * hw;
    const float scale = __fsqrt_rn((float)hw);
    const bool vec = hw % 4u == 0 && width % 4u == 0 && sp_aligned(key, 16) && sp_aligned(value, 16);
    for (unsigned item = slice; item < groups * heads * chunks; item += nblk) {
        const unsigned chunk = item % chunks, gh = item / chunks, g = gh / heads, head = gh - g * heads;
        const unsigned first = g * group_rows;
        const unsigned last = first + group_rows < valid_rows ? first + group_rows : valid_rows;
        const unsigned n = last - first;
        if (chunk * PLOW_NV_WARPS >= n) continue;
        const float* kb = key + (size_t)first * width + head * hw;
        const float* vb = value + (size_t)first * width + head * hw;
        unsigned kstride = width, vstride = width;
        if (kstaged) {
            if (vstaged && vec) {
                /* float4 loads, SPA_CHAINS per thread in flight before the first store. */
                const unsigned h4 = hw / 4u;
                for (unsigned e0 = threadIdx.x; e0 < n * h4; e0 += PLOW_NV_THREADS * SPA_CHAINS) {
                    float4 kv[SPA_CHAINS], vv[SPA_CHAINS];
#pragma unroll
                    for (int u = 0; u < SPA_CHAINS; u++) {
                        const unsigned e = e0 + u * PLOW_NV_THREADS, r = e / h4, c = (e - r * h4) * 4u;
                        if (e < n * h4) {
                            kv[u] = *(const float4*)(kb + (size_t)r * width + c);
                            vv[u] = *(const float4*)(vb + (size_t)r * width + c);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < SPA_CHAINS; u++) {
                        const unsigned e = e0 + u * PLOW_NV_THREADS, r = e / h4, c = (e - r * h4) * 4u;
                        if (e < n * h4) {
                            float* kd = ks + r * (hw + 1u) + c;
                            kd[0] = kv[u].x; kd[1] = kv[u].y; kd[2] = kv[u].z; kd[3] = kv[u].w;
                            *(float4*)(vs + r * hw + c) = vv[u];
                        }
                    }
                }
            } else {
                for (unsigned e = threadIdx.x; e < n * hw; e += PLOW_NV_THREADS) {
                    const unsigned r = e / hw, c = e - r * hw;
                    ks[r * (hw + 1u) + c] = kb[(size_t)r * width + c];
                    if (vstaged) vs[e] = vb[(size_t)r * width + c];
                }
            }
            kb = ks;
            kstride = hw + 1u;
            if (vstaged) { vb = vs; vstride = hw; }
            __syncthreads();
        }
        const unsigned row = first + chunk * PLOW_NV_WARPS + warp;
        if (row < last) {
            const float* q = query + (size_t)row * width + head * hw;
            if (qstaged) {
                for (unsigned c = lane; c < hw; c += 32u) qs[c] = q[c];
                __syncwarp();
                q = qs;
            }
            float mx = -INFINITY;
            for (unsigned j0 = lane; j0 < n; j0 += 32u * SPA_CHAINS) {
                const float* kr[SPA_CHAINS];
                float sv[SPA_CHAINS];
#pragma unroll
                for (int u = 0; u < SPA_CHAINS; u++) {
                    const unsigned j = j0 + 32u * u;
                    kr[u] = kb + (size_t)(j < n ? j : 0u) * kstride;
                    sv[u] = 0.f;
                }
#pragma unroll 4
                for (unsigned c = 0; c < hw; c++) {
                    const float qc = q[c];
#pragma unroll
                    for (int u = 0; u < SPA_CHAINS; u++) sv[u] = __fadd_rn(sv[u], __fmul_rn(qc, kr[u][c]));
                }
#pragma unroll
                for (int u = 0; u < SPA_CHAINS; u++) {
                    const unsigned j = j0 + 32u * u;
                    if (j >= n) break;
                    float v = flags & 1u ? sp_bf16(sv[u]) : sv[u];
                    v = __fdiv_rn(v, scale);
                    sc[j] = v;
                    mx = fmaxf(mx, v);
                }
            }
            mx = sp_warp_max(mx);
            for (unsigned j = lane; j < n; j += 32u) sc[j] = sp_expf(__fsub_rn(sc[j], mx));
            __syncwarp();
            float den = 0.f;
#pragma unroll 8
            for (unsigned j = 0; j < n; j++) den = __fadd_rn(den, sc[j]);
            __syncwarp();
            for (unsigned j = lane; j < n; j += 32u) {
                const float p = __fdiv_rn(sc[j], den);
                sc[j] = flags & 2u ? sp_bf16(p) : p;
            }
            __syncwarp();
            float* o = context + (size_t)row * width + head * hw;
            if (hw <= 64u) sp_ga_pv<2>(o, sc, vb, vstride, n, hw, flags, lane);
            else sp_ga_pv<SPA_CHAINS>(o, sc, vb, vstride, n, hw, flags, lane);
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
static __device__ __noinline__ void d_relative_attention_f32(float* __restrict__ context, const float* __restrict__ query,
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
static __device__ __noinline__ void d_gather_rows_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
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
static __device__ __noinline__ void d_copy_cols_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
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
static __device__ __noinline__ void d_unary_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const float* x = (const float*)SP_TEN(1);
    const float* param = (const float*)SP_TEN(2);
    const unsigned rows = in->i[0], width = in->i[1], kind = in->i[2];
    const unsigned stride = in->i[3] ? in->i[3] : width, col0 = in->i[4];
    if (!width || (uint64_t)rows * width > 0xFFFFFFFFull) return;
    const float p0 = in->fj[0].f;
    const bool vec = width % 4u == 0 && stride % 4u == 0 && col0 % 4u == 0 && sp_aligned(x, 16) &&
                     sp_aligned(out, 16) && (!param || sp_aligned(param, 16));
    if (vec) {
        const unsigned w4 = width / 4u;
        sp_with_act(kind, in->fj[1].f, [&](auto f) {
            sp_batched<2>(
                rows * w4, slice, nblk,
                [&](unsigned e) {
                    const unsigned row = e / w4;
                    return ((const float4*)(x + (size_t)row * stride + col0))[e - row * w4];
                },
                [&](unsigned e, float4 v) {
                    const unsigned row = e / w4, c = (e - row * w4) * 4u;
                    const float4 q = param ? __ldg((const float4*)(param + c)) : make_float4(p0, p0, p0, p0);
                    *(float4*)(out + (size_t)row * stride + col0 + c) =
                        make_float4(f(v.x, q.x), f(v.y, q.y), f(v.z, q.z), f(v.w, q.w));
                });
        });
        return;
    }
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

struct SpF4x2 { float4 a, b; };
/* BinaryF32 (200). */
static __device__ __noinline__ void d_binary_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    float* out = (float*)SP_TEN(0);
    const float* a = (const float*)SP_TEN(1);
    const float* b = (const float*)SP_TEN(2);
    const unsigned items = in->i[0], rows = in->i[1], width = in->i[2], op = in->i[3];
    if (!rows || !width || op > 5u || (uint64_t)items * rows * width > 0xFFFFFFFFull) return;
    const bool scaled = in->i[7] & 1u;
    const float scale = in->fj[0].f;
    const unsigned bis = in->i[4], brs = in->i[5], bcs = in->i[6];
    /* float4 form: b contiguous (column stride 1, 16B-aligned rows) or broadcast along the row. */
    const bool vec = width % 4u == 0 && (bcs == 1u || bcs == 0u) &&
                     (bcs == 0u || (bis % 4u == 0 && brs % 4u == 0 && sp_aligned(b, 16))) && sp_aligned(a, 16) &&
                     sp_aligned(out, 16);
    const auto run = [&](auto f) {
        if (vec) {
            const unsigned w4 = width / 4u;
            sp_batched<2>(
                items * rows * w4, slice, nblk,
                [&](unsigned e) {
                    const unsigned rr = e / w4, c = (e - rr * w4) * 4u, item = rr / rows, row = rr - item * rows;
                    const float* bp = b + (size_t)item * bis + (size_t)row * brs;
                    const float4 bv = bcs ? *(const float4*)(bp + c) : make_float4(*bp, *bp, *bp, *bp);
                    return SpF4x2{((const float4*)a)[e], bv};
                },
                [&](unsigned e, SpF4x2 v) {
                    float4 r = make_float4(f(v.a.x, v.b.x), f(v.a.y, v.b.y), f(v.a.z, v.b.z), f(v.a.w, v.b.w));
                    if (scaled)
                        r = make_float4(__fmul_rn(scale, r.x), __fmul_rn(scale, r.y), __fmul_rn(scale, r.z),
                                        __fmul_rn(scale, r.w));
                    ((float4*)out)[e] = r;
                });
            return;
        }
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
static __device__ __noinline__ void d_cumsum_f64(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk,
                                    float* arena) {
    arena = sp_smem;
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
static __device__ __noinline__ void d_rand_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
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
    const float* residual; const unsigned* lengths; const float* row_scale;
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
    /* oi = row * cout + o. Conv1d's optional row_scale multiplies after the output activation. */
    __device__ void store(size_t oi, unsigned o, float acc) const {
        float v = post ? sp_act_call(post, acc, alpha ? alpha[o] : slope) : acc;
        if (row_scale) v = __fmul_rn(row_scale[oi / cout], v);
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
    a.row_scale = transpose ? nullptr : (const float*)SP_TEN(7);
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
/* ConvTranspose weights [cin][ng][kernel]: consecutive n are `kernel` apart, consecutive k far. */
__device__ __forceinline__ bool sp_rowfast(const SpRowConvW& l) { return l.transpose; }
/* cp.async hooks (sp_tcx_tile). */
__device__ __forceinline__ bool sp_async(const SpRowConv1&) { return true; }
__device__ __forceinline__ bool sp_segs(const SpRowConv1& l) { return l.vec; }
__device__ __forceinline__ const float* sp_elem(const SpRowConv1& l, const SpConvPos1& s, unsigned c) {
    if (!s.ok || c >= l.K) return nullptr;
    if (l.pointwise) return s.xb + c;
    const unsigned tap = l.a.div_cg(c), ci = c - tap * l.a.cg;
    const int u = l.in_row(s, tap);
    return u < 0 ? nullptr : s.xb + (size_t)u * l.a.cin + ci;
}
__device__ __forceinline__ const float* sp_seg(const SpRowConv1& l, const SpConvPos1& s, unsigned c) {
    return sp_elem(l, s, c);
}
__device__ __forceinline__ bool sp_pre(const SpRowConv1& l) { return l.a.pre != 0u; }
__device__ __forceinline__ float sp_pre_at(const SpRowConv1& l, const SpConvPos1&, float v, unsigned c) {
    return l.a.pre_at(v, l.c0 + (l.pointwise ? c : c - l.a.div_cg(c) * l.a.cg));
}
__device__ __forceinline__ bool sp_async(const SpRowConvW& l) { return !l.a.wf16; }
__device__ __forceinline__ bool sp_segs(const SpRowConvW& l) { return l.vec; }
__device__ __forceinline__ const float* sp_elem(const SpRowConvW& l, long long n, unsigned c) {
    if (n < 0 || c >= l.K) return nullptr;
    const unsigned tap = l.a.div_cg(c), ci = c - tap * l.a.cg;
    const float* w = (const float*)l.a.w;
    if (l.transpose)
        return w + ((size_t)(l.c0 + ci) * l.a.ng + (size_t)(n - l.n0)) * l.a.kernel + l.phase + tap * l.a.stride;
    return w + ((size_t)n * l.a.cg + ci) * l.a.kernel + tap;
}
__device__ __forceinline__ const float* sp_seg(const SpRowConvW& l, long long n, unsigned c) {
    return n < 0 || c >= l.K ? nullptr : (const float*)l.a.w + (size_t)n * l.K + c;
}
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
    /* Output index of row m's column 0 (UINT64_MAX: row masked to zero). */
    __device__ size_t row_base(unsigned m) const {
        if (!transpose && !a.lengths) return (size_t)m * a.cout;
        unsigned b, t;
        bool ok;
        if (transpose) {
            b = m / n_p;
            t = (q_lo + (m - b * n_p)) * a.stride + phase - a.before;
            ok = (int)t < a.convt_len(b);
        } else {
            b = m / a.out_rows;
            t = m - b * a.out_rows;
            ok = t < a.conv_len(b);
        }
        const size_t base = ((size_t)b * a.out_rows + t) * a.cout;
        return ok ? base : ~base;
    }
};
__device__ __forceinline__ float4 sp_ep4_load(const SpConvEpi1& ep, unsigned m, unsigned n, unsigned cnt) {
    float r[4] = {0.f, 0.f, 0.f, 0.f};
    if (ep.a.residual) {
        const size_t base = ep.row_base(m);
        if ((long long)base >= 0)
            for (unsigned q = 0; q < cnt; q++) r[q] = ep.a.residual[base + ep.n0 + n + q];
    }
    return make_float4(r[0], r[1], r[2], r[3]);
}
__device__ __forceinline__ void sp_ep4(const SpConvEpi1& ep, unsigned m, unsigned n, const float* v, unsigned cnt,
                                       float4 res) {
    size_t base = ep.row_base(m);
    const bool ok = (long long)base >= 0;
    if (!ok) base = ~base;
    const float rr[4] = {res.x, res.y, res.z, res.w};
    for (unsigned q = 0; q < cnt; q++) {
        const unsigned o = ep.n0 + n + q;
        if (!ok) { ep.a.out[base + o] = 0.f; continue; }
        const float acc = ep.a.bias ? __fadd_rn(v[q], ep.a.bias[o]) : v[q];
        float y = ep.a.post ? sp_act_call(ep.a.post, acc, ep.a.alpha ? ep.a.alpha[o] : ep.a.slope) : acc;
        if (ep.a.row_scale) y = __fmul_rn(ep.a.row_scale[base / ep.a.cout], y);
        ep.a.out[base + o] = ep.a.residual ? __fadd_rn(y, rr[q]) : y;
    }
}

static __device__ __noinline__ void d_conv1d_f32(const PlowDevInst* in, void* const* T, bool transpose, unsigned slice,
                                    unsigned nblk, float* arena) {
    arena = sp_smem;
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
    /* flags bit 13: 3xTF32 tensor cores on 64x64 tiles, except when FP32 FFMA's 128x128 tiles
     * already fill the machine for a transposed conv or one with an input activation (FFMA's wider
     * tiles apply the activation to fewer copies of A, and gather fewer ConvT weights). Devgen's
     * conv1d_units mirrors the choice. */
    auto count = [&](unsigned bm, unsigned bn) {
        unsigned total = 0;
        for (unsigned p = 0; p < phases; p++) {
            unsigned rows = a.out_rows;
            if (transpose) {
                const unsigned q_lo = a.before > p ? (a.before - p + a.stride - 1) / a.stride : 0u;
                const long long qe = ((long long)a.out_rows + a.before - p + a.stride - 1) / a.stride;
                rows = qe > (long long)q_lo ? (unsigned)(qe - q_lo) : 0u;
            }
            total += a.groups * ((a.batch * rows + bm - 1) / bm) * ((a.ng + bn - 1) / bn);
        }
        return total;
    };
    const bool tc = ((in->fj[2].u >> 13) & 1u) && ((!transpose && !a.pre) || count(SPG_BM, SPG_BN) < nblk);
    const unsigned BM = tc ? SPT_BM : SPG_BM, BN = tc ? SPT_BN : SPG_BN;
    const unsigned tn = (a.ng + BN - 1) / BN;
    const unsigned total = count(BM, BN);
    for (unsigned tile = slice; tile < total; tile += nblk) {
        unsigned rem = tile, p = 0, n_p = a.out_rows, q_lo = 0, taps = a.kernel, mt = 0;
        for (;; p++) {
            if (transpose) {
                q_lo = a.before > p ? (a.before - p + a.stride - 1) / a.stride : 0u;
                const long long qe = ((long long)a.out_rows + a.before - p + a.stride - 1) / a.stride;
                n_p = qe > (long long)q_lo ? (unsigned)(qe - q_lo) : 0u;
                taps = a.kernel > p ? (a.kernel - p + a.stride - 1) / a.stride : 0u;
            }
            mt = (a.batch * n_p + BM - 1) / BM;
            if (rem < a.groups * mt * tn) break;
            rem -= a.groups * mt * tn;
        }
        const unsigned g = rem / (mt * tn), r2 = rem - g * mt * tn;
        const unsigned c0 = g * a.cg, n0 = g * a.ng, K = taps * a.cg;
        const bool bvec = !transpose && a.kernel == 1u && !a.wf16 && a.cg % 4u == 0 && sp_aligned(a.w, 16);
        const SpRowConv1 la{a, c0, K, n_p, q_lo, p, avec, transpose, !transpose && a.kernel == 1u};
        const SpRowConvW lb{a, c0, n0, K, p, bvec, transpose};
        const SpConvEpi1 ep{a, n0, n_p, q_lo, p, transpose};
        if (tc) {
            const unsigned m0 = (r2 / tn) * BM, nn0 = (r2 % tn) * BN;
            float acc[2][2][4];
            if (sp_async(lb)) sp_tcx_tile<true>(m0, nn0, a.batch * n_p, a.ng, 0u, K, la, lb, arena, acc);
            else sp_tc_tile<true>(m0, nn0, a.batch * n_p, a.ng, 0u, K, la, lb, arena, acc);
            sp_tc_store(m0, nn0, a.batch * n_p, a.ng, ep, acc, arena);
        } else {
            sp_gemm_tile((r2 / tn) * BM, (r2 % tn) * BN, a.batch * n_p, a.ng, K, la, lb, ep, arena);
        }
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
static __device__ __noinline__ void sp_attention_f32(const PlowDevInst* in, void* const* T, unsigned slice,
                                        unsigned nblk, float* arena) {
    arena = sp_smem;
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

/* AttentionF32 on 3xTF32 tensor cores (head_width 64): FA2-style, 128 queries per block, a warp
 * owning 16 query rows; 64-key K/V tiles double-buffered by cp.async. S = Q.K^T and P.V are split
 * hi/lo per operand (FP32-accurate); P moves from the accumulator layout to the A layout with quad
 * shuffles. The online softmax is the FP32 reference's (expf, running max and sum). */
#define SPQ_BQ 128
#define SPQ_BK 64
#define SPQ_LDK 68
#define SPQ_LDV 72
static_assert(2 * SPQ_BK * (SPQ_LDK + SPQ_LDV) <= SP_ARENA_FLOATS, "tc attention stages");
__device__ __forceinline__ void sp_split(float x, unsigned& h, unsigned& l) {
    h = sp_tf32(x);
    l = sp_tf32(x - __uint_as_float(h));
}
static __device__ __noinline__ void sp_attention_tc64(const PlowDevInst* in, void* const* T, unsigned slice,
                                                      unsigned nblk, float* arena) {
    arena = sp_smem;
    constexpr int HW = 64;
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
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned qtiles = (q_rows + SPQ_BQ - 1) / SPQ_BQ;
    for (unsigned item = slice; item < batch * heads * qtiles; item += nblk) {
        const unsigned qt = item % qtiles, bh = item / qtiles, h = bh % heads, b = bh / heads;
        const unsigned q0 = qt * SPQ_BQ;
        const unsigned klen = lengths && lengths[b] < kv_rows ? lengths[b] : kv_rows;
        const unsigned kend = causal ? min(klen, q0 + SPQ_BQ) : klen;
        const unsigned ra = q0 + warp * 16u + g, rb = ra + 8u;
        unsigned qh[8][4], ql[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const float* qa = query + ((size_t)b * q_rows + ra) * stride + h * HW + ks * 8 + t;
            const float* qb = query + ((size_t)b * q_rows + rb) * stride + h * HW + ks * 8 + t;
            const float x[4] = {ra < q_rows ? qa[0] : 0.f, rb < q_rows ? qb[0] : 0.f, ra < q_rows ? qa[4] : 0.f,
                                rb < q_rows ? qb[4] : 0.f};
#pragma unroll
            for (int r = 0; r < 4; r++) sp_split(x[r], qh[ks][r], ql[ks][r]);
        }
        float o[8][4], mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int r = 0; r < 4; r++) o[n][r] = 0.f;
        auto issue = [&](unsigned kt) {
            const unsigned k0 = kt * SPQ_BK;
            if (k0 < kend) {
                float* ks = arena + (kt & 1u) * SPQ_BK * (SPQ_LDK + SPQ_LDV);
                float* vs = ks + SPQ_BK * SPQ_LDK;
#pragma unroll
                for (unsigned p = 0; p < 4; p++) {
                    const unsigned e = tid + p * PLOW_NV_THREADS, j = e >> 4, d = (e & 15u) * 4u;
                    const bool ok = k0 + j < kend;
                    const size_t base = ((size_t)b * kv_rows + k0 + j) * stride + h * HW + d;
                    sp_cp16(ks + j * SPQ_LDK + d, ok ? key + base + k_col0 : key, ok);
                    sp_cp16(vs + j * SPQ_LDV + d, ok ? value + base + v_col0 : value, ok);
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        const unsigned nkt = (kend + SPQ_BK - 1) / SPQ_BK;
        issue(0);
        for (unsigned kt = 0; kt < nkt; kt++) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            issue(kt + 1);
            const float* ks = arena + (kt & 1u) * SPQ_BK * (SPQ_LDK + SPQ_LDV);
            const float* vs = ks + SPQ_BK * SPQ_LDK;
            const unsigned k0 = kt * SPQ_BK;
            float sc[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) sc[j][r] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; kk++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float* kp = ks + (j * 8 + g) * SPQ_LDK + kk * 8 + t;
                    unsigned bh0, bl0, bh1, bl1;
                    sp_split(kp[0], bh0, bl0);
                    sp_split(kp[4], bh1, bl1);
                    sp_mma_tf32(sc[j], ql[kk], bh0, bh1);
                    sp_mma_tf32(sc[j], qh[kk], bl0, bl1);
                    sp_mma_tf32(sc[j], qh[kk], bh0, bh1);
                }
            float mt[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const unsigned kj = k0 + j * 8 + 2 * t + (r & 1), row = r < 2 ? ra : rb;
                    float v = sc[j][r] * scale;
                    if (kj < kend && (!causal || kj <= row)) {
                        if (bias && row < q_rows) v += bias[(size_t)h * bias_hs + (size_t)row * kv_rows + kj];
                    } else {
                        v = -INFINITY;
                    }
                    sc[j][r] = v;
                    mt[r >> 1] = fmaxf(mt[r >> 1], v);
                }
            float corr[2], ps[2] = {0.f, 0.f};
#pragma unroll
            for (int i = 0; i < 2; i++) {
                mt[i] = fmaxf(mt[i], __shfl_xor_sync(0xffffffffu, mt[i], 1));
                mt[i] = fmaxf(mt[i], __shfl_xor_sync(0xffffffffu, mt[i], 2));
                const float mnew = fmaxf(mrow[i], mt[i]);
                corr[i] = mnew == -INFINITY ? 1.f : expf(mrow[i] - mnew);
                mrow[i] = mnew;
            }
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const float p = sc[j][r] == -INFINITY ? 0.f : expf(sc[j][r] - mrow[r >> 1]);
                    sc[j][r] = p;
                    ps[r >> 1] += p;
                }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 1);
                ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 2);
                lrow[i] = lrow[i] * corr[i] + ps[i];
            }
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int r = 0; r < 4; r++) o[n][r] *= corr[r >> 1];
            float pv[8][4];
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int r = 0; r < 4; r++) pv[n][r] = 0.f;
            const unsigned s1 = (lane & ~3u) | (t >> 1), s2 = s1 + 2u;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                /* A(row, key t / t+4) of this key block from the quad's accumulator layout. */
                float x[4];
                {
                    const float c0a = __shfl_sync(0xffffffffu, sc[kk][0], s1), c1a = __shfl_sync(0xffffffffu, sc[kk][1], s1);
                    const float c2a = __shfl_sync(0xffffffffu, sc[kk][2], s1), c3a = __shfl_sync(0xffffffffu, sc[kk][3], s1);
                    const float c0b = __shfl_sync(0xffffffffu, sc[kk][0], s2), c1b = __shfl_sync(0xffffffffu, sc[kk][1], s2);
                    const float c2b = __shfl_sync(0xffffffffu, sc[kk][2], s2), c3b = __shfl_sync(0xffffffffu, sc[kk][3], s2);
                    x[0] = t & 1u ? c1a : c0a;
                    x[1] = t & 1u ? c3a : c2a;
                    x[2] = t & 1u ? c1b : c0b;
                    x[3] = t & 1u ? c3b : c2b;
                }
                unsigned ah[4], al[4];
#pragma unroll
                for (int r = 0; r < 4; r++) sp_split(x[r], ah[r], al[r]);
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    const float* vp = vs + (kk * 8 + t) * SPQ_LDV + n * 8 + g;
                    unsigned bh0, bl0, bh1, bl1;
                    sp_split(vp[0], bh0, bl0);
                    sp_split(vp[4 * SPQ_LDV], bh1, bl1);
                    sp_mma_tf32(pv[n], al, bh0, bh1);
                    sp_mma_tf32(pv[n], ah, bl0, bl1);
                    sp_mma_tf32(pv[n], ah, bh0, bh1);
                }
            }
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int r = 0; r < 4; r++) o[n][r] += pv[n][r];
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned r = i ? rb : ra;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int n = 0; n < 8; n++)
                *(float2*)(orow + n * 8 + 2 * t) = make_float2(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
        }
    }
}

static __device__ __noinline__ void d_attention_f32(const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk,
                                       float* arena) {
    arena = sp_smem;
    const unsigned stride = in->i[5] ? in->i[5] : in->i[3] * in->i[4];
    const bool vec = stride % 4u == 0 && in->fj[1].u % 4u == 0 && in->fj[2].u % 4u == 0 &&
                     sp_aligned(SP_TEN(1), 16) && sp_aligned(SP_TEN(2), 16) && sp_aligned(SP_TEN(3), 16) &&
                     sp_aligned(SP_TEN(0), 8);
    /* flags bit 1 (i6): 3xTF32 tensor cores. */
    if (in->i[4] == 64u && (in->i[6] & 2u) && vec) sp_attention_tc64(in, T, slice, nblk, arena);
    else if (in->i[4] == 64u) sp_attention_f32<64, 128>(in, T, slice, nblk, arena);
    else if (in->i[4] == 128u) sp_attention_f32<128, 64>(in, T, slice, nblk, arena);
}

static __device__ __noinline__ void d_speech_f32(const PlowDevInst* in, void* const* T, unsigned slice,
                                    unsigned nblk, float* arena) {
    arena = sp_smem;
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
        if (in->i[7] & 32u) {
            const SpLn ln{(const float*)SP_TEN(5), (const float*)SP_TEN(6), (const float*)SP_TEN(7), in->i[4],
                          (in->i[7] & 64u) != 0u};
            d_dense_gemm_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), SP_TEN(2), (const float*)SP_TEN(3), in->i[0],
                             in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[6], in->i[7], slice, nblk, arena,
                             (float*)SP_TEN(4), &ln);
            break;
        }
        d_dense_gemm_f32((float*)SP_TEN(0), (const float*)SP_TEN(1), SP_TEN(2), (const float*)SP_TEN(3), in->i[0],
                         in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[6], in->i[7], slice,
                         nblk, arena, (float*)SP_TEN(4), nullptr);
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
    case PLOW_DOP_ROW_STATS_F32:
        d_layernorm_f32(nullptr, (const float*)SP_TEN(1), nullptr, nullptr, in->i[0], in->i[1], in->i[2], in->fj[0].f,
                        slice, nblk, arena, (float*)SP_TEN(0));
        break;
    default: __trap(); break;
    }
}
#undef SP_TEN
