/* glm_inproj.hip — several decode projections of ONE input as one split-K MFMA pass
 * (PLOW_AMD_LT_FUSE). GLM-5.3's KDA in-projection is six GEMMs of the same [M, 4096] rows
 * (q/k/v 1024, g_a/f_a 128, b 8 columns per rank): separate hipBLASLt launches leave most of the
 * GPU idle at M <= 128. Here every 16-column block of every part is one wave of one split-K slice.
 *
 * plow_glm_inproj_partial: grid (ceil(blocks/INPROJ_WAVES), S). The workgroup stages its [M, K/S] slice of x
 * in LDS once; each wave streams one 16-column block of weights (all of its K/S slice issued up
 * front) and writes f32 partials[s][m][col].
 * plow_glm_inproj_reduce: sums the S partials in s order, rounds once to bf16, scatters to the
 * parts' own [M, N] outputs.
 *
 * plow_glm_inproj_partial_f8 is the same pass over block-FP8 weights (`s` = the [N/128][K/128]
 * f32 scale grid, applied per 128-deep K block) for GLM-5.3's dense FFN layers
 * (PLOW_AMD_DENSE_NATIVE); plow_glm_dense_glu_reduce turns its gate|up partials into
 * (clamped) SwiGLU rows. */
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
extern "C" __device__ unsigned plow_glm_inproj_abi_4 = 1;
#endif

#define INPROJ_KS 512u
#define INPROJ_WAVES 8u
#define INPROJ_PARTS 6u
#define INPROJ_MAXM 128u
#define INPROJ_LDS_STRIDE (INPROJ_KS + 8u)

struct InprojPart {
    const unsigned short* w;
    unsigned short* out;
    const float* s;
    unsigned n;
    unsigned blk0;
};

struct InprojArgs {
    const unsigned short* x;
    float* partial;
    /* Explicit kernargs + the 256 hidden bytes must fit the 512-byte kernarg ring slot. */
    InprojPart parts[INPROJ_PARTS];
    unsigned m;
    unsigned k;
    unsigned nparts;
    unsigned blocks;
    /* plow_glm_dense_glu_reduce: 4 = clamped SwiGLU at `limit`, else silu. */
    unsigned act;
    float limit;
};
static_assert(sizeof(InprojArgs) == 232);

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ unsigned inproj_part(const InprojArgs& a, unsigned b) {
    unsigned p = 0;
#pragma unroll
    for (unsigned i = 1; i < INPROJ_PARTS; i++)
        if (i < a.nparts && a.parts[i].blk0 <= b) p = i;
    return p;
}
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <bool F8>
__device__ __forceinline__ void inproj_partial(const InprojArgs& a) {
    /* Dynamic LDS sized by the host to m rows, so narrow rungs keep several workgroups per CU. */
    extern __shared__ __align__(16) bf16 xs[];
    const unsigned s = blockIdx.y, kb = s * INPROJ_KS;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned r = lane & 15u, kg = lane >> 4;
    const unsigned b = blockIdx.x * INPROJ_WAVES + wave;
    const bool live = b < a.blocks;
    const unsigned p = live ? inproj_part(a, b) : 0u;
    const unsigned col = (b - a.parts[p].blk0) * 16u + r;
    const bool col_ok = live && col < a.parts[p].n;
    /* Weights first: sixteen (bf16) or eight (fp8) 16-byte loads per lane, in flight while x is
     * staged. The fp8 lane holds 16 contiguous k, so its MFMA pair reads x at the same k. */
    constexpr unsigned NW = F8 ? INPROJ_KS / 64u : INPROJ_KS / 32u;
    fp8v16 bv[NW];
    const size_t wrow = (size_t)(col_ok ? col : 0u) * a.k + kb;
    const fp8v16* wp = F8 ? (const fp8v16*)((const unsigned char*)a.parts[p].w + wrow + kg * 16u)
                          : (const fp8v16*)(a.parts[p].w + wrow + kg * 8u);
    const fp8v16 zero = {};
#pragma unroll
    for (unsigned t = 0; t < NW; t++) bv[t] = col_ok ? wp[t * 4u] : zero;
    /* x staged sixteen 16-byte loads per thread at a time, all in flight before the LDS stores. */
    const unsigned per_row = INPROJ_KS / 8u, total = a.m * per_row;
    constexpr unsigned XU = 16u;
    constexpr unsigned NT = INPROJ_WAVES * 64u;
    for (unsigned i0 = 0; i0 < total; i0 += XU * NT) {
        bf16x8 xv[XU];
#pragma unroll
        for (unsigned u = 0; u < XU; u++) {
            const unsigned i = i0 + u * NT + threadIdx.x;
            if (i < total)
                xv[u] = *(const bf16x8*)(a.x + (size_t)(i / per_row) * a.k + kb + (i % per_row) * 8u);
        }
#pragma unroll
        for (unsigned u = 0; u < XU; u++) {
            const unsigned i = i0 + u * NT + threadIdx.x;
            if (i < total) *(bf16x8*)&xs[(i / per_row) * INPROJ_LDS_STRIDE + (i % per_row) * 8u] = xv[u];
        }
    }
    __syncthreads();
    if (!live) return;
    const unsigned mb_n = (a.m + 15u) / 16u;
    f32x4 acc[INPROJ_MAXM / 16u];
#pragma unroll
    for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) acc[mb] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (!F8) {
#pragma unroll
        for (unsigned t = 0; t < INPROJ_KS / 32u; t++) {
#pragma unroll
            for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) {
                if (mb >= mb_n) break;
                const bf16x8 av = *(const bf16x8*)&xs[(mb * 16u + r) * INPROJ_LDS_STRIDE + t * 32u + kg * 8u];
                acc[mb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(av, __builtin_bit_cast(bf16x8, bv[t]),
                                                                  acc[mb], 0, 0, 0);
            }
        }
    } else {
        /* One 128-col scale row per wave (16 | 128); each 128-deep K block accumulates apart and
         * lands scaled, the block-FP8 dequantization order. */
        const float* srow = a.parts[p].s + (size_t)((b - a.parts[p].blk0) / 8u) * (a.k / 128u) + kb / 128u;
#pragma unroll
        for (unsigned q = 0; q < INPROJ_KS / 128u; q++) {
            f32x4 blk[INPROJ_MAXM / 16u];
#pragma unroll
            for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) blk[mb] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (unsigned h = 0; h < 2u; h++) {
                const unsigned t = q * 2u + h;
                bf16v8 lo, hi;
                fp8_to_bf16v8(bv[t], lo, hi);
#pragma unroll
                for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) {
                    if (mb >= mb_n) break;
                    const bf16* xr = &xs[(mb * 16u + r) * INPROJ_LDS_STRIDE + t * 64u + kg * 16u];
                    blk[mb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                        *(const bf16x8*)xr, __builtin_bit_cast(bf16x8, lo), blk[mb], 0, 0, 0);
                    blk[mb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                        *(const bf16x8*)(xr + 8), __builtin_bit_cast(bf16x8, hi), blk[mb], 0, 0, 0);
                }
            }
            const float sc = srow[q];
#pragma unroll
            for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) acc[mb] += sc * blk[mb];
        }
    }
    const unsigned ncol = a.blocks * 16u;
    float* out = a.partial + (size_t)s * a.m * ncol + b * 16u + r;
#pragma unroll
    for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) {
        if (mb >= mb_n) break;
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) {
            const unsigned row = mb * 16u + kg * 4u + i;
            if (row < a.m) out[(size_t)row * ncol] = acc[mb][i];
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(INPROJ_WAVES * 64u, 1) void plow_glm_inproj_partial(InprojArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    inproj_partial<false>(a);
#endif
}

extern "C" __global__ __launch_bounds__(INPROJ_WAVES * 64u, 1) void plow_glm_inproj_partial_f8(InprojArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    inproj_partial<true>(a);
#endif
}

extern "C" __global__ __launch_bounds__(256) void plow_glm_inproj_reduce(InprojArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned ncol = a.blocks * 16u, splits = a.k / INPROJ_KS;
    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= a.m * ncol) return;
    const unsigned row = e / ncol, c = e % ncol, b = c / 16u;
    const unsigned p = inproj_part(a, b);
    const unsigned col = c - a.parts[p].blk0 * 16u;
    if (col >= a.parts[p].n) return;
    float sum = 0.0f;
    for (unsigned s = 0; s < splits; s++) sum += a.partial[(size_t)s * a.m * ncol + e];
    a.parts[p].out[(size_t)row * a.parts[p].n + col] = f2bf(sum);
#endif
}

/* Gate part 0 and up part 1 (blk0 = n/16) of one partial pass: out0 = bf16(act(g, u)). */
extern "C" __global__ __launch_bounds__(256) void plow_glm_dense_glu_reduce(InprojArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned ncol = a.blocks * 16u, splits = a.k / INPROJ_KS, n = a.parts[0].n;
    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= a.m * n) return;
    const unsigned row = e / n, c = e % n;
    float g = 0.0f, u = 0.0f;
    for (unsigned s = 0; s < splits; s++) {
        const float* ps = a.partial + (size_t)s * a.m * ncol + (size_t)row * ncol;
        g += ps[c];
        u += ps[n + c];
    }
    if (a.act == 4u) {
        g = fminf(g, a.limit);
        u = fminf(fmaxf(u, -a.limit), a.limit);
    }
    a.parts[0].out[(size_t)row * n + c] = f2bf(g / (1.0f + __expf(-g)) * u);
#endif
}
