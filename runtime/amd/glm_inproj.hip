/* glm_inproj.hip — several decode projections of ONE input as one split-K MFMA pass
 * (PLOW_AMD_LT_FUSE). GLM-5.3's KDA in-projection is six GEMMs of the same [M, 4096] rows
 * (q/k/v 1024, g_a/f_a 128, b 8 columns per rank): separate hipBLASLt launches leave most of the
 * GPU idle at M <= 128. Here every 16-column block of every part is one wave of one split-K slice.
 *
 * plow_glm_inproj_partial: grid (ceil(blocks/4), S). The workgroup stages its [M, K/S] slice of x
 * in LDS once; each wave streams one 16-column block of weights (all of its K/S slice issued up
 * front) and writes f32 partials[s][m][col].
 * plow_glm_inproj_reduce: sums the S partials in s order, rounds once to bf16, scatters to the
 * parts' own [M, N] outputs. */
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
extern "C" __device__ unsigned plow_glm_inproj_abi_1 = 1;
#endif

#define INPROJ_KS 512u
#define INPROJ_MAXM 128u
#define INPROJ_LDS_STRIDE (INPROJ_KS + 8u)

struct InprojPart {
    const unsigned short* w;
    unsigned short* out;
    unsigned n;
    unsigned blk0;
};

struct InprojArgs {
    const unsigned short* x;
    float* partial;
    InprojPart parts[8];
    unsigned m;
    unsigned k;
    unsigned nparts;
    unsigned blocks;
};
static_assert(sizeof(InprojArgs) == 224);

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ unsigned inproj_part(const InprojArgs& a, unsigned b) {
    unsigned p = 0;
#pragma unroll
    for (unsigned i = 1; i < 8u; i++)
        if (i < a.nparts && a.parts[i].blk0 <= b) p = i;
    return p;
}
#endif

extern "C" __global__ __launch_bounds__(256, 1) void plow_glm_inproj_partial(InprojArgs a) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    /* Dynamic LDS sized by the host to m rows, so narrow rungs keep several workgroups per CU. */
    extern __shared__ __align__(16) bf16 xs[];
    const unsigned s = blockIdx.y, kb = s * INPROJ_KS;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned r = lane & 15u, kg = lane >> 4;
    const unsigned b = blockIdx.x * 4u + wave;
    const bool live = b < a.blocks;
    const unsigned p = live ? inproj_part(a, b) : 0u;
    const unsigned col = (b - a.parts[p].blk0) * 16u + r;
    const bool col_ok = live && col < a.parts[p].n;
    /* Weights first: sixteen 16-byte loads per lane, in flight while x is staged. */
    const bf16x8* wp = (const bf16x8*)(a.parts[p].w + (size_t)(col_ok ? col : 0u) * a.k + kb + kg * 8u);
    bf16x8 bv[INPROJ_KS / 32u];
    const bf16x8 zero = {};
#pragma unroll
    for (unsigned t = 0; t < INPROJ_KS / 32u; t++) bv[t] = col_ok ? wp[t * 4u] : zero;
    /* x staged sixteen 16-byte loads per thread at a time, all in flight before the LDS stores. */
    const unsigned per_row = INPROJ_KS / 8u, total = a.m * per_row;
    constexpr unsigned XU = 16u;
    for (unsigned i0 = 0; i0 < total; i0 += XU * 256u) {
        bf16x8 xv[XU];
#pragma unroll
        for (unsigned u = 0; u < XU; u++) {
            const unsigned i = i0 + u * 256u + threadIdx.x;
            if (i < total)
                xv[u] = *(const bf16x8*)(a.x + (size_t)(i / per_row) * a.k + kb + (i % per_row) * 8u);
        }
#pragma unroll
        for (unsigned u = 0; u < XU; u++) {
            const unsigned i = i0 + u * 256u + threadIdx.x;
            if (i < total) *(bf16x8*)&xs[(i / per_row) * INPROJ_LDS_STRIDE + (i % per_row) * 8u] = xv[u];
        }
    }
    __syncthreads();
    if (!live) return;
    const unsigned mb_n = (a.m + 15u) / 16u;
    f32x4 acc[INPROJ_MAXM / 16u];
#pragma unroll
    for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) acc[mb] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (unsigned t = 0; t < INPROJ_KS / 32u; t++) {
#pragma unroll
        for (unsigned mb = 0; mb < INPROJ_MAXM / 16u; mb++) {
            if (mb >= mb_n) break;
            const bf16x8 av = *(const bf16x8*)&xs[(mb * 16u + r) * INPROJ_LDS_STRIDE + t * 32u + kg * 8u];
            acc[mb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(av, bv[t], acc[mb], 0, 0, 0);
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
