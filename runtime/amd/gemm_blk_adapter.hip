#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_common.h"
#endif

// ABI marker checked by `exec::amd_gemm_blk` at load.
extern "C" __device__ unsigned plow_gemm_blk_abi_1 = 1;

struct GemmBlkQuantArgs {
    unsigned short* q;
    float* qs;
    const unsigned short* x;
    unsigned rows;
    unsigned k;
};
static_assert(sizeof(GemmBlkQuantArgs) == 32);

// The activation half of `activation_scheme: dynamic`, lifted from `plow_moe_aiter_prepare`
// with K made a parameter: one wave64 per 128-element group, scale = amax/240 (e4m3 FNUZ),
// scales stored group-major `[K/128][rows]` — the transposed layout AITER's block-scale GEMM reads.
extern "C" __global__ void plow_gemm_blk_quant(GemmBlkQuantArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned lane = threadIdx.x % 64u;
    const unsigned groups = a.k / 128u;
    for (unsigned group = first / 64u; group < a.rows * groups; group += stride / 64u) {
        const unsigned row = group / groups, col = group % groups;
        const unsigned pos = group * 128u + lane * 2u;
        const float x0 = bf2f(a.x[pos]), x1 = bf2f(a.x[pos + 1u]);
        const float amax = wave_max(fmaxf(1e-10f, fmaxf(fabsf(x0), fabsf(x1))));
        const float scale = amax * (1.0f / 240.0f);
        const float inverse = 1.0f / scale;
        a.q[pos / 2u] = (unsigned short)__builtin_amdgcn_cvt_pk_fp8_f32(
            x0 * inverse, x1 * inverse, 0, false);
        if (lane == 0) a.qs[col * a.rows + row] = scale;
    }
#endif
}
