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

// The activation half of `activation_scheme: dynamic`: the arithmetic of `plow_moe_aiter_prepare`
// (scale = amax/240, e4m3 FNUZ, scales group-major `[K/128][rows]`, the transposed layout AITER's
// block-scale GEMM reads) with K as a parameter. Sixteen lanes own one 128-element group, eight
// bf16 (16 bytes) each, so a wave covers four groups with full-width loads; the group max is the
// same value in any reduction order, so the bytes and scales are those of the one-group-per-wave
// loop it replaces, which ran at 0.8-1.4 TB/s.
extern "C" __global__ void plow_gemm_blk_quant(GemmBlkQuantArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned part = threadIdx.x % 16u;
    const unsigned groups = a.k / 128u;
    for (unsigned group = first / 16u; group < a.rows * groups; group += stride / 16u) {
        const unsigned row = group / groups, col = group % groups;
        const unsigned pos = group * 128u + part * 8u;
        const uint4 packed = *(const uint4*)(a.x + pos);
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
        float v[8];
        float amax = 1e-10f;
        for (unsigned i = 0; i < 4u; ++i) {
            v[2 * i] = __uint_as_float(words[i] << 16);
            v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
        for (unsigned offset = 8u; offset > 0u; offset /= 2u)
            amax = fmaxf(amax, __shfl_xor(amax, offset, 16));
        const float scale = amax * (1.0f / 240.0f);
        const float inverse = 1.0f / scale;
        unsigned out[2];
        for (unsigned i = 0; i < 2u; ++i) {
            const unsigned lo = __builtin_amdgcn_cvt_pk_fp8_f32(
                v[4 * i] * inverse, v[4 * i + 1] * inverse, 0, false);
            out[i] = __builtin_amdgcn_cvt_pk_fp8_f32(
                v[4 * i + 2] * inverse, v[4 * i + 3] * inverse, lo, true);
        }
        *(uint2*)((unsigned char*)a.q + pos) = make_uint2(out[0], out[1]);
        if (part == 0) a.qs[col * a.rows + row] = scale;
    }
#endif
}
