#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

extern "C" __global__ void plow_glm_fold_normalize(float* out, const float* partial, const float* ml,
                                      unsigned rows, unsigned heads, unsigned splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned d = threadIdx.x + (blockIdx.x % 2) * 256;
    const unsigned row = blockIdx.x / 2;
    const unsigned head = row % heads, token = row / heads;
    const float* m = ml + (size_t)row * splits * 2;
    float gm;
    const float gl = fa_merge_ml(m, splits, gm);
    const float inv = gl > 0.0f ? FA_RECIP(gl) : 0.0f;
    float acc = 0.0f;
    for (unsigned s = 0; s < splits; ++s) {
        const float pv = partial[((size_t)row * splits + s) * 512 + d];
        acc += m[s * 2] == FA_NEG_INF ? 0.0f : pv * FA_EXP(m[s * 2] - gm);
    }
    out[((size_t)head * rows + token) * 512 + d] = acc * inv;
#endif
}

extern "C" __global__ void plow_glm_fold_convert(unsigned short* out, const float* in,
                                    unsigned rows, unsigned heads) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned row = blockIdx.x, d = threadIdx.x;
    const unsigned head = row % heads, token = row / heads;
    out[(size_t)row * 256 + d] = f2bf(in[((size_t)head * rows + token) * 256 + d]);
#endif
}


extern "C" __global__ void plow_glm_fold_weight(float* out, const unsigned short* in) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    out[i] = bf2f(in[i]);
#endif
}
#if defined(__HIP_DEVICE_COMPILE__)
extern "C" __device__ unsigned plow_glm_fold_adapter_abi_1 = 1;
#endif
