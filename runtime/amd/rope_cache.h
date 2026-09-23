#pragma once
#include "amd_common.h"

__device__ inline void d_glm_rope_cache_bf16(float* cosb, float* sinb, unsigned ctx, float theta) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    const unsigned j = first % 32u;
    const float inverse = 1.0f / powf(theta, (float)j / 32.0f);
    for (unsigned i = first; i < ctx * 32u; i += stride) {
        const float angle = (float)(i / 32u) * inverse;
        cosb[i] = bf2f(f2bf(cosf(angle)));
        sinb[i] = bf2f(f2bf(sinf(angle)));
    }
}
