#include "rope_cache.h"

extern "C" __global__ void glm_rope_cache_bf16(float* cosb, float* sinb, unsigned ctx, float theta) {
    d_glm_rope_cache_bf16(cosb, sinb, ctx, theta);
}

extern "C" __device__ unsigned plow_rope_cache_bf16_abi_1 = 1;
