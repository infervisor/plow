#include <hip/hip_runtime.h>

extern "C" __global__ void glm_rope_indexer_cache_bf16(
    const float* cosb, const float* sinb, float* icos, float* isin, unsigned ctx) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < ctx * 64;
         i += blockDim.x * gridDim.x) {
        unsigned j = i % 64;
        unsigned src = (i / 64) * 32 + j;
        icos[i] = j < 32 ? cosb[src] : 1.0f;
        isin[i] = j < 32 ? sinb[src] : 0.0f;
    }
}

extern "C" __device__ unsigned plow_rope_indexer_cache_bf16_abi_1 = 1;
