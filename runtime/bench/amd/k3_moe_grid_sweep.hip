#include "amd_common.h"
#include "op_moe.h"

#ifndef K3_MOE_GRID_XCD
#define K3_MOE_GRID_XCD 1
#endif

__device__ __forceinline__ unsigned k3_moe_slice() {
    if (K3_MOE_GRID_XCD == 1 || gridDim.x % K3_MOE_GRID_XCD != 0) return blockIdx.x;
    return blockIdx.x % K3_MOE_GRID_XCD * (gridDim.x / K3_MOE_GRID_XCD) +
           blockIdx.x / K3_MOE_GRID_XCD;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_group_glu(bf16* fu, const bf16* x, const unsigned char* table,
                      const unsigned long long* weights,
                      const unsigned long long* scales, unsigned topk,
                      unsigned intermediate, unsigned hidden, unsigned experts) {
    d_moe_group_glu_fp8_blk(fu, x, table, weights, scales, topk, intermediate, hidden,
                            experts, PLOW_MOE_ACT_SITU, k3_moe_slice(), gridDim.x,
                            PLOW_MOE_ENC_MXFP4, 4.0f, 25.0f);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_group_down(float* partial, const bf16* fu, const unsigned char* table,
                       const unsigned long long* weights,
                       const unsigned long long* scales, unsigned topk,
                       unsigned hidden, unsigned intermediate, unsigned experts) {
    d_moe_group_down_fp8_blk(partial, fu, table, weights, scales, topk, hidden,
                             intermediate, experts, k3_moe_slice(), gridDim.x,
                             PLOW_MOE_ENC_MXFP4);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_stream(float* sink, const unsigned char* arena, unsigned long long bytes) {
    const unsigned long long thread =
        static_cast<unsigned long long>(blockIdx.x) * PLOW_THREADS + threadIdx.x;
    const unsigned long long threads =
        static_cast<unsigned long long>(gridDim.x) * PLOW_THREADS;
    const PLOW_GLOB fp4v32* words =
        (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)arena;
    float sum = 0.0f;
    for (unsigned long long i = thread; i < bytes / sizeof(fp4v32); i += threads) {
        const fp4v32 value = words[i];
        sum += static_cast<float>(value[0] ^ value[1] ^ value[2] ^ value[3]);
    }
    if (sum == 1234.5f) sink[0] = sum;
}
