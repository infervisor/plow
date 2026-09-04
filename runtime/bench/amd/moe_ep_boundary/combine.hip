#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

struct RouteEntry {
    unsigned expert;
    float gate;
};

extern "C" __device__ unsigned plow_moe_ep_combine_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_combine_fixed_slot_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_combine_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_combine_no_spill_1 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe_ep_combine_gfx950(
    __hip_bfloat16* __restrict__ out,
    const float* __restrict__ part,
    const RouteEntry* __restrict__ routes,
    unsigned tokens,
    unsigned hidden,
    unsigned topk,
    unsigned expert_begin,
    unsigned expert_end) {
    const unsigned col_tiles = (hidden + blockDim.x - 1u) / blockDim.x;
    const unsigned token = blockIdx.x / col_tiles;
    const unsigned col = (blockIdx.x - token * col_tiles) * blockDim.x + threadIdx.x;
    if (token < tokens && col < hidden) {
        float acc = 0.0f;
#pragma unroll 16
        for (unsigned slot = 0; slot < topk; ++slot) {
            const size_t p = static_cast<size_t>(token) * topk + slot;
            const unsigned expert = routes[p].expert;
            if (expert >= expert_begin && expert < expert_end)
                acc += part[p * hidden + col];
        }
        out[static_cast<size_t>(token) * hidden + col] = static_cast<__hip_bfloat16>(acc);
    }
}
