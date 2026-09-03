#define PLOW_WG_WAVES 4
#include "amd_common.h"

extern "C" __device__ unsigned plow_moe_combine_fixed_order_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_combine_fixed_order_slots16_1 = 1;
extern "C" __device__ unsigned plow_moe_combine_materialized_f32_1 = 1;
extern "C" __device__ unsigned plow_moe_combine_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe_combine_no_spill_1 = 1;

extern "C" __global__ __launch_bounds__(256, 4)
__attribute__((amdgpu_waves_per_eu(4, 8)))
void plow_moe_combine_fixed_order_gfx950(
    bf16* __restrict__ out,
    const bf16* __restrict__ residual,
    const bf16* __restrict__ shared,
    const float* __restrict__ part,
    unsigned hidden,
    unsigned topk,
    unsigned tokens) {
    if (topk != 16)
        return;

    for (unsigned token = blockIdx.x; token < tokens; token += gridDim.x) {
        const size_t output_base = (size_t)token * hidden;
        const float* token_part = part + (size_t)token * 16u * hidden;
        for (unsigned h = threadIdx.x; h < hidden; h += blockDim.x) {
            float acc = residual ? bf2f(residual[output_base + h]) : 0.0f;
            if (shared)
                acc += bf2f(shared[output_base + h]);
#pragma unroll
            for (unsigned slot = 0; slot < 16; ++slot)
                acc += token_part[(size_t)slot * hidden + h];
            st_act<bf16>(out + output_base + h, f2bf(acc));
        }
    }
}
