#define PLOW_WG_WAVES 8
#include "amd_common.h"
#include "op_moe.h"

extern "C" __device__ unsigned plow_moe_decode_grouped_mxfp4_abi_1 = 1;

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2)
void plow_moe_decode_grouped_glu_mxfp4_gfx950(
    bf16* fu, const bf16* x, const unsigned char* table,
    const unsigned long long* weights, const unsigned long long* scales,
    unsigned topk, unsigned intermediate, unsigned hidden, unsigned experts,
    unsigned act, unsigned enc, float beta, float linear_beta) {
    d_moe_group_glu_fp8_blk(fu, x, table, weights, scales, topk, intermediate, hidden,
                            experts, act, blockIdx.x, gridDim.x, enc, beta, linear_beta);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2)
void plow_moe_decode_grouped_down_mxfp4_gfx950(
    float* partial, const bf16* fu, const unsigned char* table,
    const unsigned long long* weights, const unsigned long long* scales,
    unsigned topk, unsigned hidden, unsigned intermediate, unsigned experts,
    unsigned enc, unsigned) {
    d_moe_group_down_fp8_blk(partial, fu, table, weights, scales, topk, hidden,
                             intermediate, experts, blockIdx.x, gridDim.x, enc);
}
