#include "amd_common.h"
#include "op_elementwise.h"
#include "op_moe.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_router_topk(unsigned char* table, const bf16* logits, const float* bias,
                    unsigned n_exp, unsigned topk, unsigned flags, float route_scale) {
    __shared__ float scratch[8192];
    d_moe_router_topk(table, logits, bias, n_exp, topk, flags, route_scale,
                      blockIdx.x, gridDim.x, scratch, 1, 1);
}
