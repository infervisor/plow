#define PLOW_WG_WAVES 4
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1

extern "C" __global__ __launch_bounds__(256)
void plow_moe_combine_interpreter_order_gfx950(
    bf16* out,
    const bf16* residual,
    const bf16* shared,
    const float* part,
    unsigned hidden,
    unsigned topk,
    unsigned tokens) {
    d_moe_combine_pf(out, residual, shared, part, hidden, topk, tokens,
                     blockIdx.x, gridDim.x);
}
