#ifndef PLOW_STAGE1_WG
#define PLOW_STAGE1_WG 512
#endif
#ifndef PLOW_STAGE1_MIN_OCC
#define PLOW_STAGE1_MIN_OCC 2
#endif
#define PLOW_WG_WAVES (PLOW_STAGE1_WG / 64)
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#define PLOW_MOE_PF_A4W4_DIRECT_ONLY 1
#ifndef MPF4_BM
#define MPF4_BM 64
#endif
#ifndef MPF4_BN
#define MPF4_BN 256
#endif
#ifndef MPF4_BK
#define MPF4_BK 256
#endif
#ifndef PLOW_MOE_PF_XCD_WGM
#define PLOW_MOE_PF_XCD_WGM 0
#endif
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1

extern "C" __device__ unsigned plow_moe1_schedule_abi_1 = 1;
extern "C" __device__ unsigned plow_moe1_schedule_shape_parameterized_1 = 1;
extern "C" __device__ unsigned plow_moe1_schedule_wave64_1 = 1;

extern "C" __global__ __launch_bounds__(PLOW_STAGE1_WG, PLOW_STAGE1_MIN_OCC)
__attribute__((amdgpu_waves_per_eu(PLOW_STAGE1_MIN_OCC, PLOW_STAGE1_MIN_OCC)))
void plow_moe1_mxfp4_schedule_gfx950(
    unsigned char* out,
    const bf16* activation,
    const unsigned long long* weight_table,
    const unsigned long long* weight_scale_table,
    const int* meta,
    const unsigned* row_token,
    const unsigned* row_partidx,
    unsigned char* out_scale,
    unsigned inter_dim,
    unsigned model_dim,
    unsigned experts,
    unsigned act,
    float beta,
    float linear_beta,
    unsigned reserved,
    unsigned reserved2) {
    (void)reserved;
    (void)reserved2;
    extern __shared__ unsigned char lds[];
    if (inter_dim != 384 || model_dim != 3584 || experts != 896)
        return;
    d_moe_group_pf_a4w4<true>(
        out, activation, nullptr, weight_table, weight_scale_table, meta,
        row_token, row_partidx, nullptr, out_scale, inter_dim, model_dim,
        experts, act, blockIdx.x, gridDim.x, beta, linear_beta, lds);
}
