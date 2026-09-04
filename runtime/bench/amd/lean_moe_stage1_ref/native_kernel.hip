#define PLOW_WG_WAVES 8
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#define MPF4_BK 256
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1

extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_abi_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_bm64_bn256_bk256_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_no_spill_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_dynamic_lds_119808 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_vgpr_le_192 = 1;

#ifndef PLOW_MOE1_INTER_DIM
#define PLOW_MOE1_INTER_DIM 384
#endif
#ifndef PLOW_MOE1_EXPERTS
#define PLOW_MOE1_EXPERTS 896
#endif

extern "C" __global__ __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void plow_moe1_mxfp4_bk256_gfx950(
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
    if (inter_dim != PLOW_MOE1_INTER_DIM || model_dim != 3584 || experts != PLOW_MOE1_EXPERTS)
        return;
    d_moe_group_pf_a4w4<true>(
        out, activation, nullptr, weight_table, weight_scale_table, meta,
        row_token, row_partidx, nullptr, out_scale, inter_dim, model_dim,
        experts, act, blockIdx.x, gridDim.x, beta, linear_beta, lds);
}
