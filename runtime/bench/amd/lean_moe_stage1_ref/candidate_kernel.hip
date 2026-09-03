#define PLOW_WG_WAVES 4
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#define MPF4_BM 64
#define MPF4_BN 128
#define MPF4_BK 256
#define MPF4_WNc 2
#define PLOW_MOE_PF_XCD_WGM 4
#define PLOW_MOE_PF_SITU_ONLY 1
#define PLOW_MOE_PF_A4W4_LOWREG 1
#define PLOW_MOE_PF_A4W4_BRIDGE_ALIAS 1
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1

extern "C" __device__ unsigned plow_moe1_candidate_abi_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_bm64_bn128_bk256_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_wave64_wg256_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_xcd8_wgm4_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_generic_a4w4_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_situ_pair_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_sorted_mxfp4_scale_1 = 1;
extern "C" __device__ unsigned plow_moe1_candidate_dynamic_lds_52224 = 1;

extern "C" __global__ __launch_bounds__(256, 3)
__attribute__((amdgpu_waves_per_eu(3, 3)))
void plow_moe1_mxfp4_bm64_bn128_bk256_xcd8_wgm4_gfx950(
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
    if (inter_dim != 384 || model_dim != 3584 || experts != 896 ||
        act != PLOW_MOE_ACT_SITU)
        return;
    extern __shared__ unsigned char lds[];
    d_moe_group_pf_a4w4<true>(
        out, activation, nullptr, weight_table, weight_scale_table, meta,
        row_token, row_partidx, nullptr, out_scale, inter_dim, model_dim,
        experts, act, blockIdx.x, gridDim.x, beta, linear_beta, lds);
}
