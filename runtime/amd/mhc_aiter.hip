// PLOW_GLM_HC_AITER: AITER's fused mHC decode kernels (aiter_meta/csrc/kernels/mhc_kernels.cu,
// MIT) instantiated at GLM-5.3-Flash's shape (hidden 4096, hc_mult 4, fp32 fn, bf16 residual).
// Built by scripts/build_mhc_aiter.sh against the vendor csrc tree.
#include "kernels/mhc_kernels.cu"

using PlowB = opus::bf16_t;
#define PLOW_FPP(TM)                                                                            \
    template __global__ void aiter::mhc_fused_post_pre_gemm_sqrsum_kernel<                      \
        PlowB, 4, 4, TM, 32, 32, false, false, false, false>(                                   \
        float*, float*, PlowB*, PlowB*, PlowB*, float*, float*, float*, int, int, int, int, int);
PLOW_FPP(16)
PLOW_FPP(32)
template __global__ void aiter::mhc_pre_big_fuse_rmsnorm_kernel<PlowB, 5, 4, 1, 4096, 1024, 1024, false>(
    float*, float*, PlowB*, float*, float*, float*, float*, PlowB*, PlowB*, int, int, int, int, float,
    float, float, float, float, int, int);

extern "C" __global__ void plow_mhc_aiter_1() {}
