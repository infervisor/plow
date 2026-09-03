#define PLOW_WG_WAVES 8
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_intra_cached_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_cached_bt64_d128_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_cached_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_cached_no_spill_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_cached_static_lds_114688 = 1;
extern "C" __device__ unsigned plow_kda_intra_cached_vgpr_le_96 = 1;

extern "C" __global__ __launch_bounds__(512, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void plow_kda_chunk_intra_cached_gfx950(
    float* __restrict__ aqk, float* __restrict__ ainv,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const float* __restrict__ g_prefix, const float* __restrict__ beta,
    unsigned t, unsigned heads, unsigned dim, float scale) {
    __shared__ __align__(16) bf16 scratch[57344];
    if (t < 512u || heads == 0u || dim != 128u || !(scale > 0.0f)) return;
    d_kda_chunk_intra_bt64_cached(
        aqk, ainv, q, k, g_prefix, beta, nullptr, (t + 63u) / 64u,
        t, heads, dim, scale, blockIdx.x, gridDim.x, scratch);
}
