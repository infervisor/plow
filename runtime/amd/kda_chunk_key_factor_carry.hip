#define PLOW_WG_WAVES 4
#include "kda_chunk_key_factor_common.h"

extern "C" __device__ unsigned plow_kda_key_factor_carry_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_carry_vgpr_le_160 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_carry_lds_14336 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void plow_kda_chunk_key_factor_carry_gfx950(
    bf16* __restrict__ out, float* __restrict__ state,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ key_hi, const bf16* __restrict__ key_lo,
    const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g,
    unsigned T, unsigned H, unsigned D, unsigned V, float scale) {
    __shared__ __align__(16) float st[16u * 128u];
    __shared__ __align__(16) bf16 vsm[64u * 16u];
    __shared__ __align__(16) float osm[64u * 16u];
    if (T == 0u || H == 0u || D != 128u || V != 128u || !(scale > 0.0f)) return;
    d_kda_chunk_carry_bt64<true, true>(
        out, state, q, k, W, U, Aqk, g, nullptr, (T + 63u) / 64u,
        T, H, D, V, scale, blockIdx.x, gridDim.x, st, vsm, osm, key_hi, key_lo);
}
