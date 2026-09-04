#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

#define PLOW_WG_WAVES 8
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_intra_wave_items_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_wave_items_bt64_d128_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_wave_items_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_wave_items_no_spill_1 = 1;
extern "C" __device__ unsigned plow_kda_intra_wave_items_static_lds_131072 = 1;
extern "C" __device__ unsigned plow_kda_intra_wave_items_vgpr_le_96 = 1;

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

extern "C" __global__ __launch_bounds__(512, 1)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void plow_kda_chunk_intra_wave_items_gfx950(
    float* __restrict__ aqk, float* __restrict__ ainv,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const float* __restrict__ g_prefix, const float* __restrict__ beta,
    unsigned t, unsigned heads, unsigned dim, float scale) {
    __shared__ __align__(16) float matrices[8u * 64u * 64u];
    if (t < 512u || heads == 0u || dim != 128u || !(scale > 0.0f)) return;
    d_kda_chunk_intra_bt64_wave_items(
        aqk, ainv, q, k, g_prefix, beta, nullptr, (t + 63u) / 64u,
        t, heads, dim, scale, blockIdx.x, gridDim.x, matrices);
}
