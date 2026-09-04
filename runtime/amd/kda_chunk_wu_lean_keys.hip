#include "kda_chunk_wu_lean_common.h"

extern "C" __device__ unsigned plow_kda_wu_lean_keys_1 = 1;
extern "C" __device__ unsigned plow_kda_wu_lean_scratch_pair_bf16_1 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void plow_kda_chunk_wu_lean_keys_gfx950(
    bf16* __restrict__ W, bf16* __restrict__ U, bf16* __restrict__ q,
    bf16* __restrict__ key_hi, bf16* __restrict__ key_lo, const float* __restrict__ Ainv,
    const bf16* __restrict__ k, const bf16* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, unsigned t, unsigned heads, unsigned dim,
    unsigned value_dim, float scale) {
    plow_kda_chunk_wu_lean_body<true>(W, U, q, key_hi, key_lo, Ainv, k, v, g, beta, t, heads,
                                      dim, value_dim, scale);
}
