#define PLOW_WG_WAVES 4
#include "kda_chunk_key_factor_common.h"

extern "C" __device__ unsigned plow_kda_key_factor_wu_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_wu_vgpr_le_160 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void plow_kda_chunk_key_factor_wu_gfx950(
    bf16* __restrict__ W, bf16* __restrict__ U,
    bf16* __restrict__ key_hi, bf16* __restrict__ key_lo,
    bf16* __restrict__ q, const float* __restrict__ Ainv,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    unsigned T, unsigned H, unsigned D, unsigned V, float scale) {
    if (T == 0u || H == 0u || D != 128u || V != 128u || !(scale > 0.0f)) return;
    const size_t count = (size_t)T * H * D;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned row = (unsigned)(i / ((size_t)H * D));
        const unsigned last = min((row | 63u), T - 1u);
        const size_t il = (size_t)last * H * D + i % ((size_t)H * D);
        const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
        const bf16 high = f2bf(scaled);
        key_hi[i] = high;
        key_lo[i] = f2bf(scaled - bf2f(high));
    }
    d_kda_chunk_wu_bt64(W, U, q, Ainv, k, v, g, beta, nullptr,
                        (T + 63u) / 64u, T, H, D, V, scale, true,
                        blockIdx.x, gridDim.x);
}
