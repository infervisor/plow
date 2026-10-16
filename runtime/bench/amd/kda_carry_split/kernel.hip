#define PLOW_KDA_CARRY_XCD 1
#define PLOW_WG_WAVES 4
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda_carry_regstate.h"

extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void k_carry_keyfeed(bf16* out, float* state, const bf16* q, const bf16* k, const bf16* W,
                     const bf16* U, const float* Aqk, const float* g, unsigned T, unsigned H,
                     const bf16* key_hi, const bf16* key_lo) {
    __shared__ __align__(16) bf16 lds[plow_kda_regstate::LDS_BYTES / 2u];
    d_kda_chunk_carry_bt64_regstate<false, false, true>(
        out, state, q, k, W, U, Aqk, g, (T + 63u) / 64u, T, H, blockIdx.x, gridDim.x, lds,
        nullptr, key_hi, key_lo);
}

extern "C" __global__ __launch_bounds__(256)
void k_key_precompute(bf16* key_hi, bf16* key_lo, const bf16* k, const float* g, unsigned T,
                      unsigned H) {
    const size_t count = (size_t)T * H * 128u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned row = (unsigned)(i / ((size_t)H * 128u));
        const unsigned last = min((row | 63u), T - 1u);
        const size_t il = (size_t)last * H * 128u + i % ((size_t)H * 128u);
        const float scaled = bf2f(k[i]) * exp2f(g[il] - g[i]);
        const bf16 high = f2bf(scaled);
        key_hi[i] = high;
        key_lo[i] = f2bf(scaled - bf2f(high));
    }
}
