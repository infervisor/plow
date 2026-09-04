#define PLOW_WG_WAVES 8
#include "op_elementwise.h"
#include "op_kda.h"
#include "op_kda_wu_lean.h"

/* The interpreter Wu body as the prefill interpreter runs it: eight waves, one workgroup per CU. */
extern "C" __global__ __launch_bounds__(512, 1)
void k_wu_control(bf16* W, bf16* U, bf16* q, const float* Ainv, const bf16* k,
                  const bf16* v, const float* g, const float* beta, unsigned T, unsigned H,
                  float scale) {
    d_kda_chunk_wu_bt64(W, U, q, Ainv, k, v, g, beta, nullptr, (T + 63u) / 64u, T, H, 128u,
                        128u, scale, true, blockIdx.x, gridDim.x);
}

/* The kda_chunk_key_factor pair, [T][H][D] bf16 each. */
extern "C" __global__ __launch_bounds__(256)
void k_key_reference(bf16* key_hi, bf16* key_lo, const bf16* k, const float* g, unsigned T,
                     unsigned H) {
    constexpr unsigned D = 128u;
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
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_lean(bf16* W, bf16* U, bf16* q, bf16* key_hi, bf16* key_lo, const float* Ainv,
               const bf16* k, const bf16* v, const float* g, const float* beta, unsigned T,
               unsigned H, float scale) {
    __shared__ __align__(16) bf16 lds[plow_kda_wu_lean::LDS_BYTES / 2u];
    d_kda_chunk_wu_bt64_lean<false>(W, U, q, key_hi, key_lo, Ainv, k, v, g, beta,
                                    (T + 63u) / 64u, T, H, scale, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_lean_keys(bf16* W, bf16* U, bf16* q, bf16* key_hi, bf16* key_lo, const float* Ainv,
                    const bf16* k, const bf16* v, const float* g, const float* beta,
                    unsigned T, unsigned H, float scale) {
    __shared__ __align__(16) bf16 lds[plow_kda_wu_lean::LDS_BYTES / 2u];
    d_kda_chunk_wu_bt64_lean<true>(W, U, q, key_hi, key_lo, Ainv, k, v, g, beta,
                                   (T + 63u) / 64u, T, H, scale, blockIdx.x, gridDim.x, lds);
}
