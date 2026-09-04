#define PLOW_WG_WAVES 4
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_carry_key_precompute_abi_1 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_control(bf16* W, bf16* U, bf16* q, const float* Ainv,
                  const bf16* k, const bf16* v, const float* g,
                  const float* beta, unsigned T, unsigned H, float scale) {
    d_kda_chunk_wu_bt64(W, U, q, Ainv, k, v, g, beta, nullptr,
                        (T + 63u) / 64u, T, H, 128, 128, scale, true,
                        blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_key_factors(bf16* W, bf16* U, bf16* key_hi, bf16* key_lo, bf16* q,
                      const float* Ainv, const bf16* k, const bf16* v,
                      const float* g, const float* beta, unsigned T, unsigned H,
                      float scale) {
    constexpr unsigned D = 128;
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
                        (T + 63u) / 64u, T, H, D, D, scale, true,
                        blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_carry_control(bf16* out, float* state, const bf16* q, const bf16* k,
                     const bf16* W, const bf16* U, const float* Aqk, const float* g,
                     unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + 16u * 128u);
    float* osm = reinterpret_cast<float*>(vsm + 64u * 16u);
    d_kda_chunk_carry_bt64<true, false>(out, state, q, k, W, U, Aqk, g, nullptr,
        (T + 63u) / 64u, T, H, 128, 128, 1.0f, blockIdx.x, gridDim.x,
        st, vsm, osm);
}

extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_carry_precomputed(bf16* out, float* state, const bf16* q, const bf16* k,
                         const bf16* key_hi, const bf16* key_lo, const bf16* W,
                         const bf16* U, const float* Aqk, const float* g,
                         unsigned T, unsigned H) {
    extern __shared__ __align__(16) unsigned char shared[];
    float* st = reinterpret_cast<float*>(shared);
    bf16* vsm = reinterpret_cast<bf16*>(st + 16u * 128u);
    float* osm = reinterpret_cast<float*>(vsm + 64u * 16u);
    d_kda_chunk_carry_bt64<true, true>(out, state, q, k, W, U, Aqk, g, nullptr,
        (T + 63u) / 64u, T, H, 128, 128, 1.0f, blockIdx.x, gridDim.x,
        st, vsm, osm, key_hi, key_lo);
}
