#define PLOW_WG_WAVES 8
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_intra_wave_items_abi_1 = 1;

extern "C" __global__ __launch_bounds__(512, 1)
void k_intra_control(float* aqk, float* ainv, const bf16* q, const bf16* k,
                     const float* g, const float* beta, unsigned t, unsigned heads,
                     float scale) {
    __shared__ __align__(16) float mat[64u * 64u];
    d_kda_chunk_intra_bt64(aqk, ainv, q, k, g, beta, nullptr, (t + 63u) / 64u,
                           t, heads, 128u, scale, blockIdx.x, gridDim.x, mat);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_intra_wave_items(float* aqk, float* ainv, const bf16* q, const bf16* k,
                        const float* g, const float* beta, unsigned t, unsigned heads,
                        float scale) {
    __shared__ __align__(16) float matrices[8u * 64u * 64u];
    d_kda_chunk_intra_bt64_wave_items(
        aqk, ainv, q, k, g, beta, nullptr, (t + 63u) / 64u, t, heads, 128u,
        scale, blockIdx.x, gridDim.x, matrices);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_wu_qpre(bf16* w, bf16* u, bf16* q, const float* ainv, const bf16* k,
               const bf16* v, const float* g, const float* beta, unsigned t,
               unsigned heads, float scale) {
    d_kda_chunk_wu_bt64(w, u, q, ainv, k, v, g, beta, nullptr,
                        (t + 63u) / 64u, t, heads, 128u, 128u, scale, true,
                        blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(512, 1)
void k_carry_qpre(bf16* out, float* state, const bf16* q, const bf16* k,
                  const bf16* w, const bf16* u, const float* aqk, const float* g,
                  unsigned t, unsigned heads, float scale) {
    __shared__ __align__(16) float st[16u * 128u];
    __shared__ __align__(16) bf16 vsm[64u * 16u];
    __shared__ __align__(16) float osm[64u * 16u];
    d_kda_chunk_carry_bt64<true>(
        out, state, q, k, w, u, aqk, g, nullptr, (t + 63u) / 64u, t, heads,
        128u, 128u, scale, blockIdx.x, gridDim.x, st, vsm, osm);
}
