#define PLOW_WG_WAVES 4
#include "op_elementwise.h"
#include "op_kda.h"

/* The rejected standalone key-factor Wu object body: the interpreter Wu at four waves plus the
 * key-factor pair pass (kda_chunk_key_factor_wu.hip). */
extern "C" __global__ __launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(2, 8)))
void k_wu_key_factor_object(bf16* W, bf16* U, bf16* key_hi, bf16* key_lo, bf16* q,
                            const float* Ainv, const bf16* k, const bf16* v, const float* g,
                            const float* beta, unsigned T, unsigned H, float scale) {
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
    d_kda_chunk_wu_bt64(W, U, q, Ainv, k, v, g, beta, nullptr, (T + 63u) / 64u, T, H, D, D,
                        scale, true, blockIdx.x, gridDim.x);
}
