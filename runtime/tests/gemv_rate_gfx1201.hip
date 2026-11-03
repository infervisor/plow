/* Build/run instructions are in gemv_rate_gfx1201.c. */
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_norm.h"
#include "op_gemm.h"

static_assert(PLOW_THREADS == 512, "the decode benchmark uses 64 workgroups of 512 threads");

extern "C" __global__ void gemv_rate_capacity(unsigned* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = PLOW_GEMV_MM;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemv_rate_old(bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K) {
    __shared__ __align__(16) bf16 lds[GM_LDS_HALVES];
#if PLOW_GFX12
    d_gemv_legacy(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, lds);
#else
    d_gemv(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemv_rate_new(bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K) {
    __shared__ __align__(16) bf16 lds[GM_LDS_HALVES];
    d_gemv(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemv_rate_argmax(bf16* C, const bf16* x, const bf16* W, unsigned long long* part,
                     unsigned M, unsigned N, unsigned K) {
    __shared__ __align__(16) bf16 lds[GM_LDS_HALVES];
    d_gemv_argmax(C, x, W, part, M, N, K, 0.0f, blockIdx.x, gridDim.x, lds);
}

/* Reuse the calibrated stream reader to evict prior GEMV weights before each
 * profiled dispatch. The eviction dispatch is outside the measured interval. */
#include "decode_bw_probe.hip"
