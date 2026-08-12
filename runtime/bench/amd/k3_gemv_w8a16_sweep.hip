/* Exact K3 B1 decode grids: shipping BF16 GEMV vs block-scaled E4M3 W8A16. */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

typedef union __align__(16) {
    bf16 xstage[8192];
    float part[PLOW_WAVES];
} k3_w8_smem;

#define STAGE_X(SM, X, K)                                                                          \
    for (unsigned i = threadIdx.x; i < (K); i += PLOW_THREADS) (SM).xstage[i] = (X)[i]

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_bf16(
    bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K) {
    __shared__ k3_w8_smem sm;
    for (unsigned e = 0; e < nrep; ++e) {
        STAGE_X(sm, x, K);
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(
            C, x, W + (size_t)e * N * K, nullptr, nullptr, 1u, N, K, blockIdx.x, gridDim.x,
            sm.xstage);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_bf16_aa(
    bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K) {
    __shared__ k3_w8_smem sm;
    for (unsigned e = 0; e < nrep; ++e) {
        STAGE_X(sm, x, K);
        __syncthreads();
        gemv_rows<PLOW_GEMV_MM, true, false, GV_UNROLL>(
            C, x, W + (size_t)e * N * K, nullptr, nullptr, 1u, N, K, blockIdx.x, gridDim.x,
            sm.xstage);
        __syncthreads();
    }
}

#define W8_DISPATCH(UN)                                                                            \
    gemv_rows_fp8_blk<PLOW_GEMV_MM, true, UN>(                                                     \
        C, x, Wp, scale, 1u, N, K, blockIdx.x, gridDim.x, sm.xstage)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_w8a16(
    bf16* C, const bf16* x, const unsigned char* W, const float* scale, unsigned nrep, unsigned N,
    unsigned K) {
    __shared__ k3_w8_smem sm;
    const unsigned nchunk = (K + 1023u) >> 10;
    for (unsigned e = 0; e < nrep; ++e) {
        STAGE_X(sm, x, K);
        __syncthreads();
        const unsigned char* Wp = W + (size_t)e * N * K;
        if (nchunk >= 16u)
            W8_DISPATCH(8);
        else if (nchunk >= 11u)
            W8_DISPATCH(6);
        else if (nchunk <= 1u)
            W8_DISPATCH(1);
        else if (nchunk == 2u)
            W8_DISPATCH(2);
        else if (nchunk == 4u)
            W8_DISPATCH(4);
        else
            W8_DISPATCH(3);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_w8a16_row(
    bf16* C, const bf16* x, const unsigned char* W, const float* scale, unsigned nrep, unsigned N,
    unsigned K) {
    __shared__ k3_w8_smem sm;
    for (unsigned e = 0; e < nrep; ++e) {
        STAGE_X(sm, x, K);
        __syncthreads();
        gemv_rows_fp8<PLOW_GEMV_MM, true, GV_UNROLL_FP8>(
            C, nullptr, nullptr, x, W + (size_t)e * N * K, nullptr, nullptr, scale, nullptr,
            nullptr, 1u, N, 0u, 0u, K, blockIdx.x, gridDim.x, sm.xstage);
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_mxfp4(
    bf16* C, const bf16* x, const unsigned char* W, const unsigned char* scale, unsigned nrep,
    unsigned N, unsigned K) {
    __shared__ k3_w8_smem sm;
    const size_t wstride = (size_t)N * K / 2u;
    const size_t sstride = (size_t)N * K / 32u;
    for (unsigned e = 0; e < nrep; ++e) {
        d_gemv_mxfp4(C, x, W + (size_t)e * wstride, scale + (size_t)e * sstride, 1u, N, K,
                     blockIdx.x, gridDim.x, sm.xstage);
        __syncthreads();
    }
}
