#pragma once
#include "op_gemm.cuh"

#ifndef PLOW_GEMV_TRANSPOSE_SWIZZLE
#define PLOW_GEMV_TRANSPOSE_SWIZZLE 0
#endif

// Weight rows fill MMA's m16 dimension; requests occupy n8. Thus M<=8 uses
// one MMA per weight tile without padding the request dimension to 16.
template<int RM, int BK, int STAGES, bool SPLIT>
__device__ __forceinline__ void d_gemv_transposed_tc(
        __nv_bfloat16* out, float* partial, const __nv_bfloat16* x, const __nv_bfloat16* w,
        int M, int N, int K, int splits, __nv_bfloat16* sm) {
    static_assert(RM == 8 || RM == 16 || RM == 32);
    static_assert((BK == 128 && STAGES == 3) || (BK == 256 && STAGES == 2));
    constexpr int BN = 64;
    constexpr int XS = PLOW_GEMV_TRANSPOSE_SWIZZLE ? BK : BK + 8, WS = XS;
    auto offset = [](int row, int k) {
        return row * XS + (PLOW_GEMV_TRANSPOSE_SWIZZLE ? k ^ ((row & 7) * 8) : k);
    };
    __nv_bfloat16* wx = sm;
    __nv_bfloat16* xx = sm + STAGES * BN * WS;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nt = blockIdx.x, sp = blockIdx.y;
    int steps = (K + BK - 1) / BK, per = (steps + splits - 1) / splits;
    int first = sp * per, last = min(first + per, steps);
    float acc[RM / 8][4] = {};
    auto stage = [&](int step, int buf) {
        for (int i = tid; i < BN * (BK / 8); i += 128) {
            int row = i / (BK / 8), k = (i % (BK / 8)) * 8;
            bool valid = nt * BN + row < N && step * BK + k + 8 <= K;
            const __nv_bfloat16* p = valid ? w + size_t(nt * BN + row) * K + step * BK + k : w;
            pgm_cp_async_cg16(wx + offset(buf * BN + row, k), p, valid ? 16 : 0);
        }
        for (int i = tid; i < RM * (BK / 8); i += 128) {
            int row = i / (BK / 8), k = (i % (BK / 8)) * 8;
            bool valid = row < M && step * BK + k + 8 <= K;
            const __nv_bfloat16* p = valid ? x + size_t(row) * K + step * BK + k : x;
            pgm_cp_async_cg16(xx + offset(buf * RM + row, k), p, valid ? 16 : 0);
        }
    };
    for (int s = 0; s < STAGES - 1; ++s) {
        if (first + s < last) stage(first + s, s);
        pgm_cp_commit();
    }
    for (int i = 0; i < last - first; ++i) {
        if (first + i + STAGES - 1 < last)
            stage(first + i + STAGES - 1, (i + STAGES - 1) % STAGES);
        pgm_cp_commit();
        pgm_cp_wait<STAGES - 1>();
        __syncthreads();
        int buf = i % STAGES;
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            unsigned a[4];
            pgm_ldmatrix_x4(a, wx + offset(buf * BN + warp * 16 + lane % 16, k + (lane / 16) * 8));
            #pragma unroll
            for (int r = 0; r < RM / 8; ++r) {
                unsigned b[2];
                pgm_ldmatrix_x2(b, xx + offset(buf * RM + r * 8 + (lane & 7), k + ((lane >> 3) & 1) * 8));
                pgm_mma(acc[r], a, b, acc[r]);
            }
        }
        __syncthreads();
    }
    pgm_cp_wait<0>();
    #pragma unroll
    for (int r = 0; r < RM / 8; ++r) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int n = nt * BN + warp * 16 + lane / 4 + (e / 2) * 8;
            int m = r * 8 + (lane % 4) * 2 + e % 2;
            if (m < M && n < N) {
                if constexpr (SPLIT) partial[(size_t(sp) * M + m) * N + n] = acc[r][e];
                else out[size_t(m) * N + n] = __float2bfloat16(acc[r][e]);
            }
        }
    }
}
__device__ __forceinline__ void d_gemv_reduce_parts(__nv_bfloat16* out, const float* partial, int count, int splits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = 0;
        for (int s = 0; s < splits; ++s) sum += partial[size_t(s) * count + i];
        out[i] = __float2bfloat16(sum);
    }
}
