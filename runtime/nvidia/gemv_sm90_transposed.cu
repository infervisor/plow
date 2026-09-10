#include "op_gemv_transposed.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "Transposed decode object requires sm_90a"
#endif

extern "C" {
__device__ unsigned plow_gemv_transposed_abi = 1;
__device__ unsigned plow_gemv_transposed_block = 128;
__device__ unsigned plow_gemv_transposed_max_rows = 32;
}

#define PLOW_TRANSPOSE_ENTRY(RM, BK, STAGES) \
extern "C" __global__ __launch_bounds__(128) \
void plow_gemv_bf16_m##RM##_bk##BK##_s##STAGES( \
        __nv_bfloat16* out, float* partial, const __nv_bfloat16* x, \
        const __nv_bfloat16* w, int M, int N, int K, int splits) { \
    if (blockDim.x != 128 || M < 1 || M > RM || N < 1 || K < 1 || (K & 7) || \
        splits < 1 || splits > 8 || (splits & (splits - 1))) { __trap(); return; } \
    extern __shared__ __nv_bfloat16 sm[]; \
    if (splits == 1) \
        d_gemv_transposed_tc<RM, BK, STAGES, false>(out, partial, x, w, M, N, K, splits, sm); \
    else \
        d_gemv_transposed_tc<RM, BK, STAGES, true>(out, partial, x, w, M, N, K, splits, sm); \
}

PLOW_TRANSPOSE_ENTRY(8, 128, 3)
PLOW_TRANSPOSE_ENTRY(8, 256, 2)
PLOW_TRANSPOSE_ENTRY(16, 128, 3)
PLOW_TRANSPOSE_ENTRY(16, 256, 2)
PLOW_TRANSPOSE_ENTRY(32, 128, 3)
PLOW_TRANSPOSE_ENTRY(32, 256, 2)

extern "C" __global__ void plow_gemv_bf16_reduce(
        __nv_bfloat16* out, const float* partial, int count, int splits) {
    d_gemv_reduce_parts(out, partial, count, splits);
}
