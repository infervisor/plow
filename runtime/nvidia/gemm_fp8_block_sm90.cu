/* Native block-scaled FP8 decode GEMM object (segment_roles NATIVE_DECODE_FP8_BLOCK). plowrt
 * launches `plow_gemm_fp8_block_m<MT>_s<stages>_v<variant>` (cfg = stages << 4 | variant, see
 * plow_asset::segment_roles::fp8_block_cfg) on a static grid-stride grid of at most
 * N / (64 * NC) * splits CTAs, and `plow_gemm_fp8_block_reduce` after a split launch of a
 * variant without the in-kernel reduce. */
#define PLOW_NV_HOPPER 1
#include "op_gemm_fp8_block_sm90.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "Block FP8 decode object requires sm_90a"
#endif

extern "C" {
__device__ unsigned plow_gemm_fp8_block_abi = 1;
__device__ unsigned plow_gemm_fp8_block_max_rows = 32;
}

struct alignas(64) Fp8bTensorMap {
    unsigned char bytes[128];
};

__device__ __forceinline__ unsigned fp8b_dynamic_smem() {
    unsigned v;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(v));
    return v;
}

#define PLOW_FP8B_ENTRY(MT, ST, V)                                                                 \
extern "C" __global__ void __launch_bounds__(128 * ((V & 1) ? 2 : 3))                              \
plow_gemm_fp8_block_m##MT##_s##ST##_v##V(                                                       \
        __nv_bfloat16* out, float* partial, unsigned* count,                                       \
        const __grid_constant__ Fp8bTensorMap w_map, const __grid_constant__ Fp8bTensorMap x_map, \
        const float* a_scale, const float* w_scale, unsigned M, unsigned M4, unsigned N,           \
        unsigned K, unsigned splits) {                                                             \
    constexpr unsigned NC = (V & 1) ? 1 : 2;                                                       \
    if (blockDim.x != 128 * (NC + 1) || M < 1 || M > MT || M4 < M || M4 % 4 || N % (64 * NC) ||    \
        K % PLOW_FP8B_BK || splits < 1 || splits > 8 || splits > K / PLOW_FP8B_BK ||               \
        (splits > 1 && !partial) || ((V & 4) && splits > 1 && !count) ||                           \
        fp8b_dynamic_smem() < fp8b_smem_bytes(MT, ST, NC)) {                                       \
        __trap();                                                                                  \
        return;                                                                                    \
    }                                                                                              \
    extern __shared__ __align__(1024) uint8_t arena[];                                             \
    d_gemm_fp8_block<MT, ST, NC, (V & 2) != 0, (V & 4) != 0, (V & 8) != 0>(out, partial, count, &w_map, &x_map,  \
        a_scale, w_scale, M, M4, N, K, splits, arena);                                             \
}

#define PLOW_FP8B_VARIANTS(MT, ST) \
    PLOW_FP8B_ENTRY(MT, ST, 0) PLOW_FP8B_ENTRY(MT, ST, 1) PLOW_FP8B_ENTRY(MT, ST, 2) \
    PLOW_FP8B_ENTRY(MT, ST, 3) PLOW_FP8B_ENTRY(MT, ST, 4) PLOW_FP8B_ENTRY(MT, ST, 5) \
    PLOW_FP8B_ENTRY(MT, ST, 6) PLOW_FP8B_ENTRY(MT, ST, 7) PLOW_FP8B_ENTRY(MT, ST, 8) \
    PLOW_FP8B_ENTRY(MT, ST, 9) PLOW_FP8B_ENTRY(MT, ST, 10) PLOW_FP8B_ENTRY(MT, ST, 11) \
    PLOW_FP8B_ENTRY(MT, ST, 12) PLOW_FP8B_ENTRY(MT, ST, 13) PLOW_FP8B_ENTRY(MT, ST, 14) \
    PLOW_FP8B_ENTRY(MT, ST, 15)

PLOW_FP8B_VARIANTS(8, 4)
PLOW_FP8B_VARIANTS(8, 6)
PLOW_FP8B_VARIANTS(16, 4)
PLOW_FP8B_VARIANTS(16, 6)
PLOW_FP8B_VARIANTS(32, 4)
PLOW_FP8B_VARIANTS(32, 6)

extern "C" __global__ void plow_gemm_fp8_block_reduce(
        __nv_bfloat16* out, const float* partial, unsigned M, unsigned N, unsigned splits) {
    d_fp8_block_reduce(out, partial, M, N, splits);
}
