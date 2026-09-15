#include "op_gemm.h"

static_assert(PLOW_THREADS == 512, "Gemma-4 gfx942 GLU requires eight waves");
static_assert(PLOW_WAVES == 8, "Gemma-4 gfx942 GLU requires eight waves");

extern "C" __device__ unsigned plow_gemma4_glu_abi_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_wg512_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_nospill_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_c2_128x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_c5_192x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_c5_exact_shape_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_fp8_128x256x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_glu_fp8_192x256x128_1 = 1;

extern "C" __global__ __attribute__((flatten))
__launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void plow_gemma4_gemm_glu_c2_gfx942(
    bf16* C, const bf16* A, const bf16* Bg, const bf16* Bu, unsigned M, unsigned N,
    unsigned K, unsigned act) {
    __shared__ bf16 lds[GM_LDS_HALVES_T(128, 256, 64)];
    d_gemm_t<128, 256, 64, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true,
             true>(C, A, Bg, nullptr, nullptr, M, N, K, blockIdx.x, gridDim.x, lds, Bu, act);
}

/* Exact N/K helps C5's generated address arithmetic, but increases C2 waitcnt pressure enough to
 * lose on hardware. M stays dynamic for ragged/co-packed suffixes. */
#define GEMMA4_GLU_ENTRY(NAME, BM, BN, BK, N, K)                                     \
    extern "C" __global__ __attribute__((flatten))                                  \
    __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(                       \
        bf16* C, const bf16* A, const bf16* Bg, const bf16* Bu, unsigned M,           \
        unsigned, unsigned, unsigned act) {                                           \
        __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, BK)];                            \
        d_gemm_t<BM, BN, BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true, \
                 true>(C, A, Bg, nullptr, nullptr, M, N, K, blockIdx.x, gridDim.x,    \
                       lds, Bu, act);                                                 \
    }

GEMMA4_GLU_ENTRY(plow_gemma4_12b_gemm_glu_c5_gfx942, 192, 256, 64, 15360, 3840)
GEMMA4_GLU_ENTRY(plow_gemma4_31b_gemm_glu_c5_gfx942, 192, 256, 64, 21504, 5376)

#define GEMMA4_GLU_FP8_ENTRY(NAME, BM, N, K)                                      \
    extern "C" __global__ __attribute__((flatten))                               \
    __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(                    \
        bf16* C, const unsigned char* A, const unsigned char* Bg,                  \
        const unsigned char* Bu, const float* ascale, const float* gscale,         \
        const float* uscale, unsigned M, unsigned, unsigned, unsigned act) {       \
        __shared__ unsigned char lds[(BM + 256) * 128];                            \
        d_gemm_fp8_t<BM, 256, 128, GM_WM, GM_WN, true, true>(                     \
            C, A, Bg, ascale, gscale, M, N, K, blockIdx.x, gridDim.x,             \
            (bf16*)lds, Bu, uscale, act);                                          \
    }

GEMMA4_GLU_FP8_ENTRY(plow_gemma4_12b_gemm_glu_fp8_c2_gfx942, 128, 15360, 3840)
GEMMA4_GLU_FP8_ENTRY(plow_gemma4_12b_gemm_glu_fp8_c5_gfx942, 192, 15360, 3840)
GEMMA4_GLU_FP8_ENTRY(plow_gemma4_31b_gemm_glu_fp8_c2_gfx942, 128, 21504, 5376)
GEMMA4_GLU_FP8_ENTRY(plow_gemma4_31b_gemm_glu_fp8_c5_gfx942, 192, 21504, 5376)
