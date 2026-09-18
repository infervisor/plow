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
extern "C" __device__ unsigned plow_gemma4_down_128x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_down_192x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_down_320x128x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_down_fp8_128x256x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_down_fp8_128x128x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_down_fp8_192x256x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_128x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_192x256x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_320x128x64_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_fp8_128x256x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_fp8_128x128x128_1 = 1;
extern "C" __device__ unsigned plow_gemma4_output_fp8_192x256x128_1 = 1;

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

#define GEMMA4_DOWN_ENTRY(NAME, BM, BN, N, K)                                     \
    extern "C" __global__ __attribute__((flatten))                              \
    __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(                    \
        bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned, unsigned,     \
        unsigned) {                                                               \
        __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, 64)];                         \
        d_gemm_t<BM, BN, 64, GM_WM, GM_WN, false, GM_SWZ, GM_WGM,                 \
                 (GM_PP != 0), true>(C, A, B, nullptr, nullptr, M, N, K,           \
                                     blockIdx.x, gridDim.x, lds);                  \
    }

GEMMA4_DOWN_ENTRY(plow_gemma4_12b_down_c2_gfx942, 128, 256, 3840, 15360)
GEMMA4_DOWN_ENTRY(plow_gemma4_12b_down_c5_gfx942, 192, 256, 3840, 15360)
GEMMA4_DOWN_ENTRY(plow_gemma4_12b_down_c6_gfx942, 320, 128, 3840, 15360)
GEMMA4_DOWN_ENTRY(plow_gemma4_31b_down_c2_gfx942, 128, 256, 5376, 21504)
GEMMA4_DOWN_ENTRY(plow_gemma4_31b_down_c5_gfx942, 192, 256, 5376, 21504)
GEMMA4_DOWN_ENTRY(plow_gemma4_12b_output_s_c2_gfx942, 128, 256, 3840, 4096)
GEMMA4_DOWN_ENTRY(plow_gemma4_12b_output_g_c2_gfx942, 128, 256, 3840, 8192)
GEMMA4_DOWN_ENTRY(plow_gemma4_31b_output_s_c5_gfx942, 192, 256, 5376, 8192)
GEMMA4_DOWN_ENTRY(plow_gemma4_31b_output_g_c5_gfx942, 192, 256, 5376, 16384)
GEMMA4_DOWN_ENTRY(plow_gemma4_31b_output_g_c6_gfx942, 320, 128, 5376, 16384)

#define GEMMA4_DOWN_FP8_ENTRY(NAME, BM, BN, N, K)                                \
    extern "C" __global__ __attribute__((flatten))                              \
    __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void NAME(                    \
        bf16* C, const unsigned char* A, const unsigned char* B,                  \
        const float* ascale, const float* wscale, unsigned M, unsigned, unsigned, \
        unsigned) {                                                               \
        __shared__ unsigned char lds[(BM + BN) * 128];                            \
        d_gemm_fp8_t<BM, BN, 128, GM_WM, GM_WN>(                                 \
            C, A, B, ascale, wscale, M, N, K, blockIdx.x, gridDim.x, (bf16*)lds); \
    }

GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_12b_down_fp8_c2_gfx942, 128, 256, 3840, 15360)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_12b_down_fp8_c5_gfx942, 192, 256, 3840, 15360)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_down_fp8_c2_gfx942, 128, 256, 5376, 21504)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_down_fp8_c3_gfx942, 128, 128, 5376, 21504)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_down_fp8_c5_gfx942, 192, 256, 5376, 21504)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_s_fp8_c2_gfx942, 128, 256, 5376, 8192)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_s_fp8_c3_gfx942, 128, 128, 5376, 8192)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_s_fp8_c5_gfx942, 192, 256, 5376, 8192)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_g_fp8_c2_gfx942, 128, 256, 5376, 16384)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_g_fp8_c3_gfx942, 128, 128, 5376, 16384)
GEMMA4_DOWN_FP8_ENTRY(plow_gemma4_31b_output_g_fp8_c5_gfx942, 192, 256, 5376, 16384)
