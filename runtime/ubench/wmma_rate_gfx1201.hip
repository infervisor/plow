/* Register-only gfx12 WMMA issue rate. Build/run: mfma_rate_test.c.
 * Each wave computes one 16x16x16 product per instruction (8192 operations),
 * in either wave mode. Distinct seeds keep independent chains from merging.
 */
#include <hip/hip_runtime.h>

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define RATE_WAVE 32
#define RATE_NAME_(DT, CH) wmma_##DT##_w32_ch##CH
#define RATE_BF16 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12
#define RATE_F16 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
#define RATE_FP8 __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
#define RATE_INT8 __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
#else
#define RATE_WAVE 64
#define RATE_NAME_(DT, CH) wmma_##DT##_w64_ch##CH
#define RATE_BF16 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12
#define RATE_F16 __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12
#define RATE_FP8 __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12
#define RATE_INT8 __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12
#endif

typedef __bf16 bf16_frag __attribute__((ext_vector_type(256 / RATE_WAVE)));
typedef _Float16 f16_frag __attribute__((ext_vector_type(256 / RATE_WAVE)));
typedef float f32_frag __attribute__((ext_vector_type(256 / RATE_WAVE)));
typedef int i32_frag __attribute__((ext_vector_type(256 / RATE_WAVE)));
#if RATE_WAVE == 32
typedef int packed_frag __attribute__((ext_vector_type(2)));
#else
typedef int packed_frag;
#endif

#define FLOAT_OP(OP, A, B, C) OP(A, B, C)
#define INT_OP(OP, A, B, C) OP(true, A, true, B, C, false)

#define MK(DT, CH, INPUT, ACC, ONE, OP, CALL)                                      \
    extern "C" __global__ __launch_bounds__(4 * RATE_WAVE)                        \
    void RATE_NAME_(DT, CH)(float* out, unsigned iters) {                         \
        const INPUT a = (INPUT)(ONE), b = (INPUT)(ONE);                          \
        ACC acc[CH];                                                             \
        _Pragma("unroll") for (int c = 0; c < CH; c++)                            \
            acc[c] = (ACC)(c + (threadIdx.x & 7));                               \
        for (unsigned t = 0; t < iters; t++) {                                   \
            _Pragma("unroll") for (int u = 0; u < 8; u++)                        \
                _Pragma("unroll") for (int c = 0; c < CH; c++)                    \
                    acc[c] = CALL(OP, a, b, acc[c]);                             \
        }                                                                       \
        float sum = 0;                                                          \
        _Pragma("unroll") for (int c = 0; c < CH; c++)                            \
            _Pragma("unroll") for (int e = 0; e < 256 / RATE_WAVE; e++)           \
                sum += (float)acc[c][e];                                        \
        out[blockIdx.x * blockDim.x + threadIdx.x] = sum;                         \
    }

#define SWEEP(DT, INPUT, ACC, ONE, OP, CALL)           \
    MK(DT, 1, INPUT, ACC, ONE, OP, CALL)               \
    MK(DT, 2, INPUT, ACC, ONE, OP, CALL)               \
    MK(DT, 4, INPUT, ACC, ONE, OP, CALL)               \
    MK(DT, 6, INPUT, ACC, ONE, OP, CALL)               \
    MK(DT, 8, INPUT, ACC, ONE, OP, CALL)               \
    MK(DT, 12, INPUT, ACC, ONE, OP, CALL)              \
    MK(DT, 16, INPUT, ACC, ONE, OP, CALL)

SWEEP(bf16, bf16_frag, f32_frag, 1, RATE_BF16, FLOAT_OP)
SWEEP(f16, f16_frag, f32_frag, 1, RATE_F16, FLOAT_OP)
#if (RATE_WAVE == 32 && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)) || \
    (RATE_WAVE == 64 && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12))
SWEEP(fp8, packed_frag, f32_frag, 0x38383838, RATE_FP8, FLOAT_OP)
#endif
#if (RATE_WAVE == 32 && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12)) || \
    (RATE_WAVE == 64 && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12))
SWEEP(int8, packed_frag, i32_frag, 0x01010101, RATE_INT8, INT_OP)
#endif
