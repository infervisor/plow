/* mfma_rate_gfx942.hip — what the CDNA3 matrix core actually issues, with NOTHING else running.
 *
 * No LDS, no global traffic, no barriers: a fixed number of INDEPENDENT accumulator chains fed
 * from registers. This is the ceiling any GEMM body can aim at, and the denominator for
 * "MfmaUtil" claims. Sweeps the chain count so the MFMA->MFMA RAW latency shows up directly.
 */
#include "amd_common.h"

#include "amd_arch.h"

#define MK(NAME, CH)                                                                        \
    extern "C" __global__ __launch_bounds__(PLOW_WG_THREADS, 2) void NAME(float* out,        \
                                                                         unsigned iters) {   \
        f32x16 acc[CH];                                                                      \
        _Pragma("unroll") for (int i = 0; i < CH; i++) acc[i] = (f32x16)(float)threadIdx.x;   \
        bf16x8 a, b;                                                                         \
        _Pragma("unroll") for (int i = 0; i < 8; i++) {                                       \
            a[i] = (bf16)(float)(threadIdx.x + i);                                            \
            b[i] = (bf16)(float)(threadIdx.x - i);                                            \
        }                                                                                    \
        for (unsigned t = 0; t < iters; t++) {                                                \
            _Pragma("unroll") for (int u = 0; u < 8; u++)                                      \
                _Pragma("unroll") for (int i = 0; i < CH; i++)                                 \
                    acc[i] = plow_mfma_bf16_32x32(a, b, acc[i]);                               \
        }                                                                                    \
        float s = 0;                                                                          \
        _Pragma("unroll") for (int i = 0; i < CH; i++)                                         \
            _Pragma("unroll") for (int e = 0; e < 16; e++) s += acc[i][e];                     \
        if (s == 12345.678f) out[threadIdx.x] = s;                                             \
    }

MK(mfma_ch1, 1)
MK(mfma_ch2, 2)
MK(mfma_ch4, 4)
MK(mfma_ch6, 6)
MK(mfma_ch8, 8)

/* The 16x16 twin: same MACs per chain-slot, a QUARTER of the accumulator registers. If the
 * matrix core's issue rate were the lever the two ladders would differ; if it is not, they land
 * on the same line and the MFMA-shape question is closed on CDNA3 as it was on CDNA4. */
#define MK16(NAME, CH)                                                                      \
    extern "C" __global__ __launch_bounds__(PLOW_WG_THREADS, 2) void NAME(float* out,        \
                                                                         unsigned iters) {   \
        f32x4 acc[CH];                                                                       \
        _Pragma("unroll") for (int i = 0; i < CH; i++) acc[i] = (f32x4)(float)threadIdx.x;    \
        bf16x8 a, b;                                                                         \
        _Pragma("unroll") for (int i = 0; i < 8; i++) {                                       \
            a[i] = (bf16)(float)(threadIdx.x + i);                                            \
            b[i] = (bf16)(float)(threadIdx.x - i);                                            \
        }                                                                                    \
        for (unsigned t = 0; t < iters; t++) {                                                \
            _Pragma("unroll") for (int u = 0; u < 8; u++)                                      \
                _Pragma("unroll") for (int i = 0; i < CH; i++)                                 \
                    acc[i] = plow_mfma_bf16_16x16(a, b, acc[i]);                               \
        }                                                                                    \
        float s = 0;                                                                          \
        _Pragma("unroll") for (int i = 0; i < CH; i++)                                         \
            _Pragma("unroll") for (int e = 0; e < 4; e++) s += acc[i][e];                      \
        if (s == 12345.678f) out[threadIdx.x] = s;                                             \
    }

MK16(mfma16_ch4, 4)
MK16(mfma16_ch8, 8)
MK16(mfma16_ch16, 16)
MK16(mfma16_ch24, 24)
MK16(mfma16_ch32, 32)

/* fp8, through the SAME wrapper the prefill GEMM uses (4 x 32x32x16 on CDNA3). */
#define MK8(NAME, CH)                                                                       \
    extern "C" __global__ __launch_bounds__(PLOW_WG_THREADS, 2) void NAME(float* out,        \
                                                                         unsigned iters) {   \
        f32x16 acc[CH];                                                                      \
        _Pragma("unroll") for (int i = 0; i < CH; i++) acc[i] = (f32x16)(float)threadIdx.x;   \
        mfma_f8f6f4_operand a, b;                                                            \
        _Pragma("unroll") for (int i = 0; i < 8; i++) {                                       \
            a[i] = (int)(threadIdx.x + i);                                                    \
            b[i] = (int)(threadIdx.x - i);                                                    \
        }                                                                                    \
        for (unsigned t = 0; t < iters; t++) {                                                \
            _Pragma("unroll") for (int u = 0; u < 8; u++)                                      \
                _Pragma("unroll") for (int i = 0; i < CH; i++)                                 \
                    acc[i] = plow_mfma_fp8_32x32(a, b, acc[i]);                                \
        }                                                                                    \
        float s = 0;                                                                          \
        _Pragma("unroll") for (int i = 0; i < CH; i++)                                         \
            _Pragma("unroll") for (int e = 0; e < 16; e++) s += acc[i][e];                     \
        if (s == 12345.678f) out[threadIdx.x] = s;                                             \
    }

MK8(mfma8_ch2, 2)
MK8(mfma8_ch4, 4)
MK8(mfma8_ch6, 6)

/* ------------------------------------------------------------------------------------------
 * LDS read bandwidth, and the MFMA/LDS TRADE.
 *
 * The GEMM body's ratio is (SM+SN) ds_read_b128 per SM*SN*KS matrix instructions. If the LDS
 * pipe is the wall, TF/s falls off as soon as RD/MFMA rises past the hardware's ratio -- and the
 * tile ladder has to be chosen on that ratio rather than on arithmetic intensity. LDSMIX sweeps
 * exactly that pair, with the SAME fragment layout the GEMM uses (32 rows, 16 B/lane, the
 * op_gemm XOR swizzle) so the bank behaviour is the real one.
 * ---------------------------------------------------------------------------------------- */
#define LDSMIX(NAME, RD, CH)                                                                \
    extern "C" __global__ __launch_bounds__(PLOW_WG_THREADS, 2) void NAME(float* out,        \
                                                                         unsigned iters) {   \
        __shared__ bf16 tile[8 * 1024];                                                      \
        const unsigned lane = threadIdx.x & 63, frow = lane % 32;                            \
        for (unsigned i = threadIdx.x; i < 8 * 1024; i += PLOW_WG_THREADS) tile[i] = (bf16)i; \
        __syncthreads();                                                                     \
        f32x16 acc[CH];                                                                      \
        _Pragma("unroll") for (int i = 0; i < CH; i++) acc[i] = (f32x16)(float)threadIdx.x;   \
        bf16x8 fr[RD];                                                                       \
        unsigned base[RD];                                                                   \
        _Pragma("unroll") for (int i = 0; i < RD; i++) {                                      \
            const unsigned row = (frow + i * 32) & 127;                                       \
            base[i] = row * 64 + ((8 * (lane / 32)) ^ ((row & 7) << 3));                      \
        }                                                                                    \
        for (unsigned t = 0; t < iters; t++) {                                                \
            _Pragma("unroll") for (int u = 0; u < 8; u++) {                                    \
                _Pragma("unroll") for (int i = 0; i < RD; i++)                                 \
                    __builtin_memcpy(&fr[i], &tile[base[i] + ((u & 1) << 9)], 16);             \
                _Pragma("unroll") for (int i = 0; i < CH; i++)                                 \
                    acc[i] = plow_mfma_bf16_32x32(fr[i % RD], fr[(i + 1) % RD], acc[i]);       \
            }                                                                                \
        }                                                                                    \
        float s = 0;                                                                          \
        _Pragma("unroll") for (int i = 0; i < CH; i++)                                         \
            _Pragma("unroll") for (int e = 0; e < 16; e++) s += acc[i][e];                     \
        if (s == 12345.678f) out[threadIdx.x] = s;                                             \
    }

/* Pure LDS read: the bandwidth number, same fragment layout, no matrix core at all. */
#define LDSRD(NAME, RD)                                                                     \
    extern "C" __global__ __launch_bounds__(PLOW_WG_THREADS, 2) void NAME(float* out,        \
                                                                         unsigned iters) {   \
        __shared__ bf16 tile[8 * 1024];                                                      \
        const unsigned lane = threadIdx.x & 63, frow = lane % 32;                            \
        for (unsigned i = threadIdx.x; i < 8 * 1024; i += PLOW_WG_THREADS) tile[i] = (bf16)i; \
        __syncthreads();                                                                     \
        bf16x8 fr[RD];                                                                       \
        unsigned base[RD], sink = 0;                                                         \
        _Pragma("unroll") for (int i = 0; i < RD; i++) {                                      \
            const unsigned row = (frow + i * 32) & 127;                                       \
            base[i] = row * 64 + ((8 * (lane / 32)) ^ ((row & 7) << 3));                      \
        }                                                                                    \
        for (unsigned t = 0; t < iters; t++) {                                                \
            _Pragma("unroll") for (int u = 0; u < 8; u++) {                                    \
                _Pragma("unroll") for (int i = 0; i < RD; i++)                                 \
                    __builtin_memcpy(&fr[i], &tile[base[i] + ((u & 1) << 9)], 16);             \
                _Pragma("unroll") for (int i = 0; i < RD; i++) {                               \
                    unsigned w;                                                                \
                    __builtin_memcpy(&w, &fr[i], 4);                                            \
                    sink += w;                                                                 \
                }                                                                              \
            }                                                                                \
        }                                                                                    \
        if (sink == 0xdeadbeefu) out[threadIdx.x] = (float)sink;                               \
    }

LDSRD(rd_r5, 5)
LDSRD(rd_r8, 8)
LDSMIX(mix_r5_m6, 5, 6)   /* 192x256's ratio: SM=3 SN=2 -> 5 reads, 6 MFMA pairs */
LDSMIX(mix_r6_m8, 6, 8)   /* 256x256 at 8 waves: SM=4 SN=2                        */
LDSMIX(mix_r8_m16, 8, 16) /* 256x256 at 4 waves / Tensile's ratio: SM=4 SN=4      */
LDSMIX(mix_r5_m12, 5, 12) /* same reads, twice the MFMA -- is the LDS pipe free?  */
LDSMIX(mix_r5_m24, 5, 24)
