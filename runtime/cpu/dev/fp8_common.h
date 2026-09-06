/* fp8_common.h — OCP e4m3fn decode shared by every tier.
 *
 * e4m3fn: sign | 4-bit exponent (bias 7) | 3-bit mantissa; no infinities, 0x7f/0xff = NaN,
 * subnormals at e == 0. Every finite value is exactly representable in bf16 (3 mantissa bits,
 * exponent range within bf16's), so dequant-to-bf16 is lossless and the w8a16 dot product is
 * a plain bf16 dot with the per-output-channel f32 scale applied once in the epilogue. */
#ifndef PLOW_CPU_FP8_COMMON_H
#define PLOW_CPU_FP8_COMMON_H

#include <math.h>
#include <stdint.h>
#include "cpu_dev.h"

#define PLOW_FP8_E4M3_MAX 448.0f

static inline float plow_e4m3_to_f32(uint8_t b) {
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0u) v = ldexpf((float)m, -9);                  /* m/8 * 2^-6 */
    else if (e == 15u && m == 7u) v = NAN;                   /* the only non-finite code */
    else v = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
    return (b & 0x80u) ? -v : v;
}

/* Magnitude LUT: bf16 bits of the 128 unsigned e4m3 codes (sign is OR'ed in by the caller). */
static inline void plow_fp8_mag_lut(uint16_t lut[128]) {
    for (uint32_t i = 0; i < 128u; i++) lut[i] = plow_f2bf(plow_e4m3_to_f32((uint8_t)i));
}

#if defined(__AVX512BW__) && defined(__AVX512F__)
#include <immintrin.h>

/* Vector decode state: nothing is table-driven any more; the struct stays so the tiers' init
 * call sites and the AMX pack are unchanged. */
typedef struct {
    __m512i unused;
} plow_fp8_vlut;

static inline void plow_fp8_vlut_init(plow_fp8_vlut* v) { v->unused = _mm512_setzero_si512(); }

/* 32 e4m3 bytes -> 32 bf16 (as a 512-bit integer vector), 5 vector uops.
 * Sign-extend the byte so bit 15 is the sign, shift the code left 4 (mantissa -> bf16 bits 4..6,
 * exponent -> bits 7..10) and mask to sign | 0x07F0; adding 120 << 7 re-biases the exponent
 * (bias 7 -> 127) -- exact for every normal code. Codes with e == 0 are zero-masked: +-0 is
 * exact, the 7 nonzero subnormals (|v| <= 7 * 2^-9, ~1e-4 of real fp8 weights) decode to 0,
 * and the NaN code 0x7F decodes to 480 (never present in weights). The old exact LUT path cost
 * 11 uops per 32 weights and made GEMV_FP8 uop-bound at ~57 GB/s of fp8 bytes. */
static inline __m512i plow_fp8x32_to_bf16(const plow_fp8_vlut* v, __m256i q) {
    (void)v;
    const __m512i w = _mm512_cvtepi8_epi16(q);
    const __m512i t = _mm512_and_si512(_mm512_slli_epi16(w, 4), _mm512_set1_epi16((short)0x87F0));
    const __mmask32 nrm = _mm512_test_epi16_mask(t, _mm512_set1_epi16(0x0780));
    return _mm512_maskz_add_epi16(nrm, t, _mm512_set1_epi16(0x3C00));
}
#endif

#endif /* PLOW_CPU_FP8_COMMON_H */
