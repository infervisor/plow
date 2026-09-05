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

/* Vector decode state: bf16 bits of the 8 subnormal codes (m * 2^-9), replicated so a
 * 16-bit permutexvar indexed by the mantissa picks them; everything else is arithmetic. */
typedef struct {
    __m512i sub; /* lane i (0..31): bf16(i & 7) subnormal */
} plow_fp8_vlut;

static inline void plow_fp8_vlut_init(plow_fp8_vlut* v) {
    __attribute__((aligned(64))) uint16_t t[32];
    for (uint32_t i = 0; i < 32u; i++) t[i] = plow_f2bf(plow_e4m3_to_f32((uint8_t)(i & 7u)));
    v->sub = _mm512_loadu_si512((const void*)t);
}

/* 32 e4m3 bytes -> 32 bf16 (as a 512-bit integer vector).
 * Normal codes: (mag << 4) + (120 << 7) re-biases the 4-bit exponent (bias 7) to bf16's
 * (bias 127) and drops the 3 mantissa bits into bf16's top mantissa bits — exact. e == 0 codes
 * are subnormals (m * 2^-9) from the 8-entry table; 0x7f is the NaN code. Sign is OR'ed in.
 * One shuffle (the widen) + one permute per 32 weights, the rest are p0/p1 ALU ops. */
static inline __m512i plow_fp8x32_to_bf16(const plow_fp8_vlut* v, __m256i q) {
    const __m512i w = _mm512_cvtepu8_epi16(q);
    const __m512i mag = _mm512_and_si512(w, _mm512_set1_epi16(0x7F));
    __m512i r = _mm512_add_epi16(_mm512_slli_epi16(mag, 4), _mm512_set1_epi16(0x3C00));
    const __mmask32 sub = _mm512_testn_epi16_mask(mag, _mm512_set1_epi16(0x78));
    r = _mm512_mask_permutexvar_epi16(r, sub, mag, v->sub);
    const __mmask32 nan = _mm512_cmpeq_epi16_mask(mag, _mm512_set1_epi16(0x7F));
    r = _mm512_mask_mov_epi16(r, nan, _mm512_set1_epi16(0x7FC0));
    const __m512i sign = _mm512_slli_epi16(_mm512_and_si512(w, _mm512_set1_epi16(0x80)), 8);
    return _mm512_or_si512(r, sign);
}
#endif

#endif /* PLOW_CPU_FP8_COMMON_H */
