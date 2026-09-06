/* mxfp4_common.h — OCP MXFP4 (e2m1 weights + one E8M0 scale per 32 K) decode shared by every
 * tier. Row layout (dev_isa.h op 91, byte-identical to the GPT-OSS checkpoint): W[n] is K/2 bytes,
 * LOW nibble = even k; S[n] is K/32 E8M0 bytes, scale = 2^(s-127) read as bitcast(s << 23) like
 * the GPU (s = 0 -> 0, s = 255 -> +Inf). The dot product is accumulated in f32 PER 32-BLOCK and
 * the block scale multiplied once per block, so any scale is exact and nothing under/overflows in
 * the bf16 operand. */
#ifndef PLOW_CPU_MXFP4_COMMON_H
#define PLOW_CPU_MXFP4_COMMON_H

#include <stdint.h>
#include "cpu_dev.h"

#define PLOW_MX_BLK 32u

static const float plow_e2m1_lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

static inline float plow_e8m0_to_f32(uint8_t s) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)s << 23;
    return v.f;
}

static inline float plow_mxfp4_at(const uint8_t* w, uint32_t k) {
    const uint8_t b = w[k >> 1];
    return plow_e2m1_lut[(k & 1u) ? (b >> 4) : (b & 0xFu)];
}

/* w . x over K (a partial last block scales what it has). */
static inline float plow_mxfp4_row_dot(const uint8_t* w, const uint8_t* s, const plow_bf16* x,
                                       uint32_t K) {
    float acc = 0.0f;
    for (uint32_t k0 = 0; k0 < K; k0 += PLOW_MX_BLK) {
        const uint32_t k1 = k0 + PLOW_MX_BLK < K ? k0 + PLOW_MX_BLK : K;
        float blk = 0.0f;
        for (uint32_t k = k0; k < k1; k++) blk += plow_mxfp4_at(w, k) * plow_bf2f(x[k]);
        acc += blk * plow_e8m0_to_f32(s[k0 / PLOW_MX_BLK]);
    }
    return acc;
}

#if defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512BF16__)
#include <immintrin.h>

/* Vector decode state. The unit of work is 128 weights = 64 packed bytes = 4 blocks, loaded as one
 * zmm of 32 words: word j holds weights 4j..4j+3 in its four nibbles, so nibble q of every word is
 * `word >> 4q` and one vpermw LUT lookup per q gives 32 bf16 lanes (lane j = weight 4j+q) -- no
 * byte widening, and 4 lookups per 64 loaded bytes instead of the 2 per 32 bytes a 64-weight step
 * needs. vdpbf16ps reduces lane pairs (2l, 2l+1) = weights (8l+q, 8l+4+q), so x is staged once per
 * row-set with xq[2l+h] = x[8l+4h+q] (plow_mx_stage_x) and f32 lane l of the four dots covers
 * weights 8l..8l+7: lanes 4b..4b+3 are block b of the chunk, so the four block scales apply as ONE
 * fmadd against a 4-lane-spread scale vector (vs two broadcast fmadds per 64 weights before).
 * K must be <= 32 * PLOW_MX_MAX_BLOCKS (callers fall back to golden otherwise). */
typedef struct {
    __m512i lut;   /* lane i: bf16(plow_e2m1_lut[i & 15]) (amx/ stages its bf16 tiles from this) */
    __m512i luti;  /* lane i: (int16)(2 * plow_e2m1_lut[i & 15]) -- the int16 path below */
    __m512i xi[4]; /* vpermi2w indices staging nibble position q of a 64-element half */
    __m512i sidx;  /* vpermps: lane l <- l / 4 */
} plow_mx_vlut;

#define PLOW_MX_CHUNK 128u
/* The RB / M loops below must fully unroll (constant trip counts after inlining) or acc[][]
 * lands on the stack behind store-forwarding chains; -O2 alone does not do it (measured: 3x). */
#define PLOW_MX_UNROLL _Pragma("GCC unroll 8")

static inline void plow_mx_vlut_init(plow_mx_vlut* v) {
    __attribute__((aligned(64))) uint16_t t[32], xi[4][32];
    __attribute__((aligned(64))) int16_t ti[32];
    __attribute__((aligned(64))) uint32_t si[16];
    for (uint32_t i = 0; i < 32u; i++) t[i] = plow_f2bf(plow_e2m1_lut[i & 15u]);
    for (uint32_t i = 0; i < 32u; i++) ti[i] = (int16_t)(2.0f * plow_e2m1_lut[i & 15u]);
    for (uint32_t q = 0; q < 4u; q++)
        for (uint32_t l = 0; l < 16u; l++)
            for (uint32_t h = 0; h < 2u; h++) xi[q][2u * l + h] = (uint16_t)(8u * (l & 7u) + 4u * h + q);
    for (uint32_t l = 0; l < 16u; l++) si[l] = l / 4u;
    v->lut = _mm512_load_si512((const void*)t);
    v->luti = _mm512_load_si512((const void*)ti);
    for (uint32_t q = 0; q < 4u; q++) v->xi[q] = _mm512_load_si512((const void*)xi[q]);
    v->sidx = _mm512_load_si512((const void*)si);
}

/* Elements needed for a staged x row of K elements (rounded up to whole chunks). */
static inline uint32_t plow_mx_staged_len(uint32_t K) {
    return (K + PLOW_MX_CHUNK - 1u) & ~(PLOW_MX_CHUNK - 1u);
}

static inline __mmask32 plow_mx_tail32(uint32_t n) {
    return n >= 32u ? 0xFFFFFFFFu : (__mmask32)((1u << n) - 1u);
}

/* xp[128c + 32q + 2l + h] = x[128c + 8l + 4h + q]; tail zero-padded. */
static inline void plow_mx_stage_x(const plow_mx_vlut* v, plow_bf16* xp, const plow_bf16* x,
                                   uint32_t K) {
    for (uint32_t k = 0; k < K; k += PLOW_MX_CHUNK) {
        __m512i a[4];
        if (k + PLOW_MX_CHUNK <= K) {
            for (uint32_t i = 0; i < 4u; i++) a[i] = _mm512_loadu_si512((const void*)(x + k + 32u * i));
        } else {
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t n = K - k > 32u * i ? K - k - 32u * i : 0u;
                a[i] = _mm512_maskz_loadu_epi16(plow_mx_tail32(n), x + k + 32u * i);
            }
        }
        for (uint32_t q = 0; q < 4u; q++) {
            const __m512i lo = _mm512_permutex2var_epi16(a[0], v->xi[q], a[1]);
            const __m512i hi = _mm512_permutex2var_epi16(a[2], v->xi[q], a[3]);
            _mm512_storeu_si512((void*)(xp + k + 32u * q), _mm512_mask_blend_epi16(0xFFFF0000u, lo, hi));
        }
    }
}

/* ---- int16 VNNI decode dot (opt-in; the bf16 path above stays the default) ---------------------
 * 2 * |e2m1| is {0,1,2,3,4,6,8,12}, exactly representable in int16, and the doubling folds into the
 * power-of-two E8M0 block scale: the WEIGHT side is LOSSLESS. Only x is quantized, once per staged
 * row at amax/32767. vpdpwssd retires ~3x faster than vdpbf16ps on Sapphire Rapids.
 * An f32 lane covers weights 8l..8l+7, all inside one 32-block, and the int32 accumulator is reset
 * every chunk, so it peaks at 8 * 12 * 32767 = 3.1e6 -- no overflow at any K -- and the E8M0 scale
 * is still applied exactly per block by the same f32 FMA the bf16 path uses. */

/* As plow_mx_stage_x but int16 at one scale per row; returns that scale (x ~= q * scale, already
 * halved for the doubled LUT). A row of only zeros (or a non-finite amax) stages to zeros. */
static inline float plow_mx_stage_x_i16(const plow_mx_vlut* v, int16_t* xp, const plow_bf16* x,
                                        uint32_t K) {
    const __m512i am = _mm512_set1_epi16(0x7FFF);
    __m512i mx = _mm512_setzero_si512();
    uint32_t k = 0;
    for (; k + 32u <= K; k += 32u)
        mx = _mm512_max_epu16(mx, _mm512_and_si512(_mm512_loadu_si512((const void*)(x + k)), am));
    if (k < K)
        mx = _mm512_max_epu16(
            mx, _mm512_and_si512(_mm512_maskz_loadu_epi16(plow_mx_tail32(K - k), x + k), am));
    __attribute__((aligned(64))) uint16_t mb[32];
    _mm512_store_si512((void*)mb, mx);
    uint16_t top = 0;
    for (uint32_t i = 0; i < 32u; i++) top = mb[i] > top ? mb[i] : top;
    const float amax = plow_bf2f((plow_bf16)top);
    const __m512 qv = _mm512_set1_ps(amax > 0.0f && top < 0x7F80u ? 32767.0f / amax : 0.0f);
    for (k = 0; k < K; k += PLOW_MX_CHUNK) {
        __m512i a[4];
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t n = K - k > 32u * i ? K - k - 32u * i : 0u;
            const __m512i b = k + PLOW_MX_CHUNK <= K
                                  ? _mm512_loadu_si512((const void*)(x + k + 32u * i))
                                  : _mm512_maskz_loadu_epi16(plow_mx_tail32(n), x + k + 32u * i);
            const __m512 f0 = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(b)), 16));
            const __m512 f1 = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b, 1)), 16));
            a[i] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvtps_epi32(_mm512_mul_ps(f0, qv)))),
                _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(_mm512_mul_ps(f1, qv))), 1);
        }
        for (uint32_t q = 0; q < 4u; q++) {
            const __m512i lo = _mm512_permutex2var_epi16(a[0], v->xi[q], a[1]);
            const __m512i hi = _mm512_permutex2var_epi16(a[2], v->xi[q], a[3]);
            _mm512_storeu_si512((void*)(xp + k + 32u * q), _mm512_mask_blend_epi16(0xFFFF0000u, lo, hi));
        }
    }
    return amax > 0.0f && top < 0x7F80u ? amax / (2.0f * 32767.0f) : 0.0f;
}

/* Stage one activation row into xp (int16 + row scale when i16, else bf16 and scale 1). */
static inline float plow_mx_stage(const plow_mx_vlut* v, void* xp, const plow_bf16* x, uint32_t K,
                                  int i16) {
    if (i16) return plow_mx_stage_x_i16(v, (int16_t*)xp, x, K);
    plow_mx_stage_x(v, (plow_bf16*)xp, x, K);
    return 1.0f;
}

/* Per-row f32 block scales for one row block (K/32 <= PLOW_MX_MAX_BLOCKS). The 4 zero pads after
 * the last block let a partial final chunk multiply its empty lanes by 0 instead of garbage. */
#define PLOW_MX_MAX_BLOCKS 512u /* K <= 16384 */
#define PLOW_MX_SF_STRIDE (PLOW_MX_MAX_BLOCKS + 4u)
static inline void plow_mx_scale_row(float* sf, const uint8_t* s, uint32_t nblk) {
    uint32_t b = 0;
    for (; b + 16u <= nblk; b += 16u)
        _mm512_storeu_ps(sf + b, _mm512_castsi512_ps(_mm512_slli_epi32(
                                     _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(s + b))), 23)));
    for (; b < nblk; b++) sf[b] = plow_e8m0_to_f32(s[b]);
    for (; b < nblk + 4u; b++) sf[b] = 0.0f;
}

/* One chunk (64 packed bytes, masked by wm for a partial last chunk) of RB weight rows x M staged
 * rows into acc[][]. Nibble-major: b[r] is shifted in place by 4 per step, so besides the RB*M dot
 * and RB*M accumulator registers only the LUT, RB packed words, RB decoded words and one x vector
 * are live (2*RB*M + 2*RB + 3 <= 32). */
static inline __attribute__((always_inline)) void plow_mx_chunk(
    const plow_mx_vlut* v, const uint8_t* W, size_t ldw, size_t lead, const float* sf,
    const void* XP, size_t ldx, uint32_t c, __mmask64 wm, const uint32_t RB, const uint32_t M,
    const int I16, __m512 acc[4][8]) {
    /* One accumulator array either way (f32 lanes bit-cast for the bf16 path) so the two arms cost
     * the same registers; I16 is a literal at every call site, so only one arm survives. */
    const plow_bf16* xp = (const plow_bf16*)XP;
    __m512i b[4], t[4][8];
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) {
        const uint8_t* w = W + r * ldw + (size_t)c * 64u;
        _mm_prefetch((const char*)(w + lead), _MM_HINT_T0);
        b[r] = _mm512_maskz_loadu_epi8(wm, w);
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) t[r][m] = _mm512_setzero_si512();
    }
    PLOW_MX_UNROLL for (uint32_t q = 0; q < 4u; q++) {
        __m512i wq[4];
        PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
            wq[r] = _mm512_permutexvar_epi16(q ? _mm512_srli_epi16(b[r], 4 * q) : b[r],
                                             I16 ? v->luti : v->lut);
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) {
            const __m512i xq = _mm512_loadu_si512(
                (const void*)(xp + m * ldx + (size_t)c * PLOW_MX_CHUNK + 32u * q));
            PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
                t[r][m] = I16 ? _mm512_dpwssd_epi32(t[r][m], wq[r], xq)
                              : _mm512_castps_si512(_mm512_dpbf16_ps(_mm512_castsi512_ps(t[r][m]),
                                                                     (__m512bh)wq[r], (__m512bh)xq));
        }
    }
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) {
        const __m512 sv = _mm512_permutexvar_ps(
            v->sidx, _mm512_castps128_ps512(_mm_loadu_ps(sf + r * PLOW_MX_SF_STRIDE + 4u * c)));
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++)
            acc[r][m] = _mm512_fmadd_ps(
                I16 ? _mm512_cvtepi32_ps(t[r][m]) : _mm512_castsi512_ps(t[r][m]), sv, acc[r][m]);
    }
}

/* out[r*M + m] = W[r] . X[m] over K (K % 32 == 0), RB packed weight rows (stride ldw bytes, scale
 * rows stride lds bytes) x M staged activation rows (stride ldx 2-byte elements). RB/M/I16 are
 * compile-time constants at every call site so acc[][] lives in registers. */
static inline __attribute__((always_inline)) void plow_mx_dot_rm(
    const plow_mx_vlut* v, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
    const void* XP, size_t ldx, uint32_t K, const uint32_t RB, const uint32_t M, const int I16,
    float* out) {
    __m512 acc[4][8];
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_setzero_ps();
    float sf[4u * PLOW_MX_SF_STRIDE];
    const uint32_t nb = K / PLOW_MX_BLK, nc = K / PLOW_MX_CHUNK;
    for (uint32_t r = 0; r < RB; r++) plow_mx_scale_row(sf + r * PLOW_MX_SF_STRIDE, S + r * lds, nb);
    /* Packed rows are short (K/2 bytes: 1.9 KiB at K=3840), so a fixed 1 KiB lead never covers the
     * jump to the next row block and every block restarts cold (measured: cache-warm 13 GB/s vs
     * DRAM-streaming 6 per thread). A slice's rows are contiguous, so prefetching the same chunk of
     * row r+RB (one row block ahead, at least 4 KiB) keeps one continuous stream per row. */
    const size_t lead = (size_t)RB * ldw > 4096u ? (size_t)RB * ldw : 4096u;
    for (uint32_t c = 0; c < nc; c++)
        plow_mx_chunk(v, W, ldw, lead, sf, XP, ldx, c, ~(__mmask64)0, RB, M, I16, acc);
    if (nb & 3u)
        plow_mx_chunk(v, W, ldw, lead, sf, XP, ldx, nc, (__mmask64)((1ull << (16u * (nb & 3u))) - 1ull),
                      RB, M, I16, acc);
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) out[r * M + m] = _mm512_reduce_add_ps(acc[r][m]);
}

#define PLOW_MX_DOT_CASE(RB_, M_, I16_) \
    case (RB_) * 16 + (M_): plow_mx_dot_rm(v, W, ldw, S, lds, XP, ldx, K, RB_, M_, I16_, out); break;
#define PLOW_MX_DOT_CASES(I16_)                                                                   \
    PLOW_MX_DOT_CASE(4, 1, I16_) PLOW_MX_DOT_CASE(4, 2, I16_)                                     \
    PLOW_MX_DOT_CASE(2, 1, I16_) PLOW_MX_DOT_CASE(2, 2, I16_) PLOW_MX_DOT_CASE(2, 3, I16_)        \
    PLOW_MX_DOT_CASE(2, 4, I16_)                                                                  \
    PLOW_MX_DOT_CASE(1, 1, I16_) PLOW_MX_DOT_CASE(1, 2, I16_) PLOW_MX_DOT_CASE(1, 3, I16_)        \
    PLOW_MX_DOT_CASE(1, 4, I16_) PLOW_MX_DOT_CASE(1, 5, I16_) PLOW_MX_DOT_CASE(1, 6, I16_)        \
    PLOW_MX_DOT_CASE(1, 7, I16_) PLOW_MX_DOT_CASE(1, 8, I16_)                                     \
    default:                                                                                      \
        for (uint32_t r = 0; r < RB; r++)                                                         \
            for (uint32_t m = 0; m < M; m++)                                                      \
                plow_mx_dot_rm(v, W + r * ldw, ldw, S + r * lds, lds,                             \
                               (const plow_bf16*)XP + m * ldx, ldx, K, 1, 1, I16_, out + r * M + m);

/* Row block width for M activation rows (register budget: 2*RB*M + 2*RB + 3 <= 32). */
static inline uint32_t plow_mx_rb_for(uint32_t M) { return M <= 2u ? 4u : M <= 4u ? 2u : 1u; }

/* Runtime (RB, M) -> the unrolled instance. RB in {1, 2, 4}, M in 1..8. xs != NULL selects the
 * int16 path and carries the M staged-row scales (plow_mx_stage), folded in on the way out. */
static inline void plow_mx_gemv_rows(const plow_mx_vlut* v, const uint8_t* W, size_t ldw,
                                     const uint8_t* S, size_t lds, const void* XP, size_t ldx,
                                     uint32_t K, uint32_t RB, uint32_t M, const float* xs,
                                     float* out) {
    if (xs) {
        switch (RB * 16 + M) { PLOW_MX_DOT_CASES(1) }
        for (uint32_t r = 0; r < RB; r++)
            for (uint32_t m = 0; m < M; m++) out[r * M + m] *= xs[m];
        return;
    }
    switch (RB * 16 + M) { PLOW_MX_DOT_CASES(0) }
}
#endif /* AVX-512 */

#endif /* PLOW_CPU_MXFP4_COMMON_H */
