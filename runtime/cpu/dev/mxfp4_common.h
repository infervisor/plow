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

/* Vector decode state. The unit of work is 64 weights = 32 packed bytes = 2 blocks: the bytes are
 * widened to 32 u16 lanes, low/high nibbles become two vpermw LUT lookups giving the 32 EVEN and
 * the 32 ODD elements as bf16, and x is staged once per row-set in the same even|odd order per
 * 64-chunk (plow_mx_stage_x) so the pair `vdpbf16ps` reduces is (2j, 2j+1) either way. f32 lane
 * l of the dot then covers elements 4l..4l+3: lanes 0..7 = block b, 8..15 = block b+1.
 * K must be <= 64 * PLOW_MX_MAX_BLOCKS / 2 (callers fall back to golden otherwise). */
typedef struct {
    __m512i lut;   /* lane i: bf16(plow_e2m1_lut[i & 15]) */
    __m512i ev;    /* vpermi2w indices: even elements of a 64-chunk */
    __m512i od;    /* odd elements */
} plow_mx_vlut;

#define PLOW_MX_PF_DIST 1024u /* bytes ahead of the current weight byte (measured; see gptoss test --bench) */
/* The RB / M loops below must fully unroll (constant trip counts after inlining) or acc[][]
 * lands on the stack behind store-forwarding chains; -O2 alone does not do it (measured: 3x). */
#define PLOW_MX_UNROLL _Pragma("GCC unroll 8")

static inline void plow_mx_vlut_init(plow_mx_vlut* v) {
    __attribute__((aligned(64))) uint16_t t[32], e[32], o[32];
    for (uint32_t i = 0; i < 32u; i++) {
        t[i] = plow_f2bf(plow_e2m1_lut[i & 15u]);
        e[i] = (uint16_t)(2u * i);
        o[i] = (uint16_t)(2u * i + 1u);
    }
    v->lut = _mm512_load_si512((const void*)t);
    v->ev = _mm512_load_si512((const void*)e);
    v->od = _mm512_load_si512((const void*)o);
}

/* Bytes needed for a staged x row of K elements (rounded up to whole 64-chunks). */
static inline uint32_t plow_mx_staged_len(uint32_t K) { return (K + 63u) & ~63u; }

/* xp[64c .. 64c+64) = even elements of x[64c ..) then the odd ones; tail zero-padded. */
static inline void plow_mx_stage_x(const plow_mx_vlut* v, plow_bf16* xp, const plow_bf16* x,
                                   uint32_t K) {
    uint32_t k = 0;
    for (; k + 64u <= K; k += 64u) {
        const __m512i a = _mm512_loadu_si512((const void*)(x + k));
        const __m512i b = _mm512_loadu_si512((const void*)(x + k + 32u));
        _mm512_storeu_si512((void*)(xp + k), _mm512_permutex2var_epi16(a, v->ev, b));
        _mm512_storeu_si512((void*)(xp + k + 32u), _mm512_permutex2var_epi16(a, v->od, b));
    }
    if (k < K) {
        const uint32_t n = K - k;
        const __mmask32 ma = n >= 32u ? 0xFFFFFFFFu : (__mmask32)((1u << n) - 1u);
        const __mmask32 mb = n >= 64u ? 0xFFFFFFFFu : n > 32u ? (__mmask32)((1u << (n - 32u)) - 1u) : 0u;
        const __m512i a = _mm512_maskz_loadu_epi16(ma, x + k);
        const __m512i b = _mm512_maskz_loadu_epi16(mb, x + k + 32u);
        _mm512_storeu_si512((void*)(xp + k), _mm512_permutex2var_epi16(a, v->ev, b));
        _mm512_storeu_si512((void*)(xp + k + 32u), _mm512_permutex2var_epi16(a, v->od, b));
    }
}

/* 32 packed bytes -> even / odd elements as bf16 (lane j = element 2j / 2j+1 of the 64).
 * vpermw reads 5 index bits and the LUT is replicated over 32 lanes, so the low nibble needs no
 * mask (bit 4 of the byte is harmless) and the high nibble is just the byte >> 4. */
static inline void plow_mx_dequant64(const plow_mx_vlut* v, const uint8_t* w, __m512i* ev,
                                     __m512i* od) {
    const __m512i b = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)w));
    *ev = _mm512_permutexvar_epi16(b, v->lut);
    *od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), v->lut);
}

/* Last 16 bytes of a row whose K % 64 == 32: the upper block decodes to +0. */
static inline void plow_mx_dequant32(const plow_mx_vlut* v, const uint8_t* w, __m512i* ev,
                                     __m512i* od) {
    const __m512i b = _mm512_cvtepu8_epi16(
        _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)w)));
    *ev = _mm512_permutexvar_epi16(b, v->lut);
    *od = _mm512_permutexvar_epi16(_mm512_srli_epi16(b, 4), v->lut);
}

/* Per-row f32 block scales for one row block (K/32 <= PLOW_MX_MAX_BLOCKS). Precomputed so the
 * inner loop applies them as embedded-broadcast FMA operands (pure loads, no shuffle-port work). */
#define PLOW_MX_MAX_BLOCKS 512u /* K <= 16384 */
static inline void plow_mx_scale_row(float* sf, const uint8_t* s, uint32_t nblk) {
    uint32_t b = 0;
    for (; b + 16u <= nblk; b += 16u)
        _mm512_storeu_ps(sf + b, _mm512_castsi512_ps(_mm512_slli_epi32(
                                     _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(s + b))), 23)));
    for (; b < nblk; b++) sf[b] = plow_e8m0_to_f32(s[b]);
}

/* out[r*M + m] = W[r] . X[m] over K (K % 32 == 0), RB packed weight rows (stride ldw bytes, scale
 * rows stride lds bytes) x M staged activation rows (stride ldx elements). RB/M are compile-time
 * constants at every call site so acc[][] lives in registers.
 * Per 64-chunk: t = dp(ev, xe) + dp(od, xo) holds block b in lanes 0..7 and block b+1 in lanes
 * 8..15; lo += t * s_b and hi += t * s_{b+1} as two broadcast FMAs, and the reduction takes the
 * low half of lo and the high half of hi -- cheaper than building a two-half scale vector on the
 * shuffle port every chunk. 2*RB*M <= 16 accumulators. */
static inline __attribute__((always_inline)) void plow_mx_dot_rm(
    const plow_mx_vlut* v, const uint8_t* W, size_t ldw, const uint8_t* S, size_t lds,
    const plow_bf16* XP, size_t ldx, uint32_t K, const uint32_t RB, const uint32_t M, float* out) {
    __m512 lo[4][8], hi[4][8];
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) lo[r][m] = hi[r][m] = _mm512_setzero_ps();
    float sf[4][PLOW_MX_MAX_BLOCKS];
    const uint32_t nb = K / PLOW_MX_BLK, nc = K / 64u;
    for (uint32_t r = 0; r < RB; r++) plow_mx_scale_row(sf[r], S + r * lds, nb);
    const __m512 zero = _mm512_setzero_ps();
    /* Packed rows are short (K/2 bytes: 1.9 KiB at K=3840), so a fixed 1 KiB lead never covers the
     * jump to the next row block and every block restarts cold (measured: cache-warm 13 GB/s vs
     * DRAM-streaming 6 per thread). A slice's rows are contiguous, so prefetching the same chunk of
     * row r+RB (one row block ahead, at least 4 KiB) keeps one continuous stream per row. */
    const size_t lead = (size_t)RB * ldw > 4096u ? (size_t)RB * ldw : 4096u;
    for (uint32_t c = 0; c < nc; c++) {
        __m512i ev[4], od[4];
        PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) {
            const uint8_t* w = W + r * ldw + (size_t)c * 32u;
            _mm_prefetch((const char*)(w + lead), _MM_HINT_T0);
            plow_mx_dequant64(v, w, &ev[r], &od[r]);
        }
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) {
            const __m512bh xe = (__m512bh)_mm512_loadu_si512((const void*)(XP + m * ldx + c * 64u));
            const __m512bh xo = (__m512bh)_mm512_loadu_si512((const void*)(XP + m * ldx + c * 64u + 32u));
            PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) {
                __m512 t = _mm512_dpbf16_ps(zero, (__m512bh)ev[r], xe);
                t = _mm512_dpbf16_ps(t, (__m512bh)od[r], xo);
                lo[r][m] = _mm512_fmadd_ps(t, _mm512_set1_ps(sf[r][2u * c]), lo[r][m]);
                hi[r][m] = _mm512_fmadd_ps(t, _mm512_set1_ps(sf[r][2u * c + 1u]), hi[r][m]);
            }
        }
    }
    if (K & 32u) {
        __m512i ev[4], od[4];
        PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) plow_mx_dequant32(v, W + r * ldw + (size_t)nc * 32u, &ev[r], &od[r]);
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++) {
            const __m512bh xe = (__m512bh)_mm512_loadu_si512((const void*)(XP + m * ldx + nc * 64u));
            const __m512bh xo = (__m512bh)_mm512_loadu_si512((const void*)(XP + m * ldx + nc * 64u + 32u));
            PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++) {
                __m512 t = _mm512_dpbf16_ps(zero, (__m512bh)ev[r], xe);
                t = _mm512_dpbf16_ps(t, (__m512bh)od[r], xo);
                lo[r][m] = _mm512_fmadd_ps(t, _mm512_set1_ps(sf[r][2u * nc]), lo[r][m]);
            }
        }
    }
    PLOW_MX_UNROLL for (uint32_t r = 0; r < RB; r++)
        PLOW_MX_UNROLL for (uint32_t m = 0; m < M; m++)
            out[r * M + m] = _mm512_reduce_add_ps(_mm512_mask_blend_ps(0xFF00, lo[r][m], hi[r][m]));
}

#define PLOW_MX_DOT_CASE(RB_, M_) \
    case (RB_) * 16 + (M_): plow_mx_dot_rm(v, W, ldw, S, lds, XP, ldx, K, RB_, M_, out); break;

/* Row block width for M activation rows (register budget: 2*RB*M + 2*RB + 4 <= 32). */
static inline uint32_t plow_mx_rb_for(uint32_t M) { return M <= 2u ? 4u : M <= 4u ? 2u : 1u; }

/* Runtime (RB, M) -> the unrolled instance. RB in {1, 2, 4}, M in 1..8. */
static inline void plow_mx_gemv_rows(const plow_mx_vlut* v, const uint8_t* W, size_t ldw,
                                     const uint8_t* S, size_t lds, const plow_bf16* XP, size_t ldx,
                                     uint32_t K, uint32_t RB, uint32_t M, float* out) {
    switch (RB * 16 + M) {
        PLOW_MX_DOT_CASE(4, 1) PLOW_MX_DOT_CASE(4, 2)
        PLOW_MX_DOT_CASE(2, 1) PLOW_MX_DOT_CASE(2, 2) PLOW_MX_DOT_CASE(2, 3) PLOW_MX_DOT_CASE(2, 4)
        PLOW_MX_DOT_CASE(1, 1) PLOW_MX_DOT_CASE(1, 2) PLOW_MX_DOT_CASE(1, 3) PLOW_MX_DOT_CASE(1, 4)
        PLOW_MX_DOT_CASE(1, 5) PLOW_MX_DOT_CASE(1, 6) PLOW_MX_DOT_CASE(1, 7) PLOW_MX_DOT_CASE(1, 8)
        default:
            for (uint32_t r = 0; r < RB; r++)
                for (uint32_t m = 0; m < M; m++)
                    plow_mx_dot_rm(v, W + r * ldw, ldw, S + r * lds, lds, XP + m * ldx, ldx, K, 1, 1,
                                   out + r * M + m);
    }
}
#endif /* AVX-512 */

#endif /* PLOW_CPU_MXFP4_COMMON_H */
