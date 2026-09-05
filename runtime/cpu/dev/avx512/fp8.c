/* avx512/fp8.c — fp8 (e4m3, w8a16) decode GEMV family, AVX-512 BF16.
 *
 * The weight row is uint8[K]; e4m3 -> bf16 is exact, so the inner loop is gemv.c's
 * `vdpbf16ps` over 32 K per step with a LUT dequant (two vpermi2w + blend, fp8_common.h) on
 * the weight load, RB weight rows x M activation rows of accumulators, `prefetcht0` 512 B
 * ahead (spec §2.2 / §6.6; half the bytes per output vs bf16 -> ~2x the decode roofline).
 * Per-output-channel f32 scale once in the epilogue (op_gemm.h d_gemv_fp8). Slicing is
 * golden's GV_BLOCKED column ownership. */
#include "avx512.h"
#include "../fp8_common.h"
#include "../golden/fp8.h"

#define PF8_DIST 512u /* bytes: 512 B ahead of the current weight byte */

static plow_fp8_vlut g_lut;

static inline __m512bh ld8(const uint8_t* w) {
    return (__m512bh)plow_fp8x32_to_bf16(&g_lut, _mm256_loadu_si256((const __m256i*)w));
}
static inline __m512bh ld8_mask(const uint8_t* w, __mmask32 m) {
    return (__m512bh)plow_fp8x32_to_bf16(&g_lut, _mm256_maskz_loadu_epi8(m, w));
}
static inline __m512bh ldxv(const plow_bf16* p) { return (__m512bh)_mm512_loadu_si512(p); }
static inline __m512bh ldxv_mask(const plow_bf16* p, __mmask32 m) {
    return (__m512bh)_mm512_maskz_loadu_epi16(m, p);
}

/* out[r*M + m] = W8[r] . X[m], RB weight rows x M activation rows (RB*M <= 16). */
static inline __attribute__((always_inline)) void dot8_rm(const uint8_t* W, size_t ldw,
                                                          const plow_bf16* X, size_t ldx,
                                                          uint32_t K, const uint32_t RB,
                                                          const uint32_t M, float* out) {
    __m512 acc[4][8];
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_setzero_ps();
    uint32_t k = 0;
    for (; k + 32 <= K; k += 32) {
        __m512bh xv[8];
        for (uint32_t m = 0; m < M; m++) xv[m] = ldxv(X + m * ldx + k);
        for (uint32_t r = 0; r < RB; r++) {
            const uint8_t* w = W + r * ldw + k;
            _mm_prefetch((const char*)(w + PF8_DIST), _MM_HINT_T0);
            const __m512bh wv = ld8(w);
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_dpbf16_ps(acc[r][m], wv, xv[m]);
        }
    }
    if (k < K) {
        const __mmask32 mk = (__mmask32)((1u << (K - k)) - 1u);
        __m512bh xv[8];
        for (uint32_t m = 0; m < M; m++) xv[m] = ldxv_mask(X + m * ldx + k, mk);
        for (uint32_t r = 0; r < RB; r++) {
            const __m512bh wv = ld8_mask(W + r * ldw + k, mk);
            for (uint32_t m = 0; m < M; m++) acc[r][m] = _mm512_dpbf16_ps(acc[r][m], wv, xv[m]);
        }
    }
    for (uint32_t r = 0; r < RB; r++)
        for (uint32_t m = 0; m < M; m++) out[r * M + m] = _mm512_reduce_add_ps(acc[r][m]);
}

/* M = 1: 4 weight rows, K split in two interleaved chains -> 8 accumulators (spec §2.2). */
static inline __attribute__((always_inline)) void dot8_m1_r4(const uint8_t* W, size_t ldw,
                                                             const plow_bf16* x, uint32_t K,
                                                             float* out) {
    __m512 acc[4][2];
    for (uint32_t r = 0; r < 4; r++) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
    uint32_t k = 0;
    for (; k + 64 <= K; k += 64) {
        const __m512bh x0 = ldxv(x + k), x1 = ldxv(x + k + 32);
        for (uint32_t r = 0; r < 4; r++) {
            const uint8_t* w = W + r * ldw + k;
            _mm_prefetch((const char*)(w + PF8_DIST), _MM_HINT_T0);
            acc[r][0] = _mm512_dpbf16_ps(acc[r][0], ld8(w), x0);
            acc[r][1] = _mm512_dpbf16_ps(acc[r][1], ld8(w + 32), x1);
        }
    }
    for (; k + 32 <= K; k += 32) {
        const __m512bh x0 = ldxv(x + k);
        for (uint32_t r = 0; r < 4; r++) acc[r][0] = _mm512_dpbf16_ps(acc[r][0], ld8(W + r * ldw + k), x0);
    }
    if (k < K) {
        const __mmask32 mk = (__mmask32)((1u << (K - k)) - 1u);
        const __m512bh x0 = ldxv_mask(x + k, mk);
        for (uint32_t r = 0; r < 4; r++)
            acc[r][1] = _mm512_dpbf16_ps(acc[r][1], ld8_mask(W + r * ldw + k, mk), x0);
    }
    for (uint32_t r = 0; r < 4; r++) out[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc[r][0], acc[r][1]));
}

#define DOT8_CASE(RB_, M_) \
    case (RB_) * 16 + (M_): dot8_rm(W, ldw, X, ldx, K, RB_, M_, out); break;

static inline uint32_t rb8_for(uint32_t M) { return M <= 4u ? 4u : 2u; }

static void gemv8_rows(const uint8_t* W, size_t ldw, const plow_bf16* X, size_t ldx, uint32_t K,
                       uint32_t RB, uint32_t M, float* out) {
    if (RB == 4u && M == 1u) { dot8_m1_r4(W, ldw, X, K, out); return; }
    switch (RB * 16 + M) {
        DOT8_CASE(4, 2) DOT8_CASE(4, 3) DOT8_CASE(4, 4)
        DOT8_CASE(2, 5) DOT8_CASE(2, 6) DOT8_CASE(2, 7) DOT8_CASE(2, 8)
        DOT8_CASE(1, 1) DOT8_CASE(1, 2) DOT8_CASE(1, 3) DOT8_CASE(1, 4)
        DOT8_CASE(1, 5) DOT8_CASE(1, 6) DOT8_CASE(1, 7) DOT8_CASE(1, 8)
        default:
            for (uint32_t r = 0; r < RB; r++)
                for (uint32_t m = 0; m < M; m++)
                    dot8_rm(W + r * ldw, ldw, X + m * ldx, ldx, K, 1, 1, out + r * M + m);
    }
}

/* t0=C t1=x t2=W(e4m3) t5=w_scale i0=M i1=N i2=K i4=a_row0 (i3 NRN fold -> golden poison). */
V_K(v_gemv_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const float* ws = PLOW_CPU_TEN(in, T, 5);
    if (M == 0u || M > 8u || in->i[3] != 0u || !ws) {
        g_gemv_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = rb8_for(M);
    float out[4 * 8];
    uint32_t n = n0;
    for (; n + RB <= n1; n += RB) {
        gemv8_rows(W + (size_t)n * K, K, x, K, K, RB, M, out);
        for (uint32_t r = 0; r < RB; r++)
            for (uint32_t m = 0; m < M; m++)
                C[(size_t)m * N + n + r] = plow_f2bf(out[r * M + m] * ws[n + r]);
    }
    for (; n < n1; n++) {
        gemv8_rows(W + (size_t)n * K, K, x, K, K, 1, M, out);
        for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n] = plow_f2bf(out[m] * ws[n]);
    }
}

/* t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i0=M i1=N i2=K i5=act:
 * fu = act(g * gs[n]) * (u * us[n]). Gate and up rows of one column are dotted together
 * (RB = 2 streams x M rows). */
V_K(v_gemv_glu_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float* gs = PLOW_CPU_TEN(in, T, 3);
    const float* us = PLOW_CPU_TEN(in, T, 4);
    if (M == 0u || M > 8u || act > 1u || !gs || !us) {
        g_gemv_glu_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = rb8_for(M);
    float g[4 * 8], u[4 * 8];
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rb = n + RB <= n1 ? RB : 1u;
        gemv8_rows(Wg + (size_t)n * K, K, x, K, K, rb, M, g);
        gemv8_rows(Wu + (size_t)n * K, K, x, K, K, rb, M, u);
        for (uint32_t r = 0; r < rb; r++)
            for (uint32_t m = 0; m < M; m++) {
                const float gv = g[r * M + m] * gs[n + r], uv = u[r * M + m] * us[n + r];
                const float a = act == 1u ? g_silu(gv) : g_gelu_tanh(gv);
                C[(size_t)m * N + n + r] = plow_f2bf(a * uv);
            }
        n += rb;
    }
}

void plow_cpu_register_avx512_fp8(plow_cpu_kernel_fn* tab) {
    plow_fp8_vlut_init(&g_lut);
    tab[PLOW_DOP_GEMV_FP8] = v_gemv_fp8;
    tab[PLOW_DOP_GEMV_GLU_FP8] = v_gemv_glu_fp8;
}
