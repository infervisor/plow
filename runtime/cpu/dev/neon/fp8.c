/* neon/fp8.c — fp8 (e4m3, w8a16) decode GEMV family, NEON bf16.
 *
 * The weight row is uint8[K]; e4m3 -> bf16 is exact, so the inner loop is gemv.c's
 * `vbfdotq_f32` over 8 K per step with a 2-op dequant on the weight load: bf16 bits =
 * sign << 15 | (code & 0x7F) << 4, i.e. the raw e4m3 exponent (bias 7) sits in bf16's exponent
 * field and the value is weight * 2^-120 (avx512/fp8.c plow_fp8x32_to_bf16_raw). e == 0 codes
 * land in bf16's denormal range, which BFDOT flushes to zero. The 2^120 comes back as
 * x * 2^60 (staged once per call in scratch) and 2^60 folded into the per-output-channel f32
 * scale in the epilogue: every intermediate is an exact power-of-two scaling, so rounding is
 * unchanged. Slicing is golden's GV_BLOCKED column ownership. */
#include "neon.h"
#include "../fp8_common.h"
#include "../golden/fp8.h"

#define UNROLL _Pragma("clang loop unroll(full)")
#define X_SHIFT 60
#define OUT_SCALE 0x1p60f

/* Sign-extend the code and shift 4 (one `sshll`): bits 15..11 are the sign, 10..4 the code's
 * exponent+mantissa; the mask keeps bit 15 and the 7 magnitude bits (avx512's 0x87F0 trick). */
static inline bfloat16x8_t ld8(const uint8_t* w) {
    const int16x8_t v = vshll_n_s8(vreinterpret_s8_u8(vld1_u8(w)), 4);
    return vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_s16(v), vdupq_n_u16(0x87F0)));
}

/* xs = x * 2^X_SHIFT via the exponent field. Zero/denormal x (e == 0) stay 0; Inf/NaN pass
 * through; |x| >= 2^68 would wrap (not a real activation). */
static void scale_x(plow_bf16* xs, const plow_bf16* x, size_t n) {
    const uint16x8_t em = vdupq_n_u16(0x7F80), add = vdupq_n_u16(X_SHIFT << 7);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t v = vld1q_u16(x + i);
        const uint16x8_t e = vandq_u16(v, em);
        const uint16x8_t nz = vtstq_u16(e, e);
        const uint16x8_t sat = vceqq_u16(e, em);
        const uint16x8_t r = vbslq_u16(sat, v, vandq_u16(vaddq_u16(v, add), nz));
        vst1q_u16(xs + i, r);
    }
    for (; i < n; i++) {
        const uint16_t v = x[i], e = v & 0x7F80u;
        xs[i] = e == 0u ? 0u : (e == 0x7F80u ? v : (uint16_t)(v + (X_SHIFT << 7)));
    }
}

/* out[r*M + m] = W8[r] . X[m], RB weight rows x M activation rows (RB*M <= 16). */
static inline __attribute__((always_inline)) void dot8_rm(const uint8_t* W, size_t ldw,
                                                          const plow_bf16* X, size_t ldx,
                                                          uint32_t K, const uint32_t RB,
                                                          const uint32_t M, float* out) {
    float32x4_t acc[4][8];
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) acc[r][m] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 8 <= K; k += 8) {
        bfloat16x8_t xv[8];
        UNROLL
        for (uint32_t m = 0; m < M; m++) xv[m] = n_ldbh(X + m * ldx + k);
        UNROLL
        for (uint32_t r = 0; r < RB; r++) {
            const bfloat16x8_t wv = ld8(W + r * ldw + k);
            UNROLL
            for (uint32_t m = 0; m < M; m++) acc[r][m] = vbfdotq_f32(acc[r][m], wv, xv[m]);
        }
    }
    UNROLL
    for (uint32_t r = 0; r < RB; r++)
        UNROLL
        for (uint32_t m = 0; m < M; m++) {
            float s = n_hsum(acc[r][m]);
            for (uint32_t kk = k; kk < K; kk++) {
                const uint8_t c = W[r * ldw + kk];
                const plow_bf16 wb = (plow_bf16)(((c & 0x80u) << 8) | ((c & 0x7Fu) << 4));
                s += plow_bf2f(wb) * plow_bf2f(X[m * ldx + kk]);
            }
            out[r * M + m] = s;
        }
}

/* M = 1, 4 rows, K split in two interleaved partial sums. */
static inline __attribute__((always_inline)) void dot8_m1_r4(const uint8_t* W, size_t ldw,
                                                             const plow_bf16* x, uint32_t K,
                                                             float* out) {
    float32x4_t acc[4][2];
    UNROLL
    for (uint32_t r = 0; r < 4; r++) acc[r][0] = acc[r][1] = vdupq_n_f32(0);
    uint32_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const bfloat16x8_t x0 = n_ldbh(x + k), x1 = n_ldbh(x + k + 8);
        UNROLL
        for (uint32_t r = 0; r < 4; r++) {
            const uint8_t* w = W + r * ldw + k;
            acc[r][0] = vbfdotq_f32(acc[r][0], ld8(w), x0);
            acc[r][1] = vbfdotq_f32(acc[r][1], ld8(w + 8), x1);
        }
    }
    for (; k + 8 <= K; k += 8) {
        const bfloat16x8_t x0 = n_ldbh(x + k);
        UNROLL
        for (uint32_t r = 0; r < 4; r++) acc[r][0] = vbfdotq_f32(acc[r][0], ld8(W + r * ldw + k), x0);
    }
    UNROLL
    for (uint32_t r = 0; r < 4; r++) {
        float s = n_hsum(vaddq_f32(acc[r][0], acc[r][1]));
        for (uint32_t kk = k; kk < K; kk++) {
            const uint8_t c = W[r * ldw + kk];
            const plow_bf16 wb = (plow_bf16)(((c & 0x80u) << 8) | ((c & 0x7Fu) << 4));
            s += plow_bf2f(wb) * plow_bf2f(x[kk]);
        }
        out[r] = s;
    }
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

/* x * 2^X_SHIFT staged in scratch; NULL when the ctx cannot hold M*K bf16 (-> golden). */
static plow_bf16* stage_x(PlowCpuCtx* ctx, const plow_bf16* x, uint32_t M, uint32_t K) {
    const size_t need = (size_t)M * K * sizeof(plow_bf16);
    if (M == 0u || M > 8u || !ctx || !ctx->scratch || ctx->scratch_bytes < need) return NULL;
    scale_x(ctx->scratch, x, (size_t)M * K);
    return ctx->scratch;
}

/* t0=C t1=x t2=W(e4m3) t5=w_scale i0=M i1=N i2=K i4=a_row0 (i3 NRN fold -> golden poison). */
N_K(n_gemv_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const float* ws = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const plow_bf16* xs = in->i[3] == 0u && ws ? stage_x(ctx, x, M, K) : NULL;
    if (!xs) {
        g_gemv_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = rb8_for(M);
    float out[4 * 8];
    uint32_t n = n0;
    for (; n + RB <= n1; n += RB) {
        gemv8_rows(W + (size_t)n * K, K, xs, K, K, RB, M, out);
        for (uint32_t r = 0; r < RB; r++) {
            const float sc = ws[n + r] * OUT_SCALE;
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(out[r * M + m] * sc);
        }
    }
    for (; n < n1; n++) {
        gemv8_rows(W + (size_t)n * K, K, xs, K, K, 1, M, out);
        const float sc = ws[n] * OUT_SCALE;
        for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n] = plow_f2bf(out[m] * sc);
    }
}

/* t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i0=M i1=N i2=K i5=act:
 * fu = act(g * gs[n]) * (u * us[n]). */
N_K(n_gemv_glu_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float* gs = PLOW_CPU_TEN(in, T, 3);
    const float* us = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* xs = act <= 1u && gs && us ? stage_x(ctx, x, M, K) : NULL;
    if (!xs) {
        g_gemv_glu_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = rb8_for(M);
    float g[4 * 8], u[4 * 8];
    for (uint32_t n = n0; n < n1;) {
        const uint32_t rb = n + RB <= n1 ? RB : 1u;
        gemv8_rows(Wg + (size_t)n * K, K, xs, K, K, rb, M, g);
        gemv8_rows(Wu + (size_t)n * K, K, xs, K, K, rb, M, u);
        for (uint32_t r = 0; r < rb; r++) {
            const float gsc = gs[n + r] * OUT_SCALE, usc = us[n + r] * OUT_SCALE;
            for (uint32_t m = 0; m < M; m++) {
                const float gv = g[r * M + m] * gsc, uv = u[r * M + m] * usc;
                const float a = act == 1u ? g_silu(gv) : g_gelu_tanh(gv);
                C[(size_t)m * N + n + r] = plow_f2bf(a * uv);
            }
        }
        n += rb;
    }
}

/* --- prefill: w8a16 GEMM family --------------------------------------------------------
 * The weight tile [n0, n1) x K is dequantized ONCE per tile into scratch as exact bf16 (the
 * re-biased form: + (120 << 7) on every code with a nonzero exponent; e == 0 codes decode to 0
 * like avx512's plow_fp8x32_to_bf16), then gemm.c's bf16 micro-kernel runs over it with the
 * per-output-channel scale in the epilogue. w8a8 packets (t3 = a_scale present) go to golden. */

static void dequant_rows(plow_bf16* dst, const uint8_t* W, uint32_t rows, uint32_t K) {
    const uint16x8_t mask = vdupq_n_u16(0x87F0), emask = vdupq_n_u16(0x0780), bias = vdupq_n_u16(120u << 7);
    const size_t n = (size_t)rows * K;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t t = vandq_u16(vreinterpretq_u16_s16(vshll_n_s8(vreinterpret_s8_u8(vld1_u8(W + i)), 4)), mask);
        const uint16x8_t nrm = vtstq_u16(t, emask);
        vst1q_u16(dst + i, vaddq_u16(t, vandq_u16(bias, nrm)));
    }
    for (; i < n; i++) dst[i] = plow_f2bf(plow_e4m3_to_f32(W[i]));
}

/* t0=C t1=A t2=W(e4m3) t3=a_scale? t4=w_scale i0=M i1=N i2=K i4=a_row0 i5=c_row0 */
static void gemm_fp8_op(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T,
                        PlowCpuCtx* ctx, uint32_t BM, uint32_t BN, plow_cpu_kernel_fn golden) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const float* as = PLOW_CPU_TEN(in, T, 3);
    const float* ws = PLOW_CPU_TEN(in, T, 4);
    const size_t need = (size_t)BN * K * sizeof(plow_bf16);
    if (as || !ws || !ctx || !ctx->scratch || ctx->scratch_bytes < need) {
        golden(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N;
    const plow_bf16* A = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    plow_bf16* wt = ctx->scratch;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    uint32_t cur_n0 = UINT32_MAX;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        if (n0 != cur_n0) {
            dequant_rows(wt, W + (size_t)n0 * K, n1 - n0, K);
            cur_n0 = n0;
        }
        n_gemm_tile(C, N, A, wt, n0, NULL, NULL, ws, K, m0, m1, n0, n1);
    }
}

N_K(n_gemm_fp8)       { gemm_fp8_op(in, slice, nblk, T, ctx, 256, 256, g_gemm_fp8); }
N_K(n_gemm_med_fp8)   { gemm_fp8_op(in, slice, nblk, T, ctx, 128, 128, g_gemm_med_fp8); }
N_K(n_gemm_small_fp8) { gemm_fp8_op(in, slice, nblk, T, ctx, 64, 128, g_gemm_small_fp8); }
N_K(n_gemm_wide_fp8)  { gemm_fp8_op(in, slice, nblk, T, ctx, 128, 256, g_gemm_wide_fp8); }
N_K(n_gemm_c5_fp8)    { gemm_fp8_op(in, slice, nblk, T, ctx, 192, 256, g_gemm_c5_fp8); }

/* t0=fu t1=A t2=Wg t3=a_scale? t4=g_scale t5=Wu t6=u_scale i0=M i1=N i2=K i5=act; 256x128 tile. */
N_K(n_gemm_glu_fp8) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float* as = PLOW_CPU_TEN(in, T, 3);
    const float* gs = PLOW_CPU_TEN(in, T, 4);
    const float* us = PLOW_CPU_TEN(in, T, 6);
    const uint32_t BM = 256, BN = 128;
    const size_t need = (size_t)2 * BN * K * sizeof(plow_bf16);
    if (as || !gs || !us || act > 1u || !ctx || !ctx->scratch || ctx->scratch_bytes < need) {
        g_gemm_glu_fp8(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    plow_bf16* wg = ctx->scratch;
    plow_bf16* wu = wg + (size_t)BN * K;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    uint32_t cur_n0 = UINT32_MAX;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        if (n0 != cur_n0) {
            dequant_rows(wg, Wg + (size_t)n0 * K, n1 - n0, K);
            dequant_rows(wu, Wu + (size_t)n0 * K, n1 - n0, K);
            cur_n0 = n0;
        }
        n_gemm_glu_tile(C, N, A, wg, wu, n0, NULL, NULL, gs, us, K, m0, m1, n0, n1, act, 0.0f, 0.0f);
    }
}

void plow_cpu_register_neon_fp8(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMV_FP8] = n_gemv_fp8;
    tab[PLOW_DOP_GEMV_GLU_FP8] = n_gemv_glu_fp8;
    tab[PLOW_DOP_GEMM_FP8] = n_gemm_fp8;
    tab[PLOW_DOP_GEMM_MED_FP8] = n_gemm_med_fp8;
    tab[PLOW_DOP_GEMM_SMALL_FP8] = n_gemm_small_fp8;
    tab[PLOW_DOP_GEMM_WIDE_FP8] = n_gemm_wide_fp8;
    tab[PLOW_DOP_GEMM_C5_FP8] = n_gemm_c5_fp8;
    tab[PLOW_DOP_GEMM_GLU_FP8] = n_gemm_glu_fp8;
}
