#include "avx512.h"

/* Keep the packet's nominal tile ownership: dependencies can release individual slices. */
static void dots(float out[4][4], const plow_bf16* A, const plow_bf16* B,
                 const float* rms, const plow_bf16* gamma, uint32_t K,
                 uint32_t mr, uint32_t nr) {
    __m512 acc[4][4];
    for (uint32_t m = 0; m < mr; m++)
        for (uint32_t n = 0; n < nr; n++) acc[m][n] = _mm512_setzero_ps();
    for (uint32_t k = 0; k < K; k += 32) {
        const uint32_t left = K - k;
        const __mmask32 mask = left >= 32 ? (__mmask32)-1 : (((uint32_t)1 << left) - 1);
        __m512bh a[4];
        for (uint32_t m = 0; m < mr; m++) {
            const plow_bf16* row = A + (size_t)m * K + k;
            if (rms) {
                const __mmask16 lo = (__mmask16)mask, hi = (__mmask16)(mask >> 16);
                const __m512 inv = _mm512_set1_ps(rms[m]);
                const __m512 x = _mm512_mul_ps(_mm512_mul_ps(v_load_bf16_mask(row, lo), inv),
                                              v_load_bf16_mask(gamma + k, lo));
                __m512 y = _mm512_setzero_ps();
                if (hi) y = _mm512_mul_ps(_mm512_mul_ps(v_load_bf16_mask(row + 16, hi), inv),
                                          v_load_bf16_mask(gamma + k + 16, hi));
                a[m] = _mm512_cvtne2ps_pbh(y, x);
            } else {
                a[m] = (__m512bh)_mm512_maskz_loadu_epi16(mask, row);
            }
        }
        for (uint32_t n = 0; n < nr; n++) {
            const __m512bh b = (__m512bh)_mm512_maskz_loadu_epi16(mask, B + (size_t)n * K + k);
            for (uint32_t m = 0; m < mr; m++) acc[m][n] = _mm512_dpbf16_ps(acc[m][n], a[m], b);
        }
    }
    for (uint32_t m = 0; m < mr; m++)
        for (uint32_t n = 0; n < nr; n++) out[m][n] = _mm512_reduce_add_ps(acc[m][n]);
}

static void gemm(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T,
                 uint32_t BM, uint32_t BN, int norm, int glu) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* B = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* U = glu ? PLOW_CPU_TEN(in, T, 5) : NULL;
    const float* rms = norm ? PLOW_CPU_TEN(in, T, 3) : NULL;
    const plow_bf16* gamma = norm ? PLOW_CPU_TEN(in, T, 4) : NULL;
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, glu ? 6 : 7);
    const plow_bf16* ubias = glu ? PLOW_CPU_TEN(in, T, 7) : NULL;
    if (!norm && !glu) {
        A += (size_t)in->i[4] * K;
        C += (size_t)in->i[5] * N;
    }
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = lin / tn * BM, n0 = lin % tn * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M;
        const uint32_t n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m += 4) {
            const uint32_t mr = m1 - m < 4 ? m1 - m : 4;
            for (uint32_t n = n0; n < n1; n += 4) {
                const uint32_t nr = n1 - n < 4 ? n1 - n : 4;
                float g[4][4], u[4][4];
                dots(g, A + (size_t)m * K, B + (size_t)n * K,
                     rms ? rms + m : NULL, gamma, K, mr, nr);
                if (glu) dots(u, A + (size_t)m * K, U + (size_t)n * K, NULL, NULL, K, mr, nr);
                for (uint32_t r = 0; r < mr; r++) {
                    for (uint32_t c = 0; c < nr; c++) {
                        float v = g[r][c] + (bias ? plow_bf2f(bias[n + c]) : 0.0f);
                        if (glu) {
                            const float up = u[r][c] + (ubias ? plow_bf2f(ubias[n + c]) : 0.0f);
                            v = g_glu_pair(v, up, in->i[5], in->fj[0].f, in->fj[1].f);
                        }
                        C[(size_t)(m + r) * N + n + c] = plow_f2bf(v);
                    }
                }
            }
        }
    }
}

#define GEMM(name, bm, bn, norm, glu) \
    static V_K(name) { (void)ctx; gemm(in, slice, nblk, T, bm, bn, norm, glu); }
GEMM(v_gemm, 256, 256, 0, 0)
GEMM(v_gemm_small, 64, 128, 0, 0)
GEMM(v_gemm_med, 128, 128, 0, 0)
GEMM(v_gemm_wide, 128, 256, 0, 0)
GEMM(v_gemm_c5, 192, 256, 0, 0)
GEMM(v_gemm_norm, 256, 256, 1, 0)
GEMM(v_gemm_glu, 256, 128, 0, 1)

static V_K(v_gemm_splitk) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    float* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1), *B = PLOW_CPU_TEN(in, T, 2);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m += 4) {
        const uint32_t mr = M - m < 4 ? M - m : 4;
        for (uint32_t n = n0; n < n1; n += 4) {
            const uint32_t nr = n1 - n < 4 ? n1 - n : 4;
            float values[4][4];
            dots(values, A + (size_t)m * K, B + (size_t)n * K, NULL, NULL, K, mr, nr);
            for (uint32_t r = 0; r < mr; r++)
                for (uint32_t c = 0; c < nr; c++) C[(size_t)(m + r) * N + n + c] = values[r][c];
        }
    }
}

void v_register_gemm(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMM_SPLITK] = v_gemm_splitk;
    tab[PLOW_DOP_GEMM] = v_gemm;
    tab[PLOW_DOP_GEMM_SMALL] = v_gemm_small;
    tab[PLOW_DOP_GEMM_MED] = v_gemm_med;
    tab[PLOW_DOP_GEMM_WIDE] = v_gemm_wide;
    tab[PLOW_DOP_GEMM_C5] = v_gemm_c5;
    tab[PLOW_DOP_GEMM_NORM] = v_gemm_norm;
    tab[PLOW_DOP_GEMM_GLU] = v_gemm_glu;
}
