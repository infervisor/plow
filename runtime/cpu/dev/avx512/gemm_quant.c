#include "avx512.h"
#include "../mxfp4_common.h"
#include "golden/fp8.h"
#include "golden/gptoss.h"

static __m128i fp8_encode(__m512 x) {
    const __m512i bits = _mm512_castps_si512(x);
    const __m512i sign = _mm512_and_si512(_mm512_srli_epi32(bits, 24), _mm512_set1_epi32(128));
    __m512 mag = _mm512_castsi512_ps(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)));
    const __mmask16 nan = _mm512_cmp_ps_mask(mag, mag, _CMP_UNORD_Q);
    mag = _mm512_min_ps(mag, _mm512_set1_ps(448.0f));
    __m512i u = _mm512_castps_si512(mag);
    const __m512i odd = _mm512_and_si512(_mm512_srli_epi32(u, 20), _mm512_set1_epi32(1));
    u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7ffff), odd));
    __m512i code = _mm512_srli_epi32(_mm512_sub_epi32(u, _mm512_set1_epi32(0x3c000000)), 20);
    const __m512i sub = _mm512_cvt_roundps_epi32(_mm512_mul_ps(mag, _mm512_set1_ps(512.0f)),
                                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    code = _mm512_mask_mov_epi32(code, _mm512_cmp_ps_mask(mag, _mm512_set1_ps(0x1p-6f), _CMP_LT_OQ), sub);
    code = _mm512_mask_mov_epi32(code, nan, _mm512_set1_epi32(127));
    return _mm512_cvtepi32_epi8(_mm512_or_si512(code, sign));
}

V_K(v_quant_fp8) {
    if (PLOW_CPU_TEN(in, T, 3)) { g_quant_fp8(in, slice, nblk, T, ctx); return; }
    uint8_t* q = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    float* scales = PLOW_CPU_TEN(in, T, 2);
    const uint32_t K = in->i[1];
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t m = lo; m < hi; m++) {
        const size_t row = (size_t)m * K;
        __m512 amax = _mm512_setzero_ps();
        for (uint32_t k = 0; k < K; k += 16) {
            const __mmask16 mask = K - k >= 16 ? 0xffff : (1u << (K - k)) - 1;
            const __m512i b = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, x + row + k));
            __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x7fff)), 16));
            v = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(v, v, _CMP_ORD_Q), v);
            amax = _mm512_max_ps(amax, v);
        }
        scales[m] = fmaxf(_mm512_reduce_max_ps(amax) * (1.0f / 448.0f), 1e-12f);
        const __m512 inv = _mm512_set1_ps(1.0f / scales[m]);
        for (uint32_t k = 0; k < K; k += 16) {
            const __mmask16 mask = K - k >= 16 ? 0xffff : (1u << (K - k)) - 1;
            const __m512i b = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, x + row + k));
            const __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(b, 16));
            _mm_mask_storeu_epi8(q + row + k, mask, fp8_encode(_mm512_mul_ps(v, inv)));
        }
    }
}

extern plow_mx_vlut plow_v_mx_lut;

/* Exact e4m3fn decode, including signed zero, subnormals, and NaN. */
static __m512bh fp8_load(const uint8_t* p, __mmask32 mask) {
    const __m512i q = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, p));
    const __m512i mag = _mm512_and_si512(q, _mm512_set1_epi16(127));
    const __m512i sign = _mm512_slli_epi16(_mm512_and_si512(q, _mm512_set1_epi16(128)), 8);
    const __m512i sub = _mm512_setr_epi32(0x3b000000, 0x3bc03b80, 0x3c203c00, 0x3c603c40,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i v = _mm512_add_epi16(_mm512_slli_epi16(mag, 4), _mm512_set1_epi16(0x3c00));
    v = _mm512_mask_mov_epi16(v, _mm512_cmplt_epu16_mask(mag, _mm512_set1_epi16(8)),
                              _mm512_permutexvar_epi16(mag, sub));
    v = _mm512_mask_mov_epi16(v, _mm512_cmpeq_epi16_mask(mag, _mm512_set1_epi16(127)),
                              _mm512_set1_epi16(0x7fc0));
    return (__m512bh)_mm512_or_si512(v, sign);
}

static float moe_fp8_dot(const plow_bf16* x, const uint8_t* w, uint32_t K) {
    __m512 sum = _mm512_setzero_ps();
    for (uint32_t k = 0; k < K; k += 32) {
        const __mmask32 mask = plow_mx_tail32(K-k);
        const __m512bh a = (__m512bh)_mm512_maskz_loadu_epi16(mask, x+k);
        sum = _mm512_dpbf16_ps(sum, a, fp8_load(w+k, mask));
    }
    return _mm512_reduce_add_ps(sum);
}

V_K(v_moe_expert_glu_gemma_fp8) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* routes = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* weights = PLOW_CPU_TEN(in, T, 3);
    const uint64_t* scales = PLOW_CPU_TEN(in, T, 4);
    const uint32_t k = in->i[0], I = in->i[1], H = in->i[2], E = in->i[3];
    const uint32_t rows = in->i[5] ? in->i[5] : 1;
    uint32_t lo, hi; g_range(rows * k * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi; idx++) {
        const uint32_t slot = idx / I, n = idx % I, e = routes[slot].eid;
        if (e >= E || !weights[2*e] || !scales[2*e]) continue;
        const uint8_t* w = (const uint8_t*)(uintptr_t)weights[2*e];
        const float* sc = (const float*)(uintptr_t)scales[2*e];
        const plow_bf16* a = x + (size_t)(slot / k) * H;
        const float g = moe_fp8_dot(a, w + (size_t)n * H, H) * sc[n];
        const float u = moe_fp8_dot(a, w + (size_t)(I+n) * H, H) * sc[I+n];
        out[idx] = plow_f2bf(g_gelu_tanh(g) * u);
    }
}

V_K(v_moe_expert_down_gemma_fp8) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* routes = PLOW_CPU_TEN(in, T, 2);
    const uint64_t* weights = PLOW_CPU_TEN(in, T, 3);
    const uint64_t* scales = PLOW_CPU_TEN(in, T, 4);
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3];
    const uint32_t rows = in->i[5] ? in->i[5] : 1;
    uint32_t lo, hi; g_range(rows * k * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi; idx++) {
        const uint32_t slot = idx / H, h = idx % H, e = routes[slot].eid;
        out[idx] = 0;
        if (e >= E || !weights[2*e+1] || !scales[2*e+1]) continue;
        const uint8_t* w = (const uint8_t*)(uintptr_t)weights[2*e+1];
        const float* sc = (const float*)(uintptr_t)scales[2*e+1];
        out[idx] = routes[slot].gate * (moe_fp8_dot(x + (size_t)slot * I,
                                               w + (size_t)h * I, I) * sc[h]);
    }
}

static void qdots(float out[4][4], const void* A, const uint8_t* W, const uint8_t* S,
                  uint32_t K, uint32_t mr, uint32_t nr, int mx, int a8) {
    __m512 acc[4][4];
    for (uint32_t m = 0; m < mr; m++)
        for (uint32_t n = 0; n < nr; n++) acc[m][n] = _mm512_setzero_ps();
    for (uint32_t k = 0; k < K; k += 32) {
        const uint32_t left = K - k;
        const __mmask32 mask = plow_mx_tail32(left);
        __m512bh a[4];
        for (uint32_t m = 0; m < mr; m++) {
            const size_t off = (size_t)m * K + k;
            a[m] = a8 ? fp8_load((const uint8_t*)A + off, mask)
                       : (__m512bh)_mm512_maskz_loadu_epi16(mask, (const plow_bf16*)A + off);
        }
        for (uint32_t n = 0; n < nr; n++) {
            __m512bh b;
            float scale = 1.0f;
            if (mx) {
                const uint32_t bytes = left >= 32 ? 16 : (left + 1) / 2;
                const __mmask16 wm = bytes == 16 ? 0xffff : (1u << bytes) - 1;
                const __m256i q = _mm256_cvtepu8_epi16(_mm_maskz_loadu_epi8(wm, W + (size_t)n * (K / 2) + k / 2));
                const __m256i lo = _mm256_and_si256(q, _mm256_set1_epi16(15));
                const __m256i hi = _mm256_srli_epi16(q, 4);
                const __m256i l = _mm256_unpacklo_epi16(lo, hi), h = _mm256_unpackhi_epi16(lo, hi);
                __m512i ix = _mm512_castsi256_si512(_mm256_permute2x128_si256(l, h, 0x20));
                ix = _mm512_inserti64x4(ix, _mm256_permute2x128_si256(l, h, 0x31), 1);
                b = (__m512bh)_mm512_maskz_mov_epi16(mask, _mm512_permutexvar_epi16(ix, plow_v_mx_lut.lut));
                scale = plow_e8m0_to_f32(S[(size_t)n * ((K + 31) / 32) + k / 32]);
            } else {
                b = fp8_load(W + (size_t)n * K + k, mask);
            }
            for (uint32_t m = 0; m < mr; m++) {
                if (mx) {
                    const __m512 block = _mm512_dpbf16_ps(_mm512_setzero_ps(), a[m], b);
                    acc[m][n] = _mm512_add_ps(acc[m][n], _mm512_mul_ps(block, _mm512_set1_ps(scale)));
                } else acc[m][n] = _mm512_dpbf16_ps(acc[m][n], a[m], b);
            }
        }
    }
    for (uint32_t m = 0; m < mr; m++)
        for (uint32_t n = 0; n < nr; n++) out[m][n] = _mm512_reduce_add_ps(acc[m][n]);
}

static void qgemm(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T,
                  uint32_t BM, uint32_t BN, int mx, int glu) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const uint8_t* A = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2), *U = glu ? PLOW_CPU_TEN(in, T, 5) : NULL;
    const uint8_t* S = mx ? PLOW_CPU_TEN(in, T, 3) : NULL;
    const uint8_t* US = mx && glu ? PLOW_CPU_TEN(in, T, 4) : NULL;
    const float* as = mx ? NULL : PLOW_CPU_TEN(in, T, 3);
    const float* ws = mx ? NULL : PLOW_CPU_TEN(in, T, 4);
    const float* us = !mx && glu ? PLOW_CPU_TEN(in, T, 6) : NULL;
    const plow_bf16* bias = mx && !glu ? PLOW_CPU_TEN(in, T, 7) : NULL;
    const uint32_t a0 = glu ? 0 : in->i[4], c0 = glu ? 0 : in->i[5];
    const size_t ldw = mx ? K / 2 : K, lds = (K + 31) / 32;
    C += (size_t)c0 * N;
    A += (size_t)a0 * K * (as ? 1 : 2);
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = lin / tn * BM, n0 = lin % tn * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m += 4) {
            const uint32_t mr = m1 - m < 4 ? m1 - m : 4;
            const void* a = A + (size_t)m * K * (as ? 1 : 2);
            for (uint32_t n = n0; n < n1; n += 4) {
                const uint32_t nr = n1 - n < 4 ? n1 - n : 4;
                float g[4][4], u[4][4];
                qdots(g, a, W + (size_t)n * ldw, S ? S + (size_t)n * lds : NULL, K, mr, nr, mx, as != NULL);
                if (glu) qdots(u, a, U + (size_t)n * ldw, US ? US + (size_t)n * lds : NULL, K, mr, nr, mx, as != NULL);
                for (uint32_t r = 0; r < mr; r++) {
                    const float am = as ? as[a0 + m + r] : 1.0f;
                    for (uint32_t c = 0; c < nr; c++) {
                        float v = mx ? g[r][c] : g[r][c] * am * ws[n + c];
                        if (bias) v += plow_bf2f(bias[n + c]);
                        if (glu) v = mx ? g_glu_pair(v, u[r][c], in->i[5], in->fj[0].f, in->fj[1].f)
                                         : g_act_gate_only(v, in->i[5]) * (u[r][c] * am * us[n + c]);
                        C[(size_t)(m + r) * N + n + c] = plow_f2bf(v);
                    }
                }
            }
        }
    }
}

#define QGEMM(name, bm, bn, mx, glu) \
    static V_K(name) { (void)ctx; qgemm(in, slice, nblk, T, bm, bn, mx, glu); }
QGEMM(v_gemm_fp8, 256, 256, 0, 0)
QGEMM(v_gemm_small_fp8, 64, 128, 0, 0)
QGEMM(v_gemm_med_fp8, 128, 128, 0, 0)
QGEMM(v_gemm_wide_fp8, 128, 256, 0, 0)
QGEMM(v_gemm_c5_fp8, 192, 256, 0, 0)
QGEMM(v_gemm_glu_fp8, 256, 128, 0, 1)
QGEMM(v_gemm_mxfp4, 256, 256, 1, 0)
QGEMM(v_gemm_small_mxfp4, 64, 128, 1, 0)
QGEMM(v_gemm_med_mxfp4, 128, 128, 1, 0)
QGEMM(v_gemm_wide_mxfp4, 128, 256, 1, 0)
QGEMM(v_gemm_c5_mxfp4, 192, 256, 1, 0)
QGEMM(v_gemm_glu_mxfp4, 256, 128, 1, 1)

void v_register_gemm_quant(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8] = v_moe_expert_glu_gemma_fp8;
    tab[PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8] = v_moe_expert_down_gemma_fp8;
    tab[PLOW_DOP_QUANT_FP8] = v_quant_fp8;
    tab[PLOW_DOP_GEMM_FP8] = v_gemm_fp8;
    tab[PLOW_DOP_GEMM_SMALL_FP8] = v_gemm_small_fp8;
    tab[PLOW_DOP_GEMM_MED_FP8] = v_gemm_med_fp8;
    tab[PLOW_DOP_GEMM_WIDE_FP8] = v_gemm_wide_fp8;
    tab[PLOW_DOP_GEMM_C5_FP8] = v_gemm_c5_fp8;
    tab[PLOW_DOP_GEMM_GLU_FP8] = v_gemm_glu_fp8;
    tab[PLOW_DOP_GEMM_MXFP4] = v_gemm_mxfp4;
    tab[PLOW_DOP_GEMM_SMALL_MXFP4] = v_gemm_small_mxfp4;
    tab[PLOW_DOP_GEMM_MED_MXFP4] = v_gemm_med_mxfp4;
    tab[PLOW_DOP_GEMM_WIDE_MXFP4] = v_gemm_wide_mxfp4;
    tab[PLOW_DOP_GEMM_C5_MXFP4] = v_gemm_c5_mxfp4;
    tab[PLOW_DOP_GEMM_GLU_MXFP4] = v_gemm_glu_mxfp4;
}
