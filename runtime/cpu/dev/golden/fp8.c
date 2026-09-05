/* golden/fp8.c — fp8 (e4m3) weight family: f32 accumulate over K in order, per-output-channel
 * scale applied once in the epilogue (op_gemm.h d_gemv_fp8 / op_gemm.cuh w8a16), bf16 store.
 * Slicing mirrors the bf16 twins in gemm.c (GV_BLOCKED columns for GEMV, nominal output tiles
 * in SWZ=0 order for GEMM). */
#include "fp8.h"
#include "../fp8_common.h"

/* A row m (bf16, or e4m3 with a_scale) dotted with e4m3 weight row w. */
static float dot_a_w8(const plow_bf16* a16, const uint8_t* a8, const uint8_t* w, uint32_t K) {
    float acc = 0.0f;
    if (a8) {
        for (uint32_t k = 0; k < K; k++) acc += plow_e4m3_to_f32(a8[k]) * plow_e4m3_to_f32(w[k]);
    } else {
        for (uint32_t k = 0; k < K; k++) acc += plow_bf2f(a16[k]) * plow_e4m3_to_f32(w[k]);
    }
    return acc;
}

/* t0=C t1=x t2=W t5=w_scale i0=M i1=N i2=K i4=a_row0 i3=NRN fold (unsupported -> poison). */
G_K(g_gemv_fp8) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const float* ws = PLOW_CPU_TEN(in, T, 5);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    if (in->i[3] != 0u || !ws) {
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = n0; n < n1; n++) C[(size_t)m * N + n] = G_QNAN;
        return;
    }
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = n0; n < n1; n++)
            C[(size_t)m * N + n] =
                plow_f2bf(dot_a_w8(x + (size_t)m * K, NULL, W + (size_t)n * K, K) * ws[n]);
}

/* t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i0=M i1=N i2=K i5=act */
G_K(g_gemv_glu_fp8) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    const float* gs = PLOW_CPU_TEN(in, T, 3);
    const float* us = PLOW_CPU_TEN(in, T, 4);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = n0; n < n1; n++) {
            const float g = dot_a_w8(x + (size_t)m * K, NULL, Wg + (size_t)n * K, K) * gs[n];
            const float u = dot_a_w8(x + (size_t)m * K, NULL, Wu + (size_t)n * K, K) * us[n];
            C[(size_t)m * N + n] = plow_f2bf(g_act_gate_only(g, act) * u);
        }
}

/* C[M,N] tiles of (BM, BN); acc * a_scale[m]? * w_scale[n] -> bf16. */
static void gemm_tiles_fp8(const PlowDevInst* in, void* const* T, uint32_t BM, uint32_t BN,
                           uint32_t slice, uint32_t nblk) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N;
    const void* A = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const float* as = PLOW_CPU_TEN(in, T, 3);
    const float* ws = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* a16 = as ? NULL : (const plow_bf16*)A + (size_t)in->i[4] * K;
    const uint8_t* a8 = as ? (const uint8_t*)A + (size_t)in->i[4] * K : NULL;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++) {
            const float am = as ? as[in->i[4] + m] : 1.0f;
            for (uint32_t n = n0; n < n1; n++) {
                const float acc = dot_a_w8(a16 ? a16 + (size_t)m * K : NULL,
                                           a8 ? a8 + (size_t)m * K : NULL, W + (size_t)n * K, K);
                C[(size_t)m * N + n] = plow_f2bf(acc * am * ws[n]);
            }
        }
    }
}

G_K(g_gemm_fp8)       { (void)ctx; gemm_tiles_fp8(in, T, 256, 256, slice, nblk); }
G_K(g_gemm_med_fp8)   { (void)ctx; gemm_tiles_fp8(in, T, 128, 128, slice, nblk); }
G_K(g_gemm_small_fp8) { (void)ctx; gemm_tiles_fp8(in, T, 64, 128, slice, nblk); }
G_K(g_gemm_wide_fp8)  { (void)ctx; gemm_tiles_fp8(in, T, 128, 256, slice, nblk); }
G_K(g_gemm_c5_fp8)    { (void)ctx; gemm_tiles_fp8(in, T, 192, 256, slice, nblk); }

/* t0=fu t1=A t2=Wg t3=a_scale? t4=g_scale t5=Wu t6=u_scale i0=M i1=N i2=K i5=act; 256x128 tile. */
G_K(g_gemm_glu_fp8) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const void* A = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    const float* as = PLOW_CPU_TEN(in, T, 3);
    const float* gs = PLOW_CPU_TEN(in, T, 4);
    const float* us = PLOW_CPU_TEN(in, T, 6);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const uint32_t BM = 256, BN = 128;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++) {
            const plow_bf16* a16 = as ? NULL : (const plow_bf16*)A + (size_t)m * K;
            const uint8_t* a8 = as ? (const uint8_t*)A + (size_t)m * K : NULL;
            const float am = as ? as[m] : 1.0f;
            for (uint32_t n = n0; n < n1; n++) {
                const float g = dot_a_w8(a16, a8, Wg + (size_t)n * K, K) * am * gs[n];
                const float u = dot_a_w8(a16, a8, Wu + (size_t)n * K, K) * am * us[n];
                C[(size_t)m * N + n] = plow_f2bf(g_act_gate_only(g, act) * u);
            }
        }
    }
}

void plow_cpu_register_golden_fp8(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMV_FP8] = g_gemv_fp8;
    tab[PLOW_DOP_GEMV_GLU_FP8] = g_gemv_glu_fp8;
    tab[PLOW_DOP_GEMM_FP8] = g_gemm_fp8;
    tab[PLOW_DOP_GEMM_MED_FP8] = g_gemm_med_fp8;
    tab[PLOW_DOP_GEMM_SMALL_FP8] = g_gemm_small_fp8;
    tab[PLOW_DOP_GEMM_WIDE_FP8] = g_gemm_wide_fp8;
    tab[PLOW_DOP_GEMM_C5_FP8] = g_gemm_c5_fp8;
    tab[PLOW_DOP_GEMM_GLU_FP8] = g_gemm_glu_fp8;
}
