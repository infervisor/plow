/* golden/gptoss.c — GPT-OSS family, dense part: MXFP4 (w4a16) GEMV. f32 accumulate per 32-block,
 * block scale multiplied once per block (mxfp4_common.h), bf16 store. GV_BLOCKED column slicing
 * like g_gemv. */
#include "gptoss.h"
#include "../mxfp4_common.h"

/* t0=C t1=x t2=W(fp4) t3=S(e8m0) t7=bias?(bf16 [N])  i0=M i1=N i2=K i4=x_row0 */
G_K(g_gemv_mxfp4) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K;
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 7);
    const size_t ldw = K / 2u, lds = (K + PLOW_MX_BLK - 1u) / PLOW_MX_BLK;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = n0; n < n1; n++)
            C[(size_t)m * N + n] =
                plow_f2bf(plow_mxfp4_row_dot(W + (size_t)n * ldw, S + (size_t)n * lds,
                                             x + (size_t)m * K, K) + (bias ? plow_bf2f(bias[n]) : 0.0f));
}

/* ---- MXFP4 (w4a16) PREFILL GEMM family (93/96..99, 113) ---------------------------------------
 * Same output tiles and slice order as the bf16 family in gemm.c (SWZ=0 linear tile id); only the
 * weight fetch differs. Accumulation is f32 per 32-block with the block scale applied once, as
 * plow_mxfp4_row_dot does for the decode GEMV. */
static void gemm_mx_tiles(plow_bf16* C, const plow_bf16* A, const uint8_t* W, const uint8_t* S,
                          const plow_bf16* bias, uint32_t M, uint32_t N, uint32_t K, uint32_t BM,
                          uint32_t BN, uint32_t slice, uint32_t nblk) {
    const size_t ldw = K / 2u, lds = (K + PLOW_MX_BLK - 1u) / PLOW_MX_BLK;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++)
            for (uint32_t n = n0; n < n1; n++) {
                float acc = plow_mxfp4_row_dot(W + (size_t)n * ldw, S + (size_t)n * lds,
                                               A + (size_t)m * K, K);
                if (bias) acc += plow_bf2f(bias[n]);
                C[(size_t)m * N + n] = plow_f2bf(acc);
            }
    }
}

/* t0=C t1=A t2=W(fp4) t3=wscale(e8m0) t7=bias?  i0=M i1=N i2=K i4=a_row0 i5=c_row0 */
static void gemm_mx_op(const PlowDevInst* in, void* const* T, uint32_t BM, uint32_t BN,
                       uint32_t slice, uint32_t nblk) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    gemm_mx_tiles((plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N,
                  (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K,
                  PLOW_CPU_TEN(in, T, 2), PLOW_CPU_TEN(in, T, 3), PLOW_CPU_TEN(in, T, 7), M, N, K,
                  BM, BN, slice, nblk);
}

G_K(g_gemm_mxfp4)       { (void)ctx; gemm_mx_op(in, T, 256, 256, slice, nblk); }
G_K(g_gemm_med_mxfp4)   { (void)ctx; gemm_mx_op(in, T, 128, 128, slice, nblk); }
G_K(g_gemm_small_mxfp4) { (void)ctx; gemm_mx_op(in, T, 64, 128, slice, nblk); }
G_K(g_gemm_wide_mxfp4)  { (void)ctx; gemm_mx_op(in, T, 128, 256, slice, nblk); }
G_K(g_gemm_c5_mxfp4)    { (void)ctx; gemm_mx_op(in, T, 192, 256, slice, nblk); }

/* t0=fu t1=A t2=Wg(fp4) t5=Wu(fp4) t3=Sg t4=Su  i0=M i1=N i2=K i5=act. 256x128 output tile like
 * the bf16 GEMM_GLU (a GLU tile emits BN/2 columns). */
G_K(g_gemm_glu_mxfp4) {
    (void)ctx;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    const uint8_t* Sg = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* Su = PLOW_CPU_TEN(in, T, 4);
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t ldw = K / 2u, lds = (K + PLOW_MX_BLK - 1u) / PLOW_MX_BLK;
    const uint32_t BM = 256, BN = 128;
    const uint32_t tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * BM, n0 = (lin % tn) * BN;
        const uint32_t m1 = m0 + BM < M ? m0 + BM : M, n1 = n0 + BN < N ? n0 + BN : N;
        for (uint32_t m = m0; m < m1; m++)
            for (uint32_t n = n0; n < n1; n++) {
                const plow_bf16* a = A + (size_t)m * K;
                const float g = plow_mxfp4_row_dot(Wg + (size_t)n * ldw, Sg + (size_t)n * lds, a, K);
                const float u = plow_mxfp4_row_dot(Wu + (size_t)n * ldw, Su + (size_t)n * lds, a, K);
                C[(size_t)m * N + n] = plow_f2bf(g_glu_pair(g, u, act, f0, f1));
            }
    }
}

/* t0=C t1=x t2=Wg(fp4) t5=Wu(fp4) t3=Sg t4=Su  i0=M i1=N i2=K i5=act */
G_K(g_gemv_glu_mxfp4) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2], act = in->i[5];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* Wg = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* Wu = PLOW_CPU_TEN(in, T, 5);
    const uint8_t* Sg = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* Su = PLOW_CPU_TEN(in, T, 4);
    const size_t ldw = K / 2u, lds = (K + PLOW_MX_BLK - 1u) / PLOW_MX_BLK;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = n0; n < n1; n++) {
            const plow_bf16* xm = x + (size_t)m * K;
            const float g = plow_mxfp4_row_dot(Wg + (size_t)n * ldw, Sg + (size_t)n * lds, xm, K);
            const float u = plow_mxfp4_row_dot(Wu + (size_t)n * ldw, Su + (size_t)n * lds, xm, K);
            C[(size_t)m * N + n] = plow_f2bf(g_act_gate_only(g, act) * u);
        }
}
