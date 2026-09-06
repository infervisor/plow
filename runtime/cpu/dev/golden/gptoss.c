/* golden/gptoss.c — GPT-OSS family, dense part: MXFP4 (w4a16) GEMV. f32 accumulate per 32-block,
 * block scale multiplied once per block (mxfp4_common.h), bf16 store. GV_BLOCKED column slicing
 * like g_gemv. */
#include "gptoss.h"
#include "../mxfp4_common.h"

/* t0=C t1=x t2=W(fp4) t3=S(e8m0)  i0=M i1=N i2=K */
G_K(g_gemv_mxfp4) {
    (void)ctx;
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const size_t ldw = K / 2u, lds = (K + PLOW_MX_BLK - 1u) / PLOW_MX_BLK;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = n0; n < n1; n++)
            C[(size_t)m * N + n] =
                plow_f2bf(plow_mxfp4_row_dot(W + (size_t)n * ldw, S + (size_t)n * lds,
                                             x + (size_t)m * K, K));
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
