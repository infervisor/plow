/* avx512/gptoss.c — GPT-OSS family, dense part: MXFP4 (w4a16) GEMV, AVX-512 BF16.
 *
 * Inner loop = mxfp4_common.h plow_mx_dot_rm: 32 packed bytes (2 blocks) -> two vpermw LUT
 * lookups (even / odd elements as bf16) -> two vdpbf16ps against the even|odd-staged x -> one
 * fmadd with the two block scales. RB weight rows x M activation rows of accumulators, x staged
 * once per call in scratch. Slicing is golden's GV_BLOCKED column ownership. */
#include "avx512.h"
#include "../mxfp4_common.h"
#include "../golden/gptoss.h"

plow_mx_vlut plow_v_mx_lut;

/* t0=C t1=x t2=W(fp4) t3=S(e8m0)  i0=M i1=N i2=K */
V_K(v_gemv_mxfp4) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    const size_t ldx = plow_mx_staged_len(K);
    const size_t need = (size_t)M * ldx * sizeof(plow_bf16);
    if (M == 0u || M > 8u || (K & 31u) || K / PLOW_MX_BLK > PLOW_MX_MAX_BLOCKS || !ctx || !ctx->scratch || ctx->scratch_bytes < need) {
        g_gemv_mxfp4(in, slice, nblk, T, ctx);
        return;
    }
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const size_t ldw = K / 2u, lds = K / PLOW_MX_BLK;
    plow_bf16* XP = ctx->scratch;
    for (uint32_t m = 0; m < M; m++) plow_mx_stage_x(&plow_v_mx_lut, XP + m * ldx, x + (size_t)m * K, K);
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    const uint32_t RB = plow_mx_rb_for(M);
    float out[4 * 8];
    uint32_t n = n0;
    for (; n + RB <= n1; n += RB) {
        plow_mx_gemv_rows(&plow_v_mx_lut, W + (size_t)n * ldw, ldw, S + (size_t)n * lds, lds, XP, ldx, K, RB, M, out);
        for (uint32_t r = 0; r < RB; r++)
            for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n + r] = plow_f2bf(out[r * M + m]);
    }
    for (; n < n1; n++) {
        plow_mx_gemv_rows(&plow_v_mx_lut, W + (size_t)n * ldw, ldw, S + (size_t)n * lds, lds, XP, ldx, K, 1, M, out);
        for (uint32_t m = 0; m < M; m++) C[(size_t)m * N + n] = plow_f2bf(out[m]);
    }
}

void v_register_gptoss(plow_cpu_kernel_fn* tab) {
    plow_mx_vlut_init(&plow_v_mx_lut);
    tab[PLOW_DOP_GEMV_MXFP4] = v_gemv_mxfp4;
    tab[PLOW_DOP_MOE_GLU_MX] = v_moe_glu_mx;
    tab[PLOW_DOP_MOE_DOWN_MX] = v_moe_down_mx;
    tab[PLOW_DOP_MOE_GLU_MX_PF] = v_moe_glu_mx_pf;
    tab[PLOW_DOP_MOE_DOWN_MX_PF] = v_moe_down_mx_pf;
}
