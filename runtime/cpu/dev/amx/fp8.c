/* amx/fp8.c — fp8 (e4m3, w8a16) prefill GEMM family on the AMX tile driver: the 32-column B
 * strip is dequantized to bf16 while it is packed (exact), TDPBF16PS is unchanged, and the
 * per-output-channel scale is applied in the interleaved epilogue (spec §6.5). w8a8 packets
 * (a_scale present, A e4m3) fall back to golden. */
#include "cpu_dev_internal.h"
#include "amx_common.h"
#include "golden/fp8.h"

#define X8_K(name) \
    static void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* t0=C t1=A(bf16) t2=W(e4m3) t3=a_scale? t4=w_scale i0=M i1=N i2=K i4=a_row0 i5=c_row0 */
static void gemm_fp8_op(const PlowDevInst* in, void* const* T, PlowCpuCtx* ctx, uint32_t BM,
                        uint32_t BN, uint32_t slice, uint32_t nblk,
                        void (*fallback)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*)) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    if (!plow_amx_usable(ctx, K) || PLOW_CPU_TEN(in, T, 3) || !PLOW_CPU_TEN(in, T, 4)) {
        fallback(in, slice, nblk, T, ctx);
        return;
    }
    gemm_args g = {.C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N,
                   .A = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K,
                   .Wq = PLOW_CPU_TEN(in, T, 2),
                   .ws = PLOW_CPU_TEN(in, T, 4),
                   .M = M, .N = N, .K = K, .BM = BM, .BN = BN};
    plow_amx_run_tiles(&g, slice, nblk, ctx);
}

X8_K(x_gemm_fp8)       { gemm_fp8_op(in, T, ctx, 256, 256, slice, nblk, g_gemm_fp8); }
X8_K(x_gemm_med_fp8)   { gemm_fp8_op(in, T, ctx, 128, 128, slice, nblk, g_gemm_med_fp8); }
X8_K(x_gemm_small_fp8) { gemm_fp8_op(in, T, ctx, 64, 128, slice, nblk, g_gemm_small_fp8); }
X8_K(x_gemm_wide_fp8)  { gemm_fp8_op(in, T, ctx, 128, 256, slice, nblk, g_gemm_wide_fp8); }
X8_K(x_gemm_c5_fp8)    { gemm_fp8_op(in, T, ctx, 192, 256, slice, nblk, g_gemm_c5_fp8); }

/* t0=fu t1=A t2=Wg t3=a_scale? t4=g_scale t5=Wu t6=u_scale i0=M i1=N i2=K i5=act; 256x128. */
X8_K(x_gemm_glu_fp8) {
    const uint32_t K = in->i[2];
    if (!plow_amx_usable(ctx, K) || in->i[5] > 1u || PLOW_CPU_TEN(in, T, 3) ||
        !PLOW_CPU_TEN(in, T, 4) || !PLOW_CPU_TEN(in, T, 6)) {
        g_gemm_glu_fp8(in, slice, nblk, T, ctx);
        return;
    }
    gemm_args g = {.C = PLOW_CPU_TEN(in, T, 0), .A = PLOW_CPU_TEN(in, T, 1),
                   .Wq = PLOW_CPU_TEN(in, T, 2), .Wuq = PLOW_CPU_TEN(in, T, 5),
                   .ws = PLOW_CPU_TEN(in, T, 4), .us = PLOW_CPU_TEN(in, T, 6),
                   .M = in->i[0], .N = in->i[1], .K = K, .BM = 256, .BN = 128, .act = in->i[5]};
    plow_amx_run_tiles(&g, slice, nblk, ctx);
}

void plow_cpu_register_amx_fp8(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_GEMM_FP8] = x_gemm_fp8;
    tab[PLOW_DOP_GEMM_MED_FP8] = x_gemm_med_fp8;
    tab[PLOW_DOP_GEMM_SMALL_FP8] = x_gemm_small_fp8;
    tab[PLOW_DOP_GEMM_WIDE_FP8] = x_gemm_wide_fp8;
    tab[PLOW_DOP_GEMM_C5_FP8] = x_gemm_c5_fp8;
    tab[PLOW_DOP_GEMM_GLU_FP8] = x_gemm_glu_fp8;
}
