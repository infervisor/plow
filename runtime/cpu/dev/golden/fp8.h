/* golden/fp8.h — scalar reference kernels for the fp8 (e4m3, w8a16 / w8a8) weight family. */
#ifndef PLOW_CPU_GOLDEN_FP8_H
#define PLOW_CPU_GOLDEN_FP8_H

#include "golden.h"

/* Decode (op 30/31): t0=C t1=x t2=W(e4m3) t5=w_scale(f32[N]) i0=M i1=N i2=K i4=a_row0;
 * GLU: t0=fu t1=x t2=Wg t3=g_scale t4=u_scale t5=Wu i5=act. i3 != 0 (AMD NRN fold) poisons. */
G_K(g_gemv_fp8);
G_K(g_gemv_glu_fp8);
/* Prefill (op 33-36, 100, 101): t0=C t1=A t2=W(e4m3) t3=a_scale? t4=w_scale i0=M i1=N i2=K
 * i4=a_row0. t3 present => w8a8: A is e4m3 with per-row scale; absent => A is bf16 (w8a16).
 * GLU: t0=fu t1=A t2=Wg t3=a_scale? t4=g_scale t5=Wu t6=u_scale i5=act. */
G_K(g_gemm_fp8);
G_K(g_gemm_med_fp8);
G_K(g_gemm_small_fp8);
G_K(g_gemm_wide_fp8);
G_K(g_gemm_c5_fp8);
G_K(g_gemm_glu_fp8);

#endif /* PLOW_CPU_GOLDEN_FP8_H */
