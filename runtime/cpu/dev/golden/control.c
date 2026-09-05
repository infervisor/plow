#include "golden.h"

G_K(g_nop) {
    (void)in; (void)slice; (void)nblk; (void)T; (void)ctx;
}

void plow_cpu_register_golden(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_NOP] = g_nop;
    tab[PLOW_DOP_RESIDUAL] = g_residual;
    tab[PLOW_DOP_GLU] = g_glu;
    tab[PLOW_DOP_SOFTCAP] = g_softcap;
    tab[PLOW_DOP_EMBED] = g_embed;
    tab[PLOW_DOP_ARGMAX] = g_argmax;
    tab[PLOW_DOP_ARGMAX_FIN] = g_argmax_fin;
    tab[PLOW_DOP_RMSNORM] = g_rmsnorm;
    tab[PLOW_DOP_ROWRMS] = g_rowrms;
    tab[PLOW_DOP_LAYERNORM] = g_layernorm;
    tab[PLOW_DOP_HEADNORM_ROPE] = g_headnorm_rope;
    tab[PLOW_DOP_NORM_RESIDUAL] = g_norm_residual;
    tab[PLOW_DOP_ADD_NORM] = g_add_norm;
    tab[PLOW_DOP_NORM_RESIDUAL_NORM] = g_norm_residual_norm;
    tab[PLOW_DOP_GEMM] = g_gemm;
    tab[PLOW_DOP_GEMM_SMALL] = g_gemm_small;
    tab[PLOW_DOP_GEMM_MED] = g_gemm_med;
    tab[PLOW_DOP_GEMM_WIDE] = g_gemm_wide;
    tab[PLOW_DOP_GEMM_C5] = g_gemm_c5;
    tab[PLOW_DOP_GEMM_NORM] = g_gemm_norm;
    tab[PLOW_DOP_GEMM_GLU] = g_gemm_glu;
    tab[PLOW_DOP_GEMV] = g_gemv;
    tab[PLOW_DOP_GEMV_GLU] = g_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = g_gemv_qkv;
    tab[PLOW_DOP_GEMV_ARGMAX] = g_gemv_argmax;
    tab[PLOW_DOP_FLASH_PREFILL] = g_flash_prefill;
    tab[PLOW_DOP_FLASH_DECODE] = g_flash_decode;
    tab[PLOW_DOP_FLASH_MERGE] = g_flash_merge;
    tab[PLOW_DOP_ATTN_RES] = g_attn_res;
}
