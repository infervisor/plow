/* AVX-512 tier registrar: overrides the golden entries it has a kernel for. Called by
 * plow_cpu_init after the golden registrar when cpuid reports AVX-512 F/BW/VL/BF16. */
#include "avx512.h"

void plow_cpu_register_avx512(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_RESIDUAL] = v_residual;
    tab[PLOW_DOP_GLU] = v_glu;
    tab[PLOW_DOP_SOFTCAP] = v_softcap;
    tab[PLOW_DOP_EMBED] = v_embed;
    tab[PLOW_DOP_ARGMAX] = v_argmax;
    tab[PLOW_DOP_ARGMAX_FIN] = v_argmax_fin;
    tab[PLOW_DOP_RMSNORM] = v_rmsnorm;
    tab[PLOW_DOP_ROWRMS] = v_rowrms;
    tab[PLOW_DOP_LAYERNORM] = v_layernorm;
    tab[PLOW_DOP_NORM_RESIDUAL] = v_norm_residual;
    tab[PLOW_DOP_ADD_NORM] = v_add_norm;
    tab[PLOW_DOP_NORM_RESIDUAL_NORM] = v_norm_residual_norm;
    tab[PLOW_DOP_GEMV] = v_gemv;
    tab[PLOW_DOP_GEMV_GLU] = v_gemv_glu;
    tab[PLOW_DOP_GEMV_QKV] = v_gemv_qkv;
    tab[PLOW_DOP_GEMV_ARGMAX] = v_gemv_argmax;
}
