/* gemm_group_pf.hip — DevOp::GemmGroupPf as a standalone prefill segment (GLM-5.3-Flash KDA
 * q|k|v|g_a|f_a|b, MLA q_a|kv_a_latent): one GEMM over the loader-stacked weight with the output
 * columns routed to each group's own row-major buffer.
 * Args: kernarg `GemmGroupArgs` (crates/plowrt/src/exec/amd.rs). Grid: `gemm_group_grid`. */
#include "amd_common.h"
#include "op_gemm.h"

extern "C" __device__ unsigned plow_gemm_group_pf_abi_1 = 1;

#define PLOW_GG_GROUPS 6

struct GemmGroupArgs {
    const bf16* x;
    const bf16* w;
    PlowGemmRoute r;
    unsigned t, k;
};

/* ONE GEMM over the concatenated N (the weight rows are already stacked) on the 128x384 tile,
 * output columns routed per group. Tiles agree element for element with the interpreter's
 * GemmWide (same BK and MFMA K order; measured bit-identical, gemm_group_kda_gfx950.hip). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void plow_gemm_group_pf_gfx950(GemmGroupArgs a) {
    __shared__ bf16 lds[GM_LDS_HALVES_T(GM_C8_BM, GM_C8_BN, GM_C8_BK)];
    unsigned n = 0;
    for (unsigned g = 0; g < PLOW_GG_GROUPS; g++) n += a.r.n[g];
    const unsigned tiles = ((a.t + GM_C8_BM - 1) / GM_C8_BM) * ((n + GM_C8_BN - 1) / GM_C8_BN);
    d_gemm_t<GM_C8_BM, GM_C8_BN, GM_C8_BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true,
             false, false, false, false
#if PLOW_GLM_FUSE_POST
             , false
#endif
             , false, true>(nullptr, a.x, a.w, nullptr, nullptr, a.t, n, a.k, blockIdx.x,
                            gridDim.x < tiles ? gridDim.x : tiles, lds, nullptr, 0, nullptr,
                            nullptr, nullptr
#if PLOW_GLM_FUSE_POST
                            , nullptr, nullptr, nullptr, 0
#endif
                            , 0, &a.r);
}
