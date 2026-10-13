/* gemm_group_pf.hip — DevOp::GemmGroupPf as a standalone prefill segment (GLM-5.3-Flash KDA input
 * projections: q|k|v|g_a|f_a|b in one launch). One workgroup per output tile of one group; the
 * groups share x and read consecutive row blocks of one concatenated [sum N_g, K] weight. Each
 * group keeps the interpreter's tile for its width (d_gemm_wide for N >= 512, d_gemm_small below),
 * so every output element sees the interpreter's K order.
 * Args: kernarg `GemmGroupArgs` (crates/plowrt/src/exec/amd.rs). Grid: plow_gemm_group_tiles. */
#include "amd_common.h"
#include "op_gemm.h"

extern "C" __device__ unsigned plow_gemm_group_pf_abi_1 = 1;

#define PLOW_GG_GROUPS 6

struct GemmGroupArgs {
    const bf16* x;
    const bf16* w;
    bf16* c[PLOW_GG_GROUPS];
    unsigned t, k;
    unsigned n[PLOW_GG_GROUPS];
};

static __device__ __forceinline__ unsigned gg_tiles(unsigned t, unsigned n) {
    const bool wide = n >= 512;
    const unsigned bm = wide ? GM_WD_BM : GM_SM_BM, bn = wide ? GM_WD_BN : GM_SM_BN;
    return ((t + bm - 1) / bm) * ((n + bn - 1) / bn);
}

#define GG_LDS_WIDE GM_LDS_HALVES_T(GM_WD_BM, GM_WD_BN, GM_WD_BK)
#define GG_LDS_SMALL GM_LDS_HALVES_T(GM_SM_BM, GM_SM_BN, GM_SM_BK)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void plow_gemm_group_pf_gfx950(GemmGroupArgs a) {
    __shared__ bf16 lds[GG_LDS_WIDE > GG_LDS_SMALL ? GG_LDS_WIDE : GG_LDS_SMALL];
    /* Grid-stride over the concatenated tile space: a launch narrower than the tile count (a
     * PLOW_SEG_GEOM override) still covers every tile. */
    for (unsigned id = blockIdx.x;; id += gridDim.x) {
        unsigned b = id;
        size_t row0 = 0;
        unsigned g = 0;
        for (; g < PLOW_GG_GROUPS && a.n[g]; g++) {
            const unsigned tiles = gg_tiles(a.t, a.n[g]);
            if (b < tiles) break;
            b -= tiles;
            row0 += a.n[g];
        }
        if (g == PLOW_GG_GROUPS || !a.n[g]) return;
        const unsigned n = a.n[g], tiles = gg_tiles(a.t, n);
        const bf16* w = a.w + row0 * a.k;
        if (n >= 512)
            d_gemm_wide(a.c[g], a.x, w, a.t, n, a.k, b, tiles, lds);
        else
            d_gemm_small(a.c[g], a.x, w, a.t, n, a.k, b, tiles, lds);
        __syncthreads();
    }
}
