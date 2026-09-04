/* k3_kbench_fold.hip — MLA_MERGE_FOLD body oracle + bench at Kimi-K3 TP8 decode shapes.
 *
 * Includes the production op header verbatim (the glm52_kbench_fold pattern). `k_ctl` is the
 * body the shipped decode object runs at this shape (`d_mla_merge_fold<512, 32>`, 48 work items
 * on the 256-workgroup grid); the `k_s*` kernels are `d_mla_merge_fold_split` at the tiles the
 * interpreter's PLOW_MLA_FOLD_DVT arm can pick. The driver compares every output bit against
 * k_ctl — the split body is a re-partition, not a reassociation, and this is where that is
 * checked rather than argued. */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_attention.h"

typedef union __align__(16) {
    float f[4096]; /* olds(512) + red(<= 8*32) + wgt(nsplit) */
} kbf_smem;

#define MF_CTL(NAME, VT)                                                                          \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                              \
        bf16* O, const float* Op, const float* mlp, const bf16* Wuv, unsigned nh, unsigned V,     \
        unsigned ns) {                                                                            \
        __shared__ kbf_smem sm;                                                                   \
        d_mla_merge_fold<512, VT>(O, Op, mlp, Wuv, 1u, nh, V, ns, blockIdx.x, gridDim.x, sm.f);   \
    }
#define MF_SPLIT(NAME, VT, MSG)                                                                   \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                              \
        bf16* O, const float* Op, const float* mlp, const bf16* Wuv, unsigned nh, unsigned V,     \
        unsigned ns) {                                                                            \
        __shared__ kbf_smem sm;                                                                   \
        d_mla_merge_fold_split<512, VT, 32, PLOW_MLA_FOLD_VEC, PLOW_MLA_FOLD_UN, MSG>(            \
            O, Op, mlp, Wuv, 1u, nh, V, ns, blockIdx.x, gridDim.x, sm.f);                         \
    }

MF_CTL(k_ctl, 32)
MF_SPLIT(k_s32, 32, PLOW_MLA_FOLD_DEC_MS)
MF_SPLIT(k_s16, 16, PLOW_MLA_FOLD_DEC_MS)
MF_SPLIT(k_s8, 8, PLOW_MLA_FOLD_DEC_MS)
MF_SPLIT(k_s8m8, 8, 8)
MF_SPLIT(k_s8m64, 8, 64)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_null(bf16* O, const float* Op,
                                                                  const float* mlp,
                                                                  const bf16* Wuv, unsigned nh,
                                                                  unsigned V, unsigned ns) {
    if (threadIdx.x == 1023u) O[0] = (bf16)nh;
}
