/* glm52_kbench_fold.hip — MLA_MERGE_FOLD body bench at GLM-5.2 TP4 decode shapes.
 *
 * Includes the production op header verbatim (the runtime/ubench pattern; no copies). Sweeps the
 * (VT, VEC, UN) fold map. Launch geometry mirrors the persistent interpreter: blockDim =
 * PLOW_THREADS = 512, gridDim = the workgroup count the emitter gives the packet.
 *
 * CACHE DEFEAT is load-bearing here and the earlier bench did not have it: one head's W_uv is
 * 256 KB and the whole packet's is 4 MB, which sits inside the 256 MB LLC. In the model the fold
 * reads a DIFFERENT layer's W_uv every packet, so it is cold. The driver rotates over a
 * >LLC arena of W_uv copies.
 */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_attention.h"

typedef union __align__(16) {
    float f[4096]; /* olds(512) + red(PLOW_WAVES*VT<=8*256) + wgt(nsplit) */
} kbf_smem;

#define MF_K(NAME, VT, VEC, UN)                                                                   \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(                              \
        bf16* O, const float* Op, const float* mlp, const bf16* Wuv, unsigned nh, unsigned V,     \
        unsigned ns) {                                                                            \
        __shared__ kbf_smem sm;                                                                   \
        d_mla_merge_fold<512, VT, VEC, UN>(O, Op, mlp, Wuv, 1u, nh, V, ns, blockIdx.x, gridDim.x, \
                                           sm.f);                                                 \
    }

/* the shipped baseline: one thread per column, whole-V tile */
MF_K(k_mf_base, 256, 1, 1)
/* VT sweep at the default map */
MF_K(k_mf_v16, 16, 4, 4)
MF_K(k_mf_v32, 32, 4, 4)
MF_K(k_mf_v64, 64, 4, 4)
MF_K(k_mf_v128, 128, 4, 4)
MF_K(k_mf_v256, 256, 4, 4)
/* VEC / UN sweep at VT=64 */
MF_K(k_mf_v64c2, 64, 2, 4)
MF_K(k_mf_v64c8, 64, 8, 4)
MF_K(k_mf_v64u2, 64, 4, 2)
MF_K(k_mf_v64u8, 64, 4, 8)
MF_K(k_mf_v32c8, 32, 8, 4)
MF_K(k_mf_v128c8, 128, 8, 4)
MF_K(k_mf_v16c8, 16, 8, 2)
MF_K(k_mf_v32c8u8, 32, 8, 8)

/* launch floor: same grid/blockDim, no body. Everything above is measured against this. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_null(bf16* O, const float* Op,
                                                                  const float* mlp,
                                                                  const bf16* Wuv, unsigned nh,
                                                                  unsigned V, unsigned ns) {
    if (threadIdx.x == 1023u) O[0] = (bf16)nh;
}
