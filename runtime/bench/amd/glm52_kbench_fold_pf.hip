/* glm52_kbench_fold_pf.hip — MLA_MERGE_FOLD body at GLM-5.2 TP8 PREFILL shapes.
 *
 * The existing glm52_kbench_fold.hip is a DECODE bench (n_batch=1, nh=16, ns=16, 16 rows of
 * work). Prefill is a different machine: the emitter folds the token axis into n_batch, so the
 * packet is n_batch=T rows of (token, head) — 65536 rows at T=8192, TP8 — and the op is 8.6% of
 * a prefill layer's CU budget. Nothing measured that body at that shape before this file.
 *
 * Includes the production header verbatim (the runtime/ubench pattern; no copies). Launch
 * geometry mirrors the persistent interpreter: blockDim = PLOW_THREADS = 512, gridDim = 304 (the
 * whole chip; `mla_fold_cus` hands the packet every CU because n_work >> n_cu).
 *
 * NO cache defeat here, and that is deliberate and is the point: within ONE packet the same 2 MB
 * of W_uv is read by all 65536 rows, so it is L2-resident by the second row whatever the driver
 * does. The stream this bench measures is L2->CU, not HBM->L2.
 */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#define PLOW_MLA_FOLD_TB 8
#include "op_attention.h"

typedef union __align__(16) {
    float f[12288]; /* TB=8: olds(8*512) + red(4*8*256) = 12288 floats = 48 KiB */
} kbf_smem;

#define MF_ARGS                                                                                   \
    bf16 *O, const float *Op, const float *mlp, const bf16 *Wuv, unsigned nb, unsigned nh,        \
        unsigned V, unsigned ns

/* shipped body, one (token, head) row per work item */
#define MF_BASE(NAME, VT, VEC, UN)                                                                \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(MF_ARGS) {                    \
        __shared__ kbf_smem sm;                                                                   \
        d_mla_merge_fold<512, VT, VEC, UN>(O, Op, mlp, Wuv, nb, nh, V, ns, blockIdx.x, gridDim.x, \
                                           sm.f);                                                 \
    }
/* token-blocked body, TB rows of one head per work item */
#define MF_TB(NAME, VT, TB, VEC, UN, MB)                                                          \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(MF_ARGS) {                    \
        __shared__ kbf_smem sm;                                                                   \
        d_mla_merge_fold_tb<512, VT, TB, VEC, UN, MB>(O, Op, mlp, Wuv, nb, nh, V, ns, blockIdx.x, \
                                                      gridDim.x, sm.f);                           \
    }

MF_BASE(k_pf_base, 256, 4, 4)
MF_TB(k_pf_tb2, 256, 2, 4, 4, true)
MF_TB(k_pf_tb4, 256, 4, 4, 4, true)
MF_TB(k_pf_tb8, 256, 8, 4, 4, true)
MF_TB(k_pf_tb4u8, 256, 4, 4, 8, true)
MF_TB(k_pf_tb4s, 256, 4, 4, 4, false)   /* TB4, SERIAL merge — prices the merge interleave */
MF_TB(k_pf_tb4u8s, 256, 4, 4, 8, false)
MF_TB(k_pf_tb8u8, 256, 8, 4, 8, true)

/* MERGE ONLY — the shipped merge half verbatim, fold deleted. Prices exactly what the causal
 * KV-split (`PLOW_GLM_PF_NS=2`) adds, against `k_pf_base` which is merge+fold. olds[tid&511] is
 * stored so nothing is dead. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_pf_merge(MF_ARGS) {
    __shared__ kbf_smem sm;
    float* olds = sm.f;
    const unsigned tid = threadIdx.x;
    const unsigned n_work = nb * nh;
    for (unsigned w = blockIdx.x; w < n_work; w += gridDim.x) {
        const float* ml = mlp + (size_t)w * ns * 2;
        float gm;
        const float gl = fa_merge_ml(ml, ns, gm);
        const float inv = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;
        const float* opb = Op + (size_t)w * ns * 512;
        for (unsigned d = tid; d < 512u; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned s = 0; s < ns; s++) {
                const float m = ml[s * 2];
                acc += (m == FA_NEG_INF) ? 0.0f : (opb[(size_t)s * 512 + d] * FA_EXP(m - gm));
            }
            olds[d] = acc * inv;
        }
        __syncthreads();
        if (tid < V) st_act1(&O[(size_t)w * V + tid], f2bf(olds[tid]));
        __syncthreads();
    }
}

/* launch floor: same grid/blockDim, no body. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_pf_null(MF_ARGS) {
    if (threadIdx.x == 1023u) O[0] = (bf16)nh;
}
