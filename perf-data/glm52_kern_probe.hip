/* glm_kern_probe.hip — COMPILE-ONLY resource probe for the GLM-5.2 decode kernels.
 * Wraps each production __device__ op as its own __global__ so that
 * -Rpass-analysis=kernel-resource-usage reports its INTRINSIC VGPR/LDS/occupancy,
 * separately from the megakernel's worst-case-over-all-arms allocation. */
#include "amd_common.h"
#include "op_norm.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_attention.h"
#include "op_moe.h"

typedef union __align__(16) {
    float part[PLOW_WAVES];
    bf16  gm[GM_LDS_HALVES_T(GM_BM, GM_BN, GM_BK)];
    bf16  fa[FA_LDS_HALVES(512)];
    float fd[FA_DEC_LDS_FLOATS(512, PLOW_FA_GF(512))];
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    unsigned long long am[PLOW_WAVES];
} probe_smem;

/* MLA-decode-sized LDS only (what the GLM decode path actually needs) */
typedef union __align__(16) {
    float part[PLOW_WAVES];
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    bf16  xstage[8192];
} mla_smem;

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_moe_glu_blk(bf16* fu, const bf16* x, const unsigned char* tab,
                   const unsigned long long* wt, const unsigned long long* st,
                   unsigned slot, unsigned I, unsigned H, unsigned E, unsigned act) {
    d_moe_expert_glu_fp8_blk(fu, x, tab, wt, st, slot, I, H, E, act, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_moe_down_blk(float* part, const bf16* fu, const unsigned char* tab,
                    const unsigned long long* wt, const unsigned long long* st,
                    unsigned slot, unsigned H, unsigned I, unsigned E) {
    d_moe_expert_down_fp8_blk(part, fu, tab, wt, st, slot, H, I, E, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_moe_group_glu_blk(bf16* fu, const bf16* x, const unsigned char* tab,
                         const unsigned long long* wt, const unsigned long long* st,
                         unsigned k, unsigned I, unsigned H, unsigned E, unsigned act) {
    d_moe_group_glu_fp8_blk(fu, x, tab, wt, st, k, I, H, E, act, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_dense_glu_blk(bf16* fu, const bf16* x, const unsigned char* Wg, const unsigned char* Wu,
                     const float* Sg, const float* Su, unsigned N, unsigned K, unsigned act) {
    d_dense_glu_fp8_blk(fu, x, Wg, Wu, Sg, Su, N, K, act, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_gemv_fp8_blk(bf16* C, const bf16* x, const unsigned char* W, const float* s,
                    unsigned M, unsigned N, unsigned K) {
    __shared__ mla_smem sm;
    d_gemv_fp8_blk(C, x, W, s, M, N, K, blockIdx.x, gridDim.x, sm.xstage);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_decode_gf4(float* Op, float* mlp, const bf16* Qa, const bf16* Qr, const bf16* Ckv,
                      const bf16* Kr, const int* kvlen, unsigned nb, unsigned nh, unsigned ks,
                      float sc, unsigned ns) {
    __shared__ mla_smem sm;
    d_flash_mla_decode<512, 64, 4, false>(Op, mlp, Qa, Qr, Ckv, Kr, kvlen, nb, nh, ks, 0, sc, ns,
                                          0xFFFFFFFFu, blockIdx.x, gridDim.x, sm.fmla);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_decode_gf2(float* Op, float* mlp, const bf16* Qa, const bf16* Qr, const bf16* Ckv,
                      const bf16* Kr, const int* kvlen, unsigned nb, unsigned nh, unsigned ks,
                      float sc, unsigned ns) {
    __shared__ mla_smem sm;
    d_flash_mla_decode<512, 64, 2, false>(Op, mlp, Qa, Qr, Ckv, Kr, kvlen, nb, nh, ks, 0, sc, ns,
                                          0xFFFFFFFFu, blockIdx.x, gridDim.x, sm.fmla);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_merge_fold(bf16* O, const float* Op, const float* mlp, const bf16* Wuv,
                      unsigned nb, unsigned nh, unsigned V, unsigned ns) {
    __shared__ mla_smem sm;
    d_mla_merge_fold<512, 64>(O, Op, mlp, Wuv, nb, nh, V, ns, blockIdx.x, gridDim.x, sm.fmla);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_headnorm_rope_hd64_il(bf16* o, const bf16* x, const bf16* g, const float* c,
                             const float* s, const int* p, unsigned nt, unsigned nh, float eps) {
    d_headnorm_rope<64, true>(o, x, g, c, s, p, nt, nh, eps, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                              gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_gemv_bf16(bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K) {
    __shared__ mla_smem sm;
    d_gemv(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, sm.xstage);
}
