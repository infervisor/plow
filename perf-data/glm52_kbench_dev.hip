/* glm_kbench.hip — apples-to-apples bandwidth bench of plow's PRODUCTION block-fp8 decode kernel
 * bodies at GLM-5.2 TP4 M=1 shapes. Includes the production headers verbatim; no copies.
 *
 * The comparison that matters: op 45/46 (d_moe_expert_{glu,down}_fp8_blk, built on
 * wave_dot_fp8_blk) vs op 44 (d_gemv_fp8_blk, built on gemv_rows_fp8_blk) reading the SAME
 * bytes off the SAME buffers. Both are w8a16 block-fp8 [128,128]; only the loop nest differs.
 *
 * Cache defeat: the weight arena is 800 MB, far past the 256 MB LLC, and every iteration walks a
 * different expert.
 *
 * Launch geometry MIRRORS the persistent interpreter: blockDim = PLOW_THREADS (512 = 8 waves),
 * gridDim = the packet's workgroup count (28 under GLM_MOE_CORESIDENT=2, or 256 for all-CU ops).
 */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_norm.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_attention.h"
#include "op_moe.h"

typedef union __align__(16) {
    float part[PLOW_WAVES];
    float fmla[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    bf16 xstage[8192];
} kb_smem;

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_moe_glu_blk(
    bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned slot, unsigned I, unsigned H, unsigned E, unsigned act) {
    d_moe_expert_glu_fp8_blk(fu, x, tab, wt, st, slot, I, H, E, act, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_moe_down_blk(
    float* part, const bf16* fu, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned slot, unsigned H, unsigned I, unsigned E) {
    d_moe_expert_down_fp8_blk(part, fu, tab, wt, st, slot, H, I, E, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv_fp8_blk(
    bf16* C, const bf16* x, const unsigned char* W, const float* s, unsigned M, unsigned N,
    unsigned K) {
    __shared__ kb_smem sm;
    d_gemv_fp8_blk(C, x, W, s, M, N, K, blockIdx.x, gridDim.x, sm.xstage);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemv_bf16(bf16* C, const bf16* x,
                                                                      const bf16* W, unsigned M,
                                                                      unsigned N, unsigned K) {
    __shared__ kb_smem sm;
    d_gemv(C, x, W, nullptr, nullptr, M, N, K, 0, 0.0f, blockIdx.x, gridDim.x, sm.xstage);
}
/* pure-stream reference: what the machine gives for a coalesced 16 B/lane read with UN loads
 * in flight per wave, at the interpreter's exact geometry. The MLP ceiling, nothing else. */
template <int UN>
__device__ __forceinline__ void streamk(float* out, const unsigned char* W, size_t bytes) {
    const unsigned lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const size_t wgw = (size_t)gridDim.x * PLOW_WAVES;
    const size_t widx = (size_t)blockIdx.x * PLOW_WAVES + wave;
    const size_t chunks = bytes / 1024;
    float acc = 0.f;
    for (size_t c = widx * UN; c < chunks; c += wgw * UN) {
        fp8v16 v[UN];
#pragma unroll
        for (int u = 0; u < UN; u++) {
            size_t cc = c + u < chunks ? c + u : chunks - 1;
            v[u] = *(const fp8v16*)(W + cc * 1024 + lane * 16);
        }
#pragma unroll
        for (int u = 0; u < UN; u++) {
            bf16v8 lo, hi;
            fp8_to_bf16v8(v[u], lo, hi);
            acc += bf2f(lo[0]) + bf2f(hi[0]);
        }
    }
    if (acc == 1234.5f) out[0] = acc;
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_stream1(float* o,
                                                                     const unsigned char* W,
                                                                     size_t b) {
    streamk<1>(o, W, b);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_stream3(float* o,
                                                                     const unsigned char* W,
                                                                     size_t b) {
    streamk<3>(o, W, b);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_stream8(float* o,
                                                                     const unsigned char* W,
                                                                     size_t b) {
    streamk<8>(o, W, b);
}


/* LOOPED wrappers: one launch walks `nrep` experts, mirroring the persistent interpreter where a
 * packet costs no launch. Removes the ~4.6 us/launch hipModuleLaunchKernel floor from the number. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_glu_loop(
    bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned nrep, unsigned I, unsigned H, unsigned E, unsigned act) {
    for (unsigned e = 0; e < nrep; e++)
        d_moe_expert_glu_fp8_blk(fu, x, tab, wt, st, e, I, H, E, act, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_down_loop(
    float* part, const bf16* fu, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned nrep, unsigned H, unsigned I, unsigned E) {
    for (unsigned e = 0; e < nrep; e++)
        d_moe_expert_down_fp8_blk(part, fu, tab, wt, st, e, H, I, E, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_gemvblk_loop(
    bf16* C, const bf16* x, const unsigned char* W, const float* s, unsigned nrep, unsigned N,
    unsigned K) {
    __shared__ kb_smem sm;
    for (unsigned e = 0; e < nrep; e++)
        d_gemv_fp8_blk(C, x, W + (size_t)e * N * K, s, 1u, N, K, blockIdx.x, gridDim.x, sm.xstage);
}

/* CO-RESIDENT emulation (GLM_MOE_CORESIDENT: 8 experts on disjoint 256/(tk+1)-CU slices, all at
 * once). blockIdx -> (expert, slice) exactly as b.split(tk+1, sl) does. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_glu_cores(
    bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned per, unsigned I, unsigned H, unsigned E, unsigned act) {
    unsigned sl = blockIdx.x / per, s = blockIdx.x % per;
    d_moe_expert_glu_fp8_blk(fu, x, tab, wt, st, sl, I, H, E, act, s, per);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_down_cores(
    float* part, const bf16* fu, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned per, unsigned H, unsigned I, unsigned E) {
    unsigned sl = blockIdx.x / per, s = blockIdx.x % per;
    d_moe_expert_down_fp8_blk(part, fu, tab, wt, st, sl, H, I, E, s, per);
}
/* GROUPED arm (ops 48/49) — one packet, all k slots, on the whole chip. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_group_glu(
    bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned k, unsigned I, unsigned H, unsigned E, unsigned act) {
    d_moe_group_glu_fp8_blk(fu, x, tab, wt, st, k, I, H, E, act, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_group_down(
    float* part, const bf16* fu, const unsigned char* tab, const unsigned long long* wt,
    const unsigned long long* st, unsigned k, unsigned H, unsigned I, unsigned E) {
    d_moe_group_down_fp8_blk(part, fu, tab, wt, st, k, H, I, E, blockIdx.x, gridDim.x);
}

/* ---- remaining GLM decode ops at their real shapes ---- */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rmsnorm(bf16* o, const bf16* x,
                                                                    const bf16* g, unsigned rows,
                                                                    unsigned feat, float eps) {
    __shared__ kb_smem sm;
    d_rmsnorm(o, x, g, rows, feat, eps, 0, blockIdx.x, gridDim.x, sm.part);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_residual(bf16* o, const bf16* a,
                                                                     const bf16* b, unsigned n) {
    d_residual(o, a, b, n, 1.0f, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_hnr(bf16* o, const bf16* x,
                                                                 const float* c, const float* s,
                                                                 const int* p, unsigned nt,
                                                                 unsigned nh) {
    d_headnorm_rope<64, true>(o, x, nullptr, c, s, p, nt, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1,
                              blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_qkv(bf16* q, bf16* k, bf16* v,
                                                                 const bf16* x, const bf16* Wq,
                                                                 const bf16* Wk, const bf16* Wv,
                                                                 unsigned Nq, unsigned Nk,
                                                                 unsigned Nv, unsigned K) {
    __shared__ kb_smem sm;
    d_gemv_qkv(q, k, v, x, Wq, Wk, Wv, 1u, Nq, Nk, Nv, K, blockIdx.x, gridDim.x, sm.xstage);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_mla_dec2(
    float* Op, float* mlp, const bf16* Qa, const bf16* Qr, const bf16* Ckv, const bf16* Kr,
    const int* kvlen, unsigned nh, unsigned ks, float sc, unsigned ns) {
    __shared__ kb_smem sm;
    d_flash_mla_decode<512, 64, 2, false>(Op, mlp, Qa, Qr, Ckv, Kr, kvlen, 1u, nh, ks, 0, sc, ns,
                                          0xFFFFFFFFu, blockIdx.x, gridDim.x, sm.fmla);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_mf(bf16* O, const float* Op,
                                                                const float* mlp, const bf16* Wuv,
                                                                unsigned nh, unsigned V,
                                                                unsigned ns) {
    __shared__ kb_smem sm;
    d_mla_merge_fold<512, 64>(O, Op, mlp, Wuv, 1u, nh, V, ns, blockIdx.x, gridDim.x, sm.fmla);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_dglu(bf16* fu, const bf16* x,
                                                                  const unsigned char* Wg,
                                                                  const unsigned char* Wu,
                                                                  const float* Sg, const float* Su,
                                                                  unsigned N, unsigned K) {
    d_dense_glu_fp8_blk(fu, x, Wg, Wu, Sg, Su, N, K, 1u, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_null(float* o) { if (threadIdx.x==1023) o[0]=1.f; }
/* looped tiny ops: amortize the ~2.7us hipModuleLaunchKernel floor so the BODY cost is visible */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_rmsnorm_loop(bf16* o, const bf16* x, const bf16* g, unsigned rep, unsigned feat, float eps) {
    __shared__ kb_smem sm;
    for (unsigned i=0;i<rep;i++) d_rmsnorm(o, x, g, 1u, feat, eps, 0, blockIdx.x, gridDim.x, sm.part);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_hnr_loop(bf16* o, const bf16* x, const float* c, const float* s, const int* p, unsigned rep, unsigned nh) {
    for (unsigned i=0;i<rep;i++) d_headnorm_rope<64,true>(o, x, nullptr, c, s, p, 1u, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_resid_loop(bf16* o, const bf16* a, const bf16* b, unsigned rep, unsigned n) {
    for (unsigned i=0;i<rep;i++) d_residual(o, a, b, n, 1.0f, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_mf256(bf16* O, const float* Op, const float* mlp, const bf16* Wuv, unsigned nh, unsigned V, unsigned ns) {
    __shared__ kb_smem sm;
    d_mla_merge_fold<512, 256>(O, Op, mlp, Wuv, 1u, nh, V, ns, blockIdx.x, gridDim.x, sm.fmla);
}
