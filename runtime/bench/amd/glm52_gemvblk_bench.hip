/* Block-fp8 decode GEMV (op 44) vs the bf16 GEMV (op 10) it would replace, at GLM-5.2 TP4 M=1
 * shapes — the instrument behind the gemv_rows_fp8_blk MLP fix. Production bodies included
 * verbatim from the real headers; launch geometry mirrors the persistent interpreter
 * (blockDim = PLOW_THREADS, grid = the emitter's workgroup count). x is staged ONCE outside the
 * rep loop so the number is the packet BODY, not an LDS restage repeated per rep.
 *
 * Also benches the OTHER half of the GLM_LINEAR_FP8 lever, DenseGluFp8Blk (op 47), which is a
 * different kernel (wave_dot_fp8_blk) and is NOT fixed by the op-44 patch.
 *
 * Build + run (ROCm tooling must be OUTSIDE `nix develop` — knob-contract 0a):
 *   hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -std=c++17 --genco \
 *         perf-data/glm52_gemvblk_bench.hip -o /tmp/gvb.co -Iruntime/amd -Iruntime/common
 *   g++ -O2 -std=c++17 perf-data/glm52_gemvblk_bench.cpp -o /tmp/gvb \
 *         -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64
 *   perf-data/harness/gpulease -n 1 gvb sg render -c \
 *         'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; /tmp/gvb /tmp/gvb.co 256 21'
 * `unset HIP_VISIBLE_DEVICES` is load-bearing: gpulease exports BOTH it and ROCR to the same
 * absolute id and they COMPOSE, so hipInit reports "no ROCm-capable device" on a leased GPU. */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"

typedef union __align__(16) { bf16 xstage[8192]; float part[PLOW_WAVES]; } gv_smem;

/* mirror of d_gemv_fp8_blk's dispatch with the LDS stage hoisted out of the rep loop */
#define DISP(UN) gemv_rows_fp8_blk<PLOW_GEMV_MM, true, UN>(C, x, Wp, s, 1u, N, K, blockIdx.x, gridDim.x, sm.xstage)
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_blk_ship(
    bf16* C, const bf16* x, const unsigned char* W, const float* s, unsigned nrep, unsigned N,
    unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned i = threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
    __syncthreads();
    const unsigned nchunk = (K + 1023u) >> 10;
    for (unsigned e = 0; e < nrep; e++) {
        const unsigned char* Wp = W + (size_t)e * N * K;
        if (nchunk >= 16u) DISP(8);
        else if (nchunk >= 11u) DISP(6);
        else if (nchunk <= 1u) DISP(1);
        else if (nchunk == 2u) DISP(2);
        else if (nchunk == 4u) DISP(4);
        else DISP(3);
    }
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_bf16_ship(
    bf16* C, const bf16* x, const bf16* W, unsigned nrep, unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned i = threadIdx.x; i < K; i += PLOW_THREADS) sm.xstage[i] = x[i];
    __syncthreads();
    for (unsigned e = 0; e < nrep; e++)
        gemv_rows<PLOW_GEMV_MM, true, false>(C, x, W + (size_t)e * N * K, nullptr, nullptr, 1u, N, K,
                                             blockIdx.x, gridDim.x, sm.xstage);
}

/* ---- the OTHER half of the GLM_LINEAR_FP8 lever: shared gate/up.
 * bf16 today = GemvGlu (op 19, gemv_glu_rows: 2 streams x UN=6 in flight, x in LDS).
 * fp8 arm    = DenseGluFp8Blk (op 47, d_dense_glu_fp8_blk -> wave_dot_fp8_blk: ONE output per
 *              wave, ONE load in flight, x read from GLOBAL). Different kernel, not patched here. */
#include "op_moe.h"
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_glu_bf16(
    bf16* C, const bf16* x, const bf16* Wg, unsigned nrep, unsigned N, unsigned K) {
    __shared__ gv_smem sm;
    for (unsigned e = 0; e < nrep; e++)
        d_gemv_glu(C, x, Wg + (size_t)e * N * K * 2, Wg + (size_t)e * N * K * 2 + (size_t)N * K, 1u,
                   N, K, PLOW_ACT_SILU_, blockIdx.x, gridDim.x, sm.xstage);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_glu_blk(
    bf16* C, const bf16* x, const unsigned char* Wg, const float* s, unsigned nrep, unsigned N,
    unsigned K) {
    for (unsigned e = 0; e < nrep; e++)
        d_dense_glu_fp8_blk(C, x, Wg + (size_t)e * N * K * 2, Wg + (size_t)e * N * K * 2 + (size_t)N * K,
                            s, s, N, K, PLOW_ACT_SILU_, blockIdx.x, gridDim.x);
}
