/* fa_varlen_bench.cu — PX-1 stage 2: kernel-level A/B of the batched-prefill flash.
 *
 *   serial : stage-1 semantics — R sequential d_flash_prefill launches, one per packed
 *            request (offset Q/O rows, slot-offset K/V bases), each its own full-grid
 *            cooperative-style pass with its own partial-wave tail.
 *   varlen : stage-2 — ONE d_flash_prefill_mux pass; the req table (cu_seqlens layout)
 *            enumerates every request's (q_tile, head) work items in one persistent grid.
 *
 * Shapes = gemma-4-12B prefill attention at the serving interleave quantum (2048 packed
 * rows split R ways), per-request kvlen 4096 (a 4k prompt's second chunk):
 *   sliding: hd256 BQ64 BKV32, nh16 nkv8, window 1024
 *   full   : hd512 BQ32 BKV16, nh16 nkv1, causal
 * Grid 188 x 256 (the interpreter's), smem = FA_PRE_SMEM_FLOATS.
 *
 * Build: nvcc -arch=sm_120a -O3 -I runtime/common -I runtime/nvidia \
 *          runtime/nvidia/experiments/fa_varlen_bench.cu -o /tmp/fa_varlen_bench
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm120_common.cuh"

using bf16 = __nv_bfloat16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s: %s\n",#x,cudaGetErrorString(e_)); exit(2);} } while(0)

template <int HD, int BQ, int BKV>
__global__ void k_mux(const int* req, bf16* O, const bf16* Q, const bf16* K, const bf16* V,
                      unsigned nh, unsigned nkv, unsigned win, unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill_mux<HD,BQ,BKV>(req, nullptr, nullptr, Q, K, V, O, 0, 0, nh, nkv, 0, win, 1,
                                   kv_stride, 0xFFFFFFFFu, scale, blockIdx.x, gridDim.x, sm);
}
/* Stage-1 semantics, verbatim: the requests loop SERIALLY inside one launch — every block
 * walks request r's (q_tile, head) items before moving to r+1, so each request pays its own
 * work-loop rounding/imbalance across the 188 blocks. (This is the exact retired mux body.) */
template <int HD, int BQ, int BKV>
__global__ void k_serial(const int* req, bf16* O, const bf16* Q, const bf16* K, const bf16* V,
                         unsigned nh, unsigned nkv, unsigned win, unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    const int R = req[0];
    for (int r = 0; r < R; r++) {
        const int q0 = req[1 + 4 * r], qlen = req[2 + 4 * r], slot = req[3 + 4 * r],
                  kvlen = req[4 + 4 * r];
        if (qlen <= 0) continue;
        const size_t qoff = (size_t)q0 * nh * HD;
        const size_t kvoff = (size_t)slot * nkv * (size_t)kv_stride * HD;
        d_flash_prefill<HD,BQ,BKV>(nullptr, nullptr, Q + qoff, K + kvoff, V + kvoff, O + qoff,
                                   (unsigned)qlen, (unsigned)kvlen, nh, nkv,
                                   (unsigned)(kvlen - qlen), win, 1, kv_stride, 0xFFFFFFFFu,
                                   scale, blockIdx.x, gridDim.x, sm);
    }
}

template <int HD, int BQ, int BKV>
static void bench(const char* label, unsigned nh, unsigned nkv, unsigned window, int rows,
                  int kvlen) {
    const unsigned kv_stride = 8192;
    const float scale = 1.0f / sqrtf((float)HD);
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float);
    CK(cudaFuncSetAttribute(k_mux<HD,BQ,BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_serial<HD,BQ,BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    const int RMAX = 8;
    bf16 *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, (size_t)rows * nh * HD * sizeof(bf16)));
    CK(cudaMalloc(&dO, (size_t)rows * nh * HD * sizeof(bf16)));
    CK(cudaMalloc(&dK, (size_t)RMAX * nkv * kv_stride * HD * sizeof(bf16)));
    CK(cudaMalloc(&dV, (size_t)RMAX * nkv * kv_stride * HD * sizeof(bf16)));
    CK(cudaMemset(dQ, 0x3c, (size_t)rows * nh * HD * sizeof(bf16)));
    CK(cudaMemset(dK, 0x3c, (size_t)RMAX * nkv * kv_stride * HD * sizeof(bf16)));
    CK(cudaMemset(dV, 0x3c, (size_t)RMAX * nkv * kv_stride * HD * sizeof(bf16)));

    printf("%s rows=%d kvlen=%d nh=%u nkv=%u win=%u\n", label, rows, kvlen, nh, nkv, window);
    for (int R = 1; R <= RMAX; R *= 2) {
        const int qlen = rows / R;
        std::vector<int> req;
        req.push_back(R);
        for (int r = 0; r < R; r++) {
            req.push_back(r * qlen);   /* q0   */
            req.push_back(qlen);       /* qlen */
            req.push_back(r);          /* slot */
            req.push_back(kvlen);      /* kvlen */
        }
        int* dreq; CK(cudaMalloc(&dreq, req.size() * 4));
        CK(cudaMemcpy(dreq, req.data(), req.size() * 4, cudaMemcpyHostToDevice));

        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int ITERS = 50;
        /* warm + time: varlen (one launch) */
        for (int w = 0; w < 3; w++)
            k_mux<HD,BQ,BKV><<<188,256,smem>>>(dreq, dO, dQ, dK, dV, nh, nkv, window, kv_stride, scale);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++)
            k_mux<HD,BQ,BKV><<<188,256,smem>>>(dreq, dO, dQ, dK, dV, nh, nkv, window, kv_stride, scale);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms_v; CK(cudaEventElapsedTime(&ms_v, e0, e1)); ms_v /= ITERS;
        /* warm + time: serial (device-side per-request loop, stage-1 semantics, one launch) */
        for (int w = 0; w < 3; w++)
            k_serial<HD,BQ,BKV><<<188,256,smem>>>(dreq, dO, dQ, dK, dV, nh, nkv, window, kv_stride, scale);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++)
            k_serial<HD,BQ,BKV><<<188,256,smem>>>(dreq, dO, dQ, dK, dV, nh, nkv, window, kv_stride, scale);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms_s; CK(cudaEventElapsedTime(&ms_s, e0, e1)); ms_s /= ITERS;

        printf("  R=%d qlen=%4d  serial %8.1f us  varlen %8.1f us  speedup %.2fx\n",
               R, qlen, ms_s * 1000.0, ms_v * 1000.0, ms_s / ms_v);
        CK(cudaFree(dreq)); CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
    CK(cudaFree(dQ)); CK(cudaFree(dK)); CK(cudaFree(dV)); CK(cudaFree(dO));
}

int main() {
    /* Serving interleave quantum: 2048 packed rows; per-request kv history 4096. */
    bench<256,64,32>("sliding hd256", 16, 8, 1024, 2048, 4096);
    bench<512,32,16>("full    hd512", 16, 1, 0,    2048, 4096);
    /* Cold 8k pack (no decode live: budget 8192). */
    bench<256,64,32>("sliding hd256 cold", 16, 8, 1024, 8192, 8192);
    bench<512,32,16>("full    hd512 cold", 16, 1, 0,    8192, 8192);
    return 0;
}
