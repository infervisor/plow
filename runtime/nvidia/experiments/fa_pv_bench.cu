// fa_pv_bench.cu — FLASH_PREFILL op-level A/B for the T4 register-resident mma P.V.
//
// Times d_flash_prefill<HD,BQ,BKV> (op_attention.cuh) at Gemma-4-12B prefill shapes, grid=188
// (RTX PRO 6000, 188 SMs) / 256 threads — the exact launch geometry the persistent interpreter
// uses. Compile this ONE source against two include paths to get the A/B:
//   baseline (18ea793, FFMA-serial P.V):  nvcc -I <base_inc> ...
//   T4       (register-resident mma P.V): nvcc -I runtime/nvidia ...
// The kernel body is otherwise identical, so the delta is the P.V lever alone.
//
// Shapes model the 12B layer mix (5:1 sliding:full): hd512 FULL causal (nh16/nkv1, the O(ctx^2)
// tail) and hd256 SLIDING window=1024 (nh16/nkv8, O(window)). seq_q = 8192 (the max prefill
// bucket); seq_kv swept to expose the long-ctx tail. Data is zeroed (timing is shape-driven, not
// content-driven — same convention as the prefill perf harness); nsplit=1 (the fused chunked path).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "op_attention.cuh"

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

template <int HD, int BQ, int BKV>
__global__ void k_fa_pre(float* Op, float* Ml, __nv_bfloat16* O, const __nv_bfloat16* Q,
                         const __nv_bfloat16* K, const __nv_bfloat16* V, unsigned sq, unsigned skv,
                         unsigned nh, unsigned nkv, unsigned qpos0, unsigned win, unsigned nsplit,
                         unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill<HD, BQ, BKV>(Op, Ml, Q, K, V, O, sq, skv, nh, nkv, qpos0, win, nsplit,
                                 kv_stride, /*kv_mask*/0xFFFFFFFFu, scale, blockIdx.x, gridDim.x, sm);
}

template <int HD, int BQ, int BKV>
static double bench(const char* label, unsigned nh, unsigned nkv, unsigned seq_q, unsigned seq_kv,
                    unsigned window, int iters) {
    const unsigned q_pos0 = (seq_kv > seq_q) ? (seq_kv - seq_q) : 0u; // last chunk attends full history
    const unsigned kv_stride = seq_kv;
    const float scale = 1.0f;
    size_t nQ = (size_t)seq_q * nh * HD, nKV = (size_t)nkv * seq_kv * HD, nO = (size_t)seq_q * nh * HD;
    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    CHK(cudaMalloc(&dQ, nQ * 2)); CHK(cudaMalloc(&dK, nKV * 2));
    CHK(cudaMalloc(&dV, nKV * 2)); CHK(cudaMalloc(&dO, nO * 2));
    CHK(cudaMemset(dQ, 0, nQ * 2)); CHK(cudaMemset(dK, 0, nKV * 2));
    CHK(cudaMemset(dV, 0, nKV * 2)); CHK(cudaMemset(dO, 0, nO * 2));

    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float);
    CHK(cudaFuncSetAttribute(k_fa_pre<HD, BQ, BKV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    const int grid = 188;
    // warmup
    for (int i = 0; i < 2; i++)
        k_fa_pre<HD, BQ, BKV><<<grid, 256, smem>>>(nullptr, nullptr, dO, dQ, dK, dV, seq_q, seq_kv,
                                                   nh, nkv, q_pos0, window, 1, kv_stride, scale);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        k_fa_pre<HD, BQ, BKV><<<grid, 256, smem>>>(nullptr, nullptr, dO, dQ, dK, dV, seq_q, seq_kv,
                                                   nh, nkv, q_pos0, window, 1, kv_stride, scale);
    cudaEventRecord(b); CHK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    printf("  %-52s smem=%.1fKiB  %8.3f ms/op\n", label, smem / 1024.0, ms);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

int main() {
    cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s  sm_%d%d  SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("smemOptin=%zu KiB\n\n", p.sharedMemPerBlockOptin / 1024);

    printf("== hd512 FULL causal (nh16 nkv1) — the O(ctx^2) tail ==\n");
    bench<512, 32, 16>("hd512 sq8192 skv8192  (early)",  16, 1, 8192,   8192,   0, 20);
    bench<512, 32, 16>("hd512 sq8192 skv32768 (mid)",    16, 1, 8192,   32768,  0, 10);
    bench<512, 32, 16>("hd512 sq8192 skv131072 (tail)",  16, 1, 8192,   131072, 0, 5);

    printf("\n== hd256 SLIDING window=1024 (nh16 nkv8) — O(window) ==\n");
    bench<256, 64, 32>("hd256 sq8192 skv8192  win1024",  16, 8, 8192,   8192,   1024, 20);
    bench<256, 64, 32>("hd256 sq8192 skv131072 win1024", 16, 8, 8192,   131072, 1024, 10);

    printf("\n== hd256 FULL causal (nh16 nkv8) — sanity (no window) ==\n");
    bench<256, 64, 32>("hd256 sq8192 skv8192  causal",   16, 8, 8192,   8192,   0, 20);
    bench<256, 64, 32>("hd256 sq8192 skv32768 causal",   16, 8, 8192,   32768,  0, 10);
    return 0;
}
