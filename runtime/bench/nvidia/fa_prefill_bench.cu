/* Isolated bf16 flash-attention-prefill throughput, for Gemma-4-12B's real shapes on the
 * actual shipped tiling (hd256/BQ64/BKV32 sliding, hd512/BQ32/BKV16 full), RTX 5090 sm_120a.
 * No external reference available in this sandbox (flash-attn has no prebuilt wheel for this
 * torch/cuda combo and a from-source build wasn't attempted — reported honestly, not guessed).
 * Reference point instead: PX-9's own cycle-verified bf16 mma.sync ceiling on this exact GPU
 * class, 259.2 TFLOP/s (perf-data/px9-gemm-body.md Result 1).
 *
 * Build: nvcc -O3 -arch=sm_120a -I <repo>/runtime/common -I <repo>/runtime/nvidia \
 *   fa_prefill_bench.cu -o fabench
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

typedef __nv_bfloat16 bf16;
#include "sm120_common.cuh"

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)

static const int WARM = 5, ITERS = 20;

template <int HD, int BQ, int BKV>
__global__ void k_flash_prefill(float* Op, float* Ml, bf16* O, const bf16* Q, const bf16* K,
                                const bf16* V, unsigned sq, unsigned skv, unsigned nh, unsigned nkv,
                                unsigned qpos0, unsigned win, unsigned nsplit, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill<HD,BQ,BKV>(Op, Ml, Q, K, V, O, sq, skv, nh, nkv, qpos0, win, nsplit,
                               /*kv_stride*/skv, /*kv_mask*/0xFFFFFFFFu, scale, blockIdx.x,
                               gridDim.x, sm);
}

static bf16* dev_bf16(size_t n) {
    bf16* d; CK(cudaMalloc(&d, n*sizeof(bf16)));
    CK(cudaMemset(d, 0x3c, n*sizeof(bf16)));
    return d;
}

template <int HD, int BQ, int BKV>
static void bench(const char* label, unsigned nh, unsigned nkv, unsigned seq, unsigned window, int P) {
    bf16* Q = dev_bf16((size_t)seq*nh*HD);
    bf16* K = dev_bf16((size_t)nkv*seq*HD);
    bf16* V = dev_bf16((size_t)nkv*seq*HD);
    bf16* O = dev_bf16((size_t)seq*nh*HD);
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD,BQ,BKV)*sizeof(float);
    CK(cudaFuncSetAttribute(k_flash_prefill<HD,BQ,BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    auto run = [&](){
        k_flash_prefill<HD,BQ,BKV><<<P,256,smem>>>(nullptr,nullptr,O,Q,K,V,seq,seq,nh,nkv,0,window,1,1.0f/sqrtf((float)HD));
    };
    for (int i=0;i<WARM;i++) run();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i=0;i<ITERS;i++) run();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
    CK(cudaGetLastError());

    /* Effective FLOPs: causal triangle (window==0) or causal+sliding band (window>0), not naive
     * dense seq^2 -- the kernel skips fully-masked KV tiles, so a naive seq^2 count would make
     * this look artificially slow. QK^T + P.V, 2 matmuls, 2*HD FLOPs/MAC. */
    double pairs;
    if (window == 0) {
        pairs = (double)seq*(seq+1)/2.0; /* causal triangle */
    } else { /* causal AND within window: for q, kv in [max(0,q-window+1), q] */
        pairs = 0;
        for (unsigned q=0;q<seq;q++) { unsigned lo=(q>=window)?(q-window+1):0; pairs += (double)(q-lo+1); }
    }
    double fl = 4.0 * pairs * (double)nh * (double)HD; /* QK^T + PV, 2 FLOP/MAC each */
    double tf = fl/(ms*1e-3)/1e12;
    printf("%-20s HD=%-4d nh=%-3d nkv=%-3d seq=%-6u win=%-6u  ms=%8.4f  TFLOP/s=%8.1f  %%bf16peak(259.2)=%5.1f%%\n",
           label, HD, nh, nkv, seq, window, ms, tf, 100.0*tf/259.2);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int P = prop.multiProcessorCount;
    printf("SMs=%d\n", P);
    /* Gemma-4-12B: 16 heads, hd256 sliding (kvh=8, gqa=2), hd512 full (kvh=1, gqa=16). */
    bench<256,64,32>("hd256 sliding (shipped)", 16, 8, 8192, 1024, P);
    bench<512,32,16>("hd512 full (shipped)",    16, 1, 8192, 0,    P);
    return 0;
}
