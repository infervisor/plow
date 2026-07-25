/* flashpre_slide_bw_sm120.cu — CEILING probe for the hd256 SLIDING-WINDOW flash-PREFILL arm
 * (beat-sliding-fp8mma step 1/2). Measures the CURRENT bf16 sliding kernel
 * d_flash_prefill<256,64,32> (PIPE=1 generic body) at window=1024, chunk regime, Gemma-4 26B
 * sliding geometry (16 q-heads / 8 kv-heads, hd256). Answers, BEFORE any kernel work:
 *   (1) absolute ms of one chunk-layer -> total sliding-flash time @128k -> share of TTFT.
 *   (2) achieved effective bandwidth vs L2 roofline  AND  compute floor (bf16/fp8 mma rate)
 *       -> is the sliding arm COMPUTE-bound (fp8 mma helps) or MEMORY/L2-bound (it does not)?
 *
 * Build (outside nix develop):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/flashpre_slide_bw_sm120.cu -o slide_bw
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_prefill) */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x13572468u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }

static const unsigned NH = 16, NKV = 8, HD = 256; /* Gemma-4 26B SLIDING geometry */
static const int BQ = 64, BKV = 32;
static const unsigned WINDOW = 1024;

__global__ void __launch_bounds__(256,1)
k_slide_bf16(float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, bf16* O,
             unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
             unsigned window, unsigned nsplit, unsigned kv_stride, float scale) {
    extern __shared__ float arena[];
    d_flash_prefill<256,64,32>(Opart,mlpart,Q,K,V,O,seq_q,seq_kv,n_head,n_kv_head,q_pos0,
                               window,nsplit,kv_stride,0xFFFFFFFFu,scale,
                               blockIdx.x,gridDim.x,arena,nullptr);
}

int main(int argc, char** argv){
    int iters = argc>1?atoi(argv[1]):20;
    unsigned seq_q = argc>2?(unsigned)atoi(argv[2]):8192u; /* one chunk of queries */
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("# device %s, SMs %d, iters %d\n", p.name, p.multiProcessorCount, iters);
    printf("# SLIDING geom: n_head=%u n_kv_head=%u HD=%u BQ=%d BKV=%d window=%u  seq_q(chunk)=%u\n",
           NH,NKV,HD,BQ,BKV,WINDOW,seq_q);
    const size_t smem=(size_t)FA_PRE_SMEM_FLOATS(256,64,32)*sizeof(float);
    printf("# sliding arm smem = %zu B\n", smem);
    CK(cudaFuncSetAttribute(k_slide_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    const unsigned ctxs[]={8192u,32768u,131072u};
    const float scale = 1.f/sqrtf((float)HD);
    printf("\n%-8s | %10s | %12s | %12s | %10s\n","ctx","ms/chunklayer","effBW_GBps","compFloor_ms(bf16)","cbound?");
    for (unsigned ctx : ctxs){
        const unsigned q_pos0 = ctx - seq_q; /* queries at the tail -> full 1024 windows */
        const unsigned kvs = ctx;
        const size_t nkv = (size_t)NKV*kvs*HD;
        std::vector<bf16> k(nkv), v(nkv);
        for (size_t i=0;i<nkv;i++){ k[i]=__float2bfloat16(rnd()*0.5f); v[i]=__float2bfloat16(rnd()*0.5f); }
        bf16 *dK,*dV,*dQ,*dO; float *dOp,*dMl;
        CK(cudaMalloc(&dK,nkv*2)); CK(cudaMalloc(&dV,nkv*2));
        CK(cudaMemcpy(dK,k.data(),nkv*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,v.data(),nkv*2,cudaMemcpyHostToDevice));
        std::vector<bf16> q((size_t)seq_q*NH*HD); for (auto&x:q) x=__float2bfloat16(rnd()*0.5f);
        CK(cudaMalloc(&dQ,q.size()*2)); CK(cudaMemcpy(dQ,q.data(),q.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dOp,(size_t)seq_q*NH*HD*4)); CK(cudaMalloc(&dMl,(size_t)seq_q*NH*2*4));
        CK(cudaMalloc(&dO,(size_t)seq_q*NH*HD*2));
        const unsigned n_work=(seq_q/BQ)*NH; const unsigned grid=(n_work<188u)?n_work:188u;
        auto launch=[&](){ k_slide_bf16<<<grid,256,smem>>>(dOp,dMl,dQ,dK,dV,dO,seq_q,kvs,NH,NKV,q_pos0,WINDOW,1u,kvs,scale); };
        for(int w=0;w<3;w++) launch(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        double best=1e30;
        for(int it=0;it<iters;it++){ CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b)); float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best)best=ms; }
        cudaEventDestroy(a); cudaEventDestroy(b);
        /* effective KV bytes STREAMED smem<-L2: per (head,qtile) ~ceil((window+BQ)/BKV) kv tiles,
         * each BKV*HD bf16, K+V. This is the L2-served re-read volume the arm actually moves. */
        const unsigned ntiles = (WINDOW + BQ + BKV - 1)/BKV;
        const double effbytes = (double)NH*(seq_q/BQ)*ntiles*BKV*HD*2.0*2.0;
        const double effbw = effbytes/(best*1e-3)/1e9;
        /* compute floor: QK + PV, window-bounded, bf16 mma 465 TF/s. */
        const double flop = 2.0*(double)NH*seq_q*WINDOW*(2.0*HD); /* QK + PV */
        const double cfloor = flop/465e12*1e3;
        printf("%-8u | %13.4f | %12.1f | %12.4f | %10s\n", ctx, best, effbw, cfloor,
               (cfloor > best*0.5)?"COMPUTE":"MEMORY");
        cudaFree(dK);cudaFree(dV);cudaFree(dQ);cudaFree(dO);cudaFree(dOp);cudaFree(dMl);
    }
    printf("\n# total sliding-flash @128k(26B) = ms(ctx=128k) * (131072/%u chunks) * 25 layers\n", seq_q);
    return 0;
}
