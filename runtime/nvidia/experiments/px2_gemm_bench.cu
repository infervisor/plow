// rtx-11 PX-2 bench: isolated GEMM-BODY timing for the w8a8 prefill GEMM vs bf16, at real
// Gemma-4-12B prefill projection shapes. The persistent-interpreter overhead is absent (the
// kernel IS just the tiled GEMM over the whole [M,N] tile grid on 188 blocks), so the kernel
// time ratio IS the GEMM-body ratio the campaign quotes.
//
//   T8 baseline vs PX-2: build twice, once with -I on a T8 op_gemm.cuh copy, once on HEAD.
//   nvcc -arch=sm_120a -O3 -I ../common -I .. -DPLOW_NV_GEMMA=1 -DPLOW_NV_PREFILL=1 \
//        -DPLOW_NV_FA_GF=2 px2_gemm_bench.cu -o /tmp/px2bench && /tmp/px2bench
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define PLOW_NV_GEMMA 1
#define PLOW_NV_PREFILL 1
#ifndef PLOW_NV_FA_GF
#define PLOW_NV_FA_GF 2
#endif
#include "sm120_common.cuh"
#include "op_gemm.cuh"

using bf16 = __nv_bfloat16;
#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(2);} }while(0)
static uint8_t e4m3_enc(float v){ __nv_fp8_e4m3 q(v); return *(const uint8_t*)&q; }

__global__ void k_bf16(bf16* C, const bf16* A, const bf16* B, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm(C, A, B, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_w8a8(bf16* C, const uint8_t* A, const uint8_t* B, const float* as,
                       const float* ws, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm_w8a8(C, A, B, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}

struct Shape { const char* name; unsigned n, k; };

int main() {
    // Gemma-4-12B: hidden 3840, interm 15360, q 4096 (16*256), kv 2048 (8*256).
    Shape shapes[] = {
        {"q_proj  N4096 K3840", 4096, 3840},
        {"o_proj  N3840 K4096", 3840, 4096},
        {"down    N3840 K15360", 3840, 15360},
        {"gate    N15360 K3840", 15360, 3840},   // (one arm of gate|up; GLU runs two)
    };
    unsigned Ms[] = {512, 2048, 4096};
    const size_t smem = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    const int ITER = 60, WARM = 10;
    printf("# PX-2 GEMM-body bench (sm_120, 188 blocks, %d iters)\n", ITER);
    printf("# %-22s %6s | %10s %10s %8s\n", "shape", "M", "bf16_us", "w8a8_us", "w8a8/bf16");
    for (unsigned M : Ms) {
        for (Shape s : shapes) {
            unsigned n = s.n, k = s.k;
            std::vector<float> Af((size_t)M*k), Bf((size_t)n*k), as(M,1.f), ws(n,1.f);
            std::vector<bf16> Ab((size_t)M*k), Bb((size_t)n*k);
            std::vector<uint8_t> A8((size_t)M*k), B8((size_t)n*k);
            for (size_t i=0;i<(size_t)M*k;i++){ float v=((i*2654435761u>>8)&255)/255.f-0.5f; Ab[i]=__float2bfloat16(v); A8[i]=e4m3_enc(v);}
            for (size_t i=0;i<(size_t)n*k;i++){ float v=((i*40503u>>7)&255)/255.f-0.5f; Bb[i]=__float2bfloat16(v); B8[i]=e4m3_enc(v);}
            bf16 *dC,*dAb,*dBb; uint8_t *dA8,*dB8; float *das,*dws;
            CK(cudaMalloc(&dC,(size_t)M*n*2));
            CK(cudaMalloc(&dAb,(size_t)M*k*2)); CK(cudaMalloc(&dBb,(size_t)n*k*2));
            CK(cudaMalloc(&dA8,(size_t)M*k)); CK(cudaMalloc(&dB8,(size_t)n*k));
            CK(cudaMalloc(&das,M*4)); CK(cudaMalloc(&dws,n*4));
            CK(cudaMemcpy(dAb,Ab.data(),(size_t)M*k*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dBb,Bb.data(),(size_t)n*k*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dA8,A8.data(),(size_t)M*k,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB8,B8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(das,as.data(),M*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dws,ws.data(),n*4,cudaMemcpyHostToDevice));
            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto timeit=[&](auto launch)->float{
                for(int i=0;i<WARM;i++) launch();
                CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
                for(int i=0;i<ITER;i++) launch();
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms=0; cudaEventElapsedTime(&ms,e0,e1); return ms/ITER*1000.f; // us
            };
            float tb=timeit([&]{ k_bf16<<<188,256,smem>>>(dC,dAb,dBb,M,n,k); });
            float tw=timeit([&]{ k_w8a8<<<188,256,smem>>>(dC,dA8,dB8,das,dws,M,n,k); });
            printf("  %-22s %6u | %10.2f %10.2f %8.3f\n", s.name, M, tb, tw, tw/tb);
            cudaFree(dC);cudaFree(dAb);cudaFree(dBb);cudaFree(dA8);cudaFree(dB8);cudaFree(das);cudaFree(dws);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
    }
    return 0;
}
