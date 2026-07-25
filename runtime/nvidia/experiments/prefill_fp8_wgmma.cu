// prefill_fp8_wgmma.cu — H100 sm_90a per-op PREFILL micro-bench of plow's PRODUCTION fp8 w8a8
// wgmma GEMM bodies (d_gemm_w8a8_sm90 / d_gemm_glu_w8a8_sm90) at the REAL Gemma-4-26B-A4B prefill
// projection shapes, vs plow's OWN bf16 wgmma prefill (d_gemm_sm90 / d_gemm_glu_sm90).
//
// These are the SAME device bodies the interp_sm90a_pf cubin dispatches; the persistent-interpreter
// overhead is absent (the kernel IS the tiled GEMM over the whole [M,N] tile grid on 132 blocks =
// 1 block/SM, the megakernel regime), so the measured kernel time IS the GEMM-body time.
//
// Reports per (shape, M): bf16 us, w8a8 us, TFLOP/s and %peak for each, and the w8a8-vs-bf16
// output relL2 (fp8 is lossy — this is the sanity that it is a valid fp8 result, not the bf16 one).
//
// H100 dense tensor-core peaks: fp8 e4m3 ~1979 TFLOP/s, bf16 ~989.5 TFLOP/s.
//
// Build (see task env):
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
//     -I runtime/common -I runtime/nvidia -DPLOW_NV_HOPPER=1 -DPGM90_FORK_GLU=1 \
//     -o /tmp/pfp8 runtime/nvidia/experiments/prefill_fp8_wgmma.cu
// Run under gpulease.
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

static const int NBLK = 132;              // H100 NVL SMs = 1 block/SM (megakernel regime)
static const double PEAK_FP8 = 1979.0;    // H100 e4m3 dense TFLOP/s
static const double PEAK_BF16 = 989.5;    // H100 bf16 dense TFLOP/s

__global__ void k_bf16(bf16* C, const bf16* A, const bf16* B, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm(C, A, B, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_w8a8(bf16* C, const uint8_t* A, const uint8_t* B, const float* as,
                       const float* ws, unsigned m, unsigned n, unsigned k) {
    extern __shared__ bf16 sm[];
    d_gemm_w8a8(C, A, B, as, ws, m, n, k, 0, blockIdx.x, gridDim.x, sm);
}
__global__ void k_glu_bf16(bf16* C, const bf16* A, const bf16* Wg, const bf16* Wu,
                           unsigned m, unsigned n, unsigned k, unsigned act) {
    extern __shared__ bf16 sm[];
    d_gemm_glu(C, A, Wg, Wu, m, n, k, act, blockIdx.x, gridDim.x, sm);
}
__global__ void k_glu_w8a8(bf16* C, const uint8_t* A, const uint8_t* Wg, const uint8_t* Wu,
                           const float* as, const float* sg, const float* su,
                           unsigned m, unsigned n, unsigned k, unsigned act) {
    extern __shared__ bf16 sm[];
    d_gemm_glu_w8a8(C, A, Wg, Wu, as, sg, su, m, n, k, act, blockIdx.x, gridDim.x, sm);
}

struct Shape { const char* name; unsigned n, k; };

static double relL2(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
        num += (x - y) * (x - y); den += y * y;
    }
    return sqrt(num / (den + 1e-30));
}

int main() {
    // Gemma-4-26B-A4B: hidden 2816, dense interm 2112, q 4096 (16*256), kv 2048 (8*256, v=k).
    Shape plain[] = {
        {"qkv    N6144 K2816", 6144, 2816},   // fused q(4096)+k(2048); v shares k (k_eq_v)
        {"o_proj N2816 K4096", 2816, 4096},
        {"down   N2816 K2112", 2816, 2112},
    };
    Shape glu = {"gate_up N2112 K2816", 2112, 2816};   // GLU: two GEMMs (gate,up), Gemma gelu-tanh
    unsigned Ms[] = {1024, 4096};
    const size_t smem = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_glu_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_glu_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    const int ITER = 80, WARM = 15;
    printf("# prefill fp8 w8a8 wgmma bench — Gemma-4-26B-A4B, sm_90a, %d blocks (1/SM), %d iters\n",
           NBLK, ITER);
    printf("# peaks: fp8 %.0f TFLOP/s  bf16 %.1f TFLOP/s\n", PEAK_FP8, PEAK_BF16);
    printf("# %-20s %5s | %8s %8s %6s | %8s %8s %6s %6s | %8s\n",
           "shape", "M", "bf16us", "bfTF/s", "%pk", "w8a8us", "w8TF/s", "%fp8", "%bf16", "relL2");

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timeit = [&](auto launch) -> float {
        for (int i = 0; i < WARM; i++) launch();
        CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
        for (int i = 0; i < ITER; i++) launch();
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); return ms / ITER * 1000.f; // us
    };

    for (unsigned M : Ms) {
        for (Shape s : plain) {
            unsigned n = s.n, k = s.k;
            std::vector<bf16> Ab((size_t)M*k), Bb((size_t)n*k);
            std::vector<uint8_t> A8((size_t)M*k), B8((size_t)n*k);
            std::vector<float> as(M,1.f), ws(n,1.f);
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

            float tb=timeit([&]{ k_bf16<<<NBLK,256,smem>>>(dC,dAb,dBb,M,n,k); });
            std::vector<bf16> Cbf((size_t)M*n); CK(cudaMemcpy(Cbf.data(),dC,(size_t)M*n*2,cudaMemcpyDeviceToHost));
            float tw=timeit([&]{ k_w8a8<<<NBLK,256,smem>>>(dC,dA8,dB8,das,dws,M,n,k); });
            std::vector<bf16> Cw8((size_t)M*n); CK(cudaMemcpy(Cw8.data(),dC,(size_t)M*n*2,cudaMemcpyDeviceToHost));

            double fl = 2.0*M*n*k;
            double bTF = fl/tb/1e6, wTF = fl/tw/1e6;
            printf("  %-20s %5u | %8.1f %8.1f %5.1f | %8.1f %8.1f %5.1f %5.1f | %8.2e\n",
                   s.name, M, tb, bTF, 100*bTF/PEAK_BF16, tw, wTF, 100*wTF/PEAK_FP8, 100*wTF/PEAK_BF16,
                   relL2(Cw8,Cbf));
            cudaFree(dC);cudaFree(dAb);cudaFree(dBb);cudaFree(dA8);cudaFree(dB8);cudaFree(das);cudaFree(dws);
        }
        // GLU gate_up: two GEMMs fused -> 2*(2*M*N*K) flops.
        {
            unsigned n = glu.n, k = glu.k;
            std::vector<bf16> Ab((size_t)M*k), Wgb((size_t)n*k), Wub((size_t)n*k);
            std::vector<uint8_t> A8((size_t)M*k), Wg8((size_t)n*k), Wu8((size_t)n*k);
            std::vector<float> as(M,1.f), sg(n,1.f), su(n,1.f);
            for (size_t i=0;i<(size_t)M*k;i++){ float v=((i*2654435761u>>8)&255)/255.f-0.5f; Ab[i]=__float2bfloat16(v); A8[i]=e4m3_enc(v);}
            for (size_t i=0;i<(size_t)n*k;i++){ float v=((i*40503u>>7)&255)/255.f-0.5f; Wgb[i]=__float2bfloat16(v); Wg8[i]=e4m3_enc(v);
                                                float u=((i*2246822519u>>9)&255)/255.f-0.5f; Wub[i]=__float2bfloat16(u); Wu8[i]=e4m3_enc(u);}
            bf16 *dC,*dAb,*dWgb,*dWub; uint8_t *dA8,*dWg8,*dWu8; float *das,*dsg,*dsu;
            CK(cudaMalloc(&dC,(size_t)M*n*2));
            CK(cudaMalloc(&dAb,(size_t)M*k*2)); CK(cudaMalloc(&dWgb,(size_t)n*k*2)); CK(cudaMalloc(&dWub,(size_t)n*k*2));
            CK(cudaMalloc(&dA8,(size_t)M*k)); CK(cudaMalloc(&dWg8,(size_t)n*k)); CK(cudaMalloc(&dWu8,(size_t)n*k));
            CK(cudaMalloc(&das,M*4)); CK(cudaMalloc(&dsg,n*4)); CK(cudaMalloc(&dsu,n*4));
            CK(cudaMemcpy(dAb,Ab.data(),(size_t)M*k*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWgb,Wgb.data(),(size_t)n*k*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWub,Wub.data(),(size_t)n*k*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dA8,A8.data(),(size_t)M*k,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWg8,Wg8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWu8,Wu8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(das,as.data(),M*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dsg,sg.data(),n*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dsu,su.data(),n*4,cudaMemcpyHostToDevice));

            float tb=timeit([&]{ k_glu_bf16<<<NBLK,256,smem>>>(dC,dAb,dWgb,dWub,M,n,k,PLOW_ACT_GELU_TANH_); });
            std::vector<bf16> Cbf((size_t)M*n); CK(cudaMemcpy(Cbf.data(),dC,(size_t)M*n*2,cudaMemcpyDeviceToHost));
            float tw=timeit([&]{ k_glu_w8a8<<<NBLK,256,smem>>>(dC,dA8,dWg8,dWu8,das,dsg,dsu,M,n,k,PLOW_ACT_GELU_TANH_); });
            std::vector<bf16> Cw8((size_t)M*n); CK(cudaMemcpy(Cw8.data(),dC,(size_t)M*n*2,cudaMemcpyDeviceToHost));

            double fl = 2.0*(2.0*M*n*k);   // gate + up
            double bTF = fl/tb/1e6, wTF = fl/tw/1e6;
            printf("  %-20s %5u | %8.1f %8.1f %5.1f | %8.1f %8.1f %5.1f %5.1f | %8.2e  (GLU x2)\n",
                   glu.name, M, tb, bTF, 100*bTF/PEAK_BF16, tw, wTF, 100*wTF/PEAK_FP8, 100*wTF/PEAK_BF16,
                   relL2(Cw8,Cbf));
            cudaFree(dC);cudaFree(dAb);cudaFree(dWgb);cudaFree(dWub);cudaFree(dA8);cudaFree(dWg8);cudaFree(dWu8);
            cudaFree(das);cudaFree(dsg);cudaFree(dsu);
        }
    }
    return 0;
}
