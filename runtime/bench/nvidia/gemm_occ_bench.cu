/* gemm_occ_bench.cu — PX-3 (rtx-11) standalone per-op GEMM A/B: BN=128 occ-1 vs BN=64 occ-2.
 *
 * The lean-GEMM megakernel object cannot be driven end-to-end at occ-2 without the emitter
 * re-slicing GEMM segments to 2*n_cu (the Stage-3 prerequisite, not built). But the tiled GEMM
 * body (d_gemm / d_gemm_glu) is a DEDICATED kernel whose occupancy is set by the GEMM arena +
 * registers ALONE (no flash arm in the union), exactly like the lean object. So k_gemm launched
 * at grid=2*n_cu with BN=64 gives a FAITHFUL occ-2 measurement of the same tile the lean object
 * would run — an honest per-op answer to "does BN=64/occ-2 beat BN=128/occ-1".
 *
 * Built twice: default PGM_BN=128 (occ-1 target) and -DPGM_BN=64 (occ-2 target). Each build reports
 * its k_gemm achieved occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor) and per-shape ms
 * at grid=188 (1 block/SM) and grid=376 (2 blocks/SM) at the 12B prefill shapes. */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "sm120_common.cuh"
#include "op_gemm.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA FAIL %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); exit(1);} } while(0)

static uint32_t rng=12345u; static float frand(){ rng=rng*1664525u+1013904223u; return ((rng>>8)&0xffff)/65535.0f-0.5f; }
static bf16* dev_rand(size_t n){ std::vector<float> h(n); for(size_t i=0;i<n;i++) h[i]=frand();
    std::vector<bf16> hb(n); for(size_t i=0;i<n;i++) hb[i]=__float2bfloat16(h[i]);
    bf16* d; CK(cudaMalloc(&d,n*sizeof(bf16))); CK(cudaMemcpy(d,hb.data(),n*sizeof(bf16),cudaMemcpyHostToDevice)); return d; }

__global__ void k_gemm(bf16* C, const bf16* A, const bf16* B, unsigned m, unsigned n, unsigned k, unsigned a_row0){
    extern __shared__ bf16 smg[]; d_gemm(C,A,B,m,n,k,a_row0,blockIdx.x,gridDim.x,smg);
}
__global__ void k_gemm_glu(bf16* C, const bf16* A, const bf16* Wg, const bf16* Wu, unsigned m, unsigned n, unsigned k, unsigned act){
    extern __shared__ bf16 smg[]; d_gemm_glu(C,A,Wg,Wu,m,n,k,act,blockIdx.x,gridDim.x,smg);
}

static const int ITERS=30, WARM=8;

static double time_gemm(bf16*C,bf16*A,bf16*B,unsigned m,unsigned n,unsigned k,size_t smem,int grid){
    for(int i=0;i<WARM;i++) k_gemm<<<grid,256,smem>>>(C,A,B,m,n,k,0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    for(int i=0;i<ITERS;i++) k_gemm<<<grid,256,smem>>>(C,A,B,m,n,k,0);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms=0; CK(cudaEventElapsedTime(&ms,s,e)); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms/ITERS;
}
static double time_glu(bf16*C,bf16*A,bf16*Wg,bf16*Wu,unsigned m,unsigned n,unsigned k,size_t smem,int grid){
    for(int i=0;i<WARM;i++) k_gemm_glu<<<grid,256,smem>>>(C,A,Wg,Wu,m,n,k,0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    for(int i=0;i<ITERS;i++) k_gemm_glu<<<grid,256,smem>>>(C,A,Wg,Wu,m,n,k,0);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms=0; CK(cudaEventElapsedTime(&ms,s,e)); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms/ITERS;
}

struct Shape{ const char* name; unsigned n,k; int glu; };

int main(){
    const size_t smem=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_gemm_glu,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    int occ=0,occg=0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k_gemm,256,smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occg,k_gemm_glu,256,smem));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa,k_gemm));
    printf("BN=%d arena_bf16=%d smem_bytes=%zu smem_KiB=%.1f k_gemm_regs=%d occ_gemm=%d occ_glu=%d\n",
           PGM_BN, PGM_ARENA_BF16, smem, smem/1024.0, fa.numRegs, occ, occg);

    // 12B prefill shapes (K=hidden 3840). q N4096 K3840; o N3840 K4096; down N3840 K15360;
    // gate/up GLU N15360 K3840. M = chunk rows (4k / 8k prefill).
    Shape shapes[]={
        {"q_proj",   4096, 3840, 0},
        {"o_proj",   3840, 4096, 0},
        {"down_proj",3840,15360, 0},
        {"gateup_glu",15360,3840, 1},
    };
    unsigned Ms[]={4096,8192};
    printf("shape,M,N,K,glu,grid188_ms,grid376_ms\n");
    for(unsigned mi=0;mi<2;mi++){ unsigned M=Ms[mi];
        for(auto&sh:shapes){
            bf16* A=dev_rand((size_t)M*sh.k);
            bf16* Bg=dev_rand((size_t)sh.n*sh.k);
            bf16* Bu=sh.glu?dev_rand((size_t)sh.n*sh.k):nullptr;
            bf16* C=nullptr; CK(cudaMalloc(&C,(size_t)M*sh.n*sizeof(bf16)));
            double t188,t376;
            if(sh.glu){ t188=time_glu(C,A,Bg,Bu,M,sh.n,sh.k,smem,188);
                        t376=time_glu(C,A,Bg,Bu,M,sh.n,sh.k,smem,376); }
            else{ t188=time_gemm(C,A,Bg,M,sh.n,sh.k,smem,188);
                  t376=time_gemm(C,A,Bg,M,sh.n,sh.k,smem,376); }
            printf("%s,%u,%u,%u,%d,%.4f,%.4f\n",sh.name,M,sh.n,sh.k,sh.glu,t188,t376);
            cudaFree(A);cudaFree(Bg);if(Bu)cudaFree(Bu);cudaFree(C);
        }
    }
    return 0;
}
