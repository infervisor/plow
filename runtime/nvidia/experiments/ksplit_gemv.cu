// ksplit_gemv.cu — does splitting K across blocks raise achieved HBM BW for the
// M=1 decode GEMV of the NON-headnorm dense projections? (Task 3, H100 sm_90a.)
//
// Target shapes (Gemma-4-26B-A4B, hidden 2816):
//   down_proj : y[N=2816] = W[N=2816][K=2112] . x[2112]
//   o_proj    : y[N=2816] = W[N=2816][K=4096] . x[4096]
// (The campaign flagged these as "K-split UNEXPLORED"; the MoE-expert down
//  (K=704) was already refuted in gemv_dimspec.cu at -30%.)
//
// Claim under test: N-split (one-warp-per-output-row, the production gemv_rows
// mapping) at nblk=132 already places one block on every SM, so the binding
// limit is WARPS-PER-SM occupancy, not SM fill. K-split multiplies the block
// count past the SM count (extra waves) and adds a cross-block reduce, so it
// should NOT raise achieved BW. Measures weight-BW = N*K*2/time for both.
//
// Build: /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 -Xptxas -v -o /tmp/ks ksplit_gemv.cu
// Run under gpulease.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static double HBM_SPEC = 3350.0; // H100 NVL HBM3 spec GB/s
#define BLOCK 256
#define WPB (BLOCK/32)

__device__ __forceinline__ float wred(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(~0u,v,o); return v; }
__device__ __forceinline__ float dot8b(const __nv_bfloat16* w,const __nv_bfloat16* x,float a){
  #pragma unroll
  for(int j=0;j<8;j++) a=__fmaf_rn(__bfloat162float(w[j]),__bfloat162float(x[j]),a); return a; }
__device__ __forceinline__ void ldv8(__nv_bfloat16 d[8],const __nv_bfloat16* p){ *(uint4*)d=*(const uint4*)p; }

// N-split: one warp per output row, grid-strided over N.
__global__ __launch_bounds__(BLOCK) void gv_nsplit(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y, int N, int K) {
  extern __shared__ __nv_bfloat16 xs[];
  for(int i=threadIdx.x;i<K;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  unsigned nch=(K+255)/256;
  for(int n=gw;n<N;n+=stride){
    const __nv_bfloat16* wr=W+(size_t)n*K; float acc=0;
    for(unsigned c=0;c<nch;c++){ unsigned k=c*256+lane*8; if(k<(unsigned)K){ __nv_bfloat16 wv[8]; ldv8(wv,wr+k); acc=dot8b(wv,xs+k,acc);} }
    acc=wred(acc); if(lane==0) Y[n]=__float2bfloat16(acc);
  }
}

// K-split: S warps cooperate per output row, each dots K/S; partial reduced to a
// global fp32 accumulator via atomicAdd. Adds a 2nd wave of blocks + the reduce.
__global__ __launch_bounds__(BLOCK) void gv_ksplit(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    float* __restrict__ Yacc, int N, int K, int S) {
  extern __shared__ __nv_bfloat16 xs[];
  for(int i=threadIdx.x;i<K;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int items=N*S, Ks=(K+S-1)/S;
  for(int it=gw; it<items; it+=stride){
    int n=it/S, s=it%S; int k0=s*Ks, k1=min(k0+Ks,K);
    const __nv_bfloat16* wr=W+(size_t)n*K; float acc=0;
    for(int k=k0+lane*8;k<k1;k+=256){
      if(k+8<=k1){ __nv_bfloat16 wv[8]; ldv8(wv,wr+k); acc=dot8b(wv,xs+k,acc); }
      else for(int j=k;j<k1;j++) acc=__fmaf_rn(__bfloat162float(wr[j]),__bfloat162float(xs[j]),acc);
    }
    acc=wred(acc); if(lane==0) atomicAdd(&Yacc[n],acc);
  }
}

static double time_ns(const __nv_bfloat16*W,const __nv_bfloat16*X,__nv_bfloat16*Y,int N,int K,int bl,int smem){
  cudaEvent_t a,b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
  for(int i=0;i<20;i++) gv_nsplit<<<bl,BLOCK,smem>>>(W,X,Y,N,K);
  CHK(cudaDeviceSynchronize());
  const int IT=300; CHK(cudaEventRecord(a));
  for(int i=0;i<IT;i++) gv_nsplit<<<bl,BLOCK,smem>>>(W,X,Y,N,K);
  CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b));
  float ms; CHK(cudaEventElapsedTime(&ms,a,b)); return (double)ms/IT*1e3;
}
static double time_ks(const __nv_bfloat16*W,const __nv_bfloat16*X,float*Ya,int N,int K,int S,int bl,int smem){
  cudaEvent_t a,b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
  for(int i=0;i<20;i++){ CHK(cudaMemset(Ya,0,(size_t)N*4)); gv_ksplit<<<bl,BLOCK,smem>>>(W,X,Ya,N,K,S); }
  CHK(cudaDeviceSynchronize());
  const int IT=300; CHK(cudaEventRecord(a));
  for(int i=0;i<IT;i++){ CHK(cudaMemset(Ya,0,(size_t)N*4)); gv_ksplit<<<bl,BLOCK,smem>>>(W,X,Ya,N,K,S); }
  CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b));
  float ms; CHK(cudaEventElapsedTime(&ms,a,b)); return (double)ms/IT*1e3;
}

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0));
  int nSM=p.multiProcessorCount;
  printf("device %s  SMs=%d  HBM_spec=%.0f GB/s  (K-split reduce cost NOT in the BW numerator)\n",p.name,nSM,HBM_SPEC);
  struct Shape{const char*name;int N,K;} shapes[]={
    {"down_proj (N2816,K2112)",2816,2112},
    {"o_proj    (N2816,K4096)",2816,4096},
    {"lm_head-ctrl(N32000,K2816)",32000,2816}};
  for(auto&sh:shapes){
    int N=sh.N,K=sh.K; size_t wn=(size_t)N*K; double gb=(double)wn*2/1e9;
    std::vector<__nv_bfloat16> hW(wn),hX(K);
    for(size_t i=0;i<wn;i++) hW[i]=__float2bfloat16((float)((i*1103515245u+12345u)&255)/255.f-0.5f);
    for(int i=0;i<K;i++) hX[i]=__float2bfloat16(0.01f*(i%7-3));
    __nv_bfloat16*dW,*dX,*dY; float*dYa;
    CHK(cudaMalloc(&dW,wn*2)); CHK(cudaMalloc(&dX,K*2)); CHK(cudaMalloc(&dY,N*2)); CHK(cudaMalloc(&dYa,N*4));
    CHK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dX,hX.data(),K*2,cudaMemcpyHostToDevice));
    int smem=K*2;
    printf("\n== %s  weight=%.1f MB ==\n",sh.name,gb*1e3);
    double best_ns=0; int best_b=0;
    printf("  N-split (one-warp-per-row):\n");
    for(int mult=1;mult<=4;mult++){ int bl=nSM*mult;
      double us=time_ns(dW,dX,dY,N,K,bl,smem); double bw=gb/(us/1e6);
      if(bw>best_ns){best_ns=bw;best_b=bl;}
      printf("    blocks=%-4d (%.1f/SM)  %8.2f us  %8.1f GB/s  (%2.0f%% HBM)\n",bl,(double)bl/nSM,us,bw,100*bw/HBM_SPEC);
    }
    double best_ks=0; int best_S=0,best_kb=0;
    printf("  K-split (S warps/row + atomicAdd reduce):\n");
    for(int S=2;S<=8;S*=2){
      int needw=N*S, bwant=(needw+WPB-1)/WPB, blk=std::min(std::max(bwant,nSM),nSM*4);
      double us=time_ks(dW,dX,dYa,N,K,S,blk,smem); double bw=gb/(us/1e6);
      if(bw>best_ks){best_ks=bw;best_S=S;best_kb=blk;}
      printf("    S=%d blocks=%-4d (%.1f/SM)  %8.2f us  %8.1f GB/s  (%2.0f%% HBM)\n",S,blk,(double)blk/nSM,us,bw,100*bw/HBM_SPEC);
    }
    printf("  >> best N-split %.1f GB/s (blocks=%d) vs best K-split %.1f GB/s (S=%d,blocks=%d)  => K-split %+.1f%%\n",
           best_ns,best_b,best_ks,best_S,best_kb,100*(best_ks-best_ns)/best_ns);
    cudaFree(dW);cudaFree(dX);cudaFree(dY);cudaFree(dYa);
  }
  return 0;
}
