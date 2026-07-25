// decode_seg_gemv.cu — Step 1 of the SEGMENTED-DECODE mission (Gemma-4-26B-A4B, H100 sm_90a).
//
// Question: does a LEAN high-occupancy GEMV object raise the achieved HBM% of the
// REAL 26B decode GEMV op sequence (qkv/o/gate_up/down + grouped MoE-experts) to the
// ~46-58% the isolated ksplit_gemv probe hit at 3 blocks/SM — IN CONTEXT, back-to-back,
// with weights read COLD (L2 flushed) exactly like block_op_bench.py measures vLLM?
//
// The kernel is production-faithful to op_gemm.cuh gemv_rows: one WARP owns one output
// row n; block `slice` owns columns [slice*per, slice*per+per), per=ceil(N/nblk),
// nblk=grid blocks; x read from GLOBAL (production reads ld_glob8(x+..), not smem-staged
// like ksplit) — the only representative diff from ksplit. 8 warps/block (PLOW_NV_WARPS).
//
// Occupancy is set by grid size (nblk = nSM*mult). The lean object fits `mult` blocks/SM
// (verified via cudaOccupancyMaxActiveBlocksPerMultiprocessor); the megakernel fits 1.
//
// MoE-experts: 8 active experts, FLAT one-warp-per-(expert,row) schedule (op_moe.cuh
// design: "flat, not expert-parallel" — every warp does equal work regardless of routing).
//   per expert: gate_up N=1408 K=2816, down N=2816 K=704 (moe_inter 704, top-8).
//
// Also (Step 3): per-segment relaunch cost — empty-kernel and launch+sync us.
//
// Build:  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -arch=sm_90a -O3 -Xptxas -v \
//              -o /tmp/dsg decode_seg_gemv.cu
// Run under gpulease.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const double HBM_SPEC = 3350.0;   // H100 NVL HBM3 spec GB/s
#define BLOCK 256u
#define WPB  (BLOCK/32u)                  // 8 warps/block = PLOW_NV_WARPS
#define GV_STEP 256u                      // 32 lanes * 8 elems

typedef struct { uint4 v; } bf16v8;
__device__ __forceinline__ bf16v8 ld8(const __nv_bfloat16* p){ bf16v8 r; r.v=*(const uint4*)p; return r; }
__device__ __forceinline__ float dot8(const bf16v8&a,const bf16v8&b,float acc){
  const __nv_bfloat16* A=(const __nv_bfloat16*)&a.v; const __nv_bfloat16* B=(const __nv_bfloat16*)&b.v;
  #pragma unroll
  for(int j=0;j<8;j++) acc=__fmaf_rn(__bfloat162float(A[j]),__bfloat162float(B[j]),acc);
  return acc;
}
__device__ __forceinline__ float wsum(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(~0u,v,o); return v; }

// Production gemv_rows: one warp/row, block `slice` owns column range, x from GLOBAL.
__global__ __launch_bounds__(BLOCK) void gemv_rows(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  for(unsigned n=n0+warp; n<n1; n+=WPB){
    const __nv_bfloat16* wr=W+(size_t)n*K; float acc=0.f;
    for(unsigned c=0;c<nchunk;c++){ unsigned k=c*GV_STEP+lane*8u;
      if(k<K){ acc=dot8(ld8(wr+k), ld8(x+k), acc);} }
    acc=wsum(acc); if(lane==0) C[n]=__float2bfloat16(acc);
  }
}

// FLAT grouped MoE gate_up: rows are (expert e, out-row n in [0,Ngu)); warp owns one row.
__global__ __launch_bounds__(BLOCK) void moe_gateup(
    __nv_bfloat16* __restrict__ FU, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* const* __restrict__ Wgu, unsigned E, unsigned Ngu, unsigned K, unsigned nblk){
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned rows=E*Ngu; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/Ngu, n=r%Ngu; const __nv_bfloat16* wr=Wgu[e]+(size_t)n*K; float acc=0.f;
    for(unsigned c=0;c<nchunk;c++){ unsigned k=c*GV_STEP+lane*8u; if(k<K) acc=dot8(ld8(wr+k),ld8(x+k),acc);}
    acc=wsum(acc); if(lane==0) FU[(size_t)e*Ngu+n]=__float2bfloat16(acc);
  }
}
// FLAT grouped MoE down: rows are (expert e, out-row n in [0,H)); K=moe_inter.
__global__ __launch_bounds__(BLOCK) void moe_down(
    __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ FU,
    const __nv_bfloat16* const* __restrict__ Wd, unsigned E, unsigned H, unsigned Kmi, unsigned nblk){
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(Kmi+GV_STEP-1)/GV_STEP;
  const unsigned rows=E*H; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/H, n=r%H; const __nv_bfloat16* wr=Wd[e]+(size_t)n*Kmi;
    const __nv_bfloat16* fu=FU+(size_t)e*Kmi; float acc=0.f;
    for(unsigned c=0;c<nchunk;c++){ unsigned k=c*GV_STEP+lane*8u; if(k<Kmi) acc=dot8(ld8(wr+k),ld8(fu+k),acc);}
    acc=wsum(acc); if(lane==0) Y[(size_t)e*H+n]=__float2bfloat16(acc);  // no cross-expert combine (BW test)
  }
}

__global__ void empty_k(){}

static __nv_bfloat16* rnd_dev(size_t n, unsigned seed){
  std::vector<__nv_bfloat16> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)((i*1103515245u+seed*2654435761u+12345u)); h[i]=__float2bfloat16(((r>>8)&255)/255.f-0.5f);}
  __nv_bfloat16* d; CHK(cudaMalloc(&d,n*2)); CHK(cudaMemcpy(d,h.data(),n*2,cudaMemcpyHostToDevice)); return d;
}

static char* g_flush=nullptr; static const size_t FLUSH=192ull<<20;  // >H100 50MB L2
static void flushL2(){ CHK(cudaMemset(g_flush,0,FLUSH)); }

// time a single dense gemv (W cold each rep)
static double time_gemv(__nv_bfloat16*C,__nv_bfloat16*x,__nv_bfloat16*W,unsigned N,unsigned K,unsigned nblk){
  cudaEvent_t a,b; CHK(cudaEventCreate(&a));CHK(cudaEventCreate(&b));
  for(int i=0;i<5;i++){ flushL2(); gemv_rows<<<nblk,BLOCK>>>(C,x,W,N,K,nblk);} CHK(cudaDeviceSynchronize());
  const int IT=60; double acc=0;
  for(int i=0;i<IT;i++){ flushL2(); CHK(cudaDeviceSynchronize());
    CHK(cudaEventRecord(a)); gemv_rows<<<nblk,BLOCK>>>(C,x,W,N,K,nblk); CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b)); float ms; CHK(cudaEventElapsedTime(&ms,a,b)); acc+=ms;}
  CHK(cudaEventDestroy(a));CHK(cudaEventDestroy(b)); return acc/IT*1e3; // us
}

struct Res{ std::string name; double bytes; double us[4]; };

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); int nSM=p.multiProcessorCount;
  CHK(cudaMalloc(&g_flush,FLUSH));
  printf("device %s  SMs=%d  L2=%.0f MB  HBM_spec=%.0f GB/s\n",p.name,nSM,p.l2CacheSize/1e6,HBM_SPEC);

  int occ; CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)gemv_rows,BLOCK,0));
  printf("gemv_rows: cudaOccupancyMaxActiveBlocksPerMultiprocessor = %d blocks/SM (lean object ceiling)\n\n",occ);

  const unsigned H=2816;
  struct Shape{const char*name;unsigned N,K;} sh[]={
    {"qkv_proj  (N8192,K2816)",8192,2816},
    {"o_proj    (N2816,K4096)",2816,4096},
    {"gate_up   (N4224,K2816)",4224,2816},
    {"down_proj (N2816,K2112)",2816,2112},
  };
  std::vector<Res> R;
  printf("== DENSE decode GEMV (M=1, W cold via L2 flush), occupancy = grid/nSM ==\n");
  printf("%-26s %10s | %s\n","shape","weightMB","us / GB/s / %HBM  at 1,2,3 blocks/SM");
  for(auto&s:sh){
    size_t wn=(size_t)s.N*s.K; double gb=(double)wn*2/1e9;
    __nv_bfloat16*W=rnd_dev(wn,s.N+s.K), *x=rnd_dev(s.K,7u), *C; CHK(cudaMalloc(&C,s.N*2));
    Res r; r.name=s.name; r.bytes=(double)wn*2;
    printf("%-26s %9.1f | ",s.name,gb*1e3);
    for(int m=1;m<=3;m++){ unsigned nblk=nSM*m; double us=time_gemv(C,x,W,s.N,s.K,nblk); r.us[m]=us;
      double bw=gb/(us/1e6); printf("[%d] %6.1fus %6.0f %2.0f%%  ",m,us,bw,100*bw/HBM_SPEC);}
    printf("\n"); R.push_back(r);
    cudaFree(W);cudaFree(x);cudaFree(C);
  }

  // MoE-experts: 8 experts, gate_up N=1408 K=2816, down N=2816(->H rows) K=704.
  printf("\n== MoE-experts grouped GEMV (8 active experts, flat schedule, W cold) ==\n");
  const unsigned E=8, Ngu=1408, Kmi=704;
  std::vector<__nv_bfloat16*> hWgu(E),hWd(E);
  for(unsigned e=0;e<E;e++){ hWgu[e]=rnd_dev((size_t)Ngu*H,100+e); hWd[e]=rnd_dev((size_t)H*Kmi,200+e);}
  const __nv_bfloat16 **dWgu,**dWd; CHK(cudaMalloc(&dWgu,E*sizeof(void*))); CHK(cudaMalloc(&dWd,E*sizeof(void*)));
  CHK(cudaMemcpy(dWgu,hWgu.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dWd,hWd.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  __nv_bfloat16 *xh=rnd_dev(H,9u), *FU, *Yout; CHK(cudaMalloc(&FU,(size_t)E*Ngu*2)); CHK(cudaMalloc(&Yout,(size_t)E*H*2));
  double gu_gb=(double)E*Ngu*H*2/1e9, dn_gb=(double)E*H*Kmi*2/1e9, moe_gb=gu_gb+dn_gb;
  printf("gate_up bytes=%.1fMB  down bytes=%.1fMB  total=%.1fMB\n",gu_gb*1e3,dn_gb*1e3,moe_gb*1e3);
  cudaEvent_t a,b; CHK(cudaEventCreate(&a));CHK(cudaEventCreate(&b));
  Res moeR; moeR.name="moe_experts(8)"; moeR.bytes=moe_gb*1e9;
  printf("%-26s %9.1f | ","moe_experts(gu+down)",moe_gb*1e3);
  for(int m=1;m<=3;m++){ unsigned nblk=nSM*m;
    for(int i=0;i<5;i++){ flushL2(); moe_gateup<<<nblk,BLOCK>>>(FU,xh,dWgu,E,Ngu,H,nblk);
      moe_down<<<nblk,BLOCK>>>(Yout,FU,dWd,E,H,Kmi,nblk);} CHK(cudaDeviceSynchronize());
    const int IT=60; double us=0;
    for(int i=0;i<IT;i++){ flushL2(); CHK(cudaDeviceSynchronize()); CHK(cudaEventRecord(a));
      moe_gateup<<<nblk,BLOCK>>>(FU,xh,dWgu,E,Ngu,H,nblk);
      moe_down<<<nblk,BLOCK>>>(Yout,FU,dWd,E,H,Kmi,nblk);
      CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b)); float ms; CHK(cudaEventElapsedTime(&ms,a,b)); us+=ms;}
    us=us/IT*1e3; moeR.us[m]=us; double bw=moe_gb/(us/1e6);
    printf("[%d] %6.1fus %6.0f %2.0f%%  ",m,us,bw,100*bw/HBM_SPEC);}
  printf("\n"); R.push_back(moeR);

  // ---- Sequence total: dense GEMV family + moe, at each occupancy ----
  printf("\n== decode GEMV-family SEQUENCE total (qkv+o+gate_up+down+moe_experts) ==\n");
  for(int m=1;m<=3;m++){ double tot=0,byt=0; for(auto&r:R){ tot+=r.us[m]; byt+=r.bytes;}
    double bw=byt/1e9/(tot/1e6);
    printf("  %d blocks/SM: SUM=%7.1f us   agg %5.0f GB/s (%2.0f%% HBM)\n",m,tot,bw,100*bw/HBM_SPEC);}
  printf("  vLLM GEMV-family per-op sum (measured) = 147 us ; whole decode block = 356 us\n");

  // ---- Step 3: per-segment relaunch overhead ----
  printf("\n== Step 3: per-segment relaunch/sync overhead ==\n");
  { const int IT=2000; CHK(cudaDeviceSynchronize());
    CHK(cudaEventRecord(a)); for(int i=0;i<IT;i++){ empty_k<<<nSM,BLOCK>>>(); } CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b)); float ms; CHK(cudaEventElapsedTime(&ms,a,b));
    printf("  empty-kernel launch (async, amortized): %.2f us/launch\n",ms/IT*1e3);
    double each=0; for(int i=0;i<400;i++){ CHK(cudaEventRecord(a)); empty_k<<<nSM,BLOCK>>>();
      CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b)); float m2; CHK(cudaEventElapsedTime(&m2,a,b)); each+=m2;}
    printf("  empty-kernel launch+SYNC (per-segment cost): %.2f us/launch\n",each/400*1e3);
  }
  printf("\nDONE\n");
  return 0;
}
