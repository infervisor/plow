// decode_seg_cpasync.cu — DECISIVE H100 decode experiment (Gemma-4-26B-A4B, sm_90a).
//
// Combine the TWO levers already built:
//   (1) HIGH OCCUPANCY via the lean segment object (grid = nSM*mult → mult blocks/SM),
//       the lever that took the production N-split GEMV family 272→178→149us at occ 1/2/3
//       (perf-data/segmented-decode-26b-h100.md) but STALLED at occ-3 because the
//       megakernel spilled (80-reg cap → 8.5 KB spill/block eroded the memory-bound gain).
//   (2) The LOW-REGISTER cp.async row-staged GEMV (perf-data/gemv-cpasync-h100.md,
//       gemv_cpasync_h100.cu): 30-40 regs / 0 spill / smem-stages the weight+operand.
//
// HYPOTHESIS: because cp.async is 30-40 reg / 0 spill it can run occ 3/4/5/6 WITHOUT the
// spill that eroded the megakernel's occ-3, and because it smem-stages the operand its
// in-context %HBM should climb toward the isolated 46-58% rather than plateauing at 40%.
// If the GEMV-family aggregate drops below vLLM's 147us (per-layer < vLLM 161us after
// +flash ~23us +relaunch ~12us → GEMV-family must be < ~126us), plow decode BEATS vLLM.
//
// This probe = decode_seg_gemv.cu's FAITHFUL methodology (real 26B shapes, flat grouped
// MoE, weights COLD via 192 MB L2 flush, back-to-back, median >=100 reps) but with the
// cp.async GEMV kernel (ported depth-6 winner) at blocks/SM = grid/132 ∈ {1,2,3,4,5,6}.
// N-split column ownership kept EXACT (per=ceil(N/nblk), one warp/row). bit-exact vs the
// production gemv_rows (0 mismatches).
//
// Build: env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
//   nvcc -arch=sm_90a -O3 -Xptxas -v -I runtime/common -I runtime/nvidia \
//   -o /tmp/dscp runtime/nvidia/experiments/decode_seg_cpasync.cu
// Run:   LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dscp /tmp/dscp

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const double HBM_SPEC = 3350.0;   // H100 NVL HBM3 achievable GB/s
#define BLOCK    256u
#define WPB      (BLOCK/32u)              // 8 warps/block = PLOW_NV_WARPS
#define GV_STEP  256u                     // one warp pass over K: 32 lanes * 8 bf16
#define MAXOCC   6

// ---- production primitives (byte-identical to op_gemm.cuh gemv_rows) ----
struct bf16v8 { __nv_bfloat16 x[8]; };
__device__ __forceinline__ bf16v8 ld_glob8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ bf16v8 ld_smem8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ float dot8(const bf16v8&a,const bf16v8&b,float acc){
  #pragma unroll
  for(int i=0;i<8;i++) acc=__fmaf_rn(__bfloat162float(a.x[i]),__bfloat162float(b.x[i]),acc);
  return acc;
}
__device__ __forceinline__ float warp_sum32(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o,32);
  return v;
}
__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem, int src_bytes){
  unsigned s=(unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(s),"l"(gmem),"r"(src_bytes));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"::); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

// ---------- PRODUCTION gemv_rows (x from GLOBAL) — the bit-exact reference ----------
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
      if(k<K){ acc=dot8(ld_glob8(wr+k), ld_glob8(x+k), acc);} }
    acc=warp_sum32(acc); if(lane==0) C[n]=__float2bfloat16(acc);
  }
}

// ---------- cp.async row-staged worker (SAME lane ownership + dot8 order → bit-exact) --
// each warp cp.async-stages its row's 256-elem K-chunks into a depth-D smem ring and FMAs
// from smem; a constant D commit-groups stay in flight (wait_group<D-1>). ring[stage] slot
// per warp is warp*GV_STEP; lane l owns [l*8,l*8+8) → no cross-lane smem dependency.
template<int D>
__device__ __forceinline__ float cprow(const __nv_bfloat16* wrow, const __nv_bfloat16* xs,
                                        unsigned K, __nv_bfloat16* ring, unsigned lane, unsigned warp){
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned wbase   = warp*GV_STEP;
  const unsigned sstride = WPB*GV_STEP;      // one stage = 8 warps * 256
  #pragma unroll
  for(int p=0;p<D;p++){
    unsigned c=(unsigned)p, k=c*GV_STEP+lane*8u;
    __nv_bfloat16* dst=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
    cp_async_cg16(dst, wrow+k, (c<nchunk && k<K)?16:0); cp_commit();
  }
  float acc=0.f;
  for(unsigned c=0;c<nchunk;c++){
    cp_wait<D-1>();
    unsigned k=c*GV_STEP+lane*8u;
    const __nv_bfloat16* src=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
    if(k<K) acc=dot8(ld_smem8(src), ld_smem8(xs+k), acc);
    unsigned cn=c+(unsigned)D, kn=cn*GV_STEP+lane*8u;
    __nv_bfloat16* dst=ring+(cn%(unsigned)D)*sstride+wbase+lane*8u;
    cp_async_cg16(dst, wrow+kn, (cn<nchunk && kn<K)?16:0); cp_commit();
  }
  cp_wait<0>();
  return warp_sum32(acc);
}

// ---------- dense cp.async GEMV (N-split column ownership, x smem-staged) ----------
template<int D>
__global__ __launch_bounds__(BLOCK) void gemv_cp(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[];               // xs[K] then ring[D*WPB*256]
  __nv_bfloat16* xs=sm; __nv_bfloat16* ring=sm+K;
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  for(unsigned n=n0+warp; n<n1; n+=WPB){
    float t=cprow<D>(W+(size_t)n*K, xs, K, ring, lane, warp);
    if(lane==0) C[n]=__float2bfloat16(t);
  }
}

// ---------- FLAT grouped MoE gate_up (cp.async), operand x[K] smem-staged once ----------
template<int D>
__global__ __launch_bounds__(BLOCK) void moe_gateup_cp(
    __nv_bfloat16* __restrict__ FU, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* const* __restrict__ Wgu, unsigned E, unsigned Ngu, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[]; __nv_bfloat16* xs=sm; __nv_bfloat16* ring=sm+K;
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned rows=E*Ngu; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/Ngu, n=r%Ngu;
    float t=cprow<D>(Wgu[e]+(size_t)n*K, xs, K, ring, lane, warp);
    if(lane==0) FU[(size_t)e*Ngu+n]=__float2bfloat16(t);
  }
}
// ---------- FLAT grouped MoE down (cp.async), operand FU per-expert (Kmi each) staged ---
template<int D>
__global__ __launch_bounds__(BLOCK) void moe_down_cp(
    __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ FU,
    const __nv_bfloat16* const* __restrict__ Wd, unsigned E, unsigned H, unsigned Kmi, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[]; __nv_bfloat16* xs=sm; __nv_bfloat16* ring=sm+E*Kmi;
  for(unsigned i=threadIdx.x;i<E*Kmi;i+=blockDim.x) xs[i]=FU[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned rows=E*H; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/H, n=r%H;
    float t=cprow<D>(Wd[e]+(size_t)n*Kmi, xs+(size_t)e*Kmi, Kmi, ring, lane, warp);
    if(lane==0) Y[(size_t)e*H+n]=__float2bfloat16(t);
  }
}

__global__ void empty_k(){}

// ---------------------------------------------------------------- harness
static __nv_bfloat16* rnd_dev(size_t n, unsigned seed){
  std::vector<__nv_bfloat16> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)((i*1103515245u+seed*2654435761u+12345u));
    h[i]=__float2bfloat16(((r>>8)&255)/255.f-0.5f); }
  __nv_bfloat16* d; CHK(cudaMalloc(&d,n*2)); CHK(cudaMemcpy(d,h.data(),n*2,cudaMemcpyHostToDevice));
  return d;
}
static char* g_flush=nullptr; static const size_t FLUSH=192ull<<20;
static void flushL2(){ CHK(cudaMemset(g_flush,0,FLUSH)); }

template<class F>
static double time_median(F launch, int IT){
  for(int i=0;i<5;i++){ flushL2(); launch(); } CHK(cudaDeviceSynchronize());
  std::vector<double> ms; ms.reserve(IT);
  cudaEvent_t a,b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
  for(int i=0;i<IT;i++){ flushL2(); CHK(cudaDeviceSynchronize());
    CHK(cudaEventRecord(a)); launch(); CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b)); float t; CHK(cudaEventElapsedTime(&t,a,b)); ms.push_back(t); }
  CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
  std::sort(ms.begin(),ms.end());
  return ms[ms.size()/2]*1e3;   // us
}

// dense smem for a given K and depth D
static size_t dsmem(unsigned K,int D){ return (size_t)K*2 + (size_t)D*WPB*GV_STEP*2; }

template<int D>
static void launch_dense(__nv_bfloat16*C,__nv_bfloat16*x,__nv_bfloat16*W,unsigned N,unsigned K,unsigned nblk){
  gemv_cp<D><<<nblk,BLOCK,dsmem(K,D)>>>(C,x,W,N,K,nblk);
}

struct Res{ std::string name; double bytes; double us[MAXOCC+1]; double bw[MAXOCC+1]; };

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); int nSM=p.multiProcessorCount;
  CHK(cudaMalloc(&g_flush,FLUSH));
  printf("# device %s  SMs=%d  L2=%.0f MB  smemPerSM=%.0f KB  HBM_spec=%.0f GB/s\n",
         p.name,nSM,p.l2CacheSize/1e6,p.sharedMemPerMultiprocessor/1024.0,HBM_SPEC);
  printf("# cp.async GEMV depth=6, x/operand smem-staged, weights COLD (192MB L2 flush/rep), median 120 reps\n");
  printf("# occupancy = grid/nSM = mult blocks/SM (N-split per=ceil(N/nblk), one warp/row)\n\n");

  const int D=6, IT=120;

  // achievable-occupancy ceiling for the cp kernel at representative K
  for(unsigned K : {2816u,4096u,2112u}){
    int occ=0; CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)gemv_cp<6>,BLOCK,dsmem(K,D)));
    printf("# cudaOccupancyMaxActiveBlocksPerMultiprocessor gemv_cp<6> K=%u smem=%.1fKB : %d blocks/SM\n",
           K,dsmem(K,D)/1024.0,occ);
  }
  printf("\n");

  const unsigned H=2816;
  struct Shape{const char*name;unsigned N,K;bool perlayer;} sh[]={
    {"qkv_proj  (N8192,K2816)",8192,2816,true},
    {"o_proj    (N2816,K4096)",2816,4096,true},
    {"gate_up   (N4224,K2816)",4224,2816,true},
    {"down_proj (N2816,K2112)",2816,2112,true},
  };

  std::vector<Res> R;      // per-layer GEMV family (dense parts + moe)
  Res lmR; bool haveLM=false;

  printf("== DENSE decode GEMV (M=1, cp.async D=6, W cold), us / GB/s / %%HBM at 1..6 blocks/SM ==\n");
  for(auto&s:sh){
    size_t wn=(size_t)s.N*s.K; double gb=(double)wn*2/1e9;
    __nv_bfloat16*W=rnd_dev(wn,s.N+s.K), *x=rnd_dev(s.K,7u), *C; CHK(cudaMalloc(&C,(size_t)s.N*2));
    Res r; r.name=s.name; r.bytes=(double)wn*2;
    printf("%-26s %6.1fMB |",s.name,gb*1e3);
    for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ launch_dense<6>(C,x,W,s.N,s.K,nblk); },IT);
      double bw=gb/(us/1e6); r.us[m]=us; r.bw[m]=bw;
      printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
    printf("\n"); R.push_back(r);
    cudaFree(W);cudaFree(x);cudaFree(C);
  }

  // MoE-experts: 8 experts, gate_up N=1408 K=2816, down N=2816(->H rows) K=704.
  printf("\n== MoE-experts grouped GEMV (8 experts, flat cp.async D=6, W cold), 1..6 blocks/SM ==\n");
  const unsigned E=8, Ngu=1408, Kmi=704;
  std::vector<__nv_bfloat16*> hWgu(E),hWd(E);
  for(unsigned e=0;e<E;e++){ hWgu[e]=rnd_dev((size_t)Ngu*H,100+e); hWd[e]=rnd_dev((size_t)H*Kmi,200+e);}
  const __nv_bfloat16 **dWgu,**dWd; CHK(cudaMalloc(&dWgu,E*sizeof(void*))); CHK(cudaMalloc(&dWd,E*sizeof(void*)));
  CHK(cudaMemcpy(dWgu,hWgu.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dWd,hWd.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  __nv_bfloat16 *xh=rnd_dev(H,9u), *FU, *Yout; CHK(cudaMalloc(&FU,(size_t)E*Ngu*2)); CHK(cudaMalloc(&Yout,(size_t)E*H*2));
  double gu_gb=(double)E*Ngu*H*2/1e9, dn_gb=(double)E*H*Kmi*2/1e9, moe_gb=gu_gb+dn_gb;
  size_t gu_sm=dsmem(H,D), dn_sm=(size_t)E*Kmi*2+(size_t)D*WPB*GV_STEP*2;
  printf("gate_up=%.1fMB down=%.1fMB total=%.1fMB (gu_smem=%.1fKB dn_smem=%.1fKB)\n",
         gu_gb*1e3,dn_gb*1e3,moe_gb*1e3,gu_sm/1024.0,dn_sm/1024.0);
  Res moeR; moeR.name="moe_experts(8)"; moeR.bytes=moe_gb*1e9;
  printf("%-26s %6.1fMB |","moe_experts(gu+down)",moe_gb*1e3);
  for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
    double us=time_median([&]{
      moe_gateup_cp<6><<<nblk,BLOCK,gu_sm>>>(FU,xh,dWgu,E,Ngu,H,nblk);
      moe_down_cp<6><<<nblk,BLOCK,dn_sm>>>(Yout,FU,dWd,E,H,Kmi,nblk);
    },IT);
    double bw=moe_gb/(us/1e6); moeR.us[m]=us; moeR.bw[m]=bw;
    printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
  printf("\n"); R.push_back(moeR);

  // lm_head control (once-per-token) — big N, keeps scaling
  printf("\n== lm_head control (N262144,K2816, once-per-token), cp.async D=6, 1..6 blocks/SM ==\n");
  { unsigned N=262144, K=2816; size_t wn=(size_t)N*K; double gb=(double)wn*2/1e9;
    __nv_bfloat16*W=rnd_dev(wn,123u), *x=rnd_dev(K,7u), *C; CHK(cudaMalloc(&C,(size_t)N*2));
    lmR.name="lm_head(N262144)"; lmR.bytes=(double)wn*2; haveLM=true;
    printf("%-26s %6.0fMB |","lm_head",gb*1e3);
    for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ launch_dense<6>(C,x,W,N,K,nblk); },IT);
      double bw=gb/(us/1e6); lmR.us[m]=us; lmR.bw[m]=bw;
      printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
    printf("\n"); cudaFree(W);cudaFree(x);cudaFree(C);
  }

  // ---- GEMV-family aggregate (per-layer shapes: qkv+o+gate_up+down+moe = 200MB) ----
  printf("\n== GEMV-family SEQUENCE aggregate (qkv+o+gate_up+down+moe_experts = 200MB) ==\n");
  double best=1e30; int bestm=0;
  for(int m=1;m<=MAXOCC;m++){ double tot=0,byt=0; for(auto&r:R){ tot+=r.us[m]; byt+=r.bytes; }
    double bw=byt/1e9/(tot/1e6);
    printf("  %d blk/SM: AGG=%7.1f us  %5.0f GB/s (%2.0f%% HBM)\n",m,tot,bw,100*bw/HBM_SPEC);
    if(tot<best){best=tot;bestm=m;} }
  printf("  -> aggregate MIN = %.1f us at %d blocks/SM\n",best,bestm);
  printf("  vLLM GEMV-family per-op sum = 147 us ; per-layer floor = 161 us ; BEAT-vLLM target < ~126 us\n");
  if(haveLM){ printf("  lm_head (separate, once/token) best = %.1f us at ",
    [&]{double b=1e30;for(int m=1;m<=MAXOCC;m++)if(lmR.us[m]<b)b=lmR.us[m];return b;}());
    double b=1e30;int bm=0;for(int m=1;m<=MAXOCC;m++)if(lmR.us[m]<b){b=lmR.us[m];bm=m;} printf("%d blk/SM\n",bm);}

  // ---- per-shape optimum occupancy ----
  printf("\n== per-shape OPTIMUM occupancy (small-N starvation vs big-N scaling) ==\n");
  auto report_opt=[&](Res&r){ double b=1e30;int bm=0;for(int m=1;m<=MAXOCC;m++)if(r.us[m]<b){b=r.us[m];bm=m;}
    printf("  %-24s peak %6.1f us (%2.0f%% HBM) at %d blk/SM\n",r.name.c_str(),b,100*r.bw[bm]/HBM_SPEC,bm); };
  for(auto&r:R) report_opt(r);
  if(haveLM) report_opt(lmR);

  // ---- bit-exact vs production gemv_rows (qkv shape, occ-3) ----
  printf("\n== bit-exact check: gemv_cp<6> vs production gemv_rows (qkv N8192 K2816) ==\n");
  { unsigned N=8192,K=2816; size_t wn=(size_t)N*K;
    __nv_bfloat16*W=rnd_dev(wn,N+K), *x=rnd_dev(K,7u), *Cp,*Cc;
    CHK(cudaMalloc(&Cp,(size_t)N*2)); CHK(cudaMalloc(&Cc,(size_t)N*2));
    unsigned nblk=(unsigned)nSM*3;
    gemv_rows<<<nblk,BLOCK>>>(Cp,x,W,N,K,nblk);
    launch_dense<6>(Cc,x,W,N,K,nblk); CHK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hP(N),hC(N);
    CHK(cudaMemcpy(hP.data(),Cp,(size_t)N*2,cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hC.data(),Cc,(size_t)N*2,cudaMemcpyDeviceToHost));
    int mm=0; for(unsigned i=0;i<N;i++){ uint16_t a=*(uint16_t*)&hP[i],b=*(uint16_t*)&hC[i]; if(a!=b)mm++; }
    printf("  mismatches = %d / %u  (%s)\n",mm,N,mm==0?"BIT-EXACT":"DIFFERS");
    cudaFree(W);cudaFree(x);cudaFree(Cp);cudaFree(Cc);
  }

  // ---- depth sensitivity at aggregate-best occupancy (D=4,6,8) on qkv+moe ----
  printf("\n== depth sensitivity (D=4,6,8) on qkv (big-N) at occ %d ==\n",bestm);
  { unsigned N=8192,K=2816; size_t wn=(size_t)N*K; double gb=(double)wn*2/1e9;
    __nv_bfloat16*W=rnd_dev(wn,N+K), *x=rnd_dev(K,7u), *C; CHK(cudaMalloc(&C,(size_t)N*2));
    unsigned nblk=(unsigned)nSM*bestm;
    for(int Dd : {4,6,8}){
      double us=time_median([&]{
        size_t sm=(size_t)K*2+(size_t)Dd*WPB*GV_STEP*2;
        if(Dd==4) gemv_cp<4><<<nblk,BLOCK,sm>>>(C,x,W,N,K,nblk);
        else if(Dd==6) gemv_cp<6><<<nblk,BLOCK,sm>>>(C,x,W,N,K,nblk);
        else gemv_cp<8><<<nblk,BLOCK,sm>>>(C,x,W,N,K,nblk);
      },IT);
      double bw=gb/(us/1e6); printf("  D%d: %6.1f us %2.0f%% HBM\n",Dd,us,100*bw/HBM_SPEC);
    }
    cudaFree(W);cudaFree(x);cudaFree(C);
  }

  printf("\nDONE\n");
  return 0;
}
