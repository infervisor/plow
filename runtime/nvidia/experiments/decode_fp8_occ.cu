// decode_fp8_occ.cu — fp8 (w8a16 e4m3) DECODE occupancy sweep for Gemma-4-26B-A4B, sm_90a.
//
// Fork of decode_seg_cpasync.cu (the bf16 cp.async×occupancy winner) with the WEIGHT switched
// to e4m3 (1 byte/elt, HALF the bf16 bytes) and the production dequant-on-load dot8_fp8 kernel
// from op_gemm.cuh (gemv_rows_fp8 / d_gemv_fp8). Same N-split column ownership (per=ceil(N/nblk),
// one warp/row), same 256-K warp-chunk (8 fp8/lane = a uint2), x staged in smem, weights COLD via
// 192 MB L2 flush, median 120 reps, blocks/SM ∈ {1..6}. GB/s = N*K*1byte / time.
//
// HYPOTHESIS: fp8 halves the per-layer GEMV weight bytes (200MB bf16 → ~100MB fp8), so at the SAME
// achieved GB/s the memory-bound decode should ~halve. Counter-forces: (a) at half the bytes each
// shape's working set halves, pushing MORE shapes into the HBM cold-ramp region (the 12us residual
// that held bf16 at 1.07x) so fp8 %HBM should be LOWER; (b) dequant adds registers/ALU, may erode
// occupancy. Does the fp8 GEMV-family aggregate drop below vLLM fp8's per-op target?
//
// Build: env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
//   nvcc -arch=sm_90a -O3 -Xptxas -v -I runtime/common -I runtime/nvidia \
//   -o /tmp/dfp8 runtime/nvidia/experiments/decode_fp8_occ.cu
// Run:   LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dfp8 /tmp/dfp8

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const double HBM_SPEC = 3350.0;   // H100 NVL HBM3 achievable GB/s
#define BLOCK    256u
#define WPB      (BLOCK/32u)              // 8 warps/block
#define GV_STEP  256u                     // one warp pass over K: 32 lanes * 8 elems
#define MAXOCC   6

// ---- production primitives (byte-identical to op_gemm.cuh) ----
struct bf16v8 { __nv_bfloat16 x[8]; };
__device__ __forceinline__ bf16v8 ld_glob8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ bf16v8 ld_smem8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }

// dot8_fp8: dequant 8 e4m3 weights (a uint2, 8 bytes) and fma vs 8 bf16 x. BYTE-IDENTICAL to
// op_gemm.cuh:dot8_fp8 (same __nv_cvt_fp8x2_to_halfraw2 order, same fmaf accumulation order).
__device__ __forceinline__ float dot8_fp8(const uint2& w8, const bf16v8& x, float acc){
  const uint16_t* wp=(const uint16_t*)&w8;
  #pragma unroll
  for(int j=0;j<4;j++){
    __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
    float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
    acc=fmaf(f.x, __bfloat162float(x.x[2*j]),   acc);
    acc=fmaf(f.y, __bfloat162float(x.x[2*j+1]), acc);
  }
  return acc;
}
__device__ __forceinline__ float warp_sum32(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o,32);
  return v;
}
// fp8 chunk = 8 bytes/lane → cp.async.ca (the .cg qualifier is 16-byte-only).
__device__ __forceinline__ void cp_async_ca8(void* smem, const void* gmem, int src_bytes){
  unsigned s=(unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"::"r"(s),"l"(gmem),"r"(src_bytes));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"::); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

// ---------- PRODUCTION gemv_rows_fp8 (W from GLOBAL) — bit-exact reference ----------
// mirrors op_gemm.cuh:gemv_rows_fp8<1> (the B=1 serving body): uint2 per lane, dot8_fp8, scale[n].
__global__ __launch_bounds__(BLOCK) void gemv_rows_fp8_ref(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W, const float* __restrict__ scale,
    unsigned N, unsigned K, unsigned nblk){
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  for(unsigned n=n0+warp; n<n1; n+=WPB){
    const uint8_t* wr=W+(size_t)n*K; float acc=0.f;
    for(unsigned c=0;c<nchunk;c++){ unsigned k=c*GV_STEP+lane*8u;
      if(k<K){ uint2 wv=*(const uint2*)(wr+k); acc=dot8_fp8(wv, ld_glob8(x+k), acc);} }
    acc=warp_sum32(acc); if(lane==0) C[n]=__float2bfloat16(acc*scale[n]);
  }
}

// ---------- cp.async fp8 row-staged worker (SAME lane ownership + dot8_fp8 order → bit-exact) ----
// each warp cp.async-stages its row's 256-fp8 K-chunks (8 bytes/lane) into a depth-D smem ring and
// dequant-FMAs from smem; D commit-groups stay in flight. ring slot per warp = warp*GV_STEP bytes.
template<int D>
__device__ __forceinline__ float cprow_fp8(const uint8_t* wrow, const __nv_bfloat16* xs,
                                           unsigned K, uint8_t* ring, unsigned lane, unsigned warp){
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned wbase   = warp*GV_STEP;          // bytes
  const unsigned sstride = WPB*GV_STEP;           // one stage = 8 warps * 256 bytes
  #pragma unroll
  for(int p=0;p<D;p++){
    unsigned c=(unsigned)p, k=c*GV_STEP+lane*8u;
    uint8_t* dst=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
    cp_async_ca8(dst, wrow+k, (c<nchunk && k<K)?8:0); cp_commit();
  }
  float acc=0.f;
  for(unsigned c=0;c<nchunk;c++){
    cp_wait<D-1>();
    unsigned k=c*GV_STEP+lane*8u;
    const uint8_t* src=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
    if(k<K){ uint2 wv=*(const uint2*)src; acc=dot8_fp8(wv, ld_smem8(xs+k), acc); }
    unsigned cn=c+(unsigned)D, kn=cn*GV_STEP+lane*8u;
    uint8_t* dst=ring+(cn%(unsigned)D)*sstride+wbase+lane*8u;
    cp_async_ca8(dst, wrow+kn, (cn<nchunk && kn<K)?8:0); cp_commit();
  }
  cp_wait<0>();
  return warp_sum32(acc);
}

// ---------- dense cp.async fp8 GEMV (N-split column ownership, x smem-staged) ----------
template<int D>
__global__ __launch_bounds__(BLOCK) void gemv_cp_fp8(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W, const float* __restrict__ scale,
    unsigned N, unsigned K, unsigned nblk){
  extern __shared__ uint8_t smb[];                    // xs[K] bf16 (2K bytes) then ring[D*WPB*256]
  __nv_bfloat16* xs=(__nv_bfloat16*)smb; uint8_t* ring=smb+(size_t)K*2;
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  for(unsigned n=n0+warp; n<n1; n+=WPB){
    float t=cprow_fp8<D>(W+(size_t)n*K, xs, K, ring, lane, warp);
    if(lane==0) C[n]=__float2bfloat16(t*scale[n]);
  }
}

// ---------- FLAT grouped MoE gate_up (cp.async fp8), operand x[K] smem-staged once ----------
template<int D>
__global__ __launch_bounds__(BLOCK) void moe_gateup_cp_fp8(
    __nv_bfloat16* __restrict__ FU, const __nv_bfloat16* __restrict__ x,
    const uint8_t* const* __restrict__ Wgu, const float* const* __restrict__ Sgu,
    unsigned E, unsigned Ngu, unsigned K, unsigned nblk){
  extern __shared__ uint8_t smb[]; __nv_bfloat16* xs=(__nv_bfloat16*)smb; uint8_t* ring=smb+(size_t)K*2;
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned rows=E*Ngu; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/Ngu, n=r%Ngu;
    float t=cprow_fp8<D>(Wgu[e]+(size_t)n*K, xs, K, ring, lane, warp);
    if(lane==0) FU[(size_t)e*Ngu+n]=__float2bfloat16(t*Sgu[e][n]);
  }
}
// ---------- FLAT grouped MoE down (cp.async fp8), operand FU per-expert (Kmi each) staged ----------
template<int D>
__global__ __launch_bounds__(BLOCK) void moe_down_cp_fp8(
    __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ FU,
    const uint8_t* const* __restrict__ Wd, const float* const* __restrict__ Sd,
    unsigned E, unsigned H, unsigned Kmi, unsigned nblk){
  extern __shared__ uint8_t smb[]; __nv_bfloat16* xs=(__nv_bfloat16*)smb; uint8_t* ring=smb+(size_t)E*Kmi*2;
  for(unsigned i=threadIdx.x;i<E*Kmi;i+=blockDim.x) xs[i]=FU[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned rows=E*H; const unsigned per=(rows+nblk-1)/nblk;
  const unsigned r0=blockIdx.x*per, r1=(r0+per<rows)?(r0+per):rows;
  for(unsigned r=r0+warp; r<r1; r+=WPB){
    unsigned e=r/H, n=r%H;
    float t=cprow_fp8<D>(Wd[e]+(size_t)n*Kmi, xs+(size_t)e*Kmi, Kmi, ring, lane, warp);
    if(lane==0) Y[(size_t)e*H+n]=__float2bfloat16(t*Sd[e][n]);
  }
}

// ---------------------------------------------------------------- harness
static __nv_bfloat16* rnd_bf16(size_t n, unsigned seed){
  std::vector<__nv_bfloat16> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)((i*1103515245u+seed*2654435761u+12345u));
    h[i]=__float2bfloat16(((r>>8)&255)/255.f-0.5f); }
  __nv_bfloat16* d; CHK(cudaMalloc(&d,n*2)); CHK(cudaMemcpy(d,h.data(),n*2,cudaMemcpyHostToDevice));
  return d;
}
static uint8_t* rnd_fp8(size_t n, unsigned seed){
  std::vector<uint8_t> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)((i*1103515245u+seed*2654435761u+12345u));
    float v=((r>>8)&255)/255.f-0.5f; __nv_fp8_e4m3 q(v); h[i]=*(uint8_t*)&q; }
  uint8_t* d; CHK(cudaMalloc(&d,n)); CHK(cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice));
  return d;
}
static float* rnd_scale(size_t n, unsigned seed){
  std::vector<float> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)((i*2654435761u+seed*40503u+7u)); h[i]=0.5f+((r>>8)&255)/512.f; }
  float* d; CHK(cudaMalloc(&d,n*4)); CHK(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice));
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

// dense smem for a given K and depth D: xs[K] bf16 (2K bytes) + ring D*WPB*256 fp8 bytes
static size_t dsmem_fp8(unsigned K,int D){ return (size_t)K*2 + (size_t)D*WPB*GV_STEP; }

template<int D>
static void launch_dense(__nv_bfloat16*C,__nv_bfloat16*x,uint8_t*W,float*S,unsigned N,unsigned K,unsigned nblk){
  gemv_cp_fp8<D><<<nblk,BLOCK,dsmem_fp8(K,D)>>>(C,x,W,S,N,K,nblk);
}

struct Res{ std::string name; double bytes; double us[MAXOCC+1]; double bw[MAXOCC+1]; };

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); int nSM=p.multiProcessorCount;
  CHK(cudaMalloc(&g_flush,FLUSH));
  printf("# device %s  SMs=%d  L2=%.0f MB  smemPerSM=%.0f KB  HBM_spec=%.0f GB/s\n",
         p.name,nSM,p.l2CacheSize/1e6,p.sharedMemPerMultiprocessor/1024.0,HBM_SPEC);
  printf("# fp8 (e4m3 w8a16) cp.async GEMV depth=6, x staged, W COLD (192MB L2 flush/rep), median 120 reps\n");
  printf("# GB/s = N*K*1byte/time ; occupancy = grid/nSM = mult blocks/SM (N-split per=ceil(N/nblk))\n\n");

  const int D=6, IT=120;

  for(unsigned K : {2816u,4096u,2112u}){
    int occ=0; CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)gemv_cp_fp8<6>,BLOCK,dsmem_fp8(K,D)));
    printf("# cudaOccupancyMaxActiveBlocksPerMultiprocessor gemv_cp_fp8<6> K=%u smem=%.1fKB : %d blocks/SM\n",
           K,dsmem_fp8(K,D)/1024.0,occ);
  }
  printf("\n");

  const unsigned H=2816;
  struct Shape{const char*name;unsigned N,K;} sh[]={
    {"qkv_proj  (N8192,K2816)",8192,2816},
    {"o_proj    (N2816,K4096)",2816,4096},
    {"gate_up   (N4224,K2816)",4224,2816},
    {"down_proj (N2816,K2112)",2816,2112},
  };

  std::vector<Res> R;
  Res lmR; bool haveLM=false;

  printf("== DENSE fp8 decode GEMV (M=1, cp.async D=6, W cold), us / GB/s / %%HBM at 1..6 blocks/SM ==\n");
  for(auto&s:sh){
    size_t wn=(size_t)s.N*s.K; double gb=(double)wn/1e9;   // 1 byte/weight
    uint8_t*W=rnd_fp8(wn,s.N+s.K); __nv_bfloat16*x=rnd_bf16(s.K,7u),*C; float*S=rnd_scale(s.N,s.N);
    CHK(cudaMalloc(&C,(size_t)s.N*2));
    Res r; r.name=s.name; r.bytes=(double)wn;
    printf("%-26s %6.1fMB |",s.name,gb*1e3);
    for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ launch_dense<6>(C,x,W,S,s.N,s.K,nblk); },IT);
      double bw=gb/(us/1e6); r.us[m]=us; r.bw[m]=bw;
      printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
    printf("\n"); R.push_back(r);
    cudaFree(W);cudaFree(x);cudaFree(C);cudaFree(S);
  }

  // MoE-experts: 8 experts, gate_up N=1408 K=2816, down N=2816(->H rows) K=704.
  printf("\n== MoE-experts grouped fp8 GEMV (8 experts, flat cp.async D=6, W cold), 1..6 blocks/SM ==\n");
  const unsigned E=8, Ngu=1408, Kmi=704;
  std::vector<uint8_t*> hWgu(E),hWd(E); std::vector<float*> hSgu(E),hSd(E);
  for(unsigned e=0;e<E;e++){ hWgu[e]=rnd_fp8((size_t)Ngu*H,100+e); hWd[e]=rnd_fp8((size_t)H*Kmi,200+e);
    hSgu[e]=rnd_scale(Ngu,300+e); hSd[e]=rnd_scale(H,400+e);}
  const uint8_t **dWgu,**dWd; const float **dSgu,**dSd;
  CHK(cudaMalloc(&dWgu,E*sizeof(void*))); CHK(cudaMalloc(&dWd,E*sizeof(void*)));
  CHK(cudaMalloc(&dSgu,E*sizeof(void*))); CHK(cudaMalloc(&dSd,E*sizeof(void*)));
  CHK(cudaMemcpy(dWgu,hWgu.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dWd,hWd.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dSgu,hSgu.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dSd,hSd.data(),E*sizeof(void*),cudaMemcpyHostToDevice));
  __nv_bfloat16 *xh=rnd_bf16(H,9u), *FU, *Yout; CHK(cudaMalloc(&FU,(size_t)E*Ngu*2)); CHK(cudaMalloc(&Yout,(size_t)E*H*2));
  double gu_gb=(double)E*Ngu*H/1e9, dn_gb=(double)E*H*Kmi/1e9, moe_gb=gu_gb+dn_gb;   // 1 byte/weight
  size_t gu_sm=dsmem_fp8(H,D), dn_sm=(size_t)E*Kmi*2+(size_t)D*WPB*GV_STEP;
  printf("gate_up=%.1fMB down=%.1fMB total=%.1fMB (gu_smem=%.1fKB dn_smem=%.1fKB)\n",
         gu_gb*1e3,dn_gb*1e3,moe_gb*1e3,gu_sm/1024.0,dn_sm/1024.0);
  Res moeR; moeR.name="moe_experts(8)"; moeR.bytes=moe_gb*1e9;
  printf("%-26s %6.1fMB |","moe_experts(gu+down)",moe_gb*1e3);
  for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
    double us=time_median([&]{
      moe_gateup_cp_fp8<6><<<nblk,BLOCK,gu_sm>>>(FU,xh,dWgu,dSgu,E,Ngu,H,nblk);
      moe_down_cp_fp8<6><<<nblk,BLOCK,dn_sm>>>(Yout,FU,dWd,dSd,E,H,Kmi,nblk);
    },IT);
    double bw=moe_gb/(us/1e6); moeR.us[m]=us; moeR.bw[m]=bw;
    printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
  printf("\n"); R.push_back(moeR);

  // lm_head control (once-per-token) — big N
  printf("\n== lm_head control fp8 (N262144,K2816, once-per-token), cp.async D=6, 1..6 blocks/SM ==\n");
  { unsigned N=262144, K=2816; size_t wn=(size_t)N*K; double gb=(double)wn/1e9;
    uint8_t*W=rnd_fp8(wn,123u); __nv_bfloat16*x=rnd_bf16(K,7u),*C; float*S=rnd_scale(N,999);
    CHK(cudaMalloc(&C,(size_t)N*2));
    lmR.name="lm_head(N262144)"; lmR.bytes=(double)wn; haveLM=true;
    printf("%-26s %6.0fMB |","lm_head",gb*1e3);
    for(int m=1;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ launch_dense<6>(C,x,W,S,N,K,nblk); },IT);
      double bw=gb/(us/1e6); lmR.us[m]=us; lmR.bw[m]=bw;
      printf(" [%d]%6.1f %2.0f%%",m,us,100*bw/HBM_SPEC);}
    printf("\n"); cudaFree(W);cudaFree(x);cudaFree(C);cudaFree(S);
  }

  // ---- GEMV-family aggregate (per-layer shapes: qkv+o+gate_up+down+moe = 100MB) ----
  printf("\n== fp8 GEMV-family SEQUENCE aggregate (qkv+o+gate_up+down+moe_experts) ==\n");
  double best=1e30; int bestm=0;
  for(int m=1;m<=MAXOCC;m++){ double tot=0,byt=0; for(auto&r:R){ tot+=r.us[m]; byt+=r.bytes; }
    double bw=byt/1e9/(tot/1e6);
    printf("  %d blk/SM: AGG=%7.1f us  %5.0f GB/s (%2.0f%% HBM)  totW=%.0fMB\n",m,tot,bw,100*bw/HBM_SPEC,byt/1e6);
    if(tot<best){best=tot;bestm=m;} }
  printf("  -> aggregate MIN = %.1f us at %d blocks/SM\n",best,bestm);
  if(haveLM){ double b=1e30;int bm=0;for(int m=1;m<=MAXOCC;m++)if(lmR.us[m]<b){b=lmR.us[m];bm=m;}
    printf("  lm_head (separate, once/token) best = %.1f us at %d blk/SM\n",b,bm);}

  printf("\n== per-shape OPTIMUM occupancy ==\n");
  auto report_opt=[&](Res&r){ double b=1e30;int bm=0;for(int m=1;m<=MAXOCC;m++)if(r.us[m]<b){b=r.us[m];bm=m;}
    printf("  %-24s peak %6.1f us (%2.0f%% HBM) at %d blk/SM\n",r.name.c_str(),b,100*r.bw[bm]/HBM_SPEC,bm); };
  for(auto&r:R) report_opt(r);
  if(haveLM) report_opt(lmR);

  // ---- bit-exact vs production gemv_rows_fp8 (qkv shape, occ-3) ----
  printf("\n== bit-exact: gemv_cp_fp8<6> vs production gemv_rows_fp8 (qkv N8192 K2816) ==\n");
  { unsigned N=8192,K=2816; size_t wn=(size_t)N*K;
    uint8_t*W=rnd_fp8(wn,N+K); __nv_bfloat16*x=rnd_bf16(K,7u),*Cp,*Cc; float*S=rnd_scale(N,N);
    CHK(cudaMalloc(&Cp,(size_t)N*2)); CHK(cudaMalloc(&Cc,(size_t)N*2));
    unsigned nblk=(unsigned)nSM*3;
    gemv_rows_fp8_ref<<<nblk,BLOCK>>>(Cp,x,W,S,N,K,nblk);
    launch_dense<6>(Cc,x,W,S,N,K,nblk); CHK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hP(N),hC(N);
    CHK(cudaMemcpy(hP.data(),Cp,(size_t)N*2,cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hC.data(),Cc,(size_t)N*2,cudaMemcpyDeviceToHost));
    int mm=0; float maxabs=0; for(unsigned i=0;i<N;i++){ uint16_t a=*(uint16_t*)&hP[i],b=*(uint16_t*)&hC[i];
      if(a!=b){mm++; float d=fabsf(__bfloat162float(hP[i])-__bfloat162float(hC[i])); if(d>maxabs)maxabs=d;} }
    printf("  mismatches = %d / %u   max|abs delta| = %.3e   (%s)\n",mm,N,maxabs,mm==0?"BIT-EXACT":"DIFFERS");
    cudaFree(W);cudaFree(x);cudaFree(Cp);cudaFree(Cc);cudaFree(S);
  }

  // ---- depth sensitivity at aggregate-best occupancy (D=4,6,8) on qkv ----
  printf("\n== depth sensitivity (D=4,6,8) on qkv (big-N) at occ %d ==\n",bestm);
  { unsigned N=8192,K=2816; size_t wn=(size_t)N*K; double gb=(double)wn/1e9;
    uint8_t*W=rnd_fp8(wn,N+K); __nv_bfloat16*x=rnd_bf16(K,7u),*C; float*S=rnd_scale(N,N);
    CHK(cudaMalloc(&C,(size_t)N*2)); unsigned nblk=(unsigned)nSM*bestm;
    for(int Dd : {4,6,8}){
      double us=time_median([&]{
        size_t sm=(size_t)K*2+(size_t)Dd*WPB*GV_STEP;
        if(Dd==4) gemv_cp_fp8<4><<<nblk,BLOCK,sm>>>(C,x,W,S,N,K,nblk);
        else if(Dd==6) gemv_cp_fp8<6><<<nblk,BLOCK,sm>>>(C,x,W,S,N,K,nblk);
        else gemv_cp_fp8<8><<<nblk,BLOCK,sm>>>(C,x,W,S,N,K,nblk);
      },IT);
      double bw=gb/(us/1e6); printf("  D%d: %6.1f us %2.0f%% HBM\n",Dd,us,100*bw/HBM_SPEC);
    }
    cudaFree(W);cudaFree(x);cudaFree(C);cudaFree(S);
  }

  printf("\nDONE\n");
  return 0;
}
