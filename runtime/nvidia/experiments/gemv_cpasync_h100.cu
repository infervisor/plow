// gemv_cpasync_h100.cu — SINGLE-OP decode-GEMV micro-experiment on H100 NVL (sm_90a).
//
// GOAL: beat the production plow M=1 decode GEMV (op_gemm.cuh d_gemv arena path) by
// hiding HBM3 load latency with a cp.async software pipeline on the weight stream.
// The IDENTICAL bf16 N-split GEMV runs ~96% of peak on RTX-Blackwell but only ~21% of
// peak on H100 (perf-data/live-decode-trace-26b-h100.md): H100's bandwidth-latency
// product needs far more in-flight loads than the production 8-warp/SM blocking
// ld_glob8 + GV_UNROLL prefetch can keep outstanding. c1r (perf-data/c1r-decode-
// occupancy.md, H2b) proved cp.async-staging the weight stream to smem (all warps FMA,
// NO warp specialization, NO occupancy change) saturates HBM and wins 1.25-1.30x on RTX
// for K<=4096. This probe A/Bs that on H100 at the real megakernel geometry.
//
//   (A) BASELINE  = production d_gemv inner loop VERBATIM: blocking ld_glob8 with
//                   GV_UNROLL=8 manual prefetch, x staged in smem, dot8 from smem.
//   (B) CP.ASYNC  = SAME N-split column ownership + SAME dot8/warp_sum32 epilogue and
//                   SAME chunk accumulation order (=> BYTE-IDENTICAL result), but each
//                   warp cp.async-stages its weight row's 256-elem K-chunks global->smem
//                   in a depth-{2,3,4} software pipeline and FMAs from smem.
//
// Geometry = the real megakernel: GRID=nSM(132), blockDim=256 (8 warps), 1 block/SM,
// slice=blockIdx.x, nblk=gridDim.x, weights HBM-resident, x smem-staged, M=1 bf16.
// Weights read COLD: a 192 MB L2-flush memset between every timed rep. Median of >=100.
//
// The three defects in the prior probe gemv_transport.cu (campaign lever #1) and how (B)
// avoids them:
//   D1. transport's cp.async kernel (gemv_bf16_cpasync) is NOT bit-exact vs production:
//       it accumulates x as fp32 (`float* xs`), stages a ROWS x SK block shared across a
//       warp, and its column walk (k=lane*8;k<SK;k+=WARP*8) sums in a DIFFERENT order
//       than gemv_rows' per-lane contiguous dot8 — the result is only relL2-close, never
//       byte-identical, so it can't be dropped into d_gemv. (B) keeps the EXACT lane
//       ownership (lane l owns [l*8,l*8+8) of every 256-chunk) and the EXACT dot8 order,
//       verified with a 0-mismatch memcmp against baseline.
//   D2. transport hard-wired K=2560 (Qwen3-4B) and SK=256 with `nst=K/SK` — a partial
//       last chunk (26B has K=704,2112 -> not multiples of 256) would silently drop the
//       tail or over-read. (B) carries the production `k < K` predicate on both the
//       cp.async issue (src_bytes=0 past the row end, hw zero-fills, reads no gmem) and
//       the consume (skip the dot, exactly like baseline's `if(kk>=K) continue`).
//   D3. transport used a SINGLE cp.async.commit + wait_group 0 PER STAGE (fully blocking,
//       no overlap) — it measured a transport, not a pipeline. (B) keeps a constant depth
//       of D commit-groups in flight (wait_group<D-1>), issuing chunk c+D while consuming
//       chunk c, so the next HBM chunk lands while the current one is being FMA'd.
//   (transport ALSO measured L2 not HBM for buffers < L2; this probe L2-flushes instead.)
//
// Build: env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
//          nvcc -arch=sm_90a -O3 --use_fast_math -Xptxas -v \
//          -I runtime/nvidia -I runtime/common -o /tmp/gcah gemv_cpasync_h100.cu
// Run:   LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease gcah /tmp/gcah

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

static const double HBM_SPEC = 3350.0;   // H100 NVL HBM3 achievable GB/s (perf-data)
#define BLOCK    256u
#define WPB      (BLOCK/32u)              // 8 warps/block = PLOW_NV_WARPS
#define GV_STEP  256u                     // one warp pass over K: 32 lanes * 8 bf16
#define GV_UNROLL 8                       // production d_gemv manual prefetch depth

// ---- production primitives, byte-identical to op_attention.cuh / sm120_common.cuh ----
struct bf16v8 { __nv_bfloat16 x[8]; };
__device__ __forceinline__ bf16v8 ld_glob8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ bf16v8 ld_smem8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ bf16v8 bf16v8_zero(){ bf16v8 r; *(uint4*)&r=make_uint4(0,0,0,0); return r; }
__device__ __forceinline__ float dot8(const bf16v8&a,const bf16v8&b,float acc){
  #pragma unroll
  for(int i=0;i<8;i++) acc=fmaf(__bfloat162float(a.x[i]),__bfloat162float(b.x[i]),acc);
  return acc;
}
__device__ __forceinline__ float warp_sum32(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o,32);
  return v;
}
// cp.async primitives, byte-identical to op_gemm.cuh:548-559 (pgm_cp_async_cg16 et al.)
__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem, int src_bytes){
  unsigned s=(unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(s),"l"(gmem),"r"(src_bytes));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"::); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

// ============================ (A) BASELINE: production d_gemv ============================
// Verbatim op_gemm.cuh:210-246 M=1 arena inner loop. x staged in smem, weights streamed
// from HBM with GV_UNROLL-deep blocking ld_glob8 prefetch.
__global__ __launch_bounds__(BLOCK) void gemv_base(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 xs[];                       // K bf16
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  for(unsigned n=n0+warp; n<n1; n+=WPB){
    const __nv_bfloat16* wrow=W+(size_t)n*K;
    float acc=0.f;
    for(unsigned c=0;c<nchunk;c+=GV_UNROLL){
      bf16v8 wv[GV_UNROLL]; unsigned kk[GV_UNROLL];
      #pragma unroll
      for(int u=0;u<GV_UNROLL;u++){ unsigned k=(c+(unsigned)u)*GV_STEP+lane*8u; kk[u]=k;
        wv[u]=(k<K)?ld_glob8(wrow+k):bf16v8_zero(); }
      #pragma unroll
      for(int u=0;u<GV_UNROLL;u++){ if(kk[u]>=K) continue;
        acc=dot8(wv[u], ld_smem8(xs+kk[u]), acc); }
    }
    const float t=warp_sum32(acc);
    if(lane==0) C[n]=__float2bfloat16(t);
  }
}

// ========================= (B) CP.ASYNC software-pipeline GEMV =========================
// Same N-split ownership (one warp/row), same per-lane 256-chunk dot8 order, same epilogue
// => BYTE-IDENTICAL to (A). Only the weight bytes' transport differs: each warp cp.async-
// stages its row's chunks into a depth-D smem ring and FMAs from smem. A constant D commit
// groups stay in flight (wait_group<D-1>); chunk c+D is issued while chunk c is consumed.
// Ring layout: ring[stage][warp][256 bf16]; lane l reads/writes its own [l*8,l*8+8) slot,
// so there is NO cross-lane smem dependency and wait_group alone orders it (no __syncwarp).
template<int D>
__global__ __launch_bounds__(BLOCK) void gemv_cpasync(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[];                       // xs[K] then ring[D*WPB*256]
  __nv_bfloat16* xs   = sm;
  __nv_bfloat16* ring = sm + K;
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
  const unsigned wbase = warp*GV_STEP;                        // this warp's slot in a stage
  const unsigned sstride = WPB*GV_STEP;                       // one stage = 8 warps * 256

  for(unsigned n=n0+warp; n<n1; n+=WPB){
    const __nv_bfloat16* wrow=W+(size_t)n*K;
    // prime D commit groups (chunks 0..D-1; past-end issues an empty group => instantly done)
    #pragma unroll
    for(int p=0;p<D;p++){
      unsigned c=(unsigned)p, k=c*GV_STEP+lane*8u;
      __nv_bfloat16* dst=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
      cp_async_cg16(dst, wrow+k, (c<nchunk && k<K)?16:0);
      cp_commit();
    }
    float acc=0.f;
    for(unsigned c=0;c<nchunk;c++){
      cp_wait<D-1>();                                         // chunk c now resident
      unsigned k=c*GV_STEP+lane*8u;
      const __nv_bfloat16* src=ring+(c%(unsigned)D)*sstride+wbase+lane*8u;
      if(k<K) acc=dot8(ld_smem8(src), ld_smem8(xs+k), acc);
      // keep the pipeline full: issue chunk c+D (empty group past the end)
      unsigned cn=c+(unsigned)D, kn=cn*GV_STEP+lane*8u;
      __nv_bfloat16* dst=ring+(cn%(unsigned)D)*sstride+wbase+lane*8u;
      cp_async_cg16(dst, wrow+kn, (cn<nchunk && kn<K)?16:0);
      cp_commit();
    }
    cp_wait<0>();                                             // drain before the ring is reused
    const float t=warp_sum32(acc);
    if(lane==0) C[n]=__float2bfloat16(t);
  }
}

// ---------------------------------------------------------------- harness
static __nv_bfloat16* rnd_dev(size_t n, unsigned seed){
  std::vector<__nv_bfloat16> h(n);
  for(size_t i=0;i<n;i++){ unsigned r=(unsigned)(i*1103515245u+seed*2654435761u+12345u);
    h[i]=__float2bfloat16(((r>>8)&255)/255.f-0.5f); }
  __nv_bfloat16* d; CHK(cudaMalloc(&d,n*2)); CHK(cudaMemcpy(d,h.data(),n*2,cudaMemcpyHostToDevice));
  return d;
}
static char* g_flush=nullptr; static const size_t FLUSH=192ull<<20;   // >> H100 60 MB L2
static void flushL2(){ CHK(cudaMemset(g_flush,0,FLUSH)); }

// median us of >=IT cold-weight reps of `launch`
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

struct Shape{ const char* name; unsigned N,K; };

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); int nSM=p.multiProcessorCount;
  CHK(cudaMalloc(&g_flush,FLUSH));
  printf("# device %s  SMs=%d  L2=%.0f MB  HBM_spec=%.0f GB/s\n",p.name,nSM,p.l2CacheSize/1e6,HBM_SPEC);
  printf("# geometry: GRID=%d (1 blk/SM)  blockDim=%u (8 warps)  M=1 bf16  weights COLD (192MB L2 flush/rep)\n\n",nSM,BLOCK);

  const unsigned nblk=(unsigned)nSM;
  const int IT=120;

  Shape sh[]={
    {"moe_gate_up  op71 (N1408,K2816)",1408,2816},  // biggest decode body (~37%), per-expert
    {"moe_down     op63 (N2816,K704) ",2816,704},
    {"qkv          op22 (N8192,K2816)",8192,2816},
    {"o_proj       op10 (N2816,K4096)",2816,4096},
    {"dense_gate_up op19(N4224,K2816)",4224,2816},
    {"dense_down       (N2816,K2112)",2816,2112},
    {"lm_head control  (N262144,K2816)",262144,2816},
  };

  // occupancy sanity for both kernels
  int occB=0, occC2=0;
  CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occB,(void*)gemv_base,BLOCK,4096*2));
  CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occC2,(void*)gemv_cpasync<2>,BLOCK,4096*2+2*(int)(WPB*GV_STEP)*2));
  printf("# occupancy @smem-of-K=4096: baseline=%d blk/SM  cpasync(D=2)=%d blk/SM (target 1 = megakernel)\n\n",occB,occC2);

  printf("%-34s | %-18s | %s\n","shape","baseline","cp.async depth {2,3,4}  (GB/s, %HBM, speedup, mismatch)");
  for(auto& s : sh){
    size_t wn=(size_t)s.N*s.K; double bytes=(double)wn*2;
    __nv_bfloat16 *W=rnd_dev(wn, s.N+s.K), *x=rnd_dev(s.K,7u), *Cb, *Cc;
    CHK(cudaMalloc(&Cb,(size_t)s.N*2)); CHK(cudaMalloc(&Cc,(size_t)s.N*2));
    const size_t xsB=(size_t)s.K*2;

    // baseline
    double us_b=time_median([&]{ gemv_base<<<nblk,BLOCK,xsB>>>(Cb,x,W,s.N,s.K,nblk); }, IT);
    double gb_b=bytes/1e9/(us_b/1e6);
    std::vector<__nv_bfloat16> hCb((size_t)s.N); CHK(cudaMemcpy(hCb.data(),Cb,(size_t)s.N*2,cudaMemcpyDeviceToHost));

    printf("%-34s | %6.1fus %6.0f %2.0f%% |", s.name, us_b, gb_b, 100*gb_b/HBM_SPEC);

    // cp.async depth sweep (D<=4 ties/loses to baseline's own 8-deep GV_UNROLL prefetch;
    // extend to 6,8 for a fair match to that prefetch depth)
    for(int D : {2,3,4,6,8}){
      size_t smem=xsB + (size_t)D*WPB*GV_STEP*2;
      auto launch=[&]{
        if(D==2) gemv_cpasync<2><<<nblk,BLOCK,smem>>>(Cc,x,W,s.N,s.K,nblk);
        else if(D==3) gemv_cpasync<3><<<nblk,BLOCK,smem>>>(Cc,x,W,s.N,s.K,nblk);
        else if(D==4) gemv_cpasync<4><<<nblk,BLOCK,smem>>>(Cc,x,W,s.N,s.K,nblk);
        else if(D==6) gemv_cpasync<6><<<nblk,BLOCK,smem>>>(Cc,x,W,s.N,s.K,nblk);
        else gemv_cpasync<8><<<nblk,BLOCK,smem>>>(Cc,x,W,s.N,s.K,nblk);
      };
      double us_c=time_median(launch, IT);
      double gb_c=bytes/1e9/(us_c/1e6);
      // bit-exact check vs baseline
      launch(); CHK(cudaDeviceSynchronize());
      std::vector<__nv_bfloat16> hCc((size_t)s.N); CHK(cudaMemcpy(hCc.data(),Cc,(size_t)s.N*2,cudaMemcpyDeviceToHost));
      int mism=0; for(size_t i=0;i<(size_t)s.N;i++){
        uint16_t a=*(uint16_t*)&hCb[i], b=*(uint16_t*)&hCc[i]; if(a!=b) mism++; }
      printf(" D%d %6.0f %2.0f%% %.2fx mm=%d |",D,gb_c,100*gb_c/HBM_SPEC,gb_b>0?gb_c/gb_b:0.0,mism);
    }
    printf("\n");
    cudaFree(W); cudaFree(x); cudaFree(Cb); cudaFree(Cc);
  }
  printf("\nDONE\n");
  return 0;
}
