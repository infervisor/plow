// decode_smalln_warppair.cu — closing probe for the H100 26B decode "beat vLLM" campaign.
//
// The cp.async N-split GEMV (decode_seg_cpasync.cu) floors the GEMV-family aggregate at
// 138us (43% HBM) = 1.07x vLLM. The residual is TWO small-N shapes that plateau while the
// idle warps sit doing nothing at high occupancy:
//   o_proj    (N2816,K4096)  -> 33% HBM  (per=ceil(N/nblk) collapses to ~4-6 rows/block)
//   down_proj (N2816,K2112)  -> 25% HBM
// One-warp-per-row leaves 2-4 of 8 warps idle at occ 4-6, capping in-flight loads/row.
//
// THIS PROBE tests the ONE untested lever: INTRA-block warp-pairing. Instead of 1 warp/row
// with idle warps, assign Wpr warps to ONE row, each dotting a K-slice, partials reduced
// WITHIN the block (warp-shuffle + tiny smem) — NO global atomicAdd (the cross-block
// atomic K-split is already REFUTED: down_proj -25.5%, o_proj -15.5%). More warps busy/row
// => more memory-level parallelism for the small latency-bound matrices.
//
// Sweep both shapes, weights COLD (192MB L2 flush/rep, median >=100 reps), in-context
// back-to-back — identical methodology to decode_seg_cpasync.cu:
//   baseline: 1 warp/row (gemv_cp), blocks/SM {2,3,4,5,6}  -> reproduce the 25-33% floor
//   warp-pair: Wpr in {2,4}, cp.async K-slice + shuffle/smem reduce, blocks/SM {2,3,4,5,6}
// Correctness: compare vs production gemv_rows. The Wpr-way reduce reassociates the float
// sum, so report mismatches AND max abs/rel error (within bf16 rounding or not).
//
// Build: env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
//   nvcc -arch=sm_90a -O3 -Xptxas -v -I runtime/common -I runtime/nvidia \
//   -o /tmp/dsw runtime/nvidia/experiments/decode_smalln_warppair.cu
// Run:   LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease dsw /tmp/dsw

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
#define WPB      (BLOCK/32u)              // 8 warps/block
#define GV_STEP  256u                     // one warp pass over K: 32 lanes * 8 bf16
#define MAXOCC   6

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

// ---------- BASELINE cp.async row-staged worker (1 warp/row) — matches decode_seg_cpasync -
template<int D>
__device__ __forceinline__ float cprow(const __nv_bfloat16* wrow, const __nv_bfloat16* xs,
                                        unsigned K, __nv_bfloat16* ring, unsigned lane, unsigned warp){
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned wbase   = warp*GV_STEP;
  const unsigned sstride = WPB*GV_STEP;
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
template<int D>
__global__ __launch_bounds__(BLOCK) void gemv_cp(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[];
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

// ---------- WARP-PAIR cp.async worker: this warp owns the STRIDED chunk subset ----------
// warp handles global chunks c = coff, coff+cstride, coff+2*cstride, ... (coff=sub, cstride=Wpr)
// depth-D cp.async pipeline over ITS local chunk list; returns this warp's warp_sum32 partial.
template<int D>
__device__ __forceinline__ float cprow_slice(const __nv_bfloat16* wrow, const __nv_bfloat16* xs,
    unsigned K, __nv_bfloat16* ring, unsigned lane, unsigned warp, unsigned coff, unsigned cstride){
  const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
  const unsigned wbase   = warp*GV_STEP;      // each warp keeps its own ring region
  const unsigned sstride = WPB*GV_STEP;
  #pragma unroll
  for(int p=0;p<D;p++){
    unsigned c=coff+(unsigned)p*cstride, k=c*GV_STEP+lane*8u;
    __nv_bfloat16* dst=ring+((unsigned)p)*sstride+wbase+lane*8u;   // stage = p (p<D)
    cp_async_cg16(dst, wrow+k, (c<nchunk && k<K)?16:0); cp_commit();
  }
  float acc=0.f; unsigned j=0;
  for(unsigned c=coff; c<nchunk; c+=cstride, j++){
    cp_wait<D-1>();
    unsigned k=c*GV_STEP+lane*8u;
    const __nv_bfloat16* src=ring+(j%(unsigned)D)*sstride+wbase+lane*8u;
    if(k<K) acc=dot8(ld_smem8(src), ld_smem8(xs+k), acc);
    unsigned cn=c+(unsigned)D*cstride, kn=cn*GV_STEP+lane*8u;
    __nv_bfloat16* dst=ring+((j+(unsigned)D)%(unsigned)D)*sstride+wbase+lane*8u;
    cp_async_cg16(dst, wrow+kn, (cn<nchunk && kn<K)?16:0); cp_commit();
  }
  cp_wait<0>();
  return warp_sum32(acc);
}

// ---------- WARP-PAIR GEMV: Wpr warps cooperate on one row, reduce in-block (no atomic) ---
template<int D, int Wpr>
__global__ __launch_bounds__(BLOCK) void gemv_cp_wp(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W, unsigned N, unsigned K, unsigned nblk){
  extern __shared__ __nv_bfloat16 sm[];
  __nv_bfloat16* xs=sm; __nv_bfloat16* ring=sm+K;
  __shared__ float combine[WPB];                    // one partial per warp
  for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
  const unsigned sub = warp % (unsigned)Wpr;        // which K-slice
  const unsigned grp = warp / (unsigned)Wpr;        // which row-group
  const unsigned G   = WPB / (unsigned)Wpr;         // rows covered per iteration
  const unsigned per=(N+nblk-1)/nblk;
  const unsigned n0=blockIdx.x*per; unsigned n1=n0+per; if(n1>N)n1=N;
  const unsigned iters=(per+G-1)/G;                 // uniform across all warps => syncthreads-safe
  for(unsigned it=0; it<iters; it++){
    unsigned row=n0 + it*G + grp;
    bool active = row<n1;
    float part = active ? cprow_slice<D>(W+(size_t)row*K, xs, K, ring, lane, warp, sub, (unsigned)Wpr) : 0.f;
    if(lane==0) combine[warp]=part;
    __syncthreads();
    if(active && sub==0){
      float t=0.f;
      #pragma unroll
      for(int q=0;q<Wpr;q++) t+=combine[grp*(unsigned)Wpr+(unsigned)q];   // fixed order sum
      C[row]=__float2bfloat16(t);
    }
    __syncthreads();
  }
}

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
static size_t dsmem(unsigned K,int D){ return (size_t)K*2 + (size_t)D*WPB*GV_STEP*2; }

// numeric compare vs a float reference computed the gemv_rows way
static void num_check(const char* tag, const __nv_bfloat16* Cdev, const std::vector<float>& ref, unsigned N){
  std::vector<__nv_bfloat16> h(N); CHK(cudaMemcpy(h.data(),Cdev,(size_t)N*2,cudaMemcpyDeviceToHost));
  // reference in bf16 (round the float ref) for mismatch count
  int mm=0; double maxabs=0, maxrel=0;
  for(unsigned i=0;i<N;i++){
    float got=__bfloat162float(h[i]);
    __nv_bfloat16 refb=__float2bfloat16(ref[i]); float refv=__bfloat162float(refb);
    if(*(uint16_t*)&h[i]!=*(uint16_t*)&refb) mm++;
    double a=fabs((double)got-(double)refv); maxabs=std::max(maxabs,a);
    if(fabs(refv)>1e-6) maxrel=std::max(maxrel,a/fabs((double)refv));
  }
  printf("  %-22s mismatches=%d/%u  max|abs|=%.3e  max|rel|=%.3e  %s\n",
         tag,mm,N,maxabs,maxrel, mm==0?"BIT-EXACT":"(reassoc)");
}

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); int nSM=p.multiProcessorCount;
  CHK(cudaMalloc(&g_flush,FLUSH));
  printf("# device %s  SMs=%d  L2=%.0f MB  HBM_spec=%.0f GB/s\n",p.name,nSM,p.l2CacheSize/1e6,HBM_SPEC);
  printf("# INTRA-block warp-pairing on the two small-N shapes, cp.async D=6, W COLD (192MB flush/rep), median 120 reps\n\n");

  const int D=6, IT=120;

  struct Shape{const char*name;unsigned N,K;} sh[]={
    {"o_proj    (N2816,K4096)",2816,4096},
    {"down_proj (N2816,K2112)",2816,2112},
  };

  // occupancy ceilings (same dynamic smem as baseline; combine[] is 32B static)
  for(unsigned K:{4096u,2112u}){
    int o1=0,o2=0,o4=0;
    CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o1,(void*)gemv_cp<6>,BLOCK,dsmem(K,D)));
    CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2,(void*)gemv_cp_wp<6,2>,BLOCK,dsmem(K,D)));
    CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o4,(void*)gemv_cp_wp<6,4>,BLOCK,dsmem(K,D)));
    printf("# occ ceiling K=%u smem=%.1fKB : baseline=%d  Wpr2=%d  Wpr4=%d blocks/SM\n",K,dsmem(K,D)/1024.0,o1,o2,o4);
  }
  printf("\n");

  double best_o=1e30, best_dn=1e30; std::string best_o_at, best_dn_at;

  for(auto&s:sh){
    size_t wn=(size_t)s.N*s.K; double gb=(double)wn*2/1e9;
    __nv_bfloat16*W=rnd_dev(wn,s.N+s.K), *x=rnd_dev(s.K,7u), *C; CHK(cudaMalloc(&C,(size_t)s.N*2));
    double* pbest = (s.N==2816 && s.K==4096)? &best_o : &best_dn;
    std::string* pat = (s.N==2816 && s.K==4096)? &best_o_at : &best_dn_at;

    printf("== %s  %.1fMB : us / GB/s / %%HBM at blocks/SM {2,3,4,5,6} ==\n",s.name,gb*1e3);
    // baseline
    printf("  %-10s |","baseline");
    for(int m=2;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ gemv_cp<6><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); },IT);
      double bw=gb/(us/1e6); printf(" [%d]%6.1f %4.0f %2.0f%%",m,us,bw,100*bw/HBM_SPEC);
      if(us<*pbest){*pbest=us;*pat="baseline occ-"+std::to_string(m);} }
    printf("\n");
    // Wpr=2
    printf("  %-10s |","Wpr=2");
    for(int m=2;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ gemv_cp_wp<6,2><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); },IT);
      double bw=gb/(us/1e6); printf(" [%d]%6.1f %4.0f %2.0f%%",m,us,bw,100*bw/HBM_SPEC);
      if(us<*pbest){*pbest=us;*pat="Wpr2 occ-"+std::to_string(m);} }
    printf("\n");
    // Wpr=4
    printf("  %-10s |","Wpr=4");
    for(int m=2;m<=MAXOCC;m++){ unsigned nblk=(unsigned)nSM*m;
      double us=time_median([&]{ gemv_cp_wp<6,4><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); },IT);
      double bw=gb/(us/1e6); printf(" [%d]%6.1f %4.0f %2.0f%%",m,us,bw,100*bw/HBM_SPEC);
      if(us<*pbest){*pbest=us;*pat="Wpr4 occ-"+std::to_string(m);} }
    printf("\n");

    // ---- correctness: warp-pair vs production gemv_rows (float ref computed on-device) ----
    printf("  -- correctness vs production gemv_rows (occ-4) --\n");
    unsigned nblk=(unsigned)nSM*4;
    __nv_bfloat16* Cref; CHK(cudaMalloc(&Cref,(size_t)s.N*2));
    gemv_rows<<<nblk,BLOCK>>>(Cref,x,W,s.N,s.K,nblk); CHK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hr(s.N); CHK(cudaMemcpy(hr.data(),Cref,(size_t)s.N*2,cudaMemcpyDeviceToHost));
    std::vector<float> ref(s.N); for(unsigned i=0;i<s.N;i++) ref[i]=__bfloat162float(hr[i]);
    gemv_cp<6><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); CHK(cudaDeviceSynchronize());
    num_check("baseline(1w/row)",C,ref,s.N);
    gemv_cp_wp<6,2><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); CHK(cudaDeviceSynchronize());
    num_check("Wpr=2",C,ref,s.N);
    gemv_cp_wp<6,4><<<nblk,BLOCK,dsmem(s.K,D)>>>(C,x,W,s.N,s.K,nblk); CHK(cudaDeviceSynchronize());
    num_check("Wpr=4",C,ref,s.N);
    printf("\n");
    cudaFree(Cref); cudaFree(W);cudaFree(x);cudaFree(C);
  }

  // ---- aggregate plug-in ----
  double base_agg=138.0, cur_small=20.6+14.0; // decode_seg_cpasync occ-4 aggregate & o+down contribution
  double new_small=best_o+best_dn;
  double new_agg=base_agg-cur_small+new_small;
  printf("== AGGREGATE PLUG-IN ==\n");
  printf("  best o_proj    = %.1f us (%s)\n",best_o,best_o_at.c_str());
  printf("  best down_proj = %.1f us (%s)\n",best_dn,best_dn_at.c_str());
  printf("  small-N sum: current 34.6 us -> new %.1f us\n",new_small);
  printf("  GEMV-family aggregate: 138.0 - 34.6 + %.1f = %.1f us  (beat-vLLM target < 126)\n",new_small,new_agg);
  printf("  VERDICT: %s vLLM decode (aggregate %.1f %s 126)\n",
         new_agg<126.0?"BEATS":"does NOT beat", new_agg, new_agg<126.0?"<":">=");

  printf("\nDONE\n");
  return 0;
}
