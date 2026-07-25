// gemv_dimspec.cu — dim-specific decode GEMV variants for the MoE EXPERT shapes (M=1, bf16).
//
// The MoE decode GEMVs are the launch-bound case: 8 experts, small per-op work.
//   gate/up : N=704 outputs, K=2816 (big-K, few rows)   -> Wg,Wu each [E][704][2816]
//   down    : N=2816 outputs, K=704  (small-K, many rows) -> Wd       [E][2816][704]
//
// A/B four ways of mapping warps to work, all bf16, x staged in smem, 188 blocks x 8 warps
// (the realistic decode occupancy — only MoE work live):
//   (a) flat        : one warp per output row (the current op_moe / gemv_rows mapping)
//   (b) split-K x2  : two warps per output, each dots K/2, combine via smem
//   (c) N-tile/warp : one warp owns a tile of TN consecutive outputs, x-chunk reused
//                     from smem across the tile (K-chunked staging)
//   (d) fused gu    : one warp per output, gate & up weight streams in flight together,
//                     x read ONCE for both (gate/up only)
//
// Reports weight-bandwidth (bytes = E*N*K*2) and %HBM. Register counts via ptxas -v at build.
// Build: /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -Xptxas -v -o /tmp/gd gemv_dimspec.cu

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define HBM_SPEC 1790.0
#define BLOCK 256
#define WPB   (BLOCK/32)     // 8 warps/block
#define E     8              // experts

__device__ __forceinline__ float wred(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(~0u,v,o); return v; }
__device__ __forceinline__ float dot8b(const __nv_bfloat16*w,const __nv_bfloat16*x,float a){
  #pragma unroll
  for(int j=0;j<8;j++) a=__fmaf_rn(__bfloat162float(w[j]),__bfloat162float(x[j]),a); return a; }
// vector 16B bf16 load
__device__ __forceinline__ void ldv8(__nv_bfloat16 d[8], const __nv_bfloat16* p){ *(uint4*)d=*(const uint4*)p; }

// x for gate/up is per-token K-vector shared by all experts; x for down is per-expert (E*K).
// We stage the needed x into smem. For gate/up: one x[K]. For down: expert's x_e[K].

// ---------- (a) flat one-warp-per-output ----------
// xg: [K] (gate/up, shared) OR the caller passes per-expert x for down via xstride.
__global__ __launch_bounds__(BLOCK) void gv_flat(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y, int N, int K, int xstride /*0 shared, else per-expert*/) {
  extern __shared__ __nv_bfloat16 xs[];   // [E*K] if per-expert else [K]
  int nx = xstride ? E*K : K;
  for(int i=threadIdx.x;i<nx;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int gw = blockIdx.x*WPB + (threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int total=E*N; unsigned nch=(K+255)/256;
  for(int g=gw; g<total; g+=stride){
    int e=g/N, n=g%N; const __nv_bfloat16* wr=W+((size_t)e*N+n)*K;
    const __nv_bfloat16* xe = xs + (xstride? e*K:0);
    float acc=0;
    for(unsigned c=0;c<nch;c++){ unsigned k=c*256+lane*8; if(k<(unsigned)K){ __nv_bfloat16 wv[8]; ldv8(wv,wr+k); acc=dot8b(wv,xe+k,acc);} }
    acc=wred(acc); if(lane==0) Y[(size_t)e*N+n]=__float2bfloat16(acc);
  }
}

// ---------- (b) split-K, two warps per output ----------
// block = 4 pairs. pair p (warps 2p,2p+1); half = warp&1 dots [half*K/2, +K/2).
__global__ __launch_bounds__(BLOCK) void gv_splitk(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y, int N, int K, int xstride) {
  extern __shared__ __nv_bfloat16 smem[];
  int nx = xstride ? E*K : K;
  __nv_bfloat16* xs=smem; float* part=(float*)(smem+nx);   // part[WPB/2 pairs * 2]
  for(int i=threadIdx.x;i<nx;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int warp=threadIdx.x>>5, lane=threadIdx.x&31, half=warp&1, pair=warp>>1;
  int pairsPerBlk=WPB/2, gpair=blockIdx.x*pairsPerBlk+pair, stride=gridDim.x*pairsPerBlk;
  int total=E*N; int Kh=K/2;
  for(int g=gpair; g<total; g+=stride){
    int e=g/N, n=g%N; const __nv_bfloat16* wr=W+((size_t)e*N+n)*K;
    const __nv_bfloat16* xe=xs+(xstride? e*K:0);
    float acc=0;
    for(int k=half*Kh + lane*8; k<half*Kh+Kh; k+=256){ __nv_bfloat16 wv[8]; ldv8(wv,wr+k); acc=dot8b(wv,xe+k,acc); }
    acc=wred(acc);
    if(lane==0) part[pair*2+half]=acc;
    __syncthreads();
    if(lane==0 && half==0){ float t=part[pair*2]+part[pair*2+1]; Y[(size_t)e*N+n]=__float2bfloat16(t); }
    __syncthreads();
  }
}

// ---------- (c) N-tile per warp, K-chunked x reuse ----------
#define TN 4
__global__ __launch_bounds__(BLOCK) void gv_ntile(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y, int N, int K, int xstride) {
  extern __shared__ __nv_bfloat16 xs[];
  int nx = xstride ? E*K : K;
  for(int i=threadIdx.x;i<nx;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int ntiles=(N+TN-1)/TN, total=E*ntiles; unsigned nch=(K+255)/256;
  for(int g=gw; g<total; g+=stride){
    int e=g/ntiles, t=g%ntiles, n0=t*TN;
    const __nv_bfloat16* xe=xs+(xstride? e*K:0);
    float acc[TN]; for(int i=0;i<TN;i++) acc[i]=0;
    for(unsigned c=0;c<nch;c++){ unsigned k=c*256+lane*8; if(k>=(unsigned)K) continue;
      __nv_bfloat16 xv[8]; ldv8(xv,xe+k);
      #pragma unroll
      for(int i=0;i<TN;i++){ int n=n0+i; if(n<N){ __nv_bfloat16 wv[8]; ldv8(wv,W+((size_t)e*N+n)*K+k); acc[i]=dot8b(wv,xv,acc[i]); } }
    }
    #pragma unroll
    for(int i=0;i<TN;i++){ float r=wred(acc[i]); int n=n0+i; if(lane==0 && n<N) Y[(size_t)e*N+n]=__float2bfloat16(r); }
  }
}

// ---------- (d) fused gate+up, one warp per output, x read once ----------
__global__ __launch_bounds__(BLOCK) void gv_fused_gu(
    const __nv_bfloat16* __restrict__ Wg, const __nv_bfloat16* __restrict__ Wu,
    const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Yg, __nv_bfloat16* __restrict__ Yu,
    int N, int K) {
  extern __shared__ __nv_bfloat16 xs[];
  for(int i=threadIdx.x;i<K;i+=BLOCK) xs[i]=X[i];
  __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int total=E*N; unsigned nch=(K+255)/256;
  for(int g=gw; g<total; g+=stride){
    int e=g/N, n=g%N; const __nv_bfloat16* wg=Wg+((size_t)e*N+n)*K; const __nv_bfloat16* wu=Wu+((size_t)e*N+n)*K;
    float ag=0,au=0;
    for(unsigned c=0;c<nch;c++){ unsigned k=c*256+lane*8; if(k>=(unsigned)K) continue;
      __nv_bfloat16 xv[8]; ldv8(xv,xs+k);
      __nv_bfloat16 g8[8]; ldv8(g8,wg+k); ag=dot8b(g8,xv,ag);
      __nv_bfloat16 u8[8]; ldv8(u8,wu+k); au=dot8b(u8,xv,au);
    }
    ag=wred(ag); au=wred(au);
    if(lane==0){ Yg[(size_t)e*N+n]=__float2bfloat16(ag); Yu[(size_t)e*N+n]=__float2bfloat16(au); }
  }
}

// ---------------- harness ----------------
static std::vector<__nv_bfloat16> mkrand(size_t n){ std::vector<__nv_bfloat16> v(n); for(size_t i=0;i<n;i++) v[i]=__float2bfloat16((float)(rand()%2001-1000)/4000.f); return v; }

struct Buf { std::vector<__nv_bfloat16*> W; __nv_bfloat16 *X,*Y,*Yref; int COPIES; size_t wb; };
static Buf alloc(int N,int K,int xstride){
  Buf b; b.wb=(size_t)E*N*K*2;
  b.COPIES=1; while((size_t)b.COPIES*b.wb<(size_t)400*1024*1024 && (size_t)(b.COPIES+1)*b.wb<(size_t)3ULL*1024*1024*1024 && b.COPIES<256) b.COPIES++;
  auto hW=mkrand(b.wb/2); auto hX=mkrand(xstride? (size_t)E*K : K);
  b.W.resize(b.COPIES); CHK(cudaMalloc(&b.W[0],b.wb)); CHK(cudaMemcpy(b.W[0],hW.data(),b.wb,cudaMemcpyHostToDevice));
  for(int c=1;c<b.COPIES;c++){ CHK(cudaMalloc(&b.W[c],b.wb)); CHK(cudaMemcpy(b.W[c],b.W[0],b.wb,cudaMemcpyDeviceToDevice)); }
  CHK(cudaMalloc(&b.X,(xstride?(size_t)E*K:K)*2)); CHK(cudaMemcpy(b.X,hX.data(),(xstride?(size_t)E*K:K)*2,cudaMemcpyHostToDevice));
  CHK(cudaMalloc(&b.Y,(size_t)E*N*2)); CHK(cudaMalloc(&b.Yref,(size_t)E*N*2));
  return b;
}
static void freeb(Buf&b){ for(auto p:b.W) cudaFree(p); cudaFree(b.X); cudaFree(b.Y); cudaFree(b.Yref); }

static int grid_;
template<class F> static double timeit(F launch){
  cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
  for(int i=0;i<5;i++) launch(i); CHK(cudaDeviceSynchronize());
  cudaEventRecord(e0); int R=100; for(int i=0;i<R;i++) launch(i);
  cudaEventRecord(e1); CHK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms,e0,e1);
  cudaEventDestroy(e0);cudaEventDestroy(e1); return ms/R;
}
static int verify(Buf&b,int N){ std::vector<__nv_bfloat16> a(E*N),r(E*N);
  CHK(cudaMemcpy(a.data(),b.Y,(size_t)E*N*2,cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(r.data(),b.Yref,(size_t)E*N*2,cudaMemcpyDeviceToHost));
  int m=0; for(int i=0;i<E*N;i++){ float x=__bfloat162float(a[i]),y=__bfloat162float(r[i]); if(fabsf(x-y)>0.02f*fmaxf(1.f,fabsf(y))) m++; } return m; }

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); grid_=p.multiProcessorCount;
  printf("# %s SMs=%d E=%d HBM_spec=%.0f GB/s  (achievable ceiling ~1535 GB/s)\n",p.name,grid_,E,HBM_SPEC);

  struct Sh{int N,K,xstride; const char*tag;} shs[]={ {704,2816,0,"gate/up N704 K2816"}, {2816,704,1,"down    N2816 K704 (per-expert x)"} };
  for(auto sh:shs){
    Buf b=alloc(sh.N,sh.K,sh.xstride); int N=sh.N,K=sh.K;
    size_t smx=(sh.xstride?(size_t)E*K:K)*2;
    auto bw=[&](double ms){ return (double)b.wb/(ms*1e-3)/1e9; };
    printf("\n== %s ==  weights=%.1f MB  COPIES=%d\n", sh.tag, b.wb/1048576.0, b.COPIES);

    // reference = flat
    gv_flat<<<grid_,BLOCK,smx>>>(b.W[0],b.X,b.Yref,N,K,sh.xstride); CHK(cudaDeviceSynchronize());
    double ta=timeit([&](int i){ gv_flat<<<grid_,BLOCK,smx>>>(b.W[i%b.COPIES],b.X,b.Yref,N,K,sh.xstride); });
    printf("  (a) flat        %6.3f ms  %6.0f GB/s  %4.0f%%\n", ta,bw(ta),100*bw(ta)/HBM_SPEC);

    size_t sm_b=smx+ (WPB/2)*2*sizeof(float);
    gv_splitk<<<grid_,BLOCK,sm_b>>>(b.W[0],b.X,b.Y,N,K,sh.xstride); CHK(cudaDeviceSynchronize());
    int mb=verify(b,N);
    double tb=timeit([&](int i){ gv_splitk<<<grid_,BLOCK,sm_b>>>(b.W[i%b.COPIES],b.X,b.Y,N,K,sh.xstride); });
    printf("  (b) split-K x2  %6.3f ms  %6.0f GB/s  %4.0f%%   %s\n", tb,bw(tb),100*bw(tb)/HBM_SPEC, mb?"[MISMATCH]":"ok");

    gv_ntile<<<grid_,BLOCK,smx>>>(b.W[0],b.X,b.Y,N,K,sh.xstride); CHK(cudaDeviceSynchronize());
    int mc=verify(b,N);
    double tc=timeit([&](int i){ gv_ntile<<<grid_,BLOCK,smx>>>(b.W[i%b.COPIES],b.X,b.Y,N,K,sh.xstride); });
    printf("  (c) N-tile/warp %6.3f ms  %6.0f GB/s  %4.0f%%   %s\n", tc,bw(tc),100*bw(tc)/HBM_SPEC, mc?"[MISMATCH]":"ok");
    freeb(b);
  }

  // (d) fused gate+up vs (a)+(a): two weight streams, x read once. gate/up shape only.
  {
    int N=704,K=2816; Buf g=alloc(N,K,0), u=alloc(N,K,0);
    size_t smx=(size_t)K*2; auto bw2=[&](double ms){ return 2.0*(double)g.wb/(ms*1e-3)/1e9; };
    printf("\n== fused gate+up N704 K2816 ==  weights=%.1f MB (x2 streams)\n", 2*g.wb/1048576.0);
    // baseline: two separate flat GEMVs
    double tsep=timeit([&](int i){ gv_flat<<<grid_,BLOCK,smx>>>(g.W[i%g.COPIES],g.X,g.Y,N,K,0);
                                   gv_flat<<<grid_,BLOCK,smx>>>(u.W[i%u.COPIES],u.X,u.Y,N,K,0); });
    printf("  (a+a) separate  %6.3f ms  %6.0f GB/s  %4.0f%%\n", tsep,bw2(tsep),100*bw2(tsep)/HBM_SPEC);
    double tf=timeit([&](int i){ gv_fused_gu<<<grid_,BLOCK,smx>>>(g.W[i%g.COPIES],u.W[i%u.COPIES],g.X,g.Y,u.Y,N,K); });
    printf("  (d) fused gu    %6.3f ms  %6.0f GB/s  %4.0f%%\n", tf,bw2(tf),100*bw2(tf)/HBM_SPEC);
    freeb(g); freeb(u);
  }
  return 0;
}
