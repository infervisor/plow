// fp8_gemv_moe.cu — fp8 (w8a16) weight-only decode GEMV for the MoE expert shapes (M=1).
// Extends fp8_gemv.cu's dequant-on-load dot to the launch-bound MoE case.
//
// Weight is e4m3 (1 byte/elt) row-major [E][N][K]; per-output-channel f32 scale[E*N] applied
// ONCE in the epilogue (factors out of the K-reduction). x is bf16, staged in smem.
//   gate/up : N=704 K=2816       down : N=2816 K=704       E=8 experts
//
// Variants:
//   (a) flat UN=4   : dot8_fp8, GV_UNROLL_FP8=4 (the current op_gemm.cuh setting)
//   (b) flat UN=8   : same, 8 weight loads in flight
//   (c) fused gu    : gate|up in ONE pass, each own scale, x read once (fp8 twin of d_gemv_glu_fp8)
//
// BW counts fp8 bytes (E*N*K). Question: does fp8 reach the same %HBM as the bf16 GEMV?
// Build: /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -Xptxas -v -o /tmp/f fp8_gemv_moe.cu

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define HBM_SPEC 1790.0
#define BLOCK 256
#define WPB   (BLOCK/32)
#define E     8

__device__ __forceinline__ float wred(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(~0u,v,o); return v; }

// dequant 8 e4m3 (uint2) and fma against 8 bf16 x  (byte-identical to op_gemm.cuh dot8_fp8)
__device__ __forceinline__ float dot8_fp8(const uint2& w8, const __nv_bfloat16 x[8], float acc){
  const uint16_t* wp=(const uint16_t*)&w8;
  #pragma unroll
  for(int j=0;j<4;j++){
    __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j],__NV_E4M3);
    float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
    acc=fmaf(f.x,__bfloat162float(x[2*j]),acc);
    acc=fmaf(f.y,__bfloat162float(x[2*j+1]),acc);
  }
  return acc;
}
__device__ __forceinline__ void ldx8(__nv_bfloat16 d[8], const __nv_bfloat16* p){ *(uint4*)d=*(const uint4*)p; }

// ---------- flat, templated unroll ----------
template<int UN>
__global__ __launch_bounds__(BLOCK) void gv_fp8_flat(
    const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ S, __nv_bfloat16* __restrict__ Y, int N, int K, int xstride){
  extern __shared__ __nv_bfloat16 xs[];
  int nx=xstride? E*K:K; for(int i=threadIdx.x;i<nx;i+=BLOCK) xs[i]=X[i]; __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int total=E*N; unsigned nch=(K+255)/256;
  for(int g=gw; g<total; g+=stride){
    int e=g/N, n=g%N; const uint8_t* wr=W+((size_t)e*N+n)*K; const __nv_bfloat16* xe=xs+(xstride? e*K:0);
    float acc=0;
    for(unsigned c=0;c<nch;c+=UN){
      uint2 wv[UN]; unsigned kk[UN];
      #pragma unroll
      for(int u=0;u<UN;u++){ unsigned k=(c+u)*256+lane*8; kk[u]=k; wv[u]=(k<(unsigned)K)?*(const uint2*)(wr+k):make_uint2(0,0); }
      #pragma unroll
      for(int u=0;u<UN;u++){ if(kk[u]<(unsigned)K){ __nv_bfloat16 xv[8]; ldx8(xv,xe+kk[u]); acc=dot8_fp8(wv[u],xv,acc);} }
    }
    acc=wred(acc); if(lane==0) Y[(size_t)e*N+n]=__float2bfloat16(acc*S[(size_t)e*N+n]);
  }
}

// ---------- fused gate+up single pass ----------
template<int UN>
__global__ __launch_bounds__(BLOCK) void gv_fp8_fused(
    const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu, const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ Sg, const float* __restrict__ Su,
    __nv_bfloat16* __restrict__ Yg, __nv_bfloat16* __restrict__ Yu, int N, int K){
  extern __shared__ __nv_bfloat16 xs[];
  for(int i=threadIdx.x;i<K;i+=BLOCK) xs[i]=X[i]; __syncthreads();
  int gw=blockIdx.x*WPB+(threadIdx.x>>5), lane=threadIdx.x&31, stride=gridDim.x*WPB;
  int total=E*N; unsigned nch=(K+255)/256;
  for(int g=gw; g<total; g+=stride){
    int e=g/N, n=g%N; const uint8_t* wg=Wg+((size_t)e*N+n)*K; const uint8_t* wu=Wu+((size_t)e*N+n)*K;
    float ag=0,au=0;
    for(unsigned c=0;c<nch;c+=UN){
      uint2 gv[UN],uv[UN]; unsigned kk[UN];
      #pragma unroll
      for(int u=0;u<UN;u++){ unsigned k=(c+u)*256+lane*8; kk[u]=k;
        gv[u]=(k<(unsigned)K)?*(const uint2*)(wg+k):make_uint2(0,0);
        uv[u]=(k<(unsigned)K)?*(const uint2*)(wu+k):make_uint2(0,0); }
      #pragma unroll
      for(int u=0;u<UN;u++){ if(kk[u]<(unsigned)K){ __nv_bfloat16 xv[8]; ldx8(xv,xs+kk[u]); ag=dot8_fp8(gv[u],xv,ag); au=dot8_fp8(uv[u],xv,au);} }
    }
    ag=wred(ag); au=wred(au);
    if(lane==0){ Yg[(size_t)e*N+n]=__float2bfloat16(ag*Sg[(size_t)e*N+n]); Yu[(size_t)e*N+n]=__float2bfloat16(au*Su[(size_t)e*N+n]); }
  }
}

// ---------------- harness ----------------
struct Buf { std::vector<uint8_t*> W; __nv_bfloat16 *X,*Y,*Yref; float* S; int COPIES; size_t wb; };
static Buf alloc(int N,int K,int xstride){
  Buf b; b.wb=(size_t)E*N*K; // fp8: 1 byte
  b.COPIES=1; while((size_t)b.COPIES*b.wb<(size_t)400*1024*1024 && (size_t)(b.COPIES+1)*b.wb<(size_t)3ULL*1024*1024*1024 && b.COPIES<256) b.COPIES++;
  std::vector<uint8_t> hW(b.wb); std::vector<__nv_bfloat16> hX(xstride? (size_t)E*K:K); std::vector<float> hS(E*N);
  srand(7); for(size_t i=0;i<b.wb;i++){ __nv_fp8_e4m3 q((float)(rand()%201-100)/400.f); hW[i]=*(uint8_t*)&q; }
  for(size_t i=0;i<hX.size();i++) hX[i]=__float2bfloat16((float)(rand()%2001-1000)/4000.f);
  for(int i=0;i<E*N;i++) hS[i]=0.5f+(rand()%100)/200.f;
  b.W.resize(b.COPIES); CHK(cudaMalloc(&b.W[0],b.wb)); CHK(cudaMemcpy(b.W[0],hW.data(),b.wb,cudaMemcpyHostToDevice));
  for(int c=1;c<b.COPIES;c++){ CHK(cudaMalloc(&b.W[c],b.wb)); CHK(cudaMemcpy(b.W[c],b.W[0],b.wb,cudaMemcpyDeviceToDevice)); }
  CHK(cudaMalloc(&b.X,hX.size()*2)); CHK(cudaMemcpy(b.X,hX.data(),hX.size()*2,cudaMemcpyHostToDevice));
  CHK(cudaMalloc(&b.S,E*N*4)); CHK(cudaMemcpy(b.S,hS.data(),E*N*4,cudaMemcpyHostToDevice));
  CHK(cudaMalloc(&b.Y,(size_t)E*N*2)); CHK(cudaMalloc(&b.Yref,(size_t)E*N*2));
  return b;
}
static void freeb(Buf&b){ for(auto p:b.W) cudaFree(p); cudaFree(b.X);cudaFree(b.S);cudaFree(b.Y);cudaFree(b.Yref); }
static int grid_;
template<class F> static double timeit(F launch){
  cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
  for(int i=0;i<5;i++) launch(i); CHK(cudaDeviceSynchronize());
  cudaEventRecord(e0); int R=100; for(int i=0;i<R;i++) launch(i);
  cudaEventRecord(e1); CHK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms,e0,e1);
  cudaEventDestroy(e0);cudaEventDestroy(e1); return ms/R; }
static int verify(Buf&b,int N){ std::vector<__nv_bfloat16> a(E*N),r(E*N);
  CHK(cudaMemcpy(a.data(),b.Y,(size_t)E*N*2,cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(r.data(),b.Yref,(size_t)E*N*2,cudaMemcpyDeviceToHost));
  int m=0; for(int i=0;i<E*N;i++){ float x=__bfloat162float(a[i]),y=__bfloat162float(r[i]); if(fabsf(x-y)>0.02f*fmaxf(1.f,fabsf(y))) m++; } return m; }

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0)); grid_=p.multiProcessorCount;
  printf("# %s SMs=%d E=%d HBM_spec=%.0f GB/s (achievable ~1535)  fp8 bytes=E*N*K\n",p.name,grid_,E,HBM_SPEC);
  struct Sh{int N,K,xstride;const char*tag;} shs[]={ {704,2816,0,"gate/up N704 K2816"}, {2816,704,1,"down    N2816 K704"} };
  for(auto sh:shs){
    Buf b=alloc(sh.N,sh.K,sh.xstride); int N=sh.N,K=sh.K; size_t smx=(sh.xstride?(size_t)E*K:K)*2;
    auto bw=[&](double ms){ return (double)b.wb/(ms*1e-3)/1e9; };
    printf("\n== %s ==  fp8 weights=%.1f MB  COPIES=%d\n", sh.tag, b.wb/1048576.0, b.COPIES);
    gv_fp8_flat<4><<<grid_,BLOCK,smx>>>(b.W[0],b.X,b.S,b.Yref,N,K,sh.xstride); CHK(cudaDeviceSynchronize());
    double t4=timeit([&](int i){ gv_fp8_flat<4><<<grid_,BLOCK,smx>>>(b.W[i%b.COPIES],b.X,b.S,b.Yref,N,K,sh.xstride); });
    printf("  (a) flat UN=4   %6.3f ms  %6.0f GB/s  %4.0f%%\n", t4,bw(t4),100*bw(t4)/HBM_SPEC);
    gv_fp8_flat<8><<<grid_,BLOCK,smx>>>(b.W[0],b.X,b.S,b.Y,N,K,sh.xstride); CHK(cudaDeviceSynchronize());
    int m8=verify(b,N);
    double t8=timeit([&](int i){ gv_fp8_flat<8><<<grid_,BLOCK,smx>>>(b.W[i%b.COPIES],b.X,b.S,b.Y,N,K,sh.xstride); });
    printf("  (b) flat UN=8   %6.3f ms  %6.0f GB/s  %4.0f%%   %s\n", t8,bw(t8),100*bw(t8)/HBM_SPEC, m8?"[MISMATCH]":"ok");
    freeb(b);
  }
  // fused gate+up fp8
  { int N=704,K=2816; Buf g=alloc(N,K,0),u=alloc(N,K,0); size_t smx=(size_t)K*2;
    auto bw2=[&](double ms){ return 2.0*(double)g.wb/(ms*1e-3)/1e9; };
    printf("\n== fused gate+up fp8 N704 K2816 ==  fp8 weights=%.1f MB (x2)\n", 2*g.wb/1048576.0);
    double tsep=timeit([&](int i){ gv_fp8_flat<4><<<grid_,BLOCK,smx>>>(g.W[i%g.COPIES],g.X,g.S,g.Y,N,K,0);
                                   gv_fp8_flat<4><<<grid_,BLOCK,smx>>>(u.W[i%u.COPIES],u.X,u.S,u.Y,N,K,0); });
    printf("  (a+a) separate  %6.3f ms  %6.0f GB/s  %4.0f%%\n", tsep,bw2(tsep),100*bw2(tsep)/HBM_SPEC);
    double tf=timeit([&](int i){ gv_fp8_fused<4><<<grid_,BLOCK,smx>>>(g.W[i%g.COPIES],u.W[i%u.COPIES],g.X,g.S,u.S,g.Y,u.Y,N,K); });
    printf("  (c) fused gu    %6.3f ms  %6.0f GB/s  %4.0f%%\n", tf,bw2(tf),100*bw2(tf)/HBM_SPEC);
    freeb(g); freeb(u);
  }
  return 0;
}
