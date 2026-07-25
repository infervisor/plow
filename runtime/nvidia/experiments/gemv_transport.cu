// sm_120 decode-GEMV weight-transport microbenchmark.
// Measures: direct vectorized loads (4/8/16B), __ldg, cp.async.cg, cp.async.bulk (1-D TMA),
// cp.async.bulk.tensor.2d (TMA), across bf16 and fp8 weight streams, with a grid sweep.
//
// Shape: M=1 GEMV, W[N,K] row-major stride K (matches gemma_sm120.cu:179 / op_gemm.h:5),
// K = 2560 (Qwen3-4B hidden). N chosen so the weight buffer is ~1 GB => 10x the 96 MB L2.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CKD(x) do{ CUresult e=(x); if(e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(e,&s); \
  printf("CU ERR %s @%d: %s\n",#x,__LINE__,s); exit(1);} }while(0)

static const int K = 2560;          // Qwen3-4B hidden
static const int WARP = 32;

// ---------------------------------------------------------------- mbarrier / TMA helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar){
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase){
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
               " @!p bra W%=; }" :: "r"(smem_u32(bar)), "r"(phase) : "memory");
}
// 1-D bulk copy (TMA, no descriptor)
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar){
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// 2-D tensor copy (TMA, descriptor-based)
__device__ __forceinline__ void bulk_tensor_2d(void* dst, const CUtensorMap* map,
                                               int c0, int c1, uint64_t* bar){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}

// ---------------------------------------------------------------- reductions (warp=32)
__device__ __forceinline__ float warp_sum(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------------------------------------------------------- A: direct vectorized loads
// VB = vector bytes per lane per load (4, 8, 16). LDG = use __ldg (ld.global.nc).
// One warp per output row; grid-stride over rows.
template<int VB, bool LDG>
__global__ void gemv_bf16_direct(const __nv_bfloat16* W, const float* __restrict__ x,
                                 float* __restrict__ y, int N){
  extern __shared__ float xs[];
  for(int i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const int EV = VB/2;                       // bf16 elements per vector
  const int warps = blockDim.x/WARP;
  const int wid = threadIdx.x/WARP, lane = threadIdx.x%WARP;
  for(int row = blockIdx.x*warps + wid; row < N; row += gridDim.x*warps){
    const __nv_bfloat16* wr = W + (size_t)row*K;
    float acc = 0.f;
    for(int k = lane*EV; k < K; k += WARP*EV){
      if(VB==16){
        uint4 v; const uint4* p = (const uint4*)(wr+k);
        v = LDG ? __ldg(p) : *p;
        const __nv_bfloat16* h = (const __nv_bfloat16*)&v;
        #pragma unroll
        for(int j=0;j<8;j++) acc += __bfloat162float(h[j])*xs[k+j];
      } else if(VB==8){
        uint2 v; const uint2* p = (const uint2*)(wr+k);
        v = LDG ? __ldg(p) : *p;
        const __nv_bfloat16* h = (const __nv_bfloat16*)&v;
        #pragma unroll
        for(int j=0;j<4;j++) acc += __bfloat162float(h[j])*xs[k+j];
      } else {
        uint32_t v; const uint32_t* p = (const uint32_t*)(wr+k);
        v = LDG ? __ldg(p) : *p;
        const __nv_bfloat16* h = (const __nv_bfloat16*)&v;
        #pragma unroll
        for(int j=0;j<2;j++) acc += __bfloat162float(h[j])*xs[k+j];
      }
    }
    acc = warp_sum(acc);
    if(lane==0) y[row] = acc;
  }
}

// fp8 e4m3 twin. VB bytes = VB elements. scale applied once in the epilogue
// (per-output-channel multiplier, per op_gemm.h:1439).
template<int VB, bool LDG>
__global__ void gemv_fp8_direct(const __nv_fp8_e4m3* W, const float* __restrict__ x,
                                const float* __restrict__ ws, float* __restrict__ y, int N){
  extern __shared__ float xs[];
  for(int i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const int warps = blockDim.x/WARP;
  const int wid = threadIdx.x/WARP, lane = threadIdx.x%WARP;
  for(int row = blockIdx.x*warps + wid; row < N; row += gridDim.x*warps){
    const __nv_fp8_e4m3* wr = W + (size_t)row*K;
    float acc = 0.f;
    for(int k = lane*VB; k < K; k += WARP*VB){
      if(VB==16){
        uint4 v; const uint4* p=(const uint4*)(wr+k); v = LDG?__ldg(p):*p;
        const __nv_fp8_e4m3* h=(const __nv_fp8_e4m3*)&v;
        #pragma unroll
        for(int j=0;j<16;j++) acc += float(h[j])*xs[k+j];
      } else if(VB==8){
        uint2 v; const uint2* p=(const uint2*)(wr+k); v = LDG?__ldg(p):*p;
        const __nv_fp8_e4m3* h=(const __nv_fp8_e4m3*)&v;
        #pragma unroll
        for(int j=0;j<8;j++) acc += float(h[j])*xs[k+j];
      } else {
        uint32_t v; const uint32_t* p=(const uint32_t*)(wr+k); v = LDG?__ldg(p):*p;
        const __nv_fp8_e4m3* h=(const __nv_fp8_e4m3*)&v;
        #pragma unroll
        for(int j=0;j<4;j++) acc += float(h[j])*xs[k+j];
      }
    }
    acc = warp_sum(acc);
    if(lane==0) y[row] = acc * ws[row];
  }
}

// ---------------------------------------------------------------- B: cp.async.cg staged
// Block owns ROWS consecutive rows; stage = ROWS x SK tile, double buffered.
// 256 threads, ROWS=8 (one warp per row), SK=256 bf16 = 512 B/row/stage.
#define ROWS 8
#define SK   256
__global__ void gemv_bf16_cpasync(const __nv_bfloat16* W, const float* __restrict__ x,
                                  float* __restrict__ y, int N){
  extern __shared__ char smem[];
  float* xs = (float*)smem;                                   // K floats
  __nv_bfloat16* tile = (__nv_bfloat16*)(smem + K*4);         // 2 * ROWS * SK
  for(int i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  __syncthreads();
  const int lane = threadIdx.x%WARP, wid = threadIdx.x/WARP;
  const int nst = K/SK;
  // each stage: ROWS*SK*2 bytes = 4096 B; 256 threads * 16 B = 4096 B -> 1 cp.async each
  for(int rb = blockIdx.x*ROWS; rb < N; rb += gridDim.x*ROWS){
    float acc = 0.f;
    for(int s=0; s<nst; ++s){
      __nv_bfloat16* dst = tile + (s&1)*(ROWS*SK);
      // 16 B per thread: thread t -> row t/(SK/8), col (t%(SK/8))*8
      int r = threadIdx.x/(SK/8), c = (threadIdx.x%(SK/8))*8;
      const __nv_bfloat16* src = W + (size_t)(rb+r)*K + s*SK + c;
      uint32_t d = smem_u32(dst + r*SK + c);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
      const __nv_bfloat16* mr = dst + wid*SK;
      for(int k=lane*8;k<SK;k+=WARP*8){
        #pragma unroll
        for(int j=0;j<8;j++) acc += __bfloat162float(mr[k+j])*xs[s*SK+k+j];
      }
      __syncthreads();
    }
    acc = warp_sum(acc);
    if(lane==0 && rb+wid < N) y[rb+wid] = acc;
  }
}

// ---------------------------------------------------------------- C: cp.async.bulk 1-D (TMA)
// Block owns ROWS consecutive rows = ROWS*K contiguous bf16. Stage = SR full rows (contiguous).
#define SR 2
__global__ void gemv_bf16_bulk1d(const __nv_bfloat16* W, const float* __restrict__ x,
                                 float* __restrict__ y, int N){
  extern __shared__ char smem[];
  float* xs = (float*)smem;
  __nv_bfloat16* tile = (__nv_bfloat16*)(smem + K*4);   // 2 buffers * SR * K
  __shared__ uint64_t bar[2];
  __shared__ float rowacc[ROWS];
  for(int i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  if(threadIdx.x==0){ mbar_init(&bar[0],1); mbar_init(&bar[1],1); }
  __syncthreads();
  const int STB = SR*K*2;                                // bytes per stage
  const int nst = ROWS/SR;
  const int tpr = blockDim.x/SR;                         // threads per row in a stage
  const int myr = threadIdx.x/tpr, myl = threadIdx.x%tpr;
  int it = 0;   // running stage counter: phase must be continuous across rb iterations
  for(int rb = blockIdx.x*ROWS; rb < N; rb += gridDim.x*ROWS){
    if(threadIdx.x<ROWS) rowacc[threadIdx.x]=0.f;
    __syncthreads();
    for(int s=0; s<nst; ++s,++it){
      __nv_bfloat16* dst = tile + (it&1)*(SR*K);
      if(threadIdx.x==0){
        mbar_expect(&bar[it&1], STB);
        bulk_g2s(dst, W + (size_t)(rb + s*SR)*K, STB, &bar[it&1]);
      }
      mbar_wait(&bar[it&1], (it>>1)&1);
      __syncthreads();
      float acc=0.f;
      const __nv_bfloat16* mr = dst + myr*K;
      for(int k=myl*8;k<K;k+=tpr*8){
        #pragma unroll
        for(int j=0;j<8;j++) acc += __bfloat162float(mr[k+j])*xs[k+j];
      }
      acc = warp_sum(acc);
      if((threadIdx.x%WARP)==0) atomicAdd(&rowacc[s*SR+myr], acc);
      __syncthreads();
    }
    if(threadIdx.x<ROWS && rb+threadIdx.x<N) y[rb+threadIdx.x]=rowacc[threadIdx.x];
    __syncthreads();
  }
}

// ---------------------------------------------------------------- D: cp.async.bulk.tensor.2d
// Descriptor over W viewed as [N, K] bf16; box = (SK cols, ROWS rows).
__global__ void gemv_bf16_tma2d(const __grid_constant__ CUtensorMap map,
                                const float* __restrict__ x, float* __restrict__ y, int N){
  extern __shared__ char smem[];
  // TMA requires the shared-memory destination to be 128-B aligned.
  uint32_t tb = (smem_u32(smem) + 127u) & ~127u;
  __nv_bfloat16* tile = (__nv_bfloat16*)__cvta_shared_to_generic(tb);
  float* xs = (float*)(tile + 2*ROWS*SK);
  __shared__ __align__(8) uint64_t bar[2];
  for(int i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
  if(threadIdx.x==0){ mbar_init(&bar[0],1); mbar_init(&bar[1],1); }
  __syncthreads();
  const int lane=threadIdx.x%WARP, wid=threadIdx.x/WARP;
  const int nst=K/SK, TB=ROWS*SK*2;
  int it = 0;   // running stage counter: phase must be continuous across rb iterations
  for(int rb = blockIdx.x*ROWS; rb < N; rb += gridDim.x*ROWS){
    float acc=0.f;
    for(int s=0;s<nst;++s,++it){
      __nv_bfloat16* dst = tile + (it&1)*(ROWS*SK);
      if(threadIdx.x==0){
        mbar_expect(&bar[it&1], TB);
        bulk_tensor_2d(dst, &map, s*SK, rb, &bar[it&1]);
      }
      mbar_wait(&bar[it&1], (it>>1)&1);
      __syncthreads();
      const __nv_bfloat16* mr = dst + wid*SK;
      for(int k=lane*8;k<SK;k+=WARP*8){
        #pragma unroll
        for(int j=0;j<8;j++) acc += __bfloat162float(mr[k+j])*xs[s*SK+k+j];
      }
      __syncthreads();
    }
    acc = warp_sum(acc);
    if(lane==0 && rb+wid<N) y[rb+wid]=acc;
  }
}

// ---------------------------------------------------------------- harness
struct Res { std::string name; double gbs; double ms; double rell2; int grid; int block; };

static double relL2(const std::vector<float>& a, const std::vector<float>& b, int n){
  double num=0, den=0;
  for(int i=0;i<n;i++){ double d=a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
  return sqrt(num)/sqrt(den>0?den:1);
}

int main(int argc, char** argv){
  CKD(cuInit(0));
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,dev));
  size_t freeB, totB; CK(cudaMemGetInfo(&freeB,&totB));
  printf("# device=%s SMs=%d L2=%.1f MB freeMem=%.2f GB\n",
         prop.name, prop.multiProcessorCount, prop.l2CacheSize/1048576.0, freeB/1e9);

  // ~1 GB bf16 weight buffer => 10x L2, and N a multiple of ROWS
  size_t target = 1024ull*1024*1024;
  if(freeB < target + 512ull*1024*1024) target = freeB/2;
  int N = (int)(target/(K*2));
  N = (N/ROWS)*ROWS;
  size_t wbytes_bf16 = (size_t)N*K*2, wbytes_fp8 = (size_t)N*K;
  printf("# N=%d K=%d  bf16 buffer=%.3f GB (%.1fx L2)  fp8 buffer=%.3f GB\n",
         N,K, wbytes_bf16/1e9, wbytes_bf16/(double)prop.l2CacheSize, wbytes_fp8/1e9);

  // host data
  std::vector<float> hx(K); for(int i=0;i<K;i++) hx[i] = (float)((i*37%211)-105)/105.f;
  std::vector<__nv_bfloat16> hw_head((size_t)64*K);
  srand(1234);
  // fill device buffer in chunks (avoid a 1 GB host allocation twice)
  __nv_bfloat16* dW; CK(cudaMalloc(&dW, wbytes_bf16));
  __nv_fp8_e4m3* dW8; CK(cudaMalloc(&dW8, wbytes_fp8));
  float *dx,*dy,*dws;
  CK(cudaMalloc(&dx,K*4)); CK(cudaMalloc(&dy,(size_t)N*4)); CK(cudaMalloc(&dws,(size_t)N*4));
  CK(cudaMemcpy(dx,hx.data(),K*4,cudaMemcpyHostToDevice));

  {
    const int CH = 4096;                       // rows per host chunk
    std::vector<__nv_bfloat16> buf((size_t)CH*K);
    std::vector<__nv_fp8_e4m3> buf8((size_t)CH*K);
    std::vector<float> ws(CH);
    for(int r0=0;r0<N;r0+=CH){
      int nr = std::min(CH, N-r0);
      static uint32_t st = 12345u;
      for(int r=0;r<nr;r++){
        for(int k=0;k<K;k++){
          st = st*1664525u + 1013904223u;
          float v = ((float)(st>>8)/16777216.0f - 0.5f)*0.1f;
          buf[(size_t)r*K+k]=__float2bfloat16(v);
          // quantize from the bf16 value so the host fp8 reference is not double-rounded
          buf8[(size_t)r*K+k]=__nv_fp8_e4m3(__bfloat162float(buf[(size_t)r*K+k]));
        }
        ws[r]=1.0f;
      }
      CK(cudaMemcpy(dW+(size_t)r0*K, buf.data(), (size_t)nr*K*2, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dW8+(size_t)r0*K, buf8.data(), (size_t)nr*K, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dws+r0, ws.data(), nr*4, cudaMemcpyHostToDevice));
      if(r0==0) memcpy(hw_head.data(), buf.data(), (size_t)64*K*2);
    }
  }

  // CPU reference for first 64 rows (bf16 path)
  std::vector<float> ref(64,0.f);
  for(int r=0;r<64;r++){ double a=0; for(int k=0;k<K;k++) a += (double)__bfloat162float(hw_head[(size_t)r*K+k])*hx[k]; ref[r]=(float)a; }

  std::vector<float> hy((size_t)N);
  std::vector<Res> res;
  const int ITERS = 20, WARM = 3;

  auto run = [&](const char* name, int grid, int block, size_t smem, size_t bytes,
                 auto launch)->void{
    CK(cudaMemset(dy,0,(size_t)N*4));
    for(int i=0;i<WARM;i++) launch(grid,block,smem);
    cudaError_t we = cudaDeviceSynchronize();
    if(we!=cudaSuccess){
      printf("%-34s grid=%-6d blk=%-4d  FAILED: %s\n",name,grid,block,cudaGetErrorString(we));
      cudaGetLastError(); return;
    }
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for(int i=0;i<ITERS;i++) launch(grid,block,smem);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
    CK(cudaMemcpy(hy.data(),dy,(size_t)64*4,cudaMemcpyDeviceToHost));
    double r2 = relL2(hy, ref, 64);
    double gbs = bytes/(ms*1e-3)/1e9;
    res.push_back({name, gbs, ms, r2, grid, block});
    printf("%-34s grid=%-6d blk=%-4d %8.3f ms  %8.1f GB/s  relL2=%.3e\n",
           name, grid, block, ms, gbs, r2);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  };

  int SMs = prop.multiProcessorCount;

  // ---- derived grid for the direct kernel
  int occ=0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(void*)gemv_bf16_direct<16,false>,256,K*4));
  int derived = occ*SMs;
  printf("# occupancy(direct,blk256)=%d blocks/SM -> derived grid=%d\n", occ, derived);

  printf("\n=== A. bf16 direct vectorized loads (grid=%d, blk=256) ===\n", derived);
  run("bf16 direct  4B/lane", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_direct<4,false><<<g,b,s>>>(dW,dx,dy,N);});
  run("bf16 direct  8B/lane", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_direct<8,false><<<g,b,s>>>(dW,dx,dy,N);});
  run("bf16 direct 16B/lane", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_direct<16,false><<<g,b,s>>>(dW,dx,dy,N);});
  run("bf16 __ldg  16B/lane", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_direct<16,true><<<g,b,s>>>(dW,dx,dy,N);});
  run("bf16 __ldg   8B/lane", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_direct<8,true><<<g,b,s>>>(dW,dx,dy,N);});

  printf("\n=== B. bf16 smem-staged transports (grid=%d, blk=256) ===\n", derived);
  size_t sm_cg = K*4 + 2*ROWS*SK*2;
  run("bf16 cp.async.cg", derived,256,sm_cg,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_cpasync<<<g,b,s>>>(dW,dx,dy,N);});
  size_t sm_b1 = K*4 + 2*SR*K*2;
  CK(cudaFuncSetAttribute((void*)gemv_bf16_bulk1d,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm_b1));
  run("bf16 cp.async.bulk 1-D (TMA)", derived,256,sm_b1,wbytes_bf16,[&](int g,int b,size_t s){
    gemv_bf16_bulk1d<<<g,b,s>>>(dW,dx,dy,N);});

  // TMA 2-D descriptor
  {
    CUtensorMap map; memset(&map,0,sizeof(map));
    uint64_t gdim[2] = {(uint64_t)K, (uint64_t)N};          // {inner, outer}
    uint64_t gstr[1] = {(uint64_t)K*2};                     // row stride bytes
    uint32_t bdim[2] = {(uint32_t)SK, (uint32_t)ROWS};
    uint32_t estr[2] = {1,1};
    CKD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dW,
        gdim, gstr, bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    printf("# TMA 2-D descriptor encoded OK (box %dx%d bf16, swizzle NONE)\n", ROWS, SK);
    size_t sm_t2 = 128 + K*4 + 2*ROWS*SK*2;
    CK(cudaFuncSetAttribute((void*)gemv_bf16_tma2d,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm_t2));
    run("bf16 cp.async.bulk.tensor.2d", derived,256,sm_t2,wbytes_bf16,[&](int g,int b,size_t s){
      gemv_bf16_tma2d<<<g,b,s>>>(map,dx,dy,N);});
  }

  printf("\n=== C. fp8 e4m3 direct vectorized loads ===\n");
  // fp8 reference (scale=1) computed from the same values, rounded to e4m3
  {
    std::vector<float> ref8(64,0.f);
    for(int r=0;r<64;r++){ double a=0;
      for(int k=0;k<K;k++) a += (double)float(__nv_fp8_e4m3(__bfloat162float(hw_head[(size_t)r*K+k])))*hx[k];
      ref8[r]=(float)a; }
    auto run8=[&](const char* name,int grid,int block,auto launch){
      CK(cudaMemset(dy,0,(size_t)N*4));
      for(int i=0;i<WARM;i++) launch(grid,block);
      CK(cudaDeviceSynchronize());
      cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0)); for(int i=0;i<ITERS;i++) launch(grid,block);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=ITERS;
      CK(cudaMemcpy(hy.data(),dy,(size_t)64*4,cudaMemcpyDeviceToHost));
      double r2=relL2(hy,ref8,64), gbs=wbytes_fp8/(ms*1e-3)/1e9;
      res.push_back({name,gbs,ms,r2,grid,block});
      printf("%-34s grid=%-6d blk=%-4d %8.3f ms  %8.1f GB/s  relL2=%.3e\n",name,grid,block,ms,gbs,r2);
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    };
    run8("fp8  direct  4B/lane",derived,256,[&](int g,int b){
      gemv_fp8_direct<4,false><<<g,b,K*4>>>(dW8,dx,dws,dy,N);});
    run8("fp8  direct  8B/lane",derived,256,[&](int g,int b){
      gemv_fp8_direct<8,false><<<g,b,K*4>>>(dW8,dx,dws,dy,N);});
    run8("fp8  direct 16B/lane",derived,256,[&](int g,int b){
      gemv_fp8_direct<16,false><<<g,b,K*4>>>(dW8,dx,dws,dy,N);});
    run8("fp8  __ldg  16B/lane",derived,256,[&](int g,int b){
      gemv_fp8_direct<16,true><<<g,b,K*4>>>(dW8,dx,dws,dy,N);});
  }

  printf("\n=== D. grid sweep, bf16 direct 16B/lane, blk=256 (derived=%d) ===\n", derived);
  for(double f : {0.25,0.5,1.0,1.5,2.0,3.0,4.0}){
    int g = (int)(derived*f); if(g<1) g=1;
    char nm[64]; snprintf(nm,64,"grid %.2fx derived (%d)",f,g);
    run(nm,g,256,K*4,wbytes_bf16,[&](int gg,int b,size_t s){
      gemv_bf16_direct<16,false><<<gg,b,s>>>(dW,dx,dy,N);});
  }
  printf("\n=== D2. grid = k * SMs sweep (SMs=%d), bf16 16B/lane ===\n", SMs);
  for(int k : {1,2,3,4,6,8,12,16}){
    char nm[64]; snprintf(nm,64,"grid %d x SM (%d)",k,k*SMs);
    run(nm,k*SMs,256,K*4,wbytes_bf16,[&](int gg,int b,size_t s){
      gemv_bf16_direct<16,false><<<gg,b,s>>>(dW,dx,dy,N);});
  }
  printf("\n=== D3. wrong-SM-count control: 188 vs 170 ===\n");
  for(int smc : {170,188}){
    for(int k : {2,4}){
      char nm[64]; snprintf(nm,64,"SMS=%d k=%d (grid %d)",smc,k,k*smc);
      run(nm,k*smc,256,K*4,wbytes_bf16,[&](int gg,int b,size_t s){
        gemv_bf16_direct<16,false><<<gg,b,s>>>(dW,dx,dy,N);});
    }
  }
  printf("\n=== E. block size sweep, bf16 16B/lane ===\n");
  for(int b : {64,128,256,512,1024}){
    int o=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o,(void*)gemv_bf16_direct<16,false>,b,K*4));
    char nm[64]; snprintf(nm,64,"blk=%d occ=%d",b,o);
    run(nm,o*SMs,b,K*4,wbytes_bf16,[&](int gg,int bb,size_t s){
      gemv_bf16_direct<16,false><<<gg,bb,s>>>(dW,dx,dy,N);});
  }

  // ---- negative control: perturb one weight, confirm relL2 detects
  printf("\n=== NEGATIVE CONTROL ===\n");
  {
    __nv_bfloat16 bad = __float2bfloat16(50.0f);
    CK(cudaMemcpy(dW, &bad, 2, cudaMemcpyHostToDevice));   // W[0,0] = 50
    run("perturbed W[0,0]=50 (must FAIL)", derived,256,K*4,wbytes_bf16,[&](int g,int b,size_t s){
      gemv_bf16_direct<16,false><<<g,b,s>>>(dW,dx,dy,N);});
    printf("negative control: relL2 above must be >> 1e-6 for the test to be meaningful\n");
  }

  printf("\n=== RANKED (bf16 full-pass variants) ===\n");
  std::sort(res.begin(),res.end(),[](const Res&a,const Res&b){return a.gbs>b.gbs;});
  for(auto&r:res) printf("%-34s %8.1f GB/s  %6.1f%% of 1673  relL2=%.2e\n",
      r.name.c_str(), r.gbs, 100.0*r.gbs/1673.0, r.rell2);
  return 0;
}
