// sm_120 L2 residency control for the KV cache (cudaAccessPolicyWindow).
//
// The real question is NOT "does a small KV cache fit in 96 MB L2" (it trivially does),
// but "does it SURVIVE the multi-GB weight stream that runs in the same decode step".
// So every measurement interleaves a weight-stream kernel with the flash-decode KV read.
//
// Qwen3-4B: 8 KV heads, head_dim 128, 32 q heads => gqa/GF = 4.
// KV bytes at context L = 2 * L * 8 * 128 * 2 = L * 4096 bytes.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const int HD  = 128;   // head dim
static const int KVH = 8;     // kv heads
static const int GF  = 4;     // q heads per kv head (Qwen3 gqa=4)
static const int WARP = 32;

__device__ __forceinline__ float warp_sum(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// flash-decode shaped KV read. K,V laid out [KVH, L, HD] bf16.
// grid.x = KVH * nsplit. Each warp walks a strided subset of timesteps.
// Each KV element is read ONCE and used by all GF q heads (the GF=4 win).
__global__ void flash_decode(const __nv_bfloat16* __restrict__ Kc,
                             const __nv_bfloat16* __restrict__ Vc,
                             const float* __restrict__ Q,        // [KVH*GF, HD]
                             float* __restrict__ out,            // [KVH*GF, HD] partials
                             int L, int nsplit){
  const int h  = blockIdx.x / nsplit;
  const int sp = blockIdx.x % nsplit;
  const int lane = threadIdx.x % WARP;
  const int wid  = threadIdx.x / WARP;
  const int nw   = blockDim.x / WARP;

  // lane holds 4 of the 128 dims: dims lane*4 .. lane*4+3
  float q[GF][4];
  #pragma unroll
  for(int g=0; g<GF; ++g)
    #pragma unroll
    for(int d=0; d<4; ++d)
      q[g][d] = Q[(size_t)(h*GF+g)*HD + lane*4 + d];

  float acc[GF][4];
  #pragma unroll
  for(int g=0; g<GF; ++g)
    #pragma unroll
    for(int d=0; d<4; ++d) acc[g][d] = 0.f;

  // timestep range for this split, then warp-strided within it
  const int per = (L + nsplit - 1) / nsplit;
  const int t0 = sp*per, t1 = min(L, t0+per);

  for(int t = t0 + wid; t < t1; t += nw){
    const __nv_bfloat16* kp = Kc + ((size_t)h*L + t)*HD + lane*4;
    const __nv_bfloat16* vp = Vc + ((size_t)h*L + t)*HD + lane*4;
    uint2 kv = *(const uint2*)kp;           // 8 B = 4 bf16
    uint2 vv = *(const uint2*)vp;
    const __nv_bfloat16* kh = (const __nv_bfloat16*)&kv;
    const __nv_bfloat16* vh = (const __nv_bfloat16*)&vv;
    float kf[4], vf[4];
    #pragma unroll
    for(int d=0; d<4; ++d){ kf[d]=__bfloat162float(kh[d]); vf[d]=__bfloat162float(vh[d]); }
    #pragma unroll
    for(int g=0; g<GF; ++g){
      float s = 0.f;
      #pragma unroll
      for(int d=0; d<4; ++d) s += q[g][d]*kf[d];
      s = warp_sum(s);                       // full 128-dim dot
      float w = __expf(s*0.08838834f - 8.f); // scaled, shifted; keeps values bounded
      #pragma unroll
      for(int d=0; d<4; ++d) acc[g][d] += w*vf[d];
    }
  }
  #pragma unroll
  for(int g=0; g<GF; ++g)
    #pragma unroll
    for(int d=0; d<4; ++d)
      atomicAdd(&out[(size_t)(h*GF+g)*HD + lane*4 + d], acc[g][d]);
}

// competing weight stream: reads `bytes` of weights, exactly like a decode step's GEMV
__global__ void weight_stream(const __nv_bfloat16* W, float* sink, size_t nelem){
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x*blockDim.x*8;
  for(size_t i = ((size_t)blockIdx.x*blockDim.x + threadIdx.x)*8; i < nelem; i += stride){
    uint4 v = *(const uint4*)(W + i);
    const __nv_bfloat16* h = (const __nv_bfloat16*)&v;
    #pragma unroll
    for(int j=0;j<8;j++) acc += __bfloat162float(h[j]);
  }
  if(acc == 1234.5678f) sink[0] = acc;   // never true; keeps the loop alive
}

int main(){
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,dev));
  printf("# %s SMs=%d L2=%.1f MB persistingL2Max=%.1f MB accessPolicyMaxWindow=%.1f MB\n",
    prop.name, prop.multiProcessorCount, prop.l2CacheSize/1048576.0,
    prop.persistingL2CacheMaxSize/1048576.0, prop.accessPolicyMaxWindowSize/1048576.0);

  const int SMs = prop.multiProcessorCount;
  // competing weight stream: 1 GB, ~ one Qwen3-4B layer group; enough to blow 96 MB L2 many times
  size_t wbytes = 1024ull*1024*1024;
  size_t wnel = wbytes/2;
  __nv_bfloat16* dW; CK(cudaMalloc(&dW, wbytes));
  CK(cudaMemset(dW, 0x3c, wbytes));
  float* dsink; CK(cudaMalloc(&dsink, 4));

  float* dQ; CK(cudaMalloc(&dQ, (size_t)KVH*GF*HD*4));
  {
    std::vector<float> hq((size_t)KVH*GF*HD);
    for(size_t i=0;i<hq.size();i++) hq[i] = (float)((int)(i*29%97)-48)/48.f;
    CK(cudaMemcpy(dQ,hq.data(),hq.size()*4,cudaMemcpyHostToDevice));
  }
  float* dOut; CK(cudaMalloc(&dOut,(size_t)KVH*GF*HD*4));

  cudaStream_t s; CK(cudaStreamCreate(&s));

  printf("\n%-8s %-9s %-26s %9s %9s %9s\n","ctxL","KV MB","config","KV ms","KV GB/s","step ms");
  printf("------------------------------------------------------------------------------\n");

  std::vector<int> Ls = {2048,4096,8192,16384,32768,65536};
  const int ITERS=30, WARM=5;

  struct Row { int L; double mb; const char* cfg; double kvms; double gbs; double stepms; };
  std::vector<Row> rows;

  for(int L : Ls){
    size_t kvb_each = (size_t)KVH*L*HD*2;     // K alone
    size_t kvb = 2*kvb_each;                  // K + V
    __nv_bfloat16 *dK,*dV;
    CK(cudaMalloc(&dK,kvb_each)); CK(cudaMalloc(&dV,kvb_each));
    CK(cudaMemset(dK,0x3c,kvb_each)); CK(cudaMemset(dV,0x3c,kvb_each));

    int nsplit = std::max(1, (SMs*2)/KVH);    // ~2 blocks/SM worth
    int gridK = KVH*nsplit, blkK = 256;
    int gridW = SMs*4, blkW = 256;

    for(int mode=0; mode<3; ++mode){
      const char* cfg;
      cudaStreamAttrValue av{};
      if(mode==0){
        cfg = "baseline (no policy)";
        av.accessPolicyWindow.base_ptr = dK;
        av.accessPolicyWindow.num_bytes = 0;
        av.accessPolicyWindow.hitRatio = 0.f;
        av.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
        av.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
        CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
        CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av));
      } else if(mode==1){
        cfg = "KV persisting (K only)";
        size_t win = std::min((size_t)prop.accessPolicyMaxWindowSize, kvb_each);
        size_t lim = std::min((size_t)prop.persistingL2CacheMaxSize, kvb);
        CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim));
        av.accessPolicyWindow.base_ptr = dK;
        av.accessPolicyWindow.num_bytes = win;
        av.accessPolicyWindow.hitRatio = 1.0f;
        av.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        av.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av));
      } else {
        cfg = "KV persist + W streaming";
        // same as mode 1; the weight kernel is additionally launched on a stream whose
        // window marks the (non-overlapping) weight region as streaming
        size_t win = std::min((size_t)prop.accessPolicyMaxWindowSize, kvb_each);
        size_t lim = std::min((size_t)prop.persistingL2CacheMaxSize, kvb);
        CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim));
        av.accessPolicyWindow.base_ptr = dK;
        av.accessPolicyWindow.num_bytes = win;
        av.accessPolicyWindow.hitRatio = 1.0f;
        av.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        av.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av));
      }

      // warmup: touch KV so it can be resident
      for(int i=0;i<WARM;i++){
        CK(cudaMemsetAsync(dOut,0,(size_t)KVH*GF*HD*4,s));
        weight_stream<<<gridW,blkW,0,s>>>(dW,dsink,wnel);
        flash_decode<<<gridK,blkK,0,s>>>(dK,dV,dQ,dOut,L,nsplit);
      }
      CK(cudaStreamSynchronize(s));

      // time the KV read only, with the weight stream running before it every step
      cudaEvent_t a,b,c; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventCreate(&c));
      float kvtot=0, steptot=0;
      for(int i=0;i<ITERS;i++){
        CK(cudaMemsetAsync(dOut,0,(size_t)KVH*GF*HD*4,s));
        CK(cudaEventRecord(a,s));
        weight_stream<<<gridW,blkW,0,s>>>(dW,dsink,wnel);
        CK(cudaEventRecord(b,s));
        flash_decode<<<gridK,blkK,0,s>>>(dK,dV,dQ,dOut,L,nsplit);
        CK(cudaEventRecord(c,s));
        CK(cudaEventSynchronize(c));
        float mkv,mst; CK(cudaEventElapsedTime(&mkv,b,c)); CK(cudaEventElapsedTime(&mst,a,c));
        kvtot+=mkv; steptot+=mst;
      }
      double kvms = kvtot/ITERS, stepms = steptot/ITERS;
      double gbs = kvb/(kvms*1e-3)/1e9;
      printf("%-8d %-9.1f %-26s %9.4f %9.1f %9.3f\n", L, kvb/1048576.0, cfg, kvms, gbs, stepms);
      rows.push_back({L, kvb/1048576.0, cfg, kvms, gbs, stepms});
      CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaEventDestroy(c));
    }
    // isolated KV read (no competing weight stream) = the "KV already hot in L2" bound
    {
      for(int i=0;i<WARM;i++) flash_decode<<<gridK,blkK,0,s>>>(dK,dV,dQ,dOut,L,nsplit);
      CK(cudaStreamSynchronize(s));
      cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a,s));
      for(int i=0;i<ITERS;i++) flash_decode<<<gridK,blkK,0,s>>>(dK,dV,dQ,dOut,L,nsplit);
      CK(cudaEventRecord(b,s)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms,a,b)); ms/=ITERS;
      printf("%-8d %-9.1f %-26s %9.4f %9.1f %9s\n", L, kvb/1048576.0,
             "ISOLATED (no W stream)", ms, kvb/(ms*1e-3)/1e9, "-");
      rows.push_back({L, kvb/1048576.0, "ISOLATED (no W stream)", ms, kvb/(ms*1e-3)/1e9, 0});
      CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    printf("\n");
    CK(cudaFree(dK)); CK(cudaFree(dV));
  }

  // correctness / negative control on flash_decode
  printf("=== NEGATIVE CONTROL (flash_decode sensitivity) ===\n");
  {
    int L=4096, nsplit=std::max(1,(SMs*2)/KVH);
    size_t kvb_each=(size_t)KVH*L*HD*2;
    __nv_bfloat16 *dK,*dV; CK(cudaMalloc(&dK,kvb_each)); CK(cudaMalloc(&dV,kvb_each));
    {   // realistic pseudo-random KV, not a degenerate constant fill
      std::vector<__nv_bfloat16> h(kvb_each/2);
      uint32_t st=99991u;
      for(size_t i=0;i<h.size();i++){ st=st*1664525u+1013904223u;
        h[i]=__float2bfloat16(((float)(st>>8)/16777216.0f-0.5f)*0.5f); }
      CK(cudaMemcpy(dK,h.data(),kvb_each,cudaMemcpyHostToDevice));
      for(size_t i=0;i<h.size();i++){ st=st*1664525u+1013904223u;
        h[i]=__float2bfloat16(((float)(st>>8)/16777216.0f-0.5f)*0.5f); }
      CK(cudaMemcpy(dV,h.data(),kvb_each,cudaMemcpyHostToDevice));
    }
    std::vector<float> r1(KVH*GF*HD), r2(KVH*GF*HD);
    CK(cudaMemset(dOut,0,(size_t)KVH*GF*HD*4));
    flash_decode<<<KVH*nsplit,256>>>(dK,dV,dQ,dOut,L,nsplit);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(r1.data(),dOut,r1.size()*4,cudaMemcpyDeviceToHost));
    __nv_bfloat16 bad=__float2bfloat16(7.0f);
    CK(cudaMemcpy(dV,&bad,2,cudaMemcpyHostToDevice));      // perturb V[0,0,0]
    CK(cudaMemset(dOut,0,(size_t)KVH*GF*HD*4));
    flash_decode<<<KVH*nsplit,256>>>(dK,dV,dQ,dOut,L,nsplit);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(r2.data(),dOut,r2.size()*4,cudaMemcpyDeviceToHost));
    double num=0,den=0; int nbad=0;
    for(size_t i=0;i<r1.size();i++){
      if(!std::isfinite(r1[i])||!std::isfinite(r2[i])) nbad++;
      double d=r2[i]-r1[i]; num+=d*d; den+=(double)r1[i]*r1[i]; }
    printf("r1[0..3] = %.6e %.6e %.6e %.6e\n", r1[0],r1[1],r1[2],r1[3]);
    printf("r2[0..3] = %.6e %.6e %.6e %.6e\n", r2[0],r2[1],r2[2],r2[3]);
    printf("non-finite entries: %d / %zu   num=%.6e den=%.6e\n", nbad, r1.size(), num, den);
    printf("perturbed V[0,0,0]: relL2 = %.3e (must be >> 0 for the test to be able to fail)\n",
           sqrt(num)/sqrt(den>0?den:1));
  }
  return 0;
}
