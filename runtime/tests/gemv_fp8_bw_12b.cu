/* gemv_fp8_bw_12b.cu — fp8 (e4m3 weight) M=1 DECODE GEMV bandwidth probe for Gemma-4-12B shapes
 * on sm_120 (beat12b-ctx-switch work item 2). Answers where the last ~10% of o_proj/down GEMV
 * bandwidth goes and A/Bs two candidate cheap fixes, single-block (grid = n_cu = 188, the real
 * decode partition), min-of-N timing.
 *
 * REAL 12B dims (config.json text_config): hidden 3840, intermediate 15360, 16 q-heads,
 * 8 kv-heads, head_dim 256.  Decode M=1 GEMV weight arms (W is [N,K] e4m3 row-major):
 *   qkv (fused q|k|v): here probed as q_proj  N=16*256=4096  K=3840
 *   o_proj    N=hidden=3840        K=heads*hd=4096
 *   gate/up   N=inter=15360        K=hidden=3840   (probed as a plain GEMV, not GLU, for the read)
 *   down      N=hidden=3840        K=inter=15360
 *   lm_head   N=vocab=262144       K=hidden=3840   (E5 fp8 head)
 * NOTE the campaign brief's "down K=9728" is Qwen3-4B's intermediate, not 12B's (15360).
 *
 * Arms (each a standalone kernel matching the interp's arena M=1 path):
 *   base   — d_gemv_fp8 arena path VERBATIM (uint2 8B loads, dot8_fp8).
 *   u4     — 16B (uint4) weight loads: half the load instructions, same math.
 *   bal    — base load path but BALANCED N-partition (every block gets floor/ceil rows; no idle
 *            blocks) instead of per=ceil(N/nblk) which idles the tail blocks.
 * All produce the SAME C (checked maxdiff vs base).
 *
 * Build (plain env, sm_120a):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/gemv_fp8_bw_12b.cu -o gemv_fp8_bw_12b
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh" /* bf16v8, ld_glob8, ld_smem8, dot8; pulls op_attention.cuh */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

#define GV_STEP (32u*8u)
#define WARPS   8u

static uint32_t rng_s = 0x1234567u;
static float rnd(){ rng_s^=rng_s<<13; rng_s^=rng_s>>17; rng_s^=rng_s<<5;
    return (float)((int32_t)rng_s)/2147483648.0f; }

/* fp8 dequant-fma of one lane's 8 e4m3 (uint2) against 8 bf16 x from smem. Copy of dot8_fp8. */
__device__ __forceinline__ float p_dot8(const uint2& w8, const bf16* xs, float acc){
    const uint16_t* wp=(const uint16_t*)&w8;
    bf16v8 xv = ld_smem8(xs);
#pragma unroll
    for (int j=0;j<4;j++){
        __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j],__NV_E4M3);
        float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
        acc=fmaf(f.x,__bfloat162float(xv.x[2*j]),acc);
        acc=fmaf(f.y,__bfloat162float(xv.x[2*j+1]),acc);
    }
    return acc;
}
/* ---- base: verbatim d_gemv_fp8 arena M=1 path ---- */
__global__ void __launch_bounds__(256,1)
k_base(bf16* C, const bf16* x, const uint8_t* W, const float* scale, unsigned N, unsigned K){
    extern __shared__ bf16 xs[];
    for (unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
    const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
    const unsigned per=(N+gridDim.x-1)/gridDim.x;
    const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
    for (unsigned n=n0+warp;n<n1;n+=WARPS){
        const uint8_t* wrow=W+(size_t)n*K;
        float acc=0.0f;
        for (unsigned c=0;c<nchunk;c+=8){
            uint2 wv[8]; unsigned kk[8];
#pragma unroll
            for (int u=0;u<8;u++){ unsigned k=(c+u)*GV_STEP+lane*8; kk[u]=k;
                wv[u]=(k<K)?*(const uint2*)(wrow+k):make_uint2(0u,0u); }
#pragma unroll
            for (int u=0;u<8;u++){ if(kk[u]>=K)continue; acc=p_dot8(wv[u],xs+kk[u],acc); }
        }
        float t=warp_sum32(acc);
        if(lane==0) C[n]=__float2bfloat16(t*scale[n]);
    }
}

/* ---- u4: 16B (uint4) weight loads, lane owns 16 contiguous e4m3, warp-pass = 512 K ---- */
__global__ void __launch_bounds__(256,1)
k_u4(bf16* C, const bf16* x, const uint8_t* W, const float* scale, unsigned N, unsigned K){
    extern __shared__ bf16 xs[];
    for (unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
    const unsigned STEP16=32u*16u;               /* 512 K per warp-pass */
    const unsigned nchunk=(K+STEP16-1)/STEP16;
    const unsigned per=(N+gridDim.x-1)/gridDim.x;
    const unsigned n0=blockIdx.x*per, n1=(n0+per<N)?(n0+per):N;
    for (unsigned n=n0+warp;n<n1;n+=WARPS){
        const uint8_t* wrow=W+(size_t)n*K;
        float acc=0.0f;
        for (unsigned c=0;c<nchunk;c+=4){        /* 4-deep unroll of uint4 = 2KB/lane inflight */
            uint4 wv[4]; unsigned kk[4];
#pragma unroll
            for (int u=0;u<4;u++){ unsigned k=(c+u)*STEP16+lane*16; kk[u]=k;
                wv[u]=(k<K)?*(const uint4*)(wrow+k):make_uint4(0u,0u,0u,0u); }
#pragma unroll
            for (int u=0;u<4;u++){ if(kk[u]>=K)continue;
                const uint16_t* wp=(const uint16_t*)&wv[u];
                bf16v8 xa=ld_smem8(xs+kk[u]); bf16v8 xb=ld_smem8(xs+kk[u]+8);
#pragma unroll
                for (int j=0;j<8;j++){
                    __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j],__NV_E4M3);
                    float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
                    const bf16v8& xr = (j<4)?xa:xb; int jj=(j&3);
                    acc=fmaf(f.x,__bfloat162float(xr.x[2*jj]),acc);
                    acc=fmaf(f.y,__bfloat162float(xr.x[2*jj+1]),acc);
                }
            }
        }
        float t=warp_sum32(acc);
        if(lane==0) C[n]=__float2bfloat16(t*scale[n]);
    }
}

/* ---- bal: base math, BALANCED N-partition (no idle tail blocks) ---- */
__global__ void __launch_bounds__(256,1)
k_bal(bf16* C, const bf16* x, const uint8_t* W, const float* scale, unsigned N, unsigned K){
    extern __shared__ bf16 xs[];
    for (unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31u, warp=threadIdx.x>>5;
    const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
    const unsigned base=N/gridDim.x, rem=N%gridDim.x;      /* first `rem` blocks get base+1 rows */
    const unsigned n0 = blockIdx.x*base + (blockIdx.x<rem?blockIdx.x:rem);
    const unsigned cnt = base + (blockIdx.x<rem?1u:0u);
    const unsigned n1 = n0+cnt;
    for (unsigned n=n0+warp;n<n1;n+=WARPS){
        const uint8_t* wrow=W+(size_t)n*K;
        float acc=0.0f;
        for (unsigned c=0;c<nchunk;c+=8){
            uint2 wv[8]; unsigned kk[8];
#pragma unroll
            for (int u=0;u<8;u++){ unsigned k=(c+u)*GV_STEP+lane*8; kk[u]=k;
                wv[u]=(k<K)?*(const uint2*)(wrow+k):make_uint2(0u,0u); }
#pragma unroll
            for (int u=0;u<8;u++){ if(kk[u]>=K)continue; acc=p_dot8(wv[u],xs+kk[u],acc); }
        }
        float t=warp_sum32(acc);
        if(lane==0) C[n]=__float2bfloat16(t*scale[n]);
    }
}

struct Shape { const char* name; unsigned N, K; };

/* L2-flush buffer (argv[2]=1): each in-network decode reads every weight exactly ONCE per token,
 * so the realistic bandwidth is a COLD HBM read. Without a flush the probe reads the (<=96 MB L2)
 * weight hot and reports L2 bandwidth, not the HBM ceiling the campaign's 466-823 GB/s measured. */
static uint8_t* g_flush = nullptr;
static size_t g_flush_bytes = 0;

int main(int argc, char** argv){
    int iters = argc>1?atoi(argv[1]):200;
    int flush = argc>2?atoi(argv[2]):0;
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    const unsigned NB=(unsigned)p.multiProcessorCount; /* 188 = n_cu */
    if (flush){ g_flush_bytes=(size_t)256<<20; CK(cudaMalloc(&g_flush,g_flush_bytes)); }
    printf("# device %s SMs %u iters %d flush=%d (grid = n_cu = %u, block 256, M=1 arena path)\n",
           p.name, NB, iters, flush, NB);
    printf("# fp8 weight bytes = N*K (e4m3). GBps = bytes / min_ms. %s\n\n",
           flush?"L2-FLUSHED (cold HBM read)":"L2-hot");

    const Shape shapes[] = {
        {"q_proj ", 4096, 3840},
        {"o_proj ", 3840, 4096},
        {"gate/up", 15360, 3840},
        {"down   ", 3840, 15360},
        {"lm_head", 262144, 3840},
    };
    printf("%-8s %7s %7s | %5s %10s %10s | %10s %10s %10s | %s\n",
           "arm","N","K","rows/b","base_GBps","","ms_base","ms_u4","ms_bal","maxdiff");

    for (auto& s : shapes){
        const size_t wbytes=(size_t)s.N*s.K;
        std::vector<uint8_t> hW(wbytes);
        for (size_t i=0;i<wbytes;i++){ float v=rnd()*0.3f; hW[i]=(uint8_t)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);}
        std::vector<bf16> hx(s.K); for (unsigned i=0;i<s.K;i++) hx[i]=__float2bfloat16(rnd());
        std::vector<float> hs(s.N); for (unsigned i=0;i<s.N;i++) hs[i]=0.01f+0.001f*(i%7);
        uint8_t* dW; bf16* dx; float* dsc; bf16* dC;
        CK(cudaMalloc(&dW,wbytes)); CK(cudaMalloc(&dx,s.K*2)); CK(cudaMalloc(&dsc,s.N*4));
        CK(cudaMalloc(&dC,s.N*2));
        CK(cudaMemcpy(dW,hW.data(),wbytes,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx,hx.data(),s.K*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc,hs.data(),s.N*4,cudaMemcpyHostToDevice));
        const size_t smem=(size_t)s.K*2;
        CK(cudaFuncSetAttribute(k_base,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        CK(cudaFuncSetAttribute(k_u4,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        CK(cudaFuncSetAttribute(k_bal, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

        auto run=[&](int which)->double{
            double best=1e30;
            for (int it=0;it<iters+3;it++){
                cudaMemset(dC,0,s.N*2);
                if (g_flush) cudaMemset(g_flush, (int)it, g_flush_bytes); /* evict weight from L2 */
                cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a);
                if (which==0) k_base<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
                else if (which==1) k_u4<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
                else k_bal<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms,a,b);
                cudaEventDestroy(a); cudaEventDestroy(b);
                CK(cudaGetLastError());
                if (it>=3 && ms<best) best=ms;
            }
            return best;
        };
        /* correctness: capture base C, compare u4/bal */
        auto capture=[&](int which,std::vector<float>& o){
            cudaMemset(dC,0,s.N*2);
            if (which==0) k_base<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
            else if (which==1) k_u4<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
            else k_bal<<<NB,256,smem>>>(dC,dx,dW,dsc,s.N,s.K);
            CK(cudaDeviceSynchronize());
            std::vector<bf16> hc(s.N); CK(cudaMemcpy(hc.data(),dC,s.N*2,cudaMemcpyDeviceToHost));
            o.resize(s.N); for (unsigned i=0;i<s.N;i++) o[i]=__bfloat162float(hc[i]);
        };
        std::vector<float> cb,cu,cl; capture(0,cb); capture(1,cu); capture(2,cl);
        double md=0; for (unsigned i=0;i<s.N;i++){ md=fmax(md,fabs(cu[i]-cb[i])); md=fmax(md,fabs(cl[i]-cb[i])); }
        double mb=run(0), mu=run(1), ml=run(2);
        double gb=(double)wbytes/(mb*1e-3)/1e9;
        unsigned per=(s.N+NB-1)/NB;
        printf("%-8s %7u %7u | %5u %10.1f %10s | %10.4f %10.4f %10.4f | %.4f\n",
               s.name,s.N,s.K,per,gb,"",mb,mu,ml,md);
        printf("         u4 GBps=%.1f  bal GBps=%.1f\n",
               (double)wbytes/(mu*1e-3)/1e9, (double)wbytes/(ml*1e-3)/1e9);
        cudaFree(dW);cudaFree(dx);cudaFree(dsc);cudaFree(dC);
    }
    return 0;
}
