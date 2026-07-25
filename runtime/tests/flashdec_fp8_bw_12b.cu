/* flashdec_fp8_bw_12b.cu — flash-decode OCCUPANCY microbench for the Gemma-4-12B hd512
 * FULL-attention layers on sm_120 (beat12b-fp8-margin campaign).
 *
 * 12B full-layer geometry (config.json): n_head=16, n_GLOBAL_kv_head=1, D=512 -> GQA=16.
 * ONE kv head serves all 16 query heads: with the shipped GF=2 fusion every KV byte is
 * demanded 8x (n_grp=8 groups all read kv_head 0); L2 (~96MB) absorbs part of that at
 * short ctx but 128k fp8 K+V = 134MB streams past it.
 *
 * SHIPPED TODAY (decode program, ctx>8k): GF=2, nsplit=24 (emitter: (188*2)/16 heads),
 * n_work = n_grp(8) * 24 = 192 items on 188 SMs -> RAGGED: 4 blocks run 2 items, 184 run
 * 1, and FLASH_MERGE waits for the 2x stragglers. Levers swept here, single-block level:
 *   L1 nsplit ALIGNMENT: ns=23 (184 items, 1/block) or ns=47 (376 = exactly 2/block).
 *   L2 GF up: GF=4 (4x re-read) / GF=8 (2x re-read) cut issued KV bytes; need larger ns
 *      to refill (n_grp = 16/GF). Register cost oacc[GF][8] — ptxas decides.
 * Merge cost is TIMED TOO (d_flash_merge, 16 blocks): it scales with nsplit and took back
 * the flash win in prior over-split attempts (31B ns64).
 *
 * bytes_issued = n_grp * ctx * D * elem * 2(K,V)  (demand incl. the GQA re-reads)
 * bytes_phys   = NKV(=1) * ctx * D * elem * 2     (distinct HBM bytes)
 *
 * Build (plain env, sm_120a):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/flashdec_fp8_bw_12b.cu -o flashdec_bw_12b
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_decode / d_flash_merge) */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }

/* Standalone launchers pinned to 1 block/SM, matching the megakernel's __launch_bounds__(256,1). */
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_flash_bf16(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
             const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
             float scale, unsigned nsplit){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,false>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0u/*window=full*/,scale,nsplit,
                               0xFFFFFFFFu,blockIdx.x,gridDim.x,arena,0u);
}
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_flash_fp8(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
            const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
            float scale, unsigned nsplit, const float* ks, const float* vs){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,true>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0u/*window=full*/,scale,nsplit,
                              0xFFFFFFFFu,blockIdx.x,gridDim.x,arena,0u,ks,vs);
}
template<int D>
__global__ void k_merge(bf16* O, const float* Opart, const float* mlpart,
                        unsigned nb, unsigned nh, unsigned nsplit){
    d_flash_merge<D>(O,Opart,mlpart,nb,nh,nsplit,blockIdx.x,gridDim.x);
}

static const unsigned NH = 16, NKV = 1, D = 512; /* Gemma-4-12B FULL-attn: gqa=16, ONE kv head */
static const unsigned MAXCTX = 131072u;
static std::vector<float> g_last_out; /* last run()'s merged output, f32 (ref capture) */

struct Res { double ms_flash, ms_merge; double gbps_issued, gbps_phys; float maxdiff; };

/* Device buffers shared by every config (biggest ctx): built once. */
static void *g_dK8, *g_dV8, *g_dK16, *g_dV16; static float *g_dKs, *g_dVs;
static bf16 *g_dQ; static int *g_dL;
static void setup(){
    const size_t n = (size_t)NKV*MAXCTX*D;
    std::vector<uint8_t> k8(n), v8(n); std::vector<bf16> k16(n), v16(n);
    for (size_t i=0;i<n;i++){ float a=rnd()*0.5f, b=rnd()*0.5f;
        __nv_fp8_e4m3 fa(a), fb(b); k8[i]=*(uint8_t*)&fa; v8[i]=*(uint8_t*)&fb;
        k16[i]=__float2bfloat16(a); v16[i]=__float2bfloat16(b); }
    CK(cudaMalloc(&g_dK8,n)); CK(cudaMalloc(&g_dV8,n));
    CK(cudaMemcpy(g_dK8,k8.data(),n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_dV8,v8.data(),n,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_dK16,n*2)); CK(cudaMalloc(&g_dV16,n*2));
    CK(cudaMemcpy(g_dK16,k16.data(),n*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_dV16,v16.data(),n*2,cudaMemcpyHostToDevice));
    const size_t nrows=(size_t)NKV*MAXCTX;
    std::vector<float> sc(nrows,1.0f);
    CK(cudaMalloc(&g_dKs,nrows*4)); CK(cudaMalloc(&g_dVs,nrows*4));
    CK(cudaMemcpy(g_dKs,sc.data(),nrows*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_dVs,sc.data(),nrows*4,cudaMemcpyHostToDevice));
    std::vector<bf16> q((size_t)NH*D); for (auto& x:q) x=__float2bfloat16(rnd()*0.5f);
    CK(cudaMalloc(&g_dQ,q.size()*2)); CK(cudaMemcpy(g_dQ,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_dL,4));
}

template<int GF>
static Res run(bool fp8, unsigned ctx, unsigned nsplit, int iters, const std::vector<float>& ref){
    const unsigned nb=1, kvs=MAXCTX; /* stride fixed at max like the real cache */
    const unsigned n_grp = NH/GF;
    const unsigned n_work = nb*n_grp*nsplit;
    const float scale = 1.f/sqrtf((float)D);
    int len=(int)ctx; CK(cudaMemcpy(g_dL,&len,4,cudaMemcpyHostToDevice));

    float *dOp=nullptr,*dMl=nullptr; bf16* dO=nullptr;
    CK(cudaMalloc(&dOp,(size_t)nb*NH*nsplit*D*4)); CK(cudaMalloc(&dMl,(size_t)nb*NH*nsplit*2*4));
    CK(cudaMalloc(&dO,(size_t)nb*NH*D*2));

    const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    auto lflash=[&](){
        if (fp8) k_flash_fp8<D,GF><<<n_work,256,smem>>>(dOp,dMl,g_dQ,(bf16*)g_dK8,(bf16*)g_dV8,g_dL,nb,NH,NKV,kvs,scale,nsplit,g_dKs,g_dVs);
        else     k_flash_bf16<D,GF><<<n_work,256,smem>>>(dOp,dMl,g_dQ,(bf16*)g_dK16,(bf16*)g_dV16,g_dL,nb,NH,NKV,kvs,scale,nsplit);
    };
    auto lmerge=[&](){ k_merge<D><<<nb*NH,256>>>(dO,dOp,dMl,nb,NH,nsplit); };
    if (fp8) CK(cudaFuncSetAttribute(k_flash_fp8<D,GF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    else     CK(cudaFuncSetAttribute(k_flash_bf16<D,GF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    for (int w=0;w<3;w++){ lflash(); lmerge(); }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t a,b,c; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventCreate(&c));
    double bf=1e30, bm=1e30;
    for (int it=0; it<iters; it++){
        CK(cudaEventRecord(a)); lflash(); CK(cudaEventRecord(b)); lmerge(); CK(cudaEventRecord(c));
        CK(cudaEventSynchronize(c));
        float mf=0,mm=0; CK(cudaEventElapsedTime(&mf,a,b)); CK(cudaEventElapsedTime(&mm,b,c));
        if (mf<bf) bf=mf; if (mm<bm) bm=mm;
    }
    cudaEventDestroy(a); cudaEventDestroy(b); cudaEventDestroy(c);

    /* Output check vs the reference config (guards this HARNESS's mapping, not the kernel). */
    std::vector<bf16> o((size_t)nb*NH*D); CK(cudaMemcpy(o.data(),dO,o.size()*2,cudaMemcpyDeviceToHost));
    float md=0;
    for (size_t i=0;i<ref.size();i++){
        float d=fabsf(__bfloat162float(o[i])-ref[i]); if (d>md) md=d; }
    g_last_out.resize(o.size());
    for (size_t i=0;i<o.size();i++) g_last_out[i]=__bfloat162float(o[i]);
    cudaFree(dOp); cudaFree(dMl); cudaFree(dO);

    const double elem = fp8?1.0:2.0;
    const double bytes_issued = (double)n_grp*ctx*D*elem*2.0;
    const double bytes_phys   = (double)NKV  *ctx*D*elem*2.0;
    Res r; r.ms_flash=bf; r.ms_merge=bm;
    r.gbps_issued=bytes_issued/(bf*1e-3)/1e9; r.gbps_phys=bytes_phys/(bf*1e-3)/1e9;
    r.maxdiff=md;
    return r;
}

int main(int argc, char** argv){
    int iters = argc>1?atoi(argv[1]):40;
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("# device %s, SMs %d, iters %d\n", p.name, p.multiProcessorCount, iters);
    printf("# geom: n_head=%u n_kv_head=%u gqa=%u D=%u (Gemma-4-12B FULL-attn, one kv head)\n",
           NH,NKV,NH/NKV,D);
    printf("# shipped today: GF=2 ns=24 -> n_work=192 on %d SMs (ragged)\n\n", p.multiProcessorCount);
    setup();

    struct Cfg { int gf; unsigned ns; };
    const Cfg cfgs[] = {
        {2,12},{2,16},{2,23},{2,24},{2,32},{2,47},{2,64},{2,94},
        {4,24},{4,47},{4,48},{4,64},{4,94},
        {8,47},{8,94},{8,96},{8,128},
    };
    /* {2,94} (ctx-switch campaign): the ONE-PACKET candidate — ns94 gives GF2 752 items
     * (4/block, aligned) and GF8 188 (1/block); if GF2/ns94 ~= GF2/ns47 at short ctx, a
     * single pkt serves both objects and switching is purely a function choice. */
    const unsigned ctxs[]={1024u,4096u,16384u,32768u,65536u,98304u,131072u};
    printf("%-4s %-3s %-7s %-7s %-8s | %9s %9s %9s | %10s %10s | %s\n",
           "dt","GF","ctx","nsplit","n_work","ms_flash","ms_merge","ms_sum","GBps_iss","GBps_phys","maxdiff_vs_ref");
    for (int fp8i=1; fp8i>=0; fp8i--){
      bool fp8 = fp8i==1;
      for (unsigned ci=0; ci<sizeof(ctxs)/sizeof(ctxs[0]); ci++){
        unsigned ctx=ctxs[ci];
        std::vector<float> ref; /* GF=2 ns=24 = shipped config is the ref (first with ns 24) */
        /* run ref first */
        Res rr = run<2>(fp8,ctx,24,iters,std::vector<float>());
        ref = g_last_out;
        for (auto& cfg : cfgs){
          Res r;
          if      (cfg.gf==2) r=run<2>(fp8,ctx,cfg.ns,iters,ref);
          else if (cfg.gf==4) r=run<4>(fp8,ctx,cfg.ns,iters,ref);
          else                r=run<8>(fp8,ctx,cfg.ns,iters,ref);
          unsigned n_work=(NH/cfg.gf)*cfg.ns;
          printf("%-4s %-3d %-7u %-7u %-8u | %9.4f %9.4f %9.4f | %10.1f %10.1f | %.4f\n",
                 fp8?"fp8":"bf16",cfg.gf,ctx,cfg.ns,n_work,
                 r.ms_flash,r.ms_merge,r.ms_flash+r.ms_merge,r.gbps_issued,r.gbps_phys,r.maxdiff);
        }
        printf("\n");
      }
    }
    return 0;
}
