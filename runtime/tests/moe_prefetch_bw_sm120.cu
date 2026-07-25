/* moe_prefetch_bw_sm120.cu — GO/NO-GO microbench for ROUTER-HOIST EXPERT PREFETCH
 * (26B-A4B MoE DECODE, sm_120).
 *
 * Question: the compiler hoists the MoE router before the dense MLP, so the 8 routed
 * experts' IDs are known one dense-MLP-duration before their gate_up weights are read
 * by the expert-GLU GEMV. Does a `prefetch.global.L2` packet fired in that window net
 * real time, or does it just steal HBM bandwidth from the (BW-bound) dense stream?
 *
 * Measures, at real decode geometry (H=2816, I=704, k=8, E=128, nrow=1), per dtype
 * (bf16 / fp8 per-channel — the two shipping 26B decode expert paths):
 *   glu_cold        flush L2 -> d_moe_expert_glu_gemma[_fp8]      (today's baseline)
 *   glu_warm        same, second back-to-back run                  (L2-resident bound)
 *   pf_idle + glu   flush -> prefetch alone -> GLU                 (does prefetch land at all?)
 *   dense_base      flush -> dense GEMV (~100MB bf16 / ~50MB fp8 window model)
 *   dense+pf, glu   flush -> fused kernel (PFB blocks prefetch first, all do dense) -> GLU
 * GO iff net = (glu_cold - glu_after_pf) - (dense_pf - dense_base) > ~1% of the MoE
 * step (glu_cold + down_cold) AND the dense slowdown is ~noise.
 *
 * Prefetch: prefetch.global.L2 (optionally ::evict_last), 128B lines, GLU gate_up rows
 * ONLY (60.5 MiB bf16 / 30.3 MiB fp8 for 8 experts vs 128 MiB L2), block-strided.
 *
 * Same launch shape as the megakernel: grid=SMs(188), 256 threads.
 * Build: nvcc -arch=sm_120a -O3 -DPLOW_NV_GEMMA=1 -I runtime/common -I runtime/nvidia
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "sm120_common.cuh"
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "dev_isa.h"
#include "op_moe.cuh"

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }
static void seed(uint32_t s) { rng_s = s ? s : 1; }
static float bf16_rt(float x) { return __bfloat162float(__float2bfloat16(x)); }
static std::vector<float> gen_bf16(size_t n, float amp) {
    std::vector<float> v(n); for (size_t i=0;i<n;i++) v[i]=bf16_rt(rnd()*amp); return v;
}
template<class T> static T* to_dev(const std::vector<T>& h){ T* d=nullptr; CK(cudaMalloc(&d,h.size()*sizeof(T)));
    CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return d; }
static bf16* to_dev_bf(const std::vector<float>& h){ std::vector<bf16> t(h.size());
    for(size_t i=0;i<h.size();i++) t[i]=__float2bfloat16(h[i]); return to_dev(t); }

/* per-output-channel e4m3 quant (matches the shipping 26B fp8 expert layout). */
static void quant_rows(const std::vector<float>& v, size_t rows, size_t width,
                       std::vector<uint8_t>& q, std::vector<float>& scale) {
    q.resize(rows*width); scale.resize(rows);
    for (size_t r=0;r<rows;r++){ float amax=0; for(size_t c=0;c<width;c++) amax=fmaxf(amax,fabsf(v[r*width+c]));
        float s = amax>0 ? amax/448.0f : 1.0f; scale[r]=s;
        for(size_t c=0;c<width;c++){ __nv_fp8_e4m3 e(v[r*width+c]/s); q[r*width+c]=e.__x; } }
}

/* ---- L2 prefetch of the k routed experts' gate_up rows (the candidate packet body) ----
 * Block-strided flat sweep over k * (bytes_per_exp/128) cache lines; consecutive threads
 * touch consecutive 128B lines of one expert. Fire-and-forget: no registers, no result. */
__device__ __forceinline__ void moe_pf_glu(const unsigned char* __restrict__ table,
        const unsigned long long* __restrict__ ewt, unsigned k, unsigned n_exp,
        size_t bytes_per_exp, unsigned slice, unsigned nblk, int evict_last) {
    const size_t lines = bytes_per_exp >> 7;
    const size_t total = (size_t)k * lines;
    const size_t stride = (size_t)nblk * blockDim.x;
    for (size_t i = (size_t)slice * blockDim.x + threadIdx.x; i < total; i += stride) {
        const unsigned slot = (unsigned)(i / lines);
        const size_t off = (i - (size_t)slot * lines) << 7;
        const unsigned eid = plow_moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long wb = ewt[(size_t)eid * 2 + 0];
        if (wb == 0ull) continue;
        const char* p = (const char*)(size_t)wb + off;
        if (evict_last) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p));
        else            asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
    }
}

__global__ void k_prefetch(const unsigned char* table, const unsigned long long* ewt,
                           unsigned k, unsigned E, size_t bytes_per_exp, int evict_last) {
    moe_pf_glu(table, ewt, k, E, bytes_per_exp, blockIdx.x, gridDim.x, evict_last);
}

/* ---- dense window model: warp-per-row bf16 GEMV streaming an N x H weight matrix.
 * pfb > 0: blocks [0,pfb) first issue their slice of the expert prefetch, then join the
 * dense sweep — models the hoisted prefetch packet running concurrently inside the
 * megakernel's dense phase (same grid, dense work total unchanged). */
__global__ void k_dense(bf16* __restrict__ y, const bf16* __restrict__ W,
                        const bf16* __restrict__ x, unsigned N, unsigned H,
                        const unsigned char* table, const unsigned long long* ewt,
                        unsigned k, unsigned E, size_t pf_bytes, unsigned pfb, int evict_last) {
    if (pfb && blockIdx.x < pfb)
        moe_pf_glu(table, ewt, k, E, pf_bytes, blockIdx.x, pfb, evict_last);
    const unsigned lane = threadIdx.x & (PLOW_NV_WARP - 1u);
    const unsigned gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned nwarp = (gridDim.x * blockDim.x) >> 5;
    for (unsigned n = gwarp; n < N; n += nwarp) {
        const float v = plow_warp_dot_bf16(x, W + (size_t)n * H, H, lane);
        if (lane == 0) y[n] = __float2bfloat16(v);
    }
}

/* ---- L2 flush: stream a buffer larger than L2 through it. ---- */
__global__ void k_flush(const float4* __restrict__ buf, size_t n4, float* __restrict__ sink) {
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = buf[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1e30f) sink[blockIdx.x] = acc;   /* never true; defeats DCE */
}

/* ---- decode expert op wrappers (exact megakernel bodies) ---- */
__global__ void k_glu(bf16* fu, const bf16* x, const unsigned char* table,
                      const uint64_t* ewt, unsigned k, unsigned I, unsigned H, unsigned E){
    extern __shared__ bf16 sm[];
    d_moe_expert_glu_gemma(fu, x, table, (const unsigned long long*)ewt, k, I, H, E,
                           blockIdx.x, gridDim.x, 1u, sm);
}
__global__ void k_glu_fp8(bf16* fu, const bf16* x, const unsigned char* table,
                          const uint64_t* ewt, const uint64_t* est,
                          unsigned k, unsigned I, unsigned H, unsigned E){
    extern __shared__ bf16 sm[];
    d_moe_expert_glu_gemma_fp8(fu, x, table, (const unsigned long long*)ewt,
                               (const unsigned long long*)est, k, I, H, E,
                               blockIdx.x, gridDim.x, 1u, sm);
}
__global__ void k_down(float* part, const bf16* fu, const unsigned char* table,
                       const uint64_t* ewt, unsigned k, unsigned H, unsigned I, unsigned E){
    d_moe_expert_down_gemma(part, fu, table, (const unsigned long long*)ewt, k, H, I, E,
                            blockIdx.x, gridDim.x, 1u);
}
__global__ void k_down_fp8(float* part, const bf16* fu, const unsigned char* table,
                           const uint64_t* ewt, const uint64_t* est,
                           unsigned k, unsigned H, unsigned I, unsigned E){
    d_moe_expert_down_gemma_fp8(part, fu, table, (const unsigned long long*)ewt,
                                (const unsigned long long*)est, k, H, I, E,
                                blockIdx.x, gridDim.x, 1u);
}

static double reL2(const std::vector<float>& a, const std::vector<float>& b){
    double num=0,den=0; for(size_t i=0;i<a.size();i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return den>0? sqrt(num/den) : 0.0;
}

int main(int argc, char** argv){
    int dev=0; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p,dev));
    const unsigned H=2816, E=128, K8=8, I=704;
    const int GRID=p.multiProcessorCount, ITERS=30, WARM=5;
    printf("device: %s sm_%d%d SMs=%d L2=%.0fMiB | H=%u I=%u k=%u E=%u grid=%d iters=%d\n",
           p.name,p.major,p.minor,p.multiProcessorCount,p.l2CacheSize/1048576.0,H,I,K8,E,GRID,ITERS);

    /* one expert's worth of random weights, replicated device-side across E experts
     * (distinct addresses — identical values; timing- and sanity-equivalent). */
    seed(1234);
    const size_t GU_ELEM=(size_t)2*I*H, DW_ELEM=(size_t)H*I;
    std::vector<float> gu=gen_bf16(GU_ELEM,0.05f), dw=gen_bf16(DW_ELEM,0.05f);
    std::vector<uint8_t> gu8,dw8; std::vector<float> guS,dwS;
    quant_rows(gu,(size_t)2*I,H,gu8,guS);
    quant_rows(dw,(size_t)H,I,dw8,dwS);

    bf16 *dGU,*dDW; uint8_t *dGU8,*dDW8;
    CK(cudaMalloc(&dGU,(size_t)E*GU_ELEM*sizeof(bf16)));
    CK(cudaMalloc(&dDW,(size_t)E*DW_ELEM*sizeof(bf16)));
    CK(cudaMalloc(&dGU8,(size_t)E*GU_ELEM));
    CK(cudaMalloc(&dDW8,(size_t)E*DW_ELEM));
    { bf16* t0=to_dev_bf(gu); bf16* t1=to_dev_bf(dw); uint8_t* t2=to_dev(gu8); uint8_t* t3=to_dev(dw8);
      for(unsigned e=0;e<E;e++){
        CK(cudaMemcpy(dGU+(size_t)e*GU_ELEM,t0,GU_ELEM*sizeof(bf16),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dDW+(size_t)e*DW_ELEM,t1,DW_ELEM*sizeof(bf16),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dGU8+(size_t)e*GU_ELEM,t2,GU_ELEM,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dDW8+(size_t)e*DW_ELEM,t3,DW_ELEM,cudaMemcpyDeviceToDevice));
      }
      cudaFree(t0);cudaFree(t1);cudaFree(t2);cudaFree(t3); }
    float* dGUs=to_dev(guS); float* dDWs=to_dev(dwS);   /* shared scale rows (values identical) */

    /* decode ewt/est tables: [E][2] = {gate_up, down} / {glu scales [2I], down scales [H]} */
    std::vector<uint64_t> ewt((size_t)E*2), ewt8((size_t)E*2), est8((size_t)E*2);
    for(unsigned e=0;e<E;e++){
        ewt [e*2+0]=(uint64_t)(dGU +(size_t)e*GU_ELEM); ewt [e*2+1]=(uint64_t)(dDW +(size_t)e*DW_ELEM);
        ewt8[e*2+0]=(uint64_t)(dGU8+(size_t)e*GU_ELEM); ewt8[e*2+1]=(uint64_t)(dDW8+(size_t)e*DW_ELEM);
        est8[e*2+0]=(uint64_t)dGUs;                     est8[e*2+1]=(uint64_t)dDWs;
    }
    uint64_t* dEwt=to_dev(ewt); uint64_t* dEwt8=to_dev(ewt8); uint64_t* dEst8=to_dev(est8);

    /* routing tables: NTAB disjoint sets of 8 distinct experts, rotated per iteration so the
     * previous iteration's GLU can never pre-warm this iteration's experts (Blackwell L2
     * streaming-insertion heuristics let reused lines survive a read-only flush). This also
     * models decode reality: the routed set changes every token. gate=1/8. */
    const unsigned NTAB=8;
    unsigned char* dTabs[NTAB];
    for(unsigned t=0;t<NTAB;t++){
        std::vector<unsigned char> tab((size_t)K8*8);
        for(unsigned s=0;s<K8;s++){ *(unsigned*)(tab.data()+(size_t)s*8)=t*16u+s*2u+(t&1u);
            *(float*)(tab.data()+(size_t)s*8+4)=0.125f; }
        dTabs[t]=to_dev(tab);
    }

    /* activation row, fu, part */
    seed(77); std::vector<float> xh=gen_bf16(H,1.0f);
    bf16* dX=to_dev_bf(xh);
    bf16* dFu=nullptr;  CK(cudaMalloc(&dFu,(size_t)K8*I*sizeof(bf16)));
    float* dPart=nullptr; CK(cudaMalloc(&dPart,(size_t)K8*H*sizeof(float)));

    /* dense window: N_BF16 x H bf16 ~= 100 MiB (bf16 model); fp8 model halves the window */
    const unsigned N_BF16 = (unsigned)((100.0*1048576.0)/((double)H*2.0));   /* ~18617 rows */
    const unsigned N_FP8  = N_BF16/2;
    std::vector<float> wd=gen_bf16((size_t)N_BF16*H,0.05f);
    bf16* dDenseW=to_dev_bf(wd); wd.clear(); wd.shrink_to_fit();
    bf16* dY=nullptr; CK(cudaMalloc(&dY,(size_t)N_BF16*sizeof(bf16)));

    /* flush buffer: 3x L2 */
    const size_t FLUSH_B = (size_t)p.l2CacheSize*3, FLUSH_N4 = FLUSH_B/16;
    float4* dFlush=nullptr; CK(cudaMalloc(&dFlush,FLUSH_N4*16)); CK(cudaMemset(dFlush,1,FLUSH_N4*16));
    float* dSink=nullptr; CK(cudaMalloc(&dSink,GRID*sizeof(float)));

    const size_t smem=(size_t)H*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_glu,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_glu_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    /* sanity: fp8 GLU vs bf16 GLU relL2 (wiring check; expect ~1e-2 for e4m3) */
    {
        k_glu<<<GRID,256,smem>>>(dFu,dX,dTabs[0],dEwt,K8,I,H,E); CK(cudaDeviceSynchronize());
        std::vector<bf16> t((size_t)K8*I); std::vector<float> a(t.size()),b(t.size());
        CK(cudaMemcpy(t.data(),dFu,t.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<t.size();i++) a[i]=__bfloat162float(t[i]);
        k_glu_fp8<<<GRID,256,smem>>>(dFu,dX,dTabs[0],dEwt8,dEst8,K8,I,H,E); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(t.data(),dFu,t.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<t.size();i++) b[i]=__bfloat162float(t[i]);
        printf("sanity: GLU fp8-vs-bf16 relL2 = %.2e\n", reL2(b,a));
    }

    cudaEvent_t ev[16]; for(int i=0;i<16;i++) CK(cudaEventCreate(&ev[i]));
    /* flush = 384MiB memset (write-allocate) + 384MiB read stream: defeats both LRU and
     * streaming-insertion retention. */
    auto flush=[&]{ CK(cudaMemsetAsync(dFlush,0x3f,FLUSH_N4*16));
                    k_flush<<<GRID,256>>>(dFlush,FLUSH_N4,dSink); };

    /* timed sequence runner: per-iteration the routing table rotates through the NTAB
     * disjoint expert sets; fns between consecutive events; returns per-phase means (ms). */
    auto run_seq=[&](std::vector<std::pair<const char*,std::function<void(const unsigned char*)>>> steps,
                     std::vector<double>& out){
        out.assign(steps.size(),0.0);
        for(int it=-WARM;it<ITERS;it++){
            const unsigned char* tab=dTabs[(unsigned)(it+WARM)%NTAB];
            flush();
            for(size_t s=0;s<steps.size();s++){
                CK(cudaEventRecord(ev[2*s]));
                steps[s].second(tab);
                CK(cudaEventRecord(ev[2*s+1]));
            }
            CK(cudaDeviceSynchronize());
            if(it>=0) for(size_t s=0;s<steps.size();s++){
                float ms=0; CK(cudaEventElapsedTime(&ms,ev[2*s],ev[2*s+1])); out[s]+=ms; }
        }
        for(auto& v:out) v/=ITERS;
    };

    const unsigned PFBS[]={4,16,64,188};
    struct Cfg { const char* name; bool fp8; size_t pf_bytes; unsigned Ndense; };
    const Cfg cfgs[2]={
        {"bf16", false, GU_ELEM*sizeof(bf16), N_BF16},
        {"fp8",  true,  GU_ELEM,              N_FP8 },
    };

    printf("\n== per-config phases (ms, mean of %d iters, L2 flushed each iter) ==\n",ITERS);
    for(const Cfg& c: cfgs){
        auto glu=[&](const unsigned char* tab){
            if(c.fp8) k_glu_fp8<<<GRID,256,smem>>>(dFu,dX,tab,dEwt8,dEst8,K8,I,H,E);
            else      k_glu    <<<GRID,256,smem>>>(dFu,dX,tab,dEwt ,K8,I,H,E); };
        auto down=[&](const unsigned char* tab){
            if(c.fp8) k_down_fp8<<<GRID,256>>>(dPart,dFu,tab,dEwt8,dEst8,K8,H,I,E);
            else      k_down    <<<GRID,256>>>(dPart,dFu,tab,dEwt ,K8,H,I,E); };
        const uint64_t* pfewt=c.fp8?dEwt8:dEwt;
        const double ws_mib=(double)K8*c.pf_bytes/1048576.0;
        const double dense_mib=(double)c.Ndense*H*2.0/1048576.0;
        printf("\n-- %s: prefetch working set %.1f MiB (8 experts, gate_up only); dense window %.0f MiB --\n",
               c.name, ws_mib, dense_mib);

        std::vector<double> t;
        run_seq({{"glu_cold",glu},{"glu_warm",glu},{"down",down}},t);
        const double glu_cold=t[0], glu_warm=t[1], down_cold=t[2], moe_step=glu_cold+down_cold;
        printf("%-8s glu_cold %8.4f  glu_warm %8.4f  down_cold %8.4f  (moe step %.4f, max win %.4f)\n",
               c.name,glu_cold,glu_warm,down_cold,moe_step,glu_cold-glu_warm);

        /* idle-window: does the prefetch land at all? */
        for(int el=0; el<2; el++){
            run_seq({{"pf",[&](const unsigned char* tab){
                        k_prefetch<<<GRID,256>>>(tab,(const unsigned long long*)pfewt,K8,E,c.pf_bytes,el); }},
                     {"glu",glu}},t);
            printf("%-8s pf_idle(%s) %8.4f  glu_after %8.4f  (recovered %.4f = %.0f%% of max)\n",
                   c.name, el?"evict_last":"plain", t[0], t[1], glu_cold-t[1],
                   (glu_cold-glu_warm)>0 ? 100.0*(glu_cold-t[1])/(glu_cold-glu_warm) : 0.0);
        }

        /* dense window baseline + fused dense||prefetch */
        run_seq({{"dense",[&](const unsigned char*){
                    k_dense<<<GRID,256>>>(dY,dDenseW,dX,c.Ndense,H,nullptr,nullptr,0,E,0,0,0); }},
                 {"glu",glu}},t);
        const double dense_base=t[0], glu_after_dense=t[1];
        printf("%-8s dense_base %8.4f (%.0f GB/s)  glu_after_dense %8.4f (cold check vs %.4f)\n",
               c.name,dense_base,dense_mib/1024.0/(dense_base/1e3),glu_after_dense,glu_cold);

        printf("%-8s %-14s %10s %10s %10s %10s %10s %8s\n",
               c.name,"variant","dense_ms","d_slow","glu_ms","glu_gain","net_ms","net/step");
        for(int el=0; el<2; el++) for(unsigned pfb: PFBS){
            run_seq({{"dense_pf",[&](const unsigned char* tab){
                        k_dense<<<GRID,256>>>(dY,dDenseW,dX,c.Ndense,H,tab,
                                (const unsigned long long*)pfewt,K8,E,c.pf_bytes,pfb,el); }},
                     {"glu",glu}},t);
            const double dslow=t[0]-dense_base, gain=glu_after_dense-t[1], net=gain-dslow;
            char nm[32]; snprintf(nm,sizeof nm,"%s pfb=%u",el?"evlast":"plain",pfb);
            printf("%-8s %-14s %10.4f %10.4f %10.4f %10.4f %10.4f %7.2f%%\n",
                   c.name,nm,t[0],dslow,t[1],gain,net,100.0*net/moe_step);
            fflush(stdout);
        }
    }
    printf("\nGO iff net > ~1%% of moe step AND dense slowdown ~ noise (see plan).\n");
    return 0;
}
