/* sz_ws_sm120.cu — C-1R H2 control: producer/consumer WARP-SPECIALIZED SplitZip decode GEMV
 * vs naive-sz (gemv_rows_sz) vs bf16 (gemv_rows), all at the REAL decode geometry
 * (GRID = SM count, 1 block/SM). Answers H2: at 1 block/SM, does moving the ~6-ALU-op/elem
 * reconstruct into dedicated PRODUCER warps (feeding a smem ring) that dedicated CONSUMER
 * warps FMA from, recover the win? Prediction (from C-1): NO — warp-spec adds no occupancy
 * and cannot create a BW shadow that isn't there at ~96% of peak.
 *
 * The block's 8 warps split 4 producers / 4 consumers. Producer p reconstructs SUPER chunks
 * (SUPER*GV_STEP bf16) of one output row into a smem ring slot; consumer p FMAs them against x.
 * Bit-exact gate: the accumulation order (chunk 0..nchunk-1, dot8 lane*8..+7) is IDENTICAL to
 * gemv_rows, so the f32 output must be BYTE-IDENTICAL to bf16 GEMV.
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_120a -Iinclude -Iruntime/common -Iruntime/nvidia \
 *          runtime/tests/sz_ws_sm120.cu -o /tmp/szws -lcuda
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#ifndef WS_SUPER
#define WS_SUPER 8      /* chunks (GV_STEP each) reconstructed per ring slot; matches GV_UNROLL x-MLP */
#endif
#ifndef WS_RINGD
#define WS_RINGD 2      /* ring depth (slots) per producer/consumer pair */
#endif
#define WS_NPAIR 4      /* 4 producer warps + 4 consumer warps = 8 warps */

/* smem layout: wtile[pair][slot][WS_SUPER*GV_STEP], then produced[4], consumed[4]. */
struct WsSmem {
    __nv_bfloat16 wtile[WS_NPAIR][WS_RINGD][WS_SUPER * (int)GV_STEP];
    volatile unsigned produced[WS_NPAIR];
    volatile unsigned consumed[WS_NPAIR];
};

template <int MM>
__global__ void __launch_bounds__(256, 1) k_ws(
        __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ lo, const uint8_t* __restrict__ cd,
        const unsigned* __restrict__ eoff, const unsigned* __restrict__ epos,
        const __nv_bfloat16* __restrict__ eval, unsigned exp_base,
        unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk) {
    extern __shared__ char smem_raw[];
    WsSmem* sm = (WsSmem*)smem_raw;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned ngrp = (nchunk + WS_SUPER - 1) / WS_SUPER;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    if (threadIdx.x < WS_NPAIR) { sm->produced[threadIdx.x] = 0; sm->consumed[threadIdx.x] = 0; }
    __syncthreads();

    if (warp < WS_NPAIR) {
        /* ---- PRODUCER warp `warp` : reconstruct rows n = n0+warp, +4, ... into ring ---- */
        const unsigned p = warp;
        unsigned flat = 0;                       /* running produced group count */
        for (unsigned n = n0 + p; n < n1; n += WS_NPAIR) {
            const size_t rbase = (size_t)n * K;
            const unsigned e0 = eoff[n], e1 = eoff[n + 1];
            for (unsigned g = 0; g < ngrp; ++g) {
                while (flat - sm->consumed[p] >= (unsigned)WS_RINGD) { /* spin: slot busy */ }
                __nv_bfloat16* dst = sm->wtile[p][flat % WS_RINGD];
#pragma unroll
                for (int s = 0; s < WS_SUPER; ++s) {
                    const unsigned c = g * WS_SUPER + (unsigned)s;
                    const unsigned k = c * GV_STEP + lane * 8;
                    bf16v8 wv;
                    if (c < nchunk && k < K) {
                        wv = sz_expand8(lo, cd, rbase + k, exp_base);
                        if (e1 != e0) sz_escape8(wv, rbase + k, e0, e1, epos, eval);
                    } else {
                        wv = bf16v8_zero();
                    }
                    st_glob8(dst + s * (int)GV_STEP + lane * 8, wv);
                }
                __syncwarp();
                __threadfence_block();
                if (lane == 0) sm->produced[p] = flat + 1;
                flat++;
            }
        }
    } else {
        /* ---- CONSUMER warp : FMA rows n = n0+cc, +4, ... from the ring ---- */
        const unsigned cc = warp - WS_NPAIR;
        unsigned flat = 0;
        for (unsigned n = n0 + cc; n < n1; n += WS_NPAIR) {
            float acc[MM];
#pragma unroll
            for (int m = 0; m < MM; ++m) acc[m] = 0.0f;
            for (unsigned g = 0; g < ngrp; ++g) {
                while (sm->produced[cc] <= flat) { /* spin: slot not ready */ }
                __threadfence_block();
                const __nv_bfloat16* src = sm->wtile[cc][flat % WS_RINGD];
#pragma unroll
                for (int s = 0; s < WS_SUPER; ++s) {
                    const unsigned c = g * WS_SUPER + (unsigned)s;
                    const unsigned k = c * GV_STEP + lane * 8;
                    if (c >= nchunk || k >= K) continue;
                    const bf16v8 wv = ld_smem8(src + s * (int)GV_STEP + lane * 8);
#pragma unroll
                    for (int m = 0; m < MM; ++m) {
                        if ((unsigned)m >= M) continue;
                        acc[m] = dot8(wv, ld_glob8(x + (size_t)m * K + k), acc[m]);
                    }
                }
                __syncwarp();
                __threadfence_block();
                if (lane == 0) sm->consumed[cc] = flat + 1;
                flat++;
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                const float t = warp_sum32(acc[m]);
                if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
            }
        }
    }
}

/* ---- H2b (user-proposed): cp.async-staged, INLINE-smem decompress, NO warp specialization.
 * All 8 warps do FMA (warp-per-row, exactly like gemv_rows_sz); the ONLY change is the
 * compressed operand (lo 8B + cd 4B per lane per chunk) is streamed global->smem via cp.async
 * with a D-deep software pipeline, then reconstructed FROM smem inline. This tests whether the
 * naive-sz loss at 1 block/SM is load/issue-latency (curable by cp.async overlap, no FMA-warp
 * cost) rather than pure occupancy. Bit-exact: identical FMA order to gemv_rows. */
__device__ __forceinline__ void cp_async8(void* smem, const void* g) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ bf16v8 sz_expand8_s(const uint8_t* lo8, const uint8_t* cd4,
                                               unsigned exp_base) {
    const uint2 lb = *(const uint2*)lo8;
    const unsigned cw = *(const unsigned*)cd4;
    const unsigned lw[2] = {lb.x, lb.y};
    bf16v8 r;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned b = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
        const unsigned ex = ((cw >> (e * 4)) & 0xFu) + exp_base;
        const unsigned short u = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
        r.x[e] = __ushort_as_bfloat16(u);
    }
    return r;
}
#ifndef CP_D
#define CP_D 4          /* software-pipeline depth (smem stages) */
#endif
template <int MM>
__global__ void __launch_bounds__(256, 1) k_szcp(
        __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ lo, const uint8_t* __restrict__ cd,
        const unsigned* __restrict__ eoff, const unsigned* __restrict__ epos,
        const __nv_bfloat16* __restrict__ eval, unsigned exp_base,
        unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk) {
    extern __shared__ char smem_raw[];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    uint8_t* lo_s = (uint8_t*)smem_raw;                 /* [8][CP_D][256] */
    uint8_t* cd_s = lo_s + (size_t)8 * CP_D * 256;      /* [8][CP_D][128] */
    uint8_t* my_lo = lo_s + ((size_t)warp * CP_D) * 256 + lane * 8;
    uint8_t* my_cd = cd_s + ((size_t)warp * CP_D) * 128 + lane * 4;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const size_t rbase = (size_t)n * K;
        const unsigned e0 = eoff[n], e1 = eoff[n + 1];
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        auto issue = [&](unsigned c) {
            const unsigned k = c * GV_STEP + lane * 8;
            const unsigned b = c % CP_D;
            cp_async8(my_lo + (size_t)b * 256, lo + rbase + k);
            cp_async4(my_cd + (size_t)b * 128, cd + (rbase + k) / 2);
        };
        unsigned pr = 0;
        for (; pr < (unsigned)CP_D && pr < nchunk; ++pr) { issue(pr); fa_cp_commit(); }
        for (unsigned c = 0; c < nchunk; ++c) {
            fa_cp_wait<CP_D - 1>();
            __syncwarp();
            const unsigned b = c % CP_D;
            const unsigned k = c * GV_STEP + lane * 8;
            bf16v8 wv = sz_expand8_s(my_lo + (size_t)b * 256, my_cd + (size_t)b * 128, exp_base);
            if (e1 != e0) sz_escape8(wv, rbase + k, e0, e1, epos, eval);
#pragma unroll
            for (int m = 0; m < MM; m++) {
                if ((unsigned)m >= M) continue;
                acc[m] = dot8(wv, ld_glob8(x + (size_t)m * K + k), acc[m]);
            }
            if (c + CP_D < nchunk) { issue(c + CP_D); fa_cp_commit(); }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}

template <int MM> __global__ void k_bf16(__nv_bfloat16* C, const __nv_bfloat16* x,
        const __nv_bfloat16* W, unsigned M, unsigned N, unsigned K) {
    gemv_rows<MM>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}
template <int MM> __global__ void k_sz(__nv_bfloat16* C, const __nv_bfloat16* x,
        const uint8_t* lo, const uint8_t* cd, const unsigned* eoff, const unsigned* epos,
        const __nv_bfloat16* eval, unsigned exp_base, unsigned M, unsigned N, unsigned K) {
    gemv_rows_sz<MM>(C, x, lo, cd, eoff, epos, eval, exp_base, M, N, K, blockIdx.x, gridDim.x);
}

static unsigned GRID = 188; static const unsigned BLOCK = 256, EXP_BASE = 109;

struct Comp { std::vector<uint8_t> lo, cd; std::vector<unsigned> eoff, epos; std::vector<uint16_t> eval; };
static Comp compress(const uint16_t* s, size_t n, size_t K) {
    Comp c; c.lo.resize(n); c.cd.assign(n/2,0); size_t nch=n/K; c.eoff.assign(nch+1,0);
    for (size_t i=0;i<n;++i){ uint16_t u=s[i]; unsigned ex=(u>>7)&0xFF;
        c.lo[i]=(uint8_t)(((u>>8)&0x80)|(u&0x7F)); int code=0;
        if(ex>=EXP_BASE&&ex<=EXP_BASE+15) code=(int)ex-(int)EXP_BASE;
        else{ c.epos.push_back((unsigned)i); c.eval.push_back(u); c.eoff[i/K]++; }
        c.cd[i/2]|=(uint8_t)(code<<((i&1)*4)); }
    unsigned run=0; for(size_t k=0;k<=nch;++k){unsigned t=k<nch?c.eoff[k]:0;c.eoff[k]=run;run+=t;}
    return c;
}

template <int MM> static void run_mm(const char* nm, int N, int K, int L,
        const std::vector<uint16_t>& src, int iters) {
    size_t nper=(size_t)N*K, ntot=nper*(size_t)L, nsrc=src.size();
    std::vector<uint16_t> W(ntot); for(size_t i=0;i<ntot;++i) W[i]=src[i%nsrc];
    Comp c=compress(W.data(),ntot,(size_t)K);
    double logical=ntot*2.0, comp=(double)ntot+ntot/2.0+c.eoff.size()*4.0+c.epos.size()*6.0;
    __nv_bfloat16 *dW,*dC,*dx; uint8_t *dlo,*dcd; unsigned *deoff,*depos; __nv_bfloat16* deval;
    CK(cudaMalloc(&dW,ntot*2)); CK(cudaMalloc(&dC,(size_t)N*L*MM*2));
    CK(cudaMalloc(&dlo,ntot)); CK(cudaMalloc(&dcd,ntot/2));
    CK(cudaMalloc(&deoff,c.eoff.size()*4)); CK(cudaMalloc(&depos,c.epos.size()*4+4));
    CK(cudaMalloc(&deval,c.eval.size()*2+2)); CK(cudaMalloc(&dx,(size_t)MM*K*2));
    CK(cudaMemcpy(dW,W.data(),ntot*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlo,c.lo.data(),ntot,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcd,c.cd.data(),ntot/2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(deoff,c.eoff.data(),c.eoff.size()*4,cudaMemcpyHostToDevice));
    if(c.epos.size())CK(cudaMemcpy(depos,c.epos.data(),c.epos.size()*4,cudaMemcpyHostToDevice));
    if(c.eval.size())CK(cudaMemcpy(deval,c.eval.data(),c.eval.size()*2,cudaMemcpyHostToDevice));
    std::vector<uint16_t> hx((size_t)MM*K); for(size_t i=0;i<hx.size();++i) hx[i]=src[(i*7919)%nsrc];
    CK(cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    int NL=N*L;
    unsigned shbytes = sizeof(WsSmem);
    unsigned shcp = 8u * CP_D * 256u + 8u * CP_D * 128u;
    CK(cudaFuncSetAttribute(k_ws<MM>, cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes));
    CK(cudaFuncSetAttribute(k_szcp<MM>, cudaFuncAttributeMaxDynamicSharedMemorySize, shcp));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    /* run + capture each variant's output */
    auto launch=[&](int mode){
        if(mode==0) k_bf16<MM><<<GRID,BLOCK>>>(dC,dx,dW,MM,NL,K);
        else if(mode==1) k_sz<MM><<<GRID,BLOCK>>>(dC,dx,dlo,dcd,deoff,depos,deval,EXP_BASE,MM,NL,K);
        else if(mode==2) k_ws<MM><<<GRID,BLOCK,shbytes>>>(dC,dx,dlo,dcd,deoff,depos,deval,EXP_BASE,MM,NL,K,0,GRID);
        else k_szcp<MM><<<GRID,BLOCK,shcp>>>(dC,dx,dlo,dcd,deoff,depos,deval,EXP_BASE,MM,NL,K,0,GRID);
    };
    auto bench=[&](int mode){
        for(int i=0;i<3;++i) launch(mode); CK(cudaDeviceSynchronize());CK(cudaGetLastError());
        CK(cudaEventRecord(e0));
        for(int i=0;i<iters;++i) launch(mode);
        CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms/iters;
    };
    std::vector<uint16_t> ybf((size_t)NL*MM), ysz((size_t)NL*MM), yws((size_t)NL*MM), ycp((size_t)NL*MM);
    double mbf=bench(0); CK(cudaMemcpy(ybf.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    double msz=bench(1); CK(cudaMemcpy(ysz.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    double mws=bench(2); CK(cudaMemcpy(yws.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    double mcp=bench(3); CK(cudaMemcpy(ycp.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    size_t badsz=0,badws=0,badcp=0; for(size_t i=0;i<(size_t)NL*MM;++i){badsz+=(ysz[i]!=ybf[i]);badws+=(yws[i]!=ybf[i]);badcp+=(ycp[i]!=ybf[i]);}
    double ratio=logical/comp;
    double bw_bf=logical/1e9/(mbf/1e3), bw_sz=logical/1e9/(msz/1e3), bw_ws=logical/1e9/(mws/1e3), bw_cp=logical/1e9/(mcp/1e3);
    printf("%-14s MM=%-2d | bf16 %6.3fms %7.1f | sz %6.3fms %7.1f (%.3fx) | ws %6.3fms %7.1f (%.3fx) | cp %6.3fms %7.1f (%.3fx) | sz=%s ws=%s cp=%s(%zu)\n",
           nm,MM,mbf,bw_bf,msz,bw_sz,mbf/msz,mws,bw_ws,mbf/mws,mcp,bw_cp,mbf/mcp,
           badsz==0?"OK":"BAD",badws==0?"OK":"BAD",badcp==0?"OK":"BAD",badcp);
    cudaFree(dW);cudaFree(dC);cudaFree(dlo);cudaFree(dcd);cudaFree(deoff);cudaFree(depos);cudaFree(deval);cudaFree(dx);
}

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"/tmp/g12_sample.bin"; int iters=argc>2?atoi(argv[2]):100;
    if(const char* g=getenv("SZ_GRID")) GRID=atoi(g);
    FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);return 1;}
    fseek(f,0,SEEK_END);size_t nb=ftell(f);fseek(f,0,SEEK_SET); std::vector<uint16_t> src(nb/2);
    if(fread(src.data(),2,src.size(),f)!=src.size())return 1; fclose(f);
    printf("WARP-SPEC sz vs naive-sz vs bf16 @ 12B shapes, GRID=%u (blocks/SM=%.2f), EXP_BASE=%u, SUPER=%d RINGD=%d smem=%zuB\n\n",
           GRID, GRID/188.0, EXP_BASE, WS_SUPER, WS_RINGD, sizeof(WsSmem));
    struct Sh{const char*nm;int N,K,L;};
    Sh sh[]={{"qkv    K3840",6144,3840,8},{"o_proj K4096",3840,4096,8},
             {"gate/up K3840",15360,3840,4},{"down   K15360",3840,15360,4}};
#define RUN(MM) for(auto&s:sh) run_mm<MM>(s.nm,s.N,s.K,s.L,src,iters); printf("\n");
    printf("-- MM=1 (B=1) --\n");  RUN(1)
    printf("-- MM=4 (B=4) --\n");  RUN(4)
    printf("-- MM=8 (B=8) --\n");  RUN(8)
    printf("-- MM=16 (B=16) --\n"); RUN(16)
    return 0;
}
