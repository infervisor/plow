/* sz_batch_sm120.cu — C-1 S4(ii): the REAL gemv_rows_sz vs gemv_rows batched-rung A/B,
 * launched with the interpreter's own geometry (GRID = SM count, 1 block/SM, slice=blockIdx,
 * nblk=gridDim), at gemma-4-12B decode shapes with REAL 12B weight bytes. Answers the campaign
 * question directly: does the 1.33x-fewer-weight-bytes SplitZip path speed up the shared
 * weight pass at B = 1,2,4,8,16 — or are the batched rungs already off the BW wall?
 *
 * Uses the SAME kernels the megakernel instantiates (op_gemm.cuh gemv_rows<MM> / gemv_rows_sz<MM>),
 * so the numbers are faithful to the interpreter. Bit-exact gate: sz output == bf16 output.
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_120a -Iinclude -Iruntime/common -Iruntime/nvidia \
 *          runtime/tests/sz_batch_sm120.cu -o /tmp/szbatch -lcuda
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

template <int MM> static double run_mm(const char* nm, int N, int K, int L,
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
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int NL=N*L;
    auto timeit=[&](int mode){
        for(int w=0;w<2;++w){ if(w)CK(cudaEventRecord(e0)); int it=w?iters:3;
            for(int i=0;i<it;++i){
                if(mode==0) k_bf16<MM><<<GRID,BLOCK>>>(dC,dx,dW,MM,NL,K);
                else k_sz<MM><<<GRID,BLOCK>>>(dC,dx,dlo,dcd,deoff,depos,deval,EXP_BASE,MM,NL,K);
            }
            if(w){CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));} CK(cudaDeviceSynchronize());CK(cudaGetLastError()); }
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms/iters;
    };
    double msz=timeit(1); std::vector<uint16_t> ysz((size_t)NL*MM); CK(cudaMemcpy(ysz.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    double mbf=timeit(0); std::vector<uint16_t> ybf((size_t)NL*MM); CK(cudaMemcpy(ybf.data(),dC,(size_t)NL*MM*2,cudaMemcpyDeviceToHost));
    size_t bad=0; for(size_t i=0;i<(size_t)NL*MM;++i) bad+=(ysz[i]!=ybf[i]);
    double ratio=logical/comp, bw_bf=logical/1e9/(mbf/1e3), bw_sz=logical/1e9/(msz/1e3);
    printf("%-14s MM=%-2d ratio=%.3f  bf16 %6.3fms %7.1f GB/s | sz %6.3fms %7.1f GB/s  speedup=%.3fx  real=%5.1f%%  %s(%zu)\n",
           nm,MM,ratio,mbf,bw_bf,msz,bw_sz,mbf/msz,100.0*(mbf/msz)/ratio,bad==0?"BITEXACT ":"MISMATCH ",bad);
    cudaFree(dW);cudaFree(dC);cudaFree(dlo);cudaFree(dcd);cudaFree(deoff);cudaFree(depos);cudaFree(deval);cudaFree(dx);
    return mbf/msz;
}

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"/tmp/g12_sample.bin"; int iters=argc>2?atoi(argv[2]):100;
    if(const char* g=getenv("SZ_GRID")) GRID=atoi(g);
    FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);return 1;}
    fseek(f,0,SEEK_END);size_t nb=ftell(f);fseek(f,0,SEEK_SET); std::vector<uint16_t> src(nb/2);
    if(fread(src.data(),2,src.size(),f)!=src.size())return 1; fclose(f);
    printf("REAL gemv_rows_sz vs gemv_rows @ 12B shapes, GRID=%u (1 blk/SM), EXP_BASE=%u\n\n",GRID,EXP_BASE);
    struct Sh{const char*nm;int N,K,L;};
    Sh sh[]={{"qkv    K3840",6144,3840,8},{"o_proj K4096",3840,4096,8},
             {"gate/up K3840",15360,3840,4},{"down   K15360",3840,15360,4}};
#define RUN(MM) for(auto&s:sh) run_mm<MM>(s.nm,s.N,s.K,s.L,src,iters); printf("\n");
    printf("-- MM=1 (B=1) --\n");  RUN(1)
    printf("-- MM=2 (B=2) --\n");  RUN(2)
    printf("-- MM=4 (B=4) --\n");  RUN(4)
    printf("-- MM=8 (B=8) --\n");  RUN(8)
    printf("-- MM=16 (B=16) --\n"); RUN(16)
    return 0;
}
