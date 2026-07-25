// C-1 S2 device microbench — SplitZip decode FUSED into the WS-GEMV batched rungs,
// at the REAL gemma-4-12B decode shapes with REAL 12B exponent statistics.
//
// Refresh of splitzip_gemv.cu (Qwen3-4B / MM=1) for the p9-v2 C-1 campaign:
//   * 12B shapes: qkv K=3840, o K=4096, gate/up K=3840, down K=15360, lm_head K=3840.
//   * batched rungs MM in {1,8,16} — one weight row loaded ONCE, dotted against MM x-rows,
//     mirroring gemv_rows<MM> in op_gemm.cuh (the reconstruct cost amortizes x MM).
//   * EXP_BASE from the C-0 audit (109 global for 12B; escapes <0.03%).
//
// Layout (SoA, exactly what the plow emitter would produce):
//   lo[] : 1 B/elem  = sign(bit7) | mantissa(bits6:0)
//   cd[] : 4 b/elem  = code, exponent = code + EXP_BASE
//   eoff[] u32/row prefix ; epos[] u32 flat idx ; eval[] u16 raw bf16
// Reconstruct: bf16 = ((lo&0x80)<<8) | (exp<<7) | (lo&0x7F), then sparse escape overwrite.
//
// MODE 0 = raw bf16 GEMV (reference). MODE 1 = fused SplitZip GEMV. Both share the FMA
// order => bit-IDENTICAL f32 outputs (memcmp gate). SZ x is smem-resident like the arena path.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define WARPS 8
#define BLK   (WARPS*32)
#ifndef EXP_BASE
#define EXP_BASE 109u
#endif

__device__ __forceinline__ float bf2f(unsigned short u) { return __uint_as_float((unsigned int)u << 16); }

// Batched WS-GEMV rung. One weight row is loaded once (raw or reconstructed) and dotted
// against MM activation rows staged in smem. Mirrors gemv_rows<MM>: MM accumulators/lane.
template<int MODE, int MM>
__global__ __launch_bounds__(BLK) void k_gemv_mm(
        const uint4* __restrict__ Wraw, const uint4* __restrict__ lo, const uint2* __restrict__ cd,
        const unsigned int* __restrict__ eoff, const unsigned int* __restrict__ epos,
        const unsigned short* __restrict__ eval, const unsigned short* __restrict__ x,
        float* __restrict__ y, int K, int N, int Mrows)
{
    // x read from global (L2-resident, MM*K*2 <= ~1 MB) — matches gemv_rows<MM> for M>1;
    // the B=1 arena-smem staging is an orthogonal optimization not on trial here.
    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int nj   = K >> 9;                       // 512 elems per warp-iteration (lane*16)
    long long stride = (long long)gridDim.x * WARPS;

    for (long long row = (long long)blockIdx.x * WARPS + w; row < N; row += stride) {
        long long rbase = row * (long long)K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.f;
        unsigned int e0 = 0, e1 = 0;
        if (MODE == 1) { e0 = eoff[row]; e1 = eoff[row + 1]; }
        for (int j = 0; j < nj; ++j) {
            long long base = rbase + (long long)j * 512;
            long long myel = base + lane * 16;
            unsigned short o[16];
            if (MODE == 0) {
                const uint4* p = Wraw + (myel >> 3);
                uint4 a = p[0], b = p[1];
                unsigned int aw[8] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for (int e = 0; e < 16; ++e) o[e] = (unsigned short)((aw[e >> 1] >> ((e & 1) * 16)) & 0xFFFFu);
            } else {
                uint4 l = lo[myel >> 4];
                uint2 c = cd[myel >> 4];
                unsigned int lw[4] = {l.x,l.y,l.z,l.w}, cw[2] = {c.x,c.y};
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    unsigned int bb = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
                    unsigned int cc = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
                    unsigned int ex = cc + EXP_BASE;
                    o[e] = (unsigned short)(((bb & 0x80u) << 8) | (ex << 7) | (bb & 0x7Fu));
                }
                for (unsigned int t = e0; t < e1; ++t) {
                    unsigned int p = epos[t];
                    if ((unsigned int)(p - (unsigned int)myel) < 16u) {
                        unsigned short v = eval[t]; unsigned int sl = p & 15u;
#pragma unroll
                        for (int e = 0; e < 16; ++e) if ((unsigned int)e == sl) o[e] = v;
                    }
                }
            }
            const int xo = j * 512 + lane * 16;
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const unsigned short* xr = x + (size_t)m * K + xo;
#pragma unroll
                for (int e = 0; e < 16; ++e) acc[m] = fmaf(bf2f(o[e]), bf2f(xr[e]), acc[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            float a = acc[m];
#pragma unroll
            for (int s = 16; s; s >>= 1) a += __shfl_down_sync(0xFFFFFFFFu, a, s);
            if (lane == 0) y[(size_t)m * N + row] = a;
        }
    }
}

struct Comp {
    std::vector<unsigned char>  lo, cd;
    std::vector<unsigned int>   eoff, epos;
    std::vector<unsigned short> eval;
    size_t n;
    double bytes() const { return (double)n + n / 2.0 + eoff.size() * 4.0 + epos.size() * 6.0 + 16.0; }
};
static Comp compress(const unsigned short* src, size_t n, size_t K) {
    Comp c; c.n = n; c.lo.resize(n); c.cd.assign(n / 2, 0);
    size_t nch = n / K; c.eoff.resize(nch + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        unsigned short u = src[i]; unsigned int ex = (u >> 7) & 0xFFu;
        c.lo[i] = (unsigned char)(((u >> 8) & 0x80u) | (u & 0x7Fu));
        int code = 0;
        if (ex >= EXP_BASE && ex <= EXP_BASE + 15) code = (int)ex - (int)EXP_BASE;
        else { c.epos.push_back((unsigned int)i); c.eval.push_back(u); c.eoff[i / K]++; }
        c.cd[i / 2] |= (unsigned char)(code << ((i & 1) * 4));
    }
    unsigned int run = 0;
    for (size_t k = 0; k <= nch; ++k) { unsigned int t = k < nch ? c.eoff[k] : 0; c.eoff[k] = run; run += t; }
    return c;
}

struct Dev { unsigned char *lo=0,*cd=0; unsigned int *eoff=0,*epos=0; unsigned short *eval=0,*raw=0; float* y=0; };

template<int MM> static void run_shape(const char* nm, int N, int K, int L,
        const std::vector<unsigned short>& src, cudaEvent_t ev0, cudaEvent_t ev1) {
    size_t nper=(size_t)N*K, ntot=nper*(size_t)L, nsrc=src.size();
    std::vector<unsigned short> W(ntot);
    for (size_t i=0;i<ntot;++i) W[i]=src[i%nsrc];
    Comp c=compress(W.data(),ntot,(size_t)K);
    double logical=ntot*2.0, comp=c.bytes(), ratio=logical/comp;
    Dev d;
    CK(cudaMalloc(&d.lo,ntot)); CK(cudaMalloc(&d.cd,ntot/2));
    CK(cudaMalloc(&d.eoff,c.eoff.size()*4)); CK(cudaMalloc(&d.epos,c.epos.size()*4+4));
    CK(cudaMalloc(&d.eval,c.eval.size()*2+2)); CK(cudaMalloc(&d.raw,ntot*2));
    CK(cudaMalloc(&d.y,(size_t)N*L*4*MM));
    CK(cudaMemcpy(d.lo,c.lo.data(),ntot,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.cd,c.cd.data(),ntot/2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.eoff,c.eoff.data(),c.eoff.size()*4,cudaMemcpyHostToDevice));
    if(c.epos.size()) CK(cudaMemcpy(d.epos,c.epos.data(),c.epos.size()*4,cudaMemcpyHostToDevice));
    if(c.eval.size()) CK(cudaMemcpy(d.eval,c.eval.data(),c.eval.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.raw,W.data(),ntot*2,cudaMemcpyHostToDevice));
    std::vector<unsigned short> hx((size_t)MM*K);
    for (size_t i=0;i<(size_t)MM*K;++i) hx[i]=src[(i*7919)%nsrc];
    unsigned short* d_x; CK(cudaMalloc(&d_x,(size_t)MM*K*2));
    CK(cudaMemcpy(d_x,hx.data(),(size_t)MM*K*2,cudaMemcpyHostToDevice));
    int grid=1020, N_L=N*L;
    auto timeit=[&](int mode){
        for(int wm=0;wm<2;++wm){
            if(wm)CK(cudaEventRecord(ev0));
            int it=wm?10:2;
            for(int i=0;i<it;++i){
                if(mode==0) k_gemv_mm<0,MM><<<grid,BLK>>>((uint4*)d.raw,0,0,0,0,0,d_x,d.y,K,N_L,MM);
                else        k_gemv_mm<1,MM><<<grid,BLK>>>(0,(uint4*)d.lo,(uint2*)d.cd,d.eoff,d.epos,d.eval,d_x,d.y,K,N_L,MM);
            }
            if(wm){CK(cudaEventRecord(ev1));CK(cudaEventSynchronize(ev1));}
            CK(cudaDeviceSynchronize());CK(cudaGetLastError());
        }
        float ms; CK(cudaEventElapsedTime(&ms,ev0,ev1)); return (double)ms/10.0;
    };
    double ms_c=timeit(1); std::vector<float> y_c((size_t)N_L*MM);
    CK(cudaMemcpy(y_c.data(),d.y,(size_t)N_L*MM*4,cudaMemcpyDeviceToHost));
    double ms_r=timeit(0); std::vector<float> y_r((size_t)N_L*MM);
    CK(cudaMemcpy(y_r.data(),d.y,(size_t)N_L*MM*4,cudaMemcpyDeviceToHost));
    size_t bad=0; for(size_t i=0;i<(size_t)N_L*MM;++i) bad+=(y_c[i]!=y_r[i]);
    // logical bytes DELIVERED = raw weight bytes read (compression is transparent to the consumer)
    double log_gbs_r=logical/1e9/(ms_r/1e3), log_gbs_c=logical/1e9/(ms_c/1e3);
    printf("%-16s MM=%-2d ratio=%.4f  raw %6.3fms %7.1f GB/s | SZ %6.3fms %7.1f GB/s  vsraw=%.3fx  real=%.1f%%  %s(bad=%zu)\n",
           nm, MM, ratio, ms_r, log_gbs_r, ms_c, log_gbs_c, ms_r/ms_c, 100.0*(ms_r/ms_c)/ratio,
           bad==0?"BITEXACT ":"MISMATCH ", bad);
    CK(cudaFree(d.lo));CK(cudaFree(d.cd));CK(cudaFree(d.eoff));CK(cudaFree(d.epos));
    CK(cudaFree(d.eval));CK(cudaFree(d.raw));CK(cudaFree(d.y));CK(cudaFree(d_x));
}

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"/tmp/g12_sample.bin";
    FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);return 1;}
    fseek(f,0,SEEK_END); size_t nb=ftell(f); fseek(f,0,SEEK_SET);
    size_t nsrc=nb/2; std::vector<unsigned short> src(nsrc);
    if(fread(src.data(),2,nsrc,f)!=nsrc)return 1; fclose(f);
    printf("source: %zu real bf16 elems (%.1f MB) from gemma-4-12B gate_proj  EXP_BASE=%u\n",
           nsrc, nsrc*2/1048576.0, EXP_BASE);
    // L chosen so each working set is a few hundred MB (real exponent stats, HBM-resident).
    struct Sh{const char*nm;int N,K,L;};
    Sh sh[]={{"qkv    K3840",6144,3840,8},{"o_proj K4096",3840,4096,8},
             {"gate/up K3840",15360,3840,4},{"down   K15360",3840,15360,4},
             {"lm_head K3840",262144,3840,1}};
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("\n-- MM=1 (B=1 decode; conservative — reconstruct NOT amortized) --\n");
    for(auto&s:sh) run_shape<1>(s.nm,s.N,s.K,s.L,src,e0,e1);
    printf("\n-- MM=8 (B=8 serving sweet spot; reconstruct amortized x8) --\n");
    for(auto&s:sh) run_shape<8>(s.nm,s.N,s.K,s.L,src,e0,e1);
    printf("\n-- MM=16 (B=16 rung; the 255-reg cliff watch) --\n");
    for(auto&s:sh) run_shape<16>(s.nm,s.N,s.K,s.L,src,e0,e1);
    return 0;
}
