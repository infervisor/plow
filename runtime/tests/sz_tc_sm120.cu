/* sz_tc_sm120.cu — C-1T Thread 1: tensor-core small-M decode GEMM, B-sweep crossover.
 *
 * Moves batched decode (B rows share one weight read) off the CUDA-core FFMA path (WS-GEMV,
 * gemv_rows) onto TENSOR cores (mma.sync m16n8k16, small-M tile BM=16) with the compressed
 * weight staged and decompressed INLINE in the cp.async ring's stager (Thread-2 V3 packed):
 *   stageA cp.async A tile -> stageB cp.async COMPRESSED weight tile -> expand to bf16 smem
 *   -> stageC ldmatrix + mma.  Software-pipelined by depth across ALL warps (no warp-spec).
 *
 * The mma sees the EXACT bf16 the control feeds (V3 reconstructs bit-identical bf16), so the
 * tc-sz f32 output is BYTE-IDENTICAL to the tc-bf16 output — the bit-exact gate.
 *
 * Compares per shape x B in {1,2,4,8,16,32}:
 *   FFMA-bf16 (gemv_rows)  |  FFMA-sz (gemv_rows_sz)  |  TC-bf16 (this GEMM)  |  TC-sz (this GEMM).
 * Reports the crossover B where TC-sz stops beating TC-bf16, and where TC beats FFMA.
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_120a -Xptxas -v -Iinclude -Iruntime/common -Iruntime/nvidia \
 *          runtime/tests/sz_tc_sm120.cu -o /tmp/c1t_tc -lcuda
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

/* ---- small-M GEMM tiling (BM=16 => one mma m-dim; weight-reuse factor = 16 < ~75 crossover) ---- */
#define TM 16
#define TN 128
#define TK 32
#define TKPAD 8
#define TBKS (TK + TKPAD)        /* 40  B/Bbf smem stride [n][k] */
#define TAS  (TK + TKPAD)        /* 40  A smem stride [m][k]     */
#define TWN  (TN / 8)            /* 16  per-warp N (8 warps all in N) */
#define TNFRAG (TWN / 8)         /* 2   n-fragments per warp */
#ifndef TC_STAGES
#define TC_STAGES 3
#endif
#define TABUF (TM * TAS)         /* 640  bf16/stage A */
#define TBBUF (TN * TBKS)        /* 5120 bf16 B convert / bf16-ring stage */
#define TLOBUF (TN * TK)         /* 4096 bytes lo/stage */
#define TCDBUF (TN * TK / 2)     /* 2048 bytes cd/stage */

/* ---- V3 packed reconstruct (Thread-2 winner): lo 8B + cd 4B -> bf16v8 ----
 * TC_EXPAND_V0=1 swaps in the scalar baseline to prove the TC-sz wall is the smem round-trip
 * + syncs, NOT the decompress ALU (V0 is 11.8 ops/elem vs V3 5.5). */
#if defined(TC_EXPAND_V0)
__device__ __forceinline__ bf16v8 expand_v3(const uint2 lb, const unsigned cw, unsigned base) {
    const unsigned lw[2] = {lb.x, lb.y};
    bf16v8 r;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned b = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
        const unsigned ex = ((cw >> (e * 4)) & 0xFu) + base;
        const unsigned short u = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
        r.x[e] = __ushort_as_bfloat16(u);
    }
    return r;
}
#else
__device__ __forceinline__ bf16v8 expand_v3(const uint2 lb, const unsigned cw, unsigned base) {
    const unsigned lw[2] = {lb.x, lb.y};
    const unsigned baseK = base | (base << 16);
    unsigned res[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const unsigned L = (lw[p >> 1] >> ((p & 1) * 16)) & 0xFFFFu;
        const unsigned spread = __byte_perm(L, 0u, 0x4140u);
        const unsigned b16 = (spread & 0x007F007Fu) | ((spread << 8) & 0x80008000u);
        const unsigned cwp = cw >> (8 * p);
        const unsigned e = (cwp & 0xFu) | ((cwp & 0xF0u) << 12);
        const unsigned expw = (e + baseK) << 7;
        res[p] = b16 | expw;
    }
    bf16v8 r; memcpy(&r, res, 16); return r;
}
#endif

/* ---- stagers (mirror pgm_stage_b_fp8; per-row lines because [n][k] has stride K) ---- */
__device__ __forceinline__ void tc_stage_a(__nv_bfloat16* Ad, const __nv_bfloat16* __restrict__ A,
                                            int tid, int tm, int kbase, unsigned M, unsigned K) {
    const int KCH = TK / 8;
    for (int L = tid; L < TM * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int mm = tm + row, kk = kbase + kk8;
        const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
        const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
        pgm_cp_async_cg16(&Ad[row * TAS + kk8], g, in ? 16 : 0);
    }
}
/* bf16 weight tile stage (control) */
__device__ __forceinline__ void tc_stage_b_bf16(__nv_bfloat16* Bd, const __nv_bfloat16* __restrict__ B,
                                                 int tid, int tn, int kbase, unsigned N, unsigned K) {
    const int KCH = TK / 8;
    for (int L = tid; L < TN * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int nn = tn + row, kk = kbase + kk8;
        const bool in = (nn < (int)N) && (kk + 8 <= (int)K);
        const __nv_bfloat16* g = in ? B + (size_t)nn * K + kk : B;
        pgm_cp_async_cg16(&Bd[row * TBKS + kk8], g, in ? 16 : 0);
    }
}
/* compressed weight tile stage: lo [TN][TK] bytes (2 x 16B lines/row), cd [TN][TK/2] (1 line/row) */
__device__ __forceinline__ void tc_stage_b_sz(uint8_t* lod, uint8_t* cdd, const uint8_t* __restrict__ lo,
                                               const uint8_t* __restrict__ cd, int tid, int tn,
                                               int kbase, unsigned N, unsigned K) {
    const int LCH = TK / 16;    /* 2 lo lines/row */
    for (int L = tid; L < TN * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const int nn = tn + row, kk = kbase + kk16;
        const bool in = (nn < (int)N) && (kk + 16 <= (int)K);
        const uint8_t* g = in ? lo + (size_t)nn * K + kk : lo;
        pgm_cp_async_cg16(&lod[row * TK + kk16], g, in ? 16 : 0);
    }
    /* cd: TK/2 = 16 bytes/row = exactly one 16B line */
    for (int row = tid; row < TN; row += (int)PLOW_NV_THREADS) {
        const int nn = tn + row, kk = kbase;
        const bool in = (nn < (int)N) && (kk + TK <= (int)K);
        const size_t coff = ((size_t)nn * K + kk) / 2;
        const uint8_t* g = in ? cd + coff : cd;
        pgm_cp_async_cg16(&cdd[row * (TK/2)], g, in ? 16 : 0);
    }
}
/* expand landed compressed tile -> bf16 convert tile Bbf [n][k] (BKS-padded), + escapes */
__device__ __forceinline__ void tc_expand_b_sz(__nv_bfloat16* Bbf, const uint8_t* lod,
                                                const uint8_t* cdd, int tid, int tn, int kbase,
                                                unsigned N, unsigned K, unsigned base,
                                                const unsigned* __restrict__ eoff,
                                                const unsigned* __restrict__ epos,
                                                const __nv_bfloat16* __restrict__ eval) {
    const int KCH = TK / 8;   /* 4 chunks(8 elems)/row */
    for (int L = tid; L < TN * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const uint2 lb = *(const uint2*)(lod + row * TK + kk8);
        const unsigned cw = *(const unsigned*)(cdd + row * (TK/2) + kk8/2);
        bf16v8 wv = expand_v3(lb, cw, base);
        const int nn = tn + row;
        if (nn < (int)N) {
            const size_t el = (size_t)nn * K + (kbase + kk8);
            const unsigned e0 = eoff[nn], e1 = eoff[nn + 1];
            if (e1 != e0) sz_escape8(wv, el, e0, e1, epos, eval);
        }
        *(uint4*)&Bbf[row * TBKS + kk8] = *(const uint4*)&wv;
    }
}

/* ---- local frag loaders (BM=16 / WARPS_N=8 tiling) ---- */
__device__ __forceinline__ void tc_load_afrag(unsigned (&af)[4], __nv_bfloat16* Ad, int kf, int lane) {
    const int arow = (lane % 16);
    const int acol = kf + (lane / 16) * 8;
    pgm_ldmatrix_x4(af, &Ad[arow * TAS + acol]);
}
__device__ __forceinline__ void tc_load_bfrags(unsigned (&bf)[TNFRAG][2], __nv_bfloat16* Bd,
                                               int warp, int kf, int lane) {
#pragma unroll
    for (int nj = 0; nj < TNFRAG; nj++) {
        const int n = warp * TWN + nj * 8 + (lane & 7);
        const int kcol = kf + ((lane >> 3) & 1) * 8;
        pgm_ldmatrix_x2(bf[nj], &Bd[n * TBKS + kcol]);
    }
}

/* ---- mainloop body shared shape; STORE map for m16n8k16 acc ---- */
__device__ __forceinline__ void tc_store(float (&acc)[TNFRAG][4], __nv_bfloat16* C, int tm, int tn,
                                         int warp, int lane, unsigned M, unsigned N) {
#pragma unroll
    for (int nj = 0; nj < TNFRAG; nj++) {
        const int gr = (lane / 4);
        const int gc = warp * TWN + nj * 8 + (lane % 4) * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int rr = tm + gr + (e / 2) * 8;
            const int cc = tn + gc + (e % 2);
            if (rr < (int)M && cc < (int)N) C[(size_t)rr * N + cc] = __float2bfloat16(acc[nj][e]);
        }
    }
}

/* TC bf16 control GEMM */
__global__ void __launch_bounds__(256,1) k_tc_bf16(__nv_bfloat16* __restrict__ C,
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        unsigned M, unsigned N, unsigned K) {
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    __nv_bfloat16* Bs = As + TC_STAGES * TABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_m = (M + TM - 1) / TM, tiles_n = (N + TN - 1) / TN;
    const int ntiles = tiles_m * tiles_n, ksteps = (K + TK - 1) / TK;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int tm = (tile / tiles_n) * TM, tn = (tile % tiles_n) * TN;
        float acc[TNFRAG][4];
#pragma unroll
        for (int j=0;j<TNFRAG;j++) for(int e=0;e<4;e++) acc[j][e]=0.f;
        auto stage=[&](int ks,int buf){ tc_stage_a(As+buf*TABUF,A,tid,tm,ks*TK,M,K);
            tc_stage_b_bf16(Bs+buf*TBBUF,B,tid,tn,ks*TK,N,K); };
#pragma unroll
        for (int s=0;s<TC_STAGES-1;s++){ if(s<ksteps) stage(s,s); pgm_cp_commit(); }
        for (int ks=0; ks<ksteps; ks++){
            const int fetch=ks+TC_STAGES-1; if(fetch<ksteps) stage(fetch,fetch%TC_STAGES);
            pgm_cp_commit(); pgm_cp_wait<TC_STAGES-1>(); __syncthreads();
            const int cb=ks%TC_STAGES; __nv_bfloat16* Ad=As+cb*TABUF; __nv_bfloat16* Bd=Bs+cb*TBBUF;
#pragma unroll
            for (int kf=0; kf<TK; kf+=16){
                unsigned af[4]; tc_load_afrag(af,Ad,kf,lane);
                unsigned bf[TNFRAG][2]; tc_load_bfrags(bf,Bd,warp,kf,lane);
#pragma unroll
                for (int nj=0;nj<TNFRAG;nj++) pgm_mma(acc[nj],af,bf[nj],acc[nj]);
            }
            __syncthreads();
        }
        tc_store(acc,C,tm,tn,warp,lane,M,N);
        __syncthreads();
    }
}

/* TC sz GEMM (inline V3 expand in the ring) */
__global__ void __launch_bounds__(256,1) k_tc_sz(__nv_bfloat16* __restrict__ C,
        const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ lo,
        const uint8_t* __restrict__ cd, const unsigned* __restrict__ eoff,
        const unsigned* __restrict__ epos, const __nv_bfloat16* __restrict__ eval,
        unsigned base, unsigned M, unsigned N, unsigned K) {
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    uint8_t* los = (uint8_t*)(As + TC_STAGES * TABUF);
    uint8_t* cds = los + TC_STAGES * TLOBUF;
    __nv_bfloat16* Bbf = (__nv_bfloat16*)(((size_t)(cds + TC_STAGES * TCDBUF) + 15) & ~size_t(15));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_m = (M + TM - 1) / TM, tiles_n = (N + TN - 1) / TN;
    const int ntiles = tiles_m * tiles_n, ksteps = (K + TK - 1) / TK;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int tm = (tile / tiles_n) * TM, tn = (tile % tiles_n) * TN;
        float acc[TNFRAG][4];
#pragma unroll
        for (int j=0;j<TNFRAG;j++) for(int e=0;e<4;e++) acc[j][e]=0.f;
        auto stage=[&](int ks,int buf){ tc_stage_a(As+buf*TABUF,A,tid,tm,ks*TK,M,K);
            tc_stage_b_sz(los+buf*TLOBUF,cds+buf*TCDBUF,lo,cd,tid,tn,ks*TK,N,K); };
#pragma unroll
        for (int s=0;s<TC_STAGES-1;s++){ if(s<ksteps) stage(s,s); pgm_cp_commit(); }
        for (int ks=0; ks<ksteps; ks++){
            const int fetch=ks+TC_STAGES-1; if(fetch<ksteps) stage(fetch,fetch%TC_STAGES);
            pgm_cp_commit(); pgm_cp_wait<TC_STAGES-1>(); __syncthreads();
            const int cb=ks%TC_STAGES; __nv_bfloat16* Ad=As+cb*TABUF;
            tc_expand_b_sz(Bbf, los+cb*TLOBUF, cds+cb*TCDBUF, tid, tn, ks*TK, N, K, base, eoff, epos, eval);
            __syncthreads();
#pragma unroll
            for (int kf=0; kf<TK; kf+=16){
                unsigned af[4]; tc_load_afrag(af,Ad,kf,lane);
                unsigned bf[TNFRAG][2]; tc_load_bfrags(bf,Bbf,warp,kf,lane);
#pragma unroll
                for (int nj=0;nj<TNFRAG;nj++) pgm_mma(acc[nj],af,bf[nj],acc[nj]);
            }
            __syncthreads();
        }
        tc_store(acc,C,tm,tn,warp,lane,M,N);
        __syncthreads();
    }
}

/* ============ REGISTER-STAGED sz GEMM: expand DIRECTLY into mma B-fragment regs ============
 * Removes the bf16 convert tile and its 2 syncs: cp.async compressed->smem (once), then each
 * lane loads ONLY its own mma-fragment's lo/cd bytes smem->reg and reconstructs register->
 * register into the exact m16n8k16 B-fragment (b0,b1). No ldmatrix, no expand pass, no second
 * sync. int reconstruct overlaps the mma via ILP (different pipes). Bit-exact gate: output ==
 * k_tc_bf16 (validates the hand-built B-fragment layout).
 *
 * m16n8k16 B-fragment (per PTX ISA, groupID=lane/4 -> n in [0,8), tig=lane%4 -> k-pair):
 *   b0 = {B[k=tig*2 ][n=groupID], B[k=tig*2+1][n=groupID]}
 *   b1 = {B[k=tig*2+8][n=groupID], B[k=tig*2+9][n=groupID]}  (k within the 16-wide mma step) */
__device__ __forceinline__ unsigned recon_pair(unsigned L /*2 lo bytes @b0,b1*/, unsigned cbyte,
                                               unsigned baseK) {
    const unsigned spread = __byte_perm(L, 0u, 0x4140u);           /* lo0@b0, lo1@b2 */
    const unsigned b16 = (spread & 0x007F007Fu) | ((spread << 8) & 0x80008000u);
    const unsigned e = (cbyte & 0xFu) | ((cbyte & 0xF0u) << 12);   /* c0@b0, c1@b2 */
    return b16 | ((e + baseK) << 7);
}
__global__ void __launch_bounds__(256,1) k_tc_sz_reg(__nv_bfloat16* __restrict__ C,
        const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ lo,
        const uint8_t* __restrict__ cd, const unsigned* __restrict__ eoff,
        const unsigned* __restrict__ epos, const __nv_bfloat16* __restrict__ eval,
        unsigned base, unsigned M, unsigned N, unsigned K) {
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    uint8_t* los = (uint8_t*)(As + TC_STAGES * TABUF);
    uint8_t* cds = los + TC_STAGES * TLOBUF;                        /* NO Bbf convert tile */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int groupID = lane >> 2, tig = lane & 3;
    const unsigned baseK = base | (base << 16);
    const int tiles_m = (M + TM - 1) / TM, tiles_n = (N + TN - 1) / TN;
    const int ntiles = tiles_m * tiles_n, ksteps = (K + TK - 1) / TK;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int tm = (tile / tiles_n) * TM, tn = (tile % tiles_n) * TN;
        float acc[TNFRAG][4];
#pragma unroll
        for (int j=0;j<TNFRAG;j++) for(int e=0;e<4;e++) acc[j][e]=0.f;
        auto stage=[&](int ks,int buf){ tc_stage_a(As+buf*TABUF,A,tid,tm,ks*TK,M,K);
            tc_stage_b_sz(los+buf*TLOBUF,cds+buf*TCDBUF,lo,cd,tid,tn,ks*TK,N,K); };
#pragma unroll
        for (int s=0;s<TC_STAGES-1;s++){ if(s<ksteps) stage(s,s); pgm_cp_commit(); }
        for (int ks=0; ks<ksteps; ks++){
            const int fetch=ks+TC_STAGES-1; if(fetch<ksteps) stage(fetch,fetch%TC_STAGES);
            pgm_cp_commit(); pgm_cp_wait<TC_STAGES-1>(); __syncthreads();
            const int cb=ks%TC_STAGES; __nv_bfloat16* Ad=As+cb*TABUF;
            uint8_t* lob=los+cb*TLOBUF; uint8_t* cdb=cds+cb*TCDBUF;
#pragma unroll
            for (int kf=0; kf<TK; kf+=16){
                unsigned af[4]; tc_load_afrag(af,Ad,kf,lane);
                unsigned bf[TNFRAG][2];
#pragma unroll
                for (int nj=0;nj<TNFRAG;nj++){
                    const int nl = warp*TWN + nj*8 + groupID;      /* tile row */
                    const int k0 = kf + tig*2, k1 = kf + tig*2 + 8;
                    const unsigned L0 = *(const unsigned short*)(lob + nl*TK + k0);
                    const unsigned L1 = *(const unsigned short*)(lob + nl*TK + k1);
                    const unsigned c0 = cdb[nl*(TK/2) + k0/2];
                    const unsigned c1 = cdb[nl*(TK/2) + k1/2];
                    bf[nj][0] = recon_pair(L0,c0,baseK);
                    bf[nj][1] = recon_pair(L1,c1,baseK);
                    /* escapes: patch any of the 4 (n,k) that fell out of window */
                    const int ng = tn + nl;
                    if (ng < (int)N) {
                        const unsigned e0=eoff[ng], e1=eoff[ng+1];
                        if (e1!=e0){
                            const size_t base_el = (size_t)ng*K + (size_t)ks*TK + tig*2;
                            for(unsigned t=e0;t<e1;++t){ long d=(long)epos[t]-(long)base_el;
                                unsigned ev=*(const unsigned short*)&eval[t];
                                if(d==0) bf[nj][0]=(bf[nj][0]&0xFFFF0000u)|ev;
                                else if(d==1) bf[nj][0]=(bf[nj][0]&0xFFFFu)|(ev<<16);
                                else if(d==8) bf[nj][1]=(bf[nj][1]&0xFFFF0000u)|ev;
                                else if(d==9) bf[nj][1]=(bf[nj][1]&0xFFFFu)|(ev<<16);
                            }
                        }
                    }
                }
#pragma unroll
                for (int nj=0;nj<TNFRAG;nj++) pgm_mma(acc[nj],af,bf[nj],acc[nj]);
            }
            __syncthreads();
        }
        tc_store(acc,C,tm,tn,warp,lane,M,N);
        __syncthreads();
    }
}

/* ---- FFMA references (C-1R): gemv_rows / gemv_rows_sz at the same geometry ---- */
template<int MM> __global__ void k_ff_bf16(__nv_bfloat16* C, const __nv_bfloat16* x,
        const __nv_bfloat16* W, unsigned M, unsigned N, unsigned K){
    gemv_rows<MM>(C,x,W,M,N,K,blockIdx.x,gridDim.x);
}
template<int MM> __global__ void k_ff_sz(__nv_bfloat16* C, const __nv_bfloat16* x,
        const uint8_t* lo, const uint8_t* cd, const unsigned* eoff, const unsigned* epos,
        const __nv_bfloat16* eval, unsigned base, unsigned M, unsigned N, unsigned K){
    gemv_rows_sz<MM>(C,x,lo,cd,eoff,epos,eval,base,M,N,K,blockIdx.x,gridDim.x);
}

static unsigned GRID=188; static const unsigned BLOCK=256, EXP_BASE=109;

struct Comp{ std::vector<uint8_t> lo,cd; std::vector<unsigned> eoff,epos; std::vector<uint16_t> eval; };
static Comp compress(const uint16_t* s,size_t n,size_t K){
    Comp c; c.lo.resize(n); c.cd.assign(n/2,0); size_t nch=n/K; c.eoff.assign(nch+1,0);
    for(size_t i=0;i<n;++i){ uint16_t u=s[i]; unsigned ex=(u>>7)&0xFF;
        c.lo[i]=(uint8_t)(((u>>8)&0x80)|(u&0x7F)); int code=0;
        if(ex>=EXP_BASE&&ex<=EXP_BASE+15) code=(int)ex-(int)EXP_BASE;
        else{ c.epos.push_back((unsigned)i); c.eval.push_back(u); c.eoff[i/K]++; }
        c.cd[i/2]|=(uint8_t)(code<<((i&1)*4)); }
    unsigned run=0; for(size_t k=0;k<=nch;++k){unsigned t=k<nch?c.eoff[k]:0;c.eoff[k]=run;run+=t;}
    return c;
}

/* device buffers reused across B for a shape */
struct Dev{ __nv_bfloat16 *dW,*dC,*dx; uint8_t *dlo,*dcd; unsigned *deoff,*depos; __nv_bfloat16* deval;
            size_t ntot; int N,K,L,NL; };

static void alloc_shape(Dev& d, int N,int K,int L, const std::vector<uint16_t>& src){
    size_t nper=(size_t)N*K, ntot=nper*L, nsrc=src.size(); d.N=N;d.K=K;d.L=L;d.NL=N*L;d.ntot=ntot;
    std::vector<uint16_t> W(ntot); for(size_t i=0;i<ntot;++i) W[i]=src[i%nsrc];
    Comp c=compress(W.data(),ntot,(size_t)K);
    CK(cudaMalloc(&d.dW,ntot*2)); CK(cudaMalloc(&d.dlo,ntot)); CK(cudaMalloc(&d.dcd,ntot/2));
    CK(cudaMalloc(&d.deoff,c.eoff.size()*4)); CK(cudaMalloc(&d.depos,c.epos.size()*4+4));
    CK(cudaMalloc(&d.deval,c.eval.size()*2+2)); CK(cudaMalloc(&d.dx,(size_t)32*K*2));
    CK(cudaMalloc(&d.dC,(size_t)d.NL*32*2));
    CK(cudaMemcpy(d.dW,W.data(),ntot*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.dlo,c.lo.data(),ntot,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.dcd,c.cd.data(),ntot/2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.deoff,c.eoff.data(),c.eoff.size()*4,cudaMemcpyHostToDevice));
    if(c.epos.size())CK(cudaMemcpy(d.depos,c.epos.data(),c.epos.size()*4,cudaMemcpyHostToDevice));
    if(c.eval.size())CK(cudaMemcpy(d.deval,c.eval.data(),c.eval.size()*2,cudaMemcpyHostToDevice));
    std::vector<uint16_t> hx((size_t)32*K); for(size_t i=0;i<hx.size();++i) hx[i]=src[(i*7919)%nsrc];
    CK(cudaMemcpy(d.dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
}
static void free_shape(Dev& d){ cudaFree(d.dW);cudaFree(d.dC);cudaFree(d.dx);cudaFree(d.dlo);
    cudaFree(d.dcd);cudaFree(d.deoff);cudaFree(d.depos);cudaFree(d.deval); }

static unsigned tc_smem_bf16(){ return TC_STAGES*(TABUF+TBBUF)*2u; }
static unsigned tc_smem_sz(){ return TC_STAGES*TABUF*2u + TC_STAGES*(TLOBUF+TCDBUF) + TBBUF*2u + 16u; }
static unsigned tc_smem_szreg(){ return TC_STAGES*TABUF*2u + TC_STAGES*(TLOBUF+TCDBUF); } /* no Bbf */

/* JSON accumulation */
static std::string g_json;
static void jrow(const char* shape,int B,double tcbf,double tcsz,double ffbf,double tcszr,
                 int badsz,double ratio){
    char buf[512];
    snprintf(buf,sizeof buf,
      "%s{\"shape\":\"%s\",\"B\":%d,\"tc_bf16_gbs\":%.1f,\"tc_sz_gbs\":%.1f,"
      "\"ff_bf16_gbs\":%.1f,\"tc_sz_reg_gbs\":%.1f,\"tc_sz_vs_tcbf\":%.4f,"
      "\"tc_szreg_vs_tcbf\":%.4f,\"tc_szreg_vs_ffbf\":%.4f,\"tc_bf16_vs_ffbf\":%.4f,"
      "\"bitexact\":%s,\"ratio\":%.4f}",
      g_json.empty()?"":",\n", shape,B,tcbf,tcsz,ffbf,tcszr,
      tcsz/tcbf, tcszr/tcbf, tcszr/ffbf, tcbf/ffbf, badsz==0?"true":"false", ratio);
    g_json += buf;
}

template<int MM>
static void run_B(Dev& d, const char* shape, int iters, cudaEvent_t e0, cudaEvent_t e1){
    const int B=MM; const int N=d.NL,K=d.K; const size_t outN=(size_t)N*B;
    double logical=(double)d.ntot*2.0; /* weight bytes streamed */
    unsigned sbf=tc_smem_bf16(), ssz=tc_smem_sz(), sszr=tc_smem_szreg();
    CK(cudaFuncSetAttribute(k_tc_bf16,cudaFuncAttributeMaxDynamicSharedMemorySize,sbf));
    CK(cudaFuncSetAttribute(k_tc_sz,cudaFuncAttributeMaxDynamicSharedMemorySize,ssz));
    CK(cudaFuncSetAttribute(k_tc_sz_reg,cudaFuncAttributeMaxDynamicSharedMemorySize,sszr));
    auto Lffbf=[&](){ k_ff_bf16<MM><<<GRID,BLOCK>>>(d.dC,d.dx,d.dW,B,N,K); };
    auto Ltcbf=[&](){ k_tc_bf16<<<GRID,BLOCK,sbf>>>(d.dC,d.dx,d.dW,B,N,K); };
    auto Ltcsz=[&](){ k_tc_sz<<<GRID,BLOCK,ssz>>>(d.dC,d.dx,d.dlo,d.dcd,d.deoff,d.depos,d.deval,EXP_BASE,B,N,K); };
    auto Ltcszr=[&](){ k_tc_sz_reg<<<GRID,BLOCK,sszr>>>(d.dC,d.dx,d.dlo,d.dcd,d.deoff,d.depos,d.deval,EXP_BASE,B,N,K); };
    auto bench=[&](auto fn){ for(int i=0;i<3;i++) fn(); CK(cudaDeviceSynchronize());CK(cudaGetLastError());
        CK(cudaEventRecord(e0)); for(int i=0;i<iters;i++) fn(); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));CK(cudaGetLastError()); float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
        return (double)ms/iters; };
    std::vector<uint16_t> ytcbf(outN), ytcsz(outN), ytcszr(outN);
    double mtcbf=bench(Ltcbf); CK(cudaMemcpy(ytcbf.data(),d.dC,outN*2,cudaMemcpyDeviceToHost));
    double mtcsz=bench(Ltcsz); CK(cudaMemcpy(ytcsz.data(),d.dC,outN*2,cudaMemcpyDeviceToHost));
    double mtcszr=bench(Ltcszr); CK(cudaMemcpy(ytcszr.data(),d.dC,outN*2,cudaMemcpyDeviceToHost));
    double mffbf=bench(Lffbf);
    size_t badsz=0,badszr=0; for(size_t i=0;i<outN;++i){badsz+=(ytcbf[i]!=ytcsz[i]);badszr+=(ytcbf[i]!=ytcszr[i]);}
    double gbf=logical/1e9/(mtcbf/1e3), gsz=logical/1e9/(mtcsz/1e3);
    double gszr=logical/1e9/(mtcszr/1e3), gffbf=logical/1e9/(mffbf/1e3);
    double ratio = logical / ((double)d.ntot + d.ntot/2.0);
    printf("%-14s B=%-2d | TCbf %7.1f | TCsz %7.1f (%.3fx) | TCszREG %7.1f (%.3fx) | FFbf %7.1f | TCr/FFbf %.3fx | sz=%s(%zu) szr=%s(%zu)\n",
        shape,B,gbf,gsz,gsz/gbf,gszr,gszr/gbf,gffbf,gszr/gffbf,
        badsz==0?"OK":"BAD",badsz, badszr==0?"OK":"BAD",badszr);
    jrow(shape,B,gbf,gsz,gffbf,gszr,(int)(badsz+badszr),ratio);
}

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"/tmp/g12_sample.bin"; int iters=argc>2?atoi(argv[2]):60;
    if(const char* g=getenv("SZ_GRID")) GRID=atoi(g);
    FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);return 1;}
    fseek(f,0,SEEK_END);size_t nb=ftell(f);fseek(f,0,SEEK_SET); std::vector<uint16_t> src(nb/2);
    if(fread(src.data(),2,src.size(),f)!=src.size())return 1; fclose(f);
    printf("TENSOR-CORE small-M decode GEMM (BM=%d) vs bf16 vs FFMA @ 12B shapes, GRID=%u, EXP_BASE=%u\n",TM,GRID,EXP_BASE);
    printf("smem: TC-bf16=%uB TC-sz=%uB  STAGES=%d\n\n",tc_smem_bf16(),tc_smem_sz(),TC_STAGES);
    struct Sh{const char*nm;int N,K,L;};
    Sh sh[]={{"qkv    K3840",6144,3840,8},{"o_proj K4096",3840,4096,8},
             {"gate/up K3840",15360,3840,4},{"down   K15360",3840,15360,4}};
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for(auto& s: sh){
        Dev d; alloc_shape(d,s.N,s.K,s.L,src);
        run_B<1>(d,s.nm,iters,e0,e1);  run_B<2>(d,s.nm,iters,e0,e1);
        run_B<4>(d,s.nm,iters,e0,e1);  run_B<8>(d,s.nm,iters,e0,e1);
        run_B<16>(d,s.nm,iters,e0,e1); run_B<32>(d,s.nm,iters,e0,e1);
        printf("\n"); free_shape(d);
    }
    FILE* j=fopen("/tmp/c1t_tc.json","w"); if(j){ fprintf(j,"[\n%s\n]\n",g_json.c_str()); fclose(j); }
    return 0;
}
