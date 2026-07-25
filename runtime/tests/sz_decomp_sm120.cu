/* sz_decomp_sm120.cu — C-1T Thread 2: SplitZip decompressor (sz_expand8) instruction tuning.
 *
 * Isolated microbench of decode-only reconstruct variants, each gated BIT-EXACT vs the current
 * scalar sz_expand8 output (which is itself bit-exact bf16). Two measurements per variant:
 *   (1) correctness: expand a real [R][K] weight block -> bf16, memcmp vs original bf16 bytes.
 *   (2) throughput: ALU-bound tight loop, compressed operands held in REGISTERS across ITER
 *       iterations (no HBM traffic), XOR-sink to defeat DCE -> elems/ns + expanded GB/s.
 * ptxas regs/spill printed at build (-Xptxas -v). SASS op count via cuobjdump (offline).
 *
 * Variants:
 *   V0 baseline  : current scalar per-elem (op_gemm.cuh sz_expand8), 8x scalar loop.
 *   V1 lop3      : 3-way sign|exp|mantissa merge via one lop3.b32 (a|b|c), else scalar.
 *   V2 prmt      : __byte_perm byte spread for the lo->base16 plane; exp still shifted.
 *   V3 packed    : fully packed 32-bit, 2 bf16 per u32 lane, minimal ops (the contender).
 *   V4 relayout  : PRE-SPLIT byte planes (exp0 merged into the low byte, sign+exp[7:1] a 4b
 *                  code over an EVEN base) -> decode is byte-assembly (prmt), no <<7 cross shift.
 *                  Changes the offline layout; measures the compression-ratio cost.
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_120a -Xptxas -v -Iinclude -Iruntime/common -Iruntime/nvidia \
 *          runtime/tests/sz_decomp_sm120.cu -o /tmp/c1t_decomp -lcuda
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

/* ---------------- device reconstruct variants (lo 8B + cd 4B -> bf16v8) ---------------- */

/* V0: exact copy of op_gemm.cuh sz_expand8 arithmetic (the current production form). */
__device__ __forceinline__ bf16v8 expand_v0(const uint2 lb, const unsigned cw, unsigned base) {
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

__device__ __forceinline__ unsigned lop3_or3(unsigned a, unsigned b, unsigned c) {
    unsigned d;
    asm("lop3.b32 %0, %1, %2, %3, 0xFE;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c)); /* a|b|c */
    return d;
}
/* V1: same field extraction, but the 3-way merge is one lop3.b32 instead of two ORs. */
__device__ __forceinline__ bf16v8 expand_v1(const uint2 lb, const unsigned cw, unsigned base) {
    const unsigned lw[2] = {lb.x, lb.y};
    unsigned short us[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned b = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
        const unsigned ex = ((cw >> (e * 4)) & 0xFu) + base;
        us[e] = (unsigned short)lop3_or3((b & 0x80u) << 8, ex << 7, b & 0x7Fu);
    }
    bf16v8 r;
#pragma unroll
    for (int e = 0; e < 8; e++) r.x[e] = __ushort_as_bfloat16(us[e]);
    return r;
}

/* V2: prmt (__byte_perm) spreads the 4 lo bytes of a u32 into two u32s (base16 planes), the
 * sign is repositioned with a shift, the exp path stays scalar-ish per pair. Half the lanes. */
__device__ __forceinline__ unsigned base16_pair(unsigned L /*byte0=lo0, byte1=lo1*/) {
    /* spread: bits7:0=lo0, bits23:16=lo1 */
    const unsigned spread = __byte_perm(L, 0u, 0x4140u); /* out: lo0,0,lo1,0 */
    const unsigned mant = spread & 0x007F007Fu;
    const unsigned sign = (spread << 8) & 0x80008000u;   /* bit7->15, bit23->31 */
    return mant | sign;
}
__device__ __forceinline__ bf16v8 expand_v2(const uint2 lb, const unsigned cw, unsigned base) {
    const unsigned lw[2] = {lb.x, lb.y};
    unsigned res[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const unsigned L = (lw[p >> 1] >> ((p & 1) * 16)) & 0xFFFFu;
        const unsigned b16 = base16_pair(L);
        const unsigned cwp = cw >> (8 * p);
        const unsigned c0 = (cwp & 0xFu) + base, c1 = ((cwp >> 4) & 0xFu) + base;
        const unsigned expw = (c0 << 7) | (c1 << 23);
        res[p] = b16 | expw;
    }
    bf16v8 r; memcpy(&r, res, 16); return r;
}

/* V3: fully packed 32-bit, 2 bf16 per lane, no per-elem scalar loop. */
__device__ __forceinline__ bf16v8 expand_v3(const uint2 lb, const unsigned cw, unsigned base) {
    const unsigned lw[2] = {lb.x, lb.y};
    const unsigned baseK = base | (base << 16);
    unsigned res[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const unsigned L = (lw[p >> 1] >> ((p & 1) * 16)) & 0xFFFFu;
        /* base16 (mantissa + repositioned sign) for the 2 elems */
        const unsigned spread = __byte_perm(L, 0u, 0x4140u);         /* lo0 @b0, lo1 @b2 */
        const unsigned b16 = (spread & 0x007F007Fu) | ((spread << 8) & 0x80008000u);
        /* exp: c0 @bits3:0, c1 @bits19:16 ; +base packed ; <<7 lands exp fields in both slots */
        const unsigned cwp = cw >> (8 * p);
        const unsigned e = (cwp & 0xFu) | ((cwp & 0xF0u) << 12);
        const unsigned eb = e + baseK;
        const unsigned expw = eb << 7;
        res[p] = b16 | expw;
    }
    bf16v8 r; memcpy(&r, res, 16); return r;
}

/* V4: PRE-SPLIT layout. lo2 byte = (exp&1)<<7 | mant7 (true low byte). code4 = sign<<3 |
 * (exp>>1 - baseHi), over an EVEN window base. Decode: hi byte = (sign<<7)|(baseHi+off),
 * bf16 = (hi<<8)|lo2 — byte assembly, NO <<7 cross-byte shift. */
__device__ __forceinline__ bf16v8 expand_v4(const uint2 lb2 /*lo2 bytes*/, const unsigned cw,
                                            unsigned baseHi) {
    const unsigned lw[2] = {lb2.x, lb2.y};
    const unsigned baseHiK = baseHi | (baseHi << 16);
    unsigned res[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const unsigned L = (lw[p >> 1] >> ((p & 1) * 16)) & 0xFFFFu;  /* lo2_0 @b0, lo2_1 @b1 */
        const unsigned cwp = cw >> (8 * p);
        /* offsets (3b) + sign (1b) per elem */
        const unsigned off = ((cwp & 0x7u)) | (((cwp >> 4) & 0x7u) << 16);
        const unsigned sgn = ((cwp & 0x8u) << 4) | (((cwp >> 4) & 0x8u) << 20); /* ->bit7,bit23 */
        const unsigned hi = (off + baseHiK) | sgn;   /* hi byte of each elem @b0,b2 */
        /* assemble bf16: want bytes [lo0,hi0,lo1,hi1]. hi already has hi0@b0, hi1@b2;
         * spread lo2 bytes to b0,b2 then interleave hi shifted up one byte. */
        const unsigned lop = __byte_perm(L, 0u, 0x4140u);   /* lo0 @b0, lo1 @b2 */
        res[p] = lop | (hi << 8);                           /* [lo0,hi0,lo1,hi1] */
    }
    bf16v8 r; memcpy(&r, res, 16); return r;
}

/* ---------------- host compress (V0-V3 layout) and V4 layout ---------------- */
static const unsigned EXP_BASE = 109;      /* audited window for [109,124] */
static const unsigned V4_BASE  = 108;      /* EVEN base for the pre-split hi code, window [108,123] */

struct Comp { std::vector<uint8_t> lo, cd; unsigned nesc; };
static Comp compressV03(const uint16_t* s, size_t n) {
    Comp c; c.lo.resize(n); c.cd.assign(n/2,0); c.nesc=0;
    for (size_t i=0;i<n;++i){ uint16_t u=s[i]; unsigned ex=(u>>7)&0xFF;
        c.lo[i]=(uint8_t)(((u>>8)&0x80)|(u&0x7F)); int code=0;
        if(ex>=EXP_BASE&&ex<=EXP_BASE+15) code=(int)ex-(int)EXP_BASE; else c.nesc++;
        c.cd[i/2]|=(uint8_t)(code<<((i&1)*4)); }
    return c;
}
/* V4: lo2 = (exp&1)<<7|mant7 ; code = sign<<3 | (exp>>1 - baseHi), baseHi=V4_BASE>>1 */
static Comp compressV4(const uint16_t* s, size_t n) {
    Comp c; c.lo.resize(n); c.cd.assign(n/2,0); c.nesc=0;
    const unsigned baseHi=V4_BASE>>1;
    for (size_t i=0;i<n;++i){ uint16_t u=s[i]; unsigned ex=(u>>7)&0xFF; unsigned sign=(u>>15)&1;
        unsigned mant=u&0x7F; c.lo[i]=(uint8_t)(((ex&1)<<7)|mant); int code=0;
        if(ex>=V4_BASE&&ex<=V4_BASE+15){ unsigned off=(ex>>1)-baseHi; code=(int)((sign<<3)|off); }
        else c.nesc++;
        c.cd[i/2]|=(uint8_t)(code<<((i&1)*4)); }
    return c;
}

/* ---------------- correctness kernels (expand [R][K] block -> bf16) ---------------- */
enum { MODE_V0=0, MODE_V1, MODE_V2, MODE_V3, MODE_V4 };
__global__ void k_expand(uint16_t* out, const uint8_t* lo, const uint8_t* cd, unsigned n,
                         unsigned base, int mode) {
    unsigned i8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i8 >= n) return;
    const uint2 lb = *(const uint2*)(lo + i8);
    const unsigned cw = *(const unsigned*)(cd + i8/2);
    bf16v8 r;
    switch (mode) {
        case MODE_V0: r = expand_v0(lb, cw, base); break;
        case MODE_V1: r = expand_v1(lb, cw, base); break;
        case MODE_V2: r = expand_v2(lb, cw, base); break;
        case MODE_V3: r = expand_v3(lb, cw, base); break;
        default:      r = expand_v4(lb, cw, base); break;
    }
    *(uint4*)(out + i8) = *(const uint4*)&r;
}

/* ---------------- throughput kernels (ALU-bound, operands in registers) ---------------- */
template <int MODE, int ITER>
__global__ void k_thru(uint32_t* sink, const uint8_t* lo, const uint8_t* cd, unsigned base) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint2 lb = *(const uint2*)(lo + (tid & 1023) * 8);      /* small resident footprint */
    const unsigned cw = *(const unsigned*)(cd + (tid & 1023) * 4);
    uint4 acc = make_uint4(0,0,0,0);
#pragma unroll 8
    for (int t = 0; t < ITER; t++) {
        /* perturb inputs by the (independent) loop index so the expand is neither loop-invariant
         * (would be hoisted) nor serialized (a per-iter feedback chain would measure latency). */
        const uint2 lbt = make_uint2(lb.x ^ (unsigned)t, lb.y);
        const unsigned cwt = cw ^ (unsigned)t;
        bf16v8 r;
        if (MODE==MODE_V0) r = expand_v0(lbt, cwt, base);
        else if (MODE==MODE_V1) r = expand_v1(lbt, cwt, base);
        else if (MODE==MODE_V2) r = expand_v2(lbt, cwt, base);
        else if (MODE==MODE_V3) r = expand_v3(lbt, cwt, base);
        else r = expand_v4(lbt, cwt, base);
        uint4 v = *(const uint4*)&r;
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x|acc.y|acc.z|acc.w)==0xFFFFFFFFu) sink[tid] = acc.x; /* never taken; keeps acc live */
    else if (tid==0xFFFFFFFF) sink[0]=acc.y;
}

static const int TITER = 4096;
static const unsigned TBLK = 256, TGRID = 4096; /* 1.05M threads * 8 * ITER elems */

template <int MODE> static double thru(uint32_t* sink, const uint8_t* lo, const uint8_t* cd,
                                       unsigned base, cudaEvent_t e0, cudaEvent_t e1, int reps) {
    auto L=[&](){ k_thru<MODE,TITER><<<TGRID,TBLK>>>(sink,lo,cd,base); };
    for(int i=0;i<3;i++) L(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for(int i=0;i<reps;i++) L(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
    return (double)ms/reps;
}

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"/tmp/g12_sample.bin"; int reps=argc>2?atoi(argv[2]):50;
    FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);return 1;}
    fseek(f,0,SEEK_END);size_t nb=ftell(f);fseek(f,0,SEEK_SET); std::vector<uint16_t> src(nb/2);
    if(fread(src.data(),2,src.size(),f)!=src.size())return 1; fclose(f);
    size_t n = src.size() & ~size_t(7);
    printf("SZ decompressor variants @ %zu real bf16 elems, EXP_BASE=%u V4_BASE=%u\n\n", n, EXP_BASE, V4_BASE);

    Comp c03 = compressV03(src.data(), n);
    Comp c4  = compressV4(src.data(), n);
    /* ratio: 12 b/elem fixed + escape metadata (u32 pos + u16 raw = 6 B/escape) */
    auto ratio=[&](const Comp& c){ double logical=n*2.0, comp=n+n/2.0+c.nesc*6.0; return logical/comp; };
    double r03=ratio(c03), r4=ratio(c4);
    printf("layout V0-V3: escapes=%u (%.4f%%) ratio=%.4fx\n", c03.nesc, 100.0*c03.nesc/n, r03);
    printf("layout V4   : escapes=%u (%.4f%%) ratio=%.4fx  (window shift 109->108 cost)\n\n",
           c4.nesc, 100.0*c4.nesc/n, r4);

    uint8_t *dlo,*dcd,*dlo4,*dcd4; uint16_t* dout; uint32_t* dsink;
    CK(cudaMalloc(&dlo,n)); CK(cudaMalloc(&dcd,n/2));
    CK(cudaMalloc(&dlo4,n)); CK(cudaMalloc(&dcd4,n/2));
    CK(cudaMalloc(&dout,n*2)); CK(cudaMalloc(&dsink,(size_t)TGRID*TBLK*4));
    CK(cudaMemcpy(dlo,c03.lo.data(),n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcd,c03.cd.data(),n/2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlo4,c4.lo.data(),n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcd4,c4.cd.data(),n/2,cudaMemcpyHostToDevice));

    /* correctness: expand -> memcmp vs original bf16 (only in-window elems; escapes handled
     * separately in production. We verify the fast path over the in-window majority). */
    std::vector<uint16_t> hout(n);
    auto check=[&](int mode, const char* nm, uint8_t* lo, uint8_t* cd, unsigned base){
        unsigned blocks=(n/8+TBLK-1)/TBLK;
        k_expand<<<blocks,TBLK>>>(dout,lo,cd,n,base,mode);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(hout.data(),dout,n*2,cudaMemcpyDeviceToHost));
        /* compare vs original, skipping escaped positions (out-of-window exps) */
        size_t bad=0, checked=0;
        for(size_t i=0;i<n;++i){ unsigned ex=(src[i]>>7)&0xFF; bool inwin;
            if(mode==MODE_V4) inwin=(ex>=V4_BASE&&ex<=V4_BASE+15); else inwin=(ex>=EXP_BASE&&ex<=EXP_BASE+15);
            if(!inwin) continue; checked++; if(hout[i]!=src[i]) bad++; }
        printf("  %-3s bit-exact(in-window): %s  (%zu bad / %zu checked)\n",
               nm, bad==0?"OK":"MISMATCH", bad, checked);
        return bad==0;
    };
    printf("correctness (expand real weights, in-window):\n");
    check(MODE_V0,"V0",dlo,dcd,EXP_BASE); check(MODE_V1,"V1",dlo,dcd,EXP_BASE);
    check(MODE_V2,"V2",dlo,dcd,EXP_BASE); check(MODE_V3,"V3",dlo,dcd,EXP_BASE);
    check(MODE_V4,"V4",dlo4,dcd4,V4_BASE>>1);

    /* throughput */
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double totElems = (double)TGRID*TBLK*8.0*TITER;
    printf("\nthroughput (ALU-bound, ops in registers, %d reps):\n", reps);
    struct R{const char*nm;double ms;};
    R rr[5];
    rr[0]={"V0",thru<MODE_V0>(dsink,dlo,dcd,EXP_BASE,e0,e1,reps)};
    rr[1]={"V1",thru<MODE_V1>(dsink,dlo,dcd,EXP_BASE,e0,e1,reps)};
    rr[2]={"V2",thru<MODE_V2>(dsink,dlo,dcd,EXP_BASE,e0,e1,reps)};
    rr[3]={"V3",thru<MODE_V3>(dsink,dlo,dcd,EXP_BASE,e0,e1,reps)};
    rr[4]={"V4",thru<MODE_V4>(dsink,dlo4,dcd4,V4_BASE>>1,e0,e1,reps)};
    double v0e=totElems/(rr[0].ms*1e6);
    for(int i=0;i<5;i++){ double ens=totElems/(rr[i].ms*1e6); /* elems/ns */
        double gbs=totElems*2.0/(rr[i].ms*1e6); /* expanded bytes/s: elems*2 / ns */
        printf("  %-3s %8.3f ms  %8.2f elems/ns  %8.1f GB/s(expanded)  %.3fx vs V0\n",
               rr[i].nm, rr[i].ms, ens, gbs, ens/v0e);
    }
    cudaFree(dlo);cudaFree(dcd);cudaFree(dlo4);cudaFree(dcd4);cudaFree(dout);cudaFree(dsink);
    return 0;
}
