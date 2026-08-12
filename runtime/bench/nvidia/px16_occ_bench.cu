/* px16_occ_bench.cu — what is OCCUPANCY 2 worth to the flash DECODE kernel?
 *
 * PX-7 measured occ 2 at ~1.05x on the PREFILL GEMM (compute-bound, register-limited to occ 1).
 * Nobody measured it for DECODE, which PX-10/PX-11 show is latency/map-bound at 527-608 GB/s
 * against a 1700 GB/s measured wall. Someone proposed spending real engineering to relocate
 * ~110 KB of registers into idle smem to reach occ 2. This bench prices the PRIZE first.
 *
 * Three probes, cheapest-decisive first:
 *
 *   A  FREE-OCCUPANCY UPPER BOUND. The score phase's access map (one 512 B fp8 KV row per
 *      THREAD) in a kernel whose natural register count is ~40, so occupancy 1/2/3/4 costs
 *      NOTHING -- no spills, no launch_bounds tax. This is the most occ 2 could ever buy on
 *      this map. If it is ~1.0x here, no register work can be worth anything, full stop.
 *      (PX-11 rung 0c did this for the LINEAR map and got 1.00x; the row map is the one
 *      running at 74% of the wall, so it is the one that might have headroom.)
 *
 *   B  THE REAL TRADEOFF. The SHIPPED d_flash_decode<512,GF,true>, compiled once at its
 *      natural register count (occ 1) and once under __maxnreg__(128) (occ 2, WITH whatever
 *      spills ptxas needs). Spilled-but-occ-2 vs unspilled-but-occ-1 is the actual decision.
 *
 *   C  THE CURVE. Same kernel swept over __maxnreg__, reporting the ptxas-achieved register
 *      count, the spill (local) bytes, the runtime-computed blocks/SM, and GB/s. The curve,
 *      not intuition, is the answer.
 *
 * Every arm prints registers / spill bytes / blocks-per-SM from cudaFuncGetAttributes and
 * cudaOccupancyMaxActiveBlocksPerMultiprocessor -- MEASURED, never assumed, because a
 * __maxnreg__ that ptxas ignored and an occupancy that smem blocked look identical in ms.
 *
 * Numerics: register caps only add spill stores, so every arm must be BIT-IDENTICAL to the
 * natural arm. maxdiff is printed as a self-check that the cap did not change the kernel.
 *
 * Build:  bash perf-data/px16_build.sh /tmp/px16 [extra -D...]
 * Run:    perf-data/tools/gpulease px16 bash perf-data/px16_run.sh
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh: d_flash_decode / d_flash_merge */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }

static int g_sm = 170;
static int g_iters = 20;

/* ================================================================ probe A: free-occupancy map */
__device__ unsigned long long g_sink;

/* One whole ROW per THREAD -- the default score-phase map. ROWB=512 is the fp8 hd512 row
 * (and the bf16 hd256 sliding row); ROWB=1024 is the bf16 hd512 row. W bytes/lane, U loads
 * in flight (U>1 is mandatory: PX-11 bug #2 -- without it this is latency-bound and reads
 * ~35% low). MINB is the __launch_bounds__ occupancy floor: it is what forces ptxas to fit
 * the kernel in 65536/(MINB*256) registers, and here the kernel is small enough that every
 * MINB from 1..4 is free. */
template<int ROWB, int W, int U, int MINB>
__global__ void __launch_bounds__(256, MINB)
k_row(const unsigned char* __restrict__ p, size_t nrows) {
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (size_t r = tid; r < nrows; r += nthr) {
        const unsigned char* row = p + r * (size_t)ROWB;
#pragma unroll
        for (int o = 0; o < ROWB; o += W*U) {
            uint4 v4[W==16?U:1]; uint2 v2[W==8?U:1];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const unsigned char* a = row + o + u*W;
                if constexpr (W == 16) v4[u] = __ldcs((const uint4*)a);
                else                   v2[u] = __ldcs((const uint2*)a);
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                if constexpr (W == 16) acc ^= v4[u].x ^ v4[u].y ^ v4[u].z ^ v4[u].w;
                else                   acc ^= v2[u].x ^ v2[u].y;
            }
        }
    }
    if (acc == 0xDEADBEEFu) g_sink += acc;
}

/* Row-GROUP map (the V phase): NDT = ROWB/W lanes cover one row, fully coalesced. */
template<int ROWB, int W, int U, int MINB>
__global__ void __launch_bounds__(256, MINB)
k_rowgrp(const unsigned char* __restrict__ p, size_t nrows) {
    constexpr int NDT = ROWB / W;                 /* lanes per row */
    constexpr int NG  = 256 / NDT;                /* rows per block per step */
    const int ln = threadIdx.x % NDT, gp = threadIdx.x / NDT;
    const size_t ngrp = (size_t)gridDim.x * NG;
    unsigned acc = 0;
    for (size_t r0 = (size_t)blockIdx.x * NG + gp; r0 < nrows; r0 += ngrp * U) {
        uint4 v4[W==16?U:1]; uint2 v2[W==8?U:1];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const size_t r = r0 + (size_t)u * ngrp;
            const unsigned char* a = p + (r < nrows ? r : nrows-1) * (size_t)ROWB + (size_t)ln*W;
            if constexpr (W == 16) v4[u] = __ldcs((const uint4*)a);
            else                   v2[u] = __ldcs((const uint2*)a);
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if constexpr (W == 16) acc ^= v4[u].x ^ v4[u].y ^ v4[u].z ^ v4[u].w;
            else                   acc ^= v2[u].x ^ v2[u].y;
        }
    }
    if (acc == 0xDEADBEEFu) g_sink += acc;
}

/* Linear map, for the ceiling pin (must reproduce PX-11's 1700 GB/s or the rig is wrong). */
template<int W, int U, int MINB>
__global__ void __launch_bounds__(256, MINB)
k_lin(const unsigned char* __restrict__ p, size_t nbytes) {
    const size_t chunk = (size_t)blockDim.x * U * W;
    const size_t step  = chunk * gridDim.x;
    unsigned acc = 0;
    for (size_t base = (size_t)blockIdx.x * chunk; base + chunk <= nbytes; base += step) {
        uint4 v4[W==16?U:1]; uint2 v2[W==8?U:1];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const size_t off = base + ((size_t)u * blockDim.x + threadIdx.x) * W;
            if constexpr (W == 16) v4[u] = __ldcs((const uint4*)(p + off));
            else                   v2[u] = __ldcs((const uint2*)(p + off));
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if constexpr (W == 16) acc ^= v4[u].x ^ v4[u].y ^ v4[u].z ^ v4[u].w;
            else                   acc ^= v2[u].x ^ v2[u].y;
        }
    }
    if (acc == 0xDEADBEEFu) g_sink += acc;
}

/* ============================================================ probe B/C: the SHIPPED kernel */
/* R = __maxnreg__ cap. R=0 is the natural (megakernel) arm: __launch_bounds__(256,1), which
 * is exactly what interp_sm120.cu compiles. Every other R adds ONLY a register cap. */
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_fd_nat(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
         const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
         unsigned window, float scale, unsigned nsplit, unsigned kvmask,
         const float* ks, const float* vs){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,true>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,window,scale,nsplit,
                              kvmask,blockIdx.x,gridDim.x,arena,0u,ks,vs);
}
template<int D,int GF,int R>
__global__ void __launch_bounds__(256) __maxnreg__(R)
k_fd_reg(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
         const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
         unsigned window, float scale, unsigned nsplit, unsigned kvmask,
         const float* ks, const float* vs){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,true>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,window,scale,nsplit,
                              kvmask,blockIdx.x,gridDim.x,arena,0u,ks,vs);
}
template<int D>
__global__ void k_merge(bf16* O, const float* Opart, const float* mlpart,
                        unsigned nb, unsigned nh, unsigned nsplit){
    d_flash_merge<D>(O,Opart,mlpart,nb,nh,nsplit,blockIdx.x,gridDim.x);
}

/* ------------------------------------------------------------------------------- host state */
struct Class { const char* name; int D, NH, NKV, window, ring; };
static const Class C_FULL = { "full", 512, 16, 1, 0, 0 };

struct Pool {
    unsigned char *k8=nullptr, *v8=nullptr;
    float *ks=nullptr, *vs=nullptr;
    bf16 *q=nullptr; int* len=nullptr;
};
static Pool g_p;

static void pool_alloc(const Class& c, int B, unsigned stride){
    const size_t rows = (size_t)B * c.NKV * stride;
    const size_t n = rows * c.D;
    std::vector<unsigned char> h8(n);
    for (size_t i=0;i<n;i++){ float a=rnd()*0.5f; __nv_fp8_e4m3 f(a); h8[i]=*(unsigned char*)&f; }
    CK(cudaMalloc(&g_p.k8,n)); CK(cudaMemcpy(g_p.k8,h8.data(),n,cudaMemcpyHostToDevice));
    for (size_t i=0;i<n;i++){ float a=rnd()*0.5f; __nv_fp8_e4m3 f(a); h8[i]=*(unsigned char*)&f; }
    CK(cudaMalloc(&g_p.v8,n)); CK(cudaMemcpy(g_p.v8,h8.data(),n,cudaMemcpyHostToDevice));
    std::vector<float> sc(rows,1.0f);
    CK(cudaMalloc(&g_p.ks,rows*4)); CK(cudaMemcpy(g_p.ks,sc.data(),rows*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_p.vs,rows*4)); CK(cudaMemcpy(g_p.vs,sc.data(),rows*4,cudaMemcpyHostToDevice));
    std::vector<bf16> hq((size_t)B*c.NH*c.D);
    for (auto& x:hq) x=__float2bfloat16(rnd()*0.5f);
    CK(cudaMalloc(&g_p.q,hq.size()*2)); CK(cudaMemcpy(g_p.q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_p.len,(size_t)B*4));
}
static void pool_free(){
    cudaFree(g_p.k8); cudaFree(g_p.v8); cudaFree(g_p.ks); cudaFree(g_p.vs);
    cudaFree(g_p.q); cudaFree(g_p.len); memset(&g_p,0,sizeof(g_p));
}

static std::vector<float> g_out;

/* Resource + occupancy report for one kernel, straight from the driver. */
struct Res { int regs, spill, occ_blk; size_t stat_smem; };
template<class F>
static Res resources(F f, size_t dyn_smem){
    cudaFuncAttributes a{}; CK(cudaFuncGetAttributes(&a,(const void*)f));
    int blk=0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk,(const void*)f,256,(int)dyn_smem));
    return { a.numRegs, (int)a.localSizeBytes, blk, (size_t)a.sharedSizeBytes };
}

/* Time one flash-decode arm. `gridmul` scales the launch grid: the megakernel launches
 * exactly occ*sm_count blocks (exec/gpu.rs's n_cu gate), so an occ-2 arm MUST run 2*sm
 * blocks for the extra residency to exist at all. */
/* `padsmem` is the OCCUPANCY CONTROL. Comparing grid=1x against grid=2x conflates two
 * things: residency (2 blocks per SM) and wave quantisation (n_work/grid rounding). Padding
 * the dynamic arena past 102400/2 = 51200 B forces blocks/SM back to 1 while the grid stays
 * at 2x, so grid-2x-padded vs grid-2x-natural isolates occupancy ALONE. The kernel only ever
 * indexes the first FA_DEC_SMEM_FLOATS floats, so the pad is inert. */
template<int D,int GF,int R>
static double run_fd(const Class& c, int B, unsigned ctx, unsigned stride, unsigned mask,
                     unsigned nsplit, int gridmul, Res* res, unsigned* grid_out,
                     size_t padsmem = 0){
    const unsigned n_grp = c.NH/GF;
    const unsigned n_work = (unsigned)B*n_grp*nsplit;
    const float scale = 1.f/sqrtf((float)D);
    std::vector<int> hl(B,(int)ctx);
    CK(cudaMemcpy(g_p.len,hl.data(),(size_t)B*4,cudaMemcpyHostToDevice));

    float *dOp=nullptr,*dMl=nullptr; bf16* dO=nullptr;
    CK(cudaMalloc(&dOp,(size_t)B*c.NH*nsplit*D*4));
    CK(cudaMalloc(&dMl,(size_t)B*c.NH*nsplit*2*4));
    CK(cudaMalloc(&dO ,(size_t)B*c.NH*D*2));

    size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    if (padsmem > smem) smem = padsmem;
    unsigned want = (unsigned)(gridmul * g_sm);
    const unsigned grid = (n_work < want) ? n_work : want;

    auto lf=[&](){
        if constexpr (R==0)
            k_fd_nat<D,GF><<<grid,256,smem>>>(dOp,dMl,g_p.q,(bf16*)g_p.k8,(bf16*)g_p.v8,
                g_p.len,(unsigned)B,c.NH,c.NKV,stride,(unsigned)c.window,scale,nsplit,mask,
                g_p.ks,g_p.vs);
        else
            k_fd_reg<D,GF,R><<<grid,256,smem>>>(dOp,dMl,g_p.q,(bf16*)g_p.k8,(bf16*)g_p.v8,
                g_p.len,(unsigned)B,c.NH,c.NKV,stride,(unsigned)c.window,scale,nsplit,mask,
                g_p.ks,g_p.vs);
    };
    if constexpr (R==0) {
        CK(cudaFuncSetAttribute(k_fd_nat<D,GF>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        if (res) *res = resources(k_fd_nat<D,GF>, smem);
    } else {
        CK(cudaFuncSetAttribute(k_fd_reg<D,GF,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        if (res) *res = resources(k_fd_reg<D,GF,R>, smem);
    }

    for (int w=0;w<3;w++) lf();
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best=1e30;
    for (int it=0; it<g_iters; it++){
        CK(cudaEventRecord(a)); lf(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); if (ms<best) best=ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    k_merge<D><<<(unsigned)B*c.NH,256>>>(dO,dOp,dMl,(unsigned)B,c.NH,nsplit);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> o((size_t)B*c.NH*D);
    CK(cudaMemcpy(o.data(),dO,o.size()*2,cudaMemcpyDeviceToHost));
    g_out.resize(o.size());
    for (size_t i=0;i<o.size();i++) g_out[i]=__bfloat162float(o[i]);
    cudaFree(dOp); cudaFree(dMl); cudaFree(dO);
    if (grid_out) *grid_out = grid;
    return best;
}

/* ------------------------------------------------------------------------------------ probe A */
static void probeA(size_t nbytes){
    unsigned char* p=nullptr;
    CK(cudaMalloc(&p,nbytes)); CK(cudaMemset(p,0x11,nbytes));
    printf("\n########## PROBE A — occupancy on the REAL ACCESS MAPS, with NO register tax\n");
    printf("# working set %.2f GB. These kernels use ~40 regs, so occ 1..4 costs nothing:\n"
           "# this is the CEILING on what occupancy 2 could ever be worth to this map.\n", (double)nbytes/1e9);
    printf("%-40s %5s %5s %5s %8s %10s %8s\n","arm","regs","spil","b/SM","ms","GB/s","vs occ1");

    double base[8]; int nb=0;
#define ARM(LBL,KEXPR,GRID,NELEM,GRP) do { \
        auto kf = KEXPR; \
        cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa,(const void*)kf)); \
        int blk=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk,(const void*)kf,256,0)); \
        double ms=1e30; \
        for (int w=0;w<3;w++) kf<<<(GRID),256>>>(p,(NELEM)); \
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); \
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); \
        for (int i=0;i<g_iters;i++){ cudaEventRecord(a); kf<<<(GRID),256>>>(p,(NELEM)); \
            cudaEventRecord(b); cudaEventSynchronize(b); float t; cudaEventElapsedTime(&t,a,b); if(t<ms) ms=t; } \
        cudaEventDestroy(a); cudaEventDestroy(b); \
        double gb=(double)nbytes/(ms*1e-3)/1e9; \
        if ((GRP)==nb) base[nb++]=gb; \
        printf("%-40s %5d %5d %5d %8.4f %10.1f %8.3fx\n",LBL,fa.numRegs,(int)fa.localSizeBytes, \
               blk,ms,gb,gb/base[GRP]); \
    } while(0)

    const unsigned G=(unsigned)g_sm;
    /* group 0: linear map (the PX-11 1700 GB/s pin) */
    ARM("0 linear 16B/ln U=4        occ-cap 1", (k_lin<16,4,1>), G,   nbytes, 0);
    ARM("0 linear 16B/ln U=4  grid2 occ-cap 2", (k_lin<16,4,2>), 2*G, nbytes, 0);
    ARM("0 linear 16B/ln U=4  grid4 occ-cap 4", (k_lin<16,4,4>), 4*G, nbytes, 0);
    /* group 1: the fp8 hd512 SCORE map -- 512 B row per thread (PX-11 rung 1, 74% of wall) */
    ARM("1 row/thr 512B row 8B/ln U=8       occ-cap 1", (k_row<512,8,8,1>), G,   nbytes/512, 1);
    ARM("1 row/thr 512B row 8B/ln U=8 grid2 occ-cap 2", (k_row<512,8,8,2>), 2*G, nbytes/512, 1);
    ARM("1 row/thr 512B row 8B/ln U=8 grid3 occ-cap 3", (k_row<512,8,8,3>), 3*G, nbytes/512, 1);
    ARM("1 row/thr 512B row 8B/ln U=8 grid4 occ-cap 4", (k_row<512,8,8,4>), 4*G, nbytes/512, 1);
    /* group 2: same map, U=1 -- the LATENCY-STARVED variant. If occupancy ever rescues a map,
     * it is here: extra warps are exactly the substitute for per-thread MLP. */
    ARM("2 row/thr 512B row 8B/ln U=1       occ-cap 1", (k_row<512,8,1,1>), G,   nbytes/512, 2);
    ARM("2 row/thr 512B row 8B/ln U=1 grid2 occ-cap 2", (k_row<512,8,1,2>), 2*G, nbytes/512, 2);
    ARM("2 row/thr 512B row 8B/ln U=1 grid4 occ-cap 4", (k_row<512,8,1,4>), 4*G, nbytes/512, 2);
    /* group 3: the bf16 hd512 score map -- 1024 B row per thread (PX-11 rung 1, 35% of wall) */
    ARM("3 row/thr 1024B row 16B/ln U=4       occ-cap 1", (k_row<1024,16,4,1>), G,   nbytes/1024, 3);
    ARM("3 row/thr 1024B row 16B/ln U=4 grid2 occ-cap 2", (k_row<1024,16,4,2>), 2*G, nbytes/1024, 3);
    ARM("3 row/thr 1024B row 16B/ln U=4 grid4 occ-cap 4", (k_row<1024,16,4,4>), 4*G, nbytes/1024, 3);
    /* group 4: the V-phase map at the kernel's own VU (2 at GF=8) -- PX-11 says U=2 costs 7% */
    ARM("4 rowgrp 512B row 8B/ln U=2       occ-cap 1", (k_rowgrp<512,8,2,1>), G,   nbytes/512, 4);
    ARM("4 rowgrp 512B row 8B/ln U=2 grid2 occ-cap 2", (k_rowgrp<512,8,2,2>), 2*G, nbytes/512, 4);
    ARM("4 rowgrp 512B row 8B/ln U=2 grid4 occ-cap 4", (k_rowgrp<512,8,2,4>), 4*G, nbytes/512, 4);
    /* ...and the SAME map with the unroll raised instead of the occupancy. If these two land
     * together, occupancy 2 is buying nothing that FA_DEC_VU(8): 2 -> 4 does not buy for free. */
    ARM("4 rowgrp 512B row 8B/ln U=4       occ-cap 1", (k_rowgrp<512,8,4,1>), G,   nbytes/512, 4);
    ARM("4 rowgrp 512B row 8B/ln U=8       occ-cap 1", (k_rowgrp<512,8,8,1>), G,   nbytes/512, 4);
#undef ARM
    cudaFree(p);
}

/* ------------------------------------------------------------------------------- probes B / C */
static const Class& C = C_FULL;
static int   g_B  = 8;
static unsigned g_ctx = 131072u;
static std::vector<float> g_ref;
static bool  g_have_ref = false;
static double g_base_ms = 0.0;

template<int GF,int R>
static void fd_arm(const char* lbl, unsigned ns, int gridmul, size_t pad = 0){
    const unsigned stride = g_ctx, mask = 0xFFFFFFFFu, span = g_ctx;
    Res r{}; unsigned grid=0;
    double ms = run_fd<512,GF,R>(C,g_B,g_ctx,stride,mask,ns,gridmul,&r,&grid,pad);
    const double phys = (double)g_B*C.NKV*span*C.D*1.0*2.0;
    const double iss  = (double)g_B*(C.NH/GF)*span*C.D*1.0*2.0;
    float md=0.f;
    if (!g_have_ref){ g_ref=g_out; g_have_ref=true; g_base_ms=ms; }
    else for (size_t i=0;i<g_ref.size();i++){ float d=fabsf(g_out[i]-g_ref[i]); if(d>md) md=d; }
    printf("%-26s %3d %4u %5u %5d %6d %5d %10.4f %10.1f %10.1f %8.3fx  %.3e\n",
           lbl, GF, ns, grid, r.regs, r.spill, r.occ_blk, ms,
           phys/(ms*1e-3)/1e9, iss/(ms*1e-3)/1e9, g_base_ms/ms, md);
    fflush(stdout);
}

static void hdr(){
    printf("%-26s %3s %4s %5s %5s %6s %5s %10s %10s %10s %9s  %s\n",
           "arm","GF","ns","grid","regs","spill","b/SM","ms","GBps_phys","GBps_iss","vs_nat","maxdiff");
}

int main(int argc, char** argv){
    const char* mode = argc>1?argv[1]:"all";
    g_iters = argc>2?atoi(argv[2]):20;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); g_sm=p.multiProcessorCount;
    printf("# device %s  cc %d.%d  SMs %d  iters %d\n",p.name,p.major,p.minor,g_sm,g_iters);
    printf("# sharedMemPerMultiprocessor %zu B   sharedMemPerBlockOptin %zu B   regsPerBlock(=per SM) %d\n",
           (size_t)p.sharedMemPerMultiprocessor,(size_t)p.sharedMemPerBlockOptin,p.regsPerBlock);
    int memclk=0; CK(cudaDeviceGetAttribute(&memclk,cudaDevAttrMemoryClockRate,0)); /* kHz */
    printf("# maxThreadsPerMultiProcessor %d  warpSize %d  l2 %d B  memBusWidth %d  memClk %d kHz\n",
           p.maxThreadsPerMultiProcessor,p.warpSize,p.l2CacheSize,p.memoryBusWidth,memclk);
    printf("# spec HBM pin: %.1f GB/s\n",
           2.0*(double)memclk*1e3*(p.memoryBusWidth/8.0)/1e9);
    printf("# FA_DEC_SMEM_FLOATS(512,2)=%d B  (512,4)=%d B  (512,8)=%d B\n",
           FA_DEC_SMEM_FLOATS(512,2)*4, FA_DEC_SMEM_FLOATS(512,4)*4, FA_DEC_SMEM_FLOATS(512,8)*4);

    if (!strcmp(mode,"a") || !strcmp(mode,"all")) {
        size_t ws = (size_t)2048*1024*1024ull;
        if (const char* e=getenv("PX16_WS")) ws=(size_t)atoll(e);
        probeA(ws);
        if (!strcmp(mode,"a")) return 0;
    }

    if (const char* e=getenv("PX16_B"))   g_B = atoi(e);
    if (const char* e=getenv("PX16_CTX")) g_ctx = (unsigned)atoi(e);
    printf("\n########## PROBES B/C — the SHIPPED d_flash_decode<512,GF,fp8>\n");
    printf("# class=full D=512 NH=16 NKV=1 gqa=16 B=%d ctx=%u  fp8 KV\n", g_B, g_ctx);
    printf("# 'vs_nat' is against the FIRST row (the natural occ-1 arm at the same GF/ns).\n");
    printf("# maxdiff must be 0 everywhere: a register cap adds spills, never arithmetic.\n");
    pool_alloc(C,g_B,g_ctx);

    /* ---- OCCUPANCY, ISOLATED. Grid is FIXED at 2x = 340 blocks in every row; the only
     * difference is the smem pad, which pushes blocks/SM from 2 down to 1. Any delta here is
     * occupancy and nothing else -- same grid, same wave quantisation, same register count. */
    /* Both pads are >48 KB, so both opt in to the SAME max-shared carveout and leave the same
     * L1 behind; the ONLY difference between them is that 50176 lets two blocks fit per SM and
     * 52224 lets one. That is the cleanest occupancy-only A/B available on this kernel. */
    const size_t PAD1 = 52224; /* > 102400/2 -> exactly ONE block per SM  */
    const size_t PAD2 = 50176; /* 2 x 50176 = 100352 <= 102400 -> TWO blocks per SM */
    printf("\n--- OCCUPANCY ISOLATED: grid FIXED at 2x, occ set by smem pad, carveout matched ---\n");
    printf("--- GF=4 (d_flash_decode<512,4,fp8> is 128 regs on its own -> occ 2 is FREE) ---\n");
    hdr();
    for (unsigned ns : {21u,32u,43u,64u,85u}) {
        g_have_ref=false;
        fd_arm<4,0>("occ1 pad52224", ns, 2, PAD1);
        fd_arm<4,0>("occ2 pad50176", ns, 2, PAD2);
        fd_arm<4,0>("occ2 natural ", ns, 2, 0);
        fd_arm<4,0>("occ2 grid=1x ", ns, 1, 0);
    }
    printf("--- GF=8 (168 regs natural), capped to 128 WITH spills so occ 2 is reachable ---\n");
    hdr();
    for (unsigned ns : {16u,21u,32u,43u}) {
        g_have_ref=false;
        fd_arm<8,128>("cap128 occ1 pad52224", ns, 2, PAD1);
        fd_arm<8,128>("cap128 occ2 pad50176", ns, 2, PAD2);
        fd_arm<8,128>("cap128 occ2 natural ", ns, 2, 0);
    }

    /* ---- BEST-vs-BEST. The only comparison that decides anything: sweep nsplit and grid for
     * the best occ-1 arm the kernel can reach today, and the best occ-2 arm a register cut
     * could reach, and take min over each family. Ratios of single cells are wave-quantisation
     * noise (PX-6); ratios of the two minima are not. */
    printf("\n--- BEST-vs-BEST: GF=8 nsplit sweep, occ1 (natural 168 regs) vs occ2 (cap128/96) ---\n");
    hdr();
    for (unsigned ns : {11u,16u,21u,26u,32u,43u,64u,85u}) {
        g_have_ref=false;
        fd_arm<8,0  >("occ1 nat168 grid=1x", ns, 1, 0);
        fd_arm<8,0  >("occ1 nat168 grid=2x", ns, 2, 0);
        fd_arm<8,128>("occ2 cap128 grid=2x", ns, 2, 0);
        fd_arm<8,96 >("occ2 cap96  grid=2x", ns, 2, 0);
    }

    /* ---- GF=4, ns=32: what scripts/build_sm120_cubin.sh actually deploys ---- */
    printf("\n--- GF=4 ns=32 (DEPLOYED cubin config: GF_FULL=4, NS_FULL_ABS=32) ---\n"); hdr();
    g_have_ref=false;
    fd_arm<4,0  >("nat occ1 grid=1x", 32, 1);
    fd_arm<4,0  >("nat occ1 grid=2x", 32, 2);   /* grid control: 2 waves, still 1 blk/SM */
    fd_arm<4,128>("cap128 grid=2x",   32, 2);
    fd_arm<4,128>("cap128 grid=1x",   32, 1);
    fd_arm<4,168>("cap168 grid=1x",   32, 1);
    fd_arm<4,96 >("cap96  grid=2x",   32, 2);
    fd_arm<4,80 >("cap80  grid=3x",   32, 3);
    fd_arm<4,64 >("cap64  grid=4x",   32, 4);

    /* ---- the curve: GF=4, ns=32, grid always 2x so occ 2 is reachable ---- */
    printf("\n--- CURVE: GF=4 ns=32, grid=2x (340 blocks), __maxnreg__ swept ---\n"); hdr();
    g_have_ref=false;
    fd_arm<4,0  >("nat        grid=2x", 32, 2);
    fd_arm<4,224>("cap224     grid=2x", 32, 2);
    fd_arm<4,192>("cap192     grid=2x", 32, 2);
    fd_arm<4,168>("cap168     grid=2x", 32, 2);
    fd_arm<4,152>("cap152     grid=2x", 32, 2);
    fd_arm<4,136>("cap136     grid=2x", 32, 2);
    fd_arm<4,128>("cap128     grid=2x", 32, 2);
    fd_arm<4,120>("cap120     grid=2x", 32, 2);
    fd_arm<4,104>("cap104     grid=2x", 32, 2);
    fd_arm<4,96 >("cap96      grid=2x", 32, 2);
    fd_arm<4,88 >("cap88      grid=2x", 32, 2);
    fd_arm<4,80 >("cap80      grid=2x", 32, 2);
    fd_arm<4,72 >("cap72      grid=2x", 32, 2);
    fd_arm<4,64 >("cap64      grid=2x", 32, 2);

    /* ---- GF=8 ns=21: PX-11's recommended arm, the one a future build would ship ---- */
    printf("\n--- GF=8 ns=21 (PX-11 recommended arm) ---\n"); hdr();
    g_have_ref=false;
    fd_arm<8,0  >("nat occ1 grid=1x", 21, 1);
    fd_arm<8,0  >("nat occ1 grid=2x", 21, 2);
    fd_arm<8,128>("cap128 grid=2x",   21, 2);
    fd_arm<8,96 >("cap96  grid=2x",   21, 2);

    /* ---- GF=4 ns=21: the aligned nsplit, in case ns=32 hides the effect ---- */
    printf("\n--- GF=4 ns=21 ---\n"); hdr();
    g_have_ref=false;
    fd_arm<4,0  >("nat occ1 grid=1x", 21, 1);
    fd_arm<4,0  >("nat occ1 grid=2x", 21, 2);
    fd_arm<4,128>("cap128 grid=2x",   21, 2);
    fd_arm<4,96 >("cap96  grid=2x",   21, 2);

    /* ---- B=1: fewer work items, so residency matters more ---- */
    pool_free(); g_B = 1; pool_alloc(C,g_B,g_ctx);
    printf("\n--- B=1 ctx=131072, GF=4 ns=85 ---\n"); hdr();
    g_have_ref=false;
    fd_arm<4,0  >("nat occ1 grid=1x", 85, 1);
    fd_arm<4,0  >("nat occ1 grid=2x", 85, 2);
    fd_arm<4,128>("cap128 grid=2x",   85, 2);
    fd_arm<4,96 >("cap96  grid=2x",   85, 2);
    pool_free();
    return 0;
}
