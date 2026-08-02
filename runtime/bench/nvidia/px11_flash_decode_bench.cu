/* px11_flash_decode_bench.cu — is the flash DECODE kernel at its BANDWIDTH ceiling?
 *
 * PX-9 built a cycles/QMMA ladder for a compute-bound prefill GEMM. Decode at qlen=1 is
 * GEMV-shaped and bandwidth-bound, so the ladder here is denominated in GB/s against the
 * MEASURED achievable HBM of the same silicon, rung 0 of this very binary.
 *
 *   rung 0  ideal coalesced stream, W B/lane, __ldcs      -> achievable HBM, this working set
 *   rung 1  row-per-thread stream (the K SCORE phase map) -> does the scatter cost bandwidth?
 *   rung 2  row-group stream (the V phase map)            -> the coalesced half of the kernel
 *   rung 3  the REAL d_flash_decode<D,GF,FP8>, called directly (no cubin, no dispatch)
 *
 * Denominators (both printed; a phys number above the rung-0 ceiling means the DENOMINATOR is
 * wrong, not that a wall was broken):
 *   bytes_phys   = B * NKV      * span * D * elem * 2   distinct HBM bytes
 *   bytes_issued = B * (NH/GF)  * span * D * elem * 2   demand incl. the GQA re-read
 *
 * Layer classes (Gemma-4-12B unified):
 *   FULL    D=512 NH=16 NKV=1 (gqa 16) window=0     kv_stride=ctx   mask=~0
 *   SLIDING D=256 NH=16 NKV=8 (gqa  2) window=1024  ring=16384      mask=16383
 *
 * Build (plain env, sm_120a):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint \
 *     perf-data/px11_flash_decode_bench.cu -o /tmp/px11
 * Knob arms add -DPLOW_NV_FA_KUN=.. etc. The kernel source is the SHIPPED header, unmodified.
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

/* ------------------------------------------------------------------ rung 0/1/2: stream probes */
/* All three read exactly `nbytes` from `p` per launch, with __ldcs (what the KV path uses).
 * The only difference is the ADDRESS MAP. Sink store is predicated on an impossible value so
 * ptxas cannot drop the loads but no store ever executes. */
__device__ unsigned long long g_sink;

/* W = bytes per lane per load: 16 (uint4) or 8 (uint2). U = independent loads in flight per
 * thread. WITHOUT U>1 this loop is LATENCY-bound, not request-bound, and reports a ceiling
 * ~35% low at W=8 — the bug that cost this campaign its first ladder (see the doc). The warp
 * stays fully coalesced: within one unroll step the 256 threads cover 256*W contiguous bytes. */
template<int W, int U>
__global__ void __launch_bounds__(256,1)
k_stream_lin(const unsigned char* __restrict__ p, size_t nbytes) {
    const size_t chunk = (size_t)blockDim.x * U * W;             /* bytes a block does per step */
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
/* Same map, no occupancy pin — is 1 block/SM (what the megakernel forces) itself the limit? */
template<int W, int U>
__global__ void k_stream_lin_occ(const unsigned char* __restrict__ p, size_t nbytes) {
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

/* rung 1 — one whole ROW per THREAD, exactly the default score-phase map. U = chunks staged
 * before any is consumed: U=1 is the shipped body (PLOW_NV_FA_KUN=1), U>1 is the FA_KUN arm. */
template<int ROWB, int W, int U>
__global__ void __launch_bounds__(256,1)
k_stream_row(const unsigned char* __restrict__ p, size_t nrows) {
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
                if constexpr (W == 16) v4[u] = __ldcs((const uint4*)(row + o + u*W));
                else                   v2[u] = __ldcs((const uint2*)(row + o + u*W));
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

/* rung 2 — NDT lanes cover one row, exactly the V-phase map (dbase = (tid%NDT)*8 elems).
 * U = rows in flight, i.e. the kernel's FA_DEC_VU(GF): 8 at GF=2, 4 at GF=4, 2 at GF=8. */
template<int ROWB, int W, int U>
__global__ void __launch_bounds__(256,1)
k_stream_rowgrp(const unsigned char* __restrict__ p, size_t nrows) {
    constexpr int NDT = ROWB / W;          /* threads covering one row */
    constexpr int NG  = 256 / NDT;         /* row groups per block */
    const unsigned dbase = (threadIdx.x % NDT) * W;
    const unsigned grp   = threadIdx.x / NDT;
    const size_t stride  = (size_t)gridDim.x * NG * U;
    unsigned acc = 0;
    for (size_t r0 = (size_t)blockIdx.x * NG * U + grp; r0 + (U-1)*NG < nrows; r0 += stride) {
        uint4 v4[W==16?U:1]; uint2 v2[W==8?U:1];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const unsigned char* a = p + (r0 + (size_t)u*NG) * (size_t)ROWB + dbase;
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

/* --------------------------------------------------------------- rung 3: the SHIPPED kernel */
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_fd_bf16(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
          const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
          unsigned window, float scale, unsigned nsplit, unsigned kvmask){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,false>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,window,scale,nsplit,
                               kvmask,blockIdx.x,gridDim.x,arena,0u);
}
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_fd_fp8(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
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
static const Class C_FULL  = { "full",  512, 16, 1, 0,     0     }; /* kv_stride = ctx, mask ~0 */
static const Class C_SLIDE = { "slide", 256, 16, 8, 1024,  16384 };

struct Pool {
    unsigned char *k8=nullptr, *v8=nullptr;   /* e4m3 */
    bf16 *k16=nullptr, *v16=nullptr;
    float *ks=nullptr, *vs=nullptr;
    bf16 *q=nullptr; int* len=nullptr;
    size_t rows=0;                            /* B*NKV*stride rows */
    int D=0;
};
static Pool g_p;

static void pool_alloc(const Class& c, int B, unsigned stride){
    const size_t rows = (size_t)B * c.NKV * stride;
    const size_t n = rows * c.D;
    g_p.rows = rows; g_p.D = c.D;
    std::vector<unsigned char> h8(n); std::vector<bf16> h16(n);
    for (size_t i=0;i<n;i++){ float a = rnd()*0.5f; __nv_fp8_e4m3 f(a);
        h8[i] = *(unsigned char*)&f; h16[i] = __float2bfloat16(a); }
    CK(cudaMalloc(&g_p.k8,n));  CK(cudaMemcpy(g_p.k8,h8.data(),n,cudaMemcpyHostToDevice));
    for (size_t i=0;i<n;i++){ float a = rnd()*0.5f; __nv_fp8_e4m3 f(a); h8[i]=*(unsigned char*)&f; }
    CK(cudaMalloc(&g_p.v8,n));  CK(cudaMemcpy(g_p.v8,h8.data(),n,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_p.k16,n*2)); CK(cudaMemcpy(g_p.k16,h16.data(),n*2,cudaMemcpyHostToDevice));
    for (size_t i=0;i<n;i++) h16[i]=__float2bfloat16(rnd()*0.5f);
    CK(cudaMalloc(&g_p.v16,n*2)); CK(cudaMemcpy(g_p.v16,h16.data(),n*2,cudaMemcpyHostToDevice));
    std::vector<float> sc(rows,1.0f);
    CK(cudaMalloc(&g_p.ks,rows*4)); CK(cudaMemcpy(g_p.ks,sc.data(),rows*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_p.vs,rows*4)); CK(cudaMemcpy(g_p.vs,sc.data(),rows*4,cudaMemcpyHostToDevice));
    std::vector<bf16> hq((size_t)B*c.NH*c.D);
    for (auto& x:hq) x=__float2bfloat16(rnd()*0.5f);
    CK(cudaMalloc(&g_p.q,hq.size()*2)); CK(cudaMemcpy(g_p.q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_p.len,(size_t)B*4));
}
static void pool_free(){
    cudaFree(g_p.k8); cudaFree(g_p.v8); cudaFree(g_p.k16); cudaFree(g_p.v16);
    cudaFree(g_p.ks); cudaFree(g_p.vs); cudaFree(g_p.q); cudaFree(g_p.len);
    memset(&g_p,0,sizeof(g_p));
}

static std::vector<float> g_out;

/* L2 flush. The SLIDING class touches only window*NKV*B rows -- 16.8 MB at B=8/fp8 -- which is
 * L2-RESIDENT, so a warm loop measures L2, not HBM. In a real decode step those rows were last
 * written up to 1024 steps ago with 25 GiB of traffic in between, i.e. they are cold. Launched
 * BEFORE the start event so the eviction is not in the timed window. */
static unsigned char* g_flush = nullptr;
static size_t g_flush_bytes = 192ull<<20;   /* 2x the 96 MiB L2 */
__global__ void k_flush(unsigned char* p, size_t n){
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x, s = (size_t)gridDim.x*blockDim.x;
    for (size_t i=t; i<n/16; i+=s) ((uint4*)p)[i] = make_uint4(i,i,i,i);
}
static bool g_do_flush = false;
static inline void l2_flush(){ if (g_do_flush) k_flush<<<1024,256>>>(g_flush,g_flush_bytes); }

template<int D,int GF>
static double run_fd(const Class& c, bool fp8, int B, unsigned ctx, unsigned stride,
                     unsigned mask, unsigned nsplit, int iters, unsigned* n_work_out){
    const unsigned n_grp = c.NH/GF;
    const unsigned n_work = (unsigned)B*n_grp*nsplit;
    const float scale = 1.f/sqrtf((float)D);
    std::vector<int> hl(B,(int)ctx);
    CK(cudaMemcpy(g_p.len,hl.data(),(size_t)B*4,cudaMemcpyHostToDevice));

    float *dOp=nullptr,*dMl=nullptr; bf16* dO=nullptr;
    CK(cudaMalloc(&dOp,(size_t)B*c.NH*nsplit*D*4));
    CK(cudaMalloc(&dMl,(size_t)B*c.NH*nsplit*2*4));
    CK(cudaMalloc(&dO ,(size_t)B*c.NH*D*2));

    const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    const unsigned grid = (n_work < (unsigned)g_sm) ? n_work : (unsigned)g_sm;
    auto lf=[&](){
        if (fp8) k_fd_fp8<D,GF><<<grid,256,smem>>>(dOp,dMl,g_p.q,(bf16*)g_p.k8,(bf16*)g_p.v8,
            g_p.len,(unsigned)B,c.NH,c.NKV,stride,(unsigned)c.window,scale,nsplit,mask,g_p.ks,g_p.vs);
        else     k_fd_bf16<D,GF><<<grid,256,smem>>>(dOp,dMl,g_p.q,g_p.k16,g_p.v16,
            g_p.len,(unsigned)B,c.NH,c.NKV,stride,(unsigned)c.window,scale,nsplit,mask);
    };
    if (fp8) CK(cudaFuncSetAttribute(k_fd_fp8<D,GF>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    else     CK(cudaFuncSetAttribute(k_fd_bf16<D,GF>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    for (int w=0;w<3;w++) lf();
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best=1e30;
    for (int it=0; it<iters; it++){
        l2_flush();
        CK(cudaEventRecord(a)); lf(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); if (ms<best) best=ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    /* merge once for the numeric capture (not timed here) */
    k_merge<D><<<(unsigned)B*c.NH,256>>>(dO,dOp,dMl,(unsigned)B,c.NH,nsplit);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> o((size_t)B*c.NH*D);
    CK(cudaMemcpy(o.data(),dO,o.size()*2,cudaMemcpyDeviceToHost));
    g_out.resize(o.size());
    for (size_t i=0;i<o.size();i++) g_out[i]=__bfloat162float(o[i]);
    cudaFree(dOp); cudaFree(dMl); cudaFree(dO);
    if (n_work_out) *n_work_out = n_work;
    return best;
}

/* dispatch over the (D,GF) template pairs this bench instantiates */
static double run_fd_disp(const Class& c, int GF, bool fp8, int B, unsigned ctx, unsigned stride,
                          unsigned mask, unsigned nsplit, int iters, unsigned* nw){
    if (c.D==512){
        if (GF==2) return run_fd<512,2>(c,fp8,B,ctx,stride,mask,nsplit,iters,nw);
        if (GF==4) return run_fd<512,4>(c,fp8,B,ctx,stride,mask,nsplit,iters,nw);
        if (GF==8) return run_fd<512,8>(c,fp8,B,ctx,stride,mask,nsplit,iters,nw);
    } else {
        if (GF==1) return run_fd<256,1>(c,fp8,B,ctx,stride,mask,nsplit,iters,nw);
        if (GF==2) return run_fd<256,2>(c,fp8,B,ctx,stride,mask,nsplit,iters,nw);
    }
    printf("unsupported D=%d GF=%d\n",c.D,GF); exit(3);
}

static double ms_of(void (*launch)(), int iters){
    for (int w=0;w<3;w++) launch();
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best=1e30;
    for (int it=0;it<iters;it++){
        CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); if (ms<best) best=ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b); return best;
}

/* ---------------------------------------------------------------------------------- ceiling */
static void ceiling(size_t nbytes, int iters){
    unsigned char* p=nullptr;
    CK(cudaMalloc(&p,nbytes));
    CK(cudaMemset(p,0x11,nbytes));
    const unsigned G=(unsigned)g_sm;
    printf("# ceiling ladder, working set %.2f GB, grid %u x 256\n",(double)nbytes/1e9,G);
    printf("%-34s %10s %12s\n","rung","ms","GB/s");
    struct R { const char* n; double ms; };
    std::vector<R> rs;
#define LINARM(LBL,K,W,U,GRID) { double ms=1e30; \
      for (int w=0;w<3;w++) K<W,U><<<(GRID),256>>>(p,nbytes); \
      CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); \
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); \
      for (int i=0;i<iters;i++){ cudaEventRecord(a); K<W,U><<<(GRID),256>>>(p,nbytes); \
        cudaEventRecord(b); cudaEventSynchronize(b); float t; cudaEventElapsedTime(&t,a,b); if(t<ms) ms=t; } \
      cudaEventDestroy(a); cudaEventDestroy(b); rs.push_back({LBL,ms}); }
    LINARM("0  linear 16 B/ln U=1  occ1", k_stream_lin, 16, 1, G)
    LINARM("0  linear 16 B/ln U=4  occ1", k_stream_lin, 16, 4, G)
    LINARM("0  linear 16 B/ln U=8  occ1", k_stream_lin, 16, 8, G)
    LINARM("0b linear  8 B/ln U=1  occ1", k_stream_lin,  8, 1, G)
    LINARM("0b linear  8 B/ln U=8  occ1", k_stream_lin,  8, 8, G)
    LINARM("0b linear  8 B/ln U=16 occ1", k_stream_lin,  8,16, G)
    LINARM("0c linear 16 B/ln U=4  occ4", k_stream_lin_occ, 16, 4, 4*G)
    LINARM("0c linear  8 B/ln U=8  occ4", k_stream_lin_occ,  8, 8, 4*G)
#undef LINARM
#define ROWARM(LBL,K,RB,W,U) { const size_t nr=nbytes/(RB); double ms=1e30; \
      for (int w=0;w<3;w++) K<RB,W,U><<<G,256>>>(p,nr); CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); \
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); \
      for (int i=0;i<iters;i++){ cudaEventRecord(a); K<RB,W,U><<<G,256>>>(p,nr); \
        cudaEventRecord(b); cudaEventSynchronize(b); float t; cudaEventElapsedTime(&t,a,b); if(t<ms) ms=t; } \
      cudaEventDestroy(a); cudaEventDestroy(b); rs.push_back({LBL,ms}); }
    /* bf16 hd512 row = 1024 B; bf16 hd256 row = 512 B; fp8 hd512 row = 512 B; fp8 hd256 = 256 B */
    ROWARM("1  row/thr 1024 B row 16 B/ln U=1 ", k_stream_row, 1024, 16, 1)
    ROWARM("1  row/thr 1024 B row 16 B/ln U=4 ", k_stream_row, 1024, 16, 4)
    ROWARM("1  row/thr 1024 B row 16 B/ln U=8 ", k_stream_row, 1024, 16, 8)
    ROWARM("1  row/thr  512 B row 16 B/ln U=1 ", k_stream_row,  512, 16, 1)
    ROWARM("1  row/thr  512 B row 16 B/ln U=8 ", k_stream_row,  512, 16, 8)
    ROWARM("1  row/thr  512 B row  8 B/ln U=1 ", k_stream_row,  512,  8, 1)
    ROWARM("1  row/thr  512 B row  8 B/ln U=8 ", k_stream_row,  512,  8, 8)
    ROWARM("1  row/thr  256 B row  8 B/ln U=1 ", k_stream_row,  256,  8, 1)
    ROWARM("1  row/thr  256 B row  8 B/ln U=8 ", k_stream_row,  256,  8, 8)
    /* U here is the kernel's FA_DEC_VU(GF): 8 at GF=2, 4 at GF=4, 2 at GF=8. */
    ROWARM("2  rowgrp  1024 B row 16 B/ln U=1 ", k_stream_rowgrp, 1024, 16, 1)
    ROWARM("2  rowgrp  1024 B row 16 B/ln U=2 ", k_stream_rowgrp, 1024, 16, 2)
    ROWARM("2  rowgrp  1024 B row 16 B/ln U=4 ", k_stream_rowgrp, 1024, 16, 4)
    ROWARM("2  rowgrp  1024 B row 16 B/ln U=8 ", k_stream_rowgrp, 1024, 16, 8)
    ROWARM("2  rowgrp   512 B row  8 B/ln U=1 ", k_stream_rowgrp,  512,  8, 1)
    ROWARM("2  rowgrp   512 B row  8 B/ln U=2 ", k_stream_rowgrp,  512,  8, 2)
    ROWARM("2  rowgrp   512 B row  8 B/ln U=4 ", k_stream_rowgrp,  512,  8, 4)
    ROWARM("2  rowgrp   512 B row  8 B/ln U=8 ", k_stream_rowgrp,  512,  8, 8)
    ROWARM("2  rowgrp   256 B row  8 B/ln U=8 ", k_stream_rowgrp,  256,  8, 8)
    /* 3 — WARP-per-row (what PLOW_NV_FA_WPR turns the SCORE phase into): 32 lanes cover
     *     one 512 B chunk of a row, so the row read is fully coalesced. WRB rows in flight. */
    ROWARM("3  warp/row 1024 B row 16 B/ln U=1 ", k_stream_rowgrp, 512, 16, 1)
    ROWARM("3  warp/row 1024 B row 16 B/ln U=4 ", k_stream_rowgrp, 512, 16, 4)
#undef ROWARM
    for (auto& r : rs)
        printf("%-34s %10.4f %12.1f\n", r.n, r.ms, (double)nbytes/(r.ms*1e-3)/1e9);
    printf("\n");
    cudaFree(p);
}

/* ------------------------------------------------------------------------------------- main */
int main(int argc, char** argv){
    const char* mode = argc>1?argv[1]:"all";
    int iters = argc>2?atoi(argv[2]):30;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); g_sm=p.multiProcessorCount;
    printf("# device %s, SMs %d, iters %d\n", p.name, g_sm, iters);
    printf("# knobs: FA_KUN=%d FA_WPR=%d FA_WPR_RB=%d FA_QGLOB=%d FA_REDBOUND=%d"
#ifdef PLOW_FP8_LD16
           " FP8_LD16=1"
#endif
#ifdef PLOW_FP8_FAST
           " FP8_FAST=1"
#endif
           "\n", PLOW_NV_FA_KUN, PLOW_NV_FA_WPR, PLOW_NV_FA_WPR_RB,
           PLOW_NV_FA_QGLOB, PLOW_NV_FA_REDBOUND);

    if (!strcmp(mode,"ceil") || !strcmp(mode,"all")) {
        size_t ws = (size_t)2048*1024*1024ull;              /* 2 GB, the in-tree HBM protocol */
        const char* e = getenv("PX11_WS"); if (e) ws = (size_t)atoll(e);
        ceiling(ws, iters);
        if (!strcmp(mode,"ceil")) return 0;
    }

    /* --------- the real kernel, per layer class --------- */
    const char* cls = getenv("PX11_CLASS"); if (!cls) cls="full";
    const Class& c = strcmp(cls,"slide")==0 ? C_SLIDE : C_FULL;
    const int B    = getenv("PX11_B")   ? atoi(getenv("PX11_B"))   : 8;
    const unsigned ctx = getenv("PX11_CTX") ? (unsigned)atoi(getenv("PX11_CTX")) : 131072u;
    const unsigned stride = c.ring ? (unsigned)c.ring : ctx;
    const unsigned mask   = c.ring ? (unsigned)(c.ring-1) : 0xFFFFFFFFu;
    const unsigned span   = c.window ? (ctx < (unsigned)c.window ? ctx : (unsigned)c.window) : ctx;

    g_do_flush = getenv("PX11_FLUSH") && atoi(getenv("PX11_FLUSH"));
    printf("# class=%s D=%d NH=%d NKV=%d gqa=%d window=%d ring=%u B=%d ctx=%u span=%u l2flush=%d\n",
           c.name,c.D,c.NH,c.NKV,c.NH/c.NKV,c.window,stride,B,ctx,span,(int)g_do_flush);
    pool_alloc(c,B,stride);
    if (g_do_flush) CK(cudaMalloc(&g_flush,g_flush_bytes));

    /* nsplit list: env override, else a sweep that includes the shipped grid-aligned choices */
    std::vector<unsigned> nss;
    if (const char* e = getenv("PX11_NS")) {
        char buf[256]; snprintf(buf,sizeof buf,"%s",e);
        for (char* t=strtok(buf,","); t; t=strtok(nullptr,",")) nss.push_back((unsigned)atoi(t));
    } else if (c.window) { nss = {8u,16u}; }
    else { nss = {16u,21u,32u,43u,85u}; }

    std::vector<int> gfs;
    if (const char* e = getenv("PX11_GF")) {
        char buf[64]; snprintf(buf,sizeof buf,"%s",e);
        for (char* t=strtok(buf,","); t; t=strtok(nullptr,",")) gfs.push_back(atoi(t));
    } else if (c.D==512) gfs = {2,4,8}; else gfs = {2};

    printf("%-5s %-3s %-6s %-8s | %10s | %10s %10s | %7s %7s | %s\n",
           "dt","GF","nsplit","n_work","ms","GBps_phys","GBps_iss","%ceil","reread",
           "maxdiff_vs_GF_ref@same_ns");
    for (int fp8i=1; fp8i>=0; fp8i--){
        const bool fp8 = fp8i==1;
        const double elem = fp8?1.0:2.0;
        for (unsigned ns : nss){
            std::vector<float> ref;
            for (int GF : gfs){
                if ((c.NH/c.NKV) % GF) continue;          /* gqa % GF == 0 */
                unsigned nw=0;
                double ms = run_fd_disp(c,GF,fp8,B,ctx,stride,mask,ns,iters,&nw);
                float md = 0.f;
                if (ref.empty()) ref = g_out;
                else for (size_t i=0;i<ref.size();i++){ float d=fabsf(g_out[i]-ref[i]); if (d>md) md=d; }
                const double phys = (double)B*c.NKV*span*c.D*elem*2.0;
                const double iss  = (double)B*(c.NH/GF)*span*c.D*elem*2.0;
                printf("%-5s %-3d %-6u %-8u | %10.4f | %10.1f %10.1f | %7.1f %7.1f | %.3e\n",
                       fp8?"fp8":"bf16",GF,ns,nw,ms,
                       phys/(ms*1e-3)/1e9, iss/(ms*1e-3)/1e9,
                       100.0*(phys/(ms*1e-3)/1e9)/1695.6, iss/phys, md);
                /* Cross-ARM numerics gate: PLOW_FP8_FAST/LD16 change the dequant rounding, so
                 * they are NOT bit-exact against the shipped path and need a measured bound. */
                if (const char* dp = getenv("PX11_DUMP")) {
                    char fn[512];
                    snprintf(fn,sizeof fn,"%s.%s.gf%d.ns%u.bin",dp,fp8?"fp8":"bf16",GF,ns);
                    FILE* f = fopen(fn,"wb");
                    if (f){ fwrite(g_out.data(),4,g_out.size(),f); fclose(f); }
                }
            }
        }
    }
    pool_free();
    if (g_flush) cudaFree(g_flush);
    return 0;
}
