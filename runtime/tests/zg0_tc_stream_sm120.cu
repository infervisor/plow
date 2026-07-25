/* zg0_tc_stream_sm120.cu — ZG-0 premise check for the ZipGEMM campaign.
 *
 * Question: does a PROPERLY-OPTIMIZED small-M (batch 1..64) tensor-core bf16 GEMM
 * C[M,N]=A[M,K].B[N,K]^T saturate HBM bandwidth on the RTX PRO 6000 (sm_120, 188 SM,
 * 1535 GB/s achievable)? C-1T's k_tc_bf16 only reached ~1000 GB/s (65% of wall) — was that
 * a kernel-quality artifact (occupancy / small-BK bursts / grid underutilisation) or a real
 * "small-M TC GEMM is not memory-bound" fact?
 *
 * This file measures, per shape x M:
 *   - WALL PROBE  k_stream_reduce : pure weight-read ceiling (fully-parallel grid-stride).
 *   - TC clean    k_tc<...>       : tuned small-M mma GEMM, split-K to fill the GPU, deep
 *                                   cp.async pipeline, tunable BN/BK/warps/stages.
 *   - TC baseline k_tc_bf16       : the exact C-1T kernel (BM=16, TN=128, 1 blk/SM) for A/B.
 *   - FFMA        k_ffma          : the current WS-GEMV (gemv_rows / gemv_walk).
 * Gate: bit-exact (TC clean, split=1) == k_tc_bf16, byte-identical; both within bf16 rounding
 * of an f32 device reference.
 *
 * Build (SYSTEM toolchain, NOT nix): nvcc -std=c++17 -O3 -arch=sm_120a
 *   -Iinclude -Iruntime/common -Iruntime/nvidia runtime/tests/zg0_tc_stream_sm120.cu -o /tmp/zg0
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <cuda_runtime.h>
#include "sm120_common.cuh"
#include "op_gemm.cuh"

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

/* ------------------------------------------------------------------ wall probe */
/* Pure streaming read of the whole weight [N*K] bf16 as uint4 (8 elems). Fully parallel
 * grid-stride so ALL SMs are busy regardless of N/K aspect ratio. XOR/add sink defeats DCE. */
__global__ void k_stream_reduce(const __nv_bfloat16* __restrict__ B, size_t nelem8, float* sink) {
    float acc = 0.f;
    const uint4* p = (const uint4*)B;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nelem8;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        /* touch every byte cheaply */
        acc += (float)(v.x ^ v.y ^ v.z ^ v.w);
    }
    /* one atomic per block */
    __shared__ float red[32];
    float w = acc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffffu, w, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += red[i];
        atomicAdd(sink, s);
    }
}

/* ------------------------------------------------------------------ f32 device reference */
/* Naive C[M,N] = sum_k A[m,k]*B[n,k], f32 accumulate, round to bf16. Order-tolerant oracle. */
__global__ void k_ref_f32(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B, unsigned M, unsigned N, unsigned K) {
    unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned m = blockIdx.y;
    if (n >= N || m >= M) return;
    const __nv_bfloat16* a = A + (size_t)m * K;
    const __nv_bfloat16* b = B + (size_t)n * K;
    float acc = 0.f;
    for (unsigned k = 0; k < K; k++) acc += __bfloat162float(a[k]) * __bfloat162float(b[k]);
    C[(size_t)m * N + n] = __float2bfloat16(acc);
}

/* ------------------------------------------------------------------ tuned TC GEMM (template) */
/* All NW warps share the same M rows and split BN columns; MFRAG = BM/16 m-fragments. Weight
 * B[n][k] staged in its natural [n][k] layout, read with ldmatrix.x2 non-.trans (proven in
 * op_gemm.cuh's probe). Split-K: a k-slice per block, accumulate into a global f32 buffer via
 * atomicAdd (ksplit=1 => single writer => identical to a direct store => bit-exact vs k_tc_bf16). */
template <int BM, int BN, int BK, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32) k_tc(float* __restrict__ Cf, /* [M*N] f32, pre-zeroed */
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        unsigned M, unsigned N, unsigned K, int ksplit) {
    constexpr int MFRAG = BM / 16;
    constexpr int WN    = BN / NW;         /* columns per warp */
    constexpr int NFRAG = WN / 8;
    constexpr int AS    = BK + 8;          /* A smem stride [m][k] */
    constexpr int BKS   = BK + 8;          /* B smem stride [n][k] */
    constexpr int ABUF  = BM * AS;
    constexpr int BBUF  = BN * BKS;
    constexpr int KCH   = BK / 8;          /* 16B lines per row */

    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    __nv_bfloat16* Bs = As + STAGES * ABUF;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_n = (N + BN - 1) / BN;
    const int totksteps = (K + BK - 1) / BK;
    /* split K into `ksplit` contiguous groups of k-steps */
    const int kper = (totksteps + ksplit - 1) / ksplit;
    const int njob = tiles_n * ksplit;

    for (int job = blockIdx.x; job < njob; job += gridDim.x) {
        const int nt = job / ksplit;          /* which N-tile */
        const int ksp = job % ksplit;         /* which K-slice */
        const int tn = nt * BN;
        const int ks0 = ksp * kper;
        const int ks1 = (ks0 + kper < totksteps) ? (ks0 + kper) : totksteps;
        if (ks0 >= ks1) continue;

        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
            for (int j = 0; j < NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            /* A tile [BM][BK] (k contiguous). OOB rows/cols => src_bytes 0 => HW zero-fill, no HBM. */
#pragma unroll
            for (int L = tid; L < BM * KCH; L += NW * 32) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int mm = row, kk = ks * BK + kk8;
                const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
                pgm_cp_async_cg16(&As[buf * ABUF + row * AS + kk8], g, in ? 16 : 0);
            }
            /* B tile [BN][BK] */
#pragma unroll
            for (int L = tid; L < BN * KCH; L += NW * 32) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int nn = tn + row, kk = ks * BK + kk8;
                const bool in = (nn < (int)N) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? B + (size_t)nn * K + kk : B;
                pgm_cp_async_cg16(&Bs[buf * BBUF + row * BKS + kk8], g, in ? 16 : 0);
            }
        };

        const int nks = ks1 - ks0;
#pragma unroll 1
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < nks) stage(ks0 + s, s);
            pgm_cp_commit();
        }
        for (int i = 0; i < nks; i++) {
            const int fetch = i + STAGES - 1;
            if (fetch < nks) stage(ks0 + fetch, fetch % STAGES);
            pgm_cp_commit();
            pgm_cp_wait<STAGES - 1>();
            __syncthreads();
            const int cb = i % STAGES;
            __nv_bfloat16* Ad = As + cb * ABUF;
            __nv_bfloat16* Bd = Bs + cb * BBUF;
#pragma unroll
            for (int kf = 0; kf < BK; kf += 16) {
                unsigned af[MFRAG][4];
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++) {
                    const int arow = mi * 16 + (lane % 16);
                    const int acol = kf + (lane / 16) * 8;
                    pgm_ldmatrix_x4(af[mi], &Ad[arow * AS + acol]);
                }
                unsigned bf[NFRAG][2];
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) {
                    const int nrow = warp * WN + nj * 8 + (lane & 7);
                    const int kcol = kf + ((lane >> 3) & 1) * 8;
                    pgm_ldmatrix_x2(bf[nj], &Bd[nrow * BKS + kcol]);
                }
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++)
                        pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

        /* epilogue: atomicAdd f32 partials into Cf[m*N+n] */
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) {
                const int gr = lane / 4;
                const int gc = warp * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int rr = mi * 16 + gr + (e / 2) * 8;
                    const int cc = tn + gc + (e % 2);
                    if (rr < (int)M && cc < (int)N)
                        atomicAdd(&Cf[(size_t)rr * N + cc], acc[mi][nj][e]);
                }
            }
        __syncthreads();
    }
}

__global__ void k_f32_to_bf16(__nv_bfloat16* C, const float* Cf, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = __float2bfloat16(Cf[i]);
}

/* ------------------------------------------------------------------ C-1T baseline k_tc_bf16 */
/* Ported verbatim (structure) from origin/c1t-tensorcore-bsweep runtime/tests/sz_tc_sm120.cu:
 * BM=16, TN=128, TK=32, 8 warps all in N, TC_STAGES=3, __launch_bounds__(256,1). */
#define B_TM 16
#define B_TN 128
#define B_TK 32
#define B_TKPAD 8
#define B_TBKS (B_TK + B_TKPAD)
#define B_TAS  (B_TK + B_TKPAD)
#define B_TWN  (B_TN / 8)
#define B_TNFRAG (B_TWN / 8)
#define B_STAGES 3
#define B_TABUF (B_TM * B_TAS)
#define B_TBBUF (B_TN * B_TBKS)
__global__ void __launch_bounds__(256,1) k_tc_bf16(__nv_bfloat16* __restrict__ C,
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        unsigned M, unsigned N, unsigned K) {
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)sm;
    __nv_bfloat16* Bs = As + B_STAGES * B_TABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tiles_m = (M + B_TM - 1) / B_TM, tiles_n = (N + B_TN - 1) / B_TN;
    const int ntiles = tiles_m * tiles_n, ksteps = (K + B_TK - 1) / B_TK;
    const int KCH = B_TK / 8;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int tm = (tile / tiles_n) * B_TM, tn = (tile % tiles_n) * B_TN;
        float acc[B_TNFRAG][4];
#pragma unroll
        for (int j=0;j<B_TNFRAG;j++) for(int e=0;e<4;e++) acc[j][e]=0.f;
        auto stage=[&](int ks,int buf){
            for (int L = tid; L < B_TM * KCH; L += 256) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int mm = tm + row, kk = ks*B_TK + kk8;
                const bool in = (mm < (int)M) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? A + (size_t)mm * K + kk : A;
                pgm_cp_async_cg16(&As[buf*B_TABUF + row * B_TAS + kk8], g, in ? 16 : 0);
            }
            for (int L = tid; L < B_TN * KCH; L += 256) {
                const int row = L / KCH, kk8 = (L % KCH) * 8;
                const int nn = tn + row, kk = ks*B_TK + kk8;
                const bool in = (nn < (int)N) && (kk + 8 <= (int)K);
                const __nv_bfloat16* g = in ? B + (size_t)nn * K + kk : B;
                pgm_cp_async_cg16(&Bs[buf*B_TBBUF + row * B_TBKS + kk8], g, in ? 16 : 0);
            }
        };
#pragma unroll
        for (int s=0;s<B_STAGES-1;s++){ if(s<ksteps) stage(s,s); pgm_cp_commit(); }
        for (int ks=0; ks<ksteps; ks++){
            const int fetch=ks+B_STAGES-1; if(fetch<ksteps) stage(fetch,fetch%B_STAGES);
            pgm_cp_commit(); pgm_cp_wait<B_STAGES-1>(); __syncthreads();
            const int cb=ks%B_STAGES; __nv_bfloat16* Ad=As+cb*B_TABUF; __nv_bfloat16* Bd=Bs+cb*B_TBBUF;
#pragma unroll
            for (int kf=0; kf<B_TK; kf+=16){
                unsigned af[4];
                { const int arow=lane%16, acol=kf+(lane/16)*8; pgm_ldmatrix_x4(af,&Ad[arow*B_TAS+acol]); }
                unsigned bf[B_TNFRAG][2];
#pragma unroll
                for (int nj=0;nj<B_TNFRAG;nj++){ const int n=warp*B_TWN+nj*8+(lane&7);
                    const int kcol=kf+((lane>>3)&1)*8; pgm_ldmatrix_x2(bf[nj],&Bd[n*B_TBKS+kcol]); }
#pragma unroll
                for (int nj=0;nj<B_TNFRAG;nj++) pgm_mma(acc[nj],af,bf[nj],acc[nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int nj=0;nj<B_TNFRAG;nj++){
            const int gr=lane/4, gc=warp*B_TWN+nj*8+(lane%4)*2;
#pragma unroll
            for (int e=0;e<4;e++){ const int rr=tm+gr+(e/2)*8, cc=tn+gc+(e%2);
                if (rr<(int)M && cc<(int)N) C[(size_t)rr*N+cc]=__float2bfloat16(acc[nj][e]); }
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ FFMA WS-GEMV launcher */
/* Reproduces the WS-GEMV: gemv_walk dispatches gemv_rows<MM> in blocks of GV_MM_MAX (8), so a
 * decode with M>8 re-reads the weight ceil(M/8) times — the honest "current WS-GEMV" behaviour.
 * Logical GB/s = single-weight-bytes / time (matches C-1T's convention). */
__global__ void k_ffma(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ W, unsigned M, unsigned N, unsigned K) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        constexpr int MM = decltype(mm)::v;
        gemv_rows<MM>(C + (size_t)m0 * N, x + (size_t)m0 * K, W, rows, N, K, blockIdx.x, gridDim.x);
    });
}

/* ================================================================== host harness */
static int g_sm = 188;

struct Shape { const char* name; unsigned N, K; };

/* fill deterministic small bf16 values in [-1,1)-ish so f32 accum stays sane */
static void fill_rand(std::vector<__nv_bfloat16>& v, uint64_t seed) {
    uint64_t s = seed | 1;
    for (auto& e : v) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        float f = ((int)(s & 0xffff) - 32768) / 32768.0f * 0.25f;
        e = __float2bfloat16(f);
    }
}

static double time_kernel(std::function<void()> launch, int iters, bool flush_l2, void* flushbuf,
                          size_t flushbytes, cudaStream_t st) {
    /* warmup */
    for (int i = 0; i < 3; i++) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        if (flush_l2) CK(cudaMemsetAsync(flushbuf, it & 0xff, flushbytes, st));
        CK(cudaEventRecord(e0, st));
        launch();
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return best; /* ms */
}

/* configurable TC launch: returns GB/s (logical weight bytes / time) and writes Cf->C */
template <int BM, int BN, int BK, int NW, int STAGES>
static double run_tc(float* Cf, __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                     unsigned M, unsigned N, unsigned K, int ksplit, int iters,
                     void* flushbuf, size_t flushbytes, cudaStream_t st, double* out_ms) {
    constexpr int TH = NW * 32;
    constexpr int MFRAG = BM / 16;
    constexpr int WN = BN / NW, NFRAG = WN / 8;
    (void)MFRAG; (void)NFRAG;
    size_t smem = (size_t)STAGES * (BM * (BK + 8) + BN * (BK + 8)) * sizeof(__nv_bfloat16);
    if (smem > 101376) { if (out_ms) *out_ms = 0; return 0.0; } /* > 99KB: skip */
    static size_t last = 0;
    if (smem != last) { CK(cudaFuncSetAttribute(k_tc<BM,BN,BK,NW,STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); last = smem; }
    int tiles_n = (N + BN - 1) / BN;
    int njob = tiles_n * ksplit;
    /* persistent-ish grid: cap so we don't massively oversubscribe, but cover the GPU */
    int grid = njob < g_sm * 4 ? njob : g_sm * 4;
    if (grid < 1) grid = 1;
    size_t cf_n = (size_t)M * N;
    auto launch = [&]() {
        CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
        k_tc<BM,BN,BK,NW,STAGES><<<grid, TH, smem, st>>>(Cf, A, B, M, N, K, ksplit);
    };
    double ms = time_kernel(launch, iters, true, flushbuf, flushbytes, st);
    /* materialise C for correctness check (one clean run) */
    CK(cudaMemsetAsync(Cf, 0, cf_n * sizeof(float), st));
    k_tc<BM,BN,BK,NW,STAGES><<<grid, TH, smem, st>>>(Cf, A, B, M, N, K, ksplit);
    k_f32_to_bf16<<<(cf_n + 255) / 256, 256, 0, st>>>(C, Cf, cf_n);
    CK(cudaDeviceSynchronize());
    double bytes = (double)N * K * 2.0;
    if (out_ms) *out_ms = ms;
    return bytes / (ms * 1e-3) / 1e9;
}

static double run_baseline(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                           unsigned M, unsigned N, unsigned K, int iters,
                           void* flushbuf, size_t flushbytes, cudaStream_t st) {
    size_t smem = (size_t)(B_STAGES * (B_TABUF + B_TBBUF)) * sizeof(__nv_bfloat16);
    CK(cudaFuncSetAttribute(k_tc_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int grid = g_sm;
    auto launch = [&]() { k_tc_bf16<<<grid, 256, smem, st>>>(C, A, B, M, N, K); };
    double ms = time_kernel(launch, iters, true, flushbuf, flushbytes, st);
    launch(); CK(cudaDeviceSynchronize());
    return (double)N * K * 2.0 / (ms * 1e-3) / 1e9;
}

static double run_ffma(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                       unsigned M, unsigned N, unsigned K, int iters,
                       void* flushbuf, size_t flushbytes, cudaStream_t st) {
    int grid = g_sm * 6; /* WS-GEMV wants many blocks; matches decode megakernel oversub */
    auto launch = [&]() { k_ffma<<<grid, 256, 0, st>>>(C, A, B, M, N, K); };
    double ms = time_kernel(launch, iters, true, flushbuf, flushbytes, st);
    launch(); CK(cudaDeviceSynchronize());
    return (double)N * K * 2.0 / (ms * 1e-3) / 1e9;
}

/* correctness: max relerr vs f32 ref + byte-mismatch count vs a golden bf16 buffer */
static void check(const char* tag, const std::vector<__nv_bfloat16>& got,
                  const std::vector<__nv_bfloat16>& ref_f32bf, const std::vector<__nv_bfloat16>* golden,
                  double* max_relerr, long* mism) {
    double mr = 0; long mm = 0;
    for (size_t i = 0; i < got.size(); i++) {
        float g = __bfloat162float(got[i]), r = __bfloat162float(ref_f32bf[i]);
        double d = fabs(g - r) / (fabs(r) + 1e-4);
        if (d > mr) mr = d;
        if (golden) {
            uint16_t a, b; memcpy(&a, &got[i], 2); memcpy(&b, &(*golden)[i], 2);
            if (a != b) mm++;
        }
    }
    *max_relerr = mr; *mism = golden ? mm : -1;
    (void)tag;
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_sm = prop.multiProcessorCount;
    printf("# GPU %s  SMs=%d  smem/SM=%zuKB\n", prop.name, g_sm, prop.sharedMemPerMultiprocessor/1024);

    const double WALL = 1535.0;
    std::vector<Shape> shapes = {
        {"qkv",    8192,  3840},
        {"o_proj", 3840,  4096},
        {"gate/up",15360, 3840},
        {"down",   3840,  15360},
    };
    std::vector<unsigned> Ms = {1,2,4,8,16,32,64};
    std::string mode = argc > 1 ? argv[1] : "full";

    cudaStream_t st; CK(cudaStreamCreate(&st));
    /* L2 flush buffer: > 96 MiB to evict weights (qkv/o_proj fit in L2 otherwise) */
    size_t flushbytes = 256ull * 1024 * 1024;
    void* flushbuf; CK(cudaMalloc(&flushbuf, flushbytes));
    float* d_sink; CK(cudaMalloc(&d_sink, 4));

    int iters = 30;

    /* ---------- config sweep on `down` M=8 to pick the best TC config ---------- */
    if (mode == "sweep") {
        Shape s = shapes[3]; unsigned M = 8;
        size_t wn = (size_t)s.N * s.K;
        std::vector<__nv_bfloat16> hW(wn), hA((size_t)M * s.K);
        fill_rand(hW, 1); fill_rand(hA, 2);
        __nv_bfloat16 *dW,*dA,*dC; float* dCf;
        CK(cudaMalloc(&dW, wn*2)); CK(cudaMalloc(&dA, (size_t)M*s.K*2));
        CK(cudaMalloc(&dC, (size_t)M*s.N*2)); CK(cudaMalloc(&dCf, (size_t)M*s.N*4));
        CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dA,hA.data(),(size_t)M*s.K*2,cudaMemcpyHostToDevice));
        double ms;
        printf("# SWEEP shape=%s M=%u  (GB/s, %%wall)\n", s.name, M);
#define TRY(BM,BN,BK,NW,ST,KS) { double g=run_tc<BM,BN,BK,NW,ST>(dCf,dC,dA,dW,M,s.N,s.K,KS,iters,flushbuf,flushbytes,st,&ms); \
        if(g>0) printf("  BM%d BN%d BK%-3d W%d S%d ksplit%-2d : %7.1f GB/s  %5.1f%%\n",BM,BN,BK,NW,ST,KS,g,100*g/WALL); }
        /* baseline-ish (BK32) then BK sweep at ksplit1 (N=3840 => only 30 N-tiles) */
        TRY(16,128,32,8,3,1) TRY(16,128,64,8,4,1) TRY(16,64,128,4,4,1)
        /* split-K to flood the 188 SMs for skinny-N */
        TRY(16,128,64,8,4,2) TRY(16,128,64,8,4,4) TRY(16,128,64,8,4,8)
        TRY(16,64,128,4,4,2) TRY(16,64,128,4,4,4) TRY(16,64,128,4,4,8) TRY(16,64,128,4,4,16)
        TRY(16,64,64,4,4,4) TRY(16,64,64,4,4,8) TRY(16,64,64,4,4,16)
        TRY(16,64,128,4,5,8) TRY(16,64,128,4,3,8)
        TRY(16,64,64,2,4,8) TRY(16,64,64,2,4,16)
        TRY(16,128,64,8,5,4) TRY(16,128,64,8,6,4)
        TRY(16,64,128,8,4,8) TRY(16,128,64,4,4,8)
#undef TRY
        return 0;
    }

    /* ---------- wall probe per shape ---------- */
    printf("\n## WALL PROBE (pure weight read, all SMs)\n");
    printf("%-8s %10s %8s %7s\n","shape","bytesMB","GB/s","%wall");
    for (auto& s : shapes) {
        size_t wn = (size_t)s.N * s.K;
        std::vector<__nv_bfloat16> hW(wn); fill_rand(hW,1);
        __nv_bfloat16* dW; CK(cudaMalloc(&dW, wn*2));
        CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
        size_t nelem8 = wn/8;
        int grid = g_sm*8;
        auto launch=[&](){ k_stream_reduce<<<grid,256,0,st>>>(dW,nelem8,d_sink); };
        double ms = time_kernel(launch, iters, true, flushbuf, flushbytes, st);
        double gb = (double)wn*2/(ms*1e-3)/1e9;
        printf("%-8s %10.1f %8.1f %6.1f%%\n", s.name, wn*2/1048576.0, gb, 100*gb/WALL);
        CK(cudaFree(dW));
    }

    /* ---------- main grid: shape x M ---------- */
    /* config picker (all fit < 99KB smem): BK=64, split-K to flood 188 SMs.
     * M<=16 => BM16/BN128/8w ; M<=32 => BM32/BN64/4w ; M<=64 => BM64/BN64/4w. */
    auto tc_run = [&](unsigned M, __nv_bfloat16* C, float* Cf, const __nv_bfloat16* A,
                      const __nv_bfloat16* B, unsigned N, unsigned K, int ksplit, int it, double* ms)->double {
        if (M <= 16) return run_tc<16,128,64,8,4>(Cf,C,A,B,M,N,K,ksplit,it,flushbuf,flushbytes,st,ms);
        if (M <= 32) return run_tc<32,64,64,4,4>(Cf,C,A,B,M,N,K,ksplit,it,flushbuf,flushbytes,st,ms);
        return run_tc<64,64,64,4,4>(Cf,C,A,B,M,N,K,ksplit,it,flushbuf,flushbytes,st,ms);
    };

    FILE* jf = fopen("perf-data/zg0-tc-baseline.json","w");
    fprintf(jf, "{\n  \"gpu\": \"%s\", \"sm\": %d, \"wall_achievable_gbs\": %.0f,\n", prop.name, g_sm, WALL);
    fprintf(jf, "  \"note\": \"cold single-tensor read; wall_achievable is sustained-2GB; see cold-read ceiling per shape\",\n");
    fprintf(jf, "  \"rows\": [\n");
    bool firstrow = true;

    printf("\n## SHAPE x M   (TC-clean = tuned split-K mma GEMM; wallGB = pure cold read of THIS tensor)\n");
    printf("%-8s %3s | %8s %6s | %8s %6s | %8s | %6s | %8s %6s | %s\n",
        "shape","M","TCcl GB","%1535","TCbase","%1535","FFMA GB","TC/FF","coldrd","%rd","gate");
    for (auto& s : shapes) {
        size_t wn = (size_t)s.N * s.K;
        std::vector<__nv_bfloat16> hW(wn); fill_rand(hW,1);
        __nv_bfloat16* dW; CK(cudaMalloc(&dW, wn*2));
        CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
        /* per-shape cold pure-read ceiling */
        double wallgb; {
            size_t ne8=wn/8; int grid=g_sm*8;
            auto l=[&](){ k_stream_reduce<<<grid,256,0,st>>>(dW,ne8,d_sink); };
            double ms=time_kernel(l,iters,true,flushbuf,flushbytes,st); wallgb=(double)wn*2/(ms*1e-3)/1e9;
        }
        for (unsigned M : Ms) {
            std::vector<__nv_bfloat16> hA((size_t)M*s.K); fill_rand(hA,2+M);
            __nv_bfloat16 *dA,*dC,*dCb,*dCf16; float* dCf;
            CK(cudaMalloc(&dA,(size_t)M*s.K*2)); CK(cudaMalloc(&dC,(size_t)M*s.N*2));
            CK(cudaMalloc(&dCb,(size_t)M*s.N*2)); CK(cudaMalloc(&dCf16,(size_t)M*s.N*2));
            CK(cudaMalloc(&dCf,(size_t)M*s.N*4));
            CK(cudaMemcpy(dA,hA.data(),(size_t)M*s.K*2,cudaMemcpyHostToDevice));
            double ms;
            /* ksplit so njob ~>= 4*SM (BN=128 for M<=16 else 64) */
            int bn = (M<=16)?128:64; int tiles_n=(s.N+bn-1)/bn;
            int ksplit = (4*g_sm + tiles_n - 1)/tiles_n; if(ksplit<1) ksplit=1; if(ksplit>16) ksplit=16;
            double tccl = tc_run(M,dC,dCf,dA,dW,s.N,s.K,ksplit,iters,&ms);
            double tcb  = run_baseline(dCb,dA,dW,M,s.N,s.K,iters,flushbuf,flushbytes,st);
            double ff   = run_ffma(dCf16,dA,dW,M,s.N,s.K,iters,flushbuf,flushbytes,st);

            /* correctness: RMS-relative vs f32 ref (both TC-clean & baseline) + bit-exact TCclean(split=1) vs baseline */
            std::vector<__nv_bfloat16> gTC((size_t)M*s.N), gBASE((size_t)M*s.N), gREF((size_t)M*s.N);
            CK(cudaMemcpy(gTC.data(),dC,(size_t)M*s.N*2,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gBASE.data(),dCb,(size_t)M*s.N*2,cudaMemcpyDeviceToHost));
            {   __nv_bfloat16* dR; CK(cudaMalloc(&dR,(size_t)M*s.N*2));
                dim3 gb((s.N+127)/128, M); k_ref_f32<<<gb,128,0,st>>>(dR,dA,dW,M,s.N,s.K);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(gREF.data(),dR,(size_t)M*s.N*2,cudaMemcpyDeviceToHost)); CK(cudaFree(dR)); }
            double sumsq=0, eTC=0, eBASE=0;
            for (size_t i=0;i<gREF.size();i++){ double r=__bfloat162float(gREF[i]); sumsq+=r*r;
                double a=fabs(__bfloat162float(gTC[i])-r); if(a>eTC)eTC=a;
                double b=fabs(__bfloat162float(gBASE[i])-r); if(b>eBASE)eBASE=b; }
            double rms=sqrt(sumsq/gREF.size())+1e-9;
            double rmsTC=eTC/rms, rmsBASE=eBASE/rms;
            /* bit-exact TCclean(split=1) vs baseline */
            long bit_mm=0; {
                std::vector<__nv_bfloat16> g1((size_t)M*s.N);
                tc_run(M,dC,dCf,dA,dW,s.N,s.K,1,3,&ms);
                CK(cudaMemcpy(g1.data(),dC,(size_t)M*s.N*2,cudaMemcpyDeviceToHost));
                for (size_t i=0;i<g1.size();i++){ uint16_t a,b; memcpy(&a,&g1[i],2); memcpy(&b,&gBASE[i],2); if(a!=b) bit_mm++; }
            }
            const char* gate = (rmsTC<0.05 && rmsBASE<0.05 && bit_mm==0) ? "PASS" :
                               (rmsTC<0.05 && rmsBASE<0.05) ? "rel-ok/BITX" : "FAIL";
            printf("%-8s %3u | %8.1f %5.1f%% | %8.1f %5.1f%% | %8.1f | %5.2fx | %8.1f %5.1f%% | %s(bit=%ld rmsTC=%.1e)\n",
                s.name, M, tccl, 100*tccl/WALL, tcb, 100*tcb/WALL, ff, tccl/ff, wallgb, 100*wallgb/WALL,
                gate, bit_mm, rmsTC);
            fprintf(jf, "%s    {\"shape\":\"%s\",\"N\":%u,\"K\":%u,\"M\":%u,\"tc_clean_gbs\":%.1f,"
                "\"tc_clean_pct1535\":%.1f,\"tc_clean_pct_coldread\":%.1f,\"tc_base_c1t_gbs\":%.1f,"
                "\"ffma_gbs\":%.1f,\"tc_over_ffma\":%.3f,\"cold_read_ceiling_gbs\":%.1f,"
                "\"ksplit\":%d,\"bit_mismatch\":%ld,\"rms_relerr_tc\":%.2e}",
                firstrow?"":",\n", s.name, s.N, s.K, M, tccl, 100*tccl/WALL, 100*tccl/wallgb, tcb,
                ff, tccl/ff, wallgb, ksplit, bit_mm, rmsTC);
            firstrow=false;
            CK(cudaFree(dA));CK(cudaFree(dC));CK(cudaFree(dCb));CK(cudaFree(dCf16));CK(cudaFree(dCf));
        }
        CK(cudaFree(dW));
    }
    fprintf(jf, "\n  ]\n}\n"); fclose(jf);
    printf("\n# wrote perf-data/zg0-tc-baseline.json\n");

    /* ---------- large-stream clincher: does the SAME TC GEMM reach the wall on a big cold read?
     * A synthetic K=3840 weight at growing N. If %1535 climbs toward ~95% as the footprint grows,
     * the per-tensor 65% is a small-cold-tensor ramp effect, not a TC-GEMM-quality ceiling. ---- */
    printf("\n## LARGE-STREAM (K=3840, M=8, TC-clean, cold) — footprint vs %%1535\n");
    printf("%-9s %10s %8s %6s | %8s %6s (pure read)\n","N","weightMB","TCcl GB","%1535","rd GB","%1535");
    unsigned Kbig=3840, Mbig=8;
    for (unsigned N : {8192u, 32768u, 131072u, 524288u}) {
        size_t wn=(size_t)N*Kbig; std::vector<__nv_bfloat16> hW(wn); fill_rand(hW,1);
        __nv_bfloat16* dW; CK(cudaMalloc(&dW,wn*2)); CK(cudaMemcpy(dW,hW.data(),wn*2,cudaMemcpyHostToDevice));
        std::vector<__nv_bfloat16> hA((size_t)Mbig*Kbig); fill_rand(hA,9);
        __nv_bfloat16 *dA,*dC; float* dCf;
        CK(cudaMalloc(&dA,(size_t)Mbig*Kbig*2)); CK(cudaMalloc(&dC,(size_t)Mbig*N*2)); CK(cudaMalloc(&dCf,(size_t)Mbig*N*4));
        CK(cudaMemcpy(dA,hA.data(),(size_t)Mbig*Kbig*2,cudaMemcpyHostToDevice));
        double ms; double g=run_tc<16,128,64,8,4>(dCf,dC,dA,dW,Mbig,N,Kbig,1,iters,flushbuf,flushbytes,st,&ms);
        double rd; { size_t ne8=wn/8; int grid=g_sm*8; auto l=[&](){k_stream_reduce<<<grid,256,0,st>>>(dW,ne8,d_sink);};
            double m2=time_kernel(l,iters,true,flushbuf,flushbytes,st); rd=(double)wn*2/(m2*1e-3)/1e9; }
        printf("%-9u %10.1f %8.1f %5.1f%% | %8.1f %5.1f%%\n", N, wn*2/1048576.0, g, 100*g/WALL, rd, 100*rd/WALL);
        CK(cudaFree(dW));CK(cudaFree(dA));CK(cudaFree(dC));CK(cudaFree(dCf));
    }
    return 0;
}
