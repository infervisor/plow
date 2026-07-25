/* chunk_dbuf_poc.cu — CHUNK-1: measure the prefill M-chunk cross-op pipeline thesis.
 *
 * The persistent interpreter (interp_sm120.cu) runs one packet to completion per block,
 * then claims the next from an atomic cursor and GATES on a per-consumer COUNTER
 * (`while (ctr_poll(counters[id]) < threshold) __nanosleep`). It is NOT a cooperative
 * grid.sync() barrier. BUT for UNISEG prefill `select_granularity` downgrades every edge
 * to COARSE, so the consumer's threshold == the producer's FULL block count (n_cu=188):
 * op[c+1] cannot start until op[c] finished on EVERY SM. That coarse all-SM threshold is a
 * de-facto op-to-op barrier across the whole grid -> no cross-op / cross-chunk overlap.
 *
 * This POC A/B's, on the same data and the production 128x128x32 cp.async GEMM body, a
 * 2-op prefill chain (GEMM1 -> GEMM2, 1:1 M-chunk producer/consumer):
 *
 *   mode 0 SERIAL      : full GEMM1 (all M) -> grid.sync() [ == coarse all-SM barrier ]
 *                        -> full GEMM2. The current model.
 *   mode 1 CHUNK_GQ    : split M into k chunks; per-chunk counters; ONE atomic work cursor
 *                        (op-major worklist, PLOW_NV_SCHED=1 GQ). GEMM2-chunk-c waits ONLY
 *                        on counter[c]; blocks that finished GEMM1-chunk-c roll onto
 *                        GEMM2-chunk-c while OTHER SMs still grind GEMM1-chunk-(c+1).
 *                        Cross-SM overlap, no full-op barrier, no intra-block double-buffer.
 *   mode 2 CHUNK_STATIC: per-chunk counters, but block->chunk STATICALLY pinned so a chunk's
 *                        producer AND consumer run on the SAME SM set. GEMM2-chunk-c reads
 *                        Y-chunk-c from HOT L2 (just written by the same SMs). Trades
 *                        cross-SM overlap for producer->consumer L2 locality.
 *
 * Parity: mode 1/2 output Z must be bit-identical to mode 0 (same math, reordered).
 * Overlap probe: atomic min(GEMM2 first-start clk) / max(GEMM1 last-end clk) over the grid.
 *
 * Build (plain env, NOT nix — nix poisons RUNPATH):
 *   nvcc -arch=sm_120a -O2 -std=c++17 -o /tmp/chunk_poc chunk_dbuf_poc.cu && /tmp/chunk_poc
 */
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;
typedef unsigned long long ull;

#define PLOW_NV_THREADS 256

/* ---- production GEMM tile helpers (verbatim geometry from op_gemm.cuh d_gemm) ---------- */
#define PGM_BM 128
#define PGM_BN 128
#define PGM_BK 32
#define PGM_APAD 8
#define PGM_BPAD 8
#define PGM_WARPS_M 4
#define PGM_WARPS_N 2
#define PGM_WM (PGM_BM / PGM_WARPS_M)
#define PGM_WN (PGM_BN / PGM_WARPS_N)
#define PGM_MFRAG (PGM_WM / 16)
#define PGM_NFRAG (PGM_WN / 8)
#define PGM_ASTRIDE (PGM_BK + PGM_APAD)
#define PGM_BKSTRIDE (PGM_BK + PGM_BPAD)
#define PGM_ABUF (PGM_BM * PGM_ASTRIDE)
#define PGM_BBUF (PGM_BN * PGM_BKSTRIDE)
#ifndef PGM_STAGES
#define PGM_STAGES 3
#endif
#define PGM_ARENA_BF16 (PGM_STAGES * (PGM_ABUF + PGM_BBUF)) /* 30720 bf16 = 60 KiB */

__device__ __forceinline__ void pgm_cp16(void* smem, const void* gmem, int b) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(b));
}
__device__ __forceinline__ void pgm_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void pgm_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldm_x2(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
/* Grid-consistent ns timer (same source across all SMs; clock64() is per-SM and cannot be
 * min/max'd across the grid). */
__device__ __forceinline__ ull gtime() {
    ull t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ void mma_k16(float (&d)[4], const unsigned (&a)[4],
                                        const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void stage_a(bf16* Ad, const bf16* __restrict__ A, int tid,
                                         int tm, int kbase, int m, int k) {
    const int KCH = PGM_BK / 8;
    for (int L = tid; L < PGM_BM * KCH; L += PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int mm = tm + row, kk = kbase + kk8;
        const bool in = (mm < m) && (kk + 8 <= k);
        const bf16* g = in ? A + (size_t)mm * k + kk : A;
        pgm_cp16(&Ad[row * PGM_ASTRIDE + kk8], g, in ? 16 : 0);
    }
}
__device__ __forceinline__ void stage_b(bf16* Bd, const bf16* __restrict__ B, int tid,
                                         int tn, int kbase, int n, int k) {
    const int KCH = PGM_BK / 8;
    for (int L = tid; L < PGM_BN * KCH; L += PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int nn = tn + row, kk = kbase + kk8;
        const bool in = (nn < n) && (kk + 8 <= k);
        const bf16* g = in ? B + (size_t)nn * k + kk : B;
        pgm_cp16(&Bd[row * PGM_BKSTRIDE + kk8], g, in ? 16 : 0);
    }
}

/* C[M,N] = A[M,K] . B[N,K]^T over output tiles in [tile_lo, tile_hi), grid-strided by
 * (rank, nblk). Identical inner loop to production d_gemm (bf16 IO, f32 acc, 3-stage ring). */
__device__ void gemm_band(bf16* __restrict__ C, const bf16* __restrict__ A,
                          const bf16* __restrict__ B, int m, int n, int k,
                          int tile_lo, int tile_hi, int rank, int nblk, bf16* arena) {
    bf16* As = arena;
    bf16* Bs = arena + PGM_STAGES * PGM_ABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_n = (n + PGM_BN - 1) / PGM_BN;
    const int ksteps = (k + PGM_BK - 1) / PGM_BK;

    for (int tile = tile_lo + rank; tile < tile_hi; tile += nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN;
        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
        auto stage = [&](int ks, int buf) {
            stage_a(As + buf * PGM_ABUF, A, tid, tm, ks * PGM_BK, m, k);
            stage_b(Bs + buf * PGM_BBUF, B, tid, tn, ks * PGM_BK, n, k);
        };
#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) { if (s < ksteps) stage(s, s); pgm_commit(); }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_commit();
            pgm_wait<PGM_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_STAGES;
            bf16* Ad = As + cb * PGM_ABUF;
            bf16* Bd = Bs + cb * PGM_BBUF;
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++) {
                    const int arow = wm * PGM_WM + mi * 16 + (lane % 16);
                    const int acol = kf + (lane / 16) * 8;
                    ldm_x4(af[mi], &Ad[arow * PGM_ASTRIDE + acol]);
                }
                unsigned bf[PGM_NFRAG][2];
#pragma unroll
                for (int nj = 0; nj < PGM_NFRAG; nj++) {
                    const int nn = wn * PGM_WN + nj * 8 + (lane & 7);
                    const int kcol = kf + ((lane >> 3) & 1) * 8;
                    ldm_x2(bf[nj], &Bd[nn * PGM_BKSTRIDE + kcol]);
                }
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        mma_k16(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < m && cc < n) C[(size_t)rr * n + cc] = __float2bfloat16(acc[mi][nj][e]);
                }
            }
        __syncthreads();
    }
}

/* ---- POC kernel ------------------------------------------------------------------------ */
struct POC {
    bf16 *X, *W1, *Y, *W2, *Z;
    int M, K1, N1, N2; /* GEMM1: [M,K1]x[N1,K1]->[M,N1].  GEMM2: [M,N1]x[N2,N1]->[M,N2] */
    int k;             /* M-chunks */
    int Sper;          /* slices (blocks) per chunk for GQ worklist */
    unsigned* counters;/* [k] per-chunk producer-done counters */
    unsigned* cursor;  /* [1] GQ atomic work cursor */
    ull* g1_end;       /* atomicMax: clk of last GEMM1 store, grid-wide */
    ull* g2_start;     /* atomicMin: clk of first GEMM2 start, grid-wide */
};

__device__ __forceinline__ void band_of(int c, int k, int M, int tiles_n, int& lo, int& hi) {
    const int mt = (M + PGM_BM - 1) / PGM_BM;
    const int per = (mt + k - 1) / k;
    int m0 = c * per, m1 = min(mt, (c + 1) * per);
    lo = m0 * tiles_n;
    hi = m1 * tiles_n;
}

template <int MODE>
__global__ void __launch_bounds__(256, 1) poc_kernel(POC p) {
    extern __shared__ bf16 arena[];
    const int tid = threadIdx.x;
    const int tiles_n1 = (p.N1 + PGM_BN - 1) / PGM_BN; /* GEMM1 out N1 */
    const int tiles_n2 = (p.N2 + PGM_BN - 1) / PGM_BN; /* GEMM2 out N2 */

    if (MODE == 0) {
        /* SERIAL: full GEMM1 -> coarse all-SM barrier (grid.sync) -> full GEMM2. */
        cg::grid_group grid = cg::this_grid();
        const int mt = (p.M + PGM_BM - 1) / PGM_BM;
        gemm_band(p.Y, p.X, p.W1, p.M, p.N1, p.K1, 0, mt * tiles_n1, blockIdx.x, gridDim.x, arena);
        if (tid == 0) atomicMax(p.g1_end, gtime());
        grid.sync();
        if (tid == 0) atomicMin(p.g2_start, gtime());
        gemm_band(p.Z, p.Y, p.W2, p.M, p.N2, p.N1, 0, mt * tiles_n2, blockIdx.x, gridDim.x, arena);
        return;
    }

    if (MODE == 1) {
        /* CHUNK_GQ: op-major worklist [G1(c,s)]*kS then [G2(c,s)]*kS, one atomic cursor.
         * G2-chunk-c gates on counter[c] only. Cross-SM overlap via dynamic claim. */
        __shared__ unsigned claim;
        const int total = 2 * p.k * p.Sper;
        for (;;) {
            __syncthreads();
            if (tid == 0) claim = atomicAdd(p.cursor, 1u);
            __syncthreads();
            const unsigned w = claim;
            if (w >= (unsigned)total) break;
            const bool g2 = (w >= (unsigned)(p.k * p.Sper));
            const unsigned idx = g2 ? w - p.k * p.Sper : w;
            const int c = idx / p.Sper, s = idx % p.Sper;
            int lo, hi;
            if (!g2) {
                band_of(c, p.k, p.M, tiles_n1, lo, hi);
                gemm_band(p.Y, p.X, p.W1, p.M, p.N1, p.K1, lo, hi, s, p.Sper, arena);
                __syncthreads();
                if (tid == 0) {
                    __threadfence();
                    atomicAdd(&p.counters[c], 1u);
                    atomicMax(p.g1_end, gtime());
                }
            } else {
                /* gate: wait all Sper producer slices of chunk c */
                if (tid == 0) { long z = 0; while (atomicAdd(&p.counters[c], 0u) < (unsigned)p.Sper) { __nanosleep(64); if (++z > 20000000) { printf("HANG GQ c=%d cnt=%u need=%d\n", c, atomicAdd(&p.counters[c],0u), p.Sper); break; } } }
                __syncthreads();
                if (tid == 0) atomicMin(p.g2_start, gtime());
                band_of(c, p.k, p.M, tiles_n2, lo, hi);
                gemm_band(p.Z, p.Y, p.W2, p.M, p.N2, p.N1, lo, hi, s, p.Sper, arena);
            }
        }
        return;
    }

    if (MODE == 2) {
        /* CHUNK_STATIC: contiguous cu_range placement (CHUNK-2 ChunkPlacement::StaticColocated):
         * chunk c owns SM-set [c*n_cu/k, (c+1)*n_cu/k). Producer AND consumer of chunk c run on
         * the SAME contiguous SM set -> GEMM2 reads Y-chunk-c from hot L2 written by these SMs. */
        /* g and cu_range must be MUTUALLY CONSISTENT when nc % k != 0: the set
         * {b : floor(b*k/nc)==g} == [ceil(g*nc/k), ceil((g+1)*nc/k)). Using floor boundaries
         * (c*nc/k) here would leak a boundary block into the wrong chunk and starve a counter
         * -> cooperative-grid deadlock. Ceil boundaries keep the partition exact. */
        const int nc = gridDim.x;
        const int g = (blockIdx.x * p.k) / nc;             /* this block's chunk */
        const int lo_cu = (g * nc + p.k - 1) / p.k;
        const int hi_cu = ((g + 1) * nc + p.k - 1) / p.k;
        const int rank = blockIdx.x - lo_cu;
        const int nblk = hi_cu - lo_cu;                    /* blocks in chunk g's cu_range */
        { const int c = g;
            int lo, hi;
            band_of(c, p.k, p.M, tiles_n1, lo, hi);
            gemm_band(p.Y, p.X, p.W1, p.M, p.N1, p.K1, lo, hi, rank, nblk, arena);
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicAdd(&p.counters[c], 1u);
                atomicMax(p.g1_end, gtime());
            }
            if (tid == 0) { long z = 0; while (atomicAdd(&p.counters[c], 0u) < (unsigned)nblk) { __nanosleep(64); if (++z > 20000000) { printf("HANG STATIC c=%d cnt=%u need=%d\n", c, atomicAdd(&p.counters[c],0u), nblk); break; } } }
            __syncthreads();
            if (tid == 0) atomicMin(p.g2_start, gtime());
            band_of(c, p.k, p.M, tiles_n2, lo, hi);
            gemm_band(p.Z, p.Y, p.W2, p.M, p.N2, p.N1, lo, hi, rank, nblk, arena);
        }
        return;
    }
}

/* ---- host -------------------------------------------------------------------------- */
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static void fill_rand(bf16* h, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        unsigned x = (unsigned)(i * 2654435761u + seed * 40503u);
        float f = ((int)(x & 0xffff) - 32768) / 32768.0f * 0.05f;
        h[i] = __float2bfloat16(f);
    }
}

int main(int argc, char** argv) {
    int M = 2048, K1 = 3840, N1 = 3840, N2 = 3840;
    int iters = 40;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N1 = atoi(argv[2]);
    if (argc > 3) N2 = atoi(argv[3]);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    const size_t smem = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    printf("# GPU %s  SMs %d  L2 %.0f MB  clk %.2f GHz  arena %zu B (%.1f KiB)\n",
           prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1048576.0,
           clk_khz / 1e6, smem, smem / 1024.0);
    printf("# shapes: GEMM1 [%d,%d]x[%d,%d]->[%d,%d] ; GEMM2 [%d,%d]x[%d,%d]->[%d,%d]\n",
           M, K1, N1, K1, M, N1, M, N1, N2, N1, M, N2);

    bf16 *X, *W1, *Y, *W2, *Z;
    CK(cudaMalloc(&X, (size_t)M * K1 * 2));
    CK(cudaMalloc(&W1, (size_t)N1 * K1 * 2));
    CK(cudaMalloc(&Y, (size_t)M * N1 * 2));
    CK(cudaMalloc(&W2, (size_t)N2 * N1 * 2));
    CK(cudaMalloc(&Z, (size_t)M * N2 * 2));
    { std::vector<bf16> h((size_t)M * K1); fill_rand(h.data(), h.size(), 1);
      CK(cudaMemcpy(X, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); }
    { std::vector<bf16> h((size_t)N1 * K1); fill_rand(h.data(), h.size(), 2);
      CK(cudaMemcpy(W1, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); }
    { std::vector<bf16> h((size_t)N2 * N1); fill_rand(h.data(), h.size(), 3);
      CK(cudaMemcpy(W2, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); }

    unsigned *counters, *cursor; ull *g1e, *g2s;
    CK(cudaMalloc(&counters, 256 * 4));
    CK(cudaMalloc(&cursor, 4));
    CK(cudaMalloc(&g1e, 8)); CK(cudaMalloc(&g2s, 8));

    auto setattr = [&](const void* fn) {
        CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    };
    setattr((const void*)poc_kernel<0>);
    setattr((const void*)poc_kernel<1>);
    setattr((const void*)poc_kernel<2>);

    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, (const void*)poc_kernel<0>, 256, smem));
    const int grid = bps * prop.multiProcessorCount;
    printf("# occupancy %d blk/SM  grid %d  (occ-%d)\n", bps, grid, bps);

    const int Ks[] = {2, 4, 8, 16};
    std::vector<bf16> refZ((size_t)M * N2);

    auto run_mode = [&](int mode, int k, int Sper, float& ms, double& overlap_ns, bool capture) {
        POC p{X, W1, Y, W2, Z, M, K1, N1, N2, k, Sper, counters, cursor, g1e, g2s};
        void* args[] = {&p};
        auto launch = [&](void* fn) {
            CK(cudaMemset(counters, 0, 256 * 4));
            CK(cudaMemset(cursor, 0, 4));
            ull hi = 0, lo = ~0ull;
            CK(cudaMemcpy(g1e, &hi, 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(g2s, &lo, 8, cudaMemcpyHostToDevice));
            CK(cudaLaunchCooperativeKernel(fn, dim3(grid), dim3(256), args, smem, 0));
        };
        void* fn = mode == 0 ? (void*)poc_kernel<0> : mode == 1 ? (void*)poc_kernel<1> : (void*)poc_kernel<2>;
        fprintf(stderr, "[run mode=%d k=%d Sper=%d]\n", mode, k, Sper); fflush(stderr);
        launch(fn); CK(cudaDeviceSynchronize());
        if (capture) CK(cudaMemcpy(refZ.data(), Z, refZ.size() * 2, cudaMemcpyDeviceToHost));
        launch(fn); CK(cudaDeviceSynchronize());
        ull he = 0, ls = 0; CK(cudaMemcpy(&he, g1e, 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&ls, g2s, 8, cudaMemcpyDeviceToHost));
        overlap_ns = (double)((long long)he - (long long)ls); /* %globaltimer is already ns */
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) launch(fn);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float tot; CK(cudaEventElapsedTime(&tot, a, b)); ms = tot / iters;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    };

    float ms0; double ov0;
    run_mode(0, 1, 0, ms0, ov0, true);
    printf("\nSERIAL(mode0, coarse grid.sync barrier): %.4f ms/pair   [op-boundary drain=%.1f us]\n",
           ms0, ov0 < 0 ? -ov0 / 1000.0 : 0.0);

    auto parity = [&](const char* tag) {
        std::vector<bf16> z((size_t)M * N2);
        CK(cudaMemcpy(z.data(), Z, z.size() * 2, cudaMemcpyDeviceToHost));
        size_t mism = 0; float maxad = 0;
        for (size_t i = 0; i < z.size(); i++) {
            float a = __bfloat162float(z[i]), b = __bfloat162float(refZ[i]);
            if (a != b) { mism++; maxad = fmaxf(maxad, fabsf(a - b)); }
        }
        printf("    parity[%s]: %s (%zu/%zu mism, maxabs %.3e)\n", tag,
               mism == 0 ? "BIT-IDENTICAL" : "DIFF", mism, z.size(), maxad);
    };

    printf("\n k | GQ ms   d%%    ovlp(us) | STATIC ms  d%%    ovlp(us)\n");
    printf("---+------------------------+-------------------------\n");
    for (int k : Ks) {
        int Sper = (grid + k - 1) / k;
        float ms1, ms2; double ov1, ov2;
        run_mode(1, k, Sper, ms1, ov1, false); parity("GQ");
        run_mode(2, k, Sper, ms2, ov2, false); parity("STATIC");
        printf("%2d | %.4f %+5.1f %7.1f | %.4f %+5.1f %7.1f\n",
               k, ms1, (ms1 - ms0) / ms0 * 100, ov1 > 0 ? ov1 / 1000 : 0.0,
               ms2, (ms2 - ms0) / ms0 * 100, ov2 > 0 ? ov2 / 1000 : 0.0);
    }
    printf("\n# ovlp(us) = max(GEMM1-end-clk) - min(GEMM2-start-clk), grid-wide; >0 => real cross-SM overlap.\n");
    return 0;
}
