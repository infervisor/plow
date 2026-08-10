/* CU-SUBSET FEASIBILITY PROBE for the banded two-shot all-reduce (8x MI300X).
 *
 * The banded TP seam (PLOW_GLM_XR_BAND=K) wants to run band i's collective on a
 * SUBSET of c CUs while the other 304-c CUs compute band i+1. Two quantities decide
 * whether that can ever pay, and neither was in the tree:
 *
 *   (A) rate(c)  -- peer-read bandwidth of the AG phase as a function of the workgroup
 *       count c. If the collective is CU-THROUGHPUT bound, rate(c) ~ c/304 * rate(304)
 *       and the subset buys nothing: donating c CUs to the collective slows the
 *       producer by exactly the work the collective would have done anyway. The lever
 *       needs rate(c)/rate(304) >> c/304, i.e. the fabric saturating well below the
 *       full grid.
 *
 *   (B) rendezvous(c) -- the cost of ONE two-shot rendezvous pair as a function of the
 *       participating workgroup count. gate_ag's threshold is nranks*nblk, so every
 *       workgroup on every rank does a remote system-scope RMW to every peer:
 *       nranks*nblk remote atomics per rank per collective. Banding multiplies the
 *       rendezvous COUNT by K, so this is the cost model for "K bands cost K full-grid
 *       synchronizations".
 *
 * Both kernels reproduce the shipped protocol exactly: 512 threads/WG (PLOW_THREADS),
 * staggered peer visit order (slice+rank+i)%N, scalar 2 B loads, relaxed poll +
 * one system acquire, release RMW signal.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

static const int NR = 8;
static const int THREADS = 512;                 /* PLOW_THREADS */
static const size_t SLICE = 6291456;            /* elements per slice: 12.6 MB, the @8k shape */
static const unsigned CTR_STRIDE = 32u;         /* PLOW_CTR_STRIDE, 128 B/counter */

__device__ __forceinline__ uint32_t xctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void xctr_acquire(void) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }
__device__ __forceinline__ void xctr_signal(uint32_t* p) {
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* (A) AG-phase traffic shape at an arbitrary workgroup count. Exactly
 * d_xreduce_twoshot_mega's PHASE 2: slice s = (wg + rank + i) % N, scalar 2 B,
 * grid-stride within the slice. */
__global__ void k_ag(ushort* __restrict__ dst, ushort* const* srcs, int rank, size_t slice_elems) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = wg * blockDim.x + threadIdx.x;
    const unsigned stride = nwg * blockDim.x;
    for (int i = 0; i < NR; i++) {
        const unsigned s = (wg + (unsigned)rank + (unsigned)i) % NR;
        const ushort* src = srcs[s];
        const size_t lo = slice_elems * s;
        const size_t hi = lo + slice_elems;
        for (size_t e = lo + tid; e < hi; e += stride) dst[e] = src[e - lo];
    }
}

/* (B) the two rendezvous alone, REPS times, no payload. Gate ids advance per iteration
 * so thresholds are exact; the counter region is the same 128 B-strided layout. */
__global__ void k_rdv(uint32_t* const* gates, int rank, int nwg_target, int reps) {
    const unsigned wg = blockIdx.x;
    for (int r = 0; r < reps; r++) {
        uint32_t* const* g = gates;
        const unsigned id_rs = (unsigned)(2 * r);
        const unsigned id_ag = (unsigned)(2 * r + 1);
        /* gate_rs: slice 0 speaks for the rank, threshold nranks */
        if (wg == 0 && threadIdx.x == 0)
            for (int p = 0; p < NR; p++) xctr_signal(g[p] + id_rs * CTR_STRIDE);
        if (threadIdx.x == 0) {
            uint32_t* lg = g[rank] + id_rs * CTR_STRIDE;
            while (xctr_poll(lg) < (uint32_t)NR) __builtin_amdgcn_s_sleep(2);
            xctr_acquire();
        }
        __syncthreads();
        /* gate_ag: EVERY workgroup signals every peer, threshold nranks*nblk */
        if (threadIdx.x == 0)
            for (int p = 0; p < NR; p++) xctr_signal(g[p] + id_ag * CTR_STRIDE);
        if (threadIdx.x == 0) {
            uint32_t* lg = g[rank] + id_ag * CTR_STRIDE;
            const uint32_t target = (uint32_t)(NR * nwg_target);
            while (xctr_poll(lg) < target) __builtin_amdgcn_s_sleep(2);
            xctr_acquire();
        }
        __syncthreads();
    }
}

int main() {
    ushort* bufs[NR];
    ushort* dsts[NR];
    ushort** dsrcs[NR];
    uint32_t* ctrs[NR];
    uint32_t** dctrs[NR];
    const int REPS = 200;
    const size_t CTR_WORDS = (size_t)2 * REPS * CTR_STRIDE;

    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        for (int p = 0; p < NR; p++)
            if (p != d) hipDeviceEnablePeerAccess(p, 0);
        CHECK(hipMalloc(&bufs[d], SLICE * 2));
        CHECK(hipMalloc(&dsts[d], (size_t)NR * SLICE * 2));
        CHECK(hipMemset(bufs[d], 0x3c, SLICE * 2));
        CHECK(hipMalloc(&ctrs[d], CTR_WORDS * sizeof(uint32_t)));
    }
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc(&dsrcs[d], NR * sizeof(void*)));
        CHECK(hipMemcpy(dsrcs[d], bufs, NR * sizeof(void*), hipMemcpyHostToDevice));
        CHECK(hipMalloc(&dctrs[d], NR * sizeof(void*)));
        CHECK(hipMemcpy(dctrs[d], ctrs, NR * sizeof(void*), hipMemcpyHostToDevice));
    }
    hipStream_t st[NR];
    for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamCreate(&st[d])); }

    const int wgs[] = {304, 152, 76, 64, 38, 32, 19, 16, 8};
    const int NW = (int)(sizeof(wgs) / sizeof(wgs[0]));

    printf("=== (A) AG peer-read rate vs workgroup count (2B scalar, staggered, %d thr/WG)\n", THREADS);
    printf("  full message = %.1f MB/rank read (7/8 remote)\n", (double)NR * SLICE * 2 / 1e6);
    double r304 = 0;
    for (int w = 0; w < NW; w++) {
        const int nwg = wgs[w];
        for (int d = 0; d < NR; d++) {
            CHECK(hipSetDevice(d));
            hipLaunchKernelGGL(k_ag, dim3(nwg), dim3(THREADS), 0, st[d], dsts[d], dsrcs[d], d, SLICE);
        }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        auto t0 = std::chrono::steady_clock::now();
        const int REP = 5;
        for (int rep = 0; rep < REP; rep++)
            for (int d = 0; d < NR; d++) {
                CHECK(hipSetDevice(d));
                hipLaunchKernelGGL(k_ag, dim3(nwg), dim3(THREADS), 0, st[d], dsts[d], dsrcs[d], d, SLICE);
            }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / REP;
        double gbs = (double)NR * SLICE * 2 / ms / 1e6;
        if (w == 0) r304 = gbs;
        printf("  nwg=%3d  %7.3f ms  %7.1f GB/s   rate/rate304=%.3f  cu/304=%.3f  EFF=%.2fx\n",
               nwg, ms, gbs, gbs / r304, (double)nwg / 304.0,
               (gbs / r304) / ((double)nwg / 304.0));
    }

    printf("\n=== (C) is the controlling variable WORKGROUPS or THREADS IN FLIGHT?\n");
    printf("  same AG traffic, (nwg x threads) held against each other\n");
    const int cfg[][2] = {{304, 512}, {304, 256}, {304, 128}, {304, 64},
                          {152, 512}, {152, 256}, {608, 256}, {76, 512}, {608, 128}};
    for (unsigned w = 0; w < sizeof(cfg) / sizeof(cfg[0]); w++) {
        const int nwg = cfg[w][0], nth = cfg[w][1];
        for (int d = 0; d < NR; d++) {
            CHECK(hipSetDevice(d));
            hipLaunchKernelGGL(k_ag, dim3(nwg), dim3(nth), 0, st[d], dsts[d], dsrcs[d], d, SLICE);
        }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        auto t0 = std::chrono::steady_clock::now();
        const int REP = 5;
        for (int rep = 0; rep < REP; rep++)
            for (int d = 0; d < NR; d++) {
                CHECK(hipSetDevice(d));
                hipLaunchKernelGGL(k_ag, dim3(nwg), dim3(nth), 0, st[d], dsts[d], dsrcs[d], d, SLICE);
            }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / REP;
        printf("  nwg=%3d x %3d thr = %6d threads  %7.3f ms  %7.1f GB/s\n", nwg, nth,
               nwg * nth, ms, (double)NR * SLICE * 2 / ms / 1e6);
    }

    printf("\n=== (B) two-shot RENDEZVOUS PAIR cost vs participating workgroups (no payload)\n");
    for (int w = 0; w < NW; w++) {
        const int nwg = wgs[w];
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipMemset(ctrs[d], 0, CTR_WORDS * sizeof(uint32_t))); }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipDeviceSynchronize()); }
        auto t0 = std::chrono::steady_clock::now();
        for (int d = 0; d < NR; d++) {
            CHECK(hipSetDevice(d));
            hipLaunchKernelGGL(k_rdv, dim3(nwg), dim3(THREADS), 0, st[d], dctrs[d], d, nwg, REPS);
        }
        for (int d = 0; d < NR; d++) { CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d])); }
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count() / REPS;
        printf("  nwg=%3d  %8.2f us / rendezvous pair   (%d remote RMWs per rank per pair)\n",
               nwg, us, NR * (1 + nwg));
    }
    return 0;
}
