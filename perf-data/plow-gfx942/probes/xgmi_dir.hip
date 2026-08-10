/* xGMI DIRECTION + PROTOCOL probe for the TP8 all-reduce on 8x MI300X (gfx942).
 *
 * `xrbw.hip` answered "which visit ORDER and which load WIDTH" for the PULL path and the
 * answer shipped (stagger, 2 B scalar loads). This probe answers the question that one
 * could not:
 *
 *   plow's two-shot is PULL-ONLY in both phases (reduce-scatter READS N peers' partials;
 *   all-gather READS N peers' reduced slices). RCCL and vLLM's custom all-reduce are
 *   PUSH-based (each rank WRITES into peers). A remote read is a request plus a response
 *   and is capped by outstanding-request depth; a remote write is posted. If the fabric is
 *   materially faster one way, the algorithm is wrong by DIRECTION, not by order.
 *
 * Sections, all at the real GLM TP8 traffic shape (n = 8192*6144 bf16, slice = n/8):
 *   [A] one pair only, nothing else contending -> PER-LINK read and write ceiling
 *   [B] all 8 devices concurrent -> aggregate per-rank all-gather, pull vs push
 *   [C] the reduce-scatter shapes, pull (ships) vs push
 *   [D] rendezvous cost: an N-way system-scope xctr gate round, the term that decides
 *       one-shot vs two-shot at a given message size
 *
 * Build (OUTSIDE nix):
 *   /opt/rocm-7.2.4/bin/hipcc -O3 --offload-arch=gfx942 xgmi_dir.hip -o /tmp/xgmi_dir
 */
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>

#define CHECK(x)                                                        \
    do {                                                                \
        hipError_t e = (x);                                             \
        if (e != hipSuccess) {                                          \
            printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); \
            return 1;                                                   \
        }                                                               \
    } while (0)

static const int NR = 8;
/* One all-gather slice at the widest GLM prefill bucket: 8192 rows * 6144 hidden / 8. */
static const size_t SLICE = 6291456; /* ushort elements = 12.6 MB */
static const int WG = 304, THR = 256;

/* ---- [A] one-pair link ceiling ------------------------------------------------- */
__global__ void k_pair(ushort* __restrict__ dst, const ushort* __restrict__ src, size_t n, int w) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x * blockDim.x;
    if (w == 16) {
        const uint4* s = (const uint4*)src;
        uint4* d = (uint4*)dst;
        for (size_t e = t; e < n / 8; e += st) d[e] = s[e];
    } else {
        for (size_t e = t; e < n; e += st) dst[e] = src[e];
    }
}

/* ---- [B] all-gather: PULL (ships today) vs PUSH -------------------------------- */
/* PULL: every rank reads all NR slices, staggered start -- the shipped order. */
__global__ void k_ag_pull(ushort* dst, ushort* const* srcs, int rank, size_t slice, int w) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < NR; i++) {
        const int s = (int)((wg + rank + i) % NR);
        const ushort* src = srcs[s];
        ushort* d = dst + (size_t)s * slice;
        if (w == 16) {
            const uint4* s4 = (const uint4*)src;
            uint4* d4 = (uint4*)d;
            for (size_t e = (size_t)wg * nt + tid; e < slice / 8; e += (size_t)nwg * nt)
                d4[e] = s4[e];
        } else {
            for (size_t e = (size_t)wg * nt + tid; e < slice; e += (size_t)nwg * nt)
                d[e] = src[e];
        }
    }
}
/* PUSH: every rank writes ITS OWN slice into all NR peers' destinations, staggered.
 * Same bytes on the fabric (7/8 remote), opposite direction. */
__global__ void k_ag_push(ushort* const* dsts, const ushort* src, int rank, size_t slice, int w) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < NR; i++) {
        const int p = (int)((wg + rank + i) % NR);
        ushort* d = dsts[p] + (size_t)rank * slice;
        if (w == 16) {
            const uint4* s4 = (const uint4*)src;
            uint4* d4 = (uint4*)d;
            for (size_t e = (size_t)wg * nt + tid; e < slice / 8; e += (size_t)nwg * nt)
                d4[e] = s4[e];
        } else {
            for (size_t e = (size_t)wg * nt + tid; e < slice; e += (size_t)nwg * nt)
                d[e] = src[e];
        }
    }
}

/* ---- [C] reduce-scatter: PULL (ships today) vs PUSH ----------------------------- */
/* PULL RS as SHIPPED: `nr` is a RUNTIME argument, exactly as `nranks` is in the kernel, so
 * LLVM cannot unroll the peer loop and the emitted ISA is the shipped one -- a pointer-table
 * re-load and s_waitcnt vmcnt(0), then the remote load and another vmcnt(0), per peer per
 * element. Passing NR as a compile-time constant here (the first cut of this probe) silently
 * measures the BATCHED form and reports a 2% difference that is not the difference. */
__global__ void k_rs_pull(ushort* dst, ushort* const* parts, int rank, size_t slice, int nr) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x * blockDim.x;
    const size_t lo = (size_t)rank * slice;
    for (size_t e = lo + t; e < lo + slice; e += st) {
        float acc = 0.f;
        for (int r = 0; r < nr; r++) acc += (float)parts[r][e];
        dst[e] = (ushort)acc;
    }
}
/* PULL RS, BATCHED: the same reads, but the NR peer base pointers are hoisted out of the
 * element loop and the NR remote loads are issued before any of them is consumed. The
 * shipped kernel does NEITHER -- ISA audit of d_xreduce/d_xreduce_twoshot_mega PHASE 1 on
 * gfx942 shows, per element per peer:
 *     global_load_dwordx2  (re-read peer_scratch[r] from memory)
 *     s_waitcnt vmcnt(0)
 *     global_load_ushort   (the remote bf16)
 *     s_waitcnt vmcnt(0)
 *     v_add_f32
 * i.e. TWO full memory stalls per peer and ZERO peer-level memory parallelism. Same sum
 * order (r = 0..NR-1, f32 accumulate), so a kernel built this way is bit-identical. */
__global__ void k_rs_pull_mlp(ushort* dst, ushort* const* parts, int rank, size_t slice) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x * blockDim.x;
    const size_t lo = (size_t)rank * slice;
    const ushort* p[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) p[r] = parts[r];
    for (size_t e = lo + t; e < lo + slice; e += st) {
        ushort v[NR];
#pragma unroll
        for (int r = 0; r < NR; r++) v[r] = p[r][e];
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < NR; r++) acc += (float)v[r];
        dst[e] = (ushort)acc;
    }
}

/* PUSH RS: this rank writes slice s of its own partial into peer s's staging area, so the
 * reduce that follows is entirely LOCAL. Staggered peer order, same as the AG. */
__global__ void k_rs_push(ushort* const* stage, const ushort* mypart, int rank, size_t slice,
                          int w) {
    const unsigned wg = blockIdx.x, nwg = gridDim.x;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < NR; i++) {
        const int s = (int)((wg + rank + i) % NR);
        const ushort* src = mypart + (size_t)s * slice;
        ushort* d = stage[s] + (size_t)rank * slice;
        if (w == 16) {
            const uint4* s4 = (const uint4*)src;
            uint4* d4 = (uint4*)d;
            for (size_t e = (size_t)wg * nt + tid; e < slice / 8; e += (size_t)nwg * nt)
                d4[e] = s4[e];
        } else {
            for (size_t e = (size_t)wg * nt + tid; e < slice; e += (size_t)nwg * nt)
                d[e] = src[e];
        }
    }
}

/* ---- [D] rendezvous latency ----------------------------------------------------- */
/* One xctr-style gate, exactly op_collective.h's discipline: signal every peer with a
 * system-scope release RMW, poll the local word relaxed until the arrivals land, then one
 * system acquire. `nsig` workgroups signal (1 = gate_rs, WG = gate_ag). */
__global__ void k_gate(unsigned* const* gates, unsigned* mygate, int iters, int nsig) {
    for (int it = 1; it <= iters; it++) {
        if (blockIdx.x < (unsigned)nsig && threadIdx.x == 0)
            for (int r = 0; r < NR; r++)
                __hip_atomic_fetch_add(gates[r], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (threadIdx.x == 0) {
            const unsigned target = (unsigned)(NR * nsig * it);
            while (__hip_atomic_load(mygate, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < target)
                __builtin_amdgcn_s_sleep(2);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        }
        __syncthreads();
    }
}

int main() {
    ushort *bufs[NR], *dsts[NR], *stage[NR];
    ushort **dsrcs[NR], **ddsts[NR], **dstage[NR];
    unsigned* gate[NR];
    unsigned** dgate[NR];
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        for (int p = 0; p < NR; p++)
            if (p != d) hipDeviceEnablePeerAccess(p, 0);
        CHECK(hipMalloc(&bufs[d], (size_t)NR * SLICE * 2)); /* a whole partial */
        CHECK(hipMalloc(&dsts[d], (size_t)NR * SLICE * 2));
        CHECK(hipMalloc(&stage[d], (size_t)NR * SLICE * 2));
        CHECK(hipMemset(bufs[d], 0x3c, (size_t)NR * SLICE * 2));
        CHECK(hipMalloc(&gate[d], 256));
        CHECK(hipMemset(gate[d], 0, 256));
    }
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc(&dsrcs[d], NR * sizeof(void*)));
        CHECK(hipMalloc(&ddsts[d], NR * sizeof(void*)));
        CHECK(hipMalloc(&dstage[d], NR * sizeof(void*)));
        CHECK(hipMalloc(&dgate[d], NR * sizeof(void*)));
        CHECK(hipMemcpy(dsrcs[d], bufs, NR * sizeof(void*), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ddsts[d], dsts, NR * sizeof(void*), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dstage[d], stage, NR * sizeof(void*), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dgate[d], gate, NR * sizeof(void*), hipMemcpyHostToDevice));
    }
    hipStream_t st[NR];
    for (int d = 0; d < NR; d++) {
        CHECK(hipSetDevice(d));
        CHECK(hipStreamCreate(&st[d]));
    }
    const int REP = 5;
    auto sync_all = [&]() {
        for (int d = 0; d < NR; d++) {
            hipSetDevice(d);
            hipStreamSynchronize(st[d]);
        }
    };

    printf("== [A] ONE PAIR ACTIVE (per-link ceiling), 12.6 MB, GPU0 <-> GPU1\n");
    for (int w : {2, 16})
        for (int dir = 0; dir < 2; dir++) {
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                hipSetDevice(0);
                hipStreamSynchronize(st[0]);
                auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < REP; k++) {
                    hipSetDevice(0);
                    if (dir == 0)
                        hipLaunchKernelGGL(k_pair, dim3(WG), dim3(THR), 0, st[0], dsts[0], bufs[1],
                                           SLICE, w); /* PULL: read peer */
                    else
                        hipLaunchKernelGGL(k_pair, dim3(WG), dim3(THR), 0, st[0], dsts[1], bufs[0],
                                           SLICE, w); /* PUSH: write peer */
                }
                hipSetDevice(0);
                hipStreamSynchronize(st[0]);
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count() /
                            REP;
                if (rep && ms < best) best = ms;
            }
            printf("   %2dB %-5s : %7.3f ms  %7.1f GB/s\n", w, dir ? "PUSH" : "PULL", best,
                   SLICE * 2.0 / best / 1e6);
        }

    printf("== [B] ALL-GATHER, all 8 devices concurrent (7/8 of the volume is remote)\n");
    const double ag_bytes = (double)NR * SLICE * 2;
    for (int w : {2, 16})
        for (int push = 0; push < 2; push++) {
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                sync_all();
                auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < REP; k++)
                    for (int d = 0; d < NR; d++) {
                        hipSetDevice(d);
                        if (push)
                            hipLaunchKernelGGL(k_ag_push, dim3(WG), dim3(THR), 0, st[d], ddsts[d],
                                               bufs[d], d, SLICE, w);
                        else
                            hipLaunchKernelGGL(k_ag_pull, dim3(WG), dim3(THR), 0, st[d], dsts[d],
                                               dsrcs[d], d, SLICE, w);
                    }
                sync_all();
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count() /
                            REP;
                if (rep && ms < best) best = ms;
            }
            printf("   %2dB %-5s : %7.3f ms  %7.1f GB/s per rank\n", w, push ? "PUSH" : "PULL",
                   best, ag_bytes / best / 1e6);
        }

    printf("== [C] REDUCE-SCATTER, all 8 devices concurrent\n");
    const double rs_bytes = (double)NR * SLICE * 2; /* NR partials over one slice */
    {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            sync_all();
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < REP; k++)
                for (int d = 0; d < NR; d++) {
                    hipSetDevice(d);
                    hipLaunchKernelGGL(k_rs_pull, dim3(WG), dim3(THR), 0, st[d], dsts[d], dsrcs[d],
                                       d, SLICE, NR);
                }
            sync_all();
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count() /
                REP;
            if (rep && ms < best) best = ms;
        }
        printf("      PULL + f32 reduce, SERIAL peers (ships) : %7.3f ms  %7.1f GB/s per rank\n",
               best, rs_bytes / best / 1e6);
    }
    {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            sync_all();
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < REP; k++)
                for (int d = 0; d < NR; d++) {
                    hipSetDevice(d);
                    hipLaunchKernelGGL(k_rs_pull_mlp, dim3(WG), dim3(THR), 0, st[d], dsts[d],
                                       dsrcs[d], d, SLICE);
                }
            sync_all();
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count() /
                REP;
            if (rep && ms < best) best = ms;
        }
        printf("      PULL + f32 reduce, BATCHED peers        : %7.3f ms  %7.1f GB/s per rank\n",
               best, rs_bytes / best / 1e6);
    }
    for (int w : {2, 16}) {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            sync_all();
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < REP; k++)
                for (int d = 0; d < NR; d++) {
                    hipSetDevice(d);
                    hipLaunchKernelGGL(k_rs_push, dim3(WG), dim3(THR), 0, st[d], dstage[d], bufs[d],
                                       d, SLICE, w);
                }
            sync_all();
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count() /
                REP;
            if (rep && ms < best) best = ms;
        }
        printf("   %2dB PUSH scatter only (reduce would then be LOCAL) : %7.3f ms  %7.1f GB/s\n", w,
               best, rs_bytes / best / 1e6);
    }

    printf("== [E] ALL-GATHER (2 B pull, shipped shape) vs WORKGROUP COUNT -- does the fabric\n");
    printf("       need the whole machine? Every CU inside the collective is a CU the next\n");
    printf("       packet cannot claim, so the saturation point is what PLOW_XR_CUS may be set to.\n");
    for (int wg : {304, 152, 76, 38, 19}) {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            sync_all();
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < REP; k++)
                for (int d = 0; d < NR; d++) {
                    hipSetDevice(d);
                    hipLaunchKernelGGL(k_ag_pull, dim3(wg), dim3(THR), 0, st[d], dsts[d], dsrcs[d],
                                       d, SLICE, 2);
                }
            sync_all();
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count() /
                REP;
            if (rep && ms < best) best = ms;
        }
        printf("   WG=%3d : %7.3f ms  %7.1f GB/s per rank\n", wg, best, ag_bytes / best / 1e6);
    }

    printf("== [D] RENDEZVOUS: N-way system-scope xctr gate round, 304 WGs polling\n");
    for (int nsig : {1, 304}) {
        const int ITER = 200;
        for (int d = 0; d < NR; d++) {
            hipSetDevice(d);
            CHECK(hipMemsetAsync(gate[d], 0, 256, st[d]));
        }
        sync_all();
        auto t0 = std::chrono::steady_clock::now();
        for (int d = 0; d < NR; d++) {
            hipSetDevice(d);
            hipLaunchKernelGGL(k_gate, dim3(WG), dim3(THR), 0, st[d], dgate[d], gate[d], ITER,
                               nsig);
        }
        sync_all();
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("   signallers=%3d : %7.3f us / gate round (%d rounds, launch amortised)\n", nsig,
               ms * 1000.0 / ITER, ITER);
    }
    return 0;
}
