// ctr_convergence.hip — WHERE the plow interpreter's per-packet counter protocol spends its time
// on an ALL-WORKGROUP (b=256) packet, which is the shape 1068 of K3 decode's 2459 packets have.
//
// WHY THIS EXISTS SEPARATELY FROM interp_dispatch_floor.hip. That bench measures the NARROW
// (1-workgroup) consumer's floor -- t_arrive/t_ready/t_end of ONE workgroup gated on a 256-way
// producer. It therefore charges the whole 256-way convergence to the consumer's GAP and cannot
// see what the 256 PRODUCERS individually paid. A K3 decode packet trace (PLOW_TRACE_RAW, the
// full 93-layer TP8 model with every body ablated to Nop) says the producers are where the money
// is: for a b=256 packet with an EMPTY body the median workgroup spends
//
//     t_ready -> t_end = 5.72 us
//
// on nothing but `__syncthreads(); ctr_signal(); __syncthreads();`, and the 256 workgroups' t_end
// values are spread over 5.15 us while their t_ready values are spread over only 2.22 us. So the
// spread is CREATED at the signal, not inherited from the gate. This bench isolates which part of
// the signal creates it.
//
// STRUCTURE: a chain of ALL-WORKGROUP packets, each gated on the previous, empty body -- exactly
// the interpreter's loop for the NOP b=256 -> NOP b=256 transition, with interp.hip's exact
// primitive lowering (relaxed agent-scope poll, ONE agent acquire fence after the gate, release
// agent-scope RMW to signal, PLOW_CTR_STRIDE = 128 B per counter, 512 threads, 90 KB LDS so one
// block lands per CU and all 8 XCDs light up).
//
// ARMS (-D):
//   (default)      one counter per step, threshold 256, RELEASE signal, agent ACQUIRE. Faithful.
//   -DRELAXSIG     signal RELAXED -- deletes the `buffer_wbl2` the release RMW carries.
//   -DNOACQ        drop the acquire fence -- deletes the `buffer_inv`.
//   -DFAN=<n>      SPREAD the convergence over <n> counters per step instead of 1. Workgroup w
//                  signals counter (s*FAN + w%FAN); the consumer polls all <n> with threshold
//                  256/n, one THREAD per counter, which is what interp.hip's gate loop already
//                  does for a multi-counter wait list. FAN=8 is the per-XCD hierarchy; FAN=256 is
//                  one counter per workgroup, i.e. zero contention with the identical instruction
//                  count. Both are pure COMPILER-side slot-assignment changes if they win.
//   -DXCCFAN       like -DFAN=8 but the counter is chosen from HW_REG_XCC_ID rather than from
//                  blockIdx.x, so each counter line is written only by workgroups on ONE XCD.
//
// BUILD/RUN (pin an IDLE GPU -- this box is shared):
//   hipcc --offload-arch=gfx950 -O3 -std=c++17 -w ctr_convergence.hip -o ctrconv
//   sg render -c 'LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=5 ./ctrconv 400'
//
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
namespace cg = cooperative_groups;

#define THREADS 512
#define CTR_STRIDE 32u            // PLOW_CTR_STRIDE: 128 B per counter, one cache line
#ifndef NSTEP
#define NSTEP 400
#endif
#ifndef FAN
#define FAN 1
#endif

#define CTR(base, id) ((base) + (size_t)(id) * CTR_STRIDE)

__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
#ifdef RELAXSIG
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); // + buffer_wbl2
#endif
}
__device__ __forceinline__ void ctr_acquire() {
#ifndef NOACQ
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");                          // buffer_inv
#endif
}
__device__ __forceinline__ uint64_t now() { return __builtin_amdgcn_s_memrealtime(); }

struct Args {
    uint32_t* counters;
    uint32_t* lgo;     // [NSTEP][8] per-XCD "gate is open"  (HIER2)
    uint32_t* ldn;     // [NSTEP][8] per-XCD "my stores are in this L2" (HIER2)
    uint64_t* t_arr;   // [NSTEP][NCU]
    uint64_t* t_rdy;
    uint64_t* t_end;
    int NS, ncu;
};

__global__ void __launch_bounds__(THREADS) chain(Args a) {
    __shared__ char pad[90 * 1024];   // 1 block/CU => 256 blocks across all 8 XCDs
    ((volatile char*)pad)[threadIdx.x & 1023] = 0;
    const int bid = blockIdx.x, tid = threadIdx.x;
    cg::grid_group grid = cg::this_grid();
    grid.sync();

#ifdef XCCFAN
    const unsigned mine = __builtin_amdgcn_s_getreg((20 /*HW_REG_XCC_ID*/) | (0 << 6) |
                                                    ((32 - 1) << 11)) & 0xf;
    const unsigned nfan = 8u, thresh = (unsigned)a.ncu / 8u;
#else
    const unsigned mine = (unsigned)bid % (unsigned)FAN;
    const unsigned nfan = (unsigned)FAN, thresh = (unsigned)a.ncu / (unsigned)FAN;
#endif

    /* LEADER: one workgroup per XCD. The gfx950 dispatcher is exactly round-robin over the 8
     * XCDs (runtime/tests/xcd_map_gfx950_test.hip measures 100.0%), so blocks 0..7 are one per
     * XCD. UNSAFE BY CONSTRUCTION -- it bounds the prize of doing the cache maintenance once per
     * L2 instead of once per workgroup, nothing more. */
    const bool leader = (bid < 8);

#ifdef HIER2
    /* TWO-LEVEL CACHE MAINTENANCE. The 13.2 us an empty b=256 packet costs is not the atomic and
     * is mostly not the counter line: it is that all 256 workgroups issue `buffer_wbl2` and
     * `buffer_inv`, which are per-L2 operations, so each XCD's L2 performs the SAME writeback and
     * the SAME invalidate 32 times. This arm does each once per XCD and adds two XCD-LOCAL
     * rendezvous to make that ordering real rather than racy:
     *
     *   publish  follower: __syncthreads() retires its stores into THIS XCD's L2 (s_waitcnt
     *                      vmcnt(0)), then bumps ldn[s][x] RELAXED -- no writeback, the data is
     *                      in the L2 the leader is about to write back.
     *            leader:   waits ldn[s][x] == 31, then ONE release RMW on the global counter --
     *                      its `buffer_wbl2` writes back the whole L2, which is exactly the 32
     *                      workgroups' stores, and then publishes.
     *   observe  leader:   waits the global counter, does ONE `buffer_inv` for the XCD, then bumps
     *                      lgo[s][x].
     *            follower: waits lgo[s][x] and does NO invalidate -- the leader's invalidate
     *                      happened-before the bump it just observed, and this XCD's L2 has been
     *                      fresh ever since.
     *
     * The XCD-local counters never leave their own L2, so their traffic is not fabric traffic.
     * This is the shape a real implementation would take; the arm exists to price it against
     * LEADER_MAINT, which is the same saving with the ordering simply omitted. */
    const unsigned xcd = __builtin_amdgcn_s_getreg((20 /*HW_REG_XCC_ID*/) | (0 << 6) |
                                                   ((32 - 1) << 11)) & 0xf;
    const unsigned nper = (unsigned)a.ncu / 8u;   // workgroups per XCD (round-robin dispatcher)
    for (int s = 0; s < a.NS; s++) {
        const uint64_t ta = now();
        if (s > 0) {
            if (leader) {
                if (tid == 0) {
                    while (ctr_poll(CTR(a.counters, (size_t)(s - 1) * 8 + 0)) < 8u)
                        __builtin_amdgcn_s_sleep(8);
                    ctr_acquire();                                   // ONE buffer_inv per XCD
                    __hip_atomic_fetch_add(CTR(a.lgo, (size_t)(s - 1) * 8 + xcd), 1u,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            } else if (tid == 0) {
                while (ctr_poll(CTR(a.lgo, (size_t)(s - 1) * 8 + xcd)) < 1u)
                    __builtin_amdgcn_s_sleep(8);                     // no invalidate
            }
            __syncthreads();
        }
        const uint64_t tr = now();
        __syncthreads();   // retires this workgroup's stores into its own XCD's L2
        if (tid == 0) {
            if (!leader) {
                __hip_atomic_fetch_add(CTR(a.ldn, (size_t)s * 8 + xcd), 1u,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                while (ctr_poll(CTR(a.ldn, (size_t)s * 8 + xcd)) < nper - 1u)
                    __builtin_amdgcn_s_sleep(8);
                ctr_signal(CTR(a.counters, (size_t)s * 8 + 0));      // ONE buffer_wbl2 per XCD
            }
        }
        __syncthreads();
        const uint64_t te = now();
        if (tid == 0) {
            a.t_arr[(size_t)s * a.ncu + bid] = ta;
            a.t_rdy[(size_t)s * a.ncu + bid] = tr;
            a.t_end[(size_t)s * a.ncu + bid] = te;
        }
    }
    grid.sync();
    return;
#endif

    for (int s = 0; s < a.NS; s++) {
        const uint64_t ta = now();
        if (s > 0) {
            // interp.hip's gate: ONE THREAD PER COUNTER, polled concurrently.
            for (unsigned w = tid; w < nfan; w += THREADS)
                while (ctr_poll(CTR(a.counters, (size_t)(s - 1) * nfan + w)) < thresh)
                    __builtin_amdgcn_s_sleep(8);
            __syncthreads();
#ifdef LEADER_MAINT
            if (tid == 0 && leader) ctr_acquire();
#else
            if (tid == 0) ctr_acquire();
#endif
            __syncthreads();
        }
        const uint64_t tr = now();
        /* empty body -- this bench is the protocol, not the work */
        __syncthreads();
#ifdef LEADER_MAINT
        if (tid == 0) {
            if (leader) ctr_signal(CTR(a.counters, (size_t)s * nfan + mine));
            else __hip_atomic_fetch_add(CTR(a.counters, (size_t)s * nfan + mine), 1u,
                                        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
#else
        if (tid == 0) ctr_signal(CTR(a.counters, (size_t)s * nfan + mine));
#endif
        __syncthreads();          // mirrors the trace's own barrier before t_end
        const uint64_t te = now();
        if (tid == 0) {
            a.t_arr[(size_t)s * a.ncu + bid] = ta;
            a.t_rdy[(size_t)s * a.ncu + bid] = tr;
            a.t_end[(size_t)s * a.ncu + bid] = te;
        }
    }
    grid.sync();
}

#define CK(x) do{ hipError_t e=(x); if(e){ printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(2);} }while(0)
static int cmp64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b; return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    /* argv[2] overrides the grid so the maintenance cost can be measured against WORKGROUPS PER
     * XCD: 256, 128, 64, 32 blocks is 32, 16, 8, 4 per XCD under the round-robin dispatcher. */
    const int GD = (argc > 2) ? atoi(argv[2]) : p.multiProcessorCount;
    const int NS = (argc > 1) ? atoi(argv[1]) : NSTEP;
    const char* cfg =
#if defined(HIER2)
        "HIER2 (per-XCD leader does the cache maintenance, with rendezvous)";
#elif defined(LEADER_MAINT)
        "LEADER_MAINT (UNSAFE: maintenance on 1 WG/XCD)";
#elif defined(XCCFAN)
        "XCCFAN (8 counters, one per XCD)";
#elif defined(RELAXSIG)
        "RELAXSIG (no buffer_wbl2)";
#elif defined(NOACQ)
        "NOACQ (no buffer_inv)";
#else
        (FAN == 1) ? "BASELINE (1 counter, 256-way)" : "FAN";
#endif
    printf("dev=%s CU=%d THREADS=%d NSTEP=%d FAN=%d cfg=%s\n",
           p.gcnArchName, GD, THREADS, NS, FAN, cfg);

    Args a; a.NS = NS; a.ncu = GD;
    const size_t nfan = FAN, nc = (size_t)NS * nfan * CTR_STRIDE;
    const size_t nrec = (size_t)NS * GD;
    CK(hipMalloc(&a.counters, nc * 4)); CK(hipMemset(a.counters, 0, nc * 4));
    const size_t nl = (size_t)NS * 8 * CTR_STRIDE;
    CK(hipMalloc(&a.lgo, nl * 4)); CK(hipMemset(a.lgo, 0, nl * 4));
    CK(hipMalloc(&a.ldn, nl * 4)); CK(hipMemset(a.ldn, 0, nl * 4));
    CK(hipMalloc(&a.t_arr, nrec * 8)); CK(hipMalloc(&a.t_rdy, nrec * 8));
    CK(hipMalloc(&a.t_end, nrec * 8));

    void* kargs[] = {&a};
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    CK(hipLaunchCooperativeKernel((void*)chain, dim3(GD), dim3(THREADS), kargs, 0, 0));
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1)); CK(hipDeviceSynchronize());

    uint64_t* A = (uint64_t*)malloc(nrec * 8);
    uint64_t* R = (uint64_t*)malloc(nrec * 8);
    uint64_t* E = (uint64_t*)malloc(nrec * 8);
    CK(hipMemcpy(A, a.t_arr, nrec * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(R, a.t_rdy, nrec * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(E, a.t_end, nrec * 8, hipMemcpyDeviceToHost));

    const double TPUS = 100.0;   // s_memrealtime ticks per us
    const int lo = NS / 10, hi = NS - 2, n = hi - lo;
    uint64_t *sig = (uint64_t*)malloc((size_t)n * 8), *gap = (uint64_t*)malloc((size_t)n * 8),
             *esp = (uint64_t*)malloc((size_t)n * 8), *rsp = (uint64_t*)malloc((size_t)n * 8),
             *per = (uint64_t*)malloc((size_t)n * 8);
    for (int s = lo; s < hi; s++) {
        uint64_t r0 = ~0ull, r1 = 0, e0_ = ~0ull, e1_ = 0, a0 = ~0ull;
        uint64_t med = 0;
        for (int b = 0; b < GD; b++) {
            uint64_t r = R[(size_t)s * GD + b], e = E[(size_t)s * GD + b], t = A[(size_t)s * GD + b];
            if (r < r0) r0 = r; if (r > r1) r1 = r;
            if (e < e0_) e0_ = e; if (e > e1_) e1_ = e;
            if (t < a0) a0 = t;
            med += e - r;
        }
        sig[s - lo] = med / (uint64_t)GD;      // mean per-WG t_ready->t_end (the signal cost)
        gap[s - lo] = r1 - a0;                 // packet gate: first arrive -> last ready
        rsp[s - lo] = r1 - r0;                 // t_ready spread over the 256 WGs
        esp[s - lo] = e1_ - e0_;               // t_end   spread
        per[s - lo] = e1_ - a0;                // whole packet
    }
    qsort(sig, n, 8, cmp64); qsort(gap, n, 8, cmp64); qsort(esp, n, 8, cmp64);
    qsort(rsp, n, 8, cmp64); qsort(per, n, 8, cmp64);
    printf("---- n=%d steps, kernel %.3f ms (%.3f us/step) ----\n", n, ms, ms * 1e3 / NS);
    printf(" SIGNAL  per-WG t_ready->t_end  median %.3f us\n", sig[n / 2] / TPUS);
    printf(" GATE    arrive->last ready     median %.3f us\n", gap[n / 2] / TPUS);
    printf(" spread  t_ready over 256 WGs   median %.3f us\n", rsp[n / 2] / TPUS);
    printf(" spread  t_end   over 256 WGs   median %.3f us\n", esp[n / 2] / TPUS);
    printf(" PACKET  first arrive->last end median %.3f us\n", per[n / 2] / TPUS);
    printf("RESULTLINE cfg=[%s] fan=%d signal=%.3f gate=%.3f packet=%.3f step_us=%.3f\n",
           cfg, FAN, sig[n / 2] / TPUS, gap[n / 2] / TPUS, per[n / 2] / TPUS, ms * 1e3 / NS);
    return 0;
}
