// sc1_gate.hip — is there a SOUND cross-CU handoff on gfx950 with NO cache-wide operation?
//
// THE CLAIM UNDER TEST. `runtime/amd/interp.hip`'s ctr_poll comment states: "There is no HIP scope
// between workgroup (no cross-CU) and agent (full L1+L2 inv), so agent is the cheapest sound
// choice." That is true of FENCES and false of ACCESSES. On gfx940+ every vector memory op carries
// SC0/SC1 scope bits, and this toolchain (ROCm 7.2.4 hipcc, gfx950) lowers
//
//     __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)  ->  global_store_dword ... sc1
//     __hip_atomic_load (p,    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)  ->  global_load_dword  ... sc1
//
// with NO buffer_wbl2 and NO buffer_inv. The cache-wide ops appear only when the scope is carried
// by a FENCE, because a fence has to cover the PLAIN accesses around it. So a protocol that marks
// the guarded DATA device-scope per-instruction should need no cache maintenance at all:
//
//   producer   data stores  RELAXED/AGENT (sc1);  s_waitcnt vmcnt(0);  counter bump RELAXED
//   consumer   counter poll RELAXED/AGENT (sc1);  NO acquire fence;    data loads RELAXED/AGENT (sc1)
//
// If that holds it converts plow's PLOW_GATE_NOINV + PLOW_GATE_RELAXSIG ceiling (measured -13.7%
// of the token, but a deliberate data race) into something shippable, WITHOUT the sentinel scheme's
// three blockers: no reserved NaN, no range poll, no static knowledge of the output range.
//
// WHY THE THIRD ARM IS THE WHOLE EXPERIMENT. `interp.hip` records that a workgroup-scope acquire
// fires stale reads at a 100% rate under a resident-line stress harness. Arm UNSAFE reproduces
// exactly that -- workgroup fences with PLAIN data accesses -- so if UNSAFE shows stale reads and
// SC1 shows none, the difference is the sc1 bits and not a harness that cannot see staleness.
// A run where UNSAFE is also clean proves NOTHING and must be reported as inconclusive.
//
// ARMS (-D, default BASELINE):
//   (default)   plow today: plain data, release RMW (buffer_wbl2) + agent acquire (buffer_inv).
//   -DSC1       device-scope data accesses, workgroup fences, relaxed counter. No cache-wide op.
//   -DUNSAFE    workgroup fences with PLAIN data accesses. The known race; the harness control.
//   -DSC1WIDE   like SC1 but the payload moves in 128-bit global_{load,store}_dwordx4 ... sc1 via
//               inline asm, which is what a real implementation would use for an activation slice.
//
// STALENESS TEST. Every step each workgroup stamps its own PAYLOAD slice with the step number,
// signals, waits for all 256, then reads the slice of the workgroup 128 away (guaranteed a
// DIFFERENT XCD under the round-robin gfx950 dispatcher) and counts any word not stamped with the
// current step. Slices are 4 KB apart so lines stay resident, which is the condition under which
// the race actually fires.
//
// BUILD/RUN (pin an IDLE GPU -- this box is shared; take flock /tmp/plow_gpu.lock):
//   hipcc --offload-arch=gfx950 -O3 -std=c++17 -w sc1_gate.hip -o sc1base
//   hipcc --offload-arch=gfx950 -O3 -std=c++17 -w -DSC1 sc1_gate.hip -o sc1sc1
//   sg render -c 'LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=5 ./sc1sc1 400 256 3072'
//   argv: [steps] [blocks] [payload words per workgroup]
//
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
namespace cg = cooperative_groups;

#define THREADS 512
#define CTR_STRIDE 32u            // PLOW_CTR_STRIDE: 128 B per counter, one cache line
#define SLICE_WORDS 1024u         // 4 KB apart, as ctr_convergence's counters are 128 B apart
#define CTR(base, id) ((base) + (size_t)(id) * CTR_STRIDE)

/* ---- the four protocols, as thin wrappers so the loop below is identical in every arm -------- */
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);  // global_load sc1
}
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
#if defined(SC1) || defined(SC1WIDE) || defined(UNSAFE)
    /* vmcnt(0) so this workgroup's payload stores have retired past L2 before the bump is visible.
     * This is the ONLY ordering the producer needs once the stores are sc1 -- there is nothing
     * dirty left in a cache below the device coherence point to write back. */
    __builtin_amdgcn_s_waitcnt(0x0070 /* vmcnt(0) */);
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);   // + buffer_wbl2
#endif
}
__device__ __forceinline__ void gate_acquire() {
#if defined(SC1) || defined(SC1WIDE) || defined(UNSAFE)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");                      // no buffer_inv
#else
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");                          // buffer_inv sc1
#endif
}
__device__ __forceinline__ void pay_store(uint32_t* p, uint32_t v) {
#if defined(SC1)
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);       // global_store sc1
#else
    *p = v;                                                                     // plain
#endif
}
__device__ __forceinline__ uint32_t pay_load(const uint32_t* p) {
#if defined(SC1)
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);    // global_load sc1
#else
    return *p;
#endif
}
/* 128-bit device-scope payload moves. `__hip_atomic_*` refuses 16-byte atomics, so the sc1 bit goes
 * on via inline asm -- which is what a production activation path would do anyway. */
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void pay_store4(uint32_t* p, uint32_t v) {
    u32x4 x = { v, v, v, v };
    asm volatile("global_store_dwordx4 %0, %1, off sc1" :: "v"(p), "v"(x) : "memory");
}
__device__ __forceinline__ u32x4 pay_load4(const uint32_t* p) {
    u32x4 r;
    /* The `s_waitcnt vmcnt(0)` MUST be inside the asm block. Without it the first run of this arm
     * reported 49.4% stale reads -- not a hardware result: the compiler does not know an asm-defined
     * VGPR is still in flight, so the comparison below read the register before the load landed.
     * The dword SC1 arm never had this problem because the compiler places its own waitcnts. */
    asm volatile("global_load_dwordx4 %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                 : "=v"(r) : "v"(p) : "memory");
    return r;
}

__device__ __forceinline__ uint64_t now() { return __builtin_amdgcn_s_memrealtime(); }

struct Args {
    uint32_t* counters;
    uint32_t* pay;      // [ncu][SLICE_WORDS]  -- only the first `pw` words are used
    uint32_t* stale;    // [ncu] stale-word count
    uint64_t* t_arr;    // [NS][ncu]
    uint64_t* t_rdy;
    uint64_t* t_end;
    int NS, ncu; uint32_t pw;
};

__global__ void __launch_bounds__(THREADS) chain(Args a) {
    __shared__ char pad[90 * 1024];   // 1 block/CU => 256 blocks light all 8 XCDs
    ((volatile char*)pad)[threadIdx.x & 1023] = 0;
    const int bid = blockIdx.x, tid = threadIdx.x;
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    const int peer = (bid + a.ncu / 2) % a.ncu;   // 128 away => a different XCD
    uint32_t bad = 0;

    for (int s = 0; s < a.NS; s++) {
        const uint64_t ta = now();
        if (s > 0) {
            if (tid == 0)
                while (ctr_poll(CTR(a.counters, (size_t)(s - 1))) < (unsigned)a.ncu)
                    __builtin_amdgcn_s_sleep(8);
            __syncthreads();
            if (tid == 0) gate_acquire();
            __syncthreads();
            /* OBSERVE: read the PEER's slice. It must carry AT LEAST step s-1's stamp (== s).
             * A LARGER stamp is not an error: nothing stops the peer from clearing this same gate
             * and writing step s's stamp while we are still reading, so only `value < want` is a
             * stale read. Counting `!=` would report legitimate run-ahead as a failure. */
#if defined(SC1WIDE)
            for (uint32_t i = tid * 4; i < a.pw; i += THREADS * 4) {
                u32x4 v = pay_load4(&a.pay[(size_t)peer * SLICE_WORDS + i]);
                uint32_t want = (uint32_t)s;   // stamp written at step s-1 is s-1+1 == s
                bad += (v[0] < want) + (v[1] < want) + (v[2] < want) + (v[3] < want);
            }
#else
            for (uint32_t i = tid; i < a.pw; i += THREADS)
                if (pay_load(&a.pay[(size_t)peer * SLICE_WORDS + i]) < (uint32_t)s) bad++;
#endif
        }
        const uint64_t tr = now();
        /* PUBLISH: stamp my own slice with s+1, then signal. */
#if defined(SC1WIDE)
        for (uint32_t i = tid * 4; i < a.pw; i += THREADS * 4)
            pay_store4(&a.pay[(size_t)bid * SLICE_WORDS + i], (uint32_t)s + 1u);
#else
        for (uint32_t i = tid; i < a.pw; i += THREADS)
            pay_store(&a.pay[(size_t)bid * SLICE_WORDS + i], (uint32_t)s + 1u);
#endif
        __syncthreads();
        if (tid == 0) ctr_signal(CTR(a.counters, (size_t)s));
        __syncthreads();
        const uint64_t te = now();
        if (tid == 0) {
            a.t_arr[(size_t)s * a.ncu + bid] = ta;
            a.t_rdy[(size_t)s * a.ncu + bid] = tr;
            a.t_end[(size_t)s * a.ncu + bid] = te;
        }
    }
    atomicAdd(&a.stale[bid], bad);
    grid.sync();
}

#define CK(x) do{ hipError_t e=(x); if(e){ printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(2);} }while(0)
static int cmp64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b; return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    const int NS = (argc > 1) ? atoi(argv[1]) : 400;
    const int GD = (argc > 2) ? atoi(argv[2]) : p.multiProcessorCount;
    uint32_t pw = (argc > 3) ? (uint32_t)atoi(argv[3]) : 0u;   // payload words per workgroup
    if (pw > SLICE_WORDS) pw = SLICE_WORDS;
#if defined(SC1WIDE)
    pw &= ~3u;
#endif
    const char* cfg =
#if defined(SC1WIDE)
        "SC1WIDE (dwordx4 sc1 payload, wg fences, relaxed ctr)";
#elif defined(SC1)
        "SC1 (device-scope payload accesses, wg fences, relaxed ctr)";
#elif defined(UNSAFE)
        "UNSAFE (wg fences, PLAIN payload) -- harness control, must show stale reads";
#else
        "BASELINE (plain payload, release RMW + agent acquire)";
#endif
    printf("dev=%s CU=%d THREADS=%d NSTEP=%d payload=%u words (%u B)/WG cfg=%s\n",
           p.gcnArchName, GD, THREADS, NS, pw, pw * 4u, cfg);

    Args a; a.NS = NS; a.ncu = GD; a.pw = pw;
    const size_t nrec = (size_t)NS * GD;
    CK(hipMalloc(&a.counters, (size_t)NS * CTR_STRIDE * 4)); CK(hipMemset(a.counters, 0, (size_t)NS * CTR_STRIDE * 4));
    CK(hipMalloc(&a.pay, (size_t)GD * SLICE_WORDS * 4));     CK(hipMemset(a.pay, 0, (size_t)GD * SLICE_WORDS * 4));
    CK(hipMalloc(&a.stale, (size_t)GD * 4));                 CK(hipMemset(a.stale, 0, (size_t)GD * 4));
    CK(hipMalloc(&a.t_arr, nrec * 8)); CK(hipMalloc(&a.t_rdy, nrec * 8)); CK(hipMalloc(&a.t_end, nrec * 8));

    void* kargs[] = {&a};
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    CK(hipLaunchCooperativeKernel((void*)chain, dim3(GD), dim3(THREADS), kargs, 0, 0));
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1)); CK(hipDeviceSynchronize());

    uint64_t* A = (uint64_t*)malloc(nrec * 8); uint64_t* R = (uint64_t*)malloc(nrec * 8);
    uint64_t* E = (uint64_t*)malloc(nrec * 8); uint32_t* S = (uint32_t*)malloc((size_t)GD * 4);
    CK(hipMemcpy(A, a.t_arr, nrec * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(R, a.t_rdy, nrec * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(E, a.t_end, nrec * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(S, a.stale, (size_t)GD * 4, hipMemcpyDeviceToHost));

    const double TPUS = 100.0;   // s_memrealtime ticks per us
    const int lo = NS / 10, hi = NS - 2, n = hi - lo;
    uint64_t *sig = (uint64_t*)malloc((size_t)n * 8), *per = (uint64_t*)malloc((size_t)n * 8);
    for (int s = lo; s < hi; s++) {
        uint64_t e1_ = 0, a0 = ~0ull, med = 0;
        for (int b = 0; b < GD; b++) {
            uint64_t r = R[(size_t)s * GD + b], e = E[(size_t)s * GD + b], t = A[(size_t)s * GD + b];
            if (e > e1_) e1_ = e; if (t < a0) a0 = t; med += e - r;
        }
        sig[s - lo] = med / (uint64_t)GD; per[s - lo] = e1_ - a0;
    }
    qsort(sig, n, 8, cmp64); qsort(per, n, 8, cmp64);
    uint64_t stale = 0; int wgs = 0;
    for (int b = 0; b < GD; b++) { stale += S[b]; if (S[b]) wgs++; }
    const double reads = (double)(NS - 1) * GD * pw;
    printf(" SIGNAL  per-WG t_ready->t_end  median %.3f us\n", sig[n / 2] / TPUS);
    printf(" PACKET  first arrive->last end median %.3f us\n", per[n / 2] / TPUS);
    printf(" STALE   %llu words of %.0f read (%.4f%%), in %d/%d workgroups\n",
           (unsigned long long)stale, reads, reads ? 100.0 * (double)stale / reads : 0.0, wgs, GD);
    printf("RESULTLINE cfg=[%s] pw=%u signal=%.3f packet=%.3f step_us=%.3f stale=%llu stale_pct=%.4f\n",
           cfg, pw, sig[n / 2] / TPUS, per[n / 2] / TPUS, ms * 1e3 / NS,
           (unsigned long long)stale, reads ? 100.0 * (double)stale / reads : 0.0);
    return 0;
}
