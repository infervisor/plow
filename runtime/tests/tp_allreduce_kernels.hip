/* tp_allreduce_kernels.hip — device code for the one-shot all-reduce microbench.
 *
 * Validates op_collective.h's d_xreduce_oneshot END-TO-END across two real GPUs:
 * the publish-signal, the SYSTEM-scope xctr gate, and the f32-accumulate reduction,
 * exactly as the persistent megakernel's XREDUCE packet will run it. This is the
 * standalone correctness+latency proof for the runtime collective (P1-B), reusing
 * the transport primitives tp-transport already proved.
 *
 * The kernels call op_collective.h directly, so the bench validates THE op code, not
 * a copy — the same rule every op_*.h golden test follows.
 */
#include "../amd/op_collective.h"

/* Fill this rank's partial slot with a checkable element-varying bf16 pattern.
 * val(r,e) = (r+1) * (1 + (e&7)*0.125) — all exact in bf16, and it varies per
 * element so a mis-indexed peer read or a partial write is caught, not just a
 * uniform 1+2=3. */
extern "C" __global__ void tp_fill_partial(bf16* part, uint32_t n, uint32_t rank) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += gridDim.x * blockDim.x)
        part[e] = f2bf((float)(rank + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}

/* One-shot all-reduce, timed. Loops `iters` collectives so the per-collective
 * latency is (cycles / iters); iteration i uses gate_target = i*nranks because each
 * rank's workgroup-0 bumps every peer gate once per iteration (nranks arrivals/iter).
 * All workgroups wait + reduce; only workgroup 0 signals. */
extern "C" __global__ void tp_allreduce(bf16* out, const void* const* peer_scratch,
                                        uint32_t* const* peer_gate, uint32_t nranks,
                                        uint32_t rank, uint32_t n, uint32_t slot_bytes,
                                        uint32_t iters, uint64_t deadline,
                                        uint64_t* cycles, uint32_t* status) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        d_xreduce_oneshot(out, peer_scratch, peer_gate, nranks, rank, n, slot_bytes,
                          i * nranks, blockIdx.x == 0, deadline, status,
                          blockIdx.x, gridDim.x);
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}

/* Production prefill path: reduce-scatter + all-gather, including both SYSTEM-scope
 * rendezvous. The grid is supplied by the host so the benchmark can match an emitted
 * packet exactly. Iterations use disjoint counter ids; callers refill before a parity run. */
extern "C" __global__ void tp_allreduce_twoshot(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, uint64_t xctr_byte_off, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 0; i < iters; i++)
        d_xreduce_twoshot_mega(out, peer_scratch, nranks, rank, n, slot_bytes,
                               (size_t)xctr_byte_off, 2u * i, 2u * i + 1u,
                               deadline, status, blockIdx.x, gridDim.x);
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}

extern "C" __global__ void tp_allreduce_twoshot_gather(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, uint64_t xctr_byte_off, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status,
    uint32_t gslot_bytes, uint32_t gcols) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 0; i < iters; i++)
        d_xreduce_twoshot_mega(out, peer_scratch, nranks, rank, n, slot_bytes,
                               (size_t)xctr_byte_off, 2u * i, 2u * i + 1u,
                               deadline, status, blockIdx.x, gridDim.x,
                               nullptr, nullptr, gslot_bytes, gcols);
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}

extern "C" __global__ void tp_allreduce_oneshot_gather(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, uint64_t xctr_byte_off, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status,
    uint32_t gslot_bytes, uint32_t gcols) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 0; i < iters; i++)
        d_xreduce_mega(out, peer_scratch, nranks, rank, n, slot_bytes,
                       (size_t)xctr_byte_off, i, deadline, status,
                       blockIdx.x, gridDim.x, gslot_bytes, gcols, nranks * gcols);
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}

/* ---- DECODE-LATENCY ARMS (tp_allreduce_bench TP_MODE=cold|tagged|tagged_cold) --------
 *
 * The hot loop above never rewrites a partial, so it prices the protocol with clean caches
 * and unchanged peer lines. A decode collective follows a producer that has just written
 * the partial on every XCD, and the whole point of the tagged form is the arrival path, so
 * the cold arms re-run a PRODUCER each iteration: iteration-varying values, plain stores
 * (as the GEMV epilogue writes them), then the interpreter's local agent-scope
 * release/poll/acquire gate before the collective. Partials are double-buffered by
 * iteration parity exactly as the emitter's partial_A/partial_B by layer parity: a rank can
 * only rewrite a slot after every peer has finished reading it.
 *
 * `order` selects the strict-order oracle values [2^24, 1, -2^24, 0...]: their exact f32
 * rank-0..N-1 sum is 0 and any other order is 1 or -1. */
__device__ __forceinline__ bf16 tp_val(uint32_t rank, uint32_t e, uint32_t iter, uint32_t order) {
    if (order)
        return f2bf(rank == 0u ? 16777216.0f : rank == 1u ? 1.0f : rank == 2u ? -16777216.0f
                                                                                 : 0.0f);
    return f2bf((float)(rank + 1) * (1.0f + (float)((e + iter) & 7u) * 0.125f));
}
extern "C" __global__ void tp_fill_partial_v(bf16* part, uint32_t n, uint32_t rank,
                                             uint32_t iter, uint32_t order) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
        part[e] = tp_val(rank, e, iter, order);
}
__device__ __forceinline__ void tp_local_produce(bf16* part, uint32_t n, uint32_t rank,
                                                 uint32_t iter, uint32_t order,
                                                 uint32_t* lctr) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
        st_act1(&as_glob(part)[e], tp_val(rank, e, iter, order));
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(lctr, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(lctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
               iter * gridDim.x)
            __builtin_amdgcn_s_sleep(8);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}
extern "C" __global__ void tp_allreduce_cold(bf16* out, const void* const* peer_scratch,
                                             uint32_t* const* peer_gate, uint32_t nranks,
                                             uint32_t rank, uint32_t n, uint32_t slot_bytes,
                                             uint32_t iters, uint64_t deadline,
                                             uint64_t* cycles, uint32_t* status,
                                             uint32_t* lctr, uint32_t order) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        const uint32_t sb = slot_bytes * (i & 1u);
        tp_local_produce((bf16*)((char*)peer_scratch[rank] + sb), n, rank, i, order, lctr);
        d_xreduce_oneshot(out, peer_scratch, peer_gate, nranks, rank, n, sb, i * nranks,
                          blockIdx.x == 0, deadline, status, blockIdx.x, gridDim.x);
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}
/* Tagged one-shot. Iteration i is the tag; the tagged slot alternates by parity. */
extern "C" __global__ void tp_allreduce_tagged(bf16* out, const void* const* peer_scratch,
                                               uint32_t nranks, uint32_t rank, uint32_t n,
                                               uint32_t slot_bytes, uint32_t tslot_bytes,
                                               uint32_t iters, uint64_t deadline,
                                               uint64_t* cycles, uint32_t* status,
                                               uint32_t* lctr, uint32_t order, uint32_t cold,
                                               uint32_t tstride) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        const uint32_t sb = slot_bytes * (i & 1u);
        if (cold)
            tp_local_produce((bf16*)((char*)peer_scratch[rank] + sb), n, rank, i, order, lctr);
        d_xreduce_tagged_mega(out, peer_scratch, nranks, rank, n, cold ? sb : 0u,
                              tslot_bytes + tstride * (i & 1u), i, deadline, status,
                              blockIdx.x, gridDim.x);
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}

/* ---- FABRIC LATENCY PROBE (TP_PROBE=1): one lane, dependent chains, ticks per hop ----
 *   res[0] remote system-scope 8 B load, pointer chase over a zeroed 32 KiB window
 *   res[1] remote returning relaxed system-scope atomic, value-dependent chain
 *   res[2] remote release fetch_add (non-returning) + s_waitcnt vmcnt(0): the signal loop's
 *          element, clean L2
 *   res[3] the same after every thread dirtied 128 B of local L2 (64 KiB per hop)
 *   res[4] local system-scope 4 B poll load chain (xctr_poll on the own counter)
 *   res[5] res[4] plus one system acquire fence (buffer_inv sc0 sc1) per hop */
extern "C" __global__ void tp_peer_probe(const void* const* peer_scratch,
                                         uint32_t* const* peer_gate, uint32_t rank,
                                         uint32_t peer, uint32_t hops, uint64_t* res,
                                         uint32_t probe_off, bf16* dirty) {
    if (blockIdx.x != 0) return;
    const uint64_t* win = (const uint64_t*)((const char*)peer_scratch[peer] + probe_off);
    uint32_t* rg = peer_gate[peer] + 16;
    uint32_t* own = peer_gate[rank] + 24;
    uint64_t acc = 0;
    uint64_t t0, t1;
    if (threadIdx.x == 0) {
        uint32_t idx = 0;
        t0 = __builtin_amdgcn_s_memrealtime();
        for (uint32_t h = 0; h < hops; h++) {
            const uint64_t v = xrt_load(win + idx);
            idx = (idx + (uint32_t)v + 16u) & 4095u;
        }
        t1 = __builtin_amdgcn_s_memrealtime();
        res[0] = t1 - t0;
        acc += idx;
        /* `a >> 31` is 0 for any count this probe can reach but not foldable, so each
         * atomic's operand depends on the previous one's return: a true chain. */
        uint32_t a = 0;
        t0 = __builtin_amdgcn_s_memrealtime();
        for (uint32_t h = 0; h < hops; h++)
            a = __hip_atomic_fetch_add(rg, a >> 31, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        t1 = __builtin_amdgcn_s_memrealtime();
        res[1] = t1 - t0;
        acc += a;
        t0 = __builtin_amdgcn_s_memrealtime();
        for (uint32_t h = 0; h < hops; h++) {
            __hip_atomic_fetch_add(rg, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        t1 = __builtin_amdgcn_s_memrealtime();
        res[2] = t1 - t0;
    }
    uint64_t sum = 0;
    for (uint32_t h = 0; h < hops; h++) {
        __syncthreads();
        for (uint32_t k = 0; k < 64u; k++)
            st_act1(&as_glob(dirty)[(threadIdx.x * 64u + k) & 0xffffu], (bf16)(h + k));
        __syncthreads();
        if (threadIdx.x == 0) {
            t0 = __builtin_amdgcn_s_memrealtime();
            __hip_atomic_fetch_add(rg, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            sum += __builtin_amdgcn_s_memrealtime() - t0;
        }
    }
    if (threadIdx.x == 0) {
        res[3] = sum;
        uint32_t v = 0;
        t0 = __builtin_amdgcn_s_memrealtime();
        for (uint32_t h = 0; h < hops; h++) v = xctr_poll(own + (v >> 31));
        t1 = __builtin_amdgcn_s_memrealtime();
        res[4] = t1 - t0;
        acc += v;
        t0 = __builtin_amdgcn_s_memrealtime();
        for (uint32_t h = 0; h < hops; h++) {
            v = xctr_poll(own + (v >> 31));
            xctr_acquire();
        }
        t1 = __builtin_amdgcn_s_memrealtime();
        res[5] = t1 - t0;
        res[6] = acc + v;
    }
}
