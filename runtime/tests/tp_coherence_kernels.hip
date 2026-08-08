/* tp_coherence_kernels.hip — ADVERSARIAL cross-GPU (XGMI) coherence stress test.
 *
 * Purpose: decide whether the FLAT one-shot all-reduce's cross-GPU handshake
 * (op_collective.h: xctr_signal RELEASE-SYSTEM / xctr_poll RELAXED-SYSTEM /
 * xctr_acquire = fence(ACQUIRE, "")) is a REAL happens-before on gfx950 XGMI, or
 * whether the full-model "bit-exact TP" passes only by hardware accident (the
 * weight stream evicting stale peer-VRAM lines before they are re-read).
 *
 * This is the cross-GPU / SYSTEM-scope analogue of the cross-XCD / AGENT-scope
 * CTR_FENCE stress test. Same discipline:
 *   - TINY resident peer buffer (no weight stream to evict stale lines).
 *   - strict prime-then-overwrite LOCKSTEP ping-pong (each iter's payload is a
 *     DISTINCT value; the consumer MUST observe THIS iter's value, and its own L2
 *     copy of the previous iter's value is the stale trap).
 *   - a POSITIVE CONTROL (a deliberately-unsynchronized variant the test must be
 *     shown to catch) so a 0-stale result on the shipped path is meaningful.
 *
 * The producer/consumer reproduce the EXACT lowering the interpreter emits for a
 * coarse XReduce (d_xreduce_mega): the producer writes its partial into its own
 * peer scratch (plain global stores), then system-releases and bumps the peer's
 * gate; the consumer polls its local gate, takes ONE system acquire, then reads
 * the producer's partial over XGMI. Only the RELEASE scope (producer) and the
 * ACQUIRE scope (consumer) vary between variants — the handshake counter is always
 * system-scope so the ping-pong always progresses.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

/* Scope selectors (must match the host enum). */
#define SC_SYS     0  /* system scope  — the SHIPPED xctr path                    */
#define SC_AGENT   1  /* agent/device scope — reaches L2, not XGMI                 */
#define SC_RELAXED 2  /* relaxed / none — the positive control                     */

/* The distinct per-iteration payload. Every word of the partial is set to this;
 * neighbouring iterations differ in every bit-ish, so a stale (previous-iter) or
 * torn read is caught. */
__device__ __forceinline__ uint32_t payload(uint32_t i) {
    return i * 2654435761u /* 0x9E3779B1 */ + 0xABCD1234u;
}

/* ---- producer (GPU A) ----------------------------------------------------
 * data    : this rank's partial, in A's OWN peer VRAM (consumer reads over XGMI)
 * g_ready : the consumer's gate word (peer VRAM on B) — producer bumps it
 * g_done  : this rank's gate word (peer VRAM on A) — consumer bumps it
 * Strict lockstep: overwrite iter i only after the consumer acknowledged i-1.
 * rel_scope selects the RELEASE strength paired with the ready-bump. */
extern "C" __global__ void coh_producer(
    uint32_t* data, uint32_t ndata, uint32_t* g_ready, uint32_t* g_done,
    uint32_t iters, int rel_scope, uint64_t deadline, uint32_t* status) {
    if (threadIdx.x || blockIdx.x) return;   /* single lane — the do_signal thread */
    for (uint32_t i = 1; i <= iters; i++) {
        /* wait for the consumer to finish the PREVIOUS iteration (i-1). */
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (__hip_atomic_load(g_done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < i - 1) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) { *status = 0xDEAD0000u | i; return; }
        }
        /* write THIS iteration's partial (plain global stores, as the GEMV epilogue does). */
        for (uint32_t j = 0; j < ndata; j++) data[j] = payload(i);

        /* publish + signal the consumer's gate. The RELEASE scope is the variable
         * under test; RELAXED omits the release ordering entirely (control). */
        if (rel_scope == SC_SYS)
            __hip_atomic_fetch_add(g_ready, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        else if (rel_scope == SC_AGENT)
            __hip_atomic_fetch_add(g_ready, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        else /* SC_RELAXED */
            __hip_atomic_fetch_add(g_ready, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    *status = 0;
}

/* ---- consumer (GPU B) ----------------------------------------------------
 * data     : the producer's partial, peer VRAM on A (read over XGMI)
 * g_ready  : this rank's gate word (peer VRAM on B) — producer bumped it
 * g_done   : the producer's gate word (peer VRAM on A) — consumer bumps it
 * acq_scope selects the ACQUIRE fence strength taken once the gate clears.
 * stale    : incremented whenever the observed partial != this iter's payload. */
extern "C" __global__ void coh_consumer(
    const uint32_t* data, uint32_t ndata, uint32_t* g_ready, uint32_t* g_done,
    uint32_t iters, int acq_scope, uint64_t deadline, uint32_t* stale,
    uint32_t* first_bad, uint32_t* status) {
    if (threadIdx.x || blockIdx.x) return;   /* single lane — matches the poll/acquire thread */
    uint32_t nstale = 0, fb = 0xffffffffu;
    for (uint32_t i = 1; i <= iters; i++) {
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (__hip_atomic_load(g_ready, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < i) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) { *status = 0xDEAD0000u | i; *stale = nstale; return; }
        }
        /* ONE acquire once the gate clears — exactly xctr_acquire()'s placement. */
        if (acq_scope == SC_SYS)        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        else if (acq_scope == SC_AGENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        /* SC_RELAXED: no acquire at all (control). */

        /* read the partial over XGMI (plain global loads, as d_xreduce does). */
        const uint32_t want = payload(i);
        uint32_t bad = 0;
        for (uint32_t j = 0; j < ndata; j++)
            if (data[j] != want) { bad = 1; if (fb == 0xffffffffu) fb = data[j]; }
        if (bad) nstale++;

        /* acknowledge iter i to the producer (always system-release so it progresses). */
        __hip_atomic_fetch_add(g_done, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    *stale = nstale;
    if (fb != 0xffffffffu) *first_bad = fb;
    *status = 0;
}
