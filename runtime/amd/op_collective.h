/* op_collective.h — the cross-GPU (tensor-parallel) collectives.
 *
 * These are the ONLY device ops that touch PEER VRAM. They ride the transport the
 * `tp-transport` agent proved (plans/tp-transport.md): a coarse-grained VRAM buffer
 * made visible to every GPU with hsa_amd_agents_allow_access, synchronized by a
 * SYSTEM-scope atomic on peer memory (no HSA signal, no host, ~90 ns one-way).
 *
 * The single-GPU counter-gate in interp.hip uses __HIP_MEMORY_SCOPE_AGENT (orders
 * across XCDs within ONE GPU). The ONLY change for cross-GPU is widening that scope
 * to SYSTEM so the release/acquire reaches past this device's per-XCD L2 out onto
 * XGMI to the peer — exactly the xctr_* trio below (plans/tp-design.md §6a, §12).
 *
 * SHARED with the golden wrapper (test_kernels.hip) and the 2-GPU microbench
 * (tp_allreduce_kernels.hip), so the CPU/2-GPU reference validates THIS code, not a
 * copy of it — the same discipline as every other op_*.h.
 */
#ifndef PLOW_OP_COLLECTIVE_H
#define PLOW_OP_COLLECTIVE_H

#include "amd_common.h" /* bf16, bf2f/f2bf, PLOW_THREADS */

/* ---- cross-GPU counter helpers (SYSTEM scope) ---------------------------------
 * Mirror interp.hip's agent-scope ctr_poll/ctr_acquire/ctr_signal, widened to
 * __HIP_MEMORY_SCOPE_SYSTEM. Same discipline (plans/tp-design.md §12):
 *   - POLL relaxed (a system ACQUIRE load emits a full inv on EVERY iteration —
 *     never put it in the spin),
 *   - take exactly ONE system acquire once the gate clears,
 *   - RELEASE on the signal RMW.
 * `flag` is a word in a plow_hsa_alloc_peer buffer; a peer's release store reaches
 * this rank's poll over XGMI. */
__device__ __forceinline__ uint32_t xctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void xctr_acquire(void) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  /* "" = system scope on AMDGPU */
}
__device__ __forceinline__ void xctr_signal(uint32_t* p) {
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* ---- XREDUCE: the reduce half of the one-shot all-reduce ------------------------
 * The N partials are already published — each rank's producing GEMV (o_proj/down)
 * wrote its partial H-vector straight into its own peer_scratch slot (fused into the
 * GEMV epilogue, no extra copy, plans/tp-design.md §8a) — and the SE_XCTR gate has
 * already fired, so this is a pure local reduction over the N peer slots.
 *
 *   out          : local full H-vector result (bf16)
 *   peer_scratch : [nranks] each rank's peer-mapped reduction-region base
 *   slot_bytes   : byte offset of THIS collective's partial within a region
 *                  (partial_A / partial_B, chosen by layer parity)
 *   n, nranks    : elements to reduce, TP degree
 *   slice, nblk  : the interpreter's work-share (this workgroup, of nblk)
 *
 * f32 accumulate, round to bf16 — the same reduction math as the CPU rdma.c oracle
 * and the transport tp_reduce_oneshot building block. */
__device__ __forceinline__ void d_xreduce(bf16* out, const void* const* peer_scratch,
                                          uint32_t slot_bytes, uint32_t nranks, uint32_t n,
                                          unsigned slice, unsigned nblk) {
    const unsigned base = slice * PLOW_THREADS + threadIdx.x;
    const unsigned step = nblk * PLOW_THREADS;
    for (uint32_t e = base; e < n; e += step) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nranks; r++) {
            const bf16* part = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
            acc += bf2f(as_glob(part)[e]);
        }
        as_glob(out)[e] = f2bf(acc);
    }
}

/* ---- ONE-SHOT all-reduce (publish-signal + gate + reduce), fused, no launch -----
 * The whole collective in one device call, matching plans/tp-transport.md §5's
 * tp_allreduce_oneshot signature. Used by the 2-GPU microbench and the golden
 * reference. In the persistent megakernel the WAIT is instead handled by the
 * interpreter's SE_XCTR gate (system-scope) and the body is just d_xreduce — this
 * self-contained form is what the standalone harness validates bit-exact.
 *
 *   peer_gate[r] : rank r's "arrivals" counter (system-scope, peer-mapped word).
 *                  peer_gate[rank] is this rank's LOCAL gate (peers write it).
 *   gate_target  : arrivals to wait for (nranks for a single collective; the bench
 *                  scales it per iteration to reuse the counters in a timing loop).
 *   do_signal    : only ONE workgroup announces this rank's arrival (else the gate
 *                  would need threshold nranks*nblk). All workgroups wait + reduce.
 *   deadline_ticks : s_memrealtime budget; if the fabric does not propagate the
 *                    system atomic the kernel bails (status |= 0xDEAD0000) instead of
 *                    hanging the queue. status may be NULL. */
__device__ __forceinline__ void d_xreduce_oneshot(
    bf16* out, const void* const* peer_scratch, uint32_t* const* peer_gate,
    uint32_t nranks, uint32_t rank, uint32_t n, uint32_t slot_bytes,
    uint32_t gate_target, bool do_signal, uint64_t deadline_ticks, uint32_t* status,
    unsigned slice, unsigned nblk) {
    __shared__ int bailed;
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();

    if (do_signal && threadIdx.x == 0)
        for (uint32_t r = 0; r < nranks; r++) xctr_signal(peer_gate[r]);

    if (threadIdx.x == 0) {
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(peer_gate[rank]) < gate_target) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1;
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!bailed) xctr_acquire(); /* ONE system acquire, only after the gate clears */
    }
    __syncthreads();
    if (bailed) return;

    d_xreduce(out, peer_scratch, slot_bytes, nranks, n, slice, nblk);
}

/* ---- XREDUCE for the persistent megakernel -------------------------------------
 * The interpreter's form of the one-shot all-reduce. Same three steps as
 * d_xreduce_oneshot, but the peer gate words are addressed straight out of the
 * PlowProgram peer-scratch table instead of a precomputed pointer array (the
 * megakernel cannot build one per packet). Each rank's xctr region sits at the SAME
 * byte offset inside its peer_scratch (plans/tp-design.md §7a), so:
 *     peer r's gate = PLOW_CTR(peer_scratch[r] + xctr_byte_off, gate_id)
 * where xctr_byte_off = (char*)prog.xctr - (char*)prog.peer_scratch[rank] — computed
 * by the caller from the two §12 fields, no 5th field needed.
 *
 * COARSE (1 sync/collective, plans/tp-design.md §6b): gate_target = nranks, the host
 * having zeroed xctr before the token. The producing o_proj/down GEMV has already
 * written this rank's partial into peer_scratch[rank] (tp-host binds og/dg there), and
 * the interpreter's ordinary agent-scope gate on that GEMV is this op's LOCAL wait —
 * so by the time we get here this rank's partial is visible; we then cross-signal,
 * wait N arrivals, and reduce. This is exactly the path the 2-GPU microbench proved. */
__device__ __forceinline__ void d_xreduce_mega(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, size_t xctr_byte_off, uint32_t gate_id,
    uint64_t deadline_ticks, uint32_t* status, unsigned slice, unsigned nblk) {
    __shared__ int bailed;
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();

    /* ONE workgroup announces this rank's arrival to every peer (else the threshold
     * would have to be nranks*nblk). */
    if (slice == 0 && threadIdx.x == 0) {
        for (uint32_t r = 0; r < nranks; r++) {
            uint32_t* base = (uint32_t*)((char*)peer_scratch[r] + xctr_byte_off);
            xctr_signal(PLOW_CTR(base, gate_id));
        }
    }
    if (threadIdx.x == 0) {
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_id);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1;
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!bailed) xctr_acquire();
    }
    __syncthreads();
    if (bailed) return;

    d_xreduce(out, peer_scratch, slot_bytes, nranks, n, slice, nblk);
}

/* ---- TWO-SHOT all-reduce for the LARGE prefill [T,hidden] message ----------------
 * The one-shot d_xreduce_mega has EVERY rank read ALL N peers' FULL partial: ~(N-1)*msg
 * of fabric traffic per rank. That is optimal for decode's tiny [1,hidden] latency-bound
 * message, but the prefill partial is [T,hidden] — T* bigger, BANDWIDTH-bound — so the
 * O(N) fabric is what caps TP8 prefill at 4.74x instead of 8x (plans/tp-prefill.md §4).
 *
 * Reduce-scatter + all-gather moves only ~2(N-1)/N*msg per rank (~N/2x less fabric).
 * Partition the flat [n] result into N CONTIGUOUS slices; slice s = [n*s/N, n*(s+1)/N).
 *   PHASE 1 (reduce-scatter): this rank OWNS slice `rank`. It sums the N peers' partials
 *     over its owned slice (f32 acc, r=0..N-1 in order — BIT-IDENTICAL to the one-shot)
 *     and writes the reduced values IN-PLACE over its OWN peer_scratch partial slot's
 *     slice. No reader barrier needed: slice s of ANY partial is read only by rank s, so
 *     no peer ever reads the region we overwrite (each rank touches only its own slice).
 *   PHASE 2 (all-gather): every rank copies each slice s from peer s's (now-reduced)
 *     partial slot into its local full `out`.
 * Two cross-GPU rendezvous bracket the phases: gate_rs (all partials published, same
 * gate the one-shot uses) and gate_ag (every rank's reduced slice written+visible). The
 * extra sync latency is amortised BECAUSE the message is large — the opposite trade to
 * decode, which keeps the one-shot. Same xctr discipline as d_xreduce_mega. */
__device__ __forceinline__ void d_xreduce_twoshot_mega(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, size_t xctr_byte_off,
    uint32_t gate_rs, uint32_t gate_ag, uint64_t deadline_ticks, uint32_t* status,
    unsigned slice, unsigned nblk) {
    __shared__ int bailed;
    const unsigned tid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;

    /* ---- RENDEZVOUS 1 (gate_rs): every rank has published its full partial. One
     * workgroup announces this rank to every peer; ALL workgroups wait N arrivals then
     * take ONE system acquire (never inside the spin — it would inv the L2 every poll). */
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();
    if (slice == 0 && threadIdx.x == 0)
        for (uint32_t r = 0; r < nranks; r++)
            xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_rs));
    if (threadIdx.x == 0) {
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_rs);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1; break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!bailed) xctr_acquire();
    }
    __syncthreads();
    if (bailed) return;

    /* ---- PHASE 1 reduce-scatter: reduce this rank's OWNED slice, write it in-place into
     * this rank's own (peer-visible) partial slot. All nblk workgroups collaborate on the
     * single owned slice. r=0..N-1 in-order f32 accumulate => bit-exact with the one-shot. */
    const uint32_t my_lo = (uint32_t)(((uint64_t)n * rank) / nranks);
    const uint32_t my_hi = (uint32_t)(((uint64_t)n * (rank + 1)) / nranks);
    bf16* my_part = (bf16*)((char*)peer_scratch[rank] + slot_bytes);
    for (uint32_t e = my_lo + tid; e < my_hi; e += stride) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nranks; r++) {
            const bf16* part = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
            acc += bf2f(as_glob(part)[e]);
        }
        as_glob(my_part)[e] = f2bf(acc);
    }
    __syncthreads();

    /* ---- RENDEZVOUS 2 (gate_ag): every rank's reduced slice is written + visible. ---- */
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();
    if (slice == 0 && threadIdx.x == 0)
        for (uint32_t r = 0; r < nranks; r++)
            xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_ag));
    if (threadIdx.x == 0) {
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_ag);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1; break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!bailed) xctr_acquire();
    }
    __syncthreads();
    if (bailed) return;

    /* ---- PHASE 2 all-gather: assemble the full reduced vector into `out`, each slice s
     * read from peer s's now-reduced partial slot (s==rank is a local copy). ---- */
    for (uint32_t s = 0; s < nranks; s++) {
        const uint32_t lo = (uint32_t)(((uint64_t)n * s) / nranks);
        const uint32_t hi = (uint32_t)(((uint64_t)n * (s + 1)) / nranks);
        const bf16* src = (const bf16*)((const char*)peer_scratch[s] + slot_bytes);
        for (uint32_t e = lo + tid; e < hi; e += stride) as_glob(out)[e] = as_glob(src)[e];
    }
}

#endif /* PLOW_OP_COLLECTIVE_H */
