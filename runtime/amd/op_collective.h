/* op_collective.h — the cross-GPU (tensor-parallel) collectives.
 *
 * These are the ONLY device ops that touch PEER VRAM. They ride the transport the
 * `tp-transport` agent proved: a coarse-grained VRAM buffer
 * made visible to every GPU with hsa_amd_agents_allow_access, synchronized by a
 * SYSTEM-scope atomic on peer memory (no HSA signal, no host, ~90 ns one-way).
 *
 * The single-GPU counter-gate in interp.hip uses __HIP_MEMORY_SCOPE_AGENT (orders
 * across XCDs within ONE GPU). The ONLY change for cross-GPU is widening that scope
 * to SYSTEM so the release/acquire reaches past this device's per-XCD L2 out onto
 * XGMI to the peer — exactly the xctr_* trio below.
 *
 * SHARED with the golden wrapper (test_kernels.hip) and the 2-GPU microbench
 * (tp_allreduce_kernels.hip), so the CPU/2-GPU reference validates THIS code, not a
 * copy of it — the same discipline as every other op_*.h.
 */
#ifndef PLOW_OP_COLLECTIVE_H
#define PLOW_OP_COLLECTIVE_H

#include "amd_common.h" /* bf16, bf2f/f2bf, PLOW_THREADS */

/* Ceiling instrument, off in every shipping object. See d_xreduce_mega. */
#ifndef PLOW_XR_NOWAIT
#define PLOW_XR_NOWAIT 0
#endif
/* PREFILL ceiling instrument, off in every shipping object. See d_xreduce_twoshot_mega.
 * PLOW_XR_NOWAIT deletes BOTH of the two-shot's rendezvous waits; PLOW_XR_NOWAIT_RS
 * deletes only the FIRST (gate_rs, "every rank published its partial"), which is the
 * only one a producer-side tile watermark could ever replace — gate_ag is a barrier
 * over slices that all nblk workgroups wrote collaboratively. Splitting them prices
 * the addressable half separately from the whole. */
#ifndef PLOW_XR_NOWAIT_RS
#define PLOW_XR_NOWAIT_RS 0
#endif
/* THE ABSOLUTE PROTOCOL CEILING (-DPLOW_XR_NOSIG=1): deletes the WAIT *and* the
 * SIGNALLING, keeping only the acquire and the reduce body. PLOW_XR_NOWAIT prices what
 * a redesign could win by waiting less; it CANNOT see the cost of the announcement
 * itself — and in the prefill two-shot the announcement is the dominant term by count:
 * gate_ag needs `nranks*nblk` arrivals, so all 256 workgroups signal all 4 peers =
 * 1024 remote system-scope RMWs per collective per rank, 156 times per launch.
 * A tile/watermark scheme changes exactly that traffic (N remote RMWs -> C*N remote
 * stores), so NOWAIT alone would price the wrong half and could report "no prize"
 * for a protocol whose real cost is announcement. NOSIG bounds BOTH halves at once:
 * nothing that still publishes and observes progress can beat it. Numerically wrong. */
#ifndef PLOW_XR_NOSIG
#define PLOW_XR_NOSIG 0
#endif
/* THE POSITIVE CONTROL FOR EVERY WRONG-BY-CONSTRUCTION ARM (-DPLOW_XR_SHUFFLE=1).
 *
 * NOWAIT/NOSIG are numerically wrong, and this model's MoE routing is DATA-DEPENDENT: a
 * prefill whose activations are garbage from layer 0 routes tokens differently, so its
 * grouped-expert ops may do a different amount of work and the launch can get faster for
 * a reason that has NOTHING to do with the rendezvous. That confound is fatal to the
 * measurement — it produces exactly the speedup a real protocol win would.
 *
 * This arm separates them. It keeps the ENTIRE protocol — both rendezvous, every signal,
 * every acquire, the same workgroup count, the same number of loads and stores, the same
 * slot — and only rotates the peer READ index by n/2. The output is garbage in the same
 * way, so the routing artefact (if any) appears in full; the protocol cost is untouched.
 *
 *   SHUFFLE ~ base    => the routing artefact is negligible, and NOWAIT/NOSIG measure the protocol.
 *   SHUFFLE ~ NOWAIT  => the "win" is the artefact, and the rendezvous ceiling is ~0.
 *
 * The rotation stays inside the same partial slot (indices are mod n), so it reads only
 * memory the unmodified arm also reads. */
#ifndef PLOW_XR_SHUFFLE
#define PLOW_XR_SHUFFLE 0
#endif

/* PLOW_XR_MLP=1: PEER-BATCHED REDUCE. Opt-in build axis, BIT-IDENTICAL, default OFF.
 *
 * The REDUCE bodies (d_xreduce, and the two-shot's PHASE 1) walk the N peers with a runtime
 * trip count over a pointer table that lives in memory, and the ISA that falls out of that on
 * gfx942 (llvm-objdump of a probe TU, ROCm 7.2.4) is, PER ELEMENT PER PEER:
 *
 *     global_load_dwordx2 v[4:5], v1, s[14:15]   ; re-read peer_scratch[r] from memory
 *     s_waitcnt vmcnt(0)                         ; stall on the POINTER
 *     v_lshl_add_u64 ... slot_bytes ... e*2
 *     global_load_ushort v4, v[4:5], off         ; the remote bf16
 *     s_waitcnt vmcnt(0)                         ; stall on the DATA
 *     v_add_f32_e32 v3, v3, v4
 *
 * - TWO full memory stalls per peer, and ZERO peer-level memory parallelism: a thread has one
 * outstanding remote read at a time and the N xGMI links are walked one round trip after
 * another. That is the SAME class of defect the all-gather stagger fixed (one link busy at a
 * time), one level down: there the links were serialised across workgroups, here they are
 * serialised inside a thread. It is also exactly what the FUSED decode variant already avoids
 * on purpose - d_xreduce_add_norm_mega's comment says "nranks * XRN_VEC independent issues per
 * thread, so the reduce phase has the MLP the scalar form lacked" - and the scalar form is
 * still what the plain XREDUCE (every MoE-seam decode collective) and the two-shot's
 * reduce-scatter (half of the prefill collective's fabric bytes) run.
 *
 * This arm hoists the N peer bases into registers ONCE per thread and issues all N remote
 * loads before consuming any of them, on the TP8 shape only (`nranks == 8`, the shipped
 * degree; any other N takes the untouched loop). The accumulate stays r = 0..N-1 in f32, the
 * element->thread map is unchanged, and the load WIDTH stays 2 B scalar - the xrbw probe's
 * finding that narrow beats wide on the uncached peer path is about how ONE link is walked and
 * is not disturbed by issuing eight of them at once. BIT-IDENTICAL by construction.
 *
 * Not the same knob as anything in xrbw's table: `2 B x4 unroll` there added depth on ONE
 * peer's stream (four lines from the same link) and LOST; this adds depth ACROSS peers (one
 * line from each of eight links).
 *
 * >>> MEASURED, AND IT IS A NULL. DEFAULT STAYS OFF. DO NOT RE-DERIVE. <<<
 * (perf-data/plow-gfx942/glm52-collective-tuning-mi300x.md, 8x MI300X)
 *
 *   xgmi_dir [C], the real reduce-scatter shape, 8 devices concurrent:
 *       2 B pull, SERIAL peers (this file, unchanged)  289.4 GB/s per rank
 *       2 B pull, BATCHED peers (this arm)             181.7 GB/s per rank
 *   served, bench_speed, 2 interleaved rounds, TTFT median:
 *       1024  350.5 -> 350.9  (+0.1%)   control's own round-to-round spread 1.3%
 *       4096 1010.4 -> 1021.5 (+1.1%)                                        0.7%
 *       8192 1729.2 -> 1757.2 (+1.6%)                                        1.1%
 *   Extended coherence gate (Paris / 391 / Au+jewelry / a 300-token real-text answer)
 *   CHARACTER-IDENTICAL between arms, as a bit-identical change must be.
 *
 * WHY, and it is the same reason three times now: this fabric is limited by REQUEST
 * CONCURRENCY ACROSS THREADS, not by per-thread depth. The probe's [E] section walks the
 * all-gather from 19 to 304 workgroups and the rate is DEAD LINEAR in workgroup count
 * (18.4 / 36.5 / 72.6 / 141.8 / 240.5 GB/s) — the machine never saturates the links, so the
 * concurrency is already all there is and eight loads in one thread buy nothing while
 * lengthening the loop-carried dependence. Same rule as xrbw's "wider loads lose" and
 * "2 B x4 unroll loses". The serialised, scalar, depth-1 peer walk above is NOT a defect;
 * it is the right shape for this silicon, and this knob is the record of that.
 *
 * Kept, correct and bit-identical, so the falsification is reproducible rather than a claim. */
#ifndef PLOW_XR_MLP
#define PLOW_XR_MLP 0
#endif
/* The TP degree the batched form is specialised for. Eight peer bases plus eight in-flight
 * bf16 is ~18 VGPRs on bodies that have them; a runtime-N version would spill the pointer
 * array to scratch and lose to the loop it replaces. */
#define PLOW_XR_MLP_N 8u
/* Off under the wrong-by-construction probe axis: SHUFFLE's whole job is to keep the shipped
 * body's instruction mix while rotating the read index, so it must not also change the mix. */
#define PLOW_XR_MLP_ON (PLOW_XR_MLP && !PLOW_XR_SHUFFLE)

/* PLOW_XR_AGG -- DEVICE-LOCAL AGGREGATION OF THE TWO-SHOT'S `gate_ag` SIGNAL
 * (default in gfx942/gfx950 prefill objects; objects-only, no blob or emitter change).
 *
 * WHAT IT ATTACKS. `gate_ag` is the two-shot's SECOND rendezvous and, unlike `gate_rs`, it
 * needs `nranks*nblk` arrivals -- every workgroup must speak for itself, because PHASE 1
 * writes the owned slice COLLABORATIVELY (the note on RENDEZVOUS 2 below is the argument and
 * it stands). Without aggregation, each of the `nblk` workgroups issues `nranks` SYSTEM-scope returning
 * RMWs on ONE 128 B line per peer: at nblk=304, tp=8 that is 2432 remote atomics per rank per
 * collective, all contending on eight lines.
 *
 * MEASURED (perf-data/plow-gfx942/glm52-collective-tuning-mi300x.md section 6.2): a
 * 1-signaller N-way gate round costs 8.2 us; this 304-signaller one costs 51.8 us -- 6.3x.
 * The mechanism is confirmed independently by the atomics probe in
 * glm52-packet-protocol-xcd.md section 2: a returning atomic is 392 ns on a PRIVATE line and
 * 2770 ns at 304 claimants on ONE line (7.1x). It was originally prefill-only. Batched K3
 * decode also emits `XReduceTwoShot`; B32 has 186 instances, so the same aggregation is
 * decode-critical there. B1 and ordinary decode retain the one-shot `d_xreduce_mega`.
 *
 * THE FIX, AND WHY IT NEEDS NO BLOB CHANGE. `PLOW_CTR_STRIDE` is 32 words (128 B) per
 * counter and only word 0 is ever used, so word 1 of this gate's OWN line is a free
 * device-local aggregation counter that the host already zeroes every step. Each workgroup
 * arrives with a SYSTEM-scope RELEASE RMW on word 1 -- the RMW itself is the release that
 * orders that workgroup's PHASE 1 stores, and the scope keeps the op memory-side, off the
 * XCD L2. The workgroup that closes the count takes the SYSTEM acquire and issues the
 * `nranks` remote signals on behalf of the whole device. Remote atomics per rank per
 * collective: 2432 -> 8; the 304 arrivals contend on ONE LOCAL line instead.
 *
 * THE COUNT IS UNCHANGED, DELIBERATELY. The closer adds `nblk` per peer rather than 1, so
 * word 0 still lands on exactly `nranks*nblk` and the waiter's threshold, the deadline arm
 * and `plowrt`'s host-side `audit_xctr` expectation (`gate_expectations`, `n_gpu * blocks`)
 * are all untouched. An object built with this axis and one built without it produce the same
 * final counter state; nothing outside this function can tell them apart.
 *
 * ORDERING, stated because it is the whole risk -- the first cut (release FENCE + relaxed
 * AGENT arrival, agent acquire in the closer) failed needle@3000 and is LESSONS.md #19.
 * Workgroup A: PHASE 1 stores; SYSTEM-scope RELEASE RMW on word 1. Closer B: the same RMW
 * returns `nblk-1`; B takes the SYSTEM acquire (`xctr_acquire`), then issues SYSTEM-scope
 * RELEASE RMWs to every peer's word 0. A's stores happen-before A's release RMW (same wave,
 * release semantics), which happens-before B's acquire (RMWs on word 1 form the release
 * sequence, and every edge is at SYSTEM scope -- no scope narrower than the final observer
 * appears anywhere in the chain), which happens-before B's peer signals, which happen-before
 * the peer's `xctr_acquire()`. Transitive, so a peer that observes the count sees every
 * workgroup's slice. SYSTEM scope on the arrival also keeps word 1 memory-side: an
 * agent-scope RMW would run cached in the arriving XCD's L2 on the SAME 128 B line the
 * peers' signals are updating at memory, and a stale-line writeback can lose word-0 counts.
 * BIT-IDENTICAL: no value is touched.
 *
 * ONE-SHOT UNTOUCHED. `d_xreduce_mega` already signals from slice 0 only (threshold
 * `nranks`), so it has nothing to aggregate; this axis is the two-shot's alone. */
#ifndef PLOW_XR_AGG
#define PLOW_XR_AGG 0
#endif
/* Inert under the ceiling probes, which delete the signal outright. */
#define PLOW_XR_AGG_ON (PLOW_XR_AGG && !PLOW_XR_NOSIG)

#define PLOW_XR2_SKIP_RS (PLOW_XR_NOWAIT || PLOW_XR_NOWAIT_RS || PLOW_XR_NOSIG)
#define PLOW_XR2_SKIP_AG (PLOW_XR_NOWAIT || PLOW_XR_NOSIG)
/* Marginal cost of ONE MORE system-scope acquire fence on this exact path. See the
 * instrument's own comment in d_xreduce_mega. 1 = shipping.
 *
 * This is the ONE quantity a chunked/watermark redesign is priced on: such a scheme takes
 * C acquires per gate where the protocol takes 1, so it PAYS (C-1) of these to win at most
 * a fraction of what PLOW_XR_NOWAIT deletes. None of the NOWAIT/NOSIG arms can see it —
 * they all keep the acquire, deliberately — so it needs its own knob.
 *
 * NUMERICALLY CORRECT: an extra acquire fence cannot change a value, so every ACQ_N arm
 * must reproduce the control's output exactly. That is the instrument's self-check.
 * The relaxed poll between fences is what stops LLVM folding k adjacent identical fences
 * into one; it compares against a value the counter cannot hold, so nothing else changes. */
#ifndef PLOW_XR_ACQ_N
#define PLOW_XR_ACQ_N 1
#endif
#if PLOW_XR_ACQ_N > 1
#define PLOW_XR_EXTRA_ACQ(gw)                                        \
    do {                                                             \
        const uint32_t* _g = (gw);                                   \
        for (int _a = 1; _a < (PLOW_XR_ACQ_N); _a++) {               \
            if (xctr_poll(_g) == 0xFFFFFFFFu) bailed = 1;            \
            xctr_acquire();                                          \
        }                                                            \
    } while (0)
#endif

/* ---- cross-GPU counter helpers (SYSTEM scope) ---------------------------------
 * Mirror interp.hip's agent-scope ctr_poll/ctr_acquire/ctr_signal, widened to
 * __HIP_MEMORY_SCOPE_SYSTEM. Same discipline:
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
 * GEMV epilogue, no extra copy) — and the SE_XCTR gate has
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
 * and the transport tp_reduce_oneshot building block.
 *
 * ---- THE FOLDED ALL-GATHER TERM (gcols != 0) -------------------------------------
 *
 *   gslot_bytes  : byte offset of a SECOND peer slot holding a COLUMN-PARALLEL partial
 *   gcols        : that partial's per-rank column count (0 disables the whole term)
 *   row_w        : the full row width of `out`, so a [T, row_w] result can be indexed
 *
 * A column-parallel producer leaves rank r owning output columns [r*gcols, (r+1)*gcols)
 * and needs an ALL-GATHER, not an all-reduce. Kimi-K3's LatentMoE tail has one of each,
 * added together, and they land in the same result:
 *
 *     out = sum_r shd_r            (row-parallel shared-expert down_proj -> slot_bytes)
 *         + concat_r yh_r          (column-parallel routed_expert_up_proj -> gslot_bytes)
 *
 * Folding the gather into this loop is what lets `routed_expert_up_proj` be sharded at
 * ALL: on its own the gather would need its own packet AND its own cross-rank
 * rendezvous, and this tree measures an added serial decode packet at ~5.3 us
 * (op_norm.h's d_norm_residual_norm note) — 92 MoE layers of that is 0.49 ms/token
 * against the 0.65 ms the sharding saves. Here it is one extra bf16 load per element on
 * a packet that already reads nranks of them, on the rendezvous that already happened.
 *
 * The owner rank and its LOCAL index are derived from the column, not from `e`: at
 * prefill `out` is [T, row_w] and rank r's slot holds a COMPACT [T, gcols], so row m's
 * columns live at m*gcols. At decode (T = 1, n = row_w) that degenerates to e/gcols.
 *
 * gcols == 0 leaves the loop byte-identical to the pre-existing one, which is what
 * every non-K3 collective and every tp=1 build takes. */
__device__ __forceinline__ void d_xreduce(bf16* out, const void* const* peer_scratch,
                                          uint32_t slot_bytes, uint32_t nranks, uint32_t n,
                                          unsigned slice, unsigned nblk,
                                          uint32_t gslot_bytes = 0, uint32_t gcols = 0,
                                          uint32_t row_w = 0) {
    const unsigned base = slice * PLOW_THREADS + threadIdx.x;
    const unsigned step = nblk * PLOW_THREADS;
#if PLOW_XR_MLP_ON
    /* PEER-BATCHED (see PLOW_XR_MLP): bases hoisted, N remote loads in flight, same sum
     * order and same element->thread map => bit-identical. The folded all-gather term
     * (gcols) keeps the general loop: it is K3-only and its owner-derived second load is a
     * different access pattern. */
    if (nranks == PLOW_XR_MLP_N && gcols == 0) {
        const bf16* p[PLOW_XR_MLP_N];
#pragma unroll
        for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++)
            p[r] = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
        for (uint32_t e = base; e < n; e += step) {
            bf16 v[PLOW_XR_MLP_N];
#pragma unroll
            for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++) v[r] = as_glob(p[r])[e];
            float acc = 0.0f;
#pragma unroll
            for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++) acc += bf2f(v[r]);
            st_act1(&as_glob(out)[e], f2bf(acc));
        }
        return;
    }
#endif
    for (uint32_t e = base; e < n; e += step) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nranks; r++) {
            const bf16* part = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
#if PLOW_XR_SHUFFLE
            acc += bf2f(as_glob(part)[((e) + (n >> 1)) % n]);
#else
            acc += bf2f(as_glob(part)[e]);
#endif
        }
        if (gcols) {
            /* ROUND THE REDUCTION TO BF16 FIRST, and this is not a detail — it is what makes
             * the fold BIT-EXACT against the two-packet form it replaces.
             *
             * Unfolded, K3's tail was: this collective stored `f2bf(sum_r)` to `shd`, and a
             * separate `d_residual` re-read it and computed `f2bf(yh + bf2f(shd))`. Keeping
             * `sum_r` in f32 across the gather add would skip that intermediate rounding —
             * a 1-bf16-ULP difference per element per layer, which is NOT nothing on this
             * model: 92 MoE layers of it moves the logits by ~0.03 (the same order as the
             * tp=1-vs-tp=8 residual `k3_tp_equivalence.sh` reports) and greedy decode then
             * flips a token and diverges. Measured: the un-rounded form answered
             * ". The capital of X is Y. The capital of ..." where the replicated emit answered
             * ". The population is approximately 67 million people."
             *
             * With the round, `out[e]` is EXACTLY the value the deleted `d_residual` stored,
             * so the shard is bit-neutral and the token stream is identical. */
            acc = bf2f(f2bf(acc));
            const uint32_t c = row_w ? (e % row_w) : e;
            const uint32_t m = row_w ? (e / row_w) : 0u;
            const uint32_t owner = c / gcols;
            const bf16* g = (const bf16*)((const char*)peer_scratch[owner] + gslot_bytes);
            acc += bf2f(as_glob(g)[m * gcols + (c - owner * gcols)]);
        }
        st_act1(&as_glob(out)[e], f2bf(acc));
    }
}

/* ---- ONE-SHOT all-reduce (publish-signal + gate + reduce), fused, no launch -----
 * The whole collective in one device call, matching the design notes §5's
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
 * byte offset inside its peer_scratch, so:
 *     peer r's gate = PLOW_CTR(peer_scratch[r] + xctr_byte_off, gate_id)
 * where xctr_byte_off = (char*)prog.xctr - (char*)prog.peer_scratch[rank] — computed
 * by the caller from the two §12 fields, no 5th field needed.
 *
 * COARSE (1 sync/collective): gate_target = nranks, the host
 * having zeroed xctr before the token. The producing o_proj/down GEMV has already
 * written this rank's partial into peer_scratch[rank] (tp-host binds og/dg there), and
 * the interpreter's ordinary agent-scope gate on that GEMV is this op's LOCAL wait —
 * so by the time we get here this rank's partial is visible; we then cross-signal,
 * wait N arrivals, and reduce. This is exactly the path the 2-GPU microbench proved. */
__device__ __forceinline__ void d_xreduce_mega(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, size_t xctr_byte_off, uint32_t gate_id,
    uint64_t deadline_ticks, uint32_t* status, unsigned slice, unsigned nblk,
    uint32_t gslot_bytes = 0, uint32_t gcols = 0, uint32_t row_w = 0) {
    __shared__ int bailed;
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();

    /* ONE workgroup announces this rank's arrival to every peer (else the threshold
     * would have to be nranks*nblk). */
#if !PLOW_XR_NOSIG
    if (slice == 0 && threadIdx.x == 0) {
        for (uint32_t r = 0; r < nranks; r++) {
            uint32_t* base = (uint32_t*)((char*)peer_scratch[r] + xctr_byte_off);
            xctr_signal(PLOW_CTR(base, gate_id));
        }
    }
#endif
#if PLOW_XR_NOWAIT || PLOW_XR_NOSIG
    /* CEILING INSTRUMENT ONLY (-DPLOW_XR_NOWAIT=1), never a shipping object. Skips the
     * cross-rank rendezvous: this rank still signals every peer and still reduces all N
     * peer slots, so the fabric traffic, the packet count, the gate graph and the workgroup
     * count are all UNCHANGED — the only thing removed is WAITING FOR THE SLOWEST PEER.
     * The result is numerically wrong (a peer's partial may be read before it is written),
     * which is fine: this measures scheduling, and the token delta it produces is exactly
     * the price of cross-rank arrival skew + the rendezvous protocol. Same discipline as
     * PLOW_CHAIN_BYPASS (knob-contract §7a-CHAIN): implementation cost zero, ceiling real. */
    if (threadIdx.x == 0) xctr_acquire();
    __syncthreads();
    d_xreduce(out, peer_scratch, slot_bytes, nranks, n, slice, nblk, gslot_bytes, gcols, row_w);
    return;
#endif
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
#if PLOW_XR_ACQ_N > 1
        PLOW_XR_EXTRA_ACQ(lg); /* marginal-acquire instrument; see the knob's comment */
#endif
    }
    __syncthreads();
    if (bailed) return;

    d_xreduce(out, peer_scratch, slot_bytes, nranks, n, slice, nblk, gslot_bytes, gcols, row_w);
}

/* ---- XARGMAX_FIN: the cross-rank fold for a VOCAB-COLUMN-PARALLEL lm_head ---------
 * The design notes §8d, and the reason `crates/plowrt/src/asset/shard.rs` records
 * lm_head as REPLICATED: without this fold every rank must compute the full-vocab
 * argmax to agree on the token, so all N stream the whole 1.9 GB head every step.
 *
 * Sharded, rank r owns logits [r*vocab_l, (r+1)*vocab_l) and its ARGMAX packets have
 * already reduced that shard to `nparts` packed keys — `amax_pack`'s u64 with an
 * order-preserving u32 image of the bf16 value in [63:32] and the COMPLEMENT of the
 * index in [31:0], so the whole reduction is one unsigned max and ties break toward
 * the lowest index. This op finishes the local fold, rebases the index into GLOBAL
 * vocab space, publishes one u64 to every peer and takes the cross-rank max. The
 * complement has to be re-formed around the rebase: ~(~i + off) != ~(i + off).
 *
 * The published value rides a dedicated xctr COUNTER ID rather than a peer_scratch
 * partial slot. PLOW_CTR_STRIDE is 32 words (128 B) per counter, the host zeroes the
 * whole xctr region every step, and the region sits at the same byte offset in every
 * rank's peer_scratch — so a spare id is 8 peer-visible bytes that need no host
 * binding and cannot alias a live all-reduce partial (the two-slot partial_A/partial_B
 * parity only gives one collective of slack, which is not slack you want to spend on
 * an op that runs after the whole FFN).
 *
 * `val_id` must NOT be the arrival gate's id: the gate is an atomic counter.
 * Each 128-byte line carries sixteen packed 8-byte keys; batches above 16 use consecutive lines.
 * `PLOW_XAMAX_MAX_BATCH` and the emitter's matching assert are the two halves of the bound.
 *
 * KNOWN, UNRESOLVED (perf-data/archive/k3/k3-batched-decode-design.md §9): a GSM8K run at B=4 hit ONE
 * cross-rank disagreement in ~1e4-1e5 steps where two ranks folded DIFFERENT winners that both
 * lay in the same rank's vocab shard — i.e. a peer's read of a published `myv[b]` did not match
 * what the publisher folded. Not a deadline bail (that would surface an id from the bailing
 * rank's OWN shard). The host-side agreement check in `decode_step_batched` is what catches it.
 * Passing a real pointer for `status` below, instead of the `nullptr` every call site uses, is
 * the missing instrument.
 *
 * On deadline the rank keeps its LOCAL argmax rather than hanging the queue — the same
 * bail discipline (and the same silent-wrongness) as d_xreduce_mega's. */
/* Up to 128 rows spanning eight consecutive counter lines. `val_id` carries rows 0..15
 * (16 keys x 8 B = one 128 B PLOW_CTR line, exact), with each later line carrying another 16.
 * The emitter allocates only the lines n_batch needs, so existing narrow blobs stay compatible;
 * the arrival gate stays one counter regardless. */
#define PLOW_XAMAX_MAX_BATCH 128u
#define PLOW_XAMAX_LINE 16u
extern "C" __device__ unsigned plow_xargmax_max_batch_128 = 1;
__device__ __forceinline__ void d_xargmax_fin_mega(
    int* ids, const unsigned long long* part, unsigned nparts, unsigned n_batch,
    unsigned vocab_l, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    size_t xctr_byte_off, uint32_t gate_id, uint32_t val_id, uint64_t deadline_ticks,
    uint32_t* status, unsigned slice, unsigned long long* lds) {
    if (slice != 0) return;
    const unsigned B = n_batch ? n_batch : 1u;
    if (B > PLOW_XAMAX_MAX_BATCH) {
        if (threadIdx.x == 0 && status) *status = 0xBADB0000u | (B & 0xFFFFu);
        return;
    }
    if (B == 1 && threadIdx.x != 0) return;
    auto val_at = [&](const void* base, unsigned b) -> unsigned long long* {
        return (unsigned long long*)PLOW_CTR((uint32_t*)((char*)base + xctr_byte_off),
                                             val_id + (b / PLOW_XAMAX_LINE))
               + (b % PLOW_XAMAX_LINE);
    };
    const PLOW_GLOB unsigned long long* pg = as_glob(part);

    /* Rows are independent. Wide decode used to serialize B*nparts comparisons on lane 0;
     * distribute rows across the interpreter block while keeping each row's reduction order. */
    for (unsigned b = threadIdx.x; b < B; b += blockDim.x) {
        const PLOW_GLOB unsigned long long* pb = pg + (size_t)b * nparts;
        unsigned long long best = 0;
        for (unsigned i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        const unsigned gi = ~(unsigned)(best & 0xFFFFFFFFu) + rank * vocab_l;
        *val_at(peer_scratch[rank], b) =
            (best & 0xFFFFFFFF00000000ull) | (unsigned long long)(unsigned)(~gi);
    }

    /* publish (the release on the signal RMW orders the stores above), then rendezvous */
    if (B > 1) __syncthreads();
    if (threadIdx.x == 0) {
        lds[0] = 0;
        for (uint32_t r = 0; r < nranks; r++)
            xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_id));
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_id);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                lds[0] = 1;
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
    }
    if (B > 1) __syncthreads();
    const bool bailed = lds[0] != 0;
    if (!bailed) xctr_acquire();

    for (unsigned b = threadIdx.x; b < B; b += blockDim.x) {
        unsigned long long best = *val_at(peer_scratch[rank], b);
        if (!bailed)
            for (uint32_t r = 0; r < nranks; r++) {
                const unsigned long long v = *(const unsigned long long*)as_glob(
                    (const unsigned long long*)val_at((void*)peer_scratch[r], b));
                best = v > best ? v : best;
            }
        st_act<int>(&as_glob(ids)[b], (int)~(unsigned)(best & 0xFFFFFFFFull));
    }
}

/* ---- TWO-SHOT all-reduce for the LARGE prefill [T,hidden] message ----------------
 * The one-shot d_xreduce_mega has EVERY rank read ALL N peers' FULL partial: ~(N-1)*msg
 * of fabric traffic per rank. That is optimal for decode's tiny [1,hidden] latency-bound
 * message, but the prefill partial is [T,hidden] — T* bigger, BANDWIDTH-bound — so the
 * O(N) fabric is what caps TP8 prefill at 4.74x instead of 8x.
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
    unsigned slice, unsigned nblk,
    /* FUSED RESIDUAL (t1/t2 on the packet; both null on every pre-fold blob): when `out2`
     * is set, PHASE 2 writes `out2[e] = bf16(resid[e] + reduced[e])` INSTEAD of copying
     * into `out` — the Residual packet that followed this collective is deleted and its
     * whole [n] round trip (2 reads + 1 write) plus this op's own `out` write go with it.
     * BIT-IDENTICAL to d_residual at scale=1: the reduced value is already bf16 (PHASE 1
     * rounded it), and bf2f(a)+bf2f(b) -> f2bf is exactly the Residual's arithmetic. */
    const bf16* __restrict__ resid = nullptr, bf16* __restrict__ out2 = nullptr,
    uint32_t gslot_bytes = 0, uint32_t gcols = 0
#if PLOW_XR_ATTNRES
    , bf16* __restrict__ row_prefix = nullptr, uint32_t row_w = 0
#endif
    ) {
    __shared__ int bailed;
    const unsigned tid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;

    /* ---- RENDEZVOUS 1 (gate_rs): every rank has published its full partial. One
     * workgroup announces this rank to every peer; ALL workgroups wait N arrivals then
     * take ONE system acquire (never inside the spin — it would inv the L2 every poll). */
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();
#if !PLOW_XR_NOSIG
    if (slice == 0 && threadIdx.x == 0)
        for (uint32_t r = 0; r < nranks; r++)
            xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_rs));
#endif
    if (threadIdx.x == 0) {
#if !PLOW_XR2_SKIP_RS
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_rs);
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < nranks) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1; break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
#endif
        /* The acquire STAYS in the ceiling arm — the instrument prices the WAIT, not the
         * fence. Same discipline as d_xreduce_mega's PLOW_XR_NOWAIT. */
        if (!bailed) xctr_acquire();
#if PLOW_XR_ACQ_N > 1
        PLOW_XR_EXTRA_ACQ(PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_rs));
#endif
    }
    __syncthreads();
    if (bailed) return;

    /* ---- PHASE 1 reduce-scatter: reduce this rank's OWNED slice, write it in-place into
     * this rank's own (peer-visible) partial slot. All nblk workgroups collaborate on the
     * single owned slice. r=0..N-1 in-order f32 accumulate => bit-exact with the one-shot. */
    const uint32_t my_lo = (uint32_t)(((uint64_t)n * rank) / nranks);
    const uint32_t my_hi = (uint32_t)(((uint64_t)n * (rank + 1)) / nranks);
    bf16* my_part = (bf16*)((char*)peer_scratch[rank] + slot_bytes);
#if PLOW_XR_MLP_ON
    if (nranks == PLOW_XR_MLP_N) {
        /* PEER-BATCHED reduce-scatter (see PLOW_XR_MLP). This phase is HALF the two-shot's
         * remote bytes and every one of them was a serialised round trip. */
        const bf16* p[PLOW_XR_MLP_N];
#pragma unroll
        for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++)
            p[r] = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
        for (uint32_t e = my_lo + tid; e < my_hi; e += stride) {
            bf16 v[PLOW_XR_MLP_N];
#pragma unroll
            for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++) v[r] = as_glob(p[r])[e];
            float acc = 0.0f;
#pragma unroll
            for (uint32_t r = 0; r < PLOW_XR_MLP_N; r++) acc += bf2f(v[r]);
            st_act1(&as_glob(my_part)[e], f2bf(acc));
        }
    } else
#endif
    for (uint32_t e = my_lo + tid; e < my_hi; e += stride) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nranks; r++) {
            const bf16* part = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
#if PLOW_XR_SHUFFLE
            acc += bf2f(as_glob(part)[((e) + (n >> 1)) % n]);
#else
            acc += bf2f(as_glob(part)[e]);
#endif
        }
        st_act1(&as_glob(my_part)[e], f2bf(acc));
    }
    __syncthreads();

    /* ---- RENDEZVOUS 2 (gate_ag): every rank's reduced slice is written + visible.
     *
     * EVERY WORKGROUP SIGNALS, and the threshold is nranks*nblk — NOT slice-0-signals
     * with threshold nranks, which is what this was and it was a live cross-GPU race.
     *
     * The asymmetry with gate_rs is the whole point. gate_rs announces "my full partial
     * is published", and that partial was written by the producing GEMV in an EARLIER
     * packet, whose completion the interpreter's local counter gate already guarantees
     * before ANY workgroup of this packet runs — so one workgroup may speak for the rank.
     * gate_ag announces "my reduced slice is written", and PHASE 1 above writes that
     * slice COLLABORATIVELY across all nblk workgroups (grid-stride, step nblk*THREADS).
     * __syncthreads() is workgroup-wide, so workgroup 0 reaching it says nothing about
     * workgroups 1..nblk-1. Letting it signal on their behalf let a peer's PHASE 2 read
     * a slice that was still being reduced: ranks disagreed, non-deterministically, with
     * every gate still reading its expected count — which is why the host-side xctr audit
     * saw nothing wrong. It is also why DECODE was unaffected: the one-shot writes its
     * result to LOCAL `out` that no peer ever reads, so it has no such ordering duty.
     *
     * Each workgroup's own system-scope RELEASE orders its own PHASE 1 stores, so the
     * gate reaching nranks*nblk means every workgroup on every rank has both finished
     * writing and released. */
    if (threadIdx.x == 0) bailed = 0;
    __syncthreads();
#if !PLOW_XR_NOSIG
#if PLOW_XR_AGG_ON
    /* ---- DEVICE-LOCAL SIGNAL AGGREGATION (see the PLOW_XR_AGG header note) ----
     * Word 1 of this gate's own 128 B counter line, on THIS rank's scratch only: never
     * peer-read, never audited, zeroed by the host with the rest of the region. */
    if (threadIdx.x == 0) {
        uint32_t* const agg =
            PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_ag) + 1;
        /* The arrival RMW IS the release, at SYSTEM scope — xctr_signal's exact form aimed
         * at word 1 — for two load-bearing reasons. (1) A release RMW orders the ISSUING
         * workgroup's PHASE 1 stores; the first cut's fence+relaxed-agent arrival ordered
         * nobody's and failed needle@3000 (LESSONS.md #19). (2) SYSTEM scope keeps word 1
         * memory-side like word 0: an agent-scope RMW executes cached in this XCD's L2 and
         * dirties the very line the peers' system-scope signals are updating at memory —
         * a stale-line writeback can lose remote word-0 counts. */
        const uint32_t prev =
            __hip_atomic_fetch_add(agg, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == nblk) {
            /* I closed the count for this device. Acquire the other workgroups' releases
             * at SYSTEM scope BEFORE speaking for them — the peers' visibility of every
             * other workgroup's slice runs through this fence. Then ONE remote RMW per
             * peer, carrying nblk, so word 0 still lands on exactly nranks*nblk. */
            xctr_acquire();
            for (uint32_t r = 0; r < nranks; r++)
                __hip_atomic_fetch_add(
                    PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_ag),
                    nblk, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
#else
    if (threadIdx.x == 0)
        for (uint32_t r = 0; r < nranks; r++)
            xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off), gate_ag));
#endif
#endif
    if (threadIdx.x == 0) {
#if !PLOW_XR2_SKIP_AG
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_ag);
        const uint32_t target = nranks * nblk;
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        while (xctr_poll(lg) < target) {
            if (__builtin_amdgcn_s_memrealtime() - t0 > deadline_ticks) {
                if (status) *status = 0xDEAD0000u | rank;
                bailed = 1; break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
#endif
        if (!bailed) xctr_acquire();
#if PLOW_XR_ACQ_N > 1
        PLOW_XR_EXTRA_ACQ(PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off), gate_ag));
#endif
    }
    __syncthreads();
    if (bailed) return;

#if PLOW_XR_ATTNRES
    /* A graph-selected AttnRes consumer needs one complete row per workgroup. Preserve the
     * ordinary reduced output, reproduce the intervening bf16 residual boundary into
     * `row_prefix`, then let the dispatch run AttnRes on the same token-owned rows. Requiring
     * the row count divisible by nranks makes every flat reduce-scatter slice end on a row
     * boundary, so one peer owns each complete row. */
    if (row_prefix) {
        if (row_w == 0 || n % row_w != 0) __builtin_trap();
        const uint32_t rows = n / row_w;
        if (rows % nranks != 0) __builtin_trap();
        const uint32_t rows_per_rank = rows / nranks;
        for (uint32_t m = slice; m < rows; m += nblk) {
            const size_t base = (size_t)m * row_w;
            const uint32_t owner = m / rows_per_rank;
            const bf16* src = (const bf16*)((const char*)peer_scratch[owner] + slot_bytes);
            for (uint32_t d = threadIdx.x; d < row_w; d += PLOW_THREADS) {
                const bf16 reduced = as_glob(src)[base + d];
                st_act1(&as_glob(out)[base + d], reduced);
                st_act1(&as_glob(row_prefix)[base + d],
                        resid ? f2bf(bf2f(as_glob(resid)[base + d]) + bf2f(reduced))
                              : reduced);
            }
        }
        __syncthreads();
        return;
    }
#endif

    /* ---- PHASE 2 all-gather: assemble the full reduced vector into `out`, each slice s
     * read from peer s's now-reduced partial slot (s==rank is a local copy).
     *
     * STAGGERED, and the stagger is 3.6x on this box. The original `for (s = 0; ...)` had
     * every workgroup on every rank draining peer 0, then peer 1, ... — ONE xGMI link
     * saturated at a time while the other six idled. Measured with the xrbw probe (8
     * devices concurrent, 304 WGs x 256, exactly this traffic shape): sequential order
     * moves 64 GB/s per rank — matching the shipped kernel's ~65 GB/s effective to the
     * percent — and starting workgroup w at peer (w + rank) lifts it to 233 GB/s, with
     * all links streaming. The same probe says scalar 2 B loads BEAT wider vectors on
     * the uncached peer path (233 vs 180/140/120 GB/s at 4/8/16 B): a wave already
     * coalesces 64 lanes to 128 B lines, and narrower ops mean more independent
     * requests in flight — so the load width below stays as it is on purpose.
     *
     * BIT-IDENTICAL: the (workgroup, element) -> value map is unchanged; only the ORDER
     * a workgroup visits the slices rotates. `+ rank` also decorrelates the ranks so no
     * two ranks start on the same peer. */
    for (uint32_t i = 0; i < nranks; i++) {
        const uint32_t s = (slice + rank + i) % nranks;
        const uint32_t lo = (uint32_t)(((uint64_t)n * s) / nranks);
        const uint32_t hi = (uint32_t)(((uint64_t)n * (s + 1)) / nranks);
        const bf16* src = (const bf16*)((const char*)peer_scratch[s] + slot_bytes);
        if (gcols) {
            const uint32_t row_w = nranks * gcols;
            for (uint32_t e = lo + tid; e < hi; e += stride) {
                const uint32_t c = e % row_w;
                const uint32_t m = e / row_w;
                const uint32_t owner = c / gcols;
                const bf16* g =
                    (const bf16*)((const char*)peer_scratch[owner] + gslot_bytes);
                const float reduced = bf2f(as_glob(src)[e]);
                const float gathered =
                    bf2f(as_glob(g)[m * gcols + (c - owner * gcols)]);
                st_act1(&as_glob(out)[e], f2bf(reduced + gathered));
            }
            continue;
        }
        if (out2) {
            /* fused residual: the gathered value goes straight into resid+v, and `out` is
             * NOT written — its only reader was the Residual this fold replaces. (The
             * SHUFFLE probe axis keeps the plain copy; a shuffled blob never carries t1.)
             *
             * HAND-BATCHED, because the first cut was a plain per-element
             * load-add-store loop and served 4.7x SLOWER than the split emit: the
             * uncached peer stream lives on independent requests in flight (the
             * stagger note above), and interleaving a dependent local load + add +
             * store per element collapsed that parallelism. Issuing U remote loads
             * up front restores it by construction. */
            constexpr uint32_t U = 8;
            uint32_t e = lo + tid;
            for (; e + (U - 1) * stride < hi; e += U * stride) {
                bf16 sv[U], rv[U];
#pragma unroll
                for (int u = 0; u < (int)U; u++) sv[u] = as_glob(src)[e + (uint32_t)u * stride];
#pragma unroll
                for (int u = 0; u < (int)U; u++) rv[u] = as_glob(resid)[e + (uint32_t)u * stride];
#pragma unroll
                for (int u = 0; u < (int)U; u++)
                    st_act1(&as_glob(out2)[e + (uint32_t)u * stride],
                            f2bf(bf2f(rv[u]) + bf2f(sv[u])));
            }
            for (; e < hi; e += stride)
                st_act1(&as_glob(out2)[e],
                        f2bf(bf2f(as_glob(resid)[e]) + bf2f(as_glob(src)[e])));
            continue;
        }
#if PLOW_XR_SHUFFLE
        for (uint32_t e = lo + tid; e < hi; e += stride)
            st_act1(&as_glob(out)[e], as_glob(src)[((e) + (n >> 1)) % n]);
#else
        for (uint32_t e = lo + tid; e < hi; e += stride) st_act1(&as_glob(out)[e], as_glob(src)[e]);
#endif
    }
}

#endif /* PLOW_OP_COLLECTIVE_H */
