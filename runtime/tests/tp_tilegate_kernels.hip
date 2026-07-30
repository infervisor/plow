/* tp_tilegate_kernels.hip — device code for the TILE-TRIGGERED COLLECTIVE microbench.
 *
 * THE QUESTION. A tile-triggered all-reduce splits the [hidden] message into C tiles, gives
 * each tile its own cross-GPU gate, and lets tile c's peer reads start as soon as the GEMV
 * slices producing tile c are done — instead of one gate that waits for the whole GEMV on
 * every rank. It trades ONE system-scope rendezvous for C of them, in exchange for hiding
 * whatever producer SKEW exists.
 *
 * That is a two-sided trade and neither side was ever measured, so this instrument measures
 * both, on the real decode shape (hidden = 7168, so C = ceil(7168/512) = 14 tiles):
 *
 *   COST  — `tg_ar` at skew 0. Both arms do byte-for-byte the same producer work and the same
 *           reduce work; they differ ONLY in the gate structure. The difference IS the added
 *           rendezvous cost.
 *   PRIZE — `tg_ar` swept over an injected skew ramp. The crossover skew is the amount of
 *           producer skew the tiled form needs before it repays its own rendezvous.
 *   REALITY — `tg_gemv_ts` timestamps every workgroup of the REAL decode row-GEMV
 *           (op_gemm.h's `gemv_rows`, the actual shipping code, not a copy) so the measured
 *           skew can be read against that crossover.
 *
 * STEEL-MANNING, deliberately, on three counts — a negative result is only worth something if
 * the losing side was given every advantage:
 *   1. nblk == C, so the producer->tile map is the IDENTITY. Workgroup b produces exactly
 *      tile b. No many-to-one gate fan-in, which is the cheapest a tile gate can possibly be.
 *   2. The injected skew is a LINEAR RAMP (wg 0 first, wg nblk-1 last), so tile 0 is ready
 *      immediately and the tiled arm has the maximum possible head start. Real GEMV skew is
 *      diffuse (devbuild.rs: "the straggler is diffuse rather than one persistently slow CU"),
 *      which is strictly worse for tiling.
 *   3. The tiled arm carries NO local barrier at all, while the coarse arm pays a full
 *      agent-scope one. That is the coarse form's genuine cost and it is charged in full.
 *
 * Every arm must land the SAME bit-exact sum — the host checks it on every rank at every point.
 * A gate that releases early is a silent wrong answer across ranks, so the value check is the
 * correctness proof, not a nicety. It has already earned its keep twice: it caught the harness
 * announcing a rank before all its workgroups had published (see mode 3's precondition note).
 *
 * ---- WHAT IT MEASURED (gfx950 x8, MI355X, XGMI) --------------------------------------------
 * THE ANSWER IS NO, and the reason is not cost — it is that there is no prize.
 *
 * Sweeping INJECTED producer skew from 0 to 8 us, the tiled-minus-coarse makespan gap is FLAT.
 * At decode (n=7168, 14 tiles) it is -0.12 us at TP2, TP4 and TP8 alike, and it does not move
 * as skew grows by 8 us; the isolated tile-triggering term decays toward zero (TP4: +0.182 us
 * at zero skew, +0.026 us at 8 us; TP8 it goes NEGATIVE, -0.041 us). Tile gates hide none of
 * the skew they exist to hide.
 *
 * That is not a surprise once it is written down — it is `collapse`, measured.
 * lean-plow/Plow/CounterGranularity.lean proves `max_v (max_{u in P v} finish u)` collapses to
 * `max over ALL u` whenever the consumer maps COVER the producer, for any maps whatsoever. The
 * collective's consumer (RMSNorm, and K3's AttnRes) reads the whole [hidden] row, so it covers
 * every tile, so the barrier at the end re-imposes the global straggler no matter how finely
 * the gates upstream are cut. Cutting the gate finer cannot beat the slowest producer, and the
 * slowest producer is what the coarse gate already waits for.
 *
 * And the prize would have been tiny anyway: the REAL decode row-GEMV's workgroup skew is
 * 0.196 us of a 1.056 us GEMV at TP2, 0.090 of 0.792 at TP4, and 0.042 of 0.714 at TP8 — it
 * SHRINKS as TP grows, because the shard each rank owns gets smaller and more uniform. Uniform
 * work is exactly `collapse`'s hypothesis.
 *
 * At PREFILL a tiled one-shot looks 2 us ahead of a coarse one-shot at TP4 — and that reverses
 * to 3.1-3.3 us BEHIND at TP8, because every workgroup must announce itself to every peer and
 * that traffic scales with nranks*nblk (op_collective.h's PLOW_XR_NOSIG comment predicted
 * exactly this). It never beats the shipping two-shot's fabric advantage at scale either.
 *
 * Keep this instrument. The claim "finer gates will overlap the collective" is the kind that
 * regrows, and re-running this is cheaper than re-arguing it.
 */
#include "../amd/op_collective.h"
#include "../amd/op_gemm.h"

/* AGENT-scope counter trio, byte-for-byte interp.hip's `ctr_poll`/`ctr_acquire`/`ctr_signal`
 * (interp.hip:1585-1611). Duplicated rather than included because they live inside the
 * interpreter translation unit, which cannot be pulled into a standalone object. Keeping the
 * exact scopes matters: the coarse arm's local barrier is charged its real cost — a relaxed
 * agent-scope poll, ONE agent acquire fence, and a release RMW on the signal — and understating
 * it would flatter the tiled arm, which is the arm under suspicion. */
__device__ __forceinline__ uint32_t tg_ctr_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void tg_ctr_acquire(void) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
__device__ __forceinline__ void tg_ctr_signal(uint32_t* p) {
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

/* Element-varying, bf16-exact fill: val(r,e) = (r+1) * (1 + (e&7)*0.125). Same pattern as
 * tp_allreduce_kernels.hip, so a mis-indexed peer read or a skipped tile is caught by value
 * rather than by a uniform 1+2=3 that any bug would still satisfy. */
__device__ __forceinline__ bf16 tg_val(uint32_t r, uint32_t e) {
    return f2bf((float)(r + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}

extern "C" __global__ void tg_fill(bf16* p, uint32_t n, uint32_t rank) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
        p[e] = tg_val(rank, e);
}

/* Fill a weight/activation buffer with a small bounded pattern — the GEMV timing does not
 * depend on the values, only on the bytes moved, but denormals/NaN would. */
extern "C" __global__ void tg_fill_w(bf16* p, uint32_t n, uint32_t seed) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
        p[e] = f2bf((float)((e + seed) & 7u) * 0.125f - 0.5f);
}

/* ---- K1: per-workgroup completion timestamps of the REAL decode row-GEMV ---------
 * This is the "is there any skew to hide?" measurement. `gemv_rows` is op_gemm.h's shipping
 * decode kernel; GV_BLOCKED=1 (its default) gives workgroup `slice` the contiguous output
 * range [slice*per, slice*per+per), which is precisely the partition a tile gate would need.
 *
 * ts[2*b] / ts[2*b+1] are workgroup b's first and last s_memrealtime. All workgroups are
 * resident simultaneously (the launch is sized to fit), so these timestamps are comparable:
 * the spread of ts[2*b+1] across b IS the producer skew a tile gate could hide. */
extern "C" __global__ void tg_gemv_ts(bf16* C, const bf16* x, const bf16* W,
                                      uint32_t N, uint32_t K, uint64_t* ts, uint32_t iters) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 0; i < iters; i++)
        gemv_rows<1, false, false>(C, x, W, nullptr, nullptr, 1, N, K, blockIdx.x, gridDim.x,
                                   nullptr);
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = t1; }
}

/* Busy-wait `ticks` of s_memrealtime. Models a producer workgroup that finishes late. */
__device__ __forceinline__ void tg_spin(uint64_t ticks) {
    if (!ticks) return;
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
}

/* ---- K2: the two arms, COARSE (1 gate) vs TILED (C gates) ------------------------
 *
 *   mode 0 COARSE — every workgroup writes its tile of this rank's partial, then a full
 *     agent-scope local barrier (all nblk arrived: this is what Dep::Coarse compiles to),
 *     then workgroup 0 announces this rank to every peer on gate 0, every workgroup waits
 *     nranks arrivals on gate 0, ONE system acquire, then the whole [n] is reduced
 *     grid-strided. Exactly d_xreduce_mega's shape.
 *
 *   mode 1 TILED — workgroup b writes tile b, RELEASE-announces itself on gate b of EVERY
 *     peer, waits nranks arrivals on its OWN gate b, takes ITS OWN system acquire, and
 *     reduces ONLY tile b. No local barrier, no global rendezvous: b never learns anything
 *     about tile b' != b. This is the proposal, in its most favourable form.
 *
 * MEMORY MODEL, and it is the whole of the correctness argument. The partial store must be
 * visible to a peer BEFORE that peer observes the arrival that releases it. `xctr_signal` is
 * an `__ATOMIC_RELEASE` system-scope RMW, so it orders every prior store by this thread ahead
 * of itself; the waiting rank polls RELAXED (an acquire load in a spin would invalidate L2 on
 * every iteration) and takes exactly ONE `xctr_acquire` — a system-scope acquire fence —
 * after the gate clears and before any peer read. Release-on-signal / one-acquire-after-gate
 * is the same discipline d_xreduce_mega relies on, and it is the ONLY thing making either arm
 * correct. Nothing here is protected by a device-scope fence: peer VRAM is not coherent at
 * agent scope, and an agent-scope release here would be a silent wrong answer.
 *
 * The tiled arm needs one extra property the coarse arm gets for free: the store to tile b
 * and the release on gate b must be by the SAME workgroup, because that is the only thread
 * whose release orders that store. It is — workgroup b does both. A design where GEMV
 * workgroup u produces part of tile b and workgroup v announces it would need u's release to
 * be observed by v first, i.e. a local barrier per tile, and that is a different (worse) cost
 * structure than the one measured here.
 *
 * The reduce is `r = 0..nranks-1` in ascending order with an f32 accumulator in BOTH arms and
 * each element is summed by exactly one workgroup in both, so the summation ORDER is identical
 * and the two arms are bit-comparable. The host checks that they are.
 */
extern "C" __global__ void tg_ar(bf16* out, const void* const* peer_scratch,
                                 uint32_t* const* peer_gate, uint32_t nranks, uint32_t rank,
                                 uint32_t n, uint32_t slot_bytes, uint32_t mode,
                                 uint32_t skew_ticks, uint32_t iters, uint64_t deadline,
                                 uint64_t* cycles, uint32_t* status, uint32_t* lctr,
                                 uint64_t* ts, uint32_t ntiles, uint64_t xctr_byte_off) {
    const unsigned nblk = gridDim.x;
    const unsigned b = blockIdx.x;
    /* Producer partition: workgroup b owns the contiguous output range [e0,e1), matching
     * GV_BLOCKED=1. Independent of the tiling — both arms produce identically. */
    const unsigned per = (n + nblk - 1) / nblk;
    const unsigned e0 = b * per;
    const unsigned e1 = (e0 + per < n) ? (e0 + per) : n;
    /* COMMUNICATION TILING (tiled arm only). `ntiles` gates over `nblk` workgroups: the
     * workgroups are cut into `ntiles` contiguous groups of `gsz`, group g owns tile g, and
     * gate g therefore takes `nranks*gsz` arrivals — every workgroup that produced part of
     * tile g announces itself to every peer. ntiles == nblk is the decode case (one gate per
     * workgroup, gsz = 1); ntiles < nblk is the prefill case, where the collective is 256
     * workgroups wide for bandwidth and only a handful of row-blocks are worth gating. This
     * many-to-one fan-in is the announcement cost op_collective.h's PLOW_XR_NOSIG comment
     * warns about: C*nranks*gsz remote RMWs per collective where the coarse form pays nranks. */
    const unsigned gsz = nblk / ntiles;
    const unsigned grp = b / gsz;
    const unsigned tper = (n + ntiles - 1) / ntiles;
    const unsigned t0e = grp * tper;
    const unsigned t1e = (t0e + tper < n) ? (t0e + tper) : n;
    bf16* mypart = (bf16*)((char*)peer_scratch[rank] + slot_bytes);
    /* Linear skew ramp: workgroup 0 finishes first, workgroup nblk-1 `skew_ticks` later. */
    const uint64_t myskew = nblk > 1 ? (uint64_t)skew_ticks * b / (nblk - 1) : 0;

    __shared__ int bailed;
    const uint64_t tstart = __builtin_amdgcn_s_memrealtime();

    for (uint32_t i = 1; i <= iters; i++) {
        if (threadIdx.x == 0) bailed = 0;
        __syncthreads();

        /* ---- PRODUCER: write my tile of this rank's partial, late by my skew. ---- */
        tg_spin(myskew);
        for (unsigned e = e0 + threadIdx.x; e < e1; e += blockDim.x)
            as_glob(mypart)[e] = tg_val(rank, e);
        __syncthreads();

        if (mode == 0) {
            /* ---- COARSE ---- local barrier over all nblk (what Dep::Coarse costs), then
             * ONE cross-GPU rendezvous on gate 0. */
            if (threadIdx.x == 0) tg_ctr_signal(lctr);
            if (threadIdx.x == 0)
                while (tg_ctr_poll(lctr) < i * nblk) __builtin_amdgcn_s_sleep(1);
            __syncthreads();
            if (threadIdx.x == 0) tg_ctr_acquire();
            __syncthreads();

            if (b == 0 && threadIdx.x == 0)
                for (uint32_t r = 0; r < nranks; r++) xctr_signal(PLOW_CTR(peer_gate[r], 0));
            if (threadIdx.x == 0) {
                const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
                while (xctr_poll(PLOW_CTR(peer_gate[rank], 0)) < i * nranks) {
                    if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
                        if (status) *status = 0xDEAD0000u | rank; bailed = 1; break;
                    }
                    __builtin_amdgcn_s_sleep(2);
                }
                if (!bailed) xctr_acquire();
            }
            __syncthreads();
            if (bailed) return;
            /* Whole message, grid-strided — d_xreduce's partition. */
            for (uint32_t e = b * blockDim.x + threadIdx.x; e < n; e += nblk * blockDim.x) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < nranks; r++)
                    acc += bf2f(as_glob((const bf16*)((const char*)peer_scratch[r] + slot_bytes))[e]);
                as_glob(out)[e] = f2bf(acc);
            }
        } else if (mode == 3) {
            /* ---- THE SHIPPING PREFILL ALGORITHM, for scale ----------------------------
             * op_collective.h's `d_xreduce_twoshot_mega` itself — reduce-scatter + all-gather,
             * ~N/2x less fabric than the one-shot, which is what prefill actually runs today.
             * Called, not copied, so this is the real thing.
             *
             * Without this arm the prefill comparison is rigged: modes 0/1/2 are all ONE-SHOT
             * variants, and a tiled one-shot beating a coarse one-shot says nothing about
             * whether it beats the algorithm in the tree. It has to clear the real bar.
             *
             * Its gates use ABSOLUTE thresholds (nranks, and nranks*nblk), so a timing loop
             * needs a fresh set of counter ids per iteration rather than a rising threshold —
             * hence 3i, 3i+1, 3i+2. Phase 1 also rewrites the partial slot in-place, which is
             * exactly why every arm re-publishes its partial at the top of each iteration:
             * without that, iteration 2 would reduce iteration 1's output.
             *
             * HARNESS SCAFFOLD, and it must be declared because it is charged to this arm.
             * Re-publishing creates a loop-carried hazard the single-shot packet does not have:
             * a rank that finishes iteration i can overwrite its partial slot with iteration
             * i+1's producer values while a slower peer is still all-gathering from it in
             * iteration i. Measured as a real bit-exactness failure before this barrier was
             * added, which is the instrument working. Gate 3i+2 is an end-of-iteration "everyone
             * has finished reading" rendezvous that closes it. The SHIPPING packet needs no such
             * thing — the interpreter's own counter gate on the next packet already separates
             * the collective from whatever rewrites the slot — so this arm is charged ONE
             * rendezvous the real two-shot does not pay, and its number is therefore an UPPER
             * bound on the real two-shot's cost. That bias flatters the tiled arm, not the
             * two-shot, so it is the safe direction for the conclusion drawn below. */
            /* THE TWO-SHOT'S DOCUMENTED PRECONDITION, supplied. gate_rs is announced by
             * workgroup 0 alone, and the comment on d_xreduce_twoshot_mega says why that is
             * sound: "that partial was written by the producing GEMV in an EARLIER packet,
             * whose completion the interpreter's local counter gate already guarantees before
             * ANY workgroup of this packet runs — so one workgroup may speak for the rank."
             *
             * This harness writes the partial inside the same kernel, so that guarantee does
             * not exist for free and must be reconstructed. Without this barrier workgroup 0
             * announces the rank while workgroups 1..255 are still writing, peers all-gather a
             * half-written partial, and the result is wrong — which is precisely what the
             * bit-exact check reported before it was added. Not a defect in the two-shot: a
             * defect in asking it to run without the gate it is specified to run behind. The
             * coarse arm pays the identical barrier, so the arms stay comparable. */
            if (threadIdx.x == 0) tg_ctr_signal(lctr);
            if (threadIdx.x == 0)
                while (tg_ctr_poll(lctr) < i * nblk) __builtin_amdgcn_s_sleep(1);
            __syncthreads();
            if (threadIdx.x == 0) tg_ctr_acquire();
            __syncthreads();
            d_xreduce_twoshot_mega(out, peer_scratch, nranks, rank, n, slot_bytes, xctr_byte_off,
                                   3 * i, 3 * i + 1, deadline, status, b, nblk);
            if (threadIdx.x == 0) {
                uint32_t* base = (uint32_t*)((char*)peer_scratch[rank] + xctr_byte_off);
                for (uint32_t r = 0; r < nranks; r++)
                    xctr_signal(PLOW_CTR((uint32_t*)((char*)peer_scratch[r] + xctr_byte_off),
                                         3 * i + 2));
                const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
                while (xctr_poll(PLOW_CTR(base, 3 * i + 2)) < nranks * nblk) {
                    if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
                        if (status) *status = 0xDEAD0000u | rank; break;
                    }
                    __builtin_amdgcn_s_sleep(2);
                }
            }
            __syncthreads();
        } else if (mode == 2) {
            /* ---- CONTROL: ONE gate, NO local barrier ----------------------------------
             * The isolating control, and the reason the whole instrument is trustworthy.
             *
             * The tiled arm differs from the coarse arm in TWO ways at once, not one: it has C
             * gates instead of 1, AND it has no local N-way barrier in front of the rendezvous.
             * Any difference between them could be either. This arm has the COARSE gate
             * structure — a single rendezvous, one tile, everybody waits for everybody — with
             * the local barrier deleted: instead of a barrier followed by workgroup 0
             * announcing the rank, ALL nblk workgroups announce, and the gate waits
             * `nranks*nblk` arrivals.
             *
             * That is numerically CORRECT, not a ceiling hack: each workgroup's release RMW
             * orders its OWN partial store, and nranks*nblk arrivals means every workgroup of
             * every rank has published. It is exactly the shape of d_xreduce_twoshot_mega's
             * gate_ag, which takes nranks*nblk for the same reason. So its output is checked
             * against the oracle like every other arm.
             *
             * READ IT AS: mode2 - coarse = what deleting the local barrier is worth.
             *             tiled  - mode2 = what TILE-TRIGGERING is worth, and nothing else. */
            if (threadIdx.x == 0)
                for (uint32_t r = 0; r < nranks; r++) xctr_signal(PLOW_CTR(peer_gate[r], 0));
            if (threadIdx.x == 0) {
                const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
                while (xctr_poll(PLOW_CTR(peer_gate[rank], 0)) < i * nranks * nblk) {
                    if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
                        if (status) *status = 0xDEAD0000u | rank; bailed = 1; break;
                    }
                    __builtin_amdgcn_s_sleep(2);
                }
                if (!bailed) xctr_acquire();
            }
            __syncthreads();
            if (bailed) return;
            for (uint32_t e = b * blockDim.x + threadIdx.x; e < n; e += nblk * blockDim.x) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < nranks; r++)
                    acc += bf2f(as_glob((const bf16*)((const char*)peer_scratch[r] + slot_bytes))[e]);
                as_glob(out)[e] = f2bf(acc);
            }
        } else {
            /* ---- TILED ---- my group's gate, my own acquire, my group's tile. No barrier. */
            if (threadIdx.x == 0)
                for (uint32_t r = 0; r < nranks; r++) xctr_signal(PLOW_CTR(peer_gate[r], grp));
            if (threadIdx.x == 0) {
                const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
                while (xctr_poll(PLOW_CTR(peer_gate[rank], grp)) < i * nranks * gsz) {
                    if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
                        if (status) *status = 0xDEAD0000u | rank; bailed = 1; break;
                    }
                    __builtin_amdgcn_s_sleep(2);
                }
                if (!bailed) xctr_acquire();
            }
            __syncthreads();
            if (bailed) return;
            /* ONLY my group's tile, strided across the group — same elements overall, same r
             * order, same f32 acc as the coarse arm, so the two arms are bit-comparable. */
            const unsigned lb = b - grp * gsz;
            for (uint32_t e = t0e + lb * blockDim.x + threadIdx.x; e < t1e;
                 e += gsz * blockDim.x) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < nranks; r++)
                    acc += bf2f(as_glob((const bf16*)((const char*)peer_scratch[r] + slot_bytes))[e]);
                as_glob(out)[e] = f2bf(acc);
            }
        }
        __syncthreads();
    }

    const uint64_t tend = __builtin_amdgcn_s_memrealtime();
    /* MAKESPAN, NOT WORKGROUP 0'S WALL TIME. Timing workgroup 0 alone is a trap and it is the
     * trap this instrument was originally built with: under the TILED arm workgroup 0 owns
     * tile 0, which has zero injected skew, and its gate is signalled only by the peers'
     * workgroup 0 — so it never observes the straggler at all and reports a flat 0.27 us no
     * matter how much skew is injected. That is not a speedup, it is a workgroup that finished
     * early while the collective was still running.
     *
     * The collective's real completion is the MAX over its workgroups, because its consumer is
     * the block's RMSNorm/residual, which reads the whole [hidden] vector and therefore is a
     * barrier over every tile. This is `collapse`'s "max_v max_{P v} = max over everything"
     * (lean-plow/Plow/CounterGranularity.lean) in instrument form: whatever the gates upstream
     * do, the barrier at the end re-imposes the global maximum. So record every workgroup's
     * span and let the host take it. */
    if (threadIdx.x == 0) { ts[2 * b] = tstart; ts[2 * b + 1] = tend; }
    if (b == 0 && threadIdx.x == 0 && cycles) *cycles = tend - tstart;
}
