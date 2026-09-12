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

/* Random bf16 partials for the strict-order oracle (tp_allreduce_prefill_bench TP_RANDOM=1):
 * exponents span 2^-8..2^7 so the f32 sum depends on the rank order and a mis-rounded or
 * reordered element is caught; the host computes the same words. */
__device__ __forceinline__ uint32_t tp_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
extern "C" __global__ void tp_fill_random(bf16* part, uint32_t n, uint32_t rank, uint32_t seed) {
    const uint32_t rs = tp_hash(rank * 0x9E3779B9u + seed);
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
        const uint32_t h = tp_hash(e ^ rs);
        part[e] = (bf16)(((h >> 31) << 15) | ((120u + ((h >> 20) & 15u)) << 7) | (h & 0x7fu));
    }
}

#if PLOW_SEQ_PAR_SEAMS
/* PER-PHASE timing (tp_allreduce_prefill_bench TP_PHASES=1): op 25 then op 26 on the same
 * operands, so the reduce-scatter and the all-gather are priced separately, each with its own
 * rendezvous. The packet boundary between them is stood in for by a device-local barrier
 * (system-scope release arrival, acquire on exit), which is what makes op 26's one-workgroup
 * announcement legal. cycles[0] = reduce-scatter + barrier, cycles[1] = all-gather. */
extern "C" __global__ void tp_seams_timed(bf16* out, const void* const* peer_scratch,
                                          uint32_t nranks, uint32_t rank, uint32_t n,
                                          uint32_t slot_bytes, uint64_t xctr_byte_off,
                                          uint32_t iters, uint64_t deadline, uint64_t* cycles,
                                          uint32_t* status, uint32_t* lctr) {
    uint64_t rs = 0, ag = 0;
    for (uint32_t i = 0; i < iters; i++) {
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        d_xreduce_scatter_mega(peer_scratch, nranks, rank, n, slot_bytes, (size_t)xctr_byte_off,
                               2u * i, deadline, status, blockIdx.x, gridDim.x, 0u, 0u, nullptr);
        if (threadIdx.x == 0) {
            __hip_atomic_fetch_add(lctr, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            while (__hip_atomic_load(lctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
                   (i + 1u) * gridDim.x)
                __builtin_amdgcn_s_sleep(2);
            xctr_acquire();
        }
        __syncthreads();
        const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
        d_xall_gather_mega(peer_scratch, nranks, rank, (size_t)xctr_byte_off, 2u * i + 1u,
                           deadline, status, blockIdx.x, gridDim.x, out, n, slot_bytes, nullptr,
                           0u, 0u, nullptr, 0u, 0u);
        __syncthreads();
        const uint64_t t2 = __builtin_amdgcn_s_memrealtime();
        rs += t1 - t0;
        ag += t2 - t1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) { cycles[0] = rs; cycles[1] = ag; }
}

/* ---- SEAMS-OVERLAP BOUNDING MICROBENCHES (tp_seams_overlap_bench.c) ------------------------
 * Price the two ways past the capped seam collectives without touching a packet:
 *   tp_push_rs      the reduce-scatter as a PUSH: every rank writes each band slice of its partial
 *                   into the owner's receive region, then the owner sums recv[0..7] in rank order
 *                   (the strict-order oracle's sum, one bf16 round);
 *   tp_push_busy    the same push issued from an o_proj-shaped tile loop (an MFMA burst per 256x256
 *                   tile, owner-interleaved tile order) against the same loop writing locally (today)
 *                   and writing nothing (compute only);
 *   tp_seams_banded op 26 split into K row bands, each with its own one-workgroup rendezvous;
 *   tp_seams_corun  op 26 on its capped workgroups beside an MFMA + streaming consumer on the rest.
 * ts[4*wg + {0..3}] = start, work done, rendezvous passed, end (s_memrealtime). */
__device__ __forceinline__ uint64_t tp_now(void) { return __builtin_amdgcn_s_memrealtime(); }

/* Every workgroup arrives (the two-shot's gate_ag arrival: system-scope release, device-local
 * aggregation under PLOW_XR_AGG) and waits for nranks * nblk arrivals, then one system acquire. */
__device__ __forceinline__ bool tp_push_gate(const void* const* peer_scratch, uint32_t nranks,
                                             uint32_t rank, size_t xoff, uint32_t gate,
                                             unsigned nblk, uint64_t deadline, uint32_t* status,
                                             int* bailed) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *bailed = 0;
        xr2_arrive_ag(peer_scratch, nranks, rank, xoff, gate, nblk);
        uint32_t* lg = PLOW_CTR((uint32_t*)((char*)peer_scratch[rank] + xoff), gate);
        const uint64_t t0 = tp_now();
        while (xctr_poll(lg) < nranks * nblk) {
            if (tp_now() - t0 > deadline) {
                if (status) *status = 0xDEAD0000u | rank;
                *bailed = 1;
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        if (!*bailed) xctr_acquire();
    }
    __syncthreads();
    return !*bailed;
}

/* Strict-order local sum of this rank's band from recv[0..nranks-1], 16-byte packs. */
__device__ __forceinline__ void tp_recv_sum(const bf16* recv, uint32_t nranks, uint32_t band,
                                            bf16* out, unsigned slice, unsigned nblk) {
    for (uint32_t e = (slice * PLOW_THREADS + threadIdx.x) * 8u; e < band;
         e += nblk * PLOW_THREADS * 8u) {
        float acc[8] = {};
        for (uint32_t r = 0; r < nranks; r++) {
            const bf16v8 v = ld_glob8(recv + (size_t)r * band + e);
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) acc[k] += bf2f(v[k]);
        }
        bf16v8 o;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) o[k] = f2bf(acc[k]);
        st_glob8(out + e, o);
    }
}

/* part: local [n] partial; recv region at recv_off in every rank's peer scratch, [src][band].
 * kpush workgroups push (wave w -> destination (w + rank) % 8, 16-byte stores); all workgroups
 * take the rendezvous and the local sum. Needs nranks == 8 (one wave per destination). */
extern "C" __global__ void tp_push_rs(const bf16* part, const void* const* peer_scratch,
                                      uint32_t nranks, uint32_t rank, uint32_t n,
                                      uint32_t recv_off, uint64_t xoff, uint32_t kpush, bf16* out,
                                      uint64_t deadline, uint32_t* status, uint64_t* ts) {
    __shared__ int bailed;
    const unsigned slice = blockIdx.x, nblk = gridDim.x;
    const uint64_t t0 = tp_now();
    const uint32_t band = n / nranks;
    if (slice < kpush) {
        const uint32_t wave = threadIdx.x >> 6, lane = threadIdx.x & 63u;
        const uint32_t d = (wave + rank) % nranks;
        const bf16* src = part + (size_t)d * band;
        bf16* dst = (bf16*)((char*)peer_scratch[d] + recv_off) + (size_t)rank * band;
        for (uint32_t e = (slice * 64u + lane) * 8u; e < band; e += kpush * 64u * 8u)
            st_glob8(dst + e, ld_glob8(src + e));
    }
    const uint64_t t1 = tp_now();
    if (!tp_push_gate(peer_scratch, nranks, rank, (size_t)xoff, 0u, nblk, deadline, status, &bailed))
        return;
    const uint64_t t2 = tp_now();
    tp_recv_sum((const bf16*)((const char*)peer_scratch[rank] + recv_off), nranks, band, out, slice,
                nblk);
    __syncthreads();
    if (threadIdx.x == 0 && ts) {
        ts[4 * slice] = t0; ts[4 * slice + 1] = t1; ts[4 * slice + 2] = t2; ts[4 * slice + 3] = tp_now();
    }
}

/* o_proj-shaped producer: 256x256 output tiles (65536 elements), owner-interleaved tile order
 * (tile t -> owner t % 8), `mfma_iters` 32x32 bf16 MFMAs per wave per tile, then the tile's
 * epilogue: mode 0 stores it locally into cbuf (today's partial write), mode 1 pushes it into the
 * owner's recv slot (then rendezvous + strict-order local sum into out), mode 2 stores nothing. */
extern "C" __global__ void tp_push_busy(const bf16* part, const void* const* peer_scratch,
                                        uint32_t nranks, uint32_t rank, uint32_t n,
                                        uint32_t recv_off, uint64_t xoff, uint32_t mode,
                                        uint32_t mfma_iters, bf16* cbuf, bf16* out,
                                        uint64_t deadline, uint32_t* status, uint64_t* ts) {
    __shared__ int bailed;
    const unsigned slice = blockIdx.x, nblk = gridDim.x;
    const uint64_t t0 = tp_now();
    const uint32_t band = n / nranks;
    const uint32_t tile = 65536u;
    const uint32_t ntile = n / tile;
    f32x16 acc = {};
    for (uint32_t t = slice; t < ntile; t += nblk) {
        const uint32_t owner = t % nranks, k = t / nranks;
        const uint32_t base = owner * band + k * tile;
        const bf16x8 a = __builtin_bit_cast(bf16x8, ld_glob8(part + base + (threadIdx.x & 63u) * 8u));
        for (uint32_t i = 0; i < mfma_iters; i++) acc = plow_mfma_bf16_32x32(a, a, acc);
        bf16* rdst = (bf16*)((char*)peer_scratch[owner] + recv_off) + (size_t)rank * band;
        for (uint32_t pk = threadIdx.x; pk < tile / 8u; pk += PLOW_THREADS) {
            const uint32_t e = base + pk * 8u;
            const bf16v8 v = ld_glob8(part + e);
            if (mode == 0u) st_glob8(cbuf + e, v);
            else if (mode == 1u) st_glob8(rdst + (e - owner * band), v);
        }
    }
    if (acc[0] == -1.2345e30f && status) *status = 0xACC00000u;
    const uint64_t t1 = tp_now();
    uint64_t t2 = t1;
    if (mode == 1u) {
        if (!tp_push_gate(peer_scratch, nranks, rank, (size_t)xoff, 0u, nblk, deadline, status,
                          &bailed))
            return;
        t2 = tp_now();
        tp_recv_sum((const bf16*)((const char*)peer_scratch[rank] + recv_off), nranks, band, out,
                    slice, nblk);
    }
    __syncthreads();
    if (threadIdx.x == 0 && ts) {
        ts[4 * slice] = t0; ts[4 * slice + 1] = t1; ts[4 * slice + 2] = t2; ts[4 * slice + 3] = tp_now();
    }
}

/* op 26 in `kbands` row bands of n / kbands elements each (band-major layout: band j's slices are
 * contiguous, so each band moves 7/8 of n / kbands per rank over all seven links), one
 * one-workgroup rendezvous per band (gates 1..kbands). Workgroups past the object's seam cap idle. */
extern "C" __global__ void tp_seams_banded(const void* const* peer_scratch, uint32_t nranks,
                                           uint32_t rank, uint32_t n, uint32_t slot, uint64_t xoff,
                                           uint32_t kbands, bf16* out, uint64_t deadline,
                                           uint32_t* status, uint64_t* ts) {
    const unsigned slice = blockIdx.x, nblk = gridDim.x;
    const uint64_t t0 = tp_now();
    const uint32_t nb = n / kbands;
    for (uint32_t j = 0; j < kbands; j++)
        d_xall_gather_mega(peer_scratch, nranks, rank, (size_t)xoff, 1u + j, deadline, status, slice,
                           nblk, out + (size_t)j * nb, nb, slot + j * nb * 2u, nullptr, 0u, 0u,
                           nullptr, 0u, 0u);
    __syncthreads();
    if (threadIdx.x == 0 && ts) {
        const uint64_t t = tp_now();
        ts[4 * slice] = t0; ts[4 * slice + 1] = t; ts[4 * slice + 2] = t; ts[4 * slice + 3] = t;
    }
}

/* Workgroups [0, kag) run op 26 over [n] (mode 0 or 2); workgroups [kag, nblk) run a consumer
 * (mode 1 or 2): a grid-stride stream over the local cbuf with `mfma_iters` MFMAs per 16-byte pack. */
extern "C" __global__ void tp_seams_corun(const void* const* peer_scratch, uint32_t nranks,
                                          uint32_t rank, uint32_t n, uint32_t slot, uint64_t xoff,
                                          bf16* out, const bf16* cbuf, uint32_t cn,
                                          uint32_t mfma_iters, uint32_t mode, uint32_t kag,
                                          uint64_t deadline, uint32_t* status, uint64_t* ts) {
    const unsigned slice = blockIdx.x, nblk = gridDim.x;
    const uint64_t t0 = tp_now();
    if (slice < kag) {
        if (mode != 1u)
            d_xall_gather_mega(peer_scratch, nranks, rank, (size_t)xoff, 1u, deadline, status, slice,
                               kag, out, n, slot, nullptr, 0u, 0u, nullptr, 0u, 0u);
    } else if (mode != 0u) {
        const unsigned cs = slice - kag, cnb = nblk - kag;
        f32x16 acc = {};
        for (uint32_t e = (cs * PLOW_THREADS + threadIdx.x) * 8u; e < cn;
             e += cnb * PLOW_THREADS * 8u) {
            const bf16x8 a = __builtin_bit_cast(bf16x8, ld_glob8(cbuf + e));
            for (uint32_t i = 0; i < mfma_iters; i++) acc = plow_mfma_bf16_32x32(a, a, acc);
        }
        if (acc[0] == -1.2345e30f && status) *status = 0xACC00000u;
    }
    __syncthreads();
    if (threadIdx.x == 0 && ts) {
        const uint64_t t = tp_now();
        ts[4 * slice] = t0; ts[4 * slice + 1] = t; ts[4 * slice + 2] = t; ts[4 * slice + 3] = t;
    }
}

/* Sequence-parallel seam halves (ops 25/26), for tp_seqpar_smoke.c: the reduce-scatter with
 * the FFN-seam gather fold and the local band copy, then the all-gather of up to three
 * arrays. Same bodies the interpreter dispatches. */
extern "C" __global__ void tp_seqpar_rs(const void* const* peer_scratch, uint32_t nranks,
                                        uint32_t rank, uint32_t n, uint32_t slot_bytes,
                                        uint64_t xctr_byte_off, uint32_t gate,
                                        uint64_t deadline, uint32_t* status,
                                        uint32_t gslot_bytes, uint32_t gcols,
                                        bf16* band_copy) {
    d_xreduce_scatter_mega(peer_scratch, nranks, rank, n, slot_bytes, (size_t)xctr_byte_off,
                           gate, deadline, status, blockIdx.x, gridDim.x, gslot_bytes, gcols,
                           band_copy);
}
extern "C" __global__ void tp_seqpar_ag(const void* const* peer_scratch, uint32_t nranks,
                                        uint32_t rank, uint64_t xctr_byte_off, uint32_t gate,
                                        uint64_t deadline, uint32_t* status, bf16* dst0,
                                        uint32_t n0, uint32_t slot0, bf16* dst1, uint32_t n1,
                                        uint32_t slot1, bf16* dst2, uint32_t n2,
                                        uint32_t slot2) {
    d_xall_gather_mega(peer_scratch, nranks, rank, (size_t)xctr_byte_off, gate, deadline,
                       status, blockIdx.x, gridDim.x, dst0, n0, slot0, dst1, n1, slot1, dst2,
                       n2, slot2);
}
#endif

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
/* Tagged one-shot, the production body. Gate id = i & 1023 (parity alternates, the audit
 * bump lands on 1024 xctr lines at `xctr_off`); tag = gate + 1 differs between any two
 * consecutive uses of a slot, which is all the loop needs without the host's per-token
 * zeroing. `tag_off`/`tag_slot` are the four-slot region exactly as PlowProgram carries it. */
extern "C" __global__ void tp_allreduce_tagged(bf16* out, const void* const* peer_scratch,
                                               uint32_t nranks, uint32_t rank, uint32_t n,
                                               uint32_t slot_bytes, uint32_t tag_off,
                                               uint32_t iters, uint64_t deadline,
                                               uint64_t* cycles, uint32_t* status,
                                               uint32_t* lctr, uint32_t order, uint32_t cold,
                                               uint32_t tag_slot, uint32_t xctr_off) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        const uint32_t sb = slot_bytes * (i & 1u);
        if (cold)
            tp_local_produce((bf16*)((char*)peer_scratch[rank] + sb), n, rank, i, order, lctr);
        d_xreduce_tagged_mega(out, peer_scratch, nranks, rank, n, cold ? sb : 0u,
                              (size_t)xctr_off, i & 1023u, deadline, status, blockIdx.x,
                              gridDim.x, 0u, 0u, 0u, tag_off, tag_slot);
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}
/* tp_allreduce_prefill_bench TP_TAGGED=1: the tagged body on the production one-shot
 * operands (plain or folded-gather), gate id = iteration exactly like the one-shot arm. */
extern "C" __global__ void tp_allreduce_tagged_x(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, uint64_t xctr_byte_off, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status,
    uint32_t gslot_bytes, uint32_t gcols, uint32_t tag_off, uint32_t tag_slot) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 0; i < iters; i++)
        d_xreduce_tagged_mega(out, peer_scratch, nranks, rank, n, slot_bytes,
                              (size_t)xctr_byte_off, i, deadline, status, blockIdx.x,
                              gridDim.x, gslot_bytes, gcols, gcols ? n : 0u, tag_off,
                              tag_slot);
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
