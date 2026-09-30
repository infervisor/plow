/* op_collective.cuh -- tensor-parallel collectives on the NVIDIA interpreter (ops 24, 29), the
 * semantics of runtime/amd/op_collective.h in their plain form (no gather fold, no fused residual).
 *
 * Every rank's peer region (`prog.peer_scratch[r]`, UVA, peer access enabled on every pair) holds
 * the partial slots and, at the same offset on every rank, the xctr counters. A counter is signalled
 * with a SYSTEM-scope release add and polled with a relaxed system load, then one acquire fence.
 * The host zeroes every rank's xctr before a run; each gate id is used once per run.
 *
 * Sums run r = 0..N-1 in f32 and round once to bf16, so every rank computes identical bytes and the
 * one-shot and the two-shot agree bit for bit. */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_xc {
constexpr unsigned long long DEADLINE_NS = 20ull * 1000 * 1000 * 1000;

__device__ __forceinline__ void signal(uint32_t* p) {
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}
__device__ __forceinline__ uint32_t poll(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t* ctr(const PlowProgram& prog, uint32_t r, uint32_t id) {
    const size_t off = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
    return PLOW_CTR((uint32_t*)((char*)prog.peer_scratch[r] + off), id);
}
/* thread 0 only: spin until the local gate reaches `want`, then acquire at system scope */
__device__ __forceinline__ void wait(const PlowProgram& prog, uint32_t id, uint32_t want) {
    const uint32_t* g = ctr(prog, prog.rank, id);
    const unsigned long long t0 = now_ns();
    while (poll(g) < want) {
        if (now_ns() - t0 > DEADLINE_NS) {
            printf("plow xreduce: rank %u gate %u stuck at %u/%u\n", prog.rank, id, poll(g), want);
            __trap();
        }
        __nanosleep(64);
    }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}
__device__ __forceinline__ const __nv_bfloat16* part(const PlowProgram& prog, uint32_t r, uint32_t slot) {
    return (const __nv_bfloat16*)((const char*)prog.peer_scratch[r] + slot);
}
/* out[e] = bf16(sum_r part_r[e]) over [lo, hi), grid-strided; uint4 when the range allows */
__device__ __forceinline__ void reduce_range(const PlowProgram& prog, __nv_bfloat16* out, uint32_t slot, uint32_t lo, uint32_t hi,
                                             unsigned tid, unsigned stride) {
    const uint32_t N = prog.n_gpu;
    const bool vec = (lo % 8u) == 0 && (hi % 8u) == 0 && (slot % 16u) == 0 && ((uintptr_t)out % 16u) == 0;
    if (vec) {
        for (uint32_t e = lo + tid * 8u; e < hi; e += stride * 8u) {
            float acc[8] = {};
            for (uint32_t r = 0; r < N; r++) {
                const uint4 u = __ldcg(reinterpret_cast<const uint4*>(part(prog, r, slot) + e));
                const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    acc[2 * k] += __uint_as_float(w[k] << 16);
                    acc[2 * k + 1] += __uint_as_float(w[k] & 0xffff0000u);
                }
            }
            uint32_t w[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const __nv_bfloat162 b = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
                w[k] = *reinterpret_cast<const uint32_t*>(&b);
            }
            *reinterpret_cast<uint4*>(out + e) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    } else {
        for (uint32_t e = lo + tid; e < hi; e += stride) {
            float acc = 0.f;
            for (uint32_t r = 0; r < N; r++) acc += __bfloat162float(__ldcg(part(prog, r, slot) + e));
            out[e] = __float2bfloat16_rn(acc);
        }
    }
}
}  // namespace plow_xc

/* PLOW_DOP_XREDUCE (24), one-shot: t0=out i0=H i1=n_gpu i2=slot i3=gate. Slice 0 announces this
 * rank's published partial to every peer; every slice waits N arrivals and sums all N slots. */
__device__ __forceinline__ void d_xreduce_nv(const PlowProgram& prog, __nv_bfloat16* out, uint32_t n, uint32_t slot, uint32_t gate,
                                             unsigned slice, unsigned nblk) {
    using namespace plow_xc;
    if (slice == 0 && threadIdx.x == 0)
        for (uint32_t r = 0; r < prog.n_gpu; r++) signal(ctr(prog, r, gate));
    if (threadIdx.x == 0) wait(prog, gate, prog.n_gpu);
    __syncthreads();
    reduce_range(prog, out, slot, 0, n, slice * blockDim.x + threadIdx.x, nblk * blockDim.x);
}

/* PLOW_DOP_XREDUCE2 (29), two-shot: t0=out i0=n i1=n_gpu i2=slot i3=gate_rs i4=gate_ag i5=e0.
 * Phase 1 reduces this rank's owned slice [n*rank/N, n*(rank+1)/N) in place into its own slot;
 * phase 2 gathers slice s from peer s. gate_ag counts EVERY workgroup of every rank (N * nblk):
 * the owned slice is written by all workgroups together, so none may speak for the others. */
__device__ __forceinline__ void d_xreduce_twoshot_nv(const PlowProgram& prog, __nv_bfloat16* out, uint32_t n, uint32_t slot,
                                                     uint32_t gate_rs, uint32_t gate_ag, unsigned slice, unsigned nblk) {
    using namespace plow_xc;
    const uint32_t N = prog.n_gpu, rank = prog.rank;
    const unsigned tid = slice * blockDim.x + threadIdx.x, stride = nblk * blockDim.x;
    if (slice == 0 && threadIdx.x == 0)
        for (uint32_t r = 0; r < N; r++) signal(ctr(prog, r, gate_rs));
    if (threadIdx.x == 0) wait(prog, gate_rs, N);
    __syncthreads();
    const uint32_t lo = (uint32_t)((uint64_t)n * rank / N), hi = (uint32_t)((uint64_t)n * (rank + 1) / N);
    reduce_range(prog, (__nv_bfloat16*)part(prog, rank, slot), slot, lo, hi, tid, stride);
    __syncthreads();
    if (threadIdx.x == 0) {
        /* Aggregated arrival: word 1 of the local gate line counts this rank's workgroups (its release
         * orders each one's phase-1 stores); the last one acquires them all and adds nblk to every
         * peer's word 0 -- N remote atomics per rank instead of N * nblk on one line. */
        uint32_t prev;
        asm volatile("atom.add.release.sys.global.u32 %0, [%1], 1;" : "=r"(prev) : "l"(ctr(prog, rank, gate_ag) + 1) : "memory");
        if (prev + 1u == nblk) {
            asm volatile("fence.acq_rel.sys;" ::: "memory");
            for (uint32_t r = 0; r < N; r++)
                asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(ctr(prog, r, gate_ag)), "r"(nblk) : "memory");
        }
        wait(prog, gate_ag, N * nblk);
    }
    __syncthreads();
    /* gather, starting each workgroup at a different peer so every link streams */
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t s = (i + slice + rank) % N;
        const uint32_t slo = (uint32_t)((uint64_t)n * s / N), shi = (uint32_t)((uint64_t)n * (s + 1) / N);
        const __nv_bfloat16* src = part(prog, s, slot);
        if ((slo % 8u) == 0 && (shi % 8u) == 0 && (slot % 16u) == 0 && ((uintptr_t)out % 16u) == 0) {
            for (uint32_t e = slo + tid * 8u; e < shi; e += stride * 8u)
                *reinterpret_cast<uint4*>(out + e) = __ldcg(reinterpret_cast<const uint4*>(src + e));
        } else {
            for (uint32_t e = slo + tid; e < shi; e += stride) out[e] = __ldcg(src + e);
        }
    }
}

/* PLOW_DOP_XARGMAX_FIN (28): t0=ids t1=part i0=nparts i1=n_batch i2=vocab_l i3=gate i4=first value line.
 * Folds this rank's block partials per sequence, rebases the winner to the global vocab id, publishes
 * (key | ~id) into its OWN value lines (16 u64 per 128 B counter line), then every rank takes the max
 * over all ranks' lines -- the same order everywhere, so all ranks write identical ids. */
#define PLOW_XAMAX_LINE 16u
#define PLOW_XAMAX_MAX_BATCH 128u
__device__ __forceinline__ void d_xargmax_fin_nv(const PlowProgram& prog, int* ids, const unsigned long long* part, uint32_t nparts,
                                                 uint32_t n_batch, uint32_t vocab_l, uint32_t gate, uint32_t val_id, unsigned slice) {
    using namespace plow_xc;
    if (slice != 0) return;
    const uint32_t B = n_batch ? n_batch : 1u, N = prog.n_gpu, rank = prog.rank;
    if (B > PLOW_XAMAX_MAX_BATCH) __trap();
    auto val_at = [&](uint32_t r, uint32_t b) {
        return reinterpret_cast<unsigned long long*>(ctr(prog, r, val_id + b / PLOW_XAMAX_LINE)) + b % PLOW_XAMAX_LINE;
    };
    for (uint32_t b = threadIdx.x; b < B; b += blockDim.x) {
        const unsigned long long* pb = part + (size_t)b * nparts;
        unsigned long long best = 0;
        for (uint32_t i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        const uint32_t gi = ~(uint32_t)(best & 0xFFFFFFFFu) + rank * vocab_l;
        *val_at(rank, b) = (best & 0xFFFFFFFF00000000ull) | (unsigned long long)(uint32_t)~gi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (uint32_t r = 0; r < N; r++) signal(ctr(prog, r, gate));
        wait(prog, gate, N);
    }
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < B; b += blockDim.x) {
        unsigned long long best = 0;
        for (uint32_t r = 0; r < N; r++) {
            const unsigned long long v = __ldcv(val_at(r, b));
            best = v > best ? v : best;
        }
        ids[b] = (int)~(uint32_t)(best & 0xFFFFFFFFull);
    }
}
