/* tp_alltoall_kernels.hip — device code for the row-split sparse attention's two all-to-alls
 * (rowsplit-attention-design.md §3.1 steps 2 and 4), priced standalone (tp_alltoall_bench.c).
 *
 * Both ops are pull-form, one-workgroup rendezvous, per-peer slices — the same shape as
 * d_xall_gather_mega (runtime/amd/op_collective.h), reusing its primitives (xr_rendezvous_one_wg,
 * the xctr gate) and copying its AITER access pattern (xr_ag_sched's `wave_form`): ONE WAVE (64
 * lanes) PER PEER, walking that peer's slice linearly with 16-byte (8xbf16) loads/stores, instead
 * of a scalar per-element walk — the first cut of this file used st_act1/2-byte accesses and hit
 * only ~55 GB/s; ld_glob8/st_glob8 at one-wave-per-peer is what the production all-gather uses to
 * hit ~230 GB/s (seams-overlap.md §1), and every (peer, row) block here is 8 heads wide (a
 * multiple of 8 elements), so the vector width divides both sides evenly — no scalar remainder.
 *
 * Q all-to-all (d_xa2a_q): each rank's source is [T][nh_l][d] (its own nh_l heads, all T rows,
 * contiguous). Rank `rank` pulls, from every peer p, p's rows [rank*rpr, (rank+1)*rpr) — a
 * CONTIGUOUS run of rpr*nh_l*d elements in the source — and scatters it into the local
 * [rpr][nh_total][d] destination at head offset p*nh_l (contiguous per row, strided across rows).
 *
 * Attention-output all-to-all (d_xa2a_o): each rank's source is [rpr][nh_total][d] (its own row
 * band, all heads). Rank `rank` pulls, from every peer p, p's heads [rank*nh_l, (rank+1)*nh_l) at
 * every one of p's rpr rows — strided in the source (nh_l of nh_total per row) — into a
 * CONTIGUOUS run of the local [T][nh_l][d] destination at rows [p*rpr, (p+1)*rpr).
 *
 * Both are permutations (no arithmetic), so a copied element is byte-exact against the source
 * word; tp_alltoall_bench.c checks every destination element on every rank.
 *
 * tp_ag_control_bench calls d_xall_gather_mega directly (unmodified) at the seams-overlap.md §1
 * shape (8192x6144 bf16, full-width all-gather) as a control: if THIS harness cannot reproduce
 * that op's known ~230 GB/s, the fault is in the harness's peer/table setup, not in a new kernel.
 */
#include "../amd/op_collective.h"

extern "C" __device__ uint32_t tp_a2a_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
/* Same word(rank, e) the host oracle in tp_alltoall_bench.c computes. */
extern "C" __global__ void tp_a2a_fill(bf16* part, uint32_t n, uint32_t rank, uint32_t seed) {
    const uint32_t rs = tp_a2a_hash(rank * 0x9E3779B9u + seed);
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
        const uint32_t h = tp_a2a_hash(e ^ rs);
        part[e] = (bf16)(((h >> 31) << 15) | ((120u + ((h >> 20) & 15u)) << 7) | (h & 0x7fu));
    }
}

__device__ __forceinline__ uint64_t tp_a2a_now(void) { return __builtin_amdgcn_s_memrealtime(); }

/* Q all-to-all: rows_per_rank(rpr) x nh_l x d per peer moved, nh_total heads at the destination.
 * One wave (64 lanes) per peer p = (wave + rank) % nranks (the production stagger, so every
 * rank's wave 0 does not all hit peer 0 at once); `nblk` workgroups share each peer's range.
 * row_elems (= nh_l * d) is a multiple of 8 by construction, so every (peer, row) block divides
 * into whole 8xbf16 vectors on both the contiguous source and the per-row destination slice. */
__device__ __forceinline__ void d_xa2a_q(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xctr_byte_off,
    uint32_t gate, uint64_t deadline_ticks, uint32_t* status, unsigned slice, unsigned nblk,
    bf16* __restrict__ dst, uint32_t rpr, uint32_t nh_l, uint32_t d, uint32_t nh_total,
    uint32_t slot_bytes) {
    __shared__ int bailed;
    if (!xr_rendezvous_one_wg(peer_scratch, nranks, rank, xctr_byte_off, gate, deadline_ticks,
                              status, slice, &bailed))
        return;
    const uint32_t row_elems = nh_l * d, vrow = row_elems >> 3u;
    const uint32_t chunk_vecs = (rpr * row_elems) >> 3u, dst_row_vecs = (nh_total * d) >> 3u;
    const uint32_t lane = threadIdx.x & 63u, wave = threadIdx.x >> 6u;
    const uint32_t t0 = slice * 64u + lane, tstep = nblk * 64u;
    const uint32_t p = (wave + rank) % nranks;
    const bf16* src = (const bf16*)((const char*)peer_scratch[p] + slot_bytes) +
                      (size_t)rank * rpr * row_elems;
    const uint32_t dst_head_vecs = p * vrow;
    for (uint32_t v = t0; v < chunk_vecs; v += tstep) {
        const uint32_t row = v / vrow, r_in_row = v % vrow;
        st_glob8(dst + ((size_t)row * dst_row_vecs + dst_head_vecs + r_in_row) * 8u,
                ld_glob8(src + (size_t)v * 8u));
    }
}

/* Attention-output all-to-all: rpr x nh_l x d per peer moved, nh_total heads at the source. */
__device__ __forceinline__ void d_xa2a_o(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xctr_byte_off,
    uint32_t gate, uint64_t deadline_ticks, uint32_t* status, unsigned slice, unsigned nblk,
    bf16* __restrict__ dst, uint32_t rpr, uint32_t nh_l, uint32_t d, uint32_t nh_total,
    uint32_t slot_bytes) {
    __shared__ int bailed;
    if (!xr_rendezvous_one_wg(peer_scratch, nranks, rank, xctr_byte_off, gate, deadline_ticks,
                              status, slice, &bailed))
        return;
    const uint32_t row_elems = nh_l * d, vrow = row_elems >> 3u;
    const uint32_t chunk_vecs = (rpr * row_elems) >> 3u, src_row_vecs = (nh_total * d) >> 3u;
    const uint32_t lane = threadIdx.x & 63u, wave = threadIdx.x >> 6u;
    const uint32_t t0 = slice * 64u + lane, tstep = nblk * 64u;
    const uint32_t p = (wave + rank) % nranks;
    const bf16* src = (const bf16*)((const char*)peer_scratch[p] + slot_bytes);
    const uint32_t src_head_vecs = rank * vrow;
    bf16* dp = dst + (size_t)p * chunk_vecs * 8u;
    for (uint32_t v = t0; v < chunk_vecs; v += tstep) {
        const uint32_t row = v / vrow, r_in_row = v % vrow;
        st_glob8(dp + (size_t)v * 8u,
                ld_glob8(src + ((size_t)row * src_row_vecs + src_head_vecs + r_in_row) * 8u));
    }
}

/* TP_MODE=q | o: one all-to-all, `cap` workgroups (grid == cap), 2 timestamps/wg (start, end). */
extern "C" __global__ void tp_a2a_q_bench(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xoff, uint32_t gate,
    uint64_t deadline, uint32_t* status, bf16* dst, uint32_t rpr, uint32_t nh_l, uint32_t d,
    uint32_t nh_total, uint32_t slot_bytes, uint64_t* ts) {
    const uint64_t t0 = tp_a2a_now();
    d_xa2a_q(peer_scratch, nranks, rank, xoff, gate, deadline, status, blockIdx.x, gridDim.x, dst,
            rpr, nh_l, d, nh_total, slot_bytes);
    __syncthreads();
    if (threadIdx.x == 0 && ts) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tp_a2a_now(); }
}
extern "C" __global__ void tp_a2a_o_bench(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xoff, uint32_t gate,
    uint64_t deadline, uint32_t* status, bf16* dst, uint32_t rpr, uint32_t nh_l, uint32_t d,
    uint32_t nh_total, uint32_t slot_bytes, uint64_t* ts) {
    const uint64_t t0 = tp_a2a_now();
    d_xa2a_o(peer_scratch, nranks, rank, xoff, gate, deadline, status, blockIdx.x, gridDim.x, dst,
            rpr, nh_l, d, nh_total, slot_bytes);
    __syncthreads();
    if (threadIdx.x == 0 && ts) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tp_a2a_now(); }
}

/* TP_MODE=ag: control. The unmodified production all-gather (d_xall_gather_mega) at the
 * seams-overlap.md §1 shape, `cap` workgroups, 2 timestamps/wg. `-DPLOW_XR_SCHED_AITER=1` and no
 * `PLOW_XR_SCHED_NWG` (so PLOW_XR_SCHED_CAP is 0 and `cap` alone controls the width) is what
 * build.sh compiles this file with. */
extern "C" __global__ void tp_ag_control_bench(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xoff, uint32_t gate,
    uint64_t deadline, uint32_t* status, bf16* dst, uint32_t n, uint32_t slot_bytes,
    uint64_t* ts) {
    const uint64_t t0 = tp_a2a_now();
    d_xall_gather_mega(peer_scratch, nranks, rank, xoff, gate, deadline, status, blockIdx.x,
                       gridDim.x, dst, n, slot_bytes, nullptr, 0u, 0u, nullptr, 0u, 0u);
    __syncthreads();
    if (threadIdx.x == 0 && ts) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tp_a2a_now(); }
}

/* TP_MODE=both: the Q all-to-all then the O all-to-all, same `cap` workgroups doing both, back to
 * back in one launch (separate gates 1/2, each its own one-workgroup rendezvous) — the number the
 * design's per-layer +42 ms/chunk rests on. 3 timestamps/wg: start, after Q, after O. */
extern "C" __global__ void tp_a2a_both_bench(
    const void* const* peer_scratch, uint32_t nranks, uint32_t rank, size_t xoff,
    uint64_t deadline, uint32_t* status, bf16* dst_q, uint32_t rpr_q, uint32_t nh_l_q,
    uint32_t d_q, uint32_t nh_total_q, uint32_t slot_q, bf16* dst_o, uint32_t rpr_o,
    uint32_t nh_l_o, uint32_t d_o, uint32_t nh_total_o, uint32_t slot_o, uint64_t* ts) {
    const uint64_t t0 = tp_a2a_now();
    d_xa2a_q(peer_scratch, nranks, rank, xoff, 1u, deadline, status, blockIdx.x, gridDim.x, dst_q,
            rpr_q, nh_l_q, d_q, nh_total_q, slot_q);
    __syncthreads();
    const uint64_t t1 = tp_a2a_now();
    d_xa2a_o(peer_scratch, nranks, rank, xoff, 2u, deadline, status, blockIdx.x, gridDim.x, dst_o,
            rpr_o, nh_l_o, d_o, nh_total_o, slot_o);
    __syncthreads();
    const uint64_t t2 = tp_a2a_now();
    if (threadIdx.x == 0 && ts) {
        ts[3 * blockIdx.x] = t0; ts[3 * blockIdx.x + 1] = t1; ts[3 * blockIdx.x + 2] = t2;
    }
}
