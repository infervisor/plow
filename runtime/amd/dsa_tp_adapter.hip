#define PLOW_WG_WAVES 8
#define PLOW_SEQ_PAR_SEAMS 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "dev_isa.h"
#include "op_attention.h"
#include "op_collective.h"
/* ABI 2: struct kernargs carrying an optional PlowKvSpan table. `spans == NULL` is the ABI-1
 * single-request form (kv_len[0] - rows positions). With a table, every row's position, causal
 * bound and key base come from its span; rows no span covers are skipped. plowrt loads either
 * marker and refuses to stage a packed sparse launch on an ABI-1 object. */
extern "C" __device__ unsigned plow_dsa_tp_abi_2 = 1;
#endif

struct DsaTpScoreArgs {
    float* score;
    const unsigned short* q;
    const unsigned short* k;
    const unsigned short* w;
    const int* len;
    const void* spans; /* PlowKvSpan[n_spans], or NULL (host pass cannot see dev_isa.h) */
    unsigned rows;
    unsigned stride;
    unsigned rank;
    unsigned n_spans;
};
static_assert(sizeof(DsaTpScoreArgs) == 64);

/* Per span, the ABI-1 body runs unchanged on the span's own row window: `len = &kv_len` and
 * `n_tok = row0 + n_rows` make its `q_pos0 = len - n_tok` equal `kv_row0 - row0` (mod 2^32),
 * so row t resolves to position kv_row0 + (t - row0); the key base is the span's `kv_base`.
 * Every argument is workgroup-uniform, so the body's barriers stay uniform. */
/* Slab width of the score pass, the occupancy knob `d_index_score_pf_row` documents: 64 keys is
 * 17,408 B of LDS (3 workgroups per CU), 128 keys is 34,816 B (1 workgroup). At the served
 * prefill shape the wider slab wins — half as many re-reads of the key rows per 8-query pack —
 * measured 1549 -> 1376 us (rank 0 band) and 1646 -> 1492 us (rank 7 band) for one layer of the
 * 8192-row chunk at 65k prior. -DPLOW_DSA_TP_TILE_N=64 rebuilds the previous object. */
#ifndef PLOW_DSA_TP_TILE_N
#define PLOW_DSA_TP_TILE_N 128u
#endif

extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_score(DsaTpScoreArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 tile[PLOW_DSA_TP_TILE_N * (128 + FA_PAD)];
    const unsigned band = (a.rows + 7u) / 8u;
    const unsigned lo = a.rank * band, hi = (a.rank + 1u) * band;
    if (!a.spans) {
        d_index_score_pf_row<128, 32, PLOW_DSA_TP_TILE_N>(a.score, a.q, a.k, a.w, a.len, a.rows, a.stride,
            0x1.fffffep-7f, blockIdx.x, gridDim.x, tile, lo, hi);
        return;
    }
    const PlowKvSpan* spans = (const PlowKvSpan*)a.spans;
    for (unsigned s = 0; s < a.n_spans; ++s) {
        const PlowKvSpan sp = spans[s];
        const unsigned rb = lo > sp.row0 ? lo : sp.row0;
        const unsigned re = sp.row0 + sp.n_rows;
        const unsigned rl = hi < re ? hi : re;
        if (rb >= rl) continue;
        d_index_score_pf_row<128, 32, PLOW_DSA_TP_TILE_N>(a.score, a.q, a.k + (size_t)sp.kv_base * 128u, a.w,
            (const int*)&spans[s].kv_len, re, a.stride, 0x1.fffffep-7f, blockIdx.x,
            gridDim.x, tile, rb, rl);
    }
#endif
}

struct DsaTpSelectArgs {
    int* idx;
    const float* score;
    const int* len;
    const void* const* peers;
    unsigned long long xctr_offset;
    unsigned* status;
    unsigned long long deadline;
    const void* spans; /* PlowKvSpan[n_spans], or NULL */
    unsigned rows;
    unsigned stride;
    unsigned rank;
    unsigned gate;
    unsigned n_spans;
    unsigned pad;
};
static_assert(sizeof(DsaTpSelectArgs) == 88);

extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_select(DsaTpSelectArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[SEL_NB], red[4];
    __shared__ int bailed;
    // All ranks must finish the previous collective before this slot can be overwritten.
    if (!xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate,
                              a.deadline, a.status, blockIdx.x, &bailed))
        return;
    const unsigned band = (a.rows + 7u) / 8u;
    const unsigned lo = a.rank * band, hi = (a.rank + 1u) * band;
    if (!a.spans) {
        d_index_select_pf<true>(a.idx, a.score, a.len, a.rows, 2048, a.stride,
            blockIdx.x, gridDim.x, hist, red, 1, lo, hi);
        return;
    }
    const PlowKvSpan* spans = (const PlowKvSpan*)a.spans;
    for (unsigned s = 0; s < a.n_spans; ++s) {
        const PlowKvSpan sp = spans[s];
        const unsigned rb = lo > sp.row0 ? lo : sp.row0;
        const unsigned re = sp.row0 + sp.n_rows;
        const unsigned rl = hi < re ? hi : re;
        if (rb >= rl) continue;
        d_index_select_pf<true>(a.idx, a.score, (const int*)&spans[s].kv_len, re, 2048,
            a.stride, blockIdx.x, gridDim.x, hist, red, 1, rb, rl);
    }
#endif
}

#if PLOW_DSA_TP_SELECT_THR
/* plow_dsa_tp_select with the threshold selection (d_index_select_pf_thr): same kernargs, rendezvous
 * and set. Built only with -DPLOW_DSA_TP_SELECT_THR=1; plowrt launches it under
 * PLOW_DSA_SELECT_THRESHOLD and refuses that knob on an object without it. */
extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_select_thr(DsaTpSelectArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hf[4096], hc[256], red[8];
    __shared__ unsigned long long cand[2048];
    __shared__ int bailed;
    if (!xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate,
                              a.deadline, a.status, blockIdx.x, &bailed))
        return;
    const unsigned band = (a.rows + 7u) / 8u;
    const unsigned lo = a.rank * band, hi = (a.rank + 1u) * band;
    if (!a.spans) {
        d_index_select_pf_thr<2048u>(a.idx, a.score, a.len, a.rows, 2048, a.stride,
            blockIdx.x, gridDim.x, hf, hc, cand, red, lo, hi);
        return;
    }
    const PlowKvSpan* spans = (const PlowKvSpan*)a.spans;
    for (unsigned s = 0; s < a.n_spans; ++s) {
        const PlowKvSpan sp = spans[s];
        const unsigned rb = lo > sp.row0 ? lo : sp.row0;
        const unsigned re = sp.row0 + sp.n_rows;
        const unsigned rl = hi < re ? hi : re;
        if (rb >= rl) continue;
        d_index_select_pf_thr<2048u>(a.idx, a.score, (const int*)&spans[s].kv_len, re, 2048,
            a.stride, blockIdx.x, gridDim.x, hf, hc, cand, red, rb, rl);
    }
#endif
}
#endif

struct DsaTpGatherArgs {
    uint4* out;
    const void* const* peers;
    unsigned long long xctr_offset;
    unsigned long long slot_offset;
    unsigned* status;
    unsigned long long deadline;
    unsigned rows;
    unsigned rank;
    unsigned gate;
    unsigned kv_bound; /* per-slot KV rows (the score stride); 0 = legacy host, no clamp */
};
static_assert(sizeof(DsaTpGatherArgs) == 64);

/* Every position that reaches `iidx_pf` passes through here, so this is where a bad one is
 * stopped. A position is a slot-relative KV row in [0, kv_bound) or the -1 pad; anything else
 * becomes -1. Positions are NOT guaranteed well-formed: the peer band is slot B of the TP scratch,
 * which the prefill FFN also writes (bf16 activations). If a select workgroup bails at its
 * rendezvous deadline it writes no band, the gather still meets and copies the bf16 bytes, and
 * read as i32 they are ~1e9. Unclamped, that was a write ~8-9 GiB past `iumask` in
 * `d_index_union_pf` ("Write access to a read-only page") and a wild read in the AITER sparse
 * flash ("Reason: Unknown"), both measured at 70k/C16. A bailed chunk still fails its audit;
 * this only makes it fail without touching memory it does not own.
 * `x + 1 <= bound` in unsigned arithmetic accepts exactly {-1} U [0, bound). */
#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ uint4 dsa_tp_clamp(uint4 v, unsigned bound) {
    if (!bound) return v;
    if (v.x + 1u > bound) v.x = 0xFFFFFFFFu;
    if (v.y + 1u > bound) v.y = 0xFFFFFFFFu;
    if (v.z + 1u > bound) v.z = 0xFFFFFFFFu;
    if (v.w + 1u > bound) v.w = 0xFFFFFFFFu;
    return v;
}
#endif

// Copies each peer's WHOLE row band [src*band, min((src+1)*band, rows)) from its slot: plain-form select wrote all of it after the entry gate; in the span form rows outside every span carry stale slot bytes that nothing reads (the packed flash walks spans; a body's band selects into n.iidx).
extern "C" __global__ void plow_dsa_tp_gather(DsaTpGatherArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ int bailed;
    if (!xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate,
                              a.deadline, a.status, blockIdx.x, &bailed))
        return;
    const unsigned band_vectors = ((a.rows + 7u) / 8u) * 512u;
    // Workgroup group w % 8 < 7 copies the band of peer rank + 1 + w % 8, group 7 the local
    // band, so all seven xGMI links carry a band at once; the serial loop below walked the peers
    // one after another with the whole grid on one link. One 16 B load per lane: holding four per
    // lane made the compiler promote them to an alloca (16400 B LDS, 48 B/lane scratch), and
    // IndexTp::load refuses a private segment.
    const unsigned lanes = (gridDim.x / 8u) * blockDim.x;
    if (lanes == 0u) {  // grid narrower than the eight peer groups; plowrt launches GATHER_GRID = 304
        for (unsigned step = 0; step < 8; ++step) {
            const unsigned src = (a.rank + step) & 7u;
            const unsigned lo = src * band_vectors;
            const unsigned hi = min(lo + band_vectors, a.rows * 512u);
            const auto* source = (const uint4*)((const char*)a.peers[src] + a.slot_offset);
            for (unsigned i = lo + blockIdx.x * blockDim.x + threadIdx.x;
                 i < hi; i += gridDim.x * blockDim.x)
                a.out[i] = dsa_tp_clamp(source[i], a.kv_bound);
        }
        return;
    }
    if (blockIdx.x >= (gridDim.x / 8u) * 8u) return;
    const unsigned group = blockIdx.x % 8u, sub = blockIdx.x / 8u;
    const unsigned src = group == 7u ? a.rank : ((a.rank + 1u + group) & 7u);
    const unsigned lo = src * band_vectors;
    const unsigned hi = min(lo + band_vectors, a.rows * 512u);
    const auto* source = (const uint4*)((const char*)a.peers[src] + a.slot_offset);
    for (unsigned i = lo + sub * blockDim.x + threadIdx.x; i < hi; i += lanes) a.out[i] = dsa_tp_clamp(source[i], a.kv_bound);
#endif
}

extern "C" __global__ void plow_dsa_tp_complete(DsaTpGatherArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    // A separate ordered launch ensures every gather workgroup finished reading peers.
    __shared__ int bailed;
    xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate + 1u,
                         a.deadline, a.status, 0, &bailed);
#endif
}

// Row-split sibling (local band): copies this rank's own band, which select wrote at its global
// rows of the rank's slot, into rows [0, band) of `out`. No peer reads, no rendezvous.
extern "C" __global__ void plow_dsa_tp_local(DsaTpGatherArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned band_vectors = ((a.rows + 7u) / 8u) * 512u;
    const auto* source =
        (const uint4*)((const char*)a.peers[a.rank] + a.slot_offset) + a.rank * band_vectors;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < band_vectors;
         i += gridDim.x * blockDim.x)
        a.out[i] = dsa_tp_clamp(source[i], a.kv_bound);
#endif
}
