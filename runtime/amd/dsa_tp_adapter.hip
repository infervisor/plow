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
extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_score(DsaTpScoreArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 tile[64 * (128 + FA_PAD)];
    const unsigned band = (a.rows + 7u) / 8u;
    const unsigned lo = a.rank * band, hi = (a.rank + 1u) * band;
    if (!a.spans) {
        d_index_score_pf_row<128, 32, 64>(a.score, a.q, a.k, a.w, a.len, a.rows, a.stride,
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
        d_index_score_pf_row<128, 32, 64>(a.score, a.q, a.k + (size_t)sp.kv_base * 128u, a.w,
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
    unsigned pad;
};
static_assert(sizeof(DsaTpGatherArgs) == 64);

extern "C" __global__ void plow_dsa_tp_gather(DsaTpGatherArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ int bailed;
    if (!xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate,
                              a.deadline, a.status, blockIdx.x, &bailed))
        return;
    const unsigned band_vectors = ((a.rows + 7u) / 8u) * 512u;
    for (unsigned step = 0; step < 8; ++step) {
        const unsigned src = (a.rank + step) & 7u;
        const unsigned lo = src * band_vectors;
        const unsigned hi = min(lo + band_vectors, a.rows * 512u);
        const auto* source = (const uint4*)((const char*)a.peers[src] + a.slot_offset);
        for (unsigned i = lo + blockIdx.x * blockDim.x + threadIdx.x;
             i < hi; i += gridDim.x * blockDim.x)
            a.out[i] = source[i];
    }
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
