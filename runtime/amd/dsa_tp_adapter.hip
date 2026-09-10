#define PLOW_WG_WAVES 8
#define PLOW_SEQ_PAR_SEAMS 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "dev_isa.h"
#include "op_attention.h"
#include "op_collective.h"
extern "C" __device__ unsigned plow_dsa_tp_abi_1 = 1;
#endif

extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_score(
    float* score, const unsigned short* q, const unsigned short* k,
    const unsigned short* w, const int* len, unsigned rows, unsigned stride,
    unsigned rank) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 tile[64 * (128 + FA_PAD)];
    const unsigned band = (rows + 7u) / 8u;
    d_index_score_pf_row<128, 32, 64>(score, q, k, w, len, rows, stride,
        0x1.fffffep-7f, blockIdx.x, gridDim.x, tile, rank * band, (rank + 1u) * band);
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
    unsigned rows;
    unsigned stride;
    unsigned rank;
    unsigned gate;
};
static_assert(sizeof(DsaTpSelectArgs) == 72);

extern "C" __global__ __launch_bounds__(512, 1) void plow_dsa_tp_select(DsaTpSelectArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[SEL_NB], red[4];
    __shared__ int bailed;
    // All ranks must finish the previous collective before this slot can be overwritten.
    if (!xr_rendezvous_one_wg(a.peers, 8, a.rank, a.xctr_offset, a.gate,
                              a.deadline, a.status, blockIdx.x, &bailed))
        return;
    const unsigned band = (a.rows + 7u) / 8u;
    d_index_select_pf<true>(a.idx, a.score, a.len, a.rows, 2048, a.stride,
        blockIdx.x, gridDim.x, hist, red, 1, a.rank * band, (a.rank + 1u) * band);
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
