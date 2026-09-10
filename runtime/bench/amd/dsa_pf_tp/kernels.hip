#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

template <bool RANGED>
__global__ __launch_bounds__(512, 1) void score_pf(
    float* score, const unsigned short* q, const unsigned short* k,
    const unsigned short* w, const int* len, unsigned rows, unsigned stride,
    unsigned begin, unsigned end) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 tile[64 * (128 + FA_PAD)];
    // Match the emitter's two FP32 reciprocal-square-root factors exactly.
    constexpr float scale = 0x1.fffffep-7f;
    if constexpr (RANGED)
        d_index_score_pf_row<128, 32, 64>(score, q, k, w, len, rows, stride,
            scale, blockIdx.x, gridDim.x, tile, begin, end);
    else
        d_index_score_pf_row<128, 32, 64>(score, q, k, w, len, rows, stride,
            scale, blockIdx.x, gridDim.x, tile);
#endif
}

template <bool RANGED>
__global__ __launch_bounds__(512, 1) void select_pf(
    int* idx, const float* score, const int* len, unsigned rows,
    unsigned stride, unsigned begin, unsigned end) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[SEL_NB], red[4];
    if constexpr (RANGED)
        d_index_select_pf<true>(idx, score, len, rows, 2048, stride,
            blockIdx.x, gridDim.x, hist, red, 1, begin, end);
    else
        d_index_select_pf<true>(idx, score, len, rows, 2048, stride,
            blockIdx.x, gridDim.x, hist, red);
#endif
}

extern "C" int plow_index(void* idx, void* score, void* q, void* k, void* w,
                           void* len, unsigned rows, unsigned stride,
                           unsigned begin, unsigned end, unsigned ranged,
                           void* stream) {
    const auto s = (hipStream_t)stream;
    if (ranged) {
        score_pf<true><<<304, 512, 0, s>>>((float*)score, (unsigned short*)q,
            (unsigned short*)k, (unsigned short*)w, (int*)len, rows, stride, begin, end);
        select_pf<true><<<304, 512, 0, s>>>((int*)idx, (float*)score,
            (int*)len, rows, stride, begin, end);
    } else {
        score_pf<false><<<304, 512, 0, s>>>((float*)score, (unsigned short*)q,
            (unsigned short*)k, (unsigned short*)w, (int*)len, rows, stride, begin, end);
        select_pf<false><<<304, 512, 0, s>>>((int*)idx, (float*)score,
            (int*)len, rows, stride, begin, end);
    }
    return hipGetLastError();
}

__global__ void gather_idx(uint4* out, const unsigned long long* peers,
                           unsigned band_vectors, unsigned rank) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < band_vectors * 8; i += gridDim.x * blockDim.x) {
        const unsigned src = i / band_vectors;
        if (src != rank)
            out[i] = ((const uint4*)peers[src])[i];
    }
}

extern "C" int plow_gather(void* idx, void* peers, unsigned band_rows,
                            unsigned rank, void* stream) {
    gather_idx<<<304, 256, 0, (hipStream_t)stream>>>((uint4*)idx,
        (unsigned long long*)peers, band_rows * 512, rank);
    return hipGetLastError();
}

extern "C" int plow_enable_peers() {
    int original;
    hipError_t rc = hipGetDevice(&original);
    if (rc != hipSuccess) return rc;
    for (int rank = 0; rank < 8; ++rank) {
        rc = hipSetDevice(rank);
        if (rc != hipSuccess) return rc;
        for (int peer = 0; peer < 8; ++peer) {
            if (peer == rank) continue;
            rc = hipDeviceEnablePeerAccess(peer, 0);
            if (rc != hipSuccess && rc != hipErrorPeerAccessAlreadyEnabled) return rc;
        }
    }
    return hipSetDevice(original);
}

#include "dsa_tp_adapter.hip"

extern "C" int plow_index_protocol(void* idx, void* scratch, void* score,
    void* q, void* k, void* w, void* len, void* peers, void* status,
    unsigned rows, unsigned stride, unsigned rank, unsigned gate,
    unsigned long long counter_offset, void* stream) {
    const auto s = (hipStream_t)stream;
    plow_dsa_tp_score<<<304, 512, 0, s>>>((float*)score, (unsigned short*)q,
        (unsigned short*)k, (unsigned short*)w, (int*)len, rows, stride, rank);
    const DsaTpSelectArgs select = {(int*)scratch, (float*)score, (int*)len,
        (const void* const*)peers, counter_offset, (unsigned*)status, 1000000000ull,
        rows, stride, rank, gate};
    plow_dsa_tp_select<<<304, 512, 0, s>>>(select);
    const DsaTpGatherArgs a = {(uint4*)idx, (const void* const*)peers,
        counter_offset, 0, (unsigned*)status, 1000000000ull, rows, rank, gate + 1, 0};
    plow_dsa_tp_gather<<<304, 256, 0, s>>>(a);
    plow_dsa_tp_complete<<<1, 64, 0, s>>>(a);
    return hipGetLastError();
}
