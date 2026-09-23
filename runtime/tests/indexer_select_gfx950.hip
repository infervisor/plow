#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

extern "C" __global__ __launch_bounds__(512, 1) void plow_indexer_select_bounds(
    int* indices, const float* scores, const int* lengths, unsigned rows, unsigned stride) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned hist[SEL_NB], red[4];
    d_index_select_pf<true>(indices + (size_t)row * 2048, scores + (size_t)row * stride,
        lengths + row, 1, 2048, stride, 0, 1, hist, red);
#endif
}

extern "C" __device__ unsigned plow_indexer_select_test_abi_1 = 1;
