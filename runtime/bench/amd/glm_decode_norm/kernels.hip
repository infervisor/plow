#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_norm.h"
#endif

__global__ __launch_bounds__(512) void norm_rows(
    unsigned short* out, unsigned short* resid, const unsigned short* a,
    const unsigned short* b, const unsigned short* gamma,
    unsigned rows, unsigned feat, unsigned add) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float part[PLOW_WAVES * 2];
    if (add)
        d_add_norm(out, resid, a, b, gamma, rows, feat, 1e-5f, blockIdx.x, gridDim.x, part);
    else
        d_rmsnorm(out, a, gamma, rows, feat, 1e-5f, 0, blockIdx.x, gridDim.x, part);
#endif
}

extern "C" int plow_norm_rows(void* out, void* resid, const void* a, const void* b,
                              const void* gamma, unsigned rows, unsigned feat,
                              unsigned groups, unsigned add, void* stream) {
    if (!rows || !feat || !groups || add > 1) return hipErrorInvalidValue;
    norm_rows<<<groups, 512, 0, (hipStream_t)stream>>>(
        (unsigned short*)out, (unsigned short*)resid, (const unsigned short*)a,
        (const unsigned short*)b, (const unsigned short*)gamma, rows, feat, add);
    return hipGetLastError();
}
