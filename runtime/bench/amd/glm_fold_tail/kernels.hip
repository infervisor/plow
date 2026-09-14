#include <hip/hip_runtime.h>
#define PLOW_MLA_FOLD_TB 8
#ifndef FOLD_HEADER
#define FOLD_HEADER "op_attention.h"
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#include FOLD_HEADER
#endif

#ifndef FOLD_TAIL
#define FOLD_TAIL 0
#endif
extern "C" __global__ __launch_bounds__(512) void fold(
    unsigned short* out, const float* partial, const float* ml, const unsigned short* weight,
    unsigned rows, unsigned heads, unsigned splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[PLOW_MLA_FOLD_TB_LDS];
    if ((rows % 8 == 0 || FOLD_TAIL) && (rows / 8) * heads >= gridDim.x) {
        const unsigned full = rows / 8 * 8;
        d_mla_merge_fold_tb<512, 256, 8>(out, partial, ml, weight, full, heads,
                                        256, splits, blockIdx.x, gridDim.x, lds);
        if (rows != full) {
            const size_t offset = (size_t)full * heads;
            d_mla_merge_fold<512, 256>(out + offset * 256,
                partial + offset * splits * 512, ml + offset * splits * 2,
                weight, rows - full, heads, 256, splits, blockIdx.x, gridDim.x, lds);
        }
    } else
        d_mla_merge_fold<512, 256>(out, partial, ml, weight, rows, heads,
                                  256, splits, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" int plow_fold(void* out, void* partial, void* ml, void* weight,
                         unsigned rows, unsigned heads, unsigned splits, void* stream) {
    if (!rows || !heads || !splits || splits >= 8) return hipErrorInvalidValue;
    fold<<<304, 512, 0, (hipStream_t)stream>>>((unsigned short*)out, (float*)partial,
        (float*)ml, (unsigned short*)weight, rows, heads, splits);
    return hipGetLastError();
}
