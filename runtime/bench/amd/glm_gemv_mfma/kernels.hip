#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#endif

template <int MM, int ARM>
__global__ __launch_bounds__(512) void projection(
    unsigned short* c, const unsigned short* x, const unsigned short* w,
    unsigned m, unsigned n, unsigned k, unsigned reps) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[GM_LDS_HALVES];
    for (unsigned rep = 0; rep < reps; ++rep) {
        const bf16* weight = w + (size_t)rep * n * k;
        if constexpr (ARM == 2) {
            d_gemv_t<MM>(c, x, weight, nullptr, nullptr, m, n, k, 0, 0.0f,
                         blockIdx.x, gridDim.x, lds);
            __syncthreads();
            continue;
        }
        const bool staged = (size_t)m * k <= GM_LDS_HALVES;
        if (staged) {
            stage_x_lds(lds, x, (size_t)m * k);
            __syncthreads();
        }
#define BODY(XLDS)                                                                              \
        if constexpr (ARM == 1)                                                                \
            gemv_rows_mfma4<MM, XLDS, gv_mfma4_un(MM), gv_mfma4_yt(MM)>(                          \
                c, x, weight, m, n, k, blockIdx.x, gridDim.x, lds);                              \
        else                                                                                   \
            gemv_rows<MM, XLDS, false, gv_unroll_for(MM)>(                                       \
                c, x, weight, nullptr, nullptr, m, n, k, blockIdx.x, gridDim.x, lds)
        if (staged) { BODY(true); } else { BODY(false); }
#undef BODY
        __syncthreads();
    }
#endif
}

extern "C" int plow_glm_gemv(void* c, void* x, void* w, unsigned m, unsigned n,
                            unsigned k, unsigned mm, unsigned arm, unsigned groups,
                            unsigned reps, void* stream) {
    if (m == 0 || m > mm || n == 0 || k == 0 || groups == 0 || reps == 0)
        return hipErrorInvalidValue;
    const auto s = (hipStream_t)stream;
#define LAUNCH(MM, MF) projection<MM, MF><<<groups, 512, 0, s>>>(                                 \
    (unsigned short*)c, (unsigned short*)x, (unsigned short*)w, m, n, k, reps)
#define RUNG(MM) case MM:                                                                       \
    if (arm == 2) { LAUNCH(MM, 2); }                                                           \
    else if (arm == 1) { LAUNCH(MM, 1); }                                                      \
    else { LAUNCH(MM, 0); }                                                                   \
    break
    switch (mm) {
        RUNG(2);
        RUNG(4);
        RUNG(8);
        default: return hipErrorInvalidValue;
    }
#undef RUNG
#undef LAUNCH
    return hipGetLastError();
}
