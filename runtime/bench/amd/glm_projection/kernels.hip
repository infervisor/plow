#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#endif

template <int KIND>
__global__ __launch_bounds__(512, 1) void projection(
    unsigned short* c, const unsigned short* a, const unsigned short* b,
    unsigned m, unsigned n, unsigned k) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr int BM = KIND == 0 ? GM_BM : KIND == 1 ? GM_WD_BM : KIND == 2 ? GM_MD_BM : GM_SM_BM;
    constexpr int BN = KIND == 0 ? GM_BN : KIND == 1 ? GM_WD_BN : KIND == 2 ? GM_MD_BN : GM_SM_BN;
    constexpr int BK = KIND == 0 ? GM_BK : KIND == 1 ? GM_WD_BK : KIND == 2 ? GM_MD_BK : GM_SM_BK;
    __shared__ bf16 lds[GM_LDS_HALVES_T(BM, BN, BK)];
    d_gemm_t<BM, BN, BK, GM_WM, GM_WN, false>(c, a, b, nullptr, nullptr,
        m, n, k, blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" int plow_projection(void* c, void* a, void* b, unsigned m,
    unsigned n, unsigned k, unsigned kind, void* stream) {
    const auto s = (hipStream_t)stream;
#define LAUNCH(K) projection<K><<<304, 512, 0, s>>>((unsigned short*)c, (unsigned short*)a, (unsigned short*)b, m, n, k)
    switch (kind) {
        case 0: LAUNCH(0); break;
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        default: return hipErrorInvalidValue;
    }
#undef LAUNCH
    return hipGetLastError();
}
