#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

#if PLOW_MLA_PF_PSWZ
namespace pswz_on {
#else
namespace pswz_off {
#endif

template <bool GATHER, bool FP8>
__global__ __launch_bounds__(256, 1) void attention(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* len,
    void* uni, void* scales, unsigned rows, unsigned ctx, unsigned cap,
    unsigned splits, unsigned window, unsigned rope8, float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 lds[32768];
    d_flash_mla_prefill_v2<512, 64, GATHER, FP8>(
        (float*)out, (float*)ml, (bf16*)qa, (bf16*)qr, (bf16*)ck, (bf16*)kr, (int*)len,
        1, rows, 8, ctx, window, scale, 0xffffffffu, blockIdx.x, gridDim.x, lds,
        (unsigned char*)uni, cap, splits, false, (float*)scales, rope8);
#endif
}

__global__ __launch_bounds__(512, 1) void make_union(
    void* uni, void* mask, void* idx, void* len, unsigned rows, unsigned ctx, unsigned cap) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned sc[513];
    d_index_union_pf((unsigned char*)uni, (unsigned long long*)mask, (int*)idx, (int*)len,
                     rows, 2048, ctx, cap, 8, blockIdx.x, gridDim.x, sc);
#endif
}
}

#if PLOW_MLA_PF_PSWZ
namespace impl = pswz_on;
#else
namespace impl = pswz_off;
#endif

extern "C" int plow_union(void* uni, void* mask, void* idx, void* len,
                          unsigned rows, unsigned ctx, unsigned cap, void* stream) {
    unsigned blocks = (rows + 7) / 8;
    if (blocks > 304) blocks = 304;
    impl::make_union<<<blocks, 512, 0, (hipStream_t)stream>>>(uni, mask, idx, len, rows, ctx, cap);
    return (int)hipGetLastError();
}

extern "C" int plow_attention(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* len,
    void* uni, void* scales, unsigned rows, unsigned ctx, unsigned cap,
    unsigned splits, unsigned window, unsigned rope8, unsigned mode, float scale, void* stream) {
#define RUN(G, F) impl::attention<G, F><<<304, 256, 0, (hipStream_t)stream>>>( \
    out, ml, qa, qr, ck, kr, len, uni, scales, rows, ctx, cap, splits, window, rope8, scale)
    switch (mode) {
    case 0: RUN(false, false); break;
    case 1: RUN(false, true); break;
    case 2: RUN(true, false); break;
    case 3: RUN(true, true); break;
    default: return (int)hipErrorInvalidValue;
    }
#undef RUN
    return (int)hipGetLastError();
}
