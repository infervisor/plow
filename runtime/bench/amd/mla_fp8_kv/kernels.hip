#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#include "op_norm.h"
#endif

template <bool FP8>
__global__ __launch_bounds__(512, 1) void gather(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* lengths,
    void* idx, void* scales, unsigned batch, unsigned ctx, unsigned splits, float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 4)];
    d_flash_mla_decode<512, 64, 4, true, FP8>(
        (float*)out, (float*)ml, (bf16*)qa, (bf16*)qr, (bf16*)ck, (bf16*)kr,
        (int*)lengths, batch, 8, ctx, 0, scale, splits, 0xffffffffu,
        (blockIdx.x * 13u) % gridDim.x, gridDim.x, lds, (int*)idx, 2048, 1,
        (float*)scales, 0);
#endif
}

__global__ __launch_bounds__(512, 1) void write_rows(
    void* out, void* scales, void* x, void* gamma, void* pos,
    unsigned rows, unsigned ctx, unsigned mask) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope_fp8<512>(
        (unsigned char*)out, (float*)scales, (bf16*)x, (bf16*)gamma, nullptr, nullptr,
        (int*)pos, rows, 1, 1e-6f, 0, ctx, mask, 0, blockIdx.x, gridDim.x, rows);
#endif
}

extern "C" int plow_gather(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* lengths,
    void* idx, void* scales, unsigned batch, unsigned ctx, unsigned splits,
    unsigned fp8, float scale, void* stream) {
    if (fp8)
        gather<true><<<304, 512, 0, (hipStream_t)stream>>>(
            out, ml, qa, qr, ck, kr, lengths, idx, scales, batch, ctx, splits, scale);
    else
        gather<false><<<304, 512, 0, (hipStream_t)stream>>>(
            out, ml, qa, qr, ck, kr, lengths, idx, scales, batch, ctx, splits, scale);
    return (int)hipGetLastError();
}

extern "C" int plow_write(
    void* out, void* scales, void* x, void* gamma, void* pos,
    unsigned rows, unsigned ctx, unsigned mask, void* stream) {
    write_rows<<<2, 512, 0, (hipStream_t)stream>>>(out, scales, x, gamma, pos, rows, ctx, mask);
    return (int)hipGetLastError();
}

__global__ __launch_bounds__(512, 1) void make_union(
    void* uni, void* mask, void* idx, void* len, unsigned rows, unsigned ctx, unsigned cap) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned sc[513];
    d_index_union_pf((unsigned char*)uni, (unsigned long long*)mask, (int*)idx, (int*)len,
                     rows, 2048, ctx, cap, 8, blockIdx.x, gridDim.x, sc);
#endif
}

template <bool FP8>
__global__ __launch_bounds__(256, 1) void prefill(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* len,
    void* uni, void* scales, unsigned rows, unsigned ctx, unsigned cap, float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 lds[32768];
    d_flash_mla_prefill_v2<512, 64, true, FP8>(
        (float*)out, (float*)ml, (bf16*)qa, (bf16*)qr, (bf16*)ck, (bf16*)kr, (int*)len,
        1, rows, 8, ctx, 0, scale, 0xffffffffu, blockIdx.x, gridDim.x, lds,
        (unsigned char*)uni, cap, 1, false, (float*)scales, 0);
#endif
}

extern "C" int plow_union(void* uni, void* mask, void* idx, void* len,
                          unsigned rows, unsigned ctx, unsigned cap, void* stream) {
    unsigned blocks = (rows + 7) / 8;
    if (blocks > 304) blocks = 304;
    make_union<<<blocks, 512, 0, (hipStream_t)stream>>>(uni, mask, idx, len, rows, ctx, cap);
    return (int)hipGetLastError();
}

extern "C" int plow_prefill(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* len,
    void* uni, void* scales, unsigned rows, unsigned ctx, unsigned cap,
    unsigned fp8, float scale, void* stream) {
    if (fp8)
        prefill<true><<<304, 256, 0, (hipStream_t)stream>>>(
            out, ml, qa, qr, ck, kr, len, uni, scales, rows, ctx, cap, scale);
    else
        prefill<false><<<304, 256, 0, (hipStream_t)stream>>>(
            out, ml, qa, qr, ck, kr, len, uni, scales, rows, ctx, cap, scale);
    return (int)hipGetLastError();
}
