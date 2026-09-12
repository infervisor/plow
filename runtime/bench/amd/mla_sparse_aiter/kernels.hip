#define PLOW_WG_WAVES 8
#define FA_LDS_DMA 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

__global__ __launch_bounds__(512, 1) void make_union(
    void* uni, void* mask, void* idx, void* len, unsigned rows, unsigned ctx, unsigned cap) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned sc[513];
    d_index_union_pf((unsigned char*)uni, (unsigned long long*)mask, (int*)idx, (int*)len,
                     rows, 2048, ctx, cap, 8, blockIdx.x, gridDim.x, sc);
#endif
}

__global__ __launch_bounds__(256, 1) void attend(
    void* out, void* ml, void* qa, void* qr, void* ck, void* kr, void* len, void* uni,
    unsigned rows, unsigned ctx, unsigned cap, float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __align__(16) bf16 lds[32768];
    // V2 has a fixed four-wave tile; union uses the interpreter's eight waves.
    d_flash_mla_prefill_v2<512, 64, true>(
        (float*)out, (float*)ml, (bf16*)qa, (bf16*)qr, (bf16*)ck, (bf16*)kr, (int*)len,
        1, rows, 8, ctx, 0, scale, 0xffffffffu, blockIdx.x, gridDim.x, lds,
        (unsigned char*)uni, cap);
#endif
}

extern "C" int plow_union(void* uni, void* mask, void* idx, void* len,
                          unsigned rows, unsigned ctx, unsigned cap, void* stream) {
    unsigned blocks = (rows + 7) / 8;
    if (blocks > 304) blocks = 304;
    make_union<<<blocks, 512, 0, (hipStream_t)stream>>>(uni, mask, idx, len, rows, ctx, cap);
    return (int)hipGetLastError();
}

extern "C" int plow_attend(void* out, void* ml, void* qa, void* qr, void* ck, void* kr,
                           void* len, void* uni, unsigned rows, unsigned ctx, unsigned cap,
                           float scale, void* stream) {
    attend<<<304, 256, 0, (hipStream_t)stream>>>(
        out, ml, qa, qr, ck, kr, len, uni, rows, ctx, cap, scale);
    return (int)hipGetLastError();
}

__global__ void pack_mla(uint4* out, const uint4* latent, const uint4* rope, unsigned rows) {
    const unsigned n = rows * 72u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const unsigned row = i / 72u, col = i % 72u;
        out[i] = col < 64u ? latent[row * 64u + col] : rope[row * 8u + col - 64u];
    }
}

extern "C" int plow_pack(void* out, void* latent, void* rope, unsigned rows, void* stream) {
    pack_mla<<<304, 256, 0, (hipStream_t)stream>>>(
        (uint4*)out, (const uint4*)latent, (const uint4*)rope, rows);
    return (int)hipGetLastError();
}
