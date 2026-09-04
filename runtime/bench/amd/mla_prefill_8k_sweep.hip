#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_attention.h"

extern "C" __global__ __launch_bounds__(512, 2) void k_mla_prefill_mfma(
    float* opart, float* mlpart, const bf16* qabs, const bf16* qrope, const bf16* ckv,
    const bf16* krope, const int* kv_len, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, float scale) {
    __shared__ float lds[FA_MLA_PF_LDS_FLOATS(512, 64)];
    d_flash_mla_prefill_mfma<512, 64, false>(
        opart, mlpart, qabs, qrope, ckv, krope, kv_len, 1, n_tok, n_head, kv_stride, 0,
        scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr, 0);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_mla_prefill_v2(
    float* opart, float* mlpart, const bf16* qabs, const bf16* qrope, const bf16* ckv,
    const bf16* krope, const int* kv_len, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, float scale) {
    __shared__ bf16 lds[FA_MLA_PF2_LDS_BYTES(512, 64) / 2];
    d_flash_mla_prefill_v2<512, 64>(
        opart, mlpart, qabs, qrope, ckv, krope, kv_len, 1, n_tok, n_head, kv_stride, 0,
        scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ void k_init_bf16(bf16* p, size_t n, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned x = (unsigned)i * 1664525u + seed * 1013904223u;
        p[i] = (bf16)(((x >> 31) << 15) | 0x3b00u | ((x >> 16) & 0xffu));
    }
}

extern "C" __global__ void k_cache_flush(const unsigned* in, unsigned* out, size_t n) {
    unsigned v = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v ^= in[i];
    if (v) atomicXor(out, v);
}
