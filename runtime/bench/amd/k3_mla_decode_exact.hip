// Build: nix develop --command hipcc --genco --offload-arch=gfx950 -O3 -Werror -I runtime/amd -I runtime/common runtime/bench/amd/k3_mla_decode_exact.hip -o /tmp/k3_mla_decode_exact.co
#include "amd_common.h"
#include "op_attention.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_flash_bf16(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                      const bf16* ckv, const bf16* krope, const int* kv_len,
                      unsigned n_batch, unsigned kv_stride, unsigned nsplit) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 4)];
    d_flash_mla_decode<512, 64, 4>(opart, mlpart, qabs, qrope, ckv, krope, kv_len,
                                   n_batch, 12, kv_stride, 0, 0.07216878364870322f,
                                   nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_flash_bf16_gf6(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                          const bf16* ckv, const bf16* krope, const int* kv_len,
                          unsigned n_batch, unsigned kv_stride, unsigned nsplit) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 6)];
    d_flash_mla_decode<512, 64, 6>(opart, mlpart, qabs, qrope, ckv, krope, kv_len,
                                   n_batch, 12, kv_stride, 0, 0.07216878364870322f,
                                   nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_merge_bf16(bf16* olat, const float* opart, const float* mlpart,
                      unsigned n_batch, unsigned nsplit) {
    d_flash_merge<512>(olat, opart, mlpart, n_batch, 12, nsplit, blockIdx.x, gridDim.x);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_fold_bf16(bf16* out, const bf16* olat, const bf16* wuv, unsigned n_batch) {
    __shared__ bf16 lds[512];
    d_o_uv_fold<512>(out, olat, wuv, n_batch, 12, 128, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_merge_fold_v32(bf16* out, const float* opart, const float* mlpart,
                          const bf16* wuv, unsigned n_batch, unsigned nsplit) {
    __shared__ float lds[512 + PLOW_WAVES * 32];
    d_mla_merge_fold<512, 32>(out, opart, mlpart, wuv, n_batch, 12, 128, nsplit,
                              blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_merge_fold_v128(bf16* out, const float* opart, const float* mlpart,
                           const bf16* wuv, unsigned n_batch, unsigned nsplit) {
    __shared__ float lds[512 + PLOW_WAVES * 128];
    d_mla_merge_fold<512, 128>(out, opart, mlpart, wuv, n_batch, 12, 128, nsplit,
                               blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ void k_fill_bf16(bf16* dst, size_t count, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned x = (unsigned)(i * 1664525u + seed * 1013904223u);
        const unsigned mag = 0x3c00u + ((x >> 28) << 3); // small finite bf16 values
        reinterpret_cast<unsigned short*>(dst)[i] = (unsigned short)(mag | ((x >> 16) & 0x8000u));
    }
}
