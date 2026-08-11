#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_attention.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k_mla_decode_fp8(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                      const unsigned char* ckv, const bf16* krope, const int* kv_len,
                      unsigned n_head, unsigned kv_stride, unsigned nsplit, float scale,
                      const float* kv_scale) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 4)];
    d_flash_mla_decode<512, 64, 4, false, true>(
        opart, mlpart, qabs, qrope, (const bf16*)ckv, krope, kv_len, 1, n_head, kv_stride, 0,
        scale, nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr, 0, 1, kv_scale, 0);
}

extern "C" __global__ void k_cache_flush(const unsigned* in, unsigned* out, size_t n) {
    unsigned v = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v ^= in[i];
    if (v) atomicXor(out, v);
}
