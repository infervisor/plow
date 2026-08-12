#include "amd_common.h"
#include "op_attention.h"

template <int GF>
__device__ void run_mla_gf(float* opart, float* mlpart, const bf16* qabs,
                           const bf16* qrope, const unsigned char* ckv,
                           const bf16* krope, const int* kvlen, const float* kvscale,
                           unsigned kv_stride, unsigned nsplit, float* lds) {
    d_flash_mla_decode<512, 64, GF, false, true>(
        opart, mlpart, qabs, qrope, reinterpret_cast<const bf16*>(ckv), krope, kvlen,
        1, 12, kv_stride, 0, 0.08838834764831845f, nsplit, 0xffffffffu,
        blockIdx.x, gridDim.x, lds, nullptr, 0, 1, kvscale, 0);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_gf4_fp8(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                    const unsigned char* ckv, const bf16* krope, const int* kvlen,
                    const float* kvscale, unsigned kv_stride, unsigned nsplit) {
    __shared__ __align__(16) float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 4)];
    run_mla_gf<4>(opart, mlpart, qabs, qrope, ckv, krope, kvlen, kvscale, kv_stride,
                  nsplit, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_gf6_fp8(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                    const unsigned char* ckv, const bf16* krope, const int* kvlen,
                    const float* kvscale, unsigned kv_stride, unsigned nsplit) {
    __shared__ __align__(16) float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 6)];
    run_mla_gf<6>(opart, mlpart, qabs, qrope, ckv, krope, kvlen, kvscale, kv_stride,
                  nsplit, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_gf12_fp8(float* opart, float* mlpart, const bf16* qabs, const bf16* qrope,
                     const unsigned char* ckv, const bf16* krope, const int* kvlen,
                     const float* kvscale, unsigned kv_stride, unsigned nsplit) {
    __shared__ __align__(16) float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 12)];
    run_mla_gf<12>(opart, mlpart, qabs, qrope, ckv, krope, kvlen, kvscale, kv_stride,
                   nsplit, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void k_mla_merge(bf16* out, const float* opart, const float* mlpart, const bf16* wuv,
                 unsigned nsplit) {
    __shared__ __align__(16) float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 4)];
    d_mla_merge_fold<512, 64>(out, opart, mlpart, wuv, 1, 12, 128, nsplit,
                              blockIdx.x, gridDim.x, lds);
}
