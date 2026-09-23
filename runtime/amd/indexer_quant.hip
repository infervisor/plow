#include "amd_common.h"

__device__ __forceinline__ float indexer_scale(float raw) {
    unsigned bits = __builtin_bit_cast(unsigned, raw);
    return __builtin_bit_cast(float, (bits + 0x7fffffu) & 0xff800000u);
}

extern "C" __global__ __launch_bounds__(256) void glm_indexer_query_quant(
    uint8_t* q, float* scales, float* weights_scaled, const bf16* x,
    const bf16* weights, unsigned rows) {
    unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    unsigned lane = threadIdx.x % 64;
    if (row >= rows) return;
    size_t pair = (size_t)row * 64 + lane;
    unsigned packed = ((const unsigned*)x)[pair];
    float a = bf2f((bf16)packed), b = bf2f((bf16)(packed >> 16));
    // The installed contiguous-input custom op floors scale, not amax.
    float scale = indexer_scale(fmaxf(wave_max(fmaxf(fabsf(a), fabsf(b))) * (1.0f / 448.0f), 1e-10f));
    unsigned quant = __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0, false);
    ((uint16_t*)q)[pair] = (uint16_t)quant;
    if (lane == 0) {
        scales[row] = scale;
        float w = bf2f(weights[row]) * scale;
        w *= 0.08838834764831845f;
        weights_scaled[row] = w * 0.1767766952966369f;
    }
}

extern "C" __global__ __launch_bounds__(256) void glm_indexer_key_cache(
    uint8_t* cache, const bf16* x, const int64_t* slots, unsigned rows) {
    unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    unsigned lane = threadIdx.x % 64;
    if (row >= rows) return;
    int64_t slot = slots[row];
    if (slot < 0) return;
    unsigned packed = ((const unsigned*)x)[(size_t)row * 64 + lane];
    float a = bf2f((bf16)packed), b = bf2f((bf16)(packed >> 16));
    float scale = indexer_scale(fmaxf(wave_max(fmaxf(fabsf(a), fabsf(b))), 1e-4f) * (1.0f / 448.0f));
    unsigned quant = __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0, false);
    size_t block = (uint64_t)slot / 16;
    unsigned pos = slot % 16, d = lane * 2;
    size_t offset = block * 2112 + (d / 16) * 256 + pos * 16 + d % 16;
    *(uint16_t*)(cache + offset) = (uint16_t)quant;
    if (lane == 0) *(float*)(cache + block * 2112 + 2048 + pos * 4) = scale;
}

extern "C" __device__ unsigned plow_indexer_quant_abi_1 = 1;
