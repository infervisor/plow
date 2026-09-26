#include "amd_common.h"

extern "C" __global__ __launch_bounds__(256) void glm_indexer_prefill_prepare(
    uint8_t* q, float* scales, float* weights_scaled, uint8_t* cache,
    uint8_t* keys, float* key_scales, uint32_t* starts, uint32_t* ends,
    const bf16* query, const bf16* weights, const bf16* key,
    unsigned rows, unsigned base, unsigned live) {
    unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    unsigned lane = threadIdx.x % 64;
    if (wave >= rows * 33) {
        unsigned pos = wave - rows * 33;
        if (pos >= base) return;
        size_t block = pos / 16;
        unsigned d = lane * 2;
        size_t offset = block * 2112 + (d / 16) * 256 + (pos % 16) * 16 + d % 16;
        ((uint16_t*)keys)[(size_t)pos * 64 + lane] = *(const uint16_t*)(cache + offset);
        if (lane == 0) key_scales[pos] = *(const float*)(cache + block * 2112 + 2048 + (pos % 16) * 4);
        return;
    }
    unsigned row = wave / 33, head = wave % 33;
    bool active = row < live, is_key = head == 32;
    if (is_key && lane == 0) {
        starts[row] = 0;
        ends[row] = active ? base + row + 1 : 0;
    }
    if (is_key && !active) return;
    size_t index = is_key ? row : (size_t)row * 32 + head;
    unsigned packed = active ? ((const unsigned*)(is_key ? key : query))[index * 64 + lane] : 0;
    float a = bf2f((bf16)packed), b = bf2f((bf16)(packed >> 16));
    float amax = wave_max(fmaxf(fabsf(a), fabsf(b)));
    float raw = is_key ? fmaxf(amax, 1e-4f) * (1.0f / 448.0f)
                       : fmaxf(amax * (1.0f / 448.0f), 1e-10f);
    unsigned bits = __builtin_bit_cast(unsigned, raw);
    float scale = __builtin_bit_cast(float, (bits + 0x7fffffu) & 0xff800000u);
    unsigned quant = __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0, false);
    if (is_key) {
        unsigned pos = base + row, d = lane * 2;
        size_t block = pos / 16;
        size_t offset = block * 2112 + (d / 16) * 256 + (pos % 16) * 16 + d % 16;
        *(uint16_t*)(cache + offset) = (uint16_t)quant;
        ((uint16_t*)keys)[(size_t)pos * 64 + lane] = (uint16_t)quant;
        if (lane == 0) {
            *(float*)(cache + block * 2112 + 2048 + (pos % 16) * 4) = scale;
            key_scales[pos] = scale;
        }
    } else {
        ((uint16_t*)q)[index * 64 + lane] = (uint16_t)quant;
        if (lane == 0) {
            scales[index] = scale;
            float w = active ? bf2f(weights[index]) * scale : 0.0f;
            w *= 0.08838834764831845f;
            weights_scaled[index] = w * 0.1767766952966369f;
        }
    }
}

extern "C" __device__ unsigned plow_indexer_prefill_abi_1 = 1;
