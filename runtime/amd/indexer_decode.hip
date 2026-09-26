#include "amd_common.h"

extern "C" __global__ __launch_bounds__(256) void glm_indexer_decode_prepare(
    uint8_t* q, float* scales, float* weights_scaled, uint8_t* cache,
    const bf16* query, const bf16* weights, const bf16* key,
    const int32_t* positions, const uint32_t* lengths,
    const uint32_t* parked, uint32_t* effective_lengths, unsigned rows, unsigned ctx) {
    unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    unsigned row = wave / 33, head = wave % 33, lane = threadIdx.x % 64;
    if (row >= rows) return;
    int32_t position = positions[row];
    uint32_t length = lengths[row];
    bool active = parked[row] == 0 && length > 0 && length <= ctx
        && position >= 0 && (uint32_t)position + 1 == length;
    bool is_key = head == 32;
    if (is_key && lane == 0) effective_lengths[row] = active ? length : 0;
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
        size_t slot = (size_t)row * ctx + (unsigned)position;
        size_t block = slot / 16;
        unsigned pos = slot % 16, d = lane * 2;
        size_t offset = block * 2112 + (d / 16) * 256 + pos * 16 + d % 16;
        *(uint16_t*)(cache + offset) = (uint16_t)quant;
        if (lane == 0) *(float*)(cache + block * 2112 + 2048 + pos * 4) = scale;
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

extern "C" __device__ unsigned plow_indexer_decode_abi_2 = 2;
