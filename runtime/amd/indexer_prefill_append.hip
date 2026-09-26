#include "amd_common.h"

extern "C" __global__ __launch_bounds__(256) void glm_indexer_prefill_append(
    uint8_t* cache, const bf16* key, unsigned rows, unsigned base, unsigned live) {
    unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    unsigned lane = threadIdx.x % 64;
    if (row >= rows || row >= live) return;
    unsigned packed = ((const unsigned*)key)[(size_t)row * 64 + lane];
    float a = bf2f((bf16)packed), b = bf2f((bf16)(packed >> 16));
    float amax = wave_max(fmaxf(fabsf(a), fabsf(b)));
    float raw = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
    unsigned bits = __builtin_bit_cast(unsigned, raw);
    float scale = __builtin_bit_cast(float, (bits + 0x7fffffu) & 0xff800000u);
    unsigned quant = __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0, false);
    unsigned pos = base + row, d = lane * 2;
    size_t block = pos / 16;
    size_t offset = block * 2112 + (d / 16) * 256 + (pos % 16) * 16 + d % 16;
    *(uint16_t*)(cache + offset) = (uint16_t)quant;
    if (lane == 0) *(float*)(cache + block * 2112 + 2048 + (pos % 16) * 4) = scale;
}

extern "C" __device__ unsigned plow_indexer_prefill_append_abi_1 = 1;
