#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif

extern "C" __global__ void normalize(float* out, const float* partial, const float* ml,
                                      unsigned rows, unsigned heads, unsigned splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned d = threadIdx.x + (blockIdx.x % 2) * 256;
    const unsigned row = blockIdx.x / 2;
    const unsigned head = row % heads, token = row / heads;
    const float* m = ml + (size_t)row * splits * 2;
    float gm;
    const float gl = fa_merge_ml(m, splits, gm);
    const float inv = gl > 0.0f ? FA_RECIP(gl) : 0.0f;
    float acc = 0.0f;
    for (unsigned s = 0; s < splits; ++s) {
        const float pv = partial[((size_t)row * splits + s) * 512 + d];
        acc += m[s * 2] == FA_NEG_INF ? 0.0f : pv * FA_EXP(m[s * 2] - gm);
    }
    out[((size_t)head * rows + token) * 512 + d] = acc * inv;
#endif
}

extern "C" __global__ void convert(unsigned short* out, const float* in,
                                    unsigned rows, unsigned heads) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned row = blockIdx.x, d = threadIdx.x;
    const unsigned head = row % heads, token = row / heads;
    out[(size_t)row * 256 + d] = f2bf(in[((size_t)head * rows + token) * 256 + d]);
#endif
}

extern "C" int plow_normalize(void* out, void* partial, void* ml,
                               unsigned rows, unsigned heads, unsigned splits, void* stream) {
    if (!rows || heads != 8 || !splits || splits >= 8) return hipErrorInvalidValue;
    normalize<<<rows * heads * 2, 256, 0, (hipStream_t)stream>>>((float*)out,
        (float*)partial, (float*)ml, rows, heads, splits);
    return hipGetLastError();
}
extern "C" int plow_convert(void* out, void* in, unsigned rows, unsigned heads, void* stream) {
    convert<<<rows * heads, 256, 0, (hipStream_t)stream>>>((unsigned short*)out,
        (float*)in, rows, heads);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256) void fold_mfma(unsigned short* out,
    const float* partial, const float* ml, const unsigned short* weight,
    unsigned rows, unsigned heads, unsigned splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float latent[16 * 512];
    const unsigned tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const unsigned h = blockIdx.y, t0 = blockIdx.x * 16, v0 = blockIdx.z * 64;
    const unsigned t = t0 + tid / 16;
    const float* m = ml + ((size_t)t * heads + h) * splits * 2;
    float gm = FA_NEG_INF, inv = 0;
    if (t < rows) {
        const float gl = fa_merge_ml(m, splits, gm);
        inv = gl > 0 ? FA_RECIP(gl) : 0;
    }
    for (unsigned d = tid % 16; d < 512; d += 16) {
        float value = 0;
        if (t < rows) {
            for (unsigned s = 0; s < splits; ++s) {
                float pv = partial[(((size_t)t * heads + h) * splits + s) * 512 + d];
                value += m[s * 2] == FA_NEG_INF ? 0.0f : pv * FA_EXP(m[s * 2] - gm);
            }
        }
        latent[(tid / 16) * 512 + (d ^ ((tid / 16) * 4))] = value * inv;
    }
    __syncthreads();
    typedef float f4 __attribute__((ext_vector_type(4)));
    f4 acc = {0, 0, 0, 0};
    const unsigned v = v0 + wave * 16 + lane % 16;
#pragma unroll 4
    for (unsigned kk = 0; kk < 512; kk += 4) {
        const float a = bf2f(weight[((size_t)h * 512 + kk + lane / 16) * 256 + v]);
        const float b = latent[(lane % 16) * 512 + ((kk + lane / 16) ^ ((lane % 16) * 4))];
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned row = t0 + lane % 16;
        const unsigned col = v0 + wave * 16 + (lane / 16) * 4 + i;
        if (row < rows) out[((size_t)row * heads + h) * 256 + col] = f2bf(acc[i]);
    }
#endif
}
extern "C" int plow_fold_mfma(void* out, void* partial, void* ml, void* weight,
    unsigned rows, unsigned heads, unsigned splits, void* stream) {
    if (!rows || heads != 8 || !splits || splits >= 8) return hipErrorInvalidValue;
    fold_mfma<<<dim3((rows + 15) / 16, heads, 4), 256, 0, (hipStream_t)stream>>>(
        (unsigned short*)out, (float*)partial, (float*)ml, (unsigned short*)weight,
        rows, heads, splits);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(512) void fold_bf16x2(unsigned short* out,
    const float* partial, const float* ml, const unsigned short* weight,
    unsigned rows, unsigned heads, unsigned splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned short high[16 * 512], low[16 * 512];
    const unsigned tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const unsigned h = blockIdx.y, t0 = blockIdx.x * 16, v0 = blockIdx.z * 128;
    const unsigned t = t0 + tid / 32;
    const float* m = ml + ((size_t)t * heads + h) * splits * 2;
    float gm = FA_NEG_INF, inv = 0;
    if (t < rows) {
        const float gl = fa_merge_ml(m, splits, gm);
        inv = gl > 0 ? FA_RECIP(gl) : 0;
    }
    for (unsigned d = tid % 32; d < 512; d += 32) {
        float value = 0;
        if (t < rows) {
            for (unsigned s = 0; s < splits; ++s) {
                float pv = partial[(((size_t)t * heads + h) * splits + s) * 512 + d];
                value += m[s * 2] == FA_NEG_INF ? 0.0f : pv * FA_EXP(m[s * 2] - gm);
            }
        }
        const float normalized = value * inv;
        const unsigned index = (tid / 32) * 512 + (d ^ ((tid / 32) * 16));
        const bf16 hi = f2bf(normalized);
        high[index] = hi;
        low[index] = f2bf(normalized - bf2f(hi));
    }
    __syncthreads();
    typedef float f4 __attribute__((ext_vector_type(4)));
    f4 acc = {0, 0, 0, 0};
    const unsigned v = v0 + wave * 16 + lane % 16;
    typedef short s4 __attribute__((ext_vector_type(4)));
#pragma unroll 4
    for (unsigned kk = 0; kk < 512; kk += 16) {
        s4 a, bh, bl;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned k = kk + (lane / 16) * 4 + j;
            a[j] = weight[((size_t)h * 512 + k) * 256 + v];
            const unsigned index = (lane % 16) * 512 + (k ^ ((lane % 16) * 16));
            bh[j] = high[index];
            bl[j] = low[index];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, bh, acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, bl, acc, 0, 0, 0);
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned row = t0 + lane % 16;
        const unsigned col = v0 + wave * 16 + (lane / 16) * 4 + i;
        if (row < rows) out[((size_t)row * heads + h) * 256 + col] = f2bf(acc[i]);
    }
#endif
}
extern "C" int plow_fold_bf16x2(void* out, void* partial, void* ml, void* weight,
    unsigned rows, unsigned heads, unsigned splits, void* stream) {
    if (!rows || heads != 8 || !splits || splits >= 8) return hipErrorInvalidValue;
    fold_bf16x2<<<dim3((rows + 15) / 16, heads, 2), 512, 0, (hipStream_t)stream>>>(
        (unsigned short*)out, (float*)partial, (float*)ml, (unsigned short*)weight,
        rows, heads, splits);
    return hipGetLastError();
}
