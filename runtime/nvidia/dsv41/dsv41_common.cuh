// DeepSeek-V4.1 on sm_90a: shared helpers for the per-op kernels the plowrt V4.1 engine launches.
//
// Numerics follow the checkpoint's reference (`inference/kernel.py`) bit for bit where the reference
// defines them: power-of-two ue8m0 activation scales, fp8/fp4 round-to-nearest casts after a clamp,
// and fp32 accumulation with a per-32-K-block rescale.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define DSV_EXTERN extern "C" __global__

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum; `red` holds one float per warp. Every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}
__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : -INFINITY;
    return warp_max(t);
}

// kernel.py `fast_log2_ceil` / `fast_pow2`: ceil(log2(x)) from the float's bits, then 2^n.
__device__ __forceinline__ int fast_log2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int)((b >> 23) & 0xffu);
    return e - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
}
__device__ __forceinline__ float fast_pow2(int n) { return __uint_as_float((uint32_t)(n + 127) << 23); }

// ue8m0: value 2^(e-127). A power-of-two scale encodes exactly.
__device__ __forceinline__ float e8m0_to_f(uint8_t e) { return __uint_as_float((uint32_t)e << 23); }
__device__ __forceinline__ uint8_t f_to_e8m0_pow2(float s) { return (uint8_t)((__float_as_uint(s) >> 23) & 0xffu); }

__device__ __forceinline__ float e4m3_to_f(uint8_t v) {
    __nv_fp8_e4m3 t;
    t.__x = v;
    return float(t);
}
__device__ __forceinline__ uint8_t f_to_e4m3(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
// e2m1 nibble -> float. Values: 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives.
__device__ __forceinline__ float e2m1_to_f(uint32_t nib) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[nib & 7u];
    return (nib & 8u) ? -m : m;
}
__device__ __forceinline__ float round_e2m1(float v) {
    __nv_fp4_storage_t q = __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
    return e2m1_to_f((uint32_t)q);
}
