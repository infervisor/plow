/* op_act_quant_mx.h -- PLOW_DOP_ACT_QUANT_MX (op 200), shared by the CUDA and HIP interpreters.
 *
 * DeepSeek-V4.1's activation quant ahead of every block-fp8 GEMM (kernel.py `act_quant(x, 32,
 * "ue8m0")`), as a FAKE quant: each 32-element block of a bf16 row is scaled by
 * s = 2^ceil(log2(max(amax, 1e-4) / 448)), rounded to e4m3 (nearest-even, saturating at 448) and
 * scaled back, so out = e4m3(x / s) * s -- exact in bf16 (<= 4 significant bits times a power of
 * two). GemmFp8Mx then multiplies these bf16 values with the dequantized weights, which is the
 * reference's per-block fp8 product in another summation order.
 *
 * Portable by construction: bf16 is handled as raw bits and the e4m3 rounding is float math, so
 * the CUDA and HIP arms are the same code and agree bit for bit. One thread per 32-element block;
 * `out` may alias `x`.
 *
 * `scale` non-null: the real quant instead -- `out` is the e4m3 bytes [rows][k] and `scale` the ue8m0
 * exponents [rows][k / 32], act_quant's (y, s) as GemmFp8Mx i6 = 1 reads them.
 */
#ifndef PLOW_OP_ACT_QUANT_MX_H
#define PLOW_OP_ACT_QUANT_MX_H

#include <stdint.h>

static __device__ __forceinline__ float aqmx_bf16_to_f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
static __device__ __forceinline__ uint16_t aqmx_f_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); /* round to nearest even; inputs are finite */
}
/* the e4m3 value nearest y (ties to even), |y| <= 448 */
static __device__ __forceinline__ float aqmx_round_e4m3(float y) {
    const float a = fabsf(y);
    if (a == 0.f) return y;
    const int e = (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
    const float quantum = __uint_as_float((uint32_t)((e < -6 ? -6 : e) - 3 + 127) << 23);
    const float q = fminf(rintf(a / quantum) * quantum, 448.f);
    return y < 0.f ? -q : q;
}
/* e4m3 bits of an exactly representable q */
static __device__ __forceinline__ uint32_t aqmx_e4m3_bits(float q) {
    const uint32_t u = __float_as_uint(q);
    const float a = fabsf(q);
    const uint32_t mag = a < 0.015625f ? (uint32_t)(a * 512.f) : ((((u >> 23) & 0xffu) - 120u) << 3) | ((u >> 20) & 7u);
    return ((u >> 24) & 0x80u) | mag;
}
/* e4m3 bytes of (a, b), low byte = a: nearest-even, saturating at 448. sm_89+ in one instruction. */
static __device__ __forceinline__ uint32_t aqmx_e4m3x2(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
    return r;
#else
    return aqmx_e4m3_bits(aqmx_round_e4m3(fminf(fmaxf(a, -448.f), 448.f))) |
           (aqmx_e4m3_bits(aqmx_round_e4m3(fminf(fmaxf(b, -448.f), 448.f))) << 8);
#endif
}
/* bf16 bits of e4m3(a) * s and e4m3(b) * s (exact), low half = a */
static __device__ __forceinline__ uint32_t aqmx_fake2(float a, float b, float s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)aqmx_e4m3x2(a, b)));
    float fa, fb;
    asm("cvt.f32.f16 %0, %1;" : "=f"(fa) : "h"((uint16_t)(h2 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(fb) : "h"((uint16_t)(h2 >> 16)));
    return (__float_as_uint(fa * s) >> 16) | (__float_as_uint(fb * s) & 0xffff0000u);
#else
    return (uint32_t)aqmx_f_to_bf16(aqmx_round_e4m3(fminf(fmaxf(a, -448.f), 448.f)) * s) |
           ((uint32_t)aqmx_f_to_bf16(aqmx_round_e4m3(fminf(fmaxf(b, -448.f), 448.f)) * s) << 16);
#endif
}

static __device__ __forceinline__ void d_act_quant_mx(uint16_t* __restrict__ out, const uint16_t* __restrict__ x,
                                                       unsigned rows, unsigned k, unsigned slice, unsigned nblk,
                                                       uint8_t* __restrict__ scale = nullptr) {
    const unsigned kb = k >> 5;
    const unsigned long long n = (unsigned long long)rows * kb;
    for (unsigned long long g = (unsigned long long)slice * blockDim.x + threadIdx.x; g < n;
         g += (unsigned long long)nblk * blockDim.x) {
        const unsigned long long base = (g / kb) * k + (g % kb) * 32u;
        /* 64 contiguous bytes per thread as four 16 B loads (x and out are 16 B aligned: k % 32 == 0);
         * scalar u16 loads measured 8x off the bandwidth floor because `out` may alias `x`. */
        float v[32];
        float amax = 0.f;
        const uint4* xv = reinterpret_cast<const uint4*>(x + base);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const uint4 u = xv[q];
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                v[q * 8 + 2 * j] = aqmx_bf16_to_f((uint16_t)(w[j] & 0xffffu));
                v[q * 8 + 2 * j + 1] = aqmx_bf16_to_f((uint16_t)(w[j] >> 16));
            }
        }
#pragma unroll
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
        amax = fmaxf(amax, 1e-4f);
        const float t = amax * (1.0f / 448.0f);
        const uint32_t tb = __float_as_uint(t);
        const int ce = (int)((tb >> 23) & 0xffu) - 127 + ((tb & 0x7fffffu) != 0u ? 1 : 0);
        const float s = __uint_as_float((uint32_t)(ce + 127) << 23);
        const float rs = __uint_as_float((uint32_t)(127 - ce) << 23); /* 1 / s, exact: ce in [-22, 120] */
        if (scale) {
            uint4* ov8 = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + base);
#pragma unroll
            for (int q = 0; q < 2; q++) {
                unsigned w[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float* e = v + q * 16 + j * 4;
                    w[j] = aqmx_e4m3x2(e[0] * rs, e[1] * rs) | (aqmx_e4m3x2(e[2] * rs, e[3] * rs) << 16);
                }
                ov8[q] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            scale[g] = (uint8_t)(ce + 127);
            continue;
        }
        uint4* ov = reinterpret_cast<uint4*>(out + base);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            unsigned w[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                w[j] = aqmx_fake2(v[q * 8 + 2 * j] * rs, v[q * 8 + 2 * j + 1] * rs, s);
            }
            ov[q] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

#endif /* PLOW_OP_ACT_QUANT_MX_H */
