// DeepSeek-V4.1 sm_90a: norms and activation quantizers.
#include "dsv41_common.cuh"

// y[m] = bf16(w * (x[m] * rsqrt(mean(x[m]^2) + eps))), fp32 math. One block per row.
DSV_EXTERN void dsv_rmsnorm(bf16* __restrict__ y, const bf16* __restrict__ x, const bf16* __restrict__ w,
                            int D, long long x_stride, long long y_stride, float eps) {
    __shared__ float red[32];
    const int m = blockIdx.x;
    const bf16* xr = x + (long long)m * x_stride;
    bf16* yr = y + (long long)m * y_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = bf2f(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = f2bf(bf2f(w[i]) * (bf2f(xr[i]) * r));
}

// kernel.py act_quant(x, 32, "ue8m0"): per row, per 32-element group, s = 2^ceil(log2(max(amax,1e-4)/448)).
// q != null: writes e4m3 codes and the ue8m0 scale byte. fq != null: fake quant, bf16(fp8(x/s) * s).
// One warp per group; grid.x covers M * (K/32) groups, 8 warps per block.
DSV_EXTERN void dsv_act_quant_fp8(uint8_t* __restrict__ q, uint8_t* __restrict__ s, bf16* __restrict__ fq,
                                  const bf16* __restrict__ x, int M, int K, long long x_stride) {
    const int lane = threadIdx.x & 31;
    const long long g = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kg = K >> 5;
    if (g >= (long long)M * kg) return;
    const int m = (int)(g / kg), b = (int)(g % kg);
    const float v = bf2f(x[(long long)m * x_stride + b * 32 + lane]);
    float amax = warp_max(fabsf(v));
    amax = fmaxf(amax, 1e-4f);
    const float sc = fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
    const float c = fminf(fmaxf(v / sc, -448.f), 448.f);
    const uint8_t code = f_to_e4m3(c);
    if (q) {
        q[(long long)m * K + b * 32 + lane] = code;
        if (lane == 0) s[(long long)m * kg + b] = f_to_e8m0_pow2(sc);
    }
    if (fq) fq[(long long)m * x_stride + b * 32 + lane] = f2bf(e4m3_to_f(code) * sc);
}

// kernel.py fp4_act_quant(x, gs, inplace=True): fake quant to e2m1 in place.
// e4m3_scale = 0: ue8m0 pow2 scale (indexer, gs 32). 1: e4m3-rounded scale (compressed KV, gs 16).
// One (half-)warp group per 32 or 16 elements; each thread one element.
DSV_EXTERN void dsv_fp4_fakequant(bf16* __restrict__ x, long long n_elems, int gs, int e4m3_scale) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = i < n_elems;
    const float v = live ? bf2f(x[i]) : 0.f;
    float amax = fabsf(v);
    // reduce over groups of `gs` consecutive lanes (16 or 32)
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (gs == 32) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
    float sc;
    if (e4m3_scale) {
        amax = fmaxf(amax, 6.f * 0.001953125f);  // 6 * 2^-9
        sc = e4m3_to_f(f_to_e4m3(amax / 6.f));
    } else {
        amax = fmaxf(amax, 6.f * 1.1754943508222875e-38f);  // 6 * 2^-126
        sc = fast_pow2(fast_log2_ceil(amax * (1.0f / 6.0f)));
    }
    const float c = fminf(fmaxf(v / sc, -6.f), 6.f);
    if (live) x[i] = f2bf(round_e2m1(c) * sc);
}
