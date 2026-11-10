#pragma once
#include "sm90_wgmma.cuh"

__device__ __forceinline__ void depth_mma_f16(float* d, uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a), "l"(b));
}

template <unsigned BM>
__device__ __forceinline__ void gemv_f16w_mma_tile(float* d, uint64_t a, uint64_t b) {
    if constexpr (BM == 32) depth_mma_f16(d, a, b);
    else asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(a), "l"(b));
}

template <unsigned BM>
static __device__ __forceinline__ void gemv_f16w_tiled(
    float* C, const float* x, const unsigned short* W, const float* bias, unsigned bgroup,
    const float* res, const float* gamma, const float* beta, float eps, unsigned act,
    unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena) {
    constexpr unsigned BN = 64, BK = 64, NS = 3;
    constexpr unsigned STAGE = (2 * BM + BN) * BK;
    static_assert(NS * STAGE * 2 + BM * BK * 2 + 1024 + (BM * 2 + (BM == 64 ? 2048 : 8192)) * 4 <= 98304, "f16 GEMM arena");
    auto* ring = (unsigned short*)sm90_align1024(arena);
    auto* xhalf = ring + NS * STAGE;
    float* stats = (float*)(xhalf + BM * BK);
    float* gs = stats + 2 * BM;
    float* bs = gs + K;
    if (gamma) {
        for (unsigned k = threadIdx.x * 4u; k < K; k += PLOW_NV_THREADS * 4u) {
            sm90_cp16(gs + k, gamma + k, 16);
            sm90_cp16(bs + k, beta + k, 16);
        }
        sm90_cp_commit();
        sm90_cp_wait<0>();
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned wg = warp >> 2, ww = warp & 3u;
    const unsigned nt = N / BN;
    for (unsigned tile = slice; tile < ((M + BM - 1) / BM) * nt; tile += nblk) {
        const unsigned m0 = tile / nt * BM, n0 = tile % nt * BN;
        __syncthreads();
        if (gamma) {
            for (unsigned r = warp; r < BM && m0 + r < M; r += 8u) {
                const float* xr = x + (size_t)(m0 + r) * K;
                float s = 0.f;
                for (unsigned k = lane; k < K; k += 32u) s += xr[k];
                const float mean = warp_sum32(s) / (float)K;
                float v = 0.f;
                for (unsigned k = lane; k < K; k += 32u) {
                    const float d = xr[k] - mean;
                    v = __fmaf_rn(d, d, v);
                }
                const float var = warp_sum32(v) / (float)K;
                if (!lane) { stats[2 * r] = mean; stats[2 * r + 1] = rsqrtf(var + eps); }
            }
        }
        __syncthreads();
        auto stage = [&](unsigned p) {
            auto* ws = ring + (p % NS) * STAGE;
            auto* xf = (float*)(ws + BN * BK);
            const unsigned kt = p * BK;
#pragma unroll
            for (unsigned i = threadIdx.x; i < BN * BK / 8u; i += PLOW_NV_THREADS) {
                const unsigned r = i / 8u, c = i % 8u;
                sm90_cp16(ws + sm90_swz_off<BK, 8>(r, c),
                          W + (size_t)(n0 + r) * K + kt + c * 8u, kt < K ? 16 : 0);
            }
#pragma unroll
            for (unsigned i = threadIdx.x; i < BM * BK / 4u; i += PLOW_NV_THREADS) {
                const unsigned r = i / 16u, k = (i % 16u) * 4u;
                const bool ok = m0 + r < M && kt < K;
                sm90_cp16(xf + r * BK + k, ok ? x + (size_t)(m0 + r) * K + kt + k : x,
                          ok ? 16 : 0);
            }
            sm90_cp_commit();
        };
        const unsigned nk = K / BK;
        for (unsigned p = 0; p < NS; p++) stage(p);
        float acc[BM / 4] = {};
        for (unsigned p = 0; p < nk; p++) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 1) : "memory");
            auto* ws = ring + (p % NS) * STAGE;
            const auto* xf = (const float*)(ws + BN * BK);
#pragma unroll
            for (unsigned i = threadIdx.x; i < BM * BK / 4u; i += PLOW_NV_THREADS) {
                const unsigned r = i / 16u, k = (i % 16u) * 4u;
                float4 v = *(const float4*)(xf + r * BK + k);
                if (gamma && m0 + r < M) {
                    const float4 g = *(const float4*)(gs + p * BK + k);
                    const float4 b = *(const float4*)(bs + p * BK + k);
                    const float mu = stats[2 * r], rs = stats[2 * r + 1];
                    v.x = __fmaf_rn((v.x - mu) * rs, g.x, b.x);
                    v.y = __fmaf_rn((v.y - mu) * rs, g.y, b.y);
                    v.z = __fmaf_rn((v.z - mu) * rs, g.z, b.z);
                    v.w = __fmaf_rn((v.w - mu) * rs, g.w, b.w);
                }
                auto* d = (__half2*)(xhalf + sm90_swz_off<BK, 8>(r, k / 8u) + k % 8u);
                d[0] = __floats2half2_rn(v.x, v.y);
                d[1] = __floats2half2_rn(v.z, v.w);
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            auto* xs = xhalf + wg * (BM / 2u) * BK;
            sm90_wg_fence();
#pragma unroll
            for (unsigned k = 0; k < BK; k += 16u)
                gemv_f16w_mma_tile<BM>(acc, sm90_desc(ws + k), sm90_desc(xs + k));
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
            stage(p + NS);
        }
        sm90_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < BM / 4u; c++) {
            const unsigned n = n0 + ww * 16u + lane / 4u + ((c % 4u) / 2u) * 8u;
            const unsigned m = m0 + wg * (BM / 2u) + (c / 4u) * 8u + 2u * (lane % 4u) + c % 2u;
            if (m >= M) continue;
            const size_t o = (size_t)m * N + n;
            float v = acc[c];
            if (bias) v = __fadd_rn(v, bias[(bgroup ? m / bgroup * N : 0u) + n]);
            if (res) v = __fadd_rn(v, res[o]);
            if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
            C[o] = v;
        }
    }
}
