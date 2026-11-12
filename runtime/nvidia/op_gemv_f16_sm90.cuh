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
    if constexpr (BM == 16) asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b));
    else if constexpr (BM == 32) depth_mma_f16(d, a, b);
    else if constexpr (BM == 64) asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(a), "l"(b));
    else if constexpr (BM == 128) asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b));
#if (defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE) || (defined(PLOW_NV_GEMM_F16_NODE) && PLOW_NV_GEMM_F16_NODE)
    else if constexpr (BM == 256) asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a), "l"(b));
#endif
}

template <unsigned BM, bool HALF_OUT, unsigned BK = 64
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
          , bool OUTPUT_VIEW = false
#endif
          >
static __device__
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE == 2
__noinline__
#else
__forceinline__
#endif
void gemv_f16w_half(
    void* C, const unsigned short* __restrict__ x, const unsigned short* __restrict__ W,
    const float* bias, unsigned bgroup, const float* res, unsigned act,
    unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
    , unsigned view_rows = 0, unsigned view_stride = 0, unsigned view_offset = 0
#endif
    ) {
    static_assert(PLOW_NV_THREADS == 256, "f16 operand GEMM needs two warpgroups");
    constexpr unsigned BN = 64, NS = BK == 128 ? 2 : 3, STAGE = (BM + BN) * BK;
    static_assert(NS * STAGE * 2 + 1024 <= 102400, "f16 operand GEMM arena");
    auto* ring = (unsigned short*)sm90_align1024(arena);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned wg = warp >> 2, ww = warp & 3u, nt = N / BN;
    for (unsigned tile = slice; tile < ((M + BM - 1) / BM) * nt; tile += nblk) {
        const unsigned m0 = tile / nt * BM, n0 = tile % nt * BN;
        __syncthreads();
        auto stage = [&](unsigned p) {
            auto* ws = ring + (p % NS) * STAGE;
            auto* xs = ws + BN * BK;
            const unsigned kt = p * BK;
#pragma unroll
            for (unsigned i = threadIdx.x; i < BN * BK / 8u; i += PLOW_NV_THREADS) {
                const unsigned r = i / (BK / 8u), c = i % (BK / 8u);
                sm90_cp16(ws + c / 8u * BN * 64u + sm90_swz_off<64, 8>(r, c % 8u),
                          kt < K ? W + (size_t)(n0 + r) * K + kt + c * 8u : W, kt < K ? 16 : 0);
            }
#pragma unroll
            for (unsigned i = threadIdx.x; i < BM * BK / 8u; i += PLOW_NV_THREADS) {
                const unsigned r = i / (BK / 8u), c = i % (BK / 8u);
                const bool ok = m0 + r < M && kt < K;
                sm90_cp16(xs + c / 8u * BM * 64u + sm90_swz_off<64, 8>(r, c % 8u),
                          ok ? x + (size_t)(m0 + r) * K + kt + c * 8u : x, ok ? 16 : 0);
            }
            sm90_cp_commit();
        };
        for (unsigned p = 0; p < NS; p++) stage(p);
        float acc[BM / 4] = {};
        for (unsigned p = 0; p < K / BK; p++) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 1) : "memory");
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            auto* ws = ring + (p % NS) * STAGE;
            sm90_wg_fence();
#pragma unroll
            for (unsigned plane = 0; plane < BK / 64u; plane++) {
                auto* wpanel = ws + plane * BN * 64u;
                auto* xpanel = ws + BN * BK + plane * BM * 64u + wg * (BM / 2u) * 64u;
#pragma unroll
                for (unsigned k = 0; k < 64u; k += 16u)
                    gemv_f16w_mma_tile<BM>(acc, sm90_desc(wpanel + k), sm90_desc(xpanel + k));
            }
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
            const size_t o =
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
                OUTPUT_VIEW ? (size_t)(m / view_rows) * view_stride + (m % view_rows) * N + view_offset + n :
#endif
                (size_t)m * N + n;
            float v = acc[c];
            if (bias) v = __fadd_rn(v, bias[(bgroup ? m / bgroup * N : 0u) + n]);
            if (res) v = __fadd_rn(v, res[o]);
            if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
            if constexpr (HALF_OUT) ((__half*)C)[o] = __float2half_rn(v);
            else ((float*)C)[o] = v;
        }
    }
}

#if (defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE) || (defined(PLOW_NV_GEMM_F16_NODE) && PLOW_NV_GEMM_F16_NODE)
template <unsigned BM, bool HALF_OUT, unsigned BK = 64>
static __device__ __forceinline__ void gemv_f16w_wide(
    void* C, const unsigned short* __restrict__ x, const unsigned short* __restrict__ W,
    const float* bias, unsigned bgroup, const float* res, unsigned act,
    unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena) {
    static_assert(PLOW_NV_THREADS == 256, "f16 operand GEMM needs two warpgroups");
    constexpr unsigned BN = 128, NS = BK == 128 ? 2 : 3, STAGE = (BM + BN) * BK;
    static_assert(BM == 64 || BM == 128, "wide GEMM input rows");
    static_assert(BK == 64 || BK == 128, "wide GEMM reduction tile");
    static_assert(NS * STAGE * 2 + 1024 <= 102400, "f16 operand GEMM arena");
    auto* ring = (unsigned short*)sm90_align1024(arena);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned wg = warp >> 2, ww = warp & 3u, nt = N / BN;
    for (unsigned tile = slice; tile < ((M + BM - 1) / BM) * nt; tile += nblk) {
        const unsigned m0 = tile / nt * BM, n0 = tile % nt * BN;
        __syncthreads();
        auto stage = [&](unsigned p) {
            auto* ws = ring + (p % NS) * STAGE;
            auto* xs = ws + BN * BK;
            const unsigned kt = p * BK;
#pragma unroll
            for (unsigned i = threadIdx.x; i < BN * BK / 8u; i += PLOW_NV_THREADS) {
                const unsigned r = i / (BK / 8u), c = i % (BK / 8u);
                sm90_cp16(ws + c / 8u * BN * 64u + sm90_swz_off<64, 8>(r, c % 8u),
                          kt < K ? W + (size_t)(n0 + r) * K + kt + c * 8u : W, kt < K ? 16 : 0);
            }
#pragma unroll
            for (unsigned i = threadIdx.x; i < BM * BK / 8u; i += PLOW_NV_THREADS) {
                const unsigned r = i / (BK / 8u), c = i % (BK / 8u);
                const bool ok = m0 + r < M && kt < K;
                sm90_cp16(xs + c / 8u * BM * 64u + sm90_swz_off<64, 8>(r, c % 8u),
                          ok ? x + (size_t)(m0 + r) * K + kt + c * 8u : x, ok ? 16 : 0);
            }
            sm90_cp_commit();
        };
        for (unsigned p = 0; p < NS; p++) stage(p);
        float acc[BM / 2] = {};
        for (unsigned p = 0; p < K / BK; p++) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 1) : "memory");
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            auto* ws = ring + (p % NS) * STAGE;
            sm90_wg_fence();
#pragma unroll
            for (unsigned plane = 0; plane < BK / 64u; plane++) {
                auto* wpanel = ws + plane * BN * 64u + wg * 64u * 64u;
                auto* xpanel = ws + BN * BK + plane * BM * 64u;
#pragma unroll
                for (unsigned k = 0; k < 64u; k += 16u)
                    gemv_f16w_mma_tile<2 * BM>(acc, sm90_desc(wpanel + k), sm90_desc(xpanel + k));
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
            stage(p + NS);
        }
        sm90_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < BM / 2u; c++) {
            const unsigned n = n0 + wg * 64u + ww * 16u + lane / 4u + ((c % 4u) / 2u) * 8u;
            const unsigned m = m0 + (c / 4u) * 8u + 2u * (lane % 4u) + c % 2u;
            if (m >= M) continue;
            const size_t o = (size_t)m * N + n;
            float v = acc[c];
            if (bias) v = __fadd_rn(v, bias[(bgroup ? m / bgroup * N : 0u) + n]);
            if (res) v = __fadd_rn(v, res[o]);
            if (act == 1u) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
            if constexpr (HALF_OUT) ((__half*)C)[o] = __float2half_rn(v);
            else ((float*)C)[o] = v;
        }
    }
}
#endif

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
