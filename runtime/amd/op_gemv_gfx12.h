/* Native wave64 WMMA decode. Included only by the gfx12 GEMV dispatch. */
#pragma once

#if PLOW_GFX12 && PLOW_GFX12_WMMA_GEMV
typedef __bf16 gv12_b4 __attribute__((ext_vector_type(4)));
typedef unsigned gv12_u2 __attribute__((ext_vector_type(2)));
typedef unsigned gv12_u4 __attribute__((ext_vector_type(4)));

/* A b128 load covers eight adjacent K values. Repartition four lane groups into
 * the WMMA wave64 layout: lane%16 is the column, lane/16 owns four K values.
 * The same map applies to x (A, transposed) and W (B). */
template <int HALF>
__device__ __forceinline__ gv12_b4 gv12_fragment(bf16v8 v, unsigned lane) {
    const gv12_u4 p = __builtin_bit_cast(gv12_u4, v);
    const unsigned src = (lane & 15u) + ((lane >> 5) + HALF * 2u) * 16u;
    gv12_u2 q;
    const unsigned a = __shfl(p[0], src), b = __shfl(p[1], src);
    const unsigned c = __shfl(p[2], src), d = __shfl(p[3], src);
    q[0] = (lane & 16u) ? c : a;
    q[1] = (lane & 16u) ? d : b;
    return __builtin_bit_cast(gv12_b4, q);
}

__device__ __forceinline__ bool gv12_fits(unsigned M, unsigned K, int norm) {
    return GV_BLOCKED && norm == 0 && M >= 1 && M <= 16 && K >= 32 && (K & 31u) == 0;
}

/* Every wave visits the same N tile and a disjoint, increasing sequence of K
 * chunks. The reduction order is independent of M and the object's GEMV bucket,
 * including the M=1 calls made by GEMV_ARGMAX. Scratch aliases the dead GEMM arena. */
template <bool GLU>
__device__ void gemv_gfx12_range(bf16* C, const bf16* x, const bf16* W, const bf16* Wu,
                                unsigned M, unsigned N, unsigned K,
                                unsigned n0, unsigned n1, bf16* lds,
                                unsigned act = 0, float beta = 0, float lbeta = 0) {
    static_assert(PLOW_WAVE == 64, "gfx12 GEMV requires wave64");
    static_assert((GLU ? 2 : 1) * PLOW_WAVES * 256 * sizeof(float) <= GM_LDS_HALVES * sizeof(bf16),
                  "gfx12 GEMV reduction must fit the interpreter arena");
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned col = lane & 15u, kg = lane >> 4;
    float* part = (float*)lds;
    const auto xr = buf_rsrc(x, M * K);
    for (unsigned n = n0; n < n1; n += 16) {
        const unsigned cols = min(16u, n1 - n);
        /* Descriptors must be wave-uniform: a lane-varying row base makes the
         * compiler serialize the loads through a descriptor waterfall. */
        const auto wr = buf_rsrc(W + (size_t)n * K, cols * K);
        const auto ur = buf_rsrc((GLU ? Wu : W) + (size_t)n * K, cols * K);
        f32x4 acc = {}, up = {};
        for (unsigned k0 = wave * 32; k0 < K; k0 += PLOW_WAVES * 32 * 4) {
            bf16v8 wv[4], xv[4], uv[4];
#pragma unroll
            for (unsigned u = 0; u < 4; u++) {
                const unsigned k = k0 + u * PLOW_WAVES * 32 + kg * 8;
                const unsigned wo = col < cols && k < K ? (col * K + k) * 2u : cols * K * 2u;
                const unsigned xo = col < M && k < K ? (col * K + k) * 2u : M * K * 2u;
                wv[u] = buf_ld8(wr, wo);
                if constexpr (GLU) uv[u] = buf_ld8(ur, wo);
                /* Activations are reused by every N tile; keep them cacheable. */
                xv[u] = __builtin_bit_cast(bf16v8,
                    __builtin_amdgcn_raw_buffer_load_b128(xr, xo, 0, 0));
            }
#pragma unroll
            for (unsigned u = 0; u < 4; u++) {
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(
                    gv12_fragment<0>(xv[u], lane), gv12_fragment<0>(wv[u], lane), acc);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(
                    gv12_fragment<1>(xv[u], lane), gv12_fragment<1>(wv[u], lane), acc);
                if constexpr (GLU) {
                    up = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(
                        gv12_fragment<0>(xv[u], lane), gv12_fragment<0>(uv[u], lane), up);
                    up = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(
                        gv12_fragment<1>(xv[u], lane), gv12_fragment<1>(uv[u], lane), up);
                }
            }
        }
#pragma unroll
        for (unsigned e = 0; e < 4; e++) {
            // Native wave64 accumulator groups order the lane bits as 0, 8, 4, 12.
            const unsigned row = (kg & 1u) * 8 + (kg >> 1) * 4 + e;
            part[wave * 256 + row * 16 + col] = acc[e];
            if constexpr (GLU) part[(PLOW_WAVES + wave) * 256 + row * 16 + col] = up[e];
        }
        __syncthreads();
        for (unsigned i = threadIdx.x; i < M * 16; i += PLOW_THREADS) {
            float sum = part[i];
#pragma unroll
            for (unsigned w = 1; w < PLOW_WAVES; w++) sum += part[w * 256 + i];
            if constexpr (GLU) {
                float usum = part[PLOW_WAVES * 256 + i];
#pragma unroll
                for (unsigned w = 1; w < PLOW_WAVES; w++) usum += part[(PLOW_WAVES + w) * 256 + i];
                sum = act == 2u ? k3_situ_gate(sum, beta) * k3_situ_up(usum, lbeta)
                                : act_gate_only(sum, act) * usum;
            }
            if (n + i % 16 < n1) st_act1(as_glob(C) + (size_t)(i / 16) * N + n + i % 16, f2bf(sum));
        }
        __syncthreads();
    }
}
__device__ void d_gemv_gfx12(bf16* C, const bf16* x, const bf16* W,
                            unsigned M, unsigned N, unsigned K,
                            unsigned slice, unsigned nblk, bf16* lds) {
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per, n1 = min(n0 + per, N);
    gemv_gfx12_range<false>(C, x, W, nullptr, M, N, K, n0, n1, lds);
}
#endif
