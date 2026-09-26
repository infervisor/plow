// MLA FP8 batched GEMM (MlaBmmFp8: W_UK absorb, W_UV up-projection) for prefill rows (gfx950).
//
// d_mla_bmm_fp8_m16 gives one wave one 16x16 output tile and re-quantizes its A rows for every
// tile, so at prefill each activation 128-group is re-read and re-quantized N/16 times per head.
// Here a workgroup owns 32 rows of one head: phase 1 quantizes them once into LDS with the m16
// kernel's exact rule (per-128 amax/448, +-448 clamp, PLOW_GM_FP8_PACK2), phase 2 streams the
// head's K-contiguous W rows straight into 32x32x64 f8f6f4 MFMAs and folds each 128-group by its
// row scale, then applies the scalar weight scale once. Same output layout and rope copy.
// Accumulation order differs from the m16 body (K64 vs K128 MFMA), so equality is to f32 rounding.
#pragma once

template <int K, int N>
__device__ void d_mla_bmm_fp8_pf(bf16* __restrict__ C, const bf16* __restrict__ X,
                                 const unsigned char* __restrict__ W, const float* __restrict__ wscale,
                                 unsigned M, unsigned H, unsigned slice, unsigned nblk,
                                 unsigned x_head_stride, bf16* rope, unsigned rope_dim,
                                 unsigned char* lds) {
    constexpr int G = (K + 127) / 128, KP = G * 128, RB = 32;
    constexpr int NW = N / PLOW_WAVES, NB = NW / 32;
    static_assert(NB >= 1 && NB * 32 * PLOW_WAVES == N, "N must tile 8 waves x 32 columns");
    static_assert(PLOW_THREADS == 512, "quant map: 32 groups x 16 lanes per pass");
    unsigned char* aq = lds;                           // [RB][KP] e4m3
    float* as = (float*)(lds + RB * KP);               // [RB][G]
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned xs = x_head_stride ? x_head_stride : K;
    const unsigned units = (M + RB - 1) / RB * H;
    const float ws = *wscale;

    for (unsigned u = slice; u < units; u += nblk) {
        const unsigned h = u % H, r0 = (u / H) * RB;
        // Phase 1: quantize RB rows x G groups; 16 lanes x 8 elements per 128-group.
#pragma unroll
        for (int pass = 0; pass < G; pass++) {
            const unsigned item = pass * 32u + (tid >> 4), sub = tid & 15u;
            const unsigned r = item / G, g = item % G, k0 = g * 128u + sub * 8u;
            const unsigned m = r0 + r;
            float v[8];
            float amax = 1e-10f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                v[j] = (m < M && k0 + j < (unsigned)K) ? bf2f(X[((size_t)m * H + h) * xs + k0 + j]) : 0.0f;
                amax = fmaxf(amax, fabsf(v[j]));
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
            const float scale = amax * (1.0f / 448.0f), inv = 1.0f / scale;
            unsigned pk[2];
#pragma unroll
            for (int q = 0; q < 2; q++) {
                unsigned a = 0;
#pragma unroll
                for (int pair = 0; pair < 2; pair++) {
                    const float x0 = fminf(448.0f, fmaxf(-448.0f, v[q * 4 + pair * 2] * inv));
                    const float x1 = fminf(448.0f, fmaxf(-448.0f, v[q * 4 + pair * 2 + 1] * inv));
                    a |= PLOW_GM_FP8_PACK2(x0, x1) << (pair * 16);
                }
                pk[q] = a;
            }
            __builtin_memcpy(aq + r * KP + k0, pk, 8);
            if (sub == 0) as[r * G + g] = scale;
        }
        if (rope) {  // W_UK: carry the rope tail of each (row, head) through, as COPY_ROPE does
            for (unsigned j = tid; j < RB * rope_dim; j += PLOW_THREADS) {
                const unsigned m = r0 + j / rope_dim, col = j % rope_dim;
                if (m < M) st_act1(rope + ((size_t)m * H + h) * rope_dim + col,
                                   X[((size_t)m * H + h) * xs + K + col]);
            }
        }
        __syncthreads();

        // Phase 2: wave -> NW columns (NB 32-col blocks), rows 0..31 of the block.
        f32x16 acc[NB];
#pragma unroll
        for (int j = 0; j < NB; j++) acc[j] = (f32x16)(0.0f);
        const unsigned frow = lane % 32u, khalf = 32u * (lane / 32u);
#pragma unroll
        for (int g = 0; g < G; g++) {
            f32x16 t[NB];
#pragma unroll
            for (int j = 0; j < NB; j++) t[j] = (f32x16)(0.0f);
#pragma unroll
            for (int kk = 0; kk < 128; kk += 64) {
                if (g * 128 + kk >= K) break;
                const unsigned k = g * 128u + kk + khalf;
                fp8v32 a;
                __builtin_memcpy(&a, aq + frow * KP + k, 32);
#pragma unroll
                for (int j = 0; j < NB; j++) {
                    const unsigned n = wave * NW + j * 32u + frow;
                    fp8v32 b;
                    __builtin_memcpy(&b, W + ((size_t)h * N + n) * K + k, 32);
                    t[j] = plow_mfma_fp8_32x32(a, b, t[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < NB; j++)
#pragma unroll
                for (int e = 0; e < 16; e++)
                    acc[j][e] += t[j][e] * as[mfma_acc_m(lane, (unsigned)e) * G + g];
        }
#pragma unroll
        for (int j = 0; j < NB; j++) {
            const unsigned n = wave * NW + j * 32u + mfma_acc_n(lane);
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const unsigned m = r0 + mfma_acc_m(lane, (unsigned)e);
                if (m < M) st_act1(C + ((size_t)m * H + h) * N + n, f2bf(acc[j][e] * ws));
            }
        }
        __syncthreads();
    }
}
