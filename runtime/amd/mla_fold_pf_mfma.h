// MLA prefill value fold as a batched MFMA GEMM (gfx950).
//
// With one split (prefill), MlaMergeFold is O[b,h,:] = (1/l[b,h]) * Opart[b,h,0:512] @ W_uv[h],
// a [T x 512] x [512 x V] GEMM per head. The scalar fold gives one workgroup one (row, head,
// V-tile) and re-streams a W_uv panel per row: 524,288 work items at T=8192 x 8 heads, 141 ms
// of a 1.17 s TP8 prefill on MI350X (1.5% of its HBM roof). Here a workgroup owns RB rows of one
// head: W_uv[h] is staged once per 32-row K chunk into LDS, transposed so B fragments are K
// contiguous, and the latent rows are normalized by 1/l and rounded to bf16 on the way into the
// MFMA — the same rounding point as the reference engine, whose attention output is bf16.
//
// Differs from the scalar fold only by that bf16 rounding of the normalized latent and f32
// reassociation; see the rel-L2 gate in runtime/tests/mla_fold_pf_mfma_gfx950.hip.
#pragma once

template <int WM, int V>
__device__ void d_mla_fold_pf_mfma(bf16* __restrict__ O, const float* __restrict__ Opart,
                                   const float* __restrict__ mlpart, const bf16* __restrict__ Wuv,
                                   unsigned n_batch, unsigned n_head, unsigned slice, unsigned nblk,
                                   bf16* lds /* V * (32 + 8) bf16 */) {
    constexpr int DK = 512, KC = 32, LROW = KC + 8;  // LDS row: 32 l + 8 pad (bank spread)
    constexpr int WN = PLOW_WAVES / WM, RB = 32 * WM;
    constexpr int NBLK = V / (32 * WN);
    static_assert(NBLK >= 1 && NBLK * 32 * WN == V, "V must tile the wave grid");
    static_assert(V == 256 && PLOW_THREADS == KC * 16, "staging map: 16 threads x 16 v per l row");
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave / WN, wn = wave % WN;
    const unsigned rblocks = (n_batch + RB - 1) / RB;
    const unsigned units = rblocks * n_head;

    for (unsigned u = slice; u < units; u += nblk) {
        const unsigned h = u % n_head, rb = u / n_head;
        const unsigned b = rb * RB + wm * 32u + mfma_frag_row(lane);
        const bool live = b < n_batch;
        float inv = 0.0f;
        if (live) {
            float gm;
            const float gl = fa_merge_ml(mlpart + ((size_t)b * n_head + h) * 2, 1u, gm);
            inv = gl > 0.0f ? FA_RECIP(gl) : 0.0f;
        }
        const float* arow = Opart + ((size_t)b * n_head + h) * DK;
        const bf16* w = Wuv + (size_t)h * DK * V;
        f32x16 acc[NBLK];
#pragma unroll
        for (int j = 0; j < NBLK; j++) acc[j] = (f32x16)(0.0f);

        for (unsigned kc = 0; kc < (unsigned)DK; kc += KC) {
            // Stage W_uv[h][kc:kc+32][0:V] transposed to lds[v][l]: thread -> (l = tid/16, 16 v).
            {
                const unsigned l = tid >> 4, v0 = (tid & 15u) * 16u;
                typedef bf16 bf16x16 __attribute__((ext_vector_type(16)));
                const bf16x16 q = *(const bf16x16*)(w + (size_t)(kc + l) * V + v0);
#pragma unroll
                for (int i = 0; i < 16; i++) lds[(v0 + i) * LROW + l] = q[i];
            }
            __syncthreads();
#pragma unroll
            for (unsigned kk = 0; kk < (unsigned)KC; kk += MFMA_K) {
                typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
                u16x8 ab;  // bf16 bit patterns; bf16x8 is a __bf16 vector (numeric assignment)
                {
                    const unsigned k = kc + mfma_frag_k(lane, kk);
                    float4 p0 = make_float4(0, 0, 0, 0), p1 = p0;
                    if (live) {
                        p0 = *(const float4*)(arow + k);
                        p1 = *(const float4*)(arow + k + 4);
                    }
                    ab[0] = f2bf(p0.x * inv); ab[1] = f2bf(p0.y * inv);
                    ab[2] = f2bf(p0.z * inv); ab[3] = f2bf(p0.w * inv);
                    ab[4] = f2bf(p1.x * inv); ab[5] = f2bf(p1.y * inv);
                    ab[6] = f2bf(p1.z * inv); ab[7] = f2bf(p1.w * inv);
                }
                const bf16x8 a = __builtin_bit_cast(bf16x8, ab);
#pragma unroll
                for (int j = 0; j < NBLK; j++) {
                    const unsigned n = wn * (32u * NBLK) + j * 32u + mfma_frag_row(lane);
                    bf16x8 bq;
                    __builtin_memcpy(&bq, lds + n * LROW + mfma_frag_k(lane, kk), 16);
                    acc[j] = plow_mfma_bf16_32x32(a, bq, acc[j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < NBLK; j++) {
            const unsigned n = wn * (32u * NBLK) + j * 32u + mfma_acc_n(lane);
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned r = rb * RB + wm * 32u + mfma_acc_m(lane, (unsigned)i);
                if (r < n_batch) st_act1(&O[((size_t)r * n_head + h) * V + n], f2bf(acc[j][i]));
            }
        }
    }
}
