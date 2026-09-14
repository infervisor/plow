#include "golden.h"

static float bf_add(float a, float b) { return plow_bf2f(plow_f2bf(a + b)); }

static float affine_dot(const uint32_t* w, const plow_bf16* s, const plow_bf16* b,
                        const plow_bf16* x, uint32_t K) {
    float lanes[32] = {0};
    for (uint32_t lane = 0; lane < 32; lane++) {
        for (uint32_t k = lane * 16; k < K; k += 512) {
            float dot = 0, sum = 0;
            for (uint32_t j = 0; j < 16; j += 4) {
                float a0 = plow_bf2f(x[k + j]), a1 = plow_bf2f(x[k + j + 1]);
                float a2 = plow_bf2f(x[k + j + 2]), a3 = plow_bf2f(x[k + j + 3]);
                sum += bf_add(bf_add(bf_add(a0, a1), a2), a3);
                uint32_t q = w[(k + j) / 8] >> (((k + j) % 8) * 4);
                dot += a0 * (q & 15) + a1 * ((q >> 4) & 15)
                     + a2 * ((q >> 8) & 15) + a3 * ((q >> 12) & 15);
            }
            lanes[lane] += plow_bf2f(s[k / 64]) * dot + plow_bf2f(b[k / 64]) * sum;
        }
    }
    for (uint32_t width = 16; width; width /= 2)
        for (uint32_t lane = 0; lane < width; lane++) lanes[lane] += lanes[lane + width];
    return lanes[0];
}

static float affine_gemm_dot(const uint32_t* w, const plow_bf16* s, const plow_bf16* b,
                             const plow_bf16* x, uint32_t K) {
    float sum = 0;
    for (uint32_t k = 0; k < K; k++) {
        uint32_t q = (w[k / 8] >> ((k % 8) * 4)) & 15;
        float weight = plow_bf2f(plow_f2bf(plow_bf2f(s[k / 64]) * q + plow_bf2f(b[k / 64])));
        sum += plow_bf2f(x[k]) * weight;
    }
    return sum;
}

static void affine_q4(const PlowDevInst* in, uint32_t slice, uint32_t nblk,
                      void* const* T, int gemm) {
    uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    if (!nblk || slice >= nblk || !M || !N) return;
    plow_bf16* C = PLOW_CPU_TEN(in, T, 0);
    if (!C) return;
    C += (size_t)in->i[5] * N;
    const plow_bf16* A = PLOW_CPU_TEN(in, T, 1);
    const uint32_t* W = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* S = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* B = PLOW_CPU_TEN(in, T, 4);
    int valid = K && !(K % 64) && !in->i[3] && A && W && S && B;
    if (valid) A += (size_t)in->i[4] * K;
    uint32_t n0, n1;
    g_range(N, slice, nblk, &n0, &n1);
    size_t tiles_n = ((size_t)N + 63) / 64;
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            if (gemm ? (((size_t)(m / 64) * tiles_n + n / 64) % nblk != slice)
                     : (n < n0 || n >= n1)) continue;
            float value = NAN;
            if (valid) {
                const uint32_t* w = W + (size_t)n * (K / 8);
                const plow_bf16* s = S + (size_t)n * (K / 64);
                const plow_bf16* b = B + (size_t)n * (K / 64);
                const plow_bf16* x = A + (size_t)m * K;
                value = gemm ? affine_gemm_dot(w, s, b, x, K) : affine_dot(w, s, b, x, K);
            }
            C[(size_t)m * N + n] = plow_f2bf(value);
        }
}

G_K(g_gemv_affine_q4) { (void)ctx; affine_q4(in, slice, nblk, T, 0); }
G_K(g_gemm_affine_q4) { (void)ctx; affine_q4(in, slice, nblk, T, 1); }
