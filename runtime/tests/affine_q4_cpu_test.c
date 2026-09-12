#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "golden/golden.h"

static void check(uint32_t M, uint32_t N, uint32_t K, uint32_t blocks, int gemm) {
    plow_bf16* x = calloc((M + 1) * K, 2);
    uint32_t* w = calloc(N * K / 8, 4);
    plow_bf16* s = calloc(N * K / 64, 2);
    plow_bf16* b = calloc(N * K / 64, 2);
    plow_bf16* y = malloc((M + 2) * N * 2);
    unsigned* writes = calloc(M * N, sizeof(unsigned));
    assert(x && w && s && b && y && writes);
    for (uint32_t m = 0; m <= M; m++)
        for (uint32_t k = 0; k < K; k++) x[m * K + k] = plow_f2bf(m ? (float)m / 8 : 42);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < K; k++) w[n * K / 8 + k / 8] |= (k % 16) << ((k % 8) * 4);
        for (uint32_t g = 0; g < K / 64; g++) {
            s[n * K / 64 + g] = plow_f2bf(n % 2 ? -0.25f : 0.5f);
            b[n * K / 64 + g] = plow_f2bf(g % 2 ? -1.0f : 0.5f);
        }
    }
    PlowDevInst in = {0};
    for (unsigned i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    for (unsigned i = 0; i < 5; i++) in.t[i] = i;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[4] = 1; in.i[5] = 1;
    void* tab[] = {y, x, w, s, b};
    for (uint32_t slice = 0; slice < blocks; slice++) {
        for (uint32_t i = 0; i < (M + 2) * N; i++) y[i] = 0x5a5a;
        (gemm ? g_gemm_affine_q4 : g_gemv_affine_q4)(&in, slice, blocks, tab, NULL);
        for (uint32_t i = 0; i < N; i++) assert(y[i] == 0x5a5a && y[(M + 1) * N + i] == 0x5a5a);
        for (uint32_t m = 0; m < M; m++) for (uint32_t n = 0; n < N; n++) {
            if (y[(m + 1) * N + n] == 0x5a5a) continue;
            writes[m * N + n]++;
            float total = 0;
            for (uint32_t k = 0; k < K; k++)
                total += ((k % 16) * plow_bf2f(s[n * K / 64 + k / 64])
                    + plow_bf2f(b[n * K / 64 + k / 64])) * ((float)(m + 1) / 8);
            assert(y[(m + 1) * N + n] == plow_f2bf(total));
        }
    }
    for (uint32_t i = 0; i < M * N; i++) assert(writes[i] == 1);
    in.t[4] = PLOW_TENSOR_NONE;
    (gemm ? g_gemm_affine_q4 : g_gemv_affine_q4)(&in, 0, 1, tab, NULL);
    for (uint32_t i = N; i < (M + 1) * N; i++) assert(isnan(plow_bf2f(y[i])));
    in.t[4] = 4; in.i[2] = K - 1;
    (gemm ? g_gemm_affine_q4 : g_gemv_affine_q4)(&in, 0, 1, tab, NULL);
    for (uint32_t i = N; i < (M + 1) * N; i++) assert(isnan(plow_bf2f(y[i])));
    free(x); free(w); free(s); free(b); free(y); free(writes);
}

int main(void) {
    unsigned checks = 0;
    const uint32_t shapes[][3] = {{1, 1, 64}, {2, 7, 128}, {65, 65, 64}, {2, 129, 512}, {2, 9, 576}};
    const uint32_t blocks[] = {1, 3, 64};
    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) for (unsigned j = 0; j < 3; j++)
        for (int gemm = 0; gemm < 2; gemm++, checks++) check(shapes[i][0], shapes[i][1], shapes[i][2], blocks[j], gemm);
    plow_bf16 x[64] = {0}, y[1], s[1] = {0x3f80}, b[1] = {0x3f80};
    uint32_t w[8] = {0};
    x[0] = plow_f2bf(1); x[1] = x[2] = plow_f2bf(1.0f / 256); x[3] = plow_f2bf(-1);
    void* tab[] = {y, x, w, s, b};
    PlowDevInst in = {0};
    for (unsigned i = 0; i < 5; i++) in.t[i] = i;
    in.i[0] = in.i[1] = 1; in.i[2] = 64;
    g_gemv_affine_q4(&in, 0, 1, tab, NULL); assert(y[0] == 0);
    g_gemm_affine_q4(&in, 0, 1, tab, NULL); assert(y[0] == plow_f2bf(1.0f / 128));
    printf("checks=%u guards=1 ownership=1 invalid_shapes=1 missing_bias=1 rounding_contract=1\n", checks);
    return 0;
}
