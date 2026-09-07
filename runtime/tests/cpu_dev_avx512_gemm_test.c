#define _POSIX_C_SOURCE 200809L
#include "cpu_dev_internal.h"
#include "golden/golden.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int check(uint16_t op, uint32_t M, uint32_t N, uint32_t K, int bench) {
    plow_cpu_kernel_fn gold[PLOW_CPU_DOP_TABLE] = {0};
    plow_cpu_register_golden(gold);
    const int norm = op == PLOW_DOP_GEMM_NORM, glu = op == PLOW_DOP_GEMM_GLU;
    const uint32_t offset = norm || glu ? 0 : 2;
    const size_t nc = (size_t)(M + offset) * N;
    plow_bf16* C = calloc(nc, 2), *ref = calloc(nc, 2);
    plow_bf16* A = malloc((size_t)(M + offset) * K * 2);
    plow_bf16* B = malloc((size_t)N * K * 2), *U = malloc((size_t)N * K * 2);
    plow_bf16* gamma = malloc(K * 2), *bias = malloc(N * 2);
    float* rms = malloc(M * sizeof(float));
    uint32_t state = 7;
    for (size_t i = 0; i < (size_t)(M + offset) * K; i++) {
        state = state * 1664525u + 1013904223u;
        A[i] = plow_f2bf(((int)(state >> 24) - 128) / 128.0f);
    }
    for (size_t i = 0; i < (size_t)N * K; i++) {
        state = state * 1664525u + 1013904223u;
        B[i] = plow_f2bf(((int)(state >> 24) - 128) / 1024.0f);
        U[i] = plow_f2bf(((int)(state >> 16 & 255) - 128) / 1024.0f);
    }
    for (uint32_t i = 0; i < K; i++) gamma[i] = plow_f2bf(0.75f + (i % 7) / 16.0f);
    for (uint32_t i = 0; i < N; i++) bias[i] = plow_f2bf((i % 9) / 128.0f);
    for (uint32_t i = 0; i < M; i++) rms[i] = 0.5f + (i % 3) / 8.0f;
    void* T[] = {C, A, B, rms, gamma, U, bias, bias};
    PlowDevInst in = {0};
    in.op = op;
    for (int i = 0; i < 8; i++) in.t[i] = i;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    if (!norm && !glu) { in.i[4] = offset; in.i[5] = offset; }
    if (glu) in.i[5] = 1;
    PlowCpuCtx ctx = {0};
    double start = now();
    gold[op](&in, 0, 1, T, &ctx);
    double scalar = now() - start;
    memcpy(ref, C, nc * 2);
    int bad = plow_cpu_tier_of(op) != PLOW_CPU_ISA_AVX512;
    start = now();
    for (uint32_t blocks = 1; blocks <= 7; blocks += 3) {
        memset(C, 0, nc * 2);
        for (uint32_t slice = 0; slice < blocks; slice++) {
            // Compare after each slice, including untouched output rows and columns.
            memset(ref, 0, nc * 2);
            gold[op](&in, slice, blocks, T, &ctx);
            memcpy(ref, C, nc * 2);
            memset(C, 0, nc * 2);
            plow_cpu_kernel(op)(&in, slice, blocks, T, &ctx);
            for (size_t i = 0; i < nc; i++) {
                const float a = plow_bf2f(C[i]), b = plow_bf2f(ref[i]);
                if (!isfinite(a) || fabsf(a - b) > 0.01f * fabsf(b) + 0.002f) { bad++; break; }
            }
            memset(C, 0, nc * 2);
        }
    }
    if (bench) {
        start = now();
        for (int i = 0; i < 5; i++) plow_cpu_kernel(op)(&in, 0, 1, T, &ctx);
        printf("GEMM %ux%ux%u scalar %.3f ms AVX512 %.3f ms\n", M, N, K,
               scalar * 1e3, (now() - start) * 200.0);
    }
    if (bad) printf("FAIL op=%u M=%u N=%u K=%u (%d)\n", op, M, N, K, bad);
    free(C); free(ref); free(A); free(B); free(U); free(gamma); free(bias); free(rms);
    return bad;
}

static int splitk_check(void) {
    const uint32_t M = 7, N = 19, K = 65;
    float C[M * N], ref[M * N];
    plow_bf16 A[M * K], W[N * K];
    for (uint32_t i = 0; i < M * K; i++) A[i] = plow_f2bf(((int)(i % 19) - 9) / 16.0f);
    for (uint32_t i = 0; i < N * K; i++) W[i] = plow_f2bf(((int)(i % 23) - 11) / 32.0f);
    void* T[] = {C, A, W};
    PlowDevInst in = {0}; in.op = PLOW_DOP_GEMM_SPLITK;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[3] = 4;
    PlowCpuCtx ctx = {0};
    for (uint32_t blocks = 1; blocks < 25; blocks += 3) {
        for (uint32_t slice = 0; slice < blocks; slice++) {
            memset(C, 0, sizeof C); g_gemm_splitk(&in, slice, blocks, T, &ctx);
            memcpy(ref, C, sizeof C); memset(C, 0, sizeof C);
            plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
            for (uint32_t i = 0; i < M * N; i++)
                if (fabsf(C[i] - ref[i]) > 1e-6f) { printf("FAIL splitk\n"); return 1; }
        }
    }
    return plow_cpu_tier_of(in.op) != PLOW_CPU_ISA_AVX512;
}

static int quant_check(uint16_t op, uint32_t M, uint32_t N, uint32_t K, int mx, int a8) {
    const int glu = op == PLOW_DOP_GEMM_GLU_FP8 || op == PLOW_DOP_GEMM_GLU_MXFP4;
    const uint32_t off = glu ? 0 : 2;
    const size_t nc = (size_t)(M + off) * N, na = (size_t)(M + off) * K;
    const size_t nw = (size_t)N * (mx ? K / 2 : K), ns = (size_t)N * ((K + 31) / 32);
    plow_bf16* C = calloc(nc, 2), *ref = calloc(nc, 2), *A = malloc(na * 2), *bias = calloc(N, 2);
    uint8_t* A8 = malloc(na), *W = malloc(nw), *U = malloc(nw), *S = malloc(ns), *US = malloc(ns);
    float* as = malloc((M + off) * sizeof(float)), *ws = malloc(N * sizeof(float)), *us = malloc(N * sizeof(float));
    for (size_t i = 0; i < na; i++) { A[i] = plow_f2bf(((int)(i % 19) - 9) / 16.0f); A8[i] = i % 125; }
    for (size_t i = 0; i < nw; i++) {
        W[i] = i % 256; U[i] = (i * 17) % 256;
        if (!mx && K > 1) {
            if ((W[i] & 127) == 127) W[i] &= 254;
            if ((U[i] & 127) == 127) U[i] &= 254;
        }
    }
    for (size_t i = 0; i < ns; i++) { S[i] = 120 + i % 8; US[i] = 119 + i % 7; }
    for (uint32_t i = 0; i < M + off; i++) as[i] = 0.125f + (i % 3) / 16.0f;
    for (uint32_t i = 0; i < N; i++) { ws[i] = 0.001f * (1 + i % 7); us[i] = 0.002f; bias[i] = plow_f2bf(0.125f); }
    void* T[8] = {C, a8 ? (void*)A8 : (void*)A, W, mx ? (void*)S : a8 ? (void*)as : NULL,
                   mx ? (void*)US : (void*)ws, U, us, bias};
    PlowDevInst in = {0}; in.op = op;
    for (int i = 0; i < 8; i++) in.t[i] = i;
    if (!mx && !a8) in.t[3] = PLOW_TENSOR_NONE;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    if (glu) in.i[5] = 1; else { in.i[4] = off; in.i[5] = off; }
    plow_cpu_kernel_fn gold[PLOW_CPU_DOP_TABLE] = {0};
    plow_cpu_register_golden(gold); plow_cpu_register_golden_fp8(gold);
    PlowCpuCtx ctx = {0};
    int bad = plow_cpu_tier_of(op) != PLOW_CPU_ISA_AVX512;
    for (uint32_t blocks = 1; blocks <= 7; blocks += 3) {
        for (uint32_t slice = 0; slice < blocks; slice++) {
            memset(C, 0, nc * 2);
            gold[op](&in, slice, blocks, T, &ctx);
            memcpy(ref, C, nc * 2); memset(C, 0, nc * 2);
            plow_cpu_kernel(op)(&in, slice, blocks, T, &ctx);
            for (size_t i = 0; i < nc; i++) {
                const float a = plow_bf2f(C[i]), b = plow_bf2f(ref[i]);
                if (a == b || (isnan(a) && isnan(b))) continue;
                if (!isfinite(a) || !isfinite(b) || fabsf(a-b) > 0.01f * fabsf(b) + 0.002f) {
                    bad++; break;
                }
            }
        }
    }
    if (M == 64 && N == 512 && K == 1024) {
        double start = now(); gold[op](&in, 0, 1, T, &ctx);
        double scalar = now() - start;
        start = now();
        for (int i = 0; i < 5; i++) plow_cpu_kernel(op)(&in, 0, 1, T, &ctx);
        printf("GEMM %s %ux%ux%u scalar %.3f ms AVX512 %.3f ms\n", mx ? "MXFP4" : "FP8", M, N, K,
               scalar * 1e3, (now() - start) * 200.0);
    }
    if (bad) printf("FAIL quant op=%u M=%u N=%u K=%u a8=%d (%d)\n", op, M, N, K, a8, bad);
    free(C); free(ref); free(A); free(A8); free(W); free(U); free(S); free(US); free(as); free(ws); free(us); free(bias);
    return bad;
}

int main(int argc, char** argv) {
    (void)argv;
    if (plow_cpu_init(PLOW_CPU_ISA_AVX512) < PLOW_CPU_ISA_AVX512) return 77;
    const uint16_t ops[] = {PLOW_DOP_GEMM, PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED,
        PLOW_DOP_GEMM_WIDE, PLOW_DOP_GEMM_C5, PLOW_DOP_GEMM_NORM, PLOW_DOP_GEMM_GLU};
    int bad = splitk_check();
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        bad += check(ops[i], 7, 19, 1, 0);
        bad += check(ops[i], 17, 259, 33, 0);
        bad += check(ops[i], 259, 259, 65, 0);
    }
    if (argc > 1) bad += check(PLOW_DOP_GEMM_SMALL, 64, 512, 1024, 1);
    const uint16_t fp8[] = {PLOW_DOP_GEMM_FP8, PLOW_DOP_GEMM_SMALL_FP8, PLOW_DOP_GEMM_MED_FP8,
        PLOW_DOP_GEMM_WIDE_FP8, PLOW_DOP_GEMM_C5_FP8, PLOW_DOP_GEMM_GLU_FP8};
    const uint16_t mx[] = {PLOW_DOP_GEMM_MXFP4, PLOW_DOP_GEMM_SMALL_MXFP4, PLOW_DOP_GEMM_MED_MXFP4,
        PLOW_DOP_GEMM_WIDE_MXFP4, PLOW_DOP_GEMM_C5_MXFP4, PLOW_DOP_GEMM_GLU_MXFP4};
    for (size_t i = 0; i < 6; i++) {
        for (int a8 = 0; a8 < 2; a8++) {
            bad += quant_check(fp8[i], 7, 256, 1, 0, a8);
            bad += quant_check(fp8[i], 259, 259, 33, 0, a8);
        }
        bad += quant_check(mx[i], 7, 259, 2, 1, 0);
        bad += quant_check(mx[i], 259, 259, 66, 1, 0);
    }
    if (argc > 1) {
        bad += quant_check(PLOW_DOP_GEMM_SMALL_FP8, 64, 512, 1024, 0, 0);
        bad += quant_check(PLOW_DOP_GEMM_SMALL_MXFP4, 64, 512, 1024, 1, 0);
    }
    return bad ? 1 : 0;
}
