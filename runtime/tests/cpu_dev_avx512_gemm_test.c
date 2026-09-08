#define _POSIX_C_SOURCE 200809L
#include "cpu_dev_internal.h"
#include "golden/golden.h"
#include "fp8_common.h"
#include "golden/gptoss.h"
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

static float fp8_oracle_value(unsigned code) {
    return code < 8 ? ldexpf((float)code, -9)
                    : ldexpf(1.0f + (code & 7) / 8.0f, (int)(code >> 3) - 7);
}

static uint8_t fp8_oracle(float x) {
    const unsigned sign = signbit(x) ? 128 : 0;
    if (isnan(x)) return sign | 127;
    x = fabsf(x);
    if (x >= 448) return sign | 126;
    unsigned lo = 0, hi = 126;
    while (hi - lo > 1) {
        unsigned mid = (lo + hi) / 2;
        if (fp8_oracle_value(mid) <= x) lo = mid; else hi = mid;
    }
    const float dl = x - fp8_oracle_value(lo), dh = fp8_oracle_value(hi) - x;
    return sign | (dl < dh || (dl == dh && !(lo & 1)) ? lo : hi);
}

/* v_gelu_tanh must keep the far-negative tail. Evaluating 0.5*x*(1 + tanh(c)) there cancels away
 * the whole result -- the scalar tier's tanhf saturates to exactly -1 and flushes the output to
 * zero -- so the kernel uses the identity x * sigmoid(2c). The reference is that identity in
 * double, which is the one form that stays accurate here. */
static int gelu_tail_check(void) {
    enum { N = 64 };
    plow_bf16 gate[N], up[N], out[N];
    for (unsigned i = 0; i < N; i++) {
        gate[i] = plow_f2bf(-4.0f - 5.0f * (float)i / (N - 1));
        up[i] = plow_f2bf(1.0f);
    }
    PlowDevInst in = {0}; in.op = PLOW_DOP_GLU;
    for (unsigned i = 0; i < 8; i++) in.t[i] = i < 3 ? i : PLOW_TENSOR_NONE;
    in.i[0] = N; in.i[1] = 0;
    void* T[] = {out, gate, up}; PlowCpuCtx ctx = {0};
    plow_cpu_kernel(in.op)(&in, 0, 1, T, &ctx);
    int bad = 0;
    for (unsigned i = 0; i < N; i++) {
        const double g = plow_bf2f(gate[i]);
        const double c = 2.0 * 0.7978845608028654 * (g + 0.044715 * g * g * g);
        const double want = g / (1.0 + exp(-c));
        bad += fabs(plow_bf2f(out[i]) - want) > 0.02 * fabs(want);
    }
    if (bad) printf("FAIL gelu tail (%d)\n", bad);
    return bad;
}

static int activation_quant_check(void) {
    int bad = 0;
    for (unsigned code = 0; code < 126; code++) {
        const float mid = (fp8_oracle_value(code) + fp8_oracle_value(code + 1)) * 0.5f;
        const float points[] = {fp8_oracle_value(code), nextafterf(mid, 0), mid,
                                nextafterf(mid, INFINITY)};
        for (unsigned j = 0; j < 4; j++) for (int sign = -1; sign <= 1; sign += 2)
            bad += plow_f32_to_e4m3(sign * points[j]) != fp8_oracle(sign * points[j]);
    }
    const float special[] = {0, -0.0f, INFINITY, -INFINITY, NAN, -NAN, 448, -500};
    for (unsigned i = 0; i < sizeof special / sizeof special[0]; i++)
        bad += plow_f32_to_e4m3(special[i]) != fp8_oracle(special[i]);
    plow_cpu_kernel_fn gold[PLOW_CPU_DOP_TABLE] = {0};
    plow_cpu_register_golden_fp8(gold);
    const unsigned widths[] = {1, 15, 16, 17, 257, 65537};
    for (unsigned shape = 0; shape < sizeof widths / sizeof widths[0]; shape++) {
        const unsigned K = widths[shape], M = 3;
        plow_bf16* x = malloc(M * K * 2);
        uint8_t* q = malloc(M * K), *ref = malloc(M * K);
        float scales[3], refs[3];
        for (unsigned k = 0; k < K; k++) {
            float v = plow_bf2f((plow_bf16)k);
            x[k] = plow_f2bf(isfinite(v) ? fmaxf(-448, fminf(v, 448)) : v);
            x[K+k] = plow_f2bf(k & 1 ? -0.0f : 0.0f);
            x[2*K+k] = plow_f2bf(((int)(k % 31) - 15) * 1e-14f);
        }
        x[0] = plow_f2bf(448);
        /* Exhaust finite bf16 values at unit scale; non-finites have their own row below. */
        for (unsigned k = 1; k < K; k++) if (!isfinite(plow_bf2f(x[k]))) x[k] = 0;
        PlowDevInst in = {0}; in.op = PLOW_DOP_QUANT_FP8;
        for (unsigned i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.i[0] = M; in.i[1] = K;
        void* T[] = {q, x, scales}; PlowCpuCtx ctx = {0};
        for (unsigned blocks = 1; blocks <= 5; blocks += 4) {
            for (unsigned slice = 0; slice < blocks; slice++) {
                memset(q, 0xa5, M*K); for (unsigned m = 0; m < M; m++) scales[m] = -1;
                gold[in.op](&in, slice, blocks, T, &ctx);
                memcpy(ref, q, M*K); memcpy(refs, scales, sizeof refs);
                memset(q, 0xa5, M*K); for (unsigned m = 0; m < M; m++) scales[m] = -1;
                plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
                bad += memcmp(q, ref, M*K) != 0 || memcmp(scales, refs, sizeof refs) != 0;
                for (unsigned m = 0; m < M; m++) if (scales[m] >= 0)
                    for (unsigned k = 0; k < K; k++)
                        bad += q[m*K+k] != fp8_oracle(plow_bf2f(x[m*K+k]) * (1.0f / scales[m]));
            }
        }
        x[0] = plow_f2bf(NAN); x[K] = plow_f2bf(INFINITY);
        gold[in.op](&in, 0, 1, T, &ctx); memcpy(ref, q, M*K);
        plow_cpu_kernel(in.op)(&in, 0, 1, T, &ctx);
        bad += memcmp(q, ref, M*K) != 0;
        free(x); free(q); free(ref);
    }
    for (unsigned act = 0; act < 2; act++) {
        plow_bf16 gate[51], up[51], x[51]; uint8_t q[51]; float scales[3];
        for (unsigned k = 0; k < 51; k++) {
            gate[k] = plow_f2bf(((int)k - 25) / 8.0f);
            up[k] = plow_f2bf(((int)(k % 7) - 3) / 4.0f);
        }
        PlowDevInst in = {0}; in.op = PLOW_DOP_QUANT_FP8;
        for (unsigned i = 0; i < 8; i++) in.t[i] = i < 5 ? i : PLOW_TENSOR_NONE;
        in.i[0] = 3; in.i[1] = 17; in.i[2] = act;
        void* T[] = {q, x, scales, gate, up}; PlowCpuCtx ctx = {0};
        for (unsigned slice = 0; slice < 5; slice++) plow_cpu_kernel(in.op)(&in, slice, 5, T, &ctx);
        for (unsigned k = 0; k < 51; k++) {
            const float g = plow_bf2f(gate[k]), u = plow_bf2f(up[k]);
            const float activated = act == 1 ? g / (1.0f + expf(-g))
                : 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f*g*g*g)));
            bad += fabsf(plow_bf2f(x[k]) - activated*u) > 0.005f * fabsf(activated*u) + 1e-6f;
            bad += q[k] != fp8_oracle(plow_bf2f(x[k]) * (1.0f / scales[k/17]));
        }
    }
    /* The vectorized fused gate/up path against golden, over widths that exercise the masked
     * tail and every slice split. The two tiers evaluate the activation differently, so the
     * bound is against each row's own dynamic range -- what the fp8 scale encodes -- and not
     * against bit patterns: deep in the gelu tail both tiers sit within noise of zero, while a
     * masking, slicing or indexing regression moves a value by a sizable fraction of its row.
     * The gate stride keeps every row spanning the whole range, so no row is all-tail. */
    for (unsigned act = 0; act < 2; act++) {
        const unsigned widths[] = {1, 15, 16, 17, 129, 1024};
        for (unsigned shape = 0; shape < sizeof widths / sizeof widths[0]; shape++) {
            const unsigned K = widths[shape], M = 5, n = M * K;
            plow_bf16* gate = malloc(n*2), *up = malloc(n*2), *x = malloc(n*2), *xr = malloc(n*2);
            uint8_t* q = malloc(n), *qr = malloc(n);
            float sc[5], scr[5];
            for (unsigned i = 0; i < n; i++) {
                gate[i] = plow_f2bf(((int)((i * 37) % 97) - 48) / 12.0f);
                up[i] = plow_f2bf(((int)((i * 11) % 53) - 26) / 9.0f);
            }
            PlowDevInst in = {0}; in.op = PLOW_DOP_QUANT_FP8;
            for (unsigned i = 0; i < 8; i++) in.t[i] = i < 5 ? i : PLOW_TENSOR_NONE;
            in.i[0] = M; in.i[1] = K; in.i[2] = act;
            void* Tg[] = {qr, xr, scr, gate, up}, *Tv[] = {q, x, sc, gate, up};
            PlowCpuCtx ctx = {0};
            for (unsigned blocks = 1; blocks <= 7; blocks += 6)
                for (unsigned slice = 0; slice < blocks; slice++) {
                    gold[in.op](&in, slice, blocks, Tg, &ctx);
                    plow_cpu_kernel(in.op)(&in, slice, blocks, Tv, &ctx);
                }
            for (unsigned m = 0; m < M; m++) {
                float hi = 0;
                for (unsigned k = 0; k < K; k++) hi = fmaxf(hi, fabsf(plow_bf2f(xr[m*K+k])));
                bad += fabsf(sc[m] - scr[m]) > 0.01f * scr[m];
                for (unsigned k = 0; k < K; k++) {
                    const unsigned i = m * K + k;
                    bad += fabsf(plow_bf2f(x[i]) - plow_bf2f(xr[i])) > 0.01f * hi;
                    /* fp8 codes are only comparable where the value carries signal. */
                    bad += fabsf(plow_bf2f(xr[i])) > 0.05f * hi && abs((int)q[i] - (int)qr[i]) > 1;
                }
            }
            free(gate); free(up); free(x); free(xr); free(q); free(qr);
        }
    }
    if (bad) printf("FAIL activation FP8 quantization (%d)\n", bad);
    return bad;
}

static int moe_fp8_check(void) {
    enum { H = 35, I = 17, E = 3, TOP = 4, ROWS = 3 };
    plow_bf16 x[ROWS*H], fu[ROWS*TOP*I], ref[ROWS*TOP*I];
    float out[ROWS*TOP*H], expected[ROWS*TOP*H], gs[2*I], ds[H];
    uint8_t gu[2*I*H], down[H*I];
    uint64_t weights[2*E] = {(uintptr_t)gu, (uintptr_t)down, 0, 0, (uintptr_t)gu, (uintptr_t)down};
    uint64_t scales[2*E] = {(uintptr_t)gs, (uintptr_t)ds, (uintptr_t)gs, (uintptr_t)ds, 0, 0};
    plow_moe_route routes[ROWS*TOP];
    for (unsigned j = 0; j < ROWS*H; j++) x[j] = plow_f2bf(((int)(j % 23)-11)/16.0f);
    for (unsigned j = 0; j < 2*I*H; j++) gu[j] = (j*29 % 127) | (j & 128);
    for (unsigned j = 0; j < H*I; j++) down[j] = (j*7 % 127) | (j & 128);
    for (unsigned j = 0; j < 2*I; j++) gs[j] = 0.0001f*(j+1);
    for (unsigned j = 0; j < H; j++) ds[j] = 0.0001f*(j+1);
    for (unsigned j = 0; j < ROWS*TOP; j++) {
        routes[j].eid = j % TOP;
        routes[j].gate = 0.2f + (j % 3) * 0.15f;
    }
    plow_cpu_kernel_fn gold[PLOW_CPU_DOP_TABLE] = {0}; plow_cpu_register_golden_fp8(gold);
    PlowDevInst in = {0};
    for (unsigned j = 0; j < 8; j++) in.t[j] = j < 5 ? j : PLOW_TENSOR_NONE;
    in.i[0] = TOP; in.i[3] = E;
    PlowCpuCtx ctx = {0}; int bad = 0;
    for (unsigned rows = 1; rows <= ROWS; rows += 2) {
        in.i[5] = rows == 1 ? 0 : rows;
        for (unsigned blocks = 1; blocks <= 7; blocks += 3) {
            in.op = PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8; in.i[1] = I; in.i[2] = H;
            void* T[] = {fu, x, routes, weights, scales};
            for (unsigned slice = 0; slice < blocks; slice++) {
                for (unsigned j = 0; j < ROWS*TOP*I; j++) fu[j] = plow_f2bf(7);
                gold[in.op](&in, slice, blocks, T, &ctx); memcpy(ref, fu, sizeof ref);
                for (unsigned j = 0; j < ROWS*TOP*I; j++) fu[j] = plow_f2bf(7);
                plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
                for (unsigned j = 0; j < ROWS*TOP*I; j++)
                    bad += !isfinite(plow_bf2f(fu[j])) || fabsf(plow_bf2f(fu[j])-plow_bf2f(ref[j])) > 0.01f*fabsf(plow_bf2f(ref[j]))+1e-5f;
            }
            for (unsigned j = 0; j < ROWS*TOP*I; j++) fu[j] = plow_f2bf(7);
            for (unsigned slice = 0; slice < blocks; slice++) plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
            for (unsigned slot = 0; slot < rows*TOP; slot++) for (unsigned n = 0; n < I; n++) {
                float value = 7;
                if (routes[slot].eid == 0) {
                    float g = 0, u = 0;
                    for (unsigned h = 0; h < H; h++) {
                        const uint8_t gq = gu[n*H+h], uq = gu[(I+n)*H+h];
                        const float a = plow_bf2f(x[(slot/TOP)*H+h]);
                        g += a * fp8_oracle_value(gq & 127) * (gq & 128 ? -1 : 1);
                        u += a * fp8_oracle_value(uq & 127) * (uq & 128 ? -1 : 1);
                    }
                    g *= gs[n]; u *= gs[I+n];
                    value = plow_bf2f(plow_f2bf(0.5f*g*(1+tanhf(0.7978845608028654f*(g+0.044715f*g*g*g)))*u));
                }
                bad += !isfinite(plow_bf2f(fu[slot*I+n])) || fabsf(plow_bf2f(fu[slot*I+n])-value) > 0.01f*fabsf(value)+1e-5f;
            }
            in.op = PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8; in.i[1] = H; in.i[2] = I;
            T[0] = out; T[1] = fu;
            for (unsigned j = 0; j < ROWS*TOP*I; j++) fu[j] = plow_f2bf(((int)(j % 13)-6)/16.0f);
            for (unsigned slice = 0; slice < blocks; slice++) {
                for (unsigned j = 0; j < ROWS*TOP*H; j++) out[j] = 7;
                gold[in.op](&in, slice, blocks, T, &ctx); memcpy(expected, out, sizeof expected);
                for (unsigned j = 0; j < ROWS*TOP*H; j++) out[j] = 7;
                plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
                for (unsigned j = 0; j < ROWS*TOP*H; j++)
                    bad += !isfinite(out[j]) || fabsf(out[j]-expected[j]) > 1e-5f*fabsf(expected[j])+1e-6f;
            }
            for (unsigned slice = 0; slice < blocks; slice++) plow_cpu_kernel(in.op)(&in, slice, blocks, T, &ctx);
            for (unsigned slot = 0; slot < rows*TOP; slot++) for (unsigned h = 0; h < H; h++) {
                float value = 0;
                if (routes[slot].eid == 0) {
                    for (unsigned j = 0; j < I; j++) {
                        const uint8_t q = down[h*I+j];
                        value += plow_bf2f(fu[slot*I+j]) * fp8_oracle_value(q & 127) * (q & 128 ? -1 : 1);
                    }
                    value *= ds[h] * routes[slot].gate;
                }
                bad += !isfinite(out[slot*H+h]) || fabsf(out[slot*H+h]-value) > 1e-5f*fabsf(value)+1e-6f;
            }
        }
    }
    if (bad) printf("FAIL Gemma FP8 experts (%d)\n", bad);
    return bad;
}

int main(int argc, char** argv) {
    (void)argv;
    if (plow_cpu_init(PLOW_CPU_ISA_AVX512) < PLOW_CPU_ISA_AVX512) return 77;
    const uint16_t ops[] = {PLOW_DOP_GEMM, PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED,
        PLOW_DOP_GEMM_WIDE, PLOW_DOP_GEMM_C5, PLOW_DOP_GEMM_NORM, PLOW_DOP_GEMM_GLU};
    int bad = splitk_check() + activation_quant_check() + moe_fp8_check() + gelu_tail_check();
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
