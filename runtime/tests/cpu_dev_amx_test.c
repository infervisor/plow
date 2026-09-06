/* cpu_dev_amx_test.c — AMX tier GEMM family against the golden tier.
 *
 * Every op runs over ALL slices for nblk in {1, 3, 16} at Gemma-4-12B-like shapes; results are
 * compared element-wise to the golden kernel (1e-2 relative + 1e-2 absolute, fp32 accumulate
 * with one bf16 round on both sides). `--bench`: single-thread GEMM GFLOPS. Exits 0 with a
 * message when the host has no AMX. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev.h"
#include "cpu_dev_internal.h"
#include "golden/golden.h"

static int fails = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            fails++;                                    \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
        }                                               \
    } while (0)

static uint32_t rng = 0x9E3779B9u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
}
static void fill_bf16(plow_bf16* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(frand() * scale);
}
static void fill_f32(float* p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) p[i] = lo + (frand() + 1.0f) * 0.5f * (hi - lo);
}

static PlowDevInst inst(uint16_t op) {
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    in.op = op;
    for (int k = 0; k < 8; k++) in.t[k] = PLOW_TENSOR_NONE;
    return in;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Compare `got` (AMX) to `want` (golden) over n bf16; reports the worst element. */
static void compare(const char* what, const plow_bf16* got, const plow_bf16* want, size_t n) {
    size_t bad = 0, worst = 0;
    float worst_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float g = plow_bf2f(got[i]), w = plow_bf2f(want[i]);
        const float err = fabsf(g - w), tol = 1e-2f * fabsf(w) + 1e-2f;
        if (!(err <= tol)) bad++;
        if (err > worst_err || err != err) {
            worst_err = err;
            worst = i;
        }
    }
    CHECK(bad == 0, "%s: %zu/%zu elements off (worst i=%zu got %f want %f)", what, bad, n, worst,
          plow_bf2f(got[worst]), plow_bf2f(want[worst]));
}

typedef void (*kfn)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);

static void run_all(kfn f, const PlowDevInst* in, uint32_t nblk, void* const* T, PlowCpuCtx* ctx) {
    PlowDevInst i2 = *in;
    i2.blocks = (uint16_t)nblk;
    for (uint32_t s = 0; s < nblk; s++) f(&i2, s, nblk, T, ctx);
}

static const uint32_t NBLKS[3] = {1, 3, 16};

static kfn golden_of(uint16_t op) {
    switch (op) {
        case PLOW_DOP_GEMM: return g_gemm;
        case PLOW_DOP_GEMM_SMALL: return g_gemm_small;
        case PLOW_DOP_GEMM_MED: return g_gemm_med;
        case PLOW_DOP_GEMM_WIDE: return g_gemm_wide;
        case PLOW_DOP_GEMM_C5: return g_gemm_c5;
        case PLOW_DOP_GEMM_NORM: return g_gemm_norm;
        case PLOW_DOP_GEMM_GLU: return g_gemm_glu;
        default: return NULL;
    }
}

static const char* name_of(uint16_t op) {
    switch (op) {
        case PLOW_DOP_GEMM: return "GEMM";
        case PLOW_DOP_GEMM_SMALL: return "GEMM_SMALL";
        case PLOW_DOP_GEMM_MED: return "GEMM_MED";
        case PLOW_DOP_GEMM_WIDE: return "GEMM_WIDE";
        case PLOW_DOP_GEMM_C5: return "GEMM_C5";
        case PLOW_DOP_GEMM_NORM: return "GEMM_NORM";
        case PLOW_DOP_GEMM_GLU: return "GEMM_GLU";
        default: return "?";
    }
}

/* Plain GEMM ops (+ a_row0/c_row0), GEMM_NORM, GEMM_GLU at one shape. */
static void test_shape(uint16_t op, uint32_t M, uint32_t N, uint32_t K, PlowCpuCtx* ctx) {
    const uint32_t a_row0 = (op == PLOW_DOP_GEMM_SMALL && M > 8) ? 3 : 0;
    const uint32_t c_row0 = (op == PLOW_DOP_GEMM_MED) ? 2 : 0;
    const int glu = op == PLOW_DOP_GEMM_GLU, norm = op == PLOW_DOP_GEMM_NORM;
    plow_bf16* A = malloc((size_t)(M + a_row0) * K * 2);
    plow_bf16* W = malloc((size_t)N * K * 2);
    plow_bf16* Wu = glu ? malloc((size_t)N * K * 2) : NULL;
    plow_bf16* C = malloc((size_t)(M + c_row0) * N * 2);
    plow_bf16* Cg = malloc((size_t)(M + c_row0) * N * 2);
    float* rms = norm ? malloc(M * sizeof(float)) : NULL;
    plow_bf16* gamma = norm ? malloc((size_t)K * 2) : NULL;
    fill_bf16(A, (size_t)(M + a_row0) * K, 1.0f);
    fill_bf16(W, (size_t)N * K, 0.05f);
    if (glu) fill_bf16(Wu, (size_t)N * K, 0.05f);
    if (norm) {
        fill_f32(rms, M, 0.5f, 1.5f);
        fill_bf16(gamma, K, 1.0f);
    }
    void* T[8] = {C, A, W, rms, gamma, Wu, NULL, NULL};
    PlowDevInst in = inst(op);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (norm) { in.t[3] = 3; in.t[4] = 4; }
    if (glu) { in.t[5] = 5; in.i[5] = 1; /* silu */ }
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    if (!norm && !glu) { in.i[4] = a_row0; in.i[5] = c_row0; }

    memset(C, 0, (size_t)(M + c_row0) * N * 2);
    run_all(golden_of(op), &in, 1, T, ctx); /* golden reference into C */
    memcpy(Cg, C, (size_t)(M + c_row0) * N * 2);
    kfn f = plow_cpu_kernel(op);
    CHECK(f != NULL && f != golden_of(op), "%s: AMX kernel not registered", name_of(op));
    for (int k = 0; k < 3; k++) {
        memset(C, 0, (size_t)(M + c_row0) * N * 2);
        const double t0 = now();
        run_all(f, &in, NBLKS[k], T, ctx);
        const double dt = now() - t0;
        char what[128];
        snprintf(what, sizeof what, "%s M=%u N=%u K=%u nblk=%u", name_of(op), M, N, K, NBLKS[k]);
        compare(what, C, Cg, (size_t)(M + c_row0) * N);
        if (k == 0)
            printf("  %-40s %.1f ms  %.1f GFLOPS (1 thread, all slices)\n", what, dt * 1e3,
                   2.0 * M * N * K * (glu ? 2 : 1) / dt / 1e9);
    }
    free(A); free(W); free(Wu); free(C); free(Cg); free(rms); free(gamma);
}

static void bench(PlowCpuCtx* ctx) {
    const uint32_t M = 512, N = 3840, K = 3840;
    plow_bf16* A = malloc((size_t)M * K * 2);
    plow_bf16* W = malloc((size_t)N * K * 2);
    plow_bf16* C = malloc((size_t)M * N * 2);
    fill_bf16(A, (size_t)M * K, 1.0f);
    fill_bf16(W, (size_t)N * K, 0.05f);
    void* T[3] = {C, A, W};
    PlowDevInst in = inst(PLOW_DOP_GEMM);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    in.blocks = 1;
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMM);
    f(&in, 0, 1, T, ctx); /* warm */
    double best = 1e9;
    for (int r = 0; r < 5; r++) {
        const double t0 = now();
        f(&in, 0, 1, T, ctx);
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    /* Where the time goes: strip packing vs the raw 2x2 K loop on scratch-resident data. */
    {
        extern void plow_cpu_amx_pack_b_strip(void*, const plow_bf16*, uint32_t, uint32_t,
                                              uint32_t, uint32_t, uint32_t);
        extern double plow_cpu_amx_debug_kloop(uint32_t iters, PlowCpuCtx* ctx);
        const double t0 = now();
        for (uint32_t s = 0; s < N / 32; s++)
            for (uint32_t k0 = 0; k0 < K; k0 += 1024)
                plow_cpu_amx_pack_b_strip(ctx->scratch, W, N, K, s * 32, k0,
                                          K - k0 < 1024 ? K - k0 : 1024);
        const double tp = now() - t0;
        printf("pack all of W (%u x %u bf16 = %.1f MB): %.2f ms = %.1f GB/s\n", N, K,
               (double)N * K * 2 / 1e6, tp * 1e3, (double)N * K * 2 / tp / 1e9);
        const uint32_t iters = 2000;
        const double tk = plow_cpu_amx_debug_kloop(iters, ctx);
        printf("raw 2x2 K loop (one panel, data in scratch): %.1f GFLOPS\n",
               2.0 * 32 * 32 * 512 * iters / tk / 1e9);
        extern double plow_cpu_amx_debug_tdp(uint32_t iters, PlowCpuCtx* ctx);
        const double tt = plow_cpu_amx_debug_tdp(200000, ctx);
        printf("pure TDPBF16PS (no loads): %.1f GFLOPS = TMUL ceiling on this core\n",
               2.0 * 16 * 16 * 32 * 4 * 200000 / tt / 1e9);
    }
    const double flops = 2.0 * M * N * K;
    printf("bench GEMM M=%u N=%u K=%u: %.2f ms, %.1f GFLOPS on 1 thread "
           "(peak 1024 FLOP/cycle x 3.4 GHz assumed = 3482 GFLOPS -> %.0f%%)\n",
           M, N, K, best * 1e3, flops / best / 1e9, flops / best / 1e9 / 3482.0 * 100.0);
    free(A); free(W); free(C);
}

/* ---- Batched decode GEMV on AMX (gemv_amx.c): GEMV / GEMV_GLU / GEMV_QKV vs golden, M >= 4 ---- */
static void test_gemv_amx(uint32_t M, uint32_t N, uint32_t K, int bias, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)(M + 2) * K * 2);
    plow_bf16* W = malloc((size_t)N * K * 2);
    plow_bf16* b = malloc((size_t)N * 2);
    plow_bf16 *C = calloc((size_t)M * N, 2), *R = calloc((size_t)M * N, 2);
    fill_bf16(x, (size_t)(M + 2) * K, 1.0f); fill_bf16(W, (size_t)N * K, 0.05f); fill_bf16(b, N, 0.5f);
    void* T[8] = {C, x, W, NULL, NULL, NULL, NULL, bias ? b : NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; if (bias) in.t[7] = 7;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[4] = 2; /* a_row0 */
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV);
    CHECK(plow_cpu_tier_of(PLOW_DOP_GEMV) == PLOW_CPU_ISA_AMX, "GEMV on the AMX tier");
    T[0] = R; run_all(g_gemv, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = C; memset(C, 0, (size_t)M * N * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96];
        snprintf(what, sizeof what, "amx gemv M=%u N=%u K=%u bias=%d nblk=%u", M, N, K, bias, NBLKS[k]);
        compare(what, C, R, (size_t)M * N);
    }
    free(x); free(W); free(b); free(C); free(R);
}

static void test_gemv_glu_amx(uint32_t M, uint32_t N, uint32_t K, uint32_t act, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)M * K * 2);
    plow_bf16* Wg = malloc((size_t)N * K * 2);
    plow_bf16* Wu = malloc((size_t)N * K * 2);
    plow_bf16 *C = calloc((size_t)M * N, 2), *R = calloc((size_t)M * N, 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_bf16(Wg, (size_t)N * K, 0.05f); fill_bf16(Wu, (size_t)N * K, 0.05f);
    void* T[8] = {C, x, Wg, NULL, NULL, Wu, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_GLU);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = act;
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_GLU);
    T[0] = R; run_all(g_gemv_glu, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = C; memset(C, 0, (size_t)M * N * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96];
        snprintf(what, sizeof what, "amx gemv_glu M=%u N=%u K=%u act=%u nblk=%u", M, N, K, act, NBLKS[k]);
        compare(what, C, R, (size_t)M * N);
    }
    free(x); free(Wg); free(Wu); free(C); free(R);
}

static void test_gemv_qkv_amx(uint32_t M, uint32_t Nq, uint32_t Nk, uint32_t Nv, uint32_t K, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)M * K * 2);
    plow_bf16 *Wq = malloc((size_t)Nq * K * 2), *Wk = malloc((size_t)Nk * K * 2), *Wv = malloc((size_t)Nv * K * 2);
    plow_bf16 *q = calloc((size_t)M * Nq, 2), *kk = calloc((size_t)M * Nk, 2), *v = calloc((size_t)M * Nv, 2);
    plow_bf16 *rq = calloc((size_t)M * Nq, 2), *rk = calloc((size_t)M * Nk, 2), *rv = calloc((size_t)M * Nv, 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_bf16(Wq, (size_t)Nq * K, 0.05f); fill_bf16(Wk, (size_t)Nk * K, 0.05f); fill_bf16(Wv, (size_t)Nv * K, 0.05f);
    void* T[8] = {q, x, Wq, kk, Wk, v, Wv, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_QKV);
    for (int i = 0; i < 7; i++) in.t[i] = (uint16_t)i;
    in.i[0] = M; in.i[1] = Nq; in.i[2] = K; in.i[3] = Nk; in.i[4] = Nv;
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_QKV);
    T[0] = rq; T[3] = rk; T[5] = rv; run_all(g_gemv_qkv, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = q; T[3] = kk; T[5] = v;
        memset(q, 0, (size_t)M * Nq * 2); memset(kk, 0, (size_t)M * Nk * 2); memset(v, 0, (size_t)M * Nv * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96];
        snprintf(what, sizeof what, "amx gemv_qkv M=%u nblk=%u", M, NBLKS[k]);
        compare(what, q, rq, (size_t)M * Nq); compare(what, kk, rk, (size_t)M * Nk); compare(what, v, rv, (size_t)M * Nv);
    }
    free(x); free(Wq); free(Wk); free(Wv); free(q); free(kk); free(v); free(rq); free(rk); free(rv);
}

/* ---- Quantized batched GEMV on AMX (gemv_amx.c fp8 / mxfp4 arms) vs golden ---- */
#include "golden/fp8.h"
#include "golden/gptoss.h"
static uint8_t rb8(void) { return (uint8_t)(frand() * 255.0f); }
static void fill_fp8(uint8_t* p, size_t n) { /* finite, normal e4m3 (no NaN, no nonzero subnormals) */
    for (size_t i = 0; i < n; i++) { uint8_t b = rb8(); if ((b & 0x7f) == 0x7f) b &= 0xfe; if ((b & 0x78) == 0 && (b & 7)) b |= 0x08; p[i] = b; }
}
static void fill_fp4(uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = rb8(); }
static void fill_e8m0(uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(120 + (rb8() & 7)); }

static void test_gemv_fp8_amx(uint32_t M, uint32_t N, uint32_t K, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)(M + 1) * K * 2);
    uint8_t* W = malloc((size_t)N * K);
    float* ws = malloc((size_t)N * 4);
    plow_bf16 *C = calloc((size_t)M * N, 2), *R = calloc((size_t)M * N, 2);
    fill_bf16(x, (size_t)(M + 1) * K, 1.0f); fill_fp8(W, (size_t)N * K);
    for (uint32_t n = 0; n < N; n++) ws[n] = 0.001f + 0.01f * frand();
    void* T[8] = {C, x, W, NULL, NULL, ws, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_FP8);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[4] = 1;
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_FP8);
    T[0] = R; run_all(g_gemv_fp8, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = C; memset(C, 0, (size_t)M * N * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96]; snprintf(what, sizeof what, "amx gemv_fp8 M=%u N=%u K=%u nblk=%u", M, N, K, NBLKS[k]);
        compare(what, C, R, (size_t)M * N);
    }
    free(x); free(W); free(ws); free(C); free(R);
}

static void test_gemv_glu_fp8_amx(uint32_t M, uint32_t N, uint32_t K, uint32_t act, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)M * K * 2);
    uint8_t *Wg = malloc((size_t)N * K), *Wu = malloc((size_t)N * K);
    float *gs = malloc((size_t)N * 4), *us = malloc((size_t)N * 4);
    plow_bf16 *C = calloc((size_t)M * N, 2), *R = calloc((size_t)M * N, 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_fp8(Wg, (size_t)N * K); fill_fp8(Wu, (size_t)N * K);
    for (uint32_t n = 0; n < N; n++) { gs[n] = 0.001f + 0.01f * frand(); us[n] = 0.001f + 0.01f * frand(); }
    void* T[8] = {C, x, Wg, gs, us, Wu, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_GLU_FP8);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.t[4] = 4; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = act;
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_GLU_FP8);
    T[0] = R; run_all(g_gemv_glu_fp8, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = C; memset(C, 0, (size_t)M * N * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96]; snprintf(what, sizeof what, "amx gemv_glu_fp8 M=%u N=%u act=%u nblk=%u", M, N, act, NBLKS[k]);
        compare(what, C, R, (size_t)M * N);
    }
    free(x); free(Wg); free(Wu); free(gs); free(us); free(C); free(R);
}

static void test_gemv_mx4_amx(uint32_t M, uint32_t N, uint32_t K, int glu, uint32_t act, PlowCpuCtx* ctx) {
    plow_bf16* x = malloc((size_t)M * K * 2);
    uint8_t *W = malloc((size_t)N * K / 2), *Wu = malloc((size_t)N * K / 2);
    uint8_t *S = malloc((size_t)N * K / 32), *Su = malloc((size_t)N * K / 32);
    plow_bf16 *C = calloc((size_t)M * N, 2), *R = calloc((size_t)M * N, 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_fp4(W, (size_t)N * K / 2); fill_fp4(Wu, (size_t)N * K / 2);
    fill_e8m0(S, (size_t)N * K / 32); fill_e8m0(Su, (size_t)N * K / 32);
    void* T[8] = {C, x, W, S, Su, Wu, NULL, NULL};
    PlowDevInst in = inst(glu ? PLOW_DOP_GEMV_GLU_MXFP4 : PLOW_DOP_GEMV_MXFP4);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    if (glu) { in.t[4] = 4; in.t[5] = 5; in.i[5] = act; }
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    kfn f = plow_cpu_kernel(in.op);
    T[0] = R; run_all(glu ? g_gemv_glu_mxfp4 : g_gemv_mxfp4, &in, 1, T, ctx);
    for (int k = 0; k < 3; k++) {
        T[0] = C; memset(C, 0, (size_t)M * N * 2);
        run_all(f, &in, NBLKS[k], T, ctx);
        char what[96]; snprintf(what, sizeof what, "amx gemv_mxfp4%s M=%u N=%u K=%u nblk=%u", glu ? "_glu" : "", M, N, K, NBLKS[k]);
        compare(what, C, R, (size_t)M * N);
    }
    free(x); free(W); free(Wu); free(S); free(Su); free(C); free(R);
}

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AMX);
    if (tier < PLOW_CPU_ISA_AMX) {
        printf("no AMX on this host (tier %d) — skipping AMX tests\n", tier);
        return 0;
    }
    PlowCpuCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.scratch_bytes = plow_cpu_scratch_bytes();
    ctx.scratch = aligned_alloc(64, ctx.scratch_bytes);
    memset(ctx.scratch, 0, ctx.scratch_bytes);
    CHECK(plow_cpu_thread_init(&ctx) == 0, "thread init");
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench(&ctx);
        return 0;
    }

    /* Strip pack self-check against the ORM 20.5.3 pair formula. */
    {
        const uint32_t N = 40, K = 96, n0 = 32, k0 = 32, kp = 64;
        plow_bf16* W = malloc((size_t)N * K * 2);
        for (uint32_t i = 0; i < N * K; i++) W[i] = (plow_bf16)i;
        plow_bf16* P = aligned_alloc(64, (size_t)kp * 64);
        extern void plow_cpu_amx_pack_b_strip(void*, const plow_bf16*, uint32_t, uint32_t, uint32_t,
                                              uint32_t, uint32_t);
        plow_cpu_amx_pack_b_strip(P, W, N, K, n0, k0, kp);
        int bad = 0;
        for (uint32_t kb = 0; kb < kp / 32; kb++)
            for (uint32_t t = 0; t < 2; t++)
                for (uint32_t r = 0; r < 16; r++)
                    for (uint32_t c = 0; c < 32; c++) {
                        const uint32_t n = n0 + t * 16 + c / 2, k = k0 + kb * 32 + 2 * r + (c & 1);
                        const plow_bf16 want = n < N ? W[n * K + k] : 0;
                        const plow_bf16 got = P[((kb * 2 + t) * 16 + r) * 32 + c];
                        if (got != want) bad++;
                    }
        CHECK(bad == 0, "strip pack: %d mismatches", bad);
        free(W); free(P);
    }

    const uint16_t ops[] = {PLOW_DOP_GEMM, PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED, PLOW_DOP_GEMM_WIDE,
                            PLOW_DOP_GEMM_C5, PLOW_DOP_GEMM_NORM, PLOW_DOP_GEMM_GLU};
    const uint32_t Ms[] = {5, 16, 128, 512};
    /* Golden is scalar: keep the reference work bounded (largest ~ 16 GFLOP). */
    for (size_t o = 0; o < sizeof ops / sizeof ops[0]; o++) {
        for (size_t mi = 0; mi < 4; mi++) test_shape(ops[o], Ms[mi], 3840, 3840, &ctx);
        test_shape(ops[o], 16, 4096, 3840, &ctx);
        test_shape(ops[o], 128, 4096, 3840, &ctx);
        test_shape(ops[o], 16, 3840, 15360, &ctx);
        test_shape(ops[o], 37, 15360, 3840, &ctx); /* odd M, K-panel tail, N strips */
    }
    /* Odd geometry: N not a multiple of 32, M partial second tile. */
    test_shape(PLOW_DOP_GEMM_SMALL, 21, 100, 128, &ctx);
    test_shape(PLOW_DOP_GEMM_GLU, 21, 100, 128, &ctx);
    test_shape(PLOW_DOP_GEMM_NORM, 33, 64, 2048 + 32, &ctx);
    test_gemv_amx(4, 512, 3840, 0, &ctx);
    test_gemv_amx(8, 1000, 3840, 1, &ctx);   /* ragged slices: 16- and tail-row groups */
    test_gemv_amx(5, 37, 1024, 1, &ctx);     /* all-tail slices */
    test_gemv_amx(16, 256, 2048, 0, &ctx);
    test_gemv_glu_amx(8, 500, 3840, 0, &ctx);
    test_gemv_glu_amx(6, 100, 1024, 1, &ctx);
    test_gemv_qkv_amx(8, 512, 128, 128, 3840, &ctx);
    test_gemv_qkv_amx(4, 100, 60, 60, 1024, &ctx);
    test_gemv_fp8_amx(8, 500, 3840, &ctx);
    test_gemv_fp8_amx(5, 37, 1024, &ctx);
    test_gemv_glu_fp8_amx(8, 300, 3840, 0, &ctx);
    test_gemv_glu_fp8_amx(6, 100, 1024, 1, &ctx);
    test_gemv_mx4_amx(8, 500, 3840, 0, 0, &ctx);
    test_gemv_mx4_amx(5, 37, 1024, 0, 0, &ctx);
    test_gemv_mx4_amx(8, 300, 3840, 1, 0, &ctx);
    test_gemv_mx4_amx(6, 100, 2048, 1, 1, &ctx);

    free(ctx.scratch);
    printf(fails ? "cpu_dev_amx_test: %d FAILURES\n" : "cpu_dev_amx_test: all passed\n", fails);
    return fails ? 1 : 0;
}
