/* cpu_dev_fp8_test — fp8 (e4m3, w8a16) weight family: golden vs a naive f32 reference, then
 * the AVX-512 / AMX tiers vs golden, over all slices at Gemma shapes.
 *
 *   ./cpu_dev_fp8_test            run the comparisons (exit 1 on mismatch)
 *   ./cpu_dev_fp8_test --bench    GEMV_FP8 N=15360 K=3840 M=1, 1 thread: GB/s of weight bytes */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev.h"
#include "cpu_dev_internal.h"
#include "fp8_common.h"
#include "golden/fp8.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((float)(rng >> 40) / (float)(1u << 24)) * 2.0f - 1.0f;
}
static void fill_bf16(plow_bf16* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(frand() * scale);
}
/* Random finite e4m3 codes: never NaN (0x7f / 0xff) and never a NONZERO subnormal (e == 0,
 * m != 0): the vector tiers decode those 7 codes as 0 (fp8_common.h — 1e-4 of real fp8 weights,
 * |v| <= 7 * 2^-9), while golden keeps them exact. +-0 stays in the mix (exact everywhere). */
static void fill_fp8(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint8_t b = (uint8_t)(rng >> 56);
        if ((b & 0x7f) == 0x7f) b &= 0xfe;
        if ((b & 0x78) == 0 && (b & 0x07) != 0) b |= 0x08; /* subnormal -> smallest normal */
        p[i] = b;
    }
}
static void fill_scale(float* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = 0.001f + 0.02f * (frand() * 0.5f + 0.5f);
}

static void* xmalloc(size_t n) {
    void* p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(2); }
    memset(p, 0, (n + 63) & ~(size_t)63);
    return p;
}

static int fails = 0;
static float g_floor = 1e-2f; /* relative-error denominator floor; GLU outputs (act(g)*u near 0) use 0.1 */
static void cmp(const char* what, const plow_bf16* a, const plow_bf16* b, size_t n, float tol) {
    double maxrel = 0; size_t bad = 0, where = 0;
    for (size_t i = 0; i < n; i++) {
        const float x = plow_bf2f(a[i]), y = plow_bf2f(b[i]);
        if (isnan(x) || isnan(y)) { if (isnan(x) != isnan(y)) { bad++; where = i; } continue; }
        const float d = fabsf(x - y), ref = fmaxf(fabsf(x), g_floor);
        const double rel = d / ref;
        if (rel > maxrel) { maxrel = rel; where = i; }
        if (rel > tol) bad++;
    }
    printf("  %-44s max rel %.4f  bad %zu/%zu%s\n", what, maxrel, bad, n, bad ? "  <-- FAIL" : "");
    if (bad) { fails++; printf("    e.g. [%zu] a=%g b=%g\n", where, plow_bf2f(a[where]), plow_bf2f(b[where])); }
}

typedef void (*kfn)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);

static void run_all(kfn f, const PlowDevInst* in, uint32_t nblk, void* const* T, PlowCpuCtx* ctx) {
    for (uint32_t s = 0; s < nblk; s++) f(in, s, nblk, T, ctx);
}

static PlowCpuCtx mkctx(void) {
    PlowCpuCtx c; memset(&c, 0, sizeof(c));
    c.scratch_bytes = plow_cpu_scratch_bytes();
    c.scratch = xmalloc(c.scratch_bytes);
    plow_cpu_thread_init(&c);
    return c;
}

/* ---- naive f32 references ---- */
static void ref_gemv(plow_bf16* C, const plow_bf16* x, const uint8_t* W, const float* ws,
                     uint32_t M, uint32_t N, uint32_t K, uint32_t a_row0) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            double acc = 0;
            for (uint32_t k = 0; k < K; k++)
                acc += (double)plow_bf2f(x[(size_t)(a_row0 + m) * K + k]) * plow_e4m3_to_f32(W[(size_t)n * K + k]);
            C[(size_t)m * N + n] = plow_f2bf((float)acc * ws[n]);
        }
}
static float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static float silu(float x) { return x / (1.0f + expf(-x)); }

static void ref_glu(plow_bf16* C, const plow_bf16* x, const uint8_t* Wg, const uint8_t* Wu,
                    const float* gs, const float* us, uint32_t M, uint32_t N, uint32_t K, uint32_t act) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            double g = 0, u = 0;
            for (uint32_t k = 0; k < K; k++) {
                const double xv = plow_bf2f(x[(size_t)m * K + k]);
                g += xv * plow_e4m3_to_f32(Wg[(size_t)n * K + k]);
                u += xv * plow_e4m3_to_f32(Wu[(size_t)n * K + k]);
            }
            const float gv = (float)g * gs[n], uv = (float)u * us[n];
            C[(size_t)m * N + n] = plow_f2bf((act ? silu(gv) : gelu_tanh(gv)) * uv);
        }
}

static const uint32_t NBLKS[] = {1, 3, 16};

static void test_gemv(uint32_t M, uint32_t N, uint32_t K, uint32_t a_row0, int have_v) {
    plow_bf16* x = xmalloc((size_t)(M + a_row0) * K * 2);
    uint8_t* W = xmalloc((size_t)N * K);
    float* ws = xmalloc((size_t)N * 4);
    plow_bf16* Cref = xmalloc((size_t)M * N * 2);
    plow_bf16* Cg = xmalloc((size_t)M * N * 2);
    plow_bf16* Cv = xmalloc((size_t)M * N * 2);
    fill_bf16(x, (size_t)(M + a_row0) * K, 1.0f); fill_fp8(W, (size_t)N * K); fill_scale(ws, N);
    void* T[8] = {Cg, x, W, NULL, NULL, ws, NULL, NULL};
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = PLOW_DOP_GEMV_FP8; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[4] = a_row0;
    PlowCpuCtx ctx = mkctx();
    ref_gemv(Cref, x, W, ws, M, N, K, a_row0);
    char what[128];
    for (size_t bi = 0; bi < sizeof(NBLKS) / sizeof(*NBLKS); bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(g_gemv_fp8, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "GEMV_FP8 golden M=%u N=%u K=%u r0=%u nblk=%u", M, N, K, a_row0, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 2e-2f);
        if (have_v) {
            T[0] = Cv; memset(Cv, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMV_FP8), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "GEMV_FP8 avx512 vs golden nblk=%u", nblk);
            cmp(what, Cg, Cv, (size_t)M * N, 1e-2f);
        }
    }
    free(x); free(W); free(ws); free(Cref); free(Cg); free(Cv); free(ctx.scratch);
}

static void test_glu(uint32_t M, uint32_t N, uint32_t K, uint32_t act, int have_v) {
    plow_bf16* x = xmalloc((size_t)M * K * 2);
    uint8_t* Wg = xmalloc((size_t)N * K); uint8_t* Wu = xmalloc((size_t)N * K);
    float* gs = xmalloc((size_t)N * 4); float* us = xmalloc((size_t)N * 4);
    plow_bf16* Cref = xmalloc((size_t)M * N * 2); plow_bf16* Cg = xmalloc((size_t)M * N * 2); plow_bf16* Cv = xmalloc((size_t)M * N * 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_fp8(Wg, (size_t)N * K); fill_fp8(Wu, (size_t)N * K); fill_scale(gs, N); fill_scale(us, N);
    void* T[8] = {Cg, x, Wg, gs, us, Wu, NULL, NULL};
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = PLOW_DOP_GEMV_GLU_FP8; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.t[4] = 4; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = act;
    PlowCpuCtx ctx = mkctx();
    ref_glu(Cref, x, Wg, Wu, gs, us, M, N, K, act);
    g_floor = 0.1f;
    char what[128];
    for (size_t bi = 0; bi < sizeof(NBLKS) / sizeof(*NBLKS); bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(g_gemv_glu_fp8, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "GEMV_GLU_FP8 golden M=%u N=%u K=%u act=%u nblk=%u", M, N, K, act, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 3e-2f);
        if (have_v) {
            T[0] = Cv; memset(Cv, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMV_GLU_FP8), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "GEMV_GLU_FP8 avx512 vs golden nblk=%u", nblk);
            cmp(what, Cg, Cv, (size_t)M * N, 2e-2f);
        }
    }
    g_floor = 1e-2f;
    free(x); free(Wg); free(Wu); free(gs); free(us); free(Cref); free(Cg); free(Cv); free(ctx.scratch);
}

/* GEMM_FP8 (w8a16: t3 absent) — golden vs the GEMV reference (same math, M rows), then AMX. */
static void test_gemm(uint16_t op, kfn golden, uint32_t M, uint32_t N, uint32_t K, int have_x) {
    plow_bf16* A = xmalloc((size_t)M * K * 2);
    uint8_t* W = xmalloc((size_t)N * K);
    float* ws = xmalloc((size_t)N * 4);
    plow_bf16* Cref = xmalloc((size_t)M * N * 2); plow_bf16* Cg = xmalloc((size_t)M * N * 2); plow_bf16* Cx = xmalloc((size_t)M * N * 2);
    fill_bf16(A, (size_t)M * K, 1.0f); fill_fp8(W, (size_t)N * K); fill_scale(ws, N);
    void* T[8] = {Cg, A, W, NULL, ws, NULL, NULL, NULL};
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = op; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[4] = 4;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    ref_gemv(Cref, A, W, ws, M, N, K, 0);
    char what[128];
    for (size_t bi = 0; bi < sizeof(NBLKS) / sizeof(*NBLKS); bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(golden, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "op%u golden M=%u N=%u K=%u nblk=%u", op, M, N, K, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 2e-2f);
        if (have_x && plow_cpu_tier_of(op) == PLOW_CPU_ISA_AMX) {
            T[0] = Cx; memset(Cx, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(op), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "op%u amx vs golden nblk=%u", op, nblk);
            cmp(what, Cg, Cx, (size_t)M * N, 1e-2f);
        }
    }
    free(A); free(W); free(ws); free(Cref); free(Cg); free(Cx); free(ctx.scratch);
}

static void test_gemm_glu(uint32_t M, uint32_t N, uint32_t K, uint32_t act, int have_x) {
    plow_bf16* A = xmalloc((size_t)M * K * 2);
    uint8_t* Wg = xmalloc((size_t)N * K); uint8_t* Wu = xmalloc((size_t)N * K);
    float* gs = xmalloc((size_t)N * 4); float* us = xmalloc((size_t)N * 4);
    plow_bf16* Cref = xmalloc((size_t)M * N * 2); plow_bf16* Cg = xmalloc((size_t)M * N * 2); plow_bf16* Cx = xmalloc((size_t)M * N * 2);
    fill_bf16(A, (size_t)M * K, 1.0f); fill_fp8(Wg, (size_t)N * K); fill_fp8(Wu, (size_t)N * K); fill_scale(gs, N); fill_scale(us, N);
    void* T[8] = {Cg, A, Wg, NULL, gs, Wu, us, NULL};
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = PLOW_DOP_GEMM_GLU_FP8; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[4] = 4; in.t[5] = 5; in.t[6] = 6;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = act;
    PlowCpuCtx ctx = mkctx();
    ref_glu(Cref, A, Wg, Wu, gs, us, M, N, K, act);
    g_floor = 0.1f;
    char what[128];
    for (size_t bi = 0; bi < sizeof(NBLKS) / sizeof(*NBLKS); bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(g_gemm_glu_fp8, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "GEMM_GLU_FP8 golden M=%u N=%u K=%u act=%u nblk=%u", M, N, K, act, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 3e-2f);
        if (have_x && plow_cpu_tier_of(PLOW_DOP_GEMM_GLU_FP8) == PLOW_CPU_ISA_AMX) {
            T[0] = Cx; memset(Cx, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMM_GLU_FP8), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "GEMM_GLU_FP8 amx vs golden nblk=%u", nblk);
            cmp(what, Cg, Cx, (size_t)M * N, 2e-2f);
        }
    }
    g_floor = 1e-2f;
    free(A); free(Wg); free(Wu); free(gs); free(us); free(Cref); free(Cg); free(Cx); free(ctx.scratch);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void bench(void) {
    const uint32_t M = 1, N = 15360, K = 3840;
    plow_bf16* x = xmalloc((size_t)K * 2); uint8_t* W = xmalloc((size_t)N * K); float* ws = xmalloc((size_t)N * 4);
    plow_bf16* C = xmalloc((size_t)N * 2);
    fill_bf16(x, K, 1.0f); fill_fp8(W, (size_t)N * K); fill_scale(ws, N);
    void* T[8] = {C, x, W, NULL, NULL, ws, NULL, NULL};
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = PLOW_DOP_GEMV_FP8; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5; in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_FP8);
    f(&in, 0, 1, T, &ctx);
    double best = 1e9;
    for (int r = 0; r < 5; r++) { double t0 = now(); f(&in, 0, 1, T, &ctx); double dt = now() - t0; if (dt < best) best = dt; }
    printf("bench GEMV_FP8 tier %d M=1 N=%u K=%u: %.2f ms, %.1f GB/s weight bytes (%zu MB)\n",
           plow_cpu_tier_of(PLOW_DOP_GEMV_FP8), N, K, best * 1e3, (double)N * K / best / 1e9, (size_t)N * K >> 20);
    double t0 = now(); g_gemv_fp8(&in, 0, 1, T, &ctx); double tg = now() - t0;
    printf("      golden: %.2f ms (%.1fx)\n", tg * 1e3, tg / best);
}

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AMX);
    printf("isa tier: %d\n", tier);
    if (tier < 0) return 2;
    if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(); return 0; }
    const int have_v = tier >= PLOW_CPU_ISA_AVX512, have_x = tier >= PLOW_CPU_ISA_AMX;
    for (uint16_t op = PLOW_DOP_GEMV_FP8; op <= PLOW_DOP_GEMM_GLU_FP8; op++)
        if (op != PLOW_DOP_QUANT_FP8 && !plow_cpu_has(op)) { printf("missing op %u\n", op); fails++; }
    printf("tiers: gemv_fp8=%d gemv_glu_fp8=%d gemm_fp8=%d gemm_glu_fp8=%d\n",
           plow_cpu_tier_of(PLOW_DOP_GEMV_FP8), plow_cpu_tier_of(PLOW_DOP_GEMV_GLU_FP8),
           plow_cpu_tier_of(PLOW_DOP_GEMM_FP8), plow_cpu_tier_of(PLOW_DOP_GEMM_GLU_FP8));
    test_gemv(1, 4096, 3840, 0, have_v);
    test_gemv(1, 2048, 3872, 0, have_v);   /* K tail */
    test_gemv(4, 3840, 15360, 0, have_v);
    test_gemv(8, 1024, 3840, 3, have_v);   /* a_row0 */
    test_gemv(1, 100, 3840, 0, have_v);    /* N tail vs RB */
    test_glu(1, 2048, 3840, 0, have_v);
    test_glu(3, 1024, 3840, 1, have_v);
    test_gemm(PLOW_DOP_GEMM_FP8, g_gemm_fp8, 128, 4096, 3840, have_x);
    test_gemm(PLOW_DOP_GEMM_FP8, g_gemm_fp8, 37, 512, 3840, have_x);
    test_gemm(PLOW_DOP_GEMM_MED_FP8, g_gemm_med_fp8, 128, 2048, 3840, have_x);
    test_gemm(PLOW_DOP_GEMM_SMALL_FP8, g_gemm_small_fp8, 5, 256, 3840, have_x);
    test_gemm(PLOW_DOP_GEMM_WIDE_FP8, g_gemm_wide_fp8, 512, 1024, 3840, have_x);
    test_gemm_glu(128, 1024, 3840, 0, have_x);
    test_gemm_glu(37, 512, 3840, 1, have_x);
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
