/* cpu_dev_neon_gemm_test.c — NEON tier GEMM family against the golden tier.
 *
 * The GEMM shape list of cpu_dev_amx_test.c (minus GEMM_NORM, which the NEON tier does not
 * override): every op runs over ALL slices for nblk in {1, 3, 16}, compared element-wise to
 * the golden kernel (1e-2 relative + 1e-2 absolute). `--bench`: single-thread GFLOPS. */
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
        case PLOW_DOP_GEMM_GLU: return "GEMM_GLU";
        default: return "?";
    }
}

/* Plain GEMM ops (+ a_row0/c_row0 + bias) and GEMM_GLU at one shape. */
static void test_shape(uint16_t op, uint32_t M, uint32_t N, uint32_t K, int bias, PlowCpuCtx* ctx) {
    const uint32_t a_row0 = (op == PLOW_DOP_GEMM_SMALL && M > 8) ? 3 : 0;
    const uint32_t c_row0 = (op == PLOW_DOP_GEMM_MED) ? 2 : 0;
    const int glu = op == PLOW_DOP_GEMM_GLU;
    plow_bf16* A = malloc((size_t)(M + a_row0) * K * 2);
    plow_bf16* W = malloc((size_t)N * K * 2);
    plow_bf16* Wu = glu ? malloc((size_t)N * K * 2) : NULL;
    plow_bf16* Bv = bias ? malloc((size_t)N * 2) : NULL;
    plow_bf16* C = malloc((size_t)(M + c_row0) * N * 2);
    plow_bf16* Cg = malloc((size_t)(M + c_row0) * N * 2);
    fill_bf16(A, (size_t)(M + a_row0) * K, 1.0f);
    fill_bf16(W, (size_t)N * K, 0.05f);
    if (glu) fill_bf16(Wu, (size_t)N * K, 0.05f);
    if (bias) fill_bf16(Bv, N, 0.5f);
    void* T[8] = {C, A, W, NULL, NULL, Wu, NULL, Bv};
    PlowDevInst in = inst(op);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (glu) { in.t[5] = 5; in.i[5] = 1; /* silu */ }
    if (bias) in.t[7] = 7;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    if (!glu) { in.i[4] = a_row0; in.i[5] = c_row0; }

    memset(C, 0, (size_t)(M + c_row0) * N * 2);
    run_all(golden_of(op), &in, 1, T, ctx);
    memcpy(Cg, C, (size_t)(M + c_row0) * N * 2);
    kfn f = plow_cpu_kernel(op);
    CHECK(f != NULL && f != golden_of(op), "%s: NEON kernel not registered", name_of(op));
    for (int k = 0; k < 3; k++) {
        memset(C, 0, (size_t)(M + c_row0) * N * 2);
        const double t0 = now();
        run_all(f, &in, NBLKS[k], T, ctx);
        const double dt = now() - t0;
        char what[128];
        snprintf(what, sizeof what, "%s M=%u N=%u K=%u bias=%d nblk=%u", name_of(op), M, N, K, bias, NBLKS[k]);
        compare(what, C, Cg, (size_t)(M + c_row0) * N);
        if (k == 0)
            printf("  %-44s %.1f ms  %.1f GFLOPS (1 thread, all slices)\n", what, dt * 1e3,
                   2.0 * M * N * K * (glu ? 2 : 1) / dt / 1e9);
    }
    free(A); free(W); free(Wu); free(Bv); free(C); free(Cg);
}

static void bench_gemm(uint16_t op, uint32_t M, uint32_t N, uint32_t K, PlowCpuCtx* ctx) {
    const int glu = op == PLOW_DOP_GEMM_GLU;
    plow_bf16* A = malloc((size_t)M * K * 2);
    plow_bf16* W = malloc((size_t)N * K * 2);
    plow_bf16* Wu = glu ? malloc((size_t)N * K * 2) : NULL;
    plow_bf16* C = malloc((size_t)M * N * 2);
    fill_bf16(A, (size_t)M * K, 1.0f);
    fill_bf16(W, (size_t)N * K, 0.05f);
    if (glu) fill_bf16(Wu, (size_t)N * K, 0.05f);
    void* T[8] = {C, A, W, NULL, NULL, Wu, NULL, NULL};
    PlowDevInst in = inst(op);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (glu) { in.t[5] = 5; in.i[5] = 1; }
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    kfn f = plow_cpu_kernel(op);
    run_all(f, &in, 16, T, ctx);
    double best = 1e9;
    for (int r = 0; r < 4; r++) {
        const double t0 = now();
        run_all(f, &in, 16, T, ctx);
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    const double flops = 2.0 * M * N * K * (glu ? 2 : 1);
    printf("bench %-9s M=%u N=%-5u K=%u: %7.2f ms  %6.1f GFLOPS (1 thread, 16 slices)\n",
           name_of(op), M, N, K, best * 1e3, flops / best / 1e9);
    free(A); free(W); free(Wu); free(C);
}

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_NEON);
    if (tier < PLOW_CPU_ISA_NEON) {
        printf("no NEON bf16 on this host (tier %d) — skipping\n", tier);
        return 0;
    }
    PlowCpuCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.scratch_bytes = plow_cpu_scratch_bytes();
    ctx.scratch = aligned_alloc(64, ctx.scratch_bytes);
    memset(ctx.scratch, 0, ctx.scratch_bytes);
    CHECK(plow_cpu_thread_init(&ctx) == 0, "thread init");
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench_gemm(PLOW_DOP_GEMM, 512, 3840, 3840, &ctx);
        bench_gemm(PLOW_DOP_GEMM_WIDE, 128, 3072, 3072, &ctx);
        bench_gemm(PLOW_DOP_GEMM_GLU, 512, 8192, 3072, &ctx);
        return 0;
    }
    const uint16_t ops[] = {PLOW_DOP_GEMM, PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED, PLOW_DOP_GEMM_WIDE,
                            PLOW_DOP_GEMM_C5, PLOW_DOP_GEMM_GLU};
    const uint32_t Ms[] = {5, 16, 128, 512};
    for (size_t o = 0; o < sizeof ops / sizeof ops[0]; o++) {
        for (size_t mi = 0; mi < 4; mi++) test_shape(ops[o], Ms[mi], 3840, 3840, 0, &ctx);
        test_shape(ops[o], 16, 4096, 3840, 1, &ctx);
        test_shape(ops[o], 37, 15360, 3840, 0, &ctx); /* odd M, K tail, N tail */
    }
    test_shape(PLOW_DOP_GEMM_SMALL, 21, 100, 128 + 3, 1, &ctx);
    test_shape(PLOW_DOP_GEMM_GLU, 21, 100, 128 + 3, 1, &ctx);
    if (fails == 0) printf("cpu_dev_neon_gemm_test: all passed (tier %d)\n", tier);
    else printf("cpu_dev_neon_gemm_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
