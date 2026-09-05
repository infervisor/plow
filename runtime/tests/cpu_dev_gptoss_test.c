/* cpu_dev_gptoss_test — GPT-OSS CPU kernels: MXFP4 GEMV / MoE ops (147-150), bias epilogues
 * (GEMV/GEMM/GLU families, GEMV_QKV handles), sinks merge, half-split RoPE at hd=64, swiglu_oai.
 * Golden vs a naive f64 reference, then AVX-512 (and AMX where a tier exists) vs golden, over all
 * slices at GPT-OSS shapes.
 *
 *   ./cpu_dev_gptoss_test            run the comparisons (exit 1 on mismatch)
 *   ./cpu_dev_gptoss_test --bench    GEMV_MXFP4 N=5760 K=2880 M=1, 1 thread: GB/s of packed bytes */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev.h"
#include "cpu_dev_internal.h"
#include "mxfp4_common.h"
#include "golden/gptoss.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float frand(void) { return ((float)(next64() >> 40) / (float)(1u << 24)) * 2.0f - 1.0f; }
static void fill_bf16(plow_bf16* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(frand() * scale);
}
static void fill_f32(float* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = frand() * scale;
}
/* Random fp4 nibbles (all 16 codes, incl. -0) and realistic E8M0 scales: 2^-9 .. 2^-4 (a
 * checkpoint block of |w| <= ~0.4 has max 6 * 2^e), 1 in 64 blocks tiny (2^-20) so the
 * per-block scale path is exercised without one block dominating the row sum. */
static void fill_fp4(uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(next64() >> 56); }
static void fill_e8m0(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = (uint32_t)(next64() >> 40);
        p[i] = (uint8_t)(r % 64u == 0u ? 107u : 118u + (r >> 8) % 6u);
    }
}

static void* xmalloc(size_t n) {
    void* p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(2); }
    memset(p, 0, (n + 63) & ~(size_t)63);
    return p;
}

static int fails = 0;
static float g_floor = 1e-2f; /* relative-error denominator floor */
static void cmp(const char* what, const plow_bf16* a, const plow_bf16* b, size_t n, float tol) {
    double maxrel = 0, mean = 0; size_t bad = 0, where = 0;
    for (size_t i = 0; i < n; i++) {
        const float x = plow_bf2f(a[i]), y = plow_bf2f(b[i]);
        if (isnan(x) || isnan(y)) { if (isnan(x) != isnan(y)) { bad++; where = i; } continue; }
        mean += fabsf(x);
        const float d = fabsf(x - y), ref = fmaxf(fabsf(x), g_floor);
        const double rel = d / ref;
        if (rel > maxrel) { maxrel = rel; where = i; }
        if (rel > tol) bad++;
    }
    /* A reference that is (almost) all zero would pass trivially: flag it. */
    if (n && mean / (double)n < 1e-3) { bad++; where = 0; }
    printf("  %-62s mean|a| %.3g max rel %.4f  bad %zu/%zu%s\n", what, mean / (double)(n ? n : 1), maxrel, bad, n, bad ? "  <-- FAIL" : "");
    if (bad) { fails++; printf("    e.g. [%zu] a=%g b=%g\n", where, plow_bf2f(a[where]), plow_bf2f(b[where])); }
}
static void cmp_f32(const char* what, const float* a, const float* b, size_t n, float tol) {
    double maxrel = 0, mean = 0; size_t bad = 0, where = 0;
    for (size_t i = 0; i < n; i++) {
        mean += fabsf(a[i]);
        const float d = fabsf(a[i] - b[i]), ref = fmaxf(fabsf(a[i]), g_floor);
        const double rel = d / ref;
        if (rel > maxrel) { maxrel = rel; where = i; }
        if (rel > tol || isnan(d)) bad++;
    }
    if (n && mean / (double)n < 1e-3) { bad++; where = 0; }
    printf("  %-62s mean|a| %.3g max rel %.4f  bad %zu/%zu%s\n", what, mean / (double)(n ? n : 1), maxrel, bad, n, bad ? "  <-- FAIL" : "");
    if (bad) { fails++; printf("    e.g. [%zu] a=%g b=%g\n", where, a[where], b[where]); }
}
static void check(const char* what, int ok) {
    printf("  %-62s %s\n", what, ok ? "ok" : "<-- FAIL");
    if (!ok) fails++;
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
static PlowDevInst inst(uint16_t op) {
    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = op; for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    return in;
}
static const uint32_t NBLKS[] = {1, 3, 16};
#define NNB (sizeof(NBLKS) / sizeof(*NBLKS))
static int have_v = 0, have_x = 0;
#define ALPHA 1.702f
#define LIMIT 7.0f

/* ---- f64 references ---- */
static double mx_at(const uint8_t* w, uint32_t k) {
    static const double lut[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
    const uint8_t b = w[k >> 1];
    return lut[(k & 1u) ? (b >> 4) : (b & 0xFu)];
}
static double e8m0(uint8_t s) { return ldexp(1.0, (int)s - 127); }
/* Row n of an [N][K] MXFP4 matrix dotted with x (any K, block 32). */
static double ref_mx_dot(const uint8_t* W, const uint8_t* S, uint32_t n, uint32_t K, const plow_bf16* x) {
    const uint8_t* w = W + (size_t)n * (K / 2u);
    const uint8_t* s = S + (size_t)n * (K / 32u);
    double acc = 0;
    for (uint32_t k = 0; k < K; k++) acc += mx_at(w, k) * e8m0(s[k / 32u]) * (double)plow_bf2f(x[k]);
    return acc;
}
static double ref_dot(const plow_bf16* a, const plow_bf16* b, uint32_t K) {
    double acc = 0;
    for (uint32_t k = 0; k < K; k++) acc += (double)plow_bf2f(a[k]) * (double)plow_bf2f(b[k]);
    return acc;
}
static double ref_swiglu(double g, double u) {
    g = fmin(g, LIMIT); u = fmin(fmax(u, -LIMIT), LIMIT);
    return g / (1.0 + exp(-ALPHA * g)) * (u + 1.0);
}

/* ---- GEMV_MXFP4 (91) ---- */
static void test_gemv_mxfp4(uint32_t M, uint32_t N, uint32_t K) {
    plow_bf16* x = xmalloc((size_t)M * K * 2);
    uint8_t* W = xmalloc((size_t)N * K / 2);
    uint8_t* S = xmalloc((size_t)N * K / 32);
    plow_bf16 *Cref = xmalloc((size_t)M * N * 2), *Cg = xmalloc((size_t)M * N * 2), *Cv = xmalloc((size_t)M * N * 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_fp4(W, (size_t)N * K / 2); fill_e8m0(S, (size_t)N * K / 32);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) Cref[(size_t)m * N + n] = plow_f2bf((float)ref_mx_dot(W, S, n, K, x + (size_t)m * K));
    void* T[8] = {Cg, x, W, S, NULL, NULL, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_MXFP4);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(g_gemv_mxfp4, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "GEMV_MXFP4 golden M=%u N=%u K=%u nblk=%u", M, N, K, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 1e-2f);
        if (have_v) {
            T[0] = Cv; memset(Cv, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMV_MXFP4), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "GEMV_MXFP4 avx512 vs golden M=%u nblk=%u", M, nblk);
            cmp(what, Cg, Cv, (size_t)M * N, 1e-2f);
        }
    }
    free(x); free(W); free(S); free(Cref); free(Cg); free(Cv); free(ctx.scratch);
}

/* ---- swiglu_oai helper vs the HF formula in f64 ---- */
static void test_swiglu(void) {
    double maxd = 0;
    for (int i = 0; i < 4096; i++) {
        const float g = frand() * 12.0f, u = frand() * 12.0f;
        const double ref = ref_swiglu(g, u);
        const double d = fabs(ref - (double)g_swiglu_oai(g, u, ALPHA, LIMIT)) / fmax(fabs(ref), 1e-2);
        if (d > maxd) maxd = d;
    }
    printf("  %-62s max rel %.6f%s\n", "swiglu_oai golden vs f64", maxd, maxd > 1e-4 ? "  <-- FAIL" : "");
    if (maxd > 1e-4) fails++;
}

/* ---- GLU (5) act=3 ---- */
static void test_glu_act3(uint32_t n) {
    plow_bf16 *g = xmalloc(n * 2), *u = xmalloc(n * 2), *ref = xmalloc(n * 2), *out = xmalloc(n * 2), *outv = xmalloc(n * 2);
    fill_bf16(g, n, 10.0f); fill_bf16(u, n, 10.0f);
    for (uint32_t i = 0; i < n; i++) ref[i] = plow_f2bf((float)ref_swiglu(plow_bf2f(g[i]), plow_bf2f(u[i])));
    void* T[8] = {out, g, u, NULL, NULL, NULL, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GLU);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.i[0] = n; in.i[1] = 3; in.fj[0].f = ALPHA; in.fj[1].f = LIMIT;
    PlowCpuCtx ctx = mkctx();
    g_floor = 0.05f;
    for (size_t bi = 0; bi < NNB; bi++) {
        memset(out, 0, n * 2);
        run_all(g_glu, &in, NBLKS[bi], T, &ctx);
        char what[160];
        snprintf(what, sizeof what, "GLU act=3 golden n=%u nblk=%u", n, NBLKS[bi]);
        cmp(what, ref, out, n, 1e-2f);
        if (have_v) {
            T[0] = outv; memset(outv, 0, n * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GLU), &in, NBLKS[bi], T, &ctx);
            T[0] = out;
            snprintf(what, sizeof what, "GLU act=3 avx512 vs golden nblk=%u", NBLKS[bi]);
            cmp(what, out, outv, n, 1e-2f);
        }
    }
    g_floor = 1e-2f;
    free(g); free(u); free(ref); free(out); free(outv); free(ctx.scratch);
}

/* ---- GEMV (10) / GEMM family with t7 bias ---- */
static void test_gemv_bias(uint32_t M, uint32_t N, uint32_t K) {
    plow_bf16* x = xmalloc((size_t)M * K * 2);
    plow_bf16* W = xmalloc((size_t)N * K * 2);
    plow_bf16* bias = xmalloc((size_t)N * 2);
    plow_bf16 *Cref = xmalloc((size_t)M * N * 2), *Cg = xmalloc((size_t)M * N * 2), *Cv = xmalloc((size_t)M * N * 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_bf16(W, (size_t)N * K, 0.05f); fill_bf16(bias, N, 2.0f);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++)
            Cref[(size_t)m * N + n] = plow_f2bf((float)(ref_dot(x + (size_t)m * K, W + (size_t)n * K, K) + plow_bf2f(bias[n])));
    void* T[8] = {Cg, x, W, NULL, NULL, NULL, NULL, bias};
    PlowDevInst in = inst(PLOW_DOP_GEMV);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[7] = 7;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(g_gemv, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "GEMV+bias golden M=%u N=%u K=%u nblk=%u", M, N, K, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 1e-2f);
        if (have_v) {
            T[0] = Cv; memset(Cv, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMV), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "GEMV+bias avx512 vs golden M=%u nblk=%u", M, nblk);
            cmp(what, Cg, Cv, (size_t)M * N, 1e-2f);
        }
    }
    free(x); free(W); free(bias); free(Cref); free(Cg); free(Cv); free(ctx.scratch);
}

static void test_gemm_bias(uint16_t op, kfn golden, uint32_t M, uint32_t N, uint32_t K) {
    plow_bf16* A = xmalloc((size_t)M * K * 2);
    plow_bf16* W = xmalloc((size_t)N * K * 2);
    plow_bf16* bias = xmalloc((size_t)N * 2);
    plow_bf16 *Cref = xmalloc((size_t)M * N * 2), *Cg = xmalloc((size_t)M * N * 2), *Cx = xmalloc((size_t)M * N * 2);
    fill_bf16(A, (size_t)M * K, 1.0f); fill_bf16(W, (size_t)N * K, 0.05f); fill_bf16(bias, N, 2.0f);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++)
            Cref[(size_t)m * N + n] = plow_f2bf((float)(ref_dot(A + (size_t)m * K, W + (size_t)n * K, K) + plow_bf2f(bias[n])));
    void* T[8] = {Cg, A, W, NULL, NULL, NULL, NULL, bias};
    PlowDevInst in = inst(op);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[7] = 7;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(golden, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "op%u+bias golden M=%u N=%u K=%u nblk=%u", op, M, N, K, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 1e-2f);
        if (have_x && plow_cpu_tier_of(op) == PLOW_CPU_ISA_AMX) {
            T[0] = Cx; memset(Cx, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(op), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "op%u+bias amx vs golden nblk=%u", op, nblk);
            cmp(what, Cg, Cx, (size_t)M * N, 1e-2f);
        }
    }
    free(A); free(W); free(bias); free(Cref); free(Cg); free(Cx); free(ctx.scratch);
}

/* ---- GEMV_GLU (19) / GEMM_GLU (20): t6/t7 biases + act 3 ---- */
static void test_glu_proj(uint16_t op, kfn golden, uint32_t M, uint32_t N, uint32_t K) {
    plow_bf16* x = xmalloc((size_t)M * K * 2);
    plow_bf16 *Wg = xmalloc((size_t)N * K * 2), *Wu = xmalloc((size_t)N * K * 2);
    plow_bf16 *bg = xmalloc((size_t)N * 2), *bu = xmalloc((size_t)N * 2);
    plow_bf16 *Cref = xmalloc((size_t)M * N * 2), *Cg = xmalloc((size_t)M * N * 2), *Cf = xmalloc((size_t)M * N * 2);
    fill_bf16(x, (size_t)M * K, 1.0f); fill_bf16(Wg, (size_t)N * K, 0.05f); fill_bf16(Wu, (size_t)N * K, 0.05f);
    fill_bf16(bg, N, 3.0f); fill_bf16(bu, N, 3.0f);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            const double g = ref_dot(x + (size_t)m * K, Wg + (size_t)n * K, K) + plow_bf2f(bg[n]);
            const double u = ref_dot(x + (size_t)m * K, Wu + (size_t)n * K, K) + plow_bf2f(bu[n]);
            Cref[(size_t)m * N + n] = plow_f2bf((float)ref_swiglu(g, u));
        }
    void* T[8] = {Cg, x, Wg, NULL, NULL, Wu, bg, bu};
    PlowDevInst in = inst(op);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5; in.t[6] = 6; in.t[7] = 7;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = 3; in.fj[0].f = ALPHA; in.fj[1].f = LIMIT;
    PlowCpuCtx ctx = mkctx();
    const int fast = plow_cpu_tier_of(op) > PLOW_CPU_ISA_SCALAR;
    g_floor = 0.05f;
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(Cg, 0, (size_t)M * N * 2);
        run_all(golden, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "op%u act=3+bias golden M=%u N=%u K=%u nblk=%u", op, M, N, K, nblk);
        cmp(what, Cref, Cg, (size_t)M * N, 1e-2f);
        if (fast) {
            T[0] = Cf; memset(Cf, 0, (size_t)M * N * 2);
            run_all(plow_cpu_kernel(op), &in, nblk, T, &ctx);
            T[0] = Cg;
            snprintf(what, sizeof what, "op%u act=3+bias tier%d vs golden nblk=%u", op, plow_cpu_tier_of(op), nblk);
            cmp(what, Cg, Cf, (size_t)M * N, 1e-2f);
        }
    }
    g_floor = 1e-2f;
    free(x); free(Wg); free(Wu); free(bg); free(bu); free(Cref); free(Cg); free(Cf); free(ctx.scratch);
}

/* ---- GEMV_QKV (22): i5/i6/i7 bias handles ---- */
static void test_qkv_bias(uint32_t M, uint32_t Nq, uint32_t Nk, uint32_t Nv, uint32_t K) {
    plow_bf16* x = xmalloc((size_t)M * K * 2);
    const uint32_t Ns[3] = {Nq, Nk, Nv};
    plow_bf16 *W[3], *b[3], *Cref[3], *Cg[3], *Cv[3];
    for (int s = 0; s < 3; s++) {
        W[s] = xmalloc((size_t)Ns[s] * K * 2); b[s] = xmalloc((size_t)Ns[s] * 2);
        Cref[s] = xmalloc((size_t)M * Ns[s] * 2); Cg[s] = xmalloc((size_t)M * Ns[s] * 2); Cv[s] = xmalloc((size_t)M * Ns[s] * 2);
        fill_bf16(W[s], (size_t)Ns[s] * K, 0.05f); fill_bf16(b[s], Ns[s], 2.0f);
    }
    fill_bf16(x, (size_t)M * K, 1.0f);
    for (int s = 0; s < 3; s++)
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = 0; n < Ns[s]; n++)
                Cref[s][(size_t)m * Ns[s] + n] = plow_f2bf((float)(ref_dot(x + (size_t)m * K, W[s] + (size_t)n * K, K) + plow_bf2f(b[s][n])));
    /* handles 8/9/10 carry the biases (an integer slot may hold a weight-like handle, never 0). */
    void* T[11] = {Cg[0], x, W[0], Cg[1], W[1], Cg[2], W[2], NULL, b[0], b[1], b[2]};
    PlowDevInst in = inst(PLOW_DOP_GEMV_QKV);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.t[4] = 4; in.t[5] = 5; in.t[6] = 6;
    in.i[0] = M; in.i[1] = Nq; in.i[2] = K; in.i[3] = Nk; in.i[4] = Nv; in.i[5] = 8; in.i[6] = 9; in.i[7] = 10;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        for (int s = 0; s < 3; s++) memset(Cg[s], 0, (size_t)M * Ns[s] * 2);
        run_all(g_gemv_qkv, &in, nblk, T, &ctx);
        for (int s = 0; s < 3; s++) {
            snprintf(what, sizeof what, "GEMV_QKV+bias golden M=%u stream %d nblk=%u", M, s, nblk);
            cmp(what, Cref[s], Cg[s], (size_t)M * Ns[s], 1e-2f);
        }
        if (have_v) {
            T[0] = Cv[0]; T[3] = Cv[1]; T[5] = Cv[2];
            for (int s = 0; s < 3; s++) memset(Cv[s], 0, (size_t)M * Ns[s] * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_GEMV_QKV), &in, nblk, T, &ctx);
            T[0] = Cg[0]; T[3] = Cg[1]; T[5] = Cg[2];
            for (int s = 0; s < 3; s++) {
                snprintf(what, sizeof what, "GEMV_QKV+bias avx512 vs golden stream %d nblk=%u", s, nblk);
                cmp(what, Cg[s], Cv[s], (size_t)M * Ns[s], 1e-2f);
            }
        }
    }
    /* i5..i7 = 0 must mean "no bias" (pre-bias packets). */
    in.i[5] = in.i[6] = in.i[7] = 0;
    run_all(g_gemv_qkv, &in, 1, T, &ctx);
    size_t diff = 0;
    for (uint32_t n = 0; n < Nq; n++) diff += Cg[0][n] != Cref[0][n];
    check("GEMV_QKV i5..7 = 0 reads as no bias", diff > Nq / 2);
    free(x); free(ctx.scratch);
    for (int s = 0; s < 3; s++) { free(W[s]); free(b[s]); free(Cref[s]); free(Cg[s]); free(Cv[s]); }
}

/* ---- HEADNORM_ROPE i5=2: NeoX half-split at hd=64 (skip_norm, q path) ---- */
static void test_rope_halfsplit(uint32_t ntok, uint32_t nhead, uint32_t hd) {
    const uint32_t H2 = hd / 2, maxpos = 4096, out_row0 = 5;
    plow_bf16* x = xmalloc((size_t)ntok * nhead * hd * 2);
    float* cosb = xmalloc((size_t)maxpos * H2 * 4);
    float* sinb = xmalloc((size_t)maxpos * H2 * 4);
    fill_bf16(x, (size_t)ntok * nhead * hd, 1.0f);
    for (uint32_t p = 0; p < maxpos; p++)
        for (uint32_t i = 0; i < H2; i++) {
            const double f = pow(150000.0, -2.0 * (double)i / (double)hd);
            cosb[(size_t)p * H2 + i] = (float)(1.3466 * cos((double)p * f));
            sinb[(size_t)p * H2 + i] = (float)(1.3466 * sin((double)p * f));
        }
    const size_t osz = (size_t)(out_row0 + ntok) * nhead * hd;
    plow_bf16 *ref = xmalloc(osz * 2), *out = xmalloc(osz * 2), *outv = xmalloc(osz * 2);
    for (uint32_t t = 0; t < ntok; t++)
        for (uint32_t h = 0; h < nhead; h++) {
            const plow_bf16* xr = x + ((size_t)t * nhead + h) * hd;
            plow_bf16* o = ref + ((size_t)(out_row0 + t) * nhead + h) * hd;
            const size_t p = (size_t)(out_row0 + t) * H2;
            for (uint32_t i = 0; i < H2; i++) {
                const double a = plow_bf2f(xr[i]), b = plow_bf2f(xr[i + H2]);
                const double c = cosb[p + i], s = sinb[p + i];
                o[i] = plow_f2bf((float)(a * c - b * s));
                o[i + H2] = plow_f2bf((float)(b * c + a * s));
            }
        }
    void* T[8] = {out, x, NULL, cosb, sinb, NULL, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_HEADNORM_ROPE);
    in.t[0] = 0; in.t[1] = 1; in.t[3] = 3; in.t[4] = 4;
    in.i[0] = ntok; in.i[1] = nhead; in.i[2] = hd; in.i[3] = out_row0; in.i[4] = 1; in.i[5] = 2;
    in.fj[0].f = 1e-5f;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(out, 0, osz * 2);
        run_all(g_headnorm_rope, &in, nblk, T, &ctx);
        snprintf(what, sizeof what, "HEADNORM_ROPE i5=2 golden ntok=%u nhead=%u hd=%u nblk=%u", ntok, nhead, hd, nblk);
        cmp(what, ref, out, osz, 1e-2f);
        if (have_v) {
            T[0] = outv; memset(outv, 0, osz * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_HEADNORM_ROPE), &in, nblk, T, &ctx);
            T[0] = out;
            snprintf(what, sizeof what, "HEADNORM_ROPE i5=2 avx512 vs golden nblk=%u", nblk);
            cmp(what, out, outv, osz, 1e-2f);
        }
    }
    /* i5=0 at hd=64 must still be the legacy interleaved form (differs from half-split). */
    in.i[5] = 0;
    memset(out, 0, osz * 2);
    run_all(g_headnorm_rope, &in, 1, T, &ctx);
    size_t diff = 0;
    for (size_t i = 0; i < osz; i++) diff += ref[i] != out[i];
    check("HEADNORM_ROPE i5=0 at hd=64 stays interleaved", diff > osz / 4);
    free(x); free(cosb); free(sinb); free(ref); free(out); free(outv); free(ctx.scratch);
}

/* ---- FLASH_DECODE + FLASH_MERGE with sinks (t3, bf16) vs naive softmax over [scores, sink] ---- */
static void test_decode_sinks(uint32_t n_head, uint32_t n_kv_head, uint32_t hd, uint32_t kvlen,
                              uint32_t window, uint32_t nsplit) {
    const uint32_t n_batch = 1, kv_stride = 2048, kv_mask = 2047;
    const float scale = 1.0f / sqrtf((float)hd);
    plow_bf16* Q = xmalloc((size_t)n_head * hd * 2);
    plow_bf16* K = xmalloc((size_t)n_kv_head * kv_stride * hd * 2);
    plow_bf16* V = xmalloc((size_t)n_kv_head * kv_stride * hd * 2);
    plow_bf16* sinks = xmalloc(n_head * 2);
    int32_t kv_len = (int32_t)kvlen;
    fill_bf16(Q, (size_t)n_head * hd, 1.0f); fill_bf16(K, (size_t)n_kv_head * kv_stride * hd, 1.0f);
    fill_bf16(V, (size_t)n_kv_head * kv_stride * hd, 1.0f); fill_bf16(sinks, n_head, 4.0f);
    float* Opart = xmalloc((size_t)n_head * nsplit * hd * 4);
    float* mlpart = xmalloc((size_t)n_head * nsplit * 2 * 4);
    plow_bf16 *O = xmalloc((size_t)n_head * hd * 2), *Ov = xmalloc((size_t)n_head * hd * 2), *ref = xmalloc((size_t)n_head * hd * 2);
    const uint32_t gqa = n_head / n_kv_head, first = (window && kvlen > window) ? kvlen - window : 0;
    double* p = malloc(kvlen * sizeof(double));
    for (uint32_t h = 0; h < n_head; h++) {
        const plow_bf16* q = Q + (size_t)h * hd;
        const plow_bf16* kb = K + (size_t)(h / gqa) * kv_stride * hd;
        const plow_bf16* vb = V + (size_t)(h / gqa) * kv_stride * hd;
        const double sink = plow_bf2f(sinks[h]);
        double m = sink;
        for (uint32_t j = first; j < kvlen; j++) {
            p[j] = ref_dot(q, kb + (size_t)(j & kv_mask) * hd, hd) * scale;
            if (p[j] > m) m = p[j];
        }
        double denom = exp(sink - m);
        for (uint32_t j = first; j < kvlen; j++) { p[j] = exp(p[j] - m); denom += p[j]; }
        for (uint32_t d = 0; d < hd; d++) {
            double acc = 0;
            for (uint32_t j = first; j < kvlen; j++) acc += p[j] * plow_bf2f(vb[(size_t)(j & kv_mask) * hd + d]);
            ref[(size_t)h * hd + d] = plow_f2bf((float)(acc / denom));
        }
    }
    void* T[8] = {Opart, mlpart, Q, K, V, &kv_len, NULL, NULL};
    PlowDevInst fd = inst(PLOW_DOP_FLASH_DECODE);
    fd.t[0] = 0; fd.t[1] = 1; fd.t[2] = 2; fd.t[3] = 3; fd.t[4] = 4; fd.t[5] = 5;
    fd.i[0] = n_batch; fd.i[1] = n_head; fd.i[2] = n_kv_head; fd.i[3] = kv_stride; fd.i[4] = window;
    fd.i[5] = nsplit; fd.i[6] = hd; fd.i[7] = kv_mask; fd.fj[0].f = scale;
    void* TM[8] = {O, Opart, mlpart, sinks, NULL, NULL, NULL, NULL};
    PlowDevInst mg = inst(PLOW_DOP_FLASH_MERGE);
    mg.t[0] = 0; mg.t[1] = 1; mg.t[2] = 2; mg.t[3] = 3;
    mg.i[0] = n_batch; mg.i[1] = n_head; mg.i[2] = nsplit; mg.i[3] = hd;
    PlowCpuCtx ctx = mkctx();
    char what[160];
    run_all(g_flash_decode, &fd, 16, T, &ctx);
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        memset(O, 0, (size_t)n_head * hd * 2);
        run_all(g_flash_merge, &mg, nblk, TM, &ctx);
        snprintf(what, sizeof what, "FLASH_MERGE+sinks golden H=%u kv=%u win=%u nsplit=%u nblk=%u", n_head, kvlen, window, nsplit, nblk);
        cmp(what, ref, O, (size_t)n_head * hd, 2e-2f);
        if (have_v) {
            TM[0] = Ov; memset(Ov, 0, (size_t)n_head * hd * 2);
            run_all(plow_cpu_kernel(PLOW_DOP_FLASH_MERGE), &mg, nblk, TM, &ctx);
            TM[0] = O;
            snprintf(what, sizeof what, "FLASH_MERGE+sinks avx512 vs golden nsplit=%u nblk=%u", nsplit, nblk);
            cmp(what, O, Ov, (size_t)n_head * hd, 1e-2f);
        }
    }
    /* Without sinks the same partials must NOT match the sink reference (the fold is live). */
    mg.t[3] = PLOW_TENSOR_NONE;
    run_all(g_flash_merge, &mg, 1, TM, &ctx);
    double maxd = 0;
    for (size_t i = 0; i < (size_t)n_head * hd; i++) maxd = fmax(maxd, fabs(plow_bf2f(ref[i]) - plow_bf2f(O[i])));
    check("FLASH_MERGE without sinks differs from the sink reference", maxd > 1e-3);
    free(Q); free(K); free(V); free(sinks); free(Opart); free(mlpart); free(O); free(Ov); free(ref); free(p); free(ctx.scratch);
}

/* ---- MoE fixtures ---- */
typedef struct {
    uint32_t E, I, K, H;
    uint8_t *Wgu, *Sgu, *Wd, *Sd;
    plow_bf16 *bgu, *bd;
} experts_t;

static experts_t mk_experts(uint32_t E, uint32_t I, uint32_t K, uint32_t H) {
    experts_t x = {E, I, K, H, 0, 0, 0, 0, 0, 0};
    x.Wgu = xmalloc((size_t)E * 2 * I * K / 2); x.Sgu = xmalloc((size_t)E * 2 * I * K / 32); x.bgu = xmalloc((size_t)E * 2 * I * 2);
    x.Wd = xmalloc((size_t)E * H * I / 2); x.Sd = xmalloc((size_t)E * H * I / 32); x.bd = xmalloc((size_t)E * H * 2);
    fill_fp4(x.Wgu, (size_t)E * 2 * I * K / 2); fill_e8m0(x.Sgu, (size_t)E * 2 * I * K / 32); fill_bf16(x.bgu, (size_t)E * 2 * I, 1.0f);
    fill_fp4(x.Wd, (size_t)E * H * I / 2); fill_e8m0(x.Sd, (size_t)E * H * I / 32); fill_bf16(x.bd, (size_t)E * H, 1.0f);
    return x;
}
static void free_experts(experts_t* x) { free(x->Wgu); free(x->Sgu); free(x->bgu); free(x->Wd); free(x->Sd); free(x->bd); }

/* fu[n] = swiglu(W_e[g_n] . x + b, W_e[u_n] . x + b) in f64 -> bf16 */
static void ref_glu_row(const experts_t* ex, uint32_t e, uint32_t layout, const plow_bf16* x, plow_bf16* fu) {
    const uint32_t I = ex->I, K = ex->K;
    const uint8_t* W = ex->Wgu + (size_t)e * 2 * I * (K / 2);
    const uint8_t* S = ex->Sgu + (size_t)e * 2 * I * (K / 32);
    const plow_bf16* b = ex->bgu + (size_t)e * 2 * I;
    for (uint32_t n = 0; n < I; n++) {
        const uint32_t rg = layout ? n : 2 * n, ru = layout ? I + n : 2 * n + 1;
        const double g = ref_mx_dot(W, S, rg, K, x) + plow_bf2f(b[rg]);
        const double u = ref_mx_dot(W, S, ru, K, x) + plow_bf2f(b[ru]);
        fu[n] = plow_f2bf((float)ref_swiglu(g, u));
    }
}
/* part[h] = gate * (W_e[h] . fu + b_e[h]) in f64 -> f32 */
static void ref_down_row(const experts_t* ex, uint32_t e, float gate, const plow_bf16* fu, float* part) {
    const uint32_t I = ex->I, H = ex->H;
    const uint8_t* W = ex->Wd + (size_t)e * H * (I / 2);
    const uint8_t* S = ex->Sd + (size_t)e * H * (I / 32);
    const plow_bf16* b = ex->bd + (size_t)e * H;
    for (uint32_t h = 0; h < H; h++) part[h] = (float)((double)gate * (ref_mx_dot(W, S, h, I, fu) + plow_bf2f(b[h])));
}

/* ---- MOE_GLU_MX (147) + MOE_DOWN_MX (148), decode: B tokens x k slots, one sentinel slot ---- */
static void test_moe_decode(uint32_t B, uint32_t k, uint32_t E, uint32_t I, uint32_t K, uint32_t H, uint32_t layout) {
    experts_t ex = mk_experts(E, I, K, H);
    const uint32_t S = B * k;
    plow_bf16* x = xmalloc((size_t)B * K * 2);
    plow_moe_route* tab = xmalloc(S * sizeof(plow_moe_route));
    fill_bf16(x, (size_t)B * K, 1.0f);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t j = 0; j < k; j++) {
            tab[b * k + j].eid = (uint32_t)((next64() >> 20) % E);
            for (uint32_t q = 0; q < j; q++) if (tab[b * k + q].eid == tab[b * k + j].eid) tab[b * k + j].eid = (tab[b * k + j].eid + 1) % E;
            tab[b * k + j].gate = 0.05f + 0.5f * (frand() * 0.5f + 0.5f);
        }
    tab[k - 1].eid = PLOW_EXPERT_UNUSED; /* token 0, last slot: sentinel */
    plow_bf16 *fu_ref = xmalloc((size_t)S * I * 2), *fu_g = xmalloc((size_t)S * I * 2), *fu_v = xmalloc((size_t)S * I * 2);
    float *part_ref = xmalloc((size_t)S * H * 4), *part_g = xmalloc((size_t)S * H * 4), *part_v = xmalloc((size_t)S * H * 4);
    for (uint32_t s = 0; s < S; s++)
        if (tab[s].eid < E) ref_glu_row(&ex, tab[s].eid, layout, x + (size_t)(s / k) * K, fu_ref + (size_t)s * I);
    void* TG[8] = {fu_g, x, tab, ex.Wgu, ex.Sgu, ex.bgu, NULL, NULL};
    PlowDevInst gi = inst(PLOW_DOP_MOE_GLU_MX);
    gi.t[0] = 0; gi.t[1] = 1; gi.t[2] = 2; gi.t[3] = 3; gi.t[4] = 4; gi.t[5] = 5;
    gi.i[0] = k; gi.i[1] = I; gi.i[2] = K; gi.i[3] = E; gi.i[4] = layout; gi.i[5] = 3; gi.i[6] = B;
    gi.fj[0].f = ALPHA; gi.fj[1].f = LIMIT;
    void* TD[8] = {part_g, fu_g, tab, ex.Wd, ex.Sd, ex.bd, NULL, NULL};
    PlowDevInst di = inst(PLOW_DOP_MOE_DOWN_MX);
    di.t[0] = 0; di.t[1] = 1; di.t[2] = 2; di.t[3] = 3; di.t[4] = 4; di.t[5] = 5;
    di.i[0] = k; di.i[1] = H; di.i[2] = I; di.i[3] = E; di.i[6] = B;
    PlowCpuCtx ctx = mkctx();
    const plow_bf16 poison = 0x7fc1;
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        for (size_t i = 0; i < (size_t)S * I; i++) fu_g[i] = poison;
        run_all(g_moe_glu_mx, &gi, nblk, TG, &ctx);
        int sent_ok = 1;
        for (uint32_t n = 0; n < I; n++) sent_ok &= fu_g[(size_t)(k - 1) * I + n] == poison;
        for (uint32_t n = 0; n < I; n++) fu_g[(size_t)(k - 1) * I + n] = 0; /* compare the rest */
        memset(fu_ref + (size_t)(k - 1) * I, 0, (size_t)I * 2);
        g_floor = 0.05f;
        snprintf(what, sizeof what, "MOE_GLU_MX golden B=%u k=%u E=%u I=%u K=%u lay=%u nblk=%u", B, k, E, I, K, layout, nblk);
        cmp(what, fu_ref, fu_g, (size_t)S * I, 1e-2f);
        check("MOE_GLU_MX sentinel slot left untouched", sent_ok);
        if (have_v) {
            TG[0] = fu_v; for (size_t i = 0; i < (size_t)S * I; i++) fu_v[i] = poison;
            run_all(plow_cpu_kernel(PLOW_DOP_MOE_GLU_MX), &gi, nblk, TG, &ctx);
            TG[0] = fu_g;
            for (uint32_t n = 0; n < I; n++) fu_v[(size_t)(k - 1) * I + n] = 0;
            snprintf(what, sizeof what, "MOE_GLU_MX avx512 vs golden lay=%u nblk=%u", layout, nblk);
            cmp(what, fu_g, fu_v, (size_t)S * I, 1e-2f);
        }
        g_floor = 1e-2f;
        /* DOWN from golden's fu (the same input for every tier and the reference). */
        for (uint32_t s = 0; s < S; s++) {
            if (tab[s].eid < E) ref_down_row(&ex, tab[s].eid, tab[s].gate, fu_g + (size_t)s * I, part_ref + (size_t)s * H);
            else memset(part_ref + (size_t)s * H, 0, (size_t)H * 4);
        }
        fill_f32(part_g, (size_t)S * H, 1.0f);
        run_all(g_moe_down_mx, &di, nblk, TD, &ctx);
        snprintf(what, sizeof what, "MOE_DOWN_MX golden B=%u k=%u H=%u I=%u nblk=%u", B, k, H, I, nblk);
        cmp_f32(what, part_ref, part_g, (size_t)S * H, 1e-2f);
        int zero_ok = 1;
        for (uint32_t h = 0; h < H; h++) zero_ok &= part_g[(size_t)(k - 1) * H + h] == 0.0f;
        check("MOE_DOWN_MX sentinel slot zeroed", zero_ok);
        if (have_v) {
            TD[0] = part_v; fill_f32(part_v, (size_t)S * H, 1.0f);
            run_all(plow_cpu_kernel(PLOW_DOP_MOE_DOWN_MX), &di, nblk, TD, &ctx);
            TD[0] = part_g;
            snprintf(what, sizeof what, "MOE_DOWN_MX avx512 vs golden nblk=%u", nblk);
            cmp_f32(what, part_g, part_v, (size_t)S * H, 1e-2f);
        }
    }
    free(x); free(tab); free(fu_ref); free(fu_g); free(fu_v); free(part_ref); free(part_g); free(part_v); free(ctx.scratch);
    free_experts(&ex);
}

/* ---- MOE_GLU_MX_PF (149) + MOE_DOWN_MX_PF (150): T tokens sorted by expert, segments padded to 8 ---- */
static void test_moe_prefill(uint32_t Tn, uint32_t k, uint32_t E, uint32_t I, uint32_t K, uint32_t H, uint32_t layout) {
    experts_t ex = mk_experts(E, I, K, H);
    const uint32_t PAD = 8;
    plow_bf16* x = xmalloc((size_t)Tn * K * 2);
    fill_bf16(x, (size_t)Tn * K, 1.0f);
    uint32_t* sel = xmalloc((size_t)Tn * k * 4);
    float* gate = xmalloc((size_t)Tn * k * 4);
    int32_t* meta = xmalloc((size_t)(3 * E + 1) * 4);
    for (uint32_t t = 0; t < Tn; t++)
        for (uint32_t j = 0; j < k; j++) {
            uint32_t e = (uint32_t)((next64() >> 20) % E);
            for (uint32_t q = 0; q < j; q++) if (sel[t * k + q] == e) e = (e + 1) % E;
            sel[t * k + j] = e; gate[t * k + j] = 0.05f + 0.5f * (frand() * 0.5f + 0.5f);
            meta[E + e]++;
        }
    uint32_t rows = 0;
    for (uint32_t e = 0; e < E; e++) { meta[e] = (int32_t)rows; rows += ((uint32_t)meta[E + e] + PAD - 1) / PAD * PAD; }
    uint32_t* row_token = xmalloc((size_t)rows * 4);
    uint32_t* row_partidx = xmalloc((size_t)rows * 4);
    float* row_gate = xmalloc((size_t)rows * 4);
    for (uint32_t r = 0; r < rows; r++) { row_token[r] = PLOW_EXPERT_UNUSED; row_partidx[r] = PLOW_EXPERT_UNUSED; }
    uint32_t* fillp = xmalloc((size_t)E * 4);
    for (uint32_t t = 0; t < Tn; t++)
        for (uint32_t j = 0; j < k; j++) {
            const uint32_t e = sel[t * k + j], r = (uint32_t)meta[e] + fillp[e]++;
            row_token[r] = t; row_partidx[r] = t * k + j; row_gate[r] = gate[t * k + j];
        }
    plow_bf16 *fu_ref = xmalloc((size_t)rows * I * 2), *fu_g = xmalloc((size_t)rows * I * 2), *fu_v = xmalloc((size_t)rows * I * 2);
    float *part_ref = xmalloc((size_t)Tn * k * H * 4), *part_g = xmalloc((size_t)Tn * k * H * 4), *part_v = xmalloc((size_t)Tn * k * H * 4);
    for (uint32_t e = 0; e < E; e++)
        for (uint32_t r = (uint32_t)meta[e]; r < (uint32_t)meta[e] + (uint32_t)meta[E + e]; r++)
            ref_glu_row(&ex, e, layout, x + (size_t)row_token[r] * K, fu_ref + (size_t)r * I);
    void* TG[8] = {fu_g, x, ex.Wgu, ex.Sgu, meta, row_token, ex.bgu, NULL};
    PlowDevInst gi = inst(PLOW_DOP_MOE_GLU_MX_PF);
    gi.t[0] = 0; gi.t[1] = 1; gi.t[2] = 2; gi.t[3] = 3; gi.t[4] = 4; gi.t[5] = 5; gi.t[6] = 6;
    gi.i[0] = I; gi.i[1] = K; gi.i[2] = E; gi.i[3] = layout; gi.i[5] = 3; gi.fj[0].f = ALPHA; gi.fj[1].f = LIMIT;
    void* TD[8] = {part_g, fu_g, ex.Wd, ex.Sd, meta, ex.bd, row_partidx, row_gate};
    PlowDevInst di = inst(PLOW_DOP_MOE_DOWN_MX_PF);
    for (int i = 0; i < 8; i++) di.t[i] = (uint16_t)i;
    di.i[0] = H; di.i[1] = I; di.i[2] = E;
    PlowCpuCtx ctx = mkctx();
    const plow_bf16 poison = 0x7fc1;
    char what[160];
    for (size_t bi = 0; bi < NNB; bi++) {
        const uint32_t nblk = NBLKS[bi];
        for (size_t i = 0; i < (size_t)rows * I; i++) fu_g[i] = poison;
        run_all(g_moe_glu_mx_pf, &gi, nblk, TG, &ctx);
        int pad_ok = 1;
        for (uint32_t r = 0; r < rows; r++) {
            if (row_token[r] != PLOW_EXPERT_UNUSED) continue;
            for (uint32_t n = 0; n < I; n++) pad_ok &= fu_g[(size_t)r * I + n] == poison;
            memset(fu_g + (size_t)r * I, 0, (size_t)I * 2);
        }
        g_floor = 0.05f;
        snprintf(what, sizeof what, "MOE_GLU_MX_PF golden T=%u k=%u E=%u I=%u K=%u lay=%u nblk=%u", Tn, k, E, I, K, layout, nblk);
        cmp(what, fu_ref, fu_g, (size_t)rows * I, 1e-2f);
        check("MOE_GLU_MX_PF pad rows left untouched", pad_ok);
        if (have_v) {
            TG[0] = fu_v; for (size_t i = 0; i < (size_t)rows * I; i++) fu_v[i] = poison;
            run_all(plow_cpu_kernel(PLOW_DOP_MOE_GLU_MX_PF), &gi, nblk, TG, &ctx);
            TG[0] = fu_g;
            for (uint32_t r = 0; r < rows; r++) if (row_token[r] == PLOW_EXPERT_UNUSED) memset(fu_v + (size_t)r * I, 0, (size_t)I * 2);
            snprintf(what, sizeof what, "MOE_GLU_MX_PF avx512 vs golden lay=%u nblk=%u", layout, nblk);
            cmp(what, fu_g, fu_v, (size_t)rows * I, 1e-2f);
        }
        g_floor = 1e-2f;
        for (uint32_t e = 0; e < E; e++)
            for (uint32_t r = (uint32_t)meta[e]; r < (uint32_t)meta[e] + (uint32_t)meta[E + e]; r++)
                ref_down_row(&ex, e, row_gate[r], fu_g + (size_t)r * I, part_ref + (size_t)row_partidx[r] * H);
        fill_f32(part_g, (size_t)Tn * k * H, 1.0f);
        run_all(g_moe_down_mx_pf, &di, nblk, TD, &ctx);
        snprintf(what, sizeof what, "MOE_DOWN_MX_PF golden T=%u k=%u H=%u I=%u nblk=%u", Tn, k, H, I, nblk);
        cmp_f32(what, part_ref, part_g, (size_t)Tn * k * H, 1e-2f);
        if (have_v) {
            TD[0] = part_v; fill_f32(part_v, (size_t)Tn * k * H, 1.0f);
            run_all(plow_cpu_kernel(PLOW_DOP_MOE_DOWN_MX_PF), &di, nblk, TD, &ctx);
            TD[0] = part_g;
            snprintf(what, sizeof what, "MOE_DOWN_MX_PF avx512 vs golden nblk=%u", nblk);
            cmp_f32(what, part_g, part_v, (size_t)Tn * k * H, 1e-2f);
        }
    }
    free(x); free(sel); free(gate); free(meta); free(row_token); free(row_partidx); free(row_gate); free(fillp);
    free(fu_ref); free(fu_g); free(fu_v); free(part_ref); free(part_g); free(part_v); free(ctx.scratch);
    free_experts(&ex);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void bench(void) {
    const uint32_t M = 1, N = 5760, K = 2880; /* one expert's gate_up: 8.3 MB packed */
    plow_bf16* x = xmalloc((size_t)K * 2);
    uint8_t* W = xmalloc((size_t)N * K / 2); uint8_t* S = xmalloc((size_t)N * K / 32);
    plow_bf16* C = xmalloc((size_t)N * 2);
    fill_bf16(x, K, 1.0f); fill_fp4(W, (size_t)N * K / 2); fill_e8m0(S, (size_t)N * K / 32);
    void* T[8] = {C, x, W, S, NULL, NULL, NULL, NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_MXFP4);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.i[0] = M; in.i[1] = N; in.i[2] = K;
    PlowCpuCtx ctx = mkctx();
    kfn f = plow_cpu_kernel(PLOW_DOP_GEMV_MXFP4);
    const double bytes = (double)N * K / 2 + (double)N * K / 32;
    f(&in, 0, 1, T, &ctx);
    double best = 1e9;
    for (int r = 0; r < 20; r++) { double t0 = now(); f(&in, 0, 1, T, &ctx); double dt = now() - t0; if (dt < best) best = dt; }
    printf("bench GEMV_MXFP4 tier %d M=1 N=%u K=%u (cache-warm): %.3f ms, %.1f GB/s packed (%.1f MB)\n",
           plow_cpu_tier_of(PLOW_DOP_GEMV_MXFP4), N, K, best * 1e3, bytes / best / 1e9, bytes / 1e6);
    /* DRAM-streaming: cycle through 32 experts' worth of weights (265 MB > L3). */
    const uint32_t E = 32;
    uint8_t* We = xmalloc((size_t)E * N * K / 2); uint8_t* Se = xmalloc((size_t)E * N * K / 32);
    fill_fp4(We, (size_t)E * N * K / 2); fill_e8m0(Se, (size_t)E * N * K / 32);
    double tot = 0;
    for (uint32_t e = 0; e < E; e++) {
        T[2] = We + (size_t)e * N * K / 2; T[3] = Se + (size_t)e * N * K / 32;
        double t0 = now(); f(&in, 0, 1, T, &ctx); tot += now() - t0;
    }
    printf("bench GEMV_MXFP4 DRAM-streaming over %u experts: %.3f ms/expert, %.1f GB/s packed\n", E, tot / E * 1e3, bytes * E / tot / 1e9);
    double t0 = now(); g_gemv_mxfp4(&in, 0, 1, T, &ctx); double tg = now() - t0;
    printf("      golden: %.2f ms (%.1fx)\n", tg * 1e3, tg / best);
    /* MOE_GLU_MX at the GPT-OSS layer shape, k=4 slots of one token, 1 thread (all slices). */
    {
        const uint32_t k = 4, I = 2880, KK = 2880;
        plow_moe_route tab[4] = {{0, 0.3f}, {5, 0.3f}, {9, 0.2f}, {17, 0.2f}};
        plow_bf16* fu = xmalloc((size_t)k * I * 2);
        void* TG[8] = {fu, x, tab, We, Se, NULL, NULL, NULL};
        PlowDevInst gi = inst(PLOW_DOP_MOE_GLU_MX);
        gi.t[0] = 0; gi.t[1] = 1; gi.t[2] = 2; gi.t[3] = 3; gi.t[4] = 4;
        gi.i[0] = k; gi.i[1] = I; gi.i[2] = KK; gi.i[3] = E; gi.i[5] = 3; gi.fj[0].f = ALPHA; gi.fj[1].f = LIMIT;
        kfn fm = plow_cpu_kernel(PLOW_DOP_MOE_GLU_MX);
        double bm = 1e9;
        for (int r = 0; r < 5; r++) { double t1 = now(); fm(&gi, 0, 1, TG, &ctx); double dt = now() - t1; if (dt < bm) bm = dt; }
        printf("bench MOE_GLU_MX tier %d k=4 I=K=2880 (4 experts, %.1f MB): %.3f ms, %.1f GB/s packed\n",
               plow_cpu_tier_of(PLOW_DOP_MOE_GLU_MX), 4 * bytes / 1e6, bm * 1e3, 4 * bytes / bm / 1e9);
        free(fu);
    }
}

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AMX);
    printf("isa tier: %d\n", tier);
    if (tier < 0) return 2;
    if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(); return 0; }
    have_v = tier >= PLOW_CPU_ISA_AVX512;
    have_x = tier >= PLOW_CPU_ISA_AMX;
    printf("tiers: gemv_mxfp4=%d moe_glu=%d moe_down=%d moe_glu_pf=%d moe_down_pf=%d gemv=%d gemm=%d gemm_glu=%d merge=%d rope=%d\n",
           plow_cpu_tier_of(PLOW_DOP_GEMV_MXFP4), plow_cpu_tier_of(PLOW_DOP_MOE_GLU_MX),
           plow_cpu_tier_of(PLOW_DOP_MOE_DOWN_MX), plow_cpu_tier_of(PLOW_DOP_MOE_GLU_MX_PF),
           plow_cpu_tier_of(PLOW_DOP_MOE_DOWN_MX_PF), plow_cpu_tier_of(PLOW_DOP_GEMV),
           plow_cpu_tier_of(PLOW_DOP_GEMM), plow_cpu_tier_of(PLOW_DOP_GEMM_GLU),
           plow_cpu_tier_of(PLOW_DOP_FLASH_MERGE), plow_cpu_tier_of(PLOW_DOP_HEADNORM_ROPE));
    const uint16_t need[] = {PLOW_DOP_GEMV_MXFP4, PLOW_DOP_MOE_GLU_MX, PLOW_DOP_MOE_DOWN_MX,
                             PLOW_DOP_MOE_GLU_MX_PF, PLOW_DOP_MOE_DOWN_MX_PF};
    for (size_t i = 0; i < sizeof(need) / sizeof(*need); i++)
        if (!plow_cpu_has(need[i])) { printf("missing op %u\n", need[i]); fails++; }
    test_swiglu();
    test_glu_act3(5000);
    test_gemv_mxfp4(1, 512, 2880);
    test_gemv_mxfp4(4, 256, 2880);
    test_gemv_mxfp4(8, 100, 2880);   /* N tail vs RB, M > 4 */
    test_gemv_mxfp4(1, 64, 2912);    /* K % 64 == 32 tail block */
    test_gemv_mxfp4(3, 32, 2880);    /* router shape */
    test_gemv_bias(1, 4096, 2880);   /* q_proj */
    test_gemv_bias(4, 512, 2880);    /* k/v_proj, batch */
    test_gemv_bias(1, 32, 2880);     /* router */
    test_gemm_bias(PLOW_DOP_GEMM, g_gemm, 128, 2880, 4096);      /* o_proj prefill */
    test_gemm_bias(PLOW_DOP_GEMM_SMALL, g_gemm_small, 37, 512, 2880);
    test_gemm_bias(PLOW_DOP_GEMM_MED, g_gemm_med, 64, 4096, 2880);
    test_glu_proj(PLOW_DOP_GEMV_GLU, g_gemv_glu, 1, 512, 2880);
    test_glu_proj(PLOW_DOP_GEMV_GLU, g_gemv_glu, 5, 100, 2880);
    test_glu_proj(PLOW_DOP_GEMM_GLU, g_gemm_glu, 64, 512, 2880);
    test_glu_proj(PLOW_DOP_GEMM_GLU, g_gemm_glu, 37, 200, 1024);
    test_qkv_bias(1, 4096, 512, 512, 2880);
    test_qkv_bias(3, 256, 64, 64, 2880);
    test_rope_halfsplit(6, 64, 64);
    test_rope_halfsplit(3, 8, 64);
    test_decode_sinks(64, 8, 64, 100, 0, 1);
    test_decode_sinks(64, 8, 64, 300, 128, 3);
    test_decode_sinks(64, 8, 64, 700, 0, 4);
    test_moe_decode(1, 4, 8, 2880, 2880, 2880, 0);   /* GPT-OSS layer shape, 8 experts */
    test_moe_decode(2, 4, 32, 352, 2912, 320, 0);    /* K % 64 == 32, batch 2, all 32 experts */
    test_moe_decode(1, 4, 32, 352, 2880, 320, 1);    /* blocked gate|up layout */
    test_moe_prefill(37, 4, 32, 352, 2880, 320, 0);  /* 37 tokens: segments of 1..8+ rows, pads */
    test_moe_prefill(9, 4, 8, 2880, 2880, 2880, 0);  /* layer shape */
    test_moe_prefill(20, 4, 16, 352, 2912, 320, 1);
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
