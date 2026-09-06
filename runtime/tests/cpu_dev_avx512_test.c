/* cpu_dev_avx512_test.c — the AVX-512 tier against the golden tier on random inputs.
 *
 * Every registered op runs over ALL slices for nblk in {1, 3, 16} at Gemma-like shapes, both
 * through the live table (AVX-512 after plow_cpu_init(AVX512)) and through the golden entry
 * points called directly. bf16 outputs: 1e-2 relative (+1e-2 abs); integer outputs exact.
 * `--bench`: single-thread GEMV bandwidth, N=15360 K=3840 M=1, vs a plain AVX-512 read, then
 * DRAM-streaming GEMV (weights rotated over ~1 GB) at decode shapes, M=1/8, 1 and 16 threads. */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev.h"
#include "golden/golden.h"

/* Declared by hand: avx512.h needs -mavx512* to compile, the test does not. */
void v_gemv(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx);

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
static plow_bf16* rand_bf16(size_t n, float scale) {
    plow_bf16* p = malloc(n * 2 + 64);
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(frand() * scale);
    return p;
}
static plow_bf16* zeros_bf16(size_t n) { return calloc(n + 32, 2); }

static PlowDevInst inst(uint16_t op) {
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    in.op = op;
    for (int k = 0; k < 8; k++) in.t[k] = PLOW_TENSOR_NONE;
    return in;
}

static PlowCpuCtx g_ctx;

static void run_all(plow_cpu_kernel_fn fn, const PlowDevInst* in, uint32_t nblk, void* const* T) {
    PlowDevInst i2 = *in;
    i2.blocks = (uint16_t)nblk;
    for (uint32_t s = 0; s < nblk; s++) fn(&i2, s, nblk, T, &g_ctx);
}

/* Max relative error of bf16 arrays; fails on any NaN mismatch. */
static void cmp_bf16(const char* what, const plow_bf16* got, const plow_bf16* ref, size_t n) {
    double worst = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < n; i++) {
        const float g = plow_bf2f(got[i]), r = plow_bf2f(ref[i]);
        if (isnan(r) || isnan(g)) {
            CHECK(isnan(r) == isnan(g), "%s: NaN mismatch at %zu (got %f ref %f)", what, i, g, r);
            continue;
        }
        const double e = fabs((double)g - r) / (fabs((double)r) + 1.0);
        if (e > worst) { worst = e; wi = i; }
    }
    CHECK(worst <= 1e-2, "%s: max rel err %.4g at %zu (got %f ref %f)", what, worst, wi,
          plow_bf2f(got[wi]), plow_bf2f(ref[wi]));
}
static void cmp_f32(const char* what, const float* got, const float* ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const double e = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-6);
        CHECK(e <= 1e-3, "%s: rel err %.4g at %zu (got %f ref %f)", what, e, i, got[i], ref[i]);
    }
}

static const uint32_t NBLKS[3] = {1, 3, 16};

static plow_cpu_kernel_fn avx(uint16_t op, plow_cpu_kernel_fn golden, const char* name) {
    plow_cpu_kernel_fn f = plow_cpu_kernel(op);
    CHECK(f != NULL, "%s: no kernel registered", name);
    CHECK(f != golden, "%s: table still holds the golden kernel", name);
    return f ? f : golden;
}

/* --- GEMV family --------------------------------------------------------------------- */

static void test_gemv(uint32_t M, uint32_t N, uint32_t K, uint32_t norm) {
    plow_bf16* x = rand_bf16((size_t)(M + 2) * K, 1.0f);
    plow_bf16* W = rand_bf16((size_t)N * K, 0.05f);
    plow_bf16* gamma = rand_bf16(K, 1.0f);
    float* rms = malloc(M * sizeof(float));
    for (uint32_t m = 0; m < M; m++) rms[m] = 0.5f + 0.1f * (float)m;
    plow_bf16 *out = zeros_bf16((size_t)M * N), *ref = zeros_bf16((size_t)M * N);
    void* T[5] = {out, x, W, norm == 1 ? (void*)rms : NULL, norm ? (void*)gamma : NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (norm == 1) in.t[3] = 3;
    if (norm) in.t[4] = 4;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[3] = norm; in.i[4] = 2; /* a_row0 = 2 */
    in.fj[0].f = 1e-6f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_GEMV, g_gemv, "gemv");
    for (int k = 0; k < 3; k++) {
        memset(out, 0, (size_t)M * N * 2);
        memset(ref, 0, (size_t)M * N * 2);
        T[0] = ref; run_all(g_gemv, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        char what[96];
        snprintf(what, sizeof what, "gemv M=%u N=%u K=%u norm=%u nblk=%u", M, N, K, norm, NBLKS[k]);
        cmp_bf16(what, out, ref, (size_t)M * N);
    }
    free(x); free(W); free(gamma); free(rms); free(out); free(ref);
}

static void test_gemv_glu(uint32_t M, uint32_t N, uint32_t K, uint32_t act) {
    plow_bf16* x = rand_bf16((size_t)M * K, 1.0f);
    plow_bf16* Wg = rand_bf16((size_t)N * K, 0.05f);
    plow_bf16* Wu = rand_bf16((size_t)N * K, 0.05f);
    plow_bf16 *out = zeros_bf16((size_t)M * N), *ref = zeros_bf16((size_t)M * N);
    void* T[6] = {out, x, Wg, NULL, NULL, Wu};
    PlowDevInst in = inst(PLOW_DOP_GEMV_GLU);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[5] = 5;
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[5] = act;
    in.fj[0].f = 4.0f; in.fj[1].f = act == 2 ? 3.0f : 0.0f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_GEMV_GLU, g_gemv_glu, "gemv_glu");
    for (int k = 0; k < 3; k++) {
        T[0] = ref; run_all(g_gemv_glu, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        char what[96];
        snprintf(what, sizeof what, "gemv_glu M=%u N=%u K=%u act=%u nblk=%u", M, N, K, act, NBLKS[k]);
        cmp_bf16(what, out, ref, (size_t)M * N);
    }
    free(x); free(Wg); free(Wu); free(out); free(ref);
}

static void test_gemv_qkv(uint32_t M, uint32_t K, int gnorm) {
    const uint32_t Nq = 2048, Nk = 512, Nv = 512;
    plow_bf16* x = rand_bf16((size_t)M * K, 1.0f);
    plow_bf16 *Wq = rand_bf16((size_t)Nq * K, 0.05f), *Wk = rand_bf16((size_t)Nk * K, 0.05f),
              *Wv = rand_bf16((size_t)Nv * K, 0.05f);
    plow_bf16* g = rand_bf16(K, 1.0f);
    plow_bf16 *q = zeros_bf16((size_t)M * Nq), *kk = zeros_bf16((size_t)M * Nk), *v = zeros_bf16((size_t)M * Nv);
    plow_bf16 *rq = zeros_bf16((size_t)M * Nq), *rk = zeros_bf16((size_t)M * Nk), *rv = zeros_bf16((size_t)M * Nv);
    void* T[8] = {q, x, Wq, kk, Wk, v, Wv, gnorm ? (void*)g : NULL};
    PlowDevInst in = inst(PLOW_DOP_GEMV_QKV);
    for (int t = 0; t < 7; t++) in.t[t] = (uint16_t)t;
    if (gnorm) in.t[7] = 7;
    in.i[0] = M; in.i[1] = Nq; in.i[2] = K; in.i[3] = Nk; in.i[4] = Nv;
    in.fj[0].f = 1e-6f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_GEMV_QKV, g_gemv_qkv, "gemv_qkv");
    for (int k = 0; k < 3; k++) {
        T[0] = rq; T[3] = rk; T[5] = rv; run_all(g_gemv_qkv, &in, NBLKS[k], T);
        T[0] = q; T[3] = kk; T[5] = v; run_all(f, &in, NBLKS[k], T);
        char what[96];
        snprintf(what, sizeof what, "gemv_qkv M=%u K=%u gnorm=%d nblk=%u", M, K, gnorm, NBLKS[k]);
        cmp_bf16(what, q, rq, (size_t)M * Nq);
        cmp_bf16(what, kk, rk, (size_t)M * Nk);
        cmp_bf16(what, v, rv, (size_t)M * Nv);
    }
    free(x); free(Wq); free(Wk); free(Wv); free(g);
    free(q); free(kk); free(v); free(rq); free(rk); free(rv);
}

static uint32_t key_index(uint64_t key) { return ~(uint32_t)(key & 0xFFFFFFFFull); }

static void test_gemv_argmax(uint32_t N, uint32_t K, float cap) {
    plow_bf16* x = rand_bf16(K * 3, 1.0f);
    plow_bf16* W = rand_bf16((size_t)N * K, 0.05f);
    plow_bf16 *out = zeros_bf16(N), *ref = zeros_bf16(N);
    uint64_t part[16], rpart[16];
    void* T[4] = {out, x, W, part};
    PlowDevInst in = inst(PLOW_DOP_GEMV_ARGMAX);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    in.i[0] = 1; in.i[1] = N; in.i[2] = K; in.i[4] = 1;
    in.fj[0].f = cap;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_GEMV_ARGMAX, g_gemv_argmax, "gemv_argmax");
    for (int k = 0; k < 3; k++) {
        const uint32_t nb = NBLKS[k];
        T[0] = ref; T[3] = rpart; run_all(g_gemv_argmax, &in, nb, T);
        T[0] = out; T[3] = part; run_all(f, &in, nb, T);
        char what[96];
        snprintf(what, sizeof what, "gemv_argmax N=%u K=%u cap=%g nblk=%u", N, K, cap, nb);
        cmp_bf16(what, out, ref, N);
        /* Keys: self-consistent (index points at the max of this tier's own logits) and equal to
         * golden's unless the two tiers' bf16 logits differ at either index (a rounding flip). */
        uint64_t best = 0, rbest = 0;
        for (uint32_t s = 0; s < nb; s++) { best = part[s] > best ? part[s] : best; rbest = rpart[s] > rbest ? rpart[s] : rbest; }
        const uint32_t bi = key_index(best), ri = key_index(rbest);
        CHECK(bi < N, "%s: key index %u out of range", what, bi);
        if (bi < N) {
            float mx = -INFINITY;
            for (uint32_t i = 0; i < N; i++) mx = fmaxf(mx, plow_bf2f(out[i]));
            CHECK(plow_bf2f(out[bi]) == mx, "%s: key index %u is not the max (%f vs %f)", what, bi,
                  plow_bf2f(out[bi]), mx);
            if (bi != ri && ri < N)
                CHECK(out[bi] != ref[bi] || out[ri] != ref[ri],
                      "%s: argmax index %u differs from golden %u with identical logits", what, bi, ri);
        }
    }
    free(x); free(W); free(out); free(ref);
}

/* --- pointwise ------------------------------------------------------------------------ */

static void test_residual(uint32_t n, int pre) {
    plow_bf16 *a = rand_bf16(n, 2.0f), *b = rand_bf16(n, 2.0f), *p = rand_bf16(n, 2.0f);
    plow_bf16 *out = zeros_bf16(n), *ref = zeros_bf16(n);
    void* T[4] = {out, a, b, pre ? (void*)p : NULL};
    PlowDevInst in = inst(PLOW_DOP_RESIDUAL);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (pre) in.t[3] = 3;
    in.i[0] = n; in.fj[0].f = 0.75f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_RESIDUAL, g_residual, "residual");
    for (int k = 0; k < 3; k++) {
        T[0] = ref; run_all(g_residual, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        cmp_bf16("residual", out, ref, n);
    }
    free(a); free(b); free(p); free(out); free(ref);
}

static void test_glu(uint32_t n, uint32_t act) {
    plow_bf16 *g = rand_bf16(n, 6.0f), *u = rand_bf16(n, 2.0f);
    plow_bf16 *out = zeros_bf16(n), *ref = zeros_bf16(n);
    void* T[3] = {out, g, u};
    PlowDevInst in = inst(PLOW_DOP_GLU);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = n; in.i[1] = act;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_GLU, g_glu, "glu");
    for (int k = 0; k < 3; k++) {
        T[0] = ref; run_all(g_glu, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        cmp_bf16(act ? "glu silu" : "glu gelu", out, ref, n);
    }
    free(g); free(u); free(out); free(ref);
}

static void test_softcap(uint32_t n) {
    plow_bf16* x = rand_bf16(n, 60.0f);
    plow_bf16 *out = zeros_bf16(n), *ref = zeros_bf16(n);
    void* T[2] = {out, x};
    PlowDevInst in = inst(PLOW_DOP_SOFTCAP);
    in.t[0] = 0; in.t[1] = 1; in.i[0] = n; in.fj[0].f = 30.0f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_SOFTCAP, g_softcap, "softcap");
    for (int k = 0; k < 3; k++) {
        T[0] = ref; run_all(g_softcap, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        cmp_bf16("softcap", out, ref, n);
    }
    free(x); free(out); free(ref);
}

static void test_embed(uint32_t ntok, uint32_t hidden) {
    const uint32_t vocab = 1000;
    plow_bf16* table = rand_bf16((size_t)vocab * hidden, 1.0f);
    int32_t* ids = malloc(ntok * 4);
    for (uint32_t t = 0; t < ntok; t++) ids[t] = (int32_t)((t * 397u + 13u) % vocab);
    plow_bf16 *out = zeros_bf16((size_t)ntok * hidden), *ref = zeros_bf16((size_t)ntok * hidden);
    void* T[3] = {out, table, ids};
    PlowDevInst in = inst(PLOW_DOP_EMBED);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = ntok; in.i[1] = hidden; in.fj[0].f = 62.0f;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_EMBED, g_embed, "embed");
    for (int k = 0; k < 3; k++) {
        T[0] = ref; run_all(g_embed, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        cmp_bf16("embed", out, ref, (size_t)ntok * hidden);
    }
    free(table); free(ids); free(out); free(ref);
}

static void test_argmax(uint32_t n, uint32_t B) {
    plow_bf16* x = rand_bf16((size_t)B * n, 20.0f);
    /* Force a few exact ties and negative-heavy rows to exercise the key order. */
    x[7] = x[3];
    for (uint32_t i = 0; i < n; i++) x[(size_t)(B - 1) * n + i] = plow_f2bf(-fabsf(plow_bf2f(x[(size_t)(B - 1) * n + i])) - 1.0f);
    uint64_t *part = calloc(B * 16, 8), *rpart = calloc(B * 16, 8);
    int32_t ids[4], rids[4];
    void* T[2] = {part, x};
    PlowDevInst in = inst(PLOW_DOP_ARGMAX);
    in.t[0] = 0; in.t[1] = 1; in.i[0] = n; in.i[1] = B;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_ARGMAX, g_argmax, "argmax");
    plow_cpu_kernel_fn ff = avx(PLOW_DOP_ARGMAX_FIN, g_argmax_fin, "argmax_fin");
    for (int k = 0; k < 3; k++) {
        const uint32_t nb = NBLKS[k];
        T[0] = rpart; run_all(g_argmax, &in, nb, T);
        T[0] = part; run_all(f, &in, nb, T);
        for (uint32_t i = 0; i < B * nb; i++)
            CHECK(part[i] == rpart[i], "argmax nblk=%u part[%u] %llx != %llx", nb, i,
                  (unsigned long long)part[i], (unsigned long long)rpart[i]);
        PlowDevInst fin = inst(PLOW_DOP_ARGMAX_FIN);
        fin.t[0] = 0; fin.t[1] = 1; fin.i[0] = nb; fin.i[1] = B;
        void* TF[2] = {ids, part};
        run_all(ff, &fin, 1, TF);
        void* TR[2] = {rids, rpart};
        run_all(g_argmax_fin, &fin, 1, TR);
        for (uint32_t b = 0; b < B; b++) CHECK(ids[b] == rids[b], "argmax_fin b=%u %d != %d", b, ids[b], rids[b]);
    }
    free(x); free(part); free(rpart);
}

/* --- norms ----------------------------------------------------------------------------- */

static void test_norms(uint32_t rows, uint32_t feat) {
    const uint32_t out_row0 = 2;
    plow_bf16 *x = rand_bf16((size_t)rows * feat, 3.0f), *b = rand_bf16((size_t)rows * feat, 3.0f);
    plow_bf16 *g = rand_bf16(feat, 1.0f), *g2 = rand_bf16(feat, 1.0f), *beta = rand_bf16(feat, 0.5f);
    const size_t outn = (size_t)(rows + out_row0) * feat;
    plow_bf16 *out = zeros_bf16(outn), *ref = zeros_bf16(outn);
    plow_bf16 *res = zeros_bf16(outn), *rres = zeros_bf16(outn);
    float *rms = calloc(rows, 4), *rrms = calloc(rows, 4);
    char what[96];

    /* RMSNORM, with and without gamma */
    for (int gm = 0; gm < 2; gm++) {
        void* T[3] = {out, x, gm ? (void*)g : NULL};
        PlowDevInst in = inst(PLOW_DOP_RMSNORM);
        in.t[0] = 0; in.t[1] = 1; if (gm) in.t[2] = 2;
        in.i[0] = rows; in.i[1] = feat; in.i[2] = out_row0; in.fj[0].f = 1e-6f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_RMSNORM, g_rmsnorm, "rmsnorm");
        for (int k = 0; k < 3; k++) {
            T[0] = ref; run_all(g_rmsnorm, &in, NBLKS[k], T);
            T[0] = out; run_all(f, &in, NBLKS[k], T);
            snprintf(what, sizeof what, "rmsnorm feat=%u gamma=%d nblk=%u", feat, gm, NBLKS[k]);
            cmp_bf16(what, out, ref, outn);
        }
    }
    /* ROWRMS */
    {
        void* T[2] = {rms, x};
        PlowDevInst in = inst(PLOW_DOP_ROWRMS);
        in.t[0] = 0; in.t[1] = 1; in.i[0] = rows; in.i[1] = feat; in.fj[0].f = 1e-6f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_ROWRMS, g_rowrms, "rowrms");
        for (int k = 0; k < 3; k++) {
            T[0] = rrms; run_all(g_rowrms, &in, NBLKS[k], T);
            T[0] = rms; run_all(f, &in, NBLKS[k], T);
            cmp_f32("rowrms", rms, rrms, rows);
        }
    }
    /* LAYERNORM */
    {
        void* T[4] = {out, x, g, beta};
        PlowDevInst in = inst(PLOW_DOP_LAYERNORM);
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
        in.i[0] = rows; in.i[1] = feat; in.i[3] = out_row0; in.fj[0].f = 1e-5f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_LAYERNORM, g_layernorm, "layernorm");
        for (int k = 0; k < 3; k++) {
            T[0] = ref; run_all(g_layernorm, &in, NBLKS[k], T);
            T[0] = out; run_all(f, &in, NBLKS[k], T);
            snprintf(what, sizeof what, "layernorm feat=%u nblk=%u", feat, NBLKS[k]);
            cmp_bf16(what, out, ref, outn);
        }
    }
    /* NORM_RESIDUAL */
    {
        void* T[4] = {out, x, b, g};
        PlowDevInst in = inst(PLOW_DOP_NORM_RESIDUAL);
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
        in.i[0] = rows; in.i[1] = feat; in.fj[0].f = 1e-6f; in.fj[1].f = 1.25f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_NORM_RESIDUAL, g_norm_residual, "norm_residual");
        for (int k = 0; k < 3; k++) {
            T[0] = ref; run_all(g_norm_residual, &in, NBLKS[k], T);
            T[0] = out; run_all(f, &in, NBLKS[k], T);
            snprintf(what, sizeof what, "norm_residual feat=%u nblk=%u", feat, NBLKS[k]);
            cmp_bf16(what, out, ref, (size_t)rows * feat);
        }
    }
    /* ADD_NORM (resid distinct from a) */
    {
        void* T[5] = {out, res, x, b, g};
        PlowDevInst in = inst(PLOW_DOP_ADD_NORM);
        for (int t = 0; t < 5; t++) in.t[t] = (uint16_t)t;
        in.i[0] = rows; in.i[1] = feat; in.fj[0].f = 1e-6f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_ADD_NORM, g_add_norm, "add_norm");
        for (int k = 0; k < 3; k++) {
            T[0] = ref; T[1] = rres; run_all(g_add_norm, &in, NBLKS[k], T);
            T[0] = out; T[1] = res; run_all(f, &in, NBLKS[k], T);
            snprintf(what, sizeof what, "add_norm feat=%u nblk=%u", feat, NBLKS[k]);
            cmp_bf16(what, out, ref, (size_t)rows * feat);
            cmp_bf16(what, res, rres, (size_t)rows * feat);
        }
    }
    /* NORM_RESIDUAL_NORM */
    {
        void* T[6] = {out, res, x, b, g, g2};
        PlowDevInst in = inst(PLOW_DOP_NORM_RESIDUAL_NORM);
        for (int t = 0; t < 6; t++) in.t[t] = (uint16_t)t;
        in.i[0] = rows; in.i[1] = feat; in.fj[0].f = 1e-6f; in.fj[1].f = 0.9f;
        plow_cpu_kernel_fn f = avx(PLOW_DOP_NORM_RESIDUAL_NORM, g_norm_residual_norm, "norm_residual_norm");
        for (int k = 0; k < 3; k++) {
            T[0] = ref; T[1] = rres; run_all(g_norm_residual_norm, &in, NBLKS[k], T);
            T[0] = out; T[1] = res; run_all(f, &in, NBLKS[k], T);
            snprintf(what, sizeof what, "norm_residual_norm feat=%u nblk=%u", feat, NBLKS[k]);
            cmp_bf16(what, out, ref, (size_t)rows * feat);
            cmp_bf16(what, res, rres, (size_t)rows * feat);
        }
    }
    free(x); free(b); free(g); free(g2); free(beta); free(out); free(ref); free(res); free(rres);
    free(rms); free(rrms);
}

/* --- bench ----------------------------------------------------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include <immintrin.h>
__attribute__((target("avx512f"))) static uint64_t stream_read(const void* p, size_t bytes) {
    const char* c = p;
    __m512i a = _mm512_setzero_si512(), b = _mm512_setzero_si512();
    __m512i d = _mm512_setzero_si512(), e = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 256 <= bytes; i += 256) {
        a = _mm512_xor_si512(a, _mm512_loadu_si512(c + i));
        b = _mm512_xor_si512(b, _mm512_loadu_si512(c + i + 64));
        d = _mm512_xor_si512(d, _mm512_loadu_si512(c + i + 128));
        e = _mm512_xor_si512(e, _mm512_loadu_si512(c + i + 192));
    }
    a = _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_xor_si512(d, e));
    return (uint64_t)_mm512_reduce_add_epi64(a);
}

static void bench_stream(void);
static void bench(void) {
    const uint32_t N = 15360, K = 3840, M = 1;
    plow_bf16* x = rand_bf16(K, 1.0f);
    plow_bf16* W = rand_bf16((size_t)N * K, 0.05f);
    plow_bf16* out = zeros_bf16(N);
    void* T[3] = {out, x, W};
    PlowDevInst in = inst(PLOW_DOP_GEMV);
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    in.blocks = 1;
    const double bytes = (double)N * K * 2.0;
    plow_cpu_kernel_fn f = plow_cpu_kernel(PLOW_DOP_GEMV);
    f(&in, 0, 1, T, &g_ctx); /* warm */
    double best = 1e30;
    for (int it = 0; it < 7; it++) {
        const double t0 = now_s();
        f(&in, 0, 1, T, &g_ctx);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    printf("gemv  N=%u K=%u M=1: %.1f ms  %.1f GB/s (1 thread, %.0f MB weights)\n", N, K, best * 1e3,
           bytes / best / 1e9, bytes / 1e6);
    /* Plain AVX-512 streaming read of the same buffer: one core's bandwidth ceiling. */
    double rbest = 1e30;
    volatile uint64_t sink = 0;
    for (int it = 0; it < 7; it++) {
        const double t0 = now_s();
        sink += stream_read(W, (size_t)N * K * 2);
        const double dt = now_s() - t0;
        if (dt < rbest) rbest = dt;
    }
    printf("read  same buffer, AVX-512 loads only: %.1f GB/s (gemv = %.0f%% of it)\n",
           bytes / rbest / 1e9, 100.0 * rbest / best);
    /* Golden for scale. */
    const double t0 = now_s();
    g_gemv(&in, 0, 1, T, &g_ctx);
    printf("golden gemv: %.1f ms\n", (now_s() - t0) * 1e3);
    free(x); free(W); free(out);
    bench_stream();
}

/* --- multi-thread streaming bench ---------------------------------------------------------
 * One start barrier, then every thread runs its own slice over all rotations free-running (a
 * per-rotation barrier turned into a scheduler artifact: 16 spinning threads on 16 logical
 * CPUs stalled a whole timeslice whenever one was preempted). Aggregate bandwidth = all the
 * weight bytes read divided by the wall time from the first start to the last finish. */

typedef struct {
    plow_cpu_kernel_fn f;
    PlowDevInst in;
    void*** Ts; /* per-rotation tensor tables (distinct weight buffers) */
    uint32_t nrot, nthr;
    atomic_uint arrive;
    double t0[64], t1[64];
} MtBench;
typedef struct { MtBench* b; uint32_t slice; PlowCpuCtx ctx; } MtThr;

static void* mt_run(void* p) {
    MtThr* t = p;
    MtBench* b = t->b;
    atomic_fetch_add(&b->arrive, 1);
    while (atomic_load(&b->arrive) < b->nthr) _mm_pause();
    b->t0[t->slice] = now_s();
    for (uint32_t r = 0; r < b->nrot; r++) b->f(&b->in, t->slice, b->nthr, b->Ts[r], &t->ctx);
    b->t1[t->slice] = now_s();
    return NULL;
}
/* Wall seconds for one rotation of the whole weight matrix across all threads. */
static double mt_bench(MtBench* b) {
    MtThr* th = calloc(b->nthr, sizeof *th);
    pthread_t* tid = calloc(b->nthr, sizeof *tid);
    atomic_store(&b->arrive, 0);
    for (uint32_t i = 0; i < b->nthr; i++) {
        th[i].b = b;
        th[i].slice = i;
        th[i].ctx.scratch_bytes = plow_cpu_scratch_bytes();
        th[i].ctx.scratch = aligned_alloc(64, th[i].ctx.scratch_bytes);
        plow_cpu_thread_init(&th[i].ctx);
        if (i) pthread_create(&tid[i], NULL, mt_run, &th[i]);
    }
    mt_run(&th[0]);
    for (uint32_t i = 1; i < b->nthr; i++) pthread_join(tid[i], NULL);
    double lo = b->t0[0], hi = b->t1[0];
    for (uint32_t i = 1; i < b->nthr; i++) {
        if (b->t0[i] < lo) lo = b->t0[i];
        if (b->t1[i] > hi) hi = b->t1[i];
    }
    for (uint32_t i = 0; i < b->nthr; i++) free(th[i].ctx.scratch);
    free(th); free(tid);
    return (hi - lo) / (double)b->nrot;
}

static void bench_stream(void) {
    const uint32_t shapes[3][2] = {{3840, 3840}, {5760, 2880}, {5760, 3840}};
    const size_t wmax = (size_t)5760 * 3840;
    const uint32_t R = 24; /* 24 x 44 MB > 1 GB: past L3 even on a 260 MB part */
    plow_bf16* W = aligned_alloc(64, wmax * 2 * R);
    plow_bf16* w0 = rand_bf16(wmax, 0.05f);
    for (uint32_t r = 0; r < R; r++) memcpy(W + r * wmax, w0, wmax * 2);
    free(w0);
    plow_bf16* x = rand_bf16((size_t)8 * 3840, 1.0f);
    plow_bf16* C = zeros_bf16((size_t)8 * 5760);
    void*** Ts = malloc(R * sizeof *Ts);
    void** Tbuf = malloc(R * 3 * sizeof *Tbuf);
    printf("--- DRAM-streaming GEMV bf16 (weights rotated over %u buffers) ---\n", R);
    for (int si = 0; si < 3; si++)
        for (uint32_t M = 1; M <= 8; M += 7)
            for (uint32_t thr = 1; thr <= 16; thr *= 16) {
                const uint32_t N = shapes[si][0], K = shapes[si][1];
                for (uint32_t r = 0; r < R; r++) {
                    Ts[r] = Tbuf + r * 3;
                    Ts[r][0] = C; Ts[r][1] = x; Ts[r][2] = W + r * wmax;
                }
                MtBench b = {0};
                b.f = plow_cpu_kernel(PLOW_DOP_GEMV);
                b.in = inst(PLOW_DOP_GEMV);
                b.in.t[0] = 0; b.in.t[1] = 1; b.in.t[2] = 2;
                b.in.i[0] = M; b.in.i[1] = N; b.in.i[2] = K;
                b.in.blocks = (uint16_t)thr;
                b.Ts = Ts; b.nrot = R; b.nthr = thr;
                const double best = mt_bench(&b), bytes = (double)N * K * 2.0;
                printf("gemv bf16 N=%5u K=%4u M=%u thr=%2u: %7.3f ms  %6.1f GB/s\n", N, K, M, thr,
                       best * 1e3, bytes / best / 1e9);
            }
    free(W); free(x); free(C); free(Ts); free(Tbuf);
}

/* Static-archive pitfall: dispatch.o's weak no-op registrar satisfies the symbol, so the
 * archive member with the strong definition is never extracted unless something outside the
 * archive references it. This reference forces it in; the same is needed wherever the archive
 * is linked (plowrt build.rs) — a whole-archive link or a non-weak hook is the durable fix. */
plow_cpu_kernel_fn volatile g_force_avx512_member = v_gemv;

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AVX512);
    if (tier < PLOW_CPU_ISA_AVX512) {
        printf("SKIP: host tier %d has no AVX-512 BF16\n", tier);
        return 0;
    }
    memset(&g_ctx, 0, sizeof g_ctx);
    g_ctx.scratch_bytes = plow_cpu_scratch_bytes();
    g_ctx.scratch = aligned_alloc(64, g_ctx.scratch_bytes);
    plow_cpu_thread_init(&g_ctx);
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) { bench(); return 0; }

    const uint32_t Ks[2] = {3840, 3872};
    for (int ki = 0; ki < 2; ki++) {
        const uint32_t K = Ks[ki];
        for (uint32_t norm = 0; norm < 3; norm++) {
            test_gemv(1, 3840, K, norm);
            test_gemv(2, 1024, K, norm);
            test_gemv(4, 1024, K, norm);
            test_gemv(8, 1024, K, norm);
        }
        test_gemv(3, 333, K, 0);
        test_gemv(5, 333, K, 2);
        test_gemv_glu(1, 2048, K, 0);
        test_gemv_glu(1, 2048, K, 1);
        test_gemv_glu(4, 512, K, 0);
        test_gemv_glu(8, 512, K, 2);
        test_gemv_qkv(1, K, 0);
        test_gemv_qkv(1, K, 1);
        test_gemv_qkv(4, K, 1);
        test_gemv_argmax(8192, K, 30.0f);
        test_gemv_argmax(8199, K, 0.0f);
        test_norms(5, K);
    }
    test_gemv(1, 15360, 3840, 0);
    test_residual(100003, 0);
    test_residual(100003, 1);
    test_glu(100003, 0);
    test_glu(100003, 1);
    test_softcap(100003);
    test_embed(5, 3840);
    test_embed(3, 3872);
    test_argmax(8199, 2);
    test_argmax(262144, 1);

    if (fails == 0) printf("cpu_dev_avx512_test: all passed (tier %d)\n", tier);
    else printf("cpu_dev_avx512_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
