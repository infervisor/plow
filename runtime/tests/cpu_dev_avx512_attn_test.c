/* cpu_dev_avx512_attn_test.c — AVX-512 HEADNORM_ROPE / FLASH_* against the golden tier.
 *
 * Every op runs over ALL slices for nblk in {1, 3, 16} at Gemma-4-12B shapes (hd 256 x 8 KV
 * heads / window 1024, hd 512 x 1 KV head, 16 q heads) through the live table (AVX-512) and the
 * golden entry points. bf16 outputs: 1e-2 relative (+1e-2 abs). f32 partials: 1e-2 relative.
 * `--bench`: FLASH_DECODE kvlen 1024, hd 512, 1 KV head x 16 q heads, single thread. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev.h"
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
/* Sum of 4 uniforms: close enough to N(0, 0.58) for realistic softmax sharpness. */
static float nrand(void) { return (frand() + frand() + frand() + frand()) * 0.5f; }
static plow_bf16* rand_bf16(size_t n, float scale) {
    plow_bf16* p = malloc(n * 2 + 64);
    for (size_t i = 0; i < n; i++) p[i] = plow_f2bf(nrand() * scale);
    return p;
}
static plow_bf16* zeros_bf16(size_t n) { return calloc(n + 32, 2); }
static float* zeros_f32(size_t n) { return calloc(n + 16, 4); }

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
/* Partials are f32 sums over bf16-rounded P (prefill) or f32 P (decode); the AVX-512 softmax is
 * blockwise, so the tolerance is the bf16 one, relative to the row's magnitude. */
static void cmp_f32(const char* what, const float* got, const float* ref, size_t n, double tol) {
    double worst = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < n; i++) {
        if (ref[i] == G_NEG_INF || got[i] == G_NEG_INF) {
            CHECK(ref[i] == got[i], "%s: -inf mismatch at %zu (got %g ref %g)", what, i, got[i], ref[i]);
            continue;
        }
        const double e = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1.0);
        if (e > worst) { worst = e; wi = i; }
    }
    CHECK(worst <= tol, "%s: max rel err %.4g at %zu (got %f ref %f)", what, worst, wi, got[wi],
          ref[wi]);
}

static const uint32_t NBLKS[3] = {1, 3, 16};

static plow_cpu_kernel_fn avx(uint16_t op, plow_cpu_kernel_fn golden, const char* name) {
    plow_cpu_kernel_fn f = plow_cpu_kernel(op);
    CHECK(f != NULL, "%s: no kernel registered", name);
    CHECK(f != golden, "%s: table still holds the golden kernel", name);
    CHECK(plow_cpu_tier_of(op) == PLOW_CPU_ISA_AVX512, "%s: tier %d", name, plow_cpu_tier_of(op));
    return f ? f : golden;
}

/* Real RoPE tables (theta 10000) as f32 [ctx][hd/2]. */
static void rope_tables(uint32_t ctx, uint32_t hd, float** cosb, float** sinb) {
    const uint32_t H2 = hd / 2;
    *cosb = malloc((size_t)ctx * H2 * 4);
    *sinb = malloc((size_t)ctx * H2 * 4);
    for (uint32_t p = 0; p < ctx; p++)
        for (uint32_t i = 0; i < H2; i++) {
            const double f = pow(10000.0, -2.0 * (double)i / (double)hd);
            (*cosb)[(size_t)p * H2 + i] = (float)cos((double)p * f);
            (*sinb)[(size_t)p * H2 + i] = (float)sin((double)p * f);
        }
}

/* --- HEADNORM_ROPE ------------------------------------------------------------------- */

/* mode 0: q path (out_stride 0); 1: K ring (out_stride, kv_mask); 2: batched KV with pos tensor. */
static void test_headnorm_rope(uint32_t ntok, uint32_t nhead, uint32_t hd, int with_gamma,
                               uint32_t skip_norm, int with_rope, uint32_t mode) {
    const uint32_t stride = 2048, mask = 2047, out_row0 = mode ? 1500 : 3;
    plow_bf16* x = rand_bf16((size_t)ntok * nhead * hd, 1.0f);
    plow_bf16* gamma = rand_bf16(hd, 1.0f);
    float *cosb = NULL, *sinb = NULL;
    if (with_rope) rope_tables(4096, hd, &cosb, &sinb);
    int32_t* pos = malloc(ntok * 4);
    for (uint32_t t = 0; t < ntok; t++) pos[t] = (int32_t)(out_row0 + t);
    const size_t osz = mode == 0 ? (size_t)(out_row0 + ntok) * nhead * hd
                                 : (size_t)(mode == 2 ? ntok * nhead : nhead) * stride * hd;
    plow_bf16 *out = zeros_bf16(osz), *ref = zeros_bf16(osz);
    void* T[6] = {out, x, with_gamma ? gamma : NULL, cosb, sinb, mode == 2 ? (void*)pos : NULL};
    PlowDevInst in = inst(PLOW_DOP_HEADNORM_ROPE);
    in.t[0] = 0; in.t[1] = 1;
    if (with_gamma) in.t[2] = 2;
    if (with_rope) { in.t[3] = 3; in.t[4] = 4; }
    if (mode == 2) in.t[5] = 5;
    in.i[0] = ntok; in.i[1] = nhead; in.i[2] = hd; in.i[3] = out_row0; in.i[4] = skip_norm;
    in.i[6] = mode == 2 ? 1u : 0u;
    in.fj[0].f = 1e-6f;
    if (mode) { in.fj[1].u = stride; in.fj[2].u = mask; }
    plow_cpu_kernel_fn f = avx(PLOW_DOP_HEADNORM_ROPE, g_headnorm_rope, "headnorm_rope");
    for (int k = 0; k < 3; k++) {
        memset(out, 0, osz * 2);
        memset(ref, 0, osz * 2);
        T[0] = ref; run_all(g_headnorm_rope, &in, NBLKS[k], T);
        T[0] = out; run_all(f, &in, NBLKS[k], T);
        char what[128];
        snprintf(what, sizeof what, "headnorm_rope ntok=%u nhead=%u hd=%u gamma=%d skip=%u rope=%d mode=%u nblk=%u",
                 ntok, nhead, hd, with_gamma, skip_norm, with_rope, mode, NBLKS[k]);
        cmp_bf16(what, out, ref, osz);
    }
    free(x); free(gamma); free(cosb); free(sinb); free(pos); free(out); free(ref);
}

/* --- FLASH_DECODE + FLASH_MERGE ------------------------------------------------------- */

static void test_flash_decode(uint32_t n_head, uint32_t n_kv_head, uint32_t hd, uint32_t kvlen,
                              uint32_t window, uint32_t nsplit, float scale) {
    const uint32_t stride = 2048, mask = 2047, n_batch = 1;
    plow_bf16* Q = rand_bf16((size_t)n_head * hd, 1.0f);
    plow_bf16* K = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    plow_bf16* V = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    int32_t len = (int32_t)kvlen;
    const size_t npart = (size_t)n_head * nsplit * hd, nml = (size_t)n_head * nsplit * 2;
    float *op = zeros_f32(npart), *opr = zeros_f32(npart), *ml = zeros_f32(nml), *mlr = zeros_f32(nml);
    plow_bf16 *O = zeros_bf16((size_t)n_head * hd), *Or = zeros_bf16((size_t)n_head * hd);
    void* T[6] = {op, ml, Q, K, V, &len};
    PlowDevInst in = inst(PLOW_DOP_FLASH_DECODE);
    for (int k = 0; k < 6; k++) in.t[k] = (uint16_t)k;
    in.i[0] = n_batch; in.i[1] = n_head; in.i[2] = n_kv_head; in.i[3] = stride; in.i[4] = window;
    in.i[5] = nsplit; in.i[6] = hd; in.i[7] = mask;
    in.fj[0].f = scale;
    PlowDevInst mg = inst(PLOW_DOP_FLASH_MERGE);
    mg.t[0] = 0; mg.t[1] = 1; mg.t[2] = 2;
    mg.i[0] = n_batch; mg.i[1] = n_head; mg.i[2] = nsplit; mg.i[3] = hd;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_FLASH_DECODE, g_flash_decode, "flash_decode");
    plow_cpu_kernel_fn fm = avx(PLOW_DOP_FLASH_MERGE, g_flash_merge, "flash_merge");
    for (int k = 0; k < 3; k++) {
        char what[128];
        snprintf(what, sizeof what, "flash_decode heads=%u/%u hd=%u kvlen=%u window=%u nsplit=%u scale=%.2f nblk=%u",
                 n_head, n_kv_head, hd, kvlen, window, nsplit, scale, NBLKS[k]);
        T[0] = opr; T[1] = mlr; run_all(g_flash_decode, &in, NBLKS[k], T);
        T[0] = op; T[1] = ml; run_all(f, &in, NBLKS[k], T);
        cmp_f32(what, ml, mlr, nml, 1e-2);
        cmp_f32(what, op, opr, npart, 1e-2);
        /* Merge: AVX-512 merge over the golden partials vs golden merge, then the full chain. */
        void* TM[3] = {Or, opr, mlr};
        run_all(g_flash_merge, &mg, NBLKS[k], TM);
        TM[0] = O; run_all(fm, &mg, NBLKS[k], TM);
        snprintf(what, sizeof what, "flash_merge heads=%u hd=%u nsplit=%u nblk=%u", n_head, hd, nsplit, NBLKS[k]);
        cmp_bf16(what, O, Or, (size_t)n_head * hd);
        TM[0] = O; TM[1] = op; TM[2] = ml; run_all(fm, &mg, NBLKS[k], TM);
        snprintf(what, sizeof what, "flash_decode+merge heads=%u/%u hd=%u kvlen=%u window=%u nsplit=%u nblk=%u",
                 n_head, n_kv_head, hd, kvlen, window, nsplit, NBLKS[k]);
        cmp_bf16(what, O, Or, (size_t)n_head * hd);
    }
    free(Q); free(K); free(V); free(op); free(opr); free(ml); free(mlr); free(O); free(Or);
}

/* --- FLASH_PREFILL ------------------------------------------------------------------- */

static void test_flash_prefill(uint32_t n_q, uint32_t q_pos0, uint32_t n_head, uint32_t n_kv_head,
                               uint32_t hd, uint32_t window, uint32_t nsplit, int fused) {
    const uint32_t stride = 2048, mask = 2047, n_kv = q_pos0 + n_q;
    plow_bf16* Q = rand_bf16((size_t)n_q * n_head * hd, 1.0f);
    plow_bf16* K = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    plow_bf16* V = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    const size_t npart = (size_t)n_q * n_head * nsplit * hd, nml = (size_t)n_q * n_head * nsplit * 2;
    float *op = zeros_f32(npart), *opr = zeros_f32(npart), *ml = zeros_f32(nml), *mlr = zeros_f32(nml);
    plow_bf16 *O = zeros_bf16((size_t)n_q * n_head * hd), *Or = zeros_bf16((size_t)n_q * n_head * hd);
    void* T[6] = {op, ml, Q, K, V, fused ? (void*)O : NULL};
    PlowDevInst in = inst(PLOW_DOP_FLASH_PREFILL);
    for (int k = 0; k < 5; k++) in.t[k] = (uint16_t)k;
    if (fused) in.t[5] = 5;
    in.i[0] = n_q; in.i[1] = n_kv; in.i[2] = n_head; in.i[3] = n_kv_head; in.i[4] = q_pos0;
    in.i[5] = window; in.i[6] = hd; in.i[7] = nsplit;
    in.fj[0].f = 1.0f / sqrtf((float)hd); /* Gemma uses 1.0 (q-norm absorbs it); keep logits tame */
    in.fj[1].u = stride; in.fj[2].u = mask;
    plow_cpu_kernel_fn f = avx(PLOW_DOP_FLASH_PREFILL, g_flash_prefill, "flash_prefill");
    for (int k = 0; k < 3; k++) {
        char what[160];
        snprintf(what, sizeof what, "flash_prefill n_q=%u q_pos0=%u heads=%u/%u hd=%u window=%u nsplit=%u fused=%d nblk=%u",
                 n_q, q_pos0, n_head, n_kv_head, hd, window, nsplit, fused, NBLKS[k]);
        T[0] = opr; T[1] = mlr; T[5] = fused ? (void*)Or : NULL; run_all(g_flash_prefill, &in, NBLKS[k], T);
        T[0] = op; T[1] = ml; T[5] = fused ? (void*)O : NULL; run_all(f, &in, NBLKS[k], T);
        if (fused && nsplit == 1) {
            cmp_bf16(what, O, Or, (size_t)n_q * n_head * hd);
        } else {
            /* Raw partials differ by the bf16 rounding of P at a blockwise vs per-key max
             * (both self-consistent with their own l); the merged output is the contract. */
            cmp_f32(what, ml, mlr, nml, 3e-2);
            cmp_f32(what, op, opr, npart, 3e-2);
            PlowDevInst mg = inst(PLOW_DOP_FLASH_MERGE);
            mg.t[0] = 0; mg.t[1] = 1; mg.t[2] = 2;
            mg.i[0] = n_q; mg.i[1] = n_head; mg.i[2] = nsplit; mg.i[3] = hd;
            void* TM[3] = {Or, opr, mlr};
            run_all(g_flash_merge, &mg, 1, TM);
            TM[0] = O; TM[1] = op; TM[2] = ml;
            run_all(g_flash_merge, &mg, 1, TM);
            cmp_bf16(what, O, Or, (size_t)n_q * n_head * hd);
        }
    }
    free(Q); free(K); free(V); free(op); free(opr); free(ml); free(mlr); free(O); free(Or);
}

/* --- bench ----------------------------------------------------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_decode(void) {
    const uint32_t n_head = 16, n_kv_head = 1, hd = 512, kvlen = 1024, stride = 2048, mask = 2047;
    plow_bf16* Q = rand_bf16((size_t)n_head * hd, 1.0f);
    plow_bf16* K = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    plow_bf16* V = rand_bf16((size_t)n_kv_head * stride * hd, 1.0f);
    int32_t len = (int32_t)kvlen;
    float *op = zeros_f32((size_t)n_head * hd), *ml = zeros_f32((size_t)n_head * 2);
    void* T[6] = {op, ml, Q, K, V, &len};
    PlowDevInst in = inst(PLOW_DOP_FLASH_DECODE);
    for (int k = 0; k < 6; k++) in.t[k] = (uint16_t)k;
    in.i[0] = 1; in.i[1] = n_head; in.i[2] = n_kv_head; in.i[3] = stride; in.i[4] = 0;
    in.i[5] = 1; in.i[6] = hd; in.i[7] = mask;
    in.fj[0].f = 1.0f;
    plow_cpu_kernel_fn f = plow_cpu_kernel(PLOW_DOP_FLASH_DECODE);
    /* Bytes the op must read: K and V rows for every head group (8 items x kvlen x hd x 2 x 2). */
    const double bytes = 8.0 * kvlen * hd * 2.0 * 2.0;
    for (int which = 0; which < 2; which++) {
        plow_cpu_kernel_fn fn = which ? f : g_flash_decode;
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            const double t0 = now_s();
            run_all(fn, &in, 1, T);
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("flash_decode kvlen=%u hd=%u heads=%u/%u %s: %.1f us/call, %.1f GB/s KV read\n", kvlen, hd,
               n_head, n_kv_head, which ? "avx512" : "golden", best * 1e6, bytes / best / 1e9);
    }
    free(Q); free(K); free(V); free(op); free(ml);
}

int main(int argc, char** argv) {
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AVX512);
    printf("isa tier: %d\n", tier);
    if (tier < PLOW_CPU_ISA_AVX512) {
        printf("no AVX-512 on this host; skipping\n");
        return 0;
    }
    memset(&g_ctx, 0, sizeof g_ctx);
    plow_cpu_thread_init(&g_ctx);
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench_decode();
        return 0;
    }

    /* HEADNORM_ROPE: q path (Gemma q_norm: gamma, rope, out_stride 0), K ring path, batched. */
    test_headnorm_rope(5, 16, 256, 1, 0, 1, 0);
    test_headnorm_rope(38, 16, 512, 1, 0, 1, 0);
    test_headnorm_rope(128, 16, 256, 0, 0, 1, 0);   /* weightless v_norm-style, with rope */
    test_headnorm_rope(38, 8, 256, 1, 1, 1, 0);     /* skip_norm */
    test_headnorm_rope(38, 8, 512, 1, 0, 0, 0);     /* no rope (v_norm) */
    test_headnorm_rope(38, 8, 256, 1, 0, 1, 1);     /* K path into the ring, out_row0 1500 */
    test_headnorm_rope(128, 1, 512, 1, 0, 1, 1);
    test_headnorm_rope(5, 8, 256, 1, 0, 1, 2);      /* batched KV path with pos tensor */
    test_headnorm_rope(6, 4, 64, 1, 0, 1, 0);       /* interleaved (hd 64) */

    /* FLASH_DECODE(+MERGE): sliding (hd256, 8 kv heads, window 1024) and full (hd512, 1 kv head). */
    const uint32_t lens[6] = {1, 5, 37, 129, 1024, 1500};
    for (int li = 0; li < 6; li++) {
        for (uint32_t ns = 1; ns <= 3; ns += 2) {
            test_flash_decode(16, 8, 256, lens[li], 1024, ns, 0.0625f);
            test_flash_decode(16, 1, 512, lens[li], 0, ns, 0.0442f);
        }
    }
    test_flash_decode(16, 1, 512, 129, 0, 1, 1.0f); /* Gemma scale 1.0: sharp softmax */
    test_flash_decode(16, 8, 256, 37, 1024, 3, 1.0f);

    /* FLASH_PREFILL: causal from 0, chunk continuation (q_pos0 100), window wrap, nsplit 1/3. */
    const uint32_t tq[3] = {5, 38, 128};
    for (int ti = 0; ti < 3; ti++) {
        test_flash_prefill(tq[ti], 0, 16, 8, 256, 1024, 1, 1);
        test_flash_prefill(tq[ti], 0, 16, 1, 512, 0, 1, 1);
        test_flash_prefill(tq[ti], 0, 16, 8, 256, 1024, 3, 0);
        test_flash_prefill(tq[ti], 100, 16, 1, 512, 0, 3, 0);
        test_flash_prefill(tq[ti], 100, 16, 8, 256, 1024, 1, 1);
    }
    test_flash_prefill(128, 1372, 16, 8, 256, 1024, 1, 1); /* window wrap around the 2048 ring */
    test_flash_prefill(128, 1372, 16, 8, 256, 1024, 3, 0);
    test_flash_prefill(128, 1372, 16, 1, 512, 0, 3, 0);

    if (fails == 0) printf("all avx512 attention tests passed\n");
    return fails ? 1 : 0;
}
