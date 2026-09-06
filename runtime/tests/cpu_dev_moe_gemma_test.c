/* Gemma-4 26B-A4B hybrid MoE ops vs f64 references at a small geometry: decode chain 69 -> 68 ->
 * 71 -> 63 -> 70 over B rows, and the prefill chain 73 -> 74 -> 75 -> 76 -> 77 over T tokens, which
 * must reproduce the decode chain token by token. Fused expert weights sit behind a host `ewt`
 * pointer table exactly as the engine builds it. */
#define _GNU_SOURCE /* --bench: pthread_setaffinity_np */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu_dev.h"
#include "golden/gptoss.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double nr(void) { return sqrt(-2 * log(ur() + 1e-12)) * cos(6.283185307 * ur()); }
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void fill_bf16(plow_bf16* p, size_t n, double sc) { for (size_t i = 0; i < n; i++) p[i] = plow_f2bf((float)(nr() * sc)); }
static void run_op(uint16_t op, PlowDevInst* in, void** T, uint32_t nblk, PlowCpuCtx* ctx) {
    plow_cpu_kernel_fn f = plow_cpu_kernel(op);
    in->op = op; in->blocks = (uint16_t)nblk;
    for (uint32_t s = 0; s < nblk; s++) f(in, s, nblk, T, ctx);
}
static PlowDevInst inst(void) { PlowDevInst in; memset(&in, 0, sizeof in); for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE; return in; }
static double b2d(plow_bf16 v) { return plow_bf2f(v); }
static double gelu(double x) { return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

/* ---- geometry ---- */
#ifndef MOE_BIG
enum { H = 64, I = 24, E = 6, K_ = 2, B = 3, T_ = 5 };
#else /* AMX grouped prefill: K % 32 == 0, > 32 rows per expert, partial weight tiles */
enum { H = 64, I = 32, E = 4, K_ = 2, B = 3, T_ = 72 };
#endif

typedef struct {
    plow_bf16 *gu[E], *dn[E];   /* fused gate_up [2I][H], down [H][I] */
    uint64_t ewt[E * 2];
    plow_bf16 proj[E * H], scale[H], pes[E], gamma[H], g_pf2[H];
} weights_t;

static void make_weights(weights_t* w) {
    for (int e = 0; e < E; e++) {
        w->gu[e] = malloc((size_t)2 * I * H * 2); w->dn[e] = malloc((size_t)H * I * 2);
        fill_bf16(w->gu[e], (size_t)2 * I * H, 0.3); fill_bf16(w->dn[e], (size_t)H * I, 0.3);
        w->ewt[2 * e] = (uint64_t)(uintptr_t)w->gu[e]; w->ewt[2 * e + 1] = (uint64_t)(uintptr_t)w->dn[e];
    }
    fill_bf16(w->proj, E * H, 0.2);
    for (int h = 0; h < H; h++) { w->scale[h] = plow_f2bf((float)(1.0 + 0.1 * nr())); w->gamma[h] = plow_f2bf((float)(1.0 + 0.1 * nr())); w->g_pf2[h] = plow_f2bf((float)(1.0 + 0.1 * nr())); }
    for (int e = 0; e < E; e++) w->pes[e] = plow_f2bf((float)(0.8 + 0.4 * ur()));
}

/* f64 reference of ONE token: routing table + final output (combine-norm with residual h1). */
static void ref_token(const weights_t* w, const plow_bf16* resid, const plow_bf16* h1, double root, double eps,
                      uint32_t* eid, double* gate, double* out) {
    double ss = 0; for (int h = 0; h < H; h++) ss += b2d(resid[h]) * b2d(resid[h]);
    const double inv = 1.0 / sqrt(ss / H + eps);
    double h2[H], sc[E];
    for (int h = 0; h < H; h++) h2[h] = b2d(resid[h]) * inv * b2d(w->scale[h]) * root;
    double m = -1e300;
    for (int e = 0; e < E; e++) { double a = 0; for (int h = 0; h < H; h++) a += h2[h] * b2d(w->proj[e * H + h]); sc[e] = a; if (a > m) m = a; }
    double s = 0; for (int e = 0; e < E; e++) { sc[e] = exp(sc[e] - m); s += sc[e]; }
    for (int e = 0; e < E; e++) sc[e] /= s;
    double gs = 0;
    for (int j = 0; j < K_; j++) {
        int best = -1; double bv = -1e300;
        for (int e = 0; e < E; e++) if (sc[e] > bv) { bv = sc[e]; best = e; }
        eid[j] = (uint32_t)best; gate[j] = sc[best]; gs += sc[best]; sc[best] = -1e300;
    }
    for (int j = 0; j < K_; j++) gate[j] = gate[j] / gs * b2d(w->pes[eid[j]]);
    /* experts */
    double xn[H]; for (int h = 0; h < H; h++) xn[h] = b2d(resid[h]) * inv * b2d(w->gamma[h]);
    double sum[H]; memset(sum, 0, sizeof sum);
    for (int j = 0; j < K_; j++) {
        const plow_bf16* gu = w->gu[eid[j]]; const plow_bf16* dn = w->dn[eid[j]];
        double fu[I];
        for (int n = 0; n < I; n++) {
            double g = 0, u = 0;
            for (int h = 0; h < H; h++) { g += b2d(gu[n * H + h]) * xn[h]; u += b2d(gu[(I + n) * H + h]) * xn[h]; }
            fu[n] = b2d(plow_f2bf((float)(gelu(g) * u))); /* fu is stored bf16 by the kernels */
        }
        for (int h = 0; h < H; h++) { double a = 0; for (int n = 0; n < I; n++) a += b2d(dn[h * I + n]) * fu[n]; sum[h] += gate[j] * a; }
    }
    double s2 = 0; for (int h = 0; h < H; h++) s2 += sum[h] * sum[h];
    const double inv2 = 1.0 / sqrt(s2 / H + eps);
    for (int h = 0; h < H; h++) out[h] = sum[h] * inv2 * b2d(w->g_pf2[h]) + b2d(h1[h]);
}

static void cmp_out(const char* what, const plow_bf16* got, const double* want, int n, double tol) {
    int bad = 0; double worst = 0;
    for (int i = 0; i < n; i++) { const double d = fabs(b2d(got[i]) - want[i]); const double rel = d / fmax(1.0, fabs(want[i])); if (rel > worst) worst = rel; if (rel > tol) bad++; }
    CHECK(bad == 0, "%s: %d/%d off (worst rel %.3g)", what, bad, n, worst);
}

/* ---- --bench: ops 71/63 at the 26B-A4B shape (H=2816, I=704, k=8, E=128) on 16 threads, B = 1 and
 * B = 8 (64 slots over ~50 distinct experts). Reports ms/step and GB/s of expert bytes streamed
 * (distinct experts) plus the per-slot-equivalent rate. PLOW_MOE_GEMMA_NOGROUP=1 for the A/B. ---- */
#include <pthread.h>
#include <time.h>
enum { BN_H = 2816, BN_I = 704, BN_K = 8, BN_E = 128, BN_NT = 16, BN_IT = 41 };
typedef struct {
    pthread_barrier_t bar;
    PlowDevInst in71, in63;
    void* T71[8]; void* T63[8];
    plow_moe_route* tabs[BN_IT]; /* a fresh routing per step so the expert weights stream from DRAM */
    double t71[BN_IT], t63[BN_IT];
} bench_t;
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void* bench_thread(void* arg) {
    bench_t* b = (bench_t*)((uintptr_t)arg & ~(uintptr_t)0xFF);
    const uint32_t t = (uint32_t)((uintptr_t)arg & 0xFF);
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    PlowCpuCtx ctx; memset(&ctx, 0, sizeof ctx);
    ctx.scratch_bytes = plow_cpu_scratch_bytes(); ctx.scratch = aligned_alloc(64, ctx.scratch_bytes);
    plow_cpu_thread_init(&ctx);
    plow_cpu_kernel_fn f71 = plow_cpu_kernel(PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA), f63 = plow_cpu_kernel(PLOW_DOP_MOE_EXPERT_DOWN_GEMMA);
    for (int it = 0; it < BN_IT; it++) {
        if (t == 0) { b->T71[2] = b->tabs[it]; b->T63[2] = b->tabs[it]; } /* others wait at the barrier */
        pthread_barrier_wait(&b->bar);
        double t0 = now_s();
        f71(&b->in71, t, BN_NT, b->T71, &ctx);
        pthread_barrier_wait(&b->bar);
        double t1 = now_s();
        f63(&b->in63, t, BN_NT, b->T63, &ctx);
        pthread_barrier_wait(&b->bar);
        if (t == 0) { b->t71[it] = t1 - t0; b->t63[it] = now_s() - t1; }
    }
    free(ctx.scratch);
    return NULL;
}
static void fill_fast(plow_bf16* p, size_t n, float sc) {
    for (size_t i = 0; i < n; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; p[i] = plow_f2bf(((float)(int32_t)(rs >> 32) * (1.0f / 2147483648.0f)) * sc); }
}
static int cmp_dbl(const void* a, const void* b) { const double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static void bench_one(uint32_t B, uint64_t* ewt, plow_bf16* gamma) {
    const uint32_t ns = B * BN_K;
    plow_bf16* resid = malloc((size_t)B * BN_H * 2); fill_fast(resid, (size_t)B * BN_H, 1.0f);
    bench_t* b = aligned_alloc(256, (sizeof *b + 255) & ~(size_t)255); memset(b, 0, sizeof *b);
    uint8_t used[BN_E], prev[BN_E]; /* prev: experts of the last 2 steps, avoided while E allows (no L3 reuse) */
    memset(prev, 0, sizeof prev);
    const int avoid = 3u * ns <= BN_E;
    double nd = 0;
    for (int it = 0; it < BN_IT; it++) {
        plow_moe_route* tab = b->tabs[it] = malloc(ns * sizeof *tab);
        for (uint32_t r = 0; r < B; r++) {
            memset(used, 0, sizeof used);
            for (uint32_t j = 0; j < BN_K; j++) {
                uint32_t e; do { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; e = (uint32_t)(rs >> 40) % BN_E; } while (used[e] || (avoid && prev[e]));
                used[e] = 1; tab[r * BN_K + j].eid = e; tab[r * BN_K + j].gate = 0.125f;
            }
        }
        for (int e = 0; e < BN_E; e++) prev[e] = prev[e] == 1 ? 2 : 0;
        memset(used, 0, sizeof used);
        for (uint32_t s = 0; s < ns; s++) if (!used[tab[s].eid]) { used[tab[s].eid] = 1; prev[tab[s].eid] = 1; if (it) nd += 1.0 / (BN_IT - 1); }
    }
    plow_bf16* fu = malloc((size_t)ns * BN_I * 2); float* part = malloc((size_t)ns * BN_H * 4);
    pthread_barrier_init(&b->bar, NULL, BN_NT);
    b->in71 = inst(); b->in71.op = PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA; b->in71.blocks = BN_NT;
    for (int i = 0; i < 5; i++) b->in71.t[i] = (uint16_t)i;
    b->in71.i[0] = BN_K; b->in71.i[1] = BN_I; b->in71.i[2] = BN_H; b->in71.i[3] = BN_E; b->in71.i[5] = B; b->in71.fj[0].f = 1e-6f;
    b->T71[0] = fu; b->T71[1] = resid; b->T71[3] = ewt; b->T71[4] = gamma;
    b->in63 = inst(); b->in63.op = PLOW_DOP_MOE_EXPERT_DOWN_GEMMA; b->in63.blocks = BN_NT;
    for (int i = 0; i < 4; i++) b->in63.t[i] = (uint16_t)i;
    b->in63.i[0] = BN_K; b->in63.i[1] = BN_H; b->in63.i[2] = BN_I; b->in63.i[3] = BN_E; b->in63.i[5] = B;
    b->T63[0] = part; b->T63[1] = fu; b->T63[3] = ewt;
    pthread_t th[BN_NT];
    for (uint32_t t = 0; t < BN_NT; t++) pthread_create(&th[t], NULL, bench_thread, (void*)((uintptr_t)b | t));
    for (uint32_t t = 0; t < BN_NT; t++) pthread_join(th[t], NULL);
    const double gu_b = 2.0 * BN_I * BN_H * 2, dn_b = (double)BN_H * BN_I * 2;
    qsort(b->t71 + 1, BN_IT - 1, sizeof(double), cmp_dbl); qsort(b->t63 + 1, BN_IT - 1, sizeof(double), cmp_dbl);
    const double m71 = b->t71[1], q71 = b->t71[1 + (BN_IT - 1) / 4], d71 = b->t71[1 + (BN_IT - 1) / 2];
    const double m63 = b->t63[1], q63 = b->t63[1 + (BN_IT - 1) / 4], d63 = b->t63[1 + (BN_IT - 1) / 2];
    printf("B=%u slots=%u distinct~%.1f | 71 GLU  min %.3f p25 %.3f med %.3f ms | at med: %.1f GB/s distinct (%.1f per-slot-equiv)\n", B, ns, nd, m71 * 1e3, q71 * 1e3, d71 * 1e3, nd * gu_b / d71 / 1e9, ns * gu_b / d71 / 1e9);
    printf("B=%u slots=%u distinct~%.1f | 63 DOWN min %.3f p25 %.3f med %.3f ms | at med: %.1f GB/s distinct (%.1f per-slot-equiv)\n", B, ns, nd, m63 * 1e3, q63 * 1e3, d63 * 1e3, nd * dn_b / d63 / 1e9, ns * dn_b / d63 / 1e9);
    pthread_barrier_destroy(&b->bar);
    for (int it = 0; it < BN_IT; it++) free(b->tabs[it]);
    free(b); free(resid); free(fu); free(part);
}
static int bench(void) {
    plow_cpu_init(PLOW_CPU_ISA_AMX);
    printf("bench: 71/63 on tier %d %d, %d threads, H=%d I=%d k=%d E=%d\n", plow_cpu_tier_of(71), plow_cpu_tier_of(63), BN_NT, BN_H, BN_I, BN_K, BN_E);
    uint64_t* ewt = malloc(BN_E * 2 * sizeof *ewt);
    for (int e = 0; e < BN_E; e++) {
        plow_bf16* gu = aligned_alloc(64, (size_t)2 * BN_I * BN_H * 2); plow_bf16* dn = aligned_alloc(64, (size_t)BN_H * BN_I * 2);
        fill_fast(gu, (size_t)2 * BN_I * BN_H, 0.05f); fill_fast(dn, (size_t)BN_H * BN_I, 0.05f);
        ewt[2 * e] = (uint64_t)(uintptr_t)gu; ewt[2 * e + 1] = (uint64_t)(uintptr_t)dn;
    }
    plow_bf16 gamma[BN_H]; fill_fast(gamma, BN_H, 1.0f);
    bench_one(1, ewt, gamma);
    bench_one(8, ewt, gamma);
    bench_one(32, ewt, gamma);
    for (int e = 0; e < BN_E; e++) { free((void*)(uintptr_t)ewt[2 * e]); free((void*)(uintptr_t)ewt[2 * e + 1]); }
    free(ewt);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "--bench")) return bench();
    const int tier = plow_cpu_init(PLOW_CPU_ISA_AMX);
    printf("tier %d: 69/71/63/75/76 on tier %d %d %d %d %d\n", tier, plow_cpu_tier_of(69), plow_cpu_tier_of(71),
           plow_cpu_tier_of(63), plow_cpu_tier_of(75), plow_cpu_tier_of(76));
    PlowCpuCtx ctx; memset(&ctx, 0, sizeof ctx);
    ctx.scratch_bytes = plow_cpu_scratch_bytes(); ctx.scratch = aligned_alloc(64, ctx.scratch_bytes);
    plow_cpu_thread_init(&ctx);
    const uint16_t need[] = {63, 68, 69, 70, 71, 73, 74, 75, 76, 77};
    for (size_t i = 0; i < sizeof(need) / sizeof(*need); i++) if (!plow_cpu_has(need[i])) { printf("missing op %u\n", need[i]); fails++; }
    weights_t w; make_weights(&w);
    const float root = 1.0f / sqrtf((float)H), eps = 1e-6f;

    /* rows: resid [T][H], h1 [T][H]; decode uses the first B rows as a batch */
    plow_bf16 resid[T_ * H], h1[T_ * H];
    fill_bf16(resid, T_ * H, 1.0); fill_bf16(h1, T_ * H, 1.0);
    uint32_t ref_eid[T_][K_]; double ref_gate[T_][K_], ref_out[T_][H];
    for (int t = 0; t < T_; t++) ref_token(&w, resid + t * H, h1 + t * H, root, eps, ref_eid[t], ref_gate[t], ref_out[t]);

    for (uint32_t nblk = 1; nblk <= 16; nblk *= 4) {
        /* ---- decode chain over nrow rows: 1 = per-slot 71/63, B and T_ = grouped by expert (T_ rows
         * over E experts share experts, > 8 slots per expert) ---- */
        const int nrows[3] = {1, B, T_};
        for (int nr = 0; nr < 3; nr++) {
        const int NR = nrows[nr];
        float score[T_ * E]; plow_moe_route tab[T_ * K_]; plow_bf16 fu[T_ * K_ * I]; float part[T_ * K_ * H]; plow_bf16 out[T_ * H];
        PlowDevInst in = inst();
        void* T1[8] = {score, resid, w.proj, w.scale, NULL, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.i[0] = H; in.i[1] = E; in.i[2] = NR; in.fj[0].f = root; in.fj[1].f = eps;
        run_op(PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST, &in, T1, nblk, &ctx);
        in = inst(); void* T2[8] = {tab, score, w.pes, NULL, NULL, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.i[1] = E; in.i[2] = K_; in.i[3] = NR;
        run_op(PLOW_DOP_MOE_ROUTER_GEMMA_TOPK, &in, T2, nblk, &ctx);
        for (int b = 0; b < NR; b++) for (int j = 0; j < K_; j++) {
            CHECK(tab[b * K_ + j].eid == ref_eid[b][j], "topk row %d slot %d: %u want %u (nblk %u)", b, j, tab[b * K_ + j].eid, ref_eid[b][j], nblk);
            CHECK(fabs(tab[b * K_ + j].gate - ref_gate[b][j]) < 1e-3, "gate row %d slot %d: %g want %g", b, j, tab[b * K_ + j].gate, ref_gate[b][j]);
        }
        in = inst(); void* T3[8] = {fu, resid, tab, w.ewt, w.gamma, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.t[4] = 4; in.i[0] = K_; in.i[1] = I; in.i[2] = H; in.i[3] = E; in.i[5] = NR; in.fj[0].f = eps;
        run_op(PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA, &in, T3, nblk, &ctx);
        in = inst(); void* T4[8] = {part, fu, tab, w.ewt, NULL, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.i[0] = K_; in.i[1] = H; in.i[2] = I; in.i[3] = E; in.i[5] = NR;
        run_op(PLOW_DOP_MOE_EXPERT_DOWN_GEMMA, &in, T4, nblk, &ctx);
        in = inst(); void* T5[8] = {out, part, h1, w.g_pf2, NULL, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.i[0] = H; in.i[1] = K_; in.i[2] = NR; in.fj[0].f = eps;
        run_op(PLOW_DOP_MOE_COMBINE_NORM_GEMMA, &in, T5, nblk, &ctx);
        /* B >= 2 runs the grouped kernels with a bf16 xn (the prefill xn2 numerics, = PLOW_MOE_XN_BF16=1) → the prefill tolerance */
        for (int b = 0; b < NR; b++) { char what[64]; snprintf(what, sizeof what, "decode out row %d/%d nblk %u", b, NR, nblk); cmp_out(what, out + b * H, ref_out[b], H, NR > 1 ? 5e-2 : 2e-2); }
        }

        /* ---- prefill chain over T tokens ---- */
        plow_moe_route ptab[T_ * K_]; int32_t meta[3 * E + 1];
        const uint32_t rows_max = T_ * K_ + E * 63u;
        uint32_t* row_token = malloc(rows_max * 4); uint32_t* row_partidx = malloc(rows_max * 4); float* row_gate = malloc(rows_max * 4);
        plow_bf16* fug = calloc((size_t)rows_max * I, 2); float* ppart = calloc((size_t)T_ * K_ * H, 4); plow_bf16 pout[T_ * H];
        PlowDevInst in = inst(); void* P1[8] = {ptab, resid, w.proj, w.scale, w.pes, NULL, NULL, NULL};
        in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3; in.t[4] = 4; in.i[0] = H; in.i[1] = E; in.i[2] = K_; in.i[3] = T_; in.fj[0].f = root; in.fj[1].f = eps;
        run_op(PLOW_DOP_MOE_ROUTER_GEMMA_PF, &in, P1, nblk, &ctx);
        for (int t = 0; t < T_; t++) for (int j = 0; j < K_; j++) CHECK(ptab[t * K_ + j].eid == ref_eid[t][j], "pf topk tok %d slot %d", t, j);
        in = inst(); void* P2[8] = {meta, ptab, row_token, row_partidx, row_gate, NULL, NULL, NULL};
        for (int i = 0; i < 5; i++) { in.t[i] = (uint16_t)i; } in.i[0] = T_; in.i[1] = E; in.i[2] = K_;
        run_op(PLOW_DOP_MOE_ALIGN_GEMMA_PF, &in, P2, nblk, &ctx);
        in = inst(); void* P3[8] = {fug, resid, w.ewt, meta, row_token, NULL, NULL, NULL};
        for (int i = 0; i < 5; i++) { in.t[i] = (uint16_t)i; } in.i[0] = I; in.i[1] = H; in.i[2] = E; in.i[5] = 0;
        /* 75 consumes xn2 = RMSNorm(resid, gamma) as bf16 — stage it like the emitter's RmsNorm would */
        plow_bf16 xn2[T_ * H];
        for (int t = 0; t < T_; t++) { double ss = 0; for (int h = 0; h < H; h++) ss += b2d(resid[t * H + h]) * b2d(resid[t * H + h]); const double inv = 1.0 / sqrt(ss / H + eps); for (int h = 0; h < H; h++) xn2[t * H + h] = plow_f2bf((float)(b2d(resid[t * H + h]) * inv * b2d(w.gamma[h]))); }
        P3[1] = xn2;
        run_op(PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF, &in, P3, nblk, &ctx);
        in = inst(); void* P4[8] = {ppart, fug, w.ewt, meta, row_partidx, row_gate, NULL, NULL};
        for (int i = 0; i < 6; i++) { in.t[i] = (uint16_t)i; } in.i[0] = H; in.i[1] = I; in.i[2] = E;
        run_op(PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF, &in, P4, nblk, &ctx);
        in = inst(); void* P5[8] = {pout, ppart, h1, w.g_pf2, NULL, NULL, NULL, NULL};
        for (int i = 0; i < 4; i++) { in.t[i] = (uint16_t)i; } in.i[0] = H; in.i[1] = K_; in.i[2] = T_; in.fj[0].f = eps;
        run_op(PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF, &in, P5, nblk, &ctx);
        /* xn2 is bf16-rounded on the prefill path (decode keeps xn in f32) → looser tolerance */
        for (int t = 0; t < T_; t++) { char what[64]; snprintf(what, sizeof what, "prefill out tok %d nblk %u", t, nblk); cmp_out(what, pout + t * H, ref_out[t], H, 5e-2); }
        free(row_token); free(row_partidx); free(row_gate); free(fug); free(ppart);
    }
    printf(fails ? "cpu_dev_moe_gemma: %d failures\n" : "cpu_dev_moe_gemma: all passed\n", fails);
    return fails ? 1 : 0;
}
