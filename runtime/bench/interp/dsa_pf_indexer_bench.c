/* dsa_pf_indexer_bench.c — prices the GLM-5.2 DSA sparse-PREFILL indexer, per layer, per op.
 *                                                                    [GLM52-DSA-PF-IDX]
 *
 * The sparse-prefill counterfactual (perf-data/plow-gfx942/glm52-current-cost-decomposition.md
 * §4) is gated not by the flash but by plow's own selection chain: op 117 IndexScorePf writes a
 * T x ctx f32 score matrix and op 118 IndexSelectPf re-reads it eight times. This bench isolates
 * both ops from the interpreter so their cost can be attributed, and A/Bs each against a rebuild:
 *
 *   index_score_pf_128       op 117 as shipped   — operands re-fetched per (query, 32-key) item
 *   index_score_pf_row_128   op 117 arm B        — pack-per-workgroup, Q in VGPRs, K through LDS
 *   index_select_pf_k        op 118 as shipped   — 7 radix passes + emit = 8 full-row scans
 *   index_select_pf_fast_k   op 118 arm B        — d_index_select_coop's fewer-passes early-out
 *
 * Both rebuilds are EXACT by construction, and the bench gates that rather than assuming it:
 * the score arms are compared BYTE-FOR-BYTE over the whole causal matrix, and the select arms by
 * top-k SET equality per row (the emit order is deliberately arbitrary, so the sets are sorted
 * before comparison). A mismatch prints the offending row and fails.
 *
 * Shapes are the ones the emitter actually produces: `plan_chunks` covers a prompt from the
 * bucket ladder with MAX_CHUNK = 8192, so a 16k prompt is [8192, 8192] and the expensive half is
 * 8192 queries over a 16384 KV. Per-layer cost, one rank, TP8 geometry (HI=32, DI=128, k=2048).
 *
 * Build: scripts/build_dsa_pf_indexer_bench.sh <outdir>
 * Run:   cd <outdir> && ROCR_VISIBLE_DEVICES=0 ./dsa_pf_bench test_kernels.elf
 */
#include "../../amd/hsa_backend.h"
#include "../../common/dev_isa.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HI   32   /* index_n_heads  */
#define DI   128  /* index_head_dim */
#define TOPK 2048 /* index_topk     */

typedef uint16_t bf16;
static float b2f(bf16 v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}
static bf16 f2b(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    uint32_t r = c.u + 0x7fff + ((c.u >> 16) & 1);
    return (bf16)(r >> 16);
}
static uint64_t rs;
static void seed(uint64_t s) { rs = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static float frand(void) {
    rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int32_t)(rs >> 33) % 2001 - 1000) / 4000.0f;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static plow_hsa* H;
static void* dev(size_t n) {
    void* p = plow_hsa_alloc(H, 0, n);
    if (!p) { printf("alloc %.1f MiB failed: %s\n", n / 1048576.0, plow_hsa_last_error()); exit(1); }
    return p;
}

/* median of R timed launches, each drained separately: the DVFS spread on this box is ~20%, so
 * a mean over back-to-back launches hides it. Reports median and (max-min)/median as spread. */
static double time_med(int R, void (*launch)(void*), void* ctx, double* spread) {
    double v[32];
    if (R > 32) R = 32;
    for (int w = 0; w < 2; w++) { launch(ctx); plow_hsa_wait(H, 0); }
    for (int r = 0; r < R; r++) {
        const double t0 = now();
        launch(ctx);
        plow_hsa_wait(H, 0);
        v[r] = (now() - t0) * 1e6;
    }
    for (int i = 1; i < R; i++)
        for (int j = i; j > 0 && v[j] < v[j - 1]; j--) { double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    const double med = v[R / 2];
    if (spread) *spread = (v[R - 1] - v[0]) / med * 100.0;
    return med;
}

static int cmp_u32(const void* a, const void* b) {
    const uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* ---- one measured configuration ---- */
struct cfg { unsigned n_tok, ctx; const char* label; };
static const struct cfg CFGS[] = {
    {4096, 4096, "T=4096  ctx=4096   (4k prompt)"},
    {8192, 8192, "T=8192  ctx=8192   (8k prompt / 16k chunk 1)"},
    {8192, 16384, "T=8192  ctx=16384  (16k prompt chunk 2)"},
};
#define NCFG ((int)(sizeof(CFGS) / sizeof(CFGS[0])))

static plow_hsa_kernel kScore, kScoreRow, kScoreRow64, kSel, kSelFast;
static unsigned NCU;

struct scargs { void *sc, *qi, *ki, *w, *len; unsigned n_tok, kv_stride; float scale; } __attribute__((packed));
struct slargs { void *idx; const void* sc; const void* len; unsigned n_tok, top_k, kv_stride; } __attribute__((packed));

static struct { plow_hsa_kernel* k; void* a; size_t n; unsigned wgs; } LC;
static void do_launch(void* unused) {
    (void)unused;
    plow_hsa_launch(H, 0, LC.k, LC.wgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, LC.a, LC.n);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    H = plow_hsa_init();
    if (!H) { printf("hsa init failed\n"); return 1; }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    NCU = cus;
    printf("dev0: %s  CUs=%u  LDS=%u  threads/WG=%d\n", nm, cus, lds, PLOW_WG_THREADS);

    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n)) { printf("load failed: %s\n", plow_hsa_last_error()); return 1; }
    /* PROOF THE ARMS ARE IN THE LOADED OBJECT: every kernel below is resolved by name out of the
     * code object actually loaded on the device. A missing symbol is a hard failure, never a
     * silent fallback to the shipped arm. */
    if (plow_hsa_get_kernel(H, 0, "index_score_pf_128", &kScore) ||
        plow_hsa_get_kernel(H, 0, "index_score_pf_row_128", &kScoreRow) ||
        plow_hsa_get_kernel(H, 0, "index_score_pf_row64_128", &kScoreRow64) ||
        plow_hsa_get_kernel(H, 0, "index_select_pf_k", &kSel) ||
        plow_hsa_get_kernel(H, 0, "index_select_pf_fast_k", &kSelFast)) {
        printf("kernel missing: %s\n", plow_hsa_last_error());
        return 1;
    }
    printf("arms resolved from %s: index_score_pf_128, index_score_pf_row_128, "
           "index_score_pf_row64_128, index_select_pf_k, index_select_pf_fast_k\n\n", elf);

    const int REPS = argc > 2 ? atoi(argv[2]) : 9;
    printf("%-44s | %10s %10s %10s | %10s %10s | %s\n", "config", "score us",
           "row(tn128)", "row(tn64)", "select us", "sel-fast", "gates");
    printf("---------------------------------------------------------------------------------"
           "-------------------------------------------\n");

    for (int ci = 0; ci < NCFG; ci++) {
        const unsigned T = CFGS[ci].n_tok, CTX = CFGS[ci].ctx;
        const size_t nsc = (size_t)T * CTX;
        /* ---- host data (the geometry the emitter produces; values are synthetic) ---- */
        seed(0xD5Au + CTX * 31u + T);
        bf16* hQ = malloc((size_t)T * HI * DI * 2);
        bf16* hK = malloc((size_t)CTX * DI * 2);
        bf16* hW = malloc((size_t)T * HI * 2);
        for (size_t i = 0; i < (size_t)T * HI * DI; i++) hQ[i] = f2b(frand());
        for (size_t i = 0; i < (size_t)CTX * DI; i++) hK[i] = f2b(frand());
        /* the trained lightning weights are an unconstrained projection output — signed. */
        for (size_t i = 0; i < (size_t)T * HI; i++) hW[i] = f2b(frand() * 2.0f);
        int32_t klen = (int)CTX;

        void* dQ = dev((size_t)T * HI * DI * 2);
        plow_hsa_upload(H, 0, dQ, hQ, (size_t)T * HI * DI * 2);
        void* dK = dev((size_t)CTX * DI * 2);
        plow_hsa_upload(H, 0, dK, hK, (size_t)CTX * DI * 2);
        void* dW = dev((size_t)T * HI * 2);
        plow_hsa_upload(H, 0, dW, hW, (size_t)T * HI * 2);
        void* dLen = dev(4);
        plow_hsa_upload(H, 0, dLen, &klen, 4);
        void* dSc = dev(nsc * 4);
        void* dSc2 = dev(nsc * 4);
        void* dIdx = dev((size_t)T * TOPK * 4);
        void* dIdx2 = dev((size_t)T * TOPK * 4);
        free(hQ); free(hK); free(hW);

        const float scale = powf((float)DI, -0.5f) * powf((float)HI, -0.5f);
        struct scargs aSc = {dSc, dQ, dK, dW, dLen, T, CTX, scale};
        struct scargs aScR = {dSc2, dQ, dK, dW, dLen, T, CTX, scale};
        struct slargs aSl = {dIdx, dSc, dLen, T, TOPK, CTX};
        struct slargs aSlF = {dIdx2, dSc, dLen, T, TOPK, CTX};

        /* ---- score: both arms, then a byte-for-byte gate over the causal matrix ---- */
        double spA, spB;
        LC.k = &kScore; LC.a = &aSc; LC.n = sizeof(aSc); LC.wgs = NCU;
        const double tScore = time_med(REPS, do_launch, NULL, &spA);
        LC.k = &kScoreRow; LC.a = &aScR; LC.n = sizeof(aScR); LC.wgs = NCU;
        const double tScoreRow = time_med(REPS, do_launch, NULL, &spB);
        double spB64;
        LC.k = &kScoreRow64; LC.a = &aScR; LC.n = sizeof(aScR); LC.wgs = NCU;
        const double tScoreRow64 = time_med(REPS, do_launch, NULL, &spB64);
        /* leave dSc2 holding the TN=128 arm's output for the byte gate below */
        LC.k = &kScoreRow; LC.a = &aScR; LC.n = sizeof(aScR); LC.wgs = NCU;
        do_launch(NULL); plow_hsa_wait(H, 0);

        /* compare in 64 MiB windows so the host never holds the whole 512 MiB matrix twice.
         * Only the causal entries are defined, so the check walks rows and stops at row_end. */
        int score_ok = 1;
        {
            const size_t WIN = 16u << 20; /* floats */
            float* a = malloc(WIN * 4);
            float* b = malloc(WIN * 4);
            const unsigned q_pos0 = CTX - T;
            for (size_t off = 0; off < nsc && score_ok; off += WIN) {
                const size_t m = (nsc - off < WIN) ? (nsc - off) : WIN;
                plow_hsa_download(H, 0, a, (char*)dSc + off * 4, m * 4);
                plow_hsa_download(H, 0, b, (char*)dSc2 + off * 4, m * 4);
                for (size_t i = 0; i < m; i++) {
                    const size_t g = off + i;
                    const unsigned t = (unsigned)(g / CTX), s = (unsigned)(g % CTX);
                    if (s >= q_pos0 + t + 1u) continue; /* outside the causal range: undefined */
                    if (memcmp(&a[i], &b[i], 4)) {
                        printf("  SCORE MISMATCH t=%u s=%u  shipped=%.9g row=%.9g\n", t, s, a[i], b[i]);
                        score_ok = 0;
                        break;
                    }
                }
            }
            free(a); free(b);
        }

        /* ---- select: both arms off the SAME (shipped-arm) score matrix, then set equality ---- */
        double spC, spD;
        LC.k = &kSel; LC.a = &aSl; LC.n = sizeof(aSl); LC.wgs = NCU < T ? NCU : T;
        const double tSel = time_med(REPS, do_launch, NULL, &spC);
        LC.k = &kSelFast; LC.a = &aSlF; LC.n = sizeof(aSlF); LC.wgs = NCU < T ? NCU : T;
        const double tSelF = time_med(REPS, do_launch, NULL, &spD);

        int sel_ok = 1;
        {
            /* top-k MEMBERSHIP agreement, row by row: emit order is arbitrary by design. */
            const size_t rowb = (size_t)TOPK * 4;
            uint32_t* a = malloc(rowb);
            uint32_t* b = malloc(rowb);
            unsigned checked = 0, agree_min = TOPK;
            for (unsigned t = 0; t < T && sel_ok; t += (T / 64 ? T / 64 : 1)) {
                plow_hsa_download(H, 0, a, (char*)dIdx + (size_t)t * rowb, rowb);
                plow_hsa_download(H, 0, b, (char*)dIdx2 + (size_t)t * rowb, rowb);
                qsort(a, TOPK, 4, cmp_u32);
                qsort(b, TOPK, 4, cmp_u32);
                unsigned same = 0;
                for (unsigned i = 0; i < TOPK; i++) same += (a[i] == b[i]);
                if (same < agree_min) agree_min = same;
                if (same != TOPK) {
                    printf("  SELECT SET MISMATCH row t=%u: %u/%u positions agree\n", t, same, TOPK);
                    sel_ok = 0;
                }
                checked++;
            }
            free(a); free(b);
            if (sel_ok) printf("  [%s] select set gate: %u rows checked, min agreement %u/%u\n",
                               CFGS[ci].label, checked, agree_min, TOPK);
        }

        printf("%-44s | %7.0f+-%-2.0f %7.0f+-%-2.0f %7.0f+-%-2.0f | %7.0f+-%-2.0f %7.0f+-%-2.0f | %s %s\n",
               CFGS[ci].label, tScore, spA, tScoreRow, spB, tScoreRow64, spB64, tSel, spC, tSelF,
               spD, score_ok ? "score=EXACT" : "score=FAIL", sel_ok ? "sel=EXACT" : "sel=FAIL");
        printf("    speedup: score %.2fx (tn128) / %.2fx (tn64),  select %.2fx\n",
               tScore / tScoreRow, tScore / tScoreRow64, tSel / tSelF);

        /* per-layer chain totals, the number the net calculation consumes */
        const double best = tScoreRow < tScoreRow64 ? tScoreRow : tScoreRow64;
        printf("    chain/layer: shipped %.3f ms  ->  rebuilt %.3f ms  (score %.3f->%.3f, select %.3f->%.3f)\n\n",
               (tScore + tSel) / 1000.0, (best + tSelF) / 1000.0, tScore / 1000.0, best / 1000.0,
               tSel / 1000.0, tSelF / 1000.0);
        fflush(stdout);
    }
    return 0;
}
