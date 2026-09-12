/* dsa_select_split_test.c — the gated split-row batched decode selection (op 59 i[4]=2,
 * PLOW_DSA_SELECT_SPLIT: three packets, run here as three launches) against a host top-k, with the
 * local form it replaces (i[4]=1, one workgroup per row) alongside. All are test_kernels.hip
 * wrappers around the device functions interp.hip calls.
 *
 * Gates, per row:
 *   split LIST == the host top-k as ascending positions (score desc, lowest index first; a row of
 *                 len <= top_k is the identity padded with -1), element for element;
 *   split LIST identical on a second run over the same strips (they must be left clean);
 *   local SET  == the host set (its list order comes from LDS atomic slots: reported, not gated).
 * Scores are quantized so the k-th score is tied across many positions, plus continuous, negative
 * and all-equal rows, with lengths at and around top_k. Also prints local vs split time.
 *
 * Build (gfx942):
 *   hipcc --offload-arch=gfx942 -O3 -w --genco -DPLOW_DSA_SELECT_SPLIT=1 \
 *       -Iruntime/amd -Iruntime/common runtime/amd/test_kernels.hip -o tk.co
 *   clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
 *       --input=tk.co --output=test_kernels.elf
 *   gcc -O2 -std=gnu11 -o dsa_select_split_test runtime/bench/interp/dsa_select_split_test.c \
 *       runtime/amd/hsa_backend.c -I$ROCM_PATH/include -L$ROCM_PATH/lib -lhsa-runtime64
 * Run: ./dsa_select_split_test test_kernels.elf   (exit 0 = every gate passed)
 */
#include "../../amd/hsa_backend.h"
#include "../../common/dev_isa.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 20u
#define LMAX 81920u
#define TOPK 2048u
#define NB 4096u
#define CTL 16u
#define WORDS ((LMAX + 31u) / 32u)

static plow_hsa* H;
static uint64_t rs;
static uint32_t rnd(void) {
    rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rs >> 33);
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static const float* g_row;
static int by_rank(const void* a, const void* b) {
    const int x = *(const int*)a, y = *(const int*)b;
    if (g_row[x] != g_row[y]) return g_row[x] > g_row[y] ? -1 : 1;
    return x - y;
}
static int by_value(const void* a, const void* b) {
    const int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}

/* Host top-k of one row as ascending positions (identity + -1 pad when len <= TOPK). */
static void host_topk(const float* row, unsigned len, int* out, int* scratch) {
    if (len <= TOPK) {
        for (unsigned s = 0; s < TOPK; s++) out[s] = s < len ? (int)s : -1;
        return;
    }
    for (unsigned s = 0; s < len; s++) scratch[s] = (int)s;
    g_row = row;
    qsort(scratch, len, sizeof(int), by_rank);
    memcpy(out, scratch, TOPK * sizeof(int));
    qsort(out, TOPK, sizeof(int), by_value);
}

static plow_hsa_kernel kLoc, kHist, kMark, kEmit;
static void* dS;
static void* dLen;
static void* dIdx;
static void* dHist;
static void* dCtl;
static void* dBits;
static void* dCand;

static void run_local(void) {
    struct __attribute__((packed)) { void *idx; const void *sc, *len; unsigned lmax, tk; } a =
        {dIdx, dS, dLen, LMAX, TOPK};
    plow_hsa_launch(H, 0, &kLoc, ROWS * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
}

static void run_split(unsigned g) {
    struct __attribute__((packed)) { const void *sc, *len; unsigned lmax, tk; void* hist; unsigned g; } a1 =
        {dS, dLen, LMAX, TOPK, dHist, g};
    struct __attribute__((packed)) { const void *sc, *len; unsigned lmax, tk; const void* hist;
                                     void *ctl, *bits, *cand; unsigned g; } a2 =
        {dS, dLen, LMAX, TOPK, dHist, dCtl, dBits, dCand, g};
    struct __attribute__((packed)) { void* idx; const void* len; unsigned lmax, tk; void *hist, *ctl;
                                     const void *bits, *cand; } a3 =
        {dIdx, dLen, LMAX, TOPK, dHist, dCtl, dBits, dCand};
    plow_hsa_launch(H, 0, &kHist, ROWS * g * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a1, sizeof(a1));
    plow_hsa_launch(H, 0, &kMark, ROWS * g * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a2, sizeof(a2));
    plow_hsa_launch(H, 0, &kEmit, ROWS * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a3, sizeof(a3));
}

static void poison_idx(void) {
    static int* p;
    if (!p) {
        p = malloc((size_t)ROWS * TOPK * 4);
        memset(p, 0x7f, (size_t)ROWS * TOPK * 4);
    }
    plow_hsa_upload(H, 0, dIdx, p, (size_t)ROWS * TOPK * 4);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    H = plow_hsa_init();
    if (!H) { printf("hsa init failed\n"); return 2; }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 2;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) ||
        plow_hsa_get_kernel(H, 0, "index_select_rows_local_k", &kLoc) ||
        plow_hsa_get_kernel(H, 0, "index_select_split_hist_k", &kHist) ||
        plow_hsa_get_kernel(H, 0, "index_select_split_mark_k", &kMark) ||
        plow_hsa_get_kernel(H, 0, "index_select_split_emit_k", &kEmit)) {
        printf("load failed (object built without -DPLOW_DSA_SELECT_SPLIT=1?): %s\n", plow_hsa_last_error());
        return 2;
    }
    printf("dev0: %s CUs=%u rows=%u len_max=%u top_k=%u\n", nm, cus, ROWS, LMAX, TOPK);

    const unsigned lens_fixed[ROWS] = {1000, 2047, 2048, 2049, 4096, 65000, 81920, 65000, 65000, 65000,
                                       30000, 65001, 81919, 12345, 65000, 65000, 70000, 65000, 3000, 81920};
    float* hS = malloc((size_t)ROWS * LMAX * 4);
    int32_t hLen[ROWS];
    int* want = malloc((size_t)ROWS * TOPK * 4);
    int* got = malloc((size_t)ROWS * TOPK * 4);
    int* first = malloc((size_t)ROWS * TOPK * 4);
    int* scratch = malloc((size_t)LMAX * 4);
    dS = plow_hsa_alloc(H, 0, (size_t)ROWS * LMAX * 4);
    dLen = plow_hsa_alloc(H, 0, ROWS * 4);
    dIdx = plow_hsa_alloc(H, 0, (size_t)ROWS * TOPK * 4);
    dHist = plow_hsa_alloc(H, 0, (size_t)ROWS * NB * 4);
    dCtl = plow_hsa_alloc(H, 0, (size_t)ROWS * CTL * 4);
    dBits = plow_hsa_alloc(H, 0, (size_t)ROWS * WORDS * 4);
    dCand = plow_hsa_alloc(H, 0, (size_t)ROWS * LMAX * 8);
    {
        unsigned* z = calloc((size_t)ROWS * NB, 4);
        plow_hsa_upload(H, 0, dHist, z, (size_t)ROWS * NB * 4);
        plow_hsa_upload(H, 0, dCtl, z, (size_t)ROWS * CTL * 4);
        free(z);
    }

    int bad = 0, local_unstable = 0;
    /* case 0: ~200 distinct values (ties at every boundary); 1: continuous with random lengths;
     * 2: negatives only, 64 distinct; 3: one row all-equal, the rest 8 distinct values. */
    for (unsigned cs = 0; cs < 4; cs++) {
        rs = 0x5E1EC7u + cs;
        for (unsigned r = 0; r < ROWS; r++) {
            hLen[r] = (int32_t)(cs == 1 ? (TOPK + 1 + rnd() % (LMAX - TOPK)) : lens_fixed[r]);
            float* row = hS + (size_t)r * LMAX;
            for (unsigned s = 0; s < LMAX; s++) {
                const uint32_t u = rnd();
                switch (cs) {
                case 0: row[s] = (float)(int)(u % 200u) * 0.125f - 12.0f; break;
                case 1: row[s] = (float)(int)(u % 2000001u) * 1e-5f - 10.0f; break;
                case 2: row[s] = -1.0f - (float)(u % 64u) * 0.5f; break;
                default: row[s] = r == 5 ? 3.5f : (float)(u % 8u) - 4.0f; break;
                }
            }
        }
        plow_hsa_upload(H, 0, dS, hS, (size_t)ROWS * LMAX * 4);
        plow_hsa_upload(H, 0, dLen, hLen, ROWS * 4);
        for (unsigned r = 0; r < ROWS; r++)
            host_topk(hS + (size_t)r * LMAX, (unsigned)hLen[r], want + (size_t)r * TOPK, scratch);

        /* local form: set exactness, and whether its list order repeats */
        int bl = 0, lu = 0;
        for (int pass = 0; pass < 2; pass++) {
            poison_idx();
            run_local();
            plow_hsa_wait(H, 0);
            plow_hsa_download(H, 0, got, dIdx, (size_t)ROWS * TOPK * 4);
            if (pass == 0) memcpy(first, got, (size_t)ROWS * TOPK * 4);
            else lu += memcmp(first, got, (size_t)ROWS * TOPK * 4) != 0;
            for (unsigned r = 0; r < ROWS && pass == 0; r++) {
                int* g = got + (size_t)r * TOPK;
                qsort(g, TOPK, sizeof(int), by_value);
                if (memcmp(g, want + (size_t)r * TOPK, TOPK * 4)) {
                    bl++;
                    if (bl <= 3) printf("  local row %u (len %d) != host set\n", r, hLen[r]);
                }
            }
        }
        local_unstable += lu;

        /* gated split: the LIST itself, twice per strip */
        const unsigned gs[] = {1, 4, 8, 12, 15};
        int bs = 0;
        for (unsigned gi = 0; gi < sizeof(gs) / sizeof(gs[0]); gi++) {
            if (ROWS * gs[gi] > cus) continue;
            for (int pass = 0; pass < 2; pass++) {
                poison_idx();
                run_split(gs[gi]);
                plow_hsa_wait(H, 0);
                plow_hsa_download(H, 0, got, dIdx, (size_t)ROWS * TOPK * 4);
                for (unsigned r = 0; r < ROWS; r++) {
                    if (memcmp(got + (size_t)r * TOPK, want + (size_t)r * TOPK, TOPK * 4)) {
                        bs++;
                        if (bs <= 3) printf("  split g=%u pass %d row %u (len %d) list != host\n", gs[gi], pass, r, hLen[r]);
                    }
                }
            }
        }
        printf("case %u: local set %s (list order %s across 2 runs); split list (g=1/4/8/12/15, twice each) %s\n",
               cs, bl ? "MISMATCH" : "exact", lu ? "DIFFERS" : "repeats", bs ? "MISMATCH" : "exact");
        bad += bl + bs;
    }

    /* time at the traced shape: every row at 65000 live keys */
    for (unsigned r = 0; r < ROWS; r++) hLen[r] = 65000;
    plow_hsa_upload(H, 0, dLen, hLen, ROWS * 4);
    const int R = 50;
    for (int w = 0; w < 3; w++) run_local();
    plow_hsa_wait(H, 0);
    double t0 = now();
    for (int i = 0; i < R; i++) run_local();
    plow_hsa_wait(H, 0);
    printf("time at len 65000 x %u rows: local %.1f us", ROWS, (now() - t0) / R * 1e6);
    const unsigned gt[] = {4, 8, 12, 15};
    for (unsigned gi = 0; gi < sizeof(gt) / sizeof(gt[0]); gi++) {
        if (ROWS * gt[gi] > cus) continue;
        for (int w = 0; w < 3; w++) run_split(gt[gi]);
        plow_hsa_wait(H, 0);
        t0 = now();
        for (int i = 0; i < R; i++) run_split(gt[gi]);
        plow_hsa_wait(H, 0);
        printf("; split g=%u %.1f us (3 launches)", gt[gi], (now() - t0) / R * 1e6);
    }
    printf("\nlocal list order differed between runs in %d of 4 cases\n", local_unstable);
    printf("%s\n", bad ? "FAIL" : "PASS: every split list exact and repeatable");
    plow_hsa_shutdown(H);
    return bad ? 1 : 0;
}
