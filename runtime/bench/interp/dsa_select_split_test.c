/* dsa_select_split_test.c — the split-row batched decode selection (op 59 i[4]=2,
 * PLOW_DSA_SELECT_SPLIT) against the local form it replaces (i[4]=1, one workgroup per row) and a
 * host top-k. Both run as test_kernels.hip wrappers around the device functions interp.hip calls.
 *
 * Gate: for every row, the split set == the local set == the host set (score desc, lowest index
 * first; a row of len <= top_k is the identity padded with -1). Scores are quantized so the k-th
 * score is tied across many positions (the index passes decide), plus an all-equal row, negative
 * rows and lengths at/around top_k. Every split configuration runs twice on the same strips, so a
 * control strip the first run left dirty fails the second. Also prints local vs split time.
 *
 * Build (gfx942):
 *   hipcc --offload-arch=gfx942 -O3 -w --genco -DPLOW_DSA_SELECT_SPLIT=1 \
 *       -Iruntime/amd -Iruntime/common runtime/amd/test_kernels.hip -o tk.co
 *   clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
 *       --input=tk.co --output=test_kernels.elf
 *   gcc -O2 -std=gnu11 -o dsa_select_split_test runtime/bench/interp/dsa_select_split_test.c \
 *       runtime/amd/hsa_backend.c -I$ROCM_PATH/include -L$ROCM_PATH/lib -lhsa-runtime64
 * Run: ./dsa_select_split_test test_kernels.elf   (exit 0 = every row exact)
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
#define NPASS_NB (7u * 256u)
#define CTL 16u

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

/* Host top-k of one row as a sorted set of TOPK ints (-1 pads). */
static void host_topk(const float* row, unsigned len, int* out, int* scratch) {
    if (len <= TOPK) {
        for (unsigned s = 0; s < TOPK; s++) out[s] = s < len ? (int)s : -1;
    } else {
        for (unsigned s = 0; s < len; s++) scratch[s] = (int)s;
        g_row = row;
        qsort(scratch, len, sizeof(int), by_rank);
        memcpy(out, scratch, TOPK * sizeof(int));
    }
    qsort(out, TOPK, sizeof(int), by_value);
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
    plow_hsa_kernel kLoc, kSplit;
    if (plow_hsa_load_code_object(H, 0, co, n) ||
        plow_hsa_get_kernel(H, 0, "index_select_rows_local_k", &kLoc) ||
        plow_hsa_get_kernel(H, 0, "index_select_rows_split_k", &kSplit)) {
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
    int* scratch = malloc((size_t)LMAX * 4);
    void* dS = plow_hsa_alloc(H, 0, (size_t)ROWS * LMAX * 4);
    void* dLen = plow_hsa_alloc(H, 0, ROWS * 4);
    void* dIdx = plow_hsa_alloc(H, 0, (size_t)ROWS * TOPK * 4);
    void* dHist = plow_hsa_alloc(H, 0, (size_t)ROWS * NPASS_NB * 4);
    void* dCtl = plow_hsa_alloc(H, 0, (size_t)ROWS * CTL * 4);
    {
        unsigned* z = calloc((size_t)ROWS * NPASS_NB, 4);
        plow_hsa_upload(H, 0, dHist, z, (size_t)ROWS * NPASS_NB * 4);
        plow_hsa_upload(H, 0, dCtl, z, (size_t)ROWS * CTL * 4);
        free(z);
    }
    struct __attribute__((packed)) { void *idx; const void *sc, *len; unsigned lmax, tk; } aLoc =
        {dIdx, dS, dLen, LMAX, TOPK};
    struct __attribute__((packed)) { void *idx; const void *sc, *len; unsigned lmax, tk; void *hist, *ctl; unsigned g; } aSp =
        {dIdx, dS, dLen, LMAX, TOPK, dHist, dCtl, 0};

    int bad = 0;
    /* case 0: ~200 distinct values (ties at every boundary); 1: continuous; 2: negatives only,
     * 64 distinct; 3: one row all-equal, the rest 8 distinct values. */
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

        /* local form */
        plow_hsa_launch(H, 0, &kLoc, ROWS * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aLoc, sizeof(aLoc));
        plow_hsa_wait(H, 0);
        plow_hsa_download(H, 0, got, dIdx, (size_t)ROWS * TOPK * 4);
        int bl = 0;
        for (unsigned r = 0; r < ROWS; r++) {
            int* g = got + (size_t)r * TOPK;
            qsort(g, TOPK, sizeof(int), by_value);
            if (memcmp(g, want + (size_t)r * TOPK, TOPK * 4)) { bl++; if (bl <= 3) printf("  local row %u (len %d) != host\n", r, hLen[r]); }
        }
        const unsigned gs[] = {1, 4, 8, 15};
        int bs = 0;
        for (unsigned gi = 0; gi < sizeof(gs) / sizeof(gs[0]); gi++) {
            aSp.g = gs[gi];
            if (ROWS * aSp.g > cus) continue;
            for (int pass = 0; pass < 2; pass++) {
                unsigned* poison = malloc((size_t)ROWS * TOPK * 4);
                memset(poison, 0x7f, (size_t)ROWS * TOPK * 4);
                plow_hsa_upload(H, 0, dIdx, poison, (size_t)ROWS * TOPK * 4);
                free(poison);
                plow_hsa_launch(H, 0, &kSplit, ROWS * aSp.g * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aSp, sizeof(aSp));
                plow_hsa_wait(H, 0);
                plow_hsa_download(H, 0, got, dIdx, (size_t)ROWS * TOPK * 4);
                for (unsigned r = 0; r < ROWS; r++) {
                    int* g = got + (size_t)r * TOPK;
                    qsort(g, TOPK, sizeof(int), by_value);
                    if (memcmp(g, want + (size_t)r * TOPK, TOPK * 4)) {
                        bs++;
                        if (bs <= 3) printf("  split g=%u pass %d row %u (len %d) != host\n", aSp.g, pass, r, hLen[r]);
                    }
                }
            }
        }
        printf("case %u: local %s, split (g=1/4/8/15, twice each) %s\n", cs, bl ? "MISMATCH" : "exact",
               bs ? "MISMATCH" : "exact");
        bad += bl + bs;
    }

    /* timing on case-1-style rows at 65000 live keys, the traced shape */
    for (unsigned r = 0; r < ROWS; r++) hLen[r] = 65000;
    plow_hsa_upload(H, 0, dLen, hLen, ROWS * 4);
    const int R = 50;
    for (int w = 0; w < 3; w++)
        plow_hsa_launch(H, 0, &kLoc, ROWS * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aLoc, sizeof(aLoc));
    plow_hsa_wait(H, 0);
    double t0 = now();
    for (int i = 0; i < R; i++)
        plow_hsa_launch(H, 0, &kLoc, ROWS * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aLoc, sizeof(aLoc));
    plow_hsa_wait(H, 0);
    printf("time at len 65000 x %u rows: local %.1f us", ROWS, (now() - t0) / R * 1e6);
    const unsigned gt[] = {4, 8, 12, 15};
    for (unsigned gi = 0; gi < sizeof(gt) / sizeof(gt[0]); gi++) {
        aSp.g = gt[gi];
        if (ROWS * aSp.g > cus) continue;
        for (int w = 0; w < 3; w++)
            plow_hsa_launch(H, 0, &kSplit, ROWS * aSp.g * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aSp, sizeof(aSp));
        plow_hsa_wait(H, 0);
        t0 = now();
        for (int i = 0; i < R; i++)
            plow_hsa_launch(H, 0, &kSplit, ROWS * aSp.g * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aSp, sizeof(aSp));
        plow_hsa_wait(H, 0);
        printf("; split g=%u %.1f us", aSp.g, (now() - t0) / R * 1e6);
    }
    printf("\n%s\n", bad ? "FAIL" : "PASS: every row exact");
    plow_hsa_shutdown(H);
    return bad ? 1 : 0;
}
