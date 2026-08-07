/* gemm942_shapes_test.c — plow's prefill GEMM at the EXACT shapes the hipBLASLt/Tensile
 * baseline was measured on (/workspace/bench_results/mi300x-library-baseline-20260803).
 *
 * Sweeps every gemm_c* / gemm_fp8_c* tile the loaded object exports, warms the clock, then
 * re-times the winner as a warm median-of-5. Every row is checked against an f64 oracle.
 *
 * Buffers are allocated ONCE at the maximum size and reused across shapes: filling 100M+
 * elements per shape dominated the run otherwise, and the content is arbitrary.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static unsigned long long rs = 88172645463325252ull;
static unsigned xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 11); }

/* OCP e4m3 decode, for the oracle. */
static double e4m3_decode(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
    double v;
    if (e == 0) v = ldexp((double)m, -9);
    else if (e == 15 && m == 7) v = NAN;
    else v = ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}

static int fails = 0;

static const char* BF_TILES[] = {"gemm_c0", "gemm_c1", "gemm_c2", "gemm_c3", "gemm_c4",
                                 "gemm_c5", "gemm_c6", "gemm_c7", "gemm_c8", "gemm_c9"};
static const char* BF_NAMES[] = {"256x256", "256x128", "128x256", "128x128", "64x128",
                                 "192x256", "320x128", "384x128", "128x384", "192x128"};
#define NBF (int)(sizeof(BF_TILES) / sizeof(BF_TILES[0]))
static const char* F8_TILES[] = {"gemm_fp8_c0", "gemm_fp8_c3", "gemm_fp8_c4", "gemm_fp8_c5"};
static const char* F8_NAMES[] = {"256x256", "128x128", "64x128", "192x256"};
#define NF8 (int)(sizeof(F8_TILES) / sizeof(F8_TILES[0]))

static int dcmp(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

/* time one kernel: `warm` launches to settle the clock, then MED reps of ITER launches. */
static double timeit(plow_hsa* h, plow_hsa_kernel* kc, unsigned grid, void* args, unsigned asz,
                     int iter, int med) {
    double s[9];
    plow_hsa_launch(h, 0, kc, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz);
    plow_hsa_wait(h, 0);
    for (int r = 0; r < med; r++) {
        const double t0 = now();
        for (int i = 0; i < iter; i++)
            plow_hsa_launch(h, 0, kc, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz);
        plow_hsa_wait(h, 0);
        s[r] = (now() - t0) / iter;
    }
    qsort(s, med, sizeof(double), dcmp);
    return s[med / 2];
}

typedef struct { const char* name; unsigned N, K; } shape_t;

int main(int argc, char** argv) {
    const int do_bf = !(argc > 1 && !strcmp(argv[1], "fp8"));
    const int do_f8 = !(argc > 1 && !strcmp(argv[1], "bf16"));
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B  waves/WG=%d\n", nm, cus, lds, PLOW_WG_WAVES);

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    static const shape_t SH[] = {
        {"g31_q_proj",        8192,  5376},
        {"g31_o_proj",        5376,  8192},
        {"g31_gate_up",      21504,  5376},
        {"g31_down",          5376, 21504},
        {"g12_q_proj",        4096,  3840},
        {"g12_o_proj",        3840,  4096},
        {"g12_gate_up",      15360,  3840},
        {"g12_down",          3840, 15360},
        {"glm_q_b",          16384,  2048},
        {"glm_o",             6144, 16384},
        {"glm_dense_gate_up",12288,  6144},
        {"glm_dense_down",    6144, 12288},
        {"glm_moe_gate_up",   2048,  6144},
        {"glm_moe_down",      6144,  2048},
    };
    const int NS = (int)(sizeof(SH) / sizeof(SH[0]));
    static const unsigned MS[] = {4096, 512, 128};

    /* One allocation, sized for the worst shape, reused everywhere. */
    size_t maxK = 0, maxN = 0, maxNK = 0, maxMN = 0;
    for (int i = 0; i < NS; i++) {
        if (SH[i].K > maxK) maxK = SH[i].K;
        if (SH[i].N > maxN) maxN = SH[i].N;
        if ((size_t)SH[i].N * SH[i].K > maxNK) maxNK = (size_t)SH[i].N * SH[i].K;
        if ((size_t)SH[i].N * MS[0] > maxMN) maxMN = (size_t)SH[i].N * MS[0];
    }
    const size_t maxMK = (size_t)MS[0] * maxK;

    bf16* hA = plow_hsa_alloc_host(h, maxMK * 2);
    bf16* hB = plow_hsa_alloc_host(h, maxNK * 2);
    bf16* hC = plow_hsa_alloc_host(h, maxMN * 2);
    unsigned char* qA = plow_hsa_alloc_host(h, maxMK);
    unsigned char* qB = plow_hsa_alloc_host(h, maxNK);
    float* hAs = plow_hsa_alloc_host(h, MS[0] * 4);
    float* hWs = plow_hsa_alloc_host(h, maxN * 4);
    if (!hA || !hB || !hC || !qA || !qB) { fprintf(stderr, "host alloc failed\n"); return 1; }
    printf("filling %.1f MB of operands...\n", (maxMK + maxNK) * 3.0 / 1e6);
    for (size_t i = 0; i < maxMK; i++) {
        hA[i] = f2bf(((float)(xr() % 17) - 8.0f) / 16.0f);
        qA[i] = (unsigned char)(((xr() % 2) << 7) | ((xr() % 8) << 3) | (xr() % 8));
    }
    for (size_t i = 0; i < maxNK; i++) {
        hB[i] = f2bf(((float)(xr() % 17) - 8.0f) / 16.0f);
        qB[i] = (unsigned char)(((xr() % 2) << 7) | ((xr() % 8) << 3) | (xr() % 8));
    }
    for (unsigned i = 0; i < MS[0]; i++) hAs[i] = 0.005f + 0.02f * (xr() % 8) / 8.0f;
    for (unsigned i = 0; i < maxN; i++) hWs[i] = 0.005f + 0.02f * (xr() % 8) / 8.0f;

    void* dA = plow_hsa_alloc(h, 0, maxMK * 2);
    void* dB = plow_hsa_alloc(h, 0, maxNK * 2);
    void* dC = plow_hsa_alloc(h, 0, maxMN * 2);
    void* dqA = plow_hsa_alloc(h, 0, maxMK);
    void* dqB = plow_hsa_alloc(h, 0, maxNK);
    void* dAs = plow_hsa_alloc(h, 0, MS[0] * 4);
    void* dWs = plow_hsa_alloc(h, 0, maxN * 4);
    plow_hsa_copy_h2d(h, 0, dA, hA, maxMK * 2);
    plow_hsa_copy_h2d(h, 0, dB, hB, maxNK * 2);
    plow_hsa_copy_h2d(h, 0, dqA, qA, maxMK);
    plow_hsa_copy_h2d(h, 0, dqB, qB, maxNK);
    plow_hsa_copy_h2d(h, 0, dAs, hAs, MS[0] * 4);
    plow_hsa_copy_h2d(h, 0, dWs, hWs, maxN * 4);

    const int SAMP = 128;
    printf("\n%-20s %5s %6s %6s | %-9s %8s %8s | %-9s %8s %8s\n", "shape", "M", "N", "K",
           "bf16 tile", "us", "TF/s", "fp8 tile", "us", "TF/s");
    const char* only = getenv("PLOW_ONLY");
    const char* onlym = getenv("PLOW_M");
    /* Workgroups per CU, and a HYPOTHESIS THAT DID NOT SURVIVE. The kernel walks
     * `for (lin = slice; lin < n_tiles; lin += nblk)`, so at 1 WG/CU the tile grid is assigned
     * STATICALLY and a shape whose tile count is not a multiple of 304 pays for a whole extra
     * round -- 192x256 on g31_gate_up is 1848 tiles = 6.08 rounds, so it should pay 7, a 15%
     * tail. Oversubscribing lets the hardware dispatcher hand that tail out dynamically instead.
     *
     * MEASURED (gfx942, M=4096, bf16 TF/s, GMUL 1/2/4/8): q_proj 459/468/466/461,
     * o_proj 453/452/458/451, gate_up 481/480/483/480, down 404/420/422/402. Flat inside the
     * run-to-run spread of a shared box. Kept as a knob because the reasoning is sound and the
     * effect may appear on a shape whose tile count is worse; it is not a lever today. */
    const int gmul = getenv("PLOW_GMUL") ? atoi(getenv("PLOW_GMUL")) : 1;
    const unsigned grid = cus * (unsigned)gmul * PLOW_WG_THREADS;
    for (int mi = 0; mi < 3; mi++) {
        const unsigned M = MS[mi];
        if (onlym && (unsigned)atoi(onlym) != M) continue;
        for (int si = 0; si < NS; si++) {
            const unsigned N = SH[si].N, K = SH[si].K;
            if (only && !strstr(SH[si].name, only)) continue;
            const double flops = 2.0 * M * N * K;
            printf("%-20s %5u %6u %6u |", SH[si].name, M, N, K);
            fflush(stdout);

            char bfline[64] = "-", f8line[64] = "-";
            if (do_bf) {
                struct __attribute__((packed)) {
                    void* c; const void* a; const void* b; unsigned m, n, kk;
                } args = {dC, dA, dB, M, N, K};
                double best = 1e30; int bi = -1;
                const char* onlyt = getenv("PLOW_TILE");
                for (int c = 0; c < NBF; c++) {
                    plow_hsa_kernel kc;
                    if (onlyt && strcmp(onlyt, BF_TILES[c])) continue;
                    if (plow_hsa_get_kernel(h, 0, BF_TILES[c], &kc) != 0) continue;
                    const double d = timeit(h, &kc, grid, &args, sizeof(args), 3, 3);
                    if (getenv("PLOW_VERBOSE"))
                        printf("\n      [%-8s %7.1f TF/s]", BF_NAMES[c], flops / d / 1e12);
                    if (d < best) { best = d; bi = c; }
                }
                if (bi >= 0) {
                    plow_hsa_kernel kc; plow_hsa_get_kernel(h, 0, BF_TILES[bi], &kc);
                    best = timeit(h, &kc, grid, &args, sizeof(args), 5, 5);
                    plow_hsa_launch(h, 0, &kc, grid, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &args, sizeof(args));
                    plow_hsa_wait(h, 0);
                    plow_hsa_copy_d2h(h, 0, hC, dC, (size_t)M * N * 2);
                    double worst = 0.0;
                    for (int s = 0; s < SAMP; s++) {
                        const unsigned m = xr() % M, nn = xr() % N;
                        double want = 0.0;
                        for (unsigned kk = 0; kk < K; kk++)
                            want += (double)bf2f(hA[(size_t)m * K + kk]) * bf2f(hB[(size_t)nn * K + kk]);
                        const double got = bf2f(hC[(size_t)m * N + nn]);
                        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
                        if (!(rel <= worst)) worst = rel;
                    }
                    const int ok = worst < 3e-2;
                    if (!ok) fails++;
                    snprintf(bfline, sizeof(bfline), "%-9s %8.1f %8.1f %s", BF_NAMES[bi],
                             best * 1e6, flops / best / 1e12, ok ? "" : "**FAIL**");
                }
            }
            if (do_f8) {
                struct __attribute__((packed)) {
                    void* c; const void* a; const void* b; const void* as; const void* ws;
                    unsigned m, n, kk;
                } args = {dC, dqA, dqB, dAs, dWs, M, N, K};
                double best = 1e30; int bi = -1;
                for (int c = 0; c < NF8; c++) {
                    plow_hsa_kernel kc;
                    if (plow_hsa_get_kernel(h, 0, F8_TILES[c], &kc) != 0) continue;
                    const double d = timeit(h, &kc, grid, &args, sizeof(args), 3, 3);
                    if (getenv("PLOW_VERBOSE"))
                        printf("\n      [fp8 %-8s %7.1f TF/s]", F8_NAMES[c], flops / d / 1e12);
                    if (d < best) { best = d; bi = c; }
                }
                if (bi >= 0) {
                    plow_hsa_kernel kc; plow_hsa_get_kernel(h, 0, F8_TILES[bi], &kc);
                    best = timeit(h, &kc, grid, &args, sizeof(args), 5, 5);
                    plow_hsa_launch(h, 0, &kc, grid, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &args, sizeof(args));
                    plow_hsa_wait(h, 0);
                    plow_hsa_copy_d2h(h, 0, hC, dC, (size_t)M * N * 2);
                    double worst = 0.0;
                    for (int s = 0; s < SAMP; s++) {
                        const unsigned m = xr() % M, nn = xr() % N;
                        double want = 0.0;
                        for (unsigned kk = 0; kk < K; kk++)
                            want += e4m3_decode(qA[(size_t)m * K + kk]) * e4m3_decode(qB[(size_t)nn * K + kk]);
                        want *= (double)hAs[m] * (double)hWs[nn];
                        const double got = bf2f(hC[(size_t)m * N + nn]);
                        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
                        if (!(rel <= worst)) worst = rel;
                    }
                    const int ok = worst < 3e-2;
                    if (!ok) fails++;
                    snprintf(f8line, sizeof(f8line), "%-9s %8.1f %8.1f %s", F8_NAMES[bi],
                             best * 1e6, flops / best / 1e12, ok ? "" : "**FAIL**");
                }
            }
            printf(" %s | %s\n", bfline, f8line);
            fflush(stdout);
        }
    }
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails != 0;
}
