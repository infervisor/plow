/* gemm_bench_8k.c — WHOLE-GPU standalone bf16 GEMM sweep on the Qwen3-4B prefill shapes.
 *
 * Launches a named gemm_cN kernel from test_kernels.elf across all CUs (one workgroup per
 * CU, grid-strided over tiles — the same slice/nblk loop the interpreter drives), times it,
 * and spot-checks correctness against a CPU dot product. The kernel SYMBOL is argv[1] so a
 * tile sweep only rebuilds test_kernels.hip, never this file.
 *
 *   usage: gemm_bench_8k <kernel_symbol> [M]
 *
 * Peak: 1660 TF/s sustained bf16 MFMA on this machine (256 CU @ ~1.58 GHz dense).
 */
#include "../amd/hsa_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static bf16 f2bf(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define PEAK_TFLOPS 1660.0

static plow_hsa* H;
static unsigned NCU;
static unsigned THREADS;

struct Shape { const char* name; unsigned M, N, K; };

static void bench(plow_hsa_kernel* k, const char* label, unsigned M, unsigned N, unsigned K) {
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(H, nA * 2);
    bf16* hB = plow_hsa_alloc_host(H, nB * 2);
    srand(5);
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nB; i++) hB[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    void* dA = plow_hsa_alloc(H, 0, nA * 2);
    void* dB = plow_hsa_alloc(H, 0, nB * 2);
    void* dC = plow_hsa_alloc(H, 0, nC * 2);
    plow_hsa_copy_h2d(H, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(H, 0, dB, hB, nB * 2);

    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; unsigned m, n, kk;
    } args = {dC, dA, dB, M, N, K};

    /* SUSTAINED warm-up. On this shared-package MI350X a single launch is NOT enough: the
     * governor ramps sclk over tens of ms, so an under-warmed kernel reads slow — and a sweep
     * that warms up over its first few kernels then ranks LATER kernels faster purely because
     * the clock rose. Burn ~50 launches (~25 ms) so the clock is saturated before timing. */
    for (int w = 0; w < 50; w++)
        plow_hsa_launch(H, 0, k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(H, 0);

    const int reps = 20;
    const double t0 = now();
    for (int r = 0; r < reps; r++)
        plow_hsa_launch(H, 0, k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(H, 0);
    const double dt = (now() - t0) / reps;
    const double tf = 2.0 * M * N * K / dt / 1e12;

    /* spot-check */
    bf16* hC = plow_hsa_alloc_host(H, nC * 2);
    plow_hsa_copy_d2h(H, 0, hC, dC, nC * 2);
    int bad = 0;
    for (int t = 0; t < 12; t++) {
        unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double acc = 0;
        for (unsigned kk = 0; kk < K; kk++)
            acc += (double)bf2f(hA[(size_t)m * K + kk]) * bf2f(hB[(size_t)n * K + kk]);
        double g = bf2f(hC[(size_t)m * N + n]);
        double rel = fabs(g - acc) / (fabs(acc) + 1e-3);
        if (rel > 0.03) bad++;
    }
    printf("  %-14s %5ux%6ux%5u  %8.3f ms  %7.1f TF/s  %5.1f%% peak  %s\n", label, M, N, K,
           dt * 1e3, tf, 100.0 * tf / PEAK_TFLOPS, bad ? "MISMATCH!" : "ok");

    plow_hsa_free(H, dA);
    plow_hsa_free(H, dB);
    plow_hsa_free(H, dC);
}

int main(int argc, char** argv) {
    const char* sym = argc > 1 ? argv[1] : "gemm_c0";
    unsigned Mov = argc > 2 ? (unsigned)atoi(argv[2]) : 0;
    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    NCU = cus;

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error());
        return 1;
    }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(H, 0, sym, &k) != 0) {
        fprintf(stderr, "sym %s: %s\n", sym, plow_hsa_last_error());
        return 1;
    }
    THREADS = 512; /* PLOW_THREADS: the 8-wave GEMM grid */
    printf("%s  %u CUs   kernel=%s  thr=%u  vspill=%uB  LDS=%uB\n\n", nm, NCU, sym, THREADS,
           k.private_segment_size, k.group_segment_size);

    unsigned M = Mov ? Mov : 8192;
    int gemma = argc > 3 && strcmp(argv[3], "gemma") == 0;
    struct Shape qwen[] = {
        {"q_proj",   M, 4096, 2560},
        {"kv_proj",  M, 2048, 2560},
        {"o_proj",   M, 2560, 4096},
        {"gate/up",  M, 9728, 2560},
        {"down",     M, 2560, 9728},
    };
    struct Shape gem[] = {
        {"q_proj",   M, 8192, 5376},
        {"kv_proj",  M, 4096, 5376},
        {"o_proj",   M, 5376, 8192},
        {"gate/up",  M, 21504, 5376},
        {"down",     M, 5376, 21504},
    };
    struct Shape* sh = gemma ? gem : qwen;
    printf("%s prefill GEMM (M=%u), peak %.0f TF/s:\n", gemma ? "Gemma-31B" : "Qwen3-4B", M,
           PEAK_TFLOPS);
    for (int s = 0; s < 5; s++) bench(&k, sh[s].name, sh[s].M, sh[s].N, sh[s].K);

    plow_hsa_shutdown(H);
    return 0;
}
