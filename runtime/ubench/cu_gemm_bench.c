/* cu_gemm_bench.c — SINGLE-CU inner-loop efficiency for the GEMM/GEMV ops.
 *
 * The persistent interpreter hands each CU its own tile, so the number that
 * matters is not whole-GPU throughput (which folds in scheduling, tail effects
 * and memory contention) but how close ONE workgroup on ONE CU gets to that CU's
 * roofline. This launches exactly one workgroup.
 *
 * Per-CU rooflines on MI350X (gfx950) @ 2.2 GHz:
 *   MFMA bf16 : 4 matrix cores x 512 MAC/cycle x 2 FLOP = 4096 FLOP/cycle
 *               => 9.01 TFLOP/s per CU
 *   HBM       : ~8 TB/s / 256 CU => ~31 GB/s per CU (only meaningful for GEMV,
 *               and only when all CUs are streaming; a single CU can pull more)
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
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define CU_CLOCK_GHZ 2.2
#define CU_PEAK_TFLOPS (4.0 * 512.0 * 2.0 * CU_CLOCK_GHZ / 1000.0) /* 9.01 */

static plow_hsa* H;

/* One workgroup, one output tile. M and N are the tile the kernel will pick up
 * (it grid-strides, but with 1 block and 1 tile there is exactly one pass). */
static void bench_gemm(plow_hsa_kernel* k, const char* label, unsigned M, unsigned N,
                       unsigned K) {
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(H, nA * 2);
    bf16* hB = plow_hsa_alloc_host(H, nB * 2);
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

    /* ONE workgroup => one CU. */
    plow_hsa_launch(H, 0, k, 256, 1, 1, 256, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(H, 0);

    const int reps = 20;
    const double t0 = now();
    for (int r = 0; r < reps; r++)
        plow_hsa_launch(H, 0, k, 256, 1, 1, 256, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(H, 0);
    const double dt = (now() - t0) / reps;

    const double tf = 2.0 * M * N * K / dt / 1e12;
    /* Cycles per MFMA-equivalent: how many cycles the CU actually spent per
     * 32x32x16 matrix op it had to issue. Ideal is ~32 (the instruction's own
     * latency at full rate); anything far above that is stall, not math. */
    const double mfmas = (double)(M / 32) * (N / 32) * (K / 16);
    const double cycles = dt * CU_CLOCK_GHZ * 1e9;
    printf("  %-24s %4ux%4ux%5u  %8.1f us  %6.3f TF/s  %5.1f%% of CU peak   %6.1f cyc/mfma\n",
           label, M, N, K, dt * 1e6, tf, 100.0 * tf / CU_PEAK_TFLOPS, cycles / mfmas);

    plow_hsa_free(H, dA);
    plow_hsa_free(H, dB);
    plow_hsa_free(H, dC);
}

int main(void) {
    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    printf("%s  %u CUs  %u B LDS   per-CU MFMA peak = %.2f TFLOP/s\n\n", nm, cus, lds,
           CU_PEAK_TFLOPS);

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
    if (plow_hsa_get_kernel(H, 0, "gemma_gemm_bf16", &k) != 0) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error());
        return 1;
    }
    printf("gemm: vgpr-spill(scratch)=%u B  LDS=%u B\n\n", k.private_segment_size,
           k.group_segment_size);
    srand(5);

    printf("SINGLE CU (1 workgroup, 256 threads, 4 waves):\n");
    bench_gemm(&k, "one 128x128 tile", 128, 128, 5376);   /* Gemma K */
    bench_gemm(&k, "one 128x128 tile, K=1k", 128, 128, 1024);
    bench_gemm(&k, "2 tiles (128x256)", 128, 256, 5376);
    bench_gemm(&k, "4 tiles (256x256)", 256, 256, 5376);

    plow_hsa_shutdown(H);
    return 0;
}
