/* Host runner for mfma_rate_gfx942.hip — TF/s of a pure MFMA issue stream. */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &lds);
    FILE* f = fopen("mfma_rate.elf", "rb");
    if (!f) { perror("mfma_rate.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1; }
    void* dO = plow_hsa_alloc(h, 0, 4096);

    /* name, chains, MACs per issued wrapper call, wrapper calls per inner step */
    struct { const char* sym; int ch; double macs; } K[] = {
        {"mfma_ch1", 1, 32.0*32*16}, {"mfma_ch2", 2, 32.0*32*16},
        {"mfma_ch4", 4, 32.0*32*16}, {"mfma_ch6", 6, 32.0*32*16},
        {"mfma_ch8", 8, 32.0*32*16},
        {"mfma16_ch4", 4, 16.0*16*32}, {"mfma16_ch8", 8, 16.0*16*32},
        {"mfma16_ch16", 16, 16.0*16*32}, {"mfma16_ch24", 24, 16.0*16*32},
        {"mfma16_ch32", 32, 16.0*16*32},
        {"mfma8_ch2", 2, 32.0*32*64}, {"mfma8_ch4", 4, 32.0*32*64},
        {"mfma8_ch6", 6, 32.0*32*64},
    };
    /* LDS mix: name, ds_read_b128 per step, MFMA-wrapper calls per step */
    struct { const char* sym; int rd, ch; } X[] = {
        {"rd_r5", 5, 0}, {"rd_r8", 8, 0},
        {"mix_r5_m6", 5, 6}, {"mix_r6_m8", 6, 8}, {"mix_r8_m16", 8, 16},
        {"mix_r5_m12", 5, 12}, {"mix_r5_m24", 5, 24},
    };
    const unsigned iters = 4000;
    printf("dev %s  CUs=%u   pure-MFMA issue rate, %d waves/WG, 1 WG/CU\n", nm, cus, PLOW_WG_WAVES);
    for (unsigned i = 0; i < sizeof(K) / sizeof(K[0]); i++) {
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(h, 0, K[i].sym, &k) != 0) { printf("  %-14s missing\n", K[i].sym); continue; }
        struct __attribute__((packed)) { void* o; unsigned it; } args = {dO, iters};
        plow_hsa_launch(h, 0, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            const double t0 = now();
            plow_hsa_launch(h, 0, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
            plow_hsa_wait(h, 0);
            const double d = now() - t0;
            if (d < best) best = d;
        }
        /* one wave issues iters*8*ch wrapper calls; cus*PLOW_WG_WAVES waves in flight */
        const double flops = 2.0 * K[i].macs * iters * 8.0 * K[i].ch * cus * PLOW_WG_WAVES;
        printf("  %-14s chains=%2d  %8.3f ms  %8.1f TF/s  (VGPR %u)\n", K[i].sym, K[i].ch,
               best * 1e3, flops / best / 1e12, k.group_segment_size);
    }
    printf("\nLDS read / MFMA trade (per step: RD ds_read_b128 + CH 32x32x16 wrappers):\n");
    for (unsigned i = 0; i < sizeof(X) / sizeof(X[0]); i++) {
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(h, 0, X[i].sym, &k) != 0) { printf("  %-14s missing\n", X[i].sym); continue; }
        struct __attribute__((packed)) { void* o; unsigned it; } args = {dO, iters};
        plow_hsa_launch(h, 0, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            const double t0 = now();
            plow_hsa_launch(h, 0, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
            plow_hsa_wait(h, 0);
            const double d = now() - t0;
            if (d < best) best = d;
        }
        const double steps = (double)iters * 8.0 * cus * PLOW_WG_WAVES;
        const double bytes = steps * X[i].rd * 1024.0;         /* 64 lanes * 16 B */
        const double flops = steps * X[i].ch * 2.0 * 32 * 32 * 16;
        printf("  %-14s rd=%d ch=%2d  %8.3f ms  LDS %8.1f GB/s  %8.1f TF/s\n", X[i].sym, X[i].rd,
               X[i].ch, best * 1e3, bytes / best / 1e9, flops / best / 1e12);
    }
    return 0;
}
