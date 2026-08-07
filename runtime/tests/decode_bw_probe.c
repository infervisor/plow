/* decode_bw_probe.c — host driver for decode_bw_probe.hip. Reads a `--gb` buffer with
 * each of the decode kernel's addressing shapes and reports GB/s. */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}
static int cmpd(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    int dev = 0, iters = 10, reps = 7;
    double gb = 8.0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--dev=", 6)) dev = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--gb=", 5)) gb = atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
    }
    plow_hsa* H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, dev, nm, &cus, &lds);
    fprintf(stderr, "dev%d %s CUs=%u\n", dev, nm, cus);

    FILE* f = fopen("bw_probe.elf", "rb");
    if (!f) { perror("bw_probe.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, dev, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    const size_t bytes = (size_t)(gb * 1e9) & ~(size_t)4095;
    void* dS = plow_hsa_alloc(H, dev, bytes);
    void* dO = plow_hsa_alloc(H, dev, 4096);
    if (!dS) { fprintf(stderr, "alloc failed\n"); return 1; }

    struct { const char* k; int rowd; } K[] = {
        {"bw_probe_stream", 0}, {"bw_probe_rows_512", 512}, {"bw_probe_coop_512", 512},
        {"bw_probe_rows_256", 256}, {"bw_probe_coop_256", 256},
        {"bw_probe_rows8_512", 512}, {"bw_probe_rows8_256", 256},
    };
    for (unsigned i = 0; i < sizeof(K) / sizeof(K[0]); i++) {
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(H, dev, K[i].k, &k)) { fprintf(stderr, "sym %s\n", K[i].k); continue; }
        struct __attribute__((packed)) { void* o; const void* s; unsigned long long n; } a8 =
            {dO, dS, bytes / 16};
        struct __attribute__((packed)) { void* o; const void* s; unsigned n; } a4 =
            {dO, dS, (unsigned)(bytes / (K[i].rowd ? K[i].rowd * 2 : 1024))};
        const void* args = K[i].rowd ? (const void*)&a4 : (const void*)&a8;
        size_t asz = K[i].rowd ? sizeof(a4) : sizeof(a8);
        plow_hsa_launch(H, dev, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz);
        plow_hsa_wait(H, dev);
        double ts[64];
        for (int r = 0; r < reps; r++) {
            double t0 = now_us();
            for (int it = 0; it < iters; it++)
                plow_hsa_launch(H, dev, &k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz);
            plow_hsa_wait(H, dev);
            ts[r] = (now_us() - t0) / iters;
        }
        qsort(ts, reps, sizeof(double), cmpd);
        printf("%-20s %8.1f us  %7.1f GB/s\n", K[i].k, ts[reps / 2], bytes / ts[reps / 2] / 1e3);
        fflush(stdout);
    }
    plow_hsa_shutdown(H);
    return 0;
}
