/* HSA driver for decode_bw_probe.hip; decimal GB/s over a >=4 GiB device buffer.
 * Build from the repo root:
 *
 * nix develop --command bash -c '
 *   set -eu
 *   : "${ROCM_PATH:?}"
 *   BW_OUT=/tmp/r9700-ubench
 *   mkdir -p "$BW_OUT"
 *   "$ROCM_PATH/bin/hipcc" --offload-arch=gfx1201 -mwavefrontsize64 -mcumode -O3 \
 *     --genco -Iruntime/amd -Iruntime/common runtime/tests/decode_bw_probe.hip \
 *     -o "$BW_OUT/bw_probe.co"
 *   "$ROCM_PATH/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
 *     --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
 *     --input="$BW_OUT/bw_probe.co" --output="$BW_OUT/bw_probe.elf"
 *   cc -O2 runtime/tests/decode_bw_probe.c runtime/amd/hsa_backend.c \
 *     -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
 *     -lhsa-runtime64 -o "$BW_OUT/decode_bw_probe"
 * '
 *
 * Run on R9700:
 * nix develop --command python3 scripts/bench/gpuq.py --vendor amd submit bandwidth 1 \
 *   /tmp/r9700-ubench/decode_bw_probe --object=/tmp/r9700-ubench/bw_probe.elf \
 *   --gib=8 --stream-only
 *
 * Optional: --dev=0 --iters=10 --wg-per-cu=4; omit --stream-only for decode
 * addressing shapes too. Streaming sweeps 1,2,4,8,16 WG/CU by default; other
 * shapes retain 1 WG/CU. --gb uses decimal GB; --gib uses binary GiB.
 * Both host and device use the same PLOW_WG_WAVES (default 8), wave64.
 * Reports median and best host launch+wait time; read bytes exclude the tiny
 * checksum stores. Initialization and streaming correctness checks are untimed.
 * R9700 theoretical reference: 644.6 GB/s GDDR6, not a measured ceiling.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <limits.h>
#include <math.h>
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

/* Exercise all four unrolled loads plus the tail, using exact integer sums. */
static int check_stream(plow_hsa* h, int dev, void* dst, void* out, unsigned grid) {
    const size_t n8 = (size_t)grid * 4 + 3;
    uint32_t* src = malloc(n8 * 16);
    uint32_t* sums = malloc((size_t)grid * 4);
    if (!src || !sums) { free(src); free(sums); return -1; }
    for (size_t i = 0; i < n8 * 4; i++) src[i] = (uint32_t)i;
    plow_hsa_kernel k;
    struct { void* o; const void* s; unsigned long long n; } args = {out, dst, n8};
    int rc = -1;
    if (plow_hsa_upload(h, dev, dst, src, n8 * 16) ||
        plow_hsa_get_kernel(h, dev, "bw_probe_stream", &k) ||
        plow_hsa_launch(h, dev, &k, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)) ||
        plow_hsa_wait(h, dev) || plow_hsa_download(h, dev, sums, out, (size_t)grid * 4)) goto done;
    for (unsigned t = 0; t < grid; t++) {
        uint32_t expected = 0;
        for (size_t i = t; i < n8; i += grid) expected += (uint32_t)(16 * i + 6);
        if (sums[t] != expected) {
            fprintf(stderr, "stream checksum[%u]=%u expected=%u\n", t, sums[t], expected);
            goto done;
        }
    }
    rc = 0;
done:
    free(src); free(sums);
    return rc;
}

int main(int argc, char** argv) {
    int dev = 0, iters = 10, reps = 7, stream_only = 0;
    unsigned wg_per_cu = 0;
    double buffer_bytes = 8.0 * 1024 * 1024 * 1024;
    const char* object = "bw_probe.elf";
    for (int i = 1; i < argc; i++) {
        char* end = NULL;
        double value;
        if (!strncmp(argv[i], "--object=", 9)) object = argv[i] + 9;
        else if (!strcmp(argv[i], "--stream-only")) stream_only = 1;
        else if (!strncmp(argv[i], "--gb=", 5) || !strncmp(argv[i], "--gib=", 6)) {
            int binary = argv[i][3] == 'i';
            const char* s = argv[i] + (binary ? 6 : 5);
            value = strtod(s, &end);
            if (end == s || *end || !isfinite(value)) goto usage;
            buffer_bytes = value * (binary ? 1073741824.0 : 1e9);
        } else {
            const char* s;
            if (!strncmp(argv[i], "--dev=", 6)) s = argv[i] + 6;
            else if (!strncmp(argv[i], "--iters=", 8)) s = argv[i] + 8;
            else if (!strncmp(argv[i], "--wg-per-cu=", 12)) s = argv[i] + 12;
            else goto usage;
            unsigned long n = strtoul(s, &end, 10);
            if (end == s || *s == '-' || *end || n > INT_MAX) goto usage;
            if (!strncmp(argv[i], "--dev=", 6)) dev = (int)n;
            else if (!strncmp(argv[i], "--iters=", 8)) iters = (int)n;
            else wg_per_cu = (unsigned)n;
        }
    }
    if (buffer_bytes < 4294967296.0 || buffer_bytes > 1099511627776.0 ||
        iters < 1 || iters > 10000 || wg_per_cu > 64) goto usage;
    const size_t bytes = (size_t)buffer_bytes & ~(size_t)4095;
    plow_hsa* H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    if (plow_hsa_device_info(H, dev, nm, &cus, &lds) || !cus) goto fail;
    const unsigned max_wg = wg_per_cu ? wg_per_cu : 16;
    if ((uint64_t)cus * max_wg * PLOW_WG_THREADS > UINT32_MAX) goto fail;
    const unsigned max_grid = cus * max_wg * PLOW_WG_THREADS;
    if (((size_t)max_grid * 4 + 3) * 16 > bytes) goto fail;
    printf("dev%d %s CUs=%u wave64 %d threads/WG bytes=%zu (%.3f GiB)\n",
           dev, nm, cus, PLOW_WG_THREADS, bytes, bytes / 1073741824.0);

    FILE* f = fopen(object, "rb");
    if (!f) { perror(object); return 1; }
    if (fseek(f, 0, SEEK_END)) { perror(object); return 1; }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET)) { fprintf(stderr, "invalid object\n"); return 1; }
    void* co = malloc((size_t)n);
    if (!co || fread(co, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read object failed\n"); return 1; }
    fclose(f);
    int loaded = plow_hsa_load_code_object(H, dev, co, (size_t)n);
    free(co);
    if (loaded) goto fail;
    void* dS = plow_hsa_alloc(H, dev, bytes);
    void* dO = plow_hsa_alloc(H, dev, (size_t)max_grid * sizeof(float));
    if (!dS || !dO) goto fail;
    if (check_stream(H, dev, dS, dO, max_grid)) {
        fprintf(stderr, "stream correctness check failed\n"); goto fail;
    }
    printf("stream checksum/tail check=pass\n");
    plow_hsa_kernel init;
    struct { void* s; unsigned long long n; } init_args = {dS, bytes / 16};
    if (plow_hsa_get_kernel(H, dev, "bw_probe_init", &init) ||
        plow_hsa_launch(H, dev, &init, max_grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &init_args, sizeof(init_args)) ||
        plow_hsa_wait(H, dev)) goto fail;

    struct { const char* k; int rowd; } K[] = {
        {"bw_probe_stream", 0}, {"bw_probe_rows_512", 512}, {"bw_probe_coop_512", 512},
        {"bw_probe_rows_256", 256}, {"bw_probe_coop_256", 256},
        {"bw_probe_rows8_512", 512}, {"bw_probe_rows8_256", 256},
    };
    double best_bw = 0;
    unsigned best_wg = 0;
    for (unsigned i = 0; i < (stream_only ? 1 : sizeof(K) / sizeof(K[0])); i++) {
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(H, dev, K[i].k, &k)) goto fail;
        struct __attribute__((packed)) { void* o; const void* s; unsigned long long n; } a8 =
            {dO, dS, bytes / 16};
        struct __attribute__((packed)) { void* o; const void* s; unsigned n; } a4 =
            {dO, dS, (unsigned)(bytes / (K[i].rowd ? K[i].rowd * 2 : 1024))};
        const void* args = K[i].rowd ? (const void*)&a4 : (const void*)&a8;
        size_t asz = K[i].rowd ? sizeof(a4) : sizeof(a8);
        const unsigned limit = wg_per_cu ? wg_per_cu : (K[i].rowd ? 1 : max_wg);
        for (unsigned wg = wg_per_cu ? wg_per_cu : 1; wg <= limit; wg *= 2) {
            const unsigned grid = cus * wg * PLOW_WG_THREADS;
            if (plow_hsa_launch(H, dev, &k, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz) ||
                plow_hsa_wait(H, dev)) goto fail;
            double ts[7];
            for (int r = 0; r < reps; r++) {
                double t0 = now_us();
                for (int it = 0; it < iters; it++)
                    if (plow_hsa_launch(H, dev, &k, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, args, asz)) goto fail;
                if (plow_hsa_wait(H, dev)) goto fail;
                ts[r] = (now_us() - t0) / iters;
            }
            qsort(ts, reps, sizeof(double), cmpd);
            const double bw = bytes / ts[reps / 2] / 1e3;
            printf("%-20s WG/CU=%2u median=%8.1f us %7.1f GB/s best=%7.1f GB/s\n",
                   K[i].k, wg, ts[reps / 2], bw, bytes / ts[0] / 1e3);
            if (!K[i].rowd && bw > best_bw) { best_bw = bw; best_wg = wg; }
            fflush(stdout);
        }
    }
    printf("best stream median: %.1f GB/s (WG/CU=%u, %.3f GiB)\n", best_bw, best_wg, bytes / 1073741824.0);
    plow_hsa_free(H, dO);
    plow_hsa_free(H, dS);
    plow_hsa_shutdown(H);
    return 0;
fail:
    fprintf(stderr, "%s\n", plow_hsa_last_error());
    plow_hsa_shutdown(H);
    return 1;
usage:
    fprintf(stderr, "usage: %s [--object=FILE] [--dev=N] [--gib=N|--gb=N] [--iters=N]\n"
            "  [--wg-per-cu=N] [--stream-only]; buffer must be >=4 GiB\n", argv[0]);
    return 1;
}
