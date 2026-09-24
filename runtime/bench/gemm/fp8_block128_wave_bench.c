/* Compare the production eight-wave FP8 block128 tile with a one-wave workgroup.
 * Usage: fp8_block128_wave_bench <test_kernels.elf> <M> <N> <K> [<N_first> <N_second>] */
#include "../../amd/hsa_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static int load_object(plow_hsa* h, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    void* bytes = malloc((size_t)n);
    int ok = bytes && fread(bytes, 1, (size_t)n, f) == (size_t)n &&
             plow_hsa_load_code_object(h, 0, bytes, (size_t)n) == 0;
    free(bytes);
    fclose(f);
    return ok ? 0 : -1;
}

struct __attribute__((packed)) args {
    void* c;
    void* a;
    void* b;
    void* as;
    void* ws;
    unsigned m, n, k;
};
struct __attribute__((packed)) split_args {
    void* c;
    void* c1;
    void* c2;
    void* a;
    void* b;
    void* as;
    void* ws;
    unsigned m, n, k, n_first, n_second;
};

static int launch(plow_hsa* h, plow_hsa_kernel* kernel, void* a, size_t arg_size,
                  unsigned waves_per_group, unsigned tiles) {
    unsigned groups = (tiles + waves_per_group - 1) / waves_per_group;
    unsigned threads = waves_per_group * 64;
    return plow_hsa_launch(h, 0, kernel, groups * threads, 1, 1, threads, 1, 1,
                           0, a, arg_size);
}

static double measure_one(plow_hsa* h, plow_hsa_kernel* kernel, void* a, size_t arg_size,
                          unsigned waves_per_group, unsigned tiles) {
    const int reps = 16;
    double begin = now_ms();
    for (int i = 0; i < reps; i++)
        if (launch(h, kernel, a, arg_size, waves_per_group, tiles) != 0) return -1;
    if (plow_hsa_wait(h, 0) != 0) return -1;
    return (now_ms() - begin) * 1e3 / reps;
}

static int measure_pair(plow_hsa* h, plow_hsa_kernel* ctl, plow_hsa_kernel* tr,
                        void* a, size_t arg_size, unsigned tiles,
                        double* ctl_us, double* tr_us) {
    const int groups = 8;
    double samples[2][groups];
    for (int i = 0; i < 10; i++) {
        if (launch(h, ctl, a, arg_size, 8, tiles) != 0 ||
            launch(h, tr, a, arg_size, 1, tiles) != 0) return -1;
    }
    if (plow_hsa_wait(h, 0) != 0) return -1;
    for (int g = 0; g < groups; g++) {
        int first = g & 1, second = first ^ 1;
        samples[first][g] = measure_one(h, first ? tr : ctl, a, arg_size,
                                         first ? 1 : 8, tiles);
        samples[second][g] = measure_one(h, second ? tr : ctl, a, arg_size,
                                          second ? 1 : 8, tiles);
        if (samples[first][g] <= 0 || samples[second][g] <= 0) return -1;
    }
    for (int arm = 0; arm < 2; arm++) {
        for (int i = 1; i < groups; i++)
            for (int j = i; j > 0 && samples[arm][j] < samples[arm][j - 1]; j--) {
                double x = samples[arm][j]; samples[arm][j] = samples[arm][j - 1];
                samples[arm][j - 1] = x;
            }
    }
    *ctl_us = (samples[0][3] + samples[0][4]) * 0.5;
    *tr_us = (samples[1][3] + samples[1][4]) * 0.5;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s <test_kernels.elf> <M> <N> <K> [<N_first> <N_second>]\n", argv[0]);
        return 2;
    }
    unsigned m = (unsigned)strtoul(argv[2], 0, 10);
    unsigned n = (unsigned)strtoul(argv[3], 0, 10);
    unsigned k = (unsigned)strtoul(argv[4], 0, 10);
    unsigned n_first = argc == 7 ? (unsigned)strtoul(argv[5], 0, 10) : 0;
    unsigned n_second = argc == 7 ? (unsigned)strtoul(argv[6], 0, 10) : 0;
    if (!m || !n || !k || k % 128 || m > 64 || n > 32768) return 2;
    if (argc == 7 && (!n_first || !n_second || n_first + n_second >= n)) return 2;
    size_t na = (size_t)m * k, nb = (size_t)n * k, nc = (size_t)m * n;
    size_t ns = (size_t)(k / 128) * m, nw = (size_t)((n + 127) / 128) * (k / 128);
    plow_hsa* h = plow_hsa_init();
    if (!h || load_object(h, argv[1]) != 0) {
        fprintf(stderr, "object: %s\n", plow_hsa_last_error());
        return 1;
    }
    plow_hsa_kernel ctl, tr;
    const char* ctl_name = argc == 7 ? "gemm_qkv_a8w8_block128_m16" : "gemm_a8w8_block128_m16";
    const char* tr_name = argc == 7 ? "gemm_qkv_a8w8_block128_m16_wave1" : "gemm_a8w8_block128_m16_wave1";
    if (plow_hsa_get_kernel(h, 0, ctl_name, &ctl) != 0 ||
        plow_hsa_get_kernel(h, 0, tr_name, &tr) != 0) {
        fprintf(stderr, "kernel: %s\n", plow_hsa_last_error());
        return 1;
    }
    uint8_t* a = plow_hsa_alloc_host(h, na);
    uint8_t* b = plow_hsa_alloc_host(h, nb);
    float* as = plow_hsa_alloc_host(h, ns * sizeof(float));
    float* ws = plow_hsa_alloc_host(h, nw * sizeof(float));
    uint16_t* out = plow_hsa_alloc_host(h, nc * sizeof(uint16_t));
    uint16_t* ref = malloc(nc * sizeof(uint16_t));
    void *da = plow_hsa_alloc(h, 0, na), *db = plow_hsa_alloc(h, 0, nb);
    void *das = plow_hsa_alloc(h, 0, ns * sizeof(float));
    void *dws = plow_hsa_alloc(h, 0, nw * sizeof(float));
    void *dc = plow_hsa_alloc(h, 0, nc * sizeof(uint16_t));
    if (!a || !b || !as || !ws || !out || !ref || !da || !db || !das || !dws || !dc) {
        fprintf(stderr, "allocation: %s\n", plow_hsa_last_error());
        return 1;
    }
    const uint8_t values[] = {0x30, 0x38, 0xb8, 0x40};
    uint32_t rng = 0x8365a41du;
    for (size_t i = 0; i < na; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        a[i] = values[rng & 3u];
    }
    for (size_t i = 0; i < nb; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        b[i] = values[rng & 3u];
    }
    for (size_t i = 0; i < ns; i++) as[i] = 0.0078125f * (1 + i % 7);
    for (size_t i = 0; i < nw; i++) ws[i] = 0.0078125f * (1 + i % 5);
    if (plow_hsa_copy_h2d(h, 0, da, a, na) != 0 ||
        plow_hsa_copy_h2d(h, 0, db, b, nb) != 0 ||
        plow_hsa_copy_h2d(h, 0, das, as, ns * sizeof(float)) != 0 ||
        plow_hsa_copy_h2d(h, 0, dws, ws, nw * sizeof(float)) != 0) {
        fprintf(stderr, "copy: %s\n", plow_hsa_last_error());
        return 1;
    }
    struct args args = {dc, da, db, das, dws, m, n, k};
    struct split_args split_args = {dc, (char*)dc + (size_t)m * n_first * sizeof(uint16_t),
        (char*)dc + (size_t)m * (n_first + n_second) * sizeof(uint16_t),
        da, db, das, dws, m, n, k, n_first, n_second};
    void* selected_args = argc == 7 ? (void*)&split_args : (void*)&args;
    size_t arg_size = argc == 7 ? sizeof(split_args) : sizeof(args);
    unsigned tiles = ((m + 15) / 16) * ((n + 15) / 16);
    if (launch(h, &ctl, selected_args, arg_size, 8, tiles) != 0 || plow_hsa_wait(h, 0) != 0 ||
        plow_hsa_copy_d2h(h, 0, out, dc, nc * sizeof(uint16_t)) != 0) return 1;
    memcpy(ref, out, nc * sizeof(uint16_t));
    if (launch(h, &tr, selected_args, arg_size, 1, tiles) != 0 || plow_hsa_wait(h, 0) != 0 ||
        plow_hsa_copy_d2h(h, 0, out, dc, nc * sizeof(uint16_t)) != 0) return 1;
    size_t mismatches = 0;
    for (size_t i = 0; i < nc; i++) mismatches += ref[i] != out[i];
    double ctl_us = 0, tr_us = 0;
    if (measure_pair(h, &ctl, &tr, selected_args, arg_size, tiles,
                     &ctl_us, &tr_us) != 0) return 1;
    printf("M=%u N=%u K=%u tiles=%u ctl_groups=%u tr_groups=%u mismatches=%zu ctl_us=%.3f tr_us=%.3f speedup=%.3f\n",
           m, n, k, tiles, (tiles + 7) / 8, tiles, mismatches, ctl_us, tr_us,
           ctl_us / tr_us);
    plow_hsa_free(h, a); plow_hsa_free(h, b);
    plow_hsa_free(h, as); plow_hsa_free(h, ws); plow_hsa_free(h, out);
    plow_hsa_free(h, da); plow_hsa_free(h, db);
    plow_hsa_free(h, das); plow_hsa_free(h, dws); plow_hsa_free(h, dc);
    free(ref);
    plow_hsa_shutdown(h);
    return mismatches || ctl_us <= 0 || tr_us <= 0;
}
