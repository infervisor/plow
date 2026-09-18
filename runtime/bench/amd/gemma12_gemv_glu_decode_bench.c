/* HSA driver for gemma12_gemv_glu_decode_bench.hip.
 *
 * Uses Gemma-4 12B's production batch-1 shape and 304-block grid. Each timed launch walks fresh
 * gate/up slabs larger than L2, avoiding a cache-resident microbenchmark. Arms run ABBA/BAAB and
 * must produce bit-identical BF16 output before timings are reported.
 */
#include "../../amd/hsa_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;

static float bf2f(bf16 value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

#define HSA_CHECK(call)                                                                            \
    do {                                                                                           \
        if ((call) != 0) {                                                                         \
            fprintf(stderr, "%s failed: %s\n", #call, plow_hsa_last_error());                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static bf16 pattern(size_t i, unsigned salt) {
    unsigned x = (unsigned)i * 1664525u + 1013904223u + salt;
    return (bf16)(0x3d00u | ((x >> 20) & 0x7fu));
}

static double time_arm(plow_hsa* h, plow_hsa_kernel* k, void* args, size_t args_size,
                       unsigned warmup, unsigned iters, unsigned nrep) {
    for (unsigned i = 0; i < warmup; i++)
        HSA_CHECK(plow_hsa_launch(h, 0, k, 304u * 512u, 1, 1, 512, 1, 1, 0, args, args_size));
    HSA_CHECK(plow_hsa_wait(h, 0));
    double begin = now_ms();
    for (unsigned i = 0; i < iters; i++)
        HSA_CHECK(plow_hsa_launch(h, 0, k, 304u * 512u, 1, 1, 512, 1, 1, 0, args, args_size));
    HSA_CHECK(plow_hsa_wait(h, 0));
    return (now_ms() - begin) / ((double)iters * nrep);
}

static int load_object(plow_hsa* h, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* bytes = malloc((size_t)size);
    int ok = bytes && fread(bytes, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (!ok) { free(bytes); return -1; }
    int rc = plow_hsa_load_code_object(h, 0, bytes, (size_t)size);
    free(bytes);
    return rc;
}

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/gemma12_gemv_glu_decode.elf";
    const unsigned warmup = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 4;
    const unsigned iters = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 9;
    const unsigned N = 15360, K = 3840, act = 0;
    const size_t slab = (size_t)N * K;
    const size_t arena_limit = 3ull << 30;
    unsigned nrep = (unsigned)(arena_limit / (2 * slab * sizeof(bf16)));
    if (!nrep) nrep = 1;

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    if (load_object(h, object) != 0) {
        fprintf(stderr, "load %s: %s\n", object, plow_hsa_last_error());
        return 1;
    }
    plow_hsa_kernel kernels[3];
    const char* names[3] = {"gemma12_gemv_glu_generic_un6", "gemma12_gemv_glu_runtime_un8",
                            "gemma12_gemv_glu_native_un8"};
    for (int i = 0; i < 3; i++)
        if (plow_hsa_get_kernel(h, 0, names[i], &kernels[i]) != 0) {
            fprintf(stderr, "symbol %s: %s\n", names[i], plow_hsa_last_error());
            return 1;
        }

    bf16* hx = plow_hsa_alloc_host(h, K * sizeof(bf16));
    bf16* hw = plow_hsa_alloc_host(h, slab * sizeof(bf16));
    for (size_t i = 0; i < K; i++) hx[i] = pattern(i, 17);
    for (size_t i = 0; i < slab; i++) hw[i] = pattern(i, 31);
    void* dx = plow_hsa_alloc(h, 0, K * sizeof(bf16));
    void* dg = plow_hsa_alloc(h, 0, (size_t)nrep * slab * sizeof(bf16));
    void* du = plow_hsa_alloc(h, 0, (size_t)nrep * slab * sizeof(bf16));
    void* dc[3] = {plow_hsa_alloc(h, 0, N * sizeof(bf16)),
                   plow_hsa_alloc(h, 0, N * sizeof(bf16)),
                   plow_hsa_alloc(h, 0, N * sizeof(bf16))};
    if (!hx || !hw || !dx || !dg || !du || !dc[0] || !dc[1] || !dc[2]) {
        fprintf(stderr, "allocation failed\n"); return 1;
    }
    HSA_CHECK(plow_hsa_copy_h2d(h, 0, dx, hx, K * sizeof(bf16)));
    for (unsigned r = 0; r < nrep; r++)
        HSA_CHECK(plow_hsa_copy_h2d(h, 0, (char*)dg + (size_t)r * slab * sizeof(bf16), hw,
                                    slab * sizeof(bf16)));
    for (size_t i = 0; i < slab; i++) hw[i] = pattern(i, 47);
    for (unsigned r = 0; r < nrep; r++)
        HSA_CHECK(plow_hsa_copy_h2d(h, 0, (char*)du + (size_t)r * slab * sizeof(bf16), hw,
                                    slab * sizeof(bf16)));

    struct __attribute__((packed)) {
        void* c; const void* x; const void* wg; const void* wu;
        unsigned nrep, n, k, act;
    } args[3] = {{dc[0], dx, dg, du, 1, N, K, act}, {dc[1], dx, dg, du, 1, N, K, act},
                 {dc[2], dx, dg, du, 1, N, K, act}};
    for (int i = 0; i < 3; i++) {
        HSA_CHECK(plow_hsa_launch(h, 0, &kernels[i], 304u * 512u, 1, 1, 512, 1, 1, 0,
                                  &args[i], sizeof(args[i])));
        HSA_CHECK(plow_hsa_wait(h, 0));
    }
    bf16* out[3] = {plow_hsa_alloc_host(h, N * sizeof(bf16)),
                    plow_hsa_alloc_host(h, N * sizeof(bf16)),
                    plow_hsa_alloc_host(h, N * sizeof(bf16))};
    for (int i = 0; i < 3; i++)
        HSA_CHECK(plow_hsa_copy_d2h(h, 0, out[i], dc[i], N * sizeof(bf16)));
    size_t bad = 0;
    for (unsigned i = 0; i < N; i++) bad += out[0][i] != out[1][i] || out[0][i] != out[2][i];
    if (bad) { fprintf(stderr, "correctness: %zu/%u outputs differ\n", bad, N); return 2; }
    for (unsigned sample = 0; sample < 12; sample++) {
        const unsigned n = (sample * 1301u + 17u) % N;
        double gate = 0.0, up = 0.0;
        for (unsigned k = 0; k < K; k++) {
            const float xv = bf2f(pattern(k, 17));
            gate += xv * bf2f(pattern((size_t)n * K + k, 31));
            up += xv * bf2f(pattern((size_t)n * K + k, 47));
        }
        const double c = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate);
        const double want = 0.5 * gate * (1.0 + tanh(c)) * up;
        const double got = bf2f(out[2][n]);
        if (fabs(got - want) / (fabs(want) + 1e-3) > 0.03) {
            fprintf(stderr, "oracle mismatch n=%u got=%g want=%g\n", n, got, want);
            return 2;
        }
    }

    args[0].nrep = args[1].nrep = args[2].nrep = nrep;
    double sample[6];
    int order[6] = {0, 1, 2, 2, 1, 0};
    for (int i = 0; i < 6; i++)
        sample[i] = time_arm(h, &kernels[order[i]], &args[order[i]], sizeof(args[0]), warmup,
                             iters, nrep);
    double generic = 0.5 * (sample[0] + sample[5]);
    double runtime = 0.5 * (sample[1] + sample[4]);
    double native = 0.5 * (sample[2] + sample[3]);
    printf("Gemma-12B BF16 batch-1 gate/up+GeGLU N=%u K=%u grid=304 nrep=%u\n", N, K, nrep);
    printf("correctness=bit-exact+f64-sampled generic_ms=%.6f/%.6f runtime8_ms=%.6f/%.6f "
           "native8_ms=%.6f/%.6f pair_mean_speedup_runtime=%.4fx "
           "pair_mean_speedup_native=%.4fx\n",
           sample[0], sample[5], sample[1], sample[4], sample[2], sample[3], generic / runtime,
           generic / native);
    printf("asm generic: private=%uB lds=%uB; runtime: private=%uB lds=%uB; "
           "native: private=%uB lds=%uB\n",
           kernels[0].private_segment_size, kernels[0].group_segment_size,
           kernels[1].private_segment_size, kernels[1].group_segment_size,
           kernels[2].private_segment_size, kernels[2].group_segment_size);
    plow_hsa_shutdown(h);
    return 0;
}
