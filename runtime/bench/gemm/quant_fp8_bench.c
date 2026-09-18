/* Paired d_quant_fp8 microbenchmark with bit-exact output and scale comparison.
 * Usage: quant_fp8_bench <control.elf> <candidate.elf> <M> <K> */
#include "../../amd/hsa_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;

static bf16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int load_object(plow_hsa* h, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    const long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    void* bytes = malloc((size_t)n);
    const int ok = bytes && fread(bytes, 1, (size_t)n, f) == (size_t)n &&
                   plow_hsa_load_code_object(h, 0, bytes, (size_t)n) == 0;
    free(bytes);
    fclose(f);
    return ok ? 0 : -1;
}

static int run(const char* object, const bf16* input, unsigned M, unsigned K,
               unsigned char* output, float* scales, double* ms) {
    plow_hsa* h = plow_hsa_init();
    if (!h || load_object(h, object) != 0) {
        fprintf(stderr, "%s: %s\n", object, plow_hsa_last_error());
        plow_hsa_shutdown(h);
        return -1;
    }
    int rc = -1;
    char name[64];
    uint32_t cus = 0, lds = 0;
    if (plow_hsa_device_info(h, 0, name, &cus, &lds) != 0) goto fail;
    plow_hsa_kernel kernel;
    if (plow_hsa_get_kernel(h, 0, "d_quant_fp8_k", &kernel) != 0) goto fail;

    const size_t elems = (size_t)M * K;
    void* dx = plow_hsa_alloc(h, 0, elems * sizeof(*input));
    void* dq = plow_hsa_alloc(h, 0, elems);
    void* ds = plow_hsa_alloc(h, 0, M * sizeof(*scales));
    bf16* hx = plow_hsa_alloc_host(h, elems * sizeof(*input));
    unsigned char* hq = plow_hsa_alloc_host(h, elems);
    float* hs = plow_hsa_alloc_host(h, M * sizeof(*scales));
    if (!dx || !dq || !ds || !hx || !hq || !hs) goto free_buffers;
    memcpy(hx, input, elems * sizeof(*input));
    if (plow_hsa_copy_h2d(h, 0, dx, hx, elems * sizeof(*input)) != 0) goto free_buffers;

    struct __attribute__((packed)) {
        void* q;
        void* x;
        void* scale;
        unsigned m, k, nblk;
    } args = {dq, dx, ds, M, K, cus};

    for (int i = 0; i < 50; i++) {
        if (plow_hsa_launch(h, 0, &kernel, cus * 512u, 1, 1, 512, 1, 1, 0,
                            &args, sizeof args) != 0)
            goto free_buffers;
    }
    if (plow_hsa_wait(h, 0) != 0) goto free_buffers;
    const int groups = 10, reps = 8;
    double total = 0.0;
    for (int g = 0; g < groups; g++) {
        const double begin = now();
        for (int i = 0; i < reps; i++) {
            if (plow_hsa_launch(h, 0, &kernel, cus * 512u, 1, 1, 512, 1, 1, 0,
                                &args, sizeof args) != 0)
                goto free_buffers;
        }
        if (plow_hsa_wait(h, 0) != 0) goto free_buffers;
        total += (now() - begin) / reps;
    }
    *ms = total / groups * 1e3;
    if (plow_hsa_copy_d2h(h, 0, hq, dq, elems) != 0 ||
        plow_hsa_copy_d2h(h, 0, hs, ds, M * sizeof(*scales)) != 0)
        goto free_buffers;
    memcpy(output, hq, elems);
    memcpy(scales, hs, M * sizeof(*scales));
    rc = 0;
free_buffers:
    plow_hsa_free(h, hx);
    plow_hsa_free(h, hq);
    plow_hsa_free(h, hs);
    plow_hsa_free(h, dx);
    plow_hsa_free(h, dq);
    plow_hsa_free(h, ds);
    if (rc != 0) fprintf(stderr, "%s: %s\n", object, plow_hsa_last_error());
    plow_hsa_shutdown(h);
    return rc;
fail:
    fprintf(stderr, "%s: %s\n", object, plow_hsa_last_error());
    plow_hsa_shutdown(h);
    return -1;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <control.elf> <candidate.elf> <M> <K>\n", argv[0]);
        return 2;
    }
    const unsigned M = (unsigned)strtoul(argv[3], NULL, 10);
    const unsigned K = (unsigned)strtoul(argv[4], NULL, 10);
    if (!M || !K) return 2;
    const size_t elems = (size_t)M * K;
    bf16* input = malloc(elems * sizeof(*input));
    unsigned char *control = malloc(elems), *candidate = malloc(elems);
    float *control_scale = malloc(M * sizeof(*control_scale));
    float *candidate_scale = malloc(M * sizeof(*candidate_scale));
    if (!input || !control || !candidate || !control_scale || !candidate_scale) return 1;
    for (size_t i = 0; i < elems; i++)
        input[i] = f2bf(((float)((i * 1315423911u) % 8191u) - 4095.0f) / 17.0f);

    double control_ms, candidate_ms;
    if (run(argv[1], input, M, K, control, control_scale, &control_ms) != 0 ||
        run(argv[2], input, M, K, candidate, candidate_scale, &candidate_ms) != 0)
        return 1;
    const int bytes_equal = memcmp(control, candidate, elems) == 0;
    const int scales_equal = memcmp(control_scale, candidate_scale, M * sizeof(*control_scale)) == 0;
    printf("M=%u K=%u control_ms=%.6f candidate_ms=%.6f speedup=%.6f "
           "bytes_equal=%d scales_equal=%d\n",
           M, K, control_ms, candidate_ms, control_ms / candidate_ms,
           bytes_equal, scales_equal);
    return bytes_equal && scales_equal ? 0 : 1;
}
