/* cuda_backend_test.c — verifies runtime/nvidia/cuda_backend.c.
 *
 * Two checks, both driven ONLY through the plow_hsa_* interface:
 *   (1) alloc 1 MiB, H2D a known pattern, D2H it back, memcmp == 0.
 *       Negative control: corrupt one byte of the host pattern after the
 *       upload and confirm the memcmp reports the mismatch.
 *   (2) load the interp_sm120_poc cubin, build the same 2-instruction
 *       counter-gated DAG the PoC builds, cooperative-launch it through
 *       plow_hsa_launch, and reproduce mismatches=0.
 *
 * Build:
 *   nvcc -arch=sm_120a -cubin -I runtime/common \
 *        runtime/nvidia/interp_sm120_poc.cu -o /workspace/poc.cubin
 *   gcc -O2 -I/usr/local/cuda-12.8/include -I runtime/common \
 *        runtime/tests/cuda_backend_test.c runtime/nvidia/cuda_backend.c \
 *        -L/usr/local/cuda-12.8/lib64/stubs -lcuda -lm -o /workspace/cuda_backend_test
 */
#include "../amd/hsa_backend.h"
#include "dev_isa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PoC opcodes (interp_sm120_poc.cu:41). */
enum { POC_OP_ADD1 = 1000, POC_OP_MUL2 = 1001 };

/* The PoC kernel is a plain C++ __global__, so the cubin carries the Itanium
 * mangling of `interp_sm120(PlowProgram)`. Using the mangled name means the PoC
 * file stays untouched — no forked copy of the device code. */
#define POC_KERNEL "_Z12interp_sm12011PlowProgram"

static plow_hsa* H;
static int DEV = 0;
static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

static void die(const char* what) {
    fprintf(stderr, "%s: %s\n", what, plow_hsa_last_error());
    exit(1);
}

/* --- test 1: memory round trip ------------------------------------------- */

static void test_memory(void) {
    const size_t N = 1u << 20; /* 1 MiB */
    printf("\n=== test 1: alloc %zu B / H2D / D2H / memcmp ===\n", N);

    unsigned char* src = plow_hsa_alloc_host(H, N);
    unsigned char* dst = plow_hsa_alloc_host(H, N);
    if (!src || !dst) die("plow_hsa_alloc_host");
    void* d = plow_hsa_alloc(H, DEV, N);
    if (!d) die("plow_hsa_alloc");

    /* A pattern that catches offset/stride errors, not just zero-vs-nonzero. */
    for (size_t i = 0; i < N; i++) src[i] = (unsigned char)((i * 31u + 7u) & 0xff);
    memset(dst, 0xAB, N);

    if (plow_hsa_copy_h2d(H, DEV, d, src, N) != 0) die("copy_h2d");
    if (plow_hsa_copy_d2h(H, DEV, dst, d, N) != 0) die("copy_d2h");

    int cmp = memcmp(src, dst, N);
    printf("  memcmp(src, dst, %zu) = %d\n", N, cmp);
    CHECK(cmp == 0, "round trip differs");

    /* NEGATIVE CONTROL — prove the comparison can fail. Corrupt exactly one byte
     * of the host pattern; the device copy still holds the original, so memcmp
     * must now report a difference. If this printed 0 the test above would be
     * vacuous. */
    const size_t bad_at = 500000;
    unsigned char keep = src[bad_at];
    src[bad_at] ^= 0x01;
    int cmp_bad = memcmp(src, dst, N);
    printf("  negative control: flipped src[%zu] 0x%02x -> 0x%02x, memcmp = %d\n",
           bad_at, keep, src[bad_at], cmp_bad);
    CHECK(cmp_bad != 0, "negative control did NOT detect a 1-byte corruption");
    src[bad_at] = keep;

    /* And the control-plane (arbitrary host memory) path, which must reach the
     * same bytes from a plain malloc buffer. */
    unsigned char* pageable = malloc(N);
    memset(pageable, 0, N);
    if (plow_hsa_download(H, DEV, pageable, d, N) != 0) die("download");
    int cmp_pageable = memcmp(src, pageable, N);
    printf("  download to pageable malloc: memcmp = %d\n", cmp_pageable);
    CHECK(cmp_pageable == 0, "pageable download differs");
    free(pageable);

    plow_hsa_free(H, d);
    plow_hsa_free(H, src);
    plow_hsa_free(H, dst);
}

/* --- test 2: cooperative launch of the PoC interpreter -------------------- */

static void* upload_new(size_t bytes, const void* src) {
    void* d = plow_hsa_alloc(H, DEV, bytes);
    if (!d) die("alloc");
    if (plow_hsa_upload(H, DEV, d, src, bytes) != 0) die("upload");
    return d;
}

static void test_interp(const char* cubin_path) {
    printf("\n=== test 2: cooperative launch of %s ===\n", POC_KERNEL);

    /* Load the cubin through the backend. */
    FILE* fp = fopen(cubin_path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", cubin_path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* img = malloc((size_t)sz);
    if (fread(img, 1, (size_t)sz, fp) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(fp);
    printf("  cubin: %s (%ld bytes)\n", cubin_path, sz);

    if (plow_hsa_load_code_object(H, DEV, img, (size_t)sz) != 0) die("load_code_object");

    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(H, DEV, POC_KERNEL, &k) != 0) die("get_kernel");
    printf("  kernel: kernarg_size=%u (sizeof(PlowProgram)=%zu) group=%u private=%u\n",
           k.kernarg_size, sizeof(PlowProgram), k.group_segment_size,
           k.private_segment_size);
    CHECK(k.kernarg_size == sizeof(PlowProgram),
          "kernarg_size %u != sizeof(PlowProgram) %zu", k.kernarg_size,
          sizeof(PlowProgram));

    char dname[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, DEV, dname, &cus, &lds);
    printf("  device: %s  SMs=%u  shared/SM=%u B\n", dname, cus, lds);

    /* Grid: the PoC computes blocksPerSM x SMs. We do not have the occupancy API
     * exposed through this interface, so use the value the PoC printed on this
     * GPU (6 blocks/SM at 256 threads) and let the backend's own co-residency
     * check confirm it by choosing a cooperative launch. Any G the hardware
     * cannot hold co-resident would make cuLaunchKernelEx fail, not hang. */
    const int TPB = 256;
    const int G = 6 * (int)cus;
    const unsigned N = 4u * 1024u * 1024u;
    printf("  grid=%d blocks x %d threads, N=%u floats\n", G, TPB, N);

    /* Tensors: 0=x, 1=out0, 2=out1. */
    float* h_x = malloc((size_t)N * sizeof(float));
    for (unsigned i = 0; i < N; i++) h_x[i] = (float)(i % 97);
    void* d_x = upload_new((size_t)N * sizeof(float), h_x);
    void* d_out0 = plow_hsa_alloc(H, DEV, (size_t)N * sizeof(float));
    void* d_out1 = plow_hsa_alloc(H, DEV, (size_t)N * sizeof(float));
    void* h_tensors[3] = {d_x, d_out0, d_out1};
    void* d_tensors = upload_new(sizeof(h_tensors), h_tensors);

    /* The same 2-instruction DAG as interp_sm120_poc.cu:199-216. */
    PlowDevInst insts[2];
    memset(insts, 0, sizeof(insts));
    insts[0].op = POC_OP_ADD1;
    insts[0].blocks = (uint16_t)G;
    insts[0].t[0] = 1; insts[0].t[1] = 0;
    insts[0].i[0] = N;

    insts[1].op = POC_OP_MUL2;
    insts[1].blocks = (uint16_t)G;
    insts[1].t[0] = 2; insts[1].t[1] = 1;
    insts[1].i[0] = N;

    PlowWait waits[1]; memset(waits, 0, sizeof(waits));
    waits[0].id = 0; waits[0].threshold = (uint32_t)G;
    uint32_t succs[2] = {0u, 1u};

    const int NENT = 2 * G;
    PlowStreamEnt* h_stream = calloc((size_t)NENT, sizeof(PlowStreamEnt));
    uint32_t* h_ofs = malloc((size_t)G * sizeof(uint32_t));
    uint32_t* h_len = malloc((size_t)G * sizeof(uint32_t));
    for (int b = 0; b < G; b++) {
        h_ofs[b] = (uint32_t)(2 * b);
        h_len[b] = 2;
        h_stream[2 * b + 0].inst = 0; h_stream[2 * b + 0].slice = (uint32_t)b;
        h_stream[2 * b + 1].inst = 1; h_stream[2 * b + 1].slice = (uint32_t)b;
        /* Gates live on the stream entries (64-byte PlowDevInst carries none). */
        h_stream[2 * b + 0].succ_len = 1; h_stream[2 * b + 0].succ_ofs = 0;
        h_stream[2 * b + 1].wait_len = 1; h_stream[2 * b + 1].wait_ofs = 0;
        h_stream[2 * b + 1].succ_len = 1; h_stream[2 * b + 1].succ_ofs = 1;
    }

    const int NCTR = 2;
    size_t ctr_bytes = (size_t)NCTR * PLOW_CTR_STRIDE * sizeof(uint32_t);
    uint32_t* h_ctr = calloc(1, ctr_bytes);

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts      = upload_new(sizeof(insts), insts);
    prog.stream     = upload_new((size_t)NENT * sizeof(PlowStreamEnt), h_stream);
    prog.stream_ofs = upload_new((size_t)G * sizeof(uint32_t), h_ofs);
    prog.stream_len = upload_new((size_t)G * sizeof(uint32_t), h_len);
    prog.waits      = upload_new(sizeof(waits), waits);
    prog.succs      = upload_new(sizeof(succs), succs);
    prog.counters   = upload_new(ctr_bytes, h_ctr); /* zeroed */
    prog.tensors    = d_tensors;

    /* grid_* are WORK-ITEM counts (HSA semantics); the backend divides. */
    if (plow_hsa_launch(H, DEV, &k,
                        (uint32_t)G * TPB, 1, 1,
                        (uint16_t)TPB, 1, 1,
                        0, &prog, sizeof(prog)) != 0) die("plow_hsa_launch");
    if (plow_hsa_wait(H, DEV) != 0) die("plow_hsa_wait");

    float* h_out1 = malloc((size_t)N * sizeof(float));
    if (plow_hsa_download(H, DEV, h_out1, d_out1, (size_t)N * sizeof(float)) != 0)
        die("download out1");

    unsigned bad = 0;
    double maxerr = 0;
    for (unsigned i = 0; i < N; i++) {
        float want = (h_x[i] + 1.0f) * 2.0f;
        double err = h_out1[i] - want;
        if (err < 0) err = -err;
        if (err > 1e-4) { bad++; if (err > maxerr) maxerr = err; }
    }
    printf("  verify: N=%u  mismatches=%u  maxerr=%g\n", N, bad, maxerr);
    CHECK(bad == 0, "interpreter produced %u mismatches", bad);

    /* NEGATIVE CONTROL — the checker must be able to fail. Corrupt one element
     * of the device output and re-verify. */
    float poison = -12345.0f;
    if (plow_hsa_upload(H, DEV, (char*)d_out1 + 777 * sizeof(float),
                        &poison, sizeof(poison)) != 0) die("poison upload");
    if (plow_hsa_download(H, DEV, h_out1, d_out1, (size_t)N * sizeof(float)) != 0)
        die("download out1 (2)");
    unsigned bad2 = 0;
    for (unsigned i = 0; i < N; i++) {
        float want = (h_x[i] + 1.0f) * 2.0f;
        double err = h_out1[i] - want;
        if (err < 0) err = -err;
        if (err > 1e-4) bad2++;
    }
    printf("  negative control: poisoned d_out1[777] -> mismatches=%u\n", bad2);
    CHECK(bad2 == 1, "negative control expected exactly 1 mismatch, got %u", bad2);
}

int main(int argc, char** argv) {
    const char* cubin = argc > 1 ? argv[1] : "/workspace/poc.cubin";

    H = plow_hsa_init();
    if (!H) die("plow_hsa_init");
    printf("plow_hsa_device_count = %d\n", plow_hsa_device_count(H));

    test_memory();
    test_interp(cubin);

    printf("\n%s (%d failure(s))\n", failures ? "RESULT: FAIL" : "RESULT: PASS", failures);
    plow_hsa_shutdown(H);
    return failures ? 1 : 0;
}
