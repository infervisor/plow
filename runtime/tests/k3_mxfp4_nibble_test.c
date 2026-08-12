/* k3_mxfp4_nibble_test.c — which nibble holds element 2i? One GEMV, one real K3 expert tensor.
 *                                                                                  [K3-MXFP4]
 * `perf-data/archive/k3/kimi-k3-kernel-gap.md` §4c is explicit that this is the ONE mxfp4 fact the checkpoint
 * bytes cannot settle: a nibble swap permutes elements within a byte and leaves every per-block
 * multiset — and therefore every histogram — unchanged. It is also explicit about the cost of
 * being wrong: "every mxfp4 number is garbage in a way that looks like 'the model is just bad'".
 *
 * The oracle emits BOTH readings of one real `experts.0.w1` tensor. This runs plow's
 * PLOW_DOP_GEMV_MXFP4 arm on the raw checkpoint bytes and reports which one the hardware agrees
 * with. There is no tolerance argument to make: the two answers differ by ~100% of the norm, so
 * the verdict is unambiguous or the probe is broken and says so.
 *
 *   ./mxnib_test [interp_decode.elf] [mxnib_fixture.bin]
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../amd/hsa_backend.h"
#include "k3_test_arch.h"
#include "../common/dev_isa.h"

typedef uint16_t bf16;
static float b2f(bf16 b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}
static void upload_or_die(plow_hsa* h, void* dst, const void* src, size_t bytes) {
    if (plow_hsa_upload(h, 0, dst, src, bytes)) {
        fprintf(stderr, "upload %zu failed: %s\n", bytes, plow_hsa_last_error());
        exit(1);
    }
}
static double rel(const bf16* got, const float* want, size_t n) {
    double se = 0, sw = 0;
    for (size_t i = 0; i < n; i++) {
        double d = b2f(got[i]) - (double)want[i];
        se += d * d;
        sw += (double)want[i] * want[i];
    }
    return sqrt(se / (double)n) / (sqrt(sw / (double)n) + 1e-30);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const char* fix = argc > 2 ? argv[2] : "mxnib_fixture.bin";
    setbuf(stdout, NULL);

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    printf("dev0: %s CUs=%u\n", gfx, cus);

    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel kern;
    if (plow_hsa_get_kernel(h, 0, PLOW_K3_DECODE_KERNEL, &kern)) { printf("no kernel\n"); return 1; }
    printf("kernel: private=%u B/workitem, LDS=%u B, kernarg=%u B\n",
           kern.private_segment_size, kern.group_segment_size, kern.kernarg_size);

    int fd = open(fix, O_RDONLY);
    if (fd < 0) { perror(fix); return 1; }
    struct stat st; fstat(fd, &st);
    char* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int32_t* hdr = (int32_t*)base;
    if (hdr[0] != 0x4D584E31) { printf("bad magic %x (want MXN1)\n", hdr[0]); return 1; }
    const int N = hdr[1], K = hdr[2];
    printf("mxfp4 GEMV probe: N=%d K=%d  (packed [N,K/2], scale [N,K/32] E8M0 bias 127)\n", N, K);

    size_t off = 4 * 4;
#define NEXT(cnt, elt) ({ void* _p = base + off; off += (size_t)(cnt) * (elt); _p; })
    bf16* P_x = NEXT((size_t)K, 2);
    uint8_t* P_w = NEXT((size_t)N * (K / 2), 1);
    uint8_t* P_s = NEXT((size_t)N * (K / 32), 1);
    float* R_lo = NEXT((size_t)N, 4);
    float* R_hi = NEXT((size_t)N, 4);
    if (off != (size_t)st.st_size) {
        printf("FIXTURE SIZE MISMATCH: consumed %zu, file %zu\n", off, (size_t)st.st_size);
        return 1;
    }

    void* d_x = plow_hsa_alloc(h, 0, (size_t)K * 2);
    void* d_w = plow_hsa_alloc(h, 0, (size_t)N * (K / 2));
    void* d_s = plow_hsa_alloc(h, 0, (size_t)N * (K / 32));
    void* d_c = plow_hsa_alloc(h, 0, (size_t)N * 2);
    upload_or_die(h, d_x, P_x, (size_t)K * 2);
    upload_or_die(h, d_w, P_w, (size_t)N * (K / 2));
    upload_or_die(h, d_s, P_s, (size_t)N * (K / 32));
    bf16* poison = malloc((size_t)N * 2);
    for (int i = 0; i < N; i++) poison[i] = (bf16)0x7fc0u;
    upload_or_die(h, d_c, poison, (size_t)N * 2);
    void* tens[4] = { d_c, d_x, d_w, d_s };
    void* d_tens = plow_hsa_alloc(h, 0, sizeof(tens));
    upload_or_die(h, d_tens, tens, sizeof(tens));

    PlowDevInst in; memset(&in, 0, sizeof(in));
    in.op = PLOW_DOP_GEMV_MXFP4; in.blocks = (uint16_t)cus;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    for (int i = 4; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.i[0] = 1; in.i[1] = (uint32_t)N; in.i[2] = (uint32_t)K;
    void* d_inst = plow_hsa_alloc(h, 0, sizeof(in));
    upload_or_die(h, d_inst, &in, sizeof(in));

    PlowStreamEnt* stream = calloc(cus, sizeof(PlowStreamEnt));
    uint32_t *sofs = calloc(cus, 4), *slen = calloc(cus, 4);
    for (unsigned cu = 0; cu < cus; cu++) {
        stream[cu].inst = 0; stream[cu].slice = cu;
        stream[cu].succ_ofs = 0; stream[cu].succ_len = 1;
        sofs[cu] = cu; slen[cu] = 1;
    }
    uint32_t succ0 = 0;
    void* d_stream = plow_hsa_alloc(h, 0, (size_t)cus * sizeof(PlowStreamEnt));
    void* d_sofs = plow_hsa_alloc(h, 0, 4u * cus);
    void* d_slen = plow_hsa_alloc(h, 0, 4u * cus);
    void* d_succ = plow_hsa_alloc(h, 0, 4);
    void* d_ctr = plow_hsa_alloc(h, 0, (size_t)PLOW_CTR_STRIDE * 4);
    void* d_waits = plow_hsa_alloc(h, 0, sizeof(PlowWait));
    upload_or_die(h, d_stream, stream, (size_t)cus * sizeof(PlowStreamEnt));
    upload_or_die(h, d_sofs, sofs, 4u * cus);
    upload_or_die(h, d_slen, slen, 4u * cus);
    upload_or_die(h, d_succ, &succ0, 4);
    uint32_t* zc = calloc(PLOW_CTR_STRIDE, 4);
    upload_or_die(h, d_ctr, zc, (size_t)PLOW_CTR_STRIDE * 4);

    PlowProgram prog; memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs; prog.stream_len = d_slen;
    prog.waits = d_waits; prog.succs = d_succ; prog.counters = d_ctr;
    prog.tensors = (void* const*)d_tens;
    if (plow_hsa_launch(h, 0, &kern, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog))) {
        printf("LAUNCH FAILED\n");
        return 1;
    }
    if (plow_hsa_wait(h, 0)) {
        printf("WAIT FAILED\n");
        return 1;
    }

    bf16* got = malloc((size_t)N * 2);
    plow_hsa_download(h, 0, got, d_c, (size_t)N * 2);
    plow_hsa_download(h, 0, zc, d_ctr, (size_t)PLOW_CTR_STRIDE * 4);
    printf("executed on %u of %u workgroups\n", zc[0], cus);
    double got_ss = 0.0;
    size_t got_nz = 0;
    for (int i = 0; i < N; i++) {
        const double v = b2f(got[i]);
        got_ss += v * v;
        got_nz += v != 0.0;
    }
    printf("device output: |y| %.6f, nonzero %zu / %d\n", sqrt(got_ss), got_nz, N);

    double r_lo = rel(got, R_lo, (size_t)N);
    double r_hi = rel(got, R_hi, (size_t)N);
    printf("\n  device vs LOW-nibble-is-even-k  (plow's documented layout): %10.3e\n", r_lo);
    printf("  device vs HIGH-nibble-is-even-k (the swap)                : %10.3e\n", r_hi);

    /* bf16 output of an fp32 reference: ~4e-3 is the floor, and the wrong order is ~1e0. There is
     * no band between them, so a "both are large" result means the kernel is broken, not that the
     * order is ambiguous — and that is reported as its own verdict rather than folded into one. */
    const double TOL = 1.5e-2;
    int lo_ok = r_lo < TOL, hi_ok = r_hi < TOL;
    if (lo_ok && !hi_ok) {
        printf("\n=> NIBBLE ORDER CONFIRMED: element 2i is the LOW nibble. plow's `dev.rs:679` \n"
               "   comment, its packer at op_moe.h:1317, and compressed_tensors' convention all \n"
               "   agree with the hardware on real Kimi-K3 bytes. gap-doc SS4c is CLOSED.\n");
        return 0;
    }
    if (hi_ok && !lo_ok) {
        printf("\n*** NIBBLE ORDER IS REVERSED: element 2i is the HIGH nibble. Every mxfp4 weight \n"
               "    plow decodes is permuted in pairs along K. This invalidates every 'COVERED' \n"
               "    mxfp4 verdict until the packer/unpacker is swapped. ***\n");
        return 1;
    }
    printf("\n*** PROBE INCONCLUSIVE — the device matches NEITHER reading (%.3e / %.3e). That is a \n"
           "    broken kernel or a broken fixture, not an ambiguous layout. Do not read a nibble \n"
           "    verdict out of this run. ***\n", r_lo, r_hi);
    return 1;
}
