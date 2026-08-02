/* gemm_tensile_ext.c — dispatch a hipBLASLt/Tensile ASSEMBLY code object directly from
 * plow's OWN HSA backend, and race it against plow's HIP-source GEMM on the same shape.
 *
 * WHY THIS EXISTS.  `GEMM_MFMA_SHAPE_VERDICT.md` proved that plow's prefill GEMM cannot
 * reach hipBLASLt's MfmaUtil *from HIP source* — matching the library's MFMA shape, tile
 * and accumulator-chain count left MfmaUtil at 18-30% vs the library's 52-62%.  The
 * advantage is Tensile's hand-scheduled assembly pipeline.  The only way to get it is to
 * dispatch Tensile's own code object.  This bench measures whether plow can do that with
 * no HIP runtime, no hipBLASLt host library, and no ABI guesswork — i.e. whether an
 * external kernel can be one more AQL packet in `run_segmented`'s loop.
 *
 * THE ABI (read out of the code object, not guessed — see the .note metadata):
 *   Tensile "Custom" UserArgs kernels take a FLAT 104-byte kernarg with NO hidden COv5
 *   block (kernarg_segment_size == 104 == sum of the explicit args), so plow's existing
 *   `plow_hsa_launch` passes it through untouched.
 *
 *     off  0  u32  Gemm info      -- low 30 bits = gemm count; top 2 bits = arg mode.
 *                                   mode 0 => args are INLINE at offset 16 (what we use);
 *                                   nonzero => kernel re-loads a pointer from offset 16
 *                                   and reads a UserArguments struct out of DEVICE memory.
 *     off  4  u32  kernel info0   -- bits 0..13 WorkGroupMapping, bit 14 its direction.
 *     off  8  u32  kernel info1   -- bits 16.. WorkGroupMappingXCC, bits 22.. XCC group.
 *     off 12  u32  numWG          -- total workgroups (the XCC remap needs the count).
 *     off 16  u32  SizesFree0     -- Tensile's i (free index 0)
 *     off 20  u32  SizesFree1     -- Tensile's j
 *     off 24  u32  SizesFree2     -- batch
 *     off 28  u32  SizesSum0      -- K
 *     off 32  ptr  D, C, A, B     -- 4 x 8 bytes
 *     off 64  u32  strideD0,D1, strideC0,C1, strideA0,A1, strideB0,B1
 *     off 96  f32  alpha, beta
 *
 * LAYOUT MAPPING.  plow computes C[m][n] = sum_k A[m][k]*B[n][k] with all three ROW-major
 * (`gemm_bench_8k.c` spot-check).  The kernel is `Cijk_Alik_Bljk`: Tensile's leading index
 * is the FASTEST-varying one, so it wants D fastest in i, A and B fastest in l(=K).  Feed
 * it plow's B as A, plow's A as B, and i:=N, j:=M -- then D's fastest index is plow's n,
 * which IS plow's row-major C.  No transpose, no copy; the swap is free.
 *
 *   usage: gemm_tensile_ext <tensile.co> <tensile_sym> <plow.elf> <plow_sym> M N K
 *                           [info0 info1]
 */
#include "../amd/hsa_backend.h"

#include <math.h>
#include <stdint.h>
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

#define PEAK_TFLOPS 1660.0 /* sustained whole-GPU bf16 MFMA, same denominator as the verdicts */

/* The Tensile Custom-kernel kernarg. Packed; verified 104 bytes at compile time. */
struct __attribute__((packed)) tkarg {
    uint32_t gemm_info, info0, info1, num_wg;
    uint32_t free0, free1, free2, sum0;
    uint64_t d, c, a, b;
    uint32_t sD0, sD1, sC0, sC1, sA0, sA1, sB0, sB1;
    float alpha, beta;
};
_Static_assert(sizeof(struct tkarg) == 104, "Tensile Custom UserArgs kernarg is 104 bytes");

struct __attribute__((packed)) plowarg {
    void* c;
    const void* a;
    const void* b;
    unsigned m, n, k;
};

static plow_hsa* H;
static unsigned NCU;

/* ---- reference + scoring ------------------------------------------------- */

static int check(const bf16* hA, const bf16* hB, const bf16* hC, unsigned M, unsigned N,
                 unsigned K, int nprobe) {
    int bad = 0;
    srand(11);
    for (int t = 0; t < nprobe; t++) {
        unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double acc = 0;
        for (unsigned k = 0; k < K; k++)
            acc += (double)bf2f(hA[(size_t)m * K + k]) * bf2f(hB[(size_t)n * K + k]);
        double g = bf2f(hC[(size_t)m * N + n]);
        if (fabs(g - acc) / (fabs(acc) + 1e-3) > 0.03) bad++;
    }
    return bad;
}

static double timeit(void (*fire)(void*), void* ctx, int warm, int reps) {
    for (int i = 0; i < warm; i++) fire(ctx);
    plow_hsa_wait(H, 0);
    double t0 = now();
    for (int i = 0; i < reps; i++) fire(ctx);
    plow_hsa_wait(H, 0);
    return (now() - t0) / reps;
}

struct fire_ctx {
    plow_hsa_kernel* k;
    uint32_t grid, wg;
    void* args;
    size_t nargs;
};
static void fire(void* p) {
    struct fire_ctx* c = (struct fire_ctx*)p;
    plow_hsa_launch(H, 0, c->k, c->grid, 1, 1, (uint16_t)c->wg, 1, 1, 0, c->args, c->nargs);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
                "usage: %s <tensile.co> <tensile_sym> <plow.elf> <plow_sym> M N K "
                "[info0 info1]\n",
                argv[0]);
        return 2;
    }
    const char* tco = argv[1];
    const char* tsym = argv[2];
    const char* pelf = argv[3];
    const char* psym = argv[4];
    unsigned M = (unsigned)atoi(argv[5]), N = (unsigned)atoi(argv[6]), K = (unsigned)atoi(argv[7]);
    /* Defaults come from the object's own `custom.config`: WorkGroupMapping 16,
     * WorkGroupMappingXCC 2 (encoded as the raw value the kernel bit-tests). */
    /* info0 low 16 = GlobalSplitU (the kernel does `tiles * (info0 & 0x3fff)`), high 16 =
     * the StaggerU triple. info1 low 16 (sign-extended) = WorkGroupMapping, bits 16..21 =
     * WorkGroupMappingXCC, bits 22.. = its group. All four read out of the entry-block
     * disassembly and cross-checked against the object's own `custom.config`
     * (GlobalSplitU 1, StaggerU 0, WorkGroupMapping 16, WorkGroupMappingXCC 2). */
    uint32_t info0 = argc > 8 ? (uint32_t)strtoul(argv[8], 0, 0) : 1u;
    uint32_t info1 = argc > 9 ? (uint32_t)strtoul(argv[9], 0, 0) : 0x20010u;
    int smode = argc > 10 ? atoi(argv[10]) : 1;

    H = plow_hsa_init();
    if (!H) {
        fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error());
        return 1;
    }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    NCU = cus;

    /* --- load BOTH objects into the SAME HSA context. This is the whole point:
     * plow's executable and a vendor Tensile object co-resident, dispatched off one
     * queue, exactly as `run_segmented` already dispatches its three interpreters. */
    /* `plow_hsa_load_code_object` keeps ONE hsa_executable_t per device and a second call
     * overwrites it, so resolve each kernel before loading the next object. The handles
     * stay valid: the previous executable is never destroyed. (plowrt's Rust backend does
     * NOT have this restriction -- `module_load` returns a per-module executable and
     * `get_function` takes the module, so a 4th co-resident object needs no backend work.) */
    plow_hsa_kernel kt, kp;
    const char* paths[2] = {tco, pelf};
    const char* syms[2] = {tsym, psym};
    plow_hsa_kernel* outs[2] = {&kt, &kp};
    for (int which = 0; which < 2; which++) {
        FILE* f = fopen(paths[which], "rb");
        if (!f) { perror(paths[which]); return 1; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        void* img = malloc((size_t)n);
        if (fread(img, 1, (size_t)n, f) != (size_t)n) return 1;
        fclose(f);
        if (plow_hsa_load_code_object(H, 0, img, (size_t)n) != 0) {
            fprintf(stderr, "load %s: %s\n", paths[which], plow_hsa_last_error());
            return 1;
        }
        free(img);
        if (plow_hsa_get_kernel(H, 0, syms[which], outs[which]) != 0) {
            fprintf(stderr, "sym %s: %s\n", syms[which], plow_hsa_last_error());
            return 1;
        }
    }

    printf("%s  %u CU  %u KiB LDS\n", nm, NCU, lds >> 10);
    printf("  tensile %-24s kernarg=%uB LDS=%uB spill=%uB\n", tsym, kt.kernarg_size,
           kt.group_segment_size, kt.private_segment_size);
    printf("  plow    %-24s kernarg=%uB LDS=%uB spill=%uB\n", psym, kp.kernarg_size,
           kp.group_segment_size, kp.private_segment_size);
    /* THE ONE REAL ABI SURPRISE, and it must be carried into any production design:
     * Tensile's hand-written assembly kernels leave `kernarg_size = 0` in the KERNEL
     * DESCRIPTOR (verified byte-wise: KD+8 == 0), so
     * HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE reports 0. The true size
     * lives ONLY in the msgpack `.note` metadata (`.kernarg_segment_size: 104`).
     * A host that trusts HSA here will refuse the dispatch (or under-allocate). plow must
     * therefore carry the kernarg size out-of-band, in the manifest, exactly as hipBLASLt
     * carries it in its .dat solution library. HSA itself does not care: the packet just
     * needs a kernarg buffer at least as large as what the kernel reads. */
    if (kt.kernarg_size == 0) {
        printf("  note: KD declares kernarg_size=0 (Tensile asm); using %zu from metadata\n",
               sizeof(struct tkarg));
        kt.kernarg_size = (uint32_t)sizeof(struct tkarg);
    } else if (kt.kernarg_size != sizeof(struct tkarg)) {
        fprintf(stderr,
                "ABI MISMATCH: object says kernarg=%u, this driver builds %zu. The Custom\n"
                "UserArgs layout changed; re-read the .note metadata before trusting a number.\n",
                kt.kernarg_size, sizeof(struct tkarg));
        return 1;
    }

    /* --- data ------------------------------------------------------------- */
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(H, nA * 2);
    bf16* hB = plow_hsa_alloc_host(H, nB * 2);
    bf16* hC = plow_hsa_alloc_host(H, nC * 2);
    srand(5);
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nB; i++) hB[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    void* dA = plow_hsa_alloc(H, 0, nA * 2);
    void* dB = plow_hsa_alloc(H, 0, nB * 2);
    void* dC = plow_hsa_alloc(H, 0, nC * 2);
    plow_hsa_copy_h2d(H, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(H, 0, dB, hB, nB * 2);

    const double flop = 2.0 * (double)M * N * K;

    /* --- arm 1: Tensile ---------------------------------------------------- */
    /* MT256x256: tile counts live in TENSILE index space, where i:=N and j:=M. */
    const unsigned MT_I = 256, MT_J = 256;
    const unsigned ti = (N + MT_I - 1) / MT_I, tj = (M + MT_J - 1) / MT_J;
    const unsigned nwg = ti * tj;

    struct tkarg ta;
    memset(&ta, 0, sizeof ta);
    ta.gemm_info = 1;   /* one gemm, arg mode 0 => inline args from offset 16 */
    ta.info0 = info0;
    ta.info1 = info1;
    ta.num_wg = nwg;
    ta.free0 = N;       /* i */
    ta.free1 = M;       /* j */
    ta.free2 = 1;       /* batch */
    ta.sum0 = K;
    ta.d = (uint64_t)dC;
    ta.c = (uint64_t)dC; /* beta = 0, but give it a valid pointer regardless */
    ta.a = (uint64_t)dB; /* plow's B is Tensile's A (both K-contiguous, i-major) */
    ta.b = (uint64_t)dA; /* plow's A is Tensile's B */
    /* Two readings of the two stride slots per tensor, and the metadata names alone do not
     * settle it (the inline kernarg calls them strideX0/strideX1; the PUBLIC
     * `hipblaslt::UserArguments` calls the same pair strideX1/strideX2). So make it a knob:
     *   mode 1 (default, matches the public struct): leading dimension, then batch stride.
     *   mode 0: stride of the tensor's own index 0, then index 1.
     * Index 0 is contiguous in all three tensors here, so mode 0's first slot is always 1. */
    if (smode) {
        ta.sD0 = N;      /* D is Dijk, i contiguous => leading dim = |i| = N */
        ta.sD1 = N * M;  /* batch */
        ta.sC0 = N;
        ta.sC1 = N * M;
        ta.sA0 = K;      /* A is Alik, l contiguous => lda = K */
        ta.sA1 = N * K;
        ta.sB0 = K;      /* B is Bljk, l contiguous => ldb = K */
        ta.sB1 = M * K;
    } else {
        ta.sD0 = 1;
        ta.sD1 = N;
        ta.sC0 = 1;
        ta.sC1 = N;
        ta.sA0 = 1;
        ta.sA1 = K;
        ta.sB0 = 1;
        ta.sB1 = K;
    }
    ta.alpha = 1.0f;
    ta.beta = 0.0f;

    struct fire_ctx ct = {&kt, nwg * 256u, 256u, &ta, sizeof ta};

    /* --- arm 2 runs FIRST: plow is the oracle ------------------------------
     * 16 CPU probes tell you "wrong" but not "wrong how". plow's kernel is already
     * trusted on this shape, so keep its FULL output and diff the whole matrix against
     * it -- that distinguishes "every element wrong" (bad layout) from "a band wrong"
     * (bad tile mapping) from "a few wrong" (edge case). */
    struct plowarg pa = {dC, dA, dB, M, N, K};
    memset(hC, 0, nC * 2);
    plow_hsa_copy_h2d(H, 0, dC, hC, nC * 2);
    struct fire_ctx cp = {&kp, NCU * 512u, 512u, &pa, sizeof pa};
    fire(&cp);
    if (plow_hsa_wait(H, 0) != 0) {
        fprintf(stderr, "plow dispatch: %s\n", plow_hsa_last_error());
        return 1;
    }
    bf16* ref = plow_hsa_alloc_host(H, nC * 2);
    plow_hsa_copy_d2h(H, 0, ref, dC, nC * 2);
    int pbad = check(hA, hB, ref, M, N, K, 16);
    double pdt = timeit(fire, &cp, 50, 20);

    /* --- arm 1: Tensile ---------------------------------------------------- */
    memset(hC, 0, nC * 2);
    plow_hsa_copy_h2d(H, 0, dC, hC, nC * 2);
    fire(&ct);
    if (plow_hsa_wait(H, 0) != 0) {
        fprintf(stderr, "tensile dispatch: %s\n", plow_hsa_last_error());
        return 1;
    }
    plow_hsa_copy_d2h(H, 0, hC, dC, nC * 2);
    int tbad = check(hA, hB, hC, M, N, K, 16);
    double tdt = timeit(fire, &ct, 50, 20);

    /* full-matrix diff vs the oracle */
    {
        size_t nz = 0, ndiff = 0, first = (size_t)-1;
        for (size_t i = 0; i < nC; i++) {
            if (hC[i] != 0) nz++;
            float a = bf2f(ref[i]), b = bf2f(hC[i]);
            if (fabsf(a - b) > 0.02f * (fabsf(a) + 1e-3f)) {
                if (first == (size_t)-1) first = i;
                ndiff++;
            }
        }
        printf("\n  full-matrix vs plow: %zu/%zu differ (%.2f%%), %zu/%zu nonzero",
               ndiff, nC, 100.0 * (double)ndiff / (double)nC, nz, nC);
        if (first != (size_t)-1)
            printf(", first at m=%zu n=%zu (plow %.4f, tensile %.4f)", first / N, first % N,
                   bf2f(ref[first]), bf2f(hC[first]));
        printf("\n");
    }

    printf("\n  M=%u N=%u K=%u   (tensile grid %u wg x 256 thr; plow %u wg x 512 thr)\n", M, N, K,
           nwg, NCU);
    printf("  %-10s %9.3f ms  %8.1f TF/s  %5.1f%% peak  %s\n", "tensile", tdt * 1e3,
           flop / tdt / 1e12, 100.0 * flop / tdt / 1e12 / PEAK_TFLOPS,
           tbad ? "MISMATCH" : "ok");
    printf("  %-10s %9.3f ms  %8.1f TF/s  %5.1f%% peak  %s\n", "plow", pdt * 1e3,
           flop / pdt / 1e12, 100.0 * flop / pdt / 1e12 / PEAK_TFLOPS, pbad ? "MISMATCH" : "ok");
    printf("  ratio      %9.2fx  (tensile / plow)\n", pdt / tdt);

    plow_hsa_free(H, dA);
    plow_hsa_free(H, dB);
    plow_hsa_free(H, dC);
    plow_hsa_shutdown(H);
    return (tbad || pbad) ? 3 : 0;
}
