/* gemv_row_sweep.c — sweep the DECODE GEMV over its five row buckets at one (N,K).
 *
 * THE GAP THIS FILLS. `tunedb` is GEMM-ONLY: `tunedb::gemm_op_case` is its sole shape lookup,
 * and the decode program contains ZERO `Gemm` ops — every decode matmul is
 * `Gemv`/`GemvQkv`/`GemvGlu`/`GemvFp8Blk`. So the tuner cannot touch ms/token, and the entire
 * decode path has never had a shape campaign. The row bucket `PLOW_GEMV_MM` is selected by
 * CAPACITY (does the object's compiled MM cover the packet's M) and never by measurement.
 * `runtime/ubench/gemv_bw_gfx950.hip` is a bandwidth microbench on synthetic shapes, not a
 * sweep over the shapes the compiler asks for.
 *
 *   usage: gemv_row_sweep <N> <K> [label]
 *
 * The M values are NOT on argv: they are the five rungs `PLOW_GEMV_MM` can take, which is the
 * axis being measured. The (N,K) pairs come from the compiler's own census —
 * `PLOW_TUNE_DUMP=1` now emits one `TUNEDUMP_GEMV` line per resolved GEMV shape
 * (`crates/packet/src/devbuild.rs`), and `scripts/rebench_tune_gemv.sh` derives its list from
 * that dump. Hand-authoring the list is exactly how GLM-5.2 prefill ended up 100% unmeasured
 * on the GEMM side; do not do it here.
 *
 * WHAT IT MEASURES, and what it does NOT. There is no per-shape rung to select on this path:
 * MM is a macro of the object and the K-unroll is a runtime branch inside the kernel. What the
 * numbers decide is the OBJECT: which `PLOW_GEMV_MM`, and whether `PLOW_GEMV_WALK` is on. Every
 * kernel here walks (`ceil(M/MM)` row blocks with the staging inside the loop), so at M == MM
 * it is the shipping non-walking form and at M > MM it is the walk — both from one build, which
 * is the only way they can appear in one table (separate objects would sit under separate build
 * digests and could never be compared as one measurement).
 *
 * THREE ARMS, because the decode stream's COMPOSITION changes with M and not only its width:
 *   gemv_m<MM>      plain  C[M,N] = W[N,K] . x[M,K]
 *   gemv_glu_m<MM>  fused gate|up + SwiGLU   (the `glu_fused` arm)
 *   gemv_qkv_m<MM>  fused q|k|v              (the `fuse_qkv` arm)
 * §6g-BATCH's B=16 regression (142.4 tok/s against B=8's 202.3) is TWO things at once — MM=16
 * spilling AND both fusions turning off, because devgen gated them on `t * hidden <=
 * GM_LDS_HALVES` and `16 * 5376 = 86016 > 73728`. Measuring only the plain arm would price one
 * of the two and attribute the whole loss to it.
 *
 * With PLOW_GEMV_JSONL=<path> it appends one raw-sample row per (arm, MM), which
 * `tunedb-gemv ingest` turns into qualified `kernel_measurement` records. The C side writes
 * SAMPLES and a correctness verdict and nothing else — the build digests that decide staleness
 * come from probing the interpreter, which is the Rust side's job. That split is the same one
 * `gemm_tile_sweep.c` uses, and a missing ingest step silently killed a benchmark run today.
 *
 * THE ORACLE CHECKS EVERY ROW, unlike the GEMM one's sampled elements. `gemv_rows<MM>`
 * predicates each row on `m < M` and has no outer loop of its own, so the characteristic
 * failure is not a wrong value — it is an UNTOUCHED row. That is the `PLOW_GEMV_MM` bug
 * (`scripts/build_gfx950.sh:51`: every AMD decode object compiled at MM=1, wrote row 0, and
 * left rows 1..B-1 exactly as it found them — fluent output, rms error sqrt((T-1)/T), no fault
 * anywhere) and §6g-BATCH's B=16 slots 13/14/15. A spot-check that samples random (m,n) pairs
 * can miss it; poisoning C and checking that every row moved cannot.
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

/* Streaming vec-read ceiling, not the 8 TB/s nominal: op_gemm.h records ~6.4 TB/s as the real
 * one (even the wide lm_head tops out there), and a roofline against the nominal would report
 * every arm as failing. */
#define HBM_GBPS 6400.0
#define POISON 0x7f7f /* a bf16 no dot product produces; an untouched row keeps it */

static const unsigned MMS[] = {1, 2, 4, 8, 16};
#define NMM ((int)(sizeof MMS / sizeof MMS[0]))

/* ---------------------------------------------------------------------------------------------
 * MXFP4 (OCP microscaling), the w4a16 arms.
 *
 * LAYOUT, matching `op_gemm.h`'s `gemv_rows_mxfp4` and the Kimi-K3 checkpoint byte-for-byte:
 *   W[n][k]  packed 2 fp4/byte, row stride K/2 bytes, LOW nibble = even k
 *   S[n][j]  one E8M0 byte per 32-K block, row stride K/32, value 2^(byte - 127)
 * The nibble order is the one fact the checkpoint bytes provably cannot settle (a swap permutes
 * elements within a byte and leaves every per-block statistic identical), and
 * `runtime/tests/k3_mxfp4_nibble_test.c` settled it on hardware: element 2i is the LOW nibble.
 * This encoder writes that order, so a kernel that reads the other one shows up as a MISMATCH
 * here rather than as plausible-looking numbers.
 *
 * WHY THE ORACLE DEQUANTISES RATHER THAN COMPARING AGAINST THE bf16 WEIGHTS. The reference is
 * the f64 dot of the DECODED fp4 values, not of the bf16 originals. Those differ by the
 * quantisation error, which at 4 bits is percent-scale — comparing against the originals would
 * measure the QUANTISER and drown the kernel's own error in it, so a genuinely broken decode
 * (a swapped nibble, a mis-strided scale row) would sit inside the same tolerance as a correct
 * one. Checking against the decoded values makes the tolerance tight and the check about the
 * kernel, which is the only thing this harness can actually verify. */

/* OCP e2m1: 1 sign, 2 exponent, 1 mantissa. Eight magnitudes, no inf, no NaN. */
static const float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/* Nearest-representable e2m1 code for |v| (ties to the larger magnitude, as the OCP round-to-
 * nearest-even reference does for these eight exactly-representable points).
 *
 * A binary search over the seven midpoints rather than a linear scan of `E2M1`: this runs once
 * per weight ELEMENT, and the lm_head fixture is 163840 x 7168 = 1.17e9 of them, where the
 * eight-way scan cost minutes of wall clock before the first kernel launched. */
static unsigned char fp4_code(float v) {
    const float a = fabsf(v);
    unsigned c;
    if (a < 1.75f) {
        if (a < 0.75f) c = (a < 0.25f) ? 0u : 1u;
        else c = (a < 1.25f) ? 2u : 3u;
    } else {
        if (a < 3.5f) c = (a < 2.5f) ? 4u : 5u;
        else c = (a < 5.0f) ? 6u : 7u;
    }
    return (unsigned char)(c | (v < 0.0f ? 8u : 0u));
}

/* Decode one element back, exactly as `fp4_to_bf16v8x4` folds the scale into the convert. */
static float fp4_deq(const unsigned char* W, const unsigned char* S, unsigned n, unsigned k,
                     unsigned K) {
    const unsigned char byte = W[(size_t)n * (K / 2) + (k >> 1)];
    const unsigned code = (k & 1u) ? (byte >> 4) : (byte & 0xfu); /* LOW nibble = even k */
    const float mag = E2M1[code & 7u];
    const int e = (int)S[(size_t)n * (K / 32) + (k >> 5)] - 127;
    return ldexpf((code & 8u) ? -mag : mag, e);
}

/* Quantise one bf16 weight matrix to mxfp4 in place over 32-K blocks.
 *
 * The E8M0 exponent is chosen so the block's absmax lands at the top of e2m1's range (6.0):
 * `e = floor(log2(absmax)) - 2`, which is the standard MX shared-exponent rule. A block that is
 * entirely zero takes exponent 0 (byte 127, scale 1.0) — NOT byte 0, which would mean 2^-127
 * and is the flush-to-zero encoding; either decodes this block to zeros, but 127 keeps the
 * fixture inside the byte range the real checkpoint uses. */
static void quantise_mxfp4(unsigned char* W, unsigned char* S, const bf16* src, unsigned N,
                           unsigned K, float (*b2f)(bf16)) {
    const unsigned nsb = K / 32;
    for (unsigned n = 0; n < N; n++) {
        for (unsigned j = 0; j < nsb; j++) {
            const bf16* blk = src + (size_t)n * K + (size_t)j * 32;
            float amax = 0.0f;
            for (unsigned t = 0; t < 32; t++) {
                const float a = fabsf(b2f(blk[t]));
                if (a > amax) amax = a;
            }
            int e = 0;
            if (amax > 0.0f) {
                e = (int)floorf(log2f(amax)) - 2; /* absmax -> the 4.0..6.0 top of e2m1 */
                if (e < -127) e = -127;
                if (e > 127) e = 127;
            }
            S[(size_t)n * nsb + j] = (unsigned char)(e + 127);
            const float inv = ldexpf(1.0f, -e);
            for (unsigned t = 0; t < 32; t += 2) {
                const unsigned char lo = fp4_code(b2f(blk[t]) * inv);
                const unsigned char hi = fp4_code(b2f(blk[t + 1]) * inv);
                W[(size_t)n * (K / 2) + (size_t)j * 16 + t / 2] =
                    (unsigned char)((lo & 0xfu) | (unsigned)(hi << 4)); /* LOW = even k */
            }
        }
    }
}

/* The bf16 sentinel written into C before each launch. Rows the kernel never touches come back
 * carrying it, which is the failure the whole oracle exists to catch. */
static void poison(bf16* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = POISON;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <N> <K> [label]\n", argv[0]);
        return 2;
    }
    const unsigned N = (unsigned)atoi(argv[1]), K = (unsigned)atoi(argv[2]);
    const char* label = argc > 3 ? argv[3] : "shape";

    plow_hsa* H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64];
    uint32_t cus = 0, ldsz = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &ldsz);
    const unsigned NCU = cus, THREADS = 512;

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END);
    long nby = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* co = malloc(nby);
    if (fread(co, 1, nby, f) != (size_t)nby) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, nby) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error());
        return 1;
    }

    const unsigned MMAX = MMS[NMM - 1];
    /* Q|K|V splits N three ways so the FUSED op moves the same weight bytes as the plain one at
     * the same (N,K) — otherwise the two arms would not be comparable at all. Nk=Nv is the GQA
     * shape; an odd remainder goes to q. */
    const unsigned Nk = N / 4, Nv = N / 4, Nq = N - Nk - Nv;
    const size_t nX = (size_t)MMAX * K, nW = (size_t)N * K, nC = (size_t)MMAX * N;
    bf16* hX = plow_hsa_alloc_host(H, nX * 2);
    bf16* hW = plow_hsa_alloc_host(H, nW * 2);
    bf16* hC = plow_hsa_alloc_host(H, nC * 2);
    srand(5);
    for (size_t i = 0; i < nX; i++) hX[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nW; i++) hW[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    void* dX = plow_hsa_alloc(H, 0, nX * 2);
    void* dW = plow_hsa_alloc(H, 0, nW * 2);
    void* dW2 = plow_hsa_alloc(H, 0, nW * 2); /* the up / k+v stream */
    void* dC = plow_hsa_alloc(H, 0, nC * 2);
    /* THREE SEPARATE OUTPUTS FOR Q|K|V, not one buffer sliced by column.
     * `gemv_qkv_rows` writes `Cq[m*Nq + n]`, `Ck[m*Nk + n]`, `Cv[m*Nv + n]` — each with its OWN
     * row stride. Pointing all three into one `[M, N]` buffer at column offsets makes those
     * strides disagree with N, so the writes interleave and the row-coverage oracle reports
     * every launch as leaving rows untouched. It did, on the first run: the oracle caught a
     * HARNESS bug rather than a kernel one, which is the correct outcome and the reason the
     * check is coverage-based rather than a sampled spot-check. */
    void* dCq = plow_hsa_alloc(H, 0, (size_t)MMAX * Nq * 2);
    void* dCk = plow_hsa_alloc(H, 0, (size_t)MMAX * Nk * 2);
    void* dCv = plow_hsa_alloc(H, 0, (size_t)MMAX * Nv * 2);
    plow_hsa_copy_h2d(H, 0, dX, hX, nX * 2);
    plow_hsa_copy_h2d(H, 0, dW, hW, nW * 2);
    plow_hsa_copy_h2d(H, 0, dW2, hW, nW * 2);

    /* THE MXFP4 ARMS ARE BUILT FROM THE SAME hW, so bf16 and mxfp4 at one (N,K) are the same
     * numbers at two precisions and the timing difference is the encoding alone.
     *
     * They run at ONE row bucket, not five. `d_gemv_mxfp4_k` / `d_gemv_glu_mxfp4_k` instantiate
     * `gemv_rows_mxfp4<PLOW_GEMV_MM>` — the object's compile-time macro — where the bf16 arms
     * compile five `_m<MM>` walk variants. So there is no mxfp4 M-curve to sweep from one
     * object, and no walk: at M > MM the kernel would leave rows untouched, which the oracle
     * reports as a failure rather than a slow number. The bucket the object was built at comes
     * in on PLOW_GEMV_OBJ_MM (default 1, op_gemm.h's default and the decode case), and the
     * mxfp4 arms run only in the cell where MM matches it and M <= MM. */
    unsigned objmm = 1;
    if (getenv("PLOW_GEMV_OBJ_MM")) objmm = (unsigned)atoi(getenv("PLOW_GEMV_OBJ_MM"));
    const int mx_ok = (K % 32) == 0; /* a scale block is 32 K-elements; no partial blocks here */
    void *dWq = NULL, *dSq = NULL, *dWq2 = NULL, *dSq2 = NULL;
    unsigned char *hWq = NULL, *hSq = NULL;
    if (mx_ok) {
        /* PINNED, not malloc'd: `plow_hsa_copy_h2d` hands the pointer to the SDMA engine, which
         * faults on pageable memory ("Memory access fault ... Reason: Unknown", async and with
         * no line number). Every other host buffer here is already `plow_hsa_alloc_host`. */
        hWq = plow_hsa_alloc_host(H, (size_t)N * (K / 2));
        hSq = plow_hsa_alloc_host(H, (size_t)N * (K / 32));
        quantise_mxfp4(hWq, hSq, hW, N, K, bf2f);
        dWq = plow_hsa_alloc(H, 0, (size_t)N * (K / 2));
        dSq = plow_hsa_alloc(H, 0, (size_t)N * (K / 32));
        dWq2 = plow_hsa_alloc(H, 0, (size_t)N * (K / 2));
        dSq2 = plow_hsa_alloc(H, 0, (size_t)N * (K / 32));
        plow_hsa_copy_h2d(H, 0, dWq, hWq, (size_t)N * (K / 2));
        plow_hsa_copy_h2d(H, 0, dSq, hSq, (size_t)N * (K / 32));
        plow_hsa_copy_h2d(H, 0, dWq2, hWq, (size_t)N * (K / 2));
        plow_hsa_copy_h2d(H, 0, dSq2, hSq, (size_t)N * (K / 32));
    }

    printf("%s  %u CUs\n", nm, NCU);
    printf("%s  N=%u K=%u   weights %.2f MB/stream   HBM floor %.4f ms\n\n", label, N, K,
           2.0 * (double)N * K / 1e6, 2.0 * (double)N * K / (HBM_GBPS * 1e9) * 1e3);
    printf("  %-16s %3s %3s   %9s %9s %8s %8s\n", "arm", "M", "MM", "ms", "tok/s-eq", "%hbm",
           "check");

    const char* jsonl = getenv("PLOW_GEMV_JSONL");
    FILE* jf = jsonl ? fopen(jsonl, "a") : NULL;

    /* THE M LOOP IS OUTSIDE THE MM LOOP ON PURPOSE. The interesting cell is M != MM — an MM=8
     * object serving M=16 is the §6g-WALK arm — and pairing them the other way round hides it
     * behind "the diagonal". Only M >= MM is run: MM > M is legal (the `m < M` predicate covers
     * it) but it is the same work in a wider accumulator, which is a register question this
     * harness cannot see and `plans/knob-contract.md` §6g-WALK already answers statically. */
    for (int im = 0; im < NMM; im++) {
        const unsigned M = MMS[im];
        /* Weight bytes are M-INVARIANT — one pass streams the whole weight — so per-token cost
         * FALLS with M until something else binds. That is the entire economic case for batched
         * decode and the yardstick every row here is read against. */
        for (int ib = 0; ib < NMM; ib++) {
            const unsigned MM = MMS[ib];
            if (MM > M) continue;
            const unsigned passes = (M + MM - 1) / MM;
            for (int arm = 0; arm < 5; arm++) {
                /* Arms 3/4 are the w4a16 twins of arms 0/1. `mxq` selects the ENCODING, which is
                 * both the quant string in the record and the byte count in the roofline. */
                const int mxq = arm >= 3;
                if (mxq && (!mx_ok || MM != objmm || M > objmm)) continue;
                char sym[48];
                const char* base = arm == 0   ? "gemv_m"
                                   : arm == 1 ? "gemv_glu_m"
                                   : arm == 2 ? "gemv_qkv_m"
                                   : arm == 3 ? "gemv_mxfp4_m"
                                              : "gemv_glu_mxfp4_m";
                /* The record's symbol carries the SCHEMA name (`tunedb::gemv::SYMBOLS`); the
                 * mxfp4 goldens are launched under their own `d_..._k` names because they are
                 * single instantiations rather than a `_m<MM>` family. Keeping the two apart is
                 * what lets the JSONL row key the same way an emitted op will. */
                snprintf(sym, sizeof sym, "%s%u", base, MM);
                const char* launch = arm == 3   ? "d_gemv_mxfp4_k"
                                     : arm == 4 ? "d_gemv_glu_mxfp4_k"
                                                : sym;
                plow_hsa_kernel k;
                if (plow_hsa_get_kernel(H, 0, launch, &k) != 0) continue;

                /* WEIGHT BYTES ARE PER-ARM, and this used to be one number for all three.
                 * The GLU arms read TWO matrices (gate and up) for one N, so a single count
                 * understated their traffic by 2x and printed a %hbm half of the truth. It
                 * matters more now: the whole mxfp4 question is a bytes question, and an
                 * mxfp4 element is a nibble PLUS one E8M0 byte per 32 — 0.53 B/weight against
                 * bf16's 2, so 3.76x and not the 4x a nibble alone would suggest. The ms
                 * column is unaffected either way; only the roofline reading is. */
                const double streams = (arm == 1 || arm == 4) ? 2.0 : 1.0;
                const double bpw = mxq ? (0.5 + 1.0 / 32.0) : 2.0;
                const double wbytes = bpw * (double)N * K * streams * passes;

                /* One packed kernarg per arm. Layouts mirror test_kernels.hip exactly; a
                 * mismatch here reads as a wrong answer, not a fault, so they are written out
                 * rather than shared. */
                struct __attribute__((packed)) {
                    void* c; const void* x; const void* w; const void* rms; const void* gam;
                    unsigned m, n, kk, norm;
                } a0 = {dC, dX, dW, NULL, NULL, M, N, K, 0};
                struct __attribute__((packed)) {
                    void* c; const void* x; const void* wg; const void* wu;
                    unsigned m, n, kk, act;
                } a1 = {dC, dX, dW, dW2, M, N, K, 0};
                struct __attribute__((packed)) {
                    void* cq; void* ck; void* cv; const void* x; const void* wq; const void* wk;
                    const void* wv; unsigned m, nq, nk, nv, kk;
                } a2 = {dCq, dCk, dCv, dX, dW, dW, dW, M, Nq, Nk, Nv, K};
                struct __attribute__((packed)) {
                    void* c; const void* x; const void* w; const void* s; unsigned m, n, kk;
                } a3 = {dC, dX, dWq, dSq, M, N, K};
                struct __attribute__((packed)) {
                    void* c; const void* x; const void* wg; const void* wu; const void* sg;
                    const void* su; unsigned m, n, kk, act;
                } a4 = {dC, dX, dWq, dWq2, dSq, dSq2, M, N, K, 0};
                void* ap = arm == 0   ? (void*)&a0
                           : arm == 1 ? (void*)&a1
                           : arm == 2 ? (void*)&a2
                           : arm == 3 ? (void*)&a3
                                      : (void*)&a4;
                size_t asz = arm == 0   ? sizeof a0
                             : arm == 1 ? sizeof a1
                             : arm == 2 ? sizeof a2
                             : arm == 3 ? sizeof a3
                                        : sizeof a4;

                /* 50 warm-up launches: the governor ramps sclk over tens of ms and an
                 * under-warmed kernel reads slow, which silently re-ranks the sweep. */
                for (int w = 0; w < 50; w++)
                    plow_hsa_launch(H, 0, &k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, ap, asz);
                plow_hsa_wait(H, 0);

                /* Batches of 8, ten times. `tunedb::Stats` needs >= 5 samples and refuses a win
                 * inside the noise, so a single mean is not publishable — it carries no
                 * dispersion. Each sample averages 8 launches because a decode GEMV is tens of
                 * microseconds and per-launch jitter would otherwise swamp it. */
                const int groups = 10, per = 8;
                double sample_ns[16];
                for (int g = 0; g < groups; g++) {
                    const double g0 = now();
                    for (int r = 0; r < per; r++)
                        plow_hsa_launch(H, 0, &k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, ap, asz);
                    plow_hsa_wait(H, 0);
                    sample_ns[g] = (now() - g0) / per * 1e9;
                }
                double sum = 0;
                for (int g = 0; g < groups; g++) sum += sample_ns[g];
                const double dt = sum / groups / 1e9;

                /* CORRECTNESS: poison, launch once, and require BOTH that no output element
                 * still carries the sentinel AND that a sampled element matches an f64 dot.
                 * The plain arm is the only one whose reference is a bare dot product; the two
                 * fused arms carry an epilogue (SwiGLU) or a three-way column split, so for
                 * those the ROW-COVERAGE half is the load-bearing check and the value half is
                 * applied only where the reference is exact (q's columns of the qkv arm, which
                 * are a plain dot against Wq). Under-checking is stated rather than hidden:
                 * `tunedb` will qualify what this marks correct, and a silent "pass" on an
                 * unchecked epilogue is how a fast wrong kernel ships. */
                /* Each output stream is poisoned and checked with ITS OWN row stride. */
                const unsigned NW = arm == 2 ? 3 : 1;
                void* outs[3] = {arm == 2 ? dCq : dC, dCk, dCv};
                unsigned strides[3] = {arm == 2 ? Nq : N, Nk, Nv};
                for (unsigned w = 0; w < NW; w++) {
                    poison(hC, (size_t)M * strides[w]);
                    plow_hsa_copy_h2d(H, 0, outs[w], hC, (size_t)M * strides[w] * 2);
                }
                plow_hsa_launch(H, 0, &k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, ap, asz);
                plow_hsa_wait(H, 0);
                int untouched = 0, bad = 0;
                for (unsigned w = 0; w < NW; w++) {
                    plow_hsa_copy_d2h(H, 0, hC, outs[w], (size_t)M * strides[w] * 2);
                    for (size_t i = 0; i < (size_t)M * strides[w]; i++)
                        if (hC[i] == (bf16)POISON) untouched++;
                    /* EVERY arm gets a value check, and the GLU one is not optional.
                     *
                     * It was, in the first draft, on the reasoning that the SwiGLU epilogue is
                     * awkward to reimplement and row coverage would carry the load. That is
                     * wrong, and the sweep proved it in one run: the LDS-overrun failures show
                     * up as WRONG VALUES in rows that were all written, never as untouched
                     * rows. An unchecked epilogue would have taken a Qualified record on
                     * exactly the cells that are broken.
                     *
                     * Both weight streams are the same buffer here (Wg == Wu == W), so with
                     * `act = PLOW_ACT_GELU_TANH_ = 0` the reference is `gelu_tanh(d) * d` on
                     * the same f64 dot the other arms use. Mirrors `act_gelu_tanh` in
                     * `runtime/amd/op_elementwise.h` exactly. */
                    if (w > 0) continue;
                    for (int s = 0; s < 32; s++) {
                        const unsigned m = (unsigned)(rand() % (int)M);
                        const unsigned nn = (unsigned)(rand() % (int)strides[w]);
                        double acc = 0;
                        /* The mxfp4 arms dot the DECODED fp4 values, not the bf16 originals —
                         * see the encoder's header. Decoding on demand for the sampled row
                         * costs one K-loop and avoids materialising an [N,K] f32 dequant, which
                         * at the lm_head shape would be 4.7 GB. */
                        if (mxq)
                            for (unsigned kk = 0; kk < K; kk++)
                                acc += (double)bf2f(hX[(size_t)m * K + kk]) *
                                       (double)fp4_deq(hWq, hSq, nn, kk, K);
                        else
                            for (unsigned kk = 0; kk < K; kk++)
                                acc += (double)bf2f(hX[(size_t)m * K + kk]) *
                                       bf2f(hW[(size_t)nn * K + kk]);
                        if (arm == 1) {
                            const double c = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc);
                            acc = 0.5 * acc * (1.0 + tanh(c)) * acc;
                        } else if (arm == 4) {
                            /* `act_gate_only(g, 0)` is the same GELU-tanh gate the bf16 GLU arm
                             * takes, and Wg == Wu here, so the reference is gelu(d)*d on one
                             * dot — identical in form to arm 1 and differing only in which
                             * weights produced d. */
                            const double c = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc);
                            acc = 0.5 * acc * (1.0 + tanh(c)) * acc;
                        }
                        const double g = bf2f(hC[(size_t)m * strides[w] + nn]);
                        if (fabs(g - acc) / (fabs(acc) + 1e-3) > 0.03) bad++;
                    }
                }
                const int ok = !untouched && !bad;

                printf("  %-16s %3u %3u   %9.5f %9.1f %7.1f%%  %s\n", base, M, MM, dt * 1e3,
                       (double)M / dt, 100.0 * (wbytes / (HBM_GBPS * 1e9)) / dt,
                       ok ? "ok" : untouched ? "ROWS UNTOUCHED!" : "MISMATCH!");

                if (jf) {
                    /* A FAILING check is written too, marked failed. `tunedb` will not qualify
                     * it, and keeping the negative is the point: an arm that is fast and wrong
                     * must not be silently absent from the record. */
                    fprintf(jf,
                            "{\"m\":%u,\"n\":%u,\"k\":%u,\"quant\":\"%s\",\"mm\":%u,"
                            "\"sym\":\"%s\",\"correct\":%s,\"samples_ns\":[",
                            M, N, K, mxq ? "Mxfp4" : "None", MM, sym, ok ? "true" : "false");
                    for (int g = 0; g < groups; g++)
                        fprintf(jf, "%s%.1f", g ? "," : "", sample_ns[g]);
                    fprintf(jf, "]}\n");
                }
            }
        }
    }
    if (jf) fclose(jf);
    plow_hsa_free(H, dX);
    plow_hsa_free(H, dW);
    plow_hsa_free(H, dW2);
    plow_hsa_free(H, dC);
    plow_hsa_free(H, dCq);
    plow_hsa_free(H, dCk);
    plow_hsa_free(H, dCv);
    if (mx_ok) {
        plow_hsa_free(H, dWq);
        plow_hsa_free(H, dSq);
        plow_hsa_free(H, dWq2);
        plow_hsa_free(H, dSq2);
    }
    return 0;
}
