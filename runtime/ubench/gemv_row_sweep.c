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
        const double wbytes_per_pass = 2.0 * (double)N * K;
        for (int ib = 0; ib < NMM; ib++) {
            const unsigned MM = MMS[ib];
            if (MM > M) continue;
            const unsigned passes = (M + MM - 1) / MM;
            const double wbytes = wbytes_per_pass * passes;
            for (int arm = 0; arm < 3; arm++) {
                char sym[48];
                const char* base = arm == 0 ? "gemv_m" : arm == 1 ? "gemv_glu_m" : "gemv_qkv_m";
                snprintf(sym, sizeof sym, "%s%u", base, MM);
                plow_hsa_kernel k;
                if (plow_hsa_get_kernel(H, 0, sym, &k) != 0) continue;

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
                void* ap = arm == 0 ? (void*)&a0 : arm == 1 ? (void*)&a1 : (void*)&a2;
                size_t asz = arm == 0 ? sizeof a0 : arm == 1 ? sizeof a1 : sizeof a2;

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
                        for (unsigned kk = 0; kk < K; kk++)
                            acc += (double)bf2f(hX[(size_t)m * K + kk]) *
                                   bf2f(hW[(size_t)nn * K + kk]);
                        if (arm == 1) {
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
                            "{\"m\":%u,\"n\":%u,\"k\":%u,\"quant\":\"None\",\"mm\":%u,"
                            "\"sym\":\"%s\",\"correct\":%s,\"samples_ns\":[",
                            M, N, K, MM, sym, ok ? "true" : "false");
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
    return 0;
}
