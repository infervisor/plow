/* moe_gemma_gfx950_test.c — the Gemma-4 26B-A4B MoE op family on AMD, against a CPU reference.
 *
 * Nineteen opcodes (61-77, 81/82) had NO AMD arm at all, so a Gemma-4 26B-A4B packet ran into
 * this interpreter's dispatch `default:`, which WRITES NOTHING — an untouched buffer read as a
 * result. This is the golden that says the new arms compute the model.
 *
 * TRUTH IS THE CPU REFERENCE, and it is written from runtime/nvidia/op_moe.cuh's semantics —
 * the ops' own definition — NOT from runtime/amd/op_moe.h. A reference derived from the kernel
 * validates that the kernel matches itself; a shared misreading passes silently. Every reference
 * below was transcribed from the `_gemma` bodies in the CUDA header, which is why the Gemma
 * router here has a weightless-RMS scalar, a per-channel `scale`, a `root` exponent and a
 * per-expert gate scale rather than the generic router's single `route_scale`.
 *
 * SHAPES ARE THE REAL ONES. hidden 2816, moe_intermediate 704, 128 experts, top-8 — read out of
 * /workspace/models/gemma-4-26B-A4B-it/config.json, not invented. Only the PREFILL grouped-GEMM
 * block narrows the expert count (16), so that a full set of expert weights fits in a test's
 * working set; H and I_moe — the two dimensions the GEMM tile actually sees — stay real.
 *
 * WHAT IS SAMPLED AND WHY. The grouped prefill GEMMs are checked on a RANDOM SUBSET of live
 * gathered rows plus EVERY pad row. A full reference over 1024 gathered rows is ~4 GFLOP of
 * scalar C; the subset is the same arithmetic on the same code path, and the pad rows are
 * checked exhaustively because "did the zero-fill happen" is a structural property, not a
 * numeric one.
 *
 * Needs a gfx942/gfx950 device and `test_kernels.elf` (path as argv[1]).
 *   cc -O2 -std=gnu11 -o t_moe_gemma runtime/tests/moe_gemma_gfx950_test.c \
 *      runtime/amd/hsa_backend.c -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
 * The harness MUST be built with the same PLOW_WG_WAVES as the object (dev_isa.h defaults to 8);
 * a mismatch surfaces as HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS at launch, not as a wrong
 * number.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- real gemma-4-26B-A4B text config ---- */
#define H      2816u  /* text_config.hidden_size        */
#define I_MOE   704u  /* text_config.moe_intermediate_size */
#define N_EXP   128u  /* text_config.num_experts        */
#define TOPK      8u  /* text_config.top_k_experts      */
#define EPS   1e-6f   /* text_config.rms_norm_eps       */

/* Prefill block: real H/I_moe, narrowed expert count so a full weight set fits. */
#define PF_EXP   16u
#define PF_T    128u
#define MPF_BM   64u  /* must match runtime/amd/op_moe.h — the align op pads to it */

typedef unsigned short bf16;

static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

/* OCP e4m3 -> f32, by the spec (bias 7, 0x80 = -0, 0x7f/0xff = NaN). Written here rather than
 * reused from amd_arch.h on purpose: CDNA3's HARDWARE converter reads e4m3FNUZ (bias 8), so the
 * one thing this test must not do is share a decoder with the kernel. If plow's byte convention
 * and the kernel's decode ever disagree, this is what says so. */
static float fp8_ocp_to_f32(unsigned char b) {
    const float sgn = (b & 0x80u) ? -1.0f : 1.0f;
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0u) return sgn * (float)m * 0.001953125f; /* subnormal: m * 2^-9 */
    return sgn * ldexpf(1.0f + (float)m * 0.125f, (int)e - 7);
}

/* ---- deterministic PRNG (identical bits every run) ---- */
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static void seed(uint64_t s) { rs = s ? s : 1; }
static uint32_t r32(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}
static float frand(void) { return ((float)(r32() % 4001u) - 2000.0f) / 8000.0f; } /* +-0.25 */
static float frand_pos(void) { return 0.25f + (float)(r32() % 1000u) / 2000.0f; } /* [0.25,0.75) */

static int fails = 0, checks = 0;

static void report(const char* what, double worst, double tol, size_t at, double g, double w) {
    const int ok = (worst < tol) && !isnan(worst);
    checks++;
    printf("  %-34s %s  (worst rel %.3e, tol %.0e, at %zu)\n", what, ok ? "PASS" : "FAIL", worst,
           tol, at);
    if (!ok) { printf("      got %.6g want %.6g\n", g, w); fails++; }
}

/* NaN-SAFE. `if (rel > worst)` is FALSE for NaN, so a kernel that returns all-NaN scores 0.0 and
 * a naive harness prints PASS — the exact failure mode the fp8 format divergence produced on this
 * part. A NaN here is forced to +inf so it can only fail. */
static double relerr(double g, double w) {
    if (isnan(g) || isnan(w)) return INFINITY;
    return fabs(g - w) / (fabs(w) + 1e-3);
}

static void check_bf16(const char* what, const bf16* got, const float* want, size_t n, double tol) {
    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = relerr(bf2f(got[i]), want[i]);
        if (!(d <= worst)) { worst = d; at = i; }
    }
    report(what, worst, tol, at, n ? bf2f(got[at]) : 0.0, n ? want[at] : 0.0);
}
static void check_f32(const char* what, const float* got, const float* want, size_t n, double tol) {
    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = relerr(got[i], want[i]);
        if (!(d <= worst)) { worst = d; at = i; }
    }
    report(what, worst, tol, at, n ? got[at] : 0.0, n ? want[at] : 0.0);
}
static void check_exact_u32(const char* what, const unsigned* got, const unsigned* want, size_t n) {
    size_t bad = 0, at = 0;
    for (size_t i = 0; i < n; i++) if (got[i] != want[i]) { if (!bad) at = i; bad++; }
    checks++;
    printf("  %-34s %s  (%zu/%zu mismatched)\n", what, bad ? "FAIL" : "PASS", bad, n);
    if (bad) { printf("      [%zu] got %u want %u\n", at, got[at], want[at]); fails++; }
}

static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

/* ================= CPU REFERENCE (from runtime/nvidia/op_moe.cuh) ======================== */

/* d_moe_router_gemma_row steps 1-3: the weightless RMS, the h2 transform, the expert dots. */
static void ref_router_score(const bf16* resid, const bf16* proj, const bf16* scale, unsigned Hn,
                             unsigned n_exp, float root, float eps, float* sc, float* h2) {
    double ss = 0.0;
    for (unsigned h = 0; h < Hn; h++) { const float v = bf2f(resid[h]); ss += (double)v * v; }
    const float inv = 1.0f / sqrtf((float)(ss / (double)Hn) + eps);
    for (unsigned h = 0; h < Hn; h++) h2[h] = bf2f(resid[h]) * inv * bf2f(scale[h]) * root;
    for (unsigned e = 0; e < n_exp; e++) {
        float acc = 0.0f;
        const bf16* pr = proj + (size_t)e * Hn;
        for (unsigned h = 0; h < Hn; h++) acc = fmaf(h2[h], bf2f(pr[h]), acc);
        sc[e] = acc;
    }
}

/* d_moe_router_gemma_row step 4 / d_moe_router_gemma_topk_row: softmax, k-pass masked argmax on
 * the packed key (highest prob, LOWEST id on a tie), norm_topk, per-expert scale. */
static void ref_topk_tail(float* sc, const bf16* pes, unsigned n_exp, unsigned k, unsigned* ids,
                          float* gates) {
    float m = -1e30f;
    for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
    float s = 0.0f;
    for (unsigned e = 0; e < n_exp; e++) { sc[e] = expf(sc[e] - m); s += sc[e]; }
    for (unsigned e = 0; e < n_exp; e++) sc[e] /= s;
    for (unsigned j = 0; j < k; j++) {
        unsigned long long best = 0ull; unsigned bid = 0;
        for (unsigned e = 0; e < n_exp; e++) {
            unsigned sb; const float v = sc[e]; memcpy(&sb, &v, 4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            const unsigned long long key =
                ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
            if (key > best) { best = key; bid = e; }
        }
        ids[j] = bid; gates[j] = sc[bid]; sc[bid] = -1e30f;
    }
    float gs = 0.0f;
    for (unsigned j = 0; j < k; j++) gs += gates[j];
    for (unsigned j = 0; j < k; j++) {
        if (gs != 0.0f) gates[j] /= gs;
        gates[j] *= bf2f(pes[ids[j]]);
    }
}

/* ---- table accessors, the 8-byte {u32 id, f32 gate} slot layout ---- */
static unsigned tbl_id(const unsigned char* t, unsigned slot) {
    unsigned v; memcpy(&v, t + (size_t)slot * 8, 4); return v;
}
static float tbl_gate(const unsigned char* t, unsigned slot) {
    float v; memcpy(&v, t + (size_t)slot * 8 + 4, 4); return v;
}

/* ================= host plumbing ========================================================= */
static plow_hsa* g_h;
static int g_dev = 0;

static void* dev_alloc(size_t n) {
    void* p = plow_hsa_alloc(g_h, g_dev, n);
    if (!p) { fprintf(stderr, "alloc %zu failed: %s\n", n, plow_hsa_last_error()); exit(1); }
    return p;
}
static void up(void* dst, const void* src, size_t n) {
    if (plow_hsa_upload(g_h, g_dev, dst, src, n) != 0) {
        fprintf(stderr, "upload %zu failed: %s\n", n, plow_hsa_last_error()); exit(1);
    }
}
static void down(void* dst, const void* src, size_t n) {
    if (plow_hsa_download(g_h, g_dev, dst, src, n) != 0) {
        fprintf(stderr, "download %zu failed: %s\n", n, plow_hsa_last_error()); exit(1);
    }
}
/* POISON EVERY OUTPUT BEFORE ITS LAUNCH. AMD's dispatch `default:` writes NOTHING — it does not
 * trap the way sm_120 does — so an op with no arm, or a grid map that skips part of its output,
 * leaves the buffer exactly as it found it. A freshly allocated buffer reads as zeros, and zeros
 * are a plausible-looking result. 0x7FC0 is bf16 qNaN and the same word is a NaN read as f32, so
 * one filler covers both output kinds; `relerr` forces NaN to +inf, so anything unwritten FAILS.
 * This is the harness half of the discipline `moe_pf_refuse` follows in the kernels. */
static void poison(void* dst, size_t bytes) {
    static unsigned char* pat = NULL;
    static size_t pat_n = 0;
    if (bytes > pat_n) {
        pat = realloc(pat, bytes + 4);
        for (size_t i = 0; i < (bytes + 3) / 4; i++) ((unsigned*)pat)[i] = 0x7FC07FC0u;
        pat_n = bytes;
    }
    up(dst, pat, bytes);
}
static plow_hsa_kernel kern(const char* name) {
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(g_h, g_dev, name, &k) != 0) {
        fprintf(stderr, "kernel %s not found: %s\n", name, plow_hsa_last_error()); exit(1);
    }
    return k;
}
/* THE ARGUMENT SIZE IS THE PACKED ONE, NOT sizeof(). A C struct's trailing padding is not part
 * of the kernarg segment — `struct { void*[5]; unsigned[3]; }` is 52 bytes of arguments in a
 * 56-byte struct — and the size passed here is LOAD-BEARING TWICE over: `plow_hsa_launch`
 * rejects args_size > kernarg_size ("explicit args 56 B > kernarg segment 52 B"), and it places
 * the COv5 HIDDEN BLOCK at (args_size+7)&~7. Overstate the size and gridDim is written where the
 * kernel does not read it, so `gridDim.x` comes back as garbage and every workgroup strides
 * wrong. ASZ() is the packed size: the offset of the last member plus its width.
 * (`plow_hsa_kernel::kernarg_explicit` cannot help — hsa_backend.c leaves it 0 by design,
 * because the COv5 implicit block is truncated to the fields a kernel actually references.) */
#define ASZ(a, last) (__builtin_offsetof(__typeof__(a), last) + sizeof((a).last))

static void run(const plow_hsa_kernel* k, unsigned nblk, const void* args, size_t n) {
    if (plow_hsa_launch(g_h, g_dev, k, nblk * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                        args, n) != 0) {
        fprintf(stderr, "launch failed: %s\n", plow_hsa_last_error()); exit(1);
    }
    plow_hsa_wait(g_h, g_dev);
}
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
/* Best of `rep` — the box is SHARED, so the minimum is the only stable statistic. */
static double bench(const plow_hsa_kernel* k, unsigned nblk, const void* args, size_t n, int rep) {
    double best = 1e9;
    run(k, nblk, args, n); /* warm */
    for (int i = 0; i < rep; i++) {
        const double t0 = now_ms();
        run(k, nblk, args, n);
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* ---- REAL-SHAPE GROUPED-PREFILL BENCHMARK ----------------------------------------------
 * The correctness block narrows the expert count so a full weight set is cheap; this one does
 * not. n_exp = 128 and T = 512 is a real gemma-4-26B-A4B prefill chunk, 1.5 GB of expert weights,
 * and it is the configuration that exposes what BM=64 padding actually costs at 128 experts:
 * T*k = 4096 live rows spread over 128 experts is ~32 rows each, padded to 64.
 *
 * The weight CONTENT is one generated slab uploaded to every expert. Timing does not depend on
 * the values, and generating 760M distinct bf16 on the host would dominate the run. Correctness
 * is established at the narrow shape, above, where the values ARE distinct per expert. */
static void bench_grouped_real(unsigned NCU, unsigned E, unsigned T, unsigned k) {
    const unsigned NSLOT = T * k;
    const unsigned MAXR = NSLOT + E * (MPF_BM - 1u);
    const size_t GU_N = (size_t)2 * I_MOE * H, DN_N = (size_t)H * I_MOE;

    unsigned char* h_tb = malloc((size_t)NSLOT * 8);
    for (unsigned s = 0; s < NSLOT; s++) {
        const unsigned e = r32() % E;
        const float g = 0.05f + (float)(r32() % 1000u) / 4000.0f;
        memcpy(h_tb + (size_t)s * 8, &e, 4);
        memcpy(h_tb + (size_t)s * 8 + 4, &g, 4);
    }
    void* d_tb = dev_alloc((size_t)NSLOT * 8);
    up(d_tb, h_tb, (size_t)NSLOT * 8);
    void* d_meta = dev_alloc((3 * E + 1) * sizeof(int));
    void* d_rt = dev_alloc((size_t)MAXR * sizeof(unsigned));
    void* d_rp = dev_alloc((size_t)MAXR * sizeof(unsigned));
    void* d_rg = dev_alloc((size_t)MAXR * sizeof(float));
    {
        struct { void* m; const void* t; void* rt; void* rp; void* rg;
                 unsigned T_, E_, K_; } b = { d_meta, d_tb, d_rt, d_rp, d_rg, T, E, k };
        plow_hsa_kernel ka = kern("moe_align_gemma_pf_k");
        run(&ka, 1, &b, ASZ(b, K_));
    }
    int* h_meta = malloc((3 * E + 1) * sizeof(int));
    down(h_meta, d_meta, (3 * E + 1) * sizeof(int));
    const unsigned TOT = (unsigned)h_meta[3 * E] * MPF_BM;

    bf16* xn = malloc((size_t)T * H * sizeof(bf16));
    for (size_t i = 0; i < (size_t)T * H; i++) xn[i] = f2bf(frand());
    void* d_xn = dev_alloc((size_t)T * H * sizeof(bf16));
    up(d_xn, xn, (size_t)T * H * sizeof(bf16));

    bf16* slab_gu = malloc(GU_N * sizeof(bf16));
    bf16* slab_dn = malloc(DN_N * sizeof(bf16));
    for (size_t i = 0; i < GU_N; i++) slab_gu[i] = f2bf(frand());
    for (size_t i = 0; i < DN_N; i++) slab_dn[i] = f2bf(frand());
    unsigned long long* ewt = calloc(E * 2, sizeof(unsigned long long));
    for (unsigned e = 0; e < E; e++) {
        void* a = dev_alloc(GU_N * sizeof(bf16)); up(a, slab_gu, GU_N * sizeof(bf16));
        void* b = dev_alloc(DN_N * sizeof(bf16)); up(b, slab_dn, DN_N * sizeof(bf16));
        ewt[(size_t)e * 2 + 0] = (unsigned long long)(size_t)a;
        ewt[(size_t)e * 2 + 1] = (unsigned long long)(size_t)b;
    }
    void* d_ewt = dev_alloc(E * 2 * sizeof(unsigned long long));
    up(d_ewt, ewt, E * 2 * sizeof(unsigned long long));
    void* d_fug = dev_alloc((size_t)MAXR * I_MOE * sizeof(bf16));
    void* d_part = dev_alloc((size_t)NSLOT * H * sizeof(float));

    const double wbytes = (double)E * (GU_N + DN_N) * 2.0;
    printf("  T=%u n_exp=%u k=%u: %u padded rows (live %u, %.2fx pad), %.2f GB of expert weights\n",
           T, E, k, TOT, NSLOT, (double)TOT / NSLOT, wbytes / 1e9);
    {
        struct { void* fu; const void* x; const void* w; const void* m; const void* rt;
                 unsigned I_, H_, E_, act_; } a = { d_fug, d_xn, d_ewt, d_meta, d_rt, I_MOE, H, E,
                                                    0u };
        plow_hsa_kernel kk = kern("moe_group_glu_gemma_pf_k");
        const double ms = bench(&kk, NCU, &a, ASZ(a, act_), 10);
        const double fl = 2.0 * (double)TOT * I_MOE * 2.0 * H;
        const double by = (double)E * GU_N * 2.0;
        printf("    op75 GROUP_GLU_PF   %8.3f ms  %7.1f TF/s (%4.1f%% of 1307)  %7.1f GB/s (%4.1f%% of 5325)\n",
               ms, fl / (ms * 1e9), 100.0 * fl / (ms * 1e9) / 1307.0, by / (ms * 1e6),
               100.0 * by / (ms * 1e6) / 5325.0);
    }
    {
        struct { void* p; const void* fu; const void* w; const void* m; const void* rp;
                 const void* rg; unsigned H_, I_, E_; } a = { d_part, d_fug, d_ewt, d_meta, d_rp,
                                                              d_rg, H, I_MOE, E };
        plow_hsa_kernel kk = kern("moe_group_down_gemma_pf_k");
        const double ms = bench(&kk, NCU, &a, ASZ(a, E_), 10);
        const double fl = 2.0 * (double)TOT * H * I_MOE;
        const double by = (double)E * DN_N * 2.0;
        printf("    op76 GROUP_DOWN_PF  %8.3f ms  %7.1f TF/s (%4.1f%% of 1307)  %7.1f GB/s (%4.1f%% of 5325)\n",
               ms, fl / (ms * 1e9), 100.0 * fl / (ms * 1e9) / 1307.0, by / (ms * 1e6),
               100.0 * by / (ms * 1e6) / 5325.0);
    }
    free(h_tb); free(h_meta); free(xn); free(slab_gu); free(slab_dn); free(ewt);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    const int do_bench = (argc > 2 && strcmp(argv[2], "--bench") == 0);
    setvbuf(stdout, NULL, _IOLBF, 0); /* a GPU fault must not swallow the log */

    g_h = plow_hsa_init();
    if (!g_h) { fprintf(stderr, "plow_hsa_init: %s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, ldsb = 0;
    plow_hsa_device_info(g_h, g_dev, nm, &cus, &ldsb);
    printf("dev0: %s  CUs=%u  LDS=%u B   wg=%d threads\n", nm, cus, ldsb, PLOW_WG_THREADS);
    printf("shapes: H=%u I_moe=%u n_exp=%u top_k=%u  (gemma-4-26B-A4B-it)\n\n", H, I_MOE, N_EXP,
           TOPK);

    FILE* f = fopen(elf, "rb");
    if (!f) { fprintf(stderr, "%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(g_h, g_dev, co, (size_t)co_n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    const unsigned NCU = cus;

    seed(0xC0FFEEull);

    /* =============================== DECODE: the router ================================ */
    bf16* h_resid = malloc(H * sizeof(bf16));
    bf16* h_proj = malloc((size_t)N_EXP * H * sizeof(bf16));
    bf16* h_scale = malloc(H * sizeof(bf16));
    bf16* h_pes = malloc(N_EXP * sizeof(bf16));
    for (unsigned i = 0; i < H; i++) h_resid[i] = f2bf(frand());
    for (size_t i = 0; i < (size_t)N_EXP * H; i++) h_proj[i] = f2bf(frand());
    for (unsigned i = 0; i < H; i++) h_scale[i] = f2bf(1.0f + frand());
    for (unsigned i = 0; i < N_EXP; i++) h_pes[i] = f2bf(1.0f + frand());
    const float root = 1.0f / sqrtf((float)H);

    void* d_resid = dev_alloc(H * sizeof(bf16));
    void* d_proj = dev_alloc((size_t)N_EXP * H * sizeof(bf16));
    void* d_scale = dev_alloc(H * sizeof(bf16));
    void* d_pes = dev_alloc(N_EXP * sizeof(bf16));
    void* d_table = dev_alloc(TOPK * 8);
    void* d_score = dev_alloc(N_EXP * sizeof(float));
    up(d_resid, h_resid, H * sizeof(bf16));
    up(d_proj, h_proj, (size_t)N_EXP * H * sizeof(bf16));
    up(d_scale, h_scale, H * sizeof(bf16));
    up(d_pes, h_pes, N_EXP * sizeof(bf16));

    float* ref_sc = malloc(N_EXP * sizeof(float));
    float* ref_h2 = malloc(H * sizeof(float));
    float* work = malloc(N_EXP * sizeof(float));
    unsigned ref_ids[TOPK]; float ref_gates[TOPK];
    ref_router_score(h_resid, h_proj, h_scale, H, N_EXP, root, EPS, ref_sc, ref_h2);
    memcpy(work, ref_sc, N_EXP * sizeof(float));
    ref_topk_tail(work, h_pes, N_EXP, TOPK, ref_ids, ref_gates);

    printf("[decode] router\n");
    {
        struct { void* t; const void* r; const void* p; const void* s; const void* pe;
                 unsigned H_, E_, K_, R_; float root_, eps_; } a = {
            d_table, d_resid, d_proj, d_scale, d_pes, H, N_EXP, TOPK, 1u, root, EPS };
        plow_hsa_kernel k = kern("moe_router_gemma_k");
        poison(d_table, TOPK * 8);
        run(&k, 1, &a, ASZ(a, eps_));
        unsigned char tb[TOPK * 8];
        down(tb, d_table, sizeof(tb));
        unsigned got_ids[TOPK]; float got_g[TOPK];
        for (unsigned j = 0; j < TOPK; j++) { got_ids[j] = tbl_id(tb, j); got_g[j] = tbl_gate(tb, j); }
        check_exact_u32("op61 ROUTER_GEMMA ids", got_ids, ref_ids, TOPK);
        check_f32("op61 ROUTER_GEMMA gates", got_g, ref_gates, TOPK, 2e-3);
    }
    {
        struct { void* s; const void* r; const void* p; const void* sc; unsigned H_, E_, R_;
                 float root_, eps_; } a = { d_score, d_resid, d_proj, d_scale, H, N_EXP, 1u,
                                            root, EPS };
        plow_hsa_kernel k = kern("moe_router_gemma_score_k");
        poison(d_score, N_EXP * sizeof(float));
        run(&k, NCU, &a, ASZ(a, eps_));
        float* gs = malloc(N_EXP * sizeof(float));
        down(gs, d_score, N_EXP * sizeof(float));
        /* THE EXACT SCORER IS NOT BIT-EXACT AGAINST THIS REFERENCE, AND THAT IS STRUCTURAL.
         * Its shuffle replay reproduces the legacy scalar dot's fmaf chain element for element,
         * so the DOT contributes nothing; what moves is `invrms`, a PLOW_WAVES-partial block
         * reduction on device against a sequential sum here, and it scales every h2 term. The
         * residual is ~1e-7 absolute on a logit of order 1 — the numbers printed are relative to
         * (|want| + 1e-3), so a near-zero logit shows the largest ratio. What separates op 67
         * from op 69 is visible in the measured column, not in the threshold. */
        check_f32("op67 ROUTER_GEMMA_SCORE", gs, ref_sc, N_EXP, 1e-4);

        plow_hsa_kernel kf = kern("moe_router_gemma_score_fast_k");
        poison(d_score, N_EXP * sizeof(float));
        run(&kf, NCU, &a, ASZ(a, eps_));
        down(gs, d_score, N_EXP * sizeof(float));
        /* The FAST scorer is association-changing by construction; a looser tolerance is the
         * honest one and is what distinguishes it from op 67. */
        check_f32("op69 ROUTER_GEMMA_SCORE_FAST", gs, ref_sc, N_EXP, 1e-3);

        /* op 68 consumes the scores op 67/69 produced — the split router, end to end. */
        up(d_score, ref_sc, N_EXP * sizeof(float));
        struct { void* t; const void* s; const void* pe; unsigned E_, K_, R_; } b = {
            d_table, d_score, d_pes, N_EXP, TOPK, 1u };
        plow_hsa_kernel kt = kern("moe_router_gemma_topk_k");
        poison(d_table, TOPK * 8);
        run(&kt, 1, &b, ASZ(b, R_));
        unsigned char tb[TOPK * 8];
        down(tb, d_table, sizeof(tb));
        unsigned got_ids[TOPK]; float got_g[TOPK];
        for (unsigned j = 0; j < TOPK; j++) { got_ids[j] = tbl_id(tb, j); got_g[j] = tbl_gate(tb, j); }
        check_exact_u32("op68 ROUTER_GEMMA_TOPK ids", got_ids, ref_ids, TOPK);
        check_f32("op68 ROUTER_GEMMA_TOPK gates", got_g, ref_gates, TOPK, 1e-5);
        free(gs);
    }

    /* =============================== DECODE: the experts ================================
     * Only the TOPK routed experts get weights, and `ewt` is zero everywhere else — which is
     * exactly the expert-parallel "not my expert" sentinel the bodies already honour, and it is
     * what makes a real 128-expert decode testable in 95 MB instead of 1.5 GB. */
    printf("\n[decode] experts (top-%u of %u routed; the other %u ewt slots are the EP null)\n",
           TOPK, N_EXP, N_EXP - TOPK);
    const size_t GU_N = (size_t)2 * I_MOE * H;   /* fused gate_up, [2I][H] */
    const size_t DN_N = (size_t)H * I_MOE;       /* down, [H][I]           */
    bf16** h_gu = calloc(N_EXP, sizeof(bf16*));
    bf16** h_dn = calloc(N_EXP, sizeof(bf16*));
    unsigned long long* h_ewt = calloc(N_EXP * 2, sizeof(unsigned long long));
    for (unsigned j = 0; j < TOPK; j++) {
        const unsigned e = ref_ids[j];
        h_gu[e] = malloc(GU_N * sizeof(bf16));
        h_dn[e] = malloc(DN_N * sizeof(bf16));
        for (size_t i = 0; i < GU_N; i++) h_gu[e][i] = f2bf(frand());
        for (size_t i = 0; i < DN_N; i++) h_dn[e][i] = f2bf(frand());
        void* dg = dev_alloc(GU_N * sizeof(bf16));
        void* dd = dev_alloc(DN_N * sizeof(bf16));
        up(dg, h_gu[e], GU_N * sizeof(bf16));
        up(dd, h_dn[e], DN_N * sizeof(bf16));
        h_ewt[(size_t)e * 2 + 0] = (unsigned long long)(size_t)dg;
        h_ewt[(size_t)e * 2 + 1] = (unsigned long long)(size_t)dd;
    }
    void* d_ewt = dev_alloc(N_EXP * 2 * sizeof(unsigned long long));
    up(d_ewt, h_ewt, N_EXP * 2 * sizeof(unsigned long long));

    /* Put the reference table on device so the expert ops read a table we control exactly. */
    unsigned char h_tab[TOPK * 8];
    for (unsigned j = 0; j < TOPK; j++) {
        memcpy(h_tab + (size_t)j * 8, &ref_ids[j], 4);
        memcpy(h_tab + (size_t)j * 8 + 4, &ref_gates[j], 4);
    }
    up(d_table, h_tab, sizeof(h_tab));

    bf16* h_x = malloc(H * sizeof(bf16));
    for (unsigned i = 0; i < H; i++) h_x[i] = f2bf(frand());
    void* d_x = dev_alloc(H * sizeof(bf16));
    up(d_x, h_x, H * sizeof(bf16));

    const size_t FU_N = (size_t)TOPK * I_MOE;
    const size_t PART_N = (size_t)TOPK * H;
    void* d_fu = dev_alloc(FU_N * sizeof(bf16));
    void* d_part = dev_alloc(PART_N * sizeof(float));
    bf16* g_fu = malloc(FU_N * sizeof(bf16));
    float* r_fu = malloc(FU_N * sizeof(float));
    float* g_part = malloc(PART_N * sizeof(float));
    float* r_part = malloc(PART_N * sizeof(float));

    plow_hsa_kernel k_glu = kern("moe_expert_glu_gemma_k");
    plow_hsa_kernel k_down = kern("moe_expert_down_gemma_k");
    {
        struct { void* fu; const void* x; const void* t; const void* w;
                 unsigned k_, I_, H_, E_, R_; } a = { d_fu, d_x, d_table, d_ewt, TOPK, I_MOE, H,
                                                      N_EXP, 1u };
        poison(d_fu, FU_N * sizeof(bf16));
        run(&k_glu, NCU, &a, ASZ(a, R_));
        down(g_fu, d_fu, FU_N * sizeof(bf16));
        for (unsigned s = 0; s < TOPK; s++) {
            const bf16* gu = h_gu[ref_ids[s]];
            for (unsigned n = 0; n < I_MOE; n++) {
                float gg = 0.0f, uu = 0.0f;
                const bf16* gr = gu + (size_t)n * H;
                const bf16* ur = gu + (size_t)(I_MOE + n) * H;
                for (unsigned hh = 0; hh < H; hh++) {
                    const float xv = bf2f(h_x[hh]);
                    gg = fmaf(xv, bf2f(gr[hh]), gg);
                    uu = fmaf(xv, bf2f(ur[hh]), uu);
                }
                r_fu[(size_t)s * I_MOE + n] = bf2f(f2bf(gelu_tanh(gg) * uu));
            }
        }
        check_bf16("op62 EXPERT_GLU_GEMMA", g_fu, r_fu, FU_N, 2e-2);
    }
    {
        /* DOWN reads the fu the GLU op just produced — the real chain, not a fresh random. */
        struct { void* p; const void* fu; const void* t; const void* w;
                 unsigned k_, H_, I_, E_, R_; } a = { d_part, d_fu, d_table, d_ewt, TOPK, H,
                                                      I_MOE, N_EXP, 1u };
        poison(d_part, PART_N * sizeof(float));
        run(&k_down, NCU, &a, ASZ(a, R_));
        down(g_part, d_part, PART_N * sizeof(float));
        for (unsigned s = 0; s < TOPK; s++) {
            const bf16* dn = h_dn[ref_ids[s]];
            const bf16* fs = g_fu + (size_t)s * I_MOE;
            for (unsigned hh = 0; hh < H; hh++) {
                float acc = 0.0f;
                const bf16* wr = dn + (size_t)hh * I_MOE;
                for (unsigned i = 0; i < I_MOE; i++) acc = fmaf(bf2f(fs[i]), bf2f(wr[i]), acc);
                r_part[(size_t)s * H + hh] = ref_gates[s] * acc;
            }
        }
        check_f32("op63 EXPERT_DOWN_GEMMA", g_part, r_part, PART_N, 5e-3);
    }

    /* op 71: the same gate/up with the pre-FFN RMSNorm fused in. */
    {
        bf16* h_gam = malloc(H * sizeof(bf16));
        for (unsigned i = 0; i < H; i++) h_gam[i] = f2bf(1.0f + frand());
        void* d_gam = dev_alloc(H * sizeof(bf16));
        up(d_gam, h_gam, H * sizeof(bf16));
        struct { void* fu; const void* r; const void* g; const void* t; const void* w;
                 unsigned k_, I_, H_, E_, R_; float eps_; } a = {
            d_fu, d_resid, d_gam, d_table, d_ewt, TOPK, I_MOE, H, N_EXP, 1u, EPS };
        plow_hsa_kernel k = kern("moe_expert_glu_norm_gemma_k");
        poison(d_fu, FU_N * sizeof(bf16));
        run(&k, NCU, &a, ASZ(a, eps_));
        bf16* gv = malloc(FU_N * sizeof(bf16));
        down(gv, d_fu, FU_N * sizeof(bf16));
        double ss = 0.0;
        for (unsigned hh = 0; hh < H; hh++) { const float v = bf2f(h_resid[hh]); ss += (double)v * v; }
        const float inv = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
        float* xn = malloc(H * sizeof(float));
        for (unsigned hh = 0; hh < H; hh++) xn[hh] = bf2f(h_resid[hh]) * inv * bf2f(h_gam[hh]);
        for (unsigned s = 0; s < TOPK; s++) {
            const bf16* gu = h_gu[ref_ids[s]];
            for (unsigned n = 0; n < I_MOE; n++) {
                float gg = 0.0f, uu = 0.0f;
                const bf16* gr = gu + (size_t)n * H;
                const bf16* ur = gu + (size_t)(I_MOE + n) * H;
                for (unsigned hh = 0; hh < H; hh++) {
                    gg = fmaf(xn[hh], bf2f(gr[hh]), gg);
                    uu = fmaf(xn[hh], bf2f(ur[hh]), uu);
                }
                r_fu[(size_t)s * I_MOE + n] = bf2f(f2bf(gelu_tanh(gg) * uu));
            }
        }
        /* Same tolerance as op 62: the kernel stages xn in f32 (see the note on GMOE_XN_MAX_F
         * in op_moe.h — bf16 staging measured an 8x relative error here, because act(g)*u is a
         * cancellation wherever act(g) is near zero). */
        check_bf16("op71 EXPERT_GLU_NORM_GEMMA", gv, r_fu, FU_N, 2e-2);
        free(gv); free(xn); free(h_gam);
    }

    /* ============================ DECODE: the combines =============================== */
    printf("\n[decode] combines\n");
    {
        bf16* h_gam = malloc(H * sizeof(bf16));
        for (unsigned i = 0; i < H; i++) h_gam[i] = f2bf(1.0f + frand());
        void* d_gam = dev_alloc(H * sizeof(bf16));
        void* d_out = dev_alloc(H * sizeof(bf16));
        up(d_gam, h_gam, H * sizeof(bf16));
        up(d_part, r_part, PART_N * sizeof(float)); /* a known f32 partial set */

        float* r_sum = malloc(H * sizeof(float));
        for (unsigned hh = 0; hh < H; hh++) {
            float acc = 0.0f;
            for (unsigned s = 0; s < TOPK; s++) acc += r_part[(size_t)s * H + hh];
            r_sum[hh] = acc;
        }
        {
            struct { void* o; const void* p; unsigned H_, k_; } a = { d_out, d_part, H, TOPK };
            plow_hsa_kernel k = kern("moe_combine_gemma_k");
            poison(d_out, H * sizeof(bf16));
            run(&k, NCU, &a, ASZ(a, k_));
            bf16* gv = malloc(H * sizeof(bf16));
            down(gv, d_out, H * sizeof(bf16));
            check_bf16("op64 COMBINE_GEMMA", gv, r_sum, H, 1e-2);
            free(gv);
        }
        {
            struct { void* o; const void* p; const void* r; const void* g;
                     unsigned H_, k_, R_; float eps_; } a = { d_out, d_part, d_resid, d_gam, H,
                                                              TOPK, 1u, EPS };
            plow_hsa_kernel k = kern("moe_combine_norm_gemma_k");
            poison(d_out, H * sizeof(bf16));
            run(&k, NCU, &a, ASZ(a, eps_));
            bf16* gv = malloc(H * sizeof(bf16));
            down(gv, d_out, H * sizeof(bf16));
            double ss = 0.0;
            for (unsigned hh = 0; hh < H; hh++) ss += (double)r_sum[hh] * r_sum[hh];
            const float inv = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
            float* want = malloc(H * sizeof(float));
            for (unsigned hh = 0; hh < H; hh++)
                want[hh] = r_sum[hh] * inv * bf2f(h_gam[hh]) + bf2f(h_resid[hh]);
            check_bf16("op70 COMBINE_NORM_GEMMA", gv, want, H, 1e-2);
            free(gv); free(want);
        }
        {
            /* op 72: the whole layer tail. b/x are ROUNDED to bf16 between reductions, which is
             * what makes it bit-exact to the op pair it replaces — the reference does the same. */
            bf16* h_h1 = malloc(H * sizeof(bf16));
            bf16* h_gpo = malloc(H * sizeof(bf16));
            bf16* h_gn = malloc(H * sizeof(bf16));
            bf16* h_xr = malloc(H * sizeof(bf16));
            for (unsigned i = 0; i < H; i++) {
                h_h1[i] = f2bf(frand()); h_gpo[i] = f2bf(1.0f + frand());
                h_gn[i] = f2bf(1.0f + frand()); h_xr[i] = f2bf(frand());
            }
            void* d_h1 = dev_alloc(H * sizeof(bf16));
            void* d_gpo = dev_alloc(H * sizeof(bf16));
            void* d_gn = dev_alloc(H * sizeof(bf16));
            void* d_xr = dev_alloc(H * sizeof(bf16));
            void* d_hn = dev_alloc(H * sizeof(bf16));
            up(d_h1, h_h1, H * sizeof(bf16)); up(d_gpo, h_gpo, H * sizeof(bf16));
            up(d_gn, h_gn, H * sizeof(bf16)); up(d_xr, h_xr, H * sizeof(bf16));
            const float ls = 0.5f;
            struct { void* hn; void* x; const void* p; const void* h1; const void* g2;
                     const void* go; const void* gn_; unsigned H_, k_; float eps_, ls_; } a = {
                d_hn, d_xr, d_part, d_h1, d_gam, d_gpo, d_gn, H, TOPK, EPS, ls };
            plow_hsa_kernel k = kern("moe_combine_resid_norm_gemma_k");
            poison(d_hn, H * sizeof(bf16));
            run(&k, 1, &a, ASZ(a, ls_));
            bf16* g_hn = malloc(H * sizeof(bf16));
            bf16* g_x = malloc(H * sizeof(bf16));
            down(g_hn, d_hn, H * sizeof(bf16));
            down(g_x, d_xr, H * sizeof(bf16));

            double s1 = 0.0;
            for (unsigned hh = 0; hh < H; hh++) s1 += (double)r_sum[hh] * r_sum[hh];
            const float i1 = 1.0f / sqrtf((float)(s1 / (double)H) + EPS);
            float* b = malloc(H * sizeof(float));
            double s2 = 0.0;
            for (unsigned hh = 0; hh < H; hh++) {
                b[hh] = bf2f(f2bf(r_sum[hh] * i1 * bf2f(h_gam[hh]) + bf2f(h_h1[hh])));
                s2 += (double)b[hh] * b[hh];
            }
            const float i2 = 1.0f / sqrtf((float)(s2 / (double)H) + EPS);
            float* rr = malloc(H * sizeof(float));
            double s3 = 0.0;
            for (unsigned hh = 0; hh < H; hh++) {
                rr[hh] = bf2f(f2bf((bf2f(h_xr[hh]) + b[hh] * i2 * bf2f(h_gpo[hh])) * ls));
                s3 += (double)rr[hh] * rr[hh];
            }
            const float i3 = 1.0f / sqrtf((float)(s3 / (double)H) + EPS);
            float* want_hn = malloc(H * sizeof(float));
            for (unsigned hh = 0; hh < H; hh++) want_hn[hh] = rr[hh] * i3 * bf2f(h_gn[hh]);
            check_bf16("op72 CRN_GEMMA x(resid)", g_x, rr, H, 1e-2);
            check_bf16("op72 CRN_GEMMA hn", g_hn, want_hn, H, 1e-2);
            free(g_hn); free(g_x); free(b); free(rr); free(want_hn);
            free(h_h1); free(h_gpo); free(h_gn); free(h_xr);
        }
        free(r_sum); free(h_gam);
    }

    /* ============================ DECODE: fp8 experts ================================
     * The weight BYTES are generated directly (never encoded from f32 on the host), so this
     * checks plow's OCP e4m3 convention against the kernel's decode with nothing shared between
     * them. 0x7f/0xff are excluded: they are OCP NaN, and amd_arch.h documents that the fast
     * CDNA3 decode maps them to 480 rather than NaN. */
    printf("\n[decode] fp8 experts (per-output-channel e4m3)\n");
    {
        unsigned char** h_gu8 = calloc(N_EXP, sizeof(unsigned char*));
        unsigned char** h_dn8 = calloc(N_EXP, sizeof(unsigned char*));
        float** h_sgu = calloc(N_EXP, sizeof(float*));
        float** h_sdn = calloc(N_EXP, sizeof(float*));
        unsigned long long* h_w8 = calloc(N_EXP * 2, sizeof(unsigned long long));
        unsigned long long* h_s8 = calloc(N_EXP * 2, sizeof(unsigned long long));
        for (unsigned j = 0; j < TOPK; j++) {
            const unsigned e = ref_ids[j];
            h_gu8[e] = malloc(GU_N);
            h_dn8[e] = malloc(DN_N);
            for (size_t i = 0; i < GU_N; i++) {
                unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                h_gu8[e][i] = b;
            }
            for (size_t i = 0; i < DN_N; i++) {
                unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                h_dn8[e][i] = b;
            }
            h_sgu[e] = malloc(2 * I_MOE * sizeof(float));
            h_sdn[e] = malloc(H * sizeof(float));
            for (unsigned i = 0; i < 2 * I_MOE; i++) h_sgu[e][i] = frand_pos() * 0.01f;
            for (unsigned i = 0; i < H; i++) h_sdn[e][i] = frand_pos() * 0.01f;
            void* dg = dev_alloc(GU_N); up(dg, h_gu8[e], GU_N);
            void* dd = dev_alloc(DN_N); up(dd, h_dn8[e], DN_N);
            void* sg = dev_alloc(2 * I_MOE * sizeof(float));
            up(sg, h_sgu[e], 2 * I_MOE * sizeof(float));
            void* sd = dev_alloc(H * sizeof(float));
            up(sd, h_sdn[e], H * sizeof(float));
            h_w8[(size_t)e * 2 + 0] = (unsigned long long)(size_t)dg;
            h_w8[(size_t)e * 2 + 1] = (unsigned long long)(size_t)dd;
            h_s8[(size_t)e * 2 + 0] = (unsigned long long)(size_t)sg;
            h_s8[(size_t)e * 2 + 1] = (unsigned long long)(size_t)sd;
        }
        void* d_w8 = dev_alloc(N_EXP * 2 * sizeof(unsigned long long));
        void* d_s8 = dev_alloc(N_EXP * 2 * sizeof(unsigned long long));
        up(d_w8, h_w8, N_EXP * 2 * sizeof(unsigned long long));
        up(d_s8, h_s8, N_EXP * 2 * sizeof(unsigned long long));
        {
            struct { void* fu; const void* x; const void* t; const void* w; const void* s;
                     unsigned k_, I_, H_, E_, R_; } a = { d_fu, d_x, d_table, d_w8, d_s8, TOPK,
                                                          I_MOE, H, N_EXP, 1u };
            plow_hsa_kernel k = kern("moe_expert_glu_gemma_fp8_k");
            poison(d_fu, FU_N * sizeof(bf16));
            run(&k, NCU, &a, ASZ(a, R_));
            down(g_fu, d_fu, FU_N * sizeof(bf16));
            for (unsigned s = 0; s < TOPK; s++) {
                const unsigned e = ref_ids[s];
                for (unsigned n = 0; n < I_MOE; n++) {
                    float gg = 0.0f, uu = 0.0f;
                    const unsigned char* gr = h_gu8[e] + (size_t)n * H;
                    const unsigned char* ur = h_gu8[e] + (size_t)(I_MOE + n) * H;
                    for (unsigned hh = 0; hh < H; hh++) {
                        const float xv = bf2f(h_x[hh]);
                        gg = fmaf(xv, fp8_ocp_to_f32(gr[hh]), gg);
                        uu = fmaf(xv, fp8_ocp_to_f32(ur[hh]), uu);
                    }
                    gg *= h_sgu[e][n];
                    uu *= h_sgu[e][I_MOE + n];
                    r_fu[(size_t)s * I_MOE + n] = bf2f(f2bf(gelu_tanh(gg) * uu));
                }
            }
            check_bf16("op65 EXPERT_GLU_GEMMA_FP8", g_fu, r_fu, FU_N, 2e-2);
        }
        {
            struct { void* p; const void* fu; const void* t; const void* w; const void* s;
                     unsigned k_, H_, I_, E_, R_; } a = { d_part, d_fu, d_table, d_w8, d_s8, TOPK,
                                                          H, I_MOE, N_EXP, 1u };
            plow_hsa_kernel k = kern("moe_expert_down_gemma_fp8_k");
            poison(d_part, PART_N * sizeof(float));
            run(&k, NCU, &a, ASZ(a, R_));
            down(g_part, d_part, PART_N * sizeof(float));
            for (unsigned s = 0; s < TOPK; s++) {
                const unsigned e = ref_ids[s];
                const bf16* fs = g_fu + (size_t)s * I_MOE;
                for (unsigned hh = 0; hh < H; hh++) {
                    float acc = 0.0f;
                    const unsigned char* wr = h_dn8[e] + (size_t)hh * I_MOE;
                    for (unsigned i = 0; i < I_MOE; i++)
                        acc = fmaf(bf2f(fs[i]), fp8_ocp_to_f32(wr[i]), acc);
                    r_part[(size_t)s * H + hh] = ref_gates[s] * (acc * h_sdn[e][hh]);
                }
            }
            check_f32("op66 EXPERT_DOWN_GEMMA_FP8", g_part, r_part, PART_N, 5e-3);
        }
    }

    /* =============================== PREFILL ========================================= */
    printf("\n[prefill] router / align at the REAL expert count (T=%u, n_exp=%u)\n", 32u, N_EXP);
    {
        const unsigned T = 32;
        bf16* h_r = malloc((size_t)T * H * sizeof(bf16));
        for (size_t i = 0; i < (size_t)T * H; i++) h_r[i] = f2bf(frand());
        void* d_r = dev_alloc((size_t)T * H * sizeof(bf16));
        void* d_tb = dev_alloc((size_t)T * TOPK * 8);
        up(d_r, h_r, (size_t)T * H * sizeof(bf16));
        struct { void* t; const void* r; const void* p; const void* s; const void* pe;
                 unsigned H_, E_, K_, T_; float root_, eps_; } a = {
            d_tb, d_r, d_proj, d_scale, d_pes, H, N_EXP, TOPK, T, root, EPS };
        plow_hsa_kernel k = kern("moe_router_gemma_pf_k");
        poison(d_tb, (size_t)T * TOPK * 8);
        run(&k, NCU, &a, ASZ(a, eps_));
        unsigned char* gtb = malloc((size_t)T * TOPK * 8);
        down(gtb, d_tb, (size_t)T * TOPK * 8);
        unsigned* gi = malloc((size_t)T * TOPK * sizeof(unsigned));
        unsigned* ri = malloc((size_t)T * TOPK * sizeof(unsigned));
        float* gg = malloc((size_t)T * TOPK * sizeof(float));
        float* rg = malloc((size_t)T * TOPK * sizeof(float));
        for (unsigned t = 0; t < T; t++) {
            ref_router_score(h_r + (size_t)t * H, h_proj, h_scale, H, N_EXP, root, EPS, ref_sc,
                             ref_h2);
            ref_topk_tail(ref_sc, h_pes, N_EXP, TOPK, ri + (size_t)t * TOPK,
                          rg + (size_t)t * TOPK);
            for (unsigned j = 0; j < TOPK; j++) {
                gi[(size_t)t * TOPK + j] = tbl_id(gtb, t * TOPK + j);
                gg[(size_t)t * TOPK + j] = tbl_gate(gtb, t * TOPK + j);
            }
        }
        check_exact_u32("op73 ROUTER_GEMMA_PF ids", gi, ri, (size_t)T * TOPK);
        check_f32("op73 ROUTER_GEMMA_PF gates", gg, rg, (size_t)T * TOPK, 2e-3);

        /* op 74 on that very table. The scatter uses an LDS atomic cursor, so the ROW ORDER
         * inside an expert is not reproducible — the check is on the meta (exact), on the pad
         * rows (exact), and on the MULTISET of live (token, partidx, gate) triples. */
        const unsigned MAXR = T * TOPK + N_EXP * (MPF_BM - 1u);
        void* d_meta = dev_alloc((3 * N_EXP + 1) * sizeof(int));
        void* d_rt = dev_alloc((size_t)MAXR * sizeof(unsigned));
        void* d_rp = dev_alloc((size_t)MAXR * sizeof(unsigned));
        void* d_rgt = dev_alloc((size_t)MAXR * sizeof(float));
        struct { void* m; const void* t; void* rt; void* rp; void* rg;
                 unsigned T_, E_, K_; } b = { d_meta, d_tb, d_rt, d_rp, d_rgt, T, N_EXP, TOPK };
        plow_hsa_kernel ka = kern("moe_align_gemma_pf_k");
        poison(d_meta, (3 * N_EXP + 1) * sizeof(int));
        run(&ka, 1, &b, ASZ(b, K_));
        int* gm = malloc((3 * N_EXP + 1) * sizeof(int));
        unsigned* grt = malloc((size_t)MAXR * sizeof(unsigned));
        unsigned* grp = malloc((size_t)MAXR * sizeof(unsigned));
        float* grg = malloc((size_t)MAXR * sizeof(float));
        down(gm, d_meta, (3 * N_EXP + 1) * sizeof(int));
        down(grt, d_rt, (size_t)MAXR * sizeof(unsigned));
        down(grp, d_rp, (size_t)MAXR * sizeof(unsigned));
        down(grg, d_rgt, (size_t)MAXR * sizeof(float));

        int* rm = calloc(3 * N_EXP + 1, sizeof(int));
        unsigned* cnt = calloc(N_EXP, sizeof(unsigned));
        for (unsigned s = 0; s < T * TOPK; s++) { const unsigned e = tbl_id(gtb, s); if (e < N_EXP) cnt[e]++; }
        unsigned tp = 0;
        for (unsigned e = 0; e < N_EXP; e++) {
            rm[2 * N_EXP + e] = (int)tp;
            rm[e] = (int)(tp * MPF_BM);
            rm[N_EXP + e] = (int)cnt[e];
            tp += (cnt[e] + MPF_BM - 1u) / MPF_BM;
        }
        rm[3 * N_EXP] = (int)tp;
        check_exact_u32("op74 ALIGN_GEMMA_PF meta", (unsigned*)gm, (unsigned*)rm, 3 * N_EXP + 1);

        /* per-expert multiset + pad check */
        size_t bad = 0;
        const unsigned total_pad = tp * MPF_BM;
        for (unsigned e = 0; e < N_EXP; e++) {
            const unsigned base = (unsigned)rm[e], c = cnt[e];
            const unsigned tiles = (c + MPF_BM - 1u) / MPF_BM;
            for (unsigned r = base + c; r < base + tiles * MPF_BM; r++)
                if (grp[r] != PLOW_EXPERT_UNUSED || grt[r] != PLOW_EXPERT_UNUSED) bad++;
            for (unsigned r = base; r < base + c; r++) {
                const unsigned pidx = grp[r];
                if (pidx == PLOW_EXPERT_UNUSED || pidx >= T * TOPK) { bad++; continue; }
                if (tbl_id(gtb, pidx) != e) bad++;
                if (grt[r] != pidx / TOPK) bad++;
                if (grg[r] != tbl_gate(gtb, pidx)) bad++;
            }
        }
        /* every live slot must appear exactly once */
        unsigned char* seen = calloc(T * TOPK, 1);
        for (unsigned r = 0; r < total_pad; r++)
            if (grp[r] != PLOW_EXPERT_UNUSED) { if (seen[grp[r]]++) bad++; }
        for (unsigned s = 0; s < T * TOPK; s++) if (!seen[s]) bad++;
        checks++;
        printf("  %-34s %s  (%zu structural violations over %u padded rows)\n",
               "op74 ALIGN_GEMMA_PF scatter", bad ? "FAIL" : "PASS", bad, total_pad);
        if (bad) fails++;
        free(h_r); free(gtb); free(gi); free(ri); free(gg); free(rg);
        free(gm); free(grt); free(grp); free(grg); free(rm); free(cnt); free(seen);
    }

    /* ---- grouped expert GEMM (ops 75/76 + 81/82) and the T-row combine (op 77) ---- */
    printf("\n[prefill] grouped expert GEMM (T=%u, n_exp=%u, H=%u, I_moe=%u)\n", PF_T, PF_EXP, H,
           I_MOE);
    {
        const unsigned T = PF_T, E = PF_EXP, k = TOPK;
        const unsigned NSLOT = T * k;
        const unsigned MAXR = NSLOT + E * (MPF_BM - 1u);

        /* a synthetic routing table over E experts */
        unsigned char* h_tb = malloc((size_t)NSLOT * 8);
        for (unsigned s = 0; s < NSLOT; s++) {
            const unsigned e = r32() % E;
            const float g = 0.05f + (float)(r32() % 1000u) / 4000.0f;
            memcpy(h_tb + (size_t)s * 8, &e, 4);
            memcpy(h_tb + (size_t)s * 8 + 4, &g, 4);
        }
        void* d_tb = dev_alloc((size_t)NSLOT * 8);
        up(d_tb, h_tb, (size_t)NSLOT * 8);

        void* d_meta = dev_alloc((3 * E + 1) * sizeof(int));
        void* d_rt = dev_alloc((size_t)MAXR * sizeof(unsigned));
        void* d_rp = dev_alloc((size_t)MAXR * sizeof(unsigned));
        void* d_rg = dev_alloc((size_t)MAXR * sizeof(float));
        {
            struct { void* m; const void* t; void* rt; void* rp; void* rg;
                     unsigned T_, E_, K_; } b = { d_meta, d_tb, d_rt, d_rp, d_rg, T, E, k };
            plow_hsa_kernel ka = kern("moe_align_gemma_pf_k");
            run(&ka, 1, &b, ASZ(b, K_));
        }
        int* h_meta = malloc((3 * E + 1) * sizeof(int));
        unsigned* h_rt = malloc((size_t)MAXR * sizeof(unsigned));
        unsigned* h_rp = malloc((size_t)MAXR * sizeof(unsigned));
        float* h_rg = malloc((size_t)MAXR * sizeof(float));
        down(h_meta, d_meta, (3 * E + 1) * sizeof(int));
        down(h_rt, d_rt, (size_t)MAXR * sizeof(unsigned));
        down(h_rp, d_rp, (size_t)MAXR * sizeof(unsigned));
        down(h_rg, d_rg, (size_t)MAXR * sizeof(float));
        const unsigned TOT = (unsigned)h_meta[3 * E] * MPF_BM;
        printf("  padded gathered rows: %u (live %u, %.1fx pad at BM=%u)\n", TOT, NSLOT,
               (double)TOT / NSLOT, MPF_BM);

        /* activations + weights */
        bf16* h_xn = malloc((size_t)T * H * sizeof(bf16));
        for (size_t i = 0; i < (size_t)T * H; i++) h_xn[i] = f2bf(frand());
        void* d_xn = dev_alloc((size_t)T * H * sizeof(bf16));
        up(d_xn, h_xn, (size_t)T * H * sizeof(bf16));

        bf16** gu = calloc(E, sizeof(bf16*));
        bf16** dn = calloc(E, sizeof(bf16*));
        unsigned long long* ewt = calloc(E * 2, sizeof(unsigned long long));
        for (unsigned e = 0; e < E; e++) {
            gu[e] = malloc(GU_N * sizeof(bf16));
            dn[e] = malloc(DN_N * sizeof(bf16));
            for (size_t i = 0; i < GU_N; i++) gu[e][i] = f2bf(frand());
            for (size_t i = 0; i < DN_N; i++) dn[e][i] = f2bf(frand());
            void* a = dev_alloc(GU_N * sizeof(bf16)); up(a, gu[e], GU_N * sizeof(bf16));
            void* b = dev_alloc(DN_N * sizeof(bf16)); up(b, dn[e], DN_N * sizeof(bf16));
            ewt[(size_t)e * 2 + 0] = (unsigned long long)(size_t)a;
            ewt[(size_t)e * 2 + 1] = (unsigned long long)(size_t)b;
        }
        void* d_ewt2 = dev_alloc(E * 2 * sizeof(unsigned long long));
        up(d_ewt2, ewt, E * 2 * sizeof(unsigned long long));

        void* d_fug = dev_alloc((size_t)MAXR * I_MOE * sizeof(bf16));
        void* d_partg = dev_alloc((size_t)NSLOT * H * sizeof(float));

        /* which expert owns gathered row r (from the meta this very align wrote) */
        unsigned* row_exp = malloc((size_t)TOT * sizeof(unsigned));
        for (unsigned e = 0; e < E; e++) {
            const unsigned base = (unsigned)h_meta[e];
            const unsigned tiles = (unsigned)(h_meta[2 * E + e + 1] - h_meta[2 * E + e]);
            for (unsigned r = base; r < base + tiles * MPF_BM; r++) row_exp[r] = e;
        }

        /* --- op 75 --- */
        bf16* g_fug = malloc((size_t)TOT * I_MOE * sizeof(bf16));
        {
            struct { void* fu; const void* x; const void* w; const void* m; const void* rt;
                     unsigned I_, H_, E_, act_; } a = { d_fug, d_xn, d_ewt2, d_meta, d_rt, I_MOE,
                                                        H, E, 0u };
            plow_hsa_kernel kk = kern("moe_group_glu_gemma_pf_k");
            poison(d_fug, (size_t)TOT * I_MOE * sizeof(bf16));
            run(&kk, NCU, &a, ASZ(a, act_));
            down(g_fug, d_fug, (size_t)TOT * I_MOE * sizeof(bf16));

            /* pad rows must be exactly zero — a structural property, checked exhaustively */
            size_t padbad = 0, padrows = 0;
            for (unsigned r = 0; r < TOT; r++) {
                if (h_rt[r] != PLOW_EXPERT_UNUSED) continue;
                padrows++;
                for (unsigned n = 0; n < I_MOE; n++)
                    if (g_fug[(size_t)r * I_MOE + n] != 0) { padbad++; break; }
            }
            checks++;
            printf("  %-34s %s  (%zu of %zu pad rows non-zero)\n", "op75 GROUP_GLU pad rows",
                   padbad ? "FAIL" : "PASS", padbad, padrows);
            if (padbad) fails++;

            /* a random subset of live rows, in full */
            const unsigned NS = 48;
            float* want = malloc((size_t)NS * I_MOE * sizeof(float));
            bf16* got = malloc((size_t)NS * I_MOE * sizeof(bf16));
            for (unsigned q = 0; q < NS; q++) {
                unsigned r;
                do { r = r32() % TOT; } while (h_rt[r] == PLOW_EXPERT_UNUSED);
                const bf16* x = h_xn + (size_t)h_rt[r] * H;
                const bf16* w = gu[row_exp[r]];
                for (unsigned n = 0; n < I_MOE; n++) {
                    float g = 0.0f, u = 0.0f;
                    const bf16* gr = w + (size_t)n * H;
                    const bf16* ur = w + (size_t)(I_MOE + n) * H;
                    for (unsigned hh = 0; hh < H; hh++) {
                        const float xv = bf2f(x[hh]);
                        g = fmaf(xv, bf2f(gr[hh]), g);
                        u = fmaf(xv, bf2f(ur[hh]), u);
                    }
                    want[(size_t)q * I_MOE + n] = gelu_tanh(g) * u;
                    got[(size_t)q * I_MOE + n] = g_fug[(size_t)r * I_MOE + n];
                }
            }
            check_bf16("op75 GROUP_GLU_GEMMA_PF", got, want, (size_t)NS * I_MOE, 3e-2);
            free(want); free(got);
        }
        /* --- op 76 --- */
        float* g_pg = malloc((size_t)NSLOT * H * sizeof(float));
        {
            struct { void* p; const void* fu; const void* w; const void* m; const void* rp;
                     const void* rg; unsigned H_, I_, E_; } a = { d_partg, d_fug, d_ewt2, d_meta,
                                                                  d_rp, d_rg, H, I_MOE, E };
            plow_hsa_kernel kk = kern("moe_group_down_gemma_pf_k");
            poison(d_partg, (size_t)NSLOT * H * sizeof(float));
            run(&kk, NCU, &a, ASZ(a, E_));
            down(g_pg, d_partg, (size_t)NSLOT * H * sizeof(float));
            const unsigned NS = 24;
            float* want = malloc((size_t)NS * H * sizeof(float));
            float* got = malloc((size_t)NS * H * sizeof(float));
            for (unsigned q = 0; q < NS; q++) {
                unsigned r;
                do { r = r32() % TOT; } while (h_rp[r] == PLOW_EXPERT_UNUSED);
                const bf16* fs = g_fug + (size_t)r * I_MOE;
                const bf16* w = dn[row_exp[r]];
                for (unsigned hh = 0; hh < H; hh++) {
                    float acc = 0.0f;
                    const bf16* wr = w + (size_t)hh * I_MOE;
                    for (unsigned i = 0; i < I_MOE; i++) acc = fmaf(bf2f(fs[i]), bf2f(wr[i]), acc);
                    want[(size_t)q * H + hh] = h_rg[r] * acc;
                    got[(size_t)q * H + hh] = g_pg[(size_t)h_rp[r] * H + hh];
                }
            }
            check_f32("op76 GROUP_DOWN_GEMMA_PF", got, want, (size_t)NS * H, 5e-3);
            free(want); free(got);
        }
        /* --- op 77 --- */
        {
            bf16* h_gam = malloc(H * sizeof(bf16));
            bf16* h_h1 = malloc((size_t)T * H * sizeof(bf16));
            for (unsigned i = 0; i < H; i++) h_gam[i] = f2bf(1.0f + frand());
            for (size_t i = 0; i < (size_t)T * H; i++) h_h1[i] = f2bf(frand());
            void* d_gam = dev_alloc(H * sizeof(bf16));
            void* d_h1 = dev_alloc((size_t)T * H * sizeof(bf16));
            void* d_o = dev_alloc((size_t)T * H * sizeof(bf16));
            up(d_gam, h_gam, H * sizeof(bf16));
            up(d_h1, h_h1, (size_t)T * H * sizeof(bf16));
            struct { void* o; const void* p; const void* h1; const void* g;
                     unsigned H_, k_, T_; float eps_; } a = { d_o, d_partg, d_h1, d_gam, H, k, T,
                                                              EPS };
            plow_hsa_kernel kk = kern("moe_combine_norm_gemma_pf_k");
            poison(d_o, (size_t)T * H * sizeof(bf16));
            run(&kk, NCU, &a, ASZ(a, eps_));
            bf16* gv = malloc((size_t)T * H * sizeof(bf16));
            float* want = malloc((size_t)T * H * sizeof(float));
            down(gv, d_o, (size_t)T * H * sizeof(bf16));
            for (unsigned t = 0; t < T; t++) {
                const float* pt = g_pg + (size_t)t * k * H;
                double ss = 0.0;
                float* sum = malloc(H * sizeof(float));
                for (unsigned hh = 0; hh < H; hh++) {
                    float acc = 0.0f;
                    for (unsigned s = 0; s < k; s++) acc += pt[(size_t)s * H + hh];
                    sum[hh] = acc; ss += (double)acc * acc;
                }
                const float inv = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
                for (unsigned hh = 0; hh < H; hh++)
                    want[(size_t)t * H + hh] =
                        sum[hh] * inv * bf2f(h_gam[hh]) + bf2f(h_h1[(size_t)t * H + hh]);
                free(sum);
            }
            check_bf16("op77 COMBINE_NORM_GEMMA_PF", gv, want, (size_t)T * H, 1e-2);
            free(gv); free(want); free(h_gam); free(h_h1);
        }

        /* --- ops 81 / 82: w8a8 --- */
        printf("\n[prefill] w8a8 grouped expert GEMM\n");
        {
            unsigned char* xq8 = malloc((size_t)T * H);
            float* ascale = malloc((size_t)T * sizeof(float));
            for (size_t i = 0; i < (size_t)T * H; i++) {
                unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                xq8[i] = b;
            }
            for (unsigned t = 0; t < T; t++) ascale[t] = frand_pos() * 0.02f;
            void* d_xq8 = dev_alloc((size_t)T * H); up(d_xq8, xq8, (size_t)T * H);
            void* d_as = dev_alloc((size_t)T * sizeof(float));
            up(d_as, ascale, (size_t)T * sizeof(float));

            unsigned char** gu8 = calloc(E, sizeof(unsigned char*));
            unsigned char** dn8 = calloc(E, sizeof(unsigned char*));
            float** sgu = calloc(E, sizeof(float*));
            float** sdn = calloc(E, sizeof(float*));
            unsigned long long* w8 = calloc(E * 2, sizeof(unsigned long long));
            unsigned long long* s8 = calloc(E * 2, sizeof(unsigned long long));
            for (unsigned e = 0; e < E; e++) {
                gu8[e] = malloc(GU_N); dn8[e] = malloc(DN_N);
                for (size_t i = 0; i < GU_N; i++) {
                    unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                    gu8[e][i] = b;
                }
                for (size_t i = 0; i < DN_N; i++) {
                    unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                    dn8[e][i] = b;
                }
                sgu[e] = malloc(2 * I_MOE * sizeof(float));
                sdn[e] = malloc(H * sizeof(float));
                for (unsigned i = 0; i < 2 * I_MOE; i++) sgu[e][i] = frand_pos() * 0.02f;
                for (unsigned i = 0; i < H; i++) sdn[e][i] = frand_pos() * 0.02f;
                void* a = dev_alloc(GU_N); up(a, gu8[e], GU_N);
                void* b = dev_alloc(DN_N); up(b, dn8[e], DN_N);
                void* c = dev_alloc(2 * I_MOE * sizeof(float));
                up(c, sgu[e], 2 * I_MOE * sizeof(float));
                void* d = dev_alloc(H * sizeof(float)); up(d, sdn[e], H * sizeof(float));
                w8[(size_t)e * 2 + 0] = (unsigned long long)(size_t)a;
                w8[(size_t)e * 2 + 1] = (unsigned long long)(size_t)b;
                s8[(size_t)e * 2 + 0] = (unsigned long long)(size_t)c;
                s8[(size_t)e * 2 + 1] = (unsigned long long)(size_t)d;
            }
            void* d_w8 = dev_alloc(E * 2 * sizeof(unsigned long long));
            void* d_s8 = dev_alloc(E * 2 * sizeof(unsigned long long));
            up(d_w8, w8, E * 2 * sizeof(unsigned long long));
            up(d_s8, s8, E * 2 * sizeof(unsigned long long));

            {
                struct { void* fu; const void* x; const void* as; const void* w; const void* s;
                         const void* m; const void* rt; unsigned I_, H_, E_, act_; } a = {
                    d_fug, d_xq8, d_as, d_w8, d_s8, d_meta, d_rt, I_MOE, H, E, 0u };
                plow_hsa_kernel kk = kern("moe_group_glu_gemma_pf_w8a8_k");
                poison(d_fug, (size_t)TOT * I_MOE * sizeof(bf16));
                run(&kk, NCU, &a, ASZ(a, act_));
                down(g_fug, d_fug, (size_t)TOT * I_MOE * sizeof(bf16));
                const unsigned NS = 32;
                float* want = malloc((size_t)NS * I_MOE * sizeof(float));
                bf16* got = malloc((size_t)NS * I_MOE * sizeof(bf16));
                for (unsigned q = 0; q < NS; q++) {
                    unsigned r;
                    do { r = r32() % TOT; } while (h_rt[r] == PLOW_EXPERT_UNUSED);
                    const unsigned tok = h_rt[r], e = row_exp[r];
                    const unsigned char* x = xq8 + (size_t)tok * H;
                    for (unsigned n = 0; n < I_MOE; n++) {
                        float g = 0.0f, u = 0.0f;
                        const unsigned char* gr = gu8[e] + (size_t)n * H;
                        const unsigned char* ur = gu8[e] + (size_t)(I_MOE + n) * H;
                        for (unsigned hh = 0; hh < H; hh++) {
                            const float xv = fp8_ocp_to_f32(x[hh]);
                            g = fmaf(xv, fp8_ocp_to_f32(gr[hh]), g);
                            u = fmaf(xv, fp8_ocp_to_f32(ur[hh]), u);
                        }
                        g *= ascale[tok] * sgu[e][n];
                        u *= ascale[tok] * sgu[e][I_MOE + n];
                        want[(size_t)q * I_MOE + n] = gelu_tanh(g) * u;
                        got[(size_t)q * I_MOE + n] = g_fug[(size_t)r * I_MOE + n];
                    }
                }
                check_bf16("op81 GROUP_GLU_GEMMA_PF_W8A8", got, want, (size_t)NS * I_MOE, 3e-2);
                free(want); free(got);
            }
            {
                /* the DOWN arm's A is a gathered e4m3 [padded_rows, I_moe] with a per-ROW scale */
                unsigned char* fu8 = malloc((size_t)TOT * I_MOE);
                float* fscale = malloc((size_t)TOT * sizeof(float));
                for (size_t i = 0; i < (size_t)TOT * I_MOE; i++) {
                    unsigned char b; do { b = (unsigned char)r32(); } while ((b & 0x7fu) == 0x7fu);
                    fu8[i] = b;
                }
                for (unsigned r = 0; r < TOT; r++) fscale[r] = frand_pos() * 0.02f;
                void* d_fu8 = dev_alloc((size_t)TOT * I_MOE);
                up(d_fu8, fu8, (size_t)TOT * I_MOE);
                void* d_fs = dev_alloc((size_t)TOT * sizeof(float));
                up(d_fs, fscale, (size_t)TOT * sizeof(float));
                struct { void* p; const void* fu; const void* fs; const void* w; const void* s;
                         const void* m; const void* rp; const void* rg;
                         unsigned H_, I_, E_; } a = { d_partg, d_fu8, d_fs, d_w8, d_s8, d_meta,
                                                      d_rp, d_rg, H, I_MOE, E };
                plow_hsa_kernel kk = kern("moe_group_down_gemma_pf_w8a8_k");
                poison(d_partg, (size_t)NSLOT * H * sizeof(float));
                run(&kk, NCU, &a, ASZ(a, E_));
                down(g_pg, d_partg, (size_t)NSLOT * H * sizeof(float));
                const unsigned NS = 16;
                float* want = malloc((size_t)NS * H * sizeof(float));
                float* got = malloc((size_t)NS * H * sizeof(float));
                for (unsigned q = 0; q < NS; q++) {
                    unsigned r;
                    do { r = r32() % TOT; } while (h_rp[r] == PLOW_EXPERT_UNUSED);
                    const unsigned e = row_exp[r];
                    const unsigned char* fs = fu8 + (size_t)r * I_MOE;
                    for (unsigned hh = 0; hh < H; hh++) {
                        float acc = 0.0f;
                        const unsigned char* wr = dn8[e] + (size_t)hh * I_MOE;
                        for (unsigned i = 0; i < I_MOE; i++)
                            acc = fmaf(fp8_ocp_to_f32(fs[i]), fp8_ocp_to_f32(wr[i]), acc);
                        want[(size_t)q * H + hh] = h_rg[r] * fscale[r] * sdn[e][hh] * acc;
                        got[(size_t)q * H + hh] = g_pg[(size_t)h_rp[r] * H + hh];
                    }
                }
                check_f32("op82 GROUP_DOWN_GEMMA_PF_W8A8", got, want, (size_t)NS * H, 5e-3);
                free(want); free(got); free(fu8); free(fscale);
            }
        }

        /* =============================== BENCHMARKS =============================== */
        if (do_bench) {
            printf("\n[bench] best of 20, MI300X peaks: 5325 GB/s HBM, ~1307 TF/s bf16\n");
            {
                struct { void* fu; const void* x; const void* w; const void* m; const void* rt;
                         unsigned I_, H_, E_, act_; } a = { d_fug, d_xn, d_ewt2, d_meta, d_rt,
                                                            I_MOE, H, E, 0u };
                plow_hsa_kernel kk = kern("moe_group_glu_gemma_pf_k");
                const double ms = bench(&kk, NCU, &a, ASZ(a, act_), 20);
                const double fl = 2.0 * (double)TOT * I_MOE * 2.0 * H;
                printf("  op75 GROUP_GLU_PF   T=%u E=%u  %8.3f ms  %7.1f TF/s (%.1f%% of peak)\n",
                       T, E, ms, fl / (ms * 1e9), 100.0 * fl / (ms * 1e9) / 1307.0);
            }
            {
                struct { void* p; const void* fu; const void* w; const void* m; const void* rp;
                         const void* rg; unsigned H_, I_, E_; } a = { d_partg, d_fug, d_ewt2,
                                                                      d_meta, d_rp, d_rg, H,
                                                                      I_MOE, E };
                plow_hsa_kernel kk = kern("moe_group_down_gemma_pf_k");
                const double ms = bench(&kk, NCU, &a, ASZ(a, E_), 20);
                const double fl = 2.0 * (double)TOT * H * I_MOE;
                printf("  op76 GROUP_DOWN_PF  T=%u E=%u  %8.3f ms  %7.1f TF/s (%.1f%% of peak)\n",
                       T, E, ms, fl / (ms * 1e9), 100.0 * fl / (ms * 1e9) / 1307.0);
            }
            {
                struct { void* fu; const void* x; const void* t; const void* w;
                         unsigned k_, I_, H_, E_, R_; } a = { d_fu, d_x, d_table, d_ewt, TOPK,
                                                              I_MOE, H, N_EXP, 1u };
                const double ms = bench(&k_glu, NCU, &a, ASZ(a, R_), 20);
                const double by = (double)TOPK * 2.0 * I_MOE * H * 2.0;
                printf("  op62 EXPERT_GLU     k=%u          %8.3f ms  %7.1f GB/s (%.1f%% of peak)\n",
                       TOPK, ms, by / (ms * 1e6), 100.0 * by / (ms * 1e6) / 5325.0);
            }
            {
                /* INTERLEAVED A/B for op 63's sub-group split against the whole-wave arm it
                 * replaced. Two separate runs of a 30-microsecond kernel on a SHARED box measure
                 * the box, not the kernel — the difference between two consecutive best-of-N in
                 * ONE process is the only comparison that means anything here. */
                struct { void* p; const void* fu; const void* t; const void* w;
                         unsigned k_, H_, I_, E_, R_; } a = { d_part, d_fu, d_table, d_ewt, TOPK,
                                                              H, I_MOE, N_EXP, 1u };
                plow_hsa_kernel kw = kern("moe_expert_down_gemma_wave_k");
                const double by = (double)TOPK * H * I_MOE * 2.0;
                double bs = 1e9, bw = 1e9;
                for (int r = 0; r < 6; r++) {
                    const double s1 = bench(&k_down, NCU, &a, ASZ(a, R_), 8);
                    const double w1 = bench(&kw, NCU, &a, ASZ(a, R_), 8);
                    if (s1 < bs) bs = s1;
                    if (w1 < bw) bw = w1;
                }
                printf("  op63 EXPERT_DOWN    k=%u          %8.3f ms  %7.1f GB/s (%.1f%% of peak)\n",
                       TOPK, bs, by / (bs * 1e6), 100.0 * by / (bs * 1e6) / 5325.0);
                printf("       A/B whole-wave arm            %8.3f ms  %7.1f GB/s  => split is %.2fx\n",
                       bw, by / (bw * 1e6), bw / bs);
            }
            {
                struct { void* s; const void* r; const void* p; const void* sc; unsigned H_, E_,
                         R_; float root_, eps_; } a = { d_score, d_resid, d_proj, d_scale, H,
                                                        N_EXP, 1u, root, EPS };
                plow_hsa_kernel k1 = kern("moe_router_gemma_score_k");
                plow_hsa_kernel k2 = kern("moe_router_gemma_score_fast_k");
                printf("  op67 ROUTER_SCORE                %8.3f ms (exact, ordered replay)\n",
                       bench(&k1, NCU, &a, ASZ(a, eps_), 20));
                printf("  op69 ROUTER_SCORE_FAST           %8.3f ms\n",
                       bench(&k2, NCU, &a, ASZ(a, eps_), 20));
            }
            {
                struct { void* t; const void* r; const void* p; const void* s; const void* pe;
                         unsigned H_, E_, K_, R_; float root_, eps_; } a = {
                    d_table, d_resid, d_proj, d_scale, d_pes, H, N_EXP, TOPK, 1u, root, EPS };
                plow_hsa_kernel k = kern("moe_router_gemma_k");
                printf("  op61 ROUTER (fused, 1 WG)        %8.3f ms\n",
                       bench(&k, 1, &a, ASZ(a, eps_), 20));
            }
            printf("\n[bench] grouped prefill at the REAL expert count\n");
            bench_grouped_real(NCU, N_EXP, 512u, TOPK);
        }
        free(h_tb); free(h_meta); free(h_rt); free(h_rp); free(h_rg); free(h_xn);
        free(row_exp); free(g_fug); free(g_pg);
    }

    printf("\n%d/%d checks passed\n", checks - fails, checks);
    plow_hsa_shutdown(g_h);
    return fails ? 1 : 0;
}
