/* gemm_gfx950_test.c — correctness + throughput of the MFMA bf16 GEMM.
 *
 * C[M,N] = A[M,K] . B[N,K]^T, with B stored [out_features, in_features] exactly
 * as HF stores a Linear weight.
 *
 * Shapes are the real Gemma 4 31B projections, not round numbers.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int fails = 0;

static void run(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, const char* label,
                unsigned M, unsigned N, unsigned K, int check) {
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(h, nA * 2);
    bf16* hB = plow_hsa_alloc_host(h, nB * 2);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    /* Small magnitudes: a K=5376 dot product in bf16 accumulates real error, and
     * we want to measure the kernel, not the format. */
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nB; i++) hB[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dA = plow_hsa_alloc(h, 0, nA * 2);
    void* dB = plow_hsa_alloc(h, 0, nB * 2);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    plow_hsa_copy_h2d(h, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(h, 0, dB, hB, nB * 2);

    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; unsigned m, n, kk;
    } args = {dC, dA, dB, M, N, K};

    /* Sweep the tile configs and keep the best — this is what the compiler does
     * per shape bucket. There is no single winner: a bigger tile cuts HBM
     * re-reads, but a tile that does not DIVIDE the shape leaves a ragged edge on
     * every tile-column and quantizes badly against 256 CUs. */
    static const char* cfg_name[5] = {"256x256x64", "256x128x64", "128x256x64",
                                      "128x128x64", "64x128x64"};
    double best = 1e30;
    int best_cfg = 0;
    for (int c = 0; c < 5; c++) {
        char sym[32];
        snprintf(sym, sizeof(sym), "gemm_c%d", c);
        plow_hsa_kernel kc;
        if (plow_hsa_get_kernel(h, 0, sym, &kc) != 0) continue;
        plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        const int reps = 5;
        const double t0 = now();
        for (int r = 0; r < reps; r++)
            plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        const double d = (now() - t0) / reps;
        if (getenv("PLOW_GEMM_VERBOSE"))
            printf("      [%-11s %7.3f ms  %7.1f TF/s]\n", cfg_name[c], d * 1e3,
                   2.0 * M * N * K / d / 1e12);
        if (d < best) { best = d; best_cfg = c; }
    }
    { /* re-run the winner so the check below sees its output */
        char sym[32];
        snprintf(sym, sizeof(sym), "gemm_c%d", best_cfg);
        plow_hsa_kernel kc;
        plow_hsa_get_kernel(h, 0, sym, &kc);
        plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
    }
    (void)k;
    const double dt = best;

    const double flops = 2.0 * M * N * K;
    printf("  %-22s %5u x %6u x %5u  %-11s %7.3f ms  %7.1f TF/s", label, M, N, K,
           cfg_name[best_cfg], dt * 1e3, flops / dt / 1e12);

    if (check) {
        plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);
        /* Spot-check a sample of output elements against an f64 reference. */
        double worst = 0.0;
        for (int s = 0; s < 512; s++) {
            const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
            double want = 0.0;
            for (unsigned kk = 0; kk < K; kk++)
                want += (double)bf2f(hA[(size_t)m * K + kk]) * bf2f(hB[(size_t)n * K + kk]);
            const double got = bf2f(hC[(size_t)m * N + n]);
            const double rel = fabs(got - want) / (fabs(want) + 1e-2);
            if (rel > worst) worst = rel;
        }
        /* f32 accumulate over K, one bf16 rounding at the store => a few e-3. */
        const int ok = worst < 3e-2;
        printf("   %s (worst rel %.4f)", ok ? "PASS" : "FAIL", worst);
        if (!ok) fails++;
    }
    printf("\n");

    plow_hsa_free(h, dA); plow_hsa_free(h, dB); plow_hsa_free(h, dC);
}

/* OCP e4m3 (torch.float8_e4m3fn) decode of one byte: 1 sign, 4 exp (bias 7), 3 mantissa, no inf,
 * subnormals at 2^-6, max 448. This is exactly what the gfx950 fp8 MFMA interprets the operand as,
 * so the reference below (decode -> f64 dot) measures the KERNEL, not the format. */
static double e4m3_decode(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0) v = (m / 8.0) * 0.015625;                 /* 2^-6 subnormal */
    else v = (1.0 + m / 8.0) * ldexp(1.0, e - 7);
    return s ? -v : v;
}

/* FP8 (w8a8) GEMM: correctness against a dequant-fp8 f64 reference + throughput. A/B are RANDOM
 * valid e4m3 bytes (NaN 0x7F/0xFF avoided) with random positive per-row(A)/per-channel(B) scales,
 * so this validates the fp8 load path, LDS byte swizzle, MFMA operand layout and dequant epilogue
 * end to end without needing a host float->e4m3 encoder. */
static void run_fp8(plow_hsa* h, unsigned NCU, const char* label, unsigned M, unsigned N,
                    unsigned K) {
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    unsigned char* hA = plow_hsa_alloc_host(h, nA);
    unsigned char* hB = plow_hsa_alloc_host(h, nB);
    float* hAs = plow_hsa_alloc_host(h, (size_t)M * 4);
    float* hWs = plow_hsa_alloc_host(h, (size_t)N * 4);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    /* SMALL e4m3 magnitudes (exp field <= 7 -> |v| < 2), exactly the reason the bf16 test uses
     * [-0.5,0.5]: a K=9728 dot of FULL-range fp8 (up to 448 each) has such a dynamic range that the
     * kernel's f32 MFMA accumulator and an f64 reference legitimately differ by several % from
     * catastrophic cancellation — that is the FORMAT, not the kernel. Small values keep the sum
     * well-conditioned so any real LAYOUT/MFMA bug shows as ~100% error, not a few %. Runtime
     * per-row/per-channel scales carry the real dynamic range in production. */
    for (size_t i = 0; i < nA; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2; /* e<=7 => |v|<2 */
        hA[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (size_t i = 0; i < nB; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hB[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (unsigned i = 0; i < M; i++) hAs[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;
    for (unsigned i = 0; i < N; i++) hWs[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;

    void* dA = plow_hsa_alloc(h, 0, nA);
    void* dB = plow_hsa_alloc(h, 0, nB);
    void* dAs = plow_hsa_alloc(h, 0, (size_t)M * 4);
    void* dWs = plow_hsa_alloc(h, 0, (size_t)N * 4);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    plow_hsa_copy_h2d(h, 0, dA, hA, nA);
    plow_hsa_copy_h2d(h, 0, dB, hB, nB);
    plow_hsa_copy_h2d(h, 0, dAs, hAs, (size_t)M * 4);
    plow_hsa_copy_h2d(h, 0, dWs, hWs, (size_t)N * 4);

    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; const void* as; const void* ws;
        unsigned m, n, kk;
    } args = {dC, dA, dB, dAs, dWs, M, N, K};

    static const struct { const char* sym; const char* nm; } cfgs[3] = {
        {"gemm_fp8_c0", "256x256x64"}, {"gemm_fp8_c3", "128x128x64"}, {"gemm_fp8_c4", "64x128x64"}};
    double best = 1e30; int best_i = 0;
    for (int c = 0; c < 3; c++) {
        plow_hsa_kernel kc;
        if (plow_hsa_get_kernel(h, 0, cfgs[c].sym, &kc) != 0) continue;
        plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        const int reps = 5;
        const double t0 = now();
        for (int r = 0; r < reps; r++)
            plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
        plow_hsa_wait(h, 0);
        const double d = (now() - t0) / reps;
        if (getenv("PLOW_GEMM_VERBOSE"))
            printf("      [fp8 %-11s %7.3f ms  %7.1f TF/s]\n", cfgs[c].nm, d * 1e3,
                   2.0 * M * N * K / d / 1e12);
        if (d < best) { best = d; best_i = c; }
    }
    { plow_hsa_kernel kc; plow_hsa_get_kernel(h, 0, cfgs[best_i].sym, &kc);
      plow_hsa_launch(h, 0, &kc, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
      plow_hsa_wait(h, 0); }

    const double flops = 2.0 * M * N * K;
    printf("  %-22s %5u x %6u x %5u  fp8 %-11s %7.3f ms  %7.1f TF/s", label, M, N, K,
           cfgs[best_i].nm, best * 1e3, flops / best / 1e12);

    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);
    double worst = 0.0;
    for (int s = 0; s < 512; s++) {
        const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += e4m3_decode(hA[(size_t)m * K + kk]) * e4m3_decode(hB[(size_t)n * K + kk]);
        want *= (double)hAs[m] * (double)hWs[n];
        const double got = bf2f(hC[(size_t)m * N + n]);
        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    const int ok = worst < 3e-2;
    printf("   %s (worst rel %.4f)\n", ok ? "PASS" : "FAIL", worst);
    if (!ok) fails++;

    plow_hsa_free(h, dA); plow_hsa_free(h, dB); plow_hsa_free(h, dAs); plow_hsa_free(h, dWs);
    plow_hsa_free(h, dC);
}

/* Fused RMSNorm-prologue GEMM: C = norm(A) . B^T, where the normalized A never
 * touches HBM. Two ops in the program (rowrms -> gemm_norm), one pass over A. */
static void run_norm(plow_hsa* h, plow_hsa_kernel* krms, plow_hsa_kernel* kgn, unsigned NCU,
                     unsigned M, unsigned N, unsigned K) {
    const float EPS = 1e-6f;
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(h, nA * 2);
    bf16* hB = plow_hsa_alloc_host(h, nB * 2);
    bf16* hG = plow_hsa_alloc_host(h, K * 2);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nB; i++) hB[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (unsigned i = 0; i < K; i++) hG[i] = f2bf(1.0f + 0.1f * ((rand() % 9) - 4) / 4.0f);

    void* dA = plow_hsa_alloc(h, 0, nA * 2);
    void* dB = plow_hsa_alloc(h, 0, nB * 2);
    void* dG = plow_hsa_alloc(h, 0, K * 2);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    void* dR = plow_hsa_alloc(h, 0, (size_t)M * 4);
    plow_hsa_copy_h2d(h, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(h, 0, dB, hB, nB * 2);
    plow_hsa_copy_h2d(h, 0, dG, hG, K * 2);

    struct __attribute__((packed)) {
        void* rms; const void* x; unsigned rows, feat; float eps;
    } ra = {dR, dA, M, K, EPS};
    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; const void* rms; const void* g;
        unsigned m, n, kk;
    } ga = {dC, dA, dB, dR, dG, M, N, K};

    plow_hsa_launch(h, 0, krms, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ra, sizeof(ra));
    plow_hsa_launch(h, 0, kgn, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ga, sizeof(ga));
    plow_hsa_wait(h, 0);

    const int reps = 5;
    const double t0 = now();
    for (int r = 0; r < reps; r++) {
        plow_hsa_launch(h, 0, krms, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ra, sizeof(ra));
        plow_hsa_launch(h, 0, kgn, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ga, sizeof(ga));
    }
    plow_hsa_wait(h, 0);
    const double dt = (now() - t0) / reps;
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    double worst = 0.0;
    for (int s = 0; s < 256; s++) {
        const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double ss = 0.0;
        for (unsigned k = 0; k < K; k++) {
            const double v = bf2f(hA[(size_t)m * K + k]);
            ss += v * v;
        }
        /* Gemma RMSNorm: * w, NOT * (1 + w); eps inside the power. */
        const double inv = pow(ss / K + EPS, -0.5);
        double want = 0.0;
        for (unsigned k = 0; k < K; k++)
            want += (double)bf2f(f2bf((float)(bf2f(hA[(size_t)m * K + k]) * inv *
                                             bf2f(hG[k])))) *
                    bf2f(hB[(size_t)n * K + k]);
        const double got = bf2f(hC[(size_t)m * N + n]);
        worst = fmax(worst, fabs(got - want) / (fabs(want) + 1e-2));
    }
    const double flops = 2.0 * M * N * K;
    const int ok = worst < 3e-2;
    printf("  %-26s %5u x %6u x %5u   %7.3f ms   %7.1f TFLOP/s   %s (worst rel %.4f)\n",
           "rowrms+gemm_norm (fused)", M, N, K, dt * 1e3, flops / dt / 1e12,
           ok ? "PASS" : "FAIL", worst);
    if (!ok) fails++;
}

/* Decode GEMV. Bandwidth-bound: the figure of merit is GB/s of weight streamed,
 * not FLOP/s. MI350X HBM is ~8 TB/s. */
static void run_gemv(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, unsigned M, unsigned N,
                     unsigned K) {
    const size_t nx = (size_t)M * K, nW = (size_t)N * K, nC = (size_t)M * N;
    bf16* hx = plow_hsa_alloc_host(h, nx * 2);
    bf16* hW = plow_hsa_alloc_host(h, nW * 2);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    for (size_t i = 0; i < nx; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nW; i++) hW[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dx = plow_hsa_alloc(h, 0, nx * 2);
    void* dW = plow_hsa_alloc(h, 0, nW * 2);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    plow_hsa_copy_h2d(h, 0, dx, hx, nx * 2);
    plow_hsa_copy_h2d(h, 0, dW, hW, nW * 2);

    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* rms; const void* g;
        unsigned m, n, kk, norm;
    } a = {dC, dx, dW, NULL, NULL, M, N, K, 0};

    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
    plow_hsa_wait(h, 0);
    const int reps = 10;
    const double t0 = now();
    for (int r = 0; r < reps; r++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
    plow_hsa_wait(h, 0);
    const double dt = (now() - t0) / reps;
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    double worst = 0.0;
    for (int s = 0; s < 256; s++) {
        const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += (double)bf2f(hx[(size_t)m * K + kk]) * bf2f(hW[(size_t)n * K + kk]);
        worst = fmax(worst, fabs(bf2f(hC[(size_t)m * N + n]) - want) / (fabs(want) + 1e-2));
    }
    const double gb = (double)nW * 2.0 / 1e9;
    const int ok = worst < 3e-2;
    printf("  M=%-3u %6u x %5u   %7.3f ms   %7.0f GB/s (of ~8000)   %s (worst rel %.4f)\n", M, N,
           K, dt * 1e3, gb / dt, ok ? "PASS" : "FAIL", worst);
    if (!ok) fails++;

    plow_hsa_free(h, dx); plow_hsa_free(h, dW); plow_hsa_free(h, dC);
}

int main(void) {
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B\n\n", nm, cus, lds);

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "gemma_gemm_bf16", &k) != 0) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
    }
    printf("gemm LDS=%u B, kernarg=%u B\n\n", k.group_segment_size, k.kernarg_size);
    srand(5);

    /* Gemma 4 31B, prefill T=4096. H=5376, I=21504. */
    const unsigned T = 4096, H = 5376, I = 21504;
    printf("Gemma 4 31B projections (prefill T=%u):\n", T);
    run(h, &k, cus, "q_proj  (sliding)", T, 8192, H, 1);
    run(h, &k, cus, "kv_proj (sliding)", T, 4096, H, 1);
    run(h, &k, cus, "q_proj  (global)", T, 16384, H, 1);
    run(h, &k, cus, "o_proj  (sliding)", T, H, 8192, 1);
    run(h, &k, cus, "gate/up_proj", T, I, H, 1);
    run(h, &k, cus, "down_proj", T, H, I, 1);

    /* FP8 (w8a8) — the same Qwen3-4B prefill projections, both operands e4m3, 2x-rate MFMA.
     * Compare TF/s directly against the bf16 rows above (the whole 2x question). Qwen3-4B:
     * hidden 2560, inter 9728, q 4096 (32*128), kv 1024 (8*128). */
    if (plow_hsa_get_kernel(h, 0, "gemm_fp8_c0", &k) == 0) {
        const unsigned H4 = 2560, I4 = 9728, QD = 4096, KV = 1024;
        plow_hsa_get_kernel(h, 0, "gemma_gemm_bf16", &k);
        printf("\nBF16 GEMM at the SAME Qwen3-4B shapes (apples-to-apples baseline for fp8 below):\n");
        run(h, &k, cus, "q_proj", T, QD, H4, 1);
        run(h, &k, cus, "kv_proj", T, KV, H4, 1);
        run(h, &k, cus, "o_proj", T, H4, QD, 1);
        run(h, &k, cus, "gate/up_proj", T, I4, H4, 1);
        run(h, &k, cus, "down_proj", T, H4, I4, 1);
        printf("\nFP8 w8a8 GEMM — Qwen3-4B projections (prefill T=%u), compare TF/s to bf16 above:\n", T);
        run_fp8(h, cus, "q_proj", T, QD, H4);
        run_fp8(h, cus, "kv_proj", T, KV, H4);
        run_fp8(h, cus, "o_proj", T, H4, QD);
        run_fp8(h, cus, "gate/up_proj", T, I4, H4);
        run_fp8(h, cus, "down_proj", T, H4, I4);
        printf("  --- FP8 at REAL prefill lengths ---\n");
        for (unsigned t = 128; t <= 512; t *= 4) {
            printf("  --- T=%u ---\n", t);
            run_fp8(h, cus, "q_proj", t, QD, H4);
            run_fp8(h, cus, "o_proj", t, H4, QD);
            run_fp8(h, cus, "gate/up_proj", t, I4, H4);
            run_fp8(h, cus, "down_proj", t, H4, I4);
        }
    }

    /* The shapes a REAL prefill actually runs. T=4096 fills the machine and flatters the
     * big tile; a 128- or 512-token prompt does not, and that is the common case. This is
     * the data plowc's tile heuristic is calibrated on. */
    printf("\nSame projections at REAL prefill lengths (this is what plowc must choose for):\n");
    for (unsigned t = 128; t <= 512; t *= 4) {
        printf("  --- T=%u ---\n", t);
        run(h, &k, cus, "q_proj  (sliding)", t, 8192, H, 1);
        run(h, &k, cus, "o_proj  (sliding)", t, H, 8192, 1);
        run(h, &k, cus, "gate/up_proj", t, I, H, 1);
        run(h, &k, cus, "down_proj", t, H, I, 1);
    }

    plow_hsa_kernel krms, kgn, kgv;
    if (plow_hsa_get_kernel(h, 0, "gemma_rowrms_bf16", &krms) ||
        plow_hsa_get_kernel(h, 0, "gemma_gemm_norm_bf16", &kgn) ||
        plow_hsa_get_kernel(h, 0, "gemma_gemv_bf16", &kgv)) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
    }

    printf("\nFused norm-prologue (normalized activation never reaches HBM):\n");
    run_norm(h, &krms, &kgn, cus, T, I, H); /* pre_ffn_norm -> gate_proj */
    run_norm(h, &krms, &kgn, cus, T, 8192, H); /* input_norm -> q_proj    */

    printf("\nDecode GEMV (bandwidth-bound; weight streamed once):\n");
    /* Each M uses ITS OWN compile-time bucket kernel — that is what a bucket is. */
    plow_hsa_kernel kgv4, kgv8;
    if (plow_hsa_get_kernel(h, 0, "gemma_gemv_m4", &kgv4) ||
        plow_hsa_get_kernel(h, 0, "gemma_gemv_m8", &kgv8)) {
        printf("gemv bucket kernels missing\n");
        return 1;
    }
    run_gemv(h, &kgv, cus, 1, I, H);
    run_gemv(h, &kgv4, cus, 4, I, H);
    run_gemv(h, &kgv8, cus, 8, I, H);
    run_gemv(h, &kgv, cus, 1, H, I); /* down_proj shape */

    printf("\n%s (%d failure%s)\n", fails ? "GEMM FAILED" : "GEMM FAMILY CORRECT", fails,
           fails == 1 ? "" : "s");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
