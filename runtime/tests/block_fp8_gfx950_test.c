/* block_fp8_gfx950_test.c — DeepSeek/GLM BLOCK-fp8 decode GEMV on device vs an f64 reference.
 *
 * GLM-5.2-FP8 (and DeepSeek-V3) quantise weights with weight_block_size [128,128]: the weight is
 * e4m3 and there is ONE f32 dequant scale per [128 out-channel][128 K] block, laid out as a
 * ceil(N/128) x ceil(K/128) row-major grid. This is DIFFERENT from plow's existing per-CHANNEL fp8
 * GEMV (one scale per output column, applied once in the epilogue): the block scale varies along K
 * and so must be folded into the reduction per 128-K block. This test drives the new gemv_fp8_blk
 * wrapper (d_gemv_fp8_blk -> gemv_rows_fp8_blk) and checks it against a decode-fp8-dequant f64
 * reference over the SAME block scheme, on real GLM decode shapes plus a ragged (non-128-multiple)
 * shape that exercises the ceil / overshoot-clamp path.
 *
 * w8a16: x (activation) is bf16, W (weight) is e4m3 block-scaled — the plow decode weight-stream
 * path. Truth is decode(W)*x*block_scale summed in f64 (measures the KERNEL, not the format).
 *
 * Build with scripts/build_block_fp8.sh; run under `sg render` on one GPU.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (bf16)(u >> 16);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* OCP e4m3 (torch.float8_e4m3fn) decode — same as gemm_gfx950_test.c. */
static double e4m3_decode(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0) v = (m / 8.0) * 0.015625;
    else v = (1.0 + m / 8.0) * ldexp(1.0, e - 7);
    return s ? -v : v;
}

static int fails = 0;

/* Block-fp8 decode GEMV: C[1][N] = Σ_k x[k] * decode(W[n][k]) * wscale[n/128][k/128]. */
static void run(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, const char* label, unsigned N,
                unsigned K) {
    const unsigned M = 1;
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    const size_t nW = (size_t)N * K, nS = (size_t)NB * KB, nC = (size_t)M * N;

    unsigned char* hW = plow_hsa_alloc_host(h, nW);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)M * K * 2);
    float* hS = plow_hsa_alloc_host(h, nS * 4);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);

    /* SMALL e4m3 magnitudes (exp field <= 7 -> |v| < 2) so the K-long dot stays well-conditioned;
     * the per-block scales carry the dynamic range, exactly as the per-channel fp8 GEMV test does. */
    for (size_t i = 0; i < nW; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hW[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (unsigned i = 0; i < M * K; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nS; i++) hS[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;

    void* dW = plow_hsa_alloc(h, 0, nW);
    void* dx = plow_hsa_alloc(h, 0, (size_t)M * K * 2);
    void* dS = plow_hsa_alloc(h, 0, nS * 4);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    plow_hsa_copy_h2d(h, 0, dW, hW, nW);
    plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)M * K * 2);
    plow_hsa_copy_h2d(h, 0, dS, hS, nS * 4);

    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* ws; unsigned m, n, kk;
    } args = {dC, dx, dW, dS, M, N, K};

    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    /* Perf: launch-overhead-dominated standalone (~30us floor per campaign notes), so this is a
     * per-shape PROFILE not a full-model per-op proxy. TB/s over the fp8 weight bytes streamed (N*K). */
    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double ms = (now() - t0) * 1e3 / ITERS;
    const double tbs = (double)nW / (ms * 1e-3) / 1e12;

    double worst = 0.0;
    for (unsigned n = 0; n < N; n++) {
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += (double)bf2f(hx[kk]) * e4m3_decode(hW[(size_t)n * K + kk]) *
                    (double)hS[(size_t)(n >> 7) * KB + (kk >> 7)];
        const double got = bf2f(hC[n]);
        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    const int ok = worst < 3e-2;
    printf("  %-20s N=%5u K=%5u  %s rel %.4f | %.3f ms  %.2f TB/s\n", label, N, K,
           ok ? "PASS" : "FAIL", worst, ms, tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dW); plow_hsa_free(h, dx); plow_hsa_free(h, dS); plow_hsa_free(h, dC);
}

/* Block-fp8 DENSE PREFILL GEMM (op 107, d_gemm_fp8_blk) vs the SAME f64 reference the decode GEMV
 * above uses, plus a direct cross-kernel check against `gemv_fp8_blk` on identical weights.
 *
 * THE CROSS-KERNEL CHECK IS THE POINT, not a bonus. A block-fp8 GEMM with a wrong scale block reads
 * as plausible-but-wrong output and never crashes, and the two kernels index the grid completely
 * differently — the GEMV folds `wscale[(n>>7)*KB + (k>>7)]` into a per-lane chunk partial, the GEMM
 * promotes a whole MFMA accumulator by `wscale[(n0>>7)*KB + (kt>>1)]` at a k-tile boundary. An f64
 * reference proves each is right; agreeing with each other proves they read ONE convention, which
 * is what the emitter relies on when it hands both phases the same `.weight_scale_inv` handle.
 *
 * The reference SAMPLES (m,n) pairs rather than computing all M*N: at M=512, N=6144, K=4096 the full
 * product is 12.9 G f64 MACs on one core. Sampling is not a weaker test here — a tiling, swizzle or
 * scale-index bug is systematic across the output, not concentrated in a few elements. */
static void run_gemm_blk(plow_hsa* h, plow_hsa_kernel* kg, plow_hsa_kernel* kv, unsigned NCU,
                         const char* label, unsigned M, unsigned N, unsigned K) {
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    const size_t nW = (size_t)N * K, nS = (size_t)NB * KB, nA = (size_t)M * K, nC = (size_t)M * N;

    unsigned char* hW = plow_hsa_alloc_host(h, nW);
    bf16* hA = plow_hsa_alloc_host(h, nA * 2);
    float* hS = plow_hsa_alloc_host(h, nS * 4);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    bf16* hCv = plow_hsa_alloc_host(h, (size_t)N * 2);

    /* Same conditioning as the GEMV above: exp field <= 7 (|v| < 2) so a K-long dot stays
     * well-conditioned and a real layout bug shows as ~100% error rather than a few percent of
     * legitimate f32-vs-f64 cancellation. The per-block scales carry the dynamic range. */
    for (size_t i = 0; i < nW; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hW[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nS; i++) hS[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;

    void* dW = plow_hsa_alloc(h, 0, nW);
    void* dA = plow_hsa_alloc(h, 0, nA * 2);
    void* dS = plow_hsa_alloc(h, 0, nS * 4);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    void* dCv = plow_hsa_alloc(h, 0, (size_t)N * 2);
    plow_hsa_copy_h2d(h, 0, dW, hW, nW);
    plow_hsa_copy_h2d(h, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(h, 0, dS, hS, nS * 4);

    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* ws; unsigned m, n, kk;
    } args = {dC, dA, dW, dS, M, N, K};
    plow_hsa_launch(h, 0, kg, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args,
                    sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    /* The decode GEMV on row 0 of the SAME A, W and scale grid. */
    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* ws; unsigned m, n, kk;
    } vargs = {dCv, dA, dW, dS, 1u, N, K};
    plow_hsa_launch(h, 0, kv, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &vargs,
                    sizeof(vargs));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hCv, dCv, (size_t)N * 2);

    double worst = 0.0;
    const int PROBES = 512;
    for (int p = 0; p < PROBES; p++) {
        const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += (double)bf2f(hA[(size_t)m * K + kk]) * e4m3_decode(hW[(size_t)n * K + kk]) *
                    (double)hS[(size_t)(n >> 7) * KB + (kk >> 7)];
        const double got = bf2f(hC[(size_t)m * N + n]);
        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    /* Row 0 against the decode kernel, every column. Two f32 reduction orders, so this is a
     * numeric-agreement bound, not bit-identity. */
    double xworst = 0.0;
    for (unsigned n = 0; n < N; n++) {
        const double g = bf2f(hC[n]), v = bf2f(hCv[n]);
        const double rel = fabs(g - v) / (fabs(v) + 1e-2);
        if (rel > xworst) xworst = rel;
    }
    const int ok = worst < 3e-2 && xworst < 3e-2;
    printf("  %-24s M=%5u N=%5u K=%5u  %s  ref %.4f  vs-gemv %.4f\n", label, M, N, K,
           ok ? "PASS" : "FAIL", worst, xworst);
    if (!ok) fails++;

    plow_hsa_free(h, dW); plow_hsa_free(h, dA); plow_hsa_free(h, dS);
    plow_hsa_free(h, dC); plow_hsa_free(h, dCv);
}

/* Block-fp8 MoE expert gate/up + down for ONE expert (slot 0), through the real table indirection.
 * fu = act(gate·x)*(up·x) ; part = gate_weight · (down·fu). Validates the fp8 expert decode path. */
static void run_expert(plow_hsa* h, plow_hsa_kernel* kglu, plow_hsa_kernel* kdown, unsigned NCU,
                       unsigned I_moe, unsigned H) {
    const unsigned act = 1u;              /* silu (SwiGLU — GLM) */
    const float gate_w = 0.7f;            /* the router gate weight for this slot */
    const unsigned IB = (I_moe + 127u) / 128u, HB = (H + 127u) / 128u;

    /* Wg,Wu : [I_moe][H] fp8 + [IB][HB] scale.  Wd : [H][I_moe] fp8 + [HB][IB] scale. */
    unsigned char* hWg = plow_hsa_alloc_host(h, (size_t)I_moe * H);
    unsigned char* hWu = plow_hsa_alloc_host(h, (size_t)I_moe * H);
    unsigned char* hWd = plow_hsa_alloc_host(h, (size_t)H * I_moe);
    float* hSg = plow_hsa_alloc_host(h, (size_t)IB * HB * 4);
    float* hSu = plow_hsa_alloc_host(h, (size_t)IB * HB * 4);
    float* hSd = plow_hsa_alloc_host(h, (size_t)HB * IB * 4);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)H * 2);
    bf16* hfu = plow_hsa_alloc_host(h, (size_t)I_moe * 2);
    float* hpart = plow_hsa_alloc_host(h, (size_t)H * 4);
    for (size_t i = 0; i < (size_t)I_moe * H; i++) {
        hWg[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
        hWu[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    }
    for (size_t i = 0; i < (size_t)H * I_moe; i++)
        hWd[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    for (size_t i = 0; i < (size_t)IB * HB; i++) { hSg[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f; hSu[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f; }
    for (size_t i = 0; i < (size_t)HB * IB; i++) hSd[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
    for (unsigned i = 0; i < H; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dWg = plow_hsa_alloc(h, 0, (size_t)I_moe * H); plow_hsa_copy_h2d(h, 0, dWg, hWg, (size_t)I_moe * H);
    void* dWu = plow_hsa_alloc(h, 0, (size_t)I_moe * H); plow_hsa_copy_h2d(h, 0, dWu, hWu, (size_t)I_moe * H);
    void* dWd = plow_hsa_alloc(h, 0, (size_t)H * I_moe); plow_hsa_copy_h2d(h, 0, dWd, hWd, (size_t)H * I_moe);
    void* dSg = plow_hsa_alloc(h, 0, (size_t)IB * HB * 4); plow_hsa_copy_h2d(h, 0, dSg, hSg, (size_t)IB * HB * 4);
    void* dSu = plow_hsa_alloc(h, 0, (size_t)IB * HB * 4); plow_hsa_copy_h2d(h, 0, dSu, hSu, (size_t)IB * HB * 4);
    void* dSd = plow_hsa_alloc(h, 0, (size_t)HB * IB * 4); plow_hsa_copy_h2d(h, 0, dSd, hSd, (size_t)HB * IB * 4);
    void* dx = plow_hsa_alloc(h, 0, (size_t)H * 2); plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)H * 2);
    void* dfu = plow_hsa_alloc(h, 0, (size_t)I_moe * 2);
    void* dpart = plow_hsa_alloc(h, 0, (size_t)H * 4);

    /* routing table slot 0 = {expert_id 0, gate gate_w} */
    /* Small pointer tables: use plow_hsa_upload (stages arbitrary host memory); copy_h2d requires
     * alloc_host-pinned memory and would SDMA-fault on these stack arrays. */
    unsigned char htab[8]; *(unsigned*)htab = 0u; *(float*)(htab + 4) = gate_w;
    void* dtab = plow_hsa_alloc(h, 0, 8); plow_hsa_upload(h, 0, dtab, htab, 8);
    uint64_t wtab[3] = {(uint64_t)(uintptr_t)dWg, (uint64_t)(uintptr_t)dWu, (uint64_t)(uintptr_t)dWd};
    uint64_t stab[3] = {(uint64_t)(uintptr_t)dSg, (uint64_t)(uintptr_t)dSu, (uint64_t)(uintptr_t)dSd};
    void* dwtab = plow_hsa_alloc(h, 0, sizeof(wtab)); plow_hsa_upload(h, 0, dwtab, wtab, sizeof(wtab));
    void* dstab = plow_hsa_alloc(h, 0, sizeof(stab)); plow_hsa_upload(h, 0, dstab, stab, sizeof(stab));

    struct __attribute__((packed)) {
        void* fu; const void* x; const void* table; const void* wtab; const void* stab;
        unsigned i_moe, h, n_exp, act;
    } aglu = {dfu, dx, dtab, dwtab, dstab, I_moe, H, 1u, act};
    plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hfu, dfu, (size_t)I_moe * 2);

    /* Perf profile: gate+up stream 2*I_moe*H fp8 bytes; down streams H*I_moe. */
    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    const double glu_ms = (now() - t0) * 1e3 / ITERS;
    const double glu_tbs = (double)2 * I_moe * H / (glu_ms * 1e-3) / 1e12;

    struct __attribute__((packed)) {
        void* part; const void* fu; const void* table; const void* wtab; const void* stab;
        unsigned h, i_moe, n_exp;
    } adown = {dpart, dfu, dtab, dwtab, dstab, H, I_moe, 1u};
    plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hpart, dpart, (size_t)H * 4);

    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    const double down_ms = (now() - t0) * 1e3 / ITERS;
    const double down_tbs = (double)H * I_moe / (down_ms * 1e-3) / 1e12;

    /* Reference: fu[n]=silu(Σ_k x·decode(Wg)·Sg[n/128][k/128]) * (Σ_k x·decode(Wu)·Su);
     *            part[h]=gate_w·Σ_i fu·decode(Wd)·Sd[h/128][i/128]. */
    double worst_fu = 0.0, worst_p = 0.0;
    for (unsigned n = 0; n < I_moe; n++) {
        double g = 0.0, u = 0.0;
        for (unsigned kk = 0; kk < H; kk++) {
            const double xv = bf2f(hx[kk]);
            g += xv * e4m3_decode(hWg[(size_t)n * H + kk]) * (double)hSg[(size_t)(n >> 7) * HB + (kk >> 7)];
            u += xv * e4m3_decode(hWu[(size_t)n * H + kk]) * (double)hSu[(size_t)(n >> 7) * HB + (kk >> 7)];
        }
        const double silu = g / (1.0 + exp(-g));
        const double want = silu * u;
        const double rel = fabs(bf2f(hfu[n]) - want) / (fabs(want) + 1e-2);
        if (rel > worst_fu) worst_fu = rel;
    }
    for (unsigned hh = 0; hh < H; hh++) {
        double y = 0.0;
        for (unsigned ii = 0; ii < I_moe; ii++)
            y += (double)bf2f(hfu[ii]) * e4m3_decode(hWd[(size_t)hh * I_moe + ii]) *
                 (double)hSd[(size_t)(hh >> 7) * IB + (ii >> 7)];
        const double want = (double)gate_w * y;
        const double rel = fabs(hpart[hh] - want) / (fabs(want) + 1e-2);
        if (rel > worst_p) worst_p = rel;
    }
    const int ok = worst_fu < 3e-2 && worst_p < 3e-2;
    printf("  moe expert I_moe=%u H=%u  %s (glu rel %.4f, down rel %.4f)\n", I_moe, H,
           ok ? "PASS" : "FAIL", worst_fu, worst_p);
    printf("    gate+up (fused)  %.3f ms  %.2f TB/s   |   down  %.3f ms  %.2f TB/s\n",
           glu_ms, glu_tbs, down_ms, down_tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dWg); plow_hsa_free(h, dWu); plow_hsa_free(h, dWd);
    plow_hsa_free(h, dSg); plow_hsa_free(h, dSu); plow_hsa_free(h, dSd);
    plow_hsa_free(h, dx); plow_hsa_free(h, dfu); plow_hsa_free(h, dpart);
    plow_hsa_free(h, dtab); plow_hsa_free(h, dwtab); plow_hsa_free(h, dstab);
}

/* Block-fp8 DENSE MLP gate/up (fused SwiGLU) on NAMED weights (no expert table) — GLM dense
 * layers 0-2. fu[n] = silu(gate_n·x) * (up_n·x). Validates the dense-GLU decode path (op 47). */
static void run_dense_glu(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, unsigned N, unsigned K) {
    const unsigned act = 1u; /* silu (SwiGLU) */
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    unsigned char* hWg = plow_hsa_alloc_host(h, (size_t)N * K);
    unsigned char* hWu = plow_hsa_alloc_host(h, (size_t)N * K);
    float* hSg = plow_hsa_alloc_host(h, (size_t)NB * KB * 4);
    float* hSu = plow_hsa_alloc_host(h, (size_t)NB * KB * 4);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)K * 2);
    bf16* hfu = plow_hsa_alloc_host(h, (size_t)N * 2);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        hWg[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
        hWu[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    }
    for (size_t i = 0; i < (size_t)NB * KB; i++) {
        hSg[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
        hSu[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
    }
    for (unsigned i = 0; i < K; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dWg = plow_hsa_alloc(h, 0, (size_t)N * K); plow_hsa_copy_h2d(h, 0, dWg, hWg, (size_t)N * K);
    void* dWu = plow_hsa_alloc(h, 0, (size_t)N * K); plow_hsa_copy_h2d(h, 0, dWu, hWu, (size_t)N * K);
    void* dSg = plow_hsa_alloc(h, 0, (size_t)NB * KB * 4); plow_hsa_copy_h2d(h, 0, dSg, hSg, (size_t)NB * KB * 4);
    void* dSu = plow_hsa_alloc(h, 0, (size_t)NB * KB * 4); plow_hsa_copy_h2d(h, 0, dSu, hSu, (size_t)NB * KB * 4);
    void* dx = plow_hsa_alloc(h, 0, (size_t)K * 2); plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)K * 2);
    void* dfu = plow_hsa_alloc(h, 0, (size_t)N * 2);

    struct __attribute__((packed)) {
        void* fu; const void* x; const void* wg; const void* wu; const void* sg; const void* su;
        unsigned n, kk, act;
    } args = {dfu, dx, dWg, dWu, dSg, dSu, N, K, act};
    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hfu, dfu, (size_t)N * 2);

    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double ms = (now() - t0) * 1e3 / ITERS;
    const double tbs = (double)2 * N * K / (ms * 1e-3) / 1e12;

    double worst = 0.0;
    for (unsigned n = 0; n < N; n++) {
        double g = 0.0, u = 0.0;
        for (unsigned kk = 0; kk < K; kk++) {
            const double xv = bf2f(hx[kk]);
            g += xv * e4m3_decode(hWg[(size_t)n * K + kk]) * (double)hSg[(size_t)(n >> 7) * KB + (kk >> 7)];
            u += xv * e4m3_decode(hWu[(size_t)n * K + kk]) * (double)hSu[(size_t)(n >> 7) * KB + (kk >> 7)];
        }
        const double want = (g / (1.0 + exp(-g))) * u;
        const double rel = fabs(bf2f(hfu[n]) - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    const int ok = worst < 3e-2;
    printf("  dense g/u %u->%u  %s rel %.4f | %.3f ms  %.2f TB/s\n", K, N,
           ok ? "PASS" : "FAIL", worst, ms, tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dWg); plow_hsa_free(h, dWu); plow_hsa_free(h, dSg);
    plow_hsa_free(h, dSu); plow_hsa_free(h, dx); plow_hsa_free(h, dfu);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    setbuf(stdout, NULL);
    srand(1234);
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    FILE* f = fopen(elf, "rb");
    if (!f) { perror(elf); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "gemv_fp8_blk", &k) != 0) {
        fprintf(stderr, "no kernel gemv_fp8_blk: %s\n", plow_hsa_last_error()); return 1;
    }

    const unsigned NCU = 64;
    printf("block-fp8 decode GEMV (weight_block_size [128,128])   [N=out, K=in]:\n");
    /* Real GLM-5.2-FP8 decode GEMV shapes (hidden 6144). N=out-channels, K=in. All 128-multiples
     * except kv_a's N=576 (NB=ceil(576/128)=5). See plans/glm52-arch.md. */
    /* --- attention projections --- */
    run(h, &k, NCU, "q_a  6144->2048",   2048,  6144);
    run(h, &k, NCU, "q_b  2048->16384", 16384,  2048);
    run(h, &k, NCU, "kv_a 6144->576",     576,  6144);
    run(h, &k, NCU, "kv_b 512->28672",  28672,   512);  /* narrow-K, CU-starved */
    run(h, &k, NCU, "o    16384->6144",  6144, 16384);
    /* --- dense-MLP (layers 0-2) --- */
    run(h, &k, NCU, "dense g/u 6144->12288", 12288, 6144);
    run(h, &k, NCU, "dense down 12288->6144", 6144, 12288);
    /* --- MoE expert / shared expert (per expert) --- */
    run(h, &k, NCU, "moe g/u 6144->2048",  2048, 6144);
    run(h, &k, NCU, "moe down 2048->6144",  6144, 2048); /* narrow-K, CU-starved */
    /* Ragged shape: not a 128-multiple in either dim — exercises ceil() blocks + overshoot clamp. */
    run(h, &k, NCU, "ragged (ceil+clamp)", 130, 260);

    /* Block-fp8 MoE expert path (GLM-5.2: I_moe=2048, H=6144). */
    plow_hsa_kernel kglu, kdown;
    if (plow_hsa_get_kernel(h, 0, "moe_expert_glu_fp8_blk_k", &kglu) == 0 &&
        plow_hsa_get_kernel(h, 0, "moe_expert_down_fp8_blk_k", &kdown) == 0)
        run_expert(h, &kglu, &kdown, NCU, 2048, 6144);
    else { printf("  no expert kernels\n"); fails++; }

    /* Block-fp8 DENSE MLP gate/up (op 47), GLM-5.2 dense layers 0-2: gate/up 6144->12288 fused. */
    plow_hsa_kernel kdglu;
    if (plow_hsa_get_kernel(h, 0, "dense_glu_fp8_blk_k", &kdglu) == 0)
        run_dense_glu(h, &kdglu, NCU, 12288, 6144);
    else { printf("  no dense-glu kernel\n"); fails++; }

    /* Block-fp8 DENSE PREFILL GEMM (op 107) — the T-row arm GLM_LINEAR_FP8 was blocked on.
     * These are exactly the four projections the knob re-declares, at both TP degrees the campaign
     * runs, plus a ragged N to exercise the ceil() column tail. K is always a 64-multiple because
     * the kernel is instantiated KEXACT and the emitter refuses anything else. */
    plow_hsa_kernel kgblk;
    if (plow_hsa_get_kernel(h, 0, "d_gemm_fp8_blk_k", &kgblk) == 0) {
        printf("block-fp8 DENSE PREFILL GEMM (op 107) vs f64 ref AND vs the decode GEMV:\n");
        /* TP4 (nh_l=16, v_head=256 -> o K=4096; imoe_l=512). M = the emitted bucket ladder. */
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP4",        512, 6144, 4096);
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP4 M=2048", 2048, 6144, 4096);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared gate TP4",    512,  512, 6144);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared down TP4",    512, 6144,  512);
        /* TP8 (nh_l=8 -> o K=2048; imoe_l=256). */
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP8",         512, 6144, 2048);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared gate TP8",    512,  256, 6144);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared down TP8",    512, 6144,  256);
        /* Ragged M and N (K stays a 64-multiple): tile tails on both output axes at once. */
        run_gemm_blk(h, &kgblk, &k, NCU, "ragged M,N tails",   100,  130,  256);
    } else { printf("  no d_gemm_fp8_blk_k kernel\n"); fails++; }

    printf(fails ? "FAIL (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
