/* gemm_tile_sweep.c — sweep EVERY compiled gemm_cN tile over one arbitrary (M,N,K).
 *
 * gemm_bench_8k.c hard-codes the Qwen prefill shapes, so it cannot answer "which tile is
 * best for the Gemma-26B router (128x128x2816)". This takes the shape on argv and sweeps
 * every tile symbol that test_kernels.elf actually exports, so a tile added to
 * test_kernels.hip appears here with no edit.
 *
 *   usage: gemm_tile_sweep <M> <N> <K> [label] [quant]
 *
 * `quant` is `None` (bf16, the default and what every prior campaign measured) or `Mxfp4`
 * (w4a16: bf16 activations against packed-2/byte e2m1 weights with one E8M0 scale per 32 K).
 * It is a SEPARATE ladder, not a flag on the same one — a bf16 timing and an mxfp4 timing move
 * different numbers of weight bytes, which is exactly why `tunedb::gemm_op_case` puts the quant
 * in the key. The JSONL carries it so ingest files them apart.
 *
 * With PLOW_GEMM_JSONL=<path> it also appends one raw-sample row per tile, which
 * `tunedb-gemm ingest` turns into qualified `kernel_measurement` records. The C side
 * deliberately writes SAMPLES and a correctness verdict and nothing else: the build digests
 * that decide staleness come from probing the interpreter, which is the Rust side's job.
 *
 * It also prints the TILE COUNT and the CU FILL each tile achieves, because that — not the
 * tile's own efficiency — is what the M=128 shapes are actually limited by.
 *
 * Peak: 1660 TF/s sustained bf16 MFMA (256 CU @ ~1.58 GHz dense, op_gemm.h). HBM 6200 GB/s.
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

/* OVERRIDABLE, because these two constants are gfx950/MI355X numbers and this harness also runs
 * on gfx942. On MI300X the honest denominators are the MEASURED pure-MFMA issue ceiling
 * (937 TF/s bf16 -- runtime/ubench/GEMM_MFMA_SHAPE_VERDICT.md, not the 1307 datasheet peak) and
 * 5300 GB/s of HBM3. Printing 1660/6200 there overstates the machine by ~1.77x on compute and
 * ~1.17x on bandwidth. Defaults unchanged, so every published gfx950 record still reproduces. */
#ifndef PEAK_TFLOPS
#define PEAK_TFLOPS 1660.0
#endif
#ifndef HBM_GBPS
#define HBM_GBPS 6200.0
#endif

/* Keep in sync with test_kernels.hip's GEMM_VARIANT list. Symbol lookup failing is the
 * signal that a variant is not compiled, so an entry costs nothing when absent. */
static const struct { const char* sym; unsigned bm, bn; } TILES[] = {
    {"gemm_c0", 256, 256}, {"gemm_c1", 256, 128}, {"gemm_c2", 128, 256},
    {"gemm_c3", 128, 128}, {"gemm_c4", 64, 128},  {"gemm_c5", 192, 256},
    {"gemm_c6", 320, 128}, {"gemm_c7", 384, 128}, {"gemm_c8", 128, 384},
    {"gemm_c9", 192, 128}, {"gemm_c10", 64, 256}, {"gemm_c11", 256, 384},
};
#define NTILES ((int)(sizeof TILES / sizeof TILES[0]))

/* The MXFP4 (w4a16) ladder: the FIVE SELECTABLE rungs and nothing else.
 *
 * Deliberately not the twelve-entry bf16 list. `tunedb::gemm_rung_opcode` maps exactly these
 * five tiles to `Gemm*Mxfp4` opcodes; the calibration-only tiles (320x128, 384x128, 128x384,
 * 192x128) have no mxfp4 dispatch arm, so measuring them would produce rows ingest can only
 * throw away. `test_kernels.hip`'s GEMM_MXFP4_VARIANT list is the other half of this pair. */
static const struct { const char* sym; unsigned bm, bn; } MXTILES[] = {
    {"gemm_mxfp4_c0", 256, 256}, {"gemm_mxfp4_c2", 128, 256}, {"gemm_mxfp4_c3", 128, 128},
    {"gemm_mxfp4_c4", 64, 128},  {"gemm_mxfp4_c5", 192, 256},
};
#define NMXTILES ((int)(sizeof MXTILES / sizeof MXTILES[0]))

/* OCP e2m1: three magnitude bits on the ladder 0,0.5,1,1.5,2,3,4,6 and a sign bit. This is the
 * HOST-side twin of `amd_common.h`'s `fp4_to_bf16v8`, and it is exact — every code is a small
 * dyadic rational and the E8M0 scale is a power of two — so the f64 reference below is an exact
 * dot product of the same values the kernel converts, not an approximation of them. */
static const double FP4[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

/* E8M0 byte -> 2^(b-127). `ldexp` rather than a shift-into-the-exponent bit trick because this
 * is host code with no reason to be clever. */
static double e8m0(unsigned char b) { return ldexp(1.0, (int)b - 127); }

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <M> <N> <K> [label] [quant: None|Mxfp4]\n", argv[0]);
        return 2;
    }
    const unsigned M = (unsigned)atoi(argv[1]), N = (unsigned)atoi(argv[2]),
                   K = (unsigned)atoi(argv[3]);
    const char* label = argc > 4 ? argv[4] : "shape";
    const char* quant = argc > 5 ? argv[5] : "None";
    const int mx = strcmp(quant, "Mxfp4") == 0;
    if (!mx && strcmp(quant, "None") != 0) {
        fprintf(stderr, "unknown quant %s (want None or Mxfp4)\n", quant);
        return 2;
    }
    /* REFUSED rather than measured wrong. `d_gemm_t<WFP4>` static_asserts KEXACT, its weight
     * load reads a whole u32 of 8 fp4, and its scale row is K/32 bytes — a K that is not a
     * multiple of 64 would read past the row or split an MX block, and the kernel has no
     * predicate for either. Every prefill K plow emits is a multiple of 128. */
    if (mx && K % 64u) {
        fprintf(stderr, "mxfp4 needs K %% 64 == 0 (K=%u): the w4a16 B-fetch is KEXACT and its "
                        "MX block is 32 elements\n",
                K);
        return 2;
    }

    plow_hsa* H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64];
    uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    const unsigned NCU = cus, THREADS = 512;

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error());
        return 1;
    }

    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    /* mxfp4 weights are packed 2/byte with one E8M0 byte per 32 K — a QUARTER and a
     * thirty-second of the bf16 stream. That ratio is the whole reason the rung exists. */
    const size_t nW = nB / 2, nS = (size_t)N * (K / 32);
    bf16* hA = plow_hsa_alloc_host(H, nA * 2);
    bf16* hB = mx ? NULL : plow_hsa_alloc_host(H, nB * 2);
    unsigned char* hW = mx ? plow_hsa_alloc_host(H, nW) : NULL;
    unsigned char* hS = mx ? plow_hsa_alloc_host(H, nS) : NULL;
    bf16* hC = plow_hsa_alloc_host(H, nC * 2);
    srand(5);
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    if (mx) {
        for (size_t i = 0; i < nW; i++) hW[i] = (unsigned char)(rand() & 0xff);
        /* Scales 2^-4..2^-2, VARIED per block rather than pinned at 2^0. A constant scale would
         * pass the spot-check even if the kernel dropped the scale fetch entirely, which is the
         * one thing about this path that is new. Varying it keeps |w| <= 1.5, in the same range
         * as the bf16 operand, so the f32 accumulator sees the same magnitudes either way. */
        for (size_t i = 0; i < nS; i++) hS[i] = (unsigned char)(127 - 4 + (int)(i % 3));
    } else {
        for (size_t i = 0; i < nB; i++) hB[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    }
    void* dA = plow_hsa_alloc(H, 0, nA * 2);
    void* dB = plow_hsa_alloc(H, 0, mx ? nW : nB * 2);
    void* dS = mx ? plow_hsa_alloc(H, 0, nS) : NULL;
    void* dC = plow_hsa_alloc(H, 0, nC * 2);
    plow_hsa_copy_h2d(H, 0, dA, hA, nA * 2);
    if (mx) {
        plow_hsa_copy_h2d(H, 0, dB, hW, nW);
        plow_hsa_copy_h2d(H, 0, dS, hS, nS);
    } else {
        plow_hsa_copy_h2d(H, 0, dB, hB, nB * 2);
    }
    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; unsigned m, n, kk;
    } args = {dC, dA, dB, M, N, K};
    /* The mxfp4 kernels take the scale pointer between B and M, so the kernarg layout differs
     * and a separate struct is the honest way to say so. */
    struct __attribute__((packed)) {
        void* c; const void* a; const void* b; const void* s; unsigned m, n, kk;
    } mxargs = {dC, dA, dB, dS, M, N, K};
    const void* kargs = mx ? (const void*)&mxargs : (const void*)&args;
    const size_t kargs_sz = mx ? sizeof mxargs : sizeof args;

    /* The weight stream is the memory-bound floor for these shapes: at M=128 the B operand
     * dominates and A/C are noise, so B bytes / 6200 GB/s is the achievable wall time. */
    const double flops = 2.0 * (double)M * N * K;
    const double wbytes = mx ? (double)(nW + nS) : 2.0 * (double)N * K;
    const double mem_floor_ms = wbytes / (HBM_GBPS * 1e9) * 1e3;
    printf("%s  %u CUs\n", nm, NCU);
    printf("%s  M=%u N=%u K=%u   %.1f MFLOP  weights %.2f MB  HBM floor %.4f ms "
           "(= %.1f TF/s equiv, %.1f%% of MFMA peak)\n\n",
           label, M, N, K, flops / 1e6, wbytes / 1e6, mem_floor_ms,
           flops / (mem_floor_ms * 1e-3) / 1e12,
           100.0 * flops / (mem_floor_ms * 1e-3) / 1e12 / PEAK_TFLOPS);
    printf("  %-10s %-10s %6s %6s   %9s %9s %7s %7s\n", "tile", "BMxBN", "tiles", "fill",
           "ms", "TF/s", "%peak", "%hbm");

    const char* jsonl = getenv("PLOW_GEMM_JSONL");
    FILE* jf = jsonl ? fopen(jsonl, "a") : NULL;

    const int ntiles = mx ? NMXTILES : NTILES;
    for (int t = 0; t < ntiles; t++) {
        const char* sym = mx ? MXTILES[t].sym : TILES[t].sym;
        const unsigned tbm = mx ? MXTILES[t].bm : TILES[t].bm;
        const unsigned tbn = mx ? MXTILES[t].bn : TILES[t].bn;
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(H, 0, sym, &k) != 0) continue;
        /* 50 warm-up launches: the governor ramps sclk over tens of ms and an under-warmed
         * kernel reads slow, which silently re-ranks the sweep (gemm_bench_8k.c note). */
        for (int w = 0; w < 50; w++)
            plow_hsa_launch(H, 0, &k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, kargs, kargs_sz);
        plow_hsa_wait(H, 0);
        /* TIMED IN BATCHES OF 4, ten times, rather than one mean over 20.
         * `tunedb::Stats` requires >= 5 samples and refuses a win inside the noise, so a
         * single mean is not publishable — correctly, since it carries no dispersion. Each
         * sample is still an average of 4 launches so that per-launch jitter does not swamp
         * the short M=128 shapes. */
        const int groups = 10, per = 4, reps = groups * per;
        double sample_ns[16];
        for (int g = 0; g < groups; g++) {
            const double g0 = now();
            for (int r = 0; r < per; r++)
                plow_hsa_launch(H, 0, &k, NCU * THREADS, 1, 1, THREADS, 1, 1, 0, kargs,
                                kargs_sz);
            plow_hsa_wait(H, 0);
            sample_ns[g] = (now() - g0) / per * 1e9;
        }
        double sum = 0;
        for (int g = 0; g < groups; g++) sum += sample_ns[g];
        const double dt = sum / groups / 1e9;
        (void)reps;
        const double tf = flops / dt / 1e12;

        plow_hsa_copy_d2h(H, 0, hC, dC, nC * 2);
        int bad = 0;
        for (int s = 0; s < 24; s++) {
            unsigned m = (unsigned)(rand() % (int)M), nn = (unsigned)(rand() % (int)N);
            double acc = 0;
            for (unsigned kk = 0; kk < K; kk++) {
                /* The SAME oracle either way — an f64 dot product over the values the kernel
                 * consumed — which is what lets one `GEMM_ORACLE` string key both ladders. For
                 * mxfp4 the B element is reconstructed from its nibble and block scale; both
                 * are exact, so this is the reference and not an estimate of it. */
                double b;
                if (mx) {
                    const size_t e = (size_t)nn * K + kk;
                    const unsigned char byte = hW[e >> 1];
                    const unsigned nib = (kk & 1u) ? (byte >> 4) : (byte & 0xfu);
                    b = FP4[nib] * e8m0(hS[(size_t)nn * (K / 32) + (kk >> 5)]);
                } else {
                    b = bf2f(hB[(size_t)nn * K + kk]);
                }
                acc += (double)bf2f(hA[(size_t)m * K + kk]) * b;
            }
            const double g = bf2f(hC[(size_t)m * N + nn]);
            if (fabs(g - acc) / (fabs(acc) + 1e-3) > 0.03) bad++;
        }

        const unsigned ntile = ((M + tbm - 1) / tbm) * ((N + tbn - 1) / tbn);
        char bxb[16];
        snprintf(bxb, sizeof bxb, "%ux%u", tbm, tbn);
        printf("  %-10s %-10s %6u %5.1f%%   %9.4f %9.1f %6.1f%% %6.1f%%  %s\n", sym,
               bxb, ntile, 100.0 * (ntile < NCU ? ntile : NCU) / NCU, dt * 1e3, tf,
               100.0 * tf / PEAK_TFLOPS, 100.0 * mem_floor_ms / (dt * 1e3),
               bad ? "MISMATCH!" : "ok");

        if (jf) {
            /* A FAILING spot-check is written too, marked failed. `tunedb` will not qualify
             * it, and keeping the negative is the point: a tile that is fast and wrong must
             * not be silently absent from the record. */
            fprintf(jf,
                    "{\"m\":%u,\"n\":%u,\"k\":%u,\"quant\":\"%s\",\"tile\":\"%ux%ux%u\","
                    "\"sym\":\"%s\",\"correct\":%s,\"samples_ns\":[",
                    M, N, K, quant, tbm, tbn, 64u, sym, bad ? "false" : "true");
            for (int g = 0; g < groups; g++)
                fprintf(jf, "%s%.1f", g ? "," : "", sample_ns[g]);
            fprintf(jf, "]}\n");
        }
    }
    if (jf) fclose(jf);
    plow_hsa_free(H, dA);
    plow_hsa_free(H, dB);
    if (dS) plow_hsa_free(H, dS);
    plow_hsa_free(H, dC);
    return 0;
}
