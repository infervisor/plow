/* mla_prefill_bench_gfx950.c — MLA PREFILL: the decode-body wrapper vs the tiled MFMA
 * kernel, at Kimi-K3's shapes.                                          [DEEPSEEK-MLA]
 *
 * `mla_gfx950_test.c` answers "is the tiled kernel right". This answers "is it worth it",
 * and it is a separate binary for the usual reason: the gate wants a small fixture and
 * exact comparisons, a bench wants big shapes and no host round trips between launches.
 *
 * WHAT IT SWEEPS AND WHY THOSE NUMBERS. K3 at TP8 gives n_head_local = 12, kv_lora = 512,
 * qk_rope = 64. Its prefill bucket ladder tops out at T=8192 and the blob is emitted at
 * --max-ctx 32768, so (ctx, n_tok) walks the four 8192-token chunks a 32k prompt actually
 * runs, plus the small-T tail chunks. Both kernels get the same buffers and the same grid
 * (n_cu workgroups), because that is how the interpreter dispatches them.
 *
 * The COST MODEL in the report is the causal-triangle one, counted over the rows this
 * launch really attends: for chunk row t at absolute qpos = ctx-T+t, the score is
 * (DK+DR) MACs and the PV is DK MACs, per head, per kv position in [0, qpos]. That is the
 * work both kernels must do; the difference is only which units do it and how many times
 * the latent crosses the memory system.
 *
 * Weights are UNBOUND — random bytes. The schedule and the timing are real, the numbers
 * are not, which is all a kernel bench needs. No correctness claim is made here.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;

static plow_hsa* H;
static void* dev(size_t b) { return plow_hsa_alloc(H, 0, b); }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* A cheap deterministic bf16 fill. Values land in ~[-1,1) so the softmax sees a spread
 * rather than a saturated max — a constant fill would make every score equal and let the
 * exp() path run at an unrepresentative speed. */
static void fill_bf16(bf16* p, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        /* bf16 exponent 0x3f80 is 1.0; walk the low bits of the exponent and mantissa. */
        const uint32_t sign = (s >> 31) << 15;
        const uint32_t exp = 0x3b00u + ((s >> 20) & 0x0300u);
        p[i] = (bf16)(sign | exp | ((s >> 8) & 0x7fu));
    }
}

struct pf_args {
    void *op, *ml;
    const void *qa, *qr, *ckv, *kr, *len;
    unsigned n_batch, n_tok, n_head, kv_stride, window;
    float scale;
} __attribute__((packed));

int main(int argc, char** argv) {
    const unsigned n_head = (unsigned)(argc > 1 ? atoi(argv[1]) : 12);
    const unsigned iters = (unsigned)(argc > 2 ? atoi(argv[2]) : 5);
    const unsigned DK = 512, DR = 64;

    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, ldsb = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &ldsb);
    printf("dev0: %s  CUs=%u  LDS=%u B   n_head=%u  iters=%u\n", nm, cus, ldsb, n_head, iters);

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    plow_hsa_kernel ks, km, ks8, km8;
    if (plow_hsa_get_kernel(H, 0, "mla_flash_prefill_512", &ks) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_mfma_512", &km) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_fp8_512", &ks8) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_mfma_fp8_512", &km8)) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
    }

    /* The four 8192-row chunks of a 32k prompt, then the tail-chunk sizes. `ctx` is the
     * context INCLUDING this chunk, which is what kv_len means to both kernels. */
    /* The last four straddle `mla_pf_tiled_fills`: at n_head=12 and nblk=256 the rule takes
     * the tiled kernel from n_tok=704 (132 items) up and the scalar body below it. */
    static const unsigned CTX[] = {8192, 16384, 24576, 32768, 2048, 32768,
                                   32768, 32768, 32768, 32768};
    static const unsigned TOK[] = {8192, 8192, 8192, 8192, 2048, 1024,
                                   704, 640, 512, 128};
    const unsigned NC = sizeof(CTX) / sizeof(CTX[0]);

    unsigned max_ctx = 0, max_tok = 0;
    for (unsigned i = 0; i < NC; i++) {
        if (CTX[i] > max_ctx) max_ctx = CTX[i];
        if (TOK[i] > max_tok) max_tok = TOK[i];
    }
    const size_t nckv = (size_t)max_ctx * DK, nkr = (size_t)max_ctx * DR;
    const size_t nqa = (size_t)max_tok * n_head * DK, nqr = (size_t)max_tok * n_head * DR;
    const size_t nop = (size_t)max_tok * n_head * DK, nml = (size_t)max_tok * n_head * 2;

    bf16* h = plow_hsa_alloc_host(H, (nckv + nkr + nqa + nqr) * 2);
    fill_bf16(h, nckv + nkr + nqa + nqr, 12345u);
    void* dCkv = dev(nckv * 2);
    void* dKr = dev(nkr * 2);
    void* dQa = dev(nqa * 2);
    void* dQr = dev(nqr * 2);
    plow_hsa_copy_h2d(H, 0, dCkv, h, nckv * 2);
    plow_hsa_copy_h2d(H, 0, dKr, h + nckv, nkr * 2);
    plow_hsa_copy_h2d(H, 0, dQa, h + nckv + nkr, nqa * 2);
    plow_hsa_copy_h2d(H, 0, dQr, h + nckv + nkr + nqa, nqr * 2);
    void* dOp = dev(nop * 4);
    void* dMl = dev(nml * 4);
    int* pLen = plow_hsa_alloc_host(H, 4);
    void* dLen = dev(4);

    /* The fp8 arm — op 110, what K3 actually dispatches. Same latent, half the bytes, and
     * a scale strip per row. Contents are irrelevant to timing; only the widths are. */
    unsigned char* h8 = plow_hsa_alloc_host(H, nckv);
    for (size_t i = 0; i < nckv; i++) h8[i] = (unsigned char)(0x30u | (i & 7u));
    void* dC8 = dev(nckv);
    plow_hsa_copy_h2d(H, 0, dC8, h8, nckv);
    float* hS = plow_hsa_alloc_host(H, (size_t)2 * max_ctx * 4);
    for (size_t i = 0; i < (size_t)2 * max_ctx; i++) hS[i] = 1.0f;
    void* dS = dev((size_t)2 * max_ctx * 4);
    plow_hsa_copy_h2d(H, 0, dS, hS, (size_t)2 * max_ctx * 4);

    struct pf8_args {
        void *op, *ml;
        const void *qa, *qr, *ckv, *kr, *len;
        unsigned n_batch, n_tok, n_head, kv_stride, window;
        float scale;
        const void* kvs;
        unsigned krot;
    } __attribute__((packed));

    printf("\nbf16 latent (ops 51)\n");
    printf("  %6s %6s | %11s %11s %7s | %9s %9s\n", "ctx", "n_tok", "scalar ms",
           "tiled ms", "speedup", "sc TF/s", "ti TF/s");
    for (unsigned c = 0; c < NC; c++) {
        const unsigned ctx = CTX[c], T = TOK[c];
        pLen[0] = (int)ctx;
        plow_hsa_copy_h2d(H, 0, dLen, pLen, 4);
        struct pf_args a = {dOp, dMl, dQa, dQr, dCkv, dKr, dLen, 1, T, n_head, ctx, 0,
                            0.0883883f};

        /* Causal MACs actually attended by THIS launch's rows. */
        double kvpairs = 0.0;
        for (unsigned t = 0; t < T; t++) kvpairs += (double)(ctx - T + t + 1);
        const double flop = kvpairs * (double)n_head * (double)(DK + DR + DK) * 2.0;

        double best[2] = {1e30, 1e30};
        for (int which = 0; which < 2; which++) {
            plow_hsa_kernel* k = which ? &km : &ks;
            for (unsigned it = 0; it < iters + 1; it++) {
                const double t0 = now_s();
                if (plow_hsa_launch(H, 0, k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &a, sizeof(a)) != 0) {
                    fprintf(stderr, "launch: %s\n", plow_hsa_last_error());
                    return 1;
                }
                plow_hsa_wait(H, 0);
                const double dt = now_s() - t0;
                if (it && dt < best[which]) best[which] = dt; /* it 0 warms the caches */
            }
        }
        printf("  %6u %6u | %11.3f %11.3f %6.2fx | %9.1f %9.1f\n", ctx, T, best[0] * 1e3,
               best[1] * 1e3, best[0] / best[1], flop / best[0] / 1e12, flop / best[1] / 1e12);
        fflush(stdout);
    }

    printf("\nfp8 latent (op 110 — the arm Kimi-K3 dispatches)\n");
    printf("  %6s %6s | %11s %11s %7s | %9s %9s\n", "ctx", "n_tok", "scalar ms",
           "tiled ms", "speedup", "sc TF/s", "ti TF/s");
    for (unsigned c = 0; c < NC; c++) {
        const unsigned ctx = CTX[c], T = TOK[c];
        pLen[0] = (int)ctx;
        plow_hsa_copy_h2d(H, 0, dLen, pLen, 4);
        struct pf8_args a = {dOp, dMl, dQa,  dQr, dC8, dKr, dLen, 1, T, n_head, ctx, 0,
                             0.0883883f, dS, 0};
        double kvpairs = 0.0;
        for (unsigned t = 0; t < T; t++) kvpairs += (double)(ctx - T + t + 1);
        const double flop = kvpairs * (double)n_head * (double)(DK + DR + DK) * 2.0;

        double best[2] = {1e30, 1e30};
        for (int which = 0; which < 2; which++) {
            plow_hsa_kernel* k = which ? &km8 : &ks8;
            for (unsigned it = 0; it < iters + 1; it++) {
                const double t0 = now_s();
                if (plow_hsa_launch(H, 0, k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &a, sizeof(a)) != 0) {
                    fprintf(stderr, "launch: %s\n", plow_hsa_last_error());
                    return 1;
                }
                plow_hsa_wait(H, 0);
                const double dt = now_s() - t0;
                if (it && dt < best[which]) best[which] = dt;
            }
        }
        printf("  %6u %6u | %11.3f %11.3f %6.2fx | %9.1f %9.1f\n", ctx, T, best[0] * 1e3,
               best[1] * 1e3, best[0] / best[1], flop / best[0] / 1e12, flop / best[1] / 1e12);
        fflush(stdout);
    }

    plow_hsa_shutdown(H);
    return 0;
}
