/* decode_bench_gfx942.c — plow decode-attention throughput on MI300X, measured the
 * same way the AITER baseline was measured:
 *
 *   * warm (buffers touched, kernel run once) before any timing
 *   * N dispatches enqueued back-to-back into ONE AQL queue with the barrier bit set,
 *     then a single wait — the packet processor runs them serially with no host in the
 *     loop, which is the HSA equivalent of the HIP-graph replay AITER was timed under
 *   * GB/s counts KV-CACHE BYTES READ only (2 * batch * ctx * n_kv * head_dim * esz),
 *     exactly as bench_attn_decode.py's kv_bytes() does
 *   * median of R reps
 *
 * The reported time is decode + merge, because plow's split-KV decode is not a complete
 * attention without the merge and AITER's paged_attention_v1 includes its own partition
 * reduction. `--parts` also prints the two separately.
 *
 * CSV to stdout: impl,tag,dtype,batch,ctx,n_q,n_kv,head_dim,nsplit,us_median,us_best,gbps
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
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static plow_hsa* H;
static int DEV = 0;
static uint32_t NCU = 0;

static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

static int cmpd(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Tile a host pattern across a large device buffer. Uninitialised VRAM is a fine
 * timing input but a terrible debugging one; this keeps the values bounded. */
static void* PAT = NULL;
static size_t PATSZ = 64u << 20;
static void fill_dev(void* d, size_t bytes) {
    if (!PAT) {
        PAT = plow_hsa_alloc_host(H, PATSZ);
        bf16* p = (bf16*)PAT;
        for (size_t i = 0; i < PATSZ / 2; i++)
            p[i] = f2bf(((float)((i * 1103515245u + 12345u) % 2048) / 2048.0f - 0.5f) * 0.3f);
    }
    for (size_t o = 0; o < bytes; o += PATSZ) {
        size_t n = bytes - o < PATSZ ? bytes - o : PATSZ;
        plow_hsa_copy_h2d(H, DEV, (char*)d + o, PAT, n);
    }
}

/* ---------------------------------------------------------------------------
 * Flash decode (Gemma 4): gemma_flash_decode_{256,512} + gemma_flash_merge_{256,512}
 * ------------------------------------------------------------------------- */
struct __attribute__((packed)) dec_args {
    void* op; void* ml; const void* q; const void* k; const void* v; const void* len;
    unsigned n_batch, n_head, n_kv_head, kv_stride, window; float scale; unsigned nsplit;
};
struct __attribute__((packed)) mrg_args {
    void* o; const void* op; const void* ml; unsigned n_batch, n_head, nsplit;
};

static int ITERS = 20, REPS = 5;
static int WANT_PARTS = 0;

/* Enqueue `iters` (decode, merge) pairs, one wait, return us/pair. */
static double time_pair(plow_hsa_kernel* kd, const struct dec_args* a, unsigned dwgs,
                        plow_hsa_kernel* km, const struct mrg_args* m, unsigned mwgs,
                        double* out_dec_only) {
    double ts[64];
    for (int r = 0; r < REPS; r++) {
        double t0 = now_us();
        for (int i = 0; i < ITERS; i++) {
            plow_hsa_launch(H, DEV, kd, dwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, a, sizeof(*a));
            if (km) plow_hsa_launch(H, DEV, km, mwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, m, sizeof(*m));
        }
        plow_hsa_wait(H, DEV);
        ts[r] = (now_us() - t0) / ITERS;
    }
    qsort(ts, REPS, sizeof(double), cmpd);
    if (out_dec_only) {
        double us[64];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_us();
            for (int i = 0; i < ITERS; i++)
                plow_hsa_launch(H, DEV, kd, dwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, a, sizeof(*a));
            plow_hsa_wait(H, DEV);
            us[r] = (now_us() - t0) / ITERS;
        }
        qsort(us, REPS, sizeof(double), cmpd);
        *out_dec_only = us[REPS / 2];
    }
    return ts[REPS / 2];
}

struct geom { const char* tag; int hd, nq, nkv; };

static void bench_flash(plow_hsa_kernel* kd, plow_hsa_kernel* km, const struct geom* g,
                        int batch, int ctx, const int* nsplits, int n_ns, double max_kv_gb) {
    const size_t kvb = (size_t)2 * batch * ctx * g->nkv * g->hd * 2;
    if (kvb > (size_t)(max_kv_gb * 1e9)) {
        printf("skip,%s,bf16,%d,%d,%d,%d,%d,0,nan,nan,nan,SKIP_kv=%.1fGB\n", g->tag, batch, ctx,
               g->nq, g->nkv, g->hd, kvb / 1e9);
        fflush(stdout);
        return;
    }
    const size_t nq = (size_t)batch * g->nq * g->hd;
    const size_t nkv1 = (size_t)batch * g->nkv * (size_t)ctx * g->hd;

    int max_ns = 1;
    for (int i = 0; i < n_ns; i++) if (nsplits[i] > max_ns) max_ns = nsplits[i];

    void* dQ = plow_hsa_alloc(H, DEV, nq * 2);
    void* dK = plow_hsa_alloc(H, DEV, nkv1 * 2);
    void* dV = plow_hsa_alloc(H, DEV, nkv1 * 2);
    void* dO = plow_hsa_alloc(H, DEV, nq * 2);
    void* dLen = plow_hsa_alloc(H, DEV, (size_t)batch * 4);
    void* dOp = plow_hsa_alloc(H, DEV, (size_t)batch * g->nq * max_ns * g->hd * 4);
    void* dMl = plow_hsa_alloc(H, DEV, (size_t)batch * g->nq * max_ns * 2 * 4);
    if (!dQ || !dK || !dV || !dO || !dLen || !dOp || !dMl) {
        printf("skip,%s,bf16,%d,%d,%d,%d,%d,0,nan,nan,nan,OOM\n", g->tag, batch, ctx, g->nq,
               g->nkv, g->hd);
        fflush(stdout);
        goto done;
    }
    fill_dev(dQ, nq * 2);
    fill_dev(dK, nkv1 * 2);
    fill_dev(dV, nkv1 * 2);
    {
        int* hl = plow_hsa_alloc_host(H, (size_t)batch * 4);
        for (int b = 0; b < batch; b++) hl[b] = ctx;
        plow_hsa_copy_h2d(H, DEV, dLen, hl, (size_t)batch * 4);
    }

    double best_us = 1e30, best_dec = 0; int best_ns = 0;
    for (int i = 0; i < n_ns; i++) {
        const unsigned ns = (unsigned)nsplits[i];
        struct dec_args a = {dOp, dMl, dQ, dK, dV, dLen, (unsigned)batch, (unsigned)g->nq,
                             (unsigned)g->nkv, (unsigned)ctx, 0u, 1.0f, ns};
        struct mrg_args m = {dO, dOp, dMl, (unsigned)batch, (unsigned)g->nq, ns};
        /* The merge decomposes over (batch, head) -- 16 work items at batch 1 -- and derives a
         * d-chunk axis from the workgroup count it is given (d_flash_merge's `dsplit`). Sweep it:
         * at batch 1 the merge is otherwise 16 of 304 CUs and costs as much as the decode. */
        const unsigned n_bh = (unsigned)batch * g->nq;
        for (int ds = 0; ds < 5; ds++) {
            const unsigned mwgs = (n_bh << ds) < NCU ? (n_bh << ds) : NCU;
            /* warm */
            plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
            plow_hsa_launch(H, DEV, km, mwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &m, sizeof(m));
            plow_hsa_wait(H, DEV);
            double dec = 0;
            double us = time_pair(kd, &a, NCU, km, &m, mwgs, WANT_PARTS ? &dec : NULL);
            if (WANT_PARTS)
                fprintf(stderr, "   ns=%-3u mwgs=%-4u total %8.1f us   decode %8.1f  merge %6.1f\n",
                        ns, mwgs, us, dec, us - dec);
            if (us < best_us) { best_us = us; best_dec = dec; best_ns = (int)ns; }
            if (mwgs == NCU) break;
        }
    }
    printf("plow,%s,bf16,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.1f,%.2f\n", g->tag, batch, ctx, g->nq,
           g->nkv, g->hd, best_ns, best_us, best_us, kvb / best_us / 1e3, best_dec);
    fflush(stdout);

done:
    if (dQ) plow_hsa_free(H, dQ);
    if (dK) plow_hsa_free(H, dK);
    if (dV) plow_hsa_free(H, dV);
    if (dO) plow_hsa_free(H, dO);
    if (dLen) plow_hsa_free(H, dLen);
    if (dOp) plow_hsa_free(H, dOp);
    if (dMl) plow_hsa_free(H, dMl);
}

/* ---------------------------------------------------------------------------
 * MLA decode (GLM-5.2 / DeepSeek): mla_flash_decode_512 -> merge -> o_uv_fold,
 * or the fused mla_merge_fold_512_v{256,32}. KV bytes = latent + rope, ONE
 * head-shared stream: batch * ctx * (DK + DR) * esz.  This matches
 * bench_mla_decode.py's accounting for aiter mla_decode_fwd.
 * ------------------------------------------------------------------------- */
struct __attribute__((packed)) mla_args {
    void* op; void* ml; const void* qa; const void* qr; const void* ckv; const void* kr;
    const void* len; unsigned n_batch, n_head, kv_stride, window; float scale; unsigned nsplit;
};
struct __attribute__((packed)) fold_args { /* mla_merge_fold_512_vN */
    void* o; const void* op; const void* ml; const void* wuv;
    unsigned n_batch, n_head, V, nsplit;
};

/* Gathered (DSA / sparse top-k) MLA decode: the same latent flash reading only the top_k
 * SELECTED rows through an index table. KV bytes counted are the rows actually TOUCHED --
 * batch * top_k * (DK+DR) * esz -- which is the whole point of the op. The idx table is a
 * random-without-replacement draw per (batch), so the gather is a real scatter, not a
 * contiguous prefix that would flatter the cache. */
struct __attribute__((packed)) gat_args {
    void* op; void* ml; const void* qa; const void* qr; const void* ckv; const void* kr;
    const void* len; const void* idx; unsigned top_k, n_batch, n_head, kv_stride;
    float scale; unsigned nsplit;
};

static void bench_gather(plow_hsa_kernel* kd, plow_hsa_kernel* kf, int n_head, int batch,
                         int ctx, int top_k, const int* nsplits, int n_ns, int vhd) {
    const int DK = 512, DR = 64;
    const size_t kvb = (size_t)batch * top_k * (DK + DR) * 2;
    const size_t nqa = (size_t)batch * n_head * DK, nqr = (size_t)batch * n_head * DR;
    int max_ns = 1;
    for (int i = 0; i < n_ns; i++) if (nsplits[i] > max_ns) max_ns = nsplits[i];

    void* dQa = plow_hsa_alloc(H, DEV, nqa * 2);
    void* dQr = plow_hsa_alloc(H, DEV, nqr * 2);
    void* dC = plow_hsa_alloc(H, DEV, (size_t)batch * ctx * DK * 2);
    void* dKr = plow_hsa_alloc(H, DEV, (size_t)batch * ctx * DR * 2);
    void* dLen = plow_hsa_alloc(H, DEV, (size_t)batch * 4);
    void* dIdx = plow_hsa_alloc(H, DEV, (size_t)batch * top_k * 4);
    void* dOp = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * max_ns * DK * 4);
    void* dMl = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * max_ns * 2 * 4);
    void* dW = plow_hsa_alloc(H, DEV, (size_t)n_head * DK * vhd * 2);
    void* dO = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * vhd * 2);
    if (!dQa || !dC || !dIdx || !dOp || !dW) { printf("skip,dsa-nh%d,OOM\n", n_head); return; }
    fill_dev(dQa, nqa * 2); fill_dev(dQr, nqr * 2);
    fill_dev(dC, (size_t)batch * ctx * DK * 2); fill_dev(dKr, (size_t)batch * ctx * DR * 2);
    fill_dev(dW, (size_t)n_head * DK * vhd * 2);
    {
        int* hl = plow_hsa_alloc_host(H, (size_t)batch * 4);
        for (int b = 0; b < batch; b++) hl[b] = ctx;
        plow_hsa_copy_h2d(H, DEV, dLen, hl, (size_t)batch * 4);
        int* hi = plow_hsa_alloc_host(H, (size_t)batch * top_k * 4);
        unsigned st = 12345u;
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < top_k; t++) {
                st = st * 1103515245u + 12345u;
                hi[(size_t)b * top_k + t] = (int)((st >> 8) % (unsigned)ctx);
            }
        plow_hsa_copy_h2d(H, DEV, dIdx, hi, (size_t)batch * top_k * 4);
    }

    double best = 1e30; int best_ns = 0;
    for (int i = 0; i < n_ns; i++) {
        const unsigned ns = (unsigned)nsplits[i];
        struct gat_args a = {dOp, dMl, dQa, dQr, dC, dKr, dLen, dIdx, (unsigned)top_k,
                             (unsigned)batch, (unsigned)n_head, (unsigned)ctx, 1.0f, ns};
        struct fold_args f = {dO, dOp, dMl, dW, (unsigned)batch, (unsigned)n_head, (unsigned)vhd, ns};
        const unsigned fwgs = (unsigned)batch * n_head < NCU ? (unsigned)batch * n_head : NCU;
        plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
        plow_hsa_launch(H, DEV, kf, fwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &f, sizeof(f));
        plow_hsa_wait(H, DEV);
        double ts[64];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_us();
            for (int it = 0; it < ITERS; it++) {
                plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
                plow_hsa_launch(H, DEV, kf, fwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &f, sizeof(f));
            }
            plow_hsa_wait(H, DEV);
            ts[r] = (now_us() - t0) / ITERS;
        }
        qsort(ts, REPS, sizeof(double), cmpd);
        if (WANT_PARTS) fprintf(stderr, "   dsa ns=%-3u %8.1f us\n", ns, ts[REPS / 2]);
        if (ts[REPS / 2] < best) { best = ts[REPS / 2]; best_ns = (int)ns; }
    }
    printf("plow,dsa-nh%d-topk%d,bf16,%d,%d,%d,1,512,%d,%.3f,%.3f,%.1f,0\n", n_head, top_k,
           batch, ctx, n_head, best_ns, best, best, kvb / best / 1e3);
    fflush(stdout);
    plow_hsa_free(H, dQa); plow_hsa_free(H, dQr); plow_hsa_free(H, dC); plow_hsa_free(H, dKr);
    plow_hsa_free(H, dLen); plow_hsa_free(H, dIdx); plow_hsa_free(H, dOp); plow_hsa_free(H, dMl);
    plow_hsa_free(H, dW); plow_hsa_free(H, dO);
}

static void bench_mla(plow_hsa_kernel* kd, plow_hsa_kernel* km, plow_hsa_kernel* kf,
                      int n_head, int batch, int ctx, const int* nsplits, int n_ns,
                      double max_kv_gb, int vhd) {
    const int DK = 512, DR = 64;
    const size_t kvb = (size_t)batch * ctx * (DK + DR) * 2;
    if (kvb > (size_t)(max_kv_gb * 1e9)) return;
    const size_t nqa = (size_t)batch * n_head * DK, nqr = (size_t)batch * n_head * DR;
    int max_ns = 1;
    for (int i = 0; i < n_ns; i++) if (nsplits[i] > max_ns) max_ns = nsplits[i];

    void* dQa = plow_hsa_alloc(H, DEV, nqa * 2);
    void* dQr = plow_hsa_alloc(H, DEV, nqr * 2);
    void* dC = plow_hsa_alloc(H, DEV, (size_t)batch * ctx * DK * 2);
    void* dKr = plow_hsa_alloc(H, DEV, (size_t)batch * ctx * DR * 2);
    void* dLen = plow_hsa_alloc(H, DEV, (size_t)batch * 4);
    void* dOp = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * max_ns * DK * 4);
    void* dMl = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * max_ns * 2 * 4);
    void* dOl = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * DK * 2);
    void* dW = plow_hsa_alloc(H, DEV, (size_t)n_head * DK * vhd * 2);
    void* dO = plow_hsa_alloc(H, DEV, (size_t)batch * n_head * vhd * 2);
    if (!dQa || !dC || !dOp || !dW) {
        printf("skip,mla-nh%d,bf16,%d,%d,%d,1,512,0,nan,nan,nan,OOM\n", n_head, batch, ctx, n_head);
        fflush(stdout);
        return;
    }
    fill_dev(dQa, nqa * 2); fill_dev(dQr, nqr * 2);
    fill_dev(dC, (size_t)batch * ctx * DK * 2); fill_dev(dKr, (size_t)batch * ctx * DR * 2);
    fill_dev(dW, (size_t)n_head * DK * vhd * 2);
    {
        int* hl = plow_hsa_alloc_host(H, (size_t)batch * 4);
        for (int b = 0; b < batch; b++) hl[b] = ctx;
        plow_hsa_copy_h2d(H, DEV, dLen, hl, (size_t)batch * 4);
    }

    double best = 1e30; int best_ns = 0;
    for (int i = 0; i < n_ns; i++) {
        const unsigned ns = (unsigned)nsplits[i];
        struct mla_args a = {dOp, dMl, dQa, dQr, dC, dKr, dLen, (unsigned)batch,
                             (unsigned)n_head, (unsigned)ctx, 0u, 1.0f, ns};
        struct fold_args f = {dO, dOp, dMl, dW, (unsigned)batch, (unsigned)n_head, (unsigned)vhd, ns};
        const unsigned fwgs = (unsigned)batch * n_head < NCU ? (unsigned)batch * n_head : NCU;
        plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
        plow_hsa_launch(H, DEV, kf, fwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &f, sizeof(f));
        plow_hsa_wait(H, DEV);
        double ts[64];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_us();
            for (int it = 0; it < ITERS; it++) {
                plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
                plow_hsa_launch(H, DEV, kf, fwgs * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &f, sizeof(f));
            }
            plow_hsa_wait(H, DEV);
            ts[r] = (now_us() - t0) / ITERS;
        }
        qsort(ts, REPS, sizeof(double), cmpd);
        double dus[64];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_us();
            for (int it = 0; it < ITERS; it++)
                plow_hsa_launch(H, DEV, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
            plow_hsa_wait(H, DEV);
            dus[r] = (now_us() - t0) / ITERS;
        }
        qsort(dus, REPS, sizeof(double), cmpd);
        if (WANT_PARTS) fprintf(stderr, "   mla ns=%-3u total %8.1f  decode %8.1f  fold %7.1f us\n",
                                ns, ts[REPS / 2], dus[REPS / 2], ts[REPS / 2] - dus[REPS / 2]);
        if (ts[REPS / 2] < best) { best = ts[REPS / 2]; best_ns = (int)ns; }
    }
    printf("plow,mla-nh%d,bf16,%d,%d,%d,1,512,%d,%.3f,%.3f,%.1f,0\n", n_head, batch, ctx, n_head,
           best_ns, best, best, kvb / best / 1e3);
    fflush(stdout);
    plow_hsa_free(H, dQa); plow_hsa_free(H, dQr); plow_hsa_free(H, dC); plow_hsa_free(H, dKr);
    plow_hsa_free(H, dLen); plow_hsa_free(H, dOp); plow_hsa_free(H, dMl); plow_hsa_free(H, dOl);
    plow_hsa_free(H, dW); plow_hsa_free(H, dO);
}

int main(int argc, char** argv) {
    const char* which = "flash";
    const char* mlak = "mla_flash_decode_512";
    double max_kv_gb = 90.0;
    int only_hd = 0, only_batch = 0, only_ctx = 0, only_nq = 0, fixed_ns = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--parts")) WANT_PARTS = 1;
        else if (!strncmp(argv[i], "--which=", 8)) which = argv[i] + 8;
        else if (!strncmp(argv[i], "--hd=", 5)) only_hd = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "--batch=", 8)) only_batch = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--ctx=", 6)) only_ctx = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--iters=", 8)) ITERS = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--reps=", 7)) REPS = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--maxkv=", 8)) max_kv_gb = atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--dev=", 6)) DEV = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--nq=", 5)) only_nq = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "--ns=", 5)) fixed_ns = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "--mlak=", 7)) mlak = argv[i] + 7;
    }

    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, DEV, nm, &cus, &lds);
    NCU = cus;
    fprintf(stderr, "dev%d: %s CUs=%u LDS=%u  threads/wg=%d\n", DEV, nm, cus, lds, PLOW_WG_THREADS);

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, DEV, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    printf("impl,tag,dtype,batch,ctx,n_q,n_kv,head_dim,nsplit,us_median,us_best,gbps,dec_us\n");

    int NS[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int NNS = (int)(sizeof(NS) / sizeof(NS[0]));
    if (fixed_ns) { NS[0] = fixed_ns; NNS = 1; }

    if (!strcmp(which, "flash")) {
        plow_hsa_kernel d256, d512, m256, m512;
        if (plow_hsa_get_kernel(H, DEV, "gemma_flash_decode_256", &d256) ||
            plow_hsa_get_kernel(H, DEV, "gemma_flash_decode_512", &d512) ||
            plow_hsa_get_kernel(H, DEV, "gemma_flash_merge_256", &m256) ||
            plow_hsa_get_kernel(H, DEV, "gemma_flash_merge_512", &m512)) {
            fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
        }
        fprintf(stderr, "dec256 LDS=%u  dec512 LDS=%u\n", d256.group_segment_size,
                d512.group_segment_size);
        const struct geom G[] = {
            {"gemma4-31B(32q/16kv)", 256, 32, 16}, {"gemma4-12B(16q/8kv)", 256, 16, 8},
            {"gemma4-31B(32q/16kv)", 512, 32, 16}, {"gemma4-12B(16q/8kv)", 512, 16, 8},
        };
        const int BS[] = {1, 8, 32};
        const int CS[] = {4096, 32768, 131072};
        for (unsigned gi = 0; gi < sizeof(G) / sizeof(G[0]); gi++) {
            if (only_hd && G[gi].hd != only_hd) continue;
            if (only_nq && G[gi].nq != only_nq) continue;
            for (int bi = 0; bi < 3; bi++) {
                if (only_batch && BS[bi] != only_batch) continue;
                for (int ci = 0; ci < 3; ci++) {
                    if (only_ctx && CS[ci] != only_ctx) continue;
                    bench_flash(G[gi].hd == 256 ? &d256 : &d512, G[gi].hd == 256 ? &m256 : &m512,
                                &G[gi], BS[bi], CS[ci], NS, NNS, max_kv_gb);
                }
            }
        }
    } else if (!strcmp(which, "dsa")) {
        plow_hsa_kernel gd, mf;
        if (plow_hsa_get_kernel(H, DEV, "mla_gather_decode_512", &gd) ||
            plow_hsa_get_kernel(H, DEV, "mla_merge_fold_512_v32", &mf)) {
            fprintf(stderr, "sym dsa: %s\n", plow_hsa_last_error()); return 1;
        }
        const int NH[] = {16, 32, 128};
        const int BS[] = {1, 8, 32};
        const int TK[] = {512, 2048};
        for (int hi = 0; hi < 3; hi++) {
            if (only_nq && NH[hi] != only_nq) continue;
            for (int bi = 0; bi < 3; bi++) {
                if (only_batch && BS[bi] != only_batch) continue;
                for (int ti = 0; ti < 2; ti++)
                    bench_gather(&gd, &mf, NH[hi], BS[bi], only_ctx ? only_ctx : 131072, TK[ti],
                                 NS, NNS, 128);
            }
        }
    } else if (!strcmp(which, "mla")) {
        plow_hsa_kernel md, mm, mf;
        if (plow_hsa_get_kernel(H, DEV, mlak, &md) ||
            plow_hsa_get_kernel(H, DEV, "gemma_flash_merge_512", &mm) ||
            plow_hsa_get_kernel(H, DEV, "mla_merge_fold_512_v32", &mf)) {
            fprintf(stderr, "sym mla: %s\n", plow_hsa_last_error()); return 1;
        }
        const int NH[] = {16, 32, 128};
        const int BS[] = {1, 8, 32};
        const int CS[] = {4096, 32768, 131072};
        for (int hi = 0; hi < 3; hi++) {
            if (only_nq && NH[hi] != only_nq) continue;
            for (int bi = 0; bi < 3; bi++) {
                if (only_batch && BS[bi] != only_batch) continue;
                for (int ci = 0; ci < 3; ci++) {
                    if (only_ctx && CS[ci] != only_ctx) continue;
                    bench_mla(&md, &mm, &mf, NH[hi], BS[bi], CS[ci], NS, NNS, max_kv_gb, 128);
                }
            }
        }
    }
    plow_hsa_shutdown(H);
    return 0;
}
