/* tp_dcp_gather_bench.c — DCP owner gather (ops 181/182, op_collective.h) on 8 GPUs.
 *
 * Every rank holds only its shard's KV rows (page 64, degree 8, block-cyclic, packet::dcp). Each
 * rank packs the selected records it owns; the collective then pulls every live record from its
 * owner. Every gathered byte on every rank is checked against the host oracle — byte-exact.
 *
 *   TP_MODE=decode  B=32 rows x K=2048 selected, per-row kv_len (every fourth row is shorter than
 *                   K, exercising the live clamp and `glen`); idx replicated on every rank.
 *   TP_MODE=prefix  B=1, K=81920, idx null (record j is global row j), kv_len 70001: the prefill
 *                   form.
 *   TP_MODE=scatter-decode / scatter-prefix  op 183, the owner-only write (32 decode slots at
 *                   random positions / an 8192-row chunk at 61000), untouched rows checked too.
 * env: TP_NWG (default 48) TP_REPS (default 5) TP_ELF (default dcp_gather.elf)
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define XCTR_BYTES (128u * 64u)
#define PSHIFT 6u
#define DSHIFT 3u
#define PAGE (1u << PSHIFT)
#define MAX_CTX 81920u
#define REC_BYTES 656u

static uint32_t hsh(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static uint32_t rowkey(uint32_t b, uint32_t g) { return hsh(b * 0x9E3779B9u ^ hsh(g + 0x51ED27u)); }
static uint8_t ckv_byte(uint32_t key, uint32_t d) { return (uint8_t)(hsh(key ^ (d >> 2)) >> ((d & 3u) * 8u)); }
static uint16_t krot_word(uint32_t key, uint32_t d) { return (uint16_t)hsh(key ^ (0x100u + d)); }
static uint32_t scale_bits(uint32_t key) { return hsh(key ^ 0xABCDEFu); }
static uint32_t envu(const char* k, uint32_t d) { const char* v = getenv(k); return v ? (uint32_t)strtoul(v, 0, 10) : d; }
static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static uint32_t global_row(uint32_t shard, uint32_t local) {
    return (local / PAGE) * (PAGE * NR) + shard * PAGE + local % PAGE;
}

typedef struct {
    void* slot; void* idx; void* kv_len; void* ckv; void* krot; void* kv_scale;
    void* gckv; void* gkrot; void* gscale; void* glen;
    void* peers; void* status; void* ts; size_t xoff; uint64_t deadline;
    uint32_t n_batch, K, local_stride, page_shift, degree_shift, rank, n_gpu, slot_bytes, nwg;
} TdgArgs;
/* The kernel's kernarg segment ends at the last u32; sizeof would add 4 B of tail padding. */
#define TDG_ARGS_BYTES (offsetof(TdgArgs, nwg) + sizeof(uint32_t))

static double launch_all(plow_hsa* h, int* dev, plow_hsa_kernel* k, TdgArgs* a, void** ts,
                         uint32_t nwg, uint64_t* t) {
    for (int r = NR - 1; r >= 0; r--)
        if (plow_hsa_launch(h, dev[r], &k[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &a[r], TDG_ARGS_BYTES)) {
            fprintf(stderr, "launch rank %d: %s\n", r, plow_hsa_last_error());
            exit(2);
        }
    for (int r = 0; r < NR; r++)
        if (plow_hsa_wait(h, dev[r])) {
            fprintf(stderr, "wait rank %d: %s\n", r, plow_hsa_last_error());
            exit(2);
        }
    plow_hsa_download(h, dev[0], t, ts[0], nwg * 2u * 8u);
    uint64_t t0 = ~0ull, m1 = 0;
    for (uint32_t w = 0; w < nwg; w++) {
        if (t[2 * w] < t0) t0 = t[2 * w];
        if (t[2 * w + 1] > m1) m1 = t[2 * w + 1];
    }
    return (m1 - t0) * 0.01;
}

typedef struct {
    void* pos; void* ckv_st; void* krot_st; void* scale_st; void* ckv; void* krot; void* kv_scale;
    void* ts;
    uint32_t rows, local_stride, page_shift, degree_shift, shard, batched, nwg;
} TdsArgs;
#define TDS_ARGS_BYTES (offsetof(TdsArgs, nwg) + sizeof(uint32_t))

/* TP_MODE=scatter-decode | scatter-prefix: op 183 on every rank. Local caches start at a sentinel
 * byte; afterwards every byte must be the sentinel except the rows the rank owns, which must equal
 * the staged row. */
static int scatter_main(int prefix, char** argv) {
    const uint32_t nwg = envu("TP_NWG", 48), reps = envu("TP_REPS", 5);
    const uint32_t rows = prefix ? 8192u : 32u;
    const uint32_t Ls = (MAX_CTX / PAGE + NR - 1) / NR * PAGE;
    const size_t lrows = prefix ? Ls : (size_t)rows * Ls;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);
    int32_t* pos = malloc(rows * 4u);
    for (uint32_t t = 0; t < rows; t++) pos[t] = prefix ? (int32_t)(61000u + t) : (int32_t)(hsh(t) % 70000u);
    uint8_t* cst = malloc(rows * 512u); uint16_t* kst = malloc(rows * 128u); uint32_t* sst = malloc(rows * 4u);
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t key = rowkey(t, (uint32_t)pos[t]);
        for (uint32_t d = 0; d < 512u; d++) cst[t * 512u + d] = ckv_byte(key, d);
        for (uint32_t d = 0; d < 64u; d++) kst[t * 64u + d] = krot_word(key, d);
        sst[t] = scale_bits(key);
    }
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    size_t elf_len = 0;
    void* elf = slurp(getenv("TP_ELF") ? getenv("TP_ELF") : "dcp_gather.elf", &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }
    uint8_t* hc = malloc(lrows * 512u); uint16_t* hk = malloc(lrows * 128u); uint32_t* hs = malloc(lrows * 4u);
    memset(hc, 0xA5, lrows * 512u); memset(hk, 0xA5, lrows * 128u); memset(hs, 0xA5, lrows * 4u);
    plow_hsa_kernel k[NR];
    TdsArgs a[NR];
    double* tt = calloc(reps, sizeof(double));
    uint64_t* tsh = malloc((size_t)nwg * 2u * 8u);
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tdg_scatter", &k[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        a[r] = (TdsArgs){plow_hsa_alloc(h, dev[r], rows * 4u), plow_hsa_alloc(h, dev[r], rows * 512u),
                         plow_hsa_alloc(h, dev[r], rows * 128u), plow_hsa_alloc(h, dev[r], rows * 4u),
                         plow_hsa_alloc(h, dev[r], lrows * 512u), plow_hsa_alloc(h, dev[r], lrows * 128u),
                         plow_hsa_alloc(h, dev[r], lrows * 4u), plow_hsa_alloc(h, dev[r], nwg * 16u),
                         rows, Ls, PSHIFT, DSHIFT, (uint32_t)r, prefix ? 0u : 1u, nwg};
        if (!a[r].pos || !a[r].ckv_st || !a[r].krot_st || !a[r].scale_st || !a[r].ckv || !a[r].krot ||
            !a[r].kv_scale || !a[r].ts) { fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2; }
        plow_hsa_upload(h, dev[r], a[r].pos, pos, rows * 4u);
        plow_hsa_upload(h, dev[r], a[r].ckv_st, cst, rows * 512u);
        plow_hsa_upload(h, dev[r], a[r].krot_st, kst, rows * 128u);
        plow_hsa_upload(h, dev[r], a[r].scale_st, sst, rows * 4u);
    }
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], a[r].ckv, hc, lrows * 512u);
            plow_hsa_upload(h, dev[r], a[r].krot, hk, lrows * 128u);
            plow_hsa_upload(h, dev[r], a[r].kv_scale, hs, lrows * 4u);
            if (plow_hsa_launch(h, dev[r], &k[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &a[r], TDS_ARGS_BYTES)) {
                fprintf(stderr, "launch rank %d: %s\n", r, plow_hsa_last_error()); return 2;
            }
            if (plow_hsa_wait(h, dev[r])) { fprintf(stderr, "wait: %s\n", plow_hsa_last_error()); return 2; }
        }
        plow_hsa_download(h, dev[0], tsh, a[0].ts, nwg * 16u);
        uint64_t t0 = ~0ull, m1 = 0;
        for (uint32_t w = 0; w < nwg; w++) { if (tsh[2 * w] < t0) t0 = tsh[2 * w]; if (tsh[2 * w + 1] > m1) m1 = tsh[2 * w + 1]; }
        tt[rep] = (m1 - t0) * 0.01;
    }
    qsort(tt, reps, sizeof(double), cmpd);
    size_t bad = 0, written = 0;
    uint8_t* dc = malloc(lrows * 512u); uint16_t* dk = malloc(lrows * 128u); uint32_t* ds = malloc(lrows * 4u);
    uint8_t* want = malloc(lrows);
    for (int r = 0; r < NR; r++) {
        plow_hsa_download(h, dev[r], dc, a[r].ckv, lrows * 512u);
        plow_hsa_download(h, dev[r], dk, a[r].krot, lrows * 128u);
        plow_hsa_download(h, dev[r], ds, a[r].kv_scale, lrows * 4u);
        memset(want, 0, lrows);
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t g = (uint32_t)pos[t];
            if (((g >> PSHIFT) & (NR - 1u)) != (uint32_t)r) continue;
            const uint32_t l = ((g >> (PSHIFT + DSHIFT)) << PSHIFT) | (g & (PAGE - 1u));
            const size_t lr = (prefix ? 0u : (size_t)t * Ls) + l;
            want[lr] = 1; written++;
            bad += memcmp(dc + lr * 512u, cst + (size_t)t * 512u, 512u) != 0;
            bad += memcmp(dk + lr * 64u, kst + (size_t)t * 64u, 128u) != 0;
            bad += ds[lr] != sst[t];
        }
        for (size_t lr = 0; lr < lrows; lr++) {
            if (want[lr]) continue;
            bad += memcmp(dc + lr * 512u, hc, 512u) != 0;
            bad += memcmp(dk + lr * 64u, hk, 128u) != 0;
            bad += ds[lr] != hs[0];
        }
    }
    printf("mode=scatter-%s nwg=%u rows=%u written=%zu scatter_us=%.1f reps=%u parity=%s bad=%zu\n",
           prefix ? "prefix" : "decode", nwg, rows, written, tt[reps / 2], reps, bad ? "FAIL" : "PASS", bad);
    plow_hsa_shutdown(h);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* mode = getenv("TP_MODE") ? getenv("TP_MODE") : "decode";
    if (!strcmp(mode, "scatter-decode")) return scatter_main(0, argv);
    if (!strcmp(mode, "scatter-prefix")) return scatter_main(1, argv);
    const int prefix = !strcmp(mode, "prefix");
    if (!prefix && strcmp(mode, "decode")) { fprintf(stderr, "TP_MODE must be decode|prefix\n"); return 2; }
    const uint32_t nwg = envu("TP_NWG", 48), reps = envu("TP_REPS", 5);
    const uint32_t B = prefix ? 1u : 32u, K = prefix ? MAX_CTX : 2048u;
    const uint32_t Ls = (MAX_CTX / PAGE + NR - 1) / NR * PAGE;
    const size_t n_rec = (size_t)B * K;
    const size_t xoff = n_rec * REC_BYTES;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    int32_t* len = malloc(B * sizeof(int32_t));
    for (uint32_t b = 0; b < B; b++) len[b] = prefix ? 70001 : (b % 4 == 0 ? 1500 : 60000 + (int32_t)b * 101);
    int32_t* idx = prefix ? NULL : malloc(n_rec * sizeof(int32_t));
    if (idx)
        for (uint32_t b = 0; b < B; b++)
            for (uint32_t j = 0; j < K; j++) idx[b * K + j] = (int32_t)(hsh(b * 7919u + j) % (uint32_t)len[b]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    size_t elf_len = 0;
    void* elf = slurp(getenv("TP_ELF") ? getenv("TP_ELF") : "dcp_gather.elf", &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }

    const size_t lrows = (size_t)B * Ls;
    const size_t hrows = lrows > n_rec ? lrows : n_rec;
    uint8_t* hckv = malloc(hrows * 512u);
    uint16_t* hkrot = malloc(hrows * 128u);
    uint32_t* hscale = malloc(hrows * 4u);
    plow_hsa_kernel kpack[NR], kgather[NR];
    void* scratch[NR], *table[NR], *ts[NR];
    TdgArgs a[NR];
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tdg_pack", &kpack[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tdg_gather", &kgather[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        memset(&a[r], 0, sizeof a[r]);
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], xoff + XCTR_BYTES);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        ts[r] = plow_hsa_alloc(h, dev[r], nwg * 2u * 8u);
        a[r].slot = scratch[r];
        a[r].kv_len = plow_hsa_alloc(h, dev[r], B * 4u);
        a[r].idx = idx ? plow_hsa_alloc(h, dev[r], n_rec * 4u) : NULL;
        a[r].ckv = plow_hsa_alloc(h, dev[r], lrows * 512u);
        a[r].krot = plow_hsa_alloc(h, dev[r], lrows * 128u);
        a[r].kv_scale = plow_hsa_alloc(h, dev[r], lrows * 4u);
        a[r].gckv = plow_hsa_alloc(h, dev[r], n_rec * 512u);
        a[r].gkrot = plow_hsa_alloc(h, dev[r], n_rec * 128u);
        a[r].gscale = plow_hsa_alloc(h, dev[r], n_rec * 4u);
        a[r].glen = prefix ? NULL : plow_hsa_alloc(h, dev[r], B * 4u);
        a[r].status = plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !table[r] || !ts[r] || !a[r].kv_len || !a[r].ckv || !a[r].krot ||
            !a[r].kv_scale || !a[r].gckv || !a[r].gkrot || !a[r].gscale || !a[r].status ||
            (idx && !a[r].idx) || (!prefix && !a[r].glen)) {
            fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2;
        }
        a[r].peers = table[r]; a[r].ts = ts[r]; a[r].xoff = xoff;
        a[r].deadline = freq ? freq : 1000000000ull;
        a[r].n_batch = B; a[r].K = K; a[r].local_stride = Ls; a[r].page_shift = PSHIFT;
        a[r].degree_shift = DSHIFT; a[r].rank = (uint32_t)r; a[r].n_gpu = NR;
        a[r].slot_bytes = 0; a[r].nwg = nwg;

        for (uint32_t b = 0; b < B; b++)
            for (uint32_t l = 0; l < Ls; l++) {
                const uint32_t key = rowkey(b, global_row((uint32_t)r, l));
                const size_t lr = (size_t)b * Ls + l;
                for (uint32_t d = 0; d < 512u; d++) hckv[lr * 512u + d] = ckv_byte(key, d);
                for (uint32_t d = 0; d < 64u; d++) hkrot[lr * 64u + d] = krot_word(key, d);
                hscale[lr] = scale_bits(key);
            }
        plow_hsa_upload(h, dev[r], a[r].ckv, hckv, lrows * 512u);
        plow_hsa_upload(h, dev[r], a[r].krot, hkrot, lrows * 128u);
        plow_hsa_upload(h, dev[r], a[r].kv_scale, hscale, lrows * 4u);
        plow_hsa_upload(h, dev[r], a[r].kv_len, len, B * 4u);
        if (idx) plow_hsa_upload(h, dev[r], a[r].idx, idx, n_rec * 4u);
    }
    for (int r = 0; r < NR; r++) plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);

    static uint8_t zero[XCTR_BYTES];
    double* tp = calloc(reps, sizeof(double)), *tg = calloc(reps, sizeof(double));
    uint64_t* t = malloc((size_t)nwg * 2u * 8u);
    int timeout = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + xoff, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], a[r].status, zero, 4);
            plow_hsa_wait(h, dev[r]);
        }
        tp[rep] = launch_all(h, dev, kpack, a, ts, nwg, t);
        tg[rep] = launch_all(h, dev, kgather, a, ts, nwg, t);
        for (int r = 0; r < NR; r++) { uint32_t st = 0; plow_hsa_download(h, dev[r], &st, a[r].status, 4); timeout |= st != 0; }
    }
    qsort(tp, reps, sizeof(double), cmpd);
    qsort(tg, reps, sizeof(double), cmpd);

    size_t bad = 0, live = 0;
    for (int r = 0; r < NR; r++) {
        plow_hsa_download(h, dev[r], hckv, a[r].gckv, n_rec * 512u);
        plow_hsa_download(h, dev[r], hkrot, a[r].gkrot, n_rec * 128u);
        plow_hsa_download(h, dev[r], hscale, a[r].gscale, n_rec * 4u);
        int32_t glen[32];
        if (!prefix) plow_hsa_download(h, dev[r], glen, a[r].glen, B * 4u);
        for (uint32_t b = 0; b < B; b++) {
            const uint32_t lv = (uint32_t)len[b] < K ? (uint32_t)len[b] : K;
            if (!prefix) bad += (uint32_t)glen[b] != lv;
            for (uint32_t j = 0; j < lv; j++) {
                const size_t rr = (size_t)b * K + j;
                const uint32_t key = rowkey(b, idx ? (uint32_t)idx[rr] : j);
                for (uint32_t d = 0; d < 512u; d++) bad += hckv[rr * 512u + d] != ckv_byte(key, d);
                for (uint32_t d = 0; d < 64u; d++) bad += hkrot[rr * 64u + d] != krot_word(key, d);
                bad += hscale[rr] != scale_bits(key);
                live += r == 0;
            }
        }
    }
    const uint32_t mid = reps / 2;
    const double moved = (double)live * 644.0 * 7.0 / 8.0;
    printf("mode=%s nwg=%u live=%zu pack_us=%.1f gather_us=%.1f GBps=%.1f reps=%u parity=%s bad=%zu timeout=%s\n",
           mode, nwg, live, tp[mid], tg[mid], moved / (tg[mid] * 1e3), reps, bad ? "FAIL" : "PASS",
           bad, timeout ? "YES" : "no");
    plow_hsa_shutdown(h);
    return (bad || timeout) ? 1 : 0;
}
