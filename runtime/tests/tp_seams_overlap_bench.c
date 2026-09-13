/* Bounding microbenches for the two ways past the capped seam collectives (8x MI300X, TP8):
 *   TP_MODE=push_rs    push-form reduce-scatter + strict-order local sum    (tp_push_rs)
 *   TP_MODE=push_busy  the push from an o_proj-shaped tile loop             (tp_push_busy)
 *                      TP_BUSY=0 local store (today), 1 push, 2 no store (compute only)
 *   TP_MODE=banded     op 26 in TP_K row bands                             (tp_seams_banded)
 *   TP_MODE=corun      op 26 on TP_KAG workgroups beside a consumer         (tp_seams_corun)
 *                      TP_CMODE=0 gather only, 1 consumer only, 2 both
 * Every mode checks its output on every rank against the host strict-order oracle (the
 * tp_fill_random words, as tp_allreduce_prefill_bench TP_RANDOM=1) and prints the median of
 * TP_REPS runs. One workgroup per CU (48 KiB of dynamic LDS each), like the interpreter.
 * env: TP_ROWS (8192) TP_HIDDEN (6144) TP_NWG (304) TP_KPUSH TP_MFMA TP_K TP_KAG TP_REPS TP_ELF
 * Built by scripts/build_tp_allreduce.sh-style gcc + hsa_backend.c. */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define XCTR_BYTES (128u * 64u)
#define DYN_LDS 49152u

typedef uint16_t bf16;
static float bf2f(bf16 b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u); return (bf16)(u >> 16);
}
static uint32_t tp_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static const uint32_t SEED = 20260904u;
static bf16 word(uint32_t r, uint32_t e) {  /* tp_fill_random(rank r, SEED) */
    const uint32_t h = tp_hash(e ^ tp_hash(r * 0x9E3779B9u + SEED));
    return (bf16)(((h >> 31) << 15) | ((120u + ((h >> 20) & 15u)) << 7) | (h & 0x7fu));
}
static uint32_t envu(const char* k, uint32_t d) { const char* v = getenv(k); return v ? (uint32_t)strtoul(v, 0, 10) : d; }
static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

typedef struct { void* part; uint32_t n, rank, seed; } a_fill;
typedef struct { const void* part; const void* peers; uint32_t nranks, rank, n, recv_off; uint64_t xoff;
                 uint32_t kpush, pad0; void* out; uint64_t deadline; void* status; void* ts; } a_push;
typedef struct { const void* part; const void* peers; uint32_t nranks, rank, n, recv_off; uint64_t xoff;
                 uint32_t mode, mfma; void* cbuf; void* out; uint64_t deadline; void* status; void* ts; } a_busy;
typedef struct { const void* peers; uint32_t nranks, rank, n, slot; uint64_t xoff; uint32_t k, pad0;
                 void* out; uint64_t deadline; void* status; void* ts; } a_band;
typedef struct { const void* peers; uint32_t nranks, rank, n, slot; uint64_t xoff; void* out;
                 const void* cbuf; uint32_t cn, mfma, cmode, kag; uint64_t deadline; void* status; void* ts; } a_corun;

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* mode = getenv("TP_MODE") ? getenv("TP_MODE") : "push_rs";
    const uint32_t rows = envu("TP_ROWS", 8192), hid = envu("TP_HIDDEN", 6144), nwg = envu("TP_NWG", 304);
    const uint32_t reps = envu("TP_REPS", 5), kpush = envu("TP_KPUSH", 24), mfma = envu("TP_MFMA", 0);
    const uint32_t kb = envu("TP_K", 1), kag = envu("TP_KAG", 24), busy = envu("TP_BUSY", 1), cmode = envu("TP_CMODE", 2);
    const uint64_t n64 = (uint64_t)rows * hid;
    if (n64 % (65536u * NR) || n64 % (8u * NR * (kb ? kb : 1)) || n64 > 0x7fffffffu) {
        fprintf(stderr, "unsupported n=%llu\n", (unsigned long long)n64); return 2;
    }
    const uint32_t n = (uint32_t)n64, band = n / NR;
    const uint32_t recv_off = n * 2u;                 /* part at 0, recv [src][band] after it */
    const uint64_t xoff = (uint64_t)n * 4u;           /* counters after both */
    const size_t region = (size_t)xoff + XCTR_BYTES;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    size_t elf_len = 0; void* elf = slurp(getenv("TP_ELF") ? getenv("TP_ELF") : "tp_allreduce_kernels.elf", &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }
    const char* kname = !strcmp(mode, "push_rs") ? "tp_push_rs" : !strcmp(mode, "push_busy") ? "tp_push_busy"
                      : !strcmp(mode, "banded") ? "tp_seams_banded" : !strcmp(mode, "corun") ? "tp_seams_corun" : NULL;
    if (!kname) { fprintf(stderr, "TP_MODE must be push_rs|push_busy|banded|corun\n"); return 2; }
    plow_hsa_kernel kfill[NR], kk[NR];
    void *scratch[NR], *table[NR], *out[NR], *cbuf[NR], *ts[NR]; uint32_t* status[NR];
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tp_fill_random", &kfill[r]) ||
            plow_hsa_get_kernel(h, dev[r], kname, &kk[r])) { fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2; }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], region);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        out[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        cbuf[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        ts[r] = plow_hsa_alloc(h, dev[r], (size_t)nwg * 4u * 8u);
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !table[r] || !out[r] || !cbuf[r] || !ts[r] || !status[r]) { fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2; }
    }
    for (int r = 0; r < NR; r++) plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const uint64_t deadline = freq ? freq : 1000000000ull;
    static uint8_t zero[XCTR_BYTES];
    for (int r = 0; r < NR; r++) {
        a_fill f = {scratch[r], n, (uint32_t)r, SEED};          /* partial / slot words */
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &f, sizeof f);
        a_fill g = {cbuf[r], n, (uint32_t)r, SEED + 7u};        /* consumer input */
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &g, sizeof g);
        plow_hsa_wait(h, dev[r]);
    }
    double *a1 = calloc(reps, sizeof(double)), *a2 = calloc(reps, sizeof(double)), *a3 = calloc(reps, sizeof(double));
    uint64_t* t = malloc((size_t)nwg * 32u);
    int timeout = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + xoff, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            plow_hsa_wait(h, dev[r]);
        }
        for (int r = NR - 1; r >= 0; r--) {
            a_push ap = {scratch[r], table[r], NR, (uint32_t)r, n, recv_off, xoff, kpush, 0, out[r], deadline, status[r], ts[r]};
            a_busy ab = {scratch[r], table[r], NR, (uint32_t)r, n, recv_off, xoff, busy, mfma, cbuf[r], out[r], deadline, status[r], ts[r]};
            a_band an = {table[r], NR, (uint32_t)r, n, 0u, xoff, kb, 0, out[r], deadline, status[r], ts[r]};
            a_corun ac = {table[r], NR, (uint32_t)r, n, 0u, xoff, out[r], cbuf[r], n, mfma, cmode, kag, deadline, status[r], ts[r]};
            const void* a = !strcmp(mode, "push_rs") ? (const void*)&ap : !strcmp(mode, "push_busy") ? (const void*)&ab
                          : !strcmp(mode, "banded") ? (const void*)&an : (const void*)&ac;
            const size_t az = !strcmp(mode, "push_rs") ? sizeof ap : !strcmp(mode, "push_busy") ? sizeof ab
                            : !strcmp(mode, "banded") ? sizeof an : sizeof ac;
            plow_hsa_launch(h, dev[r], &kk[r], nwg * 512u, 1, 1, 512, 1, 1, DYN_LDS, a, az);
        }
        for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        for (int r = 0; r < NR; r++) { uint32_t st = 0; plow_hsa_download(h, dev[r], &st, status[r], 4); timeout |= st != 0; }
        plow_hsa_download(h, dev[0], t, ts[0], (size_t)nwg * 32u);
        uint64_t t0 = ~0ull, m1 = 0, m2 = 0, m3 = 0, g_end = 0, c_end = 0;
        for (uint32_t w = 0; w < nwg; w++) {
            if (t[4 * w] < t0) t0 = t[4 * w];
            if (t[4 * w + 1] > m1) m1 = t[4 * w + 1];
            if (t[4 * w + 2] > m2) m2 = t[4 * w + 2];
            if (t[4 * w + 3] > m3) m3 = t[4 * w + 3];
            if (w < kag) { if (t[4 * w + 3] > g_end) g_end = t[4 * w + 3]; }
            else if (t[4 * w + 3] > c_end) c_end = t[4 * w + 3];
        }
        if (!strcmp(mode, "corun")) { a1[rep] = (g_end - t0) * 0.01; a2[rep] = (c_end - t0) * 0.01; a3[rep] = (m3 - t0) * 0.01; }
        else { a1[rep] = (m1 - t0) * 0.01; a2[rep] = (m2 - m1) * 0.01; a3[rep] = (m3 - t0) * 0.01; }
    }
    qsort(a1, reps, sizeof(double), cmpd); qsort(a2, reps, sizeof(double), cmpd); qsort(a3, reps, sizeof(double), cmpd);
    /* Oracle on every rank. */
    size_t bad = 0; int checked = 0;
    bf16* hb = malloc((size_t)n * 2u);
    for (int r = 0; r < NR; r++) {
        if (!strcmp(mode, "push_rs") || (!strcmp(mode, "push_busy") && busy == 1)) {
            plow_hsa_download(h, dev[r], hb, out[r], (size_t)band * 2u);
            for (uint32_t e = 0; e < band; e++) {
                float s = 0.0f;
                for (uint32_t src = 0; src < NR; src++) s += bf2f(word(src, (uint32_t)r * band + e));
                bad += hb[e] != f2bf(s);
            }
            checked = 1;
        } else if (!strcmp(mode, "banded") || (!strcmp(mode, "corun") && cmode != 1)) {
            const uint32_t nb = !strcmp(mode, "banded") ? n / kb : n;
            plow_hsa_download(h, dev[r], hb, out[r], (size_t)n * 2u);
            for (uint32_t e = 0; e < n; e++) {
                const uint32_t owner = (uint32_t)(((uint64_t)(e % nb) * NR) / nb);
                bad += hb[e] != word(owner, e);
            }
            checked = 1;
        }
    }
    const uint32_t mid = reps / 2;
    if (!strcmp(mode, "corun"))
        printf("mode=corun rows=%u hidden=%u nwg=%u kag=%u cmode=%u mfma=%u gather_us=%.1f consumer_us=%.1f total_us=%.1f",
               rows, hid, nwg, kag, cmode, mfma, a1[mid], a2[mid], a3[mid]);
    else if (!strcmp(mode, "banded"))
        printf("mode=banded rows=%u hidden=%u nwg=%u k=%u total_us=%.1f", rows, hid, nwg, kb, a3[mid]);
    else if (!strcmp(mode, "push_busy"))
        printf("mode=push_busy rows=%u hidden=%u nwg=%u busy=%u mfma=%u tiles_us=%.1f gate_us=%.1f total_us=%.1f",
               rows, hid, nwg, busy, mfma, a1[mid], a2[mid], a3[mid]);
    else
        printf("mode=push_rs rows=%u hidden=%u nwg=%u kpush=%u push_us=%.1f gate_us=%.1f total_us=%.1f push_GBps=%.1f",
               rows, hid, nwg, kpush, a1[mid], a2[mid], a3[mid], (double)band * 2.0 * (NR - 1) / (a1[mid] * 1e3));
    printf(" reps=%u parity=%s bad=%zu timeout=%s\n", reps, checked ? (bad ? "FAIL" : "PASS") : "n/a", bad, timeout ? "YES" : "no");
    plow_hsa_shutdown(h);
    return (bad || timeout) ? 1 : 0;
}
