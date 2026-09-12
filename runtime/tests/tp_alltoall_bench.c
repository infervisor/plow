/* tp_alltoall_bench.c — prices the two all-to-alls a query-row-split sparse attention needs on
 * the GLM 8192 prefill rung (rowsplit-attention-design.md §3.1 steps 2 and 4), standalone (no
 * packet), sibling to tp_seams_overlap_bench.c and built the same way.
 *
 *   TP_MODE=q     Q all-to-all only:      [8192][8][576]  -> [1024][64][576]  (66 MB/rank moved)
 *   TP_MODE=o     attn-out all-to-all:    [1024][64][512] -> [8192][8][512]   (59 MB/rank moved)
 *   TP_MODE=both  q then o, back to back, one launch (the design's "+42 ms/chunk" number)
 *   TP_MODE=ag    control: unmodified d_xall_gather_mega at the seams-overlap.md §1 shape
 *                 (8192x6144 bf16, 88 MB/rank moved) — must reproduce ~230 GB/s, or the fault is
 *                 in this harness's peer/table setup, not in the q/o kernels
 * env: TP_NWG (workgroup cap: 8/16/24/48/96/152/224/304) TP_REPS (default 5) TP_ELF
 *
 * Every destination element on every rank is checked against the host oracle (a pure
 * permutation of the same word(rank, e) pattern tp_alltoall_kernels.hip's tp_a2a_fill writes —
 * byte-exact, no tolerance).
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define XCTR_BYTES (128u * 64u)
#define T 8192u        /* prefill rows            */
#define RPR 1024u      /* rows per rank (T / NR)  */
#define NHL 8u         /* per-rank local heads    */
#define NHT 64u        /* total heads (NR * NHL)  */
#define DQ 576u         /* Q head dim              */
#define DO 512u         /* attn-output head dim    */
#define AGROWS 8192u    /* control all-gather: seams-overlap.md §1 shape */
#define AGHID 6144u
#define MAXCAP 304u     /* largest workgroup cap swept (one per CU)      */

typedef uint16_t bf16;
static uint32_t tp_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static const uint32_t SEED = 20260912u;
static bf16 word(uint32_t r, uint32_t e) {  /* must match tp_a2a_fill */
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

/* Q source per rank: [T][NHL][DQ]; O source per rank: [RPR][NHT][DO]. Both live in the same
 * peer-visible scratch region, one after the other; xctr counters after both. */
static const size_t S1 = (size_t)T * NHL * DQ * 2u;
static const size_t S2 = (size_t)RPR * NHT * DO * 2u;
static const size_t DSTQ = (size_t)RPR * NHT * DQ * 2u;  /* == S1 */
static const size_t DSTO = (size_t)T * NHL * DO * 2u;    /* == S2 */
/* Control all-gather: production slot convention is a FULL n-sized array per rank (the
 * reduce-scatter's in-place slot, which the all-gather then reads by global index), not a
 * band — see d_xall_gather_mega's `peer_scratch[s] + slot` indexed by the global element e. */
static const size_t AGN = (size_t)AGROWS * AGHID;
static const size_t AGS = AGN * 2u;

typedef struct { const void* peers; uint32_t nranks, rank; size_t xoff; uint32_t gate;
                 uint64_t deadline; void* status; void* dst; uint32_t rpr, nh_l, d, nh_total, slot;
                 void* ts; } a_one;
typedef struct { const void* peers; uint32_t nranks, rank; size_t xoff; uint32_t gate;
                 uint64_t deadline; void* status; void* dst; uint32_t n, slot; void* ts; } a_ag;
typedef struct { const void* peers; uint32_t nranks, rank; size_t xoff; uint64_t deadline;
                 void* status; void* dst_q; uint32_t rpr_q, nhl_q, dq, nht_q, slot_q; void* dst_o;
                 uint32_t rpr_o, nhl_o, do_, nht_o, slot_o; void* ts; } a_both;

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* mode = getenv("TP_MODE") ? getenv("TP_MODE") : "q";
    const uint32_t nwg = envu("TP_NWG", 24), reps = envu("TP_REPS", 5);
    if (strcmp(mode, "q") && strcmp(mode, "o") && strcmp(mode, "both") && strcmp(mode, "ag")) {
        fprintf(stderr, "TP_MODE must be q|o|both|ag\n"); return 2;
    }
    if (nwg > MAXCAP) { fprintf(stderr, "TP_NWG > %u\n", MAXCAP); return 2; }
    const size_t xoff = S1 + S2;
    const size_t region = xoff + XCTR_BYTES;
    const size_t agregion = AGS + XCTR_BYTES;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    size_t elf_len = 0;
    void* elf = slurp(getenv("TP_ELF") ? getenv("TP_ELF") : "alltoall.elf", &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }
    plow_hsa_kernel kfill[NR], kq[NR], ko[NR], kboth[NR], kag[NR];
    void *scratch[NR], *table[NR], *dstq[NR], *dsto[NR], *ts[NR]; uint32_t* status[NR];
    void *agscratch[NR], *agtable[NR], *agdst[NR];
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tp_a2a_fill", &kfill[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_a2a_q_bench", &kq[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_a2a_o_bench", &ko[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_a2a_both_bench", &kboth[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_ag_control_bench", &kag[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], region);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        dstq[r] = plow_hsa_alloc(h, dev[r], DSTQ);
        dsto[r] = plow_hsa_alloc(h, dev[r], DSTO);
        ts[r] = plow_hsa_alloc(h, dev[r], MAXCAP * 3u * 8u);
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        agscratch[r] = plow_hsa_alloc_peer(h, dev[r], agregion);
        agtable[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        agdst[r] = plow_hsa_alloc(h, dev[r], AGS);
        if (!scratch[r] || !table[r] || !dstq[r] || !dsto[r] || !ts[r] || !status[r] ||
            !agscratch[r] || !agtable[r] || !agdst[r]) {
            fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    for (int r = 0; r < NR; r++) {
        plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);
        plow_hsa_upload(h, dev[r], agtable[r], agscratch, sizeof agscratch);
    }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const uint64_t deadline = freq ? freq : 1000000000ull;
    for (int r = 0; r < NR; r++) {
        /* fill Q source at offset 0, O source at offset S1 */
        struct { void* part; uint32_t n, rank, seed; } fq = {scratch[r], (uint32_t)(S1 / 2u), (uint32_t)r, SEED};
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fq, sizeof fq);
        struct { void* part; uint32_t n, rank, seed; } fo = {(char*)scratch[r] + S1, (uint32_t)(S2 / 2u), (uint32_t)r, SEED};
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fo, sizeof fo);
        struct { void* part; uint32_t n, rank, seed; } fag = {agscratch[r], (uint32_t)AGN, (uint32_t)r, SEED};
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fag, sizeof fag);
        plow_hsa_wait(h, dev[r]);
    }
    static uint8_t zero[XCTR_BYTES];
    double *a1 = calloc(reps, sizeof(double)), *a2 = calloc(reps, sizeof(double)), *a3 = calloc(reps, sizeof(double));
    uint64_t* t = malloc(MAXCAP * 3u * 8u);
    int timeout = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (int r = 0; r < NR; r++) {
            if (!strcmp(mode, "ag"))
                plow_hsa_upload(h, dev[r], (char*)agscratch[r] + AGS, zero, sizeof zero);
            else
                plow_hsa_upload(h, dev[r], (char*)scratch[r] + xoff, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            plow_hsa_wait(h, dev[r]);
        }
        for (int r = NR - 1; r >= 0; r--) {
            if (!strcmp(mode, "both")) {
                a_both ab = {table[r], NR, (uint32_t)r, xoff, deadline, status[r],
                             dstq[r], RPR, NHL, DQ, NHT, 0u,
                             dsto[r], RPR, NHL, DO, NHT, (uint32_t)S1, ts[r]};
                plow_hsa_launch(h, dev[r], &kboth[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &ab, sizeof ab);
            } else if (!strcmp(mode, "q")) {
                a_one aq = {table[r], NR, (uint32_t)r, xoff, 1u, deadline, status[r],
                            dstq[r], RPR, NHL, DQ, NHT, 0u, ts[r]};
                plow_hsa_launch(h, dev[r], &kq[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &aq, sizeof aq);
            } else if (!strcmp(mode, "ag")) {
                a_ag aa = {agtable[r], NR, (uint32_t)r, AGS, 1u, deadline, status[r],
                           agdst[r], (uint32_t)AGN, 0u, ts[r]};
                plow_hsa_launch(h, dev[r], &kag[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &aa, sizeof aa);
            } else {
                a_one ao = {table[r], NR, (uint32_t)r, xoff, 2u, deadline, status[r],
                            dsto[r], RPR, NHL, DO, NHT, (uint32_t)S1, ts[r]};
                plow_hsa_launch(h, dev[r], &ko[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &ao, sizeof ao);
            }
        }
        for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        for (int r = 0; r < NR; r++) { uint32_t st = 0; plow_hsa_download(h, dev[r], &st, status[r], 4); timeout |= st != 0; }
        plow_hsa_download(h, dev[0], t, ts[0], (!strcmp(mode, "both") ? nwg * 3u : nwg * 2u) * 8u);
        uint64_t t0 = ~0ull, m1 = 0, m2 = 0;
        if (!strcmp(mode, "both")) {
            for (uint32_t w = 0; w < nwg; w++) {
                if (t[3 * w] < t0) t0 = t[3 * w];
                if (t[3 * w + 1] > m1) m1 = t[3 * w + 1];
                if (t[3 * w + 2] > m2) m2 = t[3 * w + 2];
            }
            a1[rep] = (m1 - t0) * 0.01; a2[rep] = (m2 - m1) * 0.01; a3[rep] = (m2 - t0) * 0.01;
        } else {
            for (uint32_t w = 0; w < nwg; w++) {
                if (t[2 * w] < t0) t0 = t[2 * w];
                if (t[2 * w + 1] > m1) m1 = t[2 * w + 1];
            }
            a3[rep] = (m1 - t0) * 0.01; a1[rep] = a3[rep]; a2[rep] = 0.0;
        }
    }
    qsort(a1, reps, sizeof(double), cmpd); qsort(a2, reps, sizeof(double), cmpd); qsort(a3, reps, sizeof(double), cmpd);

    /* Oracle on every rank. */
    size_t bad = 0;
    bf16* hbq = malloc(DSTQ);
    bf16* hbo = malloc(DSTO);
    bf16* hbag = !strcmp(mode, "ag") ? malloc(AGS) : NULL;
    for (int r = 0; r < NR; r++) {
        if (!strcmp(mode, "ag")) {
            plow_hsa_download(h, dev[r], hbag, agdst[r], AGS);
            for (size_t e = 0; e < AGN; e++) {
                const uint32_t owner = (uint32_t)((e * NR) / AGN);
                bad += hbag[e] != word(owner, (uint32_t)e);
            }
            continue;
        }
        if (strcmp(mode, "o")) {
            plow_hsa_download(h, dev[r], hbq, dstq[r], DSTQ);
            for (uint32_t row = 0; row < RPR; row++) {
                const uint32_t grow = (uint32_t)r * RPR + row;
                for (uint32_t head = 0; head < NHT; head++) {
                    const uint32_t p = head / NHL, lh = head % NHL;
                    const uint32_t e = (grow * NHL + lh) * DQ;
                    for (uint32_t d = 0; d < DQ; d++)
                        bad += hbq[(row * NHT + head) * DQ + d] != word(p, e + d);
                }
            }
        }
        if (strcmp(mode, "q")) {
            plow_hsa_download(h, dev[r], hbo, dsto[r], DSTO);
            for (uint32_t grow = 0; grow < T; grow++) {
                const uint32_t p = grow / RPR, lrow = grow % RPR;
                for (uint32_t lh = 0; lh < NHL; lh++) {
                    const uint32_t head = (uint32_t)r * NHL + lh;
                    const uint32_t e = (lrow * NHT + head) * DO;
                    for (uint32_t d = 0; d < DO; d++)
                        bad += hbo[(grow * NHL + lh) * DO + d] != word(p, e + d);
                }
            }
        }
    }
    const uint32_t mid = reps / 2;
    const double moved_q = (double)DSTQ * 7.0 / 8.0, moved_o = (double)DSTO * 7.0 / 8.0;
    const double moved_ag = (double)AGS * 7.0 / 8.0;
    if (!strcmp(mode, "both"))
        printf("mode=both nwg=%u q_us=%.1f o_us=%.1f total_us=%.1f GBps=%.1f",
               nwg, a1[mid], a2[mid], a3[mid], (moved_q + moved_o) / (a3[mid] * 1e3));
    else if (!strcmp(mode, "q"))
        printf("mode=q nwg=%u total_us=%.1f GBps=%.1f", nwg, a3[mid], moved_q / (a3[mid] * 1e3));
    else if (!strcmp(mode, "ag"))
        printf("mode=ag nwg=%u total_us=%.1f GBps=%.1f", nwg, a3[mid], moved_ag / (a3[mid] * 1e3));
    else
        printf("mode=o nwg=%u total_us=%.1f GBps=%.1f", nwg, a3[mid], moved_o / (a3[mid] * 1e3));
    printf(" reps=%u parity=%s bad=%zu timeout=%s\n", reps, bad ? "FAIL" : "PASS", bad, timeout ? "YES" : "no");
    plow_hsa_shutdown(h);
    return (bad || timeout) ? 1 : 0;
}
