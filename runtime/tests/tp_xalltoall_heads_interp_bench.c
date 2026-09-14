/* tp_xalltoall_heads_interp_bench.c — DevOp::XAllToAllHeads (op 160) driven through the SAME
 * `PlowDevInst` decode `runtime/amd/interp.hip`'s real dispatch arm uses (tier-1/tier-2 gate for
 * the op's OWN ABI wiring, sibling to tp_alltoall_bench.c which already priced the device body
 * standalone, byte-exact, at 291.2/257.8 us for Q/O at a 48-workgroup cap, job
 * 0-1789234094-a2a-collective-bench3).
 *
 *   TP_MODE=q   Q form  (dir=0): [8192][8][576]  per rank -> [1024][64][576] per rank
 *   TP_MODE=o   O form  (dir=1): [1024][64][512] per rank -> [8192][8][512]  per rank
 * env: TP_NWG (workgroup cap == in.blocks; default 48, must equal PLOW_XA2A_NWG for every
 *      launched workgroup to be "active") TP_REPS (default 5) TP_ELF
 *
 * Every destination element on every rank checked against the host oracle (a pure permutation
 * of the same word(rank, e) pattern tp_alltoall_kernels.hip's tp_a2a_fill / this file's txh_fill
 * write) — byte-exact, no tolerance.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define XCTR_BYTES (128u * 64u)
#define T 8192u
#define RPR 1024u
#define NHL 8u
#define NHT 64u
#define DQ 576u
#define DO 512u

typedef uint16_t bf16;
static uint32_t tp_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static const uint32_t SEED = 20260912u;
static bf16 word(uint32_t r, uint32_t e) {  /* must match txh_fill */
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

/* Q source per rank: [T][NHL][DQ]; O source per rank: [RPR][NHT][DO]. Same peer-visible
 * region layout as tp_alltoall_bench.c: Q source at offset 0, O source at offset S1. */
static const size_t S1 = (size_t)T * NHL * DQ * 2u;
static const size_t S2 = (size_t)RPR * NHT * DO * 2u;
static const size_t DSTQ = (size_t)RPR * NHT * DQ * 2u;  /* == S1 */
static const size_t DSTO = (size_t)T * NHL * DO * 2u;    /* == S2 */

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* mode = getenv("TP_MODE") ? getenv("TP_MODE") : "q";
    const uint32_t nwg = envu("TP_NWG", 48), reps = envu("TP_REPS", 5);
    const int is_o = !strcmp(mode, "o");
    if (strcmp(mode, "q") && !is_o) { fprintf(stderr, "TP_MODE must be q|o\n"); return 2; }
    const size_t xoff = S1 + S2;
    const size_t region = xoff + XCTR_BYTES;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    size_t elf_len = 0;
    void* elf = slurp(getenv("TP_ELF") ? getenv("TP_ELF") : "xalltoall_heads.elf", &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }
    plow_hsa_kernel kfill[NR], kdisp[NR];
    void *scratch[NR], *table[NR], *dst[NR], *ts[NR], *tens[NR]; uint32_t* status[NR];
    const size_t dst_bytes = is_o ? DSTO : DSTQ;
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "txh_fill", &kfill[r]) ||
            plow_hsa_get_kernel(h, dev[r], "txh_dispatch_bench", &kdisp[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], region);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        dst[r] = plow_hsa_alloc(h, dev[r], dst_bytes);
        tens[r] = plow_hsa_alloc(h, dev[r], sizeof(void*)); /* tens[0] = dst */
        ts[r] = plow_hsa_alloc(h, dev[r], nwg * 2u * 8u);
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !table[r] || !dst[r] || !tens[r] || !ts[r] || !status[r]) {
            fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    for (int r = 0; r < NR; r++) {
        plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);
        plow_hsa_upload(h, dev[r], tens[r], &dst[r], sizeof(void*));
    }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const uint64_t deadline = freq ? freq : 1000000000ull;
    for (int r = 0; r < NR; r++) {
        struct { void* part; uint32_t n, rank, seed; } fq = {scratch[r], (uint32_t)(S1 / 2u), (uint32_t)r, SEED};
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fq, sizeof fq);
        struct { void* part; uint32_t n, rank, seed; } fo = {(char*)scratch[r] + S1, (uint32_t)(S2 / 2u), (uint32_t)r, SEED};
        plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fo, sizeof fo);
        plow_hsa_wait(h, dev[r]);
    }

    /* One PlowDevInst per direction, the operand mapping devgen's emit_xalltoall_heads (and
     * interp.hip's PLOW_DOP_XALLTOALL_HEADS arm) use:
     *   i0=rpr i1=nh_l i2=d i3=nh_total i4=gate i5=n_gpu i6=slot_bytes i7=dir  t0=dst */
    PlowDevInst in; memset(&in, 0, sizeof in);
    in.op = PLOW_DOP_XALLTOALL_HEADS;
    in.blocks = (uint16_t)nwg;
    in.t[0] = 0;
    in.i[0] = RPR; in.i[1] = NHL; in.i[2] = is_o ? DO : DQ; in.i[3] = NHT;
    in.i[4] = 1u; in.i[5] = NR; in.i[6] = is_o ? (uint32_t)S1 : 0u; in.i[7] = is_o ? 1u : 0u;

    static uint8_t zero[XCTR_BYTES];
    double* a = calloc(reps, sizeof(double));
    uint64_t* t = malloc((size_t)nwg * 2u * 8u);
    int timeout = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + xoff, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            plow_hsa_wait(h, dev[r]);
        }
        for (int r = NR - 1; r >= 0; r--) {
            struct { const void* in; const void* tens; const void* peers; uint32_t n_gpu, rank;
                     size_t xoff; uint64_t deadline; void* status; void* ts; } a_in = {
                &in, tens[r], table[r], NR, (uint32_t)r, xoff, deadline, status[r], ts[r]};
            plow_hsa_launch(h, dev[r], &kdisp[r], nwg * 512u, 1, 1, 512, 1, 1, 0, &a_in, sizeof a_in);
        }
        for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        for (int r = 0; r < NR; r++) { uint32_t st = 0; plow_hsa_download(h, dev[r], &st, status[r], 4); timeout |= st != 0; }
        plow_hsa_download(h, dev[0], t, ts[0], nwg * 2u * 8u);
        uint64_t t0 = ~0ull, m1 = 0;
        for (uint32_t w = 0; w < nwg; w++) {
            if (t[2 * w] < t0) t0 = t[2 * w];
            if (t[2 * w + 1] > m1) m1 = t[2 * w + 1];
        }
        a[rep] = (m1 - t0) * 0.01;
    }
    qsort(a, reps, sizeof(double), cmpd);

    size_t bad = 0;
    bf16* hb = malloc(dst_bytes);
    for (int r = 0; r < NR; r++) {
        plow_hsa_download(h, dev[r], hb, dst[r], dst_bytes);
        if (!is_o) {
            for (uint32_t row = 0; row < RPR; row++) {
                const uint32_t grow = (uint32_t)r * RPR + row;
                for (uint32_t head = 0; head < NHT; head++) {
                    const uint32_t p = head / NHL, lh = head % NHL;
                    const uint32_t e = (grow * NHL + lh) * DQ;
                    for (uint32_t d = 0; d < DQ; d++)
                        bad += hb[(row * NHT + head) * DQ + d] != word(p, e + d);
                }
            }
        } else {
            for (uint32_t grow = 0; grow < T; grow++) {
                const uint32_t p = grow / RPR, lrow = grow % RPR;
                for (uint32_t lh = 0; lh < NHL; lh++) {
                    const uint32_t head = (uint32_t)r * NHL + lh;
                    const uint32_t e = (lrow * NHT + head) * DO;
                    for (uint32_t d = 0; d < DO; d++)
                        bad += hb[(grow * NHL + lh) * DO + d] != word(p, e + d);
                }
            }
        }
    }
    const uint32_t mid = reps / 2;
    const double moved = (double)(is_o ? DSTO : DSTQ) * 7.0 / 8.0;
    printf("mode=%s nwg=%u total_us=%.1f GBps=%.1f reps=%u parity=%s bad=%zu timeout=%s\n",
           mode, nwg, a[mid], moved / (a[mid] * 1e3), reps, bad ? "FAIL" : "PASS", bad,
           timeout ? "YES" : "no");
    plow_hsa_shutdown(h);
    return (bad || timeout) ? 1 : 0;
}
