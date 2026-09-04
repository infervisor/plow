/* tp_seqpar_smoke.c — the sequence-parallel seam halves (ops 25/26) end to end on 1..8 GPUs.
 *
 * Same peer-visible allocation and device bodies as plowrt (`d_xreduce_scatter_mega`,
 * `d_xall_gather_mega` through tp_allreduce_kernels.hip). Checks, per rank:
 *   1. the reduce-scatter's owned rows in slot 0 equal the two-shot's reduced+folded value
 *      (`f2bf(bf2f(f2bf(sum_r partial_r)) + bf2f(gathered))`), i.e. the rounding contract;
 *   2. the local band copy equals those rows;
 *   3. the all-gather assembles every rank's owned rows into the full array, for two arrays
 *      under one rendezvous.
 * A ONE-rank run is a valid smoke of the arithmetic and the dispatch (the rendezvous is
 * trivially satisfied); the protocol needs the multi-rank form.
 *
 * usage: tp_seqpar_smoke gpu0 [gpu1 ...]   env TP_ROWS (default 64) TP_HIDDEN (default 7168)
 *        TP_NWG (default 64) TP_ELF (default tp_allreduce_kernels.elf) */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXR 8
#define GATHER_OFF (32u * 1024u * 1024u)
#define XCTR_OFF (48u * 1024u * 1024u)
#define REGION_BYTES (49u * 1024u * 1024u)

typedef uint16_t bf16;
static float bf2f(bf16 b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u); return (bf16)(u >> 16);
}
static bf16 partial_val(uint32_t r, uint32_t e) {
    return f2bf((float)(r + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}
typedef struct { void* part; uint32_t n; uint32_t rank; } arg_fill;
typedef struct {
    const void* peers; uint32_t nranks, rank, n, slot_bytes; uint64_t xctr_byte_off;
    uint32_t gate; uint64_t deadline; void* status; uint32_t gslot_bytes, gcols;
    void* band_copy;
} arg_rs;
typedef struct {
    const void* peers; uint32_t nranks, rank; uint64_t xctr_byte_off; uint32_t gate;
    uint64_t deadline; void* status;
    void* dst0; uint32_t n0, slot0; void* dst1; uint32_t n1, slot1; void* dst2; uint32_t n2, slot2;
} arg_ag;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* t = getenv(name); if (!t) return fallback;
    char* end = NULL; errno = 0; unsigned long v = strtoul(t, &end, 10);
    if (errno || end == t || *end || v == 0) { fprintf(stderr, "%s: bad value '%s'\n", name, t); exit(2); }
    return (uint32_t)v;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > MAXR + 1) { fprintf(stderr, "usage: %s gpu0 [gpu1 ...]\n", argv[0]); return 2; }
    const uint32_t nr = (uint32_t)(argc - 1);
    const uint32_t rows = env_u32("TP_ROWS", 64u), hidden = env_u32("TP_HIDDEN", 7168u);
    const uint32_t nwg = env_u32("TP_NWG", 64u);
    const char* elf_path = getenv("TP_ELF") ? getenv("TP_ELF") : "tp_allreduce_kernels.elf";
    if (rows % nr != 0 || hidden % nr != 0) { fprintf(stderr, "rows and hidden must divide %u\n", nr); return 2; }
    const uint32_t n = rows * hidden, gcols = hidden / nr, band = n / nr;
    if ((uint64_t)n * 2u > GATHER_OFF || (uint64_t)rows * gcols * 2u > XCTR_OFF - GATHER_OFF) {
        fprintf(stderr, "shape too large for the region\n"); return 2;
    }
    int dev[MAXR]; for (uint32_t r = 0; r < nr; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    size_t elf_len = 0; void* elf = slurp(elf_path, &elf_len);
    if (!elf) { fprintf(stderr, "cannot open %s\n", elf_path); return 2; }

    plow_hsa_kernel fill[MAXR], krs[MAXR], kag[MAXR];
    void *scratch[MAXR], *out[MAXR], *out2[MAXR], *copy[MAXR], *table[MAXR];
    uint32_t* status[MAXR];
    for (uint32_t r = 0; r < nr; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tp_fill_partial", &fill[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_seqpar_rs", &krs[r]) ||
            plow_hsa_get_kernel(h, dev[r], "tp_seqpar_ag", &kag[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], REGION_BYTES);
        out[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        out2[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        copy[r] = plow_hsa_alloc(h, dev[r], (size_t)band * 2u);
        table[r] = plow_hsa_alloc(h, dev[r], MAXR * sizeof(void*));
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !out[r] || !out2[r] || !copy[r] || !table[r] || !status[r]) {
            fprintf(stderr, "alloc rank=%u: %s\n", r, plow_hsa_last_error()); return 2;
        }
    }
    for (uint32_t r = 0; r < nr; r++) plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);
    const uint64_t deadline = freq ? 5 * freq : 5000000000ull;

    uint8_t zero[128 * 64] = {0};
    for (uint32_t r = 0; r < nr; r++) {
        plow_hsa_upload(h, dev[r], (char*)scratch[r] + XCTR_OFF, zero, sizeof zero);
        plow_hsa_upload(h, dev[r], status[r], zero, 4);
        arg_fill a = {scratch[r], n, r};
        plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0, &a, sizeof a);
        arg_fill g = {(char*)scratch[r] + GATHER_OFF, rows * gcols, r};
        plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0, &g, sizeof g);
        plow_hsa_wait(h, dev[r]);
    }
    /* Op 25 on every rank (gate 0), then op 26 (gate 1): the packet order. */
    for (int r = (int)nr - 1; r >= 0; r--) {
        arg_rs a = {table[r], nr, (uint32_t)r, n, 0, XCTR_OFF, 0, deadline, status[r],
                    GATHER_OFF, gcols, copy[r]};
        plow_hsa_launch(h, dev[r], &krs[r], nwg * 512, 1, 1, 512, 1, 1, 0, &a, sizeof a);
    }
    for (uint32_t r = 0; r < nr; r++) plow_hsa_wait(h, dev[r]);
    for (int r = (int)nr - 1; r >= 0; r--) {
        arg_ag a = {table[r], nr, (uint32_t)r, XCTR_OFF, 1, deadline, status[r],
                    out[r], n, 0, out2[r], n, 0, NULL, 0, 0};
        plow_hsa_launch(h, dev[r], &kag[r], nwg * 512, 1, 1, 512, 1, 1, 0, &a, sizeof a);
    }
    for (uint32_t r = 0; r < nr; r++) plow_hsa_wait(h, dev[r]);

    int timeout = 0;
    for (uint32_t r = 0; r < nr; r++) {
        uint32_t st = 0; plow_hsa_download(h, dev[r], &st, status[r], 4); timeout |= st != 0;
    }
    bf16* want = (bf16*)malloc((size_t)n * 2u);
    for (uint32_t e = 0; e < n; e++) {
        float sum = 0;
        for (uint32_t r = 0; r < nr; r++) sum += bf2f(partial_val(r, e));
        const uint32_t c = e % hidden, m = e / hidden, owner = c / gcols;
        want[e] = f2bf(bf2f(f2bf(sum)) + bf2f(partial_val(owner, m * gcols + c % gcols)));
    }
    bf16* host = (bf16*)malloc((size_t)n * 2u);
    size_t bad_slot = 0, bad_copy = 0, bad_ag = 0, bad_ag2 = 0;
    for (uint32_t r = 0; r < nr; r++) {
        const uint32_t lo = r * band;
        plow_hsa_download(h, dev[r], host, scratch[r], (size_t)n * 2u);
        for (uint32_t e = lo; e < lo + band; e++) bad_slot += host[e] != want[e];
        plow_hsa_download(h, dev[r], host, copy[r], (size_t)band * 2u);
        for (uint32_t e = 0; e < band; e++) bad_copy += host[e] != want[lo + e];
        plow_hsa_download(h, dev[r], host, out[r], (size_t)n * 2u);
        for (uint32_t e = 0; e < n; e++) bad_ag += host[e] != want[e];
        plow_hsa_download(h, dev[r], host, out2[r], (size_t)n * 2u);
        for (uint32_t e = 0; e < n; e++) bad_ag2 += host[e] != want[e];
    }
    printf("seqpar smoke: ranks=%u rows=%u hidden=%u n=%u nwg=%u timeout=%d "
           "bad_slot=%zu bad_copy=%zu bad_ag=%zu bad_ag2=%zu -> %s\n",
           nr, rows, hidden, n, nwg, timeout, bad_slot, bad_copy, bad_ag, bad_ag2,
           (!timeout && !bad_slot && !bad_copy && !bad_ag && !bad_ag2) ? "PASS" : "FAIL");
    plow_hsa_shutdown(h);
    return (!timeout && !bad_slot && !bad_copy && !bad_ag && !bad_ag2) ? 0 : 1;
}
