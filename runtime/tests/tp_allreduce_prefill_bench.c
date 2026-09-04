/* Exact gfx950 TP8 prefill XReduceTwoShot benchmark.
 * Uses the same peer-visible HSA allocation and device body as plowrt. */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define DEFAULT_NWG 256
#define DEFAULT_HIDDEN 7168
#define GATHER_OFF 117440512u
#define XCTR_OFF 132120576u
#define REGION_BYTES (127u * 1024u * 1024u)

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
    void* out; const void* peers; uint32_t nranks, rank, n, slot_bytes;
    uint64_t xctr_byte_off; uint32_t iters; uint64_t deadline;
    void* cycles; void* status;
} arg_xr2;
typedef struct {
    void* out; const void* peers; uint32_t nranks, rank, n, slot_bytes;
    uint64_t xctr_byte_off; uint32_t iters; uint64_t deadline;
    void* cycles; void* status; uint32_t gslot_bytes, gcols;
} arg_xr2g;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}

static uint32_t env_u32(const char* name, uint32_t fallback, uint32_t min,
                        uint32_t max) {
    const char* text = getenv(name);
    if (!text) return fallback;
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || end == text || *end || value < min || value > max) {
        fprintf(stderr, "%s must be an integer in [%u, %u], got '%s'\n",
                name, min, max, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char** argv) {
    const int check_config = argc == 2 && strcmp(argv[1], "--check-config") == 0;
    if (!check_config && argc != NR + 1) {
        fprintf(stderr, "usage: %s gpu0 ... gpu7 | --check-config\n", argv[0]); return 2;
    }
    const uint32_t iters = env_u32("TP_ITERS", 25u, 1u, UINT32_MAX);
    const char* elf_path = getenv("TP_ELF") ? getenv("TP_ELF") : "tp_allreduce_kernels.elf";
    const uint32_t rows = env_u32("TP_ROWS", 0u, 0u, UINT32_MAX);
    const uint32_t hidden = env_u32("TP_HIDDEN", DEFAULT_HIDDEN, 1u, UINT32_MAX);
    const uint32_t nwg = env_u32("TP_NWG", DEFAULT_NWG, 1u, UINT32_MAX / 512u);
    const int gather = getenv("TP_GATHER") && atoi(getenv("TP_GATHER")) != 0;
    const int oneshot = getenv("TP_ONESHOT") && atoi(getenv("TP_ONESHOT")) != 0;
    const uint64_t n0 = (uint64_t)(rows ? rows : 512u) * hidden;
    const uint64_t n1 = rows ? 0u : (uint64_t)1024u * hidden;
    if (n0 > UINT32_MAX || n1 > UINT32_MAX || 2u * (n1 ? n1 : n0) > GATHER_OFF ||
        (gather && (hidden % NR != 0 || 2u * (n1 ? n1 : n0) / NR > XCTR_OFF - GATHER_OFF))) {
        fprintf(stderr, "unsupported shape: rows=%u hidden=%u gather=%d\n",
                rows, hidden, gather);
        return 2;
    }
    const uint32_t shape[] = {(uint32_t)n0, (uint32_t)n1};
    if (check_config) {
        for (unsigned si = 0; si < sizeof shape / sizeof shape[0] && shape[si]; si++)
            printf("rows=%u hidden=%u n=%u nblk=%u gather=%d oneshot=%d iters=%u\n",
                   shape[si] / hidden, hidden, shape[si], nwg, gather, oneshot, iters);
        return 0;
    }

    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    size_t elf_len = 0; void* elf = slurp(elf_path, &elf_len);
    if (!elf) { fprintf(stderr, "cannot open %s\n", elf_path); return 2; }

    plow_hsa_kernel fill[NR], xr2[NR];
    void *scratch[NR], *out[NR], *table[NR];
    uint32_t* status[NR];
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "tp_fill_partial", &fill[r]) ||
            plow_hsa_get_kernel(h, dev[r], gather ? (oneshot ? "tp_allreduce_oneshot_gather"
                                                             : "tp_allreduce_twoshot_gather")
                                                  : "tp_allreduce_twoshot", &xr2[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], REGION_BYTES);
        const uint32_t max_n = shape[1] ? shape[1] : shape[0];
        out[r] = plow_hsa_alloc(h, dev[r], (size_t)max_n * 2u);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !out[r] || !table[r] || !status[r]) {
            fprintf(stderr, "alloc rank=%d scratch=%p out=%p table=%p status=%p: %s\n",
                    r, scratch[r], out[r], table[r], status[r], plow_hsa_last_error()); return 2;
        }
    }
    for (int r = 0; r < NR; r++)
        plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);
    void* cycles = plow_hsa_alloc(h, dev[0], 8);
    const uint32_t max_n = shape[1] ? shape[1] : shape[0];
    bf16* host = (bf16*)malloc((size_t)max_n * 2u);
    const uint64_t deadline = freq ? freq : 1000000000ull;

    for (unsigned si = 0; si < sizeof shape / sizeof shape[0] && shape[si]; si++) {
        const uint32_t n = shape[si]; uint8_t zero[128 * 64] = {0};
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + XCTR_OFF, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            arg_fill a = {scratch[r], n, (uint32_t)r};
            plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0, &a, sizeof a);
            if (gather) {
                arg_fill g = {(char*)scratch[r] + GATHER_OFF, n / NR, (uint32_t)r};
                plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0,
                                &g, sizeof g);
            }
            plow_hsa_wait(h, dev[r]);
        }
        arg_xr2 a[NR];
        arg_xr2g ag[NR];
        for (int r = NR - 1; r >= 0; r--) {
            a[r] = (arg_xr2){out[r], table[r], NR, (uint32_t)r, n, 0, XCTR_OFF,
                              iters, deadline, r ? NULL : cycles, status[r]};
            ag[r] = (arg_xr2g){out[r], table[r], NR, (uint32_t)r, n, 0, XCTR_OFF,
                                iters, deadline, r ? NULL : cycles, status[r],
                                GATHER_OFF, hidden / NR};
            void* ka = gather ? (void*)&ag[r] : (void*)&a[r];
            size_t kaz = gather ? sizeof ag[r] : sizeof a[r];
            plow_hsa_launch(h, dev[r], &xr2[r], nwg * 512, 1, 1, 512, 1, 1, 0, ka, kaz);
        }
        for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        uint64_t ticks = 0; plow_hsa_download(h, dev[0], &ticks, cycles, 8);
        int timeout = 0;
        for (int r = 0; r < NR; r++) {
            uint32_t st = 0; plow_hsa_download(h, dev[r], &st, status[r], 4); timeout |= st != 0;
        }

        /* Refill and run once: repeated timing iterations intentionally overwrite partials. */
        memset(zero, 0, sizeof zero);
        for (int r = 0; r < NR; r++) {
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + XCTR_OFF, zero, sizeof zero);
            arg_fill f = {scratch[r], n, (uint32_t)r};
            plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0, &f, sizeof f);
            if (gather) {
                arg_fill g = {(char*)scratch[r] + GATHER_OFF, n / NR, (uint32_t)r};
                plow_hsa_launch(h, dev[r], &fill[r], 4096, 1, 1, 256, 1, 1, 0,
                                &g, sizeof g);
            }
            plow_hsa_wait(h, dev[r]);
            a[r].iters = 1; a[r].cycles = NULL;
            ag[r].iters = 1; ag[r].cycles = NULL;
        }
        for (int r = NR - 1; r >= 0; r--) {
            void* ka = gather ? (void*)&ag[r] : (void*)&a[r];
            size_t kaz = gather ? sizeof ag[r] : sizeof a[r];
            plow_hsa_launch(h, dev[r], &xr2[r], nwg * 512, 1, 1, 512, 1, 1, 0,
                            ka, kaz);
        }
        for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        size_t bad = 0;
        for (int rr = 0; rr < NR; rr++) {
            plow_hsa_download(h, dev[rr], host, out[rr], (size_t)n * 2u);
            for (uint32_t e = 0; e < n; e++) {
                float sum = 0;
                for (uint32_t r = 0; r < NR; r++) sum += bf2f(partial_val(r, e));
                bf16 want = f2bf(sum);
                if (gather) {
                    const uint32_t c = e % hidden, m = e / hidden;
                    const uint32_t owner = c / (hidden / NR);
                    const uint32_t ge = m * (hidden / NR) + c % (hidden / NR);
                    want = f2bf(bf2f(want) + bf2f(partial_val(owner, ge)));
                }
                bad += host[e] != want;
            }
        }
        const double us = (double)ticks * (1e9 / (double)freq) / (double)iters / 1e3;
        printf("rows=%u n=%u nblk=%u gather=%d oneshot=%d %.3f us/collective parity=%s timeout=%s bad=%zu\n",
               n / hidden, n, nwg, gather, oneshot, us, bad ? "FAIL" : "PASS", timeout ? "YES" : "no", bad);
        if (bad || timeout) return 1;
    }
    plow_hsa_shutdown(h); return 0;
}
