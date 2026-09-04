#include "hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8u
#define GRID 14u
#define HIDDEN 7168u
#define NBCAP 9u
#define LINE 32u
#define MAXB 16u

typedef uint16_t bf16;
typedef struct {
    float stats[NR][2u * (MAXB + 1u)];
    float probs[MAXB + 1u];
    float norm[NR];
    float inv;
    uint32_t phase[5][LINE];
    uint32_t local_arrive[NR][LINE];
    uint32_t local_done[NR][LINE];
    uint32_t control[2][LINE];
    uint32_t successor[LINE];
} D6Sync;

typedef struct {
    void* reduced;
    void* prefix;
    void* mixed;
    void* block_residual;
    const void* score_weight;
    const void* gamma;
    const void* residual_other;
    const void* residual_pre;
    const void* push_src;
    const void* peer_scratch;
    const void* peer_gate;
    void* sync;
    void* cycles;
    void* status;
    uint32_t nranks;
    uint32_t rank;
    uint32_t hidden;
    uint32_t nb;
    uint32_t nb_cap;
    uint32_t contract;
    uint32_t push;
    uint32_t use_gamma;
    uint32_t iters;
    uint64_t deadline;
    float eps;
} D6Args;
_Static_assert(sizeof(D6Args) == 168, "D6Args ABI");
_Static_assert(sizeof(D6Sync) == 4264, "D6Sync ABI");

typedef struct { void* out; uint32_t n, kind, rank; } FillBf16;
typedef struct { void* out; uint32_t n; } FillF32;
typedef struct { const void* args; } ArgPtr;

typedef struct {
    void *partial[NR], *gate[NR], *peer_partial[NR], *peer_gate[NR];
    void *reduced[NR], *prefix[NR], *mixed[NR], *blockres[NR];
    void *score[NR], *gamma[NR], *other[NR], *pre[NR], *push[NR];
    void *sync[NR], *argdev[NR], *cycles[NR];
    uint32_t* status[NR];
} Buffers;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static float bf2f(bf16 b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static bf16 f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u); return (bf16)(u >> 16);
}
static bf16 partial_value(uint32_t rank, uint32_t i) {
    return f2bf((float)(rank + 1u) * (0.25f + (float)(i & 15u) * 0.03125f));
}

static void fill_inputs(plow_hsa* h, const int* dev, plow_hsa_kernel* kb,
                        plow_hsa_kernel* kf, Buffers* b) {
    for (uint32_t r = 0; r < NR; ++r) {
        FillBf16 fp = {b->partial[r], HIDDEN, 0, r};
        FillBf16 fo = {b->other[r], HIDDEN, 1, r};
        FillBf16 fg = {b->gamma[r], HIDDEN, 2, r};
        FillBf16 fx = {b->pre[r], HIDDEN, 3, r};
        FillBf16 fpush = {b->push[r], HIDDEN, 1, r};
        FillBf16 fbr = {b->blockres[r], NBCAP * HIDDEN, 3, r};
        FillF32 fs = {b->score[r], HIDDEN};
        FillBf16 all[] = {fp, fo, fg, fx, fpush, fbr};
        for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); ++i)
            plow_hsa_launch(h, dev[r], &kb[r], 4096, 1, 1, 256, 1, 1, 0,
                            &all[i], sizeof(all[i]));
        plow_hsa_launch(h, dev[r], &kf[r], 4096, 1, 1, 256, 1, 1, 0, &fs, sizeof fs);
        plow_hsa_wait(h, dev[r]);
    }
}

static double run(plow_hsa* h, const int* dev, plow_hsa_kernel* k, int fused, Buffers* b,
                  uint32_t contract, uint32_t nb, uint32_t push, uint32_t gamma,
                  uint32_t iters, uint64_t deadline, double tick_ns) {
    uint32_t zline[LINE] = {0};
    D6Sync zs; memset(&zs, 0, sizeof zs);
    D6Args args[NR];
    for (uint32_t r = 0; r < NR; ++r) {
        plow_hsa_upload(h, dev[r], b->gate[r], zline, sizeof zline);
        plow_hsa_upload(h, dev[r], b->sync[r], &zs, sizeof zs);
        plow_hsa_upload(h, dev[r], b->status[r], zline, 4);
        plow_hsa_upload(h, dev[r], b->cycles[r], zline, 8);
        args[r] = (D6Args){
            b->reduced[r], b->prefix[r], b->mixed[r], b->blockres[r], b->score[r],
            b->gamma[r], b->other[r], b->pre[r], b->push[r], b->peer_partial[r],
            b->peer_gate[r], b->sync[r], b->cycles[r], b->status[r], NR, r, HIDDEN,
            nb, NBCAP, contract, push, gamma, iters, deadline, 1.0e-6f};
        plow_hsa_upload(h, dev[r], b->argdev[r], &args[r], sizeof args[r]);
    }
    ArgPtr kp[NR];
    for (uint32_t r = NR - 1; r >= 1; --r) {
        kp[r].args = b->argdev[r];
        plow_hsa_launch(h, dev[r], &k[r], GRID * 512, 1, 1, 512, 1, 1, 0,
                        &kp[r], sizeof kp[r]);
    }
    kp[0].args = b->argdev[0];
    plow_hsa_launch(h, dev[0], &k[0], GRID * 512, 1, 1, 512, 1, 1, 0,
                    &kp[0], sizeof kp[0]);
    for (uint32_t r = 0; r < NR; ++r) plow_hsa_wait(h, dev[r]);
    for (uint32_t r = 0; r < NR; ++r) {
        uint32_t st = 0; D6Sync got;
        plow_hsa_download(h, dev[r], &st, b->status[r], 4);
        plow_hsa_download(h, dev[r], &got, b->sync[r], sizeof got);
        uint32_t gate = 0; plow_hsa_download(h, dev[r], &gate, b->gate[r], 4);
        int counters_bad = got.successor[0] != iters * GRID || gate != iters * NR;
        if (fused) {
            counters_bad |= got.phase[0][0] != iters * NR;
            counters_bad |= got.phase[1][0] != iters * NR;
            counters_bad |= got.phase[2][0] != iters;
            counters_bad |= got.phase[3][0] != (gamma ? iters * NR : 0u);
            counters_bad |= got.phase[4][0] != (gamma ? iters : 0u);
            for (uint32_t d = 0; d < NR; ++d) {
                const uint32_t nper = d < 6u ? 2u : 1u;
                counters_bad |= got.local_arrive[d][0] != iters * nper;
                counters_bad |= got.local_done[d][0] != iters;
            }
        } else {
            counters_bad |= got.control[0][0] != iters * GRID;
            counters_bad |= got.control[1][0] != iters;
        }
        if (st || counters_bad) {
            fprintf(stderr, "status rank%u=0x%08x successor=%u want=%u peer_gate=%u want=%u\n",
                    r, st, got.successor[0], iters * GRID, gate, iters * NR);
            exit(1);
        }
    }
    uint64_t cyc = 0;
    for (uint32_t r = 0; r < NR; ++r) {
        uint64_t rank_cyc = 0;
        plow_hsa_download(h, dev[r], &rank_cyc, b->cycles[r], 8);
        if (rank_cyc > cyc) cyc = rank_cyc;
    }
    return (double)cyc * tick_ns / (double)iters / 1.0e3;
}

static int compare_case(plow_hsa* h, const int* dev, plow_hsa_kernel* kc,
                        plow_hsa_kernel* kf, plow_hsa_kernel* kfillb,
                        plow_hsa_kernel* kfillf, Buffers* b, uint32_t contract,
                        uint32_t nb, uint32_t push, uint32_t gamma,
                        uint64_t deadline, double tick_ns) {
    const size_t hbytes = HIDDEN * sizeof(bf16);
    const size_t bbytes = NBCAP * hbytes;
    bf16 *ref_mix = malloc(hbytes), *ref_pre = malloc(hbytes), *ref_red = malloc(hbytes);
    bf16 *ref_blk = malloc(bbytes), *got = malloc(bbytes);
    if (!ref_mix || !ref_pre || !ref_red || !ref_blk || !got) exit(2);
    fill_inputs(h, dev, kfillb, kfillf, b);
    run(h, dev, kc, 0, b, contract, nb, push, gamma, 1, deadline, tick_ns);
    plow_hsa_download(h, dev[0], ref_mix, b->mixed[0], hbytes);
    plow_hsa_download(h, dev[0], ref_pre, b->prefix[0], hbytes);
    plow_hsa_download(h, dev[0], ref_red, b->reduced[0], hbytes);
    plow_hsa_download(h, dev[0], ref_blk, b->blockres[0], bbytes);

    int fail = 0;
    for (uint32_t i = 0; i < HIDDEN; ++i) {
        float sum = 0.0f;
        for (uint32_t r = 0; r < NR; ++r) sum += bf2f(partial_value(r, i));
        if (ref_red[i] != f2bf(sum)) { fail = 1; fprintf(stderr, "reduce mismatch i=%u\n", i); break; }
    }
    fill_inputs(h, dev, kfillb, kfillf, b);
    run(h, dev, kf, 1, b, contract, nb, push, gamma, 1, deadline, tick_ns);
    for (uint32_t r = 0; r < NR && !fail; ++r) {
        plow_hsa_download(h, dev[r], got, b->mixed[r], hbytes);
        if (memcmp(ref_mix, got, hbytes)) { fail = 1; fprintf(stderr, "mix mismatch rank%u c%u nb%u p%u g%u\n", r, contract, nb, push, gamma); break; }
        plow_hsa_download(h, dev[r], got, b->reduced[r], hbytes);
        if (memcmp(ref_red, got, hbytes)) { fail = 1; fprintf(stderr, "reduced scratch mismatch rank%u\n", r); break; }
        if (contract) {
            plow_hsa_download(h, dev[r], got, b->prefix[r], hbytes);
            if (memcmp(ref_pre, got, hbytes)) { fail = 1; fprintf(stderr, "prefix mismatch rank%u\n", r); break; }
        }
        plow_hsa_download(h, dev[r], got, b->blockres[r], bbytes);
        if (memcmp(ref_blk, got, bbytes)) { fail = 1; fprintf(stderr, "ring mismatch rank%u\n", r); break; }
    }
    free(ref_mix); free(ref_pre); free(ref_red); free(ref_blk); free(got);
    return fail;
}

int main(int argc, char** argv) {
    if (argc != 9) { fprintf(stderr, "pass exactly eight GPU ids\n"); return 2; }
    int dev[NR]; for (uint32_t r = 0; r < NR; ++r) dev[r] = atoi(argv[r + 1]);
    const char* path = getenv("D6_ELF") ? getenv("D6_ELF") : "d6.elf";
    const uint32_t iters = getenv("D6_ITERS") ? (uint32_t)atoi(getenv("D6_ITERS")) : 200;
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 2; }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const double tick_ns = freq ? 1.0e9 / (double)freq : 1.0;
    const uint64_t deadline = freq ? freq : 1000000000ull;
    size_t elf_len = 0; void* elf = slurp(path, &elf_len);
    plow_hsa_kernel kc[NR], kf[NR], kb[NR], kff[NR], km[NR];
    for (uint32_t r = 0; r < NR; ++r) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "d6_control", &kc[r]) ||
            plow_hsa_get_kernel(h, dev[r], "d6_fused", &kf[r]) ||
            plow_hsa_get_kernel(h, dev[r], "d6_fill_bf16", &kb[r]) ||
            plow_hsa_get_kernel(h, dev[r], "d6_fill_f32", &kff[r]) ||
            plow_hsa_get_kernel(h, dev[r], "d6_xcd_map", &km[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        char arch[64] = {0}; uint32_t cus = 0, lds = 0;
        if (plow_hsa_device_info(h, dev[r], arch, &cus, &lds) || strcmp(arch, "gfx950") ||
            cus != 256u || lds < 147460u || kc[r].private_segment_size != 0u ||
            kf[r].private_segment_size != 0u || kc[r].group_segment_size != 147456u ||
            kf[r].group_segment_size != 147460u || km[r].private_segment_size != 0u ||
            km[r].group_segment_size != 147460u) {
            fprintf(stderr, "admission rank%u arch=%s cus=%u lds=%u ctl(group=%u private=%u) fused(group=%u private=%u) map(group=%u private=%u)\n",
                    r, arch, cus, lds, kc[r].group_segment_size,
                    kc[r].private_segment_size, kf[r].group_segment_size,
                    kf[r].private_segment_size, km[r].group_segment_size,
                    km[r].private_segment_size);
            return 2;
        }
    }

    Buffers b = {0};
    for (uint32_t r = 0; r < NR; ++r) {
        b.partial[r] = plow_hsa_alloc_peer(h, dev[r], HIDDEN * 2u);
        b.gate[r] = plow_hsa_alloc_peer(h, dev[r], 128u);
        b.peer_partial[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        b.peer_gate[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        b.reduced[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.prefix[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.mixed[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.blockres[r] = plow_hsa_alloc(h, dev[r], NBCAP * HIDDEN * 2u);
        b.score[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 4u);
        b.gamma[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.other[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.pre[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.push[r] = plow_hsa_alloc(h, dev[r], HIDDEN * 2u);
        b.sync[r] = plow_hsa_alloc(h, dev[r], sizeof(D6Sync));
        b.argdev[r] = plow_hsa_alloc(h, dev[r], sizeof(D6Args));
        b.cycles[r] = plow_hsa_alloc(h, dev[r], 8u);
        b.status[r] = plow_hsa_alloc(h, dev[r], 4u);
        if (!b.partial[r] || !b.gate[r] || !b.peer_partial[r] || !b.peer_gate[r] ||
            !b.reduced[r] || !b.prefix[r] || !b.mixed[r] || !b.blockres[r] ||
            !b.score[r] || !b.gamma[r] || !b.other[r] || !b.pre[r] || !b.push[r] ||
            !b.sync[r] || !b.argdev[r] || !b.cycles[r] || !b.status[r]) return 2;
    }
    for (uint32_t r = 0; r < NR; ++r) {
        plow_hsa_upload(h, dev[r], b.peer_partial[r], b.partial, NR * sizeof(void*));
        plow_hsa_upload(h, dev[r], b.peer_gate[r], b.gate, NR * sizeof(void*));
        void* mapdev = plow_hsa_alloc(h, dev[r], GRID * 4u); uint32_t map[GRID] = {0};
        struct { void* p; } ma = {mapdev};
        plow_hsa_launch(h, dev[r], &km[r], GRID * 512, 1, 1, 512, 1, 1, 0, &ma, sizeof ma);
        plow_hsa_wait(h, dev[r]); plow_hsa_download(h, dev[r], map, mapdev, sizeof map);
        uint32_t hist[NR] = {0};
        for (uint32_t i = 0; i < GRID; ++i) {
            if (map[i] >= NR) { fprintf(stderr, "XCD map rank%u block%u=%u\n", r, i, map[i]); return 1; }
            ++hist[map[i]];
        }
        for (uint32_t d = 0; d < NR; ++d) {
            const uint32_t want = d < 6u ? 2u : 1u;
            if (hist[d] != want) { fprintf(stderr, "XCD histogram rank%u domain%u=%u want%u\n", r, d, hist[d], want); return 1; }
        }
    }
    int fail = 0;
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t nb = 0; nb <= 8; ++nb)
            for (uint32_t p = 0; p <= 1; ++p)
                for (uint32_t g = 0; g <= 1; ++g)
                    fail |= compare_case(h, dev, kc, kf, kb, kff, &b, c, nb, p, g, deadline, tick_ns);
    if (fail) { fprintf(stderr, "D6 exact sweep failed\n"); return 1; }

    static const uint8_t site_count[3][9] = {
        {0, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 11, 12, 12, 12, 12, 12, 12, 9},
        {0, 12, 11, 11, 11, 11, 11, 11, 8},
    };
    double ctl = 0.0, fus = 0.0; uint32_t sites = 0;
    for (uint32_t c = 0; c < 3; ++c) for (uint32_t nb = 1; nb <= 8; ++nb) {
        const uint32_t count = site_count[c][nb];
        fill_inputs(h, dev, kb, kff, &b);
        const double a = run(h, dev, kc, 0, &b, c, nb, 0, 1, iters, deadline, tick_ns);
        fill_inputs(h, dev, kb, kff, &b);
        const double f = run(h, dev, kf, 1, &b, c, nb, 0, 1, iters, deadline, tick_ns);
        printf("contract=%u nb=%u sites=%u control=%.3f us fused=%.3f us delta=%+.3f us\n",
               c, nb, count, a, f, f - a);
        ctl += a * count; fus += f * count; sites += count;
    }
    printf("weighted sites=%u control=%.3f ms fused=%.3f ms delta=%+.3f ms exact=yes\n",
           sites, ctl / 1000.0, fus / 1000.0, (fus - ctl) / 1000.0);
    plow_hsa_shutdown(h);
    return 0;
}
