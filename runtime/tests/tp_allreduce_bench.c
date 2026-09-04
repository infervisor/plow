/* tp_allreduce_bench.c — N-RANK, BATCH-SWEPT validation of the one-shot all-reduce.
 *
 * Ranks each publish a known partial into peer VRAM; every rank runs op_collective.h's
 * d_xreduce_oneshot (publish-signal + SYSTEM-scope xctr gate + f32-accumulate reduce) and must
 * land the bit-exact element-wise sum. Per-collective latency is timed on rank 0's
 * s_memrealtime. The kernels call op_collective.h directly, so this validates THE op code and
 * not a copy — the rule every op_*.h golden test follows.
 *
 * WHY IT SWEEPS. The question this instrument exists to answer is whether plow needs the
 * ONE-STAGE / TWO-STAGE gating that launch-based engines add — ATOM ships exactly that
 * (one-stage through batch 32 at TP4) because a launched collective's fixed overhead dominates
 * at small sizes and its bandwidth term dominates at large ones, so no single algorithm wins
 * across the range. plow's collective is INLINE in the persistent kernel, so it has no launch
 * term to amortise, and the claim is that one algorithm covers the whole batch range. That is
 * a claim about a curve, and it was previously backed by ONE point (2 ranks, N=3840). This
 * sweeps ranks x batch so it becomes a measurement.
 *
 * WHAT TO READ IT AGAINST. The `us/collective` column is SYSTEM-scope (cross-GPU, peer VRAM,
 * xctr). The single-GPU AGENT-scope gate is the cheaper cousin and its cost is measured
 * separately: dropping the agent-scope acquire's buffer_inv is worth 0.30 ms of a 17.10 ms
 * Gemma-4 31B decode token (1.8%), and dropping the release RMW's buffer_wbl2 another 0.90 ms
 * (5.3%) — see the PLOW_GATE_NOINV / PLOW_GATE_RELAXSIG ceiling knobs in runtime/amd/interp.hip.
 * System scope is strictly more expensive than agent scope, so those figures are the floor
 * under anything measured here.
 *
 * MODES (TP_MODE). `oneshot` is the original hot loop: partials written once, the same peer
 * lines re-read every iteration. `cold` re-runs a producer each iteration (iteration-varying
 * plain stores + the interpreter's local agent-scope gate, double-buffered slots), which is
 * the decode condition. `tagged` / `tagged_cold` run d_xreduce_tagged_mega, the
 * signal-in-data prototype, on the same producer output. TP_ORACLE=order uses the strict
 * rank-order values [2^24, 1, -2^24, 0...] (exact rank-0..N-1 f32 sum is 0; any other
 * order is not). TP_PROBE=1 runs the fabric latency probe (rank 0 -> rank 1) first.
 *
 * Built by scripts/build_tp_allreduce.sh. Run on >= 2 idle gfx950:
 *   ./tp_allreduce_bench 1 2 3            # rank count = number of device ids given
 *   TP_HIDDEN=7168 TP_ITERS=4000 ./tp_allreduce_bench 1 2
 * env: TP_HIDDEN (default 7168), TP_NWG (default 64), TP_ITERS, TP_ELF, TP_MODE, TP_ORACLE,
 *      TP_PROBE. A single device id is accepted for a smoke run (every peer is self).
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TP_MAXR 8
#define PROBE_BYTES 65536u
#define TP_TICK_NS 10.0

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
/* The fill patterns the device kernels write (tp_val); the oracle must agree bit for bit. */
static bf16 partial_val(uint32_t r, uint32_t e, uint32_t iter, int order) {
    if (order) return f2bf(r == 0 ? 16777216.0f : r == 1 ? 1.0f : r == 2 ? -16777216.0f : 0.0f);
    return f2bf((float)(r + 1) * (1.0f + (float)((e + iter) & 7u) * 0.125f));
}

typedef struct { void* part; uint32_t n; uint32_t rank; uint32_t iter; uint32_t order; } arg_fill;
typedef struct {
    void* out; const void* peer_scratch; const void* peer_gate;
    uint32_t nranks; uint32_t rank; uint32_t n; uint32_t slot_bytes;
    uint32_t iters; uint64_t deadline; void* cycles; void* status;
} arg_ar;
typedef struct {
    void* out; const void* peer_scratch; const void* peer_gate;
    uint32_t nranks; uint32_t rank; uint32_t n; uint32_t slot_bytes;
    uint32_t iters; uint64_t deadline; void* cycles; void* status; void* lctr; uint32_t order;
} arg_cold;
typedef struct {
    void* out; const void* peer_scratch;
    uint32_t nranks; uint32_t rank; uint32_t n; uint32_t slot_bytes; uint32_t tslot_bytes;
    uint32_t iters; uint64_t deadline; void* cycles; void* status; void* lctr;
    uint32_t order; uint32_t cold; uint32_t tstride;
} arg_tag;
typedef struct {
    const void* peer_scratch; const void* peer_gate; uint32_t rank; uint32_t peer; uint32_t hops;
    void* res; uint32_t probe_off; void* dirty;
} arg_probe;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n); if (fread(p, 1, n, f) != (size_t)n) exit(2);
    fclose(f); *len = n; return p;
}

enum { MODE_ONESHOT, MODE_COLD, MODE_TAGGED, MODE_TAGGED_COLD };

int main(int argc, char** argv) {
    int dev[TP_MAXR]; uint32_t NR = 0;
    for (int i = 1; i < argc && NR < TP_MAXR; i++) dev[NR++] = atoi(argv[i]);
    if (NR == 0) { dev[0] = 0; dev[1] = 1; NR = 2; }
    const uint32_t HID = (uint32_t)(getenv("TP_HIDDEN") ? atoi(getenv("TP_HIDDEN")) : 7168);
    const uint32_t NWG = (uint32_t)(getenv("TP_NWG") ? atoi(getenv("TP_NWG")) : 64);
    const uint32_t ITERS = (uint32_t)(getenv("TP_ITERS") ? atoi(getenv("TP_ITERS")) : 4000);
    const char* elf_path = getenv("TP_ELF") ? getenv("TP_ELF") : "tp_allreduce_kernels.elf";
    const char* mode_s = getenv("TP_MODE") ? getenv("TP_MODE") : "oneshot";
    const int order = getenv("TP_ORACLE") && strcmp(getenv("TP_ORACLE"), "order") == 0;
    const int probe = getenv("TP_PROBE") && atoi(getenv("TP_PROBE")) != 0;
    int mode;
    if (!strcmp(mode_s, "oneshot")) mode = MODE_ONESHOT;
    else if (!strcmp(mode_s, "cold")) mode = MODE_COLD;
    else if (!strcmp(mode_s, "tagged")) mode = MODE_TAGGED;
    else if (!strcmp(mode_s, "tagged_cold")) mode = MODE_TAGGED_COLD;
    else { printf("TP_MODE must be oneshot|cold|tagged|tagged_cold\n"); return 2; }
    const int tagged = mode == MODE_TAGGED || mode == MODE_TAGGED_COLD;
    const int cold = mode == MODE_COLD || mode == MODE_TAGGED_COLD;
    const char* kname = mode == MODE_ONESHOT ? "tp_allreduce"
                        : mode == MODE_COLD  ? "tp_allreduce_cold"
                                             : "tp_allreduce_tagged";
    /* batch points: dense at the low end where a launched engine needs one-stage, then the
     * doublings where its bandwidth term would take over. */
    static const uint32_t BATCH[] = {1, 2, 4, 8, 16, 24, 32};
    const int NB = (int)(sizeof BATCH / sizeof BATCH[0]);
    int fails = 0;

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 2; }
    const int ndev = plow_hsa_device_count(h);
    printf("== plow one-shot all-reduce — %u ranks, batch sweep (gfx950 / XGMI) ==\n", NR);
    printf("GPUs discovered: %d   ranks:", ndev);
    for (uint32_t i = 0; i < NR; i++) printf(" GPU%d", dev[i]);
    printf("   hidden=%u  nblk=%u  iters=%u  mode=%s  oracle=%s\n\n", HID, NWG, ITERS, mode_s,
           order ? "order" : "benign");
    if (NR < 2) printf("WARNING: single rank — every peer is self, a smoke run only\n");
    if (NWG == 0) { printf("TP_NWG must be positive\n"); return 2; }
    for (uint32_t i = 0; i < NR; i++) {
        if (dev[i] >= ndev) { printf("device %d out of range (have %d)\n", dev[i], ndev); return 2; }
        for (uint32_t j = 0; j < i; j++)
            if (dev[i] == dev[j]) { printf("duplicate device %d\n", dev[i]); return 2; }
    }

    /* The kernels time with `s_memrealtime`, the 100 MHz REFCLK on gfx9 (10 ns per tick; the
     * interpreter trace calibrates it against host TTFT, scripts/k3_trace_report.py). Until
     * 2026-09-04 this harness scaled ticks by HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, which is the
     * runtime's 1 GHz profiling clock, so every number it printed was 10x too small. The host
     * wall clock of the timed launch is printed per row as the check. */
    uint64_t ts_freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ts_freq);
    const double tick_ns = TP_TICK_NS;
    printf("s_memrealtime tick = %.0f ns (HSA timestamp frequency %lu Hz is NOT this clock)\n",
           tick_ns, (unsigned long)ts_freq);

    size_t elf_len; void* elf = slurp(elf_path, &elf_len);
    plow_hsa_kernel k_fill[TP_MAXR], k_ar[TP_MAXR], k_probe[TP_MAXR];
    for (uint32_t i = 0; i < NR; i++) {
        if (plow_hsa_load_code_object(h, dev[i], elf, elf_len) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tp_fill_partial_v", &k_fill[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], kname, &k_ar[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tp_peer_probe", &k_probe[i]) != 0) {
            printf("load/get_kernel: %s\n", plow_hsa_last_error()); return 2;
        }
    }

    /* Peer region per rank: two plain slots (iteration parity), two tagged slots, then the
     * probe window. Every offset is a multiple of 256 B. */
    const uint32_t NMAX = HID * BATCH[NB - 1];
    const uint32_t SLOT = (NMAX * 2u + 255u) & ~255u;
    const uint32_t TSLOT = (((NMAX + 2u) / 3u) * 8u + 255u) & ~255u;
    const uint32_t PROBE_OFF = 2u * SLOT + 2u * TSLOT;
    const size_t REGION = (size_t)PROBE_OFF + PROBE_BYTES;
    void *scratch[TP_MAXR], *gate[TP_MAXR], *out[TP_MAXR], *scr_tbl[TP_MAXR], *gate_tbl[TP_MAXR];
    void *lctr[TP_MAXR], *dirty[TP_MAXR];
    uint32_t* stat[TP_MAXR];
    void* cyc = NULL;
    for (uint32_t i = 0; i < NR; i++) {
        scratch[i]  = plow_hsa_alloc_peer(h, dev[i], REGION);
        gate[i]     = plow_hsa_alloc_peer(h, dev[i], 256);
        out[i]      = plow_hsa_alloc(h, dev[i], (size_t)NMAX * 2);
        scr_tbl[i]  = plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        gate_tbl[i] = plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        stat[i]     = (uint32_t*)plow_hsa_alloc(h, dev[i], 4);
        lctr[i]     = plow_hsa_alloc(h, dev[i], 128);
        dirty[i]    = plow_hsa_alloc(h, dev[i], 131072);
        if (!scratch[i] || !gate[i] || !out[i] || !scr_tbl[i] || !gate_tbl[i] || !stat[i] ||
            !lctr[i] || !dirty[i]) {
            printf("alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    cyc = plow_hsa_alloc(h, dev[0], 8);
    void* scr_pv[TP_MAXR]; void* gate_pv[TP_MAXR];
    for (uint32_t i = 0; i < NR; i++) { scr_pv[i] = scratch[i]; gate_pv[i] = gate[i]; }
    for (uint32_t i = 0; i < NR; i++) {
        plow_hsa_upload(h, dev[i], scr_tbl[i], scr_pv, NR * sizeof(void*));
        plow_hsa_upload(h, dev[i], gate_tbl[i], gate_pv, NR * sizeof(void*));
    }
    const uint64_t deadline = (uint64_t)(ts_freq ? ts_freq : 1000000000);
    bf16* hb = malloc((size_t)NMAX * 2);
    unsigned char* zero_page = calloc(1, PROBE_BYTES);

    if (probe) {
        const uint32_t peer = NR > 1 ? 1u : 0u, hops = 2000;
        void* res = plow_hsa_alloc(h, dev[0], 64);
        uint32_t z[64] = {0};
        plow_hsa_upload(h, dev[peer], gate[peer], z, 256);
        plow_hsa_upload(h, dev[0], gate[0], z, 256);
        plow_hsa_upload(h, dev[peer], (char*)scratch[peer] + PROBE_OFF, zero_page, PROBE_BYTES);
        arg_probe ap = { scr_tbl[0], gate_tbl[0], 0, peer, hops, res, PROBE_OFF, dirty[0] };
        plow_hsa_launch(h, dev[0], &k_probe[0], 512, 1, 1, 512, 1, 1, 0, &ap, sizeof ap);
        plow_hsa_wait(h, dev[0]);
        uint64_t r[8] = {0}; plow_hsa_download(h, dev[0], r, res, 64);
        static const char* what[6] = {
            "remote sys-scope 8 B load (dependent chain)",
            "remote returning relaxed sys atomic (dependent chain)",
            "remote release fetch_add + s_waitcnt vmcnt(0), clean L2",
            "remote release fetch_add + s_waitcnt vmcnt(0), 64 KiB dirty L2 per hop",
            "local sys-scope 4 B poll load (dependent chain)",
            "local poll load + system acquire fence (buffer_inv sc0 sc1)"};
        printf("fabric probe: rank 0 -> rank %u, %u hops, ns per hop\n", peer, hops);
        for (int k = 0; k < 6; k++)
            printf("  %-72s %8.1f\n", what[k], (double)r[k] * tick_ns / hops);
        printf("\n");
    }

    printf("  batch      N     KB/rank   us/coll    GB/s eff   bit-exact   host us/coll\n");
    printf("  -----  -------  --------  --------  ----------  ---------   ------------\n");
    for (int b = 0; b < NB; b++) {
        const uint32_t N = HID * BATCH[b];
        /* Re-fill and re-zero every point: the gate word counts arrivals cumulatively, so a
         * fresh run per shape keeps gate_target = i*nranks honest. Tagged slots are zeroed so
         * a tag from the previous point cannot match (tags restart at 1). */
        uint32_t z[64] = {0};
        for (uint32_t i = 0; i < NR; i++) {
            plow_hsa_upload(h, dev[i], gate[i], z, 256);
            plow_hsa_upload(h, dev[i], stat[i], z, 4);
            plow_hsa_upload(h, dev[i], lctr[i], z, 128);
            for (uint32_t s = 0; tagged && s < 2u * TSLOT; s += PROBE_BYTES) {
                const size_t len = 2u * TSLOT - s < PROBE_BYTES ? 2u * TSLOT - s : PROBE_BYTES;
                plow_hsa_upload(h, dev[i], (char*)scratch[i] + 2u * SLOT + s, zero_page, len);
            }
            /* Hot arms read slot 0 filled here with the final iteration's values; cold arms
             * overwrite both slots and finish on iteration ITERS. */
            for (uint32_t s = 0; s < 2u; s++) {
                arg_fill af = { (char*)scratch[i] + s * SLOT, N, i, ITERS, (uint32_t)order };
                plow_hsa_launch(h, dev[i], &k_fill[i], 4096, 1, 1, 256, 1, 1, 0, &af, sizeof af);
            }
            plow_hsa_wait(h, dev[i]);
        }
        /* Launch the non-timed ranks first so they are already spinning, then rank 0. */
        arg_ar a[TP_MAXR]; arg_cold ac[TP_MAXR]; arg_tag at[TP_MAXR];
        struct timespec h0, h1;
        clock_gettime(CLOCK_MONOTONIC, &h0);
        for (int ii = (int)NR - 1; ii >= 0; ii--) {
            const uint32_t i = (uint32_t)ii;
            void* c = i ? NULL : cyc;
            const void* ka; size_t kz;
            if (mode == MODE_ONESHOT) {
                a[i] = (arg_ar){ out[i], scr_tbl[i], gate_tbl[i], NR, i, N, 0, ITERS, deadline, c, stat[i] };
                ka = &a[i]; kz = sizeof a[i];
            } else if (mode == MODE_COLD) {
                ac[i] = (arg_cold){ out[i], scr_tbl[i], gate_tbl[i], NR, i, N, SLOT, ITERS, deadline, c,
                                    stat[i], lctr[i], (uint32_t)order };
                ka = &ac[i]; kz = sizeof ac[i];
            } else {
                /* Tagged slots sit at 2*SLOT + parity*TSLOT. */
                at[i] = (arg_tag){ out[i], scr_tbl[i], NR, i, N, SLOT, 2u * SLOT, ITERS, deadline, c,
                                   stat[i], lctr[i], (uint32_t)order, (uint32_t)cold, TSLOT };
                ka = &at[i]; kz = sizeof at[i];
            }
            plow_hsa_launch(h, dev[i], &k_ar[i], NWG * 512, 1, 1, 512, 1, 1, 0, ka, kz);
        }
        for (uint32_t i = 0; i < NR; i++) plow_hsa_wait(h, dev[i]);
        clock_gettime(CLOCK_MONOTONIC, &h1);
        const double host_us = ((double)(h1.tv_sec - h0.tv_sec) * 1e9 +
                                (double)(h1.tv_nsec - h0.tv_nsec)) / 1e3;

        int timeout = 0;
        for (uint32_t i = 0; i < NR; i++) {
            uint32_t st = 0; plow_hsa_download(h, dev[i], &st, stat[i], 4);
            if (st) { printf("  FAIL: rank %u xctr gate did not propagate (0x%08x)\n", i, st); timeout = 1; }
        }
        if (timeout) { fails++; continue; }

        int exact = 1; size_t bad = 0;
        for (uint32_t i = 0; i < NR; i++) {
            plow_hsa_download(h, dev[i], hb, out[i], (size_t)N * 2);
            for (uint32_t e = 0; e < N; e++) {
                float sum = 0.0f;
                for (uint32_t r = 0; r < NR; r++) sum += bf2f(partial_val(r, e, ITERS, order));
                if (hb[e] != f2bf(sum)) {
                    if (exact)
                        printf("  FAIL: rank %u e=%u got %.4f want %.4f\n", i, e, bf2f(hb[e]), sum);
                    exact = 0; bad++;
                }
            }
        }
        if (!exact) { fails++; printf("  %zu wrong elements\n", bad); }
        uint64_t cycles = 0; plow_hsa_download(h, dev[0], &cycles, cyc, 8);
        const double per_us = (double)cycles * tick_ns / ITERS / 1e3;
        /* Effective bytes moved per collective: each rank reads NR-1 peer slots. */
        const double gbs = ((double)N * 2.0 * (double)(NR - 1)) / (per_us * 1e-6) / 1e9;
        printf("  %5u  %7u  %8.1f  %8.3f  %10.1f  %-9s   %8.3f\n", BATCH[b], N, N * 2 / 1024.0,
               per_us, gbs, exact ? "yes" : "NO", host_us / ITERS);
    }

    printf("\n%s\n", fails ? "== FAILURES ==" : "== ALL-REDUCE VALIDATED (bit-exact across the sweep) ==");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
