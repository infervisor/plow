/* tp_tilegate_bench.c — does a TILE-TRIGGERED collective actually win?
 *
 * The proposal under test: split the layer all-reduce into ~512-element tiles, give each tile
 * its own cross-GPU gate, and start a tile's peer reads as soon as the GEMV slices producing
 * it are done. At decode, hidden = 7168, so that replaces ONE system-scope rendezvous with
 * ceil(7168/512) = 14 of them, on every one of the ~186 collectives in a token.
 *
 * It is obviously POSSIBLE. This measures whether it WINS, which is a different question and
 * the only one worth answering. Three tables:
 *
 *   [A] GEMV SKEW — per-workgroup completion spread of the real decode row-GEMV
 *       (op_gemm.h's `gemv_rows`, GV_BLOCKED=1). This is the entire prize: a tile gate can
 *       hide producer skew and nothing else. If the workgroups finish together there is
 *       nothing to hide and the proposal is dead before the cost side is even counted.
 *
 *   [B] RENDEZVOUS COST — coarse (1 gate) vs tiled (14 gates) at ZERO injected skew. Both
 *       arms move the same bytes and do the same arithmetic; the difference is the price of
 *       the extra 13 rendezvous, per collective.
 *
 *   [C] CROSSOVER — the same two arms swept over an injected skew ramp, which says how much
 *       producer skew the tiled form needs before it repays that price. Read [A] against [C].
 *
 * Every point is checked bit-exact against the CPU oracle, and the two arms must agree with
 * each other: a tile gate that releases early is a silent wrong answer across ranks, so the
 * value check is the correctness proof, not a nicety.
 *
 * Built by scripts/build_tp_tilegate.sh. Run on >= 2 idle gfx950:
 *   ./tp_tilegate_bench 0 1 2 3
 * env: TG_HIDDEN (default 7168, Kimi-K3), TG_ITERS, TG_GEMV_BLK (default 256), TG_ELF.
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TG_MAXR 8
#define TG_CTR_STRIDE 32u /* must match PLOW_CTR_STRIDE in runtime/common/dev_isa.h */

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static bf16 tg_val(uint32_t r, uint32_t e) {
    return f2bf((float)(r + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}

typedef struct { void* p; uint32_t n; uint32_t seed; } arg_fill;
typedef struct {
    void* C; const void* x; const void* W; uint32_t N; uint32_t K; void* ts; uint32_t iters;
} arg_gemv;
typedef struct {
    void* out; const void* peer_scratch; const void* peer_gate;
    uint32_t nranks; uint32_t rank; uint32_t n; uint32_t slot_bytes;
    uint32_t mode; uint32_t skew_ticks; uint32_t iters; uint64_t deadline;
    void* cycles; void* status; void* lctr; void* ts; uint32_t ntiles; uint64_t xctr_off;
} arg_ar;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n); if (fread(p, 1, n, f) != (size_t)n) exit(2);
    fclose(f); *len = n; return p;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char** argv) {
    int dev[TG_MAXR]; uint32_t NR = 0;
    for (int i = 1; i < argc && NR < TG_MAXR; i++) dev[NR++] = atoi(argv[i]);
    if (NR == 0) { dev[0] = 0; dev[1] = 1; NR = 2; }
    const uint32_t HID = (uint32_t)(getenv("TG_HIDDEN") ? atoi(getenv("TG_HIDDEN")) : 7168);
    const uint32_t ITERS = (uint32_t)(getenv("TG_ITERS") ? atoi(getenv("TG_ITERS")) : 2000);
    const uint32_t GBLK = (uint32_t)(getenv("TG_GEMV_BLK") ? atoi(getenv("TG_GEMV_BLK")) : 256);
    const char* elf_path = getenv("TG_ELF") ? getenv("TG_ELF") : "tp_tilegate_kernels.elf";
    /* The collective's own width: emit_xreduce sizes it to ceil(xr_elems/512), and the tiled
     * arm gives each of those workgroups its own gate. That is the proposal's tile count. */
    const uint32_t NTILE = (HID + 511) / 512;
    int fails = 0;

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 2; }
    const int ndev = plow_hsa_device_count(h);
    printf("== tile-triggered collective: does it win? — %u ranks (gfx950 / XGMI) ==\n", NR);
    printf("GPUs discovered: %d   ranks:", ndev);
    for (uint32_t i = 0; i < NR; i++) printf(" GPU%d", dev[i]);
    printf("   hidden=%u  tiles=%u  iters=%u\n", HID, NTILE, ITERS);
    if (NR < 2) { printf("need >= 2 ranks\n"); return 2; }
    for (uint32_t i = 0; i < NR; i++)
        if (dev[i] >= ndev) { printf("device %d out of range\n", dev[i]); return 2; }

    uint64_t ts_freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ts_freq);
    const double tick_ns = ts_freq ? 1e9 / (double)ts_freq : 1.0;

    size_t elf_len; void* elf = slurp(elf_path, &elf_len);
    plow_hsa_kernel k_fill[TG_MAXR], k_fillw[TG_MAXR], k_gemv[TG_MAXR], k_ar[TG_MAXR];
    for (uint32_t i = 0; i < NR; i++) {
        if (plow_hsa_load_code_object(h, dev[i], elf, elf_len) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tg_fill", &k_fill[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tg_fill_w", &k_fillw[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tg_gemv_ts", &k_gemv[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tg_ar", &k_ar[i]) != 0) {
            printf("load/get_kernel: %s\n", plow_hsa_last_error()); return 2;
        }
    }

    /* ---------------- [A] real decode row-GEMV per-workgroup skew ---------------- */
    /* o_proj/down at TP: the reduce dim is sharded, the output is the full hidden. */
    const uint32_t GN = HID, GK = HID / NR;
    printf("\n[A] REAL DECODE ROW-GEMV WORKGROUP SKEW  (op_gemm.h gemv_rows, GV_BLOCKED=1)\n");
    printf("    N=%u K=%u (=hidden/tp, the row-parallel shard)  nblk=%u  GPU%d\n", GN, GK, GBLK,
           dev[0]);
    {
        const uint32_t gi = 200;
        void* gC = plow_hsa_alloc(h, dev[0], (size_t)GN * 2);
        void* gx = plow_hsa_alloc(h, dev[0], (size_t)GK * 2);
        void* gW = plow_hsa_alloc(h, dev[0], (size_t)GN * GK * 2);
        void* gts = plow_hsa_alloc(h, dev[0], (size_t)GBLK * 2 * 8);
        if (!gC || !gx || !gW || !gts) { printf("    alloc: %s\n", plow_hsa_last_error()); return 2; }
        arg_fill fx = { gx, GK, 1 }, fw = { gW, GN * GK, 7 };
        plow_hsa_launch(h, dev[0], &k_fillw[0], 4096, 1, 1, 256, 1, 1, 0, &fx, sizeof fx);
        plow_hsa_launch(h, dev[0], &k_fillw[0], 65536, 1, 1, 256, 1, 1, 0, &fw, sizeof fw);
        plow_hsa_wait(h, dev[0]);
        arg_gemv ag = { gC, gx, gW, GN, GK, gts, gi };
        plow_hsa_launch(h, dev[0], &k_gemv[0], GBLK * 512, 1, 1, 512, 1, 1, 0, &ag, sizeof ag);
        plow_hsa_wait(h, dev[0]);
        uint64_t* ts = malloc((size_t)GBLK * 2 * 8);
        plow_hsa_download(h, dev[0], ts, gts, (size_t)GBLK * 2 * 8);
        uint64_t t0min = ts[0], tend[4096];
        for (uint32_t b = 0; b < GBLK; b++) { if (ts[2*b] < t0min) t0min = ts[2*b]; }
        for (uint32_t b = 0; b < GBLK; b++) tend[b] = ts[2*b+1] - t0min;
        qsort(tend, GBLK, 8, cmp_u64);
        const double one = 1.0 / gi; /* per single GEMV */
        const double us_lo  = (double)tend[0] * tick_ns * one / 1e3;
        const double us_p50 = (double)tend[GBLK/2] * tick_ns * one / 1e3;
        const double us_p90 = (double)tend[(GBLK*9)/10] * tick_ns * one / 1e3;
        const double us_hi  = (double)tend[GBLK-1] * tick_ns * one / 1e3;
        printf("    per-GEMV workgroup FINISH times (us, relative to first workgroup start):\n");
        printf("      earliest %.3f   p50 %.3f   p90 %.3f   LAST %.3f\n", us_lo, us_p50, us_p90, us_hi);
        printf("      => GEMV wall time  %.3f us ;  SKEW (last-earliest) %.3f us  (%.1f%% of it)\n",
               us_hi, us_hi - us_lo, 100.0 * (us_hi - us_lo) / us_hi);
        printf("    The skew is the ENTIRE prize: it is the only thing a tile gate can hide.\n");
        free(ts);
    }

    /* ---------------- [B]/[C] coarse vs tiled rendezvous ---------------- */
    /* Shapes. `nblk` is how wide the collective runs; `ntiles` is how many cross-GPU gates it
     * is cut into. ntiles == 1 in the tiled arm degenerates to the coarse arm minus the local
     * barrier, so it is also the control that prices that barrier on its own.
     *
     *   DECODE  n = hidden, nblk = ceil(n/512) = 14 — emit_xreduce ALREADY narrows the
     *           collective to exactly this, so the proposal's 14 tiles means one gate per
     *           workgroup and the producer->tile map is the identity. Best possible case.
     *   PREFILL n = T*hidden. The collective runs the full machine for bandwidth, so a tile is
     *           a row-block covering many workgroups and its gate takes nranks*gsz arrivals.
     */
    struct shape { const char* tag; uint32_t n; uint32_t nblk; uint32_t ntiles; };
    const uint32_t PT = (uint32_t)(getenv("TG_PREFILL_T") ? atoi(getenv("TG_PREFILL_T")) : 512);
    struct shape SH[] = {
        { "decode  T=1",   HID,          NTILE, NTILE },
        { "prefill T=%u",  PT * HID,     256,   4     },
        { "prefill T=%u",  PT * HID,     256,   16    },
        { "prefill T=%u",  PT * HID,     256,   64    },
        { "prefill T=%u",  PT * HID,     256,   256   },
    };
    const int NSH = (int)(sizeof SH / sizeof SH[0]);

    uint32_t NMAX = 0, BMAX = 0, TMAX = 0;
    for (int s = 0; s < NSH; s++) {
        if (SH[s].n > NMAX) NMAX = SH[s].n;
        if (SH[s].nblk > BMAX) BMAX = SH[s].nblk;
        if (SH[s].ntiles > TMAX) TMAX = SH[s].ntiles;
    }

    void *scratch[TG_MAXR], *gate[TG_MAXR], *out[TG_MAXR], *scr_tbl[TG_MAXR], *gate_tbl[TG_MAXR];
    void *lctr[TG_MAXR], *arts[TG_MAXR]; uint32_t* stat[TG_MAXR];
    /* The gate counters live INSIDE the peer scratch, at a fixed byte offset, because that is
     * how the real packet lays out its `xctr` region and it is what d_xreduce_twoshot_mega
     * expects (it derives every peer's counter base as peer_scratch[r] + xctr_byte_off).
     * The two-shot's thresholds are absolute, so its timing loop burns a fresh pair of counter
     * ids per iteration: 2*ITERS + 2 of them. */
    const uint32_t NCTR = (TMAX + 1) > (3 * ITERS + 3) ? (TMAX + 1) : (3 * ITERS + 3);
    const size_t gate_bytes = (size_t)NCTR * TG_CTR_STRIDE * 4;
    const size_t part_bytes = (((size_t)NMAX * 2) + 4095) & ~(size_t)4095;
    for (uint32_t i = 0; i < NR; i++) {
        scratch[i]  = plow_hsa_alloc_peer(h, dev[i], part_bytes + gate_bytes);
        gate[i]     = scratch[i] ? (void*)((char*)scratch[i] + part_bytes) : NULL;
        out[i]      = plow_hsa_alloc(h, dev[i], (size_t)NMAX * 2);
        scr_tbl[i]  = plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        gate_tbl[i] = plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        lctr[i]     = plow_hsa_alloc(h, dev[i], 128);
        arts[i]     = plow_hsa_alloc(h, dev[i], (size_t)BMAX * 2 * 8);
        stat[i]     = (uint32_t*)plow_hsa_alloc(h, dev[i], 4);
        if (!scratch[i] || !gate[i] || !out[i] || !scr_tbl[i] || !gate_tbl[i] || !lctr[i] ||
            !arts[i] || !stat[i]) {
            printf("alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    void* cyc = plow_hsa_alloc(h, dev[0], 8);
    void* scr_pv[TG_MAXR]; void* gate_pv[TG_MAXR];
    for (uint32_t i = 0; i < NR; i++) { scr_pv[i] = scratch[i]; gate_pv[i] = gate[i]; }
    for (uint32_t i = 0; i < NR; i++) {
        plow_hsa_upload(h, dev[i], scr_tbl[i], scr_pv, NR * sizeof(void*));
        plow_hsa_upload(h, dev[i], gate_tbl[i], gate_pv, NR * sizeof(void*));
    }
    const uint64_t deadline = (uint64_t)(ts_freq ? ts_freq : 1000000000);
    bf16* hb = malloc((size_t)NMAX * 2);
    void* zbuf = calloc(1, gate_bytes);
    uint64_t* wts = malloc((size_t)BMAX * 2 * 8);

    /* Injected skew ramp, ns. 0 is the pure rendezvous-cost point [B]; the rest is [C], and
     * the shape of the delta across this sweep is the whole result: if tiling hides skew the
     * delta must GROW with it. */
    static const double SKEW_NS[] = { 0, 250, 500, 1000, 2000, 4000, 8000 };
    const int NS = (int)(sizeof SKEW_NS / sizeof SKEW_NS[0]);

    printf("\n[B]/[C] COARSE (1 cross-GPU rendezvous) vs TILED (C rendezvous), %u ranks\n", NR);
    printf("    Both arms: same workgroup count, same producer work, same reduce arithmetic,\n"
           "    same summation order (r=0..N-1, f32 acc). ONLY the gate structure differs.\n");
    printf("    TIMED AS MAKESPAN (last workgroup out - first in), because the collective's\n"
           "    consumer (RMSNorm / AttnRes) reads the whole vector and is a barrier over every\n"
           "    tile. Timing workgroup 0 alone reports a flat number that ignores the straggler\n"
           "    entirely; the `wg0` column is kept to show how large that error is.\n");

    for (int sh = 0; sh < NSH; sh++) {
        const uint32_t n = SH[sh].n, nblk = SH[sh].nblk, ntl = SH[sh].ntiles;
        char tag[64]; snprintf(tag, sizeof tag, SH[sh].tag, PT);
        printf("\n  --- %s : n=%u (%.2f MB/partial)  nblk=%u  tiles=%u  (%u wgs/tile) ---\n",
               tag, n, n * 2 / 1048576.0, nblk, ntl, nblk / ntl);
        printf("    injected   1shot     1gate      1shot     SHIPPING   tile-trigger\n");
        printf("             coarse    /nobar      tiled     two-shot    is worth\n");
        printf("    skew (us) makespan  makespan   makespan   makespan   (vs 1gate)\n");
        printf("    --------- --------  --------   --------   --------   ----------\n");
        double d0 = 0;
        for (int s = 0; s < NS; s++) {
            const uint32_t skew_ticks = (uint32_t)(SKEW_NS[s] / tick_ns);
            double got[4] = { 0, 0, 0, 0 }, wg0[4] = { 0, 0, 0, 0 };
            for (uint32_t mode = 0; mode < 4; mode++) {
                uint32_t z4 = 0;
                for (uint32_t i = 0; i < NR; i++) {
                    plow_hsa_upload(h, dev[i], gate[i], zbuf, gate_bytes);
                    plow_hsa_upload(h, dev[i], lctr[i], zbuf, 128);
                    plow_hsa_upload(h, dev[i], stat[i], &z4, 4);
                    arg_fill af = { scratch[i], n, i };
                    plow_hsa_launch(h, dev[i], &k_fill[i], 8192, 1, 1, 256, 1, 1, 0, &af, sizeof af);
                    plow_hsa_wait(h, dev[i]);
                }
                arg_ar a[TG_MAXR];
                for (uint32_t i = NR - 1; i >= 1; i--) {
                    a[i] = (arg_ar){ out[i], scr_tbl[i], gate_tbl[i], NR, i, n, 0, mode,
                                     skew_ticks, ITERS, deadline, NULL, stat[i], lctr[i],
                                     arts[i], ntl, part_bytes };
                    plow_hsa_launch(h, dev[i], &k_ar[i], nblk * 512, 1, 1, 512, 1, 1, 0,
                                    &a[i], sizeof a[i]);
                }
                a[0] = (arg_ar){ out[0], scr_tbl[0], gate_tbl[0], NR, 0, n, 0, mode,
                                 skew_ticks, ITERS, deadline, cyc, stat[0], lctr[0],
                                 arts[0], ntl, part_bytes };
                plow_hsa_launch(h, dev[0], &k_ar[0], nblk * 512, 1, 1, 512, 1, 1, 0,
                                &a[0], sizeof a[0]);
                for (uint32_t i = 0; i < NR; i++) plow_hsa_wait(h, dev[i]);

                int bad = 0;
                for (uint32_t i = 0; i < NR; i++) {
                    uint32_t st = 0; plow_hsa_download(h, dev[i], &st, stat[i], 4);
                    if (st) { printf("    FAIL: rank %u gate did not propagate (0x%08x)\n", i, st); bad = 1; }
                }
                /* Bit-exact against the CPU oracle, on EVERY rank and EVERY arm. This is what
                 * proves the tiled gates do not release early — a gate that fires before its
                 * peers published would land a partial sum, and a partial sum is visible here. */
                for (uint32_t i = 0; i < NR && !bad; i++) {
                    plow_hsa_download(h, dev[i], hb, out[i], (size_t)n * 2);
                    for (uint32_t e = 0; e < n; e++) {
                        float sum = 0.0f;
                        for (uint32_t r = 0; r < NR; r++) sum += bf2f(tg_val(r, e));
                        if (hb[e] != f2bf(sum)) {
                            printf("    FAIL: mode %u rank %u e=%u got %.4f want %.4f\n",
                                   mode, i, e, bf2f(hb[e]), sum);
                            bad = 1; break;
                        }
                    }
                }
                if (bad) fails++;
                plow_hsa_download(h, dev[0], wts, arts[0], (size_t)nblk * 2 * 8);
                uint64_t lo = wts[0], hi = wts[1];
                for (uint32_t b = 0; b < nblk; b++) {
                    if (wts[2*b] < lo) lo = wts[2*b];
                    if (wts[2*b+1] > hi) hi = wts[2*b+1];
                }
                got[mode] = (double)(hi - lo) * tick_ns / ITERS / 1e3;
                uint64_t cycles = 0; plow_hsa_download(h, dev[0], &cycles, cyc, 8);
                wg0[mode] = (double)cycles * tick_ns / ITERS / 1e3;
            }
            const double d = got[1] - got[0];
            if (s == 0) d0 = d;
            /* "barrier is worth"    = coarse - (1 gate, no barrier)
             * "tile-trigger is worth" = (1 gate, no barrier) - tiled.  Positive = a saving. */
            printf("    %9.2f %8.3f  %8.3f   %8.3f   %8.3f   %+10.3f\n", SKEW_NS[s] / 1e3,
                   got[0], got[2], got[1], got[3], got[2] - got[1]);
        }
        printf("    tiled-coarse at zero skew: %+.3f us.  The two right-hand columns split it into\n", d0);
        printf("    the part that is just deleting the local N-way barrier and the part that is\n");
        printf("    genuinely tile-triggering. If tile-triggering hid skew, ITS column would GROW\n");
        printf("    down the sweep. Read whether it does.\n");
    }

    printf("\n%s\n", fails ? "== FAILURES ==" : "== EVERY ARM BIT-EXACT AT EVERY POINT ==");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
