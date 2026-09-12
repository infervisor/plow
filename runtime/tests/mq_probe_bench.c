/* mq_probe_bench.c — multi-queue AQL overlap probe (see mq_probe_kernels.hip's header).
 *
 * Runs the REAL production collective (op_collective.h's d_xreduce_twoshot_mega, the exact
 * body XReduceTwoShot dispatches) across all NR ranks, at TP_ROWS x TP_HIDDEN, capped to
 * MQ_XR_NWG workgroups (PLOW_XR_CUS's lever; default 32, docs/flags-reference.md), and a
 * SYNTHETIC independent-GEMM proxy on rank 0 alone (no cross-GPU dependency), MQ_GEMM_NWG
 * workgroups (default 272 = 304 - 32, CU-disjoint from the collective by construction) doing
 * MQ_GEMM_ITERS dependent FMA steps.
 *
 * MQ_ARM selects what rank 0 does with the two dispatches:
 *   solo_xr    only the collective (every rank) — baseline collective duration.
 *   solo_gemm  only the GEMM proxy on rank 0, nothing on the other ranks — baseline GEMM
 *              duration.
 *   1q         TODAY: collective then GEMM proxy, same queue, both via plow_hsa_launch, so
 *              both carry the barrier bit plow's runtime always sets (device/hsa.rs,
 *              hsa_backend.c). Serialisation is expected; this arm's job is to confirm it.
 *   2q         PROPOSED: collective on rank 0's normal queue; GEMM proxy on a SECOND,
 *              independent queue opened on the same agent (plow_hsa_agent_raw +
 *              hsa_queue_create), dispatched with the barrier bit CLEARED and no ordering
 *              relationship to the collective at all. Both are issued before either is
 *              waited on.
 * All arms run every other rank's collective dispatch identically (needed for the real
 * rendezvous to complete); only rank 0 additionally carries the GEMM proxy in 1q/2q.
 *
 * Reports, on rank 0's own s_memrealtime clock (100 MHz REFCLK, one clock domain, so the two
 * kernels' windows are directly comparable): the collective's [t0,t1], the GEMM proxy's
 * [min start, max end] over its workgroups, the overlap between them, and the wall time
 * spanning both — plus each kernel's own duration, to catch CU contention (the effect
 * PLOW_AMD_DECODE_GEMM_OVERLAP measured: barrier-free WITHIN one queue grew native GEMM
 * intervals 40-50 us).
 *
 * env: MQ_ROWS (8192) MQ_HIDDEN (6144) MQ_XR_NWG (32) MQ_GEMM_NWG (272) MQ_GEMM_ITERS (20000)
 *      MQ_REPS (7) MQ_ARM (required) MQ_ELF (mq_probe_kernels.elf)
 * usage: mq_probe_bench gpu0 gpu1 ... gpu7
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define REGION_BYTES (127u * 1024u * 1024u)
#define XCTR_OFF 132120576u
#define MQ_KARG_SLOT 512u
#define MQ_Q2_SIZE 64u

typedef uint16_t bf16;

static uint32_t envu(const char* k, uint32_t d) { const char* v = getenv(k); return v ? (uint32_t)strtoul(v, 0, 10) : d; }
static const char* envs(const char* k, const char* d) { const char* v = getenv(k); return v ? v : d; }
static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static double med(double* a, uint32_t n) { qsort(a, n, sizeof(double), cmpd); return a[n / 2]; }

/* --- a caller-owned second AQL queue on one agent, mirroring hsa_backend.c's packet-write
 * body but with an explicit barrier bit and its own kernarg ring/signal, independent of the
 * backend's per-device queue. This is the exact mechanism the brief proposes: K queues, real
 * dependency edges only. */
typedef struct {
    hsa_agent_t agent;
    hsa_queue_t* q;
    uint8_t* karg_ring;
    hsa_signal_t sig; /* counting: +1 per dispatch, -1 on completion, like plow_hsa's `done` */
} q2_t;

static int q2_init(q2_t* c, hsa_agent_t agent, hsa_amd_memory_pool_t kpool) {
    c->agent = agent;
    if (hsa_queue_create(agent, MQ_Q2_SIZE, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &c->q) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_amd_memory_pool_allocate(kpool, (size_t)MQ_Q2_SIZE * MQ_KARG_SLOT, 0,
                                     (void**)&c->karg_ring) != HSA_STATUS_SUCCESS)
        return -1;
    hsa_amd_agents_allow_access(1, &agent, NULL, c->karg_ring);
    return hsa_signal_create(0, 0, NULL, &c->sig) == HSA_STATUS_SUCCESS ? 0 : -1;
}

/* 1D-grid dispatch onto `c`'s queue with an EXPLICIT barrier bit — the packet field this
 * probe exists to flip. Same packet contract as plow_hsa_launch (hsa_backend.c) otherwise:
 * agent-scope acquire/release fences on every packet either way, matching production. */
static int q2_launch(q2_t* c, const plow_hsa_kernel* k, uint32_t grid_x, uint16_t wg_x,
                     const void* args, size_t args_size, int barrier) {
    hsa_queue_t* q = c->q;
    uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
    while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}
    const uint32_t slot = (uint32_t)(idx & (q->size - 1));
    uint8_t* karg = c->karg_ring + (size_t)slot * MQ_KARG_SLOT;
    memcpy(karg, args, args_size);
    memset(karg + args_size, 0, k->kernarg_size - args_size);
    const size_t hoff = (args_size + 7u) & ~(size_t)7u;
    if (k->kernarg_size > hoff) {
        uint8_t* hid = karg + hoff;
        const size_t avail = k->kernarg_size - hoff;
#define PUT32(off, val) if (avail >= (off) + 4) *(uint32_t*)(hid + (off)) = (uint32_t)(val)
#define PUT16(off, val) if (avail >= (off) + 2) *(uint16_t*)(hid + (off)) = (uint16_t)(val)
        PUT32(0, (grid_x + wg_x - 1) / wg_x);
        PUT16(12, wg_x);
        PUT16(18, grid_x % wg_x);
        PUT16(64, 1);
#undef PUT32
#undef PUT16
    }
    hsa_kernel_dispatch_packet_t* p = (hsa_kernel_dispatch_packet_t*)q->base_address + slot;
    memset((uint8_t*)p + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x = wg_x; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
    p->grid_size_x = grid_x; p->grid_size_y = 1; p->grid_size_z = 1;
    p->kernel_object = k->kernel_object;
    p->kernarg_address = karg;
    p->group_segment_size = k->group_segment_size;
    p->private_segment_size = k->private_segment_size;
    p->completion_signal = c->sig;
    hsa_signal_add_screlease(c->sig, 1);
    uint16_t header = (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                    | ((barrier ? 1u : 0u) << HSA_PACKET_HEADER_BARRIER)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    uint16_t setup = (uint16_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
    __atomic_store_n((uint32_t*)p, ((uint32_t)setup << 16) | header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
    return 0;
}

static void q2_wait(q2_t* c) {
    while (hsa_signal_wait_scacquire(c->sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0) {}
}

typedef struct {
    void* out; const void* peers; uint32_t nranks, rank, n, slot_bytes;
    uint64_t xctr_byte_off; uint64_t deadline; void* ts; void* status;
} arg_xr;
typedef struct { void* part; uint32_t n; uint32_t rank; } arg_fill;
typedef struct { void* out; uint32_t iters; void* ts; } arg_gemm;

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* arm = envs("MQ_ARM", "");
    if (strcmp(arm, "solo_xr") && strcmp(arm, "solo_gemm") && strcmp(arm, "1q") && strcmp(arm, "2q")
        && strcmp(arm, "2qm")) {
        fprintf(stderr, "MQ_ARM must be solo_xr|solo_gemm|1q|2q|2qm\n"); return 2;
    }
    /* 2qm: hsa_amd_queue_cu_set_mask partitions the two queues' CUs by construction instead of
     * by grid-size bookkeeping — the hardware mechanism that turns "these two grids happen to
     * sum to n_cu" into a mask the packet processor enforces regardless of dispatch order or
     * grid size. Masks [0,32) to rank 0's default (collective) queue, [32,304) to q2 (GEMM). */
    const int masked = !strcmp(arm, "2qm");
    const uint32_t rows = envu("MQ_ROWS", 8192), hidden = envu("MQ_HIDDEN", 6144);
    const uint32_t xr_nwg = envu("MQ_XR_NWG", 32), gemm_nwg = envu("MQ_GEMM_NWG", 272);
    const uint32_t gemm_iters = envu("MQ_GEMM_ITERS", 20000), reps = envu("MQ_REPS", 7);
    const uint64_t n = (uint64_t)rows * hidden;
    if (n > UINT32_MAX || 2u * n > XCTR_OFF) { fprintf(stderr, "shape too big for REGION_BYTES\n"); return 2; }
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const uint64_t deadline = freq ? freq : 1000000000ull;
    size_t elf_len = 0;
    void* elf = slurp(envs("MQ_ELF", "mq_probe_kernels.elf"), &elf_len);
    if (!elf) { fprintf(stderr, "no ELF\n"); return 2; }

    plow_hsa_kernel kfill[NR], kxr[NR], kgemm;
    void *scratch[NR], *table[NR], *out[NR], *xrts[NR];
    uint32_t* status[NR];
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "mq_fill_partial", &kfill[r]) ||
            plow_hsa_get_kernel(h, dev[r], "mq_xreduce_twoshot_ts", &kxr[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], REGION_BYTES);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        out[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        xrts[r] = plow_hsa_alloc(h, dev[r], 16);
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !table[r] || !out[r] || !xrts[r] || !status[r]) {
            fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    if (plow_hsa_get_kernel(h, dev[0], "mq_gemm_proxy", &kgemm)) {
        fprintf(stderr, "load gemm: %s\n", plow_hsa_last_error()); return 2;
    }
    for (int r = 0; r < NR; r++)
        plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);

    /* Second queue on rank 0's agent, for MQ_ARM=2q. Built unconditionally (cheap) so every
     * arm's setup is identical up to the dispatch step. */
    q2_t q2;
    hsa_agent_t agent0 = { .handle = plow_hsa_agent_raw(h, dev[0]) };
    hsa_amd_memory_pool_t kpool = { .handle = plow_hsa_kernarg_pool_raw(h) };
    if (q2_init(&q2, agent0, kpool) != 0) { fprintf(stderr, "q2_init failed\n"); return 2; }
    if (masked) {
        /* 304 CUs = 10 x u32 words. mask0 = CUs [0,32) for the collective's own queue; mask1 =
         * CUs [32,304) for q2. Disjoint and exhaustive by construction. */
        uint32_t mask0[10] = {0}, mask1[10] = {0};
        for (uint32_t c = 0; c < 304; c++)
            (c < 32 ? mask0 : mask1)[c / 32] |= 1u << (c % 32);
        hsa_queue_t* q0 = (hsa_queue_t*)(uintptr_t)plow_hsa_queue_raw(h, dev[0]);
        /* num_cu_mask_count must be a multiple of 32 (hsa_ext_amd.h) — 320, not 304; the
         * high 16 bits of word 9 name CUs that don't exist and are left 0 (masked off). */
        if (hsa_amd_queue_cu_set_mask(q0, 320, mask0) != HSA_STATUS_SUCCESS ||
            hsa_amd_queue_cu_set_mask(q2.q, 320, mask1) != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "cu_set_mask failed\n"); return 2;
        }
    }

    void* gemm_out = plow_hsa_alloc(h, dev[0], (size_t)gemm_nwg * 4u);
    void* gemm_ts = plow_hsa_alloc(h, dev[0], (size_t)gemm_nwg * 2u * 8u);
    uint64_t* xr_ts_h = malloc(16);
    uint64_t* gemm_ts_h = malloc((size_t)gemm_nwg * 2u * 8u);
    double* wall = calloc(reps, sizeof(double));
    double* xr_dur = calloc(reps, sizeof(double));
    double* gemm_dur = calloc(reps, sizeof(double));
    double* overlap = calloc(reps, sizeof(double));
    int any_timeout = 0;

    const int run_xr = strcmp(arm, "solo_gemm") != 0;
    const int run_gemm = strcmp(arm, "solo_xr") != 0;

    for (uint32_t rep = 0; rep < reps; rep++) {
        uint8_t zero[8192] = {0};
        for (int r = 0; r < NR; r++) {
            if (!run_xr) break;
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + XCTR_OFF, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            arg_fill fa = {scratch[r], (uint32_t)n, (uint32_t)r};
            plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fa, sizeof fa);
            plow_hsa_wait(h, dev[r]);
        }

        arg_xr axr = {out[0], table[0], NR, 0u, (uint32_t)n, 0u, XCTR_OFF, deadline, xrts[0], status[0]};
        arg_gemm agm = {gemm_out, gemm_iters, gemm_ts};

        if (!strcmp(arm, "solo_xr")) {
            plow_hsa_launch(h, dev[0], &kxr[0], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &axr, sizeof axr);
            for (int r = 1; r < NR; r++) {
                arg_xr a = {out[r], table[r], NR, (uint32_t)r, (uint32_t)n, 0u, XCTR_OFF, deadline, NULL, status[r]};
                plow_hsa_launch(h, dev[r], &kxr[r], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &a, sizeof a);
            }
            for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
        } else if (!strcmp(arm, "solo_gemm")) {
            plow_hsa_launch(h, dev[0], &kgemm, gemm_nwg * 512u, 1, 1, 512, 1, 1, 0, &agm, sizeof agm);
            plow_hsa_wait(h, dev[0]);
        } else if (!strcmp(arm, "1q")) {
            /* TODAY: same queue, plow_hsa_launch always sets the barrier bit. */
            plow_hsa_launch(h, dev[0], &kxr[0], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &axr, sizeof axr);
            plow_hsa_launch(h, dev[0], &kgemm, gemm_nwg * 512u, 1, 1, 512, 1, 1, 0, &agm, sizeof agm);
            for (int r = 1; r < NR; r++) {
                arg_xr a = {out[r], table[r], NR, (uint32_t)r, (uint32_t)n, 0u, XCTR_OFF, deadline, NULL, status[r]};
                plow_hsa_launch(h, dev[r], &kxr[r], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &a, sizeof a);
            }
            for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
            q2_wait(&q2); /* no-op: q2 unused in this arm, but keeps the arms symmetric */
        } else { /* 2q, 2qm */
            plow_hsa_launch(h, dev[0], &kxr[0], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &axr, sizeof axr);
            q2_launch(&q2, &kgemm, gemm_nwg * 512u, 512, &agm, sizeof agm, 0 /* no barrier */);
            for (int r = 1; r < NR; r++) {
                arg_xr a = {out[r], table[r], NR, (uint32_t)r, (uint32_t)n, 0u, XCTR_OFF, deadline, NULL, status[r]};
                plow_hsa_launch(h, dev[r], &kxr[r], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &a, sizeof a);
            }
            for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
            q2_wait(&q2);
        }

        if (run_xr) { uint32_t st = 0; plow_hsa_download(h, dev[0], &st, status[0], 4); any_timeout |= st != 0; }
        if (run_xr) plow_hsa_download(h, dev[0], xr_ts_h, xrts[0], 16);
        if (run_gemm) plow_hsa_download(h, dev[0], gemm_ts_h, gemm_ts, (size_t)gemm_nwg * 2u * 8u);

        double xr0 = 0, xr1 = 0, g0 = 1e18, g1 = 0;
        if (run_xr) { xr0 = (double)xr_ts_h[0] * 0.01; xr1 = (double)xr_ts_h[1] * 0.01; }
        if (run_gemm)
            for (uint32_t w = 0; w < gemm_nwg; w++) {
                double s = (double)gemm_ts_h[2 * w] * 0.01, e = (double)gemm_ts_h[2 * w + 1] * 0.01;
                if (s < g0) g0 = s;
                if (e > g1) g1 = e;
            }
        xr_dur[rep] = run_xr ? xr1 - xr0 : 0;
        gemm_dur[rep] = run_gemm ? g1 - g0 : 0;
        if (run_xr && run_gemm) {
            const double lo = xr0 > g0 ? xr0 : g0, hi = xr1 < g1 ? xr1 : g1;
            overlap[rep] = hi > lo ? hi - lo : 0.0;
            const double wlo = xr0 < g0 ? xr0 : g0, whi = xr1 > g1 ? xr1 : g1;
            wall[rep] = whi - wlo;
        } else {
            wall[rep] = run_xr ? xr_dur[rep] : gemm_dur[rep];
        }
    }

    printf("arm=%s rows=%u hidden=%u xr_nwg=%u gemm_nwg=%u gemm_iters=%u reps=%u timeout=%s\n",
           arm, rows, hidden, xr_nwg, gemm_nwg, gemm_iters, reps, any_timeout ? "YES" : "no");
    printf("  xr_us=%.2f gemm_us=%.2f overlap_us=%.2f wall_us=%.2f\n",
           med(xr_dur, reps), med(gemm_dur, reps), med(overlap, reps), med(wall, reps));

    /* Parity: the collective's output on every rank, from the LAST rep, against the same
     * strict-rank-order f32-sum oracle tp_allreduce_prefill_bench.c uses. Proves the reduction
     * is still bit-exact when co-resident with the CU-contending GEMM proxy — timing overlap
     * without a numeric check would not answer the safety question this probe exists for. */
    size_t bad = 0;
    if (run_xr) {
        bf16* host = malloc((size_t)n * 2u);
        bf16* want = malloc((size_t)n * 2u);
        for (uint64_t e = 0; e < n; e++) {
            float sum = 0.0f;
            for (int r = 0; r < NR; r++)
                sum += (float)(r + 1) * (1.0f + (float)((e) & 7u) * 0.125f);
            uint32_t u; memcpy(&u, &sum, 4);
            u += 0x7fffu + ((u >> 16) & 1u);
            want[e] = (bf16)(u >> 16);
        }
        for (int r = 0; r < NR; r++) {
            plow_hsa_download(h, dev[r], host, out[r], (size_t)n * 2u);
            for (uint64_t e = 0; e < n; e++) bad += host[e] != want[e];
        }
        printf("  parity=%s bad=%zu (checked all %d ranks, n=%llu)\n",
               bad ? "FAIL" : "PASS", bad, NR, (unsigned long long)n);
        free(host); free(want);
    }
    plow_hsa_shutdown(h);
    return (any_timeout || bad) ? 1 : 0;
}
