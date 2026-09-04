/* aql_launch_floor.c — what does ONE AQL kernel dispatch actually cost on gfx950, via raw HSA?
 *
 * WHY THIS EXISTS. plow runs ONE persistent kernel per token whose 256 workgroups self-schedule
 * over a 2459-packet list and order themselves with global counters. That protocol measures
 * 5.7-7.4 us per packet, and it is CACHE MAINTENANCE, not contention: the consumer's agent-scope
 * acquire emits `buffer_inv` and the producer's release RMW emits `buffer_wbl2`, cache-WIDE, once
 * per workgroup, 32 workgroups per L2, serialised. See runtime/bench/ctr_convergence.hip.
 *
 * The obvious alternative is to stop doing it in software: put one AQL kernel-dispatch packet per
 * plow packet on the queue and let the AQL BARRIER BIT order them. The barrier bit is a hardware
 * dependency edge -- the packet processor will not launch packet k+1 until packet k has completed
 * -- and the packet header's scacquire/screlease fence-scope fields make the CP do the cache
 * maintenance ONCE per packet instead of once per workgroup. If the back-to-back period of an
 * empty barrier-bit-chained dispatch is well under 5.7 us, host-side op-by-op dispatch is a
 * serious contender for the whole megakernel.
 *
 * This is deliberately RAW HSA, not HIP: plow's own backend is raw HSA
 * (crates/plowrt/src/device/hsa.rs `dispatch`), and every packet field below is set to exactly
 * what that function sets, so the number is plow's launch cost and not hipLaunchKernel's.
 *
 * ARMS (--arm, or `all`):
 *   chain          barrier=1, no completion signal, AGENT/AGENT fences, doorbell per packet.
 *                  THE HEADLINE NUMBER: steady-state launch-to-launch period of an ordered chain.
 *   chain-sig      as `chain` plus plow's exact completion-signal scheme (one shared counting
 *                  signal, hsa_signal_add_screlease(+1) per dispatch). Prices the signal.
 *   chain-nofence  as `chain` with fence scope NONE on both sides. Prices the CP's cache
 *                  maintenance -- the hardware analogue of ctr_convergence's -DNOACQ -DRELAXSIG.
 *   chain-sysfence as `chain` with fence scope SYSTEM on both sides.
 *   free           barrier=0, no signal. Packets may run concurrently: raw packet-processor
 *                  ISSUE throughput, a lower bound that does NOT order anything.
 *   roundtrip      barrier=1, a fresh completion signal per packet, and the host BLOCKS on packet
 *                  k before enqueuing k+1. The cost if the host must actually see each op finish.
 *   prebuild       as `chain` but all N packets are written to the ring first and the doorbell is
 *                  rung ONCE. Removes any host-enqueue interleaving from the period.
 *   verify-prebuild the heterogeneous pub/check/bump exactness chain, reserved as one contiguous
 *                  replay and committed with one doorbell. This is the phase-chain correctness
 *                  gate, not merely another empty-kernel timing arm.
 *
 * Every arm reports us/packet plus the host-side enqueue cost, so it is visible whether the
 * measurement is GPU-bound (enqueue << total) or host-bound.
 *
 * BUILD:
 *   hipcc --genco --offload-arch=gfx950 -O3 -std=c++17 -w aql_launch_floor.hip -o aqlfloor.elf
 *   cc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -I/opt/rocm/include aql_launch_floor.c \
 *      -o aqlfloor -L/opt/rocm/lib -lhsa-runtime64
 * RUN (pin an IDLE GPU -- this box is shared):
 *   sg render -c 'LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=5 ./aqlfloor'
 */
#define _POSIX_C_SOURCE 199309L
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define QSIZE 4096u
#define KARG_SLOT 1024u

static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}
#define CHK(x, m) do { hsa_status_t _s = (x); if (_s != HSA_STATUS_SUCCESS) { \
    fprintf(stderr, "FAIL %s: %d\n", (m), (int)_s); exit(1); } } while (0)

/* --- AQL dispatch packet, transcribed like hsa.rs's HsaDispatchPacket ------------------------ */
typedef struct {
    uint16_t header, setup;
    uint16_t wg_x, wg_y, wg_z, rsvd0;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t private_segment_size, group_segment_size;
    uint64_t kernel_object, kernarg_address, rsvd2;
    hsa_signal_t completion_signal;
} pkt_t;

static hsa_agent_t g_agent, g_cpu;
static int g_found, g_found_cpu;
static hsa_status_t on_agent(hsa_agent_t a, void* d) {
    (void)d; hsa_device_type_t t;
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    if (t == HSA_DEVICE_TYPE_GPU && !g_found) { g_agent = a; g_found = 1; }
    /* The KERNARG and FINE_GRAINED system pools belong to the CPU agent, not the GPU -- picking
     * them off the GPU agent is what made the first run of this bench die with
     * "no pool with flags 0x1". `runtime/amd/hsa_backend.c:101` picks both off `h->cpu`. */
    if (t == HSA_DEVICE_TYPE_CPU && !g_found_cpu) { g_cpu = a; g_found_cpu = 1; }
    return HSA_STATUS_SUCCESS;
}
/* Every host allocation the GPU touches has to be made reachable from it. */
static void* alloc_shared(hsa_amd_memory_pool_t pool, size_t bytes) {
    void* p;
    CHK(hsa_amd_memory_pool_allocate(pool, bytes, 0, &p), "pool_allocate");
    CHK(hsa_amd_agents_allow_access(1, &g_agent, NULL, p), "allow_access");
    memset(p, 0, bytes);
    return p;
}
typedef struct { hsa_amd_memory_pool_t p; uint32_t want; int found; } pick_t;
static hsa_status_t on_pool(hsa_amd_memory_pool_t p, void* d) {
    pick_t* k = (pick_t*)d; hsa_amd_segment_t seg; uint32_t fl;
    if (k->found) return HSA_STATUS_SUCCESS;
    if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) != HSA_STATUS_SUCCESS
        || seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (fl & k->want) { k->p = p; k->found = 1; }
    return HSA_STATUS_SUCCESS;
}
static hsa_amd_memory_pool_t pick_pool(hsa_agent_t a, uint32_t want) {
    pick_t k; memset(&k, 0, sizeof k); k.want = want;
    CHK(hsa_amd_agent_iterate_memory_pools(a, on_pool, &k), "iterate_pools");
    if (!k.found) { fprintf(stderr, "no pool with flags 0x%x\n", want); exit(1); }
    return k.p;
}

/* header bit positions and fence scopes come straight from hsa.h */
static uint16_t mk_header(int barrier, int fence_scope) {
    return (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
        | ((barrier ? 1 : 0) << HSA_PACKET_HEADER_BARRIER)
        | (fence_scope << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (fence_scope << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

typedef struct {
    hsa_queue_t* q;
    uint64_t kobj;
    uint32_t karg_size;
    uint8_t* karg;                 /* one shared kernarg block; every packet passes the same arg */
    uint32_t wg, blocks, lds;
    hsa_signal_t shared_sig;
} ctx_t;

/* Write packet at ring slot `slot` (header left for the caller's release store). */
static pkt_t* fill(ctx_t* c, uint64_t idx, hsa_signal_t sig) {
    uint32_t slot = (uint32_t)(idx & (QSIZE - 1));
    pkt_t* p = (pkt_t*)((uint8_t*)c->q->base_address + (size_t)slot * 64);
    memset((uint8_t*)p + 4, 0, 60);
    p->wg_x = (uint16_t)c->wg; p->wg_y = 1; p->wg_z = 1;
    p->grid_x = c->wg * c->blocks; p->grid_y = 1; p->grid_z = 1;
    p->kernel_object = c->kobj;
    p->kernarg_address = (uint64_t)(uintptr_t)c->karg;
    p->group_segment_size = c->lds;
    p->private_segment_size = 0;
    p->completion_signal = sig;
    return p;
}
static void publish(ctx_t* c, pkt_t* p, uint64_t idx, uint16_t header) {
    uint32_t hs = ((uint32_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) << 16) | header;
    __atomic_store_n((uint32_t*)p, hs, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(c->q->doorbell_signal, (hsa_signal_value_t)idx);
}
static void ring_space(ctx_t* c, uint64_t idx) {
    while (idx - hsa_queue_load_read_index_scacquire(c->q) >= QSIZE) { }
}

/* ---- arms ---------------------------------------------------------------------------------- */
typedef struct { double us_per_pkt, enq_us_per_pkt; } res_t;

static res_t arm_stream(ctx_t* c, uint32_t n, int barrier, int fence, int use_sig) {
    hsa_signal_t none = { .handle = 0 };
    uint16_t hdr = mk_header(barrier, fence);
    if (use_sig) hsa_signal_store_screlease(c->shared_sig, 0);
    /* a tail signal so the host can tell when the LAST packet retired */
    hsa_signal_t tail; CHK(hsa_signal_create(1, 0, NULL, &tail), "sig_create tail");
    double t0 = now_us();
    for (uint32_t i = 0; i < n; i++) {
        uint64_t idx = hsa_queue_add_write_index_screlease(c->q, 1);
        ring_space(c, idx);
        hsa_signal_t s = none;
        if (i == n - 1) s = tail;
        else if (use_sig) { s = c->shared_sig; hsa_signal_add_screlease(c->shared_sig, 1); }
        pkt_t* p = fill(c, idx, s);
        publish(c, p, idx, hdr);
    }
    double t_enq = now_us();
    hsa_signal_wait_scacquire(tail, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    if (use_sig)
        hsa_signal_wait_scacquire(c->shared_sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    double t1 = now_us();
    hsa_signal_destroy(tail);
    res_t r = { (t1 - t0) / n, (t_enq - t0) / n };
    return r;
}

static res_t arm_prebuild(ctx_t* c, uint32_t n, int fence) {
    if (n > QSIZE) n = QSIZE;
    hsa_signal_t none = { .handle = 0 };
    uint16_t hdr = mk_header(1, fence);
    hsa_signal_t tail; CHK(hsa_signal_create(1, 0, NULL, &tail), "sig_create tail");
    uint64_t base = hsa_queue_add_write_index_screlease(c->q, n);
    ring_space(c, base + n - 1);
    double t0 = now_us();
    for (uint32_t i = 0; i < n; i++) {
        pkt_t* p = fill(c, base + i, i == n - 1 ? tail : none);
        uint32_t hs = ((uint32_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) << 16) | hdr;
        __atomic_store_n((uint32_t*)p, hs, __ATOMIC_RELEASE);
    }
    double t_enq = now_us();
    hsa_signal_store_screlease(c->q->doorbell_signal, (hsa_signal_value_t)(base + n - 1));
    hsa_signal_wait_scacquire(tail, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    double t1 = now_us();
    hsa_signal_destroy(tail);
    res_t r = { (t1 - t0) / n, (t_enq - t0) / n };
    return r;
}

static res_t arm_roundtrip(ctx_t* c, uint32_t n, int fence) {
    uint16_t hdr = mk_header(1, fence);
    hsa_signal_t s; CHK(hsa_signal_create(1, 0, NULL, &s), "sig_create rt");
    double t0 = now_us();
    for (uint32_t i = 0; i < n; i++) {
        hsa_signal_store_relaxed(s, 1);
        uint64_t idx = hsa_queue_add_write_index_screlease(c->q, 1);
        ring_space(c, idx);
        pkt_t* p = fill(c, idx, s);
        publish(c, p, idx, hdr);
        hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
    double t1 = now_us();
    hsa_signal_destroy(s);
    res_t r = { (t1 - t0) / n, 0.0 };
    return r;
}

/* ---- (a) DEVICE-SIDE ENQUEUE -----------------------------------------------------------------
 * `d_parent` (one workgroup on queue q1) writes AQL packets into q2 and rings q2's doorbell with
 * `__ockl_hsa_signal_store`, then spins on a flag the child sets. Both kernels read the same
 * device-wide 100 MHz `s_memrealtime`, so t_child - t_ring is the doorbell -> child-executing
 * latency with no host in the loop at all. */
typedef struct { uint64_t* t_child; uint32_t* flag; uint32_t* seq; } child_args_t;
typedef struct {
    const void* queue; uint64_t doorbell; uint64_t qbase; uint32_t qsize; uint32_t pad;
    uint64_t child_kobj, child_karg; uint32_t* flag; uint64_t* t_ring; uint32_t* seq; int n;
} parent_args_t;

static int cmpu64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b; return x < y ? -1 : x > y;
}

static void arm_devenq(ctx_t* c, hsa_executable_t exe, hsa_amd_memory_pool_t fine,
                       hsa_amd_memory_pool_t kpool, uint32_t n) {
    hsa_executable_symbol_t sp, sc;
    uint64_t pobj, cobj; uint32_t pksz, cksz;
    if (hsa_executable_get_symbol_by_name(exe, "d_parent.kd", &g_agent, &sp) != HSA_STATUS_SUCCESS ||
        hsa_executable_get_symbol_by_name(exe, "d_child.kd",  &g_agent, &sc) != HSA_STATUS_SUCCESS) {
        printf("devenq: d_parent/d_child not in the code object -- skipped\n"); return;
    }
    CHK(hsa_executable_symbol_get_info(sp, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &pobj), "pobj");
    CHK(hsa_executable_symbol_get_info(sc, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &cobj), "cobj");
    CHK(hsa_executable_symbol_get_info(sp, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &pksz), "pksz");
    CHK(hsa_executable_symbol_get_info(sc, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &cksz), "cksz");

    hsa_queue_t* q2;
    CHK(hsa_queue_create(g_agent, QSIZE, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &q2), "q2_create");

    uint64_t* t_ring  = (uint64_t*)alloc_shared(fine, (size_t)n * 8);
    uint64_t* t_child = (uint64_t*)alloc_shared(fine, (size_t)n * 8);
    uint32_t* flag    = (uint32_t*)alloc_shared(fine, (size_t)n * 4 + 64);
    uint32_t* seq     = (uint32_t*)alloc_shared(fine, 64);
    uint8_t*  pk      = (uint8_t*) alloc_shared(kpool, KARG_SLOT);
    uint8_t*  ck      = (uint8_t*) alloc_shared(kpool, KARG_SLOT);

    child_args_t ca = { t_child, flag, seq };
    memcpy(ck, &ca, sizeof ca);
    parent_args_t pa;
    memset(&pa, 0, sizeof pa);
    pa.queue = q2; pa.doorbell = q2->doorbell_signal.handle; pa.qbase = (uint64_t)q2->base_address;
    pa.qsize = QSIZE; pa.child_kobj = cobj; pa.child_karg = (uint64_t)(uintptr_t)ck;
    pa.flag = flag; pa.t_ring = t_ring; pa.seq = seq; pa.n = (int)n;
    memcpy(pk, &pa, sizeof pa);

    /* WARM q2 FROM THE HOST FIRST. A queue that has never been rung may not yet be attached to a
     * hardware queue descriptor, and then a null result would say "the device doorbell did not
     * work" when it actually said "that queue was never live". One host dispatch of the same child
     * removes the ambiguity; the counters are re-zeroed afterwards. */
    {
        ctx_t c2 = *c; c2.q = q2;
        hsa_signal_t w; CHK(hsa_signal_create(1, 0, NULL, &w), "warm sig");
        uint64_t wi = hsa_queue_add_write_index_screlease(q2, 1);
        pkt_t* wp = (pkt_t*)((uint8_t*)q2->base_address + (size_t)(wi & (QSIZE - 1)) * 64);
        memset((uint8_t*)wp + 4, 0, 60);
        wp->wg_x = 64; wp->wg_y = 1; wp->wg_z = 1; wp->grid_x = 64; wp->grid_y = 1; wp->grid_z = 1;
        wp->kernel_object = cobj; wp->kernarg_address = (uint64_t)(uintptr_t)ck;
        wp->completion_signal = w;
        publish(&c2, wp, wi, mk_header(1, HSA_FENCE_SCOPE_SYSTEM));
        if (hsa_signal_wait_scacquire(w, HSA_SIGNAL_CONDITION_EQ, 0,
                                      5ull * 1000 * 1000 * 1000, HSA_WAIT_STATE_ACTIVE) != 0) {
            printf("devenq: the CHILD QUEUE does not even work from the host -- aborting arm\n");
            return;
        }
        hsa_signal_destroy(w);
        printf("devenq          child queue warmed from host ok (seq=%u)\n", *seq);
        memset(t_child, 0, (size_t)n * 8); memset(flag, 0, (size_t)n * 4 + 64); memset(seq, 0, 64);
    }

    /* dispatch the parent on the ORIGINAL queue; children land on q2 (never the parent's own
     * queue -- an in-order queue plus a parent waiting on a child behind it is a deadlock). */
    hsa_signal_t done; CHK(hsa_signal_create(1, 0, NULL, &done), "done");
    uint64_t idx = hsa_queue_add_write_index_screlease(c->q, 1);
    ring_space(c, idx);
    uint32_t slot = (uint32_t)(idx & (QSIZE - 1));
    pkt_t* p = (pkt_t*)((uint8_t*)c->q->base_address + (size_t)slot * 64);
    memset((uint8_t*)p + 4, 0, 60);
    p->wg_x = 64; p->wg_y = 1; p->wg_z = 1; p->grid_x = 64; p->grid_y = 1; p->grid_z = 1;
    p->kernel_object = pobj; p->kernarg_address = (uint64_t)(uintptr_t)pk;
    p->completion_signal = done;
    publish(c, p, idx, mk_header(1, HSA_FENCE_SCOPE_SYSTEM));

    double t0 = now_us();
    hsa_signal_value_t v = hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_EQ, 0,
                                                     20ull * 1000 * 1000 * 1000, HSA_WAIT_STATE_ACTIVE);
    double t1 = now_us();
    if (v != 0) {
        printf("devenq: PARENT DID NOT FINISH in 20s (children seen: %u of %u). "
               "The device-written packet was not consumed.\n", *seq, n);
        return;
    }
    /* s_memrealtime ticks: 100 per us on gfx950 (the constant the other benches use). */
    const double TPUS = 100.0;
    uint64_t* d = (uint64_t*)malloc((size_t)n * 8);
    uint32_t m = 0;
    for (uint32_t i = 1; i < n; i++)                      /* drop i=0: cold */
        if (t_child[i] > t_ring[i]) d[m++] = t_child[i] - t_ring[i];
    if (!m) { printf("devenq: no usable samples\n"); return; }
    qsort(d, m, 8, cmpu64);
    printf("devenq          children=%u/%u   doorbell->child-entry: p50 %.3f us  p10 %.3f  p90 %.3f\n",
           *seq, n, d[m / 2] / TPUS, d[m / 10] / TPUS, d[(size_t)m * 9 / 10] / TPUS);
    printf("devenq          full parent round trip (ring->flag seen, serialised): %.3f us/pkt\n",
           (t1 - t0) / n);
    free(d);
}

int main(int argc, char** argv) {
    const char* kname = "d_nop.kd";
    const char* elf = "aqlfloor.elf";
    const char* only = "all";
    const char* kargloc = "vram";
    uint32_t n = 2000, wg = 512, blocks = 256, lds = 151040, spin = 0, reps = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) n = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wg") && i + 1 < argc) wg = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lds") && i + 1 < argc) lds = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kname = argv[++i];
        else if (!strcmp(argv[i], "--elf") && i + 1 < argc) elf = argv[++i];
        else if (!strcmp(argv[i], "--arm") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--spin") && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--karg") && i + 1 < argc) kargloc = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    /* unbuffered: a hang inside a timing loop must not erase the output that says where */
    setvbuf(stdout, NULL, _IONBF, 0);
    CHK(hsa_init(), "hsa_init");
    CHK(hsa_iterate_agents(on_agent, NULL), "iterate_agents");
    if (!g_found) { fprintf(stderr, "no GPU agent\n"); return 1; }
    if (!g_found_cpu) { fprintf(stderr, "no CPU agent\n"); return 1; }
    char name[64] = {0};
    CHK(hsa_agent_get_info(g_agent, HSA_AGENT_INFO_NAME, name), "agent_name");

    hsa_queue_t* q;
    CHK(hsa_queue_create(g_agent, QSIZE, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &q), "queue_create");
    if (q->size != QSIZE) { fprintf(stderr, "queue size %u != %u\n", q->size, QSIZE); return 1; }
    fprintf(stderr, "# queue ok: size=%u base=%p doorbell=%llu\n", q->size, q->base_address,
            (unsigned long long)q->doorbell_signal.handle);

    hsa_amd_memory_pool_t kpool = pick_pool(g_cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);
    hsa_amd_memory_pool_t fine  = pick_pool(g_cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
    hsa_amd_memory_pool_t vram = pick_pool(g_agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
    /* WHERE THE KERNARG BLOCK LIVES IS NOT A DETAIL. `d_nop` reads no kernarg and dispatches in
     * 1.46 us; `d_store`, which differs only by loading its two pointers, measured 25 us -- because
     * the HSA kernarg pool is HOST fine-grained memory and all 256 workgroups fetch it across the
     * host link. `hsa_kernel_dispatch_packet_t.kernarg_address` may point at any memory the agent
     * can read, so `--karg vram` stages it in device memory instead. This is the difference between
     * host-side op-by-op dispatch being viable and being hopeless. */
    uint8_t* karg;
    if (!strcmp(kargloc, "vram")) {
        CHK(hsa_amd_memory_pool_allocate(vram, KARG_SLOT, 0, (void**)&karg), "karg vram");
    } else {
        karg = (uint8_t*)alloc_shared(kpool, KARG_SLOT);
    }
    uint8_t* kstage = (uint8_t*)alloc_shared(kpool, KARG_SLOT);  /* host staging for the H2D copy */
    uint32_t* ctl = (uint32_t*)alloc_shared(fine, 4096);        /* host fine-grained: spin count */
    uint32_t* out;                                              /* DEVICE VRAM: the store target */
    CHK(hsa_amd_memory_pool_allocate(vram, 8192, 0, (void**)&out), "vram_alloc");
    ctl[1] = spin;

    /* load the code object */
    FILE* f = fopen(elf, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    hsa_code_object_reader_t rdr;
    CHK(hsa_code_object_reader_create_from_memory(buf, (size_t)sz, &rdr), "cor_create");
    hsa_executable_t exe;
    CHK(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &exe),
        "exe_create");
    CHK(hsa_executable_load_agent_code_object(exe, g_agent, rdr, NULL, NULL), "exe_load");
    CHK(hsa_executable_freeze(exe, NULL), "exe_freeze");
    hsa_executable_symbol_t sym;
    CHK(hsa_executable_get_symbol_by_name(exe, kname, &g_agent, &sym), "get_symbol");
    ctx_t c; memset(&c, 0, sizeof c);
    CHK(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &c.kobj), "kobj");
    uint32_t ksz, kernel_lds;
    CHK(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &ksz), "ksz");
    CHK(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kernel_lds), "klds");

    /* kernarg: {const u32* ctl, u32* out}, then a zeroed COv5 implicit block (the kernel reads no hidden args
     * beyond what hipcc emits, and zeros are valid for the ones d_nop/d_store/d_spin never read). */
    memset(kstage, 0, KARG_SLOT);
    memcpy(kstage, &ctl, 8);
    memcpy(kstage + 8, &out, 8);
    if (karg != kstage) {
        hsa_signal_t cs; CHK(hsa_signal_create(1, 0, NULL, &cs), "copy sig");
        CHK(hsa_amd_memory_async_copy(karg, g_agent, kstage, g_cpu, KARG_SLOT, 0, NULL, cs), "h2d karg");
        hsa_signal_wait_scacquire(cs, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
        hsa_signal_destroy(cs);
    }

    c.q = q; c.karg = karg; c.karg_size = ksz; c.wg = wg; c.blocks = blocks; c.lds = lds;
    CHK(hsa_signal_create(0, 0, NULL, &c.shared_sig), "sig_create shared");

    printf("# agent=%s kernel=%s kernarg=%uB kernel_lds=%uB\n", name, kname, ksz, kernel_lds);
    printf("# grid: %u blocks x %u thr, packet group_segment_size=%u B (occupancy 1/CU above ~80KB)\n",
           blocks, wg, lds);
    printf("# iters=%u reps=%u spin_ticks=%u queue=%u kernarg_in=%s\n", n, reps, spin, QSIZE, kargloc);
    /* SANITY: one dispatch, bounded wait. A hang here means the packet is malformed, and a silent
     * hang inside a timing loop is unreadable -- so fail loudly before any measurement. */
    {
        hsa_signal_t s; CHK(hsa_signal_create(1, 0, NULL, &s), "sig_create sanity");
        uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
        ring_space(&c, idx);
        pkt_t* p = fill(&c, idx, s);
        publish(&c, p, idx, mk_header(1, HSA_FENCE_SCOPE_AGENT));
        hsa_signal_value_t v = hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_EQ, 0,
                                                         5ull * 1000 * 1000 * 1000, HSA_WAIT_STATE_ACTIVE);
        if (v != 0) {
            fprintf(stderr, "SANITY DISPATCH TIMED OUT (signal=%lld). read_index=%llu write=%llu\n",
                    (long long)v, (unsigned long long)hsa_queue_load_read_index_scacquire(q),
                    (unsigned long long)hsa_queue_load_write_index_scacquire(q));
            return 1;
        }
        hsa_signal_destroy(s);
        fprintf(stderr, "# sanity dispatch ok\n");
    }

    printf("%-16s %10s %10s   %s\n", "arm", "us/pkt", "enq us/pkt", "note");

    struct { const char* nm; int barrier, fence, sig, kind; const char* note; } arms[] = {
        { "chain",          1, HSA_FENCE_SCOPE_AGENT,  0, 0, "barrier bit, no compl signal, agent fences" },
        { "chain-sig",      1, HSA_FENCE_SCOPE_AGENT,  1, 0, "+ plow's shared counting compl signal" },
        { "chain-nofence",  1, HSA_FENCE_SCOPE_NONE,   0, 0, "barrier bit, NO cache maintenance (UNSAFE)" },
        { "chain-sysfence", 1, HSA_FENCE_SCOPE_SYSTEM, 0, 0, "barrier bit, system-scope fences" },
        { "free",           0, HSA_FENCE_SCOPE_AGENT,  0, 0, "NO barrier bit: concurrent, orders nothing" },
        { "prebuild",       1, HSA_FENCE_SCOPE_AGENT,  0, 1, "all packets written, one doorbell" },
        { "roundtrip",      1, HSA_FENCE_SCOPE_AGENT,  0, 2, "host blocks on each packet's signal" },
    };
    for (size_t a = 0; a < sizeof arms / sizeof arms[0]; a++) {
        if (strcmp(only, "all") && strcmp(only, arms[a].nm)) continue;
        res_t best = { 1e30, 0 };
        for (uint32_t r = 0; r < reps; r++) {
            res_t x;
            uint32_t nn = arms[a].kind == 2 ? (n > 400 ? 400 : n) : n;
            if (arms[a].kind == 1) x = arm_prebuild(&c, nn, arms[a].fence);
            else if (arms[a].kind == 2) x = arm_roundtrip(&c, nn, arms[a].fence);
            else x = arm_stream(&c, nn, arms[a].barrier, arms[a].fence, arms[a].sig);
            if (x.us_per_pkt < best.us_per_pkt) best = x;
        }
        printf("%-16s %10.3f %10.3f   %s\n", arms[a].nm, best.us_per_pkt, best.enq_us_per_pkt, arms[a].note);
        fflush(stdout);
    }
    if (!strcmp(only, "verify") || !strcmp(only, "verify-prebuild")) {
        const int prebuilt = !strcmp(only, "verify-prebuild");
        /* alternate d_pub / d_chk down one barrier-bit-chained queue, plain accesses only */
        uint64_t pubo, chko, bumpo;
        hsa_executable_symbol_t sy;
        CHK(hsa_executable_get_symbol_by_name(exe, "d_pub.kd", &g_agent, &sy), "d_pub");
        CHK(hsa_executable_symbol_get_info(sy, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &pubo), "pubo");
        CHK(hsa_executable_get_symbol_by_name(exe, "d_chk.kd", &g_agent, &sy), "d_chk");
        CHK(hsa_executable_symbol_get_info(sy, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &chko), "chko");
        CHK(hsa_executable_get_symbol_by_name(exe, "d_bump.kd", &g_agent, &sy), "d_bump");
        CHK(hsa_executable_symbol_get_info(sy, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &bumpo), "bumpo");
        /* The scratch MUST be in VRAM. In host fine-grained memory this arm moves 1 MB across the
         * host link per packet in each direction and never finishes. */
        uint32_t* dbg;
        CHK(hsa_amd_memory_pool_allocate(vram, 4096 + 256 * 1024 * 4, 0, (void**)&dbg), "dbg vram");
        {   uint8_t* z = (uint8_t*)alloc_shared(fine, 4096);
            memset(z, 0, 4096);
            hsa_signal_t zs; CHK(hsa_signal_create(1, 0, NULL, &zs), "zero sig");
            CHK(hsa_amd_memory_async_copy(dbg, g_agent, z, g_cpu, 4096, 0, NULL, zs), "zero dbg");
            hsa_signal_wait_scacquire(zs, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            hsa_signal_destroy(zs); }
        ctl[2] = 1024; ctl[3] = c.blocks; ctl[4] = c.wg;   /* pw, blocks, threads */
        memcpy(kstage, &ctl, 8); memcpy(kstage + 8, &dbg, 8);
        uint8_t* vk = (uint8_t*)alloc_shared(kpool, KARG_SLOT);
        memcpy(vk, kstage, KARG_SLOT);
        hsa_signal_t tail; CHK(hsa_signal_create(1, 0, NULL, &tail), "verify tail");
        uint16_t hdr = mk_header(1, HSA_FENCE_SCOPE_AGENT);
        uint32_t np = (n / 3) * 3;
        uint64_t base = 0;
        if (prebuilt) {
            base = hsa_queue_add_write_index_screlease(q, np);
            ring_space(&c, base + np - 1);
        }
        for (uint32_t i = 0; i < np; i++) {
            uint64_t ix = prebuilt ? base + i : hsa_queue_add_write_index_screlease(q, 1);
            if (!prebuilt) ring_space(&c, ix);
            pkt_t* pp = fill(&c, ix, i == np - 1 ? tail : (hsa_signal_t){ .handle = 0 });
            uint32_t ph = i % 3;                       /* 0 pub(256), 1 chk(256), 2 bump(1) */
            pp->kernel_object = ph == 0 ? pubo : ph == 1 ? chko : bumpo;
            pp->kernarg_address = (uint64_t)(uintptr_t)vk;
            pp->group_segment_size = 0;
            if (ph == 2) { pp->grid_x = c.wg; }   /* the bump is a single workgroup */
            if (prebuilt) {
                uint32_t hs =
                    ((uint32_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) << 16) | hdr;
                __atomic_store_n((uint32_t*)pp, hs, __ATOMIC_RELEASE);
            } else {
                publish(&c, pp, ix, hdr);
            }
        }
        if (prebuilt)
            hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)(base + np - 1));
        if (hsa_signal_wait_scacquire(tail, HSA_SIGNAL_CONDITION_EQ, 0,
                                      60ull * 1000 * 1000 * 1000, HSA_WAIT_STATE_ACTIVE) != 0) {
            printf("verify          CHAIN DID NOT DRAIN in 60 s -- inconclusive\n"); return 1;
        }
        uint32_t hostdbg[4] = {0,0,0,0};
        {   uint32_t* hb = (uint32_t*)alloc_shared(fine, 4096);
            hsa_signal_t rs; CHK(hsa_signal_create(1, 0, NULL, &rs), "read sig");
            CHK(hsa_amd_memory_async_copy(hb, g_cpu, dbg, g_agent, 16, 0, NULL, rs), "d2h dbg");
            hsa_signal_wait_scacquire(rs, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            hsa_signal_destroy(rs); memcpy(hostdbg, hb, 16); }
        const uint32_t* dbgv = hostdbg;
        printf("verify%s barrier-bit chain, PLAIN accesses, %u packets (%u pub/chk pairs)\n",
               prebuilt ? "-prebuild" : "         ", np, np / 3);
        printf("verify          version reached %u, words checked %u, STALE %u\n",
               dbgv[0], dbgv[2], dbgv[1]);
        printf("verify          -> the AQL header's agent-scope fences %s order memory across XCDs\n",
               dbgv[1] == 0 && dbgv[2] > 0 ? "DO" : "DO NOT");
        return 0;
    }
    if (!strcmp(only, "all") || !strcmp(only, "devenq")) {
        arm_devenq(&c, exe, fine, kpool, n > 500 ? 500 : n);
        fflush(stdout);
    }
    return 0;
}
