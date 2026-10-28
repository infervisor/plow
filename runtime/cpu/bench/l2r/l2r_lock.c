/* l2r_lock.c <weight_bytes_per_core> <secs> <scenario>: P1 L2 weight-residency probe.
 *
 * One thread per CPU of L2R_CPUS (default the 90 isolated inference cores). Each allocates a node-local
 * THP-backed BF16 weight slice and, with L2R_LOCK=1, pins it into its own L2 through /dev/pseudo_lock
 * (runtime/cpu/driver, level 2: filled in the lock CLOS, other fills restricted to the free ways);
 * otherwise it only warms the slice (normal cache). Every step streams the whole slice through the GEMV
 * kernel (L2R_KERNEL=amx: TDPBF16PS M=16 B-tile stream; avx: VDPBF16PS batch-1) and then runs the
 * scenario's interference:
 *   A  none
 *   B  activation broadcast + per-die reduction: write 256 B to this core's slot of a per-SNC-node array,
 *      read all slots of the node (32 x 256 B), and a 10 KiB shared hidden vector rewritten by one core
 *   C  KV streaming from DRAM: L2R_KV_STEP bytes (default 256 KiB) of a per-core L2R_KV_BYTES buffer
 *      (default 256 MiB), VDPBF16PS over it, cursor advancing; L2R_GAP_NS spins that long after each KV phase (C/D/E)
 *   D  LLC-resident KV: same, over a per-core L2R_LLC_KV buffer (default 2 MiB, 180 MiB total < L3) plus
 *      L2R_HK_THREADS (default 6) housekeeping threads on cpus 0,1,32,33,64,65 streaming DRAM
 *   E  C plus an attention-like dot over a resident 64 KiB q/score block (long run: pass secs 600/1800)
 * Weight-pass time is recorded per step (rdtsc); per core we report GB/s, pass p50/p99, first- vs
 * last-10% mean (drift), and with L2R_LOCK=1 the driver's measured lines by L1/L2 / L3 / DRAM class
 * before and after the run. One JSON line per core plus a summary line. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

struct pl_caps { uint32_t version, nodes, l2_cbm_full, l2_cbm_lock, l3_cbm_full, l3_cbm_lock; uint64_t l2_lock_bytes_per_core, l3_lock_bytes_per_node; };
struct pl_lock_req { uint64_t addr, len; int32_t cpu; uint32_t level, id, pad; };
struct pl_measure { uint32_t id, pad; uint64_t lines, l1_l2, l3, dram, p50, cal_l2, cal_l3, cal_dram; };
#define PL_IOC_CAPS _IOR('P', 10, struct pl_caps)
#define PL_IOC_LOCK _IOWR('P', 11, struct pl_lock_req)
#define PL_IOC_MEASURE _IOWR('P', 13, struct pl_measure)

#define MAXT 192
#define MAXSTEP (1 << 22)
static int NT, CPUS[MAXT], LOCK, AMX, PLFD = -1;
static char SCEN;
static size_t WB, KVB, KVSTEP, LLCKV;
static uint64_t GAP_CYC;
static double SECS;
static pthread_barrier_t bar;
static volatile int stop_hk;
static uint8_t* node_slots[8];
static uint16_t hidden[5120] __attribute__((aligned(64)));
static volatile uint64_t hidden_epoch;

typedef struct { uint8_t palette, start; uint8_t r0[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) tilecfg;
typedef struct {
    double gbs, p50_us, p99_us, max_us, first_us, last_us;
    uint64_t steps;
    struct pl_measure m0, m1;
    int locked;
} result;
static result res[MAXT];

static int parse_cpus(const char* s, int* out) {
    int n = 0;
    while (*s && n < MAXT) {
        char* e; long a = strtol(s, &e, 10), b = a;
        if (*e == '-') b = strtol(e + 1, &e, 10);
        for (long c = a; c <= b && n < MAXT; c++) out[n++] = (int)c;
        if (e == s) break;
        s = *e == ',' ? e + 1 : e;
    }
    return n;
}

static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); sched_setaffinity(0, sizeof s, &s); }

static uint8_t* huge_alloc(size_t len) {
    const size_t al = (size_t)2 << 20, l = (len + al - 1) / al * al;
    uint8_t* p = mmap(NULL, l + al, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    p = (uint8_t*)(((uintptr_t)p + al - 1) & ~(al - 1));
    madvise(p, l, MADV_HUGEPAGE);
    return p;
}

static double tsc_ghz;
static double calib(void) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a); uint64_t t0 = __rdtsc();
    do clock_gettime(CLOCK_MONOTONIC, &b); while ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec) < 2e8);
    return (__rdtsc() - t0) / ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec));
}

static inline __m512 gemv_avx(const uint8_t* w, size_t n, __m512 acc) {
    const __m512bh x = (__m512bh)_mm512_set1_epi32(0x3f803f80);
    __m512 g[8] = {acc, acc, acc, acc, acc, acc, acc, acc};
    for (size_t o = 0; o < n; o += 512) {
#pragma GCC unroll 8
        for (int k = 0; k < 8; k++) g[k] = _mm512_dpbf16_ps(g[k], (__m512bh)_mm512_load_si512(w + o + 64 * k), x);
    }
    for (int k = 1; k < 8; k++) g[0] = _mm512_add_ps(g[0], g[k]);
    return g[0];
}

static inline void gemv_amx(const uint8_t* w, size_t n) {
    for (size_t o = 0; o + 4096 <= n; o += 4096) {
        _tile_loadd(5, w + o, 64); _tile_loadd(6, w + o + 1024, 64);
        _tile_dpbf16ps(0, 4, 5); _tile_dpbf16ps(1, 4, 6);
        _tile_loadd(7, w + o + 2048, 64); _tile_loadd(5, w + o + 3072, 64);
        _tile_dpbf16ps(2, 4, 7); _tile_dpbf16ps(3, 4, 5);
    }
}

static int cmpu(const void* a, const void* b) { uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b; return x < y ? -1 : x > y; }

static void* hk_run(void* arg) {
    pin((int)(intptr_t)arg);
    const size_t n = (size_t)256 << 20;
    uint8_t* b = huge_alloc(n);
    memset(b, 1, n);
    __m512i acc = _mm512_setzero_si512();
    while (!stop_hk)
        for (size_t o = 0; o < n && !stop_hk; o += 64) acc = _mm512_add_epi64(acc, _mm512_load_si512(b + o));
    static volatile long sink; sink = _mm512_reduce_add_epi64(acc);
    return NULL;
}

static void* run(void* arg) {
    const int id = (int)(intptr_t)arg, cpu = CPUS[id], node = cpu / 32 % 3;
    pin(cpu);
    result* r = &res[id];
    uint8_t* w = huge_alloc(WB);
    for (size_t i = 0; i < WB; i += 2) { uint16_t v = 0x3f80 ^ (uint16_t)((i * 131 + id) & 0x7f); memcpy(w + i, &v, 2); }
    uint8_t* kv = NULL; size_t kvn = 0, cur = 0;
    if (SCEN == 'C' || SCEN == 'E') kvn = KVB;
    if (SCEN == 'D') kvn = LLCKV;
    if (kvn) { kv = huge_alloc(kvn); memset(kv, 0x3f, kvn); }
    uint8_t* qblk = huge_alloc(64 << 10); memset(qblk, 0x3f, 64 << 10);
    if (LOCK) {
        struct pl_lock_req q = {(uintptr_t)w, (WB + 4095) / 4096 * 4096, cpu, 2, 0, 0};
        if (ioctl(PLFD, PL_IOC_LOCK, &q)) { fprintf(stderr, "cpu %d LOCK: %s\n", cpu, strerror(errno)); }
        else {
            r->locked = 1;
            r->m0.id = q.id; ioctl(PLFD, PL_IOC_MEASURE, &r->m0);
            r->m1.id = q.id;
        }
    }
    uint16_t a[16 * 32] __attribute__((aligned(64)));
    for (int i = 0; i < 16 * 32; i++) a[i] = 0x3f80 ^ (uint16_t)(i % 7);
    if (AMX) {
        tilecfg c; memset(&c, 0, sizeof c); c.palette = 1;
        for (int t = 0; t < 8; t++) { c.colsb[t] = 64; c.rows[t] = 16; }
        _tile_loadconfig(&c); _tile_loadd(4, a, 64);
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    }
    __m512 acc = _mm512_setzero_ps();
    for (int warm = 0; warm < 3; warm++) { if (AMX) gemv_amx(w, WB); else acc = gemv_avx(w, WB, acc); }
    uint32_t* t = malloc(sizeof(uint32_t) * MAXSTEP);
    uint64_t steps = 0;
    pthread_barrier_wait(&bar);
    const uint64_t tend = __rdtsc() + (uint64_t)(SECS * tsc_ghz * 1e9);
    uint64_t wcyc = 0;
    while (__rdtsc() < tend && steps < MAXSTEP) {
        const uint64_t t0 = __rdtsc();
        if (AMX) gemv_amx(w, WB); else acc = gemv_avx(w, WB, acc);
        const uint64_t d = __rdtsc() - t0;
        wcyc += d;
        t[steps++] = (uint32_t)(d > 0xffffffffu ? 0xffffffffu : d);
        if (SCEN == 'B') {
            uint8_t* slots = node_slots[node];
            memcpy(slots + (size_t)(id % 32) * 256, w, 256);
            acc = gemv_avx(slots, 32 * 256, acc);
            if (id == 0) { hidden[steps & 4095] = (uint16_t)steps; hidden_epoch = steps; }
            acc = gemv_avx((const uint8_t*)hidden, sizeof hidden, acc);
        } else if (kv) {
            if (cur + KVSTEP > kvn) cur = 0;
            acc = gemv_avx(kv + cur, KVSTEP, acc);
            cur += KVSTEP;
            if (SCEN == 'E') acc = gemv_avx(qblk, 64 << 10, acc);
            if (GAP_CYC) { const uint64_t g = __rdtsc() + GAP_CYC; while (__rdtsc() < g) _mm_pause(); }
        }
    }
    if (AMX) { float o[256]; _tile_stored(0, o, 64); acc = _mm512_add_ps(acc, _mm512_set1_ps(o[1])); _tile_release(); }
    static volatile float sink; sink = _mm512_reduce_add_ps(acc);
    if (r->locked) ioctl(PLFD, PL_IOC_MEASURE, &r->m1);
    const double us = 1e-3 / tsc_ghz;
    r->steps = steps;
    r->gbs = steps ? (double)WB * steps / (wcyc / (tsc_ghz * 1e9)) / 1e9 : 0;
    const uint64_t k = steps / 10 ? steps / 10 : 1;
    double f = 0, l = 0;
    for (uint64_t i = 0; i < k && i < steps; i++) { f += t[i]; l += t[steps - 1 - i]; }
    r->first_us = f / k * us; r->last_us = l / k * us;
    qsort(t, steps, sizeof t[0], cmpu);
    if (steps) { r->p50_us = t[steps / 2] * us; r->p99_us = t[steps * 99 / 100] * us; r->max_us = t[steps - 1] * us; }
    free(t);
    return NULL;
}

static size_t envsz(const char* k, size_t d) { const char* v = getenv(k); return v ? strtoull(v, 0, 0) : d; }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: l2r_lock <weight_bytes_per_core> <secs> <A|B|C|D|E>\n"); return 2; }
    WB = strtoull(argv[1], 0, 0) / 4096 * 4096; SECS = atof(argv[2]); SCEN = argv[3][0];
    const char* cl = getenv("L2R_CPUS");
    NT = parse_cpus(cl ? cl : "2-31,34-63,66-95", CPUS);
    LOCK = getenv("L2R_LOCK") && atoi(getenv("L2R_LOCK"));
    AMX = !getenv("L2R_KERNEL") || !strcmp(getenv("L2R_KERNEL"), "amx");
    KVB = envsz("L2R_KV_BYTES", (size_t)256 << 20); KVSTEP = envsz("L2R_KV_STEP", 256 << 10) / 512 * 512;
    LLCKV = envsz("L2R_LLC_KV", (size_t)2 << 20);
    tsc_ghz = calib();
    GAP_CYC = (uint64_t)(envsz("L2R_GAP_NS", 0) * tsc_ghz);
    if (AMX && syscall(SYS_arch_prctl, 0x1023, 18)) { perror("amx perm"); return 1; }
    struct pl_caps caps = {0};
    if (LOCK) {
        PLFD = open("/dev/pseudo_lock", O_RDWR);
        if (PLFD < 0 || ioctl(PLFD, PL_IOC_CAPS, &caps)) { perror("/dev/pseudo_lock"); return 1; }
        if (WB > caps.l2_lock_bytes_per_core) fprintf(stderr, "warning: %zu B/core > lockable %lu\n", WB, (unsigned long)caps.l2_lock_bytes_per_core);
    }
    for (int n = 0; n < 3; n++) { node_slots[n] = huge_alloc(32 * 256); memset(node_slots[n], 0x3f, 32 * 256); }
    for (int i = 0; i < 5120; i++) hidden[i] = 0x3f80;
    pthread_t hk[8]; int nhk = 0;
    if (SCEN == 'D') {
        static const int hkc[] = {0, 1, 32, 33, 64, 65};
        nhk = (int)envsz("L2R_HK_THREADS", 6); if (nhk > 6) nhk = 6;
        for (int i = 0; i < nhk; i++) pthread_create(&hk[i], NULL, hk_run, (void*)(intptr_t)hkc[i]);
    }
    pthread_barrier_init(&bar, NULL, NT);
    pthread_t th[MAXT];
    for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, run, (void*)(intptr_t)i);
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    stop_hk = 1;
    for (int i = 0; i < nhk; i++) pthread_join(hk[i], NULL);
    double tot = 0, mn = 1e30, worst99 = 0, held0 = 0, held1 = 0, lines = 0; int wc = -1, mc = -1;
    for (int i = 0; i < NT; i++) {
        result* r = &res[i];
        printf("{\"cpu\":%d,\"gbs\":%.2f,\"p50_us\":%.3f,\"p99_us\":%.3f,\"max_us\":%.3f,\"first_us\":%.3f,\"last_us\":%.3f,\"steps\":%lu",
               CPUS[i], r->gbs, r->p50_us, r->p99_us, r->max_us, r->first_us, r->last_us, (unsigned long)r->steps);
        if (r->locked)
            printf(",\"m0\":[%lu,%lu,%lu,%lu],\"m1\":[%lu,%lu,%lu,%lu]", (unsigned long)r->m0.lines, (unsigned long)r->m0.l1_l2,
                   (unsigned long)r->m0.l3, (unsigned long)r->m0.dram, (unsigned long)r->m1.lines, (unsigned long)r->m1.l1_l2,
                   (unsigned long)r->m1.l3, (unsigned long)r->m1.dram);
        printf("}\n");
        tot += r->gbs;
        if (r->gbs < mn) mn = r->gbs, mc = CPUS[i];
        if (r->p99_us > worst99) worst99 = r->p99_us, wc = CPUS[i];
        if (r->locked) { held0 += r->m0.l1_l2; held1 += r->m1.l1_l2; lines += r->m1.lines; }
    }
    printf("{\"summary\":1,\"scenario\":\"%c\",\"lock\":%d,\"kernel\":\"%s\",\"threads\":%d,\"weight_bytes_per_core\":%zu,\"secs\":%.0f,"
           "\"l2_lock_ways_cbm\":\"0x%x\",\"lockable\":%lu,\"total_gbs\":%.1f,\"per_core_mean\":%.2f,\"min_gbs\":%.2f,\"min_cpu\":%d,"
           "\"worst_p99_us\":%.3f,\"worst_cpu\":%d,\"held_l2_before\":%.4f,\"held_l2_after\":%.4f}\n",
           SCEN, LOCK, AMX ? "amx" : "avx", NT, WB, SECS, caps.l2_cbm_lock, (unsigned long)caps.l2_lock_bytes_per_core, tot, tot / NT, mn, mc,
           worst99, wc, lines ? held0 / lines : -1.0, lines ? held1 / lines : -1.0);
    if (PLFD >= 0) close(PLFD);
    return 0;
}
