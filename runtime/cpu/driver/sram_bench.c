// sram_bench: exercise /dev/pseudo_lock the way plowrt uses it. Caller-owned, 2 MiB-aligned
// THP buffers bound to the owner's node are locked into each node's L3 and into a few cores'
// L2, measured by the driver, then re-measured after other cores stream far more than the
// cache. A lock that holds keeps its lines in the L1/L2 (L2 lock) or L3 (L3 lock) class.
//
//   cc -O2 -pthread sram_bench.c -o sram_bench && ./sram_bench [stream_seconds] [l2 cpus...]
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
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

struct pl_caps {
    uint32_t version, nodes, l2_cbm_full, l2_cbm_lock, l3_cbm_full, l3_cbm_lock;
    uint64_t l2_lock_bytes_per_core, l3_lock_bytes_per_node;
};
struct pl_lock_req { uint64_t addr, len; int32_t cpu; uint32_t level, id, pad; };
struct pl_measure { uint32_t id, pad; uint64_t lines, l1_l2, l3, dram, p50, cal_l2, cal_l3, cal_dram; };
#define PL_IOC_CAPS    _IOR('P', 10, struct pl_caps)
#define PL_IOC_LOCK    _IOWR('P', 11, struct pl_lock_req)
#define PL_IOC_MEASURE _IOWR('P', 13, struct pl_measure)
struct pl_clos_req { int32_t cpu; uint32_t on; };
#define PL_IOC_WORKER  _IOW('P', 15, struct pl_clos_req)

#define MPOL_BIND 2
#define HUGE (2UL << 20)

static int node_first_cpu(int node) {
    char path[96], buf[256];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
    FILE *f = fopen(path, "r");
    if (!f || !fgets(buf, sizeof buf, f)) return -1;
    fclose(f);
    return atoi(buf);
}

static int cpu_node(int cpu) {
    for (int n = 0; n < 64; n++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
        if (access(path, F_OK) == 0) return n;
    }
    return 0;
}

static void *alloc_on(int node, size_t len) {
    size_t map = (len + HUGE - 1) / HUGE * HUGE;
    char *p = mmap(NULL, map + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    p = (char *)(((uintptr_t)p + HUGE - 1) & ~(HUGE - 1));
    unsigned long mask = 1UL << node;
    if (syscall(SYS_mbind, p, map, MPOL_BIND, &mask, 64, 0)) perror("mbind");
    madvise(p, map, MADV_HUGEPAGE);
    memset(p, 1, map);
    return p;
}

static int lock(int fd, void *p, size_t len, int cpu, int level, struct pl_measure *m) {
    struct pl_lock_req r = { (uintptr_t)p, len, cpu, level, 0, 0 };
    if (ioctl(fd, PL_IOC_LOCK, &r)) {
        fprintf(stderr, "LOCK L%d cpu %d %zu B: %s\n", level, cpu, len, strerror(errno));
        return -1;
    }
    m->id = r.id;
    return ioctl(fd, PL_IOC_MEASURE, m);
}

static void show(const char *tag, int level, int cpu, const struct pl_measure *m) {
    uint64_t hit = level == 2 ? m->l1_l2 : m->l1_l2 + m->l3;
    printf("%-7s L%d cpu %3d: %6.2f%% held (l1/l2 %llu, l3 %llu, dram %llu of %llu) p50 %llu cyc; cal L2 %llu L3 %llu DRAM %llu\n",
           tag, level, cpu, 100.0 * hit / m->lines, (unsigned long long)m->l1_l2, (unsigned long long)m->l3,
           (unsigned long long)m->dram, (unsigned long long)m->lines, (unsigned long long)m->p50,
           (unsigned long long)m->cal_l2, (unsigned long long)m->cal_l3, (unsigned long long)m->cal_dram);
}

// READ_LOCKED=1: each streamer also re-reads its node's L3-locked range between 1 GiB passes
// (a worker reusing KV/activations while streaming weights). WORKER_CLOS=1: streamer cpus run in
// the driver's worker CLOS, whose L3 victims refill the locked ways.
static volatile int stop;
static char *l3_range[64];
static size_t l3_len;
static void *streamer(void *arg) {
    int cpu = (int)(intptr_t)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
    size_t len = 1UL << 30;
    volatile uint64_t *b = alloc_on(cpu_node(cpu), len), sum = 0;
    const char *env = getenv("READ_LOCKED");
    volatile const uint64_t *hot = env && *env == '1' ? (const uint64_t *)l3_range[cpu_node(cpu)] : NULL;
    while (!stop) {
        for (size_t i = 0; i < len / 8 && !stop; i += 8) {
            sum += b[i];
            if (hot && (i & 0xffff) == 0)
                for (size_t j = 0; j < l3_len / 8; j += 8) sum += hot[j];
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    int secs = argc > 1 ? atoi(argv[1]) : 10;
    int fd = open("/dev/pseudo_lock", O_RDWR);
    if (fd < 0) { perror("/dev/pseudo_lock"); return 1; }
    struct pl_caps c;
    if (ioctl(fd, PL_IOC_CAPS, &c)) { perror("CAPS"); return 1; }
    printf("pseudo_lock v%u: %u nodes, L2 lock %#x %llu KiB/core, L3 lock %#x %llu KiB/node\n", c.version, c.nodes,
           c.l2_cbm_lock, (unsigned long long)c.l2_lock_bytes_per_core >> 10, c.l3_cbm_lock,
           (unsigned long long)c.l3_lock_bytes_per_node >> 10);
    enum { MAXR = 64 };
    struct { int level, cpu; struct pl_measure m; } r[MAXR];
    int n = 0;
    const char *l3mb = getenv("L3_MB");
    size_t l3 = l3mb ? (size_t)atoi(l3mb) << 20 : c.l3_lock_bytes_per_node / 4096 * 4096;
    l3_len = l3;
    for (int node = 0; node < (int)c.nodes && c.l3_cbm_lock && n < MAXR; node++) {
        int cpu = node_first_cpu(node);
        void *p = alloc_on(node, l3);
        if (p && lock(fd, p, l3, cpu, 3, &r[n].m) == 0) { r[n].level = 3; r[n].cpu = cpu; show("locked", 3, cpu, &r[n].m); n++; l3_range[node] = p; }
    }
    size_t l2 = c.l2_lock_bytes_per_core / 4096 * 4096;
    int l2cpus[16] = { 0, 40, 80 }, nl2 = 3;
    if (argc > 2) for (nl2 = 0; nl2 + 2 < argc && nl2 < 16; nl2++) l2cpus[nl2] = atoi(argv[nl2 + 2]);
    for (int k = 0; k < nl2 && c.l2_cbm_lock && n < MAXR; k++) {
        void *p = alloc_on(cpu_node(l2cpus[k]), l2);
        if (p && lock(fd, p, l2, l2cpus[k], 2, &r[n].m) == 0) { r[n].level = 2; r[n].cpu = l2cpus[k]; show("locked", 2, l2cpus[k], &r[n].m); n++; }
    }
    // Pressure: 32 threads stream 1 GiB each on cores the locks do not own (SMT siblings excluded).
    pthread_t th[32];
    int nt = 0;
    for (int cpu = 1; cpu < 96 && nt < 32; cpu += 3) {
        int owned = 0;
        for (int k = 0; k < nl2; k++) owned |= cpu == l2cpus[k];
        if (owned) continue;
        const char *wc = getenv("WORKER_CLOS");
        if (wc && *wc == '1') {
            struct pl_clos_req q = { cpu, 1 };
            if (ioctl(fd, PL_IOC_WORKER, &q)) perror("WORKER");
        }
        pthread_create(&th[nt++], NULL, streamer, (void *)(intptr_t)cpu);
    }
    printf("streaming %d x 1 GiB for %d s...\n", nt, secs);
    sleep(secs);
    for (int i = 0; i < n; i++) {
        struct pl_measure m = { .id = r[i].m.id };
        if (ioctl(fd, PL_IOC_MEASURE, &m) == 0) show("stress", r[i].level, r[i].cpu, &m);
    }
    stop = 1;
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    sleep(2);
    for (int i = 0; i < n; i++) {
        struct pl_measure m = { .id = r[i].m.id };
        if (ioctl(fd, PL_IOC_MEASURE, &m) == 0) show("idle", r[i].level, r[i].cpu, &m);
    }
    return 0;
}
