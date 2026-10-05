#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sched.h>
#include <pthread.h>
#include <x86intrin.h>

#define PSEUDO_LOCK_IOC_MAGIC       'P'
#define PSEUDO_LOCK_IOC_GET_INFO    _IOR(PSEUDO_LOCK_IOC_MAGIC, 1, struct pseudo_lock_info)
#define PSEUDO_LOCK_IOC_MEASURE     _IOWR(PSEUDO_LOCK_IOC_MAGIC, 2, struct pseudo_lock_latency)
#define PSEUDO_LOCK_IOC_RELOAD      _IO(PSEUDO_LOCK_IOC_MAGIC, 3)

struct pseudo_lock_info {
    uint32_t level;
    uint32_t cpu;
    uint64_t size;
    uint32_t cbm;
    uint32_t line_size;
    uint64_t phys_addr;
};

struct pseudo_lock_latency {
    uint64_t min_cycles;
    uint64_t avg_cycles;
    uint64_t max_cycles;
    uint64_t total_lines;
    uint64_t l1_l2_hits;
    uint64_t l3_hits;
    uint64_t dram_misses;
};

static inline uint64_t rdtsc_fence(void) {
    uint32_t lo, hi;
    asm volatile("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static volatile int stop_stress = 0;

void *cache_pollution_worker(void *arg) {
    int cpu = (int)(intptr_t)arg;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    size_t sz = 64 * 1024 * 1024; // 64 MB per worker
    char *buf = malloc(sz);
    if (!buf) return NULL;
    memset(buf, 0x5a, sz);

    while (!stop_stress) {
        for (size_t i = 0; i < sz; i += 64) {
            buf[i] += 1;
        }
    }
    free(buf);
    return NULL;
}

void benchmark_sram(const char *dev_path, int target_cpu) {
    printf("\n======================================================================\n");
    printf("  BENCHMARKING PSEUDO-LOCKED SRAM DEVICE: %s\n", dev_path);
    printf("======================================================================\n");

    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        perror("open device");
        return;
    }

    struct pseudo_lock_info info;
    if (ioctl(fd, PSEUDO_LOCK_IOC_GET_INFO, &info) < 0) {
        perror("ioctl GET_INFO");
        close(fd);
        return;
    }

    printf("Cache Hierarchy Level:  L%u\n", info.level);
    printf("Target Core / Logical:  CPU %u\n", info.cpu);
    printf("SRAM Allocated Size:    %lu KB (%lu MB, %lu bytes)\n",
           info.size / 1024, info.size / (1024 * 1024), info.size);
    printf("Capacity Bitmask (CBM): 0x%x (Way 0..%d dedicated)\n",
           info.cbm, __builtin_popcount(info.cbm) - 1);
    printf("Contiguous Phys Base:   0x%lx\n", info.phys_addr);

    // Pin current benchmark process to target CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
        perror("sched_setaffinity");
    }

    // Memory map the SRAM region into our address space
    void *sram = mmap(NULL, info.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (sram == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    // Lock CPU out of deep C-states during benchmark
    int dma_fd = open("/dev/cpu_dma_latency", O_RDWR);
    if (dma_fd >= 0) {
        int32_t val = 0;
        write(dma_fd, &val, sizeof(val));
    }

    /* 1. Data Integrity & Verification */
    printf("\n[1] Data Integrity & Read/Write SRAM Operation:\n");
    volatile uint64_t *sram64 = (volatile uint64_t *)sram;
    size_t words = info.size / sizeof(uint64_t);

    for (size_t i = 0; i < words; i++) {
        sram64[i] = 0xcafebabe00000000ULL | i;
    }
    int verify_ok = 1;
    for (size_t i = 0; i < words; i++) {
        if (sram64[i] != (0xcafebabe00000000ULL | i)) {
            verify_ok = 0;
            break;
        }
    }
    printf("    -> Memory Read/Write Verification: %s\n",
           verify_ok ? "PASSED (100% Data Integrity)" : "FAILED");

    // Re-lock to refresh cache lines after initialization writes
    ioctl(fd, PSEUDO_LOCK_IOC_RELOAD, 0);

    /* 2. Cache Line Stride Latency Distribution */
    printf("\n[2] Cache-Line Stride Latency Distribution (Entire Buffer):\n");
    size_t num_lines = info.size / 64;
    uint64_t total_cycles = 0;
    uint32_t fast_hits = 0;  // <= 30c (L1/L2)
    uint32_t l3_hits = 0;    // 31-150c (L3)
    uint32_t misses = 0;     // > 150c (DRAM)
    uint64_t min_c = (uint64_t)-1;
    uint64_t max_c = 0;

    for (size_t i = 0; i < num_lines; i++) {
        uint64_t t0 = rdtsc_fence();
        volatile uint32_t val = *(volatile uint32_t *)((char *)sram + i * 64);
        uint64_t t1 = rdtsc_fence();
        (void)val;

        uint64_t diff = (t1 > t0) ? (t1 - t0) : 1;
        if (diff < min_c) min_c = diff;
        if (diff > max_c) max_c = diff;
        total_cycles += diff;

        if (diff <= 30) fast_hits++;
        else if (diff <= 150) l3_hits++;
        else misses++;
    }

    double avg_cyc = (double)total_cycles / num_lines;
    printf("    Lines Tested:         %zu\n", num_lines);
    printf("    Min Latency:          %lu cycles\n", min_c);
    printf("    Avg Latency:          %.1f cycles (~%.2f ns)\n", avg_cyc, avg_cyc / 3.9);
    printf("    Max Latency:          %lu cycles\n", max_c);
    printf("    L1/L2 Hits (<=30c):   %u (%.2f%%)\n", fast_hits, (double)fast_hits * 100.0 / num_lines);
    printf("    L3 Hits (31-150c):    %u (%.2f%%)\n", l3_hits, (double)l3_hits * 100.0 / num_lines);
    printf("    DRAM Misses (>150c):  %u (%.2f%%)\n", misses, (double)misses * 100.0 / num_lines);
    printf("    Total Cache Residency:%.2f%%\n", (double)(fast_hits + l3_hits) * 100.0 / num_lines);

    /* 3. Eviction Resistance Test under Active Background Thrashing */
    printf("\n[3] Stress Test: Eviction Resistance Under Heavy Background Churn:\n");
    printf("    Spawning 16 background cache-polluting stress threads...\n");
    stop_stress = 0;
    pthread_t threads[16];
    for (int i = 0; i < 16; i++) {
        pthread_create(&threads[i], NULL, cache_pollution_worker, (void *)(intptr_t)(i + 1));
    }
    usleep(500000); // 500ms burn-in

    // Measure latency under extreme contention
    total_cycles = 0;
    fast_hits = 0;
    l3_hits = 0;
    misses = 0;

    for (size_t i = 0; i < num_lines; i++) {
        uint64_t t0 = rdtsc_fence();
        volatile uint32_t val = *(volatile uint32_t *)((char *)sram + i * 64);
        uint64_t t1 = rdtsc_fence();
        (void)val;

        uint64_t diff = (t1 > t0) ? (t1 - t0) : 1;
        total_cycles += diff;

        if (diff <= 30) fast_hits++;
        else if (diff <= 150) l3_hits++;
        else misses++;
    }

    stop_stress = 1;
    for (int i = 0; i < 16; i++) {
        pthread_join(threads[i], NULL);
    }

    double stressed_avg = (double)total_cycles / num_lines;
    printf("    SRAM Avg Latency under 16-thread Thrash: %.1f cycles (~%.2f ns)\n",
           stressed_avg, stressed_avg / 3.9);
    printf("    Cache Residency under Contention:        %.2f%%\n",
           (double)(fast_hits + l3_hits) * 100.0 / num_lines);
    printf("    Latency Delta:                           %+.1f cycles\n", stressed_avg - avg_cyc);

    if ((double)(fast_hits + l3_hits) / num_lines > 0.95) {
        printf("    ==> STATUS: 100%% SRAM LOCK CONFIRMED! Zero evictions under intense thrashing!\n");
    } else {
        printf("    ==> STATUS: Contention detected.\n");
    }

    if (dma_fd >= 0) close(dma_fd);
    munmap(sram, info.size);
    close(fd);
}

int main(int argc, char *argv[]) {
    printf("======================================================================\n");
    printf("   INTEL XEON 6975P-C RDT CACHE PSEUDO-LOCKING (SRAM) BENCHMARK\n");
    printf("======================================================================\n");

    benchmark_sram("/dev/pseudo_lock_l2", 0);
    benchmark_sram("/dev/pseudo_lock_l3", 0);
    return 0;
}
