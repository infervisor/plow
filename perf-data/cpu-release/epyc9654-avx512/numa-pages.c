#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

int main(void) {
    size_t size = 2ul << 30, huge = 2ul << 20;
    for (int trial = 0; trial < 4; ++trial) {
        int advice = trial % 2 ? MADV_NOHUGEPAGE : MADV_HUGEPAGE;
        void *raw = mmap(0, size + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { perror("mmap"); return 1; }
        uintptr_t addr = ((uintptr_t)raw + huge - 1) & ~(huge - 1);
        size_t prefix = addr - (uintptr_t)raw;
        if (prefix) munmap(raw, prefix);
        munmap((void *)(addr + size), huge - prefix);
        void *ptr = (void *)addr;
        unsigned long mask = 255;
        if (syscall(SYS_mbind, ptr, size, MPOL_INTERLEAVE, &mask, 65ul, 0ul)) { perror("mbind"); return 1; }
        if (madvise(ptr, size, advice)) { perror("madvise"); return 1; }
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        memset(ptr, 1, size);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("trial=%d huge=%d seconds=%.3f ", trial, advice == MADV_HUGEPAGE,
               end.tv_sec-start.tv_sec + 1e-9*(end.tv_nsec-start.tv_nsec));
        FILE *f = fopen("/proc/self/numa_maps", "r");
        char *line = 0; size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            unsigned long a;
            if (sscanf(line, "%lx", &a) == 1 && a == addr) { fputs(line, stdout); break; }
        }
        free(line); fclose(f); fflush(stdout);
        munmap(ptr, size);
    }
}
