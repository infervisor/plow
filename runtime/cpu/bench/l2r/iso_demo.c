/* iso_demo.c <worker-cpu> [jobs]: how a program uses an isolated core.
 * main (run it on a housekeeping core: taskset -c 0 ./iso_demo 5) pins one worker thread to <worker-cpu>.
 * The two talk through one cache line each way (epoch + payload); the worker never makes a syscall in
 * its loop. At the end the worker reports its CPU and how often the kernel took the core away. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

typedef struct { _Alignas(64) volatile uint64_t epoch, arg, out, pad[5]; } mbox_t;
static mbox_t to_worker, from_worker;
static long JOBS = 1000000;
static int WCPU;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void* worker(void* unused) {
    (void)unused;
    /* 1. pin before touching memory, so first-touch pages land on this core's NUMA node */
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(WCPU, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s)) { perror("affinity"); exit(1); }
    /* 2. private data for this core, sized to stay in its L2 (1 MiB) */
    const size_t n = 1 << 20;
    uint8_t* w = aligned_alloc(64, n);
    for (size_t i = 0; i < n; i++) w[i] = (uint8_t)(i * 7);
    struct rusage r0; getrusage(RUSAGE_THREAD, &r0);
    /* 3. the loop: spin on the input line, compute, publish; no syscalls, no allocation */
    for (uint64_t e = 1;; e++) {
        while (to_worker.epoch < e) _mm_pause();
        const uint64_t a = to_worker.arg;
        if (a == UINT64_MAX) break;                     /* stop command */
        __m512i acc = _mm512_setzero_si512();
        const uint8_t* p = w + (a % 16) * 4096;         /* touch 4 KB of the L2-resident slice */
        for (int k = 0; k < 4096; k += 64) acc = _mm512_add_epi64(acc, _mm512_load_si512(p + k));
        from_worker.out = (uint64_t)_mm512_reduce_add_epi64(acc) + a;
        from_worker.epoch = e;                          /* payload first, epoch last (x86 keeps store order) */
    }
    struct rusage r1; getrusage(RUSAGE_THREAD, &r1);
    printf("worker ran on cpu %d (asked for %d)\n", sched_getcpu(), WCPU);
    printf("kernel preemptions during the run: involuntary %ld, voluntary %ld\n",
           r1.ru_nivcsw - r0.ru_nivcsw, r1.ru_nvcsw - r0.ru_nvcsw);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <worker-cpu> [jobs]\n", argv[0]); return 1; }
    WCPU = atoi(argv[1]);
    if (argc > 2) JOBS = atol(argv[2]);
    mlockall(MCL_CURRENT | MCL_FUTURE);                 /* no page faults once running */
    pthread_t th; pthread_create(&th, NULL, worker, NULL);
    printf("main (control) on cpu %d\n", sched_getcpu());
    double t0 = 0, worst = 0;
    uint64_t check = 0;
    for (long e = 1; e <= JOBS + 1000; e++) {
        if (e == 1001) t0 = now();
        const double ts = now();
        to_worker.arg = (uint64_t)e;
        to_worker.epoch = (uint64_t)e;
        while (from_worker.epoch < (uint64_t)e) _mm_pause();
        check += from_worker.out;
        const double dt = now() - ts;
        if (e > 1000 && dt > worst) worst = dt;
    }
    const double t = now() - t0;
    to_worker.arg = UINT64_MAX;
    to_worker.epoch = (uint64_t)JOBS + 1001;
    pthread_join(th, NULL);
    printf("%ld jobs: mean round trip %.0f ns, worst %.1f us (checksum %llu)\n", JOBS, t / JOBS * 1e9, worst * 1e6,
           (unsigned long long)(check & 0xffff));
    return 0;
}
