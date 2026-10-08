/* sync.c: synchronization primitives for an all-core exchange point on Xeon 6 (SNC3, 96 cores).
 * Part A: one-way core-to-core latency (ping-pong / 2): spin+pause vs UMWAIT (C0.1), +/- CLDEMOTE.
 * Part B: one exchange of a 2,560-byte vector (each core contributes a slice, then every core reads it all):
 *   diss       dissemination barrier, then read the vector
 *   hier       per-SNC-node arrival flags -> node leader -> 3 leaders exchange -> per-node gate (homed on the node)
 *   hier-umw   same, members wait on the gate with UMWAIT
 *   fid        flag in data: each core writes one line [epoch | payload]; readers poll all 96 lines
 *   fid-cld    same + CLDEMOTE after the write
 * Buffers alternate by epoch parity, so a phase never overwrites lines still being read. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NT
#define NT 96
#endif
#define PER 32
#define VB 2560
typedef struct { _Alignas(64) volatile uint64_t v[8]; } line_t;

static line_t* arrive[3];          /* per node: PER arrival lines, page first-touched on the node */
static line_t* nflag[3];           /* per node: node arrival flag */
static line_t* gate[3];            /* per node: release gate */
static line_t fid[2][NT];
static _Alignas(64) volatile uint8_t vec[2][VB];
static line_t dis[NT][8];
static line_t pp[2];
static int MODE, ITERS;
static double res;
static pthread_barrier_t bar;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void pin(int c) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s); sched_setaffinity(0, sizeof s, &s); }

static inline void wait_ge(volatile uint64_t* p, uint64_t e, int umw) {
    if (!umw) { while (*p < e) _mm_pause(); return; }
    while (*p < e) {
        _umonitor((void*)p);
        if (*p >= e) break;
        _umwait(1, __rdtsc() + 200000);
    }
}

static void exchange(int id, uint64_t e) {
    const int node = id / PER, k = id % PER, par = (int)(e & 1);
    switch (MODE) {
    case 0: { /* diss */
        const int lo = VB * id / NT, hi = VB * (id + 1) / NT;
        for (int i = lo; i < hi; i++) vec[par][i] = (uint8_t)e;
        for (int r = 0, d = 1; d < NT; r++, d <<= 1) {
            dis[(id + d) % NT][r].v[0] = e;
            wait_ge(&dis[id][r].v[0], e, 0);
        }
        break;
    }
    case 1: case 2: { /* hier, hier-umw */
        const int lo = VB * id / NT, hi = VB * (id + 1) / NT;
        for (int i = lo; i < hi; i++) vec[par][i] = (uint8_t)e;
        if (k) {
            arrive[node][k].v[0] = e;
            wait_ge(&gate[node]->v[0], e, MODE == 2);
        } else {
            for (int j = 1; j < PER; j++) wait_ge(&arrive[node][j].v[0], e, 0);
            nflag[node]->v[0] = e;
            for (int n = 0; n < 3; n++) if (n != node) wait_ge(&nflag[n]->v[0], e, 0);
            gate[node]->v[0] = e;
        }
        break;
    }
    case 3: case 4: { /* fid, fid-cld */
        line_t* l = &fid[par][id];
        for (int i = 1; i < 8; i++) l->v[i] = e + i;
        l->v[0] = e;
        if (MODE == 4) _cldemote((const void*)l);
        uint64_t s = 0;
        for (int j = 0; j < NT; j++) { wait_ge(&fid[par][j].v[0], e, 0); s += fid[par][j].v[1]; }
        if (s == 1) res += 0;
        return;
    }
    }
    uint64_t s = 0; /* read the whole vector */
    for (int i = 0; i < VB; i += 64) s += vec[par][i];
    if (s == 1) res += 0;
}

static void* run(void* a) {
    const int id = (int)(intptr_t)a;
    pin(id);
    if (id % PER == 0) { /* first-touch the node's sync page on the node */
        const int n = id / PER;
        uint8_t* pg = aligned_alloc(4096, 4096 * 2);
        memset(pg, 0, 4096 * 2);
        arrive[n] = (line_t*)pg; nflag[n] = (line_t*)(pg + 4096); gate[n] = (line_t*)(pg + 4096 + 64);
    }
    pthread_barrier_wait(&bar);
    uint64_t e = 1;
    for (int i = 0; i < 2000; i++) exchange(id, e++);
    pthread_barrier_wait(&bar);
    const double t0 = now();
    for (int i = 0; i < ITERS; i++) exchange(id, e++);
    const double t = now() - t0;
    if (id == 0) res = t / ITERS * 1e6;
    return NULL;
}

static void* pong(void* a) {
    const intptr_t* arg = a;
    pin((int)arg[0]);
    const int umw = (int)arg[1] & 1, cld = (int)arg[1] & 2;
    for (uint64_t i = 1; i <= (uint64_t)ITERS + 2000; i++) {
        wait_ge(&pp[0].v[0], i, umw);
        pp[1].v[0] = i;
        if (cld) _cldemote((const void*)&pp[1]);
    }
    return NULL;
}

static double pingpong(int other, int flags) {
    pp[0].v[0] = pp[1].v[0] = 0;
    intptr_t arg[2] = {other, flags};
    pthread_t th; pthread_create(&th, NULL, pong, arg);
    pin(0);
    const int umw = flags & 1, cld = flags & 2;
    double t0 = 0;
    for (uint64_t i = 1; i <= (uint64_t)ITERS + 2000; i++) {
        if (i == 2001) t0 = now();
        pp[0].v[0] = i;
        if (cld) _cldemote((const void*)&pp[0]);
        wait_ge(&pp[1].v[0], i, umw);
    }
    const double t = now() - t0;
    pthread_join(th, NULL);
    return t / ITERS / 2 * 1e9;
}

int main(void) {
    ITERS = 20000;
    printf("Part A: one-way latency core 0 -> core N (ns)\n");
    const char* fl[] = {"spin", "umwait", "spin+cldemote", "umwait+cldemote"};
    const int others[] = {1, 31, 32, 64};
    for (int o = 0; o < 4; o++) {
        printf("  core %2d (node %d):", others[o], others[o] / PER);
        for (int f = 0; f < 4; f++) printf("  %s %.0f", fl[f], pingpong(others[o], f));
        printf("\n");
    }
    printf("Part B: one 96-core exchange of a 2,560-byte vector (us)\n");
    const char* nm[] = {"diss", "hier", "hier-umw", "fid", "fid-cld"};
    for (MODE = 0; MODE < 5; MODE++) {
        if (NT < 96 && (MODE == 1 || MODE == 2)) continue;
        pthread_barrier_init(&bar, NULL, NT);
        pthread_t th[NT];
        for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, run, (void*)(intptr_t)i);
        for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
        pthread_barrier_destroy(&bar);
        printf("  %-9s %.2f us\n", nm[MODE], res);
    }
    return 0;
}
