/* l2r_bw.c <bytes_per_thread> <mode> <secs> [reps]: per-core bandwidth / sustained-compute probe for the
 * L2-resident BF16 experiment. Threads pin one per CPU of L2R_CPUS (default: physical cores 2-31,34-63,66-95,
 * i.e. the isolated inference set), allocate their buffer after pinning (node-local first touch) and, with
 * L2R_HUGE=1, back it with THP so the buffer covers L2 sets uniformly.
 *
 * modes: 0 AVX-512 read sweep            (buffer bytes/s)
 *        1 AMX-BF16 weight stream        (B tiles from the buffer, TDPBF16PS vs a fixed A tile: M=16 GEMM)
 *        2 AVX-512 BF16 GEMV stream      (VDPBF16PS, one broadcast x pair per 64 B of weights: batch-1 GEMV)
 *        3 AVX-512 BF16 compute only     (VDPBF16PS on registers; for sustained frequency)
 *        4 AMX-BF16 compute only         (TDPBF16PS on resident tiles; for sustained frequency)
 * Prints one JSON line per rep: aggregate and per-core GB/s (or Gop/s for 3/4) with min/p5/p50/max. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAXT 192
static int NT, MODE, CPUS[MAXT];
static size_t SZ;
static double SECS;
static double rate[MAXT];
static pthread_barrier_t bar;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { uint8_t palette, start; uint8_t r0[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) tilecfg;

static int parse_cpus(const char* s, int* out) {
    int n = 0;
    while (*s && n < MAXT) {
        char* e; long a = strtol(s, &e, 10), b = a;
        if (*e == '-') b = strtol(e + 1, &e, 10);
        for (long c = a; c <= b && n < MAXT; c++) out[n++] = (int)c;
        s = *e == ',' ? e + 1 : e;
        if (e == s && *s) break;
    }
    return n;
}

static void* run(void* arg) {
    const int id = (int)(intptr_t)arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(CPUS[id], &s); sched_setaffinity(0, sizeof s, &s);
    const size_t al = (size_t)2 << 20, len = (SZ + al - 1) / al * al;
    uint8_t* b = mmap(NULL, len + al, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    b = (uint8_t*)(((uintptr_t)b + al - 1) & ~(al - 1));
    if (getenv("L2R_HUGE")) madvise(b, len, MADV_HUGEPAGE);
    for (size_t i = 0; i < SZ; i += 2) { const uint16_t v = 0x3f80 ^ (uint16_t)((i * 131 + id) & 0x7f); memcpy(b + i, &v, 2); }
    uint16_t a[16 * 32] __attribute__((aligned(64)));
    for (int i = 0; i < 16 * 32; i++) a[i] = 0x3f80 ^ (uint16_t)(i % 7);
    if (MODE == 1 || MODE == 4) {
        tilecfg c; memset(&c, 0, sizeof c); c.palette = 1;
        for (int t = 0; t < 8; t++) { c.colsb[t] = 64; c.rows[t] = 16; }
        _tile_loadconfig(&c);
        _tile_loadd(4, a, 64); _tile_loadd(5, b, 64); _tile_loadd(6, b + 1024, 64); _tile_loadd(7, b + 2048, 64);
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    }
    __m512 f0 = _mm512_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
    const __m512bh xa = (__m512bh)_mm512_set1_epi32(0x3f803f80);
    __m512i acc = _mm512_setzero_si512();
    pthread_barrier_wait(&bar);
    const double t0 = now();
    double t1 = t0, work = 0;
    while (t1 - t0 < SECS) {
        if (MODE == 0) {
            __m512i x0 = acc, x1 = acc, x2 = acc, x3 = acc;
            for (size_t o = 0; o < SZ; o += 256) {
                x0 = _mm512_add_epi64(x0, _mm512_load_si512(b + o));
                x1 = _mm512_add_epi64(x1, _mm512_load_si512(b + o + 64));
                x2 = _mm512_add_epi64(x2, _mm512_load_si512(b + o + 128));
                x3 = _mm512_add_epi64(x3, _mm512_load_si512(b + o + 192));
            }
            acc = _mm512_add_epi64(_mm512_add_epi64(x0, x1), _mm512_add_epi64(x2, x3));
            work += SZ;
        } else if (MODE == 1) {
            for (size_t o = 0; o + 4096 <= SZ; o += 4096) {
                _tile_loadd(5, b + o, 64);
                _tile_loadd(6, b + o + 1024, 64);
                _tile_dpbf16ps(0, 4, 5);
                _tile_dpbf16ps(1, 4, 6);
                _tile_loadd(7, b + o + 2048, 64);
                _tile_loadd(5, b + o + 3072, 64);
                _tile_dpbf16ps(2, 4, 7);
                _tile_dpbf16ps(3, 4, 5);
            }
            work += SZ;
        } else if (MODE == 2) {
            __m512 g[8] = {f0, f1, f2, f3, f0, f1, f2, f3};
            for (size_t o = 0; o < SZ; o += 512) {
#pragma GCC unroll 8
                for (int k = 0; k < 8; k++) g[k] = _mm512_dpbf16_ps(g[k], (__m512bh)_mm512_load_si512(b + o + 64 * k), xa);
            }
            f0 = _mm512_add_ps(_mm512_add_ps(g[0], g[1]), _mm512_add_ps(g[2], g[3]));
            f1 = _mm512_add_ps(_mm512_add_ps(g[4], g[5]), _mm512_add_ps(g[6], g[7]));
            work += SZ;
        } else if (MODE == 3) {
            const __m512bh w0 = (__m512bh)_mm512_load_si512(b), w1 = (__m512bh)_mm512_load_si512(b + 64);
            __m512 g[8] = {f0, f1, f2, f3, f0, f1, f2, f3};
            for (int it = 0; it < (1 << 20); it++) {
#pragma GCC unroll 8
                for (int k = 0; k < 8; k++) g[k] = _mm512_dpbf16_ps(g[k], k & 1 ? w1 : w0, xa);
            }
            for (int k = 0; k < 8; k++) f0 = _mm512_add_ps(f0, g[k]);
            work += 8.0 * (1 << 20) * 32 * 2; /* flops: 16 lanes x 2 pairs x mul+add */
        } else {
            for (int it = 0; it < (1 << 16); it++) {
                _tile_dpbf16ps(0, 4, 5); _tile_dpbf16ps(1, 4, 6); _tile_dpbf16ps(2, 4, 7); _tile_dpbf16ps(3, 4, 5);
            }
            work += 4.0 * (1 << 16) * 16 * 16 * 32 * 2;
        }
        t1 = now();
    }
    if (MODE == 1 || MODE == 4) { float o[256]; _tile_stored(0, o, 64); f0 = _mm512_set1_ps(o[3]); _tile_release(); }
    const float sink = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(f0, f1), _mm512_add_ps(f2, f3))) +
                       (float)_mm512_reduce_add_epi64(acc);
    static volatile float sinkv;
    sinkv = sink;
    rate[id] = work / (t1 - t0) / 1e9;
    munmap(b, len);
    return NULL;
}

static int cmpd(const void* x, const void* y) { double a = *(const double*)x, b = *(const double*)y; return a < b ? -1 : a > b; }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: l2r_bw <bytes_per_thread> <mode> <secs> [reps]\n"); return 2; }
    SZ = strtoull(argv[1], 0, 0); MODE = atoi(argv[2]); SECS = atof(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 1;
    const char* cl = getenv("L2R_CPUS");
    NT = parse_cpus(cl ? cl : "2-31,34-63,66-95", CPUS);
    SZ = SZ / 4096 * 4096;
    if ((MODE == 1 || MODE == 4) && syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XTILEDATA */)) { perror("amx perm"); return 1; }
    static const char* names[] = {"avx512-read", "amx-bf16-stream", "avx512-bf16-gemv", "avx512-bf16-compute", "amx-bf16-compute"};
    for (int r = 0; r < reps; r++) {
        pthread_barrier_init(&bar, NULL, NT);
        pthread_t th[MAXT];
        for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, run, (void*)(intptr_t)i);
        double tot = 0, v[MAXT];
        for (int i = 0; i < NT; i++) { pthread_join(th[i], NULL); tot += rate[i]; v[i] = rate[i]; }
        int imin = 0;
        for (int i = 1; i < NT; i++) if (rate[i] < rate[imin]) imin = i;
        qsort(v, NT, sizeof v[0], cmpd);
        printf("{\"mode\":\"%s\",\"rep\":%d,\"threads\":%d,\"bytes_per_thread\":%zu,\"huge\":%d,\"unit\":\"%s\","
               "\"total\":%.1f,\"per_core_mean\":%.2f,\"min\":%.2f,\"min_cpu\":%d,\"p5\":%.2f,\"p50\":%.2f,\"max\":%.2f}\n",
               names[MODE], r, NT, SZ, getenv("L2R_HUGE") != NULL, MODE >= 3 ? "Gflop/s" : "GB/s", tot, tot / NT, v[0],
               CPUS[imin], v[NT / 20], v[NT / 2], v[NT - 1]);
        fflush(stdout);
        pthread_barrier_destroy(&bar);
    }
    return 0;
}
