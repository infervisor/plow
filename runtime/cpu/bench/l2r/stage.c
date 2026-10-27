/* stage.c <threads> <iters> <mode>: one Gemma-4-E4B-shaped decoder layer at batch 1, INT8 weights
 * resident in the cores' L2 (W-stationary column split: each core owns N/T full-K rows per GEMV,
 * no reduction), 4 dependent GEMV phases (qkv, o, gate|up, down) separated by a spin barrier; every
 * phase reads the previous phase's whole output vector (broadcast through the mesh).
 * mode 0 = full layer, 1 = barriers + broadcast only (no GEMV), 2 = GEMV only (no barriers).
 * Prints us per layer. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NPH 4
static const int PK[NPH] = {2560, 2048, 2560, 10240};  /* input width per phase */
static const int PN[NPH] = {3072, 2560, 20480, 2560};  /* output width per phase */
static int NT, ITERS, MODE;
static uint8_t* vec[NPH + 1];                          /* phase inputs/outputs, shared */
static double tsum;
static unsigned gctr[256 * 16];

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* dissemination barrier: round r, core i flags core (i + 2^r) % NT; each flag on its own line */
#define NR 8
static _Alignas(64) struct { atomic_uint f; char pad[60]; } flag[256][NR];
static inline void barrier(unsigned* g) {
    const unsigned my = ++*g;
    const int id = (int)(g - gctr) / 16;
    for (int r = 0, d = 1; d < NT; r++, d <<= 1) {
        atomic_store_explicit(&flag[(id + d) % NT][r].f, my, memory_order_release);
        while (atomic_load_explicit(&flag[id][r].f, memory_order_acquire) < my) _mm_pause();
    }
}

static void* run(void* arg) {
    const int id = (int)(intptr_t)arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(id, &s); sched_setaffinity(0, sizeof s, &s);
    int8_t* w[NPH]; int n0[NPH], n1[NPH];
    size_t mine = 0;
    for (int p = 0; p < NPH; p++) {
        n0[p] = (int)((long)PN[p] * id / NT); n1[p] = (int)((long)PN[p] * (id + 1) / NT);
        const size_t b = (size_t)(n1[p] - n0[p]) * PK[p];
        w[p] = aligned_alloc(64, b + 64);
        for (size_t i = 0; i < b; i++) w[p][i] = (int8_t)((i * 37 + id) % 7 - 3);
        mine += b;
    }
    if (id == 0) printf("per-core weights %.2f MiB\n", mine / 1048576.0);
    unsigned* gp = &gctr[id * 16];
    barrier(gp);
    double t0 = 0;
    for (int it = -200; it < ITERS; it++) {
        if (it == 0) { barrier(gp); t0 = now(); }
        for (int p = 0; p < NPH; p++) {
            const uint8_t* x = vec[p];
            uint8_t* y = vec[p + 1];
            const int K = PK[p];
            if (MODE != 1) {
                int n = n0[p];
                for (; n + 4 <= n1[p]; n += 4) {
                    const int8_t* r = w[p] + (size_t)(n - n0[p]) * K;
                    __m512i a[8];
                    for (int j = 0; j < 8; j++) a[j] = _mm512_setzero_si512();
                    for (int k = 0; k < K; k += 128) {
                        const __m512i x0 = _mm512_load_si512(x + k), x1 = _mm512_load_si512(x + k + 64);
                        for (int j = 0; j < 4; j++) {
                            a[2 * j] = _mm512_dpbusd_epi32(a[2 * j], x0, _mm512_load_si512(r + (size_t)j * K + k));
                            a[2 * j + 1] = _mm512_dpbusd_epi32(a[2 * j + 1], x1, _mm512_load_si512(r + (size_t)j * K + k + 64));
                        }
                    }
                    for (int j = 0; j < 4; j++) y[n + j] = (uint8_t)(_mm512_reduce_add_epi32(_mm512_add_epi32(a[2 * j], a[2 * j + 1])) >> 12);
                }
                for (; n < n1[p]; n++) {
                    const int8_t* r = w[p] + (size_t)(n - n0[p]) * K;
                    __m512i a0 = _mm512_setzero_si512(), a1 = a0;
                    for (int k = 0; k < K; k += 128) {
                        a0 = _mm512_dpbusd_epi32(a0, _mm512_load_si512(x + k), _mm512_load_si512(r + k));
                        a1 = _mm512_dpbusd_epi32(a1, _mm512_load_si512(x + k + 64), _mm512_load_si512(r + k + 64));
                    }
                    y[n] = (uint8_t)(_mm512_reduce_add_epi32(_mm512_add_epi32(a0, a1)) >> 12);
                }
            } else {
                /* touch the whole input (the broadcast) and write this core's output slice */
                __m512i a = _mm512_setzero_si512();
                for (int k = 0; k < K; k += 64) a = _mm512_add_epi32(a, _mm512_load_si512(x + k));
                const uint8_t v = (uint8_t)_mm512_reduce_add_epi32(a);
                for (int n = n0[p]; n < n1[p]; n++) y[n] = v;
            }
            if (MODE != 2) barrier(gp);
        }
        if (id == 0) memcpy(vec[0], vec[NPH], 2560);   /* next layer's input */
        if (MODE != 2) barrier(gp);
    }
    if (id == 0) tsum = (now() - t0) / ITERS * 1e6;
    return NULL;
}

int main(int argc, char** argv) {
    NT = atoi(argv[1]); ITERS = atoi(argv[2]); MODE = atoi(argv[3]);
    for (int p = 0; p <= NPH; p++) { vec[p] = aligned_alloc(64, 20480 + 64); memset(vec[p], 1, 20480 + 64); }
    pthread_t th[256];
    for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, run, (void*)(intptr_t)i);
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    printf("threads %d mode %s: %.2f us/layer (%.0f layer/s)\n", NT,
           MODE == 0 ? "full" : MODE == 1 ? "sync+bcast" : "gemv-only", tsum, 1e6 / tsum);
    return 0;
}
