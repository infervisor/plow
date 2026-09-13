/* Tier-2 sweep of the pinned glm_lt Tensile kernels through plow's HSA backend.
 *
 *   sweep OBJECT KERNELS_TXT CASE M N K XFILE WFILE TOPK
 *
 * KERNELS_TXT: "name mt_i mt_j" per line. XFILE holds >= M rows of K bf16; WFILE is N x K bf16.
 * Pass 1 checks every kernel (sampled rows vs FP64, finite, guard) and times it warm at one
 * mapping; pass 2 re-checks the TOPK fastest at every mapping and times warm and rotating
 * (operand sets spanning > 768 MiB, so MALL/L2 do not carry weights between launches).
 * Output: one JSON object per line. Exit 3 on a hung kernel, 1 on any setup failure. */
#include "hsa_backend.h"
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define S 64
#define GUARD 512
#define MAXK 1024

static void die(const char* what) {
    fprintf(stderr, "%s: %s\n", what, plow_hsa_last_error());
    exit(1);
}
#define CHECK(x) do { if (x) die(#x); } while (0)
static const char* current = "";
static void hung(int sig) {
    (void)sig;
    fprintf(stderr, "kernel hung: %s\n", current);
    _exit(3);
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double bf(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static void* slurp(const char* path, size_t want, size_t* got) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f);
    rewind(f);
    if (want && n < want) { fprintf(stderr, "%s: %zu < %zu bytes\n", path, n, want); exit(1); }
    if (want) n = want;
    void* p = malloc(n);
    if (!p || fread(p, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
    if (got) *got = n;
    return p;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

typedef struct { char name[1024]; unsigned mi, mj; plow_hsa_kernel k; double warm; int ok; } Kern;

static plow_hsa* h;
static unsigned M, N, K, R;
static void **dx, **dw, **dout;
static uint16_t *hpoison, *hrows, *hguard;
static unsigned rows[S];
static double* ref;
static size_t ob;

static void args_for(uint8_t a[160], const Kern* k, unsigned set, uint32_t info1) {
    const uint32_t grid = ((N + k->mi - 1) / k->mi) * ((M + k->mj - 1) / k->mj);
    memset(a, 0, 160);
    const uint32_t dims[8] = {1, 1, info1, grid, N, M, 1, K};
    memcpy(a, dims, 32);
    const uint64_t out = (uint64_t)dout[set], w = (uint64_t)dw[set], x = (uint64_t)dx[set];
    const uint64_t ptr[4] = {out, out, w, x};
    memcpy(a + 32, ptr, 32);
    const uint32_t strides[8] = {N, N * M, N, N * M, K, K * N, K, K * M};
    memcpy(a + 64, strides, 32);
    const float alpha = 1.0f;
    memcpy(a + 96, &alpha, 4);
    memcpy(a + 140, &out, 8);
}
static void launch(const Kern* k, const uint8_t a[160]) {
    const uint32_t grid = ((N + k->mi - 1) / k->mi) * ((M + k->mj - 1) / k->mj);
    CHECK(plow_hsa_launch(h, 0, &k->k, grid * 256, 1, 1, 256, 1, 1, 0, a, 160));
}
static void drain(void) {
    alarm(10);
    CHECK(plow_hsa_wait(h, 0));
    alarm(0);
}
/* Sampled-row check on set 0. Returns 1 when finite, guard intact and rel-L2 < 5e-3. */
static int check(const Kern* k, uint32_t info1, double* rel, double* worst) {
    uint8_t a[160];
    CHECK(plow_hsa_copy_h2d(h, 0, dout[0], hpoison, ob + GUARD));
    args_for(a, k, 0, info1);
    launch(k, a);
    drain();
    for (int s = 0; s < S; s++)
        CHECK(plow_hsa_copy_d2h(h, 0, hrows + (size_t)s * N, (uint8_t*)dout[0] + (size_t)rows[s] * N * 2,
                                (size_t)N * 2));
    CHECK(plow_hsa_copy_d2h(h, 0, hguard, (uint8_t*)dout[0] + ob, GUARD));
    int ok = !memcmp(hguard, hpoison + ob / 2, GUARD);
    double num = 0, den = 0, rms = 0;
    *worst = 0;
    for (size_t i = 0; i < (size_t)S * N; i++) rms += ref[i] * ref[i];
    rms = sqrt(rms / ((double)S * N));
    for (size_t i = 0; i < (size_t)S * N; i++) {
        const double y = bf(hrows[i]), d = y - ref[i];
        if (!isfinite(y)) ok = 0;
        num += d * d;
        den += ref[i] * ref[i];
        if (fabs(d) / rms > *worst) *worst = fabs(d) / rms;
    }
    *rel = sqrt(num / den);
    return ok && *rel < 5e-3;
}
static double timed(const Kern* k, uint32_t info1, int rotate, int samples, int reps) {
    uint8_t a[64][160];
    const unsigned sets = rotate ? R : 1;
    for (unsigned s = 0; s < sets; s++) args_for(a[s], k, s, info1);
    for (int i = 0; i < 2; i++) launch(k, a[0]);
    drain();
    double v[16];
    for (int s = 0; s < samples; s++) {
        const double t0 = now();
        for (int r = 0; r < reps; r++) launch(k, a[(unsigned)(s * reps + r) % sets]);
        drain();
        v[s] = (now() - t0) * 1e6 / reps;
    }
    qsort(v, samples, sizeof(double), cmpd);
    return v[samples / 2];
}

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: sweep OBJECT KERNELS_TXT CASE M N K XFILE WFILE TOPK\n");
        return 2;
    }
    const char* label = argv[3];
    M = atoi(argv[4]); N = atoi(argv[5]); K = atoi(argv[6]);
    const int topk = atoi(argv[9]);
    signal(SIGALRM, hung);
    static Kern ks[MAXK];
    int nk = 0;
    FILE* f = fopen(argv[2], "r");
    if (!f) { perror(argv[2]); return 1; }
    while (nk < MAXK && fscanf(f, "%1023s %u %u", ks[nk].name, &ks[nk].mi, &ks[nk].mj) == 3) nk++;
    fclose(f);
    size_t eb;
    void* elf = slurp(argv[1], 0, &eb);
    h = plow_hsa_init();
    if (!h) die("init");
    char arch[64];
    uint32_t cus, lds;
    CHECK(plow_hsa_device_info(h, 0, arch, &cus, &lds));
    if (strcmp(arch, "gfx942") || cus != 304) { fprintf(stderr, "not a gfx942/304 agent\n"); return 1; }
    CHECK(plow_hsa_load_code_object(h, 0, elf, eb));
    free(elf);
    for (int i = 0; i < nk; i++) {
        CHECK(plow_hsa_get_kernel(h, 0, ks[i].name, &ks[i].k));
        if ((ks[i].k.kernarg_size && ks[i].k.kernarg_size != 160) || ks[i].k.private_segment_size) {
            fprintf(stderr, "ABI mismatch %s\n", ks[i].name);
            return 1;
        }
        ks[i].k.kernarg_size = 160;
        ks[i].k.kernarg_explicit = 160;
    }
    const size_t xb = (size_t)M * K * 2, wb = (size_t)N * K * 2;
    ob = (size_t)M * N * 2;
    uint16_t* x = slurp(argv[7], xb, NULL);
    uint16_t* w = slurp(argv[8], wb, NULL);
    const size_t per = xb + wb + ob + GUARD;
    R = (unsigned)((768ull << 20) / per) + 1;
    if (R < 2) R = 2;
    if (R > 64) R = 64;
    dx = calloc(R, sizeof(void*)); dw = calloc(R, sizeof(void*)); dout = calloc(R, sizeof(void*));
    void* hx = plow_hsa_alloc_host(h, xb);
    void* hw = plow_hsa_alloc_host(h, wb);
    hpoison = plow_hsa_alloc_host(h, ob + GUARD);
    hrows = plow_hsa_alloc_host(h, (size_t)S * N * 2);
    hguard = plow_hsa_alloc_host(h, GUARD);
    if (!hx || !hw || !hpoison || !hrows || !hguard) die("host alloc");
    memcpy(hx, x, xb);
    memcpy(hw, w, wb);
    for (size_t i = 0; i < (ob + GUARD) / 2; i++) hpoison[i] = 0x7fc1;
    for (unsigned s = 0; s < R; s++) {
        dx[s] = plow_hsa_alloc(h, 0, xb);
        dw[s] = plow_hsa_alloc(h, 0, wb);
        dout[s] = plow_hsa_alloc(h, 0, ob + GUARD);
        if (!dx[s] || !dw[s] || !dout[s]) die("device alloc");
        CHECK(plow_hsa_copy_h2d(h, 0, dx[s], hx, xb));
        CHECK(plow_hsa_copy_h2d(h, 0, dw[s], hw, wb));
        CHECK(plow_hsa_copy_h2d(h, 0, dout[s], hpoison, ob + GUARD));
    }
    for (int s = 0; s < S; s++) {
        unsigned r = (unsigned)(((uint64_t)s * M) / S + ((unsigned)s * 37u) % (M / S));
        rows[s] = s == S - 1 ? M - 1 : r;
    }
    ref = malloc(sizeof(double) * S * N);
    double* xr = malloc(sizeof(double) * K);
    for (int s = 0; s < S; s++) {
        for (unsigned kk = 0; kk < K; kk++) xr[kk] = bf(x[(size_t)rows[s] * K + kk]);
        for (unsigned n = 0; n < N; n++) {
            double acc = 0;
            const uint16_t* wr = w + (size_t)n * K;
            for (unsigned kk = 0; kk < K; kk++) acc += xr[kk] * bf(wr[kk]);
            ref[(size_t)s * N + n] = acc;
        }
    }
    const uint32_t base = (8u << 16) | 4u;
    int good = 0;
    for (int i = 0; i < nk; i++) {
        current = ks[i].name;
        double rel, worst;
        ks[i].ok = check(&ks[i], base, &rel, &worst);
        ks[i].warm = ks[i].ok ? timed(&ks[i], base, 0, 5, 10) : 1e30;
        good += ks[i].ok;
        printf("{\"case\":\"%s\",\"pass\":1,\"m\":%u,\"n\":%u,\"k\":%u,\"kernel\":%d,\"info1\":%u,\"ok\":%d,"
               "\"rel_l2\":%.3e,\"max_err_rms\":%.3e,\"warm_us\":%.2f}\n",
               label, M, N, K, i, base, ks[i].ok, rel, worst, ks[i].ok ? ks[i].warm : -1.0);
        fflush(stdout);
    }
    int order[MAXK];
    for (int i = 0; i < nk; i++) order[i] = i;
    for (int i = 0; i < nk; i++)
        for (int j = i + 1; j < nk; j++)
            if (ks[order[j]].warm < ks[order[i]].warm) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    const uint32_t maps[] = {1, (8u << 16) | 1, (8u << 16) | 2, (8u << 16) | 4, (8u << 16) | 6,
                             (8u << 16) | 8, (8u << 16) | 16};
    double best = 1e30;
    int bk = -1;
    uint32_t bm = 0;
    for (int t = 0; t < topk && t < good; t++) {
        Kern* k = &ks[order[t]];
        current = k->name;
        for (unsigned mi = 0; mi < sizeof(maps) / sizeof(maps[0]); mi++) {
            double rel, worst;
            const int ok = check(k, maps[mi], &rel, &worst);
            const double wu = ok ? timed(k, maps[mi], 0, 7, 20) : -1, ru = ok ? timed(k, maps[mi], 1, 7, 20) : -1;
            printf("{\"case\":\"%s\",\"pass\":2,\"m\":%u,\"n\":%u,\"k\":%u,\"kernel\":%d,\"info1\":%u,\"ok\":%d,"
                   "\"rel_l2\":%.3e,\"max_err_rms\":%.3e,\"warm_us\":%.2f,\"rot_us\":%.2f,\"sets\":%u,"
                   "\"mt_i\":%u,\"mt_j\":%u,\"lds\":%u}\n",
                   label, M, N, K, order[t], maps[mi], ok, rel, worst, wu, ru, R, k->mi, k->mj,
                   k->k.group_segment_size);
            fflush(stdout);
            if (ok && ru < best) { best = ru; bk = order[t]; bm = maps[mi]; }
        }
    }
    printf("{\"case\":\"%s\",\"best\":true,\"m\":%u,\"n\":%u,\"k\":%u,\"kernel\":%d,\"info1\":%u,\"rot_us\":%.2f,"
           "\"passed\":%d,\"candidates\":%d,\"name\":\"%s\"}\n",
           label, M, N, K, bk, bm, best, good, nk, bk >= 0 ? ks[bk].name : "");
    plow_hsa_shutdown(h);
    return bk >= 0 ? 0 : 1;
}
