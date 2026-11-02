/* Build from the repo root inside nix develop:
 * cc -O2 -std=gnu11 -Wall -Wextra -Werror runtime/tests/amd_opcode_test.c \
 *   runtime/amd/hsa_backend.c -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" \
 *   -Wl,-rpath,"$ROCM_PATH/lib" -lhsa-runtime64 -lm -o /tmp/amd_opcode_test
 * Run beside test_kernels.elf, through gpuq.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t bf16;
static unsigned rng = 12345;
static int fails;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float roundbf(float f) { return bf2f(f2bf(f)); }
static float random_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 257) - 128) / 256.0f;
}
static void* alloc(plow_hsa* h, size_t bytes) {
    void* p = plow_hsa_alloc_host(h, bytes);
    if (!p) { fprintf(stderr, "%s\n", plow_hsa_last_error()); exit(1); }
    memset(p, 0, bytes);
    return p;
}
static bf16* random_bf(plow_hsa* h, size_t n) {
    bf16* p = alloc(h, n * sizeof(*p));
    for (size_t i = 0; i < n; i++) p[i] = f2bf(random_value());
    return p;
}
static void launch(plow_hsa* h, const char* name, unsigned blocks, void* args, size_t size) {
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, name, &k) ||
        plow_hsa_launch(h, 0, &k, blocks * PLOW_WG_THREADS, 1, 1,
                        PLOW_WG_THREADS, 1, 1, 0, args, size) || plow_hsa_wait(h, 0)) {
        fprintf(stderr, "%s: %s\n", name, plow_hsa_last_error()); exit(1);
    }
}
static void check(const char* name, const bf16* got, const bf16* ref, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        float a = bf2f(got[i]), b = bf2f(ref[i]);
        if (!isfinite(a) || fabsf(a - b) > tol * (1.0f + fabsf(b))) {
            fprintf(stderr, "%s[%zu]: %.8g vs %.8g\n", name, i, a, b);
            fails++; return;
        }
    }
}
static uint64_t pack(bf16 b, unsigned i) {
    unsigned u = (b & 0x8000u) ? (unsigned)(bf16)~b : (unsigned)(b | 0x8000u);
    return ((uint64_t)u << 32) | (uint32_t)~i;
}

static void gemv(plow_hsa* h, unsigned M, unsigned N, unsigned K, unsigned blocks, float cap) {
    bf16* x = random_bf(h, (M + 1) * K);
    bf16* w = random_bf(h, N * K);
    bf16* out = random_bf(h, M * N);
    bf16* ref = malloc(M * N * sizeof(*ref));
    uint64_t* part = alloc(h, M * blocks * sizeof(*part));
    /* Duplicate winning rows test the packed index tie-break across blocks. */
    for (unsigned k = 0; k < K; k++) w[k] = w[(N - 1) * K + k] = f2bf(1.0f);
    for (unsigned m = 1; m <= M; m++)
        for (unsigned k = 0; k < K; k++) x[m * K + k] = f2bf(fabsf(bf2f(x[m * K + k])));
    struct { bf16 *out, *x, *w; uint64_t* part; unsigned M, N, K; float cap; }
        args = {out, x + K, w, part, M, N, K, cap};
    launch(h, "gemv_argmax_test", blocks, &args, sizeof(args));
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            double sum = 0;
            for (unsigned k = 0; k < K; k++) sum += (double)bf2f(x[(m + 1) * K + k]) * bf2f(w[n * K + k]);
            float v = roundbf((float)sum);
            ref[m * N + n] = f2bf(cap > 0 ? cap * tanhf(v * (1.0f / cap)) : v);
        }
        for (unsigned b = 0; b < blocks; b++) {
            unsigned per = (N + blocks - 1) / blocks;
            uint64_t best = 0;
            for (unsigned n = b * per; n < N && n < (b + 1) * per; n++) {
                uint64_t key = pack(out[m * N + n], n);
                if (key > best) best = key;
            }
            if (part[m * blocks + b] != best) { fprintf(stderr, "argmax partial layout/tie failure\n"); fails++; }
        }
        uint64_t best = 0;
        for (unsigned b = 0; b < blocks; b++) if (part[m * blocks + b] > best) best = part[m * blocks + b];
        if ((uint32_t)~best != 0) { fprintf(stderr, "argmax tie token failure\n"); fails++; }
    }
    check("gemv", out, ref, M * N, 0.01f);
    free(ref); plow_hsa_free(h, x); plow_hsa_free(h, w); plow_hsa_free(h, out); plow_hsa_free(h, part);
}

static void per_layer(plow_hsa* h, unsigned rows, unsigned H, unsigned P, int next) {
    const unsigned col0 = 7, stride = P + 19;
    const float eps = 1e-6f, scalar = 0.75f;
    bf16 *x = random_bf(h, rows * H), *wg = random_bf(h, P * H), *wp = random_bf(h, H * P);
    bf16 *gamma = random_bf(h, H), *ple = random_bf(h, rows * stride), *gnext = random_bf(h, H);
    bf16* hn = random_bf(h, rows * H);
    bf16* ref = malloc(rows * H * 2), *nr = malloc(rows * H * 2);
    float* a = malloc(P * sizeof(float)), *y = malloc(H * sizeof(float));
    memcpy(ref, x, rows * H * 2);
    for (unsigned t = 0; t < rows; t++) {
        for (unsigned p = 0; p < P; p++) {
            double sum = 0;
            for (unsigned j = 0; j < H; j++) sum += (double)bf2f(wg[p * H + j]) * bf2f(ref[t * H + j]);
            float v = roundbf((float)sum);
            float g = roundbf(0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
            a[p] = roundbf(g * bf2f(ple[t * stride + col0 + p]));
        }
        double ss = 0;
        for (unsigned j = 0; j < H; j++) {
            double sum = 0;
            for (unsigned p = 0; p < P; p++) sum += (double)bf2f(wp[j * P + p]) * a[p];
            y[j] = roundbf((float)sum); ss += (double)y[j] * y[j];
        }
        float inv = 1.0f / sqrtf((float)ss / H + eps);
        ss = 0;
        for (unsigned j = 0; j < H; j++) {
            float n = roundbf(y[j] * inv * (next == 2 ? 1.0f : bf2f(gamma[j])));
            ref[t * H + j] = f2bf(roundbf(bf2f(ref[t * H + j]) + n) * scalar);
            float v = bf2f(ref[t * H + j]); ss += (double)v * v;
        }
        inv = 1.0f / sqrtf((float)ss / H + eps);
        for (unsigned j = 0; j < H; j++) nr[t * H + j] = f2bf(bf2f(ref[t * H + j]) * inv * (next == 2 ? 1.0f : bf2f(gnext[j])));
    }
    struct { bf16 *x, *wg, *wp, *gamma, *ple, *hn, *gnext; unsigned rows, H, P, col0, stride; float eps, scalar; }
        args = {x, wg, wp, next == 2 ? NULL : gamma, ple, next ? hn : NULL,
                next == 2 ? NULL : gnext, rows, H, P, col0, stride, eps, scalar};
    launch(h, "per_layer_input_test", 3, &args, sizeof(args));
    check("per-layer x", x, ref, rows * H, 0.025f);
    if (next) check("per-layer next", hn, nr, rows * H, 0.025f);
    free(ref); free(nr); free(a); free(y);
    plow_hsa_free(h, x); plow_hsa_free(h, wg); plow_hsa_free(h, wp); plow_hsa_free(h, gamma);
    plow_hsa_free(h, ple); plow_hsa_free(h, gnext); plow_hsa_free(h, hn);
}

static void embeds(plow_hsa* h, unsigned width) {
    const unsigned rows = 7, vocab = 13, overlays = 3, positions = 9, K = 3, cb_size = 5;
    bf16 *table = random_bf(h, vocab * width), *pt = random_bf(h, positions * width);
    bf16 *cb = random_bf(h, K * cb_size * width), *out = random_bf(h, rows * width);
    bf16* ref = malloc(rows * width * 2);
    float *ov = alloc(h, overlays * width * 4), *nw = alloc(h, width * 4);
    unsigned *tok = alloc(h, rows * 4), *sel = alloc(h, rows * 4), *pos = alloc(h, rows * 4);
    unsigned *base = alloc(h, rows * 4), *codes = alloc(h, rows * K * 4);
    for (unsigned i = 0; i < overlays * width; i++) ov[i] = random_value() * 1.003f;
    for (unsigned i = 0; i < width; i++) nw[i] = 1.0f + random_value();
    for (unsigned r = 0; r < rows; r++) {
        tok[r] = r + 1; base[r] = 100 + r * 3; pos[r] = base[r] + r;
        sel[r] = r % 3 == 0 ? UINT32_MAX : (r % 3 == 1 ? UINT32_MAX - 1 : r % overlays);
        for (unsigned k = 0; k < K; k++) codes[r * K + k] = (r + k) % cb_size;
    }
    memcpy(ref, out, rows * width * 2);
    for (unsigned r = 0; r < rows; r++) for (unsigned i = 0; i < width; i++) {
        if (sel[r] == UINT32_MAX) ref[r * width + i] = table[tok[r] * width + i];
        else if (sel[r] != UINT32_MAX - 1) ref[r * width + i] = f2bf(ov[sel[r] * width + i]);
    }
    struct { bf16 *out, *table; unsigned* tok; float* ov; unsigned* sel; unsigned rows, width, vocab, overlays; }
        oa = {out, table, tok, ov, sel, rows, width, vocab, overlays};
    launch(h, "embed_overlay_bf16_test", 3, &oa, sizeof(oa));
    check("overlay", out, ref, rows * width, 0);
    for (unsigned r = 0; r < rows; r++) for (unsigned i = 0; i < width; i++)
        ref[r * width + i] = f2bf(bf2f(table[tok[r] * width + i]) + bf2f(pt[(pos[r] - base[r]) * width + i]));
    struct { bf16 *out, *table; unsigned* tok; bf16* pt; unsigned *pos, *base; unsigned rows, width, vocab, positions; }
        pa = {out, table, tok, pt, pos, base, rows, width, vocab, positions};
    launch(h, "embed_pos_bf16_test", 3, &pa, sizeof(pa));
    check("position", out, ref, rows * width, 0);
    for (unsigned mode = 0; mode < 4; mode++) {
        for (unsigned r = 0; r < rows; r++) {
            double ss = 0;
            for (unsigned i = 0; i < width; i++) {
                if (tok[r] - 3u >= 3u) { ref[r * width + i] = table[tok[r] * width + i]; continue; }
                float sum = 0;
                for (unsigned k = 0; k < K; k++) sum += bf2f(cb[(k * cb_size + codes[r * K + k]) * width + i]);
                float v = roundbf(sum);
                if (!(mode & 1)) v = roundbf(bf2f(table[tok[r] * width + i]) + v);
                ref[r * width + i] = f2bf(v / 1.75f);
                v = bf2f(ref[r * width + i]); ss += (double)v * v;
            }
            if ((mode & 2) && tok[r] - 3u < 3u) {
                float inv = 1.0f / sqrtf((float)ss / width + 1e-5f);
                for (unsigned i = 0; i < width; i++) ref[r * width + i] = f2bf(bf2f(ref[r * width + i]) * inv * nw[i]);
            }
        }
        struct { bf16 *out, *table; unsigned* tok; bf16* cb; unsigned* codes; float* nw;
                 unsigned rows, width, vocab, K, cb_size, frame_lo, frame_rows, mode; float div, eps; }
            fa = {out, table, tok, cb, codes, (mode & 2) ? nw : NULL,
                  rows, width, vocab, K, cb_size, 3, 3, mode, 1.75f, 1e-5f};
        launch(h, "embed_frame_bf16_test", 3, &fa, sizeof(fa));
        check("frame", out, ref, rows * width, (mode & 2) ? 0.01f : 0);
    }
    free(ref); plow_hsa_free(h, table); plow_hsa_free(h, pt); plow_hsa_free(h, cb); plow_hsa_free(h, out);
    plow_hsa_free(h, ov); plow_hsa_free(h, nw); plow_hsa_free(h, tok); plow_hsa_free(h, sel);
    plow_hsa_free(h, pos); plow_hsa_free(h, base); plow_hsa_free(h, codes);
}

int main(void) {
    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
    if (bytes <= 0) return 1;
    void* elf = malloc((size_t)bytes);
    if (!elf || fread(elf, 1, (size_t)bytes, f) != (size_t)bytes) return 1;
    fclose(f);
    plow_hsa* h = plow_hsa_init();
    if (!h || plow_hsa_load_code_object(h, 0, elf, (size_t)bytes)) {
        fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1;
    }
    free(elf);
    unsigned* threads = alloc(h, sizeof(*threads));
    struct { unsigned* out; } probe = {threads};
    launch(h, "plow_probe_wg_threads", 1, &probe, sizeof(probe));
    if (*threads != PLOW_WG_THREADS) { fprintf(stderr, "workgroup width mismatch\n"); return 1; }
    plow_hsa_free(h, threads);
    gemv(h, 1, 259, 512, 3, 0); gemv(h, 3, 259, 520, 5, 30);
    gemv(h, 2, 7, 67, 11, 0); gemv(h, 1, 259, 3840, 3, 30);
    per_layer(h, 1, 1536, 256, 0); per_layer(h, 7, 64, 32, 1);
    per_layer(h, 3, 67, 35, 2); per_layer(h, 1, 2560, 256, 1);
    embeds(h, 64); embeds(h, 67); embeds(h, 1024);
    plow_hsa_shutdown(h);
    printf("amd_opcode_test: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
