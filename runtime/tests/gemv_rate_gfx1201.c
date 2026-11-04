/* gfx1201 decode GEMV: the interpreter's old/new bodies, 64 WGs x 512 threads.
 * Build from the repository root (no GPU needed):
 *
 * nix develop --command bash -c '
 *   set -eu
 *   : "${ROCM_PATH:?}"
 *   OUT=/tmp/gemv-rate
 *   mkdir -p "$OUT"
 *   for M in 1 2 4 8 16; do
 *     "$ROCM_PATH/bin/hipcc" --offload-arch=gfx1201 -mwavefrontsize64 -mcumode -O3 \
 *       --genco -Iruntime/amd -Iruntime/common -DPLOW_GEMV_MM=$M \
 *       -DPLOW_GFX12_WMMA_GEMV=1 \
 *       -Rpass-analysis=kernel-resource-usage runtime/tests/gemv_rate_gfx1201.hip \
 *       -o "$OUT/gemv_m$M.co" 2> "$OUT/gemv_m$M.resources"
 *     "$ROCM_PATH/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
 *       --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
 *       --input="$OUT/gemv_m$M.co" --output="$OUT/gemv_m$M.elf"
 *   done
 *   cc -O3 -std=gnu11 -Wall -Wextra -Werror -fopenmp \
 *     runtime/tests/gemv_rate_gfx1201.c runtime/amd/hsa_backend.c \
 *     -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
 *     -lhsa-runtime64 -lm -o "$OUT/gemv_rate"
 * '
 *
 * Run ONLY on the owner's GPU host through gpuq:
 * nix develop --command python3 scripts/bench/gpuq.py --vendor amd submit gemv-rate 1 \
 *   /tmp/gemv-rate/gemv_rate --objects=/tmp/gemv-rate
 * Add --old-only or --new-only for independent runs; --m=1 --shape=qwen-qkv
 * selects one point. --list lists shapes without opening HSA. Stdout is CSV.
 * --op=qkv|glu selects the actual fused path (new-only); --act=0|1|2 selects
 * GELU-tanh/SiLU/SITU for GLU. --edges checks tiny ragged/empty/fallback cases;
 * their timing is diagnostic. The capacity probe checks both MM and walk.
 *
 * Median HSA dispatch timestamps exclude launch/wait overhead and an untimed
 * 256 MiB stream-read eviction before EVERY sample (including warmups). Reported
 * bytes count W once, independent of M. CPU FP32 references use signed, seeded
 * dyadic BF16 inputs; compute all requested rows once per shape, outside timing.
 * Both numerical arms must pass the 2^-8 relative bound for |ref|>=1e-2; new also
 * checks fused/unfused logits byte-for-byte, untimed. This does not replace the
 * opcode oracle or model FP32 gates. --old-only never launches a new body.
 *
 * Shapes from HF config.json, 2026-11-03:
 * https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
 * https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json
 * https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json
 * E4B QKV = (8+2*2)*head_dim, with head_dim 256 sliding / 512 global.
 * Retain the requested N=4096 case as the global Q projection (8*512).
 * gate_up concatenates the two intermediate_size projections.
 */
#include "../amd/hsa_backend.h"
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t bf16;
typedef struct { const char* name; unsigned N, K; } Shape;
static const Shape shapes[] = {
    {"qwen-qkv", 4096, 1024}, {"qwen-o", 1024, 2048},
    {"qwen-gate-up", 6144, 1024}, {"qwen-down", 1024, 3072},
    {"qwen-lm-head", 151936, 1024},
    {"e4b-qkv-sliding", 3072, 2560}, {"e4b-qkv-global", 6144, 2560},
    {"e4b-q-global", 4096, 2560},
    {"e4b-gate-up", 20480, 2560}, {"e4b-down", 2560, 10240},
    {"12b-gate-up", 30720, 3840}, {"12b-down", 3840, 15360},
    {"12b-lm-head", 262144, 3840},
};
static const Shape edge_shapes[] = {
    {"tail-qkv", 37, 96}, {"empty-kv-qkv", 19, 64}, {"fallback-qkv", 37, 40},
    {"tail-gate-up", 38, 96}, {"fallback-gate-up", 38, 40},
};
static float bf2f(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static bf16 f2bf(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    return (bf16)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static bf16 value(size_t i, unsigned salt) {
    unsigned u = (unsigned)i ^ (unsigned)(i >> 32) ^ salt;
    u ^= u >> 16; u *= 0x7feb352du; u ^= u >> 15; u *= 0x846ca68bu; u ^= u >> 16;
    return f2bf(((int)(u % 129u) - 64) / 128.0f);
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static void die(const char* what) {
    fprintf(stderr, "%s: %s\n", what, plow_hsa_last_error()); exit(1);
}
static void* allocate(size_t bytes) {
    void* p = malloc(bytes);
    if (!p) die("host allocation");
    return p;
}
static void launch(plow_hsa* h, int dev, const plow_hsa_kernel* k, const void* a, size_t size) {
    if (plow_hsa_launch(h, dev, k, 64 * 512, 1, 1, 512, 1, 1, 0, a, size) ||
        plow_hsa_wait(h, dev)) die("dispatch");
}
static void load_object(plow_hsa* h, int dev, const char* dir, unsigned M) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/gemv_m%u.elf", dir, M) >= (int)sizeof(path)) die("object path");
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fseek(f, 0, SEEK_END)) die("seek object");
    long size = ftell(f);
    if (size <= 0 || fseek(f, 0, SEEK_SET)) die("object size");
    void* data = allocate((size_t)size);
    if (fread(data, 1, (size_t)size, f) != (size_t)size) die("read object");
    fclose(f);
    if (plow_hsa_load_code_object(h, dev, data, (size_t)size)) die("load object");
    free(data);
}
static unsigned integer(const char* text) {
    char* end;
    unsigned long n = strtoul(text, &end, 10);
    if (!*text || *text == '-' || *end || n > INT_MAX) die("invalid integer option");
    return (unsigned)n;
}
typedef struct { double abs, rel; size_t over; } Error;
static Error error(const bf16* got, const float* ref, size_t count) {
    Error e = {0};
    for (size_t i = 0; i < count; i++) {
        double v = bf2f(got[i]), r = ref[i], d = fabs(v - r);
        if (!isfinite(v)) { e.over++; e.abs = e.rel = INFINITY; continue; }
        if (d > e.abs) e.abs = d;
        if (fabs(r) >= 1e-2) {
            double rel = d / fabs(r);
            if (rel > e.rel) e.rel = rel;
            if (rel > 0x1p-8) e.over++;
        }
    }
    return e;
}

int main(int argc, char** argv) {
    const char* objects = "/tmp/gemv-rate", *shape = NULL;
    unsigned only_m = 0, reps = 21, warmups = 3, dev = 0, act = 1;
    int mode = 0, list = 0, failures = 0, op = 0, edges = 0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--objects=", 10)) objects = argv[i] + 10;
        else if (!strncmp(argv[i], "--shape=", 8)) shape = argv[i] + 8;
        else if (!strncmp(argv[i], "--m=", 4)) only_m = integer(argv[i] + 4);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = integer(argv[i] + 7);
        else if (!strncmp(argv[i], "--warmups=", 10)) warmups = integer(argv[i] + 10);
        else if (!strncmp(argv[i], "--dev=", 6)) dev = integer(argv[i] + 6);
        else if (!strcmp(argv[i], "--old-only") && !mode) mode = 1;
        else if (!strcmp(argv[i], "--new-only") && !mode) mode = 2;
        else if (!strcmp(argv[i], "--edges")) edges = 1;
        else if (!strncmp(argv[i], "--act=", 6)) act = integer(argv[i] + 6);
        else if (!strcmp(argv[i], "--op=qkv")) op = 1;
        else if (!strcmp(argv[i], "--op=glu")) op = 2;
        else if (!strcmp(argv[i], "--list")) list = 1;
        else goto usage;
    }
    if ((only_m && (only_m > 16 || (only_m & (only_m - 1)))) ||
        reps < 3 || reps > 1001 || !(reps & 1u) || warmups > 100) goto usage;
    if (edges && !op) goto usage;
    if (act > 2) goto usage;
    if (op) { if (mode == 1) goto usage; mode = 2; }
    const Shape* catalog = edges ? edge_shapes : shapes;
    const unsigned n_shapes = edges ? sizeof(edge_shapes) / sizeof(edge_shapes[0]) : sizeof(shapes) / sizeof(shapes[0]);
    int found = !shape;
    for (unsigned s = 0; s < n_shapes; s++) {
        if (shape && !strcmp(shape, catalog[s].name)) found = 1;
        if (list) printf("%s N=%u K=%u weight_bytes=%zu floor_us=%.3f\n",
                         catalog[s].name, catalog[s].N, catalog[s].K,
                         (size_t)catalog[s].N * catalog[s].K * 2,
                         (double)catalog[s].N * catalog[s].K * 2 / 640e3);
    }
    if (!found) goto usage;
    if (list) return 0;

    printf("shape,N,K,M,body,weight_bytes,floor_us,median_us,GB_s,pct_640,max_abs,max_rel_ge_1e-2,over,old_new_max_abs,old_new_max_rel_ge_1e-2,old_new_byte_mismatches,fused_byte_mismatches\n");
    for (unsigned s = 0; s < n_shapes; s++) {
        const Shape sh = catalog[s];
        if (shape && strcmp(shape, sh.name)) continue;
        unsigned nq = 0, nk = 0;
        if (!strcmp(sh.name, "qwen-qkv")) { nq = 2048; nk = 1024; }
        else if (!strcmp(sh.name, "e4b-qkv-sliding")) { nq = 2048; nk = 512; }
        else if (!strcmp(sh.name, "e4b-qkv-global")) { nq = 4096; nk = 1024; }
        if (!strcmp(sh.name, "tail-qkv") || !strcmp(sh.name, "fallback-qkv")) { nq = 19; nk = 9; }
        else if (!strcmp(sh.name, "empty-kv-qkv")) { nq = 19; nk = 0; }
        if ((op == 1 && !nq) || (op == 2 && !strstr(sh.name, "gate-up"))) {
            if (shape) goto usage;
            continue;
        }
        const unsigned out_n = op == 2 ? sh.N / 2 : sh.N;
        const unsigned max_m = only_m ? only_m : 16;
        const size_t nw = (size_t)sh.N * sh.K, nx = (size_t)max_m * sh.K;
        bf16* w = allocate(nw * 2), *x = allocate(nx * 2);
        float* ref = allocate((size_t)max_m * sh.N * sizeof(float));
#pragma omp parallel for
        for (size_t i = 0; i < nw; i++) w[i] = value(i, 0x12345678);
        for (size_t i = 0; i < nx; i++) x[i] = value(i, 0x9abcdef0);
        fprintf(stderr, "CPU FP32 reference: %s M=%u\n", sh.name, max_m);
#pragma omp parallel for
        for (unsigned n = 0; n < sh.N; n++) {
            for (unsigned m = 0; m < max_m; m++) {
                float sum = 0.0f;
#pragma omp simd reduction(+:sum)
                for (unsigned k = 0; k < sh.K; k++)
                    sum += bf2f(w[(size_t)n * sh.K + k]) * bf2f(x[(size_t)m * sh.K + k]);
                ref[(size_t)m * sh.N + n] = sum;
            }
        }
        if (op == 2) {
            for (unsigned m = 0; m < max_m; m++)
                for (unsigned n = 0; n < out_n; n++) {
                    const float g = ref[(size_t)m * sh.N + n];
                    const float u = ref[(size_t)m * sh.N + out_n + n];
                    const float sigmoid = 1.0f / (1.0f + expf(-g));
                    if (act == 0)
                        ref[(size_t)m * out_n + n] = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
                    else if (act == 2)
                        ref[(size_t)m * out_n + n] = (0.75f * tanhf(g / 0.75f) * sigmoid) * (0.5f * tanhf(u / 0.5f));
                    else ref[(size_t)m * out_n + n] = (g * sigmoid) * u;
                }
        }
        for (unsigned M = only_m ? only_m : 1; M <= max_m; M *= 2) {
            /* Fresh HSA executable per bucket avoids duplicate-symbol resolution
             * choosing an earlier object's MM capacity. */
            plow_hsa* h = plow_hsa_init();
            if (!h) die("HSA init");
            char name[64]; uint32_t cus, lds;
            if (plow_hsa_device_info(h, (int)dev, name, &cus, &lds)) die("device info");
            if (strcmp(name, "gfx1201") || cus != 64) die("requires gfx1201 with 64 CUs");
            load_object(h, (int)dev, objects, M);
            const size_t no = (size_t)M * out_n, evict_bytes = 256u * 1024u * 1024u;
            void *dw = plow_hsa_alloc(h, (int)dev, nw * 2), *dx = plow_hsa_alloc(h, (int)dev, nx * 2);
            void *out = plow_hsa_alloc(h, (int)dev, no * 2), *evict = plow_hsa_alloc(h, (int)dev, evict_bytes);
            void *checksum = plow_hsa_alloc(h, (int)dev, 64 * 512 * sizeof(float));
            void *part = plow_hsa_alloc(h, (int)dev, M * 64 * sizeof(uint64_t));
            if (!dw || !dx || !out || !evict || !checksum || !part) die("device allocation");
            if (plow_hsa_upload(h, (int)dev, dw, w, nw * 2) ||
                plow_hsa_upload(h, (int)dev, dx, x, nx * 2)) die("upload");
            plow_hsa_kernel probe, init, flush, kernels[2], fused;
            if (plow_hsa_get_kernel(h, (int)dev, "gemv_rate_capacity", &probe) ||
                plow_hsa_get_kernel(h, (int)dev, "bw_probe_init", &init) ||
                plow_hsa_get_kernel(h, (int)dev, "bw_probe_stream", &flush)) die("utility symbols");
            launch(h, (int)dev, &probe, &checksum, sizeof(checksum));
            unsigned capacity[2];
            if (plow_hsa_download(h, (int)dev, capacity, checksum, sizeof(capacity)) ||
                capacity[1] > 1 || !capacity[0] || (capacity[0] != M && !capacity[1]))
                die("object GEMV capacity/walk mismatch");
            struct { void* p; unsigned long long n; } ia = {evict, evict_bytes / 16};
            struct { void *out, *src; unsigned long long n; } fa = {checksum, evict, evict_bytes / 16};
            launch(h, (int)dev, &init, &ia, sizeof(ia));
            const unsigned first = mode == 2 ? 1 : 0, count = mode ? 1 : 2;
            const char* names[] = {"gemv_rate_old", op == 1 ? "gemv_rate_qkv" : op == 2 ? "gemv_rate_glu" : "gemv_rate_new"};
            float* expected = ref;
            if (op == 1) {
                expected = allocate(no * sizeof(float));
                unsigned offset = 0;
                const unsigned widths[] = {nq, nk, sh.N - nq - nk};
                for (unsigned stream = 0; stream < 3; stream++) {
                    for (unsigned m = 0; m < M; m++)
                        memcpy(expected + (size_t)M * offset + (size_t)m * widths[stream],
                               ref + (size_t)m * sh.N + offset, widths[stream] * sizeof(float));
                    offset += widths[stream];
                }
            }
            double* times[2] = {allocate(reps * sizeof(double)), allocate(reps * sizeof(double))};
            bf16* got[2] = {allocate(no * 2), allocate(no * 2)};
            bf16* poison = allocate(no * 2);
            memset(poison, 0xff, no * 2);
            for (unsigned a = first; a < first + count; a++)
                if (plow_hsa_get_kernel(h, (int)dev, names[a], &kernels[a])) die("GEMV symbol");
            hsa_queue_t* queue = (hsa_queue_t*)(uintptr_t)plow_hsa_queue_raw(h, (int)dev);
            hsa_agent_t agent = {plow_hsa_agent_raw(h, (int)dev)};
            hsa_signal_t signal = {plow_hsa_done_signal_raw(h, (int)dev)};
            uint64_t freq;
            if (hsa_amd_profiling_set_profiler_enabled(queue, 1) != HSA_STATUS_SUCCESS ||
                hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq) != HSA_STATUS_SUCCESS || !freq)
                die("HSA profiling setup");
            struct { void *out, *x, *w; unsigned M, N, K; } ga = {out, dx, dw, M, sh.N, sh.K};
            struct { void *out, *x, *w; unsigned M, N, K, Nq, Nk; } qa = {out, dx, dw, M, sh.N, sh.K, op == 2 ? act : nq, nk};
            for (unsigned r = 0; r < warmups + reps; r++) {
                for (unsigned j = 0; j < count; j++) {
                    unsigned a = first + ((r + j) % count);
                    if (plow_hsa_upload(h, (int)dev, out, poison, no * 2)) die("poison output");
                    launch(h, (int)dev, &flush, &fa, sizeof(fa));
                    launch(h, (int)dev, &kernels[a], op ? (const void*)&qa : (const void*)&ga, op ? sizeof(qa) : sizeof(ga));
                    hsa_amd_profiling_dispatch_time_t t;
                    if (hsa_amd_profiling_get_dispatch_time(agent, signal, &t) != HSA_STATUS_SUCCESS || t.end <= t.start)
                        die("HSA dispatch timestamp");
                    if (r >= warmups) times[a][r - warmups] = (t.end - t.start) * 1e6 / freq;
                    if (r + 1 == warmups + reps && plow_hsa_download(h, (int)dev, got[a], out, no * 2))
                        die("download output");
                }
            }
            size_t fused_diff = 0, pair_diff = 0;
            Error pair = {0};
            if (mode != 1 && !op) {
                if (plow_hsa_get_kernel(h, (int)dev, "gemv_rate_argmax", &fused)) die("fused symbol");
                struct { void *out, *x, *w, *part; unsigned M, N, K; } aa = {out, dx, dw, part, M, sh.N, sh.K};
                if (plow_hsa_upload(h, (int)dev, out, poison, no * 2)) die("poison fused output");
                launch(h, (int)dev, &fused, &aa, sizeof(aa));
                bf16* logits = allocate(no * 2);
                if (plow_hsa_download(h, (int)dev, logits, out, no * 2)) die("download fused output");
                for (size_t i = 0; i < no * 2; i++)
                    fused_diff += ((unsigned char*)logits)[i] != ((unsigned char*)got[1])[i];
                free(logits);
                if (fused_diff) failures++;
            }
            if (!mode) {
                float* old = allocate(no * sizeof(float));
                for (size_t i = 0; i < no; i++) old[i] = bf2f(got[0][i]);
                pair = error(got[1], old, no);
                for (size_t i = 0; i < no * 2; i++)
                    pair_diff += ((unsigned char*)got[0])[i] != ((unsigned char*)got[1])[i];
                free(old);
            }
            for (unsigned a = first; a < first + count; a++) {
                Error e = error(got[a], expected, no);
                if (e.over) {
                    failures++;
                    for (unsigned m = 0; m < M && !op; m++) {
                        Error row = error(got[a] + (size_t)m * sh.N, ref + (size_t)m * sh.N, sh.N);
                        fprintf(stderr, "row-error %s body=%s M=%u row=%u over=%zu max_rel=%.9g samples=",
                                sh.name, a ? "new" : "old", M, m, row.over, row.rel);
                        for (unsigned n = 0; n < 4 && n < sh.N; n++)
                            fprintf(stderr, " %.9g/%.9g", bf2f(got[a][(size_t)m * sh.N + n]),
                                    ref[(size_t)m * sh.N + n]);
                        fputc('\n', stderr);
                    }
                }
                qsort(times[a], reps, sizeof(double), cmpd);
                double us = times[a][reps / 2], bw = nw * 2 / us / 1e3;
                printf("%s,%u,%u,%u,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.9g,%.9g,%zu,",
                       sh.name, sh.N, sh.K, M, a ? "new" : "old", nw * 2,
                       nw * 2 / 640e3, us, bw, bw / 6.4, e.abs, e.rel, e.over);
                if (!mode) printf("%.9g,%.9g,%zu,", pair.abs, pair.rel, pair_diff);
                else printf("NA,NA,NA,");
                if (a && !op) printf("%zu\n", fused_diff); else printf("NA\n");
                fflush(stdout);
            }
            if (op == 1) free(expected);
            free(times[0]); free(times[1]); free(got[0]); free(got[1]); free(poison);
            plow_hsa_free(h, part); plow_hsa_free(h, checksum); plow_hsa_free(h, evict);
            plow_hsa_free(h, out); plow_hsa_free(h, dx); plow_hsa_free(h, dw);
            plow_hsa_shutdown(h);
            if (only_m) break;
        }
        free(ref); free(x); free(w);
    }
    fprintf(stderr, "numerical/fused failures=%d; bandwidth qualification is separate\n", failures);
    return failures ? 1 : 0;
usage:
    fprintf(stderr, "usage: %s [--objects=DIR] [--old-only|--new-only] [--m=1|2|4|8|16]\n"
            "  [--op=qkv|glu] [--act=0|1|2] [--edges] [--shape=NAME] [--reps=ODD] [--warmups=N] [--dev=N] [--list]\n", argv[0]);
    return 2;
}
