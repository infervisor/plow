/* HSA host runner for register-only MFMA/WMMA issue rate and the gfx942 LDS mix.
 * Build from the repo root (all artifacts in scratch):
 *
 * nix develop --command bash -c '
 *   set -eu
 *   : "${ROCM_PATH:?}"
 *   RATE_OUT=/tmp/r9700-ubench
 *   mkdir -p "$RATE_OUT"
 *   for wave in 32 64; do
 *     flags=(); if [ "$wave" = 64 ]; then flags=(-mwavefrontsize64); fi
 *     "$ROCM_PATH/bin/hipcc" --offload-arch=gfx1201 -mcumode "${flags[@]}" -O3 \
 *       --genco runtime/ubench/wmma_rate_gfx1201.hip -o "$RATE_OUT/wmma_w$wave.co"
 *     "$ROCM_PATH/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
 *       --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
 *       --input="$RATE_OUT/wmma_w$wave.co" --output="$RATE_OUT/wmma_w$wave.elf"
 *   done
 *   "$ROCM_PATH/bin/hipcc" --offload-arch=gfx942 -O3 --genco \
 *     -Iruntime/amd -Iruntime/common runtime/ubench/mfma_rate_gfx942.hip \
 *     -o "$RATE_OUT/mfma_rate.co"
 *   "$ROCM_PATH/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
 *     --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
 *     --input="$RATE_OUT/mfma_rate.co" --output="$RATE_OUT/mfma_rate.elf"
 *   cc -O2 runtime/ubench/mfma_rate_test.c runtime/amd/hsa_backend.c \
 *     -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
 *     -lhsa-runtime64 -o "$RATE_OUT/mfma_rate_test"
 * '
 *
 * Run on R9700:
 * nix develop --command python3 scripts/bench/gpuq.py --vendor amd submit wmma32 1 \
 *   /tmp/r9700-ubench/mfma_rate_test --arch=gfx1201 --wave=32 \
 *   --object=/tmp/r9700-ubench/wmma_w32.elf
 * nix develop --command python3 scripts/bench/gpuq.py --vendor amd submit wmma64 1 \
 *   /tmp/r9700-ubench/mfma_rate_test --arch=gfx1201 --wave=64 \
 *   --object=/tmp/r9700-ubench/wmma_w64.elf
 * Run on gfx942:
 * nix develop --command python3 scripts/bench/gpuq.py --vendor amd submit mfma942 1 \
 *   /tmp/r9700-ubench/mfma_rate_test --arch=gfx942 --object=/tmp/r9700-ubench/mfma_rate.elf
 *
 * Optional: --dev=0 --iters=4000 --wg-per-cu=4 --kernel=wmma_bf16_w32_ch4.
 * WMMA defaults to 4 waves/WG and sweeps 1,2,4,8 WG/CU; gfx942 retains
 * PLOW_WG_WAVES wave64 waves and 1 WG/CU. Timings include host launch + wait.
 * Dense operations count multiply and add separately; int8 reports TOP/s.
 * R9700 reference supplied for this probe: ~191 TF/s bf16/f16, ~383 TF/s fp8
 * at 2.92 GHz (64 CUs); these are theoretical, not measured ceilings.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static unsigned number(const char* s, unsigned max) {
    char* end;
    unsigned long n = strtoul(s, &end, 10);
    if (!*s || *s == '-' || *end || n > max) {
        fprintf(stderr, "invalid number: %s\n", s); exit(1);
    }
    return (unsigned)n;
}

struct rate_kernel { char sym[64]; unsigned ch; double macs; unsigned dtype, rd; };

int main(int argc, char** argv) {
    const char *object = NULL, *arch = NULL, *only = NULL;
    unsigned wave = 0, wg_per_cu = 0, iters = 4000;
    int dev = 0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--object=", 9)) object = argv[i] + 9;
        else if (!strncmp(argv[i], "--arch=", 7)) arch = argv[i] + 7;
        else if (!strncmp(argv[i], "--kernel=", 9)) only = argv[i] + 9;
        else if (!strncmp(argv[i], "--wave=", 7)) wave = number(argv[i] + 7, 64);
        else if (!strncmp(argv[i], "--dev=", 6)) dev = (int)number(argv[i] + 6, INT_MAX);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = number(argv[i] + 8, 65536);
        else if (!strncmp(argv[i], "--wg-per-cu=", 12)) wg_per_cu = number(argv[i] + 12, 64);
        else {
            fprintf(stderr, "usage: %s [--object=FILE] [--arch=gfx942|gfx1201] [--wave=32|64]\n"
                    "  [--dev=N] [--iters=N] [--wg-per-cu=N] [--kernel=NAME]\n", argv[0]);
            return 1;
        }
    }
    if (!iters || (wave && wave != 32 && wave != 64) ||
        (arch && strcmp(arch, "gfx942") && strcmp(arch, "gfx1201"))) {
        fprintf(stderr, "invalid arch, wave size or iteration count\n"); return 1;
    }
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    if (plow_hsa_device_info(h, dev, nm, &cus, &lds) || !cus) goto fail;
    if (!arch) arch = nm;
    const int wmma = !strcmp(arch, "gfx1201");
    if (strcmp(arch, nm) || (!wmma && strcmp(arch, "gfx942"))) {
        fprintf(stderr, "unsupported device/arch pairing: %s / %s\n", nm, arch); return 1;
    }
    if (!wave) wave = wmma ? 32 : 64;
    if (!wmma && wave != 64) { fprintf(stderr, "gfx942 requires wave64\n"); return 1; }
    if (!object) object = wmma ? (wave == 32 ? "wmma_w32.elf" : "wmma_w64.elf") : "mfma_rate.elf";
    FILE* f = fopen(object, "rb");
    if (!f) { perror(object); return 1; }
    if (fseek(f, 0, SEEK_END)) { perror(object); return 1; }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET)) { fprintf(stderr, "invalid object\n"); return 1; }
    void* co = malloc((size_t)n);
    if (!co || fread(co, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read object failed\n"); return 1; }
    fclose(f);
    int loaded = plow_hsa_load_code_object(h, dev, co, (size_t)n);
    free(co);
    if (loaded) goto fail;

    struct rate_kernel K[32] = {
        {"mfma_ch1", 1, 32.0*32*16, 0, 0}, {"mfma_ch2", 2, 32.0*32*16, 0, 0},
        {"mfma_ch4", 4, 32.0*32*16, 0, 0}, {"mfma_ch6", 6, 32.0*32*16, 0, 0},
        {"mfma_ch8", 8, 32.0*32*16, 0, 0},
        {"mfma16_ch4", 4, 16.0*16*32, 0, 0}, {"mfma16_ch8", 8, 16.0*16*32, 0, 0},
        {"mfma16_ch16", 16, 16.0*16*32, 0, 0}, {"mfma16_ch24", 24, 16.0*16*32, 0, 0},
        {"mfma16_ch32", 32, 16.0*16*32, 0, 0},
        {"mfma8_ch2", 2, 32.0*32*64, 2, 0}, {"mfma8_ch4", 4, 32.0*32*64, 2, 0},
        {"mfma8_ch6", 6, 32.0*32*64, 2, 0},
        {"rd_r5", 0, 0, 0, 5}, {"rd_r8", 0, 0, 0, 8},
        {"mix_r5_m6", 6, 32.0*32*16, 0, 5}, {"mix_r6_m8", 8, 32.0*32*16, 0, 6},
        {"mix_r8_m16", 16, 32.0*32*16, 0, 8}, {"mix_r5_m12", 12, 32.0*32*16, 0, 5},
        {"mix_r5_m24", 24, 32.0*32*16, 0, 5},
    };
    const char* dtype[] = {"bf16", "f16", "fp8", "int8"};
    unsigned nk = 20;
    if (wmma) {
        const unsigned chains[] = {1, 2, 4, 6, 8, 12, 16};
        nk = 0;
        for (unsigned d = 0; d < 4; d++)
            for (unsigned c = 0; c < sizeof(chains) / sizeof(chains[0]); c++) {
                struct rate_kernel* k = &K[nk++];
                snprintf(k->sym, sizeof(k->sym), "wmma_%s_w%u_ch%u", dtype[d], wave, chains[c]);
                k->ch = chains[c]; k->macs = 16.0*16*16; k->dtype = d; k->rd = 0;
            }
    }
    const unsigned waves = wmma ? 4 : PLOW_WG_WAVES, threads = waves * wave;
    const unsigned max_wg = wg_per_cu ? wg_per_cu : (wmma ? 8 : 1);
    if ((uint64_t)cus * max_wg * threads > UINT32_MAX) { fprintf(stderr, "grid too large\n"); return 1; }
    const size_t out_bytes = (size_t)cus * max_wg * threads * sizeof(float);
    void* dO = plow_hsa_alloc(h, dev, out_bytes);
    float* output = wmma ? malloc(out_bytes) : NULL;
    if (!dO || (wmma && !output)) { fprintf(stderr, "output allocation failed\n"); return 1; }
    double best_rate[4] = {0};
    const char* best_name[4] = {0};
    unsigned best_wg[4] = {0}, ran = 0;
    printf("dev%d %s CUs=%u wave%u %u waves/WG iters=%u; host launch+wait, best of 5\n",
           dev, nm, cus, wave, waves, iters);
    for (unsigned i = 0; i < nk; i++) {
        if (only && strcmp(only, K[i].sym)) continue;
        plow_hsa_kernel k;
        if (plow_hsa_get_kernel(h, dev, K[i].sym, &k)) {
            if (wmma && K[i].dtype >= 2 && !only) {
                printf("  %-24s unavailable in object\n", K[i].sym); continue;
            }
            goto fail;
        }
        for (unsigned wg = wg_per_cu ? wg_per_cu : 1; wg <= max_wg; wg *= 2) {
            const unsigned grid = cus * wg * threads;
            struct __attribute__((packed)) { void* o; unsigned it; } args = {dO, iters};
            if (plow_hsa_launch(h, dev, &k, grid, 1, 1, threads, 1, 1, 0, &args, sizeof(args)) ||
                plow_hsa_wait(h, dev)) goto fail;
            if (wmma) {
                if (plow_hsa_download(h, dev, output, dO, (size_t)grid * sizeof(float))) goto fail;
                for (unsigned t = 0; t < grid; t++) {
                    const double expected = (256 / wave) * (K[i].ch * (128.0 * iters + (t & 7)) +
                                                          K[i].ch * (K[i].ch - 1) / 2.0);
                    if (!isfinite(output[t]) || fabs(output[t] - expected) > expected * 1e-5) {
                        fprintf(stderr, "%s output[%u]=%g expected=%g\n", K[i].sym, t, output[t], expected);
                        return 1;
                    }
                }
            }
            double best = 1e30;
            for (int r = 0; r < 5; r++) {
                const double t0 = now();
                if (plow_hsa_launch(h, dev, &k, grid, 1, 1, threads, 1, 1, 0, &args, sizeof(args)) ||
                    plow_hsa_wait(h, dev)) goto fail;
                const double dt = now() - t0;
                if (dt < best) best = dt;
            }
            const double steps = (double)iters * 8 * cus * wg * waves;
            const double rate = 2 * K[i].macs * K[i].ch * steps / best / 1e12;
            const char* unit = K[i].dtype == 3 ? "TOP/s" : "TF/s";
            printf("  %-24s chains=%2u WG/CU=%2u %8.3f ms %8.2f %s LDS=%u B scratch=%u B/thread",
                   K[i].sym, K[i].ch, wg, best * 1e3, rate, unit,
                   k.group_segment_size, k.private_segment_size);
            if (K[i].rd) printf(" LDS-read=%.1f GB/s", steps * K[i].rd * 1024 / best / 1e9);
            printf("%s\n", wmma ? " check=pass" : "");
            if (!K[i].rd && rate > best_rate[K[i].dtype]) {
                best_rate[K[i].dtype] = rate; best_name[K[i].dtype] = K[i].sym; best_wg[K[i].dtype] = wg;
            }
            ran++;
        }
    }
    for (unsigned d = 0; d < 4; d++)
        if (best_name[d]) printf("best %s: %.2f %s (%s, WG/CU=%u)\n", dtype[d], best_rate[d],
                                 d == 3 ? "TOP/s" : "TF/s", best_name[d], best_wg[d]);
    free(output);
    plow_hsa_free(h, dO);
    plow_hsa_shutdown(h);
    if (!ran) { fprintf(stderr, "no matching kernels\n"); return 1; }
    return 0;
fail:
    fprintf(stderr, "%s\n", plow_hsa_last_error());
    plow_hsa_shutdown(h);
    return 1;
}
