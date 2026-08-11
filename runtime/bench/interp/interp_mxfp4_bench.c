/* interp_mxfp4_bench.c — MXFP4 GEMM through the persistent AMD interpreter. */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static const double FP4[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

struct Rung {
    unsigned short op;
    const char* symbol;
    unsigned bm, bn;
};

static bf16 f2bf(float f) {
    unsigned u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double e8m0(unsigned char b) { return ldexp(1.0, (int)b - 127); }

static void json_string(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            fputc('\\', f);
            fputc(c, f);
        } else if (c >= 0x20) {
            fputc(c, f);
        }
    }
    fputc('"', f);
}

static int load_file(const char* path, void** data, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END)) return -1;
    long n = ftell(f);
    if (n < 0) return -1;
    rewind(f);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) return -1;
    fclose(f);
    *data = p;
    *size = (size_t)n;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <interp.elf> <M> <N> <K> [label]\n", argv[0]);
        return 2;
    }
    const char* object = argv[1];
    unsigned M = (unsigned)strtoul(argv[2], NULL, 10);
    unsigned N = (unsigned)strtoul(argv[3], NULL, 10);
    unsigned K = (unsigned)strtoul(argv[4], NULL, 10);
    const char* label = argc > 5 ? argv[5] : "shape";
    const char* cleared = getenv("PLOW_STAGE4_CLEARED");
    const char* gpu = getenv("PLOW_GPU");
    const char* toolchain = getenv("PLOW_TOOLCHAIN_LABEL");
    const char* build_id = getenv("PLOW_BUILD_ID");
    const char* lease = getenv("PLOW_LEASE_LABEL");
    const char* jsonl = getenv("PLOW_GEMM_JSONL");
    if (!M || !N || !K || K % 64u) {
        fprintf(stderr, "M/N/K must be nonzero and MXFP4 K must be divisible by 64\n");
        return 2;
    }
    if (!cleared || strcmp(cleared, "1")) {
        fprintf(stderr, "REFUSED: set PLOW_STAGE4_CLEARED=1 only after Stages 1-3 pass\n");
        return 2;
    }
    if (!gpu || !*gpu || !toolchain || !*toolchain || !build_id || !*build_id || !lease ||
        !*lease || !jsonl || !*jsonl) {
        fprintf(stderr,
                "PLOW_GPU, PLOW_TOOLCHAIN_LABEL, PLOW_BUILD_ID, PLOW_LEASE_LABEL, and "
                "PLOW_GEMM_JSONL are required provenance\n");
        return 2;
    }

    plow_hsa* h = plow_hsa_init();
    if (!h) {
        fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error());
        return 1;
    }
    char arch[64];
    uint32_t cus = 0, lds = 0;
    if (plow_hsa_device_info(h, 0, arch, &cus, &lds) || !cus) {
        fprintf(stderr, "device info: %s\n", plow_hsa_last_error());
        return 1;
    }
    if (strcmp(arch, "gfx942") && strcmp(arch, "gfx950")) {
        fprintf(stderr, "unsupported architecture %s\n", arch);
        return 2;
    }

    void* co = NULL;
    size_t co_n = 0;
    if (load_file(object, &co, &co_n)) {
        perror(object);
        return 1;
    }
    if (plow_hsa_load_code_object(h, 0, co, co_n)) {
        fprintf(stderr, "load %s: %s\n", object, plow_hsa_last_error());
        return 1;
    }
    char interp_symbol[96];
    snprintf(interp_symbol, sizeof interp_symbol, "plow_interp_%s", arch);
    plow_hsa_kernel kernel;
    if (plow_hsa_get_kernel(h, 0, interp_symbol, &kernel)) {
        fprintf(stderr, "kernel %s: %s\n", interp_symbol, plow_hsa_last_error());
        return 1;
    }

    unsigned default_bm = !strcmp(arch, "gfx942") ? 192u : 256u;
    const struct Rung rungs[] = {
        {PLOW_DOP_GEMM_MXFP4, "PLOW_DOP_GEMM_MXFP4", default_bm, 256},
        {PLOW_DOP_GEMM_MED_MXFP4, "PLOW_DOP_GEMM_MED_MXFP4", 128, 128},
        {PLOW_DOP_GEMM_SMALL_MXFP4, "PLOW_DOP_GEMM_SMALL_MXFP4", 64, 128},
        {PLOW_DOP_GEMM_WIDE_MXFP4, "PLOW_DOP_GEMM_WIDE_MXFP4", 128, 256},
        {PLOW_DOP_GEMM_C5_MXFP4, "PLOW_DOP_GEMM_C5_MXFP4", 192, 256},
    };

    size_t nA = (size_t)M * K, nW = (size_t)N * K / 2;
    size_t nS = (size_t)N * (K / 32), nC = (size_t)M * N;
    bf16* hA = plow_hsa_alloc_host(h, nA * 2);
    unsigned char* hW = plow_hsa_alloc_host(h, nW);
    unsigned char* hS = plow_hsa_alloc_host(h, nS);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    void* dA = plow_hsa_alloc(h, 0, nA * 2);
    void* dW = plow_hsa_alloc(h, 0, nW);
    void* dS = plow_hsa_alloc(h, 0, nS);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    if (!hA || !hW || !hS || !hC || !dA || !dW || !dS || !dC) {
        fprintf(stderr, "allocation failed: %s\n", plow_hsa_last_error());
        return 1;
    }
    srand(5);
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nW; i++) hW[i] = (unsigned char)(rand() & 0xff);
    for (size_t i = 0; i < nS; i++) hS[i] = (unsigned char)(123 + i % 3);
    if (plow_hsa_copy_h2d(h, 0, dA, hA, nA * 2) || plow_hsa_copy_h2d(h, 0, dW, hW, nW) ||
        plow_hsa_copy_h2d(h, 0, dS, hS, nS)) {
        fprintf(stderr, "input upload: %s\n", plow_hsa_last_error());
        return 1;
    }

    PlowDevInst inst;
    memset(&inst, 0, sizeof inst);
    inst.blocks = (unsigned short)cus;
    inst.t[0] = 0;
    inst.t[1] = 1;
    inst.t[2] = 2;
    inst.t[3] = 3;
    for (int i = 4; i < 8; i++) inst.t[i] = PLOW_TENSOR_NONE;
    inst.i[0] = M;
    inst.i[1] = N;
    inst.i[2] = K;

    PlowStreamEnt* stream = calloc(cus, sizeof *stream);
    unsigned* sofs = malloc(cus * sizeof *sofs);
    unsigned* slen = malloc(cus * sizeof *slen);
    if (!stream || !sofs || !slen) {
        fprintf(stderr, "host program allocation failed\n");
        return 1;
    }
    for (unsigned i = 0; i < cus; i++) {
        stream[i].slice = i;
        sofs[i] = i;
        slen[i] = 1;
    }
    void* tensors[4] = {dC, dA, dW, dS};
    void* d_inst = plow_hsa_alloc(h, 0, sizeof inst);
    void* d_stream = plow_hsa_alloc(h, 0, cus * sizeof *stream);
    void* d_sofs = plow_hsa_alloc(h, 0, cus * sizeof *sofs);
    void* d_slen = plow_hsa_alloc(h, 0, cus * sizeof *slen);
    void* d_ctr = plow_hsa_alloc(h, 0, 8 * sizeof(unsigned));
    void* d_tensors = plow_hsa_alloc(h, 0, sizeof tensors);
    unsigned counters[8] = {0};
    if (!stream || !sofs || !slen || !d_inst || !d_stream || !d_sofs || !d_slen || !d_ctr ||
        !d_tensors || plow_hsa_upload(h, 0, d_stream, stream, cus * sizeof *stream) ||
        plow_hsa_upload(h, 0, d_sofs, sofs, cus * sizeof *sofs) ||
        plow_hsa_upload(h, 0, d_slen, slen, cus * sizeof *slen) ||
        plow_hsa_upload(h, 0, d_ctr, counters, sizeof counters) ||
        plow_hsa_upload(h, 0, d_tensors, tensors, sizeof tensors)) {
        fprintf(stderr, "program allocation/upload: %s\n", plow_hsa_last_error());
        return 1;
    }
    PlowProgram prog;
    memset(&prog, 0, sizeof prog);
    prog.insts = d_inst;
    prog.stream = d_stream;
    prog.stream_ofs = d_sofs;
    prog.stream_len = d_slen;
    prog.counters = d_ctr;
    prog.tensors = d_tensors;

    FILE* jf = fopen(jsonl, "a");
    if (!jf) {
        perror(jsonl);
        return 1;
    }
    printf("%s arch=%s CUs=%u interpreter=%s M=%u N=%u K=%u\n", gpu, arch, cus,
           interp_symbol, M, N, K);
    int any_bad = 0;
    for (size_t r = 0; r < sizeof rungs / sizeof rungs[0]; r++) {
        inst.op = rungs[r].op;
        plow_hsa_upload(h, 0, d_inst, &inst, sizeof inst);
        for (size_t i = 0; i < nC; i++) hC[i] = (bf16)0x7fc0u;
        plow_hsa_copy_h2d(h, 0, dC, hC, nC * 2);
        for (int w = 0; w < 50; w++)
            plow_hsa_launch(h, 0, &kernel, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                            0, &prog, sizeof prog);
        plow_hsa_wait(h, 0);

        enum { GROUPS = 12, PER_GROUP = 4 };
        double samples[GROUPS];
        for (int g = 0; g < GROUPS; g++) {
            double t0 = now();
            for (int i = 0; i < PER_GROUP; i++)
                plow_hsa_launch(h, 0, &kernel, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1,
                                1, 0, &prog, sizeof prog);
            plow_hsa_wait(h, 0);
            samples[g] = (now() - t0) / PER_GROUP * 1e9;
        }
        plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);
        int bad = 0;
        for (size_t i = 0; i < nC; i++) {
            if (!isfinite(bf2f(hC[i]))) {
                bad++;
                break;
            }
        }
        srand(17);
        for (int s = 0; s < 24; s++) {
            unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
            double expected = 0;
            for (unsigned k = 0; k < K; k++) {
                size_t e = (size_t)n * K + k;
                unsigned char packed = hW[e >> 1];
                unsigned fp4 = k & 1u ? packed >> 4 : packed & 0xfu;
                double weight = FP4[fp4] * e8m0(hS[(size_t)n * (K / 32) + (k >> 5)]);
                expected += (double)bf2f(hA[(size_t)m * K + k]) * weight;
            }
            double got = bf2f(hC[(size_t)m * N + n]);
            if (fabs(got - expected) / (fabs(expected) + 1e-3) > 0.03) bad++;
        }
        double mean_ns = 0;
        for (int g = 0; g < GROUPS; g++) mean_ns += samples[g];
        mean_ns /= GROUPS;
        printf("  %-28s tile=%ux%ux64 mean=%.1f us %s\n", rungs[r].symbol, rungs[r].bm,
               rungs[r].bn, mean_ns / 1e3, bad ? "MISMATCH" : "ok");
        any_bad |= bad != 0;

        if (jf) {
            fputs("{\"gpu\":", jf);
            json_string(jf, gpu);
            fprintf(jf, ",\"cu\":%u,\"arch\":", cus);
            json_string(jf, arch);
            fputs(",\"toolchain\":", jf);
            json_string(jf, toolchain);
            fputs(",\"build_id\":", jf);
            json_string(jf, build_id);
            fputs(",\"lease_label\":", jf);
            json_string(jf, lease);
            fputs(",\"object\":", jf);
            json_string(jf, object);
            fprintf(jf, ",\"m\":%u,\"n\":%u,\"k\":%u,\"quant\":\"Mxfp4\",\"label\":",
                    M, N, K);
            json_string(jf, label);
            fprintf(jf, ",\"tile\":\"%ux%ux64\",\"sym\":", rungs[r].bm, rungs[r].bn);
            json_string(jf, rungs[r].symbol);
            fputs(",\"interpreter_symbol\":", jf);
            json_string(jf, interp_symbol);
            fprintf(jf, ",\"correct\":%s,\"samples_ns\":[", bad ? "false" : "true");
            for (int g = 0; g < GROUPS; g++) fprintf(jf, "%s%.1f", g ? "," : "", samples[g]);
            fputs("]}\n", jf);
            fflush(jf);
        }
    }
    fclose(jf);
    plow_hsa_shutdown(h);
    return any_bad ? 1 : 0;
}
