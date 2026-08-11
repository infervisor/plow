/* Production-interpreter comparison of fused and unfused MXFP4 GLU on gfx942.
 * Timing is opt-in because Stage 3 correctness must clear before Stage 4 starts.
 *
 * Build:
 *   nix develop --command cmake -S runtime -B runtime/build -DPLOW_ROCM=ON \
 *     -DPLOW_BENCH=ON -DPLOW_HIP_ARCH=gfx942
 *   nix develop --command cmake --build runtime/build \
 *     --target interp_mxfp4_glu_gfx942
 *
 * Run under gpulease (rc=76 means discard the JSONL row):
 *   nix develop --command env PLOW_STAGE4_CLEARED=1 \
 *     PLOW_GPU=MI325X PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix PLOW_BUILD_ID=<revision> \
 *     PLOW_LEASE_LABEL=k3-mxfp4-glu PLOW_GEMM_GLU_JSONL=<samples.jsonl> \
 *     perf-data/harness/gpulease -n 1 k3-mxfp4-glu \
 *     runtime/build/bench/interp_mxfp4_glu_gfx942 \
 *     <interp_prefill_k3_moe_a4w4.elf> <M> <N> <K> [samples]
 */
#define _POSIX_C_SOURCE 200809L
#include "dev_isa.h"
#include "hsa_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;
enum { T_OUT, T_ACT, T_WG, T_SG, T_SU, T_WU, T_UP, N_TENSORS };

typedef struct {
    PlowProgram prog;
    void *insts, *stream, *sofs, *slen, *waits, *succs, *counters;
    size_t counters_bytes;
} BenchProgram;
typedef struct {
    double nrmse, max_norm_abs;
    unsigned checked;
    int finite;
} OracleStats;

static uint32_t rng_state = 0x13579bdfu;
static uint32_t rng_u32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}
static bf16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double bf2d(bf16 b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return (double)f;
}
static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + t.tv_nsec;
}
static void fail_hsa(const char* what) {
    fprintf(stderr, "%s: %s\n", what, plow_hsa_last_error());
    exit(1);
}
static void* device_alloc(plow_hsa* h, size_t bytes) {
    void* p = plow_hsa_alloc(h, 0, bytes);
    if (!p) fail_hsa("device allocation");
    return p;
}
static void upload(plow_hsa* h, void* dst, const void* src, size_t bytes, const char* what) {
    if (plow_hsa_upload(h, 0, dst, src, bytes)) fail_hsa(what);
}
static void init_inst(PlowDevInst* in, uint16_t op, uint16_t blocks) {
    memset(in, 0, sizeof *in);
    in->op = op;
    in->blocks = blocks;
    for (unsigned j = 0; j < 8; j++) in->t[j] = PLOW_TENSOR_NONE;
}

static BenchProgram make_program(plow_hsa* h, unsigned ncu, int fused, unsigned m, unsigned n,
                                 unsigned k, void* tensors) {
    const unsigned nops = fused ? 1u : 3u;
    PlowDevInst* insts = calloc(nops, sizeof *insts);
    PlowStreamEnt* stream = calloc((size_t)ncu * nops, sizeof *stream);
    uint32_t* sofs = malloc((size_t)ncu * sizeof *sofs);
    uint32_t* slen = malloc((size_t)ncu * sizeof *slen);
    const PlowWait waits[2] = {{0, ncu}, {1, ncu}};
    const uint32_t succs[2] = {0, 1};
    if (!insts || !stream || !sofs || !slen) {
        fprintf(stderr, "host allocation failed\n");
        exit(1);
    }
    if (fused) {
        init_inst(&insts[0], PLOW_DOP_GEMM_GLU_MXFP4, (uint16_t)ncu);
        insts[0].t[0] = T_OUT;
        insts[0].t[1] = T_ACT;
        insts[0].t[2] = T_WG;
        insts[0].t[3] = T_SG;
        insts[0].t[4] = T_SU;
        insts[0].t[5] = T_WU;
        insts[0].i[0] = m;
        insts[0].i[1] = n;
        insts[0].i[2] = k;
        insts[0].i[5] = 1; /* PLOW_ACT_SILU */
    } else {
        init_inst(&insts[0], PLOW_DOP_GEMM_MXFP4, (uint16_t)ncu);
        insts[0].t[0] = T_OUT;
        insts[0].t[1] = T_ACT;
        insts[0].t[2] = T_WG;
        insts[0].t[3] = T_SG;
        insts[0].i[0] = m;
        insts[0].i[1] = n;
        insts[0].i[2] = k;
        init_inst(&insts[1], PLOW_DOP_GEMM_MXFP4, (uint16_t)ncu);
        insts[1].t[0] = T_UP;
        insts[1].t[1] = T_ACT;
        insts[1].t[2] = T_WU;
        insts[1].t[3] = T_SU;
        insts[1].i[0] = m;
        insts[1].i[1] = n;
        insts[1].i[2] = k;
        init_inst(&insts[2], PLOW_DOP_GLU, (uint16_t)ncu);
        insts[2].t[0] = T_OUT;
        insts[2].t[1] = T_OUT;
        insts[2].t[2] = T_UP;
        insts[2].i[0] = m * n;
        insts[2].i[1] = 1; /* PLOW_ACT_SILU */
    }
    for (unsigned cu = 0; cu < ncu; cu++) {
        sofs[cu] = cu * nops;
        slen[cu] = nops;
        for (unsigned op = 0; op < nops; op++) {
            PlowStreamEnt* ent = &stream[(size_t)cu * nops + op];
            ent->inst = op;
            ent->slice = cu;
            if (!fused && op < 2) {
                ent->succ_ofs = op;
                ent->succ_len = 1;
            } else if (!fused) {
                ent->wait_len = 2;
            }
        }
    }
    BenchProgram p;
    memset(&p, 0, sizeof p);
    p.counters_bytes = fused ? sizeof(uint32_t) : 2u * PLOW_CTR_STRIDE * sizeof(uint32_t);
    p.insts = device_alloc(h, nops * sizeof *insts);
    p.stream = device_alloc(h, (size_t)ncu * nops * sizeof *stream);
    p.sofs = device_alloc(h, (size_t)ncu * sizeof *sofs);
    p.slen = device_alloc(h, (size_t)ncu * sizeof *slen);
    p.waits = device_alloc(h, fused ? sizeof(PlowWait) : sizeof waits);
    p.succs = device_alloc(h, fused ? sizeof(uint32_t) : sizeof succs);
    p.counters = device_alloc(h, p.counters_bytes);
    upload(h, p.insts, insts, nops * sizeof *insts, "upload instructions");
    upload(h, p.stream, stream, (size_t)ncu * nops * sizeof *stream, "upload stream");
    upload(h, p.sofs, sofs, (size_t)ncu * sizeof *sofs, "upload stream offsets");
    upload(h, p.slen, slen, (size_t)ncu * sizeof *slen, "upload stream lengths");
    if (!fused) {
        upload(h, p.waits, waits, sizeof waits, "upload waits");
        upload(h, p.succs, succs, sizeof succs, "upload successors");
    }
    p.prog.insts = p.insts;
    p.prog.stream = p.stream;
    p.prog.stream_ofs = p.sofs;
    p.prog.stream_len = p.slen;
    p.prog.waits = p.waits;
    p.prog.succs = p.succs;
    p.prog.counters = p.counters;
    p.prog.tensors = tensors;
    free(insts);
    free(stream);
    free(sofs);
    free(slen);
    return p;
}

static void reset_counters(plow_hsa* h, const BenchProgram* p) {
    uint32_t zeros[2 * PLOW_CTR_STRIDE] = {0};
    upload(h, p->counters, zeros, p->counters_bytes, "reset counters");
    if (plow_hsa_wait(h, 0)) fail_hsa("wait after counter reset");
}
static void run_once(plow_hsa* h, const plow_hsa_kernel* kernel, unsigned ncu,
                     const BenchProgram* p) {
    reset_counters(h, p);
    if (plow_hsa_launch(h, 0, kernel, ncu * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                        &p->prog, sizeof p->prog))
        fail_hsa("interpreter launch");
    if (plow_hsa_wait(h, 0)) fail_hsa("interpreter wait");
}
static double time_once(plow_hsa* h, const plow_hsa_kernel* kernel, unsigned ncu,
                        const BenchProgram* p) {
    reset_counters(h, p);
    const double t0 = now_ns();
    if (plow_hsa_launch(h, 0, kernel, ncu * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                        &p->prog, sizeof p->prog))
        fail_hsa("timed interpreter launch");
    if (plow_hsa_wait(h, 0)) fail_hsa("timed interpreter wait");
    return now_ns() - t0;
}

static const double fp4[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
static double weight(const uint8_t* w, const uint8_t* scale, unsigned n, unsigned kk,
                     unsigned k) {
    const size_t e = (size_t)n * k + kk;
    const uint8_t packed = w[e >> 1];
    const unsigned nibble = (kk & 1u) ? packed >> 4 : packed & 15u;
    return fp4[nibble] * ldexp(1.0, (int)scale[(size_t)n * (k / 32) + kk / 32] - 127);
}
static double silu(double x) {
    if (x >= 0.0) return x / (1.0 + exp(-x));
    const double ex = exp(x);
    return x * ex / (1.0 + ex);
}
static OracleStats check_oracle(const bf16* got, size_t count, const bf16* act,
                                const uint8_t* wg, const uint8_t* sg, const uint8_t* wu,
                                const uint8_t* su, unsigned n, unsigned k) {
    OracleStats s = {0.0, 0.0, 0, 1};
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(bf2d(got[i]))) {
            s.finite = 0;
            return s;
        }
    }
    const unsigned checks = count < 128 ? (unsigned)count : 128u;
    double se = 0.0, sr = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (unsigned q = 0; q < checks; q++) {
        size_t ix = q == 0 ? 0 : q == 1 ? count - 1
                                           : ((size_t)q * 0x9e3779b1u + q * q * 17u) % count;
        const unsigned row = (unsigned)(ix / n), col = (unsigned)(ix % n);
        double gate = 0.0, up = 0.0;
        for (unsigned kk = 0; kk < k; kk++) {
            const double a = bf2d(act[(size_t)row * k + kk]);
            gate += a * weight(wg, sg, col, kk, k);
            up += a * weight(wu, su, col, kk, k);
        }
        const double ref = silu(gate) * up;
        const double err = bf2d(got[ix]) - ref;
        se += err * err;
        sr += ref * ref;
        if (fabs(err) > max_abs) max_abs = fabs(err);
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
    }
    s.nrmse = sqrt(se / (sr + 1e-30));
    s.max_norm_abs = max_abs / (max_ref + 1e-12);
    s.checked = checks;
    return s;
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double percentile(const double* samples, unsigned count, double p) {
    double* copy = malloc((size_t)count * sizeof *copy);
    if (!copy) exit(1);
    memcpy(copy, samples, (size_t)count * sizeof *copy);
    qsort(copy, count, sizeof *copy, cmp_double);
    const double pos = p * (count - 1);
    const unsigned lo = (unsigned)pos, hi = lo + 1 < count ? lo + 1 : lo;
    const double out = copy[lo] + (copy[hi] - copy[lo]) * (pos - lo);
    free(copy);
    return out;
}
static void json_string(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c == '\n') fputs("\\n", f);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}
static void json_samples(FILE* f, const double* samples, unsigned count) {
    fputc('[', f);
    for (unsigned i = 0; i < count; i++) fprintf(f, "%s%.1f", i ? "," : "", samples[i]);
    fputc(']', f);
}
static void free_program(plow_hsa* h, const BenchProgram* p) {
    plow_hsa_free(h, p->insts);
    plow_hsa_free(h, p->stream);
    plow_hsa_free(h, p->sofs);
    plow_hsa_free(h, p->slen);
    plow_hsa_free(h, p->waits);
    plow_hsa_free(h, p->succs);
    plow_hsa_free(h, p->counters);
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s <prefill-interpreter.elf> <M> <N> <K> [samples>=10]\n",
                argv[0]);
        return 2;
    }
    const char* cleared = getenv("PLOW_STAGE4_CLEARED");
    const char* gpu = getenv("PLOW_GPU");
    const char* jsonl = getenv("PLOW_GEMM_GLU_JSONL");
    const char* toolchain = getenv("PLOW_TOOLCHAIN_LABEL");
    const char* build_id = getenv("PLOW_BUILD_ID");
    const char* lease = getenv("PLOW_LEASE_LABEL");
    if (!cleared || strcmp(cleared, "1") != 0) {
        fprintf(stderr, "REFUSED: set PLOW_STAGE4_CLEARED=1 only after Stages 1-3 pass\n");
        return 2;
    }
    if (!gpu || !*gpu || !jsonl || !*jsonl || !toolchain || !*toolchain || !build_id ||
        !*build_id || !lease || !*lease) {
        fprintf(stderr, "REFUSED: GPU, JSONL, toolchain, build-id, and lease provenance required\n");
        return 2;
    }
    if (strcmp(gpu, "MI325X") != 0 || strcmp(toolchain, "rocm-7.14.0-nix") != 0) {
        fprintf(stderr, "REFUSED: this campaign requires MI325X and rocm-7.14.0-nix\n");
        return 2;
    }
    const unsigned m = (unsigned)strtoul(argv[2], NULL, 10);
    const unsigned n = (unsigned)strtoul(argv[3], NULL, 10);
    const unsigned kdim = (unsigned)strtoul(argv[4], NULL, 10);
    const unsigned nsamples = argc == 6 ? (unsigned)strtoul(argv[5], NULL, 10) : 15u;
    if (!m || !n || !kdim || kdim % 64u || nsamples < 10u || (uint64_t)m * n > UINT32_MAX) {
        fprintf(stderr, "REFUSED: nonzero M/N/K, K%%64==0, M*N<=u32, samples>=10 required\n");
        return 2;
    }
    FILE* jf = fopen(jsonl, "a");
    if (!jf) {
        perror(jsonl);
        return 1;
    }

    plow_hsa* h = plow_hsa_init();
    if (!h) fail_hsa("HSA initialization");
    char arch[64];
    uint32_t ncu = 0, lds = 0;
    if (plow_hsa_device_info(h, 0, arch, &ncu, &lds)) fail_hsa("device info");
    if (strncmp(arch, "gfx942", 6)) {
        fprintf(stderr, "REFUSED: gfx942 required, got %s\n", arch);
        return 2;
    }

    FILE* object = fopen(argv[1], "rb");
    if (!object) {
        perror(argv[1]);
        return 1;
    }
    fseek(object, 0, SEEK_END);
    const long object_bytes = ftell(object);
    fseek(object, 0, SEEK_SET);
    void* image = object_bytes > 0 ? malloc((size_t)object_bytes) : NULL;
    if (!image || fread(image, 1, (size_t)object_bytes, object) != (size_t)object_bytes) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }
    fclose(object);
    uint64_t object_fnv1a = UINT64_C(14695981039346656037);
    for (long i = 0; i < object_bytes; i++) {
        object_fnv1a ^= ((const uint8_t*)image)[i];
        object_fnv1a *= UINT64_C(1099511628211);
    }
    if (plow_hsa_load_code_object(h, 0, image, (size_t)object_bytes)) fail_hsa("load object");
    plow_hsa_kernel kernel;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_gfx942", &kernel))
        fail_hsa("resolve plow_interp_gfx942");

    const size_t na = (size_t)m * kdim, nw = (size_t)n * kdim / 2;
    const size_t nscale = (size_t)n * (kdim / 32), no = (size_t)m * n;
    bf16* ha = plow_hsa_alloc_host(h, na * sizeof *ha);
    uint8_t* hwg = plow_hsa_alloc_host(h, nw);
    uint8_t* hwu = plow_hsa_alloc_host(h, nw);
    uint8_t* hsg = plow_hsa_alloc_host(h, nscale);
    uint8_t* hsu = plow_hsa_alloc_host(h, nscale);
    bf16* hout = plow_hsa_alloc_host(h, no * sizeof *hout);
    bf16* hfused = malloc(no * sizeof *hfused);
    bf16* hunfused = malloc(no * sizeof *hunfused);
    if (!ha || !hwg || !hwu || !hsg || !hsu || !hout || !hfused || !hunfused) {
        fprintf(stderr, "operand allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < na; i++) ha[i] = f2bf(((int)(rng_u32() % 33u) - 16) / 64.0f);
    for (size_t i = 0; i < nw; i++) {
        hwg[i] = (uint8_t)rng_u32();
        hwu[i] = (uint8_t)rng_u32();
    }
    for (size_t i = 0; i < nscale; i++) {
        hsg[i] = (uint8_t)(123u + rng_u32() % 3u);
        hsu[i] = (uint8_t)(123u + rng_u32() % 3u);
    }

    void* dout = device_alloc(h, no * sizeof(bf16));
    void* dact = device_alloc(h, na * sizeof(bf16));
    void* dwg = device_alloc(h, nw);
    void* dsg = device_alloc(h, nscale);
    void* dsu = device_alloc(h, nscale);
    void* dwu = device_alloc(h, nw);
    void* dup = device_alloc(h, no * sizeof(bf16));
    if (plow_hsa_copy_h2d(h, 0, dact, ha, na * sizeof *ha) ||
        plow_hsa_copy_h2d(h, 0, dwg, hwg, nw) || plow_hsa_copy_h2d(h, 0, dwu, hwu, nw) ||
        plow_hsa_copy_h2d(h, 0, dsg, hsg, nscale) || plow_hsa_copy_h2d(h, 0, dsu, hsu, nscale))
        fail_hsa("operand upload");
    void* tensors[N_TENSORS] = {dout, dact, dwg, dsg, dsu, dwu, dup};
    void* dtensors = device_alloc(h, sizeof tensors);
    upload(h, dtensors, tensors, sizeof tensors, "upload tensor table");

    BenchProgram fused = make_program(h, ncu, 1, m, n, kdim, dtensors);
    BenchProgram unfused = make_program(h, ncu, 0, m, n, kdim, dtensors);
    for (size_t i = 0; i < no; i++) hout[i] = (bf16)0x7fc0u;
    if (plow_hsa_copy_h2d(h, 0, dout, hout, no * sizeof *hout)) fail_hsa("fused sentinel");
    run_once(h, &kernel, ncu, &fused);
    if (plow_hsa_copy_d2h(h, 0, hout, dout, no * sizeof *hout)) fail_hsa("fused result");
    memcpy(hfused, hout, no * sizeof *hout);
    const OracleStats fo = check_oracle(hfused, no, ha, hwg, hsg, hwu, hsu, n, kdim);

    for (size_t i = 0; i < no; i++) hout[i] = (bf16)0x7fc0u;
    if (plow_hsa_copy_h2d(h, 0, dout, hout, no * sizeof *hout) ||
        plow_hsa_copy_h2d(h, 0, dup, hout, no * sizeof *hout))
        fail_hsa("unfused sentinel");
    run_once(h, &kernel, ncu, &unfused);
    if (plow_hsa_copy_d2h(h, 0, hout, dout, no * sizeof *hout)) fail_hsa("unfused result");
    memcpy(hunfused, hout, no * sizeof *hout);
    const OracleStats uo = check_oracle(hunfused, no, ha, hwg, hsg, hwu, hsu, n, kdim);
    const int correct = fo.finite && uo.finite && fo.nrmse < 0.05 && uo.nrmse < 0.05 &&
                        fo.max_norm_abs < 0.08 && uo.max_norm_abs < 0.08 &&
                        fo.nrmse <= uo.nrmse * 1.10 + 0.002;
    if (!correct) {
        fprintf(stderr,
                "ORACLE FAIL: fused finite=%d nrmse=%.4e max=%.4e; unfused finite=%d "
                "nrmse=%.4e max=%.4e\n",
                fo.finite, fo.nrmse, fo.max_norm_abs, uo.finite, uo.nrmse, uo.max_norm_abs);
        return 1;
    }

    for (unsigned w = 0; w < 10; w++) run_once(h, &kernel, ncu, (w & 1u) ? &unfused : &fused);
    double* fused_ns = malloc((size_t)nsamples * sizeof *fused_ns);
    double* unfused_ns = malloc((size_t)nsamples * sizeof *unfused_ns);
    if (!fused_ns || !unfused_ns) return 1;
    for (unsigned s = 0; s < nsamples; s++) {
        if (s & 1u) {
            unfused_ns[s] = time_once(h, &kernel, ncu, &unfused);
            fused_ns[s] = time_once(h, &kernel, ncu, &fused);
        } else {
            fused_ns[s] = time_once(h, &kernel, ncu, &fused);
            unfused_ns[s] = time_once(h, &kernel, ncu, &unfused);
        }
    }
    const double fm = percentile(fused_ns, nsamples, 0.5);
    const double um = percentile(unfused_ns, nsamples, 0.5);
    printf("gfx942 CUs=%u tile=192x256x64 M=%u N=%u K=%u samples=%u\n", ncu, m, n, kdim,
           nsamples);
    printf("  fused   median %.4f ms p10 %.4f p90 %.4f\n", fm / 1e6,
           percentile(fused_ns, nsamples, 0.1) / 1e6, percentile(fused_ns, nsamples, 0.9) / 1e6);
    printf("  unfused median %.4f ms p10 %.4f p90 %.4f\n", um / 1e6,
           percentile(unfused_ns, nsamples, 0.1) / 1e6,
           percentile(unfused_ns, nsamples, 0.9) / 1e6);
    printf("  fused delta %+.2f%%; oracle fused nrmse %.3e, unfused %.3e\n",
           100.0 * (fm - um) / um, fo.nrmse, uo.nrmse);

    char stamp[32];
    time_t wall = time(NULL);
    struct tm utc;
    gmtime_r(&wall, &utc);
    strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &utc);
    fputs("{\"schema\":\"plow.interp_mxfp4_glu.v1\",\"timestamp\":", jf);
    json_string(jf, stamp);
    fputs(",\"gpu\":", jf);
    json_string(jf, gpu);
    fputs(",\"arch\":", jf);
    json_string(jf, arch);
    fprintf(jf,
            ",\"cu_count\":%u,\"tile\":\"192x256x64\",\"object_bytes\":%ld,"
            "\"object_fnv1a64\":\"%016llx\",\"object\":",
            ncu, object_bytes, (unsigned long long)object_fnv1a);
    json_string(jf, argv[1]);
    fputs(",\"kernel\":\"plow_interp_gfx942\",\"toolchain\":", jf);
    json_string(jf, toolchain);
    fputs(",\"build_id\":", jf);
    json_string(jf, build_id);
    fputs(",\"lease_label\":", jf);
    json_string(jf, lease);
    fprintf(jf,
            ",\"stage4_cleared\":true,\"m\":%u,\"n\":%u,\"k\":%u,\"samples\":%u,"
            "\"fused_op\":113,\"unfused_ops\":[93,93,5],\"correct\":true,"
            "\"oracle\":\"full-bf16-finite-sentinel+128xf64-mxfp4\","
            "\"oracle_checked\":%u,\"fused_nrmse\":%.9g,\"unfused_nrmse\":%.9g,"
            "\"fused_max_norm_abs\":%.9g,\"unfused_max_norm_abs\":%.9g,"
            "\"fused_samples_ns\":",
            m, n, kdim, nsamples, fo.checked, fo.nrmse, uo.nrmse, fo.max_norm_abs,
            uo.max_norm_abs);
    json_samples(jf, fused_ns, nsamples);
    fputs(",\"unfused_samples_ns\":", jf);
    json_samples(jf, unfused_ns, nsamples);
    fputs("}\n", jf);
    fclose(jf);

    free(fused_ns);
    free(unfused_ns);
    free(hfused);
    free(hunfused);
    free_program(h, &fused);
    free_program(h, &unfused);
    plow_hsa_free(h, dout);
    plow_hsa_free(h, dact);
    plow_hsa_free(h, dwg);
    plow_hsa_free(h, dsg);
    plow_hsa_free(h, dsu);
    plow_hsa_free(h, dwu);
    plow_hsa_free(h, dup);
    plow_hsa_free(h, dtensors);
    plow_hsa_shutdown(h);
    free(image);
    return 0;
}
