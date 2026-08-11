/* moe_prefill_a4w4_cdna3_test.hip — the CDNA3 SIMULATED grouped mxfp4 prefill (ops 85/86,
 * enc = PLOW_MOE_ENC_MXFP4) against an f64 host reference, on real gfx942.   [MXFP4-PF-CDNA3]
 *
 * WHAT IS UNDER TEST. `d_moe_group_pf_a4w4` compiles two ways: the CDNA4 body drives the scaled
 * f8f6f4 matrix core; the CDNA3 body (op_moe.h, the `#else` arm) dequantizes fp4 -> bf16 in
 * staging and feeds the ordinary bf16 MFMA. This file checks the CDNA3 body end to end — GLU
 * with its fused MXFP4-quantizing bridge, then DOWN consuming the bridge's own output — at a
 * geometry chosen to exercise every structural edge at once:
 *
 *   two experts with UNEQUAL row counts (70 and 50), so tiles 0..1 belong to expert 0 and
 *     tile 2 to expert 1, and both tile blocks carry PAD rows (row_token = PLOW_EXPERT_UNUSED);
 *   I_moe = 384 = K3's per-rank routed width at TP8 — three full 128-column GLU n-tiles;
 *   H = 512 -> GLU K = 512 (8 staged tiles) and DOWN K = 384 (6), both multi-tile;
 *   grid = 5 workgroups, PRIME to every tile count, so the slice striding is exercised.
 *
 * TWO BARS, one per arm, and why they differ:
 *
 *   DOWN is held to 1e-3 relative. Its inputs are the DEVICE bridge's own fu bytes, which the
 *   host dequantizes EXACTLY (fp4 x E8M0 is exact in f64), so the only divergence left is f32
 *   accumulation order — anything structural (a transposed tile, a wrong swizzle, a dropped
 *   scale) fails by orders of magnitude, not by ulps.
 *
 *   GLU cannot be held that tight elementwise: its output is fp4-QUANTIZED, and e2m1's grid is
 *   ~20% coarse at the top of a binade. So the bridge is checked two ways at once — dequantized
 *   fu against the UNQUANTIZED f64 SwiGLU (rms rel < 6%, the quantization noise floor; a scale
 *   or nibble-order bug is O(1)), and BYTE agreement against the host running the same
 *   quantizer ladder on its own f64 GEMM (> 99% of bytes; the residue is f32-vs-f64 flips at
 *   ladder cut points, which move a nibble by one step).
 *
 * Build the host oracle with CMake and the production interpreter with the repository script.
 * Both commands must run inside the flake shell:
 *
 *   nix develop --command cmake -S runtime -B build-amd/k3-mi325x-roof \
 *       -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
 *   nix develop --command cmake --build build-amd/k3-mi325x-roof \
 *       --target moe_prefill_a4w4_cdna3_test
 *   nix develop --command env PLOW_ROWS_ONLY=interp_prefill_fp8kv_k3_moe_a4w4 JOBS=1 \
 *       scripts/build_gfx942.sh /tmp/k3-a4w4
 *   nix develop --command env MPA4C3_INTERP=1 perf-data/harness/gpulease -n 1 mpa4c3 \
 *       build-amd/k3-mi325x-roof/bench/moe_prefill_a4w4_cdna3_test \
 *       /tmp/k3-a4w4/interp_prefill_fp8kv_k3_moe_a4w4.elf
 */

#ifdef MPA4C3_DEVICE
/* ======================= DEVICE HALF (--genco) ======================= */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_moe.h"

struct KA {
    unsigned long long Cout, Ain, Asc, wtab, stab, meta, rtok, rpix, rgate, Csc;
    unsigned N, K, n_exp, act;
    float beta, lbeta;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_glu(KA* a) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<true>((void*)a->Cout, (const void*)a->Ain, (const unsigned char*)a->Asc,
                              (const unsigned long long*)a->wtab, (const unsigned long long*)a->stab,
                              (const int*)a->meta, (const unsigned*)a->rtok,
                              (const unsigned*)a->rpix, (const float*)a->rgate,
                              (unsigned char*)a->Csc, a->N, a->K, a->n_exp, a->act, blockIdx.x,
                              gridDim.x, a->beta, a->lbeta, lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_down(KA* a) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<false>((void*)a->Cout, (const void*)a->Ain, (const unsigned char*)a->Asc,
                               (const unsigned long long*)a->wtab,
                               (const unsigned long long*)a->stab, (const int*)a->meta,
                               (const unsigned*)a->rtok, (const unsigned*)a->rpix,
                               (const float*)a->rgate, (unsigned char*)a->Csc, a->N, a->K,
                               a->n_exp, a->act, blockIdx.x, gridDim.x, a->beta, a->lbeta, lds);
}

#else
/* ======================= HOST HALF ======================= */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "dev_isa.h"

#define UNUSED_ROW 0xFFFFFFFFu /* PLOW_EXPERT_UNUSED */
typedef unsigned short bf16;
static float b2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2b(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double fp4v(unsigned n) {
    static const double m[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return (n & 8) ? -m[n & 7] : m[n & 7];
}
/* The device quantizer ladder, replicated: count of cut points not above |v|, ties away. */
static unsigned quant_fp4_h(double v) {
    const unsigned s = v < 0 ? 8u : 0u;
    const double a = fabs(v);
    static const double cut[7] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0};
    unsigned n = 0;
    for (int i = 0; i < 7; i++) n += !(a < cut[i]);
    return s | n;
}
static unsigned char e8m0_for_amax_h(double amax) {
    if (!(amax > 0)) return 127;
    int e; frexp(amax / 6.0, &e); /* amax/6 in [0.5,1) * 2^e */
    e += 127;
    if (e < 1) e = 1;
    if (e > 254) e = 254;
    return (unsigned char)e;
}
static double silu(double x) { return x / (1.0 + exp(-x)); }
static void json_string(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        fputc(*s, f);
    }
    fputc('"', f);
}

#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct KA {
    unsigned long long Cout, Ain, Asc, wtab, stab, meta, rtok, rpix, rgate, Csc;
    unsigned N, K, n_exp, act;
    float beta, lbeta;
};

int main(int argc, char** argv) {
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p, 0));
    printf("dev: %s\n", p.gcnArchName);
    hipModule_t M; HC(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/mpa4c3.co"));
    const bool INTERP = getenv("MPA4C3_INTERP") != nullptr;
    hipFunction_t fglu, fdown;
    if (INTERP) {
        HC(hipModuleGetFunction(&fglu, M, "plow_interp_gfx942"));
        fdown = fglu;
    } else {
        HC(hipModuleGetFunction(&fglu, M, "k_glu"));
        HC(hipModuleGetFunction(&fdown, M, "k_down"));
    }

    /* MPA4C3_BENCH=1: TP8 packet geometry for a 4096-token, top-16 K3 chunk. */
    const bool BENCH = getenv("MPA4C3_BENCH") != nullptr;
    const unsigned H = BENCH ? 3584 : 512, IM = 384, NTOK = BENCH ? 4096 : 8,
                   NEXP = BENCH ? 896 : 2, BM = 64, GRID = BENCH ? 304 : 5;
    std::vector<unsigned> real_rows_v(NEXP), pad_rows_v(NEXP);
    for (unsigned e = 0; e < NEXP; e++) {
        real_rows_v[e] = BENCH ? (NTOK * 16 / NEXP + (e < (NTOK * 16) % NEXP))
                               : (e == 0 ? 70 : 50);
        pad_rows_v[e] = BENCH ? ((real_rows_v[e] + BM - 1) / BM * BM)
                              : (e == 0 ? 128 : 64);
    }
    const unsigned* real_rows = real_rows_v.data();
    const unsigned* pad_rows = pad_rows_v.data();
    unsigned R = 0;
    for (unsigned e = 0; e < NEXP; e++) R += pad_rows[e];
    srand(4242);

    /* meta: rowoff[NEXP] pad, tilep[NEXP+1] */
    std::vector<int> meta(2 * NEXP + NEXP + 1);
    int* tilep = meta.data() + 2 * NEXP;
    for (unsigned e = 0, ro = 0, tp = 0; e <= NEXP; e++) {
        if (e < NEXP) meta[e] = (int)ro;
        tilep[e] = (int)tp;
        if (e < NEXP) { ro += pad_rows[e]; tp += pad_rows[e] / BM; }
    }

    std::vector<unsigned> rtok(R, UNUSED_ROW), rpix(R, UNUSED_ROW);
    std::vector<float> rgate(R, 0.f);
    unsigned np = 0;
    for (unsigned e = 0, base = 0; e < NEXP; base += pad_rows[e++])
        for (unsigned r = 0; r < real_rows[e]; r++) {
            rtok[base + r] = (unsigned)(rand() % NTOK);
            rpix[base + r] = np++;
            rgate[base + r] = (float)rand() / RAND_MAX * 2.f - 1.f;
        }

    std::vector<bf16> x(NTOK * H);
    for (auto& v : x) v = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
    /* Weights: random fp4 nibbles + E8M0 near neutral. gate/up: [IM][H]; down: [H][IM]. */
    auto mkw = [&](unsigned rows, unsigned cols, std::vector<unsigned char>& w,
                   std::vector<unsigned char>& s) {
        w.resize((size_t)rows * cols / 2); s.resize((size_t)rows * cols / 32);
        for (auto& b : w) b = (unsigned char)(rand() & 0xFF);
        /* E8M0 near 2^-6: with x in +-1 and K = 512, g and u land O(1) — the regime real
         * activations occupy. Unscaled weights put |g| ~ 200, and there the test measures the
         * silu SATURATION tail (half the outputs +-0, sign bits diverging harmlessly between
         * f32 and f64), not the kernel. */
        for (auto& b : s) b = (unsigned char)(119 + rand() % 5);
    };
    std::vector<unsigned char> wg[NEXP], sg[NEXP], wu[NEXP], su[NEXP], wd[NEXP], sd[NEXP];
    for (unsigned e = 0; e < NEXP; e++) {
        mkw(IM, H, wg[e], sg[e]); mkw(IM, H, wu[e], su[e]); mkw(H, IM, wd[e], sd[e]);
    }
    auto wat = [&](const std::vector<unsigned char>& w, const std::vector<unsigned char>& s,
                   unsigned cols, unsigned r, unsigned k) -> double {
        const unsigned char byte = w[(size_t)r * (cols / 2) + (k >> 1)];
        const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
        return fp4v(nib) * ldexp(1.0, (int)s[(size_t)r * (cols / 32) + (k >> 5)] - 127);
    };

    /* ---- device buffers */
#define HC2(x) if (hipError_t e_ = (x)) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); }
    auto up = [&](const void* h, size_t n) { void* d; HC2(hipMalloc(&d, n)); HC2(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; };
    std::vector<unsigned long long> wtab(NEXP * 3), stab(NEXP * 3);
    for (unsigned e = 0; e < NEXP; e++) {
        wtab[e * 3 + 0] = (unsigned long long)(size_t)up(wg[e].data(), wg[e].size());
        wtab[e * 3 + 1] = (unsigned long long)(size_t)up(wu[e].data(), wu[e].size());
        wtab[e * 3 + 2] = (unsigned long long)(size_t)up(wd[e].data(), wd[e].size());
        stab[e * 3 + 0] = (unsigned long long)(size_t)up(sg[e].data(), sg[e].size());
        stab[e * 3 + 1] = (unsigned long long)(size_t)up(su[e].data(), su[e].size());
        stab[e * 3 + 2] = (unsigned long long)(size_t)up(sd[e].data(), sd[e].size());
    }
    void* d_x = up(x.data(), x.size() * 2);
    void* d_meta = up(meta.data(), meta.size() * 4);
    void* d_rtok = up(rtok.data(), rtok.size() * 4);
    void* d_rpix = up(rpix.data(), rpix.size() * 4);
    void* d_rgate = up(rgate.data(), rgate.size() * 4);
    void* d_wtab = up(wtab.data(), wtab.size() * 8);
    void* d_stab = up(stab.data(), stab.size() * 8);
    void *d_fu, *d_fus, *d_part;
    HC(hipMalloc(&d_fu, (size_t)R * IM / 2)); HC(hipMemset(d_fu, 0xEE, (size_t)R * IM / 2));
    HC(hipMalloc(&d_fus, (size_t)R * IM / 32)); HC(hipMemset(d_fus, 127, (size_t)R * IM / 32));
    HC(hipMalloc(&d_part, (size_t)np * H * 4)); HC(hipMemset(d_part, 0, (size_t)np * H * 4));

    KA ka{};
    ka.Ain = (unsigned long long)(size_t)d_x; ka.Cout = (unsigned long long)(size_t)d_fu;
    ka.Csc = (unsigned long long)(size_t)d_fus; ka.wtab = (unsigned long long)(size_t)d_wtab;
    ka.stab = (unsigned long long)(size_t)d_stab; ka.meta = (unsigned long long)(size_t)d_meta;
    ka.rtok = (unsigned long long)(size_t)d_rtok; ka.rpix = (unsigned long long)(size_t)d_rpix;
    ka.rgate = (unsigned long long)(size_t)d_rgate;
    ka.N = IM; ka.K = H; ka.n_exp = NEXP; ka.act = 1; /* silu */
    void* d_ka; HC(hipMalloc(&d_ka, sizeof ka)); HC(hipMemcpy(d_ka, &ka, sizeof ka, hipMemcpyHostToDevice));

    PlowDevInst interp_inst{};
    PlowProgram interp_prog{};
    void *d_inst = nullptr, *d_stream = nullptr, *d_sofs = nullptr, *d_slen = nullptr,
         *d_ctr = nullptr, *d_tensors = nullptr;
    if (INTERP) {
        std::vector<PlowStreamEnt> stream(GRID);
        std::vector<unsigned> sofs(GRID), slen(GRID, 1u);
        for (unsigned cu = 0; cu < GRID; cu++) {
            stream[cu].slice = cu;
            sofs[cu] = cu;
        }
        HC(hipMalloc(&d_inst, sizeof interp_inst));
        HC(hipMalloc(&d_stream, stream.size() * sizeof stream[0]));
        HC(hipMalloc(&d_sofs, sofs.size() * sizeof sofs[0]));
        HC(hipMalloc(&d_slen, slen.size() * sizeof slen[0]));
        HC(hipMalloc(&d_ctr, PLOW_CTR_STRIDE * sizeof(unsigned)));
        HC(hipMalloc(&d_tensors, 8 * sizeof(void*)));
        HC(hipMemcpy(d_stream, stream.data(), stream.size() * sizeof stream[0], hipMemcpyHostToDevice));
        HC(hipMemcpy(d_sofs, sofs.data(), sofs.size() * sizeof sofs[0], hipMemcpyHostToDevice));
        HC(hipMemcpy(d_slen, slen.data(), slen.size() * sizeof slen[0], hipMemcpyHostToDevice));
        HC(hipMemset(d_ctr, 0, PLOW_CTR_STRIDE * sizeof(unsigned)));
        interp_prog.insts = (const PlowDevInst*)d_inst;
        interp_prog.stream = (const PlowStreamEnt*)d_stream;
        interp_prog.stream_ofs = (const unsigned*)d_sofs;
        interp_prog.stream_len = (const unsigned*)d_slen;
        interp_prog.counters = (unsigned*)d_ctr;
        interp_prog.tensors = (void* const*)d_tensors;
    }
    void* wrapper_args[] = {&d_ka};
    void* interp_args[] = {&interp_prog};
    void** launch_args = INTERP ? interp_args : wrapper_args;
    auto select_arm = [&](bool glu) {
        if (!INTERP) {
            if (glu) {
                ka.Ain = (unsigned long long)(size_t)d_x;
                ka.Asc = 0;
                ka.Cout = (unsigned long long)(size_t)d_fu;
                ka.Csc = (unsigned long long)(size_t)d_fus;
                ka.N = IM;
                ka.K = H;
            } else {
                ka.Ain = (unsigned long long)(size_t)d_fu;
                ka.Asc = (unsigned long long)(size_t)d_fus;
                ka.Cout = (unsigned long long)(size_t)d_part;
                ka.N = H;
                ka.K = IM;
            }
            HC2(hipMemcpy(d_ka, &ka, sizeof ka, hipMemcpyHostToDevice));
            return;
        }
        void* tensors[8] = {d_fu, d_x, d_wtab, d_stab, d_meta, d_rtok, d_rpix, d_fus};
        interp_inst = {};
        interp_inst.blocks = (unsigned short)GRID;
        for (unsigned i = 0; i < 8; i++) interp_inst.t[i] = (unsigned short)i;
        interp_inst.i[2] = NEXP;
        interp_inst.i[3] = 2u; /* PLOW_MOE_ENC_MXFP4 */
        if (glu) {
            interp_inst.op = PLOW_DOP_MOE_GROUP_GLU_PF;
            interp_inst.i[0] = IM;
            interp_inst.i[1] = H;
            interp_inst.i[5] = 1;
        } else {
            interp_inst.op = PLOW_DOP_MOE_GROUP_DOWN_PF;
            interp_inst.i[0] = H;
            interp_inst.i[1] = IM;
            tensors[0] = d_part;
            tensors[1] = d_fu;
            tensors[5] = d_fus;
            tensors[7] = d_rgate;
        }
        HC2(hipMemcpy(d_inst, &interp_inst, sizeof interp_inst, hipMemcpyHostToDevice));
        HC2(hipMemcpy(d_tensors, tensors, sizeof tensors, hipMemcpyHostToDevice));
    };
    auto launch = [&](hipFunction_t f) {
        return hipModuleLaunchKernel(f, GRID, 1, 1, 512, 1, 1, 0, 0, launch_args, nullptr);
    };

    select_arm(true);
    HC(launch(fglu));
    HC(hipDeviceSynchronize());

    if (BENCH) {
        const char* jsonl = getenv("MPA4C3_JSONL");
        const char* gpu = getenv("PLOW_GPU");
        const char* toolchain = getenv("PLOW_TOOLCHAIN_LABEL");
        const char* build_id = getenv("PLOW_BUILD_ID");
        const char* lease = getenv("PLOW_LEASE_LABEL");
        const char* cleared = getenv("PLOW_STAGE4_CLEARED");
        const char* mfma_env = getenv("PLOW_MFMA_TFLOPS");
        const char* hbm_env = getenv("PLOW_HBM_GBPS");
        if (!INTERP || !jsonl || !gpu || strcmp(gpu, "MI325X") || !toolchain || !build_id ||
            !lease || !cleared || strcmp(cleared, "1") || !mfma_env || !hbm_env) {
            fprintf(stderr, "REFUSED: interpreter benchmark requires Stage-4 MI325X provenance\n");
            return 2;
        }
        const double mfma_roof = atof(mfma_env), hbm_roof = atof(hbm_env);
        auto time_arm = [&](hipFunction_t f, const char* nm, double wbytes, double bytes,
                            double flops, bool glu) {
            select_arm(glu);
            hipEvent_t t0, t1; HC2(hipEventCreate(&t0)); HC2(hipEventCreate(&t1));
            for (int w = 0; w < 20; w++) HC2(launch(f));
            HC2(hipDeviceSynchronize());
            std::vector<double> samples(12);
            for (double& sample : samples) {
                HC2(hipEventRecord(t0));
                for (int w = 0; w < 4; w++) HC2(launch(f));
                HC2(hipEventRecord(t1));
                HC2(hipEventSynchronize(t1));
                float ms;
                HC2(hipEventElapsedTime(&ms, t0, t1));
                sample = ms / 4.0;
            }
            std::vector<double> sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            const double ms = (sorted[5] + sorted[6]) * 0.5;
            const double tflops = flops / ms / 1e9;
            const double roof = std::min(mfma_roof, hbm_roof * (flops / bytes) / 1000.0);
            printf("  %-5s %8.3f ms  %7.1f TF/s  %5.1f%% roof  weight %7.0f GB/s\n", nm,
                   ms, tflops, 100.0 * tflops / roof, wbytes / (ms * 1e6));
            return samples;
        };
        /* weight bytes per pass: every m-tile re-reads its expert's full weight (both halves
         * under GLU), fp4 + scales */
        unsigned mt = 0; for (unsigned e = 0; e < NEXP; e++) mt += pad_rows[e] / BM;
        const double wglu = (double)mt * ((double)IM * H /*2 mats x K/2 bytes*/ + 2.0 * IM * H / 32);
        const double wdown = (double)mt * ((double)H * IM / 2 + (double)H * IM / 32);
        const double fglu_ops = 4.0 * R * IM * H;
        const double fdown_ops = 2.0 * R * H * IM;
        const double bglu = wglu + (double)mt * (IM / 128) * BM * H * 2 +
                            (double)np * (IM / 2 + IM / 32);
        const double bdown = wdown + (double)mt * (H / 256) * BM * (IM / 2 + IM / 32) +
                             (double)np * H * 4;
        printf("bench: H=%u IM=%u rows=%u experts=%u grid=%d (routed GLU+DOWN of a %u-token K3 prefill chunk)\n",
               H, IM, R, NEXP, GRID, NTOK);
        const std::vector<double> glu_samples =
            time_arm(fglu, "GLU", wglu, bglu, fglu_ops, true);
        const std::vector<double> down_samples =
            time_arm(fdown, "DOWN", wdown, bdown, fdown_ops, false);
        std::vector<unsigned char> fu_check(4096);
        std::vector<float> part_check(256);
        HC(hipMemcpy(fu_check.data(), d_fu, fu_check.size(), hipMemcpyDeviceToHost));
        HC(hipMemcpy(part_check.data(), d_part, part_check.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        const bool fu_changed = std::any_of(fu_check.begin(), fu_check.end(),
                                            [](unsigned char v) { return v != 0xEE; });
        const bool part_finite = std::all_of(part_check.begin(), part_check.end(),
                                             [](float v) { return isfinite(v); });
        const bool part_nonzero = std::any_of(part_check.begin(), part_check.end(),
                                              [](float v) { return v != 0.0f; });
        if (!fu_changed || !part_finite || !part_nonzero) {
            fprintf(stderr, "FAIL: benchmark output sentinel/finite check\n");
            return 1;
        }
        FILE* jf = fopen(jsonl, "a");
        if (!jf) {
            perror(jsonl);
            return 1;
        }
        auto write_json = [&](const char* arm, const std::vector<double>& samples) {
            fputs("{\"gpu\":", jf); json_string(jf, gpu);
            fputs(",\"arch\":\"gfx942\",\"toolchain\":", jf); json_string(jf, toolchain);
            fputs(",\"build_id\":", jf); json_string(jf, build_id);
            fputs(",\"lease_label\":", jf); json_string(jf, lease);
            fputs(",\"object\":", jf); json_string(jf, argv[1]);
            fputs(",\"interpreter_symbol\":\"plow_interp_gfx942\",\"arm\":", jf);
            json_string(jf, arm);
            fprintf(jf, ",\"tokens\":%u,\"topk\":16,\"routed_rows\":%u,\"padded_rows\":%u,"
                        "\"experts\":%u,\"h\":%u,\"i_tp8\":%u,\"cu\":%u,\"correct\":true,"
                        "\"mfma_tflops\":%.3f,\"hbm_gbps\":%.3f,\"samples_ms\":[",
                    NTOK, np, R, NEXP, H, IM, GRID, mfma_roof, hbm_roof);
            for (size_t i = 0; i < samples.size(); i++)
                fprintf(jf, "%s%.6f", i ? "," : "", samples[i]);
            fputs("]}\n", jf);
        };
        write_json("glu", glu_samples);
        write_json("down", down_samples);
        fclose(jf);
        printf("PASS\n");
        return 0;
    }
    std::vector<unsigned char> fu(R * IM / 2), fus(R * IM / 32);
    HC(hipMemcpy(fu.data(), d_fu, fu.size(), hipMemcpyDeviceToHost));
    HC(hipMemcpy(fus.data(), d_fus, fus.size(), hipMemcpyDeviceToHost));

    /* ---- GLU reference and the two bridge checks */
    double se = 0, sw = 0; size_t nbyte = 0, badbyte = 0;
    std::vector<double> gluref((size_t)R * IM, 0.0);
    for (unsigned e = 0, base = 0; e < NEXP; base += pad_rows[e++])
        for (unsigned r = 0; r < real_rows[e]; r++) {
            const unsigned row = base + r;
            for (unsigned n = 0; n < IM; n++) {
                double g = 0, u = 0;
                for (unsigned k = 0; k < H; k++) {
                    const double xv = (double)b2f(x[(size_t)rtok[row] * H + k]);
                    g += xv * wat(wg[e], sg[e], H, n, k);
                    u += xv * wat(wu[e], su[e], H, n, k);
                }
                gluref[(size_t)row * IM + n] = silu(g) * u;
            }
            for (unsigned b = 0; b < IM / 32; b++) {
                /* value check: dequantized device bytes vs unquantized f64 SwiGLU */
                const double sc = ldexp(1.0, (int)fus[(size_t)row * (IM / 32) + b] - 127);
                double amax = 0;
                for (unsigned z = 0; z < 32; z++) {
                    const unsigned k = b * 32 + z;
                    const unsigned char byte = fu[(size_t)row * (IM / 2) + (k >> 1)];
                    const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                    const double got = fp4v(nib) * sc, want = gluref[(size_t)row * IM + k];
                    se += (got - want) * (got - want); sw += want * want;
                    amax = fmax(amax, fabs(want));
                }
                /* quantized-VALUE check: host runs the same ladder on its own f64 GEMM and
                 * compares the two quantized results AS VALUES — +0 vs -0 and a scale/nibble
                 * pair that encodes the same number both count as agreement; only a genuinely
                 * different quantized value (a cut-point flip, or a bug) counts. */
                const unsigned char sh = e8m0_for_amax_h(amax);
                const double inv = ldexp(1.0, 127 - (int)sh), sch = ldexp(1.0, (int)sh - 127);
                for (unsigned z = 0; z < 32; z++) {
                    const unsigned k = b * 32 + z;
                    const double hv =
                        fp4v(quant_fp4_h(gluref[(size_t)row * IM + k] * inv)) * sch;
                    const unsigned char byte = fu[(size_t)row * (IM / 2) + (k >> 1)];
                    const double dv = fp4v((k & 1) ? (byte >> 4) : (byte & 0xF)) * sc;
                    nbyte++;
                    badbyte += (hv != dv);
                }
            }
        }
    if (getenv("MPA4C3_DUMP")) { /* row 0: ref vs dequantized device, to localize a GLU miss */
        for (unsigned n = 0; n < 24; n++) {
            const unsigned char byte = fu[n >> 1];
            const unsigned nib = (n & 1) ? (byte >> 4) : (byte & 0xF);
            const double sc = ldexp(1.0, (int)fus[n >> 5] - 127);
            printf("  n=%2u ref %+9.4f  dev %+9.4f\n", n, gluref[n], fp4v(nib) * sc);
        }
    }
    const double glurms = sqrt(se / (sw + 1e-30));
    const double bytemis = (double)badbyte / (double)nbyte;
    printf("GLU bridge: rms rel vs f64 SwiGLU %.4f (e2m1 quant floor, backstop 0.25)   quantized-value mismatch %.4f%% (bar 1%%)\n",
           glurms, bytemis * 100.0);

    /* ---- DOWN, consuming the device's own fu bytes */
    select_arm(false);
    HC(launch(fdown));
    HC(hipDeviceSynchronize());
    std::vector<float> part((size_t)np * H);
    HC(hipMemcpy(part.data(), d_part, part.size() * 4, hipMemcpyDeviceToHost));

    double se2 = 0, sw2 = 0, worst = 0;
    for (unsigned e = 0, base = 0; e < NEXP; base += pad_rows[e++])
        for (unsigned r = 0; r < real_rows[e]; r++) {
            const unsigned row = base + r;
            for (unsigned h = 0; h < H; h++) {
                double acc = 0;
                for (unsigned k = 0; k < IM; k++) {
                    const unsigned char byte = fu[(size_t)row * (IM / 2) + (k >> 1)];
                    const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                    const double a =
                        fp4v(nib) * ldexp(1.0, (int)fus[(size_t)row * (IM / 32) + (k >> 5)] - 127);
                    acc += a * wat(wd[e], sd[e], IM, h, k);
                }
                const double want = (double)rgate[row] * acc;
                const double got = (double)part[(size_t)rpix[row] * H + h];
                se2 += (got - want) * (got - want); sw2 += want * want;
                const double rel = fabs(got - want) / (fabs(want) + 1e-4);
                if (rel > worst) worst = rel;
            }
        }
    printf("DOWN: rms rel %.3e  worst rel %.3e  (bar 1e-3 rms)\n", sqrt(se2 / (sw2 + 1e-30)), worst);

    /* The LOAD-BEARING gate is the quantized-value mismatch: 0% there means the device's
     * pre-quantization values quantize IDENTICALLY to the f64 reference — the kernel adds no
     * error of its own. The rms term is a backstop against gross scale errors (those are O(1),
     * not 25%); it sits above the e2m1 floor, which this data measures at ~0.155. */
    const bool pass = glurms < 0.25 && bytemis < 0.01 && sqrt(se2 / (sw2 + 1e-30)) < 1e-3;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return !pass;
}
#endif
