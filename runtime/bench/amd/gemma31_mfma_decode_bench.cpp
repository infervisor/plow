/* Host driver for gemma31_mfma_decode_bench.hip.
 *
 * Shapes and grids come from `plowrt disasm build-gemma31/assets-ctx131072-chunk8192/model.pkt
 * --program 4`; `nblk` is the emitter's own dispatch width, not a CU count.
 *
 * Three things are measured per (compiled bucket MM, runtime M, shape):
 *   1. us/packet and achieved weight GB/s for the shipped VALU body, every MFMA arm, and the
 *      A/A control (byte-identical device code under a second name). The A/A must read
 *      0.99-1.01 or nothing else on the line is trustworthy.
 *   2. relative L2 of every arm against the VALU base, and of every arm against a
 *      double-precision CPU reference over sampled output columns. The MFMA arms are NOT
 *      bit-identical to the base by construction; the A/A arm still must be.
 *   3. BATCH-WIDTH INDEPENDENCE: row 0 computed with M=1 vs row 0 of the same arm run at the
 *      bucket's full width, bit-compared. A prompt's output must not depend on how many other
 *      sequences share the step.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static const int T = 512; /* PLOW_THREADS */

static double bf2d(unsigned short b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return (double)f;
}

static double tm(hipFunction_t f, int g, void** a, int it) {
    for (int i = 0; i < 3; i++) CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    for (int i = 0; i < it; i++) {
        CK(hipEventRecord(s, 0));
        CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipEventRecord(e, 0));
        CK(hipEventSynchronize(e));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, s, e));
        v.push_back((double)ms * 1000.0);
    }
    CK(hipEventDestroy(s));
    CK(hipEventDestroy(e));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Shape {
    const char* name;
    unsigned N, K, blk;
    unsigned inst; /* packets per token at batch 4 (T=4 program) */
};

/* T=4 program, 60 layers: 50 sliding (hd 256, o K=8192) + 10 full (hd 512, o K=16384). */
static Shape SHAPES[] = {
    {"q_proj", 8192, 5376, 152, 60},    {"kv_proj", 4096, 5376, 76, 120},
    {"o_slide", 5376, 8192, 304, 50},   {"o_full", 5376, 16384, 304, 10},
    {"gate_up", 21504, 5376, 152, 120}, {"down", 5376, 21504, 304, 60},
    {"lm_head", 262144, 5376, 304, 1},
};
static const int NS = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

struct Group {
    int mm;
    std::vector<const char*> arms; /* arms[0] must be base, arms[1] the A/A twin */
    std::vector<unsigned> ms;
};

int main(int argc, char** argv) {
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/mf31.co"));
    const int IT = argc > 2 ? atoi(argv[2]) : 41;
    const char* only = argc > 3 ? argv[3] : nullptr;

    std::vector<Group> G = {
        {1,
         {"k_m1_base", "k_m1_aa", "k_m1_mf_u6y1", "k_m1_mf_u11y1", "k_m1_mf_u4y2", "k_m1_mf_u6y2",
          "k_m1_mf_u3y4"},
         {1}},
        {2,
         {"k_m2_base", "k_m2_aa", "k_m2_mf_u6y1", "k_m2_mf_u11y1", "k_m2_mf_u4y2", "k_m2_mf_u6y2",
          "k_m2_mf_u3y4", "k_m2_mf_u6y4", "k_m2_mf_u4y3"},
         {2}},
        {4,
         {"k_m4_base", "k_m4_aa", "k_m4_mf_u3y1", "k_m4_mf_u6y1", "k_m4_mf_u8y1", "k_m4_mf_u11y1",
          "k_m4_mf_u2y2", "k_m4_mf_u4y2", "k_m4_mf_u6y2", "k_m4_mf_u8y2", "k_m4_mf_u3y3",
          "k_m4_mf_u4y3", "k_m4_mf_u2y4", "k_m4_mf_u3y4"},
         {4}},
        {8,
         {"k_m8_base", "k_m8_aa", "k_m8_mf_u2y1", "k_m8_mf_u3y1", "k_m8_mf_u6y1", "k_m8_mf_u2y2",
          "k_m8_mf_u3y2", "k_m8_mf_u4y1"},
         {8}},
    };

    const size_t ARENA = 3ull << 30;
    const size_t HN = 32u << 20; /* halves of host pattern; the arena repeats it */
    unsigned short *W, *C1, *C2, *xb;
    CK(hipMalloc(&W, ARENA));
    CK(hipMalloc(&C1, 8u << 20));
    CK(hipMalloc(&C2, 8u << 20));
    CK(hipMalloc(&xb, 8u << 20));
    std::vector<unsigned short> h(HN);
    {
        unsigned st = 0x12345677u;
        for (size_t i = 0; i < h.size(); i++) {
            st = st * 1664525u + 1013904223u;
            h[i] = (unsigned short)(0x3B00u | ((st >> 20) & 0x7Fu));
        }
        for (size_t o = 0; o < ARENA; o += HN * 2)
            CK(hipMemcpy((char*)W + o, h.data(), std::min(HN * 2, ARENA - o),
                         hipMemcpyHostToDevice));
        CK(hipMemcpy(xb, h.data(), 8u << 20, hipMemcpyHostToDevice));
    }

    double aa_lo = 9, aa_hi = 0;
    int hard_fail = 0;
    for (const Group& g : G) {
        if (only && atoi(only) != g.mm) continue;
        const int NA = (int)g.arms.size();
        std::vector<hipFunction_t> fn(NA);
        for (int i = 0; i < NA; i++) CK(hipModuleGetFunction(&fn[i], M, g.arms[i]));
        for (unsigned Mr : g.ms) {
            printf("\n===== compiled bucket MM=%d, runtime M=%u   (us/packet, then speedup)\n",
                   g.mm, Mr);
            printf("%-9s %7s %6s %4s %5s | %10s %8s", "shape", "N", "K", "blk", "nrep", "base_us",
                   "GB/s");
            for (int i = 1; i < NA; i++) printf(" %11s", g.arms[i] + 5);
            printf("\n");
            std::vector<double> tot(NA, 0.0);
            std::vector<double> bestgb(NA, 0.0);
            for (int s = 0; s < NS; s++) {
                const Shape& S = SHAPES[s];
                const size_t slab = (size_t)S.N * S.K * 2;
                unsigned nrep = (unsigned)std::min<size_t>(ARENA / slab, 64);
                if (nrep < 1) nrep = 1;
                void* a[] = {&C1, &xb, &W, &nrep, (void*)&S.N, (void*)&S.K, &Mr};
                std::vector<double> us(NA, 1e30);
                for (int i = 0; i < NA; i++)
                    us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT) / nrep);
                for (int i = NA - 1; i >= 0; i--)
                    us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT) / nrep);
                const double gb = (double)slab / 1e9;
                const double aa = us[1] / us[0];
                aa_lo = std::min(aa_lo, aa);
                aa_hi = std::max(aa_hi, aa);
                printf("%-9s %7u %6u %4u %5u | %10.3f %8.0f", S.name, S.N, S.K, S.blk, nrep, us[0],
                       gb / (us[0] * 1e-6));
                for (int i = 1; i < NA; i++) printf(" %11.3f", us[0] / us[i]);
                printf("\n");
                for (int i = 0; i < NA; i++) {
                    tot[i] += us[i] * S.inst / 1e3;
                    bestgb[i] = std::max(bestgb[i], gb / (us[i] * 1e-6));
                }
            }
            printf("%-9s per-token ms over the T=4 instance counts:", "TOTAL");
            for (int i = 0; i < NA; i++) printf("  %s=%.3f", g.arms[i] + 5, tot[i]);
            printf("\n%-9s peak weight GB/s:", "");
            for (int i = 0; i < NA; i++) printf("  %s=%.0f", g.arms[i] + 5, bestgb[i]);
            printf("\n");
        }

        /* ---------------- numerics ------------------------------------------------------ */
        for (unsigned Mr : g.ms) {
            printf("\n--- numerics, MM=%d M=%u (rel L2 vs VALU base / vs CPU float64, sampled)\n",
                   g.mm, Mr);
            for (int s = 0; s < NS; s++) {
                const Shape& S = SHAPES[s];
                if (S.N > 32768) continue; /* lm_head: same story, and the CPU pass is slow */
                unsigned one = 1;
                const size_t nout = (size_t)Mr * S.N;
                std::vector<unsigned short> hb(nout), hc(nout);
                void* b1[] = {&C1, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &Mr};
                void* b2[] = {&C2, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &Mr};
                CK(hipMemset(C1, 0, 8u << 20));
                CK(hipModuleLaunchKernel(fn[0], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b1, nullptr));
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(hb.data(), C1, nout * 2, hipMemcpyDeviceToHost));
                const int NSAMP = 16;
                std::vector<double> ref((size_t)Mr * NSAMP);
                std::vector<unsigned> cols(NSAMP);
                for (int p = 0; p < NSAMP; p++) {
                    cols[p] = (unsigned)(((unsigned long long)p * (S.N - 1)) / (NSAMP - 1));
                    for (unsigned m = 0; m < Mr; m++) {
                        double acc = 0;
                        for (unsigned k = 0; k < S.K; k++)
                            acc += bf2d(h[((size_t)cols[p] * S.K + k) % HN]) *
                                   bf2d(h[(size_t)m * S.K + k]);
                        ref[(size_t)m * NSAMP + p] = acc;
                    }
                }
                printf("  %-9s", S.name);
                for (int i = 0; i < NA; i++) {
                    const unsigned short* out = hb.data();
                    if (i) {
                        CK(hipMemset(C2, 0, 8u << 20));
                        CK(hipModuleLaunchKernel(fn[i], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b2,
                                                 nullptr));
                        CK(hipDeviceSynchronize());
                        CK(hipMemcpy(hc.data(), C2, nout * 2, hipMemcpyDeviceToHost));
                        out = hc.data();
                    }
                    double sq = 0, rs = 0;
                    size_t bad = 0;
                    for (size_t j = 0; j < nout; j++) {
                        double v = bf2d(out[j]), r = bf2d(hb[j]);
                        sq += (v - r) * (v - r);
                        rs += r * r;
                        bad += (out[j] != hb[j]);
                    }
                    double csq = 0, crs = 0;
                    for (unsigned m = 0; m < Mr; m++)
                        for (int p = 0; p < NSAMP; p++) {
                            double v = bf2d(out[(size_t)m * S.N + cols[p]]);
                            double r = ref[(size_t)m * NSAMP + p];
                            csq += (v - r) * (v - r);
                            crs += r * r;
                        }
                    if (i == 1 && bad) {
                        printf("  [A/A NOT BIT-IDENTICAL: %zu]", bad);
                        hard_fail++;
                    }
                    printf("  %s=%.3e/%.3e", g.arms[i] + 5, std::sqrt(sq / (rs + 1e-30)),
                           std::sqrt(csq / (crs + 1e-30)));
                }
                printf("\n");
            }
        }

        /* ---------------- batch-width independence ------------------------------------- */
        if (g.mm > 1) {
            printf("\n--- batch-width independence, MM=%d: row 0 at M=1 vs row 0 at M=%d\n", g.mm,
                   g.mm);
            for (int s = 0; s < NS; s++) {
                const Shape& S = SHAPES[s];
                if (S.N > 32768) continue;
                unsigned one = 1, mone = 1, mfull = (unsigned)g.mm;
                std::vector<unsigned short> r1(S.N), rf((size_t)g.mm * S.N);
                printf("  %-9s", S.name);
                for (int i = 0; i < NA; i++) {
                    void* b1[] = {&C1, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &mone};
                    void* b2[] = {&C2, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &mfull};
                    CK(hipMemset(C1, 0, 8u << 20));
                    CK(hipModuleLaunchKernel(fn[i], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b1, nullptr));
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(r1.data(), C1, (size_t)S.N * 2, hipMemcpyDeviceToHost));
                    CK(hipMemset(C2, 0, 8u << 20));
                    CK(hipModuleLaunchKernel(fn[i], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b2, nullptr));
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(rf.data(), C2, (size_t)g.mm * S.N * 2, hipMemcpyDeviceToHost));
                    size_t bad = 0;
                    for (unsigned j = 0; j < S.N; j++) bad += (r1[j] != rf[j]);
                    printf("  %s=%s", g.arms[i] + 5, bad ? "DIFFER" : "ident");
                    if (bad) hard_fail++;
                }
                printf("\n");
            }
        }
    }
    printf("\nA/A control (aa / base): %.4f .. %.4f     hard failures: %d\n", aa_lo, aa_hi,
           hard_fail);
    return hard_fail ? 2 : 0;
}
