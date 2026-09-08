/* Host driver for gemma31_gemv_decode_bench.hip -- the bf16 decode GEMV at Gemma-4 31B's own
 * BF16 TP1 decode shapes.
 *
 * Shapes and grids come from `plowrt disasm build-gemma31/assets-glm53/model.pkt --program 4`
 * (batch 4) and `--program 1` (batch 1). `nblk` is the emitter's own dispatch width and is NOT
 * the CU count: gate/up/q run 152 workgroups, k/v run 76, o_proj/down/lm_head run 304.
 *
 * (MM, M) PAIRS, and why both axes exist. MM is the COMPILED bucket (`PLOW_GEMV_MM`); M is the
 * packet's runtime row count. One instantiation serves every M <= MM, and the shipping decode
 * object for this asset is MM=4 (`plow_gemv_mm_cap_4` in interp_decode_gq.elf) while the blob
 * carries programs T=1, T=2 and T=4. So (4,1) and (4,2) are real serving configurations today,
 * and (1,1) / (2,2) are what a batch-width-matched object tier would run instead.
 *
 * Arms are interleaved in ONE process, each a median-of-IT hipEvent measurement, palindromic, and
 * `*_aa` is byte-identical device code to `*_base` so the run reports its own noise floor. Every
 * arm is arithmetic-preserving by construction and is checked ELEMENTWISE on device against the
 * base arm; anything but bit-identical is a bug, not a tradeoff. */
#include <hip/hip_runtime.h>

#include <algorithm>
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
    {"q_proj", 8192, 5376, 152, 60},   {"kv_proj", 4096, 5376, 76, 120},
    {"o_slide", 5376, 8192, 304, 50},  {"o_full", 5376, 16384, 304, 10},
    {"gate_up", 21504, 5376, 152, 120}, {"down", 5376, 21504, 304, 60},
    {"lm_head", 262144, 5376, 304, 1},
};
static const int NS = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

struct Group {
    int mm;
    std::vector<const char*> arms; /* arms[0] must be base, arms[1] the A/A twin */
    std::vector<unsigned> ms;      /* runtime M values to sweep */
};

int main(int argc, char** argv) {
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/gv31.co"));
    const int IT = argc > 2 ? atoi(argv[2]) : 41;
    const char* only = argc > 3 ? argv[3] : nullptr; /* optional MM filter, e.g. "4" */

    std::vector<Group> G = {
        {1,
         {"k_m1_base", "k_m1_aa", "k_m1_un6", "k_m1_un4", "k_m1_r2u6", "k_m1_r2u4", "k_m1_r4u3",
          "k_m1_mp", "k_m1_kt"},
         {1}},
        {2,
         {"k_m2_base", "k_m2_aa", "k_m2_un6", "k_m2_un4", "k_m2_r2u6", "k_m2_r2u4", "k_m2_r4u3",
          "k_m2_mp", "k_m2_kt"},
         {2}},
        {4,
         {"k_m4_base", "k_m4_aa", "k_m4_un3", "k_m4_un4", "k_m4_un8", "k_m4_un11", "k_m4_r2u2",
          "k_m4_r2u3", "k_m4_r2u4", "k_m4_r2u6", "k_m4_r4u2", "k_m4_mp", "k_m4_kt"},
         {4, 2, 1}},
        {8,
         {"k_m8_base", "k_m8_aa", "k_m8_un2", "k_m8_un4", "k_m8_un6", "k_m8_r2u2", "k_m8_r2u3",
          "k_m8_r4u2", "k_m8_mp", "k_m8_kt"},
         {8}},
    };

    const size_t ARENA = 3ull << 30;
    unsigned short *W, *C1, *C2, *xb;
    CK(hipMalloc(&W, ARENA));
    CK(hipMalloc(&C1, 8u << 20));
    CK(hipMalloc(&C2, 8u << 20));
    CK(hipMalloc(&xb, 8u << 20));
    {
        const size_t nfill = 64u << 20;
        std::vector<unsigned short> h(nfill / 2);
        unsigned st = 0x12345677u;
        for (size_t i = 0; i < h.size(); i++) {
            st = st * 1664525u + 1013904223u;
            /* small bf16 magnitudes: exponent near 127 so the dot products stay finite */
            h[i] = (unsigned short)(0x3B00u | ((st >> 20) & 0x7Fu));
        }
        for (size_t o = 0; o < ARENA; o += nfill)
            CK(hipMemcpy((char*)W + o, h.data(), std::min(nfill, ARENA - o), hipMemcpyHostToDevice));
        CK(hipMemcpy(xb, h.data(), 8u << 20, hipMemcpyHostToDevice));
    }

    double aa_lo = 9, aa_hi = 0;
    int mismatches = 0;
    for (const Group& g : G) {
        if (only && atoi(only) != g.mm) continue;
        const int NA = (int)g.arms.size();
        std::vector<hipFunction_t> fn(NA);
        for (int i = 0; i < NA; i++) CK(hipModuleGetFunction(&fn[i], M, g.arms[i]));
        for (unsigned Mr : g.ms) {
            printf("\n===== compiled bucket MM=%d, runtime M=%u\n", g.mm, Mr);
            printf("%-9s %7s %6s %4s %5s %5s | %10s %8s", "shape", "N", "K", "blk", "inst", "nrep",
                   "base_us", "GB/s");
            for (int i = 1; i < NA; i++) printf(" %9s", g.arms[i] + 6);
            printf("\n");
            std::vector<double> tot(NA, 0.0);
            for (int s = 0; s < NS; s++) {
                const Shape& S = SHAPES[s];
                const size_t slab = (size_t)S.N * S.K * 2;
                unsigned nrep = (unsigned)std::min<size_t>(ARENA / slab, 64);
                if (nrep < 1) nrep = 1;
                /* The K-tile arm holds RC=4 columns per wave in registers; it is only defined
                 * where the workgroup's column supply fits that. */
                const unsigned gv_per = (S.N + S.blk - 1) / S.blk;
                const bool kt_ok = ((gv_per + 7u) / 8u) <= 4u;
                void* a[] = {&C1, &xb, &W, &nrep, (void*)&S.N, (void*)&S.K, &Mr};
                std::vector<double> us(NA, 1e30);
                for (int i = 0; i < NA; i++) {
                    if (!kt_ok && std::string(g.arms[i]).find("_kt") != std::string::npos) continue;
                    us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT) / nrep);
                }
                for (int i = NA - 1; i >= 0; i--) {
                    if (!kt_ok && std::string(g.arms[i]).find("_kt") != std::string::npos) continue;
                    us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT) / nrep);
                }
                /* ELEMENTWISE identity against the base arm. Every arm here only moves which
                 * wave takes which column, how many loads are batched, or where x is read
                 * from -- so every output element must match exactly. */
                {
                    unsigned one = 1;
                    void* b1[] = {&C1, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &Mr};
                    void* b2[] = {&C2, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &Mr};
                    const size_t nout = (size_t)Mr * S.N;
                    std::vector<unsigned short> h1(nout), h2(nout);
                    CK(hipMemset(C1, 0, 8u << 20));
                    CK(hipModuleLaunchKernel(fn[0], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b1, nullptr));
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(h1.data(), C1, nout * 2, hipMemcpyDeviceToHost));
                    for (int i = 1; i < NA; i++) {
                        if (!kt_ok && std::string(g.arms[i]).find("_kt") != std::string::npos)
                            continue;
                        CK(hipMemset(C2, 0, 8u << 20));
                        CK(hipModuleLaunchKernel(fn[i], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b2,
                                                 nullptr));
                        CK(hipDeviceSynchronize());
                        CK(hipMemcpy(h2.data(), C2, nout * 2, hipMemcpyDeviceToHost));
                        size_t bad = 0;
                        for (size_t j = 0; j < nout; j++) bad += (h1[j] != h2[j]);
                        if (bad) {
                            printf("!! MM=%d M=%u %s %s: %zu/%zu outputs DIFFER\n", g.mm, Mr,
                                   S.name, g.arms[i], bad, nout);
                            mismatches++;
                        }
                    }
                }
                const double gb = (double)slab / 1e9;
                const double aa = us[1] / us[0];
                aa_lo = std::min(aa_lo, aa);
                aa_hi = std::max(aa_hi, aa);
                printf("%-9s %7u %6u %4u %5u %5u | %10.3f %8.0f", S.name, S.N, S.K, S.blk, S.inst,
                       nrep, us[0], gb / (us[0] * 1e-6));
                for (int i = 1; i < NA; i++) {
                    if (us[i] > 1e29)
                        printf(" %9s", "-");
                    else
                        printf(" %9.3f", us[0] / us[i]);
                }
                printf("\n");
                for (int i = 0; i < NA; i++)
                    if (us[i] < 1e29) tot[i] += us[i] * S.inst / 1e3;
            }
            printf("%-9s per-token ms over the tabulated T=4 instance counts:", "TOTAL");
            for (int i = 0; i < NA; i++) printf("  %s=%.3f", g.arms[i] + 6, tot[i]);
            printf("\n");
        }
    }
    printf("\nA/A control (aa / base): %.4f .. %.4f     identity mismatches: %d\n", aa_lo, aa_hi,
           mismatches);
    return mismatches ? 2 : 0;
}
