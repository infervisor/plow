/* driver for glm52_kbench_fold.hip — correctness + timing of the MLA_MERGE_FOLD body.
 * Usage: kb_fold [module.co] [grid]   (grid defaults to 256, the interpreter's decode width) */
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x)                                                                                     \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_ != hipSuccess) {                                                                   \
            printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));                 \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static const unsigned NH = 16, DK = 512, VD = 256;
static unsigned NS = 16;
static const int T = 512;
static const int NCOPY = 96; /* 96 * 4 MB = 384 MB of W_uv > the 256 MB LLC */

static unsigned short f2bf_h(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf2f_h(unsigned short b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static unsigned rng_s = 12345;
static float rnd() {
    rng_s = rng_s * 1664525u + 1013904223u;
    return (float)((rng_s >> 8) & 0xffff) / 32768.0f - 1.0f;
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/glm_kfold.co";
    int grid = argc > 2 ? atoi(argv[2]) : 256;
    if (argc > 3) NS = (unsigned)atoi(argv[3]);
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, co));

    const size_t wuv_n = (size_t)NH * DK * VD;               /* halves per copy */
    const size_t op_n = (size_t)NH * NS * DK;                /* floats */
    const size_t ml_n = (size_t)NH * NS * 2;                 /* floats */
    std::vector<unsigned short> hW(wuv_n);
    std::vector<float> hOp(op_n), hMl(ml_n);
    for (size_t i = 0; i < wuv_n; i++) hW[i] = f2bf_h(rnd() * 0.05f);
    for (size_t i = 0; i < op_n; i++) hOp[i] = rnd();
    for (unsigned h = 0; h < NH; h++)
        for (unsigned s = 0; s < NS; s++) {
            hMl[(h * NS + s) * 2] = rnd() * 2.0f;
            hMl[(h * NS + s) * 2 + 1] = 1.0f + 0.5f * (rnd() + 1.0f);
        }

    /* CPU reference: sequential-order fold of the merged latent. */
    std::vector<float> ref((size_t)NH * VD);
    for (unsigned h = 0; h < NH; h++) {
        const float* ml = &hMl[(size_t)h * NS * 2];
        float gm = -INFINITY;
        for (unsigned s = 0; s < NS; s++) gm = fmaxf(gm, ml[s * 2]);
        float gl = 0.f;
        for (unsigned s = 0; s < NS; s++) gl += ml[s * 2 + 1] * exp2f(ml[s * 2] - gm);
        const float inv = gl > 0.f ? 1.f / gl : 0.f;
        std::vector<float> olat(DK);
        for (unsigned d = 0; d < DK; d++) {
            float a = 0.f;
            for (unsigned s = 0; s < NS; s++)
                a += hOp[((size_t)h * NS + s) * DK + d] * exp2f(ml[s * 2] - gm);
            olat[d] = a * inv;
        }
        for (unsigned v = 0; v < VD; v++) {
            float a = 0.f;
            for (unsigned l = 0; l < DK; l++)
                a += olat[l] * bf2f_h(hW[(size_t)h * DK * VD + (size_t)l * VD + v]);
            ref[(size_t)h * VD + v] = a;
        }
    }

    void *dW, *dO;
    float *dOp, *dMl;
    CK(hipMalloc(&dW, wuv_n * 2 * NCOPY));
    for (int c = 0; c < NCOPY; c++)
        CK(hipMemcpy((char*)dW + (size_t)c * wuv_n * 2, hW.data(), wuv_n * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dO, (size_t)NH * VD * 2));
    CK(hipMalloc(&dOp, op_n * 4));
    CK(hipMemcpy(dOp, hOp.data(), op_n * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dMl, ml_n * 4));
    CK(hipMemcpy(dMl, hMl.data(), ml_n * 4, hipMemcpyHostToDevice));

    const char* names[] = {"launch floor",      "base VT256 scalar", "VT16  vec4 un4",
                           "VT32  vec4 un4",    "VT64  vec4 un4",    "VT128 vec4 un4",
                           "VT256 vec4 un4",    "VT64  vec2 un4",    "VT64  vec8 un4",
                           "VT64  vec4 un2",    "VT64  vec4 un8",    "VT32  vec8 un4",
                           "VT128 vec8 un4",    "VT16  vec8 un2",    "VT32  vec8 un8"};
    const char* syms[] = {"k_null",     "k_mf_base",   "k_mf_v16",    "k_mf_v32",
                          "k_mf_v64",   "k_mf_v128",   "k_mf_v256",   "k_mf_v64c2",
                          "k_mf_v64c8", "k_mf_v64u2",  "k_mf_v64u8",  "k_mf_v32c8",
                          "k_mf_v128c8", "k_mf_v16c8", "k_mf_v32c8u8"};
    const int NK = (int)(sizeof(syms) / sizeof(syms[0]));
    const double bytes = (double)wuv_n * 2.0; /* the W_uv stream, the op's own roofline */

    printf("grid=%d  nh=%u DK=%u V=%u ns=%u   W_uv=%.2f MB/packet\n", grid, NH, DK, VD, NS,
           bytes / 1048576.0);
    printf("%-20s %10s %10s %8s %10s %14s %8s\n", "variant", "us/pkt", "GB/s", "%6200", "rel_rms",
           "vs-scalar diff", "max ulp");
    std::vector<unsigned short> base((size_t)NH * VD, 0);
    for (int k = 0; k < NK; k++) {
        hipFunction_t F;
        CK(hipModuleGetFunction(&F, M, syms[k]));
        unsigned nh = NH, v = VD, ns = NS;
        void* Wp = dW;
        void* a[] = {&dO, &dOp, &dMl, &Wp, &nh, &v, &ns};
        CK(hipMemset(dO, 0, (size_t)NH * VD * 2));
        CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipDeviceSynchronize());
        std::vector<unsigned short> got((size_t)NH * VD);
        CK(hipMemcpy(got.data(), dO, got.size() * 2, hipMemcpyDeviceToHost));
        double se = 0, sr = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = bf2f_h(got[i]) - ref[i];
            se += d * d;
            sr += (double)ref[i] * ref[i];
        }
        const double rms = sqrt(se / (sr > 0 ? sr : 1));
        /* The gate that separates "reassociated" from "buggy": every element must be within a
         * bf16 ulp or two of what the SHIPPED scalar body produces on the same inputs. */
        if (k == 1) base = got; /* k_mf_base, the shipped one-thread-per-column fold */
        long ndiff = 0, maxulp = 0;
        for (size_t i = 0; i < got.size(); i++) {
            long d = (long)got[i] - (long)base[i];
            if (d) ndiff++;
            if (labs(d) > maxulp) maxulp = labs(d);
        }

        const int IT = 200;
        for (int i = 0; i < 10; i++) {
            Wp = (char*)dW + (size_t)(i % NCOPY) * wuv_n * 2;
            CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        }
        CK(hipDeviceSynchronize());
        hipEvent_t s, e;
        CK(hipEventCreate(&s));
        CK(hipEventCreate(&e));
        CK(hipEventRecord(s, 0));
        for (int i = 0; i < IT; i++) {
            Wp = (char*)dW + (size_t)(i % NCOPY) * wuv_n * 2;
            CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        }
        CK(hipEventRecord(e, 0));
        CK(hipEventSynchronize(e));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, s, e));
        const double us = (double)ms * 1000.0 / IT;
        printf("%-20s %10.2f %10.1f %7.1f%% %10.2e %8ld/%-5zu %8ld%s\n", names[k], us,
               bytes / us / 1000.0, 100.0 * (bytes / us / 1000.0) / 6200.0, rms, ndiff, got.size(),
               maxulp, rms < 5e-3 ? "" : "   <-- FAIL");
        CK(hipEventDestroy(s));
        CK(hipEventDestroy(e));
    }
    return 0;
}
