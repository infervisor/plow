/* driver for glm52_kbench_fold_pf.hip — MLA_MERGE_FOLD at GLM-5.2 TP8 PREFILL shape.
 * Usage: kb_fold_pf [module.co] [grid] [n_batch] [nsplit] */
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

static const unsigned NH = 8, DK = 512, VD = 256; /* GLM-5.2 TP8: nh_l=8, kv_lora=512, v_head=256 */
static unsigned NB = 8192, NS = 2;
static const int T = 512; /* PLOW_THREADS */

static unsigned short f2bf_h(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static unsigned rng_s = 12345;
static float rnd() {
    rng_s = rng_s * 1664525u + 1013904223u;
    return (float)((rng_s >> 8) & 0xffff) / 32768.0f - 1.0f;
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/glm_kfold_pf.co";
    int grid = argc > 2 ? atoi(argv[2]) : 304;
    if (argc > 3) NB = (unsigned)atoi(argv[3]);
    if (argc > 4) NS = (unsigned)atoi(argv[4]);
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, co));

    const size_t rows = (size_t)NB * NH;
    const size_t wuv_n = (size_t)NH * DK * VD;
    const size_t op_n = rows * NS * DK;
    const size_t ml_n = rows * NS * 2;
    const size_t o_n = rows * VD;

    std::vector<unsigned short> hW(wuv_n);
    for (size_t i = 0; i < wuv_n; i++) hW[i] = f2bf_h(rnd() * 0.05f);
    std::vector<float> hOp(op_n), hMl(ml_n);
    for (size_t i = 0; i < op_n; i++) hOp[i] = rnd();
    for (size_t r = 0; r < rows; r++)
        for (unsigned s = 0; s < NS; s++) {
            hMl[(r * NS + s) * 2] = rnd() * 2.0f;
            hMl[(r * NS + s) * 2 + 1] = 1.0f + 0.5f * (rnd() + 1.0f);
        }

    void *dW, *dO;
    float *dOp, *dMl;
    CK(hipMalloc(&dW, wuv_n * 2));
    CK(hipMemcpy(dW, hW.data(), wuv_n * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dO, o_n * 2));
    CK(hipMalloc(&dOp, op_n * 4));
    CK(hipMemcpy(dOp, hOp.data(), op_n * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dMl, ml_n * 4));
    CK(hipMemcpy(dMl, hMl.data(), ml_n * 4, hipMemcpyHostToDevice));

    const char* names[] = {"launch floor",  "merge only (ns share)", "SHIPPED  base TB1",
                           "TB2  vec4 un4", "TB4  vec4 un4",         "TB8  vec4 un4",
                           "TB4  vec4 un8", "TB8  vec4 un8",         "TB4  un4 SERIAL merge",
                           "TB4  un8 SERIAL merge"};
    const char* syms[] = {"k_pf_null",  "k_pf_merge",  "k_pf_base",  "k_pf_tb2",
                          "k_pf_tb4",   "k_pf_tb8",    "k_pf_tb4u8", "k_pf_tb8u8",
                          "k_pf_tb4s",  "k_pf_tb4u8s"};
    const int NK = (int)(sizeof(syms) / sizeof(syms[0]));

    /* the op's own byte model, per packet */
    const double wuv_b_tb1 = (double)rows * DK * VD * 2.0;      /* one panel per row */
    const double op_b = (double)op_n * 4.0, o_b = (double)o_n * 2.0;
    printf("PREFILL shape: n_batch(T)=%u nh_l=%u DK=%u V=%u ns=%u  grid=%d block=%d\n", NB, NH, DK,
           VD, NS, grid, T);
    printf("  rows=%zu  Opart=%.1f MB  O=%.1f MB  W_uv panel=%.2f MB  W_uv stream @TB1=%.2f GB\n",
           rows, op_b / 1048576.0, o_b / 1048576.0, (double)wuv_n * 2 / 1048576.0,
           wuv_b_tb1 / 1e9);
    printf("  fold MAC/packet=%.3f G   merge MAC/packet=%.3f G\n",
           (double)rows * DK * VD / 1e9, (double)rows * NS * DK / 1e9);
    printf("%-24s %10s %10s %12s %10s %8s\n", "variant", "us/pkt", "ms/78lay", "W_uv TB/s",
           "GFLOP/s", "bitcmp");

    std::vector<unsigned short> ref(o_n), cur(o_n);
    for (int k = 0; k < NK; k++) {
        hipFunction_t F;
        if (hipModuleGetFunction(&F, M, syms[k]) != hipSuccess) {
            printf("%-24s  (missing)\n", names[k]);
            continue;
        }
        unsigned nb = NB, nh = NH, v = VD, ns = NS;
        void* a[] = {&dO, &dOp, &dMl, &dW, &nb, &nh, &v, &ns};
        CK(hipMemset(dO, 0, o_n * 2));
        for (int w = 0; w < 2; w++)
            CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipDeviceSynchronize());
        const int REP = 5;
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        CK(hipEventRecord(e0, 0));
        for (int r = 0; r < REP; r++)
            CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipEventRecord(e1, 0));
        CK(hipDeviceSynchronize());
        float ms = 0;
        CK(hipEventElapsedTime(&ms, e0, e1));
        const double us = (double)ms * 1000.0 / REP;
        /* effective W_uv stream: TB1 model / TB for the blocked arms */
        double tbf = 1.0;
        if (strstr(syms[k], "tb2")) tbf = 2;
        else if (strstr(syms[k], "tb4")) tbf = 4;
        else if (strstr(syms[k], "tb8")) tbf = 8;
        const double flop = (strstr(syms[k], "merge") || strstr(syms[k], "null"))
                                ? (double)rows * NS * DK * 2.0
                                : (double)rows * DK * VD * 2.0;
        CK(hipMemcpy(cur.data(), dO, o_n * 2, hipMemcpyDeviceToHost));
        const char* bc = "-";
        if (k == 2) { ref = cur; bc = "REF"; }
        else if (k > 2) bc = memcmp(ref.data(), cur.data(), o_n * 2) == 0 ? "IDENTICAL" : "DIFFERS";
        printf("%-24s %10.1f %10.2f %12.2f %10.1f %8s\n", names[k], us, us * 78 / 1000.0,
               (strstr(syms[k], "null") || strstr(syms[k], "merge"))
                   ? 0.0
                   : wuv_b_tb1 / tbf / (us * 1e-6) / 1e12,
               flop / (us * 1e-6) / 1e9, bc);
    }
    return 0;
}
