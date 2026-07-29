/* glm_khost3.cpp — the SHIPPING slice mappings of the GLM-5.2 routed-expert MoE block, one decode
 * layer's worth of work (top_k=8, TP4), measured per LAYER not per expert. */
#include <hip/hip_runtime.h>
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

static hipModule_t MOD;
static const int THREADS = 512;
static double timeit(hipFunction_t f, int grid, void** a, int iters) {
    for (int i = 0; i < 3; i++)
        CK(hipModuleLaunchKernel(f, grid, 1, 1, THREADS, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    CK(hipEventRecord(s, 0));
    for (int i = 0; i < iters; i++)
        CK(hipModuleLaunchKernel(f, grid, 1, 1, THREADS, 1, 1, 0, 0, a, nullptr));
    CK(hipEventRecord(e, 0));
    CK(hipEventSynchronize(e));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, s, e));
    return (double)ms * 1000.0 / iters;
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/glm_kdev.co";
    const char* tag = argc > 2 ? argv[2] : "slot-outer";
    CK(hipInit(0));
    CK(hipModuleLoad(&MOD, co));
    hipFunction_t Fgc, Fdc, Fgg, Fgd, Fgl, Fdl;
    CK(hipModuleGetFunction(&Fgc, MOD, "k_glu_cores"));
    CK(hipModuleGetFunction(&Fdc, MOD, "k_down_cores"));
    CK(hipModuleGetFunction(&Fgg, MOD, "k_group_glu"));
    CK(hipModuleGetFunction(&Fgd, MOD, "k_group_down"));
    CK(hipModuleGetFunction(&Fgl, MOD, "k_glu_loop"));
    CK(hipModuleGetFunction(&Fdl, MOD, "k_down_loop"));

    unsigned H = 6144, I = 512, E = 4096, TK = 8;
    /* 512 distinct experts so a repeat sweep never re-reads the LLC (512*9.4 MB = 4.8 GB) */
    int NP = 512;
    size_t gate_b = (size_t)NP * I * H, down_b = (size_t)NP * H * I;
    size_t Wb = gate_b * 2 + down_b;
    printf("[%s] arena %.2f GB, LLC 256 MB\n", tag, Wb / 1e9);
    unsigned char *W, *tabd;
    float *S, *part;
    void *x, *fu;
    unsigned long long *wt, *st;
    CK(hipMalloc(&W, Wb));
    CK(hipMemset(W, 0x3c, Wb));
    unsigned IB = (I + 127) / 128, HB = (H + 127) / 128;
    size_t Sn = (size_t)NP * 3 * IB * HB;
    CK(hipMalloc(&S, Sn * 4));
    {
        std::vector<float> h(Sn, 1.0f);
        CK(hipMemcpy(S, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    }
    CK(hipMalloc(&x, (size_t)H * 2));
    CK(hipMemset(x, 0x3c, (size_t)H * 2));
    CK(hipMalloc(&fu, (size_t)TK * I * 2));
    CK(hipMemset(fu, 0x3c, (size_t)TK * I * 2));
    CK(hipMalloc(&part, (size_t)TK * H * 4));
    std::vector<unsigned long long> hwt(NP * 3), hst(NP * 3);
    for (int p = 0; p < NP; p++) {
        hwt[p * 3 + 0] = (unsigned long long)(size_t)(W + (size_t)p * I * H);
        hwt[p * 3 + 1] = (unsigned long long)(size_t)(W + gate_b + (size_t)p * I * H);
        hwt[p * 3 + 2] = (unsigned long long)(size_t)(W + 2 * gate_b + (size_t)p * H * I);
        hst[p * 3 + 0] = (unsigned long long)(size_t)(S + (size_t)p * 3 * IB * HB);
        hst[p * 3 + 1] = (unsigned long long)(size_t)(S + ((size_t)p * 3 + 1) * IB * HB);
        hst[p * 3 + 2] = (unsigned long long)(size_t)(S + ((size_t)p * 3 + 2) * IB * HB);
    }
    CK(hipMalloc(&wt, hwt.size() * 8));
    CK(hipMemcpy(wt, hwt.data(), hwt.size() * 8, hipMemcpyHostToDevice));
    CK(hipMalloc(&st, hst.size() * 8));
    CK(hipMemcpy(st, hst.data(), hst.size() * 8, hipMemcpyHostToDevice));
    /* routing table: slot s -> expert s (8 slots) */
    std::vector<unsigned> htab(TK * 2);
    for (unsigned s = 0; s < TK; s++) {
        htab[s * 2] = s;
        float g = 1.0f;
        memcpy(&htab[s * 2 + 1], &g, 4);
    }
    CK(hipMalloc(&tabd, htab.size() * 4));
    CK(hipMemcpy(tabd, htab.data(), htab.size() * 4, hipMemcpyHostToDevice));

    const double CEIL = 6200.0;
    double b_glu = 2.0 * TK * (double)I * H, b_dn = (double)TK * H * I; /* ONE layer */
    printf("\nONE GLM-5.2 sparse layer, top_k=8, TP4 (routed experts only)\n");
    printf("%-52s %5s %9s %9s %7s\n", "layout", "grid", "us/layer", "GB/s", "%ceil");
    auto row = [&](const char* n, int g, double us, double by) {
        double gbs = by / (us * 1e-6) / 1e9;
        printf("%-52s %5d %9.2f %9.1f %6.1f%%\n", n, g, us, gbs, 100.0 * gbs / CEIL);
    };
    unsigned act = 1;
    /* A. CORESIDENT=2: 9 slices of 256/9=28 CUs, tk=8 experts concurrent (slot 8 = shared) */
    for (unsigned per : {28u, 32u}) {
        void* a[] = {&fu, &x, &tabd, &wt, &st, &per, &I, &H, &E, &act};
        row(per == 28 ? "GLU coresident: 8 experts x 28 CU, concurrent"
                      : "GLU coresident: 8 experts x 32 CU, concurrent",
            (int)(per * TK), timeit(Fgc, (int)(per * TK), a, 20), b_glu);
    }
    for (unsigned per : {28u, 32u}) {
        void* a[] = {&part, &fu, &tabd, &wt, &st, &per, &H, &I, &E};
        row(per == 28 ? "DOWN coresident: 8 experts x 28 CU, concurrent"
                      : "DOWN coresident: 8 experts x 32 CU, concurrent",
            (int)(per * TK), timeit(Fdc, (int)(per * TK), a, 20), b_dn);
    }
    /* B. CORESIDENT=0: serial, each expert on all 256 CUs */
    {
        unsigned nrep = TK;
        void* a[] = {&fu, &x, &tabd, &wt, &st, &nrep, &I, &H, &E, &act};
        row("GLU serial: 8 experts x 256 CU, one after another", 256, timeit(Fgl, 256, a, 20),
            b_glu);
        void* b[] = {&part, &fu, &tabd, &wt, &st, &nrep, &H, &I, &E};
        row("DOWN serial: 8 experts x 256 CU, one after another", 256, timeit(Fdl, 256, a ? b : b, 20),
            b_dn);
    }
    /* C. GROUPED (ops 48/49) on the whole chip */
    {
        void* a[] = {&fu, &x, &tabd, &wt, &st, &TK, &I, &H, &E, &act};
        char nm[128];
        snprintf(nm, sizeof(nm), "GLU grouped op48 (%s) x 256 CU", tag);
        row(nm, 256, timeit(Fgg, 256, a, 20), b_glu);
        void* b[] = {&part, &fu, &tabd, &wt, &st, &TK, &H, &I, &E};
        snprintf(nm, sizeof(nm), "DOWN grouped op49 (%s) x 256 CU", tag);
        row(nm, 256, timeit(Fgd, 256, b, 20), b_dn);
    }
    printf("roofline for this layer: GLU %.2f us, DOWN %.2f us @6200 GB/s\n", b_glu / 6200e3 * 1e3,
           b_dn / 6200e3 * 1e3);
    return 0;
}
