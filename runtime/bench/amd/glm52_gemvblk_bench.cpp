/* Ship-vs-ship: patched d_gemv_fp8_blk dispatch vs the bf16 GEMV, GLM TP4 M=1 shapes.
 * Interleaved A/B (contract 6b-STALE): fp8 and bf16 alternate, two passes. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x)                                                                                     \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_ != hipSuccess) { printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1); } \
    } while (0)
static hipModule_t M;
static const int T = 512;
static double tm(hipFunction_t f, int g, void** a, int it) {
    for (int i = 0; i < 3; i++) CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    hipEvent_t s, e;
    CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
    for (int i = 0; i < it; i++) {
        CK(hipEventRecord(s, 0));
        CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipEventRecord(e, 0)); CK(hipEventSynchronize(e));
        float ms = 0; CK(hipEventElapsedTime(&ms, s, e));
        v.push_back((double)ms * 1000.0);
    }
    CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
int main(int argc, char** argv) {
    CK(hipInit(0));
    CK(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/gv_dev2.co"));
    const int GRID = argc > 2 ? atoi(argv[2]) : 256;
    const int IT = argc > 3 ? atoi(argv[3]) : 21;
    hipFunction_t Ffp8, Fbf;
    CK(hipModuleGetFunction(&Ffp8, M, "k_blk_ship"));
    CK(hipModuleGetFunction(&Fbf, M, "k_bf16_ship"));
    struct Shape { const char* name; unsigned N, K; } shapes[] = {
        {"o_proj", 6144, 4096}, {"dense_down(44)", 6144, 3072}, {"shared_down", 6144, 512},
        {"shared_gate", 512, 6144}, {"kva_fusionA", 2624, 6144}, {"gate|up concat", 1024, 6144}, {"gate|up dense", 6144, 6144}, {"lm_head_like", 32768, 6144},
    };
    const size_t ARENA = 1500ull << 20;
    unsigned char* W8; void *Wbf, *C, *xb; float* S;
    CK(hipMalloc(&W8, ARENA)); CK(hipMemset(W8, 0x38, ARENA));
    CK(hipMalloc(&Wbf, ARENA)); CK(hipMemset(Wbf, 0x3c, ARENA));
    CK(hipMalloc(&C, 1 << 22)); CK(hipMalloc(&xb, 1 << 22)); CK(hipMemset(xb, 0x3c, 1 << 22));
    size_t sn = (size_t)(32768 / 128 + 1) * (16384 / 128 + 1) + 8192;
    CK(hipMalloc(&S, sn * 4));
    { std::vector<float> h(sn, 1.0f); CK(hipMemcpy(S, h.data(), h.size() * 4, hipMemcpyHostToDevice)); }
    printf("grid=%d blockDim=%d median-of-%d  denominator 6200 GB/s\n", GRID, T, IT);
    printf("%-15s %6s %6s | %8s %8s %7s | %8s %8s %7s | %s\n", "shape", "N", "K",
           "bf16 us", "GB/s", "%ceil", "fp8 us", "GB/s", "%ceil", "fp8 time vs bf16");
    for (auto& sh : shapes) {
        size_t wb = (size_t)sh.N * sh.K;
        unsigned r8 = (unsigned)std::max<size_t>(1, std::min<size_t>(64, ARENA / wb));
        unsigned rb = (unsigned)std::max<size_t>(1, std::min<size_t>(64, ARENA / (wb * 2)));
        void* a8[] = {&C, &xb, &W8, &S, &r8, &sh.N, &sh.K};
        void* ab[] = {&C, &xb, &Wbf, &rb, &sh.N, &sh.K};
        double b1 = tm(Fbf, GRID, ab, IT), f1 = tm(Ffp8, GRID, a8, IT);
        double b2 = tm(Fbf, GRID, ab, IT), f2 = tm(Ffp8, GRID, a8, IT);
        double b = std::min(b1, b2), f = std::min(f1, f2);
        double bgb = (double)wb * 2.0 * rb / (b * 1e3), fgb = (double)wb * r8 / (f * 1e3);
        printf("%-15s %6u %6u | %8.2f %8.0f %6.1f%% | %8.2f %8.0f %6.1f%% | %6.3fx\n", sh.name,
               sh.N, sh.K, b / rb, bgb, 100 * bgb / 6200, f / r8, fgb, 100 * fgb / 6200,
               (f / r8) / (b / rb));
    }
    /* --- the other half of the lever: shared gate/up, bf16 GemvGlu(19) vs DenseGluFp8Blk(47) --- */
    hipFunction_t Gb, Gf;
    if (hipModuleGetFunction(&Gb, M, "k_glu_bf16") == hipSuccess &&
        hipModuleGetFunction(&Gf, M, "k_glu_blk") == hipSuccess) {
        printf("\n-- shared gate/up (2 weight streams). bf16 GemvGlu(19) vs DenseGluFp8Blk(47), NOT patched\n");
        struct GS { unsigned N, K; } gs[] = {{512, 6144}, {3072, 6144}};
        for (auto& g : gs) {
            size_t wb = (size_t)g.N * g.K * 2; /* gate + up */
            unsigned r8 = (unsigned)std::max<size_t>(1, std::min<size_t>(32, ARENA / wb));
            unsigned rb = (unsigned)std::max<size_t>(1, std::min<size_t>(32, ARENA / (wb * 2)));
            void* a8[] = {&C, &xb, &W8, &S, &r8, &g.N, &g.K};
            void* ab[] = {&C, &xb, &Wbf, &rb, &g.N, &g.K};
            double b1 = tm(Gb, GRID, ab, IT), f1 = tm(Gf, GRID, a8, IT);
            double b2 = tm(Gb, GRID, ab, IT), f2 = tm(Gf, GRID, a8, IT);
            double b = std::min(b1, b2) / rb, f = std::min(f1, f2) / r8;
            double bgb = (double)wb * 2.0 / (b * 1e3), fgb = (double)wb / (f * 1e3);
            printf("%-15s %6u %6u | %8.2f %8.0f %6.1f%% | %8.2f %8.0f %6.1f%% | %6.3fx\n",
                   "shared gate/up", g.N, g.K, b, bgb, 100 * bgb / 6200, f, fgb,
                   100 * fgb / 6200, f / b);
        }
    }
    return 0;
}
