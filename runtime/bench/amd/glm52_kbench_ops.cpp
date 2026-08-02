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
static hipModule_t M;
static const int T = 512;
static double tm(hipFunction_t f, int g, void** a, int it) {
    for (int i = 0; i < 5; i++) CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    CK(hipEventRecord(s, 0));
    for (int i = 0; i < it; i++) CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipEventRecord(e, 0));
    CK(hipEventSynchronize(e));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, s, e));
    return (double)ms * 1000.0 / it;
}
int main(int argc, char** argv) {
    int CTX = argc > 1 ? atoi(argv[1]) : 1024;
    CK(hipInit(0));
    CK(hipModuleLoad(&M, argc > 2 ? argv[2] : "/tmp/glm_kdev.co"));
    hipFunction_t Fn, Frl, Fhl, Fsl, Fgv, Fmla, Fmf, Fdglu;
    CK(hipModuleGetFunction(&Fn, M, "k_null"));
    CK(hipModuleGetFunction(&Frl, M, "k_rmsnorm_loop"));
    CK(hipModuleGetFunction(&Fhl, M, "k_hnr_loop"));
    CK(hipModuleGetFunction(&Fsl, M, "k_resid_loop"));
    CK(hipModuleGetFunction(&Fgv, M, "k_gemv_bf16"));
    CK(hipModuleGetFunction(&Fmla, M, "k_mla_dec2"));
    CK(hipModuleGetFunction(&Fmf, M, "k_mf"));
    CK(hipModuleGetFunction(&Fdglu, M, "k_dglu"));
    unsigned H = 6144, QL = 2048, DK = 512, DR = 64, VD = 256, NHL = 16, DIL = 3072, NS = 16;
    void *x, *o, *g, *Wuv, *Ol, *lmW;
    float *Op, *mlp, *cs, *sn, *S;
    int* kvl;
    unsigned char* Wfp8;
    CK(hipMalloc(&lmW, (size_t)154880 * 6144 * 2));
    CK(hipMemset(lmW, 0x3c, (size_t)154880 * 6144 * 2));
    CK(hipMalloc(&x, 1 << 24));
    CK(hipMemset(x, 0x3c, 1 << 24));
    CK(hipMalloc(&o, 1 << 24));
    CK(hipMalloc(&g, 1 << 20));
    CK(hipMemset(g, 0x3c, 1 << 20));
    CK(hipMalloc(&Wuv, (size_t)NHL * DK * VD * 2));
    CK(hipMemset(Wuv, 0x3c, (size_t)NHL * DK * VD * 2));
    CK(hipMalloc(&Op, (size_t)NHL * NS * DK * 4));
    CK(hipMemset(Op, 0, (size_t)NHL * NS * DK * 4));
    CK(hipMalloc(&mlp, (size_t)NHL * NS * 2 * 4));
    CK(hipMemset(mlp, 0, (size_t)NHL * NS * 2 * 4));
    CK(hipMalloc(&Ol, (size_t)CTX * (DK + DR) * 2 + 4096));
    CK(hipMemset(Ol, 0x3c, (size_t)CTX * (DK + DR) * 2));
    CK(hipMalloc(&cs, 1 << 24));
    CK(hipMemset(cs, 0, 1 << 24));
    CK(hipMalloc(&sn, 1 << 24));
    CK(hipMemset(sn, 0, 1 << 24));
    CK(hipMalloc(&kvl, 64));
    CK(hipMemset(kvl, 0, 64));
    {
        int L = CTX;
        CK(hipMemcpy(kvl, &L, 4, hipMemcpyHostToDevice));
    }
    CK(hipMalloc(&Wfp8, (size_t)DIL * H * 2));
    CK(hipMemset(Wfp8, 0x3c, (size_t)DIL * H * 2));
    size_t sn2 = (size_t)(DIL / 128) * (H / 128) + 8192;
    CK(hipMalloc(&S, sn2 * 4));
    {
        std::vector<float> hh(sn2, 1.0f);
        CK(hipMemcpy(S, hh.data(), hh.size() * 4, hipMemcpyHostToDevice));
    }
    const double CEIL = 6200.0;
    unsigned M1 = 1, REP = 200;
    float eps = 1e-5f;
    {
        void* a[] = {&Op};
        printf("empty-kernel launch floor of THIS harness: grid1 %.3f us, grid256 %.3f us\n",
               tm(Fn, 1, a, 500), tm(Fn, 256, a, 500));
    }
    printf("\nGLM-5.2 TP4 decode, ctx=%d, per-rank shapes. blockDim=512. BODY cost (loop-amortised)\n",
           CTX);
    printf("%-48s %5s %9s %9s %7s %7s\n", "op @ shape", "grid", "us", "MB", "GB/s", "%ceil");
    auto row = [&](const char* n, int gr, double us, double by) {
        double gbs = by / (us * 1e-6) / 1e9;
        printf("%-48s %5d %9.3f %9.3f %7.1f %6.1f%%\n", n, gr, us, by / 1e6, gbs,
               100.0 * gbs / CEIL);
    };
    {
        void* a[] = {&o, &x, &g, &REP, &H, &eps};
        row("RmsNorm h=6144 BODY (emitted grid=1)", 1, tm(Frl, 1, a, 20) / REP, 2.0 * H * 2);
        row("RmsNorm h=6144 BODY @grid=256", 256, tm(Frl, 256, a, 20) / REP, 2.0 * H * 2);
        unsigned q = QL;
        void* b[] = {&o, &x, &g, &REP, &q, &eps};
        row("RmsNorm q_lora=2048 BODY (grid=1)", 1, tm(Frl, 1, b, 20) / REP, 2.0 * QL * 2);
        unsigned d = DK;
        void* c[] = {&o, &x, &g, &REP, &d, &eps};
        row("RmsNorm kv_lora=512 BODY (grid=1)", 1, tm(Frl, 1, c, 20) / REP, 2.0 * DK * 2);
    }
    {
        void* a[] = {&o, &x, &g, &REP, &H};
        row("Residual n=6144 BODY (emitted grid=1)", 1, tm(Fsl, 1, a, 20) / REP, 3.0 * H * 2);
    }
    {
        unsigned nh = NHL;
        void* a[] = {&o, &x, &cs, &sn, &kvl, &REP, &nh};
        row("HeadNormRope q hd=64 nh=16 BODY (grid 256)", 256, tm(Fhl, 256, a, 20) / REP,
            2.0 * NHL * 64 * 2);
        unsigned n1 = 1;
        void* b[] = {&o, &x, &cs, &sn, &kvl, &REP, &n1};
        row("HeadNormRope krope hd=64 nh=1 BODY (grid 256)", 256, tm(Fhl, 256, b, 20) / REP,
            2.0 * 64 * 2);
    }
    {
        unsigned Nv = 154880, Kv = H;
        void* b[] = {&o, &x, &lmW, &M1, &Nv, &Kv};
        row("Gemv bf16 lm_head N=154880 K=6144 UNSHARDED", 256, tm(Fgv, 256, b, 20),
            (double)Nv * Kv * 2);
    }
    {
        unsigned nh = NHL, ks = (unsigned)CTX;
        float sc = 0.1f;
        void* a[] = {&Op, &mlp, &x, &x, &Ol, &Ol, &kvl, &nh, &ks, &sc, &NS};
        row("FlashMlaDecode GF=2 nh=16 ns=16 ctx", 256, tm(Fmla, 256, a, 100),
            (double)(NHL / 2) * CTX * (DK + DR) * 2);
        unsigned V = VD;
        void* b[] = {&o, &Op, &mlp, &Wuv, &nh, &V, &NS};
        row("MlaMergeFold nh=16 V=256 ns=16", 256, tm(Fmf, 256, b, 200), (double)NHL * DK * VD * 2);
    }
    {
        unsigned N = DIL, K = H;
        void* a[] = {&o, &x, &Wfp8, &Wfp8, &S, &S, &N, &K};
        row("DenseGluFp8Blk N=3072 K=6144 (layers 0-2)", 256, tm(Fdglu, 256, a, 50), 2.0 * DIL * H);
    }
    return 0;
}
