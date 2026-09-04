/* driver for k3_kbench_fold.hip — bit-exactness oracle + timing of the MLA_MERGE_FOLD body at
 * the Kimi-K3 TP8 decode shape (nh_l=12, DK=512, V=128, nsplit=64 on a 256-workgroup grid).
 * Usage: kb_k3fold [module.co] [grid] [nh] [ns]
 * Exactness: every variant is compared bit for bit against k_ctl at the given nsplit AND at a
 * set of tail-exercising nsplits with dead splits (m = FA_NEG_INF) sprinkled in; any mismatch
 * is a FAIL and the exit code is 1. Timing rotates W_uv over a >LLC arena (cold, as in the
 * model) and reports the partials both L2-hot and cold (the model sits between: they were just
 * written by flash workgroups on other XCDs). */
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

static const unsigned DK = 512, VD = 128;
static const int T = 512;
static const float NEG_INF = -3.0e38f; /* FA_NEG_INF */

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

struct Inputs {
    std::vector<unsigned short> W;
    std::vector<float> Op, Ml;
};
static Inputs gen(unsigned nh, unsigned ns, bool dead) {
    Inputs in;
    in.W.resize((size_t)nh * DK * VD);
    in.Op.resize((size_t)nh * ns * DK);
    in.Ml.resize((size_t)nh * ns * 2);
    for (auto& w : in.W) w = f2bf_h(rnd() * 0.05f);
    for (auto& o : in.Op) o = rnd();
    for (unsigned h = 0; h < nh; h++)
        for (unsigned s = 0; s < ns; s++) {
            const bool d = dead && ((s * 7 + h) % 5 == 0);
            in.Ml[(h * ns + s) * 2] = d ? NEG_INF : rnd() * 2.0f;
            in.Ml[(h * ns + s) * 2 + 1] = d ? 0.0f : 1.0f + 0.5f * (rnd() + 1.0f);
            if (d)
                for (unsigned k = 0; k < DK; k++) in.Op[((size_t)h * ns + s) * DK + k] = 0.0f;
        }
    return in;
}

static std::vector<unsigned short> run(hipFunction_t F, int grid, unsigned nh, unsigned ns,
                                       void* dO, float* dOp, float* dMl, void* dW) {
    unsigned v = VD;
    void* a[] = {&dO, &dOp, &dMl, &dW, &nh, &v, &ns};
    CK(hipMemset(dO, 0xff, (size_t)nh * VD * 2));
    CK(hipModuleLaunchKernel(F, grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<unsigned short> got((size_t)nh * VD);
    CK(hipMemcpy(got.data(), dO, got.size() * 2, hipMemcpyDeviceToHost));
    return got;
}

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/k3_kfold.co";
    const int grid = argc > 2 ? atoi(argv[2]) : 256;
    const unsigned NH = argc > 3 ? (unsigned)atoi(argv[3]) : 12u;
    const unsigned NS = argc > 4 ? (unsigned)atoi(argv[4]) : 64u;
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, co));
    const char* syms[] = {"k_null", "k_ctl", "k_s32", "k_s16", "k_s8", "k_s8m8", "k_s8m64"};
    const char* names[] = {"launch floor",     "ctl VT32 (48 wg)", "split VT32",
                           "split VT16",       "split VT8",        "split VT8 ms8",
                           "split VT8 ms64"};
    const int NK = (int)(sizeof(syms) / sizeof(syms[0]));
    hipFunction_t F[NK];
    for (int k = 0; k < NK; k++) CK(hipModuleGetFunction(&F[k], M, syms[k]));

    /* ---- exactness: k_ctl vs every split variant, several nsplit, with and without dead splits */
    const unsigned NSS[] = {NS, 64, 48, 40, 24, 13, 8, 5, 1};
    const size_t nss = sizeof(NSS) / sizeof(NSS[0]);
    int fails = 0;
    void* dO;
    CK(hipMalloc(&dO, (size_t)NH * VD * 2));
    for (size_t i = 0; i < nss; i++)
        for (int dead = 0; dead < 2; dead++) {
            const unsigned ns = NSS[i];
            Inputs in = gen(NH, ns, dead != 0);
            void* dW;
            float *dOp, *dMl;
            CK(hipMalloc(&dW, in.W.size() * 2));
            CK(hipMalloc(&dOp, in.Op.size() * 4));
            CK(hipMalloc(&dMl, in.Ml.size() * 4));
            CK(hipMemcpy(dW, in.W.data(), in.W.size() * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(dOp, in.Op.data(), in.Op.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dMl, in.Ml.data(), in.Ml.size() * 4, hipMemcpyHostToDevice));
            std::vector<unsigned short> ctl = run(F[1], grid, NH, ns, dO, dOp, dMl, dW);
            for (int k = 2; k < NK; k++) {
                std::vector<unsigned short> got = run(F[k], grid, NH, ns, dO, dOp, dMl, dW);
                long nd = 0;
                for (size_t j = 0; j < got.size(); j++) nd += got[j] != ctl[j];
                if (nd) {
                    fails++;
                    printf("MISMATCH %-16s ns=%-3u dead=%d : %ld/%zu elements\n", names[k], ns,
                           dead, nd, got.size());
                }
            }
            CK(hipFree(dW));
            CK(hipFree(dOp));
            CK(hipFree(dMl));
        }
    printf("exactness vs k_ctl over %zu nsplits x {live,dead} x %d variants: %s\n", nss, NK - 2,
           fails ? "FAIL" : "bit-identical");

    /* ---- timing at the requested shape. Two regimes: partials HOT (one copy, L2-resident, the
     * optimistic case) and partials COLD (rotated over a >LLC arena like W_uv — in the model
     * they were just written by flash workgroups on other XCDs, so they come from MALL/HBM, and
     * every extra V-tile re-reads the whole 128 KB per row). */
    Inputs in = gen(NH, NS, false);
    const size_t wuv_b = in.W.size() * 2, op_b = in.Op.size() * 4, ml_b = in.Ml.size() * 4;
    const int NCOPY = (int)((320u << 20) / wuv_b) + 1; /* > the 256 MB LLC */
    void* dW;
    float *dOp, *dMl;
    CK(hipMalloc(&dW, wuv_b * NCOPY));
    CK(hipMalloc(&dOp, op_b * NCOPY));
    CK(hipMalloc(&dMl, ml_b * NCOPY));
    for (int c = 0; c < NCOPY; c++) {
        CK(hipMemcpy((char*)dW + (size_t)c * wuv_b, in.W.data(), wuv_b, hipMemcpyHostToDevice));
        CK(hipMemcpy((char*)dOp + (size_t)c * op_b, in.Op.data(), op_b, hipMemcpyHostToDevice));
        CK(hipMemcpy((char*)dMl + (size_t)c * ml_b, in.Ml.data(), ml_b, hipMemcpyHostToDevice));
    }
    printf("grid=%d  nh=%u DK=%u V=%u ns=%u   W_uv=%.2f MB/packet, partials %.2f MB\n", grid, NH,
           DK, VD, NS, wuv_b / 1048576.0, op_b / 1048576.0);
    printf("%-20s %14s %14s\n", "variant", "us/pkt(hot)", "us/pkt(cold)");
    for (int k = 0; k < NK; k++) {
        double us[2];
        for (int cold = 0; cold < 2; cold++) {
            unsigned nh = NH, v = VD, ns = NS;
            void* Wp = dW;
            void* Opp = dOp;
            void* Mlp = dMl;
            void* a[] = {&dO, &Opp, &Mlp, &Wp, &nh, &v, &ns};
            const int IT = 400;
            auto rot = [&](int i) {
                Wp = (char*)dW + (size_t)(i % NCOPY) * wuv_b;
                if (cold) {
                    Opp = (char*)dOp + (size_t)(i % NCOPY) * op_b;
                    Mlp = (char*)dMl + (size_t)(i % NCOPY) * ml_b;
                }
            };
            for (int i = 0; i < 20; i++) {
                rot(i);
                CK(hipModuleLaunchKernel(F[k], grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
            }
            CK(hipDeviceSynchronize());
            hipEvent_t s, e;
            CK(hipEventCreate(&s));
            CK(hipEventCreate(&e));
            CK(hipEventRecord(s, 0));
            for (int i = 0; i < IT; i++) {
                rot(i);
                CK(hipModuleLaunchKernel(F[k], grid, 1, 1, T, 1, 1, 0, 0, a, nullptr));
            }
            CK(hipEventRecord(e, 0));
            CK(hipEventSynchronize(e));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, s, e));
            us[cold] = (double)ms * 1000.0 / IT;
            CK(hipEventDestroy(s));
            CK(hipEventDestroy(e));
        }
        printf("%-20s %14.2f %14.2f\n", names[k], us[0], us[1]);
    }
    return fails ? 1 : 0;
}
