/* qkvfp8_test.hip — d_gemv_fp8 (op 30) and d_gemv_qkv_fp8 (op 115) vs a CPU f64 reference.
 *
 * TWO PASSES (op_gemm.h is device-only):
 *   hipcc --offload-arch=gfx942 -O3 -w -DPLOW_GEMV_MM=1 -DQKV_DEVICE --genco \
 *       qkvfp8_test.hip -o /tmp/qkv.co -I<repo>/runtime/amd -I<repo>/runtime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 qkvfp8_test.hip -o /tmp/qkv_test -lamdhip64
 *   /tmp/qkv_test /tmp/qkv.co
 */

struct KArgs {
    unsigned long long Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv;
    unsigned M, Nq, Nk, Nv, K, mode; /* mode 0: d_gemv_fp8 on (q only); 1: d_gemv_qkv_fp8 */
};

#ifdef QKV_DEVICE
#include <hip/hip_runtime.h>
#include "op_gemm.h"

extern "C" __global__ __launch_bounds__(512) void k_qkv(KArgs a) {
    __shared__ bf16 lds[16384];
    if (a.mode == 0) {
        d_gemv_fp8((bf16*)a.Cq, (const bf16*)a.x, (const unsigned char*)a.Wq, (const float*)a.Sq,
                   a.M, a.Nq, a.K, blockIdx.x, gridDim.x, lds);
    } else {
        d_gemv_qkv_fp8((bf16*)a.Cq, (bf16*)a.Ck, (bf16*)a.Cv, (const bf16*)a.x,
                       (const unsigned char*)a.Wq, (const unsigned char*)a.Wk,
                       (const unsigned char*)a.Wv, (const float*)a.Sq, (const float*)a.Sk,
                       (const float*)a.Sv, a.M, a.Nq, a.Nk, a.Nv, a.K, blockIdx.x, gridDim.x, lds);
    }
}
#else
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(e) do { hipError_t r = (e); if (r) { printf("ERR %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(r)); exit(1); } } while (0)

typedef unsigned short u16;
static float bf2f(u16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static u16 f2bf(float f) { unsigned u; memcpy(&u, &f, 4); unsigned r = (u + 0x7fff + ((u >> 16) & 1)) >> 16; return (u16)r; }
/* OCP e4m3: 1s 4e 3m, bias 7; the device dequant path yields the OCP value exactly. */
static float e4m3(unsigned char b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    if (e == 0xF && m == 7) return NAN;
    float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}

int main(int argc, char** argv) {
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, "k_qkv"));

    const unsigned M = 1, Nq = 4096, Nk = 2048, Nv = 2048, K = 3840, N = Nq + Nk + Nv;
    std::vector<unsigned char> hW(N * (size_t)K);
    std::vector<u16> hx(K);
    std::vector<float> hS(N);
    srand(7);
    for (auto& w : hW) { do { w = rand() & 0xFF; } while ((w & 0x7F) == 0x7F); }
    for (auto& v : hx) v = f2bf(((rand() % 2000) - 1000) / 500.0f);
    for (auto& s : hS) s = ((rand() % 100) + 1) / 50.0f;

    unsigned char *dW; u16 *dx, *dC; float *dS;
    CK(hipMalloc(&dW, hW.size())); CK(hipMalloc(&dx, K * 2)); CK(hipMalloc(&dC, N * 2)); CK(hipMalloc(&dS, N * 4));
    CK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dx, hx.data(), K * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dS, hS.data(), N * 4, hipMemcpyHostToDevice));

    /* CPU reference, f64. Concatenated column n reads row `col` of its own W with its own scale. */
    std::vector<float> ref(N);
    for (unsigned n = 0; n < N; n++) {
        double acc = 0;
        for (unsigned k = 0; k < K; k++) acc += (double)e4m3(hW[(size_t)n * K + k]) * (double)bf2f(hx[k]);
        ref[n] = (float)(acc * hS[n]);
    }

    auto run = [&](unsigned mode, const char* name, unsigned nq, unsigned nk, unsigned nv) {
        CK(hipMemset(dC, 0xEE, N * 2));
        KArgs a{};
        a.x = (unsigned long long)dx; a.M = M; a.K = K; a.mode = mode;
        a.Cq = (unsigned long long)dC;
        a.Wq = (unsigned long long)dW;
        a.Sq = (unsigned long long)dS;
        a.Nq = nq; a.Nk = nk; a.Nv = nv;
        if (mode == 1) {
            a.Ck = (unsigned long long)(dC + nq);
            a.Cv = (unsigned long long)(dC + nq + nk);
            a.Wk = (unsigned long long)(dW + (size_t)nq * K);
            a.Wv = (unsigned long long)(dW + (size_t)(nq + nk) * K);
            a.Sk = (unsigned long long)(dS + nq);
            a.Sv = (unsigned long long)(dS + nq + nk);
        }
        size_t sz = sizeof(a);
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CK(hipModuleLaunchKernel(fn, 304, 1, 1, 512, 1, 1, 0, 0, nullptr, cfg));
        CK(hipDeviceSynchronize());
        std::vector<u16> hC(N);
        CK(hipMemcpy(hC.data(), dC, N * 2, hipMemcpyDeviceToHost));
        unsigned total = nq + nk + nv, bad = 0;
        double worst = 0; unsigned worstn = 0;
        for (unsigned n = 0; n < total; n++) {
            float g = bf2f(hC[n]);
            float r = ref[n];
            double rel = std::fabs(g - r) / (std::fabs(r) + 1e-3);
            if (rel > worst) { worst = rel; worstn = n; }
            if (rel > 0.02) bad++;
        }
        printf("%-14s cols=%u bad=%u worst_rel=%.4g at n=%u (got %.4f want %.4f)\n", name, total,
               bad, worst, worstn, bf2f(hC[worstn]), ref[worstn]);
        return bad;
    };

    unsigned fail = 0;
    fail += run(0, "gemv_fp8", Nq, 0, 0);            /* op 30: q block only */
    fail += run(0, "gemv_fp8_wide", N, 0, 0);        /* op 30 over all rows (odd-tail cases) */
    fail += run(1, "qkv_fp8", Nq, Nk, Nv);           /* op 115: three streams */
    fail += run(1, "qkv_fp8_2s", Nq, Nk, 0);         /* op 115: two-stream form */
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
#endif
