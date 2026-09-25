/* gemm_fp8b128_kw_gfx950.hip — decode-rows block-FP8 GEMM: d_gemm_fp8_block128_m16 (wave per 16x16
 * tile, all of K) vs d_gemm_fp8_block128_m16_kw (workgroup per tile, K split over the 8 waves) at
 * GLM-5.3 TP8 fused qkv_a (N=2624 = 2048|512|64 split3, K=6144). rel-L2 + timing.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/gemm_fp8b128_kw_gfx950.hip
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned N = 2624, K = 6144, N0 = 2048, N1 = 512;

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_m16(bf16* C, bf16* C1, bf16* C2, const unsigned char* A,
                                                               const unsigned char* B, const float* as, const float* ws,
                                                               unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    d_gemm_fp8_block128_m16<false, false, false, false, false, true>(C, A, B, as, ws, M, N, K, blockIdx.x, gridDim.x,
        nullptr, nullptr, nullptr, 0, nullptr, 0, C1, C2, N0, N1);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_kw(bf16* C, bf16* C1, bf16* C2, const unsigned char* A,
                                                              const unsigned char* B, const float* as, const float* ws,
                                                              unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ float red[PLOW_WAVES * 64 * 4];
    d_gemm_fp8_block128_m16_kw(C, A, B, as, ws, M, N, K, blockIdx.x, gridDim.x, red, C1, C2, N0, N1);
#endif
}

static float b2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
    const unsigned kb = K / 128, ncb = (N + 127) / 128;
    std::vector<unsigned char> A(128 * K), B((size_t)N * K);
    std::vector<float> as(kb * 128), ws(ncb * kb);
    srand(1);
    for (auto& x : A) x = (unsigned char)(rand() & 0x77);
    for (auto& x : B) x = (unsigned char)(rand() & 0x77);
    for (auto& x : as) x = 0.001f * (1 + rand() % 7);
    for (auto& x : ws) x = 0.001f * (1 + rand() % 5);
    unsigned char *dA, *dB; float *das, *dws; bf16 *c[6];
    CK(hipMalloc(&dA, A.size())); CK(hipMalloc(&dB, B.size()));
    CK(hipMalloc(&dws, ws.size() * 4));
    CK(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dws, ws.data(), ws.size() * 4, hipMemcpyHostToDevice));
    for (auto& p : c) CK(hipMalloc(&p, 128 * N * 2));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int bad = 0;
    for (unsigned M : {1u, 8u, 64u, 128u}) {
        CK(hipMalloc(&das, kb * M * 4));
        std::vector<float> asm_(kb * M);
        for (unsigned g = 0; g < kb; g++) for (unsigned m = 0; m < M; m++) asm_[g * M + m] = as[g * 128 + m];
        CK(hipMemcpy(das, asm_.data(), asm_.size() * 4, hipMemcpyHostToDevice));
        auto timed = [&](auto launch) {
            std::vector<float> t;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) t.push_back(ms);
            }
            std::sort(t.begin(), t.end()); return t[t.size() / 2];
        };
        const float t0 = timed([&] { hipLaunchKernelGGL(k_m16, dim3(256), dim3(PLOW_THREADS), 0, 0, c[0], c[1], c[2], dA, dB, das, dws, M); });
        const float t1 = timed([&] { hipLaunchKernelGGL(k_kw, dim3(256), dim3(PLOW_THREADS), 0, 0, c[3], c[4], c[5], dA, dB, das, dws, M); });
        CK(hipGetLastError());
        double num = 0, den = 0;
        const unsigned w[3] = {N0, N1, N - N0 - N1};
        for (int o = 0; o < 3; o++) {
            std::vector<unsigned short> x(M * w[o]), y(M * w[o]);
            CK(hipMemcpy(x.data(), c[o], x.size() * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(y.data(), c[3 + o], y.size() * 2, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < x.size(); i++) { const double a = b2f(x[i]), b = b2f(y[i]); num += (a - b) * (a - b); den += a * a; }
        }
        const double rel = sqrt(num / (den + 1e-30));
        bad += !(rel < 5e-3);
        printf("M=%-4u m16 %.4f ms  kw %.4f ms  %.2fx  relL2 %.2e %s\n", M, t0, t1, t0 / t1, rel, rel < 5e-3 ? "OK" : "BAD");
        hipFree(das);
    }
    return bad ? 2 : 0;
}
