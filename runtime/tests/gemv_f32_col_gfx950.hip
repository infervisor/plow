/* gemv_f32_col_gfx950.hip — decode router GEMV (bf16 in, f32 out) at GLM-5.3 (N=256, K=6144):
 * d_gemv_bf16_f32 (wave per output) vs d_gemv_bf16_f32_col (wave per column, all rows). */
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
constexpr unsigned N = 256, K = 6144;
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_old(float* C, const bf16* x, const bf16* W, unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    d_gemv_bf16_f32(C, x, W, M, N, K, blockIdx.x, gridDim.x);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_col(float* C, const bf16* x, const bf16* W, unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    if (M <= 8) d_gemv_bf16_f32_col<8>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
    else d_gemv_bf16_f32_col<16>(C, x, W, M, N, K, blockIdx.x, gridDim.x);
#endif
}
int main() {
    std::vector<unsigned short> x(16 * K), W((size_t)N * K);
    srand(3);
    for (auto& v : x) { float f = (rand() % 2001 - 1000) / 1000.0f; unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
    for (auto& v : W) { float f = (rand() % 2001 - 1000) / 30000.0f; unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
    bf16 *dx, *dW; float *c0, *c1;
    CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&dW, W.size() * 2));
    CK(hipMalloc(&c0, 16 * N * 4)); CK(hipMalloc(&c1, 16 * N * 4));
    CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int bad = 0;
    for (unsigned M : {1u, 8u, 16u}) {
        auto timed = [&](auto launch) {
            std::vector<float> t;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) t.push_back(ms);
            }
            std::sort(t.begin(), t.end()); return t[t.size() / 2];
        };
        const float t0 = timed([&] { hipLaunchKernelGGL(k_old, dim3(256), dim3(PLOW_THREADS), 0, 0, c0, dx, dW, M); });
        const float t1 = timed([&] { hipLaunchKernelGGL(k_col, dim3(256), dim3(PLOW_THREADS), 0, 0, c1, dx, dW, M); });
        std::vector<float> a(M * N), b(M * N);
        CK(hipMemcpy(a.data(), c0, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), c1, b.size() * 4, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
        const double rel = sqrt(num / (den + 1e-30));
        bad += !(rel < 1e-5);
        printf("M=%-3u old %.4f ms  col %.4f ms  %.2fx  relL2 %.2e %s\n", M, t0, t1, t0 / t1, rel, rel < 1e-5 ? "OK" : "BAD");
    }
    return bad ? 2 : 0;
}
