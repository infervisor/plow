/* gemv_f32_wg_gfx950.hip — decode mHC mix GEMV (bf16 x, f32 W, f32 out) at GLM-5.3-Flash
 * (N=24, K=16384): d_gemv_f32 (wave per output) vs d_gemv_f32_wg (workgroup per output).
 * Bit-exact + timing.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/gemv_f32_wg_gfx950.hip -o /tmp/gemv_f32_wg
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#endif
#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned N = 24, K = 16384;
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_old(float* C, const bf16* x, const float* W, unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    d_gemv_f32(C, x, W, M, N, K, blockIdx.x, gridDim.x);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_wg(float* C, const bf16* x, const float* W, unsigned M) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ float part[K / 8];
    d_gemv_f32_wg(C, x, W, M, N, K, blockIdx.x, gridDim.x, part);
#endif
}
int main() {
    std::vector<unsigned short> x(16 * K);
    std::vector<float> W((size_t)N * K);
    srand(7);
    auto rnd = [] { return (rand() % 2000001 - 1000000) / 1000000.0f; };
    for (auto& v : x) { float f = rnd() * (1 << (rand() % 8)); unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
    for (auto& v : W) v = rnd() / (1 << (rand() % 12));
    bf16* dx; float *dW, *c0, *c1;
    CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&dW, W.size() * 4));
    CK(hipMalloc(&c0, 16 * N * 4)); CK(hipMalloc(&c1, 16 * N * 4));
    CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, W.data(), W.size() * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int bad = 0;
    for (unsigned M : {1u, 2u, 8u}) {
        auto timed = [&](auto launch) {
            std::vector<float> t;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) t.push_back(ms);
            }
            std::sort(t.begin(), t.end()); return t[t.size() / 2];
        };
        CK(hipMemset(c0, 0xff, 16 * N * 4)); CK(hipMemset(c1, 0x7f, 16 * N * 4));
        const float t0 = timed([&] { hipLaunchKernelGGL(k_old, dim3(256), dim3(PLOW_THREADS), 0, 0, c0, dx, dW, M); });
        const float t1 = timed([&] { hipLaunchKernelGGL(k_wg, dim3(256), dim3(PLOW_THREADS), 0, 0, c1, dx, dW, M); });
        std::vector<float> a(M * N), b(M * N);
        CK(hipMemcpy(a.data(), c0, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), c1, b.size() * 4, hipMemcpyDeviceToHost));
        const int diff = (int)(memcmp(a.data(), b.data(), a.size() * 4) != 0);
        bad += diff;
        printf("M=%-2u old %.4f ms  wg %.4f ms  %.2fx  %s  c[0]=%.9g\n", M, t0, t1, t0 / t1,
               diff ? "MISMATCH" : "bit-exact", a[0]);
    }
    return bad ? 2 : 0;
}
