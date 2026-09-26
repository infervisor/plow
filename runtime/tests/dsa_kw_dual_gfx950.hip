/* dsa_kw_dual_gfx950.hip — d_gemm_small_dual (PLOW_DSA_PREP) vs two d_gemm_small launches at the GLM
 * DSA indexer projections (wk N=128 and weights_proj N=32, K=6144), 256 workgroups x 512 threads.
 * Outputs must be byte-identical. Timing per call.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/dsa_kw_dual_gfx950.hip -o dsa_kw_dual
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#include "op_norm.h"
#include "dsa_prep.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(512, 1) void k_small(bf16* c, const bf16* a, const bf16* b, unsigned m,
                                                            unsigned n, unsigned k) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_small(c, a, b, m, n, k, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_dual(bf16* c0, bf16* c1, const bf16* a, const bf16* b0,
                                                           const bf16* b1, unsigned m, unsigned n0, unsigned n1,
                                                           unsigned k) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_small_dual(c0, c1, a, b0, b1, m, n0, n1, k, blockIdx.x, gridDim.x, lds);
#endif
}

int main() {
    const unsigned MT = 16384, K = 6144, N0 = 128, N1 = 32;
    unsigned long long r = 88172645463325252ull;
    auto fill = [&](std::vector<unsigned short>& v, float sc) {
        for (auto& e : v) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            float f = sc * ((int)((r >> 40) & 0xFFFF) - 32768) / 32768.0f;
            unsigned u; memcpy(&u, &f, 4); e = u >> 16;
        }
    };
    std::vector<unsigned short> ha((size_t)MT * K), hb0((size_t)N0 * K), hb1((size_t)N1 * K);
    fill(ha, 4.0f); fill(hb0, 0.05f); fill(hb1, 0.05f);
    bf16 *a, *b0, *b1, *s0, *s1, *d0, *d1;
    CK(hipMalloc(&a, ha.size() * 2)); CK(hipMalloc(&b0, hb0.size() * 2)); CK(hipMalloc(&b1, hb1.size() * 2));
    CK(hipMalloc(&s0, (size_t)MT * N0 * 2)); CK(hipMalloc(&s1, (size_t)MT * N1 * 2));
    CK(hipMalloc(&d0, (size_t)MT * N0 * 2)); CK(hipMalloc(&d1, (size_t)MT * N1 * 2));
    CK(hipMemcpy(a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(b0, hb0.data(), hb0.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(b1, hb1.data(), hb1.size() * 2, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    for (unsigned m : {4096u, 8192u, 16384u, 8000u, 1000u}) {
        CK(hipMemset(s0, 0x5a, (size_t)MT * N0 * 2)); CK(hipMemset(d0, 0xa5, (size_t)MT * N0 * 2));
        CK(hipMemset(s1, 0x5a, (size_t)MT * N1 * 2)); CK(hipMemset(d1, 0xa5, (size_t)MT * N1 * 2));
        float ms[2];
        for (int k = 0; k < 2; k++) {
            auto run = [&]() {
                if (k == 0) {
                    hipLaunchKernelGGL(k_small, dim3(256), dim3(512), 0, 0, s0, a, b0, m, N0, K);
                    hipLaunchKernelGGL(k_small, dim3(256), dim3(512), 0, 0, s1, a, b1, m, N1, K);
                } else {
                    hipLaunchKernelGGL(k_dual, dim3(256), dim3(512), 0, 0, d0, d1, a, b0, b1, m, N0, N1, K);
                }
            };
            run();
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(e0));
            for (int i = 0; i < 10; i++) run();
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[k], e0, e1)); ms[k] /= 10;
        }
        std::vector<unsigned short> x0((size_t)m * N0), y0((size_t)m * N0), x1((size_t)m * N1), y1((size_t)m * N1);
        CK(hipMemcpy(x0.data(), s0, x0.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(y0.data(), d0, y0.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(x1.data(), s1, x1.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(y1.data(), d1, y1.size() * 2, hipMemcpyDeviceToHost));
        const bool same = !memcmp(x0.data(), y0.data(), x0.size() * 2) && !memcmp(x1.data(), y1.data(), x1.size() * 2);
        printf("M=%5u  wk+wp separate %7.1f us  dual %7.1f us  %s\n", m, ms[0] * 1e3, ms[1] * 1e3,
               same ? "identical" : "DIFFER");
        fails += !same;
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
