/* rmsnorm_q128_gfx950.hip — d_rmsnorm + d_quant_fp8_block128 vs d_rmsnorm_q128 (byte compare of
 * out, xq and a_scale) at GLM-5.3 shapes. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#include "op_norm.h"
#endif
#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_ref_norm(bf16* out, const bf16* x, const bf16* g,
                                                                    unsigned rows, unsigned feat) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ float part[64];
    d_rmsnorm(out, x, g, rows, feat, 1e-5f, 0u, blockIdx.x, gridDim.x, part);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_ref_q(unsigned char* xq, const bf16* x, float* s,
                                                                 unsigned rows, unsigned feat) {
#ifdef __HIP_DEVICE_COMPILE__
    d_quant_fp8_block128(xq, x, s, rows, feat, blockIdx.x, gridDim.x);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_fused(bf16* out, const bf16* x, const bf16* g,
                                                                 unsigned char* xq, float* s, unsigned rows,
                                                                 unsigned feat) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ float part[64];
    d_rmsnorm_q128(out, x, g, rows, feat, 1e-5f, blockIdx.x, gridDim.x, part, xq, s);
#endif
}

int main() {
    int bad = 0;
    for (unsigned feat : {2048u, 6144u})
        for (unsigned rows : {8u, 1024u}) {
            std::vector<unsigned short> x((size_t)rows * feat), g(feat);
            srand(feat + rows);
            for (auto& v : x) { float f = (rand() % 2001 - 1000) / 250.0f; unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
            for (auto& v : g) { float f = 0.5f + (rand() % 1000) / 1000.0f; unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
            bf16 *dx, *dg, *o1, *o2; unsigned char *q1, *q2; float *s1, *s2;
            const size_t n = (size_t)rows * feat;
            CK(hipMalloc(&dx, n * 2)); CK(hipMalloc(&dg, feat * 2)); CK(hipMalloc(&o1, n * 2)); CK(hipMalloc(&o2, n * 2));
            CK(hipMalloc(&q1, n)); CK(hipMalloc(&q2, n)); CK(hipMalloc(&s1, n / 128 * 4)); CK(hipMalloc(&s2, n / 128 * 4));
            CK(hipMemcpy(dx, x.data(), n * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, g.data(), feat * 2, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(k_ref_norm, dim3(256), dim3(PLOW_THREADS), 0, 0, o1, dx, dg, rows, feat);
            hipLaunchKernelGGL(k_ref_q, dim3(256), dim3(PLOW_THREADS), 0, 0, q1, o1, s1, rows, feat);
            hipLaunchKernelGGL(k_fused, dim3(256), dim3(PLOW_THREADS), 0, 0, o2, dx, dg, q2, s2, rows, feat);
            CK(hipDeviceSynchronize());
            std::vector<unsigned char> a1(n * 2), a2(n * 2), b1(n), b2(n), c1(n / 128 * 4), c2(n / 128 * 4);
            CK(hipMemcpy(a1.data(), o1, n * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(a2.data(), o2, n * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b1.data(), q1, n, hipMemcpyDeviceToHost)); CK(hipMemcpy(b2.data(), q2, n, hipMemcpyDeviceToHost));
            CK(hipMemcpy(c1.data(), s1, n / 32, hipMemcpyDeviceToHost)); CK(hipMemcpy(c2.data(), s2, n / 32, hipMemcpyDeviceToHost));
            const bool okO = a1 == a2, okQ = b1 == b2, okS = c1 == c2;
            size_t fq = 0; while (fq < n && b1[fq] == b2[fq]) fq++;
            size_t fs = 0; while (fs < n / 128 && memcmp(&c1[fs * 4], &c2[fs * 4], 4) == 0) fs++;
            printf("feat=%u rows=%u out %s xq %s scale %s  first xq diff @%zu first scale diff @%zu\n", feat, rows,
                   okO ? "SAME" : "DIFF", okQ ? "SAME" : "DIFF", okS ? "SAME" : "DIFF", fq, fs);
            bad += !(okO && okQ && okS);
        }
    return bad ? 2 : 0;
}
