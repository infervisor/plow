/* gemm_f32_smalln_gfx950.hip — router GEMM (bf16 x bf16 -> f32, N=256, K=6144) small-N arm
 * (gemm_f32_smalln.h) against the interpreter's d_gemm_f32 and an f64 CPU reference.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/gemm_f32_smalln_gfx950.hip -o /tmp/gemm_f32_smalln
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
#include "gemm_f32_smalln.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_ref(float* C, const bf16* A, const bf16* B,
                                                                 unsigned M, unsigned N, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm_f32(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_new(float* C, const bf16* A, const bf16* B,
                                                                 unsigned M, unsigned N, unsigned K) {
#ifdef __HIP_DEVICE_COMPILE__
    d_gemm_f32_smalln(C, A, B, M, N, K, blockIdx.x, gridDim.x);
#endif
}

static unsigned short h_f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float h_bf(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
    const unsigned N = 256, K = 6144;
    int bad_total = 0;
    for (unsigned M : {64u, 128u, 512u, 1024u, 2048u, 8192u}) {
        std::vector<unsigned short> a((size_t)M * K), b((size_t)N * K);
        srand(M);
        for (auto& x : a) x = h_f2bf((float)(rand() % 2001 - 1000) / 1000.0f);
        for (auto& x : b) x = h_f2bf((float)(rand() % 2001 - 1000) / 8000.0f);
        bf16 *da, *db; float *dr, *dn;
        CK(hipMalloc(&da, a.size() * 2)); CK(hipMalloc(&db, b.size() * 2));
        CK(hipMalloc(&dr, (size_t)M * N * 4)); CK(hipMalloc(&dn, (size_t)M * N * 4));
        CK(hipMemcpy(da, a.data(), a.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        auto timed = [&](auto launch) {
            std::vector<float> v;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) v.push_back(ms);
            }
            std::sort(v.begin(), v.end()); return v[v.size() / 2];
        };
        const float tr = timed([&] { hipLaunchKernelGGL(k_ref, dim3(256), dim3(PLOW_THREADS), 0, 0, dr, da, db, M, N, K); });
        const float tn = timed([&] { hipLaunchKernelGGL(k_new, dim3(256), dim3(PLOW_THREADS), 0, 0, dn, da, db, M, N, K); });
        std::vector<float> r((size_t)M * N), n(r.size());
        CK(hipMemcpy(r.data(), dr, r.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(n.data(), dn, n.size() * 4, hipMemcpyDeviceToHost));
        double worst_new = 0, worst_ref = 0;
        for (int s = 0; s < 512; s++) {
            const unsigned m = rand() % M, c = rand() % N;
            double acc = 0;
            for (unsigned k = 0; k < K; k++) acc += (double)h_bf(a[(size_t)m * K + k]) * h_bf(b[(size_t)c * K + k]);
            const double den = fabs(acc) + 1e-3;
            worst_new = fmax(worst_new, fabs(n[(size_t)m * N + c] - acc) / den);
            worst_ref = fmax(worst_ref, fabs(r[(size_t)m * N + c] - acc) / den);
        }
        const bool ok = worst_new < 1e-4;
        bad_total += !ok;
        printf("M=%-5u new %8.4f ms  ref %8.4f ms  %6.1fx  worst rel vs f64: new %.2e ref %.2e  %s\n", M, tn, tr,
               tr / tn, worst_new, worst_ref, ok ? "PASS" : "FAIL");
        hipFree(da); hipFree(db); hipFree(dr); hipFree(dn);
    }
    return bad_total ? 2 : 0;
}
