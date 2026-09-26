/* mla_bmm_fp8_pf_gfx950.hip — prefill MlaBmmFp8 (mla_bmm_fp8_pf.h) against the production
 * d_mla_bmm_fp8_m16 at GLM-5.3 TP8 shapes: W_UK (8 heads, N=512, K=192, head stride 256, 64-col
 * rope copy) and W_UV (8 heads, N=256, K=512). Interpreter geometry: 256 x PLOW_THREADS.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/mla_bmm_fp8_pf_gfx950.hip -o /tmp/mla_bmm_pf
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
#include "mla_bmm_fp8_pf.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

template <int K, int N>
__global__ __launch_bounds__(PLOW_THREADS) void k_ref(bf16* C, const bf16* X, const unsigned char* W,
                                                      const float* ws, unsigned M, unsigned xs, bf16* rope) {
#ifdef __HIP_DEVICE_COMPILE__
    if (rope) d_mla_bmm_fp8_m16<true>(C, X, W, ws, M, 8, N, K, blockIdx.x, gridDim.x, xs, rope, 64);
    else d_mla_bmm_fp8_m16(C, X, W, ws, M, 8, N, K, blockIdx.x, gridDim.x, xs);
#endif
}
template <int K, int N>
__global__ __launch_bounds__(PLOW_THREADS) void k_pf(bf16* C, const bf16* X, const unsigned char* W,
                                                     const float* ws, unsigned M, unsigned xs, bf16* rope) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ unsigned char lds[32 * ((K + 127) / 128) * 128 + 32 * ((K + 127) / 128) * 4];
    d_mla_bmm_fp8_pf<K, N>(C, X, W, ws, M, 8, blockIdx.x, gridDim.x, xs, rope, rope ? 64 : 0, lds);
#endif
}

static unsigned short h_f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float h_bf(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

template <int K, int N>
static int run(const char* name, unsigned M, unsigned xs, bool with_rope) {
    const unsigned H = 8;
    std::vector<unsigned short> x((size_t)M * H * xs);
    std::vector<unsigned char> w((size_t)H * N * K);
    srand(11 + M);
    for (auto& v : x) v = h_f2bf((float)(rand() % 2001 - 1000) / 300.0f);
    for (auto& v : w) { unsigned char b; do { b = (unsigned char)(rand() & 0xff); } while ((b & 0x7f) == 0x7f); v = b; }
    const float wsv = 0.0123f;
    bf16 *dx, *dref, *dout, *rr = nullptr, *rp = nullptr; unsigned char* dw; float* dws;
    CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&dw, w.size())); CK(hipMalloc(&dws, 4));
    CK(hipMalloc(&dref, (size_t)M * H * N * 2)); CK(hipMalloc(&dout, (size_t)M * H * N * 2));
    if (with_rope) { CK(hipMalloc(&rr, (size_t)M * H * 64 * 2)); CK(hipMalloc(&rp, (size_t)M * H * 64 * 2)); }
    CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dws, &wsv, 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](auto launch) {
        std::vector<float> v;
        for (int it = 0; it < 21; it++) {
            CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) v.push_back(ms);
        }
        std::sort(v.begin(), v.end()); return v[v.size() / 2];
    };
    const float tr = timed([&] { hipLaunchKernelGGL((k_ref<K, N>), dim3(256), dim3(PLOW_THREADS), 0, 0, dref, dx, dw, dws, M, xs, rr); });
    const float tp = timed([&] { hipLaunchKernelGGL((k_pf<K, N>), dim3(256), dim3(PLOW_THREADS), 0, 0, dout, dx, dw, dws, M, xs, rp); });
    std::vector<unsigned short> a((size_t)M * H * N), b(a.size());
    CK(hipMemcpy(a.data(), dref, a.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), dout, b.size() * 2, hipMemcpyDeviceToHost));
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = h_bf(b[i]) - h_bf(a[i]); e += d * d; n += (double)h_bf(a[i]) * h_bf(a[i]); }
    const double rl2 = sqrt(e / n);
    bool rope_ok = true;
    if (with_rope) {
        std::vector<unsigned short> ra((size_t)M * H * 64), rb(ra.size());
        CK(hipMemcpy(ra.data(), rr, ra.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(rb.data(), rp, rb.size() * 2, hipMemcpyDeviceToHost));
        rope_ok = ra == rb;
    }
    const bool ok = std::isfinite(rl2) && rl2 < 2e-3 && rope_ok;
    printf("%-5s M=%-6u pf %8.4f ms  ref %8.4f ms  %5.1fx  rel-L2 %.2e rope %s  %s\n", name, M, tp, tr, tr / tp, rl2,
           with_rope ? (rope_ok ? "same" : "DIFF") : "-", ok ? "PASS" : "FAIL");
    hipFree(dx); hipFree(dw); hipFree(dws); hipFree(dref); hipFree(dout); if (rr) { hipFree(rr); hipFree(rp); }
    return ok ? 0 : 1;
}

int main() {
    int bad = 0;
    for (unsigned M : {128u, 1024u, 8192u, 16384u}) {
        bad += run<192, 512>("W_UK", M, 256, true);
        bad += run<512, 256>("W_UV", M, 512, false);
    }
    return bad ? 2 : 0;
}
