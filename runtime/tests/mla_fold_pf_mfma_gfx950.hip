/* mla_fold_pf_mfma_gfx950.hip — MFMA prefill value fold (mla_fold_pf_mfma.h) against the
 * production scalar fold (d_mla_merge_fold<512,256>) at GLM-5.3 TP8 prefill shapes: 8 heads,
 * V=256, one split. Interpreter launch geometry: 256 workgroups x PLOW_THREADS.
 *
 *   nix develop --command hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/mla_fold_pf_mfma_gfx950.hip -o /tmp/mla_fold_pf_mfma
 *   /tmp/mla_fold_pf_mfma [T...]        (through the GPU queue)
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__  /* the attention headers are device-only; the host pass sees stubs */
#include "op_attention.h"
#include "mla_fold_pf_mfma.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr unsigned NH = 8, V = 256, DK = 512, GRID = 256;

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_ref(bf16* O, const float* Op, const float* ml,
                                                                 const bf16* W, unsigned nb) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ float raw[DK + PLOW_WAVES * 256];
    d_mla_merge_fold<512, 256>(O, Op, ml, W, nb, NH, V, 1u, blockIdx.x, gridDim.x, raw);
#endif
}
template <int WM>
__global__ __launch_bounds__(PLOW_THREADS) void k_mfma(bf16* O, const float* Op, const float* ml,
                                                       const bf16* W, unsigned nb) {
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ bf16 lds[V * 40];
    d_mla_fold_pf_mfma<WM, 256>(O, Op, ml, W, nb, NH, blockIdx.x, gridDim.x, lds);
#endif
}

static float bf(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    std::vector<unsigned> ts;
    for (int i = 1; i < argc; i++) ts.push_back((unsigned)atoi(argv[i]));
    if (ts.empty()) ts = {1024, 2048, 4096, 8192, 16384};
    int bad_total = 0;
    for (unsigned T : ts) {
        const size_t rows = (size_t)T * NH;
        std::vector<float> op(rows * DK), ml(rows * 2);
        std::vector<unsigned short> w((size_t)NH * DK * V);
        srand(7);
        for (auto& x : op) x = (float)(rand() % 2001 - 1000) / 500.0f;
        for (size_t r = 0; r < rows; r++) { ml[r * 2] = 0.5f; ml[r * 2 + 1] = 1.0f + (r % 13) * 0.25f; }
        for (auto& x : w) { float f = (float)(rand() % 2001 - 1000) / 16000.0f; unsigned u; memcpy(&u, &f, 4); x = (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
        float *dop, *dml; bf16 *dw, *dref, *dout;
        CK(hipMalloc(&dop, op.size() * 4)); CK(hipMalloc(&dml, ml.size() * 4));
        CK(hipMalloc(&dw, w.size() * 2)); CK(hipMalloc(&dref, rows * V * 2)); CK(hipMalloc(&dout, rows * V * 2));
        CK(hipMemcpy(dop, op.data(), op.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dml, ml.data(), ml.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        auto timed = [&](auto launch) {
            std::vector<float> v;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) v.push_back(ms);
            }
            std::sort(v.begin(), v.end()); return v[v.size() / 2];
        };
        const float tref = timed([&] { hipLaunchKernelGGL(k_ref, dim3(GRID), dim3(PLOW_THREADS), 0, 0, dref, dop, dml, dw, T); });
        std::vector<unsigned short> ref(rows * V), got(rows * V);
        CK(hipMemcpy(ref.data(), dref, ref.size() * 2, hipMemcpyDeviceToHost));
        auto check = [&](const char* name, float ms) {
            CK(hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
            double e = 0, n = 0;
            for (size_t i = 0; i < got.size(); i++) { double d = bf(got[i]) - bf(ref[i]); e += d * d; n += (double)bf(ref[i]) * bf(ref[i]); }
            const double rl2 = sqrt(e / n);
            const bool ok = std::isfinite(rl2) && rl2 < 5e-3;
            bad_total += !ok;
            printf("T=%-6u %-10s %8.4f ms  (ref %8.4f ms, %5.1fx)  rel-L2 %.2e %s\n", T, name, ms, tref, tref / ms, rl2, ok ? "PASS" : "FAIL");
        };
        check("mfma WM=2", timed([&] { hipLaunchKernelGGL(k_mfma<2>, dim3(GRID), dim3(PLOW_THREADS), 0, 0, dout, dop, dml, dw, T); }));
        check("mfma WM=1", timed([&] { hipLaunchKernelGGL(k_mfma<1>, dim3(GRID), dim3(PLOW_THREADS), 0, 0, dout, dop, dml, dw, T); }));
        hipFree(dop); hipFree(dml); hipFree(dw); hipFree(dref); hipFree(dout);
    }
    return bad_total ? 2 : 0;
}
