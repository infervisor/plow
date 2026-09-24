/* flash_merge_flat_gfx950.hip — d_flash_merge<512> single-split flat arm (flat=1) against the
 * (row, head) item arm (flat=0) at GLM-5.3 TP8 prefill shapes (8 heads). Bit-exact + timing.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/flash_merge_flat_gfx950.hip -o /tmp/flash_merge_flat
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_attention.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_merge(bf16* O, const float* Op, const float* ml,
                                                                   unsigned nb, unsigned flat) {
#ifdef __HIP_DEVICE_COMPILE__
    d_flash_merge<512>(O, Op, ml, nb, 8, 1, blockIdx.x, gridDim.x, nullptr, flat);
#endif
}

int main() {
    int bad_total = 0;
    for (unsigned T : {1024u, 8192u, 16384u}) {
        const size_t rows = (size_t)T * 8;
        std::vector<float> op(rows * 512), ml(rows * 2);
        srand(T);
        for (auto& x : op) x = (float)(rand() % 20001 - 10000) / 997.0f;
        for (size_t r = 0; r < rows; r++) { ml[r * 2] = (r % 97 == 0) ? -INFINITY : 0.25f * (r % 5); ml[r * 2 + 1] = 1.0f + (r % 11); }
        float *dop, *dml; bf16 *da, *db;
        CK(hipMalloc(&dop, op.size() * 4)); CK(hipMalloc(&dml, ml.size() * 4));
        CK(hipMalloc(&da, rows * 512 * 2)); CK(hipMalloc(&db, rows * 512 * 2));
        CK(hipMemcpy(dop, op.data(), op.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dml, ml.data(), ml.size() * 4, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        auto timed = [&](bf16* out, unsigned flat) {
            std::vector<float> v;
            for (int it = 0; it < 21; it++) {
                CK(hipEventRecord(e0));
                hipLaunchKernelGGL(k_merge, dim3(256), dim3(PLOW_THREADS), 0, 0, out, dop, dml, T, flat);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) v.push_back(ms);
            }
            std::sort(v.begin(), v.end()); return v[v.size() / 2];
        };
        const float ti = timed(da, 0), tf = timed(db, 1);
        std::vector<unsigned short> a(rows * 512), b(a.size());
        CK(hipMemcpy(a.data(), da, a.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), db, b.size() * 2, hipMemcpyDeviceToHost));
        const bool ok = a == b;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i]) { printf("  first diff @%zu item %04x flat %04x\n", i, a[i], b[i]); break; }
        bad_total += !ok;
        printf("T=%-6u item %8.4f ms  flat %8.4f ms  %5.1fx  %s\n", T, ti, tf, ti / tf, ok ? "BIT-EXACT" : "DIFF");
        hipFree(dop); hipFree(dml); hipFree(da); hipFree(db);
    }
    return bad_total ? 2 : 0;
}
