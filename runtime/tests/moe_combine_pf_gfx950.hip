/* moe_combine_pf_gfx950.hip — op 87 (d_moe_combine_pf) at GLM-5.3 TP8 prefill shape (k=8,
 * H=6144, f32 part, residual + shared), bit-exact against a CPU sum in the shipped slot order,
 * and timed at the interpreter's geometry (256 x PLOW_THREADS). Build once per arm:
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common -DPLOW_COMBINE_VEC={0,1} \
 *       runtime/tests/moe_combine_pf_gfx950.hip -o /tmp/combine_vec{0,1}
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_combine(bf16* out, const bf16* r, const bf16* s,
                                                                      const float* part, unsigned H, unsigned k,
                                                                      unsigned T) {
    d_moe_combine_pf(out, r, s, part, H, k, T, blockIdx.x, gridDim.x);
}

static unsigned short h_f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float h_bf(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
    const unsigned H = 6144, k = 8;
    int bad_total = 0;
    for (unsigned T : {1024u, 8192u, 16384u}) {
        const size_t n = (size_t)T * H;
        std::vector<unsigned short> r(n), s(n), out(n);
        std::vector<float> p(n * k);
        srand(3);
        for (auto& x : r) x = h_f2bf((float)(rand() % 2001 - 1000) / 250.0f);
        for (auto& x : s) x = h_f2bf((float)(rand() % 2001 - 1000) / 500.0f);
        for (auto& x : p) x = (float)(rand() % 20001 - 10000) / 7777.0f;
        bf16 *dr, *ds, *dout; float* dp;
        CK(hipMalloc(&dr, n * 2)); CK(hipMalloc(&ds, n * 2)); CK(hipMalloc(&dout, n * 2)); CK(hipMalloc(&dp, n * k * 4));
        CK(hipMemcpy(dr, r.data(), n * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(ds, s.data(), n * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dp, p.data(), n * k * 4, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        std::vector<float> tv;
        for (int it = 0; it < 21; it++) {
            CK(hipEventRecord(e0));
            hipLaunchKernelGGL(k_combine, dim3(256), dim3(PLOW_THREADS), 0, 0, dout, dr, ds, dp, H, k, T);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) tv.push_back(ms);
        }
        std::sort(tv.begin(), tv.end());
        CK(hipMemcpy(out.data(), dout, n * 2, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            const size_t tok = i / H, h = i % H;
            float acc = h_bf(r[i]);
            acc += h_bf(s[i]);
            for (unsigned j = 0; j < k; j++) acc += p[(tok * k + j) * H + h];
            if (out[i] != h_f2bf(acc)) bad++;
        }
        const double bytes = n * (2.0 + 2.0 + 2.0) + n * k * 4.0;
        printf("COMBINE_VEC=%d T=%-5u %8.4f ms  %7.1f GB/s  %s (%zu mismatches)\n", PLOW_COMBINE_VEC, T,
               tv[tv.size() / 2], bytes / (tv[tv.size() / 2] * 1e-3) / 1e9, bad ? "FAIL" : "BIT-EXACT", bad);
        bad_total += bad != 0;
        hipFree(dr); hipFree(ds); hipFree(dout); hipFree(dp);
    }
    return bad_total ? 2 : 0;
}
