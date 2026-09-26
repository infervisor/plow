/* moe_router_pf_gfx950.hip — op 83 prefill router tail (d_moe_router_topk_pf) at GLM-5.3 routing
 * (sigmoid, f32 logits, e_score_correction_bias, norm_topk, scale 2.5, top-8 of 256, shared tail),
 * interpreter geometry 256 x PLOW_THREADS. Build twice and compare the dumped tables:
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common -DPLOW_MOE_ROUTER_PF_WAVE={0,1} \
 *       runtime/tests/moe_router_pf_gfx950.hip -o /tmp/router{0,1}
 *   /tmp/router0 OUTDIR ; /tmp/router1 OUTDIR   (writes table_T<t>_<arm>.bin, prints timing)
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_router(unsigned char* table, const float* logit,
                                                                    const float* bias, unsigned T) {
    __shared__ float lds[4096];
    d_moe_router_topk_pf(table, (const bf16*)logit, bias, 256u, 8u, 15u, 2.5f, T, blockIdx.x, gridDim.x, lds,
                         0u, 0u
#if PLOW_MOE_PF_ATOMIC
                         , nullptr, 0u
#endif
#if PLOW_MOE_PF_DET
                         , nullptr, 0u
#endif
                         , nullptr, nullptr, 1u);
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "/tmp";
    for (unsigned T : {1024u, 8192u, 16384u}) {
        std::vector<float> l((size_t)T * 256), b(256);
        srand(T);
        for (auto& x : l) x = (float)(rand() % 20001 - 10000) / 2500.0f;
        for (auto& x : b) x = (float)(rand() % 2001 - 1000) / 20000.0f;
        float *dl, *db; unsigned char* dt;
        CK(hipMalloc(&dl, l.size() * 4)); CK(hipMalloc(&db, 256 * 4)); CK(hipMalloc(&dt, (size_t)T * 9 * 8));
        CK(hipMemcpy(dl, l.data(), l.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, b.data(), 256 * 4, hipMemcpyHostToDevice));
        CK(hipMemset(dt, 0, (size_t)T * 9 * 8));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        std::vector<float> v;
        for (int it = 0; it < 21; it++) {
            CK(hipEventRecord(e0));
            hipLaunchKernelGGL(k_router, dim3(256), dim3(PLOW_THREADS), 0, 0, dt, dl, db, T);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) v.push_back(ms);
        }
        std::sort(v.begin(), v.end());
        std::vector<unsigned char> t((size_t)T * 9 * 8);
        CK(hipMemcpy(t.data(), dt, t.size(), hipMemcpyDeviceToHost));
        const std::string f = out + "/table_T" + std::to_string(T) + "_" + std::to_string(PLOW_MOE_ROUTER_PF_WAVE) + ".bin";
        FILE* fp = fopen(f.c_str(), "wb"); fwrite(t.data(), 1, t.size(), fp); fclose(fp);
        printf("ROUTER_PF_WAVE=%d T=%-6u %8.4f ms -> %s\n", PLOW_MOE_ROUTER_PF_WAVE, T, v[v.size() / 2], f.c_str());
        hipFree(dl); hipFree(db); hipFree(dt);
    }
    return 0;
}
