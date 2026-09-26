/* moe_align_pf_phases_gfx950.hip — the multi-packet d_moe_align_pf (phases 1-4) at GLM-5.3 TP8
 * prefill shape (257 experts incl. the shared fold, k=9, 64 partitions, 512 threads), timed per
 * phase, with a fingerprint of the row arrays so two builds (e.g. -DPLOW_MOE_ALIGN_WAVES={0,1})
 * can be compared bitwise.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common [-DPLOW_MOE_ALIGN_WAVES=1] \
 *       runtime/tests/moe_align_pf_phases_gfx950.hip -o /tmp/align
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#include "op_moe.h"

#define HC(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_align(int* meta, const unsigned char* tab,
                                                                    unsigned* rt, unsigned* rp, float* rg,
                                                                    unsigned T, unsigned n_exp, unsigned k,
                                                                    unsigned phase, unsigned npart) {
    __shared__ unsigned lds[8192];
    d_moe_align_pf(meta, tab, rt, rp, rg, T, n_exp, k, blockIdx.x, lds, phase, gridDim.x, npart);
}

int main() {
    const unsigned n_exp = 257, k = 9, npart = 64;
    for (unsigned T : {1024u, 8192u, 16384u}) {
        const unsigned nslot = T * k;
        std::vector<unsigned char> tab((size_t)nslot * 8);
        unsigned st = 12345u + T;
        auto rnd = [&] { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
        for (unsigned t = 0; t < T; t++)
            for (unsigned j = 0; j < k; j++) {
                const unsigned e = j == 8 ? 256u : rnd() % 256u;
                const float g = (float)(rnd() % 1000) / 1000.0f;
                memcpy(&tab[((size_t)t * k + j) * 8], &e, 4);
                memcpy(&tab[((size_t)t * k + j) * 8 + 4], &g, 4);
            }
        const size_t rows = (size_t)nslot + n_exp * 63;
        unsigned char* dtab; int* dmeta; unsigned *drt, *drp; float* drg;
        HC(hipMalloc(&dtab, tab.size()));
        HC(hipMemcpy(dtab, tab.data(), tab.size(), hipMemcpyHostToDevice));
        HC(hipMalloc(&dmeta, (3 * n_exp + 1 + (size_t)npart * n_exp) * 4));
        HC(hipMalloc(&drt, rows * 4)); HC(hipMalloc(&drp, rows * 4)); HC(hipMalloc(&drg, rows * 4));
        hipEvent_t ev[5];
        for (auto& e : ev) HC(hipEventCreate(&e));
        std::vector<float> ph[4];
        for (int it = 0; it < 21; it++) {
            HC(hipEventRecord(ev[0]));
            k_align<<<npart, PLOW_THREADS>>>(dmeta, dtab, drt, drp, drg, T, n_exp, k, 1, npart);
            HC(hipEventRecord(ev[1]));
            k_align<<<1, PLOW_THREADS>>>(dmeta, dtab, drt, drp, drg, T, n_exp, k, 2, npart);
            HC(hipEventRecord(ev[2]));
            k_align<<<npart, PLOW_THREADS>>>(dmeta, dtab, drt, drp, drg, T, n_exp, k, 3, npart);
            HC(hipEventRecord(ev[3]));
            k_align<<<npart, PLOW_THREADS>>>(dmeta, dtab, drt, drp, drg, T, n_exp, k, 4, npart);
            HC(hipEventRecord(ev[4]));
            HC(hipEventSynchronize(ev[4]));
            for (int p = 0; p < 4; p++) {
                float ms; HC(hipEventElapsedTime(&ms, ev[p], ev[p + 1]));
                if (it) ph[p].push_back(ms);
            }
        }
        std::vector<int> meta(3 * n_exp + 1);
        HC(hipMemcpy(meta.data(), dmeta, meta.size() * 4, hipMemcpyDeviceToHost));
        const size_t pad = (size_t)meta[3 * n_exp] * 64;
        std::vector<unsigned> rt(pad), rp(pad); std::vector<float> rg(pad);
        HC(hipMemcpy(rt.data(), drt, pad * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(rp.data(), drp, pad * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(rg.data(), drg, pad * 4, hipMemcpyDeviceToHost));
        unsigned long long fp = 1469598103934665603ull;
        auto mix = [&](const void* p, size_t n) {
            const unsigned char* b = (const unsigned char*)p;
            for (size_t i = 0; i < n; i++) fp = (fp ^ b[i]) * 1099511628211ull;
        };
        mix(meta.data(), meta.size() * 4); mix(rt.data(), pad * 4); mix(rp.data(), pad * 4); mix(rg.data(), pad * 4);
        size_t live = 0;
        for (size_t r = 0; r < pad; r++) live += rp[r] != PLOW_EXPERT_UNUSED;
        for (auto& v : ph) std::sort(v.begin(), v.end());
        printf("ALIGN_WAVES=%d T=%-5u phase us: %.1f %.1f %.1f %.1f  live=%zu/%u  fingerprint %016llx\n",
               PLOW_MOE_ALIGN_WAVES, T, ph[0][10] * 1e3, ph[1][10] * 1e3, ph[2][10] * 1e3, ph[3][10] * 1e3,
               live, nslot, fp);
        hipFree(dtab); hipFree(dmeta); hipFree(drt); hipFree(drp); hipFree(drg);
    }
    return 0;
}
