#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#define PLOW_MOE_PREFILL 1
#define PLOW_K3 1
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_moe.h"

#define HC(x)                                                                                   \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_) {                                                                                \
            std::fprintf(stderr, "hip %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                       \
        }                                                                                        \
    } while (0)

extern "C" __global__ void align_k(int* meta, const unsigned char* table, unsigned* row_token,
                                    unsigned* row_partidx, float* row_gate, unsigned T,
                                    unsigned n_exp, unsigned k, unsigned phase) {
    __shared__ unsigned lds[4096];
    d_moe_align_pf(meta, table, row_token, row_partidx, row_gate, T, n_exp, k, blockIdx.x, lds,
                   phase, gridDim.x, 64);
}

extern "C" __global__ void router_k(unsigned char* table, const bf16* logit, unsigned n_exp,
                                     unsigned k, unsigned T) {
    __shared__ float lds[8192];
    d_moe_router_topk_pf(table, logit, nullptr, n_exp, k, 0, 1.0f, T, blockIdx.x, gridDim.x, lds);
}

static unsigned short f2b(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void run(unsigned T) {
    constexpr unsigned E = 896, K = 16, NB = 64;
    int device;
    hipDeviceProp_t prop;
    HC(hipGetDevice(&device));
    HC(hipGetDeviceProperties(&prop, device));
    const unsigned router_blocks = std::min(T, (unsigned)prop.multiProcessorCount);
    const size_t nslot = (size_t)T * K;
    const size_t pad = nslot + E * (MPF_BM - 1u);
    std::vector<unsigned short> logit((size_t)T * E);
    for (size_t i = 0; i < logit.size(); i++)
        logit[i] = f2b((float)((i * 1103515245u + 12345u) & 0xffffu) / 32768.0f - 1.0f);

    bf16* dlogit;
    unsigned char *table, *table_prod;
    int *meta0, *meta1;
    unsigned *rt0, *rt1, *rp0, *rp1;
    float *rg0, *rg1;
    HC(hipMalloc(&dlogit, logit.size() * sizeof(unsigned short)));
    HC(hipMemcpy(dlogit, logit.data(), logit.size() * sizeof(unsigned short), hipMemcpyHostToDevice));
    HC(hipMalloc(&table, nslot * 8));
    HC(hipMalloc(&table_prod, nslot * 8));
    HC(hipMalloc(&meta0, (3 * E + 1) * sizeof(int)));
    HC(hipMalloc(&meta1, (3 * E + 1 + NB * E) * sizeof(int)));
    HC(hipMalloc(&rt0, pad * sizeof(unsigned)));
    HC(hipMalloc(&rt1, pad * sizeof(unsigned)));
    HC(hipMalloc(&rp0, pad * sizeof(unsigned)));
    HC(hipMalloc(&rp1, pad * sizeof(unsigned)));
    HC(hipMalloc(&rg0, pad * sizeof(float)));
    HC(hipMalloc(&rg1, pad * sizeof(float)));

    hipLaunchKernelGGL(router_k, dim3(router_blocks), dim3(PLOW_THREADS), 0, 0, table, dlogit, E, K,
                       T);
    hipLaunchKernelGGL(router_k, dim3(304), dim3(PLOW_THREADS), 0, 0, table_prod, dlogit, E, K, T);
    HC(hipDeviceSynchronize());
    std::vector<unsigned char> htable_prod(nslot * 8), htable_token(nslot * 8);
    HC(hipMemcpy(htable_prod.data(), table_prod, htable_prod.size(), hipMemcpyDeviceToHost));
    HC(hipMemcpy(htable_token.data(), table, htable_token.size(), hipMemcpyDeviceToHost));
    const bool router_exact = htable_prod == htable_token;
    hipLaunchKernelGGL(align_k, dim3(1), dim3(PLOW_THREADS), 0, 0, meta0, table, rt0, rp0, rg0, T,
                       E, K, 0u);
    hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1, T,
                       E, K, 1u);
    hipLaunchKernelGGL(align_k, dim3(1), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1, T,
                       E, K, 2u);
    hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1, T,
                       E, K, 3u);
    hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1, T,
                       E, K, 4u);
    HC(hipDeviceSynchronize());

    std::vector<int> hm0(3 * E + 1), hm1(3 * E + 1);
    HC(hipMemcpy(hm0.data(), meta0, hm0.size() * sizeof(int), hipMemcpyDeviceToHost));
    HC(hipMemcpy(hm1.data(), meta1, hm1.size() * sizeof(int), hipMemcpyDeviceToHost));
    const size_t live_pad = (size_t)hm0[3 * E] * MPF_BM;
    std::vector<unsigned> hrt0(live_pad), hrt1(live_pad), hrp0(live_pad), hrp1(live_pad);
    std::vector<float> hrg0(live_pad), hrg1(live_pad);
    HC(hipMemcpy(hrt0.data(), rt0, live_pad * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hrt1.data(), rt1, live_pad * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hrp0.data(), rp0, live_pad * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hrp1.data(), rp1, live_pad * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hrg0.data(), rg0, live_pad * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hrg1.data(), rg1, live_pad * 4, hipMemcpyDeviceToHost));
    std::vector<unsigned char> htable(nslot * 8);
    HC(hipMemcpy(htable.data(), table, htable.size(), hipMemcpyDeviceToHost));
    bool exact = hm0 == hm1;
    for (unsigned e = 0; e < E; e++) {
        const unsigned begin = (unsigned)hm1[e];
        const unsigned count = (unsigned)hm1[E + e];
        const unsigned end = begin + ((count + MPF_BM - 1u) / MPF_BM) * MPF_BM;
        unsigned prev = 0;
        for (unsigned i = 0; i < count; i++) {
            const unsigned r = begin + i, s = hrp1[r];
            unsigned eid;
            float gate;
            std::memcpy(&eid, htable.data() + (size_t)s * 8, 4);
            std::memcpy(&gate, htable.data() + (size_t)s * 8 + 4, 4);
            exact &= s < nslot && eid == e && hrt1[r] == s / K && hrg1[r] == gate;
            exact &= i == 0 || s > prev;
            prev = s;
        }
        for (unsigned r = begin + count; r < end; r++)
            exact &= hrt1[r] == PLOW_EXPERT_UNUSED && hrp1[r] == PLOW_EXPERT_UNUSED && hrg1[r] == 0;
    }
    const bool baseline_rows_equal = hrt0 == hrt1 && hrp0 == hrp1 && hrg0 == hrg1;

    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    std::vector<float> router_prod, router_token, serial, parallel;
    for (unsigned rep = 0; rep < 31; rep++) {
        float ms;
        HC(hipEventRecord(a));
        hipLaunchKernelGGL(router_k, dim3(304), dim3(PLOW_THREADS), 0, 0, table_prod, dlogit, E, K,
                           T);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&ms, a, b));
        if (rep >= 5) router_prod.push_back(ms);

        HC(hipEventRecord(a));
        hipLaunchKernelGGL(router_k, dim3(router_blocks), dim3(PLOW_THREADS), 0, 0, table, dlogit, E,
                           K, T);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&ms, a, b));
        if (rep >= 5) router_token.push_back(ms);

        HC(hipEventRecord(a));
        hipLaunchKernelGGL(align_k, dim3(1), dim3(PLOW_THREADS), 0, 0, meta0, table, rt0, rp0, rg0,
                           T, E, K, 0u);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&ms, a, b));
        if (rep >= 5) serial.push_back(ms);

        HC(hipEventRecord(a));
        hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1,
                           T, E, K, 1u);
        hipLaunchKernelGGL(align_k, dim3(1), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1,
                           T, E, K, 2u);
        hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1,
                           T, E, K, 3u);
        hipLaunchKernelGGL(align_k, dim3(NB), dim3(PLOW_THREADS), 0, 0, meta1, table, rt1, rp1, rg1,
                           T, E, K, 4u);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&ms, a, b));
        if (rep >= 5) parallel.push_back(ms);
    }
    const float rp = median(router_prod), rt = median(router_token), s = median(serial),
                p = median(parallel);
    std::printf("T=%u router_prod304=%.6fms router_token=%.6fms align_serial=%.6fms align_parallel=%.6fms combined=%.6fms exact=%s router_exact=%s baseline_rows_equal=%s\n",
                T, rp, rt, s, p, rt + p, exact ? "yes" : "NO", router_exact ? "yes" : "NO",
                baseline_rows_equal ? "yes" : "no(nondeterministic-contract)");
    if (!exact || !router_exact) std::exit(2);
}

int main() {
    run(1024);
    run(8192);
    return 0;
}
