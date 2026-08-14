/* v4_attn_scale_gfx942.hip — is V4 decode attention CU-starved, or slow?
 *
 * At T=1 the work index is (token, head), so the grid is H = 64 blocks on a
 * 304-CU part: 240 CUs idle. Three optimization attempts on this kernel have
 * now measured as losses or no-ops because I predicted the bottleneck instead
 * of establishing it, so this file establishes it with one cheap experiment
 * rather than a fourth hypothesis.
 *
 * THE TEST. Time the same kernel at T = 1, 2, 4, 8, 16. Work grows linearly
 * with T. If the op is CU-starved at T=1, the first few doublings are nearly
 * FREE per token (the added blocks land on idle CUs) and us/token falls; once
 * the grid passes ~304 blocks, us/token flattens and the kernel is genuinely
 * throughput-bound.
 *
 * That distinction decides the fix and they are opposite:
 *   falls   => split KEYS across blocks too (flash-decode split-K, with a
 *              cross-block combine) so a T=1 step fills the machine.
 *   flat    => the inner loop is the cost; an MFMA key tile is the answer and
 *              split-K would buy nothing.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_attn_scale_gfx942.hip -o /tmp/v4as && /tmp/v4as
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_deepseek_v4.h"

__global__ __launch_bounds__(PLOW_THREADS) void k_topk(int* idx, const float* sc, unsigned T,
                                                       unsigned NC, unsigned K, unsigned ratio,
                                                       int off) {
    __shared__ __align__(16) char raw[40960];
    d_v4_index_topk(idx, sc, T, NC, K, ratio, off, nullptr, 0, blockIdx.x, gridDim.x, (float*)raw);
}

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { ATTN_LDS = 20480, NCU = 304 };

__global__ __launch_bounds__(PLOW_THREADS) void k_attn(bf16* out, const bf16* q, const bf16* kv,
                                                       const int* idx, const float* sink,
                                                       unsigned T, unsigned H, unsigned D,
                                                       unsigned TOPK, float scale) {
    __shared__ __align__(16) char raw[ATTN_LDS];
    d_v4_sparse_attn(out, q, kv, idx, sink, T, H, D, TOPK, scale, blockIdx.x, gridDim.x,
                     (float*)raw);
}

template <typename F>
static double time_us(F f, int reps = 15) {
    for (int i = 0; i < 3; i++) f();
    HC_CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        f();
        hipEventRecord(b);
        HC_CHECK(hipEventSynchronize(b));
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0);
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(void) {
    srand48(5);
    const unsigned H = 64, D = 512, NKV = 8192, TOPK = 640;
    const unsigned TS[] = {1, 2, 4, 8, 16};
    printf("V4 sparse attention, H=%u D=%u topk=%u, on %d CUs\n", H, D, TOPK, NCU);
    printf("grid is T*H blocks; work is linear in T\n\n");
    printf("  %4s %7s %11s %13s %10s\n", "T", "blocks", "us total", "us/token", "vs T=1");

    std::vector<bf16> kvh((size_t)NKV * D);
    for (auto& v : kvh) v = (bf16)(lrand48() & 0xffff);
    bf16 *d_q, *d_kv, *d_o;
    int* d_i;
    float* d_s;
    HC_CHECK(hipMalloc(&d_kv, kvh.size() * 2));
    HC_CHECK(hipMemcpy(d_kv, kvh.data(), kvh.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMalloc(&d_s, H * 4));
    HC_CHECK(hipMemset(d_s, 0, H * 4));

    double base = 0.0;
    for (unsigned ti = 0; ti < sizeof(TS) / sizeof(TS[0]); ti++) {
        const unsigned T = TS[ti];
        HC_CHECK(hipMalloc(&d_q, (size_t)T * H * D * 2));
        HC_CHECK(hipMemset(d_q, 0x3c, (size_t)T * H * D * 2));
        HC_CHECK(hipMalloc(&d_o, (size_t)T * H * D * 2));
        std::vector<int> idx((size_t)T * TOPK);
        for (auto& v : idx) v = (int)(lrand48() % NKV);
        HC_CHECK(hipMalloc(&d_i, idx.size() * 4));
        HC_CHECK(hipMemcpy(d_i, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));

        const unsigned blocks = T * H;
        const double us = time_us(
            [&] { k_attn<<<blocks, PLOW_THREADS>>>(d_o, d_q, d_kv, d_i, d_s, T, H, D, TOPK, 0.0442f); });
        if (ti == 0) base = us;
        printf("  %4u %7u %11.2f %13.2f %9.2fx\n", T, blocks, us, us / T, (us / T) / base);

        hipFree(d_q);
        hipFree(d_o);
        hipFree(d_i);
    }
    /* THE SAME QUESTION FOR THE TOP-K, which runs ONE block per row and so one
     * block per layer at decode. If its total is flat as rows are added it is
     * starved too, and the fix is a multi-block histogram; if it scales from
     * the first row, one block already saturates something. */
    {
        const unsigned NC = 4096, KK = 512;
        printf("\n  top-k: one block per row\n");
        printf("  %4s %7s %11s %13s %10s\n", "rows", "blocks", "us total", "us/row", "vs 1");
        float* scb = nullptr;
        int* oi = nullptr;
        HC_CHECK(hipMalloc(&scb, (size_t)64 * NC * 4));
        std::vector<float> hh((size_t)64 * NC);
        for (auto& v : hh) v = (float)drand48();
        HC_CHECK(hipMemcpy(scb, hh.data(), hh.size() * 4, hipMemcpyHostToDevice));
        HC_CHECK(hipMalloc(&oi, (size_t)64 * KK * 4));
        double b0 = 0.0;
        const unsigned RS[] = {1u, 2u, 4u, 8u, 16u, 64u};
        for (unsigned ri = 0; ri < 6; ri++) {
            const unsigned rows = RS[ri];
            const double us =
                time_us([&] { k_topk<<<rows, PLOW_THREADS>>>(oi, scb, rows, NC, KK, 0, 0); });
            if (ri == 0) b0 = us;
            printf("  %4u %7u %11.2f %13.2f %9.2fx\n", rows, rows, us, us / rows,
                   (us / rows) / b0);
        }
        hipFree(scb);
        hipFree(oi);
    }

    printf("\n  If us/token falls sharply through T=4 (grid 256 < 304 CUs) the op is\n"
           "  CU-STARVED at decode and needs split-K across blocks. If it is flat,\n"
           "  the inner loop is the cost and only an MFMA tile helps.\n");
    hipFree(d_kv);
    hipFree(d_s);
    return 0;
}
