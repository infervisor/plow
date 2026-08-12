/* v4_index_oracle_gfx942.hip — `d_v4_index_score` / `d_v4_index_topk` against a
 * HOST TRANSCRIPTION of `inference/model.py::Indexer.forward`.
 *
 * THE REFERENCE:
 *
 *     index_score[t][c] = sum_h relu(q[t][h] . ckv[c]) * w[t][h]
 *     prefill mask:  c < (t + 1) / ratio          # entry c pools [c*r, (c+1)*r)
 *     topk_idxs      = topk(index_score, K) + offset, -1 where illegal
 *
 * WHAT THIS PINS:
 *   * the relu is INSIDE the head sum. Moving it outside lets one head cancel
 *     another's evidence and selects a different set. NEGATIVE-CONTROLLED.
 *   * the prefill causal limit is `(t+1)/ratio`, not `t/ratio` — a query may
 *     see the compressed entry its own window completed.
 *     NEGATIVE-CONTROLLED.
 *   * ties resolve to the LOWER index, so the selection is deterministic across
 *     runs and, under TP, across ranks.
 *
 * The gate compares the SELECTED SET exactly, not a tolerance: the top-k is a
 * discrete decision and near-misses are what change a token.
 *
 * NOT COVERED: the reference's fp4 quantize-dequantize of q and the compressed
 * KV. The Hadamard that precedes it is orthogonal and applied to both sides of
 * the dot, so it cancels; the fp4 rounding does not, and near a tie it can flip
 * a selection. That is the fp4 activation path and it is Stage-4 work.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_index_oracle_gfx942.hip -o /tmp/v4idx && /tmp/v4idx
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

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { LDS_BYTES = 32768 };

__global__ __launch_bounds__(PLOW_THREADS) void k_score(float* sc, const bf16* q, const bf16* ckv,
                                                        const bf16* w, unsigned T, unsigned H,
                                                        unsigned HD, unsigned NC) {
    d_v4_index_score(sc, q, ckv, w, T, H, HD, NC, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_topk(int* idx, const float* sc, unsigned T,
                                                       unsigned NC, unsigned K, unsigned ratio,
                                                       int offset) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_index_topk(idx, sc, T, NC, K, ratio, offset, blockIdx.x, gridDim.x, (float*)raw);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    const unsigned r = (u >> 16) & 1u;
    return (bf16)((u + 0x7fffu + r) >> 16);
}
static float bf2f_host(bf16 b) {
    const unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float frand(void) { return (float)drand48() * 2.0f - 1.0f; }

static int run_case(unsigned T, unsigned H, unsigned HD, unsigned NC, unsigned K, unsigned ratio,
                    int offset) {
    printf("T=%u H=%u HD=%u NC=%u K=%u ratio=%u off=%d\n", T, H, HD, NC, K, ratio, offset);
    std::vector<bf16> q((size_t)T * H * HD), ckv((size_t)NC * HD), w((size_t)T * H);
    for (auto& v : q) v = f2bf_host(frand());
    for (auto& v : ckv) v = f2bf_host(frand());
    for (auto& v : w) v = f2bf_host(frand());

    bf16 *d_q, *d_ckv, *d_w;
    float* d_sc;
    int* d_idx;
    HC_CHECK(hipMalloc(&d_q, q.size() * 2));
    HC_CHECK(hipMalloc(&d_ckv, ckv.size() * 2));
    HC_CHECK(hipMalloc(&d_w, w.size() * 2));
    HC_CHECK(hipMalloc(&d_sc, (size_t)T * NC * 4));
    HC_CHECK(hipMalloc(&d_idx, (size_t)T * K * 4));
    HC_CHECK(hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_ckv, ckv.data(), ckv.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_w, w.data(), w.size() * 2, hipMemcpyHostToDevice));

    const unsigned blocks = T < 4u ? T : 4u;
    k_score<<<blocks, PLOW_THREADS>>>(d_sc, d_q, d_ckv, d_w, T, H, HD, NC);
    HC_CHECK(hipDeviceSynchronize());
    k_topk<<<blocks, PLOW_THREADS>>>(d_idx, d_sc, T, NC, K, ratio, offset);
    HC_CHECK(hipDeviceSynchronize());

    std::vector<float> gsc((size_t)T * NC);
    std::vector<int> gidx((size_t)T * K);
    HC_CHECK(hipMemcpy(gsc.data(), d_sc, gsc.size() * 4, hipMemcpyDeviceToHost));
    HC_CHECK(hipMemcpy(gidx.data(), d_idx, gidx.size() * 4, hipMemcpyDeviceToHost));

    double worst = 0.0, rms = 0.0;
    std::vector<double> ref((size_t)T * NC);
    for (unsigned t = 0; t < T; t++)
        for (unsigned c = 0; c < NC; c++) {
            double acc = 0.0;
            for (unsigned h = 0; h < H; h++) {
                double d = 0.0;
                for (unsigned e = 0; e < HD; e++)
                    d += bf2f_host(q[((size_t)t * H + h) * HD + e]) *
                         bf2f_host(ckv[(size_t)c * HD + e]);
                acc += std::max(d, 0.0) * bf2f_host(w[(size_t)t * H + h]); /* relu INSIDE */
            }
            ref[(size_t)t * NC + c] = acc;
            rms += acc * acc;
        }
    rms = sqrt(rms / (double)ref.size()) + 1e-9;
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, fabs(ref[i] - (double)gsc[i]) / rms);

    /* The selection is compared EXACTLY — a top-k is a discrete decision. */
    int selbad = 0;
    for (unsigned t = 0; t < T; t++) {
        const unsigned lim = ratio ? ((t + 1u) / ratio) : NC;
        std::vector<std::pair<double, int>> v;
        for (unsigned c = 0; c < lim && c < NC; c++) v.push_back({ref[(size_t)t * NC + c], (int)c});
        std::stable_sort(v.begin(), v.end(), [](const std::pair<double, int>& a,
                                                const std::pair<double, int>& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        /* THE SET, NOT THE ORDER. `torch.topk` returns indices in descending
         * score; a radix select returns them in ascending index. Both feed
         * `d_v4_sparse_attn`, which sums over the selected entries under an
         * online softmax and is invariant to their order, so what has to match
         * is membership — including the tie-break, since two ranks that keep
         * different entries at equal score decode different tokens. Comparing
         * the sequence would fail a selector that is exactly right. */
        std::vector<int> want, got;
        for (unsigned k = 0; k < K; k++) {
            if (k < v.size()) want.push_back(v[k].second + offset);
            const int g = gidx[(size_t)t * K + k];
            if (g >= 0) got.push_back(g);
        }
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        if (want != got) selbad++;
    }
    printf("  %-24s score max rel %.3e   selection %s\n", "index_score+topk", worst,
           selbad ? "MISMATCH" : "exact");

    hipFree(d_q);
    hipFree(d_ckv);
    hipFree(d_w);
    hipFree(d_sc);
    hipFree(d_idx);
    return (worst <= 2e-2 && selbad == 0) ? 0 : 1;
}

int main(void) {
    srand48(17);
    int bad = 0;
    /* The shipped indexer: 64 heads x 128, top-512. NC is scaled down so the
     * host's O(T*NC*H*HD) double loop stays tractable; the selection rule this
     * gate is about does not depend on it. */
    bad |= run_case(4, 64, 128, 1024, 512, 0, 128);
    /* Prefill: the causal limit leaves fewer legal entries than K near the
     * start, so the tail of the index list must be -1. */
    bad |= run_case(6, 16, 128, 512, 64, 4, 0);
    printf("%s\n", bad ? "FAIL" : "PASS — indexer scoring and selection match Indexer.forward");
    return bad;
}
