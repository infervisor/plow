// d_index_select_pf_thr (the TP adapter's plow_dsa_tp_select_thr) against d_index_select_pf<true>
// and a host top-k, row set for row set. Cases: continuous scores; heavy quantized ties that
// overflow the 2048-key boundary capacity; an all-equal row; negative, +-0 and -inf-masked scores;
// rows at and below 2048 keys (identity + -1 pad); a band subrange that must leave other rows alone.
// Build (gfx942, lab ROCm):
//   hipcc --offload-arch=gfx942 -O3 -w -std=c++17 -DPLOW_DSA_TP_SELECT_THR=1 -Iruntime/amd \
//       -Iruntime/common runtime/tests/dsa_select_thr_gfx942_test.hip -o dsa_select_thr_test
// Run on one GPU: ./dsa_select_thr_test   (exit 0 = every row matches)
#define PLOW_WG_WAVES 8
#define PLOW_DSA_TP_SELECT_THR 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void check(hipError_t error) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(error));
        std::exit(1);
    }
}

__global__ __launch_bounds__(512, 1)
void select_ref(int* idx, const float* score, const int* len, unsigned rows, unsigned stride,
                unsigned lo, unsigned hi) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[SEL_NB], red[4];
    d_index_select_pf<true>(idx, score, len, rows, 2048, stride, blockIdx.x, gridDim.x, hist, red, 1,
                            lo, hi);
#endif
}

__global__ __launch_bounds__(512, 1)
void select_thr(int* idx, const float* score, const int* len, unsigned rows, unsigned stride,
                unsigned lo, unsigned hi) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hf[4096], hc[256], red[8];
    __shared__ unsigned long long cand[2048];
    d_index_select_pf_thr<2048u>(idx, score, len, rows, 2048, stride, blockIdx.x, gridDim.x, hf, hc,
                                 cand, red, lo, hi);
#endif
}

static uint64_t host_key(float sc, unsigned t, unsigned len) {
    uint32_t sb;
    std::memcpy(&sb, &sc, 4);
    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
    return (uint64_t(sb) << 24) | ((len - 1 - t) & 0xFFFFFFu);
}

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint32_t rnd() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return uint32_t(rng >> 11);
}

enum Kind { CONTINUOUS, QUANTIZED, EQUAL, SIGNED_MASKED };

struct Case {
    const char* name;
    unsigned rows, len, lo, hi;
    Kind kind;
};

static float value(Kind kind, unsigned row, unsigned pos) {
    switch (kind) {
    case CONTINUOUS: return float(rnd() % 1000003) / 1000.f - 300.f;
    case QUANTIZED: return float(rnd() % 7) * 0.25f;
    case EQUAL: return 1.5f;
    case SIGNED_MASKED:
        if ((pos + row) % 3 == 0) return -std::numeric_limits<float>::infinity();
        if (pos % 11 == 0) return (pos & 1) ? -0.0f : 0.0f;
        return -float(rnd() % 20000) / 64.f;
    }
    return 0.f;
}

int main() {
    const Case cases[] = {
        {"continuous", 40, 70000, 0, 40, CONTINUOUS},
        {"quantized ties", 40, 70000, 0, 40, QUANTIZED},
        {"all equal", 24, 30000, 0, 24, EQUAL},
        {"signed -inf masked", 40, 70000, 0, 40, SIGNED_MASKED},
        {"short rows (identity at <= 2048 keys)", 4096, 4096, 2040, 2060, CONTINUOUS},
        {"short quantized", 4096, 4096, 2044, 2056, QUANTIZED},
        {"band subrange", 64, 66000, 20, 36, QUANTIZED},
    };
    unsigned failures = 0;
    for (const Case& c : cases) {
        const unsigned stride = c.len, rows = c.rows;
        std::vector<float> score(size_t(rows) * stride, 0.f);
        const unsigned q_pos0 = c.len - rows;
        for (unsigned t = c.lo; t < c.hi; ++t)
            for (unsigned p = 0; p < q_pos0 + t + 1; ++p) score[size_t(t) * stride + p] = value(c.kind, t, p);
        int len = int(c.len);
        float* d_score;
        int* d_len;
        int* d_ref;
        int* d_thr;
        const size_t idx_bytes = size_t(rows) * 2048 * 4;
        check(hipMalloc(&d_score, score.size() * 4));
        check(hipMalloc(&d_len, 4));
        check(hipMalloc(&d_ref, idx_bytes));
        check(hipMalloc(&d_thr, idx_bytes));
        check(hipMemcpy(d_score, score.data(), score.size() * 4, hipMemcpyHostToDevice));
        check(hipMemcpy(d_len, &len, 4, hipMemcpyHostToDevice));
        std::vector<int> sentinel(size_t(rows) * 2048, -7);
        check(hipMemcpy(d_ref, sentinel.data(), idx_bytes, hipMemcpyHostToDevice));
        check(hipMemcpy(d_thr, sentinel.data(), idx_bytes, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(select_ref, dim3(304), dim3(512), 0, nullptr, d_ref, d_score, d_len, rows, stride,
                           c.lo, c.hi);
        hipLaunchKernelGGL(select_thr, dim3(304), dim3(512), 0, nullptr, d_thr, d_score, d_len, rows, stride,
                           c.lo, c.hi);
        check(hipDeviceSynchronize());
        std::vector<int> ref(sentinel.size()), thr(sentinel.size());
        check(hipMemcpy(ref.data(), d_ref, idx_bytes, hipMemcpyDeviceToHost));
        check(hipMemcpy(thr.data(), d_thr, idx_bytes, hipMemcpyDeviceToHost));
        unsigned bad = 0, bad_ref = 0, touched = 0;
        for (unsigned t = 0; t < rows; ++t) {
            const auto* a = &thr[size_t(t) * 2048];
            if (t < c.lo || t >= c.hi) {
                touched += !std::all_of(a, a + 2048, [](int v) { return v == -7; });
                continue;
            }
            const unsigned row_len = q_pos0 + t + 1;
            std::vector<int> want;
            if (row_len <= 2048) {
                for (unsigned s = 0; s < 2048; ++s) want.push_back(s < row_len ? int(s) : -1);
            } else {
                std::vector<unsigned> order(row_len);
                for (unsigned p = 0; p < row_len; ++p) order[p] = p;
                const auto* sr = &score[size_t(t) * stride];
                std::partial_sort(order.begin(), order.begin() + 2048, order.end(), [&](unsigned x, unsigned y) {
                    return host_key(sr[x], x, row_len) > host_key(sr[y], y, row_len);
                });
                want.assign(order.begin(), order.begin() + 2048);
            }
            std::sort(want.begin(), want.end());
            std::vector<int> got(a, a + 2048), got_ref(&ref[size_t(t) * 2048], &ref[size_t(t) * 2048] + 2048);
            std::sort(got.begin(), got.end());
            std::sort(got_ref.begin(), got_ref.end());
            bad += got != want;
            bad_ref += got_ref != want;
        }
        std::printf("%-40s rows [%u,%u) len %u: thr mismatches %u, reference mismatches %u, rows outside band touched %u\n",
                    c.name, c.lo, c.hi, c.len, bad, bad_ref, touched);
        failures += bad + bad_ref + touched;
        check(hipFree(d_score));
        check(hipFree(d_len));
        check(hipFree(d_ref));
        check(hipFree(d_thr));
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
