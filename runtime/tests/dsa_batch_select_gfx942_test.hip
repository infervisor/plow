#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

static void check(hipError_t error) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(error));
        std::exit(1);
    }
}

__global__ __launch_bounds__(512, 1)
void select_row(int* idx, const float* scores, unsigned ctx, unsigned topk,
                unsigned* hist, unsigned* ctl, const int* lengths, unsigned row) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned scratch[259];
    d_index_select_coop(idx, scores, ctx, topk, hist, ctl,
                        (blockIdx.x * 13u) % gridDim.x, gridDim.x,
                        scratch, scratch + 256, lengths, 1, row);
#endif
}

__global__ __launch_bounds__(512, 1)
void score_rows(float* scores, const uint16_t* q, const uint16_t* k, const uint16_t* w,
                 const int* lengths, unsigned rows, unsigned ctx) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 queries[32 * 136], keys[128 * 136];
    __shared__ float weights[32];
    static_assert(sizeof(queries) + sizeof(keys) + sizeof(weights) == 43648);
    d_index_score_mfma<128, 32, 128>(scores, q, k, w, lengths, rows, ctx, 1.f / 64.f,
                                    blockIdx.x, gridDim.x, queries, keys, weights);
#endif
}

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits >> 16;
}

static void score_oracle(const std::vector<int>& lengths, unsigned ctx, int* lens) {
    const unsigned rows = lengths.size();
    std::vector<uint16_t> q(rows * 32 * 128), k(size_t(rows) * ctx * 128), w(rows * 32);
    std::vector<float> expected(rows * 19), got(size_t(rows) * ctx, -12345.f);
    auto query = [](unsigned r, unsigned h, unsigned d) {
        return float(int((h * 7 + d * 13 + r * 5) % 17) - 8) / 32.f;
    };
    auto key = [](unsigned r, unsigned pos, unsigned d) {
        return float(int((pos * 3 + d * 11 + r) % 19) - 9) / 32.f;
    };
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned h = 0; h < 32; ++h) {
            float weight = float(int((h + r) % 7) - 3) / 16.f;
            w[r * 32 + h] = bf16_bits(weight);
            for (unsigned d = 0; d < 128; ++d)
                q[(r * 32 + h) * 128 + d] = bf16_bits(query(r, h, d));
            for (unsigned pos = 0; pos < 19; ++pos) {
                float dot = 0;
                for (unsigned d = 0; d < 128; ++d) dot += query(r, h, d) * key(r, pos, d);
                expected[r * 19 + pos] += weight * std::max(dot, 0.f) / 64.f;
            }
        }
        for (unsigned pos = 0; pos < ctx; ++pos)
            for (unsigned d = 0; d < 128; ++d)
                k[(size_t(r) * ctx + pos) * 128 + d] = bf16_bits(key(r, pos, d));
    }
    uint16_t *dq, *dk, *dw;
    float* ds;
    check(hipMalloc(&dq, q.size() * 2));
    check(hipMalloc(&dk, k.size() * 2));
    check(hipMalloc(&dw, w.size() * 2));
    check(hipMalloc(&ds, got.size() * 4));
    check(hipMemcpy(dq, q.data(), q.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(ds, got.data(), got.size() * 4, hipMemcpyHostToDevice));
    score_rows<<<304, 512>>>(ds, dq, dk, dw, lens, rows, ctx);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(got.data(), ds, got.size() * 4, hipMemcpyDeviceToHost));
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned pos = 0; pos < ctx; ++pos) {
            float want = pos < unsigned(lengths[r]) ? expected[r * 19 + pos % 19] : -12345.f;
            if (!std::isfinite(got[r * ctx + pos]) || std::abs(got[r * ctx + pos] - want) > 1e-7f) {
                std::fprintf(stderr, "FAIL score row=%u pos=%u got=%g want=%g\n", r, pos, got[r * ctx + pos], want);
                std::exit(1);
            }
        }
    std::puts("PASS: 43648-byte score tile, 8 independent query/key/weight rows, all live scores and inactive tails");
    check(hipFree(dq)); check(hipFree(dk)); check(hipFree(dw)); check(hipFree(ds));
}

int main() {
    constexpr unsigned ctx = 81920, topk = 2048;
    const std::vector<int> lengths{0, 1, 129, 2047, 2048, 8192, 65537, 70000};
    std::vector<float> scores(lengths.size() * ctx, 1e30f);
    std::vector<std::vector<int>> expected(lengths.size());
    for (unsigned row = 0; row < lengths.size(); ++row) {
        for (int i = 0; i < lengths[row]; ++i)
            scores[row * ctx + i] = float(int((i * 313u + row * 97u) % 4096u) - 2048) / 128.f;
        auto& ids = expected[row];
        ids.resize(lengths[row]);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            float sa = scores[row * ctx + a], sb = scores[row * ctx + b];
            return sa != sb ? sa > sb : a < b;
        });
        ids.resize(std::min(topk, unsigned(lengths[row])));
        std::sort(ids.begin(), ids.end());
    }
    int *idx, *lens;
    float* sc;
    unsigned *hist, *ctl;
    check(hipMalloc(&idx, lengths.size() * topk * sizeof(int)));
    check(hipMalloc(&lens, lengths.size() * sizeof(int)));
    check(hipMalloc(&sc, scores.size() * sizeof(float)));
    check(hipMalloc(&hist, 7 * 256 * sizeof(unsigned)));
    check(hipMalloc(&ctl, 3 * sizeof(unsigned)));
    check(hipMemcpy(lens, lengths.data(), lengths.size() * sizeof(int), hipMemcpyHostToDevice));
    check(hipMemcpy(sc, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice));
    check(hipMemset(hist, 0, 7 * 256 * sizeof(unsigned)));
    check(hipMemset(ctl, 0, 3 * sizeof(unsigned)));
    std::vector<int> result(lengths.size() * topk);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        check(hipMemset(idx, 0xa5, result.size() * sizeof(int)));
        for (unsigned row = 0; row < lengths.size(); ++row)
            select_row<<<32, 512>>>(idx, sc, ctx, topk, hist, ctl, lens, row);
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        check(hipMemcpy(result.data(), idx, result.size() * sizeof(int), hipMemcpyDeviceToHost));
        for (unsigned row = 0; row < lengths.size(); ++row) {
            auto begin = result.begin() + row * topk;
            std::vector<int> got(begin, begin + expected[row].size());
            std::sort(got.begin(), got.end());
            if (got != expected[row]) {
                std::fprintf(stderr, "FAIL row=%u length=%d repeat=%u\n", row, lengths[row], repeat);
                return 1;
            }
        }
    }
    std::puts("PASS: 8 distinct KV lengths, poisoned tails, tied scores, permuted slices, 3 scratch reuses");
    score_oracle(lengths, ctx, lens);
    for (void* ptr : {static_cast<void*>(idx), static_cast<void*>(lens), static_cast<void*>(sc),
                      static_cast<void*>(hist), static_cast<void*>(ctl)}) check(hipFree(ptr));
}
