#ifndef PLOW_WG_WAVES
#define PLOW_WG_WAVES 8
#endif
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1)
void union_kernel(unsigned char* uni, unsigned long long* mask, const int* idx,
                  const int* len, unsigned rows, unsigned topk, unsigned stride,
                  unsigned cap, unsigned pack) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned scratch[PLOW_WG_WAVES * 64 + 1];
    d_index_union_pf(uni, mask, idx, len, rows, topk, stride, cap, pack,
                     blockIdx.x, gridDim.x, scratch);
#endif
}

static void run(unsigned rows, unsigned len, unsigned topk, unsigned pack) {
    const unsigned tiles = (rows + pack - 1) / pack;
    const unsigned blocks = std::min(tiles, 304u);
    const unsigned cap = std::min(pack * topk, len);
    const unsigned header = (tiles * 4 + 255) / 256 * 256;
    const size_t bytes = header + (size_t)tiles * cap * 12;
    std::vector<int> indices((size_t)rows * topk, -1);
    for (unsigned q = 0; q < rows; ++q) {
        unsigned live = len - rows + q + 1;
        for (unsigned j = 0; j < std::min(live, topk); ++j)
            indices[(size_t)q * topk + j] = (q * 313u + j * 37u) % live;
    }
    unsigned char* uni;
    unsigned long long* mask;
    int *idx, *length;
    check(hipMalloc(&uni, bytes));
    check(hipMalloc(&mask, (size_t)blocks * len * 8));
    check(hipMalloc(&idx, indices.size() * sizeof(int)));
    check(hipMalloc(&length, sizeof(int)));
    check(hipMemset(uni, 0xa5, bytes));
    check(hipMemset(mask, 0xa5, (size_t)blocks * len * 8));
    check(hipMemcpy(idx, indices.data(), indices.size() * sizeof(int), hipMemcpyHostToDevice));
    check(hipMemcpy(length, &len, sizeof(int), hipMemcpyHostToDevice));
    auto launch = [&] {
        union_kernel<<<blocks, PLOW_WG_WAVES * 64>>>(uni, mask, idx, length, rows, topk, len, cap, pack);
        check(hipGetLastError());
    };
    launch();
    check(hipDeviceSynchronize());
    std::vector<uint32_t> result(bytes / 4);
    check(hipMemcpy(result.data(), uni, bytes, hipMemcpyDeviceToHost));
    std::vector<uint64_t> expected(len);
    for (unsigned tile = 0; tile < tiles; ++tile) {
        std::fill(expected.begin(), expected.end(), 0);
        for (unsigned p = 0; p < pack && tile * pack + p < rows; ++p)
            for (unsigned j = 0; j < topk; ++j) {
                int k = indices[(size_t)(tile * pack + p) * topk + j];
                if (k >= 0) expected[k] |= uint64_t(1) << p;
            }
        const size_t base = header / 4 + (size_t)tile * cap * 3;
        unsigned count = 0;
        for (unsigned k = 0; k < len; ++k) {
            if (!expected[k]) continue;
            if (count >= cap) {
                std::fprintf(stderr, "reference union exceeds capacity\n");
                std::exit(1);
            }
            uint64_t got = result[base + cap + count] |
                           (uint64_t(result[base + 2 * cap + count]) << 32);
            if (result[base + count] != k || got != expected[k]) {
                std::fprintf(stderr, "union mismatch tile=%u entry=%u key=%u\n", tile, count, k);
                std::exit(1);
            }
            ++count;
        }
        if (result[tile] != count) {
            std::fprintf(stderr, "count mismatch tile=%u got=%u expected=%u\n",
                         tile, result[tile], count);
            std::exit(1);
        }
    }
    hipEvent_t begin, end;
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
    std::vector<float> samples;
    for (int repeat = 0; repeat < 5; ++repeat) {
        check(hipEventRecord(begin));
        launch();
        check(hipEventRecord(end));
        check(hipEventSynchronize(end));
        float ms;
        check(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    std::printf("rows=%u len=%u topk=%u pack=%u exact=PASS median_ms=%.4f\n",
                rows, len, topk, pack, samples[2]);
    check(hipEventDestroy(begin));
    check(hipEventDestroy(end));
    check(hipFree(uni));
    check(hipFree(mask));
    check(hipFree(idx));
    check(hipFree(length));
}

int main() {
    run(3, 3, 16, 8);
    run(129, 4096, 128, 64);
    run(8192, 81920, 2048, 8);
}
