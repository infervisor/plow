#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>

constexpr unsigned CTX = 81920, TOPK = 2048, HIST = 7 * 256, CTL = 64;
constexpr int POISON = -1515870811;

static void check(hipError_t e) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(e));
        std::exit(1);
    }
}

__global__ __launch_bounds__(512, 1)
void select_rows(int* idx, const float* scores, const int* lengths,
                 unsigned* hist, unsigned* ctl, unsigned rows, unsigned groups,
                 unsigned mode, unsigned selected_row, bool permute) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned scratch[259];
    const unsigned logical = permute ? (blockIdx.x * 13u) % gridDim.x : blockIdx.x;
    if (mode == 2) {
        const unsigned row = logical / groups;
        d_index_select_coop(idx, scores, CTX, TOPK, hist + row * HIST, ctl + row * CTL,
                            logical % groups, groups, scratch, scratch + 256, lengths, 1, row);
    } else if (mode == 1) {
        for (unsigned row = 0; row < rows; ++row) {
            d_index_select_coop(idx, scores, CTX, TOPK, hist, ctl,
                                logical, groups, scratch, scratch + 256, lengths, 1, row);
            // Match the interpreter's all-producers-complete dependency before scratch reuse.
            dsa_grid_sync_t<true>(ctl, groups);
        }
    } else {
        d_index_select_coop(idx, scores, CTX, TOPK, hist, ctl,
                            logical, groups, scratch, scratch + 256, lengths, 1, selected_row);
    }
#endif
}

struct Buffers {
    int *idx, *lengths;
    float* scores;
    unsigned *hist, *ctl;
    unsigned rows;
    Buffers(const std::vector<float>& s, const std::vector<int>& lens) : rows(lens.size()) {
        check(hipMalloc(&idx, (rows * TOPK + 128) * sizeof(int)));
        check(hipMalloc(&lengths, rows * sizeof(int)));
        check(hipMalloc(&scores, s.size() * sizeof(float)));
        check(hipMalloc(&hist, rows * HIST * sizeof(unsigned)));
        check(hipMalloc(&ctl, rows * CTL * sizeof(unsigned)));
        check(hipMemcpy(scores, s.data(), s.size() * sizeof(float), hipMemcpyHostToDevice));
        check(hipMemcpy(lengths, lens.data(), rows * sizeof(int), hipMemcpyHostToDevice));
        check(hipMemset(hist, 0, rows * HIST * sizeof(unsigned)));
        check(hipMemset(ctl, 0, rows * CTL * sizeof(unsigned)));
    }
    ~Buffers() {
        for (void* p : {static_cast<void*>(idx), static_cast<void*>(lengths),
                       static_cast<void*>(scores), static_cast<void*>(hist), static_cast<void*>(ctl)})
            check(hipFree(p));
    }
    void launch(unsigned mode, unsigned groups, bool permute = false) {
        if (mode == 0) {
            for (unsigned row = 0; row < rows; ++row)
                select_rows<<<groups, 512>>>(idx, scores, lengths, hist, ctl, rows, groups, mode, row, permute);
        } else {
            select_rows<<<mode == 2 ? groups * rows : groups, 512>>>(
                idx, scores, lengths, hist, ctl, rows, groups, mode, 0, permute);
        }
        check(hipGetLastError());
    }
};

static std::vector<std::vector<int>> oracle(const std::vector<float>& scores,
                                           const std::vector<int>& lengths) {
    std::vector<std::vector<int>> result(lengths.size());
    for (unsigned row = 0; row < lengths.size(); ++row) {
        auto& ids = result[row];
        ids.resize(lengths[row]);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            float x = scores[size_t(row) * CTX + a], y = scores[size_t(row) * CTX + b];
            return x != y ? x > y : a < b;
        });
        ids.resize(std::min<unsigned>(ids.size(), TOPK));
        std::sort(ids.begin(), ids.end());
    }
    return result;
}

static void verify(Buffers& b, const std::vector<std::vector<int>>& expected,
                   unsigned mode, unsigned groups) {
    std::vector<int> got(b.rows * TOPK + 128);
    for (unsigned repeat = 0; repeat < 12; ++repeat) {
        check(hipMemset(b.idx, 0xa5, got.size() * sizeof(int)));
        b.launch(mode, groups, repeat % 2 != 0);
        check(hipDeviceSynchronize());
        check(hipMemcpy(got.data(), b.idx, got.size() * sizeof(int), hipMemcpyDeviceToHost));
        for (unsigned row = 0; row < b.rows; ++row) {
            auto at = got.begin() + row * TOPK;
            std::vector<int> selected(at, at + expected[row].size());
            std::sort(selected.begin(), selected.end());
            if (selected != expected[row] ||
                !std::all_of(at + expected[row].size(), at + TOPK, [](int x) { return x == POISON; })) {
                std::fprintf(stderr, "selection mismatch mode=%u groups=%u row=%u repeat=%u\n", mode, groups, row, repeat);
                std::exit(1);
            }
        }
        if (!std::all_of(got.end() - 128, got.end(), [](int x) { return x == POISON; })) std::exit(2);
    }
}

static void timing(Buffers& b, unsigned mode, unsigned groups) {
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphExec_t executable;
    check(hipStreamCreate(&stream));
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (unsigned rep = 0; rep < 10; ++rep) {
        if (mode == 0) {
            for (unsigned row = 0; row < b.rows; ++row)
                select_rows<<<groups, 512, 0, stream>>>(b.idx, b.scores, b.lengths,
                    b.hist, b.ctl, b.rows, groups, mode, row, false);
        } else {
            select_rows<<<mode == 2 ? b.rows * groups : groups, 512, 0, stream>>>(
                b.idx, b.scores, b.lengths, b.hist, b.ctl, b.rows, groups, mode, 0, false);
        }
    }
    check(hipGetLastError());
    check(hipStreamEndCapture(stream, &graph));
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    hipEvent_t begin, end;
    check(hipEventCreate(&begin)); check(hipEventCreate(&end));
    for (unsigned i = 0; i < 2; ++i) check(hipGraphLaunch(executable, stream));
    check(hipStreamSynchronize(stream));
    std::printf("[");
    for (unsigned i = 0; i < 10; ++i) {
        check(hipEventRecord(begin, stream));
        check(hipGraphLaunch(executable, stream));
        check(hipEventRecord(end, stream));
        check(hipEventSynchronize(end));
        float ms;
        check(hipEventElapsedTime(&ms, begin, end));
        std::printf("%s%.6f", i ? "," : "", ms * 100.f);
    }
    std::printf("]");
    check(hipEventDestroy(begin)); check(hipEventDestroy(end));
    check(hipGraphExecDestroy(executable)); check(hipGraphDestroy(graph));
    check(hipStreamDestroy(stream));
}

static bool first = true;
static void run(unsigned rows, unsigned pattern, const char* capture = nullptr, const char* idx = nullptr) {
    const int edges[] = {0, 1, 129, 2047, 2048, 8192, 65537, 79800};
    std::vector<int> lengths(rows);
    std::vector<float> scores(size_t(rows) * CTX, 1e30f);
    for (unsigned r = 0; r < rows; ++r) {
        lengths[r] = capture ? 65003 : pattern == 0 ? edges[r % 8] : 70000 + r;
        for (int i = 0; i < lengths[r]; ++i) {
            uint32_t x = (uint32_t(i) + r * 12345u) * 747796405u + 2891336453u;
            x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
            scores[size_t(r) * CTX + i] = pattern == 2 ? 1.f :
                pattern == 0 ? float(int(x % 4096) - 2048) / 128.f : float(int32_t(x)) / 2147483648.f;
        }
    }
    if (capture) {
        std::ifstream f(capture, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(scores.data()), scores.size() * 4)) std::exit(3);
    }
    auto expected = oracle(scores, lengths);
    if (idx) {
        std::vector<int> original(rows * TOPK);
        std::ifstream f(idx, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(original.data()), original.size() * 4)) std::exit(3);
        for (unsigned r = 0; r < rows; ++r) {
            std::vector<int> got(original.begin() + r * TOPK, original.begin() + (r + 1) * TOPK);
            std::sort(got.begin(), got.end());
            if (got != expected[r]) { std::fprintf(stderr, "capture top-k mismatch row=%u\n", r); std::exit(4); }
        }
    }
    Buffers b(scores, lengths);
    for (unsigned mode = 0; mode < 3; ++mode) {
        for (unsigned groups : {4u, 8u, 16u, 32u}) {
            if ((mode != 2 && groups != 32) || (mode == 2 && groups * rows > 304)) continue;
            verify(b, expected, mode, groups);
            std::printf("%s{\"rows\":%u,\"pattern\":%u,\"capture\":%s,\"mode\":%u,\"groups_per_row\":%u,\"passed\":true,\"us\":",
                        first ? "" : ",\n", rows, pattern, capture ? "true" : "false", mode, groups);
            first = false;
            timing(b, mode, groups);
            std::printf("}");
            std::fflush(stdout);
        }
    }
}

int main(int argc, char** argv) {
    std::printf("{\"schema\":\"plow.dsa.parallel-selection.v1\",\"cases\":[\n");
    if (argc == 3) run(8, 1, argv[1], argv[2]);
    else if (argc == 1) {
        for (unsigned rows : {1u, 2u, 4u, 8u, 16u, 20u, 32u})
            for (unsigned pattern = 0; pattern < 3; ++pattern) run(rows, pattern);
    } else return 2;
    std::printf("\n]}\n");
}
