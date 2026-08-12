#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CK(call)                                                                                  \
    do {                                                                                          \
        const hipError_t error_ = (call);                                                         \
        if (error_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,                \
                         hipGetErrorString(error_));                                               \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace {
constexpr unsigned kThreads = 512;
constexpr unsigned kTopK = 16;
constexpr unsigned kExperts = 896;
constexpr unsigned kHidden = 3584;
constexpr unsigned kIntermediate = 384;
constexpr unsigned kRotations = 56;
constexpr unsigned kSamples = 12;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t count = values.size();
    return count & 1 ? values[count / 2]
                     : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

struct Device {
    unsigned char* arena{};
    uint16_t* x{};
    uint16_t* fu{};
    float* partial{};
    unsigned char* tables{};
    unsigned long long* weights{};
    unsigned long long* scales{};
};

void launch_glu(hipFunction_t kernel, Device& d, unsigned grid, unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, intermediate = kIntermediate, hidden = kHidden, experts = kExperts;
    void* args[] = {&d.fu, &d.x, &table, &d.weights, &d.scales, &topk,
                    &intermediate, &hidden, &experts};
    CK(hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_down(hipFunction_t kernel, Device& d, unsigned grid, unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, hidden = kHidden, intermediate = kIntermediate, experts = kExperts;
    void* args[] = {&d.partial, &d.fu, &table, &d.weights, &d.scales, &topk,
                    &hidden, &intermediate, &experts};
    CK(hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

double time_us(const std::function<void(unsigned)>& launch) {
    for (unsigned i = 0; i < kRotations; ++i) launch(i);
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    std::vector<double> samples;
    samples.reserve(kSamples);
    for (unsigned sample = 0; sample < kSamples; ++sample) {
        CK(hipEventRecord(begin, nullptr));
        for (unsigned i = 0; i < kRotations; ++i) launch(i);
        CK(hipEventRecord(end, nullptr));
        CK(hipEventSynchronize(end));
        float elapsed_ms = 0.0f;
        CK(hipEventElapsedTime(&elapsed_ms, begin, end));
        samples.push_back(elapsed_ms * 1000.0 / kRotations);
    }
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    return median(std::move(samples));
}

std::vector<unsigned char> copy_bytes(const void* source, size_t bytes) {
    std::vector<unsigned char> host(bytes);
    CK(hipMemcpy(host.data(), source, bytes, hipMemcpyDeviceToHost));
    return host;
}
}  // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_moe_grid_sweep_gfx942.co";
    constexpr size_t weight_bytes = static_cast<size_t>(kIntermediate) * kHidden / 2;
    constexpr size_t scale_bytes = static_cast<size_t>(kIntermediate) * kHidden / 32;
    constexpr size_t expert_bytes = 3 * (weight_bytes + scale_bytes);
    constexpr size_t arena_bytes = expert_bytes * kExperts;
    constexpr size_t fu_bytes = static_cast<size_t>(kTopK) * kIntermediate * sizeof(uint16_t);
    constexpr size_t partial_bytes = static_cast<size_t>(kTopK) * kHidden * sizeof(float);
    constexpr double glu_bytes =
        static_cast<double>(kTopK) * 2.0 * (weight_bytes + scale_bytes);
    constexpr double down_bytes =
        static_cast<double>(kTopK) * (weight_bytes + scale_bytes);

    CK(hipInit(0));
    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t glu, down, stream;
    CK(hipModuleGetFunction(&glu, module, "k3_moe_group_glu"));
    CK(hipModuleGetFunction(&down, module, "k3_moe_group_down"));
    CK(hipModuleGetFunction(&stream, module, "k3_moe_stream"));

    Device d;
    CK(hipMalloc(&d.arena, arena_bytes));
    CK(hipMemset(d.arena, 0x42, arena_bytes));
    CK(hipMalloc(&d.x, static_cast<size_t>(kHidden) * sizeof(uint16_t)));
    CK(hipMemset(d.x, 0x3c, static_cast<size_t>(kHidden) * sizeof(uint16_t)));
    CK(hipMalloc(&d.fu, fu_bytes));
    CK(hipMalloc(&d.partial, partial_bytes));

    std::vector<unsigned long long> weights(kExperts * 3), scales(kExperts * 3);
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        unsigned char* base = d.arena + expert_bytes * expert;
        for (unsigned matrix = 0; matrix < 3; ++matrix) {
            weights[expert * 3 + matrix] = reinterpret_cast<unsigned long long>(
                base + static_cast<size_t>(matrix) * weight_bytes);
            scales[expert * 3 + matrix] = reinterpret_cast<unsigned long long>(
                base + 3 * weight_bytes + static_cast<size_t>(matrix) * scale_bytes);
        }
    }
    CK(hipMalloc(&d.weights, weights.size() * sizeof(weights[0])));
    CK(hipMalloc(&d.scales, scales.size() * sizeof(scales[0])));
    CK(hipMemcpy(d.weights, weights.data(), weights.size() * sizeof(weights[0]),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d.scales, scales.data(), scales.size() * sizeof(scales[0]),
                 hipMemcpyHostToDevice));

    std::vector<unsigned char> tables(static_cast<size_t>(kRotations) * kTopK * 8);
    for (unsigned rotation = 0; rotation < kRotations; ++rotation) {
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            unsigned char* entry = &tables[(static_cast<size_t>(rotation) * kTopK + slot) * 8];
            const unsigned expert = rotation * kTopK + slot;
            const float gate = 0.05f;
            std::memcpy(entry, &expert, sizeof(expert));
            std::memcpy(entry + 4, &gate, sizeof(gate));
        }
    }
    CK(hipMalloc(&d.tables, tables.size()));
    CK(hipMemcpy(d.tables, tables.data(), tables.size(), hipMemcpyHostToDevice));

    float* sink = d.partial;
    unsigned char* arena = d.arena;
    unsigned long long stream_bytes = 512ull << 20;
    void* stream_args[] = {&sink, &arena, &stream_bytes};
    const double stream_us = time_us([&](unsigned) {
        CK(hipModuleLaunchKernel(stream, 304, 1, 1, kThreads, 1, 1, 0, nullptr,
                                 stream_args, nullptr));
    });
    std::fprintf(stderr, "reference_stream_gbps=%.3f arena_gib=%.3f\n",
                 static_cast<double>(stream_bytes) / (stream_us * 1e-6) / 1e9,
                 static_cast<double>(arena_bytes) / (1ull << 30));

    CK(hipMemset(d.fu, 0, fu_bytes));
    CK(hipMemset(d.partial, 0, partial_bytes));
    launch_glu(glu, d, 304, 0);
    launch_down(down, d, 304, 0);
    CK(hipDeviceSynchronize());
    const auto reference_fu = copy_bytes(d.fu, fu_bytes);
    const auto reference_partial = copy_bytes(d.partial, partial_bytes);

    std::puts("grid,glu_us,down_us,chain_us,glu_gbps,down_gbps,chain_ms_x92,fu_diff,partial_diff");
    for (unsigned grid : {1u, 12u, 32u, 64u, 96u, 128u, 152u, 192u, 256u, 304u}) {
        const double glu_us = time_us([&](unsigned rotation) {
            launch_glu(glu, d, grid, rotation);
        });
        const double down_us = time_us([&](unsigned rotation) {
            launch_down(down, d, grid, rotation);
        });
        const double chain_us = time_us([&](unsigned rotation) {
            launch_glu(glu, d, grid, rotation);
            launch_down(down, d, grid, rotation);
        });

        CK(hipMemset(d.fu, 0, fu_bytes));
        CK(hipMemset(d.partial, 0, partial_bytes));
        launch_glu(glu, d, grid, 0);
        launch_down(down, d, grid, 0);
        CK(hipDeviceSynchronize());
        const auto got_fu = copy_bytes(d.fu, fu_bytes);
        const auto got_partial = copy_bytes(d.partial, partial_bytes);
        size_t fu_diff = 0;
        for (size_t i = 0; i < got_fu.size(); ++i) fu_diff += got_fu[i] != reference_fu[i];
        size_t partial_diff = 0;
        for (size_t i = 0; i < got_partial.size(); ++i) {
            partial_diff += got_partial[i] != reference_partial[i];
        }

        std::printf("%u,%.6f,%.6f,%.6f,%.3f,%.3f,%.6f,%zu,%zu\n", grid,
                    glu_us, down_us, chain_us, glu_bytes / (glu_us * 1e-6) / 1e9,
                    down_bytes / (down_us * 1e-6) / 1e9,
                    chain_us * 92.0 / 1000.0, fu_diff, partial_diff);
    }

    CK(hipModuleUnload(module));
    return 0;
}
