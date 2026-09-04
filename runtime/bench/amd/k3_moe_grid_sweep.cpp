#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
unsigned kThreads = 512;
unsigned kTreeLeaves = 4;
unsigned kTreeThreads = 512;
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
    uint16_t* out{};
    float* partial{};
    unsigned char* tables{};
    unsigned long long* weights{};
    unsigned long long* scales{};
    unsigned* control{};
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

void launch_pair(hipFunction_t kernel, Device& d, unsigned grid, unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, intermediate = kIntermediate, hidden = kHidden, experts = kExperts;
    void* args[] = {&d.partial, &d.fu, &d.x, &table, &d.weights, &d.scales, &topk,
                    &intermediate, &hidden, &experts, &d.control};
    CK(hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_handoff(hipFunction_t kernel, Device& d, unsigned grid) {
    void* args[] = {&d.control};
    CK(hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_combine(hipFunction_t kernel, Device& d) {
    unsigned topk = kTopK, hidden = kHidden;
    void* args[] = {&d.out, &d.partial, &topk, &hidden};
    CK(hipModuleLaunchKernel(kernel, 7, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_down_combine(hipFunction_t kernel, Device& d, unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, hidden = kHidden, intermediate = kIntermediate, experts = kExperts;
    void* args[] = {&d.out, &d.partial, &d.fu, &table, &d.weights, &d.scales,
                    &topk, &hidden, &intermediate, &experts, &d.control};
    CK(hipModuleLaunchKernel(kernel, 256, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_down_combine_owned(hipFunction_t kernel, Device& d, unsigned grid,
                               unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, hidden = kHidden, intermediate = kIntermediate, experts = kExperts;
    void* args[] = {&d.out, &d.partial, &d.fu, &table, &d.weights, &d.scales,
                    &topk, &hidden, &intermediate, &experts, &d.control};
    CK(hipModuleLaunchKernel(kernel, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

void launch_down_combine_tree(hipFunction_t kernel, Device& d, unsigned rotation) {
    unsigned char* table = d.tables + static_cast<size_t>(rotation % kRotations) * kTopK * 8;
    unsigned topk = kTopK, hidden = kHidden, intermediate = kIntermediate, experts = kExperts;
    void* args[] = {&d.out, &d.partial, &d.fu, &table, &d.weights, &d.scales,
                    &topk, &hidden, &intermediate, &experts, &d.control};
    const unsigned tiles = (kHidden + (kTreeThreads / 64) * 8 - 1) /
                           ((kTreeThreads / 64) * 8);
    CK(hipModuleLaunchKernel(kernel, tiles * kTreeLeaves, 1, 1, kTreeThreads, 1, 1,
                             0, nullptr, args, nullptr));
}

float bf16_to_float(uint16_t bits) {
    uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

struct Error {
    size_t different{};
    double rel_l2{};
    double max_abs{};
};

uint16_t float_to_bf16(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7fffu + ((word >> 16) & 1u);
    return static_cast<uint16_t>(word >> 16);
}

struct TreeOracle {
    Error f32;
    size_t bf16_different{};
    std::vector<unsigned char> bf16;
};

TreeOracle tree_oracle(const std::vector<unsigned char>& raw_partial) {
    const float* partial = reinterpret_cast<const float*>(raw_partial.data());
    TreeOracle oracle;
    oracle.bf16.resize(static_cast<size_t>(kHidden) * sizeof(uint16_t));
    double delta2 = 0.0, reference2 = 0.0;
    const unsigned slots_per_leaf = kTopK / kTreeLeaves;
    for (unsigned h = 0; h < kHidden; ++h) {
        float fixed = 0.0f;
        for (unsigned slot = 0; slot < kTopK; ++slot)
            fixed += partial[static_cast<size_t>(slot) * kHidden + h];
        float leaves[16]{};
        for (unsigned leaf = 0; leaf < kTreeLeaves; ++leaf) {
            float values[16]{};
            for (unsigned i = 0; i < slots_per_leaf; ++i)
                values[i] = partial[static_cast<size_t>(leaf * slots_per_leaf + i) *
                                    kHidden + h];
            for (unsigned stride = 1; stride < slots_per_leaf; stride <<= 1)
                for (unsigned i = 0; i < slots_per_leaf; i += 2 * stride)
                    values[i] += values[i + stride];
            leaves[leaf] = values[0];
        }
        for (unsigned stride = 1; stride < kTreeLeaves; stride <<= 1)
            for (unsigned leaf = 0; leaf < kTreeLeaves; leaf += 2 * stride)
                leaves[leaf] += leaves[leaf + stride];
        const double delta = static_cast<double>(leaves[0]) - fixed;
        delta2 += delta * delta;
        reference2 += static_cast<double>(fixed) * fixed;
        oracle.f32.max_abs = std::max(oracle.f32.max_abs, std::abs(delta));
        oracle.f32.different += leaves[0] != fixed;
        const uint16_t tree_bits = float_to_bf16(leaves[0]);
        const uint16_t fixed_bits = float_to_bf16(fixed);
        oracle.bf16_different += tree_bits != fixed_bits;
        std::memcpy(oracle.bf16.data() + static_cast<size_t>(h) * sizeof(tree_bits),
                    &tree_bits, sizeof(tree_bits));
    }
    oracle.f32.rel_l2 = std::sqrt(delta2 / std::max(reference2, 1e-30));
    return oracle;
}

Error compare_bf16(const std::vector<unsigned char>& candidate,
                   const std::vector<unsigned char>& reference) {
    Error error;
    double delta2 = 0.0, reference2 = 0.0;
    for (size_t i = 0; i < candidate.size(); i += sizeof(uint16_t)) {
        uint16_t a_bits, b_bits;
        std::memcpy(&a_bits, candidate.data() + i, sizeof(a_bits));
        std::memcpy(&b_bits, reference.data() + i, sizeof(b_bits));
        error.different += a_bits != b_bits;
        const double a = bf16_to_float(a_bits), b = bf16_to_float(b_bits);
        const double delta = a - b;
        delta2 += delta * delta;
        reference2 += b * b;
        error.max_abs = std::max(error.max_abs, std::abs(delta));
    }
    error.rel_l2 = std::sqrt(delta2 / std::max(reference2, 1e-30));
    return error;
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
    if (argc > 2) kThreads = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    if (argc > 3) kTreeLeaves = static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10));
    if (argc > 4) kTreeThreads = static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10));
    if (kTreeLeaves != 2 && kTreeLeaves != 4 && kTreeLeaves != 8 &&
        kTreeLeaves != 16) {
        std::fprintf(stderr, "REFUSE: tree leaves must be 2, 4, 8, or 16\n");
        return 2;
    }
    if (kTreeThreads != 256 && kTreeThreads != 512) {
        std::fprintf(stderr, "REFUSE: tree threads must be 256 or 512\n");
        return 2;
    }
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
    hipDeviceProp_t props{};
    CK(hipGetDeviceProperties(&props, 0));
    if (props.multiProcessorCount != 256) {
        std::fprintf(stderr, "REFUSE: phase grid needs 256 CUs, got %d\n",
                     props.multiProcessorCount);
        return 2;
    }
    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t glu, down, combine, pair, down_combine, down_combine_owned,
                  down_combine_ready, down_combine_tree, xcd_map, handoff, stream;
    CK(hipModuleGetFunction(&glu, module, "k3_moe_group_glu"));
    CK(hipModuleGetFunction(&down, module, "k3_moe_group_down"));
    CK(hipModuleGetFunction(&combine, module, "k3_moe_combine"));
    CK(hipModuleGetFunction(&pair, module, "k3_moe_group_pair_xcd"));
    CK(hipModuleGetFunction(&down_combine, module, "k3_moe_down_combine_xcd"));
    CK(hipModuleGetFunction(&down_combine_owned, module,
                            "k3_moe_down_combine_xcd_owned"));
    CK(hipModuleGetFunction(&down_combine_ready, module,
                            "k3_moe_down_combine_ready"));
    CK(hipModuleGetFunction(&down_combine_tree, module,
                            "k3_moe_down_combine_tree"));
    CK(hipModuleGetFunction(&xcd_map, module, "k3_moe_xcd_map"));
    CK(hipModuleGetFunction(&handoff, module, "k3_moe_xcd_handoff"));
    CK(hipModuleGetFunction(&stream, module, "k3_moe_stream"));

    const unsigned tree_tiles =
        (kHidden + (kTreeThreads / 64) * 8 - 1) / ((kTreeThreads / 64) * 8);
    const unsigned tree_grid = tree_tiles * kTreeLeaves;
    Device d;
    unsigned* d_xcd_map{};
    CK(hipMalloc(&d_xcd_map, std::max(tree_grid, 768u) * sizeof(unsigned)));
    for (unsigned grid : {tree_grid, 256u, 768u}) {
        void* map_args[] = {&d_xcd_map};
        CK(hipModuleLaunchKernel(xcd_map, grid, 1, 1, 1, 1, 1, 0, nullptr, map_args, nullptr));
        std::vector<unsigned> xcds(grid), xcd_hist(8);
        CK(hipMemcpy(xcds.data(), d_xcd_map, grid * sizeof(unsigned), hipMemcpyDeviceToHost));
        for (unsigned xcd : xcds) {
            if (xcd >= xcd_hist.size()) {
                std::fprintf(stderr, "REFUSE: hardware XCD id %u is outside [0,8)\n", xcd);
                return 2;
            }
            xcd_hist[xcd]++;
        }
        for (unsigned xcd = 0; xcd < xcd_hist.size(); ++xcd) {
            if (xcd_hist[xcd] != grid / 8) {
                std::fprintf(stderr, "REFUSE: hardware XCD %u owns %u/%u workgroups\n",
                             xcd, xcd_hist[xcd], grid);
                return 2;
            }
        }
        if (grid == tree_grid) {
            for (unsigned block = 0; block < grid; ++block) {
                if (xcds[block] != (block & 7u)) {
                    std::fprintf(stderr,
                                 "REFUSE: tree placement block %u ran on XCD %u, expected %u\n",
                                 block, xcds[block], block & 7u);
                    return 2;
                }
            }
        }
    }
    CK(hipMalloc(&d.arena, arena_bytes));
    CK(hipMemset(d.arena, 0x7f, arena_bytes));
    CK(hipMalloc(&d.x, static_cast<size_t>(kHidden) * sizeof(uint16_t)));
    CK(hipMemset(d.x, 0x3c, static_cast<size_t>(kHidden) * sizeof(uint16_t)));
    CK(hipMalloc(&d.fu, fu_bytes));
    CK(hipMalloc(&d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t)));
    CK(hipMalloc(&d.partial, partial_bytes));
    CK(hipMalloc(&d.control, 56 * sizeof(unsigned)));
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));

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

    static constexpr float route_gates[kTopK] = {
        0.0661533847f, 0.0308158584f, 0.132974565f, 0.0147974715f,
        0.109471351f, 0.0747038722f, 0.0118481694f, 0.103660278f,
        0.00765970955f, 0.0885862559f, 0.0142702451f, 0.0185310878f,
        0.0867218673f, 0.168911487f, 0.0252905823f, 0.0456038304f};
    std::vector<unsigned char> tables(static_cast<size_t>(kRotations) * kTopK * 8);
    for (unsigned rotation = 0; rotation < kRotations; ++rotation) {
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            unsigned char* entry = &tables[(static_cast<size_t>(rotation) * kTopK + slot) * 8];
            const unsigned expert = rotation * kTopK + slot;
            const float gate = route_gates[slot];
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
    CK(hipDeviceSynchronize());
    const auto reference_fu = copy_bytes(d.fu, fu_bytes);
    launch_down(down, d, 304, 0);
    CK(hipDeviceSynchronize());
    const auto grid_reference_partial = copy_bytes(d.partial, partial_bytes);

    std::vector<uint16_t> varied_fu(static_cast<size_t>(kTopK) * kIntermediate);
    for (size_t i = 0; i < varied_fu.size(); ++i) {
        const float value = static_cast<float>(static_cast<int>((i * 37) % 251) - 125) /
                            64.0f;
        varied_fu[i] = float_to_bf16(value);
    }
    CK(hipMemcpy(d.fu, varied_fu.data(), fu_bytes, hipMemcpyHostToDevice));
    launch_down(down, d, 304, 0);
    CK(hipDeviceSynchronize());
    const auto reference_partial = copy_bytes(d.partial, partial_bytes);
    const TreeOracle cpu_tree = tree_oracle(reference_partial);
    launch_combine(combine, d);
    CK(hipDeviceSynchronize());
    const auto reference_out = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));

    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_control_us = time_us([&](unsigned rotation) {
        launch_down(down, d, 256, rotation);
        launch_combine(combine, d);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_phase_us = time_us([&](unsigned rotation) {
        launch_down_combine(down_combine, d, rotation);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_768_control_us = time_us([&](unsigned rotation) {
        launch_down(down, d, 768, rotation);
        launch_combine(combine, d);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_owned_us = time_us([&](unsigned rotation) {
        launch_down_combine_owned(down_combine_owned, d, 768, rotation);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_ready_us = time_us([&](unsigned rotation) {
        launch_down_combine_owned(down_combine_ready, d, 768, rotation);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    const double down_combine_tree_us = time_us([&](unsigned rotation) {
        launch_down_combine_tree(down_combine_tree, d, rotation);
    });
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    launch_down_combine(down_combine, d, 0);
    CK(hipDeviceSynchronize());
    const auto phase_out = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));
    size_t phase_out_diff = 0;
    for (size_t i = 0; i < phase_out.size(); ++i)
        phase_out_diff += phase_out[i] != reference_out[i];
    std::printf("down_combine,control_us=%.6f,phase_us=%.6f,delta_us=%+.6f,"
                "projected_92_delta_ms=%+.6f,out_diff=%zu\n",
                down_combine_control_us, down_combine_phase_us,
                down_combine_phase_us - down_combine_control_us,
                (down_combine_phase_us - down_combine_control_us) * 92.0 / 1000.0,
                phase_out_diff);
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    launch_down_combine_owned(down_combine_owned, d, 768, 0);
    CK(hipDeviceSynchronize());
    const auto owned_out = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));
    size_t owned_out_diff = 0;
    for (size_t i = 0; i < owned_out.size(); ++i)
        owned_out_diff += owned_out[i] != reference_out[i];
    std::printf("down_combine_owned,control768_us=%.6f,owned768_us=%.6f,delta_us=%+.6f,"
                "vs_control256_us=%+.6f,projected_92_delta_ms=%+.6f,out_diff=%zu\n",
                down_combine_768_control_us, down_combine_owned_us,
                down_combine_owned_us - down_combine_768_control_us,
                down_combine_owned_us - down_combine_control_us,
                (down_combine_owned_us - down_combine_768_control_us) * 92.0 / 1000.0,
                owned_out_diff);
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    launch_down_combine_owned(down_combine_ready, d, 768, 0);
    CK(hipDeviceSynchronize());
    const auto ready_out = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));
    size_t ready_out_diff = 0;
    for (size_t i = 0; i < ready_out.size(); ++i)
        ready_out_diff += ready_out[i] != reference_out[i];
    std::printf("down_combine_ready,control768_us=%.6f,ready768_us=%.6f,delta_us=%+.6f,"
                "vs_control256_us=%+.6f,projected_92_delta_ms=%+.6f,out_diff=%zu\n",
                down_combine_768_control_us, down_combine_ready_us,
                down_combine_ready_us - down_combine_768_control_us,
                down_combine_ready_us - down_combine_control_us,
                (down_combine_ready_us - down_combine_768_control_us) * 92.0 / 1000.0,
                ready_out_diff);
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
    launch_down_combine_tree(down_combine_tree, d, 0);
    CK(hipDeviceSynchronize());
    const auto tree_out = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));
    const Error tree_error = compare_bf16(tree_out, reference_out);
    const Error tree_implementation_error = compare_bf16(tree_out, cpu_tree.bf16);
    size_t repeat_diff = 0;
    for (unsigned repeat = 0; repeat < 32; ++repeat) {
        launch_down_combine_tree(down_combine_tree, d, 0);
        CK(hipDeviceSynchronize());
        const auto repeated = copy_bytes(d.out, static_cast<size_t>(kHidden) * sizeof(uint16_t));
        for (size_t i = 0; i < repeated.size(); ++i)
            repeat_diff += repeated[i] != tree_out[i];
    }
    std::printf("down_combine_tree,leaves=%u,threads=%u,control768_us=%.6f,tree_us=%.6f,delta_us=%+.6f,"
                "vs_control256_us=%+.6f,projected_92_gain_ms=%.6f,out_diff=%zu,"
                "rel_l2=%.9g,max_abs=%.9g,repeat_diff=%zu,cpu_tree_f32_diff=%zu,"
                "cpu_tree_f32_rel_l2=%.9g,cpu_tree_f32_max_abs=%.9g,"
                "cpu_tree_bf16_diff=%zu,gpu_vs_cpu_tree_diff=%zu\n",
                kTreeLeaves, kTreeThreads, down_combine_768_control_us, down_combine_tree_us,
                down_combine_tree_us - down_combine_768_control_us,
                down_combine_tree_us - down_combine_control_us,
                (down_combine_768_control_us - down_combine_tree_us) * 92.0 / 1000.0,
                tree_error.different, tree_error.rel_l2, tree_error.max_abs, repeat_diff,
                cpu_tree.f32.different, cpu_tree.f32.rel_l2, cpu_tree.f32.max_abs,
                cpu_tree.bf16_different, tree_implementation_error.different);
    CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));

    std::puts("grid,glu_us,down_us,chain_us,pair_us,handoff_us,glu_gbps,down_gbps,chain_ms_x92,fu_diff,partial_diff,pair_fu_diff,pair_partial_diff,sync_diff");
    for (unsigned grid : {1u, 64u, 128u, 192u, 256u, 384u, 512u, 768u}) {
        std::fprintf(stderr, "grid=%u control\n", grid);
        std::fflush(stderr);
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
        double pair_us = 0.0, handoff_us = 0.0;
        if (grid == 256) {
            std::fprintf(stderr, "grid=%u pair\n", grid);
            std::fflush(stderr);
            pair_us = time_us([&](unsigned rotation) { launch_pair(pair, d, grid, rotation); });
            std::fprintf(stderr, "grid=%u handoff\n", grid);
            std::fflush(stderr);
            handoff_us = time_us([&](unsigned) { launch_handoff(handoff, d, grid); });
        }

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
            partial_diff += got_partial[i] != grid_reference_partial[i];
        }

        size_t pair_fu_diff = 0, pair_partial_diff = 0, sync_diff = 0;
        if (grid == 256) {
            CK(hipMemset(d.fu, 0, fu_bytes));
            CK(hipMemset(d.partial, 0, partial_bytes));
            CK(hipMemset(d.control, 0, 56 * sizeof(unsigned)));
            launch_pair(pair, d, grid, 0);
            CK(hipDeviceSynchronize());
            const auto pair_fu = copy_bytes(d.fu, fu_bytes);
            const auto pair_partial = copy_bytes(d.partial, partial_bytes);
            std::vector<unsigned> sync(25);
            CK(hipMemcpy(sync.data(), d.control, sync.size() * sizeof(sync[0]),
                         hipMemcpyDeviceToHost));
            for (size_t i = 0; i < pair_fu.size(); ++i)
                pair_fu_diff += pair_fu[i] != reference_fu[i];
            for (size_t i = 0; i < pair_partial.size(); ++i)
                pair_partial_diff += pair_partial[i] != grid_reference_partial[i];
            for (unsigned domain = 0; domain < 8; ++domain) {
                sync_diff += sync[domain] != 32;
                sync_diff += sync[8 + domain] != 32;
                sync_diff += sync[16 + domain] != 32;
            }
            sync_diff += sync[24] != 256;
        }

        std::printf("%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.6f,%zu,%zu,%zu,%zu,%zu\n", grid,
                    glu_us, down_us, chain_us, pair_us, handoff_us,
                    glu_bytes / (glu_us * 1e-6) / 1e9,
                    down_bytes / (down_us * 1e-6) / 1e9,
                    chain_us * 92.0 / 1000.0, fu_diff, partial_diff,
                    pair_fu_diff, pair_partial_diff, sync_diff);
        std::fflush(stdout);
    }

    CK(hipModuleUnload(module));
    return 0;
}
