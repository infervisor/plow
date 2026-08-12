#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#define CK(call)                                                                                  \
    do {                                                                                          \
        const hipError_t error_ = (call);                                                         \
        if (error_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,               \
                         hipGetErrorString(error_));                                               \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace {
constexpr unsigned kThreads = 512;
constexpr unsigned kExperts = 896;
constexpr unsigned kTopK = 16;
constexpr unsigned kLayers = 92;
constexpr unsigned kRotations = 64;
constexpr unsigned kSamples = 21;
constexpr unsigned kFlags = 1u | 2u | 4u;
constexpr float kRouteScale = 1.0f;

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float from_bf16(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(quantile * static_cast<double>(values.size() - 1));
    return values[index];
}

struct Arm {
    hipModule_t module{};
    hipFunction_t kernel{};
    unsigned char* table{};
    std::vector<double> samples;
};

void load_arm(Arm& arm, const char* object) {
    CK(hipModuleLoad(&arm.module, object));
    CK(hipModuleGetFunction(&arm.kernel, arm.module, "k3_router_topk"));
    CK(hipMalloc(&arm.table, static_cast<size_t>(kRotations) * kTopK * 8));
    CK(hipMemset(arm.table, 0xff, static_cast<size_t>(kRotations) * kTopK * 8));
}

void launch(Arm& arm, const uint16_t* logits, const float* bias, unsigned rotation) {
    unsigned char* table = arm.table + static_cast<size_t>(rotation) * kTopK * 8;
    const uint16_t* row = logits + static_cast<size_t>(rotation) * kExperts;
    unsigned experts = kExperts;
    unsigned topk = kTopK;
    unsigned flags = kFlags;
    float route_scale = kRouteScale;
    void* args[] = {&table, &row, &bias, &experts, &topk, &flags, &route_scale};
    CK(hipModuleLaunchKernel(arm.kernel, 1, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr));
}

double measure(Arm& arm, const uint16_t* logits, const float* bias) {
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    CK(hipEventRecord(begin, nullptr));
    for (unsigned rotation = 0; rotation < kRotations; ++rotation) {
        launch(arm, logits, bias, rotation);
    }
    CK(hipEventRecord(end, nullptr));
    CK(hipEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CK(hipEventElapsedTime(&elapsed_ms, begin, end));
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    return elapsed_ms * 1000.0 / kRotations;
}

std::vector<unsigned char> copy_table(const Arm& arm) {
    std::vector<unsigned char> host(static_cast<size_t>(kRotations) * kTopK * 8);
    CK(hipMemcpy(host.data(), arm.table, host.size(), hipMemcpyDeviceToHost));
    return host;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <control.elf> <local.elf>\n", argv[0]);
        return 2;
    }
    CK(hipInit(0));

    std::vector<uint16_t> host_logits(static_cast<size_t>(kRotations) * kExperts);
    std::vector<float> host_bias(kExperts);
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        host_bias[expert] = std::cos(static_cast<float>(expert) * 0.071f) * 0.2f;
    }
    for (unsigned rotation = 0; rotation < kRotations; ++rotation) {
        for (unsigned expert = 0; expert < kExperts; ++expert) {
            const float value = std::sin(static_cast<float>(expert) * 0.173f +
                                         static_cast<float>(rotation) * 0.317f) * 3.0f;
            host_logits[static_cast<size_t>(rotation) * kExperts + expert] = to_bf16(value);
        }
    }

    uint16_t* logits{};
    float* bias{};
    CK(hipMalloc(&logits, host_logits.size() * sizeof(host_logits[0])));
    CK(hipMalloc(&bias, host_bias.size() * sizeof(host_bias[0])));
    CK(hipMemcpy(logits, host_logits.data(), host_logits.size() * sizeof(host_logits[0]),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(bias, host_bias.data(), host_bias.size() * sizeof(host_bias[0]),
                 hipMemcpyHostToDevice));

    Arm control, local;
    load_arm(control, argv[1]);
    load_arm(local, argv[2]);
    for (unsigned warmup = 0; warmup < 4; ++warmup) {
        (void)measure(control, logits, bias);
        (void)measure(local, logits, bias);
    }
    for (unsigned sample = 0; sample < kSamples; ++sample) {
        if ((sample & 1u) == 0) {
            control.samples.push_back(measure(control, logits, bias));
            local.samples.push_back(measure(local, logits, bias));
        } else {
            local.samples.push_back(measure(local, logits, bias));
            control.samples.push_back(measure(control, logits, bias));
        }
    }

    const auto control_table = copy_table(control);
    const auto local_table = copy_table(local);
    size_t byte_diff = 0;
    for (size_t i = 0; i < control_table.size(); ++i) {
        byte_diff += control_table[i] != local_table[i];
    }

    size_t oracle_id_diff = 0;
    double oracle_gate_max = 0.0;
    std::vector<std::pair<float, unsigned>> candidates;
    candidates.reserve(kExperts);
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        const float score = 1.0f / (1.0f + std::exp(-from_bf16(host_logits[expert])));
        candidates.emplace_back(score + host_bias[expert], expert);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    });
    float gate_sum = 0.0f;
    for (unsigned rank = 0; rank < kTopK; ++rank) {
        const unsigned expert = candidates[rank].second;
        gate_sum += 1.0f / (1.0f + std::exp(-from_bf16(host_logits[expert])));
    }
    for (unsigned rank = 0; rank < kTopK; ++rank) {
        unsigned got_id;
        float got_gate;
        std::memcpy(&got_id, control_table.data() + static_cast<size_t>(rank) * 8, 4);
        std::memcpy(&got_gate, control_table.data() + static_cast<size_t>(rank) * 8 + 4, 4);
        const unsigned want_id = candidates[rank].second;
        const float want_gate =
            (1.0f / (1.0f + std::exp(-from_bf16(host_logits[want_id])))) / gate_sum;
        oracle_id_diff += got_id != want_id;
        oracle_gate_max = std::max(oracle_gate_max,
                                   std::abs(static_cast<double>(got_gate - want_gate)));
    }

    const double control_med = percentile(control.samples, 0.5);
    const double local_med = percentile(local.samples, 0.5);
    const double projected_ms = (control_med - local_med) * kLayers / 1000.0;
    std::puts("arm,median_us,p10_us,p90_us,projected_ms_x92");
    std::printf("control,%.6f,%.6f,%.6f,%.6f\n", control_med,
                percentile(control.samples, 0.1), percentile(control.samples, 0.9),
                control_med * kLayers / 1000.0);
    std::printf("local,%.6f,%.6f,%.6f,%.6f\n", local_med,
                percentile(local.samples, 0.1), percentile(local.samples, 0.9),
                local_med * kLayers / 1000.0);
    std::printf("result,byte_diff=%zu,oracle_id_diff=%zu,oracle_gate_max=%.9g,saving_ms_x92=%.6f\n",
                byte_diff, oracle_id_diff, oracle_gate_max, projected_ms);

    CK(hipModuleUnload(control.module));
    CK(hipModuleUnload(local.module));
    return byte_diff == 0 && oracle_id_diff == 0 && oracle_gate_max < 2e-6 ? 0 : 1;
}
