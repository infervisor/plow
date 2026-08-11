#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call)                                                                                  \
    do {                                                                                          \
        const hipError_t error_ = (call);                                                         \
        if (error_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,                 \
                         hipGetErrorString(error_));                                               \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

struct QkvgConvArgs {
    unsigned long long x, wq, wk, wv, wg;
    unsigned long long raw_q, raw_k, raw_v, raw_g;
    unsigned long long mix_q, mix_k, mix_v;
    unsigned long long conv_wq, conv_wk, conv_wv;
    unsigned long long conv_sq, conv_sk, conv_sv;
};

namespace {
constexpr unsigned kLayers = 8;
constexpr unsigned kN = 1536;
constexpr unsigned kK = 7168;
constexpr unsigned kW = 4;
constexpr unsigned kBlocks = 304;
constexpr unsigned kThreads = 512;
constexpr unsigned kSamples = 12;

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float random_value(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<float>(state & 0xffffu) / 65536.0f - 0.5f) * 0.2f;
}

template <class T>
T* allocate(size_t count) {
    T* pointer = nullptr;
    CK(hipMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

template <class T>
void upload_chunks(T* device, size_t count, uint32_t& state) {
    const size_t chunk = std::min<size_t>(count, 1u << 20);
    std::vector<T> host(chunk);
    for (size_t offset = 0; offset < count; offset += chunk) {
        const size_t n = std::min(chunk, count - offset);
        for (size_t i = 0; i < n; ++i) {
            if constexpr (sizeof(T) == 2)
                host[i] = to_bf16(random_value(state));
            else
                host[i] = random_value(state);
        }
        CK(hipMemcpy(device + offset, host.data(), n * sizeof(T), hipMemcpyHostToDevice));
    }
}

size_t byte_differences(const void* left, const void* right, size_t bytes) {
    std::vector<unsigned char> a(bytes), b(bytes);
    CK(hipMemcpy(a.data(), left, bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), right, bytes, hipMemcpyDeviceToHost));
    size_t count = 0;
    for (size_t i = 0; i < bytes; ++i) count += a[i] != b[i];
    return count;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[(values.size() - 1) / 2] + values[values.size() / 2]);
}

void launch(hipFunction_t function, QkvgConvArgs* args) {
    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(function, kBlocks, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr,
                             config));
}
}  // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_qkvg_conv_sweep_gfx942.co";
    hipDeviceProp_t properties;
    CK(hipGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount != static_cast<int>(kBlocks)) {
        std::fprintf(stderr, "expected %u CUs, got %d\n", kBlocks, properties.multiProcessorCount);
        return 2;
    }

    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t qkvg, conv, fused;
    CK(hipModuleGetFunction(&qkvg, module, "k3_qkvg_control"));
    CK(hipModuleGetFunction(&conv, module, "k3_conv3_control"));
    CK(hipModuleGetFunction(&fused, module, "k3_qkvg_conv"));

    const size_t x_count = (size_t)kLayers * kK;
    const size_t weight_count = (size_t)kLayers * 4u * kN * kK;
    const size_t raw_count = (size_t)kLayers * 4u * kN;
    const size_t mix_count = (size_t)kLayers * 3u * kN;
    const size_t conv_weight_count = (size_t)kLayers * 3u * kN * kW;
    const size_t conv_state_count = conv_weight_count;
    uint32_t random = 0x714942u;

    uint16_t* x = allocate<uint16_t>(x_count);
    uint16_t* weights = allocate<uint16_t>(weight_count);
    float* conv_weights = allocate<float>(conv_weight_count);
    float* conv_source = allocate<float>(conv_state_count);
    upload_chunks(x, x_count, random);
    upload_chunks(weights, weight_count, random);
    upload_chunks(conv_weights, conv_weight_count, random);
    upload_chunks(conv_source, conv_state_count, random);

    uint16_t* raw_control = allocate<uint16_t>(raw_count);
    uint16_t* raw_fused = allocate<uint16_t>(raw_count);
    uint16_t* mix_control = allocate<uint16_t>(mix_count);
    uint16_t* mix_fused = allocate<uint16_t>(mix_count);
    float* state_control = allocate<float>(conv_state_count);
    float* state_fused = allocate<float>(conv_state_count);

    std::vector<QkvgConvArgs*> control_args(kLayers), fused_args(kLayers);
    const size_t weight_stride = (size_t)kN * kK;
    for (unsigned layer = 0; layer < kLayers; ++layer) {
        QkvgConvArgs control{};
        control.x = (unsigned long long)(x + (size_t)layer * kK);
        control.wq = (unsigned long long)(weights + ((size_t)layer * 4u + 0u) * weight_stride);
        control.wk = (unsigned long long)(weights + ((size_t)layer * 4u + 1u) * weight_stride);
        control.wv = (unsigned long long)(weights + ((size_t)layer * 4u + 2u) * weight_stride);
        control.wg = (unsigned long long)(weights + ((size_t)layer * 4u + 3u) * weight_stride);
        control.raw_q = (unsigned long long)(raw_control + ((size_t)layer * 4u + 0u) * kN);
        control.raw_k = (unsigned long long)(raw_control + ((size_t)layer * 4u + 1u) * kN);
        control.raw_v = (unsigned long long)(raw_control + ((size_t)layer * 4u + 2u) * kN);
        control.raw_g = (unsigned long long)(raw_control + ((size_t)layer * 4u + 3u) * kN);
        control.mix_q = (unsigned long long)(mix_control + ((size_t)layer * 3u + 0u) * kN);
        control.mix_k = (unsigned long long)(mix_control + ((size_t)layer * 3u + 1u) * kN);
        control.mix_v = (unsigned long long)(mix_control + ((size_t)layer * 3u + 2u) * kN);
        const size_t conv_base = (size_t)layer * 3u * kN * kW;
        control.conv_wq = (unsigned long long)(conv_weights + conv_base + 0u * kN * kW);
        control.conv_wk = (unsigned long long)(conv_weights + conv_base + 1u * kN * kW);
        control.conv_wv = (unsigned long long)(conv_weights + conv_base + 2u * kN * kW);
        control.conv_sq = (unsigned long long)(state_control + conv_base + 0u * kN * kW);
        control.conv_sk = (unsigned long long)(state_control + conv_base + 1u * kN * kW);
        control.conv_sv = (unsigned long long)(state_control + conv_base + 2u * kN * kW);
        CK(hipMalloc(&control_args[layer], sizeof(control)));
        CK(hipMemcpy(control_args[layer], &control, sizeof(control), hipMemcpyHostToDevice));

        QkvgConvArgs candidate = control;
        candidate.raw_q = (unsigned long long)(raw_fused + ((size_t)layer * 4u + 0u) * kN);
        candidate.raw_k = (unsigned long long)(raw_fused + ((size_t)layer * 4u + 1u) * kN);
        candidate.raw_v = (unsigned long long)(raw_fused + ((size_t)layer * 4u + 2u) * kN);
        candidate.raw_g = (unsigned long long)(raw_fused + ((size_t)layer * 4u + 3u) * kN);
        candidate.mix_q = (unsigned long long)(mix_fused + ((size_t)layer * 3u + 0u) * kN);
        candidate.mix_k = (unsigned long long)(mix_fused + ((size_t)layer * 3u + 1u) * kN);
        candidate.mix_v = (unsigned long long)(mix_fused + ((size_t)layer * 3u + 2u) * kN);
        candidate.conv_sq = (unsigned long long)(state_fused + conv_base + 0u * kN * kW);
        candidate.conv_sk = (unsigned long long)(state_fused + conv_base + 1u * kN * kW);
        candidate.conv_sv = (unsigned long long)(state_fused + conv_base + 2u * kN * kW);
        CK(hipMalloc(&fused_args[layer], sizeof(candidate)));
        CK(hipMemcpy(fused_args[layer], &candidate, sizeof(candidate), hipMemcpyHostToDevice));
    }

    auto reset = [&] {
        CK(hipMemcpy(state_control, conv_source, conv_state_count * sizeof(float),
                     hipMemcpyDeviceToDevice));
        CK(hipMemcpy(state_fused, conv_source, conv_state_count * sizeof(float),
                     hipMemcpyDeviceToDevice));
    };
    auto run_control = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer) {
            launch(qkvg, control_args[layer]);
            launch(conv, control_args[layer]);
        }
    };
    auto run_fused = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer) launch(fused, fused_args[layer]);
    };

    reset();
    run_control();
    run_fused();
    CK(hipDeviceSynchronize());
    const size_t raw_diff = byte_differences(raw_control, raw_fused, raw_count * sizeof(uint16_t));
    const size_t mix_diff = byte_differences(mix_control, mix_fused, mix_count * sizeof(uint16_t));
    const size_t state_diff =
        byte_differences(state_control, state_fused, conv_state_count * sizeof(float));
    if (raw_diff || mix_diff || state_diff) {
        std::fprintf(stderr, "FAIL raw=%zu mix=%zu state=%zu\n", raw_diff, mix_diff, state_diff);
        return 3;
    }

    std::vector<double> control_times, fused_times;
    auto measure = [&](auto&& body) {
        hipEvent_t begin, end;
        CK(hipEventCreate(&begin));
        CK(hipEventCreate(&end));
        reset();
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(begin));
        body();
        CK(hipEventRecord(end));
        CK(hipEventSynchronize(end));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, begin, end));
        CK(hipEventDestroy(begin));
        CK(hipEventDestroy(end));
        return static_cast<double>(ms);
    };
    for (unsigned sample = 0; sample < kSamples + 3u; ++sample) {
        const bool reverse = sample & 1u;
        const double first = reverse ? measure(run_fused) : measure(run_control);
        const double second = reverse ? measure(run_control) : measure(run_fused);
        if (sample >= 3u) {
            control_times.push_back(reverse ? second : first);
            fused_times.push_back(reverse ? first : second);
        }
    }
    const double control_ms = median(control_times);
    const double fused_ms = median(fused_times);
    const double projected_ms = (control_ms - fused_ms) * 69.0 / kLayers;
    std::printf("{\"schema\":\"plow.k3-qkvg-conv-sweep.v1\",\"device\":\"%s\","
                "\"layers\":%u,\"blocks\":%u,\"control_ms\":%.6f,\"fused_ms\":%.6f,"
                "\"saving_ms\":%.6f,\"projected_69_ms\":%.6f,\"raw_diff\":%zu,"
                "\"mix_diff\":%zu,\"state_diff\":%zu}\n",
                properties.name, kLayers, kBlocks, control_ms, fused_ms, control_ms - fused_ms,
                projected_ms, raw_diff, mix_diff, state_diff);
    CK(hipModuleUnload(module));
    return 0;
}
