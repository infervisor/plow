#include <hip/hip_runtime.h>

#include <algorithm>
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

struct KdaNormGemvArgs {
    unsigned long long o, g, gamma, y, w, out;
};

namespace {
constexpr unsigned kLayers = 8;
constexpr unsigned kH = 12;
constexpr unsigned kD = 128;
constexpr unsigned kK = kH * kD;
constexpr unsigned kN = 7168;
constexpr unsigned kGemvBlocks = 299;
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
void upload(T* device, size_t count, uint32_t& state, float bias = 0.0f) {
    const size_t chunk = std::min<size_t>(count, 1u << 20);
    std::vector<T> host(chunk);
    for (size_t offset = 0; offset < count; offset += chunk) {
        const size_t n = std::min(chunk, count - offset);
        for (size_t i = 0; i < n; ++i) {
            const float value = bias + random_value(state);
            if constexpr (sizeof(T) == 2)
                host[i] = to_bf16(value);
            else
                host[i] = value;
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

void launch(hipFunction_t function, unsigned blocks, KdaNormGemvArgs* args) {
    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(function, blocks, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr,
                             config));
}
}  // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_kda_norm_gemv_sweep_gfx942.co";
    hipDeviceProp_t properties;
    CK(hipGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount != 304) return 2;

    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t norm, gemv, check, fused;
    CK(hipModuleGetFunction(&norm, module, "k3_kda_norm_control"));
    CK(hipModuleGetFunction(&gemv, module, "k3_kda_gemv_control"));
    CK(hipModuleGetFunction(&check, module, "k3_kda_norm_check"));
    CK(hipModuleGetFunction(&fused, module, "k3_kda_norm_gemv"));

    uint32_t random = 0x942714u;
    uint16_t* o = allocate<uint16_t>((size_t)kLayers * kK);
    uint16_t* g = allocate<uint16_t>((size_t)kLayers * kK);
    float* gamma = allocate<float>((size_t)kLayers * kD);
    uint16_t* weights = allocate<uint16_t>((size_t)kLayers * kN * kK);
    uint16_t* y_control = allocate<uint16_t>((size_t)kLayers * kK);
    uint16_t* y_check = allocate<uint16_t>((size_t)kLayers * kK);
    uint16_t* out_control = allocate<uint16_t>((size_t)kLayers * kN);
    uint16_t* out_fused = allocate<uint16_t>((size_t)kLayers * kN);
    upload(o, (size_t)kLayers * kK, random);
    upload(g, (size_t)kLayers * kK, random);
    upload(gamma, (size_t)kLayers * kD, random, 1.0f);
    upload(weights, (size_t)kLayers * kN * kK, random);

    std::vector<KdaNormGemvArgs*> control_args(kLayers), check_args(kLayers), fused_args(kLayers);
    for (unsigned layer = 0; layer < kLayers; ++layer) {
        KdaNormGemvArgs a{};
        a.o = (unsigned long long)(o + (size_t)layer * kK);
        a.g = (unsigned long long)(g + (size_t)layer * kK);
        a.gamma = (unsigned long long)(gamma + (size_t)layer * kD);
        a.y = (unsigned long long)(y_control + (size_t)layer * kK);
        a.w = (unsigned long long)(weights + (size_t)layer * kN * kK);
        a.out = (unsigned long long)(out_control + (size_t)layer * kN);
        CK(hipMalloc(&control_args[layer], sizeof(a)));
        CK(hipMemcpy(control_args[layer], &a, sizeof(a), hipMemcpyHostToDevice));
        a.y = (unsigned long long)(y_check + (size_t)layer * kK);
        CK(hipMalloc(&check_args[layer], sizeof(a)));
        CK(hipMemcpy(check_args[layer], &a, sizeof(a), hipMemcpyHostToDevice));
        a.out = (unsigned long long)(out_fused + (size_t)layer * kN);
        CK(hipMalloc(&fused_args[layer], sizeof(a)));
        CK(hipMemcpy(fused_args[layer], &a, sizeof(a), hipMemcpyHostToDevice));
    }

    auto run_control = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer) {
            launch(norm, 2, control_args[layer]);
            launch(gemv, kGemvBlocks, control_args[layer]);
        }
    };
    auto run_fused = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer)
            launch(fused, kGemvBlocks, fused_args[layer]);
    };
    run_control();
    for (unsigned layer = 0; layer < kLayers; ++layer) launch(check, 1, check_args[layer]);
    run_fused();
    CK(hipDeviceSynchronize());
    const size_t y_diff =
        byte_differences(y_control, y_check, (size_t)kLayers * kK * sizeof(uint16_t));
    const size_t out_diff =
        byte_differences(out_control, out_fused, (size_t)kLayers * kN * sizeof(uint16_t));
    if (y_diff || out_diff) {
        std::fprintf(stderr, "FAIL y=%zu out=%zu\n", y_diff, out_diff);
        return 3;
    }

    auto measure = [&](auto&& body) {
        hipEvent_t begin, end;
        CK(hipEventCreate(&begin));
        CK(hipEventCreate(&end));
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
    std::vector<double> control_times, fused_times;
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
    const double projection = (control_ms - fused_ms) * 69.0 / kLayers;
    std::printf("{\"schema\":\"plow.k3-kda-norm-gemv-sweep.v1\",\"device\":\"%s\"," 
                "\"layers\":%u,\"control_ms\":%.6f,\"fused_ms\":%.6f,"
                "\"saving_ms\":%.6f,\"projected_69_ms\":%.6f,\"y_diff\":%zu,"
                "\"out_diff\":%zu}\n",
                properties.name, kLayers, control_ms, fused_ms, control_ms - fused_ms, projection,
                y_diff, out_diff);
    return 0;
}
