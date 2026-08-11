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

namespace {
constexpr unsigned kLayers = 69;
constexpr unsigned kHeads = 12;
constexpr unsigned kDim = 128;
constexpr unsigned kBv = 8;
constexpr unsigned kWidth = 4;
constexpr unsigned kProj = kHeads * kDim;
constexpr unsigned kItems = kHeads * kDim / kBv;
constexpr unsigned kThreads = 512;
constexpr unsigned kSamples = 12;
constexpr unsigned kSharedBytes = (3 * kDim + 2 * 8 + kBv) * sizeof(float);

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float random_value(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<float>(state & 0xffffu) / 65536.0f - 0.5f) * 0.2f;
}

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    CK(hipMalloc(&device, host.size() * sizeof(T)));
    CK(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return device;
}

template <class T>
T* allocate(size_t count) {
    T* device = nullptr;
    CK(hipMalloc(&device, count * sizeof(T)));
    return device;
}

void launch(hipFunction_t function, unsigned blocks, unsigned shared, void** args) {
    CK(hipModuleLaunchKernel(function, blocks, 1, 1, kThreads, 1, 1, shared, nullptr, args,
                             nullptr));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[(values.size() - 1) / 2] + values[values.size() / 2]);
}

size_t differences(const void* left, const void* right, size_t bytes) {
    std::vector<unsigned char> a(bytes), b(bytes);
    CK(hipMemcpy(a.data(), left, bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), right, bytes, hipMemcpyDeviceToHost));
    size_t count = 0;
    for (size_t i = 0; i < bytes; ++i) count += a[i] != b[i];
    return count;
}

struct Error {
    double rel_l2;
    double max_abs;
    size_t nonfinite;
};

float bf16_value(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <class T, class Convert>
Error error(const T* left, const T* right, size_t count, Convert convert) {
    std::vector<T> a(count), b(count);
    CK(hipMemcpy(a.data(), left, count * sizeof(T), hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), right, count * sizeof(T), hipMemcpyDeviceToHost));
    double squared = 0.0, reference = 0.0, maximum = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        const double x = convert(a[i]), y = convert(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            ++nonfinite;
            continue;
        }
        const double delta = x - y;
        squared += delta * delta;
        reference += x * x;
        maximum = std::max(maximum, std::abs(delta));
    }
    return {std::sqrt(squared / std::max(reference, 1e-30)), maximum, nonfinite};
}
}  // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_kda_conv_step_db_gfx942.co";
    const size_t p3 = 3ull * kLayers * kProj;
    const size_t cs_count = p3 * kWidth;
    const size_t state_count = (size_t)kLayers * kHeads * kDim * kDim;
    const size_t output_count = (size_t)kLayers * kProj;
    uint32_t rng = 0x714942u;
    std::vector<uint16_t> raw(p3), gate(output_count), beta((size_t)kLayers * kHeads);
    std::vector<float> weight(cs_count), conv_source(cs_count), alog((size_t)kLayers * kHeads),
        dt_bias(output_count), state_source(state_count);
    for (auto& value : raw) value = bf16(random_value(rng));
    for (auto& value : gate) value = bf16(random_value(rng));
    for (auto& value : beta) value = bf16(random_value(rng));
    for (auto& value : weight) value = random_value(rng);
    for (auto& value : conv_source) value = random_value(rng);
    for (auto& value : alog) value = -2.0f + random_value(rng);
    for (auto& value : dt_bias) value = random_value(rng);
    for (auto& value : state_source) value = random_value(rng);

    CK(hipInit(0));
    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t conv, step, fused;
    CK(hipModuleGetFunction(&conv, module, "kda_conv3_control"));
    CK(hipModuleGetFunction(&step, module, "kda_step_control"));
    CK(hipModuleGetFunction(&fused, module, "kda_conv_step_db"));

    uint16_t* d_raw = upload(raw);
    uint16_t* d_gate = upload(gate);
    uint16_t* d_beta = upload(beta);
    float* d_weight = upload(weight);
    float* d_conv_source = upload(conv_source);
    float* d_alog = upload(alog);
    float* d_dt_bias = upload(dt_bias);
    float* d_state_source = upload(state_source);
    uint16_t *mix_control = allocate<uint16_t>(p3), *mix_fused = allocate<uint16_t>(p3);
    uint16_t *output_control = allocate<uint16_t>(output_count),
             *output_fused = allocate<uint16_t>(output_count);
    float *conv_control = allocate<float>(cs_count), *conv_fused = allocate<float>(cs_count);
    float *state_control = allocate<float>(state_count), *state_fused = allocate<float>(state_count);

    auto reset = [&] {
        CK(hipMemcpy(conv_control, d_conv_source, cs_count * sizeof(float), hipMemcpyDeviceToDevice));
        CK(hipMemcpy(state_control, d_state_source, state_count * sizeof(float),
                     hipMemcpyDeviceToDevice));
        CK(hipMemcpy(state_fused, d_state_source, state_count * sizeof(float),
                     hipMemcpyDeviceToDevice));
    };
    auto run_control = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer) {
            void* conv_args[] = {&mix_control, &d_raw, &d_weight, &conv_control, &layer};
            launch(conv, 304, 0, conv_args);
            void* step_args[] = {&output_control, &mix_control, &d_gate, &d_beta,
                                 &d_alog, &d_dt_bias, &state_control, &layer};
            launch(step, kItems, kSharedBytes, step_args);
        }
    };
    auto run_fused = [&] {
        for (unsigned layer = 0; layer < kLayers; ++layer) {
            void* args[] = {&output_fused, &mix_fused, &d_raw, &d_weight, &d_conv_source,
                            &conv_fused, &d_gate, &d_beta, &d_alog, &d_dt_bias, &state_fused,
                            &layer};
            launch(fused, kItems, kSharedBytes, args);
        }
    };

    reset();
    run_control();
    run_fused();
    CK(hipDeviceSynchronize());
    const size_t mix_diff = differences(mix_control, mix_fused, p3 * sizeof(uint16_t));
    const size_t conv_diff = differences(conv_control, conv_fused, cs_count * sizeof(float));
    const size_t state_diff = differences(state_control, state_fused, state_count * sizeof(float));
    const size_t output_diff =
        differences(output_control, output_fused, output_count * sizeof(uint16_t));
    const auto mix_error = error(mix_control, mix_fused, p3, bf16_value);
    const auto state_error = error(state_control, state_fused, state_count,
                                   [](float value) { return value; });
    const auto output_error = error(output_control, output_fused, output_count, bf16_value);
    if (conv_diff || mix_error.nonfinite || state_error.nonfinite || output_error.nonfinite ||
        mix_error.rel_l2 > 1e-4 || state_error.rel_l2 > 1e-4 || output_error.rel_l2 > 1e-4) {
        std::fprintf(stderr,
                     "FAIL mix=%zu/%.3e conv=%zu state=%zu/%.3e output=%zu/%.3e\n", mix_diff,
                     mix_error.rel_l2, conv_diff, state_diff, state_error.rel_l2, output_diff,
                     output_error.rel_l2);
        return 2;
    }

    auto measure = [&](auto&& body) {
        hipEvent_t begin, end;
        CK(hipEventCreate(&begin));
        CK(hipEventCreate(&end));
        std::vector<double> times;
        for (unsigned sample = 0; sample < kSamples + 3; ++sample) {
            reset();
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(begin, nullptr));
            body();
            CK(hipEventRecord(end, nullptr));
            CK(hipEventSynchronize(end));
            float milliseconds = 0.0f;
            CK(hipEventElapsedTime(&milliseconds, begin, end));
            if (sample >= 3) times.push_back(milliseconds);
        }
        CK(hipEventDestroy(begin));
        CK(hipEventDestroy(end));
        return median(std::move(times));
    };
    const double control_ms = measure(run_control);
    const double fused_ms = measure(run_fused);
    std::printf("{\"schema\":\"plow.k3-kda-conv-step-db.v1\",\"layers\":%u,"
                "\"control_ms\":%.6f,\"fused_ms\":%.6f,\"saving_ms\":%.6f,"
                "\"mix_diff\":%zu,\"conv_diff\":%zu,\"state_diff\":%zu,"
                "\"output_diff\":%zu,\"mix_rel_l2\":%.9g,\"state_rel_l2\":%.9g,"
                "\"output_rel_l2\":%.9g,\"state_max_abs\":%.9g}\n",
                kLayers, control_ms, fused_ms, control_ms - fused_ms, mix_diff, conv_diff,
                state_diff, output_diff, mix_error.rel_l2, state_error.rel_l2,
                output_error.rel_l2, state_error.max_abs);
    CK(hipModuleUnload(module));
    return 0;
}
