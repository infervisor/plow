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
        hipError_t err_ = (call);                                                                 \
        if (err_ != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,                 \
                         hipGetErrorString(err_));                                                 \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace {
constexpr unsigned kThreads = 512;
constexpr unsigned kGrid = 304;
constexpr unsigned kHeads = 12;
constexpr unsigned kLatent = 512;
constexpr unsigned kRope = 64;
constexpr unsigned kValue = 128;

struct Buffers {
    float* opart{};
    float* mlpart{};
    uint16_t* qabs{};
    uint16_t* qrope{};
    uint8_t* ckv{};
    uint16_t* krope{};
    int* kvlen{};
    float* kvscale{};
    uint16_t* wuv{};
    uint16_t* out{};
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n & 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void launch(hipFunction_t decode, hipFunction_t merge, Buffers& b, unsigned ctx,
            unsigned nsplit) {
    void* decode_args[] = {&b.opart, &b.mlpart, &b.qabs,   &b.qrope, &b.ckv,
                           &b.krope, &b.kvlen, &b.kvscale, &ctx,     &nsplit};
    CK(hipModuleLaunchKernel(decode, kGrid, 1, 1, kThreads, 1, 1, 0, nullptr,
                             decode_args, nullptr));
    void* merge_args[] = {&b.out, &b.opart, &b.mlpart, &b.wuv, &nsplit};
    CK(hipModuleLaunchKernel(merge, kGrid, 1, 1, kThreads, 1, 1, 0, nullptr,
                             merge_args, nullptr));
}

double time_chain(hipFunction_t decode, hipFunction_t merge, Buffers& b, unsigned ctx,
                  unsigned nsplit) {
    for (int i = 0; i < 2; ++i) launch(decode, merge, b, ctx, nsplit);
    CK(hipDeviceSynchronize());
    std::vector<double> samples;
    samples.reserve(12);
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    for (int sample = 0; sample < 12; ++sample) {
        CK(hipEventRecord(begin, nullptr));
        for (int repeat = 0; repeat < 4; ++repeat) launch(decode, merge, b, ctx, nsplit);
        CK(hipEventRecord(end, nullptr));
        CK(hipEventSynchronize(end));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms / 4.0);
    }
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    return median(std::move(samples));
}

std::vector<uint16_t> run_output(hipFunction_t decode, hipFunction_t merge, Buffers& b,
                                 unsigned ctx, unsigned nsplit) {
    launch(decode, merge, b, ctx, nsplit);
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> output(kHeads * kValue);
    CK(hipMemcpy(output.data(), b.out, output.size() * sizeof(uint16_t),
                 hipMemcpyDeviceToHost));
    return output;
}

void compare(const std::vector<uint16_t>& reference, const std::vector<uint16_t>& candidate,
             size_t& different, double& rel_l2) {
    different = 0;
    double err2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        different += reference[i] != candidate[i];
        const double a = bf16_to_float(reference[i]);
        const double b = bf16_to_float(candidate[i]);
        const double d = a - b;
        err2 += d * d;
        ref2 += a * a;
    }
    rel_l2 = std::sqrt(err2 / std::max(ref2, 1e-30));
}
}  // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_mla_gf_sweep.co";
    constexpr unsigned max_ctx = 128000;
    constexpr unsigned max_split = 192;

    CK(hipInit(0));
    hipModule_t module;
    CK(hipModuleLoad(&module, object));
    hipFunction_t gf4, gf6, gf12, merge;
    CK(hipModuleGetFunction(&gf4, module, "k_mla_gf4_fp8"));
    CK(hipModuleGetFunction(&gf6, module, "k_mla_gf6_fp8"));
    CK(hipModuleGetFunction(&gf12, module, "k_mla_gf12_fp8"));
    CK(hipModuleGetFunction(&merge, module, "k_mla_merge"));

    Buffers b;
    CK(hipMalloc(&b.opart, static_cast<size_t>(kHeads) * max_split * kLatent * sizeof(float)));
    CK(hipMalloc(&b.mlpart, static_cast<size_t>(kHeads) * max_split * 2 * sizeof(float)));
    CK(hipMalloc(&b.qabs, kHeads * kLatent * sizeof(uint16_t)));
    CK(hipMalloc(&b.qrope, kHeads * kRope * sizeof(uint16_t)));
    CK(hipMalloc(&b.ckv, static_cast<size_t>(max_ctx) * kLatent));
    CK(hipMalloc(&b.krope, static_cast<size_t>(max_ctx) * kRope * sizeof(uint16_t)));
    CK(hipMalloc(&b.kvlen, sizeof(int)));
    CK(hipMalloc(&b.kvscale, max_ctx * sizeof(float)));
    CK(hipMalloc(&b.wuv, static_cast<size_t>(kHeads) * kLatent * kValue * sizeof(uint16_t)));
    CK(hipMalloc(&b.out, kHeads * kValue * sizeof(uint16_t)));

    const uint16_t bf16_values[] = {0x3f80, 0xbf80, 0x3f00, 0xbe80};
    const uint8_t fp8_values[] = {0x38, 0xb8, 0x30, 0xa8};
    std::vector<uint16_t> qabs(kHeads * kLatent), qrope(kHeads * kRope);
    std::vector<uint8_t> ckv(static_cast<size_t>(max_ctx) * kLatent);
    std::vector<uint16_t> krope(static_cast<size_t>(max_ctx) * kRope);
    std::vector<float> scale(max_ctx);
    std::vector<uint16_t> wuv(static_cast<size_t>(kHeads) * kLatent * kValue);
    for (size_t i = 0; i < qabs.size(); ++i) qabs[i] = bf16_values[(i * 5 + 1) & 3];
    for (size_t i = 0; i < qrope.size(); ++i) qrope[i] = bf16_values[(i * 3 + 2) & 3];
    for (size_t i = 0; i < ckv.size(); ++i) ckv[i] = fp8_values[(i * 7 + i / kLatent) & 3];
    for (size_t i = 0; i < krope.size(); ++i) krope[i] = bf16_values[(i * 11 + i / kRope) & 3];
    for (size_t i = 0; i < scale.size(); ++i) scale[i] = 0.75f + static_cast<float>(i % 7) / 32.0f;
    for (size_t i = 0; i < wuv.size(); ++i) wuv[i] = bf16_values[(i * 13 + 3) & 3];
    CK(hipMemcpy(b.qabs, qabs.data(), qabs.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.qrope, qrope.data(), qrope.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.ckv, ckv.data(), ckv.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.krope, krope.data(), krope.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.kvscale, scale.data(), scale.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.wuv, wuv.data(), wuv.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    std::puts("ctx,ns,gf,work_items,chain_ms,speedup_vs_gf4_ns64,different,rel_l2");
    for (unsigned ctx : {8192u, 32000u, 64000u, 128000u}) {
        const int length = static_cast<int>(ctx);
        CK(hipMemcpy(b.kvlen, &length, sizeof(length), hipMemcpyHostToDevice));
        const auto reference = run_output(gf4, merge, b, ctx, 64);
        const double reference_ms = time_chain(gf4, merge, b, ctx, 64);
        for (const auto& arm : std::vector<std::pair<unsigned, hipFunction_t>>{
                 {4, gf4}, {6, gf6}, {12, gf12}}) {
            for (unsigned nsplit :
                 {32u, 64u, 72u, 80u, 88u, 96u, 112u, 128u, 144u, 152u, 160u, 192u}) {
                const auto output = run_output(arm.second, merge, b, ctx, nsplit);
                size_t different = 0;
                double rel_l2 = 0.0;
                compare(reference, output, different, rel_l2);
                const double ms = time_chain(arm.second, merge, b, ctx, nsplit);
                const unsigned work_items = (kHeads / arm.first) * nsplit;
                std::printf("%u,%u,%u,%u,%.6f,%.6f,%zu,%.9g\n", ctx, nsplit,
                            arm.first, work_items, ms, reference_ms / ms, different, rel_l2);
            }
        }
    }

    CK(hipModuleUnload(module));
    return 0;
}
