#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                     \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static constexpr unsigned TOKENS = 8192;
static constexpr unsigned HEADS = 12;
static constexpr unsigned DK = 512;
static constexpr unsigned DV = 128;
static constexpr unsigned NSPLIT = 1;
static constexpr unsigned GRID = 256;
static constexpr unsigned THREADS = 512;
static constexpr size_t FLUSH_BYTES = 512ull << 20;

static uint32_t rng_state = 0x12345678u;

static float random_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return static_cast<float>((rng_state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static void launch(hipFunction_t fn, void** args) {
    CK(hipModuleLaunchKernel(fn, GRID, 1, 1, THREADS, 1, 1, 0, nullptr, args, nullptr));
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <kernel.co> [odd-samples]\n", argv[0]);
        return 2;
    }
    const int samples = argc > 2 ? std::atoi(argv[2]) : 31;
    if (samples < 3 || !(samples & 1)) {
        std::fprintf(stderr, "samples must be odd and >= 3\n");
        return 2;
    }

    CK(hipInit(0));
    hipModule_t module;
    CK(hipModuleLoad(&module, argv[1]));
    const char* names[] = {"k_fold_v128_tb1", "k_fold_v128_tb2", "k_fold_v128_tb4",
                           "k_fold_v128_tb8"};
    hipFunction_t kernels[4], flush;
    for (unsigned i = 0; i < 4; ++i) CK(hipModuleGetFunction(&kernels[i], module, names[i]));
    CK(hipModuleGetFunction(&flush, module, "k_cache_flush"));

    const size_t rows = (size_t)TOKENS * HEADS;
    const size_t wuv_count = (size_t)HEADS * DK * DV;
    const size_t opart_count = rows * NSPLIT * DK;
    const size_t ml_count = rows * NSPLIT * 2;
    const size_t out_count = rows * DV;

    std::vector<uint16_t> host_wuv(wuv_count), reference(out_count), current(out_count);
    std::vector<float> host_opart(opart_count), host_ml(ml_count);
    for (auto& value : host_wuv) value = to_bf16(random_float() * 0.05f);
    for (auto& value : host_opart) value = random_float();
    for (size_t row = 0; row < rows; ++row) {
        host_ml[row * 2] = random_float() * 2.0f;
        host_ml[row * 2 + 1] = 1.0f + 0.5f * (random_float() + 1.0f);
    }

    uint16_t *wuv, *out;
    float *opart, *mlpart;
    uint32_t *flush_in, *flush_out;
    CK(hipMalloc(&wuv, wuv_count * sizeof(*wuv)));
    CK(hipMalloc(&out, out_count * sizeof(*out)));
    CK(hipMalloc(&opart, opart_count * sizeof(*opart)));
    CK(hipMalloc(&mlpart, ml_count * sizeof(*mlpart)));
    CK(hipMalloc(&flush_in, FLUSH_BYTES));
    CK(hipMalloc(&flush_out, sizeof(*flush_out)));
    CK(hipMemcpy(wuv, host_wuv.data(), wuv_count * sizeof(*wuv), hipMemcpyHostToDevice));
    CK(hipMemcpy(opart, host_opart.data(), opart_count * sizeof(*opart), hipMemcpyHostToDevice));
    CK(hipMemcpy(mlpart, host_ml.data(), ml_count * sizeof(*mlpart), hipMemcpyHostToDevice));
    CK(hipMemset(flush_in, 0x5a, FLUSH_BYTES));
    CK(hipMemset(flush_out, 0, sizeof(*flush_out)));

    unsigned n_batch = TOKENS, n_head = HEADS, v = DV, nsplit = NSPLIT;
    void* args[] = {&out, &opart, &mlpart, &wuv, &n_batch, &n_head, &v, &nsplit};
    size_t flush_count = FLUSH_BYTES / sizeof(*flush_in);
    void* flush_args[] = {&flush_in, &flush_out, &flush_count};

    bool failed = false;
    std::printf("shape T=%u H=%u DK=%u V=%u nsplit=%u grid=%u WG=%u wave=64\n", TOKENS,
                HEADS, DK, DV, NSPLIT, GRID, THREADS);
    std::printf("%-20s %12s %12s %12s\n", "variant", "median_us", "vs_tb1", "bitcmp");
    double control_us = 0.0;
    for (unsigned kernel = 0; kernel < 4; ++kernel) {
        CK(hipMemset(out, 0, out_count * sizeof(*out)));
        launch(kernels[kernel], args);
        CK(hipDeviceSynchronize());

        std::vector<double> times;
        times.reserve(samples);
        for (int sample = 0; sample < samples; ++sample) {
            launch(flush, flush_args);
            CK(hipDeviceSynchronize());
            hipEvent_t begin, end;
            CK(hipEventCreate(&begin));
            CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(kernels[kernel], args);
            CK(hipEventRecord(end));
            CK(hipEventSynchronize(end));
            float elapsed_ms = 0.0f;
            CK(hipEventElapsedTime(&elapsed_ms, begin, end));
            CK(hipEventDestroy(begin));
            CK(hipEventDestroy(end));
            times.push_back(elapsed_ms * 1000.0);
        }
        CK(hipMemcpy(current.data(), out, out_count * sizeof(*out), hipMemcpyDeviceToHost));
        const double us = median(times);
        const char* bitcmp;
        if (kernel == 0) {
            reference = current;
            control_us = us;
            bitcmp = "REFERENCE";
        } else {
            const bool identical = std::memcmp(reference.data(), current.data(),
                                               out_count * sizeof(*out)) == 0;
            failed |= !identical;
            bitcmp = identical ? "IDENTICAL" : "DIFFERS";
        }
        std::printf("%-20s %12.3f %11.3fx %12s\n", names[kernel], us, control_us / us, bitcmp);
    }
    return failed ? 1 : 0;
}
