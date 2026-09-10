// Build from the repository root:
// nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
//   -arch=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia \
//   runtime/nvidia/experiments/gemma_fp8_w8a16_probe.cu -o /tmp/gemma_fp8_w8a16_probe
// Standalone experiment: changed accumulation order, no activation quantization.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 16
#include "op_gemm.cuh"
#include "gemma_fp8_w8a16_mma.cuh"

#define CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)
using bf16 = __nv_bfloat16;

__global__ void production(bf16* y, const bf16* x, const unsigned char* w,
    const float* scale, unsigned M, unsigned N, unsigned K) {
    d_gemv_fp8(y, x, w, scale, M, N, K, blockIdx.x, gridDim.x);
}

__global__ void flush_cache(unsigned* memory, unsigned count) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) memory[i] += i;
}

template <class T> T* allocate(size_t count) {
    T* result; CHECK(cudaMalloc(&result, count * sizeof(T))); return result;
}

static unsigned char encode(float value) {
    __nv_fp8_e4m3 fp8(value); return *reinterpret_cast<unsigned char*>(&fp8);
}
static float decode(unsigned char value) {
    __nv_fp8_e4m3 fp8; *reinterpret_cast<unsigned char*>(&fp8) = value; return (float)fp8;
}

static void run(unsigned M, unsigned N, unsigned K, bool exact, unsigned* flush) {
    std::mt19937 random(42);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<bf16> x((size_t)M * K), result((size_t)M * N), baseline(result.size());
    std::vector<unsigned char> weights((size_t)N * K);
    std::vector<float> scales(N);
    for (auto& value : x) value = __float2bfloat16(exact ? (int)(random() % 7) - 3 : uniform(random));
    for (auto& value : weights) value = encode(exact ? (int)(random() % 7) - 3 : uniform(random));
    for (unsigned n = 0; n < N; ++n) scales[n] = exact ? std::ldexp(1.f, (int)(n % 3) - 1) : .01f + .03f * std::abs(uniform(random));
    auto* dx = allocate<bf16>(x.size()); auto* dw = allocate<unsigned char>(weights.size());
    auto* ds = allocate<float>(N); auto* dy = allocate<bf16>(result.size());
    auto* partial = allocate<float>(result.size() * 8);
    CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds, scales.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&](unsigned split, unsigned blocks) {
        if (!split) production<<<blocks, 256>>>(dy, dx, dw, ds, M, N, K);
        else if (split == 1)
            gemma_fp8_probe::w8a16_mma<false><<<dim3((N + 63) / 64, (M + 15) / 16), 256>>>(dy, partial, dx, dw, ds, M, N, K, split);
        else {
            gemma_fp8_probe::w8a16_mma<true><<<dim3((N + 63) / 64, (M + 15) / 16, split), 256>>>(dy, partial, dx, dw, ds, M, N, K, split);
            gemma_fp8_probe::reduce<<<(M * N + 255) / 256, 256>>>(dy, partial, ds, M, N, split);
        }
        CHECK(cudaGetLastError());
    };
    launch(0, 132);
    CHECK(cudaMemcpy(baseline.data(), dy, result.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    for (unsigned variant = 0; variant < 7; ++variant) {
        const unsigned split = variant < 3 ? 0 : 1u << (variant - 3);
        const unsigned blocks = 132u << std::min(variant, 2u);
        launch(split, blocks);
        CHECK(cudaMemcpy(result.data(), dy, result.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        unsigned different = 0, nonfinite = 0, ref_bad = 0;
        double diff2 = 0, base2 = 0, max_abs = 0;
        for (size_t i = 0; i < result.size(); ++i) {
            const float got = __bfloat162float(result[i]), ref = __bfloat162float(baseline[i]);
            const double diff = got - ref;
            different += diff != 0; nonfinite += !std::isfinite(got);
            diff2 += diff * diff; base2 += (double)ref * ref; max_abs = std::max(max_abs, std::abs(diff));
        }
        const unsigned samples = exact ? result.size() : 257;
        for (unsigned sample = 0; sample < samples; ++sample) {
            const size_t i = exact ? sample : (size_t)sample * 104729 % result.size();
            const unsigned m = i / N, n = i % N;
            double sum = 0;
            for (unsigned k = 0; k < K; ++k) sum += (double)__bfloat162float(x[(size_t)m * K + k]) * decode(weights[(size_t)n * K + k]);
            const float ref = __bfloat162float(__float2bfloat16((float)(sum * scales[n])));
            const float got = __bfloat162float(result[i]);
            ref_bad += exact ? got != ref : std::abs(got - ref) > .008f * std::abs(ref) + .0002f;
        }
        if (nonfinite || ref_bad || (!exact && std::sqrt(diff2 / base2) > .001)) {
            std::fprintf(stderr, "FAIL M=%u N=%u K=%u split=%u blocks=%u nonfinite=%u reference_bad=%u relL2=%g\n", M, N, K, split, blocks, nonfinite, ref_bad, std::sqrt(diff2 / base2));
            std::exit(2);
        }
        for (unsigned warm = 0; warm < 4; ++warm) launch(split, blocks);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> times;
        for (unsigned rep = 0; rep < (exact ? 1u : 15u); ++rep) {
            flush_cache<<<132 * 4, 256>>>(flush, 64 * 1024 * 1024);
            CHECK(cudaEventRecord(start)); launch(split, blocks); CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end)); float milliseconds;
            CHECK(cudaEventElapsedTime(&milliseconds, start, end)); times.push_back(milliseconds * 1000);
        }
        std::sort(times.begin(), times.end());
        std::printf("M=%u N=%u K=%u split=%u blocks=%u median_us=%.3f min_us=%.3f max_us=%.3f differing=%u relL2=%.9g max_abs=%.9g reference_samples=%u PASS\n",
            M, N, K, split, blocks, times[times.size() / 2], times.front(), times.back(), different, std::sqrt(diff2 / (base2 + 1e-30)), max_abs, samples);
        std::fflush(stdout);
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(dx)); CHECK(cudaFree(dw)); CHECK(cudaFree(ds)); CHECK(cudaFree(dy)); CHECK(cudaFree(partial));
}

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 4 || std::atoi(argv[1]) < 1 || std::atoi(argv[2]) < 1 ||
        std::atoi(argv[3]) < 256 || std::atoi(argv[3]) % 256)) {
        std::fprintf(stderr, "usage: %s [M>0 N>0 K=multiple-of-256]\n", argv[0]);
        return 1;
    }
    auto* flush = allocate<unsigned>(64 * 1024 * 1024);
    CHECK(cudaMemset(flush, 0, 256 * 1024 * 1024));
    run(8, 64, 256, true, flush); run(17, 71, 256, true, flush);
    if (argc == 4) run(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), false, flush);
    else for (unsigned batch : {8u, 16u, 32u}) {
        run(batch, 8192, 5376, false, flush);
        run(batch, 21504, 5376, false, flush);
        run(batch, 5376, 21504, false, flush);
    }
    CHECK(cudaFree(flush));
}
