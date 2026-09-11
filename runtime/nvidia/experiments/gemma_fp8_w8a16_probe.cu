// Build from the repository root:
// nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
//   -arch=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia \
//   runtime/nvidia/experiments/gemma_fp8_w8a16_probe.cu -lcuda -o /tmp/gemma_fp8_w8a16_probe
// Standalone experiment: changed accumulation order, no activation quantization.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dev_isa.h"
#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 16
#include "op_gemm.cuh"
#include "gemma_fp8_w8a16_mma.cuh"

#define CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)
using bf16 = __nv_bfloat16;
static CUmodule module;
static CUfunction interpreter;
static unsigned arena_bytes;
#define DRIVER(call) do { auto error = (call); if (error != CUDA_SUCCESS) { \
    const char* message; cuGetErrorString(error, &message); \
    std::fprintf(stderr, "%s: %s\n", #call, message); std::exit(1); } } while (0)

__global__ void production(bf16* y, const bf16* x, const unsigned char* w,
    const float* scale, unsigned M, unsigned N, unsigned K, unsigned staging_bytes) {
    extern __shared__ bf16 arena[];
    if (K * sizeof(bf16) <= staging_bytes)
        d_gemv_fp8(y, x, w, scale, M, N, K, blockIdx.x, gridDim.x, arena);
    else d_gemv_fp8(y, x, w, scale, M, N, K, blockIdx.x, gridDim.x);
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

static void run(unsigned M, unsigned N, unsigned K, bool exact, unsigned* flush, unsigned packet_blocks = 0) {
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
    PlowProgram program{};
    std::vector<void*> packet_allocations;
    std::vector<PlowStreamEnt> entries(packet_blocks);
    auto upload = [&](const void* source, size_t bytes) -> void* {
        void* pointer; CHECK(cudaMalloc(&pointer, bytes));
        CHECK(cudaMemcpy(pointer, source, bytes, cudaMemcpyHostToDevice));
        packet_allocations.push_back(pointer); return pointer;
    };
    if (module && packet_blocks) {
        PlowDevInst inst{}; inst.op = PLOW_DOP_GEMV_FP8; inst.blocks = packet_blocks;
        for (auto& t : inst.t) t = PLOW_TENSOR_NONE;
        inst.t[0] = 0; inst.t[1] = 1; inst.t[2] = 2; inst.t[5] = 3;
        inst.i[0] = M; inst.i[1] = N; inst.i[2] = K;
        for (unsigned i = 0; i < packet_blocks; ++i) {
            entries[i].slice = i; entries[i].wait_len = 1; entries[i].succ_len = 1;
        }
        uint32_t offsets[]{0, packet_blocks}, successor = 1;
        PlowWait wait{0, packet_blocks};
        std::vector<uint32_t> counters(2 * PLOW_CTR_STRIDE);
        counters[0] = packet_blocks;
        void* tensors[]{dy, dx, dw, ds};
        program.insts = (PlowDevInst*)upload(&inst, sizeof(inst));
        program.gq_stream = (PlowStreamEnt*)upload(entries.data(), entries.size() * sizeof(entries[0]));
        program.gq_seg_ofs = (uint32_t*)upload(offsets, sizeof(offsets));
        program.gq_cursor = (uint32_t*)upload(counters.data() + PLOW_CTR_STRIDE, PLOW_CTR_STRIDE * 4);
        program.counters = (uint32_t*)upload(counters.data(), counters.size() * 4);
        program.waits = (PlowWait*)upload(&wait, sizeof(wait));
        program.succs = (uint32_t*)upload(&successor, sizeof(successor));
        program.tensors = (void**)upload(tensors, sizeof(tensors));
    }
    auto launch = [&](unsigned split, unsigned blocks) {
        if (split >= 16) {
            void* args[]{&program};
            DRIVER(cuLaunchKernel(interpreter, 132, 1, 1, 256, 1, 1, arena_bytes, nullptr, args, nullptr));
        } else if (!split) production<<<blocks, 256, arena_bytes>>>(dy, dx, dw, ds, M, N, K, arena_bytes);
        else if (split == 1)
            gemma_fp8_probe::w8a16_mma<false><<<dim3((N + 63) / 64, (M + 15) / 16), 256>>>(dy, partial, dx, dw, ds, M, N, K, split);
        else {
            gemma_fp8_probe::w8a16_mma<true><<<dim3((N + 63) / 64, (M + 15) / 16, split), 256>>>(dy, partial, dx, dw, ds, M, N, K, split);
            gemma_fp8_probe::reduce<<<(M * N + 255) / 256, 256>>>(dy, partial, ds, M, N, split);
        }
        CHECK(cudaGetLastError());
    };
    auto reset_packet = [&]() {
        CHECK(cudaMemsetAsync(program.gq_cursor, 0, PLOW_CTR_STRIDE * 4));
        CHECK(cudaMemsetAsync(PLOW_CTR(program.counters, 1), 0, PLOW_CTR_STRIDE * 4));
    };
    launch(0, 132);
    CHECK(cudaMemcpy(baseline.data(), dy, result.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    for (unsigned variant = 0; variant < (packet_blocks ? (module ? 10u : 8u) : 7u); ++variant) {
        const bool packet_control = variant == 7;
        const bool loaded = variant >= 8;
        const unsigned split = variant < 3 || packet_control ? 0 : 1u << (variant - (loaded ? 4 : 3));
        const unsigned blocks = packet_control || loaded ? packet_blocks : 132u << std::min(variant, 2u);
        if (loaded) {
            if (variant == 9) {
                for (auto& entry : entries) { entry.wait_len = 0; entry.succ_len = 0; }
                CHECK(cudaMemcpy((void*)program.gq_stream, entries.data(), entries.size() * sizeof(entries[0]), cudaMemcpyHostToDevice));
            }
            reset_packet();
        }
        launch(split, blocks);
        CHECK(cudaMemcpy(result.data(), dy, result.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        if (loaded) {
            uint32_t completed;
            CHECK(cudaMemcpy(&completed, PLOW_CTR(program.counters, 1), 4, cudaMemcpyDeviceToHost));
            if (completed != (variant == 8 ? packet_blocks : 0)) std::exit(3);
        }
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
        for (unsigned warm = 0; warm < 4; ++warm) { if (loaded) reset_packet(); launch(split, blocks); }
        CHECK(cudaDeviceSynchronize());
        std::vector<float> times;
        for (unsigned rep = 0; rep < (exact ? 1u : 15u); ++rep) {
            flush_cache<<<132 * 4, 256>>>(flush, 64 * 1024 * 1024);
            if (loaded) reset_packet();
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
    for (void* pointer : packet_allocations) CHECK(cudaFree(pointer));
}

int main(int argc, char** argv) {
    if (argc != 1 && ((argc != 4 && argc != 5) || std::atoi(argv[1]) < 1 || std::atoi(argv[2]) < 1 ||
        std::atoi(argv[3]) < 256 || std::atoi(argv[3]) % 256 || (argc == 5 && std::atoi(argv[4]) < 1))) {
        std::fprintf(stderr, "usage: %s [M>0 N>0 K=multiple-of-256 [packet-blocks>0]]\n", argv[0]);
        return 1;
    }
    auto* flush = allocate<unsigned>(64 * 1024 * 1024);
    CHECK(cudaMemset(flush, 0, 256 * 1024 * 1024));
    if (const char* cubin = std::getenv("PLOW_PROBE_CUBIN")) {
        DRIVER(cuModuleLoad(&module, cubin));
        DRIVER(cuModuleGetFunction(&interpreter, module, "_Z12interp_sm90a11PlowProgram"));
        CUdeviceptr address; size_t bytes;
        DRIVER(cuModuleGetGlobal(&address, &bytes, module, "plow_arena_bytes"));
        if (bytes != sizeof(arena_bytes)) return 2;
        DRIVER(cuMemcpyDtoH(&arena_bytes, address, bytes));
        DRIVER(cuFuncSetAttribute(interpreter, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, arena_bytes));
        CHECK(cudaFuncSetAttribute(production, cudaFuncAttributeMaxDynamicSharedMemorySize, arena_bytes));
    }
    run(8, 64, 256, true, flush); run(17, 71, 256, true, flush);
    if (argc >= 4) run(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), false, flush,
        argc == 5 ? std::atoi(argv[4]) : 0);
    else for (unsigned batch : {8u, 16u, 32u}) {
        run(batch, 8192, 5376, false, flush);
        run(batch, 21504, 5376, false, flush);
        run(batch, 5376, 21504, false, flush);
    }
    CHECK(cudaFree(flush));
}
