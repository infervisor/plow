/* Gemma-4 HD512 BKV16-order screen: production BQ32 mma.sync against the latent
 * Hopper BQ64 WGMMA body. Both process BKV16 tiles in the same order; this
 * harness measures complete BF16 output drift before a packet role is considered.
 *
 * Build with the repository's clean nvcc environment (see cmake/nvcc_cubin.sh):
 *   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
 *     -I runtime/common -I runtime/nvidia \
 *     -Xptxas=-v runtime/nvidia/experiments/gemma4_hd512_wgmma_bkv16.cu \
 *     -lcuda -o /tmp/gemma4_hd512_wgmma_bkv16
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_FA_PIPE 1
#ifndef PLOW_NV_FA_TMA
#define PLOW_NV_FA_TMA 1
#endif
#define PLOW_NV_FA512_WG 1
#define PLOW_NV_FA512_KV64 0
#ifndef PLOW_NV_FA512_QK_HALVES
#define PLOW_NV_FA512_QK_HALVES 0
#endif
#include "op_attention.cuh"

using bf16 = __nv_bfloat16;

constexpr int HD = 512;
constexpr int HEADS = 16;
constexpr int KV_HEADS = 1;
constexpr int BKV = 16;
constexpr unsigned GRID = 132;
constexpr unsigned THREADS = 256;

#define CK(call)                                                                                 \
    do {                                                                                         \
        const cudaError_t error_ = (call);                                                       \
        if (error_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "%s: %s at line %d\n", #call, cudaGetErrorString(error_),     \
                         __LINE__);                                                              \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

__global__ void control_kernel(float* partial, float* stats, const bf16* q,
                               const bf16* k, const bf16* v, bf16* out,
                               unsigned rows, unsigned kv_length, unsigned stride,
                               unsigned nsplit) {
    extern __shared__ float arena[];
    d_flash_prefill_px4<HD, 32, BKV>(
        partial, stats, q, k, v, out, rows, kv_length, HEADS, KV_HEADS,
        kv_length - rows, 0, nsplit, stride, 0xffffffffu, 1.0f,
        blockIdx.x, gridDim.x, arena);
}

__global__ void candidate_kernel(float* partial, float* stats, const bf16* q,
                                 const bf16* k, const bf16* v, bf16* out,
                                 unsigned rows, unsigned kv_length, unsigned stride,
                                 unsigned nsplit) {
    extern __shared__ float arena[];
    d_flash_prefill_sm90<HD, 64, BKV>(
        partial, stats, q, k, v, out, rows, kv_length, HEADS, KV_HEADS,
        kv_length - rows, 0, nsplit, stride, 0xffffffffu, 1.0f,
        blockIdx.x, gridDim.x, arena, nullptr, nullptr);
}

__global__ void merge_kernel(bf16* out, const float* partial, const float* stats,
                             unsigned rows, unsigned nsplit) {
    d_flash_merge<HD>(out, partial, stats, rows, HEADS, nsplit,
                      blockIdx.x, gridDim.x);
}

__global__ void evict_kernel(unsigned* data, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] += 1;
}

static unsigned parse(const char* text) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || *end || value > (1u << 20)) std::exit(2);
    return unsigned(value);
}

static std::vector<bf16> values(size_t count, uint32_t seed) {
    std::vector<bf16> result(count);
    for (bf16& value : result) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        value = __float2bfloat16(float(int32_t(seed)) * (1.0f / 2147483648.0f));
    }
    return result;
}

template <class T>
static T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    CK(cudaMalloc(&device, host.size() * sizeof(T)));
    CK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

struct Buffers {
    bf16* out = nullptr;
    float* partial = nullptr;
    float* stats = nullptr;
};

static Buffers allocate_buffers(size_t output_elements, unsigned rows,
                                unsigned nsplit) {
    Buffers result;
    CK(cudaMalloc(&result.out, output_elements * sizeof(bf16)));
    CK(cudaMalloc(&result.partial,
                  output_elements * size_t(nsplit) * sizeof(float)));
    CK(cudaMalloc(&result.stats,
                  size_t(rows) * HEADS * nsplit * 2 * sizeof(float)));
    return result;
}

static void print_resources(const char* name, const void* kernel, size_t smem) {
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, kernel));
    int blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, THREADS, smem));
    std::printf("\"%s_regs\":%d,\"%s_local\":%zu,\"%s_smem\":%zu,"
                "\"%s_blocks_per_sm\":%d,", name, attr.numRegs, name,
                size_t(attr.localSizeBytes), name, smem, name, blocks);
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s QUERY_ROWS KV_LENGTH NSPLIT [SEED]\n", argv[0]);
        return 2;
    }
    const unsigned rows = parse(argv[1]);
    const unsigned kv_length = parse(argv[2]);
    const unsigned nsplit = parse(argv[3]);
    const unsigned seed = argc == 5 ? parse(argv[4]) : 1;
    if (!rows || rows > 8192 || kv_length < rows || kv_length > 16384 ||
        !nsplit || nsplit > 16)
        return 2;

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 9 || prop.minor != 0 || prop.multiProcessorCount != int(GRID)) {
        std::fprintf(stderr, "requires 132-SM H100, got sm_%d%d/%d SM\n",
                     prop.major, prop.minor, prop.multiProcessorCount);
        return 2;
    }

    const unsigned stride = (kv_length + 31u) & ~31u;
    const size_t q_elements = size_t(rows) * HEADS * HD;
    const size_t kv_elements = size_t(stride) * KV_HEADS * HD;
    bf16* q = upload(values(q_elements, 123u ^ seed * 0x9e3779b9u));
    bf16* k = upload(values(kv_elements, 456u ^ seed * 0x85ebca6bu));
    bf16* v = upload(values(kv_elements, 789u ^ seed * 0xc2b2ae35u));
    Buffers control = allocate_buffers(q_elements, rows, nsplit);
    Buffers candidate = allocate_buffers(q_elements, rows, nsplit);

#ifdef PLOW_EXPERIMENT_LAUNCH_SMEM
    constexpr size_t control_smem = PLOW_EXPERIMENT_LAUNCH_SMEM;
    constexpr size_t candidate_smem = PLOW_EXPERIMENT_LAUNCH_SMEM;
#else
    constexpr size_t control_smem =
        FA_PX4_SMEM_FLOATS(HD, 32, BKV) * sizeof(float);
    constexpr size_t candidate_smem =
        FA_SM90_PRE_FLOATS(HD, 64, BKV) * sizeof(float);
#endif
    CK(cudaFuncSetAttribute(control_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            int(control_smem)));
    CK(cudaFuncSetAttribute(candidate_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            int(candidate_smem)));

    auto launch_control = [&] {
        control_kernel<<<GRID, THREADS, control_smem>>>(
            control.partial, control.stats, q, k, v, control.out, rows,
            kv_length, stride, nsplit);
        if (nsplit > 1)
            merge_kernel<<<GRID, THREADS>>>(control.out, control.partial,
                                            control.stats, rows, nsplit);
    };
    auto launch_candidate = [&] {
        candidate_kernel<<<GRID, THREADS, candidate_smem>>>(
            candidate.partial, candidate.stats, q, k, v, candidate.out, rows,
            kv_length, stride, nsplit);
        if (nsplit > 1)
            merge_kernel<<<GRID, THREADS>>>(candidate.out, candidate.partial,
                                            candidate.stats, rows, nsplit);
    };

    CK(cudaMemset(control.out, 0xff, q_elements * sizeof(bf16)));
    CK(cudaMemset(candidate.out, 0xff, q_elements * sizeof(bf16)));
    launch_control();
    launch_candidate();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<bf16> expected(q_elements), actual(q_elements);
    CK(cudaMemcpy(expected.data(), control.out, q_elements * sizeof(bf16),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(actual.data(), candidate.out, q_elements * sizeof(bf16),
                  cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    double error2 = 0.0, reference2 = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < q_elements; ++i) {
        mismatches += std::memcmp(&expected[i], &actual[i], sizeof(bf16)) != 0;
        const float ref = __bfloat162float(expected[i]);
        const float error = __bfloat162float(actual[i]) - ref;
        error2 += double(error) * error;
        reference2 += double(ref) * ref;
        max_abs = std::max(max_abs, std::abs(error));
    }
    const double rel_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    uint64_t control_hash = 1469598103934665603ull;
    for (const bf16& value : expected) {
        uint16_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        control_hash = (control_hash ^ uint8_t(bits)) * 1099511628211ull;
        control_hash = (control_hash ^ uint8_t(bits >> 8)) * 1099511628211ull;
    }

    unsigned* trash = nullptr;
    constexpr size_t eviction_bytes = 256ull << 20;
    CK(cudaMalloc(&trash, eviction_bytes));
    CK(cudaMemset(trash, 0, eviction_bytes));
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    for (int warm = 0; warm < 3; ++warm) {
        launch_control();
        launch_candidate();
    }
    CK(cudaDeviceSynchronize());

    std::vector<float> control_us, candidate_us;
    auto timed = [&](auto launch) {
        evict_kernel<<<GRID, THREADS>>>(trash, eviction_bytes / sizeof(unsigned));
        CK(cudaEventRecord(begin));
        launch();
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        CK(cudaEventElapsedTime(&elapsed, begin, end));
        return elapsed * 1000.0f;
    };
    for (int sample = 0; sample < 15; ++sample) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (sample & 1) {
            candidate_us.push_back(timed(launch_candidate));
            control_us.push_back(timed(launch_control));
        } else {
            control_us.push_back(timed(launch_control));
            candidate_us.push_back(timed(launch_candidate));
        }
    }
    std::sort(control_us.begin(), control_us.end());
    std::sort(candidate_us.begin(), candidate_us.end());
    const float control_p50 = control_us[control_us.size() / 2];
    const float candidate_p50 = candidate_us[candidate_us.size() / 2];

    std::printf("{\"rows\":%u,\"kv_length\":%u,\"nsplit\":%u,\"seed\":%u,",
                rows, kv_length, nsplit, seed);
    print_resources("control", (const void*)control_kernel, control_smem);
    print_resources("candidate", (const void*)candidate_kernel, candidate_smem);
    std::printf("\"control_hash\":\"fnv1a64:%016llx\",\"mismatches\":%zu,"
                "\"rel_l2\":%.9g,\"max_abs\":%.9g,"
                "\"exact\":%s,\"control_p50_us\":%.6f,"
                "\"candidate_p50_us\":%.6f,\"speedup\":%.6f}\n",
                (unsigned long long)control_hash, mismatches, rel_l2, max_abs,
                mismatches ? "false" : "true",
                control_p50, candidate_p50, control_p50 / candidate_p50);

    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    CK(cudaFree(trash));
    CK(cudaFree(control.out));
    CK(cudaFree(control.partial));
    CK(cudaFree(control.stats));
    CK(cudaFree(candidate.out));
    CK(cudaFree(candidate.partial));
    CK(cudaFree(candidate.stats));
    CK(cudaFree(q));
    CK(cudaFree(k));
    CK(cudaFree(v));
    return 0;
}
