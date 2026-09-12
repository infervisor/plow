/* Isolated Gemma-4-12B Hopper flash-prefill cell harness.
 *
 * Compile this TU repeatedly with PLOW_FA_SWEEP_HD/BKV and the production
 * PLOW_NV_FA_* flags. It drives d_flash_prefill_mux plus d_flash_merge without
 * changing packet routing. Each invocation measures one (query rows, live KV,
 * nsplit) cell and checks sampled rows against an FP64 host oracle.
 */
#include <cuda.h>
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

#ifndef PLOW_FA_SWEEP_HD
#define PLOW_FA_SWEEP_HD 256
#endif
#ifndef PLOW_FA_SWEEP_BKV
#define PLOW_FA_SWEEP_BKV 32
#endif
#ifndef PLOW_FA_SWEEP_THREADS
#define PLOW_FA_SWEEP_THREADS 256
#endif
#ifndef PLOW_NV_HOPPER
#define PLOW_NV_HOPPER 1
#endif
#ifndef PLOW_NV_FA_PIPE
#define PLOW_NV_FA_PIPE 1
#endif
#ifndef PLOW_NV_FA_TMA
#define PLOW_NV_FA_TMA 1
#endif
#ifndef PLOW_NV_FA512_WG
#define PLOW_NV_FA512_WG 1
#endif

#include "op_attention.cuh"

using bf16 = __nv_bfloat16;
constexpr int HD = PLOW_FA_SWEEP_HD;
constexpr int BKV = PLOW_FA_SWEEP_BKV;
constexpr int BQ = 64;
constexpr int HEADS = 16;
constexpr int KV_HEADS = HD == 256 ? 8 : 1;
constexpr int WINDOW = HD == 256 ? 1024 : 0;
constexpr unsigned GRID = 132;
constexpr unsigned THREADS = PLOW_FA_SWEEP_THREADS;

static_assert(HD == 256 || HD == 512);
static_assert(BKV == 16 || BKV == 32 || BKV == 64);
#if PLOW_NV_FA_GQA2_PAIR
static_assert(HD == 256 && BKV == 32);
#endif
#if PLOW_NV_FA_WGITEM
static_assert(HD == 256 && BKV == 32);
#endif
#if PLOW_NV_FA_WGITEM_ONE
static_assert(THREADS == 128);
#else
static_assert(THREADS == 256);
#endif

#define CK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); std::exit(2); } } while (0)
#define CD(call) do { CUresult error = (call); if (error != CUDA_SUCCESS) { \
    const char* text = nullptr; cuGetErrorString(error, &text); \
    std::fprintf(stderr, "%s: %s\n", #call, text ? text : "driver error"); std::exit(2); } } while (0)

template <int D, int BK>
__global__ void attention_kernel(float* partial, float* stats, const bf16* q,
                                 const bf16* k, const bf16* v, bf16* out,
                                 unsigned rows, unsigned kv_length,
                                 unsigned kv_stride, unsigned kv_mask,
                                 unsigned nsplit, const void* maps) {
    extern __shared__ float arena[];
    d_flash_prefill<D, BQ, BK>(partial, stats, q, k, v, out,
        rows, kv_length, HEADS, KV_HEADS, kv_length - rows, WINDOW, nsplit,
        kv_stride, kv_mask, 1.0f / sqrtf(float(D)), blockIdx.x, gridDim.x,
        arena, nullptr, maps);
}

template <int D>
__global__ void merge_kernel(bf16* out, const float* partial, const float* stats,
                             unsigned rows, unsigned nsplit) {
    d_flash_merge<D>(out, partial, stats, rows, HEADS, nsplit,
                     blockIdx.x, gridDim.x);
}

__global__ void evict_kernel(unsigned* data, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] += 1;
}

static std::vector<bf16> values(size_t count, uint32_t seed) {
    std::vector<bf16> result(count);
    for (auto& value : result) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        value = __float2bfloat16(float(int32_t(seed)) / 2147483648.0f);
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

static unsigned parse(const char* text) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || *end || value > 1u << 20) std::exit(2);
    return unsigned(value);
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s QUERY_ROWS KV_LENGTH NSPLIT TMA [SEED]\n", argv[0]);
        return 2;
    }
    const unsigned rows = parse(argv[1]);
    const unsigned kv_length = parse(argv[2]);
    const unsigned nsplit = parse(argv[3]);
    const bool tma = parse(argv[4]) != 0;
    const bool tma_active = tma && BKV % 32 == 0;
    const unsigned seed = argc == 6 ? parse(argv[5]) : 0;
    if (!rows || rows > 8192 || kv_length < rows || kv_length > 16384 ||
        !nsplit || nsplit > 64 || (HD == 256 && kv_length > 16384)) return 2;

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 9 || prop.minor != 0 || prop.multiProcessorCount != int(GRID)) {
        std::fprintf(stderr, "requires the qualified 132-SM H100, got sm_%d%d/%d SM\n",
                     prop.major, prop.minor, prop.multiProcessorCount);
        return 2;
    }

    const unsigned stride = HD == 256 ? 16384u : (kv_length + 31u) & ~31u;
    const unsigned mask = HD == 256 ? stride - 1 : 0xffffffffu;
    const size_t q_elements = size_t(rows) * HEADS * HD;
    const size_t kv_elements = size_t(KV_HEADS) * stride * HD;
    const auto host_q = values(q_elements, 123u ^ (seed * 0x9e3779b9u));
    const auto host_k = values(kv_elements, 456u ^ (seed * 0x85ebca6bu));
    const auto host_v = values(kv_elements, 789u ^ (seed * 0xc2b2ae35u));
    bf16* device_q = upload(host_q);
    bf16* device_k = upload(host_k);
    bf16* device_v = upload(host_v);
    bf16* device_out = nullptr;
    float* device_partial = nullptr;
    float* device_stats = nullptr;
    unsigned* device_trash = nullptr;
    constexpr size_t eviction_bytes = 256ull << 20;
    CK(cudaMalloc(&device_out, q_elements * sizeof(bf16)));
    CK(cudaMalloc(&device_partial, q_elements * nsplit * sizeof(float)));
    CK(cudaMalloc(&device_stats, size_t(rows) * HEADS * nsplit * 2 * sizeof(float)));
    CK(cudaMalloc(&device_trash, eviction_bytes));
    CK(cudaMemset(device_trash, 0, eviction_bytes));

    CUtensorMap maps[2]{};
    CUtensorMap* device_maps = nullptr;
    if (tma) {
        const uint64_t dims[]{HD, stride, KV_HEADS};
        const uint64_t strides[]{HD * sizeof(bf16), uint64_t(HD) * stride * sizeof(bf16)};
        const uint32_t box[]{64, 32, 1};
        const uint32_t steps[]{1, 1, 1};
        for (int operand = 0; operand < 2; ++operand) {
            void* base = operand ? static_cast<void*>(device_v) : static_cast<void*>(device_k);
            CD(cuTensorMapEncodeTiled(&maps[operand], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                3, base, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        device_maps = upload(std::vector<CUtensorMap>{maps[0], maps[1]});
    }

    size_t smem = FA_PRE_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float);
#if PLOW_NV_FA_GQA2_PAIR
    smem = FA_SM90_GQA2_PAIR_FLOATS(HD, BQ, BKV) * sizeof(float);
#elif PLOW_NV_FA_WGITEM_ONE
    smem = FA_SM90_WGI_ONE_FLOATS(HD, BQ, BKV) * sizeof(float);
#elif PLOW_NV_FA_WGITEM
    smem = FA_SM90_WGI_FLOATS(HD, BQ, BKV) * sizeof(float);
#endif
    CK(cudaFuncSetAttribute(attention_kernel<HD, BKV>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    cudaFuncAttributes attributes{};
    CK(cudaFuncGetAttributes(&attributes, attention_kernel<HD, BKV>));

    auto launch = [&] {
        attention_kernel<HD, BKV><<<GRID, THREADS, smem>>>(
            device_partial, device_stats, device_q, device_k, device_v, device_out,
            rows, kv_length, stride, mask, nsplit, device_maps);
        if (nsplit > 1)
            merge_kernel<HD><<<GRID, THREADS>>>(device_out, device_partial,
                                                device_stats, rows, nsplit);
    };
    CK(cudaMemset(device_out, 0xff, q_elements * sizeof(bf16)));
    launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<bf16> output(q_elements);
    CK(cudaMemcpy(output.data(), device_out, q_elements * sizeof(bf16), cudaMemcpyDeviceToHost));

    double worst_rel = 0.0;
    double max_abs = 0.0;
    bool finite = true;
    unsigned checked = 0;
    for (unsigned row : {0u, rows / 2, rows - 1}) {
        for (unsigned head : {0u, 7u, 15u}) {
            const unsigned end = kv_length - rows + row + 1;
            const unsigned begin = WINDOW && end > WINDOW ? end - WINDOW : 0;
            const unsigned kv_head = head / (HEADS / KV_HEADS);
            const size_t qi = (size_t(row) * HEADS + head) * HD;
            const size_t kv_base = size_t(kv_head) * stride * HD;
            std::vector<double> scores(end - begin);
            double maximum = -INFINITY;
            for (unsigned pos = begin; pos < end; ++pos) {
                double score = 0.0;
                const size_t ki = kv_base + size_t(pos & mask) * HD;
                for (int d = 0; d < HD; ++d)
                    score += double(__bfloat162float(host_q[qi + d])) *
                             __bfloat162float(host_k[ki + d]);
                score /= std::sqrt(double(HD));
                scores[pos - begin] = score;
                maximum = std::max(maximum, score);
            }
            double sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            double error2 = 0.0;
            double reference2 = 0.0;
            for (int d = 0; d < HD; ++d) {
                double expected = 0.0;
                for (unsigned pos = begin; pos < end; ++pos)
                    expected += scores[pos - begin] *
                        __bfloat162float(host_v[kv_base + size_t(pos & mask) * HD + d]);
                expected /= sum;
                const double actual = __bfloat162float(output[qi + d]);
                finite &= std::isfinite(actual);
                const double error = actual - expected;
                error2 += error * error;
                reference2 += expected * expected;
                max_abs = std::max(max_abs, std::abs(error));
                ++checked;
            }
            worst_rel = std::max(worst_rel,
                std::sqrt(error2 / std::max(reference2, 1e-30)));
        }
    }
    const bool correct = finite && worst_rel < 0.004 && max_abs < 0.01;

    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    for (int warm = 0; warm < 3; ++warm) launch();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(begin));
    launch();
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    float probe_ms = 0.0f;
    CK(cudaEventElapsedTime(&probe_ms, begin, end));
    const unsigned repetitions = std::max(1u, std::min(32u,
        unsigned(std::ceil(1.5 / std::max(double(probe_ms), 0.001)))));
    std::vector<float> samples;
    for (int sample = 0; sample < 15; ++sample) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        evict_kernel<<<GRID, THREADS>>>(device_trash, eviction_bytes / sizeof(unsigned));
        CK(cudaEventRecord(begin));
        for (unsigned repeat = 0; repeat < repetitions; ++repeat) launch();
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        CK(cudaEventElapsedTime(&elapsed, begin, end));
        samples.push_back(elapsed * 1000.0f / repetitions);
    }
    std::sort(samples.begin(), samples.end());

    const char* warp_layout =
#if PLOW_NV_FA_GQA2_PAIR
        "gqa2_pair";
#elif PLOW_NV_FA_WGITEM_ONE
        "one_wg";
#elif PLOW_NV_FA_WGITEM
        "wgitem";
#else
        "hd_split";
#endif
    std::printf("{\"phase\":\"prefill\",\"seed\":%u,\"head_dim\":%d,\"query_rows\":%u,"
                "\"kv_length\":%u,\"window\":%d,\"nsplit\":%u,\"bq\":%d,"
                "\"bkv\":%d,\"warp_layout\":\"%s\",\"warps\":%u,"
                "\"stages\":%d,\"tma\":%s,\"smem_bytes\":%zu,\"registers\":%d,"
                "\"local_bytes\":%zu,\"checked\":%u,\"worst_rel_l2\":%.9g,"
                "\"max_abs\":%.9g,\"correct\":%s,\"repetitions\":%u,\"samples_us\":[",
                seed, HD, rows, kv_length, WINDOW, nsplit, BQ, BKV, warp_layout, THREADS / 32,
                FA_SM90_STAGES(HD, BKV), tma_active ? "true" : "false", smem,
                attributes.numRegs, size_t(attributes.localSizeBytes), checked,
                worst_rel, max_abs, correct ? "true" : "false", repetitions);
    for (size_t i = 0; i < samples.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", samples[i]);
    std::printf("]}\n");

    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    CK(cudaFree(device_q));
    CK(cudaFree(device_k));
    CK(cudaFree(device_v));
    CK(cudaFree(device_out));
    CK(cudaFree(device_partial));
    CK(cudaFree(device_stats));
    CK(cudaFree(device_trash));
    if (device_maps) CK(cudaFree(device_maps));
    return correct ? 0 : 1;
}
