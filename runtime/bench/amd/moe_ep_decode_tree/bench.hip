#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "amd_common.h"
#include "op_moe.h"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

namespace {
constexpr unsigned H = 3584, I = 768, K = 16, E = 896, OWNED = E / 2;
constexpr unsigned ROTATIONS = 56, THREADS = 512, WAVES = THREADS / 64;

extern "C" __global__ __launch_bounds__(THREADS, 2)
void ep2_glu(bf16* fu, const bf16* x, const unsigned char* table,
             const unsigned long long* weights, const unsigned long long* scales) {
    d_moe_group_glu_fp8_blk(fu, x, table, weights, scales, K, I, H, E,
                            PLOW_MOE_ACT_SILU, blockIdx.x, gridDim.x,
                            PLOW_MOE_ENC_MXFP4, 0.0f, 0.0f);
}

extern "C" __global__ __launch_bounds__(THREADS, 2)
void ep2_down(float* part, const bf16* fu, const unsigned char* table,
              const unsigned long long* weights, const unsigned long long* scales) {
    d_moe_group_down_fp8_blk(part, fu, table, weights, scales, K, H, I, E,
                             blockIdx.x, gridDim.x, PLOW_MOE_ENC_MXFP4);
}

extern "C" __global__ __launch_bounds__(256, 4)
void ep2_combine(bf16* out, const float* part) {
    d_moe_combine(out, nullptr, nullptr, part, H, K, blockIdx.x, gridDim.x);
}

/* Global slots remain the leaves. A null expert pointer contributes an explicit zero; ownership
 * never renumbers or compacts the route. One wave owns one output row across all leaves, so no
 * producer rendezvous exists and arrival order cannot affect the balanced compiler tree. */
extern "C" __global__ __launch_bounds__(THREADS, 2)
void ep2_down_tree(bf16* out, const bf16* fu, const unsigned char* table,
                   const unsigned long long* weights,
                   const unsigned long long* scales) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned kb = (I + 127u) >> 7;
    for (unsigned h = blockIdx.x * WAVES + wave; h < H; h += gridDim.x * WAVES) {
        float leaf[K];
#pragma unroll
        for (unsigned slot = 0; slot < K; ++slot) {
            const unsigned eid = moe_slot_expert(table, slot);
            const unsigned long long wb = eid < E ? weights[(size_t)eid * 3 + 2] : 0ull;
            leaf[slot] = 0.0f;
            if (wb == 0ull) continue;
            const unsigned char* w = (const unsigned char*)(size_t)wb;
            const float* sf = (const float*)(size_t)scales[(size_t)eid * 3 + 2];
            const float y = wave_dot_enc(
                PLOW_MOE_ENC_MXFP4, fu + (size_t)slot * I,
                w + (size_t)h * (I >> 1), sf + (size_t)(h >> 7) * kb,
                (const unsigned char*)sf + (size_t)h * (I >> 5), I, lane);
            leaf[slot] = moe_slot_gate(table, slot) * y;
        }
#pragma unroll
        for (unsigned stride = 1; stride < K; stride <<= 1)
#pragma unroll
            for (unsigned slot = 0; slot < K; slot += 2 * stride)
                leaf[slot] += leaf[slot + stride];
        if (lane == 0) out[h] = f2bf(leaf[0]);
    }
}

extern "C" __device__ unsigned plow_moe_ep2_tree_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_ep2_tree_global_slots16_1 = 1;
extern "C" __device__ unsigned plow_moe_ep2_tree_null_remote_zero_1 = 1;
extern "C" __device__ unsigned plow_moe_ep2_tree_balanced_1 = 1;

uint16_t f2bf_host(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7fffu + ((word >> 16) & 1u);
    return static_cast<uint16_t>(word >> 16);
}

float bf2f_host(uint16_t bits) {
    uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
}

double time_us(const std::function<void(unsigned)>& launch) {
    for (unsigned r = 0; r < ROTATIONS; ++r) launch(r);
    HC(hipDeviceSynchronize());
    hipEvent_t begin, end;
    HC(hipEventCreate(&begin)); HC(hipEventCreate(&end));
    std::vector<double> samples;
    for (unsigned sample = 0; sample < 12; ++sample) {
        HC(hipEventRecord(begin));
        for (unsigned r = 0; r < ROTATIONS; ++r) launch(r);
        HC(hipEventRecord(end)); HC(hipEventSynchronize(end));
        float ms; HC(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms * 1000.0 / ROTATIONS);
    }
    HC(hipEventDestroy(begin)); HC(hipEventDestroy(end));
    return median(std::move(samples));
}
}  // namespace

int main(int argc, char** argv) {
    const unsigned local = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 9;
    const unsigned grid = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 256;
    if (local == 0 || local > K) {
        std::fprintf(stderr, "local routes must be in [1,16]\n");
        return 2;
    }
    constexpr size_t weight_bytes = (size_t)I * H / 2;
    constexpr size_t scale_bytes = (size_t)I * H / 32;
    constexpr size_t expert_bytes = 3 * (weight_bytes + scale_bytes);
    constexpr size_t arena_bytes = expert_bytes * OWNED;

    unsigned char* arena;
    bf16 *x, *fu, *out;
    float* part;
    unsigned char* tables;
    unsigned long long *weights, *scales;
    HC(hipMalloc(&arena, arena_bytes)); HC(hipMemset(arena, 0x7f, arena_bytes));
    HC(hipMalloc(&x, H * sizeof(bf16)));
    std::vector<uint16_t> host_x(H);
    for (unsigned h = 0; h < H; ++h)
        host_x[h] = f2bf_host((static_cast<int>((h * 29) % 257) - 128) / 96.0f);
    HC(hipMemcpy(x, host_x.data(), H * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMalloc(&fu, (size_t)K * I * sizeof(bf16)));
    HC(hipMalloc(&out, H * sizeof(bf16)));
    HC(hipMalloc(&part, (size_t)K * H * sizeof(float)));

    std::vector<unsigned long long> host_weights(E * 3), host_scales(E * 3);
    for (unsigned expert = 0; expert < OWNED; ++expert) {
        unsigned char* base = arena + (size_t)expert * expert_bytes;
        for (unsigned matrix = 0; matrix < 3; ++matrix) {
            host_weights[expert * 3 + matrix] =
                (unsigned long long)(size_t)(base + matrix * weight_bytes);
            host_scales[expert * 3 + matrix] =
                (unsigned long long)(size_t)(base + 3 * weight_bytes + matrix * scale_bytes);
        }
    }
    HC(hipMalloc(&weights, host_weights.size() * sizeof(host_weights[0])));
    HC(hipMalloc(&scales, host_scales.size() * sizeof(host_scales[0])));
    HC(hipMemcpy(weights, host_weights.data(), host_weights.size() * sizeof(host_weights[0]),
                 hipMemcpyHostToDevice));
    HC(hipMemcpy(scales, host_scales.data(), host_scales.size() * sizeof(host_scales[0]),
                 hipMemcpyHostToDevice));

    static constexpr unsigned local_order[K] =
        {0, 2, 3, 5, 7, 9, 10, 12, 15, 1, 4, 6, 8, 11, 13, 14};
    static constexpr float gates[K] =
        {0.0661533847f, 0.0308158584f, 0.132974565f, 0.0147974715f,
         0.109471351f, 0.0747038722f, 0.0118481694f, 0.103660278f,
         0.00765970955f, 0.0885862559f, 0.0142702451f, 0.0185310878f,
         0.0867218673f, 0.168911487f, 0.0252905823f, 0.0456038304f};
    std::vector<unsigned char> host_tables((size_t)ROTATIONS * K * 8);
    for (unsigned rotation = 0; rotation < ROTATIONS; ++rotation) {
        for (unsigned rank = 0; rank < K; ++rank) {
            const unsigned slot = local_order[rank];
            const unsigned expert = rank < local
                ? (rotation * local + rank) % OWNED
                : OWNED + (rotation * K + rank) % (E - OWNED);
            unsigned char* entry = host_tables.data() + ((size_t)rotation * K + slot) * 8;
            std::memcpy(entry, &expert, sizeof(expert));
            std::memcpy(entry + 4, &gates[slot], sizeof(gates[slot]));
        }
    }
    HC(hipMalloc(&tables, host_tables.size()));
    HC(hipMemcpy(tables, host_tables.data(), host_tables.size(), hipMemcpyHostToDevice));

    auto control = [&](unsigned rotation) {
        const unsigned char* table = tables + (size_t)rotation * K * 8;
        hipLaunchKernelGGL(ep2_glu, dim3(grid), dim3(THREADS), 0, 0,
                           fu, x, table, weights, scales);
        hipLaunchKernelGGL(ep2_down, dim3(grid), dim3(THREADS), 0, 0,
                           part, fu, table, weights, scales);
        hipLaunchKernelGGL(ep2_combine, dim3(7), dim3(256), 0, 0, out, part);
    };
    auto candidate = [&](unsigned rotation) {
        const unsigned char* table = tables + (size_t)rotation * K * 8;
        hipLaunchKernelGGL(ep2_glu, dim3(grid), dim3(THREADS), 0, 0,
                           fu, x, table, weights, scales);
        hipLaunchKernelGGL(ep2_down_tree, dim3(grid), dim3(THREADS), 0, 0,
                           out, fu, table, weights, scales);
    };
    const double control_us = time_us(control);
    const double candidate_us = time_us(candidate);

    control(0); HC(hipDeviceSynchronize());
    std::vector<uint16_t> reference(H), got(H), repeat(H);
    std::vector<float> host_part((size_t)K * H);
    HC(hipMemcpy(reference.data(), out, H * sizeof(bf16), hipMemcpyDeviceToHost));
    HC(hipMemcpy(host_part.data(), part, host_part.size() * sizeof(float), hipMemcpyDeviceToHost));
    candidate(0); HC(hipDeviceSynchronize());
    HC(hipMemcpy(got.data(), out, H * sizeof(bf16), hipMemcpyDeviceToHost));

    size_t output_diff = 0, tree_bf16_diff = 0, f32_diff = 0, repeat_diff = 0;
    double delta2 = 0.0, reference2 = 0.0, max_abs = 0.0;
    for (unsigned h = 0; h < H; ++h) {
        float fixed = 0.0f, leaf[K];
        for (unsigned slot = 0; slot < K; ++slot) {
            leaf[slot] = host_part[(size_t)slot * H + h];
            fixed += leaf[slot];
        }
        for (unsigned stride = 1; stride < K; stride <<= 1)
            for (unsigned slot = 0; slot < K; slot += 2 * stride)
                leaf[slot] += leaf[slot + stride];
        const double delta = (double)leaf[0] - fixed;
        delta2 += delta * delta; reference2 += (double)fixed * fixed;
        max_abs = std::max(max_abs, std::abs(delta));
        f32_diff += leaf[0] != fixed;
        tree_bf16_diff += f2bf_host(leaf[0]) != f2bf_host(fixed);
        output_diff += got[h] != reference[h];
    }
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        candidate(0); HC(hipDeviceSynchronize());
        HC(hipMemcpy(repeat.data(), out, H * sizeof(bf16), hipMemcpyDeviceToHost));
        for (unsigned h = 0; h < H; ++h) repeat_diff += repeat[h] != got[h];
    }
    std::printf("local=%u,grid=%u,control_us=%.6f,candidate_us=%.6f,gain_us=%.6f,"
                "projected_92_gain_ms=%.6f,output_diff=%zu,repeat_diff=%zu,"
                "cpu_f32_diff=%zu,cpu_rel_l2=%.9g,cpu_max_abs=%.9g,cpu_bf16_diff=%zu\n",
                local, grid, control_us, candidate_us, control_us - candidate_us,
                (control_us - candidate_us) * 92.0 / 1000.0, output_diff, repeat_diff,
                f32_diff, std::sqrt(delta2 / std::max(reference2, 1e-30)), max_abs,
                tree_bf16_diff);
    return 0;
}
