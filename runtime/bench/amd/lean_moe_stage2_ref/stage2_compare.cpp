#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

#ifndef MOE2_T
#define MOE2_T 8192
#endif
#ifndef MOE2_H
#define MOE2_H 3584
#endif
#ifndef MOE2_I
#define MOE2_I 384
#endif
#ifndef MOE2_E
#define MOE2_E 896
#endif
#ifndef MOE2_TOPK
#define MOE2_TOPK 16
#endif
constexpr uint32_t T = MOE2_T, H = MOE2_H, I = MOE2_I, E = MOE2_E,
                   TOPK = MOE2_TOPK, BM = 64;
constexpr uint32_t UNUSED = ~0u;

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
__global__ void fill8(uint8_t* p, size_t n, uint32_t seed, uint32_t scale) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        p[i] = scale ? uint8_t(124u + x % 7u) : uint8_t(x);
    }
}
__global__ void flush_cache(uint32_t* p, size_t n, uint32_t salt) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) p[i] = mix32(p[i] ^ uint32_t(i) ^ salt);
}

uint32_t xorshift(uint32_t x) { x ^= x << 13; x ^= x >> 17; return x ^ (x << 5); }
uint32_t mix32_host(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}

template <class V> V* device_copy(const std::vector<V>& h) {
    V* p{}; CK(hipMalloc(&p, h.size() * sizeof(V)));
    CK(hipMemcpy(p, h.data(), h.size() * sizeof(V), hipMemcpyHostToDevice)); return p;
}

double mean(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 4 || std::string(argv[3]) != "--run") {
        std::fprintf(stderr,
            "dry by default; usage: stage2_compare SHIPPING.elf CANDIDATE.elf --run "
            "[CAND_SYMBOL] [CAND_LDS] [CAND_THREADS]\n");
        return 2;
    }
    const char* cand_symbol = argc > 4 ? argv[4] : "plow_moe2_mxfp4_16x16x128_gfx950";
    const uint32_t cand_lds = argc > 5 ? std::atoi(argv[5]) : 4352;
    const uint32_t cand_threads = argc > 6 ? std::atoi(argv[6]) : 256;
    hipModule_t ship_mod{}, cand_mod{}; hipFunction_t ship{}, cand{};
    CK(hipModuleLoad(&ship_mod, argv[1])); CK(hipModuleLoad(&cand_mod, argv[2]));
    CK(hipModuleGetFunction(&ship, ship_mod, "plow_moe2_mxfp4_16x16x128_gfx950"));
    CK(hipModuleGetFunction(&cand, cand_mod, cand_symbol));
    CK(hipFuncSetAttribute(ship, hipFuncAttributeMaxDynamicSharedMemorySize, 4352));
    CK(hipFuncSetAttribute(cand, hipFuncAttributeMaxDynamicSharedMemorySize, cand_lds));

    // Same routed distribution as reuse_compare.cpp / gate.py: 25% of routes fold into E/8.
    std::vector<std::vector<uint32_t>> buckets(E); uint32_t state = 930100;
    for (uint32_t p = 0; p < T * TOPK; ++p) {
        state = xorshift(state); uint32_t e = state % E; state = xorshift(state);
        if ((state & 3u) == 0) { state = xorshift(state); e = state % (E / 8); }
        buckets[e].push_back(p);
    }
    std::vector<int32_t> rowoff(E), counts(E), tilep(E + 1);
    std::vector<uint32_t> row_partidx; std::vector<float> row_gate;
    for (uint32_t e = 0; e < E; ++e) {
        rowoff[e] = tilep[e] * BM; counts[e] = buckets[e].size();
        uint32_t tiles = (buckets[e].size() + BM - 1) / BM; tilep[e + 1] = tilep[e] + tiles;
        for (uint32_t p : buckets[e]) {
            row_partidx.push_back(p);
            row_gate.push_back(0.03125f * float(1 + (mix32_host(p) % 31)));
        }
        row_partidx.resize(row_partidx.size() + tiles * BM - buckets[e].size(), UNUSED);
        row_gate.resize(row_gate.size() + tiles * BM - buckets[e].size(), 0.0f);
    }
    std::vector<int32_t> meta; meta.insert(meta.end(), rowoff.begin(), rowoff.end());
    meta.insert(meta.end(), counts.begin(), counts.end());
    meta.insert(meta.end(), tilep.begin(), tilep.end());
    auto d_meta = device_copy(meta); auto d_partidx = device_copy(row_partidx);
    auto d_gate = device_copy(row_gate); const size_t rows = row_partidx.size();

    // Stage-2 companions: per expert N x Kbytes shuffled payload + pad256x8 shuffled scales.
    const size_t w_bytes = size_t(H) * (I / 2);
    const size_t s_bytes = size_t((H + 255) / 256 * 256) * ((I / 32 + 7) / 8 * 8);
    void *act{}, *act_scale{}, *weight{}, *scale{};
    CK(hipMalloc(&act, rows * (I / 2))); CK(hipMalloc(&act_scale, rows * (I / 32)));
    CK(hipMalloc(&weight, size_t(E) * w_bytes)); CK(hipMalloc(&scale, size_t(E) * s_bytes));
    fill8<<<4096, 256>>>((uint8_t*)act, rows * (I / 2), 0x93020001u, 0);
    fill8<<<4096, 256>>>((uint8_t*)act_scale, rows * (I / 32), 0x93020002u, 1);
    fill8<<<4096, 256>>>((uint8_t*)weight, size_t(E) * w_bytes, 0x93020003u, 0);
    fill8<<<4096, 256>>>((uint8_t*)scale, size_t(E) * s_bytes, 0x93020004u, 1);
    CK(hipDeviceSynchronize());
    std::vector<uint64_t> wtab(E * 3, 0), stab(E * 3, 0);
    for (uint32_t e = 0; e < E; ++e) {
        wtab[e * 3 + 2] = reinterpret_cast<uint64_t>(weight) + size_t(e) * w_bytes;
        stab[e * 3 + 2] = reinterpret_cast<uint64_t>(scale) + size_t(e) * s_bytes;
    }
    auto d_wtab = device_copy(wtab); auto d_stab = device_copy(stab);

    const size_t part_bytes = size_t(T) * TOPK * H * 4;
    void *ship_part{}, *cand_part{};
    CK(hipMalloc(&ship_part, part_bytes)); CK(hipMalloc(&cand_part, part_bytes));
    // Runtime grid contract: n_tiles(256) * 2 * (ceil(T*topk/64) + E) half tiles.
    const uint32_t grid = ((H + 255) / 256) * 2u * ((T * TOPK + BM - 1) / BM + E);
    auto launch = [&](hipFunction_t f, void* part, uint32_t lds, uint32_t threads) {
        int32_t model_dim = H, inter_dim = I, experts = E, zero = 0;
        void* a[] = {&part, &act, &d_wtab, &act_scale, &d_stab, &d_meta, &d_partidx, &d_gate,
                     &model_dim, &inter_dim, &experts, &zero};
        CK(hipModuleLaunchKernel(f, grid, 1, 1, threads, 1, 1, lds, nullptr, a, nullptr));
    };
    auto launch_ship = [&] { launch(ship, ship_part, 4352, 256); };
    auto launch_cand = [&] { launch(cand, cand_part, cand_lds, cand_threads); };
    CK(hipMemset(ship_part, 0xa5, part_bytes)); CK(hipMemset(cand_part, 0xa5, part_bytes));
    launch_ship(); launch_cand(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> sp(part_bytes), cp(part_bytes);
    CK(hipMemcpy(sp.data(), ship_part, part_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(cp.data(), cand_part, part_bytes, hipMemcpyDeviceToHost));
    size_t bad = 0, untouched = 0;
    for (size_t i = 0; i < part_bytes; i += 4) {
        bad += std::memcmp(&sp[i], &cp[i], 4) != 0;
        untouched += sp[i] == 0xa5 && sp[i + 1] == 0xa5 && sp[i + 2] == 0xa5 && sp[i + 3] == 0xa5;
    }
    std::printf("geometry T=%u H=%u I=%u E=%u topk=%u rows=%zu tiles=%d grid=%u\n",
                T, H, I, E, TOPK, rows, tilep[E], grid);
    std::printf("oracle part_words=%zu bad=%zu untouched=%zu\n", part_bytes / 4, bad, untouched);
    if (bad) return 3;

    void* flush{}; constexpr size_t flush_bytes = 256u * 1024 * 1024; CK(hipMalloc(&flush, flush_bytes));
    CK(hipMemset(flush, 0x5a, flush_bytes)); std::vector<float> sv, cv;
    auto measured = [&](bool candidate, std::vector<float>& samples) {
        hipEvent_t begin{}, end{}; CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
        flush_cache<<<4096, 256>>>((uint32_t*)flush, flush_bytes / 4, uint32_t(samples.size() + 1));
        CK(hipEventRecord(begin)); if (candidate) launch_cand(); else launch_ship();
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end)); float ms{};
        CK(hipEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
        CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
    };
    for (int sample = 0; sample < 31; ++sample) {
        if (sample & 1) { measured(true, cv); measured(false, sv); }
        else { measured(false, sv); measured(true, cv); }
    }
    std::printf("shipping n=31 mean_ms=%.6f median_ms=%.6f min_ms=%.6f\n", mean(sv), median(sv),
                *std::min_element(sv.begin(), sv.end()));
    std::printf("candidate n=31 mean_ms=%.6f median_ms=%.6f min_ms=%.6f\n", mean(cv), median(cv),
                *std::min_element(cv.begin(), cv.end()));
    return 0;
}
