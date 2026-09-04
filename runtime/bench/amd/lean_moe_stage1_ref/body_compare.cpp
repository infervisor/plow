// Exact comparator for stage-1 A4-reuse GEMM bodies: the same quant/sort output feeds the
// shipping reuse GEMM and a candidate GEMM; payload/scale bytes must match, then the quant
// pass and both GEMM bodies are timed separately (31 order-alternated, cache-flushed samples).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

#ifndef MOE1_T
#define MOE1_T 8192
#endif
#ifndef MOE1_H
#define MOE1_H 3584
#endif
#ifndef MOE1_I
#define MOE1_I 384
#endif
#ifndef MOE1_E
#define MOE1_E 896
#endif
#ifndef MOE1_TOPK
#define MOE1_TOPK 16
#endif
constexpr uint32_t T = MOE1_T, H = MOE1_H, I = MOE1_I, E = MOE1_E,
                   TOPK = MOE1_TOPK, BM = 64;
constexpr uint32_t UNUSED = ~0u;

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
__global__ void fill16(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        p[i] = uint16_t((x & 0x8000u) | 0x3f00u | (x & 0x7fu));
    }
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
        std::fprintf(stderr, "dry by default; usage: body_compare SHIPPING.elf CANDIDATE.elf --run "
                             "[CAND_SYMBOL] [CAND_LDS]\n");
        return 2;
    }
    const char* cand_symbol = argc > 4 ? argv[4] : "plow_moe1_a4_reuse_16x16x128_gfx950";
    const uint32_t cand_lds = argc > 5 ? std::atoi(argv[5]) : 32768;
    hipModule_t ship_mod{}, cand_mod{}; hipFunction_t quant{}, ship{}, cand{};
    CK(hipModuleLoad(&ship_mod, argv[1])); CK(hipModuleLoad(&cand_mod, argv[2]));
    CK(hipModuleGetFunction(&quant, ship_mod, "plow_moe1_quant_sort_a4_gfx950"));
    CK(hipModuleGetFunction(&ship, ship_mod, "plow_moe1_a4_reuse_16x16x128_gfx950"));
    CK(hipModuleGetFunction(&cand, cand_mod, cand_symbol));
    CK(hipFuncSetAttribute(ship, hipFuncAttributeMaxDynamicSharedMemorySize, 32768));
    CK(hipFuncSetAttribute(cand, hipFuncAttributeMaxDynamicSharedMemorySize, cand_lds));

    std::vector<std::vector<uint32_t>> buckets(E); uint32_t state = 930100;
    for (uint32_t p = 0; p < T * TOPK; ++p) {
        state = xorshift(state); uint32_t e = state % E; state = xorshift(state);
        if ((state & 3u) == 0) { state = xorshift(state); e = state % (E / 8); }
        buckets[e].push_back(p);
    }
    std::vector<int32_t> rowoff(E), counts(E), tilep(E + 1);
    std::vector<uint32_t> row_token, row_partidx;
    for (uint32_t e = 0; e < E; ++e) {
        rowoff[e] = tilep[e] * BM; counts[e] = buckets[e].size();
        uint32_t tiles = (buckets[e].size() + BM - 1) / BM; tilep[e + 1] = tilep[e] + tiles;
        for (uint32_t p : buckets[e]) { row_token.push_back(p / TOPK); row_partidx.push_back(p); }
        row_token.resize(row_token.size() + tiles * BM - buckets[e].size(), UNUSED);
        row_partidx.resize(row_partidx.size() + tiles * BM - buckets[e].size(), UNUSED);
    }
    std::vector<int32_t> meta; meta.insert(meta.end(), rowoff.begin(), rowoff.end());
    meta.insert(meta.end(), counts.begin(), counts.end()); meta.insert(meta.end(), tilep.begin(), tilep.end());
    auto d_meta = device_copy(meta); auto d_token = device_copy(row_token);
    auto d_partidx = device_copy(row_partidx); const size_t rows = row_token.size();

    void *activation{}, *weight{}, *scale{};
    const size_t branch_w = size_t(I) * (H / 2), branch_s = size_t(I) * (H / 32);
    CK(hipMalloc(&activation, size_t(T) * H * 2));
    CK(hipMalloc(&weight, size_t(E) * 2 * branch_w)); CK(hipMalloc(&scale, size_t(E) * 2 * branch_s));
    fill16<<<4096, 256>>>((uint16_t*)activation, size_t(T) * H, 0x93010001u);
    fill8<<<4096, 256>>>((uint8_t*)weight, size_t(E) * 2 * branch_w, 0x93010002u, 0);
    fill8<<<4096, 256>>>((uint8_t*)scale, size_t(E) * 2 * branch_s, 0x93010003u, 1);
    CK(hipDeviceSynchronize());
    std::vector<uint64_t> wtab(E * 3), stab(E * 3);
    for (uint32_t e = 0; e < E; ++e) for (uint32_t b = 0; b < 2; ++b) {
        size_t ix = size_t(e) * 2 + b;
        wtab[e * 3 + b] = reinterpret_cast<uint64_t>(weight) + ix * branch_w;
        stab[e * 3 + b] = reinterpret_cast<uint64_t>(scale) + ix * branch_s;
    }
    auto d_wt = device_copy(wtab); auto d_st = device_copy(stab);

    const size_t a4_bytes = rows * (H / 2), a4_scale_bytes = rows * (H / 32);
    const size_t out_bytes = rows * (I / 2), out_scale_bytes = rows * (I / 32);
    void *a4{}, *a4s{}, *ship_out{}, *cand_out{}, *ship_os{}, *cand_os{};
    CK(hipMalloc(&a4, a4_bytes)); CK(hipMalloc(&a4s, a4_scale_bytes));
    CK(hipMalloc(&ship_out, out_bytes)); CK(hipMalloc(&cand_out, out_bytes));
    CK(hipMalloc(&ship_os, out_scale_bytes)); CK(hipMalloc(&cand_os, out_scale_bytes));
    const uint32_t grid = tilep[E] * ((I + 127) / 128);
    auto launch_quant = [&] {
        uint32_t row_capacity = rows, experts = E, hidden = H;
        void* qa[] = {&a4, &a4s, &activation, &d_token, &d_meta, &row_capacity, &experts, &hidden};
        CK(hipModuleLaunchKernel(quant, 1024, 1, 1, 256, 1, 1, 0, nullptr, qa, nullptr));
    };
    auto launch_gemm = [&](hipFunction_t f, void* out, void* os, uint32_t lds) {
        uint32_t inter = I, hidden = H, experts = E, act = 2; float beta = 4, linear = 25;
        void* ga[] = {&out, &a4, &d_wt, &a4s, &d_st, &d_meta, &d_partidx, &os,
                      &inter, &hidden, &experts, &act, &beta, &linear};
        CK(hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, lds, nullptr, ga, nullptr));
    };
    auto launch_ship = [&] { launch_gemm(ship, ship_out, ship_os, 32768); };
    auto launch_cand = [&] { launch_gemm(cand, cand_out, cand_os, cand_lds); };
    CK(hipMemset(ship_out, 0xa5, out_bytes)); CK(hipMemset(cand_out, 0xa5, out_bytes));
    CK(hipMemset(ship_os, 0xa5, out_scale_bytes)); CK(hipMemset(cand_os, 0xa5, out_scale_bytes));
    launch_quant(); launch_ship(); launch_cand(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> so(out_bytes), co(out_bytes), ss(out_scale_bytes), cs(out_scale_bytes);
    CK(hipMemcpy(so.data(), ship_out, out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(co.data(), cand_out, out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ss.data(), ship_os, out_scale_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(cs.data(), cand_os, out_scale_bytes, hipMemcpyDeviceToHost));
    size_t bad = 0, sbad = 0; for (size_t i = 0; i < out_bytes; ++i) bad += so[i] != co[i];
    for (size_t i = 0; i < out_scale_bytes; ++i) sbad += ss[i] != cs[i];
    std::printf("geometry T=%u H=%u I=%u E=%u topk=%u rows=%zu tiles=%d grid=%u\n",
                T, H, I, E, TOPK, rows, tilep[E], grid);
    std::printf("oracle payload_bytes=%zu bad=%zu scale_bytes=%zu bad=%zu\n",
                out_bytes, bad, out_scale_bytes, sbad);
    if (bad || sbad) return 3;

    void* flush{}; constexpr size_t flush_bytes = 256u * 1024 * 1024; CK(hipMalloc(&flush, flush_bytes));
    CK(hipMemset(flush, 0x5a, flush_bytes)); std::vector<float> qv, sv, cv;
    auto measured = [&](int which, std::vector<float>& samples) {
        hipEvent_t begin{}, end{}; CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
        flush_cache<<<4096, 256>>>((uint32_t*)flush, flush_bytes / 4, uint32_t(samples.size() + 1));
        CK(hipEventRecord(begin));
        if (which == 0) launch_quant(); else if (which == 1) launch_ship(); else launch_cand();
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end)); float ms{};
        CK(hipEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
        CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
    };
    for (int sample = 0; sample < 31; ++sample) {
        measured(0, qv);
        if (sample & 1) { measured(2, cv); measured(1, sv); } else { measured(1, sv); measured(2, cv); }
    }
    auto report = [&](const char* name, std::vector<float>& v) {
        std::printf("%s n=31 mean_ms=%.6f median_ms=%.6f min_ms=%.6f\n", name, mean(v), median(v),
                    *std::min_element(v.begin(), v.end()));
    };
    report("quant_sort", qv); report("shipping_gemm", sv); report("candidate_gemm", cv);
    return 0;
}
