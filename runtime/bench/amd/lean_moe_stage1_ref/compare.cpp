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

constexpr uint32_t T = 8192, H = 3584, I = 384, E = 896, TOPK = 16, BM = 64, GRID = 512;
constexpr uint32_t UNUSED = ~0u;

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}

__global__ void fill_bf16(uint16_t* out, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        out[i] = uint16_t((x & 0x8000u) | 0x3f00u | (x & 0x7fu));
    }
}

__global__ void fill_fp4(uint8_t* out, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        out[i] = uint8_t(x);
    }
}

__global__ void fill_e8m0(uint8_t* out, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        out[i] = uint8_t(124u + x % 7u);
    }
}

__global__ void compute_cache_flush(uint32_t* data, size_t count, uint32_t salt) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t value = data[i];
        data[i] = mix32(value ^ uint32_t(i) ^ salt);
    }
}

void check_launch() { CK(hipGetLastError()); }

uint32_t xorshift(uint32_t x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

struct Module {
    hipModule_t module{};
    hipFunction_t function{};
    uint32_t threads, lds, grid;
    Module(const char* path, const char* symbol, uint32_t threads_, uint32_t lds_,
           uint32_t grid_ = GRID)
        : threads(threads_), lds(lds_), grid(grid_) {
        CK(hipModuleLoad(&module, path));
        CK(hipModuleGetFunction(&function, module, symbol));
        CK(hipFuncSetAttribute(function, hipFuncAttributeMaxDynamicSharedMemorySize, lds));
    }
    void launch(void* out, void* activation, void* wtab, void* stab, void* meta,
                void* row_token, void* row_partidx, void* out_scale) {
        uint32_t inter = I, hidden = H, experts = E, act = 2, zero = 0;
        float beta = 4.0f, linear_beta = 25.0f;
        void* args[] = {&out, &activation, &wtab, &stab, &meta, &row_token, &row_partidx,
                        &out_scale, &inter, &hidden, &experts, &act, &beta, &linear_beta,
                        &zero, &zero};
        CK(hipModuleLaunchKernel(function, grid, 1, 1, threads, 1, 1, lds, nullptr, args, nullptr));
    }
};

template <class T_> T_* device_copy(const std::vector<T_>& host) {
    T_* p{}; CK(hipMalloc(&p, host.size() * sizeof(T_)));
    CK(hipMemcpy(p, host.data(), host.size() * sizeof(T_), hipMemcpyHostToDevice));
    return p;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int main(int argc, char** argv) {
    if ((argc != 4 && argc != 7 && argc != 9) || std::string(argv[3]) != "--run") {
        std::fprintf(stderr,
                     "dry by default; usage: compare SHIPPING.elf CANDIDATE.elf --run "
                     "[CANDIDATE_SYMBOL THREADS LDS [SHIPPING_GRID CANDIDATE_GRID]]\n");
        return 2;
    }
    const uint32_t shipping_grid = argc == 9
        ? static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10)) : GRID;
    const uint32_t candidate_grid = argc == 9
        ? static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10)) : GRID;
    Module shipping(argv[1], "plow_moe1_mxfp4_bk256_gfx950", 512, 119808, shipping_grid);
    const char* candidate_symbol = argc >= 7
        ? argv[4] : "plow_moe1_mxfp4_bm64_bn128_bk256_xcd8_wgm4_gfx950";
    const uint32_t candidate_threads = argc >= 7
        ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 256;
    const uint32_t candidate_lds = argc >= 7
        ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 52224;
    Module candidate(argv[2], candidate_symbol, candidate_threads, candidate_lds, candidate_grid);

    std::vector<std::vector<uint32_t>> buckets(E);
    uint32_t state = 930100;
    for (uint32_t pidx = 0; pidx < T * TOPK; ++pidx) {
        state = xorshift(state); uint32_t expert = state % E;
        state = xorshift(state);
        if ((state & 3u) == 0) { state = xorshift(state); expert = state % (E / 8); }
        buckets[expert].push_back(pidx);
    }
    std::vector<int32_t> rowoff(E), counts(E), tilep(E + 1);
    std::vector<uint32_t> row_token, row_partidx;
    for (uint32_t e = 0; e < E; ++e) {
        rowoff[e] = tilep[e] * BM;
        counts[e] = buckets[e].size();
        uint32_t tiles = (buckets[e].size() + BM - 1) / BM;
        tilep[e + 1] = tilep[e] + tiles;
        for (uint32_t pidx : buckets[e]) {
            row_token.push_back(pidx / TOPK);
            row_partidx.push_back(pidx);
        }
        row_token.resize(row_token.size() + tiles * BM - buckets[e].size(), UNUSED);
        row_partidx.resize(row_partidx.size() + tiles * BM - buckets[e].size(), UNUSED);
    }
    std::vector<int32_t> meta;
    meta.insert(meta.end(), rowoff.begin(), rowoff.end());
    meta.insert(meta.end(), counts.begin(), counts.end());
    meta.insert(meta.end(), tilep.begin(), tilep.end());
    auto* d_meta = device_copy(meta);
    auto* d_token = device_copy(row_token);
    auto* d_partidx = device_copy(row_partidx);

    void *activation{}, *weight{}, *scale{}, *wtab{}, *stab{};
    size_t activation_bytes = size_t(T) * H * 2;
    size_t branch_weight_bytes = size_t(I) * (H / 2);
    size_t branch_scale_bytes = size_t(I) * (H / 32);
    CK(hipMalloc(&activation, activation_bytes));
    CK(hipMalloc(&weight, size_t(E) * 2 * branch_weight_bytes));
    CK(hipMalloc(&scale, size_t(E) * 2 * branch_scale_bytes));
    fill_bf16<<<4096, 256>>>(static_cast<uint16_t*>(activation), size_t(T) * H, 0x93010001u);
    check_launch();
    fill_fp4<<<4096, 256>>>(static_cast<uint8_t*>(weight),
                            size_t(E) * 2 * branch_weight_bytes, 0x93010002u);
    check_launch();
    fill_e8m0<<<4096, 256>>>(static_cast<uint8_t*>(scale),
                             size_t(E) * 2 * branch_scale_bytes, 0x93010003u);
    check_launch();
    std::vector<uint64_t> hwtab(E * 3), hstab(E * 3);
    for (uint32_t e = 0; e < E; ++e) for (uint32_t branch = 0; branch < 2; ++branch) {
        hwtab[e * 3 + branch] = reinterpret_cast<uint64_t>(weight) +
                               (size_t(e) * 2 + branch) * branch_weight_bytes;
        hstab[e * 3 + branch] = reinterpret_cast<uint64_t>(scale) +
                               (size_t(e) * 2 + branch) * branch_scale_bytes;
    }
    wtab = device_copy(hwtab); stab = device_copy(hstab);

    size_t rows = row_token.size(), payload_bytes = rows * (I / 2), scale_bytes = rows * (I / 32);
    void *so{}, *co{}, *ss{}, *cs{};
    CK(hipMalloc(&so, payload_bytes)); CK(hipMalloc(&co, payload_bytes));
    CK(hipMalloc(&ss, scale_bytes)); CK(hipMalloc(&cs, scale_bytes));
    CK(hipMemset(so, 0xa5, payload_bytes)); CK(hipMemset(co, 0xa5, payload_bytes));
    CK(hipMemset(ss, 0xa5, scale_bytes)); CK(hipMemset(cs, 0xa5, scale_bytes));
    shipping.launch(so, activation, wtab, stab, d_meta, d_token, d_partidx, ss);
    candidate.launch(co, activation, wtab, stab, d_meta, d_token, d_partidx, cs);
    CK(hipDeviceSynchronize());
    std::vector<uint8_t> hso(payload_bytes), hco(payload_bytes), hss(scale_bytes), hcs(scale_bytes);
    CK(hipMemcpy(hso.data(), so, payload_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hco.data(), co, payload_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hss.data(), ss, scale_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hcs.data(), cs, scale_bytes, hipMemcpyDeviceToHost));
    size_t payload_bad = 0, scales_bad = 0;
    for (size_t i = 0; i < payload_bytes; ++i) payload_bad += hso[i] != hco[i];
    for (size_t i = 0; i < scale_bytes; ++i) scales_bad += hss[i] != hcs[i];
    std::printf("oracle rows=%zu payload_bytes=%zu bad=%zu scale_bytes=%zu bad=%zu\n",
                rows, payload_bytes, payload_bad, scale_bytes, scales_bad);
    if (payload_bad || scales_bad) return 3;

    void* flush{}; constexpr size_t flush_bytes = 256u * 1024 * 1024;
    CK(hipMalloc(&flush, flush_bytes)); CK(hipMemset(flush, 0x5a, flush_bytes));
    std::vector<float> ship_samples, cand_samples;
    auto measured = [&](Module& m, void* out, void* outscale, std::vector<float>& samples) {
        hipEvent_t begin{}, end{}; CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
        compute_cache_flush<<<4096, 256>>>(static_cast<uint32_t*>(flush), flush_bytes / 4,
                                           uint32_t(samples.size() + 1));
        check_launch();
        // Default-stream order makes the timed event wait for the compute flush without
        // including it in the measured interval.
        CK(hipEventRecord(begin));
        m.launch(out, activation, wtab, stab, d_meta, d_token, d_partidx, outscale);
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
        float ms{}; CK(hipEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
        CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
    };
    for (int sample = 0; sample < 31; ++sample) {
        if ((sample & 1) == 0) {
            measured(shipping, so, ss, ship_samples); measured(candidate, co, cs, cand_samples);
        } else {
            measured(candidate, co, cs, cand_samples); measured(shipping, so, ss, ship_samples);
        }
    }
    auto report = [](const char* name, const std::vector<float>& v) {
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("%s n=31 mean_ms=%.6f median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                    name, mean, median(v), *std::min_element(v.begin(), v.end()),
                    *std::max_element(v.begin(), v.end()));
    };
    report("shipping", ship_samples); report("candidate", cand_samples);
    return 0;
}
