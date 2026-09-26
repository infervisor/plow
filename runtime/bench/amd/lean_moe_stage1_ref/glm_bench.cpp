// GLM-5.3 TP8 stage-1 A/B: base vs candidate lean stage-1 object at the in-model launch shape
// (quant+sort launch, then the A4 reuse GEMM; 256 threads, 32 KiB dynamic LDS, grid from the
// row capacity). Routing: top-8 of 256 routed + the shared expert folded as expert 256.
// usage: glm_bench BASE.elf CAND.elf T [iters]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

constexpr uint32_t H = 6144, I = 256, E = 257, TOPK = 9, BM = 64, UNUSED = ~0u;

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
__global__ void fill16(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        p[i] = uint16_t((x & 0x8000u) | ((0x78u + (x >> 20) % 8u) << 7) | (x & 0x7fu));
    }
}
__global__ void fill8(uint8_t* p, size_t n, uint32_t seed, uint32_t scale) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        uint32_t x = mix32(uint32_t(i) ^ mix32(uint32_t(i >> 32) + seed));
        p[i] = scale ? uint8_t(118u + x % 7u) : uint8_t(x);
    }
}
__global__ void flush_cache(uint32_t* p, size_t n, uint32_t salt) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) p[i] = mix32(p[i] ^ uint32_t(i) ^ salt);
}

static uint32_t xorshift(uint32_t x) { x ^= x << 13; x ^= x >> 17; return x ^ (x << 5); }
template <class V> V* device_copy(const std::vector<V>& h) {
    V* p{}; CK(hipMalloc(&p, h.size() * sizeof(V)));
    CK(hipMemcpy(p, h.data(), h.size() * sizeof(V), hipMemcpyHostToDevice)); return p;
}
static double median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Obj {
    hipModule_t mod{};
    hipFunction_t quant{}, gemm{}, bk256{};
    bool gather = false; /* plow_moe1_a4_token_gather_1: A4 by token, GEMM takes row_token */
    explicit Obj(const char* path) {
        CK(hipModuleLoad(&mod, path));
        hipDeviceptr_t g{}; size_t gb = 0;
        gather = hipModuleGetGlobal(&g, &gb, mod, "plow_moe1_a4_token_gather_1") == hipSuccess;
        CK(hipModuleGetFunction(&quant, mod, "plow_moe1_quant_sort_a4_gfx950"));
        CK(hipModuleGetFunction(&gemm, mod, "plow_moe1_a4_reuse_16x16x128_gfx950"));
        CK(hipFuncSetAttribute(gemm, hipFuncAttributeMaxDynamicSharedMemorySize, 32768));
        CK(hipModuleGetFunction(&bk256, mod, "plow_moe1_mxfp4_bk256_gfx950"));
        CK(hipFuncSetAttribute(bk256, hipFuncAttributeMaxDynamicSharedMemorySize, 119808));
    }
};

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: glm_bench BASE.elf CAND.elf T [iters]\n"); return 2; }
    Obj base(argv[1]), cand(argv[2]);
    const uint32_t T = std::atoi(argv[3]);
    const int iters = argc > 4 ? std::atoi(argv[4]) : 21;

    std::vector<std::vector<uint32_t>> buckets(E);
    uint32_t state = 930100u + T;
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t chosen[8];
        for (uint32_t k = 0; k < 8; ++k) {
            for (;;) {
                state = xorshift(state);
                uint32_t e = state % 256u;
                bool dup = false;
                for (uint32_t j = 0; j < k; ++j) dup |= chosen[j] == e;
                if (!dup) { chosen[k] = e; break; }
            }
            buckets[chosen[k]].push_back(t * TOPK + k);
        }
        buckets[256].push_back(t * TOPK + 8);
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
    /* Runtime sizes the row buffers for the worst case; the tail is never read. */
    const uint32_t cap = ((T * TOPK + E * (BM - 1)) + BM - 1) / BM * BM;
    const size_t rows = row_token.size();
    row_token.resize(cap, UNUSED); row_partidx.resize(cap, UNUSED);
    std::vector<int32_t> meta(rowoff);
    meta.insert(meta.end(), counts.begin(), counts.end());
    meta.insert(meta.end(), tilep.begin(), tilep.end());
    auto d_meta = device_copy(meta); auto d_token = device_copy(row_token);
    auto d_partidx = device_copy(row_partidx);

    void *activation{}, *weight{}, *scale{};
    const size_t branch_w = size_t(I) * (H / 2), branch_s = size_t(I) * (H / 32);
    CK(hipMalloc(&activation, size_t(T) * H * 2));
    CK(hipMalloc(&weight, size_t(E) * 2 * branch_w)); CK(hipMalloc(&scale, size_t(E) * 2 * branch_s));
    fill16<<<4096, 256>>>((uint16_t*)activation, size_t(T) * H, 0x93010001u);
    fill8<<<4096, 256>>>((uint8_t*)weight, size_t(E) * 2 * branch_w, 0x93010002u, 0);
    fill8<<<4096, 256>>>((uint8_t*)scale, size_t(E) * 2 * branch_s, 0x93010003u, 1);
    std::vector<uint64_t> wtab(E * 3), stab(E * 3);
    for (uint32_t e = 0; e < E; ++e) for (uint32_t b = 0; b < 2; ++b) {
        size_t ix = size_t(e) * 2 + b;
        wtab[e * 3 + b] = reinterpret_cast<uint64_t>(weight) + ix * branch_w;
        stab[e * 3 + b] = reinterpret_cast<uint64_t>(scale) + ix * branch_s;
    }
    auto d_wt = device_copy(wtab); auto d_st = device_copy(stab);

    const size_t a4_bytes = size_t(cap) * (H / 2), a4s_bytes = size_t(cap) * (H / 32);
    const size_t out_bytes = size_t(cap) * (I / 2), os_bytes = size_t(cap) * (I / 32);
    void *a4{}, *a4s{}, *out[2]{}, *os[2]{};
    CK(hipMalloc(&a4, a4_bytes)); CK(hipMalloc(&a4s, a4s_bytes));
    for (int i = 0; i < 2; ++i) {
        CK(hipMalloc(&out[i], out_bytes)); CK(hipMalloc(&os[i], os_bytes));
        CK(hipMemset(out[i], 0xa5, out_bytes)); CK(hipMemset(os[i], 0xa5, os_bytes));
    }
    const uint32_t grid = cap / 64 * ((I + 127) / 128);
    auto quant = [&](Obj& o) {
        uint32_t row_capacity = cap, experts = E, hidden = H, tokens = T;
        void* qa[] = {&a4, &a4s, &activation, &d_token, &d_meta, &row_capacity, &experts, &hidden, &tokens};
        CK(hipModuleLaunchKernel(o.quant, 1024, 1, 1, 256, 1, 1, 0, nullptr, qa, nullptr));
    };
    auto gemm = [&](Obj& o, int slot) {
        uint32_t inter = I, hidden = H, experts = E, act = 0; float beta = 0, linear = 0;
        void* ga[] = {&out[slot], &a4, &d_wt, &a4s, &d_st, &d_meta, &d_partidx, &os[slot],
                      &inter, &hidden, &experts, &act, &beta, &linear, &d_token};
        CK(hipModuleLaunchKernel(o.gemm, grid, 1, 1, 256, 1, 1, 32768, nullptr, ga, nullptr));
    };
    /* Sorted row r of one arm must equal row r (sorted) or row row_token[r] (gather) of the other. */
    const size_t a4_live = rows * (H / 2), a4s_live = rows * (H / 32);
    std::vector<uint8_t> qa0(a4_live), qa1(a4_live), qs0(a4s_live), qs1(a4s_live);
    CK(hipMemset(a4, 0x5a, a4_bytes)); CK(hipMemset(a4s, 0x5a, a4s_bytes));
    quant(cand); CK(hipDeviceSynchronize());
    CK(hipMemcpy(qa1.data(), a4, a4_live, hipMemcpyDeviceToHost));
    CK(hipMemcpy(qs1.data(), a4s, a4s_live, hipMemcpyDeviceToHost));
    CK(hipMemset(a4, 0xa5, a4_bytes)); CK(hipMemset(a4s, 0xa5, a4s_bytes));
    quant(base); CK(hipDeviceSynchronize());
    CK(hipMemcpy(qa0.data(), a4, a4_live, hipMemcpyDeviceToHost));
    CK(hipMemcpy(qs0.data(), a4s, a4s_live, hipMemcpyDeviceToHost));
    size_t qdiff = 0;
    for (size_t r = 0; r < rows; ++r) {
        if (row_token[r] == UNUSED) continue;
        const size_t r0 = base.gather ? row_token[r] : r, r1 = cand.gather ? row_token[r] : r;
        qdiff += memcmp(&qa0[r0 * H / 2], &qa1[r1 * H / 2], H / 2) != 0
              || memcmp(&qs0[r0 * H / 32], &qs1[r1 * H / 32], H / 32) != 0;
    }
    const bool qbad = qdiff != 0;
    gemm(base, 0);
    if (cand.gather != base.gather) { quant(cand); CK(hipDeviceSynchronize()); }
    gemm(cand, 1); CK(hipDeviceSynchronize());
    std::vector<uint8_t> o0(out_bytes), o1(out_bytes), s0(os_bytes), s1(os_bytes);
    CK(hipMemcpy(o0.data(), out[0], out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(o1.data(), out[1], out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s0.data(), os[0], os_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s1.data(), os[1], os_bytes, hipMemcpyDeviceToHost));
    size_t bad = 0, live = 0, nz = 0;
    for (size_t r = 0; r < rows; ++r) {
        if (row_partidx[r] == UNUSED) continue;
        ++live;
        bad += memcmp(&o0[r * I / 2], &o1[r * I / 2], I / 2) || memcmp(&s0[r * I / 32], &s1[r * I / 32], I / 32);
        for (uint32_t i = 0; i < I / 2; ++i) nz += o0[r * I / 2 + i] != 0;
    }

    /* The route GLM I=256 took before the token-gather object: plow_moe1_mxfp4_bk256 (op85 body,
     * 512 threads, grid 256). Different MFMA chain, so compared by dequantized rel-L2. */
    auto ship = [&](int slot) {
        uint32_t inter = I, hidden = H, experts = E, act = 0, zero = 0; float beta = 0, linear = 0;
        void* a[] = {&out[slot], &activation, &d_wt, &d_st, &d_meta, &d_token, &d_partidx, &os[slot],
                     &inter, &hidden, &experts, &act, &beta, &linear, &zero, &zero};
        CK(hipModuleLaunchKernel(base.bk256, 256, 1, 1, 512, 1, 1, 119808, nullptr, a, nullptr));
    };
    ship(0); CK(hipDeviceSynchronize());
    std::vector<uint8_t> so(out_bytes), ss(os_bytes);
    CK(hipMemcpy(so.data(), out[0], out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ss.data(), os[0], os_bytes, hipMemcpyDeviceToHost));
    auto deq = [](const std::vector<uint8_t>& q, const std::vector<uint8_t>& sc, size_t r, uint32_t i) {
        static const float lut[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const uint8_t b = q[r * I / 2 + i / 2], n = (i & 1) ? b >> 4 : b & 15;
        const float v = lut[n & 7] * std::ldexp(1.0f, int(sc[r * I / 32 + i / 32]) - 127);
        return (n & 8) ? -v : v;
    };
    double num = 0, den = 0; size_t sdiff = 0;
    for (size_t r = 0; r < rows; ++r) {
        if (row_partidx[r] == UNUSED) continue;
        sdiff += memcmp(&so[r * I / 2], &o1[r * I / 2], I / 2) != 0;
        for (uint32_t i = 0; i < I; ++i) {
            const double a = deq(so, ss, r, i), b = deq(o1, s1, r, i);
            num += (a - b) * (a - b); den += a * a;
        }
    }
    std::vector<float> tship;

    void* flush{}; constexpr size_t flush_bytes = 512u << 20;
    CK(hipMalloc(&flush, flush_bytes)); CK(hipMemset(flush, 0x5a, flush_bytes));
    hipEvent_t ev[3]; for (auto& e : ev) CK(hipEventCreate(&e));
    std::vector<float> tq, tqc, tb, tc;
    for (int it = 0; it < iters; ++it) {
        for (int arm = 0; arm < 2; ++arm) {
            const bool c = (arm ^ (it & 1)) != 0;
            flush_cache<<<4096, 256>>>((uint32_t*)flush, flush_bytes / 4, it * 2 + arm);
            CK(hipEventRecord(ev[0])); quant(c ? cand : base);
            CK(hipEventRecord(ev[1])); gemm(c ? cand : base, c);
            CK(hipEventRecord(ev[2])); CK(hipEventSynchronize(ev[2]));
            float q, g; CK(hipEventElapsedTime(&q, ev[0], ev[1])); CK(hipEventElapsedTime(&g, ev[1], ev[2]));
            (c ? tqc : tq).push_back(q);
            (c ? tc : tb).push_back(g);
        }
    }
    for (int it = 0; it < iters; ++it) {
        flush_cache<<<4096, 256>>>((uint32_t*)flush, flush_bytes / 4, 99 + it);
        CK(hipEventRecord(ev[0])); ship(0);
        CK(hipEventRecord(ev[1])); CK(hipEventSynchronize(ev[1]));
        float g; CK(hipEventElapsedTime(&g, ev[0], ev[1])); tship.push_back(g);
    }
    const double flop = 2.0 * live * H * 2.0 * I;
    const double wbytes = double(E) * 2 * (branch_w + branch_s);
    const double roof_us = std::max(wbytes / 6.2e12, flop / 9.2e15) * 1e6;
    std::printf("T=%u rows=%zu live=%zu tiles=%d grid=%u  oracle %s (%zu/%zu rows differ, nz=%zu)\n",
                T, rows, live, tilep[E], grid, bad ? "FAIL" : "BIT-EXACT", bad, live, nz);
    std::printf("  quant_sort   : %8.1f us  cand %8.1f us  speedup %.2fx  sorted A4 %s\n", median(tq) * 1e3,
                median(tqc) * 1e3, median(tq) / median(tqc), qbad ? "DIFFER" : "BIT-EXACT");
    std::printf("  base gemm    : %8.1f us  (%4.1f%% of %.1f us roof, %.0f TFLOP/s)\n", median(tb) * 1e3,
                100 * roof_us / (median(tb) * 1e3), roof_us, flop / median(tb) * 1e-9);
    std::printf("  cand gemm    : %8.1f us  (%4.1f%% of roof, %.0f TFLOP/s)  speedup %.2fx\n",
                median(tc) * 1e3, 100 * roof_us / (median(tc) * 1e3), flop / median(tc) * 1e-9,
                median(tb) / median(tc));
    std::printf("  stage1 total : %8.1f us  cand %8.1f us  speedup %.2fx\n", (median(tq) + median(tb)) * 1e3,
                (median(tqc) + median(tc)) * 1e3, (median(tq) + median(tb)) / (median(tqc) + median(tc)));
    std::printf("  bk256 route  : %8.1f us (in-model I=256 stage-1 before)  cand total speedup %.2fx  "
                "rows differing %zu/%zu, dequantized rel-L2 %.3e\n", median(tship) * 1e3,
                median(tship) / (median(tqc) + median(tc)), sdiff, live, std::sqrt(num / den));
    return bad || qbad ? 3 : 0;
}
