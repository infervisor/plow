#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
} } while (0)

extern "C" void k_intra_control();
extern "C" void k_intra_wave_items();
extern "C" void k_wu_qpre();
extern "C" void k_carry_qpre();

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
    return uint16_t((u + bias) >> 16);
}

template<class T> static T* upload(const std::vector<T>& h) {
    T* p;
    CHECK(hipMalloc(&p, h.size() * sizeof(T)));
    CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

template<class T> static T* allocate(size_t n) {
    T* p;
    CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

template<class T> static size_t mismatches(T* d0, T* d1, size_t n) {
    std::vector<T> h0(n), h1(n);
    CHECK(hipMemcpy(h0.data(), d0, n * sizeof(T), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h1.data(), d1, n * sizeof(T), hipMemcpyDeviceToHost));
    size_t mismatch = 0;
    for (size_t i = 0; i < n; ++i)
        mismatch += std::memcmp(&h0[i], &h1[i], sizeof(T)) != 0;
    return mismatch;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const unsigned t = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192u;
    const unsigned heads = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 12u;
    const unsigned samples = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 21u;
    if (t < 512u || heads == 0u || (t & 63u) != 0u || samples == 0u) return 2;

    constexpr unsigned d = 128u, vdim = 128u;
    const size_t qn = (size_t)t * heads * d;
    const size_t an = (size_t)t * heads * 64u;
    const size_t sn = (size_t)heads * vdim * d;
    const float scale = 1.0f / std::sqrt(float(d));
    std::vector<uint16_t> q(qn), k(qn), v(qn);
    std::vector<float> g(qn), beta((size_t)t * heads), state(sn);
    for (size_t i = 0; i < qn; ++i) {
        q[i] = bf16(float(int((i * 17u) % 31u) - 15) * 0.003f);
        k[i] = bf16(float(int((i * 29u) % 37u) - 18) * 0.002f);
        v[i] = bf16(float(int((i * 47u) % 43u) - 21) * 0.001f);
        const unsigned row = unsigned(i / ((size_t)heads * d));
        g[i] = -0.0002f * float(1u + (row & 63u)) * float(1u + (i % 3u));
    }
    for (size_t i = 0; i < beta.size(); ++i)
        beta[i] = 0.25f + 0.001f * float(i % 37u);
    for (size_t i = 0; i < state.size(); ++i)
        state[i] = float(int((i * 59u) % 53u) - 26) * 0.0001f;

    uint16_t *dq0 = upload(q), *dq1 = upload(q), *dk = upload(k), *dv = upload(v);
    float *dg = upload(g), *db = upload(beta), *ds0 = upload(state), *ds1 = upload(state);
    float *daq0 = allocate<float>(an), *daq1 = allocate<float>(an);
    float *dai0 = allocate<float>(an), *dai1 = allocate<float>(an);
    uint16_t *dw0 = allocate<uint16_t>(qn), *dw1 = allocate<uint16_t>(qn);
    uint16_t *du0 = allocate<uint16_t>(qn), *du1 = allocate<uint16_t>(qn);
    uint16_t *do0 = allocate<uint16_t>(qn), *do1 = allocate<uint16_t>(qn);

    const dim3 block(512), grid(256), carry_grid(heads * 8u);
    void* i0_args[] = {&daq0, &dai0, &dq0, &dk, &dg, &db,
                       (void*)&t, (void*)&heads, (void*)&scale};
    void* i1_args[] = {&daq1, &dai1, &dq1, &dk, &dg, &db,
                       (void*)&t, (void*)&heads, (void*)&scale};
    CHECK(hipLaunchKernel((const void*)k_intra_control, grid, block, i0_args, 0, nullptr));
    CHECK(hipLaunchKernel((const void*)k_intra_wave_items, grid, block, i1_args, 0, nullptr));
    CHECK(hipDeviceSynchronize());

    const size_t aqk_mismatch = mismatches(daq0, daq1, an);
    const size_t ainv_mismatch = mismatches(dai0, dai1, an);

    void* w0_args[] = {&dw0, &du0, &dq0, &dai0, &dk, &dv, &dg, &db,
                       (void*)&t, (void*)&heads, (void*)&scale};
    void* w1_args[] = {&dw1, &du1, &dq1, &dai1, &dk, &dv, &dg, &db,
                       (void*)&t, (void*)&heads, (void*)&scale};
    CHECK(hipLaunchKernel((const void*)k_wu_qpre, grid, block, w0_args, 0, nullptr));
    CHECK(hipLaunchKernel((const void*)k_wu_qpre, grid, block, w1_args, 0, nullptr));
    void* c0_args[] = {&do0, &ds0, &dq0, &dk, &dw0, &du0, &daq0, &dg,
                       (void*)&t, (void*)&heads, (void*)&scale};
    void* c1_args[] = {&do1, &ds1, &dq1, &dk, &dw1, &du1, &daq1, &dg,
                       (void*)&t, (void*)&heads, (void*)&scale};
    CHECK(hipLaunchKernel((const void*)k_carry_qpre, carry_grid, block, c0_args, 0, nullptr));
    CHECK(hipLaunchKernel((const void*)k_carry_qpre, carry_grid, block, c1_args, 0, nullptr));
    CHECK(hipDeviceSynchronize());

    const size_t q_mismatch = mismatches(dq0, dq1, qn);
    const size_t w_mismatch = mismatches(dw0, dw1, qn);
    const size_t u_mismatch = mismatches(du0, du1, qn);
    const size_t out_mismatch = mismatches(do0, do1, qn);
    const size_t state_mismatch = mismatches(ds0, ds1, sn);
    std::printf("shape T=%u H=%u D=128 V=128 chunks=%u grid=%u waves=8\n",
                t, heads, t / 64u, grid.x);
    std::printf("oracle Aqk=%zu/%zu Ainv=%zu/%zu q=%zu/%zu W=%zu/%zu U=%zu/%zu "
                "output=%zu/%zu state=%zu/%zu\n",
                aqk_mismatch, an, ainv_mismatch, an, q_mismatch, qn,
                w_mismatch, qn, u_mismatch, qn, out_mismatch, qn, state_mismatch, sn);
    if (aqk_mismatch || ainv_mismatch || q_mismatch || w_mismatch || u_mismatch ||
        out_mismatch || state_mismatch) return 3;

    hipEvent_t begin, end;
    CHECK(hipEventCreate(&begin));
    CHECK(hipEventCreate(&end));
    auto time = [&](const char* name, const void* kernel, void** args) {
        std::vector<float> times;
        for (unsigned i = 0; i < samples + 3u; ++i) {
            CHECK(hipEventRecord(begin));
            CHECK(hipLaunchKernel(kernel, grid, block, args, 0, nullptr));
            CHECK(hipEventRecord(end));
            CHECK(hipEventSynchronize(end));
            float ms;
            CHECK(hipEventElapsedTime(&ms, begin, end));
            if (i >= 3u) times.push_back(ms);
        }
        std::printf("samples_%s_ms=", name);
        for (size_t i = 0; i < times.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", times[i]);
        std::printf("\n");
        return median(times);
    };
    const float control = time("control", (const void*)k_intra_control, i0_args);
    const float candidate = time("wave_items", (const void*)k_intra_wave_items, i1_args);
    std::printf("median_ms control=%.6f wave_items=%.6f delta=%.6f speedup=%.3fx samples=%u\n",
                control, candidate, candidate - control, control / candidate, samples);
    return 0;
}
