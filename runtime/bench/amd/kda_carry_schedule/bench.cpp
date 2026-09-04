#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
} } while (0)

extern "C" void k_carry_control();
extern "C" void k_carry_v8_wg256();
extern "C" void k_carry_v16_wg256();
extern "C" void k_carry_v32_wg512();
extern "C" void k_carry_v32_staged();
extern "C" void k_carry_tail_overlap();
extern "C" void k_carry_key_stage();
extern "C" void k_carry_key_stage_tail_overlap();

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
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

template<class T> static size_t mismatches(T* a, T* b, size_t n) {
    std::vector<T> ha(n), hb(n);
    CHECK(hipMemcpy(ha.data(), a, n * sizeof(T), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hb.data(), b, n * sizeof(T), hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i)
        bad += std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    return bad;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Arm {
    const char* name;
    const void* kernel;
    dim3 grid;
    dim3 block;
    size_t lds;
    uint16_t* out;
    float* state;
    std::vector<float> samples;
    bool enabled;
};

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192u;
    const unsigned H = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 12u;
    const unsigned nsample = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 21u;
    const bool run_v8 = argc > 4 && std::strtoul(argv[4], nullptr, 0);
    if (!T || !H || !nsample || (T & 63u)) return 2;

    constexpr unsigned D = 128u, V = 128u;
    constexpr size_t control_lds = 14336u;
    constexpr size_t v8_lds = 7168u;
    constexpr size_t v16_lds = 14336u;
    constexpr size_t v32_lds = 28672u;
    constexpr size_t staged_lds = 120320u;
    constexpr size_t key_stage_lds = control_lds + 2u * 64u * D * sizeof(uint16_t) +
                                     D * sizeof(float);
    const size_t qn = (size_t)T * H * D;
    const size_t vn = (size_t)T * H * V;
    const size_t an = (size_t)T * H * 64u;
    const size_t sn = (size_t)H * V * D;

    std::vector<uint16_t> q(qn), k(qn), w(qn), u(vn);
    std::vector<float> aqk(an), g(qn), state0(sn);
    for (size_t i = 0; i < qn; ++i) {
        q[i] = bf16(float(int((i * 17u) % 31u) - 15) * 0.003f);
        k[i] = bf16(float(int((i * 29u) % 37u) - 18) * 0.002f);
        w[i] = bf16(float(int((i * 43u) % 41u) - 20) * 0.0015f);
        const unsigned row = unsigned(i / ((size_t)H * D));
        g[i] = -0.0002f * float(1u + (row & 63u)) * float(1u + (i % 3u));
    }
    for (size_t i = 0; i < vn; ++i)
        u[i] = bf16(float(int((i * 47u) % 43u) - 21) * 0.001f);
    for (size_t i = 0; i < an; ++i)
        aqk[i] = float(int((i * 53u) % 47u) - 23) * 0.00003f;
    for (size_t i = 0; i < sn; ++i)
        state0[i] = float(int((i * 59u) % 53u) - 26) * 0.0001f;

    uint16_t* dq = upload(q);
    uint16_t* dk = upload(k);
    uint16_t* dw = upload(w);
    uint16_t* du = upload(u);
    float* da = upload(aqk);
    float* da_ref = upload(aqk);
    float* dg = upload(g);

    Arm arms[] = {
        {"control_v16_wg512", (const void*)k_carry_control, dim3(H * 8u), dim3(512),
         control_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"v8_wg256", (const void*)k_carry_v8_wg256, dim3(H * 16u), dim3(256),
         v8_lds, allocate<uint16_t>(vn), upload(state0), {}, run_v8},
        {"v16_wg256", (const void*)k_carry_v16_wg256, dim3(H * 8u), dim3(256),
         v16_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"v32_wg512", (const void*)k_carry_v32_wg512, dim3(H * 4u), dim3(512),
         v32_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"v32_staged_wg512", (const void*)k_carry_v32_staged, dim3(H * 4u), dim3(512),
         staged_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"tail_overlap_wg512", (const void*)k_carry_tail_overlap, dim3(H * 8u), dim3(512),
         control_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"key_stage_wg512", (const void*)k_carry_key_stage, dim3(H * 8u), dim3(512),
         key_stage_lds, allocate<uint16_t>(vn), upload(state0), {}, true},
        {"key_stage_tail_overlap_wg512", (const void*)k_carry_key_stage_tail_overlap,
         dim3(H * 8u), dim3(512), key_stage_lds, allocate<uint16_t>(vn),
         upload(state0), {}, true},
    };

    auto launch = [&](Arm& arm) {
        void* args[] = {&arm.out, &arm.state, &dq, &dk, &dw, &du, &da, &dg,
                        (void*)&T, (void*)&H};
        CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, arm.lds, nullptr));
    };

    for (Arm& arm : arms)
        if (arm.enabled) launch(arm);
    CHECK(hipDeviceSynchronize());
    for (unsigned i = 1; i < sizeof(arms) / sizeof(arms[0]); ++i) {
        if (!arms[i].enabled) continue;
        const size_t ob = mismatches(arms[0].out, arms[i].out, vn);
        const size_t sb = mismatches(arms[0].state, arms[i].state, sn);
        std::printf("oracle arm=%s output_mismatch=%zu/%zu state_mismatch=%zu/%zu\n",
                    arms[i].name, ob, vn, sb, sn);
        if (ob || sb) return 3;
    }
    const size_t ab = mismatches(da, da_ref, an);
    std::printf("oracle Aqk_mismatch=%zu/%zu\n", ab, an);
    if (ab) return 3;

    hipEvent_t begin, end;
    CHECK(hipEventCreate(&begin));
    CHECK(hipEventCreate(&end));
    std::vector<unsigned> active;
    for (unsigned i = 0; i < sizeof(arms) / sizeof(arms[0]); ++i)
        if (arms[i].enabled) active.push_back(i);
    const unsigned narms = active.size();
    for (unsigned rep = 0; rep < nsample + 3u; ++rep) {
        for (unsigned pos = 0; pos < narms; ++pos) {
            Arm& arm = arms[active[(rep + pos) % narms]];
            CHECK(hipEventRecord(begin));
            launch(arm);
            CHECK(hipEventRecord(end));
            CHECK(hipEventSynchronize(end));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, begin, end));
            if (rep >= 3u) arm.samples.push_back(ms);
        }
    }

    std::printf("shape T=%u H=%u D=%u V=%u BT=64 samples=%u\n", T, H, D, V, nsample);
    const float base = median(arms[0].samples);
    for (Arm& arm : arms) {
        if (!arm.enabled) continue;
        const float m = median(arm.samples);
        std::printf("samples_%s_ms=", arm.name);
        for (size_t i = 0; i < arm.samples.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", arm.samples[i]);
        std::printf("\nmedian_ms arm=%s time=%.6f delta=%.6f speedup=%.3fx\n",
                    arm.name, m, m - base, base / m);
    }
    return 0;
}
