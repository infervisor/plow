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
extern "C" void k_carry_control_timed();
extern "C" void k_carry_regstate();

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
    CHECK(hipMemset(p, 0, n * sizeof(T)));
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
};

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192u;
    const unsigned H = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 12u;
    const unsigned nsample = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 21u;
    const bool timers_on = argc > 4 && std::strtoul(argv[4], nullptr, 0);
    if (!T || !H || !nsample) return 2;

    constexpr unsigned D = 128u, V = 128u;
    constexpr size_t control_lds = 14336u;
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
    for (size_t i = 0; i < an; ++i) {
        const unsigned s = unsigned(i % 64u), m = unsigned((i / 64u) % 64u);
        aqk[i] = s <= m ? float(int((i * 53u) % 47u) - 23) * 0.00003f : 0.0f;
    }
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
         control_lds, allocate<uint16_t>(vn), upload(state0), {}},
        {"regstate_v16_wg256", (const void*)k_carry_regstate, dim3(H * 8u), dim3(256),
         0, allocate<uint16_t>(vn), upload(state0), {}},
    };
    constexpr unsigned narms = sizeof(arms) / sizeof(arms[0]);

    auto launch = [&](Arm& arm) {
        void* args[] = {&arm.out, &arm.state, &dq, &dk, &dw, &du, &da, &dg,
                        (void*)&T, (void*)&H};
        CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, arm.lds, nullptr));
    };

    for (Arm& arm : arms) launch(arm);
    CHECK(hipDeviceSynchronize());
    for (unsigned i = 1; i < narms; ++i) {
        const size_t ob = mismatches(arms[0].out, arms[i].out, vn);
        const size_t sb = mismatches(arms[0].state, arms[i].state, sn);
        std::printf("oracle arm=%s output_mismatch=%zu/%zu state_mismatch=%zu/%zu\n",
                    arms[i].name, ob, vn, sb, sn);
        if (ob || sb) return 3;
    }
    const size_t ab = mismatches(da, da_ref, an);
    std::printf("oracle Aqk_mismatch=%zu/%zu\n", ab, an);
    if (ab) return 3;

    if (timers_on) {
        const unsigned nblk = H * 8u;
        unsigned long long* dt = allocate<unsigned long long>((size_t)nblk * 8u);
        uint16_t* tout = allocate<uint16_t>(vn);
        float* tstate = upload(state0);
        void* args[] = {&tout, &tstate, &dq, &dk, &dw, &du, &da, &dg,
                        (void*)&T, (void*)&H, &dt};
        for (unsigned rep = 0; rep < 3u; ++rep)
            CHECK(hipLaunchKernel((const void*)k_carry_control_timed, dim3(nblk), dim3(512),
                                  args, control_lds, nullptr));
        CHECK(hipDeviceSynchronize());
        std::vector<unsigned long long> ht((size_t)nblk * 8u);
        CHECK(hipMemcpy(ht.data(), dt, ht.size() * sizeof(ht[0]), hipMemcpyDeviceToHost));
        const char* names[7] = {"p1_pred_vsm", "bar1", "p2_out", "bar2", "p3_keys_state",
                                "bar3", "total"};
        const unsigned n_chunks = (T + 63u) / 64u;
        double sum[7] = {0, 0, 0, 0, 0, 0, 0};
        for (unsigned b = 0; b < nblk; ++b)
            for (unsigned i = 0; i < 7u; ++i) sum[i] += double(ht[(size_t)b * 8u + i]);
        std::printf("timers wave0 s_memtime cycles per chunk, mean over %u workgroups, %u chunks\n",
                    nblk, n_chunks);
        for (unsigned i = 0; i < 7u; ++i)
            std::printf("timer %-14s cycles_per_chunk=%.0f share=%.3f\n", names[i],
                        sum[i] / nblk / n_chunks, sum[i] / sum[6]);
    }

    hipEvent_t begin, end;
    CHECK(hipEventCreate(&begin));
    CHECK(hipEventCreate(&end));
    for (unsigned rep = 0; rep < nsample + 3u; ++rep) {
        for (unsigned pos = 0; pos < narms; ++pos) {
            Arm& arm = arms[(rep + pos) % narms];
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
        const float m = median(arm.samples);
        std::printf("samples_%s_ms=", arm.name);
        for (size_t i = 0; i < arm.samples.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", arm.samples[i]);
        std::printf("\nmedian_ms arm=%s time=%.6f delta=%.6f speedup=%.3fx\n",
                    arm.name, m, m - base, base / m);
    }
    return 0;
}
