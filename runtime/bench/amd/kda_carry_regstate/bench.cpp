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
extern "C" void k_carry_regstate_timed();
extern "C" void k_carry_regstate_hwcvt();
extern "C" void k_key_precompute();
extern "C" void k_carry_regstate_keyfeed();
extern "C" void k_carry_regstate_keyfeed_timed();

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

/* MODE 0: the structured screen inputs. MODE 1: LCG uniform in [-1, 1) (g in [-0.5, 0)).
 * MODE 2: adversarial — NaN/Inf/denormal/RNE-tie sprinkles in every bf16 operand, gates spanning
 * exp2 overflow and underflow, zero rows. */
static uint32_t lcg_state = 0x9e3779b9u;
static float lcg() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return float(lcg_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}
static uint16_t adv_bf16(float base) {
    const uint32_t r = (lcg_state = lcg_state * 1664525u + 1013904223u) >> 8;
    switch (r % 61u) {
        case 0: return 0x7fc0u;             /* qNaN */
        case 1: return 0xffc1u;             /* -qNaN, payload */
        case 2: return 0x7f80u;             /* +Inf */
        case 3: return 0xff80u;             /* -Inf */
        case 4: return 0x0001u;             /* bf16 denormal */
        case 5: return 0x8000u;             /* -0 */
        case 6: return 0x7f7fu;             /* bf16 max */
        default: return bf16(base * ((r & 1u) ? 1.0f : 64.0f));
    }
}
static float adv_gate(float base) {
    const uint32_t r = (lcg_state = lcg_state * 1664525u + 1013904223u) >> 8;
    switch (r % 53u) {
        case 0: return -200.0f;             /* exp2 underflow */
        case 1: return 150.0f;              /* exp2 overflow */
        case 2: return 0.0f;
        case 3: return -1.0e-40f;           /* f32 denormal */
        default: return base;
    }
}
static float adv_f32(float base) {
    const uint32_t r = (lcg_state = lcg_state * 1664525u + 1013904223u) >> 8;
    switch (r % 47u) {
        case 0: return __builtin_nanf("");
        case 1: return __builtin_inff();
        case 2: return 1.0e-39f;
        case 3: return 3.0e38f;
        case 4: return 1.00390625f;         /* bf16 RNE tie */
        default: return base;
    }
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
    const unsigned mode = argc > 5 ? std::strtoul(argv[5], nullptr, 0) : 0u;
    if (!T || !H || !nsample || mode > 2u) return 2;

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
    if (mode == 1u) {
        for (size_t i = 0; i < qn; ++i) {
            q[i] = bf16(lcg()); k[i] = bf16(lcg()); w[i] = bf16(lcg());
            g[i] = 0.25f * (lcg() - 1.0f);
        }
        for (size_t i = 0; i < vn; ++i) u[i] = bf16(lcg());
        for (size_t i = 0; i < an; ++i)
            aqk[i] = (i % 64u) <= ((i / 64u) % 64u) ? 0.01f * lcg() : 0.0f;
        for (size_t i = 0; i < sn; ++i) state0[i] = 0.1f * lcg();
    } else if (mode == 2u) {
        for (size_t i = 0; i < qn; ++i) {
            q[i] = adv_bf16(lcg()); k[i] = adv_bf16(lcg()); w[i] = adv_bf16(lcg());
            g[i] = adv_gate(4.0f * (lcg() - 1.0f));
        }
        for (size_t i = 0; i < vn; ++i) u[i] = adv_bf16(lcg());
        for (size_t i = 0; i < an; ++i)
            aqk[i] = (i % 64u) <= ((i / 64u) % 64u) ? adv_f32(lcg()) : 0.0f;
        for (size_t i = 0; i < sn; ++i) state0[i] = adv_f32(lcg());
        for (size_t r = 5; r < T; r += 97) /* zero rows */
            for (size_t j = 0; j < (size_t)H * D; ++j) { k[r * H * D + j] = 0; w[r * H * D + j] = 0; }
    }

    uint16_t* dq = upload(q);
    uint16_t* dk = upload(k);
    uint16_t* dw = upload(w);
    uint16_t* du = upload(u);
    float* da = upload(aqk);
    float* da_ref = upload(aqk);
    float* dg = upload(g);
    uint16_t* dkh = allocate<uint16_t>(qn);
    uint16_t* dkl = allocate<uint16_t>(qn);
    {
        void* args[] = {&dkh, &dkl, &dk, &dg, (void*)&T, (void*)&H};
        CHECK(hipLaunchKernel((const void*)k_key_precompute, dim3(1024), dim3(256), args, 0,
                              nullptr));
        CHECK(hipDeviceSynchronize());
    }

    Arm arms[] = {
        {"control_v16_wg512", (const void*)k_carry_control, dim3(H * 8u), dim3(512),
         control_lds, allocate<uint16_t>(vn), upload(state0), {}},
        {"regstate_v16_wg256", (const void*)k_carry_regstate, dim3(H * 8u), dim3(256),
         0, allocate<uint16_t>(vn), upload(state0), {}},
        {"regstate_hwcvt_v16_wg256", (const void*)k_carry_regstate_hwcvt, dim3(H * 8u),
         dim3(256), 0, allocate<uint16_t>(vn), upload(state0), {}},
        {"regstate_keyfeed_v16_wg256", (const void*)k_carry_regstate_keyfeed, dim3(H * 8u),
         dim3(256), 0, allocate<uint16_t>(vn), upload(state0), {}},
    };
    constexpr unsigned narms = sizeof(arms) / sizeof(arms[0]);

    auto launch = [&](Arm& arm) {
        void* args[] = {&arm.out, &arm.state, &dq, &dk, &dw, &du, &da, &dg,
                        (void*)&T, (void*)&H, &dkh, &dkl};
        CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, arm.lds, nullptr));
    };

    for (Arm& arm : arms) launch(arm);
    CHECK(hipDeviceSynchronize());
    for (unsigned i = 1; i < narms; ++i) {
        const size_t ob = mismatches(arms[0].out, arms[i].out, vn);
        const size_t sb = mismatches(arms[0].state, arms[i].state, sn);
        std::printf("oracle arm=%s output_mismatch=%zu/%zu state_mismatch=%zu/%zu\n",
                    arms[i].name, ob, vn, sb, sn);
        if (sb && std::getenv("KDA_DEBUG")) {
            std::vector<float> a(sn), b(sn);
            CHECK(hipMemcpy(a.data(), arms[0].state, sn * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), arms[i].state, sn * 4, hipMemcpyDeviceToHost));
            unsigned byd[128] = {0}, byv[128] = {0};
            for (size_t j = 0; j < sn; ++j)
                if (std::memcmp(&a[j], &b[j], 4)) { byd[j % 128]++; byv[(j / 128) % 128]++; }
            std::printf("bad_by_d:"); for (unsigned d = 0; d < 128; ++d) std::printf(" %u", byd[d]);
            std::printf("\nbad_by_v:"); for (unsigned v = 0; v < 128; ++v) std::printf(" %u", byv[v]);
            std::printf("\n");
            for (size_t j = 0, n = 0; j < sn && n < 6; ++j)
                if (std::memcmp(&a[j], &b[j], 4)) { std::printf("  [%zu] h=%zu v=%zu d=%zu ctl=%.8g cand=%.8g\n", j, j / 16384, (j / 128) % 128, j % 128, a[j], b[j]); ++n; }
        }
        if (ob || sb) return 3;
    }
    const size_t ab = mismatches(da, da_ref, an);
    std::printf("oracle Aqk_mismatch=%zu/%zu\n", ab, an);
    if (ab) return 3;

    if (timers_on) {
        const unsigned nblk = H * 8u;
        const unsigned n_chunks = (T + 63u) / 64u;
        struct Timed { const char* name; const void* kernel; unsigned block; size_t lds;
                       const char* phases[7]; };
        const Timed timed[] = {
            {"control", (const void*)k_carry_control_timed, 512, control_lds,
             {"p1_pred_vsm", "bar1", "p2_out", "bar2", "p3_keys_state", "bar3", "total"}},
            {"regstate", (const void*)k_carry_regstate_timed, 256, 0,
             {"p1_pred_vsm", "loads+bar1", "p2_out", "p3_upd_state", "keys+loads", "bar2",
              "total"}},
            {"keyfeed", (const void*)k_carry_regstate_keyfeed_timed, 256, 0,
             {"p1_pred_vsm", "loads+bar1", "p2_out", "p3_upd_state", "keys+loads", "bar2",
              "total"}},
        };
        for (const Timed& tk : timed) {
            unsigned long long* dt = allocate<unsigned long long>((size_t)nblk * 8u);
            uint16_t* tout = allocate<uint16_t>(vn);
            float* tstate = upload(state0);
            void* args[] = {&tout, &tstate, &dq, &dk, &dw, &du, &da, &dg,
                            (void*)&T, (void*)&H, &dt, &dkh, &dkl};
            for (unsigned rep = 0; rep < 3u; ++rep)
                CHECK(hipLaunchKernel(tk.kernel, dim3(nblk), dim3(tk.block), args, tk.lds,
                                      nullptr));
            CHECK(hipDeviceSynchronize());
            std::vector<unsigned long long> ht((size_t)nblk * 8u);
            CHECK(hipMemcpy(ht.data(), dt, ht.size() * sizeof(ht[0]), hipMemcpyDeviceToHost));
            double sum[7] = {0, 0, 0, 0, 0, 0, 0};
            for (unsigned b = 0; b < nblk; ++b)
                for (unsigned i = 0; i < 7u; ++i) sum[i] += double(ht[(size_t)b * 8u + i]);
            std::printf("timers %s wave0 s_memtime cycles per chunk, mean over %u workgroups, "
                        "%u chunks\n", tk.name, nblk, n_chunks);
            for (unsigned i = 0; i < 7u; ++i)
                std::printf("timer %s %-14s cycles_per_chunk=%.0f share=%.3f\n", tk.name,
                            tk.phases[i], sum[i] / nblk / n_chunks, sum[i] / sum[6]);
        }
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

    std::printf("shape T=%u H=%u D=%u V=%u BT=64 samples=%u mode=%u\n", T, H, D, V, nsample,
                mode);
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
