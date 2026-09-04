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

extern "C" void k_wu_control();
extern "C" void k_key_reference();
extern "C" void k_wu_lean();
extern "C" void k_wu_lean_keys();
extern "C" void k_wu_key_factor_object();

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

/* MODE 0: structured screen inputs. MODE 1: LCG uniform. MODE 2: adversarial — NaN/Inf/denormal
 * /RNE-tie sprinkles in every operand, gates spanning exp2 overflow and underflow, zero rows. */
static uint32_t lcg_state = 0x9e3779b9u;
static float lcg() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return float(lcg_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}
static uint16_t adv_bf16(float base) {
    const uint32_t r = (lcg_state = lcg_state * 1664525u + 1013904223u) >> 8;
    switch (r % 61u) {
        case 0: return 0x7fc0u;
        case 1: return 0xffc1u;
        case 2: return 0x7f80u;
        case 3: return 0xff80u;
        case 4: return 0x0001u;
        case 5: return 0x8000u;
        case 6: return 0x7f7fu;
        default: return bf16(base * ((r & 1u) ? 1.0f : 64.0f));
    }
}
static float adv_gate(float base) {
    const uint32_t r = (lcg_state = lcg_state * 1664525u + 1013904223u) >> 8;
    switch (r % 53u) {
        case 0: return -200.0f;
        case 1: return 150.0f;
        case 2: return 0.0f;
        case 3: return -1.0e-40f;
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
        case 4: return 1.00390625f;
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
    for (size_t i = 0; i < n; ++i) bad += std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    return bad;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

enum Kind { CONTROL, LEAN, KEYFACTOR };

struct Arm {
    const char* name;
    const void* kernel;
    Kind kind;
    bool keys;
    dim3 grid;
    dim3 block;
    uint16_t *w, *u, *q, *kh, *kl;
    std::vector<float> samples;
};

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192u;
    const unsigned H = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 12u;
    const unsigned nsample = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 21u;
    const unsigned mode = argc > 4 ? std::strtoul(argv[4], nullptr, 0) : 0u;
    if (!T || !H || !nsample || mode > 2u) return 2;

    constexpr unsigned D = 128u;
    const size_t qn = (size_t)T * H * D;
    const size_t an = (size_t)T * H * 64u;
    const size_t bn = (size_t)T * H;
    const float scale = 1.0f / 128.0f;

    std::vector<uint16_t> q(qn), k(qn), v(qn);
    std::vector<float> ainv(an), g(qn), beta(bn);
    for (size_t i = 0; i < qn; ++i) {
        q[i] = bf16(float(int((i * 17u) % 31u) - 15) * 0.003f);
        k[i] = bf16(float(int((i * 29u) % 37u) - 18) * 0.002f);
        v[i] = bf16(float(int((i * 47u) % 43u) - 21) * 0.001f);
        const unsigned row = unsigned(i / ((size_t)H * D));
        g[i] = -0.0002f * float(1u + (row & 63u)) * float(1u + (i % 3u));
    }
    for (size_t i = 0; i < an; ++i) {
        const unsigned s = unsigned(i % 64u), m = unsigned((i / 64u) % 64u);
        ainv[i] = s < m ? float(int((i * 61u) % 43u) - 21) * 0.00002f : (s == m ? 1.0f : 0.0f);
    }
    for (size_t i = 0; i < bn; ++i) beta[i] = 0.25f + 0.001f * float(i % 37u);
    if (mode == 1u) {
        for (size_t i = 0; i < qn; ++i) {
            q[i] = bf16(lcg()); k[i] = bf16(lcg()); v[i] = bf16(lcg());
            g[i] = 0.25f * (lcg() - 1.0f);
        }
        for (size_t i = 0; i < an; ++i) {
            const unsigned s = unsigned(i % 64u), m = unsigned((i / 64u) % 64u);
            ainv[i] = s < m ? 0.05f * lcg() : (s == m ? 1.0f : 0.0f);
        }
        for (size_t i = 0; i < bn; ++i) beta[i] = 0.5f + 0.5f * lcg();
    } else if (mode == 2u) {
        for (size_t i = 0; i < qn; ++i) {
            q[i] = adv_bf16(lcg()); k[i] = adv_bf16(lcg()); v[i] = adv_bf16(lcg());
            g[i] = adv_gate(4.0f * (lcg() - 1.0f));
        }
        for (size_t i = 0; i < an; ++i) {
            const unsigned s = unsigned(i % 64u), m = unsigned((i / 64u) % 64u);
            ainv[i] = s <= m ? adv_f32(lcg()) : 0.0f;
        }
        for (size_t i = 0; i < bn; ++i) beta[i] = adv_f32(lcg());
        for (size_t r = 5; r < T; r += 97)
            for (size_t j = 0; j < (size_t)H * D; ++j) { k[r * H * D + j] = 0; v[r * H * D + j] = 0; }
    }

    uint16_t* dk = upload(k);
    uint16_t* dv = upload(v);
    float* da = upload(ainv);
    float* dg = upload(g);
    float* db = upload(beta);
    uint16_t* dq0 = upload(q);
    uint16_t* ref_kh = allocate<uint16_t>(qn);
    uint16_t* ref_kl = allocate<uint16_t>(qn);
    {
        void* args[] = {&ref_kh, &ref_kl, &dk, &dg, (void*)&T, (void*)&H};
        CHECK(hipLaunchKernel((const void*)k_key_reference, dim3(1024), dim3(256), args, 0,
                              nullptr));
    }

    const unsigned n_items = ((T + 63u) / 64u) * H;
    auto mk = [&](const char* name, const void* kernel, Kind kind, bool keys, unsigned grid,
                  unsigned block) {
        return Arm{name, kernel, kind, keys, dim3(grid), dim3(block), allocate<uint16_t>(qn),
                   allocate<uint16_t>(qn), upload(q), allocate<uint16_t>(qn),
                   allocate<uint16_t>(qn), {}};
    };
    std::vector<Arm> arms;
    arms.push_back(mk("control_wg512_g256", (const void*)k_wu_control, CONTROL, false, 256, 512));
    arms.push_back(mk("key_factor_object_wg256_g256", (const void*)k_wu_key_factor_object,
                      KEYFACTOR, true, 256, 256));
    arms.push_back(mk("lean_g256", (const void*)k_wu_lean, LEAN, false, 256, 256));
    arms.push_back(mk("lean_g512", (const void*)k_wu_lean, LEAN, false, 512, 256));
    arms.push_back(mk("lean_g768", (const void*)k_wu_lean, LEAN, false, 768, 256));
    arms.push_back(mk("lean_gitems", (const void*)k_wu_lean, LEAN, false, n_items, 256));
    arms.push_back(mk("lean_keys_g768", (const void*)k_wu_lean_keys, LEAN, true, 768, 256));
    arms.push_back(mk("lean_keys_gitems", (const void*)k_wu_lean_keys, LEAN, true, n_items, 256));

    /* q is pre-scaled in place; every timed launch restarts from the same q. */
    auto reset_q = [&](Arm& arm) {
        CHECK(hipMemcpyAsync(arm.q, dq0, qn * 2, hipMemcpyDeviceToDevice, nullptr));
    };
    auto launch = [&](Arm& arm) {
        if (arm.kind == CONTROL) {
            void* args[] = {&arm.w, &arm.u, &arm.q, &da, &dk, &dv, &dg, &db, (void*)&T,
                            (void*)&H, (void*)&scale};
            CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, 0, nullptr));
        } else if (arm.kind == KEYFACTOR) {
            void* args[] = {&arm.w, &arm.u, &arm.kh, &arm.kl, &arm.q, &da, &dk, &dv, &dg, &db,
                            (void*)&T, (void*)&H, (void*)&scale};
            CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, 0, nullptr));
        } else {
            void* args[] = {&arm.w, &arm.u, &arm.q, &arm.kh, &arm.kl, &da, &dk, &dv, &dg, &db,
                            (void*)&T, (void*)&H, (void*)&scale};
            CHECK(hipLaunchKernel(arm.kernel, arm.grid, arm.block, args, 0, nullptr));
        }
    };

    for (Arm& arm : arms) { reset_q(arm); launch(arm); }
    CHECK(hipDeviceSynchronize());
    bool ok = true;
    for (size_t i = 1; i < arms.size(); ++i) {
        const size_t wb = mismatches(arms[0].w, arms[i].w, qn);
        const size_t ub = mismatches(arms[0].u, arms[i].u, qn);
        const size_t qb = mismatches(arms[0].q, arms[i].q, qn);
        size_t hb = 0, lb = 0;
        if (arms[i].keys) {
            hb = mismatches(ref_kh, arms[i].kh, qn);
            lb = mismatches(ref_kl, arms[i].kl, qn);
        }
        std::printf("oracle arm=%s W_mismatch=%zu/%zu U_mismatch=%zu/%zu q_mismatch=%zu/%zu "
                    "key_hi_mismatch=%zu key_lo_mismatch=%zu\n",
                    arms[i].name, wb, qn, ub, qn, qb, qn, hb, lb);
        ok = ok && !wb && !ub && !qb && !hb && !lb;
    }
    if (!ok) return 3;

    hipEvent_t begin, end;
    CHECK(hipEventCreate(&begin));
    CHECK(hipEventCreate(&end));
    const unsigned narms = (unsigned)arms.size();
    for (unsigned rep = 0; rep < nsample + 3u; ++rep) {
        for (unsigned pos = 0; pos < narms; ++pos) {
            Arm& arm = arms[(rep + pos) % narms];
            reset_q(arm);
            CHECK(hipEventRecord(begin));
            launch(arm);
            CHECK(hipEventRecord(end));
            CHECK(hipEventSynchronize(end));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, begin, end));
            if (rep >= 3u) arm.samples.push_back(ms);
        }
    }

    std::printf("shape T=%u H=%u D=128 V=128 BT=64 items=%u samples=%u mode=%u\n", T, H,
                n_items, nsample, mode);
    const float base = median(arms[0].samples);
    for (Arm& arm : arms) {
        const float m = median(arm.samples);
        std::printf("median_ms arm=%s time=%.6f delta=%.6f speedup=%.3fx\n", arm.name, m,
                    m - base, base / m);
    }
    return 0;
}
