#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
} } while (0)

static hipFunction_t k_carry_keyfeed, k_key_precompute, k_seg, k_combine, k_fix;

static void launch(hipFunction_t f, unsigned grid, void** args) {
    CHECK(hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
}

struct SplitArgs {
    void* out; void* state; const void* q; const void* k; const void* W; const void* U;
    const void* Aqk; const void* g; const void* key_hi; const void* key_lo; void* scratch;
    unsigned t, heads, nseg, seg_len;
};

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
static float f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

template<class T> static T* upload(const std::vector<T>& h) {
    T* p;
    CHECK(hipMalloc(&p, h.size() * sizeof(T)));
    CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template<class T> static std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

constexpr unsigned D = 128, V = 128, BT = 64;

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192u;
    const unsigned H = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 8u;
    const unsigned nsample = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 11u;
    const bool ref_on = argc > 4 ? std::strtoul(argv[4], nullptr, 0) : 1u;
    const char* co = argc > 5 ? argv[5] : "kernel.co";
    const char* split_co = argc > 6 ? argv[6] : "split.co";
    hipModule_t mod, smod;
    CHECK(hipModuleLoad(&mod, co));
    CHECK(hipModuleLoad(&smod, split_co));
    CHECK(hipModuleGetFunction(&k_carry_keyfeed, mod, "k_carry_keyfeed"));
    CHECK(hipModuleGetFunction(&k_key_precompute, mod, "k_key_precompute"));
    CHECK(hipModuleGetFunction(&k_seg, smod, "plow_kda_chunk_carry_split_seg_gfx950"));
    CHECK(hipModuleGetFunction(&k_combine, smod, "plow_kda_chunk_carry_split_combine_gfx950"));
    CHECK(hipModuleGetFunction(&k_fix, smod, "plow_kda_chunk_carry_split_fix_gfx950"));
    const size_t qn = (size_t)T * H * D, an = (size_t)T * H * BT, sn = (size_t)H * V * D;

    /* KDA-consistent factors (log2 gates, chunk-local prefixes, q pre-scaled by D^-1/2 exp2(g)). */
    std::mt19937 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<double> qr(qn), kr(qn), vr(qn), gr(qn), br((size_t)T * H);
    for (size_t row = 0; row < (size_t)T * H; ++row) {
        double nq = 0, nk = 0;
        for (unsigned d = 0; d < D; ++d) {
            qr[row * D + d] = nd(rng); kr[row * D + d] = nd(rng); vr[row * D + d] = nd(rng);
            nq += qr[row * D + d] * qr[row * D + d]; nk += kr[row * D + d] * kr[row * D + d];
        }
        for (unsigned d = 0; d < D; ++d) { qr[row * D + d] /= std::sqrt(nq); kr[row * D + d] /= std::sqrt(nk); }
        br[row] = 1.0 / (1.0 + std::exp(-nd(rng)));
    }
    for (unsigned h = 0; h < H; ++h)
        for (unsigned d = 0; d < D; ++d) {
            const double rate = 0.002 + 0.2 * ud(rng) * ud(rng);
            double acc = 0;
            for (unsigned t = 0; t < T; ++t) {
                if (t % BT == 0) acc = 0;
                acc -= rate * (0.5 + ud(rng));
                gr[((size_t)t * H + h) * D + d] = acc;
            }
        }
    const double scale = 1.0 / std::sqrt(double(D));
    std::vector<uint16_t> q(qn), k(qn), w(qn), u(qn);
    std::vector<float> aqk(an), g(qn), state0(sn);
    for (size_t i = 0; i < qn; ++i) {
        g[i] = float(gr[i]);
        k[i] = bf16(float(kr[i]));
        q[i] = bf16(float(qr[i] * scale * std::exp2(gr[i])));
    }
    for (unsigned h = 0; h < H; ++h)
        for (unsigned c0 = 0; c0 < T; c0 += BT) {
            const unsigned n = std::min(BT, T - c0);
            auto ix = [&](unsigned i, unsigned d) { return ((size_t)(c0 + i) * H + h) * D + d; };
            std::vector<double> A(n * n, 0.0);
            for (unsigned i = 0; i < n; ++i)
                for (unsigned j = 0; j < n; ++j) {
                    double kk = 0, qk = 0;
                    for (unsigned d = 0; d < D; ++d) {
                        const double e = std::exp2(gr[ix(i, d)] - gr[ix(j, d)]);
                        if (j < i) kk += f32(k[ix(i, d)]) * f32(k[ix(j, d)]) * e;
                        if (j <= i) qk += qr[ix(i, d)] * f32(k[ix(j, d)]) * e;
                    }
                    if (j < i) A[i * n + j] = br[(size_t)(c0 + i) * H + h] * kk;
                    aqk[((size_t)(c0 + i) * H + h) * BT + j] = j <= i ? float(qk * scale) : 0.0f;
                }
            /* W = (I + A)^-1 (beta k exp2 g), U = (I + A)^-1 (beta v), forward substitution. */
            std::vector<double> Wd(n * D), Ud(n * V);
            for (unsigned i = 0; i < n; ++i) {
                const double b = br[(size_t)(c0 + i) * H + h];
                for (unsigned d = 0; d < D; ++d) {
                    double wv = b * f32(k[ix(i, d)]) * std::exp2(gr[ix(i, d)]);
                    double uv = b * vr[ix(i, d)];
                    for (unsigned j = 0; j < i; ++j) { wv -= A[i * n + j] * Wd[j * D + d]; uv -= A[i * n + j] * Ud[j * V + d]; }
                    Wd[i * D + d] = wv; Ud[i * V + d] = uv;
                }
            }
            for (unsigned i = 0; i < n; ++i)
                for (unsigned d = 0; d < D; ++d) { w[ix(i, d)] = bf16(float(Wd[i * D + d])); u[ix(i, d)] = bf16(float(Ud[i * V + d])); }
        }
    for (size_t i = 0; i < sn; ++i) state0[i] = float(0.05 * nd(rng));

    uint16_t *dq = upload(q), *dk = upload(k), *dw = upload(w), *du = upload(u);
    float *da = upload(aqk), *dg = upload(g);
    uint16_t *dkh, *dkl;
    CHECK(hipMalloc(&dkh, qn * 2)); CHECK(hipMalloc(&dkl, qn * 2));
    {
        void* args[] = {&dkh, &dkl, &dk, &dg, (void*)&T, (void*)&H};
        launch(k_key_precompute, 1024, args);
        CHECK(hipDeviceSynchronize());
    }
    std::vector<uint16_t> khh = download(dkh, qn), klh = download(dkl, qn);

    /* f64 reference of the carry's recurrence on the same bf16/f32 operands. */
    std::vector<double> ro(qn), rs(sn);
    if (ref_on) {
        for (size_t i = 0; i < sn; ++i) rs[i] = state0[i];
        std::vector<double> vp(BT * V);
        for (unsigned h = 0; h < H; ++h) {
            double* S = rs.data() + (size_t)h * V * D;
            for (unsigned c0 = 0; c0 < T; c0 += BT) {
                const unsigned n = std::min(BT, T - c0);
                auto ix = [&](unsigned i, unsigned d) { return ((size_t)(c0 + i) * H + h) * D + d; };
                for (unsigned m = 0; m < n; ++m)
                    for (unsigned v = 0; v < V; ++v) {
                        double pr = 0, fs = 0;
                        for (unsigned d = 0; d < D; ++d) { pr += f32(w[ix(m, d)]) * S[v * D + d]; fs += f32(q[ix(m, d)]) * S[v * D + d]; }
                        vp[m * V + v] = f32(u[ix(m, v)]) - pr;
                        ro[ix(m, v)] = fs;
                    }
                for (unsigned m = 0; m < n; ++m)
                    for (unsigned v = 0; v < V; ++v) {
                        double l = 0;
                        for (unsigned s = 0; s < n; ++s) l += aqk[((size_t)(c0 + m) * H + h) * BT + s] * vp[s * V + v];
                        ro[ix(m, v)] += l;
                    }
                for (unsigned v = 0; v < V; ++v)
                    for (unsigned d = 0; d < D; ++d) {
                        double a = S[v * D + d] * std::exp2(double(g[ix(n - 1, d)]));
                        for (unsigned s = 0; s < n; ++s) a += (double(f32(khh[ix(s, d)])) + f32(klh[ix(s, d)])) * vp[s * V + v];
                        S[v * D + d] = a;
                    }
            }
        }
    }
    auto report = [&](const char* name, const uint16_t* dout, const float* dstate) {
        if (!ref_on) return;
        std::vector<uint16_t> o = download(dout, qn);
        std::vector<float> s = download(dstate, sn);
        double eo = 0, no = 0, mo = 0, es = 0, ns = 0, ms = 0;
        for (size_t i = 0; i < qn; ++i) { const double e = f32(o[i]) - ro[i]; eo += e * e; no += ro[i] * ro[i]; mo = std::max(mo, std::fabs(e)); }
        for (size_t i = 0; i < sn; ++i) { const double e = s[i] - rs[i]; es += e * e; ns += rs[i] * rs[i]; ms = std::max(ms, std::fabs(e)); }
        std::printf("accuracy arm=%s out_rel_rms=%.3e out_max_abs=%.3e out_rms=%.3e state_rel_rms=%.3e state_max_abs=%.3e\n",
                    name, std::sqrt(eo / no), mo, std::sqrt(no / qn), std::sqrt(es / ns), ms);
    };

    uint16_t* dout; float* dstate;
    CHECK(hipMalloc(&dout, qn * 2)); CHECK(hipMalloc(&dstate, sn * 4));
    hipEvent_t ev0, ev1;
    CHECK(hipEventCreate(&ev0)); CHECK(hipEventCreate(&ev1));
    auto timeit = [&](auto&& run) {
        std::vector<float> v;
        for (unsigned r = 0; r < nsample + 2; ++r) {
            CHECK(hipMemcpy(dstate, state0.data(), sn * 4, hipMemcpyHostToDevice));
            CHECK(hipEventRecord(ev0)); run(); CHECK(hipEventRecord(ev1)); CHECK(hipEventSynchronize(ev1));
            float ms; CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            if (r >= 2) v.push_back(ms);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    const float base = timeit([&] {
        void* args[] = {&dout, &dstate, &dq, &dk, &dw, &du, &da, &dg, (void*)&T, (void*)&H, &dkh, &dkl};
        launch(k_carry_keyfeed, H * 8, args);
    });
    std::printf("shape T=%u H=%u\nmedian_us arm=keyfeed time=%.1f\n", T, H, base * 1000);
    report("keyfeed", dout, dstate);

    float* scratch;
    const unsigned pmax = 16;
    CHECK(hipMalloc(&scratch, 3ull * (pmax - 1) * sn * 4 + qn * 2));
    for (unsigned P : {2u, 3u, 4u, 6u, 8u, 12u, 16u}) {
        unsigned seg = ((T + P - 1) / P + BT - 1) / BT * BT;
        const unsigned nseg = (T + seg - 1) / seg;
        if (nseg < 2) continue;
        SplitArgs a{dout, dstate, dq, dk, dw, du, da, dg, dkh, dkl, scratch, T, H, nseg, seg};
        const unsigned tiles = (T - seg + BT - 1) / BT;
        float t1 = 0;
        auto run = [&] {
            void* args[] = {&a};
            launch(k_seg, H * 8 * (2 * nseg - 1), args);
            launch(k_combine, H * V / 2, args);
            launch(k_fix, H * tiles, args);
        };
        const float ms = timeit(run);
        t1 = timeit([&] {
            void* args[] = {&a};
            launch(k_seg, H * 8 * (2 * nseg - 1), args);
        });
        const float tc = timeit([&] {
            void* args[] = {&a};
            launch(k_combine, H * V / 2, args);
        });
        const float tf = timeit([&] {
            void* args[] = {&a};
            launch(k_fix, H * tiles, args);
        });
        char name[64];
        std::snprintf(name, sizeof name, "split_p%u", nseg);
        CHECK(hipMemcpy(dstate, state0.data(), sn * 4, hipMemcpyHostToDevice));
        run();
        CHECK(hipDeviceSynchronize());
        std::printf("median_us arm=%s seg=%u time=%.1f seg_pass=%.1f combine=%.1f fix=%.1f speedup=%.3fx\n", name, seg, ms * 1000, t1 * 1000, tc * 1000, tf * 1000, base / ms);
        report(name, dout, dstate);
    }
    return 0;
}
