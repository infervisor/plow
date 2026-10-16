// kda_decode_wide_v3 (op 125 v3) with PLOW_KDA_DECODE_LANES=0 vs =1 at the TP8 batched-decode shape:
// output / state agreement after one step and the per-launch time.
//   bench <l0.elf> <l1.elf> [rows=128] [heads=8] [reps=50]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); \
} } while (0)

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
template <class T> static T* up(const std::vector<T>& h) {
    T* p;
    CHECK(hipMalloc(&p, h.size() * sizeof(T)));
    CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const unsigned R = argc > 3 ? std::atoi(argv[3]) : 128, H = argc > 4 ? std::atoi(argv[4]) : 8;
    const unsigned reps = argc > 5 ? std::atoi(argv[5]) : 50;
    const unsigned D = 128, P = H * D, W = 4;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bvec = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(s * nd(rng)); return v; };
    auto fvec = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = s * nd(rng); return v; };
    auto qr = bvec((size_t)R * P, 1.f), kr = bvec((size_t)R * P, 1.f), vr = bvec((size_t)R * P, 1.f);
    auto wq = fvec((size_t)P * W, 0.5f), wk = fvec((size_t)P * W, 0.5f), wv = fvec((size_t)P * W, 0.5f);
    auto cs0 = fvec((size_t)R * P * W, 0.5f);
    auto fr = bvec((size_t)R * P, 1.f), br = bvec((size_t)R * H, 1.f), og = bvec((size_t)R * P, 1.f);
    auto alog = fvec(H, 0.5f), dtb = fvec(P, 0.5f), nw = fvec(D, 0.2f);
    for (auto& x : nw) x += 1.f;
    auto st0 = fvec((size_t)R * H * D * D, 0.05f);

    uint16_t *dq = up(qr), *dk = up(kr), *dv = up(vr), *dfr = up(fr), *dbr = up(br), *dog = up(og);
    float *dwq = up(wq), *dwk = up(wk), *dwv = up(wv), *dal = up(alog), *ddt = up(dtb), *dnw = up(nw);
    float *csq = up(cs0), *csk = up(cs0), *csv = up(cs0), *st = up(st0);
    uint16_t *y, *mq, *mk, *mv, *o;
    for (uint16_t** p : {&y, &mq, &mk, &mv, &o}) CHECK(hipMalloc(p, (size_t)R * P * 2));
    const unsigned* parked = nullptr;
    unsigned rows = R, heads = H, dim = D, bv = 8, conv_w = W, flags = 3, gate_mode = 1;
    float lb = -5.f, scale = 1.f / std::sqrt(128.f), eps = 1e-6f;
    void* args[] = {&y, &dq, &dk, &dv, &dwq, &dwk, &dwv, &csq, &csk, &csv, &dfr, &dbr, &dog, &dal,
                    &ddt, &st, &dnw, &parked, &rows, &heads, &dim, &bv, &conv_w, &flags, &gate_mode,
                    &lb, &scale, &eps, &mq, &mk, &mv, &o};
    hipFunction_t fn[2];
    for (int i = 0; i < 2; ++i) {
        hipModule_t m;
        CHECK(hipModuleLoad(&m, argv[1 + i]));
        CHECK(hipModuleGetFunction(&fn[i], m, "plow_kda_decode_wide_v3"));
    }
    auto reset = [&] {
        for (float* c : {csq, csk, csv}) CHECK(hipMemcpy(c, cs0.data(), cs0.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(st, st0.data(), st0.size() * 4, hipMemcpyHostToDevice));
    };
    auto launch = [&](int i) { CHECK(hipModuleLaunchKernel(fn[i], R * H, 1, 1, 512, 1, 1, 0, nullptr, args, nullptr)); };
    std::vector<uint16_t> ys[2];
    std::vector<float> ss[2];
    for (int i = 0; i < 2; ++i) {
        reset();
        launch(i);
        CHECK(hipDeviceSynchronize());
        ys[i] = down(y, (size_t)R * P);
        ss[i] = down(st, st0.size());
    }
    double ey = 0, ny = 0, my = 0, es = 0, ns = 0, ms = 0;
    for (size_t j = 0; j < ys[0].size(); ++j) {
        const double a = f32(ys[0][j]), b = f32(ys[1][j]);
        ey += (a - b) * (a - b); ny += a * a; my = std::max(my, std::fabs(a - b));
    }
    for (size_t j = 0; j < ss[0].size(); ++j) {
        const double a = ss[0][j], b = ss[1][j];
        es += (a - b) * (a - b); ns += a * a; ms = std::max(ms, std::fabs(a - b));
    }
    std::printf("agree y_rel_rms=%.3e y_max_abs=%.3e state_rel_rms=%.3e state_max_abs=%.3e\n",
                std::sqrt(ey / ny), my, std::sqrt(es / ns), ms);
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    for (int round = 0; round < 2; ++round)
        for (int i = 0; i < 2; ++i) {
            launch(i);
            CHECK(hipEventRecord(e0));
            for (unsigned r = 0; r < reps; ++r) launch(i);
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            float ms_ = 0;
            CHECK(hipEventElapsedTime(&ms_, e0, e1));
            std::printf("round=%d lanes=%d rows=%u H=%u us_per_launch=%.1f\n", round, i, R, H, ms_ * 1000 / reps);
        }
    return 0;
}
