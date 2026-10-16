// GLM-5.3-Flash TP8 batched-decode MLA attention: hp8 flash + fold (scalar vs MFMA), per-launch
// time and agreement. kern.hip bodies; a variant object (same symbols) can be passed second.
//   bench <kern.elf> [variant.elf|-] [rows=128] [ctx=1024] [nsplit=2] [reps=50] [kv_stride=ctx]
#include <hip/hip_runtime.h>

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

struct Obj {
    hipFunction_t hp8, fs, fm, tb[3];
};
static Obj load(const char* path) {
    hipModule_t m;
    CHECK(hipModuleLoad(&m, path));
    Obj o;
    CHECK(hipModuleGetFunction(&o.hp8, m, "mla_hp8"));
    CHECK(hipModuleGetFunction(&o.fs, m, "mla_fold_scalar"));
    CHECK(hipModuleGetFunction(&o.fm, m, "mla_fold_mfma"));
    const char* tbn[3] = {"mla_fold_tb2", "mla_fold_tb4", "mla_fold_tb8"};
    for (int i = 0; i < 3; ++i)
        if (hipModuleGetFunction(&o.tb[i], m, tbn[i]) != hipSuccess) o.tb[i] = nullptr;
    return o;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const char* var = argc > 2 && std::strcmp(argv[2], "-") ? argv[2] : nullptr;
    const unsigned B = argc > 3 ? std::atoi(argv[3]) : 128, ctx = argc > 4 ? std::atoi(argv[4]) : 1024;
    unsigned ns = argc > 5 ? std::atoi(argv[5]) : 2;
    const unsigned reps = argc > 6 ? std::atoi(argv[6]) : 50;
    const unsigned NH = 8, DK = 512, V = 256;
    const unsigned stride = argc > 7 ? std::atoi(argv[7]) : ctx; /* serving rings: 65536 */
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bvec = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(s * nd(rng)); return v; };
    auto q = up(bvec((size_t)B * NH * DK, 1.f));
    uint16_t* ckv;
    {
        auto h = bvec((size_t)B * ctx * DK, 1.f);
        CHECK(hipMalloc(&ckv, (size_t)B * stride * DK * 2));
        CHECK(hipMemcpy2D(ckv, (size_t)stride * DK * 2, h.data(), (size_t)ctx * DK * 2,
                          (size_t)ctx * DK * 2, B, hipMemcpyHostToDevice));
    }
    auto wuv = up(bvec((size_t)NH * DK * V, 0.05f));
    std::vector<int> lens(B);
    for (unsigned b = 0; b < B; ++b) lens[b] = ctx - (b * 7) % 64;
    auto kvl = up(lens);
    float *opart, *ml;
    CHECK(hipMalloc(&opart, (size_t)B * NH * 64 * DK * 4));
    CHECK(hipMalloc(&ml, (size_t)B * NH * 64 * 2 * 4));
    uint16_t* o[4];
    for (auto& p : o) CHECK(hipMalloc(&p, (size_t)B * NH * V * 2));
    const float scale = 1.0f / std::sqrt(576.0f);
    const unsigned hp8_lds = 96 * 1024; /* >= the BKV=64 variant's arena */

    Obj objs[2] = {load(argv[1]), var ? load(var) : load(argv[1])};
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    auto timeit = [&](auto&& fn) {
        fn();
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (unsigned r = 0; r < reps; ++r) fn();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        return ms * 1000.f / reps;
    };
    for (int i = 0; i < 2; ++i) {
        Obj& ob = objs[i];
        unsigned nb = B, kvs = stride, nsp = ns, nh = NH;
        float sc = scale;
        void* fa[] = {&opart, &ml, &q, &ckv, &kvl, &nb, &kvs, &sc, &nsp};
        auto flash = [&] { CHECK(hipModuleLaunchKernel(ob.hp8, 256, 1, 1, 512, 1, 1, hp8_lds, nullptr, fa, nullptr)); };
        const float t_fl = timeit(flash);
        flash();
        void* fs_a[] = {&o[2 * i], &opart, &ml, &wuv, &nb, &nh, &nsp};
        void* fm_a[] = {&o[2 * i + 1], &opart, &ml, &wuv, &nb, &nh, &nsp};
        const float t_fs = timeit([&] { CHECK(hipModuleLaunchKernel(ob.fs, 256, 1, 1, 512, 1, 1, (DK + 8 * 256) * 4, nullptr, fs_a, nullptr)); });
        const float t_fm = timeit([&] { CHECK(hipModuleLaunchKernel(ob.fm, 256, 1, 1, 512, 1, 1, V * 40 * 2, nullptr, fm_a, nullptr)); });
        std::printf("obj=%d rows=%u ctx=%u ns=%u flash_us=%.1f fold_scalar_us=%.1f fold_mfma_us=%.1f\n", i, B, ctx, ns, t_fl, t_fs, t_fm);
        for (int k = 0; k < 3; ++k) {
            if (!ob.tb[k]) continue;
            void* tb_a[] = {&o[2 * i + 1], &opart, &ml, &wuv, &nb, &nh, &nsp};
            const float t = timeit([&] { CHECK(hipModuleLaunchKernel(ob.tb[k], 256, 1, 1, 512, 1, 1, 12288 * 4, nullptr, tb_a, nullptr)); });
            CHECK(hipDeviceSynchronize());
            auto ha = down(o[2 * i], (size_t)B * NH * V), hb = down(o[2 * i + 1], (size_t)B * NH * V);
            size_t diff = 0;
            for (size_t e = 0; e < ha.size(); ++e) diff += ha[e] != hb[e];
            std::printf("obj=%d fold_tb%d_us=%.1f differing_vs_scalar=%zu\n", i, 2 << k, t, diff);
            CHECK(hipModuleLaunchKernel(ob.fm, 256, 1, 1, 512, 1, 1, V * 40 * 2, nullptr, fm_a, nullptr));
        }
    }
    CHECK(hipDeviceSynchronize());
    auto cmp = [&](const char* what, uint16_t* a, uint16_t* b) {
        auto ha = down(a, (size_t)B * NH * V), hb = down(b, (size_t)B * NH * V);
        double num = 0, den = 0, mx = 0;
        size_t bad = 0;
        for (size_t k = 0; k < ha.size(); ++k) {
            const double x = f32(ha[k]), y = f32(hb[k]);
            if (!std::isfinite(x) || !std::isfinite(y)) { ++bad; continue; }
            num += (x - y) * (x - y);
            den += x * x;
            mx = std::fmax(mx, std::fabs(x - y));
        }
        std::printf("%s rel_rms=%.3e max_abs=%.3e nonfinite=%zu\n", what, std::sqrt(num / (den + 1e-30)), mx, bad);
    };
    cmp("base: scalar vs mfma fold", o[0], o[1]);
    cmp("scalar fold: base vs variant flash", o[0], o[2]);
    cmp("mfma fold: base vs variant", o[1], o[3]);
    return 0;
}
