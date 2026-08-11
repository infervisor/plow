#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static constexpr unsigned GRID = 304;
static constexpr unsigned THREADS = 512;
static constexpr unsigned NH = 12;
static constexpr unsigned DK = 512;
static constexpr unsigned DR = 64;
static constexpr unsigned NS = 64;
static constexpr size_t FLUSH_BYTES = 512ull << 20;

static uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

struct Kernel {
    hipModule_t module;
    hipFunction_t mla;
    hipFunction_t flush;
};

static Kernel load(const char* path) {
    Kernel k{};
    CK(hipModuleLoad(&k.module, path));
    CK(hipModuleGetFunction(&k.mla, k.module, "k_mla_decode_fp8"));
    CK(hipModuleGetFunction(&k.flush, k.module, "k_cache_flush"));
    return k;
}

static void launch(hipFunction_t fn, void** args) {
    CK(hipModuleLaunchKernel(fn, GRID, 1, 1, THREADS, 1, 1, 0, 0, args, nullptr));
}

static double timed(hipFunction_t fn, void** args, hipFunction_t flush, void** flush_args) {
    launch(flush, flush_args);
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    CK(hipEventRecord(begin));
    launch(fn, args);
    CK(hipEventRecord(end));
    CK(hipEventSynchronize(end));
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, begin, end));
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    return ms * 1000.0;
}

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <control.co> <candidate.co> [samples]\n", argv[0]);
        return 2;
    }
    const int samples = argc > 3 ? std::atoi(argv[3]) : 21;
    if (samples < 3 || (samples & 1) == 0) {
        std::fprintf(stderr, "samples must be an odd integer >= 3\n");
        return 2;
    }
    CK(hipInit(0));
    Kernel control = load(argv[1]);
    Kernel candidate = load(argv[2]);
    std::printf("control=%s candidate=%s\n", argv[1], argv[2]);

    uint32_t *flush_in, *flush_out;
    CK(hipMalloc(&flush_in, FLUSH_BYTES));
    CK(hipMalloc(&flush_out, sizeof(uint32_t)));
    CK(hipMemset(flush_in, 0x5a, FLUSH_BYTES));
    CK(hipMemset(flush_out, 0, sizeof(uint32_t)));
    size_t flush_n = FLUSH_BYTES / sizeof(uint32_t);
    void* flush_args[] = {&flush_in, &flush_out, &flush_n};

    const unsigned contexts[] = {149, 8192, 32768, 128000};
    std::printf("%-8s %12s %12s %10s %14s\n", "ctx", "control_us", "candidate_us", "speedup",
                "24layer_save_ms");
    for (unsigned ctx : contexts) {
        std::vector<uint8_t> h_ckv((size_t)ctx * DK);
        std::vector<uint16_t> h_kr((size_t)ctx * DR), h_qa(NH * DK), h_qr(NH * DR);
        std::vector<float> h_scale((size_t)2 * ctx, 1.0f);
        for (size_t i = 0; i < h_ckv.size(); ++i) {
            static const uint8_t fp8[] = {0x28, 0x30, 0x34, 0x38, 0x3c, 0x40, 0xa8, 0xb4};
            h_ckv[i] = fp8[(i * 17 + 3) & 7];
        }
        for (size_t i = 0; i < h_kr.size(); ++i)
            h_kr[i] = f2bf(std::sin((double)(i % 257) * 0.031) * 0.125f);
        for (size_t i = 0; i < h_qa.size(); ++i)
            h_qa[i] = f2bf(std::cos((double)(i % 263) * 0.027) * 0.25f);
        for (size_t i = 0; i < h_qr.size(); ++i)
            h_qr[i] = f2bf(std::sin((double)(i % 127) * 0.043) * 0.25f);
        for (unsigned i = 0; i < ctx; ++i) h_scale[i] = 0.75f + 0.125f * (i & 3);

        uint8_t* d_ckv;
        uint16_t *d_kr, *d_qa, *d_qr;
        float *d_scale, *d_o0, *d_m0, *d_o1, *d_m1;
        int* d_len;
        const size_t no = (size_t)NH * NS * DK;
        const size_t nm = (size_t)NH * NS * 2;
        CK(hipMalloc(&d_ckv, h_ckv.size()));
        CK(hipMalloc(&d_kr, h_kr.size() * 2));
        CK(hipMalloc(&d_qa, h_qa.size() * 2));
        CK(hipMalloc(&d_qr, h_qr.size() * 2));
        CK(hipMalloc(&d_scale, h_scale.size() * 4));
        CK(hipMalloc(&d_len, sizeof(int)));
        CK(hipMalloc(&d_o0, no * 4));
        CK(hipMalloc(&d_m0, nm * 4));
        CK(hipMalloc(&d_o1, no * 4));
        CK(hipMalloc(&d_m1, nm * 4));
        CK(hipMemcpy(d_ckv, h_ckv.data(), h_ckv.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(d_kr, h_kr.data(), h_kr.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_qa, h_qa.data(), h_qa.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_qr, h_qr.data(), h_qr.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_scale, h_scale.data(), h_scale.size() * 4, hipMemcpyHostToDevice));
        int len = static_cast<int>(ctx);
        CK(hipMemcpy(d_len, &len, sizeof(len), hipMemcpyHostToDevice));
        unsigned nh = NH, stride = ctx, ns = NS;
        float scale = 0.0883883476f;
        void* a0[] = {&d_o0, &d_m0, &d_qa, &d_qr, &d_ckv, &d_kr, &d_len,
                      &nh,   &stride, &ns,   &scale, &d_scale};
        void* a1[] = {&d_o1, &d_m1, &d_qa, &d_qr, &d_ckv, &d_kr, &d_len,
                      &nh,   &stride, &ns,   &scale, &d_scale};
        CK(hipMemset(d_o0, 0, no * 4));
        CK(hipMemset(d_m0, 0, nm * 4));
        CK(hipMemset(d_o1, 0, no * 4));
        CK(hipMemset(d_m1, 0, nm * 4));
        launch(control.mla, a0);
        launch(candidate.mla, a1);
        CK(hipDeviceSynchronize());
        std::vector<float> o0(no), o1(no), m0(nm), m1(nm);
        CK(hipMemcpy(o0.data(), d_o0, no * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(o1.data(), d_o1, no * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(m0.data(), d_m0, nm * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(m1.data(), d_m1, nm * 4, hipMemcpyDeviceToHost));
        if (o0 != o1 || m0 != m1) {
            std::fprintf(stderr, "FAIL ctx=%u: candidate output differs from control\n", ctx);
            return 3;
        }

        std::vector<double> t0, t1;
        t0.reserve(samples);
        t1.reserve(samples);
        for (int i = 0; i < samples; ++i) {
            if ((i & 1) == 0) {
                t0.push_back(timed(control.mla, a0, control.flush, flush_args));
                t1.push_back(timed(candidate.mla, a1, candidate.flush, flush_args));
            } else {
                t1.push_back(timed(candidate.mla, a1, candidate.flush, flush_args));
                t0.push_back(timed(control.mla, a0, control.flush, flush_args));
            }
        }
        const double c = median(t0), v = median(t1);
        std::printf("%-8u %12.3f %12.3f %10.4f %14.3f\n", ctx, c, v, c / v,
                    (c - v) * 24.0 / 1000.0);

        CK(hipFree(d_ckv)); CK(hipFree(d_kr)); CK(hipFree(d_qa)); CK(hipFree(d_qr));
        CK(hipFree(d_scale)); CK(hipFree(d_len)); CK(hipFree(d_o0)); CK(hipFree(d_m0));
        CK(hipFree(d_o1)); CK(hipFree(d_m1));
    }
    return 0;
}
