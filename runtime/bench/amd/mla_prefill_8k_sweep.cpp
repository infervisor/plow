#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                  \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) {                                                                \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static constexpr unsigned GRID = 256;
static constexpr unsigned T = 8192;
static constexpr unsigned NH = 12;
static constexpr unsigned DK = 512;
static constexpr unsigned DR = 64;
static constexpr size_t FLUSH_BYTES = 512ull << 20;

struct Module {
    hipModule_t module{};
    hipFunction_t mfma{}, v2{}, init{}, flush{};
};

static Module load(const char* path) {
    Module m;
    CK(hipModuleLoad(&m.module, path));
    CK(hipModuleGetFunction(&m.mfma, m.module, "k_mla_prefill_mfma"));
    CK(hipModuleGetFunction(&m.v2, m.module, "k_mla_prefill_v2"));
    CK(hipModuleGetFunction(&m.init, m.module, "k_init_bf16"));
    CK(hipModuleGetFunction(&m.flush, m.module, "k_cache_flush"));
    return m;
}

static void launch(hipFunction_t fn, unsigned threads, void** args) {
    CK(hipModuleLaunchKernel(fn, GRID, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
}

static double timed(hipFunction_t fn, unsigned threads, void** args, hipFunction_t flush,
                    void** flush_args) {
    launch(flush, 256, flush_args);
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    CK(hipEventRecord(begin));
    launch(fn, threads, args);
    CK(hipEventRecord(end));
    CK(hipEventSynchronize(end));
    float ms = 0;
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
    const int samples = argc > 3 ? std::atoi(argv[3]) : 9;
    if (samples < 3 || !(samples & 1)) {
        std::fprintf(stderr, "samples must be odd and >=3\n");
        return 2;
    }
    CK(hipInit(0));
    Module control = load(argv[1]);
    Module candidate = load(argv[2]);

    uint16_t *qabs, *qrope, *ckv, *krope;
    float *omfma, *mmfma, *ov2, *mv2, *ocand, *mcand;
    int* kv_len;
    const size_t nqa = (size_t)T * NH * DK;
    const size_t nqr = (size_t)T * NH * DR;
    const size_t nk = (size_t)T * DK;
    const size_t nr = (size_t)T * DR;
    const size_t no = nqa;
    const size_t nm = (size_t)T * NH * 2;
    CK(hipMalloc(&qabs, nqa * 2));
    CK(hipMalloc(&qrope, nqr * 2));
    CK(hipMalloc(&ckv, nk * 2));
    CK(hipMalloc(&krope, nr * 2));
    CK(hipMalloc(&kv_len, sizeof(int)));
    CK(hipMalloc(&omfma, no * 4)); CK(hipMalloc(&mmfma, nm * 4));
    CK(hipMalloc(&ov2, no * 4)); CK(hipMalloc(&mv2, nm * 4));
    CK(hipMalloc(&ocand, no * 4)); CK(hipMalloc(&mcand, nm * 4));
    int len = T;
    CK(hipMemcpy(kv_len, &len, sizeof(len), hipMemcpyHostToDevice));

    auto init = [&](uint16_t* p, size_t n, unsigned seed) {
        void* a[] = {&p, &n, &seed};
        launch(control.init, 256, a);
    };
    init(qabs, nqa, 1); init(qrope, nqr, 2); init(ckv, nk, 3); init(krope, nr, 4);

    unsigned nt = T, nh = NH, stride = T;
    float scale = 0.0883883476f;
    void* amfma[] = {&omfma, &mmfma, &qabs, &qrope, &ckv, &krope, &kv_len,
                     &nt, &nh, &stride, &scale};
    void* av2[] = {&ov2, &mv2, &qabs, &qrope, &ckv, &krope, &kv_len,
                   &nt, &nh, &stride, &scale};
    void* acand[] = {&ocand, &mcand, &qabs, &qrope, &ckv, &krope, &kv_len,
                     &nt, &nh, &stride, &scale};
    launch(control.mfma, 512, amfma);
    launch(control.v2, 256, av2);
    launch(candidate.v2, 256, acand);
    CK(hipDeviceSynchronize());

    std::vector<float> hmfma(no), hv2(no), hcand(no), hmmfma(nm), hmv2(nm), hmcand(nm);
    CK(hipMemcpy(hmfma.data(), omfma, no * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hv2.data(), ov2, no * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hcand.data(), ocand, no * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hmmfma.data(), mmfma, nm * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hmv2.data(), mv2, nm * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hmcand.data(), mcand, nm * 4, hipMemcpyDeviceToHost));
    if (hv2 != hcand || hmv2 != hmcand) {
        std::fprintf(stderr, "FAIL: candidate V2 is not bit-exact at production shape\n");
        return 3;
    }
    double max_rel = 0;
    for (size_t row = 0; row < (size_t)T * NH; ++row) {
        const double lm = hmmfma[row * 2 + 1], lv = hmv2[row * 2 + 1];
        double mag = 0, dif = 0;
        for (unsigned d = 0; d < DK; ++d) {
            const double a = lm > 0 ? hmfma[row * DK + d] / lm : 0;
            const double b = lv > 0 ? hv2[row * DK + d] / lv : 0;
            mag = std::max(mag, std::abs(a));
            dif = std::max(dif, std::abs(a - b));
        }
        max_rel = std::max(max_rel, dif / (mag + 1e-30));
    }
    if (!(max_rel <= 2e-2)) {
        std::fprintf(stderr, "FAIL: V2 differs from validated MFMA body (max row rel %.3e)\n",
                     max_rel);
        return 4;
    }

    uint32_t *flush_in, *flush_out;
    CK(hipMalloc(&flush_in, FLUSH_BYTES));
    CK(hipMalloc(&flush_out, sizeof(uint32_t)));
    CK(hipMemset(flush_in, 0x5a, FLUSH_BYTES));
    CK(hipMemset(flush_out, 0, sizeof(uint32_t)));
    size_t flush_n = FLUSH_BYTES / 4;
    void* flush_args[] = {&flush_in, &flush_out, &flush_n};
    std::vector<double> tm, tv, tc;
    for (int i = 0; i < samples; ++i) {
        if (i & 1) {
            tc.push_back(timed(candidate.v2, 256, acand, candidate.flush, flush_args));
            tv.push_back(timed(control.v2, 256, av2, control.flush, flush_args));
            tm.push_back(timed(control.mfma, 512, amfma, control.flush, flush_args));
        } else {
            tm.push_back(timed(control.mfma, 512, amfma, control.flush, flush_args));
            tv.push_back(timed(control.v2, 256, av2, control.flush, flush_args));
            tc.push_back(timed(candidate.v2, 256, acand, candidate.flush, flush_args));
        }
    }
    const double mfma = median(tm), v2 = median(tv), cand = median(tc);
    std::printf("shape B=1 T=%u H=%u DK=%u DR=%u KV=bf16 grid=%u\n", T, NH, DK, DR, GRID);
    std::printf("mfma8=%9.3f us  v2=%9.3f us  candidate=%9.3f us\n", mfma, v2, cand);
    std::printf("v2/mfma speedup %.4fx; candidate/v2 speedup %.4fx; 24-layer delta %.3f ms\n",
                mfma / v2, v2 / cand, (v2 - cand) * 24.0 / 1000.0);
    std::printf("PASS: V2 max row rel %.3e vs MFMA; candidate bit-exact to V2\n", max_rel);
    return 0;
}
