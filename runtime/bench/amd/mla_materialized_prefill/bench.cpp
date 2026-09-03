#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
static constexpr unsigned NH = 12;
static constexpr unsigned DK_ABS = 512;
static constexpr unsigned DR = 64;
static constexpr unsigned DK_MAT = 192;
static constexpr unsigned DV = 128;
static constexpr size_t FLUSH_BYTES = 512ull << 20;

static float bf16_to_float(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

static void launch(hipFunction_t fn, unsigned grid, unsigned threads, void** args) {
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
}

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <kernel.co> [samples]\n", argv[0]);
        return 2;
    }
    const int samples = argc > 2 ? std::atoi(argv[2]) : 9;
    if (samples < 3 || !(samples & 1)) return 2;

    CK(hipInit(0));
    hipModule_t mod;
    CK(hipModuleLoad(&mod, argv[1]));
    hipFunction_t absorbed, fold, materialized, init, flush;
    CK(hipModuleGetFunction(&absorbed, mod, "k_absorbed"));
    CK(hipModuleGetFunction(&fold, mod, "k_absorbed_fold"));
    CK(hipModuleGetFunction(&materialized, mod, "k_materialized"));
    CK(hipModuleGetFunction(&init, mod, "k_make_inputs"));
    CK(hipModuleGetFunction(&flush, mod, "k_cache_flush"));

    uint32_t *flush_in, *flush_out;
    CK(hipMalloc(&flush_in, FLUSH_BYTES));
    CK(hipMalloc(&flush_out, sizeof(uint32_t)));
    CK(hipMemset(flush_in, 0x5a, FLUSH_BYTES));
    CK(hipMemset(flush_out, 0, sizeof(uint32_t)));
    size_t flush_n = FLUSH_BYTES / sizeof(uint32_t);
    void* flush_args[] = {&flush_in, &flush_out, &flush_n};

    for (unsigned nt : {1024u, 8192u}) {
        const size_t qrows = (size_t)nt * NH;
        uint16_t *qabs, *qrope, *ckv, *krope, *qmat, *kmat, *vmat, *wuv;
        uint16_t *oabs, *omat;
        float *opart_abs, *mlpart_abs, *opart_mat, *mlpart_mat;
        int* kv_len;
        CK(hipMalloc(&qabs, qrows * DK_ABS * 2));
        CK(hipMalloc(&qrope, qrows * DR * 2));
        CK(hipMalloc(&ckv, (size_t)nt * DK_ABS * 2));
        CK(hipMalloc(&krope, (size_t)nt * DR * 2));
        CK(hipMalloc(&qmat, qrows * DK_MAT * 2));
        CK(hipMalloc(&kmat, qrows * DK_MAT * 2));
        CK(hipMalloc(&vmat, qrows * DV * 2));
        CK(hipMalloc(&wuv, (size_t)NH * DK_ABS * DV * 2));
        CK(hipMalloc(&oabs, qrows * DV * 2));
        CK(hipMalloc(&omat, qrows * DV * 2));
        CK(hipMalloc(&opart_abs, qrows * DK_ABS * 4));
        CK(hipMalloc(&mlpart_abs, qrows * 2 * 4));
        CK(hipMalloc(&opart_mat, qrows * DV * 4));
        CK(hipMalloc(&mlpart_mat, qrows * 2 * 4));
        CK(hipMalloc(&kv_len, sizeof(int)));
        int len = nt;
        CK(hipMemcpy(kv_len, &len, sizeof(len), hipMemcpyHostToDevice));

        unsigned nh = NH;
        void* init_args[] = {&qabs, &qrope, &ckv, &krope, &qmat, &kmat, &vmat, &wuv, &nt,
                             &nh};
        launch(init, GRID, 256, init_args);
        unsigned stride = nt;
        float scale = 0.07216878365f;
        void* abs_args[] = {&opart_abs, &mlpart_abs, &qabs, &qrope, &ckv, &krope,
                            &kv_len, &nt, &nh, &stride, &scale};
        void* fold_args[] = {&oabs, &opart_abs, &mlpart_abs, &wuv, &nt, &nh};
        void* mat_args[] = {&opart_mat, &mlpart_mat, &omat, &qmat, &kmat, &vmat,
                            &nt, &nh, &stride, &scale};

        launch(absorbed, GRID, 256, abs_args);
        launch(fold, GRID, 256, fold_args);
        launch(materialized, GRID, 256, mat_args);
        CK(hipDeviceSynchronize());

        std::vector<uint16_t> ha(qrows * DV), hm(qrows * DV);
        CK(hipMemcpy(ha.data(), oabs, ha.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hm.data(), omat, hm.size() * 2, hipMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < ha.size(); ++i) {
            const double a = bf16_to_float(ha[i]);
            const double b = bf16_to_float(hm[i]);
            const double d = std::abs(a - b);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / (std::abs(a) + 1e-6));
            sum_sq += d * d;
            bad += d > 7.8125e-3;
        }
        const double rmse = std::sqrt(sum_sq / ha.size());
        if (max_abs > 2.0e-2 || rmse > 3.0e-3) {
            std::fprintf(stderr,
                         "FAIL T=%u oracle max_abs=%.6g max_rel=%.6g rmse=%.6g bad=%zu/%zu\n",
                         nt, max_abs, max_rel, rmse, bad, ha.size());
            return 3;
        }

        std::vector<double> ta, tm;
        for (int s = 0; s < samples; ++s) {
            launch(flush, GRID, 256, flush_args);
            CK(hipDeviceSynchronize());
            hipEvent_t begin, end;
            CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(absorbed, GRID, 256, abs_args);
            launch(fold, GRID, 256, fold_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            float ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            ta.push_back(ms * 1000.0);
            CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));

            launch(flush, GRID, 256, flush_args);
            CK(hipDeviceSynchronize());
            CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(materialized, GRID, 256, mat_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            tm.push_back(ms * 1000.0);
            CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));

        }
        const double a = median(ta), m = median(tm);
        std::printf("T=%u H=%u absorbed+fold=%.3f us materialized-attn=%.3f us speedup=%.3fx "
                    "oracle(max_abs=%.6g rmse=%.6g bad=%zu/%zu)\n",
                    nt, NH, a, m, a / m, max_abs, rmse, bad, ha.size());

        hipFree(qabs); hipFree(qrope); hipFree(ckv); hipFree(krope); hipFree(qmat);
        hipFree(kmat); hipFree(vmat); hipFree(wuv); hipFree(oabs); hipFree(omat);
        hipFree(opart_abs); hipFree(mlpart_abs); hipFree(opart_mat); hipFree(mlpart_mat);
        hipFree(kv_len);
    }
    hipFree(flush_in); hipFree(flush_out);
    hipModuleUnload(mod);
    return 0;
}
