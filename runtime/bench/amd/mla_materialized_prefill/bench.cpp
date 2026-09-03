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

struct OpusArgs {
    const void* q;
    const void* k;
    const void* v;
    void* o;
    int b, n, n_kv, h, h_kv, d_qk, d_v;
    int sq_b, sq_n, sq_h, so_b, so_n, so_h;
    int sk_b, sk_n, sk_h, sv_b, sv_n, sv_h;
    float scale;
    const int* qseq;
    const int* kseq;
    const int* qseq_pad;
    const int* kseq_pad;
    int opt;
    void* lse;
    int slse_b, slse_h;
};
static_assert(sizeof(OpusArgs) == 168);

struct PackArgs {
    void* k;
    void* v;
    const void* kv;
    const void* k_rope;
    unsigned t, heads, qk_nope, qk_rope, v_head;
};
static_assert(sizeof(PackArgs) == 56);

static float bf16_to_float(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

static void launch(hipFunction_t fn, unsigned grid, unsigned threads, void** args) {
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
}

static void launch3(hipFunction_t fn, dim3 grid, unsigned threads, void** args) {
    CK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, threads, 1, 1, 0, nullptr, args,
                             nullptr));
}

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <kernel.co> <opus.co> <pack.co> <upstream-grid.co> [samples]\n",
                     argv[0]);
        return 2;
    }
    if (argc < 3) return 2;
    if (argc < 5) return 2;
    const int samples = argc > 5 ? std::atoi(argv[5]) : 9;
    if (samples < 3 || !(samples & 1)) return 2;

    CK(hipInit(0));
    hipModule_t mod;
    CK(hipModuleLoad(&mod, argv[1]));
    hipModule_t opus_mod;
    CK(hipModuleLoad(&opus_mod, argv[2]));
    hipModule_t pack_mod;
    CK(hipModuleLoad(&pack_mod, argv[3]));
    hipModule_t upstream_mod;
    CK(hipModuleLoad(&upstream_mod, argv[4]));
    hipFunction_t absorbed, fold, materialized, materialized_lds, init, flush;
    hipFunction_t materialize_q, materialize_kv, absorb_q, absorb_qrope, init_materialize, pack;
    CK(hipModuleGetFunction(&absorbed, mod, "k_absorbed"));
    CK(hipModuleGetFunction(&fold, mod, "k_absorbed_fold"));
    CK(hipModuleGetFunction(&materialized, mod, "k_materialized"));
    CK(hipModuleGetFunction(&materialized_lds, mod, "k_materialized_lds"));
    hipFunction_t opus;
    CK(hipModuleGetFunction(
        &opus, opus_mod, "plow_mla_materialized_hd192_v128_gfx950"));
    CK(hipModuleGetFunction(&init, mod, "k_make_inputs"));
    CK(hipModuleGetFunction(&flush, mod, "k_cache_flush"));
    CK(hipModuleGetFunction(&materialize_q, mod, "k_materialize_q"));
    CK(hipModuleGetFunction(&materialize_kv, mod, "k_materialize_kv"));
    CK(hipModuleGetFunction(&absorb_q, mod, "k_absorb_q"));
    CK(hipModuleGetFunction(&absorb_qrope, mod, "k_absorb_qrope"));
    CK(hipModuleGetFunction(&init_materialize, mod, "k_init_materialize"));
    CK(hipModuleGetFunction(&pack, pack_mod, "plow_mla_materialize_pack_gfx950"));
    hipFunction_t opus_upstream;
    CK(hipModuleGetFunction(
        &opus_upstream, upstream_mod, "oracle_mla_materialized_upstream_grid"));

    uint32_t *flush_in, *flush_out;
    CK(hipMalloc(&flush_in, FLUSH_BYTES));
    CK(hipMalloc(&flush_out, sizeof(uint32_t)));
    CK(hipMemset(flush_in, 0x5a, FLUSH_BYTES));
    CK(hipMemset(flush_out, 0, sizeof(uint32_t)));
    size_t flush_n = FLUSH_BYTES / sizeof(uint32_t);
    void* flush_args[] = {&flush_in, &flush_out, &flush_n};

    bool opus_oracle_failed = false;
    for (unsigned nt : {1024u, 8192u, 1025u}) {
        const size_t qrows = (size_t)nt * NH;
        uint16_t *qabs, *qrope, *ckv, *krope, *qmat, *kmat, *vmat, *wuv;
        uint16_t *oabs, *omat, *olds, *oopus, *oupstream;
        uint16_t *qlat, *qw, *kvw, *qaw, *qrw, *qproj, *kvproj, *kproj, *vproj, *ofull;
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
        CK(hipMalloc(&olds, qrows * DV * 2));
        CK(hipMalloc(&oopus, qrows * DV * 2));
        CK(hipMalloc(&oupstream, qrows * DV * 2));
        CK(hipMalloc(&opart_abs, qrows * DK_ABS * 4));
        CK(hipMalloc(&mlpart_abs, qrows * 2 * 4));
        CK(hipMalloc(&opart_mat, qrows * DV * 4));
        CK(hipMalloc(&mlpart_mat, qrows * 2 * 4));
        CK(hipMalloc(&kv_len, sizeof(int)));
        CK(hipMalloc(&qlat, (size_t)nt * 1536 * 2));
        CK(hipMalloc(&qw, (size_t)NH * DK_MAT * 1536 * 2));
        CK(hipMalloc(&kvw, (size_t)NH * (DV * 2) * DK_ABS * 2));
        CK(hipMalloc(&qaw, (size_t)NH * DK_ABS * 1536 * 2));
        CK(hipMalloc(&qrw, (size_t)NH * DR * 1536 * 2));
        CK(hipMalloc(&qproj, qrows * DK_MAT * 2));
        CK(hipMalloc(&kvproj, qrows * (DV * 2) * 2));
        CK(hipMalloc(&kproj, qrows * DK_MAT * 2));
        CK(hipMalloc(&vproj, qrows * DV * 2));
        CK(hipMalloc(&ofull, qrows * DV * 2));
        int len = nt;
        CK(hipMemcpy(kv_len, &len, sizeof(len), hipMemcpyHostToDevice));

        unsigned nh = NH;
        void* init_args[] = {&qabs, &qrope, &ckv, &krope, &qmat, &kmat, &vmat, &wuv, &nt,
                             &nh};
        launch(init, GRID, 256, init_args);
        void* mi_args[] = {&qlat, &qw, &kvw, &qaw, &qrw, &nt, &nh};
        unsigned stride = nt;
        float scale = 0.07216878365f;
        void* abs_args[] = {&opart_abs, &mlpart_abs, &qabs, &qrope, &ckv, &krope,
                            &kv_len, &nt, &nh, &stride, &scale};
        void* fold_args[] = {&oabs, &opart_abs, &mlpart_abs, &wuv, &nt, &nh};
        void* mat_args[] = {&opart_mat, &mlpart_mat, &omat, &qmat, &kmat, &vmat,
                            &nt, &nh, &stride, &scale};
        void* lds_args[] = {&olds, &qmat, &kmat, &vmat, &nt, &nh, &stride, &scale};
        OpusArgs oa{};
        oa.q = qmat; oa.k = kmat; oa.v = vmat; oa.o = oopus;
        oa.b = 1; oa.n = nt; oa.n_kv = nt; oa.h = nh; oa.h_kv = nh;
        oa.d_qk = DK_MAT; oa.d_v = DV;
        oa.sq_b = nt * nh * DK_MAT; oa.sq_n = nh * DK_MAT; oa.sq_h = DK_MAT;
        oa.so_b = nt * nh * DV; oa.so_n = nh * DV; oa.so_h = DV;
        oa.sk_b = nh * nt * DK_MAT; oa.sk_n = DK_MAT; oa.sk_h = nt * DK_MAT;
        oa.sv_b = nh * nt * DV; oa.sv_n = DV; oa.sv_h = nt * DV;
        oa.scale = scale;
        void* opus_args[] = {&oa};
        const unsigned opus_grid = ((nt + 255) / 256) * nh;
        OpusArgs upstream_oa = oa;
        upstream_oa.o = oupstream;
        void* upstream_args[] = {&upstream_oa};
        void* qproj_args[] = {&qproj, &qlat, &qw, &nt, &nh};
        void* kvproj_args[] = {&kvproj, &ckv, &kvw, &nt, &nh};
        void* qabs_proj_args[] = {&qabs, &qlat, &qaw, &nt, &nh};
        void* qrope_proj_args[] = {&qrope, &qlat, &qrw, &nt, &nh};
        PackArgs pa{kproj, vproj, kvproj, krope, nt, nh, 128, 64, 128};
        void* pack_args[] = {&pa};
        OpusArgs full = oa;
        full.q = qproj; full.k = kproj; full.v = vproj; full.o = ofull;
        full.sk_b = nt * nh * DK_MAT; full.sk_n = nh * DK_MAT; full.sk_h = DK_MAT;
        full.sv_b = nt * nh * DV; full.sv_n = nh * DV; full.sv_h = DV;
        void* full_args[] = {&full};

        launch(absorbed, GRID, 256, abs_args);
        launch(fold, GRID, 256, fold_args);
        launch(materialized, GRID, 256, mat_args);
        launch(materialized_lds, GRID, 256, lds_args);
        CK(hipDeviceSynchronize());

        std::vector<uint16_t> ha(qrows * DV), hm(qrows * DV), hl(qrows * DV), hw(qrows * DV),
            hu(qrows * DV);
        CK(hipMemcpy(ha.data(), oabs, ha.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hm.data(), omat, hm.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hl.data(), olds, hl.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemset(oopus, 0, qrows * DV * 2));
        CK(hipMemset(oupstream, 0xff, qrows * DV * 2));
        launch(opus, opus_grid, 512, opus_args);
        launch3(opus_upstream, dim3((nt + 255) / 256, nh, 1), 512, upstream_args);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(hw.data(), oopus, hw.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hu.data(), oupstream, hu.size() * 2, hipMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
        double lds_max_abs = 0.0, lds_sum_sq = 0.0;
        double lds_vs_mat_max = 0.0, lds_vs_mat_sum_sq = 0.0, lds_peak = 0.0;
        double opus_max_abs = 0.0, opus_sum_sq = 0.0;
        size_t flat_grid_mismatch = 0;
        std::vector<bool> upstream_head_written(NH);
        std::vector<double> opus_head_max(NH), opus_head_sum_sq(NH);
        size_t bad = 0;
        size_t lds_bad = 0;
        for (size_t i = 0; i < ha.size(); ++i) {
            const double a = bf16_to_float(ha[i]);
            const double b = bf16_to_float(hm[i]);
            const double d = std::abs(a - b);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / (std::abs(a) + 1e-6));
            sum_sq += d * d;
            bad += d > 7.8125e-3;
            const double ld = std::abs(a - bf16_to_float(hl[i]));
            lds_max_abs = std::max(lds_max_abs, ld);
            lds_sum_sq += ld * ld;
            lds_bad += ld > 7.8125e-3;
            const double lv = bf16_to_float(hl[i]);
            const double lmd = std::abs(b - lv);
            lds_vs_mat_max = std::max(lds_vs_mat_max, lmd);
            lds_vs_mat_sum_sq += lmd * lmd;
            lds_peak = std::max(lds_peak, std::abs(lv));
            const double wd = std::abs(a - bf16_to_float(hw[i]));
            opus_max_abs = std::max(opus_max_abs, wd);
            opus_sum_sq += wd * wd;
            flat_grid_mismatch += hw[i] != hu[i];
            const unsigned head = (i / DV) % NH;
            upstream_head_written[head] = upstream_head_written[head] || hu[i] != 0xffff;
            opus_head_max[head] = std::max(opus_head_max[head], wd);
            opus_head_sum_sq[head] += wd * wd;
        }
        const double rmse = std::sqrt(sum_sq / ha.size());
        const double lds_rmse = std::sqrt(lds_sum_sq / ha.size());
        const double opus_rmse = std::sqrt(opus_sum_sq / ha.size());
        if (nt % 256 == 0 && (max_abs > 2.0e-2 || rmse > 3.0e-3)) {
            std::fprintf(stderr,
                         "FAIL T=%u oracle max_abs=%.6g max_rel=%.6g rmse=%.6g bad=%zu/%zu\n",
                         nt, max_abs, max_rel, rmse, bad, ha.size());
            return 3;
        }
        if (lds_max_abs > 2.0e-2 || lds_rmse > 3.0e-3) {
            std::fprintf(stderr,
                         "FAIL T=%u LDS oracle max_abs=%.6g rmse=%.6g bad=%zu/%zu "
                         "vs-mat(max_abs=%.6g rmse=%.6g) peak=%.6g\n",
                         nt, lds_max_abs, lds_rmse, lds_bad, ha.size(), lds_vs_mat_max,
                         std::sqrt(lds_vs_mat_sum_sq / ha.size()), lds_peak);
        }
        if (flat_grid_mismatch != 0) {
            std::fprintf(stderr, "FAIL T=%u flat/upstream-grid mismatch=%zu/%zu\n", nt,
                         flat_grid_mismatch, hw.size());
            opus_oracle_failed = true;
        }
        for (unsigned head = 0; head < NH; ++head) {
            if (!upstream_head_written[head]) {
                std::fprintf(stderr, "FAIL T=%u upstream-grid left head=%u untouched\n", nt,
                             head);
                opus_oracle_failed = true;
            }
        }
        if (nt % 256 == 0 && (opus_max_abs > 2.0e-2 || opus_rmse > 3.0e-3)) {
            std::fprintf(stderr, "FAIL T=%u Opus oracle max_abs=%.6g rmse=%.6g\n",
                         nt, opus_max_abs, opus_rmse);
            opus_oracle_failed = true;
        }
        for (unsigned head = 0; head < NH; ++head) {
            const double head_rmse =
                std::sqrt(opus_head_sum_sq[head] / ((size_t)nt * DV));
            if (nt % 256 == 0 &&
                (opus_head_max[head] > 2.0e-2 || head_rmse > 3.0e-3)) {
                std::fprintf(stderr,
                             "FAIL T=%u Opus head=%u coverage max_abs=%.6g rmse=%.6g\n",
                             nt, head, opus_head_max[head], head_rmse);
                opus_oracle_failed = true;
            }
        }

        launch(init_materialize, GRID, 256, mi_args);
        launch(materialize_q, GRID, 256, qproj_args);
        launch(materialize_kv, GRID, 256, kvproj_args);
        launch(pack, GRID, 256, pack_args);
        CK(hipDeviceSynchronize());
        std::vector<uint16_t> hkv(qrows * (DV * 2)), hkrope((size_t)nt * DR);
        std::vector<uint16_t> hk(qrows * DK_MAT), hv(qrows * DV);
        CK(hipMemcpy(hkv.data(), kvproj, hkv.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hkrope.data(), krope, hkrope.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hk.data(), kproj, hk.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hv.data(), vproj, hv.size() * 2, hipMemcpyDeviceToHost));
        for (size_t row = 0; row < qrows; ++row) {
            for (unsigned d = 0; d < DK_MAT; ++d) {
                const uint16_t want = d < 128 ? hkv[row * 256 + d]
                                               : hkrope[(row / NH) * DR + d - 128];
                if (hk[row * DK_MAT + d] != want) {
                    std::fprintf(stderr, "FAIL T=%u pack K row=%zu d=%u\n", nt, row, d);
                    return 5;
                }
            }
            for (unsigned d = 0; d < DV; ++d) {
                if (hv[row * DV + d] != hkv[row * 256 + 128 + d]) {
                    std::fprintf(stderr, "FAIL T=%u pack V row=%zu d=%u\n", nt, row, d);
                    return 5;
                }
            }
        }

        std::vector<double> ta, tm, tl, tw, tfull, tfull_abs;
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
            launch(absorb_q, GRID, 256, qabs_proj_args);
            launch(absorb_qrope, GRID, 256, qrope_proj_args);
            launch(absorbed, GRID, 256, abs_args);
            launch(fold, GRID, 256, fold_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            tfull_abs.push_back(ms * 1000.0);
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

            launch(flush, GRID, 256, flush_args);
            CK(hipDeviceSynchronize());
            CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(materialized_lds, GRID, 256, lds_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            tl.push_back(ms * 1000.0);
            CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));

            launch(flush, GRID, 256, flush_args);
            CK(hipDeviceSynchronize());
            CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(opus, opus_grid, 512, opus_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            tw.push_back(ms * 1000.0);
            CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));

            launch(flush, GRID, 256, flush_args);
            CK(hipDeviceSynchronize());
            CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
            CK(hipEventRecord(begin));
            launch(materialize_q, GRID, 256, qproj_args);
            launch(materialize_kv, GRID, 256, kvproj_args);
            launch(pack, GRID, 256, pack_args);
            launch(opus, opus_grid, 512, full_args);
            CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
            ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
            tfull.push_back(ms * 1000.0);
            CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
        }
        const double a = median(ta), m = median(tm), l = median(tl), w = median(tw),
                     full_us = median(tfull), full_abs_us = median(tfull_abs);
        std::printf("T=%u H=%u absorbed+fold=%.3f us materialized-attn=%.3f us "
                    "materialized-lds=%.3f us opus=%.3f us speedup=%.3fx "
                    "oracle(max_abs=%.6g rmse=%.6g bad=%zu/%zu) "
                    "lds-oracle(max_abs=%.6g rmse=%.6g bad=%zu/%zu) "
                    "opus-oracle(max_abs=%.6g rmse=%.6g heads=%u flat-grid-mismatch=%zu) "
                    "full-absorbed=%.3f us "
                    "full-materialized=%.3f us "
                    "full-vs-absorbed=%.3fx\n",
                    nt, NH, a, m, l, w, a / w, max_abs, rmse, bad, ha.size(), lds_max_abs,
                    lds_rmse, lds_bad, ha.size(), opus_max_abs, opus_rmse, NH,
                    flat_grid_mismatch, full_abs_us, full_us, full_abs_us / full_us);

        hipFree(qabs); hipFree(qrope); hipFree(ckv); hipFree(krope); hipFree(qmat);
        hipFree(kmat); hipFree(vmat); hipFree(wuv); hipFree(oabs); hipFree(omat); hipFree(olds);
        hipFree(oopus); hipFree(oupstream);
        hipFree(opart_abs); hipFree(mlpart_abs); hipFree(opart_mat); hipFree(mlpart_mat);
        hipFree(kv_len);
        hipFree(qlat); hipFree(qw); hipFree(kvw); hipFree(qaw); hipFree(qrw);
        hipFree(qproj); hipFree(kvproj);
        hipFree(kproj); hipFree(vproj); hipFree(ofull);
    }
    hipFree(flush_in); hipFree(flush_out);
    hipModuleUnload(mod);
    hipModuleUnload(opus_mod);
    hipModuleUnload(pack_mod);
    hipModuleUnload(upstream_mod);
    return opus_oracle_failed ? 4 : 0;
}
