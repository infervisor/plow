// Build:
// nix develop --command hipcc -O2 -Werror runtime/bench/amd/k3_mla_decode_exact.cpp
//   -o /tmp/k3_mla_decode_exact
// Run:
// nix develop --command perf-data/tools/gpulease -n 1 k3-mla-exact
//   /tmp/k3_mla_decode_exact /tmp/k3_mla_decode_exact.co
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

namespace {
constexpr unsigned Threads = 512, Heads = 12, DK = 512, DR = 64, V = 128;
constexpr unsigned Stride = 32768, Context = 8192, NSplit = 64;

struct Buffers {
    float *opart{}, *mlpart{};
    uint16_t *qabs{}, *qrope{}, *ckv{}, *krope{}, *olat{}, *wuv{}, *out{}, *out_fused{};
    int* kvlen{};
};

double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

double time_launch(const std::vector<std::pair<hipFunction_t, std::vector<void*>>>& launches,
                   const std::vector<dim3>& grids) {
    auto run = [&] {
        for (size_t i = 0; i < launches.size(); ++i)
            CK(hipModuleLaunchKernel(launches[i].first, grids[i].x, 1, 1, Threads, 1, 1, 0,
                                     nullptr, const_cast<void**>(launches[i].second.data()), nullptr));
    };
    for (int i = 0; i < 5; ++i) run();
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
    std::vector<double> samples;
    for (int s = 0; s < 21; ++s) {
        CK(hipEventRecord(begin, nullptr));
        for (int r = 0; r < 10; ++r) run();
        CK(hipEventRecord(end, nullptr)); CK(hipEventSynchronize(end));
        float ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms * 100.0); // ms / 10 -> microseconds
    }
    CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
    return median(std::move(samples));
}

float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16; float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

void run_case(hipModule_t module, unsigned batch) {
    hipFunction_t flash, flash_gf6, merge, fold, fused, fill;
    CK(hipModuleGetFunction(&flash, module, "k_mla_flash_bf16"));
    CK(hipModuleGetFunction(&flash_gf6, module, "k_mla_flash_bf16_gf6"));
    CK(hipModuleGetFunction(&merge, module, "k_mla_merge_bf16"));
    CK(hipModuleGetFunction(&fold, module, "k_mla_fold_bf16"));
    CK(hipModuleGetFunction(&fused, module, batch == 1 ? "k_mla_merge_fold_v32" :
                                                        "k_mla_merge_fold_v128"));
    CK(hipModuleGetFunction(&fill, module, "k_fill_bf16"));

    Buffers b;
    const size_t bh = (size_t)batch * Heads;
    CK(hipMalloc(&b.opart, bh * NSplit * DK * sizeof(float)));
    CK(hipMalloc(&b.mlpart, bh * NSplit * 2 * sizeof(float)));
    CK(hipMalloc(&b.qabs, bh * DK * sizeof(uint16_t)));
    CK(hipMalloc(&b.qrope, bh * DR * sizeof(uint16_t)));
    CK(hipMalloc(&b.ckv, (size_t)batch * Stride * DK * sizeof(uint16_t)));
    CK(hipMalloc(&b.krope, (size_t)batch * Stride * DR * sizeof(uint16_t)));
    CK(hipMalloc(&b.olat, bh * DK * sizeof(uint16_t)));
    CK(hipMalloc(&b.wuv, (size_t)Heads * DK * V * sizeof(uint16_t)));
    CK(hipMalloc(&b.out, bh * V * sizeof(uint16_t)));
    CK(hipMalloc(&b.out_fused, bh * V * sizeof(uint16_t)));
    CK(hipMalloc(&b.kvlen, batch * sizeof(int)));
    std::vector<int> lengths(batch, Context);
    CK(hipMemcpy(b.kvlen, lengths.data(), batch * sizeof(int), hipMemcpyHostToDevice));

    auto fill_one = [&](uint16_t* ptr, size_t n, unsigned seed) {
        void* args[] = {&ptr, &n, &seed};
        CK(hipModuleLaunchKernel(fill, 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    };
    fill_one(b.qabs, bh * DK, 1); fill_one(b.qrope, bh * DR, 2);
    fill_one(b.ckv, (size_t)batch * Stride * DK, 3);
    fill_one(b.krope, (size_t)batch * Stride * DR, 4);
    fill_one(b.wuv, (size_t)Heads * DK * V, 5);
    CK(hipDeviceSynchronize());

    void* flash_args[] = {&b.opart, &b.mlpart, &b.qabs, &b.qrope, &b.ckv, &b.krope,
                          &b.kvlen, &batch, const_cast<unsigned*>(&Stride),
                          const_cast<unsigned*>(&NSplit)};
    void* merge_args[] = {&b.olat, &b.opart, &b.mlpart, &batch,
                          const_cast<unsigned*>(&NSplit)};
    void* fold_args[] = {&b.out, &b.olat, &b.wuv, &batch};
    void* fused_args[] = {&b.out_fused, &b.opart, &b.mlpart, &b.wuv, &batch,
                          const_cast<unsigned*>(&NSplit)};
    auto spec = [](hipFunction_t f, void** args, size_t n) {
        return std::make_pair(f, std::vector<void*>(args, args + n));
    };
    const unsigned flash_grid = batch == 1 ? 192 : 256;
    const unsigned bh_grid = batch * Heads;
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> fs{spec(flash, flash_args, 10)};
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> fs_gf6{
        spec(flash_gf6, flash_args, 10)};
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> ms{spec(merge, merge_args, 5)};
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> ws{spec(fold, fold_args, 4)};
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> core{
        spec(flash, flash_args, 10), spec(merge, merge_args, 5)};
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>> prod{
        spec(flash, flash_args, 10), spec(fused, fused_args, 6)};
    const double flash_us = time_launch(fs, {dim3(flash_grid)});
    const double flash_gf6_us = time_launch(fs_gf6, {dim3(flash_grid)});
    const double merge_us = time_launch(ms, {dim3(bh_grid)});
    const double fold_us = time_launch(ws, {dim3(bh_grid)});
    const double core_us = time_launch(core, {dim3(flash_grid), dim3(bh_grid)});
    const double prod_us = time_launch(prod, {dim3(flash_grid), dim3(256)});
    const double compact_us = time_launch(prod, {dim3(flash_grid), dim3(96)});

    CK(hipModuleLaunchKernel(flash, flash_grid, 1, 1, Threads, 1, 1, 0, nullptr, flash_args, nullptr));
    CK(hipModuleLaunchKernel(merge, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr, merge_args, nullptr));
    CK(hipModuleLaunchKernel(fold, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr, fold_args, nullptr));
    CK(hipModuleLaunchKernel(fused, 256, 1, 1, Threads, 1, 1, 0, nullptr, fused_args, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> a(bh * V), c(bh * V);
    CK(hipMemcpy(a.data(), b.out, a.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(c.data(), b.out_fused, c.size() * 2, hipMemcpyDeviceToHost));
    CK(hipModuleLaunchKernel(flash_gf6, flash_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                             flash_args, nullptr));
    CK(hipModuleLaunchKernel(fused, 256, 1, 1, Threads, 1, 1, 0, nullptr, fused_args, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> g(bh * V);
    CK(hipMemcpy(g.data(), b.out_fused, g.size() * 2, hipMemcpyDeviceToHost));
    double e2 = 0, g2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = bf16_to_float(a[i]), y = bf16_to_float(c[i]), z = bf16_to_float(g[i]);
        e2 += (x - y) * (x - y); r2 += x * x;
        g2 += (x - z) * (x - z);
    }
    std::printf("B%u,flash_us=%.3f,flash_gf6_us=%.3f,merge_us=%.3f,attention_core_us=%.3f,"
                "bf16_wuv_us=%.3f,production_fused_chain_us=%.3f,compact96_chain_us=%.3f,"
                "fused_rel_l2=%.6g,gf6_rel_l2=%.6g\n",
                batch, flash_us, flash_gf6_us, merge_us, core_us, fold_us, prod_us, compact_us,
                std::sqrt(e2 / std::max(r2, 1e-30)), std::sqrt(g2 / std::max(r2, 1e-30)));

    CK(hipFree(b.opart)); CK(hipFree(b.mlpart)); CK(hipFree(b.qabs)); CK(hipFree(b.qrope));
    CK(hipFree(b.ckv)); CK(hipFree(b.krope)); CK(hipFree(b.olat)); CK(hipFree(b.wuv));
    CK(hipFree(b.out)); CK(hipFree(b.out_fused)); CK(hipFree(b.kvlen));
}
} // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_mla_decode_exact.co";
    CK(hipInit(0));
    hipModule_t module; CK(hipModuleLoad(&module, object));
    run_case(module, 1); run_case(module, 8);
    CK(hipModuleUnload(module));
}
