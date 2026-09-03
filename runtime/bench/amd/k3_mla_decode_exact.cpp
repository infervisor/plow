// Build: nix develop --command hipcc -O2 -Werror runtime/bench/amd/k3_mla_decode_exact.cpp -o /tmp/k3_mla_decode_exact
// Run: nix develop --command perf-data/tools/gpulease -n 1 k3-mla-exact /tmp/k3_mla_decode_exact /tmp/k3_mla_decode_exact.co /tmp/k3_mla_oracle [threads] [optional-gf6-object]
// Oracle: nix develop .#quantize --command python3 runtime/bench/amd/k3_mla_oracle.py /tmp/k3_mla_oracle
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
unsigned Threads = 512;
constexpr unsigned Heads = 12, DK = 512, DR = 64, V = 128;
constexpr unsigned Stride = 32768, Context = 8192, NSplit = 64;

struct alignas(8) MlaDecodeGf6Args {
    uint64_t opart, mlpart, qabs, qrope, ckv, krope, kv_len;
    uint32_t n_batch, n_head, kv_stride, window;
    float scale;
    uint32_t nsplit, kv_mask;
};
static_assert(sizeof(MlaDecodeGf6Args) == 88);

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
                   const std::vector<dim3>& grids, const char* sample_label = nullptr) {
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
    if (sample_label) {
        std::printf("SAMPLES_US,%s", sample_label);
        for (double value : samples) std::printf(",%.6f", value);
        std::printf("\n");
    }
    return median(std::move(samples));
}

struct PairedTiming {
    double control_us;
    double candidate_us;
    unsigned candidate_wins;
};

PairedTiming time_interleaved(
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>>& control,
    const std::vector<dim3>& control_grids,
    const std::vector<std::pair<hipFunction_t, std::vector<void*>>>& candidate,
    const std::vector<dim3>& candidate_grids) {
    auto run = [](const std::vector<std::pair<hipFunction_t, std::vector<void*>>>& launches,
                  const std::vector<dim3>& grids) {
        for (size_t i = 0; i < launches.size(); ++i)
            CK(hipModuleLaunchKernel(launches[i].first, grids[i].x, 1, 1, Threads, 1, 1, 0,
                                     nullptr, const_cast<void**>(launches[i].second.data()), nullptr));
    };
    for (int i = 0; i < 5; ++i) {
        run(control, control_grids);
        run(candidate, candidate_grids);
    }
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
    auto measure = [&](const auto& launches, const auto& grids) {
        CK(hipEventRecord(begin, nullptr));
        for (int repeat = 0; repeat < 10; ++repeat) run(launches, grids);
        CK(hipEventRecord(end, nullptr)); CK(hipEventSynchronize(end));
        float ms = 0; CK(hipEventElapsedTime(&ms, begin, end));
        return static_cast<double>(ms) * 100.0;
    };
    std::vector<double> control_samples, candidate_samples;
    unsigned candidate_wins = 0;
    for (int round = 0; round < 21; ++round) {
        double control_us, candidate_us;
        if ((round & 1) == 0) {
            control_us = measure(control, control_grids);
            candidate_us = measure(candidate, candidate_grids);
        } else {
            candidate_us = measure(candidate, candidate_grids);
            control_us = measure(control, control_grids);
        }
        control_samples.push_back(control_us);
        candidate_samples.push_back(candidate_us);
        candidate_wins += candidate_us < control_us;
    }
    CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
    return {median(std::move(control_samples)), median(std::move(candidate_samples)),
            candidate_wins};
}

float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16; float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

void dump_bf16(const std::string& path, const uint16_t* device, size_t count) {
    std::vector<uint16_t> host(count);
    CK(hipMemcpy(host.data(), device, count * sizeof(uint16_t), hipMemcpyDeviceToHost));
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file || std::fwrite(host.data(), sizeof(uint16_t), count, file) != count) {
        std::fprintf(stderr, "failed to write %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(file);
}

void run_case(hipModule_t module, hipModule_t external_module, unsigned batch, const char* dump_prefix) {
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
    if (dump_prefix) {
        dump_bf16(std::string(dump_prefix) + ".b" + std::to_string(batch) + ".ns64.olat.bf16",
                  b.olat, bh * DK);
    }
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

    {
        unsigned candidate_ns = 32;
        void* control_args[] = {&b.opart, &b.mlpart, &b.qabs, &b.qrope, &b.ckv, &b.krope,
                                &b.kvlen, &batch, const_cast<unsigned*>(&Stride),
                                &candidate_ns};
        void* candidate_args[] = {&b.opart, &b.mlpart, &b.qabs, &b.qrope, &b.ckv, &b.krope,
                                  &b.kvlen, &batch, const_cast<unsigned*>(&Stride),
                                  &candidate_ns};
        void* candidate_merge_args[] = {&b.olat, &b.opart, &b.mlpart, &batch, &candidate_ns};
        const unsigned control_grid = std::min(256u, batch * (Heads / 4) * candidate_ns);
        const unsigned candidate_grid = std::min(256u, batch * (Heads / 6) * candidate_ns);
        const std::vector<std::pair<hipFunction_t, std::vector<void*>>> control_core{
            spec(flash, control_args, 10), spec(merge, candidate_merge_args, 5)};
        const std::vector<std::pair<hipFunction_t, std::vector<void*>>> candidate_core{
            spec(flash_gf6, candidate_args, 10), spec(merge, candidate_merge_args, 5)};
        const PairedTiming paired = time_interleaved(
            control_core, {dim3(control_grid), dim3(bh_grid)},
            candidate_core, {dim3(candidate_grid), dim3(bh_grid)});
        CK(hipModuleLaunchKernel(flash_gf6, candidate_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                 candidate_args, nullptr));
        CK(hipModuleLaunchKernel(merge, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                 candidate_merge_args, nullptr));
        CK(hipDeviceSynchronize());
        if (dump_prefix) {
            dump_bf16(std::string(dump_prefix) + ".b" + std::to_string(batch) +
                          ".gf6.ns32.olat.bf16",
                      b.olat, bh * DK);
        }
        if (external_module) {
            hipFunction_t external_flash;
            CK(hipModuleGetFunction(&external_flash, external_module, "plow_mla_decode_gf6"));
            MlaDecodeGf6Args external_args{
                reinterpret_cast<uint64_t>(b.opart), reinterpret_cast<uint64_t>(b.mlpart),
                reinterpret_cast<uint64_t>(b.qabs), reinterpret_cast<uint64_t>(b.qrope),
                reinterpret_cast<uint64_t>(b.ckv), reinterpret_cast<uint64_t>(b.krope),
                reinterpret_cast<uint64_t>(b.kvlen), batch, Heads, Stride, 0,
                0.07216878364870322f, candidate_ns, 0xFFFFFFFFu};
            void* external_flash_args[] = {&external_args};
            const std::vector<std::pair<hipFunction_t, std::vector<void*>>> external_core{
                spec(external_flash, external_flash_args, 1),
                spec(merge, candidate_merge_args, 5)};
            const PairedTiming external_paired = time_interleaved(
                control_core, {dim3(control_grid), dim3(bh_grid)},
                external_core, {dim3(candidate_grid), dim3(bh_grid)});
            CK(hipModuleLaunchKernel(external_flash, candidate_grid, 1, 1, 512, 1, 1, 0,
                                     nullptr, external_flash_args, nullptr));
            CK(hipModuleLaunchKernel(merge, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                     candidate_merge_args, nullptr));
            CK(hipDeviceSynchronize());
            if (dump_prefix) {
                dump_bf16(std::string(dump_prefix) + ".b" + std::to_string(batch) +
                              ".external_gf6.ns32.olat.bf16",
                          b.olat, bh * DK);
            }
            std::printf("EXTERNAL_PAIR,B%u,ns=32,gf4_core_us=%.3f,external_gf6_core_us=%.3f,"
                        "external_speedup=%.4fx,external_wins=%u/21\n",
                        batch, external_paired.control_us, external_paired.candidate_us,
                        external_paired.control_us / external_paired.candidate_us,
                        external_paired.candidate_wins);
        }
        const double speedup = paired.control_us / paired.candidate_us;
        std::printf("PAIR,B%u,ns=32,gf4_grid=%u,gf6_grid=%u,gf4_core_us=%.3f,"
                    "gf6_core_us=%.3f,gf6_speedup=%.4fx,gf6_wins=%u/21\n",
                    batch, control_grid, candidate_grid, paired.control_us,
                    paired.candidate_us, speedup, paired.candidate_wins);
    }

    const std::vector<unsigned> split_sweep = batch == 1
        ? std::vector<unsigned>{32, 41, 64}
        : std::vector<unsigned>{16, 24, 32, 64};
    for (unsigned sweep_ns : split_sweep) {
        void* sweep_flash_args[] = {&b.opart, &b.mlpart, &b.qabs, &b.qrope, &b.ckv, &b.krope,
                                    &b.kvlen, &batch, const_cast<unsigned*>(&Stride), &sweep_ns};
        void* sweep_merge_args[] = {&b.olat, &b.opart, &b.mlpart, &batch, &sweep_ns};
        const unsigned sweep_grid = std::min(256u, batch * (Heads / 4) * sweep_ns);
        const std::vector<std::pair<hipFunction_t, std::vector<void*>>> sweep_flash{
            spec(flash, sweep_flash_args, 10)};
        const std::vector<std::pair<hipFunction_t, std::vector<void*>>> sweep_core{
            spec(flash, sweep_flash_args, 10), spec(merge, sweep_merge_args, 5)};
        const double sweep_flash_us = time_launch(sweep_flash, {dim3(sweep_grid)});
        const std::string sample_label =
            "B" + std::to_string(batch) + "_ns" + std::to_string(sweep_ns) + "_attention_core";
        const double sweep_core_us =
            time_launch(sweep_core, {dim3(sweep_grid), dim3(bh_grid)}, sample_label.c_str());

        CK(hipModuleLaunchKernel(flash, sweep_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                 sweep_flash_args, nullptr));
        CK(hipModuleLaunchKernel(merge, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                 sweep_merge_args, nullptr));
        CK(hipModuleLaunchKernel(fold, bh_grid, 1, 1, Threads, 1, 1, 0, nullptr,
                                 fold_args, nullptr));
        CK(hipDeviceSynchronize());
        if (dump_prefix && (sweep_ns == 32 || sweep_ns == 64)) {
            dump_bf16(std::string(dump_prefix) + ".b" + std::to_string(batch) + ".ns" +
                          std::to_string(sweep_ns) + ".olat.bf16",
                      b.olat, bh * DK);
        }
        std::vector<uint16_t> candidate(bh * V);
        CK(hipMemcpy(candidate.data(), b.out, candidate.size() * 2, hipMemcpyDeviceToHost));
        double candidate_e2 = 0, max_abs = 0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double x = bf16_to_float(a[i]);
            const double y = bf16_to_float(candidate[i]);
            nonfinite += !std::isfinite(y);
            const double error = std::abs(x - y);
            candidate_e2 += error * error;
            max_abs = std::max(max_abs, error);
        }
        std::printf("SWEEP,B%u,ns=%u,grid=%u,flash_us=%.3f,attention_core_us=%.3f,"
                    "rel_l2_vs_ns64=%.6g,max_abs=%.6g,nonfinite=%zu\n",
                    batch, sweep_ns, sweep_grid, sweep_flash_us, sweep_core_us,
                    std::sqrt(candidate_e2 / std::max(r2, 1e-30)), max_abs, nonfinite);
    }

    CK(hipFree(b.opart)); CK(hipFree(b.mlpart)); CK(hipFree(b.qabs)); CK(hipFree(b.qrope));
    CK(hipFree(b.ckv)); CK(hipFree(b.krope)); CK(hipFree(b.olat)); CK(hipFree(b.wuv));
    CK(hipFree(b.out)); CK(hipFree(b.out_fused)); CK(hipFree(b.kvlen));
}
} // namespace

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_mla_decode_exact.co";
    const char* dump_prefix = argc > 2 ? argv[2] : nullptr;
    Threads = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 512;
    if (Threads != 256 && Threads != 384 && Threads != 512) {
        std::fprintf(stderr, "threads must be 256, 384, or 512\n");
        return 2;
    }
    CK(hipInit(0));
    hipModule_t module; CK(hipModuleLoad(&module, object));
    hipModule_t external_module = nullptr;
    if (argc > 4) CK(hipModuleLoad(&external_module, argv[4]));
    run_case(module, external_module, 1, dump_prefix);
    run_case(module, external_module, 8, dump_prefix);
    if (external_module) CK(hipModuleUnload(external_module));
    CK(hipModuleUnload(module));
}
