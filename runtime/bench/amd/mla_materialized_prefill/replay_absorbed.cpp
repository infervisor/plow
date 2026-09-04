#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

static std::vector<uint8_t> read_exact(const char* path, size_t expected) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || static_cast<size_t>(in.tellg()) != expected) {
        std::fprintf(stderr, "invalid payload size: %s (expected %zu)\n", path, expected);
        std::exit(2);
    }
    std::vector<uint8_t> out(expected);
    in.seekg(0); in.read(reinterpret_cast<char*>(out.data()), expected);
    if (!in) std::exit(2);
    return out;
}

static void write_all(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!out) std::exit(2);
}

static void launch(hipFunction_t fn, unsigned grid, void** args) {
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
}

int main(int argc, char** argv) {
    if (argc != 15) {
        std::fprintf(stderr, "usage: %s KERNEL QLAT KVLAT KROPE QW KVW OUT T H QL KL DR DV SCALE\n", argv[0]);
        return 2;
    }
    unsigned t = std::atoi(argv[8]), h = std::atoi(argv[9]);
    unsigned ql = std::atoi(argv[10]), kl = std::atoi(argv[11]);
    unsigned dr = std::atoi(argv[12]), dv = std::atoi(argv[13]);
    float scale = std::strtof(argv[14], nullptr);
    constexpr unsigned dq = 192, nope = 128;
    if (!t || !h || ql != 1536 || kl != 512 || dr != 64 || dv != 128 || !(scale > 0)) return 2;
    auto hql = read_exact(argv[2], static_cast<size_t>(t) * ql * 2);
    auto hkl = read_exact(argv[3], static_cast<size_t>(t) * kl * 2);
    auto hkr = read_exact(argv[4], static_cast<size_t>(t) * dr * 2);
    auto hqw = read_exact(argv[5], static_cast<size_t>(h) * dq * ql * 2);
    auto hkw = read_exact(argv[6], static_cast<size_t>(h) * (nope + dv) * kl * 2);

    CK(hipInit(0));
    hipModule_t module; CK(hipModuleLoad(&module, argv[1]));
    hipFunction_t derive, qproj, rproj, attention, fold;
    CK(hipModuleGetFunction(&derive, module, "k_derive_absorbed_weights"));
    CK(hipModuleGetFunction(&qproj, module, "k_absorb_q"));
    CK(hipModuleGetFunction(&rproj, module, "k_absorb_qrope"));
    CK(hipModuleGetFunction(&attention, module, "k_absorbed"));
    CK(hipModuleGetFunction(&fold, module, "k_absorbed_fold"));
    void *qlat, *kvlat, *krope, *qw, *kvw, *qabsw, *qropew, *wuv;
    void *qabs, *qrope, *opart, *mlpart, *out;
    const size_t rows = static_cast<size_t>(t) * h;
    CK(hipMalloc(&qlat, hql.size())); CK(hipMalloc(&kvlat, hkl.size()));
    CK(hipMalloc(&krope, hkr.size())); CK(hipMalloc(&qw, hqw.size())); CK(hipMalloc(&kvw, hkw.size()));
    CK(hipMalloc(&qabsw, static_cast<size_t>(h) * kl * ql * 2));
    CK(hipMalloc(&qropew, static_cast<size_t>(h) * dr * ql * 2));
    CK(hipMalloc(&wuv, static_cast<size_t>(h) * kl * dv * 2));
    CK(hipMalloc(&qabs, rows * kl * 2)); CK(hipMalloc(&qrope, rows * dr * 2));
    CK(hipMalloc(&opart, rows * kl * 4)); CK(hipMalloc(&mlpart, rows * 2 * 4));
    CK(hipMalloc(&out, rows * dv * 2));
    CK(hipMemcpy(qlat, hql.data(), hql.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(kvlat, hkl.data(), hkl.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(krope, hkr.data(), hkr.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(qw, hqw.data(), hqw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(kvw, hkw.data(), hkw.size(), hipMemcpyHostToDevice));
    void* derive_args[] = {&qabsw, &qropew, &wuv, &qw, &kvw, &h};
    launch(derive, 256, derive_args); CK(hipDeviceSynchronize());
    void* qargs[] = {&qabs, &qlat, &qabsw, &t, &h};
    void* rargs[] = {&qrope, &qlat, &qropew, &t, &h};
    int *kv_len; CK(hipMalloc(&kv_len, sizeof(int)));
    int len = t; CK(hipMemcpy(kv_len, &len, sizeof(int), hipMemcpyHostToDevice));
    unsigned stride = t;
    void* aargs[] = {&opart, &mlpart, &qabs, &qrope, &kvlat, &krope, &kv_len,
                     &t, &h, &stride, &scale};
    void* fargs[] = {&out, &opart, &mlpart, &wuv, &t, &h};
    std::vector<uint8_t> first(rows * dv * 2), current(first.size());
    std::vector<float> timings;
    for (int repeat = 0; repeat < 3; ++repeat) {
        hipEvent_t begin, end; CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
        CK(hipEventRecord(begin));
        launch(qproj, 256, qargs); launch(rproj, 256, rargs);
        launch(attention, 256, aargs); launch(fold, 256, fargs);
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
        float ms = 0; CK(hipEventElapsedTime(&ms, begin, end)); timings.push_back(ms);
        CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
        CK(hipMemcpy(current.data(), out, current.size(), hipMemcpyDeviceToHost));
        write_all(std::string(argv[7]) + ".repeat-" + std::to_string(repeat) + ".bf16", current);
        if (repeat == 0) first = current;
        else if (current != first) return 3;
    }
    std::sort(timings.begin(), timings.end());
    std::printf("{\"absorbed_ms_median\":%.9g,\"adjacent_repeats_exact\":true}\n", timings[1]);
    hipFree(qlat); hipFree(kvlat); hipFree(krope); hipFree(qw); hipFree(kvw);
    hipFree(qabsw); hipFree(qropew); hipFree(wuv); hipFree(qabs); hipFree(qrope);
    hipFree(opart); hipFree(mlpart); hipFree(out); hipFree(kv_len); hipModuleUnload(module);
    return 0;
}
