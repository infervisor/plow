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

static std::vector<uint8_t> read_exact(const char* path, size_t expected) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || static_cast<size_t>(in.tellg()) != expected) {
        std::fprintf(stderr, "invalid payload size: %s (expected %zu)\n", path, expected);
        std::exit(2);
    }
    std::vector<uint8_t> out(expected);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), expected);
    if (!in) std::exit(2);
    return out;
}

static void write_all(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!out) std::exit(2);
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::fprintf(stderr, "usage: %s OPUS Q K V OUT_PREFIX T H DQ DV SCALE\n", argv[0]);
        return 2;
    }
    const int t = std::atoi(argv[6]), h = std::atoi(argv[7]);
    const int dq = std::atoi(argv[8]), dv = std::atoi(argv[9]);
    const float scale = std::strtof(argv[10], nullptr);
    if (t <= 0 || h <= 0 || dq <= 0 || dv <= 0 || !(scale > 0)) return 2;
    const size_t q_bytes = static_cast<size_t>(t) * h * dq * 2;
    const size_t v_bytes = static_cast<size_t>(t) * h * dv * 2;
    auto hq = read_exact(argv[2], q_bytes);
    auto hk = read_exact(argv[3], q_bytes);
    auto hv = read_exact(argv[4], v_bytes);

    CK(hipInit(0));
    hipModule_t module;
    hipFunction_t kernel;
    CK(hipModuleLoad(&module, argv[1]));
    CK(hipModuleGetFunction(&kernel, module, "plow_mla_materialized_hd192_v128_gfx950"));
    void *q, *k, *v, *o;
    CK(hipMalloc(&q, q_bytes)); CK(hipMalloc(&k, q_bytes));
    CK(hipMalloc(&v, v_bytes)); CK(hipMalloc(&o, v_bytes));
    CK(hipMemcpy(q, hq.data(), q_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(k, hk.data(), q_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(v, hv.data(), v_bytes, hipMemcpyHostToDevice));
    OpusArgs a{};
    a.q = q; a.k = k; a.v = v; a.o = o;
    a.b = 1; a.n = t; a.n_kv = t; a.h = h; a.h_kv = h;
    a.d_qk = dq; a.d_v = dv;
    a.sq_b = t * h * dq; a.sq_n = h * dq; a.sq_h = dq;
    a.sk_b = t * h * dq; a.sk_n = h * dq; a.sk_h = dq;
    a.sv_b = t * h * dv; a.sv_n = h * dv; a.sv_h = dv;
    a.so_b = t * h * dv; a.so_n = h * dv; a.so_h = dv;
    a.scale = scale;
    void* args[] = {&a};
    const unsigned grid = ((t + 255) / 256) * h;
    std::vector<uint8_t> first(v_bytes), current(v_bytes);
    std::vector<float> timings;
    for (int repeat = 0; repeat < 3; ++repeat) {
        hipEvent_t begin, end;
        CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
        CK(hipEventRecord(begin));
        CK(hipModuleLaunchKernel(kernel, grid, 1, 1, 512, 1, 1, 0, nullptr, args, nullptr));
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, begin, end));
        timings.push_back(ms);
        CK(hipEventDestroy(begin)); CK(hipEventDestroy(end));
        CK(hipMemcpy(current.data(), o, v_bytes, hipMemcpyDeviceToHost));
        write_all(std::string(argv[5]) + ".repeat-" + std::to_string(repeat) + ".bf16", current);
        if (repeat == 0) first = current;
        else if (current != first) {
            std::fprintf(stderr, "adjacent replay output differs at repeat %d\n", repeat);
            return 3;
        }
    }
    std::sort(timings.begin(), timings.end());
    std::printf("{\"attention_ms_median\":%.9g,\"adjacent_repeats_exact\":true}\n",
                timings[1]);
    hipFree(q); hipFree(k); hipFree(v); hipFree(o);
    hipModuleUnload(module);
    return 0;
}
