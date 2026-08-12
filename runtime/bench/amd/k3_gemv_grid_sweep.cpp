/* Grid sweep for k_packet in k3_gemvbf16_bench.hip at the current Kimi-K3 B1 shapes. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static constexpr unsigned THREADS = 512;
static constexpr size_t ARENA = 3ull << 30;
static constexpr size_t TARGET_STREAM = 1536ull << 20;

struct Shape {
    const char* name;
    unsigned n;
    unsigned k;
    unsigned blocks;
    unsigned instances;
};

static const Shape SHAPES[] = {
    {"o_proj", 7168, 1536, 128, 93},       {"router", 896, 7168, 128, 92},
    {"routed_up", 896, 3584, 128, 92},     {"shared_down", 7168, 768, 128, 92},
    {"latent_down", 3584, 7168, 128, 92},  {"kda_f_b", 1536, 128, 128, 69},
    {"kda_f_a", 128, 7168, 128, 69},       {"kda_b", 12, 7168, 12, 69},
    {"mla_q_a_g", 1536, 7168, 128, 48},    {"mla_k_rope_down", 64, 7168, 64, 24},
    {"mla_q_rope", 768, 1536, 128, 24},    {"mla_q_absorb", 6144, 1536, 128, 24},
    {"mla_kv_a", 512, 7168, 128, 24},      {"output_attn_res", 4224, 7168, 128, 2},
    {"lm_head", 163840, 7168, 304, 1},     {"output_attn_proj", 7168, 4224, 128, 1},
};

static const unsigned GRIDS[] = {12,  24,  32,  48,  64,  76,  96,  112, 128,
                                 152, 176, 200, 224, 256, 293, 299, 302, 304};

static double time_kernel(hipFunction_t fn, unsigned grid, void** args, int samples) {
    for (int i = 0; i < 2; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, THREADS, 1, 1, 0, 0, args, nullptr));
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    std::vector<double> times;
    times.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        CK(hipEventRecord(begin));
        CK(hipModuleLaunchKernel(fn, grid, 1, 1, THREADS, 1, 1, 0, 0, args, nullptr));
        CK(hipEventRecord(end));
        CK(hipEventSynchronize(end));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, begin, end));
        times.push_back(ms * 1000.0);
    }
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static void launch_once(hipFunction_t fn, unsigned grid, void** args) {
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, THREADS, 1, 1, 0, 0, args, nullptr));
    CK(hipDeviceSynchronize());
}

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_gemv_grid.co";
    const int samples = argc > 2 ? std::atoi(argv[2]) : 15;
    if (samples < 3 || (samples & 1) == 0) {
        std::fprintf(stderr, "samples must be an odd integer >= 3\n");
        return 2;
    }

    CK(hipInit(0));
    hipModule_t module;
    hipFunction_t packet;
    CK(hipModuleLoad(&module, object));
    CK(hipModuleGetFunction(&packet, module, "k_packet"));

    unsigned short *weights, *out_control, *out_candidate, *x;
    CK(hipMalloc(&weights, ARENA));
    CK(hipMalloc(&out_control, 1u << 22));
    CK(hipMalloc(&out_candidate, 1u << 22));
    CK(hipMalloc(&x, 1u << 22));
    const size_t fill_bytes = 64u << 20;
    std::vector<unsigned short> fill(fill_bytes / 2);
    unsigned state = 0x12345677u;
    for (auto& value : fill) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<unsigned short>(0x3B00u | ((state >> 20) & 0x7Fu));
    }
    for (size_t offset = 0; offset < ARENA; offset += fill_bytes)
        CK(hipMemcpy(reinterpret_cast<char*>(weights) + offset, fill.data(),
                     std::min(fill_bytes, ARENA - offset), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, fill.data(), 1u << 22, hipMemcpyHostToDevice));

    FILE* json = nullptr;
    if (const char* path = std::getenv("PLOW_K3_GEMV_GRID_JSONL")) {
        json = std::fopen(path, "w");
        if (!json) {
            std::perror(path);
            return 2;
        }
    }

    double control_total_ms = 0.0;
    double best_total_ms = 0.0;
    std::printf("%-18s %7s %6s %5s %6s %11s %9s %8s\n", "shape", "N", "K", "base",
                "best", "base_us", "best_us", "speedup");
    for (const Shape& shape : SHAPES) {
        const size_t slab = static_cast<size_t>(shape.n) * shape.k * 2;
        unsigned nrep = static_cast<unsigned>(std::max<size_t>(1, TARGET_STREAM / slab));
        nrep = std::min<unsigned>(nrep, 16384);
        while (static_cast<size_t>(nrep) * slab > ARENA) --nrep;
        void* args[] = {&out_control, &x, &weights, &nrep, const_cast<unsigned*>(&shape.n),
                        const_cast<unsigned*>(&shape.k)};

        double values[sizeof(GRIDS) / sizeof(GRIDS[0])];
        std::fill(std::begin(values), std::end(values), 1.0e30);
        for (size_t i = 0; i < std::size(GRIDS); ++i) {
            if (GRIDS[i] > shape.n) continue;
            values[i] = std::min(values[i], time_kernel(packet, GRIDS[i], args, samples) / nrep);
        }
        for (size_t i = std::size(GRIDS); i-- > 0;) {
            if (GRIDS[i] > shape.n) continue;
            values[i] = std::min(values[i], time_kernel(packet, GRIDS[i], args, samples) / nrep);
        }

        size_t best = 0;
        while (GRIDS[best] > shape.n) ++best;
        for (size_t i = best + 1; i < std::size(GRIDS); ++i)
            if (GRIDS[i] <= shape.n && values[i] < values[best]) best = i;
        size_t base = 0;
        while (base < std::size(GRIDS) && GRIDS[base] != shape.blocks) ++base;
        if (base == std::size(GRIDS)) {
            std::fprintf(stderr, "missing control grid %u\n", shape.blocks);
            return 2;
        }

        unsigned one = 1;
        void* control_args[] = {&out_control, &x, &weights, &one,
                                const_cast<unsigned*>(&shape.n), const_cast<unsigned*>(&shape.k)};
        void* candidate_args[] = {&out_candidate, &x, &weights, &one,
                                  const_cast<unsigned*>(&shape.n), const_cast<unsigned*>(&shape.k)};
        CK(hipMemset(out_control, 0, shape.n * 2));
        CK(hipMemset(out_candidate, 0, shape.n * 2));
        launch_once(packet, shape.blocks, control_args);
        launch_once(packet, GRIDS[best], candidate_args);
        std::vector<unsigned short> control(shape.n), candidate(shape.n);
        CK(hipMemcpy(control.data(), out_control, shape.n * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(candidate.data(), out_candidate, shape.n * 2, hipMemcpyDeviceToHost));
        if (control != candidate) {
            std::fprintf(stderr, "FAIL %s: grid %u is not bit-exact with grid %u\n", shape.name,
                         GRIDS[best], shape.blocks);
            return 3;
        }

        control_total_ms += values[base] * shape.instances / 1000.0;
        best_total_ms += values[best] * shape.instances / 1000.0;
        std::printf("%-18s %7u %6u %5u %6u %11.3f %9.3f %8.4f\n", shape.name, shape.n,
                    shape.k, shape.blocks, GRIDS[best], values[base], values[best],
                    values[base] / values[best]);
        if (json) {
            for (size_t i = 0; i < std::size(GRIDS); ++i) {
                if (GRIDS[i] > shape.n) continue;
                std::fprintf(json,
                             "{\"schema\":\"plow.k3-gemv-grid.v1\",\"shape\":\"%s\","
                             "\"n\":%u,\"k\":%u,\"grid\":%u,\"control_grid\":%u,"
                             "\"instances\":%u,\"nrep\":%u,\"us\":%.6f,\"correct\":true}\n",
                             shape.name, shape.n, shape.k, GRIDS[i], shape.blocks, shape.instances,
                             nrep, values[i]);
            }
        }
    }
    if (json) std::fclose(json);
    std::printf("weighted packet-body projection: control %.3f ms, per-shape best %.3f ms, "
                "speedup %.4fx, saving %.3f ms\n",
                control_total_ms, best_total_ms, control_total_ms / best_total_ms,
                control_total_ms - best_total_ms);
    return 0;
}
