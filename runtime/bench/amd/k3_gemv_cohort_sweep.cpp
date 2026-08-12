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
static constexpr unsigned K = 7168;
static constexpr size_t ARENA = 2ull << 30;
static constexpr size_t TARGET_STREAM = 1536ull << 20;

struct Cohort {
    const char* name;
    unsigned total_n;
    unsigned instances;
};

static const Cohort COHORTS[] = {
    {"kda_qkvg_fa_beta", 6144 + 128 + 12, 69},
    {"mla_input", 1536 + 512 + 64 + 1536, 24},
    {"moe_router_latent", 896 + 3584, 92},
};
static const unsigned GRIDS[] = {128, 192, 256, 304};

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
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_gemv_cohort_sweep_gfx942.co";
    const int samples = argc > 2 ? std::atoi(argv[2]) : 21;
    if (samples < 3 || (samples & 1) == 0) {
        std::fprintf(stderr, "samples must be an odd integer >= 3\n");
        return 2;
    }
    CK(hipInit(0));
    hipModule_t module;
    CK(hipModuleLoad(&module, object));

    unsigned short *weights, *x, *control_out, *candidate_out;
    CK(hipMalloc(&weights, ARENA));
    CK(hipMalloc(&x, K * 2));
    CK(hipMalloc(&control_out, 16384));
    CK(hipMalloc(&candidate_out, 16384));
    const size_t fill_bytes = 64u << 20;
    std::vector<unsigned short> fill(fill_bytes / 2);
    unsigned state = 0x76a5f00du;
    for (auto& value : fill) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<unsigned short>(0x3B00u | ((state >> 20) & 0x7Fu));
    }
    for (size_t offset = 0; offset < ARENA; offset += fill_bytes)
        CK(hipMemcpy(reinterpret_cast<char*>(weights) + offset, fill.data(),
                     std::min(fill_bytes, ARENA - offset), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, fill.data(), K * 2, hipMemcpyHostToDevice));

    double control_projection = 0.0;
    double fused_projection = 0.0;
    std::printf("%-22s %7s %6s %11s %10s %8s\n", "cohort", "nrep", "grid", "control_us",
                "fused_us", "speedup");
    for (unsigned id = 0; id < std::size(COHORTS); ++id) {
        const Cohort& c = COHORTS[id];
        const size_t slab = static_cast<size_t>(c.total_n) * K * 2;
        unsigned nrep = static_cast<unsigned>(std::max<size_t>(1, TARGET_STREAM / slab));
        while (static_cast<size_t>(nrep) * slab > ARENA) --nrep;
        char control_name[32], fused_name[32];
        std::snprintf(control_name, sizeof(control_name), "k_control_%u", id);
        std::snprintf(fused_name, sizeof(fused_name), "k_fused_%u", id);
        hipFunction_t control, fused;
        CK(hipModuleGetFunction(&control, module, control_name));
        CK(hipModuleGetFunction(&fused, module, fused_name));
        void* control_args[] = {&control_out, &x, &weights, &nrep, const_cast<unsigned*>(&K)};
        const double control_us = time_kernel(control, 304, control_args, samples) / nrep;

        double best_us = 1.0e30;
        unsigned best_grid = 0;
        for (unsigned grid : GRIDS) {
            void* fused_args[] = {&candidate_out, &x, &weights, &nrep, const_cast<unsigned*>(&K)};
            const double us = time_kernel(fused, grid, fused_args, samples) / nrep;
            if (us < best_us) {
                best_us = us;
                best_grid = grid;
            }
        }

        unsigned one = 1;
        void* ca[] = {&control_out, &x, &weights, &one, const_cast<unsigned*>(&K)};
        void* fa[] = {&candidate_out, &x, &weights, &one, const_cast<unsigned*>(&K)};
        CK(hipMemset(control_out, 0, c.total_n * 2));
        CK(hipMemset(candidate_out, 0, c.total_n * 2));
        launch_once(control, 304, ca);
        launch_once(fused, best_grid, fa);
        std::vector<unsigned short> a(c.total_n), b(c.total_n);
        CK(hipMemcpy(a.data(), control_out, c.total_n * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), candidate_out, c.total_n * 2, hipMemcpyDeviceToHost));
        if (a != b) {
            size_t diff = 0;
            while (diff < a.size() && a[diff] == b[diff]) ++diff;
            std::fprintf(stderr, "FAIL %s: first differing output %zu\n", c.name, diff);
            return 3;
        }
        control_projection += control_us * c.instances / 1000.0;
        fused_projection += best_us * c.instances / 1000.0;
        std::printf("%-22s %7u %6u %11.3f %10.3f %8.4f\n", c.name, nrep, best_grid,
                    control_us, best_us, control_us / best_us);
    }
    std::printf("weighted projection: control %.3f ms fused %.3f ms saving %.3f ms speedup %.4fx\n",
                control_projection, fused_projection, control_projection - fused_projection,
                control_projection / fused_projection);
    return 0;
}
