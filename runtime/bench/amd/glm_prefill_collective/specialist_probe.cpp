#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call) do { auto e = (call); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s at %d\n", #call, hipGetErrorString(e), __LINE__); std::exit(1); } } while (0)
struct Args { uint16_t* out; void** peers; uint32_t* status; uint64_t xoff; uint32_t n, rank; };
static_assert(sizeof(Args) == 40);
uint16_t bf(float value) { uint32_t x; std::memcpy(&x, &value, 4); return uint16_t((x + 0x7fff + ((x >> 16) & 1)) >> 16); }
float f32(uint16_t value) { uint32_t x = uint32_t(value) << 16; float f; std::memcpy(&f, &x, 4); return f; }
struct Device {
    void* scratch = nullptr;
    uint16_t* output = nullptr;
    void** peers = nullptr;
    uint32_t* status = nullptr;
    hipStream_t stream;
    hipEvent_t start, end;
    std::array<hipModule_t, 3> modules;
    std::array<hipFunction_t, 3> kernels;
    hipFunction_t init;
};

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s four-wave.elf eight-wave.elf wave-rs.elf\n", argv[0]); return 2; }
    constexpr uint32_t guard = 128, max_n = 8192u * 6144u;
    const size_t max_bytes = size_t(max_n) * 2 + 1024;
    std::array<Device, 8> devices;
    for (int rank = 0; rank < 8; ++rank) {
        CHECK(hipSetDevice(rank));
        for (int peer = 0; peer < 8; ++peer) if (rank != peer) {
            auto error = hipDeviceEnablePeerAccess(peer, 0);
            if (error == hipErrorPeerAccessAlreadyEnabled) hipGetLastError(); else CHECK(error);
        }
        auto& d = devices[rank];
        CHECK(hipMalloc(&d.scratch, max_bytes));
        CHECK(hipMalloc(&d.output, (size_t(max_n) + 2 * guard) * 2));
        CHECK(hipMalloc(&d.peers, 8 * sizeof(void*)));
        CHECK(hipMalloc(&d.status, 4));
        CHECK(hipStreamCreate(&d.stream));
        CHECK(hipEventCreate(&d.start)); CHECK(hipEventCreate(&d.end));
        for (int arm = 0; arm < 3; ++arm) {
            CHECK(hipModuleLoad(&d.modules[arm], argv[arm + 1]));
            CHECK(hipModuleGetFunction(&d.kernels[arm], d.modules[arm], "collective"));
        }
        CHECK(hipModuleGetFunction(&d.init, d.modules[0], "initialize"));
    }
    std::array<void*, 8> pointers;
    for (int r = 0; r < 8; ++r) pointers[r] = devices[r].scratch;
    for (int r = 0; r < 8; ++r) {
        CHECK(hipSetDevice(r));
        CHECK(hipMemcpy(devices[r].peers, pointers.data(), sizeof(pointers), hipMemcpyHostToDevice));
    }
    std::puts("rows,blocks,arm,median_max_rank_us,median_host_us,verified_reuses");
    for (uint32_t rows : {1u, 128u, 129u, 512u, 2048u, 8191u, 8192u}) {
        uint32_t n = rows * 6144;
        uint64_t xoff = (256 + uint64_t(n) * 2 + 256 + 255) / 256 * 256;
        std::vector<uint16_t> host(size_t(n) + 2 * guard);
        for (uint32_t blocks : {38u, 76u, 152u, 304u}) for (int arm : {0, 1, 2}) {
            std::vector<double> gpu_times, host_times;
            for (uint32_t repeat = 0; repeat < 14; ++repeat) {
                for (uint32_t rank = 0; rank < 8; ++rank) {
                    CHECK(hipSetDevice(rank)); auto& d = devices[rank];
                    CHECK(hipMemsetAsync(d.scratch, 0xa1, xoff + 256, d.stream));
                    CHECK(hipMemsetAsync(d.output, 0xa1, (size_t(n) + 2 * guard) * 2, d.stream));
                    CHECK(hipMemsetAsync((char*)d.scratch + xoff, 0, 256, d.stream));
                    CHECK(hipMemsetAsync(d.status, 0, 4, d.stream));
                    auto* data = (uint16_t*)((char*)d.scratch + 256);
                    auto* out = d.output + guard;
                    void* args[] = {&data, &out, &n, &rank, &repeat};
                    CHECK(hipModuleLaunchKernel(d.init, 304, 1, 1, 256, 1, 1, 0, d.stream, args, nullptr));
                }
                // All peers must finish clearing gates before anyone can signal them.
                for (int rank = 0; rank < 8; ++rank) { CHECK(hipSetDevice(rank)); CHECK(hipStreamSynchronize(devices[rank].stream)); }
                auto start = std::chrono::steady_clock::now();
                for (uint32_t rank = 0; rank < 8; ++rank) {
                    CHECK(hipSetDevice(rank)); auto& d = devices[rank];
                    Args args{d.output + guard, d.peers, d.status, xoff, n, rank};
                    size_t size = sizeof(args);
                    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
                    CHECK(hipEventRecord(d.start, d.stream));
                    CHECK(hipModuleLaunchKernel(d.kernels[arm], blocks, 1, 1, arm == 0 ? 256 : 512, 1, 1, 0, d.stream, nullptr, config));
                    CHECK(hipEventRecord(d.end, d.stream));
                }
                double maximum = 0;
                for (int rank = 0; rank < 8; ++rank) {
                    CHECK(hipSetDevice(rank)); auto& d = devices[rank];
                    CHECK(hipEventSynchronize(d.end)); float ms; CHECK(hipEventElapsedTime(&ms, d.start, d.end));
                    maximum = std::max(maximum, double(ms) * 1000);
                }
                double host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                if (repeat >= 5) { gpu_times.push_back(maximum); host_times.push_back(host_us); }
                std::array<uint16_t, 127> expected;
                for (uint32_t i = 0; i < 127; ++i) {
                    float sum = 0;
                    for (int rank = 0; rank < 8; ++rank) sum += f32(bf(float(rank + 1) + float(int(i) - 63) / 128.0f + float(repeat % 3) / 8.0f));
                    expected[i] = bf(sum);
                }
                for (int rank = 0; rank < 8; ++rank) {
                    CHECK(hipSetDevice(rank)); auto& d = devices[rank];
                    uint32_t status, gates[64];
                    CHECK(hipMemcpy(&status, d.status, 4, hipMemcpyDeviceToHost));
                    CHECK(hipMemcpy(gates, (char*)d.scratch + xoff, sizeof(gates), hipMemcpyDeviceToHost));
                    if (status || gates[0] != 8 || gates[32] != 8 * blocks) { std::fprintf(stderr, "gate failure rows%u blocks%u arm%d rank%d status%x gates%u/%u\n", rows, blocks, arm, rank, status, gates[0], gates[32]); return 3; }
                    if (repeat >= 3) continue;
                    CHECK(hipMemcpy(host.data(), d.output, host.size() * 2, hipMemcpyDeviceToHost));
                    for (uint32_t i = 0; i < guard; ++i) if (host[i] != 0xa1a1 || host[guard + n + i] != 0xa1a1) return 4;
                    for (uint32_t i = 0; i < n; ++i) if (host[guard + i] != expected[i % 127]) { std::fprintf(stderr, "oracle failure rows%u blocks%u arm%d rank%d i%u got%x want%x\n", rows, blocks, arm, rank, i, host[guard + i], expected[i % 127]); return 5; }
                    std::array<uint16_t, guard> sentinel;
                    for (uint64_t off : {uint64_t(0), uint64_t(256) + uint64_t(n) * 2}) {
                        CHECK(hipMemcpy(sentinel.data(), (char*)d.scratch + off, sizeof(sentinel), hipMemcpyDeviceToHost));
                        if (!std::all_of(sentinel.begin(), sentinel.end(), [](auto x) { return x == 0xa1a1; })) return 6;
                    }
                }
            }
            std::sort(gpu_times.begin(), gpu_times.end()); std::sort(host_times.begin(), host_times.end());
            std::printf("%u,%u,%s,%.3f,%.3f,3\n", rows, blocks, arm == 0 ? "four" : arm == 1 ? "eight" : "wave_rs", gpu_times[4], host_times[4]); std::fflush(stdout);
        }
    }
    for (int rank = 0; rank < 8; ++rank) {
        CHECK(hipSetDevice(rank)); auto& d = devices[rank];
        CHECK(hipFree(d.scratch)); CHECK(hipFree(d.output)); CHECK(hipFree(d.peers)); CHECK(hipFree(d.status));
        for (auto module : d.modules) CHECK(hipModuleUnload(module));
        CHECK(hipEventDestroy(d.start)); CHECK(hipEventDestroy(d.end)); CHECK(hipStreamDestroy(d.stream));
    }
}
