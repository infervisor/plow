// Opt-in Phase-1 dispatch-floor probe. This is not linked into plowrt.
//
// Build (Nix shell toolchain only):
//   "$PLOW_HIPCC" --offload-arch=gfx942 -O3 -std=c++17 \
//     runtime/bench/dispatch/control_queue_dispatch_floor.hip -o /tmp/plow-control-floor
// Run only under a GPU lease:
//   perf-data/harness/gpulease -n 1 control-floor /tmp/plow-control-floor
//
// The baseline and resident arms execute the same grid, block size, and synthetic one-rung body.
// This bounds outer dispatch savings; it does not execute a K3 packet program and makes no K3
// latency claim. The resident arm is refused unless a cooperative grid of that exact geometry is
// occupancy-valid. The JSON verdict is STOP when median launch removal saves <=25 us or resident
// pickup is slower.

#include "../../common/control_queue_probe.h"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr unsigned kThreads = 512;
constexpr unsigned kWarmup = 64;
constexpr unsigned kDefaultSamples = 10000;
constexpr double kStopSavingUs = 25.0;
constexpr uint64_t kSessionNonce = UINT64_C(0x435150524f424531);
constexpr uint32_t kProtocolFault = 1;

struct DeviceState {
    uint64_t* start_ticks;
    uint64_t* end_ticks;
    uint64_t* sink;
    uint32_t* active_opcode;
};

__device__ __forceinline__ uint64_t sys_load(const volatile uint64_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void sys_store(volatile uint64_t* p, uint64_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t realtime_ticks() {
    return __builtin_amdgcn_s_memrealtime();
}

__device__ __forceinline__ void probe_body(DeviceState state, uint64_t seq) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t x = seq ^ (UINT64_C(0x9e3779b97f4a7c15) * (lane + 1u));
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
    }
    state.sink[lane] = x;
}

extern "C" __global__ __launch_bounds__(kThreads)
void control_floor_once(DeviceState state, uint64_t seq) {
    if (blockIdx.x == 0 && threadIdx.x == 0) state.start_ticks[seq] = realtime_ticks();
    probe_body(state, seq);
    cg::this_grid().sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) state.end_ticks[seq] = realtime_ticks();
}

extern "C" __global__ __launch_bounds__(kThreads)
void control_floor_resident(PlowControlProbeV1* control, DeviceState state,
                            uint64_t first_seq) {
    cg::grid_group grid = cg::this_grid();
    uint64_t seq = first_seq;
    for (;;) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (sys_load(&control->command.commit_seq) != seq)
                __builtin_amdgcn_s_sleep(2);
            const bool valid = control->command.session_nonce == control->header.session_nonce &&
                               control->command.record_bytes == PLOW_CONTROL_COMMAND_BYTES &&
                               control->command.abi_minor <= control->header.abi_minor &&
                               (control->command.opcode == PLOW_CONTROL_RUN_DECODE ||
                                control->command.opcode == PLOW_CONTROL_STOP);
            *state.active_opcode = valid ? control->command.opcode : 0;
            if (!valid) {
                control->header.fault_code = kProtocolFault;
                sys_store(&control->header.fault_seq, seq);
            }
            state.start_ticks[seq] = realtime_ticks();
        }
        grid.sync();

        const uint32_t opcode = *state.active_opcode;
        if (opcode == 0) return;
        if (opcode == PLOW_CONTROL_STOP) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                sys_store(&control->header.dev_head_seq, seq);
                sys_store(&control->header.stopped_seq, seq);
            }
            return;
        }

        probe_body(state, seq);
        grid.sync();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            state.end_ticks[seq] = realtime_ticks();
            sys_store(&control->header.dev_head_seq, seq);
            sys_store(&control->header.completed_seq, seq);
        }
        ++seq;
    }
}

#define HIP_OK(call)                                                                         \
    do {                                                                                     \
        const hipError_t e_ = (call);                                                        \
        if (e_ != hipSuccess) {                                                              \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,              \
                         hipGetErrorString(e_));                                              \
            std::exit(2);                                                                    \
        }                                                                                    \
    } while (0)

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(p * static_cast<double>(values.size() - 1));
    return values[index];
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

std::vector<double> body_times_us(const std::vector<uint64_t>& start,
                                  const std::vector<uint64_t>& end) {
    constexpr double kRealtimeTicksPerUs = 100.0;
    std::vector<double> times;
    times.reserve(end.size() - kWarmup);
    for (uint64_t seq = kWarmup + 1; seq < end.size(); ++seq) {
        if (end[seq] <= start[seq]) {
            std::fprintf(stderr, "invalid device timestamps at sequence %" PRIu64 "\n", seq);
            std::exit(2);
        }
        times.push_back(static_cast<double>(end[seq] - start[seq]) / kRealtimeTicksPerUs);
    }
    return times;
}

void host_store(volatile uint64_t* p, uint64_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

uint64_t host_load(const volatile uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

void publish(PlowControlProbeV1* control, uint64_t seq, uint16_t opcode) {
    control->command.session_nonce = kSessionNonce;
    control->command.record_bytes = PLOW_CONTROL_COMMAND_BYTES;
    control->command.abi_minor = PLOW_CONTROL_ABI_MINOR;
    control->command.opcode = opcode;
    control->command.flags = 0;
    control->command.program_id = 0;
    control->command.program_generation = 1;
    control->command.program_hash = UINT64_C(0x50524f42455f5631);
    control->command.command_digest = seq ^ control->command.program_hash;
    control->command.slot_mask = 1;
    control->command.rung_rows = 1;
    control->command.highest_slot_plus1 = 1;
    host_store(&control->command.commit_seq, seq);
    host_store(&control->header.host_tail_seq, seq);
}

bool wait_completed(const PlowControlProbeV1* control, uint64_t seq) {
    while (host_load(&control->header.completed_seq) != seq) {
        if (host_load(&control->header.fault_seq) == seq) return false;
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned samples = argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10))
                                      : kDefaultSamples;
    if (samples < 2) {
        std::fprintf(stderr, "samples must be >=2\n");
        return 2;
    }

    hipDeviceProp_t properties{};
    HIP_OK(hipGetDeviceProperties(&properties, 0));
    if (!properties.cooperativeLaunch) {
        std::printf("{\"schema\":\"plow.control-floor.v1\",\"scope\":\"synthetic-one-rung\","
                    "\"verdict\":\"STOP\","
                    "\"reason\":\"cooperative_launch_unsupported\"}\n");
        return 3;
    }

    int resident_blocks_per_cu = 0;
    int once_blocks_per_cu = 0;
    HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks_per_cu, reinterpret_cast<const void*>(control_floor_resident), kThreads,
        0));
    HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &once_blocks_per_cu, reinterpret_cast<const void*>(control_floor_once), kThreads, 0));
    if (resident_blocks_per_cu < 1 || once_blocks_per_cu < 1) {
        std::printf("{\"schema\":\"plow.control-floor.v1\",\"scope\":\"synthetic-one-rung\","
                    "\"verdict\":\"STOP\","
                    "\"reason\":\"cooperative_occupancy_zero\",\"resident_blocks_per_cu\":%d,"
                    "\"baseline_blocks_per_cu\":%d}\n",
                    resident_blocks_per_cu, once_blocks_per_cu);
        return 3;
    }

    constexpr unsigned blocks_per_cu = 1;
    const unsigned blocks = static_cast<unsigned>(properties.multiProcessorCount);
    const uint64_t total_runs = static_cast<uint64_t>(kWarmup) + samples;
    uint64_t stop_seq = total_runs + 1;

    PlowControlProbeV1* control = nullptr;
    HIP_OK(hipHostMalloc(&control, sizeof(*control), hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(control, 0, sizeof(*control));
    control->header.magic = PLOW_CONTROL_MAGIC;
    control->header.abi_major = PLOW_CONTROL_ABI_MAJOR;
    control->header.abi_minor = PLOW_CONTROL_ABI_MINOR;
    control->header.header_bytes = PLOW_CONTROL_HEADER_BYTES;
    control->header.command_bytes = PLOW_CONTROL_COMMAND_BYTES;
    control->header.ring_order = 0;
    control->header.rank = 0;
    control->header.n_gpu = 1;
    control->header.required_features = PLOW_CONTROL_FEAT_RUN_DECODE;
    control->header.supported_features = PLOW_CONTROL_FEAT_RUN_DECODE;
    control->header.session_nonce = kSessionNonce;
    if (!plow_control_header_v1_valid(&control->header)) return 2;

    PlowControlProbeV1* device_control = nullptr;
    HIP_OK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_control), control, 0));

    DeviceState state{};
    HIP_OK(hipMalloc(&state.start_ticks, (stop_seq + 1) * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&state.end_ticks, (stop_seq + 1) * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&state.sink,
                     static_cast<size_t>(blocks) * kThreads * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&state.active_opcode, sizeof(uint32_t)));
    HIP_OK(hipMemset(state.start_ticks, 0, (stop_seq + 1) * sizeof(uint64_t)));
    HIP_OK(hipMemset(state.end_ticks, 0, (stop_seq + 1) * sizeof(uint64_t)));

    std::vector<double> baseline;
    baseline.reserve(samples);
    for (uint64_t seq = 1; seq <= total_runs; ++seq) {
        void* args[] = {&state, &seq};
        const auto begin = Clock::now();
        HIP_OK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(control_floor_once),
                                          dim3(blocks), dim3(kThreads), args, 0, nullptr));
        HIP_OK(hipDeviceSynchronize());
        const auto end = Clock::now();
        if (seq > kWarmup) baseline.push_back(elapsed_us(begin, end));
    }

    std::vector<uint64_t> baseline_start(total_runs + 1);
    std::vector<uint64_t> baseline_end(total_runs + 1);
    HIP_OK(hipMemcpy(baseline_start.data(), state.start_ticks,
                     (total_runs + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(baseline_end.data(), state.end_ticks,
                     (total_runs + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
    const std::vector<double> baseline_body = body_times_us(baseline_start, baseline_end);

    HIP_OK(hipMemset(state.start_ticks, 0, (stop_seq + 1) * sizeof(uint64_t)));
    HIP_OK(hipMemset(state.end_ticks, 0, (stop_seq + 1) * sizeof(uint64_t)));
    std::memset(&control->command, 0, sizeof(control->command));
    host_store(&control->header.host_tail_seq, 0);
    host_store(&control->header.dev_head_seq, 0);
    host_store(&control->header.completed_seq, 0);
    host_store(&control->header.stopped_seq, 0);

    uint64_t first_seq = 1;
    void* resident_args[] = {&device_control, &state, &first_seq};
    HIP_OK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(control_floor_resident),
                                      dim3(blocks), dim3(kThreads), resident_args, 0, nullptr));

    std::vector<double> resident;
    resident.reserve(samples);
    for (uint64_t seq = 1; seq <= total_runs; ++seq) {
        const auto begin = Clock::now();
        publish(control, seq, PLOW_CONTROL_RUN_DECODE);
        if (!wait_completed(control, seq)) {
            HIP_OK(hipDeviceSynchronize());
            std::printf("{\"schema\":\"plow.control-floor.v1\","
                        "\"scope\":\"synthetic-one-rung\",\"verdict\":\"STOP\","
                        "\"reason\":\"protocol_fault\",\"fault_seq\":%" PRIu64 "}\n",
                        seq);
            return 3;
        }
        const auto end = Clock::now();
        if (seq > kWarmup) resident.push_back(elapsed_us(begin, end));
    }
    publish(control, stop_seq, PLOW_CONTROL_STOP);
    while (host_load(&control->header.stopped_seq) != stop_seq) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    HIP_OK(hipDeviceSynchronize());

    std::vector<uint64_t> resident_start(total_runs + 1);
    std::vector<uint64_t> resident_end(total_runs + 1);
    HIP_OK(hipMemcpy(resident_start.data(), state.start_ticks,
                     (total_runs + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(resident_end.data(), state.end_ticks,
                     (total_runs + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
    const std::vector<double> resident_body = body_times_us(resident_start, resident_end);

    const double baseline_median = percentile(baseline, 0.5);
    const double resident_median = percentile(resident, 0.5);
    const double baseline_body_median = percentile(baseline_body, 0.5);
    const double resident_body_median = percentile(resident_body, 0.5);
    const double saving = baseline_median - resident_median;
    const bool go = saving > kStopSavingUs && resident_median < baseline_median;
    std::printf(
        "{\"schema\":\"plow.control-floor.v1\",\"scope\":\"synthetic-one-rung\","
        "\"device\":\"%s\",\"samples\":%u,"
        "\"blocks\":%u,\"threads\":%u,\"blocks_per_cu\":%u,"
        "\"max_resident_blocks_per_cu\":%d,\"realtime_ticks_per_us\":100.0,"
        "\"baseline_us\":{\"median\":%.3f,\"mean\":%.3f,\"p90\":%.3f},"
        "\"resident_us\":{\"median\":%.3f,\"mean\":%.3f,\"p90\":%.3f},"
        "\"device_window_us\":{\"baseline_body_median\":%.3f,"
        "\"resident_pickup_and_body_median\":%.3f},"
        "\"host_residual_us\":{\"baseline_median\":%.3f,\"resident_median\":%.3f},"
        "\"saving_us\":%.3f,\"stop_threshold_us\":%.3f,\"verdict\":\"%s\"}\n",
        properties.name, samples, blocks, kThreads, blocks_per_cu, resident_blocks_per_cu,
        baseline_median, mean(baseline), percentile(baseline, 0.9), resident_median,
        mean(resident), percentile(resident, 0.9), baseline_body_median, resident_body_median,
        baseline_median - baseline_body_median, resident_median - resident_body_median, saving,
        kStopSavingUs, go ? "GO" : "STOP");

    HIP_OK(hipFree(state.start_ticks));
    HIP_OK(hipFree(state.end_ticks));
    HIP_OK(hipFree(state.sink));
    HIP_OK(hipFree(state.active_opcode));
    HIP_OK(hipHostFree(control));
    return go ? 0 : 4;
}
