/* Exercise agent-scope f32 atomic accumulation from distinct workgroups and XCDs.
 * Unit addends make the expected result exactly representable, so any mismatch is a
 * coherence failure rather than floating-point reordering.
 */
#include <hip/hip_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define WGS 512
#define CK(expr) do { \
    hipError_t status = (expr); \
    if (status != hipSuccess) { \
        std::fprintf(stderr, "HIP error at line %d: %s\n", __LINE__, \
                     hipGetErrorString(status)); \
        std::exit(2); \
    } \
} while (0)

static unsigned parse_u(const char* value, const char* name) {
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (!value[0] || *end || parsed == 0 || parsed > UINT_MAX) {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return static_cast<unsigned>(parsed);
}

__global__ __launch_bounds__(WGS) void zero_f32(float* accum, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * WGS + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * WGS)
        accum[i] = 0.0f;
}

__global__ __launch_bounds__(WGS) void add_unit_f32(
    float* accum, size_t count, unsigned* all_xcd_mask) {
    if (threadIdx.x == 0) {
        const unsigned xcd = __builtin_amdgcn_s_getreg(
            (20 /* HW_REG_XCC_ID */) | (0 << 6) | ((32 - 1) << 11)) & 0xf;
        atomicOr(all_xcd_mask, 1u << xcd);
    }

    /* Every workgroup updates every cell, so each cell sees every observed XCD. */
    for (size_t i = threadIdx.x; i < count; i += WGS)
        __hip_atomic_fetch_add(accum + i, 1.0f, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
}

__global__ __launch_bounds__(WGS) void count_bad(
    const float* accum, size_t count, float expected, unsigned long long* bad) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * WGS + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * WGS)
        if (accum[i] != expected) atomicAdd(bad, 1ull);
}

static unsigned popcount(unsigned value) {
    return static_cast<unsigned>(__builtin_popcount(value));
}

static unsigned long long check_values(
    const float* accum, size_t count, float expected, unsigned blocks,
    unsigned long long* device_bad) {
    unsigned long long bad = 0;
    CK(hipMemset(device_bad, 0, sizeof(*device_bad)));
    count_bad<<<blocks, WGS>>>(accum, count, expected, device_bad);
    CK(hipGetLastError());
    CK(hipMemcpy(&bad, device_bad, sizeof(bad), hipMemcpyDeviceToHost));
    return bad;
}

static bool check_coverage(unsigned* device_all_mask) {
    unsigned all_mask = 0;
    CK(hipMemcpy(&all_mask, device_all_mask, sizeof(all_mask),
                 hipMemcpyDeviceToHost));
    const unsigned seen = popcount(all_mask);
    std::printf("coverage: xcds=%u mask=0x%08x %s\n", seen, all_mask,
                seen > 1 ? "PASS" : "FAIL");
    return seen > 1;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "usage: %s TOKENS HIDDEN BLOCKS COLD_REPS HOT_REPS\n",
            argv[0]);
        return 2;
    }
    const unsigned tokens = parse_u(argv[1], "tokens");
    const unsigned hidden = parse_u(argv[2], "hidden");
    unsigned blocks = parse_u(argv[3], "blocks");
    const unsigned cold_reps = parse_u(argv[4], "cold_reps");
    const unsigned hot_reps = parse_u(argv[5], "hot_reps");

    hipDeviceProp_t props{};
    CK(hipGetDeviceProperties(&props, 0));
    if (blocks == UINT_MAX) {
        blocks = static_cast<unsigned>(props.multiProcessorCount);
    }
    if (hot_reps > (1u << 24) / blocks) {
        std::fprintf(
            stderr,
            "hot_reps * blocks must be <= 2^24 for exact f32 counts\n");
        return 2;
    }
    if (tokens > std::numeric_limits<size_t>::max() / hidden) {
        std::fprintf(stderr, "tensor element count overflows size_t\n");
        return 2;
    }
    const size_t count = static_cast<size_t>(tokens) * hidden;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "tensor byte count overflows size_t\n");
        return 2;
    }

    unsigned* all_mask = nullptr;
    unsigned long long* device_bad = nullptr;
    CK(hipMalloc(&all_mask, sizeof(*all_mask)));
    CK(hipMalloc(&device_bad, sizeof(*device_bad)));

    std::printf(
        "device=%s arch=%s CUs=%d shape=(tokens=%u hidden=%u) blocks=%u\n",
        props.name, props.gcnArchName, props.multiProcessorCount, tokens, hidden,
        blocks);

    bool pass = true;
    for (unsigned rep = 0; rep < cold_reps; ++rep) {
        float* accum = nullptr;
        CK(hipMalloc(&accum, count * sizeof(*accum)));
        CK(hipMemset(all_mask, 0, sizeof(*all_mask)));
        zero_f32<<<blocks, WGS>>>(accum, count);
        add_unit_f32<<<blocks, WGS>>>(accum, count, all_mask);
        CK(hipGetLastError());
        const unsigned long long bad = check_values(
            accum, count, static_cast<float>(blocks), blocks, device_bad);
        const bool coverage = check_coverage(all_mask);
        const bool rep_pass = bad == 0 && coverage;
        std::printf("COLD rep=%u expected=%u bad=%llu %s\n", rep, blocks, bad,
                    rep_pass ? "PASS" : "FAIL");
        pass &= rep_pass;
        CK(hipFree(accum));
    }

    float* accum = nullptr;
    CK(hipMalloc(&accum, count * sizeof(*accum)));
    zero_f32<<<blocks, WGS>>>(accum, count);
    CK(hipGetLastError());
    for (unsigned rep = 0; rep < hot_reps; ++rep) {
        CK(hipMemset(all_mask, 0, sizeof(*all_mask)));
        add_unit_f32<<<blocks, WGS>>>(accum, count, all_mask);
        CK(hipGetLastError());
        const unsigned expected = (rep + 1) * blocks;
        const unsigned long long bad = check_values(
            accum, count, static_cast<float>(expected), blocks, device_bad);
        const bool coverage = check_coverage(all_mask);
        const bool rep_pass = bad == 0 && coverage;
        std::printf("HOT rep=%u expected=%u bad=%llu %s\n", rep, expected, bad,
                    rep_pass ? "PASS" : "FAIL");
        pass &= rep_pass;
    }

    CK(hipFree(accum));
    CK(hipFree(all_mask));
    CK(hipFree(device_bad));
    std::printf("RESULT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
