// From the repository root, inside nix develop:
// $PLOW_HIPCC --offload-arch=gfx942 -O3 -DPLOW_FLASH_DECODE_MIXED -Iruntime/amd -Iruntime/common -c runtime/tests/mixed_flash_decode_gfx942_test.hip -o /tmp/mixed_fd4.o
// $PLOW_HIPCC --offload-arch=gfx942 -O3 -DPLOW_FLASH_DECODE_REFERENCE -Iruntime/amd -Iruntime/common -c runtime/tests/mixed_flash_decode_gfx942_test.hip -o /tmp/mixed_fd8.o
// c++ -x c++ -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include -c runtime/tests/mixed_flash_decode_gfx942_test.hip -o /tmp/mixed_fd_main.o
// c++ /tmp/mixed_fd_main.o /tmp/mixed_fd4.o /tmp/mixed_fd8.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mixed_fd
// perf-data/tools/gpulease -n 1 mixed-flash-decode /tmp/mixed_fd
#include <hip/hip_runtime.h>

#if defined(PLOW_FLASH_DECODE_MIXED) || defined(PLOW_FLASH_DECODE_REFERENCE)
#if defined(PLOW_FLASH_DECODE_MIXED)
#define PLOW_MIXED_STEP 1
#define PLOW_WG_WAVES 4
#define PROBE 4
#else
#define PLOW_WG_WAVES 8
#define PROBE 8
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#else
using bf16 = unsigned short;
#endif
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

template <int D>
__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void CAT(decode, PROBE)(
    float* partial, float* ml, const bf16* q, const bf16* k, const bf16* v,
    const int* len, unsigned ns, unsigned window) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[FA_DEC_LDS_FLOATS(D, 2)];
    d_flash_decode<D, 2>(partial, ml, q, k, v, len, 1, 2, 1, 8192, window,
                         1.0f, ns, 8191, blockIdx.x, gridDim.x, lds);
#endif
}

template <int D>
__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void CAT(merge, PROBE)(
    bf16* out, const float* partial, const float* ml, unsigned ns) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_flash_merge<D>(out, partial, ml, 1, 2, ns, blockIdx.x, gridDim.x);
#endif
}

template <int D>
static void launch(float* partial, float* ml, bf16* out, const bf16* q,
                   const bf16* k, const bf16* v, const int* len, unsigned ns,
                   unsigned window) {
    hipLaunchKernelGGL((CAT(decode, PROBE)<D>), dim3(16), dim3(PLOW_WG_WAVES * 64),
                       0, 0, partial, ml, q, k, v, len, ns, window);
    hipLaunchKernelGGL((CAT(merge, PROBE)<D>), dim3(16), dim3(PLOW_WG_WAVES * 64),
                       0, 0, out, partial, ml, ns);
}

extern "C" void CAT(launch, PROBE)(unsigned d, float* partial, float* ml, bf16* out,
                                   const bf16* q, const bf16* k, const bf16* v,
                                   const int* len, unsigned ns, unsigned window) {
    if (d == 256) launch<256>(partial, ml, out, q, k, v, len, ns, window);
    else launch<512>(partial, ml, out, q, k, v, len, ns, window);
}

#else
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using bf16 = unsigned short;
extern "C" void launch4(unsigned, float*, float*, bf16*, const bf16*, const bf16*,
                         const bf16*, const int*, unsigned, unsigned);
extern "C" void launch8(unsigned, float*, float*, bf16*, const bf16*, const bf16*,
                         const bf16*, const int*, unsigned, unsigned);

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
static T* upload(const std::vector<T>& values) {
    T* ptr;
    check(hipMalloc(&ptr, values.size() * sizeof(T)));
    check(hipMemcpy(ptr, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    return ptr;
}

template <typename T>
static std::vector<T> readback(const T* ptr, size_t count) {
    std::vector<T> values(count);
    check(hipMemcpy(values.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
    return values;
}

static std::vector<bf16> random_bf16(size_t count, uint32_t seed) {
    std::vector<bf16> values(count);
    for (auto& value : values) {
        seed = seed * 1664525u + 1013904223u;
        const float f = (static_cast<int>(seed >> 8) - 8388608) * (0.5f / 8388608);
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        value = bits >> 16;
    }
    return values;
}

static bool finite(float value) { return std::isfinite(value); }
static bool finite(bf16 value) { return (value & 0x7f80u) != 0x7f80u; }

template <typename T>
static size_t differences(const std::vector<T>& a, const std::vector<T>& b) {
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        count += !finite(a[i]) || !finite(b[i]) ||
                 std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return count;
}

static bool run(unsigned d, unsigned len) {
    const unsigned ns = len == 8192 ? 1 : 10;
    const unsigned window = d == 256 ? 1024 : 0;
    const size_t np = 2 * ns * d, nm = 4 * ns, no = 2 * d;
    auto* q = upload(random_bf16(no, 7));
    auto* k = upload(random_bf16(8192 * d, 19));
    auto* v = upload(random_bf16(8192 * d, 37));
    auto* length = upload(std::vector<int>{static_cast<int>(len)});
    auto* p4 = upload(std::vector<float>(np, NAN));
    auto* p8 = upload(std::vector<float>(np, NAN));
    auto* m4 = upload(std::vector<float>(nm, NAN));
    auto* m8 = upload(std::vector<float>(nm, NAN));
    auto* o4 = upload(std::vector<bf16>(no, 0x7fc1));
    auto* o8 = upload(std::vector<bf16>(no, 0x7fc1));
    launch4(d, p4, m4, o4, q, k, v, length, ns, window);
    launch8(d, p8, m8, o8, q, k, v, length, ns, window);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    const auto pd = differences(readback(p4, np), readback(p8, np));
    const auto md = differences(readback(m4, nm), readback(m8, nm));
    const auto od = differences(readback(o4, no), readback(o8, no));
    std::printf("HD%u len=%u ns=%u partial_diff=%zu ml_diff=%zu output_diff=%zu %s\n",
                d, len, ns, pd, md, od, pd || md || od ? "FAIL" : "PASS");
    for (void* ptr : {static_cast<void*>(q), static_cast<void*>(k), static_cast<void*>(v),
                     static_cast<void*>(length), static_cast<void*>(p4), static_cast<void*>(p8),
                     static_cast<void*>(m4), static_cast<void*>(m8), static_cast<void*>(o4),
                     static_cast<void*>(o8)})
        check(hipFree(ptr));
    return !pd && !md && !od;
}

int main() {
    bool ok = true;
    for (unsigned d : {256u, 512u})
        for (unsigned len : {97u, 257u, 1024u, 2048u, 8192u}) ok = run(d, len) && ok;
    return ok ? 0 : 1;
}
#endif
