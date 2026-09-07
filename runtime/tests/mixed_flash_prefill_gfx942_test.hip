// In nix develop:
// $PLOW_HIPCC --offload-arch=gfx942 -O3 -Iruntime/amd -Iruntime/common -c runtime/tests/mixed_flash_prefill_gfx942_test.hip -o /tmp/mixed_flash.o
// c++ /tmp/mixed_flash.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mixed_flash
// perf-data/tools/gpulease -n 1 mixed-flash-prefill /tmp/mixed_flash
#define PLOW_MIXED_STEP 1
#define PLOW_WG_WAVES 4
#define FA_DC 256
#define FA_DBUF 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#else
using bf16 = unsigned short;
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

template <typename T>
static T* upload(const std::vector<T>& values) {
    T* ptr;
    check(hipMalloc(&ptr, values.size() * sizeof(T)));
    check(hipMemcpy(ptr, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    return ptr;
}

template <int HD>
__global__ __launch_bounds__(256, 1) void mixed_flash(
    float* partial, float* ml, bf16* out, const bf16* q, const bf16* k, const bf16* v,
    unsigned rows, unsigned row0, unsigned position) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_LDS_HALVES(HD)];
    const size_t offset = (size_t)row0 * 2 * HD;
    d_flash_prefill<HD>(partial, ml, out + offset, q + offset, k, v, rows,
                       position + rows, 2, 1, position, 64, 1.0f, 256, 255, 1,
                       blockIdx.x, gridDim.x, lds);
#endif
}

template <int HD>
static bool run(unsigned rows, unsigned row0, unsigned position) {
    constexpr bf16 sentinel = 0x7fc1;
    constexpr float partial_sentinel = -12345.0f;
    const size_t elements = (row0 + rows + 5) * 2 * HD;
    std::vector<bf16> output(elements, sentinel);
    std::vector<float> partial(rows * 2 * HD, partial_sentinel);
    std::vector<float> ml(rows * 2 * 2, partial_sentinel);
    auto* dq = upload(std::vector<bf16>(elements, 0));
    auto* dk = upload(std::vector<bf16>(256 * HD, 0));
    auto* dv = upload(std::vector<bf16>(256 * HD, 0x3f80));
    auto* dout = upload(output);
    auto* dp = upload(partial);
    auto* dm = upload(ml);
    hipLaunchKernelGGL((mixed_flash<HD>), dim3(4), dim3(256), 0, 0,
                       dp, dm, dout, dq, dk, dv, rows, row0, position);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(output.data(), dout, output.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    check(hipMemcpy(partial.data(), dp, partial.size() * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(ml.data(), dm, ml.size() * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < elements; ++i) {
        const bool active = i >= row0 * 2 * HD && i < (row0 + rows) * 2 * HD;
        ok &= output[i] == (active ? (bf16)0x3f80 : sentinel);
    }
    ok &= std::all_of(partial.begin(), partial.end(), [](float v) { return v == partial_sentinel; });
    ok &= std::all_of(ml.begin(), ml.end(), [](float v) { return v == partial_sentinel; });
    std::printf("HD%d rows=%u row0=%u position=%u direct-output=%s\n",
                HD, rows, row0, position, ok ? "PASS" : "FAIL");
    for (void* ptr : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
                      static_cast<void*>(dout), static_cast<void*>(dp), static_cast<void*>(dm)})
        check(hipFree(ptr));
    return ok;
}

int main() {
    bool ok = true;
    for (unsigned rows : {1u, 17u, 53u}) {
        ok = run<256>(rows, 1, 0) && ok;
        ok = run<512>(rows, 3, 240) && ok;
    }
    return ok ? 0 : 1;
}
