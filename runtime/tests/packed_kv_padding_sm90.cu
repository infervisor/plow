#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMMA 1
#define PLOW_NV_GEMMA_HNR_BF16 1
#define PLOW_NV_PACKED_REQUEST 1
#define PLOW_NV_MASKED_PADDING 1
#include "op_norm.cuh"

using bf16 = __nv_bfloat16;
#define CK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)

template<class T> static T* upload(const std::vector<T>& host) {
    T* device;
    CK(cudaMalloc(&device, host.size() * sizeof(T)));
    CK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template<int HD>
__global__ void write_rows(bf16* out, const bf16* x, const bf16* gamma,
                          const float* cosb, const float* sinb, const int* pos,
                          const int* slot, unsigned rows, unsigned heads, unsigned stride) {
    d_headnorm_rope<HD>(out, x, gamma, cosb, sinb, pos, rows, heads, 1e-6f,
                       0, stride, 2047, 0, blockIdx.x, gridDim.x, 0, slot);
}

template<int HD> static void check(unsigned rows, unsigned heads) {
    constexpr unsigned stride = 2048;
    const auto sentinel = __float2bfloat16(13.0f);
    std::vector<bf16> x(size_t(rows) * heads * HD);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2bfloat16(float(int(i % 131) - 65) / 67.0f);
    std::vector<bf16> kv(size_t(3) * heads * stride * HD, sentinel), expected = kv;
    std::vector<bf16> query(x.size()), gamma(HD, __float2bfloat16(1.0f));
    std::vector<float> cosb(size_t(20480) * HD / 2, 1.0f), sinb(cosb.size(), 0.0f);
    std::vector<int> slots(rows, -1), positions(rows, 0);
    for (unsigned t = 0; t < 16; ++t) {
        slots[t] = t < 7 ? 2 : 0;
        positions[t] = t < 7 ? 16381 + t : 2040 + t - 7;
    }
    bf16* dx = upload(x), *dkv = upload(kv), *dq = upload(query), *dg = upload(gamma);
    float* dc = upload(cosb), *ds = upload(sinb);
    int* dp = upload(positions), *dslot = upload(slots);
    CK(cudaMemset(dq, 0xff, x.size() * sizeof(bf16)));
    write_rows<HD><<<132, PLOW_NV_THREADS>>>(dq, dx, dg, dc, ds, dp, dslot, rows, heads, 0);
    CK(cudaGetLastError());
    write_rows<HD><<<132, PLOW_NV_THREADS>>>(dkv, dx, dg, dc, ds, dp, dslot, rows, heads, stride);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(query.data(), dq, query.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(kv.data(), dkv, kv.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    for (auto v : query) if (!std::isfinite(__bfloat162float(v))) {
        std::fprintf(stderr, "query padding was not computed\n"); std::exit(1);
    }
    for (unsigned t = 0; t < 16; ++t) for (unsigned h = 0; h < heads; ++h) {
        const size_t dst = ((size_t(slots[t]) * heads + h) * stride + (positions[t] & 2047)) * HD;
        std::copy_n(query.data() + (size_t(t) * heads + h) * HD, HD, expected.data() + dst);
    }
    if (std::memcmp(kv.data(), expected.data(), kv.size() * sizeof(bf16))) {
        std::fprintf(stderr, "FAIL rows=%u HD=%d heads=%u\n", rows, HD, heads); std::exit(1);
    }
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dkv), static_cast<void*>(dq),
                   static_cast<void*>(dg), static_cast<void*>(dc), static_cast<void*>(ds),
                   static_cast<void*>(dp), static_cast<void*>(dslot)}) CK(cudaFree(p));
    std::printf("PASS rows=%u HD=%d heads=%u: ragged slots, ring wrap, untouched KV, query padding\n",
                rows, HD, heads);
}

int main() {
    for (unsigned rows : {128, 512, 1024, 2048, 4096, 8192}) {
        check<256>(rows, 8);
        check<512>(rows, 1);
    }
}
