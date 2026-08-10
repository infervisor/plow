/* #37 item 3: is f2bf's NaN guard actually an exec-mask branch, and does a branchless
 * select beat it? glm52-flash-streamed-v.md sec 8: "The f2bf conversion of P alone costs
 * ~50 instructions of exec-mask-branched rounding per tile."
 * Arm 0 = shipped. Arm 1 = branchless select. Same value for every input (gated below). */
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned short bf16;

__device__ __forceinline__ bf16 f2bf_shipped(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
__device__ __forceinline__ bf16 f2bf_sel(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    const unsigned rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const unsigned qnan = (u >> 16) | 0x0040u;
    const bool isnan = (u & 0x7fffffffu) > 0x7f800000u;
    return (bf16)(isnan ? qnan : rne);
}

/* 64 conversions = one wave's share of a P tile, the shape the flash epilogue runs. */
extern "C" __global__ void k0(const float* __restrict__ in, bf16* __restrict__ out) {
    unsigned t = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 64; i++) out[t * 64 + i] = f2bf_shipped(in[t * 64 + i]);
}
extern "C" __global__ void k1(const float* __restrict__ in, bf16* __restrict__ out) {
    unsigned t = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 64; i++) out[t * 64 + i] = f2bf_sel(in[t * 64 + i]);
}
/* Exhaustive equality gate over all 2^32 float bit patterns, striped across threads. */
extern "C" __global__ void gate(unsigned* __restrict__ bad) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned n = 0;
    for (unsigned long long v = id; v <= 0xffffffffull; v += stride) {
        float f; unsigned u = (unsigned)v; __builtin_memcpy(&f, &u, 4);
        if (f2bf_shipped(f) != f2bf_sel(f)) n++;
    }
    if (n) atomicAdd(bad, n);
}
int main() {
    unsigned* d; (void)hipMalloc(&d, 4); (void)hipMemset(d, 0, 4);
    gate<<<304, 256>>>(d); (void)hipDeviceSynchronize();
    unsigned h; (void)hipMemcpy(&h, d, 4, hipMemcpyDeviceToHost);
    printf("exhaustive gate over all 2^32 bit patterns: %u mismatches -> %s\n", h,
           h ? "FAIL" : "IDENTICAL");
    return h != 0;
}
