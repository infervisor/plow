/* PLOW_MOE_PF_DET's three claims, on hardware (gfx942 / MI300X, 8 XCDs, 304 CU).
 *
 * The fused MoE decomposition has op 86 accumulate the k routed-expert contributions of a token
 * IN PLACE, from k workgroups that are in general on k different XCDs. PLOW_MOE_PF_ATOMIC does
 * that with `global_atomic_add_f32`, and f32 addition is not associative, so its result depends
 * on arrival order. PLOW_MOE_PF_DET adds `rint(v * 2^32)` as an INTEGER-VALUED f64 instead:
 * integer addition IS associative, so the total cannot depend on arrival order.
 *
 * This probe checks, at the exact shipped shape (T=8192, H=6144, k=8, i.e. 402.65 M atomics on a
 * 50.33 M-element accumulator):
 *   1. COHERENCE  -- agent-scope f64 atomics accumulate correctly across XCDs (the f32 twin was
 *                    already checked by atomic_coherence_probe.hip; f64 is a different
 *                    instruction and gets its own check rather than an inference).
 *   2. DETERMINISM-- repeat the SAME accumulation and hash the accumulator. f32 must be allowed
 *                    to disagree between reps; f64 fixed-point must agree BIT-EXACTLY.
 *   3. RATE       -- global_atomic_add_f32 vs global_atomic_add_f64, so the epilogue's cost can
 *                    be priced instead of guessed.
 *
 * The hash is an integer accumulation of the raw bit patterns, so the hash itself is
 * order-independent and cannot manufacture the agreement it is testing for.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define WGS 512
#define DET_SCALE 4294967296.0
#define DET_UNSCALE 2.3283064365386962890625e-10

/* A deterministic per-(row,col) contribution that SPANS BINADES -- if every addend had the same
 * exponent the f32 sum would be order-independent by accident and the probe would prove nothing. */
__device__ __forceinline__ float pv(unsigned r, unsigned c) {
    unsigned x = r * 2654435761u ^ c * 2246822519u;
    x ^= x >> 13; x *= 2654435761u; x ^= x >> 16;
    float m = (float)(int)(x & 0xFFFFFFu) * (1.0f / 8388608.0f) - 1.0f; /* [-1,1) */
    return ldexpf(m, (int)((x >> 24) & 15u) - 8);                       /* 2^-8 .. 2^7 */
}
__global__ __launch_bounds__(WGS) void k_zero32(float* a, size_t n) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        __builtin_nontemporal_store(0.0f, a + i);
}
__global__ __launch_bounds__(WGS) void k_zero64(double* a, size_t n) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        __builtin_nontemporal_store(0.0, a + i);
}
__global__ __launch_bounds__(WGS) void k_add32(float* a, unsigned nrow, unsigned H, unsigned ksh,
                                               int unit) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned r = blockIdx.x * (WGS / 64) + wv; r < nrow; r += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)(r >> ksh) * H;
        for (unsigned c = lane; c < H; c += 64u)
            __hip_atomic_fetch_add(a + base + c, unit ? 1.0f : pv(r, c), __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
    }
}
__global__ __launch_bounds__(WGS) void k_add64(double* a, unsigned nrow, unsigned H, unsigned ksh,
                                               int unit) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned r = blockIdx.x * (WGS / 64) + wv; r < nrow; r += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)(r >> ksh) * H;
        for (unsigned c = lane; c < H; c += 64u) {
            const float v = unit ? 1.0f : pv(r, c);
            __hip_atomic_fetch_add(a + base + c, __builtin_rint((double)v * DET_SCALE),
                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}
/* Order-independent hash: integer accumulation of the raw bits. */
__global__ __launch_bounds__(WGS) void k_hash32(const float* a, size_t n, unsigned long long* h) {
    unsigned long long acc = 0;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        acc += (unsigned long long)__builtin_bit_cast(unsigned, a[i]) * 0x9E3779B97F4A7C15ull
               + (unsigned long long)i;
    atomicAdd(h, acc);
}
__global__ __launch_bounds__(WGS) void k_hash64(const double* a, size_t n, unsigned long long* h) {
    unsigned long long acc = 0;
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        acc += __builtin_bit_cast(unsigned long long, a[i]) * 0x9E3779B97F4A7C15ull
               + (unsigned long long)i;
    atomicAdd(h, acc);
}
__global__ __launch_bounds__(WGS) void k_chk32(const float* a, size_t n, float w, unsigned* bad) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        if (a[i] != w) atomicAdd(bad, 1u);
}
__global__ __launch_bounds__(WGS) void k_chk64(const double* a, size_t n, double w, unsigned* bad) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        if (a[i] * DET_UNSCALE != w) atomicAdd(bad, 1u);
}
#define CK(x) do { hipError_t e=(x); if(e){printf("ERR %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
int main(void) {
    const unsigned T = 8192, H = 6144, K = 8, KSH = 3, nrow = T * K;
    const size_t n = (size_t)T * H;
    float* a32; double* a64; unsigned* bad; unsigned long long* hh;
    CK(hipMalloc(&a32, n * 4)); CK(hipMalloc(&a64, n * 8));
    CK(hipMalloc(&bad, 4)); CK(hipMalloc(&hh, 8));
    unsigned b; unsigned long long h;

    printf("shape T=%u H=%u k=%u : %.2f M elements, %.2f M atomics\n", T, H, K,
           (double)n / 1e6, (double)nrow * H / 1e6);

    puts("\n--- 1. CROSS-XCD COHERENCE (unit contributions, want exactly k) ---");
    int fail = 0;
    for (int rep = 0; rep < 3; rep++) {
        CK(hipMemset(bad, 0, 4));
        k_zero32<<<304, WGS>>>(a32, n); k_add32<<<304, WGS>>>(a32, nrow, H, KSH, 1);
        k_chk32<<<304, WGS>>>(a32, n, (float)K, bad);
        CK(hipMemcpy(&b, bad, 4, hipMemcpyDeviceToHost));
        printf("  f32 rep %d: elements != %u.0 -> %u  %s\n", rep, K, b, b ? "FAIL" : "ok");
        fail |= (b != 0);
        CK(hipMemset(bad, 0, 4));
        k_zero64<<<304, WGS>>>(a64, n); k_add64<<<304, WGS>>>(a64, nrow, H, KSH, 1);
        k_chk64<<<304, WGS>>>(a64, n, (double)K, bad);
        CK(hipMemcpy(&b, bad, 4, hipMemcpyDeviceToHost));
        printf("  f64 rep %d: elements != %u.0 -> %u  %s\n", rep, K, b, b ? "FAIL" : "ok");
        fail |= (b != 0);
    }
    printf("  cross-XCD agent-scope atomic accumulate: %s\n", fail ? "INCOHERENT" : "COHERENT");

    puts("\n--- 2. RUN-TO-RUN DETERMINISM (binade-spanning contributions, 6 reps) ---");
    unsigned long long h32[6], h64[6];
    for (int rep = 0; rep < 6; rep++) {
        CK(hipMemset(hh, 0, 8));
        k_zero32<<<304, WGS>>>(a32, n); k_add32<<<304, WGS>>>(a32, nrow, H, KSH, 0);
        k_hash32<<<304, WGS>>>(a32, n, hh);
        CK(hipMemcpy(&h32[rep], hh, 8, hipMemcpyDeviceToHost));
        CK(hipMemset(hh, 0, 8));
        k_zero64<<<304, WGS>>>(a64, n); k_add64<<<304, WGS>>>(a64, nrow, H, KSH, 0);
        k_hash64<<<304, WGS>>>(a64, n, hh);
        CK(hipMemcpy(&h64[rep], hh, 8, hipMemcpyDeviceToHost));
        printf("  rep %d  f32 hash %016llx   f64-fixed hash %016llx\n", rep, h32[rep], h64[rep]);
    }
    int d32 = 0, d64 = 0;
    for (int i = 1; i < 6; i++) { d32 += (h32[i] != h32[0]); d64 += (h64[i] != h64[0]); }
    printf("  f32 atomic       : %d of 5 reps DIFFER from rep 0  -> %s\n", d32,
           d32 ? "NONDETERMINISTIC" : "no divergence observed");
    printf("  f64 fixed-point  : %d of 5 reps DIFFER from rep 0  -> %s\n", d64,
           d64 ? "NONDETERMINISTIC (unexpected)" : "BIT-REPRODUCIBLE");

    puts("\n--- 3. ATOMIC RATE (best of 4, 402.65 M atomics) ---");
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    float best32 = 1e30f, best64 = 1e30f;
    for (int i = 0; i < 4; i++) {
        float ms;
        k_zero32<<<304, WGS>>>(a32, n); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); k_add32<<<304, WGS>>>(a32, nrow, H, KSH, 0); CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best32) best32 = ms;
        k_zero64<<<304, WGS>>>(a64, n); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); k_add64<<<304, WGS>>>(a64, nrow, H, KSH, 0); CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best64) best64 = ms;
    }
    const double N = (double)nrow * H;
    printf("  global_atomic_add_f32 -> %6.3f ms   %6.1f G atomics/s\n", best32, N / best32 / 1e6);
    printf("  global_atomic_add_f64 -> %6.3f ms   %6.1f G atomics/s   (%.2fx the f32 time)\n",
           best64, N / best64 / 1e6, best64 / best32);
    return fail;
}
