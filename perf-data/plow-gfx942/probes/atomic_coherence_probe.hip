/* Cross-XCD coherence check for the fused MoE decomposition (gfx942 / MI300X, 8 XCDs).
 *
 * WHY IT EXISTS. MI300X's L2 is PER-XCD. The fused arm has op 83 (one packet, all 304 CUs)
 * non-temporally ZERO an accumulator, and op 86 (a different packet, again all 304 CUs, and in
 * general different XCDs for the k slots of one token) then ADD into it with an
 * __HIP_MEMORY_SCOPE_AGENT atomic. If agent-scope f32 atomics were performed in the issuing
 * XCD's L2, the k contributions would land in 8 different caches and the result would be silently
 * wrong -- and "silently wrong" in an LLM reads as fluent text, which is exactly the failure the
 * campaign's own MoE-PF LDS OOB bug demonstrated.
 *
 * This reproduces the pattern EXACTLY: zero from every CU, then have T*k rows atomically add 1.0f
 * into acc[row >> 3][h], and check every element is 8.0f.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define WGS 512

__global__ __launch_bounds__(WGS) void k_zero(float* a, size_t n) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        __builtin_nontemporal_store(0.0f, a + i);
}
__global__ __launch_bounds__(WGS) void k_add(float* a, unsigned nrow, unsigned H, unsigned ksh) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned r = blockIdx.x * (WGS / 64) + wv; r < nrow; r += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)(r >> ksh) * H;
        for (unsigned c = lane; c < H; c += 64u)
            __hip_atomic_fetch_add(a + base + c, 1.0f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}
__global__ __launch_bounds__(WGS) void k_check(const float* a, size_t n, float want, unsigned* bad) {
    for (size_t i = (size_t)blockIdx.x * WGS + threadIdx.x; i < n; i += (size_t)gridDim.x * WGS)
        if (a[i] != want) atomicAdd(bad, 1u);
}
#define CK(x) do { hipError_t e=(x); if(e){printf("ERR %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
int main(void) {
    const unsigned T = 8192, H = 6144, K = 8, KSH = 3;
    const unsigned nrow = T * K;
    const size_t n = (size_t)T * H;
    float* a; unsigned* bad;
    CK(hipMalloc(&a, n * 4)); CK(hipMalloc(&bad, 4));
    int fail = 0;
    for (int rep = 0; rep < 5; rep++) {
        CK(hipMemset(bad, 0, 4));
        k_zero<<<304, WGS>>>(a, n);
        k_add<<<304, WGS>>>(a, nrow, H, KSH);
        k_check<<<304, WGS>>>(a, n, (float)K, bad);
        unsigned h; CK(hipMemcpy(&h, bad, 4, hipMemcpyDeviceToHost));
        printf("rep %d: elements != %u.0 -> %u  %s\n", rep, K, h, h ? "FAIL" : "ok");
        fail |= (h != 0);
    }
    printf("cross-XCD agent-scope f32 atomic accumulate: %s\n", fail ? "INCOHERENT" : "COHERENT");
    return fail;
}
