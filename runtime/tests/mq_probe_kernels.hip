/* mq_probe_kernels.hip — device code for the multi-queue AQL overlap probe
 * (`mq_probe_bench.c`, `plans/` multi-queue-aql brief).
 *
 * Question this pair of kernels exists to answer: on ONE gfx942 agent, does an AQL
 * dispatch on a SECOND, independent queue actually run concurrently with a dispatch
 * already in flight on the first queue, absent the barrier bit plow's runtime sets on
 * every packet today (`device/hsa.rs`, `runtime/amd/hsa_backend.c`)? And if it does,
 * does either kernel's own duration grow from sharing the die (the effect
 * `PLOW_AMD_DECODE_GEMM_OVERLAP` measured: native intervals grew 40-50 us when the
 * barrier was dropped WITHIN one queue)?
 *
 * `mq_xreduce_twoshot_ts` is `tp_allreduce_twoshot` (tp_allreduce_kernels.hip) unchanged
 * in substance — same call into op_collective.h's real `d_xreduce_twoshot_mega`, the
 * exact op XReduceTwoShot dispatches in production — renamed only so this file builds
 * standalone. It plays the "fabric-bound collective" segment, capped to TP_NWG
 * workgroups (PLOW_XR_CUS's real lever: 32 by default, docs/flags-reference.md).
 *
 * `mq_gemm_proxy` is a SYNTHETIC stand-in for the "independent GEMM" segment. It is not
 * hipBLASLt (a library call, not a raw kernel object — there is nothing to grab an ELF
 * for), so it cannot be bit-for-bit the production GemmLtPf kernel; it is a
 * dependent-FMA-chain compute kernel with a calibratable iteration count, sized (see
 * mq_probe_bench.c) to land near the ~14 us "lone GEMM body" figure decode-latency.md
 * measures. What it DOES share with the real thing, and what this probe is actually
 * testing: it is an ordinary kernel dispatch that reserves a CU footprint disjoint from
 * the collective's, with no dependency on it — the exact shape `segment_resource.rs`'s
 * `co_resident()` calls safe.
 */
#include "../amd/op_collective.h"

extern "C" __device__ uint32_t mq_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

/* Same fill pattern tp_allreduce_kernels.hip's tp_fill_partial uses, copied so this file
 * has no build-time dependency on that one (both compile the same op_collective.h). */
extern "C" __global__ void mq_fill_partial(bf16* part, uint32_t n, uint32_t rank) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += gridDim.x * blockDim.x)
        part[e] = f2bf((float)(rank + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}

__device__ __forceinline__ uint64_t mq_now(void) { return __builtin_amdgcn_s_memrealtime(); }

/* The real collective, one call, timestamped. `ts[0]` = entry, `ts[1]` = exit — written
 * only by (blockIdx.x==0, threadIdx.x==0), the same convention tp_allreduce_twoshot uses
 * for its `cycles` output, so this is directly comparable to the existing bench's
 * us/collective numbers at the same TP_ROWS/TP_HIDDEN/TP_NWG. */
extern "C" __global__ void mq_xreduce_twoshot_ts(
    bf16* out, const void* const* peer_scratch, uint32_t nranks, uint32_t rank,
    uint32_t n, uint32_t slot_bytes, uint64_t xctr_byte_off,
    uint64_t deadline, uint64_t* ts, uint32_t* status) {
    const uint64_t t0 = mq_now();
    d_xreduce_twoshot_mega(out, peer_scratch, nranks, rank, n, slot_bytes,
                           (size_t)xctr_byte_off, 0u, 1u, deadline, status,
                           blockIdx.x, gridDim.x);
    const uint64_t t1 = mq_now();
    if (blockIdx.x == 0 && threadIdx.x == 0 && ts) { ts[0] = t0; ts[1] = t1; }
}

/* Synthetic independent-GEMM proxy: `iters` dependent FMA steps per thread (the chain is
 * serially dependent — each step reads the previous result — so the compiler cannot
 * hoist or parallelise it away, and duration scales linearly with `iters`, letting
 * mq_probe_bench.c calibrate it against a wall-clock target once at startup). One
 * (start,end) timestamp pair per workgroup, like tp_alltoall_bench.c's convention, so the
 * host can find this kernel's true occupied window even if workgroups straggle. `out[]`
 * takes the live value so LLVM cannot prove the loop dead and delete it. */
extern "C" __global__ void mq_gemm_proxy(float* out, uint32_t iters, uint64_t* ts) {
    const uint64_t t0 = mq_now();
    float a = (float)(threadIdx.x + 1) * 1.0000001f;
    float b = (float)(blockIdx.x + 1) * 0.9999999f;
#pragma unroll 1
    for (uint32_t i = 0; i < iters; i++) {
        a = fmaf(a, 1.0000001f, b);
        b = fmaf(b, 0.9999999f, a);
    }
    __syncthreads();
    const uint64_t t1 = mq_now();
    /* Real, always-executed store: the compute above cannot be proven dead. */
    if (threadIdx.x == 0 && out) out[blockIdx.x] = a + b;
    if (threadIdx.x == 0 && ts) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = t1; }
}
