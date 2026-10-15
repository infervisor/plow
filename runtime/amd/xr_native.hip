/* xr_native.hip — a decode-rung XReduce / XReduceTwoShot segment as its own kernel
 * (PLOW_AMD_XR_NATIVE). The bodies are op_collective.h's, called exactly as the decode
 * interpreter calls them, so the gate protocol, the audit counts and the reduction order are
 * unchanged; what goes is the interpreter's per-segment floor around a lone collective. */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "dev_isa.h"
#include "op_collective.h"
extern "C" __device__ unsigned plow_xr_native_abi_1 = 1;
#endif

#ifndef PLOW_XCTR_DEADLINE_TICKS
#define PLOW_XCTR_DEADLINE_TICKS 1000000000ull
#endif

struct XrNativeArgs {
    unsigned short* out;
    const void* const* peers;
    unsigned* status;
    unsigned long long xoff;
    unsigned n;
    unsigned slot_bytes;
    unsigned gate_rs;
    unsigned gate_ag;
    unsigned rank;
    unsigned nranks;
    unsigned two_shot;
    /* Two-shot: the gate_ag arrivals each rank publishes per peer (the instruction's workgroup
     * count, which the host audit expects); the launch width may differ. */
    unsigned ag_count;
};
static_assert(sizeof(XrNativeArgs) == 64);

#if defined(__HIP_DEVICE_COMPILE__)
/* Two-shot with one 64-lane group per peer (AITER cross_device_reduce_2stage's access pattern):
 * every load is a 16-byte pack and the eight peers are read concurrently. Same gates and audit
 * counts as d_xreduce_twoshot_mega (gate_rs: one arrival per rank; gate_ag: one per workgroup
 * per rank) and the same r = 0..N-1 f32 order with one bf16 rounding. */
__device__ __forceinline__ bool xrn_wait(const uint32_t* g, uint32_t want, uint32_t* status,
                                         uint32_t rank) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (xctr_poll(g) < want) {
        if (__builtin_amdgcn_s_memrealtime() - t0 > PLOW_XCTR_DEADLINE_TICKS) {
            if (status) *status = 0xDEAD0000u | rank;
            return false;
        }
        __builtin_amdgcn_s_sleep(1);
    }
    xctr_acquire();
    return true;
}

__device__ void xrn_twoshot(const XrNativeArgs& a) {
    __shared__ uint4 tile[8][64];
    __shared__ int ok;
    const unsigned lane = threadIdx.x & 63u, peer = threadIdx.x >> 6;
    const unsigned packs = a.n / 8u, part = packs / 8u;
    uint32_t* gate[8];
#pragma unroll
    for (unsigned r = 0; r < 8u; r++) gate[r] = (uint32_t*)((char*)a.peers[r] + a.xoff);
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0)
            for (unsigned r = 0; r < 8u; r++) xctr_signal(PLOW_CTR(gate[r], a.gate_rs));
        ok = xrn_wait(PLOW_CTR(gate[a.rank], a.gate_rs), 8u, a.status, a.rank);
    }
    __syncthreads();
    if (!ok) return;
    const uint4* src = (const uint4*)((const char*)a.peers[peer] + a.slot_bytes);
    uint4* mine = (uint4*)((char*)a.peers[a.rank] + a.slot_bytes);
    const unsigned lo = a.rank * part;
    for (unsigned p = blockIdx.x * 64u; p < part; p += gridDim.x * 64u) {
        const unsigned q = p + lane;
        if (q < part) tile[peer][lane] = src[lo + q];
        __syncthreads();
        if (peer == 0 && q < part) {
            float acc[8] = {};
#pragma unroll
            for (unsigned r = 0; r < 8u; r++) {
                const bf16* v = (const bf16*)&tile[r][lane];
#pragma unroll
                for (unsigned j = 0; j < 8u; j++) acc[j] += bf2f(v[j]);
            }
            uint4 o;
            bf16* ov = (bf16*)&o;
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) ov[j] = f2bf(acc[j]);
            mine[lo + q] = o;
        }
        __syncthreads();
    }
    /* One remote arrival per peer instead of one per workgroup (eight peers' workgroups on one
     * counter line serialise): workgroups count themselves on this rank's own gate_rs, which every
     * peer has already finished bumping, and the last one publishes `nblk` arrivals per peer and
     * restores gate_rs, so both counters end at the audited values. */
    if (threadIdx.x == 0) {
        uint32_t* rs = PLOW_CTR(gate[a.rank], a.gate_rs);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        const uint32_t seen = __hip_atomic_fetch_add(rs, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (seen == 8u + gridDim.x - 1u) {
            __hip_atomic_fetch_sub(rs, gridDim.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            for (unsigned r = 0; r < 8u; r++)
                __hip_atomic_fetch_add(PLOW_CTR(gate[r], a.gate_ag), a.ag_count, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
        }
        ok = xrn_wait(PLOW_CTR(gate[a.rank], a.gate_ag), 8u * a.ag_count, a.status, a.rank);
    }
    __syncthreads();
    if (!ok) return;
    uint4* out = (uint4*)a.out;
    const unsigned plo = peer * part;
    for (unsigned q = blockIdx.x * 64u + lane; q < part; q += gridDim.x * 64u)
        out[plo + q] = src[plo + q];
}
#endif

extern "C" __global__ __launch_bounds__(512, 1) void plow_xr_native(XrNativeArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (a.two_shot == 2u && a.nranks == 8u && a.n % 64u == 0u)
        xrn_twoshot(a);
    else if (a.two_shot)
        d_xreduce_twoshot_mega((bf16*)a.out, a.peers, a.nranks, a.rank, a.n, a.slot_bytes, a.xoff,
                               a.gate_rs, a.gate_ag, PLOW_XCTR_DEADLINE_TICKS, a.status,
                               blockIdx.x, gridDim.x);
    else
        d_xreduce_mega((bf16*)a.out, a.peers, a.nranks, a.rank, a.n, a.slot_bytes, a.xoff,
                       a.gate_rs, PLOW_XCTR_DEADLINE_TICKS, a.status, blockIdx.x, gridDim.x);
#endif
}
