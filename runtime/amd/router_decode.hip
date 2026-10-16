/* router_decode.hip — PLOW_GLM_ROUTER_RAW: the decode MoE router as ONE raw segment.
 *
 * A GemmF32 marked ROUTER_RAW_MARK carries the score GEMV and its top-k tail (op MOE_ROUTER_TOPK
 * at one row, op 83 at `rows`). Workgroups [r*per_row, (r+1)*per_row) compute row r's logits with
 * d_gemv_bf16_f32 at M = 1 (one wave per output, so every logit is the interpreter's bits), and the
 * last of them to finish runs d_moe_router_topk over the row (op 83's per-token body, with its
 * shared-expert tail slot). Table bytes are identical to the GemmF32 -> top-k packet pair.
 * With `signal_add` set (ROUTER_RAW_SIGNAL) the top-k workgroup adds it to the interpreter
 * counter at `signal` once the table is written. */
#include "amd_common.h"
#include "op_gemm.h"
#include "op_moe.h"

extern "C" __device__ unsigned plow_router_decode_abi_3 = 1;

struct RouterDecodeArgs {
    float* logit;
    const bf16* x;
    const bf16* w;
    unsigned char* table;
    const float* bias;
    unsigned* ctr; /* [rows] counters, 32 words apart, zero between launches */
    unsigned* signal;
    unsigned rows, n_exp, k_dim, topk, flags, shared_tail, n_group, topk_group;
    float route_scale;
    unsigned per_row, signal_add, pad;
};

__device__ __forceinline__ void router_topk_row(const RouterDecodeArgs& a, unsigned row,
                                                float* lds) {
    unsigned char* const trow = a.table + (size_t)row * (a.topk + a.shared_tail) * 8u;
    d_moe_router_topk(trow, (const bf16*)(a.logit + (size_t)row * a.n_exp), a.bias, a.n_exp,
                      a.topk, a.flags, a.route_scale, 0u, 1u, lds, a.n_group, a.topk_group);
    if (a.shared_tail && threadIdx.x == 0) {
        *(unsigned*)(trow + (size_t)a.topk * 8u) = a.n_exp;
        *(float*)(trow + (size_t)a.topk * 8u + 4u) = 1.0f;
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void plow_router_decode(RouterDecodeArgs a) {
    __shared__ float lds[4096];
    __shared__ unsigned last;
    const unsigned row = blockIdx.x / a.per_row, part = blockIdx.x % a.per_row;
    if (row >= a.rows || blockDim.x != PLOW_THREADS) __builtin_trap();
    float* const lrow = a.logit + (size_t)row * a.n_exp;
    d_gemv_bf16_f32(lrow, a.x + (size_t)row * a.k_dim, a.w, 1u, a.n_exp, a.k_dim, part,
                    a.per_row);
    __syncthreads();
    unsigned* const ctr = a.ctr + (size_t)row * 32u;
    if (threadIdx.x == 0) {
        const unsigned old =
            __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        last = old == a.per_row - 1u;
        if (last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!last) return;
    asm volatile("buffer_inv" ::: "memory");
    __syncthreads();
    router_topk_row(a, row, lds);
    if (a.signal_add && threadIdx.x == 0)
        __hip_atomic_fetch_add(a.signal, a.signal_add, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
}
