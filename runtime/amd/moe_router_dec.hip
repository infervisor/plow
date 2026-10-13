/* Standalone GLM decode router (op 211 MoeRouterDec, PLOW_GLM_ROUTER_DEC).
 *
 * Replaces the decode GemmF32 -> MoeRouterTopkPf -> MoeAlignPf packets with one launch:
 *   1. grid = (M/4) x (E/32) workgroups; workgroup b's wave w owns router tile b*8+w of
 *      d_gemv_bf16_f32_tile<4,4> (the decode object's own GemmF32 body, same per-output order);
 *   2. the last workgroup of each 4-row group (device-scope counter) runs the op-83 top-k tail
 *      for those rows (one wave per token);
 *   3. the last group to finish runs op 84's single-workgroup align over all M rows.
 * Each stage is the interpreter's body on the same inputs, so table / meta / row maps carry the
 * same values (align's within-expert row order is an LDS-atomic order in both, and nothing
 * downstream depends on it). Counters are reset by their last arriver for the next launch. */
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_gemm.h"
#include "op_moe.h"

extern "C" __device__ unsigned plow_moe_router_dec_1 = 1;

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void plow_moe_router_dec_v1(
    float* logit, const bf16* x, const bf16* w, const float* bias, unsigned char* table,
    int* meta, unsigned* counters, unsigned* row_token, unsigned* row_partidx, float* row_gate,
    unsigned rows, unsigned n_exp, unsigned hidden, unsigned k, unsigned flags,
    unsigned shared_tail, unsigned n_group, unsigned topk_group, unsigned e_route,
    unsigned k_route, float route_scale, unsigned abi_pad) {
    (void)abi_pad; /* keeps the kernarg segment at the host struct's 128 B */
    constexpr unsigned R = 4, CN = 4, CHUNK = CN * PLOW_WAVES;
    const unsigned nchunk = n_exp / CHUNK;
    if (blockDim.x != PLOW_THREADS || rows == 0u || rows % R != 0u || n_exp % CHUNK != 0u ||
        hidden % (8u * PLOW_WAVE) != 0u || !(flags & 8u) || gridDim.x != rows / R * nchunk)
        __builtin_trap();
    __shared__ unsigned lds[4096];
    __shared__ unsigned last;
    d_gemv_bf16_f32_tile<R, CN>(logit, x, w, rows, n_exp, hidden, blockIdx.x, gridDim.x);

    const unsigned group = blockIdx.x / nchunk;
    const unsigned ngroups = rows / R;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[group], 1u) == nchunk - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) counters[group] = 0u;
    d_moe_router_topk_pf(table + (size_t)group * R * (k + shared_tail) * 8,
                         (const bf16*)(logit + (size_t)group * R * n_exp),
                         (flags & 4u) ? bias : nullptr, n_exp, k, flags,
                         route_scale, R, 0u, 1u, (float*)lds, n_group, topk_group, nullptr,
                         nullptr, shared_tail);

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[ngroups], 1u) == ngroups - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) counters[ngroups] = 0u;
    d_moe_align_pf(meta, table, row_token, row_partidx, row_gate, rows, e_route, k_route, 0u, lds);
}
