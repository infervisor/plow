/* tp_dcp_gather_kernels.hip — device side of tp_dcp_gather_bench.c: the DCP owner gather
 * (ops 181 DCP_KV_PACK / 182 XDCP_GATHER, op_collective.h) on 8 GPUs, byte-exact against a host
 * oracle, timed per phase. Built with -DPLOW_DCP_GATHER=1. */
#include "../common/dev_isa.h"
#include "../amd/op_collective.h"

__device__ __forceinline__ uint64_t tdg_now(void) { return __builtin_amdgcn_s_memrealtime(); }

#define TDG_PARAMS                                                                              \
    bf16* slot, const int* idx, const int* kv_len, const unsigned char* ckv, const bf16* krot,  \
    const float* kv_scale, unsigned char* gckv, bf16* gkrot, float* gscale, int* glen,          \
    const void* const* peers, uint32_t* status, uint64_t* ts, size_t xoff, uint64_t deadline,   \
    uint32_t n_batch, uint32_t K, uint32_t local_stride, uint32_t page_shift,                   \
    uint32_t degree_shift, uint32_t rank, uint32_t n_gpu, uint32_t slot_bytes, uint32_t nwg

/* Kernargs in declaration order; the host passes the same fields as one packed-by-alignment
 * struct (idx == null selects the prefix form). */
extern "C" __global__ void tdg_pack(TDG_PARAMS) {
    const uint64_t t0 = tdg_now();
    const uint32_t shard = rank & ((1u << degree_shift) - 1u);
    d_dcp_kv_pack(slot, idx, kv_len, ckv, krot, kv_scale, n_batch, K, local_stride, page_shift,
                  degree_shift, shard, blockIdx.x, nwg);
    __syncthreads();
    if (threadIdx.x == 0) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tdg_now(); }
}

extern "C" __global__ void tdg_gather(TDG_PARAMS) {
    const uint64_t t0 = tdg_now();
    d_xdcp_gather_mega(peers, n_gpu, rank, xoff, 1u, deadline, status, blockIdx.x, nwg,
                       slot_bytes, idx, kv_len, gckv, gkrot, gscale, glen, n_batch, K,
                       page_shift, degree_shift);
    __syncthreads();
    if (threadIdx.x == 0) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tdg_now(); }
}

extern "C" __global__ void tdg_scatter(const int* pos, const unsigned char* ckv_st, const bf16* krot_st,
                                       const float* scale_st, unsigned char* ckv, bf16* krot,
                                       float* kv_scale, uint64_t* ts, uint32_t rows,
                                       uint32_t local_stride, uint32_t page_shift,
                                       uint32_t degree_shift, uint32_t shard, uint32_t batched,
                                       uint32_t nwg) {
    const uint64_t t0 = tdg_now();
    d_dcp_kv_scatter(pos, ckv_st, krot_st, scale_st, ckv, krot, kv_scale, rows, local_stride,
                     page_shift, degree_shift, shard, batched, blockIdx.x, nwg);
    __syncthreads();
    if (threadIdx.x == 0) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = tdg_now(); }
}
