#include "dev_isa.h"

#define PLOW_NV_HOPPER 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_FA_PIPE 1
#define PLOW_NV_FA_TMA 1
#define PLOW_NV_FA_WPR 1
#define PLOW_NV_PACKED_REQUEST 1
#define PLOW_NV_PACKED_FA_WGMMA 1
#define PLOW_NV_PACKED_FA_TMA 1
#define PLOW_NV_MASKED_PADDING 1
#include "op_attention.cuh"

extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
extern "C" __device__ unsigned plow_attention_sm90_hd256_bkv32_abi = 1;
extern "C" __device__ unsigned plow_attention_head_dim = 256;
extern "C" __device__ unsigned plow_attention_query_tile = 64;
extern "C" __device__ unsigned plow_attention_kv_tile = 32;
extern "C" __device__ unsigned plow_attention_warps = 8;
extern "C" __device__ unsigned plow_block_pfattn_hd256_bkv32 = 256;
extern "C" __device__ unsigned plow_arena_bytes_pfattn_hd256_bkv32 =
    FA_PRE_SMEM_FLOATS(256, 64, 32) * sizeof(float);

__device__ __forceinline__ unsigned attention_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ void attention_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}

__device__ __forceinline__ PlowStreamEnt attention_stream_ent(const PlowStreamEnt* p) {
    PlowStreamEnt entry;
    reinterpret_cast<uint2*>(&entry)[0] = reinterpret_cast<const uint2*>(p)[0];
    reinterpret_cast<uint2*>(&entry)[1] = reinterpret_cast<const uint2*>(p)[1];
    reinterpret_cast<uint2*>(&entry)[2] = reinterpret_cast<const uint2*>(p)[2];
    return entry;
}

extern "C" __global__
void plow_sm90a_pfattn_hd256_bkv32(PlowProgram prog) {
    extern __shared__ float arena[];
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const unsigned index = lo + blockIdx.x;
    if (index >= hi) return;

    const PlowStreamEnt entry = attention_stream_ent(prog.gq_stream + index);
    if (entry.flags & PLOW_SE_XCTR) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
        const PlowWait wait = prog.waits[entry.wait_ofs + w];
        while (attention_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
    }
    __syncthreads();

    const PlowDevInst* const in = prog.insts + entry.inst;
    if (in->op != PLOW_DOP_FLASH_PREFILL || in->i[0] == 0 || in->i[1] == 0 ||
        in->i[2] != 16 || in->i[3] != 8 || in->i[5] != 1024 || in->i[6] != 256 ||
        in->i[7] != 1 || in->t[5] == PLOW_TENSOR_NONE || in->t[7] == PLOW_TENSOR_NONE) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const int* requests = in->t[6] == PLOW_TENSOR_NONE
        ? nullptr : static_cast<const int*>(prog.tensors[in->t[6]]);
    d_flash_prefill_mux<256, 64, 32>(
        requests, static_cast<float*>(prog.tensors[in->t[0]]),
        static_cast<float*>(prog.tensors[in->t[1]]),
        static_cast<const __nv_bfloat16*>(prog.tensors[in->t[2]]),
        static_cast<const __nv_bfloat16*>(prog.tensors[in->t[3]]),
        static_cast<const __nv_bfloat16*>(prog.tensors[in->t[4]]),
        static_cast<__nv_bfloat16*>(prog.tensors[in->t[5]]),
        in->i[0], in->i[1], 16, 8, in->i[4], 1024, 1, in->fj[1].u, in->fj[2].u,
        in->fj[0].f, entry.slice, in->blocks, arena, prog.tensors[in->t[7]]);

    __syncthreads();
    for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
        attention_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
}
