#include "dev_isa.h"

#define PLOW_NV_HOPPER 1
#define PLOW_NV_FA_PIPE 1
#define PLOW_NV_FA_TMA 1
#define PLOW_NV_FA512_WG 1
#include "op_attention.cuh"

extern "C" __device__ unsigned plow_attention_sm90_hd512_wg32_abi = 1;
extern "C" __device__ unsigned plow_attention_head_dim = 512;
extern "C" __device__ unsigned plow_attention_query_tile = 64;
extern "C" __device__ unsigned plow_attention_kv_tile = 32;
extern "C" __device__ unsigned plow_attention_warps = 8;
extern "C" __device__ unsigned plow_block_pfattn_hd512 = 256;
extern "C" __device__ unsigned plow_arena_bytes_pfattn_hd512 =
    FA_PRE_SMEM_FLOATS(512, 64, 32) * sizeof(float);

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
    const uint2 a = reinterpret_cast<const uint2*>(p)[0];
    const uint2 b = reinterpret_cast<const uint2*>(p)[1];
    const uint2 c = reinterpret_cast<const uint2*>(p)[2];
    reinterpret_cast<uint2*>(&entry)[0] = a;
    reinterpret_cast<uint2*>(&entry)[1] = b;
    reinterpret_cast<uint2*>(&entry)[2] = c;
    return entry;
}

__device__ __forceinline__ void attention_body(const PlowDevInst* in, void* const* tensors,
                                             unsigned slice, unsigned nblk, float* arena) {
    const unsigned t0 = in->t[0], t1 = in->t[1], t2 = in->t[2], t3 = in->t[3];
    const unsigned t4 = in->t[4], t5 = in->t[5], t7 = in->t[7];
    __nv_bfloat16* const output =
        t5 == PLOW_TENSOR_NONE ? nullptr : static_cast<__nv_bfloat16*>(tensors[t5]);
    d_flash_prefill<512, 64, 32>(
        static_cast<float*>(tensors[t0]), static_cast<float*>(tensors[t1]),
        static_cast<const __nv_bfloat16*>(tensors[t2]),
        static_cast<const __nv_bfloat16*>(tensors[t3]),
        static_cast<const __nv_bfloat16*>(tensors[t4]),
        output, in->i[0], in->i[1], in->i[2], in->i[3],
        in->i[4], in->i[5], in->i[7], in->fj[1].u, in->fj[2].u, in->fj[0].f, slice, nblk,
        arena, nullptr, tensors[t7]);
}

extern "C" __global__ __maxnreg__(194)
void plow_sm90a_pfattn_hd512(PlowProgram prog) {
    extern __shared__ float arena[];
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const unsigned index = lo + blockIdx.x;
    if (index >= hi) return;

    {
        const PlowStreamEnt entry = attention_stream_ent(prog.gq_stream + index);
        if (entry.flags & PLOW_SE_XCTR) {
            if (threadIdx.x == 0) __trap();
        }
        for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
            const PlowWait wait = prog.waits[entry.wait_ofs + w];
            while (attention_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
        }
        __syncthreads();

        const PlowDevInst* const in = prog.insts + entry.inst;
        attention_body(in, prog.tensors, entry.slice, in->blocks, arena);
    }

    __syncthreads();
    const PlowStreamEnt entry = attention_stream_ent(prog.gq_stream + index);
    for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
        attention_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
}
