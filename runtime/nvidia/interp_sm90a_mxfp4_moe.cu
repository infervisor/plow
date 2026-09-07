#include "dev_isa.h"

#define PLOW_NV_MXFP4_PROJ 0
#define PLOW_NV_MXFP4_MOE 1
#define PLOW_NV_MOE_COMMON 0
#include "sm120_common.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "op_moe.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "MXFP4 MoE role requires sm_90a"
#endif

extern "C" __device__ unsigned plow_mxfp4_moe_sm90_abi = 1;
extern "C" __device__ unsigned plow_block_mxfp4_moe = 256;
extern "C" __device__ unsigned plow_arena_bytes_mxfp4_moe = 4;
extern "C" __device__ unsigned plow_mxfp4_moe_ctas_per_sm = 4;

__device__ __forceinline__ unsigned mxfp4_moe_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ void mxfp4_moe_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}

__device__ __forceinline__ PlowStreamEnt mxfp4_moe_stream_ent(const PlowStreamEnt* p) {
    PlowStreamEnt entry;
    const uint2 a = reinterpret_cast<const uint2*>(p)[0];
    const uint2 b = reinterpret_cast<const uint2*>(p)[1];
    const uint2 c = reinterpret_cast<const uint2*>(p)[2];
    reinterpret_cast<uint2*>(&entry)[0] = a;
    reinterpret_cast<uint2*>(&entry)[1] = b;
    reinterpret_cast<uint2*>(&entry)[2] = c;
    return entry;
}

extern "C" __global__ __launch_bounds__(256, 4)
void plow_sm90a_mxfp4_moe(PlowProgram prog) {
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const unsigned original_blocks = hi - lo;
    if (original_blocks == 0) return;

    const PlowStreamEnt entry =
        mxfp4_moe_stream_ent(prog.gq_stream + lo + blockIdx.x % original_blocks);
    if (entry.flags & PLOW_SE_XCTR) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
        const PlowWait wait = prog.waits[entry.wait_ofs + w];
        while (mxfp4_moe_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
    }
    __syncthreads();

    const PlowDevInst* const in = prog.insts + entry.inst;
    void* const* const tensors = prog.tensors;
#define TEN(i) tensors[in->t[i]]
    switch (in->op) {
    case PLOW_DOP_MOE_GLU_MX:
        d_moe_glu_mx((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1),
                     (const unsigned char*)TEN(2), (const uint8_t*)TEN(3),
                     (const uint8_t*)TEN(4), (const __nv_bfloat16*)TEN(5), in->i[0],
                     in->i[1], in->i[2], in->i[3], in->i[4], in->i[5], in->i[6],
                     in->fj[0].f, in->fj[1].f, blockIdx.x, gridDim.x);
        break;
    case PLOW_DOP_MOE_DOWN_MX:
        d_moe_down_mx((float*)TEN(0), (const __nv_bfloat16*)TEN(1),
                      (const unsigned char*)TEN(2), (const uint8_t*)TEN(3),
                      (const uint8_t*)TEN(4), (const __nv_bfloat16*)TEN(5), in->i[0],
                      in->i[1], in->i[2], in->i[3], in->i[6], blockIdx.x, gridDim.x);
        break;
    default:
        __trap();
    }
#undef TEN

    __syncthreads();
    // One original entry signals once. The role launch and its successor are ordered on
    // the same stream, so the successor cannot observe these counters until all replicas finish.
    if (blockIdx.x < original_blocks) {
        const PlowStreamEnt signal_entry =
            mxfp4_moe_stream_ent(prog.gq_stream + lo + blockIdx.x);
        for (unsigned s = threadIdx.x; s < signal_entry.succ_len; s += blockDim.x)
            mxfp4_moe_ctr_signal(
                PLOW_CTR(prog.counters, prog.succs[signal_entry.succ_ofs + s]));
    }
}
