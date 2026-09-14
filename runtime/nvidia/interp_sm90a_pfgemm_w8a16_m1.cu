#include "dev_isa.h"
#define PLOW_NV_GEMMA 1
#define GV_MM_MAX 16
#include "op_gemm.cuh"

extern "C" __device__ unsigned plow_w8a16_prefill_m1_abi = 1;
extern "C" __device__ unsigned plow_w8a16_prefill_m1_max_rows = 1;
extern "C" __device__ unsigned plow_w8a16_prefill_m1_k_multiple = 16;
extern "C" __device__ unsigned plow_block_pfgemm_w8a16_m1 = 256;
extern "C" __device__ unsigned plow_arena_bytes_pfgemm_w8a16_m1 = 1;

extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_request_abi = 1;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_masked_padding_abi = 1;

struct W8a16M1Args {
    __nv_bfloat16* output;
    const __nv_bfloat16* x;
    const unsigned char* weights;
    const float* scales;
    unsigned m, n, k, nblk;
    bool valid;
};

__device__ __forceinline__ unsigned pfgemm_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ void pfgemm_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}

__device__ __forceinline__ PlowStreamEnt pfgemm_stream_ent(const PlowStreamEnt* p) {
    PlowStreamEnt entry;
    const uint2 a = reinterpret_cast<const uint2*>(p)[0];
    const uint2 b = reinterpret_cast<const uint2*>(p)[1];
    const uint2 c = reinterpret_cast<const uint2*>(p)[2];
    reinterpret_cast<uint2*>(&entry)[0] = a;
    reinterpret_cast<uint2*>(&entry)[1] = b;
    reinterpret_cast<uint2*>(&entry)[2] = c;
    return entry;
}

extern "C" __global__ __launch_bounds__(256)
void plow_sm90a_pfgemm_w8a16_m1(PlowProgram prog) {
    __shared__ PlowStreamEnt shared_entry;
    __shared__ W8a16M1Args args;
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    for (unsigned index = lo + blockIdx.x; index < hi; index += gridDim.x) {
        if (threadIdx.x == 0) shared_entry = pfgemm_stream_ent(prog.gq_stream + index);
        __syncthreads();
        const PlowStreamEnt entry = shared_entry;
        if (entry.flags & PLOW_SE_XCTR) __trap();
        for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
            const PlowWait wait = prog.waits[entry.wait_ofs + w];
            while (pfgemm_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const PlowDevInst* const in = prog.insts + entry.inst;
            args.valid = in->op == PLOW_DOP_GEMM_FP8 && in->i[0] == 1 &&
                in->i[1] > 0 && in->i[2] > 0 && in->i[2] % 16 == 0 && in->i[4] == 0 &&
                in->t[0] != PLOW_TENSOR_NONE && in->t[1] != PLOW_TENSOR_NONE &&
                in->t[2] != PLOW_TENSOR_NONE && in->t[4] != PLOW_TENSOR_NONE &&
                in->t[3] == PLOW_TENSOR_NONE && in->t[5] == PLOW_TENSOR_NONE &&
                in->t[6] == PLOW_TENSOR_NONE && in->t[7] == PLOW_TENSOR_NONE &&
                entry.slice < in->blocks;
            if (args.valid) {
                args.output = static_cast<__nv_bfloat16*>(prog.tensors[in->t[0]]);
                args.x = static_cast<const __nv_bfloat16*>(prog.tensors[in->t[1]]);
                args.weights = static_cast<const unsigned char*>(prog.tensors[in->t[2]]);
                args.scales = static_cast<const float*>(prog.tensors[in->t[4]]);
            }
            args.m = in->i[0];
            args.n = in->i[1];
            args.k = in->i[2];
            args.nblk = in->blocks;
        }
        __syncthreads();
        if (!args.valid) __trap();
        d_gemv_fp8(args.output, args.x, args.weights, args.scales,
                    args.m, args.n, args.k, entry.slice, args.nblk);

        __syncthreads();
        for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
            pfgemm_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
        __syncthreads();
    }
}
