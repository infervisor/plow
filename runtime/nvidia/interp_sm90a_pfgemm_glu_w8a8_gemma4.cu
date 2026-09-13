#include "dev_isa.h"

#define PLOW_NV_HOPPER 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_GEMM_ONLY 1
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_GEMMA 1
#define PLOW_NV_W8A8 1
#define PGM90_UNI_BN256 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_WS384_BN 128
#define PGM90_UNI256_NS 4
#include "op_gemm.cuh"

extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_abi = 1;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_min_rows = 4096;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_max_rows = 8192;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_n = 15360;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_k = 3840;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_stages = PGM90_GEMMA4_GLU_STAGES;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_bm = PGM90_GEMMA4_GLU_BM;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_bn = PGM90_GEMMA4_GLU_BN;
extern "C" __device__ unsigned plow_pfgemm_glu_w8a8_gemma4_bk = PGM90_GEMMA4_GLU_W8A8_BK;
extern "C" __device__ unsigned plow_block_pfgemm_glu_w8a8_gemma4 = 384;
extern "C" __device__ unsigned plow_arena_bytes_pfgemm_glu_w8a8_gemma4 =
    PGM90_GEMMA4_GLU_ARENA_BYTES;
extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_request_abi = 1;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_masked_padding_abi = 1;

__device__ __forceinline__ unsigned gemma4_glu_w8a8_ctr_poll(const unsigned* address) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void gemma4_glu_w8a8_ctr_signal(unsigned* address) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(address) : "memory");
}

__device__ __forceinline__ PlowStreamEnt gemma4_glu_w8a8_stream_ent(const PlowStreamEnt* address) {
    PlowStreamEnt entry;
    reinterpret_cast<uint2*>(&entry)[0] = reinterpret_cast<const uint2*>(address)[0];
    reinterpret_cast<uint2*>(&entry)[1] = reinterpret_cast<const uint2*>(address)[1];
    reinterpret_cast<uint2*>(&entry)[2] = reinterpret_cast<const uint2*>(address)[2];
    return entry;
}

template <bool PROD>
__device__ __forceinline__ void gemma4_glu_w8a8_role_loop(
    const PlowProgram& prog, unsigned* cursor, unsigned lo, unsigned hi,
    volatile unsigned* claim, void* arena) {
    for (;;) {
        __syncthreads();
        if (threadIdx.x == 0) *claim = atomicAdd(cursor, 1u);
        __syncthreads();
        const unsigned index = lo + *claim;
        if (index >= hi) break;

        const PlowStreamEnt entry = gemma4_glu_w8a8_stream_ent(prog.gq_stream + index);
        if (entry.flags & PLOW_SE_XCTR) __trap();
        for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
            const PlowWait wait = prog.waits[entry.wait_ofs + w];
            while (gemma4_glu_w8a8_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
        }
        __syncthreads();

        const PlowDevInst* const in = prog.insts + entry.inst;
        const bool exact_rows = in->i[0] == 4096 || in->i[0] == 8192;
        const bool tensors_valid = in->t[0] != PLOW_TENSOR_NONE &&
            in->t[1] != PLOW_TENSOR_NONE && in->t[2] != PLOW_TENSOR_NONE &&
            in->t[3] != PLOW_TENSOR_NONE && in->t[4] != PLOW_TENSOR_NONE &&
            in->t[5] != PLOW_TENSOR_NONE && in->t[6] != PLOW_TENSOR_NONE &&
            in->t[7] == PLOW_TENSOR_NONE;
        if (in->op != PLOW_DOP_GEMM_GLU_FP8 || !exact_rows || in->i[1] != 15360 ||
            in->i[2] != 3840 || !in->i[3] || in->i[4] != 0 || in->i[5] != 0 ||
            !in->i[6] || !in->i[7] || !in->blocks || entry.slice >= in->blocks ||
            !tensors_valid || in->fj[0].u || in->fj[1].u || in->fj[2].u) {
            __trap();
            return;
        }

        d_gemm_glu_w8a8_sm90_tma_ws384_gemma4_role<PROD>(
            static_cast<__nv_bfloat16*>(prog.tensors[in->t[0]]), prog.tensors[in->i[6]],
            prog.tensors[in->i[7]], prog.tensors[in->i[3]],
            static_cast<const float*>(prog.tensors[in->t[3]]),
            static_cast<const float*>(prog.tensors[in->t[4]]),
            static_cast<const float*>(prog.tensors[in->t[6]]), in->i[0], entry.slice,
            in->blocks, arena);

        __syncthreads();
        for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
            gemma4_glu_w8a8_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
    }
}

extern "C" __global__ __maxnreg__(160)
void plow_sm90a_pfgemm_glu_w8a8_gemma4(PlowProgram prog) {
    extern __shared__ unsigned char arena[];
    __shared__ unsigned claim;
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    unsigned* const cursor = PLOW_CTR(prog.gq_cursor, prog.cur_seg);
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        gemma4_glu_w8a8_role_loop<true>(prog, cursor, lo, hi, &claim, arena);
    } else {
        sm90_reg_inc(224);
        gemma4_glu_w8a8_role_loop<false>(prog, cursor, lo, hi, &claim, arena);
    }
}
