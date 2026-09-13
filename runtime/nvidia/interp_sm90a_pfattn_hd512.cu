#include "dev_isa.h"

#define PLOW_NV_HOPPER 1
#define PLOW_NV_FA_PIPE 1
#ifndef PLOW_NV_FA_TMA
#define PLOW_NV_FA_TMA 1
#endif
#ifndef PLOW_NV_FA512_WG
#define PLOW_NV_FA512_WG 0
#endif
#ifndef PLOW_NV_FA512_PX4_BQ64
#define PLOW_NV_FA512_PX4_BQ64 0
#endif
#ifndef PLOW_NV_FA_SCORE_SWIZZLE
#define PLOW_NV_FA_SCORE_SWIZZLE PLOW_NV_FA512_PX4_BQ64
#endif
#include "op_attention.cuh"

#if PLOW_NV_FA512_PX4_BQ64 && (PLOW_NV_FA512_WG || !PLOW_NV_PACKED_REQUEST)
#error "HD512 px4 BQ64 requires the packed non-WGMMA object"
#endif
#if PLOW_NV_FA_TMA_DESC && !PLOW_NV_FA512_PX4_BQ64
#error "HD512 descriptor TMA is only implemented by the px4 BQ64 object"
#endif

#if PLOW_NV_FA512_KV64 && !PLOW_NV_FA512_WG
#error "HD512 KV64 requires the WGMMA body"
#endif
#ifndef PLOW_NV_FA512_KV16
#define PLOW_NV_FA512_KV16 0
#endif
#if PLOW_NV_FA512_KV16 && !PLOW_NV_FA512_WG
#error "HD512 KV16 WGMMA requires the WGMMA body"
#endif
#if PLOW_NV_FA512_KV16 && PLOW_NV_FA512_KV64
#error "HD512 KV16 and KV64 are mutually exclusive"
#endif
constexpr int FA512_KV_TILE = PLOW_NV_FA512_KV64 ? 64 : (PLOW_NV_FA512_KV16 ? 16 : 32);
#ifndef PLOW_NV_FA512_FIXED_HEADS
#define PLOW_NV_FA512_FIXED_HEADS 0
#endif

#if PLOW_NV_PACKED_REQUEST
extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
#if defined(PLOW_NV_MASKED_PADDING) && PLOW_NV_MASKED_PADDING
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
#endif
#endif

#if PLOW_NV_FA512_PX4_BQ64
extern "C" __device__ unsigned plow_attention_sm90_hd512_px4_bq64_abi = 5;
#else
extern "C" __device__ unsigned plow_attention_sm90_hd512_wg32_abi = 1;
#endif
extern "C" __device__ unsigned plow_attention_head_dim = 512;
extern "C" __device__ unsigned plow_attention_query_tile =
    PLOW_NV_FA512_PX4_BQ64 || PLOW_NV_FA512_WG ? 64 : 32;
extern "C" __device__ unsigned plow_attention_kv_tile = PLOW_NV_FA512_WG ? FA512_KV_TILE : 16;
extern "C" __device__ unsigned plow_attention_warps = PLOW_NV_FA512_PX4_BQ64 ? 16 : 8;
extern "C" __device__ unsigned plow_attention_score_partitions =
    PLOW_NV_FA512_N_SPLIT && FA512_KV_TILE == 64 ? 2 : 1;
#if PLOW_NV_FA512_PX4_BQ64
extern "C" __device__ unsigned plow_attention_packed_only = 1;
extern "C" __device__ unsigned plow_attention_n_head = 16;
extern "C" __device__ unsigned plow_attention_n_kv_head = 1;
extern "C" __device__ unsigned plow_attention_global = 1;
extern "C" __device__ unsigned plow_attention_nsplit = 1;
extern "C" __device__ unsigned plow_attention_direct_entry = 1;
extern "C" __device__ unsigned plow_attention_score_swizzle = PLOW_NV_FA_SCORE_SWIZZLE;
extern "C" __device__ unsigned plow_attention_tma_desc = PLOW_NV_FA_TMA_DESC;
extern "C" __device__ unsigned plow_block_pfattn_hd512_px4_bq64 = 512;
extern "C" __device__ unsigned plow_arena_bytes_pfattn_hd512_px4_bq64 =
#if PLOW_NV_FA_TMA_DESC
    FA_PX4_TMA_DESC_SMEM_FLOATS(512, 64, 16) * sizeof(float);
#else
    FA_PX4_SMEM_FLOATS(512, 64, 16) * sizeof(float);
#endif
#else
extern "C" __device__ unsigned plow_block_pfattn_hd512 = 256;
extern "C" __device__ unsigned plow_arena_bytes_pfattn_hd512 =
    (PLOW_NV_FA512_WG ? FA_PRE_SMEM_FLOATS(512, 64, FA512_KV_TILE)
                      : FA_PX4_SMEM_FLOATS(512, 32, 16)) *
    sizeof(float);
#endif

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

#if PLOW_NV_PACKED_REQUEST && !PLOW_NV_FA512_WG
__device__ __noinline__ void attention_packed_bkv16(
    const int* __restrict__ requests, float* __restrict__ opart,
    float* __restrict__ mlpart, const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output, unsigned seq_q, unsigned seq_kv,
    unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    unsigned nsplit, unsigned kv_stride, unsigned kv_mask, float scale,
    unsigned slice, unsigned nblk, const void* mapkv, float* arena) {
#if PLOW_NV_FA512_PX4_BQ64
    if (!requests) {
        __trap();
        return;
    }
#else
    if (!requests) {
        d_flash_prefill_px4<512, PLOW_NV_FA512_PX4_BQ64 ? 64 : 32, 16, false,
                            PLOW_NV_FA512_PX4_BQ64 ? 512 : 256>(
            opart, mlpart, q, k, v, output, seq_q, seq_kv, n_head, n_kv_head,
            q_pos0, window, nsplit, kv_stride, kv_mask, scale, slice, nblk, arena);
        return;
    }
#endif

    const unsigned count = (unsigned)requests[0];
    for (unsigned r = 0; r < count; ++r) {
        const unsigned q0 = (unsigned)requests[1 + 4 * r];
        const unsigned qlen = (unsigned)requests[2 + 4 * r];
        const unsigned slot = (unsigned)requests[3 + 4 * r];
        const unsigned kvlen = (unsigned)requests[4 + 4 * r];
        const size_t qoff = (size_t)q0 * (PLOW_NV_FA512_PX4_BQ64 ? 16 : n_head) * 512;
        const size_t kvoff =
            (size_t)slot * (PLOW_NV_FA512_PX4_BQ64 ? 1 : n_kv_head) * kv_stride * 512;
        const void* descriptor = mapkv ? (const void*)((const uint64_t*)mapkv)[slot] : nullptr;
        if (PLOW_NV_FA_TMA_DESC && !descriptor) {
            __trap();
            return;
        }
        d_flash_prefill_px4<512, PLOW_NV_FA512_PX4_BQ64 ? 64 : 32, 16, false,
                            PLOW_NV_FA512_PX4_BQ64 ? 512 : 256, PLOW_NV_FA_TMA_DESC,
                            PLOW_NV_FA_TMA_DESC>(
            opart + qoff * (PLOW_NV_FA512_PX4_BQ64 ? 1 : nsplit),
            mlpart + (size_t)q0 * (PLOW_NV_FA512_PX4_BQ64 ? 32 : n_head * nsplit * 2),
            q + qoff, k + kvoff, v + kvoff, output ? output + qoff : nullptr,
            qlen, kvlen, PLOW_NV_FA512_PX4_BQ64 ? 16 : n_head,
            PLOW_NV_FA512_PX4_BQ64 ? 1 : n_kv_head, kvlen - qlen,
            PLOW_NV_FA512_PX4_BQ64 ? 0 : window, PLOW_NV_FA512_PX4_BQ64 ? 1 : nsplit,
            kv_stride, kv_mask, scale, slice, nblk, arena, nullptr, nullptr, descriptor);
        __syncthreads();
    }

    if (output) {
        unsigned real = 0;
        if (count) {
            const unsigned last = count - 1;
            real =
                (unsigned)requests[1 + 4 * last] + (unsigned)requests[2 + 4 * last];
        }
        const size_t begin = (size_t)real * n_head * 512;
        const size_t end = (size_t)seq_q * n_head * 512;
        for (size_t i = begin + (size_t)slice * blockDim.x + threadIdx.x; i < end;
             i += (size_t)nblk * blockDim.x)
            output[i] = __float2bfloat16(0.0f);
    }
}
#endif

template <bool GEMMA = false>
__device__ __forceinline__ void attention_body(const PlowDevInst* in, void* const* tensors,
                                             unsigned slice, unsigned nblk, float* arena) {
    const unsigned t0 = in->t[0], t1 = in->t[1], t2 = in->t[2], t3 = in->t[3];
    const unsigned t4 = in->t[4], t5 = in->t[5], t7 = in->t[7];
    __nv_bfloat16* const output =
        t5 == PLOW_TENSOR_NONE ? nullptr : static_cast<__nv_bfloat16*>(tensors[t5]);
    const int* requests = nullptr;
#if PLOW_NV_PACKED_REQUEST
    if (in->t[6] != PLOW_TENSOR_NONE) requests = static_cast<const int*>(tensors[in->t[6]]);
#endif
#if PLOW_NV_FA512_WG
    if (in->op != PLOW_DOP_FLASH_PREFILL || in->i[6] != 512 || !output || in->i[7] != 1) {
        __trap();
        return;
    }
    d_flash_prefill_mux<512, 64, FA512_KV_TILE>(requests,
#elif PLOW_NV_PACKED_REQUEST
#if PLOW_NV_FA512_PX4_BQ64
    if (!requests || in->i[2] != 16 || in->i[3] != 1 || in->i[5] != 0 || in->i[7] != 1 ||
        !output) {
        __trap();
        return;
    }
#endif
    attention_packed_bkv16(requests,
#else
    d_flash_prefill<512, 32, 16>(
#endif
        static_cast<float*>(tensors[t0]), static_cast<float*>(tensors[t1]),
        static_cast<const __nv_bfloat16*>(tensors[t2]),
        static_cast<const __nv_bfloat16*>(tensors[t3]),
        static_cast<const __nv_bfloat16*>(tensors[t4]),
        output, in->i[0], in->i[1], GEMMA ? 16 : in->i[2], GEMMA ? 1 : in->i[3],
        in->i[4], GEMMA ? 0 : in->i[5], GEMMA ? 1 : in->i[7], in->fj[1].u,
        in->fj[2].u, in->fj[0].f, slice, nblk,
#if PLOW_NV_FA512_WG
        arena, t7 == PLOW_TENSOR_NONE ? nullptr : tensors[t7]);
#elif PLOW_NV_PACKED_REQUEST
        t7 == PLOW_TENSOR_NONE ? nullptr : tensors[t7], arena);
#else
        arena, nullptr, tensors[t7]);
#endif
}

#if PLOW_NV_FA512_PX4_BQ64
typedef struct {
    const int* requests;
    float* opart;
    float* mlpart;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    __nv_bfloat16* output;
    const void* mapkv;
    const PlowStreamEnt* entries;
    const unsigned* succs;
    unsigned* counters;
    unsigned seq_q;
    unsigned kv_stride;
} PlowHd512Px4Direct;
static_assert(sizeof(PlowHd512Px4Direct) == 96, "HD512 direct role ABI");

extern "C" __global__ __launch_bounds__(512, 1)
void plow_sm90a_pfattn_hd512_px4_bq64_direct(PlowHd512Px4Direct args) {
    extern __shared__ float arena[];
    const unsigned count = (unsigned)args.requests[0];
    for (unsigned r = 0; r < count; ++r) {
        const unsigned q0 = (unsigned)args.requests[1 + 4 * r];
        const unsigned qlen = (unsigned)args.requests[2 + 4 * r];
        const unsigned slot = (unsigned)args.requests[3 + 4 * r];
        const unsigned kvlen = (unsigned)args.requests[4 + 4 * r];
        const size_t qoff = (size_t)q0 * 16 * 512;
        const size_t kvoff = (size_t)slot * args.kv_stride * 512;
        const void* descriptor = args.mapkv ? (const void*)((const uint64_t*)args.mapkv)[slot]
                                            : nullptr;
        if (!descriptor) {
            __trap();
            return;
        }
        d_flash_prefill_px4<512, 64, 16, false, 512, PLOW_NV_FA_TMA_DESC,
                            PLOW_NV_FA_TMA_DESC>(
            args.opart + qoff, args.mlpart + (size_t)q0 * 32, args.q + qoff,
            args.k + kvoff, args.v + kvoff, args.output + qoff, qlen, kvlen, 16, 1,
            kvlen - qlen, 0, 1, args.kv_stride, 0xffffffffu, 1.0f,
            blockIdx.x, gridDim.x, arena, nullptr, nullptr, descriptor);
        __syncthreads();
    }

    unsigned real = 0;
    if (count) {
        const unsigned last = count - 1;
        real = (unsigned)args.requests[1 + 4 * last] +
               (unsigned)args.requests[2 + 4 * last];
    }
    const size_t begin = (size_t)real * 16 * 512;
    const size_t end = (size_t)args.seq_q * 16 * 512;
    for (size_t i = begin + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < end;
         i += (size_t)gridDim.x * blockDim.x)
        args.output[i] = __float2bfloat16(0.0f);

    __syncthreads();
    const PlowStreamEnt entry = attention_stream_ent(args.entries + blockIdx.x);
    for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
        attention_ctr_signal(PLOW_CTR(args.counters, args.succs[entry.succ_ofs + s]));
}
#endif

#if PLOW_NV_FA512_PX4_BQ64
#define PLOW_HD512_ENTRY plow_sm90a_pfattn_hd512_px4_bq64
#define PLOW_HD512_BLOCK 512
#else
#define PLOW_HD512_ENTRY plow_sm90a_pfattn_hd512
#define PLOW_HD512_BLOCK 256
#endif
extern "C" __global__ __launch_bounds__(PLOW_HD512_BLOCK, 1)
void PLOW_HD512_ENTRY(PlowProgram prog) {
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
        if (PLOW_NV_FA512_FIXED_HEADS && PLOW_NV_FA512_WG && in->i[2] == 16 && in->i[3] == 1 &&
            in->i[7] == 1 && in->i[5] == 0)
            attention_body<true>(in, prog.tensors, entry.slice, in->blocks, arena);
        else
            attention_body(in, prog.tensors, entry.slice, in->blocks, arena);
    }

    __syncthreads();
    const PlowStreamEnt entry = attention_stream_ent(prog.gq_stream + index);
    for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
        attention_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
}
