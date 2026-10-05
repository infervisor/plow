// Role wrapper for generated flash-prefill bodies (scripts/gen_kernels/build_catalog.py).
// PLOW_GEN_BODY names the generated header: it defines plow_gen_call(bx, by, Q, K, V, O, heads,
// qlen, kvlen, scale, window, kv_mask), one (query tile bx, head by) item of causal attention over
// Q/O [qlen][heads][hd] and one KV head's K/V rows (position p at row p & kv_mask; window 0 =
// global), plus the PLOW_GEN_* geometry macros.
// The wrapper owns the packet contract: the packed request table, the KV slot layout
// [slot][kv head][kv_stride][hd], zeroing the padded rows, and the segment successor counters.
#include "dev_isa.h"
#include PLOW_GEN_BODY

static_assert(PLOW_GEN_ARENA >= 116 * 1024,
              "generated flash prefill must claim one CTA per SM (role grid == packet grid)");

extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
#ifndef PLOW_GEN_FP8_KV
#define PLOW_GEN_FP8_KV 0
#endif
// FP8-KV mode (FlashPrefillFp8): e4m3 K/V, one f32 scale per (position, KV head) row passed in
// the opart / mlpart slots (ABI 2), the packed request table in the op's i[4] handle.
#if PLOW_GEN_FP8_KV
extern "C" __device__ unsigned plow_gen_flash_prefill_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_request_abi = 1;
extern "C" __device__ __constant__ unsigned plow_pf_fp8_masked_padding_abi = 1;
typedef fp8_e4_t plow_gen_kv_t;
#else
extern "C" __device__ unsigned plow_gen_flash_prefill_abi = 1;
typedef bfloat16_t plow_gen_kv_t;
#endif
extern "C" __device__ unsigned plow_gen_block = PLOW_GEN_THREADS;
extern "C" __device__ unsigned plow_gen_arena_bytes = PLOW_GEN_ARENA;
extern "C" __device__ unsigned plow_attention_head_dim = PLOW_GEN_HEAD_DIM;
extern "C" __device__ unsigned plow_attention_query_tile = PLOW_GEN_BM;
extern "C" __device__ unsigned plow_attention_kv_tile = PLOW_GEN_BN;
extern "C" __device__ unsigned plow_attention_warps = PLOW_GEN_THREADS / 32;

typedef struct {
    const int* requests;
    float* opart;
    float* mlpart;
    const bfloat16_t* q;
    const plow_gen_kv_t* k;
    const plow_gen_kv_t* v;
    bfloat16_t* output;
    const void* mapkv;
    const PlowStreamEnt* entries;
    const unsigned* succs;
    unsigned* counters;
    unsigned seq_q;
    unsigned seq_kv;
    unsigned q_pos0;
    unsigned kv_stride;
    unsigned kv_mask;
    float scale;
    unsigned n_head;
    unsigned n_kv_head;
    unsigned window;
    unsigned reserved;
} PlowGenFlashPrefill;
static_assert(sizeof(PlowGenFlashPrefill) == 128, "generated flash prefill role ABI");

__device__ __forceinline__ void gen_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}

__device__ __forceinline__ unsigned gen_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ PlowStreamEnt gen_stream_ent(const PlowStreamEnt* p) {
    PlowStreamEnt entry;
    for (int i = 0; i < 3; ++i)
        reinterpret_cast<uint2*>(&entry)[i] = reinterpret_cast<const uint2*>(p)[i];
    return entry;
}

// Items run heaviest-first inside each request (descending query tile, heads inner) and are
// dealt to the persistent CTAs in zig-zag rounds, which evens the causal triangle. `a` lives in
// param or shared space and the request row is re-read every round: only `round` stays live
// across the generated body, which already uses the whole register file.
__device__ __forceinline__ void gen_run(const PlowGenFlashPrefill& a) {
    for (unsigned round = 0;; ++round) {
        unsigned local = round * gridDim.x +
                         ((round & 1) ? gridDim.x - 1 - blockIdx.x : blockIdx.x);
        const unsigned count = a.requests ? (unsigned)a.requests[0] : 1u;
        // Unpacked: one request of seq_kv - q_pos0 real rows (an unpacked chunk's last bucket keeps
        // i[0] at the rung, so rows past them are padding).
        unsigned r = 0, q0 = 0, qlen = a.seq_kv - a.q_pos0, slot = 0, kvlen = a.seq_kv;
        for (; r < count; ++r) {
            if (a.requests) {
                q0 = (unsigned)a.requests[1 + 4 * r];
                qlen = (unsigned)a.requests[2 + 4 * r];
                slot = (unsigned)a.requests[3 + 4 * r];
                kvlen = (unsigned)a.requests[4 + 4 * r];
            }
            const unsigned items = (qlen + PLOW_GEN_BM - 1) / PLOW_GEN_BM * a.n_head;
            if (local < items) break;
            local -= items;
        }
        if (r == count) break;
        const unsigned tiles = (qlen + PLOW_GEN_BM - 1) / PLOW_GEN_BM;
        const unsigned tile = tiles - 1 - local / a.n_head, head = local % a.n_head;
        const size_t qoff = (size_t)q0 * a.n_head * PLOW_GEN_HEAD_DIM;
        const size_t kvrow =
            ((size_t)slot * a.n_kv_head + head / (a.n_head / a.n_kv_head)) * a.kv_stride;
        const size_t kvoff = kvrow * PLOW_GEN_HEAD_DIM;
        __syncthreads();
#if PLOW_GEN_FP8_KV
        plow_gen_call_fp8((int)tile, (int)head, a.q + qoff, a.k + kvoff, a.v + kvoff,
                          a.opart + kvrow, a.mlpart + kvrow, a.output + qoff, (int)a.n_head,
                          (int)qlen, (int)kvlen, a.scale, (int)a.window, a.kv_mask);
#else
        (void)kvrow;
        plow_gen_call((int)tile, (int)head, a.q + qoff, a.k + kvoff, a.v + kvoff,
                      a.output + qoff, (int)a.n_head, (int)qlen, (int)kvlen, a.scale,
                      (int)a.window, a.kv_mask);
#endif
    }

    unsigned real = a.seq_kv - a.q_pos0;
    if (a.requests) {
        const unsigned count = (unsigned)a.requests[0];
        real = count ? (unsigned)a.requests[4 * count - 3] + (unsigned)a.requests[4 * count - 2]
                     : 0u;
    }
    const size_t begin = (size_t)real * a.n_head * PLOW_GEN_HEAD_DIM;
    const size_t end = (size_t)a.seq_q * a.n_head * PLOW_GEN_HEAD_DIM;
    for (size_t i = begin + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < end;
         i += (size_t)gridDim.x * blockDim.x)
        a.output[i] = bfloat16_t(0.0f);
}

__device__ __forceinline__ void gen_signal(const PlowStreamEnt* entry_ptr, const unsigned* succs,
                                           unsigned* counters) {
    __syncthreads();
    const PlowStreamEnt entry = gen_stream_ent(entry_ptr);
    for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
        gen_ctr_signal(PLOW_CTR(counters, succs[entry.succ_ofs + s]));
}

extern "C" __global__ __launch_bounds__(PLOW_GEN_THREADS, 1)
void plow_gen_flash_prefill_direct(const __grid_constant__ PlowGenFlashPrefill args) {
    if (!args.requests || args.n_kv_head == 0 ||
        args.n_head % args.n_kv_head) {
        __trap();
        return;
    }
    gen_run(args);
    gen_signal(args.entries + blockIdx.x, args.succs, args.counters);
}

// Packet entry: one FlashPrefill instruction per segment, `blocks` == gridDim.x.
extern "C" __global__ __launch_bounds__(PLOW_GEN_THREADS, 1)
void plow_gen_flash_prefill(PlowProgram prog) {
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const unsigned index = lo + blockIdx.x;
    if (index >= hi) return;
    const PlowStreamEnt entry = gen_stream_ent(prog.gq_stream + index);
    if (entry.flags & PLOW_SE_XCTR) {
        if (threadIdx.x == 0) __trap();
    }
    for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
        const PlowWait wait = prog.waits[entry.wait_ofs + w];
        while (gen_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {}
    }
    __syncthreads();
    const PlowDevInst* in = prog.insts + entry.inst;
    void* const* t = prog.tensors;
#if PLOW_GEN_FP8_KV
    const unsigned op = PLOW_DOP_FLASH_PREFILL_FP8;
#else
    const unsigned op = PLOW_DOP_FLASH_PREFILL;
#endif
    if (in->op != op || in->i[6] != PLOW_GEN_HEAD_DIM ||
        in->i[7] != 1 || in->t[5] == PLOW_TENSOR_NONE ||
        in->i[3] == 0 || in->i[2] % in->i[3]) {
        __trap();
        return;
    }
    __shared__ PlowGenFlashPrefill a;
    if (threadIdx.x == 0) {
#if PLOW_GEN_FP8_KV
        const unsigned handle = in->i[4] & ~(1u << 31);
        const bool packed = in->i[4] & (1u << 31);
        a.requests = packed && handle < PLOW_TENSOR_NONE ? static_cast<const int*>(t[handle])
                                                         : nullptr;
        if (packed && !a.requests) __trap();
        a.opart = static_cast<float*>(t[in->t[6]]);
        a.mlpart = static_cast<float*>(t[in->t[7]]);
        a.q_pos0 = packed ? 0u : in->i[4];
#else
        a.requests = in->t[6] == PLOW_TENSOR_NONE ? nullptr : static_cast<const int*>(t[in->t[6]]);
        a.q_pos0 = in->i[4];
#endif
        a.q = static_cast<const bfloat16_t*>(t[in->t[2]]);
        a.k = static_cast<const plow_gen_kv_t*>(t[in->t[3]]);
        a.v = static_cast<const plow_gen_kv_t*>(t[in->t[4]]);
        a.output = static_cast<bfloat16_t*>(t[in->t[5]]);
        a.seq_q = in->i[0];
        a.seq_kv = in->i[1];
        a.kv_stride = in->fj[1].u;
        a.kv_mask = in->fj[2].u;
        a.scale = in->fj[0].f;
        a.n_head = in->i[2];
        a.n_kv_head = in->i[3];
        a.window = in->i[5];
        if (!a.requests && (a.seq_kv <= a.q_pos0 || a.q_pos0 + a.seq_q < a.seq_kv)) __trap();
    }
    __syncthreads();
    gen_run(a);
    gen_signal(prog.gq_stream + prog.gq_seg_ofs[prog.cur_seg] + blockIdx.x, prog.succs,
               prog.counters);
}
