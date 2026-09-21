/* attn_softmax_sm90a.cu — causal masked softmax over a dense score tile, in place.
 *
 * The vendor-GEMM prefill attention route (PLOW_PF_ATTN_GEMM) computes S = Q.K^T with one
 * cuBLASLt GEMM per query-row tile, runs this kernel over S, then P.V with a second GEMM.
 * S is [rows][ld], row-major, rows = query_rows * heads (score row i belongs to query row
 * i / heads). Query row q of the tile sees columns [0, first + q); everything from there to
 * `cols` is written as 0 so the P.V GEMM can run over the whole tile.
 *
 * Four entries, same argument block:
 *   plow_attn_softmax_w      S is bf16, P overwrites it (row pitch ld bf16 elements); one score
 *                            row per WARP, coalesced. The runtime's entry.
 *   plow_attn_softmax_w_f32  S is f32 (the GEMM widens), P is bf16 at the START of each f32 row,
 *                            i.e. row pitch 2*ld bf16 elements. The bf16 store at element j sits
 *                            at byte 2j, never ahead of the f32 load at byte 4j.
 *   plow_attn_softmax / _f32 the same with one row per thread (strided across the warp; kept
 *                            for the harness A/B: 0.46 vs 0.34 ms on a 32768 x 4096 tile).
 *
 * Scores arrive in the LOG2 domain (the GEMM's alpha carries scale * log2(e)), so every exp
 * is the hardware exp2 the native flash bodies use (op_attention.cuh FA_EXP).
 *
 * Grid-strided plain launches, no counters, no smem. `ld` must be a multiple of 8 so every row
 * starts on a 32-byte boundary and whole-vector loads/stores of a row's last partial vector
 * stay inside the row.
 */
#include <cuda_bf16.h>

#ifndef PLOW_ATTN_SM_UN
#define PLOW_ATTN_SM_UN 4
#endif
#define PLOW_ATTN_SM_NEG_INF (-3.0e38f)

/* The harness (experiments/attn_gemm_softmax_h100.cu) includes this file into an executable,
 * where device globals do not link. */
#ifndef PLOW_ATTN_SM_HARNESS
extern "C" __device__ unsigned plow_attn_softmax_abi = 1;
extern "C" __device__ unsigned plow_block_attn_softmax = 256;
#endif

struct asm_bf16v8 {
    __nv_bfloat16 x[8];
};
struct asm_f32v8 {
    float x[8];
};
__device__ __forceinline__ void asm_st8(__nv_bfloat16* p, const asm_bf16v8& v) {
    *(uint4*)p = *(const uint4*)&v;
}
__device__ __forceinline__ float asm_ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
/* Elements [j, j+8) of a score row as f32. */
template <bool F32>
__device__ __forceinline__ asm_f32v8 asm_ld8(const void* row, unsigned j) {
    asm_f32v8 r;
    if (F32) {
        const float4* p = (const float4*)((const float*)row + j);
        const float4 a = p[0], b = p[1];
        r.x[0] = a.x, r.x[1] = a.y, r.x[2] = a.z, r.x[3] = a.w;
        r.x[4] = b.x, r.x[5] = b.y, r.x[6] = b.z, r.x[7] = b.w;
    } else {
        asm_bf16v8 v;
        *(uint4*)&v = *(const uint4*)((const __nv_bfloat16*)row + j);
#pragma unroll
        for (int e = 0; e < 8; e++) r.x[e] = __bfloat162float(v.x[e]);
    }
    return r;
}

typedef struct {
    void* s;
    unsigned rows;
    unsigned cols;
    unsigned ld;
    unsigned heads;
    unsigned first;
    unsigned pad;
} PlowAttnSoftmax;
static_assert(sizeof(PlowAttnSoftmax) == 32, "attention softmax ABI");

template <bool F32>
__device__ __forceinline__ void asm_rows(const PlowAttnSoftmax& a) {
    constexpr int UN = PLOW_ATTN_SM_UN;
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nthr = gridDim.x * blockDim.x;
    const unsigned end8 = (a.cols + 7u) & ~7u;
    const size_t row_bytes = (size_t)a.ld * (F32 ? 4u : 2u);
    for (unsigned row = tid; row < a.rows; row += nthr) {
        const void* const s = (const char*)a.s + (size_t)row * row_bytes;
        __nv_bfloat16* const p = (__nv_bfloat16*)((char*)a.s + (size_t)row * row_bytes);
        unsigned lim = a.first + row / a.heads;
        if (lim > a.cols) lim = a.cols;
        const unsigned lim8 = lim & ~7u;
        const unsigned tail = lim - lim8;

        /* Pass A: blocked online softmax, UN vectors loaded before any is consumed. */
        float m = PLOW_ATTN_SM_NEG_INF, l = 0.0f;
        unsigned j = 0;
        for (; j + (unsigned)UN * 8u <= lim8; j += (unsigned)UN * 8u) {
            asm_f32v8 v[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) v[u] = asm_ld8<F32>(s, j + (unsigned)u * 8u);
            float bm = PLOW_ATTN_SM_NEG_INF;
#pragma unroll
            for (int e = 0; e < UN * 8; e++) bm = v[e >> 3].x[e & 7] > bm ? v[e >> 3].x[e & 7] : bm;
            const float nm = bm > m ? bm : m;
            float acc = l * asm_ex2(m - nm);
#pragma unroll
            for (int e = 0; e < UN * 8; e++) acc += asm_ex2(v[e >> 3].x[e & 7] - nm);
            l = acc;
            m = nm;
        }
        for (; j < lim; j += 8u) {
            asm_f32v8 v = asm_ld8<F32>(s, j);
            const unsigned live = lim - j;
            float bm = PLOW_ATTN_SM_NEG_INF;
#pragma unroll
            for (unsigned e = 0; e < 8u; e++) {
                v.x[e] = e < live ? v.x[e] : PLOW_ATTN_SM_NEG_INF;
                bm = v.x[e] > bm ? v.x[e] : bm;
            }
            const float nm = bm > m ? bm : m;
            float acc = l * asm_ex2(m - nm);
#pragma unroll
            for (unsigned e = 0; e < 8u; e++) acc += asm_ex2(v.x[e] - nm);
            l = acc;
            m = nm;
        }
        const float inv = 1.0f / l;

        /* Pass B: probabilities; the masked remainder of the tile row becomes 0. */
        j = 0;
        for (; j + (unsigned)UN * 8u <= lim8; j += (unsigned)UN * 8u) {
            asm_f32v8 v[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) v[u] = asm_ld8<F32>(s, j + (unsigned)u * 8u);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                asm_bf16v8 o;
#pragma unroll
                for (int e = 0; e < 8; e++)
                    o.x[e] = __float2bfloat16(asm_ex2(v[u].x[e] - m) * inv);
                asm_st8(p + j + (unsigned)u * 8u, o);
            }
        }
        for (; j < lim8; j += 8u) {
            const asm_f32v8 v = asm_ld8<F32>(s, j);
            asm_bf16v8 o;
#pragma unroll
            for (int e = 0; e < 8; e++) o.x[e] = __float2bfloat16(asm_ex2(v.x[e] - m) * inv);
            asm_st8(p + j, o);
        }
        if (tail) {
            const asm_f32v8 v = asm_ld8<F32>(s, lim8);
            asm_bf16v8 o;
#pragma unroll
            for (unsigned e = 0; e < 8u; e++)
                o.x[e] = e < tail ? __float2bfloat16(asm_ex2(v.x[e] - m) * inv)
                                  : __float2bfloat16(0.0f);
            asm_st8(p + lim8, o);
            j = lim8 + 8u;
        }
        asm_bf16v8 zero;
        *(uint4*)&zero = make_uint4(0u, 0u, 0u, 0u);
        for (; j < end8; j += 8u) asm_st8(p + j, zero);
    }
}

/* One score row per WARP: lane l owns columns [l*8 + 256*k, +8), so every warp-level load and
 * store touches 512 contiguous bytes. Pass A keeps a per-lane online (max, sum) that the warp
 * merges once; pass B writes P (and the zero tail) with the same striding. With f32 scores the
 * bf16 stores of an iteration land below that iteration's loads (bytes [2j, 2j+512) vs
 * [4j, 4j+1024)), and the syncwarp orders the lanes' loads before any lane's store. */
template <bool F32>
__device__ __forceinline__ void asm_rows_warp(const PlowAttnSoftmax& a) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned nwarps = (gridDim.x * blockDim.x) >> 5;
    const unsigned end8 = (a.cols + 7u) & ~7u;
    const size_t row_bytes = (size_t)a.ld * (F32 ? 4u : 2u);
    asm_bf16v8 zero;
    *(uint4*)&zero = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned row = warp; row < a.rows; row += nwarps) {
        const void* const s = (const char*)a.s + (size_t)row * row_bytes;
        __nv_bfloat16* const p = (__nv_bfloat16*)((char*)a.s + (size_t)row * row_bytes);
        unsigned lim = a.first + row / a.heads;
        if (lim > a.cols) lim = a.cols;

        float m = PLOW_ATTN_SM_NEG_INF, l = 0.0f;
        for (unsigned j = lane * 8u; j < lim; j += 256u) {
            asm_f32v8 v = asm_ld8<F32>(s, j);
            const unsigned live = lim - j;
            float bm = PLOW_ATTN_SM_NEG_INF;
#pragma unroll
            for (unsigned e = 0; e < 8u; e++) {
                v.x[e] = e < live ? v.x[e] : PLOW_ATTN_SM_NEG_INF;
                bm = v.x[e] > bm ? v.x[e] : bm;
            }
            const float nm = bm > m ? bm : m;
            float acc = l * asm_ex2(m - nm);
#pragma unroll
            for (unsigned e = 0; e < 8u; e++) acc += asm_ex2(v.x[e] - nm);
            l = acc;
            m = nm;
        }
        float M = m;
#pragma unroll
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
        l *= asm_ex2(m - M);
#pragma unroll
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        const float inv = 1.0f / l;

        /* Uniform trip count: every lane reaches the syncwarp. */
        for (unsigned j0 = 0; j0 < end8; j0 += 256u) {
            const unsigned j = j0 + lane * 8u;
            asm_bf16v8 o = zero;
            if (j < lim) {
                const asm_f32v8 v = asm_ld8<F32>(s, j);
                const unsigned live = lim - j;
#pragma unroll
                for (unsigned e = 0; e < 8u; e++)
                    o.x[e] = e < live ? __float2bfloat16(asm_ex2(v.x[e] - M) * inv)
                                      : __float2bfloat16(0.0f);
            }
            if (F32) __syncwarp();
            if (j < end8) asm_st8(p + j, o);
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void plow_attn_softmax(PlowAttnSoftmax a) {
    asm_rows<false>(a);
}

extern "C" __global__ __launch_bounds__(256)
void plow_attn_softmax_w(PlowAttnSoftmax a) {
    asm_rows_warp<false>(a);
}

extern "C" __global__ __launch_bounds__(256)
void plow_attn_softmax_w_f32(PlowAttnSoftmax a) {
    asm_rows_warp<true>(a);
}

extern "C" __global__ __launch_bounds__(256)
void plow_attn_softmax_f32(PlowAttnSoftmax a) {
    asm_rows<true>(a);
}
