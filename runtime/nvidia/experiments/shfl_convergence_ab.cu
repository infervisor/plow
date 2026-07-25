/* shfl_convergence_ab.cu — why does __shfl_xor_sync cost ~4.5 SASS instructions inside the
 * plow megakernel and 1 in a standalone kernel?
 *
 * FINDING BEING INVESTIGATED (measured on the shipped sm_90a DECODE cubin):
 *   op_attention.cuh:228 (the warp_sum32 body) accounts for 10,368 SASS instructions covering
 *   2,319 SHFLs = 6.85% of the megakernel. ptxas wraps most of them in
 *   WARPSYNC.COLLECTIVE / SHFL / ENDCOLLECTIVE plus MOVs that re-materialise the mask and the
 *   xor delta into registers, instead of the bare `SHFL.BFLY PT, Rd, Rs, 0x10, 0x1f`.
 *
 * ATTRIBUTION (nvdisasm -gi, FULL inline chain — see attrib.py method note): the cost is NOT in
 * the attention arms.  85% of it is gemv_rows (op_gemm.cuh:182) and its GLU/fp8 twins.  The
 * op_norm.cuh block_sum path (sm120_common.cuh:41) in the SAME cubin emits BARE shuffles.
 * So the trigger is per-call-site, not whole-kernel.
 *
 * This TU isolates the structural difference.  Every kernel below computes the same warp
 * reduction; only the surrounding control flow / spelling changes.  Build:
 *
 *   nvcc -arch=sm_90a -O3 -cubin -o /tmp/shfl_ab.cubin shfl_convergence_ab.cu
 *   nvdisasm -c /tmp/shfl_ab.cubin | awk '/^\t\t\.text\./{f=$0} /SHFL|WARPSYNC/{print f, $0}'
 *
 * and for the timing half:
 *   nvcc -arch=sm_90a -O3 -gencode arch=compute_90a,code=sm_90a -o /tmp/shfl_ab shfl_convergence_ab.cu
 *   LD_LIBRARY_PATH=/usr/local/cuda/compat flock /tmp/plow_gpu.lock /tmp/shfl_ab
 */
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unistd.h>

#define LANES 32
#define WARPS 8
#define THREADS (LANES * WARPS)

/* ---- the reducer under test, in three spellings ------------------------------------------- */

/* (a) exactly what op_attention.cuh:226 ships */
__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
    return v;
}
/* (b) fix candidate: __syncwarp() first, so ptxas has a convergence point to reason from */
__device__ __forceinline__ float warp_sum32_sw(float v) {
    __syncwarp();
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
    return v;
}
/* (c) fix candidate: raw PTX, bypassing the intrinsic's implicit convergence contract.
 * shfl.sync.bfly.b32 d, a, b, c, membermask;  b = xor lane delta, c = clamp/segmask 0x1f.
 * Numerically identical: same instruction, same operands, same order of the FADD tree. */
__device__ __forceinline__ float shfl_xor_asm(float v, int off) {
    float r;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(r)
                 : "f"(v), "r"(off));
    return r;
}
__device__ __forceinline__ float warp_sum32_asm(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += shfl_xor_asm(v, off);
    return v;
}

/* ---- the dot body, a stand-in for op_gemm.cuh's dot8 --------------------------------------- */
struct bf16v8 {
    __nv_bfloat16 x[8];
};
__device__ __forceinline__ bf16v8 ld8(const __nv_bfloat16* p) {
    bf16v8 r;
    *(uint4*)&r = *(const uint4*)p;
    return r;
}
__device__ __forceinline__ bf16v8 z8() {
    bf16v8 r;
    *(uint4*)&r = make_uint4(0u, 0u, 0u, 0u);
    return r;
}
__device__ __forceinline__ float dot8(const bf16v8& a, const bf16v8& b, float acc) {
#pragma unroll
    for (int i = 0; i < 8; i++) acc = fmaf(__bfloat162float(a.x[i]), __bfloat162float(b.x[i]), acc);
    return acc;
}

#define GV_STEP (LANES * 8)

/* ============================================================================================
 * V0  straight line: reduce once, uniform control flow, no enclosing loop.
 * ========================================================================================== */
__global__ void k00_plain(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                          unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    float acc = 0.f;
    for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(W + k), ld8(x + k), acc);
    const float t = warp_sum32(acc);
    if (lane == 0) C[blockIdx.x] = __float2bfloat16(t);
}

/* ============================================================================================
 * V1  reduce inside a data-dependent OUTER loop whose start differs per warp
 *     (`n = n0 + warp`), exactly gemv_rows' shape.  n1 is a runtime value.
 * ========================================================================================== */
__global__ void k01_warploop(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                             unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        const float t = warp_sum32(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* V1b same, with __syncwarp() in the reducer */
__global__ void k02_warploop_syncwarp(__nv_bfloat16* C, const __nv_bfloat16* x,
                                      const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        const float t = warp_sum32_sw(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* V1c same, raw PTX reducer */
__global__ void k03_warploop_asm(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                 unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        const float t = warp_sum32_asm(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* ============================================================================================
 * V2  the `continue` inside the unrolled inner loop (op_gemm.cuh:171) added on top of V1.
 * ========================================================================================== */
#define UN 4
__global__ void k04_inner_continue(__nv_bfloat16* C, const __nv_bfloat16* x,
                                   const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wrow + k) : z8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld8(x + kk[u]), acc);
            }
        }
        const float t = warp_sum32(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* ============================================================================================
 * V3  MM>1: an unrolled array of accumulators reduced back to back, plus the
 *     `m < M` runtime predicate on the store — op_gemm.cuh:180-184 verbatim.
 * ========================================================================================== */
template <int MM>
__device__ __forceinline__ void gemv_rows(__nv_bfloat16* __restrict__ C,
                                          const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ W, unsigned M,
                                          unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wrow + k) : z8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}
template <int MM>
__device__ __forceinline__ void gemv_rows_sw(__nv_bfloat16* __restrict__ C,
                                             const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ W, unsigned M,
                                             unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wrow + k) : z8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32_sw(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}
template <int MM>
__device__ __forceinline__ void gemv_rows_asm(__nv_bfloat16* __restrict__ C,
                                              const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ W, unsigned M,
                                              unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wrow + k) : z8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32_asm(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}

__global__ void k05_gemvrows1(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                              unsigned M, unsigned N, unsigned K) {
    gemv_rows<1>(C, x, W, M, N, K);
}
__global__ void k06_gemvrows8(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                              unsigned M, unsigned N, unsigned K) {
    gemv_rows<8>(C, x, W, M, N, K);
}
__global__ void k07_gemvrows8_sw(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                 unsigned M, unsigned N, unsigned K) {
    gemv_rows_sw<8>(C, x, W, M, N, K);
}
__global__ void k08_gemvrows8_asm(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                  unsigned M, unsigned N, unsigned K) {
    gemv_rows_asm<8>(C, x, W, M, N, K);
}

/* ============================================================================================
 * V4  the megakernel shape: the same body reached through a runtime `switch` over an opcode,
 *     inside a work-item loop.  Tests "is it the giant switch".
 * ========================================================================================== */
__global__ void k09_switch(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                           const int* prog, unsigned M, unsigned N, unsigned K, unsigned nop) {
    for (unsigned i = 0; i < nop; i++) {
        switch (prog[i]) {
        case 0:
            gemv_rows<1>(C, x, W, M, N, K);
            break;
        case 1:
            gemv_rows<8>(C, x, W, M, N, K);
            break;
        case 2: {
            const unsigned lane = threadIdx.x & 31u;
            float acc = 0.f;
            for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(W + k), ld8(x + k), acc);
            const float t = warp_sum32(acc);
            if (lane == 0) C[0] = __float2bfloat16(t);
            break;
        }
        default:
            break;
        }
        __syncthreads();
    }
}

/* ============================================================================================
 * V5  block_sum shape — the call site in the shipped cubin that emits BARE SHFLs.
 *     warp_sum32 is the FIRST thing in the callee and is followed by __syncthreads().
 * ========================================================================================== */
extern __shared__ float g_part[];
__device__ __forceinline__ float block_sum(float v, float* part) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    v = warp_sum32(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float r = part[0];
#pragma unroll
    for (unsigned w = 1; w < WARPS; w++) r += part[w];
    __syncthreads();
    return r;
}
__global__ void k10_blocksum(__nv_bfloat16* C, const __nv_bfloat16* x, unsigned K) {
    float ss = 0.f;
    for (unsigned k = threadIdx.x; k < K; k += THREADS) {
        const float v = __bfloat162float(x[k]);
        ss += v * v;
    }
    const float s = block_sum(ss, g_part);
    if (threadIdx.x == 0) C[blockIdx.x] = __float2bfloat16(s);
}

/* V5b block_sum inside the same warp-strided outer loop as V1 — isolates "enclosing loop" alone */
__global__ void k11_blocksum_loop(__nv_bfloat16* C, const __nv_bfloat16* x, unsigned N,
                                  unsigned K) {
    for (unsigned n = 0; n < N; n++) {
        float ss = 0.f;
        for (unsigned k = threadIdx.x; k < K; k += THREADS) {
            const float v = __bfloat162float(x[(size_t)n * K + k]);
            ss += v * v;
        }
        const float s = block_sum(ss, g_part);
        if (threadIdx.x == 0) C[n] = __float2bfloat16(s);
        __syncthreads();
    }
}

/* ============================================================================================
 * V6  isolate the two remaining suspects on top of the k01 shape:
 *     (a) the `if (lane == 0 && m < M)` DIVERGENT STORE that FOLLOWS the reduction
 *     (b) nothing else changed.
 * ========================================================================================== */
__global__ void k12_warploop_nostore(__nv_bfloat16* C, const __nv_bfloat16* x,
                                     const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        const float t = warp_sum32(acc);
        C[n * 32 + lane] = __float2bfloat16(t); /* all 32 lanes store: no divergence after */
    }
}

/* V6b: uniform outer loop (no `+ warp` skew), divergent store kept */
__global__ void k13_uniloop(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                            unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = 0; n < N; n++) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        const float t = warp_sum32(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* V6c: k01 but the inner k-loop bound made lane-INDEPENDENT (trip count uniform per lane).
 * `k = lane*8; k < K; k += 256` already has a lane-dependent START, so different lanes can
 * execute different trip counts when K is not a multiple of 256 -> the loop exit is DIVERGENT. */
__global__ void k14_warploop_uniformk(__nv_bfloat16* C, const __nv_bfloat16* x,
                                      const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = K / GV_STEP; /* uniform trip count, no lane in the bound */
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned c = 0; c < nchunk; c++) {
            const unsigned k = c * GV_STEP + lane * 8;
            acc = dot8(ld8(wrow + k), ld8(x + k), acc);
        }
        const float t = warp_sum32(acc);
        if (lane == 0) C[n] = __float2bfloat16(t);
    }
}

/* ============================================================================================
 * host harness — correctness + timing for the three reducer spellings on the gemv_rows shape
 * ========================================================================================== */
#ifndef SHFL_AB_NO_MAIN
#define CK(e)                                                                                      \
    do {                                                                                           \
        cudaError_t _e = (e);                                                                      \
        if (_e != cudaSuccess) {                                                                   \
            printf("CUDA %s @%d\n", cudaGetErrorString(_e), __LINE__);                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static float sm_clock_mhz() {
    int c = 0;
    cudaDeviceGetAttribute(&c, cudaDevAttrClockRate, 0);
    return c / 1000.0f;
}

__global__ void k23_gemvrows8_sw1(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
                                  unsigned, unsigned, unsigned); /* fwd: defined in round 2 */

int main() {
    const unsigned M = 1, N = 4096, K = 3840; /* gemma4-12b o_proj-ish */
    const int BLK = 132;
    size_t nW = (size_t)N * K, nX = (size_t)8 * K, nC = (size_t)8 * N;
    __nv_bfloat16 *dW, *dX, *dC;
    CK(cudaMalloc(&dW, nW * 2));
    CK(cudaMalloc(&dX, nX * 2));
    CK(cudaMalloc(&dC, nC * 2 * 4));
    __nv_bfloat16* hW = (__nv_bfloat16*)malloc(nW * 2);
    __nv_bfloat16* hX = (__nv_bfloat16*)malloc(nX * 2);
    srand(1);
    for (size_t i = 0; i < nW; i++) hW[i] = __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
    for (size_t i = 0; i < nX; i++) hX[i] = __float2bfloat16((rand() / (float)RAND_MAX - 0.5f));
    CK(cudaMemcpy(dW, hW, nW * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, hX, nX * 2, cudaMemcpyHostToDevice));

    __nv_bfloat16* ref = (__nv_bfloat16*)malloc(nC * 2);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    const double bytes = (double)nW * 2.0;

    /* -------- GROUP 1: the DECODE shape (MM=1, warp-strided outer loop = the shipped d_gemv).
     *  k01 scaffolded (intrinsic) vs k02 bare (syncwarp-in-reducer) vs k03 bare (raw PTX). ----- */
    typedef void (*KNK)(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, unsigned,
                        unsigned);
    struct V1 {
        const char* name;
        KNK k;
    } g1[] = {
        {"k01 decode MM=1 intrinsic (SCAFFOLDED)", k01_warploop},
        {"k02 decode MM=1 syncwarp  (bare)      ", k02_warploop_syncwarp},
        {"k03 decode MM=1 raw-ptx   (scaffolded)", k03_warploop_asm},
    };
    /* -------- GROUP 2: the batched shape (MM=8 back-to-back reductions). ----------------------- */
    typedef void (*KMK)(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, unsigned,
                        unsigned, unsigned);
    struct V2 {
        const char* name;
        KMK k;
    } g2[] = {
        {"k06 MM=8 intrinsic (SCAFFOLDED)       ", k06_gemvrows8},
        {"k07 MM=8 syncwarp-in-reducer          ", k07_gemvrows8_sw},
        {"k08 MM=8 raw-ptx                       ", k08_gemvrows8_asm},
        {"k23 MM=8 one-syncwarp-before-fan (bare)", k23_gemvrows8_sw1},
    };

    /* correctness: group 1 vs k01, group 2 vs k06 -- must be BIT identical */
    printf("== correctness (bf16 bit-exact) ==\n");
    for (int v = 0; v < 3; v++) {
        CK(cudaMemset(dC, 0, nC * 2));
        g1[v].k<<<BLK, THREADS>>>(dC, dX, dW, N, K);
        CK(cudaDeviceSynchronize());
        __nv_bfloat16* h = (__nv_bfloat16*)malloc(nC * 2);
        CK(cudaMemcpy(h, dC, nC * 2, cudaMemcpyDeviceToHost));
        if (v == 0) memcpy(ref, h, nC * 2);
        size_t bad = 0;
        for (size_t i = 0; i < N; i++)
            if (*(unsigned short*)&h[i] != *(unsigned short*)&ref[i]) bad++;
        printf("  %s : %s (%zu/%u)\n", g1[v].name, bad ? "MISMATCH" : "bit-identical", bad, N);
        free(h);
    }
    for (int v = 0; v < 4; v++) {
        CK(cudaMemset(dC, 0, nC * 2));
        g2[v].k<<<BLK, THREADS>>>(dC, dX, dW, M, N, K);
        CK(cudaDeviceSynchronize());
        __nv_bfloat16* h = (__nv_bfloat16*)malloc(nC * 2);
        CK(cudaMemcpy(h, dC, nC * 2, cudaMemcpyDeviceToHost));
        if (v == 0) memcpy(ref, h, nC * 2);
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)M * N; i++)
            if (*(unsigned short*)&h[i] != *(unsigned short*)&ref[i]) bad++;
        printf("  %s : %s (%zu/%u)\n", g2[v].name, bad ? "MISMATCH" : "bit-identical", bad, M * N);
        free(h);
    }

    /* timing: min-of-12, rotated round-robin, 25 ms gaps, report SM clock */
    printf("\n== timing  N=%u K=%u blocks=%d  SMclock=%.0f MHz  (weights=%.1f MB) ==\n", N, K, BLK,
           sm_clock_mhz(), bytes / 1e6);
    float b1[3];
    for (int v = 0; v < 3; v++) b1[v] = 1e30f;
    for (int rep = 0; rep < 12; rep++)
        for (int vv = 0; vv < 3; vv++) {
            int v = (vv + rep) % 3;
            g1[v].k<<<BLK, THREADS>>>(dC, dX, dW, N, K);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int it = 0; it < 20; it++) g1[v].k<<<BLK, THREADS>>>(dC, dX, dW, N, K);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms;
            CK(cudaEventElapsedTime(&ms, a, b));
            if (ms / 20.f < b1[v]) b1[v] = ms / 20.f;
            usleep(25000);
        }
    for (int v = 0; v < 3; v++)
        printf("  %s %8.3f us  %6.0f GB/s  (%+.2f%% vs k01)\n", g1[v].name, b1[v] * 1e3,
               bytes / (b1[v] * 1e-3) / 1e9, 100.0 * (b1[v] / b1[0] - 1.0));

    float b2[4];
    for (int v = 0; v < 4; v++) b2[v] = 1e30f;
    for (int rep = 0; rep < 12; rep++)
        for (int vv = 0; vv < 4; vv++) {
            int v = (vv + rep) % 4;
            g2[v].k<<<BLK, THREADS>>>(dC, dX, dW, M, N, K);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int it = 0; it < 20; it++) g2[v].k<<<BLK, THREADS>>>(dC, dX, dW, M, N, K);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms;
            CK(cudaEventElapsedTime(&ms, a, b));
            if (ms / 20.f < b2[v]) b2[v] = ms / 20.f;
            usleep(25000);
        }
    for (int v = 0; v < 4; v++)
        printf("  %s %8.3f us  %6.0f GB/s  (%+.2f%% vs k06)\n", g2[v].name, b2[v] * 1e3,
               bytes / (b2[v] * 1e-3) / 1e9, 100.0 * (b2[v] / b2[0] - 1.0));
    return 0;
}
#endif

/* ============================================================================================
 * ROUND 2 — isolate the trigger found in round 1.
 *
 * Round 1 result: k13_uniloop (`for n=0; n<N; n++`)  -> 5 SHFL, all immediate, 0 WARPSYNC.
 *                 k01_warploop (`for n=warp; n<N; n+=8`) -> 10 SHFL, 5 WARPSYNC, 5 ENDCOLLECTIVE.
 * The ONLY difference is that the loop bound/induction depends on `threadIdx.x >> 5`.
 * These kernels separate "derived from threadIdx" from "strided" from "runtime-uniform".
 * ========================================================================================== */
#define R2_BODY                                                                                    \
    const __nv_bfloat16* wrow = W + (size_t)n * K;                                                 \
    float acc = 0.f;                                                                               \
    for (unsigned k = lane * 8; k < K; k += GV_STEP) acc = dot8(ld8(wrow + k), ld8(x + k), acc);   \
    const float t = warp_sum32(acc);                                                               \
    if (lane == 0) C[n] = __float2bfloat16(t);

/* strided by WARPS but START is 0: threadIdx-independent trip count */
__global__ void k15_stride_nowarp(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                  unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = 0; n < N; n += WARPS) {
        R2_BODY
    }
}
/* start is a RUNTIME value from memory (uniform, but unknown to the compiler) */
__global__ void k16_runtime_start(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                  const unsigned* s, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = s[0]; n < N; n += WARPS) {
        R2_BODY
    }
}
/* start derived from threadIdx.x >> 5 -- the shipped shape (== k01, restated for the table) */
__global__ void k17_warp_start(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                               unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = threadIdx.x >> 5; n < N; n += WARPS) {
        R2_BODY
    }
}
/* start derived from blockIdx (uniform) */
__global__ void k18_block_start(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = blockIdx.x; n < N; n += WARPS) {
        R2_BODY
    }
}
/* start derived from threadIdx.x >> 5, laundered through a full-warp broadcast shuffle */
__global__ void k19_warp_bcast(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                               unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0, 32);
    for (unsigned n = warp; n < N; n += WARPS) {
        R2_BODY
    }
}
/* warp start + __syncwarp() at the TOP of the loop body (rather than inside the reducer) */
__global__ void k20_warp_syncwarp_top(__nv_bfloat16* C, const __nv_bfloat16* x,
                                      const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned n = warp; n < N; n += WARPS) {
        __syncwarp();
        R2_BODY
    }
}
/* warp start + a single __syncwarp() BEFORE the loop (does it survive the backedge?) */
__global__ void k21_warp_syncwarp_pre(__nv_bfloat16* C, const __nv_bfloat16* x,
                                      const __nv_bfloat16* W, unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    __syncwarp();
    for (unsigned n = warp; n < N; n += WARPS) {
        R2_BODY
    }
}
/* warp start, lane-varying start too (true divergence) — the pessimistic reference */
__global__ void k22_lane_start(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                               unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned n = lane; n < N; n += 32) {
        R2_BODY
    }
}
/* gemv_rows<8> with __syncwarp() once before the whole MM reduction fan, not per reduction */
template <int MM>
__device__ __forceinline__ void gemv_rows_sw1(__nv_bfloat16* __restrict__ C,
                                              const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ W, unsigned M,
                                              unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned n = warp; n < N; n += WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wrow + k) : z8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
        __syncwarp(); /* ONE convergence point for the whole fan of MM reductions */
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}
__global__ void k23_gemvrows8_sw1(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                  unsigned M, unsigned N, unsigned K) {
    gemv_rows_sw1<8>(C, x, W, M, N, K);
}
__global__ void k24_gemvrows1_sw1(__nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                                  unsigned M, unsigned N, unsigned K) {
    gemv_rows_sw1<1>(C, x, W, M, N, K);
}
