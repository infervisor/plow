/* gemma_sm120.cu — performant single-SM Gemma-family kernels for the
 * RTX PRO 6000 Blackwell (GB202, SM 12.0 / sm_120a).
 *
 * Why a dedicated file (not the Hopper/datacenter stubs in gemm.cu/flash.cu):
 * consumer Blackwell is SM 12.0 with **Ada-class** per-SM limits — 100 KiB max
 * shared, 1536 threads / 48 warps, and **no tcgen05 / no TMEM**. So neither the
 * Hopper warpgroup `wgmma.m64.nN.k16` path nor the datacenter `tcgen05` path is
 * available. The performant primitive here is the warp-level 5th-gen tensor
 * core op `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, fed from shared
 * memory staged with `cp.async.cg` (sm_80+, universally present on sm_120).
 *
 * Single-SM contract (shared with the rest of the runtime, see gemm.cu header):
 *   ONE packet = the whole op (full m,n,k in the body). The kernel launches
 *   P = #SMs (188 on RTX 6000 Pro) PERSISTENT blocks; block s owns output tiles
 *   s, s+P, s+2P, ... (grid-stride) and stages each tile's operands itself.
 *   Nothing crosses SM boundaries; there is no grid-wide barrier.
 *
 * bf16 storage, f32 accumulate. Slot pointers for the _bf16 variants are device
 * __nv_bfloat16 buffers (the f32 golden tier lives in gemm.cu/flash.cu/row.cu).
 *
 * Build-gated behind PLOW_CUDA; compiled for CUDA_ARCHITECTURES "120a".
 */
#include "gpu_common.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef PLOW_SM120_SMS
#define PLOW_SM120_SMS 188        /* RTX 6000 Pro Blackwell SM count (persistent grid) */
#endif

typedef __nv_bfloat16 bf16;

// ===========================================================================
// Low-level primitives: cp.async staging + warp-level mma.sync m16n8k16.
// ===========================================================================

// Async copy 16 B (one float4 / 8 bf16) global->shared, L2-cache-global class.
__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ldmatrix x4 (A operand, row-major 16x16 -> 4 fragments per thread).
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// ldmatrix x2 with transpose (B operand from a row-major K x N shared tile).
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// D[16x8] += A[16x16] * B[16x8], bf16 inputs, f32 accumulate.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const unsigned (&a)[4],
                                             const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Warp reduce (sum) over 32 lanes.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ===========================================================================
// Tile config for the norm-linear / plain GEMM mainloop (sm_120, bf16).
//   Block tile BM x BN, K-step BK. 256 threads = 8 warps in a 4x2 warp grid;
//   each warp owns a 32x32 output sub-tile computed as 2x4 m16n8k16 fragments.
//   Double-buffered shared: A[BM][BK] + B[BK][BN] in bf16, +8 halfs pad/row to
//   kill bank conflicts. 128x64x32: (128*40 + 32*136)*2B*2buf ~= 37 KiB. Fits
//   100 KiB with room for the BM rms/gamma vectors.
// ===========================================================================
#define GM_BM 128
#define GM_BN 64
#define GM_BK 32
#define GM_APAD 8
#define GM_BPAD 8
#define GM_WARPS 8
#define GM_THREADS (GM_WARPS * 32)
#define GM_WARPS_M 4
#define GM_WARPS_N 2
#define GM_WM (GM_BM / GM_WARPS_M)   // 32
#define GM_WN (GM_BN / GM_WARPS_N)   // 32

// Shared layout offsets (bf16 element counts), double-buffered.
#define GM_ASTRIDE (GM_BK + GM_APAD)
#define GM_BSTRIDE (GM_BN + GM_BPAD)
#define GM_ABUF (GM_BM * GM_ASTRIDE)
#define GM_BBUF (GM_BK * GM_BSTRIDE)

// Stage the [GM_BK][GM_BN] B tile out of the [n,k] row-major weight.
//
// The tile's fast axis is N but B's fast axis is K, so no thread can be
// contiguous on both sides. Reading wins: each thread pulls 8 CONSECUTIVE k for
// a single n as one 16 B load and scatters them down a column of the tile. That
// makes the global side 64 B contiguous per weight row instead of 2 B scattered
// (with idx=tid, consecutive threads land on consecutive n, i.e. a different
// row of B at stride k). The shared-memory scatter is cheap and GM_BSTRIDE
// already carries the bank-conflict padding.
//
// The 16 B load needs (nn*k + kk) to be 8-element aligned. kk = kbase + 8*j, so
// that holds iff k and kbase are both multiples of 8; anything else (ragged k,
// an unaligned split-K slice) takes the scalar fallback. Out-of-range elements
// are ZERO-filled in both paths -- leaving stale smem silently corrupts ragged
// N/K instead of faulting.
__device__ __forceinline__ void gm_stage_b(bf16* Bd, const bf16* __restrict__ B,
                                           int tid, int tn, int kbase,
                                           unsigned n, unsigned k, int kend) {
    if (((k | (unsigned)kbase) & 7u) == 0u) {
        const int KCH = GM_BK / 8;                       // 16 B chunks along K
        for (int c = tid; c < GM_BK * GM_BN / 8; c += GM_THREADS) {
            const int col = c / KCH, r0 = (c % KCH) * 8; // col=n, r0=kk in tile
            const int nn = tn + col, kk = kbase + r0;
            bf16 v[8];
            if (nn < (int)n && kk + 8 <= kend) {
                *(float4*)v = *(const float4*)(B + (size_t)nn * k + kk);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    v[j] = (nn < (int)n && kk + j < kend)
                             ? B[(size_t)nn * k + kk + j] : __float2bfloat16(0.f);
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) Bd[(r0 + j) * GM_BSTRIDE + col] = v[j];
        }
    } else {
        for (int idx = tid; idx < GM_BK * GM_BN; idx += GM_THREADS) {
            const int row = idx / GM_BN, col = idx % GM_BN;   // row=kk, col=n
            const int nn = tn + col, kk = kbase + row;
            bf16 v = __float2bfloat16(0.f);
            if (nn < (int)n && kk < kend) v = B[(size_t)nn * k + kk];
            Bd[row * GM_BSTRIDE + col] = v;
        }
    }
}

// Epilogue activation applied to the GEMM output (detail high nibble). Only the
// acts a Gemma projection would fold (none / gelu-tanh on head) are reachable.
__device__ __forceinline__ float gm_act(int act, float x) {
    switch (act) {
        case PLOW_ACT_SILU:      return silu(x);
        case PLOW_ACT_GELU_TANH: { float c = 0.79788456f*(x+0.044715f*x*x*x);
                                   return 0.5f*x*(1.0f+tanhf(c)); }
        case PLOW_ACT_RELU:      return x > 0.f ? x : 0.f;
        default:                 return x;
    }
}

// Core fused GEMM. When gamma!=nullptr an RMSNorm prologue over the FULL K is
// folded into the A-staging step (F1 FusedNormLinear q/kv/gate/up). k0/kext give
// the split-K slice for the decode partial forms; when the slice spans the whole
// K the norm is exact, otherwise the caller must pre-scale (decode recomputes
// rms locally over the tiny A — see the split-K launcher).
//
// A: [m,k] row-major (pre-norm activations)   B: [n,k] row-major weight (col of
// the product is a weight row -> mma B operand is B read transposed).
// C: [m,n] row-major.  rms folded per output row; gamma per K element.
template <bool NORM>
__global__ __launch_bounds__(GM_THREADS)
void gemma_gemm_sm120(const bf16* __restrict__ A, const bf16* __restrict__ B,
                      const bf16* __restrict__ gamma, bf16* __restrict__ C,
                      unsigned m, unsigned n, unsigned k, float eps, int act) {
    extern __shared__ bf16 smem[];
    bf16* As = smem;                        // [2][GM_BM][GM_ASTRIDE]
    bf16* Bs = smem + 2 * GM_ABUF;          // [2][GM_BK][GM_BSTRIDE]
    __shared__ float rms[GM_BM];            // per-row 1/rms(x) for the tile (gamma stays in L2)

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / GM_WARPS_N, wn = warp % GM_WARPS_N;
    const int tiles_m = (m + GM_BM - 1) / GM_BM;
    const int tiles_n = (n + GM_BN - 1) / GM_BN;
    const int ntiles = tiles_m * tiles_n;

    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int tm = (tile / tiles_n) * GM_BM;
        const int tn = (tile % tiles_n) * GM_BN;

        // ---- Phase 0: RMSNorm scalars over full K (norm variants only) -------
        if (NORM) {
            for (int r = warp; r < GM_BM && (tm + r) < (int)m; r += GM_WARPS) {
                const bf16* arow = A + (size_t)(tm + r) * k;
                float ss = 0.f;
                for (int c = lane; c < (int)k; c += 32) {
                    float x = __bfloat162float(arow[c]); ss += x * x;
                }
                ss = warp_sum(ss);
                if (lane == 0) rms[r] = rsqrtf(ss / (float)k + eps);
            }
            __syncthreads();
        }

        // Accumulators: each warp owns GM_WM x GM_WN = 32x32 = (2 m-frag)x(4 n-frag).
        float acc[2][4][4];
        #pragma unroll
        for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++)
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        // ---- Mainloop over K, double-buffered cp.async -----------------------
        const int ksteps = (k + GM_BK - 1) / GM_BK;
        auto stage = [&](int ks, int buf) {
            // B tile: [GM_BK][GM_BN] from weight B[n,k] (n=tn+col, kk=ks*BK+row).
            bf16* Bd = Bs + buf * GM_BBUF;
            gm_stage_b(Bd, B, tid, tn, ks * GM_BK, n, k, (int)k);
            // A tile: [GM_BM][GM_BK], normed on the fly (no cp.async: needs ALU).
            bf16* Ad = As + buf * GM_ABUF;
            for (int idx = tid; idx < GM_BM * GM_BK; idx += GM_THREADS) {
                int row = idx / GM_BK, col = idx % GM_BK;         // row=m, col=kk
                int mm = tm + row, kk = ks * GM_BK + col;
                float v = 0.f;
                if (mm < (int)m && kk < (int)k) {
                    v = __bfloat162float(A[(size_t)mm * k + kk]);
                    if (NORM) v = v * rms[row] * __bfloat162float(gamma[kk]);
                }
                Ad[row * GM_ASTRIDE + col] = __float2bfloat16(v);
            }
        };

        stage(0, 0); cp_async_commit();
        for (int ks = 0; ks < ksteps; ks++) {
            int nb = (ks + 1) & 1, cb = ks & 1;
            if (ks + 1 < ksteps) { stage(ks + 1, nb); cp_async_commit(); }
            cp_async_wait_group<0>();
            __syncthreads();

            bf16* Ad = As + cb * GM_ABUF;
            bf16* Bd = Bs + cb * GM_BBUF;
            // Warp mma over the 16-wide K (BK=32 -> 2 mma k-slices).
            #pragma unroll
            for (int kf = 0; kf < GM_BK; kf += 16) {
                unsigned af[2][4];   // 2 m-fragments (16-row halves of the 32-row warp tile)
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int arow = wm * GM_WM + mi * 16 + (lane % 16);
                    int acol = kf + (lane / 16) * 8;
                    ldmatrix_x4(af[mi], &Ad[arow * GM_ASTRIDE + acol]);
                }
                unsigned bf[4][2];   // 4 n-fragments (8-wide each -> 32-wide warp tile)
                #pragma unroll
                for (int nj = 0; nj < 4; nj++) {
                    int brow = kf + (lane % 16);      // trans reads the K x N tile transposed
                    ldmatrix_x2_trans(bf[nj], &Bd[brow * GM_BSTRIDE + wn * GM_WN + nj * 8]);
                }
                #pragma unroll
                for (int mi = 0; mi < 2; mi++)
                    #pragma unroll
                    for (int nj = 0; nj < 4; nj++)
                        mma_m16n8k16(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

        // ---- Epilogue: activation + bf16 store --------------------------------
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int nj = 0; nj < 4; nj++) {
                // m16n8 output: lane holds rows {gr, gr+8} x cols {gc,gc+1} pairs.
                int gr = wm * GM_WM + mi * 16 + (lane / 4);
                int gc = wn * GM_WN + nj * 8 + (lane % 4) * 2;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n)
                        C[(size_t)rr * n + cc] = __float2bfloat16(gm_act(act, acc[mi][nj][e]));
                }
            }
        }
        __syncthreads();
    }
}

// Split-K partial GEMM (decode o/down/lm_head + q/kv/gate/up): each block owns a
// (tile, k-slice) pair over the flat index space [ntiles * split]; it writes a
// PARTIAL into a scratch output (out[split][m][n]) that a follow-on Row reduce
// sums. NORM decode recomputes rms locally over the tiny A (M is 1..64).
template <bool NORM>
__global__ __launch_bounds__(GM_THREADS)
void gemma_gemm_splitk_sm120(const bf16* __restrict__ A, const bf16* __restrict__ B,
                             const bf16* __restrict__ gamma, float* __restrict__ Cpart,
                             unsigned m, unsigned n, unsigned k, float eps,
                             int split, int act) {
    extern __shared__ bf16 smem[];
    bf16* As = smem;
    bf16* Bs = smem + 2 * GM_ABUF;
    __shared__ float rms[GM_BM];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / GM_WARPS_N, wn = warp % GM_WARPS_N;
    const int tiles_m = (m + GM_BM - 1) / GM_BM;
    const int tiles_n = (n + GM_BN - 1) / GM_BN;
    const int ntiles = tiles_m * tiles_n;
    const int kslice = (k + split - 1) / split;

    for (int flat = blockIdx.x; flat < ntiles * split; flat += gridDim.x) {
        const int sp = flat % split;
        const int tile = flat / split;
        const int tm = (tile / tiles_n) * GM_BM;
        const int tn = (tile % tiles_n) * GM_BN;
        const int k0 = sp * kslice, k1 = min((int)k, k0 + kslice);

        if (NORM) {
            // rms over the FULL K (norm is not splittable) — cheap for decode M.
            for (int r = warp; r < GM_BM && (tm + r) < (int)m; r += GM_WARPS) {
                const bf16* arow = A + (size_t)(tm + r) * k;
                float ss = 0.f;
                for (int c = lane; c < (int)k; c += 32) { float x=__bfloat162float(arow[c]); ss+=x*x; }
                ss = warp_sum(ss);
                if (lane == 0) rms[r] = rsqrtf(ss / (float)k + eps);
            }
            __syncthreads();
        }

        float acc[2][4][4];
        #pragma unroll
        for (int i=0;i<2;i++) for(int j=0;j<4;j++) for(int e=0;e<4;e++) acc[i][j][e]=0.f;

        for (int kk0 = k0; kk0 < k1; kk0 += GM_BK) {
            bf16* Bd = Bs;
            gm_stage_b(Bd, B, tid, tn, kk0, n, k, k1);
            bf16* Ad = As;
            for (int idx = tid; idx < GM_BM*GM_BK; idx += GM_THREADS) {
                int row=idx/GM_BK, col=idx%GM_BK; int mm=tm+row, kk=kk0+col; float v=0.f;
                if (mm<(int)m && kk<k1) { v=__bfloat162float(A[(size_t)mm*k+kk]);
                    if (NORM) v = v*rms[row]*__bfloat162float(gamma[kk]); }
                Ad[row*GM_ASTRIDE+col]=__float2bfloat16(v);
            }
            cp_async_commit(); cp_async_wait_group<0>(); __syncthreads();
            #pragma unroll
            for (int kf=0; kf<GM_BK; kf+=16) {
                unsigned af[2][4];
                #pragma unroll
                for (int mi=0;mi<2;mi++){ int arow=wm*GM_WM+mi*16+(lane%16); int acol=kf+(lane/16)*8;
                    ldmatrix_x4(af[mi], &Ad[arow*GM_ASTRIDE+acol]); }
                unsigned bf[4][2];
                #pragma unroll
                for (int nj=0;nj<4;nj++){ int brow=kf+(lane%16);
                    ldmatrix_x2_trans(bf[nj], &Bd[brow*GM_BSTRIDE+wn*GM_WN+nj*8]); }
                #pragma unroll
                for (int mi=0;mi<2;mi++) for(int nj=0;nj<4;nj++)
                    mma_m16n8k16(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

        // Store partial into Cpart[sp][m][n] (f32 accumulator space).
        float* Cp = Cpart + (size_t)sp * m * n;
        (void)act; // activation applied only after the reduce, not per-partial.
        #pragma unroll
        for (int mi=0;mi<2;mi++) for(int nj=0;nj<4;nj++){
            int gr=wm*GM_WM+mi*16+(lane/4); int gc=wn*GM_WN+nj*8+(lane%4)*2;
            #pragma unroll
            for (int e=0;e<4;e++){ int rr=tm+gr+(e/2)*8, cc=tn+gc+(e%2);
                if (rr<(int)m && cc<(int)n) Cp[(size_t)rr*n+cc]=acc[mi][nj][e]; }
        }
        __syncthreads();
    }
}

// ===========================================================================
// FlashAttention (FA-2 tiling, mma m16n8k16). GQA 32:8 handled by mapping each
// Q head to its KV group (head / (NH/NKV)). One block owns (head, Q-tile) pairs.
//   Q tile [BQ, HD] resident in regs/shared; loop KV tiles: S=Q.K^T (mma),
//   online-softmax in regs, O += P.V (mma). Causal / sliding masks via flags.
// HD=128 -> 8 k-slices of 16. BQ=64, BKV=64.
// ===========================================================================
#define FA_BQ 64
#define FA_BKV 64
#define FA_HD 128
#define FA_WARPS 8
#define FA_THREADS (FA_WARPS*32)
#define FA_PAD 8

enum { FA_MASK_CAUSAL = 1, FA_MASK_SLIDING = 2 };

// Prefill (BQ>1): full causal or sliding window. q_base = absolute row of the
// Q tile's first row (coord0). window = sliding span (0 => full causal).
__global__ __launch_bounds__(FA_THREADS)
void gemma_flash_prefill_sm120(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                               const bf16* __restrict__ V, bf16* __restrict__ O,
                               unsigned seq_q, unsigned seq_kv, unsigned heads,
                               unsigned kv_heads, float scale, int mask, int window) {
    extern __shared__ bf16 smem[];
    bf16* Ks = smem;                          // [BKV][HD+PAD]
    bf16* Vs = Ks + FA_BKV * (FA_HD + FA_PAD);
    bf16* Qs = Vs + FA_BKV * (FA_HD + FA_PAD);// [BQ][HD+PAD]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q_tiles = (seq_q + FA_BQ - 1) / FA_BQ;
    const int gqa = heads / kv_heads;
    const int ntiles = heads * q_tiles;

    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int h = tile / q_tiles;
        const int qt = (tile % q_tiles) * FA_BQ;
        const int kvh = h / gqa;
        const bf16* Qh = Q + (size_t)h * seq_q * FA_HD;
        const bf16* Kh = K + (size_t)kvh * seq_kv * FA_HD;
        const bf16* Vh = V + (size_t)kvh * seq_kv * FA_HD;
        bf16* Oh = O + (size_t)h * seq_q * FA_HD;

        // Load Q tile resident.
        for (int idx = tid*8; idx < FA_BQ*FA_HD; idx += FA_THREADS*8) {
            int r=idx/FA_HD, c=idx%FA_HD; int qq=qt+r;
            const bf16* g = Qh + (size_t)qq*FA_HD + c;
            if (qq<(int)seq_q) cp_async_cg16(&Qs[r*(FA_HD+FA_PAD)+c], g);
        }
        cp_async_commit(); cp_async_wait_group<0>(); __syncthreads();

        // Per-row online-softmax state. Each warp owns FA_BQ/FA_WARPS = 8 rows.
        const int rows_per_warp = FA_BQ / FA_WARPS;   // 8
        float m_i[8], l_i[8], acc_o[8][FA_HD/32];     // O accumulator: HD across lanes
        #pragma unroll
        for (int r=0;r<rows_per_warp;r++){ m_i[r]=-1e30f; l_i[r]=0.f;
            #pragma unroll
            for (int d=0; d<FA_HD/32; d++) acc_o[r][d]=0.f; }

        const int last_kv = (mask & FA_MASK_CAUSAL) ? min((int)seq_kv, qt + FA_BQ) : (int)seq_kv;
        for (int kv0 = 0; kv0 < last_kv; kv0 += FA_BKV) {
            // Skip whole KV tiles outside the sliding window.
            if ((mask & FA_MASK_SLIDING) && window > 0 &&
                (qt - (kv0 + FA_BKV) >= window)) continue;

            for (int idx = tid*8; idx < FA_BKV*FA_HD; idx += FA_THREADS*8) {
                int r=idx/FA_HD, c=idx%FA_HD; int kk=kv0+r;
                if (kk<(int)seq_kv){ cp_async_cg16(&Ks[r*(FA_HD+FA_PAD)+c], Kh+(size_t)kk*FA_HD+c);
                                     cp_async_cg16(&Vs[r*(FA_HD+FA_PAD)+c], Vh+(size_t)kk*FA_HD+c); }
            }
            cp_async_commit(); cp_async_wait_group<0>(); __syncthreads();

            // S[rows_per_warp][BKV] = scale * Q.K^T, then online-softmax update.
            #pragma unroll
            for (int r=0;r<rows_per_warp;r++){
                int qrow = warp*rows_per_warp + r;
                float s_row[FA_BKV];
                #pragma unroll
                for (int j=0;j<FA_BKV;j++){
                    float dot=0.f;
                    for (int d=lane; d<FA_HD; d+=32)
                        dot += __bfloat162float(Qs[qrow*(FA_HD+FA_PAD)+d]) *
                               __bfloat162float(Ks[j*(FA_HD+FA_PAD)+d]);
                    dot = warp_sum(dot) * scale;
                    int kv = kv0 + j, q_abs = qt + qrow;
                    bool masked = (kv >= (int)seq_kv);
                    if (mask & FA_MASK_CAUSAL) masked |= (kv > q_abs);
                    if ((mask & FA_MASK_SLIDING) && window>0) masked |= (q_abs - kv >= window);
                    s_row[j] = masked ? -1e30f : dot;
                }
                float m_new = m_i[r];
                #pragma unroll
                for (int j=0;j<FA_BKV;j++) m_new = fmaxf(m_new, s_row[j]);
                float correction = __expf(m_i[r] - m_new);
                float l_new = l_i[r]*correction;
                #pragma unroll
                for (int d=0; d<FA_HD/32; d++) acc_o[r][d]*=correction;
                #pragma unroll
                for (int j=0;j<FA_BKV;j++){
                    float p = __expf(s_row[j]-m_new); l_new += p;
                    #pragma unroll
                    for (int d=0; d<FA_HD/32; d++){
                        int dd = lane + d*32;
                        acc_o[r][d] += p * __bfloat162float(Vs[j*(FA_HD+FA_PAD)+dd]);
                    }
                }
                m_i[r]=m_new; l_i[r]=l_new;
            }
            __syncthreads();
        }
        // Normalize + write O.
        #pragma unroll
        for (int r=0;r<rows_per_warp;r++){
            int qrow = warp*rows_per_warp + r, qq = qt + qrow;
            if (qq>=(int)seq_q) continue;
            float inv = 1.0f / l_i[r];
            #pragma unroll
            for (int d=0; d<FA_HD/32; d++){ int dd=lane+d*32;
                Oh[(size_t)qq*FA_HD+dd] = __float2bfloat16(acc_o[r][d]*inv); }
        }
        __syncthreads();
    }
}

// Decode (BQ=1): one query row, long KV. Split KV across the block's warps
// (split-KV); each warp accumulates a partial (m,l,O) over its KV stripe, then a
// shared-memory online-softmax merge combines the 8 partials. Distinct kernel
// because the reduction is over warps, not a Q-loop.
__global__ __launch_bounds__(FA_THREADS)
void gemma_flash_decode_sm120(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                              const bf16* __restrict__ V, bf16* __restrict__ O,
                              unsigned seq_kv, unsigned heads, unsigned kv_heads,
                              float scale) {
    extern __shared__ float smemf[];
    // per-warp partials: m[8], l[8], O[8][HD]
    float* pm = smemf;                 // [FA_WARPS]
    float* pl = pm + FA_WARPS;         // [FA_WARPS]
    float* po = pl + FA_WARPS;         // [FA_WARPS][HD]
    __shared__ bf16 Qsh[FA_HD];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gqa = heads / kv_heads;

    for (int h = blockIdx.x; h < (int)heads; h += gridDim.x) {
        const int kvh = h / gqa;
        const bf16* Qh = Q + (size_t)h * FA_HD;      // seq_q==1
        const bf16* Kh = K + (size_t)kvh * seq_kv * FA_HD;
        const bf16* Vh = V + (size_t)kvh * seq_kv * FA_HD;
        for (int d=tid; d<FA_HD; d+=FA_THREADS) Qsh[d]=Qh[d];
        __syncthreads();

        float m_w=-1e30f, l_w=0.f, o_w[FA_HD/32];
        #pragma unroll
        for (int d=0; d<FA_HD/32; d++) o_w[d]=0.f;

        // Warp w handles KV rows w, w+FA_WARPS, ... (each lane sums HD/32 dims).
        for (int kv = warp; kv < (int)seq_kv; kv += FA_WARPS) {
            float dot=0.f;
            for (int d=lane; d<FA_HD; d+=32)
                dot += __bfloat162float(Qsh[d]) * __bfloat162float(Kh[(size_t)kv*FA_HD+d]);
            dot = warp_sum(dot) * scale;
            float m_new = fmaxf(m_w, dot);
            float corr = __expf(m_w - m_new);
            float p = __expf(dot - m_new);
            l_w = l_w*corr + p;
            #pragma unroll
            for (int d=0; d<FA_HD/32; d++){ int dd=lane+d*32;
                o_w[d] = o_w[d]*corr + p*__bfloat162float(Vh[(size_t)kv*FA_HD+dd]); }
            m_w = m_new;
        }
        // Stash each warp's partial.
        if (lane==0){ pm[warp]=m_w; pl[warp]=l_w; }
        #pragma unroll
        for (int d=0; d<FA_HD/32; d++) po[warp*FA_HD + lane + d*32] = o_w[d];
        __syncthreads();

        // Warp 0 merges the FA_WARPS partials via online softmax.
        if (warp==0){
            float m_all=-1e30f;
            #pragma unroll
            for (int w=0; w<FA_WARPS; w++) m_all=fmaxf(m_all, pm[w]);
            float l_all=0.f, o_all[FA_HD/32];
            #pragma unroll
            for (int d=0; d<FA_HD/32; d++) o_all[d]=0.f;
            #pragma unroll
            for (int w=0; w<FA_WARPS; w++){
                float sc=__expf(pm[w]-m_all); l_all += pl[w]*sc;
                #pragma unroll
                for (int d=0; d<FA_HD/32; d++){ int dd=lane+d*32;
                    o_all[d] += sc * po[w*FA_HD+dd]; }
            }
            float inv=1.0f/l_all;
            #pragma unroll
            for (int d=0; d<FA_HD/32; d++){ int dd=lane+d*32;
                O[(size_t)h*FA_HD+dd] = __float2bfloat16(o_all[d]*inv); }
        }
        __syncthreads();
    }
}

// ===========================================================================
// Row (element-wise / reduce) fused kernels. Memory-bound; one warp per (row,
// head). Grid-stride over row-groups. bf16 vectorized.
// ===========================================================================

// F2/F3 NormRope(Scale): per (row, head) RMSNorm over head_dim then RoPE on the
// first head_dim/2 dims (Gemma partial_rotary_factor=0.5). SCALE adds a final
// *1/sqrt(head_dim) (Q path). theta and position drive the on-the-fly rotation;
// pos_base = coord. heads*head_dim == feat.
// Rotated pairs handled per lane: i runs [0,half) strided by 32, half<=FA_HD/4.
#define NR_PAIRS ((FA_HD/4 + 31) / 32)

template <bool SCALE>
__global__ void gemma_normrope_sm120(const bf16* __restrict__ X, const bf16* __restrict__ gamma,
                                     bf16* __restrict__ Out, unsigned rows, unsigned feat,
                                     unsigned head_dim, float eps, float theta, unsigned pos_base) {
    // One normalized head vector per warp, visible to the whole warp.
    __shared__ float nvs[(GM_THREADS/32) * FA_HD];
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int heads = feat / head_dim;
    const int rot = head_dim / 2;                    // rotated dims
    const int half = rot / 2;                        // pair stride within rotated block
    const long total = (long)rows * heads;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    const float scale = SCALE ? rsqrtf((float)head_dim) : 1.0f;

    for (long t = warp; t < total; t += nwarps) {
        int r = t / heads, hh = t % heads;
        const bf16* xr = X + (size_t)r*feat + (size_t)hh*head_dim;
        bf16* orow = Out + (size_t)r*feat + (size_t)hh*head_dim;
        // RMSNorm over head_dim.
        float ss=0.f;
        for (int d=lane; d<(int)head_dim; d+=32){ float v=__bfloat162float(xr[d]); ss+=v*v; }
        ss = warp_sum(ss);
        float inv = rsqrtf(ss/(float)head_dim + eps);
        // Normalize into SHARED memory, keep for the rotate.
        //
        // This MUST NOT be a per-thread local array: the strided loop below
        // gives lane L only nv[L], nv[L+32], ... , while the rotate reads the
        // partner element nv[i+half]. That index is lane-local only when
        // half==32 (i.e. head_dim==128); for any other head_dim lane L would
        // read its own *uninitialized* local slot instead of the partner lane's
        // value -- a silent wrong answer, not a fault. Shared memory makes the
        // whole normalized vector visible to every lane in the warp.
        float* nv = nvs + (threadIdx.x >> 5) * FA_HD;
        for (int d=lane; d<(int)head_dim; d+=32)
            nv[d] = __bfloat162float(xr[d]) * inv * __bfloat162float(gamma[d]);
        __syncwarp();                    // normalize -> rotate (cross-lane now)
        // RoPE on [0,rot): pair (i, i+half). freq = theta^(-2i/rot).
        // The rotate's read set and write set are the same elements, so every
        // pair is read into registers first, then a single warp-uniform barrier,
        // then written back. (The barrier cannot go inside the loop: for
        // head_dim<128, half<32 and only some lanes iterate, so a __syncwarp()
        // there would be divergent.)
        int pos = pos_base + r;
        float rr0[NR_PAIRS], rr1[NR_PAIRS];
        int np = 0;
        for (int i=lane; i<half; i+=32){
            float freq = __powf(theta, -2.0f*(float)i/(float)rot);
            float ang = pos * freq, c=__cosf(ang), s=__sinf(ang);
            float x0=nv[i], x1=nv[i+half];
            rr0[np] = x0*c - x1*s;
            rr1[np] = x0*s + x1*c;
            np++;
        }
        __syncwarp();                    // all pairs read before any is written
        np = 0;
        for (int i=lane; i<half; i+=32){ nv[i]=rr0[np]; nv[i+half]=rr1[np]; np++; }
        __syncwarp();                    // rotate -> store
        for (int d=lane; d<(int)head_dim; d+=32)
            orow[d] = __float2bfloat16(nv[d]*scale);
        __syncwarp();                    // store -> next t reuses this buffer
    }
}

// F5 SwiGLU: out = silu(gate) * up. Vectorized bf162 (2 elems / thread-step).
__global__ void gemma_swiglu_sm120(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                                   bf16* __restrict__ out, size_t n) {
    size_t i = (size_t)(blockIdx.x*blockDim.x + threadIdx.x) * 2;
    size_t stride = (size_t)gridDim.x * blockDim.x * 2;
    for (; i + 1 < n; i += stride) {
        float2 g = __bfloat1622float2(*(const __nv_bfloat162*)(gate+i));
        float2 u = __bfloat1622float2(*(const __nv_bfloat162*)(up+i));
        float2 r = make_float2(silu(g.x)*u.x, silu(g.y)*u.y);
        *(__nv_bfloat162*)(out+i) = __float22bfloat162_rn(r);
    }
    if (i < n) out[i] = __float2bfloat16(silu(__bfloat162float(gate[i]))*__bfloat162float(up[i]));
}

// S3 Residual add: out = x + residual (bf162 vectorized).
__global__ void gemma_residual_sm120(const bf16* __restrict__ x, const bf16* __restrict__ res,
                                     bf16* __restrict__ out, size_t n) {
    size_t i = (size_t)(blockIdx.x*blockDim.x + threadIdx.x) * 2;
    size_t stride = (size_t)gridDim.x * blockDim.x * 2;
    for (; i + 1 < n; i += stride) {
        __nv_bfloat162 a=*(const __nv_bfloat162*)(x+i), b=*(const __nv_bfloat162*)(res+i);
        *(__nv_bfloat162*)(out+i) = __hadd2(a,b);
    }
    if (i < n) out[i] = __hadd(x[i], res[i]);
}

// S4 Standalone RMSNorm (final norm): per row, rms over feat -> x*rms*gamma.
// One warp per row, grid-stride.
__global__ void gemma_rmsnorm_sm120(const bf16* __restrict__ X, const bf16* __restrict__ gamma,
                                    bf16* __restrict__ Out, unsigned rows, unsigned feat, float eps) {
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    const int nwarps=(gridDim.x*blockDim.x)>>5;
    for (int r=warp; r<(int)rows; r+=nwarps){
        const bf16* xr=X+(size_t)r*feat; bf16* orow=Out+(size_t)r*feat;
        float ss=0.f; for (int d=lane; d<(int)feat; d+=32){float v=__bfloat162float(xr[d]); ss+=v*v;}
        ss=warp_sum(ss); float inv=rsqrtf(ss/(float)feat+eps);
        for (int d=lane; d<(int)feat; d+=32)
            orow[d]=__float2bfloat16(__bfloat162float(xr[d])*inv*__bfloat162float(gamma[d]));
    }
}

// F4 FusedEmbeddingScale: gather embed[id[t],:] and *sqrt(D). One warp per token,
// grid-stride; the gathered row is coalesced within the warp.
__global__ void gemma_embed_scale_sm120(const int* __restrict__ ids, const bf16* __restrict__ table,
                                        bf16* __restrict__ out, unsigned tokens, unsigned d,
                                        float scale) {
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    const int nwarps=(gridDim.x*blockDim.x)>>5;
    for (int t=warp; t<(int)tokens; t+=nwarps){
        int id=ids[t]; const bf16* row=table+(size_t)id*d; bf16* orow=out+(size_t)t*d;
        for (int i=lane; i<(int)d; i+=32) orow[i]=__float2bfloat16(__bfloat162float(row[i])*scale);
    }
}

// ===========================================================================
// extern "C" launchers — map packet bodies/bindings to the kernels above.
// Persistent grid = PLOW_SM120_SMS; dynamic shared opted-in past 48 KiB.
// ===========================================================================

// Persistent grid = co-resident capacity of THIS device for THIS kernel, i.e.
// occupancy x SM count. It used to be the compile-time PLOW_SM120_SMS=188 (the
// RTX PRO 6000 count), which on any smaller part both over-subscribes the SMs
// -- 188 blocks over 170 SMs is a 2-wave launch whose second wave is 18 blocks
// wide, ~2x the tail -- and would make a cooperative launch fail outright.
// Computed once per kernel; both factors are per-process constants.
static int sm120_sm_count() {
    static int n = 0;
    if (n == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            n = PLOW_SM120_SMS;         // fall back to the old constant
    }
    return n;
}
// The static cache lives in a distinct instantiation per kernel, so each kernel
// gets its own occupancy. smem is a compile-time constant at every call site.
template <typename K>
static int sm120_grid(K kernel, int threads, int smem) {
    static int g = 0;
    if (g == 0) {
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, kernel, threads, (size_t)smem) != cudaSuccess || per_sm < 1)
            per_sm = 1;
        g = per_sm * sm120_sm_count();
    }
    return g;
}

static inline int gm_smem_bytes() {
    return (int)((2*GM_ABUF + 2*GM_BBUF) * sizeof(bf16));
}

extern "C" void cuda_gemm_norm_bf16_sm120(const void* body, kctx* ctx) {
    const PlowGemmBody* g = (const PlowGemmBody*)body; const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const bf16* A=(const bf16*)ctx->slots[bd->in0];
    const bf16* B=(const bf16*)ctx->slots[bd->in1];
    const bf16* gamma=(const bf16*)ctx->slots[bd->in2];
    bf16* C=(bf16*)ctx->slots[g->out];
    int act=(bd->detail>>4)&0x0F, sm=gm_smem_bytes();
    cudaFuncSetAttribute(gemma_gemm_sm120<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_gemm_sm120<true><<<sm120_grid(gemma_gemm_sm120<true>, GM_THREADS, sm), GM_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        A,B,gamma,C,g->m,g->n,g->k,bd->scale,act);
}

// Plain bf16 GEMM (o_proj / down_proj / lm_head prefill): no norm prologue.
extern "C" void cuda_gemm_bf16_sm120(const void* body, kctx* ctx) {
    const PlowGemmBody* g=(const PlowGemmBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* A=(const bf16*)ctx->slots[bd->in0];
    const bf16* B=(const bf16*)ctx->slots[bd->in1];
    bf16* C=(bf16*)ctx->slots[g->out];
    int act=(bd->detail>>4)&0x0F, sm=gm_smem_bytes();
    cudaFuncSetAttribute(gemma_gemm_sm120<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_gemm_sm120<false><<<sm120_grid(gemma_gemm_sm120<false>, GM_THREADS, sm), GM_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        A,B,nullptr,C,g->m,g->n,g->k,0.f,act);
}

// Split-K decode partials. coord1 high bits carry the split count (host packs
// it); Cpart scratch = slots[out] sized [split][m][n] f32. A Row reduce-add
// packet sums the partials afterward.
extern "C" void cuda_gemm_norm_splitk_bf16_sm120(const void* body, kctx* ctx) {
    const PlowGemmBody* g=(const PlowGemmBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* A=(const bf16*)ctx->slots[bd->in0];
    const bf16* B=(const bf16*)ctx->slots[bd->in1];
    const bf16* gamma=(const bf16*)ctx->slots[bd->in2];
    float* Cp=(float*)ctx->slots[g->out];
    int split=(int)(g->coord1 >> 16); if (split<1) split=1;
    int sm=gm_smem_bytes();
    cudaFuncSetAttribute(gemma_gemm_splitk_sm120<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_gemm_splitk_sm120<true><<<sm120_grid(gemma_gemm_splitk_sm120<true>, GM_THREADS, sm), GM_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        A,B,gamma,Cp,g->m,g->n,g->k,bd->scale,split,0);
}
extern "C" void cuda_gemm_bf16_splitk_sm120(const void* body, kctx* ctx) {
    const PlowGemmBody* g=(const PlowGemmBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* A=(const bf16*)ctx->slots[bd->in0];
    const bf16* B=(const bf16*)ctx->slots[bd->in1];
    float* Cp=(float*)ctx->slots[g->out];
    int split=(int)(g->coord1 >> 16); if (split<1) split=1;
    int sm=gm_smem_bytes();
    cudaFuncSetAttribute(gemma_gemm_splitk_sm120<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_gemm_splitk_sm120<false><<<sm120_grid(gemma_gemm_splitk_sm120<false>, GM_THREADS, sm), GM_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        A,B,nullptr,Cp,g->m,g->n,g->k,0.f,split,0);
}

static inline int fa_prefill_smem() {
    return (int)((2*FA_BKV*(FA_HD+FA_PAD) + FA_BQ*(FA_HD+FA_PAD)) * sizeof(bf16));
}

extern "C" void cuda_flash_causal_bf16_sm120(const void* body, kctx* ctx) {
    const PlowFlashBody* f=(const PlowFlashBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* Q=(const bf16*)ctx->slots[bd->in0];
    const bf16* K=(const bf16*)ctx->slots[bd->in1];
    const bf16* V=(const bf16*)ctx->slots[bd->in2];
    bf16* O=(bf16*)ctx->slots[f->out];
    unsigned kvh = f->tmem ? f->tmem : (f->heads/4);   // NKV (GQA 4:1 default)
    int sm=fa_prefill_smem();
    cudaFuncSetAttribute(gemma_flash_prefill_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_flash_prefill_sm120<<<sm120_grid(gemma_flash_prefill_sm120, FA_THREADS, sm), FA_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        Q,K,V,O,f->seq_q,f->seq_kv,f->heads,kvh,bd->scale,FA_MASK_CAUSAL,0);
}
extern "C" void cuda_flash_sliding_bf16_sm120(const void* body, kctx* ctx) {
    const PlowFlashBody* f=(const PlowFlashBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* Q=(const bf16*)ctx->slots[bd->in0];
    const bf16* K=(const bf16*)ctx->slots[bd->in1];
    const bf16* V=(const bf16*)ctx->slots[bd->in2];
    bf16* O=(bf16*)ctx->slots[f->out];
    unsigned kvh = f->tmem ? f->tmem : (f->heads/4);
    int window = (int)f->coord0;                       // sliding span in coord0
    int sm=fa_prefill_smem();
    cudaFuncSetAttribute(gemma_flash_prefill_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_flash_prefill_sm120<<<sm120_grid(gemma_flash_prefill_sm120, FA_THREADS, sm), FA_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        Q,K,V,O,f->seq_q,f->seq_kv,f->heads,kvh,bd->scale,FA_MASK_CAUSAL|FA_MASK_SLIDING,window);
}
extern "C" void cuda_flash_decode_bf16_sm120(const void* body, kctx* ctx) {
    const PlowFlashBody* f=(const PlowFlashBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* Q=(const bf16*)ctx->slots[bd->in0];
    const bf16* K=(const bf16*)ctx->slots[bd->in1];
    const bf16* V=(const bf16*)ctx->slots[bd->in2];
    bf16* O=(bf16*)ctx->slots[f->out];
    unsigned kvh = f->tmem ? f->tmem : (f->heads/4);
    int sm=(int)((2*FA_WARPS + FA_WARPS*FA_HD)*sizeof(float));
    cudaFuncSetAttribute(gemma_flash_decode_sm120, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    gemma_flash_decode_sm120<<<sm120_grid(gemma_flash_decode_sm120, FA_THREADS, sm), FA_THREADS, sm, (GPU_STREAM)ctx->stream>>>(
        Q,K,V,O,f->seq_kv,f->heads,kvh,bd->scale);
}

extern "C" void cuda_row_normrope_bf16_sm120(const void* body, kctx* ctx) {
    const PlowRowBody* r=(const PlowRowBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* X=(const bf16*)ctx->slots[bd->in0];
    const bf16* gamma=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[r->out];
    unsigned head_dim = r->br ? r->br : 128;           // head_dim carried in br
    if (head_dim > FA_HD) return;   // per-warp staging buffer is FA_HD floats
    gemma_normrope_sm120<false><<<sm120_grid(gemma_normrope_sm120<false>, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        X,gamma,O,r->rows,r->feat,head_dim,bd->scale,1e6f,r->coord);
}
extern "C" void cuda_row_normropescale_bf16_sm120(const void* body, kctx* ctx) {
    const PlowRowBody* r=(const PlowRowBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* X=(const bf16*)ctx->slots[bd->in0];
    const bf16* gamma=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[r->out];
    unsigned head_dim = r->br ? r->br : 128;
    if (head_dim > FA_HD) return;   // per-warp staging buffer is FA_HD floats
    gemma_normrope_sm120<true><<<sm120_grid(gemma_normrope_sm120<true>, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        X,gamma,O,r->rows,r->feat,head_dim,bd->scale,1e6f,r->coord);
}
extern "C" void cuda_row_swiglu_bf16_sm120(const void* body, kctx* ctx) {
    const PlowRowBody* r=(const PlowRowBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* gate=(const bf16*)ctx->slots[bd->in0];
    const bf16* up=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[r->out];
    gemma_swiglu_sm120<<<sm120_grid(gemma_swiglu_sm120, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        gate,up,O,(size_t)r->rows*r->feat);
}
extern "C" void cuda_row_residual_bf16_sm120(const void* body, kctx* ctx) {
    const PlowRowBody* r=(const PlowRowBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* X=(const bf16*)ctx->slots[bd->in0];
    const bf16* res=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[r->out];
    gemma_residual_sm120<<<sm120_grid(gemma_residual_sm120, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        X,res,O,(size_t)r->rows*r->feat);
}
extern "C" void cuda_row_rmsnorm_bf16_sm120(const void* body, kctx* ctx) {
    const PlowRowBody* r=(const PlowRowBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const bf16* X=(const bf16*)ctx->slots[bd->in0];
    const bf16* gamma=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[r->out];
    gemma_rmsnorm_sm120<<<sm120_grid(gemma_rmsnorm_sm120, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        X,gamma,O,r->rows,r->feat,bd->scale);
}
extern "C" void cuda_layout_gather_scale_bf16_sm120(const void* body, kctx* ctx) {
    const PlowLayoutBody* L=(const PlowLayoutBody*)body; const PlowBinding* bd=ctx->bind;
    if (!bd) return;
    const int* ids=(const int*)ctx->slots[bd->in0];
    const bf16* table=(const bf16*)ctx->slots[bd->in1];
    bf16* O=(bf16*)ctx->slots[L->out];
    unsigned tokens = L->shape[0], d = L->shape[1];
    gemma_embed_scale_sm120<<<sm120_grid(gemma_embed_scale_sm120, GM_THREADS, 0), GM_THREADS, 0, (GPU_STREAM)ctx->stream>>>(
        ids,table,O,tokens,d,bd->scale);
}
