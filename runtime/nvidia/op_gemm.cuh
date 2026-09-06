/* op_gemm.cuh — decode GEMV family for sm_120 (warp32).
 *
 * Ported from runtime/amd/op_gemm.h (gemv_rows / gemv_glu_rows / gemv_qkv). These are
 * MFMA-free on AMD — a GEMV is bandwidth-bound, one row of x against N weight rows, and
 * there is no tensor-core work to port. So the changes are:
 *   (a) wave64 -> warp32: one WARP owns one output column, lane holds 8 contiguous halves,
 *       so one pass covers 32*8 = 256 elements of K (the AMD pass covers 512),
 *   (b) the wave_sum tree re-derived for 32 lanes (warp_sum32 from op_attention.cuh),
 *   (c) the __amdgpu_buffer_rsrc_t weight descriptor REMOVED. That is the one structural
 *       loss. AMD sets num_records = K so every chunk past the end of a row returns zero
 *       from the hardware and fetches nothing — no tail, no predication, nothing to break
 *       the software pipeline. NVIDIA has no equivalent, so the k-loop here carries an
 *       explicit `k < K` predicate. K is a multiple of 256 for every Qwen3 GEMV
 *       (2560/4096/9728 — 9728 = 38*256), so the predicate is uniform across the warp and
 *       costs a branch, not divergence.
 *
 * COLUMN OWNERSHIP is BLOCKED (the AMD GV_BLOCKED=1 default): block `slice` owns the
 * contiguous run [slice*per, slice*per + per). This is not a free choice — the packet
 * compiler's fine-grained gemv->headnorm dependency map ASSUMES blocked ownership, and
 * building interleaved against a PLOW_FINE packet emits dependencies that are silently
 * WRONG (a wrong answer, not an error). Keep them in step.
 */
#pragma once
#include "sm120_common.cuh"
#include <cuda_fp8.h>

/* One pass over K per warp: 32 lanes x 8 halves. */
#define GV_STEP (32u * 8u)
/* Weight loads issued before any is consumed. The AMD default is 11 for the single stream
 * and 6 per stream for the fused GLU; the same reasoning (memory-level parallelism, not
 * instruction slots, is the binding resource) carries over unchanged. */
#ifndef GV_UNROLL
#define GV_UNROLL 8
#endif
#ifndef GV_UNROLL_GLU
#define GV_UNROLL_GLU 4
#endif
/* fp8 weights are 1 byte/elt, so a chunk is 8 B/lane instead of 16 B — the same MLP depth
 * needs its own knob once the batched (MM>1) bodies raise register pressure. Kept at
 * GV_UNROLL by default so the B=1 fp8 path is bit- and schedule-identical to pre-batch. */
#ifndef GV_UNROLL_FP8
#define GV_UNROLL_FP8 GV_UNROLL
#endif
#ifndef GV_UNROLL_GLU_FP8
#define GV_UNROLL_GLU_FP8 GV_UNROLL_GLU
#endif

/* BATCH>1 DECODE row-block walk. GV_MM_MAX is the widest gemv_*_rows<MM> instantiated: one
 * weight row is loaded once and dotted against MM activation rows, so a batch of B costs
 * ceil(B/GV_MM_MAX) weight passes, NOT B. That ceiling is the whole scaling story above B=8 —
 * at GV_MM_MAX=8 a B=32 step re-reads all 22 GiB of weights four times. Raising it trades
 * accumulator registers (MM per stream; the GLU arms hold 2*MM) for weight traffic.
 * gemv_walk hands each block the smallest instantiated rung that covers it, so ragged B
 * (5, 17, ...) costs the same passes as the next power of two, and M<=1 dispatches
 * gemv_*_rows<1> exactly as before — the B=1 serving path is untouched.
 *
 * WHY 8 AND NOT 16. Wider rungs halve the weight traffic at B>=16 but SPILL, and because the
 * interpreter is one megakernel the spill tax lands on every arm — including B=1. Measured,
 * Gemma-4-12B bf16 on RTX PRO 6000 (per-slot TPOT ms / aggregate tok/s):
 *
 *      B     GV_MM_MAX=8      =16              =32
 *      1     18.39 /  54.4    18.59 /  53.8    18.83 /  53.1
 *      8     22.53 / 355.1    27.21 / 294.1    27.31 / 292.9
 *     16     41.34 / 387.0    30.80 / 519.6    57.00 / 280.7
 *     32     75.09 / 426.1    58.87 / 543.6    59.57 / 537.2
 *
 * ptxas: 8 -> 212 regs, 0 spill. 16 -> 255 regs, 72 B spill. 32 -> 255 regs, 1162 B spill
 * stores / 3364 B loads (and it shows: =32 is the worst arm at B=16).
 *
 * So 16 is NOT free: it costs 1.1% at B=1 and 17% at B=8 to buy 34% at B=16 and 28% at B=32.
 * B=1 must not regress (the serving engine's default) and B=8 is the current serving sweet
 * spot, so the DEFAULT STAYS 8. Deployments that pin B>=16 should build the cubin with
 * -DGV_MM_MAX=16; that is a real 1.3x aggregate win and the only knob that moves it. */
#ifndef GV_MM_MAX
#define GV_MM_MAX 8
#endif
template <int N> struct gv_mm { static constexpr int v = N; };

/* WS-BATCHED wide rungs: per-MM unroll depth. The batched bodies are already WEIGHT-STATIONARY
 * (each weight row is loaded ONCE and dotted against all MM x-rows), so the ONLY thing capping
 * the MM ladder at 8 was register pressure: the wide rungs held MM accumulators AND the default
 * UN weight vectors, which spilled at MM=16 (255 regs, 80 B spill) and worse at 32. Because the
 * interpreter is one megakernel, its register count is the MAX over all instantiated rungs, and
 * that spill tax landed on B=1 too — the reason GV_MM_MAX stayed 8.
 *
 * The fix: give the wide rungs a SHALLOWER unroll. Fewer weight vectors in flight frees exactly
 * the registers the extra accumulators need, so a MM=16/32 rung fits under the 212-reg B=1
 * ceiling and adds NOTHING to the megakernel's register count — B=1 stays reg- and byte-identical.
 * MM<=8 keeps the tuned unroll (byte-identical to pre-WS); only 16/32 shrink UN. Weight traffic is
 * unchanged (still one pass per rung); only memory-level parallelism per rung drops, which the
 * MM-way accumulator ILP more than replaces. Knobs are build-overridable for autotune. */
#ifndef GV_UN16
#define GV_UN16 4
#endif
#ifndef GV_UN32
#define GV_UN32 2
#endif
#ifndef GV_UN_GLU16
#define GV_UN_GLU16 2
#endif
#ifndef GV_UN_GLU32
#define GV_UN_GLU32 1
#endif
template <int MM> struct gv_un         { static constexpr int v = GV_UNROLL; };
template <> struct gv_un<16>           { static constexpr int v = GV_UN16; };
template <> struct gv_un<32>           { static constexpr int v = GV_UN32; };
template <int MM> struct gv_un_glu     { static constexpr int v = GV_UNROLL_GLU; };
template <> struct gv_un_glu<16>       { static constexpr int v = GV_UN_GLU16; };
template <> struct gv_un_glu<32>       { static constexpr int v = GV_UN_GLU32; };
template <int MM> struct gv_un_fp8     { static constexpr int v = GV_UNROLL_FP8; };
template <> struct gv_un_fp8<16>       { static constexpr int v = GV_UN16; };
template <> struct gv_un_fp8<32>       { static constexpr int v = GV_UN32; };
template <int MM> struct gv_un_glu_fp8 { static constexpr int v = GV_UNROLL_GLU_FP8; };
template <> struct gv_un_glu_fp8<16>   { static constexpr int v = GV_UN_GLU16; };
template <> struct gv_un_glu_fp8<32>   { static constexpr int v = GV_UN_GLU32; };

/* f(gv_mm<MM>{}, m0, rows): one row-block of `rows` (<= MM) live rows starting at row m0. */
template <class F> __device__ __forceinline__ void gemv_walk(unsigned M, F f) {
    if (M <= 1) { f(gv_mm<1>{}, 0u, M); return; }
    unsigned m0 = 0;
    for (; m0 + GV_MM_MAX <= M; m0 += GV_MM_MAX) f(gv_mm<GV_MM_MAX>{}, m0, (unsigned)GV_MM_MAX);
    const unsigned rem = M - m0;
    if (rem == 0) return;
    if (rem <= 2) f(gv_mm<2>{}, m0, rem);
    else if (rem <= 4) f(gv_mm<4>{}, m0, rem);
#if GV_MM_MAX > 8
    else if (rem <= 8) f(gv_mm<8>{}, m0, rem);
#if GV_MM_MAX > 16
    else if (rem <= 16) f(gv_mm<16>{}, m0, rem);
#endif
#endif
    else f(gv_mm<GV_MM_MAX>{}, m0, rem);
}

/* C[m][n] = dot(x[m][:], W[n][:]). W is [N, K] — HF nn.Linear layout, row n is output n. */
template <int MM, int UN = gv_un<MM>::v, bool BIAS = false>
__device__ __forceinline__ void gemv_rows(__nv_bfloat16* __restrict__ C,
                                          const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ W, unsigned M,
                                          unsigned N, unsigned K, unsigned slice, unsigned nblk,
                                          const __nv_bfloat16* __restrict__ bias = nullptr) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;

        /* CONTRACT: K is a multiple of 8. The 8-wide ld_glob8 below is guarded only by `k < K`,
         * so an unaligned K would over-read the final vector past the row. plowc enforces this at
         * emit time (gemma4.rs: hidden/intermediate/head_dim % 8 == 0); this and the QKV/GLU GEMV
         * variants share the assumption. */
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
            /* Issue all UN loads before touching any of them. */
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld_glob8(wrow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue; /* wv is zero anyway; this also guards the x read */
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    const bf16v8 xv = ld_glob8(x + (size_t)m * K + kk[u]);
                    acc[m] = dot8(wv[u], xv, acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            float t = warp_sum32(acc[m]);
            if constexpr (BIAS) t += __bfloat162float(bias[n]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t);
        }
    }
}

/* BATCH>1 DECODE (serving pending #4). gemv_rows<MM> loads each weight row ONCE and dots it
 * against all MM rows of x, so B batched decode rows cost ~1 weight read instead of B (the whole
 * batching thesis; measured 3.7-4.1x at MM=8 on HBM-resident weights, runtime/tests/gemv_batch_sm120.cu).
 * d_gemv used to be a bare gemv_rows<1>, which left rows 1..M UNWRITTEN at M>1 — fluent wrong text,
 * no crash. Dispatch the smallest instantiated MM that covers M; M==1 stays byte-identical (the
 * B=1 serving blob depends on it). MM ladder is {1,2,4,8}; M>8 walks in blocks of 8. */
template <int MM>
__device__ __forceinline__ void gemv_block(__nv_bfloat16* C, const __nv_bfloat16* x,
                                           const __nv_bfloat16* W, unsigned M, unsigned N,
                                           unsigned K, unsigned slice, unsigned nblk) {
    gemv_rows<MM>(C, x, W, M, N, K, slice, nblk);
}
static __device__ void d_gemv(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ W, unsigned M, unsigned N, unsigned K,
                       unsigned slice, unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_rows<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, W, rows, N, K,
                                   slice, nblk);
    });
}

#if PLOW_PACKET_LINEAR_BIAS
static __device__ void d_gemv_bias(__nv_bfloat16* __restrict__ C,
                                   const __nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ W,
                                   const __nv_bfloat16* __restrict__ bias, unsigned M,
                                   unsigned N, unsigned K, unsigned slice, unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_rows<decltype(mm)::v, gv_un<decltype(mm)::v>::v, true>(
            C + (size_t)m0 * N, x + (size_t)m0 * K, W, rows, N, K, slice, nblk, bias);
    });
}
#endif

/* ---- ROW-BLOCKED M=1 DECODE GEMV CORE (PLOW_NV_GEMV_RB, H100 campaign E2) ----------------
 * One warp owning ONE output row keeps only UN weight loads in flight. On H100 at the
 * megakernel's 1 block/SM that is ~2.5x short of the achievable read bandwidth
 * (runtime/nvidia/experiments/hbm_ceiling_h100.cu: a pure read goes 1222 -> 2490 GB/s as
 * loads-in-flight go 1 -> 8). Giving a warp GV_RB rows makes RB*UN loads outstanding at
 * unchanged occupancy — measured 1.4x on the real decode shapes
 * (runtime/nvidia/experiments/gemv_lab_h100.cu).
 * The K walk and the FMA order WITHIN a row are untouched, so every C[n] is BIT-IDENTICAL
 * to the RB=1 body; only which warp owns a row changes. Default OFF => sm_120 objects stay
 * byte-identical. See perf-data/gemma26b-h100-gemv-mlp.md. */
/* MEASURED OFF (26B/H100, ctx1024 B=1, TPOT vs the MoE-only build at 7.898 ms):
 *   d_gemv (o_proj/down)      8.125  (+0.227)
 *   d_gemv_argmax (lm_head)   7.921  (+0.023)
 *   d_gemv_qkv                8.093  (+0.195)
 * Row-blocking wins these shapes in ISOLATION (gemv_lab_h100.cu: qkv 1.43x, lm_head 1.42x)
 * but loses in the megakernel, and not through registers — a REG:177 variant measured slower
 * than a REG:229 one, and the o_proj/down arm costs 0.227 ms even when its runtime guard keeps
 * the blocked loop from ever executing. Inlining the addresses (gv_rb_smemx_contig) did not
 * change it either. The MoE arms, whose defect was scalar loads rather than row count, keep
 * their full win. Left compiled-out but intact: flip a flag to re-test on another part. */
#ifndef PLOW_NV_GEMV_LS
#define PLOW_NV_GEMV_LS 0
#endif
#ifndef PLOW_NV_FP8_RB
#define PLOW_NV_FP8_RB 1
#endif
#ifndef PLOW_NV_GEMV_NOSTAGE
#define PLOW_NV_GEMV_NOSTAGE 0
#endif
#ifndef PLOW_NV_GEMV_STAGE_MINROWS
#define PLOW_NV_GEMV_STAGE_MINROWS 16u
#endif
#ifndef GV_LS_SG
#define GV_LS_SG 4u
#endif
#ifndef PLOW_NV_RB_GEMV
#define PLOW_NV_RB_GEMV 0
#endif
#ifndef PLOW_NV_RB_LMHEAD
#define PLOW_NV_RB_LMHEAD 0
#endif
#ifndef PLOW_NV_RB_QKV
#define PLOW_NV_RB_QKV 0
#endif
#if PLOW_NV_GEMV_RB
#ifndef GV_RB
#define GV_RB 4
#endif
#ifndef GV_UNROLL_RB
#define GV_UNROLL_RB 4
#endif
/* Contiguous-row twin: rows n0..n0+RB-1 of W. Addresses are formed inline as base + immediate
 * so ptxas emits one base register plus constant offsets, instead of indexing a pointer array
 * (the array form measurably costs address arithmetic in the megakernel). */
template <int RB, int UN>
__device__ __forceinline__ void gv_rb_smemx_contig(float* __restrict__ acc,
                                                   const __nv_bfloat16* __restrict__ W,
                                                   unsigned n0, const __nv_bfloat16* __restrict__ xs,
                                                   unsigned K, unsigned lane, unsigned nrows) {
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const __nv_bfloat16* base = W + (size_t)n0 * K;
    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 wv[RB][UN];
        unsigned kk[UN];
#pragma unroll
        for (int u = 0; u < UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
        for (int r = 0; r < RB; r++) {
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[r][u] = ((unsigned)r < nrows && kk[u] < K)
                               ? ld_glob8(base + (size_t)r * K + kk[u])
                               : bf16v8_zero();
        }
#pragma unroll
        for (int u = 0; u < UN; u++) {
            if (kk[u] >= K) continue;
            const bf16v8 xv = ld_smem8(xs + kk[u]);
#pragma unroll
            for (int r = 0; r < RB; r++) acc[r] = dot8(wv[r][u], xv, acc[r]);
        }
    }
}
/* Accumulate RB weight rows (wrow[r], any provenance) against a smem-staged x row. */
template <int RB, int UN>
__device__ __forceinline__ void gv_rb_smemx(float* __restrict__ acc,
                                            const __nv_bfloat16* const* __restrict__ wrow,
                                            const __nv_bfloat16* __restrict__ xs, unsigned K,
                                            unsigned lane, unsigned nrows) {
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 wv[RB][UN];
        unsigned kk[UN];
#pragma unroll
        for (int u = 0; u < UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
        for (int r = 0; r < RB; r++) {
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[r][u] = ((unsigned)r < nrows && kk[u] < K) ? ld_glob8(wrow[r] + kk[u])
                                                             : bf16v8_zero();
        }
#pragma unroll
        for (int u = 0; u < UN; u++) {
            if (kk[u] >= K) continue;
            const bf16v8 xv = ld_smem8(xs + kk[u]);
#pragma unroll
            for (int r = 0; r < RB; r++) acc[r] = dot8(wv[r][u], xv, acc[r]);
        }
    }
}
#endif

/* Arena-aware M=1 decode GEMV: x staged into smem, weights stream from GDDR exclusively. */
static __device__ void d_gemv(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ W, unsigned M, unsigned N, unsigned K,
                       unsigned slice, unsigned nblk, __nv_bfloat16* __restrict__ arena) {
    if (M > 1) { d_gemv(C, x, W, M, N, K, slice, nblk); return; }
#if PLOW_NV_GEMV_NOSTAGE
    /* Staging x costs K element-copies + a __syncthreads PER BLOCK, and it only pays if the
     * block then reads many weight rows against it. o_proj/down are N=2816, so at nblk=264 a
     * block owns per=ceil(N/nblk)=11 rows and the staging is no longer amortised -- while QKV
     * (N=8192, per=63) is at 91% of the pure-read ceiling with the same arm. Below the
     * threshold, read x straight from global (it is a few KiB and L2-resident). */
    if ((N + nblk - 1u) / nblk < PLOW_NV_GEMV_STAGE_MINROWS) {
        d_gemv(C, x, W, M, N, K, slice, nblk);
        return;
    }
#endif
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

#if PLOW_NV_GEMV_LS
    /* LANE-SPLIT GEMV (H100 round 6). Row-blocking wants more output rows in flight per warp,
     * but wv[RB][UN] costs registers and RB>2 blows the megakernel's budget. Splitting the warp
     * into GV_LS_SG sub-groups of 32/GV_LS_SG lanes gives the same rows-in-flight with ONE
     * accumulator per lane and no wv[][] array at all, plus a shorter reduction. Each sub-group's
     * lanes read contiguous bytes, so a warp instruction is GV_LS_SG contiguous bursts.
     * The per-lane K partition and the reduction width change, so results are numerically
     * equivalent, not bit-identical. Needs K % (lanes_per_group*8) == 0. */
    {
        constexpr unsigned SG = GV_LS_SG, SL = 32u / GV_LS_SG, CH = SL * 8u;
        if ((K % CH) == 0u) {
            const unsigned sg = lane / SL, sl = lane % SL;
            const unsigned nch = K / CH;
            for (unsigned nb = n0 + warp * SG; nb < n1; nb += PLOW_NV_WARPS * SG) {
                const unsigned n = nb + sg;
                float acc = 0.0f;
                if (n < n1) {
                    const __nv_bfloat16* wrow = W + (size_t)n * K;
                    unsigned c = 0;
                    for (; c + 2u <= nch; c += 2u) {
                        const unsigned k0 = c * CH + sl * 8u, k1 = (c + 1u) * CH + sl * 8u;
                        const bf16v8 w0 = ld_glob8(wrow + k0), w1 = ld_glob8(wrow + k1);
                        acc = dot8(w0, ld_smem8(xs + k0), acc);
                        acc = dot8(w1, ld_smem8(xs + k1), acc);
                    }
                    for (; c < nch; c++) {
                        const unsigned k0 = c * CH + sl * 8u;
                        acc = dot8(ld_glob8(wrow + k0), ld_smem8(xs + k0), acc);
                    }
                }
#pragma unroll
                for (int o = 1; o < (int)SL; o <<= 1) acc += __shfl_xor_sync(~0u, acc, o);
                if (sl == 0u && M && n < n1) C[n] = __float2bfloat16(acc);
            }
            return;
        }
    }
#endif
#if PLOW_NV_RB_GEMV
    /* A warp takes GV_RB CONSECUTIVE rows: the RB weight rows are one contiguous span, which
     * keeps the DRAM/L2 locality the strided variant destroyed (measured: striding costs more
     * than the imbalance it fixes). Consecutive blocking only balances when a block owns at
     * least WARPS*RB rows, so ops with a thin per (o_proj/down: per=22 at nblk=132) keep the
     * unblocked body below. */
    if (per >= PLOW_NV_WARPS * (unsigned)GV_RB)
    for (unsigned nb = n0 + warp * GV_RB; nb < n1; nb += PLOW_NV_WARPS * GV_RB) {
        const unsigned nrows = (nb + GV_RB <= n1) ? (unsigned)GV_RB : (n1 - nb);
        float acc[GV_RB];
#pragma unroll
        for (int r = 0; r < GV_RB; r++) acc[r] = 0.0f;
        gv_rb_smemx_contig<GV_RB, GV_UNROLL_RB>(acc, W, nb, xs, K, lane, nrows);
#pragma unroll
        for (int r = 0; r < GV_RB; r++) {
            const float t = warp_sum32(acc[r]);
            if (lane == 0 && M && (unsigned)r < nrows) C[nb + (unsigned)r] = __float2bfloat16(t);
        }
    }
    else
#endif
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL) {
            bf16v8 wv[GV_UNROLL];
            unsigned kk[GV_UNROLL];
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld_glob8(wrow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld_smem8(xs + kk[u]), acc);
            }
        }
        const float t = warp_sum32(acc);
        if (lane == 0 && M) C[n] = __float2bfloat16(t);
    }
}

/* E5 (rtx-19): lm_head decode GEMV with the greedy-argmax epilogue FUSED in (PLOW_FUSE_ARGMAX).
 * Byte-identical twin of d_gemv (M=1 arena path) → SoftCap → Argmax: each block owns the SAME
 * contiguous vocab slice [n0,n1) it would compute, and instead of leaving the argmax to two extra
 * packets it folds the slice into one packed-u64 partial part[slice] right here. The stored logit
 * AND the reduced key both use the SOFTCAPPED bf16 value (reproducing d_softcap's arithmetic bit
 * for bit, cap=f0, 0=none), so C matches the post-SoftCap logits tensor and the token matches
 * ARGMAX_FIN's — the max over all N packed keys is partition-independent (ties break on ~index,
 * carried in the key). ArgmaxFin then folds `nblk` partials instead of AMAX_BLOCKS. */
static __device__ void d_gemv_argmax(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ W, unsigned long long* __restrict__ part,
                       unsigned N, unsigned K, float cap, unsigned slice, unsigned nblk,
                       __nv_bfloat16* __restrict__ arena) {
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;
    const float inv = cap > 0.0f ? 1.0f / cap : 0.0f;

    unsigned long long best = 0;
#if PLOW_NV_RB_LMHEAD
    /* A warp takes GV_RB CONSECUTIVE rows: the RB weight rows are one contiguous span, which
     * keeps the DRAM/L2 locality the strided variant destroyed (measured: striding costs more
     * than the imbalance it fixes). Consecutive blocking only balances when a block owns at
     * least WARPS*RB rows, so ops with a thin per (o_proj/down: per=22 at nblk=132) keep the
     * unblocked body below. */
    if (per >= PLOW_NV_WARPS * (unsigned)GV_RB)
    for (unsigned nb = n0 + warp * GV_RB; nb < n1; nb += PLOW_NV_WARPS * GV_RB) {
        const unsigned nrows = (nb + GV_RB <= n1) ? (unsigned)GV_RB : (n1 - nb);
        float acc[GV_RB];
#pragma unroll
        for (int r = 0; r < GV_RB; r++) acc[r] = 0.0f;
        gv_rb_smemx_contig<GV_RB, GV_UNROLL_RB>(acc, W, nb, xs, K, lane, nrows);
#pragma unroll
        for (int r = 0; r < GV_RB; r++) {
            const float t = warp_sum32(acc[r]);
            const __nv_bfloat16 lg = __float2bfloat16(t);
            const __nv_bfloat16 sc =
                cap > 0.0f ? __float2bfloat16(cap * tanhf(__bfloat162float(lg) * inv)) : lg;
            if (lane == 0 && (unsigned)r < nrows) {
                const unsigned n = nb + (unsigned)r;
                C[n] = sc;
                const unsigned long long key = amax_pack(sc, n);
                best = key > best ? key : best;
            }
        }
    }
    else
#endif
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL) {
            bf16v8 wv[GV_UNROLL];
            unsigned kk[GV_UNROLL];
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld_glob8(wrow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld_smem8(xs + kk[u]), acc);
            }
        }
        const float t = warp_sum32(acc); /* all 32 lanes hold the sum */
        /* Reproduce Gemv's bf16 store, then d_softcap, then amax_pack — bit for bit. */
        const __nv_bfloat16 lg = __float2bfloat16(t);
        const __nv_bfloat16 sc =
            cap > 0.0f ? __float2bfloat16(cap * tanhf(__bfloat162float(lg) * inv)) : lg;
        if (lane == 0) {
            C[n] = sc;
            const unsigned long long key = amax_pack(sc, n);
            best = key > best ? key : best;
        }
    }
    __syncthreads(); /* all xs reads retired before block_max_u64 reuses arena as scratch */
    best = block_max_u64(best, (unsigned long long*)arena);
    if (threadIdx.x == 0) part[slice] = best;
}

/* FUSED QKV: one x row against three weight matrices, ownership BLOCKED over the
 * CONCATENATED output [0,Nq) u [Nq,Nq+Nk) u [Nq+Nk,Nq+Nk+Nv). x is read once per column
 * regardless, but fusing collapses three packets (three global gates, three single-row ops
 * that 169 blocks stall behind) into one. */
/* BATCH>1: one weight row feeds all MM x-rows (Cq/Ck/Cv are each [M][Nx]). MM==1 is
 * byte-identical to the old scalar-accumulator body (the B=1 serving path). */
template <int MM, int UN = gv_un<MM>::v, bool BIAS = false>
__device__ __forceinline__ void gemv_qkv_rows(__nv_bfloat16* Cq, __nv_bfloat16* Ck,
                           __nv_bfloat16* Cv, const __nv_bfloat16* x, const __nv_bfloat16* Wq,
                           const __nv_bfloat16* Wk, const __nv_bfloat16* Wv, unsigned M, unsigned Nq,
                           unsigned Nk, unsigned Nv, unsigned K, unsigned slice, unsigned nblk,
                           const __nv_bfloat16* bq = nullptr,
                           const __nv_bfloat16* bk = nullptr,
                           const __nv_bfloat16* bv = nullptr) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned N = Nq + Nk + Nv;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned g0 = slice * per;
    const unsigned g1 = (g0 + per < N) ? (g0 + per) : N;

    for (unsigned g = g0 + warp; g < g1; g += PLOW_NV_WARPS) {
        /* Route the concatenated column to its matrix. */
        const __nv_bfloat16* W;
        const __nv_bfloat16* bias;
        __nv_bfloat16* C;
        unsigned Nx, n;
        if (g < Nq) {
            W = Wq; C = Cq; Nx = Nq; n = g; bias = bq;
        } else if (g < Nq + Nk) {
            W = Wk; C = Ck; Nx = Nk; n = g - Nq; bias = bk;
        } else {
            W = Wv; C = Cv; Nx = Nv; n = g - Nq - Nk; bias = bv;
        }
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
                wv[u] = (k < K) ? ld_glob8(wrow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld_glob8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            float t = warp_sum32(acc[m]);
            if constexpr (BIAS) t += __bfloat162float(bias[n]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * Nx + n] = __float2bfloat16(t);
        }
    }
}

#if PLOW_PACKET_LINEAR_BIAS
static __device__ void d_gemv_qkv_bias(
    __nv_bfloat16* __restrict__ Cq, __nv_bfloat16* __restrict__ Ck,
    __nv_bfloat16* __restrict__ Cv, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ Wq, const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv, const __nv_bfloat16* __restrict__ bq,
    const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv,
    unsigned M, unsigned Nq, unsigned Nk, unsigned Nv, unsigned K, unsigned slice,
    unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_qkv_rows<decltype(mm)::v, gv_un<decltype(mm)::v>::v, true>(
            Cq + (size_t)m0 * Nq, Ck + (size_t)m0 * Nk, Cv + (size_t)m0 * Nv,
            x + (size_t)m0 * K, Wq, Wk, Wv, rows, Nq, Nk, Nv, K, slice, nblk, bq,
            bk, bv);
    });
}
#endif
static __device__ void d_gemv_qkv(__nv_bfloat16* __restrict__ Cq, __nv_bfloat16* __restrict__ Ck,
                           __nv_bfloat16* __restrict__ Cv, const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ Wq,
                           const __nv_bfloat16* __restrict__ Wk,
                           const __nv_bfloat16* __restrict__ Wv, unsigned M, unsigned Nq,
                           unsigned Nk, unsigned Nv, unsigned K, unsigned slice, unsigned nblk) {
    /* Before the walk this was a bare ladder topping out at MM=8: feeding M=32 straight to
     * gemv_qkv_rows<8> left rows 8..31 UNWRITTEN — fluent wrong text, no crash. */
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_qkv_rows<decltype(mm)::v>(Cq + (size_t)m0 * Nq, Ck + (size_t)m0 * Nk,
                                       Cv + (size_t)m0 * Nv, x + (size_t)m0 * K, Wq, Wk, Wv,
                                       rows, Nq, Nk, Nv, K, slice, nblk);
    });
}

/* Arena-aware M=1 decode QKV GEMV: x staged into smem. */
static __device__ void d_gemv_qkv(__nv_bfloat16* __restrict__ Cq, __nv_bfloat16* __restrict__ Ck,
                           __nv_bfloat16* __restrict__ Cv, const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ Wq,
                           const __nv_bfloat16* __restrict__ Wk,
                           const __nv_bfloat16* __restrict__ Wv, unsigned M, unsigned Nq,
                           unsigned Nk, unsigned Nv, unsigned K, unsigned slice, unsigned nblk,
                           __nv_bfloat16* __restrict__ arena) {
    if (M > 1) { d_gemv_qkv(Cq, Ck, Cv, x, Wq, Wk, Wv, M, Nq, Nk, Nv, K, slice, nblk); return; }
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned N = Nq + Nk + Nv;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned g0 = slice * per;
    const unsigned g1 = (g0 + per < N) ? (g0 + per) : N;

#if PLOW_NV_RB_QKV
    /* Row-blocked twin. The RB rows of a group can straddle the Q|K|V concatenation, so each
     * carries its own weight-row and destination pointer; the routing arithmetic is unchanged. */
    if (per >= PLOW_NV_WARPS * (unsigned)GV_RB)
    for (unsigned gb = g0 + warp * GV_RB; gb < g1; gb += PLOW_NV_WARPS * GV_RB) {
        const unsigned nrows = (gb + GV_RB <= g1) ? (unsigned)GV_RB : (g1 - gb);
        const __nv_bfloat16* wrow[GV_RB];
        __nv_bfloat16* dst[GV_RB];
        float acc[GV_RB];
#pragma unroll
        for (int r = 0; r < GV_RB; r++) {
            const unsigned g = gb + (unsigned)r;
            const unsigned gc = (g < g1) ? g : g0; /* clamp: masked off by nrows below */
            const __nv_bfloat16* W;
            __nv_bfloat16* C;
            unsigned n;
            if (gc < Nq) {
                W = Wq; C = Cq; n = gc;
            } else if (gc < Nq + Nk) {
                W = Wk; C = Ck; n = gc - Nq;
            } else {
                W = Wv; C = Cv; n = gc - Nq - Nk;
            }
            wrow[r] = W + (size_t)n * K;
            dst[r] = C + n;
            acc[r] = 0.0f;
        }
        gv_rb_smemx<GV_RB, GV_UNROLL_RB>(acc, wrow, xs, K, lane, nrows);
#pragma unroll
        for (int r = 0; r < GV_RB; r++) {
            const float t = warp_sum32(acc[r]);
            if (lane == 0 && M && (unsigned)r < nrows) *dst[r] = __float2bfloat16(t);
        }
    }
    else
#endif
    for (unsigned g = g0 + warp; g < g1; g += PLOW_NV_WARPS) {
        const __nv_bfloat16* W;
        __nv_bfloat16* C;
        unsigned n;
        if (g < Nq) {
            W = Wq; C = Cq; n = g;
        } else if (g < Nq + Nk) {
            W = Wk; C = Ck; n = g - Nq;
        } else {
            W = Wv; C = Cv; n = g - Nq - Nk;
        }
        const __nv_bfloat16* wrow = W + (size_t)n * K;
        float acc = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL) {
            bf16v8 wv[GV_UNROLL];
            unsigned kk[GV_UNROLL];
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? ld_glob8(wrow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld_smem8(xs + kk[u]), acc);
            }
        }
        const float t = warp_sum32(acc);
        if (lane == 0 && M) C[n] = __float2bfloat16(t);
    }
}

/* ============================ PREFILL TILED GEMM (mma.sync m16n8k16) ==============================
 * Ported from gemma_sm120.cu's gemma_gemm_sm120<false> (validated single-SM prefill GEMM), refactored
 * to a __device__ body for the megakernel: (a) slices output TILES by the packet's (slice, nblk)
 * instead of (blockIdx, gridDim) — the block IS the "CU" inside the interpreter, its share is carried
 * in the stream entry; (b) takes a `bf16* arena` off the interpreter's union instead of its own
 * `extern __shared__`; (c) honours a_row0 (i4) so lm_head can be M=1 over the LAST prompt row.
 *
 * bf16 storage, f32 accumulate. Block tile 128x64x32, 8 warps in a 4x2 grid, each warp a 32x32
 * output sub-tile = 2 m-frag x 4 n-frag. Double-buffered smem, cp.async.cg staged. The math is
 * byte-identical to the legacy kernel; the oracle (sm120_interp_op_test) diffs it against an f32 CPU
 * reference, and the three tile opcodes (GEMM 256x256, GEMM_MED 128x128, GEMM_SMALL 64x128) all
 * dispatch here — the tile geometry is a perf knob, not a correctness one, so one body serves all
 * three; a tuned per-opcode tiling is a later A/B. */
/* Block tile 128x128x32 (T2: BN 64->128 halves the activation re-read traffic — A is streamed
 * once per N-tile, so a wider N-tile is a direct global-bandwidth win on the memory-bound prefill
 * GEMM). Per warp owns PGM_WM x PGM_WN = 32x64 = 2 m-frag x 8 n-frag; acc grows 32->64 f32/thread.
 * The megakernel's register ceiling is set by flash-decode<512,2>, which still dominates.
 *
 * T3: REAL cp.async pipeline. Both operands are staged in their NATURAL, K-contiguous layout
 * (A [m][k], B [n][k]) so each smem row is a run of contiguous bf16 that cp.async.cg can fill with
 * 16-byte (8-bf16) gmem->smem line copies. The B operand no longer stores TRANSPOSED [k][n]
 * (the T2 blocker: a scatter of 8 elements to 8 rows, which cp.async cannot express). Instead B is
 * staged [n][k] and the mma B fragment is read with ldmatrix.x2 NON-.trans — proven bit-identical to
 * the old [k][n]+ldmatrix.x2.trans path by runtime/nvidia/experiments/t3_pipe_probe.cu (0/2048 reg
 * mismatches). Because the mma operands are identical bf16 values in identical lanes, the f32
 * accumulation is bit-exact vs T2: greedy tokens are unchanged. */
/* PX-13: overridable so the prefill tile space can be swept. Default 128 is unchanged. BM must
 * keep PGM_WM = BM/WARPS_M a multiple of 16 (one m-fragment). */
#ifndef PGM_BM
#define PGM_BM 128
#endif
/* PX-3: BN overridable so the lean occ-2 GEMM segment object can shrink the N-tile to 64. A
 * 64-wide N-tile drops the plain arena to 3*(ABUF+BBUF)=45 KiB (from 60 KiB at BN=128), which
 * fits 2 blocks/SM under the 100 KiB dynamic-smem cap WITHOUT halving the 3-stage pipeline (the
 * T10 wall). Default 128 is the T2 occ-1 win and is unchanged for every other object. */
#ifndef PGM_BN
#define PGM_BN 128
#endif
#define PGM_BK 32
#define PGM_APAD 8
#define PGM_BPAD 8
#define PGM_WARPS_M 4
#define PGM_WARPS_N 2
#define PGM_WM (PGM_BM / PGM_WARPS_M)   /* 32 */
#define PGM_WN (PGM_BN / PGM_WARPS_N)   /* 64 */
#define PGM_MFRAG (PGM_WM / 16)          /* 2  m-fragments per warp */
#define PGM_NFRAG (PGM_WN / 8)           /* 8  n-fragments per warp */
#define PGM_ASTRIDE (PGM_BK + PGM_APAD)  /* 40  A smem: [m][k], k contiguous */
#define PGM_BKSTRIDE (PGM_BK + PGM_BPAD) /* 40  B smem: [n][k], k contiguous (was [k][n] transposed) */
#define PGM_ABUF (PGM_BM * PGM_ASTRIDE)  /* 5120 bf16 per stage */
#define PGM_BBUF (PGM_BN * PGM_BKSTRIDE) /* 5120 bf16 per stage */
/* GEMM_GLU's own N-tile, independent of the plain-GEMM PGM_BN above. GLU carries TWO
 * accumulator sets (gate, up) instead of one, so its register cost per N-tile is double the
 * plain body's at the same width — a narrower N-tile trades re-read bandwidth for register
 * headroom specifically on this arm. Default equals PGM_BN (no behavioral change unless
 * overridden). Must stay a multiple of PGM_WARPS_N*8 (one n-fragment per warp-column). */
#ifndef PGM_BN_GLU
#define PGM_BN_GLU PGM_BN
#endif
#define PGM_WN_GLU (PGM_BN_GLU / PGM_WARPS_N)
#define PGM_NFRAG_GLU (PGM_WN_GLU / 8)
#define PGM_BBUF_GLU (PGM_BN_GLU * PGM_BKSTRIDE)
/* Pipeline depth. Plain GEMM rings STAGES buffers of (A,B); GEMM_GLU rings GLU_STAGES of (A,Bg,Bu).
 * Arena (bf16) = the max claim. STAGES=3 plain -> 3*(ABUF+BBUF)=30720; GLU_STAGES=2 -> 2*(ABUF+2*BBUF)
 * =30720. Both = 30720 bf16 (60.0 KiB). That is under flash-prefill's 77.5 KiB, so the megakernel's
 * smem UNION (and therefore its occupancy) is unchanged from T2; but it is over 48 KiB, so the
 * oracle's standalone k_gemm launch now opts in via cudaFuncSetAttribute. */
/* Overridable so the lean occ-2 GEMM segment object (PLOW_NV_SEG_GEMM) can shrink the pipeline
 * depth to fit the arena under the 100 KiB dynamic-smem cap at 2 blocks/SM (T10). Default depths
 * (3 / 2) are the tuned megakernel values and are unchanged for every other object. */
#ifndef PGM_STAGES
#define PGM_STAGES 3
#endif
#ifndef PGM_GLU_STAGES
#define PGM_GLU_STAGES 2
#endif
#define PGM_ARENA_PLAIN (PGM_STAGES * (PGM_ABUF + PGM_BBUF))
#define PGM_ARENA_GLU (PGM_GLU_STAGES * (PGM_ABUF + 2 * PGM_BBUF_GLU))
#define PGM_ARENA_PGM (PGM_ARENA_PLAIN > PGM_ARENA_GLU ? PGM_ARENA_PLAIN : PGM_ARENA_GLU)
/* sm_90a fork (op_gemm_sm90.cuh): the wgmma 128-BYTE-swizzle descriptor needs each logical smem
 * row to be exactly 128 B, so the K-tile deepens to 64 bf16 / 128 e4m3 and one pipeline stage is
 * A[128][128 B] + B[128][128 B] = 32 KiB (vs the mma.sync body's 20 KiB). PGM90_STAGES of that,
 * plus 1024 B of slack for the swizzle's mandatory 1024 B tile alignment. Kept in step with the
 * header by a static_assert there. The arena is a UNION with flash prefill, so this claim only
 * moves PLOW_NV_ARENA_FLOATS when it exceeds flash's 19840 floats (77.5 KiB) — at STAGES=3 it
 * does: 19840 -> 24832 floats (77.5 -> 97.0 KiB). Prefill runs 1 block/SM and H100 allows 227 KiB
 * so that is affordable; PGM90_STAGES=2 (65 KiB) keeps the arena at 77.5 KiB if it ever isn't.
 * NOTE: the lean occ-2 object (PLOW_NV_SEG_GEMM) wants <=50 KiB and this claim breaks that — it
 * is an sm_120a-only build configuration and is never set for sm_90a. */
#if defined(PLOW_NV_HOPPER)
#ifndef PGM90_STAGES
#define PGM90_STAGES 3
#endif
#define PGM_ARENA_SM90 (PGM90_STAGES * (128 * 64 + 128 * 64) + 512)
/* The GLU fork holds TWO m64n128 accumulator sets (128 f32/thread, same as the mma.sync GLU) and
 * is the most register-hungry of the four wgmma bodies. It measures 1.15-1.22x (bf16) / 2.1-2.3x
 * (w8a8) standalone, but in the megakernel it costs a further ~680 spill instructions on top of
 * the plain-GEMM fork's ~910 — the prefill object is already at the 255-register ceiling.
 *
 * DEFAULT 0, measured on the combined tree (all four op_* forks live, PLOW_NV_W8A8=0):
 *     PGM90_FORK_GLU=1 : 1526 spills, 29 HGMMA, 128 HMMA, STACK 2144
 *     PGM90_FORK_GLU=0 :  959 spills, 21 HGMMA, 192 HMMA, STACK 1744
 * The GLU fork buys 8 HGMMA for 567 spills (~71 spills/HGMMA) where the rest of the integration
 * buys 21 for ~797 (~38 spills/HGMMA) — i.e. it is half as register-efficient AND has the
 * smallest speedup of the four. Worse, its pressure spills OTHER arms: op_attention.cuh +252 and
 * interp_sm120.cu +261 spill instructions, and those run for every op, not just GEMM_GLU. That is
 * the "megakernel resources are global" hazard, so the marginal fork is off by default.
 *
 * This is a MEASUREMENT-GATED default, not a verdict on the kernel. End-to-end prefill cannot be
 * measured without a checkpoint; set to 1 once a real prefill run shows the 1.15-1.22x on
 * GEMM_GLU outweighs the extra spills, or once TMA + setmaxnreg warp specialization relieves the
 * 255-register ceiling (which is the actual fix — see the design notes). */
#ifndef PGM90_FORK_GLU
#define PGM90_FORK_GLU 0
#endif
#else
#define PGM_ARENA_SM90 0
#define PGM90_FORK_GLU 0
#endif
#define PGM_ARENA_BF16 (PGM_ARENA_PGM > PGM_ARENA_SM90 ? PGM_ARENA_PGM : PGM_ARENA_SM90)

#if defined(PLOW_NV_HOPPER)
#include "op_gemm_sm90.cuh" /* d_gemm_sm90 / d_gemm_w8a8_sm90 — the wgmma bodies dispatched below */
#endif

/* cp.async 16B line copy with a src-size operand: when src_bytes < 16 the remaining bytes are
 * zero-filled by the hardware and NOT read from gmem — this is how out-of-range M/N/K lines are
 * zeroed without a separate memset (pass src_bytes=0 and a safe in-bounds gmem pointer). */
__device__ __forceinline__ void pgm_cp_async_cg16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void pgm_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void pgm_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pgm_ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
/* NON-.trans x2: reads the mma B fragment straight out of a [n][k] smem tile (proven in the probe).
 * lanes 0-7 address n-rows 0-7 at k-half 0, lanes 8-15 the same n-rows at k-half 1 (16 addresses,
 * 2 stacked 8x8 tiles); lanes 16-31 are ignored by x2 but must still form in-bounds addresses. */
__device__ __forceinline__ void pgm_ldmatrix_x2(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void pgm_mma(float (&d)[4], const unsigned (&a)[4],
                                        const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

/* cp.async-stage a [BM][BK] tile of A ([m][k], k contiguous) into Ad. A is read once per N-tile; the
 * a_row0 offset lets lm_head slice its M=1 out of the LAST prompt row. Requires k % 8 == 0 (true for
 * every Gemma projection K: 3840/4096/9216/... are all multiples of 128). */
__device__ __forceinline__ void pgm_stage_a(__nv_bfloat16* Ad, const __nv_bfloat16* __restrict__ A,
                                            int tid, int tm, int kbase, unsigned m, unsigned k,
                                            int a_row0) {
    const int KCH = PGM_BK / 8; /* 4 lines of 8 bf16 per m-row */
    for (int L = tid; L < PGM_BM * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int mm = tm + row, kk = kbase + kk8;
        const bool in = (mm < (int)m) && (kk + 8 <= (int)k);
        const __nv_bfloat16* g = in ? A + (size_t)(a_row0 + mm) * k + kk : A;
        pgm_cp_async_cg16(&Ad[row * PGM_ASTRIDE + kk8], g, in ? 16 : 0);
    }
}
/* cp.async-stage a [BN][BK] tile of B (weight [n][k], row n = output n) into its NATURAL [n][k] smem
 * layout — the mma B fragment is later read with ldmatrix.x2 non-.trans. Templated on BN so
 * GEMM_GLU can stage its own (narrower) N-tile through the same body — see PGM_BN_GLU. */
template <int BN>
__device__ __forceinline__ void pgm_stage_b_t(__nv_bfloat16* Bd, const __nv_bfloat16* __restrict__ B,
                                            int tid, int tn, int kbase, unsigned n, unsigned k,
                                            int kend) {
    const int KCH = PGM_BK / 8;
    for (int L = tid; L < BN * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        const int nn = tn + row, kk = kbase + kk8;
        const bool in = (nn < (int)n) && (kk + 8 <= kend);
        const __nv_bfloat16* g = in ? B + (size_t)nn * k + kk : B;
        pgm_cp_async_cg16(&Bd[row * PGM_BKSTRIDE + kk8], g, in ? 16 : 0);
    }
}
__device__ __forceinline__ void pgm_stage_b(__nv_bfloat16* Bd, const __nv_bfloat16* __restrict__ B,
                                            int tid, int tn, int kbase, unsigned n, unsigned k,
                                            int kend) {
    pgm_stage_b_t<PGM_BN>(Bd, B, tid, tn, kbase, n, k, kend);
}
/* Read one warp's B fragments for k-slice kf out of a [n][k] stage buffer. Templated on
 * (WN, NFRAG) for the same reason as pgm_stage_b_t above. */
template <int WN, int NFRAG>
__device__ __forceinline__ void pgm_load_bfrags_t(unsigned (&bf)[NFRAG][2], __nv_bfloat16* Bd,
                                                int wn, int kf, int lane) {
#pragma unroll
    for (int nj = 0; nj < NFRAG; nj++) {
        const int n = wn * WN + nj * 8 + (lane & 7);
        const int kcol = kf + ((lane >> 3) & 1) * 8;
        pgm_ldmatrix_x2(bf[nj], &Bd[n * PGM_BKSTRIDE + kcol]);
    }
}
__device__ __forceinline__ void pgm_load_bfrags(unsigned (&bf)[PGM_NFRAG][2], __nv_bfloat16* Bd,
                                                int wn, int kf, int lane) {
    pgm_load_bfrags_t<PGM_WN, PGM_NFRAG>(bf, Bd, wn, kf, lane);
}
/* Read one warp's A fragments (PGM_MFRAG) for k-slice kf out of a [m][k] stage buffer. */
__device__ __forceinline__ void pgm_load_afrags(unsigned (&af)[PGM_MFRAG][4], __nv_bfloat16* Ad,
                                                int wm, int kf, int lane) {
#pragma unroll
    for (int mi = 0; mi < PGM_MFRAG; mi++) {
        const int arow = wm * PGM_WM + mi * 16 + (lane % 16);
        const int acol = kf + (lane / 16) * 8;
        pgm_ldmatrix_x4(af[mi], &Ad[arow * PGM_ASTRIDE + acol]);
    }
}

/* C[M,N] = A[M,K] . B[N,K]^T, bf16, NO epilogue activation (every plain Gemma projection —
 * q/k/v/o/down/lm_head — is unactivated; the only activated prefill matmul is GEMM_GLU below).
 * a_row0 skips a_row0 rows of A (C row r reads A row a_row0+r). One body for GEMM/GEMM_MED/GEMM_SMALL. */
static __device__ void d_gemm(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                       const __nv_bfloat16* __restrict__ B, unsigned m, unsigned n, unsigned k,
                       unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
#if defined(PLOW_NV_HOPPER)
    d_gemm_sm90(C, A, B, m, n, k, a_row0, slice, nblk, arena);
#else
    __nv_bfloat16* As = arena;                            /* [STAGES][BM][ASTRIDE]  A: [m][k] */
    __nv_bfloat16* Bs = arena + PGM_STAGES * PGM_ABUF;    /* [STAGES][BN][BKSTRIDE] B: [n][k] */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_BN - 1) / PGM_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK - 1) / PGM_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN;

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            pgm_stage_a(As + buf * PGM_ABUF, A, tid, tm, ks * PGM_BK, m, k, (int)a_row0);
            pgm_stage_b(Bs + buf * PGM_BBUF, B, tid, tn, ks * PGM_BK, n, k, (int)k);
        };

        /* Prologue: issue the first STAGES-1 K-tiles, one commit_group each. */
#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit();
            /* Keep at most STAGES-1 groups in flight -> the K-tile for `ks` is landed. */
            pgm_cp_wait<PGM_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            __nv_bfloat16* Bd = Bs + cb * PGM_BBUF;
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2];
                pgm_load_bfrags(bf, Bd, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n)
                        C[(size_t)rr * n + cc] = __float2bfloat16(acc[mi][nj][e]);
                }
            }
        __syncthreads();
    }
#endif /* PLOW_NV_HOPPER */
}

#if defined(PLOW_NV_HOPPER) && PLOW_PACKET_LINEAR_BIAS
static __device__ void d_gemm_bias(__nv_bfloat16* __restrict__ C,
                                   const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   const __nv_bfloat16* __restrict__ bias, unsigned m,
                                   unsigned n, unsigned k, unsigned a_row0, unsigned slice,
                                   unsigned nblk, __nv_bfloat16* arena) {
    d_gemm_sm90_bias(C, A, B, bias, m, n, k, a_row0, slice, nblk, arena);
}
#endif

/* GEMM over gate|up in ONE pass, act(gate)*up in the epilogue — the prefill twin of d_gemv_glu.
 * fu = act(A.Wg^T) * (A.Wu^T). Stages TWO B tiles (Wg, Wu) per K-step, keeps two accumulators,
 * applies the GLU at the store. Single-buffered (correctness-first); the arena carries the second
 * B stream (PGM_ARENA_BF16 sizes for it). i5=act: 0 GeGLU (Gemma), 1 SwiGLU. */
static __device__ void d_gemm_glu(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                           const __nv_bfloat16* __restrict__ Wg, const __nv_bfloat16* __restrict__ Wu,
                           unsigned m, unsigned n, unsigned k, unsigned act, unsigned slice,
                           unsigned nblk, __nv_bfloat16* arena) {
#if PGM90_FORK_GLU
    d_gemm_glu_sm90(C, A, Wg, Wu, m, n, k, act, slice, nblk, arena);
#else
    /* Ring of GLU_STAGES buffers, each holding (A, Bg, Bu) for one K-tile. Bg/Bu use the
     * GLU-specific N-tile (PGM_BN_GLU), independent of the plain body's PGM_BN. */
    __nv_bfloat16* As = arena;                                    /* [GLU_STAGES][BM][ASTRIDE] */
    __nv_bfloat16* Bgs0 = arena + PGM_GLU_STAGES * PGM_ABUF;      /* [GLU_STAGES][BN_GLU][BKSTRIDE] */
    __nv_bfloat16* Bus0 = Bgs0 + PGM_GLU_STAGES * PGM_BBUF_GLU;   /* [GLU_STAGES][BN_GLU][BKSTRIDE] */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_BN_GLU - 1) / PGM_BN_GLU;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK - 1) / PGM_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN_GLU;
        float accg[PGM_MFRAG][PGM_NFRAG_GLU][4], accu[PGM_MFRAG][PGM_NFRAG_GLU][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG_GLU; j++)
                for (int e = 0; e < 4; e++) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a(As + buf * PGM_ABUF, A, tid, tm, ks * PGM_BK, m, k, 0);
            pgm_stage_b_t<PGM_BN_GLU>(Bgs0 + buf * PGM_BBUF_GLU, Wg, tid, tn, ks * PGM_BK, n, k, (int)k);
            pgm_stage_b_t<PGM_BN_GLU>(Bus0 + buf * PGM_BBUF_GLU, Wu, tid, tn, ks * PGM_BK, n, k, (int)k);
        };

#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            __nv_bfloat16* Bgd = Bgs0 + cb * PGM_BBUF_GLU;
            __nv_bfloat16* Bud = Bus0 + cb * PGM_BBUF_GLU;
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bg[PGM_NFRAG_GLU][2], bu[PGM_NFRAG_GLU][2];
                pgm_load_bfrags_t<PGM_WN_GLU, PGM_NFRAG_GLU>(bg, Bgd, wn, kf, lane);
                pgm_load_bfrags_t<PGM_WN_GLU, PGM_NFRAG_GLU>(bu, Bud, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG_GLU; nj++) {
                        pgm_mma(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG_GLU; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN_GLU + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n) {
                        float g = accg[mi][nj][e];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * accu[mi][nj][e]);
                    }
                }
            }
        __syncthreads();
    }
#endif /* PGM90_FORK_GLU */
}

/* ============================ fp8 (w8a16) PREFILL TILED GEMM ==========================
 * T6 lever L2. The prefill GEMM is HBM-bound on the 22 GiB of bf16 weights (T3 measured
 * ~46% of the bf16 mma peak — the weight stream, not the mma, is the wall). Storing the
 * weight as e4m3 (1 byte/elt) HALVES that stream. The chosen recipe is w8a16 — fp8 WEIGHT,
 * bf16 ACTIVATION — implemented as DEQUANT-TO-bf16-IN-SMEM: the fp8 tile is cp.async'd (half
 * the bytes of the bf16 tile), expanded to bf16 in smem, and the EXISTING bf16 mma inner
 * loop (pgm_load_afrags/pgm_load_bfrags/pgm_mma) runs UNCHANGED. Nothing about the mma, the
 * accumulator layout, or the store map changes, so the whole T3/T4 correctness argument
 * carries; only the B stage differs (fp8 load + convert) and the epilogue multiplies the
 * f32 accumulator by the per-output-channel dequant scale[n] (which factors out of the K
 * reduction exactly as in gemv_rows_fp8: C[m][n] = scale[n]·Σ_k A[m][k]·Wfp8[n][k]).
 *
 * The A ring is byte-identical to d_gemm's. The B ring stages e4m3 bytes (PGM_BN×PGM_BK per
 * stage) and a single bf16 convert tile holds the current K-step's dequantized weight. The
 * arena claim is SMALLER than the bf16 GEMM's, so the megakernel smem union is unchanged. */
#define PGM_B8BUF (PGM_BN * PGM_BK)                  /* e4m3 bytes per staged B tile (contiguous [n][k]) */
#define PGM_ARENA_FP8 (PGM_STAGES * PGM_ABUF /*As bf16*/ \
                       + (PGM_STAGES * PGM_B8BUF + 1) / 2 /*fp8 ring, bf16 units*/ \
                       + PGM_BBUF /*one bf16 convert tile*/)
/* fp8 GLU stages two weight streams; one shared A ring, two fp8 rings, two convert tiles. */
#define PGM_ARENA_GLU_FP8 (PGM_GLU_STAGES * PGM_ABUF \
                           + (PGM_GLU_STAGES * 2 * PGM_B8BUF + 1) / 2 \
                           + 2 * PGM_BBUF)
static_assert(PGM_ARENA_FP8 <= PGM_ARENA_BF16, "fp8 GEMM arena must fit the bf16 GEMM claim");
static_assert(PGM_ARENA_GLU_FP8 <= PGM_ARENA_BF16, "fp8 GLU arena must fit the bf16 GEMM claim");

/* cp.async-stage a [BN][BK] e4m3 tile of B (weight [n][k], row n = output n) into a contiguous
 * [n][k] fp8 staging buffer. BK is a multiple of 16 (32), so each n-row is BK/16 lines of 16B. */
__device__ __forceinline__ void pgm_stage_b_fp8(uint8_t* Bd8, const uint8_t* __restrict__ B,
                                                int tid, int tn, int kbase, unsigned n, unsigned k,
                                                int kend) {
    const int LCH = PGM_BK / 16; /* 16 e4m3 per cp.async line */
    for (int L = tid; L < PGM_BN * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const int nn = tn + row, kk = kbase + kk16;
        const bool in = (nn < (int)n) && (kk + 16 <= kend);
        const uint8_t* g = in ? B + (size_t)nn * k + kk : B;
        pgm_cp_async_cg16(&Bd8[row * PGM_BK + kk16], g, in ? 16 : 0);
    }
}
/* Dequant a landed [BN][BK] e4m3 staging tile into the bf16 [n][k] convert tile the mma reads
 * (BKSTRIDE-padded, exactly d_gemm's B smem layout). No scale here — it factors into the epilogue. */
__device__ __forceinline__ void pgm_convert_b_fp8(__nv_bfloat16* Bbf, const uint8_t* Bd8, int tid) {
    const int KCH = PGM_BK / 8; /* 8 e4m3 per thread-line */
    for (int L = tid; L < PGM_BN * KCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / KCH, kk8 = (L % KCH) * 8;
        uint2 w8 = *(const uint2*)(Bd8 + row * PGM_BK + kk8);
        const uint16_t* wp = (const uint16_t*)&w8;
        __nv_bfloat16* dst = &Bbf[row * PGM_BKSTRIDE + kk8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
            float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
            dst[2 * j] = __float2bfloat16(f.x);
            dst[2 * j + 1] = __float2bfloat16(f.y);
        }
    }
}

/* w8a16 twin of d_gemm. A bf16 (t1), B e4m3 (t2) with per-output-channel scale (t4). */
static __device__ void d_gemm_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                       const uint8_t* __restrict__ B, const float* __restrict__ scale, unsigned m,
                       unsigned n, unsigned k, unsigned a_row0, unsigned slice, unsigned nblk,
                       __nv_bfloat16* arena) {
    __nv_bfloat16* As = arena;                                 /* [STAGES][BM][ASTRIDE] bf16 */
    uint8_t* Bs8 = (uint8_t*)(As + PGM_STAGES * PGM_ABUF);     /* [STAGES][BN][BK] e4m3 ring */
    __nv_bfloat16* Bbf = (__nv_bfloat16*)(((size_t)(Bs8 + PGM_STAGES * PGM_B8BUF) + 15) & ~(size_t)15);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_BN - 1) / PGM_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK - 1) / PGM_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN;

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            pgm_stage_a(As + buf * PGM_ABUF, A, tid, tm, ks * PGM_BK, m, k, (int)a_row0);
            pgm_stage_b_fp8(Bs8 + buf * PGM_B8BUF, B, tid, tn, ks * PGM_BK, n, k, (int)k);
        };

#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            pgm_convert_b_fp8(Bbf, Bs8 + cb * PGM_B8BUF, tid);
            __syncthreads();
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2];
                pgm_load_bfrags(bf, Bbf, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n)
                        C[(size_t)rr * n + cc] = __float2bfloat16(acc[mi][nj][e] * scale[cc]);
                }
            }
        __syncthreads();
    }
}

/* w8a16 twin of d_gemm_glu. A bf16 (t1), Wg/Wu e4m3 (t2/t5) with per-channel scales sg/su (t4/t6). */
static __device__ void d_gemm_glu_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                       const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                       const float* __restrict__ sg, const float* __restrict__ su, unsigned m,
                       unsigned n, unsigned k, unsigned act, unsigned slice, unsigned nblk,
                       __nv_bfloat16* arena) {
    __nv_bfloat16* As = arena;                                      /* [GLU_STAGES][BM][ASTRIDE] */
    uint8_t* Bg8 = (uint8_t*)(As + PGM_GLU_STAGES * PGM_ABUF);      /* [GLU_STAGES][BN][BK] e4m3 */
    uint8_t* Bu8 = Bg8 + PGM_GLU_STAGES * PGM_B8BUF;                /* [GLU_STAGES][BN][BK] e4m3 */
    __nv_bfloat16* Bgbf = (__nv_bfloat16*)(((size_t)(Bu8 + PGM_GLU_STAGES * PGM_B8BUF) + 15) & ~(size_t)15);
    __nv_bfloat16* Bubf = Bgbf + PGM_BBUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_BN - 1) / PGM_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK - 1) / PGM_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN;
        float accg[PGM_MFRAG][PGM_NFRAG][4], accu[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int e = 0; e < 4; e++) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a(As + buf * PGM_ABUF, A, tid, tm, ks * PGM_BK, m, k, 0);
            pgm_stage_b_fp8(Bg8 + buf * PGM_B8BUF, Wg, tid, tn, ks * PGM_BK, n, k, (int)k);
            pgm_stage_b_fp8(Bu8 + buf * PGM_B8BUF, Wu, tid, tn, ks * PGM_BK, n, k, (int)k);
        };

#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
            __nv_bfloat16* Ad = As + cb * PGM_ABUF;
            pgm_convert_b_fp8(Bgbf, Bg8 + cb * PGM_B8BUF, tid);
            pgm_convert_b_fp8(Bubf, Bu8 + cb * PGM_B8BUF, tid);
            __syncthreads();
#pragma unroll
            for (int kf = 0; kf < PGM_BK; kf += 16) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags(af, Ad, wm, kf, lane);
                unsigned bg[PGM_NFRAG][2], bu[PGM_NFRAG][2];
                pgm_load_bfrags(bg, Bgbf, wn, kf, lane);
                pgm_load_bfrags(bu, Bubf, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++) {
                        pgm_mma(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n) {
                        float g = accg[mi][nj][e] * sg[cc];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * (accu[mi][nj][e] * su[cc]));
                    }
                }
            }
        __syncthreads();
    }
}

/* ============================ fp8 (w8a8) PREFILL TILED GEMM — rtx-07 T7 L2 ==========================
 * The COMPUTE-bound fix T6 identified: w8a16 (§L2) halves the DRAM weight stream but the prefill GEMM
 * is COMPUTE-bound at large M (each weight reused across many output rows), so it measured NEGATIVE.
 * w8a8 attacks the compute wall directly with mma.sync.m16n8k32.e4m3 — 2x the bf16 tensor throughput
 * (rtx-05: fp8 peak 503.8 vs bf16 209.5 TFLOP/s) AND one k32 mma per BK=32 tile vs the bf16 path's
 * two k16 mmas. BOTH operands are e4m3: the weight is per-output-channel e4m3 (T6's twins, unchanged),
 * the ACTIVATION is per-M-row e4m3 with an f32 row-scale from d_quant_fp8 below. The epilogue dequant
 * multiplies acc by a_scale[m]*w_scale[n] (both factor out of the K reduction exactly as w8a16's did).
 *
 * FRAGMENT LAYOUT: empirically DERIVED + bit-exact verified in experiments/fp8_verify.cu. Critically
 * the m16n8k32 f32 ACCUMULATOR layout is IDENTICAL to the bf16 m16n8k16 accumulator, so the d_gemm
 * epilogue store map (row=(L>>2)+8*(e>>1), col=2*(L&3)+(e&1)) carries over UNCHANGED. Operands load
 * with PLAIN uint32 smem reads (NOT ldmatrix — 8-bit has no ldmatrix): A lane L holds rows (L>>2) and
 * (L>>2)+8 with k = 8*(L&3)..+7 (4 u32); B lane L holds col (L>>2) with k = 8*(L&3)..+7 (2 u32).
 *
 * PX-2 NATIVE FULL-TILE MAINLOOP (rtx-11 PX-2). T8 shipped BK=32 — the bf16 K-step — so the fp8
 * mma ran at bf16 CADENCE: one k32 mma per K-tile vs bf16's two k16, i.e. HALF the compute per
 * cp.async stage + __syncthreads, so the pipeline/sync overhead per K element was doubled and the
 * 2x tensor cores measured only -28% GEMM cyc. PX-2 deepens the fp8 K-tile to BK8=64 (fp8 packs 2x
 * per 128-bit cp.async line, so a 64-deep fp8 tile is the SAME smem bytes as bf16's 32-deep tile):
 * two k32 mmas per K-tile, HALF the __syncthreads / commit / wait_group per K element — the fp8
 * mma now runs at full cadence. acc[MFRAG][NFRAG][4] = 64 f32/thread and the wave grid are unchanged.
 *
 * fp8-NATIVE SWIZZLE (rtx-03): the frag read is a PLAIN uint32 (fp8 has no ldmatrix); at stride BK8
 * an unswizzled [m][k]/[n][k] tile puts all 8 warp rows on the SAME 8 banks (4-way conflict). pgm_sw8
 * XORs the 16-byte-line slot index with the low row bits (a CuTe Swizzle over the fp8 K-major tile)
 * so successive rows land on disjoint banks. It is applied IDENTICALLY on the cp.async store and the
 * frag read, so it is a pure address bijection: the mma sees the same e4m3 bytes in the same lanes
 * and the oracle is bit-unchanged. The tiles are BYTES so the arena still fits under the bf16 claim. */
#define PGM_BK8 64                    /* fp8 K-tile depth (2x bf16's BK; same smem bytes/stage) */
#define PGM_A8BUF8 (PGM_BM * PGM_BK8) /* e4m3 bytes per staged w8a8 A tile ([m][k], stride BK8) */
#define PGM_B8BUF8 (PGM_BN * PGM_BK8) /* e4m3 bytes per staged w8a8 B tile ([n][k], stride BK8) */
#define PGM_A8BUF (PGM_BM * PGM_BK)   /* (retained: legacy BK=32 A tile size, unused post-PX-2) */
/* PX-13: the N-tile is PER-OPCODE for the w8a8 pair. PX-9 §5 swept PGM_BN globally and found it
 * pulls in OPPOSITE directions on the two bodies: BN=64 is +9.6% on GEMM_GLU_FP8 and -10% on the
 * plain projections. GEMM_GLU_FP8 holds TWO accumulator sets, so at BN=128 (NFRAG 8) it costs 128
 * f32/thread and is register-limited to 1 block/SM; BN=64 (NFRAG 4) puts it at 64 f32/thread —
 * the same as the plain body — at the price of re-reading A twice as often. The plain body has
 * only one accumulator set, is not register-limited, and only loses the A reuse. So the GLU arm
 * gets its own tile width.
 *
 * DEFAULT 128 — i.e. the knob is implemented but NOT taken, because PX-13 measured it end to end
 * and it LOSES. Isolated it is worth +9.3% on gate|up at M=8192 and +6.1% at M=1024 (the largest
 * prefill bucket the runtime actually launches), bit-exact, at unchanged occupancy. On the real
 * 127k prefill it is 33.21 s vs 32.46 s — a reproducible 2.3% REGRESSION, 25x outside the +-0.03 s
 * run-to-run band. The microbench and the runtime disagree in SIGN, so the microbench does not get
 * to pick this constant. -DPGM_GLU_BN=64 re-enables the arm for anyone re-testing it. */
#ifndef PGM_GLU_BN
#define PGM_GLU_BN 128
#endif
#define PGM_GLU_WN (PGM_GLU_BN / PGM_WARPS_N)
#define PGM_GLU_NFRAG (PGM_GLU_WN / 8)
#define PGM_GLU_B8BUF8 (PGM_GLU_BN * PGM_BK8)
#define PGM_ARENA_W8A8 (((PGM_STAGES * (PGM_A8BUF8 + PGM_B8BUF8)) + 1) / 2)          /* bf16 units */
#define PGM_ARENA_GLU_W8A8 (((PGM_GLU_STAGES * (PGM_A8BUF8 + 2 * PGM_GLU_B8BUF8)) + 1) / 2)
static_assert(PGM_ARENA_W8A8 <= PGM_ARENA_BF16, "w8a8 GEMM arena must fit the bf16 GEMM claim");
static_assert(PGM_ARENA_GLU_W8A8 <= PGM_ARENA_BF16, "w8a8 GLU arena must fit the bf16 GEMM claim");
static_assert(PGM_GLU_BN % (8 * PGM_WARPS_N) == 0, "GLU N-tile must give whole 8-wide n-fragments");
static_assert(PGM_BK8 == 64, "PX-2 mainloop reads two k32 subgroups per K-tile");

/* fp8 K-major swizzle: XOR the 16-byte-line slot (bits[4,7)) with the low row bits (bits[7,10)) —
 * CuTe Swizzle<3,4,3> over the [row][k] fp8 tile (MBase=4 keeps the 16-elem cp.async line contiguous;
 * BBits=3/SShift=3 spread 8 rows across 8 slots). off is a BYTE offset into the tile; the XOR flips
 * only bits 4-6, so it permutes whole 16-byte lines (cp.async store granularity) and leaves the
 * uint32 interior (bits 0-3) intact. Bijective ⇒ store and read agree ⇒ mma bytes unchanged. */
/* PX-9 V2 SWIZZLE. Swizzle<3,4,3> assumes a 128-BYTE fast dimension, but a BK8=64 fp8 row is only
 * 64 B, so SShift=3 reads bits[7,10) = row bits [1,4) and NEVER SEES row bit 0 — two adjacent rows
 * get the SAME line permutation. Enumerated over the whole (row, lane) space that costs one bank
 * conflict way that the tile does not have to pay:
 *     read granularity   Swizzle<3,4,3> (shipped)   off ^ ((off>>2)&0x30) (V2)
 *     4-byte  (LDS.32)         2-way                      2-way   (structural: only even words)
 *     8-byte  (LDS.64)         2-way                      1-way   CONFLICT-FREE
 * V2 XORs the 2-bit 16-byte-line slot (bits[4,6)) with the low 2 row bits (bits[6,8)), i.e. a
 * Swizzle<2,4,2> matched to the ACTUAL 64-byte row. It permutes lines strictly WITHIN a row, so it
 * is trivially bijective, it still moves only bits >= 4 (whole cp.async 16-byte lines), and it
 * still satisfies sw8(off+4) == sw8(off)+4 so the LDS.64 fragment read stays legal.
 * Store and read use the same map either way, so the mma bytes — and the result — are unchanged.
 * -DPGM_SW8_V2=0 restores the shipped Swizzle<3,4,3>; -DPGM_SW8_OFF disables swizzling entirely. */
#ifndef PGM_SW8_V2
#define PGM_SW8_V2 1
#endif
#ifndef PGM_SW8_OFF
#if PGM_SW8_V2
__device__ __forceinline__ int pgm_sw8(int off) { return off ^ ((off >> 2) & 0x30); }
#else
__device__ __forceinline__ int pgm_sw8(int off) { return off ^ (((off >> 7) & 7) << 4); }
#endif
#else
__device__ __forceinline__ int pgm_sw8(int off) { return off; }   /* A/B: swizzle disabled */
#endif

/* m16n8k32 e4m3 tensor-core mma (2x-rate on sm_120), f32 accumulate. Both operands e4m3, row.col. */
__device__ __forceinline__ void pgm_mma_fp8_k32(float (&d)[4], const unsigned (&a)[4],
                                                const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
/* cp.async-stage a [BM][BK8] e4m3 tile of A ([m][k], k contiguous), a_row0 offset for lm_head M-slice.
 * Stride is BK8 (64, no pad); the swizzle pgm_sw8 permutes the 16-byte lines so the frag reader hits
 * disjoint banks. BK8=64 is 4 cp.async 16B lines per m-row. */
__device__ __forceinline__ void pgm_stage_a8(uint8_t* Ad8, const uint8_t* __restrict__ A, int tid,
                                             int tm, int kbase, unsigned m, unsigned k, int a_row0) {
    const int LCH = PGM_BK8 / 16;
    for (int L = tid; L < PGM_BM * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const int mm = tm + row, kk = kbase + kk16;
        const bool in = (mm < (int)m) && (kk + 16 <= (int)k);
        const uint8_t* g = in ? A + (size_t)(a_row0 + mm) * k + kk : A;
        pgm_cp_async_cg16(&Ad8[pgm_sw8(row * PGM_BK8 + kk16)], g, in ? 16 : 0);
    }
}
/* cp.async-stage a [BN][BK8] e4m3 tile of B (weight [n][k], row n = output n) into a swizzled [n][k]
 * fp8 tile (stride BK8). BK8=64 is 4 cp.async 16B lines per n-row. BN is a template parameter
 * (PX-13) because the plain and GLU w8a8 bodies run different N-tile widths. */
template <int BN>
__device__ __forceinline__ void pgm_stage_b8(uint8_t* Bd8, const uint8_t* __restrict__ B, int tid,
                                             int tn, int kbase, unsigned n, unsigned k, int kend) {
    const int LCH = PGM_BK8 / 16;
    for (int L = tid; L < BN * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const int nn = tn + row, kk = kbase + kk16;
        const bool in = (nn < (int)n) && (kk + 16 <= kend);
        const uint8_t* g = in ? B + (size_t)nn * k + kk : B;
        pgm_cp_async_cg16(&Bd8[pgm_sw8(row * PGM_BK8 + kk16)], g, in ? 16 : 0);
    }
}
/* PX-9: 8-BYTE FRAGMENT READS. The fp8 fragment map gives lane L byte offset 8*(L&3) inside its
 * row, so a 4-byte read touches only EVEN 4-byte words — at most 16 of the 32 banks, i.e. a
 * structural 2-way conflict that no XOR swizzle can remove (verified over the whole lane x row
 * space; pgm_sw8 does lift 4-way -> 2-way, it just cannot reach 1-way). The pair (kb, kb+4) is
 * always 8-byte aligned and never crosses a 16-byte line, so pgm_sw8 — which permutes whole
 * 16-byte lines — satisfies sw8(off+4) == sw8(off)+4 there (checked exhaustively). Reading the
 * pair as ONE uint2 therefore covers both word parities, is conflict-free, and halves both the
 * LDS instruction count and the swizzle address arithmetic (which SASS showed dominating the
 * mainloop: 96 LOP3 + 26 SHF + 23 IADD3 per 32 QMMA).
 * SAME bytes into the SAME lanes -> the mma sees identical operands -> bit-identical accumulate.
 * -DPGM_W8A8_LDS64=0 restores the pre-PX-9 scalar reads for A/B. */
#ifndef PGM_W8A8_LDS64
#define PGM_W8A8_LDS64 1
#endif
/* Read one warp's A m-fragments (4 u32 = 16 e4m3) for k-subgroup kf (0 or 32) from a landed swizzled
 * [m][k] fp8 tile (stride BK8). Each read is a single 16-byte line, so pgm_sw8 is the store's inverse. */
__device__ __forceinline__ void pgm_load_afrags_w8a8(unsigned (&af)[PGM_MFRAG][4],
                                                     const uint8_t* Ad8, int wm, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < PGM_MFRAG; mi++) {
        const int rlo = wm * PGM_WM + mi * 16 + (lane >> 2);
        const int rhi = rlo + 8;
#if PGM_W8A8_LDS64
        const uint2 lo = *(const uint2*)(Ad8 + pgm_sw8(rlo * PGM_BK8 + kb));
        const uint2 hi = *(const uint2*)(Ad8 + pgm_sw8(rhi * PGM_BK8 + kb));
        af[mi][0] = lo.x; af[mi][2] = lo.y;
        af[mi][1] = hi.x; af[mi][3] = hi.y;
#else
        af[mi][0] = *(const unsigned*)(Ad8 + pgm_sw8(rlo * PGM_BK8 + kb));
        af[mi][2] = *(const unsigned*)(Ad8 + pgm_sw8(rlo * PGM_BK8 + kb + 4));
        af[mi][1] = *(const unsigned*)(Ad8 + pgm_sw8(rhi * PGM_BK8 + kb));
        af[mi][3] = *(const unsigned*)(Ad8 + pgm_sw8(rhi * PGM_BK8 + kb + 4));
#endif
    }
}
/* Read one warp's B n-fragments (2 u32 = 8 e4m3) for k-subgroup kf (0 or 32) from a landed swizzled
 * [n][k] fp8 tile (stride BK8). WN/NFRAG are template parameters (PX-13, per-opcode N-tile). */
template <int WN, int NFRAG>
__device__ __forceinline__ void pgm_load_bfrags_w8a8(unsigned (&bf)[NFRAG][2],
                                                     const uint8_t* Bd8, int wn, int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int nj = 0; nj < NFRAG; nj++) {
        const int col = wn * WN + nj * 8 + (lane >> 2);
#if PGM_W8A8_LDS64
        const uint2 v = *(const uint2*)(Bd8 + pgm_sw8(col * PGM_BK8 + kb));
        bf[nj][0] = v.x; bf[nj][1] = v.y;
#else
        bf[nj][0] = *(const unsigned*)(Bd8 + pgm_sw8(col * PGM_BK8 + kb));
        bf[nj][1] = *(const unsigned*)(Bd8 + pgm_sw8(col * PGM_BK8 + kb + 4));
#endif
    }
}

/* w8a8 twin of d_gemm. A e4m3 [m][k] (t1) + a_scale f32[M] (t3), B e4m3 [n][k] (t2) + w_scale f32[N]
 * (t4). One k32 mma per BK tile; epilogue dequant acc*a_scale[m]*w_scale[n]. Store map == bf16 d_gemm. */

#if defined(PLOW_NV_HOPPER) && defined(PLOW_NV_FP8_M1) && PLOW_NV_FP8_M1

#ifndef PLOW_NV_FP8_M1_BLOCKED
#define PLOW_NV_FP8_M1_BLOCKED 0
#endif
#ifndef PLOW_NV_FP8_M1_XCACHE
#define PLOW_NV_FP8_M1_XCACHE 0
#endif
#if PLOW_NV_FP8_M1_XCACHE && (!defined(PLOW_NV_FP8_M1_BK1024) || !PLOW_NV_FP8_M1_BK1024 || PLOW_NV_FP8_M1_BLOCKED)
#error "FP8 M1 activation cache requires BK1024 and cyclic ownership"
#endif
#if PLOW_NV_FP8_M1_BLOCKED && defined(PLOW_NV_FP8_M1_PIPE) && PLOW_NV_FP8_M1_PIPE
#error "FP8 M1 blocked ownership excludes PIPE"
#endif
#ifndef PLOW_NV_FP8_M1_BK256
#define PLOW_NV_FP8_M1_BK256 0
#endif
#ifndef PLOW_NV_FP8_M1_BK512
#define PLOW_NV_FP8_M1_BK512 0
#endif
#ifndef PLOW_NV_FP8_M1_BK1024
#define PLOW_NV_FP8_M1_BK1024 0
#endif
#if (PLOW_NV_FP8_M1_BK256 + PLOW_NV_FP8_M1_BK512 + PLOW_NV_FP8_M1_BK1024) > 1
#error "FP8 M1 larger K tiles are separate comparison candidates"
#endif
#if (PLOW_NV_FP8_M1_BK256 || PLOW_NV_FP8_M1_BK512 || PLOW_NV_FP8_M1_BK1024) && (!defined(PLOW_NV_FP8_M1_FAST_ACCUM) || !PLOW_NV_FP8_M1_FAST_ACCUM || (defined(PLOW_NV_FP8_M1_PIPE) && PLOW_NV_FP8_M1_PIPE))
#error "FP8 M1 larger K tiles require FAST_ACCUM and exclude PIPE"
#endif
#if defined(PLOW_NV_FP8_M1_PIPE) && PLOW_NV_FP8_M1_PIPE && defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
#error "FP8 M1 PIPE and FAST_ACCUM are separate comparison candidates"
#endif
#if defined(PLOW_NV_FP8_M1_PIPE) && PLOW_NV_FP8_M1_PIPE
static __device__ __forceinline__ void pgm_m1_stage64(uint8_t* dst, const uint8_t* src,
    unsigned rows, unsigned row0, unsigned kb, unsigned n, unsigned k) {
    for (unsigned line = threadIdx.x; line < rows * 4; line += blockDim.x) {
        const unsigned row = line / 4, col = (line % 4) * 16;
        const unsigned linear = row * 64 + col;
        const unsigned gr = row0 + row, gk = kb + col;
        const bool valid = gr < n && gk < k;
        const unsigned bytes = valid ? min(16u, k - gk) : 0;
        sm90_cp16(dst + (linear ^ ((linear >> 3) & 0x30)),
            valid ? src + (size_t)gr * k + gk : src, bytes);
    }
}
static __device__ __forceinline__ uint64_t pgm_m1_desc64(const void* ptr) {
    return (sm90_make_desc(ptr, 16, 512) & ~(3ull << 62)) | (2ull << 62);
}
/* Two 4608-byte stages. SW64 offsets match CUTLASS Layout_K_SW64_Atom.
 * Promote after each 128 K elements, preserving the single-stage arithmetic. */
static __device__ void d_gemm_w8a8_m1_pipe_sm90(__nv_bfloat16* C, const uint8_t* A,
    const uint8_t* W, const float* ascale, const float* wscale, unsigned n, unsigned k,
    unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    const unsigned tid = threadIdx.x;
    uint8_t* base = (uint8_t*)sm90_align1024(arena);
    const uint8_t* x = A + (size_t)a_row0 * k;
    const unsigned steps = (k + 63) / 64;
    for (unsigned tile = slice; tile < (n + 63) / 64; tile += nblk) {
        float total[4] = {}, partial[4] = {};
        pgm_m1_stage64(base, W, 64, tile * 64, 0, n, k);
        pgm_m1_stage64(base + 4096, x, 8, 0, 0, 1, k);
        sm90_cp_commit();
        for (unsigned step = 0; step < steps; ++step) {
            if (step + 1 < steps) {
                uint8_t* next = base + ((step + 1) & 1) * 4608;
                pgm_m1_stage64(next, W, 64, tile * 64, (step + 1) * 64, n, k);
                pgm_m1_stage64(next + 4096, x, 8, 0, (step + 1) * 64, 1, k);
            }
            sm90_cp_commit();
            sm90_cp_wait<1>();
            __syncthreads();
            if (tid < 128) {
                uint8_t* current = base + (step & 1) * 4608;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const uint64_t dw = pgm_m1_desc64(current + sub * 32);
                    const uint64_t dx = pgm_m1_desc64(current + 4096 + sub * 32);
                    asm volatile(
                        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
                        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
                        "{%0,%1,%2,%3}, %4, %5, p, 1, 1; }\n"
                        : "+f"(partial[0]), "+f"(partial[1]), "+f"(partial[2]), "+f"(partial[3])
                        : "l"(dw), "l"(dx), "r"((int)((step & 1) || sub)));
                }
                sm90_wg_commit();
                sm90_wg_wait<0>();
                if ((step & 1) || step + 1 == steps) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) total[i] += partial[i];
                }
            }
            __syncthreads();
        }
        sm90_cp_wait<0>();
        if (tid < 128 && (tid & 3) == 0) {
            const unsigned r0 = tile * 64 + (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const unsigned row = r0 + hi * 8;
                if (row < n)
                    C[row] = __float2bfloat16(total[hi * 2] * ascale[0] * wscale[row]);
            }
        }
        __syncthreads();
    }
}
#endif

/* W[64,K] x X[1,K]^T uses native m64n8 FP8 (the other seven columns are zero).
 * The fragment matches CUTLASS MMA_64x8x32_F32E4M3E4M3_SS_TN. One 256-thread CTA
 * needs 9216 bytes plus at most 1023 bytes of alignment padding. */
static __device__ void d_gemm_w8a8_m1_sm90(__nv_bfloat16* C, const uint8_t* A,
    const uint8_t* W, const float* ascale, const float* wscale, unsigned n, unsigned k,
    unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    const unsigned tid = threadIdx.x;
    uint8_t* weights = (uint8_t*)sm90_align1024(arena);
    constexpr unsigned panels = PLOW_NV_FP8_M1_BK1024 ? 8 : PLOW_NV_FP8_M1_BK512 ? 4 : PLOW_NV_FP8_M1_BK256 ? 2 : 1;
#if PLOW_NV_FP8_M1_XCACHE
    const bool cached = k <= 17408;
    const unsigned panel_bytes = cached ? 64 * 128 : (64 + 8) * 128;
    uint8_t* cached_x = weights + 8 * 64 * 128;
#else
    constexpr bool cached = false;
    constexpr unsigned panel_bytes = (64 + 8) * 128;
#endif
    const uint8_t* x = A + (size_t)a_row0 * k;
#if PLOW_NV_FP8_M1_BLOCKED
    const bool blocked = n >= 1024;
    const unsigned rows_per_slice = (n + nblk - 1) / nblk;
    const unsigned first = blocked ? slice * rows_per_slice : slice * 64;
    const unsigned end = blocked ? min(first + rows_per_slice, n) : n;
    const unsigned stride = blocked ? 64 : nblk * 64;
#else
    const unsigned first = slice * 64, end = n, stride = nblk * 64;
#endif
#if PLOW_NV_FP8_M1_XCACHE
    if (cached && first < end) {
        const unsigned chunks = ((k + 127) / 128) * 64;
        for (unsigned batch = 0; batch < chunks; batch += 8 * PLOW_NV_THREADS) {
#pragma unroll
            for (unsigned copy = 0; copy < 8; ++copy) {
                const unsigned line = batch + copy * PLOW_NV_THREADS + tid;
                if (line < chunks) {
                    const unsigned panel = line / 64, row = (line % 64) / 8, col = line % 8;
                    const unsigned gk = panel * 128 + col * 16;
                    const bool valid = row == 0 && gk < k;
                    sm90_cp16(cached_x + panel * 1024 + sm90_swz_off<128, 16>(row, col),
                        valid ? x + gk : x, valid ? min(16u, k - gk) : 0);
                }
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
        }
        __syncthreads();
    }
#endif
    for (unsigned row0 = first; row0 < end; row0 += stride) {
        float total[4] = {}, partial[4] = {};
        for (unsigned kb = 0; kb < k; kb += panels * 128) {
#pragma unroll
            for (unsigned panel = 0; panel < panels; ++panel) {
                uint8_t* buffer = weights + panel * panel_bytes;
                pgm90_stage_fp8(buffer, W, tid, 64, row0, kb + panel * 128, end, k);
                if (!cached) pgm90_stage_fp8(buffer + 64 * 128, x, tid, 8, 0, kb + panel * 128, 1, k);
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
            __syncthreads();
            if (tid < 128) {
                sm90_wg_fence();
#pragma unroll
                for (unsigned sub = 0; sub < panels * 4; ++sub) {
                    if (kb + (sub / 4) * 128 >= k) break;
                    uint8_t* buffer = weights + (sub / 4) * panel_bytes;
                    const uint64_t dw = sm90_desc(buffer + (sub % 4) * 32);
#if PLOW_NV_FP8_M1_XCACHE
                    const uint8_t* xp = cached ? cached_x + (kb / 128 + sub / 4) * 1024 : buffer + 64 * 128;
#else
                    const uint8_t* xp = buffer + 64 * 128;
#endif
                    const uint64_t dx = sm90_desc(xp + (sub % 4) * 32);
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    const int accumulate = kb != 0 || sub != 0;
#else
                    const int accumulate = sub != 0;
#endif
                    asm volatile(
                        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
                        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
                        "{%0,%1,%2,%3}, %4, %5, p, 1, 1; }\n"
                        : "+f"(partial[0]), "+f"(partial[1]), "+f"(partial[2]), "+f"(partial[3])
                        : "l"(dw), "l"(dx), "r"(accumulate));
                }
                sm90_wg_commit();
                sm90_wg_wait<0>();
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    total[i] = partial[i];
#else
                    total[i] += partial[i];
#endif
                }
            }
            __syncthreads();
        }
        if (tid < 128 && (tid & 3) == 0) {
            const unsigned r0 = row0 + (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const unsigned row = r0 + hi * 8;
                if (row < end) {
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    /* vLLM M<=16 swaps both operands and epilogue scales. */
                    C[row] = __float2bfloat16(__fmul_rn(wscale[row],
                        __fmul_rn(ascale[0], total[hi * 2])));
#else
                    C[row] = __float2bfloat16(total[hi * 2] * ascale[0] * wscale[row]);
#endif
                }
            }
        }
        __syncthreads();
    }
}
#ifndef PLOW_NV_FP8_M1_TMA
#define PLOW_NV_FP8_M1_TMA 0
#endif
#if PLOW_NV_FP8_M1_TMA
#if !PLOW_NV_FP8_M1_XCACHE || !PLOW_NV_TMA_GEMM || !PLOW_NV_FP8_M1_FAST_ACCUM
#error "FP8 M1 TMA requires XCACHE, FAST_ACCUM and TMA helpers"
#endif
#ifndef PLOW_NV_FP8_M1_PROMOTE_K512
#define PLOW_NV_FP8_M1_PROMOTE_K512 0
#endif
#if PLOW_NV_FP8_M1_PROMOTE_K512 != 0 && PLOW_NV_FP8_M1_PROMOTE_K512 != 1
#error "PLOW_NV_FP8_M1_PROMOTE_K512 must be0 or1"
#endif
static __device__ void d_gemm_w8a8_m1_tma_sm90(__nv_bfloat16* C, const uint8_t* A,
    const uint8_t* W, const void* mapW, const float* ascale, const float* wscale, unsigned n, unsigned k,
    unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    const unsigned tid = threadIdx.x;
    if (!mapW || k > 17408) {
        d_gemm_w8a8_m1_sm90(C, A, W, ascale, wscale, n, k, a_row0, slice, nblk, arena);
        return;
    }
    // Two barriers precede the aligned 64 KiB ring and 139264-byte activation cache.
    uint64_t* ready = (uint64_t*)arena;
    uint8_t* weights = (uint8_t*)sm90_align1024(arena + 8);
    constexpr unsigned panels = 4;
    constexpr bool cached = true;
    constexpr unsigned panel_bytes = 8192;
    uint8_t* cached_x = weights + 65536;
    static_assert(16 + 1023 + 65536 + 139264 <= 205840, "FP8 M1 TMA arena");
    static_assert(205840 + 2448 <= 232448, "FP8 M1 TMA shared-memory limit");
    if (tid < 2) {
        sm90_mbar_init(ready + tid, 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    unsigned stage = 0;
    const uint8_t* x = A + (size_t)a_row0 * k;
    const unsigned first = slice * 64, end = n, stride = nblk * 64;
    if (cached && first < end) {
        const unsigned chunks = ((k + 127) / 128) * 64;
        for (unsigned batch = 0; batch < chunks; batch += 8 * PLOW_NV_THREADS) {
#pragma unroll
            for (unsigned copy = 0; copy < 8; ++copy) {
                const unsigned line = batch + copy * PLOW_NV_THREADS + tid;
                if (line < chunks) {
                    const unsigned panel = line / 64, row = (line % 64) / 8, col = line % 8;
                    const unsigned gk = panel * 128 + col * 16;
                    const bool valid = row == 0 && gk < k;
                    sm90_cp16(cached_x + panel * 1024 + sm90_swz_off<128, 16>(row, col),
                        valid ? x + gk : x, valid ? min(16u, k - gk) : 0);
                }
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
        }
        __syncthreads();
    }
    for (unsigned row0 = first; row0 < end; row0 += stride) {
        float total[4] = {}, partial[4] = {};
        const unsigned steps = (k + 511) / 512;
        auto issue = [&](unsigned serial, unsigned kb) {
            const unsigned slot = serial & 1;
            sm90_mbar_expect(ready + slot, 32768);
            for (unsigned panel = 0; panel < 4; ++panel)
                sm90_tma2d(sm90_su32(weights + slot * 32768 + panel * 8192),
                    mapW, kb + panel * 128, row0, sm90_su32(ready + slot));
        };
        if (tid == 0) {
            issue(stage, 0);
            if (steps > 1) issue(stage + 1, 512);
        }
        for (unsigned step = 0, kb = 0; step < steps; ++step, kb += 512, ++stage) {
            const unsigned slot = stage & 1;
            sm90_mbar_wait(ready + slot, (stage / 2) & 1);
            if (tid < 128) {
                sm90_wg_fence();
#pragma unroll
                for (unsigned sub = 0; sub < panels * 4; ++sub) {
                    if (kb + (sub / 4) * 128 >= k) break;
                    uint8_t* buffer = weights + slot * 32768 + (sub / 4) * panel_bytes;
                    const uint64_t dw = sm90_desc(buffer + (sub % 4) * 32);
                    const uint8_t* xp = cached_x + (kb / 128 + sub / 4) * 1024;
                    const uint64_t dx = sm90_desc(xp + (sub % 4) * 32);
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    const int accumulate = PLOW_NV_FP8_M1_PROMOTE_K512 ? (sub != 0) : (kb != 0 || sub != 0);
#else
                    const int accumulate = sub != 0;
#endif
                    asm volatile(
                        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
                        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
                        "{%0,%1,%2,%3}, %4, %5, p, 1, 1; }\n"
                        : "+f"(partial[0]), "+f"(partial[1]), "+f"(partial[2]), "+f"(partial[3])
                        : "l"(dw), "l"(dx), "r"(accumulate));
                }
                sm90_wg_commit();
                sm90_wg_wait<0>();
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    if (PLOW_NV_FP8_M1_PROMOTE_K512) total[i] += partial[i];
                    else total[i] = partial[i];
#else
                    total[i] += partial[i];
#endif
                }
            }
            __syncthreads();
            if (tid == 0 && step + 2 < steps) issue(stage + 2, kb + 1024);
        }
        if (tid < 128 && (tid & 3) == 0) {
            const unsigned r0 = row0 + (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const unsigned row = r0 + hi * 8;
                if (row < end) {
#if defined(PLOW_NV_FP8_M1_FAST_ACCUM) && PLOW_NV_FP8_M1_FAST_ACCUM
                    /* vLLM M<=16 swaps both operands and epilogue scales. */
                    C[row] = __float2bfloat16(__fmul_rn(wscale[row],
                        __fmul_rn(ascale[0], total[hi * 2])));
#else
                    C[row] = __float2bfloat16(total[hi * 2] * ascale[0] * wscale[row]);
#endif
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (tid < 2) sm90_mbar_inval(ready + tid);
    __syncthreads();
}
#endif

#endif

static __device__ void d_gemm_w8a8(__nv_bfloat16* __restrict__ C, const uint8_t* __restrict__ A,
                       const uint8_t* __restrict__ B, const float* __restrict__ ascale,
                       const float* __restrict__ wscale, unsigned m, unsigned n, unsigned k,
                       unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
#if defined(PLOW_NV_HOPPER)
#if defined(PLOW_NV_FP8_M1) && PLOW_NV_FP8_M1
    if (m == 1) {
#if defined(PLOW_NV_FP8_M1_PIPE) && PLOW_NV_FP8_M1_PIPE
        d_gemm_w8a8_m1_pipe_sm90(C, A, B, ascale, wscale, n, k, a_row0, slice, nblk, arena);
#else
        d_gemm_w8a8_m1_sm90(C, A, B, ascale, wscale, n, k, a_row0, slice, nblk, arena);
#endif
        return;
    }
#endif
    d_gemm_w8a8_sm90(C, A, B, ascale, wscale, m, n, k, a_row0, slice, nblk, arena);
#else
    uint8_t* As = (uint8_t*)arena;                       /* [STAGES][BM][BK8] e4m3 */
    uint8_t* Bs = As + PGM_STAGES * PGM_A8BUF8;           /* [STAGES][BN][BK8] e4m3 */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_BN - 1) / PGM_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK8 - 1) / PGM_BK8;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_BN;

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8(As + buf * PGM_A8BUF8, A, tid, tm, ks * PGM_BK8, m, k, (int)a_row0);
            pgm_stage_b8<PGM_BN>(Bs + buf * PGM_B8BUF8, B, tid, tn, ks * PGM_BK8, n, k, (int)k);
        };

#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_STAGES;
            /* Two k32 mmas per BK8 tile: subgroup kf=0 then kf=32 (native fp8 cadence). */
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2];
                pgm_load_bfrags_w8a8<PGM_WN, PGM_NFRAG>(bf, Bs + cb * PGM_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n)
                        C[(size_t)rr * n + cc] =
                            __float2bfloat16(acc[mi][nj][e] * ascale[rr] * wscale[cc]);
                }
            }
        __syncthreads();
    }
#endif /* PLOW_NV_HOPPER */
}

/* w8a8 twin of d_gemm_glu. A e4m3 (t1) + a_scale (t3); Wg/Wu e4m3 (t2/t5) + per-channel sg/su (t4/t6).
 * act(a_scale*sg*gate) * (a_scale*su*up) — the activation row-scale multiplies BOTH streams. */
static __device__ void d_gemm_glu_w8a8(__nv_bfloat16* __restrict__ C, const uint8_t* __restrict__ A,
                       const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                       const float* __restrict__ ascale, const float* __restrict__ sg,
                       const float* __restrict__ su, unsigned m, unsigned n, unsigned k, unsigned act,
                       unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
#if PGM90_FORK_GLU
    d_gemm_glu_w8a8_sm90(C, A, Wg, Wu, ascale, sg, su, m, n, k, act, slice, nblk, arena);
#else
    uint8_t* As = (uint8_t*)arena;                            /* [GLU_STAGES][BM][BK8] e4m3 */
    uint8_t* Bg = As + PGM_GLU_STAGES * PGM_A8BUF8;            /* [GLU_STAGES][GLU_BN][BK8] e4m3 */
    uint8_t* Bu = Bg + PGM_GLU_STAGES * PGM_GLU_B8BUF8;        /* [GLU_STAGES][GLU_BN][BK8] e4m3 */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;
    const int tiles_m = (m + PGM_BM - 1) / PGM_BM;
    const int tiles_n = (n + PGM_GLU_BN - 1) / PGM_GLU_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + PGM_BK8 - 1) / PGM_BK8;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM_BM;
        const int tn = (tile % tiles_n) * PGM_GLU_BN;
        float accg[PGM_MFRAG][PGM_GLU_NFRAG][4], accu[PGM_MFRAG][PGM_GLU_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++)
            for (int j = 0; j < PGM_GLU_NFRAG; j++)
                for (int e = 0; e < 4; e++) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8(As + buf * PGM_A8BUF8, A, tid, tm, ks * PGM_BK8, m, k, 0);
            pgm_stage_b8<PGM_GLU_BN>(Bg + buf * PGM_GLU_B8BUF8, Wg, tid, tn, ks * PGM_BK8, n, k, (int)k);
            pgm_stage_b8<PGM_GLU_BN>(Bu + buf * PGM_GLU_B8BUF8, Wu, tid, tn, ks * PGM_BK8, n, k, (int)k);
        };

#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) {
            if (s < ksteps) { stage(s, s); }
            pgm_cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit();
            pgm_cp_wait<PGM_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
            /* Two k32 mmas per BK8 tile: subgroup kf=0 then kf=32 (native fp8 cadence). */
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4];
                pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bg[PGM_GLU_NFRAG][2], bu[PGM_GLU_NFRAG][2];
                pgm_load_bfrags_w8a8<PGM_GLU_WN, PGM_GLU_NFRAG>(bg, Bg + cb * PGM_GLU_B8BUF8, wn, kf, lane);
                pgm_load_bfrags_w8a8<PGM_GLU_WN, PGM_GLU_NFRAG>(bu, Bu + cb * PGM_GLU_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_GLU_NFRAG; nj++) {
                        pgm_mma_fp8_k32(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma_fp8_k32(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_GLU_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_GLU_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int rr = tm + gr + (e / 2) * 8;
                    int cc = tn + gc + (e % 2);
                    if (rr < (int)m && cc < (int)n) {
                        const float as = ascale[rr];
                        float g = accg[mi][nj][e] * as * sg[cc];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * (accu[mi][nj][e] * as * su[cc]));
                    }
                }
            }
        __syncthreads();
    }
#endif /* PGM90_FORK_GLU */
}

/* Per-row (per-token) fp8 ACTIVATION quant — the w8a8 prefill's activation half (op PLOW_DOP_QUANT_FP8).
 * ONE WARP owns each M-row: the 32 lanes stride K for the row absmax, warp-reduce it, set a_scale =
 * max(absmax/448, 1e-12), then xq = round_e4m3(x/a_scale). e4m3 (fn) has no inf, max 448. Rows are
 * sliced across nblk blocks; a block takes PLOW_NV_WARPS rows at a time. bf16 in, e4m3 out, f32 scale.
 * The device round-to-nearest e4m3 is the SAME rounding the offline quantizer uses, so the oracle's
 * dequantized-f32 reference sees identical activation values (isolates the mma from the quant error). */
/* T11 GLU FUSION (t3=gate, t4=up, i2=act; both null on legacy packets): when `gate` is set,
 * `x` is an OUTPUT — the warp computes fu = act(gate)*up inline (bf16-rounded, exactly what
 * the separate Glu packet would have written), stores it, and quantizes from the rounded
 * value, deleting the Glu packet + its gate + the full inter-width fu re-read per layer. */
static __device__ void d_quant_fp8(uint8_t* __restrict__ xq, __nv_bfloat16* __restrict__ x,
                            float* __restrict__ ascale, unsigned M, unsigned K, unsigned slice,
                            unsigned nblk, const __nv_bfloat16* __restrict__ gate = nullptr,
                            const __nv_bfloat16* __restrict__ up = nullptr, unsigned act = 0) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned per = (M + nblk - 1) / nblk;
    const unsigned m0 = slice * per;
    const unsigned m1 = (m0 + per < M) ? (m0 + per) : M;
    const bool v8 = (K & 7u) == 0; /* every emitted K (hidden/q_dim/inter) is 8-aligned */
    for (unsigned mm = m0 + warp; mm < m1; mm += PLOW_NV_WARPS) {
        const size_t row = (size_t)mm * K;
        float amax = 0.0f;
        if (gate) {
            if (v8) {
                for (unsigned kk = lane * 8u; kk < K; kk += 256u) {
                    const bf16v8 vg = ld_glob8(gate + row + kk), vu = ld_glob8(up + row + kk);
                    bf16v8 vo;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = __bfloat162float(vg.x[j]);
                        const float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        vo.x[j] = __float2bfloat16(a * __bfloat162float(vu.x[j]));
                        amax = fmaxf(amax, fabsf(__bfloat162float(vo.x[j])));
                    }
                    st_glob8(x + row + kk, vo);
                }
            } else {
                for (unsigned kk = lane; kk < K; kk += 32u) {
                    const float g = __bfloat162float(gate[row + kk]);
                    const float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                    const __nv_bfloat16 fb = __float2bfloat16(a * __bfloat162float(up[row + kk]));
                    x[row + kk] = fb;
                    amax = fmaxf(amax, fabsf(__bfloat162float(fb)));
                }
            }
        } else if (v8) {
            for (unsigned kk = lane * 8u; kk < K; kk += 256u) {
                const bf16v8 v = ld_glob8(x + row + kk);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    amax = fmaxf(amax, fabsf(__bfloat162float(v.x[j])));
            }
        } else {
            for (unsigned kk = lane; kk < K; kk += 32u)
                amax = fmaxf(amax, fabsf(__bfloat162float(x[row + kk])));
        }
        amax = warp_max32(amax);
#if defined(PLOW_NV_QUANT_FP8_VLLM) && PLOW_NV_QUANT_FP8_VLLM
        /* vLLM 2cf0a6915 dynamic_per_token_scaled_fp8_quant: true division,
         * including under --use_fast_math; the floor is one E4M3 subnormal / 448. */
        const float as = fmaxf(__fdiv_rn(amax, 448.0f), 1.0f / (448.0f * 512.0f));
#else
        const float as = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
        const float inv = 1.0f / as;
#endif
        if (lane == 0) ascale[mm] = as;
        if (v8) {
            for (unsigned kk = lane * 8u; kk < K; kk += 256u) {
                const bf16v8 v = ld_glob8(x + row + kk);
                uint8_t q8[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
#if defined(PLOW_NV_QUANT_FP8_VLLM) && PLOW_NV_QUANT_FP8_VLLM
                    const float scaled = __fdiv_rn(__bfloat162float(v.x[j]), as);
                    __nv_fp8_e4m3 q(fmaxf(-448.0f, fminf(scaled, 448.0f)));
#else
                    __nv_fp8_e4m3 q(__bfloat162float(v.x[j]) * inv);
#endif
                    q8[j] = *(const uint8_t*)&q;
                }
                *(uint2*)(xq + row + kk) = *(const uint2*)q8;
            }
        } else {
            for (unsigned kk = lane; kk < K; kk += 32u) {
#if defined(PLOW_NV_QUANT_FP8_VLLM) && PLOW_NV_QUANT_FP8_VLLM
                const float scaled = __fdiv_rn(__bfloat162float(x[row + kk]), as);
                __nv_fp8_e4m3 q(fmaxf(-448.0f, fminf(scaled, 448.0f)));
#else
                __nv_fp8_e4m3 q(__bfloat162float(x[row + kk]) * inv);
#endif
                xq[row + kk] = *(const uint8_t*)&q;
            }
        }
    }
}

/* GEMV WITH A FUSED GLU EPILOGUE — gate and up in ONE pass, act(g)*u applied at the write.
 *
 * The DIRECTION is the whole point, and the AMD header's measurement stands: fuse into the
 * PRODUCER's epilogue, never into the consumer's prologue. Here the warp that owns column n
 * holds BOTH gate[n] and up[n], so the GLU runs exactly once per element. Folding it into
 * the down-projection's prologue instead lost 39x, because `fu` is down's K axis — the axis
 * it reduces over — so all of down's blocks stage the whole of it and each recomputes the
 * entire GLU. A producer feeding a consumer's K dimension is replicated by that consumer's
 * block count. */
/* BATCH>1: gate|up weight rows loaded ONCE, dotted against all MM x-rows, GLU per (row,col).
 * C is [M][N]. MM==1 is byte-identical to the old scalar body (the B=1 serving path). */
#ifndef PLOW_NV_GEMMA_GLU_BF16
#define PLOW_NV_GEMMA_GLU_BF16 0
#endif
static __device__ __forceinline__ __nv_bfloat16 gemma_glu_epilogue(float gate, float up,
                                                                 unsigned act) {
#if defined(PLOW_NV_GEMMA) && PLOW_NV_GEMMA && PLOW_NV_GEMMA_GLU_BF16
    if (act != PLOW_ACT_SILU_) {
        gate = __bfloat162float(__float2bfloat16(gate));
        up = __bfloat162float(__float2bfloat16(up));
        const float activated = __bfloat162float(__float2bfloat16(act_gelu_tanh(gate)));
        return __float2bfloat16(activated * up);
    }
#endif
    const float activated = (act == PLOW_ACT_SILU_) ? act_silu(gate) : act_gelu_tanh(gate);
    return __float2bfloat16(activated * up);
}

template <int MM, int UN = gv_un_glu<MM>::v>
__device__ __forceinline__ void gemv_glu_rows(__nv_bfloat16* C, const __nv_bfloat16* x,
                           const __nv_bfloat16* Wg, const __nv_bfloat16* Wu, unsigned M, unsigned N,
                           unsigned K, unsigned act, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* grow = Wg + (size_t)n * K;
        const __nv_bfloat16* urow = Wu + (size_t)n * K;
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 gv[UN], uv[UN];
            unsigned kk[UN];
            /* Both streams issued before either is consumed — 2*UN vectors in flight. */
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                gv[u] = (k < K) ? ld_glob8(grow + k) : bf16v8_zero();
                uv[u] = (k < K) ? ld_glob8(urow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    const bf16v8 xv = ld_glob8(x + (size_t)m * K + kk[u]);
                    ag[m] = dot8(gv[u], xv, ag[m]);
                    au[m] = dot8(uv[u], xv, au[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float tg = warp_sum32(ag[m]), tu = warp_sum32(au[m]);
            if (lane == 0 && (unsigned)m < M) {
                C[(size_t)m * N + n] = gemma_glu_epilogue(tg, tu, act);
            }
        }
    }
}
static __device__ void d_gemv_glu(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ Wg,
                           const __nv_bfloat16* __restrict__ Wu, unsigned M, unsigned N,
                           unsigned K, unsigned act, unsigned slice, unsigned nblk) {
    /* Same unwritten-rows bug as d_gemv_qkv above M=8 before the walk. */
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_glu_rows<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, Wg, Wu, rows, N, K,
                                       act, slice, nblk);
    });
}

/* Arena-aware M=1 decode GLU GEMV: x staged into smem. */
static __device__ void d_gemv_glu(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ Wg,
                           const __nv_bfloat16* __restrict__ Wu, unsigned M, unsigned N,
                           unsigned K, unsigned act, unsigned slice, unsigned nblk,
                           __nv_bfloat16* __restrict__ arena) {
    if (M > 1) { d_gemv_glu(C, x, Wg, Wu, M, N, K, act, slice, nblk); return; }
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const __nv_bfloat16* grow = Wg + (size_t)n * K;
        const __nv_bfloat16* urow = Wu + (size_t)n * K;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_GLU) {
            bf16v8 gv[GV_UNROLL_GLU], uv[GV_UNROLL_GLU];
            unsigned kk[GV_UNROLL_GLU];
#pragma unroll
            for (int u = 0; u < GV_UNROLL_GLU; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                gv[u] = (k < K) ? ld_glob8(grow + k) : bf16v8_zero();
                uv[u] = (k < K) ? ld_glob8(urow + k) : bf16v8_zero();
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL_GLU; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld_smem8(xs + kk[u]);
                ag = dot8(gv[u], xv, ag);
                au = dot8(uv[u], xv, au);
            }
        }
        const float tg = warp_sum32(ag), tu = warp_sum32(au);
        if (lane == 0 && M) {
            C[n] = gemma_glu_epilogue(tg, tu, act);
        }
    }
}

/* ======================= SplitZip (bf16 lossless) DECODE GEMV ==========================
 * p9-v2 C-1. The bf16 weight is stored byte-plane split + 4-bit affine exponent codes:
 *   lo[N*K]      : 1 B/elem = sign(bit7) | mantissa(bits6:0)
 *   cd[N*K/2]    : 4 b/elem = code, exponent = code + exp_base  (per-tensor base from C-0)
 *   eoff[N+1]    : u32 prefix into the per-ROW escape lists
 *   epos[]/eval[]: u32 flat elem index / u16 raw bf16 for out-of-window exponents (<0.03%)
 * Reconstruct on load (~4 int ops/elem, no smem tables): the EXACT bf16 bits are rebuilt, so
 * the FMA order matches gemv_rows and the f32 outputs are BIT-IDENTICAL to the bf16 GEMV.
 * 12 b/elem fixed stride => 1.33x fewer weight bytes cross HBM (the whole thesis). Twin of
 * gemv_rows: one WARP owns one output row n; each lane rebuilds 8 K-elems/chunk then fmas. */
__device__ __forceinline__ bf16v8 sz_expand8(const uint8_t* __restrict__ lo,
                                             const uint8_t* __restrict__ cd, size_t el,
                                             unsigned exp_base) {
    const uint2 lb = *(const uint2*)(lo + el);          /* 8 lo bytes            */
    const unsigned cw = *(const unsigned*)(cd + (el >> 1)); /* 8 exponent nibbles */
    const unsigned lw[2] = {lb.x, lb.y};
    bf16v8 r;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned b = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
        const unsigned ex = ((cw >> (e * 4)) & 0xFu) + exp_base;
        const unsigned short u = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
        r.x[e] = __ushort_as_bfloat16(u);
    }
    return r;
}
/* Overwrite any of the 8 rebuilt elems whose exponent fell outside the window (warp-uniform
 * trip count over the row's escape slice — the body runs ~0.6x per 4096 elems, no divergence
 * in the loop bound). el is the flat index of r.x[0]. */
__device__ __forceinline__ void sz_escape8(bf16v8& r, size_t el, unsigned e0, unsigned e1,
                                           const unsigned* __restrict__ epos,
                                           const __nv_bfloat16* __restrict__ eval) {
    for (unsigned t = e0; t < e1; ++t) {
        const unsigned d = (unsigned)(epos[t] - (unsigned)el);
        if (d < 8u) r.x[d] = eval[t];
    }
}

template <int MM, int UN = gv_un<MM>::v>
__device__ __forceinline__ void gemv_rows_sz(__nv_bfloat16* __restrict__ C,
                                             const __nv_bfloat16* __restrict__ x,
                                             const uint8_t* __restrict__ lo,
                                             const uint8_t* __restrict__ cd,
                                             const unsigned* __restrict__ eoff,
                                             const unsigned* __restrict__ epos,
                                             const __nv_bfloat16* __restrict__ eval,
                                             unsigned exp_base, unsigned M, unsigned N, unsigned K,
                                             unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const size_t rbase = (size_t)n * K;
        const unsigned e0 = eoff[n], e1 = eoff[n + 1];
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
                if (k < K) {
                    const size_t el = rbase + k;
                    wv[u] = sz_expand8(lo, cd, el, exp_base);
                    if (e1 != e0) sz_escape8(wv[u], el, e0, e1, epos, eval);
                } else {
                    wv[u] = bf16v8_zero();
                }
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8(wv[u], ld_glob8(x + (size_t)m * K + kk[u]), acc[m]);
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

/* Fused gate|up SplitZip GEMV — twin of gemv_glu_rows, two compressed weight streams. */
template <int MM, int UN = gv_un_glu<MM>::v>
__device__ __forceinline__ void gemv_glu_rows_sz(
        __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ glo, const uint8_t* __restrict__ gcd,
        const unsigned* __restrict__ geoff, const unsigned* __restrict__ gepos,
        const __nv_bfloat16* __restrict__ geval, const uint8_t* __restrict__ ulo,
        const uint8_t* __restrict__ ucd, const unsigned* __restrict__ ueoff,
        const unsigned* __restrict__ uepos, const __nv_bfloat16* __restrict__ ueval,
        unsigned exp_base, unsigned M, unsigned N, unsigned K, unsigned act, unsigned slice,
        unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const size_t rbase = (size_t)n * K;
        const unsigned ge0 = geoff[n], ge1 = geoff[n + 1], ue0 = ueoff[n], ue1 = ueoff[n + 1];
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 gv[UN], uv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                if (k < K) {
                    const size_t el = rbase + k;
                    gv[u] = sz_expand8(glo, gcd, el, exp_base);
                    if (ge1 != ge0) sz_escape8(gv[u], el, ge0, ge1, gepos, geval);
                    uv[u] = sz_expand8(ulo, ucd, el, exp_base);
                    if (ue1 != ue0) sz_escape8(uv[u], el, ue0, ue1, uepos, ueval);
                } else {
                    gv[u] = bf16v8_zero(); uv[u] = bf16v8_zero();
                }
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    const bf16v8 xv = ld_glob8(x + (size_t)m * K + kk[u]);
                    ag[m] = dot8(gv[u], xv, ag[m]);
                    au[m] = dot8(uv[u], xv, au[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float tg = warp_sum32(ag[m]), tu = warp_sum32(au[m]);
            if (lane == 0 && (unsigned)m < M) {
                const float a = (act == PLOW_ACT_SILU_) ? act_silu(tg) : act_gelu_tanh(tg);
                C[(size_t)m * N + n] = __float2bfloat16(a * tu);
            }
        }
    }
}

/* SplitZip weight BLOB — one SELF-DESCRIBING device tensor per weight. Layout (16-byte-aligned
 * sections): hdr[16] | lo[N*K] | cd[N*K/2] | eoff[(N+1)*4] | epos[maxesc*4] | eval[maxesc*2].
 * hdr = {u32 nesc, u32 exp_base, u32 pad, u32 pad}. The escape sections are sized to a FIXED
 * reservation maxesc = sz_maxesc(N,K) (a function of N,K only) so the byte layout is known at
 * emit time WITHOUT the data-dependent escape count; the loader writes the actual nesc/exp_base
 * into the header and fills [0,nesc) of epos/eval. Every other size is a function of N,K.
 * SZ_ESC_PERMIL is the reserved escape budget (‰); the C-0 audit measured worst-class 0.18‰,
 * so 2‰ is ~10x headroom — a tensor that exceeds it must be emitted raw (emitter opt-out). */
#ifndef SZ_ESC_PERMIL
#define SZ_ESC_PERMIL 2u
#endif
__host__ __device__ __forceinline__ size_t sz_maxesc(unsigned N, unsigned K) {
    return (size_t)N * K * SZ_ESC_PERMIL / 1000u + 64u;
}
__host__ __device__ __forceinline__ size_t sz_blob_bytes(unsigned N, unsigned K) {
    const size_t nk = (size_t)N * K, me = sz_maxesc(N, K);
    size_t o = 16;                                   /* header               */
    o = ((o + nk + 15) & ~(size_t)15);               /* + lo                 */
    o = ((o + nk / 2 + 15) & ~(size_t)15);           /* + cd                 */
    o = ((o + (size_t)(N + 1) * 4 + 15) & ~(size_t)15); /* + eoff            */
    o = ((o + me * 4 + 15) & ~(size_t)15);           /* + epos               */
    o = ((o + me * 2 + 15) & ~(size_t)15);           /* + eval               */
    return o;
}
struct SzBlob {
    const uint8_t* lo; const uint8_t* cd; const unsigned* eoff;
    const unsigned* epos; const __nv_bfloat16* eval; unsigned nesc; unsigned exp_base;
};
__device__ __forceinline__ SzBlob sz_blob(const uint8_t* b, unsigned N, unsigned K) {
    const size_t nk = (size_t)N * K, me = sz_maxesc(N, K);
    const size_t o_lo   = 16;
    const size_t o_cd   = (o_lo + nk + 15) & ~(size_t)15;
    const size_t o_eoff = (o_cd + nk / 2 + 15) & ~(size_t)15;
    const size_t o_epos = (o_eoff + (size_t)(N + 1) * 4 + 15) & ~(size_t)15;
    const size_t o_eval = (o_epos + me * 4 + 15) & ~(size_t)15;
    SzBlob s;
    s.nesc = ((const unsigned*)b)[0];
    s.exp_base = ((const unsigned*)b)[1];
    s.lo = b + o_lo; s.cd = b + o_cd;
    s.eoff = (const unsigned*)(b + o_eoff);
    s.epos = (const unsigned*)(b + o_epos);
    s.eval = (const __nv_bfloat16*)(b + o_eval);
    return s;
}

/* Walk dispatchers (twins of d_gemv / d_gemv_glu). Non-arena — batched decode reads x from
 * global; the M=1 arena-smem variant is an orthogonal optimization not added here (B=1 is the
 * accuracy row, and the batched path is where C-1's multi-user thesis lives). */
static __device__ void d_gemv_sz(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ blob, unsigned M, unsigned N, unsigned K, unsigned slice,
        unsigned nblk) {
    const SzBlob s = sz_blob(blob, N, K);
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_rows_sz<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, s.lo, s.cd, s.eoff,
                                      s.epos, s.eval, s.exp_base, rows, N, K, slice, nblk);
    });
}
static __device__ void d_gemv_glu_sz(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ gblob, const uint8_t* __restrict__ ublob, unsigned M,
        unsigned N, unsigned K, unsigned act, unsigned slice, unsigned nblk) {
    const SzBlob g = sz_blob(gblob, N, K);
    const SzBlob u = sz_blob(ublob, N, K);
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_glu_rows_sz<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, g.lo, g.cd, g.eoff,
                                          g.epos, g.eval, u.lo, u.cd, u.eoff, u.epos, u.eval,
                                          g.exp_base, rows, N, K, act, slice, nblk);
    });
}

/* ============================ fp8 (w8a16) DECODE GEMV ==================================
 * The settled decode recipe (refuting mma.sync/TMA/warp-spec at M=1):
 * FFMA dequant-on-load. The weight is e4m3 (1 byte/elt), HALF the bytes of the bf16 GEMV
 * above and therefore ~2x the bandwidth-bound roofline. The math is IDENTICAL to gemv_rows:
 * one WARP owns one output row n, each lane fmas 8 consecutive K elements per chunk, the warp
 * reduces, and the per-output-channel dequant scale[n] is applied ONCE in the epilogue on the
 * reduced sum (it factors out of the whole K-reduction — runtime/amd/op_gemm.h:1440).
 *
 * COLUMN OWNERSHIP is BLOCKED, byte-for-byte the same map as gemv_rows: the emitter's
 * decode fine-grained gemv->headnorm dependency (gemv_wgs_for_cols) assumes it and the fp8
 * GEMV feeds the same headnorm, so it MUST match. A K-split scheme would break that map;
 * the fill comes from the N-column partition across all 188 blocks, exactly as bf16 decode.
 *
 * The fp8 chunk is 8 fp8/lane = 256 K per warp-pass — the SAME GV_STEP as bf16 (which is 8
 * bf16/lane), so nchunk is identical and every decode K (3840/4096/2048/512/15360, all
 * multiples of 256) divides cleanly; the k<K predicate is uniform across the warp. */

/* One lane's chunk: dequant 8 e4m3 weights (a uint2, 8 bytes) and fma against 8 bf16 x. */
/* Twin of dot8_fp8 that takes x ALREADY WIDENED to f32. The bf16 form spends 8 of its ~28
 * inner-loop instructions per 8 bytes re-widening x with __bfloat162float; fp8 halves the
 * weight bytes but not that cost, which is why the fp8 dense GEMV measured 1046 GB/s -- the
 * slowest arm in either precision, and compute-bound rather than bandwidth-bound. Staging x
 * once as f32 removes those 8. Same FMA order, same values (bf16->f32 is exact). */
__device__ __forceinline__ float dot8_fp8_fx(const uint2& w8, const float* __restrict__ xs,
                                             float acc) {
    const uint16_t* wp = (const uint16_t*)&w8;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        acc = fmaf(f.x, xs[2 * j], acc);
        acc = fmaf(f.y, xs[2 * j + 1], acc);
    }
    return acc;
}

__device__ __forceinline__ float dot8_fp8(const uint2& w8, const bf16v8& x, float acc) {
    const uint16_t* wp = (const uint16_t*)&w8; /* 4 packed fp8x2 pairs */
#pragma unroll
    for (int j = 0; j < 4; j++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        acc = fmaf(f.x, __bfloat162float(x.x[2 * j]), acc);
        acc = fmaf(f.y, __bfloat162float(x.x[2 * j + 1]), acc);
    }
    return acc;
}

/* C[m][n] = scale[n] * dot(x[m][:], dequant(W8[n][:])). W8 is [N, K] e4m3 row-major.
 *
 * BATCH>1 DECODE: exactly the bf16 gemv_rows<MM> structure — each e4m3 weight row is loaded and
 * dequanted ONCE and fma'd against all MM rows of x, so B batched rows cost ~1 weight read
 * instead of B. Before this the fp8 arms were scalar-accumulator (row 0 only), which left rows
 * 1..B UNWRITTEN — fluent wrong text for every slot but the first, hence the compiler's
 * `fp8 && dbatch>1` refusal. MM==1 is byte-identical to that old body (the B=1 serving path).
 * The dequant (dot8_fp8) stays on the FFMA path: at these M it beats mma,
 * and amortising it over MM rows only widens the margin. */
template <int MM, int UN = gv_un_fp8<MM>::v>
__device__ __forceinline__ void gemv_rows_fp8(__nv_bfloat16* __restrict__ C,
                                              const __nv_bfloat16* __restrict__ x,
                                              const uint8_t* __restrict__ W,
                                              const float* __restrict__ scale, unsigned M,
                                              unsigned N, unsigned K, unsigned slice,
                                              unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const uint8_t* wrow = W + (size_t)n * K;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            uint2 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                /* 8 fp8 bytes; the k<K guard is uniform (K % 256 == 0). */
                wv[u] = (k < K) ? *(const uint2*)(wrow + k) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    acc[m] = dot8_fp8(wv[u], ld_glob8(x + (size_t)m * K + kk[u]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = warp_sum32(acc[m]);
            if (lane == 0 && (unsigned)m < M) C[(size_t)m * N + n] = __float2bfloat16(t * scale[n]);
        }
    }
}

/* Non-arena overload: standalone test kernels that don't run in the persistent interpreter.
 * MM ladder {1,2,4,8} + block-walk for M>8, identical shape to d_gemv. */
static __device__ void d_gemv_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                           const uint8_t* __restrict__ W, const float* __restrict__ scale,
                           unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_rows_fp8<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, W, scale, rows, N,
                                       K, slice, nblk);
    });
}

/* Arena-aware overload: persistent interpreter stages x into smem. */
static __device__ void d_gemv_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                           const uint8_t* __restrict__ W, const float* __restrict__ scale,
                           unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk,
                           __nv_bfloat16* __restrict__ arena) {
    /* The arena stages ONE x row; B>1 decode has M rows, so fall back to the batched global
     * path (same rule as bf16 d_gemv). Without this the fp8 arms silently computed row 0 only. */
    if (M > 1) { d_gemv_fp8(C, x, W, scale, M, N, K, slice, nblk); return; }
    /* NOTE: staging x pre-widened to f32 here would remove the per-row __bfloat162float work,
     * but the arena is sized by the EMITTER for K*2 bytes (bf16), so writing K*4 overruns it and
     * faults with CUDA_ERROR_ILLEGAL_ADDRESS. Row-blocking below gets the same saving without
     * touching the packet's arena claim. */
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

#if PLOW_NV_FP8_RB > 1
    /* ROW-BLOCKED fp8 GEMV. fp8 halves the weight bytes but NOT the per-element x widening, so
     * the arm spends ~8 of its ~28 inner-loop instructions per 8 bytes on __bfloat162float(x) --
     * it measured 1046 GB/s, the slowest arm in either precision, and is compute-bound rather
     * than bandwidth-bound. Owning GV_FP8_RB rows lets the warp widen each x chunk ONCE and
     * reuse it across all RB weight rows, and gives RB independent weight streams. Register cost
     * is small (uint2 per element, not uint4). Per-row FMA order is unchanged. */
    for (unsigned nb = n0 + warp * PLOW_NV_FP8_RB; nb < n1; nb += PLOW_NV_WARPS * PLOW_NV_FP8_RB) {
        const unsigned nrows = (nb + PLOW_NV_FP8_RB <= n1) ? (unsigned)PLOW_NV_FP8_RB : (n1 - nb);
        float acc[PLOW_NV_FP8_RB];
#pragma unroll
        for (int r = 0; r < PLOW_NV_FP8_RB; r++) acc[r] = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_FP8) {
            uint2 wv[PLOW_NV_FP8_RB][GV_UNROLL_FP8];
            unsigned kk[GV_UNROLL_FP8];
#pragma unroll
            for (int u = 0; u < GV_UNROLL_FP8; u++)
                kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8;
#pragma unroll
            for (int r = 0; r < PLOW_NV_FP8_RB; r++) {
                const uint8_t* wr = W + (size_t)(nb + (unsigned)r) * K;
#pragma unroll
                for (int u = 0; u < GV_UNROLL_FP8; u++)
                    wv[r][u] = ((unsigned)r < nrows && kk[u] < K) ? *(const uint2*)(wr + kk[u])
                                                                 : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL_FP8; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld_smem8(xs + kk[u]);
                float xf[8]; /* widen ONCE, reuse across the RB rows */
#pragma unroll
                for (int j = 0; j < 8; j++) xf[j] = __bfloat162float(xv.x[j]);
#pragma unroll
                for (int r = 0; r < PLOW_NV_FP8_RB; r++) acc[r] = dot8_fp8_fx(wv[r][u], xf, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < PLOW_NV_FP8_RB; r++) {
            const float t = warp_sum32(acc[r]);
            if (lane == 0 && M && (unsigned)r < nrows)
                C[nb + (unsigned)r] = __float2bfloat16(t * scale[nb + (unsigned)r]);
        }
    }
#else
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const uint8_t* wrow = W + (size_t)n * K;
        float acc = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_FP8) {
            uint2 wv[GV_UNROLL_FP8];
            unsigned kk[GV_UNROLL_FP8];
#pragma unroll
            for (int u = 0; u < GV_UNROLL_FP8; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                wv[u] = (k < K) ? *(const uint2*)(wrow + k) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL_FP8; u++) {
                if (kk[u] >= K) continue;
                acc = dot8_fp8(wv[u], ld_smem8(xs + kk[u]), acc);
            }
        }
        const float t = warp_sum32(acc);
        if (lane == 0 && M) C[n] = __float2bfloat16(t * scale[n]);
    }
#endif
}

/* BATCH>1: gate|up fp8 rows dequanted ONCE, fma'd against all MM x-rows, GLU per (row,col).
 * C is [M][N]. MM==1 is byte-identical to the old scalar body (the B=1 serving path). */
template <int MM, int UN = gv_un_glu_fp8<MM>::v>
__device__ __forceinline__ void gemv_glu_rows_fp8(__nv_bfloat16* __restrict__ C,
                               const __nv_bfloat16* __restrict__ x,
                               const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                               const float* __restrict__ sg, const float* __restrict__ su,
                               unsigned M, unsigned N, unsigned K, unsigned act, unsigned slice,
                               unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const uint8_t* grow = Wg + (size_t)n * K;
        const uint8_t* urow = Wu + (size_t)n * K;
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        for (unsigned c = 0; c < nchunk; c += UN) {
            uint2 gv[UN], uv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                gv[u] = (k < K) ? *(const uint2*)(grow + k) : make_uint2(0u, 0u);
                uv[u] = (k < K) ? *(const uint2*)(urow + k) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if ((unsigned)m >= M) continue;
                    const bf16v8 xv = ld_glob8(x + (size_t)m * K + kk[u]);
                    ag[m] = dot8_fp8(gv[u], xv, ag[m]);
                    au[m] = dot8_fp8(uv[u], xv, au[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float tg = warp_sum32(ag[m]) * sg[n], tu = warp_sum32(au[m]) * su[n];
            if (lane == 0 && (unsigned)m < M) {
                const float a = (act == PLOW_ACT_SILU_) ? act_silu(tg) : act_gelu_tanh(tg);
                C[(size_t)m * N + n] = __float2bfloat16(a * tu);
            }
        }
    }
}

/* Non-arena overload: standalone test kernels. Ladder {1,2,4,8} + block-walk for M>8. */
static __device__ void d_gemv_glu_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                               const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                               const float* __restrict__ sg, const float* __restrict__ su,
                               unsigned M, unsigned N, unsigned K, unsigned act, unsigned slice,
                               unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_glu_rows_fp8<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K, Wg, Wu, sg, su,
                                           rows, N, K, act, slice, nblk);
    });
}

#if PLOW_NV_MXFP4_PROJ || PLOW_NV_MXFP4_MOE
/* OCP MXFP4 weight-only GEMV. Hopper has no FP4 tensor-core operand, so decode expands one
 * 32-element microscale block in registers and keeps the activation in bf16. The reduction and
 * scaling order mirrors the packet's CPU reference: f32 dot within each 32-element block, then
 * multiply its exact power-of-two E8M0 scale and accumulate across blocks. */
__device__ __forceinline__ float plow_mxfp4_e2m1(unsigned v) {
    const unsigned mag = v & 7u;
    float x;
    switch (mag) {
    case 0: x = 0.0f; break;
    case 1: x = 0.5f; break;
    case 2: x = 1.0f; break;
    case 3: x = 1.5f; break;
    case 4: x = 2.0f; break;
    case 5: x = 3.0f; break;
    case 6: x = 4.0f; break;
    default: x = 6.0f; break;
    }
    return (v & 8u) ? -x : x;
}

__device__ __forceinline__ float plow_mxfp4_scale(unsigned v) {
    return __uint_as_float(v << 23);
}

template <int MM>
__device__ __forceinline__ void gemv_rows_mxfp4(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
    const __nv_bfloat16* __restrict__ bias, unsigned M, unsigned N, unsigned K,
    unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nscale = (K + 31u) >> 5;
    const unsigned wstride = (K + 1u) >> 1;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = min(n0 + per, N);

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const uint8_t* wrow = W + (size_t)n * wstride;
        const uint8_t* srow = S + (size_t)n * nscale;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;
        for (unsigned b = 0; b < nscale; b++) {
            const unsigned k = (b << 5) + lane;
            const uint8_t packed = k < K ? wrow[k >> 1] : 0u;
            const unsigned q = (k & 1u) ? packed >> 4 : packed & 15u;
            const float w = plow_mxfp4_e2m1(q);
            const float scale = plow_mxfp4_scale(srow[b]);
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const float part = ((unsigned)m < M && k < K)
                                       ? w * __bfloat162float(x[(size_t)m * K + k])
                                       : 0.0f;
                acc[m] += warp_sum32(part) * scale;
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            if (lane == 0 && (unsigned)m < M) {
                const float v = acc[m] + (bias ? __bfloat162float(bias[n]) : 0.0f);
                C[(size_t)m * N + n] = __float2bfloat16(v);
            }
        }
    }
}

static __device__ void d_gemv_mxfp4(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
    const __nv_bfloat16* __restrict__ bias, unsigned M, unsigned N, unsigned K,
    unsigned slice, unsigned nblk) {
    gemv_walk(M, [&](auto mm, unsigned m0, unsigned rows) {
        gemv_rows_mxfp4<decltype(mm)::v>(C + (size_t)m0 * N, x + (size_t)m0 * K,
                                         W, S, bias, rows, N, K, slice, nblk);
    });
}
#endif

/* Arena-aware overload: persistent interpreter stages x into smem. */
static __device__ void d_gemv_glu_fp8(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ x,
                               const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                               const float* __restrict__ sg, const float* __restrict__ su,
                               unsigned M, unsigned N, unsigned K, unsigned act, unsigned slice,
                               unsigned nblk, __nv_bfloat16* __restrict__ arena) {
    /* B>1: the arena holds one x row only — fall back to the batched global path. */
    if (M > 1) { d_gemv_glu_fp8(C, x, Wg, Wu, sg, su, M, N, K, act, slice, nblk); return; }
    __nv_bfloat16* xs = arena;
    for (unsigned i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const unsigned lane = threadIdx.x & PLOW_NV_LANE_MASK;
    const unsigned warp = threadIdx.x >> PLOW_NV_WARP_SHIFT;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + warp; n < n1; n += PLOW_NV_WARPS) {
        const uint8_t* grow = Wg + (size_t)n * K;
        const uint8_t* urow = Wu + (size_t)n * K;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GV_UNROLL_GLU_FP8) {
            uint2 gv[GV_UNROLL_GLU_FP8], uv[GV_UNROLL_GLU_FP8];
            unsigned kk[GV_UNROLL_GLU_FP8];
#pragma unroll
            for (int u = 0; u < GV_UNROLL_GLU_FP8; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8;
                kk[u] = k;
                gv[u] = (k < K) ? *(const uint2*)(grow + k) : make_uint2(0u, 0u);
                uv[u] = (k < K) ? *(const uint2*)(urow + k) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < GV_UNROLL_GLU_FP8; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld_smem8(xs + kk[u]);
                ag = dot8_fp8(gv[u], xv, ag);
                au = dot8_fp8(uv[u], xv, au);
            }
        }
        const float tg = warp_sum32(ag) * sg[n], tu = warp_sum32(au) * su[n];
        if (lane == 0 && M) {
            const float a = (act == PLOW_ACT_SILU_) ? act_silu(tg) : act_gelu_tanh(tg);
            C[n] = __float2bfloat16(a * tu);
        }
    }
}
