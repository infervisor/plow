/* op_gemm_sm90.cuh — Hopper (sm_90a) WGMMA fork of the prefill tiled GEMM bodies.
 *
 * Included ONLY from op_gemm.cuh under `#if defined(PLOW_NV_HOPPER)`; sm_120a keeps the
 * shared mma.sync bodies byte-for-byte. Each d_* here is a drop-in replacement for the
 * matching op_gemm.cuh body: SAME signature, SAME (slice, nblk) persistent tile loop, SAME
 * 128x128 output tile, SAME arena pointer. Only the tensor-core path changes.
 *
 * WHY: on sm_90a the shared bodies emit HMMA.16816 (Ampere mma.sync) and — worse — the fp8
 * w8a8 body's mma.sync.m16n8k32.e4m3 is EMULATED as 12x F2FP + 2x HMMA, so it gets zero fp8
 * tensor-core benefit. wgmma is the only route to Hopper's HGMMA/QGMMA pipes.
 *
 * The primitives (descriptor encoder, 128B swizzle, alignment, the two MMAs, the accumulator
 * map) are NOT re-derived here — they come from sm90_wgmma.cuh, which records the descriptor /
 * 1024 B alignment / +32 B substep gotchas. The mainloops are ports of the oracle-validated
 * probes runtime/nvidia/experiments/wgmma_bf16_probe.cu (177 TF/s) and wgmma_fp8_probe.cu.
 *
 * ---------------------------------------------------------------- 256 threads = 2 warpgroups
 * The megakernel block shape is fixed at PLOW_NV_THREADS=256 (op_attention.cuh:35), i.e. exactly
 * two 128-thread warpgroups. Both are CONSUMERS (no producer/consumer split — that needs
 * setmaxnreg + TMA, a later stage). The 128x128 output tile is split along M:
 *     warpgroup 0 -> rows [0,64)   warpgroup 1 -> rows [64,128)
 * so each warpgroup issues one m64n128 wgmma per k-substep and holds 64 f32 accumulators —
 * the SAME 64 f32/thread the mma.sync body already holds (acc[MFRAG][NFRAG][4] = 2*8*4). Both
 * warpgroups read the SAME B tile (n is not split), which is what makes the B-side smem
 * traffic identical to the single-warpgroup probe. Staging is done cooperatively by all 256
 * threads.
 *
 * Keeping BM=BN=128 means tiles_m/tiles_n/ntiles are IDENTICAL to the mma.sync body, so the
 * packet's (slice, nblk) work split and every wave-count property of the scheduler is
 * unchanged; this is a pure inner-loop swap.
 *
 * ---------------------------------------------------------------- cross-warpgroup smem hazard
 * The probe's single-warpgroup loop refills buffer (ks-1)%STAGES right after `wgmma.wait_group 1`
 * retires the group that read it. wait_group is WARPGROUP-scoped, so with two warpgroups
 * warpgroup 0 could refill a buffer that warpgroup 1's still-in-flight group is reading. The fix
 * is one extra __syncthreads() between the wait and the refill — the same two-barrier-per-k-step
 * cadence the existing mma.sync body already pays.
 *
 * ---------------------------------------------------------------- K-tile depth
 * The 128 B swizzle requires each logical smem row to be exactly 128 B, so BK is fixed by the
 * dtype: 64 bf16 or 128 e4m3. Either way one stage is A[128][128 B] + B[128][128 B] = 32 KiB.
 * PGM90_STAGES buffers of that is the arena claim (see PGM_ARENA_SM90 in op_gemm.cuh).
 */
#ifndef PLOW_OP_GEMM_SM90_CUH
#define PLOW_OP_GEMM_SM90_CUH

#include "sm90_wgmma.cuh"

/* ---- tile geometry (M/N deliberately equal to PGM_BM/PGM_BN) ---- */
#define PGM90_BM 128
#define PGM90_BN 128
#define PGM90_WGS 2                        /* warpgroups = PLOW_NV_THREADS/128 */
#define PGM90_MSLAB (PGM90_BM / PGM90_WGS) /* 64 = wgmma M */
#define PGM90_NACC 64                      /* f32 accumulators/thread for m64n128 */

#define PGM90_BK 64                        /* bf16 elements per row = 128 B swizzle atom row */
#define PGM90_KSUB (PGM90_BK / 16)         /* 4 wgmma k16 substeps per stage */
#define PGM90_CH 8                         /* 16-byte chunks per row (128 B / 16) */
#define PGM90_ABUF (PGM90_BM * PGM90_BK)   /* bf16 per staged A tile = 16 KiB */
#define PGM90_BBUF (PGM90_BN * PGM90_BK)   /* bf16 per staged B tile = 16 KiB */

#define PGM90_BK8 128                      /* e4m3 elements per row = 128 B */
#define PGM90_KSUB8 (PGM90_BK8 / 32)       /* 4 wgmma k32 substeps per stage */
#define PGM90_A8BUF (PGM90_BM * PGM90_BK8) /* BYTES per staged A tile = 16 KiB */
#define PGM90_B8BUF (PGM90_BN * PGM90_BK8) /* BYTES per staged B tile = 16 KiB */

#ifndef PGM90_STAGES
#define PGM90_STAGES 3
#endif
/* GLU rings (A, Bg, Bu) = 48 KiB per stage, so 2 stages is exactly the plain path's 3-stage
 * claim; the arena does not grow for the GLU fork. */
#ifndef PGM90_GLU_STAGES
#define PGM90_GLU_STAGES 2
#endif
/* 1024 B (512 bf16) of slack so the tile base can be rounded up to the swizzle's alignment. */
#define PGM90_PAD 512
#define PGM90_ARENA (PGM90_STAGES * (PGM90_ABUF + PGM90_BBUF) + PGM90_PAD)
#define PGM90_ARENA_GLU (PGM90_GLU_STAGES * (PGM90_ABUF + 2 * PGM90_BBUF) + PGM90_PAD)
static_assert(PGM90_ARENA <= PGM_ARENA_BF16, "sm90 wgmma GEMM arena must fit the bf16 claim");
static_assert(PGM90_ARENA_GLU <= PGM_ARENA_BF16, "sm90 wgmma GLU arena must fit the bf16 claim");
static_assert(PGM90_STAGES * (PGM90_A8BUF + PGM90_B8BUF) + 2 * PGM90_PAD <= 2 * PGM_ARENA_BF16,
              "sm90 wgmma w8a8 arena must fit the bf16 claim");
static_assert(PGM90_GLU_STAGES * (PGM90_A8BUF + 2 * PGM90_B8BUF) + 2 * PGM90_PAD <=
                  2 * PGM_ARENA_BF16,
              "sm90 wgmma w8a8 GLU arena must fit the bf16 claim");
static_assert(PLOW_NV_THREADS == 128u * PGM90_WGS, "sm90 wgmma bodies assume 2 warpgroups");


/* ---- cooperative 128B-swizzled staging (all PLOW_NV_THREADS threads) ------------------------
 * Out-of-range rows and the K tail are zero-filled by cp.async's src-size operand, exactly as
 * pgm_stage_a/pgm_stage_b do. `src` is already advanced past a_row0 by the caller. */
__device__ __forceinline__ void pgm90_stage_bf16(__nv_bfloat16* dst,
                                                 const __nv_bfloat16* __restrict__ src, int tid,
                                                 int rows, int row0, int kbase, int R, int K) {
    for (int L = tid; L < rows * PGM90_CH; L += (int)PLOW_NV_THREADS) {
        const int row = L / PGM90_CH, c = L - row * PGM90_CH;
        const int gr = row0 + row, gk = kbase + c * 8;
        int bytes = 0;
        const __nv_bfloat16* g = src;
        if (gr < R && gk < K) {
            g = src + (size_t)gr * K + gk;
            const int rem = K - gk;
            bytes = rem >= 8 ? 16 : rem * 2;
        }
        sm90_cp16(&dst[sm90_swz_off<PGM90_BK, 8>(row, c)], g, bytes);
    }
}
__device__ __forceinline__ void pgm90_stage_fp8(uint8_t* dst, const uint8_t* __restrict__ src,
                                                int tid, int rows, int row0, int kbase, int R,
                                                int K) {
    for (int L = tid; L < rows * PGM90_CH; L += (int)PLOW_NV_THREADS) {
        const int row = L / PGM90_CH, c = L - row * PGM90_CH;
        const int gr = row0 + row, gk = kbase + c * 16;
        int bytes = 0;
        const uint8_t* g = src;
        if (gr < R && gk < K) {
            g = src + (size_t)gr * K + gk;
            const int rem = K - gk;
            bytes = rem >= 16 ? 16 : rem;
        }
        sm90_cp16(&dst[sm90_swz_off<PGM90_BK8, 16>(row, c)], g, bytes);
    }
}

/* ================================ bf16 plain GEMM =========================================== */
/* C[m,n] = A[m,k] . B[n,k]^T. Drop-in for d_gemm. */
static __device__ void d_gemm_sm90(__nv_bfloat16* __restrict__ C,
                                   const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B, unsigned m, unsigned n,
                                   unsigned k, unsigned a_row0, unsigned slice, unsigned nblk,
                                   __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 8u)) return; /* swizzle staging needs K%8==0, k>0 (bf16 chunk = 8 elems) */
    __nv_bfloat16* base = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* As = base;                                /* [STAGES][128][64] swizzled */
    __nv_bfloat16* Bs = base + PGM90_STAGES * PGM90_ABUF;    /* [STAGES][128][64] swizzled */
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7;              /* warpgroup: owns A rows [64*wg, 64*wg+64) */
    const int wiw = (tid >> 5) & 3;       /* warp within the warpgroup */
    const int lane = tid & 31;
    const __nv_bfloat16* Ab = A + (size_t)a_row0 * k;

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK - 1) / PGM90_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        float acc[PGM90_NACC];  /* seeded by the first wgmma (scale-d = 0), no zeroing pass */

        sm90_cp_wait<0>();   /* drain the previous tile so the group count starts clean */
        __syncthreads();
#pragma unroll
        for (int s = 0; s < PGM90_STAGES - 1; s++) {
            if (s < ksteps) {
                pgm90_stage_bf16(As + s * PGM90_ABUF, Ab, tid, PGM90_BM, tm, s * PGM90_BK, (int)m,
                                 (int)k);
                pgm90_stage_bf16(Bs + s * PGM90_BBUF, B, tid, PGM90_BN, tn, s * PGM90_BK, (int)n,
                                 (int)k);
            }
            sm90_cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % PGM90_STAGES;
            sm90_cp_wait<PGM90_STAGES - 2>();
            __syncthreads();
            const __nv_bfloat16* Ac = As + cur * PGM90_ABUF + wg * PGM90_MSLAB * PGM90_BK;
            const __nv_bfloat16* Bc = Bs + cur * PGM90_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB; sub++)
                wgmma_m64n128k16(acc, sm90_desc(Ac + sub * 16), sm90_desc(Bc + sub * 16),
                                 (ks == 0 && sub == 0) ? 0 : 1);
            sm90_wg_commit();
            sm90_wg_wait<1>();   /* group ks-1 retired -> its buffer is free (this warpgroup) */
            __syncthreads();     /* ...and for the OTHER warpgroup too, before we refill it */
            const int nxt = ks + PGM90_STAGES - 1;
            if (nxt < ksteps) {
                const int nb = nxt % PGM90_STAGES;
                pgm90_stage_bf16(As + nb * PGM90_ABUF, Ab, tid, PGM90_BM, tm, nxt * PGM90_BK,
                                 (int)m, (int)k);
                pgm90_stage_bf16(Bs + nb * PGM90_BBUF, B, tid, PGM90_BN, tn, nxt * PGM90_BK,
                                 (int)n, (int)k);
            }
            sm90_cp_commit();
        }
        sm90_wg_wait<0>();

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n)
                        C[(size_t)rr * n + cc] = __float2bfloat16(acc[4 * g + 2 * hi + lo]);
                }
            }
        __syncthreads();
    }
}

#if PGM90_FORK_GLU
/* ================================ bf16 fused gate|up GLU ==================================== */
/* Drop-in for d_gemm_glu: C = act(A.Wg^T) * (A.Wu^T). One ring of (A, Bg, Bu) per K-step and two
 * m64n128 accumulator sets — the SAME 128 f32/thread the mma.sync body holds (accg + accu). */
static __device__ void d_gemm_glu_sm90(__nv_bfloat16* __restrict__ C,
                                       const __nv_bfloat16* __restrict__ A,
                                       const __nv_bfloat16* __restrict__ Wg,
                                       const __nv_bfloat16* __restrict__ Wu, unsigned m, unsigned n,
                                       unsigned k, unsigned act, unsigned slice, unsigned nblk,
                                       __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 8u)) return; /* swizzle staging needs K%8==0, k>0 (bf16 chunk = 8 elems) */
    __nv_bfloat16* base = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* As = base;
    __nv_bfloat16* Bgs = base + PGM90_GLU_STAGES * PGM90_ABUF;
    __nv_bfloat16* Bus = Bgs + PGM90_GLU_STAGES * PGM90_BBUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK - 1) / PGM90_BK;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        float accg[PGM90_NACC], accu[PGM90_NACC];

        sm90_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int s = 0; s < PGM90_GLU_STAGES - 1; s++) {
            if (s < ksteps) {
                pgm90_stage_bf16(As + s * PGM90_ABUF, A, tid, PGM90_BM, tm, s * PGM90_BK, (int)m,
                                 (int)k);
                pgm90_stage_bf16(Bgs + s * PGM90_BBUF, Wg, tid, PGM90_BN, tn, s * PGM90_BK, (int)n,
                                 (int)k);
                pgm90_stage_bf16(Bus + s * PGM90_BBUF, Wu, tid, PGM90_BN, tn, s * PGM90_BK, (int)n,
                                 (int)k);
            }
            sm90_cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % PGM90_GLU_STAGES;
            sm90_cp_wait<PGM90_GLU_STAGES - 2>();
            __syncthreads();
            const __nv_bfloat16* Ac = As + cur * PGM90_ABUF + wg * PGM90_MSLAB * PGM90_BK;
            const __nv_bfloat16* Bg = Bgs + cur * PGM90_BBUF;
            const __nv_bfloat16* Bu = Bus + cur * PGM90_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                const uint64_t da = sm90_desc(Ac + sub * 16);
                wgmma_m64n128k16(accg, da, sm90_desc(Bg + sub * 16), sd);
                wgmma_m64n128k16(accu, da, sm90_desc(Bu + sub * 16), sd);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            __syncthreads();
            const int nxt = ks + PGM90_GLU_STAGES - 1;
            if (nxt < ksteps) {
                const int nb = nxt % PGM90_GLU_STAGES;
                pgm90_stage_bf16(As + nb * PGM90_ABUF, A, tid, PGM90_BM, tm, nxt * PGM90_BK, (int)m,
                                 (int)k);
                pgm90_stage_bf16(Bgs + nb * PGM90_BBUF, Wg, tid, PGM90_BN, tn, nxt * PGM90_BK,
                                 (int)n, (int)k);
                pgm90_stage_bf16(Bus + nb * PGM90_BBUF, Wu, tid, PGM90_BN, tn, nxt * PGM90_BK,
                                 (int)n, (int)k);
            }
            sm90_cp_commit();
        }
        sm90_wg_wait<0>();

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n) {
                        const int r = 4 * g + 2 * hi + lo;
                        const float gt = accg[r];
                        const float a = (act == PLOW_ACT_SILU_) ? act_silu(gt) : act_gelu_tanh(gt);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * accu[r]);
                    }
                }
            }
        __syncthreads();
    }
}

#endif /* PGM90_FORK_GLU */

/* ================================ w8a8 (e4m3 x e4m3) GEMM ================================== */
/* Two-level accumulation (DeepGEMM): Hopper's fp8 wgmma does NOT accumulate in true f32 and the
 * error grows with K. Folding the wgmma accumulator into an f32 shadow every PGM90_BK8=128
 * k-elements cuts it ~10x. It costs the async overlap (wait_group 0 each step) and a second
 * 64-register accumulator, so it is a knob; default OFF because the op-test gate measures the
 * error directly and the un-promoted arm sits inside it. */
#ifndef PGM90_FP8_PROMOTE
#define PGM90_FP8_PROMOTE 0
#endif

/* Drop-in for d_gemm_w8a8. */
static __device__ void d_gemm_w8a8_sm90(__nv_bfloat16* __restrict__ C,
                                        const uint8_t* __restrict__ A,
                                        const uint8_t* __restrict__ B,
                                        const float* __restrict__ ascale,
                                        const float* __restrict__ wscale, unsigned m, unsigned n,
                                        unsigned k, unsigned a_row0, unsigned slice, unsigned nblk,
                                        __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return; /* swizzle staging needs K%16==0, k>0 (e4m3 chunk = 16 bytes) */
    uint8_t* base = (uint8_t*)sm90_align1024(arena);
    uint8_t* As = base;                                 /* [STAGES][128][128] e4m3, swizzled */
    uint8_t* Bs = base + PGM90_STAGES * PGM90_A8BUF;    /* [STAGES][128][128] e4m3, swizzled */
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const uint8_t* Ab = A + (size_t)a_row0 * k;

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        float acc[PGM90_NACC];
#if PGM90_FP8_PROMOTE
        float pacc[PGM90_NACC];
#pragma unroll
        for (int i = 0; i < PGM90_NACC; i++) pacc[i] = 0.f;
#endif

        sm90_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int s = 0; s < PGM90_STAGES - 1; s++) {
            if (s < ksteps) {
                pgm90_stage_fp8(As + s * PGM90_A8BUF, Ab, tid, PGM90_BM, tm, s * PGM90_BK8, (int)m,
                                (int)k);
                pgm90_stage_fp8(Bs + s * PGM90_B8BUF, B, tid, PGM90_BN, tn, s * PGM90_BK8, (int)n,
                                (int)k);
            }
            sm90_cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % PGM90_STAGES;
            sm90_cp_wait<PGM90_STAGES - 2>();
            __syncthreads();
            const uint8_t* Ac = As + cur * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bc = Bs + cur * PGM90_B8BUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB8; sub++) {
#if PGM90_FP8_PROMOTE
                const int sd = (sub == 0) ? 0 : 1;   /* restart the wgmma acc every k-tile */
#else
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
#endif
                wgmma_m64n128k32(acc, sm90_desc(Ac + sub * 32), sm90_desc(Bc + sub * 32), sd);
            }
            sm90_wg_commit();
#if PGM90_FP8_PROMOTE
            sm90_wg_wait<0>();
#pragma unroll
            for (int i = 0; i < PGM90_NACC; i++) pacc[i] += acc[i];
#else
            sm90_wg_wait<1>();
#endif
            __syncthreads();
            const int nxt = ks + PGM90_STAGES - 1;
            if (nxt < ksteps) {
                const int nb = nxt % PGM90_STAGES;
                pgm90_stage_fp8(As + nb * PGM90_A8BUF, Ab, tid, PGM90_BM, tm, nxt * PGM90_BK8,
                                (int)m, (int)k);
                pgm90_stage_fp8(Bs + nb * PGM90_B8BUF, B, tid, PGM90_BN, tn, nxt * PGM90_BK8,
                                (int)n, (int)k);
            }
            sm90_cp_commit();
        }
        sm90_wg_wait<0>();
#if PGM90_FP8_PROMOTE
#pragma unroll
        for (int i = 0; i < PGM90_NACC; i++) acc[i] = pacc[i];
#endif

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const float as = ascale[rr];
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n)
                        C[(size_t)rr * n + cc] =
                            __float2bfloat16(acc[4 * g + 2 * hi + lo] * as * wscale[cc]);
                }
            }
        __syncthreads();
    }
}

#if PGM90_FORK_GLU
/* ================================ w8a8 fused gate|up GLU ==================================== */
/* Drop-in for d_gemm_glu_w8a8. act(a_scale*sg*gate) * (a_scale*su*up). */
static __device__ void d_gemm_glu_w8a8_sm90(__nv_bfloat16* __restrict__ C,
                                            const uint8_t* __restrict__ A,
                                            const uint8_t* __restrict__ Wg,
                                            const uint8_t* __restrict__ Wu,
                                            const float* __restrict__ ascale,
                                            const float* __restrict__ sg,
                                            const float* __restrict__ su, unsigned m, unsigned n,
                                            unsigned k, unsigned act, unsigned slice, unsigned nblk,
                                            __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return; /* swizzle staging needs K%16==0, k>0 (e4m3 chunk = 16 bytes) */
    uint8_t* base = (uint8_t*)sm90_align1024(arena);
    uint8_t* As = base;
    uint8_t* Bgs = base + PGM90_GLU_STAGES * PGM90_A8BUF;
    uint8_t* Bus = Bgs + PGM90_GLU_STAGES * PGM90_B8BUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        float accg[PGM90_NACC], accu[PGM90_NACC];

        sm90_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int s = 0; s < PGM90_GLU_STAGES - 1; s++) {
            if (s < ksteps) {
                pgm90_stage_fp8(As + s * PGM90_A8BUF, A, tid, PGM90_BM, tm, s * PGM90_BK8, (int)m,
                                (int)k);
                pgm90_stage_fp8(Bgs + s * PGM90_B8BUF, Wg, tid, PGM90_BN, tn, s * PGM90_BK8, (int)n,
                                (int)k);
                pgm90_stage_fp8(Bus + s * PGM90_B8BUF, Wu, tid, PGM90_BN, tn, s * PGM90_BK8, (int)n,
                                (int)k);
            }
            sm90_cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % PGM90_GLU_STAGES;
            sm90_cp_wait<PGM90_GLU_STAGES - 2>();
            __syncthreads();
            const uint8_t* Ac = As + cur * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bg = Bgs + cur * PGM90_B8BUF;
            const uint8_t* Bu = Bus + cur * PGM90_B8BUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                const uint64_t da = sm90_desc(Ac + sub * 32);
                wgmma_m64n128k32(accg, da, sm90_desc(Bg + sub * 32), sd);
                wgmma_m64n128k32(accu, da, sm90_desc(Bu + sub * 32), sd);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            __syncthreads();
            const int nxt = ks + PGM90_GLU_STAGES - 1;
            if (nxt < ksteps) {
                const int nb = nxt % PGM90_GLU_STAGES;
                pgm90_stage_fp8(As + nb * PGM90_A8BUF, A, tid, PGM90_BM, tm, nxt * PGM90_BK8,
                                (int)m, (int)k);
                pgm90_stage_fp8(Bgs + nb * PGM90_B8BUF, Wg, tid, PGM90_BN, tn, nxt * PGM90_BK8,
                                (int)n, (int)k);
                pgm90_stage_fp8(Bus + nb * PGM90_B8BUF, Wu, tid, PGM90_BN, tn, nxt * PGM90_BK8,
                                (int)n, (int)k);
            }
            sm90_cp_commit();
        }
        sm90_wg_wait<0>();

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const float as = ascale[rr];
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n) {
                        const int r = 4 * g + 2 * hi + lo;
                        const float gt = accg[r] * as * sg[cc];
                        const float a = (act == PLOW_ACT_SILU_) ? act_silu(gt) : act_gelu_tanh(gt);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * (accu[r] * as * su[cc]));
                    }
                }
            }
        __syncthreads();
    }
}

#endif /* PGM90_FORK_GLU */

#endif /* PLOW_OP_GEMM_SM90_CUH */
