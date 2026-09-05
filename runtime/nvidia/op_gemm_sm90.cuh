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

#ifndef PLOW_NV_FP8_PF_SCALE_WFIRST
#define PLOW_NV_FP8_PF_SCALE_WFIRST 0
#endif

#ifndef PLOW_NV_SEG_M64N64
#define PLOW_NV_SEG_M64N64 0
#endif
#ifndef PLOW_NV_SEG_M64N128
#define PLOW_NV_SEG_M64N128 0
#endif
#if PLOW_NV_SEG_M64N64 && PLOW_NV_SEG_M64N128
#error "select one M64 segment tile"
#endif

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
/* T13 L2 TILE RASTERIZATION (PGM90_TILE_BAND): the ws GEMM measures ~450 TF/s — exactly the
 * 128 FLOP/byte roofline of a 128x128 tile whose B (weight) tile is re-read from DRAM for
 * every m-row of tiles. Remapping the linear tile index into BAND_M-row bands (m fastest
 * within a band, then n, then band) makes the ~264 concurrently-claimed tiles share
 * ~264/BAND_M distinct B tiles, so L2 serves the other BAND_M-1 reads. Bijection over
 * [0, ntiles) including the partial last band. 0 disables (identity). */
#ifndef PGM90_TILE_BAND
#define PGM90_TILE_BAND 16
#endif
__device__ __forceinline__ void sm90_tile_remap(int t, int tiles_m, int tiles_n, int* tm,
                                                int* tn) {
#if PGM90_TILE_BAND > 0
    const int band_sz = PGM90_TILE_BAND * tiles_n;
    const int band = t / band_sz;
    const int bm0 = band * PGM90_TILE_BAND;
    const int bh = (tiles_m - bm0 < PGM90_TILE_BAND) ? (tiles_m - bm0) : PGM90_TILE_BAND;
    const int rem = t - band * band_sz;
    *tm = bm0 + rem % bh;
    *tn = rem / bh;
#else
    *tm = t / tiles_n;
    *tn = t % tiles_n;
#endif
}



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
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
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

/* ================================ TMA + warp-specialized bf16 GEMM ==========================
 * PLOW_NV_TMA_GEMM=1 opt-in. Port of the winning tma_ws_gemm_bf16.cu configuration
 * (BM=BN=128, CWG=1, MS=2: warpgroup 0 = TMA producer, warpgroup 1 = the one consumer,
 * measured 352/259 TF/s vs 175/219 for the uniform cp.async body at 1 blk/SM — the probe's
 * ws_tma row; the 391/332 rows additionally need the entry-reg clamp + 2 blk/SM, which the
 * 255-reg megakernel cannot express). Correctness identical to the cp.async body: TMA
 * zero-fills ragged M and the K tail (relL2 3.79e-06 across the probe's gate shapes).
 *
 * The tensor maps arrive as GENERIC POINTERS to host-encoded 128 B CUtensorMap blobs
 * (CU_TENSOR_MAP_SWIZZLE_128B, boxDim {64, 128}, globalDim {K, rows}) resolved from the
 * packet by the dispatch — see interp_sm120.cu's GEMM case. Both fall back to d_gemm_sm90
 * when a map handle is absent, so pre-TMA packets run unchanged.
 *
 * Ring depth: probe's compromise default NS=4 (381/263/231 TF/s across the three gate
 * shapes). Stage = A[128][64] + B[128][64] bf16 = 32 KiB, so the arena claim grows to
 * NS*32 KiB + mbar + pad — raised via the same grow-only override op_moe_sm90.cuh uses.
 * mbarriers live at the arena base: 2*NS u64, then the 1024 B-aligned tile ring. */
#ifndef PLOW_NV_TMA_GEMM
#define PLOW_NV_TMA_GEMM 0
#endif
#if PLOW_NV_TMA_GEMM

#ifndef PGM90_TMA_STAGES
#define PGM90_TMA_STAGES 4
#endif
#define PGM90_TMA_TXB ((PGM90_ABUF + PGM90_BBUF) * 2) /* TMA bytes per stage */
#define PGM90_TMA_MBAR_BF16 (PGM90_TMA_STAGES * 2 * 4) /* 2*NS u64 as bf16 elems */
#define PGM90_TMA_ARENA                                                                            \
    (PGM90_TMA_MBAR_BF16 + PGM90_TMA_STAGES * (PGM90_ABUF + PGM90_BBUF) + PGM90_PAD)
/* GROW-ONLY: never shrink what op_gemm.cuh already claims (same pattern as op_moe_sm90.cuh). */
#if PGM_ARENA_BF16 < PGM90_TMA_ARENA
#undef PGM_ARENA_BF16
#define PGM_ARENA_BF16 PGM90_TMA_ARENA
#endif

/* C[m,n] = A[a_row0+m,k] . B[n,k]^T, maps over the FULL A/B tensors (a_row0/tile offsets are
 * TMA coordinates, not new descriptors). Drop-in for d_gemm_sm90 when both maps are present.
 *
 * UNIFORM TMA, not producer/consumer: at the megakernel's fixed 256 threads a dedicated
 * producer warpgroup HALVES the math warpgroups (probe shapes used 384-thread blocks), and
 * a first integration measured exactly that: 1368 ms vs the cp.async body's 750 at T=4096.
 * tma_ws_moe_group.cu already showed the win is the LOAD ENGINE, not the split ("TMA alone
 * 1.6-1.75x; warp specialization on top is a WASH"). So both warpgroups consume their m64
 * slab exactly as d_gemm_sm90 does, and ONE elected thread replaces the ~2048 cp.async of
 * a stage with two cp.async.bulk.tensor issues on an mbarrier ring.
 *
 * INLINE, deliberately: the warp-spec draft used __noinline__ to dodge its 128-acc spills,
 * but an ABI call inside the megakernel serialized the wgmma pipeline (ptxas C7510) and
 * caller-saves bled into the dispatch loop — measured as a UNIFORM ~1.8x on every prefill
 * packet, not just GEMM. This body holds 64 accs like d_gemm_sm90; it inlines cleanly. */
static __device__ void d_gemm_sm90_tma(__nv_bfloat16* __restrict__ C, const void* mapA,
                                       const void* mapB, unsigned m, unsigned n, unsigned k,
                                       unsigned a_row0, unsigned slice, unsigned nblk,
                                       __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 8u)) return; /* host mints maps only for K%8==0; keep the loud trap */
    constexpr int NS = PGM90_TMA_STAGES;
    uint64_t* bfull = (uint64_t*)arena; /* dynamic smem is 16 B aligned */
    uint64_t* bempty = bfull + NS;
    __nv_bfloat16* base = (__nv_bfloat16*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    __nv_bfloat16* As = base;                             /* [NS][128][64] swizzled BY TMA */
    __nv_bfloat16* Bs = base + NS * PGM90_ABUF;           /* [NS][128][64] swizzled BY TMA */
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7;              /* warpgroup: owns A rows [64*wg, 64*wg+64) */
    const int wiw = (tid >> 5) & 3, lane = tid & 31;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);  /* the elected issuer arrives (+ tx bytes)          */
        sm90_mbar_init(bempty + tid, 2); /* one elected thread per (consumer) warpgroup      */
    }
    __syncthreads();
    /* NO tensormap-proxy acquire here, deliberately. The maps are host-encoded ONCE at
     * model load, before the persistent kernel ever launches, and nothing ever issues a
     * device-side tensormap.replace — so launch ordering already covers visibility. The
     * probe's fence exists for device-PATCHED maps; here fence.proxy...acquire.gpu is a
     * full-L2-scope fence issued per GEMM op per block, and its collateral is the L2 the
     * surrounding ops wanted (the PLOW_XR_CUS lesson). Re-add it ONLY behind whatever
     * future path device-patches a descriptor. */

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK - 1) / PGM90_BK;

    /* Elected-thread stage issue. `st` counts stages CONTINUOUSLY across tiles (the ring and
     * the mbarrier phases never reset), so issue order == consume order and the bempty wait
     * for a buffer's previous life is always against the right phase. */
    int ist = 0; /* issue cursor (thread 0 only) */
    auto issue = [&](int tile, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
        sm90_mbar_expect(bfull + s, PGM90_TMA_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        sm90_tma2d(sm90_su32(As + s * PGM90_ABUF), mapA, ks * PGM90_BK, (int)a_row0 + tm, bar);
        sm90_tma2d(sm90_su32(Bs + s * PGM90_BBUF), mapB, ks * PGM90_BK, tn, bar);
        ist++;
    };
    /* (tile, ks) at linear stage index i — the issue cursor runs NS-1 ahead of consume. */
    auto stage_at = [&](int i, int& tile, int& ks) {
        tile = (int)slice + (i / ksteps) * (int)nblk;
        ks = i % ksteps;
    };
    const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;

    if (tid == 0) { /* prologue: fill the ring minus one */
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
        float acc[PGM90_NACC];
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const __nv_bfloat16* Ac = As + s * PGM90_ABUF + wg * PGM90_MSLAB * PGM90_BK;
            const __nv_bfloat16* Bc = Bs + s * PGM90_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB; sub++)
                wgmma_m64n128k16(acc, sm90_desc(Ac + sub * 16), sm90_desc(Bc + sub * 16),
                                 (ks == 0 && sub == 0) ? 0 : 1);
            sm90_wg_commit();
            sm90_wg_wait<1>(); /* group st-1 retired (this warpgroup)...                   */
            if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
            prev = s;
            /* Refill AFTER the arrive (bottom of loop, like the cp.async body): the
             * issuer's bempty wait for buffer (st-1)%NS needs BOTH warpgroups' arrivals
             * for stage st-1, and this warpgroup's own arrive is the line above — issuing
             * at the top would deadlock the issuer on its own warpgroup's progress. */
            if (tid == 0 && st + NS - 1 < total) { /* keep the ring NS-1 ahead */
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);

        /* epilogue: byte-identical mapping to d_gemm_sm90 */
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
    }
    /* Rejoin, then invalidate the mbarriers so the next op can reuse the arena as plain
     * data (PTX requires inval before repurposing mbarrier memory). */
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}

/* ---- fused gate|up GLU via the same uniform-TMA ring --------------------------------------
 * Three tiles per stage (A, Wg, Wu) = 48 KiB, so the ring is NS=2 inside the same arena
 * claim. TWO m64n128 accumulator sets (128 f32/thread) — the register shape the cp.async
 * GLU fork pays ~570 spills for; here the ~2048-cp.async-per-stage staging is gone, which
 * is the tradeoff the PGM90_FORK_GLU gate note said would change the answer. Separate
 * opt-in from that gate: dispatch picks this body on i6/i7/i3 map handles alone. */
#define PGM90_TMA_GLU_NS 2
#define PGM90_TMA_GLU_TXB (3 * PGM90_ABUF * 2) /* == 3*A8BUF for the e4m3 twin */
/* The GLU ring (2 stages x 3 tiles = 96 KiB) only exists when the plain ring's arena claim
 * covers it — i.e. PGM90_TMA_STAGES >= 3. A shrunken-arena build (occ-2 experiments run
 * TMA_STAGES=2 so two blocks fit an SM) compiles the GLU TMA arms OUT and its packets are
 * emitted UNFUSED (PLOW_NO_GLU_FUSE=1), so the dispatch never needs them. */
/* Not in the LEAN SEG_GEMM object: its 128-register entry cannot hold the GLU's 128
 * accumulators uniformly (measured: spills eat the occupancy), and its packets are
 * emitted UNFUSED anyway. */
#define PGM90_TMA_HAS_GLU                                                                          \
    (!PLOW_NV_SEG_GEMM &&                                                                          \
     PGM90_TMA_STAGES * (PGM90_ABUF + PGM90_BBUF) >= PGM90_TMA_GLU_NS * 3 * PGM90_ABUF)
#if PGM90_TMA_HAS_GLU
static_assert(PGM90_TMA_MBAR_BF16 >= PGM90_TMA_GLU_NS * 2 * 4, "GLU ring reuses the mbar slab");
static_assert(PGM90_TMA_GLU_NS * 3 * PGM90_ABUF + PGM90_TMA_MBAR_BF16 + PGM90_PAD <=
                  PGM90_TMA_ARENA,
              "GLU TMA ring must fit the TMA arena claim");
#endif

#if PGM90_TMA_HAS_GLU
template <bool E4M3>
static __device__ void d_gemm_glu_sm90_tma_body(__nv_bfloat16* __restrict__ C, const void* mapA,
                                                const void* mapG, const void* mapU,
                                                const float* __restrict__ ascale,
                                                const float* __restrict__ sg,
                                                const float* __restrict__ su, unsigned m,
                                                unsigned n, unsigned k, unsigned act,
                                                unsigned slice, unsigned nblk,
                                                __nv_bfloat16* arena) {
    if (sm90_bad_k(k, E4M3 ? 16u : 8u)) return;
    constexpr int NS = PGM90_TMA_GLU_NS;
    constexpr int BK = E4M3 ? PGM90_BK8 : PGM90_BK;
    constexpr int KSUB = E4M3 ? PGM90_KSUB8 : PGM90_KSUB;
    constexpr int TBUF = E4M3 ? PGM90_A8BUF : PGM90_ABUF; /* elems==bytes for e4m3 */
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    char* base = (char*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    const int esz = E4M3 ? 1 : 2;
    char* As = base; /* [NS][128][BK] swizzled BY TMA, then Bg, Bu */
    char* Gs = base + (size_t)NS * TBUF * esz;
    char* Us = Gs + (size_t)NS * TBUF * esz;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 2);
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + BK - 1) / BK;

    int ist = 0;
    auto issue = [&](int tile, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        sm90_mbar_expect(bfull + s, PGM90_TMA_GLU_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        sm90_tma2d(sm90_su32(As + (size_t)s * TBUF * esz), mapA, ks * BK, tm, bar);
        sm90_tma2d(sm90_su32(Gs + (size_t)s * TBUF * esz), mapG, ks * BK, tn, bar);
        sm90_tma2d(sm90_su32(Us + (size_t)s * TBUF * esz), mapU, ks * BK, tn, bar);
        ist++;
    };
    auto stage_at = [&](int i, int& tile, int& ks) {
        tile = (int)slice + (i / ksteps) * (int)nblk;
        ks = i % ksteps;
    };
    const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;
    if (tid == 0) {
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int tm = (tile / tiles_n) * PGM90_BM;
        const int tn = (tile % tiles_n) * PGM90_BN;
        float accg[PGM90_NACC], accu[PGM90_NACC];
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const char* Ac = As + (size_t)s * TBUF * esz + (size_t)wg * PGM90_MSLAB * BK * esz;
            const char* Gc = Gs + (size_t)s * TBUF * esz;
            const char* Uc = Us + (size_t)s * TBUF * esz;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                const uint64_t da = sm90_desc(Ac + sub * 32);
                if constexpr (E4M3) {
                    wgmma_m64n128k32(accg, da, sm90_desc(Gc + sub * 32), sd);
                    wgmma_m64n128k32(accu, da, sm90_desc(Uc + sub * 32), sd);
                } else {
                    wgmma_m64n128k16(accg, da, sm90_desc(Gc + sub * 32), sd);
                    wgmma_m64n128k16(accu, da, sm90_desc(Uc + sub * 32), sd);
                }
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
            prev = s;
            if (tid == 0 && st + NS - 1 < total) {
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const float as = E4M3 ? ascale[rr] : 1.0f;
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int cc = c0 + 8 * g + lo;
                    if (cc < (int)n) {
                        const int r = 4 * g + 2 * hi + lo;
                        float gv = accg[r], uv = accu[r];
                        if constexpr (E4M3) {
                            gv *= as * sg[cc];
                            uv *= as * su[cc];
                        }
                        const float a = (act == PLOW_ACT_SILU_) ? act_silu(gv) : act_gelu_tanh(gv);
                        C[(size_t)rr * n + cc] = __float2bfloat16(a * uv);
                    }
                }
            }
    }
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}

static __device__ void d_gemm_glu_sm90_tma(__nv_bfloat16* C, const void* mA, const void* mG,
                                           const void* mU, unsigned m, unsigned n, unsigned k,
                                           unsigned act, unsigned slice, unsigned nblk,
                                           __nv_bfloat16* arena) {
    d_gemm_glu_sm90_tma_body<false>(C, mA, mG, mU, nullptr, nullptr, nullptr, m, n, k, act,
                                    slice, nblk, arena);
}
#endif /* PGM90_TMA_HAS_GLU */

/* The n256 and ws384 BF16 bodies share this TMA implementation with FP8. */
#if PLOW_NV_W8A8 || PGM90_UNI_BN256 || PLOW_NV_SEG_M64N64 || PLOW_NV_SEG_M64N128
/* w8a8 twin of d_gemm_sm90_tma: same uniform-TMA ring (stage bytes are IDENTICAL — 128 e4m3
 * = 64 bf16 = 128 B rows), e4m3 maps (GEN_TMAP_E4M3, inner box 128), m64n128k32 QGMMA, and
 * the two-scale epilogue + PGM90_FP8_PROMOTE two-level accumulation of d_gemm_w8a8_sm90. */
static __device__ void d_gemm_w8a8_sm90_tma(__nv_bfloat16* __restrict__ C, const void* mapA,
                                            const void* mapB, const float* __restrict__ ascale,
                                            const float* __restrict__ wscale, unsigned m,
                                            unsigned n, unsigned k, unsigned a_row0,
                                            unsigned slice, unsigned nblk,
                                            __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    constexpr int NS = PGM90_TMA_STAGES;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    uint8_t* As = base;                       /* [NS][128][128] e4m3, swizzled BY TMA */
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 2);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    int ist = 0;
    auto issue = [&](int tile, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
        sm90_mbar_expect(bfull + s, PGM90_TMA_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK8, (int)a_row0 + tm, bar);
        sm90_tma2d(sm90_su32(Bs + s * PGM90_B8BUF), mapB, ks * PGM90_BK8, tn, bar);
        ist++;
    };
    auto stage_at = [&](int i, int& tile, int& ks) {
        tile = (int)slice + (i / ksteps) * (int)nblk;
        ks = i % ksteps;
    };
    const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;
    if (tid == 0) {
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
        float acc[PGM90_NACC];
#if PGM90_FP8_PROMOTE
        float pacc[PGM90_NACC];
#pragma unroll
        for (int i = 0; i < PGM90_NACC; i++) pacc[i] = 0.f;
#endif
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const uint8_t* Ac = As + s * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bc = Bs + s * PGM90_B8BUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB8; sub++) {
#if PGM90_FP8_PROMOTE
                const int sd = (sub == 0) ? 0 : 1; /* restart the wgmma acc every k-tile */
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
            if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
            prev = s;
            if (tid == 0 && st + NS - 1 < total) {
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
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
                    if (cc < (int)n) {
                        const float v = acc[4 * g + 2 * hi + lo];
#if PLOW_NV_FP8_PF_SCALE_WFIRST
                        C[(size_t)rr * n + cc] = __float2bfloat16(__fmul_rn(as, __fmul_rn(wscale[cc], v)));
#else
                        C[(size_t)rr * n + cc] = __float2bfloat16(v * as * wscale[cc]);
#endif
                    }
                }
            }
    }
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
/* ---- UNIFORM m128n256 occ-1 body (PGM90_UNI_BN256, T15) -----------------------------------
 * Both bodies above are SMEM-BANDWIDTH bound: per 128x128 stage they move 32 KiB of TMA
 * writes + 32 KiB of wgmma SS reads in the ~1108 cycles the tensor core needs — ~118 B/cyc
 * against the SM's ~128 B/cyc, at occ-1 or occ-2 alike. The m128n256 tile moves 48 KiB write
 * + 48 KiB read per 8.4 MFLOP stage = ~88 B/cyc at full-rate TC — 1.45x smem headroom. Shape:
 * occ-1, both warpgroups math (m64n256 slab each, 128 acc/thread via wgmma_m64n256k32, needs
 * the 255-reg budget — NOT legal under the 128-reg ws-entry cap), elected thread 0 issues
 * 3 TMA boxes/stage (A 128 rows + B 2x128 rows). No FP8_PROMOTE (no room for the shadow
 * accumulator; same unpromoted-numerics contract as the ws body). */
#ifndef PGM90_UNI_BN256
#define PGM90_UNI_BN256 0
#endif
#if PGM90_UNI_BN256
#ifndef PGM90_UNI256_NS
#define PGM90_UNI256_NS 4
#endif
#define PGM90_U256_BN 256
#define PGM90_U256_BBUF (PGM90_U256_BN * PGM90_BK8) /* 32 KiB */
#define PGM90_U256_TXB (PGM90_A8BUF + PGM90_U256_BBUF)
#define PGM90_U256_MBAR_BF16 (PGM90_UNI256_NS * 2 * 4)
#define PGM90_U256_ARENA                                                                       \
    ((PGM90_U256_MBAR_BF16 + PGM90_UNI256_NS * (PGM90_A8BUF + PGM90_U256_BBUF) / 2) + PGM90_PAD)
#if PGM_ARENA_BF16 < PGM90_U256_ARENA
#undef PGM_ARENA_BF16
#define PGM_ARENA_BF16 PGM90_U256_ARENA
#endif

/* T20b: bf16 twin of the fp8 uni256 body below — SAME staging (BK 64 bf16 = 128 B rows, so
 * A/B box bytes are identical), m64n256k16 wgmma, no dequant scales in the epilogue. */
static __device__ void d_gemm_sm90_tma_uni256(__nv_bfloat16* __restrict__ C, const void* mapA,
                                              const void* mapB, unsigned m, unsigned n,
                                              unsigned k, unsigned a_row0, unsigned slice,
                                              unsigned nblk, __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 8u)) return;
    if (n & 1u) { /* the paired bf16x2 epilogue needs even N — every shipped model is */
        if (threadIdx.x == 0) __trap();
        return;
    }
    constexpr int NS = PGM90_UNI256_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_U256_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 2);
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_U256_BN - 1) / PGM90_U256_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK - 1) / PGM90_BK; /* BK = 64 bf16 = one 128 B row */

    int ist = 0;
    auto issue = [&](int tile, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        sm90_mbar_expect(bfull + s, PGM90_U256_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK, (int)a_row0 + tm, bar);
        uint8_t* bs = Bs + s * PGM90_U256_BBUF;
        sm90_tma2d(sm90_su32(bs), mapB, ks * PGM90_BK, tn, bar);
        sm90_tma2d(sm90_su32(bs + 128 * PGM90_BK8), mapB, ks * PGM90_BK, tn + 128, bar);
        ist++;
    };
    auto stage_at = [&](int i, int& tile, int& ks) {
        tile = (int)slice + (i / ksteps) * (int)nblk;
        ks = i % ksteps;
    };
    const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;
    if (tid == 0) {
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        float acc[2 * PGM90_NACC];
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const uint8_t* Ac = As + s * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bc = Bs + s * PGM90_U256_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < 4; sub++) { /* 4 x k16 substeps, +32 B each */
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                wgmma_m64n256k16(acc, sm90_desc(Ac + sub * 32), sm90_desc(Bc + sub * 32), sd);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
            prev = s;
            if (tid == 0 && st + NS - 1 < total) {
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_U256_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const int cc = c0 + 8 * g;
                if (cc + 1 < (int)n) {
                    *(__nv_bfloat162*)(C + (size_t)rr * n + cc) = __floats2bfloat162_rn(
                        acc[4 * g + 2 * hi + 0], acc[4 * g + 2 * hi + 1]);
                }
            }
    }
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}

static __device__ void d_gemm_w8a8_sm90_tma_uni256(__nv_bfloat16* __restrict__ C,
                                                   const void* mapA, const void* mapB,
                                                   const float* __restrict__ ascale,
                                                   const float* __restrict__ wscale, unsigned m,
                                                   unsigned n, unsigned k, unsigned a_row0,
                                                   unsigned slice, unsigned nblk,
                                                   __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    if (n & 1u) { /* the paired bf16x2 epilogue needs even N — every shipped model is */
        if (threadIdx.x == 0) __trap();
        return;
    }
    constexpr int NS = PGM90_UNI256_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_U256_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;
    const int wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 2); /* one arrival per warpgroup */
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_U256_BN - 1) / PGM90_U256_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    int ist = 0;
    auto issue = [&](int tile, int ks) {
        const int s = ist % NS;
        if (ist >= NS) sm90_mbar_wait(bempty + s, ((ist / NS) + 1) & 1);
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        sm90_mbar_expect(bfull + s, PGM90_U256_TXB);
        const uint32_t bar = sm90_su32(bfull + s);
        sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK8, (int)a_row0 + tm, bar);
        uint8_t* bs = Bs + s * PGM90_U256_BBUF;
        sm90_tma2d(sm90_su32(bs), mapB, ks * PGM90_BK8, tn, bar);
        sm90_tma2d(sm90_su32(bs + 128 * PGM90_BK8), mapB, ks * PGM90_BK8, tn + 128, bar);
        ist++;
    };
    auto stage_at = [&](int i, int& tile, int& ks) {
        tile = (int)slice + (i / ksteps) * (int)nblk;
        ks = i % ksteps;
    };
    const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;
    if (tid == 0) {
        for (int i = 0; i < NS - 1 && i < total; i++) {
            int tl, ks;
            stage_at(i, tl, ks);
            issue(tl, ks);
        }
    }

    int st = 0;
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_U256_BN;
        float acc[2 * PGM90_NACC]; /* 128: this warpgroup's m64n256 slab */
        int prev = -1;
        for (int ks = 0; ks < ksteps; ks++, st++) {
            const int s = st % NS;
            sm90_mbar_wait(bfull + s, (st / NS) & 1);
            const uint8_t* Ac = As + s * PGM90_A8BUF + wg * PGM90_MSLAB * PGM90_BK8;
            const uint8_t* Bc = Bs + s * PGM90_U256_BBUF;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                wgmma_m64n256k32(acc, sm90_desc(Ac + sub * 32), sm90_desc(Bc + sub * 32), sd);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);
            prev = s;
            if (tid == 0 && st + NS - 1 < total) {
                int tl, kk;
                stage_at(st + NS - 1, tl, kk);
                issue(tl, kk);
            }
        }
        sm90_wg_wait<0>();
        if (prev >= 0 && (tid & 127) == 0) sm90_mbar_arrive(bempty + prev);

        const int r0 = tm + wg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
        const int c0 = tn + 2 * (lane & 3);
#pragma unroll
        for (int g = 0; g < PGM90_U256_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = r0 + 8 * hi;
                if (rr >= (int)m) continue;
                const float as = ascale[rr];
                const int cc = c0 + 8 * g;
                if (cc + 1 < (int)n) { /* n is a multiple of 256, cc even — one 4 B store */
                    *(__nv_bfloat162*)(C + (size_t)rr * n + cc) = __floats2bfloat162_rn(
                        acc[4 * g + 2 * hi + 0] * as * wscale[cc],
                        acc[4 * g + 2 * hi + 1] * as * wscale[cc + 1]);
                }
            }
    }
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
#endif /* PGM90_UNI_BN256 */

#if PGM90_UNI_BN256 && defined(PLOW_NV_SEG_WS384) && PLOW_NV_SEG_WS384
#if defined(PLOW_NV_LANE_MASK) /* interp TU only — standalone probes lack the GEMV helpers */
/* T36: QuantFp8 (incl. the fused-GLU form) on the ws384 object's CONSUMER warpgroups —
 * quant packets classed 8 merge the [gate/up, glu-quant, down] and [flash, attn-quant, o]
 * chains into single class-8 runs (saves ~2 launches/layer). Producer wg skips (32 regs);
 * the 8 consumer warps re-index the row loop. Vectorized exactly like d_quant_fp8. */
template <bool PROD>
static __device__ void d_quant_fp8_ws384(uint8_t* __restrict__ xq, __nv_bfloat16* __restrict__ x,
                                         float* __restrict__ ascale, unsigned M, unsigned K,
                                         unsigned slice, unsigned nblk,
                                         const __nv_bfloat16* __restrict__ gate,
                                         const __nv_bfloat16* __restrict__ up, unsigned act) {
    if (PROD) return;
    const int lt = (int)threadIdx.x - 128; /* 0..255 over the two consumer wgs */
    const unsigned lane = (unsigned)lt & PLOW_NV_LANE_MASK;
    const unsigned warp = (unsigned)lt >> PLOW_NV_WARP_SHIFT; /* 0..7 */
    const unsigned per = (M + nblk - 1) / nblk;
    const unsigned m0 = slice * per;
    const unsigned m1 = (m0 + per < M) ? (m0 + per) : M;
    const bool v8 = (K & 7u) == 0;
    for (unsigned mm = m0 + warp; mm < m1; mm += 8u) {
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
                for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(__bfloat162float(v.x[j])));
            }
        } else {
            for (unsigned kk = lane; kk < K; kk += 32u)
                amax = fmaxf(amax, fabsf(__bfloat162float(x[row + kk])));
        }
        amax = warp_max32(amax);
        const float as = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
        const float inv = 1.0f / as;
        if (lane == 0) ascale[mm] = as;
        if (v8) {
            for (unsigned kk = lane * 8u; kk < K; kk += 256u) {
                const bf16v8 v = ld_glob8(x + row + kk);
                uint8_t q8[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    __nv_fp8_e4m3 q(__bfloat162float(v.x[j]) * inv);
                    q8[j] = *(const uint8_t*)&q;
                }
                *(uint2*)(xq + row + kk) = *(const uint2*)q8;
            }
        } else {
            for (unsigned kk = lane; kk < K; kk += 32u) {
                __nv_fp8_e4m3 q(__bfloat162float(x[row + kk]) * inv);
                xq[row + kk] = *(const uint8_t*)&q;
            }
        }
    }
}

#endif /* PLOW_NV_LANE_MASK (T36 quant) */
/* T38 (PGM90_WS384_SMEPI): smem-staged coalesced epilogue for the ws384 consumers. ncu on
 * the standalone body shows ~27% est. loss on uncoalesced C stores (16/32 bytes/sector).
 * Each consumer warpgroup stages its m64 slab through the DEAD ring slot of the tile's last
 * stage (safe until its bempty arrive, which moves to AFTER the drain) and streams it out
 * with 16 B fully-coalesced stores; wg-scoped named barriers (ids 1/2), no cross-wg or
 * producer coordination. cwg0 uses Bs[prev] (32 KiB, one pass); cwg1 uses As[prev]
 * (16 KiB, two half-passes). */
#ifndef PGM90_WS384_SMEPI
#define PGM90_WS384_SMEPI 0
#endif
__device__ __forceinline__ void ws384_wg_bar(int cwg) {
    asm volatile("bar.sync %0, %1;" ::"r"(cwg + 1), "r"(128) : "memory");
}

/* ---- T31: 384-THREAD WS BODY (the CUTLASS shape) ------------------------------------------
 * cuBLASLt fp8 measures 1324-1468 TF/s at the 12B shapes on this box vs the 256-thread
 * uniform body's 950-1170: the missing structure is a DEDICATED producer warpgroup. This
 * object launches at 384 threads (wg0 = TMA producer, setmaxnreg-dec to 32; wg1/wg2 =
 * consumers at 224, one m64n256 slab each) — the register pool 128x32 + 256x224 = 61440
 * fits the 64K file; kernel entry is __maxnreg__(160) so ptxas can compile the 128-acc
 * consumer path (needs >=154). Same m128n256 tile, ring and epilogue as the uniform body;
 * the producer never does math, the consumers never issue, so neither stalls the other.
 * Template <PROD, E4M3>: one function, both roles, aligned __syncthreads counts. */
template <bool PROD, bool E4M3>
static __device__ void d_gemm_sm90_tma_ws384_role(__nv_bfloat16* __restrict__ C,
                                                  const void* mapA, const void* mapB,
                                                  const float* __restrict__ ascale,
                                                  const float* __restrict__ wscale, unsigned m,
                                                  unsigned n, unsigned k, unsigned a_row0,
                                                  unsigned slice, unsigned nblk,
                                                  __nv_bfloat16* arena) {
    if (sm90_bad_k(k, E4M3 ? 16u : 8u)) return;
    if (n & 1u) { /* the paired bf16x2 epilogue needs even N — every shipped model is */
        if (threadIdx.x == 0) __trap();
        return;
    }
    constexpr int NS = PGM90_UNI256_NS;
    constexpr int BKB = 128; /* bytes per staged k-step row (128 e4m3 = 64 bf16) */
    const int kelem = E4M3 ? 128 : 64;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_U256_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;

    if (PROD && tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 2); /* one rep per CONSUMER warpgroup */
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_U256_BN - 1) / PGM90_U256_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + kelem - 1) / kelem;

    if (PROD) {
        if (tid == 0) {
            int ist = 0;
            const int total = ((ntiles - (int)slice) + (int)nblk - 1) / (int)nblk * ksteps;
            for (int i = 0; i < total; i++) {
                const int tile = (int)slice + (i / ksteps) * (int)nblk;
                const int ks = i % ksteps;
                const int st = ist % NS;
                if (ist >= NS) sm90_mbar_wait(bempty + st, ((ist / NS) + 1) & 1);
                int tmi, tni;
                sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
                const int tm = tmi * PGM90_BM;
                const int tn = tni * PGM90_U256_BN;
                sm90_mbar_expect(bfull + st, PGM90_U256_TXB);
                const uint32_t bar = sm90_su32(bfull + st);
                sm90_tma2d(sm90_su32(As + st * PGM90_A8BUF), mapA, ks * kelem, (int)a_row0 + tm,
                           bar);
                uint8_t* bs = Bs + st * PGM90_U256_BBUF;
                sm90_tma2d(sm90_su32(bs), mapB, ks * kelem, tn, bar);
                sm90_tma2d(sm90_su32(bs + 128 * BKB), mapB, ks * kelem, tn + 128, bar);
                ist++;
            }
        }
    } else {
        const int cwg = (tid >> 7) - 1; /* consumer warpgroup 0/1 -> m64 slab */
        const int lt = tid & 127;
        const int wiw = lt >> 5, lane = lt & 31;
        int st = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            int tmi, tni;
            sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
            const int tm = tmi * PGM90_BM;
            const int tn = tni * PGM90_U256_BN;
            float acc[2 * PGM90_NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                sm90_mbar_wait(bfull + s, (st / NS) & 1);
                const uint8_t* Ac = As + s * PGM90_A8BUF + cwg * PGM90_MSLAB * BKB;
                const uint8_t* Bc = Bs + s * PGM90_U256_BBUF;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < 4; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    if constexpr (E4M3)
                        wgmma_m64n256k32(acc, sm90_desc(Ac + sub * 32),
                                         sm90_desc(Bc + sub * 32), sd);
                    else
                        wgmma_m64n256k16(acc, sm90_desc(Ac + sub * 32),
                                         sm90_desc(Bc + sub * 32), sd);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);
                prev = s;
            }
            sm90_wg_wait<0>();
#if !PGM90_WS384_SMEPI
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);

            const int r0 = tm + cwg * PGM90_MSLAB + wiw * 16 + (lane >> 2);
            const int c0 = tn + 2 * (lane & 3);
#pragma unroll
            for (int g = 0; g < PGM90_U256_BN / 8; g++)
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int rr = r0 + 8 * hi;
                    if (rr >= (int)m) continue;
                    const float as_ = ascale ? ascale[rr] : 1.0f;
                    const int cc = c0 + 8 * g;
                    if (cc + 1 < (int)n) {
                        const float w0 = wscale ? wscale[cc] : 1.0f;
                        const float w1 = wscale ? wscale[cc + 1] : 1.0f;
                        *(__nv_bfloat162*)(C + (size_t)rr * n + cc) = __floats2bfloat162_rn(
                            acc[4 * g + 2 * hi + 0] * as_ * w0,
                            acc[4 * g + 2 * hi + 1] * as_ * w1);
                    }
                }
#else
            /* T38: stage the slab in the dead ring slot, then coalesced 16 B stores.
             * bempty(prev) arrival moves to AFTER the drain — the producer must not refill
             * this slot while it doubles as the C stage. */
            {
                /* Both consumer wgs must have DRAINED before either stages into the shared
                 * ring slots (they drift by a stage; cwg0's B stage is cwg1's operand and
                 * cwg1's A stage covers cwg0's slab). Consumers-only barrier, id 3. */
                asm volatile("bar.sync 3, 256;" ::: "memory");
                const int pv = prev < 0 ? 0 : prev;
                const int lr0 = wiw * 16 + (lane >> 2);
                const int lc0 = 2 * (lane & 3);
                /* cwg0: Bs slot (32 KiB, whole slab). cwg1: As slot (16 KiB, two halves). */
                __nv_bfloat16* stg =
                    cwg == 0 ? (__nv_bfloat16*)(Bs + (size_t)pv * PGM90_U256_BBUF)
                             : (__nv_bfloat16*)(As + (size_t)pv * PGM90_A8BUF);
                const int halves = cwg == 0 ? 1 : 2;
                const int rows_per = PGM90_MSLAB / halves; /* 64 or 32 */
                for (int h = 0; h < halves; h++) {
#pragma unroll
                    for (int g = 0; g < PGM90_U256_BN / 8; g++)
#pragma unroll
                        for (int hi = 0; hi < 2; hi++) {
                            const int lr = lr0 + 8 * hi;
                            if (lr < h * rows_per || lr >= (h + 1) * rows_per) continue;
                            const unsigned gr = (unsigned)(tm + cwg * PGM90_MSLAB + lr);
                            const float as_ = ascale ? ascale[gr < m ? gr : 0u] : 1.0f;
                            const int lc = lc0 + 8 * g;
                            const float w0 = wscale ? wscale[tn + lc] : 1.0f;
                            const float w1 = wscale ? wscale[tn + lc + 1] : 1.0f;
                            *(__nv_bfloat162*)(stg +
                                               (size_t)(lr - h * rows_per) * PGM90_U256_BN +
                                               lc) =
                                __floats2bfloat162_rn(acc[4 * g + 2 * hi + 0] * as_ * w0,
                                                      acc[4 * g + 2 * hi + 1] * as_ * w1);
                        }
                    ws384_wg_bar(cwg);
                    /* 128 threads x 16 B chunks over rows_per x 256 bf16. */
                    const int chunks = rows_per * (PGM90_U256_BN / 8);
                    for (int i = lt; i < chunks; i += 128) {
                        const int row = i >> 5, ch = i & 31;
                        const int gr = tm + cwg * PGM90_MSLAB + h * rows_per + row;
                        /* The column guard is NOT redundant with `gr < m`: this stores a
                         * whole 8-wide chunk, so an n that is not a multiple of the 256-col
                         * tile (e.g. n%256==128) would write past each row's end into the
                         * next tensor. The non-SMEPI epilogue checks `cc + 1 < n` for the
                         * same reason. */
                        if (gr < (int)m && tn + ch * 8 + 8 <= (int)n)
                            *(uint4*)(C + (size_t)gr * n + tn + ch * 8) =
                                *(const uint4*)(stg + (size_t)row * PGM90_U256_BN + ch * 8);
                    }
                    ws384_wg_bar(cwg);
                }
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);
            }
#endif
        }
    }
    __syncthreads();
    if (PROD && tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
#endif /* PGM90_UNI_BN256 && PLOW_NV_SEG_WS384 */

#if PLOW_NV_SEG_M64N64 || PLOW_NV_SEG_M64N128
__device__ __forceinline__ void plow_wgmma_m64n64k16(float* d, uint64_t da, uint64_t db, int scale_d) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0; }\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(scale_d) : "memory");
}

template <bool PROD, int BN>
static __device__ void d_gemm_sm90_tma_m64_role(__nv_bfloat16* __restrict__ C,
    const void* mapA, const void* mapB, unsigned m, unsigned n, unsigned k,
    unsigned a_row0, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    static_assert(BN == 64 || BN == 128);
    if (sm90_bad_k(k, 8u)) return;
    if (n & 1u) { if (threadIdx.x == 0) __trap(); return; }
    constexpr int NS = PGM90_TMA_STAGES;
    // Preserve the full 128-row tensor-map transfer for either output tile.
    constexpr int slab_bytes = 128 * 128;
    uint64_t* full = (uint64_t*)arena;
    uint64_t* empty = full + NS;
    uint8_t* As = (uint8_t*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    uint8_t* Bs = As + NS * slab_bytes;
    const int tid = (int)threadIdx.x;
    if (PROD && tid < NS) {
        sm90_mbar_init(full + tid, 1);
        sm90_mbar_init(empty + tid, 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    const int tiles_m = ((int)m + 63) / 64, tiles_n = ((int)n + BN - 1) / BN;
    const int ntiles = tiles_m * tiles_n, ksteps = ((int)k + 63) / 64;
    if (PROD) {
        if (tid == 0) {
            int stage = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                const int tm = (tile / tiles_n) * 64, tn = (tile % tiles_n) * BN;
                for (int ks = 0; ks < ksteps; ks++, stage++) {
                    const int slot = stage % NS;
                    if (stage >= NS) sm90_mbar_wait(empty + slot, ((stage / NS) + 1) & 1);
                    sm90_mbar_expect(full + slot, 2 * slab_bytes);
                    const uint32_t bar = sm90_su32(full + slot);
                    sm90_tma2d(sm90_su32(As + slot * slab_bytes), mapA, ks * 64, (int)a_row0 + tm, bar);
                    sm90_tma2d(sm90_su32(Bs + slot * slab_bytes), mapB, ks * 64, tn, bar);
                }
            }
        }
    } else {
        const int lt = tid & 127, warp = lt >> 5, lane = lt & 31;
        int stage = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            const int tm = (tile / tiles_n) * 64, tn = (tile % tiles_n) * BN;
            float acc[BN / 2];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, stage++) {
                const int slot = stage % NS;
                sm90_mbar_wait(full + slot, (stage / NS) & 1);
                const uint8_t* a = As + slot * slab_bytes;
                const uint8_t* b = Bs + slot * slab_bytes;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < 4; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    if constexpr (BN == 64)
                        plow_wgmma_m64n64k16(acc, sm90_desc(a + sub * 32), sm90_desc(b + sub * 32), sd);
                    else
                        wgmma_m64n128k16(acc, sm90_desc(a + sub * 32), sm90_desc(b + sub * 32), sd);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(empty + prev);
                prev = slot;
            }
            sm90_wg_wait<0>();
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(empty + prev);
            const int row = tm + warp * 16 + (lane >> 2), col = tn + 2 * (lane & 3);
#pragma unroll
            for (int g = 0; g < BN / 8; g++)
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int rr = row + 8 * hi, cc = col + 8 * g;
                    if (rr < (int)m && cc + 1 < (int)n)
                        *(__nv_bfloat162*)(C + (size_t)rr * n + cc) =
                            __floats2bfloat162_rn(acc[4 * g + 2 * hi], acc[4 * g + 2 * hi + 1]);
                }
        }
    }
    __syncthreads();
    if (PROD && tid < NS) {
        sm90_mbar_inval(full + tid);
        sm90_mbar_inval(empty + tid);
    }
    __syncthreads();
}
#endif

#if PLOW_NV_SEG_GEMM
/* ---- WARP-SPECIALIZED + setmaxnreg twin, LEAN SEG_GEMM OBJECT ONLY ------------------------
 * The probe's zero-spill occ-2 recipe (tma_ws_gemm_bf16.cu ws_tma_smr: entry 128, producer
 * dec->32 / consumer inc->224, 391 TF/s, 0 spills at 2 blocks/SM). This lost 1.8x inside the
 * 256-thread MEGAKERNEL (a producer warpgroup halves the math warpgroups at occ-1), but in
 * the lean object the math is inverted: occ-2 x 1 consumer warpgroup = the same 2 math
 * warpgroups/SM the uniform occ-1 shape has, PLUS dedicated staging, PLUS zero spills where
 * the capped uniform body spilled its way to a 30% loss (measured, dd5d2b2).
 *
 * Register choreography per op invocation: wg0 dec 128->32 for the issue loop, inc back at
 * the end; wg1 inc 128->224 (waits on wg0's dec via the CTA pool) for the 128-accumulator
 * mainloop, dec back before the final rejoin — so every OTHER op body in this object still
 * runs uniformly at the 128-register entry. dec never blocks, so the restore order
 * (consumer dec frees, producer inc reclaims) cannot deadlock. */
/* Ring shares the PGM90_TMA_STAGES arena claim; the lean build sets STAGES=3
 * (3 x 32 KiB + mbar + pad = 97.7 KiB <= the occ-2 113 KiB/block cap). NOTE: no
 * PGM90_FP8_PROMOTE here — 128 acc + 128 shadow does not fit 224 registers; the
 * unpromoted relL2 (1.14e-3 @K=3840) still passes the 1.6e-3 oracle bar. */
#define PGM90_WS_NS PGM90_TMA_STAGES

/* PLOW_NV_SEG_WS_ENTRY=1: the warpgroup register split happens ONCE at kernel entry
 * (interp_sm120.cu) instead of per op body. The per-op dec/inc restore cycle is the
 * measured deadlock (it reproduced even in the arm-stripped PLOW_NV_GEMM_ONLY TU, so it
 * is the cycling itself, not TU pollution); no shipped kernel cycles setmaxnreg — probe
 * and CUTLASS split once per launch. Only legal on a pure-GEMM packet stream
 * (PLOW_SEG_PURE_GEMM emit + PLOW_PF_SEG_PURE serve): every packet this object claims
 * runs the role-split body, so no uniform body ever executes at producer registers. */
#ifndef PLOW_NV_SEG_WS_ENTRY
#define PLOW_NV_SEG_WS_ENTRY 0
#endif

__device__ __forceinline__ void sm90_reg_dec(int n) {
    switch (n) { /* immediate-only PTX operand */
    case 32: asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n"); break;
    default: asm volatile("setmaxnreg.dec.sync.aligned.u32 128;\n"); break;
    }
}
__device__ __forceinline__ void sm90_reg_inc(int n) {
    switch (n) {
    case 224: asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n"); break;
    default: asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n"); break;
    }
}

static __device__ void d_gemm_w8a8_sm90_tma_ws(__nv_bfloat16* __restrict__ C, const void* mapA,
                                               const void* mapB, const float* __restrict__ ascale,
                                               const float* __restrict__ wscale, unsigned m,
                                               unsigned n, unsigned k, unsigned a_row0,
                                               unsigned slice, unsigned nblk,
                                               __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    constexpr int NS = PGM90_WS_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;

    if (tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 1); /* ONE consumer warpgroup */
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    if (tid < 128) {
        /* ---------------- producer warpgroup: dec to 32, one elected issuer -------------- */
#if !PLOW_NV_SEG_WS_ENTRY
        sm90_reg_dec(32);
#endif
        if (tid == 0) {
            int st = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                const int tm = (tile / tiles_n) * PGM90_BM;
                const int tn = (tile % tiles_n) * PGM90_BN;
                for (int ks = 0; ks < ksteps; ks++, st++) {
                    const int s = st % NS;
                    if (st >= NS) sm90_mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                    sm90_mbar_expect(bfull + s, PGM90_TMA_TXB);
                    const uint32_t bar = sm90_su32(bfull + s);
                    sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK8,
                               (int)a_row0 + tm, bar);
                    sm90_tma2d(sm90_su32(Bs + s * PGM90_B8BUF), mapB, ks * PGM90_BK8, tn, bar);
                }
            }
        }
#if !PLOW_NV_SEG_WS_ENTRY
        sm90_reg_inc(128); /* restore for the next (uniform) op body */
#endif
    } else {
        /* ------------- consumer warpgroup: inc to 224, MS=2 m64 slabs (128 accs) --------- */
#if !PLOW_NV_SEG_WS_ENTRY
        sm90_reg_inc(224);
#endif
        const int lt = tid - 128;
        const int wiw = lt >> 5, lane = lt & 31;
        int st = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            const int tm = (tile / tiles_n) * PGM90_BM;
            const int tn = (tile % tiles_n) * PGM90_BN;
            float acc[2][PGM90_NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                sm90_mbar_wait(bfull + s, (st / NS) & 1);
                const uint8_t* Ac = As + s * PGM90_A8BUF;
                const uint8_t* Bc = Bs + s * PGM90_B8BUF;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    const uint64_t db = sm90_desc(Bc + sub * 32);
#pragma unroll
                    for (int sl = 0; sl < 2; sl++)
                        wgmma_m64n128k32(acc[sl],
                                         sm90_desc(Ac + sl * 64 * PGM90_BK8 + sub * 32), db, sd);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);
                prev = s;
            }
            sm90_wg_wait<0>();
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);

            const int c0 = tn + 2 * (lane & 3);
#pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                const int r0 = tm + sl * 64 + wiw * 16 + (lane >> 2);
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
                                C[(size_t)rr * n + cc] = __float2bfloat16(
                                    acc[sl][4 * g + 2 * hi + lo] * as * wscale[cc]);
                        }
                    }
            }
        }
#if !PLOW_NV_SEG_WS_ENTRY
        sm90_reg_dec(128); /* free the extra registers before the rejoin */
#endif
    }
    __syncthreads();
    if (tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
/* T13 (PGM90_WS_BN256): the ws role body at BM=64 / BN=256 — one m64n256k32 QGMMA per
 * k-substep instead of two m64n128k32, halving the issue stream at the same 128 accs/thread.
 * Stage = A 16 KiB (128-row TMA box, top 64 rows used; TMA zero-fills past the M edge) +
 * B 32 KiB (two contiguous 128-row boxes) = 48 KiB; NS=2 fits the occ-2 smem budget.
 * Every emitted N (512/3840/4096/15360) is a multiple of 256. */
#ifndef PGM90_WS_BN256
#define PGM90_WS_BN256 0
#endif
#if PGM90_WS_BN256
#define PGM90_WSB_BM 64
#define PGM90_WSB_BN 256
#define PGM90_WSB_NS 2
#define PGM90_WSB_ABUF (128 * PGM90_BK8)            /* bytes: full 128-row box */
#define PGM90_WSB_BBUF (PGM90_WSB_BN * PGM90_BK8)   /* bytes */
#define PGM90_WSB_TXB (PGM90_WSB_ABUF + PGM90_WSB_BBUF)
static_assert(PGM90_WSB_NS * (PGM90_WSB_ABUF + PGM90_WSB_BBUF) <=
                  PGM90_TMA_STAGES * (PGM90_A8BUF + PGM90_B8BUF),
              "BN256 ws ring must fit the TMA arena claim (build with PGM90_TMA_STAGES=3)");
#endif

#if PLOW_NV_SEG_WS_ENTRY
/* Role-split twin of d_gemm_w8a8_sm90_tma_ws for the ONCE-per-launch register split
 * (PLOW_NV_SEG_WS_ENTRY): each warpgroup runs its OWN template instantiation from its own
 * role loop (interp_sm120.cu plow_ws_role_loop), so ptxas never sees the roles reconverge
 * and can honor the entry setmaxnreg (a reconverged split is dropped with C7507). The two
 * instantiations execute the SAME __syncthreads() count (3) so block barriers stay aligned
 * across the divergent loops. Body math is d_gemm_w8a8_sm90_tma_ws's, verbatim. */
template <bool PROD>
static __device__ void d_gemm_w8a8_sm90_tma_ws_role(__nv_bfloat16* __restrict__ C,
                                                    const void* mapA, const void* mapB,
                                                    const float* __restrict__ ascale,
                                                    const float* __restrict__ wscale, unsigned m,
                                                    unsigned n, unsigned k, unsigned a_row0,
                                                    unsigned slice, unsigned nblk,
                                                    __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    constexpr int NS = PGM90_WS_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_A8BUF;
    const int tid = (int)threadIdx.x;

    if (PROD && tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 1);
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_BN - 1) / PGM90_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    if (PROD) {
        if (tid == 0) {
            int st = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                int tmi, tni;
                sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
                const int tm = tmi * PGM90_BM;
                const int tn = tni * PGM90_BN;
                for (int ks = 0; ks < ksteps; ks++, st++) {
                    const int s = st % NS;
                    if (st >= NS) sm90_mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                    sm90_mbar_expect(bfull + s, PGM90_TMA_TXB);
                    const uint32_t bar = sm90_su32(bfull + s);
                    sm90_tma2d(sm90_su32(As + s * PGM90_A8BUF), mapA, ks * PGM90_BK8,
                               (int)a_row0 + tm, bar);
                    sm90_tma2d(sm90_su32(Bs + s * PGM90_B8BUF), mapB, ks * PGM90_BK8, tn, bar);
                }
            }
        }
    } else {
        const int lt = tid - 128;
        const int wiw = lt >> 5, lane = lt & 31;
        int st = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            int tmi, tni;
            sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
            const int tm = tmi * PGM90_BM;
            const int tn = tni * PGM90_BN;
            float acc[2][PGM90_NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                sm90_mbar_wait(bfull + s, (st / NS) & 1);
                const uint8_t* Ac = As + s * PGM90_A8BUF;
                const uint8_t* Bc = Bs + s * PGM90_B8BUF;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    const uint64_t db = sm90_desc(Bc + sub * 32);
#pragma unroll
                    for (int sl = 0; sl < 2; sl++)
                        wgmma_m64n128k32(acc[sl],
                                         sm90_desc(Ac + sl * 64 * PGM90_BK8 + sub * 32), db, sd);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);
                prev = s;
            }
            sm90_wg_wait<0>();
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);

            const int c0 = tn + 2 * (lane & 3);
#pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                const int r0 = tm + sl * 64 + wiw * 16 + (lane >> 2);
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
                                C[(size_t)rr * n + cc] = __float2bfloat16(
                                    acc[sl][4 * g + 2 * hi + lo] * as * wscale[cc]);
                        }
                    }
            }
        }
    }
    __syncthreads();
    if (PROD && tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
#if PGM90_WS_BN256
/* BN=256 twin of the role body above (see PGM90_WSB_* for the tile derivation). */
template <bool PROD>
static __device__ void d_gemm_w8a8_sm90_tma_ws_role256(__nv_bfloat16* __restrict__ C,
                                                       const void* mapA, const void* mapB,
                                                       const float* __restrict__ ascale,
                                                       const float* __restrict__ wscale,
                                                       unsigned m, unsigned n, unsigned k,
                                                       unsigned a_row0, unsigned slice,
                                                       unsigned nblk, __nv_bfloat16* arena) {
    if (sm90_bad_k(k, 16u)) return;
    constexpr int NS = PGM90_WSB_NS;
    uint64_t* bfull = (uint64_t*)arena;
    uint64_t* bempty = bfull + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_TMA_MBAR_BF16);
    uint8_t* As = base;
    uint8_t* Bs = base + NS * PGM90_WSB_ABUF;
    const int tid = (int)threadIdx.x;

    if (PROD && tid < NS) {
        sm90_mbar_init(bfull + tid, 1);
        sm90_mbar_init(bempty + tid, 1);
    }
    __syncthreads();

    const int tiles_m = ((int)m + PGM90_WSB_BM - 1) / PGM90_WSB_BM;
    const int tiles_n = ((int)n + PGM90_WSB_BN - 1) / PGM90_WSB_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = ((int)k + PGM90_BK8 - 1) / PGM90_BK8;

    if (PROD) {
        if (tid == 0) {
            int st = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                const int tm = (tile / tiles_n) * PGM90_WSB_BM;
                const int tn = (tile % tiles_n) * PGM90_WSB_BN;
                for (int ks = 0; ks < ksteps; ks++, st++) {
                    const int s = st % NS;
                    if (st >= NS) sm90_mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                    sm90_mbar_expect(bfull + s, PGM90_WSB_TXB);
                    const uint32_t bar = sm90_su32(bfull + s);
                    sm90_tma2d(sm90_su32(As + s * PGM90_WSB_ABUF), mapA, ks * PGM90_BK8,
                               (int)a_row0 + tm, bar);
                    uint8_t* bs = Bs + s * PGM90_WSB_BBUF;
                    sm90_tma2d(sm90_su32(bs), mapB, ks * PGM90_BK8, tn, bar);
                    sm90_tma2d(sm90_su32(bs + 128 * PGM90_BK8), mapB, ks * PGM90_BK8, tn + 128,
                               bar);
                }
            }
        }
    } else {
        const int lt = tid - 128;
        const int wiw = lt >> 5, lane = lt & 31;
        int st = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            const int tm = (tile / tiles_n) * PGM90_WSB_BM;
            const int tn = (tile % tiles_n) * PGM90_WSB_BN;
            float acc[2 * PGM90_NACC]; /* 128: m64n256 per-thread accumulators */
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                sm90_mbar_wait(bfull + s, (st / NS) & 1);
                const uint8_t* Ac = As + s * PGM90_WSB_ABUF;
                const uint8_t* Bc = Bs + s * PGM90_WSB_BBUF;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < PGM90_KSUB8; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    wgmma_m64n256k32(acc, sm90_desc(Ac + sub * 32), sm90_desc(Bc + sub * 32),
                                     sd);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);
                prev = s;
            }
            sm90_wg_wait<0>();
            if (prev >= 0 && lt == 0) sm90_mbar_arrive(bempty + prev);

            const int c0 = tn + 2 * (lane & 3);
            const int r0 = tm + wiw * 16 + (lane >> 2);
#pragma unroll
            for (int g = 0; g < PGM90_WSB_BN / 8; g++)
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
        }
    }
    __syncthreads();
    if (PROD && tid < NS) {
        sm90_mbar_inval(bfull + tid);
        sm90_mbar_inval(bempty + tid);
    }
    __syncthreads();
}
#endif /* PGM90_WS_BN256 */
#endif /* PLOW_NV_SEG_WS_ENTRY */
#endif /* PLOW_NV_SEG_GEMM */

#if PGM90_TMA_HAS_GLU
static __device__ void d_gemm_glu_w8a8_sm90_tma(__nv_bfloat16* C, const void* mA, const void* mG,
                                                const void* mU, const float* ascale,
                                                const float* sg, const float* su, unsigned m,
                                                unsigned n, unsigned k, unsigned act,
                                                unsigned slice, unsigned nblk,
                                                __nv_bfloat16* arena) {
    d_gemm_glu_sm90_tma_body<true>(C, mA, mG, mU, ascale, sg, su, m, n, k, act, slice, nblk,
                                   arena);
}
#endif /* PGM90_TMA_HAS_GLU */
#endif /* PLOW_NV_W8A8 || PGM90_UNI_BN256 */

#endif /* PLOW_NV_TMA_GEMM */

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
        int tmi, tni;
        sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
        const int tm = tmi * PGM90_BM;
        const int tn = tni * PGM90_BN;
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


#ifndef PLOW_NV_GEMV_XREG
#define PLOW_NV_GEMV_XREG 0
#endif
#if PLOW_NV_GEMV_XREG
template <unsigned K>
__device__ __forceinline__ void d_gemv_sm90_xreg(__nv_bfloat16* C,
    const __nv_bfloat16* x, const __nv_bfloat16* W, unsigned N,
    unsigned slice, unsigned nblk) {
    static_assert(K == 5120 || K == 6144);
    constexpr unsigned chunks = K / GV_STEP;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    bf16v8 xv[chunks];
#pragma unroll
    for (unsigned c = 0; c < chunks; c++)
        xv[c] = ld_glob8(x + c * GV_STEP + lane * 8u);
    for (unsigned n = first + warp; n < end; n += blockDim.x / 32u) {
        const __nv_bfloat16* row = W + (size_t)n * K;
        float acc = 0.f;
#pragma unroll
        for (unsigned c = 0; c < chunks; c += GV_UNROLL) {
            bf16v8 weights[GV_UNROLL];
#pragma unroll
            for (unsigned u = 0; u < GV_UNROLL; u++)
                if (c + u < chunks)
                    weights[u] = ld_glob8(row + (c + u) * GV_STEP + lane * 8u);
#pragma unroll
            for (unsigned u = 0; u < GV_UNROLL; u++)
                if (c + u < chunks) acc = dot8(weights[u], xv[c + u], acc);
        }
        const float total = warp_sum32(acc);
        if (lane == 0) C[n] = __float2bfloat16(total);
    }
}
#endif



#ifndef PLOW_NV_GEMV_KPANEL
#define PLOW_NV_GEMV_KPANEL 0
#endif
#ifndef PLOW_NV_GEMV_KPANEL_F32
#define PLOW_NV_GEMV_KPANEL_F32 0
#endif
#if PLOW_NV_GEMV_KPANEL
__device__ __forceinline__ void d_gemv_sm90_kpanel(__nv_bfloat16* C,
    const __nv_bfloat16* x, const __nv_bfloat16* W, unsigned slice, unsigned nblk) {
    constexpr unsigned K = 17408, N = 5120, panel_chunks = 8;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    // Dispatch guarantees 256 threads and <=40 owned rows: at most five rows/warp.
    float acc[5] = {};
#pragma unroll 1
    for (unsigned panel = 0; panel < K; panel += panel_chunks * GV_STEP) {
#if PLOW_NV_GEMV_KPANEL_F32
        float xv[panel_chunks][8];
#pragma unroll
        for (unsigned c = 0; c < panel_chunks; c++) {
            if (panel + c * GV_STEP < K) {
                const bf16v8 packed = ld_glob8(x + panel + c * GV_STEP + lane * 8u);
#pragma unroll
                for (unsigned i = 0; i < 8; i++) xv[c][i] = __bfloat162float(packed.x[i]);
            }
        }
#else
        bf16v8 xv[panel_chunks];
#pragma unroll
        for (unsigned c = 0; c < panel_chunks; c++)
            if (panel + c * GV_STEP < K)
                xv[c] = ld_glob8(x + panel + c * GV_STEP + lane * 8u);
#endif
#pragma unroll
        for (unsigned r = 0; r < 5; r++) {
            const unsigned n = first + warp + r * 8u;
            if (n >= end) continue;
            const __nv_bfloat16* row = W + (size_t)n * K + panel;
#pragma unroll
            for (unsigned c = 0; c < panel_chunks; c += 8) {
                bf16v8 weights[8];
#pragma unroll
                for (unsigned u = 0; u < 8; u++)
                    if (panel + (c + u) * GV_STEP < K)
                        weights[u] = ld_glob8(row + (c + u) * GV_STEP + lane * 8u);
#pragma unroll
                for (unsigned u = 0; u < 8; u++)
                    if (panel + (c + u) * GV_STEP < K) {
#if PLOW_NV_GEMV_KPANEL_F32
#pragma unroll
                        for (unsigned i = 0; i < 8; i++)
                            acc[r] = fmaf(__bfloat162float(weights[u].x[i]), xv[c + u][i], acc[r]);
#else
                        acc[r] = dot8(weights[u], xv[c + u], acc[r]);
#endif
                    }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < 5; r++) {
        const unsigned n = first + warp + r * 8u;
        if (n < end) {
            const float total = warp_sum32(acc[r]);
            if (lane == 0) C[n] = __float2bfloat16(total);
        }
    }
}
#endif


#ifndef PLOW_NV_GEMV_M16_MMA
#define PLOW_NV_GEMV_M16_MMA 0
#endif
#if PLOW_NV_GEMV_M16_MMA

#ifndef PLOW_NV_GEMV_M16_PIPE
#define PLOW_NV_GEMV_M16_PIPE 0
#endif
#if PLOW_NV_GEMV_M16_PIPE
static __device__ void d_gemv_sm90_m16_pipeline(__nv_bfloat16* C, const __nv_bfloat16* x,
    const __nv_bfloat16* W, unsigned N, unsigned K, unsigned slice,
    unsigned nblk, __nv_bfloat16* arena) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    auto* memory = (__nv_bfloat16*)(((uintptr_t)arena + 15u) & ~(uintptr_t)15u) + warp * 768;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    for (unsigned col = first + warp * 8; col < end; col += 64) {
        float acc[4] = {};
        auto stage = [&](unsigned kb, unsigned buffer) {
            auto* a = memory + buffer * 384;
            auto* b = a + 256;
            const unsigned row = lane / 2, k = (lane & 1) * 8;
            sm90_cp16(a + row * 16 + k, x + (size_t)row * K + kb + k, 16);
            if (lane < 16) {
                const bool valid = col + row < end;
                const auto* source = valid ? W + (size_t)(col + row) * K + kb + k : W;
                sm90_cp16(b + row * 16 + k, source, valid ? 16 : 0);
            }
            sm90_cp_commit();
        };
        stage(0, 0);
        for (unsigned kb = 0; kb < K; kb += 16) {
            const unsigned current = (kb / 16) & 1u;
            if (kb + 16 < K) stage(kb + 16, current ^ 1u);
            // The empty tail group lets wait<1> retire the final data group.
            else sm90_cp_commit();
            sm90_cp_wait<1>();
            __syncwarp();
            auto* a = memory + current * 384;
            auto* b = a + 256;
            unsigned af[4], bf[2];
            const unsigned ap = (unsigned)__cvta_generic_to_shared(
                a + (lane % 16) * 16 + (lane / 16) * 8);
            const unsigned bp = (unsigned)__cvta_generic_to_shared(
                b + (lane & 7) * 16 + ((lane >> 3) & 1) * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(ap));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                : "=r"(bf[0]), "=r"(bf[1]) : "r"(bp));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
            __syncwarp();
        }
        sm90_cp_wait<0>();
        const unsigned row = lane >> 2, column = col + (lane & 3) * 2;
#pragma unroll
        for (unsigned hi = 0; hi < 2; hi++)
#pragma unroll
            for (unsigned lo = 0; lo < 2; lo++)
                if (column + lo < end)
                    C[(size_t)(row + hi * 8) * N + column + lo] =
                        __float2bfloat16(acc[hi * 2 + lo]);
        __syncwarp();
    }
}
#endif

static __device__ void d_gemv_sm90_m16(__nv_bfloat16* C, const __nv_bfloat16* x,
    const __nv_bfloat16* W, unsigned N, unsigned K, unsigned slice,
    unsigned nblk, __nv_bfloat16* arena) {
#if PLOW_NV_GEMV_M16_PIPE
    d_gemv_sm90_m16_pipeline(C, x, W, N, K, slice, nblk, arena);
#else
    constexpr unsigned stride = 72;
    auto* a = (__nv_bfloat16*)(((uintptr_t)arena + 15u) & ~(uintptr_t)15u);
    auto* b = a + 16 * stride;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    for (unsigned col = first; col < end; col += 64) {
        float acc[4] = {};
        for (unsigned kb = 0; kb < K; kb += 64) {
            for (unsigned v = tid; v < 16 * 8; v += 256) {
                const unsigned row = v / 8, k = v % 8 * 8;
                sm90_cp16(a + row * stride + k, x + (size_t)row * K + kb + k, 16);
            }
            for (unsigned v = tid; v < 64 * 8; v += 256) {
                const unsigned row = v / 8, k = v % 8 * 8;
                const bool valid = col + row < end;
                const auto* source = valid ? W + (size_t)(col + row) * K + kb + k : W;
                sm90_cp16(b + row * stride + k, source, valid ? 16 : 0);
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
            __syncthreads();
#pragma unroll
            for (unsigned kk = 0; kk < 64; kk += 16) {
                unsigned af[4], bf[2];
                const unsigned ap = (unsigned)__cvta_generic_to_shared(
                    a + (lane % 16) * stride + kk + (lane / 16) * 8);
                const unsigned bp = (unsigned)__cvta_generic_to_shared(
                    b + (warp * 8 + (lane & 7)) * stride + kk + ((lane >> 3) & 1) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(ap));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(bf[0]), "=r"(bf[1]) : "r"(bp));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
            }
            __syncthreads();
        }
        const unsigned row = lane >> 2, column = col + warp * 8 + (lane & 3) * 2;
#pragma unroll
        for (unsigned hi = 0; hi < 2; hi++)
#pragma unroll
            for (unsigned lo = 0; lo < 2; lo++)
                if (column + lo < end)
                    C[(size_t)(row + hi * 8) * N + column + lo] =
                        __float2bfloat16(acc[hi * 2 + lo]);
        __syncthreads();
    }
#endif
}
#endif

#endif /* PLOW_OP_GEMM_SM90_CUH */
