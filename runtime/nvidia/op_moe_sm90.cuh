/* op_moe_sm90.cuh — Hopper (sm_90a) WGMMA fork of the GROUPED MoE prefill GEMMs.
 *
 * Included ONLY from op_moe.cuh under PLOW_NV_HOPPER. It defines the SAME four bodies with
 * BYTE-IDENTICAL signatures, so interp_sm120.cu's dispatch is untouched:
 *     d_moe_group_glu_gemma_pf       d_moe_group_down_gemma_pf
 *     d_moe_group_glu_gemma_pf_w8a8  d_moe_group_down_gemma_pf_w8a8   (PLOW_NV_W8A8)
 *
 * WHY: sm_90a has no ldmatrix-fed 2x-rate fp8 mma (mma.sync.m16n8k32.e4m3 is EMULATED as
 * 12x F2FP + 2x HMMA there) and its bf16 mma.sync tops out well below the warpgroup MMA.
 * wgmma.mma_async is the only route to Hopper's real tensor-core rate.
 *
 * STRUCTURE (validated in runtime/nvidia/experiments/wgmma_moe_group_probe.cu, 1.50-1.60x over
 * mma.sync): the grouped expert indirection resolves ENTIRELY BEFORE the mainloop — the flat
 * (tile -> expert, m_tile, n_tile) worklist yields rowbase / Wexp, after which the mainloop is a
 * plain dense TN GEMM that never sees an expert. That is exactly the shipped rowoff[e]/tilep[e]
 * contract, so PGM_BM/PGM_BN and d_moe_align_gemma_pf are UNCHANGED.
 *
 * TILE (re-tuned from the probe's 128-thread/BM=64 shape to the megakernel's fixed
 * PLOW_NV_THREADS=256 = 2 warpgroups):
 *   BM=PGM_BM=128, BN=PGM_BN=128. Warpgroup wg owns A rows [64*wg, 64*wg+64) and issues
 *   wgmma.m64n128; both warpgroups read the SAME B tile. 64 f32 acc/thread (plain), 128 (GLU:
 *   gate+up) — identical to the mma.sync path's PGM_MFRAG*PGM_NFRAG*4 counts.
 *   BK is FIXED by the 128-byte swizzle: one logical row must be exactly 128 B, i.e. 64 bf16 or
 *   128 e4m3. See sm90_wgmma.cuh — the swizzle is a PERFORMANCE requirement (~2.2x) and its
 *   1024-byte tile alignment is a CORRECTNESS requirement (silently wrong otherwise).
 *
 * RAGGED TILES: unchanged semantics. A rows are cp.async src-size-0 zero-filled (pad rows /
 * K tail), and the epilogue masks the N tail; pad rows are skipped/zeroed exactly as before.
 *
 * STAGE 1: cp.async staging only. No TMA (that needs a CUtensorMap ABI extension).
 */
#ifndef PLOW_NV_OP_MOE_SM90_CUH
#define PLOW_NV_OP_MOE_SM90_CUH

#include "sm90_wgmma.cuh"

/* ---- tile geometry ---------------------------------------------------------------------- */
#define MOE90_BM PGM_BM /* 128 — MUST equal the align op's tile height (meta contract) */
#define MOE90_BN PGM_BN /* 128 — one wgmma n-block */
#define MOE90_BK 64     /* bf16 elems/row = 128 B = one 128B-swizzle atom row */
#define MOE90_BK8 128   /* e4m3 elems/row = 128 B */
#define MOE90_STAGES 3     /* plain (down) ring depth */
#ifndef MOE90_DOWN_PIPE
#define MOE90_DOWN_PIPE 1 /* overlap the mma with the next stage: bit-exact, -10% down */
#endif
/* DOWN on the m64n256 tile. DOWN's N is H (2816 = 22 n128 tiles) and its K only I_moe (11
 * k-steps), so the gathered A rows are RE-READ once per n-tile: 22x. One n256 stream holds 128
 * accumulators — the register footprint GLU's two n128 streams already have — and halves the
 * n-tiles, the A re-reads and the tile count. The B tile doubles (32 KiB), so the ring goes
 * 2-deep to stay inside GLU's arena claim, which gives up MOE90_DOWN_PIPE (needs 3). */
#ifndef MOE90_DOWN_N256
#define MOE90_DOWN_N256 1
#endif
#define MOE90_DN_BN (MOE90_DOWN_N256 ? 256 : MOE90_BN)
#define MOE90_DN_BBUF (MOE90_DN_BN * MOE90_BK)
#define MOE90_DN_STAGES (MOE90_DOWN_N256 ? 2 : MOE90_STAGES)
#define MOE90_GLU_STAGES 2 /* gate/up ring depth (3 tiles per stage) */
#define MOE90_ABUF (MOE90_BM * MOE90_BK)   /* bf16 elems per staged A tile */
#define MOE90_BBUF (MOE90_BN * MOE90_BK)   /* bf16 elems per staged B tile */
#define MOE90_A8BUF (MOE90_BM * MOE90_BK8) /* e4m3 BYTES per staged A tile */
#define MOE90_B8BUF (MOE90_BN * MOE90_BK8)
#define MOE90_CH (MOE90_BK / 8)    /* 16-byte chunks per bf16 row */
#define MOE90_CH8 (MOE90_BK8 / 16) /* 16-byte chunks per e4m3 row */
#define MOE90_NBLK (MOE90_BN / 8)  /* wgmma n-blocks in the accumulator */

/* ---- ARENA CLAIM (reported delta) --------------------------------------------------------
 * Both rings land on the same size: plain 3*(A+B) and GLU 2*(A+2B) = 49152 bf16 = 96 KiB, plus
 * 512 bf16 (1024 B) so the tile base can be rounded up to the swizzle's 1024 B alignment.
 * That is larger than op_gemm.cuh's mma.sync claim (PGM_ARENA_BF16 = 30720 bf16 = 60 KiB), and
 * op_gemm.cuh is owned elsewhere, so the claim is raised HERE. op_moe.cuh is included after
 * op_gemm.cuh and before interp_sm120.cu's PLOW_NV_PRE_B / the op-test's `smg` are *used*, so
 * the override propagates to both. HOPPER-ONLY: sm_120a never sees it. */
#ifndef MOE90_NB
#if MOE90_HALF
/* a mixed pair stages one B set per warpgroup: down A+2B, gate/up A+4B per ring slot. */
#define MOE90_NB 2
#else
#define MOE90_NB 1
#endif
#endif
#define PGM_MOE_ARENA_SM90                                                                         \
    ((MOE90_GLU_STAGES * (MOE90_ABUF + 2 * MOE90_NB * MOE90_BBUF) >                                \
              MOE90_STAGES * (MOE90_ABUF + MOE90_NB * MOE90_BBUF)                                  \
          ? MOE90_GLU_STAGES * (MOE90_ABUF + 2 * MOE90_NB * MOE90_BBUF)                            \
          : MOE90_STAGES * (MOE90_ABUF + MOE90_NB * MOE90_BBUF)) +                                 \
     512)
static_assert(MOE90_DN_STAGES * (MOE90_ABUF + MOE90_NB * MOE90_DN_BBUF) + 512 <= PGM_MOE_ARENA_SM90,
              "the n256 DOWN ring must fit the existing MoE arena claim");
/* GROW-ONLY: never shrink whatever op_gemm.cuh (a sibling-owned file) already claims. */
#if PGM_ARENA_BF16 < PGM_MOE_ARENA_SM90
#undef PGM_ARENA_BF16
#define PGM_ARENA_BF16 PGM_MOE_ARENA_SM90
#endif
static_assert(MOE90_STAGES * (MOE90_ABUF + MOE90_NB * MOE90_BBUF) + 512 <= PGM_ARENA_BF16,
              "sm90 MoE plain ring must fit the bf16 arena claim");
static_assert(MOE90_GLU_STAGES * (MOE90_ABUF + 2 * MOE90_NB * MOE90_BBUF) + 512 <= PGM_ARENA_BF16,
              "sm90 MoE GLU ring must fit the bf16 arena claim");
static_assert(MOE90_STAGES * (MOE90_A8BUF + MOE90_B8BUF) + 1024 <= 2 * PGM_ARENA_BF16,
              "sm90 MoE w8a8 plain ring must fit the bf16 arena claim");
static_assert(MOE90_GLU_STAGES * (MOE90_A8BUF + 2 * MOE90_B8BUF) + 1024 <= 2 * PGM_ARENA_BF16,
              "sm90 MoE w8a8 GLU ring must fit the bf16 arena claim");
static_assert(MOE90_BM == 128 && MOE90_BN == 128, "sm90 MoE tile assumes 2 warpgroups x m64n128");
static_assert(PLOW_NV_THREADS == 256u, "sm90 MoE mainloop is exactly 2 warpgroups");

/* ---- cp.async stagers (128B-swizzled, K-major) -------------------------------------------
 * Every tile is [rows][BK] with the row XOR-swizzled by sm90_swz_off; a src_bytes=0 copy
 * zero-fills without touching gmem, which is how pad rows and the K tail are handled. */

/* A, GATHERED: gathered row `rowbase+row` reads source token rowsrc[rowbase+row]. */
__device__ __forceinline__ void moe90_stage_a_gather(__nv_bfloat16* Ad,
                                                     const __nv_bfloat16* __restrict__ A,
                                                     const unsigned* __restrict__ rowsrc, int tid,
                                                     int rowbase, int kbase, unsigned k) {
    for (int L = tid; L < MOE90_BM * MOE90_CH; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH, c = L % MOE90_CH;
        const unsigned src = rowsrc[rowbase + row];
        const int kk = kbase + c * 8;
        const bool in = (src != PLOW_EXPERT_UNUSED) && (kk + 8 <= (int)k);
        const __nv_bfloat16* g = in ? A + (size_t)src * k + kk : A;
        sm90_cp16(&Ad[sm90_swz_off<MOE90_BK, 8>(row, c)], g, in ? 16 : 0);
    }
}
/* A, CONTIGUOUS: rows rowbase..rowbase+BM-1 of a [rows][k] matrix (all rows valid). */
__device__ __forceinline__ void moe90_stage_a(__nv_bfloat16* Ad,
                                              const __nv_bfloat16* __restrict__ A, int tid,
                                              int rowbase, int kbase, unsigned k) {
    for (int L = tid; L < MOE90_BM * MOE90_CH; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH, c = L % MOE90_CH;
        const int kk = kbase + c * 8;
        const bool in = (kk + 8 <= (int)k);
        const __nv_bfloat16* g = in ? A + (size_t)(rowbase + row) * k + kk : A;
        sm90_cp16(&Ad[sm90_swz_off<MOE90_BK, 8>(row, c)], g, in ? 16 : 0);
    }
}
/* B: weight [n][k], tile row = output channel tn+row. ROWS = the mma's N (128, or 256 for the
 * n256 DOWN tile). */
template <int ROWS = MOE90_BN>
__device__ __forceinline__ void moe90_stage_b(__nv_bfloat16* Bd,
                                              const __nv_bfloat16* __restrict__ B, int tid, int tn,
                                              int kbase, unsigned n, unsigned k) {
    for (int L = tid; L < ROWS * MOE90_CH; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH, c = L % MOE90_CH;
        const int nn = tn + row, kk = kbase + c * 8;
        const bool in = (nn < (int)n) && (kk + 8 <= (int)k);
        const __nv_bfloat16* g = in ? B + (size_t)nn * k + kk : B;
        sm90_cp16(&Bd[sm90_swz_off<MOE90_BK, 8>(row, c)], g, in ? 16 : 0);
    }
}

/* ---- the mainloop, shared by every arm ---------------------------------------------------
 * `Ad`/`Bd` are the LANDED stage buffers; wg selects the 64-row A sub-tile. A k16 (bf16) or
 * k32 (e4m3) substep advances the descriptor START ADDRESS by +32 B only. */
/* `live` rides the instruction's own predicate (setp.ne on the last operand): a warpgroup
 * whose 64 rows are ALL padding issues a predicated-off mma, so the commit/wait/sync
 * structure is byte-identical across warpgroups and only the tensor-core work is elided. */
template <int NACC>
__device__ __forceinline__ void moe90_wgmma_k(float (&d)[NACC], const __nv_bfloat16* Ad,
                                              const __nv_bfloat16* Bd, int wg, int live = 1) {
    const __nv_bfloat16* Aw = Ad + wg * 64 * MOE90_BK;
#pragma unroll
    for (int kk = 0; kk < MOE90_BK; kk += 16)
        wgmma_m64n128k16(d, sm90_desc(Aw + kk), sm90_desc(Bd + kk), live);
}

/* Valid rows in m-tile `mtile` of expert `e`: the align op pads each expert's segment to
 * BM, so only the LAST tile of a segment is partial. With 128 experts and a short chunk
 * every tile is that last tile — 8 valid rows of 128 at T=128, 64 at T=1024. */
__device__ __forceinline__ int moe90_tile_rows(const int* meta, int n_exp, int e, int mtile) {
    const int cnt = meta[n_exp + e];
    const int done = (mtile - meta[2 * n_exp + e]) * MOE90_BM;
    const int left = cnt - done;
    return left < MOE90_BM ? left : MOE90_BM;
}

/* ==========================================================================================
 * GROUPED GATE/UP GEMM + GLU  (PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF)
 * ========================================================================================== */
static __device__ void d_moe_group_glu_gemma_pf(__nv_bfloat16* __restrict__ fu,
                                                const __nv_bfloat16* __restrict__ xn2,
                                                const unsigned long long* __restrict__ ewt,
                                                const int* __restrict__ meta,
                                                const unsigned* __restrict__ row_token,
                                                unsigned I_moe, unsigned H, unsigned n_exp,
                                                unsigned act, unsigned slice, unsigned nblk,
                                                __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)I_moe + MOE90_BN - 1) / MOE90_BN;
    const unsigned K = H;
    const int ksteps = ((int)K + MOE90_BK - 1) / MOE90_BK;

    __nv_bfloat16* As = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* Bgs0 = As + MOE90_GLU_STAGES * MOE90_ABUF;
    __nv_bfloat16* Bus0 = Bgs0 + MOE90_GLU_STAGES * MOE90_BBUF;
    /* second B set (mixed pairs) sits AFTER the first, so a same-expert pair keeps the
     * original ring spacing — interleaving the sets cost 19% on all-full-tile routing. */
    const int bset = 2 * MOE90_GLU_STAGES * MOE90_BBUF;
    const int tid = threadIdx.x, wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

#if MOE90_HALF
    const int ntiles = (total_tiles + 1) / 2 * tiles_n; /* meta counts 64-row half-tiles */
#else
    const int ntiles = total_tiles * tiles_n;
#endif
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n;
        const int ntile = tile % tiles_n;
        const int tn = ntile * MOE90_BN;
#if MOE90_HALF
        /* half-tiles 2*mtile (warpgroup 0) and 2*mtile+1 (warpgroup 1) are row-contiguous, so A
         * stages exactly as before; only B is per warpgroup, and only when the experts differ. */
        const int rowbase = mtile * MOE90_BM;
        const int e = pgm_moe_expert_of_mtile(tilep, 2 * mtile, (int)n_exp);
        const int h1 = 2 * mtile + 1;
        int e1 = e; /* the next half-tile is this expert's or a later one's: step, don't search */
        while (h1 < total_tiles && e1 + 1 < (int)n_exp && tilep[e1 + 1] <= h1) e1++;
        const int mixed = e1 != e;
        const __nv_bfloat16* Wg1 = (const __nv_bfloat16*)(size_t)ewt[(size_t)e1 * 2 + 0];
        const __nv_bfloat16* Wu1 = Wg1 + (size_t)I_moe * H;
        const int bsel = mixed ? wg : 0; /* which staged B set this thread's warpgroup reads */
#else
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * MOE90_BM;
        const int bsel = 0;
#endif
        const __nv_bfloat16* Wg = (const __nv_bfloat16*)(size_t)ewt[(size_t)e * 2 + 0];
        const __nv_bfloat16* Wu = Wg + (size_t)I_moe * H;
#if defined(MOE90_DBG_NOMMA)
        const int live = 0; /* harness attribution: k-step cost with the tensor core OFF */
#elif MOE90_HALF
        const int live = 2 * mtile + wg < total_tiles;
#else
        const int live = moe90_tile_rows(meta, (int)n_exp, e, mtile) > wg * 64;
#endif
        /* ---- from here down the mainloop is dense; nothing is MoE-aware ---- */

        float accg[64], accu[64];
#pragma unroll
        for (int i = 0; i < 64; i++) { accg[i] = 0.f; accu[i] = 0.f; }

        auto stage = [&](int ks, int buf) {
#if defined(MOE90_DBG_NOSTAGE)
            if (ks > 0) return; /* harness attribution: k-step cost with staging OFF */
#endif
            moe90_stage_a_gather(As + buf * MOE90_ABUF, xn2, row_token, tid, rowbase,
                                 ks * MOE90_BK, K);
#if defined(MOE90_DBG_NOSTAGE_B)
            if (ks > 0) return; /* harness attribution: the bound on a free B load engine */
#endif
            moe90_stage_b(Bgs0 + buf * MOE90_BBUF, Wg, tid, tn, ks * MOE90_BK, I_moe, K);
            moe90_stage_b(Bus0 + buf * MOE90_BBUF, Wu, tid, tn, ks * MOE90_BK, I_moe, K);
#if MOE90_HALF
            if (mixed) {
                moe90_stage_b(Bgs0 + bset + buf * MOE90_BBUF, Wg1, tid, tn, ks * MOE90_BK,
                              I_moe, K);
                moe90_stage_b(Bus0 + bset + buf * MOE90_BBUF, Wu1, tid, tn, ks * MOE90_BK,
                              I_moe, K);
            }
#endif
        };
#pragma unroll
        for (int s = 0; s < MOE90_GLU_STAGES - 1; s++) {
            if (s < ksteps) stage(s, s);
            sm90_cp_commit();
        }
        sm90_wg_fence(); /* accumulators were just written by register ops */

        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + MOE90_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % MOE90_GLU_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<MOE90_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % MOE90_GLU_STAGES;
            moe90_wgmma_k(accg, As + cb * MOE90_ABUF, Bgs0 + bsel * bset + cb * MOE90_BBUF, wg,
                          live);
            moe90_wgmma_k(accu, As + cb * MOE90_ABUF, Bus0 + bsel * bset + cb * MOE90_BBUF, wg,
                          live);
            sm90_wg_commit();
            /* wait<0>: the next iteration refills THIS buffer (ring depth == prefetch+1). */
            sm90_wg_wait<0>();
            __syncthreads();
        }

#pragma unroll
        for (int g = 0; g < MOE90_NBLK; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = wg * 64 + sm90_acc_row(wiw, lane, hi);
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int r = sm90_acc_reg(g, hi, lo);
                    const int cc = tn + sm90_acc_col(g, lane, lo);
                    if (cc < (int)I_moe) {
                        const float gv = accg[r];
                        const float a =
                            (act == PLOW_ACT_SILU_) ? act_silu(gv) : act_gelu_tanh(gv);
                        fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(a * accu[r]);
                    }
                }
            }
        __syncthreads();
    }
}

/* ==========================================================================================
 * GROUPED DOWN GEMM + gate-scale + scatter  (PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF)
 * ========================================================================================== */
static __device__ void d_moe_group_down_gemma_pf(float* __restrict__ part,
                                                 const __nv_bfloat16* __restrict__ fu,
                                                 const unsigned long long* __restrict__ ewt,
                                                 const int* __restrict__ meta,
                                                 const unsigned* __restrict__ row_partidx,
                                                 const float* __restrict__ row_gate, unsigned H,
                                                 unsigned I_moe, unsigned n_exp, unsigned slice,
                                                 unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)H + MOE90_DN_BN - 1) / MOE90_DN_BN;
    const unsigned K = I_moe;
    const int ksteps = ((int)K + MOE90_BK - 1) / MOE90_BK;

    __nv_bfloat16* As = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* Bs = As + MOE90_DN_STAGES * MOE90_ABUF;
    const int tid = threadIdx.x, wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

#if MOE90_HALF
    const int ntiles = (total_tiles + 1) / 2 * tiles_n; /* see d_moe_group_glu_gemma_pf */
#else
    const int ntiles = total_tiles * tiles_n;
#endif
    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n;
        const int ntile = tile % tiles_n;
        const int tn = ntile * MOE90_DN_BN;
#if MOE90_HALF
        const int rowbase = mtile * MOE90_BM;
        const int e = pgm_moe_expert_of_mtile(tilep, 2 * mtile, (int)n_exp);
        const int h1 = 2 * mtile + 1;
        int e1 = e; /* the next half-tile is this expert's or a later one's: step, don't search */
        while (h1 < total_tiles && e1 + 1 < (int)n_exp && tilep[e1 + 1] <= h1) e1++;
        const int mixed = e1 != e;
        const __nv_bfloat16* Wd1 = (const __nv_bfloat16*)(size_t)ewt[(size_t)e1 * 2 + 1];
        const int bsel = mixed ? wg : 0;
        const int live = 2 * mtile + wg < total_tiles;
#else
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * MOE90_BM;
        const int bsel = 0;
        const int live = moe90_tile_rows(meta, (int)n_exp, e, mtile) > wg * 64;
#endif
        const __nv_bfloat16* Wd = (const __nv_bfloat16*)(size_t)ewt[(size_t)e * 2 + 1];

        float acc[MOE90_DN_BN / 2];
#pragma unroll
        for (int i = 0; i < MOE90_DN_BN / 2; i++) acc[i] = 0.f;

        auto stage = [&](int ks, int buf) {
            moe90_stage_a(As + buf * MOE90_ABUF, fu, tid, rowbase, ks * MOE90_BK, K);
            moe90_stage_b<MOE90_DN_BN>(Bs + buf * MOE90_DN_BBUF, Wd, tid, tn, ks * MOE90_BK, H, K);
#if MOE90_HALF
            if (mixed)
                moe90_stage_b<MOE90_DN_BN>(Bs + (MOE90_DN_STAGES + buf) * MOE90_DN_BBUF, Wd1, tid,
                                           tn, ks * MOE90_BK, H, K);
#endif
        };
#if MOE90_DOWN_PIPE && !MOE90_DOWN_N256
        /* PIPELINED: prefetch ONE stage and wait only on the PREVIOUS mma group, so group ks
         * runs while stage ks+1 lands. Safe on the 3-deep ring: iteration ks stages buffer
         * (ks+1)%3 and computes on ks%3, and the buffer it will restage next, (ks+2)%3, is
         * group ks-1's, which wait<1> has just retired. */
        if (0 < ksteps) stage(0, 0);
        sm90_cp_commit();
        sm90_wg_fence();

        for (int ks = 0; ks < ksteps; ks++) {
            if (ks + 1 < ksteps) stage(ks + 1, (ks + 1) % MOE90_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<1>();
            __syncthreads();
            const int cb = ks % MOE90_STAGES;
            moe90_wgmma_k(acc, As + cb * MOE90_ABUF,
                          Bs + (bsel * MOE90_STAGES + cb) * MOE90_BBUF, wg, live);
            sm90_wg_commit();
            sm90_wg_wait<1>();
            __syncthreads();
        }
        sm90_wg_wait<0>();
#else
#pragma unroll
        for (int s = 0; s < MOE90_DN_STAGES - 1; s++) {
            if (s < ksteps) stage(s, s);
            sm90_cp_commit();
        }
        sm90_wg_fence();

        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + MOE90_DN_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % MOE90_DN_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<MOE90_DN_STAGES - 1>();
            __syncthreads();
            const int cb = ks % MOE90_DN_STAGES;
#if MOE90_DOWN_N256
            {
                const __nv_bfloat16* Aw = As + cb * MOE90_ABUF + wg * 64 * MOE90_BK;
                const __nv_bfloat16* Bw = Bs + (bsel * MOE90_DN_STAGES + cb) * MOE90_DN_BBUF;
#pragma unroll
                for (int kk = 0; kk < MOE90_BK; kk += 16)
                    wgmma_m64n256k16(acc, sm90_desc(Aw + kk), sm90_desc(Bw + kk), live);
            }
#else
            moe90_wgmma_k(acc, As + cb * MOE90_ABUF,
                          Bs + (bsel * MOE90_STAGES + cb) * MOE90_BBUF, wg, live);
#endif
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
        }
#endif

#pragma unroll
        for (int g = 0; g < MOE90_DN_BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = wg * 64 + sm90_acc_row(wiw, lane, hi);
                const unsigned pidx = row_partidx[rowbase + rr];
                if (pidx == PLOW_EXPERT_UNUSED) continue;
                const float gate = row_gate[rowbase + rr];
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int r = sm90_acc_reg(g, hi, lo);
                    const int cc = tn + sm90_acc_col(g, lane, lo);
                    if (cc < (int)H) part[(size_t)pidx * H + cc] = gate * acc[r];
                }
            }
        __syncthreads();
    }
}

/* ==========================================================================================
 * T-TOKEN ROUTER through the tensor core  (PLOW_DOP_MOE_ROUTER_GEMMA_PF, T >= MOE90_RT_MIN_T)
 *
 * One m-tile = 128 tokens; logits = h2 [128,H] x proj^T [n_exp<=128,H] is ONE m128n128 wgmma
 * tile swept over K=H. h2 = resid*invrms*scale*root is computed by the threads straight into
 * the swizzled A stage as bf16 — the dtype vLLM's router feeds its GateLinear — and the f32
 * accumulators are the fp32 logits that layer asks for. proj stages by cp.async like any B.
 * After the k sweep the ring is dead, so it holds the [128][128] f32 logits for the
 * thread-per-token softmax/top-k (the scalar body's exact selection code).
 * NOT bit-identical to the scalar body: bf16 h2 and the core's K association. Selection is
 * checked against it in runtime/tests/moe_group_gemma26b_repro.cu.
 * ========================================================================================== */
static_assert(MOE90_STAGES * (MOE90_ABUF + MOE90_BBUF) * sizeof(__nv_bfloat16) >=
                  (size_t)MOE90_BM * MOE90_BN * sizeof(float),
              "router logits reuse the dead stage ring");
static __device__ void moe90_router_gemma_pf(unsigned char* __restrict__ table,
                                             const bf16* __restrict__ resid,
                                             const bf16* __restrict__ proj,
                                             const bf16* __restrict__ scale,
                                             const bf16* __restrict__ pes, unsigned H,
                                             unsigned n_exp, unsigned k, unsigned T, float root,
                                             float eps, unsigned slice, unsigned nblk,
                                             float* __restrict__ arena) {
    const int mtiles = ((int)T + MOE90_BM - 1) / MOE90_BM;
    const int ksteps = ((int)H + MOE90_BK - 1) / MOE90_BK;
    __nv_bfloat16* As = (__nv_bfloat16*)sm90_align1024(arena);
    __nv_bfloat16* Bs = As + MOE90_STAGES * MOE90_ABUF;
    float* logit = (float*)As;
    const int tid = threadIdx.x, wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float inv[MOE90_BM];

    for (int tile = (int)slice; tile < mtiles; tile += (int)nblk) {
        const int rowbase = tile * MOE90_BM;

        /* weightless RMS: warp-per-row, 16 rounds x 8 warps = the 128 rows of the tile. */
        for (int i = 0; i < MOE90_BM / 8; i++) {
            const int row = warp * (MOE90_BM / 8) + i;
            const bool in = rowbase + row < (int)T;
            const bf16* r = resid + (size_t)(in ? rowbase + row : 0) * H;
            float part = 0.0f;
            for (unsigned h = (unsigned)lane; h < H; h += 32u) {
                const float v = __bfloat162float(r[h]);
                part += v * v;
            }
            part = plow_warp_sum(part);
            if (lane == 0) inv[row] = rsqrtf(part / (float)H + eps);
        }
        __syncthreads();

        float acc[64];
#pragma unroll
        for (int i = 0; i < 64; i++) acc[i] = 0.f;

        auto stage = [&](int ks, int buf) {
            __nv_bfloat16* Ad = As + buf * MOE90_ABUF;
            for (int L = tid; L < MOE90_BM * MOE90_CH; L += (int)PLOW_NV_THREADS) {
                const int row = L / MOE90_CH, c = L % MOE90_CH;
                const int kk = ks * MOE90_BK + c * 8;
                const bool in = rowbase + row < (int)T;
                const bf16* r = resid + (size_t)(in ? rowbase + row : 0) * H;
                const float iv = inv[row];
                alignas(16) __nv_bfloat16 v[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const bool ok = in && kk + j < (int)H;
                    const int h = ok ? kk + j : 0;
                    const float x = __bfloat162float(r[h]) * iv * __bfloat162float(scale[h]) * root;
                    v[j] = __float2bfloat16(ok ? x : 0.0f);
                }
                *(uint4*)(Ad + sm90_swz_off<MOE90_BK, 8>(row, c)) = *(const uint4*)v;
            }
            /* Generic shared stores must be visible to WGMMA's async proxy. */
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            moe90_stage_b(Bs + buf * MOE90_BBUF, proj, tid, 0, ks * MOE90_BK, n_exp, H);
        };
#pragma unroll
        for (int s = 0; s < MOE90_STAGES - 1; s++) {
            if (s < ksteps) stage(s, s);
            sm90_cp_commit();
        }
        sm90_wg_fence();

        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + MOE90_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % MOE90_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<MOE90_STAGES - 1>();
            __syncthreads();
            const int cb = ks % MOE90_STAGES;
            moe90_wgmma_k(acc, As + cb * MOE90_ABUF, Bs + cb * MOE90_BBUF, wg);
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
        }

#pragma unroll
        for (int g = 0; g < MOE90_NBLK; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = wg * 64 + sm90_acc_row(wiw, lane, hi);
#pragma unroll
                for (int lo = 0; lo < 2; lo++)
                    logit[rr * MOE90_BN + sm90_acc_col(g, lane, lo)] =
                        acc[sm90_acc_reg(g, hi, lo)];
            }
        __syncthreads();

        if (tid < MOE90_BM && rowbase + tid < (int)T)
            plow_moe_gemma_softmax_topk(table + (size_t)(rowbase + tid) * k * 8,
                                        logit + tid * MOE90_BN, pes, n_exp, k);
        __syncthreads();
    }
}

#if PLOW_NV_W8A8
/* ==========================================================================================
 * w8a8 (e4m3) twins. Hopper has NO native fp8 mma.sync, so wgmma.m64n128k32.f32.e4m3.e4m3 is
 * the only path to the fp8 tensor core here. Scale semantics are unchanged from the mma.sync
 * bodies: activation per-token/-row f32 scale x weight per-output-channel f32 scale.
 * NOTE (sm90_wgmma.cuh): the fp8 wgmma accumulator is not true f32 and its error grows with K.
 * ========================================================================================== */
__device__ __forceinline__ void moe90_stage_a8_gather(uint8_t* Ad, const uint8_t* __restrict__ A,
                                                      const unsigned* __restrict__ rowsrc, int tid,
                                                      int rowbase, int kbase, unsigned k) {
    for (int L = tid; L < MOE90_BM * MOE90_CH8; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH8, c = L % MOE90_CH8;
        const unsigned src = rowsrc[rowbase + row];
        const int kk = kbase + c * 16;
        const bool in = (src != PLOW_EXPERT_UNUSED) && (kk + 16 <= (int)k);
        const uint8_t* g = in ? A + (size_t)src * k + kk : A;
        sm90_cp16(&Ad[sm90_swz_off<MOE90_BK8, 16>(row, c)], g, in ? 16 : 0);
    }
}
__device__ __forceinline__ void moe90_stage_a8(uint8_t* Ad, const uint8_t* __restrict__ A, int tid,
                                               int rowbase, int kbase, unsigned k) {
    for (int L = tid; L < MOE90_BM * MOE90_CH8; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH8, c = L % MOE90_CH8;
        const int kk = kbase + c * 16;
        const bool in = (kk + 16 <= (int)k);
        const uint8_t* g = in ? A + (size_t)(rowbase + row) * k + kk : A;
        sm90_cp16(&Ad[sm90_swz_off<MOE90_BK8, 16>(row, c)], g, in ? 16 : 0);
    }
}
__device__ __forceinline__ void moe90_stage_b8(uint8_t* Bd, const uint8_t* __restrict__ B, int tid,
                                               int tn, int kbase, unsigned n, unsigned k) {
    for (int L = tid; L < MOE90_BN * MOE90_CH8; L += (int)PLOW_NV_THREADS) {
        const int row = L / MOE90_CH8, c = L % MOE90_CH8;
        const int nn = tn + row, kk = kbase + c * 16;
        const bool in = (nn < (int)n) && (kk + 16 <= (int)k);
        const uint8_t* g = in ? B + (size_t)nn * k + kk : B;
        sm90_cp16(&Bd[sm90_swz_off<MOE90_BK8, 16>(row, c)], g, in ? 16 : 0);
    }
}
template <int NACC>
__device__ __forceinline__ void moe90_wgmma8_k(float (&d)[NACC], const uint8_t* Ad,
                                               const uint8_t* Bd, int wg) {
    const uint8_t* Aw = Ad + wg * 64 * MOE90_BK8;
#pragma unroll
    for (int kk = 0; kk < MOE90_BK8; kk += 32)
        wgmma_m64n128k32(d, sm90_desc(Aw + kk), sm90_desc(Bd + kk), 1);
}

static __device__ void d_moe_group_glu_gemma_pf_w8a8(
        __nv_bfloat16* __restrict__ fu, const uint8_t* __restrict__ xq8,
        const float* __restrict__ ascale, const unsigned long long* __restrict__ ewt,
        const unsigned long long* __restrict__ est, const int* __restrict__ meta,
        const unsigned* __restrict__ row_token, unsigned I_moe, unsigned H, unsigned n_exp,
        unsigned act, unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)I_moe + MOE90_BN - 1) / MOE90_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = H;
    const int ksteps = ((int)K + MOE90_BK8 - 1) / MOE90_BK8;

    uint8_t* As = (uint8_t*)sm90_align1024(arena);
    uint8_t* Bg = As + MOE90_GLU_STAGES * MOE90_A8BUF;
    uint8_t* Bu = Bg + MOE90_GLU_STAGES * MOE90_B8BUF;
    const int tid = threadIdx.x, wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * MOE90_BM;
        const int tn = ntile * MOE90_BN;
        const uint8_t* Wg = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 0];
        const uint8_t* Wu = Wg + (size_t)I_moe * H;
        const float* sc = (const float*)(size_t)est[(size_t)e * 2 + 0]; /* [2*I_moe] */

        float accg[64], accu[64];
#pragma unroll
        for (int i = 0; i < 64; i++) { accg[i] = 0.f; accu[i] = 0.f; }

        auto stage = [&](int ks, int buf) {
            moe90_stage_a8_gather(As + buf * MOE90_A8BUF, xq8, row_token, tid, rowbase,
                                  ks * MOE90_BK8, K);
            moe90_stage_b8(Bg + buf * MOE90_B8BUF, Wg, tid, tn, ks * MOE90_BK8, I_moe, K);
            moe90_stage_b8(Bu + buf * MOE90_B8BUF, Wu, tid, tn, ks * MOE90_BK8, I_moe, K);
        };
#pragma unroll
        for (int s = 0; s < MOE90_GLU_STAGES - 1; s++) {
            if (s < ksteps) stage(s, s);
            sm90_cp_commit();
        }
        sm90_wg_fence();

        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + MOE90_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % MOE90_GLU_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<MOE90_GLU_STAGES - 1>();
            __syncthreads();
            const int cb = ks % MOE90_GLU_STAGES;
            moe90_wgmma8_k(accg, As + cb * MOE90_A8BUF, Bg + cb * MOE90_B8BUF, wg);
            moe90_wgmma8_k(accu, As + cb * MOE90_A8BUF, Bu + cb * MOE90_B8BUF, wg);
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
        }

#pragma unroll
        for (int g = 0; g < MOE90_NBLK; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = wg * 64 + sm90_acc_row(wiw, lane, hi);
                const unsigned tok = row_token[rowbase + rr];
                const float as = (tok == PLOW_EXPERT_UNUSED) ? 0.f : ascale[tok];
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int r = sm90_acc_reg(g, hi, lo);
                    const int cc = tn + sm90_acc_col(g, lane, lo);
                    if (cc >= (int)I_moe) continue;
                    if (tok == PLOW_EXPERT_UNUSED) { /* pad row: fu=0 (bf16 parity) */
                        fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(0.f);
                        continue;
                    }
                    const float gv = accg[r] * as * sc[cc];
                    const float uv = accu[r] * as * sc[I_moe + cc];
                    const float a = (act == PLOW_ACT_SILU_) ? act_silu(gv) : act_gelu_tanh(gv);
                    fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(a * uv);
                }
            }
        __syncthreads();
    }
}

static __device__ void d_moe_group_down_gemma_pf_w8a8(
        float* __restrict__ part, const uint8_t* __restrict__ fu8, const float* __restrict__ fscale,
        const unsigned long long* __restrict__ ewt, const unsigned long long* __restrict__ est,
        const int* __restrict__ meta, const unsigned* __restrict__ row_partidx,
        const float* __restrict__ row_gate, unsigned H, unsigned I_moe, unsigned n_exp,
        unsigned slice, unsigned nblk, __nv_bfloat16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)H + MOE90_BN - 1) / MOE90_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = I_moe;
    const int ksteps = ((int)K + MOE90_BK8 - 1) / MOE90_BK8;

    uint8_t* As = (uint8_t*)sm90_align1024(arena);
    uint8_t* Bs = As + MOE90_STAGES * MOE90_A8BUF;
    const int tid = threadIdx.x, wg = tid >> 7, wiw = (tid >> 5) & 3, lane = tid & 31;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * MOE90_BM;
        const int tn = ntile * MOE90_BN;
        const uint8_t* Wd = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 1];
        const float* dsc = (const float*)(size_t)est[(size_t)e * 2 + 1]; /* [H] */

        float acc[64];
#pragma unroll
        for (int i = 0; i < 64; i++) acc[i] = 0.f;

        auto stage = [&](int ks, int buf) {
            moe90_stage_a8(As + buf * MOE90_A8BUF, fu8, tid, rowbase, ks * MOE90_BK8, K);
            moe90_stage_b8(Bs + buf * MOE90_B8BUF, Wd, tid, tn, ks * MOE90_BK8, H, K);
        };
#pragma unroll
        for (int s = 0; s < MOE90_STAGES - 1; s++) {
            if (s < ksteps) stage(s, s);
            sm90_cp_commit();
        }
        sm90_wg_fence();

        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + MOE90_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % MOE90_STAGES);
            sm90_cp_commit();
            sm90_cp_wait<MOE90_STAGES - 1>();
            __syncthreads();
            const int cb = ks % MOE90_STAGES;
            moe90_wgmma8_k(acc, As + cb * MOE90_A8BUF, Bs + cb * MOE90_B8BUF, wg);
            sm90_wg_commit();
            sm90_wg_wait<0>();
            __syncthreads();
        }

#pragma unroll
        for (int g = 0; g < MOE90_NBLK; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++) {
                const int rr = wg * 64 + sm90_acc_row(wiw, lane, hi);
                const unsigned pidx = row_partidx[rowbase + rr];
                if (pidx == PLOW_EXPERT_UNUSED) continue;
                const float rs = row_gate[rowbase + rr] * fscale[rowbase + rr];
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    const int r = sm90_acc_reg(g, hi, lo);
                    const int cc = tn + sm90_acc_col(g, lane, lo);
                    if (cc < (int)H) part[(size_t)pidx * H + cc] = rs * dsc[cc] * acc[r];
                }
            }
        __syncthreads();
    }
}
#endif /* PLOW_NV_W8A8 */

#endif /* PLOW_NV_OP_MOE_SM90_CUH */
