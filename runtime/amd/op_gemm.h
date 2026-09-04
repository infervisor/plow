/* op_gemm.h — bf16 MFMA GEMM family (CDNA4 / gfx950).
 *
 *   C[M,N] = A[M,K] . B[N,K]^T
 *
 * B is [N,K] because that is how a Linear weight is stored: [out_features,
 * in_features]. So A and B are BOTH row-major with K innermost, which makes the
 * two MFMA operand fetches symmetric and 16-byte contiguous. No transpose, ever.
 *
 * Three variants:
 *   d_gemm      — prefill. 128x128 tile, BK=64, double-buffered LDS.
 *   d_gemm_norm — same, with an RMSNorm folded into the A-operand load. The
 *                 normalized activation never touches HBM. Feeds q/k/v (from
 *                 input_layernorm) and gate/up (from pre_feedforward_layernorm),
 *                 i.e. 4 of the 6 projections in every block.
 *   d_gemv      — decode (small M). Pure bandwidth: the weight is streamed
 *                 exactly once and MFMA is not used at all (at M<=16 a 32x32
 *                 matrix core is >50% idle lanes; a coalesced dot product is
 *                 strictly better).
 *
 * See amd_common.h for the MFMA register layout, which was confirmed on hardware.
 */
#ifndef PLOW_OP_GEMM_H
#define PLOW_OP_GEMM_H

#include "amd_common.h"
#include "op_elementwise.h" /* act_gelu_tanh / act_silu: the GLU epilogue */

/* ---------------------------------------------------------------------------
 * MEASURED CEILINGS ON THIS MACHINE (gfx950, 8x MI350X). Do not trust the
 * datasheet numbers; both of these were wrong in earlier revisions of this file.
 *
 *   bf16 MFMA peak, whole GPU, SUSTAINED : 1660 TF/s
 *       A single busy CU hits 9.0 TF/s at the 2.2 GHz boost clock, but 256 busy
 *       CUs cannot hold that clock: dense MFMA pulls the governor down to
 *       ~1.58 GHz (junction is only 57 C, so this is power, not heat).
 *       256 x 4096 FLOP/clk x 1.58 GHz = 1660 TF/s. Extrapolating one CU's
 *       boost-clock number to 256 CUs gives 2308 TF/s and is simply not real.
 *   HBM read, whole GPU                  : 6200 GB/s
 *       Reached at 4 waves/CU IF each lane keeps >=4 loads in flight. Loads in
 *       flight, not wave count, is what saturates HBM: 4 waves x unroll-4 gives
 *       5908 GB/s, 16 waves x unroll-1 gives 6173. Both are the same ceiling.
 *   LDS per workgroup                    : 160 KiB (163840 B), all usable.
 *
 * WAVE COUNT AND THE PING-PONG SCHEDULE
 *
 * 8 waves (512 threads), not 4. A persistent interpreter runs exactly ONE
 * workgroup per CU, so there is no second workgroup to switch to when a global
 * load stalls; the only latency cover available is other waves in this same
 * workgroup. The 8 waves are two groups of 4, and `wcol = wave % 4` puts exactly
 * one wave of each group on each of the 4 SIMDs.
 *
 * The mainloop is four clusters MEM, MFMA, MEM, MFMA separated by s_barrier. A
 * SINGLE extra s_barrier, executed by group 1 only, before the loop, offsets that
 * group by one cluster for the rest of the kernel -- so group 0 is always in a
 * memory phase exactly when group 1 is in an MFMA phase. That one line is the
 * whole ping-pong. s_setprio(1) around each MFMA burst stops the co-resident
 * memory wave from stealing issue slots on the shared SIMD.
 *
 * Do NOT replace this with producer/consumer wave specialisation. AMD partitions
 * registers statically across all waves of a workgroup, so a dedicated producer
 * wave burns its whole register allocation while producing no output.
 *
 * Measured THROUGH THE INTERPRETER (the serving path), mean over the six Gemma-31B
 * projections, after the LDS alignment fix:
 *     8 waves, 2x4 grid, ping-pong, BN=256   573 TF/s   <- current
 *     4 waves, 2x2 grid, no ping-pong, BN=256 533
 *     8 waves, BN=128                        531
 *     4 waves, BN=128                        484
 *
 * So 8 waves is worth about +7% over 4, not the 2.2x an earlier revision of this
 * comment claimed. That claim was measured through a crippled LDS path (the
 * interpreter's arena was 4-byte aligned, so ds_read_b128 silently degraded to pairs
 * of ds_read2_b32) and it did not survive the fix. Keep 8 waves -- it is still the
 * best config and it is free -- but do not spend anything to defend it.
 *
 * TILE
 *
 * 256x256x64, double-buffered = 144 KiB of the 160 KiB LDS. The 2x4 wave grid
 * gives each wave a 128x64 output tile = 8 f32x16 accumulators = 128 AccVGPRs,
 * which is what lets the kernel fit 2 waves/SIMD (a wave may use at most 256
 * registers at that occupancy). A 4x4 per-wave tile is 256 AccVGPRs -- the entire
 * accumulator file -- and cannot run 8 waves at all.
 *
 * BK=64 and not 32: BK is the amount of MFMA work available to cover the
 * prefetch, and at BK=32 the kernel loses ~40%.
 *
 * NON-POWER-OF-2 TILES AND THE POWER-REGIME / K DEPENDENCE (qwen-prefill-gemm sweep).
 * The 128-AGPR accumulator was suspected to bracket every MFMA with v_accvgpr moves and
 * cap the tile at ~34% of peak; the proposed fix was a 192x256 tile whose 96-register
 * accumulator stays in ARCH VGPRs (0 AGPR, measured). The answer is REGIME-DEPENDENT and
 * every measurement below is INTERLEAVED (256 vs 192 back-to-back — see the warm-up note
 * in ubench/gemm_bench_8k.c; a cold-vs-warm clock ramp will otherwise invert the ranking
 * and manufacture a fake winner):
 *
 *   - STANDALONE, idle GPU / abundant power: 256x256 wins (q 50, gate 56 vs 192's 36/40).
 *     With power to spare, the bigger tile's higher arithmetic intensity dominates and the
 *     AGPR moves are free.
 *   - THROUGH THE INTERPRETER (256 CUs of sustained dense MFMA — the power-limited regime
 *     that produces the 1660 TF/s "sustained" ceiling, and the ONLY regime serving runs in):
 *       Qwen3-4B  (K=2560):  192x256 BEATS 256x256 — o +32%, down +21%, gate +11%, q +6%
 *                            (kv N=2048 -8%); the AGPR moves cost power the governor took away.
 *       Gemma-31B (K=5376):  256x256 beats 192x256 (q/o/gate/down -4..-8%). Larger K shifts
 *                            the balance back to the higher-AI tile.
 *
 * So the best tile is K- AND power-dependent; there is no single global winner. 256x256
 * stays the DEFAULT (it does not regress Gemma, and it wins the low-load / single-request
 * latency case for every model). A Qwen prefill object should be built with -DGM_BM=192
 * (+~15% weighted on the compute-heavy projections through the interpreter). The clean fix
 * is per-shape tile selection. Also measured, and both a WASH:
 * compile-time-K specialisation (+0.4%; the mainloop is schedule-bound, not loop-bound) and
 * removing flash from the object (the 128/128 register split is the op union, not flash).
 * A 4x4 per-wave tile (256x256 all-arch) is 256 AccVGPRs and cannot run 8 waves.
 * ------------------------------------------------------------------------- */
/* Overridable at compile time so plowc can bucket the tile per shape without an ISA
 * change, and so a Qwen prefill object can be built with -DGM_BM=192 (see the sweep above).
 * Keep in sync with GFX950_TILES in crates/plowc/src/bin/gemma4.rs. */
/* The DEFAULT rung -- what `d_gemm` (and therefore GemmGlu) instantiates. Its default is
 * arch-conditional for the same reason the ladder below is: 256x256 at BK=64 is a 147,456 B
 * double-buffered stage, 2.25x over CDNA3's 64 KiB, so gfx942 shipped it single-buffered and
 * lost the ping-pong on the op that costs the most (GEMM_GLU is 24% of a Gemma-4 prefill
 * chunk's body). 128x256 at BK=32 fits DOUBLE-buffered in 61,440 B. BN stays 256 because the
 * fused-GLU epilogue's SN==2 assert pins it there at 8 waves.
 *
 * These were previously forced from the outside (`-DGM_BM=... -DGM_BN=...` in
 * scripts/build_gfx942.sh). Defaulting them here means a CDNA3 build that forgets the flags
 * still gets a tile that FITS, rather than one that fails the LDS limit at link time. Still
 * overridable, which is what plowc's per-shape bucketing and every A/B sweep rely on. */
#ifndef GM_BM
#if PLOW_CDNA4
#define GM_BM 256
#else
#define GM_BM 192
#endif
#endif
#ifndef GM_BN
#define GM_BN 256
#endif
/* Overridable like GM_BM/GM_BN, and for the same reason plus one more: on a 64 KiB-LDS part the
 * stage is 4*(BM+BN)*(BK+8) bytes and BK is the only axis that shrinks it WITHOUT touching BN --
 * which matters because the fused-GLU epilogue pins BN through the SN==2 assert below. */
#ifndef GM_BK
#define GM_BK 64
#endif
/* The wave grid follows PLOW_WAVES, and the ping-pong only EXISTS at 8 waves.
 *
 * The schedule needs (wave / 4) to be the group and (wave % 4) to be the SIMD, so
 * that each SIMD hosts one wave of each group and the two can trade places. That
 * requires WN == 4, i.e. 8 waves. At 4 waves there is exactly one wave per SIMD --
 * there is no co-resident partner to hand the SIMD to -- so ping-pong and s_setprio
 * are meaningless and PP compiles out. */
#if PLOW_WAVES == 8
#define GM_WM 2 /* wave grid rows: also the two ping-pong groups */
#define GM_WN 4 /* wave grid cols: one wave per SIMD             */
#define GM_PP 1
#elif PLOW_WAVES == 4
#define GM_WM 2
#define GM_WN 2
#define GM_PP 0
#else
#error "GEMM wave grid is defined for 4 or 8 waves"
#endif
/* Pad the LDS row stride. A fragment read walks 32 consecutive rows at one k, so
 * an unpadded stride of BK*2 bytes lands every lane on the same bank. +8 halves
 * shifts each row by 16 B and breaks the conflict. (An XOR swizzle would save the
 * 16 KiB the padding costs, but we are not short of LDS, so it buys nothing.) */
#define GM_STRIDE(BK) ((BK) + 8)
/* LDS BUFFERS FOR THE A|B STAGE. 2 = the double-buffered default: the next k-tile is committed
 * into the idle buffer while the current one is still being read, so the commit hides entirely
 * behind the MFMA clusters.
 *
 * 1 = single-buffered, and it exists because on a 64 KiB part double-buffering IS the tile
 * ceiling. Measured on gfx942, where the fused-GLU epilogue pins BN to 64*GM_WN:
 *
 *   4 waves,  64x128  ->  55,296 B   fits
 *   4 waves, 128x128  ->  73,736 B   over
 *   8 waves,  64x256  ->  92,168 B   over
 *   8 waves, 192x256  -> 129,032 B   over        <- the tile gfx950 actually runs
 *
 * so no tile bigger than 64x128 exists for CDNA3 while double-buffered, and 64x128 gives each
 * wave SM*SN = 1*2 accumulator tiles against 192x256's 3*2. Halving the stage buys the tile
 * back: 192x256 single-buffered is 64,512 B and fits.
 *
 * What it costs is ONE barrier per k-tile and the LDS write itself, which no longer overlaps the
 * MFMA. What it does NOT cost is the global load: GM_FETCH still issues into registers at the
 * top of the tile and lands behind the whole MFMA run, so only the LDS store is exposed. */
/* DEFAULTED PER ARCH, not passed in by the build. gfx950's 160 KiB holds two buffers at the
 * shipped 256x256/BK=64 tile; CDNA3's 64 KiB does not hold two at ANY tile the ladder uses
 * (192x256/BK=64 double-buffered is 129,024 B), so it is single-stage. This used to be
 * `-DGM_DBUF=1` in scripts/build_gfx942.sh, which meant a CDNA3 build that forgot the flag
 * silently asked for a 129 KiB arena on a 64 KiB part. See the ladder note above GM_SM_BM for
 * the measurement that says buying the second buffer back by halving BK does not pay. */
#ifndef GM_DBUF
#if PLOW_CDNA4
#define GM_DBUF 2
#else
#define GM_DBUF 1
#endif
#endif
/* PER-CLUSTER BARRIERS. The mainloop splits each staged k-tile into NSL MEM/MFMA cluster pairs
 * and puts an s_barrier on BOTH sides of every MFMA burst. Those barriers are the ping-pong's
 * ONLY mechanism: they are what keeps group 0 in a MEM cluster exactly while group 1 is in an
 * MFMA cluster (see PPE below). The ping-pong needs two LDS buffers to trade, so at GM_DBUF=1
 * it does not exist -- and then every one of those 2*NSL barriers per k-tile is doing NOTHING.
 *
 * With one buffer the stage is written once per k-tile and only READ inside the sl loop, so no
 * wave can observe a partial write there. Correctness needs exactly two barriers per k-tile:
 * one after the last read (every wave is done with the buffer) and one after the commit (the
 * new tile is visible). At BK=64/SLICE=16 that is 2 instead of 9.
 *
 * The sched_barrier(0) fences go with them for the same reason: they exist to stop the compiler
 * hoisting the global loads across a cluster boundary and dissolving the ping-pong. With no
 * ping-pong to protect they only forbid the one thing that pays here -- overlapping cluster
 * sl+1's ds_reads with cluster sl's MFMAs, which is the software pipeline the barriers were
 * standing in the way of.
 *
 * Overridable so the two halves can be A/B'd independently; both default to "on iff the
 * ping-pong is on", which leaves gfx950 (GM_DBUF=2) bit-identical. */
#ifndef GM_CLBAR
#define GM_CLBAR (GM_DBUF == 2)
#endif
#if GM_DBUF == 2 && !GM_CLBAR
/* The double-buffered mainloop's only ordering between GM_COMMIT(buf^1) and the next tile's
 * fragment reads IS the cluster barrier — without it the ping-pong is a data race. */
#error "GM_DBUF=2 requires GM_CLBAR=1: no barrier orders the commit against the next tile's reads"
#endif
#ifndef GM_CLFENCE
#define GM_CLFENCE (GM_DBUF == 2)
#endif
/* Local-register prefetch of the NEXT cluster's fragments. Only reachable when the per-cluster
 * barriers are off (see PLR in the body); costs one extra bank of fragment registers. */
#ifndef GM_PLR
#define GM_PLR 1
#endif
/* ...and its fp8 twin, priced separately because an fp8 fragment is twice as wide. See PLR in
 * d_gemm_fp8_t for the numbers that put this at 0. */
#ifndef GM_PLR8
#define GM_PLR8 0
#endif
#ifndef GM_PRIO
#define GM_PRIO 1
#endif
/* Two-deep global prefetch (Tensile's PGR2), in a second bank of staging registers.
 *
 * MEASURED AND REJECTED on gfx942, and the reason is register pressure, not prefetch depth.
 * The single-bank schedule gives every global load ONE k-tile of MFMA to land behind; doubling
 * that to two costs (APT+BPT) more halves of staging registers, and at 8 waves / occupancy 2
 * that is exactly the budget the fragment registers and the accumulator were already using:
 *
 *   tile      VGPR PGR1 -> PGR2      g31_gate_up M=4096, TF/s
 *   256x128    186 -> 238             459 -> 379
 *   128x256    176 -> 228             433 -> 372
 *   128x128    112 -> 156             391 -> 304
 *    64x128     78 -> 102             264 -> 219
 *   320x128    168 -> 256 (19 spill)  455 -> 368
 *
 * Every rung regressed, monotonically with the register growth. The depth is not what the
 * mainloop is short of -- see the GM_HACK_NOFETCH probe, which shows the global path costing
 * ~25% even though the load already has a full k-tile of cover. Kept as a knob, defaulted OFF. */
#ifndef GM_PGR2
#define GM_PGR2 0
#endif
/* DIAGNOSTIC ONLY (never set in a shipping build): stage k-tile 0 on every iteration so every
 * global load hits L1/L2. The result is numerically WRONG; it exists to price the global-load
 * latency the mainloop is failing to cover, against the same schedule. */
#ifndef GM_HACK_NOFETCH
#define GM_HACK_NOFETCH 0
#endif
#define GM_LDS_HALVES_T(BM, BN, BK) (GM_DBUF * ((BM) + (BN)) * GM_STRIDE(BK))
/* The interpreter allocates ONE LDS arena, so it must fit the largest config any
 * packet may select: 256x256x64 => 144 KiB of the 160 KiB available. */
#define GM_LDS_HALVES GM_LDS_HALVES_T(GM_BM, GM_BN, GM_BK)

#define GM_NUM_XCD 8

/* Two-stage workgroup->tile remap for L2 locality.
 *
 * MI350X has 8 XCDs, each with its OWN 4 MB L2, and the hardware assigns
 * workgroup i to XCD (i % 8) round-robin -- so (i % 8) IS the XCD id. Under a
 * linear tile map the 32 workgroups resident on one XCD are strided 8 apart in
 * tile space, so between them they touch nearly every column-block of B and the
 * L2 thrashes. Stage 1 inverts the round-robin so same-XCD workgroups get
 * CONTIGUOUS tile ids. Stage 2 then walks those ids in column-major groups of WGM
 * tile-rows, so a group shares one A row-block and a narrow band of B.
 *
 * MEASURED, at last, on the real model (3326-token prefill; the old "+5..10%" here was never
 * checked against anything):
 *
 *     SWZ=0  linear                    828 ms
 *     SWZ=1  XCD un-round-robin only   801        <- 27 ms, just from stage 1
 *     SWZ=2  + grouped-M, WGM=8        780        <- 21 ms more.  TOTAL 5.8%
 *
 * AND WGM=8 IS ALREADY OPTIMAL — a Hilbert or Morton curve cannot beat it, and the reason is
 * geometric, not empirical. 32 workgroups are co-resident per XCD. Grouped-M(8) arranges them as
 * 8 tile-rows x 4 tile-cols, which touches 8 A-blocks + 4 B-blocks = 12 fetches for 32 tiles —
 * the MINIMUM-PERIMETER rectangle of area 32. The sweep confirms the geometry exactly:
 *
 *     WGM=4   -> 4x8, also 12 fetches   780 ms   (identical, as it must be)
 *     WGM=8   -> 8x4, 12 fetches        780
 *     WGM=16  -> 16x2, 18 fetches       787      (worse, in proportion)
 *
 * Space-filling curves earn their keep when locality is needed at MANY scales at once. Here the
 * concurrency is fixed at 32 per XCD and the optimum is a single min-perimeter rectangle, which
 * grouped-M already emits. There is no headroom for a fancier ordering.
 *
 * NOR IS THERE NUMA TO EXPLOIT: the GPU runs in SPX / NPS1 (verified with rocm-smi and
 * hipGetDeviceProperties: 8 devices x 256 CU, one HBM domain). Memory is interleaved across every
 * stack, so every XCD has uniform HBM access and there is no near/far to schedule for. The ONLY
 * chiplet-private resource is this 4 MB L2 — which is what this remap is for. */
template <int SWZ, int WGM>
__device__ __forceinline__ unsigned gm_remap(unsigned lin, unsigned n_tiles, unsigned tm,
                                             unsigned tn) {
    if (SWZ == 0) return lin;
    /* Stage 1: undo the XCD round-robin. A bijection on [0, limit); identity on the
     * ragged tail so the map stays total. */
    unsigned id = lin;
    const unsigned per = n_tiles / GM_NUM_XCD;
    if (lin < per * GM_NUM_XCD) id = (lin % GM_NUM_XCD) * per + lin / GM_NUM_XCD;
    if (SWZ == 1) return id;
    /* Stage 2: grouped-M. Bijective on each full group of WGM tile-rows; the last
     * group is short when tm % WGM != 0 and is handled by clamping gm. */
    const unsigned in_grp = WGM * tn;
    const unsigned first_m = (id / in_grp) * WGM;
    if (first_m >= tm) return id;
    unsigned gm = tm - first_m;
    if (gm > (unsigned)WGM) gm = WGM;
    const unsigned r = id % in_grp;
    if (r >= gm * tn) return id; /* short-group tail */
    return (first_m + (r % gm)) * tn + (r / gm);
}

/* One shape-specialised GEMM.
 *
 *   BM x BN output tile per workgroup, contracted BK at a time.
 *   4 waves in a 2x2 grid; each wave owns a (BM/2) x (BN/2) quadrant, held as an
 *   (SM x SN) grid of 32x32 f32 accumulators — SM*SN*16 AccVGPRs.
 *
 * NORM folds an RMSNorm into the A-operand load (rms + gamma non-null), so the
 * normalized activation never round-trips through HBM. NOTE: measured, this is a
 * LOSS at large N — the A tile is re-fetched once per N-tile, so per-element work
 * in the A path is multiplied by N/BN. Use it only when N is small.
 */
#ifndef GM_SWZ
#define GM_SWZ 2   /* 0 = linear, 1 = XCD un-round-robin only, 2 = + grouped-M */
#endif
#ifndef GM_WGM
#define GM_WGM 8   /* grouped-M: tile-rows per column-major group */
#endif
template <int BM, int BN, int BK, int WM, int WN, bool NORM, int SWZ = GM_SWZ, int WGM = GM_WGM,
          bool PP = (GM_PP != 0), bool KEXACT = true, bool GLU = false, bool WFP4 = false,
          bool WFP8BLK = false>
__device__ void d_gemm_t(bf16* __restrict__ C, const bf16* __restrict__ A,
                         const bf16* __restrict__ B, const float* __restrict__ rms,
                         const bf16* __restrict__ gamma, unsigned M, unsigned N, unsigned K,
                         unsigned slice, unsigned nblk, bf16* lds,
                         const bf16* __restrict__ B2 = nullptr, unsigned act = 0,
                         const unsigned char* __restrict__ bscale = nullptr,
                         const float* __restrict__ bsblk = nullptr,
                         const unsigned char* __restrict__ bscale2 = nullptr) {
    (void)B2;
    (void)bscale;
    (void)bsblk;
    (void)bscale2;
    /* WFP4 (w4a16 mxfp4 weights): B is a packed-2/byte fp4 tensor (row stride K/2 bytes) and
     * `bscale` its E8M0 scale rows (K/32 bytes/row). The B-fetch below dequants fp4->bf16 with the
     * MX scale folded EXACTLY (fp4_to_bf16v8), then stages bf16 into LDS — so the A-operand, LDS
     * swizzle, bf16 MFMA and epilogue are byte-for-byte the bf16 path (if constexpr discards this
     * branch entirely when WFP4=false). Activations stay bf16 (w4a16). Requires KEXACT and
     * K % 32 == 0; prefill K (a multiple of 128) satisfies both. HARDWARE-VALIDATE the numerics
     * against the golden d_gemm_mxfp4_k. */
    static_assert(!WFP4 || KEXACT, "the w4a16 fp4 B-fetch requires KEXACT (K % BK == 0)");
    static_assert(!WFP4 || !NORM, "fp4 weights + fused RMSNorm-A is not a combination plow emits");
    /* WFP8BLK (w8a16 block-fp8 weights, GLM/DeepSeek `weight_block_size: [128,128]`): B is an
     * e4m3 tensor at 1 B/elt (row stride K bytes) and `bsblk` its `[ceil(N/128)][ceil(K/128)]` f32
     * `weight_scale_inv` grid, row-major with K innermost — the SAME grid `gemv_rows_fp8_blk` (op
     * 44) and `d_moe_group_pf_t<FP8=true>` (ops 85/86) read, indexed `bsblk[(n>>7)*KB + (k>>7)]`
     * with `KB = ceil(K/128)`. One convention, three kernels; a second convention here would be
     * exactly the silent-corruption class this file's headers keep warning about.
     *
     * PROMOTED, NOT FOLDED — copied deliberately from op_moe.h's grouped block-fp8 arm, which is
     * the numeric family this joins. The scale is an ARBITRARY f32, so it cannot ride the cvt's
     * scalef32 operand (that is E8M0/exponent-only and discards the mantissa; amd_common.h records
     * a measured ~22% error on a real GLM scale), and folding it in software before the bf16 store
     * would round an exact fp8 value AFTER scaling and lose the precision fp8 had. So LDS holds the
     * EXACT fp8->bf16 decode (e4m3 has 3 mantissa bits, bf16 has 7 — lossless), the MFMA runs on
     * that, and the f32 accumulator is multiplied by the block scale every 128 K and promoted into
     * a second accumulator. Cost: one extra accumulator set (SM*SN*16 f32) and nothing else.
     *
     * BK=64 divides the 128-element K scale block exactly, so the promotion always lands on a
     * k-tile boundary and never inside an MFMA burst. The N-block index is per-lane CONSTANT: the
     * 32x32 MFMA gives lane l the single output column n = l%32 and n is the weight ROW here, so
     * `(n>>7)` does not vary across the 16 accumulator elements a lane holds — no cross-lane scale
     * shuffle exists in this kernel either. BN % 128 == 0 keeps `n0` 128-aligned, which is what
     * makes the per-(wn,j) 32-column range sit inside ONE N-scale block. */
    static_assert(!WFP8BLK || KEXACT, "the block-fp8 B-fetch requires KEXACT (K % BK == 0)");
    static_assert(!WFP8BLK || !NORM, "block-fp8 weights + fused RMSNorm-A is not emitted");
    static_assert(!WFP8BLK || !GLU, "block-fp8 gate|up is emitted as two GEMMs + Glu, not fused");
    static_assert(!WFP8BLK || !WFP4, "one weight encoding at a time");
    static_assert(!WFP8BLK || BK == 64, "the 128-K scale block must be exactly two BK tiles");
    static_assert(!WFP8BLK || BN % 128 == 0, "n0 must be 128-aligned for a per-lane N-scale block");
    (void)act;
    constexpr int THREADS = WM * WN * PLOW_WAVE;
    /* COMPACT row stride (no +8 pad). global_load_lds writes the 64 lanes CONTIGUOUSLY from a
     * uniform M0, so per-row padding cannot be applied during the DMA. An XOR swizzle on the
     * 16-byte column (GM_XORSWZ) replaces the pad and is conflict-free for the ds_read_b128 fragment
     * groups (verified below). The interpreter arena is still sized by the padded GM_STRIDE, so
     * this only makes each tile SMALLER inside the same arena -- no cross-op behaviour changes. */
    constexpr int STRIDE = BK;
    static_assert(BK % 8 == 0, "the XOR swizzle works at 16-byte (8-half) column granularity");
    constexpr int SM = BM / WM / MFMA_M; /* 32x32 accumulator tiles per wave, m */
    constexpr int SN = BN / WN / MFMA_N; /* ... and n                           */
    /* k consumed per MFMA cluster. SLICE sets the MEM/MFMA cluster count per BK tile
     * (NSL = BK/SLICE) and therefore the s_barrier count: SLICE=16 -> 4 clusters -> 8
     * barriers/tile; SLICE=32 -> 2 clusters -> 4 barriers/tile.
     *
     * The barrier count is pure scheduling overhead and gates MFMA-issue whenever the
     * matrix pipe is under-fed. MEASURED (fp8-gemm profiling, standalone M=4096, gfx950):
     * the 192x256 tile (c5, the Qwen prefill object, -DGM_BM=192) is barrier/LDS-latency
     * bound, NOT memory bound -- MfmaUtil 30%, MemUnitStalled ~0. Halving the barriers
     * with SLICE=32 raises MfmaUtil 30.5 -> 33.2% and TF/s: gate/up +9% (714->778),
     * down +7% (807->861), q +2.5%. It costs nothing on this tile (c5 has register
     * headroom: 197 VGPR / 0 AGPR / occ-2 either way).
     *
     * But SLICE is TILE-dependent and MUST stay off the 256x256 default (c0): c0 already
     * spills (880B->940B) and SLICE=32 pushes it further over -> 533 -> 322 TF/s, a HARD
     * regression. The default stays 16.
     *
     * AND THE STANDALONE WIN DOES NOT SURVIVE THE INTERPRETER. Measured end-to-end (Qwen3-4B
     * live prefill, static sched, argmax device==host, median of 3): SLICE 16 -> 32 gives
     * 4k 182->181 ms, 8k 491->490 ms -- ~0.3-0.5%, NEUTRAL. The +7-9% standalone gain
     * compresses away: GEMM is ~61% of prefill, the gain is only on 2 of 5 projections
     * (gate/up, down), and the 256-CU power/L2 serving regime is not the idle-GPU regime the
     * standalone bench measures. So GM_SLICE=32 is NOT wired into the production Qwen build --
     * it is kept as a knob for the occ-1 deep-pipeline work, where breaking the read->MFMA
     * dependency chain is the actual lever (see the fp8 verdict below). Measure THROUGH THE
     * SERVING PATH, don't reason from standalone. */
#ifndef GM_SLICE
#define GM_SLICE 16
#endif
    constexpr int SLICE = GM_SLICE;
    constexpr int NSL = BK / SLICE;      /* MEM/MFMA cluster pairs per staged tile */
    constexpr int KS = SLICE / MFMA_K;   /* MFMA k-steps per cluster (= 1)         */
    constexpr int APT = BM * BK / THREADS;
    constexpr int BPT = BN * BK / THREADS;
    constexpr int APASS = APT / 8;       /* 16-byte passes                      */
    constexpr int BPASS = BPT / 8;
    constexpr int TILE = (BM + BN) * STRIDE;
    constexpr bool DBUF = (GM_DBUF == 2); /* one stage buffer, or two */
    constexpr bool CLBAR = (GM_CLBAR);    /* per-cluster barriers: the ping-pong's mechanism */
    constexpr bool CLFENCE = (GM_CLFENCE);
    /* PLR (local-register prefetch) rotates the fragment registers so cluster sl+1's ds_reads
     * issue before cluster sl's MFMA burst. It is mutually exclusive with the per-cluster
     * barriers -- a barrier between the read and its consumer is exactly what it exists to
     * remove -- so it defaults OFF wherever the ping-pong is ON, leaving gfx950 untouched. */
    constexpr bool PLR = (GM_PLR) && !CLBAR;
    /* s_setprio(1) around the MFMA burst exists for the ping-pong: it stops the co-resident
     * MEMORY wave stealing issue slots from the matrix pipe. With the ping-pong gone the two
     * waves on a SIMD are not phase-locked, so raising priority for the whole burst can instead
     * starve the partner's ds_read issue. Kept on by default (it is what gfx950 ships) and
     * exposed so the CDNA3 schedule can be A/B'd against it. */
    constexpr bool PRIO = (GM_PRIO) != 0;

    static_assert(APT % 8 == 0 && BPT % 8 == 0, "tile must stage in 16-byte units");
    static_assert(BM % (WM * MFMA_M) == 0 && BN % (WN * MFMA_N) == 0, "tile must fit the wave grid");
    static_assert(NSL >= 2 || !CLBAR, "the ping-pong needs at least 2 MEM/MFMA cluster pairs");
    static_assert(THREADS == PLOW_THREADS, "GEMM wave grid must match the interpreter's");
    /* THE GLU EPILOGUE RIDES THE EXISTING TILE -- no extra registers, no extra LDS, no extra
     * MFMA. The trick is the WAVE->COLUMN map. Normally `j` (the SN axis) selects a column
     * block, so a wave owns BN/WN contiguous columns. Under GLU, `j` selects GATE(0) vs UP(1)
     * instead, and the wave owns MFMA_N columns of BOTH:
     *
     *     normal:  nn = n0 + wn*(BN/WN) + j*MFMA_N + acc_n(lane)
     *     GLU:     nn = n0 + wn*MFMA_N + acc_n(lane),  j = 0 gate / 1 up
     *
     * So acc[i][0][e] and acc[i][1][e] are gate and up OF THE SAME OUTPUT ELEMENT, in the same
     * lane -- the epilogue is a plain act(g)*u with no shuffle and no LDS exchange. The B tile
     * still stages BN rows: Wg[n0, n0+BN/2) in the low half, Wu[n0, n0+BN/2) in the high half.
     * A workgroup therefore emits BN/2 FUSED columns using exactly the MFMA count it used to
     * spend on BN raw ones -- which is the same arithmetic, since every output needs both. */
    static_assert(!GLU || SN == 2, "the GLU epilogue uses the SN axis to select gate vs up");
    constexpr int NB = GLU ? BN / 2 : BN; /* output columns a tile emits */

    /* DIRECT global->LDS staging (global_load_lds_dwordx4, 16 B/lane, no register round-trip) is
     * used for the common fast path. NORM must transform each element in registers, and a ragged
     * K tile (KEXACT==false) would run the fixed 16 B load off the row end -- both keep the
     * register fetch+commit path, which writes the SAME swizzled layout. GLU stages a second
     * weight (Wu) into the B tile's high half, which the single-source DMA below does not express,
     * so it too stays on the fetch+commit path (its layout is still the swizzled one). */
    /* WFP4 dequants fp4->bf16 in registers, so it MUST use the fetch+commit path (the direct
     * global_load_lds DMA streams raw weight bytes to LDS with no register round-trip to convert
     * in). Adding !WFP4 leaves the bf16 predicate — and thus the bf16 codegen — untouched. */
    /* AND IT IS CDNA4-ONLY, WHICH IS NOT A DETAIL -- ON CDNA3 IT IS A PESSIMISATION.
     *
     * `cp_async16` has no CDNA3 hardware behind it: gfx942's global_load_lds tops out at 4 B/lane
     * and a 4-byte issue lays the lanes down at lane*4, not the lane*16 stride every fragment read
     * assumes, so amd_arch.h/amd_common.h lower the 16 B/lane call to a VGPR-STAGED COPY -- a
     * global_load_dwordx4 with the matching ds_write_b128 fused to it. That is the same work
     * GM_FETCH+GM_COMMIT do, with one difference that costs everything: the load and the LDS store
     * are welded together, so the prefetch DISTANCE is zero.
     *
     * At GM_DBUF=1 that lands the whole staging sequence in the k-tile TAIL (there is no idle
     * buffer to stream into early), immediately followed by cp_async_wait() -- i.e. a full HBM
     * round trip exposed on EVERY k-tile, ~84 of them at K=5376, with no MFMA over it. The
     * register path instead issues GM_FETCH(kn) at the TOP of the tile, where it has all NSL
     * clusters of MFMA to land behind, and leaves only the ds_write in the tail.
     *
     * MEASURED (gfx942, GM_DBUF=1, g31_gate_up M=4096 x N=21504 x K=5376, bf16 TF/s, warm
     * median-of-5): 192x256 431 -> 430 is not the story -- the tile ladder as a whole moves
     * 369 -> 449 best-of, and the small tiles move most because their k-tiles are shortest:
     * 64x128 231 -> 268, 128x128 317 -> 378, 128x256 359 -> 451.
     * gfx950 is untouched: PLOW_CDNA4 is 1 there, the builtin is real, and DBUF=2 gives the DMA
     * its idle buffer to stream into a cluster ahead. */
    constexpr bool DIRECT = PLOW_CDNA4 && !NORM && !GLU && KEXACT && !WFP4 && !WFP8BLK;
    /* Two-deep GLOBAL prefetch, the register-bank analogue of Tensile's PGR2. Only reachable on
     * the single-buffered register-staging path -- with two LDS buffers the DMA already has the
     * idle buffer to stream into, and DIRECT has no register bank to double. */
    constexpr bool PGR2 = (GM_PGR2) && !DBUF && !DIRECT;

#define GM_ASM(b) (lds + (b) * TILE)
#define GM_BSM(b) (lds + (b) * TILE + BM * STRIDE)
/* 16-byte-column XOR swizzle. `off` is the half-offset within a row (a multiple of 8); `row` is
 * the LOCAL tile row. The compact stride is BK halves = 32 dwords = exactly 32 banks, so without
 * this every row of a fragment read lands on the same bank; XORing the 16-B column with row&7
 * scatters the 32 consecutive rows of a ds_read_b128 across all 32 banks. Self-inverse. */
#define GM_XORSWZ(row, off) ((off) ^ (((row) & (BK / 8 - 1)) << 3))

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / WN; /* 0/1 : the two ping-pong groups */
    const unsigned wn = wave % WN; /* 0..3: one wave per SIMD        */
    const unsigned frow = mfma_frag_row(lane);

    const unsigned tn = (N + NB - 1) / NB, tm = (M + BM - 1) / BM;
    const unsigned n_tiles = tm * tn;

    /* Walk the LINEAR tile index and remap each one. Remapping the START tile
     * instead (and striding by nblk) is WRONG unless the remap is a permutation of
     * [0, nblk): the grouped map returns absolute tile ids spread far beyond nblk,
     * so some tiles would be computed twice and others never -- less work, faster,
     * and silently incorrect. Only shapes with n_tiles <= nblk would look right. */
    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned tile = gm_remap<SWZ, WGM>(lin, n_tiles, tm, tn);
        const unsigned m0 = (tile / tn) * BM, n0 = (tile % tn) * NB;

        f32x16 acc[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) acc[i][j] = (f32x16)(0.0f);

        /* The block-fp8 PROMOTION accumulator, and this lane's N-scale-block row — constant for
         * the whole tile (see the WFP8BLK note at the top of this function). Sized 1x1 and never
         * touched off the block-fp8 arm, so the bf16 and fp4 register allocations are unmoved. */
        constexpr int FSM = WFP8BLK ? SM : 1, FSN = WFP8BLK ? SN : 1;
        f32x16 accf[FSM][FSN];
        unsigned nsblk[FSN];
        const unsigned KB = (K + 127u) >> 7; /* 128-K scale blocks per output channel */
        if constexpr (WFP8BLK) {
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) accf[i][j] = (f32x16)(0.0f);
#pragma unroll
            for (int j = 0; j < SN; j++) nsblk[j] = (n0 + wn * (BN / WN) + j * MFMA_N) >> 7;
        }
        (void)KB;

        /* Staging registers. Indices MUST be compile-time constants: a runtime
         * index forces these arrays out of registers and into scratch (HBM).
         *
         * `ra2`/`rb2` are the SECOND global-prefetch bank (PGR2, below). Sized 1 element when
         * PGR2 is off so the single-bank register allocation -- and gfx950's codegen -- are
         * bit-identical to what they were. */
        __align__(16) bf16 ra[APT], rb[BPT];
        __align__(16) bf16 ra2[PGR2 ? APT : 1], rb2[PGR2 ? BPT : 1];
        (void)ra2;
        (void)rb2;

/* KEXACT (K % BK == 0) is the common case and plowc knows K per bucket, so the
 * ragged-K path below compiles away entirely for every Gemma shape. It must
 * exist though: the load is 8 halves wide, so a chunk starting at kk < K can
 * still run off the end of the row and pull in the NEXT row's values, which the
 * MFMA then silently accumulates. Checking only `kk < K` is NOT enough. */
#define GM_FETCH_TO(ra, rb, k0)                                                              \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned r = m0 + e / BK;                                                       \
        const unsigned kk = (k0) + (e % BK);                                                  \
        if constexpr (NORM) {                                                                 \
            const float sc = (r < M) ? rms[r] : 0.0f;                                         \
            __align__(16) bf16 tv[8], tg[8];                                                  \
            if (r < M) {                                                                      \
                *(bf16v8*)tv = ld_glob8(as_glob(A) + (size_t)r * K + kk);                     \
                *(bf16v8*)tg = ld_glob8(as_glob(gamma) + kk);                                 \
                _Pragma("unroll") for (int j = 0; j < 8; j++)                                 \
                    ra[it * 8 + j] = f2bf(bf2f(tv[j]) * sc * bf2f(tg[j]));                    \
            } else {                                                                          \
                _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;             \
            }                                                                                 \
        } else if constexpr (KEXACT) {                                                        \
            if (r < M) *(bf16v8*)&ra[it * 8] = ld_glob8(as_glob(A) + (size_t)r * K + kk);     \
            else _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;            \
        } else {                                                                              \
            if (r < M && kk + 8u <= K)                                                        \
                *(bf16v8*)&ra[it * 8] = ld_glob8(as_glob(A) + (size_t)r * K + kk);            \
            else                                                                              \
                _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] =                \
                    (r < M && kk + j < K) ? A[(size_t)r * K + kk + j] : (bf16)0;              \
        }                                                                                     \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned br = e / BK;                                                          \
        /* Under GLU the B tile's low half is W_gate and its high half is W_up, both at the    \
         * SAME output rows [n0, n0+BN/2). Both are addrspace(1), so selecting between them    \
         * keeps the pointer global -- it does NOT become a generic pointer the way an         \
         * LDS/global ternary would (see XLDS at the gemv). */                                 \
        const bf16* bsrc = B;                                                                  \
        /* ...and, under WFP4, the E8M0 scale rows that belong to it. `bscale` is bound to the   \
         * SAME matrix as `bsrc`, so the gate/up select has to move both: a fused fp4 gate|up     \
         * that switched the weight and kept the gate's scale row would read Wu's nibbles with     \
         * Wg's exponents -- finite, fluent, and wrong by a per-block power of two. */             \
        const unsigned char* ssrc = bscale;                                                     \
        unsigned r = n0 + br;                                                                  \
        if constexpr (GLU) {                                                                   \
            const bool up = (br >= BN / 2);                                                    \
            bsrc = up ? B2 : B;                                                                \
            ssrc = up ? bscale2 : bscale;                                                       \
            r = n0 + (up ? br - BN / 2 : br);                                                  \
        }                                                                                      \
        (void)ssrc;                                                                            \
        const unsigned kk = (k0) + (e % BK);                                                 \
        if constexpr (WFP4) {                                                                 \
            /* dequant-on-load: 8 fp4 (one u32) + one E8M0 scale -> 8 bf16. An 8-element load     \
             * never crosses a 32-elem MX block (32 % 8 == 0), so one scale byte covers it. */    \
            if (r < N) {                                                                      \
                const unsigned* wp = reinterpret_cast<const unsigned*>(as_glob(bsrc));         \
                const unsigned w32 = wp[((size_t)r * K + kk) >> 3];                            \
                const unsigned char sc = as_glob(ssrc)[(size_t)r * (K >> 5) + (kk >> 5)];      \
                *(bf16v8*)&rb[it * 8] = fp4_to_bf16v8(w32, e8m0_to_f32(sc));                   \
            } else _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;          \
        } else if constexpr (WFP8BLK) {                                                       \
            /* EXACT fp8 -> bf16, NO scale here. `bsrc` is the e4m3 weight (passed through the    \
             * B slot cast to bf16*, exactly as WFP4 passes its packed fp4); the block scale is    \
             * applied to the f32 accumulator at the 128-K boundary below. */                     \
            if (r < N)                                                                        \
                *(bf16v8*)&rb[it * 8] = fp8v8_to_bf16v8(ld_glob_fp8v8(                        \
                    reinterpret_cast<const unsigned char*>(bsrc) + (size_t)r * K + kk));       \
            else _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;            \
        } else if constexpr (KEXACT) {                                                        \
            if (r < N) *(bf16v8*)&rb[it * 8] = ld_glob8(as_glob(bsrc) + (size_t)r * K + kk);  \
            else _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;            \
        } else {                                                                              \
            if (r < N && kk + 8u <= K)                                                        \
                *(bf16v8*)&rb[it * 8] = ld_glob8(as_glob(bsrc) + (size_t)r * K + kk);         \
            else                                                                              \
                _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] =                \
                    (r < N && kk + j < K) ? bsrc[(size_t)r * K + kk + j] : (bf16)0;           \
        }                                                                                     \
    }

#define GM_COMMIT_FROM(ra, rb, buf)                                                          \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&GM_ASM(buf)[(e / BK) * STRIDE + GM_XORSWZ(e / BK, e % BK)],            \
                         &ra[it * 8], 16);                                                    \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        __builtin_memcpy(&GM_BSM(buf)[(e / BK) * STRIDE + GM_XORSWZ(e / BK, e % BK)],            \
                         &rb[it * 8], 16);                                                    \
    }

#define GM_FETCH(k0) GM_FETCH_TO(ra, rb, k0)
#define GM_COMMIT(buf) GM_COMMIT_FROM(ra, rb, buf)

/* DIRECT global->LDS staging: global_load_lds_dwordx4 (16 B/lane) streams HBM straight into LDS
 * with NO register round-trip -- no ra/rb, freeing those VGPRs. The 64 lanes of a wave land
 * CONTIGUOUSLY from a uniform M0 (lane l -> M0 + l*16 B), so the swizzle cannot be applied to the
 * LDS destination; instead it is applied to the per-lane GLOBAL SOURCE column, which produces the
 * IDENTICAL swizzled layout GM_COMMIT writes (physical col p holds logical col p^(row&7)). Rows
 * past M / cols past N are clamped to the last valid line -- the DMA must issue on every lane to
 * keep the destination contiguous, and those outputs are discarded by the mm<M / nn<N store
 * guards, so a clamp (not a mask) is what is needed and it cannot fault. KEXACT only. */
#define GM_DMA(buf, k0)                                                                      \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned Rloc = e / BK;                                                         \
        const unsigned r = m0 + Rloc;                                                         \
        const unsigned rc = (r < M) ? r : (M - 1);                                            \
        cp_async16(as_glob(A) + (size_t)rc * K + (k0) + GM_XORSWZ(Rloc, e % BK),                 \
                   &GM_ASM(buf)[(threadIdx.x & ~63u) * 8 + it * (THREADS * 8)]);              \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned Rloc = e / BK;                                                         \
        const unsigned r = n0 + Rloc;                                                         \
        const unsigned rc = (r < N) ? r : (N - 1);                                            \
        cp_async16(as_glob(B) + (size_t)rc * K + (k0) + GM_XORSWZ(Rloc, e % BK),                 \
                   &GM_BSM(buf)[(threadIdx.x & ~63u) * 8 + it * (THREADS * 8)]);              \
    }

/* One cluster's worth of A/B fragments, LDS -> registers (ds_read_b128), into a NAMED pair of
 * register banks. The banks are declared by the caller so the PLR rotation below can hold two
 * clusters at once; GM_READ_FRAGS keeps the old declare-and-fill spelling for the CLBAR path. */
#define GM_READ_INTO(af, bfr, buf, sl)                                                       \
    _Pragma("unroll") for (int i = 0; i < SM; i++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned arow = wm * (BM / WM) + i * MFMA_M + frow;                         \
            __builtin_memcpy(                                                                  \
                &af[i][q],                                                                     \
                &GM_ASM(buf)[arow * STRIDE +                                                   \
                             GM_XORSWZ(arow, mfma_frag_k(lane, (sl) * SLICE + q * MFMA_K))], 16); \
        }                                                                                     \
    _Pragma("unroll") for (int j = 0; j < SN; j++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned brow =                                                              \
                (GLU ? j * (BN / 2) + wn * MFMA_N : wn * (BN / WN) + j * MFMA_N) + frow;       \
            __builtin_memcpy(                                                                  \
                &bfr[j][q],                                                                    \
                &GM_BSM(buf)[brow * STRIDE +                                                   \
                             GM_XORSWZ(brow, mfma_frag_k(lane, (sl) * SLICE + q * MFMA_K))], 16); \
        }

#define GM_READ_FRAGS(buf, sl)                                                               \
    bf16x8 af[SM][KS], bfr[SN][KS];                                                           \
    _Pragma("unroll") for (int i = 0; i < SM; i++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned arow = wm * (BM / WM) + i * MFMA_M + frow;                         \
            __builtin_memcpy(                                                                  \
                &af[i][q],                                                                     \
                &GM_ASM(buf)[arow * STRIDE +                                                   \
                             GM_XORSWZ(arow, mfma_frag_k(lane, (sl) * SLICE + q * MFMA_K))], 16); \
        }                                                                                     \
    _Pragma("unroll") for (int j = 0; j < SN; j++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned brow =                                                              \
                (GLU ? j * (BN / 2) + wn * MFMA_N : wn * (BN / WN) + j * MFMA_N) + frow;       \
            __builtin_memcpy(                                                                  \
                &bfr[j][q],                                                                    \
                &GM_BSM(buf)[brow * STRIDE +                                                   \
                             GM_XORSWZ(brow, mfma_frag_k(lane, (sl) * SLICE + q * MFMA_K))], 16); \
        }

/* s_setprio(1) for the burst: the memory wave sharing this SIMD must not steal
 * issue slots from the matrix pipe. */
#define GM_MFMA_FROM(af, bfr)                                                                \
    if constexpr (PRIO) __builtin_amdgcn_s_setprio(1);                                        \
    _Pragma("unroll") for (int q = 0; q < KS; q++)                                            \
        _Pragma("unroll") for (int i = 0; i < SM; i++)                                        \
            _Pragma("unroll") for (int j = 0; j < SN; j++) acc[i][j] =                        \
                plow_mfma_bf16_32x32(af[i][q], bfr[j][q], acc[i][j]);                         \
    if constexpr (PRIO) __builtin_amdgcn_s_setprio(0);
#define GM_MFMA_BURST() GM_MFMA_FROM(af, bfr)

/* sched_barrier(0) lets NOTHING cross. Without these the compiler hoists the
 * global loads across the cluster boundaries and the ping-pong evaporates.
 * With no ping-pong (GM_DBUF=1) that hoisting is the WHOLE POINT -- see GM_CLFENCE. */
#define GM_FENCE()                                                                           \
    do {                                                                                      \
        if constexpr (CLFENCE) __builtin_amdgcn_sched_barrier(0);                             \
    } while (0)
#define GM_CLUSTER_BARRIER()                                                                 \
    do {                                                                                      \
        if constexpr (CLBAR) __builtin_amdgcn_s_barrier();                                    \
    } while (0)

        __syncthreads(); /* previous tile's readers must be done with the LDS */
        if constexpr (DIRECT) {
            /* Stream tile 0 straight into LDS. cp_async_wait drains vmcnt BEFORE the barrier so
             * the DMA's LDS writes are published to the other waves. */
            GM_DMA(0, 0);
            cp_async_wait();
        } else {
            GM_FETCH(0);
            GM_COMMIT(0);
        }
        __syncthreads();

        const unsigned NT = (K + BK - 1) / BK;
        unsigned buf = 0;
        /* PGR2 PRIMES BANK 0 WITH K-TILE 1 HERE, one k-tile ahead of the loop.
         * From then on iteration kt fetches tile kt+2 into the bank it is NOT about to commit,
         * so every global load has a FULL k-tile plus the current one to land behind instead of
         * just the current one. That distance is what the single-bank schedule cannot buy: its
         * fetch is issued at cluster 0 of tile kt and waited on at the END of tile kt. */
        if constexpr (PGR2) {
            if (BK < K) GM_FETCH_TO(ra, rb, (unsigned)BK)
        }

        /* THE PING-PONG PRIMITIVE. Group 1 eats one extra barrier here, which
         * offsets it by exactly one cluster for the rest of the kernel: from now on
         * group 0 is in a MEM cluster exactly when group 1 is in an MFMA cluster. */
        constexpr bool PPE = PP && DBUF; /* the ping-pong trades BUFFERS; one buffer, no trade */
        if (PPE && wm == 1) __builtin_amdgcn_s_barrier();

#pragma unroll 1
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1) * BK;
            const unsigned kn2 = (kt + 2) * BK;
            (void)kn2;

/* The global traffic a cluster owes, identical under both schedules. */
#define GM_CLUSTER_GLOBAL(sl)                                                                \
    if constexpr (DIRECT) {                                                                   \
        /* Issue the next tile's global->LDS DMA at the START of the MEM cluster so it lands   \
         * behind NSL-1 clusters of MFMA; drain it at the LAST cluster, before the barrier, so \
         * buf^1 is fully written when the next kt reads it. Single-buffered there is no idle  \
         * buffer to land in, so it moves below. */                                            \
        if (DBUF && (sl) == 0 && kn < K) { GM_DMA(buf ^ 1, kn); }                              \
        if (DBUF && (sl) == NSL - 1 && kn < K) { cp_async_wait(); }                            \
    } else {                                                                                  \
        /* The global->REGISTER fetch stays at the top of the tile under both layouts: it      \
         * touches no LDS, so it lands behind the whole MFMA run either way. Only the LDS       \
         * commit has to wait for the reads when there is one buffer. */                        \
        if constexpr (!PGR2) {                                                                 \
            if ((sl) == 0 && kn < K) { GM_FETCH(GM_HACK_NOFETCH ? 0u : kn); }                  \
            if (DBUF && (sl) == NSL - 1 && kn < K) { GM_COMMIT(buf ^ 1); }                      \
        } else if ((sl) == 0 && kn2 < K) {                                                     \
            /* bank kt&1 holds tile kt+1 and is committed at the tail; fetch into the other. */ \
            if (kt & 1u) { GM_FETCH_TO(ra, rb, kn2) } else { GM_FETCH_TO(ra2, rb2, kn2) }       \
        }                                                                                      \
    }

            if constexpr (!PLR) {
#pragma unroll
                for (int sl = 0; sl < NSL; sl++) {
                    /* ---- MEM cluster: LDS -> fragments, plus this tile's global traffic. */
                    GM_FENCE();
                    GM_READ_FRAGS(buf, sl)
                    GM_CLUSTER_GLOBAL(sl)
                    GM_FENCE();
                    GM_CLUSTER_BARRIER();

                    /* ---- MFMA cluster */
                    GM_MFMA_BURST()
                    GM_FENCE();
                    GM_CLUSTER_BARRIER();
                }
            } else {
                /* PLR: the ds_read for cluster sl+1 is ISSUED BEFORE cluster sl's MFMA burst, so
                 * its ~120-cycle LDS latency and its 8-cycle-per-read issue occupancy sit under
                 * SM*SN*KS matrix instructions instead of in front of them. Straight out of the
                 * disassembly of the non-PLR loop, which reads:
                 *
                 *     ds_read_b128 x10 ; s_waitcnt lgkmcnt(0) ; v_mfma x24 ; ds_read_b128 x10 ;
                 *     s_waitcnt lgkmcnt(0) ; v_mfma x24
                 *
                 * -- every read burst fully exposed. The rotation costs one extra bank of fragment
                 * registers ((SM+SN)*KS*16 B/lane) and nothing else; the compiler sinks the
                 * waitcnt to the first consumer on its own once the producer is a cluster early. */
                bf16x8 afp[2][SM][KS], bfp[2][SN][KS];
                GM_READ_INTO(afp[0], bfp[0], buf, 0)
#pragma unroll
                for (int sl = 0; sl < NSL; sl++) {
                    if (sl + 1 < NSL) { GM_READ_INTO(afp[(sl + 1) & 1], bfp[(sl + 1) & 1], buf, sl + 1) }
                    GM_CLUSTER_GLOBAL(sl)
                    GM_MFMA_FROM(afp[sl & 1], bfp[sl & 1])
                }
            }
            /* BLOCK-FP8 PROMOTION. A 128-element K scale block is exactly two BK=64 tiles, so the
             * boundary always falls here — between k-tiles, outside every MFMA burst and outside
             * the ping-pong's barrier pairs, so it perturbs neither. */
            if constexpr (WFP8BLK) {
                if ((kt & 1u) == 1u || kt == NT - 1u) {
                    unsigned kb = kt >> 1;
                    if (kb >= KB) kb = KB - 1;
#pragma unroll
                    for (int i = 0; i < SM; i++)
#pragma unroll
                        for (int j = 0; j < SN; j++) {
                            const float bs = as_glob(bsblk)[(size_t)nsblk[j] * KB + kb];
                            accf[i][j] += acc[i][j] * bs;
                            acc[i][j] = (f32x16)(0.0f);
                        }
                }
            }
            if constexpr (!DBUF) {
                /* Every wave has now read the whole tile, so the single buffer is free. This is
                 * the barrier double-buffering does not pay, and the LDS store it does not
                 * expose -- the price of the bigger tile.
                 *
                 * When the per-cluster barriers are OFF this is the only place the "everyone has
                 * finished reading" edge exists, so the barrier below the reads is REQUIRED, not
                 * an optimisation: without it a fast wave overwrites the stage while a slow one
                 * is still issuing ds_read for the current k-tile. */
                /* __syncthreads(), not a bare s_barrier: with the per-cluster fences gone these
                 * two are the ONLY things ordering the ds_reads above against the ds_writes
                 * below, so they have to carry the workgroup memory fence as well as the
                 * hardware barrier. (Under CLBAR the cluster barriers already separated them and
                 * the sched_barrier(0) fences pinned the schedule; that path is unchanged.) */
                if (kn < K) {
                    if constexpr (!CLBAR) __syncthreads();
                    if constexpr (DIRECT) {
                        GM_DMA(0, kn);
                        cp_async_wait();
                    } else if constexpr (PGR2) {
                        if (kt & 1u) { GM_COMMIT_FROM(ra2, rb2, 0) } else { GM_COMMIT_FROM(ra, rb, 0) }
                    } else {
                        GM_COMMIT(0);
                    }
                    if constexpr (CLBAR) {
                        GM_FENCE();
                        __builtin_amdgcn_s_barrier();
                    } else {
                        __syncthreads();
                    }
                }
            }
            if constexpr (DBUF) buf ^= 1;
        }
        /* Hand the promoted result back to the shared epilogue below. */
        if constexpr (WFP8BLK) {
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) acc[i][j] = accf[i][j];
        }
        /* Rebalance: group 0 must execute as many barriers as group 1 did. */
        if (PPE && wm == 0) __builtin_amdgcn_s_barrier();
        __syncthreads();

        /* The store is already COALESCED -- lanes 0-31 hold consecutive columns, so each of
         * these is a clean 64-byte transaction. It was just on the GENERIC path: `C` is a
         * plain bf16* out of the tensor table, so every one of them compiled to
         * flat_store_short (188 of them in the prefill kernel), carrying a full 64-bit address
         * per lane and tracking on lgkmcnt as well as vmcnt. as_glob is free. */
        auto* const Cg = as_glob(C);
        if constexpr (GLU) {
            /* THE FUSED EPILOGUE. acc[i][0] is gate and acc[i][1] is up FOR THE SAME OUTPUT
             * ELEMENT, in the same lane -- that is what the wave->column remap above bought.
             * No shuffle, no LDS exchange, and `fu` is the only thing that reaches HBM: the
             * gate and up tiles never leave the register file. */
            const unsigned nn = n0 + wn * MFMA_N + mfma_acc_n(lane);
            if (nn < N) {
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int e = 0; e < 16; e++) {
                        const unsigned mm = m0 + wm * (BM / WM) + i * MFMA_M + mfma_acc_m(lane, e);
                        if (mm >= M) continue;
                        const float g = acc[i][0][e];
                        const float u = acc[i][1][e];
                        const float sg = act_gate_only(g, act);
                        st_act1(&Cg[(size_t)mm * N + nn], f2bf(sg * u));
                    }
            }
        } else {
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) {
                const unsigned nn = n0 + wn * (BN / WN) + j * MFMA_N + mfma_acc_n(lane);
                if (nn >= N) continue;
#pragma unroll
                for (int e = 0; e < 16; e++) {
                    const unsigned mm = m0 + wm * (BM / WM) + i * MFMA_M + mfma_acc_m(lane, e);
                    if (mm < M) st_act1(&Cg[(size_t)mm * N + nn], f2bf(acc[i][j][e]));
                }
            }
        }
#undef GM_FETCH
#undef GM_COMMIT
#undef GM_DMA
#undef GM_READ_FRAGS
#undef GM_MFMA_BURST
#undef GM_FENCE
#undef GM_ASM
#undef GM_BSM
#undef GM_XORSWZ
    }
}

/* Tile configs, selectable per shape bucket by the compiler. All use the 2x4 wave
 * grid (8 waves), so BM % 64 == 0 and BN % 128 == 0.
 *
 * Measured on the real Gemma-4-31B projections, whole GPU, against the 1660 TF/s
 * sustained bf16 MFMA ceiling (the ONLY number worth comparing to; see above):
 *
 *   shape                                   old (4w)   new (8w ping-pong)
 *   q_proj  sliding 4096 x  8192 x  5376      502          887
 *   kv_proj sliding 4096 x  4096 x  5376      487          993
 *   q_proj  global  4096 x 16384 x  5376      453          963
 *   o_proj  sliding 4096 x  5376 x  8192      386          777
 *   gate/up_proj    4096 x 21504 x  5376      405          910
 *   down_proj       4096 x  5376 x 21504      331          692
 *
 * The tile must DIVIDE the shape: a ragged edge on every tile-column wastes MFMA
 * lanes and quantizes badly against 256 CUs. 192x192 was the best tile on a
 * single CU (17.1% vs 10.4% of CU peak) and is still slower end-to-end on M=4096,
 * because 4096 % 192 != 0. Divisibility beats per-CU efficiency.
 *
 * The tile is a COMPILE-TIME parameter and has to be: a runtime switch over the
 * variants makes the register allocator budget for the worst arm, and then EVERY
 * arm spills. */
enum {
    PLOW_GM_T256 = 0,   /* 256x256x64 — 144 KiB LDS — default, best on every Gemma shape */
    PLOW_GM_T128 = 1,   /* 128x256x64 — 108 KiB — narrow M                                */
    PLOW_GM_T128SQ = 2, /* 128x128x64 —  72 KiB — smallest                                */
};

/* The tile is baked in. There is deliberately NO runtime switch over the configs:
 * register allocation is per-kernel, so a switch makes the allocator budget for the
 * worst arm and then EVERY arm spills. Measured through the interpreter, a
 * three-arm d_gemm_cfg switch cost 285 TF/s against 770 for the same tile compiled
 * on its own -- the arms it never takes still poison the one it does. The variants
 * live in test_kernels.hip, which benchmarks them so plowc can pick GM_BN per
 * bucket. */
__device__ void d_gemm(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N, unsigned K,
                       unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K, slice,
                                                       nblk, lds);
}

/* MXFP4 (w4a16) PREFILL GEMM. A is bf16 activations, W is packed-2/byte fp4 weights (row stride
 * K/2 bytes) with E8M0 scale rows `wscale` (K/32 bytes/row). Reuses the entire bf16 d_gemm_t —
 * only the B-fetch dequants fp4->bf16 with the MX scale folded exactly (WFP4=true), so the LDS
 * staging, wide-K bf16 MFMA and epilogue are identical to the bf16 prefill. This buys the fp4
 * WEIGHT-BANDWIDTH win at M>1 without an activation-quant op; the native cbsz/blgp=4 f8f6f4 2x-
 * compute path is a separate future kernel (op_gemm.h fp8 verdict: that path measured only PARITY
 * anyway until the occ-1 deep-pipeline rewrite, so w4a16 is the correct first prefill GEMM). */
/* ONE TILE PER SHAPE, not one tile for every shape. `d_gemm_mxfp4` used to hard-code
 * GM_BM/GM_BN (256x256) with no selection at all, which is why Kimi's mxfp4 kv_a_proj
 * (M=128, N=576) ran at ~0.4% of peak: at that shape 256x256 is THREE tiles on 256 CUs.
 * The bf16 family has had `pick_tile` since the small/med rungs landed; this is the same
 * family with a different B-fetch, so it gets the same rungs and the same selector.
 * `PLOW_GM_MXFP4_TILE(NAME, BM, BN)` keeps the five bodies textually identical -- the
 * previous divergence was a copy of the default-tile call that nobody updated. */
/* BK IS A PARAMETER, not the global `GM_BK`, and that is a fix rather than a generalisation.
 * The comment above promises these twins are the same tiles as their bf16 rungs; with `GM_BK`
 * hardcoded they were the same BM/BN against the DEFAULT rung's BK. On gfx950 that is invisible
 * because every rung is BK=64 == GM_BK. On CDNA3, where the rungs carry different BKs to fit a
 * DOUBLE-BUFFERED stage in 64 KiB, it breaks the build outright:
 *   d_gemm_t<192, 256, 32, ...>  static_assert(APT % 8 == 0) -- 192*32/512 = 12.
 * Note these are instantiated UNCONDITIONALLY, not behind PLOW_MXFP4, so an arch that never
 * serves mxfp4 still has to compile them. */
#define PLOW_GM_MXFP4_TILE(NAME, BM, BN, BK)                                                   \
    __device__ void NAME(bf16* C, const bf16* A, const unsigned char* W,                       \
                         const unsigned char* wscale, unsigned M, unsigned N, unsigned K,      \
                         unsigned slice, unsigned nblk, bf16* lds) {                           \
        d_gemm_t<BM, BN, BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true, false,   \
                 true>(C, A, (const bf16*)W, nullptr, nullptr, M, N, K, slice, nblk, lds,       \
                       nullptr, 0, wscale);                                                    \
    }
PLOW_GM_MXFP4_TILE(d_gemm_mxfp4, GM_BM, GM_BN, GM_BK)

/* Small tile, for shapes that cannot fill the machine at 256x256.
 *
 * BM=64 (not 256) so a prefill of 128 tokens is 2 row-tiles instead of 1 half-empty one,
 * and BN=128 so N/BN doubles. Per-wave accumulator is 1 f32x16 = 16 AccVGPRs, which is why
 * a second instantiation of d_gemm_t is affordable at all. It has WORSE arithmetic
 * intensity than the 256x256 tile (A is re-read N/128 times instead of N/256) -- it is a
 * trade of HBM traffic for CU occupancy, and it only wins when the big tile leaves most of
 * the GPU idle. plowc decides. */
/* MEASURED on the real Gemma-31B projections, whole GPU, best-of sweep:
 *
 *   shape                        256x256   128x128   64x128
 *   T=128  q_proj  128x8192x5376    48.3     131.0    150.7
 *   T=128  o_proj  128x5376x8192    35.5      93.9    110.9
 *   T=128  gate/up 128x21504x5376  118.9     326.7    221.1
 *   T=128  down    128x5376x21504   38.1     103.3    122.8
 *   T=512  q_proj  512x8192x5376   186.6     483.7    333.6
 *   T=512  o_proj  512x5376x8192   138.9     362.2    243.3
 *   T=4096 (any)                   ~700-930   ~400     ~250
 *
 * So the big tile is 3x too SLOW for a real prefill and 2-3x too FAST to give up at long
 * context. There is no single right tile -- which is the whole reason the tile is a
 * compile-time bucket and plowc picks it per shape. */
/* THE RUNG LADDER IS SHARED BY BOTH ARCHS, and CDNA3 runs it SINGLE-buffered (GM_DBUF=1,
 * defaulted per-arch above). At DBUF=1 every rung below fits 64 KiB unchanged -- 27,648 /
 * 36,864 / 55,296 / 64,512 B -- so there is nothing to re-cut.
 *
 * RESTORING THE PING-PONG ON CDNA3 WAS BUILT AND MEASURED, AND IT LOSES. The idea: BK is the
 * only axis that shrinks the stage without touching BN (BN is pinned by the fused-GLU SN==2
 * assert), so halve it to 32 and the second buffer fits -- 128x256 double-buffered is 61,440 B.
 * That requires re-cutting the ladder, because at BK=32 the staging assert (APT/BPT % 8 == 0)
 * demands BM % 128 == 0 and BN % 128 == 0, which 192 and 64 fail. Built it: MD 128x128 BK=32,
 * WD 128x256 BK=32, C5 256x128 BK=32, default 128x256 BK=32, GM_DBUF=2. It compiles, it fits
 * (61,448 B, 256 VGPR, spill 2) and it PASSES the golden GEMM oracle. Measured on the real
 * Gemma-4-12B asset, 4096-token prefill, gfx942:
 *
 *     192x256  BK=64  DBUF=1   575.2 ms   <- shipped
 *     128x256  BK=64  DBUF=1   585.9 ms   +1.9%
 *     128x256  BK=32  DBUF=2   642.6 ms   +11.7%
 *
 * The ping-pong does not pay for the BK it costs. Halving BK doubles the k-tile count and its
 * barriers, and drops NSL from 4 to 2 -- the ping-pong's minimum -- so there are half as many
 * cluster pairs to trade while the register-staged global prefetch (GM_FETCH at the top of the
 * tile) loses half the MFMA it had to land behind. On a 64 KiB part the long single-buffered
 * k-tile is worth more than the overlap. Do not re-try this without a way to keep BK=64. */
#define GM_SM_BM 64
#define GM_SM_BN 128
#define GM_SM_BK 64
#define GM_MD_BM 128
#define GM_MD_BN 128
#define GM_MD_BK 64
/* THE TWO MISSING RUNGS (tile-inventory campaign, measured 2026-07-27, runtime/ubench/
 * gemm_tile_sweep.c, whole GPU, 20 reps after a 50-launch clock warm-up, leased card).
 *
 * The three shipped tiles (256x256, 128x128, 64x128) leave a 1.3-1.8x hole for every
 * M >= 1024 shape, because between "one 256-wide tile that fills 50-65% of the CUs" and
 * "128x128, which fills them at a quarter of the arithmetic intensity" there was NOTHING.
 * BN=256 with BM cut is the rung that fills the machine WITHOUT giving up the A-reuse:
 *
 *   shape                       M     256x256  128x128  64x128 | 128x256  192x256
 *   g31b q_proj  Nx8192xK5376  1024     523      684     435   |   926      839
 *   g31b q_proj                2048     929      737     449   |  1064      895
 *   g31b q_proj                8192    1115      636     458   |   898     1236
 *   g31b gate/up N21504 K5376  2048     974      570     332   |   746     1033
 *   g31b down    N5376 K21504  2048     794      542     300   |   611     1033
 *   qwen o_proj  N2560 K4096   4096     588      598     426   |   673      941
 *   qwen gate/up N9728 K2560   4096     763      655     422   |   957      956
 *                                                                (TF/s of 1660 sustained)
 *
 * 128x256 owns the M=1024-2048 serving chunk (it is the smallest tile that still fills 256
 * CUs there: M=1024 x N=8192 is exactly 8 x 32 = 256 tiles); 192x256 owns M>=4096 and every
 * K-heavy shape, where filling is free and arithmetic intensity is the whole game. At
 * M=1024 128x256's 925.5 TF/s is PARITY with the Tensile assembly kernel measured on this
 * same shape (924.6, knob-contract 0-EXT-RESULT) -- the gap that measurement attributed to
 * Tensile's hand-scheduled software pipeline was, at that M, a missing tile.
 *
 * 192x256 is the tile every earlier sweep calls "c5" (test_kernels.hip, the Qwen prefill
 * -DGM_BM=192 object, the Tensile A/B). The name is kept so those records still resolve.
 * Its accumulator is 96 registers, and 128x256's is 64, against the 256x256 arm's 128 --
 * BOTH are strictly cheaper than an arm this object already carries, which is why a
 * megakernel sitting exactly at the 256/occ-2 cliff absorbs them for free (measured: total
 * 256 -> 256, occ 2 -> 2, spill 2 -> 2). LDS likewise: 126 KiB and 108 KiB against the
 * arena's 144 KiB.
 *
 * These are SEPARATE INSTANTIATIONS BEHIND SEPARATE OPCODES, not a runtime tile parameter.
 * That is settled, twice, by measurement already in the tree: a three-arm runtime d_gemm_cfg
 * switch cost 285 TF/s against 770 for the same tile compiled alone (see d_gemm below), and
 * test_kernels.hip:70 measured it spilling 3588 B/thread. The allocator budgets for the
 * worst arm of a SWITCH; it does not budget for the worst arm of a `case` in the interpreter
 * dispatch, whose live ranges do not overlap. GemmMed/GemmSmall have always worked this way.
 */
#define GM_WD_BM 128
#define GM_WD_BN 256
#define GM_WD_BK 64
#define GM_C5_BM 192
#define GM_C5_BN 256
#define GM_C5_BK 64
#define GM_C8_BM 128
#define GM_C8_BN 384
#define GM_C8_BK 64

__device__ void d_gemm_small(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                             unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_SM_BM, GM_SM_BN, GM_SM_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}
__device__ void d_gemm_med(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                           unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_MD_BM, GM_MD_BN, GM_MD_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}
__device__ void d_gemm_wide(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                            unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_WD_BM, GM_WD_BN, GM_WD_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}
__device__ void d_gemm_c5(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                          unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_C5_BM, GM_C5_BN, GM_C5_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}
#if PLOW_CDNA4
/* Internal gfx950 specialization of GEMM_WIDE. A 128x384 tile consumes 144 KiB of LDS, so it
 * cannot exist in the gfx942 object. It has no opcode of its own: the u8 opcode space is full,
 * and exec_gemm_wide selects it only when this tile maps exactly one workgroup to every CU. */
__device__ void d_gemm_c8(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                          unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_C8_BM, GM_C8_BN, GM_C8_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}
#endif

/* The mxfp4 (w4a16) twins of the four non-default rungs. Same tiles, same selector; only the
 * B-fetch differs (WFP4). Declared here rather than next to `d_gemm_mxfp4` because the tile
 * constants above are what they instantiate. */
PLOW_GM_MXFP4_TILE(d_gemm_med_mxfp4, GM_MD_BM, GM_MD_BN, GM_MD_BK)
PLOW_GM_MXFP4_TILE(d_gemm_small_mxfp4, GM_SM_BM, GM_SM_BN, GM_SM_BK)
PLOW_GM_MXFP4_TILE(d_gemm_wide_mxfp4, GM_WD_BM, GM_WD_BN, GM_WD_BK)
PLOW_GM_MXFP4_TILE(d_gemm_c5_mxfp4, GM_C5_BM, GM_C5_BN, GM_C5_BK)

/* ---------------------------------------------------------------------------
 * DENSE PREFILL BLOCK-FP8 GEMM (w8a16) — `d_gemm_fp8_blk`, opcode GEMM_FP8_BLK.
 *
 * The T-row twin of `gemv_rows_fp8_blk` (op 44) for a PLAIN [N,K] weight, against DeepSeek/GLM's
 * `weight_block_size: [128,128]` grid of arbitrary-f32 `weight_scale_inv`. It is the arm whose
 * absence made `GLM_LINEAR_FP8` decode-only: `emit_glm_mla_prefill`'s `o_proj` and
 * `emit_glm_block_prefill`'s three `shared_experts.*` projections had no T-row block-fp8 opcode to
 * lower to, so a stacked (prefill + decode) blob would have read fp8 bytes as bf16.
 *
 * NOT a re-route of what already existed, and the distinction is the point:
 *   - `GemmFp8`/`GemmGluFp8` (33/36) are the w8a8 rung — ONE f32 per output CHANNEL plus a per-row
 *     activation scale from `QuantFp8`. They cannot address a [128,128] grid and they need an fp8
 *     A operand this path does not produce.
 *   - ops 85/86 (`MoeGroupGluPf`/`MoeGroupDownPf`) DO carry a real block-fp8 prefill body, but they
 *     are grouped-MoE ops: expert weight/scale TABLES, `MoeAlignPf`'s row-count meta, `row_token`
 *     gather indices, `row_partidx`/`row_gate` scatter+scale maps, f32 `part[T*k,H]` output. A
 *     plain `o_proj` has none of that contract. (The DENSE FFN prefill borrows them anyway, with
 *     degenerate 1-expert routing — see `emit_glm_dense_block_prefill`. That trick works because a
 *     dense FFN IS an expert; it does not extend to `o_proj`.)
 *
 * ONE TILE RUNG, DELIBERATELY, and this is the one place this family departs from the bf16 / fp8 /
 * mxfp4 five-rung pattern. The promotion accumulator DOUBLES the register cost of a tile, and the
 * prefill object's worst case is set by the 256x256 arm's 128 accumulator registers:
 *
 *     tile      acc    +promotion   verdict
 *     64x128     16        32       fits
 *     128x128    32        64       fits            <- this rung
 *     128x256    64       128       ties the worst case
 *     192x256    96       192       OVER
 *     256x256   128       256       the whole AGPR file; cannot run 8 waves at all
 *
 * So a five-rung block-fp8 family is not merely expensive, its top two rungs do not exist. 128x128
 * is the largest rung that stays STRICTLY under the object's existing worst case, so adding it
 * cannot move the 256/occ-2 cliff `scripts/build_gfx950.sh` gates on. It is also the better of the
 * two feasible fills at GLM's TP4 shapes: `o_proj` (N=6144,K=4096) is 48 column-tiles here against
 * 24 at 128x256, and the shared gate/up (N=512) is 4 against 2. Because there is one rung it is
 * emitted DIRECTLY rather than through `pick_tile`/`gfx950_gemm_inventory` — a `QuantScheme::
 * BlockFp8` row there would have to name five opcodes, three of which cannot be built, and adding
 * rungs to the inventory re-stales the tunedb for every shape. If a second rung is ever wanted,
 * add 64x128 (narrow M) first and put BOTH in the inventory at that point.
 *
 * PREFILL-BUCKET ONLY. `interp.hip` dispatches this under `#if PLOW_BUCKET_PREFILL`, so the decode
 * object never sees it — the GF=8 lesson (a register-neutral arm that was a +32% decode regression
 * purely by growing the decode object inside the persistent megakernel) applies to any new MFMA
 * body, and this one has no business in a decode program: decode's block-fp8 is op 44/47. */
/* THE BLOCK-FP8 TILE IS ACCUMULATOR-BOUND, NOT LDS-BOUND. Do not size it from the LDS the way
 * the bf16 tile is sized -- that reasoning is the trap.
 *
 * WFP8BLK carries a SECOND accumulator: `accf` promotes `acc` at every 128-K scale boundary, so
 * the register cost is 2*SM*SN*16, twice the bf16 body's. Measured on gfx942 for
 * d_gemm_fp8_blk_k, all at LDS that fits:
 *
 *   192x256   256 VGPR, 63 SPILL, occ 2      <- the bf16 tile, and it spills badly here
 *   128x256   236 VGPR,  0 spill, occ 2
 *   128x128   148 VGPR,  0 spill, occ 2      <- the default, and what gfx950 runs
 *
 * So this stays at 128x128 even on a build whose bf16 tile is 192x256. Overriding it to match
 * the bf16 tile costs 63 spilled registers and, measured on the GLM block-fp8 prefill shapes,
 * up to 2.97x throughput. 128x256 only pays back at M >= 2048.
 *
 * Still overridable, because a 64 KiB part may need to shrink it further for LDS reasons:
 * 128x128x64 stages 73,728 B double-buffered, which a CDNA3 workgroup cannot hold, so a
 * double-buffered gfx942 build wants 64x128x64 (55,296 B). Single-buffered it fits as-is. */
#ifndef GM_BLK_BM
#define GM_BLK_BM 128
#endif
#ifndef GM_BLK_BN
#define GM_BLK_BN 128
#endif
#ifndef GM_BLK_BK
#define GM_BLK_BK 64
#endif
__device__ void d_gemm_fp8_blk(bf16* C, const bf16* A, const unsigned char* W,
                               const float* wscale, unsigned M, unsigned N, unsigned K,
                               unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_BLK_BM, GM_BLK_BN, GM_BLK_BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0),
             true, false, false, true>(C, A, (const bf16*)W, nullptr, nullptr, M, N, K, slice, nblk,
                                       lds, nullptr, 0, nullptr, wscale);
}

/* GEMM over gate|up in ONE pass, act(g)*u applied in the EPILOGUE. The prefill twin of
 * d_gemv_glu: three packets (gemm, gemm, glu) collapse to one, gt/ut never reach HBM, and
 * the GLU's own global gate goes with them. Same tile, same registers, same MFMA count --
 * see the wave->column remap in d_gemm_t. */
/* The GLU epilogue's wave->column map needs SN==2, i.e. the 8-wave grid. A 4-wave build (the
 * segmented-dispatch flash code object) omits it; plowc keeps gate/up as the tiled GEMM triple. */
#if PLOW_WAVES == 8
__device__ void d_gemm_glu(bf16* C, const bf16* A, const bf16* Bg, const bf16* Bu, unsigned M,
                           unsigned N, unsigned K, unsigned act, unsigned slice, unsigned nblk,
                           bf16* lds) {
    d_gemm_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true, true>(
        C, A, Bg, nullptr, nullptr, M, N, K, slice, nblk, lds, Bu, act);
}

/* MXFP4 (w4a16) GEMM over gate|up in ONE pass -- the fp4 twin of d_gemm_glu, and the arm whose
 * absence made the SHARED-EXPERT PREFILL the one place an mxfp4 packet paid for its encoding in
 * HBM TRAFFIC rather than only in a different weight fetch. Without it the pair unfuses into two
 * d_gemm_mxfp4 plus a Glu, which MATERIALISES gt and ut ([M, N] bf16 each) to HBM and reads them
 * both back: 8 B per output element of avoidable traffic, plus two extra packets and their gates.
 *
 * IT IS THE COMPOSITION OF TWO FLAGS THAT ALREADY EXISTED, not a third body. GLU owns the
 * EPILOGUE (the SN axis selects gate vs up, so both halves of an output element land in the same
 * lane) and WFP4 owns the B-FETCH (fp4 -> bf16 with the E8M0 scale folded exactly). They are
 * disjoint -- the fp4 dequant finishes before the tile reaches LDS, and the epilogue never sees a
 * weight -- so the two compose with ONE addition: the scale row has to follow the gate/up select,
 * which is `ssrc` in GM_FETCH. Same tile, same accumulator count, same MFMA count as
 * `d_gemm_mxfp4`; a workgroup emits BN/2 fused columns for the MFMA it used to spend on BN raw
 * ones, which is the same arithmetic because every output needs both halves anyway.
 *
 * Sg/Su are the gate and up E8M0 scale rows, K/32 bytes per row -- one per weight matrix, exactly
 * as `d_gemv_glu_mxfp4` takes at decode. 256x256 only, like its bf16 twin (the epilogue's
 * wave->column remap needs SN == 2); devgen takes this arm only where 256x256 is the winning fp4
 * rung at (M, 2N, K) -- see `glu_fusion_wins_mxfp4`, and note the 2N: a GLU tile emits NB = BN/2
 * columns, so the fused grid at width N is a plain 256x256 grid at width 2N.
 *
 * MEASURED at the K3 shared-expert prefill shape, 15 (T, N) points, gfx950 / 256 CU
 * (runtime/tests/gemm_glu_mxfp4_gfx950_test):
 *
 *   vs the SAME tile unfused   -38.8% .. -48.7% at 13 of the 15, +5% at the other two
 *   vs the BEST unfused rung   -20% .. -35% where the gate fires, +9% .. +35% where it does not
 *
 * and the mechanism behind BOTH columns is the wave count, not the byte count. The fused arm does
 * exactly the MFMA of the two GEMMs it replaces -- same tile count, same work per tile -- in ONE
 * kernel. So it wins a whole wave-time whenever its ceil(M/BM) x ceil(N/(BN/2)) grid fits one pass
 * over the 256 CUs where the unfused pair needs two kernels, and it is a WASH (the +5% pair: 384
 * tiles either way, 1.5 waves rounded to 2) when both need the same number of passes. The 8 B per
 * output element of round trip it also removes is the smaller term. */
__device__ void d_gemm_glu_mxfp4(bf16* C, const bf16* A, const unsigned char* Wg,
                                 const unsigned char* Wu, const unsigned char* Sg,
                                 const unsigned char* Su, unsigned M, unsigned N, unsigned K,
                                 unsigned act, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN, false, GM_SWZ, GM_WGM, (GM_PP != 0), true, true,
             true>(C, A, (const bf16*)Wg, nullptr, nullptr, M, N, K, slice, nblk, lds,
                   (const bf16*)Wu, act, Sg, nullptr, Su);
}
#endif

__device__ void d_gemm_norm(bf16* C, const bf16* A, const bf16* B, const float* rms,
                            const bf16* gamma, unsigned M, unsigned N, unsigned K, unsigned slice,
                            unsigned nblk, bf16* lds) {
    d_gemm_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN, true>(C, A, B, rms, gamma, M, N, K, slice,
                                                     nblk, lds);
}

/* ===========================================================================
 * FP8 (w8a8) PREFILL GEMM — a PARITY finding, NOT a 2x win. Kept as the substrate
 * for the occ-1 rewrite; NOT wired into dispatch.
 *
 * VERDICT (fp8-gemm profiling, standalone M=4096, Qwen shapes, gemm_fp8_c5 192x256):
 * the wide-K K64 f8f6f4 MFMA is correct (PASS, worst rel <0.0025) but delivers only
 * ~PARITY with the bf16 tile — q 594 vs bf16 603, gate/up 730 vs 778 (slower!), o 760
 * vs 696, down 972 vs 861 TF/s — NOT the ~2x the K64 MFMA rate allows, and 3-4x short of
 * hipBLASLt fp8 (q 2612, gate/up 2329, down 2779). rocprofv3: MfmaUtil DROPPED 30% (bf16)
 * -> 15% (fp8), MemUnitStalled ~0. The K64 op does 4x the work per instruction with 4x
 * fewer barriers, so the matrix pipe finishes even faster and idles MORE — the 4x barrier
 * cut bought nothing. This PROVES the wall is the LDS-read -> MFMA DEPENDENCY CHAIN itself
 * (each s_barrier is a fixed full-workgroup sync latency; each ds_read feeds the very next
 * MFMA with no independent work between), not the barrier COUNT and not memory. Two
 * independent measurements agree: bf16 barrier-halving gave only +7-9%; fp8's 4x barrier
 * cut gave parity. The ONLY path to the library's 1450 bf16 / 2600 fp8 is the occ-1 4-wave
 * deep-pipeline rewrite (512-reg budget -> deep REGISTER prefetch that breaks the
 * read->MFMA dependency, as CK's Intrawave v4/v5 pipelines do). fp8 rides on top of THAT.
 *
 * THE 2x IS REAL ON gfx950 BUT ONLY THROUGH THE WIDE-K INSTRUCTION. MEASURED (pure back-to-back
 * MFMA, this GPU): mfma_f32_32x32x16_bf16 == mfma_f32_32x32x16_fp8_fp8 == 2265 TF/s — the K16 fp8
 * THE 2x IS REAL ON gfx950 BUT ONLY THROUGH THE WIDE-K INSTRUCTION. MEASURED (pure back-to-back
 * MFMA, this GPU): mfma_f32_32x32x16_bf16 == mfma_f32_32x32x16_fp8_fp8 == 2265 TF/s — the K16 fp8
 * MFMA the "frozen contract" specified is NOT faster than bf16. The CDNA4 2x is delivered by
 * mfma_scale_f32_32x32x64_f8f6f4 (K=64, the MX f8/f6/f4 family): MEASURED 4532 TF/s = exactly 2x.
 * So this kernel uses THAT instruction. (The dense wide-K fp8 builtins mfma_f32_32x32x64_fp8 /
 * 16x16x128_fp8 do not exist in this ROCm 7.0.2 clang; only the *scale* f8f6f4 form and the sparse
 * smfmac do, so the scaled form with a NEUTRAL block scale is the dense path.)
 *
 * Differences from the bf16 d_gemm_t:
 *   1. Operands are 1 byte (fp8), so the LDS tile is bytes; the compact STRIDE is FBK=128 BYTES.
 *      A fragment is 32 fp8 = v8i32 (ds_read_b256), so the XOR swizzle moves 32-byte groups
 *      (GM8_XORSWZ, mask FBK/32-1<<5). The K-tile is FBK=128 = two K64 MFMAs, keeping the
 *      two-cluster ping-pong (FNSL=2). LDS at the 256 tile = 128 KiB < the bf16 144 KiB, so the
 *      256/occ-2 cliff is not moved (accumulator is the unchanged f32x16).
 *   2. The MFMA is mfma_scale_f32_32x32x64_f8f6f4 with cbsz=blgp=0 (both e4m3) and scale bytes 0
 *      (verified NEUTRAL = 2^0 on device). Its f32 accumulator layout is identical to the bf16
 *      32x32 MFMA, so mfma_acc_m/n, the wave->column GLU remap and the epilogue carry over. The
 *      A/B operand layout is: lane l holds row=l%32, k = kbase + 32*(l/32) + [0..31].
 *   3. The epilogue dequantizes acc[m][n] * a_scale[m] * w_scale[n]: a_scale the per-ROW activation
 *      scale (d_quant_fp8), w_scale the per-CHANNEL weight scale (offline). Applied once, never
 *      per element. (The MX per-32-block scale is left neutral; plow's scheme is per-row/channel.)
 *
 * DIRECT global_load_lds is NOT used (byte-granular per-lane DMA is fiddly, and the kernel is
 * MFMA-bound); the register FETCH+COMMIT path stages the tile. GLU rides the bf16 twin's SN trick. */
template <int BM, int BN, int BK, int WM, int WN, bool KEXACT = true, bool GLU = false>
__device__ void d_gemm_fp8_t(bf16* __restrict__ C, const unsigned char* __restrict__ A,
                             const unsigned char* __restrict__ B, const float* __restrict__ ascale,
                             const float* __restrict__ wscale, unsigned M, unsigned N, unsigned K,
                             unsigned slice, unsigned nblk, bf16* lds,
                             const unsigned char* __restrict__ B2 = nullptr,
                             const float* __restrict__ wscale2 = nullptr, unsigned act = 0) {
    (void)B2; (void)wscale2; (void)act;
    (void)BK;
    constexpr int THREADS = WM * WN * PLOW_WAVE;
    /* K64 MX-fp8 MFMA: one mfma_scale_f32_32x32x64_f8f6f4 consumes K=64. The K-tile is FBK=128 so
     * the ping-pong still has two MEM/MFMA cluster pairs (FNSL=2), each a single K64 MFMA. */
    constexpr int FMK = 64;                /* K per MFMA (vs bf16's 16) */
    constexpr int FBK = 128;               /* K-tile staged in LDS      */
    constexpr int STRIDE = FBK;            /* LDS row stride in BYTES (fp8), compact, no pad */
    constexpr int SM = BM / WM / MFMA_M;
    constexpr int SN = BN / WN / MFMA_N;
    constexpr int SLICE = FMK;             /* 64 */
    constexpr int NSL = FBK / SLICE;       /* 2 cluster pairs */
    constexpr int KS = SLICE / FMK;        /* == 1 */
    constexpr int APT = BM * FBK / THREADS;
    constexpr int BPT = BN * FBK / THREADS;
    constexpr int APASS = APT / 8;         /* 8-byte (8-fp8) load/commit passes */
    constexpr int BPASS = BPT / 8;
    constexpr int TILE = (BM + BN) * STRIDE; /* bytes per LDS buffer */
    constexpr bool DBUF = (GM_DBUF == 2);    /* must track the bf16 arena that sizes the LDS */
    constexpr bool CLBAR = (GM_CLBAR);       /* see the bf16 body: dead barriers at GM_DBUF=1 */
    constexpr bool CLFENCE = (GM_CLFENCE);
    /* PLR pays in the bf16 body and does NOT pay here, and the reason is the fragment WIDTH: an
     * fp8 K64 fragment is 32 B/lane (ds_read_b256, 8 VGPRs) against bf16's 16 B, so one extra
     * bank costs (SM+SN)*8 registers instead of (SM+SN)*4. Measured on gfx942, g31_gate_up
     * M=4096 / g31_q_proj M=128, fp8 TF/s: 192x256 560 -> 563 but 240 -> 256 VGPR with 14 spills,
     * and 64x128 at M=128 172 -> 148. Off by default; the knob stays so it can be re-priced if
     * the fp8 accumulator ever shrinks. */
    constexpr bool PLR = (GM_PLR8) && !CLBAR; /* local-register fragment prefetch */
    constexpr bool PRIO = (GM_PRIO) != 0;

    static_assert(APT % 8 == 0 && BPT % 8 == 0, "tile must stage in 8-byte (8-fp8) units");
    static_assert(BM % (WM * MFMA_M) == 0 && BN % (WN * MFMA_N) == 0, "tile must fit the wave grid");
    static_assert(NSL >= 2 || !CLBAR, "the ping-pong needs at least 2 MEM/MFMA cluster pairs");
    static_assert(THREADS == PLOW_THREADS, "GEMM wave grid must match the interpreter's");
    static_assert(!GLU || SN == 2, "the GLU epilogue uses the SN axis to select gate vs up");
    constexpr int NB = GLU ? BN / 2 : BN;
    constexpr bool PP = (GM_PP != 0) && DBUF; /* the ping-pong trades BUFFERS */

    unsigned char* const lds8 = (unsigned char*)lds;
#define GM8_ASM(b) (lds8 + (b) * TILE)
#define GM8_BSM(b) (lds8 + (b) * TILE + BM * STRIDE)
    /* 32-byte-granular XOR swizzle: a K64 fragment is 32 CONSECUTIVE fp8 (v8i32, ds_read_b256)
     * starting at a multiple of 32, so the permutation must move whole 32-byte groups to stay
     * aligned — XOR the 32-byte column with (row & (FBK/32-1)). Self-inverse; COMMIT (8-byte) and
     * READ_FRAGS (32-byte) apply it consistently, so a 32-byte read reassembles four 8-byte writes. */
#define GM8_XORSWZ(row, off) ((off) ^ (((row) & (FBK / 32 - 1)) << 5))
    /* k-byte this lane supplies for a K64 A/B fragment at cluster k-base kk: 32 fp8/lane, the two
     * lane-halves covering k[kk,kk+32) and k[kk+32,kk+64). */
    auto frag_k64 = [](unsigned ln, unsigned kk) { return kk + 32u * (ln / 32u); };

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / WN;
    const unsigned wn = wave % WN;
    const unsigned frow = mfma_frag_row(lane);

    const unsigned tn = (N + NB - 1) / NB, tm = (M + BM - 1) / BM;
    const unsigned n_tiles = tm * tn;

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned tile = gm_remap<GM_SWZ, GM_WGM>(lin, n_tiles, tm, tn);
        const unsigned m0 = (tile / tn) * BM, n0 = (tile % tn) * NB;

        f32x16 acc[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) acc[i][j] = (f32x16)(0.0f);

        __align__(8) unsigned char ra[APT], rb[BPT];

#define GM8_FETCH(k0)                                                                        \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned r = m0 + e / FBK;                                                      \
        const unsigned kk = (k0) + (e % FBK);                                                 \
        if constexpr (KEXACT) {                                                               \
            if (r < M) __builtin_memcpy(&ra[it * 8], as_glob(A) + (size_t)r * K + kk, 8);     \
            else _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;            \
        } else {                                                                              \
            _Pragma("unroll") for (int j = 0; j < 8; j++)                                     \
                ra[it * 8 + j] = (r < M && kk + j < K) ? A[(size_t)r * K + kk + j] : 0;       \
        }                                                                                     \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        const unsigned br = e / FBK;                                                          \
        const unsigned char* bsrc = B;                                                        \
        unsigned r = n0 + br;                                                                 \
        if constexpr (GLU) {                                                                  \
            const bool up = (br >= BN / 2);                                                   \
            bsrc = up ? B2 : B;                                                               \
            r = n0 + (up ? br - BN / 2 : br);                                                 \
        }                                                                                     \
        const unsigned kk = (k0) + (e % FBK);                                                 \
        if constexpr (KEXACT) {                                                               \
            if (r < N) __builtin_memcpy(&rb[it * 8], as_glob(bsrc) + (size_t)r * K + kk, 8);  \
            else _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;            \
        } else {                                                                              \
            _Pragma("unroll") for (int j = 0; j < 8; j++)                                     \
                rb[it * 8 + j] = (r < N && kk + j < K) ? bsrc[(size_t)r * K + kk + j] : 0;    \
        }                                                                                     \
    }

/* CDNA3 ONLY: the matrix core reads e4m3FNUZ, in which 0x80 is a NaN rather than OCP's -0.
 * Zeroing it here -- on the 8-byte group already sitting in registers, two SWAR words, before it
 * ever reaches LDS -- is the whole cost of running plow's OCP bytes through it. Every other byte
 * is exactly half its OCP value and PLOW_FP8_MMA_FIX puts that back in the epilogue.
 * A weight that quantised to negative zero would otherwise NaN the entire output tile. */
#if PLOW_CDNA4
#define GM8_FIX8(p) ((void)0)
#else
#define GM8_FIX8(p)                                                                          \
    do {                                                                                      \
        unsigned w_[2];                                                                       \
        __builtin_memcpy(w_, (p), 8);                                                         \
        w_[0] = plow_fp8_mask_neg0(w_[0]);                                                    \
        w_[1] = plow_fp8_mask_neg0(w_[1]);                                                    \
        __builtin_memcpy((p), w_, 8);                                                         \
    } while (0)
#endif
#define GM8_COMMIT(buf)                                                                      \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        GM8_FIX8(&ra[it * 8]);                                                                \
        __builtin_memcpy(&GM8_ASM(buf)[(e / FBK) * STRIDE + GM8_XORSWZ(e / FBK, e % FBK)],       \
                         &ra[it * 8], 8);                                                     \
    }                                                                                         \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                    \
        const unsigned e = threadIdx.x * 8 + it * (THREADS * 8);                              \
        GM8_FIX8(&rb[it * 8]);                                                                \
        __builtin_memcpy(&GM8_BSM(buf)[(e / FBK) * STRIDE + GM8_XORSWZ(e / FBK, e % FBK)],       \
                         &rb[it * 8], 8);                                                     \
    }

#define GM8_READ_INTO(af, bfr, buf, sl)                                                      \
    _Pragma("unroll") for (int i = 0; i < SM; i++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned arow = wm * (BM / WM) + i * MFMA_M + frow;                         \
            __builtin_memcpy(&af[i][q],                                                        \
                &GM8_ASM(buf)[arow * STRIDE +                                                  \
                              GM8_XORSWZ(arow, frag_k64(lane, (sl) * SLICE + q * FMK))], 32);  \
        }                                                                                     \
    _Pragma("unroll") for (int j = 0; j < SN; j++)                                            \
        _Pragma("unroll") for (int q = 0; q < KS; q++) {                                      \
            const unsigned brow =                                                              \
                (GLU ? j * (BN / 2) + wn * MFMA_N : wn * (BN / WN) + j * MFMA_N) + frow;       \
            __builtin_memcpy(&bfr[j][q],                                                       \
                &GM8_BSM(buf)[brow * STRIDE +                                                  \
                              GM8_XORSWZ(brow, frag_k64(lane, (sl) * SLICE + q * FMK))], 32);  \
        }

#define GM8_READ_FRAGS(buf, sl)                                                              \
    fp8v32 af[SM][KS], bfr[SN][KS];                                                           \
    GM8_READ_INTO(af, bfr, buf, sl)

/* cbsz=0 (A e4m3), blgp=0 (B e4m3), opsel=0, scale_a=0, scale_b=0.
 *
 * THE ZEROS ARE SAFE HERE FOR A REASON THAT DOES NOT GENERALISE, and this comment used to state
 * the reason wrongly ("scale byte 0 is the NEUTRAL (2^0=1) MX scale, verified on device"). E8M0
 * is BIASED BY 127: byte b means 2^(b-127), so neutral is 127 and byte 0 is 2^-127, which
 * flushes the product to exactly 0.0 (measured, rel err 1.0 — see the A4W4 contract in
 * amd_common.h). What makes it harmless in THIS kernel is that the scale arguments are
 * COMPILE-TIME CONSTANTS, so the backend selects the UNSCALED v_mfma_f32_32x32x64_f8f6f4 and
 * drops them entirely (verified in this object's disassembly: 34x unscaled, zero
 * v_mfma_scale_*). A RUNTIME scale operand selects the scaled form and byte 0 then silently
 * zeroes the output — use PLOW_E8M0_ONE, never 0. Per-row/per-channel dequant is applied in the
 * epilogue; the MX per-32-block scale is not used by this kernel at all. */
#define GM8_MFMA_FROM(af, bfr)                                                               \
    if constexpr (PRIO) __builtin_amdgcn_s_setprio(1);                                        \
    _Pragma("unroll") for (int q = 0; q < KS; q++)                                            \
        _Pragma("unroll") for (int i = 0; i < SM; i++)                                        \
            _Pragma("unroll") for (int j = 0; j < SN; j++) acc[i][j] =                        \
                plow_mfma_fp8_32x32(af[i][q], bfr[j][q], acc[i][j]);                          \
    if constexpr (PRIO) __builtin_amdgcn_s_setprio(0);
#define GM8_MFMA_BURST() GM8_MFMA_FROM(af, bfr)

#define GM8_FENCE()                                                                          \
    do {                                                                                      \
        if constexpr (CLFENCE) __builtin_amdgcn_sched_barrier(0);                             \
    } while (0)
#define GM8_CLUSTER_BARRIER()                                                                \
    do {                                                                                      \
        if constexpr (CLBAR) __builtin_amdgcn_s_barrier();                                    \
    } while (0)

        __syncthreads();
        GM8_FETCH(0);
        GM8_COMMIT(0);
        __syncthreads();

        const unsigned NT = (K + FBK - 1) / FBK;
        unsigned buf = 0;
        if (PP && wm == 1) __builtin_amdgcn_s_barrier();

#pragma unroll 1
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1) * FBK;
            if constexpr (!PLR) {
#pragma unroll
                for (int sl = 0; sl < NSL; sl++) {
                    GM8_FENCE();
                    GM8_READ_FRAGS(buf, sl)
                    if (sl == 0 && kn < K) { GM8_FETCH(kn); }
                    if (DBUF && sl == NSL - 1 && kn < K) { GM8_COMMIT(buf ^ 1); }
                    GM8_FENCE();
                    GM8_CLUSTER_BARRIER();

                    GM8_MFMA_BURST()
                    GM8_FENCE();
                    GM8_CLUSTER_BARRIER();
                }
            } else {
                /* Same rotation as the bf16 body: cluster sl+1's ds_read_b256 issues before
                 * cluster sl's MFMA burst. A fragment here is 32 B/lane, so one bank is
                 * (SM+SN)*KS*8 VGPRs -- twice the bf16 cost per fragment, which is why this is
                 * priced separately and not assumed to carry over. */
                fp8v32 afp[2][SM][KS], bfp[2][SN][KS];
                GM8_READ_INTO(afp[0], bfp[0], buf, 0)
#pragma unroll
                for (int sl = 0; sl < NSL; sl++) {
                    if (sl + 1 < NSL) { GM8_READ_INTO(afp[(sl + 1) & 1], bfp[(sl + 1) & 1], buf, sl + 1) }
                    if (sl == 0 && kn < K) { GM8_FETCH(kn); }
                    GM8_MFMA_FROM(afp[sl & 1], bfp[sl & 1])
                }
            }
            if constexpr (!DBUF) {
                /* One buffer: the commit waits until every wave has read the tile. Same trade as
                 * the bf16 body -- and required here too, because this arena is sized by
                 * GM_LDS_HALVES_T, so a two-buffer fp8 stage in a one-buffer allocation runs off
                 * the end. That is exactly what it did: 4/4 fp8 GEMM shapes wrong at GM_DBUF=1. */
                if (kn < K) {
                    if constexpr (!CLBAR) __syncthreads(); /* every wave done reading the stage */
                    GM8_COMMIT(0);
                    if constexpr (CLBAR) {
                        GM8_FENCE();
                        __builtin_amdgcn_s_barrier();
                    } else {
                        __syncthreads();
                    }
                }
            }
            if constexpr (DBUF) buf ^= 1;
        }
        if (PP && wm == 0) __builtin_amdgcn_s_barrier();
        __syncthreads();

        auto* const Cg = as_glob(C);
        if constexpr (GLU) {
            const unsigned nn = n0 + wn * MFMA_N + mfma_acc_n(lane);
            if (nn < N) {
                const float gs = wscale[nn] * PLOW_FP8_MMA_FIX,
                            us = wscale2[nn] * PLOW_FP8_MMA_FIX;
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int e = 0; e < 16; e++) {
                        const unsigned mm = m0 + wm * (BM / WM) + i * MFMA_M + mfma_acc_m(lane, e);
                        if (mm >= M) continue;
                        const float as = ascale[mm];
                        const float g = acc[i][0][e] * as * gs;
                        const float u = acc[i][1][e] * as * us;
                        const float sg = act_gate_only(g, act);
                        st_act1(&Cg[(size_t)mm * N + nn], f2bf(sg * u));
                    }
            }
        } else {
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (BN / WN) + j * MFMA_N + mfma_acc_n(lane);
                    if (nn >= N) continue;
                    const float ws = wscale[nn] * PLOW_FP8_MMA_FIX;
#pragma unroll
                    for (int e = 0; e < 16; e++) {
                        const unsigned mm = m0 + wm * (BM / WM) + i * MFMA_M + mfma_acc_m(lane, e);
                        if (mm < M)
                            st_act1(&Cg[(size_t)mm * N + nn],
                                    f2bf(acc[i][j][e] * ascale[mm] * ws));
                    }
                }
        }
#undef GM8_FETCH
#undef GM8_COMMIT
#undef GM8_READ_FRAGS
#undef GM8_MFMA_BURST
#undef GM8_FENCE
#undef GM8_ASM
#undef GM8_BSM
#undef GM8_XORSWZ
    }
}

__device__ void d_gemm_fp8(bf16* C, const unsigned char* A, const unsigned char* B,
                           const float* ascale, const float* wscale, unsigned M, unsigned N,
                           unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN>(C, A, B, ascale, wscale, M, N, K, slice, nblk,
                                                    lds);
}
__device__ void d_gemm_med_fp8(bf16* C, const unsigned char* A, const unsigned char* B,
                               const float* ascale, const float* wscale, unsigned M, unsigned N,
                               unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_MD_BM, GM_MD_BN, GM_MD_BK, GM_WM, GM_WN>(C, A, B, ascale, wscale, M, N, K, slice,
                                                            nblk, lds);
}
__device__ void d_gemm_small_fp8(bf16* C, const unsigned char* A, const unsigned char* B,
                                 const float* ascale, const float* wscale, unsigned M, unsigned N,
                                 unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_SM_BM, GM_SM_BN, GM_SM_BK, GM_WM, GM_WN>(C, A, B, ascale, wscale, M, N, K, slice,
                                                            nblk, lds);
}
/* The two added rungs, fp8 twins. They are here so PRECISION is a real selector input rather
 * than a label: fp8 halves the operand bytes without halving the MFMA work, so the tile that
 * balances fill against arithmetic intensity is not the same one bf16 picks, and a selector
 * that cannot express a different answer per precision cannot act on that. */
__device__ void d_gemm_wide_fp8(bf16* C, const unsigned char* A, const unsigned char* B,
                                const float* ascale, const float* wscale, unsigned M, unsigned N,
                                unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_WD_BM, GM_WD_BN, GM_WD_BK, GM_WM, GM_WN>(C, A, B, ascale, wscale, M, N, K,
                                                             slice, nblk, lds);
}
__device__ void d_gemm_c5_fp8(bf16* C, const unsigned char* A, const unsigned char* B,
                              const float* ascale, const float* wscale, unsigned M, unsigned N,
                              unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_C5_BM, GM_C5_BN, GM_C5_BK, GM_WM, GM_WN>(C, A, B, ascale, wscale, M, N, K,
                                                             slice, nblk, lds);
}
#if PLOW_WAVES == 8
__device__ void d_gemm_glu_fp8(bf16* C, const unsigned char* A, const unsigned char* Bg,
                               const unsigned char* Bu, const float* ascale, const float* gscale,
                               const float* uscale, unsigned M, unsigned N, unsigned K, unsigned act,
                               unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_fp8_t<GM_BM, GM_BN, GM_BK, GM_WM, GM_WN, true, true>(C, A, Bg, ascale, gscale, M, N, K,
                                                               slice, nblk, lds, Bu, uscale, act);
}
#endif

/* Per-row (per-token) fp8 activation quant — the w8a8 prefill's activation half. Each of the M rows
 * is owned by ONE wave: the 64 lanes stride K, wave-reduce the row absmax, set a_scale = absmax/448,
 * then quantize xq = round_e4m3(x / a_scale) with the native packed cvt. Rows are sliced across the
 * `nblk` workgroups; a workgroup takes PLOW_WAVES rows at a time. e4m3 has no inf (fn), max 448. */
/* T11 GLU FOLD (t3=gate, t4=up, i2=act on the packet; all absent on legacy ones). With `gate`
 * set, `x` is an OUTPUT: the wave that owns the row PRODUCES fu = act(gate)*up itself, stores
 * it, and takes the row amax from the bf16-ROUNDED value it just wrote — so xq/a_scale are
 * BIT-IDENTICAL to the `Glu` packet + a re-reading QuantFp8 this replaces (`act_gate_only` is
 * the same elementwise helper `d_glu` calls, and fmax is order-independent). What it deletes is
 * that packet, its global gate, and one full inter-width pass over `fu`.
 *
 * THIS ARM IS WHY `plow_t11_gluquant_arm` EXISTS. The emitter DROPS the `Glu` packet when it
 * folds, so an object without this arm quantizes an `fu` that was never computed: no fault, no
 * NaN, just a garbage FFN output, a wrong KV cache, and fluent wrong tokens. That is exactly
 * what shipped — the AMD dispatch ignored t3/t4 while the NVIDIA one honoured them — so the
 * pairing is refused at load rather than trusted (`PREFILL_ARM_MARKERS` in exec/amd.rs).
 *
 * THE LANE->ELEMENT MAP IS THE SAME IN BOTH LOOPS, AND IT HAS TO BE. The quant loop re-reads the
 * `fu` this function just stored; with the produce loop striding by lane (k = lane, +64) and the
 * quant loop by pairs (k = 2*lane, +128), lane l would read back bytes lane l^1 wrote, which is
 * a cross-lane dependency through memory with no waitcnt to order it. Both loops walk
 * {2l, 2l+1, 2l+128, ...}, so every element is re-read by the lane that produced it. */
__device__ void d_quant_fp8(unsigned char* __restrict__ xq_, bf16* __restrict__ x_,
                            float* __restrict__ ascale_, unsigned M, unsigned K, unsigned slice,
                            unsigned nblk, const bf16* __restrict__ gate_ = nullptr,
                            const bf16* __restrict__ up_ = nullptr, unsigned act = 0) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    auto* const xq = as_glob(xq_);
    auto* const xw = as_glob(x_);
    const auto* const x = as_glob((const bf16*)x_);
    const auto* const gg = as_glob(gate_);
    const auto* const ug = as_glob(up_);
    auto* const ascale = as_glob(ascale_);
    const unsigned per = (M + nblk - 1) / nblk;
    const unsigned m0 = slice * per;
    const unsigned m1 = (m0 + per < M) ? (m0 + per) : M;
    for (unsigned m = m0 + wave; m < m1; m += PLOW_WAVES) {
        const size_t row = (size_t)m * K;
        float amax = 0.0f;
        if (gate_) {
            for (unsigned k = lane * 2; k < K; k += PLOW_WAVE * 2) {
                for (unsigned j = k; j < k + 2 && j < K; j++) {
                    const bf16 fb = f2bf(act_gate_only(bf2f(gg[row + j]), act) * bf2f(ug[row + j]));
                    st_act1(&xw[row + j], fb);
                    amax = fmaxf(amax, fabsf(bf2f(fb)));
                }
            }
        } else {
            for (unsigned k = lane; k < K; k += PLOW_WAVE)
                amax = fmaxf(amax, fabsf(bf2f(x[row + k])));
        }
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, PLOW_WAVE));
        const float as = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
        const float inv = 1.0f / as;
        if (lane == 0) st_act<float>(&ascale[m], as);
        /* Quantize two elements at a time into one packed fp8 pair (byte lo/hi), then store the
         * byte. cvt_pk_fp8_f32(a, b, old, sel): sel=false writes bytes[0,1], and we only keep
         * bytes[0,1]; a single-element tail uses b=0. */
        for (unsigned k = lane * 2; k < K; k += PLOW_WAVE * 2) {
            const float a = bf2f(x[row + k]) * inv;
            const float b = (k + 1 < K) ? bf2f(x[row + k + 1]) * inv : 0.0f;
#if PLOW_CDNA4
            const unsigned pk = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
#else
            /* CDNA3's packed encoder is e4m3FNUZ, a DIFFERENT format (amd_arch.h). */
            const unsigned pk = (unsigned)plow_f32_to_fp8_ocp(a) |
                                ((unsigned)plow_f32_to_fp8_ocp(b) << 8);
#endif
            st_act1_u8(&xq[row + k], (unsigned char)(pk & 0xffu));
            if (k + 1 < K) st_act1_u8(&xq[row + k + 1], (unsigned char)((pk >> 8) & 0xffu));
        }
    }
}

/* ------------------------------- decode ---------------------------------- *
 * M is the decode batch (1..PLOW_GEMV_MAXM). One wave per output column n; the
 * 64 lanes split K, each issuing 16-byte loads, so the weight row is read fully
 * coalesced and exactly once. At M<=16 this beats the MFMA path outright: the
 * kernel is pinned to HBM bandwidth (31B decode reads 62.5 GB of weights, which
 * at ~8 TB/s is a ~7.8 ms/token floor), and a 32x32 matrix core would sit >50%
 * idle on the M axis while adding nothing.
 *
 * `x` is re-read once per n, but it is only M*K*2 bytes (11 KB at M=1) and lives
 * in L2, so it costs no HBM traffic. */
/* Overridable so the WIDE-ARM experiment (PLOW_GEMV_WALK, see below) can price MM=32/64
 * without editing the header. 16 stays the shipped ceiling. */
#ifndef PLOW_GEMV_MAXM
#define PLOW_GEMV_MAXM 16
#endif

/* MM is a compile-time batch bucket, and it MUST be: the accumulator has to be
 * indexed by a constant to stay in registers, and `break` inside the unrolled m
 * loop defeats the unroll (which spills it).
 *
 * Bucketing also protects the roofline. Each lane loads 16 B of W per step and
 * does 8*MM FMAs on it, i.e. MM FLOP/byte. The scalar FMA ceiling is ~72 TFLOP/s
 * (256 CU x 64 lanes x 2 FLOP x 2.2 GHz) against ~8 TB/s of HBM, so anything
 * above ~9 FLOP/byte stops being bandwidth-bound and starts being compute-bound.
 * A fixed MM=16 would therefore make M=1 decode — the single most latency-
 * critical path in the whole model — 16x more expensive than it needs to be. */
/* `norm`: 0 = none; 1 = apply a PRECOMPUTED row RMS from `rms` (a separate ROWRMS packet
 * produced it); 2 = compute the row RMS HERE, from the x this kernel already staged in LDS,
 * and delete the RMSNORM packet and its gate outright.
 *
 * MODE 2 WAS REMOVED ONCE, FOR GEMMA, AND THE REASON WAS FAN-OUT -- not the mode itself.
 * Measured 22.4 -> 24.4 ms/token there, because the norm feeding attention has FIVE
 * consumers (q, k, v off the input norm; gate, up off the pre-FFN norm), so folding it in
 * turns ONE shared reduction into FIVE redundant ones, each on its own GEMV's critical path.
 * The two gates saved did not pay for that. THE LAW STANDS: fusion that duplicates a
 * reduction across N consumers costs (N-1) extra reductions -- check N first.
 *
 * It is back because K3 answers that check differently, and the census is in
 * `perf-data/archive/k3/k3-decode-counter-graph.md` plus the fan-out pass in `k3.rs fuse_norm_gemv`:
 * 92 of K3's 116 decode RMSNORMs have fan-out ONE (`routed_expert_norm` -> the latent up
 * projection), so mode 2 there costs ZERO extra reductions, not four. K3 also pays far more
 * per packet than Gemma did: its decode graph is a 1831-level chain at mean width 1.41, so a
 * deleted packet is a deleted CHAIN LEVEL at ~5.7 us, with nothing else ready to fill it.
 * Measured: the 116 folds take the critical path 1831 -> 1715.
 *
 * THE SHAPE OF THE FUSION IS WHAT MAKES IT BIT-EXACT, and it is deliberately not mode 1's.
 * Mode 1 multiplies `x * rms * gamma` inside the k-loop, in f32, per element -- that is a
 * different number from what the deleted RMSNORM packet stored, which was `f2bf(x*inv*g)`,
 * ROUNDED TO BF16 and re-read from HBM by the GEMV. So mode 2 does not touch the hot loop at
 * all. It normalizes the STAGED COPY IN PLACE, rounding to bf16 exactly as `d_rmsnorm` does,
 * and then runs the ORDINARY un-normed hot loop over it. The bytes the k-loop reads out of
 * LDS are the same bytes the unfused pair wrote to and read from HBM, so the fused and
 * unfused programs are bit-identical by construction rather than by measurement.
 *
 * `gemv_norm_lds` below replicates `d_rmsnorm`'s `fits` path element-for-element -- same
 * PLOW_THREADS, same per-thread element map, same serial accumulation order, same
 * `block_sum`, same `rsqrtf(ss/feat + eps)` -- because the reduction ORDER is what fixes the
 * low bit of `inv`, and a "mathematically equivalent" reduction is not the same number.
 *
 * PRECONDITION: mode 2 requires the LDS-staged arm (`M*K <= GM_LDS_HALVES`) and the `fits`
 * path's bound (`K <= RN_REG*PLOW_THREADS`, `K % 8 == 0`). The emitter guarantees both --
 * `k3.rs fuse_norm_gemv` refuses otherwise -- and the arm re-checks and falls back to the
 * un-normed path rather than reading an unstaged LDS arena. */

/* Halves reserved at the TOP of the GEMM arena for the fused norm's cross-wave reduction,
 * kept clear of the staged row. PLOW_WAVES floats = 2*PLOW_WAVES halves, rounded to 16 B. */
#define GV_NORM_SCRATCH ((2u * PLOW_WAVES + 7u) & ~7u)

/* PLOW_GLM_FUSE_QNORM -- THE Q-NORM FOLD ARM on op 22 `GemvQkv` (opt-in build axis, default
 * OFF; the emit knob of the same name is what puts `t[7]` on the packet).
 *
 * GLM-5.2 decode runs `GemvQkv-A -> RmsNorm(q_a_layernorm) -> GemvQkv-G` and the middle packet
 * is ONE WORKGROUP. The decode census (perf-data/plow-gfx942/glm52-decode-packet-folds.md
 * section 1b) traces the window between the two GEMVs at 12.2 us for a 4.6 us body -- the
 * largest packet-boundary window left on the chain -- because a b=1 packet waits for the
 * slowest of ~149 workgroups, runs alone on 1 CU while 303 idle, and only then opens the next
 * gate. 63.7% of all in-packet CU time in that census is gate-wait.
 *
 * This arm computes the norm inside fusion G's staging, exactly the way `d_gemv_t`'s
 * `norm == 2` does, and the emitter stops emitting the packet. It is a BUILD AXIS as well as
 * an emit knob so an object built without it is byte-identical to one built before this
 * existed; a blob that carries the fold is refused on such an object by the
 * `plow_glm_fuse_qnorm_arm` marker (exec/amd.rs DECODE_ARM_MARKERS), because the failure mode
 * otherwise is an UNNORMED q_a row -- finite, fluent and wrong. */
#ifndef PLOW_GLM_FUSE_QNORM
#define PLOW_GLM_FUSE_QNORM 0
#endif

/* Normalize the LDS-staged activation in place: lds[k] = f2bf(bf2f(lds[k]) * inv * gamma[k]).
 * Structurally d_rmsnorm's `fits` path with the row read from LDS instead of HBM. */
__device__ __forceinline__ void gemv_norm_lds(bf16* __restrict__ lds, const bf16* __restrict__ g_,
                                              unsigned M, unsigned K, float eps, float* part) {
    const auto* const gg = as_glob(g_);
    for (unsigned m = 0; m < M; m++) {
        const size_t base = (size_t)m * K;
        /* PRODUCE: the row and its weight together, one round trip between them. */
        bf16v8 v[RN_VEC], w[RN_VEC];
#pragma unroll
        for (int c = 0; c < RN_VEC; c++) {
            const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
            v[c] = bf16v8_zero();
            w[c] = bf16v8_zero();
            if (i < K) {
                v[c] = ld_lds8(lds + base + i);
                if (gg) w[c] = ld_glob8(gg + i);
            }
        }
        float ss = 0.0f;
#pragma unroll
        for (int c = 0; c < RN_VEC; c++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = bf2f(v[c][j]);
                ss += f * f;
            }
        /* block_sum syncthreads on both sides, so the whole workgroup agrees on `inv` before
         * anyone overwrites the staged row it was reduced from. */
        const float inv = rsqrtf(block_sum(ss, part) / (float)K + eps);
#pragma unroll
        for (int c = 0; c < RN_VEC; c++) {
            const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
            if (i < K) {
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float gv = gg ? bf2f(w[c][j]) : 1.0f;
                    o[j] = f2bf(bf2f(v[c][j]) * inv * gv);
                }
                *(bf16v8*)(lds + base + i) = o;
            }
        }
    }
    /* The write-back is the LAST thing here, and block_sum's trailing __syncthreads happens
     * BEFORE it — so the barrier the hot loop needs is this one, not that one. Without it a
     * wave reads columns of the staged row that another wave has not renormalized yet: a
     * partially-normed activation, which is finite, plausible and wrong. */
    __syncthreads();
}

/* Memory-level parallelism. A single 16-byte load per lane per iteration leaves the
 * kernel LATENCY-bound, not bandwidth-bound: an HBM round trip is ~1 us and there is
 * nothing to overlap it with. Issuing GV_UNROLL independent loads up front puts that
 * many requests in flight per lane, which is what actually saturates HBM.
 *
 * This is the hottest loop in the whole model: decode streams all 57 GiB of weights for
 * every single token, so token latency IS weights / achieved-bandwidth. */
#ifndef GV_UNROLL
/* 10, not 8, and the reason is the RAGGED TAIL — not latency hiding.
 *
 * The main loop advances by `big` = GV_UNROLL * step = GV_UNROLL * 512 halves, and everything it
 * does not cover falls to a SCALAR tail that issues one load, waits for it, consumes it, and
 * repeats — no memory-level parallelism at all. So what matters is how well `big` divides K:
 *
 *     K = 5376  (q/k/v/gate/up — 67% of all the weight bytes)
 *         UN=8  -> big 4096, main covers 4096, TAIL = 1280 = 24% of every row
 *         UN=10 -> big 5120, main covers 5120, tail =  256 =  5%
 *     K = 21504 (down)      UN=10 -> 4 main passes + a 5% tail
 *     K = 8192  (o_proj)    UN=8 divides it EXACTLY; UN=10 leaves 37% in the tail  <- the one loss
 *
 * Measured standalone (N=16384, K=5376): fixing the tail is worth +16% at 256 CUs and +26% at
 * 128. Through the interpreter, across all the real shapes, UN=10 nets 16.9 -> 16.7 ms/token —
 * smaller, because o_proj pays for what the K=5376 shapes gain. A per-shape unroll would take
 * both; it needs a compile-time UN per bucket, and the registers are now there for it.
 *
 * §10 measured UN=10 as "+1% and 252 of 256 VGPRs — four from the cliff that has already rejected
 * an 8-wave dispatch once" and left it at 8. That objection is GONE: the GQA fusion reshaped the
 * register landscape and UN=10 now costs 211 VGPRs, FEWER than UN=8's 213. Free, and it stays.
 *
 * Do NOT "fix" the tail by unrolling it with predicated loads. TRIED: it issues GV_UNROLL loads
 * where only 2-3 are live, which is ~53% more instructions per row, and the real kernel went
 * 16.9 -> 20.6 ms/token (gemv 52 -> 70 us/inst). The GEMV is sensitive to instruction count, not
 * just to bytes — dead work is not free here. */
/* 11, and it is the number of 512-half chunks in a K=5376 row — which is not a coincidence.
 *
 * With `buf_ld8` there is no scalar tail: the k-loop runs over CHUNK INDEX and overshoots into
 * the hardware bounds check. So the only cost of a bad GV_UNROLL is DEAD LOADS in the last group
 * — which fetch nothing, but do occupy an instruction slot, and this kernel is measurably
 * sensitive to instruction count. Pick the unroll that divides the row.
 *
 *   K = 5376  -> 11 chunks (67% of all weight bytes)   K = 8192 -> 16   K = 21504 -> 42
 *
 *   GV_UNROLL   dead loads (5376/8192/21504)   token
 *        4            1 / 0 / 2                16.5 ms
 *        6            1 / 2 / 0                16.1
 *        8            5 / 0 / 6                16.5
 *       11            0 / 6 / 2                16.0     <- chosen
 *
 * And UN=11 was previously IMPOSSIBLE. The old loop tested `k + (UN-1)*step < K` with `k`
 * carrying the lane offset, so at UN=11 the condition went lane-DIVERGENT (only lanes < 32
 * entered). The chunk-indexed loop a buffer load allows is lane-independent, so buffer_load did
 * not just delete the tail — it also removed the constraint that capped the unroll at 10.
 *
 * A per-shape unroll (11 for K=5376, 8 for K=8192, 6 for K=21504) would take the last ~0.1 ms;
 * it needs a second GEMV opcode and plowc picking from K. Not yet done. */
#ifndef GV_UNROLL
#define GV_UNROLL 11
#endif
#endif
/* Non-temporal weight loads: the weight is streamed once and never re-read. */
#ifndef GV_NT
#define GV_NT 1
#endif
#if GV_NT
#define GV_LDW(p) ld_glob8_nt(p)
#else
#define GV_LDW(p) ld_glob8(p)
#endif

/* STAGE x INTO LDS, 16 BYTES AT A TIME.
 *
 * This was `for (i = threadIdx.x; i < n; i += PLOW_THREADS) lds[i] = x[i]` — a HALF at a time. At
 * K=7168 that is 14 `global_load_ushort` plus 14 `ds_write_b16` per thread, where two `dwordx4`
 * pairs would do: 7x the instructions, and the 2-byte LDS write is itself 2-way bank-conflicted
 * (64 lanes x 2 B lands two lanes per 4-byte bank) where `ds_write_b128` is not.
 *
 * It is only ~14 KB, so it costs no HBM traffic — but decode runs it 816 times a token in front of
 * a GEMV whose whole job is to keep the vector-memory pipe full, and the staging owns that pipe
 * until it finishes. MEASURED as 0.65-0.80 us of a 2.2 us packet at the narrow K=7168 shapes
 * (perf-data/k3_gemvbf16_bench, `stage_us` column). Widening it is worth, per packet:
 *
 *     f_a / k_rope_d / b_proj / kv_a  1.26-1.27x     router_gate 1.20x     q_a 1.12x
 *     the wide shapes (gv_per >= 14)  1.02-1.04x     lm_head 1.00x
 *
 * The tail loop keeps the half-at-a-time form for a ragged `n`; every K3 and Gemma projection has
 * K % 8 == 0 so it runs zero times there. */
__device__ __forceinline__ void stage_x_lds(bf16* __restrict__ lds, const bf16* __restrict__ x_,
                                            size_t n) {
    const auto* const x = as_glob(x_);
    size_t i0 = 0;
    /* Wave-uniform, so one scalar branch per packet. The alignment holds for every projection this
     * tree ships (K is a multiple of 64 and the tensor table is page-aligned), but `gemv_walk`
     * offsets x by `m0 * K`, so a future K with K % 8 != 0 would silently fault a 16-B load. Fall
     * back rather than fault. */
    if (((((uintptr_t)x_) | ((uintptr_t)lds)) & 15u) == 0u) {
        const size_t nv = n >> 3;
        for (size_t i = threadIdx.x; i < nv; i += PLOW_THREADS)
            *(bf16v8*)(lds + i * 8) = ld_glob8(x + i * 8);
        i0 = nv << 3;
    }
    for (size_t i = i0 + threadIdx.x; i < n; i += PLOW_THREADS) lds[i] = x[i];
}


/* XLDS selects where x lives, at COMPILE time, and that is the whole point of the split.
 *
 * This used to pick the source with a ternary:
 *
 *     const bf16* xsrc = x_lds ? lds : x;
 *
 * which merges an LDS pointer and a global pointer into one value -- so it is a GENERIC
 * pointer, and InferAddressSpaces gives up on the function. Every load in here then
 * compiled to `flat_load`, which resolves the aperture at run time and tracks on lgkmcnt
 * AND vmcnt. Not just the x reads: the whole WEIGHT STREAM went flat, and the 16-byte
 * loads decayed into `flat_load_ushort` -- two bytes at a time, in the loop that moves
 * 57 GiB per token. One ternary. */
/* PLOW_GV_DYNCLAIM — PROTOTYPE, default OFF: dynamic row claiming inside a decode GEMV packet.
 *
 * THE TAIL IT TARGETS (gfx950 TP8 K3 decode trace, scripts/k3_trace_wg.py): a b=256 GEMV packet
 * spans 14.9 us from its first gate-open to its last workgroup's end against a 9.2 us median
 * body. The last workgroup is not slow — it is LATE: it was the straggler of the previous packet,
 * arrives 2.6 us after the gate opened, and then still has to run a whole static slice
 * ([slice*per, slice*per+per) under GV_BLOCKED). Bodies also vary 3.3 us min-to-max. Both are
 * load-balance, and the interpreter's global queue only balances at SLICE granularity.
 *
 * MECHANISM. Every slice keeps a STATIC prefix of (100-GV_DYN_POOL_PCT)% of its rows — the
 * workgroup that claimed the slice always runs those, no atomic — and donates the rest to a
 * per-packet POOL of `rb`-row items (rb = one row per wave, or R per wave on the R-split arm).
 * Items are claimed with one agent-scope fetch-add by lane 0 of the LAST wave, broadcast through
 * one LDS word, and the claim for item i+1 is issued before item i is computed so its round trip
 * hides behind the loads (vmcnt is in-order, so the claimer is the wave with the least static
 * work). A late workgroup finds the pool drained and finishes after its prefix alone.
 *
 * WHERE THE CURSOR LIVES: u32[GV_DYN_CTR_WORD] of the packet's OWN counter line. The line is
 * unique to the packet (one counter per packet, dev_isa.h), 128 B wide with only u32[0] in use,
 * and re-zeroed with the counters every token. No host or blob change. Refused for SE_FINE
 * producers (their per-slice counters mean "these columns are done") and cross-GPU entries.
 *
 * BIT-EXACT: a row's chunk order, lane->k map, accumulation and wave_sum never depend on which
 * workgroup or wave runs it; only the row->workgroup map moves. `PLOW_FINE`'s gemv->headnorm
 * column map is the one consumer of that map, hence the SE_FINE refusal above.
 *
 * MEASURED IN ISOLATION AND NOT PROFITABLE AS BUILT (gfx950, runtime/bench/amd/k3_gemvbf16_bench
 * k_dyn vs k_dyn0, grid 256, bit-exact on every shape): o_proj 5.16 -> 6.05 us, shared_down
 * 4.73 -> 5.68, moe_down_latent 7.19 -> 8.12, q_absorb 5.09 -> 5.92, and the 2-4-row-per-slice
 * shapes double (routed_up 2.25 -> 4.73). Two costs: vmcnt is IN-ORDER, so the claiming wave's
 * first weight loads of every range wait for the returning atomic (~1 us more than an HBM load),
 * and each pool range costs two workgroup barriers that drain every wave's pipeline. Against
 * that, the true-critical-path analysis (scripts/k3_trace_critpath.py) puts the balanced-claim
 * ceiling on the b=256 GEMVs at ~1.5 us/packet (the tail there is random per-packet body
 * variance, (max-median)/median p50 0.29, not slice- or CU-bound), i.e. about what the
 * mechanism costs. Kept as a knob for a cheaper claim (a wave with no weight stream, or a
 * per-wave claim without barriers); do not ship this form. */
#ifndef PLOW_GV_DYNCLAIM
#define PLOW_GV_DYNCLAIM 0
#endif
/* Decode objects only. The flag arrives through PLOW_HSACO_EXTRA_DEFINES, which every object
 * sees; the prefill buckets must stay byte-identical so a decode A/B is a decode A/B. */
#if PLOW_GV_DYNCLAIM && !PLOW_BUCKET_DECODE
#undef PLOW_GV_DYNCLAIM
#define PLOW_GV_DYNCLAIM 0
#endif
#if PLOW_GV_DYNCLAIM
#ifndef GV_DYN_POOL_PCT
#define GV_DYN_POOL_PCT 50
#endif
#ifndef GV_DYN_MIN_BLK
#define GV_DYN_MIN_BLK 64
#endif
#define GV_DYN_CTR_WORD 16u
#define GV_DYN_SLOT_HALVES 8u
#ifndef GV_DYN_SLOT_OFS
#define GV_DYN_SLOT_OFS (GM_LDS_HALVES - GV_NORM_SCRATCH - GV_DYN_SLOT_HALVES)
#endif
#define GV_DYN_CLAIM_TID (PLOW_THREADS - PLOW_WAVE)
#define GV_DYN_PARAM , uint32_t* gv_cur
#define GV_DYN_PARAM_DEF , uint32_t* gv_cur = nullptr
#define GV_DYN_ARG , gv_cur
__device__ __forceinline__ bool gv_dyn_on(const uint32_t* cur, unsigned nblk, bool xlds,
                                          size_t staged_halves) {
    return cur != nullptr && xlds && nblk >= (unsigned)GV_DYN_MIN_BLK &&
           staged_halves + GV_NORM_SCRATCH + GV_DYN_SLOT_HALVES <= GM_LDS_HALVES;
}
/* Walk this slice's static prefix, then pool items until the packet's pool is empty. `run(r0, r1)`
 * computes rows [r0, r1) with every wave participating; it is called a workgroup-uniform number of
 * times, so it may contain barriers of its own. Every thread returns together. */
template <class F>
__device__ __forceinline__ void gv_dyn_walk(uint32_t* cur, const bf16* lds, unsigned per,
                                            unsigned n0, unsigned n1, unsigned nblk, unsigned N,
                                            unsigned rb, F&& run) {
    const unsigned pool = per * (unsigned)GV_DYN_POOL_PCT / 100u;
    const unsigned sp = per - pool;
    const unsigned ipb = (pool + rb - 1u) / rb;
    const unsigned n_items = nblk * ipb;
    unsigned* const slot = (unsigned*)(const_cast<bf16*>(lds) + GV_DYN_SLOT_OFS);
    const bool claimer = threadIdx.x == (unsigned)GV_DYN_CLAIM_TID;
    unsigned nxt = 0;
    if (claimer) nxt = __hip_atomic_fetch_add(cur, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    unsigned r0 = n0, r1 = (n0 + sp < n1) ? n0 + sp : n1;
    for (;;) {
        run(r0, r1);
        __syncthreads(); /* every wave is done with the previous range; the slot is free */
        if (claimer) *slot = nxt;
        __syncthreads();
        const unsigned it = __builtin_amdgcn_readfirstlane(*slot);
        if (it >= n_items) break;
        if (claimer) nxt = __hip_atomic_fetch_add(cur, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned s = it / ipb;
        const unsigned e = (s * per + per < N) ? s * per + per : N;
        r0 = s * per + sp + (it - s * ipb) * rb;
        r1 = (r0 + rb < e) ? r0 + rb : e;
    }
}
#else
#define GV_DYN_PARAM
#define GV_DYN_PARAM_DEF
#define GV_DYN_ARG
#endif
template <int MM, bool XLDS, bool NORM, int UN = GV_UNROLL>
__device__ __forceinline__ void gemv_rows(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                          const bf16* __restrict__ W_, const float* __restrict__ rms_,
                                          const bf16* __restrict__ gamma_, unsigned M, unsigned N,
                                          unsigned K, unsigned slice, unsigned nblk,
                                          const bf16* lds GV_DYN_PARAM_DEF) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned n_waves = nblk * PLOW_WAVES;
    const unsigned step = PLOW_WAVE * 8; /* one pass: 64 lanes x 8 halves */
    /* Chunks of one weight row. The k-loop covers ceil(K/step) of them and OVERSHOOTS into the
     * hardware bounds check rather than dropping the remainder into a scalar tail. */
    const unsigned nchunk = (K + step - 1) / step;

    /* Everything that came out of the tensor table, promoted to addrspace(1). See as_glob()
     * in amd_common.h: without this the weight stream is flat_load, two bytes at a time. */
    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    const auto* const rms = as_glob(rms_);
    const auto* const gamma = as_glob(gamma_);

    /* x, from wherever it lives — the choice is made at COMPILE time (XLDS), never with a
     * ternary. `xsrc = x_lds ? lds : x` would merge an LDS pointer with a global one into a
     * GENERIC pointer, and InferAddressSpaces then gives up on the whole function. */
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* WHICH OUTPUT COLUMNS THIS WORKGROUP OWNS. A compile-time choice, never a runtime one.
     *
     * INTERLEAVED (default, GV_BLOCKED=0): workgroup `slice` owns the columns
     * `= [8*slice, 8*slice+8) (mod nblk*8)` -- scattered across the whole of N. At any instant
     * all `nblk*8` waves are walking one contiguous window of W's rows, so the machine's
     * aggregate read front stays contiguous. That is the case FOR this form.
     *
     * BLOCKED (GV_BLOCKED=1): workgroup `slice` owns the contiguous run
     * `[slice*per, slice*per + per)`. This is what per-slice gates need: a head is 256
     * consecutive columns of the q projection, so under the interleaved form head `h` is
     * produced by EVERY workgroup (verified: 128 of 128), and a fine gate on gemv->headnorm
     * degenerates to the coarse one. Blocked, head `h` needs only ceil(256/per) of them.
     *
     * The cost is 128 scattered sequential streams instead of one moving front. HBM likes many
     * open pages, so this is probably free -- but "probably" is what the head-major KV relayout
     * said before it returned 3% against a predicted 2x. MEASURE IT. */
/* DEFAULT 1, on both counts: it is FASTER (16.8 vs 17.1 ms/token on the real 31B decode,
 * repeated — 688 KB of dead-sequential W per workgroup is what the memory system wants, and
 * the "contiguous aggregate read front" argument for interleaving simply did not survive
 * measurement), and PLOW_FINE's gemv->headnorm dependency map ASSUMES it. Building the kernel
 * with GV_BLOCKED=0 and a packet with PLOW_FINE=1 would emit dependencies that are silently
 * WRONG — the failure mode is a wrong answer, not an error. Keep them in step. */
/* PROBE / CANDIDATE: ONE WAVE-UNIFORM DESCRIPTOR PER WORKGROUP, not one per ROW.
 *
 * The waterfall note above the row loop prices `buf_rsrc(W + n*K, K)` at ~13 extra
 * instructions per 16-byte load: LLVM cannot prove the base wave-uniform, so it wraps every
 * `buffer_load_dwordx4` in a readfirstlane rebuild loop. THE REASON IT CANNOT is that `n`
 * descends from `threadIdx.x >> 6` -- uniform within a wave, divergent as far as the compiler
 * is concerned.
 *
 * Under GV_BLOCKED the workgroup's rows are CONTIGUOUS in W -- `[gv_n0, gv_n1)` -- so one
 * descriptor spans all of them, and `gv_n0 = slice * gv_per` descends only from kernel
 * arguments. That base IS provably uniform, which is a different thing from readfirstlane'ing
 * a divergent one (tried, and recorded above as having failed to remove the waterfall).
 *
 * The cost is that the per-row bounds check goes with it: `num_records` now covers the whole
 * block, so a chunk past row `n`'s end reads into row `n+1` instead of returning zero. That
 * overshoot is real (nchunk = ceil(K/512); at K=3840 chunk 7 runs 256 halves past the row), so
 * the offset is steered out of bounds explicitly. One v_cndmask per load against ~13 waterfall
 * instructions.
 *
 * MEASURED: EXACTLY NOTHING. Gemma-4-12B, ctx 4096, 48 steps, three interleaved triples,
 * ms/token -- base 21.322 21.270 21.329 (mean 21.307) vs GV_URSRC=1 21.306 21.292 21.314
 * (mean 21.304). Which CONFIRMS the waterfall note's own conclusion by a different route: at
 * M=1 the issue pipe is ~23% busy, so removing instructions from it cannot buy time. Kept at 0
 * -- the shipped per-row descriptor is simpler and measures the same. */
#ifndef GV_URSRC
#define GV_URSRC 0
#endif
/* Force every GEMV row descriptor into SGPRs (buf_rsrc_u) so the load clause is not wrapped in a
 * per-load exec-mask waterfall. Orthogonal to GV_URSRC: that one changes WHICH descriptor, this
 * one changes whether the compiler can keep it SCALAR.
 *
 * In a megakernel the compiler can never work this out for itself -- the op's operands are read
 * out of the packet into VGPRs, so `W + n*K` is VGPR-resident however wave-invariant its value
 * is, and LLVM wraps every buffer_load_dwordx4 in a readfirstlane rebuild loop that ends in a
 * BACKWARD BRANCH. That branch is the real cost: it puts each load in its own basic block, so
 * the UN loads the source issues "before touching any of them" are not a clause at all.
 *
 * Applied to ALL the GEMV bodies, not just gemv_rows. The shipped decode object carried the
 * waterfall in d_gemv_fp8_blk, d_gemv_qkv and d_gemv_glu too -- and d_gemv_glu pays it TWICE per
 * row, once for gate and once for up. GemvGlu + GemvQkv are 57% of the decode GEMV span, so
 * patching only gemv_rows measured almost nothing (-0.45%) while patching all of them measures:
 *
 *   interp_decode_gq   v_readfirstlane_b32 1385 -> 967   s_cbranch_execnz 579 -> 461
 *   Gemma-4-12B ctx 4096, 48 steps, three interleaved pairs, ms/token:
 *     off 21.023 21.036 21.063   on 20.955 20.907 20.912     -0.55%, ranges DISJOINT
 *
 * Register-neutral (253 VGPR, 0 spill) and token-identical over the real 12B asset.
 *
 * CDNA3 only, for the same reason FA_FAST_RCP is: this changes codegen on a shipped, validated
 * target and there is no gfx950 here to re-run its goldens on. The note above gemv_rows' loop
 * records an earlier gfx950 attempt at this that "does nothing" -- worth re-measuring there. */
#ifndef GV_USCALAR
#define GV_USCALAR (!PLOW_CDNA4)
#endif
#if GV_USCALAR
#define PLOW_GV_RSRC(p, n) buf_rsrc_u((p), (n))
#else
#define PLOW_GV_RSRC(p, n) buf_rsrc((p), (n))
#endif
/* PROBE ONLY, AND IT COMPUTES THE WRONG ANSWER. Replaces the per-row cross-lane reduction with
 * a lane-local value to price `wave_sum` -- one 6-step DPP reduction per output row. Never
 * ship it; it exists so "the per-row constant" can be split into its two halves.
 *
 * MEASURED: 21.273 vs 21.307 ms/token, i.e. deleting the ENTIRE cross-lane reduction -- and
 * returning a wrong answer -- buys 0.16%. Together with GV_URSRC's zero, that retires "the
 * per-row constant" as an explanation for the 1.5-1.7x against hipBLASLt: the descriptor
 * rebuild costs nothing and the reduction costs 0.16%. The GEMV's time is in the LOADS. */
#ifndef GV_HACK_NOSUM
#define GV_HACK_NOSUM 0
#endif
#ifndef GV_BLOCKED
#define GV_BLOCKED 1
#endif
#if GV_BLOCKED
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    (void)n_waves;
#if GV_URSRC
    /* Built ONCE, from a base that descends only from `slice` -- see GV_URSRC above. */
    const unsigned blk_rows = (gv_n1 > gv_n0) ? (gv_n1 - gv_n0) : 0u;
    const unsigned blk_halves = blk_rows * K;
    const __amdgpu_buffer_rsrc_t wrb = buf_rsrc(W + (size_t)gv_n0 * K, blk_halves);
#endif
#if PLOW_GV_DYNCLAIM
    /* The per-row body, so the dynamic-claim item loop below can run the SAME code over a
     * claimed range. Bit-exact by construction: nothing in a row depends on `slice`. */
    auto gv_row = [&](unsigned n) {
#else
    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES) {
#endif
#else
    for (unsigned n = slice * PLOW_WAVES + wave; n < N; n += n_waves) {
#endif
        /* ONE DESCRIPTOR PER ROW, and THEREFORE NO TAIL.
         *
         * `num_records = K` halves, so every chunk past the end of the row returns zero from the
         * hardware and fetches nothing. The row is a single uniform unrolled loop: no scalar
         * tail, no predication, no divergence, and nothing to break the compiler's software
         * pipeline (it runs ~20 loads deep across this loop; the old dependent scalar tail is
         * what used to stop it). See buf_ld8 in amd_common.h for the measurements.
         *
         * The OOB chunks need no `live` flag anywhere: wv == 0, so dot8 contributes exactly 0.
         * The ONLY guard needed is on the `x` index — LDS past the staged activation is whatever
         * the previous op left in the arena, and `0 * NaN` is NaN, not 0. One cndmask per chunk,
         * hoisted out of the m loop. */
        /* A readfirstlane'd (SGPR) descriptor was TRIED HERE AND IT DOES NOTHING — but NOT for the
         * reason first recorded here, and the correction matters because the wrong reason invites
         * the next reader to re-buy it.
         *
         * The waterfall IS IN THE SHIPPED MEGAKERNEL. `buf_rsrc` takes a VGPR-resident base, LLVM
         * cannot prove it wave-uniform, and wraps every `buffer_load_dwordx4` in a rebuild loop.
         * Disassembling `plow_exec` in interp_decode_fp8kv_k3.elf, one 14-load clause of THIS body
         * contains 52 `v_readfirstlane_b32`, 26 `v_cmp_eq_u64_e64`, 14 `s_and_saveexec_b64`,
         * 13 `s_xor_b64 exec`, 13 `s_cbranch_execnz` and 14 `s_nop` around its 14 loads — ~13 extra
         * instructions per 16-byte load, in all four instantiations. The earlier note read "526 vs
         * 526 `s_cbranch_execnz`, so there is no waterfall here"; the equal counts actually say the
         * readfirstlane transform FAILED TO REMOVE one that is very much present.
         *
         * It still does not pay, because at M=1 the ISSUE PIPE IS NOT THE BINDING RESOURCE. Per
         * chunk a wave issues ~6 useful instructions (one buffer_load, one ds_read_b128, four
         * v_dot2c_f32_bf16) and ~14 waterfall ones; per CU that is 8 waves x ~20 = ~160 wave-
         * instructions over 4 SIMDs at 4 cycles = ~160 cycles, against 8 KB of weights that take
         * ~690 cycles at the 25 GB/s a CU gets out of a 6.4 TB/s budget shared 256 ways. The pipe
         * is ~23% busy and the waterfall is ~15% of it, so deleting it outright cannot buy more
         * than a couple of percent — which is inside the 37.159 vs 36.720 ms/token the transform
         * measured end to end. Do not re-buy it; `buf_rsrc_fp8_u` exists for the paths where the
         * arithmetic is different. What DOES pay on this body is in-flight WORK, not issue slots:
         * see gemv_rows_r. */
#if GV_URSRC
        const unsigned row_off = (n - gv_n0) * K; /* halves, into the block descriptor */
#else
        const __amdgpu_buffer_rsrc_t wr = PLOW_GV_RSRC(W + (size_t)n * K, K);
#endif
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;

        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            /* Issue all UN loads before touching any of them. */
#pragma unroll
            for (int u = 0; u < UN; u++) {
#if GV_URSRC
                const unsigned e = (c + (unsigned)u) * step + lane * 8;
                /* Past THIS row the block descriptor would hand back row n+1, so steer the
                 * offset beyond num_records and let the same hardware check return zero. */
                wv[u] = buf_ld8(wrb, (e < K) ? ((row_off + e) * 2u) : (blk_halves * 2u));
#else
                wv[u] = buf_ld8(wr, ((c + (unsigned)u) * step + lane * 8) * 2u);
#endif
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 8;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
                bf16v8 gv;
                if constexpr (NORM) gv = ld_glob8(gamma + kx);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xv = xv8(live ? m : 0, kx);
                    float a = 0.0f;
                    if constexpr (NORM) {
#pragma unroll
                        for (int j = 0; j < 8; j++) a += bf2f(wv[u][j]) * bf2f(xv[j]) * bf2f(gv[j]);
                        a *= rms[live ? m : 0]; /* per-ROW scalar: hoisted out of j, not per element */
                    } else {
                        a = dot8(wv[u], xv, a);
                    }
                    acc[m] += live ? a : 0.0f;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
#if GV_HACK_NOSUM
            const float t = acc[m]; /* WRONG ANSWER BY CONSTRUCTION -- see GV_HACK_NOSUM */
#else
            const float t = wave_sum(acc[m]);
#endif
            if (lane == 0 && (unsigned)m < M) st_act1(&C[(size_t)m * N + n], f2bf(t));
        }
#if PLOW_GV_DYNCLAIM && GV_BLOCKED
    };
    if (gv_dyn_on(gv_cur, nblk, XLDS, (size_t)M * K)) {
        gv_dyn_walk(gv_cur, lds, gv_per, gv_n0, gv_n1, nblk, N, PLOW_WAVES,
                    [&](unsigned r0, unsigned r1) {
                        for (unsigned n = r0 + wave; n < r1; n += PLOW_WAVES) gv_row(n);
                    });
        return;
    }
    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES) gv_row(n);
#else
    }
#endif
}

/* ------------- R OUTPUT COLUMNS PER WAVE-STEP, for the SHORT-ROW shapes ------------------
 *
 * THE DEFECT THIS FIXES. A wave keeps `min(UN, nchunk)` 16-byte weight loads outstanding, and
 * `nchunk = ceil(K/512)` — so on a SHORT row the unroll is capped by the ROW, not by UN, and no
 * value of GV_UNROLL can help. The wave then runs `ceil(gv_per/PLOW_WAVES)` SERIAL rounds of
 * "issue nchunk loads, wait the full HBM latency, consume, wave_sum". K3 emits four such shapes
 * and they are 20% of the family's bytes:
 *
 *     o_proj      N=7168 K=1536  nchunk=3   93/token     shared_down N=7168 K= 768 nchunk=2  92
 *     q_absorb    N=6144 K=1536  nchunk=3   24/token     moe_up_lat  N=7168 K=3584 nchunk=7  92
 *
 * R independent columns multiply the outstanding loads by R and divide the round count by R. It
 * is the same transform `gemv_blk_rows_r` (op 44) and `gemv_mx_rows_r` already carry, and for the
 * same reason; the bf16 row simply never had it because at K=5376/7168 nchunk already exceeds UN.
 *
 * MEASURED, gfx950, grid = the emitter's own nblk, blockDim 512, in-kernel rep loop over a 1.5 GB
 * arena, hipEvent median-of-41, palindromic interleave, A/A control 0.998-1.003
 * (perf-data/k3_gemvbf16_bench.{hip,cpp}). Speedup over the shipping single-column body:
 *
 *     o_proj 1.42x   shared_down 1.43x   q_absorb 1.26x   moe_up_latent 1.14x   q_rope 1.04x
 *     nchunk >= 8 shapes: 0.99-1.00x  (which is why they keep the single-column arm)
 *
 * R=2 AND NOT R=4, and the reason is the RAGGED TAIL. R=4/UN=3 was measured too: it reads 1.21x at
 * o_proj against R=2's 1.42x, because gv_per=28 deals waves 0-3 four columns and waves 4-7 three,
 * so half the waves complete ZERO full R=4 groups and run their whole column count through the
 * R=1 tail. R=2 divides every K3 gv_per far more often. UN=7 keeps R*UN = 14 vectors = 56 VGPRs,
 * exactly the register footprint `GV_UNROLL=14` already budgets, so this is register-neutral.
 *
 * BIT-EXACT: each column keeps its own chunk order, lane->k map, accumulation order and wave_sum;
 * only WHICH WAVE takes which column moves. Checked elementwise on device across all 14 K3 shapes.
 * The workgroup's column OWNERSHIP is unchanged ([gv_n0, gv_n1)), so PLOW_FINE's gemv->headnorm
 * dependency map is unaffected.
 *
 * CONFIRMED IN THE INTERPRETER, not only standalone. `PLOW_TRACE_RAW` (see amd-bench) records
 * (t_arrive, t_ready, t_end) per (workgroup, packet) off `s_memrealtime`; summing
 * max_wg(t_end) - max_wg(t_ready) over the 816 `Gemv` packets of a steady-state decode step on the
 * full 93-layer TP8 fp8-KV asset at ctx 32000, this change plus the wide activation staging is
 *
 *     op 10 body   base 11.251 11.063 11.051 11.078 10.974 10.924 11.154   mean 11.071 ms
 *                  new  10.774 10.576 10.512 10.523 10.596 10.637 10.608   mean 10.604 ms
 *
 * -0.467 ms, -4.2%, with the two RANGES DISJOINT (min base 10.924 > max new 10.774). No other op
 * family moves. End to end, quiet-box runs only: 39.138 -> 38.768 ms/token (-0.37). */
/* The R-split's shape, as knobs so a sweep never needs to edit the dispatch. R*UN must stay <= the
 * 14 vectors GV_UNROLL already budgets, or the arm grows the register union. GV_RS_R=0 disables
 * the split outright (the arm is then never instantiated). */
#ifndef GV_RS_R
#define GV_RS_R 2
#endif
#ifndef GV_RS_UN
#define GV_RS_UN 7
#endif
/* Take the R-split only when `nchunk` is BELOW this. Measured: at nchunk >= 8 the single-column
 * body already holds 8-14 loads per wave and the split reads 0.98-1.00x, so 8 is where it stops
 * paying — not R*UN, which is a register bound and not a bandwidth one. */
#ifndef GV_RS_MAXNCH
#define GV_RS_MAXNCH 8
#endif
template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_rows_r(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                            const bf16* __restrict__ W_, unsigned M, unsigned N,
                                            unsigned K, unsigned n, unsigned lane,
                                            const bf16* lds) {
    const unsigned step = PLOW_WAVE * 8;
    const unsigned nchunk = (K + step - 1) / step;
    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };
    __amdgpu_buffer_rsrc_t wr[R];
#pragma unroll
    for (int r = 0; r < R; r++)
        wr[r] = PLOW_GV_RSRC(W + (size_t)(n + (unsigned)r * PLOW_WAVES) * K, K);
    float acc[R][MM];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MM; m++) acc[r][m] = 0.0f;

    for (unsigned c = 0; c < nchunk; c += UN) {
        bf16v8 wv[R][UN];
        /* every column's loads issued before any is touched: R*min(UN,nchunk) outstanding */
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[r][u] = buf_ld8(wr[r], ((c + (unsigned)u) * step + lane * 8) * 2u);
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * step + lane * 8;
            const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                const bf16v8 xv = xv8(live ? m : 0, kx);
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][m] += live ? dot8(wv[r][u], xv, 0.0f) : 0.0f;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; m++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wave_sum(acc[r][m]);
            if (lane == 0 && (unsigned)m < M)
                st_act1(&C[(size_t)m * N + n + (unsigned)r * PLOW_WAVES], f2bf(t));
        }
}

/* FULL R-GROUPS ONLY, then the leftover columns one at a time. The tail runs at UN = GV_UNROLL so
 * it is the SAME instantiation shape the single-column arm already carries — the R-split therefore
 * costs the object exactly TWO new bodies, not four. */
template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_rows_rs(bf16* __restrict__ C, const bf16* __restrict__ x,
                                             const bf16* __restrict__ W, unsigned M, unsigned N,
                                             unsigned K, unsigned slice, unsigned nblk,
                                             const bf16* lds GV_DYN_PARAM_DEF) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
#if PLOW_GV_DYNCLAIM
    auto run = [&](unsigned r0, unsigned r1) {
        unsigned n = r0 + wave;
        for (; n + (unsigned)(R - 1) * PLOW_WAVES < r1; n += PLOW_WAVES * (unsigned)R)
            gemv_rows_r<MM, XLDS, UN, R>(C, x, W, M, N, K, n, lane, lds);
        for (; n < r1; n += PLOW_WAVES)
            gemv_rows_r<MM, XLDS, GV_UNROLL, 1>(C, x, W, M, N, K, n, lane, lds);
    };
    if (gv_dyn_on(gv_cur, nblk, XLDS, (size_t)M * K)) {
        gv_dyn_walk(gv_cur, lds, gv_per, gv_n0, gv_n1, nblk, N, PLOW_WAVES * (unsigned)R, run);
        return;
    }
    run(gv_n0, gv_n1);
#else
    unsigned n = gv_n0 + wave;
    for (; n + (unsigned)(R - 1) * PLOW_WAVES < gv_n1; n += PLOW_WAVES * (unsigned)R)
        gemv_rows_r<MM, XLDS, UN, R>(C, x, W, M, N, K, n, lane, lds);
    for (; n < gv_n1; n += PLOW_WAVES)
        gemv_rows_r<MM, XLDS, GV_UNROLL, 1>(C, x, W, M, N, K, n, lane, lds);
#endif
}

/* ---- OPT-IN (PLOW_GEMV_LG=1): NARROW-K LANE-GROUP bf16 GEMV ----------- [BF16-GEMV-NARROWK-LG]
 *
 * THE SHAPE THIS FIXES, and it is the bf16 twin of `moe_down_lg_fp8_blk` (op_moe.h) one kernel
 * over. `gemv_rows`/`gemv_rows_r` give ONE OUTPUT ROW to a whole 64-lane wave and hand lane L the
 * 8 halves at k = 8*L, so a wave covers 512 halves per chunk. GLM-5.2 at TP8 runs the SHARED-EXPERT
 * DOWN projection at N=6144, K=256 (`layers.*.mlp.shared_experts.down_proj`, b=48, 75 packets a
 * token), and 256 < 512: only lanes 0..31 are in range. Half the wave loads bytes the buffer
 * descriptor returns as zero and then spends the full 16 VALU of widen+FMA multiplying them,
 * `wave_sum`'s leading butterfly step adds an exact +0.0, and `nchunk = ceil(256/512) = 1` so the
 * R-split's `UN` unroll has nothing to unroll over: of the R*UN = 14 `buffer_load_dwordx4` the
 * shipped clause issues per two-column round, TWELVE are entirely past `num_records` and fetch
 * nothing. Measured on the box (`PLOW_TRACE_RAW`, ctx 1024, one decode step): 1506 CU-us of busy
 * over 48 workgroups for 3.0 MB of weights -- 86 GB/s, against the 1.24 TB/s the SAME body reaches
 * on o_proj (N=6144 K=2048) in the same layer. Its 35.0 us span is the LONGEST of any Gemv packet
 * in a GLM MoE layer despite moving an eighth of o_proj's bytes.
 *
 * THE MAP. Split the wave into RG row-groups of LPG = 64/RG lanes; group `g` owns its own output
 * row and lane `sub = lane%LPG` owns k = 8*sub. At RG=2 that is LPG=32 lanes x 8 halves = 256 K =
 * exactly GLM's contraction, so every lane is live and one `global_load_dwordx4` retires TWO WHOLE
 * CONSECUTIVE ROWS as a fully-coalesced 1024-byte fetch. Then UNR row-BATCHES are issued back to
 * back before any is consumed, so RG*UNR rows are in flight at once -- the discipline the aiter
 * diff identifies as the real difference, and the one thing `nchunk = 1` denies the shipped body.
 *
 * BIT-IDENTICAL, by the argument `moe_down_lg_fp8_blk` makes. The (lane -> k) fragment set for a
 * row is unchanged (lane `sub` holds exactly what lane `sub` held), each row's partials are summed
 * by the SAME xor-butterfly, and the ONE leading offset the 64-lane `wave_sum` applies (32) adds a
 * lane whose `dot8` started at +0.0f and accumulated only products of a zero weight -- i.e. an
 * exact +0.0, and `v + 0.0` is bitwise `v` for every `v` this reduction can produce (a `dot8` chain
 * seeded at +0.0f is never -0.0). Which wave owns which row moves, and that is unobservable: `C[n]`
 * is written by exactly one lane either way. The workgroup's column OWNERSHIP is unchanged
 * ([gv_n0, gv_n1)), so PLOW_FINE's gemv->headnorm dependency map is unaffected.
 *
 * GUARDED to K <= LPG*8 (the whole contraction must fit one fragment per lane), to an 8-multiple K
 * and to M == 1; anything else falls through to the shipped R-split/single-column walk, which is
 * exact at any width. o_proj (K=2048) and the router gate (K=6144) therefore keep the shipped body
 * -- correctly, because at K >= 512 every lane is already live and neither has this defect.
 *
 * Weight rows are read as plain 16-byte global vectors rather than through a `buf_rsrc`, because
 * the row index is now LANE-VARYING and a descriptor built from a divergent base compiles to the
 * readfirstlane waterfall this file documents at length. The `live` guard is what the descriptor's
 * bounds check was doing. */
#ifndef PLOW_GEMV_LG
#define PLOW_GEMV_LG 0
#endif
#ifndef PLOW_GEMV_LG_RG
#define PLOW_GEMV_LG_RG 2
#endif
/* UNR=8 = RG*UNR = 16 rows in flight. 16 is not arbitrary at the GLM shape: gv_per = 6144/48 = 128
 * rows a workgroup over PLOW_WAVES = 8 waves is exactly 16 rows a wave, so every wave runs ONE
 * balanced pass and there is no ragged tail. Register-wise it is CHEAPER than what it replaces:
 * 8 x bf16v8 = 32 VGPRs of weight against the R-split's R*UN = 14 vectors = 56. */
#ifndef PLOW_GEMV_LG_UNR
#define PLOW_GEMV_LG_UNR 8
#endif
template <unsigned RG, unsigned UNR>
__device__ __forceinline__ void gemv_lg_batch(const bf16* __restrict__ lds,
                                              const bf16* __restrict__ W_, unsigned n0, unsigned N,
                                              unsigned K, unsigned lane, float (&o)[UNR]) {
    constexpr unsigned LPG = PLOW_WAVE / RG; /* lanes per row-group */
    const unsigned g = lane / LPG, sub = lane % LPG;
    const unsigned k = sub * 8u;
    const bool live = (k + 8u <= K);
    const unsigned kx = live ? k : 0u;
    const auto* const W = as_glob(W_);
    bf16v8 wv[UNR];
    /* RG*UNR consecutive rows, every load issued before any is consumed */
#pragma unroll
    for (unsigned u = 0; u < UNR; u++) {
        const unsigned n = n0 + u * RG + g;
        const unsigned nc = (n < N) ? n : (N - 1u); /* clamp: the caller drops the value */
        wv[u] = GV_LDW(W + (size_t)nc * K + kx);
    }
    const bf16v8 xv = ld_lds8(lds + kx); /* read once, used UNR times */
#pragma unroll
    for (unsigned u = 0; u < UNR; u++) {
        float a = live ? dot8(wv[u], xv, 0.0f) : 0.0f;
#pragma unroll
        for (unsigned off = LPG / 2; off > 0; off >>= 1) a += __shfl_xor(a, (int)off, PLOW_WAVE);
        o[u] = a;
    }
}

template <unsigned RG, unsigned UNR>
__device__ __forceinline__ void gemv_rows_lg(bf16* __restrict__ C_, const bf16* __restrict__ W,
                                             unsigned N, unsigned K, unsigned slice, unsigned nblk,
                                             const bf16* lds) {
    constexpr unsigned LPG = PLOW_WAVE / RG;
    constexpr unsigned PER = RG * UNR;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned sub = lane % LPG, grp = lane / LPG;
    auto* const C = as_glob(C_);
    /* IDENTICAL column ownership to `gemv_rows`/`gemv_rows_rs` -- only which wave takes which row
     * inside [gv_n0, gv_n1) moves. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    for (unsigned n0 = gv_n0 + wave * PER; n0 < gv_n1; n0 += PLOW_WAVES * PER) {
        float o[UNR];
        gemv_lg_batch<RG, UNR>(lds, W, n0, N, K, lane, o);
        if (sub == 0u) {
#pragma unroll
            for (unsigned u = 0; u < UNR; u++) {
                const unsigned n = n0 + u * RG + grp;
                if (n < gv_n1) st_act1(&C[n], f2bf(o[u]));
            }
        }
    }
}

/* ------------- MFMA GEMV: THE MATRIX CORE INSTEAD OF THE EMULATED DOT ----------------------
 *
 * WHY. gfx950 has v_dot2c_f32_bf16 and multiplies a bf16 PAIR in one instruction; gfx942 has
 * neither that nor a packed bf16 dot, so plow_dot2_bf16 falls back to widen+FMA and the decode
 * GEMV pays 24 VALU ops per 16 bytes of weight. MEASURED cost of that emulation, by collapsing
 * it to one op (GV_HACK_CHEAPDOT, wrong answer): -3.2% of the token. The matrix core is idle
 * during all of it.
 *
 * THE MAPPING, and it is the whole trick: do NOT put 32 OUTPUT ROWS in the M dimension. That is
 * the obvious GEMV-as-GEMM shape and it is wrong here -- it needs 32 rows per wave when these
 * shapes supply 1.6 (N=3840 over 2432 resident waves), and it reads 32 rows at K*2 stride
 * instead of one row contiguously. Instead put 32 K-CHUNKS OF ONE ROW in M:
 *
 *     A[m][j] = W[n][ s*512 + 16*m + j ]      m = lane%32, j = 8*(lane/32) + 0..7
 *     B[j][n] = x[   s*512 + 16*n + j ]       n = lane%32  -- the SAME index as A
 *     D[m][n] = sum_j A[m][j] * B[j][n]  =>  D[m][m] = sum_j W[n][16m+j] * x[16m+j]
 *
 * so the DIAGONAL of the accumulator holds 32 partial dot products, and chaining MFMAs across
 * `s` accumulates them. Each step consumes 1024 CONTIGUOUS bytes of one row per wave -- byte for
 * byte the access pattern the shipped kernel already has, so nothing about the memory side
 * changes. Only the arithmetic moves: 2 mfma_32x32x8 (the CDNA3 halves of the K=16 wrapper)
 * replace 24 VALU ops per lane.
 *
 * Matrix utilisation is 1/32, which does not matter: one mfma_f32_32x32x8bf16_1k retires 512 B
 * of W in ~32 cycles, ~40 TB/s aggregate across 304 CUs, against an HBM budget of 5.3.
 *
 * THE DIAGONAL EXTRACT is an unrolled compare, not an indexed read. `acc[di]` with a runtime
 * `di` forces the f32x16 accumulator out of registers into scratch -- the trap this file
 * already documents for the fragment arrays. 16 v_cmp/cndmask per ROW is cheaper than that.
 *
 * MEASURED, AND IT LOSES BY 1.6%. Gemma-4-12B, ctx 4096, 48 steps, three interleaved pairs:
 *
 *     VALU  21.139 21.213 21.186        MFMA  21.492 21.521 21.554
 *
 * Ranges disjoint. It is CORRECT -- GEMM FAMILY CORRECT (0 failures) at worst rel 0.0037, the
 * same as the VALU path, and token-identical on the real asset -- and it is faster STANDALONE
 * (3152 GB/s vs ~2900 on the cache-resident decode-GEMV rung, which is exactly what a rung
 * that fits in Infinity Cache measures: the arithmetic). In the model it is slower.
 *
 * WHY: the accumulator chain. Every step is `acc = mfma(a, b, acc)` into ONE f32x16, and on
 * CDNA3 the K=16 wrapper is TWO chained mfma_32x32x8. At K=3840 a row is 7.5 steps = 15
 * dependent MFMAs with nothing between them to hide ~32 cycles of latency each. The VALU path's
 * 24 ops per 16 B are ugly but they are INDEPENDENT across the UN unroll, so they pipeline.
 * The fix would be two accumulators over two rows -- 32 accumulator VGPRs against a decode
 * object already at 253 of 256, which is the same register wall that killed hand-rolled
 * software pipelining (see THE DMA ENGINE below).
 *
 * The arithmetic this arm can win back was bounded before it was written: GV_HACK_CHEAPDOT
 * collapses the emulated dot to one op and buys 3.2%. That was the ceiling, and the dependent
 * MFMA chain gives back more than it. Kept as a knob, defaulted OFF. */
#ifndef GV_MFMA
#define GV_MFMA 0
#endif
template <bool XLDS>
__device__ __forceinline__ void gemv_rows_mfma(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                               const bf16* __restrict__ W_, unsigned N, unsigned K,
                                               unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned frow = lane % 32;        /* m of A, n of B and of the accumulator */
    const unsigned koct = 8u * (lane / 32); /* which k-octet of the 16-wide block   */
    constexpr unsigned KSTEP = 32 * MFMA_K; /* 512 k per MFMA = 1024 B per wave      */

    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    const unsigned nstep = (K + KSTEP - 1) / KSTEP;

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    auto xv8 = [&](unsigned kk) -> bf16v8 {
        if constexpr (XLDS)
            return ld_lds8(lds + kk);
        else
            return ld_glob8(x + kk);
    };

    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES) {
        const __amdgpu_buffer_rsrc_t wr = PLOW_GV_RSRC(W + (size_t)n * K, K);
        f32x16 acc = (f32x16)(0.0f);
        for (unsigned s = 0; s < nstep; s++) {
            const unsigned kb = s * KSTEP + 16u * frow + koct;
            /* W past the row returns zero from the descriptor and fetches nothing; clamp only
             * the LDS index, so a zero weight never meets a NaN left in the arena. */
            const unsigned kx = (kb < K) ? kb : 0u;
            const bf16x8 a = __builtin_bit_cast(bf16x8, buf_ld8(wr, kb * 2u));
            const bf16x8 b = __builtin_bit_cast(bf16x8, xv8(kx));
            acc = plow_mfma_bf16_32x32(a, b, acc);
        }
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (mfma_acc_m(lane, (unsigned)i) == frow) d = acc[i];
        const float t = wave_sum(d);
        if (lane == 0) st_act1(&C[n], f2bf(t));
    }
}

/* ------------- SPLIT-K: KS WAVES SHARE ONE OUTPUT ROW -----------------------------------
 *
 * THE DEFECT THIS FIXES, and it is the ROW axis running out, not the unroll.
 *
 * `gemv_rows` deals whole output rows to waves: `gv_per = ceil(N/nblk)` rows per workgroup,
 * round-robined over PLOW_WAVES. At the wide shapes that is fine -- Gemma-4-12B's gate/up is
 * N=15360, so gv_per=51 and a wave gets ~6.4 rows, ~51 KB of streaming. At the NARROW ones it
 * collapses:
 *
 *     N=3840, nblk=304  ->  gv_per = 13 rows/WG  ->  waves 0-4 get 2 rows, waves 5-7 get 1
 *     row K=4096 = 8 chunks = 8 KB  ->  16 KB per wave behind a 6-step cross-lane reduction
 *                                       and a descriptor rebuild PER ROW
 *
 * Two costs at once: the per-row constant stops amortising, and 13 rows over 8 waves is a 2x
 * ragged tail (efficiency 13/16 = 81%, against 51/56 = 91% at the wide shape). Splitting K
 * turns one row into KS tasks, which both lengthens each wave's sequential stream and makes
 * the tail finer.
 *
 * WHY K AND NOT N. plow already has an N-split (`GV_SLICE`, s x n_cu shares) and it measures
 * monotonically WORSE -- 2939 / 2748 / 2315 GB/s at s = 1 / 2 / 4 on the standalone bench --
 * which is what cutting an already-too-small row axis predicts. hipBLASLt/Tensile reaches
 * 1760-3494 GB/s on these same shapes by splitting K for skinny GEMMs; this is that.
 *
 * THE REDUCTION IS WITHIN THE WORKGROUP, via LDS, so it needs no counter and no global
 * traffic: waves `rs*KS .. rs*KS+KS-1` own row `rs`, each sums a CONTIGUOUS quarter/half of the
 * row's chunks (contiguous, not strided, so each wave still walks one sequential stream), and
 * ks==0 adds the KS partials and stores.
 *
 * THE ITERATION COUNT IS WORKGROUP-UNIFORM ON PURPOSE. `iters` is derived from
 * (N, nblk, slice) only -- all wave-invariant -- so every wave reaches exactly the same number
 * of __syncthreads(). A loop bounded by each wave's own row supply would let a wave with one
 * fewer row exit early and hang the workgroup on the next barrier. Dead slots walk the loop
 * with `live=false`, computing a discarded partial over row gv_n0 and skipping only the store.
 */
/* MEASURED AND REJECTED, DEFAULTED OFF. Kept as a knob and a body because the arm is correct
 * and the negative result is worth more than the deleted code.
 *
 * Gemma-4-12B, ctx 4096, 48 steps, three INTERLEAVED pairs, ms/token:
 *
 *     GV_KS=1 (off)   21.274  21.302  21.266      GV_KS=2 (on)   21.599  21.542  21.558
 *
 * Ranges disjoint, +1.3%. AND IT IS MONOTONIC IN KS, which is the part that settles the whole
 * idea. KS=8 puts the WHOLE WORKGROUP on one row -- 512 threads x 16 B = 8 KB CONTIGUOUS per
 * step, against the shipped decomposition's 8 waves on 8 separate rows at 1 KB each. That is
 * the larger-contiguous-access shape a skinny-GEMM library is assumed to want, and it is the
 * WORST of the three: 21.832 21.863 21.832 vs 20.909 21.039 20.990, +4.1%, ranges disjoint by
 * a wide margin. Contiguity is not what this kernel is short of; the barrier and the cross-wave
 * reduction cost more than the access shape buys. An earlier shape that reduced per ROW rather than banking the
 * partials -- 8 workgroup barriers per packet instead of 2 -- read +0.95%, i.e. cutting the
 * barrier count fourfold made it slightly WORSE. So the barriers were not the cost, and the
 * premise was wrong:
 *
 *   SPLIT-K DOES NOT LENGTHEN A WAVE'S SEQUENTIAL STREAM, IT SHORTENS IT. The wave moves the
 *   same total bytes either way; splitting K just walks K/KS per row over KS times as many
 *   row-iterations. What starves the narrow shapes is the PER-ROW CONSTANT (descriptor
 *   rebuild, 6-step cross-lane reduction), and split-K pays that constant MORE often, not
 *   less. It also does not fix the ragged tail it was aimed at: 13 rows over 4 slots is
 *   13/16 utilisation, exactly what 13 rows over 8 slots already was.
 *
 * Split-K is the right transform when a kernel is short of PARALLELISM -- too few output rows
 * to fill the machine. plow is not: N=3840 over 2432 resident waves is 1.58 rows per wave, so
 * every wave already has work. hipBLASLt reaches for split-K on skinny GEMMs because it
 * launches a grid sized to the shape; the persistent interpreter is already machine-filling.
 *
 * What remains unexplained is the 1.7x against hipBLASLt on the same shapes. It is NOT
 * bandwidth-per-wave, NOT loads-in-flight (see GV_UNROLL / GV_RS_MAXNCH, both null), and NOT
 * row parallelism. The per-row constant is the remaining candidate and it wants a body that
 * amortises it across rows -- gemv_rows_r's R-column split is that shape, and it is measured
 * null at nchunk >= 8. Next probe: the descriptor rebuild (buf_rsrc per row) and wave_sum. */
#ifndef GV_KS
#define GV_KS 1 /* K-splits per row. 1 disables the arm (it is then never instantiated). */
#endif
/* Rows-per-wave ceiling for taking the arm. Kept at 3 as the value the rejected experiment
 * ran with. NOTE for anyone re-opening this: the standalone decode-GEMV rungs in
 * test_kernels.hip are too noisy to set this threshold -- the same shape read 2708-3214 GB/s
 * across runs, and a MAXRPW=5 build measured FASTER than a MAXRPW=6 one on every shape, which
 * is impossible since 6 is a superset of 5. Set it from an interleaved end-to-end A/B only. */
#ifndef GV_KS_MAXRPW
#define GV_KS_MAXRPW 3
#endif
#ifndef GV_KS_UN
#define GV_KS_UN 8
#endif
/* One float per (row-iteration, wave), in halves, padded to keep the arena 16-B aligned.
 * `iters = ceil(gv_per/NW)` and the arm is only taken at `gv_per <= GV_KS_MAXRPW*PLOW_WAVES`,
 * with `NW = PLOW_WAVES/KS`, so `iters <= GV_KS_MAXRPW*KS`. Banking every iteration is what
 * lets the whole packet reduce behind ONE barrier instead of two per row. */
#define GV_KS_ITERS_MAX ((unsigned)(GV_KS_MAXRPW) * (unsigned)(GV_KS))
#define GV_KS_SCRATCH ((2u * PLOW_WAVES * GV_KS_ITERS_MAX + 15u) & ~15u)

template <bool XLDS, int UN, int KS>
__device__ __forceinline__ void gemv_rows_ksplit(bf16* __restrict__ C_,
                                                 const bf16* __restrict__ x_,
                                                 const bf16* __restrict__ W_, unsigned N,
                                                 unsigned K, unsigned slice, unsigned nblk,
                                                 const bf16* lds, float* part) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    constexpr unsigned NW = PLOW_WAVES / KS; /* row slots per workgroup */
    const unsigned rs = wave / (unsigned)KS; /* which row slot this wave serves */
    const unsigned ks = wave % (unsigned)KS; /* which K-range within that row   */

    const unsigned step = PLOW_WAVE * 8;
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned per = (nchunk + (unsigned)KS - 1) / (unsigned)KS;
    const unsigned c0 = ks * per;
    const unsigned c1 = (c0 + per < nchunk) ? (c0 + per) : nchunk;

    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    const unsigned rows = (gv_n1 > gv_n0) ? (gv_n1 - gv_n0) : 0u;
    const unsigned iters = (rows + NW - 1) / NW; /* wave-invariant: see the note above */

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    auto xv8 = [&](unsigned kk) -> bf16v8 {
        if constexpr (XLDS)
            return ld_lds8(lds + kk);
        else
            return ld_glob8(x + kk);
    };

    for (unsigned it = 0; it < iters; it++) {
        const unsigned n = gv_n0 + rs + it * NW;
        const bool live = (n < gv_n1);
        const __amdgpu_buffer_rsrc_t wr = PLOW_GV_RSRC(W + (size_t)(live ? n : gv_n0) * K, K);
        float acc = 0.0f;
        for (unsigned c = c0; c < c1; c += UN) {
            bf16v8 wv[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned ci = c + (unsigned)u;
                /* Past THIS split's range the byte offset is pushed beyond `num_records`, so
                 * the buffer load returns zero and FETCHES NOTHING. Without it the tail of one
                 * split would read -- and add -- chunks that belong to the next split. The
                 * same hardware bounds check is what lets the row loop have no scalar tail. */
                wv[u] = buf_ld8(wr, (ci < c1) ? ((ci * step + lane * 8) * 2u) : (K * 2u));
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 8;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN */
                acc = dot8(wv[u], xv8(kx), acc);
            }
        }
        const float t = wave_sum(acc);
        /* PARTIALS ARE BANKED, NOT REDUCED HERE. A reduce-per-row costs TWO workgroup barriers
         * per row-iteration; at N=3840 that is 4 iterations x 2 = 8 barriers per packet and
         * ~800 per token, and MEASURED that way split-K was a consistent +0.95% REGRESSION
         * (21.464-21.513 vs 21.257-21.324 ms/token, ranges disjoint over 3 interleaved pairs).
         * Banking every iteration's partial and reducing once costs TWO barriers per PACKET.
         * `iters` is bounded by GV_KS_MAXRPW*KS, which is what sizes the scratch. */
        if (lane == 0) part[it * PLOW_WAVES + wave] = t;
    }
    __syncthreads();
    if (ks == 0 && lane == 0) {
        for (unsigned it = 0; it < iters; it++) {
            const unsigned n = gv_n0 + rs + it * NW;
            if (n >= gv_n1) continue;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < KS; j++) s += part[it * PLOW_WAVES + rs * (unsigned)KS + (unsigned)j];
            st_act1(&C[n], f2bf(s));
        }
    }
    /* The next packet stages its activation over the low end of this arena and barriers after
     * doing so -- but a LARGER M*K would reach into `part` while these reads are still in
     * flight. One barrier on the way out, rather than a rule about who may follow a GEMV. */
    __syncthreads();
}


/* GEMV WITH A FUSED GLU EPILOGUE -- gate and up in ONE pass, act(g)*u applied at the write.
 *
 * This is the fusion every BLAS ships (cuBLASLt epilogue, CK, hipBLASLt), and the DIRECTION is
 * the whole thing. There are two ways to "fuse the GLU" and only one of them can ever pay:
 *
 *   EPILOGUE, into the PRODUCER (this).  The GEMV that computes gate[n] and up[n] applies
 *       act(gate)*up as it writes its own output. Output-stationary: the workgroup that owns
 *       column n is the ONLY one that touches it, so the GLU runs EXACTLY ONCE per element.
 *
 *   PROLOGUE, into the CONSUMER (tried; LOST 39x).  Fold the GLU into the down-GEMV's LDS
 *       staging instead. But `fu` is down's K dimension -- the axis it REDUCES over -- so all
 *       256 of its workgroups stage the whole of it, and every one recomputes the entire GLU.
 *       One consuming OP; 256 consuming WORKGROUPS. It cost 174,688 CU-ticks/layer to save
 *       4,472. This is also exactly why norm mode 2 lost (line 517): same mistake, same shape.
 *
 *   Fuse into the producer's EPILOGUE, never into the consumer's PROLOGUE. A producer feeding
 *   a consumer's K dimension is replicated by that consumer's workgroup count.
 *
 * Merging gate and up into ONE GEMV is what MAKES the epilogue possible: they are otherwise two
 * separate GEMVs on DISJOINT CU sets (split2, 128/128), so no single workgroup ever holds both
 * gate[n] and up[n], and there is nothing to fuse. Here one workgroup owns column n of both.
 *
 * Two weight streams in flight, so the unroll is halved: 2*6 = 12 vectors against 11 for the
 * single-stream GEMV -- same memory-level parallelism, same register budget.
 *
 * UN=6 leaves ONE dead load per stream (K=5376 -> 11 chunks -> two groups of 6). UN=11 makes it
 * exact and was measured DEAD EVEN (15.5 both ways) while costing 21 more registers, 156 -> 177.
 * Same law as the per-shape-unroll experiment: a dead buffer_load fetches nothing, and this GEMV
 * is bandwidth-bound, so instruction slots are not the binding resource. Keep the cheap one. */
#ifndef GV_UNROLL_GLU
/* MEASURED LIMIT, gfx942 8 waves: the FUSED gate|up GEMV holds TWO weight streams per row, so
 * its register cost is double the plain GEMV's and the unroll cannot buy it back. Spill for
 * gemv_glu_m{1,4,8,16} against UN:
 *
 *   UN=6   246/0   256/200   256/536   256/1284
 *   UN=4   246/0   256/200   256/517   256/1167
 *   UN=3   246/0   256/200   256/524   256/1153
 *   UN=2   246/0   256/200   256/522   256/1151
 *
 * i.e. flat past UN=4 -- unlike the plain GEMV, where the same sweep was worth 9x. M=1 (the
 * default decode bucket) is clean at 0 spill; M>=4 is not, and the fix is not an unroll but
 * emitting gate|up as TWO GEMVs plus a Glu, which is what the block-fp8 path already does
 * (see the WFP8BLK/GLU static_assert above). Left as-is: that is an emitter choice, not a
 * kernel knob, and batch-1 decode does not reach it. */
#define GV_UNROLL_GLU 6
#endif

template <int MM, int UN = GV_UNROLL_GLU>
__device__ __forceinline__ void gemv_glu_rows(bf16* __restrict__ C_, const bf16* __restrict__ Wg_,
                                              const bf16* __restrict__ Wu_, unsigned M, unsigned N,
                                              unsigned K, unsigned act, unsigned slice,
                                              unsigned nblk, const bf16* lds, float beta = 0.0f,
                                              float lbeta = 0.0f) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 8;
    const unsigned nchunk = (K + step - 1) / step;

    auto* const C = as_glob(C_);
    const auto* const Wg = as_glob(Wg_);
    const auto* const Wu = as_glob(Wu_);
    /* x is ALWAYS staged in LDS here: plowc emits this op only when M*K fits GM_LDS_HALVES. */
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 { return ld_lds8(lds + (size_t)m * K + kk); };

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + wave; n < n1; n += PLOW_WAVES) {
        /* One descriptor per weight row, per stream: num_records = K halves, so the overshoot
         * past the row end returns zero AND issues no memory request. No tail, no predication. */
        /* TWO descriptors per row here, so d_gemv_glu pays the waterfall twice. */
        const __amdgpu_buffer_rsrc_t rg = PLOW_GV_RSRC(Wg + (size_t)n * K, K);
        const __amdgpu_buffer_rsrc_t ru = PLOW_GV_RSRC(Wu + (size_t)n * K, K);
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wg[UN], wu[UN];
            /* Both streams issued before either is touched: 12 loads in flight per lane. */
#pragma unroll
            for (int u = 0; u < UN; u++)
                wg[u] = buf_ld8(rg, ((c + (unsigned)u) * step + lane * 8) * 2u);
#pragma unroll
            for (int u = 0; u < UN; u++)
                wu[u] = buf_ld8(ru, ((c + (unsigned)u) * step + lane * 8) * 2u);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 8;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; w is 0 anyway */
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xv = xv8(live ? m : 0, kx);
                    ag[m] += live ? dot8(wg[u], xv, 0.0f) : 0.0f;
                    au[m] += live ? dot8(wu[u], xv, 0.0f) : 0.0f;
                }
            }
        }
        /* THE EPILOGUE. Both halves are in registers, in the one workgroup that owns column n.
         *
         * `act == 2` (SITU) takes the PAIR form and cannot go through `act_gate_only`, which
         * returns NaN for it BY DESIGN — situ transforms the UP branch too, so a gate-only helper
         * has no correct answer and poisons instead of quietly computing the wrong model. This is
         * the same split `moe_glu` makes (op_moe.h:187); it is open-coded here because op_gemm.h is
         * included BEFORE op_moe.h in interp.hip, and `k3_situ_gate`/`k3_situ_up` live in
         * amd_common.h precisely so both translation units can reach them.
         *
         * Additive by construction: every act this arm did not already serve reached here and
         * produced NaN, so nothing can have depended on the old behaviour. */
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float g = wave_sum(ag[m]);
            const float u = wave_sum(au[m]);
            if (lane == 0 && (unsigned)m < M) {
                const float o = (act == 2u) ? (k3_situ_gate(g, beta) * k3_situ_up(u, lbeta))
                                            : (act_gate_only(g, act) * u);
                st_act1(&C[(size_t)m * N + n], f2bf(o));
            }
        }
    }
}

/* ===========================================================================
 * FP8 (w8a16) DECODE GEMV — half the weight bytes, ~2x the bandwidth-bound roofline.
 *
 * Structurally identical to gemv_rows above; the ONLY differences are the weight load and its
 * dequant (see amd_common.h fp8_to_bf16v8): the weight row is uint8[K], a b128 load pulls 16 fp8
 * per lane (vs 8 bf16), and each 16-fp8 group is converted to two bf16v8 that feed the SAME fdot2
 * `dot8`. The per-output-channel dequant `w_scale[n]` is a per-row scalar, uniform across the
 * wave, and is applied ONCE in the epilogue on wave_sum(acc) — never per element.
 *
 * GV_BLOCKED column ownership is copied verbatim from gemv_rows: the decode fine-grained
 * gemv->headnorm dependency map (plowc gemv_wgs_for_cols) assumes it, and it must stay in step. */
/* One fp8 chunk is PLOW_WAVE*16 = 1024 K-elements (2x the bf16 chunk), so nchunk = ceil(K/1024) is
 * SMALL — and unlike the bf16 GEMV, a dead overshoot load here still costs a full CONVERT (16 cvt/
 * shift on zeros), which is pure wasted VALU on a near-compute-bound kernel. So UN must DIVIDE the
 * row, not just cover it: plowc's decode K's are 2560 (q/k/v/gate/up -> 3 chunks), 4096 (o -> 4),
 * 9728 (down -> 10). d_gemv_fp8 picks UN per K; the template keeps it a compile-time constant. */
#ifndef GV_UNROLL_FP8
#define GV_UNROLL_FP8 3
#endif
/* ABLATION LADDER (probe only, 0 = shipped code, results are WRONG at every other level). The
 * decode GEMV packet costs ~33 us before any bytes move, against a 5-10 us floor on the norm
 * packets in the same trace. This strips one phase per level so the fixed term can be attributed
 * instead of guessed: 1 = no wave_sum, 2 = no dequant/dot (loads still issued), 3 = no loads
 * either. Each level is measured in perf-data/plow-gfx942/gemv-mlp-and-tensile.md. */
#ifndef PLOW_GV_ABL
#define PLOW_GV_ABL 0
#endif
/* PICK UN BY THE CHUNK COUNT, NOT BY K. The rule the comment above states -- UN must DIVIDE the
 * row -- was implemented by hardcoding the K's of ONE model family (2560/4096/9728), so any model
 * whose hidden size is not on that list silently runs dead chunks. Gemma-4 is exactly that case:
 * K=3840 and K=4096 both give nchunk=4, and UN=3 issues c=0,3 => 6 chunk-steps for 4 real ones.
 * The dead loads cost no MEMORY (the buffer bound returns zero) but pay a full convert + two dot8
 * each, and MEASURED standalone at Gemma-4's shapes on this MI300X (fp8gemv probe, same loop, same
 * grid) that is worth 28-37%:
 *
 *     K=3840 nchunk=4  q_proj 152wg   UN=3 10.65us   UN=4  7.94us   -25%
 *     K=3840 nchunk=4  k/v     76wg   UN=3  9.96us   UN=4  7.25us   -27%
 *     K=4096 nchunk=4  o_proj 304wg   UN=3  8.79us   UN=4  7.24us   -18%
 *     K=15360 nchunk=15 down  304wg   UN=3 19.82us   UN=4 20.71us   UN=3 already right
 *
 * Selecting the divisor reproduces every choice the old hardcoded list made -- 2560 -> nchunk 3 ->
 * 3, 9728 -> nchunk 10 -> 5, 15360 -> nchunk 15 -> 3 -- and additionally fixes 3840/4096/8192. It
 * is arch-neutral: this is a dead-work bug, not a CDNA3 one, so gfx950 gets the same fix (its
 * Llama K=4096 moves 3 -> 4, which is the "one free overshoot" that comment already flags). */
__device__ __forceinline__ unsigned gv_un_fp8(unsigned nchunk) {
#if PLOW_CDNA4
    /* gfx950: constant, so the switches below fold and the object keeps its single pre-fix
     * UN instantiation per op. The K-divisor rule was measured on gfx942 only, and the GF=8
     * lesson in this file records object growth alone regressing gfx950 decode 32% — CDNA3
     * only until someone with a gfx950 measures the rule there. */
    (void)nchunk;
    return GV_UNROLL_FP8;
#endif
#ifdef PLOW_GV_UN_BIG
    /* PROBE: prefer the LARGEST divisor, not the first one found. The rule below tries 4 before
     * 5, so nchunk=15 (down_proj, K=15360) takes UN=3 -- five sequential loop iterations with
     * 2*3=6 loads in flight. UN=5 also divides 15 and gives three iterations with 10 in flight.
     * `down` measures 1519 GB/s against gate|up's 1999 on the SAME kernel, and its distinguishing
     * feature is exactly this: 1.6 columns per wave vs 6.3, so it has the least other work to
     * hide a load behind and the most to gain from a deeper unroll. */
    if (nchunk % 5 == 0) return 5;
    if (nchunk % 4 == 0) return 4;
    if (nchunk % 3 == 0) return 3;
    if (nchunk % 2 == 0) return 2;
    return GV_UNROLL_FP8;
#endif
#ifdef PLOW_GV_UN_LEGACY /* A/B escape hatch: the pre-fix constant, so the counterfactual object
                          * can be built from THIS tree rather than differenced against an older
                          * one. Reproduces the old path exactly for any K != 9728. */
    (void)nchunk;
    return GV_UNROLL_FP8;
#endif
    if (nchunk % 4 == 0) return 4;
    if (nchunk % 3 == 0) return 3;
    if (nchunk % 5 == 0) return 5;
    if (nchunk % 2 == 0) return 2;
    return GV_UNROLL_FP8;
}
/* ONE OUTPUT COLUMN, for the ODD TAIL of gemv_rows_fp8's pair loop — the per-channel-fp8 sibling of
 * `gemv_mx_rows_1`, deleting the same duplicate-and-discard defect for the same reason. See that
 * function for the mechanism, for why the PAIR path deliberately keeps its hand-written body rather
 * than becoming an R=2 instantiation, and for the measurement.
 *
 * THREE STREAMS, ONE BODY — the fp8 answer to `gemv_mx_rows_1`, for the same register-allocation
 * reason: `PLOW_DOP_GEMV_FP8` reaches this with Nk = Nv = 0 and the k/v pointers null, so the
 * stream-select is dead on that path and the 1-stream numerics are BIT-IDENTICAL to before. */
template <int MM, bool XLDS, int UN>
__device__ __forceinline__ void gemv_fp8_rows_1(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                                bf16* __restrict__ Cv_, const bf16* __restrict__ x_,
                                                const unsigned char* __restrict__ Wq_,
                                                const unsigned char* __restrict__ Wk_,
                                                const unsigned char* __restrict__ Wv_,
                                                const float* __restrict__ Sq_,
                                                const float* __restrict__ Sk_,
                                                const float* __restrict__ Sv_, unsigned M,
                                                unsigned Nq, unsigned Nk, unsigned Nv, unsigned K,
                                                unsigned n, unsigned lane, const bf16* lds) {
    const unsigned step = PLOW_WAVE * 16; /* one pass: 64 lanes x 16 fp8 */
    const unsigned nchunk = (K + step - 1) / step;
    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    const auto* const x = as_glob(x_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Sq = as_glob(Sq_);
    const auto* const Sk = as_glob(Sk_);
    const auto* const Sv = as_glob(Sv_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };
    /* Which projection this concatenated column belongs to. Wave-uniform, therefore scalar. */
    auto W = Wq;
    auto Sc = Sq;
    auto Co = Cq;
    unsigned col = n, Nout = Nq;
    if (n >= Nq + Nk) { W = Wv; Sc = Sv; Co = Cv; col = n - Nq - Nk; Nout = Nv; }
    else if (n >= Nq) { W = Wk; Sc = Sk; Co = Ck; col = n - Nq;      Nout = Nk; }
    const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(W + (size_t)col * K, K);
    float acc[MM];
#pragma unroll
    for (int m = 0; m < MM; m++) acc[m] = 0.0f;

    for (unsigned c = 0; c < nchunk; c += UN) {
        fp8v16 wv[UN];
#pragma unroll
        for (int u = 0; u < UN; u++) wv[u] = buf_ld_fp8(wr, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * step + lane * 16;
            const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
            bf16v8 wlo, whi;
            fp8_to_bf16v8(wv[u], wlo, whi);
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                const bf16v8 xlo = xv8(live ? m : 0, kx);
                const bf16v8 xhi = xv8(live ? m : 0, kx + 8);
                acc[m] += live ? dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) : 0.0f;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; m++) {
        const float t = wave_sum(acc[m]) * Sc[col]; /* per-row dequant, once, in the epilogue */
        if (lane == 0 && (unsigned)m < M) st_act1(&Co[(size_t)m * Nout + col], f2bf(t));
    }
}
/* THREE STREAMS, ONE BODY — `PLOW_DOP_GEMV_QKV_FP8` (op 115) is this at Nk/Nv != 0, and
 * `PLOW_DOP_GEMV_FP8` (op 30) is this with Nk = Nv = 0 and the k/v pointers null, on the register
 * argument `gemv_rows_mxfp4` (op 91/114) makes: interp.hip inlines every arm into one kernel whose
 * VGPR budget is the WORST CASE over all of them, so a second inlined fp8 sweep is what must not
 * exist. With Nk = Nv = 0 the stream test is false for every column in range, the k/v arms are dead
 * and the 1-stream numerics are BIT-IDENTICAL to before. The extra pointers are wave-uniform ->
 * SGPRs, and `auto` keeps each selected pointer in address_space(1) (see gemv_rows' XLDS note). */
template <int MM, bool XLDS, int UN = GV_UNROLL_FP8>
__device__ __forceinline__ void gemv_rows_fp8(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                              bf16* __restrict__ Cv_, const bf16* __restrict__ x_,
                                              const unsigned char* __restrict__ Wq_,
                                              const unsigned char* __restrict__ Wk_,
                                              const unsigned char* __restrict__ Wv_,
                                              const float* __restrict__ Sq_,
                                              const float* __restrict__ Sk_,
                                              const float* __restrict__ Sv_, unsigned M,
                                              unsigned Nq, unsigned Nk, unsigned Nv, unsigned K,
                                              unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 16; /* one pass: 64 lanes x 16 fp8 */
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned N = Nq + Nk + Nv; /* the CONCATENATED column count */

    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    const auto* const x = as_glob(x_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Sq = as_glob(Sq_);
    const auto* const Sk = as_glob(Sk_);
    const auto* const Sv = as_glob(Sv_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* TWO OUTPUT COLUMNS PER WAVE-STEP. The fp8 chunk is 1024 K-elements (b128 = 16 fp8/lane), so
     * nchunk = ceil(K/1024) is HALF the bf16 GEMV's — and for K=2560 (q/k/v/gate/up) that is only 3,
     * a hard ceiling of 3 loads in flight per column no matter how large UN is. That starves the
     * memory-level parallelism a persistent megakernel (no occupancy to hide latency) lives on. So
     * each wave carries TWO independent weight rows, doubling the outstanding loads to 2*min(UN,
     * nchunk) — the same reason the 2-stream fp8 GLU fares better than this single stream. The
     * workgroup's column OWNERSHIP is unchanged (still [slice*per, slice*per+per)), so the decode
     * fine-grained gemv->headnorm map is unaffected.
     *
     * THE ODD TAIL IS ITS OWN COLUMN COUNT, not a masked second stream. `n2 = has2 ? n + PLOW_WAVES
     * : n` used to make a wave with an ODD column count recompute its last column TWICE and discard
     * one copy; the loop below runs FULL PAIRS ONLY and hands a leftover column to
     * gemv_fp8_rows_1. Same defect and same fix as gemv_rows_fp8_blk and gemv_rows_mxfp4 — see
     * gemv_mx_rows_1 for the mechanism, the measurement, and why the pair path is NOT folded into
     * an R-parameterised body. NOT SEPARATELY MEASURED on this arm: it serves Qwen/Llama w8a16
     * shapes that no A/B on this branch has covered, and it is landed on the mxfp4 numbers plus the
     * structural argument that a discarded column-pass cannot be worth having. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
#if PLOW_GV_ABL >= 4
    /* ABLATION 4: return IMMEDIATELY. Level 3 still walks the column/chunk loop (address math,
     * predicates, wave_sum, stores) with fake data, so it does NOT measure "an infinitely fast
     * GEMV" -- it measures "a GEMV with free memory". This level retires the packet body
     * entirely, so what remains is the interpreter's per-packet machinery plus the ops the fp8
     * GEMV does not own. Wrong results by construction. */
    return;
#endif
    unsigned n = gv_n0 + wave;
    for (; n + PLOW_WAVES < gv_n1; n += PLOW_WAVES * 2) {
        const unsigned n2 = n + PLOW_WAVES;
        /* Which projection each concatenated column belongs to. Wave-uniform, therefore scalar;
         * the pair (n, n2) may straddle a stream boundary and each carries its own selection. */
        auto W = Wq, W2 = Wq;
        auto Sc = Sq, Sc2 = Sq;
        auto Co = Cq, Co2 = Cq;
        unsigned col = n, col2 = n2, Nout = Nq, Nout2 = Nq;
        if (n >= Nq + Nk) { W = Wv; Sc = Sv; Co = Cv; col = n - Nq - Nk; Nout = Nv; }
        else if (n >= Nq) { W = Wk; Sc = Sk; Co = Ck; col = n - Nq;      Nout = Nk; }
        if (n2 >= Nq + Nk) { W2 = Wv; Sc2 = Sv; Co2 = Cv; col2 = n2 - Nq - Nk; Nout2 = Nv; }
        else if (n2 >= Nq) { W2 = Wk; Sc2 = Sk; Co2 = Ck; col2 = n2 - Nq;      Nout2 = Nk; }
        /* num_records = K BYTES (fp8 is 1 byte): the overshoot past the row end returns zero and
         * issues no memory request — same free bounds check the bf16 GEMV relies on. */
        const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(W + (size_t)col * K, K);
        const __amdgpu_buffer_rsrc_t wr2 = buf_rsrc_fp8(W2 + (size_t)col2 * K, K);
        float acc[MM], acc2[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { acc[m] = 0.0f; acc2[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp8v16 wv[UN], wv2[UN];
#if PLOW_GV_ABL >= 3
            /* ABLATION 3: delete the weight LOADS as well, leaving only the packet's control flow.
             * Whatever time survives this is dispatch, gating and loop overhead. */
#pragma unroll
            for (int u = 0; u < UN; u++) { wv[u] = (fp8v16)(c + (unsigned)u); wv2[u] = wv[u]; }
#else
            /* Both columns' loads issued before either is touched: 2*UN in flight per lane. */
#pragma unroll
            for (int u = 0; u < UN; u++) wv[u] = buf_ld_fp8(wr, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) wv2[u] = buf_ld_fp8(wr2, (c + (unsigned)u) * step + lane * 16);
#endif
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 16;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
#if PLOW_GV_ABL >= 2
                /* ABLATION 2: keep the LOADS live, delete the dequant and the dots. Prices the
                 * whole VALU body against pure weight streaming. Wrong results by construction. */
                (void)kx;
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    acc[m] += __builtin_bit_cast(float, wv[u][0]);
                    acc2[m] += __builtin_bit_cast(float, wv2[u][0]);
                }
#else
                bf16v8 wlo, whi, wlo2, whi2;
                fp8_to_bf16v8(wv[u], wlo, whi);
                fp8_to_bf16v8(wv2[u], wlo2, whi2);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xlo = xv8(live ? m : 0, kx);
                    const bf16v8 xhi = xv8(live ? m : 0, kx + 8);
                    acc[m] += live ? dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) : 0.0f;
                    acc2[m] += live ? dot8(whi2, xhi, dot8(wlo2, xlo, 0.0f)) : 0.0f;
                }
#endif
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            /* ABLATION 1: skip the 6-step cross-lane reduction, keep everything else. Prices the
             * wave_sum chain, which runs once per column PAIR. Wrong results by construction. */
#if PLOW_GV_ABL == 1
            const float t = acc[m] * Sc[col];
            const float t2 = acc2[m] * Sc2[col2];
#else
            const float t = wave_sum(acc[m]) * Sc[col]; /* per-row dequant, once, in the epilogue */
            const float t2 = wave_sum(acc2[m]) * Sc2[col2];
#endif
            if (lane == 0 && (unsigned)m < M) {
                st_act1(&Co[(size_t)m * Nout + col], f2bf(t));
                st_act1(&Co2[(size_t)m * Nout2 + col2], f2bf(t2));
            }
        }
    }
    if (n < gv_n1)
        gemv_fp8_rows_1<MM, XLDS, UN>(Cq_, Ck_, Cv_, x_, Wq_, Wk_, Wv_, Sq_, Sk_, Sv_, M, Nq, Nk,
                                      Nv, K, n, lane, lds);
}

/* MXFP4 DECODE GEMV (w4a16) — OCP microscaling e2m1 weights, one E8M0 scale per 32 K-elements,
 * bf16 activation. Structurally the fp8 GEMV above with a 4-bit weight stream.
 *
 * WHY THIS IS THE FP4 SHAPE THAT PAYS. Decode is weight-bandwidth-bound, not MFMA-bound — the
 * whole reason the fp8 GEMV exists. fp4 halves the weight stream again on top of fp8 (0.5 vs 1
 * byte/elt, plus 1 scale byte per 32 = 4.25 bits/elt effective), so the roofline moves ~1.88x over
 * fp8 and ~3.76x over bf16. The MFMA fp4 GEMM (cbsz/blgp=4) is the PREFILL shape and is a separate
 * kernel; it buys nothing here because a GEMV never fills an MFMA.
 *
 * THE SCALE IS FREE (see fp4_to_bf16v8x4 in amd_common.h): an MX scale is E8M0, i.e. a power of two
 * by construction, so it folds exactly into the cvt's scalef32 operand. Contrast the block-fp8 twin
 * below, whose arbitrary-f32 weight_scale_inv forces a separate per-block multiply. mxfp4 dequant is
 * therefore the SAME VALU cost as fp8 dequant, and the epilogue has no dequant at all.
 *
 * LAYOUT (matches the OCP MX spec and what quantisers emit):
 *   W[n][k]  packed 2 fp4/byte, row stride K/2 bytes, low nibble = even k
 *   S[n][j]  one E8M0 byte per 32-K block, row stride K/32 bytes, j = k/32
 * One lane's b128 load is 16 bytes = 32 fp4 = exactly one block, so a lane reads exactly one scale
 * byte per chunk and no scale ever straddles a fragment.
 *
 * Bounds: the weight buffer resource returns 0 past num_records (free OOB check, as the fp8 path),
 * and fp4 zero decodes to 0.0, so an overshoot chunk contributes nothing. The scale byte is guarded
 * explicitly — it is 1/16 the traffic of the weights and stays in L1/L2, so a predicated scalar
 * load is not worth a buffer resource of its own. */
#ifndef PLOW_MXFP4_DEC_NT
#define PLOW_MXFP4_DEC_NT 1
#endif
__device__ __forceinline__ fp8v16 buf_ld_mxfp4(__amdgpu_buffer_rsrc_t r,
                                               unsigned byte_off) {
#if PLOW_MXFP4_DEC_NT
    return buf_ld_fp8(r, byte_off);
#else
    return __builtin_bit_cast(
        fp8v16, __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, 1));
#endif
}
/* ONE BODY, THREE STREAMS -- and the 1-stream form is this with Nk = Nv = 0.
 *
 * `PLOW_DOP_GEMV_MXFP4` (op 91) reaches this function with Nk = Nv = 0 and the k/v weight, scale
 * and output pointers null, so the fused q|k|v form costs the interpreter NO second inlined body.
 * That is a register-allocation fact and not tidiness, for the reason `gemv_qkvg_rows` states for
 * the bf16 family: interp.hip inlines every arm into one kernel whose VGPR budget is the WORST
 * CASE over all of them, and the decode object sits within two registers of the occ-2 cliff. With
 * Nk = Nv = 0 the `n < Nq` test is true for every column in range, so the k and v arms are dead
 * code on that path and the 1-stream numerics are BIT-IDENTICAL to before.
 *
 * The extra pointers are wave-uniform, so they are SGPRs: `n` is the loop induction variable of a
 * wave-strided loop, the stream test is on `n` alone, and `auto` keeps every selected pointer in
 * address_space(1) (a `const unsigned char*` local would merge them into a GENERIC pointer and
 * silently demote the fp4 weight stream to flat loads -- see gemv_rows' XLDS note).
 *
 * BYTE-EXACTNESS is the bar and it is structural: a column's value depends only on its own weight
 * row, its own scale row and the shared `x`, accumulated over the same `nchunk` in the same order
 * whatever `Ntot` is. Concatenating the three sweeps changes only WHICH workgroup owns a column,
 * never what that column computes. Verified elementwise on gfx950 --
 * runtime/tests/gemv_qkv_mxfp4_gfx950_test. */
/* ONE OUTPUT COLUMN, for the ODD TAIL of the pair loop below — the fp4 answer to what
 * `gemv_blk_rows_r<...,1>` is for the block-fp8 twin, and it exists for the same defect.
 *
 * The pair loop used to run `n2 = has2 ? n + PLOW_WAVES : n` and guard only the STORE, so when the
 * pair ran off the end of the workgroup's column range the wave recomputed column `n` a SECOND time
 * — same 16-byte fp4 load, same 4 `fp4_to_bf16v8x4` (16 `cvt_scalef32_pk_bf16_fp4`), same 4 dot8,
 * same wave_sum — and threw the result away. It bites when a wave's column count is ODD, and what
 * it costs is set by the SLOWEST wave in the workgroup: with `qmax = ceil(gv_per/PLOW_WAVES)`
 * columns on the critical wave the old loop makes `2*ceil(qmax/2)` column-passes to produce
 * `qmax`, so odd `qmax` wastes `1/(qmax+1)` of the sweep and even `qmax` wastes nothing at all.
 *
 * WHY THE PAIR PATH IS NOT ALSO R-PARAMETERISED, unlike `gemv_blk_rows_r`. It was, first: one
 * `gemv_mx_rows_r<MM,XLDS,UN,R>` with the caller picking R=2 or R=1, mirroring the fp8_blk twin
 * exactly. Instruction counts came out within 13 of the hand-written dual body (2156 against 2143)
 * and VGPRs one LOWER — and it still measured 2.5-3% slower on the shapes with an EVEN column
 * count, where it executes exactly the same work. Attributed rather than assumed: an A/A control
 * (the same body compiled into two objects) reads 1.000x on this instrument, and an arm that drives
 * the R-parameterised inner function from the OLD always-R=2 loop — so no R=1 instantiation exists
 * to change the code footprint — still loses, at 0.965x. So it is the R=2 code generation itself,
 * and the scheduler's answer for the hand-written pair is simply better. Keeping the measured-best
 * code on the hot path and adding only the tail leaves the even shapes BYTE-IDENTICAL to before,
 * which is the whole reason the fp8_blk A/B could not be transplanted here on faith.
 * Bench: perf-data/k3_gemvmx_bench.{hip,cpp}, which carries both bodies and checks them against
 * each other elementwise on device. */
template <int MM, bool XLDS, int UN>
__device__ __forceinline__ void gemv_mx_rows_1(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                               bf16* __restrict__ Cv_,
                                               const bf16* __restrict__ x_,
                                               const unsigned char* __restrict__ Wq_,
                                               const unsigned char* __restrict__ Wk_,
                                               const unsigned char* __restrict__ Wv_,
                                               const unsigned char* __restrict__ Sq_,
                                               const unsigned char* __restrict__ Sk_,
                                               const unsigned char* __restrict__ Sv_, unsigned M,
                                               unsigned Nq, unsigned Nk, unsigned Nv, unsigned K,
                                               unsigned wbytes, unsigned nsb, unsigned n,
                                               unsigned lane, const bf16* lds) {
    const unsigned step = PLOW_WAVE * 32; /* one pass: 64 lanes x 32 fp4 = 2048 K */
    const unsigned nchunk = (K + step - 1) / step;
    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    const auto* const x = as_glob(x_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Sq = as_glob(Sq_);
    const auto* const Sk = as_glob(Sk_);
    const auto* const Sv = as_glob(Sv_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* Which projection this concatenated column belongs to. Wave-uniform, therefore scalar. */
    auto Wr = Wq;
    auto Sc = Sq;
    auto Co = Cq;
    unsigned col = n, Nout = Nq;
    if (n >= Nq + Nk) { Wr = Wv; Sc = Sv; Co = Cv; col = n - Nq - Nk; Nout = Nv; }
    else if (n >= Nq) { Wr = Wk; Sc = Sk; Co = Ck; col = n - Nq;      Nout = Nk; }
    const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(Wr + (size_t)col * wbytes, wbytes);
    const auto* const sr = Sc + (size_t)col * nsb;
    float acc[MM];
#pragma unroll
    for (int m = 0; m < MM; m++) acc[m] = 0.0f;

    for (unsigned c = 0; c < nchunk; c += UN) {
        fp4v32 wv[UN];
#pragma unroll
        for (int u = 0; u < UN; u++)
            wv[u] = __builtin_bit_cast(
                fp4v32, buf_ld_mxfp4(wr, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * step + lane * 32;
            const unsigned kx = (k < K) ? k : 0u;
            const unsigned sb = k >> 5; /* this lane's block index; one per fragment */
            const float sc = (sb < nsb) ? e8m0_to_f32(sr[sb]) : 0.0f;
            const fp4_frag32 wf = fp4_prep32(wv[u], sc);
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                const bf16v8 x0 = xv8(live ? m : 0, kx);
                const bf16v8 x1 = xv8(live ? m : 0, kx + 8);
                const bf16v8 x2 = xv8(live ? m : 0, kx + 16);
                const bf16v8 x3 = xv8(live ? m : 0, kx + 24);
                acc[m] += live ? fp4_dot32(wf, x0, x1, x2, x3, 0.0f) : 0.0f;
            }
        }
    }
    /* No dequant here — unlike the fp8 twin's `* wscale[n]`, the MX scale is already folded
     * into every converted element by fp4_to_bf16v8x4. */
#pragma unroll
    for (int m = 0; m < MM; m++) {
        const float t = wave_sum(acc[m]);
        if (lane == 0 && (unsigned)m < M) st_act1(&Co[(size_t)m * Nout + col], f2bf(t));
    }
}
template <int MM, bool XLDS, int UN = 3>
__device__ __forceinline__ void gemv_rows_mxfp4(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                                bf16* __restrict__ Cv_,
                                                const bf16* __restrict__ x_,
                                                const unsigned char* __restrict__ Wq_,
                                                const unsigned char* __restrict__ Wk_,
                                                const unsigned char* __restrict__ Wv_,
                                                const unsigned char* __restrict__ Sq_,
                                                const unsigned char* __restrict__ Sk_,
                                                const unsigned char* __restrict__ Sv_, unsigned M,
                                                unsigned Nq, unsigned Nk, unsigned Nv, unsigned K,
                                                unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 32; /* one pass: 64 lanes x 32 fp4 = 2048 K */
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned wbytes = K >> 1;   /* packed weight row stride */
    const unsigned nsb = (K + 31) >> 5; /* scale bytes per row     */
    const unsigned N = Nq + Nk + Nv;    /* the CONCATENATED column count */

    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    const auto* const x = as_glob(x_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Sq = as_glob(Sq_);
    const auto* const Sk = as_glob(Sk_);
    const auto* const Sv = as_glob(Sv_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* TWO OUTPUT COLUMNS PER WAVE-STEP, for the reason the fp8 twin documents: fp4 doubles the
     * K-elements per load again, so nchunk halves again and the in-flight-load ceiling per column
     * gets even tighter. Two independent rows keep 2*min(UN,nchunk) loads outstanding.
     *
     * The pair (n, n2) may straddle a stream boundary. That is fine and needs no special case:
     * each of the two carries its own weight row, scale row, output base and output width. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    /* FULL PAIRS ONLY, then at most one tail column. `n + PLOW_WAVES < gv_n1` rather than
     * `n < gv_n1`: the second column of the pair is now unconditionally live, so the clamp that
     * made an odd tail recompute column `n` twice is gone along with the flag that drove it. At
     * most ONE column can be left over, because the step is 2*PLOW_WAVES. See gemv_mx_rows_1. */
    unsigned n = gv_n0 + wave;
    for (; n + PLOW_WAVES < gv_n1; n += PLOW_WAVES * 2) {
        const unsigned n2 = n + PLOW_WAVES;
        /* Which projection each concatenated column belongs to. Wave-uniform, therefore scalar. */
        auto Wr = Wq, Wr2 = Wq;
        auto Sc = Sq, Sc2 = Sq;
        auto Co = Cq, Co2 = Cq;
        unsigned col = n, col2 = n2, Nout = Nq, Nout2 = Nq;
        if (n >= Nq + Nk) { Wr = Wv; Sc = Sv; Co = Cv; col = n - Nq - Nk; Nout = Nv; }
        else if (n >= Nq) { Wr = Wk; Sc = Sk; Co = Ck; col = n - Nq;      Nout = Nk; }
        if (n2 >= Nq + Nk) { Wr2 = Wv; Sc2 = Sv; Co2 = Cv; col2 = n2 - Nq - Nk; Nout2 = Nv; }
        else if (n2 >= Nq) { Wr2 = Wk; Sc2 = Sk; Co2 = Ck; col2 = n2 - Nq;      Nout2 = Nk; }
        const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(Wr + (size_t)col * wbytes, wbytes);
        const __amdgpu_buffer_rsrc_t wr2 = buf_rsrc_fp8(Wr2 + (size_t)col2 * wbytes, wbytes);
        const auto* const sr = Sc + (size_t)col * nsb;
        const auto* const sr2 = Sc2 + (size_t)col2 * nsb;
        float acc[MM], acc2[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { acc[m] = 0.0f; acc2[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp4v32 wv[UN], wv2[UN];
            /* Both columns' weight loads issued before either is consumed: 2*UN in flight. */
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_mxfp4(wr, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv2[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_mxfp4(wr2, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 32;
                const unsigned kx = (k < K) ? k : 0u;
                const unsigned sb = k >> 5; /* this lane's block index; one per fragment */
                const float sc = (sb < nsb) ? e8m0_to_f32(sr[sb]) : 0.0f;
                const float sc2 = (sb < nsb) ? e8m0_to_f32(sr2[sb]) : 0.0f;
                const fp4_frag32 wf = fp4_prep32(wv[u], sc);
                const fp4_frag32 wf2 = fp4_prep32(wv2[u], sc2);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 x0 = xv8(live ? m : 0, kx);
                    const bf16v8 x1 = xv8(live ? m : 0, kx + 8);
                    const bf16v8 x2 = xv8(live ? m : 0, kx + 16);
                    const bf16v8 x3 = xv8(live ? m : 0, kx + 24);
                    acc[m] += live ? fp4_dot32(wf, x0, x1, x2, x3, 0.0f) : 0.0f;
                    acc2[m] += live ? fp4_dot32(wf2, x0, x1, x2, x3, 0.0f) : 0.0f;
                }
            }
        }
        /* No dequant here — unlike the fp8 twin's `* wscale[n]`, the MX scale is already folded
         * into every converted element by fp4_to_bf16v8x4. */
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]);
            const float t2 = wave_sum(acc2[m]);
            if (lane == 0 && (unsigned)m < M) {
                st_act1(&Co[(size_t)m * Nout + col], f2bf(t));
                st_act1(&Co2[(size_t)m * Nout2 + col2], f2bf(t2));
            }
        }
    }
    if (n < gv_n1)
        gemv_mx_rows_1<MM, XLDS, UN>(Cq_, Ck_, Cv_, x_, Wq_, Wk_, Wv_, Sq_, Sk_, Sv_, M, Nq, Nk, Nv,
                                     K, wbytes, nsb, n, lane, lds);
}

/* BLOCK-SCALED FP8 GEMV — the DeepSeek/GLM block-fp8 quant scheme (weight_block_size [128,128]).
 * Unlike gemv_rows_fp8 (one per-CHANNEL scale wscale[n], factored out of the whole K-reduction in
 * the epilogue), the block scheme has a per-[128 out][128 K] scale grid `wscale[n/128][k/128]`
 * (row-major, KB = ceil(K/128) columns). Because the scale varies along K it CANNOT be pulled out
 * of the reduction — each 128-K block's partial must be dequantised by its own block scale before
 * summing. The trick that keeps this cheap: a lane owns 16 CONSECUTIVE fp8 (k..k+15), and 128 is a
 * multiple of 16, so a lane's 16-element partial lies ENTIRELY within one 128-K block. Scaling is
 * linear, so multiplying each lane's per-chunk partial by ITS block scale before wave_sum yields the
 * exact block-scaled reduction — one extra FMA per chunk, no cross-lane reshuffle. (The block scale
 * CANNOT be folded into the fp8->bf16 cvt: v_cvt_scalef32_pk_bf16_fp8's scalef32 operand is E8M0 /
 * exponent-only — it uses only 2^floor(log2(scale)) and discards the mantissa, correct for MX
 * microscaling but NOT for GLM/DeepSeek's arbitrary-f32 weight_scale_inv. Probed on gfx950 2026-07-17:
 * scale 0.01 folds as 2^-7=0.0078, a ~22% per-block error. So the multiply stays separate/f32.)
 * w8a16: x is bf16
 * (dynamic-activation fp8 quantises x separately; the decode weight-stream path keeps x bf16). */
/* R OUTPUT ROWS PER WAVE-STEP, all of them LIVE. R is a compile-time count, never a runtime mask:
 * a mask needs an exec manipulation inside the chunk loop, which breaks the load batching that is
 * the whole point. Measured on gfx950 at N=6144 K=3072: this form 6633 GB/s, the same R rows with
 * a runtime `live` count 4085 GB/s. The caller instantiates R=2 for the body, R=1 for the odd tail.
 * Bench: perf-data/glm52_gemvblk_bench.{hip,cpp}.
 *
 * as_glob is re-applied here even though the caller already did it: these parameters are plain
 * pointers, and a generic pointer reaching the weight load is what once turned the whole stream
 * into flat_load_ushort (see gemv_rows' XLDS note). Idempotent cast, no code. */
template <int MM, bool XLDS, int UN, int R>
__device__ __forceinline__ void gemv_blk_rows_r(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                                const unsigned char* __restrict__ W_,
                                                const float* __restrict__ wscale_, unsigned M,
                                                unsigned N, unsigned K, unsigned KB, unsigned n,
                                                unsigned lane, const bf16* lds) {
    const unsigned step = PLOW_WAVE * 16;
    const unsigned nchunk = (K + step - 1) / step;
    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    const auto* const wscale = as_glob(wscale_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };
    __amdgpu_buffer_rsrc_t wr[R];
    unsigned nrow[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const unsigned nr = n + (unsigned)r * PLOW_WAVES;
        wr[r] = buf_rsrc_fp8_u(W + (size_t)nr * K, K);
        nrow[r] = (nr >> 7) * KB; /* base of this channel's block-scale row */
    }
    float acc[R][MM];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MM; m++) acc[r][m] = 0.0f;

    for (unsigned c = 0; c < nchunk; c += UN) {
        fp8v16 wv[R][UN];
        /* every row's loads issued before any is touched: R*UN outstanding per lane */
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[r][u] = buf_ld_fp8(wr[r], (c + (unsigned)u) * step + lane * 16);
#pragma unroll
        for (int u = 0; u < UN; u++) {
            const unsigned k = (c + (unsigned)u) * step + lane * 16;
            const unsigned kx = (k < K) ? k : 0u;
            unsigned kb = k >> 7; /* this lane's 128-K block index for chunk (c+u) */
            if (kb >= KB) kb = KB - 1; /* overshoot lanes: partial is 0, keep the read in-bounds */
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                const bf16v8 xlo = xv8(live ? m : 0, kx);
                const bf16v8 xhi = xv8(live ? m : 0, kx + 8);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    bf16v8 wlo, whi;
                    fp8_to_bf16v8(wv[r][u], wlo, whi);
                    const float p = dot8(whi, xhi, dot8(wlo, xlo, 0.0f));
                    /* dequant THIS block before summing */
                    acc[r][m] += live ? p * wscale[nrow[r] + kb] : 0.0f;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; m++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wave_sum(acc[r][m]); /* block scales already folded in per chunk */
            if (lane == 0 && (unsigned)m < M)
                st_act1(&C[(size_t)m * N + n + (unsigned)r * PLOW_WAVES], f2bf(t));
        }
}
template <int MM, bool XLDS, int UN = GV_UNROLL_FP8>
__device__ __forceinline__ void gemv_rows_fp8_blk(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                                  const unsigned char* __restrict__ W_,
                                                  const float* __restrict__ wscale_, unsigned M,
                                                  unsigned N, unsigned K, unsigned slice,
                                                  unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned KB = (K + 127u) >> 7; /* number of 128-K blocks per output channel */

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);
    const auto* const wscale = as_glob(wscale_);

    /* THE ODD TAIL IS A SEPARATE ROW COUNT, not a masked second stream.
     *
     * This loop used to run `n2 = has2 ? n + PLOW_WAVES : n` and guard only the STORE. When the
     * pair ran off the end of the workgroup's column range the wave therefore recomputed column
     * `n` a SECOND time — same loads, same 16 converts per chunk, same dots — and threw the
     * result away. It is not free: the duplicate is an L2 hit rather than HBM traffic, but it
     * doubles the VMEM issue slots and the dequant VALU of that whole iteration.
     *
     * It bites exactly when the wave's column count is ODD, which is GLM's every-day case:
     * N = 6144 over nblk = 256 is gv_per = 24 = 3 columns/wave, so one of every two iterations
     * was half-dead — 4 column-passes to produce 3, i.e. 25% of the work discarded. Measured on
     * gfx950 at the GLM TP4 shapes (grid 256, blockDim 512, 6200 GB/s denominator):
     *
     *   N=6144 K=3072 (dense down, op 44 today)  UN=3   5088 -> 6633 GB/s  (82.1% -> 107.0%)
     *   N=6144 K=4096 (o_proj under GLM_LINEAR_FP8) UN=4 5852 -> 6114     (94.4% ->  98.6%)
     *   N=2624 K=6144 (fusion-A concat)          UN=3   4812 -> 6670      (77.6% -> 107.6%)
     *   N=32768 K=6144 (wide N, 16 cols/wave, NO tail) UN=3 6417 -> 6773  (103.5% -> 109.2%)
     *
     * The last row is the control: with an even column count there is no duplicate and the fix is
     * worth nothing, which is what it measures. `has2` is wave-uniform (n is derived from `wave`),
     * so the branch is scalar and never diverges, and it sits OUTSIDE the chunk loop.
     *
     * BIT-IDENTICAL by construction: same chunk order, same lane->k mapping, same accumulation
     * order, same wave_sum. Only the discarded second stream is gone. */
    const unsigned gv_per = (N + nblk - 1) / nblk;
    const unsigned gv_n0 = slice * gv_per;
    const unsigned gv_n1 = (gv_n0 + gv_per < N) ? (gv_n0 + gv_per) : N;
    for (unsigned n = gv_n0 + wave; n < gv_n1; n += PLOW_WAVES * 2) {
        if (n + PLOW_WAVES < gv_n1)
            gemv_blk_rows_r<MM, XLDS, UN, 2>(C, x, W, wscale, M, N, K, KB, n, lane, lds);
        else
            gemv_blk_rows_r<MM, XLDS, UN, 1>(C, x, W, wscale, M, N, K, KB, n, lane, lds);
    }
}

/* FP8 GEMV WITH A FUSED GLU EPILOGUE — the fp8 twin of gemv_glu_rows. Two fp8 weight streams
 * (gate, up), each with its own per-output-channel dequant scale applied in the epilogue. */
/* gate|up are always K=hidden: Qwen 2560 -> 3 chunks, Llama 4096 -> 4. UN=3 divides Qwen exactly
 * (no dead converts) and leaves one free overshoot on Llama; two fp8 streams, so 2*3 loads/group. */
#ifndef GV_UNROLL_GLU_FP8
#define GV_UNROLL_GLU_FP8 3
#endif
template <int MM, int UN = GV_UNROLL_GLU_FP8>
__device__ __forceinline__ void gemv_glu_rows_fp8(bf16* __restrict__ C_,
                                                  const unsigned char* __restrict__ Wg_,
                                                  const unsigned char* __restrict__ Wu_,
                                                  const float* __restrict__ gscale_,
                                                  const float* __restrict__ uscale_, unsigned M,
                                                  unsigned N, unsigned K, unsigned act,
                                                  unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 16;
    const unsigned nchunk = (K + step - 1) / step;

    auto* const C = as_glob(C_);
    const auto* const Wg = as_glob(Wg_);
    const auto* const Wu = as_glob(Wu_);
    const auto* const gscale = as_glob(gscale_);
    const auto* const uscale = as_glob(uscale_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 { return ld_lds8(lds + (size_t)m * K + kk); };

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + wave; n < n1; n += PLOW_WAVES) {
        const __amdgpu_buffer_rsrc_t rg = buf_rsrc_fp8(Wg + (size_t)n * K, K);
        const __amdgpu_buffer_rsrc_t ru = buf_rsrc_fp8(Wu + (size_t)n * K, K);
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp8v16 wg[UN], wu[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) wg[u] = buf_ld_fp8(rg, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) wu[u] = buf_ld_fp8(ru, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 16;
                const unsigned kx = (k < K) ? k : 0u;
                bf16v8 glo, ghi, ulo, uhi;
                fp8_to_bf16v8(wg[u], glo, ghi);
                fp8_to_bf16v8(wu[u], ulo, uhi);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xlo = xv8(live ? m : 0, kx);
                    const bf16v8 xhi = xv8(live ? m : 0, kx + 8);
                    ag[m] += live ? dot8(ghi, xhi, dot8(glo, xlo, 0.0f)) : 0.0f;
                    au[m] += live ? dot8(uhi, xhi, dot8(ulo, xlo, 0.0f)) : 0.0f;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float g = wave_sum(ag[m]) * gscale[n];
            const float u = wave_sum(au[m]) * uscale[n];
            if (lane == 0 && (unsigned)m < M) {
                const float s = act_gate_only(g, act);
                st_act1(&C[(size_t)m * N + n], f2bf(s * u));
            }
        }
    }
}

/* FUSED Q|K|V|G GEMV -- up to FOUR projections that share x and K, in ONE packet.
 *
 * Decode emits q_proj, k_proj, v_proj as three separate GEMVs on disjoint CU sets (split3). They
 * already overlap, but that is THREE packets = three counter gates, and three CU partitions whose
 * cross-op arrival imbalance is on the critical path (the 171/42/43 CU split for N=4096/1024/1024).
 * Concatenating their output columns into one N=Nq+Nk+Nv GEMV deletes two gates AND fills every CU
 * uniformly. Same math, same bytes: concatenated column n<Nq is q, n<Nq+Nk is k, else v -- each
 * column reads exactly one weight row of the appropriate matrix and writes its own output.
 *
 * THE FOURTH STREAM is Kimi-K3's KDA output gate `g_proj`, which reads the SAME pre-normed x as
 * q/k/v and writes a fourth disjoint [H*D] buffer. It is the same merge along the OUTPUT axis, one
 * stream wider: Ntot = Nq+Nk+Nv+Ng and the ownership run each workgroup takes is still a
 * contiguous slice of that concatenation, so all 256 CUs stay uniformly loaded. This is NOT the
 * `GLM_GROUP=1` collapse (6g-KNOBS, 38% fewer ops for +2.88 ms): that one
 * merged along a LOOP dimension and serialised work that had been running on disjoint CU slices.
 * Here the per-CU column count RISES from Ntot/256 to (Ntot+Ng)/256 -- the op gets WIDER, never
 * narrower, and nothing that ran in parallel starts running in sequence.
 *
 * ONE BODY, NOT TWO. `PLOW_DOP_GEMV_QKV` reaches this same function with Ng = 0 and Wg = Cg =
 * nullptr, so the 4-stream form costs the interpreter no second inlined instantiation. That is a
 * register-allocation fact, not tidiness: interp.hip inlines every arm into one kernel whose VGPR
 * budget is the WORST CASE over all of them, and the decode object already sits at 254 of 256.
 * With Ng = 0 the `n < Nq+Nk+Nv` test is true for every column in range, so the g arm is dead code
 * on that path and the 3-stream numerics are bit-identical to before. MEASURED, decode bucket at
 * PLOW_GEMV_MM=4 / PLOW_K3=1: 254 VGPR, occupancy 2 waves/SIMD, 0 VGPR spills -- IDENTICAL to the
 * 3-stream-only object. Scratch 812 -> 796 B/lane, SGPR spills 62 -> 63.
 *
 * MEASURED on the K3 decode shape (runtime/tests/gemv_qkvg_gfx950_test): the fused sweep is
 * BIT-IDENTICAL to four separate d_gemv calls on all 4 x 12288 outputs, and 0.111 ms against
 * their 0.151 ms -- 6362 vs 4663 GB/s of weight stream, before any counter gate is counted.
 *
 * Structurally a single-stream d_gemv_t (one weight row per column), so it keeps the deep unroll
 * and the plain-GEMV register budget -- NOT the doubled 12-vector budget of the GLU fusion. The
 * per-column branch is on `n`, which is wave-uniform, so it is scalar and never diverges; the two
 * extra pointers the fourth stream adds are SGPRs for the same reason. The matrix/output pointers
 * are all address-space(1) (as_glob), so selecting among them with `auto` keeps them global -- a
 * `const bf16*` local would merge them into a GENERIC pointer and silently demote the weight
 * stream to flat loads (see gemv_rows' XLDS note). */
template <int MM, int UN>
__device__ __forceinline__ void gemv_qkvg_rows(bf16* __restrict__ Cq_, bf16* __restrict__ Ck_,
                                               bf16* __restrict__ Cv_, bf16* __restrict__ Cg_,
                                               const bf16* __restrict__ Wq_,
                                               const bf16* __restrict__ Wk_,
                                               const bf16* __restrict__ Wv_,
                                               const bf16* __restrict__ Wg_, unsigned M,
                                               unsigned Nq, unsigned Nk, unsigned Nv, unsigned Ng,
                                               unsigned K, unsigned slice, unsigned nblk,
                                               const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 8;
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned Ntot = Nq + Nk + Nv + Ng;

    auto* const Cq = as_glob(Cq_);
    auto* const Ck = as_glob(Ck_);
    auto* const Cv = as_glob(Cv_);
    auto* const Cg = as_glob(Cg_);
    const auto* const Wq = as_glob(Wq_);
    const auto* const Wk = as_glob(Wk_);
    const auto* const Wv = as_glob(Wv_);
    const auto* const Wg = as_glob(Wg_);
    /* x is ALWAYS staged in LDS here: plowc emits this op only when M*K fits GM_LDS_HALVES. */
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 { return ld_lds8(lds + (size_t)m * K + kk); };

    /* GV_BLOCKED column ownership over the CONCATENATED N: workgroup `slice` owns the contiguous
     * run [slice*per, slice*per+per) of the q|k|v|g span (6144 wide for Llama's 3-stream form,
     * 49152 for K3's 4-stream one). Every CU gets ~the same count. */
    const unsigned per = (Ntot + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < Ntot) ? (n0 + per) : Ntot;

    for (unsigned n = n0 + wave; n < n1; n += PLOW_WAVES) {
        /* Which projection this concatenated column belongs to. `auto` preserves addrspace(1).
         * The last arm is unreachable when Ng == 0, which is how the 3-stream op shares this. */
        auto Wrow = Wq;
        auto Cout = Cq;
        unsigned col, Nout;
        if (n < Nq) {
            col = n;
            Nout = Nq;
        } else if (n < Nq + Nk) {
            Wrow = Wk;
            Cout = Ck;
            col = n - Nq;
            Nout = Nk;
        } else if (n < Nq + Nk + Nv) {
            Wrow = Wv;
            Cout = Cv;
            col = n - Nq - Nk;
            Nout = Nv;
        } else {
            Wrow = Wg;
            Cout = Cg;
            col = n - Nq - Nk - Nv;
            Nout = Ng;
        }
        const __amdgpu_buffer_rsrc_t wr = PLOW_GV_RSRC(Wrow + (size_t)col * K, K);
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) acc[m] = 0.0f;

        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
#pragma unroll
            for (int u = 0; u < UN; u++)
                wv[u] = buf_ld8(wr, ((c + (unsigned)u) * step + lane * 8) * 2u);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 8;
                const unsigned kx = (k < K) ? k : 0u; /* keep a NaN out of 0*NaN; wv is 0 anyway */
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 xv = xv8(live ? m : 0, kx);
                    acc[m] += live ? dot8(wv[u], xv, 0.0f) : 0.0f;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]);
            if (lane == 0 && (unsigned)m < M) st_act1(&Cout[(size_t)m * Nout + col], f2bf(t));
        }
    }
}

/* ---------------------------------------------------------------------------
 * THE DMA ENGINE (GV_DMA=1). Not a prefetcher — an ISSUE PATTERN.
 *
 * Three attempts to PREFETCH ahead of a gate all lost (sprint plan §11-§12), and the law that
 * came out of them is that there is no idle bandwidth to harvest at batch 1: the idleness IS
 * the critical path. This is the one idea those autopsies left standing, and it is a different
 * mechanism entirely — it moves the SAME bytes at the SAME time, and only changes how the loads
 * are issued.
 *
 * THE HOLE. Today a compute wave does:
 *     8x global_load_dwordx4  ->  s_waitcnt vmcnt(0)  ->  ~40 VALU of dot2c  ->  repeat
 * The memory pipe EMPTIES on every waitcnt. Little's law wants ~31.5 KB in flight per CU to
 * saturate 6.2 TB/s; averaged over the waitcnt holes we land at 5.50 TB/s — 89% of it, and the
 * missing 11% is 1.26 ms of a 16.8 ms token (measured: gemv busy 11.16 ms, floor 9.91 ms).
 *
 * WHY REGISTERS BLOCK THE OBVIOUS FIX. Deepening the pipeline with ordinary loads costs 4 VGPRs
 * per lane per 16 B in flight. GV_UNROLL=8 already holds 32. Decode sits at 232 of 256 registers
 * (and GV_UNROLL=10 was rejected at 252 — four from the cliff that has already killed an 8-wave
 * dispatch once). There is no room. Hand-rolled software pipelining was tried and was SLOWER
 * (§10) for exactly this reason.
 *
 * `global_load_lds` VOIDS THAT. It streams HBM straight into LDS with ZERO VGPRs, tracked on
 * vmcnt. So the ring can be GV_RING deep for free, and the wave never stops issuing.
 *
 * AND THAT IS WHY THERE IS NO DEDICATED "DMA WAVE". A wave can hold up to 63 outstanding vmem
 * ops on its own vmcnt, so the continuous-issue property comes from RING DEPTH, not from wave
 * specialisation. Dedicating one of the 8 waves to moving data would cost 1/8 of the dot2c
 * throughput and buy no extra memory-level parallelism. It would also re-open the objection this
 * file has always made (AMD partitions registers statically, so a producer wave burns its whole
 * allocation producing nothing) — which `global_load_lds` only voids for a wave that never
 * touches a VGPR, i.e. for a ring like this one, issued by the consumer itself.
 *
 * THE CONTRACT, and it is sharp: cp_async16's LDS destination comes from M0, which is UNIFORM.
 * The hardware lays the 64 lanes down CONTIGUOUSLY from it (lane l -> base + l*16 B). The global
 * source may be per-lane; the destination may NOT. So one call moves exactly GV_CHUNK = 512
 * halves = 1 KB of one weight row, and the ring slot must be that same 1 KB run.
 *
 * PIPELINED ACROSS ROWS, not within one. A wave owns several output columns; the chunk stream is
 * flattened to a single index `g` over (row, chunk) so the ring never drains at a row boundary.
 * Draining per row would stall on a just-issued load once per row and give most of the win back.
 * ------------------------------------------------------------------------- */
/* DEFAULT 0 — BUILT, MEASURED, AND IT LOSES. Kept because the reason is worth more than the code.
 *
 *     baseline                        16.8 ms/token
 *     DMA ring=16 batch=8             18.4
 *     DMA ring=16 batch=4             19.3
 *     DMA ring=4  batch=1             27.3
 *
 * THE PREMISE WAS FALSE, and three lines of ISA say so. Sprint-plan §12 justified this whole
 * idea with: "the compute waves issue 8x global_load_dwordx4 -> s_waitcnt vmcnt(0) -> ~40 VALU
 * of dot2c -> repeat. The memory pipe EMPTIES on every waitcnt." It does not. What the compiler
 * actually emits for the loop below is TWENTY back-to-back global_load_dwordx4 -- software-
 * pipelined ACROSS iterations, ~20 KB in flight per wave held in v[108:167] -- and then
 * staggered s_waitcnt vmcnt(7), vmcnt(6), vmcnt(5) ... consuming one chunk while the other
 * seven are still in flight. It NEVER drains.
 *
 * So there was no hole to fill. cp_async's one real advantage -- an in-flight chunk costs zero
 * VGPRs, where a register-held one costs 4/lane -- buys nothing when the register allocator has
 * already found 60 VGPRs to hold 20 KB in flight. All the DMA path adds is a second memory
 * instruction per kilobyte (global_load_lds, then ds_read_b128 to get it back) and an LDS
 * round-trip in the dependent chain.
 *
 * This closes the ENTIRE prefetch/DMA family: prefetch-on-arrival, mover-gate, mover+throttle
 * (§11-§12), and now the DMA engine. The gemv's remaining 11% (5.50 of 6.2 TB/s) is NOT an
 * issue-pattern problem, and the 1.26 ms §12 attributed to one is not there to be had.
 *
 * READ THE ISA BEFORE THEORISING ABOUT THE DESIGN. That lesson is written at §4c of the sprint
 * plan, and this is the second time it has cost a session. */
#ifndef GV_DMA
#define GV_DMA 0
#endif
#ifndef GV_RING
#define GV_RING 16 /* 1 KB chunks IN FLIGHT per wave — free, they hold no VGPRs */
#endif
#ifndef GV_BATCH
#define GV_BATCH 8 /* chunks materialised in registers at once: 4 VGPRs/lane each */
#endif
#define GV_CHUNK (PLOW_WAVE * 8) /* halves moved by ONE cp_async16: 64 lanes x 8 */

#if GV_DMA
/* LDS the ring needs, in halves. Sits after the staged activation in the GEMM arena. */
#define GV_RING_HALVES (PLOW_WAVES * GV_RING * GV_CHUNK)
_Static_assert(GV_RING >= 2 * GV_BATCH, "the ring must stay ahead of the batch");
_Static_assert(GV_RING_HALVES + 8192 <= GM_LDS_HALVES, "gemv DMA ring does not fit the arena");

template <int MM, bool XLDS>
__device__ __forceinline__ void gemv_rows_dma(bf16* __restrict__ C_, const bf16* __restrict__ x_,
                                              const bf16* __restrict__ W_, unsigned M, unsigned N,
                                              unsigned K, unsigned slice, unsigned nblk,
                                              const bf16* lds, bf16* ring) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;

    auto* const C = as_glob(C_);
    const auto* const x = as_glob(x_);
    const auto* const W = as_glob(W_);

    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 {
        const size_t xo = (size_t)m * K + kk;
        if constexpr (XLDS)
            return ld_lds8(lds + xo);
        else
            return ld_glob8(x + xo);
    };

    /* The output columns this wave owns — same decomposition as the scalar path. */
#if GV_BLOCKED
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;
    const unsigned n_step = PLOW_WAVES;
#else
    const unsigned n0 = 0, n1 = N, n_step = nblk * PLOW_WAVES;
    (void)n0;
#endif
    const unsigned n_first = n0 + wave;
    const unsigned rows = (n_first < n1) ? ((n1 - n_first + n_step - 1) / n_step) : 0u;

    const unsigned nchunk = K / GV_CHUNK; /* full 1 KB chunks of a weight row */
    const unsigned tail0 = nchunk * GV_CHUNK;
    if (rows == 0 || nchunk == 0) return;

    bf16* const myring = ring + wave * (GV_RING * GV_CHUNK);
    const unsigned G = rows * nchunk; /* the whole (row, chunk) stream for this wave */

    /* Address of chunk `g` of the stream, and the LDS slot it lands in. */
    auto src_of = [&](unsigned g) {
        const unsigned n = n_first + (g / nchunk) * n_step;
        return W + (size_t)n * K + (g % nchunk) * GV_CHUNK + lane * 8;
    };
    auto slot_of = [&](unsigned g) { return myring + (g % GV_RING) * GV_CHUNK; };

    /* Prologue: fill the ring. */
#pragma unroll
    for (unsigned r = 0; r < GV_RING; r++)
        if (r < G) cp_async16(src_of(r), slot_of(r));

    float acc[MM];
#pragma unroll
    for (int m = 0; m < MM; m++) acc[m] = 0.0f;

    /* One `g` = one 1 KB chunk. `steady` is the region where a chunk is still being issued for
     * every one consumed; after it the ring only drains, so ONE vmcnt(0) covers the rest.
     * (s_waitcnt takes a compile-time immediate, so the depth cannot be a variable — which is
     * exactly why the drain is split out instead of being folded into the loop.) */
    const unsigned steady = (G > GV_RING) ? (G - GV_RING) : 0u;

    auto consume = [&](unsigned g) {
        const bf16v8 wv = ld_lds8(slot_of(g) + lane * 8);
        const unsigned k = (g % nchunk) * GV_CHUNK + lane * 8;
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const bool live = ((unsigned)m < M);
            acc[m] += live ? dot8(wv, xv8(live ? m : 0, k), 0.0f) : 0.0f;
        }
    };

    /* Finish a row: its ragged K tail (K % 1024 B), then the wave reduction and the store. */
    auto retire = [&](unsigned g) {
        const unsigned n = n_first + (g / nchunk) * n_step;
        const auto* wrow = W + (size_t)n * K;
        for (unsigned k = tail0 + lane * 8; k < K; k += GV_CHUNK) {
            const bf16v8 wv = GV_LDW(wrow + k);
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                acc[m] += live ? dot8(wv, xv8(live ? m : 0, k), 0.0f) : 0.0f;
            }
        }
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float t = wave_sum(acc[m]);
            if (lane == 0 && (unsigned)m < M) st_act1(&C[(size_t)m * N + n], f2bf(t));
            acc[m] = 0.0f;
        }
    };

    /* Consume in BATCHES, and this is the whole design, not a tuning knob.
     *
     * Consuming one 1 KB chunk per s_waitcnt was the first thing tried and it was 62% SLOWER
     * (27.3 vs 16.8 ms/token): it puts an `s_waitcnt` and an `ds_read` latency in the dependent
     * chain of every single kilobyte, with only ~5 instructions of dot2c to hide them, and it
     * left only GV_RING=4 chunks in flight where the old register path already had 8.
     *
     * The point of cp_async is NOT that it is a different load — it is that a chunk IN FLIGHT
     * costs ZERO VGPRs, where a register-held one costs 4/lane. So the ring can be DEEPER than
     * the old pipeline for the same register budget: GV_RING in flight, but only GV_BATCH of
     * them ever materialised in registers at once. */
    unsigned g = 0;
    for (; g + GV_BATCH <= steady; g += GV_BATCH) {
        cp_async_depth<GV_RING - GV_BATCH>(); /* the GV_BATCH oldest chunks have landed */
        bf16v8 wv[GV_BATCH];
#pragma unroll
        for (int u = 0; u < GV_BATCH; u++) wv[u] = ld_lds8(slot_of(g + u) + lane * 8);
#pragma unroll
        for (int u = 0; u < GV_BATCH; u++) {
            const unsigned gu = g + (unsigned)u;
            const unsigned k = (gu % nchunk) * GV_CHUNK + lane * 8;
#pragma unroll
            for (int m = 0; m < MM; m++) {
                const bool live = ((unsigned)m < M);
                acc[m] += live ? dot8(wv[u], xv8(live ? m : 0, k), 0.0f) : 0.0f;
            }
            if (gu % nchunk == nchunk - 1) retire(gu);
        }
        /* Refill: in the steady region g <= G - GV_RING - GV_BATCH, so every index is < G. */
#pragma unroll
        for (int u = 0; u < GV_BATCH; u++)
            cp_async16(src_of(g + GV_RING + (unsigned)u), slot_of(g + GV_RING + (unsigned)u));
    }
    cp_async_wait(); /* past the steady region the ring only drains */
    for (; g < G; g++) {
        consume(g);
        if (g % nchunk == nchunk - 1) retire(g);
    }
}
#endif /* GV_DMA */

/* ---------------------------------------------------------------------------
 * SPLIT-K GEMV: BUILT, CORRECT, AND A LOSS. Do not rebuild it without reading this.
 *
 * The motivation was solid and is still true: the default GEMV gives each WAVE its own output
 * columns, so at N=5376 on 256 CUs that is 2.6 columns per wave, the wave grid quantizes (87.5%
 * fill) and the per-column wave_sum never amortizes. Measured per-CU bandwidth tracks
 * columns-per-wave exactly -- gate/up (21 cols/wave) 27.2 GB/s, o_proj (2.6) 20.7 -- and AMD's
 * own MI300 skinny-GEMM writeup reaches for split-K for exactly this shape.
 *
 * Built it: wave w takes a strided share of the K-chunks and works on EVERY column of a block,
 * partials folded across waves in LDS. Correct ("2087" intact). Measured, o_proj + down_proj:
 *
 *     default GEMV   4.28 ms          token 16.7 ms
 *     split-K        4.39 ms          token 16.9 - 17.0     <- LOSS, every run
 *
 * WHY, AND IT IS THE INTERESTING PART. A wave needs several loads in flight or it stalls, and
 * with K split 8 ways there are only nc/8 = 2 chunks at K=8192 -- two loads. So the columns must
 * be BATCHED (GV_CB of them) to get the parallelism back from the N axis. And that batching
 * REINTRODUCES THE QUANTIZATION IT WAS MEANT TO REMOVE: with per=21 columns and GV_CB=8 a
 * workgroup does ceil(21/8) = 3 blocks, i.e. 21/24 = 87.5% utilisation -- the SAME 87.5% the
 * wave grid had. The quantization moved from waves to column-blocks; it did not go away.
 *
 * The sweep proves the tension is real, and which side wins:
 *
 *     GV_CB = 8   16.8 ms      GV_CB = 4   17.4      GV_CB = 2   17.9
 *
 * Smaller batches quantize LESS and run WORSE -- memory-level parallelism dominates. And the only
 * batch size that actually fixes the fill (GV_CB = 1, so a work unit is one (column, k-split) and
 * 5376*8 = 43008 units divide 2048 waves exactly) is precisely the one with no parallelism left.
 *
 * THE LAW: fill wants FINE work units; memory-level parallelism wants COARSE ones. On a
 * bandwidth-bound op they fight, and MLP wins. A fill argument alone is not a reason to split.
 * ------------------------------------------------------------------------- */

/* THE UNROLL IS A REGISTER BUDGET, AND IT IS SHARED WITH THE M ROWS.
 *
 * GV_UNROLL is how many independent weight loads a wave puts in flight, and the comment above
 * picks 11 from `nchunk` for the M=1 decode. But each of the MM activation rows also holds live
 * vectors, so at MM>1 the same unroll asks for MM times the registers -- and on CDNA3 it asks
 * for more still, because there is no v_dot2c_f32_bf16 and the software dot has to materialise
 * f32 operands (amd_arch.h). Past the budget the allocator starts using AGPRs as spill relief,
 * which is exactly what happened: the M=8 and M=16 buckets landed at 256 VGPR + 256 AGPR and
 * every accumulate paid a v_accvgpr round trip.
 *
 * MEASURED on MI300X, 4-wave object, Gemma-4 31B shapes (ms, lower is better):
 *
 * SWEPT AT THE SHIPPING CONFIG -- 8 waves, single-buffered stage (ms, min of runs):
 *
 *            gate/up 21504x5376            down 5376x21504
 *   UN     M=1    M=4    M=8   M=16      M=1    M=4    M=8   M=16
 *   11    0.059  0.305  0.767  2.420    0.044  0.145  0.337  0.932
 *    6    0.068  0.098  0.206  1.804    0.044  0.047  0.092  0.706
 *    4    0.070  0.101  0.172  1.566    0.047  0.051  0.079  0.585
 *    3    0.070  0.102  0.160  0.265    0.046  0.053  0.075  0.119
 *    2    0.074  0.104  0.165  0.275    0.051  0.057  0.070  0.114
 *
 * The optimum FALLS with MM: 11 at M=1, 6 at M=4, 3 at M>=8. Taking it is 4.8x at M=8 and 9.1x
 * at M=16 over the flat 11 on gate/up. The earlier 4-wave sweep put M=16 at 2; at 8 waves the
 * allocator caps itself at 256 instead of reaching for AGPRs, and 3 wins -- which is why this
 * table was re-measured rather than carried over.
 *
 * CDNA4 keeps the flat 11 -- it has the packed dot, its buckets never reached the AGPRs, and
 * these numbers were not measured there. */
#if PLOW_CDNA4
#define GV_UNROLL_M4 GV_UNROLL
#define GV_UNROLL_M8 GV_UNROLL
#else
#ifndef GV_UNROLL_M4
#define GV_UNROLL_M4 6
#endif
#ifndef GV_UNROLL_M8
#define GV_UNROLL_M8 3
#endif
#endif
__device__ __host__ constexpr int gv_unroll_for(int mm) {
    return mm >= 8 ? GV_UNROLL_M8 : (mm >= 4 ? GV_UNROLL_M4 : GV_UNROLL);
}

template <int MM, int UN = gv_unroll_for(MM)>
__device__ void d_gemv_t(bf16* __restrict__ C, const bf16* __restrict__ x,
                         const bf16* __restrict__ W, const float* __restrict__ rms,
                         const bf16* __restrict__ gamma, unsigned M, unsigned N, unsigned K,
                         int norm, float eps, unsigned slice, unsigned nblk,
                         bf16* lds GV_DYN_PARAM_DEF) {
    /* norm == 2 folds the producing RMSNORM in. It is legal ONLY on the staged arm and only
     * for a row d_rmsnorm's `fits` path would have taken; the emitter enforces both, and this
     * demotes rather than trusting it, because the failure mode otherwise is reducing over an
     * LDS arena that was never filled — finite, plausible and wrong. */
    const bool fold = (norm == 2) && ((size_t)M * K + GV_NORM_SCRATCH <= GM_LDS_HALVES) &&
                      (K <= RN_REG * PLOW_THREADS) && ((K & 7u) == 0);
    if (norm == 2 && !fold) norm = 0;
    /* `part` MUST NOT be the caller's block_sum scratch here, and this is not defensive coding.
     * `plow_smem` (interp.hip) is a UNION: `sm->part` and `sm->gm` are THE SAME ADDRESS, so the
     * interpreter hands this function two aliases of one buffer. block_sum writes part[0..WAVES)
     * = the first 16 halves of the arena — i.e. straight through the staged activation this fold
     * is in the middle of reducing. Taking the scratch from the TOP of the arena instead, past
     * the row, removes the alias outright. A standalone test cannot see this bug: its `lds` and
     * `part` are separate __shared__ arrays, so it passes while the interpreter does not.
     *
     * The scratch is DERIVED from the arena rather than passed in, so no caller can hand this
     * function an aliasing buffer by accident — which is exactly how the bug got in. */
    float* const nscratch = (float*)(lds + (GM_LDS_HALVES - GV_NORM_SCRATCH));
    /* Stage the activation in LDS when it fits. x is re-read for every one of the N output
     * columns, and although it is tiny (11 KB at M=1) and therefore L2-resident, each
     * re-read still consumes a VECTOR-MEMORY issue slot -- the same pipe the weight stream
     * needs. At M=1 that is half of all load slots spent on data that is not the
     * bottleneck. Moving it to LDS leaves the vector path entirely to W. */
    /* Collect the prefetch the interpreter issued before the gate, THEN stage x behind the
     * ring. Both live in the one arena: [ ring | x ]. */
    if ((size_t)M * K <= GM_LDS_HALVES) {
        stage_x_lds(lds, x, (size_t)M * K);
        __syncthreads();
        /* Fold the RMSNORM into the staged copy, then fall through to the ORDINARY un-normed
         * hot loop — the k-loop never learns this happened, so it keeps its register
         * allocation and its unroll. `gemv_norm_lds` barriers on exit. */
        if (fold) gemv_norm_lds(lds, gamma, M, K, eps, nscratch);
        if (norm == 1)
            gemv_rows<MM, true, true, UN>(C, x, W, rms, gamma, M, N, K, slice, nblk, lds);
        else {
#if GV_DMA
            /* The arena is [ x | ring ]. x is M*K halves and 16-B aligned (K is a multiple of
             * 64 for every Gemma projection), so the ring behind it is too — which ds_read_b128
             * requires and which a 4-byte-aligned arena once silently cost this GEMM 2x. */
            gemv_rows_dma<MM, true>(C, x, W, M, N, K, slice, nblk, lds,
                                    lds + (((size_t)M * K + 7u) & ~7u));
#else
            /* SHORT ROWS TAKE THE R-SPLIT. `nchunk = ceil(K/512)` is the hard ceiling on the loads
             * a single-column wave can keep outstanding, so below UN it is the ROW and not the
             * unroll that starves the memory pipe — see gemv_rows_r. MM==1 only: the R-split
             * carries `acc[R][MM]`, and at a prefill bucket's MM=16 that doubles a 16-float
             * accumulator into the one kernel whose register union already sets the cliff. */
            /* NARROW-N TAKES SPLIT-K, and it is tried FIRST because it addresses a different
             * starvation from the R-split: the R-split fixes too few CHUNKS per row (short K),
             * this one fixes too few ROWS per wave (small N). A shape can be neither, either,
             * or -- as Gemma-4-12B's o_proj (N=3840, K=4096) is -- both.
             *
             * `M*K + GV_NORM_SCRATCH + GV_KS_SCRATCH <= GM_LDS_HALVES` because the partials sit
             * BELOW the norm arm's reduction scratch at the top of the same arena. Below, not
             * on top of it: when norm==2 the fold has already run and `nscratch` is dead by
             * here, but a future arm that reads it after this point would be silently corrupted,
             * and the two regions cost 40 halves together. */
#if GV_MFMA
            /* The matrix-core arm. MM==1 only: it accumulates into a 32x32 tile whose diagonal
             * is one output row, so a batched decode would need MM of them. */
            if constexpr (MM == 1) {
                gemv_rows_mfma<true>(C, x, W, N, K, slice, nblk, lds);
                return;
            }
#endif
#if GV_KS > 1
            if constexpr (MM == 1) {
                const unsigned gvp = (N + nblk - 1) / nblk;
                const unsigned rpw = (gvp + PLOW_WAVES - 1) / PLOW_WAVES;
                if (rpw <= (unsigned)GV_KS_MAXRPW &&
                    (size_t)M * K + GV_NORM_SCRATCH + GV_KS_SCRATCH <= GM_LDS_HALVES) {
                    gemv_rows_ksplit<true, GV_KS_UN, GV_KS>(
                        C, x, W, N, K, slice, nblk, lds,
                        (float*)(lds + (GM_LDS_HALVES - GV_NORM_SCRATCH - GV_KS_SCRATCH)));
                    return;
                }
            }
#endif
#if PLOW_GEMV_LG
            /* NARROW-K TAKES THE LANE-GROUP MAP, and it is tried BEFORE the R-split because the
             * R-split cannot help here: at K <= 256 `nchunk` is 1, so its `UN` unroll has nothing
             * to unroll over and its extra column only doubles a load clause that is 12/14 dead.
             * See gemv_rows_lg [BF16-GEMV-NARROWK-LG]. M == 1 exactly, because the arm carries no
             * per-m accumulator and the `live` predication the other bodies do is what would pay
             * for one. */
            if constexpr (MM == 1) {
                constexpr unsigned LG_RG = PLOW_GEMV_LG_RG, LG_UNR = PLOW_GEMV_LG_UNR;
                constexpr unsigned LG_LPG = PLOW_WAVE / LG_RG;
                if (M == 1u && K <= LG_LPG * 8u && (K & 7u) == 0u) {
                    gemv_rows_lg<LG_RG, LG_UNR>(C, W, N, K, slice, nblk, lds);
                    return;
                }
            }
#endif
            if constexpr (MM == 1 && GV_RS_R > 1) {
                if ((K + PLOW_WAVE * 8 - 1) / (PLOW_WAVE * 8) >= (unsigned)GV_RS_MAXNCH)
                    gemv_rows<MM, true, false, UN>(C, x, W, rms, gamma, M, N, K, slice, nblk,
                                                   lds GV_DYN_ARG);
                else
                    gemv_rows_rs<MM, true, GV_RS_UN, GV_RS_R>(C, x, W, M, N, K, slice, nblk,
                                                              lds GV_DYN_ARG);
            } else
                gemv_rows<MM, true, false, UN>(C, x, W, rms, gamma, M, N, K, slice, nblk,
                                               lds GV_DYN_ARG);
#endif
        }
    } else {
        /* Unstaged: `fold` is false here by construction, so norm == 2 was already demoted to
         * 0 above and this is mode 1 or nothing. */
        if (norm == 1)
            gemv_rows<MM, false, true, UN>(C, x, W, rms, gamma, M, N, K, slice, nblk, lds);
        else
            gemv_rows<MM, false, false, UN>(C, x, W, rms, gamma, M, N, K, slice, nblk, lds);
    }
}

/* The decode batch bucket is a COMPILE-TIME constant, and it has to be.
 *
 * This used to be a runtime chain (`if (M<=1) d_gemv_t<1> else if (M<=2) ...`),
 * which inlined ALL FIVE instantiations into the interpreter. Register allocation
 * is per-kernel, so the allocator budgeted for the worst arm (MM=16) and the merged
 * live ranges pushed the whole persistent kernel to arch(148) + agpr(128) = 276
 * registers -- over the 256 a wave may use at 2 waves/SIMD. That single runtime
 * switch was what blocked the 8-wave dispatch for every other op, GEMM included:
 * removing it drops the interpreter to 120 arch + 6 agpr and restores occupancy 2.
 *
 * It is the same trap the GEMM tile switch has (see PLOW_GM_T256), and the same
 * answer: plowc knows the batch bucket, so bake it in. d_gemv_t<MM> already takes a
 * runtime M and predicates on `m < M`, so one instantiation serves every M <= MM. */
#ifndef PLOW_GEMV_MM
#define PLOW_GEMV_MM 1 /* decode: one token per sequence. plowc overrides per bucket. */
#endif
PLOW_SASSERT(PLOW_GEMV_MM <= PLOW_GEMV_MAXM, "gemv bucket exceeds the GEMV path's range");

/* ------------------------- THE OUTER LOOP over M > MM ---------------------------------
 * PRICED, NOT SHIPPED. Default OFF: every object this tree builds today is byte-identical
 * with PLOW_GEMV_WALK=0 (the walk collapses to `f(0, M)`, the exact call that was there).
 *
 * WHAT THE 276-REGISTER MEASUREMENT ACTUALLY MEASURED, and why it does not apply here.
 * The removed construct was a runtime LADDER — `if (M<=1) d_gemv_t<1> else if (M<=2)
 * d_gemv_t<2> ...` — which inlines FIVE bodies whose live ranges merge, so the allocator
 * budgets for their UNION. NVIDIA's `gemv_walk` (op_gemm.cuh:118) is that same ladder,
 * because over there MM is a pure performance knob and B arrives ragged at RUNTIME.
 * Here MM is COMPILED (plowc knows the bucket), so the walk needs exactly ONE rung and the
 * ragged tail is already handled by the `m < M` predicate every row body carries. One body,
 * one live range: the register union that cost 276 never forms.
 *
 * SECOND, AND IT IS THE PART NOBODY PRICED: this loop FIXES the LDS staging bound rather
 * than fighting it. `d_gemv_t`/`d_gemv_glu`/`d_gemv_qkv` stage M*K halves of x on-chip and
 * need `M*K <= GM_LDS_HALVES` (73728). At K=5376 that is M <= 13 — BELOW the advertised
 * bucket of 16, which is exactly the silent corruption §6g-BATCH found at B=16 (slots
 * 13/14/15 fluent-but-wrong). Staging INSIDE the walk stages `rows*K <= MM*K`, a bound that
 * no longer depends on M at all. The walk is therefore a correctness fix for the bucket
 * that already ships, not only a capacity lift.
 *
 * COST MODEL, so nobody mistakes this for free throughput. The walk runs ceil(M/MM) weight
 * passes, so it is EXACTLY the traffic of ceil(M/MM) sequential launches at width MM — it
 * buys CAPACITY (M is no longer bounded by the compiled bucket) and saves only the launch
 * and gate overhead. Cutting the weight passes needs a WIDER MM, and that is the axis where
 * registers bind. The two are independent; keep them that way when reading the numbers.
 *
 * `f(m0, rows)` gets one row block of `rows` (<= PLOW_GEMV_MM) live rows starting at m0. */
#ifndef PLOW_GEMV_WALK
#define PLOW_GEMV_WALK 0
#endif
template <class F> __device__ __forceinline__ void gemv_walk(unsigned M, F f) {
#if PLOW_GEMV_WALK
    /* Uniform trip count across the workgroup: M is a wave-uniform runtime immediate off the
     * packet, so the `__syncthreads()` each staged body issues stays convergent. */
    for (unsigned m0 = 0; m0 < M; m0 += (unsigned)PLOW_GEMV_MM) {
        const unsigned rem = M - m0;
        f(m0, rem < (unsigned)PLOW_GEMV_MM ? rem : (unsigned)PLOW_GEMV_MM);
    }
#else
    f(0u, M);
#endif
}

/* THE BUCKET, MADE OBSERVABLE TO THE LOADER. This is the whole point of the symbol.
 *
 * `d_gemv_t<MM>` carries `float acc[MM]`, predicates each row on `m < M` and writes
 * `C[m*N + n]` — and has NO outer loop over M > MM (NVIDIA's `gemv_walk` in
 * op_gemm.cuh is exactly that loop; it was built here, measured at 276 registers and
 * REMOVED, see the note above). So an object compiled at MM=1 handed a packet whose
 * GEMV instructions carry M=8 writes ROW 0 and leaves rows 1..7 exactly as it found
 * them. No fault, no zero page, no trap: fluent output off stale rows, rms error
 * sqrt((T-1)/T). The packet carries M as a runtime immediate and the capacity is
 * baked into the object, so until this symbol existed NOTHING compared them.
 *
 * A `__device__` variable NAMED FOR ITS VALUE, not one holding it: the loader reads
 * the ELF `.symtab`/`.dynsym` before the object is on a device (the same reader
 * `PREFILL_ARM_MARKERS` uses in crates/plowrt/src/exec/amd.rs), so a name it can
 * parse costs nothing while a value it would have to memcpy off the GPU costs a
 * device round-trip on the load path. `extern "C"` keeps the name unmangled.
 *
 * DERIVED, never restated: the token is `PLOW_GEMV_MM` itself, so the advertised
 * capacity cannot disagree with the compiled one. Changing the bucket changes the
 * symbol in the same preprocessor expansion.
 *
 * Present in EVERY object, not just decode: `case PLOW_DOP_GEMV` is unconditional in
 * the prefill bucket too (the M=1 lm_head arm), and prefill/flash take the default
 * MM=1 — so they must advertise 1 rather than advertise nothing. */
#define PLOW_GEMV_CAP_SYM_(n) plow_gemv_mm_cap_##n
#define PLOW_GEMV_CAP_SYM(n) PLOW_GEMV_CAP_SYM_(n)
extern "C" __device__ unsigned PLOW_GEMV_CAP_SYM(PLOW_GEMV_MM) = PLOW_GEMV_MM;

/* THE WALK, MADE OBSERVABLE THE SAME WAY — because it changes what the bucket MEANS.
 *
 * Without the walk, PLOW_GEMV_MM is a hard CAPACITY: `gemv_rows<MM>` writes rows 0..MM-1 and
 * leaves the rest exactly as it found them, so plowrt's `check_gemv_capacity` must refuse any
 * packet whose widest GEMV exceeds the advertised bucket (fluent output off stale rows, rms
 * error sqrt((T-1)/T), no fault anywhere).
 *
 * WITH the walk, the same object serves any M in ceil(M/MM) row blocks, and the LDS staging
 * bound becomes min(MM, M)*K rather than M*K. Both facts flip the loader's decision, and
 * NEITHER is visible in the capacity symbol — an MM=8 walking object and an MM=8 non-walking
 * object advertise the identical `plow_gemv_mm_cap_8`. Refusing to serve M=16 on the first
 * would be a false refusal; ACCEPTING it on the second is the silent-corruption bug the
 * capacity marker was added to end. So the loader has to be told, and the same argument that
 * put the bucket in the ELF applies verbatim: the loader reads `.symtab`/`.dynsym` before the
 * object is on a device, so a name costs nothing where a value costs a device round-trip.
 *
 * DERIVED, never restated — the symbol exists iff the macro is on, in one preprocessor pass.
 * An object built with the walk OFF is byte-identical to one built before this existed. */
#if PLOW_GEMV_WALK
extern "C" __device__ unsigned plow_gemv_walk_1 = 1;
#endif

/* Stage x on-chip, then run the fused gate|up GEMV. PRECONDITION: M*K <= GM_LDS_HALVES.
 * plowc checks it and falls back to the unfused (gemv, gemv, glu) triple if it does not hold. */
__device__ void d_gemv_glu(bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu, unsigned M,
                           unsigned N, unsigned K, unsigned act, unsigned slice, unsigned nblk,
                           bf16* lds, float beta, float lbeta) {
  gemv_walk(M, [&](unsigned m0, unsigned M_) {
    const bf16* x_ = x + (size_t)m0 * K;
    bf16* C_ = C + (size_t)m0 * N;
    stage_x_lds(lds, x_, (size_t)M_ * K);
    __syncthreads();
    /* SHAPE-SPECIALISED UNROLL. The default UN=6 was chosen for Gemma's K=5376 (nchunk 11 -> two
     * groups of 6). For a K that is an exact multiple of the 512-half chunk it under-tiles: Llama
     * and Qwen have K=hidden=4096 (nchunk EXACTLY 8), so UN=6 runs TWO passes (12 chunks, the
     * second half-drained) where UN=8 runs ONE clean pass. Measured on Llama-3.1-8B decode: 5.6 ->
     * 5.4 ms/token. Both variants are MM=1 and within 3 VGPRs of each other, so the runtime switch
     * does NOT trip the register-merge cliff the M-switch did (that one merged MM up to 16). */
#ifndef PLOW_GLU_K4096_UN
#define PLOW_GLU_K4096_UN 8
#endif
    if (K == 4096)
        gemv_glu_rows<PLOW_GEMV_MM, PLOW_GLU_K4096_UN>(C_, Wg, Wu, M_, N, K, act, slice, nblk, lds, beta, lbeta);
    else
        gemv_glu_rows<PLOW_GEMV_MM>(C_, Wg, Wu, M_, N, K, act, slice, nblk, lds, beta, lbeta);
    /* Re-staging next block into the SAME arena: every wave must be done reading it. */
    if (PLOW_GEMV_WALK) __syncthreads();
  });
}

/* Stage x on-chip, then run the fused q|k|v|g GEMV. PRECONDITION: M*K <= GM_LDS_HALVES.
 * UN divides the K sweep with no dead loads: K=2560 (Qwen) -> 5 chunks, K=4096 (Llama) -> 8,
 * K=7168 (Kimi-K3) -> 14, so 7 runs two clean passes where the default 6 runs three with four
 * dead chunk slots. Every rung is MM-identical and differs only in the unroll factor, so the
 * runtime switch does not trip the register-merge cliff the M-switch did. MEASURED on the K3
 * decode shape (M=1, 4x N=12288, K=7168, 256 blocks, gfx950), runtime/tests/gemv_qkvg_gfx950_test:
 * UN=7 0.1108 ms vs UN=6 0.1276 ms -- 13% of the fused kernel, for one more rung.
 *
 * `Ng == 0` (with Wg/Cg null) is the 3-stream q|k|v form -- see gemv_qkvg_rows' ONE BODY note. */
__device__ void d_gemv_qkvg(bf16* Cq, bf16* Ck, bf16* Cv, bf16* Cg, const bf16* x, const bf16* Wq,
                          const bf16* Wk, const bf16* Wv, const bf16* Wg, unsigned M, unsigned Nq,
                          unsigned Nk, unsigned Nv, unsigned Ng, unsigned K, unsigned slice,
                          unsigned nblk, bf16* lds, const bf16* gnorm = nullptr,
                          float eps = 0.0f) {
  gemv_walk(M, [&](unsigned m0, unsigned M_) {
    const bf16* x_ = x + (size_t)m0 * K;
    bf16* Cq_ = Cq + (size_t)m0 * Nq;
    bf16* Ck_ = Ck + (size_t)m0 * Nk;
    bf16* Cv_ = Cv + (size_t)m0 * Nv;
    bf16* Cg_ = Ng ? Cg + (size_t)m0 * Ng : Cg;
    stage_x_lds(lds, x_, (size_t)M_ * K);
    __syncthreads();
#if PLOW_GLM_FUSE_QNORM
    /* Q-NORM FOLD (emit knob PLOW_GLM_FUSE_QNORM, `t[7]` on op 22): normalize the staged copy
     * IN PLACE, then fall through to the ORDINARY un-normed hot loop — the k-loop never learns
     * this happened, so it keeps its register allocation and its unroll. `gemv_norm_lds`
     * barriers on exit. See its header note for the bit-exactness argument; this is d_gemv_t's
     * `norm == 2` arm, applied to the fused three-stream sweep.
     *
     * A RUNTIME BRANCH, not a template parameter, and that is the GF=8 lesson (op_attention.h):
     * a second INSTANTIATION of `gemv_qkvg_rows` would grow the decode object for every packet
     * in the program, inside a persistent megakernel where they all share one instruction
     * stream. The branch is wave-uniform (`gnorm` comes from the packet), so it costs one
     * scalar test.
     *
     * TRAP, DO NOT FALL BACK. There is no packet to fall back TO — the emitter deleted it —
     * so a shape the fold cannot take must stop the run rather than feed the GEMV an unnormed
     * row. `glm_fuse_qnorm` in mla.rs refuses the same shapes at emit; this is the half that
     * cannot be bypassed by a hand-written blob. Scratch from the TOP of the arena, past the
     * staged row: `plow_smem` is a union, so the interpreter's `part` aliases `gm`. */
    if (gnorm) {
        if (!(((size_t)M_ * K + GV_NORM_SCRATCH <= GM_LDS_HALVES) &&
              (K <= RN_REG * PLOW_THREADS) && ((K & 7u) == 0)))
            __builtin_trap();
        gemv_norm_lds(lds, gnorm, M_, K, eps,
                      (float*)(lds + (GM_LDS_HALVES - GV_NORM_SCRATCH)));
    }
#endif
    if (K == 2560)
        gemv_qkvg_rows<PLOW_GEMV_MM, 5>(Cq_, Ck_, Cv_, Cg_, Wq, Wk, Wv, Wg, M_, Nq, Nk, Nv, Ng, K, slice, nblk, lds);
    else if (K == 4096)
        gemv_qkvg_rows<PLOW_GEMV_MM, 8>(Cq_, Ck_, Cv_, Cg_, Wq, Wk, Wv, Wg, M_, Nq, Nk, Nv, Ng, K, slice, nblk, lds);
    else if (K == 7168)
        gemv_qkvg_rows<PLOW_GEMV_MM, 7>(Cq_, Ck_, Cv_, Cg_, Wq, Wk, Wv, Wg, M_, Nq, Nk, Nv, Ng, K, slice, nblk, lds);
    else
        gemv_qkvg_rows<PLOW_GEMV_MM, 6>(Cq_, Ck_, Cv_, Cg_, Wq, Wk, Wv, Wg, M_, Nq, Nk, Nv, Ng, K, slice, nblk, lds);
    if (PLOW_GEMV_WALK) __syncthreads();
  });
}

/* The 3-stream q|k|v form. A forwarder, not a second body: see gemv_qkvg_rows' ONE BODY note. */
__device__ void d_gemv_qkv(bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x, const bf16* Wq,
                          const bf16* Wk, const bf16* Wv, unsigned M, unsigned Nq, unsigned Nk,
                          unsigned Nv, unsigned K, unsigned slice, unsigned nblk, bf16* lds,
                          const bf16* gnorm = nullptr, float eps = 0.0f) {
    d_gemv_qkvg(Cq, Ck, Cv, nullptr, x, Wq, Wk, Wv, nullptr, M, Nq, Nk, Nv, 0u, K, slice, nblk,
                lds, gnorm, eps);
}

/* PER-SHAPE UNROLL for the plain decode GEMV — BUILT, MEASURED, and DEFAULT-OFF because it drops
 * decode occupancy. Kept (behind PLOW_GEMV_PERK) because the sweep and the reason are worth more
 * than the code. This is the gemv-autotune investigation (full-grid roofline on the exact 31B
 * shapes, MI350X, 256 CUs).
 *
 * THE SWEEP. The 31B narrow decode GEMVs run SUB-CEILING, and it is NOT a weight-bandwidth deficit
 * (fp8 halved the bytes for only 1.0-1.12x) — it is IN-FLIGHT-WORK starvation: a wave owning few
 * columns of a SHORT K row issues too few loads to saturate HBM, and the per-column wave_sum does
 * not amortize. Issuing ALL of a row's chunks in ONE group (UN = nchunk) hides that latency at the
 * fixed 8-wave occupancy. MEASURED standalone, full-grid at nblk=256 (the real persistent grid),
 * streaming vec-read ceiling ~6.4 TB/s (~83% of the 8 TB/s nominal — even the wide lm_head tops out
 * there, so ~6.4 is the real ceiling, not 8):
 *
 *   o_proj  K=8192  (nchunk 16): UN 11 -> 16   4.45 -> 5.13 TB/s  (+15.4%, all 16 chunks in flight)
 *   down    K=21504 (nchunk 42): UN 11 -> 14   5.41 -> 5.63 TB/s  (+3.9%, 42/14 = 3 groups, 0 dead)
 *   lm_head K=5376  (nchunk 11): UN 11 (already one full group; at ceiling)
 *   Qwen o_proj K=4096 (nchunk 8): UN 8   Qwen K<=2560 (nchunk<=5): UN 5   Qwen down K=9728: UN 11
 *
 * This also BEATS the two other levers the sweep tried: split-K (built, lost — see §SPLIT-K) and a
 * 12-wave decode-GEMV segment (which needs its own launch class and, on top of the deep UN, added
 * only +3% to o_proj / ~0 to down — the deep UN already buys the latency-hiding a higher occupancy
 * would). And it is bit-exact: UN changes only how many loads issue before they are consumed, never
 * the sequential chunk/accumulation order.
 *
 * WHY IT IS OFF. REGISTER COUPLING is the binding constraint, not per-op bandwidth. The persistent
 * decode kernel allocates for the UNION of every inlined arm (the §PLOW_GEMV_MM M-switch trap), and
 * on the CURRENT tp decode object — already at 179 VGPR / occ-2 with the TP collectives + fp8 +
 * flash-decode merged in — there is no headroom: adding the UN=16 arm (its wv[16] alone is 64 VGPR)
 * tips the object to VGPR+AGPR 258 / occ-1 (build_gfx950.sh `check decode` FAILS at >256). occ-1
 * halves the latency-hiding of EVERY decode op, a far bigger loss than o_proj's +15%. The allocator
 * is chaotic at this size — the same arm reads 256 in one probe and 258 in the real build — so it is
 * not a 2-register nudge away either. Capturing the win needs register headroom freed on the decode
 * object, or the narrow GEMVs dispatched as a separate segment (§segmented-dispatch) with their own
 * allocation. Until then the plain path keeps the global GV_UNROLL=11. Turn PLOW_GEMV_PERK on only
 * after re-clearing `check decode`. */
#ifndef GV_UN_K8192
#define GV_UN_K8192 16
#endif
__device__ void d_gemv(bf16* C, const bf16* x, const bf16* W, const float* rms,
                       const bf16* gamma, unsigned M, unsigned N, unsigned K, int norm,
                       float eps, unsigned slice, unsigned nblk, bf16* lds GV_DYN_PARAM_DEF) {
  gemv_walk(M, [&](unsigned m0, unsigned M_) {
    /* `rms` is the PER-ROW norm scalar (gamma is per-K and does not move). d_gemv_t's own
     * LDS-fit test now sees M_ <= PLOW_GEMV_MM, which is what makes the staged arm reachable
     * at any M — see the walk's note on GM_LDS_HALVES. */
    const bf16* x_ = x + (size_t)m0 * K;
    bf16* C_ = C + (size_t)m0 * N;
    const float* rms_ = (norm == 1) ? rms + m0 : rms;
#if defined(PLOW_GEMV_PERK)
    if (K == 8192)        /* o_proj (31B): nchunk 16 -> all 16 in flight, one group */
        d_gemv_t<PLOW_GEMV_MM, GV_UN_K8192>(C_, x_, W, rms_, gamma, M_, N, K, norm, eps, slice, nblk, lds GV_DYN_ARG);
    else if (K == 4096)   /* o_proj (Qwen): nchunk 8 */
        d_gemv_t<PLOW_GEMV_MM, 8>(C_, x_, W, rms_, gamma, M_, N, K, norm, eps, slice, nblk, lds GV_DYN_ARG);
    else if (K <= 2560)   /* lm_head / small-K (Qwen): nchunk <= 5 */
        d_gemv_t<PLOW_GEMV_MM, 5>(C_, x_, W, rms_, gamma, M_, N, K, norm, eps, slice, nblk, lds GV_DYN_ARG);
    else                  /* down (31B) K=21504, lm_head (31B) K=5376, Qwen down K=9728: keep 11 */
        d_gemv_t<PLOW_GEMV_MM>(C_, x_, W, rms_, gamma, M_, N, K, norm, eps, slice, nblk, lds GV_DYN_ARG);
#else
    d_gemv_t<PLOW_GEMV_MM>(C_, x_, W, rms_, gamma, M_, N, K, norm, eps, slice, nblk, lds GV_DYN_ARG);
#endif
    if (PLOW_GEMV_WALK) __syncthreads();
  });
}

/* FP8 decode GEMV. x is staged in LDS when M*K fits (always true at decode M=1), leaving the whole
 * vector-memory path to the fp8 weight stream — exactly as the bf16 d_gemv_t does. */
__device__ void d_gemv_fp8(bf16* C, const bf16* x, const unsigned char* W, const float* wscale,
                           unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk,
                           bf16* lds) {
    /* UN per CHUNK COUNT so it DIVIDES the row (no dead converts) -- see gv_un_fp8. This picks the
     * same UN the old per-K list did for the K's it knew (2560 -> 3, 9728 -> 5) and additionally
     * covers the ones it did not (Gemma-4's 3840/15360, and 4096/8192). */
    /* PROBE (PLOW_GV_NOLDS): take the EXISTING global-x arm instead of staging, to price the
     * stage_x_lds + __syncthreads pair. Every GEMV packet pays it and the norm packets -- which
     * floor at 5-10 us against GEMV's ~33 us -- do not, which made it the first suspect for the
     * fixed per-packet cost. Correctness-safe: this arm already ships for M*K > GM_LDS_HALVES.
     *
     * MEASURED NULL, so the suspect is eliminated: Gemma-4-12B fp8+occ4, ctx 4096, 48 steps,
     * three interleaved pairs -- staged 15.362/15.386/15.406 (mean 15.385) vs NOLDS
     * 15.465/15.394/15.428 (mean 15.429), i.e. +0.29%, no better and if anything worse. Staging
     * x and its barrier are NOT what the GEMV packet pays. Kept as a probe rather than deleted
     * so the next reader does not re-run it. */
#if PLOW_GV_ABL >= 5
    /* ABLATION 5: retire the ENTIRE op, staging included. Level 4 returns from the row loop but
     * d_gemv_fp8 has already run stage_x_lds + __syncthreads -- 30 KB of LDS fill at K=15360 --
     * so level 4 is NOT a zero-cost GEMV either. */
    return;
#endif
#ifdef PLOW_GV_NOLDS
    const bool lds_ok = false;
#else
    const bool lds_ok = (size_t)M * K <= GM_LDS_HALVES;
#endif
    if (lds_ok)
        stage_x_lds(lds, x, (size_t)M * K);
    if (lds_ok) __syncthreads();
#define PLOW_GV_FP8_ARM(UNV)                                                                       \
    do {                                                                                           \
        if (lds_ok)                                                                                \
            gemv_rows_fp8<PLOW_GEMV_MM, true, UNV>(C, nullptr, nullptr, x, W, nullptr, nullptr,    \
                                                   wscale, nullptr, nullptr, M, N, 0u, 0u, K,      \
                                                   slice, nblk, lds);                              \
        else                                                                                       \
            gemv_rows_fp8<PLOW_GEMV_MM, false, UNV>(C, nullptr, nullptr, x, W, nullptr, nullptr,   \
                                                    wscale, nullptr, nullptr, M, N, 0u, 0u, K,     \
                                                    slice, nblk, lds);                             \
    } while (0)
    switch (gv_un_fp8((K + PLOW_WAVE * 16 - 1) / (PLOW_WAVE * 16))) {
        case 4: PLOW_GV_FP8_ARM(4); break;
        case 5: PLOW_GV_FP8_ARM(5); break;
        case 2: PLOW_GV_FP8_ARM(2); break;
        default: PLOW_GV_FP8_ARM(3); break;
    }
#undef PLOW_GV_FP8_ARM
}

/* NRN-FOLD STAGING — compute the WHOLE NormResidualNorm (op 23) into the GEMV's LDS arena, so the
 * end-of-layer sandwich packet disappears from the decode chain.
 *
 * The Gemma decode chain is qkv -> hnr -> flash -> merge -> o -> NRN1 -> glu -> down -> NRN2 ->
 * (next) qkv: nine serial counter gates per layer at ~10 us each, and the b=1 NRN packets are pure
 * chain levels (one workgroup computing 3840 elements while 303 CUs wait on the counter). NRN1
 * cannot fold into its consumer (GemvGluFp8 has two free t slots against four operands) — this is
 * the "norm fusion capped by t[8]" item — but NRN2's consumers are the next layer's q/k/v
 * GemvFp8s, which have exactly the four free slots (t3/t4/t6/t7, with t1 repurposed as the
 * residual input since no normed x exists to read). Each of the three projections computes the
 * NRN redundantly from operands it can load itself (~2 us of extra reduction per packet, off the
 * weight-stream pipe) and ONE designated packet's slice 0 stores the residual.
 *
 * WHY THE RESIDUAL PING-PONGS (in -> out are DIFFERENT tensors, unlike op 23's in-place form):
 * q, k and v run CONCURRENTLY on disjoint CU sets. If the q packet stored the new residual over
 * act.x while a k workgroup was still loading act.x as its `a` input, k would compute the NRN of
 * a half-updated row — finite, plausible, wrong. So NRN1 (still a packet) writes its residual to
 * act.xr, this fold reads a = act.xr and stores resid_out = act.x, and the buffers alternate at
 * every half-layer with the SAME assignment in every layer. The last layer keeps the unfused op
 * 23 (its second norm is the model's FINAL norm, whose consumer is the lm_head, not a GEMV).
 *
 * BIT-EXACT BY CONSTRUCTION, same argument as `gemv_norm_lds`: this replicates
 * `d_norm_residual_norm`'s `fits` arm element-for-element — same per-thread map, same serial
 * accumulation order, same `block_sum`, same bf16 ROUNDING of the residual before the second
 * reduction — and then the ordinary un-normed hot loop runs over LDS bytes identical to the HBM
 * bytes the deleted packet would have written. The emitter refuses shapes the `fits` arm would
 * not take; the wrapper traps rather than falling back, because there is no packet to fall back
 * TO. */
__device__ __forceinline__ void gemv_nrn_lds(bf16* __restrict__ lds, bf16* __restrict__ resid_,
                                             const bf16* __restrict__ a_,
                                             const bf16* __restrict__ b_,
                                             const bf16* __restrict__ gb_,
                                             const bf16* __restrict__ gn_, unsigned M, unsigned K,
                                             float eps, float scale, bool store, float* part) {
    const auto* const ag = as_glob(a_);
    const auto* const bg = as_glob(b_);
    const auto* const gbg = as_glob(gb_);
    const auto* const gng = as_glob(gn_);
    auto* const rg = as_glob(resid_);
    for (unsigned m = 0; m < M; m++) {
        const size_t base = (size_t)m * K;
        /* PRODUCE: residual a, sublayer output b, and BOTH weights — all issued together. */
        bf16v8 av[RN_VEC], bv[RN_VEC], wb[RN_VEC], wn[RN_VEC];
#pragma unroll
        for (int c = 0; c < RN_VEC; c++) {
            const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
            av[c] = bf16v8_zero();
            bv[c] = bf16v8_zero();
            wb[c] = bf16v8_zero();
            wn[c] = bf16v8_zero();
            if (i < K) {
                av[c] = ld_glob8(ag + base + i);
                bv[c] = ld_glob8(bg + base + i);
                if (gbg) wb[c] = ld_glob8(gbg + i);
                if (gng) wn[c] = ld_glob8(gng + i);
            }
        }
        float ssb = 0.0f;
#pragma unroll
        for (int c = 0; c < RN_VEC; c++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = bf2f(bv[c][j]);
                ssb += f * f;
            }
        const float invb = rsqrtf(block_sum(ssb, part) / (float)K + eps);
        /* resid = (a + norm(b)*gb) * scale, ROUNDED to bf16; the SECOND reduction runs over the
         * rounded value, exactly reproducing op 23's bf16 store + reload. */
        bf16v8 rv[RN_VEC];
        float ssr = 0.0f;
#pragma unroll
        for (int c = 0; c < RN_VEC; c++) {
            bf16v8 r;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float g = gbg ? bf2f(wb[c][j]) : 1.0f;
                const float f = (bf2f(av[c][j]) + bf2f(bv[c][j]) * invb * g) * scale;
                r[j] = f2bf(f);
                const float rf = bf2f(r[j]);
                ssr += rf * rf;
            }
            rv[c] = r;
        }
        const float invr = rsqrtf(block_sum(ssr, part) / (float)K + eps);
#pragma unroll
        for (int c = 0; c < RN_VEC; c++) {
            const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
            if (i < K) {
                bf16v8 no;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gng ? bf2f(wn[c][j]) : 1.0f;
                    no[j] = f2bf(bf2f(rv[c][j]) * invr * g);
                }
                *(bf16v8*)(lds + base + i) = no;
                if (store) st_glob8(rg + base + i, rv[c]);
            }
        }
    }
    /* block_sum's trailing barrier ran BEFORE the LDS write-back; the barrier the hot loop
     * needs is this one (see gemv_norm_lds). */
    __syncthreads();
}

/* op 30 with the NRN fold: stage via gemv_nrn_lds instead of stage_x_lds, then run the ordinary
 * staged hot loop — the k-loop never learns this happened. `store` marks the ONE packet of the
 * q/k/v trio that owns the residual write; within it, slice 0 stores (all slices compute
 * identical bytes, so this is a traffic choice, not a correctness one). */
__device__ void d_gemv_fp8_nrn(bf16* C, bf16* resid_out, const bf16* a, const bf16* b,
                               const bf16* gb, const bf16* gn, const unsigned char* W,
                               const float* wscale, unsigned M, unsigned N, unsigned K, float eps,
                               float scale, unsigned store, unsigned slice, unsigned nblk,
                               bf16* lds) {
    const bool ok = ((size_t)M * K + GV_NORM_SCRATCH <= GM_LDS_HALVES) &&
                    (K <= RN_REG * PLOW_THREADS) && ((K & 7u) == 0);
    if (!ok) __builtin_trap(); /* the emitter refuses these shapes; no packet exists to fall back to */
    /* Scratch from the TOP of the arena, past the staged row — sm->part aliases sm->gm
     * (see d_gemv_t's note); deriving it here removes the alias outright. */
    float* const nscratch = (float*)(lds + (GM_LDS_HALVES - GV_NORM_SCRATCH));
    gemv_nrn_lds(lds, resid_out, a, b, gb, gn, M, K, eps, scale, store != 0 && slice == 0,
                 nscratch);
#define PLOW_GV_NRN_ARM(UNV)                                                                       \
    gemv_rows_fp8<PLOW_GEMV_MM, true, UNV>(C, nullptr, nullptr, nullptr, W, nullptr, nullptr,      \
                                           wscale, nullptr, nullptr, M, N, 0u, 0u, K, slice, nblk, \
                                           lds)
    switch (gv_un_fp8((K + PLOW_WAVE * 16 - 1) / (PLOW_WAVE * 16))) {
        case 4: PLOW_GV_NRN_ARM(4); break;
        case 5: PLOW_GV_NRN_ARM(5); break;
        case 2: PLOW_GV_NRN_ARM(2); break;
        default: PLOW_GV_NRN_ARM(3); break;
    }
#undef PLOW_GV_NRN_ARM
}

/* FUSED Q|K|V fp8 DECODE GEMV — op 22's per-channel-fp8 twin, and op 30's own body with
 * Nk = Nv = 0 (see gemv_rows_fp8's header note). The three f32[N] dequant-scale rows arrive as
 * TENSOR HANDLES in i5/i6/i7 (dev_isa.h PLOW_DOP_GEMV_QKV_FP8: ten pointers do not fit t[8], and
 * op 114 set the rule for which operand is demoted). Byte-exact to the three d_gemv_fp8 calls it
 * replaces: a column's value depends only on its own weight row, its own scale and the shared x —
 * concatenating the sweeps changes which workgroup owns a column, never what it computes. */
__device__ void d_gemv_qkv_fp8(bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x,
                               const unsigned char* Wq, const unsigned char* Wk,
                               const unsigned char* Wv, const float* Sq, const float* Sk,
                               const float* Sv, unsigned M, unsigned Nq, unsigned Nk, unsigned Nv,
                               unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    const bool lds_ok = (size_t)M * K <= GM_LDS_HALVES;
    if (lds_ok)
        stage_x_lds(lds, x, (size_t)M * K);
    if (lds_ok) __syncthreads();
#define PLOW_GV_QKV_FP8_ARM(UNV)                                                                   \
    do {                                                                                           \
        if (lds_ok)                                                                                \
            gemv_rows_fp8<PLOW_GEMV_MM, true, UNV>(Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv, M, Nq,   \
                                                   Nk, Nv, K, slice, nblk, lds);                   \
        else                                                                                       \
            gemv_rows_fp8<PLOW_GEMV_MM, false, UNV>(Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv, M, Nq,  \
                                                    Nk, Nv, K, slice, nblk, lds);                  \
    } while (0)
    /* Same dead-chunk rule as d_gemv_fp8: all three streams share K, so one UN serves them all. */
    switch (gv_un_fp8((K + PLOW_WAVE * 16 - 1) / (PLOW_WAVE * 16))) {
        case 4: PLOW_GV_QKV_FP8_ARM(4); break;
        case 5: PLOW_GV_QKV_FP8_ARM(5); break;
        case 2: PLOW_GV_QKV_FP8_ARM(2); break;
        default: PLOW_GV_QKV_FP8_ARM(3); break;
    }
#undef PLOW_GV_QKV_FP8_ARM
}

/* BLOCK-scaled fp8 decode GEMV — DeepSeek/GLM weight_block_size [128,128]. wscale is the
 * ceil(N/128) x ceil(K/128) f32 scale grid (row-major). Otherwise identical to d_gemv_fp8.
 *
 * UN per K, keyed on the fp8 chunk count nchunk = ceil(K/1024) (one chunk = PLOW_WAVE*16 = 1024
 * K-elements). This kernel streams 2 output columns/wave, so UN loads/col = 2*UN in flight; more
 * in-flight loads fill the memory pipe on long-K, but a dead overshoot chunk still costs a full
 * 16-cvt CONVERT, so UN must DIVIDE nchunk (no wasted VALU). MEASURED (gfx950, standalone GLM
 * shapes, block_fp8_gfx950_test) vs the old fixed UN=3, weight-stream TB/s:
 *   o_proj  K=16384 (16 chunks): 1.65 -> 2.16 (UN=8, +31%)   dense down K=12288 (12): 1.68 -> 1.97 (UN=6, +17%)
 *   q_b/moe_down K=2048 (2):    1.14/0.83 -> 1.37/0.97 (UN=2, +20/17%)   kv_b K=512 (1): 0.33 -> 0.42 (UN=2)
 *   q_a/kv_a/moe_glu K=6144 (6): 1.10 (UN=3, unchanged — the default already divides 6).
 * Occupancy stays 2 waves/SIMD, 0 spill through UN=8 (VGPRs 111/139/166). Standalone numbers are
 * launch-floor-diluted; in-model the linears are already near-roofline so the realized gain is small
 * (the campaign's win is MLA/MoE MFMA, not the linears) — but the adaptive UN is strictly >= UN=3 on
 * every shape at matched K, zero risk (overshoot stays correct at any UN). */
#define GEMV_FP8_BLK_DISP(UN)                                                                       \
    do {                                                                                            \
        if (lds_ok) gemv_rows_fp8_blk<PLOW_GEMV_MM, true, UN>(C, x, W, wscale, M, N, K, slice, nblk, lds);  \
        else gemv_rows_fp8_blk<PLOW_GEMV_MM, false, UN>(C, x, W, wscale, M, N, K, slice, nblk, lds); \
    } while (0)
__device__ void d_gemv_fp8_blk(bf16* C, const bf16* x, const unsigned char* W, const float* wscale,
                               unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk,
                               bf16* lds) {
    const bool lds_ok = (size_t)M * K <= GM_LDS_HALVES;
    if (lds_ok)
        stage_x_lds(lds, x, (size_t)M * K);
    if (lds_ok) __syncthreads();
    /* UN must DIVIDE nchunk. Two GLM TP4 shapes fell through to a UN that does not, and both were
     * measured losing double digits on gfx950 (grid 256, blockDim 512, 6200 GB/s denominator):
     *   nchunk = 4 (K=4096, GLM o_proj under GLM_LINEAR_FP8): UN=3 issued 6 loads for 4 live
     *       chunks — 2 dead converts per group.   UN 3 -> 4:  4245 -> 5852 GB/s (+38%)
     *   nchunk = 1 (K<=1024, GLM shared/MoE down at TP4): UN=2 issued 2 loads for 1 live chunk —
     *       HALF the issue slots dead.            UN 2 -> 1:  1306 -> 1581 GB/s (+21%)
     * nchunk = 6 keeps UN=3 (two clean groups) and NOT UN=6: measured 6670 vs 6165 GB/s at
     * N=2624 K=6144 — one group of 6 has more loads in flight but a longer dependence chain to the
     * first wave_sum, and at 2-3 columns/wave there is no second column to overlap it with. */
    const unsigned nchunk = (K + 1023u) >> 10;
    if (nchunk >= 16u) GEMV_FP8_BLK_DISP(8);       /* o_proj 16384: 16 chunks -> 2 groups of 8 */
    else if (nchunk >= 11u) GEMV_FP8_BLK_DISP(6);  /* dense down 12288: 12 chunks -> 2 groups of 6 */
    else if (nchunk <= 1u) GEMV_FP8_BLK_DISP(1);   /* K <= 1024: ONE chunk, so UN=1 or half is dead */
    else if (nchunk == 2u) GEMV_FP8_BLK_DISP(2);   /* short-K (q_b/kv_b): fewest dead converts */
    else if (nchunk == 4u) GEMV_FP8_BLK_DISP(4);   /* K=4096 (o_proj): one exact group of 4 */
    else GEMV_FP8_BLK_DISP(3);                     /* K=6144: 6 chunks -> 2 clean groups of 3 */
}
#undef GEMV_FP8_BLK_DISP

/* MXFP4 decode GEMV entry. Mirrors d_gemv_fp8: stage x in LDS when it fits, then pick UN so it
 * DIVIDES the chunk count (a dead overshoot chunk still costs a full 16-convert dequant, so an UN
 * that does not divide nchunk burns VALU for nothing).
 *
 * An mxfp4 chunk is 2048 K-elements — twice the fp8 chunk, four times the bf16 one — so nchunk is
 * small on real shapes and the table is correspondingly short: K=4096 -> 2, K=8192 -> 4,
 * K=16384 -> 8. UN=2 is the floor because two columns are already in flight per wave.
 *
 * FUSED Q|K|V, the mxfp4 twin of d_gemv_qkv. Kimi-K3's three MLA down-projections (q_a -> 1536,
 * kv_a -> 512, k_rope -> 64) all read the SAME pre-normed x[7168], so their output columns
 * concatenate into one N = Nq+Nk+Nv sweep exactly as the bf16 form's do. At batch-1 decode these
 * kernels are LAUNCH-BOUND, not bandwidth-bound (a GEMV census measured every K3 decode GEMV bar
 * lm_head pinned at a ~0.032 ms launch floor at under 3% of achievable bandwidth), so removing
 * packets is the whole win -- and k_rope's 64 columns over 256 workgroups is the extreme case:
 * `per = 1` leaves 192 of the 256 CUs owning NOTHING at all.
 *
 * OUTPUT-DIMENSION MERGE, which is the direction that is safe. Per-CU column count RISES
 * (6 / 2 / 1 for the three split sweeps -> 9 for the fused one at K3's geometry), so the op gets
 * WIDER and nothing that ran in parallel starts running in sequence. Contrast knob-contract
 * 6g-KNOBS' GLM_GROUP=1, which removed 38% of the ops for +2.88 ms by collapsing disjoint-CU work
 * into a loop.
 *
 * THE NINTH AND TENTH POINTERS ARE THE SCALE ROWS, and they live in i[]. Three outputs + three
 * fp4 weights + three E8M0 scale rows + x is TEN pointers against DevInst's eight `t` slots and a
 * fixed 64-byte wire instruction. `PLOW_DOP_GEMV_QKVG` set the rule for which operand gets
 * demoted -- a WEIGHT, because a wrong weight handle reads the wrong bytes and the output is
 * visibly garbage, where a wrong OUTPUT handle silently overwrites an unrelated tensor. An E8M0
 * scale row is the same kind of operand and strictly safer: it is read-only, it is HALF of the
 * weight (the fp4 nibbles are meaningless without their exponents), and a wrong one is off by a
 * per-block power of two -- garbage that is visible in the first token. So `t[0..6]` stays
 * byte-for-byte `PLOW_DOP_GEMV_QKV`'s and the three scale handles take i5/i6/i7, the three
 * integer slots op 22 leaves empty. No output is demoted and nothing is packed into a blob.
 * See dev_isa.h PLOW_DOP_GEMV_QKV_MXFP4 for the packet spec and the rejected alternatives. */
__device__ void d_gemv_qkv_mxfp4(bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x,
                                 const unsigned char* Wq, const unsigned char* Wk,
                                 const unsigned char* Wv, const unsigned char* Sq,
                                 const unsigned char* Sk, const unsigned char* Sv, unsigned M,
                                 unsigned Nq, unsigned Nk, unsigned Nv, unsigned K, unsigned slice,
                                 unsigned nblk, bf16* lds) {
    const bool lds_ok = (size_t)M * K <= GM_LDS_HALVES;
    if (lds_ok) {
        stage_x_lds(lds, x, (size_t)M * K);
        __syncthreads();
    }
    const unsigned nchunk = (K + PLOW_WAVE * 32 - 1) / (PLOW_WAVE * 32);
    /* The UN table is keyed on K ALONE, so a fused packet and the split packets it replaces pick
     * the SAME unroll at the same K. That is what makes the two accumulate identical chunk
     * sequences, and therefore what makes the fusion byte-exact rather than merely close. */
#define GEMV_MXFP4_DISP(UN)                                                                      \
    do {                                                                                         \
        if (lds_ok)                                                                              \
            gemv_rows_mxfp4<PLOW_GEMV_MM, true, UN>(Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv, M, Nq, \
                                                    Nk, Nv, K, slice, nblk, lds);                 \
        else                                                                                     \
            gemv_rows_mxfp4<PLOW_GEMV_MM, false, UN>(Cq, Ck, Cv, x, Wq, Wk, Wv, Sq, Sk, Sv, M,    \
                                                     Nq, Nk, Nv, K, slice, nblk, lds);            \
    } while (0)
    if (nchunk >= 8u) GEMV_MXFP4_DISP(4);      /* K>=16384: 8 chunks -> 2 clean groups of 4 */
    else if (nchunk >= 6u) GEMV_MXFP4_DISP(3); /* K=12288: 6 -> 2 groups of 3               */
    else GEMV_MXFP4_DISP(2);                   /* K<=8192: 2 or 4 chunks                    */
#undef GEMV_MXFP4_DISP
}

/* The 1-stream form. A forwarder, not a second body: see gemv_rows_mxfp4's ONE BODY note. */
__device__ void d_gemv_mxfp4(bf16* C, const bf16* x, const unsigned char* W,
                             const unsigned char* S, unsigned M, unsigned N, unsigned K,
                             unsigned slice, unsigned nblk, bf16* lds) {
    d_gemv_qkv_mxfp4(C, nullptr, nullptr, x, W, nullptr, nullptr, S, nullptr, nullptr, M, N, 0u, 0u,
                     K, slice, nblk, lds);
}

/* MXFP4 decode fused gate|up GEMV+GLU (w4a16) — the mxfp4 twin of d_gemv_glu_fp8, built on the SAME
 * verified fp4 machinery as gemv_rows_mxfp4: packed-2/byte weight loads (offset >>1), one E8M0 scale
 * byte per 32 K folded into fp4_to_bf16v8x4 — so there is NO per-channel epilogue dequant (contrast
 * the fp8 twin's *gscale / *uscale, whose arbitrary-f32 scale must stay a separate multiply). Gate and
 * up are two fp4 weight matrices Wg/Wu each with its own E8M0 scale row Sg/Su; the epilogue is
 * act(g)*u. One wave owns one output column (like the fp8 twin), so 4 fp4->bf16 converts + 4 fdot2
 * per fragment for each of gate and up. HARDWARE-VALIDATE numerics against golden d_gemv_glu_mxfp4_k
 * (test_kernels.hip); the fp4 convert + fdot2 primitives are already device-validated by
 * gemv_rows_mxfp4, so only the gate/up fusion + SwiGLU epilogue are new. */
#ifndef GV_UNROLL_GLU_MXFP4
#define GV_UNROLL_GLU_MXFP4 2
#endif
template <int MM, int UN = GV_UNROLL_GLU_MXFP4>
__device__ __forceinline__ void gemv_glu_rows_mxfp4(
    bf16* __restrict__ C_, const unsigned char* __restrict__ Wg_,
    const unsigned char* __restrict__ Wu_, const unsigned char* __restrict__ Sg_,
    const unsigned char* __restrict__ Su_, unsigned M, unsigned N, unsigned K, unsigned act,
    unsigned slice, unsigned nblk, const bf16* lds) {
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned step = PLOW_WAVE * 32;   /* one pass: 64 lanes x 32 fp4 = 2048 K */
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned wbytes = K >> 1;         /* packed weight row stride (2 fp4/byte) */
    const unsigned nsb = (K + 31) >> 5;     /* E8M0 scale bytes per row              */

    auto* const C = as_glob(C_);
    const auto* const Wg = as_glob(Wg_);
    const auto* const Wu = as_glob(Wu_);
    const auto* const Sg = as_glob(Sg_);
    const auto* const Su = as_glob(Su_);
    auto xv8 = [&](unsigned m, unsigned kk) -> bf16v8 { return ld_lds8(lds + (size_t)m * K + kk); };

    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = slice * per;
    const unsigned n1 = (n0 + per < N) ? (n0 + per) : N;

    for (unsigned n = n0 + wave; n < n1; n += PLOW_WAVES) {
        const __amdgpu_buffer_rsrc_t rg = buf_rsrc_fp8(Wg + (size_t)n * wbytes, wbytes);
        const __amdgpu_buffer_rsrc_t ru = buf_rsrc_fp8(Wu + (size_t)n * wbytes, wbytes);
        const unsigned char* const sg = Sg + (size_t)n * nsb;
        const unsigned char* const su = Su + (size_t)n * nsb;
        float ag[MM], au[MM];
#pragma unroll
        for (int m = 0; m < MM; m++) { ag[m] = 0.0f; au[m] = 0.0f; }

        for (unsigned c = 0; c < nchunk; c += UN) {
            fp4v32 wg[UN], wu[UN];
            /* Both gate and up weight loads issued before either is consumed: 2*UN in flight. */
#pragma unroll
            for (int u = 0; u < UN; u++)
                wg[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_mxfp4(rg, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++)
                wu[u] = __builtin_bit_cast(
                    fp4v32, buf_ld_mxfp4(ru, ((c + (unsigned)u) * step + lane * 32) >> 1));
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 32;
                const unsigned kx = (k < K) ? k : 0u;
                const unsigned sb = k >> 5; /* this lane's block index; one E8M0 scale per fragment */
                const float scg = (sb < nsb) ? e8m0_to_f32(sg[sb]) : 0.0f;
                const float scu = (sb < nsb) ? e8m0_to_f32(su[sb]) : 0.0f;
                const fp4_frag32 gf = fp4_prep32(wg[u], scg);
                const fp4_frag32 uf = fp4_prep32(wu[u], scu);
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    const bool live = ((unsigned)m < M);
                    const bf16v8 x0 = xv8(live ? m : 0, kx);
                    const bf16v8 x1 = xv8(live ? m : 0, kx + 8);
                    const bf16v8 x2 = xv8(live ? m : 0, kx + 16);
                    const bf16v8 x3 = xv8(live ? m : 0, kx + 24);
                    ag[m] += live ? fp4_dot32(gf, x0, x1, x2, x3, 0.0f) : 0.0f;
                    au[m] += live ? fp4_dot32(uf, x0, x1, x2, x3, 0.0f) : 0.0f;
                }
            }
        }
        /* No per-channel dequant: the MX E8M0 scale is already folded into every converted element
         * by fp4_to_bf16v8x4 (contrast d_gemv_glu_fp8's `* gscale[n]` / `* uscale[n]`). */
#pragma unroll
        for (int m = 0; m < MM; m++) {
            const float g = wave_sum(ag[m]);
            const float uu = wave_sum(au[m]);
            if (lane == 0 && (unsigned)m < M) {
                const float s = act_gate_only(g, act);
                st_act1(&C[(size_t)m * N + n], f2bf(s * uu));
            }
        }
    }
}

/* MXFP4 decode fused gate|up GEMV entry. Mirrors d_gemv_glu_fp8: stage x in LDS (PRECONDITION
 * M*K <= GM_LDS_HALVES), then one wave per output column computes gate & up and fuses act(g)*u. */
__device__ void d_gemv_glu_mxfp4(bf16* C, const bf16* x, const unsigned char* Wg,
                                 const unsigned char* Wu, const unsigned char* Sg,
                                 const unsigned char* Su, unsigned M, unsigned N, unsigned K,
                                 unsigned act, unsigned slice, unsigned nblk, bf16* lds) {
    stage_x_lds(lds, x, (size_t)M * K);
    __syncthreads();
    gemv_glu_rows_mxfp4<PLOW_GEMV_MM>(C, Wg, Wu, Sg, Su, M, N, K, act, slice, nblk, lds);
}

/* FP8 decode fused gate|up GEMV+GLU. PRECONDITION (as the bf16 twin): M*K <= GM_LDS_HALVES. */
__device__ void d_gemv_glu_fp8(bf16* C, const bf16* x, const unsigned char* Wg,
                              const unsigned char* Wu, const float* gscale, const float* uscale,
                              unsigned M, unsigned N, unsigned K, unsigned act, unsigned slice,
                              unsigned nblk, bf16* lds) {
#if PLOW_GV_ABL >= 4
    /* THE GLU ARM IS A SEPARATE FUNCTION, so PLOW_GV_ABL levels 1-3 (which patch gemv_rows_fp8)
     * never touched it -- and it is the LARGEST op in the decode layer (57-59 us, 27% of it).
     * Any "GEMV-free floor" measured without this line is not GEMV-free. */
    return;
#endif
    stage_x_lds(lds, x, (size_t)M * K);
    __syncthreads();
    /* Same dead-chunk rule as d_gemv_fp8, which this path did not have AT ALL -- it always ran
     * GV_UNROLL_GLU_FP8=3, tuned for Qwen's K=2560. Gemma-4's gate|up is K=3840 (nchunk=4), and
     * this is the single largest GEMV consumer in the decode trace (3.954 ms of a 14.87 ms token),
     * so a 50% dead-convert rate here is the most expensive instance of the bug. */
    /* PROBE (PLOW_GLU_UN): the GLU arm carries FOUR weight streams (gate|up x two columns), so UN
     * loads in flight is 4*UN fp8v16 = 16*UN VGPRs of weights alone -- 64 at UN=4 against this
     * object's 104-VGPR budget. The divisor rule is right about DEAD WORK but says nothing about
     * register pressure, and this is the largest single op in the decode layer.
     *
     * MEASURED (MI300X, Gemma-4-12B fp8+bf16KV, cap blob, 3 reps, ms/token) -- the rule WINS and
     * the register argument does not bite:
     *
     *     ctx     UN=4 (the rule)   UN=3        UN=2
     *     4096    12.160            12.565      12.117
     *     8192    12.231            12.620      12.216
     *
     * UN=3 is +3.3%: nchunk = 3840/1024 = 4 is not divisible by 3, so a third of the converts run
     * on zeros -- an independent confirmation of gv_un_fp8 on the op that costs the most. UN=2
     * also divides 4 and halves the weight registers, but its edge is -0.35%/-0.12%, at or below
     * this box's 0.4% run-to-run drift. Left on the rule; probe kept so this is not re-run. */
#ifdef PLOW_GLU_UN
    switch (PLOW_GLU_UN) {
#else
    switch (gv_un_fp8((K + PLOW_WAVE * 16 - 1) / (PLOW_WAVE * 16))) {
#endif
        case 4:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 4>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        case 5:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 5>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        case 2:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 2>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        default:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 3>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
    }
}

/* d_gemv_glu_fp8 WITH the NRN1 fold: the post-attention NormResidualNorm (op 23) computed into
 * this GEMV's staging — see gemv_nrn_lds for the mechanism, the bit-exactness argument and the
 * residual ping-pong. This op is ONE packet (no q/k/v trio), so slice 0 alone stores the residual
 * and there is no sibling race. The five extra operands ride the wire as t1 (a, replacing the x
 * that no longer exists) plus FOUR TENSOR HANDLES in i3/i4/i6/i7 (op 114's demotion — the "GLU
 * fusion needs 9 slots" note predates that precedent), eps/scale in f0/f1, flag in fj[2].u.
 *
 * A SEPARATE FUNCTION, not a branch inside d_gemv_glu_fp8, and that is a MEASURED requirement:
 * the branched form left resource stats untouched (104 VGPR / 0 spill / same scratch) and still
 * cost +0.5 ms/token on blobs that never take the fold — the `if (nrn)` around the staging was
 * enough to degrade the hot loop's schedule. Keeping the original function byte-for-byte and
 * paying the duplicated dispatch here restores it. */
__device__ void d_gemv_glu_fp8_nrn(bf16* C, bf16* resid_out, const bf16* a, const bf16* b,
                                   const bf16* gb, const bf16* gn, const unsigned char* Wg,
                                   const unsigned char* Wu, const float* gscale,
                                   const float* uscale, unsigned M, unsigned N, unsigned K,
                                   unsigned act, float eps, float scale, unsigned slice,
                                   unsigned nblk, bf16* lds) {
#if PLOW_GV_ABL >= 4
    return;
#endif
    const bool ok = ((size_t)M * K + GV_NORM_SCRATCH <= GM_LDS_HALVES) &&
                    (K <= RN_REG * PLOW_THREADS) && ((K & 7u) == 0);
    if (!ok) __builtin_trap(); /* no packet exists to fall back to */
    float* const nscratch = (float*)(lds + (GM_LDS_HALVES - GV_NORM_SCRATCH));
    gemv_nrn_lds(lds, resid_out, a, b, gb, gn, M, K, eps, scale, slice == 0, nscratch);
#ifdef PLOW_GLU_UN
    switch (PLOW_GLU_UN) {
#else
    switch (gv_un_fp8((K + PLOW_WAVE * 16 - 1) / (PLOW_WAVE * 16))) {
#endif
        case 4:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 4>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        case 5:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 5>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        case 2:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 2>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
        default:
            gemv_glu_rows_fp8<PLOW_GEMV_MM, 3>(C, Wg, Wu, gscale, uscale, M, N, K, act, slice, nblk, lds);
            break;
    }
}


#endif /* PLOW_OP_GEMM_H */
