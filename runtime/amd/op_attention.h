/* op_attention.h — FlashAttention (CDNA4 / gfx950), bf16, GQA, causal + sliding.
 *
 * Two shapes must both work, and they are very different:
 *
 *   sliding layers : head_dim 256, GQA 2:1  (31B) — 1024-token window, causal
 *   global  layers : head_dim 512, GQA 8:1  (31B) / MQA 16:1 (12B) — full causal
 *
 * head_dim=512 is the constraint that shapes everything. The design that makes
 * it fit:
 *
 *   Q lives in REGISTERS as MFMA A-fragments, not in LDS.
 *     Q as LDS would be 128 rows x 512 x 2B = 128 KiB, leaving no room for K and
 *     V. As registers it is D/2 halves per lane = 128 VGPRs at D=512, and the O
 *     accumulator is D/32 f32x16 = 256 AccVGPRs. gfx950 has 256 arch + 256 acc
 *     per lane, so that fits with ~50 VGPRs to spare — at exactly 1 wave/SIMD,
 *     which is all a persistent 256-thread workgroup needs anyway.
 *   Only K and V are staged in LDS: 2 x 32 x (D+8) x 2B = 65 KiB at D=512.
 *
 * Each workgroup owns one (q-tile, head); its 4 waves each own 32 query rows, so
 * a q-tile is 128 rows and all four waves share the same staged K/V tile.
 *
 * NUMERICS:
 *   scale = 1.0 for Gemma 4. HF literally sets `self.scaling = 1.0` — there is
 *   NO 1/sqrt(head_dim), and no `query_pre_attn_scalar` exists in any Gemma 4
 *   config. The softmax temperature is absorbed into the learned q_norm weight.
 *   It is a parameter here only so the op is reusable by other models.
 *
 *   Sliding window is INCLUSIVE of the current token: keep iff
 *   0 <= q_idx - kv_idx <= window-1. An off-by-one here shifts every layer's
 *   receptive field by one token and still produces fluent output.
 *
 * V is a separate tensor from K even on the global layers. `attention_k_eq_v`
 * means there is no v_proj — but K and V are still different tensors: both come
 * from k_proj(x), then K gets k_norm + RoPE while V gets the weightless v_norm
 * and no RoPE. Feeding K in as V is a real and tempting bug.
 */
#ifndef PLOW_OP_ATTENTION_H
#define PLOW_OP_ATTENTION_H

/* V rows in flight per thread in the flash-decode accumulate. The row loop used to issue one
 * 2-byte load, wait for it, and branch -- no memory-level parallelism at all, in the op that
 * moves 2.8 GB per token at ctx=3326. */
#ifndef FA_DEC_V_UNROLL
#define FA_DEC_V_UNROLL 8
#endif

#include "amd_common.h"

#define FA_BQ 32   /* query rows per wave; 4 waves => 128-row q-tile */
#define FA_BKV 32  /* KV rows staged per step (one MFMA N tile)      */
#define FA_PAD 8   /* LDS row padding, in halves, to break bank conflicts */

/* KV rows staged per step, PER HEAD DIM. Default 32 everywhere — the shipped block.
 *
 * FA_BKV_D128=64 IS BUILT, CORRECT, AND A LOSS. Do not turn it on without reading this.
 * The idea: at D=128 stage 64 KV rows and run the two 32x32 QK/PV subtiles under ONE
 * online-softmax pass (one max/exp/rescale, one K/V HBM->LDS staging, half the barriers).
 * It targets the softmax+barrier overhead that dominates the small-D tile. It is exact
 * (golden ATTENTION CORRECT at hd=128, all nsplit) and it FITS (2 waves/SIMD, LDS 66 KiB
 * < the GEMM-sized union). It is simply slower, measured on MI350X gfx950:
 *
 *   standalone flash prefill (gemma_flash_prefill_128, ROCR dev, best of 40):
 *     T=4096  BKV32 1.85 ms / 74 TF/s   ->  BKV64 2.66 ms / 52 TF/s   (-30%)
 *     T=8192  BKV32 6.22 ms / 88 TF/s   ->  BKV64 9.00 ms / 61 TF/s   (-31%)
 *   end-to-end Qwen3-4B prefill (segmented, PLOW_FLASH_THREADS=512, median of 5):
 *     4k  145 ms -> 174 ms      8k  367 ms -> 468 ms
 *
 * WHY. The combined softmax needs BOTH 32x32 score accumulators (s[2] = 32 VGPR) and a
 * doubled p[2][16] live at once; on top of the oacc[4] f32x16 (in VGPR — the MFMA writes
 * VGPR, AGPRs=0) that pushes the flash kernel to 97 VGPRs of SCRATCH SPILL (vs 16 at
 * BKV=32), and a spilling flash inner loop costs more than the barriers it removes. And
 * the barriers were never the bottleneck: the combined-P variant removes the MOST barriers
 * yet is the SLOWEST, so D=128 flash at 8 waves is bound by LDS/MFMA/softmax throughput —
 * work that BKV=64 reorganises but does not reduce. See plans/qwen-flash-plus.md.
 *
 * D=256/512 (Gemma) is unconditionally 32: those arms size the FA_LDS_HALVES(512) union and
 * live in the 4-wave/512-reg flash object where the second accumulator would not fit. */
#ifndef FA_BKV_D128
#define FA_BKV_D128 32
#endif
#define FA_BKV_OF(D) (((D) <= 128) ? FA_BKV_D128 : 32)

#define FA_NEG_INF (-3.0e38f)

/* EXP2, NOT EXP. AMD's transcendental unit has `v_exp_f32`, and it computes 2^x -- there is no
 * base-e exponential in hardware. `__expf(x)` therefore lowers to `v_mul_f32 (x * log2(e))`
 * followed by `v_exp_f32`, i.e. one extra VALU op on EVERY score element of EVERY KV row.
 *
 * softmax is invariant to the base as long as it is used consistently, so fold log2(e) into the
 * score scale ONCE, at the single place the raw q.k dot is scaled, and every exp in the family
 * becomes a bare `v_exp_f32`. The (m, l) partials then live in log2 units -- which is why
 * d_flash_merge has to change with the flash kernels, not independently of them. */
#define FA_LOG2E 1.4426950408889634f
#ifndef FA_USE_EXP2
#define FA_USE_EXP2 1
#endif
#if FA_USE_EXP2
#define FA_EXP(x) __builtin_amdgcn_exp2f(x)
#define FA_SCALE(s) ((s) * FA_LOG2E)
#else
#define FA_EXP(x) __expf(x)
#define FA_SCALE(s) (s)
#endif

/* Output-dim chunk. The O accumulator is DC/32 f32x16 = 4*DC AccVGPRs per lane;
 * Q as MFMA fragments is D/2 halves = D/4 VGPRs. At D=512 an unchunked kernel
 * wants 256 AccVGPRs + 128 VGPRs, which does not fit and SPILLS TO SCRATCH
 * (measured: 1260 B/thread) — and a spilling flash inner loop is worse than any
 * amount of recompute. Chunking O at 256 caps the accumulator at 128 AccVGPRs.
 *
 * FA_DC is 128, not 256, because the workgroup is 8 waves (2 per SIMD) and a wave
 * may therefore use at most 256 registers. At FA_DC=256 the accumulator alone is
 * 128 AccVGPRs and Q-as-fragments is another 128 VGPRs at D=512 — 256 before a
 * single address register, so the whole INTERPRETER (which inlines every op and
 * is allocated for its worst case) exceeded the budget and the dispatch was
 * rejected outright with HSA_STATUS_ERROR_INVALID_ISA.
 *
 * Cost: the KV loop runs D/FA_DC times and QK^T is recomputed each pass. That is
 * paid only by the 10 global layers of the 31B, and it buys 8-wave latency hiding
 * for every other op — which is worth roughly 2x on the GEMM that dominates.
 *
 * Overridable so a STANDALONE flash kernel (no GEMM co-residency) can try FA_DC=256 at
 * 1 wave/SIMD — no QK^T recompute — to de-risk the segmented-packet carve-out. */
#ifndef FA_DC
#define FA_DC 128
#endif

/* K/V DOUBLE-BUFFER (4-wave flash object only). At 1 wave/SIMD no co-resident wave hides the
 * K/V HBM->LDS staging latency, so prefetch the NEXT tile into registers during the current
 * tile's compute and commit it at the top of the next iteration. Costs ~48 VGPR held across the
 * compute, so it is paired with PARTIAL Q-hoist (QCH=16) -- a register-budget tradeoff vs full
 * Q-hoist (D13). Toggle for the study; default off so the shipped path is full Q-hoist. */
#ifndef FA_DBUF
#define FA_DBUF 0
#endif

/* LDS: K needs the full D (it is the QK^T contraction axis); V only needs the
 * current output chunk. */
#define FA_LDS_HALVES(D)                                                        \
    (FA_BKV_OF(D) * ((D) + FA_PAD) + FA_BKV_OF(D) * (FA_DC + FA_PAD) +           \
     PLOW_WAVES * FA_BQ * FA_BKV_OF(D))

/* ---------------------------------------------------------------------------
 * Prefill.
 *
 *   Q  [n_q,  n_head,    D]     positions q0 .. q0+n_q
 *   K  [n_kv_head, kv_stride, D]   the cache, HEAD-MAJOR; positions 0 .. n_kv of each head
 *   V  [n_kv_head, kv_stride, D]
 *   O  [n_q,  n_head,    D]
 *
 * `q_pos0` is the absolute position of Q row 0 (so prefill of a continuation
 * chunk masks correctly against cached history).
 * `window == 0` means full causal.
 * ------------------------------------------------------------------------- */
template <int D, bool FP8KV = false>
__device__ void d_flash_prefill(float* __restrict__ Opart, float* __restrict__ mlpart,
                                bf16* __restrict__ O_final,
                                const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                const bf16* __restrict__ V, unsigned n_q, unsigned n_kv,
                                unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                                unsigned window, float scale, unsigned kv_stride, unsigned kv_mask,
                                unsigned nsplit, unsigned slice,
                                unsigned nblk, bf16* lds,
                                const float* __restrict__ k_scale = nullptr,
                                const float* __restrict__ v_scale = nullptr) {
    constexpr int NK = D / MFMA_K;    /* QK^T k-steps                       */
    /* KV block and the number of 32-wide MFMA N-subtiles it spans. BKV==32 => NKT==1
     * (the D=256/512 path, unchanged); BKV==64 => NKT==2 (D=128), two subtiles under
     * one softmax pass. */
    constexpr int BKV = FA_BKV_OF(D);
    constexpr int NKT = BKV / MFMA_N;
    /* Output chunk width, CAPPED AT D. FA_DC is the register-budget cap (128 at 8 waves, 256 in
     * the 4-wave flash object), but a model with head_dim < FA_DC (Llama/Qwen: D=128) must not
     * chunk wider than the head — otherwise NCH = D/FA_DC rounds to 0 and the output loop never
     * runs. DCH = min(FA_DC, D) makes NCH >= 1 for any D and is a no-op when D >= FA_DC. */
    constexpr int DCH = (FA_DC < D) ? FA_DC : D;
    constexpr int NDT = DCH / MFMA_N; /* output d-tiles per chunk           */
    constexpr int NCH = D / DCH;      /* output chunks: 1 at D<=FA_DC, 2 at D=512 */
    constexpr int STRIDE = D + FA_PAD;
    /* V stays [kv][d] in LDS, and the 8 x ds_read_u16 in the PV loop below stays too.
     *
     * It LOOKS like the textbook bug -- the MFMA's B operand wants eight consecutive kv for one
     * d, and stored this way they are VSTRIDE apart, so the inner loop reads them one at a time:
     * 128 ds_read_u16 per thread. Storing V transposed turns that into one ds_read_b128 and was
     * tried twice. Both LOST (prefill 771 -> 853, then -> 841 ms even with a conflict-free write).
     *
     * The reason: `accn` is lane % 32, so those 2-byte reads have LANES 0-31 ON CONSECUTIVE
     * COLUMNS -- each one is already a clean, conflict-free 128-byte LDS transaction. It moves
     * exactly the same bytes as the b128; it only costs more ISSUE SLOTS, and LDS BANDWIDTH is
     * the limit here, not issue. Meanwhile the transposed WRITE is a genuine 32-way bank
     * conflict that no padding escapes (the b128 read needs VSTRIDE % 8 == 0, and every such
     * stride puts all 32 lanes on bank 0), and fixing THAT by re-mapping threads costs the
     * coalesced HBM read of V.
     *
     * "2-byte access = bad" is a good heuristic and it is wrong here. Measure. */
    constexpr int VSTRIDE = DCH + FA_PAD;

    bf16* Ksm = lds;
    bf16* Vsm = lds + BKV * STRIDE;
    bf16* Psm = lds + BKV * STRIDE + BKV * VSTRIDE; /* [PLOW_WAVES][BQ][BKV] */

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned frow = mfma_frag_row(lane); /* l % 32 */
    const unsigned accn = mfma_acc_n(lane);    /* l % 32 */

    /* SPLIT-KV, and it is what makes prefill usable on 256 CUs.
     *
     * One workgroup per (q-tile, head) gives only
     *   ceil(n_q / (PLOW_WAVES*FA_BQ)) * n_head
     * work units -- at Gemma-31B with n_q=512 that is 2 * 32 = 64, so 192 of 256 CUs sit
     * idle while the 64 that are busy each walk the whole causal triangle. Measured on the
     * real network: flash was HALF the total runtime at a mean of 66 us of work per
     * workgroup spread over a 1700 us span. That is not slow code, it is an empty machine.
     *
     * So each (q-tile, head) is split `nsplit` ways along KV. Every split emits an
     * UNNORMALIZED partial (o, m, l) and d_flash_merge combines them with the standard
     * online-softmax rescale -- exactly the machinery the decode path already used. The
     * split is over the q-tile's OWN causal/window-valid KV range, not over [0, n_kv), so
     * the splits within a tile are balanced rather than mostly-masked. */
    const unsigned q_tiles = (n_q + PLOW_WAVES * FA_BQ - 1) / (PLOW_WAVES * FA_BQ);
    const unsigned n_work = q_tiles * n_head * nsplit;
    const unsigned gqa = n_head / n_kv_head;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned h = (w / nsplit) % n_head;
        const unsigned qt = w / (nsplit * n_head);
        const unsigned hkv = h / gqa;
        const unsigned q_base = qt * PLOW_WAVES * FA_BQ; /* first row of this q-tile */
        const unsigned my_q0 = q_base + wave * FA_BQ; /* first row of THIS wave     */

        /* Q fragments. Fragment layout: lane supplies m = l%32 and the k-octet
         * 8*(l/32) at each k-step, so one wave's Q tile is
         *   FA_BQ(32) rows * D dims / 64 lanes / 8 halves = D/16 fragments,
         * i.e. exactly D/4 VGPRs. At D=256 that is 64 VGPRs and we hoist the whole
         * thing out of the KV loop, which is free.
         *
         * At D=512 it is 128 VGPRs, and that single array is what pushes the
         * INTERPRETER over the 256-register limit that 2 waves/SIMD imposes -- i.e.
         * it would force the whole persistent kernel, GEMM included, down to 4
         * waves. No choice of MFMA shape avoids it: 16x16x32 needs the same 32
         * fragments, because it is the same data. It is 128 VGPRs of Q, or nothing.
         *
         * So at D>256 we hold only QCH fragments and re-read Q per KV tile. Q is a
         * 32 KiB per-wave tile and stays L2-resident, and only 10 of the 60 layers
         * are D=512 (the rest are sliding, D=256) -- while the GEMM those 8 waves
         * buy back is ~96% of prefill FLOPs. The trade is heavily in our favour. */
        /* Q fragments live at once. At 4 waves (the flash-only segmented object, 512-reg budget)
         * hoist ALL of them: no co-resident wave hides the per-KV-tile Q re-read, and the registers
         * are there once the GEMM is not co-resident. At 8 waves the 256-reg cap forces QCH=4. Under
         * FA_DBUF the K/V prefetch needs ~48 VGPR, so drop to partial hoist (16) to make room. */
        /* QCH must divide NK (assert below), and D=128 (Llama/Qwen) has NK=8 — below both the
         * 4-wave partial-hoist depth (16) and, at register budget, the 8-wave depth (4). Cap for both:
         *   4-wave flash object: FA_DBUF partial hoist is min(16,NK) so d_flash_prefill<128> COMPILES
         *     (QCH=16 was 16%8=8 != 0 -> static_assert failure; this is why build_gfx950.sh's flash
         *     object build aborts on any interp that instantiates <128>). D>=256 keeps 16.
         *   8-wave inline flash (PLOW_FLASH_HD128, D=128): QCH=4 costs 16 VGPR of hoisted Q and pushes
         *     the co-resident GEMM+flash interpreter to 260 regs -> 1 wave/SIMD -> INVALID_ISA under
         *     waves_per_eu(2,2). QCH=2 (still | 8) trims 8 VGPR to land at occ=2; the only cost is
         *     re-reading the L2-resident Q tile twice as often, negligible at D=128. D>=256 keeps 4. */
        constexpr int QCH = (PLOW_WAVES <= 4) ? (FA_DBUF ? (NK < 16 ? NK : 16) : NK)
                                              : (NK <= 8 ? 2 : 4);
        constexpr bool QHOIST = (QCH == NK);
        static_assert(NK % QCH == 0, "Q fragment chunk must divide the k-steps");

#define FA_LOAD_QF(ST0)                                                                  \
    _Pragma("unroll") for (int _i = 0; _i < QCH; _i++) {                                  \
        const unsigned qrow = my_q0 + frow;                                               \
        const unsigned d0 = mfma_frag_k(lane, ((ST0) + _i) * MFMA_K);                     \
        bf16 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};                                             \
        if (qrow < n_q) __builtin_memcpy(t, &Q[((size_t)qrow * n_head + h) * D + d0], 16);\
        _Pragma("unroll") for (int j = 0; j < 8; j++) {                                    \
            bf16_t v;                                                                     \
            __builtin_memcpy(&v, &t[j], 2);                                               \
            qf[_i][j] = v;                                                                \
        }                                                                                 \
    }

        bf16x8 qf[QCH];
        if constexpr (QHOIST) { FA_LOAD_QF(0); }

        /* KV loop bounds MUST be uniform across the whole workgroup, not per-wave:
         * the loop body contains __syncthreads(), so if the four waves computed
         * different trip counts (each wave owns different query rows, hence a
         * different causal horizon) they would hit different numbers of barriers.
         * That is undefined behaviour, and in practice it silently corrupts the
         * output rather than hanging. Bound by the LAST row of the whole q-tile
         * and let the per-element mask below do the exact work. */
        const unsigned q_tile_last = q_pos0 + q_base + PLOW_WAVES * FA_BQ - 1;
        const unsigned kv_end = (q_tile_last + 1 < n_kv) ? (q_tile_last + 1) : n_kv;
        /* Likewise the sliding-window skip: a tile may only be skipped if it is
         * outside the window for EVERY row in the workgroup, i.e. for the
         * earliest query row. */
        const unsigned q_tile_first = q_pos0 + q_base;
        const unsigned win_lo =
            (window && q_tile_first >= window) ? (q_tile_first - window + 1) : 0;

        /* Carve this split's share out of [kv_lo, kv_end), in whole FA_BKV tiles. An empty
         * share is fine and normal (a sliding layer's early splits): it writes m = -inf,
         * l = 0 and the merge drops it. */
        const unsigned kv_lo = (win_lo / BKV) * BKV;
        const unsigned tiles_kv =
            (kv_end > kv_lo) ? ((kv_end - kv_lo + BKV - 1) / BKV) : 0u;
        const unsigned per = (tiles_kv + nsplit - 1) / nsplit;
        const unsigned my_lo = kv_lo + sp * per * BKV;
        unsigned my_hi = kv_lo + (sp + 1) * per * BKV;
        if (my_hi > kv_end) my_hi = kv_end;

        /* Output-chunk loop. NCH == 1 at D=256 (no recompute at all); NCH == 2 at
         * D=512, which halves the accumulator and keeps the kernel in registers. */
        for (int ch = 0; ch < NCH; ch++) {
            const unsigned d_off = ch * DCH;

            f32x16 oacc[NDT];
#pragma unroll
            for (int t = 0; t < NDT; t++) oacc[t] = (f32x16)(0.0f);
            float m_st[16], l_st[16];
#pragma unroll
            for (int i = 0; i < 16; i++) { m_st[i] = FA_NEG_INF; l_st[i] = 0.0f; }

#if FA_DBUF
            /* Register-held double buffer: nk/nv hold the PREFETCHED next tile across the compute,
             * committed to LDS at the top of the next iteration while the next tile's loads fly.
             * The window-skip `continue` is dropped -- my_lo already starts at the window and the
             * per-element softmax mask excludes any out-of-window rows. */
            constexpr int KPT = BKV * D / PLOW_THREADS / 8;
            constexpr int VPT = BKV * FA_DC / PLOW_THREADS / 8;
            bf16v8 nk[KPT], nv[VPT];
#define FA_DB_LOAD(KVB)                                                                          \
    _Pragma("unroll") for (int it = 0; it < KPT; it++) {                                         \
        const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                            \
        const unsigned kv = (KVB) + e / D;                                                       \
        nk[it] = (kv < n_kv) ? ld_glob8(as_glob(K) + ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + e % D) \
                             : bf16v8_zero();                                                     \
    }                                                                                            \
    _Pragma("unroll") for (int it = 0; it < VPT; it++) {                                         \
        const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                            \
        const unsigned kv = (KVB) + e / FA_DC;                                                   \
        nv[it] = (kv < n_kv) ? ld_glob8(as_glob(V) + ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + d_off + e % FA_DC) \
                             : bf16v8_zero();                                                     \
    }
            FA_DB_LOAD(my_lo); /* prime */
            for (unsigned kv0 = my_lo; kv0 < my_hi; kv0 += BKV) {
                __syncthreads(); /* previous tile's Ksm/Vsm reads are done */
#pragma unroll
                for (int it = 0; it < KPT; it++) {
                    const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);
                    __builtin_memcpy(&Ksm[(e / D) * STRIDE + e % D], &nk[it], 16);
                }
#pragma unroll
                for (int it = 0; it < VPT; it++) {
                    const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);
                    __builtin_memcpy(&Vsm[(e / FA_DC) * VSTRIDE + e % FA_DC], &nv[it], 16);
                }
                if (kv0 + BKV < my_hi) FA_DB_LOAD(kv0 + BKV); /* prefetch next; loads fly during compute */
                __syncthreads(); /* Ksm/Vsm visible */
#else
            for (unsigned kv0 = my_lo; kv0 < my_hi; kv0 += BKV) {
                /* Skip KV tiles wholly outside the sliding window. Uniform across
                 * the workgroup (see win_lo). At window=1024 and seq=4096 this
                 * drops ~3/4 of the tiles. */
                if (window && kv0 + BKV <= win_lo) continue;
                __syncthreads();
                /* K: full D (the contraction axis). V: only this output chunk.
                 * FP8 KV: the cache is uint8 e4m3 with a per-row f32 scale. DEQUANTIZE at the LDS
                 * stage — fp8 -> bf16 (b64, HALF the HBM bytes) times the row scale — so the MFMA
                 * below reads bf16 from Ksm/Vsm exactly as the bf16 path and is BYTE-IDENTICAL. */
                for (unsigned e = threadIdx.x * 8; e < BKV * D; e += PLOW_THREADS * 8) {
                    const unsigned r = e / D, c = e % D;
                    const unsigned kv = kv0 + r;
                    bf16 tk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (kv < n_kv) {
                        const size_t off = ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + c;
                        if constexpr (FP8KV) {
                            const bf16v8 dv = fp8v8_to_bf16v8(
                                ld_glob_fp8v8((const unsigned char*)K + off));
                            const float ks = k_scale[(size_t)hkv * kv_stride + (kv & kv_mask)];
#pragma unroll
                            for (int j = 0; j < 8; j++) tk[j] = f2bf(bf2f(dv[j]) * ks);
                        } else {
                            __builtin_memcpy(tk, &K[off], 16);
                        }
                    }
                    __builtin_memcpy(&Ksm[r * STRIDE + c], tk, 16);
                }
                for (unsigned e = threadIdx.x * 8; e < BKV * DCH; e += PLOW_THREADS * 8) {
                    const unsigned r = e / DCH, c = e % DCH;
                    const unsigned kv = kv0 + r;
                    bf16 tv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (kv < n_kv) {
                        const size_t off =
                            ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + d_off + c;
                        if constexpr (FP8KV) {
                            const bf16v8 dv = fp8v8_to_bf16v8(
                                ld_glob_fp8v8((const unsigned char*)V + off));
                            const float vs = v_scale[(size_t)hkv * kv_stride + (kv & kv_mask)];
#pragma unroll
                            for (int j = 0; j < 8; j++) tv[j] = f2bf(bf2f(dv[j]) * vs);
                        } else {
                            __builtin_memcpy(tv, &V[off], 16);
                        }
                    }
                    __builtin_memcpy(&Vsm[r * VSTRIDE + c], tv, 16);
                }
                __syncthreads();
#endif

                /* Two code paths, chosen at compile time by the number of MFMA N-subtiles
                 * the KV block spans. NKT==1 (D=256/512, BKV=32) is the ORIGINAL single-
                 * accumulator kernel, kept BYTE-IDENTICAL: the 8-wave Gemma prefill
                 * interpreter sits exactly on the 256-reg / 2-waves-per-SIMD cliff, and even
                 * an array-of-one `s[1]`/`p[1][16]` perturbs the allocator enough to drop it
                 * to 1 wave/SIMD. NKT==2 (D=128, BKV=64) runs the two subtiles under one
                 * shared softmax pass. */
                if constexpr (NKT == 1) {
                    /* S = Q . K^T  ->  acc layout: n = kv = l%32, m = mfma_acc_m(l,i).
                     * Accumulated over D in chunks of QCH fragments. */
                    f32x16 s = (f32x16)(0.0f);
                    /* unroll 1 is LOAD-BEARING when !QHOIST: fully unrolling this loop
                     * lets the compiler hoist all NK Q loads to the top, which puts all
                     * NK fragments live at once and undoes the chunking entirely. */
#pragma unroll 1
                    for (int c = 0; c < NK; c += QCH) {
                        if constexpr (!QHOIST) { FA_LOAD_QF(c); }
#pragma unroll
                        for (int q = 0; q < QCH; q++) {
                            bf16x8 kfrag;
                            const unsigned d0 = mfma_frag_k(lane, (c + q) * MFMA_K);
                            bf16 t[8];
                            __builtin_memcpy(t, &Ksm[frow * STRIDE + d0], 16);
#pragma unroll
                            for (int j = 0; j < 8; j++) {
                                bf16_t v;
                                __builtin_memcpy(&v, &t[j], 2);
                                kfrag[j] = v;
                            }
                            s = __builtin_amdgcn_mfma_f32_32x32x16_bf16(qf[q], kfrag, s, 0, 0, 0);
                        }
                    }

                    /* Mask, then online softmax. A row of S lives entirely inside ONE
                     * HALF of the wave (lanes 0-31 or 32-63), so the row reductions
                     * must stop at 32 lanes — a full 64-lane reduce folds two rows
                     * together and silently corrupts the softmax. */
                    float p[16];
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        const unsigned qi = my_q0 + mfma_acc_m(lane, i);
                        const unsigned qg = q_pos0 + qi;
                        const unsigned kg = kv0 + accn;
                        /* sliding window is INCLUSIVE of the current token:
                         * keep iff 0 <= qg - kg <= window-1 */
                        const bool valid = (qi < n_q) && (kg < n_kv) && (kg <= qg) &&
                                           (!window || (qg - kg) < window);
                        p[i] = valid ? (s[i] * FA_SCALE(scale)) : FA_NEG_INF;
                    }
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        const float rmax = half_wave_max(p[i]);
                        const float mnew = fmaxf(m_st[i], rmax);
                        const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                        const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
                        l_st[i] = l_st[i] * corr + half_wave_sum(pe);
                        m_st[i] = mnew;
                        p[i] = pe;
#pragma unroll
                        for (int t = 0; t < NDT; t++) oacc[t][i] *= corr;
                    }

                    /* P sits in the accumulator layout but MFMA needs it as an A operand
                     * (m = l%32, k = kv). Transpose through LDS — it is only 32x32. */
                    bf16* myP = Psm + wave * FA_BQ * BKV;
                    __syncthreads();
#pragma unroll
                    for (int i = 0; i < 16; i++)
                        myP[mfma_acc_m(lane, i) * BKV + accn] = f2bf(p[i]);
                    __syncthreads();

                    /* O += P . V */
#pragma unroll
                    for (int t = 0; t < NDT; t++) {
#pragma unroll
                        for (int ks = 0; ks < BKV; ks += MFMA_K) {
                            const unsigned kk = mfma_frag_k(lane, ks);
                            bf16x8 pf, vf;
                            bf16 tp[8];
                            __builtin_memcpy(tp, &myP[frow * BKV + kk], 16);
#pragma unroll
                            for (int j = 0; j < 8; j++) {
                                bf16_t v;
                                __builtin_memcpy(&v, &tp[j], 2);
                                pf[j] = v;
                                /* Strided by VSTRIDE, so 8 x u16 rather than a b128 -- and that
                                 * is FINE: lanes 0-31 sit on consecutive columns, so each read
                                 * is a full 128-byte LDS transaction. See the VSTRIDE comment. */
                                bf16 vv = Vsm[(kk + j) * VSTRIDE + t * MFMA_N + accn];
                                bf16_t vb;
                                __builtin_memcpy(&vb, &vv, 2);
                                vf[j] = vb;
                            }
                            oacc[t] =
                                __builtin_amdgcn_mfma_f32_32x32x16_bf16(pf, vf, oacc[t], 0, 0, 0);
                        }
                    }
                } else {
                    /* S = Q . K^T over NKT column-subtiles, each one 32x32 MFMA N-tile
                     * (n = kv-within-subtile = l%32, m = mfma_acc_m(l,i)). The subtiles
                     * cover the BKV staged rows and SHARE the softmax pass below. */
                    f32x16 s[NKT];
#pragma unroll
                    for (int n = 0; n < NKT; n++) s[n] = (f32x16)(0.0f);
#pragma unroll 1
                    for (int c = 0; c < NK; c += QCH) {
                        if constexpr (!QHOIST) { FA_LOAD_QF(c); }
#pragma unroll
                        for (int q = 0; q < QCH; q++) {
                            const unsigned d0 = mfma_frag_k(lane, (c + q) * MFMA_K);
#pragma unroll
                            for (int n = 0; n < NKT; n++) {
                                bf16x8 kfrag;
                                bf16 t[8];
                                __builtin_memcpy(t, &Ksm[(n * MFMA_N + frow) * STRIDE + d0], 16);
#pragma unroll
                                for (int j = 0; j < 8; j++) {
                                    bf16_t v;
                                    __builtin_memcpy(&v, &t[j], 2);
                                    kfrag[j] = v;
                                }
                                s[n] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(qf[q], kfrag, s[n], 0, 0, 0);
                            }
                        }
                    }

                    /* Mask + scale, per subtile. Row reductions stop at 32 lanes (a row
                     * lives in one half-wave). */
                    float p[NKT][16];
#pragma unroll
                    for (int n = 0; n < NKT; n++) {
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const unsigned qi = my_q0 + mfma_acc_m(lane, i);
                            const unsigned qg = q_pos0 + qi;
                            const unsigned kg = kv0 + n * MFMA_N + accn;
                            /* sliding window INCLUSIVE: keep iff 0 <= qg - kg <= window-1 */
                            const bool valid = (qi < n_q) && (kg < n_kv) && (kg <= qg) &&
                                               (!window || (qg - kg) < window);
                            p[n][i] = valid ? (s[n][i] * FA_SCALE(scale)) : FA_NEG_INF;
                        }
                    }
                    /* Online softmax across ALL subtiles at once: one max, one exp per
                     * element, ONE rescale of the O accumulator per BKV rows. */
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        float rmax = FA_NEG_INF;
#pragma unroll
                        for (int n = 0; n < NKT; n++) rmax = fmaxf(rmax, half_wave_max(p[n][i]));
                        const float mnew = fmaxf(m_st[i], rmax);
                        const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                        float sum = 0.0f;
#pragma unroll
                        for (int n = 0; n < NKT; n++) {
                            const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[n][i] - mnew);
                            p[n][i] = pe;
                            sum += half_wave_sum(pe);
                        }
                        l_st[i] = l_st[i] * corr + sum;
                        m_st[i] = mnew;
#pragma unroll
                        for (int t = 0; t < NDT; t++) oacc[t][i] *= corr;
                    }

                    /* P -> LDS as MFMA A operand (m = l%32, k = kv), the whole BKV-wide P
                     * in one barrier pair — both subtiles written to adjacent columns. */
                    bf16* myP = Psm + wave * FA_BQ * BKV;
                    __syncthreads();
#pragma unroll
                    for (int n = 0; n < NKT; n++)
#pragma unroll
                        for (int i = 0; i < 16; i++)
                            myP[mfma_acc_m(lane, i) * BKV + n * MFMA_N + accn] = f2bf(p[n][i]);
                    __syncthreads();

                    /* O += P . V, contracting over all BKV staged rows. */
#pragma unroll
                    for (int t = 0; t < NDT; t++) {
#pragma unroll
                        for (int ks = 0; ks < BKV; ks += MFMA_K) {
                            const unsigned kk = mfma_frag_k(lane, ks);
                            bf16x8 pf, vf;
                            bf16 tp[8];
                            __builtin_memcpy(tp, &myP[frow * BKV + kk], 16);
#pragma unroll
                            for (int j = 0; j < 8; j++) {
                                bf16_t v;
                                __builtin_memcpy(&v, &tp[j], 2);
                                pf[j] = v;
                                bf16 vv = Vsm[(kk + j) * VSTRIDE + t * MFMA_N + accn];
                                bf16_t vb;
                                __builtin_memcpy(&vb, &vv, 2);
                                vf[j] = vb;
                            }
                            oacc[t] =
                                __builtin_amdgcn_mfma_f32_32x32x16_bf16(pf, vf, oacc[t], 0, 0, 0);
                        }
                    }
                }
            }

#if FA_DBUF
#undef FA_DB_LOAD
#endif

            /* nsplit==1: there is nothing for d_flash_merge to combine, so normalize in
             * place and write the FINAL bf16 output straight to n.at, skipping the f32
             * partial round-trip through HBM entirely. The single split covers the whole
             * causal/window range, so l_st IS the final normalizer and no exp-rescale is
             * needed. The l>0 guard reproduces merge's empty-range behaviour (write 0).
             * This is bit-identical to running d_flash_merge with nsplit==1. Gated on a
             * non-null O_final so the golden wrappers (which pass null and validate the
             * partial+merge path even at nsplit==1) still exercise the old path.
             *
             * Compiled into the standalone flash object (PLOW_BUCKET_FLASH, Gemma) AND into the
             * D=128 inline path (PLOW_FLASH_HD128, Llama/Qwen): both actually RUN flash and so
             * both need the fused write. It is kept OUT of the 8-wave Gemma prefill interpreter
             * (neither macro) where the D=256/512 arms are compiled-but-never-run: adding the
             * fused write there pushes it 258 > 256 over the cliff for no benefit. */
#if defined(PLOW_BUCKET_FLASH) || defined(PLOW_FLASH_HD128)
            if (nsplit == 1 && O_final != nullptr) {
                const unsigned qd = n_head * D; /* row stride of n.at */
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const unsigned qi = my_q0 + mfma_acc_m(lane, i);
                    if (qi >= n_q) continue;
                    const float inv = (l_st[i] > 0.0f) ? (1.0f / l_st[i]) : 0.0f;
                    /* One base pointer per row (as the partial path does), so the epilogue's
                     * 64-bit address math stays out of the unrolled store loop and does not
                     * blow up register pressure / scratch for the hot KV loop. */
                    bf16* orow = O_final + ((size_t)qi * qd + h * D + d_off + accn);
#pragma unroll
                    for (int t = 0; t < NDT; t++) {
                        /* Branchless RNE f32->bf16. A softmax-normalized attention output is
                         * always finite, so the NaN/Inf guard in f2bf() is dead weight -- and
                         * its branch, replicated across the fully-unrolled 128-element epilogue,
                         * wrecks the register allocator (scratch 520 -> 1936 B/lane, spilling
                         * the hot KV loop). Drop the guard here. */
                        float fv = oacc[t][i] * inv;
                        unsigned u;
                        __builtin_memcpy(&u, &fv, 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        orow[t * MFMA_N] = (bf16)(u >> 16);
                    }
                }
                continue;
            }
#endif
            /* Store the UNNORMALIZED partial. d_flash_merge does the rescale, so the
             * division by l must NOT happen here -- each split has its own l and they are
             * only comparable after the global max is known. */
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned qi = my_q0 + mfma_acc_m(lane, i);
                if (qi >= n_q) continue;
                float* op = Opart + ((size_t)(qi * n_head + h) * nsplit + sp) * D;
#pragma unroll
                for (int t = 0; t < NDT; t++) op[d_off + t * MFMA_N + accn] = oacc[t][i];
                /* 32 lanes share this qi (they hold different d), so exactly one writes
                 * (m, l) -- and only once, not once per output chunk. */
                if (ch == 0 && accn == 0) {
                    float* ml = mlpart + ((size_t)(qi * n_head + h) * nsplit + sp) * 2;
                    ml[0] = m_st[i];
                    ml[1] = l_st[i];
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Decode: one query row per (batch, head). MFMA is the wrong tool here — a 32x32
 * matrix core would run with 1 of 32 M-lanes live — so this is a plain dot
 * product, and it is bandwidth-bound on the KV cache.
 *
 * Split-KV: at batch=1 there are only n_head workgroups of work (32 for the 31B),
 * which would leave 224 of 256 CUs idle. Each (b, h) is therefore split `nsplit`
 * ways over the KV axis; each split emits a partial (o, m, l) and d_flash_merge
 * combines them with the usual online-softmax rescale.
 *
 *   Opart [b][h][split][D]  f32
 *   mlpart[b][h][split][2]  f32   (running max, running sum)
 * ------------------------------------------------------------------------- */
#define FA_DEC_TILE PLOW_THREADS /* KV rows per pass: one per thread */

/* The LDS d_flash_decode needs, in floats: [scores | block-reduction scratch | the query row].
 *
 * This macro exists because every caller used to size the arena itself, and when the query row
 * was added to the layout the golden wrapper's __shared__ array silently became too small --
 * the kernel wrote past the end of it and the decode tests failed while the real model, whose
 * arena is a 144 KB union, kept working. An op must own its own LDS footprint. */
/* Threads that cover one V row with 16-byte loads, and how many ROW-GROUPS that leaves. */
#define FA_DEC_NDT(D) ((D) / 8)                        /* 32 at D=256, 64 at D=512 */
#define FA_DEC_NG(D) (PLOW_THREADS / FA_DEC_NDT(D))    /* 16 at D=256,  8 at D=512 */

/* [ scores | block-reduction scratch | query row | cross-group O partials ]
 *
 * This macro exists because every caller used to size the arena itself, and when the query row
 * was added the golden wrapper's __shared__ array silently became too small -- the kernel wrote
 * past the end of it and only the TEST failed, because the real model's arena is a 144 KB
 * union. An op must own its own LDS footprint. */
/* ---------------------------------------------------------------------------
 * GQA FUSION. The KV cache is the ONLY tensor in the network with un-exploited reuse.
 *
 * A work item used to be (query_head, split), so each KV row was streamed once PER QUERY HEAD
 * that shares it — GQA times. On Gemma-4 31B:
 *
 *     sliding (50 layers)  32 heads / 16 kv  ->  GQA 2:1
 *     full    (10 layers)  32 heads /  4 kv  ->  GQA 8:1   <-- each row read EIGHT times
 *
 * The full layers are 10 of 60 but 57% of all KV traffic, because they hold the whole context
 * (3326 rows at hd=512) where sliding layers hold a 1024 window. Measured, per token:
 *
 *     KV bytes   today 3.86 GB      GQA-fused 1.11 GB      3.5x less
 *     flash_dec  1.91 ms @ 2.0 TB/s -> floor 0.55 ms       ~1.36 ms
 *
 * A work item is now (kv_head, split), and the workgroup carries all GF = GQA query heads that
 * share that KV head. The K row is loaded ONCE and dotted against GF query rows; the V row is
 * loaded ONCE and accumulated into GF outputs. This is NOT a prefetch — it does not move bytes
 * earlier, it moves FEWER bytes, which is the one category the prefetch autopsies (§11-§12) left
 * standing.
 *
 * The price is parallelism: n_work drops from n_head*nsplit to n_kv_head*nsplit, so plowc must
 * raise nsplit by GF to keep the machine full. PLOW_FA_GF (dev_isa.h) is where the two agree.
 *
 * The softmax is now GF-way. The per-head reductions run on DIFFERENT WAVES (wave g reduces head
 * g), so all GF of them happen concurrently and the tile still costs 3 barriers, not 3*GF.
 * ------------------------------------------------------------------------- */
#define FA_DEC_LDS_FLOATS(D, GF) \
    ((GF) * FA_DEC_TILE + 2 * PLOW_WAVES + (GF) * ((D) / 2) + FA_DEC_NG(D) * (D))

/* V rows in flight. A fused row feeds GF accumulators, so there is 8x more arithmetic per load
 * to hide its latency with — and 8 accumulators of 8 floats is already 64 VGPRs. Trade depth for
 * registers as GF grows, or the decode kernel walks off the 256-register cliff. */
#define FA_DEC_VU(GF) ((GF) >= 8 ? 2 : ((GF) >= 4 ? 4 : FA_DEC_V_UNROLL))

/* V-stream pipelining: prefetch the first V-group over the softmax barriers (see the loop). Off by
 * default so the shipped path is byte-identical; the build script turns it on for the A/B study. */
#ifndef FA_DEC_VPIPE
#define FA_DEC_VPIPE 0
#endif

#undef FA_LOAD_QF

template <int D, int GF, bool FP8KV = false>
__device__ void d_flash_decode(float* __restrict__ Opart, float* __restrict__ mlpart,
                               const bf16* __restrict__ Q, const bf16* __restrict__ K,
                               const bf16* __restrict__ V, const int* __restrict__ kv_len,
                               unsigned n_batch, unsigned n_head, unsigned n_kv_head,
                               unsigned kv_stride, unsigned window, float scale, unsigned nsplit,
                               unsigned kv_mask, unsigned slice, unsigned nblk, float* lds,
                               const float* __restrict__ k_scale = nullptr,
                               const float* __restrict__ v_scale = nullptr) {
    /* A work item carries GF CONSECUTIVE query heads. They share a KV head as long as GF divides
     * GQA — which is the only thing this kernel needs, and is weaker than GF == GQA. On Gemma the
     * two coincide (hd 256 -> GQA 2 -> GF 2; hd 512 -> GQA 8 -> GF 8), but a true-MQA model has
     * GQA = n_head, and then this simply forms n_head/GF groups per KV head and still reads each
     * row once per group. Indexing by head-group rather than by kv_head is what makes that work;
     * the earlier form silently computed only the first GF heads of each KV head. */
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_grp = n_head / GF; /* head-groups; == n_kv_head when GF == gqa */
    const unsigned n_work = n_batch * n_grp * nsplit;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 6, lane = tid & 63;

    float* Ssm = lds;                                   /* [GF][FA_DEC_TILE] scores        */
    float* hmax = lds + GF * FA_DEC_TILE;               /* [PLOW_WAVES] per-head tile max   */
    float* hsum = hmax + PLOW_WAVES;                    /* [PLOW_WAVES] per-head tile sum   */
    bf16* qsm = (bf16*)(hsum + PLOW_WAVES);             /* [GF][D] query rows, staged once  */
    float* osm = (float*)(qsm + GF * D);                /* [NG][D] O, per row-group         */

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned hg = (w / nsplit) % n_grp;
        const unsigned b = w / (nsplit * n_grp);
        const unsigned h0 = hg * GF;   /* the GF consecutive query heads this item carries */
        const unsigned hkv = h0 / gqa; /* they all share this KV head (GF divides gqa) */

        const unsigned len = (unsigned)kv_len[b];
        const unsigned qpos = len - 1; /* the query token is the newest one */

        /* THIS SPLIT'S KV RANGE — CLAMPED TO THE WINDOW.
         *
         * The range used to be split over [0, len) and rows outside the window were then MASKED
         * inside the loop. That is correct and it is O(ctx): at ctx=128k a SLIDING layer walked
         * 128000 rows — 250 tile iterations, each with its softmax barriers — in order to read
         * 1024. Fifty of Gemma's sixty layers are sliding, so most of the machine's flash-decode
         * time at long context was spent iterating over rows it had already decided to discard.
         *
         * A windowed layer only ever reads [len-window, len), so START there. The work then does
         * not grow with context at all, which is what a sliding window is FOR. (It is also what
         * makes the KV ring sound: those are exactly the rows the ring holds.) */
        const unsigned first = (window && len > window) ? (len - window) : 0u;
        const unsigned span = len - first;
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi = (lo + per < len) ? (lo + per) : len;

        /* HEAD-MAJOR: this head's rows are CONTIGUOUS. See dev_isa.h -- token-major put
         * n_kv_head*D of stride between consecutive rows of one head (8 KB around 512 bytes of
         * payload at Gemma-31B), so a workgroup spanned 512 KB to read 256 KB. */
        const auto* kbase = as_glob(K) + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const auto* vbase = as_glob(V) + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        /* FP8 KV: the cache is uint8[...] (1 byte/elem) with a PER-ROW f32 scale, head-major like
         * the bf16 cache. These bases are byte pointers (byte offset == element index) and the
         * per-(kv_head) scale slice; both unused on the bf16 instantiation. */
        const unsigned char* kb8 = (const unsigned char*)K + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const unsigned char* vb8 = (const unsigned char*)V + ((size_t)b * n_kv_head + hkv) * kv_stride * D;
        const float* ksc = k_scale + ((size_t)b * n_kv_head + hkv) * kv_stride;
        const float* vsc = v_scale + ((size_t)b * n_kv_head + hkv) * kv_stride;

        /* All GF query rows into LDS, once. */
        for (unsigned i = tid; i < GF * D; i += PLOW_THREADS)
            qsm[i] = Q[((size_t)b * n_head + h0 + i / D) * D + i % D];
        __syncthreads();

        float m_st[GF], l_st[GF];
#pragma unroll
        for (int g = 0; g < GF; g++) {
            m_st[g] = FA_NEG_INF;
            l_st[g] = 0.0f;
        }

        /* THE THREAD MAP, shared by both phases: this thread owns eight CONSECUTIVE d, and one
         * row out of every NG. Both K and V then read 16 bytes per lane, contiguously.
         *
         * O accumulator: this thread owns those same eight d, and a stride of KV rows.
         *
         * It used to own ONE d and every row -- so each V element was a separate 2-byte load,
         * i.e. a 128-byte request per wave where 1024 was available, in the op that streams 3 GB
         * of KV cache per token. Measured: 813 GB/s, 13% of HBM. Eight consecutive d makes it a
         * global_load_dwordx4 like everything else.
         *
         * The cost is that the rows are now split across FA_DEC_NG row-groups, so each group
         * holds a PARTIAL O and they must be summed once at the end. The online-softmax state
         * (m, l, corr) is block-wide and unaffected: it is computed from Ssm by all threads, as
         * before, and `corr` rescales every group's partial identically. */
        constexpr unsigned NDT = FA_DEC_NDT(D); /* threads covering one row  */
        constexpr unsigned NG = FA_DEC_NG(D);   /* row-groups                */
        const unsigned dbase = (tid % NDT) * 8;
        const unsigned grp = tid / NDT;
        float oacc[GF][8];
#pragma unroll
        for (int g = 0; g < GF; g++)
#pragma unroll
            for (int u = 0; u < 8; u++) oacc[g][u] = 0.0f;

        for (unsigned kv0 = lo; kv0 < hi; kv0 += FA_DEC_TILE) {
            /* SCORES: each thread streams one whole K row. Contiguous per thread, scattered
             * across the wave -- 64 lanes on 64 different rows, strided n_kv_head*D apart.
             *
             * That LOOKS like it wants the cooperative treatment the V phase got (NDT lanes per
             * row, one contiguous 1024-byte request). It was tried and it LOST: flash_decode's
             * summed work fell 505 -> 443 ms but its WALL span rose 1.97 -> 2.27 ms and the
             * token went 17.2 -> 17.5. The scatter costs nothing, because each 16-byte request
             * pulls a 128-byte line that the SAME thread then consumes across its next seven d
             * iterations -- while the cooperative form adds log2(NDT) = 5-6 shuffles per row,
             * 32 rows per tile, and makes the CUs more uneven. Leave it alone. */
            /* THE FUSION. Each thread owns kv row `kv` and loads its K row ONCE, then dots it
             * against all GF query rows out of LDS. That single change is what removes the GQA
             * re-read: the row crosses HBM once instead of GF times. */
            const unsigned kv = kv0 + tid;
            float s[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = FA_NEG_INF;
            if (kv < hi && kv <= qpos && (!window || (qpos - kv) < window)) {
                float dot[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                if constexpr (FP8KV) {
                    /* fp8: the K row is e4m3 (HALF the bytes of bf16). Load 8-wide (b64) and decode
                     * to ONE bf16v8 per step, exactly mirroring the bf16 loop's live-register
                     * footprint — a 16-wide (b128) load holds lo+hi+the fp8v16 across the dot and
                     * the allocator then spills the O accumulators to AGPRs, which starves the
                     * co-resident decode GEMV (measured: 130 vs 229 VGPR, ~1.8x slower GEMV). The
                     * per-row dequant scale multiplies the score ONCE, after the dot. */
                    const unsigned char* krow = kb8 + (size_t)(kv & kv_mask) * D;
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const bf16v8 kv8 = fp8v8_to_bf16v8(ld_glob_fp8v8(krow + d));
#pragma unroll
                        for (int g = 0; g < GF; g++)
                            dot[g] = dot8(kv8, ld_lds8(qsm + g * D + d), dot[g]);
                    }
                    const float ks = ksc[kv & kv_mask];
#pragma unroll
                    for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale) * ks;
                } else {
                    const auto* krow = kbase + (size_t)(kv & kv_mask) * D; /* sliding RING */
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const bf16v8 kv8 = ld_glob8(krow + d); /* <-- read ONCE */
#pragma unroll
                        for (int g = 0; g < GF; g++)
                            dot[g] = dot8(kv8, ld_lds8(qsm + g * D + d), dot[g]); /* used GF times */
                    }
#pragma unroll
                    for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale);
                }
            }
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = s[g];

#if FA_DEC_VPIPE
            /* PIPELINE: the KV loop is single-tile per split at the shipped nsplit=16 (span/16 ==
             * FA_DEC_TILE at 8k), so there is no "next tile" to double-buffer. The one overlap the
             * compiler cannot take is the V stream: a __syncthreads() is a hard scheduling barrier,
             * so V loads are pinned AFTER the whole softmax reduction (5 barriers, waves >= GF idle,
             * HBM drained). Issue this thread's FIRST V-group loads HERE -- their addresses do not
             * depend on the softmax result, only the multiply does -- so the HBM read of V flies
             * DURING the reduction instead of after it. Costs VU bf16v8 held across softmax. */
            constexpr int VU_ = FA_DEC_VPIPE; /* prefetch depth (rows/thread) held over softmax */
            const unsigned rmax_pf = (hi - kv0 < FA_DEC_TILE) ? (hi - kv0) : FA_DEC_TILE;
            /* vpipe_ok: the whole first VU_-group is in range (a full 512-row tile, i.e. long
             * context). Only then does the peel below consume vpre. NOTE: the load MUST stay a
             * branchless predicated load -- wrapping it in `if (vpipe_ok)` makes the allocator spill
             * vpre to ~118 AGPRs and doubles flash_decode. The per-element predicate compiles to a
             * clean set of global_load_dwordx4 with no branch, and lands at 237 VGPR/occ2. */
            const bool vpipe_ok = grp + (unsigned)(VU_ - 1) * NG < rmax_pf;
            bf16v8 vpre[VU_];
            /* VPIPE prefetches V as bf16; the fp8 cache would need a converted+scaled prefetch, which
             * is not wired here. Skip it for FP8KV so the object may still be BUILT with the bf16
             * decode object's FA_DEC_VPIPE flag (the fp8 V loads happen in the loops below). */
            if constexpr (!FP8KV) {
#pragma unroll
                for (int c = 0; c < VU_; c++) {
                    const unsigned rr = grp + (unsigned)c * NG;
                    vpre[c] = (vpipe_ok || rr < rmax_pf)
                                  ? ld_glob8(vbase + (size_t)((kv0 + rr) & kv_mask) * D + dbase)
                                  : bf16v8_zero();
                }
            }
#endif
            __syncthreads();

            /* GF softmax reductions, ONE PER WAVE, so they all run concurrently: the tile still
             * costs 3 barriers, not 3*GF. (There are PLOW_WAVES=8 waves and GF <= 8.) */
            if (wave < GF) {
                float mx = FA_NEG_INF;
                for (unsigned i = lane; i < FA_DEC_TILE; i += 64)
                    mx = fmaxf(mx, Ssm[wave * FA_DEC_TILE + i]);
#pragma unroll
                for (int off = 32; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
                if (lane == 0) hmax[wave] = mx;
            }
            __syncthreads();

            float mnew[GF], corr[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) {
                mnew[g] = fmaxf(m_st[g], hmax[g]);
                corr[g] = (m_st[g] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[g] - mnew[g]);
            }
            float pe[GF];
#pragma unroll
            for (int g = 0; g < GF; g++)
                pe[g] = (mnew[g] == FA_NEG_INF || s[g] == FA_NEG_INF)
                            ? 0.0f
                            : FA_EXP(s[g] - mnew[g]);
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = pe[g];
            __syncthreads();

            if (wave < GF) {
                float sm = 0.0f;
                for (unsigned i = lane; i < FA_DEC_TILE; i += 64)
                    sm += Ssm[wave * FA_DEC_TILE + i];
                sm = wave_sum(sm);
                if (lane == 0) hsum[wave] = sm;
            }
            __syncthreads();

#pragma unroll
            for (int g = 0; g < GF; g++) {
                l_st[g] = l_st[g] * corr[g] + hsum[g];
                m_st[g] = mnew[g];
            }

            /* o[dbase..dbase+8) += sum_r p[r] * V[r][dbase..dbase+8), over this group's rows.
             * One 16-byte global load per row per thread, FA_DEC_V_UNROLL rows in flight.
             *
             * `if (pw != 0)` is gone too: it made every V load conditional, so the compiler
             * could not batch them at all. Multiplying by an exact zero costs one FMA. */
#pragma unroll
            for (int g = 0; g < GF; g++)
#pragma unroll
                for (int u = 0; u < 8; u++) oacc[g][u] *= corr[g];

            /* The other half of the fusion: the V row is loaded ONCE and accumulated into all GF
             * outputs. Same bytes off HBM, GF times the arithmetic — which is also why the unroll
             * can shrink as GF grows (FA_DEC_VU): there is far more work per load to hide it. */
            constexpr int VU = FA_DEC_VU(GF);
            const unsigned rmax = (hi - kv0 < FA_DEC_TILE) ? (hi - kv0) : FA_DEC_TILE;
            unsigned r = grp;
#if FA_DEC_VPIPE
            /* First VU-group was PREFETCHED over the softmax barriers (vpre). Consume it here
             * instead of re-loading; the loads already flew during the reduction. Only when the
             * whole first group was in range (vpipe_ok) -- otherwise fall through to the loops
             * below, which re-load with the proper bounds. */
            if constexpr (!FP8KV) if (vpipe_ok) {
#pragma unroll
                for (int c = 0; c < FA_DEC_VPIPE; c++) {
#pragma unroll
                    for (int g = 0; g < GF; g++) {
                        const float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG];
#pragma unroll
                        for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(vpre[c][u]);
                    }
                }
                r += FA_DEC_VPIPE * NG;
            }
#endif
            for (; r + (VU - 1) * NG < rmax; r += VU * NG) {
                bf16v8 vv[VU];
                float vsf[VU];
#pragma unroll
                for (int c = 0; c < VU; c++) {
                    const unsigned rr = (kv0 + r + (unsigned)c * NG) & kv_mask;
                    if constexpr (FP8KV) {
                        /* fp8: HALF the bytes — this lane reads only its 8 owned head-dims (b64),
                         * decodes to bf16, and folds the per-row dequant scale into pw. */
                        vv[c] = fp8v8_to_bf16v8(ld_glob_fp8v8(vb8 + (size_t)rr * D + dbase));
                        vsf[c] = vsc[rr];
                    } else {
                        vv[c] = ld_glob8(vbase + (size_t)rr * D + dbase);
                        vsf[c] = 1.0f;
                    }
                }
#pragma unroll
                for (int c = 0; c < VU; c++) {
#pragma unroll
                    for (int g = 0; g < GF; g++) {
                        const float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG] * vsf[c];
#pragma unroll
                        for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(vv[c][u]);
                    }
                }
            }
            for (; r < rmax; r += NG) {
                const unsigned rr = (kv0 + r) & kv_mask;
                bf16v8 v;
                float vsf;
                if constexpr (FP8KV) {
                    v = fp8v8_to_bf16v8(ld_glob_fp8v8(vb8 + (size_t)rr * D + dbase));
                    vsf = vsc[rr];
                } else {
                    v = ld_glob8(vbase + (size_t)rr * D + dbase);
                    vsf = 1.0f;
                }
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    const float pw = Ssm[g * FA_DEC_TILE + r] * vsf;
#pragma unroll
                    for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(v[u]);
                }
            }
            __syncthreads();
        }

        /* Fold the row-groups, one query head at a time. `osm` is REUSED across the GF heads
         * rather than sized GF x NG x D — at GF=8, D=512 that would be 128 KB of LDS on its own.
         * The barriers cost nothing here: this runs once per work item, not once per KV tile. */
#pragma unroll
        for (int g = 0; g < GF; g++) {
            __syncthreads();
#pragma unroll
            for (int u = 0; u < 8; u++) osm[grp * D + dbase + u] = oacc[g][u];
            __syncthreads();

            const unsigned h = h0 + (unsigned)g;
            float* op = Opart + ((size_t)(b * n_head + h) * nsplit + sp) * D;
            for (unsigned d = tid; d < D; d += PLOW_THREADS) {
                float acc = 0.0f;
#pragma unroll
                for (unsigned gg = 0; gg < NG; gg++) acc += osm[gg * D + d];
                op[d] = acc;
            }
            if (tid == 0) {
                float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_st[g];
                ml[1] = l_st[g];
            }
        }
    }
}

/* Combine the split partials: standard online-softmax merge.
 *
 * NOT split over the feature axis, and that is a MEASURED decision. The work unit is
 * (batch, head) -- 32 items in Gemma decode -- so the merge runs on 32 of 256 CUs, which looks
 * like the textbook "a reduction must not be narrower than its producer" bug. It was rewritten to
 * decompose into (batch, head, d-chunk) with the split axis folded across waves, taking n_work to
 * 128/256. That is CORRECT (golden suite green) and it made the merge itself faster (0.56 -> 0.46
 * ms) -- and it made the TOKEN slower (16.7 -> 16.9), twice, at every width swept.
 *
 * Why: widening an op WIDENS ITS CONSUMER'S GATE. o_proj waits for the merge coarsely, so it now
 * waits for the slowest of 149 workgroups instead of the slowest of 32 -- a max over more samples
 * opens later. The merge's stall went 0.83 -> 1.38 ms and ate the 0.10 ms the op saved.
 *
 * It also did not buy what it was for. Raising `nsplit` to fill the machine with flash work is
 * self-defeating regardless of the merge: Q staging and Opart traffic BOTH scale with nsplit
 * (n_head * nsplit * D), so flash_decode gets slower, not faster, as it is split more finely
 * (1.83 -> 3.08 ms at nsplit 16 -> 64). Split-KV's overhead is the ceiling, not the merge.
 *
 * Do not "fix" this without re-measuring the token. */
template <int D>
__device__ void d_flash_merge(bf16* __restrict__ O_, const float* __restrict__ Opart_,
                              const float* __restrict__ mlpart_, unsigned n_batch,
                              unsigned n_head, unsigned nsplit, unsigned slice, unsigned nblk) {
    /* All three came out of the tensor table, so all three were generic: the Opart reads were
     * flat_load_dword and the O writes flat_store_short. Nothing about the ACCESS was wrong --
     * they are coalesced -- they were just on the slow path. */
    auto* const O = as_glob(O_);
    const auto* const Opart = as_glob(Opart_);
    const auto* const mlpart = as_glob(mlpart_);
    const unsigned n_work = n_batch * n_head;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head, b = w / n_head;
        const auto* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;

        float gm = FA_NEG_INF;
        for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s * 2]);
        float gl = 0.0f;
        for (unsigned s = 0; s < nsplit; s++) {
            if (ml[s * 2] == FA_NEG_INF) continue;
            gl += ml[s * 2 + 1] * FA_EXP(ml[s * 2] - gm);
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

        for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned s = 0; s < nsplit; s++) {
                if (ml[s * 2] == FA_NEG_INF) continue;
                const float sc = FA_EXP(ml[s * 2] - gm);
                acc += Opart[((size_t)(b * n_head + h) * nsplit + s) * D + d] * sc;
            }
            O[((size_t)b * n_head + h) * D + d] = f2bf(acc * inv);
        }
    }
}

/* ============================================================================
 * MLA (Multi-head Latent Attention) decode — DeepSeek.               [DEEPSEEK-MLA]
 *
 * The KV cache stores a single low-rank LATENT per token instead of full K/V:
 *   C_kv  [b][ctx][DK]   the compressed latent (DeepSeek: kv_lora_rank = 512)
 *   K_rope[b][ctx][DR]   the shared RoPE key   (DeepSeek: qk_rope_dim  = 64)
 * both HEAD-SHARED (one logical "kv head") — this is the ~9x smaller KV stream and
 * the long-context decode-bandwidth win.
 *
 * The ABSORPTION trick folds W_uk into Q so the O(ctx) loop NEVER reconstructs the
 * per-head K/V.  Per head g, at kv position j:
 *   score[g] = q_abs[g] . C_kv[j]  +  q_rope[g] . K_rope[j]
 *   oacc[g] += p[g] * C_kv[j]                          (latent-wide, DK)
 * and the W_uv fold  o[g] = W_uv[g]^T . oacc[g]  runs ONCE per query (d_o_uv_fold),
 * O(n_q), not per KV position.  So this kernel is exactly the D=DK flash-decode
 * shape with V == C_kv (k_eq_v on the latent), GQA-fused as the GF=n_head extreme,
 * plus one extra DR rope score term.  It emits latent-wide (O_partial, m, l) — reuse
 * d_flash_merge<DK> to combine the splits, then d_o_uv_fold to reach v_head_dim.
 *
 *   gqa = n_head, n_kv_head = 1  (all query heads read the SAME latent row).
 * The register/occupancy budget is the validated D=512 decode shape (op ~700); at
 * GF=8 this is the Gemma hd=512 MQA decode profile.  n_head > GF forms n_head/GF
 * head-groups that each re-stream the latent (the §5 head-chunking knob).
 * ==========================================================================*/
#define FA_MLA_DEC_LDS_FLOATS(DK, DR, GF)                                                \
    ((GF) * FA_DEC_TILE + 2 * PLOW_WAVES + (GF) * ((DK) / 2) + (GF) * ((DR) / 2) +       \
     FA_DEC_NG(DK) * (DK))

/* MLA head-fusion factor (GLM decode). The MLA latent is HEAD-SHARED (MQA-extreme: all
 * n_head query heads read the SAME latent row), so GF query heads processed together per
 * workgroup re-stream the compact latent [ctx][DK+DR] just ONCE per head-group. n_head/GF
 * groups => latent HBM traffic scales as n_head/GF. At GLM's n_head=64:
 *   GF=2 -> 32 groups (32x latent re-read), GF=8 -> 8 groups (8x): a 4x cut in the
 *   dominant long-context decode stream. Register cost (standalone D=512 decode, gfx950):
 *   GF=2 126 VGPR, GF=4 134, GF=8 170 — all occ-2 (<=256, 2 waves/SIMD). The decode
 * interpreter bucket takes the worst-case over all ops, so this is capped where GF=8 keeps
 * the whole decode kernel occ-2. Fill: n_grp*nsplit work items on 256 CUs => pair with a
 * ctx-adaptive nsplit (~4*GF at long ctx) so the machine stays full. */
#ifndef GLM_MLA_GF
#define GLM_MLA_GF 4
#endif

/* GATHER (sparse top-k / DSA compose, sparse-attn-design.md §3.5): when GATHER, the O(ctx)
 * walk runs over the top_k SELECTED positions instead of the dense [lo,hi) range — the latent
 * row for slot t is idx[b*top_k + t] (the on-device index table, the attention twin of the MoE
 * routing_table). Everything else — the absorbed score, online softmax, split, merge — is
 * byte-identical; the selected set is assumed causal so no window/causal mask is applied. The
 * dense instantiation (GATHER=false) compiles to the exact original code (idx/top_k dead). */
/* PREFILL (n_tok > 1) is the SAME kernel. Decode is the T=1 case of it, which is why this is one
 * inner loop and not two: MLA's absorption trick already removes the per-head K/V reconstruction
 * that makes a dense prefill kernel structurally different from its decode twin, and the causal
 * mask prefill needs (`kv <= qpos`) is ALREADY in the dense `keep` predicate below — decode simply
 * never exercises it, because its single query sits at the end of the context.
 *
 * So the generalisation is a query-token axis on the work decomposition plus a per-token `qpos`:
 *
 *   qpos = len - n_tok + t     query token t of this chunk is at absolute position len-n_tok+t
 *
 * At n_tok=1, t=0 this is `len - 1` and EVERY index below collapses to the original expression
 * ((b*1 + 0)*n_head + h == b*n_head + h), so the decode instantiations are bit-identical to before
 * the generalisation. That is the point: there is no second copy of this loop to drift.
 *
 * CAUSAL CLAMP. `hi` is clamped to qpos+1 so an early query token does not walk the whole context
 * and mask it away — correctness comes from `keep`, but the clamp is what makes prefill O(T*ctx/2)
 * instead of O(T*ctx). For decode qpos+1 == len >= hi, so the clamp is a no-op and costs nothing.
 *
 * NSPLIT MUST BE 1 FOR PREFILL. Splitting the KV range across `nsplit` partials is a decode trick
 * for filling 256 CUs off a single query; prefill already has n_tok*n_grp work items and does not
 * need it. Worse, it is unsafe here: with a per-token causal bound an early token's later splits
 * are EMPTY (lo > qpos), and an empty split emits m=-inf, l=0 which d_flash_merge would divide by.
 * The emitter passes nsplit=1; this is a precondition, not a preference. */
template <int DK, int DR, int GF, bool GATHER = false>
__device__ void d_flash_mla_decode(float* __restrict__ Opart, float* __restrict__ mlpart,
                                   const bf16* __restrict__ Qabs, const bf16* __restrict__ Qrope,
                                   const bf16* __restrict__ Ckv, const bf16* __restrict__ Krope,
                                   const int* __restrict__ kv_len, unsigned n_batch,
                                   unsigned n_head, unsigned kv_stride, unsigned window,
                                   float scale, unsigned nsplit, unsigned kv_mask, unsigned slice,
                                   unsigned nblk, float* lds, const int* __restrict__ idx = nullptr,
                                   unsigned top_k = 0, unsigned n_tok = 1) {
    const unsigned n_grp = n_head / GF;                 /* head-groups; latent re-read per group */
    const unsigned n_work = n_batch * n_tok * n_grp * nsplit;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 6, lane = tid & 63;

    float* Ssm = lds;                                   /* [GF][FA_DEC_TILE] scores        */
    float* hmax = Ssm + GF * FA_DEC_TILE;               /* [PLOW_WAVES]                    */
    float* hsum = hmax + PLOW_WAVES;                    /* [PLOW_WAVES]                    */
    bf16* qsm = (bf16*)(hsum + PLOW_WAVES);             /* [GF][DK] absorbed query rows    */
    bf16* qrsm = qsm + GF * DK;                         /* [GF][DR] rope query rows        */
    float* osm = (float*)(qrsm + GF * DR);              /* [NG][DK] O, per row-group       */

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned hg = (w / nsplit) % n_grp;
        const unsigned t = (w / (nsplit * n_grp)) % n_tok; /* query token; always 0 at n_tok=1 */
        const unsigned b = w / (nsplit * n_grp * n_tok);
        const unsigned h0 = hg * GF; /* the GF consecutive query heads this item carries */

        const unsigned len = (unsigned)kv_len[b];
        /* Query token t of this chunk sits at len-n_tok+t. Decode (n_tok=1) gives len-1. */
        const unsigned qpos = len - n_tok + t;
        /* The causal end for THIS query. Dense only: a gathered set is assumed causal already
         * (the selector produced it), which is why GATHER applies no mask at all below. */
        const unsigned cend = qpos + 1;
        /* GATHER splits over the fixed top_k selected slots; dense splits over the KV window,
         * which for prefill is the per-token window ending at qpos, not at len. */
        const unsigned first = GATHER ? 0u : ((window && cend > window) ? (cend - window) : 0u);
        const unsigned span = GATHER ? top_k : (cend - first);
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi = GATHER ? (lo + per < top_k ? lo + per : top_k)
                                   : (lo + per < cend ? lo + per : cend);

        /* ONE latent "head": the cache base is just this batch's latent block. */
        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const auto* rbase = as_glob(Krope) + (size_t)b * kv_stride * DR;
        /* Selected-index table. One row of top_k per QUERY, so prefill indexes it per (b,t) —
         * a gathered prefill selects a different set for every query token. Collapses to
         * `idx + b*top_k` at n_tok=1. */
        const int* ibase = GATHER ? idx + ((size_t)b * n_tok + t) * top_k : nullptr;

        /* Stage GF absorbed-query rows (DK) and GF rope-query rows (DR), once.
         * Q is [b][t][head][D] — at n_tok=1 the token term vanishes and this is the original
         * [b][head][D] expression. */
        const size_t qrow = ((size_t)b * n_tok + t) * n_head;
        for (unsigned i = tid; i < GF * DK; i += PLOW_THREADS)
            qsm[i] = Qabs[(qrow + h0 + i / DK) * DK + i % DK];
        for (unsigned i = tid; i < GF * DR; i += PLOW_THREADS)
            qrsm[i] = Qrope[(qrow + h0 + i / DR) * DR + i % DR];
        __syncthreads();

        float m_st[GF], l_st[GF];
#pragma unroll
        for (int g = 0; g < GF; g++) { m_st[g] = FA_NEG_INF; l_st[g] = 0.0f; }

        constexpr unsigned NDT = FA_DEC_NDT(DK);
        constexpr unsigned NG = FA_DEC_NG(DK);
        const unsigned dbase = (tid % NDT) * 8;
        const unsigned grp = tid / NDT;
        float oacc[GF][8];
#pragma unroll
        for (int g = 0; g < GF; g++)
#pragma unroll
            for (int u = 0; u < 8; u++) oacc[g][u] = 0.0f;

        for (unsigned kv0 = lo; kv0 < hi; kv0 += FA_DEC_TILE) {
            /* SCORES: each thread owns latent row `kv`; loads C_kv row ONCE + K_rope row ONCE,
             * dots each against all GF staged queries — the latent crosses HBM once per group. */
            const unsigned kv = kv0 + tid;
            /* dense: cache row (kv & kv_mask); gather: the selected index ibase[kv]. */
            const unsigned row = GATHER ? (kv < hi ? (unsigned)ibase[kv] : 0u) : (kv & kv_mask);
            const bool keep = GATHER ? (kv < hi)
                                     : (kv < hi && kv <= qpos && (!window || (qpos - kv) < window));
            float s[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = FA_NEG_INF;
            if (keep) {
                float dot[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                const auto* crow = cbase + (size_t)row * DK;
#pragma unroll
                for (int d = 0; d < DK; d += 8) {
                    const bf16v8 c8 = ld_glob8(crow + d);
#pragma unroll
                    for (int g = 0; g < GF; g++)
                        dot[g] = dot8(c8, ld_lds8(qsm + g * DK + d), dot[g]);
                }
                const auto* rrow = rbase + (size_t)row * DR;
#pragma unroll
                for (int d = 0; d < DR; d += 8) {
                    const bf16v8 r8 = ld_glob8(rrow + d);
#pragma unroll
                    for (int g = 0; g < GF; g++)
                        dot[g] = dot8(r8, ld_lds8(qrsm + g * DR + d), dot[g]);
                }
#pragma unroll
                for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale);
            }
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = s[g];
            __syncthreads();

            /* GF softmax reductions, one per wave (identical to d_flash_decode). */
            if (wave < GF) {
                float mx = FA_NEG_INF;
                for (unsigned i = lane; i < FA_DEC_TILE; i += 64)
                    mx = fmaxf(mx, Ssm[wave * FA_DEC_TILE + i]);
#pragma unroll
                for (int off = 32; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
                if (lane == 0) hmax[wave] = mx;
            }
            __syncthreads();

            float mnew[GF], corr[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) {
                mnew[g] = fmaxf(m_st[g], hmax[g]);
                corr[g] = (m_st[g] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[g] - mnew[g]);
            }
            float pe[GF];
#pragma unroll
            for (int g = 0; g < GF; g++)
                pe[g] = (mnew[g] == FA_NEG_INF || s[g] == FA_NEG_INF) ? 0.0f
                                                                      : FA_EXP(s[g] - mnew[g]);
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = pe[g];
            __syncthreads();

            if (wave < GF) {
                float sm = 0.0f;
                for (unsigned i = lane; i < FA_DEC_TILE; i += 64) sm += Ssm[wave * FA_DEC_TILE + i];
                sm = wave_sum(sm);
                if (lane == 0) hsum[wave] = sm;
            }
            __syncthreads();

#pragma unroll
            for (int g = 0; g < GF; g++) {
                l_st[g] = l_st[g] * corr[g] + hsum[g];
                m_st[g] = mnew[g];
            }
#pragma unroll
            for (int g = 0; g < GF; g++)
#pragma unroll
                for (int u = 0; u < 8; u++) oacc[g][u] *= corr[g];

            /* PV: V IS the latent C_kv, DK wide.  Cooperative column read, one row per NG group. */
            constexpr int VU = FA_DEC_VU(GF);
            const unsigned rmax = (hi - kv0 < FA_DEC_TILE) ? (hi - kv0) : FA_DEC_TILE;
            unsigned r = grp;
            for (; r + (VU - 1) * NG < rmax; r += VU * NG) {
                bf16v8 vv[VU];
#pragma unroll
                for (int c = 0; c < VU; c++) {
                    const unsigned t = kv0 + r + (unsigned)c * NG;
                    const size_t vrow = GATHER ? (size_t)(unsigned)ibase[t] : (t & kv_mask);
                    vv[c] = ld_glob8(cbase + vrow * DK + dbase);
                }
#pragma unroll
                for (int c = 0; c < VU; c++) {
#pragma unroll
                    for (int g = 0; g < GF; g++) {
                        const float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG];
#pragma unroll
                        for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(vv[c][u]);
                    }
                }
            }
            for (; r < rmax; r += NG) {
                const size_t vrow = GATHER ? (size_t)(unsigned)ibase[kv0 + r] : ((kv0 + r) & kv_mask);
                const bf16v8 v = ld_glob8(cbase + vrow * DK + dbase);
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    const float pw = Ssm[g * FA_DEC_TILE + r];
#pragma unroll
                    for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(v[u]);
                }
            }
            __syncthreads();
        }

        /* Fold the row-groups and write latent-wide partials, one head at a time. */
#pragma unroll
        for (int g = 0; g < GF; g++) {
            __syncthreads();
#pragma unroll
            for (int u = 0; u < 8; u++) osm[grp * DK + dbase + u] = oacc[g][u];
            __syncthreads();

            /* Partials are [b][t][head][split][DK] — again the original [b][head][split][DK]
             * once n_tok=1. d_flash_merge<DK> and d_o_uv_fold then run per (b,t,head). */
            const size_t oh = qrow + h0 + (unsigned)g;
            float* op = Opart + (oh * nsplit + sp) * DK;
            for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
                float acc = 0.0f;
#pragma unroll
                for (unsigned gg = 0; gg < NG; gg++) acc += osm[gg * DK + d];
                op[d] = acc;
            }
            if (tid == 0) {
                float* ml = mlpart + (oh * nsplit + sp) * 2;
                ml[0] = m_st[g];
                ml[1] = l_st[g];
            }
        }
    }
}

/* ============================================================================
 * MLA PREFILL and GATHERED MLA PREFILL.                              [DEEPSEEK-MLA]
 *
 * PLOW_DOP_FLASH_MLA_PREFILL (51) and PLOW_DOP_FLASH_GATHER_PREFILL (55). Until these
 * landed, Kimi K2.7 / DeepSeek V2-V3 / GLM 5.2 could DECODE on gfx950 but could not
 * prefill through their own attention — the opcodes were in the ISA and absent from the
 * AMD switch, so a real prompt had nothing to run. That was the largest functional gap on
 * the AMD path, and it was not a dtype problem: bf16 MLA prefill was missing too.
 *
 * These are wrappers, not kernels. The whole prefill body is d_flash_mla_decode with
 * n_tok > 1, for the reason its header explains: MLA's absorption removes the per-head K/V
 * reconstruction that makes a DENSE prefill kernel structurally different from its decode
 * twin, and the causal mask is already in that loop's `keep`. Writing a second inner loop
 * here would buy nothing and would be a second thing to keep correct.
 *
 * SHAPES (both):
 *   Qabs   [b][t][n_head][DK]   absorbed query  (W_uk folded in by the emitter)
 *   Qrope  [b][t][n_head][DR]   rope query
 *   Ckv    [b][ctx][DK]         latent cache, HEAD-SHARED
 *   Krope  [b][ctx][DR]         shared rope key, HEAD-SHARED
 *   Opart  [b][t][n_head][nsplit][DK] + mlpart [..][2]  -> d_flash_merge<DK> -> d_o_uv_fold
 *   kv_len [b]                  total context INCLUDING this chunk, so query t is at
 *                               kv_len[b] - n_tok + t
 *
 * nsplit MUST be 1 (see d_flash_mla_decode's header: a per-token causal bound makes an
 * early token's later splits empty, and an empty split emits l=0 for d_flash_merge to
 * divide by). Prefill has n_tok*n_grp work items and does not need the split anyway.
 * ==========================================================================*/
template <int DK, int DR, int GF>
__device__ void d_flash_mla_prefill(float* __restrict__ Opart, float* __restrict__ mlpart,
                                    const bf16* __restrict__ Qabs, const bf16* __restrict__ Qrope,
                                    const bf16* __restrict__ Ckv, const bf16* __restrict__ Krope,
                                    const int* __restrict__ kv_len, unsigned n_batch,
                                    unsigned n_tok, unsigned n_head, unsigned kv_stride,
                                    unsigned window, float scale, unsigned kv_mask, unsigned slice,
                                    unsigned nblk, float* lds) {
    d_flash_mla_decode<DK, DR, GF, false>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch,
                                          n_head, kv_stride, window, scale, /*nsplit*/ 1, kv_mask,
                                          slice, nblk, lds, nullptr, 0, n_tok);
}

/* GATHERED prefill (GLM DSA / DeepSeek sparse attention). `idx` is one top_k row PER QUERY —
 * [b][t][top_k] — because a sparse prefill selects a different set for every query token,
 * which is exactly the axis the dense decode gather did not have. The selected set is assumed
 * causal (the selector produced it), so no mask is applied, matching FLASH_GATHER_DECODE. */
template <int DK, int DR, int GF>
__device__ void d_flash_gather_prefill(float* __restrict__ Opart, float* __restrict__ mlpart,
                                       const bf16* __restrict__ Qabs,
                                       const bf16* __restrict__ Qrope,
                                       const bf16* __restrict__ Ckv,
                                       const bf16* __restrict__ Krope,
                                       const int* __restrict__ kv_len,
                                       const int* __restrict__ idx, unsigned top_k,
                                       unsigned n_batch, unsigned n_tok, unsigned n_head,
                                       unsigned kv_stride, float scale, unsigned kv_mask,
                                       unsigned slice, unsigned nblk, float* lds) {
    d_flash_mla_decode<DK, DR, GF, true>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch,
                                         n_head, kv_stride, /*window*/ 0, scale, /*nsplit*/ 1,
                                         kv_mask, slice, nblk, lds, idx, top_k, n_tok);
}

/* ============================================================================
 * MLA decode, HEAD-PACKED MFMA (GLM/DeepSeek).                        [DEEPSEEK-MLA]
 *
 * STATUS: CORRECT (mla_test PASS incl n_head=64 dense+gather) but NON-DEFAULT — measured
 * SLOWER than the scalar d_flash_mla_decode above on plow. Head-packing forces n_grp=1, so
 * filling 256 CUs needs nsplit~256, and plow's SEPARATE O(nsplit) d_flash_merge then explodes
 * (110us at ns256); at low nsplit only nsplit CUs run. The latent-read-once win is negated —
 * both kernels are latency-bound at 256-CU fill. Realizing this lever needs a FUSED persistent
 * split-reduce + double-buffered per-CU MFMA (AITER's structure), an architectural change, not
 * a kernel port. Kept as the validated foundation for that follow-up. See
 * plans/glm-mla-flash-tuning.md. The production path is the scalar GF/nsplit kernel.
 *
 * The scalar d_flash_mla_decode above re-streams the latent once per HEAD-GROUP
 * (GF heads/group => 64/GF groups), because the O accumulator oacc[GF][8] is in
 * arch-VGPR and GF=64 would need 512 VGPR. This variant packs ALL n_head query heads
 * into the MFMA M-dimension (heads == the "query rows" of a D=512 flash), so the O
 * accumulator lives in AGPR (matrix-core output) and the whole workgroup reads the
 * compact latent tile from LDS exactly ONCE per KV tile. GF is effectively n_head:
 * latent HBM traffic drops to 1x, the roofline floor (op_attention design note).
 *
 * Structure = a D=DK flash-decode with:
 *   - M-rows = the n_head query heads (no causal mask: every head is at qpos, attends
 *     the full [0,len) latent; only the head>=n_head and kv>=hi tails are masked),
 *   - QK contraction = DK latent (absorbed Q . C_kv) PLUS a DR rope addendum
 *     (Q_rope . K_rope), both accumulated into the same 32x32 score tile,
 *   - PV = P . C_kv over the DK latent (k_eq_v on the latent), N = DK.
 * The 8-wave workgroup tiles the work as (M-tile x N-quarter): n_mtile=ceil(n_head/32)
 * M-tiles, 8/n_mtile waves per M-tile each owning DK/(8/n_mtile) output columns. The
 * score tile is recomputed per column-group (cheap; the latent is read from LDS, so no
 * extra HBM). Emits the SAME latent-wide (Opart,m,l) partials as the scalar kernel, so
 * d_flash_merge<DK> + d_o_uv_fold consume it unchanged. GATHER walks the idx[] top-k.
 *
 * Reuses the empirically-validated 32x32x16_bf16 fragment maps (amd_common.h:437) and
 * the prefill online-softmax + P-transpose-through-LDS (op_attention.h d_flash_prefill).
 * ==========================================================================*/
#define FA_MLA_MFMA_STRIDE(DK, DR) ((DK) + (DR) + FA_PAD)
#define FA_MLA_MFMA_LDS_FLOATS(DK, DR)                                                    \
    /* Ksm[BKV=32][DK+DR+pad] + Psm[PLOW_WAVES][32][BKV=32], in floats (bf16 => /2). */   \
    ((32 * FA_MLA_MFMA_STRIDE(DK, DR) + PLOW_WAVES * 32 * 32 + 1) / 2)

template <int DK, int DR, bool GATHER = false>
__device__ void d_flash_mla_decode_mfma(float* __restrict__ Opart, float* __restrict__ mlpart,
                                        const bf16* __restrict__ Qabs, const bf16* __restrict__ Qrope,
                                        const bf16* __restrict__ Ckv, const bf16* __restrict__ Krope,
                                        const int* __restrict__ kv_len, unsigned n_batch,
                                        unsigned n_head, unsigned kv_stride, unsigned window,
                                        float scale, unsigned nsplit, unsigned kv_mask,
                                        unsigned slice, unsigned nblk, float* lds_,
                                        const int* __restrict__ idx = nullptr, unsigned top_k = 0) {
    (void)window; (void)kv_mask;
    constexpr int BKV = 32;                            /* one MFMA N-tile of KV per pass  */
    constexpr int STRIDE = FA_MLA_MFMA_STRIDE(DK, DR); /* Ksm row stride (latent|rope)    */
    constexpr int NK_ABS = DK / MFMA_K;                /* QK k-steps over the latent      */
    constexpr int NK_ROPE = DR / MFMA_K;               /* + rope k-steps (DR=64 => 4)     */
    constexpr int NK = NK_ABS + NK_ROPE;
    /* Out d-tiles/wave = cpw/32, cpw = DK/(PLOW_WAVES/n_mtile). n_head<=64 => n_mtile<=2 =>
     * wpm>=PLOW_WAVES/2 => cpw <= DK/(PLOW_WAVES/2), so MAXNDT bounds oacc for occ-2. */
    constexpr int MAXNDT = DK / (PLOW_WAVES / 2) / MFMA_N;

    bf16* lds = (bf16*)lds_;
    bf16* Ksm = lds;                                   /* [BKV][STRIDE] latent|rope tile  */
    bf16* Psm = Ksm + BKV * STRIDE;                    /* [PLOW_WAVES][32][BKV] P transpose */

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63;
    const unsigned wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);         /* l % 32 */
    const unsigned accn = mfma_acc_n(lane);            /* l % 32 (== frow) */

    /* (M-tile x column-group) wave map. n_mtile in {1,2} for n_head<=64. */
    const unsigned n_mtile = (n_head + 31) / 32;
    const unsigned wpm = PLOW_WAVES / n_mtile;         /* waves per M-tile (8 or 4)       */
    const unsigned m_tile = wave % n_mtile;            /* which 32-head block             */
    const unsigned cgrp = wave / n_mtile;              /* which column-group              */
    const unsigned cpw = DK / wpm;                     /* output cols this wave owns       */
    const unsigned ncol0 = cgrp * cpw;                 /* first output col                 */
    const unsigned ndt = cpw / MFMA_N;                 /* out d-tiles this wave (<=MAXNDT) */
    const unsigned h_tile0 = m_tile * 32;              /* first head of this M-tile        */

    const unsigned n_work = n_batch * nsplit;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned b = w / nsplit;
        const unsigned len = (unsigned)kv_len[b];
        const unsigned first = 0u;
        const unsigned span = GATHER ? top_k : (len - first);
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi = GATHER ? (lo + per < top_k ? lo + per : top_k)
                                   : (lo + per < len ? lo + per : len);
        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const auto* rbase = as_glob(Krope) + (size_t)b * kv_stride * DR;
        const int* ibase = GATHER ? idx + (size_t)b * top_k : nullptr;

        f32x16 oacc[MAXNDT];
#pragma unroll
        for (int t = 0; t < MAXNDT; t++) oacc[t] = (f32x16)(0.0f);
        float m_st[16], l_st[16];
#pragma unroll
        for (int i = 0; i < 16; i++) { m_st[i] = FA_NEG_INF; l_st[i] = 0.0f; }

        for (unsigned kv0 = lo; kv0 < hi; kv0 += BKV) {
            /* Stage the latent|rope tile to LDS ONCE, cooperatively across all 512 threads. */
            __syncthreads();
            for (unsigned e = tid; e < BKV * DK; e += PLOW_THREADS) {
                const unsigned r = e / DK, c = e % DK, kv = kv0 + r;
                bf16 v = 0;
                if (kv < hi) {
                    const size_t row = GATHER ? (size_t)(unsigned)ibase[kv] : (size_t)kv;
                    v = cbase[row * DK + c];
                }
                Ksm[r * STRIDE + c] = v;
            }
            for (unsigned e = tid; e < BKV * DR; e += PLOW_THREADS) {
                const unsigned r = e / DR, c = e % DR, kv = kv0 + r;
                bf16 v = 0;
                if (kv < hi) {
                    const size_t row = GATHER ? (size_t)(unsigned)ibase[kv] : (size_t)kv;
                    v = rbase[row * DR + c];
                }
                Ksm[r * STRIDE + DK + c] = v;
            }
            __syncthreads();

            /* QK: S[head, kv] = Q_abs.C_kv + Q_rope.K_rope, one 32x32 tile (this M-tile). */
            f32x16 s = (f32x16)(0.0f);
#pragma unroll
            for (int kk = 0; kk < NK; kk++) {
                const unsigned d0 = mfma_frag_k(lane, kk * MFMA_K);
                bf16x8 qf, kf;
                /* Q fragment: abs for k-steps over [0,DK), rope for [DK,DK+DR). */
                const unsigned qh = h_tile0 + frow;
                const bool qv = (qh < n_head);
                bf16 q8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (qv) {
                    const bf16* qp = (d0 < (unsigned)DK)
                        ? as_glob(Qabs) + ((size_t)b * n_head + qh) * DK + d0
                        : as_glob(Qrope) + ((size_t)b * n_head + qh) * DR + (d0 - DK);
                    __builtin_memcpy(q8, qp, 16);
                }
#pragma unroll
                for (int j = 0; j < 8; j++) { bf16_t v; __builtin_memcpy(&v, &q8[j], 2); qf[j] = v; }
                bf16 t8[8];
                __builtin_memcpy(t8, &Ksm[frow * STRIDE + d0], 16);
#pragma unroll
                for (int j = 0; j < 8; j++) { bf16_t v; __builtin_memcpy(&v, &t8[j], 2); kf[j] = v; }
                s = __builtin_amdgcn_mfma_f32_32x32x16_bf16(qf, kf, s, 0, 0, 0);
            }

            /* Online softmax over this KV tile (row = head lives in one half-wave). */
            float p[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned qh = h_tile0 + mfma_acc_m(lane, i);
                const unsigned kg = kv0 + accn;
                const bool valid = (qh < n_head) && (kg < hi);
                p[i] = valid ? (s[i] * FA_SCALE(scale)) : FA_NEG_INF;
            }
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float rmax = half_wave_max(p[i]);
                const float mnew = fmaxf(m_st[i], rmax);
                const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
                l_st[i] = l_st[i] * corr + half_wave_sum(pe);
                m_st[i] = mnew;
                p[i] = pe;
#pragma unroll
                for (int t = 0; t < MAXNDT; t++) oacc[t][i] *= corr;
            }

            /* P -> LDS as the PV A-operand (m = head%32, k = kv). */
            bf16* myP = Psm + wave * 32 * BKV;
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 16; i++) myP[mfma_acc_m(lane, i) * BKV + accn] = f2bf(p[i]);
            __syncthreads();

            /* O += P . V, V = the latent slice (Ksm cols [ncol0, ncol0+cpw)). */
            for (unsigned t = 0; t < ndt; t++) {
#pragma unroll
                for (int ks = 0; ks < BKV; ks += MFMA_K) {
                    const unsigned kk = mfma_frag_k(lane, ks);
                    bf16x8 pf, vf;
                    bf16 tp[8];
                    __builtin_memcpy(tp, &myP[frow * BKV + kk], 16);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        bf16_t v; __builtin_memcpy(&v, &tp[j], 2); pf[j] = v;
                        bf16 vv = Ksm[(kk + j) * STRIDE + ncol0 + t * MFMA_N + accn];
                        bf16_t vb; __builtin_memcpy(&vb, &vv, 2); vf[j] = vb;
                    }
                    oacc[t] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(pf, vf, oacc[t], 0, 0, 0);
                }
            }
        }

        /* Emit latent-wide partials (o, m, l) — same layout the scalar kernel writes. */
        for (unsigned t = 0; t < ndt; t++) {
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned mh = mfma_acc_m(lane, i);
                const unsigned qh = h_tile0 + mh;
                if (qh >= n_head) continue;
                const unsigned d = ncol0 + t * MFMA_N + accn;
                float* op = Opart + ((size_t)(b * n_head + qh) * nsplit + sp) * DK + d;
                *op = oacc[t][i];
            }
        }
        /* (m,l): one column-group (cgrp==0) owns each head-row's normalizer. */
        if (cgrp == 0) {
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned mh = mfma_acc_m(lane, i);
                const unsigned qh = h_tile0 + mh;
                if (qh >= n_head) continue;
                if (accn == 0) { /* one lane per row writes */
                    float* ml = mlpart + ((size_t)(b * n_head + qh) * nsplit + sp) * 2;
                    ml[0] = m_st[i];
                    ml[1] = l_st[i];
                }
            }
        }
        __syncthreads();
    }
}

/* ATTN_SELECT — on-device top-k KV selection (DeepSeek DSA).            [DEEPSEEK-MLA]
 * sparse-attn-design.md §3.2. The attention twin of the MoE router: scores the indexer
 * query against every KV position's index key, keeps the top_k positions per batch, and
 * writes the idx table d_flash_mla_decode<...,GATHER=true> reads (idx[b*top_k + t]).
 *
 * Selection is a deterministic RANK: for KV position t,
 *   key[t] = (ordered_bits(score[t]) << 20) | (len-1 - t)
 * (the SAME monotone f32->u32 packing the router uses), so a plain unsigned compare orders
 * by score and, on a tie, by LOWEST index — the reproducible tie-break the whole sparse
 * path rests on. Position t is selected iff rank(t) = #{u : key[u] > key[t]} < top_k, and it
 * lands at slot `rank` (score-descending order; the gather's online softmax is set-order
 * invariant). One workgroup per batch (decode n_batch is tiny); scores staged in LDS as
 * packed keys. Scores-only, no O accumulator — the cheapest op of the family. */
__device__ void d_attn_select(int* __restrict__ idx, const bf16* __restrict__ Qidx,
                              const bf16* __restrict__ Kidx, const int* __restrict__ kv_len,
                              unsigned n_batch, unsigned index_dim, unsigned kv_stride,
                              unsigned top_k, float scale, unsigned slice, unsigned nblk,
                              unsigned long long* keys) {
    const auto* qg = as_glob(Qidx);
    const auto* kg = as_glob(Kidx);
    for (unsigned b = slice; b < n_batch; b += nblk) {
        const unsigned len = (unsigned)kv_len[b];
        const auto* q = qg + (size_t)b * index_dim;
        const auto* kb = kg + (size_t)b * kv_stride * index_dim;
        int* ib = idx + (size_t)b * top_k;
        /* 1. score every KV position -> packed key in LDS. */
        for (unsigned t = threadIdx.x; t < len; t += PLOW_THREADS) {
            float d = 0.0f;
            const auto* kr = kb + (size_t)t * index_dim;
            for (unsigned i = 0; i < index_dim; i++) d += bf2f(q[i]) * bf2f(kr[i]);
            float sc = d * scale;
            unsigned sb;
            __builtin_memcpy(&sb, &sc, 4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            keys[t] = ((unsigned long long)sb << 20) | (unsigned long long)((len - 1 - t) & 0xFFFFFu);
        }
        __syncthreads();
        /* 2. rank each position; the top_k highest keys write their index at slot=rank. */
        for (unsigned t = threadIdx.x; t < len; t += PLOW_THREADS) {
            const unsigned long long kt = keys[t];
            unsigned rank = 0;
            for (unsigned u = 0; u < len; u++) rank += (keys[u] > kt);
            if (rank < top_k) ib[rank] = (int)t;
        }
        __syncthreads();
    }
}

/* DSA LIGHTNING INDEXER score (DeepSeek-V3.2 eq.1, GLM-5.2 glm_moe_dsa).       [DSA]
 *
 *   score[b][t] = sum_{h=0..HI-1}  w[b][h] * ReLU( q_idx[b][h] . k_idx[b][t] )
 *
 * q_idx  [b][HI][DI]  the HI (=index_n_heads=32) indexer query heads (already RoPE'd upstream)
 * k_idx  [b][ctx][DI] the shared indexer key per KV position (k_norm'd + RoPE'd upstream, cached)
 * w      [b][HI]      the per-head "lightning" weights (weights_proj @ x)
 * -> score[b][ctx]    f32, consumed by the top-k SELECT (d_index_select).
 *
 * DI = index_head_dim = 128. The score is a small GEMM ([HI x DI] . [DI x ctx]) with a ReLU and a
 * w-weighted head reduction; positions are spread across ALL workgroups*threads (grid-stride) so it
 * FILLS the chip (unlike the one-WG d_attn_select). q (HI*DI bf16 = 8 KiB) + w (HI) are staged in
 * LDS once per batch and reused across positions. Scale is applied but selection is scale-invariant
 * (monotone), so the exact indexer scale only matters if score is consumed numerically.
 * Correctness: a plain weighted sum of ReLU dot-products — mirror in the Rust/CPU oracle verbatim. */
template <int DI>
__device__ void d_index_score(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                              const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                              const int* __restrict__ kv_len, unsigned n_batch, unsigned index_heads,
                              unsigned kv_stride, float scale, unsigned slice, unsigned nblk,
                              bf16* qlds /* index_heads*DI bf16 */) {
    auto* const Sc = as_glob(Score);
    const auto* const Qg = as_glob(Qidx);
    const auto* const Kg = as_glob(Kidx);
    const auto* const Wg = as_glob(W);
    const unsigned tid = threadIdx.x;
    const unsigned HI = index_heads;
    for (unsigned b = 0; b < n_batch; b++) {
        const unsigned len = (unsigned)kv_len[b];
        /* stage this batch's HI*DI query + HI weights into LDS (reused across all positions). */
        for (unsigned i = tid; i < HI * (unsigned)DI; i += PLOW_THREADS) qlds[i] = Qg[(size_t)b * HI * DI + i];
        __syncthreads();
        /* w cached in registers per thread (HI<=32 small); read from global (coalesced enough). */
        const auto* wb = Wg + (size_t)b * HI;
        /* positions spread across the whole grid: workgroup `slice`, then threads, grid-strided. */
        for (unsigned t = slice * PLOW_THREADS + tid; t < len; t += nblk * PLOW_THREADS) {
            const auto* kt = Kg + ((size_t)b * kv_stride + t) * DI;
            float s = 0.0f;
            for (unsigned h = 0; h < HI; h++) {
                const bf16* qh = qlds + (size_t)h * DI;
                float d = 0.0f;
                for (int i = 0; i < DI; i++) d += bf2f(qh[i]) * bf2f(kt[i]);
                s += bf2f(wb[h]) * (d > 0.0f ? d : 0.0f); /* ReLU */
            }
            Sc[(size_t)b * kv_stride + t] = s * scale;
        }
        __syncthreads();
    }
}

/* DSA lightning-indexer score — COALESCED/register-cached variant (perf follow-up a).      [DSA]
 * Identical math to d_index_score, but reordered i-outer / h-inner so each KV key element kt[i] is
 * read from HBM exactly ONCE per position and reused across all HI heads via HI register
 * accumulators (the scalar kernel re-reads the whole 128-d key HI=32 times — 32x redundant HBM
 * traffic, which dominates at long ctx). HI is fixed to the template HIc (index_n_heads=32 on GLM)
 * so the head loop unrolls and acc[] stays in VGPRs. Bit-identical reduction order to the scalar
 * path (sum over i then over h) up to fp associativity — the CPU oracle mirrors i-outer. */
template <int DI, int HIc>
__device__ void d_index_score_fast(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                                   const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                                   const int* __restrict__ kv_len, unsigned n_batch,
                                   unsigned kv_stride, float scale, unsigned slice, unsigned nblk,
                                   bf16* qlds /* HIc*DI bf16 */) {
    auto* const Sc = as_glob(Score);
    const auto* const Qg = as_glob(Qidx);
    const auto* const Kg = as_glob(Kidx);
    const auto* const Wg = as_glob(W);
    const unsigned tid = threadIdx.x;
    for (unsigned b = 0; b < n_batch; b++) {
        const unsigned len = (unsigned)kv_len[b];
        for (unsigned i = tid; i < HIc * (unsigned)DI; i += PLOW_THREADS)
            qlds[i] = Qg[(size_t)b * HIc * DI + i];
        __syncthreads();
        const auto* wb = Wg + (size_t)b * HIc;
        float wr[HIc];
#pragma unroll
        for (int h = 0; h < HIc; h++) wr[h] = bf2f(wb[h]);
        for (unsigned t = slice * PLOW_THREADS + tid; t < len; t += nblk * PLOW_THREADS) {
            const auto* kt = Kg + ((size_t)b * kv_stride + t) * DI;
            float acc[HIc];
#pragma unroll
            for (int h = 0; h < HIc; h++) acc[h] = 0.0f;
            for (int i = 0; i < DI; i++) {
                const float kv = bf2f(kt[i]);
#pragma unroll
                for (int h = 0; h < HIc; h++) acc[h] += bf2f(qlds[(size_t)h * DI + i]) * kv;
            }
            float s = 0.0f;
#pragma unroll
            for (int h = 0; h < HIc; h++) s += wr[h] * (acc[h] > 0.0f ? acc[h] : 0.0f);
            Sc[(size_t)b * kv_stride + t] = s * scale;
        }
        __syncthreads();
    }
}

/* DSA lightning-indexer score — WIDE-K MFMA (perf floor 1: ~94us -> ~10us).      [DSA]
 *
 * score[t] = sum_h w[h]*ReLU(q_idx[h].k_idx[t]) is exactly a [HI x DI].[DI x ctx] GEMM (contract
 * over DI=index_head_dim) followed by a per-head-weighted ReLU reduction over h. The register-cached
 * `_fast` kernel is HBM-BANDWIDTH-STARVED: one thread per position reads its 128-d key with a
 * DI-strided (256B) gather across the wave, so a whole cache line is fetched to deliver ~2 useful
 * bytes/lane and effective BW sits ~350 GB/s (32 MB / 93us @128k). This kernel instead STREAMS the
 * key matrix contiguously: the whole workgroup coalesce-loads a TILE_N=PLOW_WAVES*32 slab of keys
 * into LDS (contiguous [pos][dim], one b128/lane), then each wave runs a 32-position MFMA subtile
 * against the HI=32 query rows. With DI=128 the contraction is 8 wide-K MFMA k-steps (32x32x16 bf16),
 * so the [32-head x 32-pos] score-dot tile falls out of the accumulator in one shot.
 *
 * Accumulator D[m=h][n=pos]: lane l holds n = l%32 (the position within the subtile) and, across its
 * 16 acc elements, m = mfma_acc_m(l,i) (16 of the 32 heads). Lanes l and l+32 hold the SAME position
 * with complementary head-halves, so each lane ReLU/w-weights+sums its 16 heads and a single
 * __shfl_xor(.,32) folds the two halves into score[pos]. Q fragments (A operand, rows = heads) are
 * hoisted to VGPR once per batch and reused across every key slab; only K streams. bf16 in, fp32
 * accumulate — same math as the scalar/fast path (relmax 0.0000 vs the CPU weighted-ReLU ref). */
template <int DI, int HIc>
__device__ void d_index_score_mfma(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                                   const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                                   const int* __restrict__ kv_len, unsigned n_batch,
                                   unsigned kv_stride, float scale, unsigned nblk,
                                   bf16* qlds /* HIc*QSTRIDE bf16 */, bf16* ktile /* TILE_N*KSTRIDE */,
                                   float* wlds /* HIc floats */) {
    static_assert(DI % 16 == 0, "DI must be a whole number of MFMA k-steps");
    static_assert(HIc == 32, "MFMA subtile assumes index_n_heads == 32 (one 32x32 acc tile)");
    constexpr int NK = DI / MFMA_K;          /* wide-K contraction steps (DI=128 -> 8)   */
    constexpr int QSTRIDE = DI + FA_PAD;     /* padded LDS row to break bank conflicts    */
    constexpr int KSTRIDE = DI + FA_PAD;
    constexpr int TILE_N = PLOW_WAVES * 32;  /* positions per workgroup slab (8*32 = 256) */
    auto* const Sc = as_glob(Score);
    const auto* const Qg = as_glob(Qidx);
    const auto* const Kg = as_glob(Kidx);
    const auto* const Wg = as_glob(W);
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63, wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);   /* m for Q (=head), n for K (=pos-in-subtile) */
    for (unsigned b = 0; b < n_batch; b++) {
        const unsigned len = (unsigned)kv_len[b];
        __syncthreads();
        /* stage the HI query rows (padded [h][i]) + the HI head weights into LDS. */
        for (unsigned e = tid; e < (unsigned)HIc * (DI / 8); e += PLOW_THREADS) {
            const unsigned h = e / (DI / 8), c8 = (e % (DI / 8)) * 8;
            *(bf16v8*)&qlds[h * QSTRIDE + c8] = ld_glob8(&Qg[(size_t)b * HIc * DI + h * DI + c8]);
        }
        for (unsigned i = tid; i < (unsigned)HIc; i += PLOW_THREADS) wlds[i] = bf2f(Wg[(size_t)b * HIc + i]);
        __syncthreads();
        /* hoist this batch's Q as MFMA A-fragments (rows = heads); reused across every key slab. */
        bf16x8 qf[NK];
#pragma unroll
        for (int ks = 0; ks < NK; ks++) {
            const unsigned d0 = mfma_frag_k(lane, ks * MFMA_K);
            qf[ks] = __builtin_bit_cast(bf16x8, ld_lds8(&qlds[frow * QSTRIDE + d0]));
        }
        const unsigned nslab = (len + TILE_N - 1) / TILE_N;
        for (unsigned st = blockIdx.x; st < nslab; st += nblk) {
            const unsigned base = st * TILE_N;
            __syncthreads(); /* previous slab's MFMA readers done before we overwrite ktile */
            /* coalesced contiguous key load: [TILE_N][DI], zero-filled past len. */
            for (unsigned c = tid; c < (unsigned)TILE_N * (DI / 8); c += PLOW_THREADS) {
                const unsigned row = c / (DI / 8), c8 = (c % (DI / 8)) * 8;
                const unsigned pos = base + row;
                *(bf16v8*)&ktile[row * KSTRIDE + c8] =
                    (pos < len) ? ld_glob8(&Kg[((size_t)b * kv_stride + pos) * DI + c8]) : bf16v8_zero();
            }
            __syncthreads(); /* key slab visible to all waves before the MFMA reads */
            /* this wave's 32-position subtile: D[head][pos] = Qidx . Kidx over DI. */
            const unsigned krow0 = wave * 32; /* first ktile row this wave owns */
            f32x16 acc = (f32x16)(0.0f);
#pragma unroll
            for (int ks = 0; ks < NK; ks++) {
                const unsigned d0 = mfma_frag_k(lane, ks * MFMA_K);
                const bf16x8 kfrag = __builtin_bit_cast(bf16x8, ld_lds8(&ktile[(krow0 + frow) * KSTRIDE + d0]));
                acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(qf[ks], kfrag, acc, 0, 0, 0);
            }
            /* epilogue: this lane owns pos = lane%32, and 16 of the 32 heads (l and l+32 split). */
            const unsigned mbase = 4 * (lane / 32);
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned h = mbase + (i % 4) + 8 * (i / 4);
                const float d = acc[i];
                part += wlds[h] * (d > 0.0f ? d : 0.0f); /* w-weighted ReLU */
            }
            part += __shfl_xor(part, 32, PLOW_WAVE); /* fold the two head-halves for this pos */
            if (lane < 32) {
                const unsigned pos = base + krow0 + lane;
                if (pos < len) Sc[(size_t)b * kv_stride + pos] = part * scale;
            }
        }
    }
}

/* DSA top-k SELECT via radix threshold (G4 — the hard selector).               [DSA]
 * Consumes the f32 indexer scores in HBM (d_index_score output) and writes the top_k position
 * indices (idx[b*top_k + slot]) the gather flash reads. Uses the SAME monotone packed key as
 * d_attn_select: key[t] = (ordered_bits(score[t]) << 20) | (len-1-t). The low 20 index bits make
 * every key UNIQUE, so "the top_k largest keys" == "the top_k highest scores, lowest-index on a
 * score tie" — the reproducible tie-break the whole sparse path rests on. It finds the EXACT
 * top_k-th largest key by MSB-first 8-bit radix select (7 byte-passes over the 52-bit key), then
 * emits every position whose key >= that threshold: exactly top_k, emission order immaterial (the
 * gather's online softmax is set-order invariant).
 *
 * SCALABLE in ONE cooperative launch across all CUs. A single-WG radix is LDS-atomic-bound
 * (~500us@128k on 1/256 CUs); splitting the passes into separate kernel launches is launch-overhead
 * bound (~300us for 16 in-order dispatches). Instead this is ONE launch of exactly gridDim==nCU
 * workgroups (all co-resident) that grid-synchronise between passes via a manual sense-reversing
 * barrier over global counters (the persistent-interp co-residency trick). Per pass: every WG
 * histograms its grid-strided slice into an LDS 256-bin table, flushes to a global histogram, then
 * ALL WGs read the completed histogram and independently compute the SAME boundary digit (no single
 * decider) — so prefix/k_rem stay in registers, no per-pass HBM state. Full HBM bandwidth (256 CUs)
 * over the 7 passes => a few us of work + ~21 barriers. The kernel leaves gHist/gCtl clean for
 * re-launch. gCtl[0]=arrive gCtl[1]=generation gCtl[2]=emit-slot (host zeroes gHist+gCtl once). */
__device__ __forceinline__ unsigned long long dsa_pack_key(float sc, unsigned t, unsigned len) {
    unsigned sb;
    __builtin_memcpy(&sb, &sc, 4);
    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
    return ((unsigned long long)sb << 20) | (unsigned long long)((len - 1 - t) & 0xFFFFFu);
}
/* BYTE-ALIGNED packed key for the fewer-passes fast path: score in the TOP 4 bytes, index tie-break
 * in the LOW 3 bytes (24 bits; len < 2^24 always). Same monotone ordering (score desc, index asc) as
 * dsa_pack_key, so it selects the IDENTICAL top_k set — but now the 32-bit score occupies EXACTLY the
 * first 4 radix bytes (passes 0..3) and the index the last 3 (passes 4..6), so after 4 score passes
 * the score threshold is fully resolved and, absent an exact-score tie at the boundary, the selection
 * is decided in 4 passes instead of 7 (the index passes only resolve genuine score ties). */
__device__ __forceinline__ unsigned long long dsa_pack_key_a(float sc, unsigned t, unsigned len) {
    unsigned sb;
    __builtin_memcpy(&sb, &sc, 4);
    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
    return ((unsigned long long)sb << 24) | (unsigned long long)((len - 1 - t) & 0xFFFFFFu);
}
/* grid-wide barrier over co-resident WGs (requires gridDim <= resident capacity). Only lane 0
 * participates. NO threadfence: the histogram is communicated exclusively through L2-coherent
 * ATOMICS (atomicAdd commits to L2 before the instruction retires, and this WG's histogram
 * atomicAdds precede its arrival atomicAdd in program order), so the atomic counter handshake alone
 * orders writers-before-readers. Spun on with a plain volatile load. Device-wide fences on 256 CUs
 * cost ~26us each and dominate otherwise. */
__device__ __forceinline__ void dsa_grid_sync(unsigned* ctl, unsigned nwg) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = (volatile unsigned*)&ctl[1];
        const unsigned g = *gen;
        const unsigned a = atomicAdd(&ctl[0], 1u) + 1u;
        if (a == nwg) { atomicExch(&ctl[0], 0u); *gen = g + 1u; } /* last: reset count, bump gen */
        else { while (*gen == g) { /* spin */ } }
    }
    __syncthreads();
}
/* EXPERIMENT (barrier latency): the release of the generation counter above is a PLAIN volatile
 * store (`*gen = g+1`) with no fence, so its visibility to the spinning WGs on other CUs waits on
 * L2 write-back timing — measured ~20us/barrier even at 4 co-resident WGs (contention-free). This
 * variant releases and polls `gen` through L2-COHERENT ATOMICS (atomicExch to publish, atomicAdd(&,0)
 * to poll), which commit to / read from L2 immediately, so the sense-reversal propagates at L2
 * latency instead of write-back latency. Ordering is UNCHANGED: the histogram is still communicated
 * only via L2-coherent atomicAdds, ordered writers-before-readers by the arrival-count handshake;
 * making `gen` atomic only accelerates the handshake signal itself. ATOMIC selects the variant. */
template <bool ATOMIC>
__device__ __forceinline__ void dsa_grid_sync_t(unsigned* ctl, unsigned nwg) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (ATOMIC) {
            const unsigned g = atomicAdd(&ctl[1], 0u);            /* coherent read of gen */
            const unsigned a = atomicAdd(&ctl[0], 1u) + 1u;
            if (a == nwg) { atomicExch(&ctl[0], 0u); atomicExch(&ctl[1], g + 1u); }
            else { while (atomicAdd(&ctl[1], 0u) == g) { /* spin on coherent gen */ } }
        } else {
            volatile unsigned* gen = (volatile unsigned*)&ctl[1];
            const unsigned g = *gen;
            const unsigned a = atomicAdd(&ctl[0], 1u) + 1u;
            if (a == nwg) { atomicExch(&ctl[0], 0u); *gen = g + 1u; }
            else { while (*gen == g) { /* spin */ } }
        }
    }
    __syncthreads();
}
/* Radix geometry (perf floor 2 — RE-DIAGNOSED). 8-bit / 7-pass / 256-bin is optimal for this radix
 * (wider digits LOST — the bigger histogram's flush + read-back traffic dwarfs the barriers saved).
 * The floor was WRONGLY attributed to the grid barrier: a DSA_SELWG sweep shows reducing the WG count
 * (less barrier contention) does NOT reduce the select time, and swapping the barrier's generation
 * signal to L2-coherent atomics (dsa_grid_sync_t<true>) is nearly neutral on its own — so the barrier
 * is NOT the ~25us/pass cost. The real cost was the per-pass BOUNDARY SCAN: a serial lane-0 dependent
 * chain of up to 256 global-atomic reads (each an L2 round-trip), run redundantly by every WG. The
 * three levers, all EXACT, that d_index_select_coop's template flags select (defaults = ON, shipped):
 *   PAR_SCAN  — read the completed 256-bin histogram into LDS with PARALLEL coherent atomics, then
 *               accumulate serially over LDS: 256 reads in flight vs a 256-deep dependency chain.
 *   FAST      — a byte-aligned key (dsa_pack_key_a: score in the top 4 bytes) that resolves the whole
 *               32-bit score threshold in 4 passes; absent an exact-score tie at the boundary the
 *               selection is decided there (5 barriers, not 8), falling back to the 3 index passes
 *               only to split a genuine tie by lowest index (correctness-preserving early-out).
 *   ATOMIC_SYNC — coherent-atomic generation counter (a small residual win once the scan is parallel).
 * Measured (dsa_gather_bench, tp8 nh8, 32-WG, MI350X gfx950): select 178->58 (8k), 201->58 (32k),
 * 146->70 (128k), 220->85us (256k) — ~2.1-3.4x, set == CPU radix top-k EXACT incl. a tie-stress. */
#define SEL_DIGIT 8u
#define SEL_NB    (1u << SEL_DIGIT) /* 256 bins/pass */
#define SEL_NPASS 7u                /* 4 score bytes + 3 index bytes = 56-bit byte-aligned key */
/* Defaults = the SHIPPED fast configuration (measured ~2.1-3.4x over the flat 7-pass/serial-scan
 * baseline, EXACT at every ctx incl. tie-stress): coherent-atomic grid barrier + PARALLEL histogram
 * read-back + the byte-aligned-key fewer-passes fast path. The interp calls this unqualified so it
 * inherits the fast path with no ABI change; the bench pins <false,false,false> for the baseline. */
template <bool ATOMIC_SYNC = true, bool PAR_SCAN = true, bool FAST = true>
__device__ void d_index_select_coop(int* __restrict__ idx, const float* __restrict__ Score,
                                    unsigned len, unsigned top_k, unsigned* __restrict__ gHist,
                                    unsigned* __restrict__ gCtl, unsigned nwg,
                                    unsigned* lh /* [SEL_NB] LDS */, unsigned* red /* [2] LDS */) {
    const auto* const Sc = as_glob(Score);
    int* const ib = as_glob(idx);
    unsigned* const Hg = as_glob(gHist);
    unsigned* const Cg = as_glob(gCtl);
    const unsigned bid = blockIdx.x, tid = threadIdx.x;
    if (bid == 0 && tid == 0) Cg[2] = 0u; /* reset emit slot */
    /* clear all SEL_NPASS per-pass histograms cooperatively (idempotent), then barrier. */
    for (unsigned i = bid * PLOW_THREADS + tid; i < SEL_NPASS * SEL_NB; i += nwg * PLOW_THREADS)
        atomicExch(&Hg[i], 0u);
    dsa_grid_sync_t<ATOMIC_SYNC>(Cg, nwg);
    unsigned long long prefix = 0, himask = 0;
    unsigned k_rem = top_k;
    for (unsigned pass = 0; pass < SEL_NPASS; pass++) { /* MSB-first, 13-bit digits over 52 bits */
        const unsigned sh = (SEL_NPASS - 1u - pass) * SEL_DIGIT;
        unsigned* const Hp = Hg + (size_t)pass * SEL_NB; /* this pass's own histogram */
        for (unsigned i = tid; i < SEL_NB; i += PLOW_THREADS) lh[i] = 0u;
        __syncthreads();
        for (unsigned t = bid * PLOW_THREADS + tid; t < len; t += nwg * PLOW_THREADS) {
            const unsigned long long key = FAST ? dsa_pack_key_a(Sc[t], t, len) : dsa_pack_key(Sc[t], t, len);
            if ((key & himask) == prefix) atomicAdd(&lh[(unsigned)((key >> sh) & (SEL_NB - 1))], 1u);
        }
        __syncthreads();
        for (unsigned i = tid; i < SEL_NB; i += PLOW_THREADS)
            if (lh[i]) atomicAdd(&Hp[i], lh[i]);
        dsa_grid_sync_t<ATOMIC_SYNC>(Cg, nwg); /* all WGs contributed to Hp */
        /* boundary digit: top-down cumulative over the 256 bins until it crosses k_rem. Two forms:
         * PAR_SCAN reads the histogram into LDS in parallel then accumulates over LDS (the shipped
         * default — the serial global-atomic form below is the measured bottleneck); the else branch
         * is the original serial lane-0 chain of coherent global-atomic reads, kept as the baseline. */
        if (PAR_SCAN) {
            /* PARALLEL coherent read-back: the serial lane-0 scan above is a dependent chain of up to
             * 256 global-atomic reads (each an L2 round-trip) — latency-bound and run redundantly by
             * every WG. Instead all threads atomic-read the (now-complete) 256-bin Hp into LDS in
             * parallel, then only the tiny top-down accumulate runs serially over LDS. Same coherent
             * reads, but 256 in flight instead of 256 in a dependent chain. */
            for (unsigned i = tid; i < SEL_NB; i += PLOW_THREADS) lh[i] = atomicAdd(&Hp[i], 0u);
            __syncthreads();
            if (tid == 0) {
                unsigned acc = 0, dsel = 0, bnd = 0;
                for (int d = (int)SEL_NB - 1; d >= 0; d--) {
                    const unsigned hd = lh[d];
                    if (acc + hd >= k_rem) { dsel = (unsigned)d; bnd = hd; break; }
                    acc += hd;
                }
                red[0] = dsel;
                red[1] = acc;
                if (FAST) red[2] = bnd; /* count in the boundary bin (the tied group at this digit) */
            }
            __syncthreads();
        } else {
            if (tid == 0) {
                unsigned acc = 0, dsel = 0, bnd = 0;
                for (int d = (int)SEL_NB - 1; d >= 0; d--) {
                    const unsigned hd = atomicAdd(&Hp[d], 0u);
                    if (acc + hd >= k_rem) { dsel = (unsigned)d; bnd = hd; break; }
                    acc += hd;
                }
                red[0] = dsel;
                red[1] = acc;
                if (FAST) red[2] = bnd;
            }
            __syncthreads();
        }
        k_rem -= red[1];
        prefix |= ((unsigned long long)red[0] << sh);
        himask |= ((unsigned long long)(SEL_NB - 1) << sh);
        /* FAST fewer-passes exit: after the 4 SCORE bytes (pass 3) the 32-bit score threshold is
         * fully resolved. `k_rem` now counts positions still needed from the boundary bin, whose
         * size is red[2] (all share the EXACT boundary score). If they match, the whole tied group
         * is selected and the 3 index passes would emit the same set => break at 4 passes. Only a
         * genuine score tie that must be split by index (red[2] > k_rem) continues. All WGs read the
         * same red[] => the branch is uniform => no barrier divergence. Exactness is unchanged. */
        if (FAST && pass == 3u && red[2] == k_rem) break;
    }
    /* prefix == top_k-th largest key (unique keys => #{key>=prefix} == top_k). */
    for (unsigned t = bid * PLOW_THREADS + tid; t < len; t += nwg * PLOW_THREADS) {
        const unsigned long long key = FAST ? dsa_pack_key_a(Sc[t], t, len) : dsa_pack_key(Sc[t], t, len);
        if (key >= prefix) {
            const unsigned slot = atomicAdd(&Cg[2], 1u);
            if (slot < top_k) ib[slot] = (int)t;
        }
    }
}

/* W_uv fold: o[b][h][v] = sum_l  O_latent[b][h][l] * W_uv[h][l][v].     [DEEPSEEK-MLA]
 * The per-query epilogue of MLA (§2.5): folds the merged latent accumulator down to
 * v_head_dim.  An ordinary small O(n_q) GEMV — the placeholder for a plain GEMV /
 * PLOW_DOP_O_UV_FOLD.  W_uv is [n_head][DK][V] (l-major); the reduction runs l = 0..DK
 * in order so a golden that mirrors it stays bit-exact-to-reference. */
template <int DK>
__device__ void d_o_uv_fold(bf16* __restrict__ O_, const bf16* __restrict__ Olat_,
                            const bf16* __restrict__ Wuv_, unsigned n_batch, unsigned n_head,
                            unsigned V, unsigned slice, unsigned nblk, bf16* lds) {
    auto* const O = as_glob(O_);
    const auto* const Olat = as_glob(Olat_);
    const auto* const Wuv = as_glob(Wuv_);
    const unsigned n_work = n_batch * n_head;
    const unsigned tid = threadIdx.x;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head, b = w / n_head;
        const auto* ol = Olat + (size_t)(b * n_head + h) * DK;
        for (unsigned i = tid; i < DK; i += PLOW_THREADS) lds[i] = ol[i];
        __syncthreads();
        const auto* wv = Wuv + (size_t)h * DK * V;
        for (unsigned v = tid; v < V; v += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned l = 0; l < DK; l++) acc += bf2f(lds[l]) * bf2f(wv[(size_t)l * V + v]);
            O[(size_t)(b * n_head + h) * V + v] = f2bf(acc);
        }
        __syncthreads();
    }
}

/* FUSED MLA merge + W_uv fold.                                            [DEEPSEEK-MLA]
 *
 * Kills the SEPARATE O(nsplit) d_flash_merge pass and its Olat HBM round-trip: one kernel
 * online-softmax-merges the nsplit latent partials (Opart/mlpart) into olat[DK] in LDS, then
 * folds olat @ W_uv[head] -> o[head][V] straight from LDS. Two structural wins over merge+fold:
 *   - no Olat[n_head][DK] write-then-read round-trip (the merge's output stays resident);
 *   - one kernel launch + one dependency gate instead of two (the coarse merge->fold stall goes).
 *
 * OCCUPANCY: the standalone merge/fold run on only n_batch*n_head workgroups (nh_l — 8 at tp=8),
 * starving the 256-CU chip. Here work is split over (b, head, V-tile): `VT` output columns per
 * workgroup => n_batch*n_head*ceil(V/VT) workgroups. Each v-tile WG re-merges olat in LDS
 * (redundant across a head's v-tiles, but the merge is a cheap nsplit*DK read of the compact f32
 * partials); the fold — the 2 MB W_uv read + DK reduction that dominated the standalone fold — is
 * fully spread. VT trades merge-redundancy against fold-occupancy; swept empirically.
 *
 * Correctness: the split reduction equals the sequential attention sum for any nsplit
 * (Plow.SplitK.split_k_two_way; online softmax is associative). olat is kept f32 in LDS (the
 * standalone path rounds it to bf16 before the fold) — strictly MORE accurate, within the
 * mla_test tolerance (rel_rms < 5e-3). */
template <int DK, int VT>
__device__ void d_mla_merge_fold(bf16* __restrict__ O_, const float* __restrict__ Opart_,
                                 const float* __restrict__ mlpart_, const bf16* __restrict__ Wuv_,
                                 unsigned n_batch, unsigned n_head, unsigned V, unsigned nsplit,
                                 unsigned slice, unsigned nblk, float* olds /* DK floats */) {
    auto* const O = as_glob(O_);
    const auto* const Opart = as_glob(Opart_);
    const auto* const mlpart = as_glob(mlpart_);
    const auto* const Wuv = as_glob(Wuv_);
    const unsigned tid = threadIdx.x;
    const unsigned vtiles = (V + VT - 1) / VT;
    const unsigned n_work = n_batch * n_head * vtiles;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned vt = w % vtiles;
        const unsigned bh = w / vtiles;
        const unsigned h = bh % n_head, b = bh / n_head;
        const auto* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;

        /* global max / sum over the nsplit partials (online-softmax LSE combine). */
        float gm = FA_NEG_INF;
        for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s * 2]);
        float gl = 0.0f;
        for (unsigned s = 0; s < nsplit; s++) {
            if (ml[s * 2] == FA_NEG_INF) continue;
            gl += ml[s * 2 + 1] * FA_EXP(ml[s * 2] - gm);
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

        /* merge the DK-wide latent into LDS (rescaled, normalized). */
        const auto* opb = Opart + (size_t)(b * n_head + h) * nsplit * DK;
        for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned s = 0; s < nsplit; s++) {
                if (ml[s * 2] == FA_NEG_INF) continue;
                acc += opb[(size_t)s * DK + d] * FA_EXP(ml[s * 2] - gm);
            }
            olds[d] = acc * inv;
        }
        __syncthreads();

        /* fold this workgroup's V-tile: o[v] = sum_l olat[l] * W_uv[h][l][v]. */
        const auto* wv = Wuv + (size_t)h * DK * V;
        const unsigned v0 = vt * VT, v1 = (v0 + VT < V) ? (v0 + VT) : V;
        for (unsigned v = v0 + tid; v < v1; v += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned l = 0; l < DK; l++) acc += olds[l] * bf2f(wv[(size_t)l * V + v]);
            O[(size_t)(b * n_head + h) * V + v] = f2bf(acc);
        }
        __syncthreads();
    }
}

#endif /* PLOW_OP_ATTENTION_H */
