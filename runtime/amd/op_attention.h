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
 * work that BKV=64 reorganises but does not reduce. See the design notes.
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

/* SOFTMAX RECIPROCAL. `1.0f / l` lowers to the correctly-rounded IEEE sequence -- v_div_scale
 * x2, v_rcp, a Newton chain, v_div_fmas, v_div_fixup, ~12 instructions -- and this softmax does
 * not need that. `l` is a sum of exp() terms taken after the running-max subtraction, so it is
 * >= 1 and bounded by the KV span: never denormal, never inf, and the zero case is already
 * guarded at every call site. v_rcp_f32 is 1 ULP against the divide's 0.5, feeding a bf16 store
 * with 8 mantissa bits. It is the same builtin `sigmoid`/`tanh` in amd_common.h already use.
 *
 * Counted in the shipped gfx942 objects before this: 135 IEEE divides in interp_decode_gq, 99
 * in interp_prefill_gq, 48 in interp_flash_gq.
 *
 * CDNA3 ONLY, and that is a caution rather than a claim about the hardware -- v_rcp_f32 exists
 * on both. gfx950 is a shipped, validated target and this changes its last bit; there is no
 * gfx950 in this machine to re-run its goldens on, so it stays bit-identical there until
 * someone measures it. Flip FA_FAST_RCP to 1 on that box and the note above is the argument. */
#ifndef FA_FAST_RCP
#define FA_FAST_RCP (!PLOW_CDNA4)
#endif
#if FA_FAST_RCP
#define FA_RECIP(x) __builtin_amdgcn_rcpf(x)
#else
#define FA_RECIP(x) (1.0f / (x))
#endif

/* MERGE-UNROLL4 gate (see the [MERGE-UNROLL4] note at d_flash_merge). The 4-banked accumulators
 * REASSOCIATE the f32 split sums, so the merged output moves in the last ulp. Same policy as
 * FA_FAST_RCP above: CDNA3 only, gfx950 stays bit-identical until someone with that hardware
 * re-runs its goldens and flips this to 1 there. */
#ifndef FA_MERGE_UNROLL4
#define FA_MERGE_UNROLL4 (!PLOW_CDNA4)
#endif

/* LAZY RESCALE. The online-softmax rescale multiplies every O-accumulator element by
 * corr = exp(m_old - m_new) on EVERY KV tile — 16 exps + NDT*16 v_mul per lane per tile
 * (128 muls at DCH=256) — but once the running max stops moving, m_new == m_old and
 * corr == 1.0f EXACTLY (exp(0) is exact), so the whole rescale is a no-op that still
 * costs issue slots. At 1 wave/SIMD (the 4-wave flash object) that VALU is fully
 * exposed against the MFMAs: the BKV64 study concluded this tile is bound by
 * LDS/MFMA/softmax throughput, and this deletes softmax work instead of reorganising it.
 *
 * The skip is a WAVE vote (`__ballot`): every lane checks whether ANY of its 16
 * accumulator rows saw a new max this tile; only if no lane did is the corr/rescale
 * block skipped. There is no barrier inside the skipped region, so wave-level
 * divergence between waves is safe. BIT-IDENTICAL by construction: the skipped path is
 * exactly the corr == 1.0f path (and the m == -inf virgin state has l == 0 and
 * o == 0, which the skip leaves untouched — the same values the corr = 0 full path
 * writes). Default OFF until measured. */
#ifndef FA_LAZY_RESCALE
#define FA_LAZY_RESCALE 0
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
    /* FP8 KV IN THE PREFETCH PATH. This branch had none, and it is the only path the 4-wave
     * flash object takes (it is the object built -DFA_DBUF=1). `ld_glob8` on a `const bf16*`
     * reads 16 BYTES for 8 elements; the e4m3 cache is 1 byte per element, so the prefetch read
     * two rows' worth per row, ran off the end of the cache, and never applied the row scale.
     * That is the fp8-KV prefill memory fault: `d_flash_prefill<D,true>` compiles perfectly
     * under FA_DBUF=1 and silently reads the wrong dtype.
     *
     * The #else (synchronous) branch below has always handled FP8KV, which is why the same
     * packet runs clean on the 8-wave prefill object (FA_DBUF=0) and faults the moment
     * segmented dispatch sends its class-4 segments to the object built for them. An arm that
     * exists, is correct in one branch of a #if, and is silently absent from the other.
     *
     * Dequant is identical to the #else branch: fp8 -> bf16 times the per-row f32 scale, so
     * Ksm/Vsm hold exactly the same bf16 the bf16 path would and the MFMA is unchanged. */
    /* The bf16 expressions below are UNCHANGED, character for character, from before the FP8KV
     * branch was added — `if constexpr (!FP8KV)` guards them rather than an `else` on a
     * refactored address, because hoisting the row index (arithmetically identical) moved the
     * bf16 flash object's spill 228 -> 214. That is a codegen change to a shipped object bought
     * for nothing, in the one object whose spill is a deliberate, measured trade. */
#define FA_DB_FP8(DST, PTR, SCALE, ELT)                                                          \
    {                                                                                             \
        const size_t row_ = (size_t)hkv * kv_stride + (kv & kv_mask);                             \
        const bf16v8 dv_ = fp8v8_to_bf16v8(                                                       \
            ld_glob_fp8v8((const unsigned char*)(PTR) + row_ * D + (ELT)));                       \
        const float s_ = (SCALE)[row_];                                                           \
        _Pragma("unroll") for (int j_ = 0; j_ < 8; j_++) (DST)[j_] = f2bf(bf2f(dv_[j_]) * s_);     \
    }
#define FA_DB_LOAD(KVB)                                                                          \
    _Pragma("unroll") for (int it = 0; it < KPT; it++) {                                         \
        const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                            \
        const unsigned kv = (KVB) + e / D;                                                       \
        if constexpr (FP8KV) {                                                                    \
            if (kv < n_kv) FA_DB_FP8(nk[it], K, k_scale, e % D) else nk[it] = bf16v8_zero();      \
        } else {                                                                                  \
        nk[it] = (kv < n_kv) ? ld_glob8(as_glob(K) + ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + e % D) \
                             : bf16v8_zero();                                                     \
        }                                                                                         \
    }                                                                                            \
    _Pragma("unroll") for (int it = 0; it < VPT; it++) {                                         \
        const unsigned e = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                            \
        const unsigned kv = (KVB) + e / FA_DC;                                                   \
        if constexpr (FP8KV) {                                                                    \
            if (kv < n_kv) FA_DB_FP8(nv[it], V, v_scale, d_off + e % FA_DC) else nv[it] = bf16v8_zero(); \
        } else {                                                                                  \
        nv[it] = (kv < n_kv) ? ld_glob8(as_glob(V) + ((size_t)hkv * kv_stride + (kv & kv_mask)) * D + d_off + e % FA_DC) \
                             : bf16v8_zero();                                                     \
        }                                                                                         \
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
                            s = plow_mfma_bf16_32x32(qf[q], kfrag, s);
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
#if FA_LAZY_RESCALE
                    /* [FA_LAZY_RESCALE] see the knob note: skip corr/rescale when no lane's
                     * running max moved this tile — the skipped block is exactly corr == 1.
                     * No rm[] array: the vote recomputes half_wave_max and the full path
                     * repeats it, so nothing new stays live across the branch (a held
                     * rm[16] tripled the flash object's scratch spill, 217 -> 620 B). */
                    bool upd = false;
#pragma unroll
                    for (int i = 0; i < 16; i++) upd |= half_wave_max(p[i]) > m_st[i];
                    if (__ballot(upd) == 0ull) {
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const float pe =
                                (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - m_st[i]);
                            l_st[i] += half_wave_sum(pe);
                            p[i] = pe;
                        }
                    } else
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const float rmax = half_wave_max(p[i]);
                            const float mnew = fmaxf(m_st[i], rmax);
                            const float corr =
                                (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                            const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
                            l_st[i] = l_st[i] * corr + half_wave_sum(pe);
                            m_st[i] = mnew;
                            p[i] = pe;
#pragma unroll
                            for (int t = 0; t < NDT; t++) oacc[t][i] *= corr;
                        }
#else
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
#endif

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
                                plow_mfma_bf16_32x32(pf, vf, oacc[t]);
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
                                s[n] = plow_mfma_bf16_32x32(qf[q], kfrag, s[n]);
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
                     * element, ONE rescale of the O accumulator per BKV rows.
                     * (FA_LAZY_RESCALE applies only to the NKT==1 branch above: this D=128
                     * arm is register-tight and the extra live state tripled the flash
                     * object's spill when the skip was drafted here too.) */
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
                                plow_mfma_bf16_32x32(pf, vf, oacc[t]);
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
/* v_rcp_f32, not the IEEE divide. `1.0f / l` lowers to the correctly-rounded sequence --
                     * v_div_scale x2, v_rcp, a Newton chain, v_div_fmas, v_div_fixup, ~12
                     * instructions -- for a reciprocal this softmax does not need that precisely.
                     * `l` is a sum of exp() terms after the running-max subtraction, so it is >= 1
                     * and bounded by the KV span: never denormal, never inf, and the zero case is
                     * already guarded. v_rcp_f32 is 1 ULP against the divide's 0.5, into a bf16
                     * store with 8 mantissa bits. Counted in the shipped objects: 135 IEEE divides
                     * in interp_decode_gq, 99 in interp_prefill_gq, 48 in interp_flash_gq. */
                    const float inv = (l_st[i] > 0.0f) ? FA_RECIP(l_st[i]) : 0.0f;
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
                        st_act1(&orow[t * MFMA_N], (bf16)(u >> 16));
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
                for (int t = 0; t < NDT; t++)
                    st_act<float>(&op[d_off + t * MFMA_N + accn], oacc[t][i]);
                /* 32 lanes share this qi (they hold different d), so exactly one writes
                 * (m, l) -- and only once, not once per output chunk. */
                if (ch == 0 && accn == 0) {
                    float* ml = mlpart + ((size_t)(qi * n_head + h) * nsplit + sp) * 2;
                    st_act<float>(&ml[0], m_st[i]);
                    st_act<float>(&ml[1], l_st[i]);
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
/* Interleave the K-phase row->wave map instead of blocking it. Default OFF (byte-identical);
 * see the note at the map itself for why it matters when a split is shorter than the tile. */
#ifndef FA_DEC_ILV
#define FA_DEC_ILV 0
#endif
/* Bound the flash-decode softmax reductions by the split's LIVE row count instead of the full
 * FA_DEC_TILE. Exactly equivalent (dead slots are -inf / 0); default OFF. */
#ifndef FA_DEC_LIVE
#define FA_DEC_LIVE 0
#endif
/* Flash-decode ablation: 1 = skip the V accumulation. Probe only, results wrong. */
#ifndef FA_ABL
#define FA_ABL 0
#endif

/* ==========================================================================================
 * CLOSED: the "MLA decode is wrong at eight waves on gfx942" defect was a TEST-HARNESS
 * WAVE-COUNT MISMATCH, not a kernel fault. d_flash_mla_decode is correct at 4 and at 8.
 *
 * This block previously carried an OPEN DEFECT and a #warning telling anyone building a
 * GLM/MLA object to drop to four waves. That instruction was wrong and is removed.
 *
 * ROOT CAUSE, reproduced on demand. The device object and the host harness resolve
 * PLOW_WG_THREADS in two SEPARATE compilations, and scripts/build_mla.sh passed the wave
 * count to neither. Build the object at eight waves and run it under a host built at four
 * and the harness launches 256 threads at a 512-thread kernel. That is a LEGAL dispatch --
 * waves 4..7 simply never exist -- so:
 *
 *   - `if (wave < GF)` never runs for the missing waves, leaving hmax[]/hsum[] partly unwritten;
 *   - osm[] row-groups 4..7 are never stored, because `grp` only ever takes values 0..3;
 *   - and the output fold sums all NG of them anyway: `for (gg=0; gg<NG; gg++) acc += osm[...]`,
 *     with NG = PLOW_WAVES = 8 compiled into the OBJECT.
 *
 * So the fold reads uninitialised LDS. MEASURED, that mismatch reproduces the recorded
 * signature exactly: dense n_head=8 ctx=4096 nsplit=1 at max rel 0.9238 (recorded: 0.97),
 * every nsplit>1 case at 1e24..1e35 against |O|max ~30, and `rms nan`.
 *
 * The reverse mismatch (256-thread object, 512-thread launch) is NOT silent -- hardware
 * rejects it with HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS. Only this direction corrupts.
 *
 * WITH BOTH HALVES BUILT AT THE SAME WIDTH, on MI300X against the 12-case mla_ref.rs fixture:
 *
 *   -DPLOW_WG_WAVES=4 -DGM_DBUF=1 -DGM_BM=64  -DGM_BN=128    MLA CORRECT (0 failures)
 *   -DPLOW_WG_WAVES=8 -DGM_DBUF=1 -DGM_BM=192 -DGM_BN=256    MLA CORRECT (0 failures)
 *
 * All twelve decode cases agree to max rel 0.0028 of |O|max, and the two wave counts produce
 * IDENTICAL numbers case for case -- not merely both-within-tolerance. The bf16 tiled-MFMA
 * prefill (22 checks) and the fp8-latent prefill (4 checks) pass at both. Repeated on three
 * GPUs, twice each: 0 failures every time.
 *
 * This is why the old note's elimination rounds all came back negative, and why its own
 * differential probe found the four- and eight-wave kernels AGREE: the fault was never in
 * this function. It cost a multi-round hunt through the arena, the index maps, the
 * head-fusion factor, FA_DEC_KL, races and the merge.
 *
 * MADE UNREPEATABLE, both ends: `plow_probe_wg_threads` (test_kernels.hip) has the object
 * report its own width and the harness aborts on disagreement before running a single case,
 * and build_mla.sh now passes one $PLOW_WG_WAVES to both compiles. Note also that `dOp`/`dMl`
 * in mla_gfx950_test.c are allocated per case and never zeroed, which is what turned unwritten
 * slots into a previous case's values rather than an obvious zero.
 * ========================================================================================== */

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

/* LANES PER K ROW in the flash-decode score phase. See [K-PHASE-KL8] in d_flash_decode for the
 * measurement that picks 8; 1 restores the historical row-per-lane map. Must divide 64, and KL*8
 * must divide the head dim.
 *
 * DEFAULT 8 ON CDNA3, 1 ON CDNA4, and the asymmetry is deliberate. The number that picks 8 is a
 * gfx942 measurement of the gfx942 memory system (decode_bw_probe.hip, 304 CU, 16 GB) -- CDNA4
 * has a different L1, a different LDS budget and a different wave-per-SIMD budget, so the same
 * probe has to be re-run there before the map changes under a shipped, tuned gfx950 object. Until
 * it is, the CDNA4 arm compiles the historical map and its objects stay byte-identical. */
#ifndef FA_DEC_KL
#if PLOW_CDNA4
#define FA_DEC_KL 1
#else
#define FA_DEC_KL 8
#endif
#endif
/* How many of the KPASS row-passes to unroll. Every unrolled pass holds another D/(KL*8) loads
 * in flight -- more memory-level parallelism, more live registers. */
#ifndef FA_DEC_KUNROLL
#define FA_DEC_KUNROLL 8
#endif
/* Same knob for the MLA latent dot, which needs its OWN value: at GLM_MLA_GF=4 the fully unrolled
 * DK=512 chunk loop keeps GF q-fragments live per chunk, and that is what put the standalone MLA
 * decode at 256 VGPR with 1836 bytes/lane of scratch spill. */
#ifndef FA_MLA_KUNROLL
#define FA_MLA_KUNROLL 1
#endif
#ifndef FA_MLA_KDU
#if PLOW_CDNA4
#define FA_MLA_KDU 64 /* >= any trip count: FULL unroll, the historical CDNA4 lowering */
#else
#define FA_MLA_KDU 4
#endif
#endif

/* MINIMUM WAVES PER SIMD the STANDALONE decode kernel is allocated for (amdgpu-waves-per-eu).
 *
 * The interpreter's decode object is one inlined kernel allocated for its worst op, so it cannot
 * carry a per-op register budget; a standalone `gemma_flash_decode_*` can. At the shipped 2 the
 * allocator spends 164 VGPR and the CU holds ONE workgroup (2 waves/SIMD) even though the 22.6 KiB
 * LDS arena would fit three. Raising it to 4 caps VGPR at 128 and lets a second workgroup in.
 * Defaults to the shipped value so nothing changes unless the build asks. */
#ifndef FA_DEC_WPEU
#define FA_DEC_WPEU (PLOW_WAVES / 4)
#endif

#undef FA_LOAD_QF

/* NRF (norm-rope fold, template arm): the three HeadNormRope packets between q/k/v and this op
 * computed into THIS op — one whole serial gate level off the decode chain. Under NRF the `Q`
 * param carries the RAW q projection (act.qg) and `nrf_kg`/`nrf_vg` the raw k/v projections;
 * `K`/`V` stay the caches (now WRITTEN here too).
 *
 *   - Q staging: one WAVE per query head replicates d_headnorm_rope element-for-element (same
 *     lane->element map, same wave_sum order, same rsqrt, same half-split RoPE pairing, same
 *     f2bf rounding) into qsm — the bytes are the bytes the deleted packet would have written
 *     to HBM and this staging would have re-read.
 *   - Current-token K/V row: the ONE (hg, split) item whose range covers qpos computes
 *     norm(γk)+RoPE for K and the weightless RMS for V (Gemma norms V; skip via nrf_skip for
 *     Llama-style models), STORES both rows to the cache at slot qpos & kv_mask, then takes an
 *     agent-scope acq_rel fence (s_waitcnt + buffer_inv — the gate's own acquire) so the
 *     UNTOUCHED tile loop below reads them back fresh. Reading back through the ordinary loop,
 *     rather than special-casing the row, is what keeps the softmax reduction ORDER — and
 *     therefore the numerics — identical to the unfused program. MQA (several head-groups per
 *     kv head) makes several owners write the same bytes: idempotent.
 *
 * A TEMPLATE arm, not a runtime branch, on the d_gemv_glu_fp8_nrn lesson: a runtime branch
 * around staging cost +0.5 ms on packets that never took it. NRF=false instantiations are
 * byte-equivalent to before. FP8KV composes with NRF only as !NRF (the emitter refuses). */
/* The split (m, l) reduction both merge kernels open with: global max, then the LSE-combined
 * denominator.                                                              [MERGE-ML-UNROLL4]
 *
 * `ml` is UNIFORM across the workgroup, so this lowers to SCALAR loads -- and a scalar loop with a
 * runtime trip count is one s_load per iteration, each behind its own `s_waitcnt lgkmcnt(0)`. At
 * nsplit=32 that was 64 serial round trips before the first Opart byte was even requested, on an
 * op that at batch 1 owns 16 of 304 CUs and has no co-resident wave to hide any of it. Four in
 * flight per pass; the tails keep any nsplit legal. `gm` is returned through the reference so the
 * per-split weights below can be recomputed from the same value. */
__device__ __forceinline__ float fa_merge_ml(const float* __restrict__ ml, unsigned nsplit,
                                             float& gm_out) {
#if !FA_MERGE_UNROLL4
    /* The pre-UNROLL4 serial merge, kept bit-identical for CDNA4 (see FA_MERGE_UNROLL4). */
    float gm = FA_NEG_INF;
    for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s * 2]);
    gm_out = gm;
    float gl = 0.0f;
    for (unsigned s = 0; s < nsplit; s++) {
        if (ml[s * 2] == FA_NEG_INF) continue;
        gl += ml[s * 2 + 1] * FA_EXP(ml[s * 2] - gm);
    }
    return gl;
#else
    float m0 = FA_NEG_INF, m1 = FA_NEG_INF, m2 = FA_NEG_INF, m3 = FA_NEG_INF;
    unsigned s = 0;
    for (; s + 4 <= nsplit; s += 4) {
        m0 = fmaxf(m0, ml[(s + 0) * 2]); m1 = fmaxf(m1, ml[(s + 1) * 2]);
        m2 = fmaxf(m2, ml[(s + 2) * 2]); m3 = fmaxf(m3, ml[(s + 3) * 2]);
    }
    for (; s < nsplit; s++) m0 = fmaxf(m0, ml[s * 2]);
    const float gm = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    gm_out = gm;

    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    for (s = 0; s + 4 <= nsplit; s += 4) {
        const float a = ml[(s + 0) * 2], b = ml[(s + 1) * 2];
        const float c = ml[(s + 2) * 2], d = ml[(s + 3) * 2];
        l0 += ml[(s + 0) * 2 + 1] * ((a == FA_NEG_INF) ? 0.0f : FA_EXP(a - gm));
        l1 += ml[(s + 1) * 2 + 1] * ((b == FA_NEG_INF) ? 0.0f : FA_EXP(b - gm));
        l2 += ml[(s + 2) * 2 + 1] * ((c == FA_NEG_INF) ? 0.0f : FA_EXP(c - gm));
        l3 += ml[(s + 3) * 2 + 1] * ((d == FA_NEG_INF) ? 0.0f : FA_EXP(d - gm));
    }
    for (; s < nsplit; s++) {
        const float a = ml[s * 2];
        l0 += ml[s * 2 + 1] * ((a == FA_NEG_INF) ? 0.0f : FA_EXP(a - gm));
    }
    return (l0 + l1) + (l2 + l3);
#endif /* FA_MERGE_UNROLL4 */
}

template <int D, int GF, bool FP8KV = false, bool NRF = false>
__device__ void d_flash_decode(float* __restrict__ Opart, float* __restrict__ mlpart,
                               const bf16* __restrict__ Q, const bf16* __restrict__ K,
                               const bf16* __restrict__ V, const int* __restrict__ kv_len,
                               unsigned n_batch, unsigned n_head, unsigned n_kv_head,
                               unsigned kv_stride, unsigned window, float scale, unsigned nsplit,
                               unsigned kv_mask, unsigned slice, unsigned nblk, float* lds,
                               const float* __restrict__ k_scale = nullptr,
                               const float* __restrict__ v_scale = nullptr,
                               const bf16* __restrict__ nrf_kg = nullptr,
                               const bf16* __restrict__ nrf_vg = nullptr,
                               const bf16* __restrict__ nrf_gq = nullptr,
                               const bf16* __restrict__ nrf_gk = nullptr,
                               const float* __restrict__ nrf_cos = nullptr,
                               const float* __restrict__ nrf_sin = nullptr, float nrf_eps = 0.0f,
                               unsigned nrf_skip = 0, unsigned* mrg_ctr = nullptr,
                               bf16* o_final = nullptr) {
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

        /* All GF query rows into LDS, once.
         *
         * TRIED AND REVERTED: `as_glob(Q)` + a 16-byte `ld_glob8` here, on the theory that a bare
         * `const bf16*` is a generic align-2 pointer and this was issuing `flat_load_ushort` (the
         * trap amd_common.h's RN_VEC note describes, and K/V two lines above DO use as_glob).
         * Both halves turned out not to matter: the object's flat_load count is IDENTICAL either
         * way, so LLVM's address-space inference had already resolved Q to global; and the
         * vectorisation moved the traced FlashDecode span only 32.56 -> 32.06 us, which is
         * 0.2% of the token and did not survive at the token level (12.019 -> 12.046, i.e. inside
         * this box's drift). GF*D is only 512 halves -- 1 KB per workgroup -- so there was never
         * much here to win. Left as the simple scalar loop.
         *
         * THE REVERT ITSELF DELETED THIS LOOP and left only the comment (6748e5b): qsm was never
         * filled, flash decode dotted stale LDS, and every object rebuilt from that commit decoded
         * fluent garbage. It shipped because the commit landed AFTER the session's last serve
         * validation and the perf A/B (timing-only) cannot see it. Caught by the serve token gate
         * on the next rebuild. If you touch this staging again, re-run the serve gate afterwards. */
        if constexpr (!NRF) {
            for (unsigned i = tid; i < GF * D; i += PLOW_THREADS)
                qsm[i] = Q[((size_t)b * n_head + h0 + i / D) * D + i % D];
            __syncthreads();
        } else {
            /* NRF: d_headnorm_rope replicated per head, one WAVE per head (the layout that keeps
             * each RoPE pair inside one lane). See the header note for why this is bit-exact. */
            constexpr unsigned E = D / 64;   /* elements per lane */
            constexpr unsigned H2 = D / 2;
            constexpr unsigned EH = H2 / 64; /* lane-local stride to the half-split partner */
            const size_t p = (size_t)qpos * H2;
            const auto* cg = as_glob(nrf_cos);
            const auto* sg = as_glob(nrf_sin);
            /* Phase 1: the GF query heads -> qsm. */
            if (wave < GF) {
                const auto* xg = as_glob(Q) + ((size_t)b * n_head + h0 + wave) * D;
                const auto* gg = as_glob(nrf_gq);
                float v[E], gv[E];
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    v[e] = bf2f(xg[lane + e * 64]);
                    gv[e] = gg ? bf2f(gg[lane + e * 64]) : 1.0f;
                }
                float inv = 1.0f;
                if (!nrf_skip) {
                    float ss = 0.0f;
#pragma unroll
                    for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
                    inv = rsqrtf(wave_sum(ss) / (float)D + nrf_eps);
                }
#pragma unroll
                for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * gv[e];
                float r[E];
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 64;
                    const unsigned j = (i < H2) ? i : (i - H2);
                    const float c = cg[p + j], s = sg[p + j];
                    r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s) : (v[e] * c + v[e - EH] * s);
                }
#pragma unroll
                for (unsigned e = 0; e < E; e++) qsm[(size_t)wave * D + lane + e * 64] = f2bf(r[e]);
            }
            /* Phase 2 (the ONE owning item): current-token K and V rows -> the cache. */
            if (qpos >= lo && qpos < hi) {
                const size_t slot =
                    (((size_t)b * n_kv_head + hkv) * kv_stride + (qpos & kv_mask)) * D;
                if (wave == 0) { /* K: norm(gamma_k) + RoPE, exactly hnr's k arm */
                    const auto* xg = as_glob(nrf_kg) + ((size_t)b * n_kv_head + hkv) * D;
                    const auto* gg = as_glob(nrf_gk);
                    auto* og = as_glob((bf16*)K);
                    float v[E], gv[E];
#pragma unroll
                    for (unsigned e = 0; e < E; e++) {
                        v[e] = bf2f(xg[lane + e * 64]);
                        gv[e] = gg ? bf2f(gg[lane + e * 64]) : 1.0f;
                    }
                    float inv = 1.0f;
                    if (!nrf_skip) {
                        float ss = 0.0f;
#pragma unroll
                        for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
                        inv = rsqrtf(wave_sum(ss) / (float)D + nrf_eps);
                    }
#pragma unroll
                    for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * gv[e];
                    float r[E];
#pragma unroll
                    for (unsigned e = 0; e < E; e++) {
                        const unsigned i = lane + e * 64;
                        const unsigned j = (i < H2) ? i : (i - H2);
                        const float c = cg[p + j], s = sg[p + j];
                        r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s) : (v[e] * c + v[e - EH] * s);
                    }
#pragma unroll
                    for (unsigned e = 0; e < E; e++) st_act1(&og[slot + lane + e * 64], f2bf(r[e]));
                } else if (wave == 1) { /* V: weightless RMS (Gemma norms V), no RoPE, no gamma */
                    const auto* xg = as_glob(nrf_vg) + ((size_t)b * n_kv_head + hkv) * D;
                    auto* og = as_glob((bf16*)V);
                    float v[E];
#pragma unroll
                    for (unsigned e = 0; e < E; e++) v[e] = bf2f(xg[lane + e * 64]);
                    float inv = 1.0f;
                    if (!nrf_skip) {
                        float ss = 0.0f;
#pragma unroll
                        for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
                        inv = rsqrtf(wave_sum(ss) / (float)D + nrf_eps);
                    }
#pragma unroll
                    for (unsigned e = 0; e < E; e++) st_act1(&og[slot + lane + e * 64], f2bf(v[e] * inv));
                }
                /* Drain the stores (s_waitcnt, NO cache ops) before the barrier. An agent-scope
                 * fence here was the first attempt and it cost +0.6 ms/token: its buffer_inv is
                 * a FULL L1+L2 invalidate, issued by every owning item, destroying the KV lines
                 * every other workgroup was streaming. It is also unnecessary: the packet's gate
                 * acquire already invalidated this CU's L1, and nothing re-reads the written row
                 * between that and this store — so the tile loop's loads MISS L1 and hit the
                 * store in L2. Workgroup release + barrier is the whole requirement. */
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
            }
            __syncthreads();
        }

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

        /* ABLATION (FA_ABL=2): retire the WHOLE kv loop -- K phase, softmax and V. What is left
         * is the packet's own cost (gate, dispatch, Q staging, epilogue), which separates "this
         * kernel is slow" from "this packet is expensive". Probe only. */
#if FA_ABL >= 2
        for (unsigned kv0 = lo; kv0 < lo; kv0 += FA_DEC_TILE) { /* ablation: loop never runs */
#else
        for (unsigned kv0 = lo; kv0 < hi; kv0 += FA_DEC_TILE) {
#endif
            /* SCORES.  EIGHT LANES PER K ROW.                            [K-PHASE-KL8]
             *
             * The row-per-lane map this replaced was measured, on this exact kernel, to cost 20%
             * of the machine's read bandwidth. A wave issuing one 16-byte load per lane across 64
             * DIFFERENT rows makes 64 distinct cache-line requests from one instruction; the line
             * is then re-requested by that lane's next seven loads, and each re-request costs an
             * address-processing slot even though it hits. runtime/tests/decode_bw_probe.hip
             * strips everything else away and reads a 16 GB buffer in each shape (gfx942, 304 CU):
             *
             *   contiguous stream                     4119 GB/s
             *   row-per-lane, D=512  (the old map)    3348 GB/s   <- 19% below stream
             *   8 lanes/row, D=512   (this map)       4030 GB/s
             *   64 lanes/row, D=512  (the V phase)    4050 GB/s
             *
             * EIGHT is the number, not 64. A lane loads 16 bytes, so 8 lanes span exactly one
             * 128-byte line: one instruction is 8 rows x 1 line, the SAME line count per
             * instruction as the 64-lane form, and it reaches the same bandwidth. But the dot it
             * implies is reduced over 8 lanes, not 64 -- 3 lane-exchanges per row-group instead of
             * 6, and a wave-pass covers 8 rows instead of 1, so the exchanges per ROW fall by 16x
             * (0.75 vs 12 at GF=2). That is what kills the objection the old comment recorded:
             * the earlier cooperative attempt used NDT = D/8 lanes per row and drowned in shuffles.
             *
             * The row a lane contributes to is (wave*ROWS_PER_WAVE + pass*KR + lane/KL), so the
             * tile is still covered exactly once and every Ssm slot is still written -- including
             * the out-of-range ones, which must be -inf for the block-wide max.
             *
             * THE FUSION is unchanged: the row's 16-byte chunk is loaded ONCE and dotted against
             * all GF query rows out of LDS, so the row still crosses HBM once, not GQA times. */
            constexpr unsigned KL = FA_DEC_KL;              /* lanes per K row              */
            constexpr unsigned KR = 64 / KL;                /* K rows per wave-pass         */
            constexpr unsigned KRW = FA_DEC_TILE / PLOW_WAVES; /* rows a wave owns per tile */
            constexpr unsigned KPASS = KRW / KR;            /* passes to cover them         */
            constexpr unsigned KSTEP = KL * 8;              /* elems a pass advances        */
            const unsigned ksub = lane % KL, krl = lane / KL;
#pragma unroll FA_DEC_KUNROLL
            for (unsigned p = 0; p < KPASS; p++) {
                /* ROW->WAVE MAP. The default BLOCKS the tile: wave w owns rows [w*KRW,(w+1)*KRW),
                 * which is exactly right when the tile is full. It is pathological when it is not.
                 * A split covers `per = ceil(span/nsplit)` rows of a FA_DEC_TILE(=512)-row tile,
                 * and at Gemma-4's 1024-token sliding window with nsplit=38 that is 27 -- so every
                 * live row lands in wave 0's block and SEVEN OF EIGHT WAVES DO NOTHING. Measured
                 * 233 GB/s against the 4030 GB/s this same load map reaches on a full tile.
                 *
                 * FA_DEC_ILV INTERLEAVES instead: pass p covers rows [p*W*KR,(p+1)*W*KR) spread
                 * across all waves, so the first `per` rows occupy ceil(per/KR) waves rather than
                 * one. Same bijection onto [0,FA_DEC_TILE) -- every Ssm slot is still written
                 * exactly once, which is what the block-wide max and the `Ssm[g*TILE+rl]` store
                 * below require -- and a pass is now CONTIGUOUS in kv (rows p*64..p*64+63) where
                 * the blocked map strided it by KRW across waves. */
                /* NO DEAD-PASS EARLY EXIT HERE, and the reason is worth recording because the
                 * optimisation looks free. Under FA_DEC_ILV pass p covers [kv0+p*W*KR, ...) for the
                 * whole workgroup, so `if (kv0 + p*W*KR >= hi) break` would retire 7 of 8 passes at
                 * per=27 -- but a retired pass never writes its Ssm slots, and the softmax max
                 * below is `for (i = lane; i < FA_DEC_TILE; i += 64) fmaxf(mx, Ssm[...])`, i.e.
                 * UNBOUNDED over the tile. The skipped slots would hold the previous tile's scores
                 * and poison the block-wide max. Retiring them needs the -inf fill hoisted out of
                 * the pass loop first; until then the predicate below is what keeps them correct. */
                const unsigned rl = FA_DEC_ILV ? (p * (PLOW_WAVES * KR) + wave * KR + krl)
                                               : (wave * KRW + p * KR + krl);
                const unsigned kv = kv0 + rl;
                float s0[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) s0[g] = FA_NEG_INF;
                if (kv < hi && kv <= qpos && (!window || (qpos - kv) < window)) {
                    float dot[GF];
#pragma unroll
                    for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                    float ks = 1.0f; /* fp8-KV per-row dequant scale; 1.0 (dead) for bf16 KV */
                    if constexpr (FP8KV) {
                        /* fp8: the K row is e4m3 (HALF the bytes of bf16), so a lane's 8-element
                         * chunk is a b64 load and KL lanes span 64 bytes rather than a full line.
                         * Still 8 lines per instruction instead of 64, and the b64 width is what
                         * keeps the O accumulators out of the AGPRs (a b128 fp8 load holds
                         * lo+hi+the fp8v16 across the dot; measured 130 vs 229 VGPR). */
                        const unsigned char* krow = kb8 + (size_t)(kv & kv_mask) * D;
#pragma unroll
                        for (unsigned d = ksub * 8; d < D; d += KSTEP) {
                            const bf16v8 kv8 = fp8v8_to_bf16v8(ld_glob_fp8v8(krow + d));
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dot[g] = dot8(kv8, ld_lds8(qsm + g * D + d), dot[g]);
                        }
                        ks = ksc[kv & kv_mask];
                    } else {
                        const auto* krow = kbase + (size_t)(kv & kv_mask) * D; /* sliding RING */
#pragma unroll
                        for (unsigned d = ksub * 8; d < D; d += KSTEP) {
                            const bf16v8 kv8 = ld_glob8(krow + d); /* <-- read ONCE */
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dot[g] = dot8(kv8, ld_lds8(qsm + g * D + d), dot[g]);
                        }
                    }
                    /* Close the row: sum the KL lane partials. XOR butterfly over the low
                     * log2(KL) lanes only, so the KR row-groups of the wave reduce concurrently
                     * and independently -- one exchange instruction serves all of them. */
#pragma unroll
                    for (int g = 0; g < GF; g++) {
#pragma unroll
                        for (unsigned off = 1; off < KL; off <<= 1)
                            dot[g] += __shfl_xor(dot[g], (int)off, 64);
                        s0[g] = dot[g] * FA_SCALE(scale);
                        /* AFTER the reduction and AFTER the scale — (dot*SCALE)*ks is the
                         * association the pre-restructure code used; keep fp8-KV bit-identical. */
                        if constexpr (FP8KV) s0[g] *= ks;
                    }
                }
                if (ksub == 0) {
#pragma unroll
                    for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + rl] = s0[g];
                }
            }
            /* This thread's OWN row's scores, for the pe below: it computed partials for KPASS
             * other rows, not for row `tid`. One LDS read per head after the barrier, which the
             * softmax needs anyway. */
            float s[GF];

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
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];

            /* BOUND THE SOFTMAX REDUCTIONS BY THE LIVE ROW COUNT (FA_DEC_LIVE).
             *
             * Both reductions below sweep the whole 512-row tile, but a split only covers
             * `per = ceil(span/nsplit)` rows -- 64 at the measured-best nsplit=16 with win=1024, so
             * SEVEN EIGHTHS of each sweep reads slots that cannot contribute. This is EXACTLY
             * equivalent, not an approximation: the K phase fills dead slots with -inf, so they add
             * nothing to the max, and their `pe` is consequently 0, adding nothing to the sum. The
             * -inf fill itself must STAY -- the `s[g]` read below and the V phase rely on it -- so
             * this bounds the READS only. Same quantity `rmax_pf` already computes for VPIPE. */
            const unsigned live_ =
                ((hi - kv0) < (unsigned)FA_DEC_TILE) ? (hi - kv0) : (unsigned)FA_DEC_TILE;
            const unsigned red_n = FA_DEC_LIVE ? live_ : (unsigned)FA_DEC_TILE;
            /* GF softmax reductions, ONE PER WAVE, so they all run concurrently: the tile still
             * costs 3 barriers, not 3*GF. (There are PLOW_WAVES=8 waves and GF <= 8.) */
            if (wave < GF) {
                float mx = FA_NEG_INF;
                for (unsigned i = lane; i < red_n; i += 64)
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
                for (unsigned i = lane; i < red_n; i += 64)   /* see FA_DEC_LIVE above */
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
            /* ABLATION (FA_ABL=1): rmax 0 retires every V loop below, leaving the K phase and
             * the online-softmax reduction. Splits this op's ~32 us into scores+softmax vs the V
             * stream. Wrong results by construction; probe only, default 0. */
#if FA_ABL
            const unsigned rmax = 0u; /* ablation: retires every V loop below */
#else
            const unsigned rmax = (hi - kv0 < FA_DEC_TILE) ? (hi - kv0) : FA_DEC_TILE;
#endif
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

#if FA_ABL >= 3
        /* ABLATION 3: skip the row-group fold and the Opart/mlpart writes entirely. With the KV
         * loop already retired at level 2, what remains is the packet itself plus Q staging --
         * so (level2 - level3) prices the fold + partial writes. Probe only. */
        if (false)
#endif
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
                st_act<float>(&op[d], acc);
            }
            if (tid == 0) {
                float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
                st_act<float>(&ml[0], m_st[g]);
                st_act<float>(&ml[1], l_st[g]);
            }
        }

        /* [MERGE-FOLD] The LAST-arriving split of this (b, head-group) merges the partials
         * itself and the FlashMerge packet is never emitted — one fewer claim+gate+dispatch on
         * the serial chain per layer (the NRN-fold precedent priced a deleted serial packet at
         * ~4 µs; 48 layers ≈ −0.19 ms/token). NOT the widening §above kills: the consumer's
         * gate does not widen — o_proj waits on THIS packet's own coarse counter, and the
         * merger IS one of its workgroups, so packet completion still means "merged".
         *
         * Rendezvous: every item, after its Opart/ml stores, bumps its group's counter with an
         * agent-scope RELEASE (publishing the stores); the item that sees nsplit-1 is last,
         * takes ONE agent acquire (the full-L1-invalidate lesson: once per merge, never per
         * item), merges its GF heads, and resets the counter to 0 — self-cleaning, so the
         * compiler-owned counter tensor is zeroed exactly once at load. The reset is ordered
         * before this packet's own release-signal, and the NEXT token's flash for this layer
         * sits transitively behind this packet's consumers, so the reset cannot race it. */
        if (NRF && mrg_ctr != nullptr) {
            __syncthreads(); /* every lane's op[]/ml[] stores are issued */
            if (tid == 0)
                hmax[0] = (float)__hip_atomic_fetch_add(&mrg_ctr[b * n_grp + hg], 1u,
                                                        __ATOMIC_RELEASE,
                                                        __HIP_MEMORY_SCOPE_AGENT);
            __syncthreads();
            if ((unsigned)hmax[0] == nsplit - 1u) {
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    const unsigned h = h0 + (unsigned)g;
                    const float* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;
                    float gm;
                    const float gl = fa_merge_ml(ml, nsplit, gm);
                    const float inv = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;
                    const float* obase = Opart + (size_t)(b * n_head + h) * nsplit * (size_t)D;
                    auto* const orow = as_glob(o_final) + (size_t)(b * n_head + h) * D;
                    for (unsigned d = tid; d < (unsigned)D; d += PLOW_THREADS) {
#if !FA_MERGE_UNROLL4
                        float acc = 0.0f;
                        for (unsigned s = 0; s < nsplit; s++) {
                            if (ml[s * 2] == FA_NEG_INF) continue;
                            acc += obase[(size_t)s * D + d] * FA_EXP(ml[s * 2] - gm);
                        }
#else
                        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                        unsigned s = 0;
                        for (; s + 4 <= nsplit; s += 4) {
                            const float m0 = ml[(s + 0) * 2], m1 = ml[(s + 1) * 2];
                            const float m2 = ml[(s + 2) * 2], m3 = ml[(s + 3) * 2];
                            const float v0 = obase[((size_t)s + 0) * D + d];
                            const float v1 = obase[((size_t)s + 1) * D + d];
                            const float v2 = obase[((size_t)s + 2) * D + d];
                            const float v3 = obase[((size_t)s + 3) * D + d];
                            a0 += v0 * ((m0 == FA_NEG_INF) ? 0.0f : FA_EXP(m0 - gm));
                            a1 += v1 * ((m1 == FA_NEG_INF) ? 0.0f : FA_EXP(m1 - gm));
                            a2 += v2 * ((m2 == FA_NEG_INF) ? 0.0f : FA_EXP(m2 - gm));
                            a3 += v3 * ((m3 == FA_NEG_INF) ? 0.0f : FA_EXP(m3 - gm));
                        }
                        for (; s < nsplit; s++) {
                            const float m0 = ml[s * 2];
                            a0 += obase[(size_t)s * D + d] *
                                  ((m0 == FA_NEG_INF) ? 0.0f : FA_EXP(m0 - gm));
                        }
                        const float acc = (a0 + a1) + (a2 + a3);
#endif
                        st_act1(&orow[d], f2bf(acc * inv));
                    }
                }
                __syncthreads(); /* merge stores issued before the reset can publish */
                if (tid == 0)
                    __hip_atomic_store(&mrg_ctr[b * n_grp + hg], 0u, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
}

/* Combine the split partials: standard online-softmax merge.
 *
 * WORK DECOMPOSITION IS (batch, head, d-chunk), and the d-chunk axis is DERIVED FROM `nblk`, not
 * passed in. `dsplit = ceil(nblk / (n_batch*n_head))`, so when the emitter hands this op exactly
 * `n_batch*n_head` workgroups (or fewer) `dsplit == 1`, `dchunk == D`, and every index below
 * collapses to the original `w = b*n_head + h` expression -- the same work, in the same order,
 * to the same addresses. The axis only appears when the emitter asks for it by widening the CU
 * list, which is what makes the widening a *compiler* knob (`PLOW_FLASH_MERGE_DSPLIT`) served by
 * ONE kernel: there is no second copy of this loop to drift, and no arm to leave unrouted.
 *
 * Splitting D needs no new reduction and no new gate: a merge item folds the `nsplit` partials of
 * its own (row, head) over its own D-chunk and touches nothing else. The (m,l) reduction is
 * 2*nsplit floats and stays REPLICATED across the dsplit workgroups of a (row, head) -- it is
 * already redundant across all 512 threads of one workgroup, so replicating it costs uniform,
 * L2-resident loads and buys the absence of a cross-workgroup reduction.
 *
 * WIDENING IS MEASURED DEAD. TWICE, BY TWO AGENTS, ON DIFFERENT IMPLEMENTATIONS. Do not
 * re-propose it; re-read this instead.
 *
 * First kill: the same decomposition with the split folded across WAVES. The merge itself got
 * faster (0.56 -> 0.46 ms) and the TOKEN got slower (16.7 -> 16.9), at every width swept.
 *
 * Second kill (2026-07-27, this code): folded across WORKGROUPS, which is the shape an ideal-
 * schedule simulation on the real DAG predicted would be worth -0.805 ms/token -- the largest
 * single decode lever on the board. Gemma-4-31B bf16, ctx 1024, MI355X, `plowrt amd-bench`,
 * interleaved arms, SAME code object in every arm (only the blob's merge CU count differs):
 *
 *   dsplit  merge wgs   wg-packets/token   median ms/token
 *      1        32          79,947            17.517   (n=28)
 *      2        64          81,867            +0.243
 *      4       128          85,707            +0.348
 *      8       256          93,387            18.072   (n=28, +0.555)
 *
 * Monotone in WIDTH, three independent interleaved sets, and every arm token-identical. The
 * predicted -0.805 came back as +0.555, so the simulation is wrong about this class of lever,
 * not the kernel.
 *
 * Mechanism (and it is the first kill's, re-confirmed): widening an op WIDENS ITS CONSUMER'S
 * GATE. o_proj depends on the merge COARSELY -- a GEMV workgroup reads all of `n.at`, so that
 * edge is genuinely dense and cannot be made fine -- so it waits on a max over 256 stragglers
 * instead of a max over 32, and a max over more samples opens later. Gates themselves are not
 * the cost: +100 gates/token measured FASTER elsewhere, <=0.64 us/gate. Here the loss is
 * ~9 us on each of 60 packets, an order of magnitude too big to be gate protocol. It is the
 * straggler tail (+-19% on identical work, random per-(CU,packet)) sampled 8x more often.
 *
 * The corollary generalises: "this op is narrow, widen it" is NOT sound on this machine
 * whenever the consumer's edge is dense. Price the consumer's gate, not the op.
 *
 * Also note raising `nsplit` to fill the machine with flash work is self-defeating regardless of
 * the merge: Q staging and Opart traffic BOTH scale with nsplit (n_head * nsplit * D), so
 * flash_decode gets slower as it is split more finely (1.83 -> 3.08 ms at nsplit 16 -> 64).
 * Split-KV's overhead is the ceiling, not the merge.
 *
 * The axis is kept, defaulted to 1, purely as the reproduction vehicle, and it is free at that
 * default: the emitted blob is BYTE-IDENTICAL to the pre-change emitter's, the objects' registers
 * are unchanged (decode 248 VGPR / occ 2 / spill 0, flash 256+256 / occ 1 / spill 228 -- the
 * intentional Q-hoist), and the token is unchanged within noise (pre-change 17.472 vs dsplit=1
 * 17.275 median, n=9 interleaved, sd 0.35 / 0.62). Do not change the default. */

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
    const unsigned n_bh = n_batch * n_head;
    /* MUST match flash_merge_map() in crates/devgen/src/lib.rs. A mismatch is a silent wrong
     * token, not an error: the fine dep would gate a workgroup on the wrong flash slices. */
    const unsigned dsplit = (nblk + n_bh - 1) / n_bh;
    const unsigned dchunk = ((unsigned)D + dsplit - 1) / dsplit;
    const unsigned n_work = n_bh * dsplit;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned dp = w % dsplit, hb = w / dsplit;
        const unsigned h = hb % n_head, b = hb / n_head;
        const unsigned d0 = dp * dchunk;
        const unsigned d1 = (d0 + dchunk < (unsigned)D) ? (d0 + dchunk) : (unsigned)D;
        const auto* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;

        float gm;
        const float gl = fa_merge_ml(ml, nsplit, gm);
        /* v_rcp_f32 -- see the note at d_flash_prefill's epilogue. */
        const float inv = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;

        /* FOUR SPLITS IN FLIGHT, and no `continue`.                   [MERGE-UNROLL4]
         *
         * `nsplit` is a runtime value, so the split loop could not be unrolled, and its body was
         * load -> wait -> accumulate: ONE 4-byte load outstanding per thread at a time, each from
         * a different D*4-byte-strided line. At batch 1 this op has only n_batch*n_head work items
         * (16 for the 12B), so it runs on 16 of 304 CUs with no co-resident wave to hide any of
         * that latency, and it cost as much as the whole decode it was folding (17.8 us against
         * 76.9 at ctx=32k, nsplit=32 -- the merge was 19% of the token's attention).
         *
         * Four independent accumulators let four loads fly at once. The `continue` had to go for
         * the unroll to be branch-free, and it can: an EMPTY split still runs d_flash_decode's
         * final row-group fold, so its Opart slice is written and is exactly 0.0f, and its scale
         * here is exactly 0.0f -- the product is 0, not NaN. (That is why the guard is on
         * `FA_NEG_INF` and not on a garbage-valued slice.)  The gl reduction above keeps its
         * guard: it reads ml, not Opart, and runs once per work item.  */
        const auto* obase = Opart + (size_t)(b * n_head + h) * nsplit * (size_t)D;
        for (unsigned d = d0 + threadIdx.x; d < d1; d += PLOW_THREADS) {
#if !FA_MERGE_UNROLL4
            /* The pre-UNROLL4 serial merge, kept bit-identical for CDNA4 (see FA_MERGE_UNROLL4). */
            float acc = 0.0f;
            for (unsigned s = 0; s < nsplit; s++) {
                if (ml[s * 2] == FA_NEG_INF) continue;
                const float sc = FA_EXP(ml[s * 2] - gm);
                acc += obase[(size_t)s * D + d] * sc;
            }
#else
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            unsigned s = 0;
            for (; s + 4 <= nsplit; s += 4) {
                const float m0 = ml[(s + 0) * 2], m1 = ml[(s + 1) * 2];
                const float m2 = ml[(s + 2) * 2], m3 = ml[(s + 3) * 2];
                const float v0 = obase[((size_t)s + 0) * D + d], v1 = obase[((size_t)s + 1) * D + d];
                const float v2 = obase[((size_t)s + 2) * D + d], v3 = obase[((size_t)s + 3) * D + d];
                a0 += v0 * ((m0 == FA_NEG_INF) ? 0.0f : FA_EXP(m0 - gm));
                a1 += v1 * ((m1 == FA_NEG_INF) ? 0.0f : FA_EXP(m1 - gm));
                a2 += v2 * ((m2 == FA_NEG_INF) ? 0.0f : FA_EXP(m2 - gm));
                a3 += v3 * ((m3 == FA_NEG_INF) ? 0.0f : FA_EXP(m3 - gm));
            }
            for (; s < nsplit; s++) {
                const float m0 = ml[s * 2];
                a0 += obase[(size_t)s * D + d] * ((m0 == FA_NEG_INF) ? 0.0f : FA_EXP(m0 - gm));
            }
            const float acc = (a0 + a1) + (a2 + a3);
#endif /* FA_MERGE_UNROLL4 */
            st_act1(&O[((size_t)b * n_head + h) * D + d], f2bf(acc * inv));
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

/* The LARGEST GF `exec_flash_mla_decode` instantiates. GLM_MLA_GF above is only the DEFAULT for a
 * packet whose i[7] is 0; the per-packet i[7] selects 2, 4 or 8. The interpreter's LDS union must
 * be sized for the MAX, not the default, because the arena reaches the kernel as a bare `float*`
 * and a GF=8 body lays out GF*(FA_DEC_TILE + DK/2 + DR/2) floats into it — sizing that member at
 * GF=4 would let the GF=8 arm write 12.8 KB past its own member. It does NOT change the object's
 * LDS: the union is dominated by the GEMM arena (147456 B) and this is 42048 B at GF=8. */
#define GLM_MLA_GF_MAX 8

/* PLOW_GLM_GF8_ARM — compile the GF=8 flash-decode instantiation. DEFAULT 0, and that default
 * is a MEASURED REGRESSION FIX, not caution.
 *
 * The arm landed in 9dc27bb validated on REGISTERS ALONE (+6 VGPR, 248 -> 254, occupancy 2 held,
 * zero spill, LDS unmoved) and that pass concluded "GF=8 fits". It does fit. It is also a +32%
 * DECODE REGRESSION, bisected on a BYTE-IDENTICAL model.pkt with only the device object varying
 * (TP8, ctx 32768, median ITL):
 *     pre-6153189   248 VGPR   i_decode.co 313,024   27.98 ms
 *     c9b6bae       248 VGPR              313,024   27.98 ms
 *     9dc27bb       254 VGPR              361,896   37.01 ms   <-- the arm
 *     HEAD          254 VGPR              362,184   36.87 ms
 *
 * AND IT IS NOT THE ARM RUNNING. A HEAD packet with PLOW_GLM_GF=4 pinned still measures 36.12.
 * It is the arm being PRESENT: a second `d_flash_mla_decode` instantiation grows the decode
 * object 15.6% inside a PERSISTENT MEGAKERNEL, where every packet body shares one instruction
 * stream. A register-only pass structurally cannot see this — registers are per-wave, object
 * size is per-kernel. Record that as the general lesson: for this interpreter, "the arm fits in
 * the register budget" is necessary and NOT sufficient; the object must be weighed too.
 *
 * The code is kept, not deleted, because GF=8 has still never been measured on its merits
 * (its Phase B A/B never ran). To measure it, build with -DPLOW_GLM_GF8_ARM=1 and compare
 * GF=4 vs GF=8 ON THE SAME OBJECT by varying PLOW_GLM_GF at emit — otherwise the comparison is
 * confounded by exactly the object-size effect above.
 *
 * With the arm compiled out, an emitted GF=8 request falls to the `else` (GF=4), which is the
 * pre-9dc27bb behaviour and the configuration every published number was measured in. */
#ifndef PLOW_GLM_GF8_ARM
#define PLOW_GLM_GF8_ARM 0
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
/* FP8 (e4m3) LATENT KV, the MLA twin of `d_flash_decode`'s FP8KV.       [MLA-FP8-KV]
 *
 * At FP8 the `Ckv` pointer is a `uint8[b][ctx][DK]` e4m3 cache with a PER-ROW f32 dequant scale,
 * and `Krope` is EITHER the untouched bf16 rope cache (`krot_fp8 == 0`, the shipped form) or its
 * own e4m3 cache (`krot_fp8 != 0`). Both scales live in ONE `kv_scale` array because the dense
 * MLA decode has exactly ONE free tensor slot (t7 is the gather `idx`, and every other slot is
 * live):
 *
 *     kv_scale[               b*kv_stride + row ]   ckv  row scale
 *     kv_scale[ n_batch*kv_stride + b*kv_stride + row ]   krot row scale   (krot_fp8 only)
 *
 * PER-ROW, not per-tensor, for the same reason `d_headnorm_rope_fp8` picked it: a KV row is
 * written once at its own step and never revisited, so a row scale costs one f32 per 512 bytes
 * and needs no second pass. Per-tensor would have to be chosen before the context exists.
 *
 * WHY THE TWO DOTS ARE ACCUMULATED SEPARATELY UNDER FP8. The bf16 path sums the ckv dot and the
 * krot dot into ONE accumulator, because both operands are the same dtype and the score is
 * `dot * scale`. Under fp8 the two halves carry DIFFERENT dequant scales, so the score is
 * `(dot_ckv*s_ckv + dot_krot*s_krot) * scale` and one accumulator would be wrong. The bf16
 * expressions below are therefore left character-for-character intact under `if constexpr
 * (!FP8)` rather than refactored into a common shape — same discipline as `FA_DB_FP8` above,
 * and for the same reason: the bf16 MLA decode is a shipped object at 254 of 256 VGPRs.
 *
 * THE ERROR IS SHARED, WHICH DENSE GQA's IS NOT. One latent row is read by EVERY query head of
 * the layer (and, through `q_absorb` / `MlaMergeFold`, through two learned projections), so a
 * quantization error here is common-mode across all `n_head` heads rather than confined to one.
 * That is a reason to MEASURE this family separately from the dense one, not a reason it cannot
 * work; the numbers are in perf-data/mla-fp8-kv.md. */
template <int DK, int DR, int GF, bool GATHER = false, bool FP8 = false>
__device__ void d_flash_mla_decode(float* __restrict__ Opart, float* __restrict__ mlpart,
                                   const bf16* __restrict__ Qabs, const bf16* __restrict__ Qrope,
                                   const bf16* __restrict__ Ckv, const bf16* __restrict__ Krope,
                                   const int* __restrict__ kv_len, unsigned n_batch,
                                   unsigned n_head, unsigned kv_stride, unsigned window,
                                   float scale, unsigned nsplit, unsigned kv_mask, unsigned slice,
                                   unsigned nblk, float* lds, const int* __restrict__ idx = nullptr,
                                   unsigned top_k = 0, unsigned n_tok = 1,
                                   const float* __restrict__ kv_scale = nullptr,
                                   unsigned krot_fp8 = 0,
                                   /* [HNR-FOLD] non-null => `Qrope` is the RAW q_rope projection
                                    * and this kernel applies the interleaved RoPE itself. See the
                                    * staging loop. Dense bf16 arm only. */
                                   const float* __restrict__ qr_cos = nullptr,
                                   const float* __restrict__ qr_sin = nullptr) {
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
        /* GATHER splits over the SELECTED slots; dense splits over the KV window, which for prefill
         * is the per-token window ending at qpos, not at len.
         *
         * `tk_live = min(top_k, len)` and not `top_k`: `i[6]` is the emit-time `index_topk` (2048),
         * but only `min(top_k, kv_len)` slots of `idx[]` were ever produced — `d_index_select_coop`
         * clamps to exactly the same expression off the same `kv_len` operand. Walking the full
         * `top_k` on a short context read STALE index slots and, because GATHER applies no mask
         * (`keep` is just `kv < hi`), gathered latent rows that had never been written. `top_k`
         * itself still stripes `ibase` below: that is the table's ALLOCATION stride and does not
         * shrink with the live length. */
        const unsigned tk_live = GATHER ? (top_k < len ? top_k : len) : 0u;
        const unsigned first = GATHER ? 0u : ((window && cend > window) ? (cend - window) : 0u);
        const unsigned span = GATHER ? tk_live : (cend - first);
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi = GATHER ? (lo + per < tk_live ? lo + per : tk_live)
                                   : (lo + per < cend ? lo + per : cend);

        /* ONE latent "head": the cache base is just this batch's latent block. */
        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const auto* rbase = as_glob(Krope) + (size_t)b * kv_stride * DR;
        /* FP8 twins of the same two bases, plus the two per-row scale strips. `cb8`/`rb8` are the
         * SAME allocations viewed as e4m3 bytes (half the width per element); `rb8`/`rsc` are only
         * dereferenced when `krot_fp8`. */
        const unsigned char* cb8 = (const unsigned char*)Ckv + (size_t)b * kv_stride * DK;
        const unsigned char* rb8 = (const unsigned char*)Krope + (size_t)b * kv_stride * DR;
        const float* csc = FP8 ? kv_scale + (size_t)b * kv_stride : nullptr;
        const float* rsc = FP8 ? kv_scale + (size_t)(n_batch + b) * kv_stride : nullptr;
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
        /* ROPE FOLD ([HNR-FOLD], emit knob PLOW_GLM_FUSE_ROPE): when `qr_cos` is non-null the
         * `Qrope` pointer is the RAW q_rope projection and the interleaved RoPE that a separate
         * `HEADNORM_ROPE` packet used to apply is done HERE, in the staging that already reads
         * every one of those GF*DR elements.
         *
         * Why this is BIT-IDENTICAL and not merely equivalent: the q-side rope packet runs
         * `d_headnorm_rope<64, INTERLEAVE=true>` with `gamma == nullptr` and `skip_norm == 1`, so
         * its `v[e]` is exactly `bf2f(x)` — no norm, no gamma — and its store is
         * `f2bf(v*c -/+ partner*s)`. The three lines below are that expression, character for
         * character, with the partner READ FROM MEMORY instead of `__shfl_xor(.,1)`. The shuffle
         * is not reusable here: it assumes lane == element index, which holds in the rope kernel
         * (one wave per head, `lane` IS the element) and does NOT hold in this staging loop
         * (thread `tid` walks a flat GF*DR range with stride PLOW_THREADS). A second global load
         * of a bf16 that is already in L1 is the cheap way to be exactly right.
         *
         * `qpos` is the position operand: the rope packet reads `pos[0]`, and every decode entry
         * in the tree (amd.rs `decode_step`, `decode_step_batched`, serve/engine.rs) calls with
         * `kvlen == pos + 1`, so `qpos = kv_len[b] - 1 == pos[b]`. That identity is what lets the
         * fold cost NO operand slot for the position — `kv_len` is already t6.
         *
         * Dense bf16 arm only. GATHER puts its `idx` table in t7 and FP8 puts its scale strip
         * there, which is where `cos` has to live; the emitter refuses both combinations.
         *
         * The fold is a RUNTIME branch, not a template parameter, on purpose: this kernel is at
         * 254 of 256 VGPRs and the GF=8 post-mortem above is the record of what a second
         * INSTANTIATION costs a persistent megakernel (+15.6% object, +32% decode) even when the
         * registers fit. A branch outside the KV loop costs the default path a scalar test. */
        if (qr_cos) {
            constexpr unsigned H2 = DR >> 1;
            const float* cg = as_glob(qr_cos);
            const float* sg = as_glob(qr_sin);
            const size_t pbase = (size_t)qpos * H2;
            for (unsigned i = tid; i < GF * DR; i += PLOW_THREADS) {
                const unsigned d = i % DR;
                const size_t rb = (qrow + h0 + i / DR) * DR;
                const float v = bf2f(Qrope[rb + d]);
                const float partner = bf2f(Qrope[rb + (d ^ 1u)]);
                const float c = cg[pbase + (d >> 1)], s = sg[pbase + (d >> 1)];
                qrsm[i] = f2bf((d & 1u) == 0u ? (v * c - partner * s) : (v * c + partner * s));
            }
        } else {
            for (unsigned i = tid; i < GF * DR; i += PLOW_THREADS)
                qrsm[i] = Qrope[(qrow + h0 + i / DR) * DR + i % DR];
        }
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
            /* SCORES.  EIGHT LANES PER LATENT ROW, exactly d_flash_decode's [K-PHASE-KL8].
             *
             * The row-per-lane map made one instruction touch 64 distinct cache lines (measured
             * 3348 vs 4030 GB/s in decode_bw_probe.hip), and here it did worse than that: at
             * GF=4, DK=512 the fully unrolled 64-chunk latent dot with four live accumulators
             * blew the register file -- 256 VGPR and 1836 BYTES PER LANE OF SCRATCH SPILL in the
             * standalone object -- so every latent chunk paid a scratch round trip. That is why
             * the bf16 MLA decode measured 130 GB/s of latent while the dense flash decode on the
             * same silicon measured 3500.
             *
             * The KL=8 map fixes both at once: a pass is DK/(KL*8) = 8 chunks instead of 64, so
             * the loop's live footprint is an eighth, and the 8 lanes covering a row span exactly
             * one 128-byte line. The rope row (DR=64) is ONE chunk per lane -- the whole 128-byte
             * rope row in a single instruction, where the old map read it 2 bytes-per-useful-lane
             * at a time across 64 strided rows. */
            constexpr unsigned KL = FA_DEC_KL;
            constexpr unsigned KR = 64 / KL;
            constexpr unsigned KRW = FA_DEC_TILE / PLOW_WAVES;
            constexpr unsigned KPASS = KRW / KR;
            constexpr unsigned KSTEP = KL * 8;
            const unsigned ksub = lane % KL, krl = lane / KL;
#pragma unroll FA_MLA_KUNROLL
            for (unsigned p = 0; p < KPASS; p++) {
                const unsigned rl = wave * KRW + p * KR + krl;
                const unsigned kv = kv0 + rl;
                /* dense: cache row (kv & kv_mask); gather: the selected index ibase[kv]. */
                const unsigned row = GATHER ? (kv < hi ? (unsigned)ibase[kv] : 0u) : (kv & kv_mask);
                const bool keep =
                    GATHER ? (kv < hi)
                           : (kv < hi && kv <= qpos && (!window || (qpos - kv) < window));
                float s0[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) s0[g] = FA_NEG_INF;
                if (keep) {
                    if constexpr (FP8) {
                        /* e4m3 latent: HALF the HBM bytes. 8-wide (b64) loads decoded to one
                         * bf16v8 per step, mirroring the bf16 loop's live-register footprint
                         * exactly — the same reason `d_flash_decode`'s FP8KV arm does not use
                         * b128. The per-row dequant multiplies the DOT once, never an element. */
                        float dotc[GF], dotr[GF];
#pragma unroll
                        for (int g = 0; g < GF; g++) { dotc[g] = 0.0f; dotr[g] = 0.0f; }
                        const unsigned char* crow = cb8 + (size_t)row * DK;
#pragma unroll FA_MLA_KDU
                        for (unsigned d = ksub * 8; d < DK; d += KSTEP) {
                            const bf16v8 c8 = fp8v8_to_bf16v8(ld_glob_fp8v8(crow + d));
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dotc[g] = dot8(c8, ld_lds8(qsm + g * DK + d), dotc[g]);
                        }
                        const float cs = csc[row];
                        if (krot_fp8) {
                            const unsigned char* rrow = rb8 + (size_t)row * DR;
#pragma unroll FA_MLA_KDU
                            for (unsigned d = ksub * 8; d < DR; d += KSTEP) {
                                const bf16v8 r8 = fp8v8_to_bf16v8(ld_glob_fp8v8(rrow + d));
#pragma unroll
                                for (int g = 0; g < GF; g++)
                                    dotr[g] = dot8(r8, ld_lds8(qrsm + g * DR + d), dotr[g]);
                            }
                            const float rs = rsc[row];
#pragma unroll
                            for (int g = 0; g < GF; g++) s0[g] = dotc[g] * cs + dotr[g] * rs;
                        } else {
                            const auto* rrow = rbase + (size_t)row * DR;
#pragma unroll FA_MLA_KDU
                            for (unsigned d = ksub * 8; d < DR; d += KSTEP) {
                                const bf16v8 r8 = ld_glob8(rrow + d);
#pragma unroll
                                for (int g = 0; g < GF; g++)
                                    dotr[g] = dot8(r8, ld_lds8(qrsm + g * DR + d), dotr[g]);
                            }
#pragma unroll
                            for (int g = 0; g < GF; g++) s0[g] = dotc[g] * cs + dotr[g];
                        }
                    } else {
                        float dot[GF];
#pragma unroll
                        for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                        const auto* crow = cbase + (size_t)row * DK;
#pragma unroll FA_MLA_KDU
                        for (unsigned d = ksub * 8; d < DK; d += KSTEP) {
                            const bf16v8 c8 = ld_glob8(crow + d);
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dot[g] = dot8(c8, ld_lds8(qsm + g * DK + d), dot[g]);
                        }
                        const auto* rrow = rbase + (size_t)row * DR;
#pragma unroll FA_MLA_KDU
                        for (unsigned d = ksub * 8; d < DR; d += KSTEP) {
                            const bf16v8 r8 = ld_glob8(rrow + d);
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dot[g] = dot8(r8, ld_lds8(qrsm + g * DR + d), dot[g]);
                        }
#pragma unroll
                        for (int g = 0; g < GF; g++) s0[g] = dot[g];
                    }
                    /* Close the row across the KL lanes, then scale ONCE. The two fp8 dots are
                     * already dequantized above, so the same butterfly serves both dtypes. */
#pragma unroll
                    for (int g = 0; g < GF; g++) {
#pragma unroll
                        for (unsigned off = 1; off < KL; off <<= 1)
                            s0[g] += __shfl_xor(s0[g], (int)off, 64);
                        s0[g] *= FA_SCALE(scale);
                    }
                }
                if (ksub == 0) {
#pragma unroll
                    for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + rl] = s0[g];
                }
            }
            /* This thread's OWN row's scores, for the pe below. */
            float s[GF];
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];

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
            /* FP8: `vs[c]` is the row dequant, folded into the softmax weight `pw` — ONE multiply
             * per (row, head-group) instead of one per element, and the f32 accumulation order is
             * otherwise identical to bf16. The scale load is the SAME address for all NDT=64 lanes
             * covering a row, so it is a broadcast off one cache line, not 64 requests. */
            unsigned r = grp;
            for (; r + (VU - 1) * NG < rmax; r += VU * NG) {
                bf16v8 vv[VU];
                float vsf[VU];
#pragma unroll
                for (int c = 0; c < VU; c++) {
                    const unsigned t = kv0 + r + (unsigned)c * NG;
                    const size_t vrow = GATHER ? (size_t)(unsigned)ibase[t] : (t & kv_mask);
                    if constexpr (FP8) {
                        vv[c] = fp8v8_to_bf16v8(ld_glob_fp8v8(cb8 + vrow * DK + dbase));
                        vsf[c] = csc[vrow];
                    } else {
                        vv[c] = ld_glob8(cbase + vrow * DK + dbase);
                    }
                }
#pragma unroll
                for (int c = 0; c < VU; c++) {
#pragma unroll
                    for (int g = 0; g < GF; g++) {
                        float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG];
                        if constexpr (FP8) pw *= vsf[c];
#pragma unroll
                        for (int u = 0; u < 8; u++) oacc[g][u] += pw * bf2f(vv[c][u]);
                    }
                }
            }
            for (; r < rmax; r += NG) {
                const size_t vrow = GATHER ? (size_t)(unsigned)ibase[kv0 + r] : ((kv0 + r) & kv_mask);
                bf16v8 v;
                float vsf = 1.0f;
                if constexpr (FP8) {
                    v = fp8v8_to_bf16v8(ld_glob_fp8v8(cb8 + vrow * DK + dbase));
                    vsf = csc[vrow];
                } else {
                    v = ld_glob8(cbase + vrow * DK + dbase);
                }
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    float pw = Ssm[g * FA_DEC_TILE + r];
                    if constexpr (FP8) pw *= vsf;
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
                st_act<float>(&op[d], acc);
            }
            if (tid == 0) {
                float* ml = mlpart + (oh * nsplit + sp) * 2;
                st_act<float>(&ml[0], m_st[g]);
                st_act<float>(&ml[1], l_st[g]);
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

/* FP8 (e4m3) latent-KV MLA PREFILL. Same wrapper, same body, `FP8=true`.       [MLA-FP8-KV]
 *
 * It "falls out" of the decode kernel for the reason the bf16 prefill did: absorption removes the
 * per-head K/V reconstruction that would make a prefill loop structurally different, and the
 * causal mask is already in `keep`. What it does NOT get for free is the WRITE side — a prefill
 * chunk quantizes `clen` latent rows, one scale each, and `d_headnorm_rope_fp8` already does that
 * per (token, head) with `ntok = clen`. */
template <int DK, int DR, int GF>
__device__ void d_flash_mla_prefill_fp8(float* __restrict__ Opart, float* __restrict__ mlpart,
                                        const bf16* __restrict__ Qabs,
                                        const bf16* __restrict__ Qrope,
                                        const bf16* __restrict__ Ckv,
                                        const bf16* __restrict__ Krope,
                                        const int* __restrict__ kv_len, unsigned n_batch,
                                        unsigned n_tok, unsigned n_head, unsigned kv_stride,
                                        unsigned window, float scale, unsigned kv_mask,
                                        unsigned slice, unsigned nblk, float* lds,
                                        const float* __restrict__ kv_scale, unsigned krot_fp8) {
    d_flash_mla_decode<DK, DR, GF, false, /*FP8=*/true>(
        Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len, n_batch, n_head, kv_stride, window, scale,
        /*nsplit*/ 1, kv_mask, slice, nblk, lds, nullptr, 0, n_tok, kv_scale, krot_fp8);
}

/* ============================================================================
 * MLA PREFILL, TILED MFMA — the kernel a prefill actually wants.     [DEEPSEEK-MLA]
 *
 * The wrappers above run the DECODE body with n_tok > 1. That is exact, and it is the
 * wrong shape: `d_flash_mla_decode`'s work item is ONE query token, so a T-token chunk
 * re-streams the whole causal latent prefix T times and does every dot product on the
 * VECTOR ALU. Both costs are structural, and both are what this kernel removes:
 *
 *   latent traffic  T * n_grp rows  ->  ceil(T/BQ) * n_head rows   (BQ query rows share
 *                                       ONE staged LDS tile; at BQ=64, n_head_local=12,
 *                                       GF=4 that is 16x less)
 *   score+PV math   scalar v_fma    ->  32x32x16 bf16 MFMA
 *
 * WHY THIS IS NOT `d_flash_prefill<512>` WITH AN EXTRA TERM. Two departures, both
 * forced by MLA's absorption:
 *   1. K == V == the latent. `d_flash_prefill` stages K (full D) and V (one output
 *      chunk) as SEPARATE LDS tiles and re-stages V per output chunk. Here they are the
 *      same rows, so ONE tile serves both and the output chunking disappears with it.
 *   2. The score carries a DR-wide rope addendum against a SEPARATE cache, so the QK
 *      contraction is DK+DR while the PV contraction is DK. They are staged adjacent in
 *      one padded row (`[latent | rope]`) so a k-step reads either without a branch.
 *
 * WAVE MAP. The D=512 output is what sets the register budget: a wave owning all 512
 * output columns needs 16 f32x16 = 256 AccVGPRs and there is no room for it. So the
 * PLOW_WAVES waves split as (M-tile x column-group): WPM waves share one 32-row query
 * M-tile and divide DK between them, giving CPW = DK/WPM columns and NDT = CPW/32
 * accumulators per wave. At WPM=4 that is 64 AccVGPRs — the same accumulator budget
 * `d_flash_prefill` proved at D=512.
 *
 * The cost is that the WPM waves of an M-tile each recompute the same QK^T. That is not
 * a regression: `d_flash_prefill` pays exactly the same 4x at D=512 through its NCH
 * output chunks. The difference is where the redundancy lands — its chunks re-stage the
 * cache from HBM, these waves re-read ONE staged tile from LDS.
 *
 * Q LIVES IN LDS, staged once per work item. `d_flash_prefill` re-reads Q fragments from
 * global on every KV tile because it has no LDS to spare after K and V; folding V away
 * frees exactly the room to hoist Q, and it must be hoisted here because WPM waves would
 * otherwise issue the same global read WPM times per tile.
 *
 * FP8 LATENT: the cache is e4m3 with one f32 scale per row, and NOTHING is dequantized
 * into LDS. e4m3 is exactly representable in bf16 (3 mantissa bits into 7), so the staged
 * tile is the raw value and the scale is applied where it is free and exact:
 *   - score: the latent and rope QK go to SEPARATE accumulators, and a lane's whole
 *     accumulator column is one kv row, so `s_lat * cs + s_rope * rs` is one multiply per
 *     lane — the same association the scalar kernel uses (`dotc * cs + dotr * rs`).
 *   - PV: the scale varies along the CONTRACTION axis and cannot leave the sum, so it is
 *     folded into P, which is rounded to bf16 anyway. l_st must NOT see it: cs belongs to
 *     V's magnitude, not to the probability.
 * Dequantizing into LDS instead (what the dense `d_flash_prefill` FP8KV arm does) would
 * round scale*value into bf16 and lose precision this ordering keeps.
 *
 * CAUSAL LOAD BALANCE. Work is one (batch, q-tile, head); q-tile qt attends ~qt tiles, so
 * a plain index makes the last block do NQ times the first block's work. `mla_pf_fold`
 * pairs the ends (0, NQ-1, 1, NQ-2, ...) so the tiles any one block draws average out.
 *
 * NOT USED BY THE GATHERED PREFILL. `d_flash_gather_prefill` selects a DIFFERENT top_k
 * set per query token, so a tile of query rows has no common KV range to stage — the
 * tiling premise fails and op 55 stays on the scalar body below.
 * ==========================================================================*/
/* Route ops 51/110 at the tiled kernel. Default ON; 0 restores the decode-body wrappers,
 * which is what the numeric gate diffs against — keep both reachable so "the tiled kernel
 * agrees with the exact one" stays a runnable claim and not a historical note. */
#ifndef PLOW_MLA_PF_MFMA
#define PLOW_MLA_PF_MFMA 1
#endif
/* WAVES PER QUERY M-TILE. Four on CDNA4; on CDNA3 every wave shares ONE M-tile.
 *
 * This is an LDS decision, not a tuning one. The arena is Qsm[BQ][DK+DR+PAD] + Ksm + P, and
 * BQ = (PLOW_WAVES / WPM) * 32 -- so at eight waves WPM=4 gives BQ=64 and a Qsm of 37,376 B
 * before K or P is placed. MEASURED on gfx942, DK=512/DR=64, every combination that could
 * fit 64 KiB:
 *
 *   waves  WPM  KSPLIT   LDS      fits 65536
 *       8    4       1   128,520  no
 *       8    4       2    99,848  no
 *       8    4       4    91,656  no
 *       8    8       1    91,144  no
 *       8    8       2    64,520  YES  (VGPR 256, AGPR 0, occ 2, spill 6)
 *       4    4       2    60,416  YES
 *
 * KSPLIT=8 does not exist to try: it trips the `DKC % CPW == 0` static assert. So WPM=PLOW_WAVES
 * with KSPLIT=2 is the ONLY eight-wave layout on CDNA3, and it is also the better one -- it keeps
 * occupancy 2, where the four-wave flash object runs at occupancy 1 with 98 VGPR spills.
 *
 * Until this was set, `FA_MLA_PF_LDS_FLOATS(512,64)*4 <= PLOW_LDS_MAX_BYTES` was FALSE on gfx942,
 * which compiled the tiled MFMA MLA prefill -- the shipped path, `PLOW_MLA_PF_MFMA` -- out of the
 * object entirely. GLM/DeepSeek/Kimi had no MLA prefill kernel on CDNA3 at all, and because the
 * guard is a `#if` around the definition rather than an error, the object built clean and the
 * absence surfaced only as a missing HSA symbol at load. */
#ifndef PLOW_MLA_PF_WPM
#if PLOW_CDNA4
#define PLOW_MLA_PF_WPM 4
#else
#define PLOW_MLA_PF_WPM PLOW_WAVES
#endif
#endif
/* Waves sharing one 32-row query M-tile, and the M-tiles that leaves. Clamped so the
 * 4-wave flash object (PLOW_WAVES=4) degrades to one M-tile instead of dividing to 0. */
#define FA_MLA_PF_WPM ((PLOW_WAVES < PLOW_MLA_PF_WPM) ? PLOW_WAVES : PLOW_MLA_PF_WPM)
#define FA_MLA_PF_NMT (PLOW_WAVES / FA_MLA_PF_WPM)
#define FA_MLA_PF_BQ (FA_MLA_PF_NMT * 32)
/* One padded row holds [latent DK | rope DR] so a k-step reads either side unbranched. */
#define FA_MLA_PF_STRIDE(DK, DR) ((DK) + (DR) + FA_PAD)

/* LATENT SPLIT — how many passes the KV tile's latent is staged in.
 *
 * 1 is the historical layout and stays byte-identical. >1 exists because this arena does not
 * fit a 64 KiB workgroup and CANNOT be packed to: at DK=512 the two bf16 tiles alone,
 * Qsm[32][512] + Ksm[32][512], are EXACTLY 65,536 B, so rope and P have nowhere to go. There
 * is no padding trick left to find -- the only lever is staging less of the latent at once.
 *
 * The cost is re-staging. QK needs the whole latent, PV needs only this wave's CPW-wide column
 * slice, so a tile is staged KSPLIT times for QK and then ONE chunk is re-staged for the waves
 * whose slice is not the one left resident: KSPLIT+1 stagings instead of 1. Nearly all of that
 * is L2 -- the chunk was read moments earlier by this same workgroup -- and MLA prefill sits at
 * ~1000 FLOP/byte against MI300X's ~245 roofline, so it stays compute-bound with room to spare. */
/* One on CDNA4, which has the 160 KiB to stage the whole latent. Two on CDNA3 -- see the
 * measured table above `PLOW_MLA_PF_WPM`: it is the only split that fits 64 KiB at either
 * wave count, and the re-staging it costs is L2-resident against a ~1000 FLOP/byte body. */
#ifndef PLOW_MLA_PF_KSPLIT
#if PLOW_CDNA4
#define PLOW_MLA_PF_KSPLIT 1
#else
#define PLOW_MLA_PF_KSPLIT 2
#endif
#endif
/* The K tile only ever holds one latent chunk (plus the whole rope strip). */
#define FA_MLA_PF_KSTRIDE(DK, DR) ((DK) / PLOW_MLA_PF_KSPLIT + (DR) + FA_PAD)
/* PLOW_MLA_PF_QK1 — compute QK^T + softmax on ONE wave per M-tile and share the result.
 *
 * On CDNA3 the 64 KiB arena forces WPM = PLOW_WAVES (the measured table above), so ALL
 * EIGHT waves of a workgroup run the IDENTICAL QK score (NK_CH*KSP + NK_ROPE = 20 MFMAs)
 * and the identical softmax, then do only NDT*BKV/MFMA_K = 4 PV MFMAs each — ~5/6 of the
 * workgroup's matrix-core issue is redundant recomputation. gfx950 pays 4x and has the
 * LDS to not care; at 8x it is the dominant waste of the CDNA3 MLA prefill.
 *
 * The P matrix is ALREADY shared through Psm, so the only thing the other waves lack is
 * the per-row online-softmax correction factor for their O accumulators. Under this knob
 * column-group 0 computes QK + softmax, publishes P (as before) plus corr[BQ] through a
 * small LDS strip (Csm), and every wave applies the SAME corr after the existing P
 * barrier — identical values, identical multiplies, byte-identical output by
 * construction. Default 0 compiles the historical redundant path verbatim. */
#ifndef PLOW_MLA_PF_QK1
#define PLOW_MLA_PF_QK1 0
#endif
/* PLOW_MLA_PF_ABL — ABLATION PROBES for the tiled MLA prefill body. WRONG OUTPUT BY
 * CONSTRUCTION; ceiling instruments only, never a serve asset (same class as PLOW_XR_NOWAIT).
 * Each arm deletes ONE cost term while keeping the loop structure, barrier count and
 * work-item map intact, so (baseline - arm) prices that term on the real model:
 *   1  no-softmax     row-max/exp/corr replaced by a mask-only P (data-dependent on the
 *                     score so the QK MFMAs stay live)
 *   2  no-QK-MFMA     score MFMA loops skipped; staging, softmax(P=junk), PV all remain
 *   3  stage-once     K/rope staged only for the FIRST kv tile; later tiles reuse stale LDS
 *   4  no-PV-restage  the PV phase's KSPLIT re-stage skipped; PV reads the resident chunk */
#ifndef PLOW_MLA_PF_ABL
#define PLOW_MLA_PF_ABL 0
#endif
/* PLOW_MLA_PF_SMX — SPLIT the online softmax across the WPM column-group waves.
 *
 * Under the CDNA3 KSPLIT layout every wave of an M-tile computes the IDENTICAL row
 * reductions, exps and corrections for all 32 rows — measured on the 2026-08-07 ablation
 * (perf-data/plow-gfx942/glm52-mla-pf-decomposition.md): 5.57 of the 18.34 ms per-layer
 * flash span at T=8192, as large as the entire QK MFMA term. PLOW_MLA_PF_QK1 attacked the
 * same redundancy by SERIALIZING on column-group 0 and measured NET NEGATIVE (+1.0 ms):
 * seven waves idle at a barrier while one does 8x work. This knob splits instead: each
 * column-group OWNS 32/WPM query rows of the M-tile — the guard `(row*WPM)/32 == cgrp` is
 * half-wave-uniform under mfma_acc_m's layout, so the owning half-wave's shfl partners all
 * stay active — computes the reductions, exp and P store for those rows only, and publishes
 * the per-row correction through the Xsm strip. Every wave applies the SAME correction to
 * its accumulators after the existing P barrier, exactly where PLOW_MLA_PF_QK1 applied its:
 * identical values, identical multiply order, BIT-IDENTICAL output by construction; the
 * redundant row work drops WPM-fold with no extra barrier.
 *
 * Default ON for CDNA3 (WPM=8, where the redundancy is 8x); OFF on CDNA4 (WPM=4, unmeasured
 * there). PLOW_MLA_PF_SMX=0 restores the historical redundant path verbatim. */
#ifndef PLOW_MLA_PF_SMX
#if PLOW_CDNA4
#define PLOW_MLA_PF_SMX 0
#else
#define PLOW_MLA_PF_SMX 1
#endif
#endif
#if PLOW_MLA_PF_SMX && PLOW_MLA_PF_QK1
#error "PLOW_MLA_PF_SMX and PLOW_MLA_PF_QK1 are alternative softmax dedups; pick one"
#endif
/* P buffers. Every wave of an M-tile computes the SAME softmax -- they differ only in which
 * output COLUMNS they own -- so one buffer per M-tile is enough. Only claimed when the split is
 * on, so the unsplit layout (and gfx950's object) is unchanged. Under PLOW_MLA_PF_QK1 only
 * column-group 0 writes P, so one buffer per M-tile regardless of the split. */
#define FA_MLA_PF_NP                                                                      \
    (((PLOW_MLA_PF_KSPLIT > 1) || PLOW_MLA_PF_QK1 || PLOW_MLA_PF_SMX) ? FA_MLA_PF_NMT     \
                                                                      : PLOW_WAVES)
/* Qsm[BQ][STRIDE] + Ksm[BKV=32][KSTRIDE] + Psm[NP][32][BKV], in floats (bf16 => /2),
 * plus corr[BQ] floats when QK1 or SMX shares the softmax correction. */
#define FA_MLA_PF_LDS_FLOATS(DK, DR)                                                      \
    ((FA_MLA_PF_BQ * FA_MLA_PF_STRIDE(DK, DR) + 32 * FA_MLA_PF_KSTRIDE(DK, DR) +          \
      FA_MLA_PF_NP * 32 * 32 + 1) / 2 +                                                   \
     ((PLOW_MLA_PF_QK1 || PLOW_MLA_PF_SMX) ? FA_MLA_PF_BQ : 0))

/* Pair the causal ends: 0, NQ-1, 1, NQ-2, ... A bijection on [0,NQ) for either parity. */
__device__ __forceinline__ unsigned mla_pf_fold(unsigned qt, unsigned nq) {
    return (qt & 1u) ? (nq - 1u - (qt >> 1)) : (qt >> 1);
}

/* IS THE TILED KERNEL THE RIGHT ONE FOR THIS PACKET — a FILL question, and the reason the
 * scalar body is still compiled in rather than deleted.
 *
 * The tiling buys arithmetic and traffic and SPENDS work items: it turns n_tok*n_grp items
 * into ceil(n_tok/BQ)*n_head. On a full prefill chunk that is still thousands and the trade
 * is enormously good. On a short chunk — the ragged tail `plan_chunks` leaves, or a short
 * prompt that is one small chunk — it is not: at n_tok=128 and K3's TP8 n_head=12 the tiled
 * kernel has 24 items for 256 CUs and 91% of the machine idles.
 *
 * Measured on gfx950, n_head=12, fp8 latent, ctx=32768 (tiled vs scalar), items = the
 * tiled kernel's work-item count against nblk=256:
 *     n_tok 8192  50.7 vs 141.6 ms   2.79x   (1536 items, full)
 *     n_tok 1024   9.7 vs  19.9 ms   2.06x   ( 192, 75%)
 *     n_tok  704   9.7 vs  15.0 ms   1.54x   ( 132, 52%)
 *     n_tok  640   9.7 vs  13.4 ms   1.38x   ( 120, 47%)
 *     n_tok  512   9.7 vs  10.1 ms   1.04x   (  96, 38%)  <- the crossover, measured
 *     n_tok  128   9.7 vs   3.4 ms   0.35x   (  24,  9%)  <- the regression
 * Full-fill tiled is ~120 TF/s against the scalar's ~43, so the tiled kernel is ahead
 * exactly while its fill exceeds 43/120 = 0.36 — which is where the sweep turns over, to
 * within one step. The rule is that ratio, 3/8, and it is expressed against nblk (the real
 * grid) rather than a token count so it tracks a packet emitted for a different CU count.
 *
 * Taken from the FP8 arm deliberately: bf16 scalar is 3x slower again (13.7 TF/s, it moves
 * twice the bytes), so bf16 crosses over much earlier and this threshold merely gives up a
 * little there. Erring toward the shipped arm is the safe direction.
 *
 * The DEEPER fix is to stop spending items: split the KV range `nsplit` ways as
 * `d_flash_prefill` does, which is what makes a short chunk fill the machine instead of
 * choosing between two decompositions. That needs the emitter to size Opart for nsplit>1
 * and to tell `MlaMergeFold`, so it is a packet change and not a kernel one. Until then
 * this picks the better of the two and never the worse. */
__device__ __forceinline__ bool mla_pf_tiled_fills(unsigned n_batch, unsigned n_tok,
                                                   unsigned n_head, unsigned nblk) {
    const unsigned n_qt = (n_tok + FA_MLA_PF_BQ - 1) / FA_MLA_PF_BQ;
    return (unsigned long long)n_batch * n_qt * n_head * 8ull >= (unsigned long long)nblk * 3ull;
}

template <int DK, int DR, bool FP8 = false>
__device__ void d_flash_mla_prefill_mfma(float* __restrict__ Opart,
                                         float* __restrict__ mlpart,
                                         const bf16* __restrict__ Qabs,
                                         const bf16* __restrict__ Qrope,
                                         const bf16* __restrict__ Ckv,
                                         const bf16* __restrict__ Krope,
                                         const int* __restrict__ kv_len, unsigned n_batch,
                                         unsigned n_tok, unsigned n_head, unsigned kv_stride,
                                         unsigned window, float scale, unsigned kv_mask,
                                         unsigned slice, unsigned nblk, float* lds_,
                                         const float* __restrict__ kv_scale,
                                         unsigned krot_fp8) {
    constexpr int BKV = 32;                       /* one MFMA N-tile of KV per pass */
    constexpr int WPM = FA_MLA_PF_WPM;
    constexpr int NMT = FA_MLA_PF_NMT;
    constexpr int BQ = FA_MLA_PF_BQ;              /* query rows per workgroup       */
    constexpr int STRIDE = FA_MLA_PF_STRIDE(DK, DR);
    constexpr int KSP = PLOW_MLA_PF_KSPLIT;       /* latent chunks per KV tile      */
    constexpr int DKC = DK / KSP;                 /* latent columns staged at once  */
    constexpr int KSTRIDE = FA_MLA_PF_KSTRIDE(DK, DR);
    constexpr int NK_CH = DKC / MFMA_K;           /* QK k-steps within one chunk    */
    constexpr int NK_ROPE = DR / MFMA_K;          /* + rope k-steps (DR=64 => 4)    */
    constexpr int CPW = DK / WPM;                 /* output cols this wave owns     */
    constexpr int NDT = CPW / MFMA_N;             /* output d-tiles this wave       */
    static_assert(DK % (WPM * MFMA_N) == 0, "DK must divide into WPM column-groups of 32");
    static_assert(DK % KSP == 0, "the latent must split into whole chunks");
    static_assert(DKC % MFMA_K == 0, "a latent chunk must be a whole number of k-steps");
    /* PV reads only this wave's CPW-wide slice, so that slice must sit inside ONE chunk --
     * otherwise a wave would need two chunks resident at once and the split buys nothing. */
    static_assert(DKC % CPW == 0, "a latent chunk must hold whole column-groups");
    static_assert(DR % MFMA_K == 0, "rope width must be a whole number of k-steps");

    bf16* lds = (bf16*)lds_;
    bf16* Qsm = lds;                              /* [BQ][STRIDE]   latent|rope query rows */
    bf16* Ksm = Qsm + BQ * STRIDE;                /* [BKV][KSTRIDE] one latent CHUNK|rope  */
    bf16* Psm = Ksm + BKV * KSTRIDE;              /* [NP][32][BKV]  P transpose            */
#if PLOW_MLA_PF_QK1 || PLOW_MLA_PF_SMX
    /* [BQ] per-row online-softmax corrections, published with P (by cgrp 0 under QK1, by
     * each row's OWNING column-group under SMX). The Psm block is an even bf16 count, so
     * this lands 4-byte aligned off the float* arena base. */
    float* Csm = (float*)(Psm + FA_MLA_PF_NP * 32 * BKV);
#endif

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63;
    const unsigned wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);    /* l % 32: A-row and B-col alike */
    const unsigned accn = mfma_acc_n(lane);       /* l % 32: this lane's kv column */
    const unsigned m_tile = wave / WPM;           /* which 32-row query block      */
    const unsigned cgrp = wave % WPM;             /* which output column-group     */
    const unsigned ncol0 = cgrp * CPW;
    const int my_ch = (int)(ncol0 / DKC);         /* the latent chunk PV needs (wave-uniform) */

    const unsigned n_qt = (n_tok + BQ - 1) / BQ;
    const unsigned n_work = n_batch * n_qt * n_head;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head;
        const unsigned rest = w / n_head;
        const unsigned qt = mla_pf_fold(rest % n_qt, n_qt);
        const unsigned b = rest / n_qt;

        const unsigned len = (unsigned)kv_len[b];
        /* Chunk row 0 sits at len - n_tok, so chunk row qi is at q_pos0 + qi. */
        const unsigned q_pos0 = len - n_tok;
        const unsigned q_base = qt * BQ;          /* first chunk row of this q-tile */
        const unsigned my_q0 = q_base + m_tile * 32;

        /* KV bounds MUST be workgroup-uniform — the loop body has __syncthreads() and the
         * M-tiles have different causal horizons. Bound by the LAST row of the whole tile
         * and let the per-element mask do the exact work (same rule as d_flash_prefill). */
        const unsigned q_tile_last = q_pos0 + q_base + BQ - 1;
        const unsigned kv_end = (q_tile_last + 1 < len) ? (q_tile_last + 1) : len;
        const unsigned q_tile_first = q_pos0 + q_base;
        const unsigned win_lo =
            (window && q_tile_first >= window) ? (q_tile_first - window + 1) : 0u;
        const unsigned kv_lo = (win_lo / BKV) * BKV;

        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const auto* rbase = as_glob(Krope) + (size_t)b * kv_stride * DR;
        const unsigned char* cb8 = (const unsigned char*)Ckv + (size_t)b * kv_stride * DK;
        const unsigned char* rb8 = (const unsigned char*)Krope + (size_t)b * kv_stride * DR;
        /* Two strips, latent then rope, exactly as the scalar kernel splits them. */
        const float* csc = FP8 ? kv_scale + (size_t)b * kv_stride : nullptr;
        const float* rsc = FP8 ? kv_scale + (size_t)(n_batch + b) * kv_stride : nullptr;

        /* Q -> LDS, ONCE per work item: all WPM waves of an M-tile read these rows. */
        __syncthreads();
        for (unsigned e = tid * 8; e < (unsigned)(BQ * DK); e += PLOW_THREADS * 8) {
            const unsigned r = e / DK, c = e % DK;
            const unsigned qi = q_base + r;
            bf16 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            /* ld_glob8, not a memcpy off the pointer: an ADDRESS-SPACE-qualified source
             * makes __builtin_memcpy fall back to the host symbol and the device compile
             * fails with no location. Load to a value first, copy from that. */
            if (qi < n_tok) {
                const bf16v8 v =
                    ld_glob8(&Qabs[((((size_t)b * n_tok + qi) * n_head) + h) * DK + c]);
                __builtin_memcpy(t, &v, 16);
            }
            __builtin_memcpy(&Qsm[r * STRIDE + c], t, 16);
        }
        for (unsigned e = tid * 8; e < (unsigned)(BQ * DR); e += PLOW_THREADS * 8) {
            const unsigned r = e / DR, c = e % DR;
            const unsigned qi = q_base + r;
            bf16 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (qi < n_tok) {
                const bf16v8 v =
                    ld_glob8(&Qrope[((((size_t)b * n_tok + qi) * n_head) + h) * DR + c]);
                __builtin_memcpy(t, &v, 16);
            }
            __builtin_memcpy(&Qsm[r * STRIDE + DK + c], t, 16);
        }

        f32x16 oacc[NDT];
#pragma unroll
        for (int t = 0; t < NDT; t++) oacc[t] = (f32x16)(0.0f);
        float m_st[16], l_st[16];
#pragma unroll
        for (int i = 0; i < 16; i++) { m_st[i] = FA_NEG_INF; l_st[i] = 0.0f; }

        for (unsigned kv0 = kv_lo; kv0 < kv_end; kv0 += BKV) {
            /* Whole tile outside the sliding window — uniform, so every wave skips it. */
            if (window && kv0 + BKV <= win_lo) continue;

            /* Stage the KV tile's latent one CHUNK at a time (KSP=1 => the whole thing, the
             * historical path). QK needs every chunk, so the score accumulates across passes;
             * the rope strip rides the last one because it is small and needed exactly once. */
            auto stage_k_chunk = [&](int ch) {
#if PLOW_MLA_PF_ABL == 3
                if (kv0 != kv_lo) return; /* probe: later tiles reuse stale LDS */
#endif
                for (unsigned e = tid * 8; e < (unsigned)(BKV * DKC); e += PLOW_THREADS * 8) {
                    const unsigned r = e / DKC, c = e % DKC;
                    const unsigned kv = kv0 + r;
                    bf16 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (kv < kv_end) {
                        const size_t row = (size_t)(kv & kv_mask);
                        const unsigned sc = (unsigned)ch * DKC + c;
                        if constexpr (FP8) {
                            /* RAW e4m3 -> bf16, lossless. The row scale is applied after the
                             * MFMA (score) or folded into P (PV) — never here. One 16-byte copy
                             * of the whole octet: bf16v8 and bf16[8] are the same 16 bytes, and
                             * element-wise `bf16_t -> bf16` is the one direction that does not
                             * fold and asks the compiler for a host memcpy. */
                            const bf16v8 dv = fp8v8_to_bf16v8(ld_glob_fp8v8(cb8 + row * DK + sc));
                            __builtin_memcpy(t, &dv, 16);
                        } else {
                            const bf16v8 dv = ld_glob8(cbase + row * DK + sc);
                            __builtin_memcpy(t, &dv, 16);
                        }
                    }
                    __builtin_memcpy(&Ksm[r * KSTRIDE + c], t, 16);
                }
            };

            /* S = Q_abs . C_kv (+ Q_rope . K_rope), kept in SEPARATE accumulators so the
             * fp8 row scales can be applied per kv column below. A lane's whole 16-entry
             * accumulator column is ONE kv row, which is what makes that one multiply. */
            f32x16 s_lat = (f32x16)(0.0f);
            f32x16 s_rope = (f32x16)(0.0f);
#pragma unroll 1
            for (int ch = 0; ch < KSP; ch++) {
                __syncthreads();                   /* previous chunk's (or tile's) reads are done */
                stage_k_chunk(ch);
                if (
#if PLOW_MLA_PF_ABL == 3
                    kv0 == kv_lo &&
#endif
                    ch == KSP - 1) {
                    for (unsigned e = tid * 8; e < (unsigned)(BKV * DR); e += PLOW_THREADS * 8) {
                        const unsigned r = e / DR, c = e % DR;
                        const unsigned kv = kv0 + r;
                        bf16 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        if (kv < kv_end) {
                            const size_t row = (size_t)(kv & kv_mask);
                            if constexpr (FP8) {
                                if (krot_fp8) {
                                    const bf16v8 dv =
                                        fp8v8_to_bf16v8(ld_glob_fp8v8(rb8 + row * DR + c));
                                    __builtin_memcpy(t, &dv, 16);
                                } else {
                                    const bf16v8 dv = ld_glob8(rbase + row * DR + c);
                                    __builtin_memcpy(t, &dv, 16);
                                }
                            } else {
                                const bf16v8 dv = ld_glob8(rbase + row * DR + c);
                                __builtin_memcpy(t, &dv, 16);
                            }
                        }
                        __builtin_memcpy(&Ksm[r * KSTRIDE + DKC + c], t, 16);
                    }
                }
                __syncthreads();                   /* Qsm (first pass) and this chunk visible */

#if PLOW_MLA_PF_QK1
                if (cgrp == 0)
#endif
#if PLOW_MLA_PF_ABL != 2
#pragma unroll 1
                for (int kk = 0; kk < NK_CH; kk++) {
                    const unsigned d0 = mfma_frag_k(lane, kk * MFMA_K);
                    bf16x8 qf, kf;
                    bf16 tq[8], tk[8];
                    __builtin_memcpy(tq, &Qsm[(m_tile * 32 + frow) * STRIDE + ch * DKC + d0], 16);
                    __builtin_memcpy(tk, &Ksm[frow * KSTRIDE + d0], 16);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        bf16_t a, c;
                        __builtin_memcpy(&a, &tq[j], 2);
                        __builtin_memcpy(&c, &tk[j], 2);
                        qf[j] = a;
                        kf[j] = c;
                    }
                    s_lat = plow_mfma_bf16_32x32(qf, kf, s_lat);
                }
#endif
                if (
#if PLOW_MLA_PF_QK1
                    cgrp == 0 &&
#endif
                    ch == KSP - 1) {
#if PLOW_MLA_PF_ABL != 2
#pragma unroll
                    for (int kk = 0; kk < NK_ROPE; kk++) {
                        const unsigned d0 = mfma_frag_k(lane, kk * MFMA_K);
                        bf16x8 qf, kf;
                        bf16 tq[8], tk[8];
                        __builtin_memcpy(tq, &Qsm[(m_tile * 32 + frow) * STRIDE + DK + d0], 16);
                        __builtin_memcpy(tk, &Ksm[frow * KSTRIDE + DKC + d0], 16);
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            bf16_t a, c;
                            __builtin_memcpy(&a, &tq[j], 2);
                            __builtin_memcpy(&c, &tk[j], 2);
                            qf[j] = a;
                            kf[j] = c;
                        }
                        s_rope = plow_mfma_bf16_32x32(qf, kf, s_rope);
                    }
#endif
                }
            }

            /* This lane's kv row, and its two dequant scales. */
            const unsigned kg = kv0 + accn;
            const bool kv_in = (kg < kv_end);
            const size_t krow = (size_t)(kg & kv_mask);
            const float cs = (FP8 && kv_in) ? csc[krow] : 1.0f;
            const float rs = (FP8 && kv_in && krot_fp8) ? rsc[krow] : 1.0f;

            /* Mask, then online softmax. A row of S lives entirely inside ONE half-wave,
             * so the row reductions must stop at 32 lanes. Under PLOW_MLA_PF_QK1 only
             * column-group 0 holds S; it computes the softmax, scales ITS accumulators here
             * (same in-loop position as the historical path) and publishes corr[row] through
             * Csm; the other waves apply the identical factors after the P barrier below. */
            float p[16];
#if PLOW_MLA_PF_QK1
            if (cgrp == 0) {
#endif
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned qi = my_q0 + mfma_acc_m(lane, i);
                const unsigned qg = q_pos0 + qi;
                const bool valid = (qi < n_tok) && kv_in && (kg <= qg) &&
                                   (!window || (qg - kg) < window);
                const float sv = s_lat[i] * cs + s_rope[i] * rs;
                p[i] = valid ? (sv * FA_SCALE(scale)) : FA_NEG_INF;
            }
#if PLOW_MLA_PF_ABL == 1
            /* probe: mask-only P, still data-dependent on the score so QK stays live;
             * no row reductions, no exp, no accumulator corrections. */
#pragma unroll
            for (int i = 0; i < 16; i++) {
                p[i] = (p[i] == FA_NEG_INF) ? 0.0f : fmaxf(p[i] * 1e-30f, 0.0f) + 1e-6f;
                l_st[i] += p[i];
            }
#elif PLOW_MLA_PF_SMX
            /* SPLIT SOFTMAX (see the PLOW_MLA_PF_SMX header): this column-group owns rows
             * [cgrp*32/WPM, (cgrp+1)*32/WPM) of the M-tile. The ownership guard is
             * HALF-WAVE-uniform — mfma_acc_m depends on the lane only through lane/32 — so
             * the owning half-wave's shfl partners are all active. Non-owned slots leave
             * p[] unread and m_st/l_st untouched; every wave (owner included) applies the
             * published correction from Csm after the P barrier below, which keeps the
             * multiply in the same position relative to the PV accumulate as the
             * historical in-loop scaling: identical values, identical order. */
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned rm = mfma_acc_m(lane, i);
                if (rm * (unsigned)WPM / 32u != cgrp) continue;
                const float rmax = half_wave_max(p[i]);
                const float mnew = fmaxf(m_st[i], rmax);
                const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
                l_st[i] = l_st[i] * corr + half_wave_sum(pe);
                m_st[i] = mnew;
                p[i] = pe;
                /* One lane per kv column: the accn==0 lane of the owning half-wave writes
                 * each owned row exactly once. */
                if (accn == 0) Csm[m_tile * 32 + rm] = corr;
            }
#else
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
#if PLOW_MLA_PF_QK1
                /* One lane per accumulator column suffices: row m appears in every lane of
                 * the half-wave, so the accn==0 lanes cover all 32 rows exactly once. */
                if (accn == 0) Csm[m_tile * 32 + mfma_acc_m(lane, i)] = corr;
#endif
            }
#endif /* PLOW_MLA_PF_ABL == 1 */
#if PLOW_MLA_PF_QK1
            }
#endif

            /* P -> LDS as the PV A-operand (m = query row, k = kv). The fp8 latent scale
             * rides here: O = sum_kv (p * cs) * C_raw, while l_st above stays scale-free.
             *
             * Under the split the waves of an M-tile SHARE one buffer: they ran the same QK over
             * the same rows and the same kv range, so p[] is identical across cgrp -- the (m, l)
             * store at the bottom of this kernel already relies on that. cgrp 0 writes it. */
            bf16* myP =
                Psm +
                ((KSP > 1 || PLOW_MLA_PF_QK1 || PLOW_MLA_PF_SMX) ? m_tile : wave) * 32 * BKV;
            __syncthreads();
#if PLOW_MLA_PF_SMX
            /* Each owner stores ITS rows; across the WPM groups that is every row once. */
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned rm = mfma_acc_m(lane, i);
                if (rm * (unsigned)WPM / 32u == cgrp)
                    myP[rm * BKV + accn] = f2bf(FP8 ? (p[i] * cs) : p[i]);
            }
#else
            if ((KSP == 1 && !PLOW_MLA_PF_QK1) || cgrp == 0) {
#pragma unroll
                for (int i = 0; i < 16; i++)
                    myP[mfma_acc_m(lane, i) * BKV + accn] = f2bf(FP8 ? (p[i] * cs) : p[i]);
            }
#endif
            __syncthreads();
#if PLOW_MLA_PF_QK1
            /* The shared corrections are visible with P. cgrp 0 already scaled in-loop. */
            if (cgrp != 0) {
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const float c = Csm[m_tile * 32 + mfma_acc_m(lane, i)];
#pragma unroll
                    for (int t = 0; t < NDT; t++) oacc[t][i] *= c;
                }
            }
#elif PLOW_MLA_PF_SMX
            /* EVERY wave (owner included) applies the published correction here — before any
             * PV accumulate, exactly where the historical path multiplied in-loop. */
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float c = Csm[m_tile * 32 + mfma_acc_m(lane, i)];
#pragma unroll
                for (int t = 0; t < NDT; t++) oacc[t][i] *= c;
            }
#endif

            /* O += P . V, V = this wave's latent column slice.
             *
             * The LAST chunk staged above is still resident, so the waves whose slice lives in it
             * go first and pay nothing. The rest need their chunk back; that re-stage is the
             * split's whole running cost, and it reads what this workgroup just read. */
#pragma unroll 1
            for (int ch = KSP - 1; ch >= 0; ch--) {
                if (KSP > 1 && ch != KSP - 1) {
                    __syncthreads();               /* the previous chunk's PV reads are done */
#if PLOW_MLA_PF_ABL != 4
                    stage_k_chunk(ch);
#endif
                    __syncthreads();
                }
                if (my_ch != ch) continue;         /* wave-uniform: never splits a barrier */
                const unsigned vcol = ncol0 - (unsigned)ch * DKC;
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
                            bf16 vv = Ksm[(kk + j) * KSTRIDE + vcol + t * MFMA_N + accn];
                            bf16_t vb;
                            __builtin_memcpy(&vb, &vv, 2);
                            vf[j] = vb;
                        }
                        oacc[t] = plow_mfma_bf16_32x32(pf, vf, oacc[t]);
                    }
                }
            }
        }

        /* UNNORMALIZED latent-wide partials, the layout d_flash_merge<DK> + d_o_uv_fold
         * already consume: [b][t][head][nsplit=1][DK]. */
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const unsigned qi = my_q0 + mfma_acc_m(lane, i);
            if (qi >= n_tok) continue;
            const size_t oh = ((size_t)b * n_tok + qi) * n_head + h;
            float* op = Opart + oh * DK;
#pragma unroll
            for (int t = 0; t < NDT; t++)
                st_act<float>(&op[ncol0 + t * MFMA_N + accn], oacc[t][i]);
            /* One wave and one lane per row owns (m, l): cgrp 0 on the historical path
             * (every cgrp computed the same softmax), the row's OWNER under SMX (only it
             * holds that row's m/l). */
#if PLOW_MLA_PF_SMX
            if (accn == 0 && mfma_acc_m(lane, i) * (unsigned)WPM / 32u == cgrp) {
#else
            if (cgrp == 0 && accn == 0) {
#endif
                float* ml = mlpart + oh * 2;
                st_act<float>(&ml[0], m_st[i]);
                st_act<float>(&ml[1], l_st[i]);
            }
        }
        __syncthreads();                           /* Qsm reuse by the next work item */
    }
}

/* ============================================================================
 * MLA PREFILL V2 — full-column waves for the 4-WAVE FLASH OBJECT.   [CDNA3-V2]
 *
 * The 8-wave kernel above is shaped by the 256-register/occ-2 budget: all 8 waves share ONE
 * 32-row Q tile and split the 512 output columns (WPM=8), so the QK^T MFMA is issued 8×
 * redundantly and every KV tile costs a barrier-heavy cross-wave choreography. The 4-wave
 * flash object (PLOW_WG_WAVES=4) runs 512 VGPR + 256 AGPR at occ 1 — the budget for a wave
 * to own ALL 512 output columns exists there, and this kernel is that layout:
 *
 *   - wave w owns Q rows [w*16, w*16+16) of a BQ=64 tile; its 16×512 f32 accumulators are
 *     128 AccVGPRs. No column split, no redundant QK, no cross-wave softmax.
 *   - Q lives ENTIRELY in registers as pre-formed A-fragments (18 × bf16x8 = 72 VGPRs),
 *     loaded once per work item. No Q LDS, no Q barrier.
 *   - one 32-row KV slab in LDS serves BOTH matmuls: MLA's K IS its V (the latent), so the
 *     PV pass reads the same staged rows — the re-stage the 8-wave kernel priced at
 *     0.53 ms/layer does not exist here, and neither does a separate V stream.
 *   - P transposes through a PER-WAVE private LDS strip (1 KiB): same-wave ds ordering
 *     needs no barrier. The whole KV tile costs TWO __syncthreads (stage fence pair)
 *     against the 8-wave kernel's ~8.
 *   - 16×16×(2×16) MFMA via plow_mfma_bf16_16x16 — same MACs/cycle as 32×32 on CDNA3,
 *     lower fragment pressure; maps verified against d_moe_group_glu_mfma:
 *     A: m=lane%16, k=8*(lane/16)+j · B: n=lane%16, k=8*(lane/16)+j · C: n=lane%16,
 *     m=4*(lane/16)+i.
 *
 * Same operand contract as d_flash_mla_prefill (Opart/mlpart unnormalized partials at
 * nsplit=1, FA_SCALE'd base-2 (m,l)); bf16 KV only — the fp8-KV arm keeps the 8-wave body.
 * Dispatched from the flash bucket only (PLOW_MLA_PF_V2_ARM); the host routes
 * FlashMlaPrefill segments there per-program when PLOW_MLA_PF_V2=1 and the bucket is big
 * enough to fill the machine at BQ=64 (exec/amd.rs derive_segments).
 * ========================================================================================== */
#define FA_MLA_PF2_BKV 32
#define FA_MLA_PF2_PAD 8
/* Ablation probes (build axis PLOW_MLA_PF2_ABL, WRONG OUTPUT, never a serve asset):
 * 1 = no K-slab global loads (zeros staged), 2 = no QK MFMA, 3 = no softmax math
 * (pe := 1, no shuffles/exp), 4 = no PV (no P write, no PV MFMA). One term each. */
#ifndef PLOW_MLA_PF2_ABL
#define PLOW_MLA_PF2_ABL 0
#endif
/* Register-prefetch K-slab pipeline, DEFAULT ON (PLOW_MLA_PF2_DBUF=0 restores the
 * fence-exposed staging for A/B). The V2 kernel runs ONE wave per SIMD (occ 1 at its
 * 512-reg budget), so there is no co-resident wave to hide the staging latency behind —
 * ping-pong scheduling is structurally unavailable. Instead each thread ISSUES the next
 * tile's 9 global loads (36 VGPRs) right after the current slab's visibility fence and
 * consumes them at the next commit: the global latency overlaps the whole QK/softmax/PV
 * body, and only the LDS writes + fence pair stay exposed. BIT-IDENTICAL: same bytes
 * into the same LDS slots before the same fence. */
#ifndef PLOW_MLA_PF2_DBUF
#define PLOW_MLA_PF2_DBUF 1
#endif
/* A lazy corr-rescale (skip the identity rescale when the running max did not move) was
 * TRIED HERE AND REJECTED, twice over: the arm measured logits-DIFFERENT from the
 * unconditional path (the identity-skip argument fails somewhere subtle) AND SLOWER
 * (spill 248 -> 293; the divergent per-row branch costs more scheduling room than the
 * skipped multiplies are worth at occ 1). Do not re-try as a branch; a rescale saving
 * here must come from a deferred-rescale restructure, not a guard. */

/* ------------------------------------------------------------------------------------
 * PLOW_MLA_PF_SV — the V-STAGE arm. OPT-IN, default OFF, BIT-IDENTICAL (it moves LDS
 * addresses and load ISSUE ORDER only; every MFMA sees the same operands in the same
 * accumulation order). Three pieces, all aimed at the PV half of the inner loop:
 *
 *  (1) KV-ROW BLOCK SWIZZLE. The PV pass needs V^T (the MFMA B-fragment is B[n=d][k=kv],
 *      so the CONTRACTION axis is kv, which is the MINOR axis of the [kv][d] slab). It
 *      pays that transpose as 8 strided `ds_read_u16` per output tile, 256 per KV tile.
 *      With a row stride of KSTR halves the LDS bank of lane (fr,kg) reading
 *      Ksm[(kg*8+j)*KSTR + t*16+fr] is
 *          bank = ( (KSTR/2)*(8*kg+j) + 8*t + fr/2 ) mod 32
 *      and 8*kg*(KSTR/2) == 0 (mod 32) FOR EVERY KSTR THAT IS A MULTIPLE OF 8 — which is
 *      forced, because the QK pass reads 16-byte `ds_read_b128` fragments out of the same
 *      rows. So kg drops out of the bank index: the four k-groups always collide, four
 *      distinct addresses land in one bank, and every one of the 256 reads is a 4-WAY
 *      BANK CONFLICT (the fr/fr+1 pair shares a dword and broadcasts, so the conflict is
 *      exactly 4x, not 8x). This arm shifts each 8-row kv BLOCK by +16 halves:
 *          krow(kv) = kv*KSTR + 16*(kv>>3)
 *      16 halves = 8 dwords, so the bank index gains an 8*kg term; kg spreads over
 *      {0,8,16,24}, fr/2 over {0..7}, and the 64 lanes cover all 32 banks with one
 *      address each — CONFLICT-FREE. 16 is also a multiple of 8 halves, so the QK
 *      b128 alignment survives, and the swizzle is FREE IN THE ISA: on the PV path
 *      16*kg is j- and t-independent so it folds into the lane's base address register;
 *      on the QK path 16*(2*nt + fr/8) is loop-invariant per lane; on the store path
 *      r>>3 is the compile-time constant it/2 (dbuf) or the per-thread constant `wave`
 *      (rope). Cost: 3*16 halves = 96 B of LDS.
 *  (2) QK K-fragment double buffer — one ds_read_b128 in flight while the previous
 *      fragment feeds its two MFMAs, so the wait is lgkmcnt(1) instead of the full
 *      lgkmcnt(0) drain the single-buffered form forces 36 times per KV tile.
 *  (3) PV V-fragment double buffer — the next output tile's 8 u16 reads issue before the
 *      current tile's MFMAs, deepening the lgkm pipeline from 8 to 16 outstanding.
 *
 * (2) and (3) each add one live bf16x8 (4 VGPR) plus u16 staging registers on a register
 * file that is already 100% allocated at occupancy 1 — see
 * perf-data/plow-gfx942/glm52-flash-streamed-v.md for the measured spill delta. Nothing
 * here streams V from GLOBAL: that variant is priced and REFUTED in the same report (the
 * transpose makes it 256 scattered 2-byte loads per lane per KV tile). */
#ifndef PLOW_MLA_PF_SV
#define PLOW_MLA_PF_SV 0
#endif
/* Extra halves the kv-block swizzle adds to the slab (blocks 1..3 shifted by 16 each). */
#if PLOW_MLA_PF_SV
#define FA_MLA_PF2_SWZ 16
#else
#define FA_MLA_PF2_SWZ 0
#endif
/* LDS bytes: the KV slab (+ swizzle slack) + 4 per-wave P strips. 41,472 B at (512,64),
 * 41,568 B with PLOW_MLA_PF_SV — inside the flash object's 58,368 B `fa` arena, asserted
 * at the dispatch site. */
#define FA_MLA_PF2_LDS_BYTES(DK, DR)                                                     \
    ((FA_MLA_PF2_BKV * ((DK) + (DR) + FA_MLA_PF2_PAD) + 3 * FA_MLA_PF2_SWZ +             \
      4 * 16 * FA_MLA_PF2_BKV) *                                                         \
     2)

template <int DK, int DR, bool GATHER = false, bool FP8 = false>
__device__ void d_flash_mla_prefill_v2(float* __restrict__ Opart, float* __restrict__ mlpart,
                                       const bf16* __restrict__ Qabs,
                                       const bf16* __restrict__ Qrope,
                                       const bf16* __restrict__ Ckv,
                                       const bf16* __restrict__ Krope,
                                       const int* __restrict__ kv_len, unsigned n_batch,
                                       unsigned n_tok, unsigned n_head, unsigned kv_stride,
                                       unsigned window, float scale, unsigned kv_mask,
                                       unsigned slice, unsigned nblk, bf16* lds,
                                       const unsigned char* __restrict__ uni = nullptr,
                                       unsigned cap = 0, unsigned ns = 1, bool ofold = false,
                                       const float* __restrict__ kv_scale = nullptr,
                                       unsigned krot_fp8 = 0) {
    constexpr int RW = 16;                 /* q rows per wave                    */
    constexpr int BQ = 4 * RW;             /* 64 — the 4-wave tile               */
    constexpr int BKV = FA_MLA_PF2_BKV;    /* kv rows per staged slab            */
    constexpr unsigned WG = 256;            /* V2 is always a four-wave kernel    */
    constexpr int D = DK + DR;             /* 576: latent | rope, one padded row */
    constexpr int KSTR = D + FA_MLA_PF2_PAD;
    constexpr int NKT = D / 32;            /* QK k-tiles (32-deep shim)          */
    constexpr int NT = DK / 16;            /* output n-tiles                     */
    static_assert(D % 32 == 0 && DK % 16 == 0, "MLA dims must tile");

    /* Halves the staged slab occupies: BKV rows of KSTR, plus the 3*FA_MLA_PF2_SWZ of
     * block-swizzle slack PLOW_MLA_PF_SV adds (0 when the arm is off). */
    constexpr int KSLAB = BKV * KSTR + 3 * FA_MLA_PF2_SWZ;
    bf16* Ksm = lds;                       /* [BKV][KSTR] (+ block swizzle)      */
    bf16* Pw0 = lds + KSLAB;               /* [4][RW][BKV] per-wave P strips     */
    /* GATHER only: the staged slab's per-row u64 membership masks (bit q = local query q
     * of this 64-row tile selected the row). 256 B after the P strips; the byte offset
     * (KSLAB + 4*RW*BKV)*2 = 41,472 (41,568 under PLOW_MLA_PF_SV) is 8-divisible, so the
     * cast is aligned. */
    unsigned long long* Msm = (unsigned long long*)(Pw0 + 4 * RW * BKV);

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u, wave = tid >> 6;
    const unsigned fr = lane & 15u;        /* A-row / B-col selector             */
    const unsigned kg = lane >> 4;         /* k-group: this lane's 8*kg slice    */
    bf16* Pw = Pw0 + wave * RW * BKV;
    /* kv row base in HALVES. FA_MLA_PF2_SWZ=0 is the shipped identity map; =16 is the
     * PLOW_MLA_PF_SV block swizzle that de-conflicts the PV transpose read (see the
     * header note). Both are multiples of 8 halves, so every ds_read_b128 stays 16-byte
     * aligned. */
    auto krow = [&](unsigned kv) -> unsigned {
        return kv * (unsigned)KSTR + (unsigned)FA_MLA_PF2_SWZ * (kv >> 3);
    };

    /* GATHER = the HEAD-BATCHED sparse decomposition (B2). Phase B's per-64-query union
     * walk kept heads as the work axis and measured NET-NEGATIVE: 64 adjacent queries'
     * top-k sets union to 45-80% of the causal range, so the gather saved 20% while the
     * indexer cost more (glm52-dsa-sparse-prefill.md). Here a work item is one PACK of
     * QP=8 queries and the 64 M-rows are (query, head) pairs — all 8 per-rank heads ride
     * the MFMA M dimension, so the pack's union is walked ONCE for every head. Measured
     * on real membership masks (umask probe, 16k prompt): union-of-8 costs 398 KV
     * rows/query vs dense V2's 1512 — 3.8x fewer rows AND 3.8x less score/PV math,
     * growing with ctx (top-k is capped). Requires n_head == 8 (the emitter gates). */
    constexpr unsigned QP = 8; /* queries per pack; 64 rows = QP * 8 heads (GATHER) */
    const unsigned n_qt = GATHER ? (n_tok + QP - 1) / QP : (n_tok + BQ - 1) / BQ;
    /* ns = CAUSAL KV-SPLIT (dense only; i6 on the packet, 0/1 = off). A work item becomes
     * (q-tile, head, split) and each split owns a ceil-equal share of the TILE's OWN causal
     * range — so splits are near-equal work and the item count multiplies, which is what
     * fixes the 3.37-round tail quantization the 8k trace measured (34% of the machine idle
     * inside the packet). A split past the tile's causal tiles is DEAD: its walk is empty,
     * the epilogue writes (m=-inf, l=0, O=0), and d_mla_merge_fold weighs it 0 by the same
     * branch-free select the decode splits use. */
    const unsigned nsp = GATHER ? 1u : (ns ? ns : 1u);
    const unsigned n_work = GATHER ? n_batch * n_qt : n_batch * n_qt * n_head * nsp;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = GATHER ? 0u : w % nsp;
        const unsigned wq = GATHER ? w : w / nsp;
        const unsigned h = GATHER ? 0u : wq % n_head;
        const unsigned rest = GATHER ? wq : wq / n_head;
        const unsigned qt = mla_pf_fold(rest % n_qt, n_qt);
        const unsigned b = rest / n_qt;

        const unsigned len = (unsigned)kv_len[b];
        const unsigned q_pos0 = len - n_tok;
        const unsigned q_base = GATHER ? qt * QP : qt * BQ;
        const unsigned my_q0 = q_base + wave * RW; /* dense row base; GATHER remaps below */
        const unsigned my_r0 = wave * RW;          /* GATHER: this wave's (q,h) row base  */

        /* Workgroup-uniform KV bounds (the loop has barriers); exact work via the mask. */
        const unsigned q_tile_last = q_pos0 + q_base + BQ - 1;
        const unsigned kv_end = (q_tile_last + 1 < len) ? (q_tile_last + 1) : len;
        const unsigned q_tile_first = q_pos0 + q_base;
        const unsigned win_lo =
            (window && q_tile_first >= window) ? (q_tile_first - window + 1) : 0u;
        const unsigned kv_lo = (win_lo / BKV) * BKV;
        /* GATHER: this 64-query tile's union block (built by op 119). The walk covers
         * [0, ucount) union entries instead of the causal range; per-query exactness comes
         * from the staged mask words in the softmax below. */
        const int* upos = nullptr;
        const unsigned* mlo = nullptr;
        const unsigned* mhi = nullptr;
        unsigned ucount = 0;
        if constexpr (GATHER) {
            const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
            ucount = as_glob((const unsigned*)uni)[qt];
            upos = (const int*)(uni + hdr + (size_t)qt * cap * 12u);
            mlo = (const unsigned*)(uni + hdr + (size_t)qt * cap * 12u + (size_t)cap * 4u);
            mhi = (const unsigned*)(uni + hdr + (size_t)qt * cap * 12u + (size_t)cap * 8u);
        }
        unsigned walk_end = GATHER ? ucount : kv_end;
        unsigned walk_lo = GATHER ? 0u : kv_lo;
        if (!GATHER && nsp > 1) {
            /* ceil-equal tile share; mid-split bounds are BKV-aligned so no row is walked
             * twice, and the last split's ragged end is masked by kv_end as before. */
            const unsigned ntl = (kv_end - kv_lo + (unsigned)BKV - 1) / (unsigned)BKV;
            const unsigned cpt = (ntl + nsp - 1) / nsp;
            walk_lo = kv_lo + sp * cpt * (unsigned)BKV;
            const unsigned hi = kv_lo + (sp + 1u) * cpt * (unsigned)BKV;
            walk_end = (hi < kv_end) ? hi : kv_end;
            if (walk_lo >= kv_end) walk_end = walk_lo; /* dead split */
        }

        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const auto* rbase = as_glob(Krope) + (size_t)b * kv_stride * DR;
        const auto* cb8 = (const unsigned char*)Ckv + (size_t)b * kv_stride * DK;
        const auto* rb8 = (const unsigned char*)Krope + (size_t)b * kv_stride * DR;
        const float* csc = FP8 ? kv_scale + (size_t)b * kv_stride : nullptr;
        const float* rsc = FP8 ? kv_scale + (size_t)(n_batch + b) * kv_stride : nullptr;

        /* Q -> REGISTERS, once: lane's A-fragment row is my_q0+fr, k-slice kt*32 + kg*8. */
        bf16x8 qa[NKT];
        {
            /* GATHER: row my_r0+fr = (query q_base + row/8, head row%8). Dense: row = query. */
            const unsigned row = GATHER ? my_r0 + fr : 0u;
            const unsigned qi = GATHER ? q_base + (row >> 3) : my_q0 + fr;
            const unsigned hh = GATHER ? (row & 7u) : h;
            const bool live = qi < n_tok;
            const size_t qrow = ((size_t)b * n_tok + (live ? qi : 0)) * n_head + hh;
#pragma unroll
            for (int kt = 0; kt < NKT; kt++) {
                const unsigned c = (unsigned)kt * 32 + kg * 8;
                bf16v8 v = bf16v8_zero();
                if (live)
                    v = (c < (unsigned)DK) ? ld_glob8(&Qabs[qrow * DK + c])
                                           : ld_glob8(&Qrope[qrow * DR + (c - DK)]);
                __builtin_memcpy(&qa[kt], &v, 16);
            }
        }

        f32x4 oacc[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) oacc[t] = (f32x4)(0.0f);
        float m_st[4], l_st[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            m_st[i] = FA_NEG_INF;
            l_st[i] = 0.0f;
        }

#if PLOW_MLA_PF2_DBUF
        /* This thread's 9 slab fragments (8 latent + 1 rope; the loops below cover the
         * slab EXACTLY at 256 threads * 8 halves): loaded for tile n+1 while tile n
         * computes, committed to LDS at the next fence. */
        bf16v8 rl[9];
        auto ld_slab = [&](unsigned base) {
#pragma unroll
            for (int it = 0; it < 8; it++) {
                const unsigned e = tid * 8 + (unsigned)it * (WG * 8);
                const unsigned r = e / DK, c = e % DK;
                bf16v8 v = bf16v8_zero();
#if PLOW_MLA_PF2_ABL != 1
                if constexpr (GATHER) {
                    const unsigned u = base + r;
                    if (u < ucount) {
                        const unsigned kv = (unsigned)as_glob(upos)[u];
                        if constexpr (FP8)
                            v = fp8v8_to_bf16v8(
                                ld_glob_fp8v8(cb8 + (size_t)(kv & kv_mask) * DK + c));
                        else
                            v = ld_glob8(cbase + (size_t)(kv & kv_mask) * DK + c);
                    }
                } else {
                    const unsigned kv = base + r;
                    if (kv < kv_end) {
                        if constexpr (FP8)
                            v = fp8v8_to_bf16v8(
                                ld_glob_fp8v8(cb8 + (size_t)(kv & kv_mask) * DK + c));
                        else
                            v = ld_glob8(cbase + (size_t)(kv & kv_mask) * DK + c);
                    }
                }
#endif
                rl[it] = v;
            }
            {
                const unsigned e = tid * 8;
                const unsigned r = e / DR, c = e % DR;
                bf16v8 v = bf16v8_zero();
#if PLOW_MLA_PF2_ABL != 1
                if constexpr (GATHER) {
                    const unsigned u = base + r;
                    if (u < ucount) {
                        const unsigned kv = (unsigned)as_glob(upos)[u];
                        if constexpr (FP8) {
                            if (krot_fp8)
                                v = fp8v8_to_bf16v8(
                                    ld_glob_fp8v8(rb8 + (size_t)(kv & kv_mask) * DR + c));
                            else
                                v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                        } else
                            v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                    }
                } else {
                    const unsigned kv = base + r;
                    if (kv < kv_end) {
                        if constexpr (FP8) {
                            if (krot_fp8)
                                v = fp8v8_to_bf16v8(
                                    ld_glob_fp8v8(rb8 + (size_t)(kv & kv_mask) * DR + c));
                            else
                                v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                        } else
                            v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                    }
                }
#endif
                rl[8] = v;
            }
        };
        ld_slab(walk_lo); /* prologue: tile 0's loads in flight before the first fence */
#endif

        for (unsigned kv0 = walk_lo; kv0 < walk_end; kv0 += BKV) {
            if (!GATHER && window && kv0 + BKV <= win_lo) continue;

            /* ---- stage [BKV][latent|rope]; the pair of fences is the tile's WHOLE
             * barrier budget. GATHER resolves each staged row through the union list and
             * also stages the row's mask word. ---- */
            __syncthreads(); /* every wave's PV reads of the previous slab are done */
            if constexpr (GATHER) {
                for (unsigned r = tid; r < (unsigned)BKV; r += WG) {
                    const unsigned u = kv0 + r;
                    Msm[r] = (u < ucount)
                                 ? (((unsigned long long)as_glob(mhi)[u] << 32) |
                                    (unsigned long long)as_glob(mlo)[u])
                                 : 0ull;
                }
            }
#if PLOW_MLA_PF2_DBUF
            /* Commit the fragments loaded during the PREVIOUS tile's compute (or the
             * prologue). The `continue` above cannot desynchronize this: it fires only for
             * a tile entirely below win_lo, and walk_lo already floors to win_lo's tile,
             * so no reachable tile is ever skipped (GLM MLA runs window=0 besides). */
#pragma unroll
            for (int it = 0; it < 8; it++) {
                const unsigned e = tid * 8 + (unsigned)it * (WG * 8);
                __builtin_memcpy(&Ksm[krow(e / DK) + e % DK], &rl[it], 16);
            }
            {
                const unsigned e = tid * 8;
                __builtin_memcpy(&Ksm[krow(e / DR) + DK + e % DR], &rl[8], 16);
            }
            __syncthreads(); /* slab visible to every wave */
            if (kv0 + BKV < walk_end)
                ld_slab(kv0 + BKV); /* next tile's globals issue NOW, land behind compute */
#else
#if PLOW_MLA_PF2_ABL != 1
            for (unsigned e = tid * 8; e < (unsigned)(BKV * DK); e += WG * 8) {
                const unsigned r = e / DK, c = e % DK;
                bf16v8 v = bf16v8_zero();
                if constexpr (GATHER) {
                    const unsigned u = kv0 + r;
                    if (u < ucount) {
                        const unsigned kv = (unsigned)as_glob(upos)[u];
                        if constexpr (FP8)
                            v = fp8v8_to_bf16v8(
                                ld_glob_fp8v8(cb8 + (size_t)(kv & kv_mask) * DK + c));
                        else
                            v = ld_glob8(cbase + (size_t)(kv & kv_mask) * DK + c);
                    }
                } else {
                    const unsigned kv = kv0 + r;
                    if (kv < kv_end) {
                        if constexpr (FP8)
                            v = fp8v8_to_bf16v8(
                                ld_glob_fp8v8(cb8 + (size_t)(kv & kv_mask) * DK + c));
                        else
                            v = ld_glob8(cbase + (size_t)(kv & kv_mask) * DK + c);
                    }
                }
                __builtin_memcpy(&Ksm[krow(r) + c], &v, 16);
            }
            for (unsigned e = tid * 8; e < (unsigned)(BKV * DR); e += WG * 8) {
                const unsigned r = e / DR, c = e % DR;
                bf16v8 v = bf16v8_zero();
                if constexpr (GATHER) {
                    const unsigned u = kv0 + r;
                    if (u < ucount) {
                        const unsigned kv = (unsigned)as_glob(upos)[u];
                        if constexpr (FP8) {
                            if (krot_fp8)
                                v = fp8v8_to_bf16v8(
                                    ld_glob_fp8v8(rb8 + (size_t)(kv & kv_mask) * DR + c));
                            else
                                v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                        } else
                            v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                    }
                } else {
                    const unsigned kv = kv0 + r;
                    if (kv < kv_end) {
                        if constexpr (FP8) {
                            if (krot_fp8)
                                v = fp8v8_to_bf16v8(
                                    ld_glob_fp8v8(rb8 + (size_t)(kv & kv_mask) * DR + c));
                            else
                                v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                        } else
                            v = ld_glob8(rbase + (size_t)(kv & kv_mask) * DR + c);
                    }
                }
                __builtin_memcpy(&Ksm[krow(r) + DK + c], &v, 16);
            }
#endif
            __syncthreads(); /* slab visible to every wave */
#endif

            /* ---- S = Q·K^T over the full 576, registers × LDS, zero redundancy ---- */
            f32x4 sacc[2] = {(f32x4)(0.0f), (f32x4)(0.0f)};
            f32x4 srope[2] = {(f32x4)(0.0f), (f32x4)(0.0f)};
#if PLOW_MLA_PF2_ABL != 2
#if PLOW_MLA_PF_SV
            /* SV(2): flattened + double-buffered. The shipped form below reloads `kf` into
             * the SAME register the next MFMA pair consumes, so the compiler emits
             * `ds_read_b128; s_waitcnt lgkmcnt(0); mfma; mfma` 36 times — ONE LDS read in
             * flight, its full latency exposed every fragment. Issuing s+1's read before
             * s's MFMAs lets the wait drop to lgkmcnt(1). Same fragments, same order. */
            if constexpr (!FP8) {
                bf16x8 kf[2];
                auto kload = [&](int s, int slot) {
                    __builtin_memcpy(&kf[slot],
                                     &Ksm[krow((unsigned)(s / NKT) * 16 + fr) +
                                          (unsigned)(s % NKT) * 32 + kg * 8],
                                     16);
                };
                kload(0, 0);
#pragma unroll
                for (int st = 0; st < 2 * NKT; st++) {
                    if (st + 1 < 2 * NKT) kload(st + 1, (st + 1) & 1);
                    sacc[st / NKT] =
                        plow_mfma_bf16_16x16(qa[st % NKT], kf[st & 1], sacc[st / NKT]);
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                    for (int kt = 0; kt < DK / 32; kt++) {
                        bf16x8 kf;
                        __builtin_memcpy(
                            &kf, &Ksm[krow(nt * 16 + fr) + (unsigned)kt * 32 + kg * 8], 16);
                        sacc[nt] = plow_mfma_bf16_16x16(qa[kt], kf, sacc[nt]);
                    }
#pragma unroll
                    for (int kt = DK / 32; kt < NKT; kt++) {
                        bf16x8 kf;
                        __builtin_memcpy(
                            &kf, &Ksm[krow(nt * 16 + fr) + (unsigned)kt * 32 + kg * 8], 16);
                        srope[nt] = plow_mfma_bf16_16x16(qa[kt], kf, srope[nt]);
                    }
                }
            }
#else
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                if constexpr (FP8) {
#pragma unroll
                    for (int kt = 0; kt < DK / 32; kt++) {
                        bf16x8 kf;
                        __builtin_memcpy(
                            &kf, &Ksm[krow(nt * 16 + fr) + (unsigned)kt * 32 + kg * 8], 16);
                        sacc[nt] = plow_mfma_bf16_16x16(qa[kt], kf, sacc[nt]);
                    }
#pragma unroll
                    for (int kt = DK / 32; kt < NKT; kt++) {
                        bf16x8 kf;
                        __builtin_memcpy(
                            &kf, &Ksm[krow(nt * 16 + fr) + (unsigned)kt * 32 + kg * 8], 16);
                        srope[nt] = plow_mfma_bf16_16x16(qa[kt], kf, srope[nt]);
                    }
                } else {
#pragma unroll
                    for (int kt = 0; kt < NKT; kt++) {
                        bf16x8 kf;
                        __builtin_memcpy(
                            &kf, &Ksm[krow(nt * 16 + fr) + (unsigned)kt * 32 + kg * 8], 16);
                        sacc[nt] = plow_mfma_bf16_16x16(qa[kt], kf, sacc[nt]);
                    }
                }
            }
#endif
#endif

            /* ---- mask + per-wave online softmax. Row m = kg*4+i lives in the 16 lanes of
             * this k-group (one kv column each per n-tile), so the row reductions are
             * quarter-wave shuffles. No wave shares a row with any other wave. ---- */
            float pe[2][4];
#if PLOW_MLA_PF2_ABL == 3
            /* no softmax math: unit P, no shuffles/exp, no corr rescale */
#pragma unroll
            for (int i = 0; i < 4; i++) {
                pe[0][i] = 1.0f;
                pe[1][i] = 1.0f;
                m_st[i] = 0.0f;
                l_st[i] += 2.0f * 16.0f;
            }
#else
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned row_i = my_r0 + kg * 4 + (unsigned)i; /* GATHER (q,h) row */
                const unsigned qi = GATHER ? q_base + (row_i >> 3) : my_q0 + kg * 4 + i;
                const unsigned qg = q_pos0 + qi;
                float sv[2];
                float rmax = FA_NEG_INF;
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    const unsigned kvg = kv0 + nt * 16 + fr;
                    bool valid;
                    if constexpr (GATHER) {
                        /* per-query membership: the selection is causal by construction, so
                         * the mask bit IS the whole validity test (plus the tile tails). */
                        const unsigned ql = qi - q_base; /* pack-local query, bit 0..QP-1 */
                        valid = (qi < n_tok) && (kvg < ucount) &&
                                (((Msm[nt * 16 + fr] >> ql) & 1ull) != 0ull);
                    } else {
                        valid = (qi < n_tok) && (kvg < kv_end) && (kvg <= qg) &&
                                (!window || (qg - kvg) < window);
                    }
                    float score = sacc[nt][i];
                    if constexpr (FP8) {
                        const unsigned row = kvg & kv_mask;
                        const float cs = csc[row];
                        const float rs = krot_fp8 ? rsc[row] : 1.0f;
                        score = score * cs + srope[nt][i] * rs;
                    }
                    sv[nt] = valid ? score * FA_SCALE(scale) : FA_NEG_INF;
                    rmax = fmaxf(rmax, sv[nt]);
                }
#pragma unroll
                for (int d = 1; d < 16; d <<= 1) rmax = fmaxf(rmax, __shfl_xor(rmax, d, PLOW_WAVE));
                const float mnew = fmaxf(m_st[i], rmax);
                const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                float lsum = 0.0f;
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    const float p = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(sv[nt] - mnew);
                    pe[nt][i] = p;
                    lsum += p;
                }
#pragma unroll
                for (int d = 1; d < 16; d <<= 1) lsum += __shfl_xor(lsum, d, PLOW_WAVE);
                l_st[i] = l_st[i] * corr + lsum;
                m_st[i] = mnew;
#pragma unroll
                for (int t = 0; t < NT; t++) oacc[t][i] *= corr;
            }
#endif /* PLOW_MLA_PF2_ABL == 3 */

#if PLOW_MLA_PF2_ABL != 4
            /* ---- P through this wave's PRIVATE strip: same-wave ds_write -> ds_read is
             * hardware-ordered (lgkmcnt), no barrier. ---- */
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float p = pe[nt][i];
                    if constexpr (FP8)
                        p *= csc[(kv0 + (unsigned)nt * 16 + fr) & kv_mask];
                    Pw[(kg * 4 + i) * BKV + (unsigned)nt * 16 + fr] = f2bf(p);
                }

            /* ---- O += P·V, V = the latent columns of the SAME slab ---- */
            bf16x8 pf;
            __builtin_memcpy(&pf, &Pw[fr * BKV + kg * 8], 16);
#if PLOW_MLA_PF_SV
            /* SV(3): the next output tile's 8 transpose reads issue before this tile's
             * MFMAs, so the lgkm pipeline holds 16 outstanding u16 reads instead of 8.
             * Under SV(1) each of those reads is bank-conflict-free. */
            {
                bf16x8 vf[2];
                auto vload = [&](int t, int slot) {
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        bf16 vv = Ksm[krow(kg * 8 + (unsigned)j) + (unsigned)t * 16 + fr];
                        bf16_t vb;
                        __builtin_memcpy(&vb, &vv, 2);
                        vf[slot][j] = vb;
                    }
                };
                vload(0, 0);
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    if (t + 1 < NT) vload(t + 1, (t + 1) & 1);
                    oacc[t] = plow_mfma_bf16_16x16(pf, vf[t & 1], oacc[t]);
                }
            }
#else
#pragma unroll
            for (int t = 0; t < NT; t++) {
                bf16x8 vf;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    bf16 vv = Ksm[krow(kg * 8 + (unsigned)j) + (unsigned)t * 16 + fr];
                    bf16_t vb;
                    __builtin_memcpy(&vb, &vv, 2);
                    vf[j] = vb;
                }
                oacc[t] = plow_mfma_bf16_16x16(pf, vf, oacc[t]);
            }
#endif
#endif /* PLOW_MLA_PF2_ABL != 4 */
        }

        /* ---- UNNORMALIZED partials + (m,l), the d_flash_merge/d_o_uv_fold layout ----
         *
         * OFOLD arm (i[6] on the dense packet): the W_ofold fusion deletes MlaMergeFold, so
         * nothing downstream can normalize — this epilogue does it (each lane already holds
         * its rows' `l` after the quarter-wave reduce, so it is a per-value multiply) and
         * writes NORMALIZED bf16 rows into the SAME Opart allocation, [t][head][DK]
         * contiguous — exactly the [T, nh_l*DK] A operand the fused o-GEMM reads. The
         * mlpart write is skipped: the merge that consumed it no longer exists. `inv` uses
         * FA_RECIP to match the merge's v_rcp path (numerics gate class, not bit-exact —
         * the fused weight reassociated anyway). Dense arm only; the emitter refuses
         * ofold+GATHER and ofold+fp8kv. */
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned row_i = my_r0 + kg * 4 + (unsigned)i;
            const unsigned qi = GATHER ? q_base + (row_i >> 3) : my_q0 + kg * 4 + i;
            const unsigned hh = GATHER ? (row_i & 7u) : h;
            if (qi >= n_tok) continue;
            const size_t oh = (((size_t)b * n_tok + qi) * n_head + hh) * nsp + sp;
            if (ofold) {
                /* W_ofold epilogue: normalized bf16 rows for the fused o-GEMM. ns==1 is the
                 * emit contract (the fold consumes the un-split l), so nsp==1/sp==0 and oh
                 * is the un-split index. */
                bf16* ob = (bf16*)Opart + oh * DK;
                const float inv = (l_st[i] > 0.0f) ? FA_RECIP(l_st[i]) : 0.0f;
#pragma unroll
                for (int t = 0; t < NT; t++)
                    st_act1(&ob[(unsigned)t * 16 + fr], f2bf(oacc[t][i] * inv));
                continue;
            }
            float* op = Opart + oh * DK;
#pragma unroll
            for (int t = 0; t < NT; t++) st_act<float>(&op[(unsigned)t * 16 + fr], oacc[t][i]);
            if (fr == 0) {
                float* ml = mlpart + oh * 2;
                st_act<float>(&ml[0], m_st[i]);
                st_act<float>(&ml[1], l_st[i]);
            }
        }
    }
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
 * the design notes. The production path is the scalar GF/nsplit kernel.
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
                s = plow_mfma_bf16_32x32(qf, kf, s);
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
                    oacc[t] = plow_mfma_bf16_32x32(pf, vf, oacc[t]);
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
                    st_act<float>(&ml[0], m_st[i]);
                    st_act<float>(&ml[1], l_st[i]);
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
                                   unsigned kv_stride, float scale, unsigned slice, unsigned nblk,
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
        /* `slice`, NOT `blockIdx.x`. Under the persistent interpreter the workgroup that runs a
         * stream entry is NOT the entry's logical slice: the GLOBAL-QUEUE scheduler (the DEFAULT
         * for the decode phase, `exec/amd.rs` `sched_decode`) hands an entry to whichever workgroup
         * reaches the shared cursor first, and passes the logical index as `e.slice`. Striding on
         * `blockIdx.x` therefore covered an ARBITRARY subset of the slabs — duplicating some and
         * LEAVING OTHERS UNWRITTEN — while `dsa_gather_bench`, which launches this as a standalone
         * kernel with `gridDim == nblk`, saw `blockIdx.x == slice` and reported EXACT. That is why
         * the op-level oracle was clean and the end-to-end path was degenerate. Every other kernel
         * in this tree already takes (slice, nblk); see the convention note at op_kda.h:30. */
        for (unsigned st = slice; st < nslab; st += nblk) {
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
                acc = plow_mfma_bf16_32x32(qf[ks], kfrag, acc);
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
                if (pos < len) st_act<float>(&Sc[(size_t)b * kv_stride + pos], part * scale);
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
/* `kv_len` IS AN OPERAND, NOT AN OPTIMISATION — the whole DSA path was wrong without it.
 *
 * `len_max` is the packet's MAX ctx (`i[0]`, baked at emit time), but `d_index_score_mfma` writes
 * `Score[pos]` only `if (pos < kv_len[b])` — the live cache occupancy. Scanning `len_max` therefore
 * ranked `len_max - kv_len` floats THE SCORE KERNEL NEVER WROTE. DSA only arms above a 64k crossover,
 * so on a real decode step that is tens of thousands of stale/uninitialised words against a few
 * thousand real ones, and the indexer score is `sum_h w[h]*ReLU(q.k)*scale` with `w` UNSIGNED-free
 * (weights_proj emits negatives), so genuine scores go negative and a 0.0 hole OUTRANKS them. The
 * selector then handed the gather positions >= kv_len, the gather applies NO mask (the set is
 * "assumed causal because the selector produced it"), and attention read latent rows that were never
 * written. That is the recorded "degenerate output even on the decode-only bundle", and it is
 * DETERMINISTIC, not a race.
 *
 * It is also why `top_k` has to be clamped here: with `kv_len < top_k` (any short prompt) there are
 * not top_k rows in existence, and the old code padded the difference out of the uninitialised tail.
 * `d_flash_mla_decode<...,GATHER=true>` derives the SAME `min(top_k, kv_len)` from the same operand,
 * so the two agree by construction rather than by a matching pair of emit-time constants.
 *
 * `runtime/bench/dsa_gather_bench.c` could not see any of this: it uploads `kv_len = ctx` and passes
 * `len = ctx`, so score-written and select-scanned lengths are equal by construction there. */
template <bool ATOMIC_SYNC = true, bool PAR_SCAN = true, bool FAST = true>
__device__ void d_index_select_coop(int* __restrict__ idx, const float* __restrict__ Score,
                                    unsigned len_max, unsigned top_k_max, unsigned* __restrict__ gHist,
                                    unsigned* __restrict__ gCtl, unsigned slice, unsigned nwg,
                                    unsigned* lh /* [SEL_NB] LDS */, unsigned* red /* [3] LDS */,
                                    const int* __restrict__ kv_len = nullptr) {
    const auto* const Sc = as_glob(Score);
    int* const ib = as_glob(idx);
    unsigned* const Hg = as_glob(gHist);
    unsigned* const Cg = as_glob(gCtl);
    /* Only the rows the score kernel actually wrote, and only as many as exist. */
    const unsigned nkv = kv_len ? (unsigned)as_glob(kv_len)[0] : len_max;
    const unsigned len = nkv < len_max ? nkv : len_max;
    const unsigned top_k = top_k_max < len ? top_k_max : len;
    /* `bid` IS THE LOGICAL SLICE, NOT `blockIdx.x`, AND THE DIFFERENCE IS THE WHOLE END-TO-END BUG.
     *
     * This kernel partitions the score array by `bid` in three places (the histogram pass, the
     * histogram clear, and the final emit loop), so `bid` must enumerate `0..nwg-1` EXACTLY ONCE
     * across the participating workgroups. Under the persistent interpreter that is `e.slice`, the
     * index the compiler assigned the stream entry — NOT the workgroup id. The decode phase runs
     * the GLOBAL-QUEUE scheduler by default (`sched_decode` in `crates/plowrt/src/exec/amd.rs`),
     * where entries are claimed from one shared cursor, so the 32 select entries land on 32
     * ARBITRARY workgroups. With `bid = blockIdx.x` the covered set was an arbitrary subset of
     * [0, n_cu): at a short context (`len = 533`, `nwg = 32`) only `bid` 0 and 1 have any rows at
     * all, so unless those two workgroup ids happened to claim a select entry the emit loop wrote
     * NOTHING and `idx[]` stayed at its zero-filled bind value — every gathered row became latent
     * row 0. That is the recorded "first token right (dense prefill), every token after it wrong".
     *
     * `dsa_gather_bench` cannot see it: a standalone launch has `blockIdx.x == slice` by
     * construction, which is exactly why the op-level oracle reported EXACT at every ctx while the
     * model was degenerate. Every other kernel here already takes (slice, nblk) — op_kda.h:30. */
    const unsigned bid = slice, tid = threadIdx.x;
    /* PLAIN store, deliberately, and MEASURED — do not "fix" this to atomicExch.
     * The suspicion was that on 8-XCD gfx950 a plain store lands dirty in one XCD's L2 while the
     * `atomicAdd(&Cg[2],1u)` below is performed at a coherent point, so the reset would be missed on
     * every layer after the first. It is not. Disassembly of `index_select_coop_a` (ROCm 7.2.4,
     * gfx950): the reset is `global_store_dword ... offset:8` and the emit-slot bump is
     * `global_atomic_add ... offset:8 sc0` — `sc0` is the return-previous-value bit, NOT a scope bit,
     * and NEITHER instruction carries `sc1`. Both therefore operate in the same (hardware-coherent)
     * L2 domain. Confirmed end-to-end by dsa_gather_bench's tie-stress, which re-launches the
     * selector over a DIFFERENT score array sharing one gCtl and still reports the set EXACT at
     * ctx 8k/32k/128k/256k. The histogram reset below uses atomicExch to CLEAR CONCURRENTLY from
     * every WG, which is a different requirement (no single owner), not a coherence one. */
    if (bid == 0 && tid == 0) st_act<unsigned>(&Cg[2], 0u); /* reset emit slot */
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
            if (slot < top_k) st_act<int>(&ib[slot], (int)t);
        }
    }
}

/* =============================================================================================
 * DSA sparse PREFILL — ops 117/118/119 + the GATHER arm of d_flash_mla_prefill_v2. [GLM52-DSA-PF]
 *
 * The decode indexer scores ONE query; a prefill selection is one top-k row PER QUERY TOKEN.
 * These three kernels produce exactly that and package it for the V2 flash: per-token scores
 * (117, the decode MFMA subtile under a per-query work axis), per-row EXACT top-k (118, the
 * op-59 radix run whole inside one workgroup — no grid sync), and a per-64-query-tile UNION
 * table (119) so the gathered flash stages each tile's selected KV rows ONCE and masks per
 * query, instead of vLLM's per-token random gather with zero reuse.
 * ============================================================================================= */

/* op 117: T-row lightning-indexer score. score[t][s] = Σ_h w[t][h]·ReLU(q_idx[t][h]·k_idx[s])
 * for s <= q_pos0 + t (q_pos0 = kv_len - n_tok). One (query, 32-position) subtile per WAVE:
 * the A operand is the query's 32 head rows loaded straight from global (each query row is
 * touched len/32 times — L1/L2-hot), the B operand the 32 keys, also straight from global
 * (every key row is re-read by every later query — the whole key matrix is L2-resident at
 * these sizes). No LDS, no barriers: items are fully independent, so the wave-granular
 * grid-stride fills 304 CUs at any T. Math identical to d_index_score_mfma (same fragments,
 * same w-weighted ReLU epilogue, same scale). */
template <int DI, int HIc>
__device__ void d_index_score_pf(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                                 const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                                 const int* __restrict__ kv_len, unsigned n_tok,
                                 unsigned kv_stride, float scale, unsigned slice, unsigned nblk) {
    static_assert(DI % 16 == 0, "DI must be a whole number of MFMA k-steps");
    static_assert(HIc == 32, "MFMA subtile assumes index_n_heads == 32");
    constexpr int NK = DI / MFMA_K;
    auto* const Sc = as_glob(Score);
    const auto* const Qg = as_glob(Qidx);
    const auto* const Kg = as_glob(Kidx);
    const auto* const Wg = as_glob(W);
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    constexpr unsigned NW = PLOW_THREADS / 64u;
    const unsigned frow = mfma_frag_row(lane);
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const unsigned q_pos0 = len - n_tok;
    const unsigned n_s32 = (len + 31u) / 32u;
    const unsigned n_work = n_tok * n_s32;
    for (unsigned w = slice * NW + wave; w < n_work; w += nblk * NW) {
        const unsigned t = w / n_s32, s32 = w % n_s32;
        const unsigned pos0 = s32 * 32u;
        const unsigned row_end = q_pos0 + t + 1u; /* causal bound for this query row */
        if (pos0 >= row_end) continue;
        /* A: the query's 32 head rows (rows = heads via frow), straight from global. */
        f32x16 acc = (f32x16)(0.0f);
#pragma unroll
        for (int ks = 0; ks < NK; ks++) {
            const unsigned d0 = mfma_frag_k(lane, ks * MFMA_K);
            const bf16x8 qf = __builtin_bit_cast(
                bf16x8, ld_glob8(&Qg[((size_t)t * HIc + frow) * DI + d0]));
            const unsigned pos = pos0 + frow;
            bf16v8 kv = bf16v8_zero();
            if (pos < len) kv = ld_glob8(&Kg[(size_t)pos * DI + d0]);
            acc = plow_mfma_bf16_32x32(qf, __builtin_bit_cast(bf16x8, kv), acc);
        }
        /* epilogue: lane owns pos = lane%32 and 16 of the 32 heads (l / l+32 halves). */
        const unsigned mbase = 4u * (lane / 32u);
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const unsigned h = mbase + ((unsigned)i % 4u) + 8u * ((unsigned)i / 4u);
            const float d = acc[i];
            part += bf2f(Wg[(size_t)t * HIc + h]) * (d > 0.0f ? d : 0.0f);
        }
        part += __shfl_xor(part, 32, PLOW_WAVE);
        if (lane < 32u) {
            const unsigned pos = pos0 + lane;
            if (pos < row_end) st_act<float>(&Sc[(size_t)t * kv_stride + pos], part * scale);
        }
    }
}

/* op 117, ARM B — the same score, ROW-RESIDENT.                        [GLM52-DSA-PF-IDX]
 *
 * The shipped arm above re-fetches BOTH MFMA operands from global for every (query, 32-key)
 * work item: 8 KiB of query rows + 8 KiB of key rows to produce 32 scores, i.e. 16 KiB of load
 * per 8 MFMA. At T=8192 over a 16384 KV that is 3.15e6 items x 16 KiB = 51 GB of cache traffic
 * per layer to do 825 GFLOP of MFMA — the kernel is VMEM-issue bound on operand re-fetch, not
 * on the MFMA pipe, and no amount of L2 residency fixes an instruction-issue limit.
 *
 * This arm changes ONLY the decomposition, never the arithmetic:
 *   - work item = (PACK of PLOW_WAVES query rows, SPAN of kv positions); wave w owns query
 *     p*NW + w, so the whole workgroup shares one key stream;
 *   - the query's A-fragments AND its 16 lane-local lightning weights are hoisted to VGPRs once
 *     per work item and reused across every key subtile in the span (the shipped arm reloads
 *     both per subtile);
 *   - keys stream contiguously through LDS one TILE_N slab at a time, so each key row is
 *     read from global ONCE per pack instead of once per (query, subtile).
 * Traffic falls ~8x on the key side and ~SPAN/32 on the query side. Same 8 k-steps in the same
 * order into the same accumulator, same w-weighted ReLU epilogue, same scale => BIT-IDENTICAL
 * output; `dsa_pf_indexer_bench` gates it byte-for-byte against the arm above.
 *
 * SPAN exists so the grid still fills: one item per pack would be n_tok/8 items (512 at T=4096,
 * under two per CU, and the causal length grows with the pack index so the tail is ragged).
 * Splitting the kv axis restores a fine grid-stride at negligible cost — the only thing SPAN
 * repeats is the query fragment load, 8 KiB per pack per span.
 *
 * IDXPF_SPAN MUST be a multiple of TILE_N: the slab loader zero-fills past `s_hi`, and
 * that is only safe on the LAST span of a pack (where no query in the pack has a causal bound
 * beyond it). Alignment guarantees every earlier span ends exactly on a slab boundary. */
#ifndef IDXPF_SPAN
#define IDXPF_SPAN 1024u /* kv positions per work item */
#endif

/* TILE_N is the LDS slab width and it is the OCCUPANCY knob: 128 keys is 34,816 B, one workgroup
 * per CU (2 waves/SIMD); 64 keys is 17,408 B, three workgroups (6 waves/SIMD). A wider slab means
 * fewer global key re-reads per pack; more occupancy means more latency hiding for the slab load
 * itself. Which side wins is measured, not assumed — `dsa_pf_indexer_bench` runs both. */
template <int DI, int HIc, unsigned TILE_N>
__device__ void d_index_score_pf_row(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                                     const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                                     const int* __restrict__ kv_len, unsigned n_tok,
                                     unsigned kv_stride, float scale, unsigned slice,
                                     unsigned nblk, bf16* ktile /* TILE_N * KSTRIDE */) {
    static_assert(DI % 16 == 0, "DI must be a whole number of MFMA k-steps");
    static_assert(HIc == 32, "MFMA subtile assumes index_n_heads == 32");
    static_assert(IDXPF_SPAN % TILE_N == 0, "span must end on a slab boundary");
    static_assert(TILE_N % 32u == 0, "slab is walked in 32-key MFMA subtiles");
    constexpr int NK = DI / MFMA_K;
    constexpr int KSTRIDE = DI + FA_PAD;
    constexpr unsigned NW = PLOW_THREADS / 64u;
    auto* const Sc = as_glob(Score);
    const auto* const Qg = as_glob(Qidx);
    const auto* const Kg = as_glob(Kidx);
    const auto* const Wg = as_glob(W);
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u, wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);
    const unsigned mbase = 4u * (lane / 32u);
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const unsigned q_pos0 = len - n_tok;
    const unsigned n_pack = (n_tok + NW - 1u) / NW;
    const unsigned n_span = (len + IDXPF_SPAN - 1u) / IDXPF_SPAN;
    const unsigned n_work = n_pack * n_span;

    /* `w`, and everything derived from it, is workgroup-UNIFORM: every __syncthreads below is
     * reached by all 512 threads the same number of times. Only `live`/`row_end` vary by wave,
     * and those gate compute only, never a barrier. */
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned p = w / n_span, sp = w % n_span;
        unsigned pack_last = p * NW + (NW - 1u);
        if (pack_last >= n_tok) pack_last = n_tok - 1u;
        const unsigned pack_end = q_pos0 + pack_last + 1u; /* causal bound of the pack's last row */
        const unsigned s_lo = sp * IDXPF_SPAN;
        if (s_lo >= pack_end) continue; /* whole pack is causally below this span */
        const unsigned s_hi = (s_lo + IDXPF_SPAN < pack_end) ? (s_lo + IDXPF_SPAN) : pack_end;

        const unsigned t = p * NW + wave;
        const bool live = t < n_tok;
        const unsigned row_end = live ? (q_pos0 + t + 1u) : 0u;
        /* A-fragments + this row's lane-local head weights: loaded ONCE for the whole span. */
        bf16x8 qf[NK];
        float wv[16];
        if (live) {
#pragma unroll
            for (int ks = 0; ks < NK; ks++) {
                const unsigned d0 = mfma_frag_k(lane, ks * MFMA_K);
                qf[ks] = __builtin_bit_cast(bf16x8,
                                            ld_glob8(&Qg[((size_t)t * HIc + frow) * DI + d0]));
            }
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const unsigned h = mbase + ((unsigned)i % 4u) + 8u * ((unsigned)i / 4u);
                wv[i] = bf2f(Wg[(size_t)t * HIc + h]);
            }
        }
        for (unsigned b = s_lo; b < s_hi; b += TILE_N) {
            __syncthreads(); /* previous slab's MFMA readers done before we overwrite ktile */
            for (unsigned c = tid; c < TILE_N * (DI / 8u); c += PLOW_THREADS) {
                const unsigned row = c / (DI / 8u), c8 = (c % (DI / 8u)) * 8u;
                const unsigned pos = b + row;
                *(bf16v8*)&ktile[row * KSTRIDE + c8] =
                    (pos < s_hi) ? ld_glob8(&Kg[(size_t)pos * DI + c8]) : bf16v8_zero();
            }
            __syncthreads(); /* slab visible to every wave before the MFMA reads */
            if (!live) continue;
#pragma unroll 1
            for (unsigned st = 0; st < TILE_N; st += 32u) {
                const unsigned pos0 = b + st;
                if (pos0 >= row_end) break; /* causal: this row is done inside this slab */
                f32x16 acc = (f32x16)(0.0f);
#pragma unroll
                for (int ks = 0; ks < NK; ks++) {
                    const unsigned d0 = mfma_frag_k(lane, ks * MFMA_K);
                    const bf16x8 kf =
                        __builtin_bit_cast(bf16x8, ld_lds8(&ktile[(st + frow) * KSTRIDE + d0]));
                    acc = plow_mfma_bf16_32x32(qf[ks], kf, acc);
                }
                float part = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const float d = acc[i];
                    part += wv[i] * (d > 0.0f ? d : 0.0f);
                }
                part += __shfl_xor(part, 32, PLOW_WAVE);
                if (lane < 32u) {
                    const unsigned pos = pos0 + lane;
                    if (pos < row_end) st_act<float>(&Sc[(size_t)t * kv_stride + pos], part * scale);
                }
            }
        }
    }
}

/* op 118: per-row EXACT top-k select. One WORKGROUP per query row (grid-strided over rows):
 * the op-59 radix — same monotone byte-aligned key (score desc, lowest index tie-break) —
 * with the histogram entirely in LDS and __syncthreads instead of the grid barrier. 7
 * MSB-first byte passes refine `prefix` to the exact key of the k-th element (keys are
 * unique, position bits break ties), then one emit pass appends every key >= prefix in
 * arbitrary order (the union build is order-blind). Rows with len <= top_k emit the
 * identity and pad with -1. `lh` is a [256] u32 LDS strip + [4] scratch.
 *
 * FAST_EXIT ports d_index_select_coop's measured fewer-passes early-out, which this kernel was
 * written without: `dsa_pack_key_a` puts the whole 32-bit score in the TOP FOUR bytes, so after
 * pass 3 the score threshold is fully resolved and the remaining three passes exist only to
 * split a genuine exact-score tie by index. When the boundary bin's population equals the
 * still-needed count, the whole tied group is selected and those three passes would emit the
 * SAME set — so the loop breaks at 4. Every pass costs a full row re-read, so this removes 3 of
 * the 8 row scans. Exactness is unchanged (the early-out condition is precisely "the index bits
 * cannot change the set"), and the branch is workgroup-uniform. */
template <bool FAST_EXIT>
__device__ void d_index_select_pf(int* __restrict__ idx, const float* __restrict__ Score,
                                  const int* __restrict__ kv_len, unsigned n_tok, unsigned top_k,
                                  unsigned kv_stride, unsigned slice, unsigned nblk,
                                  unsigned* lh, unsigned* red) {
    const auto* const Sc = as_glob(Score);
    int* const ib = as_glob(idx);
    const unsigned tid = threadIdx.x;
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const unsigned q_pos0 = len - n_tok;
    for (unsigned t = slice; t < n_tok; t += nblk) {
        const unsigned row_len = q_pos0 + t + 1u;
        int* const row = ib + (size_t)t * top_k;
        if (row_len <= top_k) {
            for (unsigned s = tid; s < top_k; s += PLOW_THREADS)
                st_act<int>(&row[s], s < row_len ? (int)s : -1);
            __syncthreads();
            continue;
        }
        const float* const sr = Sc + (size_t)t * kv_stride;
        unsigned long long prefix = 0ull, himask = 0ull;
        unsigned k_rem = top_k;
        for (unsigned pass = 0; pass < SEL_NPASS; pass++) {
            const unsigned sh = (SEL_NPASS - 1u - pass) * SEL_DIGIT;
            for (unsigned i = tid; i < SEL_NB; i += PLOW_THREADS) lh[i] = 0u;
            __syncthreads();
            for (unsigned s = tid; s < row_len; s += PLOW_THREADS) {
                const unsigned long long key = dsa_pack_key_a(sr[s], s, row_len);
                if ((key & himask) == prefix)
                    atomicAdd(&lh[(unsigned)((key >> sh) & (SEL_NB - 1u))], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                unsigned acc = 0, dsel = 0, bnd = 0;
                for (int d = (int)SEL_NB - 1; d >= 0; d--) {
                    const unsigned hd = lh[d];
                    if (acc + hd >= k_rem) {
                        dsel = (unsigned)d;
                        bnd = hd;
                        break;
                    }
                    acc += hd;
                }
                red[0] = dsel;
                red[1] = acc;
                red[3] = bnd; /* population of the boundary bin (the tied group at this digit) */
            }
            __syncthreads();
            prefix |= (unsigned long long)red[0] << sh;
            himask |= 0xFFull << sh;
            k_rem -= red[1];
            const unsigned bnd = red[3];
            __syncthreads();
            /* after the 4 SCORE bytes the threshold score is exact; if the tied group is exactly
             * what is still needed, the 3 index passes cannot change the emitted set. */
            if (FAST_EXIT && pass == 3u && bnd == k_rem) break;
        }
        /* prefix == the exact k-th key; emit all >= it (exactly top_k, keys unique). */
        if (tid == 0) red[2] = 0u;
        __syncthreads();
        for (unsigned s = tid; s < row_len; s += PLOW_THREADS) {
            const unsigned long long key = dsa_pack_key_a(sr[s], s, row_len);
            if (key >= prefix) {
                const unsigned slot = atomicAdd(&red[2], 1u);
                if (slot < top_k) st_act<int>(&row[slot], (int)s);
            }
        }
        __syncthreads();
    }
}

/* op 119: per-64-query-tile UNION build. Scatters the tile's 64 selected-index rows into a
 * u64 membership word per kv position (bit q = local query q selected this position), then
 * compacts positions ASCENDING into the union table the gathered flash walks:
 *   [0, hdr)                      u32 count[n_qt], hdr = 256-aligned
 *   hdr + qt*cap*12 ..            i32 pos[cap] | u32 maskLo[cap] | u32 maskHi[cap]
 * Every umask access is an L2-coherent ATOMIC (exch to zero, or to scatter, or-0 to read):
 * plain loads after atomics could hit a stale L1 line on gfx9. cap >= any possible union
 * (min(64*top_k, kv_stride)), so the clamp below never truncates a real build. */
__device__ void d_index_union_pf(unsigned char* __restrict__ uni,
                                 unsigned long long* __restrict__ umask,
                                 const int* __restrict__ idx, const int* __restrict__ kv_len,
                                 unsigned n_tok, unsigned top_k, unsigned kv_stride, unsigned cap,
                                 unsigned tile_p, unsigned slice, unsigned nblk,
                                 unsigned* sc /* [PLOW_THREADS+1] */) {
    const unsigned tid = threadIdx.x;
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const unsigned q_pos0 = len - n_tok;
    /* tile_p = queries per union tile (i4; 0 = the legacy 64). The B2 head-batched walk
     * uses tile_p = 8: a pack's 8 queries share one union, all 8 heads share one walk —
     * measured 3.8x fewer rows/query than the 64-tile at a 16k prompt (umask probe). */
    const unsigned P = tile_p ? tile_p : 64u;
    const unsigned n_qt = (n_tok + P - 1u) / P;
    const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
    unsigned* const cnt = (unsigned*)uni;
    for (unsigned qt = slice; qt < n_qt; qt += nblk) {
        const unsigned q_hi = (qt * P + P - 1u < n_tok - 1u) ? qt * P + P - 1u : n_tok - 1u;
        const unsigned tile_end = q_pos0 + q_hi + 1u; /* strictest causal bound in the tile */
        /* SLICE-indexed scratch row, not qt-indexed: at tile_p=8 the tile count is 8x the
         * legacy sizing and a per-qt umask would cost ~600 MB; only `nblk` tiles are ever
         * in flight, and each workgroup zeroes its row per tile anyway. */
        unsigned long long* const mrow = as_glob(umask) + (size_t)slice * kv_stride;
        int* const upos = (int*)(uni + hdr + (size_t)qt * cap * 12u);
        unsigned* const ulo = (unsigned*)(uni + hdr + (size_t)qt * cap * 12u + (size_t)cap * 4u);
        unsigned* const uhi = (unsigned*)(uni + hdr + (size_t)qt * cap * 12u + (size_t)cap * 8u);
        for (unsigned s = tid; s < tile_end; s += PLOW_THREADS) atomicExch(&mrow[s], 0ull);
        __syncthreads();
        for (unsigned e = tid; e < P * top_k; e += PLOW_THREADS) {
            const unsigned ql = e / top_k;
            const unsigned qi = qt * P + ql;
            if (qi >= n_tok) continue;
            const int s = as_glob(idx)[(size_t)qi * top_k + (e % top_k)];
            if (s >= 0) atomicOr(&mrow[s], 1ull << ql);
        }
        __syncthreads();
        /* ordered compaction: chunked block scan over [0, tile_end). */
        unsigned base = 0;
        for (unsigned c0 = 0; c0 < tile_end; c0 += PLOW_THREADS) {
            const unsigned s = c0 + tid;
            const unsigned long long m =
                (s < tile_end) ? atomicOr(&mrow[s], 0ull) : 0ull;
            const unsigned flag = m != 0ull;
            sc[tid] = flag;
            __syncthreads();
            for (unsigned off = 1; off < PLOW_THREADS; off <<= 1) {
                const unsigned v = (tid >= off) ? sc[tid - off] : 0u;
                __syncthreads();
                sc[tid] += v;
                __syncthreads();
            }
            const unsigned rank = sc[tid] - flag;
            const unsigned total = sc[PLOW_THREADS - 1];
            if (flag && base + rank < cap) {
                st_act<int>(&as_glob(upos)[base + rank], (int)s);
                st_act<unsigned>(&as_glob(ulo)[base + rank], (unsigned)(m & 0xFFFFFFFFull));
                st_act<unsigned>(&as_glob(uhi)[base + rank], (unsigned)(m >> 32));
            }
            base += total;
            __syncthreads();
        }
        if (tid == 0) st_act<unsigned>(&cnt[qt], base < cap ? base : cap);
        __syncthreads();
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
            st_act1(&O[(size_t)(b * n_head + h) * V + v], f2bf(acc));
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
 * mla_test tolerance (rel_rms < 5e-3).
 *
 * ---------------------------------------------------------------------------------------------
 * FOLD MAP: WAVE-COOPERATIVE OVER DK, VECTOR-WIDE OVER V. (rewritten 2026-07-28)
 *
 * The fold used to give ONE thread a whole output column: `for l in 0..DK: acc += olds[l] *
 * wv[l*V+v]`. That is a DK-deep dependent chain with one strided global load per step and nothing
 * to overlap it with — measured 217 ns/iteration, i.e. exactly one uncached HBM latency, 512 times.
 * It also caps the op at n_head*V = 4096 live lanes on a 131 072-lane machine, and at V=256 <
 * PLOW_THREADS half of every workgroup idled. Cost: 111 us/packet, 0.6% of the 6200 GB/s ceiling,
 * 8.69 ms of a 34.7 ms GLM-5.2 token (perf-data/glm52-decode-attribution.md §0).
 *
 * The transform is the one `wave_dot_fp8_blk` (op_moe.h:237) calls "the ~1000x lever" and the one
 * `gemv_rows` (op_gemm.h:1330) already runs at 83-106% of ceiling: make the reduction axis
 * COOPERATIVE instead of one-thread-per-output. It is adapted rather than reused verbatim because
 * W_uv is stored l-major ([DK][V], v contiguous), the transpose of a GEMV weight ([N][K], k
 * contiguous) — so lanes must span V (to keep the loads coalesced) and the DK split has to go
 * across thread SLICES, with a cross-wave fold instead of `wave_sum`:
 *
 *   NV = VT/VEC lane-slots cover the tile's VT columns, VEC contiguous bf16 each (one dwordx2 /
 *                dwordx4 per lane; a whole tile row per NV lanes, fully coalesced).
 *   LS = PLOW_THREADS/NV  l-slices split the DK reduction; slice r owns the CONTIGUOUS block
 *                [r*BL, (r+1)*BL), BL = DK/LS.
 *   UN independent loads are issued per group before any is consumed (the `gemv_rows` idiom —
 *                this is what turns a latency chain into memory-level parallelism). Blocked slices
 *                make those UN loads consecutive W_uv rows.
 *   The LS partials fold in two steps: __shfl_xor over the l-slices that share a wave (free, and
 *   because the slices are blocked it is a binary tree over ADJACENT l-blocks), then PLOW_WAVES
 *   rows in LDS (`red`, PLOW_WAVES*VT floats behind olds) summed in increasing-l order.
 *
 * Every thread now does BL iterations of a VEC-wide FMA instead of DK of a scalar one, and all
 * PLOW_THREADS are live for any VT >= VEC. VT can therefore be lowered to fill the chip (VT=32 =>
 * 8 v-tiles => 128 workgroups at GLM tp4) without the per-workgroup collapse the old map suffered.
 *
 * NUMERICS — and BIT-IDENTITY IS NOT REACHABLE HERE, which is worth stating plainly. Reproducing
 * the old l=0..DK-1 order exactly requires one thread per output column, i.e. n_head*V = 4096
 * threads = 64 waves for the WHOLE op; at UN loads of 2 B each that is 8*UN KB in flight, against
 * the ~3 MB Little's-law figure needed to saturate 6200 GB/s at HBM latency. Any fix that reaches
 * the roofline must reassociate. What this map does instead is reassociate MINIMALLY: blocked
 * partials combined pairwise is textbook pairwise summation, whose error bound is strictly better
 * than the sequential sum it replaces. Measured against the shipped scalar body on identical
 * inputs at the GLM tp4 shape (perf-data/glm52_kbench_fold.*): **2 of 4096 outputs differ, each by
 * exactly 1 bf16 ulp** — the output is bf16, whose quantum (2^-9) is ~4000x the f32 reassociation
 * error, so a differing element is a coin-flip at a rounding boundary and nothing else.
 * The shapes that do not fit the map (VT not a multiple of VEC, NV > PLOW_WAVE, a ragged last tile,
 * V not a multiple of VEC) keep the original scalar body verbatim. */
#ifndef PLOW_MLA_FOLD_VEC
#define PLOW_MLA_FOLD_VEC 4 /* bf16 per lane-load: 4 = dwordx2. NV=VT/VEC lanes cover a tile row. */
#endif
#ifndef PLOW_MLA_FOLD_UN
#define PLOW_MLA_FOLD_UN 4 /* W_uv loads issued before any is consumed (the gemv_rows UN idiom) */
#endif
#ifndef PLOW_MLA_FOLD_VT
#define PLOW_MLA_FOLD_VT 32 /* V-tile when the op is given more workgroups than n_batch*n_head */
#endif
/* -DPLOW_MLA_FOLD_MAP=0 forces the ORIGINAL scalar fold back on while keeping the dispatch fix.
 * It exists for one open question and not as general flexibility: the new map is 7.7x faster and
 * differs from the old one by 1 bf16 ulp on 2 of 4096 outputs, which is enough to move GLM's
 * greedy trajectory off the recorded 24-token reference. Whoever owns that reference needs to be
 * able to A/B the two without editing code. Delete this knob once the call is made. */
#ifndef PLOW_MLA_FOLD_MAP
#define PLOW_MLA_FOLD_MAP 1
#endif
template <int VEC>
struct mla_fold_vec { /* clang lowers this to one global_load_dwordx{1,2,4} */
    typedef bf16 v __attribute__((ext_vector_type(VEC)));
};
template <int DK, int VT, int VEC = PLOW_MLA_FOLD_VEC, int UNW = PLOW_MLA_FOLD_UN>
__device__ void d_mla_merge_fold(bf16* __restrict__ O_, const float* __restrict__ Opart_,
                                 const float* __restrict__ mlpart_, const bf16* __restrict__ Wuv_,
                                 unsigned n_batch, unsigned n_head, unsigned V, unsigned nsplit,
                                 unsigned slice, unsigned nblk,
                                 float* olds /* DK + PLOW_WAVES*VT floats */) {
    auto* const O = as_glob(O_);
    const auto* const Opart = as_glob(Opart_);
    const auto* const mlpart = as_glob(mlpart_);
    const auto* const Wuv = as_glob(Wuv_);
    const unsigned tid = threadIdx.x;
    const unsigned vtiles = (V + VT - 1) / VT;
    const unsigned n_work = n_batch * n_head * vtiles;
    /* fold map, all compile-time. NV <= PLOW_WAVE keeps the l-slice fold inside one wave + one LDS
     * row per wave; BL % UN == 0 keeps the unrolled group exact, with no tail and no predication. */
    constexpr int NV = VT / VEC;
    constexpr int LS = NV > 0 ? (PLOW_THREADS / (NV > 0 ? NV : 1)) : 1;
    constexpr int BL = (LS > 0 && LS <= DK) ? (DK / LS) : 0; /* l-block per slice */
    constexpr int UN = (BL >= UNW) ? UNW : (BL > 0 ? BL : 1);
    constexpr bool MAP = PLOW_MLA_FOLD_MAP && (VEC > 0) && (VT % VEC == 0) && (NV > 0) &&
                         (NV <= PLOW_WAVE) && (PLOW_THREADS % NV == 0) && (LS <= DK) &&
                         (DK % LS == 0) && (BL > 0) && (UN >= 1) && (BL % UN == 0);
    typedef typename mla_fold_vec<VEC>::v wvec;
    float* const red = olds + DK; /* PLOW_WAVES*VT floats: the cross-wave l-slice fold */
    const unsigned wave = tid >> 6, lane = tid & (PLOW_WAVE - 1u);
    const unsigned cg = MAP ? (tid % (unsigned)(NV > 0 ? NV : 1)) : 0u;
    const unsigned rr = MAP ? (tid / (unsigned)(NV > 0 ? NV : 1)) : 0u;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned vt = w % vtiles;
        const unsigned bh = w / vtiles;
        const unsigned h = bh % n_head, b = bh / n_head;
        const auto* ml = mlpart + (size_t)(b * n_head + h) * nsplit * 2;

        /* global max / sum over the nsplit partials (online-softmax LSE combine). */
        float gm;
        const float gl = fa_merge_ml(ml, nsplit, gm);
        /* v_rcp_f32 -- see the note at d_flash_prefill's epilogue. */
        const float inv = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;

        /* Merge the DK-wide latent into LDS (rescaled, normalized). MS partial loads are issued
         * before any is consumed, and the per-split weight is BRANCH-FREE (a dead split weighs 0).
         * The old body carried `if (ml[s*2] == FA_NEG_INF) continue;` inside this loop, so every
         * one of the nsplit accumulations was a data-dependent branch over a global load: the
         * compiler could not batch them and the merge was a second latency chain behind the fold's.
         * Measured (ns=16 vs ns=1, VT=64): the merge cost ~6 us of the packet, now ~1.5 us.
         *
         * BIT-IDENTICAL to the old merge, deliberately: same `s` order, and the dead splits it used
         * to `continue` past now add `Opart*0`, which is exactly 0 because d_flash_mla_decode
         * ALWAYS writes op[d] (a dead split leaves oacc at 0 and stores 0.0f — op_attention.h:1509),
         * so no NaN can reach the select. `inv` stays OUT of the weight and is applied once at the
         * end, as before. The merge feeds all V output columns, so keeping it exact costs nothing
         * and removes it from the list of things that could have moved a token. */
        constexpr int MS = 8;
        const auto* opb = Opart + (size_t)(b * n_head + h) * nsplit * DK;
        for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
            float acc = 0.0f;
            unsigned s = 0;
            for (; s + MS <= nsplit; s += MS) {
                float pv[MS], wv[MS];
#pragma unroll
                for (int u = 0; u < MS; u++) pv[u] = opb[(size_t)(s + (unsigned)u) * DK + d];
#pragma unroll
                for (int u = 0; u < MS; u++) {
                    const float m = ml[(s + (unsigned)u) * 2];
                    wv[u] = (m == FA_NEG_INF) ? 0.0f : FA_EXP(m - gm);
                }
#pragma unroll
                for (int u = 0; u < MS; u++) acc += pv[u] * wv[u];
            }
            for (; s < nsplit; s++) {
                const float m = ml[s * 2];
                acc += (m == FA_NEG_INF) ? 0.0f : (opb[(size_t)s * DK + d] * FA_EXP(m - gm));
            }
            olds[d] = acc * inv;
        }
        __syncthreads();

        /* fold this workgroup's V-tile: o[v] = sum_l olat[l] * W_uv[h][l][v]. */
        const auto* wv = Wuv + (size_t)h * DK * V;
        const unsigned v0 = vt * VT, v1 = (v0 + VT < V) ? (v0 + VT) : V;
        auto* const orow = O + (size_t)(b * n_head + h) * V;
        if (MAP && v1 - v0 == (unsigned)VT && (V % (unsigned)VEC) == 0u) {
            const auto* const wcol = wv + v0 + cg * (unsigned)VEC;
            float acc[VEC];
#pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
            /* UN loads in flight, then consume: the whole point of the rewrite. Slice rr owns the
             * CONTIGUOUS l-block [rr*BL, (rr+1)*BL) rather than a strided share, so each partial is
             * a run of the original sequential sum and the cross-slice fold is textbook pairwise
             * summation — the reassociation with the SMALLEST departure from the old l=0..DK-1
             * order, which is what the token oracle is sensitive to. It also makes the UN loads
             * consecutive W_uv rows. */
            for (unsigned i = 0; i < (unsigned)BL; i += (unsigned)UN) {
                const unsigned l = rr * (unsigned)BL + i;
                wvec wq[UN];
                float sq[UN];
#pragma unroll
                for (int u = 0; u < UN; u++)
                    wq[u] = *(const PLOW_GLOB wvec*)(const PLOW_GLOB void*)(
                        wcol + (size_t)(l + (unsigned)u) * V);
#pragma unroll
                for (int u = 0; u < UN; u++) sq[u] = olds[l + (unsigned)u];
#pragma unroll
                for (int u = 0; u < UN; u++)
#pragma unroll
                    for (int k = 0; k < VEC; k++) acc[k] += sq[u] * bf2f(wq[u][k]);
            }
            /* fold the l-slices that share a wave (lanes cg, cg+NV, cg+2NV, ...), then the
             * PLOW_WAVES survivors through LDS. */
#pragma unroll
            for (int k = 0; k < VEC; k++) {
#pragma unroll
                for (int off = NV; off < PLOW_WAVE; off <<= 1)
                    acc[k] += __shfl_xor(acc[k], off, PLOW_WAVE);
            }
            if (lane < (unsigned)NV) {
#pragma unroll
                for (int k = 0; k < VEC; k++) red[wave * (unsigned)VT + cg * (unsigned)VEC + k] = acc[k];
            }
            __syncthreads();
            for (unsigned t = tid; t < (unsigned)VT; t += PLOW_THREADS) {
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < PLOW_WAVES; q++) s += red[(unsigned)q * (unsigned)VT + t];
                st_act1(&orow[v0 + t], f2bf(s));
            }
        } else {
            for (unsigned v = v0 + tid; v < v1; v += PLOW_THREADS) {
                float acc = 0.0f;
                for (unsigned l = 0; l < DK; l++) acc += olds[l] * bf2f(wv[(size_t)l * V + v]);
                st_act1(&orow[v], f2bf(acc));
            }
        }
        __syncthreads();
    }
}

/* TOKEN-BLOCKED merge+fold.  OPT-IN: -DPLOW_MLA_FOLD_TB=<G>, default 1 = this file is inert.
 *
 * WHY. `d_mla_merge_fold` above gives ONE workgroup ONE (row, V-tile), where a prefill "row" is
 * one (token, head) — the emitter folds the token axis into `n_batch` (crates/devgen/src/mla.rs,
 * "the token axis folds into n_batch"). At GLM-5.2 TP8 prefill that is n_batch=T, n_head=8,
 * V=256, DK=512, so VT=256, vtiles=1 and `n_work = T*8` rows over 304 workgroups. Each row
 * streams the WHOLE `W_uv[h]` panel — 512*256*2 = 256 KiB — and uses every byte of it exactly
 * ONCE, for a single olat vector. The panel is 256 KiB against a 32 KiB vector L1, so nothing is
 * caught on chip and the op re-reads it out of L2 once per token: at T=8192 that is
 *
 *     8192 tokens * 8 heads * 256 KiB = 16.8 GB of W_uv traffic PER LAYER,
 *
 * against 268 MB of Opart and 34 MB of O. The fold is 98% of the op's bytes and 99.2% of its
 * flops (512*256 MAC per row vs nsplit*512 for the merge), which is the first thing to say
 * plainly: MlaMergeFold is not priced by the causal KV-split. It is a batched GEMM
 * (O[T,256] = Olat[T,512] @ W_uv[h]) executed one M-row at a time, with the K x N panel
 * re-fetched for every row. The KV-split merge rides along at ~2% of it.
 *
 * WHAT THIS DOES. Give a workgroup TB CONSECUTIVE tokens of the same head instead of one. The
 * W_uv element a lane holds in a register is then consumed by TB accumulators instead of one, so
 * the panel is fetched once per TB tokens and the L2 stream divides by TB. Nothing else moves:
 * the lane->column map (NV), the l-slice split (LS/BL), the unroll (UN) and the two-stage
 * (shfl, LDS) fold are the ones `d_mla_merge_fold` already uses.
 *
 * BIT-IDENTITY. Held, deliberately, and it is the reason the loop nest is written in this order.
 * For a fixed token g the sequence of adds into `acc[g][k]` is exactly the sequence the scalar
 * body makes into `acc[k]`: same outer `i` order, same `u` order inside a group, same `l`-block
 * per wave, same shfl tree, same increasing-wave LDS sum. `TB` only interleaves INDEPENDENT
 * accumulator chains; it never reassociates one. The merge half is untouched per token. So this
 * is a pure memory-traffic transform and the output is bit-for-bit the shipped kernel's — which
 * is what makes it gateable against a character-identical control rather than a tolerance.
 *
 * LDS. `olds` grows to TB*DK floats. `red` does NOT grow to TB*PLOW_WAVES*VT (16384 floats at
 * TB=8, VT=256 — 64 KiB, over the arena on top of olds); the cross-wave fold runs in chunks of
 * RB tokens, so the buffer is RB*PLOW_WAVES*VT and the barrier count is 2*TB/RB per work item
 * rather than 2*TB. RB is the largest power of two that fits PLOW_MLA_FOLD_TB_LDS floats.
 *
 * SHAPE PRECONDITIONS, all checked by the caller (runtime/amd/interp.hip), never here:
 * n_batch % TB == 0, and the fast map must be reachable (v1-v0 == VT, V % VEC == 0) — a shape
 * that falls into the scalar `else` would gain nothing and this body does not carry that arm. */
#ifndef PLOW_MLA_FOLD_TB
#define PLOW_MLA_FOLD_TB 1 /* tokens per workgroup in the fold. 1 = arm absent (default) */
#endif
#ifndef PLOW_MLA_FOLD_TB_LDS
#define PLOW_MLA_FOLD_TB_LDS 12288 /* floats of `olds` arena the TB arm may use (48 KiB) */
#endif
#if PLOW_MLA_FOLD_TB > 1
template <int DK, int VT, int TB, int VEC = PLOW_MLA_FOLD_VEC, int UNW = PLOW_MLA_FOLD_UN,
          bool MB = true>
__device__ void d_mla_merge_fold_tb(bf16* __restrict__ O_, const float* __restrict__ Opart_,
                                    const float* __restrict__ mlpart_,
                                    const bf16* __restrict__ Wuv_, unsigned n_batch,
                                    unsigned n_head, unsigned V, unsigned nsplit, unsigned slice,
                                    unsigned nblk, float* olds /* TB*DK + RB*PLOW_WAVES*VT */) {
    auto* const O = as_glob(O_);
    const auto* const Opart = as_glob(Opart_);
    const auto* const mlpart = as_glob(mlpart_);
    const auto* const Wuv = as_glob(Wuv_);
    const unsigned tid = threadIdx.x;
    const unsigned vtiles = (V + VT - 1) / VT;
    const unsigned n_bg = n_batch / (unsigned)TB; /* caller guarantees n_batch % TB == 0 */
    const unsigned n_work = n_bg * n_head * vtiles;
    constexpr int NV = VT / VEC;
    constexpr int LS = PLOW_THREADS / NV;
    constexpr int BL = DK / LS;
    constexpr int UN = (BL >= UNW) ? UNW : BL;
    /* RB: tokens whose cross-wave partials are live in LDS at once. */
    constexpr int RB_FIT = (PLOW_MLA_FOLD_TB_LDS - TB * DK) / (PLOW_WAVES * VT);
    constexpr int RB = RB_FIT >= TB ? TB : (RB_FIT >= 4 ? 4 : (RB_FIT >= 2 ? 2 : 1));
    static_assert(VT % VEC == 0 && NV > 0 && NV <= PLOW_WAVE && PLOW_THREADS % NV == 0 &&
                      LS <= DK && DK % LS == 0 && BL > 0 && BL % UN == 0 && TB % RB == 0 &&
                      TB * DK + RB * PLOW_WAVES * VT <= PLOW_MLA_FOLD_TB_LDS,
                  "PLOW_MLA_FOLD_TB map does not close");
    typedef typename mla_fold_vec<VEC>::v wvec;
    float* const red = olds + (size_t)TB * DK;
    const unsigned wave = tid >> 6, lane = tid & (PLOW_WAVE - 1u);
    const unsigned cg = tid % (unsigned)NV;
    const unsigned rr = tid / (unsigned)NV;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned vt = w % vtiles;
        const unsigned r = w / vtiles;
        const unsigned h = r % n_head, bg = r / n_head;
        const unsigned b0 = bg * (unsigned)TB;
        /* ---- merge: TB latents into LDS.
         *
         * MB (default) INTERLEAVES the TB tokens instead of running them back to back. That is
         * not cosmetic: the merge half of this op is LATENCY-bound, not bandwidth-bound. Measured
         * standalone at the T=8192 TP8 shape it is 348 us/packet moving 288 MB = 828 GB/s, one
         * sixth of this part's HBM ceiling — because per token a thread issues ONE ml load whose
         * result gates the exp, ONE (nsplit=2) Opart load, an LDS store and a barrier, with only
         * 8 waves resident and nothing else to overlap. Serialising TB of those costs TB
         * latencies; interleaving them costs one, because the TB rows are independent and their
         * loads all issue before any is consumed.
         *
         * Bit-identity is unaffected: for each token the adds into `acc[g]` run over `s` in the
         * same order, and `inv` is still applied once at the end. Reordering across INDEPENDENT
         * tokens is not a reassociation. The `s`-blocked MS=8 form the scalar body uses is a
         * scheduling device with the same value (a dead split weighs 0 either way), so the plain
         * `s` loop here is exact for every nsplit, not just the prefill nsplit<=2. */
        if constexpr (MB) {
            float gm[TB], inv[TB];
            const float* mlg[TB];
            const float* opg[TB];
#pragma unroll
            for (int g = 0; g < TB; g++) {
                const unsigned row = (b0 + (unsigned)g) * n_head + h;
                mlg[g] = mlpart + (size_t)row * nsplit * 2;
                opg[g] = Opart + (size_t)row * nsplit * DK;
                const float gl = fa_merge_ml(mlg[g], nsplit, gm[g]);
                inv[g] = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;
            }
            for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
                float acc[TB];
#pragma unroll
                for (int g = 0; g < TB; g++) acc[g] = 0.0f;
                unsigned s = 0;
                /* NO MS=8 block here, unlike the scalar body, and that is a PRECONDITION not an
                 * omission. Carrying it would cost TB*MS*2 = 128 live floats at TB=8 (measured
                 * 231 VGPRs against 130 without) in a block that prefill never enters — and it
                 * would not even be reachable: the caller routes here only for nsplit < MS, so
                 * the scalar body is on its tail path too and the two agree bit for bit. A
                 * nsplit >= MS packet MUST NOT reach this arm; interp.hip refuses it.
                 *
                 * Tail, written in the scalar body's EXACT expression shape — the
                 * select feeding the add, not a hoisted weight multiplied in. `acc += pv * w`
                 * CONTRACTS to one v_fmac_f32 (one rounding); `acc += select(0, pv * w)` cannot
                 * (the select breaks the pattern), so it is a v_mul + v_add, two roundings. The
                 * two differ, and at nsplit=2 — the shipped prefill ns — this tail is the ONLY
                 * path taken, so writing it the natural way silently forfeits bit-identity.
                 * Measured: the hoisted-weight form matched the shipped kernel at ns=1 (where
                 * acc is still 0 and fma(a,b,0) == a*b) and DIFFERED at ns=2. */
                for (; s < nsplit; s++) {
                    float pv[TB], mm[TB], ex[TB];
#pragma unroll
                    for (int g = 0; g < TB; g++) pv[g] = opg[g][(size_t)s * DK + d];
#pragma unroll
                    for (int g = 0; g < TB; g++) {
                        mm[g] = mlg[g][s * 2];
                        ex[g] = FA_EXP(mm[g] - gm[g]);
                    }
#pragma unroll
                    for (int g = 0; g < TB; g++)
                        acc[g] += (mm[g] == FA_NEG_INF) ? 0.0f : (pv[g] * ex[g]);
                }
#pragma unroll
                for (int g = 0; g < TB; g++) olds[(size_t)g * DK + d] = acc[g] * inv[g];
            }
        } else
#pragma unroll 1
        for (int g = 0; g < TB; g++) {
            const unsigned row = (b0 + (unsigned)g) * n_head + h;
            const auto* ml = mlpart + (size_t)row * nsplit * 2;
            float gm;
            const float gl = fa_merge_ml(ml, nsplit, gm);
            const float inv = (gl > 0.0f) ? FA_RECIP(gl) : 0.0f;
            constexpr int MS = 8;
            const auto* opb = Opart + (size_t)row * nsplit * DK;
            for (unsigned d = tid; d < DK; d += PLOW_THREADS) {
                float acc = 0.0f;
                unsigned s = 0;
                for (; s + MS <= nsplit; s += MS) {
                    float pv[MS], wvv[MS];
#pragma unroll
                    for (int u = 0; u < MS; u++) pv[u] = opb[(size_t)(s + (unsigned)u) * DK + d];
#pragma unroll
                    for (int u = 0; u < MS; u++) {
                        const float m = ml[(s + (unsigned)u) * 2];
                        wvv[u] = (m == FA_NEG_INF) ? 0.0f : FA_EXP(m - gm);
                    }
#pragma unroll
                    for (int u = 0; u < MS; u++) acc += pv[u] * wvv[u];
                }
                for (; s < nsplit; s++) {
                    const float m = ml[s * 2];
                    acc += (m == FA_NEG_INF) ? 0.0f : (opb[(size_t)s * DK + d] * FA_EXP(m - gm));
                }
                olds[(size_t)g * DK + d] = acc * inv;
            }
        }
        __syncthreads();
        /* ---- fold: one W_uv panel, TB accumulator chains. */
        const auto* wv = Wuv + (size_t)h * DK * V;
        const unsigned v0 = vt * VT;
        const auto* const wcol = wv + v0 + cg * (unsigned)VEC;
        float acc[TB][VEC];
#pragma unroll
        for (int g = 0; g < TB; g++)
#pragma unroll
            for (int k = 0; k < VEC; k++) acc[g][k] = 0.0f;
        for (unsigned i = 0; i < (unsigned)BL; i += (unsigned)UN) {
            const unsigned l = rr * (unsigned)BL + i;
            wvec wq[UN];
#pragma unroll
            for (int u = 0; u < UN; u++)
                wq[u] = *(const PLOW_GLOB wvec*)(const PLOW_GLOB void*)(
                    wcol + (size_t)(l + (unsigned)u) * V);
            float sq[TB][UN];
#pragma unroll
            for (int g = 0; g < TB; g++)
#pragma unroll
                for (int u = 0; u < UN; u++) sq[g][u] = olds[(size_t)g * DK + l + (unsigned)u];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                float wf[VEC];
#pragma unroll
                for (int k = 0; k < VEC; k++) wf[k] = bf2f(wq[u][k]);
#pragma unroll
                for (int g = 0; g < TB; g++)
#pragma unroll
                    for (int k = 0; k < VEC; k++) acc[g][k] += sq[g][u] * wf[k];
            }
        }
        /* fold the l-slices sharing a wave (inert when NV == PLOW_WAVE), then the PLOW_WAVES
         * survivors through LDS in increasing-l order — the scalar body's tree, per token. */
#pragma unroll
        for (int g = 0; g < TB; g++)
#pragma unroll
            for (int k = 0; k < VEC; k++)
#pragma unroll
                for (int off = NV; off < PLOW_WAVE; off <<= 1)
                    acc[g][k] += __shfl_xor(acc[g][k], off, PLOW_WAVE);
#pragma unroll 1
        for (int gb = 0; gb < TB; gb += RB) {
            __syncthreads(); /* the previous chunk's readers are done with `red` */
            if (lane < (unsigned)NV) {
#pragma unroll
                for (int j = 0; j < RB; j++)
#pragma unroll
                    for (int k = 0; k < VEC; k++)
                        red[((unsigned)j * PLOW_WAVES + wave) * (unsigned)VT +
                            cg * (unsigned)VEC + k] = acc[gb + j][k];
            }
            __syncthreads();
#pragma unroll 1
            for (int j = 0; j < RB; j++) {
                auto* const orow = O + (size_t)((b0 + (unsigned)(gb + j)) * n_head + h) * V;
                for (unsigned t = tid; t < (unsigned)VT; t += PLOW_THREADS) {
                    float s = 0.0f;
#pragma unroll
                    for (int q = 0; q < PLOW_WAVES; q++)
                        s += red[((unsigned)j * PLOW_WAVES + (unsigned)q) * (unsigned)VT + t];
                    st_act1(&orow[v0 + t], f2bf(s));
                }
            }
        }
        __syncthreads();
    }
}
#endif /* PLOW_MLA_FOLD_TB > 1 */

#endif /* PLOW_OP_ATTENTION_H */
