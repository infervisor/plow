/* op_moe.h — the MoE data-dependent counter-gate kernels (gfx950).
 *
 * plans/moe-plow-design.md §3-§4, plans/moe-ep-kernels.md §2-§3. Four ops implement the
 * dispatch CORE: a router that writes a routing table, K expert-body slots that read it and
 * compute-or-skip (streaming ONLY the chosen experts' weights via a two-level pointer
 * indirection), and a deterministic combine. The interpreter's gate/signal loop is
 * UNCHANGED: each expert packet always signals its completion counter — the "am I done?"
 * question is static; only "should I run?" is data-dependent, and that lives here in the body.
 *
 * BIT-EXACTNESS. Every GEMV here is ONE OUTPUT ELEMENT PER THREAD with a SEQUENTIAL f32 dot
 * over K (no wave-tree reduction), and every op-boundary rounds to bf16 (RNE). This makes the
 * summation order deterministic and trivially reproducible by an independent CPU reference —
 * the discipline net_gemma_block_test.c calls "truth is the CPU reference". Partials are kept
 * f32 into the combine (moe-ep-kernels §3b). Perf is out of scope here (that is M4); these are
 * the correctness kernels. The GEMV swaps to a tiled/fp8 path later without touching the
 * dispatch structure.
 *
 * ROUTING TABLE entry = { uint32 expert_id ; float gate } (8 bytes), PLOW_EXPERT_UNUSED marks
 * an unused slot. EXPERT WEIGHT TABLE = uint64 device pointers, layout [n_exp][3] =
 * { gate_base, up_base, down_base }, filled at load by orch/moe.rs::resolve_expert_tables.
 */
#ifndef PLOW_OP_MOE_H
#define PLOW_OP_MOE_H

#include "amd_common.h"

/* Grouped-expert body: 0 = slot-outer (proven bit-identical), 1 = flattened (slot,channel) sweep
 * for the M=1 decode tail. Build-time A/B knob (see d_moe_group_*). */
#ifndef PLOW_MOE_GROUP_FLAT
#define PLOW_MOE_GROUP_FLAT 0
#endif
/* Grouped gate/up on the XDL matrix cores (fp8->bf16 16x16x32 MFMA, M=1 padded) vs the fdot2 VALU
 * path. 1 = MFMA. Bit-close (not bit-identical). Build-time A/B knob. */
#ifndef PLOW_MOE_MFMA
#define PLOW_MOE_MFMA 0
#endif

/* GLU activation, matching op_elementwise.h / the Rust reference. act: 0 = gelu_tanh, 1 = silu. */
__device__ __forceinline__ float moe_act(float x, unsigned act) {
    if (act == 1) return x / (1.0f + expf(-x)); /* silu (SwiGLU — GLM) */
    const float k = 0.7978845608028654f;         /* gelu_tanh */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* Sequential f32 dot of a bf16 row against a bf16 vector, bf16-rounded — the decode GEMV
 * element. `x` has K entries, `w` is one output row [K]. */
__device__ __forceinline__ float moe_dot_bf16(const bf16* x, const bf16* w, unsigned K) {
    float acc = 0.0f;
    for (unsigned i = 0; i < K; i++) acc += bf2f(x[i]) * bf2f(w[i]);
    return bf2f(f2bf(acc)); /* the op stores bf16 */
}

/* --- kernel 1: ROUTER (PLOW_DOP_MOE_ROUTER, moe-ep-kernels §2b) --------------------------
 * table(out) = [k] of {u32 expert_id, f32 gate}. Runs on ONE workgroup (n_exp<=256 is tiny).
 *   logit[e] = bf16_round(x . Wr[e])   (parallel over experts)
 *   score[e] = sigmoid|softmax(logit)
 *   top-k via k-pass masked argmax, LOWEST-ID tie-break (packed key)   (serial, thread 0)
 *   if norm_topk: gate_j /= Σ gate ; gate_j *= route_scale ; write table
 * `bias` (DeepSeek-V3 / GLM-5.2 noaux_tc `e_score_correction_bias`, or NULL) is added to the score
 * for SELECTION RANKING ONLY — the gate value stays the raw unbiased score (moe-plow-design §5a).
 * `lds` is float[n_exp] score scratch. */
__device__ void d_moe_router(unsigned char* table, const bf16* x, const bf16* Wr,
                             const float* bias, unsigned H, unsigned n_exp, unsigned k,
                             unsigned flags, float route_scale, unsigned slice, unsigned nblk,
                             float* lds) {
    (void)nblk;
    if (slice != 0) return; /* emitted on 1 CU; guard is belt-and-braces */
    const bool sigmoid = (flags & 1u) != 0;
    const bool norm_topk = (flags & 2u) != 0;

    /* logits + per-expert score into LDS, parallel over experts. */
    for (unsigned e = threadIdx.x; e < n_exp; e += PLOW_THREADS) {
        float logit = moe_dot_bf16(x, Wr + (size_t)e * H, H);
        lds[e] = logit;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        /* score transform (in place). Softmax needs the global max/sum; both are cheap here. */
        if (sigmoid) {
            for (unsigned e = 0; e < n_exp; e++) lds[e] = 1.0f / (1.0f + expf(-lds[e]));
        } else {
            float m = -1e30f;
            for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, lds[e]);
            float s = 0.0f;
            for (unsigned e = 0; e < n_exp; e++) { lds[e] = expf(lds[e] - m); s += lds[e]; }
            for (unsigned e = 0; e < n_exp; e++) lds[e] /= s;
        }

        /* k-pass masked argmax over packed keys: (ordered_bits(score+bias)<<20) | (n_exp-1-id).
         * A plain unsigned max picks the top BIASED score and, on a tie, the LOWEST id. The gate is
         * the winner's UNBIASED score (captured from lds before the kill); killing lds[best_id]
         * also kills its biased key (bias is finite), so the next pass cannot re-pick it. */
        unsigned winner[8]; /* top_k <= 8 for every target config (GLM/DS/Qwen/Mixtral) */
        float gate[8];
        for (unsigned j = 0; j < k; j++) {
            unsigned long long best = 0ull;
            unsigned best_id = 0;
            for (unsigned e = 0; e < n_exp; e++) {
                unsigned sb;
                float sc = lds[e] + (bias ? bias[e] : 0.0f); /* biased score = selection key only */
                __builtin_memcpy(&sb, &sc, 4);
                sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u); /* monotone f32->u32 */
                unsigned long long key =
                    ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1 - e) & 0xFFFFFu);
                if (key > best) { best = key; best_id = e; }
            }
            winner[j] = best_id;
            gate[j] = lds[best_id];   /* the gate = the winner's unbiased score */
            lds[best_id] = -1e30f;    /* kill so the next pass cannot re-pick it */
        }

        float sum = 0.0f;
        for (unsigned j = 0; j < k; j++) sum += gate[j];
        for (unsigned j = 0; j < k; j++) {
            if (norm_topk && sum != 0.0f) gate[j] /= sum;
            gate[j] *= route_scale;
        }

        for (unsigned j = 0; j < k; j++) {
            unsigned* idp = (unsigned*)(table + (size_t)j * 8);
            float* gp = (float*)(table + (size_t)j * 8 + 4);
            idp[0] = winner[j];
            gp[0] = gate[j];
        }
    }
}

/* --- kernel 1b: ROUTER TOP-K (PLOW_DOP_MOE_ROUTER_TOPK) -----------------------------------
 * The router SPLIT: the score matmul logit[e]=x·Wr[e] is the ordinary multi-CU wave-cooperative GEMV
 * (DevOp::Gemv over the 256 experts), and THIS op is the cheap tail: score transform + group-limited
 * top-k + norm + scale over the <=256 precomputed logits. `logit` is bf16[n_exp] (the GEMV output;
 * the old d_moe_router rounded the dot to bf16 identically). sigmoid/softmax, e_score_correction_bias
 * on SELECTION only, lowest-id tie-break, norm_topk, route_scale are byte-for-byte the old d_moe_router
 * so the 8-of-256 SET is unchanged (byte-identical token stream).
 *
 * WORKGROUP-PARALLEL (the fix): the whole selection used to run on THREAD 0 of the single workgroup
 * (256 expf + k passes of a 256-scan) — ~190us/MoE-layer, the #1 straggler (replicated, FLAT in ctx
 * and TP). The sigmoid transform and the top-k selection are now spread across the workgroup's
 * PLOW_THREADS lanes; the selection is done by RANK in ONE all-pairs pass (see below) rather than k
 * sequential masked-argmax passes, so the packed-key tie-break (ordered(score+bias)<<20 | (n_exp-1-id))
 * still gives highest biased score / lowest id on tie — the IDENTICAL winners the serial scan produced.
 * gate = the UNBIASED score. softmax (rare, non-GLM) stays on thread 0 to preserve its exact serial
 * max/sum. */
__device__ void d_moe_router_topk(unsigned char* table, const bf16* logit, const float* bias,
                                  unsigned n_exp, unsigned k, unsigned flags, float route_scale,
                                  unsigned slice, unsigned nblk, float* lds) {
    (void)nblk;
    if (slice != 0) return; /* emitted on 1 CU; guard is belt-and-braces */
    const bool sigmoid = (flags & 1u) != 0;
    const bool norm_topk = (flags & 2u) != 0;
    const unsigned tid = threadIdx.x;
    /* key/winner scratch carved from the tail of the LDS score arena (past the n_exp scores, 16-byte
     * aligned) — the raw arena (>= flash tiles) easily holds n_exp scores + n_exp u64 keys + wl. NOT
     * sm->am, which unions over sm->raw (== lds) and would clobber the scores. */
    unsigned long long* keys = (unsigned long long*)(lds + ((n_exp + 3u) & ~3u));
    unsigned* wl = (unsigned*)(keys + n_exp);

    if (sigmoid) { /* the hot GLM/DS path — parallel, per-expert independent */
        for (unsigned e = tid; e < n_exp; e += PLOW_THREADS)
            lds[e] = 1.0f / (1.0f + expf(-bf2f(logit[e])));
    } else {       /* softmax needs a global max+sum; keep it exactly serial on thread 0 (rare) */
        for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) lds[e] = bf2f(logit[e]);
        __syncthreads();
        if (tid == 0) {
            float m = -1e30f;
            for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, lds[e]);
            float s = 0.0f;
            for (unsigned e = 0; e < n_exp; e++) { lds[e] = expf(lds[e] - m); s += lds[e]; }
            for (unsigned e = 0; e < n_exp; e++) lds[e] /= s;
        }
    }
    __syncthreads();

    /* SELECTION is RANK, not k passes (the lever): the top-k was 8 SEQUENTIAL block_max_u64 passes,
     * each a full workgroup reduction (2 barriers) plus a kill+barrier — ~24 dependent barriers on the
     * ONE workgroup that runs while 255 CUs wait. Because the packed key (ordered(score+bias)<<20 |
     * (n_exp-1-id)) is UNIQUE per expert (distinct id low bits), the winners are exactly the k experts
     * of LOWEST rank, rank(e) = #{f : key_f > key_e}. So the whole selection is ONE all-pairs pass with
     * just TWO extra barriers: pack every key into LDS, then each expert counts how many keys beat it
     * and drops itself into wl[rank] when rank<k. winner[rank]=e is the same descending-key order the
     * masked argmax produced (rank 0 = highest key); gate = the winner's UNBIASED score; norm/scale
     * unchanged — so the 8-of-256 SET, ORDER, and GATES are BYTE-IDENTICAL. Measured (gfx950 standalone):
     * 11.6 -> 7.6 us (1.52x). The scan is LDS-bandwidth bound (unroll 32 keeps ~8 ds_read_b64 in flight);
     * the sigmoid's 256 expf is the bit-exact-locked floor. */
    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) {
        unsigned sb;
        float sc = lds[e] + (bias ? bias[e] : 0.0f); /* biased score = SELECTION key only */
        __builtin_memcpy(&sb, &sc, 4);
        sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u); /* monotone f32->u32 */
        keys[e] = ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1 - e) & 0xFFFFFu);
    }
    __syncthreads();

    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) {
        const unsigned long long myk = keys[e];
        unsigned rank = 0;
#pragma unroll 32
        for (unsigned f = 0; f < n_exp; f++) rank += (keys[f] > myk);
        if (rank < k) wl[rank] = e; /* winner of rank `rank` (rank 0 == highest key) */
    }
    __syncthreads();

    if (tid == 0) {
        float gate[8];
        for (unsigned j = 0; j < k; j++) gate[j] = lds[wl[j]]; /* winner's UNBIASED score */
        float sum = 0.0f;
        for (unsigned j = 0; j < k; j++) sum += gate[j];
        for (unsigned j = 0; j < k; j++) {
            if (norm_topk && sum != 0.0f) gate[j] /= sum;
            gate[j] *= route_scale;
            unsigned* idp = (unsigned*)(table + (size_t)j * 8);
            float* gp = (float*)(table + (size_t)j * 8 + 4);
            idp[0] = wl[j];
            gp[0] = gate[j];
        }
    }
}

/* Read this slot's expert id from the routing table. */
__device__ __forceinline__ unsigned moe_slot_expert(const unsigned char* table, unsigned slot) {
    return *(const unsigned*)(table + (size_t)slot * 8);
}
__device__ __forceinline__ float moe_slot_gate(const unsigned char* table, unsigned slot) {
    return *(const float*)(table + (size_t)slot * 8 + 4);
}

/* Block-fp8 (DeepSeek/GLM weight_block_size [128,128]) WAVE-REDUCED dot of ONE output channel — the
 * fp8 twin of moe_dot_bf16, but the 64 lanes of a wave COOPERATE on the K reduction (each lane owns
 * 16 consecutive fp8) instead of one thread walking all of K. This is the parallel decode structure
 * of gemv_rows_fp8_blk brought to the expert body — the ~1000x lever over the old one-output-per-
 * thread sequential dot. Exact scaled-cvt fp8->bf16 decode, per-128-block f32 scale multiplied into
 * each lane's chunk partial (a lane's 16 fp8 lie within one 128-K block, so one scale/chunk), then
 * wave_sum (all-reduce → valid on every lane). `x` is bf16[K] global; `Wrow` is e4m3[K]; `srow` is
 * this channel's block-scale row (KB=ceil(K/128) f32). K is a 128-multiple for every real GLM expert
 * (H=6144, I_moe=2048); the KB-1 clamp keeps an overshoot lane's read in-bounds (its partial is 0). */
__device__ __forceinline__ float wave_dot_fp8_blk(const bf16* x, const unsigned char* Wrow,
                                                  const float* srow, unsigned K, unsigned lane) {
    const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8_u(Wrow, K);
    const unsigned step = PLOW_WAVE * 16; /* 64 lanes x 16 fp8 = 1024 K per pass = 8 blocks of 128 */
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned KB = (K + 127u) >> 7;
    /* NOTE: a UN-deep software prefetch pipeline (the gemv_rows_fp8_blk idiom, op_gemm.h:1504) was
     * tried here to hide the per-chunk `s_waitcnt vmcnt(0)` HBM latency. It is BIT-EXACT (verified
     * vs the CPU oracle, moe_block_gfx950_test) but MEASURED A NET REGRESSION at M=1 decode: the
     * DOWN op (K=I_moe=2048 -> only nchunk=2) is too shallow to pipeline and regressed +17-19%
     * (per-op trace, TP4 ctx1024), while GLU (K=H=6144 -> nchunk=6) was flat at UN=2 and only
     * ~-8% at UN=3 — net +1.6-2% tpot either way. The block-fp8 expert GEMV is NOT weight-load-
     * latency-bound the way the ISA suggested; keep the serial one-load-per-chunk form. */
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * step + lane * 16;
        unsigned kb = k >> 7;
        if (kb >= KB) kb = KB - 1;
        const unsigned kx = (k < K) ? k : 0u;
        const float bs = srow[kb];       /* arbitrary-f32 block scale: separate multiply (E4M3->bf16
                                          * decode is exact; scale is NOT foldable into the cvt — the
                                          * scalef32 operand is E8M0/power-of-2 only, see gemv path) */
        fp8v16 wv = buf_ld_fp8(wr, k);   /* 16 fp8; past-K bytes return 0 */
        bf16v8 wlo, whi;
        fp8_to_bf16v8(wv, wlo, whi);
        acc += dot8(whi, ld_glob8(x + kx + 8), dot8(wlo, ld_glob8(x + kx), 0.0f)) * bs;
    }
    return wave_sum(acc);
}

/* --- kernel 2a-fp8: EXPERT GATE/UP, block-fp8 (PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK) ---------------
 * Block-fp8 twin of d_moe_expert_glu, PARALLEL (wave-per-output). Weight bases from wtab[eid*3+{0,1}]
 * (fp8 rows); block-scale grid bases from stab[eid*3+{0,1}] ([I_moe/128][H/128] f32, row-major). x
 * stays bf16 (w8a16). ONE OUTPUT CHANNEL PER WAVE: the 64 lanes wave-reduce gate·x and up·x via
 * wave_dot_fp8_blk, then SwiGLU. Sentinel-skip + two-level wtab/stab indirection + SwiGLU semantics
 * unchanged. (NOTE: a hand-FUSED gate+up loop was tried for x-read sharing/ILP but it miscompiled in
 * the interp megakernel — a stray store faulted despite 0 spill; the two-call form is correct in both
 * the standalone and the full B4 interp, and the x re-read is a cheap L2 hit while gate/up WEIGHTS —
 * the bandwidth-bound term — are read once each either way.) */
__device__ void d_moe_expert_glu_fp8_blk(bf16* fu, const bf16* x, const unsigned char* table,
                                         const unsigned long long* wtab,
                                         const unsigned long long* stab, unsigned slot,
                                         unsigned I_moe, unsigned H, unsigned n_exp, unsigned act,
                                         unsigned slice, unsigned nblk) {
    const unsigned eid = moe_slot_expert(table, slot);
    if (eid >= n_exp) return; /* sentinel: skip, stream zero bytes — interp still signals */
    const unsigned long long wg_base = wtab[(size_t)eid * 3 + 0];
    if (wg_base == 0ull) return; /* EP: expert not owned by this rank (null base) — skip */
    const unsigned char* Wg = (const unsigned char*)(size_t)wg_base;
    const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
    const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
    const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
    const unsigned KB = (H + 127u) >> 7;
    bf16* fu_slot = fu + (size_t)slot * I_moe;

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    for (unsigned n = slice * PLOW_WAVES + wave; n < I_moe; n += wstride) {
        const unsigned nrow = (n >> 7) * KB; /* this channel's block-scale row */
        const float g = wave_dot_fp8_blk(x, Wg + (size_t)n * H, Sg + nrow, H, lane);
        const float u = wave_dot_fp8_blk(x, Wu + (size_t)n * H, Su + nrow, H, lane);
        if (lane == 0) fu_slot[n] = f2bf(moe_act(g, act) * u);
    }
}

/* --- kernel 2b-fp8: EXPERT DOWN, block-fp8 (PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK) -----------------
 * Block-fp8 twin of d_moe_expert_down, PARALLEL (wave-per-output). Wd base from wtab[eid*3+2]; scale
 * grid from stab[eid*3+2] ([H/128][I_moe/128] f32). fu (this expert's gate/up output) stays bf16.
 * ONE OUTPUT ROW h PER WAVE; the sentinel-skip zeroes the slot's partial across ALL threads (the
 * fixed-slot combine sums a deterministic zero), and the gate multiply lands on the f32 partial. */
__device__ void d_moe_expert_down_fp8_blk(float* part, const bf16* fu, const unsigned char* table,
                                          const unsigned long long* wtab,
                                          const unsigned long long* stab, unsigned slot, unsigned H,
                                          unsigned I_moe, unsigned n_exp, unsigned slice,
                                          unsigned nblk) {
    const unsigned eid = moe_slot_expert(table, slot);
    float* part_slot = part + (size_t)slot * H;
    /* skip (sentinel OR EP-non-local null base): zero this slot's partial (all threads share row) */
    if (eid >= n_exp || wtab[(size_t)eid * 3 + 2] == 0ull) {
        const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
        const unsigned stride = nblk * PLOW_THREADS;
        for (unsigned h = gid; h < H; h += stride) part_slot[h] = 0.0f;
        return;
    }
    const float gate = moe_slot_gate(table, slot);
    const unsigned char* Wd = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 2];
    const float* Sd = (const float*)(size_t)stab[(size_t)eid * 3 + 2];
    const unsigned KB = (I_moe + 127u) >> 7;
    const bf16* fu_slot = fu + (size_t)slot * I_moe;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    for (unsigned h = slice * PLOW_WAVES + wave; h < H; h += wstride) {
        const float y = wave_dot_fp8_blk(fu_slot, Wd + (size_t)h * I_moe,
                                         Sd + (size_t)(h >> 7) * KB, I_moe, lane);
        if (lane == 0) part_slot[h] = gate * y; /* f32 partial into the combine */
    }
}

/* --- GROUPED block-fp8 experts (PLOW_DOP_MOE_GROUP_{GLU,DOWN}_FP8_BLK 48/49) -----------------
 * ONE packet computes ALL k chosen expert slots that the per-slot ops 45/46 did one-at-a-time. The k
 * separate counter-gated packets collapse to a single packet (one counter edge, one interp dispatch)
 * — the OP-OVERHEAD lever for GLM M=1 decode. Two body variants, selectable at build time:
 *
 *  (a) SLOT-OUTER (default, PROVEN bit-identical to per-slot): the per-output work is byte-for-byte
 *      the per-slot kernel; only the outer slot loop moved inside. Simple, deadlock-safe.
 *  (b) FLATTENED (PLOW_MOE_GROUP_FLAT=1): the grid walks a SINGLE flat (slot,channel) index space of
 *      k*I_moe (glu) / k*H (down) outputs, so all k experts' channels are load-balanced across every
 *      wave in ONE sweep instead of k tail-underutilised sweeps. Same per-output arithmetic, so still
 *      bit-identical; the win is fewer idle waves in the M=1 tail (the AITER per_1x128 M=1 idea: one
 *      grouped launch over the whole [k,N] output space, weight base resolved per slot from the table).
 *
 * BIT-IDENTICAL to the per-slot path either way: the per-output wave_dot_fp8_blk gate/up/down, SwiGLU,
 * gate-scale, and fixed-slot part write are the same; only the loop nest differs. Sentinel/non-local
 * slots (eid>=n_exp OR null weight base) are skipped exactly as the per-slot ops. */
#if PLOW_MOE_MFMA
__device__ void d_moe_group_glu_mfma(bf16*, const bf16*, const unsigned char*,
                                     const unsigned long long*, const unsigned long long*, unsigned,
                                     unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
#endif
__device__ void d_moe_group_glu_fp8_blk(bf16* fu, const bf16* x, const unsigned char* table,
                                        const unsigned long long* wtab,
                                        const unsigned long long* stab, unsigned k, unsigned I_moe,
                                        unsigned H, unsigned n_exp, unsigned act, unsigned slice,
                                        unsigned nblk) {
#if PLOW_MOE_MFMA
    d_moe_group_glu_mfma(fu, x, table, wtab, stab, k, I_moe, H, n_exp, act, slice, nblk);
#elif PLOW_MOE_GROUP_FLAT
    /* Flat (slot,channel) sweep: one wave per (slot, output-channel) over the whole k*I_moe space.
     * The activation `x` is shared across all slots and channels; only the expert weight base changes
     * per slot. wchan = KB is constant; resolve the per-slot bases once per channel-group is not
     * possible without a division, so we resolve per output — a couple L1-hot scalar loads. */
    const unsigned KB = (H + 127u) >> 7;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    const unsigned total = k * I_moe;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += wstride) {
        const unsigned slot = f / I_moe;
        const unsigned n = f - slot * I_moe;
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;                       /* sentinel/non-local: skip */
        const unsigned long long wg = wtab[(size_t)eid * 3 + 0];
        if (wg == 0ull) continue;                          /* EP: expert not owned by this rank */
        const unsigned char* Wg = (const unsigned char*)(size_t)wg;
        const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
        const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
        const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
        const unsigned nrow = (n >> 7) * KB;
        const float g = wave_dot_fp8_blk(x, Wg + (size_t)n * H, Sg + nrow, H, lane);
        const float u = wave_dot_fp8_blk(x, Wu + (size_t)n * H, Su + nrow, H, lane);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(moe_act(g, act) * u);
    }
#else
    for (unsigned slot = 0; slot < k; slot++)
        d_moe_expert_glu_fp8_blk(fu, x, table, wtab, stab, slot, I_moe, H, n_exp, act, slice, nblk);
#endif
}

__device__ void d_moe_group_down_fp8_blk(float* part, const bf16* fu, const unsigned char* table,
                                         const unsigned long long* wtab,
                                         const unsigned long long* stab, unsigned k, unsigned H,
                                         unsigned I_moe, unsigned n_exp, unsigned slice,
                                         unsigned nblk) {
#if PLOW_MOE_GROUP_FLAT
    const unsigned KB = (I_moe + 127u) >> 7;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    const unsigned total = k * H;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += wstride) {
        const unsigned slot = f / H;
        const unsigned h = f - slot * H;
        const unsigned eid = moe_slot_expert(table, slot);
        float* part_slot = part + (size_t)slot * H;
        if (eid >= n_exp || wtab[(size_t)eid * 3 + 2] == 0ull) { /* skip: deterministic zero partial */
            if (lane == 0) part_slot[h] = 0.0f;
            continue;
        }
        const float gate = moe_slot_gate(table, slot);
        const unsigned char* Wd = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 2];
        const float* Sd = (const float*)(size_t)stab[(size_t)eid * 3 + 2];
        const bf16* fu_slot = fu + (size_t)slot * I_moe;
        const float y = wave_dot_fp8_blk(fu_slot, Wd + (size_t)h * I_moe, Sd + (size_t)(h >> 7) * KB,
                                         I_moe, lane);
        if (lane == 0) part_slot[h] = gate * y;
    }
#else
    for (unsigned slot = 0; slot < k; slot++)
        d_moe_expert_down_fp8_blk(part, fu, table, wtab, stab, slot, H, I_moe, n_exp, slice, nblk);
#endif
}

/* --- GROUPED block-fp8 experts, fp8->bf16 MFMA gate/up (PLOW_MOE_MFMA=1) — NEGATIVE EXPERIMENT ---
 * The matrix-core lever (glm-kernel-tricks.md §3): the M=1 token is padded to the 16-row MFMA tile
 * and the top-k experts' gate+up run as fused 16x16x32 bf16 MFMA GEMMs (weights dequantized fp8->bf16
 * per fragment and folded with the per-128-block scale), instead of the wave-per-output fdot2 VALU.
 * w8a16 (activation stays bf16 — no per-token quant), so it is meant to be bit-CLOSE to fdot2.
 *
 * MEASURED VERDICT (EP2+grouped, MI350X, 6L): SLOWER (4.746 vs 3.855 ms/tok, +23%) AND currently
 * INCORRECT (decode tokens diverge — a fragment/layout bug I did not root-cause). The perf verdict is
 * decisive regardless of the bug: at M=1 only 1 of the 16 padded MFMA rows is real, so the XDL pass
 * reads the SAME expert-weight bytes as fdot2 (decode is weight-bandwidth-bound — MEMORY.md "decode at
 * ceiling") while adding matrix-pipe + accumulator-register pressure that drops occupancy. The
 * AITER path wins at M>32 (real rows fill the 16-tile) and uses a8w8 (fp8 activation via a per-token
 * quant + 16x16x128 f8f6f4 scaled-MFMA), NOT this w8a16 16x16x32. Kept FLAG-GATED (default 0) as the
 * scaffold for a future a8w8 attempt; the shipping grouped path stays fdot2. ONE WAVE handles one
 * (slot, 16-channel N-tile); A holds x in row 0 (lanes lane%16==0), zero-pads rows 1..15; the m=0 acc
 * slot (lanes 0..15) carries the 16 outputs. gate/up share the A fragment (x read once).
 *
 * 16x16x32 bf16 MFMA lane layout (mirrors gemm_mfma16_poc.hip):
 *   A[16][32]: lane l -> m=l%16, k=8*(l/16)+j (j=0..7)   B[32][16]: lane l -> n=l%16, k=8*(l/16)+j
 *   D[16][16]: lane l -> n=l%16, m=4*(l/16)+e (e=0..3)   => m=0 lives in lanes 0..15, e=0. */
__device__ __forceinline__ bf16x8 moe_scaled_fp8x8(const unsigned char* p, float sc) {
    const bf16v8 d = fp8v8_to_bf16v8(ld_glob_fp8v8(p)); /* exact e4m3->bf16 decode */
    bf16x8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = (bf16_t)f2bf(bf2f(d[i]) * sc); /* fold per-128-block scale */
    return out;
}
__device__ void d_moe_group_glu_mfma(bf16* fu, const bf16* x, const unsigned char* table,
                                     const unsigned long long* wtab, const unsigned long long* stab,
                                     unsigned k, unsigned I_moe, unsigned H, unsigned n_exp,
                                     unsigned act, unsigned slice, unsigned nblk) {
    typedef float f32x4 __attribute__((ext_vector_type(4)));
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    const unsigned frow = lane & 15u;   /* n within the 16-tile (== the acc m/n row selector) */
    const unsigned kgrp = lane >> 4;    /* 0..3 -> this lane's k-offset group (8*kgrp) */
    const unsigned KB = (H + 127u) >> 7;
    const unsigned ntile = I_moe >> 4;  /* 16-channel N-tiles per expert */
    const unsigned total = k * ntile;
    for (unsigned t = slice * PLOW_WAVES + wave; t < total; t += wstride) {
        const unsigned slot = t / ntile;
        const unsigned n0 = (t - slot * ntile) << 4;      /* first output channel of this tile */
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;                        /* sentinel/EP-non-local: leave fu unwritten */
        const unsigned long long wgb = wtab[(size_t)eid * 3 + 0];
        if (wgb == 0ull) continue;                          /* EP: expert not owned by this rank */
        const unsigned char* Wg = (const unsigned char*)(size_t)wgb;
        const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
        const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
        const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
        const unsigned srow = (n0 >> 7) * KB; /* all 16 channels share the (n0>>7) block-scale row */
        const unsigned n = n0 + frow;         /* this lane's weight output row */
        f32x4 accg = {0, 0, 0, 0}, accu = {0, 0, 0, 0};
        for (unsigned s = 0; s < H; s += 32) {
            const unsigned kk = s + 8u * kgrp;             /* this lane's 8 k-values */
            const float sg = Sg[srow + (s >> 7)];          /* 32-aligned frag is within one 128-block */
            const float su = Su[srow + (s >> 7)];
            bf16x8 af;
            if (frow == 0) af = *(const bf16x8*)(x + kk);  /* m=0 row = x; m=1..15 pad */
            else af = (bf16x8)((bf16_t)0);
            const bf16x8 bfg = moe_scaled_fp8x8(Wg + (size_t)n * H + kk, sg);
            const bf16x8 bfu = moe_scaled_fp8x8(Wu + (size_t)n * H + kk, su);
            accg = __builtin_amdgcn_mfma_f32_16x16x32_bf16(af, bfg, accg, 0, 0, 0);
            accu = __builtin_amdgcn_mfma_f32_16x16x32_bf16(af, bfu, accu, 0, 0, 0);
        }
        if (kgrp == 0) /* lanes 0..15 hold m=0 (e=0) for the 16 output channels */
            fu[(size_t)slot * I_moe + n0 + frow] = f2bf(moe_act(accg[0], act) * accu[0]);
    }
}

/* --- DENSE MLP GATE/UP, block-fp8 (intended opcode PLOW_DOP_DENSE_GLU_FP8_BLK = 47) -----------
 * GLM-5.2 dense layers 0-2 (first_k_dense_replace=3) run a straight SwiGLU MLP on NAMED weights —
 * NO routing/expert-table indirection, NO sentinel skip, NO gate multiply. This is the block-fp8
 * twin of gemv_glu_rows_fp8 (per-channel op 31): fused gate+up+SwiGLU, direct weight+scale pointers.
 *   fu[n] = act(gate_n·x) * (up_n·x),   n in [0,N)   (N = intermediate, K = hidden)
 * Wg,Wu are e4m3 [N][K]; Sg,Su are the block-scale grids [ceil(N/128)][ceil(K/128)] f32 (row-major),
 * read directly (weight_scale_inv). Wave-per-output + wave_dot_fp8_blk (scaled-cvt decode + separate
 * per-128-block f32 multiply), non-fused two-pass for interp-megakernel safety (see expert glu note).
 * The DENSE DOWN projection (N_dense->H) is a plain block-fp8 GEMV = the existing GEMV_FP8_BLK op
 * (gemv_rows_fp8_blk) — no separate dense-down op is needed; the emitter emits GEMV_FP8_BLK for it. */
__device__ void d_dense_glu_fp8_blk(bf16* fu, const bf16* x, const unsigned char* Wg,
                                    const unsigned char* Wu, const float* Sg, const float* Su,
                                    unsigned N, unsigned K, unsigned act, unsigned slice,
                                    unsigned nblk) {
    const unsigned KB = (K + 127u) >> 7;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    auto* const fg = as_glob(fu);
    for (unsigned n = slice * PLOW_WAVES + wave; n < N; n += wstride) {
        const unsigned nrow = (n >> 7) * KB;
        const float g = wave_dot_fp8_blk(x, Wg + (size_t)n * K, Sg + nrow, K, lane);
        const float u = wave_dot_fp8_blk(x, Wu + (size_t)n * K, Su + nrow, K, lane);
        if (lane == 0) fg[n] = f2bf(moe_act(g, act) * u);
    }
}

/* --- kernel 2a: EXPERT GATE/UP (PLOW_DOP_MOE_EXPERT_GLU, moe-ep-kernels §3a) --------------
 * fu[slot] = act(gate·x) * (up·x), streaming ONLY the chosen expert's weights. Sentinel skip.
 * Weight bases resolved from expert_weight_table[expert_id] (two-level indirection). */
__device__ void d_moe_expert_glu(bf16* fu, const bf16* x, const unsigned char* table,
                                 const unsigned long long* wtab, unsigned slot, unsigned I_moe,
                                 unsigned H, unsigned n_exp, unsigned act, unsigned slice,
                                 unsigned nblk) {
    const unsigned eid = moe_slot_expert(table, slot);
    if (eid >= n_exp) return; /* sentinel: skip, stream zero bytes — interp still signals */
    if (wtab[(size_t)eid * 3 + 0] == 0ull) return; /* EP: expert not owned by this rank — skip */
    const bf16* Wg = (const bf16*)(size_t)wtab[(size_t)eid * 3 + 0];
    const bf16* Wu = (const bf16*)(size_t)wtab[(size_t)eid * 3 + 1];
    bf16* fu_slot = fu + (size_t)slot * I_moe;
    const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;
    for (unsigned n = gid; n < I_moe; n += stride) {
        float g = moe_dot_bf16(x, Wg + (size_t)n * H, H);
        float u = moe_dot_bf16(x, Wu + (size_t)n * H, H);
        fu_slot[n] = f2bf(moe_act(g, act) * u);
    }
}

/* --- kernel 2b: EXPERT DOWN (PLOW_DOP_MOE_EXPERT_DOWN, moe-ep-kernels §3a) -----------------
 * part[slot] = gate · (down·fu[slot]), f32 partial. Sentinel skip zeroes the partial so the
 * combine sums a deterministic zero. */
__device__ void d_moe_expert_down(float* part, const bf16* fu, const unsigned char* table,
                                  const unsigned long long* wtab, unsigned slot, unsigned H,
                                  unsigned I_moe, unsigned n_exp, unsigned slice, unsigned nblk) {
    const unsigned eid = moe_slot_expert(table, slot);
    float* part_slot = part + (size_t)slot * H;
    const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;
    if (eid >= n_exp || wtab[(size_t)eid * 3 + 2] == 0ull) { /* skip (sentinel/EP-non-local): zero */
        for (unsigned h = gid; h < H; h += stride) part_slot[h] = 0.0f;
        return;
    }
    const float gate = moe_slot_gate(table, slot);
    const bf16* Wd = (const bf16*)(size_t)wtab[(size_t)eid * 3 + 2];
    const bf16* fu_slot = fu + (size_t)slot * I_moe;
    for (unsigned h = gid; h < H; h += stride) {
        float y = moe_dot_bf16(fu_slot, Wd + (size_t)h * I_moe, I_moe);
        part_slot[h] = gate * y; /* f32 partial into the combine */
    }
}

/* --- kernel 3: COMBINE (PLOW_DOP_MOE_COMBINE, moe-ep-kernels §3b) --------------------------
 * out = residual + shared + Σ_{j=0..k-1} part[j], f32 accumulate in FIXED slot order, bf16 out.
 * Deterministic regardless of which expert finished first. shared==nullptr for 0-shared. */
__device__ void d_moe_combine(bf16* out, const bf16* residual, const bf16* shared,
                              const float* part, unsigned H, unsigned k, unsigned slice,
                              unsigned nblk) {
    const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;
    for (unsigned h = gid; h < H; h += stride) {
        float acc = bf2f(residual[h]);
        if (shared) acc += bf2f(shared[h]);
        for (unsigned j = 0; j < k; j++) acc += part[(size_t)j * H + h];
        out[h] = f2bf(acc);
    }
}

#endif /* PLOW_OP_MOE_H */
