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

/* THE ONLY HARD BOUND IN THIS FILE IS top_k, AND IT IS 16 (was 8 until Kimi-K3).
 *
 * Both routers select into `unsigned winner[MAX]` / `float gate[MAX]`, and `d_moe_router_topk`
 * additionally writes `wl[rank]` into an LDS carve of exactly MAX entries, from loops that ran to
 * the caller's k with no check. The clamp that used to guard this sat AFTER the LDS write and only
 * covered the stack arrays; see `moe_bound_topk` below for what that actually did at k=16.
 *
 * RAISED TO 16 for Kimi-K3 (top-16 of 896). Measured with
 * `hipcc --offload-arch=gfx950 -O3 -DPLOW_BUCKET_DECODE=1 -Rpass-analysis=kernel-resource-usage`:
 * 8 vs 16 give the IDENTICAL decode object budget — TotalSGPRs 106, VGPRs 248, occupancy 2
 * waves/SIMD, SGPR spill 80, VGPR spill 0, LDS 147464 B. The arrays are small enough that the
 * raise is free, and it is inert at k <= 8: `moe_bound_topk` returns immediately, every loop still
 * runs to k, and the only layout change is `moe_group_mask`'s scratch base moving 32 B further
 * down a fixed-size arena. So every shipping GLM / DeepSeek / Qwen / Mixtral / Gemma packet is
 * byte-identical.
 *
 * NOT YET EXECUTED ON HARDWARE ABOVE k=8. The k<=16 path is correct by construction (every write
 * is now in bounds and the renormalisation covers every selected slot) but no GPU has run it, so
 * treat the first top-16 bring-up as bring-up.
 *
 * n_exp, by contrast, is NOT bounded here — every n_exp-sized allocation is carved from the LDS
 * arena at runtime, not from a fixed array (PLOW_MOE_MAXE is an NVIDIA-only constant and has no
 * AMD counterpart). Measured against the interpreter's 147464 B arena:
 *
 *   n_exp   d_moe_router_topk   d_moe_align_pf   d_moe_router   headroom
 *     256          3104 B            2052 B          1024 B      144360 B
 *     384          4640 B            3076 B          1536 B      142824 B   <- Kimi K2.7
 *     512          6176 B            4100 B          2048 B      141288 B
 *
 * so Kimi's 384 experts sit at ~3% of the arena and the structural ceiling is ~n_exp 12000. The
 * packed selection key gives the id 20 bits (n_exp <= 1048575). Neither is close.
 *
 * THE BOUND IS ENFORCED IN TWO PLACES, and the compile-time one is the real gate:
 *   1. `devgen::MOE_MAX_TOPK` (crates/devgen/src/lib.rs) refuses to EMIT a packet whose top_k
 *      exceeds this number, naming the model and the limit. A drift test parses the `#define`
 *      below out of this file, so the two cannot disagree. Raise them together or not at all.
 *   2. `moe_bound_topk` below is the runtime backstop. It used to be a bare
 *      `if (k > MAX) k = MAX;` and that was NOT the "deterministic truncation" its comment
 *      claimed — see the note there. */
#define PLOW_MOE_MAX_TOPK 16u

/* The top-k backstop, made SAFE and DETERMINISTIC. Returns the number of slots the selection can
 * actually fill; the caller must use the RETURN VALUE everywhere, including the rank pass.
 *
 * What the old bare clamp did at k > the bound, all three of which were invisible:
 *   1. `d_moe_router_topk`'s rank pass ran on the UNCLAMPED k, so `wl[rank] = e` wrote k u32
 *      into a MAX-u32 LDS carve and straight through `wl + PLOW_MOE_MAX_TOPK` — the base of
 *      `moe_group_mask`'s scratch. It survived only because the mask runs BEFORE the rank pass
 *      and n_group<=1 makes it a no-op. At DeepSeek's n_group=8 the ordering is the only thing
 *      standing between that and corrupted group keys.
 *   2. Table slots above the bound were never written, but every expert body loops to the
 *      packet's top_k operand — which is UNCLAMPED. So those slots were read out of uninitialised
 *      scratch: any garbage id below n_exp is a real expert summed with a garbage gate.
 *   3. The renorm denominator covered only the kept gates, so the survivors were rescaled to
 *      sum to 1. Plausible logits, wrong model.
 *
 * Filling the unreachable slots with the skip sentinel (id 0xFFFFFFFF, gate 0) is the house idiom
 * — `eid >= n_exp` is already the "skip this slot" test in eight expert bodies (op_moe.h:442,
 * :483, :551, :586, :652, :714, :738, :969) — and it turns (2) from random experts into an
 * actual deterministic truncation. It does NOT make the model correct: a top-k model routed to
 * fewer experts is still wrong, and that is what gate 1 above exists to prevent. This function only
 * guarantees the wrongness is memory-safe, reproducible, and the same on every rank.
 *
 * NOT a `__trap()`: this interpreter's dispatch `default:` is a deliberate silent NOP
 * (interp.hip:895) and introducing the first device-side trap into the megakernel is the kernel
 * owner's call, not a backstop's. The compile-time refusal is where "loud" belongs. */
__device__ __forceinline__ unsigned moe_bound_topk(unsigned char* table, unsigned k) {
    if (k <= PLOW_MOE_MAX_TOPK) return k; /* the only path any shipping model takes */
    for (unsigned j = PLOW_MOE_MAX_TOPK; j < k; j++) {
        *(unsigned*)(table + (size_t)j * 8) = 0xFFFFFFFFu; /* >= n_exp => every body skips it */
        *(float*)(table + (size_t)j * 8 + 4) = 0.0f;
    }
    return PLOW_MOE_MAX_TOPK;
}

/* A4W4 grouped MoE prefill: MXFP4 on BOTH operands through the matrix core, instead of
 * dequantizing weights to bf16 (w4a16) or carrying block-fp8 weights against bf16 activations
 * (w8a16). COMPILE-TIME, like every other precision axis here, and for the reason interp.hip
 * states: the interpreter inlines every arm, so a RUNTIME encoding switch makes the register
 * allocator budget for the worst body and then every body spills. A run is all-fp4 or it is
 * not; it is never both. */
#ifndef PLOW_MOE_PF_A4W4
#define PLOW_MOE_PF_A4W4 0
#endif
/* i[3] on ops 85/86 selects the WEIGHT ENCODING at runtime, so switching a model between
 * precisions is a field change in the packet and not a re-emit (the emitter's contract). The
 * A4W4 body is still gated by the COMPILE-TIME flag above -- see the measurement in the commit
 * message: carrying all three bodies costs nothing at occ-2, but an object built for a bf16
 * model should not have to contain an fp4 GEMM it can never reach. */
#define PLOW_MOE_ENC_BF16   0u
#define PLOW_MOE_ENC_FP8BLK 1u
#define PLOW_MOE_ENC_MXFP4  2u

/* GLU activation, matching op_elementwise.h / the Rust reference. act: 0 = gelu_tanh, 1 = silu. */
#define PLOW_MOE_ACT_GELU_TANH 0u
#define PLOW_MOE_ACT_SILU      1u
/* Kimi-K3's `situ`, and it is deliberately NOT a third `moe_act` value: `moe_act` returns the GATE
 * transform alone, and situ transforms the UP branch too. A third code inside `moe_act` would
 * compile, would leave `up` un-clipped, and would be a growing-with-the-tail error on every one of
 * K3's 896 routed experts — plausible output, wrong model. The pair form below is the only entry
 * point that can express it, so the epilogue calls THAT and never `moe_act` directly. */
#define PLOW_MOE_ACT_SITU      2u

__device__ __forceinline__ float moe_act(float x, unsigned act) {
    if (act == PLOW_MOE_ACT_SILU) return x / (1.0f + expf(-x)); /* silu (SwiGLU — GLM) */
    /* POISON, on purpose. `situ` cannot be expressed by a gate-only transform, so any caller that
     * reaches here with it is a GLU epilogue that has not been converted to `moe_glu` — and the
     * default `else` below would hand it gelu_tanh(g)*u, which is finite, plausible, and the wrong
     * model. This interpreter's dispatch `default:` is a silent NOP and there is no device trap,
     * so a NaN is the loudest primitive available: it propagates to the residual immediately.
     * Only the epilogues converted to `moe_glu` may see act = 2. */
    if (act == PLOW_MOE_ACT_SITU) return __builtin_nanf("");
    const float k = 0.7978845608028654f; /* gelu_tanh */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* The GLU epilogue as a PAIR: `A(gate) * B(up)`. For every activation but situ, `B` is the
 * identity and this is byte-identical to the `moe_act(g, act) * u` it replaces. */
__device__ __forceinline__ float moe_glu(float g, float u, unsigned act, float beta, float lbeta) {
    if (act == PLOW_MOE_ACT_SITU) return k3_situ_gate(g, beta) * k3_situ_up(u, lbeta);
    return moe_act(g, act) * u;
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
        unsigned winner[PLOW_MOE_MAX_TOPK]; /* GLM/DS/Qwen/Mixtral top-8; Kimi-K3 top-16 */
        float gate[PLOW_MOE_MAX_TOPK];
        k = moe_bound_topk(table, k); /* safe backstop; the real gate is devgen::MOE_MAX_TOPK */
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

/* --- GROUP-LIMITED TOP-K (DeepSeek-V3 / Kimi K2 `n_group` + `topk_group`) ------------------
 * THIS WAS MISSING AND IT SILENTLY ROUTED TO THE WRONG EXPERTS. The router's own header said
 * "group-limited" and the body applied a FLAT top-k over all n_exp. At n_group <= 1 the two are
 * identical, which is exactly why GLM (n_group=1) matched its oracle and this went unnoticed —
 * and why it is a bug you cannot find by testing the model you have. At DeepSeek-V3's 8 groups
 * / topk_group 4 it selects a DIFFERENT EXPERT SET: fluent output, wrong model.
 *
 * The rule (noaux_tc): experts are partitioned into n_group contiguous groups; each group is
 * scored by the SUM OF ITS TOP-2 BIASED scores; the top `topk_group` groups are selected; and
 * the top-k then runs only over experts inside them.
 *
 * Implemented as a MASK on the packed selection key rather than as a second selection pass, so
 * the existing one-pass all-pairs rank below is completely unchanged and stays a single fused
 * kernel — the whole point of that rewrite was to get the router off ~24 dependent barriers,
 * and a two-pass group filter would hand most of that back. A masked expert keeps its unique id
 * bits and loses its score bits, so it sorts below every live expert and ranks stay unique.
 *
 * Cost: n_group serial scans of n_exp/n_group (one thread per group) + an all-pairs rank over
 * n_group. At DeepSeek's 8x48 that is 48 iterations on 8 threads and 64 comparisons — noise
 * against the 256 expf the sigmoid already pays.
 *
 * `sc` is the unbiased score array and `bias` the selection bias (either may be null-biased);
 * the biased score is recomputed here rather than staged, which costs one add and saves an
 * n_exp-sized LDS array. `keys` is masked in place. */
#define PLOW_MOE_MAX_GROUPS 64u
__device__ __forceinline__ void moe_group_mask(unsigned long long* keys, const float* sc,
                                               const float* bias, unsigned n_exp,
                                               unsigned n_group, unsigned topk_group,
                                               unsigned long long* gk, unsigned char* gsel) {
    if (n_group <= 1u || topk_group == 0u || topk_group >= n_group) return;
    if (n_group > PLOW_MOE_MAX_GROUPS) return; /* backstop; no real config is close */
    /* NON-DIVISIBLE PARTITIONS, the third backstop. The mask below indexes `gsel[e / gsz]` for
     * every e < n_exp, and `gsel` is n_group bytes. With gsz = n_exp / n_group truncating, the
     * remainder experts e >= n_group*gsz give e / gsz == n_group — one byte PAST the array, read
     * out of uninitialised LDS, so a whole group's worth of experts is kept or dropped on a coin
     * flip. `crates/devgen/src/mla.rs:202` already refuses such a config at emit, but this header
     * states plainly that its guards are the RUNTIME backstop for exactly this class (see the
     * `moe_bound_topk` note above), and this one was missing: a packet reaching the kernel by any
     * other route — a hand-built blob, a future emitter, a raised MAX_GROUPS — had nothing here.
     * Returning leaves `keys` untouched, i.e. flat top-k, which is the pre-grouping behaviour. */
    if (n_exp % n_group) return;
    const unsigned gsz = n_exp / n_group;
    const unsigned tid = threadIdx.x;
    for (unsigned g = tid; g < n_group; g += PLOW_THREADS) {
        float m1 = -3.0e38f, m2 = -3.0e38f; /* top-2 of the group's biased scores */
        for (unsigned i = 0; i < gsz; i++) {
            const unsigned e_ = g * gsz + i;
            const float s = sc[e_] + (bias ? bias[e_] : 0.0f);
            if (s > m1) { m2 = m1; m1 = s; } else if (s > m2) m2 = s;
        }
        const float gs = m1 + m2;
        unsigned sb;
        __builtin_memcpy(&sb, &gs, 4);
        sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u); /* same monotone f32->u32 as above */
        /* Pack the group id low so ties break to the LOWEST id, matching the expert-level rule. */
        gk[g] = ((unsigned long long)sb << 8) | (unsigned long long)((n_group - 1u - g) & 0xFFu);
    }
    __syncthreads();
    for (unsigned g = tid; g < n_group; g += PLOW_THREADS) {
        const unsigned long long my = gk[g];
        unsigned rank = 0;
        for (unsigned f = 0; f < n_group; f++) rank += (gk[f] > my);
        gsel[g] = (unsigned char)(rank < topk_group);
    }
    __syncthreads();
    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS)
        if (!gsel[e / gsz]) keys[e] = (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
    __syncthreads();
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
                                  unsigned slice, unsigned nblk, float* lds,
                                  unsigned n_group = 0, unsigned topk_group = 0) {
    (void)nblk;
    if (slice != 0) return; /* emitted on 1 CU; guard is belt-and-braces */
    const bool sigmoid = (flags & 1u) != 0;
    const bool norm_topk = (flags & 2u) != 0;
    const unsigned tid = threadIdx.x;
    /* Bound k HERE, before anything uses it. The rank pass below writes `wl[rank]` for every
     * rank < k, so a k past the bound overruns the `wl` carve — the clamp used to sit after that
     * loop and therefore protected only the gate array, not the LDS. `if (k <= 8) return k` makes
     * this inert (and the emit byte-identical) for every shipping model. */
    if (tid == 0) (void)moe_bound_topk(table, k); /* sentinel-fill slots the bound cannot reach */
    if (k > PLOW_MOE_MAX_TOPK) k = PLOW_MOE_MAX_TOPK;
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

    /* GROUP-LIMITED masking, before the rank pass so the pass itself is untouched. Inert at
     * n_group <= 1, which is why every existing GLM/Qwen/Mixtral packet stays bit-identical. */
    moe_group_mask(keys, lds, bias, n_exp, n_group, topk_group,
                   (unsigned long long*)(wl + PLOW_MOE_MAX_TOPK),
                   (unsigned char*)((unsigned long long*)(wl + PLOW_MOE_MAX_TOPK) +
                                    PLOW_MOE_MAX_GROUPS));

    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) {
        const unsigned long long myk = keys[e];
        unsigned rank = 0;
#pragma unroll 32
        for (unsigned f = 0; f < n_exp; f++) rank += (keys[f] > myk);
        if (rank < k) wl[rank] = e; /* winner of rank `rank` (rank 0 == highest key) */
    }
    __syncthreads();

    if (tid == 0) {
        float gate[PLOW_MOE_MAX_TOPK]; /* k is already bounded at the top of the function */
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

/* MXFP4 (OCP e2m1 + one E8M0 scale per 32) wave-reduced dot of ONE output channel — the fp4
 * twin of wave_dot_fp8_blk, for the DECODE experts. [MXFP4-DECODE]
 *
 * NOT the A4W4 matrix-core path the PREFILL grouped GEMM uses, and deliberately so: at M=1 there
 * is one activation row, so a 32x32 MFMA would be 1/32 utilised and the op is weight-bandwidth
 * bound anyway. w4a16 through fdot2 is the right shape here, exactly as d_gemv_mxfp4 already
 * concluded for the dense projections.
 *
 * THE MX SCALE FOLDS INTO THE CONVERT EXACTLY, which is what makes this cheaper than the
 * block-fp8 twin rather than merely narrower. E8M0 is a bare power-of-two exponent, so
 * v_cvt_scalef32_pk_bf16_fp4's scalef32 operand represents it with no error at all — there is
 * no separate scale multiply and nothing in the epilogue. (The block-fp8 path CANNOT do this:
 * DeepSeek/GLM block scales are arbitrary f32 and the hardware would silently discard the
 * mantissa — amd_common.h measured ~22% error. Same instruction, opposite conclusion, because
 * the formats differ.)
 *
 * A lane owns 32 fp4 = 16 bytes = EXACTLY one MX block, so one load consumes one scale byte and
 * no scale ever varies within a lane's fragment. K is a multiple of 128 for every real expert
 * (Kimi H=7168 I=2048, GLM H=6144 I=2048), so the k+32<=K guard is exact, not a clamp. */
__device__ __forceinline__ float wave_dot_mxfp4(const bf16* x, const unsigned char* Wrow,
                                                const unsigned char* srow, unsigned K,
                                                unsigned lane) {
    const unsigned step = PLOW_WAVE * 32; /* 64 lanes x 32 fp4 = 2048 K per pass */
    const unsigned nchunk = (K + step - 1) / step;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * step + lane * 32;
        if (k + 32u <= K) {
            const fp4v32 w = *(const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)(Wrow + (k >> 1));
            bf16v8 a, b, cc, d;
            fp4_to_bf16v8x4(w, e8m0_to_f32(srow[k >> 5]), a, b, cc, d);
            acc = dot8(a, ld_glob8(x + k), acc);
            acc = dot8(b, ld_glob8(x + k + 8), acc);
            acc = dot8(cc, ld_glob8(x + k + 16), acc);
            acc = dot8(d, ld_glob8(x + k + 24), acc);
        }
    }
    return wave_sum(acc);
}

/* ONE quantized-expert dot, encoding selected at runtime. PLOW_MOE_ENC_* as on ops 85/86.
 * `srow_f` is the block-fp8 scale row (f32), `srow_b` the MXFP4 E8M0 row; the caller passes
 * whichever its encoding uses and the other is ignored. */
__device__ __forceinline__ float wave_dot_enc(unsigned enc, const bf16* x,
                                              const unsigned char* Wrow, const float* srow_f,
                                              const unsigned char* srow_b, unsigned K,
                                              unsigned lane) {
    if (enc == PLOW_MOE_ENC_MXFP4) return wave_dot_mxfp4(x, Wrow, srow_b, K, lane);
    return wave_dot_fp8_blk(x, Wrow, srow_f, K, lane);
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
                                         unsigned slice, unsigned nblk, unsigned enc, float beta,
                                         float lbeta) {
    const unsigned eid = moe_slot_expert(table, slot);
    if (eid >= n_exp) return; /* sentinel: skip, stream zero bytes — interp still signals */
    const unsigned long long wg_base = wtab[(size_t)eid * 3 + 0];
    if (wg_base == 0ull) return; /* EP: expert not owned by this rank (null base) — skip */
    const unsigned char* Wg = (const unsigned char*)(size_t)wg_base;
    const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
    const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
    const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
    const unsigned char* Bg = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 0];
    const unsigned char* Bu = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 1];
    /* Weight/scale row strides differ by encoding: block-fp8 is 1 byte/elt with a
     * [N/128][K/128] f32 grid; MXFP4 is 2 elts/byte with a [N][K/32] E8M0 row. */
    const bool mx = (enc == PLOW_MOE_ENC_MXFP4);
    const unsigned wstr = mx ? (H >> 1) : H;
    const unsigned KB = (H + 127u) >> 7;
    bf16* fu_slot = fu + (size_t)slot * I_moe;

    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    for (unsigned n = slice * PLOW_WAVES + wave; n < I_moe; n += wstride) {
        const unsigned nrow = (n >> 7) * KB;          /* block-fp8 scale row  */
        const size_t brow = (size_t)n * (H >> 5);     /* MXFP4 E8M0 scale row */
        const float g = wave_dot_enc(enc, x, Wg + (size_t)n * wstr, Sg + nrow, Bg + brow, H, lane);
        const float u = wave_dot_enc(enc, x, Wu + (size_t)n * wstr, Su + nrow, Bu + brow, H, lane);
        /* moe_glu, not moe_act: Kimi-K3's `situ` transforms the UP branch too, so the epilogue
         * shape is A(g)*B(u). For every other activation B is the identity and this is
         * byte-identical to what it replaces. */
        if (lane == 0) fu_slot[n] = f2bf(moe_glu(g, u, act, beta, lbeta));
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
                                          unsigned nblk, unsigned enc) {
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
    const unsigned char* Bd = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 2];
    const bool mx = (enc == PLOW_MOE_ENC_MXFP4);
    const unsigned wstr = mx ? (I_moe >> 1) : I_moe;
    const unsigned KB = (I_moe + 127u) >> 7;
    const bf16* fu_slot = fu + (size_t)slot * I_moe;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    for (unsigned h = slice * PLOW_WAVES + wave; h < H; h += wstride) {
        const float y = wave_dot_enc(enc, fu_slot, Wd + (size_t)h * wstr,
                                     Sd + (size_t)(h >> 7) * KB,
                                     Bd + (size_t)h * (I_moe >> 5), I_moe, lane);
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
                                        unsigned nblk, unsigned enc, float beta,
                                        float lbeta) {
#if PLOW_MOE_MFMA
    /* BLOCK-FP8 ONLY, and the drop is deliberate rather than an oversight — but it is a real
     * limitation, not a no-op: `d_moe_group_glu_mfma` has no MXFP4 arm, so under this non-default
     * axis an MXFP4 packet's gate/up would read fp4 nibbles as block-fp8. The DOWN twin and both
     * other GLU bodies honour `enc`. If this axis is ever turned on for a Kimi-K3 (MXFP4) build,
     * give the MFMA body an fp4 arm first; there is nothing at runtime that would say so. */
    (void)enc;
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
        const unsigned char* Bg = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 0];
        const unsigned char* Bu = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 1];
        /* ENCODING-AWARE, exactly as the wave-per-output body above and the DOWN twin below.
         * This body previously hard-coded `wave_dot_fp8_blk` at row stride H and dropped `enc` on
         * the floor, while its own down twin honoured it — so on an MXFP4 packet the gate/up GEMV
         * read fp4 nibbles as one byte per element with an E8M0 row reinterpreted as an f32 grid
         * (garbage activations, no fault) and only the down projection was right. The asymmetry
         * made it present as a gate/up numerics bug rather than a dropped operand. */
        const bool mx = (enc == PLOW_MOE_ENC_MXFP4);
        const unsigned wstr = mx ? (H >> 1) : H;
        const unsigned nrow = (n >> 7) * KB;      /* block-fp8 scale row  */
        const size_t brow = (size_t)n * (H >> 5); /* MXFP4 E8M0 scale row */
        const float g = wave_dot_enc(enc, x, Wg + (size_t)n * wstr, Sg + nrow, Bg + brow, H, lane);
        const float u = wave_dot_enc(enc, x, Wu + (size_t)n * wstr, Su + nrow, Bu + brow, H, lane);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(moe_glu(g, u, act, beta, lbeta));
    }
#else
    for (unsigned slot = 0; slot < k; slot++)
        d_moe_expert_glu_fp8_blk(fu, x, table, wtab, stab, slot, I_moe, H, n_exp, act, slice, nblk,
                                 enc, beta, lbeta);
#endif
}

__device__ void d_moe_group_down_fp8_blk(float* part, const bf16* fu, const unsigned char* table,
                                         const unsigned long long* wtab,
                                         const unsigned long long* stab, unsigned k, unsigned H,
                                         unsigned I_moe, unsigned n_exp, unsigned slice,
                                         unsigned nblk, unsigned enc) {
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
        const unsigned wstr = (enc == PLOW_MOE_ENC_MXFP4) ? (I_moe >> 1) : I_moe;
        const float y = wave_dot_enc(enc, fu_slot, Wd + (size_t)h * wstr,
                                     Sd + (size_t)(h >> 7) * KB,
                                     (const unsigned char*)Sd + (size_t)h * (I_moe >> 5), I_moe,
                                     lane);
        if (lane == 0) part_slot[h] = gate * y;
    }
#else
    for (unsigned slot = 0; slot < k; slot++)
        d_moe_expert_down_fp8_blk(part, fu, table, wtab, stab, slot, H, I_moe, n_exp, slice, nblk,
                                  enc);
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
        /* residual == nullptr is LEGAL and is what Kimi-K3's Stable LatentMoE needs: the routed
         * experts run at `routed_expert_hidden_size` (3584), so their combine has no hidden-width
         * residual to add — the residual add happens after the up-projection back to 7168. `shared`
         * was already optional; `residual` was not, and TEN() maps TENSOR_NONE to nullptr, so the
         * old code faulted rather than skipping. */
        float acc = residual ? bf2f(residual[h]) : 0.0f;
        if (shared) acc += bf2f(shared[h]);
        for (unsigned j = 0; j < k; j++) acc += part[(size_t)j * H + h];
        out[h] = f2bf(acc);
    }
}

/* ==========================================================================================
 * MoE PREFILL (T > 1) — the token-sorted grouped expert GEMM.       [KIMI/GLM/DEEPSEEK-PREFILL]
 *
 * Ops 83-87. Until these landed the AMD PREFILL bucket had NO MoE arm of any kind: ops 40-49
 * and 56 are all inside the decode branch of interp.hip, and every one of them is M=1
 * wave-per-output with no M parameter at all. So a Kimi K2.7 / GLM-5.2 / DeepSeek prefill was
 * attention-complete and FFN-incomplete — the MLA prefill arms (51/55) would run and the
 * expert packets behind them would fall through to `default:` and write nothing.
 *
 * WHY A GROUPED GEMM AND NOT A LOOP OVER THE DECODE OPS. The decode expert kernel streams one
 * expert's weights to serve ONE token. Replaying it T times reads each expert's weights once
 * per token that chose it. Sorting the T*k (token, expert) slots by expert instead lets one
 * pass over an expert's weights serve every token that routed to it — for Kimi at T=1024 that
 * is 8192 slots over 384 experts, so the weight stream drops ~21x. That is the whole op.
 *
 * AND THE WEIGHT STREAM IS THE COST, which is what sizes every decision below. Kimi at
 * T=1024: 2*T*k*3*I*H = 720 GFLOP of math against 384*3*I*H = 16.9 GB of fp8 weights, i.e.
 * ~0.7 ms of MFMA behind ~2.1 ms of HBM at peak. This kernel is memory-bound even at T=1024,
 * so it is deliberately NOT the deep-pipelined ping-pong d_gemm_t is: it double-buffers, keeps
 * the weight loads coalesced, and spends its register budget on correctness instead. There is
 * no point pipelining a matrix pipe that is already waiting on HBM.
 *
 * TILE. BM=64 x BN=256 x BK=64 on the standard 2x4 wave grid (SM=1, SN=2).
 * BM is SMALL on purpose and it is the one number worth re-tuning per model. Rows are padded
 * to a BM boundary PER EXPERT, and these MoEs are extremely sparse: Kimi routes 8 of 384, so
 * at T=1024 an expert holds ~21 rows on average. BM=128 (the NVIDIA PGM_BM) would pad that to
 * 128 and run 6x the MFMA rows; BM=64 costs 3x. The padding is pure MFMA waste and MFMA is the
 * cheap side of this kernel, so 3x is affordable — but on a denser MoE (or a longer chunk) BM
 * should go UP, and the align op writes the tile boundaries from this constant, so changing it
 * is a one-line change that stays consistent.
 *
 * BLOCK-FP8 IS PROMOTED, NOT FOLDED ON LOAD. GLM/DeepSeek/Kimi ship weight_block_size
 * [128,128] with ARBITRARY f32 scales. It is tempting to fold the scale into the fp8->bf16
 * convert on the way to LDS (one multiply, no extra registers) and it is wrong twice over:
 * v_cvt_scalef32_* takes an E8M0 (power-of-two) scale and silently discards the mantissa
 * (amd_common.h measured ~22% error on a real GLM scale), and even done in software it would
 * round an exact fp8 value through bf16 AFTER scaling and lose the precision the fp8 had. So
 * LDS holds the EXACT fp8->bf16 decode (e4m3 has 3 mantissa bits, bf16 has 7 — lossless), the
 * MFMA runs on that, and the f32 accumulator is multiplied by the block scale every 128 K and
 * promoted into a second accumulator. That is the same shape as the decode path's
 * `acc += dot8(...) * bs` in wave_dot_fp8_blk, so prefill and decode stay in one numeric
 * family. It costs one extra accumulator set (SM*SN*16 f32) and nothing else.
 *
 * BK=64 divides the 128-element scale block exactly, so a staged tile never straddles two
 * K-scale blocks and the promotion lands on a k-tile boundary. The N-scale block index is
 * per-lane CONSTANT: the 32x32 MFMA accumulator gives lane l the single output column
 * n = l%32 (amd_common.h), and n is the weight ROW here, so `(n>>7)` never varies across the
 * 16 accumulator elements a lane holds. No cross-lane scale shuffle exists in this kernel.
 *
 * W8A16, NOT W8A8. Activations stay bf16, exactly as the decode block-fp8 experts do. The 2x
 * fp8 MFMA needs both operands fp8, which would mean a per-token activation quant (QUANT_FP8)
 * and a second numeric family to validate; the weight stream — the term that actually costs —
 * is fp8 either way. Upgrading to w8a8 is the follow-up lever, not the blocker.
 * ========================================================================================== */

/* Rows are padded to this per expert; the align op and both GEMMs must agree. See the tile
 * note above for why it is 64 and when to raise it. */
#define MPF_BM 64
#define MPF_BN 256
#define MPF_BK 64
#define MPF_WM 2
#define MPF_WN 4
#define MPF_SM (MPF_BM / MPF_WM / MFMA_M) /* 1 */
#define MPF_SN (MPF_BN / MPF_WN / MFMA_N) /* 2 */
#define MPF_STRIDE MPF_BK
#define MPF_TILE ((MPF_BM + MPF_BN) * MPF_STRIDE)
/* Halves of LDS the grouped GEMM needs (double-buffered). 81920 B — comfortably inside the
 * interpreter's existing arena (147464 B), so this op does not move the union's high-water
 * mark and cannot change any other op's occupancy. */
#define MPF_LDS_HALVES (2 * MPF_TILE)
/* Same 16-byte-column XOR swizzle d_gemm_t uses, for the same reason: the compact BK-half row
 * stride is exactly 32 banks, so without it every row of a ds_read_b128 fragment collides. */
#define MPF_XORSWZ(row, off) ((off) ^ (((row) & (MPF_BK / 8 - 1)) << 3))

/* meta layout (int32), written by d_moe_align_pf and read by both grouped GEMMs:
 *   [0, n_exp)              rowoff    padded gathered-row start of expert e
 *   [n_exp, 2*n_exp)        cnt       live rows of expert e
 *   [2*n_exp, 3*n_exp + 1)  tilep     m-tile prefix; tilep[n_exp] == total m-tiles */
#define MPF_META_INTS(n_exp) (3u * (n_exp) + 1u)
/* Upper bound on padded gathered rows, which is what sizes fu_g / row_token / row_partidx /
 * row_gate. Every expert wastes at most MPF_BM-1 rows. */
#define MPF_MAX_ROWS(T, k, n_exp) ((T) * (k) + (n_exp) * (MPF_BM - 1u))

/* --- op 83: T-TOKEN ROUTER TAIL (PLOW_DOP_MOE_ROUTER_TOPK_PF) ------------------------------
 * Block-per-token loop of d_moe_router_topk. The decode router is already a whole-workgroup
 * kernel over one token's logit row, so the prefill form is that kernel with a token loop
 * around it and the table/logit rows advanced — which makes it BIT-IDENTICAL per token to the
 * decode router by construction, not by coincidence. The [T, n_exp] logit matrix itself is an
 * ordinary GEMM (op 8), already in the prefill bucket; only this tail was missing.
 *   t0=table([T*k] entries) t1=logit([T,n_exp] bf16) t3=bias   i1=n_exp i2=k i3=flags i4=T
 *   f0=route_scale */
__device__ void d_moe_router_topk_pf(unsigned char* table, const bf16* logit, const float* bias,
                                     unsigned n_exp, unsigned k, unsigned flags, float route_scale,
                                     unsigned T, unsigned slice, unsigned nblk, float* lds,
                                     unsigned n_group = 0, unsigned topk_group = 0) {
    for (unsigned tok = slice; tok < T; tok += nblk) {
        /* slice=0 so the callee's single-workgroup guard passes; this workgroup owns `tok`. */
        d_moe_router_topk(table + (size_t)tok * k * 8, logit + (size_t)tok * n_exp, bias, n_exp, k,
                          flags, route_scale, 0, 1, lds, n_group, topk_group);
        __syncthreads(); /* the callee reuses `lds` for scores/keys on the next token */
    }
}

/* --- op 84: ALIGN / SORT (PLOW_DOP_MOE_ALIGN_PF) -------------------------------------------
 * ONE workgroup. Histogram the T*k routing slots by expert, build a MPF_BM-padded prefix, and
 * scatter each live slot into its expert's contiguous gathered-row range. Pad rows are marked
 * PLOW_EXPERT_UNUSED so the A-gather zero-fills them and the DOWN scatter drops them.
 *
 * Single-workgroup and serial-prefix on thread 0 is not laziness: n_exp <= 512 and T*k is a few
 * thousand, so this is microseconds, and it must be ONE workgroup because the padded prefix is
 * a global scan. The 255 other CUs are gated behind it by the counter DAG exactly as they are
 * behind the decode router.
 *
 * The scatter uses an LDS atomic cursor, so the row ORDER within an expert is not deterministic
 * across runs. That is safe HERE and only here: row_partidx carries each row's destination in
 * part[T*k, H], so the DOWN op scatters to a fixed address regardless of which gathered row it
 * landed in, and the combine sums part in FIXED slot order. Nothing downstream depends on the
 * gathered order. (This is the one place a reader should stop and check — an order-dependent
 * epilogue here would be a nondeterminism bug that only shows under contention.)
 *   t0=meta(i32) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)
 *   i0=T i1=n_exp i2=k
 *
 * --- SYNTHETIC ROUTING: `table == nullptr` (t1 = PLOW_TENSOR_NONE) --------------------------
 * THE ONE PLACE the dense-FFN-as-1-expert construction exists. Do not replicate it elsewhere,
 * and do NOT mistake it for real routing — nothing is being routed.
 *
 * WHY IT EXISTS. GLM-5.2 / DeepSeek / Kimi run their first `first_k_dense_replace` layers
 * (3 / 3 / 1) as a DENSE block-fp8 SwiGLU MLP, not as MoE. Those layers' decode ops
 * (DENSE_GLU_FP8_BLK, GEMV_FP8_BLK) are M=1 wave-per-output, and the ISA has no T-row block-fp8
 * GEMM to lower them to at prefill — opcodes 8/14/15/20 are bf16 and 33-36 carry a PER-CHANNEL
 * fp8 scale, not the [128,128] `weight_scale_inv` grid these checkpoints use. That gap blocked
 * a whole-model prefill program outright: a prefill blob has to cover EVERY layer, so three
 * dense layers with no T-row arm meant no prefill blob for a 78-layer model.
 *
 * But this op and ops 85/86 ALREADY implement exactly the required kernel — a T-row block-fp8
 * GEMM against a [128,128] scale grid — for the routed-expert case. A dense FFN is that same
 * GEMM with the routing degenerated: ONE expert, every token assigned to it, gate == 1. So the
 * dense path reuses the grouped arms verbatim (`n_exp = 1`, `k = 1`) and this branch supplies
 * the degenerate routing the align op would otherwise have read out of a router's table. No new
 * opcode, no second MFMA body, and — measured on gfx950/ROCm 7.2.4 — no register cost at all:
 * the prefill object is 256 VGPR / occ 2 / spill 2 with and without the MoE prefill arms.
 *
 * The alternative was a real `GemmGluFp8Blk` + `GemmFp8Blk` opcode pair. HALF of that pair now
 * exists — `d_gemm_fp8_blk` (op 107, op_gemm.h) — and this construction deliberately still does
 * NOT use it. It was added for the case these grouped arms CANNOT serve: `o_proj` and the shared
 * expert, which have no expert tables, no gather/scatter row maps and no f32 `part` output, and
 * which are what `GLM_LINEAR_FP8` converts. A dense FFN is a genuine expert, so it keeps the
 * zero-cost reuse. There is still no `GemmGluFp8Blk`; the fused gate|up is emitted as two op-107
 * packets plus a `Glu`, exactly as the MXFP4 prefill arm unfuses for the same missing-fusion
 * reason.
 *
 * (The register claim above stands for op 107 as well, and it had to be checked rather than
 * assumed: an arbitrary-f32 block scale must be PROMOTED into a second accumulator, which doubles
 * a tile's accumulator cost, so op 107 carries ONE 128x128 rung — 32 + 32 = 64 registers, strictly
 * under the 256x256 bf16 arm's 128. Measured with it in: prefill 256 VGPR / 0 AGPR / occ 2 /
 * spill 2, unchanged.)
 *
 * The degenerate case is genuinely degenerate, which is why it is safe: with n_exp == 1 every
 * slot histograms to expert 0, so the padded prefix is the single range [0, ceil(T/MPF_BM)),
 * `row_partidx[pos] = s == token` makes the DOWN scatter the identity, and `gate == 1` makes it
 * unscaled. The atomic-cursor row order is still nondeterministic and still does not matter,
 * for the same reason it does not matter above. */
__device__ void d_moe_align_pf(int* meta, const unsigned char* table, unsigned* row_token,
                               unsigned* row_partidx, float* row_gate, unsigned T, unsigned n_exp,
                               unsigned k, unsigned slice, unsigned* lds) {
    if (slice != 0) return;
    /* Synthetic (dense-FFN) routing reads no table; see the header block above. Hoisted to a
     * uniform bool so the two slot loops below stay branch-free per lane. */
    const bool synth = (table == nullptr);
    const unsigned tid = threadIdx.x;
    unsigned* cnt = lds;             /* [n_exp] */
    unsigned* cur = lds + n_exp;     /* [n_exp] */
    unsigned* tot = cur + n_exp;     /* [1] total padded rows */

    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) cnt[e] = 0u;
    __syncthreads();

    const unsigned nslot = T * k;
    for (unsigned s = tid; s < nslot; s += PLOW_THREADS) {
        const unsigned eid = synth ? 0u : moe_slot_expert(table, s);
        if (eid < n_exp) __hip_atomic_fetch_add(&cnt[eid], 1u, __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    __syncthreads();

    if (tid == 0) {
        int* rowoff = meta;
        int* mcnt = meta + n_exp;
        int* tilep = meta + 2u * n_exp;
        unsigned tp = 0u;
        for (unsigned e = 0; e < n_exp; e++) {
            tilep[e] = (int)tp;
            rowoff[e] = (int)(tp * MPF_BM);
            mcnt[e] = (int)cnt[e];
            cur[e] = tp * MPF_BM;
            tp += (cnt[e] + MPF_BM - 1u) / MPF_BM;
        }
        tilep[n_exp] = (int)tp;
        *tot = tp * MPF_BM;
    }
    __syncthreads();

    const unsigned total_pad = *tot;
    for (unsigned r = tid; r < total_pad; r += PLOW_THREADS) {
        row_token[r] = PLOW_EXPERT_UNUSED;
        row_partidx[r] = PLOW_EXPERT_UNUSED;
        row_gate[r] = 0.0f;
    }
    __syncthreads();

    for (unsigned s = tid; s < nslot; s += PLOW_THREADS) {
        const unsigned eid = synth ? 0u : moe_slot_expert(table, s);
        if (eid >= n_exp) continue;
        const unsigned pos = __hip_atomic_fetch_add(&cur[eid], 1u, __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_WORKGROUP);
        row_token[pos] = s / k;  /* source token row of xn2 */
        row_partidx[pos] = s;    /* destination row of part[T*k, H] == token*k + slot */
        /* gate == 1 under synthetic routing: a dense FFN applies no routing weight, and the DOWN
         * op multiplies by this unconditionally. */
        row_gate[pos] = synth ? 1.0f : moe_slot_gate(table, s);
    }
}

/* Which expert owns m-tile `mt`. n_exp <= 512 and the scan is uniform across the workgroup, so
 * a linear walk costs a few scalar ops and no memory traffic worth naming. */
__device__ __forceinline__ unsigned mpf_expert_of_tile(const int* tilep, unsigned mt,
                                                       unsigned n_exp) {
    unsigned e = 0;
    while (e + 1u < n_exp && (unsigned)tilep[e + 1] <= mt) e++;
    return e;
}

/* Exact fp8 -> bf16 for 8 consecutive weight bytes. NO scale: the block scale is applied to the
 * f32 accumulator (see the promotion note in the family header), so this stays lossless. */
__device__ __forceinline__ bf16v8 mpf_ld_w8(const unsigned char* p) {
    return fp8v8_to_bf16v8(ld_glob_fp8v8(p));
}

/* --- ops 85 / 86: the GROUPED EXPERT GEMM -------------------------------------------------
 * ONE body, two modes, because they differ only in where A comes from and where C goes:
 *
 *   GLU=true  (op 85)  gate/up. A = xn2 rows GATHERED by row_token; B = the expert's gate and
 *                      up weights staged into the low/high halves of one BN tile so the SN axis
 *                      selects gate vs up (d_gemm_t's trick — both halves of an output element
 *                      land in the same lane, so the GLU epilogue needs no shuffle). A tile
 *                      therefore emits BN/2 = 128 fused columns. C = fu_g[row][I_moe], bf16.
 *   GLU=false (op 86)  down. A = fu_g rows, already contiguous per expert segment; B = the
 *                      expert's down weight; C = part[row_partidx[row]][H], f32, SCATTERED and
 *                      multiplied by that row's gate. Pad rows are dropped here.
 *
 * FP8=true takes the weight bases from wtab/stab (block-fp8); FP8=false reads bf16 weights from
 * wtab and ignores stab, so an all-bf16 model gets the same kernel with the dequant and the
 * promotion compiled out. */
template <bool FP8, bool GLU>
__device__ void d_moe_group_pf_t(void* __restrict__ Cout, const bf16* __restrict__ A,
                                 const unsigned long long* __restrict__ wtab,
                                 const unsigned long long* __restrict__ stab,
                                 const int* __restrict__ meta,
                                 const unsigned* __restrict__ row_token,
                                 const unsigned* __restrict__ row_partidx,
                                 const float* __restrict__ row_gate, unsigned N, unsigned K,
                                 unsigned n_exp, unsigned act, unsigned slice, unsigned nblk,
                                 bf16* lds) {
    constexpr int SM = MPF_SM, SN = MPF_SN;
    constexpr int APT = MPF_BM * MPF_BK / PLOW_THREADS; /* 8  halves/thread */
    constexpr int BPT = MPF_BN * MPF_BK / PLOW_THREADS; /* 32 halves/thread */
    constexpr int APASS = APT / 8, BPASS = BPT / 8;
    /* Output columns a tile emits: GLU fuses two B halves into one column block. */
    constexpr unsigned NB = GLU ? (MPF_BN / 2) : MPF_BN;

    const unsigned lane = threadIdx.x & 63u;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / MPF_WN, wn = wave % MPF_WN;
    const unsigned frow = mfma_frag_row(lane);

    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned total_tiles = (unsigned)tilep[n_exp];
    const unsigned tn = (N + NB - 1u) / NB;
    const unsigned n_tiles = total_tiles * tn;
    const unsigned NT = (K + MPF_BK - 1u) / MPF_BK; /* k-tiles */
    const unsigned KB = (K + 127u) >> 7;            /* scale blocks along K */

#define MPF_ASM(b) (lds + (b) * MPF_TILE)
#define MPF_BSM(b) (lds + (b) * MPF_TILE + MPF_BM * MPF_STRIDE)

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned mt = lin / tn, nt = lin % tn;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF_BM;
        const unsigned n0 = nt * NB;

        /* Weight + scale bases for THIS tile's expert. A null base is the EP "not my expert"
         * sentinel the decode ops already honour; skip rather than fault. */
        const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
        if (wb0 == 0ull) continue;
        const unsigned char* W0 = (const unsigned char*)(size_t)wb0;
        const unsigned char* W1 =
            GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : nullptr;
        const float* S0 = FP8 ? (const float*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)] : nullptr;
        const float* S1 = (FP8 && GLU) ? (const float*)(size_t)stab[(size_t)e * 3 + 1] : nullptr;

        f32x16 acc[SM][SN], accf[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) { acc[i][j] = (f32x16)(0.0f); accf[i][j] = (f32x16)(0.0f); }

        /* This lane's output column, constant for the whole tile — and therefore so is its
         * N-scale-block row. GLU: j picks gate/up at the SAME column. */
        const unsigned nn_lane = n0 + (GLU ? wn * MFMA_N : wn * (MPF_BN / MPF_WN)) + mfma_acc_n(lane);
        const unsigned nblk_row[SN] = {
            (nn_lane) >> 7,
            (GLU ? nn_lane : nn_lane + MFMA_N) >> 7,
        };

        __align__(16) bf16 ra[APT], rb[BPT];

/* Stage A: GLU gathers row_token[rowbase+r] out of xn2; DOWN reads fu_g contiguously from
 * rowbase. An UNUSED (pad) row zero-fills, which contributes nothing to any live output. */
#define MPF_FETCH_A(k0)                                                                        \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        const unsigned r = el / MPF_BK, kk = (k0) + (el % MPF_BK);                              \
        unsigned src;                                                                           \
        if constexpr (GLU) src = row_token[rowbase + r];                                        \
        else src = rowbase + r;                                                                 \
        if (src != PLOW_EXPERT_UNUSED)                                                          \
            *(bf16v8*)&ra[it * 8] = ld_glob8(as_glob(A) + (size_t)src * K + kk);                \
        else                                                                                    \
            _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;                   \
    }

/* Stage B: the expert's weight rows [n0, n0+NB) at k0. Under GLU the tile's low half is the
 * gate weight and its high half is the up weight, both at the same output rows. fp8 decodes
 * EXACTLY to bf16 here; the block scale is applied later, to the f32 accumulator. */
#define MPF_FETCH_B(k0)                                                                        \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                             \
        unsigned r = n0 + br;                                                                   \
        const unsigned char* wsrc = W0;                                                         \
        if constexpr (GLU) {                                                                    \
            const bool up = (br >= MPF_BN / 2);                                                 \
            wsrc = up ? W1 : W0;                                                                \
            r = n0 + (up ? br - MPF_BN / 2 : br);                                               \
        }                                                                                       \
        if (r < N) {                                                                            \
            if constexpr (FP8)                                                                  \
                *(bf16v8*)&rb[it * 8] = mpf_ld_w8(wsrc + (size_t)r * K + kk);                   \
            else                                                                                \
                *(bf16v8*)&rb[it * 8] =                                                         \
                    ld_glob8(as_glob((const bf16*)wsrc) + (size_t)r * K + kk);                  \
        } else                                                                                  \
            _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;                   \
    }

#define MPF_COMMIT(buf)                                                                        \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        __builtin_memcpy(&MPF_ASM(buf)[(el / MPF_BK) * MPF_STRIDE +                             \
                                       MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                   \
                         &ra[it * 8], 16);                                                      \
    }                                                                                           \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        __builtin_memcpy(&MPF_BSM(buf)[(el / MPF_BK) * MPF_STRIDE +                             \
                                       MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                   \
                         &rb[it * 8], 16);                                                      \
    }

        __syncthreads(); /* the previous tile's fragment readers must be done with the LDS */
        MPF_FETCH_A(0)
        MPF_FETCH_B(0)
        MPF_COMMIT(0)
        __syncthreads();

        unsigned buf = 0;
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1u) * MPF_BK;
            if (kn < K) { MPF_FETCH_A(kn) MPF_FETCH_B(kn) }

#pragma unroll
            for (int q = 0; q < MPF_BK / MFMA_K; q++) {
                bf16x8 af[SM], bfr[SN];
#pragma unroll
                for (int i = 0; i < SM; i++) {
                    const unsigned arow = wm * (MPF_BM / MPF_WM) + i * MFMA_M + frow;
                    __builtin_memcpy(&af[i],
                                     &MPF_ASM(buf)[arow * MPF_STRIDE +
                                                   MPF_XORSWZ(arow, mfma_frag_k(lane, q * MFMA_K))],
                                     16);
                }
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned brow =
                        (GLU ? j * (MPF_BN / 2) + wn * MFMA_N
                             : wn * (MPF_BN / MPF_WN) + j * MFMA_N) + frow;
                    __builtin_memcpy(&bfr[j],
                                     &MPF_BSM(buf)[brow * MPF_STRIDE +
                                                   MPF_XORSWZ(brow, mfma_frag_k(lane, q * MFMA_K))],
                                     16);
                }
                __builtin_amdgcn_s_setprio(1);
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int j = 0; j < SN; j++)
                        acc[i][j] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(af[i], bfr[j],
                                                                            acc[i][j], 0, 0, 0);
                __builtin_amdgcn_s_setprio(0);
            }

            /* PROMOTION. A 128-element K scale block is exactly two BK=64 tiles, so the scale
             * boundary always falls here and never inside an MFMA. */
            if constexpr (FP8) {
                if ((kt & 1u) == 1u || kt == NT - 1u) {
                    const unsigned kb = kt >> 1;
#pragma unroll
                    for (int i = 0; i < SM; i++)
#pragma unroll
                        for (int j = 0; j < SN; j++) {
                            const float* Sj = (GLU && j == 1) ? S1 : S0;
                            const float bs = Sj[(size_t)nblk_row[j] * KB + kb];
                            accf[i][j] += acc[i][j] * bs;
                            acc[i][j] = (f32x16)(0.0f);
                        }
                }
            }

            if (kn < K) { MPF_COMMIT(buf ^ 1) }
            __syncthreads();
            buf ^= 1;
        }
        if constexpr (!FP8) {
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) accf[i][j] = acc[i][j];
        }

        /* ---- epilogue */
        if constexpr (GLU) {
            /* accf[i][0] is gate and accf[i][1] is up FOR THE SAME ELEMENT, same lane. */
            auto* const fu = as_glob((bf16*)Cout);
            if (nn_lane < N) {
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        fu[(size_t)(rowbase + rr) * N + nn_lane] =
                            f2bf(moe_act(accf[i][0][el], act) * accf[i][1][el]);
                    }
            }
        } else {
            float* const part = (float*)Cout;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    if (nn >= N) continue;
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        const unsigned pidx = row_partidx[rowbase + rr];
                        if (pidx == PLOW_EXPERT_UNUSED) continue; /* pad row */
                        part[(size_t)pidx * N + nn] = row_gate[rowbase + rr] * accf[i][j][el];
                    }
                }
        }
        __syncthreads();
    }
#undef MPF_FETCH_A
#undef MPF_FETCH_B
#undef MPF_COMMIT
#undef MPF_ASM
#undef MPF_BSM
}

/* ==========================================================================================
 * A4W4 — the grouped expert GEMM with MXFP4 on BOTH operands.        [PLOW_MOE_PF_A4W4]
 *
 * The body above is w8a16/w16a16: fp8 or bf16 weights, bf16 activations, bf16 MFMA. That is
 * the same shape plow's mxfp4 path has everywhere — dequantize to bf16, feed a bf16 matrix
 * core — and it leaves two things on the floor. The matrix core runs at the bf16 rate when
 * CDNA4 can do 4-bit at 2x, and the activation still crosses HBM and LDS at 16 bits when it
 * could cross at 4. For the MoE prefill that activation is the LARGEST tensor in the layer
 * (the gathered [padded_rows][I_moe] intermediate), so it is not a rounding error.
 *
 * This body feeds v_mfma_scale_f32_32x32x64_f8f6f4 with cbsz=blgp=4 and REAL per-32 E8M0
 * scales, which is the instruction's intended use. Everything it relies on is measured in
 * runtime/tests/a4w4_gfx950_test.hip — read that before changing anything here, in particular:
 *
 *   THE HARDWARE APPLIES BOTH SCALES, so the software promotion accumulator the block-fp8 body
 *   needs DISAPPEARS. That is why this is cheaper in registers than the path it replaces, not
 *   more expensive: one f32x16 accumulator set instead of two.
 *
 *   E8M0 IS BIASED BY 127 AND BYTE 0 IS 2^-127, NOT NEUTRAL. A zero scale byte silently
 *   produces exactly 0.0. Pad rows below are therefore given PLOW_E8M0_ONE and zero DATA, not
 *   a zero scale.
 *
 *   THE SCALE MUST BE A RUNTIME VALUE. If the scale arguments are compile-time constants the
 *   backend selects the UNSCALED v_mfma_f32_32x32x64_f8f6f4 and drops the microscaling
 *   entirely — correct-looking, silently unscaled. asm_expect_gfx950.json gates this object on
 *   the SCALED mnemonic for exactly that reason.
 *
 * TILE. BM=64 x BN=256 x BK=128. BK is 128 (not the bf16 body's 64) because one A4W4 MFMA
 * consumes K=64, so a 128-K tile is two MFMAs and the same staging cadence. fp4 halves every
 * byte count: the LDS row stride is BK/2 = 64 BYTES and a fragment is 16 B (ds_read_b128), not
 * the fp8 path's 32 B. BM stays 64 for the reason it is 64 in the bf16 body — Kimi routes
 * 8-of-384, so an expert holds ~21 rows at T=1024 and per-expert padding to 128 would run 6x
 * the MFMA rows. That reasoning is about the ROUTING, not the precision, so it is unchanged.
 * ========================================================================================== */
#define MPF4_BM 64
#define MPF4_BN 256
#define MPF4_BK 128                    /* K per staged tile = 2 MFMAs of K=64 */
#define MPF4_RB (MPF4_BK / 2)          /* LDS row stride, BYTES (fp4 = 2/byte) = 64 */
#define MPF4_SPR (MPF4_BK / 32)        /* E8M0 scale bytes per row per tile     = 4  */
#define MPF4_ATB (MPF4_BM * MPF4_RB)   /* A tile bytes  = 4096  */
#define MPF4_BTB (MPF4_BN * MPF4_RB)   /* B tile bytes  = 16384 */
#define MPF4_WMc 2                     /* wave grid: the standard 2x4 (8 waves) */
#define MPF4_WNc 4
/* 16-byte-column XOR swizzle over the 64-byte row: 4 groups, row&3 picks the rotation. Same
 * purpose as the bf16 body's — without it every row of a fragment read hits the same bank. */
#define MPF4_XORSWZ(row, off) ((off) ^ (((row) & 3u) << 4))

/* LDS map (bytes), all carved from the interpreter's existing arena:
 *   [0, 2*(ATB+BTB))          double-buffered fp4 A|B tiles                 40960
 *   as[2][BM][SPR]            A scale bytes                                   512
 *   bs[2][BN][SPR]            B scale bytes                                  2048
 *   bridge f32[BM][BN/2]      GLU epilogue transpose (see the bridge note)  32768
 * = 76288 B, inside the 147464 B arena, so this op does not move the LDS high-water mark. */
#define MPF4_LDS_BYTES (2u * (MPF4_ATB + MPF4_BTB) + 2u * MPF4_BM * MPF4_SPR + \
                        2u * MPF4_BN * MPF4_SPR + MPF4_BM * (MPF4_BN / 2) * 4u)

/* Quantize 32 consecutive bf16 to one MX block: 16 fp4 bytes + one E8M0 byte. This is the
 * A-side of A4W4 and it runs IN THE GEMM'S STAGING PATH, not as a separate pass — the
 * activation is quantized on its way from HBM into LDS, so it is never written back at 16
 * bits and never re-read. */
__device__ __forceinline__ void mpf4_quant_block(const bf16* src, unsigned char* dst16,
                                                 unsigned char* scale_out) {
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const bf16v8 w = ld_glob8(src + i * 8);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            v[i * 8 + j] = bf2f(w[j]);
            amax = fmaxf(amax, fabsf(v[i * 8 + j]));
        }
    }
    const unsigned char sb = e8m0_for_amax(amax);
    const float inv = 1.0f / e8m0_to_f32(sb);
#pragma unroll
    for (int i = 0; i < 16; i++)
        dst16[i] = (unsigned char)(quant_fp4(v[i * 2] * inv) | (quant_fp4(v[i * 2 + 1] * inv) << 4));
    *scale_out = sb;
}

/* GLU=true: gate/up (A = gathered+quantized xn2, B = the expert's mxfp4 gate|up, fused bridge
 *            epilogue writes fu_g as MXFP4 + E8M0 in the SORTED layout gemm2 wants).
 * GLU=false: down    (A = fu_g, already MXFP4, staged raw; scatter epilogue to part[]). */
template <bool GLU>
__device__ void d_moe_group_pf_a4w4(void* __restrict__ Cout, const void* __restrict__ Ain,
                                    const unsigned char* __restrict__ Ascale,
                                    const unsigned long long* __restrict__ wtab,
                                    const unsigned long long* __restrict__ stab,
                                    const int* __restrict__ meta,
                                    const unsigned* __restrict__ row_token,
                                    const unsigned* __restrict__ row_partidx,
                                    const float* __restrict__ row_gate,
                                    unsigned char* __restrict__ Cscale, unsigned N, unsigned K,
                                    unsigned n_exp, unsigned act, unsigned slice, unsigned nblk,
                                    void* ldsv) {
    constexpr int SMa = MPF4_BM / MPF4_WMc / MFMA_M; /* 1 */
    constexpr int SNa = MPF4_BN / MPF4_WNc / MFMA_N; /* 2 */
    constexpr unsigned NB = GLU ? (MPF4_BN / 2) : MPF4_BN;

    unsigned char* L = (unsigned char*)ldsv;
    unsigned char* Atl = L;                                   /* [2][BM][RB] */
    unsigned char* Btl = Atl + 2u * MPF4_ATB;                 /* [2][BN][RB] */
    unsigned char* Asc = Btl + 2u * MPF4_BTB;                 /* [2][BM][SPR] */
    unsigned char* Bsc = Asc + 2u * MPF4_BM * MPF4_SPR;       /* [2][BN][SPR] */
    float* Bridge = (float*)(Bsc + 2u * MPF4_BN * MPF4_SPR);  /* [BM][BN/2] */

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave / MPF4_WNc, wn = wave % MPF4_WNc;
    const unsigned frow = mfma_frag_row(lane);   /* lane % 32 */
    const unsigned khalf = lane / 32u;           /* which 32-k half this lane supplies */

    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned total_tiles = (unsigned)tilep[n_exp];
    const unsigned tnc = (N + NB - 1u) / NB;
    const unsigned n_tiles = total_tiles * tnc;
    const unsigned NT = (K + MPF4_BK - 1u) / MPF4_BK;
    const unsigned KS = K >> 1;      /* fp4 row stride in BYTES */
    const unsigned KSC = K >> 5;     /* E8M0 scale bytes per row */

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned mt = lin / tnc, nt = lin % tnc;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF4_BM;
        const unsigned n0 = nt * NB;

        const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
        if (wb0 == 0ull) continue; /* EP: expert not local — the decode ops skip the same way */
        const unsigned char* W0 = (const unsigned char*)(size_t)wb0;
        const unsigned char* W1 = GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1]
                                      : nullptr;
        const unsigned char* SW0 = (const unsigned char*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)];
        const unsigned char* SW1 = GLU ? (const unsigned char*)(size_t)stab[(size_t)e * 3 + 1]
                                       : nullptr;

        f32x16 acc[SMa][SNa];
#pragma unroll
        for (int i = 0; i < SMa; i++)
#pragma unroll
            for (int j = 0; j < SNa; j++) acc[i][j] = (f32x16)(0.0f);

/* --- A staging. GLU quantizes bf16 -> MXFP4 on the way in (one thread owns one 32-element MX
 * block, so the block amax is a thread-local reduction and needs no shuffle); DOWN copies fp4
 * that the bridge already wrote. A pad row gets ZERO DATA and a NEUTRAL scale — never a zero
 * scale byte, which would be 2^-127 rather than "off". */
#define MPF4_STAGE_A(buf, k0)                                                                  \
    {                                                                                          \
        const unsigned nb_ = MPF4_BM * MPF4_SPR; /* 256 blocks per tile */                     \
        for (unsigned t_ = tid; t_ < nb_; t_ += PLOW_THREADS) {                                \
            const unsigned r_ = t_ / MPF4_SPR, blk_ = t_ % MPF4_SPR;                           \
            unsigned src_;                                                                     \
            if constexpr (GLU) src_ = row_token[rowbase + r_];                                 \
            else src_ = rowbase + r_;                                                          \
            unsigned char* dq_ = Atl + (buf) * MPF4_ATB + r_ * MPF4_RB +                       \
                                 MPF4_XORSWZ(r_, blk_ * 16u);                                  \
            unsigned char* ds_ = Asc + (buf) * MPF4_BM * MPF4_SPR + r_ * MPF4_SPR + blk_;      \
            if (src_ == PLOW_EXPERT_UNUSED) {                                                  \
                _Pragma("unroll") for (int z = 0; z < 16; z++) dq_[z] = 0;                      \
                *ds_ = (unsigned char)PLOW_E8M0_ONE;                                            \
            } else if constexpr (GLU) {                                                        \
                mpf4_quant_block((const bf16*)Ain + (size_t)src_ * K + (k0) + blk_ * 32u, dq_,  \
                                 ds_);                                                          \
            } else {                                                                            \
                __builtin_memcpy(dq_, (const unsigned char*)Ain + (size_t)src_ * KS +           \
                                          ((k0) >> 1) + blk_ * 16u, 16);                        \
                *ds_ = Ascale[(size_t)src_ * KSC + ((k0) >> 5) + blk_];                         \
            }                                                                                   \
        }                                                                                       \
    }

/* --- B staging. Weights are already MXFP4 on disk; this is a byte copy plus its scale byte.
 * Under GLU the tile's low half is the gate weight and its high half the up weight, at the
 * SAME output rows, so the SN axis selects gate vs up in the epilogue with no shuffle. */
#define MPF4_STAGE_B(buf, k0)                                                                  \
    {                                                                                          \
        const unsigned nb_ = MPF4_BN * MPF4_SPR;                                               \
        for (unsigned t_ = tid; t_ < nb_; t_ += PLOW_THREADS) {                                \
            const unsigned br_ = t_ / MPF4_SPR, blk_ = t_ % MPF4_SPR;                          \
            unsigned r_ = n0 + br_;                                                            \
            const unsigned char* ws_ = W0;                                                     \
            const unsigned char* ss_ = SW0;                                                    \
            if constexpr (GLU) {                                                               \
                const bool up_ = (br_ >= MPF4_BN / 2);                                         \
                ws_ = up_ ? W1 : W0;                                                            \
                ss_ = up_ ? SW1 : SW0;                                                          \
                r_ = n0 + (up_ ? br_ - MPF4_BN / 2 : br_);                                      \
            }                                                                                   \
            unsigned char* dq_ = Btl + (buf) * MPF4_BTB + br_ * MPF4_RB +                       \
                                 MPF4_XORSWZ(br_, blk_ * 16u);                                  \
            unsigned char* ds_ = Bsc + (buf) * MPF4_BN * MPF4_SPR + br_ * MPF4_SPR + blk_;      \
            if (r_ < N) {                                                                       \
                __builtin_memcpy(dq_, ws_ + (size_t)r_ * KS + ((k0) >> 1) + blk_ * 16u, 16);    \
                *ds_ = ss_[(size_t)r_ * KSC + ((k0) >> 5) + blk_];                              \
            } else {                                                                            \
                _Pragma("unroll") for (int z = 0; z < 16; z++) dq_[z] = 0;                       \
                *ds_ = (unsigned char)PLOW_E8M0_ONE;                                             \
            }                                                                                   \
        }                                                                                       \
    }

        __syncthreads();
        MPF4_STAGE_A(0, 0)
        MPF4_STAGE_B(0, 0)
        __syncthreads();

        unsigned buf = 0;
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1u) * MPF4_BK;
#pragma unroll
            for (int q = 0; q < MPF4_BK / 64; q++) { /* two K=64 MFMAs per staged tile */
                /* A lane supplies 32 fp4 = 16 B at k = q*64 + 32*khalf. */
                const unsigned boff = (unsigned)q * 32u + khalf * 16u;
                const unsigned sblk = (unsigned)q * 2u + khalf;
                mfma_f8f6f4_operand af[SMa], bfr[SNa];
                int sa[SMa], sb[SNa];
#pragma unroll
                for (int i = 0; i < SMa; i++) {
                    const unsigned ar = wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + frow;
                    af[i] = fp4_frag(Atl + buf * MPF4_ATB + ar * MPF4_RB +
                                     MPF4_XORSWZ(ar, boff));
                    sa[i] = (int)Asc[buf * MPF4_BM * MPF4_SPR + ar * MPF4_SPR + sblk];
                }
#pragma unroll
                for (int j = 0; j < SNa; j++) {
                    const unsigned br = (GLU ? j * (MPF4_BN / 2) + wn * MFMA_N
                                             : wn * (MPF4_BN / MPF4_WNc) + j * MFMA_N) + frow;
                    bfr[j] = fp4_frag(Btl + buf * MPF4_BTB + br * MPF4_RB +
                                      MPF4_XORSWZ(br, boff));
                    sb[j] = (int)Bsc[buf * MPF4_BN * MPF4_SPR + br * MPF4_SPR + sblk];
                }
                __builtin_amdgcn_s_setprio(1);
#pragma unroll
                for (int i = 0; i < SMa; i++)
#pragma unroll
                    for (int j = 0; j < SNa; j++)
                        acc[i][j] = mfma_a4w4(af[i], bfr[j], acc[i][j], sa[i], sb[j]);
                __builtin_amdgcn_s_setprio(0);
            }
            if (kn < K) {
                MPF4_STAGE_A(buf ^ 1, kn)
                MPF4_STAGE_B(buf ^ 1, kn)
            }
            __syncthreads();
            buf ^= 1;
        }

        if constexpr (GLU) {
            /* --- THE FUSED BRIDGE ---------------------------------------------------------
             * SwiGLU, then MXFP4-quantize the result, then write it with its E8M0 scales in
             * the SORTED gathered layout gemm2 reads. The intermediate never exists in HBM at
             * 16 bits and there is no second quantization pass over it.
             *
             * IT NEEDS AN LDS TRANSPOSE AND THAT IS NOT AVOIDABLE. gemm2's MX blocks run along
             * I_moe, i.e. along the COLUMN axis of this output — but the 32x32 MFMA gives each
             * lane 16 different ROWS at ONE column, which is exactly the wrong axis to take a
             * 32-element block amax on. So the tile goes to LDS as f32 and comes back out by
             * (row, 32-column block). 32 KiB, inside the arena, once per tile. */
            float* const Br = Bridge;
            const unsigned cc = wn * MFMA_N + mfma_acc_n(lane);
            if (n0 + cc < N) {
#pragma unroll
                for (int i = 0; i < SMa; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + mfma_acc_m(lane, el);
                        Br[rr * (MPF4_BN / 2) + cc] =
                            moe_act(acc[i][0][el], act) * acc[i][1][el];
                    }
            }
            __syncthreads();
            unsigned char* const fq = (unsigned char*)Cout;
            const unsigned nblocks = MPF4_BM * (NB / 32u);
            for (unsigned t = tid; t < nblocks; t += PLOW_THREADS) {
                const unsigned r = t / (NB / 32u), blk = t % (NB / 32u);
                if (row_partidx[rowbase + r] == PLOW_EXPERT_UNUSED) continue; /* pad row */
                const unsigned c0 = blk * 32u;
                if (n0 + c0 >= N) continue;
                float amax = 0.0f;
#pragma unroll
                for (int z = 0; z < 32; z++)
                    amax = fmaxf(amax, fabsf(Br[r * (MPF4_BN / 2) + c0 + z]));
                const unsigned char sbv = e8m0_for_amax(amax);
                const float inv = 1.0f / e8m0_to_f32(sbv);
                unsigned char* o = fq + (size_t)(rowbase + r) * (N >> 1) + ((n0 + c0) >> 1);
#pragma unroll
                for (int z = 0; z < 16; z++)
                    o[z] = (unsigned char)(quant_fp4(Br[r * (MPF4_BN / 2) + c0 + z * 2] * inv) |
                                           (quant_fp4(Br[r * (MPF4_BN / 2) + c0 + z * 2 + 1] * inv)
                                            << 4));
                Cscale[(size_t)(rowbase + r) * (N >> 5) + ((n0 + c0) >> 5)] = sbv;
            }
        } else {
            float* const part = (float*)Cout;
#pragma unroll
            for (int i = 0; i < SMa; i++)
#pragma unroll
                for (int j = 0; j < SNa; j++) {
                    const unsigned nn =
                        n0 + wn * (MPF4_BN / MPF4_WNc) + j * MFMA_N + mfma_acc_n(lane);
                    if (nn >= N) continue;
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + mfma_acc_m(lane, el);
                        const unsigned pidx = row_partidx[rowbase + rr];
                        if (pidx == PLOW_EXPERT_UNUSED) continue;
                        part[(size_t)pidx * N + nn] = row_gate[rowbase + rr] * acc[i][j][el];
                    }
                }
        }
        __syncthreads();
    }
#undef MPF4_STAGE_A
#undef MPF4_STAGE_B
}

/* op 85: grouped gate/up + GLU. t0=fu_g t1=xn2([T,H]) t2=wtab t3=stab t4=meta t5=row_token
 *   i0=I_moe(N) i1=H(K) i2=n_exp i3=fp8 i5=act */
__device__ void d_moe_group_glu_pf(bf16* fu, const bf16* xn2, const unsigned long long* wtab,
                                   const unsigned long long* stab, const int* meta,
                                   const unsigned* row_token, unsigned I_moe, unsigned H,
                                   unsigned n_exp, unsigned enc, unsigned act, unsigned slice,
                                   unsigned nblk, bf16* lds, const unsigned char* xscale = nullptr,
                                   const unsigned* row_partidx = nullptr,
                                   unsigned char* fu_scale = nullptr) {
    (void)xscale;
#if PLOW_MOE_PF_A4W4
    if (enc == PLOW_MOE_ENC_MXFP4) {
        /* A4W4. `fu` is the MXFP4 gathered intermediate and `fu_scale` its E8M0 rows; the
         * epilogue IS the fused bridge (SwiGLU + quantize + scale write in the sorted layout),
         * so no bf16 intermediate exists anywhere on this path and there is no separate bridge
         * op. row_partidx is needed here, not just in DOWN, so the bridge can skip PAD rows. */
        d_moe_group_pf_a4w4<true>((void*)fu, (const void*)xn2, nullptr, wtab, stab, meta,
                                  row_token, row_partidx, nullptr, fu_scale, I_moe, H, n_exp, act,
                                  slice, nblk, (void*)lds);
        return;
    }
#else
    (void)row_partidx; (void)fu_scale;
#endif
    if (enc == PLOW_MOE_ENC_FP8BLK)
        d_moe_group_pf_t<true, true>(fu, xn2, wtab, stab, meta, row_token, nullptr, nullptr, I_moe,
                                     H, n_exp, act, slice, nblk, lds);
    else
        d_moe_group_pf_t<false, true>(fu, xn2, wtab, stab, meta, row_token, nullptr, nullptr, I_moe,
                                      H, n_exp, act, slice, nblk, lds);
}

/* op 86: grouped down + gate-scale + scatter. t0=part([T*k,H] f32) t1=fu_g t2=wtab t3=stab
 *   t4=meta t6=row_partidx t7=row_gate   i0=H(N) i1=I_moe(K) i2=n_exp i3=fp8 */
__device__ void d_moe_group_down_pf(float* part, const bf16* fu, const unsigned long long* wtab,
                                    const unsigned long long* stab, const int* meta,
                                    const unsigned* row_partidx, const float* row_gate, unsigned H,
                                    unsigned I_moe, unsigned n_exp, unsigned enc, unsigned slice,
                                    unsigned nblk, bf16* lds,
                                    const unsigned char* fu_scale = nullptr) {
#if PLOW_MOE_PF_A4W4
    if (enc == PLOW_MOE_ENC_MXFP4) { /* A = the bridge's MXFP4 output + its E8M0 rows */
        d_moe_group_pf_a4w4<false>((void*)part, (const void*)fu, fu_scale, wtab, stab, meta,
                                   nullptr, row_partidx, row_gate, nullptr, H, I_moe, n_exp, 0,
                                   slice, nblk, (void*)lds);
        return;
    }
#else
    (void)fu_scale;
#endif
    if (enc == PLOW_MOE_ENC_FP8BLK)
        d_moe_group_pf_t<true, false>(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate, H,
                                      I_moe, n_exp, 0, slice, nblk, lds);
    else
        d_moe_group_pf_t<false, false>(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate,
                                       H, I_moe, n_exp, 0, slice, nblk, lds);
}

/* --- op 87: T-TOKEN COMBINE (PLOW_DOP_MOE_COMBINE_PF) --------------------------------------
 * out[t] = residual[t] + shared[t] + Σ_slot part[t*k + slot], f32 accumulate in FIXED slot
 * order — the same expression and the same order as the decode d_moe_combine, so at T=1 this
 * is bit-identical to it. Grid-strided over (token, h) so it fills every CU at any T.
 *   t0=out t1=residual([T,H]) t2=shared([T,H] or none) t3=part([T*k,H] f32)  i0=H i1=k i2=T */
__device__ void d_moe_combine_pf(bf16* out, const bf16* residual, const bf16* shared,
                                 const float* part, unsigned H, unsigned k, unsigned T,
                                 unsigned slice, unsigned nblk) {
    const size_t total = (size_t)T * H;
    const size_t gid = (size_t)slice * PLOW_THREADS + threadIdx.x;
    const size_t stride = (size_t)nblk * PLOW_THREADS;
    for (size_t i = gid; i < total; i += stride) {
        const unsigned tok = (unsigned)(i / H), h = (unsigned)(i - (size_t)tok * H);
        float acc = residual ? bf2f(residual[i]) : 0.0f; /* optional — see d_moe_combine */
        if (shared) acc += bf2f(shared[i]);
        const float* pt = part + (size_t)tok * k * H;
        for (unsigned j = 0; j < k; j++) acc += pt[(size_t)j * H + h];
        out[i] = f2bf(acc);
    }
}

#endif /* PLOW_OP_MOE_H */
