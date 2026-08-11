/* op_moe.h — the MoE data-dependent counter-gate kernels (gfx950).
 *
 * The design notes §3-§4, §2-§3. Four ops implement the
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
/* Router selection: 0 = one all-pairs rank pass; 1 = k parallel block-max passes.
 * K3's 896 experts/top-16 benefit from changing O(E²/threads) comparisons into
 * O(k*E/threads) plus 3*k barriers. Preserve the established path for other models. */
#ifndef PLOW_MOE_ROUTER_SELECT
#define PLOW_MOE_ROUTER_SELECT PLOW_K3
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
/* This used to #error on CDNA3 ("no fp4"). The flag no longer demands the scaled f8f6f4 matrix
 * core: without PLOW_HAS_MX_MMA the same ops compile as the SIMULATED arm — fp4 dequantized to
 * bf16 in staging (exact: <= 3 significant bits, power-of-two scale) and fed to the ordinary
 * bf16 MFMA. Same packet contract, same object name; see the CDNA3 body's header below. */
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

/* THE FAST SPELLINGS WERE TRIED HERE AND MEASURED SLOWER. DO NOT RE-DERIVE THEM.
 *
 * `x * rcp(1 + __expf(-x))` for silu and `fast_tanhf` for gelu_tanh (both from amd_common.h, both
 * exhaustively swept in runtime/tests/situ_identity_gfx950_test.hip, both correct) cut this
 * function from 26 and 38 VALU to 5 and 13. They are still not here, because on K3's MoE prefill
 * geometry (T=1024, 896 experts, grid 512, gfx950) they cost op85 0.18 ms:
 *
 *     op85 GLU+bridge                          act=silu    act=situ
 *     situ fast, moe_act as written below       7.609 ms    7.738 ms
 *     situ fast, moe_act fast too               7.752       7.921
 *
 * Note WHICH column moved: act=situ, the arm that never calls `moe_act` at all. Both activation
 * arms are compiled into `d_moe_group_pf_a4w4` and selected by a uniform branch, so shrinking the
 * one that is not running still re-schedules the kernel that is — and this kernel has no slack to
 * re-schedule into (165 VGPR, 2 waves/SIMD; see the same effect, from the other direction, in
 * `k3_situ_gate`'s note in amd_common.h). The epilogue's VALU is hidden by a 28-K-tile MFMA main
 * loop; its INSTRUCTION SCHEDULE is not.
 *
 * ONE THING THIS LEAVES STANDING, recorded because it is a real divergence and not a typo:
 * op_elementwise.h's `act_silu` is `x / (1 + __expf(-x))` and this one is `x / (1 + expf(-x))` —
 * the fast exponential versus OCML's. GLM-5.2 runs BOTH (dense FFN GLU through op_elementwise.h,
 * routed experts through here), so it computes two silus that differ in the last bit. Converging
 * them is the 0.18 ms above, so they stay diverged until something makes that cheap.
 *
 * The `-w` build hides no warning here: `expf` is the OCML symbol and it is deliberate. */
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

#if PLOW_MOE_ROUTER_SELECT
    for (unsigned j = 0; j < k; j++) {
        unsigned long long mine = 0ull;
        for (unsigned e = tid; e < n_exp; e += PLOW_THREADS)
            mine = keys[e] > mine ? keys[e] : mine;
        const unsigned long long best =
            block_max_u64(mine, (unsigned long long*)(wl + PLOW_MOE_MAX_TOPK));
        if (tid == 0) {
            const unsigned eid = n_exp - 1u - (unsigned)(best & 0xFFFFFu);
            wl[j] = eid;
            keys[eid] = 0ull;
        }
        __syncthreads();
    }
#else
    for (unsigned e = tid; e < n_exp; e += PLOW_THREADS) {
        const unsigned long long myk = keys[e];
        unsigned rank = 0;
#pragma unroll 32
        for (unsigned f = 0; f < n_exp; f++) rank += (keys[f] > myk);
        if (rank < k) wl[rank] = e; /* winner of rank `rank` (rank 0 == highest key) */
    }
    __syncthreads();
#endif

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

/* ---- OPT-IN (PLOW_MOE_DEC_X2=1): the fp8 twin of `wave_dot_mxfp4_x2`, for GLM-5.2 decode ----
 *                                                                            [FP8-DECODE-X2]
 * TWO INDEPENDENT WEIGHT STREAMS IN ONE WAVE. `wave_dot_fp8_blk` above walks its K chunks one
 * load at a time: issue, `s_waitcnt vmcnt(0)`, dequant, dot, next. At GLM-5.2's TP8 routed
 * geometry (K = H = 6144, step = 1024) that is SIX serial HBM round trips per output channel
 * with ONE load in flight, and the megakernel's grid gives 8 waves/CU (2/SIMD) so there is
 * nothing else on the SIMD to hide them behind. The traced cost is in
 * `perf-data/plow-gfx942/glm52-decode-gemv-aiter.md`.
 *
 * The header of `wave_dot_fp8_blk` records that a UN-deep prefetch of the SAME stream was tried
 * and regressed; this is the other axis and the one aiter actually uses (`fmoe pf3` keeps 13-39
 * loads outstanding across partial waits). gate and up are DIFFERENT weights at the SAME output
 * channel, so one wave can carry both: `2*UN` loads are issued before any is consumed, and the
 * activation fragments are read ONCE and used twice. `wave_dot_mxfp4_x2` measured 1.44x for
 * exactly this pairing on the fp4 twin; block-fp8 was left out of it only because GLM-5.2 had to
 * stay byte-identical, which an opt-in build axis is the sanctioned way around.
 *
 * BIT-IDENTICAL to two separate `wave_dot_fp8_blk` calls: each accumulator sees the same chunk
 * partials, in the same order, scaled by the same per-128-block scale, and reduced by the same
 * `wave_sum`. UN>1 only reorders ISSUE, never accumulation: chunk c+u is still added at step
 * c+u, and an overshoot chunk (k >= K) loads zeros through the buffer descriptor and adds an
 * exact +0.0, which is what NOT running it would have done.
 *
 * `UN` should divide `nchunk` or the tail iteration does dead converts — the divisor rule from
 * `gv_un_fp8` (op_gemm.h). GLM-5.2: nchunk = 6144/1024 = 6, and UN=3 divides it exactly. */
/* Both DECODE axes default OFF: with them unset this file compiles to the byte-identical
 * shipped bodies (the two helpers are templates, so an uninstantiated one emits nothing).
 * `scripts/build_gfx942.sh` carries PLOW_MOE_DEC_X2 / PLOW_MOE_DEC_LG as decode-row build axes. */
#ifndef PLOW_MOE_DEC_X2
#define PLOW_MOE_DEC_X2 0
#endif
#ifndef PLOW_MOE_DEC_LG
#define PLOW_MOE_DEC_LG 0
#endif
/* CEILING INSTRUMENT ONLY — see the block in `d_moe_expert_down_fp8_blk`. */
#ifndef PLOW_MOE_DEC_ABL
#define PLOW_MOE_DEC_ABL 0
#endif
/* UN=1 IS THE MEASURED DEFAULT, and the reason is the megakernel's register budget, not the
 * dependency graph. In a standalone TU (`-Rpass` unconstrained) UN=3 compiles to the textbook
 * aiter shape: 18 loads issued, ZERO `vmcnt(0)`, every wait partial (vmcnt(11)..vmcnt(1)). In
 * `interp_decode_fp8_gq` — 108 VGPR, the union over every arm — the allocator cannot hold
 * 2*UN fp8v16 fragments and RE-SERIALIZES, and the drains come back. Audited per chunk-PAIR in
 * the shipped object (`_Z24d_moe_expert_glu_fp8_blk...`, llvm-objdump --mcpu=gfx942):
 *
 *   shipped (2 separate wave_dot_fp8_blk)  282 insts   8 gload   2 full vmcnt(0)
 *   UN=1                                   241         6         1 full + 3 x vmcnt(1)
 *   UN=2                                   236         6         4 full
 *   UN=3                                   235         6         3 full
 *
 * So the pairing pays at UN=1 and the deeper unrolls give the drains straight back. Keep the UN
 * parameter — the shape IS reachable and a future object with register headroom wants it — but
 * ship the depth the 108-register object can actually hold. */
#ifndef PLOW_MOE_DEC_X2_UN
#define PLOW_MOE_DEC_X2_UN 1
#endif
template <unsigned UN = PLOW_MOE_DEC_X2_UN>
__device__ __forceinline__ void wave_dot_fp8_blk_x2(const bf16* x, const unsigned char* W0,
                                                    const float* S0, const unsigned char* W1,
                                                    const float* S1, unsigned K, unsigned lane,
                                                    float& o0, float& o1) {
    const __amdgpu_buffer_rsrc_t r0 = buf_rsrc_fp8_u(W0, K);
    const __amdgpu_buffer_rsrc_t r1 = buf_rsrc_fp8_u(W1, K);
    const unsigned step = PLOW_WAVE * 16;
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned KB = (K + 127u) >> 7;
    const auto* const s0 = as_glob(S0);
    const auto* const s1 = as_glob(S1);
    float a0 = 0.0f, a1 = 0.0f;
    for (unsigned c = 0; c < nchunk; c += UN) {
        fp8v16 w0[UN], w1[UN];
        /* every load of the group issued before ANY of them is touched: 2*UN outstanding */
#pragma unroll
        for (unsigned u = 0; u < UN; u++) w0[u] = buf_ld_fp8(r0, (c + u) * step + lane * 16);
#pragma unroll
        for (unsigned u = 0; u < UN; u++) w1[u] = buf_ld_fp8(r1, (c + u) * step + lane * 16);
#pragma unroll
        for (unsigned u = 0; u < UN; u++) {
            const unsigned k = (c + u) * step + lane * 16;
            unsigned kb = k >> 7;
            if (kb >= KB) kb = KB - 1;
            const unsigned kx = (k < K) ? k : 0u;
            const bf16v8 xlo = ld_glob8(x + kx), xhi = ld_glob8(x + kx + 8); /* read once, used 2x */
            bf16v8 wlo, whi;
            fp8_to_bf16v8(w0[u], wlo, whi);
            a0 += dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) * s0[kb];
            fp8_to_bf16v8(w1[u], wlo, whi);
            a1 += dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) * s1[kb];
        }
    }
    o0 = wave_sum(a0);
    o1 = wave_sum(a1);
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
 * (Kimi H=7168 I=2048, GLM H=6144 I=2048), so the k+32<=K guard is exact, not a clamp.
 *
 * SOFTWARE PREFETCH WAS TRIED HERE AND MEASURED A REGRESSION, which is the SAME verdict the
 * block-fp8 twin above records and it is worth stating separately because the reasoning that
 * predicted a win is sound and still wrong. At Kimi-K3's routed geometry — K = the 3584 LATENT
 * width on gate/up, I_moe = 3072 on down — a fp4 chunk covers 2048 K, so `nchunk` is TWO and
 * exactly ONE weight load is in flight per wave. At the interpreter's occupancy 2 (8 waves/CU,
 * one workgroup) that is ~8 KB of bytes-in-flight per CU against the ~17 KB HBM latency asks for,
 * and the op does measure at half the achievable stream. Adding one chunk of lookahead should
 * therefore have closed it. Measured on gfx950, top-16 of 64 experts, 1.06 GB resident,
 * nblk = 256 = the interpreter's launch shape:
 *
 *   | gate/up (K=3584, I_moe=3072) | down (K=3072, N=3584) |
 *   |---|---|
 *   | 93.0 us / 2012 GB/s  (shipped, this loop) | 46.7 us / 2006 GB/s |
 *   | 118.0 us / 1587 GB/s (1-chunk lookahead, conditional load) | 59.4 us / 1576 GB/s |
 *   | 114.1 us / 1641 GB/s (1-chunk lookahead, branchless index)  | 58.5 us / 1600 GB/s |
 *
 * Both forms are ~25% SLOWER, and the branchless one — which cannot be blamed on a lost load
 * hoist — is barely better than the conditional. So the missing MLP is not something this loop
 * can supply: with only two iterations, rotating the fragment across them adds live state and a
 * loop-carried dependency to a loop that was already too short to amortise either. Where the
 * remaining ~2x lives is recorded at the call sites, not here.
 *
 * The reference for "achievable" is a plain streaming read at the same launch shape: 4059 GB/s at
 * nblk=256, 6241 at nblk=512. So this op is at ~50% of a stream it should be able to match, and
 * `wave_dot_mxfp4_x2` below is what closes most of it. */
__device__ __forceinline__ float wave_dot_mxfp4(const bf16* x, const unsigned char* Wrow,
                                                const unsigned char* srow, unsigned K,
                                                unsigned lane) {
    const unsigned step = PLOW_WAVE * 32; /* 64 lanes x 32 fp4 = 2048 K per pass */
    const unsigned nchunk = (K + step - 1) / step;
    /* Byte offset k/2 in fp4v32 (16 B) units is k>>5 — the SAME index as the E8M0 scale row, which
     * is the alignment property the header above is about. */
    const PLOW_GLOB fp4v32* const W = (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)Wrow;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * step + lane * 32;
        if (k + 32u <= K) {
            const fp4_frag32 wf = fp4_prep32(W[k >> 5], e8m0_to_f32(srow[k >> 5]));
            acc = fp4_dot32(wf, ld_glob8(x + k), ld_glob8(x + k + 8), ld_glob8(x + k + 16),
                            ld_glob8(x + k + 24), acc);
        }
    }
    return wave_sum(acc);
}

/* TWO MXFP4 rows in ONE loop — the decode MoE's bandwidth lever, and the answer to the missing
 * MLP the note above measures.                                              [MXFP4-DECODE-X2]
 *
 * WHAT IS ACTUALLY SCARCE. At Kimi-K3's routed geometry a fp4 chunk is 2048 K, so a whole
 * gate/up dot is TWO chunks and a wave has ONE weight load outstanding at a time. Multiply by
 * the interpreter's occupancy 2 — 8 waves per CU, one workgroup — and the op offers HBM ~8 KB of
 * bytes-in-flight per CU where the latency-bandwidth product asks for ~17. Prefetching inside
 * the K loop cannot fix that (measured above: it makes it worse; two iterations are too few to
 * rotate a fragment through). Raising occupancy is not available: the megakernel's budget is set
 * by the union of every op it carries. What IS available is a SECOND INDEPENDENT WEIGHT STREAM
 * in the same wave, which is exactly what the extra occupancy would have supplied, and both
 * decode expert bodies have one sitting right there:
 *
 *   GATE/UP  gate and up are different weights at the SAME output channel, so the pair also
 *            shares the four activation fragments — read once, used twice.
 *   DOWN     two adjacent output rows h, h+1 against the same activation.
 *
 * MEASURED THROUGH THE SHIPPING BODIES (gfx950, K3 latent geometry K=3584 / I_moe=3072, top-16
 * of 64 experts so 1.06 GB is resident and the LLC cannot hold it, nblk=256 = the interpreter's
 * launch shape, 50 iterations, interleaved A/B; stream reference 4059 GB/s at the same shape):
 *
 *   | op | one stream | two streams | |
 *   |---|--:|--:|--:|
 *   | gate/up (d_moe_expert_glu_fp8_blk)  | 92.1 us / 2033 GB/s | 63.8 us / 2934 GB/s | 1.44x |
 *   | down    (d_moe_expert_down_fp8_blk) | 47.5 us / 1970 GB/s | 31.5 us / 2972 GB/s | 1.51x |
 *
 * i.e. from ~50% of the achievable stream to ~73%. A standalone hand-written probe of the same
 * pairing (no slot/sentinel/table indirection) reached 3260 / 3129 GB/s, so the remaining gap is
 * in the per-output epilogue and the row walk, not in this loop.
 *
 * NOT YET MEASURED AT THE TOKEN. This is an isolated-kernel number, and this file's own history
 * says that is not the same thing: `wave_dot_fp8_blk` above and `glm_shared_glu_split` in
 * mla.rs both record isolated wins that did not survive the interpreter. Decode is
 * weight-bandwidth-bound and this halves nothing about the byte count — it only raises how much
 * of it is in flight — so it should carry, but treat the endpoint as unproven until a K3 blob
 * runs.
 *
 * MXFP4 ONLY, AND THAT IS A HARD LINE, not a scoping convenience. The block-fp8 twin is what
 * GLM-5.2 ships on and it must stay byte-identical, so `wave_dot_fp8_blk` and every fp8 call
 * site are untouched — the pairing lives entirely behind `enc == PLOW_MOE_ENC_MXFP4`. There is a
 * second reason to keep it there: `d_moe_expert_glu_fp8_blk`'s header records that a hand-fused
 * gate+up loop MISCOMPILED in the interpreter megakernel (a stray store faulted at 0 spill), and
 * that history is why the fp8 body is two calls. This arm is a different body under a different
 * compiler; the decode object was re-measured at 248 VGPR / 0 VGPR-spill / occupancy 2 /
 * LDS 147464 B with it in — unchanged — but the first K3 bring-up on real weights should treat
 * a fault here as that bug returning rather than as something new.
 *
 * BIT-IDENTICAL to two separate `wave_dot_mxfp4` calls: each accumulator sees the same fragments
 * in the same order, and the two `wave_sum` reductions are the same reductions. */
__device__ __forceinline__ void wave_dot_mxfp4_x2(const bf16* x, const unsigned char* W0,
                                                  const unsigned char* S0, const unsigned char* W1,
                                                  const unsigned char* S1, unsigned K,
                                                  unsigned lane, float& o0, float& o1) {
    const unsigned step = PLOW_WAVE * 32;
    const unsigned nchunk = (K + step - 1) / step;
    const PLOW_GLOB fp4v32* const A = (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)W0;
    const PLOW_GLOB fp4v32* const B = (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)W1;
    float a0 = 0.0f, a1 = 0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        const unsigned k = c * step + lane * 32;
        if (k + 32u <= K) {
            const fp4v32 w0 = A[k >> 5];
            const fp4v32 w1 = B[k >> 5]; /* independent of w0 — the point of the whole function */
            const float s0 = e8m0_to_f32(S0[k >> 5]), s1 = e8m0_to_f32(S1[k >> 5]);
            const bf16v8 x0 = ld_glob8(x + k), x1 = ld_glob8(x + k + 8),
                         x2 = ld_glob8(x + k + 16), x3 = ld_glob8(x + k + 24);
            a0 = fp4_dot32(fp4_prep32(w0, s0), x0, x1, x2, x3, a0);
            a1 = fp4_dot32(fp4_prep32(w1, s1), x0, x1, x2, x3, a1);
        }
    }
    o0 = wave_sum(a0);
    o1 = wave_sum(a1);
}

/* TWO MXFP4 rows in one loop, LANE-GROUPED — the DOWN projection's map at a NARROW contraction.
 *                                                                        [MXFP4-DECODE-LANEGRP]
 * `wave_dot_mxfp4_x2` above assumes the contraction is wide enough to keep a wave busy: a lane
 * owns 32 elements, so 64 lanes cover 2048 K. At K3's TP8 DOWN the contraction is `I_moe`, the
 * PER-RANK expert intermediate, and that is **384** — twelve lanes' worth. Fifty-two of every
 * wave's sixty-four lanes are idle, and the wave issues a 192-byte load where the coalescer wants
 * a kilobyte. Measured at the emitted shape (H=3584, I_moe=384, top-16 of 896, nblk=256, a 1.97 GB
 * arena rotated per rep so nothing is LLC-resident, stream reference 4064 GB/s):
 *
 *   DOWN, shipped `wave_dot_mxfp4_x2` walk       22.86 us   512 GB/s
 *   DOWN, this map + the flattened slot sweep    12.70 us   921 GB/s   1.80x
 *
 * THE MAP. Split the wave into `RG` groups of `LPG = 64/RG` lanes; group `g` owns output row
 * `h0+g`. Because the DOWN rows are CONTIGUOUS (`wstr = I_moe/2` bytes apart) the RG groups'
 * fragments are contiguous in memory too, so the RG separate 192-byte loads merge into ONE
 * coalesced `RG*192`-byte request. Two row-octets are carried per wave for the same reason
 * `wave_dot_mxfp4_x2` carries two rows: a second independent stream is the missing MLP.
 *
 * BIT-EXACT, and the argument is exact rather than empirical. Each lane still owns exactly ONE
 * 32-element fragment of exactly ONE row (that is what `K <= LPG*32` buys, and the caller checks
 * it), so the per-row `dot8` FMA chain is character-for-character the shipped one. The reduction
 * is a butterfly over the group's LPG lanes with offsets LPG/2..1, which is `wave_sum`'s
 * offsets 32..1 applied to a wave whose other lanes hold +0.0 — the leading offsets add nothing.
 * `runtime/tests/` has no MoE decode golden at this shape; the microbenchmark byte-compares the
 * whole `part` array against the shipped body and they are identical. */
template <unsigned RG>
__device__ __forceinline__ void wave_dot_mxfp4_lg2(const bf16* x, const unsigned char* W,
                                                   const unsigned char* S, unsigned wstr,
                                                   unsigned sstr, unsigned ha, unsigned hb,
                                                   unsigned K, unsigned lane, float& o0,
                                                   float& o1) {
    constexpr unsigned LPG = PLOW_WAVE / RG; /* lanes per row-group */
    const unsigned sub = lane % LPG;
    const PLOW_GLOB fp4v32* const A =
        (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)(W + (size_t)ha * wstr);
    const PLOW_GLOB fp4v32* const B =
        (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)(W + (size_t)hb * wstr);
    const unsigned char* Sa = S + (size_t)ha * sstr;
    const unsigned char* Sb = S + (size_t)hb * sstr;
    float a = 0.0f, b = 0.0f;
    const unsigned k = sub * 32;
    if (k + 32u <= K) {
        const fp4v32 wa = A[k >> 5], wb = B[k >> 5];
        const float sa = e8m0_to_f32(Sa[k >> 5]), sb = e8m0_to_f32(Sb[k >> 5]);
        const bf16v8 x0 = ld_glob8(x + k), x1 = ld_glob8(x + k + 8), x2 = ld_glob8(x + k + 16),
                     x3 = ld_glob8(x + k + 24);
        a = fp4_dot32(fp4_prep32(wa, sa), x0, x1, x2, x3, a);
        b = fp4_dot32(fp4_prep32(wb, sb), x0, x1, x2, x3, b);
    }
#pragma unroll
    for (unsigned off = LPG / 2; off > 0; off >>= 1) {
        a += __shfl_xor(a, (int)off, PLOW_WAVE);
        b += __shfl_xor(b, (int)off, PLOW_WAVE);
    }
    o0 = a;
    o1 = b;
}

/* ---- OPT-IN (PLOW_MOE_DEC_LG=1): NARROW-K LANE-GROUP DOWN, block-fp8 decode experts ----
 *                                                                       [FP8-DECODE-DOWN-LG]
 * THE SHAPE THIS FIXES. `wave_dot_fp8_blk` gives ONE OUTPUT ROW to a whole 64-lane wave and
 * hands lane L the 16 fp8 at k = 16*L. GLM-5.2 at TP8 routes DOWN with K = I_moe = 256, so
 * only lanes 0..15 are in range: 48 of 64 lanes convert and dot bytes the buffer descriptor
 * returned as zero, the wave's `buffer_load_dwordx4` covers 256 useful bytes of a 1024-byte
 * request, and `wave_sum`'s first two butterfly steps add nothing. Then the row's whole cost is
 * ONE dependent load -> wait -> dequant -> dot -> reduce chain, repeated H/(nblk*PLOW_WAVES) = 24
 * times per wave with ONE load in flight. Traced on the box (a 32-CU expert slice, ctx 1024):
 * 1613 CU-us for 49 KB of weights per workgroup, i.e. ~1 GB/s per CU and the LARGEST single
 * packet in a GLM MoE layer at 45-55 us — more than o_proj, which moves EIGHT TIMES the bytes
 * in 23 us.
 *
 * THE MAP. Split the wave into RG row-groups of LPG = 64/RG lanes; group `g` owns its own output
 * row and lane `sub = lane%LPG` owns k = 16*sub. At RG=4 that is LPG=16 lanes x 16 fp8 = 256 K =
 * exactly GLM's contraction, so every lane is live. Then UNR row-BATCHES are issued back to back
 * before any is consumed, so RG*UNR rows — 12 at the defaults — are in flight at once, which is
 * the one discipline the aiter/Tensile diff (`glm52-asm-innerloop-diff.md`) identifies as the
 * real difference: 13-39 outstanding loads, not a leaner instruction mix. The rows of a pass are
 * CONSECUTIVE, so each of the UNR loads is a fully-coalesced 1024-byte fetch of RG whole rows
 * instead of a quarter-used one.
 *
 * BIT-IDENTICAL, by the same argument `wave_dot_mxfp4_lg2` makes: the set of (lane -> k) fragments
 * for a row is unchanged, each row's partials are summed by the SAME xor-butterfly, and the two
 * leading offsets the 64-lane `wave_sum` applies (32 then 16) add lanes that held an exact +0.0.
 * The only observable difference is the SIGN of a zero result (-0.0 + 0.0 = +0.0), which the f32
 * `MoeCombine` that consumes `part` cannot distinguish. Which wave owns which row also moves, and
 * that is unobservable: `part_slot[h]` is written by exactly one lane either way.
 *
 * GUARDED to K <= LPG*16 (the whole contraction must fit one fragment per lane) and to a
 * 16-multiple K; anything wider falls through to the shipped per-row walk, which is exact at any
 * width. Weight rows are loaded as plain 16-byte global vectors rather than through a buffer
 * descriptor because the row index is now LANE-VARYING — a `buf_rsrc` built from a divergent base
 * compiles to a readfirstlane waterfall, which is the cost this arm exists to remove. The `live`
 * guard is what the descriptor's bounds check was doing. */
#ifndef PLOW_MOE_DEC_LG_RG
#define PLOW_MOE_DEC_LG_RG 4
#endif
/* UNR=6 = 24 rows in flight, swept on the ISA in the shipped 108-VGPR object (per OUTPUT ROW,
 * `_Z25d_moe_expert_down_fp8_blk...`; the shipped wave-per-row body is the first line):
 *
 *   shipped   142 insts/row   122 VALU/row   4.0 gload/row   2.00 full vmcnt(0)/row
 *   UNR=2      35.0            28.5           0.75            0.63
 *   UNR=3      34.2            27.6           0.67            0.50
 *   UNR=4      33.4            27.2           0.63            0.44
 *   UNR=6      32.7            26.8           0.58            0.375
 *
 * Monotone, VGPR flat at 108 the whole way, and 6 is where GLM-5.2 stops being arbitrary: at
 * H=6144 over nblk*PLOW_WAVES = 256 waves it makes `ng` exactly 256, so every wave runs ONE
 * balanced pass over 24 consecutive rows instead of a ragged 1-or-2. */
#ifndef PLOW_MOE_DEC_LG_UNR
#define PLOW_MOE_DEC_LG_UNR 6
#endif
template <unsigned RG, unsigned UNR>
__device__ __forceinline__ void moe_down_lg_fp8_blk(const bf16* x, const unsigned char* W,
                                                    const float* S, unsigned h0, unsigned H,
                                                    unsigned K, unsigned KB, unsigned lane,
                                                    float (&o)[UNR]) {
    constexpr unsigned LPG = PLOW_WAVE / RG; /* lanes per row-group */
    const unsigned g = lane / LPG, sub = lane % LPG;
    const unsigned k = sub * 16u;
    const bool live = (k + 16u <= K);
    const unsigned kx = live ? k : 0u;
    const unsigned kb = kx >> 7;
    const auto* const Wg = as_glob(W);
    const auto* const Sg = as_glob(S);
    fp8v16 wv[UNR];
    float bs[UNR];
    /* RG*UNR consecutive rows, every load issued before any is consumed */
#pragma unroll
    for (unsigned u = 0; u < UNR; u++) {
        const unsigned h = h0 + u * RG + g;
        const unsigned hc = (h < H) ? h : (H - 1u); /* clamp: the caller drops the value */
        wv[u] = ld_glob_fp8v16((const unsigned char*)(Wg + (size_t)hc * K + kx));
        bs[u] = Sg[(size_t)(hc >> 7) * KB + kb];
    }
    const bf16v8 xlo = ld_glob8(x + kx), xhi = ld_glob8(x + kx + 8); /* read once, used UNR times */
#pragma unroll
    for (unsigned u = 0; u < UNR; u++) {
        bf16v8 wlo, whi;
        fp8_to_bf16v8(wv[u], wlo, whi);
        float a = live ? dot8(whi, xhi, dot8(wlo, xlo, 0.0f)) * bs[u] : 0.0f;
#pragma unroll
        for (unsigned off = LPG / 2; off > 0; off >>= 1) a += __shfl_xor(a, (int)off, PLOW_WAVE);
        o[u] = a;
    }
}

/* ONE quantized-expert dot, encoding selected at runtime. PLOW_MOE_ENC_* as on ops 85/86.
 * `srow_f` is the block-fp8 scale row (f32), `srow_b` the MXFP4 E8M0 row; the caller passes
 * whichever its encoding uses and the other is ignored.
 *
 * THIS IS THE ENCODING CHOKE POINT for every DECODE expert body (ops 40/43 per-slot, 48/49
 * grouped, both loop variants), so the refusal below covers all six of them at one site.
 *
 * IT USED TO BE `if (mxfp4) ... else fp8_blk`, i.e. every OTHER value of the field — including
 * PLOW_MOE_ENC_BF16, which is ZERO and therefore what an `i[6]` nobody wrote also reads as —
 * silently decoded bf16 (or garbage) as e4m3 against an f32 `[N/128][K/128]` grid that is not
 * one. Two bytes per element read as one, a scale row read out of weight data: no fault, no
 * trap, finite numbers, wrong model. That is the exact failure this file's `moe_act` poison
 * exists to prevent, and the encoding operand had no equivalent.
 *
 * BF16 IS REFUSED RATHER THAN IMPLEMENTED, deliberately. A bf16 MoE already has its own
 * opcodes — `d_moe_expert_glu` / `d_moe_expert_down` (41/42) — and `mla.rs:3298`
 * (`use_fp8 = enc != MoeEnc::Bf16`) routes to them, so no shipping emitter can land here with
 * enc 0. `k3.rs:418` CAN: it emits `MoeExpertGluFp8Blk` unconditionally and passes `c.enc`
 * through, so a bf16 K3 config would arrive here with 0 — and did, silently. A wave-reduced
 * bf16 arm is ~10 lines, but adding a fourth body nothing emits is speculative; a NaN says
 * "this object has no body for that field value" on the first token instead.
 *
 * NUMERICS: enc == PLOW_MOE_ENC_FP8BLK is unchanged (one uniform scalar compare ahead of the
 * same call), so GLM-5.2 stays byte-identical. */
__device__ __forceinline__ float wave_dot_enc(unsigned enc, const bf16* x,
                                              const unsigned char* Wrow, const float* srow_f,
                                              const unsigned char* srow_b, unsigned K,
                                              unsigned lane) {
    if (enc == PLOW_MOE_ENC_MXFP4) return wave_dot_mxfp4(x, Wrow, srow_b, K, lane);
    if (enc != PLOW_MOE_ENC_FP8BLK) return __builtin_nanf(""); /* POISON — see the note above */
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
        float g, u;
        if (mx) {
            /* gate and up are two INDEPENDENT weight streams at the same channel, so one wave
             * can carry both and the activation is read once. 1.44x here — see
             * `wave_dot_mxfp4_x2`, including why this pairing is MXFP4-only. */
            wave_dot_mxfp4_x2(x, Wg + (size_t)n * wstr, Bg + brow, Wu + (size_t)n * wstr,
                              Bu + brow, H, lane, g, u);
#if PLOW_MOE_DEC_X2
        } else if (enc == PLOW_MOE_ENC_FP8BLK) {
            /* OPT-IN — gate|up as ONE loop with both weight streams in flight and the activation
             * fragments read once. Bit-identical to the two calls below; see
             * `wave_dot_fp8_blk_x2`. */
            wave_dot_fp8_blk_x2<>(x, Wg + (size_t)n * wstr, Sg + nrow, Wu + (size_t)n * wstr,
                                  Su + nrow, H, lane, g, u);
#endif
        } else {
            g = wave_dot_enc(enc, x, Wg + (size_t)n * wstr, Sg + nrow, Bg + brow, H, lane);
            u = wave_dot_enc(enc, x, Wu + (size_t)n * wstr, Su + nrow, Bu + brow, H, lane);
        }
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
    if (mx) {
        /* TWO ADJACENT OUTPUT ROWS PER WAVE. DOWN has no gate/up pair to exploit, so the second
         * independent stream is row h+1 — same reason, same 1.51x (`wave_dot_mxfp4_x2`). This
         * moves which wave owns which row and NOTHING else: each row's dot, reduction and store
         * are unchanged, so the output is bit-identical to the one-row-per-wave walk. The odd-H
         * tail duplicates row h rather than reading past the weight (H is even for every real
         * expert; the guard is the backstop). */
        for (unsigned h = (slice * PLOW_WAVES + wave) * 2u; h < H; h += wstride * 2u) {
            const unsigned h1 = (h + 1u < H) ? h + 1u : h;
            float y0, y1;
            wave_dot_mxfp4_x2(fu_slot, Wd + (size_t)h * wstr, Bd + (size_t)h * (I_moe >> 5),
                              Wd + (size_t)h1 * wstr, Bd + (size_t)h1 * (I_moe >> 5), I_moe, lane,
                              y0, y1);
            if (lane == 0) {
                part_slot[h] = gate * y0;
                if (h + 1u < H) part_slot[h + 1u] = gate * y1;
            }
        }
        return;
    }
#if PLOW_MOE_DEC_LG && !PLOW_MOE_DEC_ABL
    /* OPT-IN — narrow-K lane-group arm. GLM-5.2 TP8 lands here (I_moe = 256 = LPG*16 at RG=4);
     * anything wider falls through to the shipped per-row walk below. See `moe_down_lg_fp8_blk`. */
    {
        constexpr unsigned RG = PLOW_MOE_DEC_LG_RG, UNR = PLOW_MOE_DEC_LG_UNR;
        constexpr unsigned LPG = PLOW_WAVE / RG;
        if (enc == PLOW_MOE_ENC_FP8BLK && I_moe <= LPG * 16u && (I_moe & 15u) == 0u) {
            const unsigned per = RG * UNR;
            const unsigned ng = (H + per - 1u) / per;
            const unsigned sub = lane % LPG, grp = lane / LPG;
            for (unsigned f = slice * PLOW_WAVES + wave; f < ng; f += wstride) {
                const unsigned h0 = f * per;
                float o[UNR];
                moe_down_lg_fp8_blk<RG, UNR>(fu_slot, Wd, Sd, h0, H, I_moe, KB, lane, o);
                if (sub == 0u) {
#pragma unroll
                    for (unsigned u = 0; u < UNR; u++) {
                        const unsigned h = h0 + u * RG + grp;
                        if (h < H) part_slot[h] = gate * o[u];
                    }
                }
            }
            return;
        }
    }
#endif
#if PLOW_MOE_DEC_ABL
    /* CEILING INSTRUMENT ONLY (PLOW_MOE_DEC_ABL=1|2) — WRONG OUTPUT BY CONSTRUCTION, never a
     * serve asset. 1 = keep the walk and the store, delete every load and the dot (prices the
     * body's memory + VALU); 2 = retire the op entirely (prices the packet: gate poll, L2
     * maintenance, counter release, launch). The pair is what turns "the arm changed the ISA 4x
     * and the packet did not move" from a puzzle into a located bottleneck. */
    for (unsigned h = slice * PLOW_WAVES + wave; h < H; h += wstride) {
#if PLOW_MOE_DEC_ABL == 1
        if (lane == 0) part_slot[h] = gate;
#endif
    }
    return;
#endif
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
                                     unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, float, float);
#endif
__device__ void d_moe_group_glu_fp8_blk(bf16* fu, const bf16* x, const unsigned char* table,
                                        const unsigned long long* wtab,
                                        const unsigned long long* stab, unsigned k, unsigned I_moe,
                                        unsigned H, unsigned n_exp, unsigned act, unsigned slice,
                                        unsigned nblk, unsigned enc, float beta,
                                        float lbeta) {
#if PLOW_MOE_MFMA
    /* `enc` IS NOW HONOURED HERE. It used to be dropped on the floor with a comment saying so:
     * `d_moe_group_glu_mfma` was block-fp8 only, so under this non-default axis an MXFP4 packet's
     * gate/up read fp4 nibbles as block-fp8 while the DOWN twin and both other GLU bodies read
     * them correctly — a dropped operand presenting as a gate/up numerics bug, with nothing at
     * runtime that would say so. The MFMA body has an fp4 arm now (see it for why the arm is ~20
     * lines and why the MX scale fold is EXACT there), and any enc it still has no body for is
     * refused loudly rather than aliased onto block-fp8. */
    d_moe_group_glu_mfma(fu, x, table, wtab, stab, k, I_moe, H, n_exp, act, slice, nblk, enc, beta,
                         lbeta);
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
    /* MXFP4: FLATTEN THE (slot, channel) SPACE. The slot-outer walk below gives ONE WAVE PER
     * OUTPUT CHANNEL, and at K3's TP8 geometry `I_moe` is the PER-RANK expert intermediate — 384.
     * With the interpreter's 256 workgroups x 8 waves that is 384 of 2048 waves, i.e. 48 of 256
     * CUs, and the k=16 slots then run one after another. Flattening to k*I_moe = 6144 outputs
     * puts three on every wave and lights every CU. Measured at the emitted shape (H=3584,
     * I_moe=384, top-16 of 896, nblk=256, 1.97 GB arena rotated per rep, stream ref 4064 GB/s):
     *
     *   GLU slot-outer (below)                  37.06 us    631 GB/s
     *   GLU flat, gate|up unpaired              16.19 us   1446 GB/s
     *   GLU flat + `wave_dot_mxfp4_x2`          12.71 us   1841 GB/s   2.92x
     *
     * i.e. the flattening is worth 2.29x on its own and the pairing the other 1.27x.
     *
     * THIS IS NOT `PLOW_MOE_GROUP_FLAT`, and the difference is why that knob measured as a LOSS.
     * That arm flattens BOTH ops and drops the mxfp4 pairing (two `wave_dot_enc` calls); this one
     * is MXFP4-only, keeps the pairing, and leaves DOWN — which does not want the same treatment —
     * to its own map. The knob's -0.76 ms was also measured on a run with NO expert weights bound
     * (`bind_packed_experts` needs a `--checkpoint`, and the K3 checkpoint here is config-only), so
     * every routed slot returned at the `wg_base == 0` test and the "body" it compared was empty.
     *
     * BIT-IDENTICAL to the slot-outer walk: the per-output dot, its lane fragments, its `wave_sum`
     * and its epilogue are untouched — only which wave owns which (slot, channel) moves. Byte-
     * compared against the slot-outer body over the whole `fu` array. Block-fp8 (GLM-5.2) does not
     * enter here at all. */
    if (enc == PLOW_MOE_ENC_MXFP4) {
        const unsigned lane = threadIdx.x & 63;
        const unsigned wave = threadIdx.x >> 6;
        const unsigned wstride = nblk * PLOW_WAVES;
        const unsigned total = k * I_moe;
        for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += wstride) {
            const unsigned slot = f / I_moe;
            const unsigned n = f - slot * I_moe;
            const unsigned eid = moe_slot_expert(table, slot);
            if (eid >= n_exp) continue;                    /* sentinel: skip */
            const unsigned long long wg = wtab[(size_t)eid * 3 + 0];
            if (wg == 0ull) continue;                      /* EP: not this rank's expert */
            const unsigned char* Wg = (const unsigned char*)(size_t)wg;
            const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
            const unsigned char* Bg = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 0];
            const unsigned char* Bu = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 1];
            const size_t brow = (size_t)n * (H >> 5);
            float g, u;
            wave_dot_mxfp4_x2(x, Wg + (size_t)n * (H >> 1), Bg + brow, Wu + (size_t)n * (H >> 1),
                              Bu + brow, H, lane, g, u);
            if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(moe_glu(g, u, act, beta, lbeta));
        }
        return;
    }
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
    /* MXFP4 with a NARROW contraction: the lane-group map plus the flattened (slot, row-octet)
     * sweep. See `wave_dot_mxfp4_lg2` for the map and for why it is bit-exact; the flattening is
     * the same argument as the GLU twin's — the shipped walk covers `H/2` row-pairs per slot with
     * one wave each and then runs the k slots in sequence, and flattening to `k*H/8` items fills
     * every wave with independent work instead. Measured together at the emitted shape:
     *
     *   DOWN slot-outer, `wave_dot_mxfp4_x2`      22.86 us   512 GB/s
     *   DOWN flat + lane-group, 8 rows/wave       12.70 us   921 GB/s   1.80x
     *
     * `RG = 4` (16 lanes per row-group) is the widest group that still keeps ONE 32-element
     * fragment per lane at `I_moe = 384`; the guard is `I_moe <= LPG*32 = 512`, and anything wider
     * falls through to the shipped walk, which is exact at any width. RG=8 measured the same
     * 12.7 us but puts TWO chunks in a lane at this K, which changes the FMA association — so the
     * bit-exact map is the one that ships. */
    if (enc == PLOW_MOE_ENC_MXFP4 && I_moe <= 512u && (I_moe & 31u) == 0u) {
        constexpr unsigned RG = 4, LPG = PLOW_WAVE / RG;
        const unsigned lane = threadIdx.x & 63;
        const unsigned wave = threadIdx.x >> 6;
        const unsigned wstride = nblk * PLOW_WAVES;
        const unsigned ng = (H + 2 * RG - 1) / (2 * RG); /* row-octets per slot */
        const unsigned total = k * ng;
        const unsigned wstr = I_moe >> 1, sstr = I_moe >> 5;
        for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += wstride) {
            const unsigned slot = f / ng;
            const unsigned h = (f - slot * ng) * 2u * RG;
            const unsigned ra = h + lane / LPG, rb = h + RG + lane / LPG;
            const unsigned eid = moe_slot_expert(table, slot);
            float* ps = part + (size_t)slot * H;
            if (eid >= n_exp || wtab[(size_t)eid * 3 + 2] == 0ull) {
                if ((lane % LPG) == 0) { /* deterministic zero partial, as the shipped skip */
                    if (ra < H) ps[ra] = 0.0f;
                    if (rb < H) ps[rb] = 0.0f;
                }
                continue;
            }
            const float gate = moe_slot_gate(table, slot);
            const unsigned char* Wd = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 2];
            const unsigned char* Bd = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 2];
            /* The odd tail clamps to a row already covered, exactly as the shipped `h1` guard. */
            const unsigned ca = ra < H ? ra : h, cb = rb < H ? rb : ca;
            float y0, y1;
            wave_dot_mxfp4_lg2<RG>(fu + (size_t)slot * I_moe, Wd, Bd, wstr, sstr, ca, cb, I_moe,
                                   lane, y0, y1);
            if ((lane % LPG) == 0) {
                if (ra < H) ps[ra] = gate * y0;
                if (rb < H) ps[rb] = gate * y1;
            }
        }
        return;
    }
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
 * MEASURED VERDICT (EP2+grouped, MI350X, 6L): SLOWER (4.746 vs 3.855 ms/tok, +23%). It was ALSO
 * recorded as INCORRECT ("decode tokens diverge — a fragment/layout bug I did not root-cause");
 * that half is now RESOLVED and it was never a layout bug — see `moe_scaled_fp8x8` below for the
 * one-line cause and `runtime/tests/moe_mxfp4_decode_gfx950_test.hip` for the measurement that
 * closed it (3.8e8 -> 2.7e-3 relative, i.e. bf16 output rounding, against an f64 oracle). The
 * perf verdict is unaffected and stands on its own: at M=1 only 1 of the 16 padded MFMA rows is
 * real, so the XDL pass
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
/* THE "fragment/layout bug I did not root-cause" IN THE HEADER ABOVE WAS HERE, AND IT WAS NOT A
 * LAYOUT BUG AT ALL. `f2bf` returns a `bf16`, which this file defines as `unsigned short` — a BIT
 * PATTERN, not a number (amd_common.h:14, "we keep bf16 as a bare u16 in memory"). `bf16_t` is
 * `__bf16`, a real floating type. So `(bf16_t)f2bf(v)` is a NUMERIC conversion of the bit pattern:
 * 1.0f became 16256.0, not 1.0. Every weight fragment fed to the matrix core was its own encoding
 * read as a decimal. Measured against the f64 oracle at K3 latent geometry it is 3.8e8 relative
 * error — which is why the axis was recorded as "decode tokens diverge", and why that was filed
 * as a layout mystery: the A fragment, the B fragment, the accumulator extraction and the scale
 * indexing in this body are all correct, and the test below now proves it by driving the SAME
 * body's MXFP4 arm (which bit-casts) to fdot2 agreement.
 *
 * `__builtin_bit_cast` is the conversion that was meant. It does NOT rehabilitate the axis — the
 * MEASURED VERDICT above is a perf verdict at M=1 and stands on its own — but a scaffold kept for
 * a future a8w8 attempt has to be arithmetically sound, or the next person to switch it on
 * re-derives this. */
__device__ __forceinline__ bf16x8 moe_scaled_fp8x8(const unsigned char* p, float sc) {
    const bf16v8 d = fp8v8_to_bf16v8(ld_glob_fp8v8(p)); /* exact e4m3->bf16 decode */
    bf16x8 out;
#pragma unroll                                              /* fold per-128-block scale */
    for (int i = 0; i < 8; i++) out[i] = __builtin_bit_cast(bf16_t, f2bf(bf2f(d[i]) * sc));
    return out;
}

/* THE MXFP4 B-FRAGMENT — the fp4 twin of moe_scaled_fp8x8, and the whole of the arm this body
 * was missing. 8 consecutive fp4 are 4 BYTES, i.e. ONE u32, which is exactly the operand
 * `fp4_to_bf16v8` takes, so the fragment shape the MFMA wants and the fragment shape the
 * hardware converter emits already agree — that is why this is four lines and not a rewrite.
 *
 * AND THE SCALE FOLD IS EXACT HERE, where the fp8 twin's is not. `moe_scaled_fp8x8` multiplies
 * an arbitrary f32 block scale in software and RE-ROUNDS to bf16, losing precision the e4m3
 * value had (amd_common.h: the cvt's scalef32 operand is E8M0 and would discard the mantissa).
 * An MX scale IS a power of two, so `cvt_scalef32_pk_bf16_fp4` applies it with no error at all
 * and there is no multiply and no second rounding. The fp4 arm is therefore strictly the
 * CLEANER of the two, not a degraded version of it. */
__device__ __forceinline__ bf16x8 moe_scaled_fp4x8(const unsigned char* p, float sc) {
    const unsigned w = *(const PLOW_GLOB unsigned*)(const PLOW_GLOB void*)p; /* 8 fp4 = 4 B */
    return __builtin_bit_cast(bf16x8, fp4_to_bf16v8(w, sc)); /* both are 8 x 16-bit, 16 B */
}

__device__ void d_moe_group_glu_mfma(bf16* fu, const bf16* x, const unsigned char* table,
                                     const unsigned long long* wtab, const unsigned long long* stab,
                                     unsigned k, unsigned I_moe, unsigned H, unsigned n_exp,
                                     unsigned act, unsigned slice, unsigned nblk, unsigned enc,
                                     float beta, float lbeta) {
    typedef float f32x4 __attribute__((ext_vector_type(4)));
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    const unsigned frow = lane & 15u;   /* n within the 16-tile (== the acc m/n row selector) */
    const unsigned kgrp = lane >> 4;    /* 0..3 -> this lane's k-offset group (8*kgrp) */
    const unsigned KB = (H + 127u) >> 7;
    const unsigned ntile = I_moe >> 4;  /* 16-channel N-tiles per expert */
    const unsigned total = k * ntile;
    /* Two arms, and no third. Aliasing an unknown encoding onto either one is the bug this
     * function's header used to DOCUMENT rather than fix; a NaN sweep of the whole gate/up
     * intermediate reaches the residual on the first token instead. Uniform branch, so the
     * block-fp8 and MXFP4 paths pay one scalar compare. */
    if (enc != PLOW_MOE_ENC_FP8BLK && enc != PLOW_MOE_ENC_MXFP4) {
        const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
        const unsigned stride = nblk * PLOW_THREADS;
        for (unsigned i = gid; i < k * I_moe; i += stride) fu[i] = f2bf(__builtin_nanf(""));
        return;
    }
    for (unsigned t = slice * PLOW_WAVES + wave; t < total; t += wstride) {
        const unsigned slot = t / ntile;
        const unsigned n0 = (t - slot * ntile) << 4;      /* first output channel of this tile */
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;                        /* sentinel/EP-non-local: leave fu unwritten */
        const unsigned long long wgb = wtab[(size_t)eid * 3 + 0];
        if (wgb == 0ull) continue;                          /* EP: expert not owned by this rank */
        const unsigned char* Wg = (const unsigned char*)(size_t)wgb;
        const unsigned char* Wu = (const unsigned char*)(size_t)wtab[(size_t)eid * 3 + 1];
        /* stab[] holds ONE pointer per projection and the encoding decides how to READ it: an f32
         * [N/128][K/128] `weight_scale_inv` grid under block-fp8, a byte [N][K/32] E8M0 row under
         * MXFP4. Both spellings are taken here and only the live one is dereferenced. */
        const float* Sg = (const float*)(size_t)stab[(size_t)eid * 3 + 0];
        const float* Su = (const float*)(size_t)stab[(size_t)eid * 3 + 1];
        const unsigned char* Bg = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 0];
        const unsigned char* Bu = (const unsigned char*)(size_t)stab[(size_t)eid * 3 + 1];
        const unsigned srow = (n0 >> 7) * KB; /* all 16 channels share the (n0>>7) block-scale row */
        const unsigned n = n0 + frow;         /* this lane's weight output row */
        /* Row strides by encoding, exactly as the fdot2 bodies compute them: block-fp8 is 1 B/elt,
         * MXFP4 is 2 elt/B. `mx` is packet-uniform, so every branch on it below is scalar. */
        const bool mx = (enc == PLOW_MOE_ENC_MXFP4);
        const unsigned wstr = mx ? (H >> 1) : H;
        const size_t brow = (size_t)n * (H >> 5); /* this row's E8M0 scale row (MXFP4) */
        f32x4 accg = {0, 0, 0, 0}, accu = {0, 0, 0, 0};
        for (unsigned s = 0; s < H; s += 32) {
            const unsigned kk = s + 8u * kgrp;             /* this lane's 8 k-values */
            bf16x8 af;
            if (frow == 0) af = *(const bf16x8*)(x + kk);  /* m=0 row = x; m=1..15 pad */
            else af = (bf16x8)((bf16_t)0);
            bf16x8 bfg, bfu;
            if (mx) {
                /* kk is 8-aligned and s is 32-aligned, so kk>>5 == s>>5: a lane's 8 fp4 never
                 * straddle two MX blocks and one scale byte covers the whole fragment. The byte
                 * offset kk>>1 is 4-aligned, so the u32 load is aligned by construction. */
                const float sgm = e8m0_to_f32(Bg[brow + (kk >> 5)]);
                const float sum = e8m0_to_f32(Bu[brow + (kk >> 5)]);
                bfg = moe_scaled_fp4x8(Wg + (size_t)n * wstr + (kk >> 1), sgm);
                bfu = moe_scaled_fp4x8(Wu + (size_t)n * wstr + (kk >> 1), sum);
            } else {
                const float sg = Sg[srow + (s >> 7)];      /* 32-aligned frag is within one 128-block */
                const float su = Su[srow + (s >> 7)];
                bfg = moe_scaled_fp8x8(Wg + (size_t)n * wstr + kk, sg);
                bfu = moe_scaled_fp8x8(Wu + (size_t)n * wstr + kk, su);
            }
            accg = plow_mfma_bf16_16x16(af, bfg, accg);
            accu = plow_mfma_bf16_16x16(af, bfu, accu);
        }
        if (kgrp == 0) /* lanes 0..15 hold m=0 (e=0) for the 16 output channels */
            fu[(size_t)slot * I_moe + n0 + frow] =
                f2bf(moe_glu(accg[0], accu[0], act, beta, lbeta));
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
                                    unsigned nblk, float beta, float lbeta) {
    const unsigned KB = (K + 127u) >> 7;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wstride = nblk * PLOW_WAVES;
    auto* const fg = as_glob(fu);
    for (unsigned n = slice * PLOW_WAVES + wave; n < N; n += wstride) {
        const unsigned nrow = (n >> 7) * KB;
        const float g = wave_dot_fp8_blk(x, Wg + (size_t)n * K, Sg + nrow, K, lane);
        const float u = wave_dot_fp8_blk(x, Wu + (size_t)n * K, Su + nrow, K, lane);
        if (lane == 0) st_act1(&fg[n], f2bf(moe_glu(g, u, act, beta, lbeta)));
    }
}

/* --- kernel 2a: EXPERT GATE/UP (PLOW_DOP_MOE_EXPERT_GLU, moe-ep-kernels §3a) --------------
 * fu[slot] = act(gate·x) * (up·x), streaming ONLY the chosen expert's weights. Sentinel skip.
 * Weight bases resolved from expert_weight_table[expert_id] (two-level indirection). */
__device__ void d_moe_expert_glu(bf16* fu, const bf16* x, const unsigned char* table,
                                 const unsigned long long* wtab, unsigned slot, unsigned I_moe,
                                 unsigned H, unsigned n_exp, unsigned act, unsigned slice,
                                 unsigned nblk, float beta, float lbeta) {
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
        fu_slot[n] = f2bf(moe_glu(g, u, act, beta, lbeta));
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
        st_act1(&out[h], f2bf(acc));
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
 * note above for why it is 64 and when to raise it.
 *
 * MPF_BK is overridable (-DMPF_BK=32) for the BATCHED-DECODE object: the OCC4 decode arena
 * is 30,720 B and the (64+256)*64 tile needs 40,960, so the decode row builds at BK=32 with
 * MPF_DBUF=1 — (64+256)*32*2 = 20,480 B. Halving BK doubles the k-passes but each expert
 * weight byte still crosses HBM exactly once, so the stream the grouped form exists to
 * amortise is unchanged; BN (and with it SN, which the epilogue's initializers hardcode at
 * 2) stays put. The #ifndef is digest-safe: unset, the preprocessed expansion is identical. */
#ifndef MPF_BM
#define MPF_BM 64
#endif
#define MPF_BN 256
#ifndef MPF_BK
#define MPF_BK 64
#endif
#define MPF_WM 2
#define MPF_WN 4
#define MPF_SM (MPF_BM / MPF_WM / MFMA_M) /* 1 */
#define MPF_SN (MPF_BN / MPF_WN / MFMA_N) /* 2 */
#define MPF_STRIDE MPF_BK
#define MPF_TILE ((MPF_BM + MPF_BN) * MPF_STRIDE)
/* Halves of LDS the grouped GEMM needs. Double-buffered (81,920 B) where the arena holds two
 * tiles — CDNA4's 160 KiB does. CDNA3's `plow_smem` is 64,512 B, and the OLD unconditional
 * `2 * MPF_TILE` layout was a MEASURED out-of-bounds on gfx942: the second buffer's B tile
 * spans bytes [49,152, 81,920), i.e. 17,408 B past the union and 16,384 B past the 64 KiB
 * physical LDS, and a hardware probe (lds_oob_probe) shows accesses past the allocation do
 * not round-trip on gfx942 — so the GLU arm's odd-k-tile `up` weights and most of DOWN's odd
 * k-tiles read garbage, silently. Same fix as the Gemma twin (`GMPF_DBUF`): one buffer on a
 * 64 KiB part, with the extra barrier the DB==1 commit needs. The static_assert in interp.hip
 * (mirroring the GMPF one) is what keeps this true if either side moves. */
/* Overridable (-DMPF_DBUF=1) for the batched-decode object: the selector below compares
 * against PHYSICAL LDS, but the OCC4 decode arena is 30,720 B — smaller than physical — so
 * the decode row pins DBUF=1 alongside its BN=128. Digest-safe like MPF_BN's guard. */
#ifndef MPF_DBUF
#if (2 * MPF_TILE * 2) <= PLOW_LDS_MAX_BYTES
#define MPF_DBUF 2
#else
#define MPF_DBUF 1
#endif
#endif
#define MPF_LDS_HALVES (MPF_DBUF * MPF_TILE)
/* Same 16-byte-column XOR swizzle d_gemm_t uses, for the same reason: the compact BK-half row
 * stride is exactly 32 banks, so without it every row of a ds_read_b128 fragment collides. */
#define MPF_XORSWZ(row, off) ((off) ^ (((row) & (MPF_BK / 8 - 1)) << 3))
/* SUB-QUANTUM A TILE (the OCC4 batched-decode recut, BM=64/BK=32): BM*BK/THREADS = 4 halves
 * per thread, below the one-16-byte-vector staging quantum — APT/8 truncates to zero and the
 * unfixed loop stages NO A tile (the recorded silent-wrong failure). The arm keyed on this
 * macro MASKS the pass instead of narrowing the quantum: the first BM*BK/8 threads — whole
 * waves, the predicate is wave-uniform — each stage one full 8-half vector in one pass.
 * Loads stay 16-byte and the swizzled LDS cell map is unchanged (element k still lands at
 * cell k ^ ((row & 3) << 3); the XOR touches bits 3-4 only, so an aligned 8-half store spans
 * the same cells at any BK >= 8). */
#define MPF_SUBQ ((MPF_BM * MPF_BK / PLOW_THREADS) < 8)
/* Preshuffled-B address for element k = k0 + (el % BK) of output row r (the PRESHUFFLED note
 * in MPF_FETCH_B). At BK=64 the k0/el form is exact: k0 is 64-aligned and el % BK == el & 63.
 * At BK < 64 odd k-tiles start MID-slab and el & 63 aliases thread parity into the byte
 * index, so both the slab index and the byte offset must derive from the element k itself. */
#if MPF_BK == 64
#define MPF_SHUF_WP(wsrc, k0, el, kk, r) \
    ((wsrc) + ((((size_t)(k0) >> 6) * N + (r)) << 6) + ((el) & 63u))
#else
#define MPF_SHUF_WP(wsrc, k0, el, kk, r) \
    ((wsrc) + ((((size_t)(kk) >> 6) * N + (r)) << 6) + ((kk) & 63u))
#endif

/* PLOW_MOE_PF_PIPE — the aiter-shape two-tile REGISTER pipeline for the grouped prefill
 * GEMM's k-loop (CDNA3 default ON; =0 restores the shipped single-stage loop verbatim).
 *
 * WHY (glm52-asm-innerloop-diff.md): the shipped loop's waits are DATAFLOW-FORCED — the
 * loop-head loads feed the SAME iteration's single-buffer LDS commit, so the commit drains
 * to vmcnt(0) 3-4x per k-tile while aiter's fmoe `pf3` sustains 13-39 outstanding loads
 * with partial waits. No scheduler hint can fix a dependency (PLOW_MOE_PF_SCHED proved
 * null by construction). This restructure changes the DATAFLOW instead: two register
 * stages rotate so that when stage S commits to LDS, the OTHER stage's loads for the
 * tile after next are already in flight — the commit's wait becomes a partial vmcnt by
 * construction, not by hint. The fp8 dequant moves from fetch to commit for the same
 * reason: converting at fetch anchors a VALU consumer next to the loads and re-forces
 * the drain. Raw staging is also SMALLER (fp8 bytes: 16 dwords/stage vs 32 converted).
 * Bit-identical: same bytes to the same LDS cells, same MFMA and promotion order.
 * CDNA4 keeps its LDS ping-pong (DB==2) and is out of scope. */
/* MEASURED PERFORMANCE-NEUTRAL, so DEFAULT OFF (opt-in build axis). The restructure DID
 * land the aiter pipeline shape — ISA-confirmed: the B-weight loads for the next tile now
 * issue before the current commit, so the MFMAs run against LDS with partial `lgkmcnt`
 * waits between them instead of the old vmcnt(0) drain. But served TTFT is a wash
 * (1040.5 vs 1044.1 @4k, 1929.3 vs 1929.7 @8k, bit-identical logits): the vmcnt(0) drains
 * that remain in the big loop (39 per glu body) are the DEPENDENT A-GATHER loads
 * (row_token[...] -> A row), which the asm diff mis-attributed to weight-stream depth.
 * No register pipeline of the B stream removes a dependent gather. This confirms the D2
 * triple-falsification at the ISA level: the grouped-MoE prefill pair is gather/scatter
 * bound, not weight-pipeline bound. Kept as the documented restructure; the real MoE lever
 * is a gather-hiding or scatter-reducing rewrite, not deeper B prefetch. */
#ifndef PLOW_MOE_PF_PIPE
#define PLOW_MOE_PF_PIPE 0
#endif

/* PLOW_MOE_PF_GH — GATHER HIDING for the grouped-prefill A-gather (opt-in build axis,
 * default OFF; wired from scripts/build_gfx942.sh like PLOW_MOE_PF_SCHED / _PIPE).
 *
 * WHAT IT ATTACKS. The register-pipeline record above localized the pair's remaining
 * `vmcnt(0)` drains to the DEPENDENT A-gather, and the ISA says exactly that: the shipped
 * k-loop opens
 *      flat_load_dword v14, v[182:183]   ; row_token[rowbase + r]
 *      s_waitcnt vmcnt(0)                ; <-- FULL DRAIN, on one dword
 *      v_mad_u64_u32 ... v14 ...         ; the A row address
 *      global_load_dwordx4 ...           ; the A row itself
 * once PER K-TILE. The index load and the data load it addresses cannot overlap — that is a
 * true dependency — so every k-tile pays a whole round trip before its A row can even issue.
 *
 * WHY IT IS REDUNDANT WORK, NOT MERELY BADLY ORDERED WORK. `row_token[rowbase + r]` is
 * K-INVARIANT: `r` is `threadIdx.x / (MPF_BK/8)` and `rowbase` is fixed for the output tile,
 * so all NT k-tiles load THE SAME DWORD FROM THE SAME ADDRESS. LLVM cannot hoist it itself —
 * the k-loop is fenced by `__syncthreads()`, a memory barrier the load may not move across.
 * The MPF4 (a4w4) body already resolves its gather row once per tile for exactly this reason
 * (see its `asrc_` note); this is that hoist applied to the bf16/fp8 body.
 *
 *   PLOW_MOE_PF_GH=1  hoist only. The index resolves ONCE per output tile; NT-1 dependent
 *                     index loads and their `vmcnt(0)` drains disappear, and the A row
 *                     address becomes a register-resident base for the whole k-loop.
 *   PLOW_MOE_PF_GH=2  hoist + software pipeline. The one surviving index load is issued a
 *                     full OUTPUT TILE ahead: tile i's head loads the indices tile i+nblk
 *                     will gather with, so its latency is covered by tile i's entire k-loop
 *                     and the next tile's gather issues from a register with no exposed
 *                     wait. The prefetch is issued BEFORE the `wb0 == 0` EP skip so the
 *                     pipeline stays in step across skipped tiles; `d_moe_align_pf`
 *                     initialises row_token over the WHOLE padded range (`total_pad =
 *                     tilep[n_exp] * MPF_BM`), so the read is in bounds for every tile index
 *                     this loop can visit.
 *
 * VALUE-IDENTICAL at both levels: same addresses, same dwords, same A rows, same LDS cells.
 * GLU arm only — DOWN's A row is `rowbase + r` with no indirection, so the hoist costs it
 * nothing and changes nothing. */
#ifndef PLOW_MOE_PF_GH
#define PLOW_MOE_PF_GH 0
#endif

/* Both arms below stage/resolve A by ALL threads at the 8-half quantum (PIPE also hardcodes
 * the BK=64 promotion cadence, kb = KTV >> 1), so they are refused with the sub-quantum tile
 * rather than left silently wrong. Both are prefill-only opt-ins; the prefill tile is BK=64. */
#if MPF_SUBQ && (PLOW_MOE_PF_PIPE || PLOW_MOE_PF_GH)
#error "sub-quantum grouped-MoE A staging (MPF_SUBQ) does not support PLOW_MOE_PF_PIPE/GH"
#endif
#if PLOW_MOE_PF_PIPE && MPF_BK != 64
#error "PLOW_MOE_PF_PIPE hardcodes the BK=64 promotion cadence (kb = kt >> 1)"
#endif

/* PLOW_MOE_PF_ATOMIC -- FUSE ops 86 -> 87 by ACCUMULATING the routed-expert outputs in place
 * instead of materialising the per-slot `part` buffer (opt-in build axis, default OFF; wired
 * from scripts/build_gfx942.sh like PLOW_MOE_PF_EPI, and additionally gated per-packet by
 * `i[4]` so an object that HAS the arm still runs an unflagged blob verbatim).
 *
 * WHAT IT REMOVES. Today op 86 scatters `part[row_partidx[row]][H]` as f32 and op 87 reads all
 * `k` slots of a token back to sum them. At T=8192, TP8, H=6144, k=8 that is
 * `T*k*H*4` = 1.611 GB WRITTEN by 86 and 1.611 GB READ by 87, per layer per rank -- against a
 * 2.848 GB pair footprint that does not even count 87's read. aiter's fused
 * `fmoe_..._g1u1` never writes it: disassembling the shipped gfx942 object
 * `fmoe_fp8_blockscale_g1u1_subGU_256.co` shows 96 `global_atomic_pk_add_bf16` and no
 * intermediate scatter at all. This is that decomposition.
 *
 * WHY IT MUST BE AN ATOMIC AND NOT AN LDS BRIDGE. The k slots of one token are produced by k
 * DIFFERENT experts, so in the expert-sorted row order they live in k different m-tiles and, in
 * general, k different workgroups on k different CUs. No LDS is shared between them; the k-way
 * reduction is inherently cross-workgroup. The only two implementations are a second pass (which
 * is what op 87 IS) or a global atomic. See glm52-moe-fusion.md for the LDS arithmetic that also
 * rules out the 85->86 bridge (it needs 73,728 B against a 64,512 B arena).
 *
 * THE DECOMPOSITION.
 *   op 83 (router) zeroes `acc[T*H]` f32 as a grid-strided prologue -- 201 MB, ~59 us/layer at
 *         stream rate, on a packet that already precedes align -> GLU -> DOWN.
 *   op 86 (DOWN)   `atomicAdd(&acc[tok*H + nn], gate * value)` instead of the non-temporal store
 *         to `part[pidx*H + nn]`. `pidx = tok*k + slot` by construction (d_moe_align_pf), so
 *         `tok = pidx >> log2(k)` -- one shift, no extra load and no division. The arm therefore
 *         requires a power-of-two k, which the emit side asserts.
 *   op 87 (combine) is UNCHANGED code: it is emitted with `k = 1` against the same buffer, so it
 *         reads ONE contiguous f32 stream instead of k streams at H*4 = 24 KB stride.
 *
 * WHY THIS IS NOT PLOW_MOE_PF_PART16 AGAIN. part16 halved the same stream's BYTES and measured
 * ~0% (glm52-moepf-activation-arms.md) because it left the SHAPE alone: op 87 still issued k
 * strided loads per output element and op 86 still issued one store per element. This changes the
 * shape -- 87 goes from k=8 strided streams to one contiguous stream -- and that is a different
 * hypothesis, not a repeat of a falsified one.
 *
 * NUMERICS: NOT bit-identical, and the class is stated plainly. The k slot values are summed in
 * f32 (as today) but in NON-DETERMINISTIC ORDER (atomic arrival), where the shipped combine sums
 * them in fixed slot order. Same precision, different association => differences at the f32 ulp,
 * then rounded to bf16 for the layer output. This is a strictly gentler class than part16, which
 * rounded every slot to bf16 BEFORE summing (and flipped top-1). It is also RUN-TO-RUN
 * nondeterministic, which the shipped path is not. */
#ifndef PLOW_MOE_PF_ATOMIC
#define PLOW_MOE_PF_ATOMIC 0
#endif
/* The trailing argument, present only when the axis is compiled in — see the signature note. */
#if PLOW_MOE_PF_ATOMIC
#define MPF_ATOM_ARG , atom_ksh
#else
#define MPF_ATOM_ARG
#endif

/* PLOW_MOE_PF_DET -- the DETERMINISTIC form of the same 86->87 fusion (opt-in build axis,
 * default OFF, mutually exclusive with PLOW_MOE_PF_ATOMIC).
 *
 * WHY IT EXISTS. PLOW_MOE_PF_ATOMIC's win is real and its numerics are not: the k slot values
 * of a token are summed in f32 in ATOMIC ARRIVAL ORDER, so the arm is neither bit-identical to
 * the shipped combine nor reproducible against ITSELF (glm52-moe-fusion.md 9). The trace also
 * shows the atomic itself buys nothing -- it COSTS 44.8 us/layer -- and that the entire
 * -980 us/layer lives in op 87 reading ONE accumulator stream instead of k slot streams. So the
 * fusion does not need atomic ARITHMETIC; it needs a cross-workgroup accumulator whose value
 * does not depend on arrival order.
 *
 * THE MECHANISM. f32 addition is commutative but NOT associative, which is the whole defect.
 * INTEGER addition is both. So this arm accumulates each contribution as an INTEGER-VALUED f64
 * -- rint(gate * value * 2^32) -- and adds it with a device-scope f64 atomic. Every partial
 * sum is an integer of magnitude < 2^53, i.e. EXACT in f64, so every add is exact and the total
 * is INDEPENDENT OF ARRIVAL ORDER by construction, not by luck. op 87 scales by 2^-32 on the
 * way out. Run-to-run bit-reproducibility is therefore a property of the arithmetic, not of the
 * scheduler.
 *
 * THE EXACTNESS BOUND, stated as an inequality rather than a hope. Contributions are clamped to
 * +-MPF_DET_CLAMP = 2^17 before scaling, so |addend| <= 2^49 and the k-way total is bounded by
 * k * 2^49 <= 2^53 for k <= 16 -- inside f64's exact-integer range. The emit refuses k > 16.
 * Quantisation is 2^-32 = 2.3e-10 ABSOLUTE, against a layer output that is rounded to bf16
 * (8 mantissa bits) two lines later; a contribution below ~1e-9 loses relative precision it
 * could not have carried into the output anyway.
 *
 * WHAT IT IS NOT. It is NOT bit-identical to the shipped combine, and it cannot be: the shipped
 * combine sums the k slots in SLOT order (router score rank), whereas any in-place accumulator
 * receives them in ROW order (expert id). Those two orders are uncorrelated, and enforcing slot
 * order across workgroups costs either a stall or the `part` DRAM pass this fusion exists to
 * delete -- see glm52-moe-deterministic-writer.md for the arithmetic on both. What this arm
 * gives up is bit-identity with a particular association; what it gains over the atomic arm is
 * that its own result is FIXED. */
#ifndef PLOW_MOE_PF_DET
#define PLOW_MOE_PF_DET 0
#endif
#if PLOW_MOE_PF_DET && PLOW_MOE_PF_ATOMIC
#error "PLOW_MOE_PF_DET and PLOW_MOE_PF_ATOMIC are two writers for one accumulator - pick one"
#endif
#if PLOW_MOE_PF_DET
#define MPF_DET_ARG , det_ksh
/* 2^32 as the fixed-point scale and 2^17 as the pre-scale clamp: see the exactness bound above.
 * Both are exact powers of two, so the scale multiply and the op-87 unscale are exact. */
#define MPF_DET_SCALE 4294967296.0
#define MPF_DET_UNSCALE 2.3283064365386962890625e-10
#define MPF_DET_CLAMP 131072.0f
/* One contribution, in the fixed-point domain. The clamp is what makes the k-way bound above an
 * inequality instead of an assumption; it is 2^17 against activations that live near 1, so it is
 * a guard, not a transform. NaN clamps (IEEE maxNum returns the non-NaN operand) rather than
 * poisoning the whole accumulator row -- which is a DIFFERENCE from the shipped path, and the
 * only one that is not a pure re-association. */
__device__ __forceinline__ double mpf_det_q(float v) {
    return __builtin_rint((double)fminf(fmaxf(v, -MPF_DET_CLAMP), MPF_DET_CLAMP) * MPF_DET_SCALE);
}
#else
#define MPF_DET_ARG
#endif

/* PLOW_MOE_PF_EPI — DOWN-EPILOGUE ROW-METADATA HOIST (opt-in build axis, default OFF;
 * wired from scripts/build_gfx942.sh like PLOW_MOE_PF_SCHED / _PIPE / _GH).
 *
 * WHAT IT ATTACKS. Disassembled from the shipped megakernel
 * (`interp_prefill_fp8_mla_moe.elf`, `_Z19d_moe_group_down_pf...` @ 0x38114), the DOWN
 * per-output-tile epilogue `0x3b81c..0x3d730` is 1,310 instructions carrying 128
 * `flat_load_dword` and 128 full `s_waitcnt vmcnt(0) lgkmcnt(0)` at MAX OUTSTANDING 1:
 *      flat_load_dword v16, v[28:29]     ; row_partidx[rowbase + rr]
 *      s_waitcnt vmcnt(0) lgkmcnt(0)     ; <-- FULL DRAIN, on one dword
 *      v_cmp_ne_u32 -1, v16              ; the PLOW_EXPERT_UNUSED test
 *      flat_load_dword v25, v[30:31]     ; row_gate[rowbase + rr]
 *      s_waitcnt vmcnt(0) lgkmcnt(0)     ; <-- FULL DRAIN, on one dword
 *      global_store_short ... nt         ; one output element
 * repeated once per output ELEMENT (64 run, 64 sit in the other store-width branch).
 *
 * WHY IT IS REDUNDANT WORK. `row_partidx[rowbase + rr]` / `row_gate[rowbase + rr]` depend on
 * neither k nor n: `rowbase` is the m-tile's padded row start and `rr < MPF_BM` is a pure
 * lane/element function. The same MPF_BM = 64 rows -- 512 bytes -- are re-fetched once per
 * element, for every one of the `tn = N/MPF_BN = 24` n-tiles of every m-tile. At T=8192 that
 * is 27,648 output tiles x 64 = 1.77 M serialized dependent round trips per layer per rank.
 * LLVM cannot hoist them itself: the k-loop between the address and the use is fenced by
 * `__syncthreads()`, which the load may not move across.
 *
 * THE HOIST. `MPF_BM == PLOW_WAVE == 64`, so one wave holds the whole row block one row per
 * lane. Lane L loads `(row_partidx[rowbase+L], row_gate[rowbase+L])` -- two coalesced
 * wave-wide dwords, both in flight at once -- and the epilogue reads row `rr` back out of
 * lane `rr` with `ds_bpermute_b32`. Wave-local: no LDS cell (the tile already sits at 64,560
 * of 65,536 B), no barrier, and 2 loop-carried VGPRs rather than the 32 an
 * `unsigned[16] + float[16]` per-element cache would cost at a 256-VGPR/occupancy-2 kernel.
 * 64 exposed round trips per output tile -> 2, and at EPI=1 those 2 are issued before the
 * k-loop so its NT k-tiles cover them.
 *
 *   PLOW_MOE_PF_EPI=1  hoist, issued at tile head (latency covered by the k-loop).
 *   PLOW_MOE_PF_EPI=2  hoist, issued just before the epilogue. Same 64 -> 2 round-trip
 *                      collapse with no value live across the k-loop -- the control for
 *                      "did the 2 extra loop-carried VGPRs cost more than the hiding won".
 *
 * BYTE-IDENTICAL: the lanes collectively read exactly the address set the shipped epilogue
 * read (rows `rowbase + 0 .. rowbase + MPF_BM-1`, which `d_moe_align_pf` initialises over the
 * whole padded range), the same dwords reach the same elements, and the arithmetic is
 * untouched. DOWN arm only -- GLU passes `row_partidx = row_gate = nullptr`. */
#ifndef PLOW_MOE_PF_EPI
#define PLOW_MOE_PF_EPI 0
#endif
#if PLOW_MOE_PF_EPI && (MPF_BM != PLOW_WAVE)
#error "PLOW_MOE_PF_EPI assumes one wave covers the m-tile's rows (MPF_BM == PLOW_WAVE)"
#endif

/* PLOW_MOE_PF_EPI_SIB -- THE SAME HOIST AT THE TWO SIBLING SITES (opt-in, default OFF).
 *
 * `PLOW_MOE_PF_EPI` above acts on `d_moe_group_pf_t`, the grouped fp8/bf16 prefill GEMM GLM
 * and DeepSeek run. TWO other DOWN epilogues in this file carry the IDENTICAL pattern --
 * `row_partidx[rowbase + rr]` and `row_gate[rowbase + rr]`, both k- and n-invariant, re-read
 * once per OUTPUT ELEMENT with a full `s_waitcnt vmcnt(0)` between each -- and were left alone
 * when that landed:
 *
 *   * `d_moe_group_pf_a4w4` (PLOW_MOE_PF_A4W4, native CDNA4 and simulated CDNA3).
 *   * `d_moe_group_gemma_pf_t` (ops 75/76 bf16 and 81/82 w8a8, the Gemma-4 MoE twin). Its
 *     W8A8 arm reads a THIRD k/n-invariant per-row quantity, `ascale[rowbase + rr]`, so this
 *     hoist takes all three: leaving one behind would leave the per-element drain in place and
 *     the collapse would not happen.
 *
 * Mechanism, register cost, EXEC discipline and the byte-identity argument are all exactly the
 * PLOW_MOE_PF_EPI note above -- read it, this is the same transform applied twice more. It is
 * a SEPARATE flag so the GLM canonical object recipe (`PLOW_MOE_PF_EPI=1`) is unperturbed by
 * a change to kernels GLM never dispatches.
 *
 * GATED ON THE MODEL THAT USES EACH PATH, not on GLM: neither site is on GLM's hot path. */
#ifndef PLOW_MOE_PF_EPI_SIB
#define PLOW_MOE_PF_EPI_SIB 0
#endif
#if PLOW_MOE_PF_EPI_SIB && (MPF_BM != PLOW_WAVE)
#error "PLOW_MOE_PF_EPI_SIB assumes one wave covers the m-tile's rows (MPF_BM == PLOW_WAVE)"
#endif

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
                                     unsigned n_group = 0, unsigned topk_group = 0
#if PLOW_MOE_PF_ATOMIC
                                     ,
                                     float* acc = nullptr, unsigned acc_n = 0
#endif
#if PLOW_MOE_PF_DET
                                     /* PLOW_MOE_PF_DET: the same prologue against an f64
                                      * accumulator -- 8 bytes per element, not 4. INSIDE the
                                      * #if for the mangled-name reason the atomic arm records. */
                                     ,
                                     double* acc = nullptr, unsigned acc_n = 0
#endif
                                     ) {
#if PLOW_MOE_PF_ATOMIC
    /* PLOW_MOE_PF_ATOMIC: zero the [T,H] f32 accumulator op 86 will atomically add into. This
     * packet is the earliest one in the MoE chain (router -> align -> GLU -> DOWN), so the
     * existing dependency edges already order it; no new packet and no new gate. ~201 MB at
     * T=8192/H=6144 = ~59 us/layer of pure stream, non-temporal because nothing reads these
     * lines until op 87, a full 304-CU round later. The `acc == nullptr` case is every
     * unflagged blob and compiles to the shipped loop with one predicated branch in front. */
    if (acc) {
        const size_t total = (size_t)T * acc_n;
        for (size_t i = (size_t)slice * PLOW_THREADS + threadIdx.x; i < total;
             i += (size_t)nblk * PLOW_THREADS)
            __builtin_nontemporal_store(0.0f, as_glob(acc) + i);
        __syncthreads(); /* the token loop below reuses `lds`; keep the phases separate */
    }
#endif
#if PLOW_MOE_PF_DET
    /* PLOW_MOE_PF_DET: identical prologue, f64 accumulator (402 MB at T=8192/H=6144 against the
     * atomic arm's 201 MB -- the price of an order-independent accumulator, priced in the
     * report). Non-temporal for the same reason: nothing reads these lines until op 87. */
    if (acc) {
        const size_t total = (size_t)T * acc_n;
        for (size_t i = (size_t)slice * PLOW_THREADS + threadIdx.x; i < total;
             i += (size_t)nblk * PLOW_THREADS)
            __builtin_nontemporal_store(0.0, as_glob(acc) + i);
        __syncthreads(); /* the token loop below reuses `lds`; keep the phases separate */
    }
#endif
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

/* HALVED fp8 decode for the CDNA3 B stage — ASM-probed at 14 VALU per 4 bytes against the
 * shipping plow_fp8x4_ocp_to_bf16 chain's 21: the OCP x2 is NOT applied here (gfx942's
 * cvt_pk_f32_fp8 decodes FNUZ = OCP/2, and the promotion below folds the x2 into the block
 * scale instead — one scalar mul per 128-K block, not one per weight), and the bf16 pack is
 * one v_perm_b32 per pair instead of lshr+and+or. BIT-IDENTICAL end to end: e4m3/2 is exact
 * in bf16 (exponent shift only), sum(w_i/2 * a_i) == sum(w_i * a_i)/2 exactly in f32 (a
 * power-of-two scale commutes with addition), and 2*bs is exact in f32. The 0x80 neg-0 mask
 * is DROPPED here (6 VALU per 4 bytes) — the loader scrubs expert payloads at stage time, see
 * the note inside mpf_fp8x4_to_bf16_h. CDNA3 ONLY: gfx950's decode is OCP-native, so
 * doubling the scale there would be a 2x error. */
/* Scheduler-shaping axis for the grouped k-loop — see the sched_group_barrier note at the
 * MFMA site. Default off: the barriers change instruction ORDER only (bit-identical), but
 * order is exactly what an A/B must judge. */
#ifndef PLOW_MOE_PF_SCHED
#define PLOW_MOE_PF_SCHED 0
#endif
#if PLOW_CDNA4
#define MPF_W_HALF 0
/* CDNA4 twins of the halved helpers below, SAME signatures and element order, so the fetch
 * macros compile on both arches. No halving here: gfx950's convert decodes OCP e4m3 directly
 * (the FNUZ=OCP/2 identity is CDNA3 silicon), so these decode full values and the epilogue's
 * (MPF_W_HALF ? 2.0f : 1.0f) stays 1.0. scale=1.0 for the reason fp8_to_bf16v8 gives: the
 * block scale folds into the f32 epilogue, never the convert. These used to not exist at all
 * — the a8 branch is a runtime `if` inside d_moe_group_pf_t, so a toolchain that instantiates
 * eagerly (ROCm 7.2.3) refused EVERY CDNA4 object over them, while 7.2.4+ only compiled them
 * out of never-emitted instantiations. */
__device__ __forceinline__ void mpf_fp8x4_to_bf16_h(unsigned w, unsigned& lo, unsigned& hi) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    const bf16_2 a = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w, 1.0f, false); /* bytes 0,1 */
    const bf16_2 c = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w, 1.0f, true);  /* bytes 2,3 */
    lo = __builtin_bit_cast(unsigned, a);
    hi = __builtin_bit_cast(unsigned, c);
}
__device__ __forceinline__ void mpf_fp8v16_to_bf16_h(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    fp8_to_bf16v8(w, lo, hi);
}
#else
#define MPF_W_HALF 1
__device__ __forceinline__ unsigned mpf_perm_pack(float lo, float hi) {
    unsigned a, b;
    __builtin_memcpy(&a, &lo, 4);
    __builtin_memcpy(&b, &hi, 4);
    return __builtin_amdgcn_perm(b, a, 0x07060302u); /* (a>>16) | (b & 0xffff0000) */
}
__device__ __forceinline__ void mpf_fp8x4_to_bf16_h(unsigned w, unsigned& lo, unsigned& hi) {
    /* NO neg-0 mask (ASM-probed: 6 VALU per 4 bytes against the masked form's 14). The loader
     * SCRUBS 0x80 -> 0x00 out of every block-fp8 expert payload as it stages it
     * (`scrub_fp8_neg0` in plowrt exec/amd.rs, one SWAR pass inside the pinned-slab copy;
     * value-identical, -0 == +0 in every product), so the FNUZ decoder's one poison byte can
     * never reach this path. That contract is exactly as wide as this helper's use: the B
     * operands of ops 85/86, which are always expert-table payloads. Anything decoding
     * UN-scrubbed checkpoint bytes must keep plow_fp8_mask_neg0. */
    const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(w, false); /* bytes 0,1 */
    const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(w, true);  /* bytes 2,3 */
    lo = mpf_perm_pack(a[0], a[1]);
    hi = mpf_perm_pack(c[0], c[1]);
}
/* fp8v16 form of the above: 16 HALVED bf16 weights, same element order as fp8_to_bf16v8. */
__device__ __forceinline__ void mpf_fp8v16_to_bf16_h(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    union { bf16v8 v; unsigned u[4]; } ol, oh;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        auto& o = (i < 2) ? ol : oh;
        const int b = (i & 1) * 2;
        mpf_fp8x4_to_bf16_h((unsigned)w[i], o.u[b + 0], o.u[b + 1]);
    }
    lo = ol.v;
    hi = oh.v;
}
#endif

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
                                 float beta, float lbeta, unsigned shuf, bf16* lds,
                                 /* ACTIVATION-SIDE arms (the pair's real cost — see the
                                  * preshuffle record's corrected traffic model). Both are
                                  * runtime flags carried on the PACKET, so one object serves
                                  * old and new blobs; the loader refuses old OBJECTS via the
                                  * plow_moe_pf_*_arm markers below.
                                  *   a8 (GLU only): A rows are fp8 (aq) + per-token f32 scale
                                  *     (as_row), written by d_rmsnorm's fused quant. Decoded
                                  *     with the maskless chain (the encoder never emits 0x80),
                                  *     which yields OCP/2 — the x2 folds into as_row at the
                                  *     epilogue, before the GLU nonlinearity.
                                  *   part16 (DOWN only): the part scatter is bf16, halving the
                                  *     pair's largest stream and the combine's readback. */
                                 const unsigned char* __restrict__ aq = nullptr,
                                 const float* __restrict__ as_row = nullptr, unsigned a8 = 0,
                                 unsigned part16 = 0
#if PLOW_MOE_PF_ATOMIC
                                 /* PLOW_MOE_PF_ATOMIC (DOWN only): 0 = the shipped `part`
                                  * scatter; otherwise log2(k)+1, and Cout is the [T,H] f32
                                  * accumulator op 87 reads with k=1. See the header note.
                                  * INSIDE the #if so the DEFAULT object's mangled names -- and
                                  * therefore its .strtab, and therefore every byte of it -- are
                                  * unchanged by this commit. */
                                 ,
                                 unsigned atom_ksh = 0
#endif
#if PLOW_MOE_PF_DET
                                 /* PLOW_MOE_PF_DET (DOWN only): 0 = the shipped `part` scatter;
                                  * otherwise log2(k)+1, and Cout is the [T,H] f64 fixed-point
                                  * accumulator op 87 reads with k=1. Same #if placement rule. */
                                 ,
                                 unsigned det_ksh = 0
#endif
                                 ) {
    constexpr int SM = MPF_SM, SN = MPF_SN;
    constexpr int APT = MPF_BM * MPF_BK / PLOW_THREADS; /* 8  halves/thread */
    constexpr int BPT = MPF_BN * MPF_BK / PLOW_THREADS; /* 32 halves/thread */
#if MPF_SUBQ
    /* Sub-quantum A tile (APT < 8) — the masked arm, see the MPF_SUBQ header note. Waves
     * 0 .. AACT/64-1 stage the whole A tile at the full quantum in one pass; the guard is
     * wave-uniform and encloses no barrier. B is unaffected (BPT stays quantum-sized). */
    constexpr int APASS = 1, BPASS = BPT / 8;
    constexpr unsigned AACT = MPF_BM * MPF_BK / 8u; /* A-staging threads */
    static_assert(AACT * 8u == MPF_BM * MPF_BK && APT > 0, "masked A pass must cover the tile");
    static_assert(AACT % PLOW_WAVE == 0 && AACT < PLOW_THREADS,
                  "masked A staging must be wave-uniform");
    static_assert(BPT >= 8 && BPT % 8 == 0, "grouped-MoE B tile below the 8-half staging quantum");
#else
    constexpr int APASS = APT / 8, BPASS = BPT / 8;
    /* The staging quantum is one 16-byte vector (8 halves) per thread per pass. A tile whose
     * per-thread share is below that truncates APASS/BPASS to ZERO and the loop stages
     * NOTHING — the GEMM then runs on whatever the arena held, fluently and wrongly. That is
     * not hypothetical: MPF_BM=64/MPF_BK=32 shipped exactly this (APT=4, APASS=0) and cost a
     * day of bisecting. Tiles with BM*BK < 8*PLOW_THREADS take the MPF_SUBQ masked arm. */
    static_assert(APT >= 8 && APT % 8 == 0, "grouped-MoE A tile below the 8-half staging quantum");
    static_assert(BPT >= 8 && BPT % 8 == 0, "grouped-MoE B tile below the 8-half staging quantum");
#endif
    /* Output columns a tile emits: GLU fuses two B halves into one column block. */
    constexpr unsigned NB = GLU ? (MPF_BN / 2) : MPF_BN;
    constexpr unsigned DB = MPF_DBUF; /* see the MPF_DBUF header: 1 on a 64 KiB part */

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

/* The gathered row of A pass `it`, for the tile whose padded start is `rb`. One expression,
 * used by the hoist site below AND by the GH=2 prefetch, so the two can never disagree about
 * which dword a pass consumes. */
#define MPF_GROW(rb, it) \
    row_token[(rb) + (threadIdx.x * 8u + (unsigned)(it) * (PLOW_THREADS * 8u)) / MPF_BK]

#if PLOW_MOE_PF_EPI
/* The hoisted fetch itself (see the PLOW_MOE_PF_EPI header note). One statement, used by both
 * the EPI=1 tile-head site and the EPI=2 pre-epilogue site so the two cannot disagree about
 * which dword a lane owns. as_glob keeps these off the flat path, so they carry no lgkmcnt. */
#define MPF_EPI_LOAD                                                                           \
    epi_pidx = as_glob(row_partidx)[rowbase + lane];                                           \
    epi_gate = __builtin_bit_cast(int, as_glob(row_gate)[rowbase + lane]);
#endif

#if PLOW_MOE_PF_GH >= 2
    /* One output tile of gather lookahead. APASS == 1 at the CDNA3 tile, so this is a single
     * loop-carried VGPR. Primed for the first tile this workgroup owns; a workgroup with no
     * tile never enters the loop and never reads it. */
    unsigned ghn_[APASS];
    if constexpr (GLU) {
#pragma unroll
        for (int it = 0; it < APASS; it++) ghn_[it] = PLOW_EXPERT_UNUSED;
        if (slice < n_tiles) {
            const unsigned mt0 = slice / tn;
            const unsigned e0 = mpf_expert_of_tile(tilep, mt0, n_exp);
            const unsigned rb0 = (unsigned)rowoff[e0] + (mt0 - (unsigned)tilep[e0]) * MPF_BM;
#pragma unroll
            for (int it = 0; it < APASS; it++) ghn_[it] = MPF_GROW(rb0, it);
        }
    }
#endif

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned mt = lin / tn, nt = lin % tn;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF_BM;
        const unsigned n0 = nt * NB;

#if PLOW_MOE_PF_GH
        /* ---- GATHER HIDING (see the PLOW_MOE_PF_GH header note) ----
         * The A-gather row, resolved ONCE for this whole output tile instead of once per
         * k-tile. Under GH=2 it was already loaded a tile ago and this is a register read. */
        unsigned asrc_[APASS];
#pragma unroll
        for (int it = 0; it < APASS; it++) {
            if constexpr (!GLU) {
                asrc_[it] = rowbase +
                            (threadIdx.x * 8u + (unsigned)it * (PLOW_THREADS * 8u)) / MPF_BK;
            } else {
#if PLOW_MOE_PF_GH >= 2
                asrc_[it] = ghn_[it];
#else
                asrc_[it] = MPF_GROW(rowbase, it);
#endif
            }
        }
#if PLOW_MOE_PF_GH >= 2
        /* ISSUE THE NEXT TILE'S INDICES NOW — before the weight-table read, before the EP
         * skip, before any k-loop traffic — so the whole tile below is what covers them.
         * Unconditional w.r.t. the skip below on purpose: the pipeline is indexed by LOOP
         * ITERATION, not by tiles that survive the skip. */
        if constexpr (GLU) {
            const unsigned lin2 = lin + nblk;
            if (lin2 < n_tiles) {
                const unsigned mt2 = lin2 / tn;
                const unsigned e2 = mpf_expert_of_tile(tilep, mt2, n_exp);
                const unsigned rb2 = (unsigned)rowoff[e2] + (mt2 - (unsigned)tilep[e2]) * MPF_BM;
#pragma unroll
                for (int it = 0; it < APASS; it++) ghn_[it] = MPF_GROW(rb2, it);
            }
        }
#endif
#endif

        /* Weight + scale bases for THIS tile's expert. A null base is the EP "not my expert"
         * sentinel the decode ops already honour; skip rather than fault. */
        const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
        if (wb0 == 0ull) continue;
        const unsigned char* W0 = (const unsigned char*)(size_t)wb0;
        const unsigned char* W1 =
            GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : nullptr;
        const float* S0 = FP8 ? (const float*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)] : nullptr;
        const float* S1 = (FP8 && GLU) ? (const float*)(size_t)stab[(size_t)e * 3 + 1] : nullptr;

#if PLOW_MOE_PF_EPI
        /* ---- DOWN-EPILOGUE ROW-METADATA HOIST (see the PLOW_MOE_PF_EPI header note) ----
         * Lane L takes row L of this m-tile; the epilogue bpermutes row rr out of lane rr.
         * Both dwords issue back to back with nothing between them, so they are in flight
         * together, and at EPI=1 the whole k-loop below covers them. */
        unsigned epi_pidx = 0u;
        int epi_gate = 0;
#if PLOW_MOE_PF_EPI == 1
        if constexpr (!GLU) {
            MPF_EPI_LOAD
        }
#endif
#endif

/* The A-gather source for pass `it` at tile row `r`. GH=0 expands to the shipped statement
 * VERBATIM, token for token, so the default object is byte-identical. */
#if PLOW_MOE_PF_GH
#define MPF_A_SRC(it, r) const unsigned src = asrc_[it];
#else
#define MPF_A_SRC(it, r)                                                                       \
    unsigned src;                                                                              \
    if constexpr (GLU) src = row_token[rowbase + (r)];                                         \
    else src = rowbase + (r);
#endif

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

#if PLOW_MOE_PF_PIPE
        /* ---- two-tile register pipeline (see the PLOW_MOE_PF_PIPE header note) ----
         * Stage s holds tile t where t % 2 == s, RAW: A as loaded bf16 (or dequanted fp8
         * under a8 — small), B as raw fp8 bytes (fp8v16) or loaded bf16. The k-loop is
         * unrolled x2 so every stage array is indexed by a compile-time constant. */
        __align__(16) bf16 ra0[APT], ra1[APT];
        fp8v16 rbq0[FP8 ? BPASS / 2 : 1], rbq1[FP8 ? BPASS / 2 : 1];
        __align__(16) bf16 rbh0[FP8 ? 1 : BPT], rbh1[FP8 ? 1 : BPT];

#define MPF_FETCH_A_S(k0, ras)                                                                 \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        const unsigned r = el / MPF_BK, kk = (k0) + (el % MPF_BK);                              \
        MPF_A_SRC(it, r)                                                                        \
        if (src == PLOW_EXPERT_UNUSED) {                                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) ras[it * 8 + j] = 0;                  \
        } else if (GLU && a8) {                                                                 \
            const fp8v8 w8 = ld_glob_fp8v8(aq + (size_t)src * K + kk);                          \
            union { bf16v8 v; unsigned u[4]; } d8;                                              \
            mpf_fp8x4_to_bf16_h((unsigned)w8[0], d8.u[0], d8.u[1]);                             \
            mpf_fp8x4_to_bf16_h((unsigned)w8[1], d8.u[2], d8.u[3]);                             \
            *(bf16v8*)&ras[it * 8] = d8.v;                                                      \
        } else {                                                                                \
            *(bf16v8*)&ras[it * 8] = ld_glob8(as_glob(A) + (size_t)src * K + kk);               \
        }                                                                                       \
    }

/* B fetch stays RAW — the dequant happens at commit so the loads have no consumer until
 * the other stage's loads are already in flight. Address selection (shuf / GLU up half /
 * r >= N zero-fill) is byte-for-byte the legacy macro's. fp8 0x00 decodes to 0.0 on both
 * cvt arms, so the raw zero-fill lands the same LDS zeros the legacy converted fill did. */
#define MPF_FETCH_B_S(k0, rbqs, rbhs)                                                          \
    if constexpr (FP8) {                                                                       \
        _Pragma("unroll") for (int it = 0; it < BPASS / 2; it++) {                              \
            const unsigned el = threadIdx.x * 16 + it * (PLOW_THREADS * 16);                    \
            const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                         \
            unsigned r = n0 + br;                                                               \
            const unsigned char* wsrc = W0;                                                     \
            if constexpr (GLU) {                                                                \
                const bool up = (br >= MPF_BN / 2);                                             \
                wsrc = up ? W1 : W0;                                                            \
                r = n0 + (up ? br - MPF_BN / 2 : br);                                           \
            }                                                                                   \
            if (r < N) {                                                                        \
                const unsigned char* wp =                                                       \
                    shuf ? wsrc + ((((size_t)(k0) >> 6) * N + r) << 6) + (el & 63u)             \
                         : wsrc + (size_t)r * K + kk;                                           \
                rbqs[it] = ld_glob_fp8v16(wp);                                                  \
            } else                                                                              \
                rbqs[it] = (fp8v16)(0u);                                                        \
        }                                                                                       \
    } else {                                                                                    \
        _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                  \
            const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                      \
            const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                         \
            unsigned r = n0 + br;                                                               \
            const unsigned char* wsrc = W0;                                                     \
            if constexpr (GLU) {                                                                \
                const bool up = (br >= MPF_BN / 2);                                             \
                wsrc = up ? W1 : W0;                                                            \
                r = n0 + (up ? br - MPF_BN / 2 : br);                                           \
            }                                                                                   \
            if (r < N)                                                                          \
                *(bf16v8*)&rbhs[it * 8] =                                                       \
                    ld_glob8(as_glob((const bf16*)wsrc) + (size_t)r * K + kk);                  \
            else                                                                                \
                _Pragma("unroll") for (int j = 0; j < 8; j++) rbhs[it * 8 + j] = 0;             \
        }                                                                                       \
    }

#define MPF_COMMIT_S(ras, rbqs, rbhs)                                                          \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        __builtin_memcpy(&MPF_ASM(0)[(el / MPF_BK) * MPF_STRIDE +                               \
                                     MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                     \
                         &ras[it * 8], 16);                                                     \
    }                                                                                           \
    if constexpr (FP8) { /* dequant AT COMMIT — same map, same LDS cells as legacy */           \
        _Pragma("unroll") for (int it = 0; it < BPASS / 2; it++) {                              \
            const unsigned el = threadIdx.x * 16 + it * (PLOW_THREADS * 16);                    \
            const unsigned br = el / MPF_BK, kk = el % MPF_BK;                                  \
            bf16v8 blo, bhi;                                                                    \
            if constexpr (MPF_W_HALF) mpf_fp8v16_to_bf16_h(rbqs[it], blo, bhi);                 \
            else fp8_to_bf16v8(rbqs[it], blo, bhi);                                             \
            __builtin_memcpy(&MPF_BSM(0)[br * MPF_STRIDE + MPF_XORSWZ(br, kk)], &blo, 16);      \
            __builtin_memcpy(&MPF_BSM(0)[br * MPF_STRIDE + MPF_XORSWZ(br, kk + 8)], &bhi, 16);  \
        }                                                                                       \
    } else {                                                                                    \
        _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                  \
            const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                      \
            __builtin_memcpy(&MPF_BSM(0)[(el / MPF_BK) * MPF_STRIDE +                           \
                                         MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                 \
                             &rbhs[it * 8], 16);                                                \
        }                                                                                       \
    }

/* The MFMA + promotion body for tile KTV — the legacy block verbatim, LDS buffer 0. */
#define MPF_MFMA_PROMO(KTV)                                                                    \
    _Pragma("unroll") for (int q = 0; q < MPF_BK / MFMA_K; q++) {                               \
        bf16x8 af[SM], bfr[SN];                                                                 \
        _Pragma("unroll") for (int i = 0; i < SM; i++) {                                        \
            const unsigned arow = wm * (MPF_BM / MPF_WM) + i * MFMA_M + frow;                   \
            __builtin_memcpy(&af[i],                                                            \
                             &MPF_ASM(0)[arow * MPF_STRIDE +                                    \
                                         MPF_XORSWZ(arow, mfma_frag_k(lane, q * MFMA_K))],      \
                             16);                                                               \
        }                                                                                       \
        _Pragma("unroll") for (int j = 0; j < SN; j++) {                                        \
            const unsigned brow = (GLU ? j * (MPF_BN / 2) + wn * MFMA_N                         \
                                       : wn * (MPF_BN / MPF_WN) + j * MFMA_N) +                 \
                                  frow;                                                         \
            __builtin_memcpy(&bfr[j],                                                           \
                             &MPF_BSM(0)[brow * MPF_STRIDE +                                    \
                                         MPF_XORSWZ(brow, mfma_frag_k(lane, q * MFMA_K))],      \
                             16);                                                               \
        }                                                                                       \
        __builtin_amdgcn_s_setprio(1);                                                          \
        _Pragma("unroll") for (int i = 0; i < SM; i++)                                          \
            _Pragma("unroll") for (int j = 0; j < SN; j++)                                       \
                acc[i][j] = plow_mfma_bf16_32x32(af[i], bfr[j], acc[i][j]);                      \
        __builtin_amdgcn_s_setprio(0);                                                          \
    }                                                                                           \
    if constexpr (FP8) {                                                                       \
        if (((KTV) & 1u) == 1u || (KTV) == NT - 1u) {                                           \
            const unsigned kb = (KTV) >> 1;                                                     \
            _Pragma("unroll") for (int i = 0; i < SM; i++)                                      \
                _Pragma("unroll") for (int j = 0; j < SN; j++) {                                 \
                const float* Sj = (GLU && j == 1) ? S1 : S0;                                    \
                const float bs =                                                                \
                    Sj[(size_t)nblk_row[j] * KB + kb] * (MPF_W_HALF ? 2.0f : 1.0f);             \
                accf[i][j] += acc[i][j] * bs;                                                   \
                acc[i][j] = (f32x16)(0.0f);                                                     \
            }                                                                                   \
        }                                                                                       \
    }

        __syncthreads(); /* the previous tile's fragment readers must be done with the LDS */
        MPF_FETCH_A_S(0, ra0) MPF_FETCH_B_S(0, rbq0, rbh0)
        if (1u < NT) { MPF_FETCH_A_S(MPF_BK, ra1) MPF_FETCH_B_S(MPF_BK, rbq1, rbh1) }
        MPF_COMMIT_S(ra0, rbq0, rbh0)
        __syncthreads();
        for (unsigned kt = 0; kt < NT; kt += 2u) {
            /* EVEN: LDS holds tile kt; stage 1 holds kt+1; stage 0 is free. Its refill for
             * kt+2 issues FIRST, so the commit of stage 1 below waits on a PARTIAL vmcnt. */
            if (kt + 2u < NT) {
                MPF_FETCH_A_S((kt + 2u) * MPF_BK, ra0)
                MPF_FETCH_B_S((kt + 2u) * MPF_BK, rbq0, rbh0)
            }
            /* PIN THE ISSUE ORDER. The dataflow above PERMITS a partial wait at the commit,
             * but the scheduler was measured sinking the refill loads below it (register
             * pressure heuristic), which restores the vmcnt(0) drain. This full fence is the
             * legitimate cousin of the PLOW_MOE_PF_SCHED null: that one tried to create depth
             * the dependencies forbade; this one stops the scheduler from DESTROYING depth
             * the restructure created. */
            __builtin_amdgcn_sched_barrier(0);
            MPF_MFMA_PROMO(kt)
            if (kt + 1u < NT) {
                __syncthreads(); /* every reader done with the one buffer */
                MPF_COMMIT_S(ra1, rbq1, rbh1)
                __syncthreads();
                /* ODD: LDS holds kt+1; stage 0 holds kt+2; stage 1 refills for kt+3. */
                if (kt + 3u < NT) {
                    MPF_FETCH_A_S((kt + 3u) * MPF_BK, ra1)
                    MPF_FETCH_B_S((kt + 3u) * MPF_BK, rbq1, rbh1)
                }
                __builtin_amdgcn_sched_barrier(0);
                MPF_MFMA_PROMO(kt + 1u)
                if (kt + 2u < NT) {
                    __syncthreads();
                    MPF_COMMIT_S(ra0, rbq0, rbh0)
                    __syncthreads();
                }
            }
        }
#undef MPF_FETCH_A_S
#undef MPF_FETCH_B_S
#undef MPF_COMMIT_S
#undef MPF_MFMA_PROMO
#else /* !PLOW_MOE_PF_PIPE — the shipped single-stage loop, verbatim */
#if MPF_SUBQ
        __align__(16) bf16 ra[8], rb[BPT]; /* an ACTIVE A-staging thread carries a full quantum */
#else
        __align__(16) bf16 ra[APT], rb[BPT];
#endif

/* Stage A: GLU gathers row_token[rowbase+r] out of xn2; DOWN reads fu_g contiguously from
 * rowbase. An UNUSED (pad) row zero-fills, which contributes nothing to any live output. */
#if MPF_SUBQ
/* Masked arm (MPF_SUBQ): same body at it = 0, staged by waves 0 .. AACT/64-1 only. */
#define MPF_FETCH_A(k0)                                                                        \
    if (threadIdx.x < AACT) {                                                                  \
        const unsigned el = threadIdx.x * 8;                                                    \
        const unsigned r = el / MPF_BK, kk = (k0) + (el % MPF_BK);                              \
        MPF_A_SRC(0, r)                                                                         \
        if (src == PLOW_EXPERT_UNUSED) {                                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) ra[j] = 0;                            \
        } else if (GLU && a8) {                                                                 \
            /* fp8 gathered A: 8 bytes -> 8 HALVED bf16 (maskless — the quantizer never       \
             * emits 0x80); the x2 and the per-token scale apply at the epilogue. */            \
            const fp8v8 w8 = ld_glob_fp8v8(aq + (size_t)src * K + kk);                          \
            union { bf16v8 v; unsigned u[4]; } d8;                                              \
            mpf_fp8x4_to_bf16_h((unsigned)w8[0], d8.u[0], d8.u[1]);                             \
            mpf_fp8x4_to_bf16_h((unsigned)w8[1], d8.u[2], d8.u[3]);                             \
            *(bf16v8*)&ra[0] = d8.v;                                                            \
        } else {                                                                                \
            *(bf16v8*)&ra[0] = ld_glob8(as_glob(A) + (size_t)src * K + kk);                     \
        }                                                                                       \
    }
#else
#define MPF_FETCH_A(k0)                                                                        \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        const unsigned r = el / MPF_BK, kk = (k0) + (el % MPF_BK);                              \
        MPF_A_SRC(it, r)                                                                        \
        if (src == PLOW_EXPERT_UNUSED) {                                                        \
            _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;                   \
        } else if (GLU && a8) {                                                                 \
            /* fp8 gathered A: 8 bytes -> 8 HALVED bf16 (maskless — the quantizer never       \
             * emits 0x80); the x2 and the per-token scale apply at the epilogue. */            \
            const fp8v8 w8 = ld_glob_fp8v8(aq + (size_t)src * K + kk);                          \
            union { bf16v8 v; unsigned u[4]; } d8;                                              \
            mpf_fp8x4_to_bf16_h((unsigned)w8[0], d8.u[0], d8.u[1]);                             \
            mpf_fp8x4_to_bf16_h((unsigned)w8[1], d8.u[2], d8.u[3]);                             \
            *(bf16v8*)&ra[it * 8] = d8.v;                                                       \
        } else {                                                                                \
            *(bf16v8*)&ra[it * 8] = ld_glob8(as_glob(A) + (size_t)src * K + kk);                \
        }                                                                                       \
    }
#endif

/* Stage B: the expert's weight rows [n0, n0+NB) at k0. Under GLU the tile's low half is the
 * gate weight and its high half is the up weight, both at the same output rows. fp8 decodes
 * EXACTLY to bf16 here; the block scale is applied later, to the f32 accumulator.
 *
 * The fp8 arm fetches 16 WEIGHTS PER LOAD (fp8v16 + the packed fp8_to_bf16v8 pair), not 8:
 * B is the stream that costs (the whole-kernel header), and 16 consecutive fp8 bytes stay
 * inside one BK=64 row for every thread (BK/16 = 4 threads per row), so this halves the
 * B-side global-load count and cuts the dequant VALU ~3x (amd_common.h fp8_to_bf16v8) while
 * fetching the SAME bytes to the SAME rb elements — bit-identical by construction. The
 * element->thread map changes with it, so the B half of MPF_COMMIT below mirrors the same
 * map; LDS cell contents are unchanged. */
#define MPF_FETCH_B(k0)                                                                        \
    if constexpr (FP8) {                                                                       \
        _Pragma("unroll") for (int it = 0; it < BPASS / 2; it++) {                              \
            const unsigned el = threadIdx.x * 16 + it * (PLOW_THREADS * 16);                    \
            const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                         \
            unsigned r = n0 + br;                                                               \
            const unsigned char* wsrc = W0;                                                     \
            if constexpr (GLU) {                                                                \
                const bool up = (br >= MPF_BN / 2);                                             \
                wsrc = up ? W1 : W0;                                                            \
                r = n0 + (up ? br - MPF_BN / 2 : br);                                           \
            }                                                                                   \
            if (r < N) {                                                                        \
                /* PRESHUFFLED layout (PLOW_MOE_PF_SHUF, loader-built 2nd slab):                \
                 * B'[K/64][N][64] -- this k-tile's B block is contiguous, full 128 B           \
                 * lines instead of 64 B row-slices at K-stride. Same bytes to the same         \
                 * rb elements => bit-identical. N == the projection buffer's rows on           \
                 * BOTH arms (GLU: per-rank I_moe per gate/up buffer; DOWN: H). The             \
                 * address form is BK-dependent -- see MPF_SHUF_WP at the tile macros. */       \
                const unsigned char* wp =                                                       \
                    shuf ? MPF_SHUF_WP(wsrc, k0, el, kk, r)                                     \
                         : wsrc + (size_t)r * K + kk;                                           \
                const fp8v16 w = ld_glob_fp8v16(wp);                                            \
                if constexpr (MPF_W_HALF)                                                       \
                    mpf_fp8v16_to_bf16_h(w, *(bf16v8*)&rb[it * 16], *(bf16v8*)&rb[it * 16 + 8]); \
                else                                                                            \
                    fp8_to_bf16v8(w, *(bf16v8*)&rb[it * 16], *(bf16v8*)&rb[it * 16 + 8]);       \
            } else                                                                              \
                _Pragma("unroll") for (int j = 0; j < 16; j++) rb[it * 16 + j] = 0;             \
        }                                                                                       \
    } else {                                                                                    \
        _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                  \
            const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                      \
            const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                         \
            unsigned r = n0 + br;                                                               \
            const unsigned char* wsrc = W0;                                                     \
            if constexpr (GLU) {                                                                \
                const bool up = (br >= MPF_BN / 2);                                             \
                wsrc = up ? W1 : W0;                                                            \
                r = n0 + (up ? br - MPF_BN / 2 : br);                                           \
            }                                                                                   \
            if (r < N)                                                                          \
                *(bf16v8*)&rb[it * 8] =                                                         \
                    ld_glob8(as_glob((const bf16*)wsrc) + (size_t)r * K + kk);                  \
            else                                                                                \
                _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;               \
        }                                                                                       \
    }

#if MPF_SUBQ
/* Masked arm (MPF_SUBQ): the same 16-byte store at it = 0, by the A-staging waves only. */
#define MPF_COMMIT_A(buf)                                                                      \
    if (threadIdx.x < AACT) {                                                                  \
        const unsigned el = threadIdx.x * 8;                                                    \
        __builtin_memcpy(&MPF_ASM(buf)[(el / MPF_BK) * MPF_STRIDE +                             \
                                       MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                   \
                         &ra[0], 16);                                                           \
    }
#else
#define MPF_COMMIT_A(buf)                                                                      \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                      \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                          \
        __builtin_memcpy(&MPF_ASM(buf)[(el / MPF_BK) * MPF_STRIDE +                             \
                                       MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                   \
                         &ra[it * 8], 16);                                                      \
    }
#endif

#define MPF_COMMIT(buf)                                                                        \
    MPF_COMMIT_A(buf)                                                                          \
    if constexpr (FP8) { /* same 16-element map as the fp8 MPF_FETCH_B above */                 \
        _Pragma("unroll") for (int it = 0; it < BPASS / 2; it++) {                              \
            const unsigned el = threadIdx.x * 16 + it * (PLOW_THREADS * 16);                    \
            const unsigned br = el / MPF_BK, kk = el % MPF_BK;                                  \
            __builtin_memcpy(&MPF_BSM(buf)[br * MPF_STRIDE + MPF_XORSWZ(br, kk)],               \
                             &rb[it * 16], 16);                                                 \
            __builtin_memcpy(&MPF_BSM(buf)[br * MPF_STRIDE + MPF_XORSWZ(br, kk + 8)],           \
                             &rb[it * 16 + 8], 16);                                             \
        }                                                                                       \
    } else {                                                                                    \
        _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                  \
            const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                      \
            __builtin_memcpy(&MPF_BSM(buf)[(el / MPF_BK) * MPF_STRIDE +                         \
                                           MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],               \
                             &rb[it * 8], 16);                                                  \
        }                                                                                       \
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
                        acc[i][j] = plow_mfma_bf16_32x32(af[i], bfr[j], acc[i][j]);
                __builtin_amdgcn_s_setprio(0);
#if PLOW_MOE_PF_SCHED
                /* OPT-IN (PLOW_MOE_PF_SCHED=1 build axis) — MEASURED NULL-BY-CONSTRUCTION,
                 * kept as the documented do-not-retry. The asm diff against aiter's
                 * fmoe/bf16gemm asm (glm52-asm-innerloop-diff.md) shows their loops keep
                 * 13-39 loads OUTSTANDING (a single partial `vmcnt(22) lgkmcnt(0)` is the
                 * bf16gemm loop's ONLY wait) while ours drains to vmcnt(0) 3-4x per k-tile.
                 * These group barriers were the HIP-level attempt to close that: they moved
                 * exactly 24 instructions (12 ds_read + their address adds) and left every
                 * waitcnt in place, because the waits are DATAFLOW-forced — the loop-head
                 * loads feed this same iteration's LDS commit (single buffer), and no
                 * reordering changes that dependency. aiter's depth needs a multi-tile
                 * register pipeline (their `pf3`) or asm waitcnt control, i.e. the asm-class
                 * restructure, not a scheduler hint. */
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); /* 2 MFMA      */
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); /* 1 VMEM read */
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); /* 2 DS read   */
                __builtin_amdgcn_sched_group_barrier(0x002, 8, 0); /* 8 VALU      */
#endif
            }

            /* PROMOTION. A 128-element K scale block is exactly two BK=64 tiles, so the scale
             * boundary always falls here and never inside an MFMA. */
            if constexpr (FP8) {
#if MPF_BK == 64
                if ((kt & 1u) == 1u || kt == NT - 1u) {
                    const unsigned kb = kt >> 1;
#else
                /* At BK < 64 a scale block is 128/BK k-tiles, not two, and BOTH the cadence
                 * and the index are element-derived. The cadence must cut the f32 accumulation
                 * chain at the SCALE-BLOCK edge only: a mid-block cut (kt & 1 at BK=32) is
                 * numerically fine but groups the sum differently from the BK=64 kernel and
                 * the rung-1 decode path — the batch-determinism probe sees the ulps and the
                 * ladder's solo-vs-paired byte-identity promise breaks. With the cut at the
                 * block edge the MFMA chain runs k = 0..127 in the same order as BK=64 and
                 * the promotion is bit-identical to it. (kt >> 1 as the INDEX additionally
                 * charged the second half of every block to the NEXT block's scale.) */
                constexpr unsigned KTPB = 128u / MPF_BK; /* k-tiles per scale block */
                static_assert(128 % MPF_BK == 0 && (KTPB & (KTPB - 1u)) == 0,
                              "scale-block promotion needs a power-of-two k-tile count");
                if ((kt & (KTPB - 1u)) == KTPB - 1u || kt == NT - 1u) {
                    const unsigned kb = kt / KTPB;
#endif
#pragma unroll
                    for (int i = 0; i < SM; i++)
#pragma unroll
                        for (int j = 0; j < SN; j++) {
                            const float* Sj = (GLU && j == 1) ? S1 : S0;
                            /* MPF_W_HALF staged the weights at OCP/2 — the x2 folds into the
                             * block scale here, exactly (power-of-two f32 mul). */
                            const float bs = Sj[(size_t)nblk_row[j] * KB + kb] *
                                             (MPF_W_HALF ? 2.0f : 1.0f);
                            accf[i][j] += acc[i][j] * bs;
                            acc[i][j] = (f32x16)(0.0f);
                        }
                }
            }

            if (kn < K) {
                if constexpr (DB == 1) __syncthreads(); /* every reader done with the one buffer */
                const unsigned cb = (DB == 2) ? (buf ^ 1u) : 0u;
                MPF_COMMIT(cb)
            }
            __syncthreads();
            if constexpr (DB == 2) buf ^= 1u;
        }
#endif /* PLOW_MOE_PF_PIPE */
        if constexpr (!FP8) {
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) accf[i][j] = acc[i][j];
        }

#if PLOW_MOE_PF_EPI
#if PLOW_MOE_PF_EPI >= 2
        if constexpr (!GLU) {
            MPF_EPI_LOAD
        }
#endif
/* Row `rr`'s hoisted metadata, read back out of the lane that loaded it.
 *
 * EXEC DISCIPLINE, and it is load-bearing: `ds_bpermute_b32` honours EXEC on the READ side --
 * a lane sourcing from a lane that is masked off does NOT get that lane's value. Lane L wants
 * row `rr(L)`, held by lane `rr(L)`, whose OWN activity is decided by row `rr(rr(L))` -- a
 * different row. So neither the `nn < N` tail guard nor the PLOW_EXPERT_UNUSED pad-row test
 * may be live when these issue: both fetches are emitted BEFORE either test (see
 * MPF_EPI_NGUARD / MPF_ROWMETA), at the full wave EXEC the epilogue is entered with. The
 * intrinsic is `convergent`, so LLVM may not sink it back into the divergent region. */
#define MPF_ROW_PIDX(rr) \
    ((unsigned)__builtin_amdgcn_ds_bpermute((int)((rr) << 2), (int)epi_pidx))
#define MPF_ROW_GATE(rr) \
    __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((rr) << 2), epi_gate))
/* The n-tail guard becomes a predicate instead of a `continue`, so it does not mask EXEC
 * above the fetches; the pad-row test then folds into one skip. Same stores, same values. */
#define MPF_EPI_NGUARD(nn) const bool epi_nok = ((nn) < N)
#define MPF_ROWMETA(rr, pidx, gv)                                                              \
    const unsigned pidx = MPF_ROW_PIDX(rr);                                                    \
    const float gv = MPF_ROW_GATE(rr);                                                         \
    if (!epi_nok || pidx == PLOW_EXPERT_UNUSED) continue /* n tail or pad row */
#else
#define MPF_EPI_NGUARD(nn) if ((nn) >= N) continue
#define MPF_ROWMETA(rr, pidx, gv)                                                              \
    const unsigned pidx = row_partidx[rowbase + (rr)];                                         \
    if (pidx == PLOW_EXPERT_UNUSED) continue; /* pad row */                                    \
    const float gv = row_gate[rowbase + (rr)]
#endif

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
                        float g = accf[i][0][el], u = accf[i][1][el];
                        if (a8) {
                            /* Per-TOKEN A scale (and the fp8 decode's x2), applied to BOTH
                             * branches BEFORE the GLU nonlinearity — silu(g*s) != s*silu(g).
                             * A pad row has no token; its fu row is never read, write 0. */
                            const unsigned tok = row_token[rowbase + rr];
                            const float sr =
                                (tok == PLOW_EXPERT_UNUSED) ? 0.0f : as_row[tok] * 2.0f;
                            g *= sr;
                            u *= sr;
                        }
                        st_act1(&fu[(size_t)(rowbase + rr) * N + nn_lane],
                                f2bf(moe_glu(g, u, act, beta, lbeta)));
                    }
            }
#if PLOW_MOE_PF_ATOMIC
        } else if (atom_ksh) {
            /* FUSED 86->87 (PLOW_MOE_PF_ATOMIC). Same value, same gate multiply, same guard
             * chain -- only the DESTINATION changes: the per-slot row `pidx` collapses onto its
             * token's accumulator row, and the store becomes a device-scope f32 atomic add so
             * the k slots reduce in place. `pidx = tok*k + slot` by construction, so `tok` is
             * one shift; no extra load, no division, no LDS.
             *
             * RELAXED / AGENT is the weakest correct ordering: nothing in this packet reads the
             * accumulator back, and op 87's read is separated by the packet gate, which already
             * carries the release. AGENT (not WORKGROUP) is required -- the k contributions come
             * from k different workgroups, in general on k different XCDs. */
            float* const acc = (float*)Cout;
            const unsigned ksh = atom_ksh - 1u;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    MPF_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        MPF_ROWMETA(rr, pidx, gv);
                        __hip_atomic_fetch_add(
                            (PLOW_GLOB float*)&acc[(size_t)(pidx >> ksh) * N + nn],
                            gv * accf[i][j][el], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    }
                }
#endif
#if PLOW_MOE_PF_DET
        } else if (det_ksh) {
            /* FUSED 86->87, DETERMINISTICALLY. Identical to the atomic arm in destination and in
             * guard chain -- `pidx >> log2(k)` collapses the slot row onto its token's row -- and
             * different in exactly one thing: the value crosses the atomic in the FIXED-POINT
             * domain, so the k-way sum is an INTEGER sum and is therefore independent of the
             * order the k workgroups happen to arrive in. That is the whole arm.
             *
             * RELAXED / AGENT for the reasons the atomic arm states: nothing here reads the
             * accumulator back, op 87's read is behind the packet gate, and AGENT (not WORKGROUP)
             * is required because the k contributions come from k different XCDs. */
            double* const acc = (double*)Cout;
            const unsigned ksh = det_ksh - 1u;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    MPF_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        MPF_ROWMETA(rr, pidx, gv);
                        __hip_atomic_fetch_add(
                            (PLOW_GLOB double*)&acc[(size_t)(pidx >> ksh) * N + nn],
                            mpf_det_q(gv * accf[i][j][el]), __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT);
                    }
                }
#endif
        } else if (part16) {
            /* bf16 part: halves the pair's largest stream (the scatter) and the combine's
             * readback. Same slots, same gate multiply; only the store width changes. */
            bf16* const part = (bf16*)Cout;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    MPF_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        MPF_ROWMETA(rr, pidx, gv);
                        __builtin_nontemporal_store(
                            f2bf(gv * accf[i][j][el]),
                            (PLOW_GLOB bf16*)&part[(size_t)pidx * N + nn]);
                    }
                }
        } else {
            float* const part = (float*)Cout;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    MPF_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        MPF_ROWMETA(rr, pidx, gv);
                        /* NON-TEMPORAL: part is written once here and read back only by the
                         * combine, after a full 304-CU round — at T=2048 it is ~400 MB per
                         * rank, far past L2, so caching these lines only evicts the weight
                         * stream this kernel is bound on. Same value, same address. */
                        __builtin_nontemporal_store(
                            gv * accf[i][j][el],
                            (PLOW_GLOB float*)&part[(size_t)pidx * N + nn]);
                    }
                }
        }
        __syncthreads();
    }
#undef MPF_FETCH_A
#undef MPF_FETCH_B
#undef MPF_COMMIT
#undef MPF_COMMIT_A
#undef MPF_A_SRC
#undef MPF_GROW
#undef MPF_ROW_PIDX
#undef MPF_ROW_GATE
#undef MPF_EPI_NGUARD
#undef MPF_ROWMETA
#if PLOW_MOE_PF_EPI
#undef MPF_EPI_LOAD
#endif
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
#if PLOW_HAS_MX_MMA
#define MPF4_LDS_BYTES (2u * (MPF4_ATB + MPF4_BTB) + 2u * MPF4_BM * MPF4_SPR + \
                        2u * MPF4_BN * MPF4_SPR + MPF4_BM * (MPF4_BN / 2) * 4u)
#else
/* The CDNA3 simulated arm: single-buffered bf16 tiles at BK=64 (128-B rows), the GLU bridge
 * aliased over them. 40,960 B — under the 64 KiB part's arena where the CDNA4 plan (76,288 B)
 * is not, which is half the reason that arm stages bf16 the way it does. */
#define MPF4_LDS_BYTES ((MPF4_BM + MPF4_BN) * 128u)
#endif

/* 16 bytes of fp4 payload held in registers between a staging phase's ISSUE and COMMIT halves
 * — one MX block for A (DOWN) or B. Vector-typed so the copy in and the copy out are each a
 * single wide access rather than sixteen byte ones. */
#if PLOW_MOE_PF_EPI_SIB && (MPF4_BM != PLOW_WAVE)
#error "PLOW_MOE_PF_EPI_SIB assumes one wave covers the m-tile's rows (MPF4_BM == PLOW_WAVE)"
#endif

typedef unsigned mpf4_b16 __attribute__((ext_vector_type(4), aligned(4)));

/* BY VALUE, both directions. `__builtin_memcpy` straight into an `mpf4_b16` array element left
 * the array as a stack object — 48 B/lane of scratch, and a `s_waitcnt vmcnt(0)` immediately
 * behind the load to feed the scratch_store, which is precisely the stall the split exists to
 * remove. Routing through an SSA return value keeps the array promotable.
 *
 * THE SOURCE IS ADDRESS-SPACE-1 ON PURPOSE. Expert weight pointers arrive as integers in
 * `wtab`/`stab`, so they are GENERIC and lower to `flat_load` — and SIInsertWaitcnts cannot
 * order a flat load against a global one, so it plants a `s_waitcnt vmcnt(0)` in front of every
 * flat load in the block. That wait lands between A's reads and B's and undoes the split. The
 * pointers are device allocations; saying so turns the flat loads back into global ones.
 *
 * Alignment 4, not 16: the same claim `__builtin_memcpy(...,16)` produced here before. MXFP4
 * needs K % 32 == 0, so the fp4 row stride K/2 is a multiple of 16 and these are in fact
 * 16-aligned, but nothing downstream needs that asserted. */
__device__ __forceinline__ mpf4_b16 mpf4_ld16(const PLOW_GLOB unsigned char* p) {
    return *(const PLOW_GLOB mpf4_b16*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ void mpf4_st16(unsigned char* p, mpf4_b16 v) {
    *(mpf4_b16*)(void*)p = v;
}

/* Quantize 32 consecutive bf16 to one MX block: 16 fp4 bytes + one E8M0 byte. This is the
 * A-side of A4W4 and it runs IN THE GEMM'S STAGING PATH, not as a separate pass — the
 * activation is quantized on its way from HBM into LDS, so it is never written back at 16
 * bits and never re-read.
 *
 * SPLIT IN TWO ON PURPOSE. The block amax is a reduction over all 32 values, so the moment any
 * of the arithmetic below runs it drags an `s_waitcnt vmcnt(0)` back to sit directly behind the
 * loads. Keeping the four reads (`_load`) separable from everything that touches them
 * (`_commit`) is what lets the main loop put the reads in flight ACROSS the MFMA block and pay
 * for them afterwards. The arithmetic itself is unchanged: same reduction order, same
 * e8m0_for_amax, same fp4 ladder, so the bytes produced are the bytes that were produced
 * before. */
struct mpf4_a32 { bf16v8 w[4]; }; /* returned BY VALUE, so it stays in registers */
__device__ __forceinline__ mpf4_a32 mpf4_quant_load(const bf16* src) {
    mpf4_a32 r;
#pragma unroll
    for (int i = 0; i < 4; i++) r.w[i] = ld_glob8(src + i * 8);
    return r;
}
__device__ __forceinline__ void mpf4_quant_commit(mpf4_a32 a, unsigned char* dst16,
                                                  unsigned char* scale_out) {
    /* NO `float v[32]` HERE, DELIBERATELY. Widening the f32 copies of the block up front and
     * keeping them across the amax reduction is 32 live VGPR, and once `quant_fp4` went
     * branchless there was nothing left to stop the scheduler hoisting all 32 multiplies to the
     * top of that range: +26 VGPR on the GLU arm, which is 3 waves/SIMD down to 2. The raw bf16
     * are already in hand as 16 registers and `bf2f` is a shift, so the second pass re-derives
     * them for free rather than storing them. Same values, same reduction order. */
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bf2f(a.w[i][j])));
    const unsigned char sb = e8m0_for_amax(amax);
    const float inv = e8m0_inv_f32(sb);
    /* Packed a word at a time and written as ONE b128. Byte i of the block holds element 2i in
     * its low nibble and 2i+1 in its high nibble, so word k holds elements 8k..8k+7 at nibble
     * j — the same bytes the sixteen `dst16[i] = ...` stores produced, as one `ds_write_b128`
     * instead of sixteen `ds_write_b8`. */
    mpf4_b16 q;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned w = 0u;
#pragma unroll
        for (int j = 0; j < 8; j++) w |= quant_fp4(bf2f(a.w[i][j]) * inv) << (j * 4);
        q[i] = w;
    }
    mpf4_st16(dst16, q);
    *scale_out = sb;
}

/* What a thread holds for its A block between ISSUE and COMMIT: 32 raw bf16 under GLU, 16 fp4
 * bytes under DOWN. Selected rather than declared side by side, so the arm that does not use a
 * carrier never declares one. */
template <bool C, class A, class B> struct mpf4_sel { using type = A; };
template <class A, class B> struct mpf4_sel<false, A, B> { using type = B; };
#define MPF4_ACELL typename mpf4_sel<GLU, mpf4_a32, mpf4_b16>::type

#if PLOW_HAS_MX_MMA /* the scaled f8f6f4 matrix core; CDNA3 has no fp4 path */
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
                                    float beta, float lbeta, void* ldsv) {
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
        const PLOW_GLOB unsigned char* W0 = as_glob((const unsigned char*)(size_t)wb0);
        const PLOW_GLOB unsigned char* W1 =
            as_glob(GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : nullptr);
        const PLOW_GLOB unsigned char* SW0 =
            as_glob((const unsigned char*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)]);
        const PLOW_GLOB unsigned char* SW1 =
            as_glob(GLU ? (const unsigned char*)(size_t)stab[(size_t)e * 3 + 1] : nullptr);

#if PLOW_MOE_PF_EPI_SIB
        /* ---- DOWN-EPILOGUE ROW-METADATA HOIST, A4W4 TWIN (see the PLOW_MOE_PF_EPI_SIB
         * header note). MPF4_BM == PLOW_WAVE == 64, so one wave holds the whole row block one
         * row per lane; the epilogue bpermutes row rr out of lane rr. Issued at the tile head
         * so the k-loop covers the latency. DOWN arm only -- the GLU arm passes
         * row_partidx = row_gate = nullptr and its own pad-row read is per 32-column BLOCK,
         * not per element, so it is not this pattern. */
        unsigned epi_pidx = 0u;
        int epi_gate = 0;
        if constexpr (!GLU) {
            epi_pidx = as_glob(row_partidx)[rowbase + lane];
            epi_gate = __builtin_bit_cast(int, as_glob(row_gate)[rowbase + lane]);
        }
#endif

        f32x16 acc[SMa][SNa];
#pragma unroll
        for (int i = 0; i < SMa; i++)
#pragma unroll
            for (int j = 0; j < SNa; j++) acc[i][j] = (f32x16)(0.0f);

/* --- STAGING, SPLIT INTO ISSUE AND COMMIT ---------------------------------------------------
 * THE DEFECT THIS FIXES. This loop was double-buffered in STORAGE but not in EXECUTION: the
 * `buf ^ 1` ping-pong was there, but the next tile's global reads were ISSUED after the current
 * tile's MFMA block, so the order was MFMA -> load -> wait -> barrier -> MFMA and the matrix
 * pipeline sat idle across every staging phase. Worse, the staging itself was three separate
 * full-latency exposures per K tile — `row_token`, then A, then B twice — each with its own
 * `s_waitcnt vmcnt(0)` immediately behind the load.
 *
 * The staging of one K tile is now four guard-free bodies:
 *
 *   ..._ISSUE1     the GLOBAL READS, and nothing else — no LDS traffic, no quantizer
 *                  arithmetic, and therefore no `s_waitcnt vmcnt`.
 *   ..._ADDR       the LDS destinations.
 *   ..._WRITE1     the quantizer (GLU) or the byte copy, then the LDS write.
 *   ..._PAD1       what a pad row / an out-of-range weight row gets instead: ZERO DATA and a
 *                  NEUTRAL scale, never a zero scale byte, which would be 2^-127 not "off".
 *
 * composed as MPF4_ISSUE(k0) ... MFMA ... MPF4_COMMIT(buf), so every read of the next tile is
 * in flight while the matrix core works through the current one, and the four separate waits
 * collapse to one. The double buffer makes that safe with the barrier count UNCHANGED: COMMIT
 * writes `buf ^ 1` while the MFMA block reads `buf`, and the single `__syncthreads()` at the
 * bottom of the iteration publishes those writes to the next one.
 *
 * WHERE THE SPLIT FALLS, AND WHY THERE. The block amax is a reduction over all 32 values, so
 * anything that touches them drags an `s_waitcnt vmcnt(0)` back to sit directly behind the
 * loads — which is exactly why the quantizer lives in WRITE1 and not in ISSUE1. Put it in ISSUE
 * and the wait moves back above the MFMA and nothing is overlapped. ONE THREAD STILL OWNS ONE
 * WHOLE MX BLOCK, so that amax stays a thread-local reduction and needs no shuffle, and the
 * bytes it produces are the bytes it produced before. What a thread carries between the halves
 * is 32 bf16 for GLU's A and 16 fp4 bytes plus a scale byte for every other stream.
 *
 * THE BODIES CARRY NO GUARDS, AND THAT IS LOAD-BEARING. WRITE1 and PAD1 are separate so the
 * COMPOSITION decides how many branches exist, and COMMIT puts them under ONE. An earlier shape
 * that guarded each body for itself — "read if live", then a second "if live" to choose write
 * over pad — cost the DOWN arm 12 VGPR, which is exactly the step from 4 waves/SIMD to 3, for
 * no reason but the duplicated branch. Measured; do not fold the guards back in. */
#define MPF4_ANB (MPF4_BM * MPF4_SPR)                              /* 256  A blocks per tile */
#define MPF4_BNB (MPF4_BN * MPF4_SPR)                              /* 1024 B blocks per tile */
#define MPF4_AIT ((MPF4_ANB + PLOW_THREADS - 1) / PLOW_THREADS)
#define MPF4_BIT ((MPF4_BNB + PLOW_THREADS - 1) / PLOW_THREADS)

/* --- A staging. GLU quantizes bf16 -> MXFP4 on the way in; DOWN copies fp4 that the bridge
 * already wrote. `sr` is the gathered row, and PLOW_EXPERT_UNUSED doubles as the marker for a
 * thread that owns no block at all, so one test covers the pad row and the idle thread. */
#define MPF4_ASRC(nm, t)                                                                       \
    unsigned nm;                                                                               \
    if ((t) >= (unsigned)MPF4_ANB) nm = PLOW_EXPERT_UNUSED;                                    \
    else if constexpr (GLU) nm = row_token[rowbase + (t) / MPF4_SPR];                          \
    else nm = rowbase + (t) / MPF4_SPR;

#define MPF4_A_ISSUE1(t, k0, sr, c, sc)                                                        \
    {                                                                                          \
        const unsigned ab_ = (t) % MPF4_SPR;                                                   \
        if constexpr (GLU) {                                                                   \
            (c) = mpf4_quant_load((const bf16*)Ain + (size_t)(sr)*K + (k0) + ab_ * 32u);       \
        } else {                                                                               \
            (c) = mpf4_ld16(as_glob((const unsigned char*)Ain) + (size_t)(sr)*KS +             \
                            ((k0) >> 1) + ab_ * 16u);                                          \
            (sc) = Ascale[(size_t)(sr)*KSC + ((k0) >> 5) + ab_];                               \
        }                                                                                      \
    }

#define MPF4_A_ADDR(t, buf)                                                                    \
    const unsigned ar_ = (t) / MPF4_SPR, ab_ = (t) % MPF4_SPR;                                 \
    unsigned char* adq_ =                                                                      \
        Atl + (buf)*MPF4_ATB + ar_ * MPF4_RB + MPF4_XORSWZ(ar_, ab_ * 16u);                    \
    unsigned char* ads_ = Asc + (buf)*MPF4_BM * MPF4_SPR + ar_ * MPF4_SPR + ab_;

#define MPF4_A_WRITE1(c, sc)                                                                   \
    {                                                                                          \
        if constexpr (GLU) mpf4_quant_commit((c), adq_, ads_);                                 \
        else {                                                                                 \
            mpf4_st16(adq_, (c));                                                              \
            *ads_ = (sc);                                                                      \
        }                                                                                      \
    }

#define MPF4_A_PAD1                                                                            \
    {                                                                                          \
        _Pragma("unroll") for (int z_ = 0; z_ < 16; z_++) adq_[z_] = 0;                        \
        *ads_ = (unsigned char)PLOW_E8M0_ONE;                                                  \
    }

/* --- B staging. Weights are already MXFP4 on disk; this is a byte copy plus its scale byte.
 * Under GLU the tile's low half is the gate weight and its high half the up weight, at the SAME
 * output rows, so the SN axis selects gate vs up in the epilogue with no shuffle. */
#define MPF4_BROW(t)                                                                           \
    const unsigned bb_ = (t) % MPF4_SPR, br_ = (t) / MPF4_SPR;                                 \
    const bool up_ = GLU && (br_ >= MPF4_BN / 2);                                              \
    const unsigned bn_ = n0 + (up_ ? br_ - MPF4_BN / 2 : br_);
/* The weight base is ISSUE's business only — the write side wants the row, not the pointer. */
#define MPF4_BPTR                                                                              \
    const PLOW_GLOB unsigned char* bw_ = up_ ? W1 : W0;                                        \
    const PLOW_GLOB unsigned char* bs_ = up_ ? SW1 : SW0;

#define MPF4_B_ISSUE1(k0, q, sc)                                                               \
    {                                                                                          \
        (q) = mpf4_ld16(bw_ + (size_t)bn_ * KS + ((k0) >> 1) + bb_ * 16u);                     \
        (sc) = bs_[(size_t)bn_ * KSC + ((k0) >> 5) + bb_];                                     \
    }

#define MPF4_B_ADDR(buf)                                                                       \
    unsigned char* bdq_ =                                                                      \
        Btl + (buf)*MPF4_BTB + br_ * MPF4_RB + MPF4_XORSWZ(br_, bb_ * 16u);                    \
    unsigned char* bds_ = Bsc + (buf)*MPF4_BN * MPF4_SPR + br_ * MPF4_SPR + bb_;

#define MPF4_B_WRITE1(q, sc)                                                                   \
    {                                                                                          \
        mpf4_st16(bdq_, (q));                                                                  \
        *bds_ = (sc);                                                                          \
    }

#define MPF4_B_PAD1                                                                            \
    {                                                                                          \
        _Pragma("unroll") for (int z_ = 0; z_ < 16; z_++) bdq_[z_] = 0;                        \
        *bds_ = (unsigned char)PLOW_E8M0_ONE;                                                  \
    }

/* Every read of the tile first, every write after — the MFMA block goes in between. */
#define MPF4_ISSUE(k0)                                                                         \
    _Pragma("unroll") for (int it_ = 0; it_ < MPF4_AIT; it_++) {                               \
        const unsigned t_ = tid + (unsigned)it_ * PLOW_THREADS;                                \
        if (asrc_[it_] != PLOW_EXPERT_UNUSED) MPF4_A_ISSUE1(t_, k0, asrc_[it_], aw_[it_],      \
                                                            asc_[it_])                         \
    }                                                                                          \
    _Pragma("unroll") for (int it_ = 0; it_ < MPF4_BIT; it_++) {                               \
        const unsigned t_ = tid + (unsigned)it_ * PLOW_THREADS;                                \
        if (t_ < (unsigned)MPF4_BNB) {                                                         \
            MPF4_BROW(t_)                                                                      \
            MPF4_BPTR                                                                          \
            if (bn_ < N) MPF4_B_ISSUE1(k0, bq_[it_], bsc_[it_])                                \
        }                                                                                      \
    }
#define MPF4_COMMIT(buf)                                                                       \
    _Pragma("unroll") for (int it_ = 0; it_ < MPF4_AIT; it_++) {                               \
        const unsigned t_ = tid + (unsigned)it_ * PLOW_THREADS;                                \
        if (t_ < (unsigned)MPF4_ANB) {                                                         \
            MPF4_A_ADDR(t_, buf)                                                               \
            if (asrc_[it_] != PLOW_EXPERT_UNUSED) MPF4_A_WRITE1(aw_[it_], asc_[it_])           \
            else MPF4_A_PAD1                                                                   \
        }                                                                                      \
    }                                                                                          \
    _Pragma("unroll") for (int it_ = 0; it_ < MPF4_BIT; it_++) {                               \
        const unsigned t_ = tid + (unsigned)it_ * PLOW_THREADS;                                \
        if (t_ < (unsigned)MPF4_BNB) {                                                         \
            MPF4_BROW(t_)                                                                      \
            MPF4_B_ADDR(buf)                                                                   \
            if (bn_ < N) MPF4_B_WRITE1(bq_[it_], bsc_[it_])                                    \
            else MPF4_B_PAD1                                                                   \
        }                                                                                      \
    }

/* --- the MFMA block. Two K=64 MFMAs per staged tile; `s_setprio` brackets them so the wave
 * holds issue priority for exactly as long as the matrix core is being fed. */
#define MPF4_MFMA(buf)                                                                         \
    _Pragma("unroll") for (int q_ = 0; q_ < MPF4_BK / 64; q_++) {                              \
        /* A lane supplies 32 fp4 = 16 B at k = q*64 + 32*khalf. */                            \
        const unsigned boff_ = (unsigned)q_ * 32u + khalf * 16u;                               \
        const unsigned sblk_ = (unsigned)q_ * 2u + khalf;                                      \
        mfma_f8f6f4_operand af_[SMa], bfr_[SNa];                                               \
        int sa_[SMa], sb_[SNa];                                                                \
        _Pragma("unroll") for (int i_ = 0; i_ < SMa; i_++) {                                   \
            const unsigned ar_ = wm * (MPF4_BM / MPF4_WMc) + i_ * MFMA_M + frow;               \
            af_[i_] = fp4_frag(Atl + (buf)*MPF4_ATB + ar_ * MPF4_RB + MPF4_XORSWZ(ar_, boff_)); \
            sa_[i_] = (int)Asc[(buf)*MPF4_BM * MPF4_SPR + ar_ * MPF4_SPR + sblk_];             \
        }                                                                                      \
        _Pragma("unroll") for (int j_ = 0; j_ < SNa; j_++) {                                   \
            const unsigned br_ = (GLU ? j_ * (MPF4_BN / 2) + wn * MFMA_N                       \
                                      : wn * (MPF4_BN / MPF4_WNc) + j_ * MFMA_N) + frow;       \
            bfr_[j_] = fp4_frag(Btl + (buf)*MPF4_BTB + br_ * MPF4_RB +                         \
                                MPF4_XORSWZ(br_, boff_));                                      \
            sb_[j_] = (int)Bsc[(buf)*MPF4_BN * MPF4_SPR + br_ * MPF4_SPR + sblk_];             \
        }                                                                                      \
        __builtin_amdgcn_s_setprio(1);                                                         \
        _Pragma("unroll") for (int i_ = 0; i_ < SMa; i_++)                                     \
            _Pragma("unroll") for (int j_ = 0; j_ < SNa; j_++)                                 \
                acc[i_][j_] = mfma_a4w4(af_[i_], bfr_[j_], acc[i_][j_], sa_[i_], sb_[j_]);     \
        __builtin_amdgcn_s_setprio(0);                                                         \
    }                                                                                          \
    /* PIN THE MATRIX BLOCK HERE. `acc` is a loop-carried PHI whose only use is in the latch, so \
     * LLVM sinks every MFMA past the staging code below and out of its own `s_setprio` window — \
     * which put them AFTER the `s_waitcnt vmcnt` they were supposed to hide, and kept every LDS \
     * fragment alive across the staging block on top of that. A scheduling barrier does not stop \
     * it, the motion is cross-block; an asm read/write of the accumulators does. Emits nothing, \
     * and is worth 30 VGPR on the GLU arm. */                                                  \
    _Pragma("unroll") for (int i_ = 0; i_ < SMa; i_++)                                         \
        _Pragma("unroll") for (int j_ = 0; j_ < SNa; j_++) asm volatile("" : "+v"(acc[i_][j_]));

        /* The gathered row index is K-INVARIANT — `row_token` does not move with the K tile —
         * but reading it inside the staging path put a DEPENDENT global load (row_token, then
         * the A row it addresses) in front of every one of the NT staging phases. Resolved once
         * per output tile instead. */
        unsigned asrc_[MPF4_AIT];
#pragma unroll
        for (int it_ = 0; it_ < MPF4_AIT; it_++) {
            MPF4_ASRC(sr_, tid + (unsigned)it_ * PLOW_THREADS)
            asrc_[it_] = sr_;
        }
        /* What a thread holds between ISSUE and COMMIT, i.e. across the MFMA block. */
        MPF4_ACELL aw_[MPF4_AIT];
        mpf4_b16 bq_[MPF4_BIT];
        unsigned char asc_[MPF4_AIT], bsc_[MPF4_BIT];

        __syncthreads();
        MPF4_ISSUE(0)
        MPF4_COMMIT(0)
        __syncthreads();

        unsigned buf = 0;
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1u) * MPF4_BK;
            /* THE NEXT TILE'S READS GO OUT FIRST. Everything from here down to COMMIT is what
             * they hide behind; nothing in between waits on vmcnt. */
            if (kn < K) { MPF4_ISSUE(kn) }
            MPF4_MFMA(buf)
            /* Now pay for them, into the OTHER buffer. The MFMA block read `buf` and did so
             * through registers, so these writes race nothing; the barrier below is the only one
             * the iteration needs, exactly as before. */
            if (kn < K) { MPF4_COMMIT(buf ^ 1) }
            __syncthreads();
            buf ^= 1;
        }

/* Row `rr`'s hoisted metadata, read back out of the lane that loaded it. Same EXEC discipline
 * as the grouped GEMM's MPF_ROWMETA: `ds_bpermute_b32` honours EXEC on the READ side, so both
 * fetches are emitted BEFORE the lane-varying `nn < N` tail guard and the pad-row test, and
 * those become a MASK rather than EXEC. */
#if PLOW_MOE_PF_EPI_SIB
#define MPF4_EPI_NGUARD(nn) const bool epi_nok = ((nn) < N)
#define MPF4_ROWMETA(rr, pidx, gv)                                                             \
    const unsigned pidx =                                                                      \
        (unsigned)__builtin_amdgcn_ds_bpermute((int)((rr) << 2), (int)epi_pidx);               \
    const float gv =                                                                           \
        __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((rr) << 2), epi_gate));   \
    if (!epi_nok || pidx == PLOW_EXPERT_UNUSED) continue /* n tail or pad row */
#else
#define MPF4_EPI_NGUARD(nn) if ((nn) >= N) continue
#define MPF4_ROWMETA(rr, pidx, gv)                                                             \
    const unsigned pidx = row_partidx[rowbase + (rr)];                                         \
    if (pidx == PLOW_EXPERT_UNUSED) continue;                                                  \
    const float gv = row_gate[rowbase + (rr)]
#endif

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
                            moe_glu(acc[i][0][el], acc[i][1][el], act, beta, lbeta);
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
                /* One base pointer, so the two passes strength-reduce to `ds_read_b128` runs
                 * rather than 64 separate `ds_read_b32` off a recomputed index. The block is
                 * read TWICE on purpose — the amax has to be complete before anything is scaled,
                 * and holding 32 floats across it costs more than re-reading LDS. */
                const float* const bs = Br + r * (MPF4_BN / 2) + c0;
                float amax = 0.0f;
#pragma unroll
                for (int z = 0; z < 32; z++) amax = fmaxf(amax, fabsf(bs[z]));
                const unsigned char sbv = e8m0_for_amax(amax);
                const float inv = e8m0_inv_f32(sbv);
                unsigned char* o = fq + (size_t)(rowbase + r) * (N >> 1) + ((n0 + c0) >> 1);
                /* Same byte layout as the sixteen `o[z] = ...` stores it replaces — byte i is
                 * element 2i low, 2i+1 high — emitted as one 16-byte store. Both N and the block
                 * origin are multiples of 32 elements, so `o` is 16-byte aligned. */
                mpf4_b16 q;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    unsigned w = 0u;
#pragma unroll
                    for (int j = 0; j < 8; j++) w |= quant_fp4(bs[k * 8 + j] * inv) << (j * 4);
                    q[k] = w;
                }
                mpf4_st16(o, q);
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
                    MPF4_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + mfma_acc_m(lane, el);
                        MPF4_ROWMETA(rr, pidx, gv);
                        part[(size_t)pidx * N + nn] = gv * acc[i][j][el];
                    }
                }
        }
        __syncthreads();
    }
#undef MPF4_ASRC
#undef MPF4_A_ISSUE1
#undef MPF4_A_ADDR
#undef MPF4_A_WRITE1
#undef MPF4_A_PAD1
#undef MPF4_BROW
#undef MPF4_BPTR
#undef MPF4_B_ISSUE1
#undef MPF4_B_ADDR
#undef MPF4_B_WRITE1
#undef MPF4_B_PAD1
#undef MPF4_ISSUE
#undef MPF4_COMMIT
#undef MPF4_MFMA
#undef MPF4_EPI_NGUARD
#undef MPF4_ROWMETA
}
#else /* !PLOW_HAS_MX_MMA ------------------------------------------------------------------------
 * CDNA3: THE SAME OPS, SIMULATED THROUGH THE bf16 MATRIX CORE. gfx942 has no scaled f8f6f4 MFMA
 * and no fp4 datatype, so true A4W4 cannot be lowered here — but the PACKET does not ask for an
 * instruction, it asks for a computation, and every piece of that computation is expressible:
 * fp4 x 2^e8m0 is EXACT in bf16 (<= 3 significant bits, power-of-two scale), so dequantizing
 * both operands to bf16 and feeding the ordinary bf16 MFMA computes the same products with f32
 * accumulation, differing from the scaled-MFMA path only in accumulation GROUPING — the same
 * license the fp8 matrix-core quartering in amd_arch.h already takes.
 *
 * The CONTRACT is byte-compatible with the CDNA4 body in both directions: same wtab/stab/meta
 * layout, and the GLU epilogue still writes `fu` as MXFP4 + E8M0 in the sorted layout —
 * an emitted packet cannot tell which arch ran it. Two deliberate divergences, both toward
 * accuracy: (1) the GLU A side stages the gathered bf16 activation RAW instead of quantizing it
 * to fp4 first — CDNA4 quantizes because its matrix core demands fp4 operands; this one does
 * not, so the A-quantization error term simply does not exist here; (2) accumulation grouping,
 * as above. Cross-arch outputs agree to quantization/rounding tolerance, NOT bit-for-bit.
 *
 * WHERE THE DEQUANT LIVES — IN STAGING, NOT IN THE MFMA BLOCK. The CDNA4 body stages raw fp4 and
 * lets the matrix instruction apply the scales. Here each staged element must become bf16
 * SOMEWHERE, and the two candidate homes differ by real VALU: a B fragment is re-read by
 * MPF4_WMc waves and an A fragment by MPF4_WNc, so read-time dequant runs 2-4x per element,
 * while COMMIT-time dequant runs exactly once — and lands in the staging phase, whose latency
 * the in-flight global reads already cover. The MFMA block is then pure ds_read_b128 + MFMA
 * with no scale traffic at all.
 *
 * WHAT THAT COSTS IN LDS, and why the tile is BK=64 single-buffered. bf16 staging doubles the
 * tile bytes and CDNA3 halves the arena (64 KiB, PLOW_LDS_MAX_BYTES): the CDNA4 plan
 * (BK=128, double-buffered fp4, 76,288 B) is over budget before the first bf16 byte. At BK=64,
 * single-buffered bf16:
 *
 *   Atl bf16 [64][128 B]     8,192
 *   Btl bf16 [256][128 B]   32,768
 *   Bridge f32 [64][128]    32,768 — ALIASED over Atl+Btl (40,960 >= 32,768), which is safe
 *                                    because the K loop's final barrier retires the tiles
 *                                    before the GLU epilogue's first Bridge write
 *   = 40,960 B against the gfx942 interpreter arena. The BM=64 x BN=256 OUTPUT tile, the 2x4
 * wave grid and SNa=2 are unchanged — SNa=2 is what the GLU gate|up-in-one-lane trick and the
 * unchanged epilogues depend on.
 *
 * Single-buffered is not un-pipelined: the next tile's global reads still issue BEFORE the MFMA
 * block and land in registers across it (the ISSUE/COMMIT split, kept from the CDNA4 body); the
 * cost against double buffering is one extra barrier per K tile, not exposed load latency. */
#ifndef PLOW_MOE_PF_A4W4_C3_BK
#define PLOW_MOE_PF_A4W4_C3_BK 64u
#endif
#if PLOW_MOE_PF_A4W4_C3_BK != 32u && PLOW_MOE_PF_A4W4_C3_BK != 64u
#error "PLOW_MOE_PF_A4W4_C3_BK must be 32 or 64"
#endif
#ifndef PLOW_MOE_PF_A4W4_PRIO
#define PLOW_MOE_PF_A4W4_PRIO 1
#endif
#define MPF4_C3_BK PLOW_MOE_PF_A4W4_C3_BK  /* K per staged tile */
#define MPF4_C3_RB (MPF4_C3_BK * 2u)        /* bf16 LDS row stride */
/* XOR swizzle over 16-byte chunks, bits 4..6 — in-row for a 128-B row, and THREE bits on
 * purpose: a 128-B row is exactly 8 chunks, and a fragment read walks 32 consecutive rows at a
 * FIXED chunk column (row stride 128 B = 0 mod 8 chunks), so anything less than a full 3-bit
 * spread leaves the ds_read_b128 bank-conflicted. MEASURED, interleaved in one lease at the
 * bench shape (H=3584, IM=3072, 4096 rows, grid 304 — moe_prefill_a4w4_cdna3_test.hip
 * MPA4C3_BENCH=1): 1-bit 737 GB/s, 2-bit 928, 3-bit 963 GB/s weight stream = 232 TF/s on the
 * GLU, ~71% of this part's bf16 MFMA peak. ONE function for the write and every read; a second
 * copy that drifts is the transposed-tile class of bug. */
#define MPF4_C3_SWZ(row, off) \
    ((off) ^ (((row) & (MPF4_C3_RB / 16u - 1u)) << 4))

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
                                    float beta, float lbeta, void* ldsv) {
    constexpr int SMa = MPF4_BM / MPF4_WMc / MFMA_M; /* 1 */
    constexpr int SNa = MPF4_BN / MPF4_WNc / MFMA_N; /* 2 */
    constexpr unsigned NB = GLU ? (MPF4_BN / 2) : MPF4_BN;
    /* Staging items per tile: A is 16-B bf16 chunks under GLU (BM * BK/8) and 32-element MX
     * blocks under DOWN (BM * BK/32); B is always MX blocks (BN * BK/32). */
    constexpr unsigned C3_ANB = GLU ? (MPF4_BM * (MPF4_C3_BK / 8u)) : (MPF4_BM * (MPF4_C3_BK / 32u));
    constexpr unsigned C3_BNB = MPF4_BN * (MPF4_C3_BK / 32u);
    constexpr int C3_AIT = (int)((C3_ANB + PLOW_THREADS - 1) / PLOW_THREADS);
    constexpr int C3_BIT = (int)((C3_BNB + PLOW_THREADS - 1) / PLOW_THREADS);

    unsigned char* const L = (unsigned char*)ldsv;
    unsigned char* const Atl = L;                            /* bf16 [BM][C3_RB] */
    unsigned char* const Btl = Atl + MPF4_BM * MPF4_C3_RB;   /* bf16 [BN][C3_RB] */
    float* const Bridge = (float*)L;                         /* GLU epilogue only; aliases tiles */
    static_assert(MPF4_BM * (MPF4_BN / 2) * 4u <= PLOW_LDS_MAX_BYTES,
                  "the GLU bridge exceeds the gfx942 LDS arena");

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave / MPF4_WNc, wn = wave % MPF4_WNc;
    const unsigned frow = mfma_frag_row(lane); /* lane % 32 */
    const unsigned khalf = lane / 32u;         /* which 8-k half of a K=16 step this lane feeds */

    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned total_tiles = (unsigned)tilep[n_exp];
    const unsigned tnc = (N + NB - 1u) / NB;
    const unsigned n_tiles = total_tiles * tnc;
    const unsigned NT = (K + MPF4_C3_BK - 1u) / MPF4_C3_BK;
    const unsigned KS = K >> 1;  /* fp4 row stride in BYTES */
    const unsigned KSC = K >> 5; /* E8M0 scale bytes per row */

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned mt = lin / tnc, nt = lin % tnc;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF4_BM;
        const unsigned n0 = nt * NB;

        const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
        if (wb0 == 0ull) continue; /* EP: expert not local */
        const PLOW_GLOB unsigned char* W0 = as_glob((const unsigned char*)(size_t)wb0);
        const PLOW_GLOB unsigned char* W1 =
            as_glob(GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : nullptr);
        const PLOW_GLOB unsigned char* SW0 =
            as_glob((const unsigned char*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)]);
        const PLOW_GLOB unsigned char* SW1 =
            as_glob(GLU ? (const unsigned char*)(size_t)stab[(size_t)e * 3 + 1] : nullptr);

#if PLOW_MOE_PF_EPI_SIB
        unsigned epi_pidx = 0u;
        int epi_gate = 0;
        if constexpr (!GLU) {
            epi_pidx = as_glob(row_partidx)[rowbase + lane];
            epi_gate = __builtin_bit_cast(int, as_glob(row_gate)[rowbase + lane]);
        }
#endif

        f32x16 acc[SMa][SNa];
#pragma unroll
        for (int i = 0; i < SMa; i++)
#pragma unroll
            for (int j = 0; j < SNa; j++) acc[i][j] = (f32x16)(0.0f);

        /* K-invariant per-item metadata, resolved once per output tile (CDNA4's hoist, kept). */
        unsigned asrc_[C3_AIT]; /* gathered A row, PLOW_EXPERT_UNUSED = pad or idle */
        unsigned brow_[C3_BIT]; /* weight row bn, or ~0u for the n tail */
#pragma unroll
        for (int it = 0; it < C3_AIT; it++) {
            const unsigned t = tid + (unsigned)it * PLOW_THREADS;
            constexpr unsigned APR = GLU ? (MPF4_C3_BK / 8u) : (MPF4_C3_BK / 32u);
            if (t >= C3_ANB) asrc_[it] = PLOW_EXPERT_UNUSED;
            else if constexpr (GLU) asrc_[it] = row_token[rowbase + t / APR];
            else asrc_[it] = rowbase + t / APR;
        }
#pragma unroll
        for (int it = 0; it < C3_BIT; it++) {
            const unsigned t = tid + (unsigned)it * PLOW_THREADS;
            unsigned bn = ~0u;
            if (t < C3_BNB) {
                const unsigned br = t / (MPF4_C3_BK / 32u);
                const bool up = GLU && (br >= MPF4_BN / 2);
                bn = n0 + (up ? br - MPF4_BN / 2 : br);
                if (bn >= N) bn = ~0u;
            }
            brow_[it] = bn;
        }

/* --- staging, ISSUE/COMMIT split exactly as the CDNA4 body reasons: reads carry no LDS
 * traffic and no dequant arithmetic, so no s_waitcnt lands between them and the MFMA block
 * they hide behind. COMMIT dequants (the once-per-element home, see the header) and writes. */
#define C3_A_ISSUE(k0)                                                                         \
    _Pragma("unroll") for (int it = 0; it < C3_AIT; it++) {                                    \
        const unsigned t = tid + (unsigned)it * PLOW_THREADS;                                  \
        if (asrc_[it] != PLOW_EXPERT_UNUSED) {                                                 \
            if constexpr (GLU) {                                                               \
                const unsigned ac = t % (MPF4_C3_BK / 8u);                                     \
                aw_[it] = ld_glob8((const bf16*)Ain + (size_t)asrc_[it] * K + (k0) + ac * 8u); \
            } else {                                                                           \
                const unsigned ab = t % (MPF4_C3_BK / 32u);                                    \
                aq_[it] = mpf4_ld16(as_glob((const unsigned char*)Ain) +                       \
                                    (size_t)asrc_[it] * KS + ((k0) >> 1) + ab * 16u);          \
                asc_[it] = Ascale[(size_t)asrc_[it] * KSC + ((k0) >> 5) + ab];                 \
            }                                                                                  \
        }                                                                                      \
    }
#define C3_B_ISSUE(k0)                                                                         \
    _Pragma("unroll") for (int it = 0; it < C3_BIT; it++) {                                    \
        if (brow_[it] != ~0u) {                                                                \
            const unsigned t = tid + (unsigned)it * PLOW_THREADS;                              \
            const unsigned bb = t % (MPF4_C3_BK / 32u);                                        \
            const unsigned br = t / (MPF4_C3_BK / 32u);                                        \
            const bool up = GLU && (br >= MPF4_BN / 2);                                        \
            const PLOW_GLOB unsigned char* bw = up ? W1 : W0;                                  \
            const PLOW_GLOB unsigned char* bs = up ? SW1 : SW0;                                \
            bq_[it] = mpf4_ld16(bw + (size_t)brow_[it] * KS + ((k0) >> 1) + bb * 16u);         \
            bsc_[it] = bs[(size_t)brow_[it] * KSC + ((k0) >> 5) + bb];                         \
        }                                                                                      \
    }
/* Dequant an MX block (16 fp4 bytes + scale) to 32 bf16 and write its four 16-B chunks through
 * the swizzle. Zero fill for pads — bf16 zero needs no neutral-scale bookkeeping. */
#define C3_BLK_COMMIT(dst_row_base, r_, blk_, q_, sc_, live_)                                  \
    {                                                                                          \
        const unsigned off0 = (blk_)*64u;                                                      \
        if (live_) {                                                                           \
            bf16v8 d0, d1, d2, d3;                                                             \
            fp4_to_bf16v8x4(__builtin_bit_cast(fp4v32, (q_)), e8m0_to_f32(sc_), d0, d1, d2,    \
                            d3);                                                               \
            *(bf16v8*)((dst_row_base) + MPF4_C3_SWZ(r_, off0)) = d0;                           \
            *(bf16v8*)((dst_row_base) + MPF4_C3_SWZ(r_, off0 + 16u)) = d1;                     \
            *(bf16v8*)((dst_row_base) + MPF4_C3_SWZ(r_, off0 + 32u)) = d2;                     \
            *(bf16v8*)((dst_row_base) + MPF4_C3_SWZ(r_, off0 + 48u)) = d3;                     \
        } else {                                                                               \
            _Pragma("unroll") for (unsigned q4 = 0; q4 < 4u; q4++) {                           \
                *(bf16v8*)((dst_row_base) + MPF4_C3_SWZ(r_, off0 + q4 * 16u)) = (bf16v8)(bf16)0; \
            }                                                                                  \
        }                                                                                      \
    }
#define C3_COMMIT                                                                              \
    _Pragma("unroll") for (int it = 0; it < C3_AIT; it++) {                                    \
        const unsigned t = tid + (unsigned)it * PLOW_THREADS;                                  \
        if (t < C3_ANB) {                                                                      \
            if constexpr (GLU) {                                                               \
                const unsigned ar = t / (MPF4_C3_BK / 8u), ac = t % (MPF4_C3_BK / 8u);         \
                bf16v8 v = (asrc_[it] != PLOW_EXPERT_UNUSED) ? aw_[it] : (bf16v8)(bf16)0;      \
                *(bf16v8*)(Atl + ar * MPF4_C3_RB + MPF4_C3_SWZ(ar, ac * 16u)) = v;             \
            } else {                                                                           \
                const unsigned ar = t / (MPF4_C3_BK / 32u), ab = t % (MPF4_C3_BK / 32u);       \
                C3_BLK_COMMIT(Atl + ar * MPF4_C3_RB, ar, ab, aq_[it], asc_[it],                \
                              asrc_[it] != PLOW_EXPERT_UNUSED)                                 \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
    _Pragma("unroll") for (int it = 0; it < C3_BIT; it++) {                                    \
        const unsigned t = tid + (unsigned)it * PLOW_THREADS;                                  \
        if (t < C3_BNB) {                                                                      \
            const unsigned br = t / (MPF4_C3_BK / 32u), bb = t % (MPF4_C3_BK / 32u);           \
            C3_BLK_COMMIT(Btl + br * MPF4_C3_RB, br, bb, bq_[it], bsc_[it], brow_[it] != ~0u)  \
        }                                                                                      \
    }

        /* Carriers across the MFMA block. */
        bf16v8 aw_[GLU ? C3_AIT : 1];
        mpf4_b16 aq_[GLU ? 1 : C3_AIT];
        mpf4_b16 bq_[C3_BIT];
        unsigned char asc_[C3_AIT], bsc_[C3_BIT];
        (void)asc_;

        __syncthreads(); /* the previous tile's epilogue may still be reading the arena */
        C3_A_ISSUE(0u)
        C3_B_ISSUE(0u)
        C3_COMMIT
        __syncthreads();

        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1u) * MPF4_C3_BK;
            if (kn < K) {
                C3_A_ISSUE(kn)
                C3_B_ISSUE(kn)
            }
            /* Four K=16 bf16 MFMA steps over the staged tile: pure fragment reads, no dequant,
             * no scales — COMMIT already folded them. Lane feeds k = s*16 + 8*khalf + [0..7]. */
#pragma unroll
            for (int s = 0; s < (int)(MPF4_C3_BK / 16u); s++) {
                const unsigned boff = (unsigned)s * 32u + khalf * 16u; /* bf16 bytes */
                bf16x8 af[SMa], bf[SNa];
#pragma unroll
                for (int i = 0; i < SMa; i++) {
                    const unsigned ar = wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + frow;
                    af[i] = __builtin_bit_cast(
                        bf16x8, *(const bf16v8*)(Atl + ar * MPF4_C3_RB + MPF4_C3_SWZ(ar, boff)));
                }
#pragma unroll
                for (int j = 0; j < SNa; j++) {
                    const unsigned br = (GLU ? j * (MPF4_BN / 2) + wn * MFMA_N
                                             : wn * (MPF4_BN / MPF4_WNc) + j * MFMA_N) +
                                        frow;
                    bf[j] = __builtin_bit_cast(
                        bf16x8, *(const bf16v8*)(Btl + br * MPF4_C3_RB + MPF4_C3_SWZ(br, boff)));
                }
#if PLOW_MOE_PF_A4W4_PRIO
                __builtin_amdgcn_s_setprio(1);
#endif
#pragma unroll
                for (int i = 0; i < SMa; i++)
#pragma unroll
                    for (int j = 0; j < SNa; j++)
                        acc[i][j] = plow_mfma_bf16_32x32(af[i], bf[j], acc[i][j]);
#if PLOW_MOE_PF_A4W4_PRIO
                __builtin_amdgcn_s_setprio(0);
#endif
            }
#pragma unroll
            for (int i = 0; i < SMa; i++)
#pragma unroll
                for (int j = 0; j < SNa; j++) asm volatile("" : "+v"(acc[i][j]));
            __syncthreads(); /* everyone done READING the tile */
            if (kn < K) { C3_COMMIT }
            __syncthreads(); /* publish the next tile (or retire the last: Bridge aliases it) */
        }

#if PLOW_MOE_PF_EPI_SIB
#define C3_EPI_NGUARD(nn) const bool epi_nok = ((nn) < N)
#define C3_ROWMETA(rr, pidx, gv)                                                               \
    const unsigned pidx =                                                                      \
        (unsigned)__builtin_amdgcn_ds_bpermute((int)((rr) << 2), (int)epi_pidx);               \
    const float gv =                                                                           \
        __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((rr) << 2), epi_gate));   \
    if (!epi_nok || pidx == PLOW_EXPERT_UNUSED) continue
#else
#define C3_EPI_NGUARD(nn) if ((nn) >= N) continue
#define C3_ROWMETA(rr, pidx, gv)                                                               \
    const unsigned pidx = row_partidx[rowbase + (rr)];                                         \
    if (pidx == PLOW_EXPERT_UNUSED) continue;                                                  \
    const float gv = row_gate[rowbase + (rr)]
#endif

        if constexpr (GLU) {
            /* --- the fused bridge, VERBATIM in structure from the CDNA4 arm (see its header
             * for why the LDS transpose is unavoidable): SwiGLU -> Bridge f32 -> per-(row,
             * 32-col block) amax -> MXFP4 + E8M0 in the sorted layout gemm2 reads. Bridge
             * ALIASES the tile arena; the barrier above retired the tiles. */
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
                            moe_glu(acc[i][0][el], acc[i][1][el], act, beta, lbeta);
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
                const float* const bs = Br + r * (MPF4_BN / 2) + c0;
                float amax = 0.0f;
#pragma unroll
                for (int z = 0; z < 32; z++) amax = fmaxf(amax, fabsf(bs[z]));
                const unsigned char sbv = e8m0_for_amax(amax);
                const float inv = e8m0_inv_f32(sbv);
                unsigned char* o = fq + (size_t)(rowbase + r) * (N >> 1) + ((n0 + c0) >> 1);
                mpf4_b16 q;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    unsigned w = 0u;
#pragma unroll
                    for (int j = 0; j < 8; j++) w |= quant_fp4(bs[k * 8 + j] * inv) << (j * 4);
                    q[k] = w;
                }
                mpf4_st16(o, q);
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
                    C3_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr =
                            wm * (MPF4_BM / MPF4_WMc) + i * MFMA_M + mfma_acc_m(lane, el);
                        C3_ROWMETA(rr, pidx, gv);
                        part[(size_t)pidx * N + nn] = gv * acc[i][j][el];
                    }
                }
        }
        __syncthreads();
    }
#undef C3_A_ISSUE
#undef C3_B_ISSUE
#undef C3_BLK_COMMIT
#undef C3_COMMIT
#undef C3_EPI_NGUARD
#undef C3_ROWMETA
}
#endif /* PLOW_HAS_MX_MMA */

/* --- THE PREFILL ENCODING REFUSAL ---------------------------------------------------------
 * Ops 85/86 select their body from `i[3]`, but the MXFP4 body is COMPILE-TIME
 * (`PLOW_MOE_PF_A4W4`, for the register reason interp.hip states). So an object built without
 * that flag has NO body for enc 2 — and the arm selection below was `if (fp8) fp8 else BF16`,
 * which made "no body" mean "read the fp4 weight bytes as bf16". Two elements per byte read as
 * one at 2 B/elt, an E8M0 scale row ignored, and a 4x overrun of every expert tensor: no fault,
 * finite output, wrong model.
 *
 * THIS IS NOT HYPOTHETICAL, and it is the reason this refusal is in the kernel and not only in
 * the host manifest. `scripts/build_gfx950.sh:307` builds the shipping MoE-prefill object as
 * `-DPLOW_BUCKET_DECODE=0 -DPLOW_MLA_PREFILL=1 -DPLOW_MOE_PREFILL=1` — no `PLOW_MOE_PF_A4W4`.
 * `runtime/CMakeLists.txt:657` builds an a4w4 row and `crates/devgen/src/manifest.rs:591` makes
 * an `a4w4` packet REQUIRE the flag, so the host pairing check is the real gate; but the two
 * build systems disagree about which objects exist, and the pairing check is a different
 * process from the one that runs the GEMM.
 *
 * WHAT "LOUD" IS HERE. `part` is f32 `[T*k][H]` under EVERY encoding, so the DOWN refusal writes
 * a real NaN to exactly the destinations the live rows own — that alone poisons the combine and
 * the residual. `fu_g` is NOT layout-stable (bf16 `[rows][I]` under fp8/bf16, fp4 `[rows][I/2]`
 * + a scale row under MXFP4), and the packet that reached us believes the MXFP4 sizing, so the
 * GLU refusal fills only `rows * (N/2)` BYTES — in bounds under both layouts, and 0xFF is a bf16
 * NaN (sign 1, exp 0xFF, mantissa 0x7F) under the wider one. Belt to the DOWN op's braces. */
__device__ __forceinline__ void moe_pf_refuse(void* Cout, const int* meta,
                                              const unsigned* row_partidx, unsigned N,
                                              unsigned n_exp, unsigned slice, unsigned nblk,
                                              bool glu) {
    const unsigned rows = (unsigned)meta[3 * n_exp] * MPF_BM; /* tilep[n_exp] padded rows */
    const unsigned tid = threadIdx.x, tstride = PLOW_THREADS;
    if (glu) {
        unsigned char* o = (unsigned char*)Cout;
        const size_t nb = (size_t)rows * (N >> 1); /* min-safe byte extent over both layouts */
        for (size_t i = (size_t)slice * tstride + tid; i < nb; i += (size_t)nblk * tstride)
            o[i] = 0xFFu;
        return;
    }
    float* const part = (float*)Cout;
    const float nanv = __builtin_nanf("");
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned pidx = row_partidx[r];
        if (pidx == PLOW_EXPERT_UNUSED) continue; /* pad row: it owns no destination */
        for (unsigned c = tid; c < N; c += tstride) part[(size_t)pidx * N + c] = nanv;
    }
}

/* op 85: grouped gate/up + GLU. t0=fu_g t1=xn2([T,H]) t2=wtab t3=stab t4=meta t5=row_token
 *   i0=I_moe(N) i1=H(K) i2=n_exp i3=fp8 i5=act f0/f1=situ betas
 * The betas ride in f0/f1 exactly as on the decode `d_moe_expert_glu_fp8_blk` path, and both
 * epilogues below call `moe_glu` (the pair form), never `moe_act` — situ transforms the UP
 * branch too, so a gate-only call would leave `up` un-clipped. For every other activation
 * `moe_glu` is byte-identical to the `moe_act(g, act) * u` it replaces. */
__device__ void d_moe_group_glu_pf(bf16* fu, const bf16* xn2, const unsigned long long* wtab,
                                   const unsigned long long* stab, const int* meta,
                                   const unsigned* row_token, unsigned I_moe, unsigned H,
                                   unsigned n_exp, unsigned enc, unsigned act, unsigned slice,
                                   unsigned nblk, float beta, float lbeta, bf16* lds,
                                   const unsigned char* xscale = nullptr,
                                   const unsigned* row_partidx = nullptr,
                                   unsigned char* fu_scale = nullptr, unsigned shuf = 0,
                                   unsigned a8 = 0) {
    (void)xscale;
    /* a8: on the block-fp8/bf16 arms t6/t7 carry the fp8 A rows + per-token f32 scales
     * (A4W4 owns those slots for its own operands and never sets a8). */
    const unsigned char* aq = a8 ? (const unsigned char*)row_partidx : nullptr;
    const float* as_row = a8 ? (const float*)fu_scale : nullptr;
#if PLOW_MOE_PF_A4W4
    if (enc == PLOW_MOE_ENC_MXFP4) {
        /* A4W4. `fu` is the MXFP4 gathered intermediate and `fu_scale` its E8M0 rows; the
         * epilogue IS the fused bridge (SwiGLU + quantize + scale write in the sorted layout),
         * so no bf16 intermediate exists anywhere on this path and there is no separate bridge
         * op. row_partidx is needed here, not just in DOWN, so the bridge can skip PAD rows. */
        d_moe_group_pf_a4w4<true>((void*)fu, (const void*)xn2, nullptr, wtab, stab, meta,
                                  row_token, row_partidx, nullptr, fu_scale, I_moe, H, n_exp, act,
                                  slice, nblk, beta, lbeta, (void*)lds);
        return;
    }
#else
    (void)fu_scale;
#endif
    /* No body for this encoding in THIS object — refuse, do not alias onto bf16. See the
     * `moe_pf_refuse` header. Under PLOW_MOE_PF_A4W4 the MXFP4 case already returned above, so
     * what reaches here is MXFP4-without-the-flag or an out-of-range field. */
    if (enc > PLOW_MOE_ENC_FP8BLK) {
        moe_pf_refuse((void*)fu, meta, row_partidx, I_moe, n_exp, slice, nblk, true);
        return;
    }
    if (enc == PLOW_MOE_ENC_FP8BLK)
        d_moe_group_pf_t<true, true>(fu, xn2, wtab, stab, meta, row_token, nullptr, nullptr, I_moe,
                                     H, n_exp, act, slice, nblk, beta, lbeta, shuf, lds, aq,
                                     as_row, a8);
    else
        d_moe_group_pf_t<false, true>(fu, xn2, wtab, stab, meta, row_token, nullptr, nullptr, I_moe,
                                      H, n_exp, act, slice, nblk, beta, lbeta, 0u, lds, aq, as_row,
                                      a8);
}

/* op 86: grouped down + gate-scale + scatter. t0=part([T*k,H] f32) t1=fu_g t2=wtab t3=stab
 *   t4=meta t6=row_partidx t7=row_gate   i0=H(N) i1=I_moe(K) i2=n_exp i3=fp8 */
__device__ void d_moe_group_down_pf(float* part, const bf16* fu, const unsigned long long* wtab,
                                    const unsigned long long* stab, const int* meta,
                                    const unsigned* row_partidx, const float* row_gate, unsigned H,
                                    unsigned I_moe, unsigned n_exp, unsigned enc, unsigned slice,
                                    unsigned nblk, bf16* lds,
                                    const unsigned char* fu_scale = nullptr, unsigned shuf = 0,
                                    unsigned part16 = 0
#if PLOW_MOE_PF_ATOMIC
                                    ,
                                    unsigned atom_ksh = 0
#endif
#if PLOW_MOE_PF_DET
                                    ,
                                    unsigned det_ksh = 0
#endif
                                    ) {
#if PLOW_MOE_PF_A4W4
    if (enc == PLOW_MOE_ENC_MXFP4) { /* A = the bridge's MXFP4 output + its E8M0 rows */
        d_moe_group_pf_a4w4<false>((void*)part, (const void*)fu, fu_scale, wtab, stab, meta,
                                   nullptr, row_partidx, row_gate, nullptr, H, I_moe, n_exp, 0,
                                   slice, nblk, 0.0f, 0.0f, (void*)lds);
        return;
    }
#else
    (void)fu_scale;
#endif
    if (enc > PLOW_MOE_ENC_FP8BLK) { /* no body for this encoding here — see moe_pf_refuse */
        moe_pf_refuse((void*)part, meta, row_partidx, H, n_exp, slice, nblk, false);
        return;
    }
    if (enc == PLOW_MOE_ENC_FP8BLK)
        d_moe_group_pf_t<true, false>(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate, H,
                                      I_moe, n_exp, 0, slice, nblk, 0.0f, 0.0f, shuf, lds, nullptr,
                                      nullptr, 0u, part16 MPF_ATOM_ARG MPF_DET_ARG);
    else
        d_moe_group_pf_t<false, false>(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate,
                                       H, I_moe, n_exp, 0, slice, nblk, 0.0f, 0.0f, 0u, lds,
                                       nullptr, nullptr, 0u, part16 MPF_ATOM_ARG MPF_DET_ARG);
}

/* ARM MARKERS for the activation-side flags (packet fields i[7] on ops 85/86/87). The loader's
 * `check_prefill_object` refuses a blob that requires an arm the object's symbol table cannot
 * prove — without these, a part16 blob on an old object would store f32 into a half-sized part
 * buffer (heap overrun) and an a8 blob would matmul fp8 bytes as bf16. Unconditional, like
 * op_gemm.h's capacity marker: any object built from this source HAS the arms. */
extern "C" __device__ unsigned plow_moe_pf_part16_arm = 1;
extern "C" __device__ unsigned plow_moe_pf_a8_arm = 1;
#if PLOW_MOE_PF_A4W4
extern "C" __device__ unsigned plow_moe_pf_a4w4_arm = 1;
#endif
#if PLOW_MOE_PF_ATOMIC
/* CONDITIONAL, unlike the two above: PLOW_MOE_PF_ATOMIC is a BUILD axis (default off, so the
 * default object is byte-identical), and a blob emitted with the fused decomposition on an
 * object without it would have op 86 scatter into a [T,H]-sized accumulator as if it were the
 * [T*k,H] part buffer -- k-way heap overrun and silent garbage. The marker is what lets
 * `check_prefill_object` refuse that pairing. */
extern "C" __device__ unsigned plow_moe_pf_atomic_arm = 1;
#endif
#if PLOW_MOE_PF_DET
/* CONDITIONAL for the same reason: a DET blob on an object without the arm would have op 86 take
 * the `part` scatter branch with Cout pointing at a [T,H] f64 accumulator -- a k/2-fold heap
 * overrun -- and op 87 would read f64 bytes as f32. */
extern "C" __device__ unsigned plow_moe_pf_det_arm = 1;
#endif

/* --- op 87: T-TOKEN COMBINE (PLOW_DOP_MOE_COMBINE_PF) --------------------------------------
 * out[t] = residual[t] + shared[t] + Σ_slot part[t*k + slot], f32 accumulate in FIXED slot
 * order — the same expression and the same order as the decode d_moe_combine, so at T=1 this
 * is bit-identical to it. Grid-strided over (token, h) so it fills every CU at any T.
 *   t0=out t1=residual([T,H]) t2=shared([T,H] or none) t3=part([T*k,H] f32)  i0=H i1=k i2=T */
__device__ void d_moe_combine_pf(bf16* out, const bf16* residual, const bf16* shared,
                                 const float* part, unsigned H, unsigned k, unsigned T,
                                 unsigned slice, unsigned nblk, unsigned part16 = 0
#if PLOW_MOE_PF_DET
                                 /* PLOW_MOE_PF_DET: `part` is the [T,H] f64 FIXED-POINT
                                  * accumulator op 86 summed in place (k == 1), so one contiguous
                                  * stream, unscaled here. See the axis header. */
                                 ,
                                 unsigned det = 0
#endif
) {
    const size_t total = (size_t)T * H;
    const size_t gid = (size_t)slice * PLOW_THREADS + threadIdx.x;
    const size_t stride = (size_t)nblk * PLOW_THREADS;
    const bf16* part_h = (const bf16*)part; /* part16: DOWN scattered bf16, same slot layout */
    for (size_t i = gid; i < total; i += stride) {
        const unsigned tok = (unsigned)(i / H), h = (unsigned)(i - (size_t)tok * H);
        float acc = residual ? bf2f(residual[i]) : 0.0f; /* optional — see d_moe_combine */
        if (shared) acc += bf2f(shared[i]);
#if PLOW_MOE_PF_DET
        if (det) {
            acc += (float)(((const double*)part)[i] * MPF_DET_UNSCALE);
        } else
#endif
        if (part16) {
            const bf16* pt = part_h + (size_t)tok * k * H;
            for (unsigned j = 0; j < k; j++) acc += bf2f(pt[(size_t)j * H + h]);
        } else {
            const float* pt = part + (size_t)tok * k * H;
            for (unsigned j = 0; j < k; j++) acc += pt[(size_t)j * H + h];
        }
        st_act1(&out[i], f2bf(acc));
    }
}

/* ==========================================================================================
 * GEMMA-4 26B-A4B MoE — the AMD twins of runtime/nvidia/op_moe.cuh's `_gemma` family.
 *                                                                        [GEMMA4-MOE-AMD]
 *
 * WHY THIS IS A SEPARATE FAMILY AND NOT `d_moe_router` RENAMED. The Gemma-4 router is not the
 * generic MoE router this file already has, and the four differences are all load-bearing:
 *
 *   1. a WEIGHTLESS RMS over the residual (no gamma) scales the router input,
 *   2. a PER-CHANNEL `scale[H]` multiplies it, and
 *   3. a `root` exponent (H^-0.5) folds in with it, so the router sees
 *          h2[h] = resid[h] * rsqrt(mean(resid^2)+eps) * scale[h] * root,
 *   4. and the gate is `softmax -> top-k -> norm_topk -> * per_expert_scale[winner]` — a
 *      PER-EXPERT multiplier (`pes`), where the generic router has one scalar `route_scale`.
 *
 * `d_moe_router` has none of those: it takes x already normalised, has no scale/root/pes
 * operands, and its `flags` select sigmoid-vs-softmax and norm_topk. Substituting it would
 * compile, would produce plausible logits, and would route to a DIFFERENT EXPERT SET.
 *
 * The EXPERT weights are also laid out differently. Gemma fuses gate and up into ONE tensor per
 * expert, `ewt[e*2+0]` -> [2*I_moe, H] with gate rows [0,I) and up rows [I,2I), and
 * `ewt[e*2+1]` -> down [H, I_moe]. That is a TWO-slot expert table; the generic AMD path uses a
 * THREE-slot one ({gate, up, down}). Two ULLs per expert, not three: an off-by-one here reads
 * another expert's weights and is finite, fluent and wrong.
 *
 * BIT-PARITY WITH sm_120, AND WHERE IT STOPS. Every arithmetic expression below is the sm_120
 * body's, in the same association, with two documented exceptions that are structural to a
 * 64-lane machine:
 *   - block reductions fold PLOW_WAVES wave partials where sm_120 folds (nth/32) warp partials,
 *     so `invrms` and the combine's RMS differ in the last f32 ulp;
 *   - the vectorised wave dots partition K over 64 lanes, not 32.
 * The SELECTION (top-k) is bit-exact by construction: it runs on one thread over an f32 score
 * array with the same packed-key/lowest-id rule, so the expert SET is reproducible.
 *
 * TWO FLAGS, for the reason every other axis here has one — this interpreter inlines every arm:
 *   PLOW_MOE_GEMMA     ops 61-72  (decode; 65/66 additionally need PLOW_FP8)
 *   PLOW_MOE_GEMMA_PF  ops 73-77 + 81/82 (prefill; the grouped-expert MFMA body)
 * Both default 0, so every object built before this section is byte-identical.
 * ========================================================================================== */
#ifndef PLOW_MOE_GEMMA
#define PLOW_MOE_GEMMA 0
#endif
#ifndef PLOW_MOE_GEMMA_PF
#define PLOW_MOE_GEMMA_PF 0
#endif

#if PLOW_MOE_GEMMA || PLOW_MOE_GEMMA_PF

/* Scratch carve for every Gemma body that needs BOTH a block reduction and a working array.
 * `sm->part` cannot serve: `plow_smem` is a union, so it aliases the arena these ops reduce
 * OVER (interp.hip:793 records the same trap for the fused-norm GEMV). 16 floats keeps the
 * working area 16-byte aligned for the float4-ish accesses below. */
#define GMOE_RED(a)   (a)
#define GMOE_WORK(a)  ((a) + 16)

/* gelu_pytorch_tanh, in the same fma form as sm120_common.cuh act_gelu_tanh and op_moe.cuh
 * plow_moe_gelu_tanh. NOT `moe_act(x, 0)`: that one spells the polynomial differently
 * (0.5*x*(1+tanh(k*(x + 0.044715 x^3)))) and Gemma's reference is this one. Same value, and
 * they are written separately so neither drifts onto the other's caller. */
__device__ __forceinline__ float gmoe_gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

/* K elements one 64-lane wave consumes per vector pass: 64 lanes x 8 bf16. */
#define GMOE_STEP (PLOW_WAVE * 8u)
/* Independent weight loads a wave keeps in flight. 4 is the depth op_gemm.h's fp8 GEMV settled
 * on for CDNA3; the GLU arm holds TWO streams (gate and up) so it uses half. */
#ifndef GMOE_UNROLL
#define GMOE_UNROLL 4
#endif
/* GMOE_UNROLL_GLU = 4 WAS BUILT AND A/B'd, AND THE EFFECT IS NOT THERE. Doubling the depth (8
 * weight loads in flight instead of 4, gate and up together) was the obvious lever on the GLU
 * arm's 26-36% of HBM peak. Alternating the two objects six times inside one GPU lease, the
 * numbers cluster into TWO STATES — op62 0.035/op63 0.024 ms, and op62 0.046/op63 0.039 —
 * and BOTH objects land in both, with op63 (which the unroll does not touch at all) moving in
 * lockstep. That is the box's clock/power state, not the kernel. This is exactly the trap the
 * branch's own A/B notes record ("the variance is probably thermal carryover — the box is
 * SHARED"), and it is why the op-63 sub-group split below is measured INTERLEAVED in one process
 * instead. Depth 2 stands until something can measure a difference. */
#ifndef GMOE_UNROLL_GLU
#define GMOE_UNROLL_GLU 2
#endif

/* bf16 dot of one output row against one activation row, reduced across a 64-lane wave.
 * 16-byte loads, GMOE_UNROLL in flight before any is consumed — the same shape as the dense
 * decode GEMV. K need not be a GMOE_STEP multiple: the overshoot lanes load nothing and skip. */
__device__ __forceinline__ float gmoe_wave_dot_bf16(const bf16* x, const bf16* w, unsigned K,
                                                    unsigned lane) {
    const unsigned nchunk = (K + GMOE_STEP - 1u) / GMOE_STEP;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c += GMOE_UNROLL) {
        bf16v8 wv[GMOE_UNROLL];
        unsigned kk[GMOE_UNROLL];
#pragma unroll
        for (int u = 0; u < GMOE_UNROLL; u++) {
            kk[u] = (c + (unsigned)u) * GMOE_STEP + lane * 8u;
            wv[u] = (kk[u] < K) ? ld_glob8(as_glob(w) + kk[u]) : bf16v8_zero();
        }
#pragma unroll
        for (int u = 0; u < GMOE_UNROLL; u++) {
            if (kk[u] >= K) continue;
            acc = dot8(wv[u], ld_glob8(as_glob(x) + kk[u]), acc);
        }
    }
    return wave_sum(acc);
}

/* Per-output-channel e4m3 twin. The weight bytes are OCP e4m3 and `fp8v8_to_bf16v8` decodes
 * them EXACTLY (3 mantissa bits into bf16's 7) — on CDNA3 through the software OCP path in
 * amd_arch.h, never the hardware fnuz convert. The per-channel scale is the caller's epilogue. */
__device__ __forceinline__ float gmoe_wave_dot_fp8(const bf16* x, const unsigned char* w,
                                                   unsigned K, unsigned lane) {
    const unsigned nchunk = (K + GMOE_STEP - 1u) / GMOE_STEP;
    float acc = 0.0f;
    for (unsigned c = 0; c < nchunk; c += GMOE_UNROLL) {
        fp8v8 wq[GMOE_UNROLL];
        unsigned kk[GMOE_UNROLL];
#pragma unroll
        for (int u = 0; u < GMOE_UNROLL; u++) {
            kk[u] = (c + (unsigned)u) * GMOE_STEP + lane * 8u;
            wq[u] = (kk[u] < K) ? ld_glob_fp8v8(w + kk[u]) : (fp8v8)(0u);
        }
#pragma unroll
        for (int u = 0; u < GMOE_UNROLL; u++) {
            if (kk[u] >= K) continue;
            acc = dot8(fp8v8_to_bf16v8(wq[u]), ld_glob8(as_glob(x) + kk[u]), acc);
        }
    }
    return wave_sum(acc);
}

/* Per-row weightless-RMS scalars for `nrow` rows, into `inv` (an LDS carve). Identical
 * reduction shape to the single-row bodies below, so a batched row is bit-identical to its
 * own B=1 result. */
__device__ __forceinline__ void gmoe_row_rms(float* inv, float* red, const bf16* resid,
                                             unsigned H, unsigned nrow, float eps) {
    for (unsigned r = 0; r < nrow; r++) {
        const bf16* rr = resid + (size_t)r * H;
        float part = 0.0f;
        for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
            const float v = bf2f(rr[h]);
            part += v * v;
        }
        const float s = block_sum(part, red);
        if (threadIdx.x == 0) inv[r] = rsqrtf(s / (float)H + eps);
        __syncthreads();
    }
}

/* The softmax -> top-k -> norm_topk -> per-expert-scale tail, on ONE thread, over an f32
 * score array it owns. Shared by the fused router (op 61), the split TOPK tail (op 68) and the
 * T-token prefill router (op 73) so the three cannot drift.
 *
 * `sc` is CONSUMED (turned into probabilities, then killed entry by entry). The routing-table
 * slots are the win/gate scratch, exactly as op_moe.cuh does it. */
__device__ __forceinline__ void gmoe_softmax_topk_tail(unsigned char* table, float* sc,
                                                       const bf16* pes, unsigned n_exp,
                                                       unsigned k) {
    float m = -1e30f;
    for (unsigned e = 0; e < n_exp; e++) m = fmaxf(m, sc[e]);
    float s = 0.0f;
    for (unsigned e = 0; e < n_exp; e++) { sc[e] = __expf(sc[e] - m); s += sc[e]; }
    for (unsigned e = 0; e < n_exp; e++) sc[e] /= s; /* prob */

    for (unsigned j = 0; j < k; j++) {
        unsigned long long best = 0ull;
        unsigned bid = 0;
        for (unsigned e = 0; e < n_exp; e++) {
            unsigned sb;
            const float scv = sc[e];
            __builtin_memcpy(&sb, &scv, 4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u); /* monotone f32 -> u32 */
            const unsigned long long key =
                ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
            if (key > best) { best = key; bid = e; }
        }
        *(unsigned*)(table + (size_t)j * 8) = bid;
        *(float*)(table + (size_t)j * 8 + 4) = sc[bid];
        sc[bid] = -1e30f; /* kill so the next pass cannot re-pick it */
    }
    float gs = 0.0f;
    for (unsigned j = 0; j < k; j++) gs += *(float*)(table + (size_t)j * 8 + 4);
    for (unsigned j = 0; j < k; j++) {
        const unsigned win = *(unsigned*)(table + (size_t)j * 8);
        float gate = *(float*)(table + (size_t)j * 8 + 4);
        if (gs != 0.0f) gate /= gs;      /* norm_topk — always on for Gemma */
        gate *= bf2f(pes[win]);          /* PER-EXPERT scale, not a scalar route_scale */
        *(float*)(table + (size_t)j * 8 + 4) = gate;
    }
}
#endif /* PLOW_MOE_GEMMA || PLOW_MOE_GEMMA_PF */

#if PLOW_MOE_GEMMA || PLOW_MOE_GEMMA_PF
/* --- ONE ROW of the fused router (op 61 / op 73) ------------------------------------------
 *   r     = weightless_rms(resid)
 *   h2[h] = r[h] * scale[h] * root
 *   sc[e] = sum_h h2[h] * proj[e][h]          (WAVE per expert; f32 tree)
 *   table = softmax -> top-k(lowest-id tie) -> norm_topk -> * pes[winner]
 * `arena` holds red[16] | h2[H] | sc[n_exp]. */
__device__ void d_moe_router_gemma_row(unsigned char* table, const bf16* resid, const bf16* proj,
                                       const bf16* scale, const bf16* pes, unsigned H,
                                       unsigned n_exp, unsigned k, float root, float eps,
                                       float* arena) {
    float* red = GMOE_RED(arena);
    float* h2 = GMOE_WORK(arena);
    float* sc = h2 + H;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;

    float part = 0.0f;
    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
        const float v = bf2f(resid[h]);
        part += v * v;
    }
    const float ss = block_sum(part, red);
    const float invrms = rsqrtf(ss / (float)H + eps);

    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS)
        h2[h] = bf2f(resid[h]) * invrms * bf2f(scale[h]) * root;
    __syncthreads();

    /* WAVE per expert: consecutive lanes read consecutive h (coalesced), and all PLOW_WAVES
     * waves stay busy. The sm_120 decode body assigns one SCALAR THREAD per expert, whose
     * global loads are strided by H and fully uncoalesced; its own prefill twin (op 73) already
     * moved to warp-per-expert for exactly this reason, so this is the shape that survived. */
    for (unsigned e = wave; e < n_exp; e += PLOW_WAVES) {
        const bf16* pr = proj + (size_t)e * H;
        float acc = 0.0f;
        for (unsigned h = lane; h < H; h += PLOW_WAVE) acc = __builtin_fmaf(h2[h], bf2f(pr[h]), acc);
        acc = wave_sum(acc);
        if (lane == 0) sc[e] = acc;
    }
    __syncthreads();

    if (threadIdx.x == 0) gmoe_softmax_topk_tail(table, sc, pes, n_exp, k);
    __syncthreads(); /* arena is reused by the next row */
}
#endif

#if PLOW_MOE_GEMMA
/* --- op 61: PLOW_DOP_MOE_ROUTER_GEMMA -----------------------------------------------------
 * Blocks stride the B batched rows; table/resid rows are [B][k]/[B][H]. */
__device__ void d_moe_router_gemma(unsigned char* table, const bf16* resid, const bf16* proj,
                                   const bf16* scale, const bf16* pes, unsigned H, unsigned n_exp,
                                   unsigned k, float root, float eps, unsigned slice,
                                   unsigned nblk, unsigned nrow, float* arena) {
    const unsigned stride = nblk ? nblk : 1u;
    for (unsigned row = slice; row < nrow; row += stride)
        d_moe_router_gemma_row(table + (size_t)row * k * 8, resid + (size_t)row * H, proj, scale,
                               pes, H, n_exp, k, root, eps, arena);
}

/* --- op 67: PLOW_DOP_MOE_ROUTER_GEMMA_SCORE (the EXACT scorer) ----------------------------
 * One wave per (row, expert) pair. Lanes read adjacent residual/scale/projection elements, then
 * LANE 0 consumes them through shuffles IN INCREASING HIDDEN-INDEX ORDER — so the fmaf chain is
 * the legacy scalar dot's, element for element, while the loads stay coalesced. That ordered
 * replay is the whole point of this opcode; op 69 is the fast twin that gives it up.
 *
 * The weightless-RMS scalar is intentionally recomputed by every block: it is 2*H bf16 bytes,
 * L2-hot, and it avoids a third packet and a global-scratch gate. */
__device__ void d_moe_router_gemma_score(float* score, const bf16* resid, const bf16* proj,
                                         const bf16* scale, unsigned H, unsigned n_exp, float root,
                                         float eps, unsigned slice, unsigned nblk, unsigned nrow,
                                         float* arena) {
    float* red = GMOE_RED(arena);
    float* invs = GMOE_WORK(arena);
    gmoe_row_rms(invs, red, resid, H, nrow, eps);

    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned npair = nrow * n_exp;
    for (unsigned idx = slice * PLOW_WAVES + wave; idx < npair; idx += nblk * PLOW_WAVES) {
        const unsigned row = idx / n_exp;
        const unsigned e = idx - row * n_exp;
        const bf16* rr = resid + (size_t)row * H;
        const bf16* pr = proj + (size_t)e * H;
        const float invrms = invs[row];
        float acc = 0.0f;
        for (unsigned h0 = 0; h0 < H; h0 += PLOW_WAVE) {
            const unsigned h = h0 + lane;
            float term_h2 = 0.0f, term_w = 0.0f;
            if (h < H) {
                term_h2 = bf2f(rr[h]) * invrms * bf2f(scale[h]) * root;
                term_w = bf2f(pr[h]);
            }
#pragma unroll 8
            for (unsigned src = 0; src < PLOW_WAVE; src++) {
                const float a = __shfl(term_h2, (int)src, PLOW_WAVE);
                const float b = __shfl(term_w, (int)src, PLOW_WAVE);
                if (lane == 0 && h0 + src < H) acc = __builtin_fmaf(a, b, acc);
            }
        }
        if (lane == 0) score[(size_t)row * n_exp + e] = acc;
    }
}

/* --- op 69: PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST -----------------------------------------
 * Association-CHANGING twin of op 67: same block/expert mapping and the same RMS transform, but
 * an ordinary strided lane dot plus a wave reduction replaces the ordered shuffle replay. Kept a
 * DISTINCT opcode so a fast experiment can never silently change an exact packet. */
__device__ void d_moe_router_gemma_score_fast(float* score, const bf16* resid, const bf16* proj,
                                              const bf16* scale, unsigned H, unsigned n_exp,
                                              float root, float eps, unsigned slice, unsigned nblk,
                                              unsigned nrow, float* arena) {
    float* red = GMOE_RED(arena);
    float* invs = GMOE_WORK(arena);
    gmoe_row_rms(invs, red, resid, H, nrow, eps);

    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned npair = nrow * n_exp;
    for (unsigned idx = slice * PLOW_WAVES + wave; idx < npair; idx += nblk * PLOW_WAVES) {
        const unsigned row = idx / n_exp;
        const unsigned e = idx - row * n_exp;
        const bf16* rr = resid + (size_t)row * H;
        const bf16* pr = proj + (size_t)e * H;
        const float invrms = invs[row];
        float acc = 0.0f;
        for (unsigned h = lane; h < H; h += PLOW_WAVE)
            acc = __builtin_fmaf(bf2f(rr[h]) * invrms * bf2f(scale[h]) * root, bf2f(pr[h]), acc);
        acc = wave_sum(acc);
        if (lane == 0) score[(size_t)row * n_exp + e] = acc;
    }
}

/* --- op 68: PLOW_DOP_MOE_ROUTER_GEMMA_TOPK ------------------------------------------------
 * The tail of the split router: one block per row (blocks stride B), the serial
 * softmax/selection/norm/per-expert-scale order preserved verbatim. */
__device__ void d_moe_router_gemma_topk(unsigned char* table, const float* score, const bf16* pes,
                                        unsigned n_exp, unsigned k, unsigned slice, unsigned nblk,
                                        unsigned nrow, float* arena) {
    float* sc = GMOE_WORK(arena);
    const unsigned stride = nblk ? nblk : 1u;
    for (unsigned row = slice; row < nrow; row += stride) {
        for (unsigned e = threadIdx.x; e < n_exp; e += PLOW_THREADS)
            sc[e] = score[(size_t)row * n_exp + e];
        __syncthreads();
        if (threadIdx.x == 0)
            gmoe_softmax_topk_tail(table + (size_t)row * k * 8, sc, pes, n_exp, k);
        __syncthreads(); /* arena reused by the next row */
    }
}

/* --- op 62: PLOW_DOP_MOE_EXPERT_GLU_GEMMA -------------------------------------------------
 * fu[slot][n] = gelu_tanh(gate_e . x) * (up_e . x), FUSED gate_up weight: gate row n at
 * ewt[e*2+0] + n*H, up row n at the SAME base + (I_moe+n)*H. One WAVE per (slot, channel).
 * A skipped slot (sentinel id, or a null base under expert parallelism) writes nothing —
 * the DOWN op zeroes its partial, so the combine still sums a deterministic zero.
 * B>1: x is [B][H], table [B][k], fu [B][k][I_moe]; the sweep is over the B*k slots. */
__device__ void d_moe_expert_glu_gemma(bf16* fu, const bf16* x, const unsigned char* table,
                                       const unsigned long long* ewt, unsigned k, unsigned I_moe,
                                       unsigned H, unsigned n_exp, unsigned slice, unsigned nblk,
                                       unsigned nrow) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned nchunk = (H + GMOE_STEP - 1u) / GMOE_STEP;

    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += nblk * PLOW_WAVES) {
        /* CHANNEL-MAJOR at B>1 so slots sharing an expert hit its weight rows in L2 instead of
         * re-reading HBM; slot-major at B==1, which is what the single-row packet emits. */
        unsigned slot, n;
        if (nrow == 1u) { slot = f / I_moe; n = f - slot * I_moe; }
        else            { n = f / nslot;    slot = f - n * nslot; }
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
        if (gub == 0ull) continue;
        const bf16* gu = (const bf16*)(size_t)gub;
        const bf16* grow = gu + (size_t)n * H;
        const bf16* urow = gu + (size_t)(I_moe + n) * H;
        const bf16* xr = x + (size_t)(slot / k) * H;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GMOE_UNROLL_GLU) {
            bf16v8 gv[GMOE_UNROLL_GLU], uv[GMOE_UNROLL_GLU];
            unsigned kk[GMOE_UNROLL_GLU];
#pragma unroll
            for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                kk[i] = (c + (unsigned)i) * GMOE_STEP + lane * 8u;
                gv[i] = (kk[i] < H) ? ld_glob8(as_glob(grow) + kk[i]) : bf16v8_zero();
            }
#pragma unroll
            for (int i = 0; i < GMOE_UNROLL_GLU; i++)
                uv[i] = (kk[i] < H) ? ld_glob8(as_glob(urow) + kk[i]) : bf16v8_zero();
#pragma unroll
            for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_glob8(as_glob(xr) + kk[i]);
                ag = dot8(gv[i], xv, ag);
                au = dot8(uv[i], xv, au);
            }
        }
        const float g = wave_sum(ag), u = wave_sum(au);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(gmoe_gelu_tanh(g) * u);
    }
}

/* --- op 71: PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA --------------------------------------------
 * op 62 with the pre-FFN RMSNorm FUSED IN: takes the RAW residual plus gamma and forms
 * xn = resid * rsqrt(mean(resid^2)+eps) * gamma once per block, in the arena, instead of a
 * separate RmsNorm packet and its counter gate. Staging xn is what makes the fusion free: the
 * unfused sm_120 body recomputes it from global for EVERY output channel — 5632 channels x
 * H=2816 of redundant reads per layer.
 *
 * xn IS STAGED AS f32, NOT bf16, AND THAT COSTS NOTHING ON CDNA3. The bf16 staging is what the
 * H100 twin does (op_moe.cuh's PLOW_MOE_XN_BF16) and it was tried here first; it MEASURED an
 * 8x relative error on the small-magnitude outputs, because gelu_tanh(g) is a cancellation for
 * g < 0 — the whole output is act(g)*u, so a 2^-9 perturbation of every xn element lands
 * amplified wherever act(g) is near zero. That is a real accuracy loss, not a tolerance
 * question, and it is avoidable HERE: CDNA3 has no packed bf16 dot (amd_arch.h), so
 * `plow_dot2_bf16` already widens both operands to f32 — feeding an f32 activation skips a
 * widening rather than adding one. `gmoe_dot8_wx` below is that dot.
 *
 * The f32 row is 4 bytes/element, so a wide batch will not fit the arena; past GMOE_XN_MAX_F
 * the body falls back to recomputing xn per output channel, which is the sm_120 default body's
 * shape — slower, same numbers, and it is the path any nrow beyond the arena takes. */
#ifndef GMOE_XN_MAX_F
#define GMOE_XN_MAX_F 8192u
#endif

/* dot8 against an f32 activation. On CDNA3 this is CHEAPER than dot8: plow_dot2_bf16's gfx942
 * arm bit-casts both bf16 operands up to f32 and issues two FMAs, so half that work is already
 * done here. */
__device__ __forceinline__ float gmoe_dot8_wx(bf16v8 w, const float* xs, float acc) {
#pragma unroll
    for (int j = 0; j < 8; j++) acc = __builtin_fmaf(bf2f(w[j]), xs[j], acc);
    return acc;
}

__device__ void d_moe_expert_glu_norm_gemma(bf16* fu, const bf16* resid, const bf16* gamma,
                                            const unsigned char* table,
                                            const unsigned long long* ewt, unsigned k,
                                            unsigned I_moe, unsigned H, unsigned n_exp, float eps,
                                            unsigned slice, unsigned nblk, unsigned nrow,
                                            bf16* arena) {
    float* red = (float*)arena;
    float* invs = (float*)arena + 16;
    float* xn = invs + ((nrow + 3u) & ~3u);
    const bool staged = (size_t)nrow * H <= GMOE_XN_MAX_F;
    gmoe_row_rms(invs, red, resid, H, nrow, eps);
    if (staged) {
        for (unsigned i = threadIdx.x; i < nrow * H; i += PLOW_THREADS) {
            const unsigned r = i / H, h = i - r * H;
            xn[i] = bf2f(resid[i]) * invs[r] * bf2f(gamma[h]);
        }
    }
    __syncthreads();

    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned nchunk = (H + GMOE_STEP - 1u) / GMOE_STEP;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += nblk * PLOW_WAVES) {
        unsigned slot, n;
        if (nrow == 1u) { slot = f / I_moe; n = f - slot * I_moe; }
        else            { n = f / nslot;    slot = f - n * nslot; }
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long gub = ewt[(size_t)eid * 2 + 0];
        if (gub == 0ull) continue;
        const bf16* gu = (const bf16*)(size_t)gub;
        const bf16* grow = gu + (size_t)n * H;
        const bf16* urow = gu + (size_t)(I_moe + n) * H;
        const unsigned row = slot / k;
        float ag = 0.0f, au = 0.0f;
        if (staged) {
            const float* xr = xn + (size_t)row * H;
            for (unsigned c = 0; c < nchunk; c += GMOE_UNROLL_GLU) {
                bf16v8 gv[GMOE_UNROLL_GLU], uv[GMOE_UNROLL_GLU];
                unsigned kk[GMOE_UNROLL_GLU];
#pragma unroll
                for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                    kk[i] = (c + (unsigned)i) * GMOE_STEP + lane * 8u;
                    gv[i] = (kk[i] < H) ? ld_glob8(as_glob(grow) + kk[i]) : bf16v8_zero();
                }
#pragma unroll
                for (int i = 0; i < GMOE_UNROLL_GLU; i++)
                    uv[i] = (kk[i] < H) ? ld_glob8(as_glob(urow) + kk[i]) : bf16v8_zero();
#pragma unroll
                for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                    if (kk[i] >= H) continue;
                    ag = gmoe_dot8_wx(gv[i], xr + kk[i], ag);
                    au = gmoe_dot8_wx(uv[i], xr + kk[i], au);
                }
            }
        } else {
            const bf16* rr = resid + (size_t)row * H;
            const float inv = invs[row];
            for (unsigned h = lane; h < H; h += PLOW_WAVE) {
                const float xv = bf2f(rr[h]) * inv * bf2f(gamma[h]);
                ag = __builtin_fmaf(xv, bf2f(grow[h]), ag);
                au = __builtin_fmaf(xv, bf2f(urow[h]), au);
            }
        }
        const float g = wave_sum(ag), u = wave_sum(au);
        if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(gmoe_gelu_tanh(g) * u);
    }
}

/* --- op 63: PLOW_DOP_MOE_EXPERT_DOWN_GEMMA ------------------------------------------------
 * part[slot][h] = gate_slot * (down_e[h] . fu[slot]), f32, FIXED slot order so the combine is
 * a deterministic sum. A skipped slot writes an explicit 0.0f — the interpreter's dispatch
 * default writes nothing, so an unwritten partial is indistinguishable from a dead op.
 *
 * SUB-GROUP SPLIT, and the reason is a MEASUREMENT, not symmetry with the GLU arm. DOWN has a
 * SHORT K: I_moe = 704 against a 64-lane wave's 512-element vector pass, so a whole-wave dot
 * runs TWO chunks of which the second is 37% used, and the per-row overhead (expert lookup, a
 * 6-step wave reduction) is amortised over almost nothing. Measured on MI300X at the real shape
 * it was the WORST arm in the block: 663-994 GB/s where the GLU arm reaches 1400-1900.
 *
 * So split the wave into GMOE_DOWN_SG sub-groups, one output channel each: GMOE_DOWN_SG rows in
 * flight per wave with ONE accumulator per lane (no wv[][] array at all), each lane getting a
 * longer contiguous run, and a reduction that is log2(64/SG) steps instead of 6.
 *
 * SG = 8 (eight lanes per sub-group, 64 K-elements per sub-group pass) IS FORCED BY THE SHAPE.
 * The obvious SG = 4 gives 16 lanes and a 128-element pass, and 704 % 128 = 64 — a partial pass
 * on every row, which is what this is trying to remove. 704 = 11 * 64 divides exactly at SG = 8.
 * The guard below falls back to the whole-wave dot on any K that does not divide, so a different
 * model's I_moe is slower, never wrong.
 *
 * NUMERICALLY EQUIVALENT, NOT BIT-IDENTICAL: an 8-lane reduction tree sums the K partition in a
 * different order than a 64-lane one. The CPU reference is a sequential f32 dot either way. */
#ifndef GMOE_DOWN_SG
#define GMOE_DOWN_SG 8u
#endif
#define GMOE_DOWN_SL (PLOW_WAVE / GMOE_DOWN_SG) /* lanes per sub-group */
#define GMOE_DOWN_CH (GMOE_DOWN_SL * 8u)        /* K elements a sub-group covers per pass */

/* The WHOLE-WAVE arm: one output channel per 64-lane wave. Kept as its own function, not folded
 * into an `else`, because it is the A/B NEGATIVE CONTROL for the sub-group split above — a
 * standalone wrapper drives it (test_kernels.hip) so the two can be timed INTERLEAVED in one
 * process. On a shared box that is the only way the comparison means anything; the same
 * discipline the branch's `a6df46a` A/B harness records. It is also the live path for any I_moe
 * the split does not divide. */
__device__ void d_moe_expert_down_gemma_wave(float* part, const bf16* fu,
                                             const unsigned char* table,
                                             const unsigned long long* ewt, unsigned k, unsigned H,
                                             unsigned I_moe, unsigned n_exp, unsigned slice,
                                             unsigned nblk, unsigned nrow) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * H;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += nblk * PLOW_WAVES) {
        unsigned slot, h;
        if (nrow == 1u) { slot = f / H; h = f - slot * H; }
        else            { h = f / nslot; slot = f - h * nslot; }
        const unsigned eid = moe_slot_expert(table, slot);
        float* pslot = part + (size_t)slot * H;
        const unsigned long long db = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
        if (db == 0ull) {
            if (lane == 0) pslot[h] = 0.0f; /* deterministic zero partial */
            continue;
        }
        const bf16* Wd = (const bf16*)(size_t)db;
        const float y = gmoe_wave_dot_bf16(fu + (size_t)slot * I_moe, Wd + (size_t)h * I_moe,
                                           I_moe, lane);
        if (lane == 0) pslot[h] = moe_slot_gate(table, slot) * y;
    }
}

__device__ void d_moe_expert_down_gemma(float* part, const bf16* fu, const unsigned char* table,
                                        const unsigned long long* ewt, unsigned k, unsigned H,
                                        unsigned I_moe, unsigned n_exp, unsigned slice,
                                        unsigned nblk, unsigned nrow) {
    if ((I_moe % GMOE_DOWN_CH) != 0u) { /* a K the split does not divide: the whole-wave arm */
        d_moe_expert_down_gemma_wave(part, fu, table, ewt, k, H, I_moe, n_exp, slice, nblk, nrow);
        return;
    }
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * H;
    {
        const unsigned sg = lane / GMOE_DOWN_SL, sl = lane % GMOE_DOWN_SL;
        const unsigned nch = I_moe / GMOE_DOWN_CH;
        for (unsigned fb = (slice * PLOW_WAVES + wave) * GMOE_DOWN_SG; fb < total;
             fb += nblk * PLOW_WAVES * GMOE_DOWN_SG) {
            const unsigned f = fb + sg;
            bool valid = false, live = false;
            float gate = 0.0f;
            float* dst = nullptr;
            const bf16* wr = nullptr;
            const bf16* xr = nullptr;
            if (f < total) {
                valid = true;
                unsigned slot, h;
                if (nrow == 1u) { slot = f / H; h = f - slot * H; }
                else            { h = f / nslot; slot = f - h * nslot; }
                dst = part + (size_t)slot * H + h;
                const unsigned eid = moe_slot_expert(table, slot);
                const unsigned long long db = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
                if (db != 0ull) {
                    live = true;
                    gate = moe_slot_gate(table, slot);
                    wr = (const bf16*)(size_t)db + (size_t)h * I_moe;
                    xr = fu + (size_t)slot * I_moe;
                }
            }
            float acc = 0.0f;
            if (live) {
                unsigned c = 0;
                for (; c + 2u <= nch; c += 2u) { /* two passes pre-issued */
                    const unsigned k0 = c * GMOE_DOWN_CH + sl * 8u;
                    const unsigned k1 = (c + 1u) * GMOE_DOWN_CH + sl * 8u;
                    const bf16v8 w0 = ld_glob8(as_glob(wr) + k0), w1 = ld_glob8(as_glob(wr) + k1);
                    const bf16v8 x0 = ld_glob8(as_glob(xr) + k0), x1 = ld_glob8(as_glob(xr) + k1);
                    acc = dot8(w0, x0, acc);
                    acc = dot8(w1, x1, acc);
                }
                for (; c < nch; c++) {
                    const unsigned k0 = c * GMOE_DOWN_CH + sl * 8u;
                    acc = dot8(ld_glob8(as_glob(wr) + k0), ld_glob8(as_glob(xr) + k0), acc);
                }
            }
#pragma unroll
            for (int o = 1; o < (int)GMOE_DOWN_SL; o <<= 1) acc += __shfl_xor(acc, o, PLOW_WAVE);
            if (sl == 0u && valid) *dst = live ? gate * acc : 0.0f;
        }
    }
}

/* --- op 64: PLOW_DOP_MOE_COMBINE_GEMMA ----------------------------------------------------
 * moe[h] = sum_slot part[slot][h], f32 in fixed slot order, rounded to bf16. Gemma's own
 * post-norms / residual / layer_scalar are ordinary norm+residual ops (or op 70/72 below). */
__device__ void d_moe_combine_gemma(bf16* moe, const float* part, unsigned H, unsigned k,
                                    unsigned slice, unsigned nblk) {
    const unsigned gid = slice * PLOW_THREADS + threadIdx.x;
    const unsigned stride = nblk * PLOW_THREADS;
    for (unsigned h = gid; h < H; h += stride) {
        float acc = 0.0f;
        for (unsigned slot = 0; slot < k; slot++) acc += part[(size_t)slot * H + h];
        st_act1(&moe[h], f2bf(acc));
    }
}

/* --- op 70: PLOW_DOP_MOE_COMBINE_NORM_GEMMA -----------------------------------------------
 *   sum[h] = Σ_slot part[slot][h]                                  (f32 combine)
 *   out[h] = sum[h] * rsqrt(mean(sum^2)+eps) * gamma[h] + resid[h] (norm + residual)
 * Replaces the three-op tail (combine -> RmsNorm -> Residual) and its two counter gates.
 * ONE BLOCK PER ROW (blocks stride B), so each row is bit-identical to its own B=1 result. */
__device__ void d_moe_combine_norm_gemma(bf16* out, const float* part, const bf16* resid,
                                         const bf16* gamma, unsigned H, unsigned k, float eps,
                                         unsigned slice, unsigned nblk, unsigned nrow,
                                         float* arena) {
    float* red = GMOE_RED(arena);
    float* acc = GMOE_WORK(arena);
    const unsigned stride = nblk ? nblk : 1u;
    for (unsigned row = slice; row < nrow; row += stride) {
        const float* pt = part + (size_t)row * k * H;
        const bf16* res = resid + (size_t)row * H;
        bf16* o = out + (size_t)row * H;
        float ss = 0.0f;
        for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
            float a = 0.0f;
            for (unsigned slot = 0; slot < k; slot++) a += pt[(size_t)slot * H + h];
            acc[h] = a;
            ss += a * a;
        }
        const float t = block_sum(ss, red);
        const float inv = rsqrtf(t / (float)H + eps);
        for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS)
            st_act1(&o[h], f2bf(acc[h] * inv * bf2f(gamma[h]) + bf2f(res[h])));
        __syncthreads(); /* arena reused by the next row */
    }
}

/* --- op 72: PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA -----------------------------------------
 * The whole MoE layer tail in ONE packet — (op 70 -> NormResidualNorm):
 *   b  = h1 + RMSNorm(Σ_slot part, g_pf2)
 *   x  = (x + RMSNorm(b, g_po)) * ls        (the running residual, ROUNDED to bf16)
 *   hn = RMSNorm(x, gn)                      (the next sublayer's input)
 * BIT-EXACT to the pair it replaces: b and the new residual are rounded to bf16 before the
 * next reduction reads them, reproducing the two ops' HBM round trips without the traffic.
 * ONE block. `arena` holds red[16] + one f32[H] staging row, overwritten pass to pass. */
__device__ void d_moe_combine_resid_norm_gemma(bf16* hn, bf16* x, const float* part,
                                               const bf16* h1, const bf16* g_pf2,
                                               const bf16* g_po, const bf16* gn, unsigned H,
                                               unsigned k, float eps, float ls, unsigned slice,
                                               float* arena) {
    if (slice != 0) return;
    float* red = GMOE_RED(arena);
    float* w = GMOE_WORK(arena);

    float ss = 0.0f;
    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
        float a = 0.0f;
        for (unsigned slot = 0; slot < k; slot++) a += part[(size_t)slot * H + h];
        w[h] = a;
        ss += a * a;
    }
    const float inv1 = rsqrtf(block_sum(ss, red) / (float)H + eps);

    ss = 0.0f;
    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
        const bf16 bh = f2bf(w[h] * inv1 * bf2f(g_pf2[h]) + bf2f(h1[h]));
        const float bf = bf2f(bh);
        w[h] = bf;
        ss += bf * bf;
    }
    __syncthreads();
    const float inv2 = rsqrtf(block_sum(ss, red) / (float)H + eps);

    ss = 0.0f;
    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
        const float rf = bf2f(f2bf((bf2f(x[h]) + w[h] * inv2 * bf2f(g_po[h])) * ls));
        w[h] = rf;
        ss += rf * rf;
    }
    __syncthreads();
    const float inv3 = rsqrtf(block_sum(ss, red) / (float)H + eps);

    for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
        const float rf = w[h];
        st_act1(&x[h], f2bf(rf));
        st_act1(&hn[h], f2bf(rf * inv3 * bf2f(gn[h])));
    }
}
#endif /* PLOW_MOE_GEMMA */

#if PLOW_MOE_GEMMA
/* --- ops 65 / 66: PER-OUTPUT-CHANNEL e4m3 GEMMA EXPERTS -----------------------------------
 * The offline quantizer stores one e4m3 row scale per output channel, so the scale factors OUT
 * of the K reduction. `ewt` points at the fp8 gate_up / down rows and `est` at their f32 row
 * scales ([2*I_moe] for gate_up, [H] for down), per expert. The fused gate_up layout, the
 * gelu_tanh epilogue and the sentinel-skip rule are the bf16 bodies', unchanged.
 *
 * THE BYTES ARE OCP e4m3 AND CDNA3's CONVERTER IS NOT. Everything here goes through
 * fp8v8_to_bf16v8, which on gfx942 takes amd_arch.h's software OCP path; calling
 * __builtin_amdgcn_cvt_*_fp8 directly would halve every weight and turn every -0 into a NaN. */
__device__ void d_moe_expert_glu_gemma_fp8(bf16* fu, const bf16* x, const unsigned char* table,
                                           const unsigned long long* ewt,
                                           const unsigned long long* est, unsigned k,
                                           unsigned I_moe, unsigned H, unsigned n_exp,
                                           unsigned slice, unsigned nblk, unsigned nrow) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * I_moe;
    const unsigned nchunk = (H + GMOE_STEP - 1u) / GMOE_STEP;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += nblk * PLOW_WAVES) {
        unsigned slot, n;
        if (nrow == 1u) { slot = f / I_moe; n = f - slot * I_moe; }
        else            { n = f / nslot;    slot = f - n * nslot; }
        const unsigned eid = moe_slot_expert(table, slot);
        if (eid >= n_exp) continue;
        const unsigned long long wb = ewt[(size_t)eid * 2 + 0];
        const unsigned long long sb = est[(size_t)eid * 2 + 0];
        if (wb == 0ull || sb == 0ull) continue;
        const unsigned char* gw = (const unsigned char*)(size_t)wb;
        const unsigned char* grow = gw + (size_t)n * H;
        const unsigned char* urow = gw + (size_t)(I_moe + n) * H;
        const float* sc = (const float*)(size_t)sb;
        const bf16* xr = x + (size_t)(slot / k) * H;
        float ag = 0.0f, au = 0.0f;
        for (unsigned c = 0; c < nchunk; c += GMOE_UNROLL_GLU) {
            fp8v8 gq[GMOE_UNROLL_GLU], uq[GMOE_UNROLL_GLU];
            unsigned kk[GMOE_UNROLL_GLU];
#pragma unroll
            for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                kk[i] = (c + (unsigned)i) * GMOE_STEP + lane * 8u;
                gq[i] = (kk[i] < H) ? ld_glob_fp8v8(grow + kk[i]) : (fp8v8)(0u);
                uq[i] = (kk[i] < H) ? ld_glob_fp8v8(urow + kk[i]) : (fp8v8)(0u);
            }
#pragma unroll
            for (int i = 0; i < GMOE_UNROLL_GLU; i++) {
                if (kk[i] >= H) continue;
                const bf16v8 xv = ld_glob8(as_glob(xr) + kk[i]);
                ag = dot8(fp8v8_to_bf16v8(gq[i]), xv, ag);
                au = dot8(fp8v8_to_bf16v8(uq[i]), xv, au);
            }
        }
        const float g = wave_sum(ag) * sc[n];
        const float u = wave_sum(au) * sc[I_moe + n];
        if (lane == 0) fu[(size_t)slot * I_moe + n] = f2bf(gmoe_gelu_tanh(g) * u);
    }
}

__device__ void d_moe_expert_down_gemma_fp8(float* part, const bf16* fu,
                                            const unsigned char* table,
                                            const unsigned long long* ewt,
                                            const unsigned long long* est, unsigned k, unsigned H,
                                            unsigned I_moe, unsigned n_exp, unsigned slice,
                                            unsigned nblk, unsigned nrow) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nslot = nrow * k;
    const unsigned total = nslot * H;
    for (unsigned f = slice * PLOW_WAVES + wave; f < total; f += nblk * PLOW_WAVES) {
        unsigned slot, h;
        if (nrow == 1u) { slot = f / H; h = f - slot * H; }
        else            { h = f / nslot; slot = f - h * nslot; }
        const unsigned eid = moe_slot_expert(table, slot);
        float* pslot = part + (size_t)slot * H;
        const unsigned long long wb = (eid < n_exp) ? ewt[(size_t)eid * 2 + 1] : 0ull;
        const unsigned long long sb = (eid < n_exp) ? est[(size_t)eid * 2 + 1] : 0ull;
        if (wb == 0ull || sb == 0ull) {
            if (lane == 0) pslot[h] = 0.0f;
            continue;
        }
        const unsigned char* Wd = (const unsigned char*)(size_t)wb;
        const float* sc = (const float*)(size_t)sb;
        const float y = gmoe_wave_dot_fp8(fu + (size_t)slot * I_moe, Wd + (size_t)h * I_moe,
                                          I_moe, lane) * sc[h];
        if (lane == 0) pslot[h] = moe_slot_gate(table, slot) * y;
    }
}
#endif /* PLOW_MOE_GEMMA (the fp8 expert BODIES; interp.hip gates the CASES on PLOW_FP8, which
        * is the weight-encoding axis they belong to) */

#if PLOW_MOE_GEMMA_PF
/* ==========================================================================================
 * GEMMA-4 grouped-MoE PREFILL (T > 1) — ops 73-77, 81/82.
 *
 * The align/sort op is EXACTLY d_moe_align_pf (see the op 74 wrapper below): same histogram,
 * same MPF_BM-padded prefix, same meta layout, same (row_token, row_partidx, row_gate) scatter.
 * Reusing it rather than copying it is what keeps the pad convention — and therefore the two
 * grouped GEMMs' tile arithmetic — provably consistent with the rest of this file.
 *
 * The two grouped GEMMs are a Gemma-shaped twin of `d_moe_group_pf_t`, and the reason it is a
 * twin rather than a fourth instantiation of that template is the WEIGHT TABLE: Gemma carries
 * TWO ULLs per expert with gate and up FUSED into one [2I,H] tensor, where every other MoE in
 * this file carries THREE with gate and up separate. That is a different indexing rule in the
 * innermost staging macro, not a flag.
 *
 * ---- LDS, AND WHY CDNA3 SINGLE-BUFFERS ----
 * The BM=64 x BN=256 x BK=64 tile is 40,960 B. Double-buffered that is 81,920 B, which fits
 * CDNA4's 160 KiB workgroup and does NOT fit CDNA3's 64 KiB — the compiler enforces the cap
 * ("local memory (81920) exceeds limit (65536)"), so this is a hard structural bound, not a
 * tuning choice. GMPF_DBUF drops to one buffer on CDNA3, which is the same answer the dense
 * GEMM reached on this branch (GM_DBUF=1): the global fetch still issues into REGISTERS at the
 * top of the k-tile and lands behind the whole MFMA run, so only the LDS store is exposed, and
 * one extra barrier per k-tile is the whole cost.
 * ========================================================================================== */
#if (2 * MPF_TILE * 2) <= PLOW_LDS_MAX_BYTES
#define GMPF_DBUF 2
#else
#define GMPF_DBUF 1
#endif
#define GMPF_LDS_HALVES (GMPF_DBUF * MPF_TILE)

/* --- op 73: PLOW_DOP_MOE_ROUTER_GEMMA_PF --------------------------------------------------
 * Block-per-token loop of the exact decode router row. Bit-identical per token to decode by
 * construction: it is the same function, with the table/resid rows advanced. */
__device__ void d_moe_router_gemma_pf(unsigned char* table, const bf16* resid, const bf16* proj,
                                      const bf16* scale, const bf16* pes, unsigned H,
                                      unsigned n_exp, unsigned k, unsigned T, float root,
                                      float eps, unsigned slice, unsigned nblk, float* arena) {
    for (unsigned tok = slice; tok < T; tok += nblk)
        d_moe_router_gemma_row(table + (size_t)tok * k * 8, resid + (size_t)tok * H, proj, scale,
                               pes, H, n_exp, k, root, eps, arena);
}

/* --- op 77: PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF --------------------------------------------
 * Block-per-token loop of op 70: out[t] = RMSNorm(Σ_slot part[t][slot], gamma) + h1[t]. */
__device__ void d_moe_combine_norm_gemma_pf(bf16* out, const float* part, const bf16* h1,
                                            const bf16* gamma, unsigned H, unsigned k, unsigned T,
                                            float eps, unsigned slice, unsigned nblk,
                                            float* arena) {
    float* red = GMOE_RED(arena);
    float* acc = GMOE_WORK(arena);
    for (unsigned tok = slice; tok < T; tok += nblk) {
        const float* pt = part + (size_t)tok * k * H;
        const bf16* res = h1 + (size_t)tok * H;
        bf16* o = out + (size_t)tok * H;
        float ss = 0.0f;
        for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS) {
            float a = 0.0f;
            for (unsigned slot = 0; slot < k; slot++) a += pt[(size_t)slot * H + h];
            acc[h] = a;
            ss += a * a;
        }
        const float t = block_sum(ss, red);
        const float inv = rsqrtf(t / (float)H + eps);
        for (unsigned h = threadIdx.x; h < H; h += PLOW_THREADS)
            st_act1(&o[h], f2bf(acc[h] * inv * bf2f(gamma[h]) + bf2f(res[h])));
        __syncthreads(); /* arena reused by the next token */
    }
}

/* --- ops 75 / 76 (+ 81 / 82): THE GROUPED GEMMA EXPERT GEMM -------------------------------
 * ONE body, four modes:
 *
 *   GLU=true   gate/up. A = xn2 rows GATHERED by row_token; B = the expert's FUSED gate_up
 *              tensor, its gate half staged into the low BN/2 rows of the tile and its up half
 *              into the high ones, so the SN axis selects gate-vs-up at the SAME output column
 *              and the GeGLU epilogue needs no cross-lane shuffle. C = fu_g[row][I_moe], bf16.
 *   GLU=false  down. A = fu_g, contiguous per expert segment; B = the expert's down tensor;
 *              C = part[row_partidx[row]][H], f32, SCATTERED and multiplied by row_gate. Pad
 *              rows (row_partidx == PLOW_EXPERT_UNUSED) are dropped.
 *   W8A8=false bf16 A and B (ops 75/76).
 *   W8A8=true  BOTH operands e4m3 with per-row (A) and per-output-channel (B) f32 scales
 *              (ops 81/82). The bytes are decoded EXACTLY to bf16 on the way into LDS and the
 *              bf16 matrix core runs them.
 *
 * WHY W8A8 DECODES TO bf16 INSTEAD OF USING THE fp8 MATRIX CORE, and it is not a shortcut.
 * Every e4m3 value is exact in bf16 (3 mantissa bits into 7) and every e4m3 x e4m3 product is
 * exact in f32, so decode-then-bf16-MFMA computes the SAME products as an fp8 MFMA and differs
 * only in accumulation grouping. On CDNA3 it also costs nothing: the part's fp8 matrix core is
 * K=16 and runs at the SAME MACs/cycle as its bf16 one (amd_arch.h), so the 2x an fp8 core buys
 * on CDNA4 does not exist here — fp8 buys memory footprint, which this path keeps in full. And
 * it side-steps the e4m3fnuz/OCP divergence in the matrix core entirely.
 *
 * `Ain` is bf16* when W8A8 is false and unsigned char* when it is true. `ascale` is the A-side
 * per-row f32 scale (per TOKEN on the GLU arm, indexed by row_token; per GATHERED ROW on the
 * DOWN arm) and is unread when W8A8 is false. */
template <bool GLU, bool W8A8>
__device__ void d_moe_group_gemma_pf_t(void* __restrict__ Cout, const void* __restrict__ Ain,
                                       const float* __restrict__ ascale,
                                       const unsigned long long* __restrict__ ewt,
                                       const unsigned long long* __restrict__ est,
                                       const int* __restrict__ meta,
                                       const unsigned* __restrict__ row_token,
                                       const unsigned* __restrict__ row_partidx,
                                       const float* __restrict__ row_gate, unsigned N, unsigned K,
                                       unsigned n_exp, unsigned act, unsigned slice, unsigned nblk,
                                       bf16* lds) {
    constexpr int SM = MPF_SM, SN = MPF_SN; /* 1, 2 */
    constexpr int APT = MPF_BM * MPF_BK / PLOW_THREADS;
    constexpr int BPT = MPF_BN * MPF_BK / PLOW_THREADS;
    constexpr int APASS = APT / 8, BPASS = BPT / 8;
    /* This twin shares the MPF tile macros but has NO sub-quantum arm: at MPF_BK=32 (the
     * batched-decode escape hatch) APASS truncates to zero and it would stage no A tile —
     * the recorded silent-wrong failure. Refuse; build such objects with PLOW_MOE_GEMMA_PF=0
     * (the OCC4 batched recipe already drops the _PF arm). */
    static_assert(APT >= 8 && APT % 8 == 0, "Gemma grouped-MoE A tile below the 8-half staging quantum");
    static_assert(BPT >= 8 && BPT % 8 == 0, "Gemma grouped-MoE B tile below the 8-half staging quantum");
    constexpr unsigned NB = GLU ? (MPF_BN / 2) : MPF_BN;
    constexpr unsigned DB = GMPF_DBUF;

    const unsigned lane = threadIdx.x & 63u;
    const unsigned wave = threadIdx.x >> 6;
    const unsigned wm = wave / MPF_WN, wn = wave % MPF_WN;
    const unsigned frow = mfma_frag_row(lane);

    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned total_tiles = (unsigned)tilep[n_exp];
    const unsigned tn = (N + NB - 1u) / NB;
    const unsigned n_tiles = total_tiles * tn;
    const unsigned NT = (K + MPF_BK - 1u) / MPF_BK;

    const bf16* A16 = (const bf16*)Ain;
    const unsigned char* A8 = (const unsigned char*)Ain;

#define GMPF_ASM(b) (lds + (b) * MPF_TILE)
#define GMPF_BSM(b) (lds + (b) * MPF_TILE + MPF_BM * MPF_STRIDE)

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned mt = lin / tn, nt = lin % tn;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF_BM;
        const unsigned n0 = nt * NB;

        /* TWO ULLs per expert (gate_up fused, down) — NOT three. */
        const unsigned long long wb = ewt[(size_t)e * 2 + (GLU ? 0 : 1)];
        if (wb == 0ull) continue; /* expert-parallel "not mine" sentinel; skip, do not fault */

#if PLOW_MOE_PF_EPI_SIB
        /* ---- DOWN-EPILOGUE ROW-METADATA HOIST, GEMMA TWIN (see the PLOW_MOE_PF_EPI_SIB
         * header note). Lane L takes row L of this m-tile; the epilogue bpermutes row rr out
         * of lane rr. Issued at the tile head so the k-loop below covers the latency, and
         * THREE dwords rather than the grouped GEMM's two: this arm's W8A8 branch also reads
         * a k- and n-invariant per-row `ascale[rowbase + rr]`, and leaving one of the three
         * behind would leave the per-element `s_waitcnt vmcnt(0)` drain exactly where it is. */
        unsigned epi_pidx = 0u;
        int epi_gate = 0, epi_asc = 0;
        if constexpr (!GLU) {
            epi_pidx = as_glob(row_partidx)[rowbase + lane];
            epi_gate = __builtin_bit_cast(int, as_glob(row_gate)[rowbase + lane]);
            if constexpr (W8A8) epi_asc = __builtin_bit_cast(int, as_glob(ascale)[rowbase + lane]);
        }
#endif
        const bf16* W16 = (const bf16*)(size_t)wb;
        const unsigned char* W8 = (const unsigned char*)(size_t)wb;
        /* `if constexpr`, not a ternary: `est` is nullptr on the bf16 arms and a ternary would
         * still evaluate the subscript. */
        const float* wsc = nullptr;
        if constexpr (W8A8) wsc = (const float*)(size_t)est[(size_t)e * 2 + (GLU ? 0 : 1)];

        f32x16 acc[SM][SN];
#pragma unroll
        for (int i = 0; i < SM; i++)
#pragma unroll
            for (int j = 0; j < SN; j++) acc[i][j] = (f32x16)(0.0f);

        __align__(16) bf16 ra[APT], rb[BPT];

/* A stage. GLU gathers row_token[rowbase+r] out of the [T,K] activation; DOWN reads the
 * gathered intermediate contiguously from rowbase. A pad row (UNUSED) zero-fills, and zero
 * contributes nothing to any live output. */
#define GMPF_FETCH_A(k0)                                                                       \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                     \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                         \
        const unsigned r = el / MPF_BK, kk = (k0) + (el % MPF_BK);                             \
        unsigned src;                                                                          \
        if constexpr (GLU) src = row_token[rowbase + r];                                       \
        else src = rowbase + r;                                                                \
        if (src != PLOW_EXPERT_UNUSED) {                                                       \
            if constexpr (W8A8)                                                                \
                *(bf16v8*)&ra[it * 8] = mpf_ld_w8(A8 + (size_t)src * K + kk);                  \
            else                                                                               \
                *(bf16v8*)&ra[it * 8] = ld_glob8(as_glob(A16) + (size_t)src * K + kk);         \
        } else                                                                                 \
            _Pragma("unroll") for (int j = 0; j < 8; j++) ra[it * 8 + j] = 0;                  \
    }

/* B stage. Under GLU the tile's low half is the FUSED tensor's gate rows [0,I) and its high
 * half the up rows [I,2I) AT THE SAME OUTPUT COLUMN — one base, an +N row offset, which is the
 * whole difference from the three-pointer grouped GEMM above. */
#define GMPF_FETCH_B(k0)                                                                       \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                     \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                         \
        const unsigned br = el / MPF_BK, kk = (k0) + (el % MPF_BK);                            \
        unsigned r = n0 + br, wrow = r;                                                        \
        if constexpr (GLU) {                                                                   \
            const bool up = (br >= MPF_BN / 2);                                                \
            r = n0 + (up ? br - MPF_BN / 2 : br);                                              \
            wrow = up ? r + N : r;                                                             \
        }                                                                                      \
        if (r < N) {                                                                           \
            if constexpr (W8A8)                                                                \
                *(bf16v8*)&rb[it * 8] = mpf_ld_w8(W8 + (size_t)wrow * K + kk);                 \
            else                                                                               \
                *(bf16v8*)&rb[it * 8] = ld_glob8(as_glob(W16) + (size_t)wrow * K + kk);        \
        } else                                                                                 \
            _Pragma("unroll") for (int j = 0; j < 8; j++) rb[it * 8 + j] = 0;                  \
    }

#define GMPF_COMMIT(buf)                                                                       \
    _Pragma("unroll") for (int it = 0; it < APASS; it++) {                                     \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                         \
        __builtin_memcpy(&GMPF_ASM(buf)[(el / MPF_BK) * MPF_STRIDE +                           \
                                        MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                 \
                         &ra[it * 8], 16);                                                     \
    }                                                                                          \
    _Pragma("unroll") for (int it = 0; it < BPASS; it++) {                                     \
        const unsigned el = threadIdx.x * 8 + it * (PLOW_THREADS * 8);                         \
        __builtin_memcpy(&GMPF_BSM(buf)[(el / MPF_BK) * MPF_STRIDE +                           \
                                        MPF_XORSWZ(el / MPF_BK, el % MPF_BK)],                 \
                         &rb[it * 8], 16);                                                     \
    }

        __syncthreads(); /* the previous tile's fragment readers must be done with the LDS */
        GMPF_FETCH_A(0)
        GMPF_FETCH_B(0)
        GMPF_COMMIT(0)
        __syncthreads();

        unsigned buf = 0;
        for (unsigned kt = 0; kt < NT; kt++) {
            const unsigned kn = (kt + 1u) * MPF_BK;
            /* Issue the next k-tile's global loads into REGISTERS before the MFMA run, in both
             * buffering modes. At DB==1 that is what keeps single-buffering cheap: the load
             * lands behind the MFMAs and only the LDS store is exposed. */
            if (kn < K) { GMPF_FETCH_A(kn) GMPF_FETCH_B(kn) }

#pragma unroll
            for (int q = 0; q < MPF_BK / MFMA_K; q++) {
                bf16x8 af[SM], bfr[SN];
#pragma unroll
                for (int i = 0; i < SM; i++) {
                    const unsigned arow = wm * (MPF_BM / MPF_WM) + i * MFMA_M + frow;
                    __builtin_memcpy(&af[i],
                                     &GMPF_ASM(buf)[arow * MPF_STRIDE +
                                                    MPF_XORSWZ(arow, mfma_frag_k(lane, q * MFMA_K))],
                                     16);
                }
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned brow =
                        (GLU ? j * (MPF_BN / 2) + wn * MFMA_N
                             : wn * (MPF_BN / MPF_WN) + j * MFMA_N) + frow;
                    __builtin_memcpy(&bfr[j],
                                     &GMPF_BSM(buf)[brow * MPF_STRIDE +
                                                    MPF_XORSWZ(brow, mfma_frag_k(lane, q * MFMA_K))],
                                     16);
                }
                __builtin_amdgcn_s_setprio(1);
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int j = 0; j < SN; j++)
                        acc[i][j] = plow_mfma_bf16_32x32(af[i], bfr[j], acc[i][j]);
                __builtin_amdgcn_s_setprio(0);
            }

            if (kn < K) {
                if constexpr (DB == 1) __syncthreads(); /* every reader done with the one buffer */
                const unsigned cb = (DB == 2) ? (buf ^ 1u) : 0u;
                GMPF_COMMIT(cb)
            }
            __syncthreads();
            if constexpr (DB == 2) buf ^= 1u;
        }

/* Row `rr`'s hoisted metadata, read back out of the lane that loaded it.
 *
 * EXEC DISCIPLINE, load-bearing and the reason this is a macro pair rather than three
 * inline reads: `ds_bpermute_b32` honours EXEC on the READ side, so a lane sourcing from a
 * masked-off lane does NOT get that lane's value. Lane L wants row `rr(L)`, held by lane
 * `rr(L)`, whose OWN activity is decided by row `rr(rr(L))` -- a different row. So neither
 * the `nn < N` tail guard (lane-varying, via mfma_acc_n) nor the PLOW_EXPERT_UNUSED pad-row
 * test may be live when these issue: all three fetches are emitted BEFORE either test, at
 * the full wave EXEC the epilogue is entered with, and the guards are then applied as a
 * MASK. The intrinsic is `convergent`, so LLVM may not sink it back into the divergent
 * region. Verified in the ISA (perf-data/plow-gfx942/glm52-dropped-items.md). */
#if PLOW_MOE_PF_EPI_SIB
#define GMPF_ROW_BP(v, rr) __builtin_amdgcn_ds_bpermute((int)((rr) << 2), (v))
#define GMPF_EPI_NGUARD(nn) const bool epi_nok = ((nn) < N)
#define GMPF_ROWMETA(rr, pidx, gv)                                                             \
    const unsigned pidx = (unsigned)GMPF_ROW_BP((int)epi_pidx, (rr));                          \
    const float gv = __builtin_bit_cast(float, GMPF_ROW_BP(epi_gate, (rr)));                   \
    const float epi_av = __builtin_bit_cast(float, GMPF_ROW_BP(epi_asc, (rr)));                \
    if (!epi_nok || pidx == PLOW_EXPERT_UNUSED) continue /* n tail or pad row */
/* The third fetch has to ride the SAME pre-EXEC window, so it is read unconditionally here
 * and consumed only on the W8A8 arm. A bpermute of a register that arm never loaded is
 * harmless; a bpermute issued under the pad-row mask would not be. */
#define GMPF_ROW_ASC(rr) epi_av
#else
#define GMPF_EPI_NGUARD(nn) if ((nn) >= N) continue
#define GMPF_ROWMETA(rr, pidx, gv)                                                             \
    const unsigned pidx = row_partidx[rowbase + (rr)];                                         \
    if (pidx == PLOW_EXPERT_UNUSED) continue; /* pad row */                                    \
    const float gv = row_gate[rowbase + (rr)]
#define GMPF_ROW_ASC(rr) ascale[rowbase + (rr)]
#endif

        /* ---- epilogue */
        const unsigned nn_lane =
            n0 + (GLU ? wn * MFMA_N : wn * (MPF_BN / MPF_WN)) + mfma_acc_n(lane);
        if constexpr (GLU) {
            /* acc[i][0] is gate and acc[i][1] is up FOR THE SAME ELEMENT, in the same lane. */
            bf16* const fu = (bf16*)Cout;
            if (nn_lane < N) {
#pragma unroll
                for (int i = 0; i < SM; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        float g = acc[i][0][el], u = acc[i][1][el];
                        if constexpr (W8A8) {
                            const unsigned tok = row_token[rowbase + rr];
                            if (tok == PLOW_EXPERT_UNUSED) {
                                /* pad row: write 0, matching the bf16 arm's zero-filled A */
                                st_act1(&fu[(size_t)(rowbase + rr) * N + nn_lane], f2bf(0.0f));
                                continue;
                            }
                            const float as = ascale[tok];
                            g *= as * wsc[nn_lane];
                            u *= as * wsc[N + nn_lane];
                        }
                        const float a = (act == PLOW_MOE_ACT_SILU) ? (g / (1.0f + expf(-g)))
                                                                   : gmoe_gelu_tanh(g);
                        st_act1(&fu[(size_t)(rowbase + rr) * N + nn_lane], f2bf(a * u));
                    }
            }
        } else {
            float* const part = (float*)Cout;
#pragma unroll
            for (int i = 0; i < SM; i++)
#pragma unroll
                for (int j = 0; j < SN; j++) {
                    const unsigned nn = n0 + wn * (MPF_BN / MPF_WN) + j * MFMA_N + mfma_acc_n(lane);
                    GMPF_EPI_NGUARD(nn);
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * (MPF_BM / MPF_WM) + i * MFMA_M + mfma_acc_m(lane, el);
                        GMPF_ROWMETA(rr, pidx, gv);
                        float y = gv * acc[i][j][el];
                        if constexpr (W8A8) y *= GMPF_ROW_ASC(rr) * wsc[nn];
                        part[(size_t)pidx * N + nn] = y;
                    }
                }
        }
        __syncthreads();
    }
#undef GMPF_FETCH_A
#undef GMPF_FETCH_B
#undef GMPF_COMMIT
#undef GMPF_ASM
#undef GMPF_BSM
#undef GMPF_EPI_NGUARD
#undef GMPF_ROWMETA
#undef GMPF_ROW_ASC
#if PLOW_MOE_PF_EPI_SIB
#undef GMPF_ROW_BP
#endif
}

/* --- op 75: PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF ----------------------------------------------- */
__device__ void d_moe_group_glu_gemma_pf(bf16* fu, const bf16* xn2,
                                         const unsigned long long* ewt, const int* meta,
                                         const unsigned* row_token, unsigned I_moe, unsigned H,
                                         unsigned n_exp, unsigned act, unsigned slice,
                                         unsigned nblk, bf16* lds) {
    d_moe_group_gemma_pf_t<true, false>(fu, xn2, nullptr, ewt, nullptr, meta, row_token, nullptr,
                                        nullptr, I_moe, H, n_exp, act, slice, nblk, lds);
}

/* --- op 76: PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF ---------------------------------------------- */
__device__ void d_moe_group_down_gemma_pf(float* part, const bf16* fu,
                                          const unsigned long long* ewt, const int* meta,
                                          const unsigned* row_partidx, const float* row_gate,
                                          unsigned H, unsigned I_moe, unsigned n_exp,
                                          unsigned slice, unsigned nblk, bf16* lds) {
    d_moe_group_gemma_pf_t<false, false>(part, fu, nullptr, ewt, nullptr, meta, nullptr,
                                         row_partidx, row_gate, H, I_moe, n_exp, 0, slice, nblk,
                                         lds);
}

/* --- op 81: PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8 ------------------------------------------ */
__device__ void d_moe_group_glu_gemma_pf_w8a8(bf16* fu, const unsigned char* xq8,
                                              const float* ascale,
                                              const unsigned long long* ewt,
                                              const unsigned long long* est, const int* meta,
                                              const unsigned* row_token, unsigned I_moe,
                                              unsigned H, unsigned n_exp, unsigned act,
                                              unsigned slice, unsigned nblk, bf16* lds) {
    d_moe_group_gemma_pf_t<true, true>(fu, xq8, ascale, ewt, est, meta, row_token, nullptr,
                                       nullptr, I_moe, H, n_exp, act, slice, nblk, lds);
}

/* --- op 82: PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8 ----------------------------------------- */
__device__ void d_moe_group_down_gemma_pf_w8a8(float* part, const unsigned char* fu8,
                                               const float* fscale,
                                               const unsigned long long* ewt,
                                               const unsigned long long* est, const int* meta,
                                               const unsigned* row_partidx, const float* row_gate,
                                               unsigned H, unsigned I_moe, unsigned n_exp,
                                               unsigned slice, unsigned nblk, bf16* lds) {
    d_moe_group_gemma_pf_t<false, true>(part, fu8, fscale, ewt, est, meta, nullptr, row_partidx,
                                        row_gate, H, I_moe, n_exp, 0, slice, nblk, lds);
}
#endif /* PLOW_MOE_GEMMA_PF */

#endif /* PLOW_OP_MOE_H */
