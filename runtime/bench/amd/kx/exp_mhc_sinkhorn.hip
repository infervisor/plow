/* kx experiment: mhc_sinkhorn — what does the mHC Sinkhorn loop's 640 divides per token cost?
 *
 * THE QUESTION. `d_hyperconn_pre` (op 121) normalizes a 4x4 combine matrix to doubly stochastic
 * by alternating row and column normalizes, `sinkhorn_repeat = 20` times, ON LANE 0 ALONE. Each
 * normalize divides every element of a row (or column) by the SAME denominator, and an IEEE f32
 * divide is not strength-reducible to a reciprocal-and-multiply by the compiler. The static
 * gfx942 ISA of the shipped body confirms it: 73 `v_div_fixup_f32` in the probe, i.e. every
 * source divide expanded in full, none shared. At n=4 that is
 *
 *     16 (initial softmax) + 16 (first column pass) + 19 * 32 (the repeat loop) = 640
 *
 * full divide expansions per token per call, serialized on one lane while the other 511 threads
 * wait at the `__syncthreads()` that follows.
 *
 * WHAT THE ARMS CHANGE, and nothing else. `PLOW_HC_SINKHORN_RCP` (op_hyperconn.h, default 0)
 * takes ONE reciprocal per row/column instead of one divide per element:
 *
 *     0  shipped — one IEEE divide per element (640)
 *     1  one IEEE divide for the reciprocal, then multiplies (160 divides)
 *     2  as 1, with `v_rcp_f32` (~1 ULP) for the reciprocal (0 divides)
 *
 * Both CHANGE ROUNDING, so neither is exact and neither is a default. The numerical price is not
 * measured here — it is measured against the f64 mHC oracle in
 * `runtime/tests/hyperconn_sinkhorn_gfx942_test.hip`, which is where a rounding claim belongs.
 * This file answers only "how much of the op is the divide?".
 *
 * WHAT THE INPUT IS, and why that is honest for THIS question. The arena fill is a fixed
 * small-magnitude bf16 pattern, so `mixes` read as f32 is uniformly tiny and the Sinkhorn's fixed
 * point is nearly its starting point. That would be a bad input for an ACCURACY claim and it is
 * fine for a TIMING one: `v_div_scale_f32`/`v_div_fmas_f32`/`v_div_fixup_f32` are fixed-latency
 * on gfx942 and the loop is unconditional — no data-dependent branch, no early exit, the same
 * 640 expansions on every input. The op's own `expf` count (24) is likewise fixed.
 *
 * WHAT IT DOES NOT ANSWER. There is NO end-to-end number behind this experiment, because no blob
 * on this box emits `HyperConnPre`: GLM-5.3 `tp4-long` is MLA + MoE and contains zero
 * hyperconnection ops (`plowrt disasm`, 1865 instructions, 33 opcodes, none of them mHC). The
 * body IS compiled into every interpreter object — the dispatch is unconditional — so the arms
 * move the object's code size and register allocation and nothing a served token executes. A
 * body win here is a body win, and the object contract in `scripts/build_gfx942.sh` is where its
 * cost to the megakernel is read. Do not promote either arm on this file alone.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_hyperconn.h"

/* The shipped body, called exactly as the interpreter calls it (interp.hip's
 * PLOW_DOP_HYPER_CONN_PRE arm): n = 4, sinkhorn_repeat = 20, rms_eps/hc_eps from the emitter's
 * own `d.f[0] = c.eps` / `d.f[1] = 1e-6` (crates/devgen/src/mla.rs).
 *
 * OUTPUT LAYOUT. comb_mix FIRST, because it is the Sinkhorn's own result and the only output the
 * arms can move; post_mix follows it so the correctness gate covers both f32 outputs in one span.
 * layer_input is bf16 and sits past them, uncompared — reduction 2 is untouched by every arm. */
KX_BODY(b_hc) {
    const unsigned T = kx_d0, hidden = kx_d1;
    (void)kx_d2;
    __shared__ float part[PLOW_WAVES];

    float* const comb_mix = (float*)kx_out;
    float* const post_mix = comb_mix + (size_t)T * 16u;
    bf16* const layer_input = (bf16*)(post_mix + (size_t)T * 4u);
    const float* const mixes = (const float*)kx_x;
    const float* const hc_scale = (const float*)kx_aux;
    const float* const hc_base = hc_scale + 8u;

    for (unsigned e = 0; e < kx_nrep; e++) {
        const bf16* const residual = (const bf16*)kx_w + (size_t)e * T * 4u * hidden;
        d_hyperconn_pre(post_mix, comb_mix, layer_input, mixes, residual, hc_scale, hc_base, T, 4u,
                        hidden, 20u, 1e-5f, 1e-6f, blockIdx.x, kx_nblk, part);
    }
}
KX_ARM_OF(k_hc, b_hc)
KX_ARM_OF(k_hc_aa, b_hc)

#else /* ------------------------------------------------------- host descriptor */

/* The variant axis IS the question: one -D, three objects, one source. */
static const kx_variant VAR[] = {
    {"div", "-DPLOW_HC_SINKHORN_RCP=0"},
    {"rcp", "-DPLOW_HC_SINKHORN_RCP=1"},
    {"nrcp", "-DPLOW_HC_SINKHORN_RCP=2"},
};

static const kx_arm ARM[] = {
    {"k_hc", 0, KX_BASE | KX_REF, "the shipped body: one IEEE divide per matrix element"},
    {"k_hc_aa", 0, KX_AA, "byte-identical device code to k_hc"},
    {"k_hc", 1, 0, "one IEEE divide per row/column, then multiplies (640 -> 160 divides)"},
    {"k_hc", 2, 0, "one v_rcp_f32 per row/column, then multiplies (0 divides)"},
};

/* grid is the EMITTER'S OWN count, `rows.min(b.n_cu())` at `emit_glm53_hc_pre`
 * (crates/devgen/src/mla.rs) — one workgroup per token, capped at 304 CUs. hidden = 6144 is
 * GLM-5.3's, from `build-glm53/tp4-long/model.pkt`.
 *
 * `inst` is 0 on every shape: a per-token total needs an instances-per-token count from a real
 * packet, and no packet on this box emits this op. Printing one would invent it. */
static const kx_shape SHP[] = {
    /* name, d0=T, d1=hidden, d2, grid, inst, slab_bytes, reps, out_elems */
    {"decode", 1, 6144, 0, 1, 0, 1.0 * 4 * 6144 * 2, 0, 1 * 20},
    {"pf304", 304, 6144, 0, 304, 0, 304.0 * 4 * 6144 * 2, 0, 304 * 20},
    {"pf2048", 2048, 6144, 0, 304, 0, 2048.0 * 4 * 6144 * 2, 0, 2048 * 20},
};

static const kx_exp EXP = {
    "mhc_sinkhorn",
    "how much of op 121 is the Sinkhorn's 640 serial IEEE divides?",
    "runtime/bench/amd/kx/exp_mhc_sinkhorn.hip",
    /* The decode object's own defines. This body reads only PLOW_THREADS/PLOW_WAVES out of them;
     * the dispatch that reaches it is unconditional in every object. */
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8",
    "gfx942", 512, KX_DT_F32,
    VAR, 3, ARM, 4, SHP, 3,
    0.98, 1.02, /* A/A tolerance */
    1e-6,       /* comb_mix entries are ~0.25 and the arms move them by ULPs, not by structure */
    "static gfx942 ISA of the whole probe kernel: 1932 instructions and 73 divide expansions at "
    "RCP=0, 1396/25 at RCP=1, 1300/9 at RCP=2",
};
extern "C" const kx_exp* kx_experiment(void) { return &EXP; }
#endif
