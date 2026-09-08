/* kx experiment: kda_gate — what does the KDA unbounded forget gate cost, and how far is it from
 * the mathematics?                                 [F6, plans/gpu-kernel-static-audit-20260908.md]
 *
 * THE QUESTION. The `PLOW_KDA_GATE_SOFTPLUS` branch computes
 *
 *      g[t,h,d] = -exp(A_log[h]) * softplus(g_raw + dt_bias)      (log2 units, then prefix-summed)
 *
 * and the result multiplies a PERSISTENT recurrent state once per token. `softplus` was
 * `__logf(1.0f + __expf(x))`: for x <= -16.6355 the `1.0f +` absorbs and the expression returns
 * EXACTLY ZERO, so the derived per-step decay is exactly 1 and the gate stops forgetting. That is
 * not a rounding difference in one output, it is a missing operation repeated for the length of
 * the sequence.
 *
 * Three formulations, three code objects, one source:
 *   fla     `__logf(1 + __expf(x))`         -- what [fla] writes; the shipped body until F6
 *   branch  series `e - e^2/2` below -8     -- what op_kda.h now compiles
 *   log1p   `log1pf(expf(x))`               -- op_qwen_gdn.h's `qwen_softplus`, verbatim; correct
 *                                              in the tail and the expensive way to get there
 *
 * The REFERENCE is a device f64 evaluation of the same recurrence, so the reported max error is
 * distance from the mathematics rather than distance from the incumbent. `tol_max` is set wide
 * enough to ADMIT the fla arm on purpose: the result of this experiment is the error column, not
 * a pass/fail, and an arm that is refused prints no number.
 *
 * WHAT THE OUTPUT IS. Not `softplus(x)` -- the absorption is invisible in a single value's
 * absolute error (it is 4e-8 at x = -17). The output is the chunk gate's own accumulator, the
 * cumulative log2-domain decay over `reps` steps, which is what `d_kda_chunk_gate_prefix_bt64`
 * actually writes and what the recurrence actually consumes. Over 8192 steps at the checkpoint's
 * maximum A = exp(A_log) = 11.776 the fla arm is 5.8e-3 away from the mathematics at x = -17 and
 * the branch arm is 1e-9 away; over 2^20 steps (K3's max_position_embeddings) the same channel
 * retains 0.5998 of its state mathematically and 1.0000 under fla.
 *
 * THE SWEEP is `s = g_raw + dt_bias` over [-30, +4] -- the K3-measured `dt_bias in [-7.894,
 * 0.179]` (docs/kimi-k3-kda.md 3.1) puts the interesting region there -- crossed with
 * `a = exp(A_log)` cycled over the measured [0.471, 11.776]. No checkpoint on this box ships the
 * softplus branch (K3 sets `gate_lower_bound` and takes the sigmoid branch), so the sweep is the
 * contract, not a sample of one.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_elementwise.h" /* act_silu, used by op_kda's conv (same order as interp.hip) */
#include "op_kda.h"        /* kda_softplus, variant-selected */
#include "op_qwen_gdn.h"   /* qwen_softplus: log1pf(expf(x)), the shipped precise form */

#define KX_KDA_A_LO 0.471f
#define KX_KDA_A_HI 11.776f
#define KX_KDA_S_LO (-30.0f)
#define KX_KDA_S_HI 4.0f

__device__ __forceinline__ float kx_kda_s(unsigned i, unsigned n) {
    return KX_KDA_S_LO + (KX_KDA_S_HI - KX_KDA_S_LO) * ((float)i / (float)(n - 1u));
}
__device__ __forceinline__ float kx_kda_a(unsigned i) {
    return KX_KDA_A_LO + (KX_KDA_A_HI - KX_KDA_A_LO) * (float)(i % 97u) / 96.0f;
}

/* `d_kda_chunk_gate_prefix_bt64`'s own accumulator, with the token axis as a plain loop:
 * `gate = -a * softplus(s); acc += gate * RCP_LN2`, and the OUTPUT is `exp2(acc)` -- the fraction
 * of the state this channel still holds after `d1` steps.
 *
 * exp2 OF the accumulator, rather than the accumulator, is deliberate. It puts every arm on one
 * bounded scale [0,1] where the harness's max-error column IS the quantity F6 claims: how much
 * state was retained that should have decayed. On the raw accumulator the max error sits at the
 * top of the sweep (one ulp of a large log2 sum) and hides the tail entirely.
 *
 * THE RECURRENCE LENGTH IS `d1`, NOT `kx_nrep`. The harness runs the correctness launch at nrep=1
 * by design, so a drift carried by the rep loop would never be compared; `kx_nrep` stays the
 * timing repeat it is in every other experiment.
 *
 * Each step perturbs `s` from the activation arena, because in the real op every token carries its
 * own `g_raw`. A loop whose summand is loop-INVARIANT gets collapsed to one multiply (measured on
 * the first cut of this file: 8192 "steps" ran FASTER than 64), and the timing column is then
 * fiction. The arena values are ~1e-2 wide, negligible against the sweep's 34-wide span. */
template <float (*SP)(float)>
__device__ __forceinline__ void kx_kda_gate_body(KX_SIG) {
    constexpr float RCP_LN2 = 1.4426950408889634f;
    (void)kx_w;
    (void)kx_aux;
    (void)kx_d2;
    const unsigned n = kx_d0, steps = kx_d1;
    const bf16* const g = (const bf16*)kx_x;
    float* const o = (float*)kx_out;
    for (unsigned e = 0; e < kx_nrep; e++)
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += kx_nblk * blockDim.x) {
            const float s0 = kx_kda_s(i, n);
            const float a = kx_kda_a(i);
            float acc = 0.0f;
            for (unsigned t = 0; t < steps; t++)
                acc += (-a * SP(s0 + bf2f(g[(i + t) & 0xFFFFFFu]))) * RCP_LN2;
            o[i] = exp2f(acc);
        }
}

/* The reference: the same recurrence in f64 over the same arena samples. `log1p(exp(x))` in double
 * does not absorb anywhere in the swept range (its threshold is x < -36.7), so this is the
 * mathematics to fp32. */
KX_BODY(b_f64) {
    constexpr double RCP_LN2 = 1.4426950408889634;
    (void)kx_w;
    (void)kx_aux;
    (void)kx_d2;
    const unsigned n = kx_d0, steps = kx_d1;
    const bf16* const g = (const bf16*)kx_x;
    float* const o = (float*)kx_out;
    for (unsigned e = 0; e < kx_nrep; e++)
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += kx_nblk * blockDim.x) {
            const double s0 = (double)kx_kda_s(i, n);
            const double a = (double)kx_kda_a(i);
            double acc = 0.0;
            for (unsigned t = 0; t < steps; t++)
                acc += -a * log1p(exp(s0 + (double)bf2f(g[(i + t) & 0xFFFFFFu]))) * RCP_LN2;
            o[i] = (float)exp2(acc);
        }
}
KX_ARM_OF(k_gate_f64, b_f64)

KX_BODY(b_kda) { kx_kda_gate_body<kda_softplus>(KX_FWD); }
KX_ARM_OF(k_gate, b_kda)
KX_ARM_OF(k_gate_aa, b_kda)

KX_BODY(b_log1p) { kx_kda_gate_body<qwen_softplus>(KX_FWD); }
KX_ARM_OF(k_gate_log1p, b_log1p)

#else /* ------------------------------------------------------- host descriptor */

/* The variant axis is the FORMULATION. `k_gate` appears twice with different variants: same
 * symbol, two code objects, and the -D is the whole difference. */
static const kx_variant VAR[] = {
    {"fla", "-DPLOW_KDA_SOFTPLUS_FLA_COMPAT=1"},
    {"branch", "-DPLOW_KDA_SOFTPLUS_FLA_COMPAT=0"},
};

static const kx_arm ARM[] = {
    {"k_gate", 0, KX_BASE, "[fla]'s __logf(1 + __expf(x)) -- the body before F6"},
    {"k_gate_aa", 0, KX_AA, "byte-identical device code to fla:k_gate"},
    {"k_gate_f64", 0, KX_REF | KX_SLOWOK, "f64 log1p(exp(x)): the mathematics, not a candidate"},
    {"k_gate", 1, 0, "F6: series e - e^2/2 below x = -8, [fla]'s expression above it"},
    {"k_gate_log1p", 0, 0, "op_qwen_gdn.h's qwen_softplus = log1pf(expf(x)), verbatim"},
};

/* d1 IS THE RECURRENCE LENGTH -- 64 is one BT64 chunk, 8192 a long context. d0 is the sweep width
 * and the compared element count; `reps` is only the timing repeat. grid 224 is a K3 gate packet's
 * own workgroup count. */
static const kx_shape SHP[] = {
    {"chunk64", 114688, 64, 0, 224, 1, 0.0, 8, 114688},
    {"seq8192", 114688, 8192, 0, 224, 1, 0.0, 1, 114688},
};

static const kx_exp EXP = {
    "kda_gate",
    "does the KDA softplus gate lose the tail, and what does fixing it cost?",
    "runtime/bench/amd/kx/exp_kda_gate.hip",
    /* scripts/build_gfx942.sh's K3 decode row. */
    "-DPLOW_K3=1 -DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8 -DPLOW_GEMV_MM=1 -DGV_UNROLL=14",
    "gfx942",
    512,
    KX_DT_F32,
    VAR,
    2,
    ARM,
    5,
    SHP,
    2,
    0.98,
    1.02,
    /* Wide ON PURPOSE. The fla arm's error against the f64 reference is the RESULT; refusing it
     * would suppress the only number this experiment exists to print. */
    1.0,
    "fla loses the whole tail below s = -16.6355 (gate exactly 0 where the mathematics says it is "
    "not); branch and log1p do not. Instruction cost: branch is +14 in d_kda_gate over fla; "
    "log1pf(expf) is the 159-vs-43 whole-probe row of the audit's cost table.",
};

extern "C" const kx_exp* kx_experiment(void) { return &EXP; }

#endif
