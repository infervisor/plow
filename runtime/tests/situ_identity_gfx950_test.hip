/* situ_identity_gfx950_test.hip — EXHAUSTIVE numeric gate for the ACTIVATION primitives that
 * the expensive-instruction sweep rewrote.                                  [KIMI-K3] [GLM-5.2]
 *
 * WHY THIS FILE EXISTS, AND WHY IT IS NOT A BIT-IDENTITY TEST.
 * `fp4_quant_identity_gfx950_test.hip` next door proves the MXFP4 quantizer primitives are
 * BIT-IDENTICAL over all 2^32 floats, and that is the right bar there: those primitives decide
 * scale BYTES that another test compares exactly, so any difference is a difference.
 *
 * The activations are not that. `v_exp_f32` is ~2 ULP where OCML's `expf` is correctly rounded,
 * and `v_rcp_f32` is ~1 ULP where an IEEE division is half. Substituting them CANNOT be
 * bit-identical and claiming otherwise would be a lie. So this test measures the thing that
 * actually decides admissibility: **the activation's output is rounded to BF16 (8 mantissa
 * bits) before anything downstream reads it**, so the question is how many of the 2^32 inputs
 * produce a DIFFERENT BF16, by how many bf16 ULP, and at what magnitudes. A change that moves
 * no bf16 at all, or moves it by one ULP only where the value is 30 binades below anything the
 * model can represent, is neutral in the only sense the runtime has.
 *
 * It carries FROZEN copies of every primitive exactly as it stood before the rewrite. Those
 * copies are the definition of "the numerics the shipped objects were measured with". If a
 * future edit needs them to change, that is the conversation, and this is where it happens.
 *
 * WHAT IS SWEPT (every one of the 4294967296 float bit patterns — normals, subnormals, both
 * zeros, every NaN payload, both infinities). The table has TWO halves and they are held to
 * DIFFERENT standards:
 *
 * CHANGED — a tolerance applies, stated below:
 *   k3_sigmoid          1/(1+__expf(-x))        -> rcp(1+__expf(-x))
 *   k3_situ_gate        beta*tanhf(g/beta)*sig  -> beta*fast_tanhf(g/beta)*sig          beta=4
 *   k3_situ_up          lb*tanhf(u/lb)          -> lb*fast_tanhf(u/lb)                  lb=25
 *   fast_tanhf          tanhf(y)                -> 1-2*rcp(__expf(2y)+1)   (the new primitive)
 *
 * UNCHANGED — must be EXACTLY ZERO differences, and the test fails on one:
 *   act_silu            x/(1+__expf(-x))        GLM-5.2's dense-FFN SwiGLU
 *   moe_act SILU        x/(1+expf(-x))          GLM-5.2's ROUTED-expert SwiGLU
 *   act_gelu_tanh       0.5x(1+tanhf(c))        Gemma's GLU
 *
 * The second half is not decoration. All three were rewritten to the fast spelling during this
 * sweep, all three were correct, and all three were REVERTED because they measured slower or
 * unmeasurable (see the notes in op_moe.h's `moe_act` and op_elementwise.h). Sweeping them
 * against their own frozen bodies is what makes "GLM-5.2's numerics did not move" a checked
 * statement rather than a claim, and it is what will catch the next person who re-derives the
 * substitution without re-doing the measurement.
 *
 * WHAT IT CANNOT SWEEP, AND THE ARGUMENT INSTEAD. The shipped `situ` is the PRODUCT
 * `k3_situ_gate(g)*k3_situ_up(u)`, a two-argument function with 2^64 inputs. Each factor is
 * swept separately and the product's relative error is bounded by the sum of the factors'
 * (to first order), which the report prints; there is no cancellation between them because both
 * factors are multiplied, never subtracted.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/situ_identity_gfx950_test.hip -o /tmp/situ_ident
 *   sg render -c 'ROCR_VISIBLE_DEVICES=<n> /tmp/situ_ident'
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "amd_common.h"
#include "op_elementwise.h"
#define PLOW_MOE_PREFILL 1
#include "op_moe.h"

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

/* K3's two situ immediates (modeling_kimi_linear.py: activation_situ_beta / _linear_beta). */
#define SITU_BETA  4.0f
#define SITU_LB   25.0f

/* ---- FROZEN REFERENCES. Verbatim from the tree before the expensive-instruction sweep. ----
 * Do not "tidy" these. Their spelling IS the claim. */
__device__ __forceinline__ float ref_k3_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float ref_situ_gate(float g, float beta) {
    return beta * tanhf(g / beta) * ref_k3_sigmoid(g);
}
__device__ __forceinline__ float ref_situ_up(float u, float lb) {
    return lb > 0.0f ? lb * tanhf(u / lb) : u;
}
__device__ __forceinline__ float ref_act_silu(float x) { return x / (1.0f + __expf(-x)); }
/* op_moe.h's silu was NOT op_elementwise.h's: it used OCML's `expf`, not `__expf`. That
 * divergence between GLM-5.2's dense GLU and its routed experts is what the sweep converged. */
__device__ __forceinline__ float ref_moe_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float ref_gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}
__device__ __forceinline__ float ref_tanh(float y) { return tanhf(y); }

/* ---- what is measured, and why it is stratified by output magnitude -----------------------
 *
 * A raw "how many of the 2^32 inputs move a bf16" is a MISLEADING headline for these functions
 * and the first version of this test printed one: it came out at 43% for `fast_tanhf`, which
 * sounds catastrophic and means nothing. tanh(y) -> y as y -> 0, and roughly 40% of all float
 * bit patterns have |y| below 1e-7, where the exponential-based form's fixed ~1e-7 ABSOLUTE
 * error is larger than the value itself. Those inputs move a bf16 from 1e-20 to 0. Nothing in a
 * neural network can observe that, and counting them swamps the inputs that matter.
 *
 * So every quantity is reported twice: over the WHOLE domain, and restricted to |ref| >= 1e-2 —
 * the region where the activation is actually carrying signal into the down projection. The
 * PASS criterion lives on the restricted numbers plus a global ABSOLUTE bound, because these
 * values are consumed by a dot product: a term wrong by E perturbs a sum by at most E
 * regardless of how small the term is, and the sum's own bf16 rounding is 2^-9 of its
 * magnitude. E below ~1e-5 against an O(1) sum is two orders under that.
 *
 * BF16 ULP IS MEASURED ON THE MONOTONE ORDER, not on the raw bits. `(int)a - (int)b` over
 * sign-magnitude bf16 reports 32768 for a +0/-0 pair and ~46000 for a sign flip near zero,
 * which is how the first version of this test produced "46001 ulp" for a 7e-7 absolute error.
 * Mapping to the total order first makes the distance mean what the name says, and makes +0 and
 * -0 distance ZERO, which is correct: they are numerically equal. */
struct Acc {
    unsigned long long n_diff;     /* inputs whose BF16 output moved                          */
    unsigned long long n_diff_big; /* ... of those, ones with |ref| >= BIG                     */
    unsigned long long n_big;      /* inputs with |ref| >= BIG (the denominator for n_diff_big)*/
    unsigned max_ulp;              /* worst bf16 ulp distance, whole domain                    */
    unsigned max_ulp_big;          /* worst bf16 ulp distance where |ref| >= BIG               */
    unsigned max_abs;              /* bits of max |new-ref|, finite ref, whole domain          */
    unsigned max_rel_big;          /* bits of max |new-ref|/|ref| where |ref| >= BIG           */
    unsigned absx_lo, absx_hi;     /* bits of the |x| range over the BF16-differing inputs     */
    unsigned n_class;              /* inputs where finite/inf/NaN CLASS disagrees              */
    unsigned worst_x;              /* bits of the input attaining max_ulp_big                  */
};
#define NFN 7
#define BIG 1.0e-2f

/* bf16 sign-magnitude bits -> monotone total order (and +0 == -0). */
__device__ __forceinline__ unsigned bford(bf16 b) {
    return (b & 0x8000u) ? (0x8000u - (unsigned)(b & 0x7fffu)) : (0x8000u + (unsigned)(b & 0x7fffu));
}

__device__ __forceinline__ void note(Acc* a, float x, float r, float v) {
    /* CLASS first: a finite-vs-NaN or +inf-vs-finite swap is a different kind of event from a
     * last-bit move and must never be averaged into an error norm. */
    const bool rn = isnan(r), vn = isnan(v);
    const bool ri = isinf(r), vi = isinf(v);
    if (rn != vn || ri != vi || (ri && vi && ((r > 0.0f) != (v > 0.0f)))) {
        atomicAdd(&a->n_class, 1u);
        return;
    }
    if (rn) return; /* both NaN: payload is not a contract anywhere in this tree */
    const bool big = !ri && fabsf(r) >= BIG;
    if (big) atomicAdd(&a->n_big, 1ull);
    if (!ri) {
        const float ae = fabsf(v - r);
        atomicMax(&a->max_abs, __builtin_bit_cast(unsigned, ae));
        if (big) atomicMax(&a->max_rel_big, __builtin_bit_cast(unsigned, ae / fabsf(r)));
    }
    /* THE GATE: does the value the runtime actually stores move, and by how much? */
    const unsigned orr = bford(f2bf(r)), orv = bford(f2bf(v));
    if (orr != orv) {
        const unsigned d = orr > orv ? orr - orv : orv - orr;
        atomicAdd(&a->n_diff, 1ull);
        atomicMax(&a->max_ulp, d);
        const unsigned ax = __builtin_bit_cast(unsigned, fabsf(x));
        atomicMin(&a->absx_lo, ax);
        atomicMax(&a->absx_hi, ax);
        if (big) {
            atomicAdd(&a->n_diff_big, 1ull);
            if (atomicMax(&a->max_ulp_big, d) < d)
                a->worst_x = __builtin_bit_cast(unsigned, x);
        }
    }
}

__global__ __launch_bounds__(256) void sweep(Acc* a, unsigned long long lo, unsigned long long n) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const float x = __builtin_bit_cast(float, (unsigned)(lo + i));
        note(a + 0, x, ref_k3_sigmoid(x), k3_sigmoid(x));
        note(a + 1, x, ref_situ_gate(x, SITU_BETA), k3_situ_gate(x, SITU_BETA));
        note(a + 2, x, ref_situ_up(x, SITU_LB), k3_situ_up(x, SITU_LB));
        note(a + 3, x, ref_act_silu(x), act_silu(x));
        note(a + 4, x, ref_moe_silu(x), moe_act(x, PLOW_MOE_ACT_SILU));
        note(a + 5, x, ref_gelu_tanh(x), act_gelu_tanh(x));
        note(a + 6, x, ref_tanh(x), fast_tanhf(x));
    }
}

/* ---- expert-selection identity for the ROUTER ---------------------------------------------
 * The router's sigmoid is the one place where an error invisible in a bf16 activation is NOT
 * automatically harmless: its output feeds a TOP-K ARGMAX, so a last-bit move can flip WHICH
 * expert is chosen — a discrete change in the token stream, not a rounding difference. The
 * sweep above cannot see that, because the question is about a 896-vector, not a scalar.
 *
 * So this kernel builds random logit vectors at K3's and GLM's widths, ranks them with the
 * frozen sigmoid and with the live one using the SAME packed-key rule the router uses
 * (ordered(score+bias)<<20 | (n_exp-1-id)), and compares the winning SETS and ORDER. It
 * deliberately includes NEAR-TIE vectors — logits drawn from a lattice so that many experts
 * share a score to within a few ULP — because a uniformly random vector almost never exercises
 * the case the argmax is sensitive to.
 *
 * This runs whether or not the router was changed. If it ever IS changed, this is the gate. */
__device__ __forceinline__ unsigned long long rkey(float sc, unsigned e, unsigned n_exp) {
    unsigned sb = __builtin_bit_cast(unsigned, sc);
    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
    return ((unsigned long long)sb << 20) | (unsigned long long)((n_exp - 1u - e) & 0xFFFFFu);
}
__device__ __forceinline__ unsigned xs(unsigned& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
/* One expert's (bf16 logit, selection bias) from the vector seed, so the reference ranking and
 * the live one see BYTE-IDENTICAL inputs and the only difference on the table is the sigmoid.
 *
 * `mode` picks the BIAS, and the bias is the whole story. Three tie-generators built around the
 * LOGITS all reported zero movement before this one, and each of those zeros was the fixture's
 * fault rather than evidence:
 *
 *   - Experts sharing a logit exactly cannot reorder: both sigmoids move by the same amount and
 *     a constant shift is order-preserving.
 *   - Experts on a lattice of adjacent bf16 logits near 0.5 cannot reorder either. The logistic's
 *     slope is ~1/4 there, so one bf16 logit ulp is worth ~5e-4 of score, and no ~6e-8
 *     perturbation bridges that however fine the bias is.
 *   - Experts on SATURATED logits (10..14, where one bf16 logit ulp is worth only ~1e-8 of score)
 *     also reported zero — because that is exactly where the reciprocal is EASY. `1+exp(-12)` is
 *     within 2 ulp of 1.0, and v_rcp_f32 is exact there, so there is no perturbation left to
 *     flip anything.
 *
 * The structural fact underneath all three: `1.0f/D` and `v_rcp_f32(D)` are both MONOTONE in D,
 * so on their own they induce the SAME order and can differ only where rcp collapses two
 * distinct quotients onto one f32. Monotonicity is most of why this substitution is as safe as
 * it is. The e_score_correction_bias is what breaks the argument, because it is added AFTER the
 * sigmoid and is per-expert, so the shared-monotone reasoning no longer applies. Hence the modes
 * below vary the bias and leave the logits realistically spread.
 *
 *   mode 0  bias on the 1/65536 grid GLM/DeepSeek-shaped biases actually live on
 *   mode 1  bias at FULL f32 granularity over the same range — many more near-ties
 *   mode 2  NO bias at all — the monotonicity control, which must come out at exactly zero
 */
__device__ __forceinline__ void gen(unsigned s, unsigned e, unsigned mode, float& lg, float& bias) {
    unsigned se = s ^ (e * 0x85ebca6bu);
    for (int w = 0; w < 3; w++) (void)xs(se);
    lg = bf2f(f2bf((float)((int)(se & 0xffffu) - 32768) * (1.0f / 4096.0f)));
    if (mode == 0) bias = (float)((int)((se >> 16) & 0xffu) - 128) * (1.0f / 65536.0f);
    else if (mode == 1) bias = ((float)(se >> 8) * (1.0f / 16777216.0f) - 0.5f) * (1.0f / 256.0f);
    else bias = 0.0f;
}

__global__ __launch_bounds__(64) void router_sel(unsigned* diff_set, unsigned* diff_order,
                                                 unsigned n_exp, unsigned topk, unsigned tie,
                                                 unsigned trials) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= trials) return;
    unsigned s = 0x9e3779b9u ^ (t * 2654435761u);
    for (int w = 0; w < 4; w++) (void)xs(s);
    /* Rank by exhaustive comparison against a rebuilt key each time — O(n_exp^2) but this is a
     * correctness fixture, not a kernel. Scores are regenerated from the seed so no storage is
     * needed beyond the top-k lists. */
    unsigned wr[16], wv[16];
    for (unsigned j = 0; j < topk; j++) { wr[j] = 0xffffffffu; wv[j] = 0xffffffffu; }
    for (unsigned e = 0; e < n_exp; e++) {
        float lg, bias;
        gen(s, e, tie, lg, bias);
        const unsigned long long rk = rkey(ref_k3_sigmoid(lg) + bias, e, n_exp);
        const unsigned long long vk = rkey(k3_sigmoid(lg) + bias, e, n_exp);
        unsigned rr = 0, vv = 0;
        for (unsigned f = 0; f < n_exp; f++) {
            float lf, bf_;
            gen(s, f, tie, lf, bf_);
            rr += (rkey(ref_k3_sigmoid(lf) + bf_, f, n_exp) > rk);
            vv += (rkey(k3_sigmoid(lf) + bf_, f, n_exp) > vk);
        }
        if (rr < topk) wr[rr] = e;
        if (vv < topk) wv[vv] = e;
    }
    bool order_ok = true, set_ok = true;
    for (unsigned j = 0; j < topk; j++) if (wr[j] != wv[j]) order_ok = false;
    for (unsigned j = 0; j < topk && set_ok; j++) {
        bool found = false;
        for (unsigned i = 0; i < topk; i++) if (wv[i] == wr[j]) found = true;
        if (!found) set_ok = false;
    }
    if (!set_ok) atomicAdd(diff_set, 1u);
    if (!order_ok) atomicAdd(diff_order, 1u);
}

static const char* NAME[NFN] = {"k3_sigmoid",  "k3_situ_gate(beta=4)", "k3_situ_up(lb=25)",
                                "act_silu",    "moe_act SILU",         "act_gelu_tanh",
                                "fast_tanhf"};
/* 1 = this primitive was NOT changed and must sweep to exactly zero differences. Indices match
 * the `note()` calls in `sweep` and the NAME table above. */
static const int FROZEN[NFN] = {0, 0, 0, 1, 1, 1, 0};

static float fb(unsigned b) { float f; memcpy(&f, &b, 4); return f; }

int main() {
    Acc* d;
    HC(hipMalloc(&d, sizeof(Acc) * NFN));
    Acc h[NFN];
    memset(h, 0, sizeof(h));
    for (int i = 0; i < NFN; i++) h[i].absx_lo = 0xffffffffu;
    HC(hipMemcpy(d, h, sizeof(h), hipMemcpyHostToDevice));

    const unsigned long long TOT = 1ull << 32, CH = 1ull << 26;
    printf("sweeping all 2^32 float bit patterns over %d primitives ...\n", NFN);
    for (unsigned long long c = 0; c < TOT / CH; c++) {
        hipLaunchKernelGGL(sweep, dim3(8192), dim3(256), 0, 0, d, c * CH, CH);
        HC(hipDeviceSynchronize());
    }
    HC(hipGetLastError());
    HC(hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost));

    int fail = 0;
    printf("\nWHOLE DOMAIN (all 2^32 inputs)            | SIGNAL REGION  |ref| >= %g\n", (double)BIG);
    printf("%-22s %11s %4s %10s | %10s %4s %10s\n", "primitive", "bf16 moved", "ulp", "max |abs|",
           "bf16 moved", "ulp", "max |rel|");
    for (int i = 0; i < NFN; i++) {
        printf("%-22s %11llu %4u %10.3g | %10llu %4u %10.3g %s\n", NAME[i],
               (unsigned long long)h[i].n_diff, h[i].max_ulp, (double)fb(h[i].max_abs),
               (unsigned long long)h[i].n_diff_big, h[i].max_ulp_big,
               (double)fb(h[i].max_rel_big), FROZEN[i] ? "  <FROZEN>" : "");
        /* FROZEN primitives are the ones this sweep left alone. Anything but a clean zero here
         * means a shipping model's numerics moved, which is never a tolerance question. */
        if (FROZEN[i] && (h[i].n_diff || h[i].max_abs || h[i].n_class)) {
            printf("   FAIL %s is FROZEN but differs from its reference: %llu bf16, %u class\n",
                   NAME[i], (unsigned long long)h[i].n_diff, h[i].n_class);
            fail = 1;
            continue;
        }
        /* A CLASS change (finite <-> NaN <-> inf) is never acceptable and is not a tolerance. */
        if (h[i].n_class) {
            printf("   FAIL %s: %u inputs changed finite/inf/NaN CLASS\n", NAME[i], h[i].n_class);
            fail = 1;
        }
        /* In the signal region a substitution must be a LAST-BIT event and nothing more. */
        if (h[i].max_ulp_big > 1u) {
            printf("   FAIL %s: bf16 moved %u ulp at |ref| >= %g (input %.9g)\n", NAME[i],
                   h[i].max_ulp_big, (double)BIG, (double)fb(h[i].worst_x));
            fail = 1;
        }
        /* NOT a relative-error criterion, deliberately. `max |rel|` above is printed because it
         * is informative, but it is NOT independent evidence: it is dominated by inputs sitting
         * exactly ON the |ref| >= 1e-2 boundary, where the fixed ~2.4e-7 absolute error is
         * mechanically 2.4e-5 relative. One bf16 ULP AT that magnitude is 1e-2 * 2^-8 = 3.9e-5,
         * so a 2.4e-5 relative error is BELOW one bf16 ulp and the `ulp` column already says so.
         * An earlier revision of this file failed on a 1e-6 relative bound and it was the bound
         * that was wrong, not the code. The ulp column is the criterion. */
        if (fb(h[i].max_rel_big) > 3.9e-3f) { /* one bf16 ulp, as a backstop on the backstop */
            printf("   FAIL %s: relative error %.3g at |ref| >= %g exceeds one bf16 ulp\n",
                   NAME[i], (double)fb(h[i].max_rel_big), (double)BIG);
            fail = 1;
        }
        /* Outside the signal region the only admissible claim is an ABSOLUTE bound, because the
         * consumer is a dot product: a term wrong by E moves the sum by E whatever the term is.
         * 1e-5 against an O(1) sum is two orders below that sum's own bf16 rounding. */
        if (fb(h[i].max_abs) > 1.0e-5f) {
            printf("   FAIL %s: absolute error %.3g exceeds 1e-5 somewhere in the domain\n",
                   NAME[i], (double)fb(h[i].max_abs));
            fail = 1;
        }
    }
    /* The whole-domain `ulp` column is large and that is EXPECTED, not a warning sign: near the
     * origin every one of these functions is asymptotically linear (tanh(y)->y, silu(x)->x/2,
     * gelu(x)->x/2), the exponential form's error there is a fixed ~1e-7 ABSOLUTE, and a fixed
     * absolute error against a value going to zero is unboundedly many ulp of a denormal bf16.
     * The `max |abs|` column beside it is the real content: nothing anywhere in the 2^32 domain
     * moves by more than that, whatever the ulp count says. */
    printf("\nwhere the bf16 moved at all, over |x| in:  (n with |ref| >= %g, the denominator\n"
           "                                            for the SIGNAL REGION column above)\n",
           (double)BIG);
    for (int i = 0; i < NFN; i++) {
        if (h[i].n_diff)
            printf("  %-22s [%.4g, %.4g]   n=%llu\n", NAME[i], (double)fb(h[i].absx_lo),
                   (double)fb(h[i].absx_hi), (unsigned long long)h[i].n_big);
        else
            printf("  %-22s (never)          n=%llu\n", NAME[i],
                   (unsigned long long)h[i].n_big);
    }

    /* ---- the router, which is a MEASUREMENT and not an assertion ---------------------------
     * op_moe.h's router still computes its sigmoid as `1.0f / (1.0f + expf(-logit))`, the slow
     * spelling, ON PURPOSE. This block is the evidence for that decision, so it prints numbers
     * and does not fail the test: it asks what WOULD happen to expert selection if the router
     * adopted the fast `k3_sigmoid`. If a future change makes the router use it, these lines
     * become the gate and `ROUTER_IS_A_GATE` below should be flipped to 1. */
    printf("\nROUTER — what adopting the fast sigmoid would do to expert selection\n");
    printf("  (frozen 1/(1+__expf) vs rcp(1+__expf); same logits, same packed-key rule)\n");
    struct { unsigned n_exp, topk, tie; const char* what; } cfg[] = {
        {896, 16, 0, "K3      896/top-16 grid bias"},
        {896, 16, 1, "K3      896/top-16 f32 bias"},
        {896, 16, 2, "K3      896/top-16 NO bias  "},
        {256, 8, 0, "GLM/DS  256/top-8  grid bias"},
        {256, 8, 1, "GLM/DS  256/top-8  f32 bias"},
        {256, 8, 2, "GLM/DS  256/top-8  NO bias  "},
    };
    unsigned *dset, *dord;
    HC(hipMalloc(&dset, 4)); HC(hipMalloc(&dord, 4));
    unsigned tot_set = 0, tot_ord = 0;
    for (unsigned c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        const unsigned trials = 8192;
        HC(hipMemset(dset, 0, 4)); HC(hipMemset(dord, 0, 4));
        hipLaunchKernelGGL(router_sel, dim3((trials + 63) / 64), dim3(64), 0, 0, dset, dord,
                           cfg[c].n_exp, cfg[c].topk, cfg[c].tie, trials);
        HC(hipDeviceSynchronize());
        unsigned hs = 0, ho = 0;
        HC(hipMemcpy(&hs, dset, 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(&ho, dord, 4, hipMemcpyDeviceToHost));
        printf("  %-28s %5u vectors : chosen SET differs %5u   ORDER differs %5u\n", cfg[c].what,
               trials, hs, ho);
        tot_set += hs; tot_ord += ho;
    }
    printf("  => totals: chosen SET differs on %u vectors, ORDER on %u.\n", tot_set, tot_ord);
    printf("     Selecting a DIFFERENT EXPERT is a discrete change in the token stream, not a\n"
           "     rounding difference. The router's sigmoid is ~1 expf per thread against an\n"
           "     O(n_exp^2) LDS rank pass, i.e. a few percent of that kernel, so it is not worth\n"
           "     ANY selection movement. op_moe.h keeps the slow spelling deliberately.\n");
#define ROUTER_IS_A_GATE 0
#if ROUTER_IS_A_GATE
    if (tot_set || tot_ord) fail = 1;
#endif

    printf("\n%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
