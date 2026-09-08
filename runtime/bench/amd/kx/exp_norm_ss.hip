/* kx experiment: norm_ss — what would F5's scale-safe RMSNorm cost, and what does the
 * default-off range check cost?                    [F5, plans/gpu-kernel-static-audit-20260908.md]
 *
 * THE QUESTION. `d_rmsnorm` reduces sum(x_i^2) in FP32 over bf16 inputs. bf16 carries FP32's
 * exponent range, so a finite input can square to +inf and the row then normalizes to ZERO
 * (rsqrt(inf) = 0) with no error anywhere. The textbook remedy is a scale-safe two pass: take
 * max|x| first, divide it out, reduce the scaled squares, and put the scale back. It is exact for
 * every finite bf16 row. It also costs a second workgroup-wide reduction and a second barrier on
 * an op that a decode step runs on ONE CU with 303 stalled behind it, and it changes the rounding
 * of every token of every layer.
 *
 * Whether to pay that is a measurement, not an opinion, and the measured input ranges say the
 * margin is already 1e16x (op_norm.h's contract note). This experiment is the other half of that
 * decision: the price.
 *
 * WHAT IS INCLUDED VERBATIM. `d_rmsnorm` from op_norm.h, called with the shipped argument shape.
 * The candidate is the SAME body with the reduction replaced -- same register staging, same
 * `rn_index` element map, same producer/consumer load order -- so the delta is the reduction and
 * nothing else.
 *
 * THE THIRD ARM IS THE SHIPPED BODY UNDER `-DPLOW_NORM_RANGE_CHECK=1`. It must be bit-identical
 * (the check only traps), and the point of measuring it is to price the debug build a range
 * campaign actually runs.
 *
 * WHAT THIS CANNOT ANSWER: whether the two-pass rounding change moves a served completion. It is
 * a different arithmetic and the answer to that is a greedy-identity campaign, not a body bench.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_norm.h"

/* SCALE-SAFE RMSNorm: bench-local, deliberately NOT in op_norm.h -- nothing shipped may reach it.
 * `d_rmsnorm`'s `fits` path with one change: max|x| is reduced first and divided out before the
 * squares, so no intermediate can overflow for any finite bf16 row.
 *
 *      inv = rsqrt(sum((x/t)^2)/feat + eps/t^2) / t     with t = max|x| (t == 0 -> inv = rsqrt(eps))
 *
 * Two block-wide reductions and two barriers where the shipped body has one. */
__device__ void kx_rmsnorm_safe(bf16* __restrict__ out, const bf16* __restrict__ x,
                                const bf16* __restrict__ gamma, unsigned rows, unsigned feat,
                                float eps, unsigned slice, unsigned nblk, float* part) {
    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    auto* og = as_glob(out);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        bf16v8 v[rn_vecs], w[rn_vecs];
#pragma unroll
        for (int c = 0; c < rn_vecs; c++) {
            const unsigned i = rn_index(c);
            v[c] = bf16v8_zero();
            w[c] = bf16v8_zero();
            if (i < feat) {
                v[c] = ld_glob8(xg + base + i);
                if (gamma) w[c] = ld_glob8(gg + i);
            }
        }
        float amax = 0.0f;
#pragma unroll
        for (int c = 0; c < rn_vecs; c++)
#pragma unroll
            for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bf2f(v[c][j])));
        const float t = block_max(amax, part);
        const float rt = t > 0.0f ? 1.0f / t : 0.0f;
        float ss = 0.0f;
#pragma unroll
        for (int c = 0; c < rn_vecs; c++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = bf2f(v[c][j]) * rt;
                ss += f * f;
            }
        const float m = block_sum(ss, part) / (float)feat;
        const float inv = t > 0.0f ? rsqrtf(m + eps * rt * rt) * rt : rsqrtf(eps);
#pragma unroll
        for (int c = 0; c < rn_vecs; c++) {
            const unsigned i = rn_index(c);
            if (i < feat) {
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gamma ? bf2f(w[c][j]) : 1.0f;
                    o[j] = f2bf(bf2f(v[c][j]) * inv * g);
                }
                st_glob8(og + i + (size_t)row * feat, o);
            }
        }
    }
}

/* kx_w is the 3 GiB arena and holds the ACTIVATIONS -- `nrep` walks a fresh slab out of it so a
 * prefill shape streams instead of replaying one L2-resident row. kx_x (64 MiB) holds gamma.
 * d0 = rows, d1 = feat; rows*feat*2 must stay inside the 64 MiB output buffer. */
__device__ __forceinline__ void kx_norm(bool safe, KX_SIG) {
    __shared__ float part[PLOW_WAVES * 2];
    (void)kx_aux;
    (void)kx_d2;
    const unsigned rows = kx_d0, feat = kx_d1;
    const bf16* g = (const bf16*)kx_x;
    for (unsigned e = 0; e < kx_nrep; e++) {
        const bf16* xin = (const bf16*)kx_w + (size_t)e * rows * feat;
        if (safe)
            kx_rmsnorm_safe((bf16*)kx_out, xin, g, rows, feat, 1e-5f, blockIdx.x, kx_nblk, part);
        else
            d_rmsnorm((bf16*)kx_out, xin, g, rows, feat, 1e-5f, 0u, blockIdx.x, kx_nblk, part);
    }
}

KX_BODY(b_ship) { kx_norm(false, KX_FWD); }
KX_ARM_OF(k_norm, b_ship)
KX_ARM_OF(k_norm_aa, b_ship)

KX_BODY(b_safe) { kx_norm(true, KX_FWD); }
KX_ARM_OF(k_norm_safe, b_safe)

#else /* ------------------------------------------------------- host descriptor */

/* The variant axis is the RANGE CHECK: same source, one -D, two objects. Variant 1's `k_norm`
 * must be bit-identical to variant 0's and must not cost anything measurable -- that is the claim
 * "default off means the shipped objects carry no instruction from it" being checked rather than
 * asserted. (The stronger form of the same claim is in the commit: a full 45-object gfx942 build
 * with the check off disassembles IDENTICALLY to one from before it existed.) */
static const kx_variant VAR[] = {
    {"off", "-DPLOW_NORM_RANGE_CHECK=0"},
    {"chk", "-DPLOW_NORM_RANGE_CHECK=1"},
};

static const kx_arm ARM[] = {
    {"k_norm", 0, KX_BASE | KX_REF, "shipped d_rmsnorm (one block_sum of raw squares)"},
    {"k_norm_aa", 0, KX_AA, "byte-identical device code to off:k_norm"},
    {"k_norm", 1, KX_EXACT, "PLOW_NORM_RANGE_CHECK=1: the same body plus a trap predicate"},
    {"k_norm_safe", 0, 0, "F5's remedy: block_max, scale, reduce, rescale. Overflow-proof, and a "
                          "second reduction + barrier per row"},
};

/* GLM-5.3 TP4's own RmsNorm shapes and workgroup counts, from
 * `plowrt disasm build-glm53/tp4-long/model.pkt`:
 *   decode program T=1   78 x rows=1 feat=2048 b=1     (q_a_layernorm)
 *                         2 x rows=1 feat=6144 b=1     (the two unfused residual norms)
 *   prefill T=8192      157 x rows=8192 feat=6144 b=304   (run here at 2432 rows, see below)
 * The decode rows are the ones that matter: one workgroup, one row, everything else stalled. */
static const kx_shape SHP[] = {
    {"glm_dec_qa", 1, 2048, 0, 1, 78, 0.0, 64, 2048},
    {"glm_dec_x", 1, 6144, 0, 1, 2, 0.0, 64, 6144},
    /* 2432 rows, not the packet's 8192: rows*feat*2 has to fit the harness's 64 MiB output
     * buffer. The per-workgroup share (8 rows at grid 304) is what the body actually measures. */
    {"glm_pf_x", 2432, 6144, 0, 304, 157, 2432.0 * 6144.0 * 2.0, 0, 6144},
};

static const kx_exp EXP = {
    "norm_ss",
    "what would a scale-safe RMSNorm cost, and does the default-off range check cost anything?",
    "runtime/bench/amd/kx/exp_norm_ss.hip",
    /* The decode object's own shape: eight waves, bucketed decode. */
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8 -DPLOW_GEMV_MM=1",
    "gfx942",
    512,
    KX_DT_BF16,
    VAR,
    2,
    ARM,
    4,
    SHP,
    3,
    0.98,
    1.02,
    /* The safe arm divides by max|x| and multiplies it back, so its bf16 outputs differ in the
     * last mantissa bit or two; 0.02 admits that on the arena's magnitudes and still rejects an
     * arm that normalized by the wrong row. */
    0.02,
    "the range check is FREE and bit-identical (it is an identity when off); the scale-safe "
    "two-pass costs a second block reduction per row and is the arm F5 does not justify buying.",
};

extern "C" const kx_exp* kx_experiment(void) { return &EXP; }

#endif
