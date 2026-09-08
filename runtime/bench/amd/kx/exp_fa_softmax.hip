/* kx experiment: fa_softmax — what does the flash-prefill row softmax actually issue?
 *
 * THE QUESTION. `__shfl_xor` has exactly ONE lowering on gfx9: `ds_bpermute_b32`, an LDS crossbar
 * op with LDS latency and an `s_waitcnt lgkmcnt` behind it. The online-softmax epilogue of
 * `d_flash_prefill` runs that 16 accumulator rows x 5 steps x {max, sum} = 160 times per KV tile,
 * against 64 MFMAs of actual work -- and pays 163 `s_waitcnt` for it. `PLOW_WAVE_RED_DPP` replaces
 * the four intra-row steps with DPP (a VALU operand modifier, no LDS at all) and keeps only the
 * xor-16 step on `ds_swizzle`: 160 LDS permutes become 64.
 *
 * This is the experiment that most needs `--isa`. The timing here is real but small; the ANSWER is
 * an instruction count, and "what does this body actually issue?" is a flag.
 *
 * WHAT IS INCLUDED VERBATIM. `half_wave_max` and `half_wave_sum` are the shipped reductions from
 * amd_common.h, selected by the same `-DPLOW_WAVE_RED_DPP` the flash object is built with, and the
 * loop below is `d_flash_prefill`'s own NKT==1 rescale epilogue (op_attention.h:654-667) with the
 * `oacc` rescale dropped -- the reduction is what varies, the PV accumulate is not.
 *
 * WHY THE ORDER OF THE SUM IS NOT THE ORDER OF THE MAX, and why it is not a mistake: f32 add is
 * not associative, so `half_wave_sum` keeps the original 16,8,4,2,1 order and pays LDS swizzle for
 * xor-16/8/4; `fmaxf` is associative and exact, so `half_wave_max` runs ascending and pays LDS for
 * one step. That asymmetry is what makes the DPP form BIT-IDENTICAL to the shfl form, and this
 * experiment declares it KX_EXACT so the claim is checked rather than repeated.
 *
 * AND ONE TRAP, on purpose. `k_trunc` reduces over 16 lanes instead of 32 by dropping the xor-16
 * step. It is faster -- 128 permutes instead of 160 -- and it is WRONG: each row's softmax then
 * covers half the columns it owns. This is the exact shape of the FA_MLA_PF2_BKV=40 result, which
 * cut TTFT 12153 -> 10903 ms (-10.3%) and was fast because it was silently dropping 8 of every 40
 * KV rows. The harness must reject it, and the point of shipping it here is that the rejection is
 * demonstrated on every run rather than asserted in a comment.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_attention.h"

/* THE TRAP, bench-local: `half_wave_max`/`half_wave_sum` with the xor-16 step deleted, so the
 * reduction spans 16 lanes where the MFMA 32x32 accumulator layout needs 32. Deliberately NOT in
 * amd_common.h -- nothing shipped may reach it. */
__device__ __forceinline__ float kx_quarter_max(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, PLOW_WAVE));
    return v;
}
__device__ __forceinline__ float kx_quarter_sum(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_xor(v, off, PLOW_WAVE);
    return v;
}

/* One KV tile of the shipped online-softmax epilogue. RMAX/RSUM are the reduction pair under
 * test; everything else -- the 16 rows, the running (m, l) state, the correction factor, the
 * FA_NEG_INF guards, the order of operations -- is op_attention.h:654-667 unchanged. */
template <float (*RMAX)(float), float (*RSUM)(float)>
__device__ __forceinline__ void kx_softmax_tiles(KX_SIG) {
    const float* const s0g = (const float*)kx_x;
    float* const o = (float*)kx_out;
    (void)kx_w;
    (void)kx_aux;
    (void)kx_nblk;
    (void)kx_d0;
    (void)kx_d1;
    (void)kx_d2;
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    float s0[16], p[16], m_st[16], l_st[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        /* Bounded, lane-varying scores; the & keeps every rep inside the seeded buffer. */
        s0[i] = s0g[(tid * 16u + (unsigned)i) & 0xFFFFFu] * 1e-2f;
        m_st[i] = FA_NEG_INF;
        l_st[i] = 0.0f;
    }

    /* `nrep` KV tiles. p is re-seeded per tile from registers rather than reloaded, so the
     * instruction mix stays the reduction's and not a memory system's. */
    for (unsigned e = 0; e < kx_nrep; e++) {
#pragma unroll
        for (int i = 0; i < 16; i++) p[i] = s0[i] + (float)e * 0.125f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float rmax = RMAX(p[i]);
            const float mnew = fmaxf(m_st[i], rmax);
            const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
            const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
            l_st[i] = l_st[i] * corr + RSUM(pe);
            m_st[i] = mnew;
        }
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        o[tid * 32u + (unsigned)i] = m_st[i];
        o[tid * 32u + 16u + (unsigned)i] = l_st[i];
    }
}

KX_BODY(b_soft) { kx_softmax_tiles<half_wave_max, half_wave_sum>(KX_FWD); }
KX_ARM_OF(k_soft, b_soft)
KX_ARM_OF(k_soft_aa, b_soft)

KX_BODY(b_trunc) { kx_softmax_tiles<kx_quarter_max, kx_quarter_sum>(KX_FWD); }
KX_ARM_OF(k_trunc, b_trunc)

/* WHAT THIS EXPERIMENT DELIBERATELY DOES NOT ANSWER. FA_LAZY_RESCALE (op_attention.h:158) -- the
 * wave-voted skip of the corr/rescale when no running max moved -- was ported here and then
 * REMOVED, because a single-block harness cannot answer it honestly. Whether the vote pays is a
 * property of the SCORE DISTRIBUTION the MFMA produces, not of the body: against this bench's
 * synthetic scores the arm read 1.25x FASTER than plain DPP, the opposite of the measured tree
 * result (+1.2% / +0.3% TTFT on top of DPP+FASTMASK, and +13.7% prefill with 622 B/thread of
 * induced scratch in its 2026-08 form). A data-dependent branch belongs in a campaign against
 * real activations. Shipping the arm would have made this harness produce exactly the kind of
 * confident wrong number it exists to prevent. */

#else /* ------------------------------------------------------- host descriptor */

/* The variant axis IS the question: one -D, two objects, the same source. */
static const kx_variant VAR[] = {
    {"shfl", "-DPLOW_WAVE_RED_DPP=0"},
    {"dpp", "-DPLOW_WAVE_RED_DPP=1"},
};

static const kx_arm ARM[] = {
    {"k_soft", 0, KX_BASE | KX_REF, "shipped half_wave_max/sum via __shfl_xor -> ds_bpermute"},
    {"k_soft_aa", 0, KX_AA, "byte-identical device code to shfl:k_soft"},
    {"k_soft", 1, KX_EXACT, "PLOW_WAVE_RED_DPP: DPP + one ds_swizzle. Claimed BIT-IDENTICAL"},
    {"k_trunc", 0, KX_EXACT, "TRAP: reduces over 16 lanes, not 32. Faster and WRONG"},
};

/* `reps` is KV tiles per launch: 32 tiles is a 1024-column row-block, 128 is 4096. The reduction
 * is per-wave, so `grid` only changes how many CUs are busy; 304 is one workgroup per CU on
 * MI300X, which is what the flash bucket dispatches at prefill widths. */
static const kx_shape SHP[] = {
    {"kv_32tiles", 0, 0, 0, 304, 0, 0.0, 32, 304u * 256u * 32u},
    {"kv_128tiles", 0, 0, 0, 304, 0, 0.0, 128, 304u * 256u * 32u},
};

static const kx_exp EXP = {
    "fa_softmax",
    "does the DPP row-softmax cut the flash tile's LDS permutes, and stay bit-identical?",
    "runtime/bench/amd/kx/exp_fa_softmax.hip",
    /* scripts/build_gfx942.sh's AX_FLASH shape: the flash object is FOUR waves, not eight. Without
     * PLOW_WG_WAVES=4 this measures the interpreter's decode geometry instead. */
    "-DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DFA_FASTMASK=1",
    "gfx942",
    256,
    KX_DT_F32,
    VAR,
    2,
    ARM,
    4,
    SHP,
    2,
    0.98,
    1.02,
    0.0,
    "DPP: 160 ds_bpermute + 163 s_waitcnt per tile -> 64 ds_swizzle + 91, bit-identical "
    "(amd_common.h:1144, build_gfx942.sh:266). k_trunc must be REJECTED as wrong.",
};

extern "C" const kx_exp* kx_experiment(void) { return &EXP; }

#endif
