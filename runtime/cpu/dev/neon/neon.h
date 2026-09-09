/* neon.h — private helpers for the aarch64 vector tier (armv8.6 NEON + bf16 + i8mm).
 * The x86 counterpart is avx512/avx512.h; the kernels keep the same shapes so the two tiers
 * stay comparable. Lane width is 4 f32 / 8 bf16; the bf16 dot is `vbfdotq_f32`. */
#ifndef PLOW_CPU_NEON_H
#define PLOW_CPU_NEON_H

#include <arm_neon.h>
#include "golden/golden.h" /* g_range, g_amax_pack, G_QNAN, scalar act refs, golden fallbacks */

#define N_K(name) \
    void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* gemv.c */
N_K(n_gemv);
N_K(n_gemv_glu);
N_K(n_gemv_qkv);
N_K(n_gemv_argmax);
/* gemm.c */
N_K(n_gemm);
N_K(n_gemm_small);
N_K(n_gemm_med);
N_K(n_gemm_wide);
N_K(n_gemm_c5);
N_K(n_gemm_glu);
/* pointwise.c */
N_K(n_residual);
N_K(n_glu);
N_K(n_softcap);
N_K(n_embed);
N_K(n_argmax);
N_K(n_argmax_fin);
/* norm.c */
N_K(n_rmsnorm);
N_K(n_rowrms);
N_K(n_layernorm);
N_K(n_norm_residual);
N_K(n_add_norm);
N_K(n_norm_residual_norm);
/* rope.c */
N_K(n_headnorm_rope);
void n_gemm_tile(plow_bf16* C, size_t ldc, const plow_bf16* A, const plow_bf16* B, size_t b_row0,
                 const plow_bf16* bias, const float* rs, const float* cs, uint32_t K, uint32_t m0,
                 uint32_t m1, uint32_t n0, uint32_t n1);
void n_gemm_glu_tile(plow_bf16* C, size_t ldc, const plow_bf16* A, const plow_bf16* Wg,
                     const plow_bf16* Wu, size_t w_row0, const plow_bf16* bg, const plow_bf16* bu,
                     const float* gs, const float* us, uint32_t K, uint32_t m0, uint32_t m1,
                     uint32_t n0, uint32_t n1, uint32_t act, float f0, float f1);
/* fp8.c */
N_K(n_gemv_fp8);
N_K(n_gemv_glu_fp8);
N_K(n_gemm_fp8);
N_K(n_gemm_med_fp8);
N_K(n_gemm_small_fp8);
N_K(n_gemm_wide_fp8);
N_K(n_gemm_c5_fp8);
N_K(n_gemm_glu_fp8);
/* attention.c */
N_K(n_flash_prefill);
N_K(n_flash_decode);
N_K(n_flash_merge);

/* --- bf16 <-> f32 ------------------------------------------------------------------- */

static inline bfloat16x8_t n_ldbh(const plow_bf16* p) {
    return vreinterpretq_bf16_u16(vld1q_u16(p));
}
/* Widen 8 bf16 to two f32x4 (bf16 is the top half of f32). */
static inline float32x4_t n_lo_f32(uint16x8_t v) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
}
static inline float32x4_t n_hi_f32(uint16x8_t v) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16));
}
static inline float32x4_t n_load4(const plow_bf16* p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
/* RNE f32 -> bf16 (FPCR default rounding; NaN keeps its quiet bit) — matches plow_f2bf. */
static inline void n_store4(plow_bf16* p, float32x4_t v) {
    vst1_u16(p, vreinterpret_u16_bf16(vcvt_bf16_f32(v)));
}
static inline float32x4_t n_round4(float32x4_t v) {
    return vreinterpretq_f32_u32(vshll_n_u16(vreinterpret_u16_bf16(vcvt_bf16_f32(v)), 16));
}
static inline float n_hsum(float32x4_t v) { return vaddvq_f32(v); }

/* Tail-safe 4-lane load/store: n in 1..4. */
static inline float32x4_t n_load4_n(const plow_bf16* p, uint32_t n) {
    plow_bf16 t[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < n; i++) t[i] = p[i];
    return n_load4(t);
}
static inline void n_store4_n(plow_bf16* p, float32x4_t v, uint32_t n) {
    plow_bf16 t[4];
    n_store4(t, v);
    for (uint32_t i = 0; i < n; i++) p[i] = t[i];
}

/* --- exp / activations (ggml_v_expf port, 4 lanes) ---------------------------------- */

static inline float32x4_t n_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
        u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static inline float32x4_t n_sigmoid(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, n_expf(vnegq_f32(x))));
}
static inline float32x4_t n_silu(float32x4_t x) { return vmulq_f32(x, n_sigmoid(x)); }
/* tanh(x) = 1 - 2 / (1 + exp(2x)); exp overflow/underflow lands on +-1 exactly. */
static inline float32x4_t n_tanh(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
    const float32x4_t e = n_expf(vmulq_f32(two, x));
    return vsubq_f32(one, vdivq_f32(two, vaddq_f32(one, e)));
}
static inline float32x4_t n_gelu_tanh(float32x4_t x) {
    const float32x4_t c = vmulq_f32(
        vdupq_n_f32(0.7978845608028654f),
        vfmaq_f32(x, vdupq_n_f32(0.044715f), vmulq_f32(vmulq_f32(x, x), x)));
    return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), x), vaddq_f32(vdupq_n_f32(1.0f), n_tanh(c)));
}
static inline float32x4_t n_act_gate(float32x4_t g, uint32_t act) {
    if (act == 1u) return n_silu(g);
    if (act == 0u) return n_gelu_tanh(g);
    return vdupq_n_f32(NAN);
}
static inline float32x4_t n_swiglu_oai(float32x4_t g, float32x4_t u, float alpha, float limit) {
    const float32x4_t vl = vdupq_n_f32(limit);
    g = vminq_f32(g, vl);
    u = vmaxq_f32(vminq_f32(u, vl), vnegq_f32(vl));
    const float32x4_t glu = vmulq_f32(g, n_sigmoid(vmulq_f32(g, vdupq_n_f32(alpha))));
    return vmulq_f32(glu, vaddq_f32(u, vdupq_n_f32(1.0f)));
}
/* golden g_glu_pair, 4 lanes. */
static inline float32x4_t n_glu_pair(float32x4_t g, float32x4_t u, uint32_t act, float f0, float f1) {
    if (act == 2u) {
        const float32x4_t vb = vdupq_n_f32(f0);
        const float32x4_t gate = vmulq_f32(vmulq_f32(vb, n_tanh(vdivq_f32(g, vb))), n_sigmoid(g));
        float32x4_t up = u;
        if (f1 > 0.0f) {
            const float32x4_t vl = vdupq_n_f32(f1);
            up = vmulq_f32(vl, n_tanh(vdivq_f32(u, vl)));
        }
        return vmulq_f32(gate, up);
    }
    if (act == 3u) return f1 > 0.0f ? n_swiglu_oai(g, u, f0, f1) : vdupq_n_f32(NAN);
    return vmulq_f32(n_act_gate(g, act), u);
}

/* --- row helpers ---------------------------------------------------------------------- */

/* Sum of squares of a bf16 row, f32 accumulate, two chains. */
static inline float n_row_ss(const plow_bf16* x, uint32_t n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t v = vld1q_u16(x + i);
        const float32x4_t lo = n_lo_f32(v), hi = n_hi_f32(v);
        a0 = vfmaq_f32(a0, lo, lo);
        a1 = vfmaq_f32(a1, hi, hi);
    }
    float s = n_hsum(vaddq_f32(a0, a1));
    for (; i < n; i++) {
        const float v = plow_bf2f(x[i]);
        s += v * v;
    }
    return s;
}

/* o = bf16(x * inv * gamma?). */
static inline void n_scale_row(plow_bf16* o, const plow_bf16* x, const plow_bf16* gamma, float inv,
                               uint32_t n) {
    const float32x4_t vinv = vdupq_n_f32(inv);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_f32(n_load4(x + i), vinv);
        if (gamma) v = vmulq_f32(v, n_load4(gamma + i));
        n_store4(o + i, v);
    }
    for (; i < n; i++) {
        const float g = gamma ? plow_bf2f(gamma[i]) : 1.0f;
        o[i] = plow_f2bf(plow_bf2f(x[i]) * inv * g);
    }
}

#endif /* PLOW_CPU_NEON_H */
