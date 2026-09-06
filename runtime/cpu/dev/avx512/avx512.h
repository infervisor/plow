/* avx512.h — private helpers for the AVX-512 tier (F/BW/VL/BF16). Inner-loop choices cite
 * plans/cpu-kernel-innerloops.md ("spec") by section. */
#ifndef PLOW_CPU_AVX512_H
#define PLOW_CPU_AVX512_H

#include <immintrin.h>
#include "golden/golden.h" /* g_range, g_amax_pack, G_QNAN, scalar act refs, golden fallbacks */

#define V_K(name) \
    void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* gemv.c */
V_K(v_gemv);
V_K(v_gemv_glu);
V_K(v_gemv_qkv);
V_K(v_gemv_argmax);
/* pointwise.c */
V_K(v_residual);
V_K(v_glu);
V_K(v_softcap);
V_K(v_embed);
V_K(v_argmax);
V_K(v_argmax_fin);
/* norm.c */
V_K(v_rmsnorm);
V_K(v_rowrms);
V_K(v_layernorm);
V_K(v_norm_residual);
V_K(v_add_norm);
V_K(v_norm_residual_norm);
/* rope.c */
V_K(v_headnorm_rope);
/* attention.c */
V_K(v_flash_prefill);
V_K(v_flash_decode);
V_K(v_flash_merge);
void v_register_attention(plow_cpu_kernel_fn* tab);
/* gptoss.c / moe.c */
V_K(v_gemv_mxfp4);
V_K(v_moe_glu_mx);
V_K(v_moe_down_mx);
V_K(v_moe_glu_mx_b);  /* 147 at B >= 2: grouped by expert */
V_K(v_moe_down_mx_b); /* 148 at B >= 2 */
V_K(v_moe_glu_mx_pf);
V_K(v_moe_down_mx_pf);
void v_register_gptoss(plow_cpu_kernel_fn* tab);
void v_register_moe_gemma(plow_cpu_kernel_fn* tab); /* moe_gemma.c: Gemma-4 26B-A4B MoE ops */

/* --- bf16 <-> f32 (spec §5: explicit widen, cvtneps_pbh on store) ------------------- */

static inline __m512 v_load_bf16(const plow_bf16* p) {
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256((const __m256i*)p));
}
static inline __m512 v_load_bf16_mask(const plow_bf16* p, __mmask16 m) {
    return _mm512_cvtpbh_ps((__m256bh)_mm256_maskz_loadu_epi16(m, p));
}
static inline void v_store_bf16(plow_bf16* p, __m512 v) {
    _mm256_storeu_si256((__m256i*)p, (__m256i)_mm512_cvtneps_pbh(v));
}
static inline void v_store_bf16_mask(plow_bf16* p, __mmask16 m, __m512 v) {
    _mm256_mask_storeu_epi16(p, m, (__m256i)_mm512_cvtneps_pbh(v));
}
/* Round through bf16 and back (the GPU's intermediate bf16 round points). */
static inline __m512 v_round_bf16(__m512 v) {
    return _mm512_cvtpbh_ps(_mm512_cvtneps_pbh(v));
}
static inline __mmask16 v_tail16(uint32_t n) { return (__mmask16)((1u << n) - 1u); }

/* --- exp / activations (spec §5: ggml_v_expf, 1.45+0.5 ulp, vscalefps) ---------------- */

static inline __m512 v_expf(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192.0f), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)),
                        u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0f)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) return res;
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ),
                                            _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

static inline __m512 v_sigmoid(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, v_expf(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}
static inline __m512 v_silu(__m512 x) { return _mm512_mul_ps(x, v_sigmoid(x)); }
/* tanh(x) = 1 - 2 / (1 + exp(2x)); exp overflow/underflow lands on +-1 exactly. */
static inline __m512 v_tanh(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f), two = _mm512_set1_ps(2.0f);
    const __m512 e = v_expf(_mm512_mul_ps(two, x));
    return _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(one, e)));
}
static inline __m512 v_gelu_tanh(__m512 x) {
    const __m512 c = _mm512_mul_ps(
        _mm512_set1_ps(0.7978845608028654f),
        _mm512_fmadd_ps(_mm512_set1_ps(0.044715f), _mm512_mul_ps(_mm512_mul_ps(x, x), x), x));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x),
                         _mm512_add_ps(_mm512_set1_ps(1.0f), v_tanh(c)));
}
/* act 0 = gelu_tanh, 1 = silu, 2 = situ pair form (caller handles); NaN otherwise like golden. */
static inline __m512 v_act_gate(__m512 g, uint32_t act) {
    if (act == 1u) return v_silu(g);
    if (act == 0u) return v_gelu_tanh(g);
    return _mm512_set1_ps(NAN);
}
/* act 3 = swiglu_oai pair form (golden g_swiglu_oai). */
static inline __m512 v_swiglu_oai(__m512 g, __m512 u, float alpha, float limit) {
    const __m512 vl = _mm512_set1_ps(limit);
    g = _mm512_min_ps(g, vl);
    u = _mm512_max_ps(_mm512_min_ps(u, vl), _mm512_sub_ps(_mm512_setzero_ps(), vl));
    const __m512 glu = _mm512_mul_ps(g, v_sigmoid(_mm512_mul_ps(g, _mm512_set1_ps(alpha))));
    return _mm512_mul_ps(glu, _mm512_add_ps(u, _mm512_set1_ps(1.0f)));
}
/* golden g_glu_pair, 16 lanes. */
static inline __m512 v_glu_pair(__m512 g, __m512 u, uint32_t act, float f0, float f1) {
    if (act == 2u) {
        const __m512 vb = _mm512_set1_ps(f0);
        const __m512 gate = _mm512_mul_ps(_mm512_mul_ps(vb, v_tanh(_mm512_div_ps(g, vb))), v_sigmoid(g));
        __m512 up = u;
        if (f1 > 0.0f) {
            const __m512 vl = _mm512_set1_ps(f1);
            up = _mm512_mul_ps(vl, v_tanh(_mm512_div_ps(u, vl)));
        }
        return _mm512_mul_ps(gate, up);
    }
    if (act == 3u) return f1 > 0.0f ? v_swiglu_oai(g, u, f0, f1) : _mm512_set1_ps(NAN);
    return _mm512_mul_ps(v_act_gate(g, act), u);
}

/* --- row helpers shared by norm.c and the gemv norm folds --------------------------------- */

/* Sum of squares of a bf16 row, f32 accumulate (two chains per 32 elements). */
static inline float v_row_ss(const plow_bf16* x, uint32_t n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m512 a = v_load_bf16(x + i), b = v_load_bf16(x + i + 16);
        acc0 = _mm512_fmadd_ps(a, a, acc0);
        acc1 = _mm512_fmadd_ps(b, b, acc1);
    }
    for (; i < n; i += 16) {
        const __mmask16 m = i + 16 <= n ? 0xFFFF : v_tail16(n - i);
        const __m512 a = v_load_bf16_mask(x + i, m);
        acc0 = _mm512_fmadd_ps(a, a, acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

/* o = bf16(x * inv * gamma?) */
static inline void v_scale_row(plow_bf16* o, const plow_bf16* x, const plow_bf16* gamma, float inv,
                               uint32_t n) {
    const __m512 vinv = _mm512_set1_ps(inv);
    for (uint32_t i = 0; i < n; i += 16) {
        const __mmask16 m = i + 16 <= n ? 0xFFFF : v_tail16(n - i);
        __m512 v = _mm512_mul_ps(v_load_bf16_mask(x + i, m), vinv);
        if (gamma) v = _mm512_mul_ps(v, v_load_bf16_mask(gamma + i, m));
        v_store_bf16_mask(o + i, m, v);
    }
}

/* --- argmax key (golden g_amax_pack, 16 lanes) ----------------------------------------- */

/* Order-preserving 32-bit key of 16 bf16 bit patterns held in epi32 lanes. */
static inline __m512i v_amax_key(__m512i b) {
    const __m512i sign = _mm512_set1_epi32(0x8000);
    const __mmask16 neg = _mm512_test_epi32_mask(b, sign);
    const __m512i pos = _mm512_or_si512(b, sign);
    const __m512i negk = _mm512_and_si512(_mm512_xor_si512(b, _mm512_set1_epi32(-1)),
                                          _mm512_set1_epi32(0xFFFF));
    return _mm512_mask_blend_epi32(neg, pos, negk);
}

/* Reduce per-lane (key, index) bests to golden's packed u64 max. */
static inline uint64_t v_amax_fold(__m512i keys, __m512i idx, uint64_t best) {
    uint32_t k[16], ix[16];
    _mm512_storeu_si512(k, keys);
    _mm512_storeu_si512(ix, idx);
    for (int l = 0; l < 16; l++) {
        if (k[l] == 0u) continue; /* lane never saw an element */
        const uint64_t p = ((uint64_t)k[l] << 32) | (uint64_t)(~ix[l]);
        best = p > best ? p : best;
    }
    return best;
}

#endif /* PLOW_CPU_AVX512_H */
