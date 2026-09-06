/* golden.h — scalar reference kernels for the device ISA (one file per family). */
#ifndef PLOW_CPU_GOLDEN_H
#define PLOW_CPU_GOLDEN_H

#include <math.h>
#include <string.h>
#include "cpu_dev.h"

#define G_K(name) \
    void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

/* control.c */
G_K(g_nop);
/* elementwise.c */
G_K(g_residual);
G_K(g_glu);
G_K(g_softcap);
G_K(g_embed);
G_K(g_argmax);
G_K(g_argmax_fin);
/* norm.c */
G_K(g_rmsnorm);
G_K(g_rowrms);
G_K(g_layernorm);
G_K(g_headnorm_rope);
G_K(g_norm_residual);
G_K(g_add_norm);
G_K(g_norm_residual_norm);
/* gemm.c */
G_K(g_gemm);
G_K(g_gemm_small);
G_K(g_gemm_med);
G_K(g_gemm_wide);
G_K(g_gemm_c5);
G_K(g_gemm_norm);
G_K(g_gemm_glu);
G_K(g_gemv);
G_K(g_gemv_glu);
G_K(g_gemv_qkv);
G_K(g_gemv_argmax);
/* attention.c */
G_K(g_flash_prefill);
G_K(g_flash_decode);
G_K(g_flash_merge);
/* k3.c */
G_K(g_attn_res);

/* --- shared math (matches runtime/amd/amd_common.h + op_elementwise.h) ------------- */

#define G_WAVES 8u /* PLOW_WG_WAVES: per-wave work items are packed 8 per workgroup */
#define G_NEG_INF (-3.0e38f)
#define G_QNAN ((plow_bf16)0x7fc1u)

static inline float g_rsqrt(float x) { return 1.0f / sqrtf(x); }

static inline float g_gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}
static inline float g_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float g_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* act 0 = gelu_tanh, 1 = silu; 2 (situ) and 3 (swiglu_oai) are pair forms and poison here, as
 * on the GPU. */
static inline float g_act_gate_only(float g, uint32_t act) {
    if (act >= 2u) return NAN;
    return act == 1u ? g_silu(g) : g_gelu_tanh(g);
}
/* act 3 = swiglu_oai (GPT-OSS, HF modeling_gpt_oss): g = min(g, limit), u = clamp(u, +-limit),
 * out = g * sigmoid(alpha * g) * (u + 1). */
static inline float g_swiglu_oai(float g, float u, float alpha, float limit) {
    g = g < limit ? g : limit;
    u = u < -limit ? -limit : (u > limit ? limit : u);
    return g * g_sigmoid(alpha * g) * (u + 1.0f);
}
static inline float g_situ_gate(float g, float beta) {
    return beta * tanhf(g / beta) * g_sigmoid(g);
}
static inline float g_situ_up(float u, float lb) { return lb > 0.0f ? lb * tanhf(u / lb) : u; }
/* Every GLU-family epilogue: act 0/1 gate-only, 2 situ (f0 beta, f1 lbeta), 3 swiglu_oai
 * (f0 alpha, f1 limit; f1 <= 0 poisons, dev_isa.h op 5). */
static inline float g_glu_pair(float g, float u, uint32_t act, float f0, float f1) {
    if (act == 2u) return g_situ_gate(g, f0) * g_situ_up(u, f1);
    if (act == 3u) return f1 > 0.0f ? g_swiglu_oai(g, u, f0, f1) : NAN;
    return g_act_gate_only(g, act) * u;
}

/* Order-preserving packed argmax key (op_elementwise.h amax_pack). */
static inline uint64_t g_amax_pack(plow_bf16 b, uint32_t i) {
    const uint32_t key = (b & 0x8000u) ? (uint32_t)(uint16_t)~b : (uint32_t)(b | 0x8000u);
    return ((uint64_t)key << 32) | (uint64_t)(~i);
}

/* GEMV_QKV i5/i6/i7: a bias tensor handle carried in an integer slot, 0 = absent (dev_isa.h op 22). */
#define G_QKV_BIAS(in, T, k) ((in)->i[(k)] ? (const plow_bf16*)(T)[(in)->i[(k)]] : (const plow_bf16*)0)

/* Contiguous share [lo, hi) of n items for `slice` of `nblk` (GV_BLOCKED ownership). */
static inline void g_range(uint32_t n, uint32_t slice, uint32_t nblk, uint32_t* lo, uint32_t* hi) {
    const uint32_t per = (n + nblk - 1) / nblk;
    const uint32_t a = slice * per, b = a + per;
    *lo = a > n ? n : a;
    *hi = b > n ? n : b;
}

/* FLASH_MERGE.t3 sinks element type (dev_isa.h op 13: bf16 [n_head]). */
#define PLOW_SINK_T plow_bf16
#define PLOW_SINK_LOAD(v) plow_bf2f(v)

static inline void g_poison_row(plow_bf16* row, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) row[i] = G_QNAN;
}

#endif /* PLOW_CPU_GOLDEN_H */
