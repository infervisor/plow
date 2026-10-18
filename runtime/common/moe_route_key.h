/* moe_route_key.h — the MoE router SELECTION contract, shared by the NVIDIA, AMD and host code.
 *
 * Selection key of expert e: ord(sel_e), sel_e = score_e (+ bias_e), ord the monotone f32 -> u32
 * map below. Ties break to the LOWEST id. Because (ord, id) is unique, the k-pass masked argmax and
 * the one-pass rank select (slot j = the expert with exactly j keys above it) pick the same experts
 * in the same order — on the DOMAIN only: every sel finite and above PLOW_ROUTE_KILL, the
 * serial reference's kill sentinel, so a killed expert never outranks a live one. ord orders by
 * bits, so -0 < +0 (both existing serial bodies already do this).
 *
 * Group-limited routing (noaux_tc): n_group contiguous groups of gsz experts, group score = sum of
 * its top-2 sel, top topk_group groups by (ord, lowest group id) kept; a dropped expert's ord
 * becomes 0, below every in-domain key and below the kill sentinel. With fewer than k kept experts
 * (refused by plow_route_group_ok) a sentinel-kill reference re-picks a killed expert ahead of the
 * dropped ones, so rank select and k-pass differ.
 *
 * Gates: unbiased score of each slot, summed in SLOT order, then
 * g_j = (norm && sum != 0 ? g_j / sum : g_j) * scale. */
#ifndef PLOW_MOE_ROUTE_KEY_H
#define PLOW_MOE_ROUTE_KEY_H

#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define PLOW_RK_FN __host__ __device__ static inline
#else
#define PLOW_RK_FN static inline
#endif

#define PLOW_ROUTE_KILL (-1e30f)
#define PLOW_ROUTE_MAX_GROUPS 64u

PLOW_RK_FN uint32_t plow_route_ord(float x) {
    uint32_t b;
    __builtin_memcpy(&b, &x, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/* (ka, ia) outranks (kb, ib). */
PLOW_RK_FN int plow_route_beats(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) {
    return ka > kb || (ka == kb && ia < ib);
}

/* The 64-bit packed form of the same order (AMD): needs n_exp <= 2^20. */
PLOW_RK_FN uint64_t plow_route_key64(uint32_t ord, uint32_t e, uint32_t n_exp) {
    return ((uint64_t)ord << 20) | (uint64_t)((n_exp - 1u - e) & 0xFFFFFu);
}

PLOW_RK_FN int plow_route_in_domain(float sel) {
    uint32_t b;
    __builtin_memcpy(&b, &sel, 4);
    return (b & 0x7F800000u) != 0x7F800000u && plow_route_ord(sel) > plow_route_ord(PLOW_ROUTE_KILL);
}

/* Slot of expert e: how many experts outrank it. */
PLOW_RK_FN uint32_t plow_route_rank(const uint32_t* ord, uint32_t n_exp, uint32_t e) {
    uint32_t r = 0;
    for (uint32_t f = 0; f < n_exp; f++) r += (uint32_t)plow_route_beats(ord[f], f, ord[e], e);
    return r;
}

PLOW_RK_FN int plow_route_group_active(uint32_t n_group, uint32_t topk_group) {
    return n_group > 1u && topk_group > 0u && topk_group < n_group;
}

/* The group shapes for which rank select equals the serial reference. */
PLOW_RK_FN int plow_route_group_ok(uint32_t n_exp, uint32_t k, uint32_t n_group, uint32_t topk_group) {
    if (!plow_route_group_active(n_group, topk_group)) return 1;
    return n_group <= PLOW_ROUTE_MAX_GROUPS && n_exp % n_group == 0u && n_exp / n_group >= 2u &&
           topk_group * (n_exp / n_group) >= k;
}

/* Group score: the top-2 sum of one group's sel values, in index order. */
PLOW_RK_FN float plow_route_group_top2(const float* sel, uint32_t gsz) {
    float m1 = -3.0e38f, m2 = -3.0e38f;
    for (uint32_t i = 0; i < gsz; i++) {
        const float s = sel[i];
        if (s > m1) { m2 = m1; m1 = s; } else if (s > m2) m2 = s;
    }
    return m1 + m2;
}

PLOW_RK_FN float plow_route_gate_sum(const float* gate, uint32_t k) {
    float s = 0.0f;
    for (uint32_t j = 0; j < k; j++) s += gate[j];
    return s;
}

PLOW_RK_FN float plow_route_gate(float g, float sum, int norm, float scale) {
    if (norm && sum != 0.0f) g /= sum;
    return g * scale;
}

#endif /* PLOW_MOE_ROUTE_KEY_H */
