/* quant_group.cuh — the 1x128 group FP8 activation quant's scale and quotient (d_quant_fp8_group:
 * s = RN(max(amax, 1e-10) / 448), q = RN(v / s)) without __fdiv_rn's per-divide slow-path call,
 * which serializes the divides of a thread. One refined reciprocal per divisor, then the
 * Markstein correction; operands outside the certified range take __fdiv_rn.
 *
 * Certified output contract (not a general division proof): the scale bits and the clamped
 * quotient bits equal __fdiv_rn's on sm_90a (nvcc 12.9, no ftz/fast-math), checked exhaustively on
 * an H100 for every positive bf16 amax (incl. inf, NaN) against every bf16 v, both signs (checker
 * in the qwen-fp8-decode campaign scratch). rcp.approx's bits are implementation-defined, so every
 * other target takes __fdiv_rn until certified there; packets ship sm_90a cubins, never PTX a newer
 * GPU could JIT with this path. */
#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define QG_CERTIFIED 1
#else
#define QG_CERTIFIED 0
#endif

static __device__ __forceinline__ float qg_rcp(float s) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
    return __fmaf_rn(r, __fmaf_rn(r, -s, 1.0f), r);
}

/* v / s given r = qg_rcp(s); `in_range` keeps v, s and v / s within [2^-100, 2^100] (v = 0 keeps
 * its sign through v * r), the range the certificate covers. */
static __device__ __forceinline__ float qg_div(float v, float s, float r, bool& in_range) {
    in_range = in_range && s >= 0x1p-100f && s <= 0x1p100f &&
               (v == 0.0f ||
                (fabsf(v) >= 0x1p-100f && fabsf(v) <= 0x1p100f && fabsf(v) >= __fmul_rn(s, 0x1p-100f) &&
                 fabsf(v) <= __fmul_rn(s, 0x1p100f)));
    const float q = __fmul_rn(v, r);
    return v == 0.0f ? q : __fmaf_rn(r, __fmaf_rn(q, -s, v), q);
}

/* The group's scale from its amax (already reduced over the group). */
static __device__ __forceinline__ float qg_scale(float amax) {
    const float a = fmaxf(amax, 1e-10f);
    if (!QG_CERTIFIED) return __fdiv_rn(a, 448.0f);
    bool in_range = a <= 0x1p100f;
    const float s = qg_div(a, 448.0f, qg_rcp(448.0f), in_range);
    return in_range ? s : __fdiv_rn(a, 448.0f);
}

/* Four quotients of one lane, clamped to e4m3's range, against the group's scale s = qg_scale(amax).
 * One bound pair per group instead of qg_div's per-element check: lo <= |v| <= amax <= 2^100 (or
 * v = 0) implies it, since s >= amax / 449 and s >= 1e-10 / 449; NaN and anything else falls back. */
static __device__ __forceinline__ void qg_quant4(const float (&v)[4], float amax, float s, float (&q)[4]) {
    bool in_range = QG_CERTIFIED && amax <= 0x1p100f;
    if (QG_CERTIFIED) {
        const float r = qg_rcp(s), lo = fmaxf(0x1p-100f, __fmul_rn(s, 0x1p-100f));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float a = fabsf(v[j]);
            in_range = in_range && (v[j] == 0.0f || (a >= lo && a <= amax));
            const float qj = __fmul_rn(v[j], r);
            q[j] = v[j] == 0.0f ? qj : __fmaf_rn(r, __fmaf_rn(qj, -s, v[j]), qj);
        }
    }
    if (__builtin_expect(!in_range, 0)) {
#pragma unroll
        for (int j = 0; j < 4; j++) q[j] = __fdiv_rn(v[j], s);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = fminf(fmaxf(q[j], -448.0f), 448.0f);
}
