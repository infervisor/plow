/* avx512/route.c — MoE combine (87), the one routing op with real per-element work; top-k and
 * align stay golden (per-token scalar selection). Same slice partition as golden/moe_route.c. */
#include "avx512.h"
#include "../golden/gptoss.h"

/* t0=out([T,H] bf16) t1=residual? t2=shared? t3=part([T*k,H] f32)  i0=H i1=k i2=T.
 * A slice owns a flat [lo, hi) run of (token, h); the k partials of one h are summed in the
 * golden order, so the result is bit-identical. */
V_K(v_moe_combine_pf) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* residual = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* shared = PLOW_CPU_TEN(in, T, 2);
    const float* part = PLOW_CPU_TEN(in, T, 3);
    const uint32_t H = in->i[0], k = in->i[1], nt = in->i[2] ? in->i[2] : 1u;
    uint32_t lo, hi;
    g_range(nt * H, slice, nblk, &lo, &hi);
    uint32_t i = lo;
    while (i < hi) {
        const uint32_t tok = i / H, h0 = i - tok * H;
        const uint32_t n = h0 + (hi - i) < H ? (uint32_t)(hi - i) : H - h0;
        const float* pt = part + (size_t)tok * k * H;
        uint32_t h = h0;
        for (; h < h0 + n; h += 16u) {
            const __mmask16 m = h0 + n - h >= 16u ? (__mmask16)0xFFFF : v_tail16(h0 + n - h);
            __m512 acc = residual ? v_load_bf16_mask(residual + (size_t)tok * H + h, m) : _mm512_setzero_ps();
            if (shared) acc = _mm512_add_ps(acc, v_load_bf16_mask(shared + (size_t)tok * H + h, m));
            for (uint32_t j = 0; j < k; j++)
                acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(m, pt + (size_t)j * H + h));
            v_store_bf16_mask(out + (size_t)tok * H + h, m, acc);
        }
        i += n;
    }
}

void v_register_route(plow_cpu_kernel_fn* tab) { tab[PLOW_DOP_MOE_COMBINE_PF] = v_moe_combine_pf; }
