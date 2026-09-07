#include "avx512.h"

V_K(v_attn_res) {
    const uint32_t rows = in->i[0], H = in->i[1], nb = in->i[2], push_row = in->i[3], cap = in->i[4];
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0), *ring = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* prefix = PLOW_CPU_TEN(in, T, 1), *push = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 5);
    const float* weight = PLOW_CPU_TEN(in, T, 3);
    const float eps = in->fj[0].f;
    if (nb > 16 || cap < nb || (push && push_row >= cap)) {
        g_attn_res(in, slice, nblk, T, ctx);
        return;
    }
    for (uint32_t t = slice; t < rows; t += nblk) {
        const size_t po = (size_t)t * H, ro = (size_t)t * cap * H;
        if (push) memcpy(ring + ro + (size_t)push_row * H, push + po, (size_t)H * 2);
        const plow_bf16* x[17];
        float score[17], max = -INFINITY, sum = 0;
        for (uint32_t r = 0; r <= nb; r++) {
            x[r] = r < nb ? ring + ro + (size_t)r * H : prefix + po;
            __m512 ss = _mm512_setzero_ps(), sw = _mm512_setzero_ps();
            for (uint32_t d = 0; d < H; d += 16) {
                const __mmask16 mask = H - d >= 16 ? 0xffff : v_tail16(H - d);
                const __m512 v = v_load_bf16_mask(x[r] + d, mask);
                ss = _mm512_add_ps(ss, _mm512_mul_ps(v, v));
                sw = _mm512_add_ps(sw, _mm512_mul_ps(v, _mm512_maskz_loadu_ps(mask, weight + d)));
            }
            score[r] = _mm512_reduce_add_ps(sw) * g_rsqrt(_mm512_reduce_add_ps(ss) / (float)H + eps);
            max = max > score[r] ? max : score[r];
        }
        for (uint32_t r = 0; r <= nb; r++) { score[r] = expf(score[r] - max); sum += score[r]; }
        for (uint32_t r = 0; r <= nb; r++) score[r] /= sum;
        for (uint32_t d = 0; d < H; d += 16) {
            const __mmask16 mask = H - d >= 16 ? 0xffff : v_tail16(H - d);
            __m512 v = _mm512_setzero_ps();
            for (uint32_t r = 0; r <= nb; r++)
                v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_set1_ps(score[r]), v_load_bf16_mask(x[r] + d, mask)));
            v_store_bf16_mask(out + po + d, mask, v);
        }
        if (gamma) {
            const float inv = g_rsqrt(v_row_ss(out + po, H) / (float)H + eps);
            v_scale_row(out + po, out + po, gamma, inv, H);
        }
    }
}
