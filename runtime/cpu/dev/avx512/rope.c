/* rope.c — HEADNORM_ROPE (AVX-512): per-(token, head) RMSNorm then RoPE, golden semantics
 * (norm.c g_headnorm_rope), vectorised over hd. Spec §5 (RMSNorm two f32 passes, RoPE
 * half-rotated form on f32 chunks, one bf16 round on store). */
#include "avx512.h"

V_K(v_headnorm_rope) {
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* gamma = PLOW_CPU_TEN(in, T, 2);
    const float* cosb = PLOW_CPU_TEN(in, T, 3);
    const float* sinb = PLOW_CPU_TEN(in, T, 4);
    const int32_t* pos = PLOW_CPU_TEN(in, T, 5);
    const uint32_t ntok = in->i[0], nhead = in->i[1], hd = in->i[2], out_row0 = in->i[3];
    const uint32_t skip_norm = in->i[4], n_batch_kv = in->i[6];
    const uint32_t out_stride = in->fj[1].u, kv_mask = in->fj[2].u;
    const float eps = in->fj[0].f;
    const int interleave = (hd == 64u) || (hd == 128u && in->i[5] == 1u);
    const uint32_t H2 = hd >> 1, total = ntok * nhead;
    /* H2 must be a whole 16-lane chunk; the interleaved form needs 16-lane pair groups. */
    if (hd > 512u || (hd & 31u)) {
        g_headnorm_rope(in, slice, nblk, T, ctx);
        return;
    }
    float v[512] __attribute__((aligned(64)));
    const __m512i dup2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    const __m512 sgn = _mm512_set_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);

    for (uint32_t w0 = slice * G_WAVES; w0 < total; w0 += nblk * G_WAVES) {
        for (uint32_t wi = 0; wi < G_WAVES && w0 + wi < total; wi++) {
            const uint32_t w = w0 + wi, t = w / nhead, hh = w % nhead;
            const uint32_t position = pos ? (uint32_t)pos[t] : out_row0 + t;
            const plow_bf16* xr = x + ((size_t)t * nhead + hh) * hd;
            const size_t obase =
                out_stride
                    ? (n_batch_kv != 0
                           ? ((size_t)(t * nhead + hh) * out_stride + (position & kv_mask)) * hd
                           : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask)) * hd)
                    : ((size_t)(out_row0 + t) * nhead + hh) * hd;

            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            for (uint32_t i = 0; i < hd; i += 32) {
                const __m512 a = v_load_bf16(xr + i), b = v_load_bf16(xr + i + 16);
                _mm512_store_ps(v + i, a);
                _mm512_store_ps(v + i + 16, b);
                acc0 = _mm512_fmadd_ps(a, a, acc0);
                acc1 = _mm512_fmadd_ps(b, b, acc1);
            }
            const float ss = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            const float inv = skip_norm ? 1.0f : g_rsqrt(ss / (float)hd + eps);
            const __m512 vinv = _mm512_set1_ps(inv);
            for (uint32_t i = 0; i < hd; i += 16) {
                __m512 a = _mm512_mul_ps(_mm512_load_ps(v + i), vinv);
                if (gamma) a = _mm512_mul_ps(a, v_load_bf16(gamma + i));
                _mm512_store_ps(v + i, a);
            }

            if (cosb) {
                const size_t p = (size_t)position * H2;
                if (!interleave) {
                    for (uint32_t i = 0; i < H2; i += 16) {
                        const __m512 c = _mm512_loadu_ps(cosb + p + i), s = _mm512_loadu_ps(sinb + p + i);
                        const __m512 lo = _mm512_load_ps(v + i), hi = _mm512_load_ps(v + i + H2);
                        _mm512_store_ps(v + i, _mm512_fmsub_ps(lo, c, _mm512_mul_ps(hi, s)));
                        _mm512_store_ps(v + i + H2, _mm512_fmadd_ps(hi, c, _mm512_mul_ps(lo, s)));
                    }
                } else {
                    /* GPT-J pairs: lanes (2k, 2k+1) share cos/sin[k]; partner = lane ^ 1. */
                    for (uint32_t i = 0; i < hd; i += 16) {
                        const __m512 c = _mm512_permutexvar_ps(
                            dup2, _mm512_castps256_ps512(_mm256_loadu_ps(cosb + p + (i >> 1))));
                        const __m512 s = _mm512_permutexvar_ps(
                            dup2, _mm512_castps256_ps512(_mm256_loadu_ps(sinb + p + (i >> 1))));
                        const __m512 a = _mm512_load_ps(v + i);
                        const __m512 partner = _mm512_permute_ps(a, 0xB1);
                        _mm512_store_ps(v + i, _mm512_fmadd_ps(_mm512_mul_ps(partner, sgn), s,
                                                               _mm512_mul_ps(a, c)));
                    }
                }
            }
            for (uint32_t i = 0; i < hd; i += 16) v_store_bf16(out + obase + i, _mm512_load_ps(v + i));
        }
    }
}
