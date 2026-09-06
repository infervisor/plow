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
    /* i5: 0 = the legacy rule (hd 64 interleaved, hd 128 half-split), 1 = force interleaved at
     * hd 128, 2 = force half-split (GPT-OSS NeoX at hd 64). */
    const int interleave = in->i[5] == 2u ? 0 : (hd == 64u) || (hd == 128u && in->i[5] == 1u);
    const uint32_t H2 = hd >> 1, total = ntok * nhead;
    /* H2 must be a whole 16-lane chunk; the interleaved form needs 16-lane pair groups. */
    if (hd > 512u || (hd & 31u)) {
        g_headnorm_rope(in, slice, nblk, T, ctx);
        return;
    }
    /* WV heads staged at once: the 1/sqrt is ~35 cycles of pure latency between the two passes,
     * and one head's second pass cannot start under it. Four lanes of vsqrtps/vdivps pay it once
     * for WV heads and are bit-identical to WV scalar 1/sqrtf. */
#define WV 4u
    float v[WV][512] __attribute__((aligned(64)));
    float ss[WV] __attribute__((aligned(16))), inv[WV] __attribute__((aligned(16)));
    const __m512i dup2 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    const __m512 sgn = _mm512_set_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);

    for (uint32_t w0 = slice * G_WAVES; w0 < total; w0 += nblk * G_WAVES) {
        const uint32_t wend = w0 + G_WAVES < total ? w0 + G_WAVES : total;
        for (uint32_t wb = w0; wb < wend; wb += WV) {
            const uint32_t nw = wend - wb < WV ? wend - wb : WV;
            for (uint32_t k = 0; k < nw; k++) {
                const uint32_t w = wb + k, t = w / nhead, hh = w % nhead;
                const plow_bf16* xr = x + ((size_t)t * nhead + hh) * hd;
                __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
                for (uint32_t i = 0; i < hd; i += 32) {
                    const __m512 a = v_load_bf16(xr + i), b = v_load_bf16(xr + i + 16);
                    _mm512_store_ps(v[k] + i, a);
                    _mm512_store_ps(v[k] + i + 16, b);
                    acc0 = _mm512_fmadd_ps(a, a, acc0);
                    acc1 = _mm512_fmadd_ps(b, b, acc1);
                }
                ss[k] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            }
            if (skip_norm) {
                for (uint32_t k = 0; k < nw; k++) inv[k] = 1.0f;
            } else {
                const __m128 q = _mm_add_ps(_mm_div_ps(_mm_load_ps(ss), _mm_set1_ps((float)hd)),
                                            _mm_set1_ps(eps));
                _mm_store_ps(inv, _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(q)));
            }

            for (uint32_t k = 0; k < nw; k++) {
                const uint32_t w = wb + k, t = w / nhead, hh = w % nhead;
                const uint32_t position = pos ? (uint32_t)pos[t] : out_row0 + t;
                const size_t obase =
                    out_stride
                        ? (n_batch_kv != 0
                               ? ((size_t)(t * nhead + hh) * out_stride + (position & kv_mask)) * hd
                               : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask)) * hd)
                        : ((size_t)(out_row0 + t) * nhead + hh) * hd;
                const __m512 vinv = _mm512_set1_ps(inv[k]);
                float* vk = v[k];
                for (uint32_t i = 0; i < hd; i += 16) {
                    __m512 a = _mm512_mul_ps(_mm512_load_ps(vk + i), vinv);
                    if (gamma) a = _mm512_mul_ps(a, v_load_bf16(gamma + i));
                    _mm512_store_ps(vk + i, a);
                }

                if (cosb) {
                    const size_t p = (size_t)position * H2;
                    if (!interleave) {
                        for (uint32_t i = 0; i < H2; i += 16) {
                            const __m512 c = _mm512_loadu_ps(cosb + p + i), s = _mm512_loadu_ps(sinb + p + i);
                            const __m512 lo = _mm512_load_ps(vk + i), hi = _mm512_load_ps(vk + i + H2);
                            _mm512_store_ps(vk + i, _mm512_fmsub_ps(lo, c, _mm512_mul_ps(hi, s)));
                            _mm512_store_ps(vk + i + H2, _mm512_fmadd_ps(hi, c, _mm512_mul_ps(lo, s)));
                        }
                    } else {
                        /* GPT-J pairs: lanes (2k, 2k+1) share cos/sin[k]; partner = lane ^ 1. */
                        for (uint32_t i = 0; i < hd; i += 16) {
                            const __m512 c = _mm512_permutexvar_ps(
                                dup2, _mm512_castps256_ps512(_mm256_loadu_ps(cosb + p + (i >> 1))));
                            const __m512 s = _mm512_permutexvar_ps(
                                dup2, _mm512_castps256_ps512(_mm256_loadu_ps(sinb + p + (i >> 1))));
                            const __m512 a = _mm512_load_ps(vk + i);
                            const __m512 partner = _mm512_permute_ps(a, 0xB1);
                            _mm512_store_ps(vk + i, _mm512_fmadd_ps(_mm512_mul_ps(partner, sgn), s,
                                                                    _mm512_mul_ps(a, c)));
                        }
                    }
                }
                for (uint32_t i = 0; i < hd; i += 16) v_store_bf16(out + obase + i, _mm512_load_ps(vk + i));
            }
        }
    }
#undef WV
}
