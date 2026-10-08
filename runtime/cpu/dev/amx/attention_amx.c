/* attention_amx.c — FLASH_PREFILL on AMX tiles.
 *
 * The AVX-512 kernel's work units (128-row q tile, head, split), absolute KV splits, partial
 * format and blockwise online softmax, with QK^T and PV on TDPBF16PS: per XA_BK-key block a VNNI
 * K^T and a VNNI V are built once and the unit's 16-row q tiles run 2x2-blocked against them.
 * Every score and output element is one row's dot product in a fixed order, so a row's result
 * does not depend on the other rows of its tile, and blocks sit at absolute key positions:
 * packed and chunked prefill stay invariant as on the AVX-512 kernel. */
#include <stdlib.h>
#include <string.h>
#include "cpu_dev_internal.h"
#include "../avx512/avx512.h"
#include "amx_common.h"

#define XA_BQ PLOW_PF_TILE
#define XA_BK 64u
#define XA_NQT (XA_BQ / 16u)
_Static_assert(PLOW_PF_TILE % 32u == 0u, "q tiles pair up");
_Static_assert(XA_BK % 32u == 0u && XA_BK <= 64u, "key blocks are whole TDP K steps, one u64 key mask");

/* Scratch at hd 512: acc 256 KiB, Q 128 KiB, K^T 64 KiB, V 64 KiB, S 32 KiB, P 16 KiB. */
#define XA_ACC_OFF 0u
#define XA_QS_OFF (XA_ACC_OFF + XA_BQ * 512u * 4u)
#define XA_KT_OFF (XA_QS_OFF + XA_BQ * 512u * 2u)
#define XA_VP_OFF (XA_KT_OFF + XA_BK * 512u * 2u)
#define XA_S_OFF (XA_VP_OFF + XA_BK * 512u * 2u)
#define XA_P_OFF (XA_S_OFF + XA_BQ * XA_BK * 4u)
#define XA_PL_OFF (XA_P_OFF + XA_BQ * XA_BK * 2u)
#define XA_SCRATCH (XA_PL_OFF + XA_BQ * XA_BK * 2u)

/* PLOW_CPU_AMX_ATTN_SPLIT_P=1: P stays f32-accurate through the bf16 PV as P = hi + lo (both
 * bf16, two TDPBF16PS per key step) and l sums the unrounded P, as the AVX-512 decode kernel
 * does. Off: P = bf16(exp(s - m)), the MFMA operand round shared with the AVX-512 prefill. */
static int xa_split_p(void) {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("PLOW_CPU_AMX_ATTN_SPLIT_P");
        on = e && *e == '1';
    }
    return on;
}

/* K^T tile (key tile t, K step k) = [16 d pairs][16 keys] u32: B of S = Q K^T. */
static void xa_build_kt(uint8_t* kt, const plow_bf16* kbase, const uint32_t* row, uint32_t D) {
    const uint32_t nks = D / 32u;
    for (uint32_t t = 0; t < XA_BK / 16u; t++)
        for (uint32_t k = 0; k < nks; k++) {
            __m512i r[16];
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16u; i++)
                r[i] = _mm512_loadu_si512((const void*)(kbase + (size_t)row[t * 16u + i] * D + k * 32u));
            plow_amx_tr16x16(r);
            uint8_t* dst = kt + ((size_t)t * nks + k) * 1024u;
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16u; i++) _mm512_store_si512((void*)(dst + i * 64u), r[i]);
        }
}

/* V tile (d tile, key step) = [16 key pairs][16 d] of (V[2kp][d], V[2kp+1][d]): B of O += P V. */
static void xa_build_vp(uint8_t* vp, const plow_bf16* vbase, const uint32_t* row, uint32_t D) {
    const __m512i i0 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    const __m512i i1 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const uint32_t nst = XA_BK / 32u;
    for (uint32_t kp = 0; kp < XA_BK / 2u; kp++) {
        const plow_bf16* a = vbase + (size_t)row[2u * kp] * D;
        const plow_bf16* b = vbase + (size_t)row[2u * kp + 1u] * D;
        uint8_t* dst = vp + ((size_t)(kp / 16u) * 16u + kp % 16u) * 64u;
        for (uint32_t c = 0; c < D; c += 32u) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + c)), vb = _mm512_loadu_si512((const void*)(b + c));
            const __m512i lo = _mm512_unpacklo_epi16(va, vb), hi = _mm512_unpackhi_epi16(va, vb);
            const size_t dt = c / 16u;
            _mm512_store_si512((void*)(dst + dt * nst * 1024u), _mm512_permutex2var_epi64(lo, i0, hi));
            _mm512_store_si512((void*)(dst + (dt + 1u) * nst * 1024u), _mm512_permutex2var_epi64(lo, i1, hi));
        }
    }
}

/* t0=Opart t1=mlpart t2=Q t3=K t4=V t5=O_final? — the operands and immediates of v_flash_prefill. */
static void x_flash_prefill(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx) {
    float* const Opart0 = PLOW_CPU_TEN(in, T, 0);
    float* const mlpart0 = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* const Q0 = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* const K0 = PLOW_CPU_TEN(in, T, 3);
    const plow_bf16* const V0 = PLOW_CPU_TEN(in, T, 4);
    plow_bf16* const O_final0 = PLOW_CPU_TEN(in, T, 5);
    const uint32_t n_q0 = in->i[0], n_kv0 = in->i[1], n_head = in->i[2], n_kv_head = in->i[3];
    const uint32_t q_pos00 = in->i[4], window = in->i[5], D = in->i[6];
    const uint32_t nsplit0 = in->i[7] ? in->i[7] : 1u;
    const float scale = in->fj[0].f;
    const uint32_t kv_stride = in->fj[1].u, kv_mask = in->fj[2].u;
    if (D > 512u || (D & 31u) || !ctx || !ctx->scratch || ctx->scratch_bytes < XA_SCRATCH) {
        v_flash_prefill(in, slice, nblk, T, ctx);
        return;
    }
    const uint32_t gqa = n_head / n_kv_head, nks = D / 32u;
    const PlowCpuPack* pk = ctx->pack;
    const uint32_t n_work = plow_pf_units(pk, n_q0, n_head, nsplit0);
    float* const Opart = pk ? pk->opart : Opart0;
    float* const mlpart = pk ? pk->mlpart : mlpart0;
    uint8_t* const sc = ctx->scratch;
    float* const acc = (float*)(sc + XA_ACC_OFF);
    uint8_t* const qs = sc + XA_QS_OFF;
    uint8_t* const kt = sc + XA_KT_OFF;
    uint8_t* const vp = sc + XA_VP_OFF;
    float* const s = (float*)(sc + XA_S_OFF);
    plow_bf16* const p = (plow_bf16*)(sc + XA_P_OFF);
    plow_bf16* const plo = (plow_bf16*)(sc + XA_PL_OFF);
    const int split = xa_split_p();
    float m[XA_BQ], l[XA_BQ];
    float corr[XA_BQ] __attribute__((aligned(64)));
    uint32_t jlo[XA_BQ], jhi[XA_BQ], row[XA_BK];
    const __m512 vscale = _mm512_set1_ps(scale);

    for (uint32_t w0 = slice; w0 < n_work; w0 += nblk) {
        const PlowPfView pv = plow_pf_view(pk, w0, n_q0, q_pos00, n_kv0, nsplit0, n_head,
                                           (size_t)n_kv_head * kv_stride * D);
        const uint32_t n_q = pv.n_q, n_kv = pv.n_kv, q_pos0 = pv.q_pos0, nsplit = pv.nsplit;
        const plow_bf16* Q = Q0 + (size_t)pv.row0 * n_head * D;
        const plow_bf16* K = K0 + pv.kv_off;
        const plow_bf16* V = V0 + pv.kv_off;
        plow_bf16* O_final = O_final0 ? O_final0 + (size_t)pv.row0 * n_head * D : NULL;
        const uint32_t h = pv.h, hkv = h / gqa, q_base = pv.qt * XA_BQ;
        if (q_base >= n_q) continue;
        const uint32_t n_rows = n_q - q_base < XA_BQ ? n_q - q_base : XA_BQ;
        const uint32_t nqt = (n_rows + 15u) / 16u;
        const uint32_t q_tile_first = q_pos0 + q_base, q_tile_last = q_tile_first + XA_BQ - 1u;
        const uint32_t kv_end = q_tile_last + 1u < n_kv ? q_tile_last + 1u : n_kv;
        const uint32_t win_lo = (window && q_tile_first >= window) ? q_tile_first - window + 1u : 0u;
        uint32_t my_lo, my_hi;
        plow_pf_split_range(&pv, win_lo / XA_BK * XA_BK, kv_end, XA_BK, &my_lo, &my_hi);
        const plow_bf16* kbase = K + (size_t)hkv * kv_stride * D;
        const plow_bf16* vbase = V + (size_t)hkv * kv_stride * D;

        for (uint32_t t = 0; t < nqt; t++)
            for (uint32_t k = 0; k < nks; k++) {
                uint8_t* dst = qs + ((size_t)t * nks + k) * 1024u;
                for (uint32_t i = 0; i < 16u; i++) {
                    const uint32_t qi = q_base + t * 16u + i;
                    const __m512i v = qi < n_q ? _mm512_loadu_si512((const void*)(Q + ((size_t)qi * n_head + h) * D + k * 32u))
                                               : _mm512_setzero_si512();
                    _mm512_store_si512((void*)(dst + i * 64u), v);
                }
            }
        memset(acc, 0, (size_t)nqt * 16u * D * sizeof(float));
        for (uint32_t r = 0; r < XA_BQ; r++) m[r] = G_NEG_INF, l[r] = 0.0f;

        for (uint32_t k0 = my_lo; k0 < my_hi; k0 += XA_BK) {
            const uint32_t nk = my_hi - k0 < XA_BK ? my_hi - k0 : XA_BK;
            /* Keys past the range alias the block's first row: finite, and masked to P = 0. */
            for (uint32_t j = 0; j < XA_BK; j++) row[j] = (k0 + (j < nk ? j : 0u)) & kv_mask;
            uint32_t qt_lo = nqt, qt_hi = 0;
            for (uint32_t r = 0; r < n_rows; r++) {
                const uint32_t qg = q_tile_first + r;
                uint32_t hi_ = k0 <= qg ? qg - k0 + 1u : 0u;
                if (hi_ > nk) hi_ = nk;
                if (k0 + hi_ > n_kv) hi_ = n_kv > k0 ? n_kv - k0 : 0u;
                uint32_t lo_ = 0;
                if (window && qg + 1u > window) {
                    const uint32_t wl = qg + 1u - window;
                    lo_ = wl > k0 ? wl - k0 : 0u;
                }
                jlo[r] = lo_;
                jhi[r] = hi_;
                if (lo_ < hi_) {
                    if (r / 16u < qt_lo) qt_lo = r / 16u;
                    qt_hi = r / 16u + 1u;
                }
            }
            if (qt_lo >= qt_hi) continue;
            xa_build_kt(kt, kbase, row, D);
            xa_build_vp(vp, vbase, row, D);

            for (uint32_t qa = qt_lo; qa < qt_hi; qa += 2u) {
                const int two = qa + 1u < qt_hi;
                for (uint32_t kb = 0; kb < XA_BK / 16u; kb += 2u) {
                    _tile_zero(0);
                    _tile_zero(1);
                    _tile_zero(2);
                    _tile_zero(3);
                    for (uint32_t k = 0; k < nks; k++) {
                        _tile_loadd(4, qs + ((size_t)qa * nks + k) * 1024u, 64);
                        _tile_loadd(6, kt + ((size_t)kb * nks + k) * 1024u, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, kt + ((size_t)(kb + 1u) * nks + k) * 1024u, 64);
                        _tile_dpbf16ps(1, 4, 7);
                        if (two) {
                            _tile_loadd(5, qs + ((size_t)(qa + 1u) * nks + k) * 1024u, 64);
                            _tile_dpbf16ps(2, 5, 6);
                            _tile_dpbf16ps(3, 5, 7);
                        }
                    }
                    float* so = s + (size_t)qa * 16u * XA_BK + kb * 16u;
                    _tile_stored(0, so, XA_BK * 4);
                    _tile_stored(1, so + 16, XA_BK * 4);
                    if (two) {
                        _tile_stored(2, so + 16u * XA_BK, XA_BK * 4);
                        _tile_stored(3, so + 16u * XA_BK + 16, XA_BK * 4);
                    }
                }
            }

            /* Online softmax over the block, P = bf16(exp(s - m)) (the MFMA operand round), l sums
             * the rounded values; rows without keys here keep corr 1 and P 0. */
            for (uint32_t r = qt_lo * 16u; r < qt_hi * 16u; r++) {
                plow_bf16* pr = p + (size_t)r * XA_BK;
                plow_bf16* plr = plo + (size_t)r * XA_BK;
                corr[r] = 1.0f;
                if (r >= n_rows || jlo[r] >= jhi[r]) {
                    memset(pr, 0, XA_BK * sizeof(plow_bf16));
                    if (split) memset(plr, 0, XA_BK * sizeof(plow_bf16));
                    continue;
                }
                float* sr = s + (size_t)r * XA_BK;
                const uint64_t vb = (jhi[r] >= 64u ? ~0ull : (1ull << jhi[r]) - 1ull) & ~((1ull << jlo[r]) - 1ull);
                float bm = G_NEG_INF;
                for (uint32_t c = 0; c < XA_BK; c += 16u) {
                    const __mmask16 mk = (__mmask16)(vb >> c);
                    const __m512 v = _mm512_mul_ps(_mm512_load_ps(sr + c), vscale);
                    _mm512_store_ps(sr + c, v);
                    if (mk) {
                        const float x = _mm512_mask_reduce_max_ps(mk, v);
                        bm = x > bm ? x : bm;
                    }
                }
                const float mold = m[r], mnew = mold > bm ? mold : bm;
                const __m512 mv = _mm512_set1_ps(mnew);
                __m512 ls = _mm512_setzero_ps();
                for (uint32_t c = 0; c < XA_BK; c += 32u) {
                    const __m512 f0 = _mm512_maskz_mov_ps((__mmask16)(vb >> c), v_expf(_mm512_sub_ps(_mm512_load_ps(sr + c), mv)));
                    const __m512 f1 =
                        _mm512_maskz_mov_ps((__mmask16)(vb >> (c + 16u)), v_expf(_mm512_sub_ps(_mm512_load_ps(sr + c + 16u), mv)));
                    const __m512 p0 = v_round_bf16(f0), p1 = v_round_bf16(f1);
                    if (split) {
                        ls = _mm512_add_ps(ls, _mm512_add_ps(f0, f1));
                        _mm512_store_si512((void*)(plr + c),
                                           (__m512i)_mm512_cvtne2ps_pbh(_mm512_sub_ps(f1, p1), _mm512_sub_ps(f0, p0)));
                    } else {
                        ls = _mm512_add_ps(ls, _mm512_add_ps(p0, p1));
                    }
                    _mm512_store_si512((void*)(pr + c), (__m512i)_mm512_cvtne2ps_pbh(p1, p0));
                }
                m[r] = mnew;
                /* First keys of the row: acc and l are still zero, nothing to rescale. */
                corr[r] = mold == G_NEG_INF ? 1.0f : _mm512_cvtss_f32(v_expf(_mm512_set1_ps(mold - mnew)));
                l[r] = (mold == G_NEG_INF ? 0.0f : l[r] * corr[r]) + _mm512_reduce_add_ps(ls);
            }
            for (uint32_t r = qt_lo * 16u; r < qt_hi * 16u && r < n_rows; r++) {
                if (corr[r] == 1.0f) continue;
                float* ar = acc + (size_t)r * D;
                const __m512 cr = _mm512_set1_ps(corr[r]);
                for (uint32_t d = 0; d < D; d += 16u) _mm512_store_ps(ar + d, _mm512_mul_ps(_mm512_load_ps(ar + d), cr));
            }

            const uint32_t nst = XA_BK / 32u;
            for (uint32_t qa = qt_lo; qa < qt_hi; qa += 2u) {
                const int two = qa + 1u < qt_hi;
                const plow_bf16* p0 = p + (size_t)qa * 16u * XA_BK;
                const plow_bf16* pl0 = plo + (size_t)qa * 16u * XA_BK;
                for (uint32_t dt = 0; dt < D / 16u; dt += 2u) {
                    float* a0 = acc + (size_t)qa * 16u * D + dt * 16u;
                    _tile_loadd(0, a0, D * 4);
                    _tile_loadd(1, a0 + 16, D * 4);
                    if (two) {
                        _tile_loadd(2, a0 + 16u * D, D * 4);
                        _tile_loadd(3, a0 + 16u * D + 16, D * 4);
                    }
                    for (uint32_t ks = 0; ks < nst; ks++) {
                        _tile_loadd(4, p0 + ks * 32u, XA_BK * 2);
                        _tile_loadd(6, vp + ((size_t)dt * nst + ks) * 1024u, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, vp + ((size_t)(dt + 1u) * nst + ks) * 1024u, 64);
                        _tile_dpbf16ps(1, 4, 7);
                        if (two) {
                            _tile_loadd(5, p0 + 16u * XA_BK + ks * 32u, XA_BK * 2);
                            _tile_dpbf16ps(2, 5, 6);
                            _tile_dpbf16ps(3, 5, 7);
                        }
                        if (split) {
                            _tile_loadd(4, pl0 + ks * 32u, XA_BK * 2);
                            _tile_dpbf16ps(0, 4, 6);
                            _tile_dpbf16ps(1, 4, 7);
                            if (two) {
                                _tile_loadd(5, pl0 + 16u * XA_BK + ks * 32u, XA_BK * 2);
                                _tile_dpbf16ps(2, 5, 6);
                                _tile_dpbf16ps(3, 5, 7);
                            }
                        }
                    }
                    _tile_stored(0, a0, D * 4);
                    _tile_stored(1, a0 + 16, D * 4);
                    if (two) {
                        _tile_stored(2, a0 + 16u * D, D * 4);
                        _tile_stored(3, a0 + 16u * D + 16, D * 4);
                    }
                }
            }
        }

        for (uint32_t r = 0; r < n_rows; r++) {
            const uint32_t qi = q_base + r;
            const float* ar = acc + (size_t)r * D;
            if (nsplit == 1u && O_final && !pk) {
                const __m512 vinv = _mm512_set1_ps(l[r] > 0.0f ? 1.0f / l[r] : 0.0f);
                plow_bf16* orow = O_final + ((size_t)qi * n_head + h) * D;
                for (uint32_t d = 0; d < D; d += 16u) v_store_bf16(orow + d, _mm512_mul_ps(_mm512_load_ps(ar + d), vinv));
                continue;
            }
            const size_t rs = pk ? pk->row_off[pv.row0 + qi] : (size_t)qi * nsplit;
            const size_t at = rs * n_head + (size_t)h * nsplit + pv.sp;
            memcpy(Opart + at * D, ar, (size_t)D * sizeof(float));
            float* ml = mlpart + at * 2;
            ml[0] = m[r];
            ml[1] = l[r];
        }
    }
}

void plow_cpu_register_amx_attention(plow_cpu_kernel_fn* tab) {
    tab[PLOW_DOP_FLASH_PREFILL] = x_flash_prefill;
}
