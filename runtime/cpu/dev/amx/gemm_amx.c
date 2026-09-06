/* gemm_amx.c — AMX-BF16 GEMM family (tier X): GEMM/_SMALL/_MED/_WIDE/_C5, GEMM_NORM, GEMM_GLU.
 *
 * Same slice partition and output as golden/gemm.c (nominal tile per op, SWZ=0 linear order);
 * only the math inside a tile changes. Inner loop per plans/cpu-kernel-innerloops.md §1.2/§6.3:
 * 2x2 fp32 accumulators (tmm0-3), A row-major via TILELOADD (tmm4/5), B packed on the fly into a
 * per-thread 32-column VNNI strip (tmm6/7), K outermost of the tile loops, TILESTORED into a
 * ping-pong fp32 C buffer whose epilogue is interleaved into the next block's K loop.
 *
 * K is paneled (KP) so a strip set + A pad + C partials fit the 1 MiB scratch for any K; C
 * partials live in scratch as fp32 tiles between panels (TILESTORED/TILELOADD, never AVX stores,
 * so ORM 20.15 does not apply to them).
 *
 * Cache blocking (measured here, Xeon 8581C): a 2x2 block consumes 4 KiB of tiles per 64 TDP
 * cycles, which is the L2->L1 fill rate, so operands must come from L1. KP=512 keeps one packed
 * B strip (32 KiB) L1-resident while the strip loop is OUTER and the M-block loop INNER; A is
 * the streamed operand and uses TILELOADDT1 so it does not evict the strip (ORM 20.8.1). */
#define _POSIX_C_SOURCE 199309L /* clock_gettime under the cc build's -std=c11 */
#include <errno.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cpu_dev_internal.h"
#include "golden/golden.h"
#include "amx_common.h"
#include "../avx512/avx512.h" /* v_glu_pair / v_load_bf16_mask for the GLU epilogue */
#include "../fp8_common.h"

#define KP 512u   /* K panel, bf16 elements (16 tile steps); one strip = 32 KiB < L1 */
#define NSTRIP 8u /* strips per tile: BN <= 256 (GLU: 4 output strips x {gate, up}) */
#define MBLK 8u   /* 32-row blocks per tile: BM <= 256 */
#define STRIP_BYTES (KP * 64u)      /* 2 x 1 KiB tiles per 32 K */
#define ABLK_BYTES (32u * KP * 2u)  /* 32 padded/normed A rows of one panel */
#define AP_BYTES (MBLK * ABLK_BYTES) /* every M block of the tile, staged once per panel */
#define CP_BYTES (MBLK * NSTRIP * 4096u) /* fp32 32x32 partials, one per (mb, strip) */
#define CB_BYTES (2u * 2u * 4096u)  /* ping-pong x {gate, up} final blocks */
#define BP_OFF 0u
#define AP_OFF (BP_OFF + NSTRIP * STRIP_BYTES)
#define CP_OFF (AP_OFF + AP_BYTES)
#define CB_OFF (CP_OFF + CP_BYTES)
#define SCRATCH_NEED (CB_OFF + CB_BYTES)

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg_t;

/* §6.2: palette 1, all eight tiles 16 rows x 64 B, configured once per thread. */
int plow_cpu_thread_init_amx(PlowCpuCtx* ctx) {
    (void)ctx;
    __attribute__((aligned(64))) tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    for (int i = 0; i < 8; i++) {
        cfg.colsb[i] = 64;
        cfg.rows[i] = 16;
    }
    _tile_loadconfig(&cfg);
    return 0;
}

/* ---- B strip pack: W[N][K] row-major -> [kb][t][r][c] = W[n0+16t+c/2][k0+32kb+2r+(c&1)] ----- */

/* 16x16 transpose of 32-bit lanes: tile row r (a k-pair) gathers pair r of 16 weight rows. */
static inline void tr16x16_epi32(__m512i* r) {
    __m512i t[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; i++) {
        r[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        r[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int i = 0; i < 4; i++) {
        t[i] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0x88);
        t[i + 4] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0xdd);
        t[i + 8] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0x88);
        t[i + 12] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0xdd);
    }
    for (int i = 0; i < 8; i++) {
        r[i] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0x88);
        r[i + 8] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0xdd);
    }
}

/* Pack columns [n0, n0+32) x K range [k0, k0+kp) of row-major W[N][K] into `dst`
 * (kp multiple of 32; columns >= N and K beyond `K` read as zero). */
/* The pack is the DRAM-facing side of the GEMM: 32 weight rows advance 64 B per K step, sixteen
 * independent streams per thread that the L2 streamer does not cover with 16 threads active.
 * Measured: the same kernel does 813 GFLOPS single-thread from L3 but ~17 GFLOPS/thread in-model.
 * Prefetch each row PACK_PF_B ahead (into L2) so ~128 lines are in flight per thread. */
#define PACK_PF_B 512u

void plow_cpu_amx_pack_b_strip(void* dst, const plow_bf16* W, uint32_t N, uint32_t K, uint32_t n0,
                                uint32_t k0, uint32_t kp) {
    uint8_t* out = dst;
    const uint32_t nkb = kp / 32u;
    for (uint32_t kb = 0; kb < nkb; kb++) {
        const uint32_t k = k0 + kb * 32u;
        for (uint32_t t = 0; t < 2; t++) {
            __m512i r[16];
            for (uint32_t j = 0; j < 16; j++) {
                const uint32_t n = n0 + t * 16u + j;
                if (n < N) _mm_prefetch((const char*)(W + (size_t)n * K + k) + PACK_PF_B, _MM_HINT_T1);
                r[j] = (n < N && k + 32u <= K)
                           ? _mm512_loadu_si512((const void*)(W + (size_t)n * K + k))
                           : _mm512_setzero_si512();
            }
            tr16x16_epi32(r);
            uint8_t* tile = out + ((size_t)kb * 2u + t) * 1024u;
            for (uint32_t rr = 0; rr < 16; rr++) _mm512_storeu_si512((void*)(tile + rr * 64u), r[rr]);
        }
    }
}

/* Same strip from an e4m3 weight: dequant 32 bytes -> 32 bf16 (exact) ahead of the transpose. */
plow_fp8_vlut plow_amx_fp8_lut;

static void pack_b_strip_fp8(void* dst, const uint8_t* W, uint32_t N, uint32_t K, uint32_t n0,
                             uint32_t k0, uint32_t kp) {
    uint8_t* out = dst;
    const uint32_t nkb = kp / 32u;
    for (uint32_t kb = 0; kb < nkb; kb++) {
        const uint32_t k = k0 + kb * 32u;
        for (uint32_t t = 0; t < 2; t++) {
            __m512i r[16];
            for (uint32_t j = 0; j < 16; j++) {
                const uint32_t n = n0 + t * 16u + j;
                if (n < N) _mm_prefetch((const char*)(W + (size_t)n * K + k) + PACK_PF_B / 2u, _MM_HINT_T1);
                r[j] = (n < N && k + 32u <= K)
                           ? plow_fp8x32_to_bf16(&plow_amx_fp8_lut,
                                                 _mm256_loadu_si256((const __m256i*)(W + (size_t)n * K + k)))
                           : _mm512_setzero_si512();
            }
            tr16x16_epi32(r);
            uint8_t* tile = out + ((size_t)kb * 2u + t) * 1024u;
            for (uint32_t rr = 0; rr < 16; rr++) _mm512_storeu_si512((void*)(tile + rr * 64u), r[rr]);
        }
    }
}

/* ---- A staging (tail rows / RMSNorm fold): 32 rows x kp bf16 at stride KP, zero padded ---- */

static inline __m512 bf16x16_to_f32(const plow_bf16* p) {
    const __m256i h = _mm256_loadu_si256((const void*)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

/* Stores march kb-outer so the chunk TILELOADD reads first was written longest ago (ORM 20.15). */
static void stage_a(plow_bf16* ap, const plow_bf16* A, uint32_t rows, uint32_t K, uint32_t k0,
                    uint32_t kp, const float* rms, const plow_bf16* gamma) {
    for (uint32_t kk = 0; kk < kp; kk += 32u) {
        const uint32_t k = k0 + kk;
        for (uint32_t r = 0; r < 32u; r++) {
            plow_bf16* d = ap + (size_t)r * KP + kk;
            if (r >= rows) {
                _mm512_storeu_si512((void*)d, _mm512_setzero_si512());
                continue;
            }
            const plow_bf16* s = A + (size_t)r * K + k;
            if (!rms) {
                _mm512_storeu_si512((void*)d, _mm512_loadu_si512((const void*)s));
            } else {
                const __m512 sc = _mm512_set1_ps(rms[r]);
                const __m512 lo = _mm512_mul_ps(_mm512_mul_ps(bf16x16_to_f32(s), sc),
                                                bf16x16_to_f32(gamma + k));
                const __m512 hi = _mm512_mul_ps(_mm512_mul_ps(bf16x16_to_f32(s + 16), sc),
                                                bf16x16_to_f32(gamma + k + 16));
                _mm512_storeu_si512((void*)d, (__m512i)_mm512_cvtne2ps_pbh(hi, lo));
            }
        }
    }
}

/* ---- Deferred epilogue (ILS): the previous block's fp32 C buffer -> bf16 rows ----------- */

typedef struct {
    const float* cb;    /* [32][32] fp32, gate (or plain) */
    const float* cb_up; /* GLU only */
    plow_bf16* dst;     /* C + m*N + n */
    const float* sc;    /* fp8: per-column w_scale for these 32 columns, else NULL */
    const float* sc_up; /* fp8 GLU: up scale */
    const plow_bf16* bias; /* per-column bias for these 32 columns, else NULL (GLU: gate bias) */
    const plow_bf16* bias_up;
    uint32_t ldc, rows, cols, done, act;
    float f0, f1;
} ils_t;

static inline void ils_row(const ils_t* e, uint32_t r) {
    const float* c = e->cb + (size_t)r * 32u;
    plow_bf16* d = e->dst + (size_t)r * e->ldc;
    if (!e->cb_up) {
        __m512 lo = _mm512_loadu_ps(c), hi = _mm512_loadu_ps(c + 16);
        if (e->sc || e->bias) {
            const __mmask16 mlo = e->cols >= 16u ? 0xFFFF : (__mmask16)((1u << e->cols) - 1u);
            const __mmask16 mhi = e->cols >= 32u ? 0xFFFF
                                  : e->cols > 16u ? (__mmask16)((1u << (e->cols - 16u)) - 1u) : 0;
            if (e->sc) {
                lo = _mm512_mul_ps(lo, _mm512_maskz_loadu_ps(mlo, e->sc));
                hi = _mm512_mul_ps(hi, _mm512_maskz_loadu_ps(mhi, e->sc + 16));
            }
            if (e->bias) {
                lo = _mm512_add_ps(lo, _mm512_castsi512_ps(_mm512_slli_epi32(
                                           _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mlo, e->bias)), 16)));
                hi = _mm512_add_ps(hi, _mm512_castsi512_ps(_mm512_slli_epi32(
                                           _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mhi, e->bias + 16)), 16)));
            }
        }
        const __m512i v = (__m512i)_mm512_cvtne2ps_pbh(hi, lo);
        if (e->cols == 32u) _mm512_storeu_si512((void*)d, v);
        else _mm512_mask_storeu_epi16((void*)d, (__mmask32)((1u << e->cols) - 1u), v);
    } else {
        /* Vector GLU epilogue: the scalar tanhf/expf loop cost ~20 us per 32x32 block against
         * ~1 us of tile math (GEMM_GLU was 386 ms/thr at T=128, 1356 at T=512). */
        const float* u = e->cb_up + (size_t)r * 32u;
        for (uint32_t j = 0; j < e->cols; j += 16u) {
            const __mmask16 m = e->cols - j >= 16u ? 0xFFFF : (__mmask16)((1u << (e->cols - j)) - 1u);
            __m512 g = _mm512_maskz_loadu_ps(m, c + j);
            __m512 uu = _mm512_maskz_loadu_ps(m, u + j);
            if (e->sc) g = _mm512_mul_ps(g, _mm512_maskz_loadu_ps(m, e->sc + j));
            if (e->sc_up) uu = _mm512_mul_ps(uu, _mm512_maskz_loadu_ps(m, e->sc_up + j));
            if (e->bias) g = _mm512_add_ps(g, v_load_bf16_mask(e->bias + j, m));
            if (e->bias_up) uu = _mm512_add_ps(uu, v_load_bf16_mask(e->bias_up + j, m));
            const __m512 o = v_glu_pair(g, uu, e->act, e->f0, e->f1);
            _mm256_mask_storeu_epi16((void*)(d + j), m, (__m256i)_mm512_cvtneps_pbh(o));
        }
    }
}

static inline void ils_step(ils_t* e, uint32_t n) {
    while (n-- && e->done < e->rows) ils_row(e, e->done++);
}

static inline void ils_drain(ils_t* e) {
    while (e->done < e->rows) ils_row(e, e->done++);
}

/* ---- One 32x32 block over one K panel ------------------------------------------------- */

/* acc in tmm0-3 (zeroed or reloaded from `part`), K loop with the ILS epilogue of `pend`
 * sprinkled in, then TILESTORED to `out` (fp32 stride 128 B). A1 == NULL: 16-row block. */
static void block(const plow_bf16* A0, const plow_bf16* A1, size_t lda, const uint8_t* bp,
                  uint32_t nkb, const float* part, float* out, ils_t* pend) {
    if (part) {
        _tile_loadd(0, part, 128);
        _tile_loadd(1, part + 16, 128);
        _tile_loadd(2, part + 16 * 32, 128);
        _tile_loadd(3, part + 16 * 32 + 16, 128);
    } else {
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
    }
    const uint32_t per = pend ? (pend->rows + nkb - 1u) / nkb : 0u;
    for (uint32_t kb = 0; kb < nkb; kb++) {
        const uint8_t* b = bp + (size_t)kb * 2048u;
        _tile_stream_loadd(4, A0 + kb * 32u, lda); /* A streams (T1 hint), B stays in L1 */
        _tile_loadd(6, b, 64);
        _tile_dpbf16ps(0, 4, 6);
        _tile_loadd(7, b + 1024, 64);
        _tile_dpbf16ps(1, 4, 7);
        if (A1) {
            _tile_stream_loadd(5, A1 + kb * 32u, lda);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        if (pend) ils_step(pend, per); /* §6.3: epilogue of block i-1 under block i's TDPs */
    }
    _tile_stored(0, out, 128);
    _tile_stored(1, out + 16, 128);
    _tile_stored(2, out + 16 * 32, 128);
    _tile_stored(3, out + 16 * 32 + 16, 128);
}

/* ---- Tile driver shared by every op --------------------------------------------------- */

/* PLOW_AMX_DEBUG=nopack|notdp|nostage: attribution knobs (WRONG RESULTS) — skip the B strip pack,
 * skip the tile math, or read A in place instead of staging. Read once, off by default. */
enum { AMX_DBG_NOPACK = 1, AMX_DBG_NOTDP = 2, AMX_DBG_NOSTAGE = 4 };
static int amx_debug_flags(void) {
    static int f = -1;
    if (f < 0) {
        const char* e = getenv("PLOW_AMX_DEBUG");
        f = 0;
        if (e && strstr(e, "nopack")) f |= AMX_DBG_NOPACK;
        if (e && strstr(e, "notdp")) f |= AMX_DBG_NOTDP;
        if (e && strstr(e, "nostage")) f |= AMX_DBG_NOSTAGE;
    }
    return f;
}

static void tile_amx(const gemm_args* g, uint32_t m0, uint32_t m1, uint32_t n0, uint32_t n1,
                     uint8_t* scratch) {
    const int dbg = amx_debug_flags();
    uint8_t* bp = scratch + BP_OFF;
    plow_bf16* ap = (plow_bf16*)(scratch + AP_OFF);
    float* cp = (float*)(scratch + CP_OFF);
    float* cb = (float*)(scratch + CB_OFF);
    const int glu = g->Wu != NULL || g->Wuq != NULL;
    const uint32_t nb = (n1 - n0 + 31u) / 32u, mb_n = (m1 - m0 + 31u) / 32u;
    const uint32_t npanel = (g->K + KP - 1u) / KP;
    ils_t pend = {0};
    uint32_t cur = 0;
    for (uint32_t p = 0; p < npanel; p++) {
        const uint32_t k0 = p * KP, kp = g->K - k0 < KP ? g->K - k0 : KP, nkb = kp / 32u;
        const int first = p == 0, last = p + 1 == npanel;
        for (uint32_t s = 0; s < nb && !(dbg & AMX_DBG_NOPACK); s++) {
            const uint32_t n = n0 + s * 32u;
            if (g->Wq) {
                if (!glu) {
                    pack_b_strip_fp8(bp + s * STRIP_BYTES, g->Wq, g->N, g->K, n, k0, kp);
                } else {
                    pack_b_strip_fp8(bp + (2 * s) * STRIP_BYTES, g->Wq, g->N, g->K, n, k0, kp);
                    pack_b_strip_fp8(bp + (2 * s + 1) * STRIP_BYTES, g->Wuq, g->N, g->K, n, k0, kp);
                }
            } else if (!glu) {
                plow_cpu_amx_pack_b_strip(bp + s * STRIP_BYTES, g->W, g->N, g->K, n, k0, kp);
            } else {
                plow_cpu_amx_pack_b_strip(bp + (2 * s) * STRIP_BYTES, g->W, g->N, g->K, n, k0, kp);
                plow_cpu_amx_pack_b_strip(bp + (2 * s + 1) * STRIP_BYTES, g->Wu, g->N, g->K, n, k0, kp);
            }
        }
        /* A operands for every M block of this panel, staged once (norm fold / row tails). */
        const plow_bf16* A0s[MBLK];
        const plow_bf16* A1s[MBLK];
        size_t ldas[MBLK];
        for (uint32_t mb = 0; mb < mb_n; mb++) {
            const uint32_t m = m0 + mb * 32u, rows = m1 - m < 32u ? m1 - m : 32u;
            /* Always stage when the A row stride crosses a page: TILELOADD of 16 rows at a
             * multi-KiB stride touches 16 pages per tile (act buffers < 2 MiB are not THP-backed),
             * and the tile loop re-reads A per strip; staged rows are 1 KiB apart. */
            if (g->rms || (rows != 32u && rows != 16u) ||
                ((size_t)g->K * 2u > 4096u && !(dbg & AMX_DBG_NOSTAGE))) {
                plow_bf16* a = ap + (size_t)mb * (ABLK_BYTES / 2u);
                stage_a(a, g->A + (size_t)m * g->K, rows, g->K, k0, kp, g->rms ? g->rms + m : NULL,
                        g->gamma);
                A0s[mb] = a;
                A1s[mb] = rows > 16u ? a + 16u * KP : NULL;
                ldas[mb] = KP * 2u;
            } else {
                A0s[mb] = g->A + (size_t)m * g->K + k0;
                A1s[mb] = rows == 32u ? A0s[mb] + 16u * g->K : NULL;
                ldas[mb] = (size_t)g->K * 2u;
            }
        }
        /* Strip outer, M block inner: the strip stays in L1, A streams through (§1.2 blocking). */
        for (uint32_t s = 0; s < nb; s++) {
            const uint32_t n = n0 + s * 32u, cols = n1 - n < 32u ? n1 - n : 32u;
            const uint8_t* bs = bp + (glu ? 2 * s : s) * STRIP_BYTES;
            for (uint32_t mb = 0; mb < mb_n; mb++) {
                const uint32_t m = m0 + mb * 32u, rows = m1 - m < 32u ? m1 - m : 32u;
                /* 8 partial slots per M block: strips, or {gate, up} pairs for GLU's 4 strips. */
                float* part = cp + ((size_t)mb * NSTRIP + (glu ? 2u * s : s)) * 1024u;
                const float* in_part = first ? NULL : part;
                float* out = last ? cb + (size_t)cur * 2048u : part;
                if (!(dbg & AMX_DBG_NOTDP)) {
                    block(A0s[mb], A1s[mb], ldas[mb], bs, nkb, in_part, out, pend.rows ? &pend : NULL);
                    if (glu)
                        block(A0s[mb], A1s[mb], ldas[mb], bs + STRIP_BYTES, nkb,
                              in_part ? in_part + 1024 : NULL, out + 1024, pend.rows ? &pend : NULL);
                } else if (pend.rows) {
                    ils_drain(&pend);
                }
                if (last) {
                    ils_drain(&pend);
                    pend = (ils_t){.cb = out,
                                   .cb_up = glu ? out + 1024 : NULL,
                                   .dst = g->C + (size_t)m * g->N + n,
                                   .sc = g->ws ? g->ws + n : NULL,
                                   .sc_up = g->us ? g->us + n : NULL,
                                   .bias = g->bias ? g->bias + n : NULL,
                                   .bias_up = g->bias_up ? g->bias_up + n : NULL,
                                   .f0 = g->f0,
                                   .f1 = g->f1,
                                   .ldc = g->N,
                                   .rows = rows,
                                   .cols = cols,
                                   .done = 0,
                                   .act = g->act};
                    cur ^= 1u;
                }
            }
        }
    }
    ils_drain(&pend);
}

int plow_amx_usable(const PlowCpuCtx* ctx, uint32_t K) {
    return ctx && ctx->scratch && ctx->scratch_bytes >= SCRATCH_NEED && (K % 32u) == 0u && K > 0;
}

void plow_amx_run_tiles(const gemm_args* g, uint32_t slice, uint32_t nblk, PlowCpuCtx* ctx) {
    const uint32_t tm = (g->M + g->BM - 1u) / g->BM, tn = (g->N + g->BN - 1u) / g->BN;
    for (uint32_t lin = slice; lin < tm * tn; lin += nblk) {
        const uint32_t m0 = (lin / tn) * g->BM, n0 = (lin % tn) * g->BN;
        const uint32_t m1 = m0 + g->BM < g->M ? m0 + g->BM : g->M;
        const uint32_t n1 = n0 + g->BN < g->N ? n0 + g->BN : g->N;
        tile_amx(g, m0, m1, n0, n1, ctx->scratch);
    }
}

/* t0=C t1=A t2=B t7=bias?  i0=M i1=N i2=K i4=a_row0 i5=c_row0 (golden gemm_op) */
static void gemm_op(const PlowDevInst* in, void* const* T, PlowCpuCtx* ctx, uint32_t BM,
                    uint32_t BN, uint32_t slice, uint32_t nblk, void (*fallback)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*)) {
    const uint32_t M = in->i[0], N = in->i[1], K = in->i[2];
    if (!plow_amx_usable(ctx, K)) {
        fallback(in, slice, nblk, T, ctx);
        return;
    }
    gemm_args g = {.C = (plow_bf16*)PLOW_CPU_TEN(in, T, 0) + (size_t)in->i[5] * N,
                   .A = (const plow_bf16*)PLOW_CPU_TEN(in, T, 1) + (size_t)in->i[4] * K,
                   .W = PLOW_CPU_TEN(in, T, 2), .bias = PLOW_CPU_TEN(in, T, 7),
                   .M = M, .N = N, .K = K, .BM = BM, .BN = BN};
    plow_amx_run_tiles(&g, slice, nblk, ctx);
}

#define X_K(name) \
    static void name(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, PlowCpuCtx* ctx)

X_K(x_gemm)       { gemm_op(in, T, ctx, 256, 256, slice, nblk, g_gemm); }
X_K(x_gemm_small) { gemm_op(in, T, ctx, 64, 128, slice, nblk, g_gemm_small); }
X_K(x_gemm_med)   { gemm_op(in, T, ctx, 128, 128, slice, nblk, g_gemm_med); }
X_K(x_gemm_wide)  { gemm_op(in, T, ctx, 128, 256, slice, nblk, g_gemm_wide); }
X_K(x_gemm_c5)    { gemm_op(in, T, ctx, 192, 256, slice, nblk, g_gemm_c5); }

/* t0=C t1=A t2=B t3=rms(f32) t4=gamma t7=bias?  i0=M i1=N i2=K */
X_K(x_gemm_norm) {
    const uint32_t K = in->i[2];
    if (!plow_amx_usable(ctx, K) || !PLOW_CPU_TEN(in, T, 3) || !PLOW_CPU_TEN(in, T, 4)) {
        g_gemm_norm(in, slice, nblk, T, ctx);
        return;
    }
    gemm_args g = {.C = PLOW_CPU_TEN(in, T, 0), .A = PLOW_CPU_TEN(in, T, 1), .W = PLOW_CPU_TEN(in, T, 2),
                   .rms = PLOW_CPU_TEN(in, T, 3), .gamma = PLOW_CPU_TEN(in, T, 4),
                   .bias = PLOW_CPU_TEN(in, T, 7),
                   .M = in->i[0], .N = in->i[1], .K = K, .BM = 256, .BN = 256};
    plow_amx_run_tiles(&g, slice, nblk, ctx);
}

/* t0=fu t1=x t2=W_gate t5=W_up t6=bias_gate? t7=bias_up?  i0=M i1=N i2=K i5=act f0/f1; nominal
 * 256x128 output tile. */
X_K(x_gemm_glu) {
    const uint32_t K = in->i[2];
    if (!plow_amx_usable(ctx, K)) {
        g_gemm_glu(in, slice, nblk, T, ctx);
        return;
    }
    gemm_args g = {.C = PLOW_CPU_TEN(in, T, 0), .A = PLOW_CPU_TEN(in, T, 1), .W = PLOW_CPU_TEN(in, T, 2),
                   .Wu = PLOW_CPU_TEN(in, T, 5), .bias = PLOW_CPU_TEN(in, T, 6),
                   .bias_up = PLOW_CPU_TEN(in, T, 7), .M = in->i[0], .N = in->i[1], .K = K, .BM = 256,
                   .BN = 128, .act = in->i[5], .f0 = in->fj[0].f, .f1 = in->fj[1].f};
    plow_amx_run_tiles(&g, slice, nblk, ctx);
}

/* Test/bench probe: `iters` 32x32xKP blocks over scratch-resident A/B, no pack, no epilogue. */
double plow_cpu_amx_debug_kloop(uint32_t iters, PlowCpuCtx* ctx) {
    uint8_t* sc = ctx->scratch;
    const plow_bf16* ap = (const plow_bf16*)(sc + AP_OFF);
    float* cb = (float*)(sc + CB_OFF);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < iters; i++)
        block(ap, ap + 16u * KP, KP * 2u, sc + BP_OFF, KP / 32u, NULL, cb, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* Test/bench probe: TMUL ceiling — `iters` x 4 TDPBF16PS on resident tiles, no loads. */
double plow_cpu_amx_debug_tdp(uint32_t iters, PlowCpuCtx* ctx) {
    uint8_t* sc = ctx->scratch;
    _tile_loadd(4, sc, 64);
    _tile_loadd(5, sc + 1024, 64);
    _tile_loadd(6, sc + 2048, 64);
    _tile_loadd(7, sc + 3072, 64);
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t i = 0; i < iters; i++) {
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, sc + CB_OFF, 64);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

void plow_cpu_register_amx(plow_cpu_kernel_fn* tab) {
    plow_fp8_vlut_init(&plow_amx_fp8_lut);
    tab[PLOW_DOP_GEMM] = x_gemm;
    tab[PLOW_DOP_GEMM_SMALL] = x_gemm_small;
    tab[PLOW_DOP_GEMM_MED] = x_gemm_med;
    tab[PLOW_DOP_GEMM_WIDE] = x_gemm_wide;
    tab[PLOW_DOP_GEMM_C5] = x_gemm_c5;
    tab[PLOW_DOP_GEMM_NORM] = x_gemm_norm;
    tab[PLOW_DOP_GEMM_GLU] = x_gemm_glu;
}
