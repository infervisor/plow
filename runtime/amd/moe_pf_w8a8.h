#pragma once
/* PLOW_MOE_PF_W8A8 (CDNA4): ops 85/86 with packet i[7] == 2 (DOWN: 2 = f32 part, 3 = bf16 part).
 *
 * W8A8 block-FP8 grouped MoE prefill, AITER's fused-MoE numerics: activations carry one f32
 * scale per token per 128 K (1x128, amax/448), weights their checkpoint [128,128] grid, the
 * product of each 128-K block is an fp8 x fp8 MFMA (mfma_scale 32x32x64, neutral MX scales)
 * accumulated in f32 and promoted with as[row] * ws[col block]. GLU quantizes its own SwiGLU
 * output per 128 columns for DOWN, so the intermediate never exists in bf16.
 *
 *   GLU  A = xq[T,H] fp8 gathered by row_token, xs[H/128][T] f32 (QuantFp8Block128's layout)
 *        -> fq[pad_rows,I] fp8 + fs[pad_rows][I/128] f32
 *   DOWN A = fq rows (sorted), fs -> part[row_partidx][H] = gate * acc (f32 or bf16)
 *
 * Tile: the align's 64-row expert tile x 256 (GLU, gate|up of 32 columns per wave) or 512
 * (DOWN, 64 columns per wave) output columns. A is staged through LDS (one 8 KiB k-tile,
 * double-buffered, one barrier per k-tile); B is read straight into MFMA fragments (each wave
 * owns its weight rows, so LDS would add a copy), optionally prefetched a k-tile ahead (MW8_BPF).
 * Requires 8 waves, K % 128 == 0, GLU I % 256 == 0. */

#include "op_moe.h"


#define MW8_A_BYTES (MPF_BM * 128u)     /* one k-tile of A */
#define MW8_OFF_S (4u * MW8_A_BYTES)     /* f32 [2][64] A scales; A = [2 buf][hi, lo] k-tiles */
#define MW8_OFF_PIDX (MW8_OFF_S + 512u)  /* DOWN: u32 [64] row_partidx */
#define MW8_OFF_GATE (MW8_OFF_PIDX + 256u)
#define MW8_OFF_RED (MW8_OFF_GATE + 256u) /* GLU: f32 [8][64] per-wave row amax */
#define MW8_OFF_FS (MW8_OFF_RED + 2048u)  /* GLU: f32 [2][64] output scales */
#define MW8_LDS_BYTES (MW8_OFF_FS + 512u)

/* MW8_BPF: B fragments of the next k-tile prefetched into registers (1) or loaded at the top of
 * their own k-tile (0). */
#ifndef MW8_BPF
#define MW8_BPF 0
#endif

/* MW8_XCD: hand each XCD a contiguous tile range (tile = (lin % 8) * per + lin / 8), so the
 * m-tiles of one expert, which share its weights, run under one L2. 0 = linear order. */
#ifndef MW8_XCD
#define MW8_XCD 1
#endif
#ifndef MW8_SCHED
#define MW8_SCHED 1
#endif
#ifndef MW8_LO_E8M0
#define MW8_LO_E8M0 123
#endif

#if PLOW_CDNA4
static_assert(MPF_BM == 64 && PLOW_WAVES == 8, "W8A8 MoE tile is 64 rows over 8 waves");

/* 16-byte chunk c of row r: XOR-swizzled so the 32 rows a fragment read touches spread over
 * the 8 chunk positions. */
__device__ __forceinline__ unsigned mw8_swz(unsigned r, unsigned c) {
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

__device__ __forceinline__ unsigned mw8_remap(unsigned lin, unsigned n) {
#if MW8_XCD
    const unsigned per = n / 8u;
    if (lin < per * 8u) return (lin % 8u) * per + lin / 8u;
#endif
    return lin;
}

/* SPLIT: x ~= s * (hi + lo / 16), both e4m3. |x/s| <= 448 leaves a residual of at most half an
 * ulp (16 at the top binade), so lo * 16 stays <= 256: the pair carries ~7 significand bits, the
 * bf16 activation's precision, where a single e4m3 term (3 bits) costs ~2.5% relL2 per operand. */
__device__ __forceinline__ void mw8_split4(float a, float b, float c, float d, unsigned& hi,
                                           unsigned& lo) {
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, hi, true);
    const f32x2 x = __builtin_amdgcn_cvt_pk_f32_fp8(hi, false);
    const f32x2 y = __builtin_amdgcn_cvt_pk_f32_fp8(hi, true);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32((a - x[0]) * 16.0f, (b - x[1]) * 16.0f, 0, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32((c - y[0]) * 16.0f, (d - y[1]) * 16.0f, lo, true);
}

/* acc + 2^-4 * (a . b): the lo term's 1/16 rides the A operand's E8M0 MX scale (123 = 2^-4). */
__device__ __forceinline__ f32x16 mw8_mfma_lo(fp8v32 a, fp8v32 b, f32x16 acc) {
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, acc, 0, 0, 0, MW8_LO_E8M0, 0, 127);
}

__device__ __forceinline__ fp8v32 mw8_ld_frag(const unsigned char* p) {
    const fp8v16 lo = ld_glob_fp8v16(p), hi = ld_glob_fp8v16(p + 16);
    fp8v32 v;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[j] = (int)lo[j];
        v[4 + j] = (int)hi[j];
    }
    return v;
}

/* SPLIT (i[7] = 4/5): two-term FP8 activations. GLU reads the bf16 rows (Aq = bf16 x, As unused)
 * and splits each 1x128 group in its A staging; it writes fq rows as [hi I | lo I] (2*I bytes, the
 * bf16 fu footprint). DOWN reads those rows; both run hi and lo through the MFMA chain. */
template <bool GLU, bool SPLIT = false>
__device__ void d_moe_pf_w8a8(void* __restrict__ Cout, float* __restrict__ fs_out,
                              const unsigned char* __restrict__ Aq, const float* __restrict__ As,
                              unsigned as_stride, const unsigned long long* __restrict__ wtab,
                              const unsigned long long* __restrict__ stab,
                              const int* __restrict__ meta, const unsigned* __restrict__ rowmap,
                              const float* __restrict__ row_gate, unsigned N, unsigned K,
                              unsigned n_exp, unsigned act, float beta, float lbeta,
                              unsigned part16, unsigned slice, unsigned nblk,
                              unsigned char* lds) {
    constexpr unsigned NB = GLU ? 256u : 512u; /* output columns per tile */
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned l32 = lane & 31u, hi = lane >> 5;
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned tn = (N + NB - 1u) / NB;
    const unsigned n_tiles = (unsigned)tilep[n_exp] * tn;
    const unsigned KB = K >> 7;
    float* const lds_s = (float*)(lds + MW8_OFF_S);
    unsigned* const lds_pidx = (unsigned*)(lds + MW8_OFF_PIDX);
    float* const lds_gate = (float*)(lds + MW8_OFF_GATE);
    float* const lds_red = (float*)(lds + MW8_OFF_RED);
    float* const lds_fs = (float*)(lds + MW8_OFF_FS);
    /* A staging: thread -> (row, 16-byte chunk) of the 64 x 128 k-tile. */
    const unsigned sr = tid >> 3, sc = tid & 7u;

    for (unsigned lin = slice; lin < n_tiles; lin += nblk) {
        const unsigned tile = mw8_remap(lin, n_tiles);
        const unsigned mt = tile / tn, nt = tile % tn;
        const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * MPF_BM;
        const unsigned n0 = nt * NB;
        const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
        if (wb0 == 0ull) continue; /* EP: not this rank's expert (uniform: no barrier skew) */
        const unsigned char* W0 = (const unsigned char*)(size_t)wb0;
        const unsigned char* W1 =
            GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : W0;
        const float* S0 = (const float*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)];
        const float* S1 = GLU ? (const float*)(size_t)stab[(size_t)e * 3 + 1] : S0;

        /* This lane's B rows (null past N) and their scale rows, one per SN slot. */
        const unsigned cb0 = GLU ? n0 + 32u * wave : n0 + 64u * wave;
        const unsigned cb1 = GLU ? cb0 : cb0 + 32u;
        const unsigned brow0 = cb0 + l32, brow1 = cb1 + l32;
        const unsigned char* const bp0 = brow0 < N ? W0 + (size_t)brow0 * K + hi * 32u : nullptr;
        const unsigned char* const bp1 = brow1 < N ? W1 + (size_t)brow1 * K + hi * 32u : nullptr;
        const float* const sp0 = S0 + (size_t)(cb0 >> 7) * KB;
        const float* const sp1 = S1 + (size_t)(cb1 >> 7) * KB;

        const unsigned asrc = GLU ? rowmap[rowbase + sr] : rowbase + sr;
        const bool alive = !GLU || asrc != PLOW_EXPERT_UNUSED;
        /* Row / k-tile strides in bytes: fp8 K / 128; SPLIT GLU bf16 2K / 256; SPLIT DOWN
         * [hi K | lo K] 2K / 128 with lo at +K. */
        const size_t arow = SPLIT ? 2u * (size_t)K : (size_t)K;
        const unsigned akb = (SPLIT && GLU) ? 256u : 128u;
        const unsigned char* const ap =
            alive ? Aq + (size_t)asrc * arow + sc * ((SPLIT && GLU) ? 32u : 16u) : nullptr;
        const float* const asp = ((SPLIT && GLU) || sc != 0u || !alive) ? nullptr
                                 : GLU ? As + asrc : As + (size_t)asrc * KB;
        const unsigned as_step = GLU ? as_stride : 1u;
        const unsigned a2off = GLU ? 16u : K;
#define MW8_LOAD_A(kb) (ap ? ld_glob_fp8v16(ap + (kb) * akb) : (fp8v16)(0u))
#define MW8_LOAD_A2(kb) (ap ? ld_glob_fp8v16(ap + (kb) * akb + a2off) : (fp8v16)(0u))
#define MW8_LOAD_S(kb) (asp ? asp[(size_t)(kb) * as_step] : 0.0f)
#define MW8_LOAD_B(kb, b, ws)                                                                  \
    do {                                                                                       \
        b[0][0] = bp0 ? mw8_ld_frag(bp0 + (kb) * 128u) : (fp8v32)(0);                          \
        b[0][1] = bp0 ? mw8_ld_frag(bp0 + (kb) * 128u + 64u) : (fp8v32)(0);                    \
        b[1][0] = bp1 ? mw8_ld_frag(bp1 + (kb) * 128u) : (fp8v32)(0);                          \
        b[1][1] = bp1 ? mw8_ld_frag(bp1 + (kb) * 128u + 64u) : (fp8v32)(0);                    \
        ws[0] = bp0 ? sp0[kb] : 0.0f;                                                          \
        ws[1] = bp1 ? sp1[kb] : 0.0f;                                                          \
    } while (0)

        fp8v16 areg = MW8_LOAD_A(0);
        fp8v16 areg2 = (fp8v16)(0u);
        if constexpr (SPLIT) areg2 = MW8_LOAD_A2(0);
        float sreg = MW8_LOAD_S(0);
        /* Write the staged k-tile (and its row scales) into LDS buffer `b`. */
        auto stage = [&](unsigned b) {
            unsigned char* const tb = lds + b * 2u * MW8_A_BYTES;
            float* const sb = lds_s + b * MPF_BM;
            if constexpr (SPLIT && GLU) {
                /* areg|areg2 = 16 bf16 of this row's 128-group; the group is 8 adjacent lanes. */
                float v[16];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    v[2 * q] = __uint_as_float(areg[q] << 16);
                    v[2 * q + 1] = __uint_as_float(areg[q] & 0xffff0000u);
                    v[8 + 2 * q] = __uint_as_float(areg2[q] << 16);
                    v[8 + 2 * q + 1] = __uint_as_float(areg2[q] & 0xffff0000u);
                }
                float m = 0.0f;
#pragma unroll
                for (int q = 0; q < 16; q++) m = fmaxf(m, fabsf(v[q]));
                m = fmaxf(m, __shfl_xor(m, 1));
                m = fmaxf(m, __shfl_xor(m, 2));
                m = fmaxf(m, __shfl_xor(m, 4));
                const float s = fmaxf(m, 1e-10f) * (1.0f / 448.0f);
                const float inv = 1.0f / s;
                fp8v16 h, l;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    unsigned hq, lq;
                    mw8_split4(v[4 * q] * inv, v[4 * q + 1] * inv, v[4 * q + 2] * inv,
                               v[4 * q + 3] * inv, hq, lq);
                    h[q] = hq;
                    l[q] = lq;
                }
                *(fp8v16*)(tb + mw8_swz(sr, sc)) = h;
                *(fp8v16*)(tb + MW8_A_BYTES + mw8_swz(sr, sc)) = l;
                if (sc == 0u) sb[sr] = alive ? s : 0.0f;
            } else {
                *(fp8v16*)(tb + mw8_swz(sr, sc)) = areg;
                if constexpr (SPLIT) *(fp8v16*)(tb + MW8_A_BYTES + mw8_swz(sr, sc)) = areg2;
                if (sc == 0u) sb[sr] = sreg;
            }
        };
        fp8v32 bcur[2][2];
        float wcur[2];
#if MW8_BPF
        fp8v32 bnxt[2][2];
        float wnxt[2];
        MW8_LOAD_B(0u, bcur, wcur);
#endif

        __syncthreads(); /* the previous tile's epilogue is done with LDS */
        stage(0u);
        if (!GLU && tid < MPF_BM) {
            lds_pidx[tid] = rowmap[rowbase + tid];
            lds_gate[tid] = row_gate[rowbase + tid];
        }
        __syncthreads();

        f32x16 accf[2][2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) accf[i][j] = (f32x16)(0.0f);

        for (unsigned kb = 0; kb < KB; kb++) {
            const unsigned buf = kb & 1u;
            const bool more = kb + 1u < KB;
            if (more) {
                areg = MW8_LOAD_A(kb + 1u);
                if constexpr (SPLIT) areg2 = MW8_LOAD_A2(kb + 1u);
                sreg = MW8_LOAD_S(kb + 1u);
#if MW8_BPF
                MW8_LOAD_B(kb + 1u, bnxt, wnxt);
#endif
            }
#if !MW8_BPF
            MW8_LOAD_B(kb, bcur, wcur);
#endif
            const unsigned char* abuf = lds + buf * 2u * MW8_A_BYTES;
            const float* sbuf = lds_s + buf * MPF_BM;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned r = 32u * (unsigned)i + l32;
                fp8v32 af[2], al[2];
#pragma unroll
                for (int s = 0; s < 2; s++) {
                    const unsigned c = (unsigned)s * 4u + hi * 2u;
                    const fp8v16 lo = *(const fp8v16*)(abuf + mw8_swz(r, c));
                    const fp8v16 up = *(const fp8v16*)(abuf + mw8_swz(r, c + 1u));
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        af[s][q] = (int)lo[q];
                        af[s][4 + q] = (int)up[q];
                    }
                    if constexpr (SPLIT) {
                        const fp8v16 lo2 = *(const fp8v16*)(abuf + MW8_A_BYTES + mw8_swz(r, c));
                        const fp8v16 up2 =
                            *(const fp8v16*)(abuf + MW8_A_BYTES + mw8_swz(r, c + 1u));
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            al[s][q] = (int)lo2[q];
                            al[s][4 + q] = (int)up2[q];
                        }
                    }
                }
                float as[16];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float4 v =
                        *(const float4*)(sbuf + 32u * (unsigned)i + 8u * (unsigned)q + 4u * hi);
                    as[4 * q + 0] = v.x;
                    as[4 * q + 1] = v.y;
                    as[4 * q + 2] = v.z;
                    as[4 * q + 3] = v.w;
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    f32x16 t = plow_mfma_fp8_32x32(af[0], bcur[j][0], (f32x16)(0.0f));
                    t = plow_mfma_fp8_32x32(af[1], bcur[j][1], t);
                    if constexpr (SPLIT) {
                        t = mw8_mfma_lo(al[0], bcur[j][0], t);
                        t = mw8_mfma_lo(al[1], bcur[j][1], t);
                    }
#pragma unroll
                    for (int el = 0; el < 16; el++)
                        accf[i][j][el] = fmaf(t[el], as[el] * wcur[j], accf[i][j][el]);
                }
#if MW8_SCHED
                /* One row block's fragments, MFMAs and promotions at a time: interleaving both
                 * keeps four 16-wide MFMA results live and spills. */
                __builtin_amdgcn_sched_barrier(0);
#endif
            }
            if (more) {
                stage(buf ^ 1u);
#if MW8_BPF
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    bcur[j][0] = bnxt[j][0];
                    bcur[j][1] = bnxt[j][1];
                    wcur[j] = wnxt[j];
                }
#endif
            }
            __syncthreads();
        }

        if constexpr (GLU) {
            /* SwiGLU in f32, then the 1x128 output quant. A 128-column group is four waves'
             * 32 columns: lane-reduce within the wave, then LDS across the four. */
            float v[2][16];
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int el = 0; el < 16; el++) {
                    v[i][el] = moe_glu(accf[i][0][el], accf[i][1][el], act, beta, lbeta);
                    float m = fabsf(v[i][el]);
                    m = fmaxf(m, __shfl_xor(m, 1));
                    m = fmaxf(m, __shfl_xor(m, 2));
                    m = fmaxf(m, __shfl_xor(m, 4));
                    m = fmaxf(m, __shfl_xor(m, 8));
                    m = fmaxf(m, __shfl_xor(m, 16));
                    if (l32 == 0u) lds_red[wave * MPF_BM + 32u * i + mfma_acc_m(lane, el)] = m;
                }
            __syncthreads();
            const unsigned groups = N >> 7;
            if (tid < 2u * MPF_BM) {
                const unsigned r = tid & (MPF_BM - 1u), g = tid >> 6;
                float amax = 1e-10f; /* AITER floors amax, not the scale */
#pragma unroll
                for (int w = 0; w < 4; w++)
                    amax = fmaxf(amax, lds_red[(4u * g + (unsigned)w) * MPF_BM + r]);
                const float s = amax * (1.0f / 448.0f);
                lds_fs[g * MPF_BM + r] = s;
                st_act<float>(as_glob(fs_out) + (size_t)(rowbase + r) * groups + (n0 >> 7) + g, s);
            }
            __syncthreads();
            unsigned char* const fq = (unsigned char*)Cout;
            const unsigned g = wave >> 2, col = n0 + 32u * wave + l32;
            const size_t frow = SPLIT ? 2u * (size_t)N : (size_t)N;
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int el = 0; el < 16; el++) {
                    const unsigned r = 32u * i + mfma_acc_m(lane, el);
                    const float inv = 1.0f / lds_fs[g * MPF_BM + r];
                    const float xv = v[i][el] * inv;
                    const unsigned q = __builtin_amdgcn_cvt_pk_fp8_f32(xv, 0.0f, 0u, false);
                    const size_t at = (size_t)(rowbase + r) * frow + col;
                    st_act1_u8(as_glob(fq) + at, (unsigned char)q);
                    if constexpr (SPLIT) {
                        const float d = __builtin_amdgcn_cvt_pk_f32_fp8(q, false)[0];
                        const unsigned ql =
                            __builtin_amdgcn_cvt_pk_fp8_f32((xv - d) * 16.0f, 0.0f, 0u, false);
                        st_act1_u8(as_glob(fq) + at + N, (unsigned char)ql);
                    }
                }
        } else {
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int el = 0; el < 16; el++) {
                    const unsigned r = 32u * i + mfma_acc_m(lane, el);
                    const unsigned pidx = lds_pidx[r];
                    if (pidx == PLOW_EXPERT_UNUSED) continue;
                    const float gv = lds_gate[r];
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned bcol = j ? brow1 : brow0;
                        if (bcol >= N) continue;
                        const float val = gv * accf[i][j][el];
                        if (part16)
                            __builtin_nontemporal_store(
                                f2bf(val), (PLOW_GLOB bf16*)&((bf16*)Cout)[(size_t)pidx * N + bcol]);
                        else
                            __builtin_nontemporal_store(
                                val, (PLOW_GLOB float*)&((float*)Cout)[(size_t)pidx * N + bcol]);
                    }
                }
        }
    }
}

#undef MW8_LOAD_A
#undef MW8_LOAD_A2
#undef MW8_LOAD_S
#undef MW8_LOAD_B
extern "C" __device__ unsigned plow_moe_pf_w8a8_arm = 1;
extern "C" __device__ unsigned plow_moe_pf_w8a8_split_arm = 1;
#endif
